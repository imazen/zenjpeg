//! Progressive JPEG scan decoding.
//!
//! This module handles the incremental refinement of DCT coefficients
//! across multiple scans in progressive JPEG files.
//!
//! Progressive scan types:
//! - DC-first (ss=0, se=0, ah=0): Initial DC coefficient values
//! - DC-refine (ss=0, se=0, ah>0): Refine DC coefficient precision
//! - AC-first (ss>0, ah=0): Initial AC coefficient values for range [ss, se]
//! - AC-refine (ss>0, ah>0): Refine AC coefficient precision for range [ss, se]

use crate::entropy::EntropyDecoder;
use crate::error::{Error, Result, ScanRead};
use crate::foundation::alloc::{checked_size_2d, try_alloc_dct_blocks};
use crate::foundation::consts::MAX_HUFFMAN_TABLES;
use crate::huffman::HuffmanDecodeTable;
use enough::Stop;

use super::super::{DecodeWarning, Strictness};
use super::JpegParser;

/// Progressive scan decoding methods for JpegParser.
impl<'a> JpegParser<'a> {
    /// Decode a progressive scan (DC or AC, first or refine).
    ///
    /// Progressive JPEG encodes coefficients incrementally:
    /// - First, DC coefficients are sent for all blocks
    /// - Then AC coefficients in bands [ss, se]
    /// - Refinement scans improve precision bit by bit
    ///
    /// The `stop` parameter allows cancellation of long-running decodes.
    pub(super) fn decode_progressive_scan(
        &mut self,
        scan_components: &[(usize, u8, u8)],
        ss: u8,
        se: u8,
        ah: u8,
        al: u8,
        stop: &impl Stop,
    ) -> Result<()> {
        // DNL mode not supported for progressive decode
        if self.height == 0 {
            return Err(Error::unsupported_feature(
                "DNL mode (height=0 in SOF) not supported for progressive decode",
            ));
        }

        // Calculate max sampling factors to determine MCU structure
        let mut max_h_samp = 1u8;
        let mut max_v_samp = 1u8;
        for i in 0..self.num_components as usize {
            max_h_samp = max_h_samp.max(self.components[i].h_samp_factor);
            max_v_samp = max_v_samp.max(self.components[i].v_samp_factor);
        }

        // MCU dimensions in pixels
        let mcu_width = (max_h_samp as usize) * 8;
        let mcu_height = (max_v_samp as usize) * 8;

        // Number of MCUs
        let mcu_cols = (self.width as usize + mcu_width - 1) / mcu_width;
        let mcu_rows = (self.height as usize + mcu_height - 1) / mcu_height;

        // Initialize coefficient storage if not already done.
        //
        // CRITICAL: Storage must use MCU-padded block counts (mcu_cols * h_samp),
        // matching the output path (CompInfo.comp_blocks_h) and baseline decoder.
        // The interleaved DC scan writes padding blocks at MCU boundaries, and the
        // output path indexes coefficients using MCU-padded stride. Using smaller
        // component-based counts (ceil(scaled_w/8)) causes misaligned reads.
        if self.coeffs.is_empty() {
            for i in 0..self.num_components as usize {
                let h_samp = self.components[i].h_samp_factor as usize;
                let v_samp = self.components[i].v_samp_factor as usize;
                // MCU-padded block counts: matches baseline decoder and output path
                let padded_blocks_h = mcu_cols * h_samp;
                let padded_blocks_v = mcu_rows * v_samp;
                let num_blocks = checked_size_2d(padded_blocks_h, padded_blocks_v)?;
                self.coeffs.push(try_alloc_dct_blocks(
                    num_blocks,
                    "allocating DCT coefficients",
                )?);
                // For progressive, we don't know coeff counts until all scans are done
                // Default to 64 (full IDCT) - tiered IDCT is mainly for baseline
                self.coeff_counts.push(vec![64u8; num_blocks]);
                // Nonzero bitmap: all zeros initially (no coefficients placed yet)
                self.nonzero_bitmaps.push(vec![0u64; num_blocks]);
            }
        }

        // Set up entropy decoder
        let scan_data = &self.data[self.position..];

        let mut decoder = EntropyDecoder::new(scan_data);

        // Enable lenient/permissive error recovery
        if matches!(
            self.strictness,
            Strictness::Lenient | Strictness::Permissive
        ) {
            decoder.set_lenient(true);
        }
        if self.strictness == Strictness::Permissive {
            decoder.set_permissive_rst(true);
        }

        for (_comp_idx, dc_table, ac_table) in scan_components {
            let dc_idx = (*dc_table as usize).min(MAX_HUFFMAN_TABLES - 1);
            let ac_idx = (*ac_table as usize).min(MAX_HUFFMAN_TABLES - 1);

            // Use explicit table if provided, otherwise use standard JPEG tables.
            // MJPEG files often omit DHT markers and expect standard tables.
            // Tables are borrowed, not cloned (~1.5KB savings per table).
            let dc_table_ref: &HuffmanDecodeTable = match &self.dc_tables[dc_idx] {
                Some(table) => table,
                None => {
                    if dc_idx == 0 {
                        HuffmanDecodeTable::std_dc_luminance()
                    } else {
                        HuffmanDecodeTable::std_dc_chrominance()
                    }
                }
            };
            decoder.set_dc_table(dc_idx, dc_table_ref);

            let ac_table_ref: &HuffmanDecodeTable = match &self.ac_tables[ac_idx] {
                Some(table) => table,
                None => {
                    if ac_idx == 0 {
                        HuffmanDecodeTable::std_ac_luminance()
                    } else {
                        HuffmanDecodeTable::std_ac_chrominance()
                    }
                }
            };
            decoder.set_ac_table(ac_idx, ac_table_ref);
        }

        // Determine scan type
        let is_dc_scan = ss == 0 && se == 0;
        let is_first_scan = ah == 0;

        // Restart marker handling
        let mut mcu_count = 0u32;
        let restart_interval = self.restart_interval as u32;
        let mut next_restart_num = 0u8;
        let mut had_progressive_truncation = false;

        if is_dc_scan {
            // DC scan - can be interleaved (multiple components) or non-interleaved (single component)
            // For non-interleaved scans, blocks are in raster order (like AC scans)
            // For interleaved scans, blocks follow MCU order

            if scan_components.len() == 1 {
                // Non-interleaved DC scan: blocks in raster order.
                // The JPEG spec says non-interleaved scans encode ceil(X_i/8) blocks
                // per row, but storage uses MCU-padded stride (mcu_cols * h_samp) to
                // match the output path. Iterate actual block counts, store with
                // padded stride.
                let (comp_idx, dc_table, _ac_table) = scan_components[0];
                let h_samp = self.components[comp_idx].h_samp_factor as usize;
                let v_samp = self.components[comp_idx].v_samp_factor as usize;
                let width = self.width as usize;
                let height = self.height as usize;
                let max_h = max_h_samp as usize;
                let max_v = max_v_samp as usize;
                let scaled_w = (width * h_samp + max_h - 1) / max_h;
                let scaled_h = (height * v_samp + max_v - 1) / max_v;
                let comp_blocks_h = (scaled_w + 7) / 8;
                let comp_blocks_v = (scaled_h + 7) / 8;
                // Storage stride: MCU-padded (matches output path)
                let padded_blocks_h = mcu_cols * h_samp;

                'ni_dc: for block_y in 0..comp_blocks_v {
                    // Check for cancellation at each block row
                    if stop.should_stop() {
                        return Err(Error::cancelled());
                    }

                    for block_x in 0..comp_blocks_h {
                        // Check for restart marker
                        if restart_interval > 0
                            && mcu_count > 0
                            && mcu_count % restart_interval == 0
                        {
                            decoder.align_to_byte();
                            decoder.read_restart_marker(next_restart_num)?;
                            next_restart_num = (next_restart_num + 1) & 7;
                            decoder.reset_dc();
                        }

                        let block_idx = block_y * padded_blocks_h + block_x;
                        if is_first_scan {
                            match decoder.decode_dc_first(comp_idx, dc_table as usize, al)? {
                                ScanRead::Value(dc) => {
                                    self.coeffs[comp_idx][block_idx][0] = dc;
                                }
                                ScanRead::EndOfScan | ScanRead::Truncated => {
                                    had_progressive_truncation = true;
                                    break 'ni_dc;
                                }
                            }
                        } else {
                            match decoder.decode_dc_refine(al)? {
                                ScanRead::Value(bit) => {
                                    self.coeffs[comp_idx][block_idx][0] |= bit;
                                }
                                ScanRead::EndOfScan | ScanRead::Truncated => {
                                    had_progressive_truncation = true;
                                    break 'ni_dc;
                                }
                            }
                        }

                        mcu_count += 1;
                    }
                }
            } else {
                // Interleaved DC scan: blocks in MCU order.
                // Storage is MCU-padded, so all blocks (including padding) fit.
                'dc_scan: for mcu_y in 0..mcu_rows {
                    // Check for cancellation at each MCU row
                    if stop.should_stop() {
                        return Err(Error::cancelled());
                    }

                    for mcu_x in 0..mcu_cols {
                        // Check for restart marker
                        if restart_interval > 0
                            && mcu_count > 0
                            && mcu_count % restart_interval == 0
                        {
                            decoder.align_to_byte();
                            decoder.read_restart_marker(next_restart_num)?;
                            next_restart_num = (next_restart_num + 1) & 7;
                            decoder.reset_dc();
                        }

                        for (comp_idx, dc_table, _ac_table) in scan_components {
                            let h_samp = self.components[*comp_idx].h_samp_factor as usize;
                            let v_samp = self.components[*comp_idx].v_samp_factor as usize;
                            let padded_blocks_h = mcu_cols * h_samp;

                            for v in 0..v_samp {
                                for h in 0..h_samp {
                                    let block_x = mcu_x * h_samp + h;
                                    let block_y = mcu_y * v_samp + v;
                                    let block_idx = block_y * padded_blocks_h + block_x;

                                    if is_first_scan {
                                        match decoder.decode_dc_first(
                                            *comp_idx,
                                            *dc_table as usize,
                                            al,
                                        )? {
                                            ScanRead::Value(dc) => {
                                                self.coeffs[*comp_idx][block_idx][0] = dc;
                                            }
                                            ScanRead::EndOfScan | ScanRead::Truncated => {
                                                had_progressive_truncation = true;
                                                break 'dc_scan;
                                            }
                                        }
                                    } else {
                                        match decoder.decode_dc_refine(al)? {
                                            ScanRead::Value(bit) => {
                                                self.coeffs[*comp_idx][block_idx][0] |= bit;
                                            }
                                            ScanRead::EndOfScan | ScanRead::Truncated => {
                                                had_progressive_truncation = true;
                                                break 'dc_scan;
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        mcu_count += 1;
                    }
                }
            }
        } else {
            // AC scan (single component only for progressive)
            // Progressive AC scans can only have one component
            if scan_components.len() != 1 {
                return Err(Error::invalid_jpeg_data(
                    "progressive AC scan must have single component",
                ));
            }

            let (comp_idx, _dc_table, ac_table) = scan_components[0];
            let h_samp = self.components[comp_idx].h_samp_factor as usize;
            let v_samp = self.components[comp_idx].v_samp_factor as usize;

            // Non-interleaved AC scans encode blocks in raster order.
            // The JPEG spec says ceil(X_i/8) blocks per row, but storage uses
            // MCU-padded stride to match the output path. Use 2D loop: iterate
            // actual block counts, store with padded stride.
            let width = self.width as usize;
            let height = self.height as usize;
            let max_h = max_h_samp as usize;
            let max_v = max_v_samp as usize;
            let scaled_w = (width * h_samp + max_h - 1) / max_h;
            let scaled_h = (height * v_samp + max_v - 1) / max_v;
            let comp_blocks_h = (scaled_w + 7) / 8;
            let comp_blocks_v = (scaled_h + 7) / 8;
            // Storage stride: MCU-padded (matches output path)
            let padded_blocks_h = mcu_cols * h_samp;

            // Fused AC scan: entire block grid processed in one call.
            // Eliminates per-block ScanResult wrapping, function call overhead,
            // and HuffmanResult→ScanRead conversion. Uses fast_ac combined lookup
            // for AC first scans and pre-refilled bit reads for refinement.
            if is_first_scan {
                if !decoder.decode_ac_first_scan(
                    &mut self.coeffs,
                    &mut self.nonzero_bitmaps,
                    comp_idx,
                    ac_table as usize,
                    ss,
                    se,
                    al,
                    comp_blocks_h,
                    comp_blocks_v,
                    padded_blocks_h,
                    restart_interval,
                    stop,
                )? {
                    had_progressive_truncation = true;
                }
            } else if !decoder.decode_ac_refine_scan(
                &mut self.coeffs,
                &mut self.nonzero_bitmaps,
                comp_idx,
                ac_table as usize,
                ss,
                se,
                al,
                comp_blocks_h,
                comp_blocks_v,
                padded_blocks_h,
                restart_interval,
                stop,
            )? {
                had_progressive_truncation = true;
            }
        }

        // Extract warning flags before dropping decoder
        let had_ac_overflow = decoder.had_ac_overflow;
        let had_invalid_huffman = decoder.had_invalid_huffman;
        self.position += decoder.position();

        // Emit warning for progressive scan truncation (or error in Strict mode)
        if had_progressive_truncation {
            self.warn(DecodeWarning::TruncatedProgressiveScan)?;
        }
        if had_ac_overflow {
            self.warn(DecodeWarning::AcIndexOverflow)?;
        }
        if had_invalid_huffman {
            self.warn(DecodeWarning::InvalidHuffmanCode)?;
        }

        Ok(())
    }
}
