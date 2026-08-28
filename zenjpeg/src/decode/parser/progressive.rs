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
use crate::foundation::alloc::{checked_size_2d, try_alloc_dct_blocks_pref};
use crate::foundation::consts::{DCT_BLOCK_SIZE, MAX_HUFFMAN_TABLES};
use crate::huffman::HuffmanDecodeTable;
use crate::types::Component;
use enough::Stop;

use super::super::{DecodeWarning, Strictness};
use super::JpegParser;

/// Geometry derived from sampling factors, shared across all four scan paths.
#[derive(Clone, Copy)]
struct ProgressiveGeometry {
    max_h_samp: u8,
    max_v_samp: u8,
    mcu_cols: usize,
    mcu_rows: usize,
}

/// Per-component block grid for non-interleaved DC and AC scans.
struct ComponentBlockGrid {
    /// Actual coefficient rows in the JPEG bitstream (`ceil(scaled_h/8)`).
    comp_blocks_v: usize,
    /// Actual coefficient columns in the JPEG bitstream (`ceil(scaled_w/8)`).
    comp_blocks_h: usize,
    /// Storage stride — MCU-padded to match the output path
    /// (`mcu_cols * h_samp`).
    padded_blocks_h: usize,
}

/// Compute MCU geometry and max sampling factors from frame metadata.
fn compute_progressive_geometry(
    width: u32,
    height: u32,
    num_components: u8,
    components: &[Component],
) -> ProgressiveGeometry {
    let mut max_h_samp = 1u8;
    let mut max_v_samp = 1u8;
    for i in 0..num_components as usize {
        max_h_samp = max_h_samp.max(components[i].h_samp_factor);
        max_v_samp = max_v_samp.max(components[i].v_samp_factor);
    }

    let mcu_width = (max_h_samp as usize) * 8;
    let mcu_height = (max_v_samp as usize) * 8;
    let mcu_cols = (width as usize + mcu_width - 1) / mcu_width;
    let mcu_rows = (height as usize + mcu_height - 1) / mcu_height;

    ProgressiveGeometry {
        max_h_samp,
        max_v_samp,
        mcu_cols,
        mcu_rows,
    }
}

/// Build the per-component block grid (actual vs MCU-padded stride).
fn component_block_grid(
    components: &[Component],
    comp_idx: usize,
    width: u32,
    height: u32,
    geom: &ProgressiveGeometry,
) -> ComponentBlockGrid {
    let h_samp = components[comp_idx].h_samp_factor as usize;
    let v_samp = components[comp_idx].v_samp_factor as usize;
    let width = width as usize;
    let height = height as usize;
    let max_h = geom.max_h_samp as usize;
    let max_v = geom.max_v_samp as usize;
    let scaled_w = (width * h_samp + max_h - 1) / max_h;
    let scaled_h = (height * v_samp + max_v - 1) / max_v;
    let comp_blocks_h = (scaled_w + 7) / 8;
    let comp_blocks_v = (scaled_h + 7) / 8;
    let padded_blocks_h = geom.mcu_cols * h_samp;
    ComponentBlockGrid {
        comp_blocks_v,
        comp_blocks_h,
        padded_blocks_h,
    }
}

/// Install DC/AC Huffman tables on the entropy decoder for each scan
/// component, falling back to standard JPEG tables (MJPEG support).
fn install_progressive_huffman_tables<'t>(
    decoder: &mut EntropyDecoder<'_, 't>,
    dc_tables: &'t [Option<HuffmanDecodeTable>],
    ac_tables: &'t [Option<HuffmanDecodeTable>],
    scan_components: &[(usize, u8, u8)],
) {
    for (_comp_idx, dc_table, ac_table) in scan_components {
        let dc_idx = (*dc_table as usize).min(MAX_HUFFMAN_TABLES - 1);
        let ac_idx = (*ac_table as usize).min(MAX_HUFFMAN_TABLES - 1);

        // Use explicit table if provided, otherwise use standard JPEG tables.
        // MJPEG files often omit DHT markers and expect standard tables.
        // Tables are borrowed, not cloned (~1.5KB savings per table).
        let dc_table_ref: &HuffmanDecodeTable = match &dc_tables[dc_idx] {
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

        let ac_table_ref: &HuffmanDecodeTable = match &ac_tables[ac_idx] {
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
}

/// Apply lenient/permissive flags onto the entropy decoder.
fn configure_decoder_strictness(decoder: &mut EntropyDecoder<'_, '_>, strictness: Strictness) {
    if strictness.lenient_entropy_recovery() {
        decoder.set_lenient(true);
    }
    if strictness.recovers_data_errors() {
        decoder.set_permissive_rst(true);
    }
}

/// Non-interleaved DC scan: blocks in raster order, single component.
///
/// The JPEG spec says non-interleaved scans encode ceil(X_i/8) blocks per
/// row, but storage uses MCU-padded stride (mcu_cols * h_samp) to match the
/// output path. Iterate actual block counts, store with padded stride.
///
/// Returns `true` if the scan was truncated.
fn decode_dc_scan_non_interleaved(
    decoder: &mut EntropyDecoder<'_, '_>,
    coeffs: &mut [Vec<[i16; DCT_BLOCK_SIZE]>],
    scan_component: (usize, u8, u8),
    is_first_scan: bool,
    al: u8,
    grid: &ComponentBlockGrid,
    restart_interval: u32,
    mut jbrd_padding_bits: Option<&mut alloc::vec::Vec<u8>>,
    stop: &impl Stop,
) -> Result<bool> {
    let (comp_idx, dc_table, _ac_table) = scan_component;
    let mut mcu_count = 0u32;
    let mut next_restart_num = 0u8;
    let mut had_progressive_truncation = false;

    'ni_dc: for block_y in 0..grid.comp_blocks_v {
        // Check for cancellation at each block row
        if stop.should_stop() {
            return Err(Error::cancelled());
        }

        for block_x in 0..grid.comp_blocks_h {
            // Check for restart marker
            if restart_interval > 0 && mcu_count > 0 && mcu_count % restart_interval == 0 {
                // JBRD: capture per-RST entropy-segment pad bits.
                if let Some(buf) = jbrd_padding_bits.as_deref_mut() {
                    buf.extend_from_slice(&decoder.partial_byte_padding_bits());
                }
                decoder.align_to_byte();
                if !decoder.read_restart_marker_tolerant(next_restart_num)? {
                    had_progressive_truncation = true;
                    break 'ni_dc;
                }
                next_restart_num = (next_restart_num + 1) & 7;
                decoder.reset_dc();
            }

            let block_idx = block_y * grid.padded_blocks_h + block_x;
            if is_first_scan {
                match decoder.decode_dc_first(comp_idx, dc_table as usize, al)? {
                    ScanRead::Value(dc) => {
                        coeffs[comp_idx][block_idx][0] = dc;
                    }
                    ScanRead::EndOfScan | ScanRead::Truncated => {
                        had_progressive_truncation = true;
                        break 'ni_dc;
                    }
                }
            } else {
                match decoder.decode_dc_refine(al)? {
                    ScanRead::Value(bit) => {
                        coeffs[comp_idx][block_idx][0] |= bit;
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

    Ok(had_progressive_truncation)
}

/// Interleaved DC scan: blocks in MCU order, multiple components.
/// Storage is MCU-padded, so all blocks (including padding) fit.
///
/// Returns `true` if the scan was truncated.
fn decode_dc_scan_interleaved(
    decoder: &mut EntropyDecoder<'_, '_>,
    coeffs: &mut [Vec<[i16; DCT_BLOCK_SIZE]>],
    components: &[Component],
    scan_components: &[(usize, u8, u8)],
    is_first_scan: bool,
    al: u8,
    geom: &ProgressiveGeometry,
    restart_interval: u32,
    mut jbrd_padding_bits: Option<&mut alloc::vec::Vec<u8>>,
    stop: &impl Stop,
) -> Result<bool> {
    let mut mcu_count = 0u32;
    let mut next_restart_num = 0u8;
    let mut had_progressive_truncation = false;

    'dc_scan: for mcu_y in 0..geom.mcu_rows {
        // Check for cancellation at each MCU row
        if stop.should_stop() {
            return Err(Error::cancelled());
        }

        for mcu_x in 0..geom.mcu_cols {
            // Check for restart marker
            if restart_interval > 0 && mcu_count > 0 && mcu_count % restart_interval == 0 {
                // JBRD: capture per-RST entropy-segment pad bits.
                if let Some(buf) = jbrd_padding_bits.as_deref_mut() {
                    buf.extend_from_slice(&decoder.partial_byte_padding_bits());
                }
                decoder.align_to_byte();
                if !decoder.read_restart_marker_tolerant(next_restart_num)? {
                    had_progressive_truncation = true;
                    break 'dc_scan;
                }
                next_restart_num = (next_restart_num + 1) & 7;
                decoder.reset_dc();
            }

            for (comp_idx, dc_table, _ac_table) in scan_components {
                let h_samp = components[*comp_idx].h_samp_factor as usize;
                let v_samp = components[*comp_idx].v_samp_factor as usize;
                let padded_blocks_h = geom.mcu_cols * h_samp;

                for v in 0..v_samp {
                    for h in 0..h_samp {
                        let block_x = mcu_x * h_samp + h;
                        let block_y = mcu_y * v_samp + v;
                        let block_idx = block_y * padded_blocks_h + block_x;

                        if is_first_scan {
                            match decoder.decode_dc_first(*comp_idx, *dc_table as usize, al)? {
                                ScanRead::Value(dc) => {
                                    coeffs[*comp_idx][block_idx][0] = dc;
                                }
                                ScanRead::EndOfScan | ScanRead::Truncated => {
                                    had_progressive_truncation = true;
                                    break 'dc_scan;
                                }
                            }
                        } else {
                            match decoder.decode_dc_refine(al)? {
                                ScanRead::Value(bit) => {
                                    coeffs[*comp_idx][block_idx][0] |= bit;
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

    Ok(had_progressive_truncation)
}

/// AC scan (first or refine): single component, fused per-grid decoder.
///
/// Non-interleaved AC scans encode blocks in raster order. The JPEG spec says
/// ceil(X_i/8) blocks per row, but storage uses MCU-padded stride to match
/// the output path.
///
/// The fused decoder methods eliminate per-block ScanResult wrapping, function
/// call overhead, and HuffmanResult→ScanRead conversion. They use fast_ac
/// combined lookup for AC first scans and pre-refilled bit reads for
/// refinement.
///
/// Returns `true` if the scan was truncated.
///
/// When `jbrd` is `Some`, populates the JBRD per-scan signals
/// (`reset_points`, `extra_zero_runs`) per libjxl's `DecodeDCTBlock` /
/// `RefineDCTBlock` semantics. `None` keeps the legacy zero-overhead path.
#[allow(clippy::too_many_arguments)]
fn decode_ac_scan(
    decoder: &mut EntropyDecoder<'_, '_>,
    coeffs: &mut [Vec<[i16; DCT_BLOCK_SIZE]>],
    nonzero_bitmaps: &mut [Vec<u64>],
    scan_component: (usize, u8, u8),
    is_first_scan: bool,
    ss: u8,
    se: u8,
    al: u8,
    grid: &ComponentBlockGrid,
    restart_interval: u32,
    jbrd: Option<&mut crate::decode::image::JbrdScanInfo>,
    jbrd_padding_bits: Option<&mut alloc::vec::Vec<u8>>,
    stop: &impl Stop,
) -> Result<bool> {
    let (comp_idx, _dc_table, ac_table) = scan_component;

    let completed = if is_first_scan {
        decoder.decode_ac_first_scan_tracked(
            coeffs,
            nonzero_bitmaps,
            comp_idx,
            ac_table as usize,
            ss,
            se,
            al,
            grid.comp_blocks_h,
            grid.comp_blocks_v,
            grid.padded_blocks_h,
            restart_interval,
            jbrd,
            jbrd_padding_bits,
            stop,
        )?
    } else {
        decoder.decode_ac_refine_scan_tracked(
            coeffs,
            nonzero_bitmaps,
            comp_idx,
            ac_table as usize,
            ss,
            se,
            al,
            grid.comp_blocks_h,
            grid.comp_blocks_v,
            grid.padded_blocks_h,
            restart_interval,
            jbrd,
            jbrd_padding_bits,
            stop,
        )?
    };

    Ok(!completed)
}

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

        let geom = compute_progressive_geometry(
            self.width,
            self.height,
            self.num_components,
            &self.components,
        );
        self.init_progressive_coeff_storage(&geom)?;

        // JBRD: if tracking is enabled, append a fresh scan_info record for
        // this SOS. AC scan helpers will fill `reset_points` and (first-scan
        // only) `extra_zero_runs`. DC scans push the record too — both stay
        // empty there since JBRD signals only fire on AC scans.
        if let Some(jbrd) = self.jbrd_scans.as_mut() {
            jbrd.push(crate::decode::image::JbrdScanInfo {
                ss,
                se,
                ah,
                al,
                reset_points: Vec::new(),
                extra_zero_runs: Vec::new(),
            });
        }

        let scan_data = &self.data[self.position..];
        let mut decoder = EntropyDecoder::new(scan_data);
        configure_decoder_strictness(&mut decoder, self.strictness);
        install_progressive_huffman_tables(
            &mut decoder,
            &self.dc_tables,
            &self.ac_tables,
            scan_components,
        );

        // Determine scan type
        let is_dc_scan = ss == 0 && se == 0;
        let is_first_scan = ah == 0;
        let restart_interval = self.restart_interval as u32;

        // JBRD tracker for THIS scan, if tracking is enabled. Pulled out of
        // `self.jbrd_scans` / `self.jbrd_padding_bits` so the inner closures
        // can borrow them mutably while the rest of `self` (coeffs, bitmaps,
        // components) is also mutably borrowed alongside.
        let current_jbrd_scan = self.jbrd_scans.as_mut().and_then(|v| v.last_mut());
        let mut current_jbrd_padding_bits = self.jbrd_padding_bits.as_mut();

        let had_progressive_truncation = if is_dc_scan {
            // DC scan: interleaved (multi-component) or non-interleaved (single).
            if scan_components.len() == 1 {
                let grid = component_block_grid(
                    &self.components,
                    scan_components[0].0,
                    self.width,
                    self.height,
                    &geom,
                );
                decode_dc_scan_non_interleaved(
                    &mut decoder,
                    &mut self.coeffs,
                    scan_components[0],
                    is_first_scan,
                    al,
                    &grid,
                    restart_interval,
                    current_jbrd_padding_bits.as_deref_mut(),
                    stop,
                )?
            } else {
                decode_dc_scan_interleaved(
                    &mut decoder,
                    &mut self.coeffs,
                    &self.components,
                    scan_components,
                    is_first_scan,
                    al,
                    &geom,
                    restart_interval,
                    current_jbrd_padding_bits.as_deref_mut(),
                    stop,
                )?
            }
        } else {
            // AC scan (single component only for progressive)
            if scan_components.len() != 1 {
                return Err(Error::invalid_jpeg_data(
                    "progressive AC scan must have single component",
                ));
            }
            let grid = component_block_grid(
                &self.components,
                scan_components[0].0,
                self.width,
                self.height,
                &geom,
            );
            decode_ac_scan(
                &mut decoder,
                &mut self.coeffs,
                &mut self.nonzero_bitmaps,
                scan_components[0],
                is_first_scan,
                ss,
                se,
                al,
                &grid,
                restart_interval,
                current_jbrd_scan,
                current_jbrd_padding_bits.as_deref_mut(),
                stop,
            )?
        };
        // JBRD: capture end-of-scan pad bits (the partial-byte padding
        // before the NEXT marker — could be another SOS, EOI, or
        // anything else). Mirrors libjxl's terminal `FinishStream` call
        // in `ProcessScan`.
        if let Some(buf) = current_jbrd_padding_bits {
            buf.extend_from_slice(&decoder.partial_byte_padding_bits());
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

    /// Initialize coefficient storage if not already done.
    ///
    /// CRITICAL: Storage must use MCU-padded block counts (mcu_cols * h_samp),
    /// matching the output path (CompInfo.comp_blocks_h) and baseline decoder.
    /// The interleaved DC scan writes padding blocks at MCU boundaries, and
    /// the output path indexes coefficients using MCU-padded stride. Using
    /// smaller component-based counts (ceil(scaled_w/8)) causes misaligned
    /// reads.
    fn init_progressive_coeff_storage(&mut self, geom: &ProgressiveGeometry) -> Result<()> {
        if !self.coeffs.is_empty() {
            return Ok(());
        }
        for i in 0..self.num_components as usize {
            let h_samp = self.components[i].h_samp_factor as usize;
            let v_samp = self.components[i].v_samp_factor as usize;
            // MCU-padded block counts: matches baseline decoder and output path
            let padded_blocks_h = geom.mcu_cols * h_samp;
            let padded_blocks_v = geom.mcu_rows * v_samp;
            let num_blocks = checked_size_2d(padded_blocks_h, padded_blocks_v)?;
            // Full-frame coefficient storage sized from the (untrusted) SOF
            // dimensions → default fallible.
            self.coeffs.push(try_alloc_dct_blocks_pref(
                self.alloc_pref,
                true,
                num_blocks,
                "allocating DCT coefficients",
            )?);
            // For progressive, we don't know coeff counts until all scans are
            // done. Default to 64 (full IDCT) — tiered IDCT is mainly for
            // baseline. Use fallible allocation so an OOM here cannot panic
            // after the parallel try_alloc_dct_blocks above succeeded with
            // lazy-committed pages. Same untrusted size → default fallible.
            self.coeff_counts
                .push(crate::foundation::alloc::try_alloc_filled_pref(
                    self.alloc_pref,
                    true,
                    num_blocks,
                    64u8,
                    "allocating progressive coefficient counts",
                )?);
            // Nonzero bitmap: all zeros initially (no coefficients placed yet).
            // Use fallible allocation for the same reason.
            self.nonzero_bitmaps
                .push(crate::foundation::alloc::try_alloc_filled_pref(
                    self.alloc_pref,
                    true,
                    num_blocks,
                    0u64,
                    "allocating progressive nonzero bitmaps",
                )?);
        }
        Ok(())
    }
}
