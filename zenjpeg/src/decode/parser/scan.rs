//! SOS (Start of Scan) parsing and baseline entropy decoding.
//!
//! This module handles:
//! - SOS marker parsing
//! - Baseline sequential scan decoding
//! - Streaming decode for 4:4:4 YCbCr images

use crate::entropy::EntropyDecoder;
use crate::error::{Error, Result, ScanRead};
use crate::foundation::alloc::{checked_size_2d, try_alloc_dct_blocks, try_alloc_maybeuninit};
use crate::foundation::consts::{DCT_BLOCK_SIZE, MAX_HUFFMAN_TABLES};
use crate::huffman::HuffmanDecodeTable;
use crate::quant::dequantize_unzigzag_i32_into_partial;
use crate::types::JpegMode;
use enough::Stop;

use super::super::idct_int::{idct_int_dc_only, idct_int_tiered};
use super::super::{DecodeWarning, Strictness};
use super::JpegParser;
use crate::color::ycbcr_planes_i16_to_rgb_u8;

/// Scan parsing and baseline decoding methods for JpegParser.
impl<'a> JpegParser<'a> {
    /// Parse and decode a scan (SOS marker + entropy-coded data).
    ///
    /// The `stop` parameter allows cancellation of long-running decodes.
    pub(super) fn parse_scan(&mut self, stop: &impl Stop) -> Result<()> {
        let _length = self.read_u16()?;
        let num_components = self.read_u8()?;

        // Validate num_components in scan
        if num_components == 0 {
            return Err(Error::invalid_jpeg_data("SOS num_components is zero"));
        }
        if num_components > self.num_components {
            return Err(Error::invalid_jpeg_data(
                "SOS num_components exceeds frame components",
            ));
        }
        if num_components > crate::foundation::consts::MAX_COMPONENTS as u8 {
            return Err(Error::invalid_jpeg_data("SOS num_components too large"));
        }

        let mut scan_components = Vec::with_capacity(num_components as usize);

        for _ in 0..num_components {
            let component_id = self.read_u8()?;
            let tables = self.read_u8()?;
            let dc_table = tables >> 4;
            let ac_table = tables & 0x0F;

            // Validate Huffman table indexes
            if dc_table as usize >= MAX_HUFFMAN_TABLES {
                return Err(Error::invalid_jpeg_data(
                    "SOS DC Huffman table index out of range",
                ));
            }
            if ac_table as usize >= MAX_HUFFMAN_TABLES {
                return Err(Error::invalid_jpeg_data(
                    "SOS AC Huffman table index out of range",
                ));
            }

            // Find component index
            let comp_idx = self.components[..self.num_components as usize]
                .iter()
                .position(|c| c.id == component_id)
                .ok_or(Error::invalid_jpeg_data("unknown component in scan"))?;

            scan_components.push((comp_idx, dc_table, ac_table));
        }

        let ss = self.read_u8()?; // Spectral selection start
        let se = self.read_u8()?; // Spectral selection end
        let ah_al = self.read_u8()?;
        let ah = ah_al >> 4;
        let al = ah_al & 0x0F;

        // Validate spectral selection (must be 0-63)
        if ss > 63 {
            return Err(Error::invalid_jpeg_data(
                "SOS Ss (spectral start) out of range",
            ));
        }
        if se > 63 {
            return Err(Error::invalid_jpeg_data(
                "SOS Se (spectral end) out of range",
            ));
        }

        // Decode entropy-coded segment based on mode
        match self.mode {
            JpegMode::Progressive => {
                self.decode_progressive_scan(&scan_components, ss, se, ah, al, stop)?;
            }
            JpegMode::ArithmeticSequential => {
                self.decode_arithmetic_scan(&scan_components, stop)?;
            }
            JpegMode::ArithmeticProgressive => {
                self.decode_arithmetic_progressive_scan(&scan_components, ss, se, ah, al, stop)?;
            }
            _ => {
                // Baseline/Extended Huffman modes
                // Try fused parallel decode first (MCU-row-aligned DRI only)
                #[cfg(feature = "parallel")]
                let used_fused = self.try_fused_parallel_decode(&scan_components)?;
                #[cfg(not(feature = "parallel"))]
                let used_fused = false;

                if !used_fused {
                    if self.prefer_streaming
                        && self.can_use_streaming()
                        && self.streaming_rgb.is_none()
                    {
                        // Use streaming decode for baseline 4:4:4 - fuses decode + IDCT + color
                        let rgb = self.decode_baseline_streaming_rgb(&scan_components, stop)?;
                        self.streaming_rgb = Some(rgb);
                    } else {
                        self.decode_scan(&scan_components, stop)?;
                    }
                }
            }
        }

        Ok(())
    }

    /// Decode a baseline sequential scan (all coefficients at once).
    ///
    /// The `stop` parameter allows cancellation of long-running decodes.
    pub(super) fn decode_scan(
        &mut self,
        scan_components: &[(usize, u8, u8)],
        stop: &impl Stop,
    ) -> Result<()> {
        // DNL mode (height=0 in SOF) requires dynamic buffer growth during decode,
        // which is not yet implemented. For now, we need height before decoding.
        if self.height == 0 {
            return Err(Error::unsupported_feature(
                "DNL mode (height=0 in SOF) not yet supported for scan decoding",
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

        // Initialize coefficient storage - size depends on component's sampling factor
        if self.coeffs.is_empty() {
            for i in 0..self.num_components as usize {
                let h_samp = self.components[i].h_samp_factor as usize;
                let v_samp = self.components[i].v_samp_factor as usize;
                let comp_blocks_h = checked_size_2d(mcu_cols, h_samp)?;
                let comp_blocks_v = checked_size_2d(mcu_rows, v_samp)?;
                let num_blocks = checked_size_2d(comp_blocks_h, comp_blocks_v)?;
                self.coeffs.push(try_alloc_dct_blocks(
                    num_blocks,
                    "allocating DCT coefficients",
                )?);
                // Allocate parallel storage for coefficient counts (tiered IDCT)
                self.coeff_counts.push(vec![64u8; num_blocks]);
            }
        }

        // Set up entropy decoder
        let scan_data = &self.data[self.position..];
        let mut decoder = EntropyDecoder::new(scan_data);

        // Enable lenient mode for maximum error recovery
        if self.strictness == Strictness::Lenient {
            decoder.set_lenient(true);
        }

        // Check for missing DHT and emit warning/error BEFORE borrowing tables.
        // This avoids borrow conflicts between self.warn() (mutable) and self.dc_tables (immutable).
        {
            let mut any_missing = false;
            for (_comp_idx, dc_table, ac_table) in scan_components {
                let dc_idx = (*dc_table as usize).min(MAX_HUFFMAN_TABLES - 1);
                let ac_idx = (*ac_table as usize).min(MAX_HUFFMAN_TABLES - 1);
                if self.dc_tables[dc_idx].is_none() || self.ac_tables[ac_idx].is_none() {
                    any_missing = true;
                    break;
                }
            }
            if any_missing {
                self.warn(DecodeWarning::MissingHuffmanTables)?;
            }
        }

        for (_comp_idx, dc_table, ac_table) in scan_components {
            let dc_idx = (*dc_table as usize).min(MAX_HUFFMAN_TABLES - 1);
            let ac_idx = (*ac_table as usize).min(MAX_HUFFMAN_TABLES - 1);

            // Use explicit table if provided, otherwise fall back to standard tables.
            // (Warning already emitted above; Strict mode returned Err above.)
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

        // Decode MCUs with proper interleaving
        let mut mcu_count = 0u32;
        let restart_interval = self.restart_interval as u32;
        let mut next_restart_num = 0u8;

        // Track previous coefficient count per component for smart zeroing (zero-copy optimization).
        // Start with 64 to force full zeroing on first block of each component.
        let mut prev_coeff_counts: [u8; 4] = [64; 4];
        let mut had_padding_error = false;
        let mut truncation_mcu: Option<u32> = None;

        // Pre-compute per-component invariants outside the MCU loop.
        // These values are constant for the entire scan but were being recomputed
        // per MCU × per component (~1.5M times), costing ~57M instructions.
        struct CompScanInfo {
            comp_idx: usize,
            dc_table: usize,
            ac_table: usize,
            h_samp: usize,
            v_samp: usize,
            comp_blocks_h: usize,
            actual_blocks_h: usize,
            actual_blocks_v: usize,
            is_single_component_oversample: bool,
            has_any_padding: bool,
        }
        let comp_scan_infos: Vec<CompScanInfo> = scan_components
            .iter()
            .map(|(comp_idx, dc_table, ac_table)| {
                let h_samp = self.components[*comp_idx].h_samp_factor as usize;
                let v_samp = self.components[*comp_idx].v_samp_factor as usize;
                let comp_blocks_h = mcu_cols * h_samp;
                let comp_width =
                    (self.width as usize * h_samp + max_h_samp as usize - 1) / max_h_samp as usize;
                let comp_height =
                    (self.height as usize * v_samp + max_v_samp as usize - 1) / max_v_samp as usize;
                let actual_blocks_h = (comp_width + 7) / 8;
                let actual_blocks_v = (comp_height + 7) / 8;
                let is_single_component_oversample =
                    scan_components.len() == 1 && (h_samp > 1 || v_samp > 1);
                let has_any_padding =
                    actual_blocks_h < comp_blocks_h || actual_blocks_v < mcu_rows * v_samp;
                CompScanInfo {
                    comp_idx: *comp_idx,
                    dc_table: *dc_table as usize,
                    ac_table: *ac_table as usize,
                    h_samp,
                    v_samp,
                    comp_blocks_h,
                    actual_blocks_h,
                    actual_blocks_v,
                    is_single_component_oversample,
                    has_any_padding,
                }
            })
            .collect();

        for mcu_y in 0..mcu_rows {
            // Check for cancellation at each MCU row
            if stop.should_stop() {
                return Err(Error::cancelled());
            }

            for mcu_x in 0..mcu_cols {
                // Check for restart marker
                if restart_interval > 0 && mcu_count > 0 && mcu_count % restart_interval == 0 {
                    // Align to byte boundary (discard padding bits)
                    decoder.align_to_byte();
                    // Read and verify restart marker
                    decoder.read_restart_marker(next_restart_num)?;
                    // Update expected marker number (cycles 0-7)
                    next_restart_num = (next_restart_num + 1) & 7;
                    // Reset DC predictors
                    decoder.reset_dc();
                    // Reset smart zeroing hints (force full zero after restart)
                    prev_coeff_counts = [64; 4];
                }

                // For each component in the scan (using pre-computed invariants)
                for info in &comp_scan_infos {
                    // Hoist block coordinate base outside v/h loops
                    let base_block_x = mcu_x * info.h_samp;
                    let base_block_y = mcu_y * info.v_samp;

                    // Decode all blocks for this component in this MCU
                    for v in 0..info.v_samp {
                        let block_y = base_block_y + v;
                        for h in 0..info.h_samp {
                            let block_x = base_block_x + h;
                            let block_idx = block_y * info.comp_blocks_h + block_x;

                            // Check if this block is beyond actual image bounds (padding).
                            // Skip the check entirely for MCU-aligned components (no padding possible).
                            let is_padding = info.has_any_padding
                                && (block_x >= info.actual_blocks_h
                                    || block_y >= info.actual_blocks_v);

                            if is_padding && info.is_single_component_oversample {
                                // Single-component with oversampling: skip padding blocks
                                // These encoders typically omit them
                                self.coeffs[info.comp_idx][block_idx] = [0i16; 64];
                                self.coeff_counts[info.comp_idx][block_idx] = 1; // DC-only (zeros)
                                continue;
                            }

                            if is_padding {
                                // For padding blocks in multi-component images, behavior depends on strictness:
                                // - Strict: require all padding blocks (error if missing)
                                // - Balanced/Lenient: speculatively decode, fill with zeros if missing
                                //   (matches mozjpeg: missing padding blocks produce zero-filled output)

                                if self.strictness == Strictness::Strict {
                                    // Strict: require padding blocks, propagate errors
                                    let count = match decoder.decode_block_into(
                                        &mut self.coeffs[info.comp_idx][block_idx],
                                        prev_coeff_counts[info.comp_idx],
                                        info.comp_idx,
                                        info.dc_table,
                                        info.ac_table,
                                    )? {
                                        ScanRead::Value(c) => c,
                                        ScanRead::EndOfScan | ScanRead::Truncated => {
                                            return Err(Error::invalid_jpeg_data(
                                                "padding blocks missing (encoder omitted MCU padding)",
                                            ));
                                        }
                                    };
                                    self.coeff_counts[info.comp_idx][block_idx] = count;
                                    prev_coeff_counts[info.comp_idx] = count;
                                } else {
                                    // Balanced/Lenient: speculative decoding with recovery
                                    let saved_state = decoder.save_state();
                                    match decoder.decode_block_into(
                                        &mut self.coeffs[info.comp_idx][block_idx],
                                        prev_coeff_counts[info.comp_idx],
                                        info.comp_idx,
                                        info.dc_table,
                                        info.ac_table,
                                    ) {
                                        Ok(ScanRead::Value(count)) => {
                                            self.coeff_counts[info.comp_idx][block_idx] = count;
                                            prev_coeff_counts[info.comp_idx] = count;
                                        }
                                        Ok(ScanRead::EndOfScan | ScanRead::Truncated) => {
                                            decoder.restore_state(saved_state);
                                            self.coeffs[info.comp_idx][block_idx] = [0i16; 64];
                                            self.coeff_counts[info.comp_idx][block_idx] = 1;
                                            prev_coeff_counts[info.comp_idx] = 64;
                                            had_padding_error = true;
                                        }
                                        Err(_e) => {
                                            decoder.restore_state(saved_state);
                                            self.coeffs[info.comp_idx][block_idx] = [0i16; 64];
                                            self.coeff_counts[info.comp_idx][block_idx] = 1;
                                            prev_coeff_counts[info.comp_idx] = 64;
                                            had_padding_error = true;
                                        }
                                    }
                                }
                            } else {
                                // Non-padding block: decode with strictness-aware truncation handling
                                let count = match decoder.decode_block_into(
                                    &mut self.coeffs[info.comp_idx][block_idx],
                                    prev_coeff_counts[info.comp_idx],
                                    info.comp_idx,
                                    info.dc_table,
                                    info.ac_table,
                                )? {
                                    ScanRead::Value(c) => c,
                                    ScanRead::EndOfScan | ScanRead::Truncated => {
                                        // Truncation: Strict errors via warn(), Balanced/Lenient fills zeros
                                        if truncation_mcu.is_none() {
                                            truncation_mcu = Some(mcu_count);
                                        }
                                        self.coeffs[info.comp_idx][block_idx] = [0i16; 64];
                                        self.coeff_counts[info.comp_idx][block_idx] = 1;
                                        prev_coeff_counts[info.comp_idx] = 64;
                                        continue;
                                    }
                                };
                                self.coeff_counts[info.comp_idx][block_idx] = count;
                                prev_coeff_counts[info.comp_idx] = count;
                            }
                        }
                    }
                }

                mcu_count += 1;
            }
        }

        // Extract warning flags (decoder borrows self.dc_tables/ac_tables)
        let had_ac_overflow = decoder.had_ac_overflow;
        let had_invalid_huffman = decoder.had_invalid_huffman;
        self.position += decoder.position();

        // Emit warnings for any issues detected during decode
        let total_mcus = (mcu_rows * mcu_cols) as u32;
        if let Some(at_mcu) = truncation_mcu {
            self.warn(DecodeWarning::TruncatedScan {
                blocks_decoded: at_mcu,
                blocks_expected: total_mcus,
            })?;
        }
        if had_padding_error {
            self.warn(DecodeWarning::PaddingBlockError)?;
        }
        if had_ac_overflow {
            self.warn(DecodeWarning::AcIndexOverflow)?;
        }
        if had_invalid_huffman {
            self.warn(DecodeWarning::InvalidHuffmanCode)?;
        }

        Ok(())
    }

    /// Check if streaming decode can be used.
    /// Streaming is only possible for baseline 4:4:4 YCbCr images.
    pub(super) fn can_use_streaming(&self) -> bool {
        // Must be baseline (not progressive)
        if self.mode != JpegMode::Baseline {
            return false;
        }
        // Must have 3 components (YCbCr)
        if self.num_components != 3 {
            return false;
        }
        // Must be 4:4:4 (all components have same sampling factors)
        let h0 = self.components[0].h_samp_factor;
        let v0 = self.components[0].v_samp_factor;
        for i in 1..3 {
            if self.components[i].h_samp_factor != h0 || self.components[i].v_samp_factor != v0 {
                return false;
            }
        }
        // Must have 1x1 sampling (no subsampling)
        if h0 != 1 || v0 != 1 {
            return false;
        }
        true
    }

    /// Streaming decode for baseline 4:4:4 YCbCr images.
    /// Combines Huffman decode + dequantize + IDCT + color convert in one pass.
    /// No coefficient storage - processes MCU row by row directly to RGB output.
    ///
    /// The `stop` parameter allows cancellation of long-running decodes.
    pub(super) fn decode_baseline_streaming_rgb(
        &mut self,
        scan_components: &[(usize, u8, u8)],
        stop: &impl Stop,
    ) -> Result<Vec<u8>> {
        // DNL mode not supported for streaming decode
        if self.height == 0 {
            return Err(Error::unsupported_feature(
                "DNL mode (height=0 in SOF) not supported for streaming decode",
            ));
        }

        let width = self.width as usize;
        let height = self.height as usize;

        // For 4:4:4, MCU = 8x8 pixels (single block per component)
        let mcu_cols = (width + 7) / 8;
        let mcu_rows = (height + 7) / 8;
        let strip_width = mcu_cols * 8;

        // Check for missing DHT FIRST (before any immutable borrows of self).
        // warn() needs mutable self, which conflicts with quant/table borrows below.
        {
            let mut any_missing = false;
            for (_comp_idx, dc_table, ac_table) in scan_components {
                let dc_idx = (*dc_table as usize).min(MAX_HUFFMAN_TABLES - 1);
                let ac_idx = (*ac_table as usize).min(MAX_HUFFMAN_TABLES - 1);
                if self.dc_tables[dc_idx].is_none() || self.ac_tables[ac_idx].is_none() {
                    any_missing = true;
                    break;
                }
            }
            if any_missing {
                self.warn(DecodeWarning::MissingHuffmanTables)?;
            }
        }

        // Get quantization tables
        let quant_y = self.quant_tables[self.components[0].quant_table_idx as usize]
            .as_ref()
            .ok_or(Error::internal("missing Y quantization table"))?;
        let quant_cb = self.quant_tables[self.components[1].quant_table_idx as usize]
            .as_ref()
            .ok_or(Error::internal("missing Cb quantization table"))?;
        let quant_cr = self.quant_tables[self.components[2].quant_table_idx as usize]
            .as_ref()
            .ok_or(Error::internal("missing Cr quantization table"))?;

        // Set up entropy decoder
        let scan_data = &self.data[self.position..];
        let mut decoder = EntropyDecoder::new(scan_data);

        // Enable lenient mode for maximum error recovery
        if self.strictness == Strictness::Lenient {
            decoder.set_lenient(true);
        }

        for (comp_idx, dc_table, ac_table) in scan_components {
            let dc_idx = (*dc_table as usize).min(MAX_HUFFMAN_TABLES - 1);
            let ac_idx = (*ac_table as usize).min(MAX_HUFFMAN_TABLES - 1);

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
            decoder.set_dc_table(*comp_idx, dc_table_ref);

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
            decoder.set_ac_table(*comp_idx, ac_table_ref);
        }

        // Allocate strip buffers for one MCU row (8 rows of pixels)
        // Note: All elements are written by IDCT before color conversion reads them
        let strip_size = strip_width * 8;
        let mut y_strip: Vec<i16> = try_alloc_maybeuninit(strip_size, "Y strip buffer")?;
        let mut cb_strip: Vec<i16> = try_alloc_maybeuninit(strip_size, "Cb strip buffer")?;
        let mut cr_strip: Vec<i16> = try_alloc_maybeuninit(strip_size, "Cr strip buffer")?;

        // Allocate output RGB buffer
        // Note: All pixels are written by color conversion before return
        let rgb_size = checked_size_2d(width, height).and_then(|s| checked_size_2d(s, 3))?;
        let mut rgb: Vec<u8> = try_alloc_maybeuninit(rgb_size, "RGB output buffer")?;

        let mut mcu_count = 0u32;
        let restart_interval = self.restart_interval as u32;
        let mut next_restart_num = 0u8;

        // Reusable buffers - avoids allocation per block
        let mut dequant_buf = [0i32; DCT_BLOCK_SIZE];
        let mut coeffs = [0i16; DCT_BLOCK_SIZE];
        // Track previous coefficient count per component for smart zeroing
        let mut prev_coeff_counts: [u8; 4] = [64; 4];
        let mut streaming_truncation_mcu: Option<u32> = None;

        // Process MCU row by row
        for mcu_y in 0..mcu_rows {
            // Check for cancellation at each MCU row
            if stop.should_stop() {
                return Err(Error::cancelled());
            }

            // Decode one MCU row's worth of blocks
            for mcu_x in 0..mcu_cols {
                // Check for restart marker
                if restart_interval > 0 && mcu_count > 0 && mcu_count % restart_interval == 0 {
                    decoder.align_to_byte();
                    decoder.read_restart_marker(next_restart_num)?;
                    next_restart_num = (next_restart_num + 1) & 7;
                    decoder.reset_dc();
                    prev_coeff_counts = [64; 4]; // Force full zero after restart
                }

                // Decode, dequantize, and IDCT each component's block directly to strip
                for (comp_idx, dc_table, ac_table) in scan_components {
                    // Zero-copy decode into reusable buffer with smart zeroing
                    let coeff_count = match decoder.decode_block_into(
                        &mut coeffs,
                        prev_coeff_counts[*comp_idx],
                        *comp_idx,
                        *dc_table as usize,
                        *ac_table as usize,
                    )? {
                        ScanRead::Value(c) => c,
                        ScanRead::EndOfScan | ScanRead::Truncated => {
                            // Truncation: record for warning, Strict will error after loop
                            if streaming_truncation_mcu.is_none() {
                                streaming_truncation_mcu = Some(mcu_count);
                            }
                            prev_coeff_counts[*comp_idx] = 64;
                            continue;
                        }
                    };
                    // Track maximum, not just previous, for reusable buffer correctness
                    prev_coeff_counts[*comp_idx] = prev_coeff_counts[*comp_idx].max(coeff_count);

                    let quant = match *comp_idx {
                        0 => quant_y,
                        1 => quant_cb,
                        _ => quant_cr,
                    };
                    let strip = match *comp_idx {
                        0 => &mut y_strip,
                        1 => &mut cb_strip,
                        _ => &mut cr_strip,
                    };

                    // IDCT directly to strip buffer
                    let dst_offset = mcu_x * 8;
                    if coeff_count <= 1 {
                        let dc = coeffs[0] as i32 * quant[0] as i32;
                        idct_int_dc_only(dc, &mut strip[dst_offset..], strip_width);
                    } else {
                        dequantize_unzigzag_i32_into_partial(
                            &coeffs,
                            quant,
                            &mut dequant_buf,
                            coeff_count,
                        );
                        idct_int_tiered(
                            &mut dequant_buf,
                            &mut strip[dst_offset..],
                            strip_width,
                            coeff_count,
                        );
                    }
                }

                mcu_count += 1;
            }

            // Color convert this MCU row directly to RGB output
            let y_start = mcu_y * 8;
            let rows_this_mcu = 8.min(height.saturating_sub(y_start));
            let cols_this_mcu = width.min(strip_width);
            let is_rgb = self.is_rgb_jpeg();

            for row in 0..rows_this_mcu {
                let strip_offset = row * strip_width;
                let rgb_offset = (y_start + row) * width * 3;

                if is_rgb {
                    // RGB JPEG: interleave planes without YCbCr→RGB matrix
                    for px in 0..cols_this_mcu {
                        let i = strip_offset + px;
                        let o = rgb_offset + px * 3;
                        rgb[o] = y_strip[i].clamp(0, 255) as u8;
                        rgb[o + 1] = cb_strip[i].clamp(0, 255) as u8;
                        rgb[o + 2] = cr_strip[i].clamp(0, 255) as u8;
                    }
                } else {
                    ycbcr_planes_i16_to_rgb_u8(
                        &y_strip[strip_offset..strip_offset + cols_this_mcu],
                        &cb_strip[strip_offset..strip_offset + cols_this_mcu],
                        &cr_strip[strip_offset..strip_offset + cols_this_mcu],
                        &mut rgb[rgb_offset..rgb_offset + cols_this_mcu * 3],
                    );
                }
            }
        }

        // Extract warning flags (decoder borrows self tables)
        let had_ac_overflow = decoder.had_ac_overflow;
        let had_invalid_huffman = decoder.had_invalid_huffman;
        self.position += decoder.position();

        // Emit truncation warning (or error in Strict mode)
        let total_mcus = (mcu_rows * mcu_cols) as u32;
        if let Some(at_mcu) = streaming_truncation_mcu {
            self.warn(DecodeWarning::TruncatedScan {
                blocks_decoded: at_mcu,
                blocks_expected: total_mcus,
            })?;
        }
        if had_ac_overflow {
            self.warn(DecodeWarning::AcIndexOverflow)?;
        }
        if had_invalid_huffman {
            self.warn(DecodeWarning::InvalidHuffmanCode)?;
        }

        Ok(rgb)
    }
}
