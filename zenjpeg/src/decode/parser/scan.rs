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
use super::baseline_streaming;
use crate::color::{ycbcr_planes_i16_to_rgb_u8, ycbcr_planes_i16_to_xrgba_u8};
use crate::types::PixelFormat;

/// Scan parsing and baseline decoding methods for JpegParser.
impl<'a> JpegParser<'a> {
    /// Parse and decode a scan (SOS marker + entropy-coded data).
    ///
    /// The `stop` parameter allows cancellation of long-running decodes.
    pub(super) fn parse_scan(&mut self, stop: &impl Stop) -> Result<()> {
        let length = self.read_u16()?;
        let num_components = self.read_u8()?;

        // Validate num_components in scan.
        //
        // IJG libjpeg v9+ with `-block 16` produces progressive JPEGs where the
        // first SOS has num_components=0 (a non-standard extension for large DCT
        // block sizes). Detect this and return UnsupportedFeature instead of a
        // confusing "SOS num_components is zero" parse error.
        if num_components == 0 {
            // SOS length=6 with Ns=0 is the IJG block_size>8 signature:
            // the scan structure uses 0-component scans for coefficient selection
            // ranges beyond the standard 0..63.
            return Err(Error::unsupported_feature(
                "This JPEG uses non-standard DCT block sizes (SOS with zero components) and is not standards-compliant. It was likely created with IJG libjpeg v9+ or v10, which support proprietary block size extensions that no other decoder implements.",
            ));
        }
        if num_components > self.num_components {
            return Err(Error::invalid_jpeg_data(
                "SOS num_components exceeds frame components",
            ));
        }
        if num_components > crate::foundation::consts::MAX_COMPONENTS as u8 {
            return Err(Error::invalid_jpeg_data("SOS num_components too large"));
        }

        // Validate SOS marker length: must be 6 + 2 * num_components
        // (2-byte length + 1-byte Ns + 2*Ns component specs + 3 bytes Ss/Se/AhAl)
        let expected_length = 6 + 2 * num_components as u16;
        if length != expected_length {
            // Don't error — warn and continue. The reads below will still
            // consume exactly the right number of bytes for the declared
            // num_components. A length mismatch indicates corruption but
            // is recoverable as long as num_components is correct.
            self.warn(DecodeWarning::MalformedSegmentSkipped)?;
        }

        let mut scan_components = Vec::with_capacity(num_components as usize);

        let permissive = self.strictness == Strictness::Permissive;

        for _ in 0..num_components {
            let component_id = self.read_u8()?;
            let tables = self.read_u8()?;
            let mut dc_table = tables >> 4;
            let mut ac_table = tables & 0x0F;

            // Validate Huffman table indexes
            if dc_table as usize >= MAX_HUFFMAN_TABLES {
                if permissive {
                    dc_table = 0; // Fallback to table 0
                } else {
                    return Err(Error::invalid_jpeg_data(
                        "SOS DC Huffman table index out of range",
                    ));
                }
            }
            if ac_table as usize >= MAX_HUFFMAN_TABLES {
                if permissive {
                    ac_table = 0; // Fallback to table 0
                } else {
                    return Err(Error::invalid_jpeg_data(
                        "SOS AC Huffman table index out of range",
                    ));
                }
            }

            // Find component index
            let comp_idx = self.components[..self.num_components as usize]
                .iter()
                .position(|c| c.id == component_id)
                .ok_or(Error::invalid_jpeg_data("unknown component in scan"))?;

            // Reject duplicate component IDs within a single scan.
            // A duplicate would write the same coefficient buffer twice,
            // producing corrupted output. (libjpeg-turbo also rejects this.)
            if scan_components.iter().any(|&(idx, _, _)| idx == comp_idx) {
                return Err(Error::invalid_jpeg_data("duplicate component in scan"));
            }

            scan_components.push((comp_idx, dc_table, ac_table));
        }

        let ss = self.read_u8()?; // Spectral selection start
        let se = self.read_u8()?; // Spectral selection end
        let ah_al = self.read_u8()?;
        let ah = ah_al >> 4;
        let al = ah_al & 0x0F;

        // Validate spectral selection (must be 0-63, and Ss <= Se)
        //
        // IJG libjpeg v9+ with `-block N` (N > 8) produces non-standard JPEGs
        // where Se can be up to N*N-1 (e.g., block 16 → Se up to 255). These
        // require fundamentally different DCT sizes and are not supported by any
        // decoder except IJG's own.
        if ss > 63 || se > 63 {
            return Err(Error::unsupported_feature(
                "This JPEG uses non-standard DCT block sizes (spectral selection beyond 0..63) and is not standards-compliant. It was likely created with IJG libjpeg v9+ or v10, which support proprietary block size extensions that no other decoder implements.",
            ));
        }
        if ss > se {
            return Err(Error::invalid_jpeg_data(
                "SOS Ss (spectral start) exceeds Se (spectral end)",
            ));
        }

        // Validate successive approximation (Ah and Al must be 0-13)
        if ah > 13 {
            return Err(Error::invalid_jpeg_data(
                "SOS Ah (successive approximation high) out of range (max 13)",
            ));
        }
        if al > 13 {
            return Err(Error::invalid_jpeg_data(
                "SOS Al (successive approximation low) out of range (max 13)",
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
                    if self.decode_mode == super::DecodeMode::Auto
                        && self.can_use_streaming()
                        && self.streaming_rgb.is_none()
                    {
                        // Use streaming decode for all baseline subsampling modes
                        let rgb = self.decode_baseline_streaming(&scan_components, stop)?;
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
                // Allocate parallel storage for coefficient counts (tiered IDCT).
                // Use the fallible try_alloc_filled rather than vec![64u8; ..] so
                // that an OOM on this allocation surfaces as Error::AllocationFailed
                // — without this, a successful try_alloc_dct_blocks (which on Linux
                // can succeed with lazy-committed zero pages) followed by an
                // infallible vec! could panic on physical commit.
                self.coeff_counts
                    .push(crate::foundation::alloc::try_alloc_filled(
                        num_blocks,
                        64u8,
                        "allocating coefficient counts",
                    )?);
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
        // Enable RST resync for all non-Strict modes. Zero overhead on valid
        // input (only gates error-path recovery). On mismatch, resync_to_restart()
        // scans forward for the next RST marker and continues decoding.
        if self.strictness != Strictness::Strict {
            decoder.set_permissive_rst(true);
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

        // JBRD padding-bit accumulator borrow. Pulled out of `self` here so
        // the MCU loop body can mutate it without conflicting with the
        // immutable borrow of `self.dc_tables`/`self.ac_tables` held by the
        // entropy decoder.
        let mut jbrd_padding_bits = self.jbrd_padding_bits.as_mut();

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
                    // JBRD: capture entropy-segment pad bits before aligning.
                    // Per-RST padding bits are part of the source bitstream;
                    // byte-exact JPEG-XL transcoding (djxl --reconstruct_jpeg)
                    // needs them to mirror the source encoder's zero-padding
                    // behaviour. No-op when JBRD tracking is disabled.
                    if let Some(buf) = jbrd_padding_bits.as_deref_mut() {
                        buf.extend_from_slice(&decoder.partial_byte_padding_bits());
                    }
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

        // JBRD: capture end-of-scan entropy-segment pad bits. The next marker
        // is whatever follows the scan in the bitstream (DHT/SOS/EOI/...).
        // Mirrors libjxl's `FinishStream` call at `ProcessScan` end. No-op
        // when JBRD tracking is disabled.
        if let Some(buf) = jbrd_padding_bits {
            buf.extend_from_slice(&decoder.partial_byte_padding_bits());
        }

        // Extract warning flags (decoder borrows self.dc_tables/ac_tables)
        let had_ac_overflow = decoder.had_ac_overflow;
        let had_invalid_huffman = decoder.had_invalid_huffman;
        let rst_resyncs = decoder.rst_resync_count();
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
        if rst_resyncs > 0 {
            self.warn(DecodeWarning::RestartMarkerResync { count: rst_resyncs })?;
        }

        Ok(())
    }

    /// Check if streaming decode can be used.
    ///
    /// Streaming is supported for baseline grayscale, 4:4:4, 4:2:0, and 4:2:2.
    /// Only standard sampling factor combinations are accepted.
    pub(super) fn can_use_streaming(&self) -> bool {
        // Must be baseline (not progressive — progressive needs multi-scan coefficient storage)
        if self.mode != JpegMode::Baseline {
            return false;
        }
        // f32 IDCT required (dimension-swapping transforms need symmetric IDCT) —
        // streaming path uses integer IDCT only, so fall back to buffered path.
        if self.force_f32_idct {
            return false;
        }
        // XYB color space needs coefficient-based decode: the streaming path
        // assumes YCbCr→RGB conversion, which produces wrong colors for XYB.
        // Coefficient storage is required so the output stage can run the
        // XYB→linear→sRGB conversion.
        if self
            .icc_profile
            .as_ref()
            .map(|p| crate::color::icc::is_xyb_profile(p))
            .unwrap_or(false)
        {
            return false;
        }
        // Grayscale (1 component) and YCbCr (3 components) are supported.
        // Grayscale streaming produces 1bpp gray or 4bpp BGRA depending on
        // streaming_output_format. Other component counts are unsupported.
        if self.num_components != 1 && self.num_components != 3 {
            return false;
        }
        // Grayscale: streamable only when sampling factors are 1x1.
        //
        // JPEG allows single-component frames with Hi/Vi > 1 (e.g. h_samp=2,
        // v_samp=2 on a grayscale file). Per ISO/IEC 10918-1 A.2.3, a
        // non-interleaved scan (Ns=1) has MCU = 1 data unit regardless of
        // sampling factors — but our streaming buffer sizing assumes the
        // component fills a max_h_samp × max_v_samp MCU, which is wrong
        // when max_h_samp/max_v_samp are clamped to 1 for grayscale. Fall
        // back to the coefficient-buffering path for these files.
        if self.num_components == 1 {
            let h = self.components[0].h_samp_factor;
            let v = self.components[0].v_samp_factor;
            return h == 1 && v == 1;
        }
        // Only accept standard sampling factor combinations
        {
            let y_h = self.components[0].h_samp_factor;
            let y_v = self.components[0].v_samp_factor;
            let c_h = self.components[1].h_samp_factor;
            let c_v = self.components[1].v_samp_factor;
            let c2_h = self.components[2].h_samp_factor;
            let c2_v = self.components[2].v_samp_factor;
            // Cb and Cr must have matching sampling factors
            if c_h != c2_h || c_v != c2_v {
                return false;
            }
            // Supported standard modes only:
            // 4:4:4 = Y(1x1) Cb(1x1) Cr(1x1)
            // 4:2:0 = Y(2x2) Cb(1x1) Cr(1x1)
            // 4:2:2 = Y(2x1) Cb(1x1) Cr(1x1)
            match (y_h, y_v, c_h, c_v) {
                (1, 1, 1, 1) => true, // 4:4:4
                (2, 2, 1, 1) => true, // 4:2:0
                (2, 1, 1, 1) => true, // 4:2:2
                _ => false,
            }
        }
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

        // Enable lenient/permissive error recovery
        if matches!(
            self.strictness,
            Strictness::Lenient | Strictness::Permissive
        ) {
            decoder.set_lenient(true);
        }
        // Enable RST resync for all non-Strict modes. Zero overhead on valid
        // input (only gates error-path recovery). On mismatch, resync_to_restart()
        // scans forward for the next RST marker and continues decoding.
        if self.strictness != Strictness::Strict {
            decoder.set_permissive_rst(true);
        }

        for (_comp_idx, dc_table, ac_table) in scan_components {
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
            // Install at the FILE table index (matches how decode_block_into
            // looks it up via ac_table_idx/dc_table_idx), not comp_idx.
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

        // Allocate strip buffers for one MCU row (8 rows of pixels)
        // Note: All elements are written by IDCT before color conversion reads them
        let strip_size = strip_width * 8;
        let mut y_strip: Vec<i16> = try_alloc_maybeuninit(strip_size, "Y strip buffer")?;
        let mut cb_strip: Vec<i16> = try_alloc_maybeuninit(strip_size, "Cb strip buffer")?;
        let mut cr_strip: Vec<i16> = try_alloc_maybeuninit(strip_size, "Cr strip buffer")?;

        // Determine output pixel format: 4bpp direct BGRA/RGBA when hinted.
        let (out_bpp, out_4bpp, swap_rb) = match self.streaming_output_format {
            Some(PixelFormat::Bgra | PixelFormat::Bgrx) => (4usize, true, true),
            Some(PixelFormat::Rgba) => (4, true, false),
            _ => (3, false, false),
        };

        // Allocate output buffer (3bpp RGB or 4bpp BGRA/RGBA)
        let rgb_size = checked_size_2d(width, height).and_then(|s| checked_size_2d(s, out_bpp))?;
        let mut rgb: Vec<u8> = try_alloc_maybeuninit(rgb_size, "output buffer")?;

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
                        match self.idct_method {
                            super::super::IdctMethod::Libjpeg => {
                                super::super::idct_int::idct_int_tiered_libjpeg(
                                    &mut dequant_buf,
                                    &mut strip[dst_offset..],
                                    strip_width,
                                    coeff_count,
                                );
                            }
                            super::super::IdctMethod::Jpegli => {
                                idct_int_tiered(
                                    &mut dequant_buf,
                                    &mut strip[dst_offset..],
                                    strip_width,
                                    coeff_count,
                                );
                            }
                        }
                    }
                }

                mcu_count += 1;
            }

            // Color convert this MCU row directly to output buffer
            let y_start = mcu_y * 8;
            let rows_this_mcu = 8.min(height.saturating_sub(y_start));
            let cols_this_mcu = width.min(strip_width);
            let is_rgb = self.is_rgb_jpeg();

            for row in 0..rows_this_mcu {
                let strip_offset = row * strip_width;
                let rgb_offset = (y_start + row) * width * out_bpp;

                if is_rgb && out_4bpp {
                    for px in 0..cols_this_mcu {
                        let i = strip_offset + px;
                        let o = rgb_offset + px * 4;
                        if swap_rb {
                            rgb[o] = cr_strip[i].clamp(0, 255) as u8;
                            rgb[o + 1] = cb_strip[i].clamp(0, 255) as u8;
                            rgb[o + 2] = y_strip[i].clamp(0, 255) as u8;
                        } else {
                            rgb[o] = y_strip[i].clamp(0, 255) as u8;
                            rgb[o + 1] = cb_strip[i].clamp(0, 255) as u8;
                            rgb[o + 2] = cr_strip[i].clamp(0, 255) as u8;
                        }
                        rgb[o + 3] = 255;
                    }
                } else if is_rgb {
                    for px in 0..cols_this_mcu {
                        let i = strip_offset + px;
                        let o = rgb_offset + px * 3;
                        rgb[o] = y_strip[i].clamp(0, 255) as u8;
                        rgb[o + 1] = cb_strip[i].clamp(0, 255) as u8;
                        rgb[o + 2] = cr_strip[i].clamp(0, 255) as u8;
                    }
                } else if out_4bpp {
                    ycbcr_planes_i16_to_xrgba_u8(
                        &y_strip[strip_offset..strip_offset + cols_this_mcu],
                        &cb_strip[strip_offset..strip_offset + cols_this_mcu],
                        &cr_strip[strip_offset..strip_offset + cols_this_mcu],
                        &mut rgb[rgb_offset..rgb_offset + cols_this_mcu * 4],
                        swap_rb,
                    );
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
        let rst_resyncs = decoder.rst_resync_count();
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
        if rst_resyncs > 0 {
            self.warn(DecodeWarning::RestartMarkerResync { count: rst_resyncs })?;
        }

        Ok(rgb)
    }

    /// Streaming decode for baseline subsampled images (4:2:0, 4:2:2, grayscale).
    /// Combines Huffman decode + dequantize + IDCT + upsample + color convert in one pass.
    /// No coefficient storage — processes MCU row by row directly to RGB output.
    ///
    /// For fancy h2v2, uses double-buffered Y and chroma strips with a 1-row lag
    /// so that each MCU row's chroma has correct above and below context for
    /// triangle filter interpolation.
    pub(super) fn decode_baseline_streaming(
        &mut self,
        scan_components: &[(usize, u8, u8)],
        stop: &impl Stop,
    ) -> Result<Vec<u8>> {
        if self.height == 0 {
            return Err(Error::unsupported_feature(
                "DNL mode (height=0 in SOF) not supported for streaming decode",
            ));
        }

        // ---- Phase 1: geometry ----
        let geom = baseline_streaming::StreamingGeometry::from_parser(self);

        // Delegate to 4:4:4 path (already optimized)
        if !geom.is_grayscale && geom.max_h_samp == 1 && geom.max_v_samp == 1 {
            return self.decode_baseline_streaming_rgb(scan_components, stop);
        }

        // ---- Phase 2: validation ----
        // Check for missing DHT — warn (and propagate if escalated to error).
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

        let quant_tables: Vec<&[u16; DCT_BLOCK_SIZE]> = (0..self.num_components as usize)
            .map(|i| {
                self.quant_tables[self.components[i].quant_table_idx as usize]
                    .as_ref()
                    .ok_or(Error::internal("missing quantization table"))
            })
            .collect::<Result<Vec<_>>>()?;

        // ---- Phase 3: buffer allocation ----
        let mut bufs = baseline_streaming::StreamingBuffers::allocate(
            &geom,
            self.chroma_upsampling,
            self.streaming_output_format,
        )?;

        // ---- Phase 4: kernel selection ----
        let use_fused_box = !geom.is_grayscale
            && geom.h_ratio == 2
            && geom.v_ratio == 2
            && matches!(
                self.chroma_upsampling,
                super::super::ChromaUpsampling::NearestNeighbor
            );
        let upsample_fn =
            baseline_streaming::select_upsample_fn(&geom, self.chroma_upsampling, use_fused_box)?;
        let idct_fn = baseline_streaming::select_idct_fn(self.idct_method);
        let is_rgb = !geom.is_grayscale && self.is_rgb_jpeg();

        // ---- Phase 5: entropy decoder setup ----
        let mut decoder = baseline_streaming::setup_entropy_decoder(self, scan_components);

        // ---- Phase 6: main MCU-row dispatch ----
        let c_data_offset = if bufs.need_fancy {
            geom.c_strip_width
        } else {
            0
        };
        let downsampled_w = (geom.width + geom.h_ratio - 1) / geom.h_ratio;

        let mut state = baseline_streaming::McuRowState::new();
        let inputs = baseline_streaming::LoopInputs {
            scan_components,
            components: &self.components,
            quant_tables: &quant_tables,
            restart_interval: self.restart_interval as u32,
            idct_fn,
            is_rgb,
        };

        if bufs.need_fancy {
            let upsample = upsample_fn.expect("fancy h2v2 requires an upsample fn");
            baseline_streaming::run_fancy_h2v2_loop(
                &mut decoder,
                &inputs,
                &geom,
                &mut bufs,
                &mut state,
                upsample,
                c_data_offset,
                downsampled_w,
                stop,
            )?;
        } else {
            baseline_streaming::run_simple_loop(
                &mut decoder,
                &inputs,
                &geom,
                &mut bufs,
                &mut state,
                upsample_fn,
                use_fused_box,
                c_data_offset,
                stop,
            )?;
        }

        // ---- Phase 7: finalize (warnings + position) ----
        // Snapshot stats in a sub-scope so the decoder's borrow of `self`
        // is released before `finalize_streaming` re-borrows mutably.
        let stats = {
            let stats = baseline_streaming::DecoderStats::snapshot(&decoder);
            let _ = decoder;
            stats
        };
        baseline_streaming::finalize_streaming(self, stats, &state, geom.mcu_rows, geom.mcu_cols)?;

        Ok(bufs.rgb)
    }
}
