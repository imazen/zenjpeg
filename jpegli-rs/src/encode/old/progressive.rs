//! Progressive JPEG encoding functions.
//!
//! These methods handle multi-scan progressive encoding:
//! - DC and AC scan encoding
//! - Successive approximation (refinement scans)
//! - Scan script generation

use super::super::{pad_ycbcr_planes_subsampled, Encoder, ProgressiveScan};
use crate::alloc::try_with_capacity;
use crate::consts::{DCT_BLOCK_SIZE, DCT_SIZE, MARKER_EOI, XYB_ICC_PROFILE};
use crate::entropy::EntropyEncoder;
use crate::error::{Error, Result};
use crate::huffman::optimize::{
    ContextConfig, FrequencyCounter, OptimizedHuffmanTables, OptimizedTable, ProgressiveTokenBuffer,
};
use crate::huffman::HuffmanEncodeTable;
use crate::quant::aq::compute_aq_strength_map;
use crate::quant::{self, QuantTable, ZeroBiasParams};
use crate::types::{ChromaDownsampling, PixelFormat, Subsampling};
use enough::Stop;

impl Encoder {
    /// Builds OptimizedHuffmanTables from the clustered tables.
    /// Currently unused - kept for potential debugging or future use.
    #[allow(dead_code)]
    pub(crate) fn build_progressive_huffman_tables(
        &self,
        tables: &[OptimizedTable],
        num_components: usize,
        num_dc_tables: usize,
    ) -> Result<OptimizedHuffmanTables> {
        // Tables are arranged: DC clusters first, then AC clusters
        // num_dc_tables tells us where DC ends and AC begins

        let dc_luma = tables.first().cloned().unwrap_or_else(|| {
            // Create a minimal default table using jpegli algorithm
            let mut counter = FrequencyCounter::new();
            counter.count(0);
            counter
                .generate_table_with_method(crate::types::HuffmanMethod::JpegliCreateTree)
                .unwrap()
        });

        // DC chroma is the second DC table if it exists
        let dc_chroma = if num_components > 1 && num_dc_tables > 1 {
            tables.get(1).cloned().unwrap_or_else(|| dc_luma.clone())
        } else {
            dc_luma.clone()
        };

        // AC tables start after DC tables
        let ac_luma = tables.get(num_dc_tables).cloned().unwrap_or_else(|| {
            let mut counter = FrequencyCounter::new();
            counter.count(0);
            counter
                .generate_table_with_method(crate::types::HuffmanMethod::JpegliCreateTree)
                .unwrap()
        });

        // AC chroma is the second AC table if it exists
        let ac_chroma = if num_components > 1 && tables.len() > num_dc_tables + 1 {
            tables
                .get(num_dc_tables + 1)
                .cloned()
                .unwrap_or_else(|| ac_luma.clone())
        } else {
            ac_luma.clone()
        };

        Ok(OptimizedHuffmanTables {
            dc_luma,
            ac_luma,
            dc_chroma,
            ac_chroma,
        })
    }

    /// Replays tokens for a progressive scan with optimized tables.
    ///
    /// # Arguments
    /// * `context_config` - Context configuration for proper AC context lookup
    /// * `tables` - All Huffman tables (DC tables first, then AC tables)
    /// * `num_dc_tables` - Number of DC tables in the tables vector
    /// * `context_map` - Maps context indices to table indices (from clustering)
    ///   - DC contexts 0..ac_offset map to DC table indices (0..num_dc_tables)
    ///   - AC contexts ac_offset.. map to total table indices (num_dc_tables + offset)
    /// * `ac_slot_ids` - Maps AC table index to JPEG slot ID (0-3)
    /// * `tables_emitted` - Number of tables emitted so far (DC + AC)
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn replay_progressive_scan(
        &self,
        token_buffer: &ProgressiveTokenBuffer,
        scan_idx: usize,
        scan: &ProgressiveScan,
        is_color: bool,
        context_config: &ContextConfig,
        tables: &[OptimizedTable],
        num_dc_tables: usize,
        context_map: &[usize],
        ac_slot_ids: &[usize],
        tables_emitted: usize,
    ) -> Result<Vec<u8>> {
        // Estimate output size from token count (~2 bytes per token average)
        let scan_info = token_buffer.scan_info.get(scan_idx);
        let estimated_tokens = scan_info
            .map(|s| s.num_tokens + s.ref_tokens.len())
            .unwrap_or(1024);
        let mut encoder = EntropyEncoder::with_capacity(estimated_tokens * 2);
        let num_components = if is_color { 3 } else { 1 };

        // Set up DC Huffman tables (up to 4)
        for (i, table) in tables.iter().take(num_dc_tables).enumerate() {
            encoder.set_dc_table(i, &table.table);
        }

        // Set up AC Huffman tables using slot IDs
        // Only load tables that have been emitted via DHT markers
        let num_ac_emitted = tables_emitted.saturating_sub(num_dc_tables);
        for (i, table) in tables
            .iter()
            .skip(num_dc_tables)
            .take(num_ac_emitted)
            .enumerate()
        {
            // Use the slot ID from ac_slot_ids (cycles 0-3)
            let slot = ac_slot_ids.get(i).copied().unwrap_or(i % 4);
            encoder.set_ac_table(slot, &table.table);
        }

        if self.config.restart_interval > 0 {
            encoder.set_restart_interval(self.config.restart_interval);
        }

        // Get scan info
        let scan_info = token_buffer
            .scan_info
            .get(scan_idx)
            .ok_or(Error::InternalError {
                reason: "Scan info not found",
            })?;

        if scan.ss == 0 && scan.se == 0 {
            // DC scan: replay DC tokens
            // Use context_map directly for DC (component index -> table index)
            let tokens = token_buffer.scan_tokens(scan_idx);
            let dc_context_map: Vec<usize> = (0..4)
                .map(|c| {
                    if c < num_components && c < context_map.len() {
                        context_map[c]
                    } else {
                        0
                    }
                })
                .collect();
            encoder.write_dc_tokens(tokens, &dc_context_map)?;
        } else if scan.ah == 0 {
            // AC first scan: replay AC tokens
            // Use context_config for per-scan AC context lookup
            let ac_context = context_config.ac_context(scan_idx, 0);
            let table_idx = if ac_context < context_map.len() {
                context_map[ac_context].saturating_sub(num_dc_tables)
            } else {
                0
            };
            // Convert table index to slot ID
            let slot_id = ac_slot_ids.get(table_idx).copied().unwrap_or(table_idx % 4);
            let tokens = token_buffer.scan_tokens(scan_idx);
            encoder.write_ac_first_tokens(tokens, slot_id)?;
        } else {
            // AC refinement scan: replay refinement tokens
            // Use context_config for per-scan AC context lookup
            let ac_context = context_config.ac_context(scan_idx, 0);
            let table_idx = if ac_context < context_map.len() {
                context_map[ac_context].saturating_sub(num_dc_tables)
            } else {
                0
            };
            // Convert table index to slot ID
            let slot_id = ac_slot_ids.get(table_idx).copied().unwrap_or(table_idx % 4);
            // Debug dump if DUMP_RUST_AC_REFINEMENT env var is set
            if std::env::var("DUMP_RUST_AC_REFINEMENT").is_ok() {
                scan_info.debug_dump(scan_idx);
            }
            encoder.write_ac_refinement_tokens(scan_info, slot_id)?;
        }

        Ok(encoder.finish())
    }

    /// Returns the progressive scan script for level 2.
    pub(crate) fn get_progressive_scan_script(&self, is_color: bool) -> Vec<ProgressiveScan> {
        let num_components = if is_color { 3 } else { 1 };
        let mut scans = Vec::new();

        // For XYB mode, always use non-interleaved DC scans (matches C++ jpegli)
        // For 4:4:4 YCbCr subsampling, DC can be interleaved
        let dc_interleaved =
            !self.config.use_xyb && matches!(self.config.subsampling, Subsampling::S444);

        // DC first scan
        if dc_interleaved && is_color {
            // Interleaved DC for all components
            scans.push(ProgressiveScan {
                components: vec![0, 1, 2],
                ss: 0,
                se: 0,
                ah: 0,
                al: 0,
            });
        } else {
            // Non-interleaved DC
            for c in 0..num_components {
                scans.push(ProgressiveScan {
                    components: vec![c],
                    ss: 0,
                    se: 0,
                    ah: 0,
                    al: 0,
                });
            }
        }

        // AC scans are always non-interleaved
        // Progressive Level 2 with successive approximation (matches C++ jpegli)
        //
        // IMPORTANT: Scan order must match C++ (encode.cc:141-152):
        // Iterate over scan TYPES first, then components.
        // This groups similar spectral bands together for better histogram clustering.
        // C++ order: [all AC 1-2] then [all AC 3-63 first] then [all refinements]
        // NOT: [Y all scans] then [Cb all scans] then [Cr all scans]
        let use_refinement = true;

        if use_refinement {
            // Level 2: with successive approximation
            // AC 1-2: full precision (low frequency, most visible) - all components
            for c in 0..num_components {
                scans.push(ProgressiveScan {
                    components: vec![c],
                    ss: 1,
                    se: 2,
                    ah: 0,
                    al: 0,
                });
            }

            // AC 3-63 first pass: top bits only (Al=2 means bits 2+) - all components
            for c in 0..num_components {
                scans.push(ProgressiveScan {
                    components: vec![c],
                    ss: 3,
                    se: 63,
                    ah: 0,
                    al: 2,
                });
            }

            // AC 3-63 refinement: bit 1 (Ah=2, Al=1) - all components
            for c in 0..num_components {
                scans.push(ProgressiveScan {
                    components: vec![c],
                    ss: 3,
                    se: 63,
                    ah: 2,
                    al: 1,
                });
            }

            // AC 3-63 refinement: bit 0 (Ah=1, Al=0) - all components
            for c in 0..num_components {
                scans.push(ProgressiveScan {
                    components: vec![c],
                    ss: 3,
                    se: 63,
                    ah: 1,
                    al: 0,
                });
            }
        } else {
            // Level 0: no successive approximation (simpler, works)
            for c in 0..num_components {
                scans.push(ProgressiveScan {
                    components: vec![c],
                    ss: 1,
                    se: 63,
                    ah: 0,
                    al: 0,
                });
            }
        }

        scans
    }

    /// Encodes a single progressive scan.
    pub(crate) fn encode_progressive_scan(
        &self,
        y_blocks: &[[i16; DCT_BLOCK_SIZE]],
        cb_blocks: &[[i16; DCT_BLOCK_SIZE]],
        cr_blocks: &[[i16; DCT_BLOCK_SIZE]],
        scan: &ProgressiveScan,
        is_color: bool,
        tables: &Option<OptimizedHuffmanTables>,
    ) -> Result<Vec<u8>> {
        // Estimate output size: DC scans ~10 bytes/block, AC scans ~50 bytes/block
        let total_blocks = y_blocks.len() + cb_blocks.len() + cr_blocks.len();
        let bytes_per_block = if scan.ss == 0 { 10 } else { 50 };
        let mut encoder = EntropyEncoder::with_capacity(total_blocks * bytes_per_block);

        // Set up Huffman tables
        if let Some(ref opt_tables) = tables {
            encoder.set_dc_table(0, &opt_tables.dc_luma.table);
            encoder.set_ac_table(0, &opt_tables.ac_luma.table);
            if is_color {
                encoder.set_dc_table(1, &opt_tables.dc_chroma.table);
                encoder.set_ac_table(1, &opt_tables.ac_chroma.table);
            }
        } else {
            encoder.set_dc_table(0, HuffmanEncodeTable::std_dc_luminance());
            encoder.set_ac_table(0, HuffmanEncodeTable::std_ac_luminance());
            if is_color {
                encoder.set_dc_table(1, HuffmanEncodeTable::std_dc_chrominance());
                encoder.set_ac_table(1, HuffmanEncodeTable::std_ac_chrominance());
            }
        }

        if self.config.restart_interval > 0 {
            encoder.set_restart_interval(self.config.restart_interval);
        }

        let width = self.config.width as usize;
        let height = self.config.height as usize;
        let blocks_h = (width + DCT_SIZE - 1) / DCT_SIZE;
        let blocks_v = (height + DCT_SIZE - 1) / DCT_SIZE;

        // Determine scan type and encode accordingly
        if scan.ss == 0 && scan.se == 0 {
            // DC scan (first or refinement)
            self.encode_dc_scan(
                &mut encoder,
                y_blocks,
                cb_blocks,
                cr_blocks,
                scan,
                blocks_h,
                blocks_v,
                is_color,
            )?;
        } else if scan.ah == 0 {
            // AC first scan
            self.encode_ac_first_scan(
                &mut encoder,
                y_blocks,
                cb_blocks,
                cr_blocks,
                scan,
                blocks_h,
                blocks_v,
                is_color,
            )?;
        } else {
            // AC refinement scan
            self.encode_ac_refine_scan(
                &mut encoder,
                y_blocks,
                cb_blocks,
                cr_blocks,
                scan,
                blocks_h,
                blocks_v,
                is_color,
            )?;
        }

        Ok(encoder.finish())
    }

    /// Encodes DC scan (first or refinement).
    #[allow(clippy::too_many_arguments)]
    fn encode_dc_scan(
        &self,
        encoder: &mut EntropyEncoder,
        y_blocks: &[[i16; DCT_BLOCK_SIZE]],
        cb_blocks: &[[i16; DCT_BLOCK_SIZE]],
        cr_blocks: &[[i16; DCT_BLOCK_SIZE]],
        scan: &ProgressiveScan,
        blocks_h: usize,
        blocks_v: usize,
        is_color: bool,
    ) -> Result<()> {
        for by in 0..blocks_v {
            for bx in 0..blocks_h {
                let block_idx = by * blocks_h + bx;

                for (comp_num, &comp_idx) in scan.components.iter().enumerate() {
                    let blocks: &[[i16; DCT_BLOCK_SIZE]] = match comp_idx {
                        0 => y_blocks,
                        1 => cb_blocks,
                        2 => cr_blocks,
                        _ => {
                            return Err(Error::InternalError {
                                reason: "Invalid component index",
                            })
                        }
                    };

                    if block_idx >= blocks.len() {
                        continue;
                    }

                    let dc = blocks[block_idx][0];
                    // For XYB: all components use table 0
                    // For YCbCr: luma uses 0, chroma uses 1
                    let table = if self.config.use_xyb {
                        0
                    } else if is_color && comp_idx > 0 {
                        1
                    } else {
                        0
                    };

                    encoder.encode_dc_progressive(dc, comp_num, table, scan.al, scan.ah)?;
                }
            }
        }

        Ok(())
    }

    /// Encodes AC first scan (Ah=0, ss>0).
    #[allow(clippy::too_many_arguments)]
    fn encode_ac_first_scan(
        &self,
        encoder: &mut EntropyEncoder,
        y_blocks: &[[i16; DCT_BLOCK_SIZE]],
        cb_blocks: &[[i16; DCT_BLOCK_SIZE]],
        cr_blocks: &[[i16; DCT_BLOCK_SIZE]],
        scan: &ProgressiveScan,
        blocks_h: usize,
        blocks_v: usize,
        is_color: bool,
    ) -> Result<()> {
        // AC first scan is always non-interleaved (single component)
        assert_eq!(scan.components.len(), 1);
        let comp_idx = scan.components[0];

        let blocks: &[[i16; DCT_BLOCK_SIZE]] = match comp_idx {
            0 => y_blocks,
            1 => cb_blocks,
            2 => cr_blocks,
            _ => {
                return Err(Error::InternalError {
                    reason: "Invalid component index",
                })
            }
        };

        // For XYB: all components use table 0
        // For YCbCr: luma uses 0, chroma uses 1
        let table_idx = if self.config.use_xyb {
            0
        } else if is_color && comp_idx > 0 {
            1
        } else {
            0
        };

        let mut eob_run = 0u16;

        for by in 0..blocks_v {
            for bx in 0..blocks_h {
                let block_idx = by * blocks_h + bx;

                if block_idx >= blocks.len() {
                    continue;
                }

                encoder.encode_ac_progressive_first(
                    &blocks[block_idx],
                    table_idx,
                    scan.ss,
                    scan.se,
                    scan.al,
                    &mut eob_run,
                )?;
            }
        }

        // Flush remaining EOB run
        encoder.flush_eob_run(table_idx, eob_run)?;

        Ok(())
    }

    /// Encodes AC refinement scan (Ah>0, ss>0).
    #[allow(clippy::too_many_arguments)]
    fn encode_ac_refine_scan(
        &self,
        encoder: &mut EntropyEncoder,
        y_blocks: &[[i16; DCT_BLOCK_SIZE]],
        cb_blocks: &[[i16; DCT_BLOCK_SIZE]],
        cr_blocks: &[[i16; DCT_BLOCK_SIZE]],
        scan: &ProgressiveScan,
        blocks_h: usize,
        blocks_v: usize,
        is_color: bool,
    ) -> Result<()> {
        // AC refinement scan is always non-interleaved
        assert_eq!(scan.components.len(), 1);
        let comp_idx = scan.components[0];

        let blocks: &[[i16; DCT_BLOCK_SIZE]] = match comp_idx {
            0 => y_blocks,
            1 => cb_blocks,
            2 => cr_blocks,
            _ => {
                return Err(Error::InternalError {
                    reason: "Invalid component index",
                })
            }
        };

        // For XYB: all components use table 0
        // For YCbCr: luma uses 0, chroma uses 1
        let table_idx = if self.config.use_xyb {
            0
        } else if is_color && comp_idx > 0 {
            1
        } else {
            0
        };

        let mut eob_run = 0u16;

        for by in 0..blocks_v {
            for bx in 0..blocks_h {
                let block_idx = by * blocks_h + bx;

                if block_idx >= blocks.len() {
                    continue;
                }

                encoder.encode_ac_progressive_refine(
                    &blocks[block_idx],
                    table_idx,
                    scan.ss,
                    scan.se,
                    scan.ah,
                    scan.al,
                    &mut eob_run,
                )?;
            }
        }

        // Flush remaining EOB run
        encoder.flush_refine_eob(table_idx, eob_run)?;

        Ok(())
    }
    /// Encodes progressive JPEG using XYB color space.
    ///
    /// This uses the same progressive scan structure as YCbCr encoding
    /// but with XYB color conversion and appropriate headers (ICC profile, APP14).
    fn encode_progressive_xyb(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Progressive mode requires Huffman optimization (already validated in encode_progressive)
        // But adding explicit check here for safety in case this is called directly
        if !self.config.optimize_huffman {
            return Err(Error::UnsupportedFeature {
                feature: "Progressive mode with fixed Huffman codes (use optimize_huffman=true)",
            });
        }

        // Use optimized Huffman tables if enabled (2-pass encoding)
        if self.config.optimize_huffman {
            return self.encode_progressive_xyb_optimized(data);
        }

        let mut output =
            crate::foundation::alloc::try_with_capacity(data.len() / 4, "progressive xyb output")?;
        let width = self.config.width as usize;
        let height = self.config.height as usize;

        // Convert sRGB to scaled XYB
        let (x_plane, y_plane, b_plane) = self.convert_to_scaled_xyb(data)?;

        // Downsample B channel to match C++ XYB behavior (2x2,2x2,1x1 subsampling)
        let b_downsampled = self.downsample_2x2_f32(&b_plane, width, height)?;
        let b_width = (width + 1) / 2;
        let b_height = (height + 1) / 2;

        // Generate XYB quantization tables
        // Use separate quantization tables for each component (matches C++ jpegli XYB mode)
        let x_quant = self.gen_quant_table(0, true, false);
        let y_quant = self.gen_quant_table(1, true, false);
        let b_quant = self.gen_quant_table(2, true, false);

        // Quantize all blocks for progressive encoding
        // Use X, Y, B as if they were Y, Cb, Cr for the progressive structure
        // Note: B is downsampled, so use b_downsampled instead of b_plane
        let (x_blocks, y_blocks, b_blocks) = self.quantize_all_blocks_xyb(
            &x_plane,
            &y_plane,
            &b_downsampled,
            width,
            height,
            b_width,
            b_height,
            &x_quant,
            &y_quant,
            &b_quant,
        )?;
        let is_color = self.config.pixel_format != PixelFormat::Gray;

        // Write XYB-specific headers
        self.write_header_xyb(&mut output)?;
        // Write APP14 Adobe marker for RGB (required by some decoders)
        self.write_app14_adobe(&mut output, 0)?; // 0 = RGB (no transform)
                                                 // Write XYB ICC profile
        self.write_icc_profile(&mut output, &XYB_ICC_PROFILE)?;
        // Write quantization tables
        self.write_quant_tables(&mut output, &x_quant, &y_quant, &b_quant)?;
        // Write SOF2 frame header for progressive XYB (with correct component IDs and subsampling)
        self.write_frame_header_xyb_progressive(&mut output)?;

        // Use standard Huffman tables (optimized tables could be added later)
        self.write_huffman_tables(&mut output)?;
        let tables: Option<OptimizedHuffmanTables> = None;

        if self.config.restart_interval > 0 {
            self.write_restart_interval(&mut output)?;
        }

        // Get progressive scan script
        let scans = self.get_progressive_scan_script(is_color);

        // Encode each scan (reusing the YCbCr progressive scan logic)
        for scan in &scans {
            self.write_progressive_scan_header(&mut output, scan, is_color)?;
            let scan_data = self.encode_progressive_scan(
                &x_blocks, &y_blocks, &b_blocks, scan, is_color, &tables,
            )?;
            output.extend_from_slice(&scan_data);
        }

        // Write EOI
        output.push(0xFF);
        output.push(MARKER_EOI);

        Ok(output)
    }

    /// Encodes progressive XYB JPEG with optimized Huffman tables (2-pass).
    ///
    /// This is similar to encode_progressive_optimized() but for XYB color space.
    /// It performs 2-pass encoding: first tokenizes all scans to collect statistics,
    /// then builds optimized Huffman tables and replays tokens.
    fn encode_progressive_xyb_optimized(&self, data: &[u8]) -> Result<Vec<u8>> {
        let mut output = crate::foundation::alloc::try_with_capacity(
            data.len() / 4,
            "progressive xyb optimized output",
        )?;
        let width = self.config.width as usize;
        let height = self.config.height as usize;

        // Convert sRGB to scaled XYB
        let (x_plane, y_plane, b_plane) = self.convert_to_scaled_xyb(data)?;

        // Downsample B channel (2x2,2x2,1x1 subsampling for XYB)
        let b_downsampled = self.downsample_2x2_f32(&b_plane, width, height)?;
        let b_width = (width + 1) / 2;
        let b_height = (height + 1) / 2;

        // Generate XYB quantization tables
        let x_quant = self.gen_quant_table(0, true, false);
        let y_quant = self.gen_quant_table(1, true, false);
        let b_quant = self.gen_quant_table(2, true, false);

        // Compute AQ map from Y plane (same as baseline XYB, using SIMD scaling)
        let y_plane_scaled = crate::encode_simd::scale_f32_slice_simd(&y_plane, 255.0)?;
        let y_quant_01 = y_quant.values[1];
        let aq_map = compute_aq_strength_map(&y_plane_scaled, width, height, y_quant_01)?;

        // Generate zero-bias parameters (same as baseline XYB)
        let effective_distance = quant::quant_vals_to_distance(&x_quant, &y_quant, &b_quant);
        let x_zero_bias = ZeroBiasParams::for_ycbcr(effective_distance, 0); // X uses luma params
        let y_zero_bias = ZeroBiasParams::for_ycbcr(effective_distance, 0); // Y uses luma params
        let b_zero_bias = ZeroBiasParams::for_ycbcr(effective_distance, 1); // B uses chroma params

        // Quantize all blocks WITH adaptive quantization (same as baseline)
        let (x_blocks, y_blocks, b_blocks) = self.quantize_all_blocks_xyb_with_aq_simple(
            &x_plane,
            &y_plane,
            &b_downsampled,
            width,
            height,
            b_width,
            b_height,
            &x_quant,
            &y_quant,
            &b_quant,
            &aq_map,
            &x_zero_bias,
            &y_zero_bias,
            &b_zero_bias,
        )?;

        // quantize_all_blocks_xyb_with_aq_simple produces blocks in MCU order:
        // - x_blocks[mcu_idx*4..mcu_idx*4+4] = 4 X blocks for mcu_idx
        // - y_blocks[mcu_idx*4..mcu_idx*4+4] = 4 Y blocks for mcu_idx
        // - b_blocks[mcu_idx] = 1 B block for mcu_idx
        //
        // But for non-interleaved progressive scans, the JPEG decoder expects
        // blocks in RASTER order (row by row), not MCU order.
        // So we must reorder X and Y blocks from MCU order to raster order.
        // B blocks don't need reordering since B has 1×1 sampling (1 block per MCU).
        let blocks_x = (width + 7) / 8;
        let blocks_y = (height + 7) / 8;
        let x_blocks_raster = Self::reorder_mcu_to_raster(&x_blocks, blocks_x, blocks_y)?;
        let y_blocks_raster = Self::reorder_mcu_to_raster(&y_blocks, blocks_x, blocks_y)?;

        let is_color = self.config.pixel_format != PixelFormat::Gray;
        let num_components = if is_color { 3 } else { 1 };

        // Define progressive scan script
        let scans = self.get_progressive_scan_script(is_color);

        // ========== PASS 1: TOKENIZATION ==========
        let mut token_buffer = ProgressiveTokenBuffer::new(num_components, scans.len());

        for scan in scans.iter() {
            // Calculate context for this scan
            // XYB: all components share same Huffman table (context 0 for DC, 3 for AC)
            // YCbCr: component-specific contexts for luma/chroma split
            let context = if scan.ss == 0 && scan.se == 0 {
                // DC scan: XYB uses context 0 for all, YCbCr uses component index
                if self.config.use_xyb {
                    0 // All XYB components use DC context 0
                } else {
                    scan.components[0] // YCbCr: component-specific DC context
                }
            } else {
                // AC scan: XYB uses context 3 for all, YCbCr uses component-specific
                if self.config.use_xyb {
                    num_components as u8 // All XYB components use AC context 3
                } else {
                    (num_components as u8) + scan.components[0] // YCbCr: offset contexts
                }
            };

            if scan.ss == 0 && scan.se == 0 {
                // DC scan - for XYB, DC scans are non-interleaved (one component per scan)
                // Use raster-ordered blocks for X and Y (decoder expects raster order)
                let blocks: Vec<&[[i16; DCT_BLOCK_SIZE]]> = scan
                    .components
                    .iter()
                    .map(|&c| match c {
                        0 => x_blocks_raster.as_slice(),
                        1 => y_blocks_raster.as_slice(),
                        2 => b_blocks.as_slice(), // B has 1×1 sampling, already in raster order
                        _ => &[][..],
                    })
                    .collect();
                let component_indices: Vec<usize> =
                    scan.components.iter().map(|&c| c as usize).collect();
                token_buffer.tokenize_dc_scan(&blocks, &component_indices, scan.al, scan.ah);
            } else if scan.ah == 0 {
                // AC first scan - use raster-ordered blocks for X and Y
                let blocks: &[[i16; DCT_BLOCK_SIZE]] = match scan.components[0] {
                    0 => &x_blocks_raster,
                    1 => &y_blocks_raster,
                    2 => &b_blocks, // B has 1×1 sampling, already in raster order
                    _ => {
                        return Err(Error::InternalError {
                            reason: "Invalid component",
                        })
                    }
                };
                token_buffer.tokenize_ac_first_scan(blocks, context, scan.ss, scan.se, scan.al);
            } else {
                // AC refinement scan - use raster-ordered blocks for X and Y
                let blocks: &[[i16; DCT_BLOCK_SIZE]] = match scan.components[0] {
                    0 => &x_blocks_raster,
                    1 => &y_blocks_raster,
                    2 => &b_blocks, // B has 1×1 sampling, already in raster order
                    _ => {
                        return Err(Error::InternalError {
                            reason: "Invalid component",
                        })
                    }
                };
                token_buffer.tokenize_ac_refinement_scan(
                    blocks, context, scan.ss, scan.se, scan.ah, scan.al,
                );
            }
        }

        // ========== GENERATE OPTIMIZED TABLES ==========
        // XYB mode uses merged tables (all components share one DC and one AC table)
        let opt_tables = token_buffer.generate_xyb_tables(num_components)?;

        // Create context config for XYB (sequential-style, ac_offset=4)
        let xyb_context_config = ContextConfig::for_sequential(num_components);

        // For XYB, all contexts map to table 0 (DC contexts 0..4 -> 0, AC contexts 4.. -> 1)
        let xyb_context_map: Vec<usize> = (0..xyb_context_config.num_contexts)
            .map(|c| {
                if c < xyb_context_config.ac_offset {
                    0
                } else {
                    1
                }
            })
            .collect();

        // Convert OptimizedHuffmanTables to vec format: [DC table 0, AC table 0]
        let xyb_tables_vec = vec![opt_tables.dc_luma.clone(), opt_tables.ac_luma.clone()];

        // ========== WRITE JPEG STRUCTURE ==========
        self.write_header_xyb(&mut output)?;
        self.write_app14_adobe(&mut output, 0)?; // 0 = RGB (no transform)
        self.write_icc_profile(&mut output, &XYB_ICC_PROFILE)?;
        self.write_quant_tables(&mut output, &x_quant, &y_quant, &b_quant)?;
        self.write_frame_header_xyb_progressive(&mut output)?;

        // Write optimized Huffman tables (XYB uses only table 0)
        self.write_huffman_tables_optimized(&mut output, &opt_tables)?;

        if self.config.restart_interval > 0 {
            self.write_restart_interval(&mut output)?;
        }

        // ========== PASS 2: REPLAY TOKENS ==========
        // XYB uses 1 DC table and 1 AC table (all components share)
        let xyb_num_dc_tables = 1;
        let xyb_ac_slot_ids = vec![0]; // Single AC table uses slot 0
        let xyb_tables_emitted = 2; // All tables emitted upfront (1 DC + 1 AC)
        for (scan_idx, scan) in scans.iter().enumerate() {
            self.write_progressive_scan_header_with_context(
                &mut output,
                scan_idx,
                scan,
                is_color,
                &xyb_context_config,
                &xyb_context_map,
                xyb_num_dc_tables,
            )?;
            let scan_data = self.replay_progressive_scan(
                &token_buffer,
                scan_idx,
                scan,
                is_color,
                &xyb_context_config,
                &xyb_tables_vec,
                xyb_num_dc_tables,
                &xyb_context_map,
                &xyb_ac_slot_ids,
                xyb_tables_emitted,
            )?;
            output.extend_from_slice(&scan_data);
        }

        // Write EOI
        output.push(0xFF);
        output.push(MARKER_EOI);

        Ok(output)
    }

    /// Encodes as progressive JPEG with cancellation support.
    ///
    /// Progressive level 2 uses the following scan script:
    /// 1. DC first: Ss=0, Se=0, Ah=0, Al=0 (DC only, full precision)
    /// 2. AC 1-2: Ss=1, Se=2, Ah=0, Al=0 (low AC, full precision)
    /// 3. AC 3-63 first: Ss=3, Se=63, Ah=0, Al=2 (high AC, top bits)
    /// 4. AC 3-63 refine: Ss=3, Se=63, Ah=2, Al=1 (bit 1 refinement)
    /// 5. AC 3-63 refine: Ss=3, Se=63, Ah=1, Al=0 (bit 0 refinement)
    pub(crate) fn encode_progressive_with_stop(
        &self,
        data: &[u8],
        _stop: &impl Stop,
    ) -> Result<Vec<u8>> {
        // TODO: Thread stop through progressive encoding pipeline
        // For now, cancellation is checked at the quantization level
        self.encode_progressive(data)
    }

    /// Encodes as progressive JPEG (level 2, matching cjpegli default).
    pub(crate) fn encode_progressive(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Progressive mode requires Huffman optimization because standard JPEG Huffman tables
        // are designed for baseline/sequential encoding and produce massive bloat (10-100×)
        // when used with progressive AC refinement scans. This matches C++ cjpegli behavior.
        if !self.config.optimize_huffman {
            return Err(Error::UnsupportedFeature {
                feature: "Progressive mode with fixed Huffman codes (use optimize_huffman=true)",
            });
        }

        // XYB progressive mode - route to specialized encoder
        if self.config.use_xyb {
            return self.encode_progressive_xyb(data);
        }

        // Use tokenization-based approach when optimizing Huffman tables
        if self.config.optimize_huffman {
            return self.encode_progressive_optimized(data);
        }

        let mut output = crate::foundation::alloc::try_with_capacity(
            data.len() / 4,
            "progressive ycbcr output",
        )?;
        let width = self.config.width as usize;
        let height = self.config.height as usize;
        let mcu_size = self.config.subsampling.mcu_size();

        // Convert to YCbCr using f32 precision
        let (y_plane, cb_plane, cr_plane) = self.convert_to_ycbcr_f32(data)?;

        // Pad planes to MCU-aligned dimensions for consistent edge handling.
        // For 4:4:4, all planes have the same dimensions.
        let ((y_padded, cb_padded, cr_padded), padded_w, padded_h, _, _) =
            pad_ycbcr_planes_subsampled(
                &y_plane,
                width,
                height,
                &cb_plane,
                &cr_plane,
                width,
                height,
                mcu_size,
                self.config.edge_padding,
            )?;

        // Generate quantization tables (3 separate tables like C++ cjpegli)
        // Progressive mode uses 4:4:4, so is_420 = false
        let y_quant = self.gen_quant_table(0, false, false);
        let cb_quant = self.gen_quant_table(1, false, false);
        let cr_quant = self.gen_quant_table(2, false, false);

        // Quantize all blocks to get full-precision coefficients (using padded planes)
        // For 4:4:4, all planes have the same padded dimensions
        let (y_blocks, cb_blocks, cr_blocks) = self.quantize_all_blocks_subsampled(
            &y_padded, padded_w, padded_h, &cb_padded, &cr_padded, padded_w, padded_h, &y_quant,
            &cb_quant, &cr_quant, None, // No cancellation support in this path
        )?;
        let is_color = self.config.pixel_format != PixelFormat::Gray;

        // Write JPEG structure
        self.write_header(&mut output)?;
        self.write_quant_tables(&mut output, &y_quant, &cb_quant, &cr_quant)?;
        self.write_frame_header(&mut output)?; // Uses SOF2 for progressive

        // For non-optimized progressive, use standard Huffman tables
        self.write_huffman_tables(&mut output)?;
        let tables: Option<OptimizedHuffmanTables> = None;

        if self.config.restart_interval > 0 {
            self.write_restart_interval(&mut output)?;
        }

        // Define progressive scan script (level 2)
        // For 4:4:4 (no subsampling), DC can be interleaved
        let scans = self.get_progressive_scan_script(is_color);

        // Encode each scan
        for scan in &scans {
            // Write SOS header for this scan
            self.write_progressive_scan_header(&mut output, scan, is_color)?;

            // Encode the scan data
            let scan_data = self.encode_progressive_scan(
                &y_blocks, &cb_blocks, &cr_blocks, scan, is_color, &tables,
            )?;
            output.extend_from_slice(&scan_data);
        }

        // Write EOI
        output.push(0xFF);
        output.push(MARKER_EOI);

        Ok(output)
    }

    /// Encodes progressive JPEG with optimized Huffman tables using two-pass tokenization.
    ///
    /// This approach:
    /// 1. Tokenizes all scans first to collect actual symbol usage
    /// 2. Builds histograms from actual tokens (not estimated baseline statistics)
    /// 3. Clusters similar histograms to minimize table overhead
    /// 4. Generates optimal Huffman tables from clustered histograms
    /// 5. Replays tokens with optimized tables
    fn encode_progressive_optimized(&self, data: &[u8]) -> Result<Vec<u8>> {
        let mut output = crate::foundation::alloc::try_with_capacity(
            data.len() / 4,
            "progressive optimized output",
        )?;
        let width = self.config.width as usize;
        let height = self.config.height as usize;
        let mcu_size = self.config.subsampling.mcu_size();

        // Get YCbCr planes with appropriate chroma handling based on downsampling method
        let (y_plane, cb_plane_final, cr_plane_final, c_width, c_height) =
            match self.config.chroma_downsampling {
                ChromaDownsampling::GammaAware | ChromaDownsampling::GammaAwareIterative => {
                    let use_iterative =
                        self.config.chroma_downsampling == ChromaDownsampling::GammaAwareIterative;
                    match self.config.subsampling {
                        Subsampling::S420 => self.convert_gamma_aware_420(data, use_iterative)?,
                        Subsampling::S422 => self.convert_gamma_aware_422(data, use_iterative)?,
                        Subsampling::S440 => self.convert_gamma_aware_440(data, use_iterative)?,
                        // No downsampling needed for 4:4:4
                        Subsampling::S444 => self.convert_intrinsic_with_subsampling(data)?,
                    }
                }
                ChromaDownsampling::Box => self.convert_intrinsic_with_subsampling(data)?,
            };

        // Pad planes to MCU-aligned dimensions for consistent edge handling.
        // This matches C++ jpegli's RowBuffer padding strategy.
        let ((y_padded, cb_padded, cr_padded), padded_w, padded_h, padded_cw, padded_ch) =
            pad_ycbcr_planes_subsampled(
                &y_plane,
                width,
                height,
                &cb_plane_final,
                &cr_plane_final,
                c_width,
                c_height,
                mcu_size,
                self.config.edge_padding,
            )?;

        // Generate quantization tables (3 separate tables like C++ cjpegli)
        // Apply 4:2:0 quality compensation if using 4:2:0 subsampling
        let is_420 = self.config.subsampling == Subsampling::S420;
        let y_quant = self.gen_quant_table(0, false, is_420);
        let cb_quant = self.gen_quant_table(1, false, is_420);
        let cr_quant = self.gen_quant_table(2, false, is_420);

        // Quantize all blocks using padded planes and dimensions
        let (y_blocks_padded, cb_blocks_padded, cr_blocks_padded) = self
            .quantize_all_blocks_subsampled(
                &y_padded, padded_w, padded_h, &cb_padded, &cr_padded, padded_cw, padded_ch,
                &y_quant, &cb_quant, &cr_quant, None, // No cancellation support in this path
            )?;

        // For non-interleaved progressive scans, the block count is based on ORIGINAL
        // dimensions, not padded dimensions. This differs from interleaved baseline scans
        // which use MCU-aligned block counts.
        //
        // Per JPEG spec for non-interleaved scan with component i:
        //   MCUx = ceil(X * Hi / (8 * Hmax))
        //   MCUy = ceil(Y * Vi / (8 * Vmax))
        //
        // For Y (Hi=Hmax, Vi=Vmax): ceil(width/8) x ceil(height/8) blocks
        // For Cb/Cr with 4:2:0: ceil(width/16) x ceil(height/16) blocks
        let orig_y_blocks_h = (width + 7) / 8;
        let orig_y_blocks_v = (height + 7) / 8;
        let padded_y_blocks_h = padded_w / 8;
        let padded_y_blocks_v = padded_h / 8;

        let y_blocks =
            if padded_y_blocks_h != orig_y_blocks_h || padded_y_blocks_v != orig_y_blocks_v {
                // Filter Y blocks: extract blocks that correspond to original dimensions
                let mut filtered =
                    try_with_capacity(orig_y_blocks_h * orig_y_blocks_v, "progressive Y blocks")?;
                for by in 0..orig_y_blocks_v {
                    for bx in 0..orig_y_blocks_h {
                        let padded_idx = by * padded_y_blocks_h + bx;
                        filtered.push(y_blocks_padded[padded_idx]);
                    }
                }
                filtered
            } else {
                y_blocks_padded
            };

        // Chroma blocks: same logic but using chroma dimensions
        let orig_c_blocks_h = (c_width + 7) / 8;
        let orig_c_blocks_v = (c_height + 7) / 8;
        let padded_c_blocks_h = padded_cw / 8;
        let padded_c_blocks_v = padded_ch / 8;

        let (cb_blocks, cr_blocks) =
            if padded_c_blocks_h != orig_c_blocks_h || padded_c_blocks_v != orig_c_blocks_v {
                let mut cb_filtered =
                    try_with_capacity(orig_c_blocks_h * orig_c_blocks_v, "progressive Cb blocks")?;
                let mut cr_filtered =
                    try_with_capacity(orig_c_blocks_h * orig_c_blocks_v, "progressive Cr blocks")?;
                for by in 0..orig_c_blocks_v {
                    for bx in 0..orig_c_blocks_h {
                        let padded_idx = by * padded_c_blocks_h + bx;
                        cb_filtered.push(cb_blocks_padded[padded_idx]);
                        cr_filtered.push(cr_blocks_padded[padded_idx]);
                    }
                }
                (cb_filtered, cr_filtered)
            } else {
                (cb_blocks_padded, cr_blocks_padded)
            };

        let is_color = self.config.pixel_format != PixelFormat::Gray;
        let num_components = if is_color { 3 } else { 1 };

        // Define progressive scan script
        let scans = self.get_progressive_scan_script(is_color);

        // ========== CREATE CONTEXT CONFIG ==========
        // Per C++ design (encode.cc:340-383): DC contexts are 0..num_components,
        // AC contexts start at 4 with one per component per AC scan.
        let context_config = ContextConfig::for_progressive(
            num_components,
            scans.iter().map(|s| (s.ss, s.se, s.components.len())),
        );

        // ========== PASS 1: TOKENIZATION ==========
        // Tokenize all scans to collect symbol statistics.
        // Use context_config.num_contexts to allocate proper histogram count.
        let mut token_buffer =
            ProgressiveTokenBuffer::new(num_components, context_config.num_contexts);

        for (scan_idx, scan) in scans.iter().enumerate() {
            // Calculate context for this scan using C++ context assignment:
            // - DC: component index (0-3)
            // - AC: context_config.ac_context(scan_idx, comp_in_scan)
            let context = if scan.ss == 0 && scan.se == 0 {
                // DC scan: use component index as context
                context_config.dc_context(scan.components[0] as usize) as u8
            } else {
                // AC scan: use per-scan context from config
                context_config.ac_context(scan_idx, 0) as u8
            };

            if scan.ss == 0 && scan.se == 0 {
                // DC scan
                let blocks: Vec<&[[i16; DCT_BLOCK_SIZE]]> = scan
                    .components
                    .iter()
                    .map(|&c| match c {
                        0 => y_blocks.as_slice(),
                        1 => cb_blocks.as_slice(),
                        2 => cr_blocks.as_slice(),
                        _ => &[][..],
                    })
                    .collect();
                let component_indices: Vec<usize> =
                    scan.components.iter().map(|&c| c as usize).collect();
                token_buffer.tokenize_dc_scan(&blocks, &component_indices, scan.al, scan.ah);
            } else if scan.ah == 0 {
                // AC first scan
                let blocks: &[[i16; DCT_BLOCK_SIZE]] = match scan.components[0] {
                    0 => &y_blocks,
                    1 => &cb_blocks,
                    2 => &cr_blocks,
                    _ => {
                        return Err(Error::InternalError {
                            reason: "Invalid component",
                        })
                    }
                };
                token_buffer.tokenize_ac_first_scan(blocks, context, scan.ss, scan.se, scan.al);
            } else {
                // AC refinement scan
                let blocks: &[[i16; DCT_BLOCK_SIZE]] = match scan.components[0] {
                    0 => &y_blocks,
                    1 => &cb_blocks,
                    2 => &cr_blocks,
                    _ => {
                        return Err(Error::InternalError {
                            reason: "Invalid component",
                        })
                    }
                };
                token_buffer.tokenize_ac_refinement_scan(
                    blocks, context, scan.ss, scan.se, scan.ah, scan.al,
                );
            }
        }

        // ========== GENERATE OPTIMIZED TABLES ==========
        // Use histogram clustering to find optimal table assignments.
        // Per C++ design: progressive mode can have more than 4 AC tables,
        // with slot IDs cycling through 0-3 and redefinition via DHT markers.
        // We allow up to 12 AC clusters (one per AC scan) to enable this.
        let (context_map, num_dc_tables, tables, ac_slot_ids) = token_buffer
            .generate_optimized_tables(
                4,                        // max DC clusters
                12,                       // max AC clusters (allows per-scan specialization)
                context_config.ac_offset, // num_dc_contexts (always 4 per C++ design, but clamped to actual)
                false,                    // force_baseline
            )?;

        // Debug: print context map and scan-to-table assignment
        if std::env::var("DUMP_CONTEXT_MAP").is_ok() {
            eprintln!("=== Context Map Debug ===");
            eprintln!("num_dc_tables: {}", num_dc_tables);
            eprintln!("context_map: {:?}", context_map);
            eprintln!("ac_slot_ids: {:?}", ac_slot_ids);
            for (scan_idx, scan) in scans.iter().enumerate() {
                let scan_type = if scan.ss == 0 && scan.se == 0 {
                    "DC"
                } else if scan.ah == 0 {
                    "AC_first"
                } else {
                    "AC_refine"
                };
                let ac_context = context_config.ac_context(scan_idx, 0);
                let table_idx = if scan.ss == 0 {
                    0
                } else {
                    context_map.get(ac_context).copied().unwrap_or(0)
                };
                let ac_table_idx = table_idx.saturating_sub(num_dc_tables);
                let slot = ac_slot_ids.get(ac_table_idx).copied().unwrap_or(0);
                eprintln!(
                    "Scan {}: {} ss={} se={} ah={} al={} comp={} -> ctx={} table={} slot={}",
                    scan_idx,
                    scan_type,
                    scan.ss,
                    scan.se,
                    scan.ah,
                    scan.al,
                    scan.components[0],
                    ac_context,
                    table_idx,
                    slot
                );
            }
            // Dump histogram symbol counts for AC contexts
            for (i, scan) in scans.iter().enumerate() {
                if scan.ss > 0 {
                    let ac_context = context_config.ac_context(i, 0);
                    if let Some(counter) = token_buffer.counter(ac_context) {
                        let mut syms: Vec<u8> = Vec::new();
                        for s in 0u8..=255 {
                            if counter.get_count(s) > 0 {
                                syms.push(s);
                            }
                        }
                        eprintln!("  Scan {} context {} symbols: {:02x?}", i, ac_context, syms);
                    }
                }
            }
            eprintln!("=========================");
        }

        // ========== WRITE JPEG STRUCTURE ==========
        self.write_header(&mut output)?;
        self.write_quant_tables(&mut output, &y_quant, &cb_quant, &cr_quant)?;
        self.write_frame_header(&mut output)?; // Uses SOF2 for progressive

        // Write initial Huffman tables (all DC + up to 4 AC)
        // Like C++ jpegli, additional AC tables are emitted on-demand before scans that need them.
        let mut next_dht_index = self.write_huffman_tables_progressive_initial(
            &mut output,
            &tables,
            num_dc_tables,
            4, // max_initial_ac: emit up to 4 AC tables initially
        )?;

        if self.config.restart_interval > 0 {
            self.write_restart_interval(&mut output)?;
        }

        // ========== PASS 2: REPLAY TOKENS ==========
        // Encode each scan by replaying tokens with optimized tables
        for (scan_idx, scan) in scans.iter().enumerate() {
            // For AC scans (Ss > 0), check if we need to emit a new Huffman table
            // This matches C++ jpegli behavior: emit tables on-demand for progressive AC scans
            if scan.ss > 0 {
                // Get the AC context for this scan
                let ac_context = context_config.ac_context(scan_idx, 0);
                // Get the table index from context_map
                if let Some(&table_idx) = context_map.get(ac_context) {
                    // If this scan needs the "next" table, emit it now
                    if table_idx == next_dht_index && table_idx < tables.len() {
                        // Get the AC table slot ID from ac_slot_ids
                        let cluster_idx = table_idx.saturating_sub(num_dc_tables);
                        let ac_slot = ac_slot_ids
                            .get(cluster_idx)
                            .copied()
                            .unwrap_or(cluster_idx % 4);
                        self.write_single_ac_table(&mut output, &tables[table_idx], ac_slot)?;
                        next_dht_index += 1;
                    }
                }
            }

            // Write SOS header with context-based table selection
            self.write_progressive_scan_header_with_slot_ids(
                &mut output,
                scan_idx,
                scan,
                is_color,
                &context_config,
                &context_map,
                num_dc_tables,
                &ac_slot_ids,
            )?;

            // Replay tokens for this scan
            let scan_data = self.replay_progressive_scan(
                &token_buffer,
                scan_idx,
                scan,
                is_color,
                &context_config,
                &tables,
                num_dc_tables,
                &context_map,
                &ac_slot_ids,
                next_dht_index, // tables_emitted so far
            )?;
            output.extend_from_slice(&scan_data);
        }

        // Write EOI
        output.push(0xFF);
        output.push(MARKER_EOI);

        Ok(output)
    }

    /// Encodes pre-computed blocks as progressive JPEG.
    ///
    /// This is used by the strip-based encoder which computes blocks during
    /// strip processing and then needs to encode them as progressive.
    ///
    /// # Arguments
    /// * `y_blocks` - Y channel quantized DCT blocks (zigzag order)
    /// * `cb_blocks` - Cb channel quantized DCT blocks
    /// * `cr_blocks` - Cr channel quantized DCT blocks
    /// * `y_quant` - Y quantization table
    /// * `cb_quant` - Cb quantization table
    /// * `cr_quant` - Cr quantization table
    pub(crate) fn encode_progressive_from_blocks(
        &self,
        y_blocks: &[[i16; DCT_BLOCK_SIZE]],
        cb_blocks: &[[i16; DCT_BLOCK_SIZE]],
        cr_blocks: &[[i16; DCT_BLOCK_SIZE]],
        y_quant: &QuantTable,
        cb_quant: &QuantTable,
        cr_quant: &QuantTable,
    ) -> Result<Vec<u8>> {
        let width = self.config.width as usize;
        let height = self.config.height as usize;

        let mut output = crate::foundation::alloc::try_with_capacity(
            width * height / 4,
            "progressive from blocks output",
        )?;

        let is_color = self.config.pixel_format != PixelFormat::Gray;
        let num_components = if is_color { 3 } else { 1 };

        // Define progressive scan script
        let scans = self.get_progressive_scan_script(is_color);

        // ========== CREATE CONTEXT CONFIG ==========
        let context_config = ContextConfig::for_progressive(
            num_components,
            scans.iter().map(|s| (s.ss, s.se, s.components.len())),
        );

        // ========== PASS 1: TOKENIZATION ==========
        let mut token_buffer =
            ProgressiveTokenBuffer::new(num_components, context_config.num_contexts);

        for (scan_idx, scan) in scans.iter().enumerate() {
            let context = if scan.ss == 0 && scan.se == 0 {
                context_config.dc_context(scan.components[0] as usize) as u8
            } else {
                context_config.ac_context(scan_idx, 0) as u8
            };

            if scan.ss == 0 && scan.se == 0 {
                // DC scan
                let blocks: Vec<&[[i16; DCT_BLOCK_SIZE]]> = scan
                    .components
                    .iter()
                    .map(|&c| match c {
                        0 => y_blocks,
                        1 => cb_blocks,
                        2 => cr_blocks,
                        _ => &[][..],
                    })
                    .collect();
                let component_indices: Vec<usize> =
                    scan.components.iter().map(|&c| c as usize).collect();
                token_buffer.tokenize_dc_scan(&blocks, &component_indices, scan.al, scan.ah);
            } else if scan.ah == 0 {
                // AC first scan
                let blocks: &[[i16; DCT_BLOCK_SIZE]] = match scan.components[0] {
                    0 => y_blocks,
                    1 => cb_blocks,
                    2 => cr_blocks,
                    _ => {
                        return Err(Error::InternalError {
                            reason: "Invalid component",
                        })
                    }
                };
                token_buffer.tokenize_ac_first_scan(blocks, context, scan.ss, scan.se, scan.al);
            } else {
                // AC refinement scan
                let blocks: &[[i16; DCT_BLOCK_SIZE]] = match scan.components[0] {
                    0 => y_blocks,
                    1 => cb_blocks,
                    2 => cr_blocks,
                    _ => {
                        return Err(Error::InternalError {
                            reason: "Invalid component",
                        })
                    }
                };
                token_buffer.tokenize_ac_refinement_scan(
                    blocks, context, scan.ss, scan.se, scan.ah, scan.al,
                );
            }
        }

        // ========== GENERATE OPTIMIZED TABLES ==========
        let (context_map, num_dc_tables, tables, ac_slot_ids) = token_buffer
            .generate_optimized_tables(
                4,  // max DC clusters
                12, // max AC clusters
                context_config.ac_offset,
                false, // force_baseline
            )?;

        // ========== WRITE JPEG STRUCTURE ==========
        if self.config.use_xyb {
            // XYB mode: use XYB-specific headers
            self.write_header_xyb(&mut output)?;
            // Write APP14 Adobe marker for RGB colorspace (required by decoders)
            self.write_app14_adobe(&mut output, 0)?; // 0 = RGB (no transform)
                                                     // Write XYB ICC profile so decoders can interpret the colors correctly
            self.write_icc_profile(&mut output, &XYB_ICC_PROFILE)?;
            self.write_quant_tables_xyb(&mut output, y_quant, cb_quant, cr_quant)?;
            self.write_frame_header_xyb_progressive(&mut output)?;
        } else {
            // YCbCr mode: use standard headers
            self.write_header(&mut output)?;
            self.write_quant_tables(&mut output, y_quant, cb_quant, cr_quant)?;
            self.write_frame_header(&mut output)?; // Uses SOF2 for progressive
        }

        // Write initial Huffman tables
        let mut next_dht_index = self.write_huffman_tables_progressive_initial(
            &mut output,
            &tables,
            num_dc_tables,
            4, // max_initial_ac
        )?;

        if self.config.restart_interval > 0 {
            self.write_restart_interval(&mut output)?;
        }

        // ========== PASS 2: REPLAY TOKENS ==========
        for (scan_idx, scan) in scans.iter().enumerate() {
            // Emit AC table on-demand if needed
            if scan.ss > 0 {
                let ac_context = context_config.ac_context(scan_idx, 0);
                if let Some(&table_idx) = context_map.get(ac_context) {
                    if table_idx == next_dht_index && table_idx < tables.len() {
                        let cluster_idx = table_idx.saturating_sub(num_dc_tables);
                        let ac_slot = ac_slot_ids
                            .get(cluster_idx)
                            .copied()
                            .unwrap_or(cluster_idx % 4);
                        self.write_single_ac_table(&mut output, &tables[table_idx], ac_slot)?;
                        next_dht_index += 1;
                    }
                }
            }

            // Write SOS header
            self.write_progressive_scan_header_with_slot_ids(
                &mut output,
                scan_idx,
                scan,
                is_color,
                &context_config,
                &context_map,
                num_dc_tables,
                &ac_slot_ids,
            )?;

            // Replay tokens for this scan
            let scan_data = self.replay_progressive_scan(
                &token_buffer,
                scan_idx,
                scan,
                is_color,
                &context_config,
                &tables,
                num_dc_tables,
                &context_map,
                &ac_slot_ids,
                next_dht_index,
            )?;
            output.extend_from_slice(&scan_data);
        }

        // Write EOI
        output.push(0xFF);
        output.push(MARKER_EOI);

        Ok(output)
    }
}
