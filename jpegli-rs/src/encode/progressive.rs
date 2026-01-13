//! Progressive JPEG encoding functions.
//!
//! These methods handle multi-scan progressive encoding:
//! - DC and AC scan encoding
//! - Successive approximation (refinement scans)
//! - Scan script generation

#![allow(deprecated)] // This module implements methods for the deprecated Encoder struct

use super::{Encoder, ProgressiveScan};
use crate::consts::{DCT_BLOCK_SIZE, MARKER_EOI, XYB_ICC_PROFILE};
use crate::entropy::EntropyEncoder;
use crate::error::{Error, Result};
use crate::huffman::optimize::{ContextConfig, OptimizedTable, ProgressiveTokenBuffer};
use crate::quant::QuantTable;
use crate::types::Subsampling;

impl Encoder {
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

        let is_color = !self.config.pixel_format.is_grayscale();
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
