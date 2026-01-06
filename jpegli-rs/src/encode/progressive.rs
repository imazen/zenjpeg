//! Progressive JPEG encoding functions.
//!
//! These methods handle multi-scan progressive encoding:
//! - DC and AC scan encoding
//! - Successive approximation (refinement scans)
//! - Scan script generation

use super::*;

impl Encoder {
    /// Builds OptimizedHuffmanTables from the clustered tables.
    /// Currently unused - kept for potential debugging or future use.
    #[allow(dead_code)]
    pub(super) fn build_progressive_huffman_tables(
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
    pub(super) fn replay_progressive_scan(
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
        let mut encoder = EntropyEncoder::new();
        let num_components = if is_color { 3 } else { 1 };

        // Set up DC Huffman tables (up to 4)
        for (i, table) in tables.iter().take(num_dc_tables).enumerate() {
            encoder.set_dc_table(i, table.table.clone());
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
            encoder.set_ac_table(slot, table.table.clone());
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
    pub(super) fn get_progressive_scan_script(&self, is_color: bool) -> Vec<ProgressiveScan> {
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
    pub(super) fn encode_progressive_scan(
        &self,
        y_blocks: &[[i16; DCT_BLOCK_SIZE]],
        cb_blocks: &[[i16; DCT_BLOCK_SIZE]],
        cr_blocks: &[[i16; DCT_BLOCK_SIZE]],
        scan: &ProgressiveScan,
        is_color: bool,
        tables: &Option<OptimizedHuffmanTables>,
    ) -> Result<Vec<u8>> {
        let mut encoder = EntropyEncoder::new();

        // Set up Huffman tables
        if let Some(ref opt_tables) = tables {
            encoder.set_dc_table(0, opt_tables.dc_luma.table.clone());
            encoder.set_ac_table(0, opt_tables.ac_luma.table.clone());
            if is_color {
                encoder.set_dc_table(1, opt_tables.dc_chroma.table.clone());
                encoder.set_ac_table(1, opt_tables.ac_chroma.table.clone());
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
}
