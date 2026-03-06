//! Progressive JPEG encoding functions.
//!
//! These methods handle multi-scan progressive encoding:
//! - DC and AC scan encoding
//! - Successive approximation (refinement scans)
//! - Scan script generation

use super::ProgressiveScan;
use super::config::ComputedConfig;
use super::encoder_types::ScanStrategy;
use crate::entropy::EntropyEncoder;
use crate::error::{Error, Result};
use crate::foundation::consts::{DCT_BLOCK_SIZE, MARKER_EOI, XYB_ICC_PROFILE};
use crate::huffman::optimize::{ContextConfig, OptimizedTable, ProgressiveTokenBuffer};
use crate::quant::QuantTable;
use crate::types::Subsampling;

/// Resolve the AC Huffman table cluster index and JPEG slot ID for a scan.
///
/// The cluster_idx indexes into AC tables (offset from `num_dc_tables`),
/// and slot_id is the JPEG DHT slot (0-3) with modular cycling.
fn ac_scan_slot(
    scan_idx: usize,
    context_config: &ContextConfig,
    context_map: &[usize],
    num_dc_tables: usize,
    ac_slot_ids: &[usize],
) -> (usize, usize) {
    let ac_context = context_config.ac_context(scan_idx, 0);
    let cluster_idx = if ac_context < context_map.len() {
        context_map[ac_context].saturating_sub(num_dc_tables)
    } else {
        0
    };
    let slot_id = ac_slot_ids
        .get(cluster_idx)
        .copied()
        .unwrap_or(cluster_idx % 4);
    (cluster_idx, slot_id)
}

impl ComputedConfig {
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

        // Set up AC Huffman table for this specific scan.
        //
        // We load only the table needed by the current scan, not all AC tables.
        // JPEG allows only 4 AC table slots (0-3), but the optimizer may produce
        // more than 4 AC clusters. With slot cycling, loading all tables at once
        // would cause later tables to overwrite earlier ones in the same slot,
        // leaving the wrong table active for earlier scans.
        if scan.ss > 0 {
            let (cluster_idx, slot) = ac_scan_slot(
                scan_idx,
                context_config,
                context_map,
                num_dc_tables,
                ac_slot_ids,
            );
            if let Some(table) = tables.get(num_dc_tables + cluster_idx) {
                encoder.set_ac_table(slot, &table.table);
            }
        }

        if self.restart_interval > 0 {
            encoder.set_restart_interval(self.restart_interval);
        }

        // Get scan info (use ok_or_else for lazy error creation)
        let scan_info = token_buffer
            .scan_info
            .get(scan_idx)
            .ok_or_else(|| Error::internal("Scan info not found"))?;

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
            encoder.write_dc_tokens(tokens, &dc_context_map, &scan_info.restarts)?;
        } else if scan.ah == 0 {
            // AC first scan: replay AC tokens
            let (_cluster_idx, slot_id) = ac_scan_slot(
                scan_idx,
                context_config,
                context_map,
                num_dc_tables,
                ac_slot_ids,
            );
            let tokens = token_buffer.scan_tokens(scan_idx);
            encoder.write_ac_first_tokens(tokens, slot_id, &scan_info.restarts)?;
        } else {
            // AC refinement scan: replay refinement tokens
            let (_cluster_idx, slot_id) = ac_scan_slot(
                scan_idx,
                context_config,
                context_map,
                num_dc_tables,
                ac_slot_ids,
            );
            if cfg!(debug_assertions) && std::env::var("DUMP_RUST_AC_REFINEMENT").is_ok() {
                scan_info.debug_dump(scan_idx);
            }
            encoder.write_ac_refinement_tokens(scan_info, slot_id, &scan_info.restarts)?;
        }

        Ok(encoder.finish())
    }

    /// Returns the progressive scan script for level 2.
    pub(crate) fn get_progressive_scan_script(&self, is_color: bool) -> Vec<ProgressiveScan> {
        match self.scan_strategy {
            ScanStrategy::Default => self.get_jpegli_scan_script(is_color),
            ScanStrategy::Search => self.get_jpegli_scan_script(is_color), // Search uses trial encoding
            ScanStrategy::Mozjpeg => self.get_mozjpeg_scan_script(is_color),
        }
    }

    /// jpegli-style progressive scan script.
    ///
    /// - Frequency split at AC 2/3
    /// - Successive approximation for all components (Al=2 → 1 → 0)
    fn get_jpegli_scan_script(&self, is_color: bool) -> Vec<ProgressiveScan> {
        let num_components = if is_color { 3 } else { 1 };
        let mut scans = Vec::new();

        // For XYB mode, always use non-interleaved DC scans (matches C++ jpegli)
        // For 4:4:4 YCbCr subsampling, DC can be interleaved
        let dc_interleaved = !self.use_xyb && matches!(self.subsampling, Subsampling::S444);

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

        scans
    }

    /// mozjpeg-style progressive scan script.
    ///
    /// - Frequency split at AC 8/9
    /// - No successive approximation for chroma (full precision in one pass)
    /// - Successive approximation for luma only
    fn get_mozjpeg_scan_script(&self, is_color: bool) -> Vec<ProgressiveScan> {
        let num_components = if is_color { 3 } else { 1 };
        let mut scans = Vec::new();

        // DC scans (separate per component)
        for c in 0..num_components {
            scans.push(ProgressiveScan {
                components: vec![c],
                ss: 0,
                se: 0,
                ah: 0,
                al: 0,
            });
        }

        // Luma AC with successive approximation
        // AC 1-8 at Al=0 (full precision for low frequencies)
        scans.push(ProgressiveScan {
            components: vec![0],
            ss: 1,
            se: 8,
            ah: 0,
            al: 0,
        });

        // AC 9-63 first pass at Al=1
        scans.push(ProgressiveScan {
            components: vec![0],
            ss: 9,
            se: 63,
            ah: 0,
            al: 1,
        });

        // AC 9-63 refinement at Al=0
        scans.push(ProgressiveScan {
            components: vec![0],
            ss: 9,
            se: 63,
            ah: 1,
            al: 0,
        });

        // Chroma AC - no successive approximation (full precision)
        if is_color {
            // Cb: AC 1-63 full precision
            scans.push(ProgressiveScan {
                components: vec![1],
                ss: 1,
                se: 63,
                ah: 0,
                al: 0,
            });

            // Cr: AC 1-63 full precision
            scans.push(ProgressiveScan {
                components: vec![2],
                ss: 1,
                se: 63,
                ah: 0,
                al: 0,
            });
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
        let mut output = Vec::new();
        self.encode_progressive_from_blocks_into(
            y_blocks,
            cb_blocks,
            cr_blocks,
            y_quant,
            cb_quant,
            cr_quant,
            &mut output,
        )?;
        Ok(output)
    }

    /// Encodes progressive JPEG from pre-processed blocks into provided buffer.
    ///
    /// Same as `encode_progressive_from_blocks` but writes directly to provided buffer.
    pub(crate) fn encode_progressive_from_blocks_into(
        &self,
        y_blocks: &[[i16; DCT_BLOCK_SIZE]],
        cb_blocks: &[[i16; DCT_BLOCK_SIZE]],
        cr_blocks: &[[i16; DCT_BLOCK_SIZE]],
        y_quant: &QuantTable,
        cb_quant: &QuantTable,
        cr_quant: &QuantTable,
        output: &mut Vec<u8>,
    ) -> Result<()> {
        let is_color = !self.pixel_format.is_grayscale();
        let num_components = if is_color { 3 } else { 1 };

        if self.scan_strategy == ScanStrategy::Search && !self.use_xyb {
            // Generate multiple candidate scan scripts and trial-encode each.
            // The frequency estimator can't accurately compare scripts with
            // different numbers of scans (Huffman clustering effects), so we
            // use actual encoding for the final selection.
            let candidates = super::scan_optimize::generate_candidate_scripts(
                y_blocks,
                cb_blocks,
                cr_blocks,
                num_components as u8,
            )?;

            let mut best_output = Vec::new();
            for (i, candidate) in candidates.iter().enumerate() {
                let mut trial_output = Vec::new();
                self.encode_progressive_with_scans(
                    candidate,
                    y_blocks,
                    cb_blocks,
                    cr_blocks,
                    y_quant,
                    cb_quant,
                    cr_quant,
                    &mut trial_output,
                    is_color,
                )?;

                if i == 0 || trial_output.len() < best_output.len() {
                    best_output = trial_output;
                }
            }

            *output = best_output;
            return Ok(());
        }

        let scans = self.get_progressive_scan_script(is_color);
        self.encode_progressive_with_scans(
            &scans, y_blocks, cb_blocks, cr_blocks, y_quant, cb_quant, cr_quant, output, is_color,
        )
    }

    /// Core progressive encode pipeline: tokenize, optimize Huffman, write JPEG.
    ///
    /// Takes an explicit scan script and produces a complete progressive JPEG.
    #[allow(clippy::too_many_arguments)]
    fn encode_progressive_with_scans(
        &self,
        scans: &[ProgressiveScan],
        y_blocks: &[[i16; DCT_BLOCK_SIZE]],
        cb_blocks: &[[i16; DCT_BLOCK_SIZE]],
        cr_blocks: &[[i16; DCT_BLOCK_SIZE]],
        y_quant: &QuantTable,
        cb_quant: &QuantTable,
        cr_quant: &QuantTable,
        output: &mut Vec<u8>,
        is_color: bool,
    ) -> Result<()> {
        let width = self.width as usize;
        let height = self.height as usize;
        let num_components = if is_color { 3 } else { 1 };

        output.clear();
        output.try_reserve(width * height / 4).map_err(|_| {
            Error::allocation_failed(width * height / 4, "progressive from blocks output")
        })?;

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
                token_buffer.tokenize_dc_scan(
                    &blocks,
                    &component_indices,
                    scan.al,
                    scan.ah,
                    self.restart_interval,
                );
            } else if scan.ah == 0 {
                // AC first scan
                let blocks: &[[i16; DCT_BLOCK_SIZE]] = match scan.components[0] {
                    0 => y_blocks,
                    1 => cb_blocks,
                    2 => cr_blocks,
                    _ => return Err(Error::internal("Invalid component")),
                };
                token_buffer.tokenize_ac_first_scan(
                    blocks,
                    context,
                    scan.ss,
                    scan.se,
                    scan.al,
                    self.restart_interval,
                );
            } else {
                // AC refinement scan
                let blocks: &[[i16; DCT_BLOCK_SIZE]] = match scan.components[0] {
                    0 => y_blocks,
                    1 => cb_blocks,
                    2 => cr_blocks,
                    _ => return Err(Error::internal("Invalid component")),
                };
                token_buffer.tokenize_ac_refinement_scan(
                    blocks,
                    context,
                    scan.ss,
                    scan.se,
                    scan.ah,
                    scan.al,
                    self.restart_interval,
                )?;
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
        if self.use_xyb {
            // XYB mode: use XYB-specific headers
            self.write_header_xyb(output)?;
            // Write APP14 Adobe marker for RGB colorspace (required by decoders)
            self.write_app14_adobe(output, 0)?; // 0 = RGB (no transform)
            // Write XYB ICC profile so decoders can interpret the colors correctly
            self.write_icc_profile(output, &XYB_ICC_PROFILE)?;
            self.write_quant_tables_xyb(output, y_quant, cb_quant, cr_quant)?;
            self.write_frame_header_xyb_progressive(output)?;
        } else {
            // YCbCr mode: use standard headers
            self.write_header(output)?;
            self.write_quant_tables(output, y_quant, cb_quant, cr_quant)?;
            self.write_frame_header(output)?; // Uses SOF2 for progressive
        }

        // Write initial DHT tables: DC tables only. AC tables are emitted
        // on-demand per scan to handle slot cycling correctly.
        let _next_dht_index = self.write_huffman_tables_progressive_initial(
            output,
            &tables,
            num_dc_tables,
            0, // AC tables emitted per-scan
        )?;

        if self.restart_interval > 0 {
            self.write_restart_interval(output)?;
        }

        // Track which AC cluster is currently loaded in each JPEG slot (0-3).
        let mut slot_cluster: [Option<usize>; 4] = [None; 4];

        // ========== PASS 2: REPLAY TOKENS ==========
        for (scan_idx, scan) in scans.iter().enumerate() {
            // Emit AC table on-demand before each AC scan
            if scan.ss > 0 {
                let (cluster_idx, ac_slot) = ac_scan_slot(
                    scan_idx,
                    &context_config,
                    &context_map,
                    num_dc_tables,
                    &ac_slot_ids,
                );
                // Only emit if this slot doesn't already have the right table
                if slot_cluster[ac_slot] != Some(cluster_idx) {
                    if let Some(table) = tables.get(num_dc_tables + cluster_idx) {
                        self.write_single_ac_table(output, table, ac_slot)?;
                    }
                    slot_cluster[ac_slot] = Some(cluster_idx);
                }
            }

            // Write SOS header
            self.write_progressive_scan_header_with_slot_ids(
                output,
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
            )?;
            output.extend_from_slice(&scan_data);
        }

        // Write EOI
        output.push(0xFF);
        output.push(MARKER_EOI);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    #[cfg(feature = "decoder")]
    fn test_restart_marker_overhead_progressive() {
        use crate::encode::encoder_config::EncoderConfig;
        use crate::encode::encoder_types::ChromaSubsampling;

        let width = 512u32;
        let height = 512u32;

        // Generate a gradient + noise pattern (not a pure gradient, which
        // produces degenerate DCT coefficients).
        let mut pixels = Vec::with_capacity((width * height) as usize);
        for y in 0..height {
            for x in 0..width {
                // Hash-based noise layered on a diagonal gradient
                let grad = ((x + y) * 255 / (width + height)) as u8;
                let noise = ((x.wrapping_mul(31337) ^ y.wrapping_mul(7919))
                    .wrapping_mul(2654435761)
                    >> 24) as u8;
                let r = grad.wrapping_add(noise & 0x1F);
                let g = grad.wrapping_add((noise >> 2) & 0x1F).wrapping_add(30);
                let b = grad.wrapping_add((noise >> 4) & 0x1F).wrapping_add(80);
                pixels.push(rgb::RGB { r, g, b });
            }
        }

        let restart_rows_values: &[(u16, &str)] = &[
            (0, "disabled"),
            (1, "1 row"),
            (4, "4 rows (default)"),
            (8, "8 rows"),
        ];

        let mut sizes: Vec<(u16, &str, usize)> = Vec::new();

        for &(rows, label) in restart_rows_values {
            let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
                .progressive(true)
                .restart_mcu_rows(rows);

            let jpeg = config.encode(&pixels, width, height).unwrap();
            let file_size = jpeg.len();
            sizes.push((rows, label, file_size));

            // Verify the output decodes successfully
            #[allow(deprecated)]
            let decoded = crate::decode::Decoder::new()
                .decode(&jpeg, enough::Unstoppable)
                .unwrap_or_else(|_| panic!("decode failed for restart_mcu_rows={}", rows));
            assert_eq!(decoded.width, width);
            assert_eq!(decoded.height, height);
            assert_eq!(
                decoded.pixels_u8().unwrap().len(),
                (width * height * 3) as usize,
                "wrong pixel count for restart_mcu_rows={}",
                rows,
            );
        }

        // Print results (visible with --nocapture)
        let baseline_size = sizes[0].2; // restart disabled
        eprintln!();
        eprintln!("Progressive JPEG restart marker overhead (512x512 Q85 4:2:0):");
        eprintln!("{:<22} {:>8} {:>10}", "Setting", "Bytes", "Overhead");
        eprintln!("{}", "-".repeat(42));
        for &(rows, label, size) in &sizes {
            let overhead_pct = if rows == 0 {
                0.0
            } else {
                (size as f64 - baseline_size as f64) / baseline_size as f64 * 100.0
            };
            eprintln!("{:<22} {:>8} {:>+9.3}%", label, size, overhead_pct);
        }
        eprintln!();

        // Assert that the default 4-row restart overhead is less than 2%.
        // Real photographic images see ~0.04% overhead; this synthetic 512x512
        // test image is worst-case due to its small size (fewer MCUs means each
        // restart marker is a larger fraction of the total bitstream).
        let default_size = sizes.iter().find(|s| s.0 == 4).unwrap().2;
        let default_overhead =
            (default_size as f64 - baseline_size as f64) / baseline_size as f64 * 100.0;
        assert!(
            default_overhead < 2.0,
            "4-row restart overhead {:.3}% exceeds 2% limit (baseline={}, 4-row={})",
            default_overhead,
            baseline_size,
            default_size,
        );
    }
}
