//! Lossless JPEG restructuring.
//!
//! Change the scan structure, add restart markers, or convert between
//! sequential and progressive encoding — without touching coefficient values.

use alloc::vec;
use alloc::vec::Vec;

use crate::decode::{ComponentCoefficients, DecodeConfig, PreserveConfig};
use crate::encode::config::ProgressiveScan;
use crate::entropy::encoder::EntropyEncoder;
use crate::error::{Error, Result};
use crate::foundation::consts::{
    DCT_BLOCK_SIZE, MARKER_DHT, MARKER_EOI, MARKER_SOF2, MARKER_SOI, MARKER_SOS,
};
use crate::huffman::optimize::{ContextConfig, ProgressiveTokenBuffer};
use enough::Stop;

use super::coeff_transform::{TransformConfig, TransformedCoefficients, transform_coefficients};
use super::pipeline::{
    component_to_blocks, encode_from_coefficients, write_marker_segment, write_quant_tables,
};

/// Output scan structure for restructured JPEG.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputMode {
    /// Baseline sequential (SOF0) — single scan, all components interleaved.
    Sequential,
    /// Progressive (SOF2) — multiple scans with frequency splitting and
    /// successive approximation, using jpegli-style scan script.
    Progressive,
}

/// Restart marker interval for restructured output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestartInterval {
    /// No restart markers.
    None,
    /// Insert restart markers every N MCUs (raw DRI value).
    EveryMcus(u16),
    /// Insert restart markers every N MCU rows.
    /// Computed from image dimensions at encoding time.
    EveryMcuRows(u16),
}

/// Configuration for lossless JPEG restructuring.
#[derive(Debug, Clone)]
pub struct RestructureConfig {
    /// Output scan structure.
    pub output_mode: OutputMode,
    /// Restart marker interval.
    pub restart_interval: RestartInterval,
    /// Optional spatial transform to apply simultaneously.
    pub transform: Option<TransformConfig>,
}

impl Default for RestructureConfig {
    fn default() -> Self {
        Self {
            output_mode: OutputMode::Sequential,
            restart_interval: RestartInterval::None,
            transform: None,
        }
    }
}

/// Losslessly restructure a JPEG.
///
/// Changes the scan structure (sequential ↔ progressive) and/or adds restart
/// markers without touching coefficient values. Optionally applies a spatial
/// transform at the same time.
///
/// # Examples
///
/// ```rust,ignore
/// use zenjpeg::lossless::{restructure, RestructureConfig, OutputMode, RestartInterval};
///
/// // Convert to progressive with restart markers every 10 MCU rows
/// let config = RestructureConfig {
///     output_mode: OutputMode::Progressive,
///     restart_interval: RestartInterval::EveryMcuRows(1),
///     ..Default::default()
/// };
/// let result = restructure(&jpeg_data, &config, enough::Unstoppable)?;
/// ```
pub fn restructure(
    jpeg_data: &[u8],
    config: &RestructureConfig,
    stop: impl Stop,
) -> Result<Vec<u8>> {
    stop.check()?;

    // Step 1: Decode to coefficients + metadata
    let decoder = DecodeConfig::new().preserve(PreserveConfig::all());
    let (decoded_coeffs, extras) = decoder.decode_coefficients_with_extras(jpeg_data, &stop)?;

    stop.check()?;

    // Step 2: Optionally transform coefficients
    let coeffs = if let Some(ref transform_config) = config.transform {
        transform_coefficients(&decoded_coeffs, transform_config)
            .map_err(|e| Error::io_error(alloc::format!("{e}")))?
    } else {
        TransformedCoefficients {
            width: decoded_coeffs.width,
            height: decoded_coeffs.height,
            components: decoded_coeffs.components,
            quant_tables: decoded_coeffs.quant_tables,
        }
    };

    stop.check()?;

    // Step 3: Compute restart interval in MCUs
    let restart_mcus = compute_restart_interval(&coeffs.components, config.restart_interval);

    // Step 4: Encode with the requested structure
    let preserved = extras.as_ref().map(|e| e.segments());

    match config.output_mode {
        OutputMode::Sequential => encode_from_coefficients(&coeffs, preserved, restart_mcus, &stop),
        OutputMode::Progressive => {
            encode_progressive_from_coefficients(&coeffs, preserved, restart_mcus, &stop)
        }
    }
}

/// Compute the restart interval in MCUs from the configuration.
fn compute_restart_interval(
    components: &[ComponentCoefficients],
    interval: RestartInterval,
) -> u16 {
    match interval {
        RestartInterval::None => 0,
        RestartInterval::EveryMcus(n) => n,
        RestartInterval::EveryMcuRows(rows) => {
            if components.is_empty() || rows == 0 {
                return 0;
            }
            // MCU width = number of MCU columns
            // For 4:4:4: MCU = 1 block per component, so mcus_wide = blocks_wide
            // For 4:2:0: MCU = 2x2 luma blocks, so mcus_wide = blocks_wide / h_samp
            let max_h_samp = components.iter().map(|c| c.h_samp).max().unwrap_or(1) as usize;
            let max_v_samp = components.iter().map(|c| c.v_samp).max().unwrap_or(1) as usize;
            let _ = max_v_samp; // MCU rows is about horizontal MCU count

            // Luma blocks_wide / max_h_samp = MCU columns
            let luma = &components[0];
            let mcus_wide = (luma.blocks_wide + max_h_samp - 1) / max_h_samp;

            let mcus_per_interval = mcus_wide * rows as usize;
            mcus_per_interval.min(u16::MAX as usize) as u16
        }
    }
}

/// Generate jpegli-style progressive scan script.
///
/// Standalone version that doesn't require `ComputedConfig`.
/// Uses the same scan structure as `get_jpegli_scan_script`:
/// - DC first (interleaved for 4:4:4, non-interleaved for subsampled)
/// - AC 1-2 full precision per component
/// - AC 3-63 with successive approximation (Al=2 → 1 → 0)
fn jpegli_scan_script(num_components: usize, is_subsampled: bool) -> Vec<ProgressiveScan> {
    let mut scans = Vec::new();

    // DC scans
    if !is_subsampled && num_components >= 3 {
        // Interleaved DC for 4:4:4
        scans.push(ProgressiveScan {
            components: (0..num_components as u8).collect(),
            ss: 0,
            se: 0,
            ah: 0,
            al: 0,
        });
    } else {
        // Non-interleaved DC
        for c in 0..num_components as u8 {
            scans.push(ProgressiveScan {
                components: vec![c],
                ss: 0,
                se: 0,
                ah: 0,
                al: 0,
            });
        }
    }

    // AC 1-2: full precision (low frequency) — all components
    for c in 0..num_components as u8 {
        scans.push(ProgressiveScan {
            components: vec![c],
            ss: 1,
            se: 2,
            ah: 0,
            al: 0,
        });
    }

    // AC 3-63 first pass: top bits (Al=2) — all components
    for c in 0..num_components as u8 {
        scans.push(ProgressiveScan {
            components: vec![c],
            ss: 3,
            se: 63,
            ah: 0,
            al: 2,
        });
    }

    // AC 3-63 refinement: bit 1 (Ah=2, Al=1) — all components
    for c in 0..num_components as u8 {
        scans.push(ProgressiveScan {
            components: vec![c],
            ss: 3,
            se: 63,
            ah: 2,
            al: 1,
        });
    }

    // AC 3-63 refinement: bit 0 (Ah=1, Al=0) — all components
    for c in 0..num_components as u8 {
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

/// Encode coefficients as progressive JPEG.
///
/// Two-pass: tokenize all scans → optimize Huffman tables → replay with
/// optimized tables.
fn encode_progressive_from_coefficients(
    coeffs: &TransformedCoefficients,
    preserved_segments: Option<&[crate::decode::PreservedSegment]>,
    _restart_interval: u16,
    stop: &impl Stop,
) -> Result<Vec<u8>> {
    let num_components = coeffs.components.len();
    let is_color = num_components >= 3;

    // Determine if subsampled
    let is_subsampled = if is_color {
        coeffs.components[0].h_samp > 1 || coeffs.components[0].v_samp > 1
    } else {
        false
    };

    // Convert to block arrays
    let all_blocks: Vec<Vec<[i16; DCT_BLOCK_SIZE]>> =
        coeffs.components.iter().map(component_to_blocks).collect();

    stop.check()?;

    // Generate scan script
    let scans = jpegli_scan_script(num_components, is_subsampled);

    // Create context config
    let context_config = ContextConfig::for_progressive(
        num_components,
        scans.iter().map(|s| (s.ss, s.se, s.components.len())),
    );

    // ========== PASS 1: TOKENIZATION ==========
    let mut token_buffer = ProgressiveTokenBuffer::new(num_components, context_config.num_contexts);

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
                .map(|&c| all_blocks[c as usize].as_slice())
                .collect();
            let component_indices: Vec<usize> =
                scan.components.iter().map(|&c| c as usize).collect();
            token_buffer.tokenize_dc_scan(&blocks, &component_indices, scan.al, scan.ah, 0);
        } else if scan.ah == 0 {
            // AC first scan
            let blocks = &all_blocks[scan.components[0] as usize];
            token_buffer.tokenize_ac_first_scan(blocks, context, scan.ss, scan.se, scan.al, 0);
        } else {
            // AC refinement scan
            let blocks = &all_blocks[scan.components[0] as usize];
            token_buffer.tokenize_ac_refinement_scan(
                blocks, context, scan.ss, scan.se, scan.ah, scan.al, 0,
            )?;
        }
    }

    stop.check()?;

    // ========== GENERATE OPTIMIZED TABLES ==========
    let (context_map, num_dc_tables, tables, ac_slot_ids) = token_buffer
        .generate_optimized_tables(
            4,  // max DC clusters
            12, // max AC clusters
            context_config.ac_offset,
            false, // force_baseline
        )?;

    stop.check()?;

    // ========== WRITE JPEG STRUCTURE ==========
    let mut output =
        Vec::with_capacity(all_blocks.iter().map(|b| b.len()).sum::<usize>() * 32 + 2048);

    // SOI
    output.push(0xFF);
    output.push(MARKER_SOI);

    // Preserved metadata
    if let Some(segments) = preserved_segments {
        for seg in segments {
            write_marker_segment(&mut output, seg.marker, &seg.data);
        }
    }

    // DQT
    write_quant_tables(&mut output, &coeffs.quant_tables, num_components);

    // SOF2 (progressive)
    write_sof2(&mut output, coeffs.width, coeffs.height, &coeffs.components);

    // DHT — DC tables initially
    write_progressive_dc_tables(&mut output, &tables, num_dc_tables);

    // Note: restart markers in progressive scans are not yet supported
    // because the token replay infrastructure doesn't handle them.
    // DRI is intentionally omitted.

    // Track which AC cluster is in each slot
    let mut slot_cluster: [Option<usize>; 4] = [None; 4];

    // ========== PASS 2: REPLAY TOKENS ==========
    for (scan_idx, scan) in scans.iter().enumerate() {
        // Emit AC table on-demand before each AC scan
        if scan.ss > 0 {
            let ac_context = context_config.ac_context(scan_idx, 0);
            let cluster_idx = context_map
                .get(ac_context)
                .map(|&t| t.saturating_sub(num_dc_tables))
                .unwrap_or(0);
            let ac_slot = ac_slot_ids
                .get(cluster_idx)
                .copied()
                .unwrap_or(cluster_idx % 4);

            if slot_cluster[ac_slot] != Some(cluster_idx) {
                if let Some(table) = tables.get(num_dc_tables + cluster_idx) {
                    write_single_ac_dht(&mut output, table, ac_slot);
                }
                slot_cluster[ac_slot] = Some(cluster_idx);
            }
        }

        // SOS header
        write_progressive_sos(
            &mut output,
            scan,
            scan_idx,
            &coeffs.components,
            &context_config,
            &context_map,
            num_dc_tables,
            &ac_slot_ids,
        );

        // Replay tokens
        let scan_data = replay_scan(
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

    // EOI
    output.push(0xFF);
    output.push(MARKER_EOI);

    Ok(output)
}

/// Write SOF2 (progressive) frame header using actual component IDs.
fn write_sof2(output: &mut Vec<u8>, width: u32, height: u32, components: &[ComponentCoefficients]) {
    let num_components = components.len();
    let len = 2 + 1 + 2 + 2 + 1 + num_components * 3;

    output.push(0xFF);
    output.push(MARKER_SOF2);
    output.push((len >> 8) as u8);
    output.push((len & 0xFF) as u8);
    output.push(8); // Sample precision
    output.push((height >> 8) as u8);
    output.push((height & 0xFF) as u8);
    output.push((width >> 8) as u8);
    output.push((width & 0xFF) as u8);
    output.push(num_components as u8);

    for comp in components {
        output.push(comp.id);
        output.push((comp.h_samp << 4) | comp.v_samp);
        output.push(comp.quant_table_idx);
    }
}

/// Write DC Huffman tables as a single DHT marker.
fn write_progressive_dc_tables(
    output: &mut Vec<u8>,
    tables: &[crate::huffman::optimize::OptimizedTable],
    num_dc_tables: usize,
) {
    if num_dc_tables == 0 {
        return;
    }

    output.push(0xFF);
    output.push(MARKER_DHT);

    let mut total_len = 2;
    for table in tables.iter().take(num_dc_tables) {
        total_len += 1 + 16 + table.values.len();
    }
    output.push((total_len >> 8) as u8);
    output.push(total_len as u8);

    for (i, table) in tables.iter().take(num_dc_tables).enumerate() {
        output.push(i as u8); // class 0 (DC), id = i
        output.extend_from_slice(&table.bits);
        output.extend_from_slice(&table.values);
    }
}

/// Write a single AC DHT marker for on-demand table emission.
fn write_single_ac_dht(
    output: &mut Vec<u8>,
    table: &crate::huffman::optimize::OptimizedTable,
    slot_id: usize,
) {
    output.push(0xFF);
    output.push(MARKER_DHT);

    let total_len = 2 + 1 + 16 + table.values.len();
    output.push((total_len >> 8) as u8);
    output.push(total_len as u8);

    output.push(0x10 | (slot_id as u8)); // class 1 (AC), id = slot_id
    output.extend_from_slice(&table.bits);
    output.extend_from_slice(&table.values);
}

/// Write SOS header for a progressive scan.
#[allow(clippy::too_many_arguments)]
fn write_progressive_sos(
    output: &mut Vec<u8>,
    scan: &ProgressiveScan,
    scan_idx: usize,
    components: &[ComponentCoefficients],
    context_config: &ContextConfig,
    context_map: &[usize],
    num_dc_tables: usize,
    ac_slot_ids: &[usize],
) {
    let num_scan_components = scan.components.len() as u8;
    let length = 6u16 + num_scan_components as u16 * 2;

    output.push(0xFF);
    output.push(MARKER_SOS);
    output.push((length >> 8) as u8);
    output.push(length as u8);
    output.push(num_scan_components);

    for (comp_in_scan, &comp_idx) in scan.components.iter().enumerate() {
        // Use actual component ID from source
        let comp_id = components
            .get(comp_idx as usize)
            .map(|c| c.id)
            .unwrap_or(comp_idx + 1);
        output.push(comp_id);

        // DC table selector
        let dc_context = context_config.dc_context(comp_idx as usize);
        let dc_table = context_map.get(dc_context).copied().unwrap_or(0);

        // AC table selector
        let ac_context = context_config.ac_context(scan_idx, comp_in_scan);
        let cluster_idx = context_map
            .get(ac_context)
            .map(|&t| t.saturating_sub(num_dc_tables))
            .unwrap_or(0);
        let ac_table = ac_slot_ids
            .get(cluster_idx)
            .copied()
            .unwrap_or(cluster_idx % 4);

        output.push(((dc_table as u8) << 4) | (ac_table as u8));
    }

    output.push(scan.ss);
    output.push(scan.se);
    output.push((scan.ah << 4) | scan.al);
}

/// Replay tokens for a single progressive scan.
#[allow(clippy::too_many_arguments)]
fn replay_scan(
    token_buffer: &ProgressiveTokenBuffer,
    scan_idx: usize,
    scan: &ProgressiveScan,
    is_color: bool,
    context_config: &ContextConfig,
    tables: &[crate::huffman::optimize::OptimizedTable],
    num_dc_tables: usize,
    context_map: &[usize],
    ac_slot_ids: &[usize],
) -> Result<Vec<u8>> {
    let scan_info = token_buffer.scan_info.get(scan_idx);
    let estimated_tokens = scan_info
        .map(|s| s.num_tokens + s.ref_tokens.len())
        .unwrap_or(1024);
    let mut encoder = EntropyEncoder::with_capacity(estimated_tokens * 2);
    let num_components = if is_color { 3 } else { 1 };

    // Set up DC tables
    for (i, table) in tables.iter().take(num_dc_tables).enumerate() {
        encoder.set_dc_table(i, &table.table);
    }

    // Set up AC table for this scan
    if scan.ss > 0 {
        let ac_context = context_config.ac_context(scan_idx, 0);
        let cluster_idx = context_map
            .get(ac_context)
            .map(|&t| t.saturating_sub(num_dc_tables))
            .unwrap_or(0);
        let slot = ac_slot_ids
            .get(cluster_idx)
            .copied()
            .unwrap_or(cluster_idx % 4);
        if let Some(table) = tables.get(num_dc_tables + cluster_idx) {
            encoder.set_ac_table(slot, &table.table);
        }
    }

    let scan_info = token_buffer
        .scan_info
        .get(scan_idx)
        .ok_or_else(|| Error::internal("Scan info not found"))?;

    if scan.ss == 0 && scan.se == 0 {
        // DC scan
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
        encoder.write_dc_tokens(tokens, &dc_context_map, &[])?;
    } else if scan.ah == 0 {
        // AC first scan
        let ac_context = context_config.ac_context(scan_idx, 0);
        let cluster_idx = context_map
            .get(ac_context)
            .map(|&t| t.saturating_sub(num_dc_tables))
            .unwrap_or(0);
        let slot_id = ac_slot_ids
            .get(cluster_idx)
            .copied()
            .unwrap_or(cluster_idx % 4);
        let tokens = token_buffer.scan_tokens(scan_idx);
        encoder.write_ac_first_tokens(tokens, slot_id, &[])?;
    } else {
        // AC refinement scan
        let ac_context = context_config.ac_context(scan_idx, 0);
        let cluster_idx = context_map
            .get(ac_context)
            .map(|&t| t.saturating_sub(num_dc_tables))
            .unwrap_or(0);
        let slot_id = ac_slot_ids
            .get(cluster_idx)
            .copied()
            .unwrap_or(cluster_idx % 4);
        encoder.write_ac_refinement_tokens(scan_info, slot_id, &[])?;
    }

    Ok(encoder.finish())
}
