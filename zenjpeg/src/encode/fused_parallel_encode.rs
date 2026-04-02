//! Fused parallel encoder: color convert → AQ → DCT → quantize → entropy in parallel.
//!
//! Two modes:
//! - **Fixed tables**: single pass — each segment encodes directly to bytes.
//! - **Optimized tables**: two passes — pass 1 quantizes + collects frequencies in parallel,
//!   merge frequencies → build tables, pass 2 entropy-encodes in parallel.
//!
//! Each rayon task processes a horizontal band of MCU rows. AQ at segment boundaries
//! uses edge clamping (same as image edges) — imperceptible quality difference.

use rayon::prelude::*;

use crate::encode::blocks::HuffmanSymbolFrequencies;
use crate::encode::dct::simd::forward_dct_8x8_wide;
use crate::encode::deringing::preprocess_deringing_block;
use crate::encode::layout::LayoutParams;
use crate::entropy::encoder::EntropyEncoder;
use crate::error::Result;
use crate::foundation::consts::DCT_BLOCK_SIZE;
use crate::foundation::simd_types::Block8x8f;
use crate::huffman::optimize::HuffmanTableSet;
use crate::huffman::HuffmanEncodeTable;
use crate::quant::aq::streaming::StreamingAQ;
use crate::types::Subsampling;

use super::parallel::ParallelEntropyConfig;

/// Minimum MCU rows per segment to justify parallel overhead.
const MIN_MCU_ROWS_PER_SEGMENT: usize = 2;

/// Shared read-only config for all parallel segments.
struct SharedEncodeConfig {
    width: usize,
    height: usize,
    padded_width: usize,
    blocks_w: usize,
    mcu_cols: usize,
    mcu_height: usize,
    h_samp: usize,
    v_samp: usize,
    subsampling: Subsampling,
    y_quant_values: [u16; DCT_BLOCK_SIZE],
    cb_quant_values: [u16; DCT_BLOCK_SIZE],
    cr_quant_values: [u16; DCT_BLOCK_SIZE],
    y_quant_01: u16,
    deringing: bool,
    aq_enabled: bool,
}

/// Quantized coefficients for one segment, ready for entropy encoding.
struct SegmentCoefficients {
    /// Quantized Y blocks in MCU scan order.
    y_blocks: Vec<[i16; DCT_BLOCK_SIZE]>,
    /// Quantized Cb blocks in MCU scan order.
    cb_blocks: Vec<[i16; DCT_BLOCK_SIZE]>,
    /// Quantized Cr blocks in MCU scan order.
    cr_blocks: Vec<[i16; DCT_BLOCK_SIZE]>,
    /// Symbol frequencies collected during quantization.
    frequencies: HuffmanSymbolFrequencies,
}

/// Entropy-encoded segment.
struct EncodedSegment {
    data: Vec<u8>,
    restart_num: u8,
}

/// Fused parallel encode with **fixed** Huffman tables (single pass).
///
/// Returns raw scan data with RST markers between segments.
pub fn fused_parallel_encode_fixed(
    rgb_pixels: &[u8],
    width: u32,
    height: u32,
    subsampling: Subsampling,
    tables: &HuffmanTableSet,
    y_quant: &[u16; DCT_BLOCK_SIZE],
    cb_quant: &[u16; DCT_BLOCK_SIZE],
    cr_quant: &[u16; DCT_BLOCK_SIZE],
    restart_mcu_rows: usize,
    deringing: bool,
    aq_enabled: bool,
) -> Result<Vec<u8>> {
    let shared = build_shared_config(
        width, height, subsampling, y_quant, cb_quant, cr_quant, deringing, aq_enabled,
    );
    let (mcu_rows, rows_per_seg, num_segments) = compute_segments(&shared, restart_mcu_rows)?;

    let entropy_tables = ParallelEntropyConfig {
        dc_luma: tables.dc_luma.table.clone(),
        ac_luma: tables.ac_luma.table.clone(),
        dc_chroma: tables.dc_chroma.table.clone(),
        ac_chroma: tables.ac_chroma.table.clone(),
    };

    // Single pass: quantize + entropy encode in parallel
    let segments: Vec<Result<EncodedSegment>> = (0..num_segments)
        .into_par_iter()
        .map(|seg_idx| {
            let mcu_row_start = seg_idx * rows_per_seg;
            let mcu_row_count = rows_per_seg.min(mcu_rows - mcu_row_start);
            let restart_num = (seg_idx % 8) as u8;

            let coeffs = quantize_segment(rgb_pixels, &shared, mcu_row_start, mcu_row_count)?;
            let data = entropy_encode_segment(&coeffs, &shared, &entropy_tables, mcu_row_count);
            Ok(EncodedSegment { data, restart_num })
        })
        .collect();

    combine_segments(segments)
}

/// Fused parallel encode with **optimized** Huffman tables (two passes).
///
/// Pass 1: quantize + collect frequencies in parallel.
/// Merge: combine frequencies → build optimal tables.
/// Pass 2: entropy encode in parallel with optimal tables.
///
/// Returns raw scan data with RST markers between segments.
pub fn fused_parallel_encode_optimized(
    rgb_pixels: &[u8],
    width: u32,
    height: u32,
    subsampling: Subsampling,
    y_quant: &[u16; DCT_BLOCK_SIZE],
    cb_quant: &[u16; DCT_BLOCK_SIZE],
    cr_quant: &[u16; DCT_BLOCK_SIZE],
    restart_mcu_rows: usize,
    deringing: bool,
    aq_enabled: bool,
) -> Result<(Vec<u8>, HuffmanTableSet)> {
    let shared = build_shared_config(
        width, height, subsampling, y_quant, cb_quant, cr_quant, deringing, aq_enabled,
    );
    let (mcu_rows, rows_per_seg, num_segments) = compute_segments(&shared, restart_mcu_rows)?;

    // Pass 1: quantize + collect frequencies (parallel)
    let segment_coeffs: Vec<Result<SegmentCoefficients>> = (0..num_segments)
        .into_par_iter()
        .map(|seg_idx| {
            let mcu_row_start = seg_idx * rows_per_seg;
            let mcu_row_count = rows_per_seg.min(mcu_rows - mcu_row_start);
            quantize_segment(rgb_pixels, &shared, mcu_row_start, mcu_row_count)
        })
        .collect();

    // Collect results and merge frequencies
    let mut all_coeffs = Vec::with_capacity(num_segments);
    let mut merged_freqs = HuffmanSymbolFrequencies {
            dc_luma: crate::huffman::optimize::FrequencyCounter::new(),
            ac_luma: crate::huffman::optimize::FrequencyCounter::new(),
            dc_chroma: crate::huffman::optimize::FrequencyCounter::new(),
            ac_chroma: crate::huffman::optimize::FrequencyCounter::new(),
        };
    for r in segment_coeffs {
        let coeffs = r?;
        merged_freqs.add(&coeffs.frequencies);
        all_coeffs.push(coeffs);
    }

    // Build optimal tables from merged frequencies
    let tables = merged_freqs.generate_tables()?;

    let entropy_tables = ParallelEntropyConfig {
        dc_luma: tables.dc_luma.table.clone(),
        ac_luma: tables.ac_luma.table.clone(),
        dc_chroma: tables.dc_chroma.table.clone(),
        ac_chroma: tables.ac_chroma.table.clone(),
    };

    // Pass 2: entropy encode with optimal tables (parallel)
    let segments: Vec<Result<EncodedSegment>> = all_coeffs
        .into_par_iter()
        .enumerate()
        .map(|(seg_idx, coeffs)| {
            let mcu_row_start = seg_idx * rows_per_seg;
            let mcu_row_count = rows_per_seg.min(mcu_rows - mcu_row_start);
            let restart_num = (seg_idx % 8) as u8;

            let data = entropy_encode_segment(&coeffs, &shared, &entropy_tables, mcu_row_count);
            Ok(EncodedSegment { data, restart_num })
        })
        .collect();

    let scan_data = combine_segments(segments)?;
    Ok((scan_data, tables))
}

// =============================================================================
// Internal helpers
// =============================================================================

fn build_shared_config(
    width: u32,
    height: u32,
    subsampling: Subsampling,
    y_quant: &[u16; DCT_BLOCK_SIZE],
    cb_quant: &[u16; DCT_BLOCK_SIZE],
    cr_quant: &[u16; DCT_BLOCK_SIZE],
    deringing: bool,
    aq_enabled: bool,
) -> SharedEncodeConfig {
    let width = width as usize;
    let height = height as usize;
    let (h_samp, v_samp) = match subsampling {
        Subsampling::S444 => (1, 1),
        Subsampling::S422 => (2, 1),
        Subsampling::S420 => (2, 2),
        Subsampling::S440 => (1, 2),
    };
    let mcu_height = v_samp * 8;
    let padded_width = ((width + h_samp * 8 - 1) / (h_samp * 8)) * h_samp * 8;
    let blocks_w = padded_width / 8;
    let mcu_cols = padded_width / (h_samp * 8);

    SharedEncodeConfig {
        width,
        height,
        padded_width,
        blocks_w,
        mcu_cols,
        mcu_height,
        h_samp,
        v_samp,
        subsampling,
        y_quant_values: *y_quant,
        cb_quant_values: *cb_quant,
        cr_quant_values: *cr_quant,
        y_quant_01: y_quant[1],
        deringing,
        aq_enabled,
    }
}

fn compute_segments(
    shared: &SharedEncodeConfig,
    restart_mcu_rows: usize,
) -> Result<(usize, usize, usize)> {
    let mcu_rows = (shared.height + shared.mcu_height - 1) / shared.mcu_height;
    let rows_per_seg = restart_mcu_rows.max(MIN_MCU_ROWS_PER_SEGMENT);
    let num_segments = (mcu_rows + rows_per_seg - 1) / rows_per_seg;

    if num_segments <= 1 {
        return Err(crate::error::Error::unsupported_feature(
            "fused parallel encode requires multiple restart segments",
        ));
    }

    Ok((mcu_rows, rows_per_seg, num_segments))
}

fn combine_segments(segments: Vec<Result<EncodedSegment>>) -> Result<Vec<u8>> {
    let mut total_size = 0;
    let mut results = Vec::with_capacity(segments.len());
    for r in segments {
        let seg = r?;
        total_size += seg.data.len() + 2;
        results.push(seg);
    }

    let mut output = Vec::with_capacity(total_size);
    for (i, seg) in results.iter().enumerate() {
        output.extend_from_slice(&seg.data);
        if i < results.len() - 1 {
            output.push(0xFF);
            output.push(0xD0 + seg.restart_num);
        }
    }
    Ok(output)
}

/// Pass 1: color convert → AQ → DCT → quantize for one segment.
/// Also collects Huffman symbol frequencies.
fn quantize_segment(
    rgb_pixels: &[u8],
    shared: &SharedEncodeConfig,
    mcu_row_start: usize,
    mcu_row_count: usize,
) -> Result<SegmentCoefficients> {
    let width = shared.width;
    let height = shared.height;
    let padded_width = shared.padded_width;
    let mcu_height = shared.mcu_height;
    let h_samp = shared.h_samp;
    let v_samp = shared.v_samp;
    let mcu_cols = shared.mcu_cols;

    let pixel_row_start = mcu_row_start * mcu_height;
    let pixel_row_end = ((mcu_row_start + mcu_row_count) * mcu_height).min(height);
    let seg_pixel_height = pixel_row_end - pixel_row_start;
    let seg_padded_height = mcu_row_count * mcu_height;

    // 1. Color convert
    let y_stride = padded_width;
    let mut y_plane = vec![0.0f32; seg_padded_height * y_stride];
    let c_width = (padded_width + h_samp - 1) / h_samp;
    let c_height = (seg_padded_height + v_samp - 1) / v_samp;
    let mut cb_plane = vec![0.0f32; c_height * c_width];
    let mut cr_plane = vec![0.0f32; c_height * c_width];

    color_convert_segment(
        rgb_pixels, width, height, pixel_row_start, seg_pixel_height, padded_width,
        h_samp, v_samp, &mut y_plane, &mut cb_plane, &mut cr_plane, c_width,
    );

    // 2. AQ
    let seg_blocks_w = shared.blocks_w;
    let seg_blocks_h = mcu_row_count * v_samp;
    let aq_strengths = if shared.aq_enabled {
        compute_segment_aq(
            &y_plane, width, seg_pixel_height, y_stride,
            shared.subsampling, shared.y_quant_01,
        )?
    } else {
        vec![0.0f32; seg_blocks_w * seg_blocks_h]
    };

    // 3. DCT + quantize + collect frequencies
    let y_dc_quant = shared.y_quant_values[0];
    let total_mcus = mcu_cols * mcu_row_count;
    let y_blocks_per_mcu = h_samp * v_samp;

    let mut y_blocks = Vec::with_capacity(total_mcus * y_blocks_per_mcu);
    let mut cb_blocks = Vec::with_capacity(total_mcus);
    let mut cr_blocks = Vec::with_capacity(total_mcus);
    let mut freqs = HuffmanSymbolFrequencies {
            dc_luma: crate::huffman::optimize::FrequencyCounter::new(),
            ac_luma: crate::huffman::optimize::FrequencyCounter::new(),
            dc_chroma: crate::huffman::optimize::FrequencyCounter::new(),
            ac_chroma: crate::huffman::optimize::FrequencyCounter::new(),
        };

    let mut prev_dc_y: i16 = 0;
    let mut prev_dc_cb: i16 = 0;
    let mut prev_dc_cr: i16 = 0;

    for local_mcu_row in 0..mcu_row_count {
        for mcu_col in 0..mcu_cols {
            // Y blocks
            for vy in 0..v_samp {
                for hx in 0..h_samp {
                    let block_x = mcu_col * h_samp + hx;
                    let block_y = local_mcu_row * v_samp + vy;

                    let mut block = extract_block_from_plane(
                        &y_plane, y_stride, block_x * 8, block_y * 8, seg_padded_height,
                    );
                    if shared.deringing {
                        preprocess_deringing_block(&mut block, y_dc_quant);
                    }
                    let dct_block = forward_dct_8x8_wide(&block);

                    let aq_idx = block_y * seg_blocks_w + block_x;
                    let aq_strength = aq_strengths.get(aq_idx).copied().unwrap_or(0.0);
                    let quantized = quantize_block_with_aq(
                        &dct_block, &shared.y_quant_values, aq_strength,
                    );

                    collect_frequencies(&quantized, prev_dc_y, &mut freqs.dc_luma, &mut freqs.ac_luma);
                    prev_dc_y = quantized[0];
                    y_blocks.push(quantized);
                }
            }

            // Cb
            {
                let cb_dct = extract_and_dct_chroma(&cb_plane, c_width, mcu_col, local_mcu_row, c_height);
                let cb_q = quantize_block_no_aq(&cb_dct, &shared.cb_quant_values);
                collect_frequencies(&cb_q, prev_dc_cb, &mut freqs.dc_chroma, &mut freqs.ac_chroma);
                prev_dc_cb = cb_q[0];
                cb_blocks.push(cb_q);
            }

            // Cr
            {
                let cr_dct = extract_and_dct_chroma(&cr_plane, c_width, mcu_col, local_mcu_row, c_height);
                let cr_q = quantize_block_no_aq(&cr_dct, &shared.cr_quant_values);
                collect_frequencies(&cr_q, prev_dc_cr, &mut freqs.dc_chroma, &mut freqs.ac_chroma);
                prev_dc_cr = cr_q[0];
                cr_blocks.push(cr_q);
            }
        }
    }

    Ok(SegmentCoefficients { y_blocks, cb_blocks, cr_blocks, frequencies: freqs })
}

/// Pass 2: entropy encode pre-quantized coefficients.
fn entropy_encode_segment(
    coeffs: &SegmentCoefficients,
    shared: &SharedEncodeConfig,
    tables: &ParallelEntropyConfig,
    mcu_row_count: usize,
) -> Vec<u8> {
    let mcu_cols = shared.mcu_cols;
    let h_samp = shared.h_samp;
    let v_samp = shared.v_samp;
    let total_mcus = mcu_cols * mcu_row_count;
    let y_blocks_per_mcu = h_samp * v_samp;

    let est = (total_mcus * (y_blocks_per_mcu + 2)) * 3;
    let mut encoder = EntropyEncoder::with_capacity(est);
    encoder.set_dc_table(0, &tables.dc_luma);
    encoder.set_ac_table(0, &tables.ac_luma);
    encoder.set_dc_table(1, &tables.dc_chroma);
    encoder.set_ac_table(1, &tables.ac_chroma);

    let mut y_idx = 0;
    let mut c_idx = 0;

    for mcu_idx in 0..total_mcus {
        for _ in 0..y_blocks_per_mcu {
            encoder.encode_block(&coeffs.y_blocks[y_idx], 0, 0, 0);
            y_idx += 1;
        }
        encoder.encode_block(&coeffs.cb_blocks[c_idx], 1, 1, 1);
        encoder.encode_block(&coeffs.cr_blocks[c_idx], 2, 1, 1);
        c_idx += 1;

        if mcu_idx + 1 < total_mcus {
            encoder.check_restart();
        }
    }

    encoder.finish()
}

/// Collect Huffman symbol frequencies from a quantized block.
fn collect_frequencies(
    block: &[i16; DCT_BLOCK_SIZE],
    prev_dc: i16,
    dc_freq: &mut crate::huffman::optimize::FrequencyCounter,
    ac_freq: &mut crate::huffman::optimize::FrequencyCounter,
) {
    // DC: category of the DC difference
    let dc_diff = block[0] - prev_dc;
    let dc_cat = crate::entropy::category(dc_diff);
    dc_freq.count(dc_cat as u8);

    // AC: run-length encoding symbols
    let mut zero_run = 0u8;
    for &coeff in &block[1..] {
        if coeff == 0 {
            zero_run += 1;
        } else {
            while zero_run >= 16 {
                ac_freq.count(0xF0); // ZRL
                zero_run -= 16;
            }
            let cat = crate::entropy::category(coeff);
            ac_freq.count((zero_run << 4) | cat as u8);
            zero_run = 0;
        }
    }
    if zero_run > 0 {
        ac_freq.count(0x00); // EOB
    }
}

// =============================================================================
// Pixel-level helpers (same as before)
// =============================================================================

fn color_convert_segment(
    rgb_pixels: &[u8], width: usize, height: usize,
    pixel_row_start: usize, seg_pixel_height: usize, padded_width: usize,
    h_samp: usize, v_samp: usize,
    y_plane: &mut [f32], cb_plane: &mut [f32], cr_plane: &mut [f32], c_width: usize,
) {
    let rgb_stride = width * 3;
    for local_y in 0..seg_pixel_height {
        let global_y = pixel_row_start + local_y;
        if global_y >= height { break; }
        let rgb_row = &rgb_pixels[global_y * rgb_stride..(global_y + 1) * rgb_stride];
        let y_off = local_y * padded_width;
        for x in 0..width {
            let r = rgb_row[x * 3] as f32;
            let g = rgb_row[x * 3 + 1] as f32;
            let b = rgb_row[x * 3 + 2] as f32;
            y_plane[y_off + x] = 0.299 * r + 0.587 * g + 0.114 * b;
            if h_samp == 1 && v_samp == 1 {
                cb_plane[local_y * c_width + x] = 128.0 + (-0.168736 * r - 0.331264 * g + 0.5 * b);
                cr_plane[local_y * c_width + x] = 128.0 + (0.5 * r - 0.418688 * g - 0.081312 * b);
            }
        }
        if width < padded_width {
            let v = y_plane[y_off + width - 1];
            for x in width..padded_width { y_plane[y_off + x] = v; }
        }
    }
    if h_samp > 1 || v_samp > 1 {
        box_downsample_chroma(
            rgb_pixels, width, height, pixel_row_start, seg_pixel_height,
            cb_plane, cr_plane, c_width, h_samp, v_samp,
        );
    }
}

fn box_downsample_chroma(
    rgb_pixels: &[u8], width: usize, height: usize,
    pixel_row_start: usize, seg_pixel_height: usize,
    cb_plane: &mut [f32], cr_plane: &mut [f32], c_width: usize,
    h_samp: usize, v_samp: usize,
) {
    let rgb_stride = width * 3;
    let c_height = (seg_pixel_height + v_samp - 1) / v_samp;
    for cy in 0..c_height {
        for cx in 0..c_width.min((width + h_samp - 1) / h_samp) {
            let mut cb_sum = 0.0f32;
            let mut cr_sum = 0.0f32;
            let mut count = 0.0f32;
            for dy in 0..v_samp {
                for dx in 0..h_samp {
                    let px = cx * h_samp + dx;
                    let py = cy * v_samp + dy;
                    let gy = pixel_row_start + py;
                    if px < width && gy < height {
                        let off = gy * rgb_stride + px * 3;
                        let r = rgb_pixels[off] as f32;
                        let g = rgb_pixels[off + 1] as f32;
                        let b = rgb_pixels[off + 2] as f32;
                        cb_sum += 128.0 + (-0.168736 * r - 0.331264 * g + 0.5 * b);
                        cr_sum += 128.0 + (0.5 * r - 0.418688 * g - 0.081312 * b);
                        count += 1.0;
                    }
                }
            }
            if count > 0.0 {
                let idx = cy * c_width + cx;
                cb_plane[idx] = cb_sum / count;
                cr_plane[idx] = cr_sum / count;
            }
        }
    }
}

fn compute_segment_aq(
    y_plane: &[f32], width: usize, seg_height: usize, y_stride: usize,
    subsampling: Subsampling, y_quant_01: u16,
) -> Result<Vec<f32>> {
    let layout = LayoutParams::new(width, seg_height, subsampling, false);
    let mut aq = StreamingAQ::new(&layout, y_quant_01, true)?;
    let imcu_height: usize = match subsampling {
        Subsampling::S420 | Subsampling::S440 => 16,
        _ => 8,
    };
    let mut all = Vec::new();
    for strip_y in (0..seg_height).step_by(imcu_height) {
        let h = imcu_height.min(seg_height - strip_y);
        let data = &y_plane[strip_y * y_stride..(strip_y + h) * y_stride];
        if let Some(s) = aq.process_y_strip(data, strip_y, h) {
            all.extend_from_slice(s);
        }
    }
    if let Some(s) = aq.flush() { all.extend_from_slice(s); }
    Ok(all)
}

#[inline]
fn extract_block_from_plane(
    plane: &[f32], stride: usize, bx: usize, by: usize, h: usize,
) -> Block8x8f {
    let mut block = Block8x8f::ZERO;
    for row in 0..8 {
        let sy = if by + row < h { by + row } else { h.saturating_sub(1) };
        let src = sy * stride + bx;
        if src + 8 <= plane.len() {
            block.rows[row].copy_from_slice(&plane[src..src + 8]);
        }
    }
    block
}

#[inline]
fn extract_and_dct_chroma(
    plane: &[f32], c_width: usize, mcu_col: usize, mcu_row: usize, c_height: usize,
) -> Block8x8f {
    let block = extract_block_from_plane(plane, c_width, mcu_col * 8, mcu_row * 8, c_height);
    forward_dct_8x8_wide(&block)
}

#[inline]
fn quantize_block_with_aq(
    dct: &Block8x8f, quant: &[u16; DCT_BLOCK_SIZE], aq_strength: f32,
) -> [i16; DCT_BLOCK_SIZE] {
    let mut result = [0i16; DCT_BLOCK_SIZE];
    let scale = if aq_strength != 0.0 { 2.0_f32.powf(aq_strength) } else { 1.0 };
    for row in 0..8 {
        for col in 0..8 {
            let i = row * 8 + col;
            let q = quant[i] as f32 * scale;
            result[i] = if q > 0.0 { (dct.rows[row][col] / q).round() as i16 } else { 0 };
        }
    }
    result
}

#[inline]
fn quantize_block_no_aq(
    dct: &Block8x8f, quant: &[u16; DCT_BLOCK_SIZE],
) -> [i16; DCT_BLOCK_SIZE] {
    let mut result = [0i16; DCT_BLOCK_SIZE];
    for row in 0..8 {
        for col in 0..8 {
            let i = row * 8 + col;
            let q = quant[i] as f32;
            result[i] = if q > 0.0 { (dct.rows[row][col] / q).round() as i16 } else { 0 };
        }
    }
    result
}
