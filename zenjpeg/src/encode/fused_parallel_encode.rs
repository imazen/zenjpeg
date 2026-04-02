//! Fused parallel encoder: color convert → AQ → DCT → quantize → entropy in parallel.
//!
//! Two modes:
//! - **Fixed tables**: single pass — each segment encodes directly to bytes.
//! - **Optimized tables**: two passes — pass 1 quantizes + collects frequencies in parallel,
//!   merge frequencies → build tables, pass 2 entropy-encodes in parallel.
//!
//! Each rayon task processes a horizontal band of MCU rows. AQ at segment boundaries
//! uses edge clamping (same as image edges) — imperceptible quality difference.
//!
//! Uses the same SIMD paths as the sequential encoder:
//! - `rgb_to_ycbcr_420_reuse` / `rgb_to_ycbcr_strided_fast` for color conversion
//! - `extract_block_from_strip_wide` for block extraction (with -128 level shift)
//! - `QuantTableSimd::quantize_with_zero_bias_zigzag` for fused quantize+zigzag
//! - `collect_block_frequencies_simd` for frequency collection
//! - `forward_dct_dispatch` for DCT

use rayon::prelude::*;

use crate::encode::blocks::{collect_block_frequencies_simd, HuffmanSymbolFrequencies};
use crate::encode::dct::simd::forward_dct_8x8_wide;
use crate::encode::deringing::preprocess_deringing_block;
use crate::encode::layout::LayoutParams;
use crate::encode::strip::extract_block_from_strip_wide;
use crate::entropy::encoder::EntropyEncoder;
use crate::error::Result;
use crate::foundation::consts::DCT_BLOCK_SIZE;
use crate::foundation::simd_types::{Block8x8f, QuantTableSimd, ZeroBiasSimd};
use crate::huffman::optimize::{FrequencyCounter, HuffmanTableSet};
use crate::quant::aq::streaming::StreamingAQ;
use crate::quant::ZeroBiasParams;
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
    y_quant: QuantTableSimd,
    cb_quant: QuantTableSimd,
    cr_quant: QuantTableSimd,
    y_zero_bias: ZeroBiasSimd,
    cb_zero_bias: ZeroBiasSimd,
    cr_zero_bias: ZeroBiasSimd,
    y_quant_01: u16,
    y_dc_quant: u16,
    deringing: bool,
    aq_enabled: bool,
}

/// Quantized coefficients for one segment, ready for entropy encoding.
struct SegmentCoefficients {
    y_blocks: Vec<[i16; DCT_BLOCK_SIZE]>,
    cb_blocks: Vec<[i16; DCT_BLOCK_SIZE]>,
    cr_blocks: Vec<[i16; DCT_BLOCK_SIZE]>,
    frequencies: HuffmanSymbolFrequencies,
}

/// Entropy-encoded segment.
struct EncodedSegment {
    data: Vec<u8>,
    restart_num: u8,
}

/// Fused parallel encode with **fixed** Huffman tables (single pass).
pub fn fused_parallel_encode_fixed(
    rgb_pixels: &[u8],
    width: u32,
    height: u32,
    subsampling: Subsampling,
    tables: &HuffmanTableSet,
    y_quant_values: &[u16; DCT_BLOCK_SIZE],
    cb_quant_values: &[u16; DCT_BLOCK_SIZE],
    cr_quant_values: &[u16; DCT_BLOCK_SIZE],
    y_zero_bias: &ZeroBiasParams,
    cb_zero_bias: &ZeroBiasParams,
    cr_zero_bias: &ZeroBiasParams,
    restart_mcu_rows: usize,
    deringing: bool,
    aq_enabled: bool,
) -> Result<Vec<u8>> {
    let shared = build_shared_config(
        width, height, subsampling, y_quant_values, cb_quant_values, cr_quant_values,
        y_zero_bias, cb_zero_bias, cr_zero_bias, deringing, aq_enabled,
    );
    let (mcu_rows, rows_per_seg, num_segments) = compute_segments(&shared, restart_mcu_rows)?;

    let entropy_tables = ParallelEntropyConfig {
        dc_luma: tables.dc_luma.table.clone(),
        ac_luma: tables.ac_luma.table.clone(),
        dc_chroma: tables.dc_chroma.table.clone(),
        ac_chroma: tables.ac_chroma.table.clone(),
    };

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
pub fn fused_parallel_encode_optimized(
    rgb_pixels: &[u8],
    width: u32,
    height: u32,
    subsampling: Subsampling,
    y_quant_values: &[u16; DCT_BLOCK_SIZE],
    cb_quant_values: &[u16; DCT_BLOCK_SIZE],
    cr_quant_values: &[u16; DCT_BLOCK_SIZE],
    y_zero_bias: &ZeroBiasParams,
    cb_zero_bias: &ZeroBiasParams,
    cr_zero_bias: &ZeroBiasParams,
    restart_mcu_rows: usize,
    deringing: bool,
    aq_enabled: bool,
) -> Result<(Vec<u8>, HuffmanTableSet)> {
    let shared = build_shared_config(
        width, height, subsampling, y_quant_values, cb_quant_values, cr_quant_values,
        y_zero_bias, cb_zero_bias, cr_zero_bias, deringing, aq_enabled,
    );
    let (mcu_rows, rows_per_seg, num_segments) = compute_segments(&shared, restart_mcu_rows)?;

    // Pass 1: parallel quantize + frequency collection
    let segment_coeffs: Vec<Result<SegmentCoefficients>> = (0..num_segments)
        .into_par_iter()
        .map(|seg_idx| {
            let mcu_row_start = seg_idx * rows_per_seg;
            let mcu_row_count = rows_per_seg.min(mcu_rows - mcu_row_start);
            quantize_segment(rgb_pixels, &shared, mcu_row_start, mcu_row_count)
        })
        .collect();

    let mut all_coeffs = Vec::with_capacity(num_segments);
    let mut merged_freqs = new_frequencies();
    for r in segment_coeffs {
        let coeffs = r?;
        merged_freqs.add(&coeffs.frequencies);
        all_coeffs.push(coeffs);
    }

    let tables = merged_freqs.generate_tables()?;
    let entropy_tables = ParallelEntropyConfig {
        dc_luma: tables.dc_luma.table.clone(),
        ac_luma: tables.ac_luma.table.clone(),
        dc_chroma: tables.dc_chroma.table.clone(),
        ac_chroma: tables.ac_chroma.table.clone(),
    };

    // Pass 2: parallel entropy encode with optimal tables
    let segments: Vec<Result<EncodedSegment>> = all_coeffs
        .into_par_iter()
        .enumerate()
        .map(|(seg_idx, coeffs)| {
            let mcu_row_count = rows_per_seg.min(mcu_rows - seg_idx * rows_per_seg);
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

fn new_frequencies() -> HuffmanSymbolFrequencies {
    HuffmanSymbolFrequencies {
        dc_luma: FrequencyCounter::new(),
        ac_luma: FrequencyCounter::new(),
        dc_chroma: FrequencyCounter::new(),
        ac_chroma: FrequencyCounter::new(),
    }
}

fn build_shared_config(
    width: u32, height: u32, subsampling: Subsampling,
    y_quant_values: &[u16; DCT_BLOCK_SIZE],
    cb_quant_values: &[u16; DCT_BLOCK_SIZE],
    cr_quant_values: &[u16; DCT_BLOCK_SIZE],
    y_zero_bias: &ZeroBiasParams,
    cb_zero_bias: &ZeroBiasParams,
    cr_zero_bias: &ZeroBiasParams,
    deringing: bool, aq_enabled: bool,
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

    SharedEncodeConfig {
        width,
        height,
        padded_width,
        blocks_w: padded_width / 8,
        mcu_cols: padded_width / (h_samp * 8),
        mcu_height,
        h_samp,
        v_samp,
        subsampling,
        y_quant: QuantTableSimd::from_values(y_quant_values),
        cb_quant: QuantTableSimd::from_values(cb_quant_values),
        cr_quant: QuantTableSimd::from_values(cr_quant_values),
        y_zero_bias: ZeroBiasSimd::from_params(y_zero_bias),
        cb_zero_bias: ZeroBiasSimd::from_params(cb_zero_bias),
        cr_zero_bias: ZeroBiasSimd::from_params(cr_zero_bias),
        y_quant_01: y_quant_values[1],
        y_dc_quant: y_quant_values[0],
        deringing,
        aq_enabled,
    }
}

fn compute_segments(shared: &SharedEncodeConfig, restart_mcu_rows: usize) -> Result<(usize, usize, usize)> {
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
    let mut results = Vec::with_capacity(segments.len());
    let mut total_size = 0;
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

/// Pass 1: color convert → AQ → DCT → quantize + collect frequencies.
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
    let blocks_w = shared.blocks_w;

    let pixel_row_start = mcu_row_start * mcu_height;
    let pixel_row_end = ((mcu_row_start + mcu_row_count) * mcu_height).min(height);
    let seg_pixel_height = pixel_row_end - pixel_row_start;
    let seg_padded_height = mcu_row_count * mcu_height;

    // 1. Color convert using SIMD paths
    let y_stride = padded_width;
    let mut y_plane = vec![0.0f32; seg_padded_height * y_stride];
    let c_width = (padded_width + h_samp - 1) / h_samp;
    let c_height = (seg_padded_height + v_samp - 1) / v_samp;
    let mut cb_plane = vec![0.0f32; c_height * c_width];
    let mut cr_plane = vec![0.0f32; c_height * c_width];

    // Use the SIMD color conversion (same as strip processor)
    let bpp = 3; // RGB8
    let seg_rgb = &rgb_pixels[pixel_row_start * width * bpp..];
    let seg_rgb_len = seg_pixel_height * width * bpp;

    if h_samp == 2 && v_samp == 2 {
        // 4:2:0: fused color convert + box downsample
        let mut temp_y = vec![0u8; seg_pixel_height * padded_width];
        let mut temp_cb = vec![0u8; seg_pixel_height * padded_width];
        let mut temp_cr = vec![0u8; seg_pixel_height * padded_width];
        crate::color::fast_yuv::rgb_to_ycbcr_420_reuse(
            &seg_rgb[..seg_rgb_len],
            &mut y_plane,
            &mut cb_plane,
            &mut cr_plane,
            &mut temp_y,
            &mut temp_cb,
            &mut temp_cr,
            width,
            seg_pixel_height,
            y_stride,
            bpp,
        );
    } else {
        // 4:4:4: full-resolution color convert
        crate::color::fast_yuv::rgb_to_ycbcr_strided_fast(
            &seg_rgb[..seg_rgb_len],
            &mut y_plane,
            &mut cb_plane,
            &mut cr_plane,
            width,
            seg_pixel_height,
            y_stride,
            bpp,
        );
    }

    // Edge-replicate Y to padded width
    if width < padded_width {
        for row in 0..seg_pixel_height {
            let off = row * y_stride;
            let last = y_plane[off + width - 1];
            for x in width..padded_width {
                y_plane[off + x] = last;
            }
        }
    }

    // 2. AQ
    let seg_blocks_h = mcu_row_count * v_samp;
    let aq_strengths = if shared.aq_enabled {
        compute_segment_aq(&y_plane, width, seg_pixel_height, y_stride, shared.subsampling, shared.y_quant_01)?
    } else {
        vec![0.0f32; blocks_w * seg_blocks_h]
    };

    // 3. DCT + quantize using SIMD paths (extract_block_from_strip_wide + QuantTableSimd)
    let total_mcus = mcu_cols * mcu_row_count;
    let y_blocks_per_mcu = h_samp * v_samp;

    let mut y_blocks = Vec::with_capacity(total_mcus * y_blocks_per_mcu);
    let mut cb_blocks = Vec::with_capacity(total_mcus);
    let mut cr_blocks = Vec::with_capacity(total_mcus);
    let mut freqs = new_frequencies();

    let mut prev_dc_y: i16 = 0;
    let mut prev_dc_cb: i16 = 0;
    let mut prev_dc_cr: i16 = 0;

    for local_mcu_row in 0..mcu_row_count {
        for mcu_col in 0..mcu_cols {
            // Y blocks — use extract_block_from_strip_wide (SIMD, applies -128 level shift)
            for vy in 0..v_samp {
                for hx in 0..h_samp {
                    let block_x = mcu_col * h_samp + hx;
                    let block_y = local_mcu_row * v_samp + vy;

                    let mut block = extract_block_from_strip_wide(
                        &y_plane, block_x, block_y, y_stride,
                    );

                    if shared.deringing {
                        preprocess_deringing_block(&mut block, shared.y_dc_quant);
                    }

                    let dct_block = forward_dct_8x8_wide(&block);

                    // SIMD quantize + zigzag (matches sequential encoder exactly)
                    let aq_idx = block_y * blocks_w + block_x;
                    let aq_strength = aq_strengths.get(aq_idx).copied().unwrap_or(0.0);
                    let quantized = shared.y_quant.quantize_with_zero_bias_zigzag(
                        &dct_block, &shared.y_zero_bias, aq_strength,
                    );

                    collect_block_frequencies_simd(&quantized, prev_dc_y, &mut freqs.dc_luma, &mut freqs.ac_luma);
                    prev_dc_y = quantized[0];
                    y_blocks.push(quantized);
                }
            }

            // Cb — extract from chroma plane (which is at half resolution for 4:2:0)
            {
                let cb_block = extract_block_from_strip_wide(&cb_plane, mcu_col, local_mcu_row, c_width);
                let cb_dct = forward_dct_8x8_wide(&cb_block);
                let cb_q = shared.cb_quant.quantize_with_zero_bias_zigzag(&cb_dct, &shared.cb_zero_bias, 0.0);
                collect_block_frequencies_simd(&cb_q, prev_dc_cb, &mut freqs.dc_chroma, &mut freqs.ac_chroma);
                prev_dc_cb = cb_q[0];
                cb_blocks.push(cb_q);
            }

            // Cr
            {
                let cr_block = extract_block_from_strip_wide(&cr_plane, mcu_col, local_mcu_row, c_width);
                let cr_dct = forward_dct_8x8_wide(&cr_block);
                let cr_q = shared.cr_quant.quantize_with_zero_bias_zigzag(&cr_dct, &shared.cr_zero_bias, 0.0);
                collect_block_frequencies_simd(&cr_q, prev_dc_cr, &mut freqs.dc_chroma, &mut freqs.ac_chroma);
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
    let y_blocks_per_mcu = shared.h_samp * shared.v_samp;
    let total_mcus = mcu_cols * mcu_row_count;

    let est = total_mcus * (y_blocks_per_mcu + 2) * 3;
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

/// Compute AQ strengths for a segment using an independent StreamingAQ instance.
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
