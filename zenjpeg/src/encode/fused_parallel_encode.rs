//! Fused parallel encoder: color convert → AQ → DCT → quantize → entropy in parallel.
//!
//! **Optimized-table mode (two passes, minimal memory):**
//! - Pass 1 (parallel): color convert → AQ → DCT → quantize → collect frequencies.
//!   Stores only AQ strengths (~150KB) and frequency counters (~8KB per segment).
//!   Planes and coefficients are discarded.
//! - Merge: combine frequencies → build optimal Huffman tables.
//! - Pass 2 (parallel): recompute color convert → DCT → quantize → entropy encode.
//!   Uses stored AQ strengths + optimal tables. ~22% more compute, zero coeff storage.
//!
//! Thread-local buffer pools avoid per-segment allocation overhead.
//!
//! Uses the same SIMD paths as the sequential encoder:
//! - `rgb_to_ycbcr_420_reuse` / `rgb_to_ycbcr_strided_fast` for color conversion
//! - `extract_block_from_strip_wide` for block extraction (with -128 level shift)
//! - `QuantTableSimd::quantize_with_zero_bias_zigzag` for fused quantize+zigzag
//! - `collect_block_frequencies_simd` for frequency collection
//! - `forward_dct_8x8_wide` for DCT

use core::cell::RefCell;
use rayon::prelude::*;

use crate::encode::blocks::{collect_block_frequencies_simd, HuffmanSymbolFrequencies};
use crate::encode::dct::simd::forward_dct_8x8_wide;
use crate::encode::deringing::preprocess_deringing_block;
use crate::encode::layout::LayoutParams;
use crate::encode::strip::extract_block_from_strip_wide;
use crate::entropy::encoder::EntropyEncoder;
use crate::error::Result;
use crate::foundation::consts::DCT_BLOCK_SIZE;
use crate::foundation::simd_types::{QuantTableSimd, ZeroBiasSimd};
use crate::huffman::optimize::{FrequencyCounter, HuffmanTableSet};
use crate::quant::aq::streaming::StreamingAQ;
use crate::quant::ZeroBiasParams;
use crate::types::Subsampling;

use super::parallel::ParallelEntropyConfig;

/// Minimum MCU rows per segment to justify parallel overhead.
const MIN_MCU_ROWS_PER_SEGMENT: usize = 2;

// =============================================================================
// Thread-local buffer pool
// =============================================================================

/// Reusable buffers for one segment's processing. Avoids per-segment allocation.
struct SegmentBuffers {
    y_plane: Vec<f32>,
    cb_plane: Vec<f32>,
    cr_plane: Vec<f32>,
    temp_y: Vec<u8>,
    temp_cb: Vec<u8>,
    temp_cr: Vec<u8>,
}

impl SegmentBuffers {
    fn ensure_capacity(
        &mut self,
        y_size: usize,
        c_size: usize,
        temp_size: usize,
    ) {
        resize_reuse(&mut self.y_plane, y_size);
        resize_reuse(&mut self.cb_plane, c_size);
        resize_reuse(&mut self.cr_plane, c_size);
        resize_reuse_u8(&mut self.temp_y, temp_size);
        resize_reuse_u8(&mut self.temp_cb, temp_size);
        resize_reuse_u8(&mut self.temp_cr, temp_size);
    }
}

impl Default for SegmentBuffers {
    fn default() -> Self {
        Self {
            y_plane: Vec::new(),
            cb_plane: Vec::new(),
            cr_plane: Vec::new(),
            temp_y: Vec::new(),
            temp_cb: Vec::new(),
            temp_cr: Vec::new(),
        }
    }
}

/// Resize vec to `len`, reusing existing allocation. Fills new elements with 0.
#[inline]
fn resize_reuse(v: &mut Vec<f32>, len: usize) {
    v.clear();
    v.resize(len, 0.0);
}

#[inline]
fn resize_reuse_u8(v: &mut Vec<u8>, len: usize) {
    v.clear();
    v.resize(len, 0);
}

thread_local! {
    static BUFFERS: RefCell<SegmentBuffers> = RefCell::new(SegmentBuffers::default());
}

// =============================================================================
// Shared config
// =============================================================================

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

/// Lightweight result from pass 1: AQ strengths + frequencies. No coefficients.
struct Pass1Result {
    aq_strengths: Vec<f32>,
    frequencies: HuffmanSymbolFrequencies,
}

struct EncodedSegment {
    data: Vec<u8>,
    restart_num: u8,
}

// =============================================================================
// Public API
// =============================================================================

/// Fused parallel encode with optimized Huffman tables.
///
/// Two passes: pass 1 collects frequencies (parallel), merge + build tables,
/// pass 2 encodes (parallel). Only AQ strengths stored between passes (~150KB).
/// Color convert + DCT + quantize recomputed in pass 2 (~22% extra compute,
/// but zero coefficient storage and no cache pressure).
///
/// Returns `(scan_data, optimal_tables)` — caller wraps scan_data in JPEG headers.
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

    // =========================================================================
    // Pass 1: parallel color convert → AQ → DCT → quantize → collect frequencies
    // Stores only AQ strengths + frequency counters. Discards planes + coefficients.
    // =========================================================================
    let pass1_results: Vec<Result<Pass1Result>> = (0..num_segments)
        .into_par_iter()
        .map(|seg_idx| {
            let mcu_row_start = seg_idx * rows_per_seg;
            let mcu_row_count = rows_per_seg.min(mcu_rows - mcu_row_start);
            pass1_segment(rgb_pixels, &shared, mcu_row_start, mcu_row_count)
        })
        .collect();

    // Merge frequencies
    let mut all_pass1 = Vec::with_capacity(num_segments);
    let mut merged_freqs = new_frequencies();
    for r in pass1_results {
        let p1 = r?;
        merged_freqs.add(&p1.frequencies);
        all_pass1.push(p1);
    }

    // Build optimal tables
    let tables = merged_freqs.generate_tables()?;
    let entropy_tables = ParallelEntropyConfig {
        dc_luma: tables.dc_luma.table.clone(),
        ac_luma: tables.ac_luma.table.clone(),
        dc_chroma: tables.dc_chroma.table.clone(),
        ac_chroma: tables.ac_chroma.table.clone(),
    };

    // =========================================================================
    // Pass 2: parallel color convert → DCT → quantize → entropy encode
    // Reuses stored AQ strengths. Recomputes color convert + DCT + quantize.
    // =========================================================================
    let segments: Vec<Result<EncodedSegment>> = all_pass1
        .into_par_iter()
        .enumerate()
        .map(|(seg_idx, p1)| {
            let mcu_row_start = seg_idx * rows_per_seg;
            let mcu_row_count = rows_per_seg.min(mcu_rows - mcu_row_start);
            let restart_num = (seg_idx % 8) as u8;
            let data = pass2_segment(
                rgb_pixels, &shared, &entropy_tables,
                mcu_row_start, mcu_row_count, &p1.aq_strengths,
            )?;
            Ok(EncodedSegment { data, restart_num })
        })
        .collect();

    let scan_data = combine_segments(segments)?;
    Ok((scan_data, tables))
}

// =============================================================================
// Pass 1: AQ + frequency collection (no coefficient storage)
// =============================================================================

/// Pass 1 for one segment: color convert → AQ → DCT → quantize → collect frequencies.
/// Returns AQ strengths and frequency counters only.
fn pass1_segment(
    rgb_pixels: &[u8],
    shared: &SharedEncodeConfig,
    mcu_row_start: usize,
    mcu_row_count: usize,
) -> Result<Pass1Result> {
    let blocks_w = shared.blocks_w;
    let mcu_cols = shared.mcu_cols;
    let h_samp = shared.h_samp;
    let v_samp = shared.v_samp;

    let (seg_pixel_height, seg_padded_height, c_width, c_height) =
        segment_dimensions(shared, mcu_row_start, mcu_row_count);

    // Color convert into thread-local buffers
    BUFFERS.with(|cell| {
        let mut bufs = cell.borrow_mut();
        let y_size = seg_padded_height * shared.padded_width;
        let c_size = c_height * c_width;
        let temp_size = seg_pixel_height * shared.padded_width;
        bufs.ensure_capacity(y_size, c_size, temp_size);

        color_convert_segment(
            rgb_pixels, shared, mcu_row_start, seg_pixel_height, seg_padded_height,
            c_width, &mut bufs,
        );

        // AQ
        let seg_blocks_h = mcu_row_count * v_samp;
        let aq_strengths = if shared.aq_enabled {
            compute_segment_aq(
                &bufs.y_plane, shared.width, seg_pixel_height,
                shared.padded_width, shared.subsampling, shared.y_quant_01,
            )?
        } else {
            vec![0.0f32; blocks_w * seg_blocks_h]
        };

        // DCT + quantize + collect frequencies (coefficients discarded)
        let total_mcus = mcu_cols * mcu_row_count;
        let mut freqs = new_frequencies();
        let mut prev_dc_y: i16 = 0;
        let mut prev_dc_cb: i16 = 0;
        let mut prev_dc_cr: i16 = 0;

        for local_mcu_row in 0..mcu_row_count {
            for mcu_col in 0..mcu_cols {
                for vy in 0..v_samp {
                    for hx in 0..h_samp {
                        let bx = mcu_col * h_samp + hx;
                        let by = local_mcu_row * v_samp + vy;
                        let mut block = extract_block_from_strip_wide(
                            &bufs.y_plane, bx, by, shared.padded_width,
                        );
                        if shared.deringing {
                            preprocess_deringing_block(&mut block, shared.y_dc_quant);
                        }
                        let dct = forward_dct_8x8_wide(&block);
                        let aq_idx = by * blocks_w + bx;
                        let aq_s = aq_strengths.get(aq_idx).copied().unwrap_or(0.0);
                        let q = shared.y_quant.quantize_with_zero_bias_zigzag(
                            &dct, &shared.y_zero_bias, aq_s,
                        );
                        collect_block_frequencies_simd(&q, prev_dc_y, &mut freqs.dc_luma, &mut freqs.ac_luma);
                        prev_dc_y = q[0];
                    }
                }

                {
                    let cb = extract_block_from_strip_wide(&bufs.cb_plane, mcu_col, local_mcu_row, c_width);
                    let cb_q = shared.cb_quant.quantize_with_zero_bias_zigzag(
                        &forward_dct_8x8_wide(&cb), &shared.cb_zero_bias, 0.0,
                    );
                    collect_block_frequencies_simd(&cb_q, prev_dc_cb, &mut freqs.dc_chroma, &mut freqs.ac_chroma);
                    prev_dc_cb = cb_q[0];
                }
                {
                    let cr = extract_block_from_strip_wide(&bufs.cr_plane, mcu_col, local_mcu_row, c_width);
                    let cr_q = shared.cr_quant.quantize_with_zero_bias_zigzag(
                        &forward_dct_8x8_wide(&cr), &shared.cr_zero_bias, 0.0,
                    );
                    collect_block_frequencies_simd(&cr_q, prev_dc_cr, &mut freqs.dc_chroma, &mut freqs.ac_chroma);
                    prev_dc_cr = cr_q[0];
                }

                let _ = total_mcus; // suppress unused warning
            }
        }

        Ok(Pass1Result { aq_strengths, frequencies: freqs })
    })
}

// =============================================================================
// Pass 2: encode with optimal tables (recomputes color convert + DCT + quantize)
// =============================================================================

/// Pass 2 for one segment: recompute pipeline + entropy encode with optimal tables.
fn pass2_segment(
    rgb_pixels: &[u8],
    shared: &SharedEncodeConfig,
    tables: &ParallelEntropyConfig,
    mcu_row_start: usize,
    mcu_row_count: usize,
    aq_strengths: &[f32],
) -> Result<Vec<u8>> {
    let blocks_w = shared.blocks_w;
    let mcu_cols = shared.mcu_cols;
    let h_samp = shared.h_samp;
    let v_samp = shared.v_samp;
    let total_mcus = mcu_cols * mcu_row_count;

    let (seg_pixel_height, seg_padded_height, c_width, c_height) =
        segment_dimensions(shared, mcu_row_start, mcu_row_count);

    BUFFERS.with(|cell| {
        let mut bufs = cell.borrow_mut();
        let y_size = seg_padded_height * shared.padded_width;
        let c_size = c_height * c_width;
        let temp_size = seg_pixel_height * shared.padded_width;
        bufs.ensure_capacity(y_size, c_size, temp_size);

        // Recompute color conversion
        color_convert_segment(
            rgb_pixels, shared, mcu_row_start, seg_pixel_height, seg_padded_height,
            c_width, &mut bufs,
        );

        // Entropy encode
        let est = total_mcus * (h_samp * v_samp + 2) * 3;
        let mut encoder = EntropyEncoder::with_capacity(est);
        encoder.set_dc_table(0, &tables.dc_luma);
        encoder.set_ac_table(0, &tables.ac_luma);
        encoder.set_dc_table(1, &tables.dc_chroma);
        encoder.set_ac_table(1, &tables.ac_chroma);

        for local_mcu_row in 0..mcu_row_count {
            for mcu_col in 0..mcu_cols {
                for vy in 0..v_samp {
                    for hx in 0..h_samp {
                        let bx = mcu_col * h_samp + hx;
                        let by = local_mcu_row * v_samp + vy;
                        let mut block = extract_block_from_strip_wide(
                            &bufs.y_plane, bx, by, shared.padded_width,
                        );
                        if shared.deringing {
                            preprocess_deringing_block(&mut block, shared.y_dc_quant);
                        }
                        let dct = forward_dct_8x8_wide(&block);
                        let aq_idx = by * blocks_w + bx;
                        let aq_s = aq_strengths.get(aq_idx).copied().unwrap_or(0.0);
                        let q = shared.y_quant.quantize_with_zero_bias_zigzag(
                            &dct, &shared.y_zero_bias, aq_s,
                        );
                        encoder.encode_block(&q, 0, 0, 0);
                    }
                }

                {
                    let cb = extract_block_from_strip_wide(&bufs.cb_plane, mcu_col, local_mcu_row, c_width);
                    let cb_q = shared.cb_quant.quantize_with_zero_bias_zigzag(
                        &forward_dct_8x8_wide(&cb), &shared.cb_zero_bias, 0.0,
                    );
                    encoder.encode_block(&cb_q, 1, 1, 1);
                }
                {
                    let cr = extract_block_from_strip_wide(&bufs.cr_plane, mcu_col, local_mcu_row, c_width);
                    let cr_q = shared.cr_quant.quantize_with_zero_bias_zigzag(
                        &forward_dct_8x8_wide(&cr), &shared.cr_zero_bias, 0.0,
                    );
                    encoder.encode_block(&cr_q, 2, 1, 1);
                }

                let mcu_idx = local_mcu_row * mcu_cols + mcu_col;
                if mcu_idx + 1 < total_mcus {
                    encoder.check_restart();
                }
            }
        }

        Ok(encoder.finish())
    })
}

// =============================================================================
// Shared helpers
// =============================================================================

fn segment_dimensions(
    shared: &SharedEncodeConfig,
    mcu_row_start: usize,
    mcu_row_count: usize,
) -> (usize, usize, usize, usize) {
    let pixel_row_start = mcu_row_start * shared.mcu_height;
    let pixel_row_end = ((mcu_row_start + mcu_row_count) * shared.mcu_height).min(shared.height);
    let seg_pixel_height = pixel_row_end - pixel_row_start;
    let seg_padded_height = mcu_row_count * shared.mcu_height;
    let c_width = (shared.padded_width + shared.h_samp - 1) / shared.h_samp;
    let c_height = (seg_padded_height + shared.v_samp - 1) / shared.v_samp;
    (seg_pixel_height, seg_padded_height, c_width, c_height)
}

fn color_convert_segment(
    rgb_pixels: &[u8],
    shared: &SharedEncodeConfig,
    mcu_row_start: usize,
    seg_pixel_height: usize,
    _seg_padded_height: usize,
    _c_width: usize,
    bufs: &mut SegmentBuffers,
) {
    let pixel_row_start = mcu_row_start * shared.mcu_height;
    let bpp = 3;
    let seg_rgb = &rgb_pixels[pixel_row_start * shared.width * bpp..];
    let seg_rgb_len = seg_pixel_height * shared.width * bpp;

    if shared.h_samp == 2 && shared.v_samp == 2 {
        crate::color::fast_yuv::rgb_to_ycbcr_420_reuse(
            &seg_rgb[..seg_rgb_len],
            &mut bufs.y_plane,
            &mut bufs.cb_plane,
            &mut bufs.cr_plane,
            &mut bufs.temp_y,
            &mut bufs.temp_cb,
            &mut bufs.temp_cr,
            shared.width,
            seg_pixel_height,
            shared.padded_width,
            bpp,
        );
    } else {
        crate::color::fast_yuv::rgb_to_ycbcr_strided_fast(
            &seg_rgb[..seg_rgb_len],
            &mut bufs.y_plane,
            &mut bufs.cb_plane,
            &mut bufs.cr_plane,
            shared.width,
            seg_pixel_height,
            shared.padded_width,
            bpp,
        );
    }

    // Edge-replicate Y to padded width
    if shared.width < shared.padded_width {
        for row in 0..seg_pixel_height {
            let off = row * shared.padded_width;
            let last = bufs.y_plane[off + shared.width - 1];
            for x in shared.width..shared.padded_width {
                bufs.y_plane[off + x] = last;
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
        width, height, padded_width,
        blocks_w: padded_width / 8,
        mcu_cols: padded_width / (h_samp * 8),
        mcu_height, h_samp, v_samp, subsampling,
        y_quant: QuantTableSimd::from_values(y_quant_values),
        cb_quant: QuantTableSimd::from_values(cb_quant_values),
        cr_quant: QuantTableSimd::from_values(cr_quant_values),
        y_zero_bias: ZeroBiasSimd::from_params(y_zero_bias),
        cb_zero_bias: ZeroBiasSimd::from_params(cb_zero_bias),
        cr_zero_bias: ZeroBiasSimd::from_params(cr_zero_bias),
        y_quant_01: y_quant_values[1],
        y_dc_quant: y_quant_values[0],
        deringing, aq_enabled,
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
