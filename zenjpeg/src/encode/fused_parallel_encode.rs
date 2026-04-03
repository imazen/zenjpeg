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

use crate::encode::blocks::HuffmanSymbolFrequencies;
use crate::encode::dct::simd::forward_dct_8x8_wide;
use crate::encode::deringing::preprocess_deringing_block;
use crate::encode::layout::LayoutParams;
use crate::encode::strip::extract_block_from_strip_wide;
use crate::error::Result;
use crate::foundation::consts::DCT_BLOCK_SIZE;
use crate::foundation::simd_types::{QuantTableSimd, ZeroBiasSimd};
use crate::huffman::optimize::{FrequencyCounter, HuffmanTableSet};
use crate::quant::aq::streaming::StreamingAQ;
use crate::quant::ZeroBiasParams;
use crate::types::Subsampling;

use super::symbol_stream::{SymbolStream, block_to_symbols};



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

/// Result from single pass: symbol stream + frequencies. No coefficients stored.
struct SegmentSymbols {
    stream: super::symbol_stream::SymbolStream,
    frequencies: HuffmanSymbolFrequencies,
}

struct EncodedSegment {
    data: Vec<u8>,
    restart_num: u8,
}

// =============================================================================
// Public API
// =============================================================================

/// Fused parallel encode with optimized Huffman tables (single-pass + remap).
///
/// 1. **Single pass (parallel):** color convert → AQ → DCT → quantize → R-D optimize
///    → capture symbol stream + collect frequencies. No recomputation needed.
/// 2. **Merge (serial):** combine frequencies → build optimal Huffman tables.
/// 3. **Remap (parallel):** encode each segment's symbol stream with optimal tables.
///    This is a cheap linear scan — no DCT, no quantize, no color convert.
///
/// Symbol streams are ~3MB total for 4K (vs 24MB for coefficient storage or
/// 2× full pipeline for the two-pass recompute approach).
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
    // Single pass: parallel color convert → AQ → DCT → quantize → symbol capture
    // =========================================================================

    // Pre-allocate result slots to avoid per-segment Vec allocation in rayon closures
    let est_symbols_per_seg = {
        let mcu_cols = shared.mcu_cols;
        let seg_mcus = mcu_cols * rows_per_seg;
        seg_mcus * (shared.h_samp * shared.v_samp + 2) * 5
    };
    let mut all_symbols: Vec<Option<SegmentSymbols>> = (0..num_segments)
        .map(|_| Some(SegmentSymbols {
            stream: SymbolStream::with_capacity(est_symbols_per_seg),
            frequencies: new_frequencies(),
        }))
        .collect();

    // Limit parallelism to physical cores (SMT threads hurt due to L1/L2 contention).
    // Group segments into chunks so we spawn ~physical_cores tasks, not num_segments.
    let max_tasks = (rayon::current_num_threads() / 2).max(2).min(num_segments);
    let chunk_size = (num_segments + max_tasks - 1) / max_tasks;

    // Run parallel quantization in grouped chunks
    let errors: Vec<Option<crate::error::Error>> = all_symbols
        .par_chunks_mut(chunk_size)
        .enumerate()
        .map(|(chunk_idx, chunk)| {
            let base_seg = chunk_idx * chunk_size;
            for (i, slot) in chunk.iter_mut().enumerate() {
                let seg_idx = base_seg + i;
                let mcu_row_start = seg_idx * rows_per_seg;
                let mcu_row_count = rows_per_seg.min(mcu_rows - mcu_row_start);
                let restart_num = (seg_idx % 8) as u8;
                let mut seg = slot.take().unwrap();
                seg.stream.symbols.clear();
                seg.stream.flags.clear();
                match quantize_to_symbols_into(
                    rgb_pixels, &shared, mcu_row_start, mcu_row_count, restart_num, &mut seg,
                ) {
                    Ok(()) => { *slot = Some(seg); }
                    Err(e) => return Some(e),
                }
            }
            None
        })
        .collect();

    // Check errors
    for e in errors.into_iter().flatten() {
        return Err(e);
    }

    // Merge frequencies
    let mut merged_freqs = new_frequencies();
    for seg in all_symbols.iter().flatten() {
        merged_freqs.add(&seg.frequencies);
    }

    // Build optimal tables
    let tables = merged_freqs.generate_tables()?;

    // =========================================================================
    // Remap: parallel encode symbol streams with optimal tables (cheap linear scan)
    // =========================================================================
    let segments: Vec<Result<EncodedSegment>> = all_symbols
        .into_par_iter()
        .enumerate()
        .map(|(seg_idx, slot)| {
            let seg = slot.expect("segment should be filled");
            let restart_num = (seg_idx % 8) as u8;
            let data = seg.stream.encode_with_tables(
                &tables.dc_luma.table,
                &tables.ac_luma.table,
                &tables.dc_chroma.table,
                &tables.ac_chroma.table,
            );
            Ok(EncodedSegment { data, restart_num })
        })
        .collect();

    let scan_data = combine_segments(segments)?;
    Ok((scan_data, tables))
}

// =============================================================================
// Pass 1: AQ + frequency collection (no coefficient storage)
// =============================================================================

/// Single pass: color convert → AQ → DCT → quantize → R-D optimize → symbol capture.
/// Writes into a pre-allocated SegmentSymbols (symbols + frequencies).
fn quantize_to_symbols_into(
    rgb_pixels: &[u8],
    shared: &SharedEncodeConfig,
    mcu_row_start: usize,
    mcu_row_count: usize,
    _restart_num: u8,
    out: &mut SegmentSymbols,
) -> Result<()> {
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

        // DCT + quantize + R-D optimize + capture to symbol stream
        let total_block_rows = mcu_row_count * v_samp;
        let total_mcus = mcu_cols * mcu_row_count;
        let mut prev_dc_y: i16 = 0;
        let mut prev_dc_cb: i16 = 0;
        let mut prev_dc_cr: i16 = 0;
        let lambda = 0.001f32;

        // Use out.stream directly (mutable borrow through function parameter)

        for local_mcu_row in 0..mcu_row_count {
            let is_first_mcu_row = local_mcu_row == 0;
            for mcu_col in 0..mcu_cols {
                let is_first_mcu = is_first_mcu_row && mcu_col == 0;
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
                        let raw_aq = aq_strengths.get(aq_idx).copied().unwrap_or(0.0);
                        let aq_s = boundary_aq_adjust(raw_aq, by, total_block_rows);
                        let aq_scale = if aq_s != 0.0 { 2.0_f32.powf(aq_s) } else { 1.0 };
                        let mut q = shared.y_quant.quantize_with_zero_bias_zigzag(
                            &dct, &shared.y_zero_bias, aq_s,
                        );
                        optimize_block_rd(&mut q, &shared.y_quant.values, aq_scale, lambda);
                        bias_dc_at_restart(&mut q, is_first_mcu);
                        block_to_symbols(&mut out.stream, &q, &mut prev_dc_y, false);
                    }
                }
                {
                    let cb = extract_block_from_strip_wide(&bufs.cb_plane, mcu_col, local_mcu_row, c_width);
                    let mut cb_q = shared.cb_quant.quantize_with_zero_bias_zigzag(
                        &forward_dct_8x8_wide(&cb), &shared.cb_zero_bias, 0.0,
                    );
                    optimize_block_rd(&mut cb_q, &shared.cb_quant.values, 1.0, lambda);
                    block_to_symbols(&mut out.stream, &cb_q, &mut prev_dc_cb, true);
                }
                {
                    let cr = extract_block_from_strip_wide(&bufs.cr_plane, mcu_col, local_mcu_row, c_width);
                    let mut cr_q = shared.cr_quant.quantize_with_zero_bias_zigzag(
                        &forward_dct_8x8_wide(&cr), &shared.cr_zero_bias, 0.0,
                    );
                    optimize_block_rd(&mut cr_q, &shared.cr_quant.values, 1.0, lambda);
                    block_to_symbols(&mut out.stream, &cr_q, &mut prev_dc_cr, true);
                }
            }
        }

        // Collect frequencies from the symbol stream
        out.frequencies = new_frequencies();
        out.stream.collect_frequencies(
            &mut out.frequencies.dc_luma, &mut out.frequencies.ac_luma,
            &mut out.frequencies.dc_chroma, &mut out.frequencies.ac_chroma,
        );

        Ok(())
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

// =============================================================================
// R-D post-quantization optimizations (A-D)
// =============================================================================

/// Estimate bits to encode one AC coefficient with the given zero run.
#[inline]
fn estimate_ac_bits(value: i16, _zero_run: u8) -> u32 {
    if value == 0 {
        return 0;
    }
    let cat = crate::entropy::category(value) as u32;
    // Huffman code for (run, cat) ≈ 4-8 bits + cat magnitude bits
    // Approximate: larger categories and runs cost more
    4 + cat
}

/// R-D optimize a quantized block: drop trailing coefficients where the
/// bit savings exceed the perceptual distortion cost.
///
/// Scans from position 63 down to 1 (zigzag order). For each nonzero coefficient
/// that is currently the last nonzero: computes bits saved (the coefficient's
/// encoding + EOB advancement) vs distortion cost (reconstruction error weighted
/// by perceptual sensitivity). Drops the coefficient if savings exceed cost.
///
/// `quant_values` are the raw quantization table values (NOT scaled by AQ).
/// `aq_scale` is `2^aq_strength` — the AQ multiplier for this block.
/// `lambda` controls the aggressiveness: higher = more dropping = smaller file.
#[inline]
fn optimize_block_rd(
    block: &mut [i16; DCT_BLOCK_SIZE],
    quant_values: &[u16; DCT_BLOCK_SIZE],
    aq_scale: f32,
    lambda: f32,
) {
    // Find last nonzero AC coefficient
    let mut last_nz = 63;
    while last_nz > 0 && block[last_nz] == 0 {
        last_nz -= 1;
    }
    if last_nz == 0 {
        return; // DC only, nothing to optimize
    }

    // Scan from the end, dropping trailing coefficients
    for k in (1..=last_nz).rev() {
        if block[k] == 0 {
            continue;
        }

        // This is the current last nonzero — dropping it advances EOB
        let q_val = block[k];

        // Only consider dropping ±1 coefficients (the marginal ones)
        if q_val.unsigned_abs() > 1 {
            break; // coefficient too large to drop without visible quality loss
        }

        // Bits saved by dropping this coefficient (advancing EOB)
        let mut run = 0u8;
        for j in (1..k).rev() {
            if block[j] != 0 { break; }
            run += 1;
        }
        let bits_saved = estimate_ac_bits(q_val, run) + (run as u32 / 16) * 4;

        // Distortion: dropping ±1 at position k. Quality cost is inversely
        // proportional to the quant step — large Q means the coefficient was
        // barely significant to begin with.
        let q_step = quant_values[k] as f32 * aq_scale;
        // Normalized distortion: 1/Q² — smaller for high-freq (large Q) positions
        let distortion = 1.0 / (q_step * q_step);

        if (bits_saved as f32) * lambda > distortion {
            block[k] = 0;
            // Continue scanning — next nonzero becomes the new tail candidate
        } else {
            break; // This coefficient is worth keeping, so are all earlier ones
        }
    }
}

/// Widen zero bias at segment boundaries (option B).
/// For blocks in the first/last block row of a segment, scale the AQ strength
/// slightly toward more aggressive quantization to stabilize the zero/nonzero
/// boundary decisions.
#[inline]
fn boundary_aq_adjust(aq_strength: f32, local_block_row: usize, total_block_rows: usize) -> f32 {
    // Only affect the first and last block row
    if local_block_row == 0 || local_block_row + 1 >= total_block_rows {
        // Nudge toward more quantization (more negative = more aggressive)
        // 0.02 corresponds to ~1.4% more quantization — enough to stabilize
        // boundary coefficients without visible quality loss
        aq_strength - 0.02
    } else {
        aq_strength
    }
}

/// Bias DC quantization at restart boundaries (option D).
/// At the first MCU of a segment, the DC prediction resets to 0, so the full
/// DC value is encoded. Rounding toward a smaller magnitude saves bits.
#[inline]
fn bias_dc_at_restart(block: &mut [i16; DCT_BLOCK_SIZE], is_first_mcu: bool) {
    if !is_first_mcu {
        return;
    }
    // If DC is ±1 from a rounding boundary, bias toward 0
    // This saves 1 category bit on average for the first MCU
    let dc = block[0];
    if dc.unsigned_abs() <= 1 {
        block[0] = 0;
    }
}

/// Compute full-image AQ map and per-segment AQ maps for comparison.
/// Returns (full_map, segmented_map, blocks_w, blocks_h, restart_mcu_rows).
#[cfg(test)]
pub(crate) fn compare_aq_maps(
    rgb_pixels: &[u8],
    width: usize,
    height: usize,
    subsampling: Subsampling,
    y_quant_01: u16,
    restart_mcu_rows: usize,
) -> Result<(Vec<f32>, Vec<f32>, usize, usize, usize)> {
    let (h_samp, v_samp) = match subsampling {
        Subsampling::S444 => (1, 1),
        Subsampling::S422 => (2, 1),
        Subsampling::S420 => (2, 2),
        Subsampling::S440 => (1, 2),
    };
    let mcu_height = v_samp * 8;
    let padded_width = ((width + h_samp * 8 - 1) / (h_samp * 8)) * h_samp * 8;
    let blocks_w = padded_width / 8;
    let mcu_rows = (height + mcu_height - 1) / mcu_height;

    // Color convert full image to Y
    let mut y_plane = vec![0.0f32; height * padded_width];
    let mut cb = vec![0.0f32; height * padded_width];
    let mut cr = vec![0.0f32; height * padded_width];
    crate::color::fast_yuv::rgb_to_ycbcr_strided_fast(
        rgb_pixels, &mut y_plane, &mut cb, &mut cr, width, height, padded_width, 3,
    );
    if width < padded_width {
        for row in 0..height {
            let off = row * padded_width;
            let v = y_plane[off + width - 1];
            for x in width..padded_width { y_plane[off + x] = v; }
        }
    }

    // Full-image AQ
    let full_map = compute_segment_aq(
        &y_plane, width, height, padded_width, subsampling, y_quant_01,
    )?;
    let blocks_h = full_map.len() / blocks_w;

    // Per-segment AQ
    let rows_per_seg = restart_mcu_rows;
    let num_segments = (mcu_rows + rows_per_seg - 1) / rows_per_seg;
    let mut seg_map = vec![0.0f32; full_map.len()];

    for seg_idx in 0..num_segments {
        let mcu_start = seg_idx * rows_per_seg;
        let mcu_count = rows_per_seg.min(mcu_rows - mcu_start);
        let px_start = mcu_start * mcu_height;
        let px_end = ((mcu_start + mcu_count) * mcu_height).min(height);
        let seg_h = px_end - px_start;

        let seg_y = &y_plane[px_start * padded_width..];
        let seg_aq = compute_segment_aq(seg_y, width, seg_h, padded_width, subsampling, y_quant_01)?;

        let block_row_start = mcu_start * v_samp;
        let n = seg_aq.len().min(mcu_count * v_samp * blocks_w);
        let dst = block_row_start * blocks_w;
        seg_map[dst..dst + n].copy_from_slice(&seg_aq[..n]);
    }

    Ok((full_map, seg_map, blocks_w, blocks_h, restart_mcu_rows))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires test image
    fn analyze_aq_boundary_pattern() {
        let img = std::fs::read("/mnt/v/input/BRAG/karwin-luo-4k-420-q85-baseline.jpg").unwrap();
        let decoded = crate::decode::Decoder::new()
            .output_format(crate::decoder::PixelFormat::Rgb)
            .decode(&img, enough::Unstoppable).unwrap();
        let (w, h) = (decoded.width as usize, decoded.height as usize);
        let pixels = decoded.pixels_u8().unwrap();

        let enc = crate::encode::EncoderConfig::ycbcr(85.0, crate::encode::ChromaSubsampling::Quarter)
            .progressive(false);
        let inner = enc.encode_from_bytes(w as u32, h as u32, crate::encode::PixelLayout::Rgb8Srgb).unwrap();
        let y_quant_01 = inner.inner().quant_context().y_quant.values[1];

        let (full, seg, bw, bh, rmr) = compare_aq_maps(
            pixels, w, h, Subsampling::S420, y_quant_01, 4,
        ).unwrap();

        let v_samp = 2;
        eprintln!("Block grid: {}x{}, segments of {} MCU rows ({} block rows)\n",
            bw, bh, rmr, rmr * v_samp);

        eprintln!("{:>4} {:>10} {:>10} {:>10} {:>4}",
            "row", "max_abs", "mean_diff", "rms", "seg");

        let mut boundary_rms_sum = 0.0f64;
        let mut boundary_count = 0;
        let mut interior_rms_sum = 0.0f64;
        let mut interior_count = 0;

        for by in 0..bh {
            let start = by * bw;
            let end = (start + bw).min(full.len()).min(seg.len());
            if start >= end { break; }

            let mut max_abs = 0.0f32;
            let mut sum = 0.0f64;
            let mut sum_sq = 0.0f64;
            for i in start..end {
                let d = seg[i] - full[i];
                max_abs = max_abs.max(d.abs());
                sum += d as f64;
                sum_sq += (d * d) as f64;
            }
            let n = (end - start) as f64;
            let rms = (sum_sq / n).sqrt();

            let mcu_row = by / v_samp;
            let local_block = by % (rmr * v_samp);
            let is_first = local_block < 2;
            let is_last = local_block >= rmr * v_samp - 2;
            let is_boundary = (is_first && mcu_row > 0) || (is_last && mcu_row + 1 < (h + 15) / 16);
            let seg_idx = mcu_row / rmr;

            if is_boundary {
                boundary_rms_sum += sum_sq;
                boundary_count += end - start;
            } else {
                interior_rms_sum += sum_sq;
                interior_count += end - start;
            }

            if rms > 0.0001 || is_boundary {
                eprintln!("{:>4} {:>10.6} {:>10.6} {:>10.6} {:>4}{}",
                    by, max_abs, sum / n, rms, seg_idx,
                    if is_boundary { " <<<" } else { "" });
            }
        }

        eprintln!("\nBoundary blocks RMS: {:.6} ({} blocks)",
            (boundary_rms_sum / boundary_count.max(1) as f64).sqrt(), boundary_count);
        eprintln!("Interior blocks RMS: {:.6} ({} blocks)",
            (interior_rms_sum / interior_count.max(1) as f64).sqrt(), interior_count);
        eprintln!("Blocks with |diff| > 0.01: {}",
            full.iter().zip(seg.iter()).filter(|(f, s)| (*s - *f).abs() > 0.01).count());
        eprintln!("Blocks with |diff| > 0.001: {}",
            full.iter().zip(seg.iter()).filter(|(f, s)| (*s - *f).abs() > 0.001).count());
    }
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
