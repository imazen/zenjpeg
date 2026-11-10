//! Fused parallel encoder with optimized Huffman tables.
//!
//! Single-pass quantization + symbol capture, then cheap Huffman remap:
//!
//! 1. **Quantize (parallel):** Each segment independently runs
//!    color convert → AQ → DCT → quantize → R-D optimize → symbol capture.
//!    Outputs a [`SymbolStream`] (~3MB for 4K) + frequency counters (~8KB).
//!
//! 2. **Merge (serial):** Combine per-segment frequencies → build optimal tables.
//!
//! 3. **Remap (parallel):** Encode each symbol stream with the optimal tables.
//!    Cheap linear scan — no DCT, no quantize, just Huffman lookups.
//!
//! Segments are chunked to approximately physical core count (`rayon_threads / 2`)
//! to avoid SMT contention while still using the global rayon pool.
//!
//! Per-segment R-D optimization drops trailing ±1 AC coefficients where bit savings
//! exceed perceptual cost, typically producing files 0.3-0.5% smaller than sequential.

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
use crate::quant::ZeroBiasParams;
use crate::quant::aq::streaming::StreamingAQ;
use crate::types::Subsampling;

use super::symbol_stream::{SymbolStream, block_to_symbols};

// =============================================================================
// Configuration
// =============================================================================

/// Minimum segment count for parallel overhead to be worthwhile.
const MIN_SEGMENTS: usize = 2;

/// Shared immutable configuration for all parallel segments.
struct Config {
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
    /// Quant table position [1] for AQ initialization.
    y_quant_01: u16,
    /// DC quant value for deringing threshold.
    y_dc_quant: u16,
    deringing: bool,
    aq_enabled: bool,
    /// R-D lambda: higher = more aggressive trailing coefficient dropping.
    /// Scaled by `(85/quality)²` from the base value of 0.001 at Q85.
    rd_lambda: f32,
}

/// Per-segment output: symbol stream + frequencies.
struct SegmentOutput {
    stream: SymbolStream,
    frequencies: HuffmanSymbolFrequencies,
}

// =============================================================================
// Public API
// =============================================================================

/// Fused parallel encode with optimized Huffman tables.
///
/// Returns `(scan_data_with_rst_markers, optimal_tables)`.
/// Caller is responsible for wrapping in JPEG headers.
pub(crate) fn fused_parallel_encode(
    rgb_pixels: &[u8],
    width: u32,
    height: u32,
    subsampling: Subsampling,
    quality: f32,
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
    let cfg = build_config(
        width,
        height,
        subsampling,
        quality,
        y_quant_values,
        cb_quant_values,
        cr_quant_values,
        y_zero_bias,
        cb_zero_bias,
        cr_zero_bias,
        deringing,
        aq_enabled,
    );
    let mcu_rows = (cfg.height + cfg.mcu_height - 1) / cfg.mcu_height;
    let rows_per_seg = restart_mcu_rows.max(2);
    let num_segments = (mcu_rows + rows_per_seg - 1) / rows_per_seg;

    if num_segments < MIN_SEGMENTS {
        return Err(crate::error::Error::unsupported_feature(
            "fused parallel encode needs ≥2 restart segments",
        ));
    }

    // --- Phase 1: parallel quantize + symbol capture ---

    // Chunk segments to ~physical_cores tasks for SMT-aware scheduling.
    // Global rayon pool stays shared; other users aren't starved.
    let max_tasks = (rayon::current_num_threads() / 2).max(2).min(num_segments);
    let chunk_size = (num_segments + max_tasks - 1) / max_tasks;

    // Estimate symbols per segment for pre-allocation
    let est_syms = cfg.mcu_cols * rows_per_seg * (cfg.h_samp * cfg.v_samp + 2) * 5;

    let mut segments: Vec<Option<SegmentOutput>> = (0..num_segments)
        .map(|_| {
            Some(SegmentOutput {
                stream: SymbolStream::with_capacity(est_syms),
                frequencies: new_freqs(),
            })
        })
        .collect();

    let errors: Vec<Option<crate::error::Error>> = segments
        .par_chunks_mut(chunk_size)
        .enumerate()
        .map(|(chunk_idx, chunk)| {
            let base = chunk_idx * chunk_size;
            for (i, slot) in chunk.iter_mut().enumerate() {
                let seg_idx = base + i;
                let mcu_start = seg_idx * rows_per_seg;
                let mcu_count = rows_per_seg.min(mcu_rows - mcu_start);
                let mut out = slot.take().unwrap();
                out.stream.clear();
                if let Err(e) = quantize_segment(rgb_pixels, &cfg, mcu_start, mcu_count, &mut out) {
                    return Some(e);
                }
                *slot = Some(out);
            }
            None
        })
        .collect();

    if let Some(e) = errors.into_iter().flatten().next() {
        return Err(e);
    }

    // --- Phase 2: merge frequencies + build optimal tables ---

    let mut merged = new_freqs();
    for seg in segments.iter().flatten() {
        merged.add(&seg.frequencies);
    }
    let tables = merged.generate_tables()?;

    // --- Phase 3: parallel remap symbols → bitstream ---

    let encoded: Vec<Vec<u8>> = segments
        .par_iter()
        .map(|slot| {
            let seg = slot.as_ref().unwrap();
            seg.stream.encode_to_bytes(
                &tables.dc_luma.table,
                &tables.ac_luma.table,
                &tables.dc_chroma.table,
                &tables.ac_chroma.table,
            )
        })
        .collect();

    // --- Phase 4: concatenate with RST markers ---

    let total_size: usize = encoded.iter().map(|s| s.len() + 2).sum();
    let mut scan_data = Vec::with_capacity(total_size);
    for (i, data) in encoded.iter().enumerate() {
        scan_data.extend_from_slice(data);
        if i + 1 < encoded.len() {
            scan_data.push(0xFF);
            scan_data.push(0xD0 + (i as u8 & 7));
        }
    }

    Ok((scan_data, tables))
}

// =============================================================================
// Per-segment pipeline
// =============================================================================

/// Full pipeline for one segment: color convert → AQ → DCT → quantize → symbols.
fn quantize_segment(
    rgb_pixels: &[u8],
    cfg: &Config,
    mcu_row_start: usize,
    mcu_row_count: usize,
    out: &mut SegmentOutput,
) -> Result<()> {
    let SegDims {
        pixel_h,
        padded_h,
        c_width,
        c_height,
    } = segment_dims(cfg, mcu_row_start, mcu_row_count);

    // Allocate planes (per-segment, not thread-local — clean drop on return)
    let pw = cfg.padded_width;
    let mut y_plane = vec![0.0f32; padded_h * pw];
    let mut cb_plane = vec![0.0f32; c_height * c_width];
    let mut cr_plane = vec![0.0f32; c_height * c_width];

    // Color convert
    color_convert(
        rgb_pixels,
        cfg,
        mcu_row_start,
        pixel_h,
        &mut y_plane,
        &mut cb_plane,
        &mut cr_plane,
        c_width,
    );

    // AQ
    let blocks_w = cfg.blocks_w;
    let seg_block_rows = mcu_row_count * cfg.v_samp;
    let aq = if cfg.aq_enabled {
        compute_aq(
            &y_plane,
            cfg.width,
            pixel_h,
            pw,
            cfg.subsampling,
            cfg.y_quant_01,
        )?
    } else {
        vec![0.0f32; blocks_w * seg_block_rows]
    };
    debug_assert!(
        aq.len() >= blocks_w * seg_block_rows,
        "AQ coverage: got {} values, need {} ({}×{})",
        aq.len(),
        blocks_w * seg_block_rows,
        blocks_w,
        seg_block_rows,
    );

    // DCT + quantize + R-D optimize + symbol capture
    let mut prev_dc_y: i16 = 0;
    let mut prev_dc_cb: i16 = 0;
    let mut prev_dc_cr: i16 = 0;
    let lambda = cfg.rd_lambda;

    for local_mcu_row in 0..mcu_row_count {
        for mcu_col in 0..cfg.mcu_cols {
            // Y blocks
            for vy in 0..cfg.v_samp {
                for hx in 0..cfg.h_samp {
                    let bx = mcu_col * cfg.h_samp + hx;
                    let by = local_mcu_row * cfg.v_samp + vy;

                    let mut block = extract_block_from_strip_wide(&y_plane, bx, by, pw);
                    if cfg.deringing {
                        preprocess_deringing_block(&mut block, cfg.y_dc_quant);
                    }
                    let dct = forward_dct_8x8_wide(&block);

                    let aq_s = aq[by * blocks_w + bx];
                    let aq_scale = if aq_s != 0.0 { 2.0_f32.powf(aq_s) } else { 1.0 };
                    let mut q =
                        cfg.y_quant
                            .quantize_with_zero_bias_zigzag(&dct, &cfg.y_zero_bias, aq_s);
                    drop_trailing_ones(&mut q, &cfg.y_quant.values, aq_scale, lambda);
                    block_to_symbols(&mut out.stream, &q, &mut prev_dc_y, false);
                }
            }

            // Cb
            let cb = extract_block_from_strip_wide(&cb_plane, mcu_col, local_mcu_row, c_width);
            let mut cb_q = cfg.cb_quant.quantize_with_zero_bias_zigzag(
                &forward_dct_8x8_wide(&cb),
                &cfg.cb_zero_bias,
                0.0,
            );
            drop_trailing_ones(&mut cb_q, &cfg.cb_quant.values, 1.0, lambda);
            block_to_symbols(&mut out.stream, &cb_q, &mut prev_dc_cb, true);

            // Cr
            let cr = extract_block_from_strip_wide(&cr_plane, mcu_col, local_mcu_row, c_width);
            let mut cr_q = cfg.cr_quant.quantize_with_zero_bias_zigzag(
                &forward_dct_8x8_wide(&cr),
                &cfg.cr_zero_bias,
                0.0,
            );
            drop_trailing_ones(&mut cr_q, &cfg.cr_quant.values, 1.0, lambda);
            block_to_symbols(&mut out.stream, &cr_q, &mut prev_dc_cr, true);
        }
    }

    // Collect frequencies from the completed symbol stream
    out.frequencies = new_freqs();
    out.stream.collect_frequencies(
        &mut out.frequencies.dc_luma,
        &mut out.frequencies.ac_luma,
        &mut out.frequencies.dc_chroma,
        &mut out.frequencies.ac_chroma,
    );

    Ok(())
}

// =============================================================================
// R-D optimization
// =============================================================================

/// Drop trailing ±1 AC coefficients where bit savings exceed distortion cost.
///
/// Scans from position 63 backward. Only considers |coeff| == 1 (the marginal
/// coefficients). Larger coefficients are always kept — their distortion cost
/// dominates any bit savings.
///
/// Distortion is normalized by `1/Q²` — high-frequency positions (large Q)
/// have low distortion when dropped, matching perceptual expectations.
#[inline]
fn drop_trailing_ones(
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
        return;
    }

    for k in (1..=last_nz).rev() {
        if block[k] == 0 {
            continue;
        }
        if block[k].unsigned_abs() > 1 {
            break; // large coefficient — worth keeping, stop scanning
        }

        // Bit cost of this coefficient: Huffman symbol + category bits
        let cat = crate::entropy::category(block[k]) as u32;
        let bits_saved = 4 + cat; // approximate: symbol ≈ 4 bits + cat magnitude

        // Distortion from dropping: inversely proportional to quantization step.
        // Large Q → coefficient was barely significant → low distortion.
        let q_step = quant_values[k] as f32 * aq_scale;
        let distortion = 1.0 / (q_step * q_step);

        if bits_saved as f32 * lambda > distortion {
            block[k] = 0;
        } else {
            break;
        }
    }
}

// =============================================================================
// Color conversion
// =============================================================================

fn color_convert(
    rgb_pixels: &[u8],
    cfg: &Config,
    mcu_row_start: usize,
    seg_pixel_h: usize,
    y_plane: &mut [f32],
    cb_plane: &mut [f32],
    cr_plane: &mut [f32],
    c_width: usize,
) {
    let px_start = mcu_row_start * cfg.mcu_height;
    let bpp = 3;
    let rgb = &rgb_pixels[px_start * cfg.width * bpp..];
    let rgb_len = seg_pixel_h * cfg.width * bpp;

    if cfg.h_samp == 2 && cfg.v_samp == 2 {
        // 4:2:0: fused color convert + box downsample
        let mut ty = vec![0u8; seg_pixel_h * cfg.padded_width];
        let mut tcb = vec![0u8; seg_pixel_h * cfg.padded_width];
        let mut tcr = vec![0u8; seg_pixel_h * cfg.padded_width];
        crate::color::fast_yuv::rgb_to_ycbcr_420_reuse(
            &rgb[..rgb_len],
            y_plane,
            cb_plane,
            cr_plane,
            &mut ty,
            &mut tcb,
            &mut tcr,
            cfg.width,
            seg_pixel_h,
            cfg.padded_width,
            bpp,
        );
    } else if cfg.h_samp == 1 && cfg.v_samp == 1 {
        crate::color::fast_yuv::rgb_to_ycbcr_strided_fast(
            &rgb[..rgb_len],
            y_plane,
            cb_plane,
            cr_plane,
            cfg.width,
            seg_pixel_h,
            cfg.padded_width,
            bpp,
        );
    } else {
        // 422/440 conversion first produces full-resolution packed chroma.
        // The segment output planes have subsampled geometry, so they cannot
        // serve as the converter's full-resolution scratch buffers.
        let mut cb_full = vec![0.; cfg.width * seg_pixel_h];
        let mut cr_full = vec![0.; cfg.width * seg_pixel_h];
        crate::color::fast_yuv::rgb_to_ycbcr_strided_fast(
            &rgb[..rgb_len],
            y_plane,
            &mut cb_full,
            &mut cr_full,
            cfg.width,
            seg_pixel_h,
            cfg.padded_width,
            bpp,
        );
        for (src, dst) in [(&cb_full, &mut *cb_plane), (&cr_full, &mut *cr_plane)] {
            if cfg.h_samp == 2 {
                crate::encode_simd::downsample_2x1_simd_inplace(src, cfg.width, seg_pixel_h, dst);
            } else {
                crate::encode_simd::downsample_1x2_simd_inplace(src, cfg.width, seg_pixel_h, dst);
            }
        }
    }

    // Chroma converters/downsamplers produce packed rows. DCT uses padded
    // chroma stride. Rearrange from the end to preserve overlapping rows.
    let visible_c_width = cfg.width.div_ceil(cfg.h_samp);
    let visible_c_height = seg_pixel_h.div_ceil(cfg.v_samp);
    for plane in [&mut *cb_plane, &mut *cr_plane] {
        for row in (0..visible_c_height).rev() {
            let src = row * visible_c_width;
            let dst = row * c_width;
            plane.copy_within(src..src + visible_c_width, dst);
            let edge = plane[dst + visible_c_width - 1];
            plane[dst + visible_c_width..dst + c_width].fill(edge);
        }
        let last = (visible_c_height - 1) * c_width;
        for row in visible_c_height..plane.len() / c_width {
            plane.copy_within(last..last + c_width, row * c_width);
        }
    }

    // Edge-replicate Y to padded width
    if cfg.width < cfg.padded_width {
        for row in 0..seg_pixel_h {
            let off = row * cfg.padded_width;
            let v = y_plane[off + cfg.width - 1];
            for x in cfg.width..cfg.padded_width {
                y_plane[off + x] = v;
            }
        }
    }
    let last = (seg_pixel_h - 1) * cfg.padded_width;
    for row in seg_pixel_h..y_plane.len() / cfg.padded_width {
        y_plane.copy_within(last..last + cfg.padded_width, row * cfg.padded_width);
    }
}

// =============================================================================
// AQ
// =============================================================================

fn compute_aq(
    y_plane: &[f32],
    width: usize,
    seg_h: usize,
    y_stride: usize,
    subsampling: Subsampling,
    y_quant_01: u16,
) -> Result<Vec<f32>> {
    let layout = LayoutParams::new(width, seg_h, subsampling, false);
    let mut aq = StreamingAQ::new(&layout, y_quant_01, true)?;
    let imcu_h: usize = match subsampling {
        Subsampling::S420 | Subsampling::S440 => 16,
        _ => 8,
    };
    let mut out = Vec::new();
    for strip_y in (0..seg_h).step_by(imcu_h) {
        let h = imcu_h.min(seg_h - strip_y);
        let data = &y_plane[strip_y * y_stride..(strip_y + h) * y_stride];
        if let Some(s) = aq.process_y_strip(data, strip_y, h) {
            out.extend_from_slice(s);
        }
    }
    if let Some(s) = aq.flush() {
        out.extend_from_slice(s);
    }
    pad_aq_grid(
        out,
        &layout,
        y_stride / 8,
        seg_h.div_ceil(imcu_h) * (imcu_h / 8),
    )
}

/// Extend a visible AQ grid to the segment's MCU block grid.
fn pad_aq_grid(
    out: Vec<f32>,
    layout: &LayoutParams,
    padded_blocks_w: usize,
    padded_blocks_h: usize,
) -> Result<Vec<f32>> {
    // AQ describes visible 8x8 blocks. Segment entropy also visits dummy
    // blocks needed to complete an MCU, at padded rather than visible stride.
    // Clamp the AQ grid in both axes just like the corresponding pixel plane.
    if out.len() != layout.blocks_w * layout.blocks_h {
        return Err(crate::error::Error::internal(
            "incomplete fused segment AQ grid",
        ));
    }
    if padded_blocks_w == layout.blocks_w && padded_blocks_h == layout.blocks_h {
        return Ok(out);
    }
    let mut padded = Vec::with_capacity(padded_blocks_w * padded_blocks_h);
    for by in 0..padded_blocks_h {
        for bx in 0..padded_blocks_w {
            padded.push(
                out[by.min(layout.blocks_h - 1) * layout.blocks_w + bx.min(layout.blocks_w - 1)],
            );
        }
    }
    Ok(padded)
}

// =============================================================================
// Helpers
// =============================================================================

struct SegDims {
    pixel_h: usize,
    padded_h: usize,
    c_width: usize,
    c_height: usize,
}

fn segment_dims(cfg: &Config, mcu_row_start: usize, mcu_row_count: usize) -> SegDims {
    let px_start = mcu_row_start * cfg.mcu_height;
    let px_end = ((mcu_row_start + mcu_row_count) * cfg.mcu_height).min(cfg.height);
    SegDims {
        pixel_h: px_end - px_start,
        padded_h: mcu_row_count * cfg.mcu_height,
        c_width: (cfg.padded_width + cfg.h_samp - 1) / cfg.h_samp,
        c_height: (mcu_row_count * cfg.mcu_height + cfg.v_samp - 1) / cfg.v_samp,
    }
}

fn build_config(
    width: u32,
    height: u32,
    subsampling: Subsampling,
    quality: f32,
    y_qv: &[u16; 64],
    cb_qv: &[u16; 64],
    cr_qv: &[u16; 64],
    y_zb: &ZeroBiasParams,
    cb_zb: &ZeroBiasParams,
    cr_zb: &ZeroBiasParams,
    deringing: bool,
    aq_enabled: bool,
) -> Config {
    let (w, h) = (width as usize, height as usize);
    let (hs, vs) = match subsampling {
        Subsampling::S444 => (1, 1),
        Subsampling::S422 => (2, 1),
        Subsampling::S420 => (2, 2),
        Subsampling::S440 => (1, 2),
    };
    // StreamingAQ reads the layout's padded row stride (also 16-aligned
    // for 440). Keep plane storage consistent with that contract while
    // retaining the JPEG sampling grid's actual MCU column count.
    let pw = LayoutParams::new(w, h, subsampling, false).padded_width;

    // Lambda scales inversely with quality squared: lower quality → more dropping.
    // Base: 0.001 at Q85 (tuned on karwin-luo 4K, -0.53% size, +2 max pixel diff).
    let rd_lambda = 0.001 * (85.0 / quality.max(1.0)).powi(2);

    Config {
        width: w,
        height: h,
        padded_width: pw,
        blocks_w: pw / 8,
        mcu_cols: w.div_ceil(hs * 8),
        mcu_height: vs * 8,
        h_samp: hs,
        v_samp: vs,
        subsampling,
        y_quant: QuantTableSimd::from_values(y_qv),
        cb_quant: QuantTableSimd::from_values(cb_qv),
        cr_quant: QuantTableSimd::from_values(cr_qv),
        y_zero_bias: ZeroBiasSimd::from_params(y_zb),
        cb_zero_bias: ZeroBiasSimd::from_params(cb_zb),
        cr_zero_bias: ZeroBiasSimd::from_params(cr_zb),
        y_quant_01: y_qv[1],
        y_dc_quant: y_qv[0],
        deringing,
        aq_enabled,
        rd_lambda,
    }
}

fn new_freqs() -> HuffmanSymbolFrequencies {
    HuffmanSymbolFrequencies {
        dc_luma: FrequencyCounter::new(),
        ac_luma: FrequencyCounter::new(),
        dc_chroma: FrequencyCounter::new(),
        ac_chroma: FrequencyCounter::new(),
    }
}

#[cfg(test)]
mod padding_tests {
    use super::*;

    fn check(sub: Subsampling) {
        let _dispatch = archmage::testing::lock_token_testing();
        let q = [16; 64];
        let zb = ZeroBiasParams::for_ycbcr(1.0, 0);
        let mut cells = 0;
        for w in (1..=33).chain([63, 64, 65, 127, 128, 129]) {
            for tail in 1..=16 {
                for start in [0, 2, 4, 8] {
                    let mut cfg =
                        build_config(w, 1, sub, 40.0, &q, &q, &q, &zb, &zb, &zb, false, false);
                    cfg.height = start * cfg.mcu_height + tail;
                    let count = tail.div_ceil(cfg.mcu_height);
                    let dims = segment_dims(&cfg, start, count);
                    let mut rgb = vec![0; w as usize * cfg.height * 3];
                    for (i, v) in rgb.iter_mut().enumerate() {
                        *v = ((i * 37 + i / 7 * 23) % 251) as u8;
                    }
                    let w = w as usize;
                    let h = dims.pixel_h;
                    let cw = w.div_ceil(cfg.h_samp);
                    let ch = h.div_ceil(cfg.v_samp);
                    let input = &rgb[start * cfg.mcu_height * w * 3..];
                    let mut y = vec![-9876.; dims.padded_h * cfg.padded_width];
                    let mut cb = vec![-9876.; dims.c_width * dims.c_height];
                    let mut cr = cb.clone();
                    color_convert(&rgb, &cfg, start, h, &mut y, &mut cb, &mut cr, dims.c_width);
                    // Conversion oracle has PACKED output; scalar coordinate clamping
                    // independently supplies every right/bottom padding sample.
                    let mut ry = vec![0.; w * h];
                    let mut rc = vec![0.; cw * ch];
                    let mut rr = rc.clone();
                    if sub == Subsampling::S420 {
                        let mut ty = vec![0; w * h];
                        let mut tc = vec![0; cw * ch];
                        let mut tr = tc.clone();
                        crate::color::fast_yuv::rgb_to_ycbcr_420_reuse(
                            input, &mut ry, &mut rc, &mut rr, &mut ty, &mut tc, &mut tr, w, h, w, 3,
                        );
                    } else {
                        let mut fc = vec![0.; w * h];
                        let mut fr = fc.clone();
                        crate::color::fast_yuv::rgb_to_ycbcr_strided_fast(
                            input, &mut ry, &mut fc, &mut fr, w, h, w, 3,
                        );
                        for (dst, src) in [(&mut rc, &fc), (&mut rr, &fr)] {
                            for yy in 0..ch {
                                for xx in 0..cw {
                                    let mut sum = 0.;
                                    for dy in 0..cfg.v_samp {
                                        for dx in 0..cfg.h_samp {
                                            sum += src[(yy * cfg.v_samp + dy).min(h - 1) * w
                                                + (xx * cfg.h_samp + dx).min(w - 1)];
                                        }
                                    }
                                    dst[yy * cw + xx] = sum / (cfg.h_samp * cfg.v_samp) as f32;
                                }
                            }
                        }
                    }
                    for (name, actual, stride, rows, expected, vw, vh) in [
                        ("Y", &y, cfg.padded_width, dims.padded_h, &ry, w, h),
                        ("Cb", &cb, dims.c_width, dims.c_height, &rc, cw, ch),
                        ("Cr", &cr, dims.c_width, dims.c_height, &rr, cw, ch),
                    ] {
                        for yy in 0..rows {
                            for xx in 0..stride {
                                assert_eq!(
                                    actual[yy * stride + xx],
                                    expected[yy.min(vh - 1) * vw + xx.min(vw - 1)],
                                    "{sub:?} {w}x{} start {start} {name} ({xx},{yy})",
                                    cfg.height
                                );
                            }
                        }
                    }
                    cells += 1;
                }
            }
        }
        eprintln!("fused parallel {sub:?}: {cells} segment geometries, 3 planes each");
    }

    #[test]
    fn fused_420_edges() {
        check(Subsampling::S420);
    }
    #[test]
    fn fused_444_edges() {
        check(Subsampling::S444);
    }
    #[test]
    fn fused_422_edges() {
        check(Subsampling::S422);
    }
    #[test]
    fn fused_440_edges() {
        check(Subsampling::S440);
    }

    #[test]
    fn fused_packed_chroma_stride_witness() {
        let q = [16; 64];
        let zb = ZeroBiasParams::for_ycbcr(1., 0);
        let cfg = build_config(
            9,
            16,
            Subsampling::S420,
            40.,
            &q,
            &q,
            &q,
            &zb,
            &zb,
            &zb,
            false,
            false,
        );
        let d = segment_dims(&cfg, 0, 1);
        let mut y = vec![-9876.; d.padded_h * cfg.padded_width];
        let mut cb = vec![-9876.; d.c_width * d.c_height];
        let mut cr = cb.clone();
        color_convert(
            &[81; 9 * 16 * 3],
            &cfg,
            0,
            16,
            &mut y,
            &mut cb,
            &mut cr,
            d.c_width,
        );
        assert!(cb.iter().all(|&v| v == 128.));
        assert!(cr.iter().all(|&v| v == 128.));
    }

    fn full_chroma_storage_witness(sub: Subsampling) {
        let q = [16; 64];
        let zb = ZeroBiasParams::for_ycbcr(1., 0);
        let cfg = build_config(32, 16, sub, 40., &q, &q, &q, &zb, &zb, &zb, false, false);
        let d = segment_dims(&cfg, 0, 16 / cfg.mcu_height);
        let mut y = vec![-9876.; d.padded_h * cfg.padded_width];
        let mut cb = vec![-9876.; d.c_width * d.c_height];
        let mut cr = cb.clone();
        color_convert(
            &[81; 32 * 16 * 3],
            &cfg,
            0,
            16,
            &mut y,
            &mut cb,
            &mut cr,
            d.c_width,
        );
        assert!(cb.iter().all(|&v| v == 128.));
        assert!(cr.iter().all(|&v| v == 128.));
    }
    #[test]
    fn fused_422_full_resolution_storage_witness() {
        full_chroma_storage_witness(Subsampling::S422);
    }
    #[test]
    fn fused_440_full_resolution_storage_witness() {
        full_chroma_storage_witness(Subsampling::S440);
    }

    #[test]
    fn fused_440_aq_input_stride_witness() {
        let q = [16; 64];
        let zb = ZeroBiasParams::for_ycbcr(1., 0);
        let cfg = build_config(
            17,
            1,
            Subsampling::S440,
            40.,
            &q,
            &q,
            &q,
            &zb,
            &zb,
            &zb,
            false,
            false,
        );
        assert_eq!(
            cfg.padded_width,
            LayoutParams::new(17, 1, Subsampling::S440, false).padded_width
        );
        compute_aq(
            &vec![81.; cfg.padded_width * 16],
            17,
            1,
            cfg.padded_width,
            Subsampling::S440,
            16,
        )
        .unwrap();
    }

    #[test]
    fn fused_aq_padding_coordinate_oracle() {
        // Token permutations are process-wide. Both AQ passes must see the
        // same dispatch while other library tests exercise fallback tiers.
        let _dispatch = archmage::testing::lock_token_testing();
        let mut cells = 0;
        for sub in [
            Subsampling::S444,
            Subsampling::S422,
            Subsampling::S420,
            Subsampling::S440,
        ] {
            for w in 1..=33 {
                for h in 1..=33 {
                    let l = LayoutParams::new(w, h, sub, false);
                    let pw = l.padded_width;
                    let ph = h.div_ceil(l.strip_height) * l.strip_height;
                    let y: Vec<f32> = (0..pw * ph)
                        .map(|i| ((i * 37 + i / 11 * 19) % 251) as f32)
                        .collect();
                    let actual = compute_aq(&y, w, h, pw, sub, 16).unwrap();
                    let mut reference = StreamingAQ::new(&l, 16, true).unwrap();
                    let mut visible = Vec::new();
                    for start in (0..h).step_by(l.strip_height) {
                        let rows = l.strip_height.min(h - start);
                        if let Some(s) = reference.process_y_strip(
                            &y[start * pw..(start + rows) * pw],
                            start,
                            rows,
                        ) {
                            visible.extend_from_slice(s);
                        }
                    }
                    visible.extend_from_slice(
                        reference
                            .flush()
                            .expect("nonempty image leaves a final pending iMCU"),
                    );
                    assert_eq!(visible.len(), l.blocks_w * l.blocks_h);
                    assert_eq!(
                        actual.len(),
                        pw / 8 * ph / 8,
                        "{sub:?} {w}x{h} padded AQ size"
                    );
                    for by in 0..ph / 8 {
                        for bx in 0..pw / 8 {
                            assert_eq!(
                                actual[by * (pw / 8) + bx],
                                visible
                                    [by.min(l.blocks_h - 1) * l.blocks_w + bx.min(l.blocks_w - 1)],
                                "{sub:?} {w}x{h} AQ ({bx},{by})"
                            );
                        }
                    }
                    cells += 1;
                }
            }
        }
        eprintln!("fused AQ coordinate oracle: {cells} geometries");
    }

    #[test]
    fn malformed_aq_grid_is_rejected_before_padding() {
        let layout = LayoutParams::new(17, 17, Subsampling::S420, false);
        assert_eq!(layout.blocks_w * layout.blocks_h, 9);
        for actual in [0, 8, 10] {
            let err = pad_aq_grid(vec![0.; actual], &layout, 4, 4).unwrap_err();
            assert!(err.to_string().contains("incomplete fused segment AQ grid"));
        }
    }
}
