//! Empirical progressive & sequential trellis rate-distortion optimization.
//!
//! Internal (`pub(crate)`) module powering `HybridBaseline`, `HybridProgressive`,
//! `HybridMaxCompression`, and `auto_optimize(true)` with zero public API surface
//! changes, zero `unsafe`, fallible allocations, and deterministic `1T == MT`
//! parallel execution.
//!
//! Techniques implemented here:
//! - **Empirical sequential rate pricing (`T3a`)**: image-specific Huffman code
//!   lengths learned from Pass-1 quantized blocks for baseline/sequential encodes.
//! - **Exact progressive band + SA bitplane trellis (`T3b`)**: image-specific
//!   per-scan Huffman tables + amortized EOBRUN costs directly from
//!   `ProgressiveTokenBuffer`, with exact multi-stage Viterbi DP matching both
//!   first-pass spectral bands (1-, 2-, and 3-band scripts) and full-range
//!   successive-approximation refinement scans (`ah -> al`).
//! - **Cached scan-matched search**: when `ScanStrategy::Search` is active,
//!   selects the winning progressive scan script once, prices Pass-2 trellis
//!   against that exact script, and returns the winning script for emission.
//! - **Empirical DC trellis (`T4`)**: row-by-row DC trellis using empirical DC
//!   Huffman tables, correct XYB component strides (`b_blocks_w`), and extended
//!   DC clamp (`16383`) for XYB SOF1.
//! - **4:4:4 step-ratio frequency weighting**: `step_ratio^(2 * 0.15)` relative
//!   distortion weighting on YCbCr 4:4:4 (`Galaxy4594/jpegli-redux`, BSD-3-Clause).

use alloc::vec::Vec;

#[cfg(target_arch = "x86_64")]
use archmage::SimdToken;

use crate::decode::idct::inverse_dct_8x8;
#[cfg(target_arch = "x86_64")]
use crate::decode::idct::inverse_dct_8x8_into_with_token;
use crate::encode::config::{ComputedConfig, ProgressiveScan};
use crate::encode::encoder_types::ScanStrategy;
use crate::encode::layout::LayoutParams;
use crate::encode::strip::StripProcessorOutput;
use crate::encode::trellis::ac::jpeg_nbits;
use crate::encode::trellis::compat::{TrellisConfig, TrellisSpeedMode};
use crate::encode::trellis::hybrid::{StandardRateTables, dct_f32_to_i32};
use crate::encode::trellis::rate::RateTable;
use crate::error::{Error, Result};
use crate::foundation::consts::{DCT_BLOCK_SIZE, JPEG_NATURAL_ORDER, JPEG_ZIGZAG_ORDER};
use crate::foundation::simd_types::Block8x8f;
use crate::huffman::optimize::{ContextConfig, FrequencyCounter, ProgressiveTokenBuffer};
use crate::quant::QuantTable;
use crate::types::{JpegMode, Subsampling};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Cap on retained raw f32 DCT memory before falling back to single-pass legacy
/// trellis (128 MiB ≈ 32 MPx 4:2:0 or 16 MPx 4:4:4). Raw DCT buffers are freed
/// immediately after Pass-2 trellis finishes and before progressive tokenization,
/// so peak RSS stays bounded.
pub(crate) const MAX_RDO_RAW_DCT_BYTES: usize = 128 * 1024 * 1024;

/// Default fallback code length (bits) for symbols unseen in the Pass-1 histogram.
const UNSEEN_SYMBOL_BITS: u8 = 16;

// XYB neutral-gray level-shifted offsets in [-128, 127]:
// X_neutral = (0.015386134 * 22.995788804) * 255.0 - 128.0
// B_neutral = (0.27770459 * 1.502141333) * 255.0 - 128.0
const XYB_X_NEUTRAL: f32 = -37.777_46;
const XYB_B_NEUTRAL: f32 = -21.627_144;

// Validated perceptual seam/activity RDO parameters (`HybridMaxCompression`):
// DampedNorm(0.65, -0.2625) + T2_imij(0.18, sat=0.35) + T1(0.15, seam=1.5, chroma=0.5, unified)
const PERCEPTUAL_NORM_SCALE: f32 = 0.65;
const PERCEPTUAL_SHIFT_S1: f32 = -0.2625;
const T2_STRENGTH: f32 = 0.18;
const T2_SAT_WEIGHT: f32 = 0.35;
const T1_WEIGHT: f32 = 0.15;
const T1_SEAM_WEIGHT: f64 = 1.5;
const T1_CHROMA_WEIGHT: f64 = 0.5;

/// Stripe height (in 8×8 block rows) for O(width) stripe-parallel T1 seam/visibility
/// evaluation. Keeps per-worker IDCT edge scratch at O(blocks_w) while preserving
/// bit-exact `1T == MT` IEEE-754 summation order.
const T1_STRIPE_ROWS: usize = 16;

/// Internal trellis rate-distortion mode wired by optimization presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum TrellisRdoMode {
    /// Legacy single-pass Annex-K trellis executed inside `StripProcessor`.
    /// Preserved for `Mozjpeg*` presets and explicit `.trellis(...)` callers.
    #[default]
    Legacy,
    /// Fast empirical RDO: SIMD zero-bias pilot in `StripProcessor`, followed by
    /// one pass of exact empirical progressive/sequential trellis + empirical DC.
    /// Used by `HybridProgressive` and `auto_optimize(true)`.
    EmpiricalFast,
    /// Two-pass empirical RDO: Annex-K trellis pilot -> exact empirical
    /// progressive/sequential trellis + cached scan-matched search + empirical DC.
    /// Used by `HybridBaseline`.
    EmpiricalTwoPass,
    /// Full perceptual multi-pass RDO: damped AC-norm lambda base + open-loop
    /// 3×3/chroma-saturation activity prior (`T2`) + closed-loop 2D unfiltered-seam
    /// & visibility-weighted error feedback (`T1`, streamed with O(width) sliding
    /// IDCT) + exact empirical progressive/sequential trellis (`T3b`/`T3a`) + `R8`
    /// cached scan-matched search + empirical DC (`T4`).
    /// Used by `HybridMaxCompression`.
    PerceptualFull,
}

/// Base block-level lambda formula before per-block perceptual multipliers.
#[derive(Debug, Clone, Copy, PartialEq)]
enum LambdaBaseMode {
    MozjpegNorm,
    DampedNorm { norm_scale: f32, shift_s1: f32 },
}

/// Fallible value-initialized `Vec` allocation.
fn try_vec<T: Clone>(len: usize, val: T, ctx: &'static str) -> Result<Vec<T>> {
    let mut v = Vec::new();
    v.try_reserve_exact(len)
        .map_err(|_| Error::allocation_failed(len * core::mem::size_of::<T>(), ctx))?;
    v.resize(len, val);
    Ok(v)
}

/// Fallible empty `Vec` with exact capacity.
fn try_vec_cap<T>(cap: usize, ctx: &'static str) -> Result<Vec<T>> {
    let mut v = Vec::new();
    v.try_reserve_exact(cap)
        .map_err(|_| Error::allocation_failed(cap * core::mem::size_of::<T>(), ctx))?;
    Ok(v)
}

// ============================================================================
// Fast SIMD IDCT helpers (zero-allocation stack blocks)
// ============================================================================

#[derive(Copy, Clone, Default)]
struct IdctToken {
    #[cfg(target_arch = "x86_64")]
    v3: Option<archmage::X64V3Token>,
}

impl IdctToken {
    #[inline]
    fn summon() -> Self {
        Self {
            #[cfg(target_arch = "x86_64")]
            v3: archmage::X64V3Token::summon(),
        }
    }
}

/// Reconstruct 8×8 level-shifted spatial pixels `[-128, 127]` from raw f32 DCT.
#[inline]
fn idct_raw_block(tok: IdctToken, dct: &Block8x8f) -> [f32; DCT_BLOCK_SIZE] {
    let mut natural = [0.0f32; DCT_BLOCK_SIZE];
    for r in 0..8 {
        let base = r * 8;
        for c in 0..8 {
            natural[base + c] = dct.rows[r][c] * 8.0;
        }
    }
    let mut out = [0.0f32; DCT_BLOCK_SIZE];
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = tok.v3 {
        inverse_dct_8x8_into_with_token(t, &natural, &mut out);
        return out;
    }
    let _ = tok;
    out = inverse_dct_8x8(&natural);
    out
}

/// Reconstruct 8×8 level-shifted spatial pixels `[-128, 127]` from zigzag i16
/// quantized coefficients and natural-order quantization table.
#[inline]
fn idct_quantized_zigzag(
    tok: IdctToken,
    zigzag: &[i16; DCT_BLOCK_SIZE],
    qvals_natural: &[u16; DCT_BLOCK_SIZE],
) -> [f32; DCT_BLOCK_SIZE] {
    let mut any_ac = false;
    for &c in &zigzag[1..] {
        if c != 0 {
            any_ac = true;
            break;
        }
    }
    if !any_ac {
        let dc = (zigzag[0] as f32) * (qvals_natural[0] as f32) * 0.125;
        return [dc; DCT_BLOCK_SIZE];
    }

    let mut natural = [0.0f32; DCT_BLOCK_SIZE];
    for (nat_idx, &zz_idx) in JPEG_ZIGZAG_ORDER.iter().enumerate() {
        let q = zigzag[zz_idx as usize];
        if q != 0 {
            natural[nat_idx] = (q as f32) * (qvals_natural[nat_idx] as f32);
        }
    }
    let mut out = [0.0f32; DCT_BLOCK_SIZE];
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = tok.v3 {
        inverse_dct_8x8_into_with_token(t, &natural, &mut out);
        return out;
    }
    let _ = tok;
    out = inverse_dct_8x8(&natural);
    out
}

/// Build the per-coefficient distortion weight table (`w[k] / q[k]^2`).
///
/// When `freq_power != 0.0`, applies the `step_ratio^(2 * freq_power)` relative
/// frequency tilt (clamped to `[0.60, 1.65]`, adapted from `jpegli-redux`
/// BSD-3-Clause), which consistently improves all 4 perceptual metrics on YCbCr 4:4:4.
fn build_lambda_tbl(
    qvals_natural: &[u16; DCT_BLOCK_SIZE],
    freq_power: f32,
) -> [f32; DCT_BLOCK_SIZE] {
    let mut tbl = [0.0f32; DCT_BLOCK_SIZE];
    if freq_power == 0.0 {
        for k in 0..DCT_BLOCK_SIZE {
            let q = qvals_natural[k].max(1) as f32;
            tbl[k] = 1.0 / (q * q);
        }
        return tbl;
    }

    let mut mean_step = 0.0f32;
    for &q in &qvals_natural[1..] {
        mean_step += q.max(1) as f32;
    }
    mean_step = (mean_step / 63.0).max(1.0);

    for k in 0..DCT_BLOCK_SIZE {
        let q = qvals_natural[k].max(1) as f32;
        let w = if k > 0 {
            let step_ratio = (q / mean_step).clamp(0.60, 1.65);
            step_ratio.powf(2.0 * freq_power)
        } else {
            1.0
        };
        tbl[k] = w / (q * q);
    }
    tbl
}

/// Run empirical progressive/sequential trellis RDO over retained raw DCT blocks.
///
/// Returns `Some(winning_scans)` when `ScanStrategy::Search` selected and priced
/// a progressive scan script so the caller can emit it directly without repeating
/// the scan search.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_trellis_rdo(
    config: &ComputedConfig,
    y_quant: &QuantTable,
    cb_quant: &QuantTable,
    cr_quant: &QuantTable,
    layout: &LayoutParams,
    strip: &mut StripProcessorOutput,
) -> Result<Option<Vec<ProgressiveScan>>> {
    if config.trellis_rdo == TrellisRdoMode::Legacy || strip.y_raw_dct.is_empty() {
        return Ok(None);
    }
    let Some(trellis_cfg) = config.trellis else {
        return Ok(None);
    };
    if !trellis_cfg.is_enabled() {
        return Ok(None);
    }

    let is_color = !config.writes_gray() && !strip.cb_raw_dct.is_empty();
    let use_xyb = config.use_xyb;

    #[cfg(feature = "parallel")]
    let use_parallel = config.parallel;
    #[cfg(not(feature = "parallel"))]
    let use_parallel = false;

    let n_y = strip.y_raw_dct.len();
    let n_cb = strip.cb_raw_dct.len();
    let n_cr = strip.cr_raw_dct.len();

    let (y_bw, y_bh) = (layout.y_blocks_w, layout.y_blocks_h);
    let (cb_bw, cb_bh) = if use_xyb {
        (layout.y_blocks_w, layout.y_blocks_h)
    } else {
        (layout.c_blocks_w, layout.c_blocks_h)
    };
    let (cr_bw, cr_bh) = if use_xyb {
        (layout.b_blocks_w, layout.b_blocks_h)
    } else {
        (layout.c_blocks_w, layout.c_blocks_h)
    };

    // Ensure quantized output buffers are sized when StripProcessor skipped initial quantize.
    if strip.y_blocks.len() != n_y {
        strip.y_blocks = try_vec(n_y, [0i16; DCT_BLOCK_SIZE], "rdo y_blocks")?;
    }
    if is_color && strip.cb_blocks.len() != n_cb {
        strip.cb_blocks = try_vec(n_cb, [0i16; DCT_BLOCK_SIZE], "rdo cb_blocks")?;
    }
    if is_color && strip.cr_blocks.len() != n_cr {
        strip.cr_blocks = try_vec(n_cr, [0i16; DCT_BLOCK_SIZE], "rdo cr_blocks")?;
    }

    // Channel roles & DC range: XYB stores (X, Y, B) in (y, cb, cr) with SOF1 DC range.
    let y_is_luma = !use_xyb;
    let cb_is_luma = use_xyb;
    let max_dc_val = if use_xyb { 16383 } else { 1023 };

    let perceptual = config.trellis_rdo == TrellisRdoMode::PerceptualFull;
    let base_mode = if perceptual {
        LambdaBaseMode::DampedNorm {
            norm_scale: PERCEPTUAL_NORM_SCALE,
            shift_s1: PERCEPTUAL_SHIFT_S1,
        }
    } else {
        LambdaBaseMode::MozjpegNorm
    };

    // Step-ratio frequency weighting (0.15) generalizes strongly across all 4 metrics
    // on YCbCr 4:4:4; keep neutral 1/q^2 on 4:2:0, XYB, RGB, and grayscale.
    let step_power =
        if !use_xyb && !config.use_rgb && is_color && config.subsampling == Subsampling::S444 {
            0.15f32
        } else {
            0.0f32
        };
    let y_lambda_tbl = build_lambda_tbl(&y_quant.values, step_power);
    let cb_lambda_tbl = build_lambda_tbl(&cb_quant.values, step_power);
    let cr_lambda_tbl = build_lambda_tbl(&cr_quant.values, step_power);

    // ------------------------------------------------------------------------
    // Optional Pass A (PerceptualFull): O(width) open-loop T2_imij lambda maps
    // ------------------------------------------------------------------------
    let (mut luma_lambda_map, mut cb_lambda_map, mut cr_lambda_map) = if perceptual {
        let luma_map = build_t2_imij_lambda_map_streaming(
            use_xyb,
            is_color,
            y_bw,
            y_bh,
            cb_bw,
            cb_bh,
            cr_bw,
            cr_bh,
            &strip.y_raw_dct,
            &strip.cb_raw_dct,
            &strip.cr_raw_dct,
            y_quant,
            cb_quant,
            cr_quant,
            T2_STRENGTH,
            T2_SAT_WEIGHT,
            true,
            use_parallel,
        )?;
        let cb_map = if is_color {
            map_luma_to_plane(&luma_map, y_bw, y_bh, cb_bw, cb_bh)?
        } else {
            Vec::new()
        };
        let cr_map = if is_color {
            map_luma_to_plane(&luma_map, y_bw, y_bh, cr_bw, cr_bh)?
        } else {
            Vec::new()
        };
        (Some(luma_map), Some(cb_map), Some(cr_map))
    } else {
        (None, None, None)
    };

    // ------------------------------------------------------------------------
    // Pass 1 (pilot): EmpiricalFast reuses StripProcessor's SIMD ZeroBias pilot;
    // EmpiricalTwoPass & PerceptualFull run Annex-K trellis as Pass 1.
    // ------------------------------------------------------------------------
    if matches!(
        config.trellis_rdo,
        TrellisRdoMode::EmpiricalTwoPass | TrellisRdoMode::PerceptualFull
    ) {
        let std_tables = StandardRateTables::new();
        quantize_plane_sequential_trellis(
            &strip.y_raw_dct,
            &mut strip.y_blocks,
            &y_quant.values,
            &y_lambda_tbl,
            if y_is_luma {
                &std_tables.luma_ac
            } else {
                &std_tables.chroma_ac
            },
            &trellis_cfg,
            base_mode,
            luma_lambda_map.as_deref(),
            max_dc_val,
            use_parallel,
        );
        if is_color {
            quantize_plane_sequential_trellis(
                &strip.cb_raw_dct,
                &mut strip.cb_blocks,
                &cb_quant.values,
                &cb_lambda_tbl,
                if cb_is_luma {
                    &std_tables.luma_ac
                } else {
                    &std_tables.chroma_ac
                },
                &trellis_cfg,
                base_mode,
                cb_lambda_map.as_deref(),
                max_dc_val,
                use_parallel,
            );
            quantize_plane_sequential_trellis(
                &strip.cr_raw_dct,
                &mut strip.cr_blocks,
                &cr_quant.values,
                &cr_lambda_tbl,
                &std_tables.chroma_ac,
                &trellis_cfg,
                base_mode,
                cr_lambda_map.as_deref(),
                max_dc_val,
                use_parallel,
            );
        }
    }

    // ------------------------------------------------------------------------
    // Optional Pass B (PerceptualFull): O(width) closed-loop T1 seam/visibility
    // ------------------------------------------------------------------------
    if perceptual
        && let (Some(ym), Some(cbm), Some(crm)) = (
            luma_lambda_map.as_mut(),
            cb_lambda_map.as_mut(),
            cr_lambda_map.as_mut(),
        )
    {
        update_t1_closed_loop_streaming(
            use_xyb,
            is_color,
            y_bw,
            y_bh,
            cb_bw,
            cb_bh,
            cr_bw,
            cr_bh,
            &strip.y_raw_dct,
            &strip.cb_raw_dct,
            &strip.cr_raw_dct,
            &strip.y_blocks,
            &strip.cb_blocks,
            &strip.cr_blocks,
            y_quant,
            cb_quant,
            cr_quant,
            ym,
            cbm,
            crm,
            T1_WEIGHT,
            T1_SEAM_WEIGHT,
            T1_CHROMA_WEIGHT,
            use_parallel,
        )?;
    }

    // ------------------------------------------------------------------------
    // Pass 2: Build empirical Huffman rate models & run exact AC trellis
    // ------------------------------------------------------------------------
    let (dc_tables, cached_scans) = if config.mode == JpegMode::Progressive {
        let (prog_models, dc_tables, winning_scans) = build_empirical_progressive_models(
            config,
            &strip.y_blocks,
            &strip.cb_blocks,
            &strip.cr_blocks,
            is_color,
            UNSEEN_SYMBOL_BITS,
            use_parallel,
        )?;

        quantize_plane_progressive_trellis(
            &strip.y_raw_dct,
            &mut strip.y_blocks,
            &y_quant.values,
            &y_lambda_tbl,
            &prog_models[0],
            &trellis_cfg,
            base_mode,
            luma_lambda_map.as_deref(),
            max_dc_val,
            use_parallel,
        );
        if is_color {
            quantize_plane_progressive_trellis(
                &strip.cb_raw_dct,
                &mut strip.cb_blocks,
                &cb_quant.values,
                &cb_lambda_tbl,
                &prog_models[1],
                &trellis_cfg,
                base_mode,
                cb_lambda_map.as_deref(),
                max_dc_val,
                use_parallel,
            );
            quantize_plane_progressive_trellis(
                &strip.cr_raw_dct,
                &mut strip.cr_blocks,
                &cr_quant.values,
                &cr_lambda_tbl,
                &prog_models[2],
                &trellis_cfg,
                base_mode,
                cr_lambda_map.as_deref(),
                max_dc_val,
                use_parallel,
            );
        }
        (dc_tables, winning_scans)
    } else {
        let (ac_tables, dc_tables) = build_empirical_sequential_tables(
            config,
            &strip.y_blocks,
            &strip.cb_blocks,
            &strip.cr_blocks,
            is_color,
            UNSEEN_SYMBOL_BITS,
        )?;
        quantize_plane_sequential_trellis(
            &strip.y_raw_dct,
            &mut strip.y_blocks,
            &y_quant.values,
            &y_lambda_tbl,
            &ac_tables[0],
            &trellis_cfg,
            base_mode,
            luma_lambda_map.as_deref(),
            max_dc_val,
            use_parallel,
        );
        if is_color {
            quantize_plane_sequential_trellis(
                &strip.cb_raw_dct,
                &mut strip.cb_blocks,
                &cb_quant.values,
                &cb_lambda_tbl,
                &ac_tables[1],
                &trellis_cfg,
                base_mode,
                cb_lambda_map.as_deref(),
                max_dc_val,
                use_parallel,
            );
            quantize_plane_sequential_trellis(
                &strip.cr_raw_dct,
                &mut strip.cr_blocks,
                &cr_quant.values,
                &cr_lambda_tbl,
                &ac_tables[2],
                &trellis_cfg,
                base_mode,
                cr_lambda_map.as_deref(),
                max_dc_val,
                use_parallel,
            );
        }
        (dc_tables, None)
    };

    // ------------------------------------------------------------------------
    // Step 3: Empirical DC trellis (when dc_enabled)
    // ------------------------------------------------------------------------
    if trellis_cfg.is_dc_enabled() {
        let delta_dc_weight = trellis_cfg.get_delta_dc_weight();
        let dc_lambda1 = match base_mode {
            LambdaBaseMode::MozjpegNorm => trellis_cfg.lambda_log_scale1,
            LambdaBaseMode::DampedNorm { shift_s1, .. } => trellis_cfg.lambda_log_scale1 + shift_s1,
        };
        run_dc_trellis_plane(
            &strip.y_raw_dct,
            &mut strip.y_blocks,
            y_bw,
            y_quant.values[0],
            &dc_tables[0],
            dc_lambda1,
            trellis_cfg.lambda_log_scale2,
            delta_dc_weight,
            max_dc_val,
        )?;
        if is_color {
            run_dc_trellis_plane(
                &strip.cb_raw_dct,
                &mut strip.cb_blocks,
                cb_bw,
                cb_quant.values[0],
                &dc_tables[1],
                dc_lambda1,
                trellis_cfg.lambda_log_scale2,
                delta_dc_weight,
                max_dc_val,
            )?;
            run_dc_trellis_plane(
                &strip.cr_raw_dct,
                &mut strip.cr_blocks,
                cr_bw,
                cr_quant.values[0],
                &dc_tables[2],
                dc_lambda1,
                trellis_cfg.lambda_log_scale2,
                delta_dc_weight,
                max_dc_val,
            )?;
        }
    }

    Ok(cached_scans)
}

// ============================================================================
// Perceptual Layer: O(width) Sliding-IDCT Open-Loop (T2) & Closed-Loop (T1)
// ============================================================================

/// Map a luma-grid multiplier map `(y_bw, y_bh)` onto a (possibly subsampled)
/// target plane grid `(dst_bw, dst_bh)` via log-space geometric mean.
fn map_luma_to_plane(
    luma_map: &[f32],
    y_bw: usize,
    y_bh: usize,
    dst_bw: usize,
    dst_bh: usize,
) -> Result<Vec<f32>> {
    let nb = dst_bw * dst_bh;
    if dst_bw == y_bw && dst_bh == y_bh && luma_map.len() == nb {
        let mut out = try_vec_cap(nb, "rdo plane_map_clone")?;
        out.extend_from_slice(luma_map);
        return Ok(out);
    }
    let mut out = try_vec(nb, 1.0f32, "rdo plane_map_down")?;
    if y_bw == 0 || y_bh == 0 || dst_bw == 0 || dst_bh == 0 || luma_map.is_empty() {
        return Ok(out);
    }
    let sx = (y_bw + dst_bw - 1) / dst_bw;
    let sy = (y_bh + dst_bh - 1) / dst_bh;
    for by in 0..dst_bh {
        for bx in 0..dst_bw {
            let mut sum = 0.0f64;
            let mut cnt = 0usize;
            for dy in 0..sy {
                let ly = by * sy + dy;
                if ly >= y_bh {
                    continue;
                }
                for dx in 0..sx {
                    let lx = bx * sx + dx;
                    if lx >= y_bw {
                        continue;
                    }
                    sum += (luma_map[ly * y_bw + lx] as f64).ln();
                    cnt += 1;
                }
            }
            if cnt > 0 {
                out[by * dst_bw + bx] = (sum / cnt as f64).exp() as f32;
            }
        }
    }
    Ok(out)
}

#[inline]
fn coarse_damp_factor(s1_luma: f32, enabled: bool) -> f64 {
    if !enabled || s1_luma <= 16.0 {
        1.0
    } else {
        (1.0 - 0.55 * ((s1_luma - 16.0) / 48.0).clamp(0.0, 1.0)) as f64
    }
}

#[inline]
fn chroma_quadrant_mag_grad_from_block(
    blk: &[f32; DCT_BLOCK_SIZE],
    sub_x: bool,
    sub_y: bool,
    bx: usize,
    by: usize,
    neutral: f32,
    s1_c: f32,
) -> (f32, f32) {
    let x0 = if sub_x { (bx & 1) * 4 } else { 0 };
    let y0 = if sub_y { (by & 1) * 4 } else { 0 };
    let xlen = if sub_x { 4 } else { 8 };
    let ylen = if sub_y { 4 } else { 8 };
    let inv_s1 = 1.0 / s1_c.max(1e-4);

    let (mut mag, mut grad) = (0.0f32, 0.0f32);
    for dy in 0..ylen {
        let y = y0 + dy;
        let y1 = (y + 1).min(y0 + ylen - 1);
        for dx in 0..xlen {
            let x = x0 + dx;
            let x1 = (x + 1).min(x0 + xlen - 1);
            let v = blk[y * 8 + x];
            mag += (v - neutral).abs() * inv_s1;
            grad += ((v - blk[y * 8 + x1]).abs() + (v - blk[y1 * 8 + x]).abs()) * inv_s1;
        }
    }
    let cnt = (xlen * ylen) as f32;
    (mag / cnt, grad / (2.0 * cnt))
}

/// Streaming open-loop 4×4-quadrant min/mean + 3×3 neighborhood + smooth chroma
/// saturation lambda multiplier map (`T2_imij`).
///
/// Reconstructs raw blocks on the stack inside 2×2-block tiles (zero full-plane
/// spatial buffers; O(1) spatial working memory per worker) and produces results
/// bit-identical to full-plane reconstruction.
#[allow(clippy::too_many_arguments)]
fn build_t2_imij_lambda_map_streaming(
    use_xyb: bool,
    is_color: bool,
    y_bw: usize,
    y_bh: usize,
    cb_bw: usize,
    cb_bh: usize,
    cr_bw: usize,
    cr_bh: usize,
    y_raw: &[Block8x8f],
    cb_raw: &[Block8x8f],
    cr_raw: &[Block8x8f],
    y_quant: &QuantTable,
    cb_quant: &QuantTable,
    cr_quant: &QuantTable,
    strength: f32,
    sat_weight: f32,
    coarse_damp: bool,
    use_parallel: bool,
) -> Result<Vec<f32>> {
    let nb = y_bw * y_bh;
    if nb == 0 {
        return Ok(Vec::new());
    }

    let (luma_raw, c0_raw, c1_raw, luma_s1, c0_s1, c1_s1, c0_bw, c0_bh, c1_bw, c1_bh, n0, n1) =
        if use_xyb {
            (
                cb_raw,
                y_raw,
                cr_raw,
                cb_quant.values[1].max(1) as f32,
                y_quant.values[1].max(1) as f32,
                cr_quant.values[1].max(1) as f32,
                y_bw,
                y_bh,
                cr_bw,
                cr_bh,
                XYB_X_NEUTRAL,
                XYB_B_NEUTRAL,
            )
        } else {
            (
                y_raw,
                cb_raw,
                cr_raw,
                y_quant.values[1].max(1) as f32,
                cb_quant.values[1].max(1) as f32,
                cr_quant.values[1].max(1) as f32,
                cb_bw,
                cb_bh,
                cr_bw,
                cr_bh,
                0.0f32,
                0.0f32,
            )
        };

    let mut min_q = try_vec(nb, 0.0f32, "rdo t2_min_q")?;
    let mut mean_q = try_vec(nb, 0.0f32, "rdo t2_mean_q")?;
    let mut sat_smooth = try_vec(nb, 0.0f32, "rdo t2_sat")?;

    let do_chroma = is_color && sat_weight != 0.0 && !c0_raw.is_empty() && !c1_raw.is_empty();
    let c0_sub_x = c0_bw < y_bw;
    let c0_sub_y = c0_bh < y_bh;
    let c1_sub_x = c1_bw < y_bw;
    let c1_sub_y = c1_bh < y_bh;
    // Choose inner scan order inside each 2×2 tile so 4:2:2 (horizontal-only subsampling)
    // and 4:4:0 (vertical-only subsampling) both get 100% stack cache hits.
    let dy_outer = c0_sub_x && !c0_sub_y;

    let compute_row_pair =
        |by_pair: usize, min_pair: &mut [f32], mean_pair: &mut [f32], sat_pair: &mut [f32]| {
            let tok = IdctToken::summon();
            let by0 = by_pair * 2;
            let rows_in_pair = (y_bh - by0).min(2);
            let mut c0_cache: (usize, [f32; DCT_BLOCK_SIZE]) = (usize::MAX, [0.0; DCT_BLOCK_SIZE]);
            let mut c1_cache: (usize, [f32; DCT_BLOCK_SIZE]) = (usize::MAX, [0.0; DCT_BLOCK_SIZE]);

            for bx_pair in 0..y_bw.div_ceil(2) {
                let bx0 = bx_pair * 2;
                let cols_in_pair = (y_bw - bx0).min(2);

                let mut visit = |dx: usize, dy: usize| {
                    let bx = bx0 + dx;
                    let by = by0 + dy;
                    let local = dy * y_bw + bx;
                    let blk = idct_raw_block(tok, &luma_raw[by * y_bw + bx]);

                    let mut q_act = [0.0f32; 4];
                    for qy in 0..2 {
                        for qx in 0..2 {
                            let mut sum = 0.0f32;
                            for py in 0..4 {
                                let y = qy * 4 + py;
                                let y1 = (y + 1).min(7);
                                for px in 0..4 {
                                    let x = qx * 4 + px;
                                    let x1 = (x + 1).min(7);
                                    let v = blk[y * 8 + x];
                                    let vx = blk[y * 8 + x1];
                                    let vy = blk[y1 * 8 + x];
                                    sum += (v - vx).abs() + (v - vy).abs();
                                }
                            }
                            q_act[qy * 2 + qx] = sum * (1.0 / 32.0) / luma_s1;
                        }
                    }
                    min_pair[local] = q_act[0].min(q_act[1]).min(q_act[2]).min(q_act[3]);
                    mean_pair[local] = 0.25 * (q_act[0] + q_act[1] + q_act[2] + q_act[3]);

                    if do_chroma {
                        let cbx0 = if c0_sub_x {
                            (bx / 2).min(c0_bw - 1)
                        } else {
                            bx.min(c0_bw - 1)
                        };
                        let cby0 = if c0_sub_y {
                            (by / 2).min(c0_bh - 1)
                        } else {
                            by.min(c0_bh - 1)
                        };
                        let cidx0 = cby0 * c0_bw + cbx0;
                        if c0_cache.0 != cidx0 {
                            c0_cache = (cidx0, idct_raw_block(tok, &c0_raw[cidx0]));
                        }
                        let (mag0, grad0) = chroma_quadrant_mag_grad_from_block(
                            &c0_cache.1,
                            c0_sub_x,
                            c0_sub_y,
                            bx,
                            by,
                            n0,
                            c0_s1,
                        );

                        let cbx1 = if c1_sub_x {
                            (bx / 2).min(c1_bw - 1)
                        } else {
                            bx.min(c1_bw - 1)
                        };
                        let cby1 = if c1_sub_y {
                            (by / 2).min(c1_bh - 1)
                        } else {
                            by.min(c1_bh - 1)
                        };
                        let cidx1 = cby1 * c1_bw + cbx1;
                        if c1_cache.0 != cidx1 {
                            c1_cache = (cidx1, idct_raw_block(tok, &c1_raw[cidx1]));
                        }
                        let (mag1, grad1) = chroma_quadrant_mag_grad_from_block(
                            &c1_cache.1,
                            c1_sub_x,
                            c1_sub_y,
                            bx,
                            by,
                            n1,
                            c1_s1,
                        );

                        let avg_mag = 0.5 * (mag0 + mag1);
                        let avg_grad = 0.5 * (grad0 + grad1);
                        let smooth = 1.0 / (1.0 + avg_grad * avg_grad);
                        sat_pair[local] = (avg_mag / (avg_mag + 8.0)) * smooth;
                    }
                };

                if dy_outer {
                    for dy in 0..rows_in_pair {
                        for dx in 0..cols_in_pair {
                            visit(dx, dy);
                        }
                    }
                } else {
                    for dx in 0..cols_in_pair {
                        for dy in 0..rows_in_pair {
                            visit(dx, dy);
                        }
                    }
                }
            }
        };

    let pair_stride = 2 * y_bw;
    #[cfg(feature = "parallel")]
    if use_parallel && y_bh >= 4 {
        min_q
            .par_chunks_mut(pair_stride)
            .zip(mean_q.par_chunks_mut(pair_stride))
            .zip(sat_smooth.par_chunks_mut(pair_stride))
            .enumerate()
            .for_each(|(by_pair, ((mn, me), st))| compute_row_pair(by_pair, mn, me, st));
    } else {
        for by_pair in 0..y_bh.div_ceil(2) {
            let start = by_pair * pair_stride;
            let end = (start + pair_stride).min(nb);
            compute_row_pair(
                by_pair,
                &mut min_q[start..end],
                &mut mean_q[start..end],
                &mut sat_smooth[start..end],
            );
        }
    }
    #[cfg(not(feature = "parallel"))]
    {
        let _ = use_parallel;
        for by_pair in 0..y_bh.div_ceil(2) {
            let start = by_pair * pair_stride;
            let end = (start + pair_stride).min(nb);
            compute_row_pair(
                by_pair,
                &mut min_q[start..end],
                &mut mean_q[start..end],
                &mut sat_smooth[start..end],
            );
        }
    }

    // Pass 2: 3×3 neighborhood + strict ascending f64 reduction for mean_raw
    let mut raw = try_vec(nb, 0.0f64, "rdo t2_raw")?;
    let mut active_sum = 0.0f64;
    let mut active_n = 0usize;
    let mut all_sum = 0.0f64;
    let sw = sat_weight as f64;

    for by in 0..y_bh {
        let y0 = by.saturating_sub(1);
        let y1 = (by + 1).min(y_bh - 1);
        for bx in 0..y_bw {
            let x0 = bx.saturating_sub(1);
            let x1 = (bx + 1).min(y_bw - 1);
            let mut s_min = f32::INFINITY;
            let mut s_sum = 0.0f32;
            let mut s_cnt = 0.0f32;
            for yy in y0..=y1 {
                let row = &mean_q[yy * y_bw..(yy + 1) * y_bw];
                for &v in &row[x0..=x1] {
                    if v < s_min {
                        s_min = v;
                    }
                    s_sum += v;
                    s_cnt += 1.0;
                }
            }
            let idx = by * y_bw + bx;
            let act = 0.45 * min_q[idx] + 0.30 * s_min + 0.25 * (s_sum / s_cnt);
            let score = -((act as f64) + 0.25).ln() + sw * (sat_smooth[idx] as f64);
            raw[idx] = score;
            all_sum += score;
            if mean_q[idx] > 0.05 {
                active_sum += score;
                active_n += 1;
            }
        }
    }

    let mean_raw = if active_n > 0 {
        active_sum / active_n as f64
    } else {
        all_sum / nb as f64
    };
    let st = (strength as f64) * coarse_damp_factor(luma_s1, coarse_damp);
    let mut out = try_vec(nb, 1.0f32, "rdo t2_lambda")?;
    for i in 0..nb {
        let lm = (st * (raw[i] - mean_raw)).clamp(-0.85, 0.85);
        out[i] = lm.exp() as f32;
    }
    Ok(out)
}

/// Local contrast-masking visibility on an 8×8 luma patch (3×3 eroded max-neighbor gradient).
#[inline]
fn block_visibility(
    orig: &[f32; DCT_BLOCK_SIZE],
    step1: f32,
    mask_tau: f32,
) -> [f32; DCT_BLOCK_SIZE] {
    let mut grad = [0.0f32; DCT_BLOCK_SIZE];
    for y in 0..8usize {
        let ym = y.saturating_sub(1);
        let yp = (y + 1).min(7);
        for x in 0..8usize {
            let xm = x.saturating_sub(1);
            let xp = (x + 1).min(7);
            let v = orig[y * 8 + x];
            grad[y * 8 + x] = (v - orig[y * 8 + xm])
                .abs()
                .max((v - orig[y * 8 + xp]).abs())
                .max((v - orig[ym * 8 + x]).abs())
                .max((v - orig[yp * 8 + x]).abs());
        }
    }
    let inv_tau = 1.0 / (mask_tau * step1).max(1e-4);
    let mut vis = [0.0f32; DCT_BLOCK_SIZE];
    for y in 0..8usize {
        let ym = y.saturating_sub(1);
        let yp = (y + 1).min(7);
        for x in 0..8usize {
            let xm = x.saturating_sub(1);
            let xp = (x + 1).min(7);
            let mut mn = grad[y * 8 + x];
            for yy in ym..=yp {
                for xx in xm..=xp {
                    let g = grad[yy * 8 + xx];
                    if g < mn {
                        mn = g;
                    }
                }
            }
            let act = 0.5 * (grad[y * 8 + x] + mn);
            let r = act * inv_tau;
            vis[y * 8 + x] = 1.0 / (1.0 + r * r);
        }
    }
    vis
}

/// Unfiltered 1-pixel block-boundary step discontinuity cost.
#[inline]
fn seam_pair_cost(
    a_orig: f32,
    b_orig: f32,
    e_a: f32,
    e_b: f32,
    inv_seam_tau_s1: f64,
    inv_s1_sq: f64,
) -> f64 {
    let d_src = (b_orig - a_orig) as f64;
    let r = d_src * inv_seam_tau_s1;
    let gate = 1.0 / (1.0 + r * r);
    let de = (e_b - e_a) as f64;
    gate * (de * de) * inv_s1_sq
}

#[derive(Clone, Copy)]
enum VisMode<'a> {
    None,
    SelfLuma {
        luma_s1: f32,
    },
    ExternalLuma {
        luma_raw: &'a [Block8x8f],
        luma_s1: f32,
    },
}

/// Edge strips exported by one T1 stripe for exact cross-stripe vertical seam fusion.
struct T1StripeEdges {
    top_orig: Vec<[f32; 8]>,
    top_rec: Vec<[f32; 8]>,
    bot_orig: Vec<[f32; 8]>,
    bot_rec: Vec<[f32; 8]>,
    /// Deferred vertical seam between row `r0` and row `r0 + 1` for stripes `s > 0`,
    /// applied to row `r0` right after `(r0 - 1, r0)` so IEEE-754 summation order
    /// is bit-identical to full-plane processing regardless of thread count.
    first_row_bot_vs: Vec<f64>,
}

/// Compute per-block visibility-weighted interior + 2D (H+V) unfiltered seam errors
/// using O(width) stripe-local sliding IDCT edge buffers.
///
/// Parallelizable across stripes with bit-exact `1T == MT` determinism and bit-exact
/// equivalence to full-plane reconstruction.
#[allow(clippy::too_many_arguments)]
fn plane_closed_loop_errors_streaming(
    raw_dct: &[Block8x8f],
    quant_blocks: &[[i16; DCT_BLOCK_SIZE]],
    qvals_natural: &[u16; DCT_BLOCK_SIZE],
    vis_mode: VisMode<'_>,
    bw: usize,
    bh: usize,
    s1: f32,
    seam_weight: f64,
    use_parallel: bool,
) -> Result<Vec<f64>> {
    let nb = bw * bh;
    if nb == 0 {
        return Ok(Vec::new());
    }
    let inv_s1_sq = 1.0 / ((s1 as f64) * (s1 as f64)).max(1e-12);
    let inv_seam_tau = 1.0 / ((1.5 * s1) as f64).max(1e-6);

    let mut blk_err = try_vec(nb, 0.0f64, "rdo t1_blk_err")?;
    let stripe_blocks = T1_STRIPE_ROWS * bw;

    let run_stripe = |stripe_idx: usize, stripe_err: &mut [f64]| -> Result<T1StripeEdges> {
        let tok = IdctToken::summon();
        let r0 = stripe_idx * T1_STRIPE_ROWS;
        let rows = stripe_err.len() / bw;

        let mut prev_bot_orig = if seam_weight != 0.0 {
            try_vec(bw, [0.0f32; 8], "rdo t1_prev_orig")?
        } else {
            Vec::new()
        };
        let mut prev_bot_rec = if seam_weight != 0.0 {
            try_vec(bw, [0.0f32; 8], "rdo t1_prev_rec")?
        } else {
            Vec::new()
        };
        let mut top_orig = if seam_weight != 0.0 && stripe_idx > 0 {
            try_vec(bw, [0.0f32; 8], "rdo t1_top_orig")?
        } else {
            Vec::new()
        };
        let mut top_rec = if seam_weight != 0.0 && stripe_idx > 0 {
            try_vec(bw, [0.0f32; 8], "rdo t1_top_rec")?
        } else {
            Vec::new()
        };
        let mut cur_vs = if seam_weight != 0.0 {
            try_vec(bw, 0.0f64, "rdo t1_cur_vs")?
        } else {
            Vec::new()
        };
        let mut first_row_bot_vs = if seam_weight != 0.0 && stripe_idx > 0 && rows > 1 {
            try_vec(bw, 0.0f64, "rdo t1_defer_vs")?
        } else {
            Vec::new()
        };

        for local_by in 0..rows {
            let by = r0 + local_by;
            let row_off = local_by * bw;
            let mut left_orig_col = [0.0f32; 8];
            let mut left_rec_col = [0.0f32; 8];

            for bx in 0..bw {
                let idx = by * bw + bx;
                let ob = idct_raw_block(tok, &raw_dct[idx]);
                let rb = idct_quantized_zigzag(tok, &quant_blocks[idx], qvals_natural);

                let vis_arr = match vis_mode {
                    VisMode::None => None,
                    VisMode::SelfLuma { luma_s1 } => Some(block_visibility(&ob, luma_s1, 2.0)),
                    VisMode::ExternalLuma { luma_raw, luma_s1 } => {
                        let lo = idct_raw_block(tok, &luma_raw[idx]);
                        Some(block_visibility(&lo, luma_s1, 2.0))
                    }
                };

                let mut acc = 0.0f64;
                for k in 0..DCT_BLOCK_SIZE {
                    let e = (ob[k] - rb[k]) as f64;
                    let e2 = (e * e * inv_s1_sq).min(64.0);
                    let vw = vis_arr.as_ref().map_or(1.0f64, |v| v[k] as f64);
                    acc += vw * (e2 + 0.15 * e2 * e2);
                }

                if seam_weight != 0.0 && bx > 0 {
                    let mut hs = 0.0f64;
                    for y in 0..8 {
                        let oa = left_orig_col[y];
                        let ob_px = ob[y * 8];
                        let ea = oa - left_rec_col[y];
                        let eb = ob_px - rb[y * 8];
                        hs += seam_pair_cost(oa, ob_px, ea, eb, inv_seam_tau, inv_s1_sq);
                    }
                    let term = seam_weight * hs;
                    acc += term;
                    stripe_err[row_off + (bx - 1)] += term;
                }
                stripe_err[row_off + bx] += acc;

                if seam_weight != 0.0 {
                    for y in 0..8 {
                        left_orig_col[y] = ob[y * 8 + 7];
                        left_rec_col[y] = rb[y * 8 + 7];
                    }
                    if local_by == 0 && stripe_idx > 0 {
                        top_orig[bx].copy_from_slice(&ob[..8]);
                        top_rec[bx].copy_from_slice(&rb[..8]);
                    }
                    if local_by > 0 {
                        let mut vs = 0.0f64;
                        for x in 0..8 {
                            let oa = prev_bot_orig[bx][x];
                            let ob_px = ob[x];
                            let ea = oa - prev_bot_rec[bx][x];
                            let eb = ob_px - rb[x];
                            vs += seam_pair_cost(oa, ob_px, ea, eb, inv_seam_tau, inv_s1_sq);
                        }
                        cur_vs[bx] = seam_weight * vs;
                    }
                    prev_bot_orig[bx].copy_from_slice(&ob[7 * 8..7 * 8 + 8]);
                    prev_bot_rec[bx].copy_from_slice(&rb[7 * 8..7 * 8 + 8]);
                }
            }

            // Apply vertical seams AFTER all horizontal seams in row `local_by`
            // completed so addition order matches full-plane bit-for-bit.
            if seam_weight != 0.0 && local_by > 0 {
                let prev_off = (local_by - 1) * bw;
                if local_by == 1 && stripe_idx > 0 {
                    first_row_bot_vs.copy_from_slice(&cur_vs);
                    for bx in 0..bw {
                        stripe_err[row_off + bx] += cur_vs[bx];
                    }
                } else {
                    for bx in 0..bw {
                        let term = cur_vs[bx];
                        stripe_err[prev_off + bx] += term;
                        stripe_err[row_off + bx] += term;
                    }
                }
            }
        }

        Ok(T1StripeEdges {
            top_orig,
            top_rec,
            bot_orig: prev_bot_orig,
            bot_rec: prev_bot_rec,
            first_row_bot_vs,
        })
    };

    #[cfg(feature = "parallel")]
    let boundaries_res: Vec<Result<T1StripeEdges>> = if use_parallel && bh > T1_STRIPE_ROWS {
        blk_err
            .par_chunks_mut(stripe_blocks)
            .enumerate()
            .map(|(s, chunk)| run_stripe(s, chunk))
            .collect()
    } else {
        blk_err
            .chunks_mut(stripe_blocks)
            .enumerate()
            .map(|(s, chunk)| run_stripe(s, chunk))
            .collect()
    };

    #[cfg(not(feature = "parallel"))]
    let boundaries_res: Vec<Result<T1StripeEdges>> = {
        let _ = use_parallel;
        blk_err
            .chunks_mut(stripe_blocks)
            .enumerate()
            .map(|(s, chunk)| run_stripe(s, chunk))
            .collect()
    };

    let mut boundaries = try_vec_cap(boundaries_res.len(), "rdo t1_bounds")?;
    for res in boundaries_res {
        boundaries.push(res?);
    }

    // Fuse stripe-boundary vertical seams in strict ascending order, applying
    // `(r0 - 1, r0)` before row `r0`'s deferred `(r0, r0 + 1)` seam.
    if seam_weight != 0.0 {
        for s in 1..boundaries.len() {
            let r0 = s * T1_STRIPE_ROWS;
            let prev_bot_o = &boundaries[s - 1].bot_orig;
            let prev_bot_r = &boundaries[s - 1].bot_rec;
            let cur_top_o = &boundaries[s].top_orig;
            let cur_top_r = &boundaries[s].top_rec;
            let deferred_bot = &boundaries[s].first_row_bot_vs;
            for bx in 0..bw {
                let mut vs = 0.0f64;
                for x in 0..8 {
                    let oa = prev_bot_o[bx][x];
                    let ob = cur_top_o[bx][x];
                    let ea = oa - prev_bot_r[bx][x];
                    let eb = ob - cur_top_r[bx][x];
                    vs += seam_pair_cost(oa, ob, ea, eb, inv_seam_tau, inv_s1_sq);
                }
                let term = seam_weight * vs;
                blk_err[(r0 - 1) * bw + bx] += term;
                blk_err[r0 * bw + bx] += term;
                if !deferred_bot.is_empty() {
                    blk_err[r0 * bw + bx] += deferred_bot[bx];
                }
            }
        }
    }

    Ok(blk_err)
}

/// Apply global log-centered T1 update to a lambda map in place (deterministic
/// strict ascending f64 sum).
fn apply_log_centered_feedback(lambda_map: &mut [f32], errs: &[f64], weight: f32) {
    let n = lambda_map.len().min(errs.len());
    if n == 0 || weight == 0.0 {
        return;
    }
    let mut sum_log = 0.0f64;
    let mut cnt = 0usize;
    for &e in &errs[..n] {
        if e > 1e-4 {
            sum_log += (e + 0.25).ln();
            cnt += 1;
        }
    }
    if cnt == 0 {
        return;
    }
    let ref_log = sum_log / cnt as f64;
    let w = weight as f64;
    for i in 0..n {
        let e = errs[i];
        let delta = ((e + 0.25).ln() - ref_log).clamp(-0.75, 1.15);
        lambda_map[i] = ((lambda_map[i] as f64) * (w * delta).exp()).clamp(0.30, 3.5) as f32;
    }
}

/// Streaming closed-loop T1 update (unified perceptual error map, O(width) IDCT).
#[allow(clippy::too_many_arguments)]
fn update_t1_closed_loop_streaming(
    use_xyb: bool,
    is_color: bool,
    y_bw: usize,
    y_bh: usize,
    cb_bw: usize,
    cb_bh: usize,
    cr_bw: usize,
    cr_bh: usize,
    y_raw: &[Block8x8f],
    cb_raw: &[Block8x8f],
    cr_raw: &[Block8x8f],
    y_blocks: &[[i16; DCT_BLOCK_SIZE]],
    cb_blocks: &[[i16; DCT_BLOCK_SIZE]],
    cr_blocks: &[[i16; DCT_BLOCK_SIZE]],
    y_quant: &QuantTable,
    cb_quant: &QuantTable,
    cr_quant: &QuantTable,
    luma_lambda_map: &mut [f32],
    cb_lambda_map: &mut [f32],
    cr_lambda_map: &mut [f32],
    weight: f32,
    seam_weight: f64,
    chroma_weight: f64,
    use_parallel: bool,
) -> Result<()> {
    let n_luma = y_bw * y_bh;

    if !use_xyb {
        let luma_s1 = y_quant.values[1].max(1) as f32;
        let mut y_err = plane_closed_loop_errors_streaming(
            y_raw,
            y_blocks,
            &y_quant.values,
            VisMode::SelfLuma { luma_s1 },
            y_bw,
            y_bh,
            luma_s1,
            seam_weight,
            use_parallel,
        )?;

        if is_color {
            let full_res_chroma = cb_bw == y_bw && cb_bh == y_bh;
            let cb_vis = if full_res_chroma {
                VisMode::ExternalLuma {
                    luma_raw: y_raw,
                    luma_s1,
                }
            } else {
                VisMode::None
            };
            let cb_err = plane_closed_loop_errors_streaming(
                cb_raw,
                cb_blocks,
                &cb_quant.values,
                cb_vis,
                cb_bw,
                cb_bh,
                cb_quant.values[1].max(1) as f32,
                seam_weight,
                use_parallel,
            )?;
            let cr_err = plane_closed_loop_errors_streaming(
                cr_raw,
                cr_blocks,
                &cr_quant.values,
                cb_vis,
                cr_bw,
                cr_bh,
                cr_quant.values[1].max(1) as f32,
                seam_weight,
                use_parallel,
            )?;

            if full_res_chroma {
                for i in 0..n_luma {
                    y_err[i] += chroma_weight * (cb_err[i] + cr_err[i]);
                }
                apply_log_centered_feedback(luma_lambda_map, &y_err, weight);
                cb_lambda_map.copy_from_slice(luma_lambda_map);
                cr_lambda_map.copy_from_slice(luma_lambda_map);
            } else {
                if chroma_weight > 0.0 {
                    for by in 0..y_bh {
                        let cby = (by * cb_bh / y_bh).min(cb_bh - 1);
                        for bx in 0..y_bw {
                            let cbx = (bx * cb_bw / y_bw).min(cb_bw - 1);
                            let ci = cby * cb_bw + cbx;
                            y_err[by * y_bw + bx] +=
                                chroma_weight * 0.5 * (cb_err[ci] + cr_err[ci]);
                        }
                    }
                }
                apply_log_centered_feedback(luma_lambda_map, &y_err, weight);
                let mapped = map_luma_to_plane(luma_lambda_map, y_bw, y_bh, cb_bw, cb_bh)?;
                cb_lambda_map.copy_from_slice(&mapped);
                cr_lambda_map.copy_from_slice(&mapped);
            }
        } else {
            apply_log_centered_feedback(luma_lambda_map, &y_err, weight);
        }
    } else {
        // XYB: comp 0 (y_*) = X, comp 1 (cb_*) = perceptual Y, comp 2 (cr_*) = B
        let luma_s1 = cb_quant.values[1].max(1) as f32;
        let x_err = plane_closed_loop_errors_streaming(
            y_raw,
            y_blocks,
            &y_quant.values,
            VisMode::ExternalLuma {
                luma_raw: cb_raw,
                luma_s1,
            },
            y_bw,
            y_bh,
            y_quant.values[1].max(1) as f32,
            seam_weight,
            use_parallel,
        )?;
        let mut y_luma_err = plane_closed_loop_errors_streaming(
            cb_raw,
            cb_blocks,
            &cb_quant.values,
            VisMode::SelfLuma { luma_s1 },
            cb_bw,
            cb_bh,
            luma_s1,
            seam_weight,
            use_parallel,
        )?;
        let b_vis = if cr_bw == y_bw && cr_bh == y_bh {
            VisMode::ExternalLuma {
                luma_raw: cb_raw,
                luma_s1,
            }
        } else {
            VisMode::None
        };
        let b_err = plane_closed_loop_errors_streaming(
            cr_raw,
            cr_blocks,
            &cr_quant.values,
            b_vis,
            cr_bw,
            cr_bh,
            cr_quant.values[1].max(1) as f32,
            seam_weight,
            use_parallel,
        )?;

        if chroma_weight > 0.0 {
            for by in 0..y_bh {
                let bby = (by * cr_bh / y_bh).min(cr_bh.saturating_sub(1));
                for bx in 0..y_bw {
                    let bbx = (bx * cr_bw / y_bw).min(cr_bw.saturating_sub(1));
                    let i = by * y_bw + bx;
                    let bi = bby * cr_bw + bbx;
                    let be = b_err.get(bi).copied().unwrap_or(0.0);
                    y_luma_err[i] += chroma_weight * (x_err[i] + 0.5 * be);
                }
            }
        }

        apply_log_centered_feedback(cb_lambda_map, &y_luma_err, weight);
        luma_lambda_map.copy_from_slice(cb_lambda_map);
        let mapped = map_luma_to_plane(cb_lambda_map, cb_bw, cb_bh, cr_bw, cr_bh)?;
        cr_lambda_map.copy_from_slice(&mapped);
    }

    Ok(())
}

// ============================================================================
// Empirical Sequential & Progressive Rate Models
// ============================================================================

/// Build empirical sequential `(AC[3], DC[3])` rate tables from Pass-1 blocks.
fn build_empirical_sequential_tables(
    config: &ComputedConfig,
    y_blocks: &[[i16; DCT_BLOCK_SIZE]],
    cb_blocks: &[[i16; DCT_BLOCK_SIZE]],
    cr_blocks: &[[i16; DCT_BLOCK_SIZE]],
    is_color: bool,
    unseen_bits: u8,
) -> Result<([RateTable; 3], [RateTable; 3])> {
    if config.use_xyb || config.use_rgb {
        // Sequential XYB and RGB passthrough emit a shared single (DC, AC) pair.
        let mut dc_freq = FrequencyCounter::new();
        let mut ac_freq = FrequencyCounter::new();
        count_single_plane_seq(y_blocks, &mut dc_freq, &mut ac_freq);
        if is_color {
            count_single_plane_seq(cb_blocks, &mut dc_freq, &mut ac_freq);
            count_single_plane_seq(cr_blocks, &mut dc_freq, &mut ac_freq);
        }
        let dc_opt = dc_freq.generate_table()?;
        let ac_opt = ac_freq.generate_table()?;
        let dc_rt = RateTable::from_lengths_with_fallback(&dc_opt.lengths, unseen_bits);
        let ac_rt = RateTable::from_lengths_with_fallback(&ac_opt.lengths, unseen_bits);
        return Ok((
            [ac_rt.clone(), ac_rt.clone(), ac_rt],
            [dc_rt.clone(), dc_rt.clone(), dc_rt],
        ));
    }

    let freqs = config.count_block_frequencies(y_blocks, cb_blocks, cr_blocks, is_color);
    let tables = freqs.generate_tables()?;
    let ac_l = RateTable::from_lengths_with_fallback(&tables.ac_luma.table.lengths, unseen_bits);
    let ac_c = RateTable::from_lengths_with_fallback(&tables.ac_chroma.table.lengths, unseen_bits);
    let dc_l = RateTable::from_lengths_with_fallback(&tables.dc_luma.table.lengths, unseen_bits);
    let dc_c = RateTable::from_lengths_with_fallback(&tables.dc_chroma.table.lengths, unseen_bits);
    Ok(([ac_l, ac_c.clone(), ac_c], [dc_l, dc_c.clone(), dc_c]))
}

fn count_single_plane_seq(
    blocks: &[[i16; DCT_BLOCK_SIZE]],
    dc_freq: &mut FrequencyCounter,
    ac_freq: &mut FrequencyCounter,
) {
    let mut prev_dc: i16 = 0;
    for blk in blocks {
        let diff = blk[0].wrapping_sub(prev_dc);
        prev_dc = blk[0];
        dc_freq.count(jpeg_nbits(diff));
        let mut zero_run = 0u8;
        for &c in &blk[1..] {
            if c == 0 {
                zero_run += 1;
            } else {
                while zero_run >= 16 {
                    ac_freq.count(0xF0);
                    zero_run -= 16;
                }
                let cat = jpeg_nbits(c);
                ac_freq.count((zero_run << 4) | cat);
                zero_run = 0;
            }
        }
        if zero_run > 0 {
            ac_freq.count(0x00);
        }
    }
}

/// Empirical rate model for one progressive AC scan (first-pass or refinement).
#[derive(Clone, Debug)]
struct ProgScanRate {
    ss: u8,
    se: u8,
    al: u8,
    table: RateTable,
    /// Amortized EOB cost for an all-zero block in this scan.
    eob_allzero: f32,
    /// Marginal EOB cost for a nonzero block ending before `se`.
    eob_trailing: f32,
}

/// Exact empirical progressive AC rate model for one color component.
///
/// Separates first-pass (`ah == 0`) spectral bands from refinement (`ah > 0`)
/// scans so multi-band first passes paired with a single full-range `1..63`
/// refinement scan (such as 3-band `R8` scripts) are priced exactly without
/// splitting refinement zero-runs at first-pass band boundaries.
#[derive(Clone, Debug)]
struct ComponentProgModel {
    /// First-pass AC scans (`ah == 0`), sorted ascending by `ss`.
    first_scans: Vec<ProgScanRate>,
    /// Refinement AC scans (`ah > 0`), sorted descending by `al` then ascending by `ss`.
    refine_scans: Vec<ProgScanRate>,
}

type PricedScans = (
    ContextConfig,
    ProgressiveTokenBuffer,
    Vec<usize>,
    usize,
    Vec<crate::huffman::optimize::OptimizedTable>,
    u64,
);

/// Tokenize a candidate progressive scan script and compute its exact clustered
/// Huffman + extra-bits + refbits cost (plus SOS/DHT overhead estimate).
fn tokenize_and_price_scans(
    config: &ComputedConfig,
    scans: &[ProgressiveScan],
    y_blocks: &[[i16; DCT_BLOCK_SIZE]],
    cb_blocks: &[[i16; DCT_BLOCK_SIZE]],
    cr_blocks: &[[i16; DCT_BLOCK_SIZE]],
    is_color: bool,
) -> Result<PricedScans> {
    let num_components = if is_color { 3 } else { 1 };
    let context_config = ContextConfig::for_progressive(
        num_components,
        scans.iter().map(|s| (s.ss, s.se, s.components.len())),
    );
    let mut token_buffer = ProgressiveTokenBuffer::new(num_components, context_config.num_contexts);

    for (scan_idx, scan) in scans.iter().enumerate() {
        let context = if scan.ss == 0 && scan.se == 0 {
            context_config.dc_context(scan.components[0] as usize) as u8
        } else {
            context_config.ac_context(scan_idx, 0) as u8
        };
        if scan.ss == 0 && scan.se == 0 {
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
            let comp_idx: Vec<usize> = scan.components.iter().map(|&c| c as usize).collect();
            token_buffer.tokenize_dc_scan(
                &blocks,
                &comp_idx,
                scan.al,
                scan.ah,
                config.restart_interval,
            );
        } else if scan.ah == 0 {
            let blocks = match scan.components[0] {
                0 => y_blocks,
                1 => cb_blocks,
                2 => cr_blocks,
                _ => &[][..],
            };
            token_buffer.tokenize_ac_first_scan(
                blocks,
                context,
                scan.ss,
                scan.se,
                scan.al,
                config.restart_interval,
            );
        } else {
            let blocks = match scan.components[0] {
                0 => y_blocks,
                1 => cb_blocks,
                2 => cr_blocks,
                _ => &[][..],
            };
            token_buffer.tokenize_ac_refinement_scan(
                blocks,
                context,
                scan.ss,
                scan.se,
                scan.ah,
                scan.al,
                config.restart_interval,
            )?;
        }
    }

    let (context_map, num_dc_tables, tables, _ac_slot_ids) =
        token_buffer.generate_optimized_tables(4, 12, context_config.ac_offset, false)?;

    // Exact Huffman symbol bits + extra bits + refinement bits + DHT/SOS overhead estimate
    let mut total_bits: u64 = (scans.len() as u64) * 14 * 8 + (tables.len() as u64) * 24 * 8;
    for (ctx, counter) in token_buffer.counters.iter().enumerate() {
        let Some(&tidx) = context_map.get(ctx) else {
            continue;
        };
        let Some(opt) = tables.get(tidx) else {
            continue;
        };
        let is_dc = ctx < context_config.ac_offset;
        for sym in 0..=255u8 {
            let c = counter.get_count(sym) as u64;
            if c == 0 {
                continue;
            }
            let hlen = opt.table.lengths[sym as usize] as u64;
            let extra = if is_dc {
                sym as u64
            } else {
                let lo = sym & 0x0F;
                if lo == 0 {
                    // EOBRUN: high nibble = extra bits (except 0xF0 ZRL has 0 extra bits)
                    if sym == 0xF0 { 0 } else { (sym >> 4) as u64 }
                } else {
                    lo as u64
                }
            };
            total_bits += c * (hlen + extra);
        }
    }
    for info in &token_buffer.scan_info {
        total_bits += info.refbits.len() as u64;
    }

    Ok((
        context_config,
        token_buffer,
        context_map,
        num_dc_tables,
        tables,
        total_bits,
    ))
}

/// Build exact progressive empirical rate models directly from `ProgressiveTokenBuffer`.
///
/// When `ScanStrategy::Search` is active, evaluates all candidate scan scripts
/// (including `R8` per-component 3-band candidates and deduped mozjpeg script),
/// builds the trellis model for the winning script, and returns `Some(winning_scans)`
/// so the progressive encoder emits that script directly without repeating the search.
#[allow(clippy::too_many_arguments)]
fn build_empirical_progressive_models(
    config: &ComputedConfig,
    y_blocks: &[[i16; DCT_BLOCK_SIZE]],
    cb_blocks: &[[i16; DCT_BLOCK_SIZE]],
    cr_blocks: &[[i16; DCT_BLOCK_SIZE]],
    is_color: bool,
    unseen_bits: u8,
    use_parallel: bool,
) -> Result<(
    [ComponentProgModel; 3],
    [RateTable; 3],
    Option<Vec<ProgressiveScan>>,
)> {
    let num_components = if is_color { 3 } else { 1 };
    let is_search =
        config.scan_strategy == ScanStrategy::Search && !config.use_xyb && !config.use_rgb;

    let candidate_scripts: Vec<Vec<ProgressiveScan>> = if is_search {
        let interleave_dc_444 = is_color && matches!(config.subsampling, Subsampling::S444);
        let mut cands = crate::encode::scan_optimize::generate_candidate_scripts(
            y_blocks,
            cb_blocks,
            cr_blocks,
            num_components as u8,
            interleave_dc_444,
            #[cfg(feature = "parallel")]
            use_parallel,
        )?;
        let mut moz_cfg = config.clone();
        moz_cfg.scan_strategy = ScanStrategy::Mozjpeg;
        let moz_script = moz_cfg.get_progressive_scan_script(is_color);
        if !cands
            .iter()
            .any(|c| crate::encode::scan_optimize::scripts_equivalent(c, &moz_script))
        {
            cands.push(moz_script);
        }
        cands
    } else {
        alloc::vec![config.get_progressive_scan_script(is_color)]
    };

    #[cfg(feature = "parallel")]
    let priced_all: Vec<Result<PricedScans>> = if use_parallel && candidate_scripts.len() > 1 {
        candidate_scripts
            .par_iter()
            .map(|script| {
                tokenize_and_price_scans(config, script, y_blocks, cb_blocks, cr_blocks, is_color)
            })
            .collect()
    } else {
        candidate_scripts
            .iter()
            .map(|script| {
                tokenize_and_price_scans(config, script, y_blocks, cb_blocks, cr_blocks, is_color)
            })
            .collect()
    };

    #[cfg(not(feature = "parallel"))]
    let priced_all: Vec<Result<PricedScans>> = {
        let _ = use_parallel;
        candidate_scripts
            .iter()
            .map(|script| {
                tokenize_and_price_scans(config, script, y_blocks, cb_blocks, cr_blocks, is_color)
            })
            .collect()
    };

    let mut best_choice: Option<(Vec<ProgressiveScan>, PricedScans)> = None;
    let mut best_bits = u64::MAX;
    for (script, priced_res) in candidate_scripts.into_iter().zip(priced_all) {
        let priced = priced_res?;
        if priced.5 < best_bits {
            best_bits = priced.5;
            best_choice = Some((script, priced));
        }
    }
    let (scans, (context_config, token_buffer, context_map, num_dc_tables, tables, _)) =
        best_choice.ok_or_else(|| Error::internal("no progressive scan script candidates"))?;

    // DC rate tables per component
    let std_dc = RateTable::standard_luma_dc();
    let dc_tables: [RateTable; 3] = core::array::from_fn(|c| {
        if c < num_components && c < context_map.len() {
            let tidx = context_map[c];
            if let Some(t) = tables.get(tidx) {
                return RateTable::from_lengths_with_fallback(&t.table.lengths, unseen_bits);
            }
        }
        std_dc.clone()
    });

    // Separate AC scans into first-pass (ah == 0) and refinement (ah > 0) per component
    let mut models = [
        ComponentProgModel {
            first_scans: Vec::new(),
            refine_scans: Vec::new(),
        },
        ComponentProgModel {
            first_scans: Vec::new(),
            refine_scans: Vec::new(),
        },
        ComponentProgModel {
            first_scans: Vec::new(),
            refine_scans: Vec::new(),
        },
    ];

    for (scan_idx, scan) in scans.iter().enumerate() {
        if scan.ss == 0 {
            continue;
        }
        let comp = scan.components[0] as usize;
        if comp >= 3 {
            continue;
        }
        let ac_ctx = context_config.ac_context(scan_idx, 0);
        let cluster_idx = if ac_ctx < context_map.len() {
            context_map[ac_ctx].saturating_sub(num_dc_tables)
        } else {
            0
        };
        let opt_table = tables.get(num_dc_tables + cluster_idx);
        let rt = if let Some(t) = opt_table {
            RateTable::from_lengths_with_fallback(&t.table.lengths, unseen_bits)
        } else {
            RateTable::standard_luma_ac()
        };

        let plane_blocks = match comp {
            0 => y_blocks,
            1 => cb_blocks,
            _ => cr_blocks,
        };
        let (eob_allzero, eob_trailing) =
            estimate_eob_costs(token_buffer.counter(ac_ctx), &rt, plane_blocks, scan);

        let sr = ProgScanRate {
            ss: scan.ss,
            se: scan.se,
            al: scan.al,
            table: rt,
            eob_allzero,
            eob_trailing,
        };

        if scan.ah == 0 {
            models[comp].first_scans.push(sr);
        } else {
            models[comp].refine_scans.push(sr);
        }
    }

    for m in &mut models {
        m.first_scans.sort_by_key(|s| s.ss);
        // Refinement scans execute from highest `al` down to `al = 0`.
        m.refine_scans
            .sort_by_key(|s| (core::cmp::Reverse(s.al), s.ss));
    }

    let cached_scans = is_search.then_some(scans);
    Ok((models, dc_tables, cached_scans))
}

fn estimate_eob_costs(
    counter: Option<&FrequencyCounter>,
    rt: &RateTable,
    blocks: &[[i16; DCT_BLOCK_SIZE]],
    scan: &ProgressiveScan,
) -> (f32, f32) {
    let eob0_len = rt.get_code_length(0x00).max(1) as f32;
    let Some(cnt) = counter else {
        return (eob0_len, eob0_len);
    };
    let mut total_eob_bits = 0.0f64;
    let mut total_eob_huff_bits = 0.0f64;
    let mut total_eob_syms = 0u64;
    for s in 0..=14u8 {
        let sym = s << 4;
        let c = cnt.get_count(sym);
        if c > 0 {
            let len = rt.get_code_length(sym).max(1) as f64;
            total_eob_bits += (c as f64) * (len + s as f64);
            total_eob_huff_bits += (c as f64) * len;
            total_eob_syms += c as u64;
        }
    }
    if total_eob_syms == 0 || blocks.is_empty() {
        return (eob0_len.min(6.0), eob0_len.min(6.0));
    }

    let se = scan.se as usize;
    let thresh = 1i16 << scan.al;
    let mut eob_blocks = 0u64;
    for blk in blocks {
        let last_nz = if scan.ah == 0 {
            blk[se].abs() >= thresh
        } else {
            let v = blk[se].abs();
            v >= thresh && v < (1i16 << scan.ah)
        };
        if !last_nz {
            eob_blocks += 1;
        }
    }
    let avg_per_eob_block = (total_eob_bits / (eob_blocks.max(1) as f64)) as f32;
    let avg_eob_huff = (total_eob_huff_bits / (total_eob_syms as f64)) as f32;
    let eob_allzero = avg_per_eob_block.clamp(0.1, eob0_len.min(6.0));
    let eob_trailing = avg_eob_huff.clamp(eob_allzero, eob0_len.min(6.0));
    (eob_allzero, eob_trailing)
}

// ============================================================================
// AC Trellis: Sequential & Exact Progressive Bitplane Viterbi DP
// ============================================================================

#[inline]
fn compute_block_lambda(
    src: &[i32; DCT_BLOCK_SIZE],
    config: &TrellisConfig,
    base_mode: LambdaBaseMode,
    mult: f32,
) -> f32 {
    let mut norm = 0.0f32;
    for &c in &src[1..] {
        let cf = c as f32;
        norm += cf * cf;
    }
    norm /= 63.0;

    let base = match base_mode {
        LambdaBaseMode::MozjpegNorm => {
            if config.lambda_log_scale2 > 0.0 {
                let s1 = 2.0_f32.powf(config.lambda_log_scale1);
                let s2 = 2.0_f32.powf(config.lambda_log_scale2);
                s1 / (s2 + norm)
            } else {
                2.0_f32.powf(config.lambda_log_scale1 - 12.0)
            }
        }
        LambdaBaseMode::DampedNorm {
            norm_scale,
            shift_s1,
        } => {
            let s1 = 2.0_f32.powf(config.lambda_log_scale1 + shift_s1);
            let s2 = 2.0_f32.powf(config.lambda_log_scale2);
            s1 / (s2 + norm_scale * norm)
        }
    };
    base * mult
}

#[allow(clippy::too_many_arguments)]
fn quantize_plane_sequential_trellis(
    raw_dct: &[Block8x8f],
    out_zigzag: &mut [[i16; DCT_BLOCK_SIZE]],
    qvals_natural: &[u16; DCT_BLOCK_SIZE],
    lambda_tbl: &[f32; DCT_BLOCK_SIZE],
    ac_table: &RateTable,
    config: &TrellisConfig,
    base_mode: LambdaBaseMode,
    lambda_map: Option<&[f32]>,
    max_dc_val: i32,
    use_parallel: bool,
) {
    let eob_len = ac_table.get_code_length(0x00).max(1) as f32;

    let quant_one = |(idx, (raw, dst)): (usize, (&Block8x8f, &mut [i16; DCT_BLOCK_SIZE]))| {
        let mult = lambda_map.and_then(|m| m.get(idx).copied()).unwrap_or(1.0);
        let src = dct_f32_to_i32(&raw.to_array());
        let lambda = compute_block_lambda(&src, config, base_mode, mult);
        let mut natural = [0i16; DCT_BLOCK_SIZE];
        {
            let x = src[0].abs();
            let sign = if src[0] < 0 { -1i16 } else { 1i16 };
            let q = 8 * qvals_natural[0] as i32;
            let qval = ((x + q / 2) / q).min(max_dc_val);
            natural[0] = (qval as i16) * sign;
        }
        trellis_band_al0(
            &src,
            &mut natural,
            qvals_natural,
            lambda_tbl,
            lambda,
            ac_table,
            config.speed_mode,
            1,
            63,
            eob_len,
            eob_len,
        );
        for j in 0..DCT_BLOCK_SIZE {
            dst[JPEG_ZIGZAG_ORDER[j] as usize] = natural[j];
        }
    };

    #[cfg(feature = "parallel")]
    if use_parallel && raw_dct.len() >= 64 {
        raw_dct
            .par_iter()
            .zip(out_zigzag.par_iter_mut())
            .enumerate()
            .for_each(quant_one);
        return;
    }
    let _ = use_parallel;
    for item in raw_dct.iter().zip(out_zigzag.iter_mut()).enumerate() {
        quant_one(item);
    }
}

/// Exact sequential/band Viterbi DP for zigzag positions `ss..=se` at `al == 0`.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
fn trellis_band_al0(
    src: &[i32; DCT_BLOCK_SIZE],
    quantized: &mut [i16; DCT_BLOCK_SIZE],
    qtable: &[u16; DCT_BLOCK_SIZE],
    lambda_tbl: &[f32; DCT_BLOCK_SIZE],
    lambda: f32,
    ac_table: &RateTable,
    speed_mode: TrellisSpeedMode,
    ss: usize,
    se: usize,
    eob_allzero: f32,
    eob_trailing: f32,
) {
    let mut accumulated_zero_dist = [0.0f32; DCT_BLOCK_SIZE];
    let mut accumulated_cost = [f32::MAX; DCT_BLOCK_SIZE];
    let mut run_start = [0usize; DCT_BLOCK_SIZE];

    let mut nonzero_count = 0i32;
    for i in ss..=se {
        let z = JPEG_NATURAL_ORDER[i] as usize;
        let x = src[z].abs();
        let q = 8 * qtable[z] as i32;
        if (x + q / 2) / q > 0 {
            nonzero_count += 1;
        }
    }
    let (max_lookback, max_candidates) = speed_mode.get_limits(nonzero_count);
    let anchor = ss - 1;
    accumulated_zero_dist[anchor] = 0.0;
    accumulated_cost[anchor] = 0.0;

    let zrl_size = ac_table.get_code_length(0xF0);

    for i in ss..=se {
        let z = JPEG_NATURAL_ORDER[i] as usize;
        let x = src[z].abs();
        let sign = if src[z] < 0 { -1i16 } else { 1i16 };
        let q = 8 * qtable[z] as i32;

        let zero_dist = (x as f32).powi(2) * lambda * lambda_tbl[z];
        accumulated_zero_dist[i] = zero_dist + accumulated_zero_dist[i - 1];

        let qval = ((x + q / 2) / q).min(1023);
        if qval == 0 {
            quantized[z] = 0;
            accumulated_cost[i] = f32::MAX;
            run_start[i] = i - 1;
            continue;
        }

        let num_candidates = (jpeg_nbits(qval as i16) as usize).min(max_candidates);
        let mut candidates = [(0i32, 0u8, 0.0f32); 16];
        for k in 0..num_candidates {
            let candidate_val = if k < num_candidates - 1 {
                (2 << k) - 1
            } else {
                qval
            };
            let delta = candidate_val * q - x;
            let dist = (delta as f32).powi(2) * lambda * lambda_tbl[z];
            candidates[k] = (candidate_val, (k + 1) as u8, dist);
        }

        accumulated_cost[i] = f32::MAX;
        let j_start = i.saturating_sub(max_lookback).max(anchor);
        for j in j_start..i {
            if j != anchor {
                let zz = JPEG_NATURAL_ORDER[j] as usize;
                if quantized[zz] == 0 {
                    continue;
                }
            }
            let zero_run = i - 1 - j;
            let zrl_cost = if zero_run >= 16 {
                if zrl_size == 0 {
                    continue;
                }
                (zero_run / 16) * zrl_size as usize
            } else {
                0
            };
            let run_mod_16 = zero_run & 15;
            let zero_run_dist = accumulated_zero_dist[i - 1] - accumulated_zero_dist[j];
            let prev_cost = if j == anchor {
                0.0
            } else {
                accumulated_cost[j]
            };

            for k in 0..num_candidates {
                let (candidate_val, candidate_bits, candidate_dist) = candidates[k];
                let symbol = ((run_mod_16 as u8) << 4) | candidate_bits;
                let code_size = ac_table.get_code_length(symbol);
                if code_size == 0 {
                    continue;
                }
                let rate = code_size as usize + candidate_bits as usize + zrl_cost;
                let cost = rate as f32 + candidate_dist + zero_run_dist + prev_cost;
                if cost < accumulated_cost[i] {
                    quantized[z] = (candidate_val as i16) * sign;
                    accumulated_cost[i] = cost;
                    run_start[i] = j;
                }
            }
        }
    }

    let mut best_cost = accumulated_zero_dist[se] + eob_allzero;
    let mut last_idx = anchor;
    for i in ss..=se {
        let z = JPEG_NATURAL_ORDER[i] as usize;
        if quantized[z] != 0 {
            let tail_zero_dist = accumulated_zero_dist[se] - accumulated_zero_dist[i];
            let mut cost = accumulated_cost[i] + tail_zero_dist;
            if i < se {
                cost += eob_trailing;
            }
            if cost < best_cost {
                best_cost = cost;
                last_idx = i;
            }
        }
    }

    let mut i = se;
    while i >= ss {
        while i > last_idx {
            let z = JPEG_NATURAL_ORDER[i] as usize;
            quantized[z] = 0;
            i -= 1;
        }
        if i >= ss {
            last_idx = run_start[i];
            i -= 1;
        }
    }
}

/// Precomputed per-position block data shared across SA first-pass and refinement stages.
struct SaBlockPrep {
    x_abs: [i32; DCT_BLOCK_SIZE],
    signs: [i16; DCT_BLOCK_SIZE],
    qs: [i32; DCT_BLOCK_SIZE],
    qvals: [i32; DCT_BLOCK_SIZE],
    ws: [f32; DCT_BLOCK_SIZE],
    d0: [f32; DCT_BLOCK_SIZE],
    ref_est: [f32; 8],
}

impl SaBlockPrep {
    #[inline]
    fn best_below(&self, i: usize, al_bound: u8) -> f32 {
        let mut best = self.d0[i];
        for a in 0..al_bound {
            let lo = 1i32 << a;
            let hi = (2i32 << a) - 1;
            let cand = self.qvals[i].clamp(lo, hi);
            let dist = ((cand * self.qs[i] - self.x_abs[i]) as f32).powi(2) * self.ws[i];
            let est = dist + self.ref_est[a as usize];
            if est < best {
                best = est;
            }
        }
        best
    }
}

/// Exact progressive first-pass + refinement Viterbi DP across a full plane.
#[allow(clippy::too_many_arguments)]
fn quantize_plane_progressive_trellis(
    raw_dct: &[Block8x8f],
    out_zigzag: &mut [[i16; DCT_BLOCK_SIZE]],
    qvals_natural: &[u16; DCT_BLOCK_SIZE],
    lambda_tbl: &[f32; DCT_BLOCK_SIZE],
    model: &ComponentProgModel,
    config: &TrellisConfig,
    base_mode: LambdaBaseMode,
    lambda_map: Option<&[f32]>,
    max_dc_val: i32,
    use_parallel: bool,
) {
    let mut ref_est = [4.0f32; 8];
    for r in &model.refine_scans {
        let a = r.al as usize;
        if a < ref_est.len() {
            ref_est[a] = r.table.get_code_length(0x01).max(1) as f32 + 1.0 + (r.al as f32);
        }
    }
    let has_sa = model.first_scans.iter().any(|s| s.al > 0) || !model.refine_scans.is_empty();

    let quant_one = |(idx, (raw, dst)): (usize, (&Block8x8f, &mut [i16; DCT_BLOCK_SIZE]))| {
        let mult = lambda_map.and_then(|m| m.get(idx).copied()).unwrap_or(1.0);
        let src = dct_f32_to_i32(&raw.to_array());
        let lambda = compute_block_lambda(&src, config, base_mode, mult);
        let mut natural = [0i16; DCT_BLOCK_SIZE];

        // DC rounding
        {
            let x = src[0].abs();
            let sign = if src[0] < 0 { -1i16 } else { 1i16 };
            let q = 8 * qvals_natural[0] as i32;
            let qval = ((x + q / 2) / q).min(max_dc_val);
            natural[0] = (qval as i16) * sign;
        }

        // Precompute per-position arrays once when any SA scan is present
        let prep = if has_sa {
            let mut p = SaBlockPrep {
                x_abs: [0; DCT_BLOCK_SIZE],
                signs: [1; DCT_BLOCK_SIZE],
                qs: [1; DCT_BLOCK_SIZE],
                qvals: [0; DCT_BLOCK_SIZE],
                ws: [0.0; DCT_BLOCK_SIZE],
                d0: [0.0; DCT_BLOCK_SIZE],
                ref_est,
            };
            for i in 1..DCT_BLOCK_SIZE {
                let z = JPEG_NATURAL_ORDER[i] as usize;
                let x = src[z].abs();
                let q = 8 * qvals_natural[z] as i32;
                let w = lambda * lambda_tbl[z];
                let qv = ((x + q / 2) / q).min(1023);
                p.x_abs[i] = x;
                p.signs[i] = if src[z] < 0 { -1 } else { 1 };
                p.qs[i] = q;
                p.qvals[i] = qv;
                p.ws[i] = w;
                p.d0[i] = (x as f32).powi(2) * w;
            }
            Some(p)
        } else {
            None
        };

        // Stage 1: All first-pass scans (disjoint bands covering 1..=63)
        for first in &model.first_scans {
            if first.al == 0 {
                trellis_band_al0(
                    &src,
                    &mut natural,
                    qvals_natural,
                    lambda_tbl,
                    lambda,
                    &first.table,
                    config.speed_mode,
                    first.ss as usize,
                    first.se as usize,
                    first.eob_allzero,
                    first.eob_trailing,
                );
            } else if let Some(ref p) = prep {
                trellis_band_sa_first(p, &mut natural, first, config.speed_mode);
            }
        }

        // Stage 2: All refinement scans (descending al, operating directly on each
        // refinement scan's exact [ss..=se] range)
        if let Some(ref p) = prep {
            for ref_scan in &model.refine_scans {
                trellis_band_sa_refine(p, &mut natural, ref_scan);
            }
        }

        for j in 0..DCT_BLOCK_SIZE {
            dst[JPEG_ZIGZAG_ORDER[j] as usize] = natural[j];
        }
    };

    #[cfg(feature = "parallel")]
    if use_parallel && raw_dct.len() >= 64 {
        raw_dct
            .par_iter()
            .zip(out_zigzag.par_iter_mut())
            .enumerate()
            .for_each(quant_one);
        return;
    }
    let _ = use_parallel;
    for item in raw_dct.iter().zip(out_zigzag.iter_mut()).enumerate() {
        quant_one(item);
    }
}

/// First-pass SA Viterbi DP for a spectral band `[ss..=se]` at `al = hi_first.al > 0`.
#[allow(clippy::needless_range_loop)]
fn trellis_band_sa_first(
    p: &SaBlockPrep,
    quantized: &mut [i16; DCT_BLOCK_SIZE],
    hi_first: &ProgScanRate,
    speed_mode: TrellisSpeedMode,
) {
    let ss = hi_first.ss as usize;
    let se = hi_first.se as usize;
    let al_init = hi_first.al;
    let shift_init = al_init as u32;
    let mask_lo = (1i32 << shift_init) - 1;
    let refbits_init = al_init as f32;

    let mut nz_init = 0i32;
    for i in ss..=se {
        if (p.qvals[i] >> shift_init) > 0 {
            nz_init += 1;
        }
    }

    let (max_lookback, max_candidates) = speed_mode.get_limits(nz_init);
    let anchor = ss - 1;
    let mut acc_skip = [0.0f32; DCT_BLOCK_SIZE];
    let mut acc_cost = [f32::MAX; DCT_BLOCK_SIZE];
    let mut run_start = [anchor; DCT_BLOCK_SIZE];
    let mut chosen_full = [0i16; DCT_BLOCK_SIZE];
    acc_skip[anchor] = 0.0;
    acc_cost[anchor] = 0.0;
    let zrl_len = hi_first.table.get_code_length(0xF0);

    for i in ss..=se {
        acc_skip[i] = acc_skip[i - 1] + p.best_below(i, al_init);

        let q_hi_max = p.qvals[i] >> shift_init;
        if q_hi_max == 0 {
            chosen_full[i] = 0;
            acc_cost[i] = f32::MAX;
            run_start[i] = i - 1;
            continue;
        }

        let num_c = (jpeg_nbits(q_hi_max as i16) as usize).min(max_candidates);
        let mut cands = [(0i32, 0u8, 0.0f32); 16];
        for k in 0..num_c {
            let q_hi = if k < num_c - 1 {
                (2 << k) - 1
            } else {
                q_hi_max
            };
            let base_full = q_hi << shift_init;
            let ideal_lo = (p.qvals[i] - base_full).clamp(0, mask_lo);
            let full_val = (base_full + ideal_lo).min(1023);
            let delta = full_val * p.qs[i] - p.x_abs[i];
            let dist = (delta as f32).powi(2) * p.ws[i];
            cands[k] = (full_val, (k + 1) as u8, dist);
        }

        acc_cost[i] = f32::MAX;
        let j_start = i.saturating_sub(max_lookback).max(anchor);
        for j in j_start..i {
            if j != anchor && chosen_full[j] == 0 {
                continue;
            }
            let zero_run = i - 1 - j;
            let zrl_cost = if zero_run >= 16 {
                if zrl_len == 0 {
                    continue;
                }
                (zero_run / 16) * zrl_len as usize
            } else {
                0
            };
            let run_mod_16 = zero_run & 15;
            let skip_run_cost = acc_skip[i - 1] - acc_skip[j];
            let prev_cost = if j == anchor { 0.0 } else { acc_cost[j] };

            for k in 0..num_c {
                let (full_val, cbits, cdist) = cands[k];
                let sym = ((run_mod_16 as u8) << 4) | cbits;
                let clen = hi_first.table.get_code_length(sym);
                if clen == 0 {
                    continue;
                }
                let rate = (clen as usize + cbits as usize + zrl_cost) as f32 + refbits_init;
                let cost = rate + cdist + skip_run_cost + prev_cost;
                if cost < acc_cost[i] {
                    chosen_full[i] = (full_val as i16) * p.signs[i];
                    acc_cost[i] = cost;
                    run_start[i] = j;
                }
            }
        }
    }

    let mut best_cost = acc_skip[se] + hi_first.eob_allzero;
    let mut last_idx = anchor;
    for i in ss..=se {
        if chosen_full[i] != 0 {
            let tail_skip = acc_skip[se] - acc_skip[i];
            let mut cost = acc_cost[i] + tail_skip;
            if i < se {
                cost += hi_first.eob_trailing;
            }
            if cost < best_cost {
                best_cost = cost;
                last_idx = i;
            }
        }
    }

    let mut i = se;
    while i >= ss {
        while i > last_idx {
            let z = JPEG_NATURAL_ORDER[i] as usize;
            quantized[z] = 0;
            i -= 1;
        }
        if i >= ss {
            let z = JPEG_NATURAL_ORDER[i] as usize;
            quantized[z] = chosen_full[i];
            last_idx = run_start[i];
            i -= 1;
        }
    }
}

/// Refinement-pass SA Viterbi DP for `[ref_scan.ss..=ref_scan.se]` at `ref_scan.al`.
#[allow(clippy::needless_range_loop)]
fn trellis_band_sa_refine(
    p: &SaBlockPrep,
    quantized: &mut [i16; DCT_BLOCK_SIZE],
    ref_scan: &ProgScanRate,
) {
    let ss = ref_scan.ss as usize;
    let se = ref_scan.se as usize;
    let al = ref_scan.al;
    let lo_val = 1i32 << al;
    let hi_val = (2i32 << al) - 1;
    let extra_bits = 1usize + (al as usize); // 1 sign bit + `al` subsequent lower refbits

    let mut rem = [0usize; DCT_BLOCK_SIZE];
    let mut n_rem = 0usize;
    for i in ss..=se {
        let z = JPEG_NATURAL_ORDER[i] as usize;
        if quantized[z] == 0 {
            rem[n_rem] = i;
            n_rem += 1;
        }
    }
    if n_rem == 0 {
        return;
    }

    let mut acc_skip = [0.0f32; DCT_BLOCK_SIZE + 1];
    let mut acc_cost = [f32::MAX; DCT_BLOCK_SIZE + 1];
    let mut prev_p = [0usize; DCT_BLOCK_SIZE + 1];
    let mut chosen = [0i16; DCT_BLOCK_SIZE + 1];
    acc_skip[0] = 0.0;
    acc_cost[0] = 0.0;
    let zrl_len = ref_scan.table.get_code_length(0xF0);

    for idx in 1..=n_rem {
        let i = rem[idx - 1];
        acc_skip[idx] = acc_skip[idx - 1] + p.best_below(i, al);
        if p.qvals[i] < lo_val {
            chosen[idx] = 0;
            continue;
        }
        let cand_abs = p.qvals[i].clamp(lo_val, hi_val);
        let cdist = ((cand_abs * p.qs[i] - p.x_abs[i]) as f32).powi(2) * p.ws[i];
        let val = (cand_abs as i16) * p.signs[i];

        for q_prev in 0..idx {
            if q_prev != 0 && chosen[q_prev] == 0 {
                continue;
            }
            let zero_run = idx - 1 - q_prev;
            let zrl_cost = if zero_run >= 16 {
                if zrl_len == 0 {
                    continue;
                }
                (zero_run / 16) * zrl_len as usize
            } else {
                0
            };
            let sym = (((zero_run & 15) as u8) << 4) | 1;
            let clen = ref_scan.table.get_code_length(sym);
            if clen == 0 {
                continue;
            }
            let rate = (clen as usize + extra_bits + zrl_cost) as f32;
            let cost = rate + cdist + (acc_skip[idx - 1] - acc_skip[q_prev]) + acc_cost[q_prev];
            if cost < acc_cost[idx] {
                acc_cost[idx] = cost;
                prev_p[idx] = q_prev;
                chosen[idx] = val;
            }
        }
    }

    let mut best_cost = acc_skip[n_rem] + ref_scan.eob_allzero;
    let mut last_p = 0usize;
    for idx in 1..=n_rem {
        if chosen[idx] != 0 {
            let tail_skip = acc_skip[n_rem] - acc_skip[idx];
            let mut cost = acc_cost[idx] + tail_skip;
            if idx < n_rem {
                cost += ref_scan.eob_trailing;
            }
            if cost < best_cost {
                best_cost = cost;
                last_p = idx;
            }
        }
    }

    let mut idx = last_p;
    while idx > 0 {
        let i = rem[idx - 1];
        let z = JPEG_NATURAL_ORDER[i] as usize;
        quantized[z] = chosen[idx];
        idx = prev_p[idx];
    }
}

// ============================================================================
// Empirical DC Trellis (Row-by-Row In-Place)
// ============================================================================

#[allow(clippy::too_many_arguments)]
fn run_dc_trellis_plane(
    raw_dct: &[Block8x8f],
    blocks_zigzag: &mut [[i16; DCT_BLOCK_SIZE]],
    blocks_w: usize,
    dc_quantval: u16,
    dc_table: &RateTable,
    lambda_log_scale1: f32,
    lambda_log_scale2: f32,
    delta_dc_weight: f32,
    max_dc_val: i32,
) -> Result<()> {
    let n = blocks_zigzag.len().min(raw_dct.len());
    if n == 0 || blocks_w == 0 || dc_quantval == 0 {
        return Ok(());
    }
    let blocks_h = (n + blocks_w - 1) / blocks_w;
    let mut cur_raw = try_vec(blocks_w, 0i32, "rdo dc cur_raw")?;
    let mut above_raw = if delta_dc_weight > 0.0 {
        try_vec(blocks_w, 0i32, "rdo dc above_raw")?
    } else {
        Vec::new()
    };
    let mut above_q = if delta_dc_weight > 0.0 {
        try_vec(blocks_w, 0i16, "rdo dc above_q")?
    } else {
        Vec::new()
    };
    let mut scratch = crate::encode::trellis::dc::DcTrellisScratch::new();
    let mut last_dc: i16 = 0;

    for row in 0..blocks_h {
        let row_start = row * blocks_w;
        let row_end = (row_start + blocks_w).min(n);
        let row_len = row_end - row_start;

        for bi in 0..row_len {
            cur_raw[bi] = (raw_dct[row_start + bi].rows[0][0] * 64.0).round() as i32;
        }

        let above_data = if delta_dc_weight > 0.0 && row > 0 {
            Some((&above_raw[..row_len], &above_q[..row_len]))
        } else {
            None
        };

        last_dc = crate::encode::trellis::dc::dc_trellis_optimize_row_direct(
            &cur_raw[..row_len],
            &mut blocks_zigzag[row_start..row_end],
            dc_quantval,
            dc_table,
            last_dc,
            lambda_log_scale1,
            lambda_log_scale2,
            delta_dc_weight,
            above_data,
            max_dc_val,
            &mut scratch,
        )?;

        if delta_dc_weight > 0.0 {
            above_raw[..row_len].copy_from_slice(&cur_raw[..row_len]);
            for bi in 0..row_len {
                above_q[bi] = blocks_zigzag[row_start + bi][0];
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::encoder_config::EncoderConfig;
    use crate::encode::encoder_types::{
        ChromaSubsampling, OptimizationPreset, PixelLayout, ProgressiveScanMode, XybSubsampling,
    };

    fn make_patterned_rgb(w: usize, h: usize) -> Vec<u8> {
        let mut rgb = Vec::with_capacity(w * h * 3);
        for y in 0..h {
            for x in 0..w {
                if x < w / 3 && y < h / 3 {
                    rgb.extend_from_slice(&[136, 144, 152]);
                } else {
                    let hash = (((x as u32).wrapping_mul(31337) ^ (y as u32).wrapping_mul(7919))
                        .wrapping_mul(2654435761)
                        >> 24) as u8;
                    let r = ((x * 255 / w) as u8).saturating_add(hash & 0x3F);
                    let g = ((y * 255 / h) as u8).saturating_add((hash >> 2) & 0x3F);
                    let b = (((x + y) * 127 / (w + h)) as u8).saturating_add((hash >> 4) & 0x3F);
                    rgb.extend_from_slice(&[r, g, b]);
                }
            }
        }
        rgb
    }

    fn verify_decode(jpg: &[u8], w: u32, h: u32, label: &str) {
        let dec = crate::decode::Decoder::new()
            .decode(jpg, enough::Unstoppable)
            .unwrap_or_else(|e| panic!("{label}: zenjpeg decode failed: {e}"));
        assert_eq!((dec.width, dec.height), (w, h), "{label}: dimensions");

        #[cfg(not(target_arch = "wasm32"))]
        {
            use zune_core::bytestream::ZCursor;
            use zune_core::colorspace::ColorSpace;
            use zune_core::options::DecoderOptions;
            let opts = DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::RGB);
            let mut zd = zune_jpeg::JpegDecoder::new_with_options(ZCursor::new(jpg), opts);
            let px = zd
                .decode()
                .unwrap_or_else(|e| panic!("{label}: zune-jpeg decode failed: {e:?}"));
            assert_eq!(px.len(), (w * h * 3) as usize, "{label}: zune pixel len");
        }
    }

    #[test]
    fn hybrid_presets_roundtrip_and_deterministic_parallel() {
        let (w, h) = (256u32, 160u32);
        let rgb = make_patterned_rgb(w as usize, h as usize);

        let cases = [
            (
                "hybrid_prog_420",
                EncoderConfig::ycbcr(75.0, ChromaSubsampling::Quarter)
                    .optimization(OptimizationPreset::HybridProgressive),
            ),
            (
                "hybrid_prog_444",
                EncoderConfig::ycbcr(75.0, ChromaSubsampling::None)
                    .optimization(OptimizationPreset::HybridProgressive),
            ),
            (
                "hybrid_max_420",
                EncoderConfig::ycbcr(75.0, ChromaSubsampling::Quarter)
                    .optimization(OptimizationPreset::HybridMaxCompression),
            ),
            (
                "hybrid_max_444",
                EncoderConfig::ycbcr(75.0, ChromaSubsampling::None)
                    .optimization(OptimizationPreset::HybridMaxCompression),
            ),
            (
                "hybrid_base_420",
                EncoderConfig::ycbcr(75.0, ChromaSubsampling::Quarter)
                    .optimization(OptimizationPreset::HybridBaseline),
            ),
            (
                "auto_opt_420",
                EncoderConfig::ycbcr(80.0, ChromaSubsampling::Quarter).auto_optimize(true),
            ),
            (
                "auto_opt_444",
                EncoderConfig::ycbcr(80.0, ChromaSubsampling::None).auto_optimize(true),
            ),
            (
                "hybrid_prog_xyb",
                EncoderConfig::xyb(75.0, XybSubsampling::BQuarter)
                    .optimization(OptimizationPreset::HybridProgressive),
            ),
            (
                "hybrid_max_xyb",
                EncoderConfig::xyb(75.0, XybSubsampling::BQuarter)
                    .optimization(OptimizationPreset::HybridMaxCompression),
            ),
            (
                "hybrid_prog_gray",
                EncoderConfig::grayscale(75.0).optimization(OptimizationPreset::HybridProgressive),
            ),
            (
                "hybrid_max_gray",
                EncoderConfig::grayscale(75.0)
                    .optimization(OptimizationPreset::HybridMaxCompression),
            ),
        ];

        for (label, cfg) in cases {
            let seq = cfg
                .encode_bytes(&rgb, w, h, PixelLayout::Rgb8Srgb)
                .unwrap_or_else(|e| panic!("{label}: seq encode failed: {e}"));
            verify_decode(&seq, w, h, label);

            #[cfg(feature = "parallel")]
            {
                let par_cfg = cfg
                    .clone()
                    .parallel(crate::encode::encoder_types::ParallelEncoding::Auto);
                let par = par_cfg
                    .encode_bytes(&rgb, w, h, PixelLayout::Rgb8Srgb)
                    .unwrap_or_else(|e| panic!("{label}: par encode failed: {e}"));
                assert_eq!(
                    seq, par,
                    "{label}: parallel encode must be byte-identical to sequential"
                );
            }
        }
    }

    #[test]
    fn r8_progressive_search_is_lossless_and_le_default() {
        let (w, h) = (192u32, 128u32);
        let rgb = make_patterned_rgb(w as usize, h as usize);

        for sub in [ChromaSubsampling::Quarter, ChromaSubsampling::None] {
            let def_cfg = EncoderConfig::ycbcr(80.0, sub).progressive(true);
            let srch_cfg =
                EncoderConfig::ycbcr(80.0, sub).progressive(ProgressiveScanMode::ProgressiveSearch);

            let def = def_cfg
                .encode_bytes(&rgb, w, h, PixelLayout::Rgb8Srgb)
                .unwrap();
            let srch = srch_cfg
                .encode_bytes(&rgb, w, h, PixelLayout::Rgb8Srgb)
                .unwrap();

            assert!(
                srch.len() <= def.len(),
                "{sub:?}: search ({}) must be <= default ({})",
                srch.len(),
                def.len()
            );

            // Lossless scan search must decode to identical pixels
            let px_def = crate::decode::Decoder::new()
                .decode(&def, enough::Unstoppable)
                .unwrap()
                .pixels_u8()
                .unwrap()
                .to_vec();
            let px_srch = crate::decode::Decoder::new()
                .decode(&srch, enough::Unstoppable)
                .unwrap()
                .pixels_u8()
                .unwrap()
                .to_vec();
            assert_eq!(px_def, px_srch, "{sub:?}: scan search must be lossless");
        }
    }

    #[test]
    fn three_band_sa_refinement_prices_full_1_to_63_range() {
        // Verify that a 3-band R8 script ({1,2},{3,9},{10,63} at al=2 + 1..63 refines)
        // produces nonzero low-magnitude coefficients in all 3 bands during refinement.
        let mut raw = [0.0f32; DCT_BLOCK_SIZE];
        raw[0] = 50.0;
        // Put coefficients whose rounded quantized magnitude is 1 (survives only al=0 refine)
        // across all three bands: zz=2 (band 1), zz=5 (band 2), zz=15 (band 3).
        let qvals = [8u16; DCT_BLOCK_SIZE];
        for zz in [2usize, 5, 15] {
            let nat = JPEG_NATURAL_ORDER[zz] as usize;
            // dct_f32_to_i32 multiplies by 64, and q = 8*8 = 64 -> raw=1.0 -> qval=1
            raw[nat] = 1.0;
        }
        // Plus one large coeff in band 2 (zz=4 -> qval=6, survives al=2 first pass)
        raw[JPEG_NATURAL_ORDER[4] as usize] = 6.0;

        let mut rows = [[0.0f32; 8]; 8];
        for r in 0..8 {
            rows[r].copy_from_slice(&raw[r * 8..(r + 1) * 8]);
        }
        let block = Block8x8f { rows };

        let ac_tbl = RateTable::standard_luma_ac();
        let model = ComponentProgModel {
            first_scans: alloc::vec![
                ProgScanRate {
                    ss: 1,
                    se: 2,
                    al: 2,
                    table: ac_tbl.clone(),
                    eob_allzero: 1.0,
                    eob_trailing: 2.0,
                },
                ProgScanRate {
                    ss: 3,
                    se: 9,
                    al: 2,
                    table: ac_tbl.clone(),
                    eob_allzero: 1.0,
                    eob_trailing: 2.0,
                },
                ProgScanRate {
                    ss: 10,
                    se: 63,
                    al: 2,
                    table: ac_tbl.clone(),
                    eob_allzero: 1.0,
                    eob_trailing: 2.0,
                },
            ],
            refine_scans: alloc::vec![
                ProgScanRate {
                    ss: 1,
                    se: 63,
                    al: 1,
                    table: ac_tbl.clone(),
                    eob_allzero: 1.0,
                    eob_trailing: 2.0,
                },
                ProgScanRate {
                    ss: 1,
                    se: 63,
                    al: 0,
                    table: ac_tbl,
                    eob_allzero: 1.0,
                    eob_trailing: 2.0,
                },
            ],
        };

        let cfg = TrellisConfig::default()
            .lambda_scales(16.0, 16.5) // high lambda so distortion beats rate -> keeps qval=1
            .speed_mode(TrellisSpeedMode::Thorough);
        let lambda_tbl = build_lambda_tbl(&qvals, 0.0);
        let mut out = [[0i16; DCT_BLOCK_SIZE]];
        quantize_plane_progressive_trellis(
            &[block],
            &mut out,
            &qvals,
            &lambda_tbl,
            &model,
            &cfg,
            LambdaBaseMode::MozjpegNorm,
            None,
            1023,
            false,
        );
        assert_eq!(out[0][4], 6, "first-pass al=2 coeff in band 2 preserved");
        assert_eq!(out[0][2], 1, "al=0 refine in band 1 (1..2) must run");
        assert_eq!(out[0][5], 1, "al=0 refine in band 2 (3..9) must run");
        assert_eq!(out[0][15], 1, "al=0 refine in band 3 (10..63) must run");
    }

    /// Reference full-plane T2 open-loop lambda map for bit-exact equivalence verification.
    #[allow(clippy::too_many_arguments)]
    fn reference_full_plane_t2(
        use_xyb: bool,
        y_bw: usize,
        y_bh: usize,
        cb_bw: usize,
        cb_bh: usize,
        cr_bw: usize,
        cr_bh: usize,
        y_raw: &[Block8x8f],
        cb_raw: &[Block8x8f],
        cr_raw: &[Block8x8f],
        y_quant: &QuantTable,
        cb_quant: &QuantTable,
        cr_quant: &QuantTable,
    ) -> Vec<f32> {
        let tok = IdctToken::summon();
        let oy: Vec<[f32; 64]> = y_raw.iter().map(|b| idct_raw_block(tok, b)).collect();
        let ocb: Vec<[f32; 64]> = cb_raw.iter().map(|b| idct_raw_block(tok, b)).collect();
        let ocr: Vec<[f32; 64]> = cr_raw.iter().map(|b| idct_raw_block(tok, b)).collect();

        let (lo, c0, c1, ls1, c0s1, c1s1, c0bw, c0bh, c1bw, c1bh, n0, n1) = if use_xyb {
            (
                &ocb,
                &oy,
                &ocr,
                cb_quant.values[1].max(1) as f32,
                y_quant.values[1].max(1) as f32,
                cr_quant.values[1].max(1) as f32,
                y_bw,
                y_bh,
                cr_bw,
                cr_bh,
                XYB_X_NEUTRAL,
                XYB_B_NEUTRAL,
            )
        } else {
            (
                &oy,
                &ocb,
                &ocr,
                y_quant.values[1].max(1) as f32,
                cb_quant.values[1].max(1) as f32,
                cr_quant.values[1].max(1) as f32,
                cb_bw,
                cb_bh,
                cr_bw,
                cr_bh,
                0.0f32,
                0.0f32,
            )
        };

        let nb = y_bw * y_bh;
        let mut min_q = vec![0.0f32; nb];
        let mut mean_q = vec![0.0f32; nb];
        let mut sat_smooth = vec![0.0f32; nb];
        for by in 0..y_bh {
            for bx in 0..y_bw {
                let idx = by * y_bw + bx;
                let blk = &lo[idx];
                let mut qa = [0.0f32; 4];
                for qy in 0..2 {
                    for qx in 0..2 {
                        let mut sum = 0.0f32;
                        for dy in 0..4 {
                            let y = qy * 4 + dy;
                            let y1 = (y + 1).min(7);
                            for dx in 0..4 {
                                let x = qx * 4 + dx;
                                let x1 = (x + 1).min(7);
                                let v = blk[y * 8 + x];
                                sum += (v - blk[y * 8 + x1]).abs() + (v - blk[y1 * 8 + x]).abs();
                            }
                        }
                        qa[qy * 2 + qx] = sum * (1.0 / 32.0) / ls1;
                    }
                }
                min_q[idx] = qa[0].min(qa[1]).min(qa[2]).min(qa[3]);
                mean_q[idx] = 0.25 * (qa[0] + qa[1] + qa[2] + qa[3]);

                let c0_sx = c0bw < y_bw;
                let c0_sy = c0bh < y_bh;
                let cbx0 = if c0_sx {
                    (bx / 2).min(c0bw - 1)
                } else {
                    bx.min(c0bw - 1)
                };
                let cby0 = if c0_sy {
                    (by / 2).min(c0bh - 1)
                } else {
                    by.min(c0bh - 1)
                };
                let (m0, g0) = chroma_quadrant_mag_grad_from_block(
                    &c0[cby0 * c0bw + cbx0],
                    c0_sx,
                    c0_sy,
                    bx,
                    by,
                    n0,
                    c0s1,
                );
                let c1_sx = c1bw < y_bw;
                let c1_sy = c1bh < y_bh;
                let cbx1 = if c1_sx {
                    (bx / 2).min(c1bw - 1)
                } else {
                    bx.min(c1bw - 1)
                };
                let cby1 = if c1_sy {
                    (by / 2).min(c1bh - 1)
                } else {
                    by.min(c1bh - 1)
                };
                let (m1, g1) = chroma_quadrant_mag_grad_from_block(
                    &c1[cby1 * c1bw + cbx1],
                    c1_sx,
                    c1_sy,
                    bx,
                    by,
                    n1,
                    c1s1,
                );
                let am = 0.5 * (m0 + m1);
                let ag = 0.5 * (g0 + g1);
                sat_smooth[idx] = (am / (am + 8.0)) * (1.0 / (1.0 + ag * ag));
            }
        }
        let mut raw = vec![0.0f64; nb];
        let (mut asum, mut an, mut all) = (0.0f64, 0usize, 0.0f64);
        for by in 0..y_bh {
            let y0 = by.saturating_sub(1);
            let y1 = (by + 1).min(y_bh - 1);
            for bx in 0..y_bw {
                let x0 = bx.saturating_sub(1);
                let x1 = (bx + 1).min(y_bw - 1);
                let (mut s_min, mut s_sum, mut s_cnt) = (f32::INFINITY, 0.0f32, 0.0f32);
                for yy in y0..=y1 {
                    for xx in x0..=x1 {
                        let v = mean_q[yy * y_bw + xx];
                        if v < s_min {
                            s_min = v;
                        }
                        s_sum += v;
                        s_cnt += 1.0;
                    }
                }
                let idx = by * y_bw + bx;
                let act = 0.45 * min_q[idx] + 0.30 * s_min + 0.25 * (s_sum / s_cnt);
                let sc =
                    -((act as f64) + 0.25).ln() + (T2_SAT_WEIGHT as f64) * (sat_smooth[idx] as f64);
                raw[idx] = sc;
                all += sc;
                if mean_q[idx] > 0.05 {
                    asum += sc;
                    an += 1;
                }
            }
        }
        let mraw = if an > 0 {
            asum / an as f64
        } else {
            all / nb as f64
        };
        let st = (T2_STRENGTH as f64) * coarse_damp_factor(ls1, true);
        raw.into_iter()
            .map(|r| (st * (r - mraw)).clamp(-0.85, 0.85).exp() as f32)
            .collect()
    }

    /// Reference full-plane T1 seam+interior error map for bit-exact equivalence verification.
    fn reference_full_plane_t1_errors(
        raw_dct: &[Block8x8f],
        quant_blocks: &[[i16; DCT_BLOCK_SIZE]],
        qvals: &[u16; DCT_BLOCK_SIZE],
        vis_luma: Option<(&[Block8x8f], f32)>,
        bw: usize,
        bh: usize,
        s1: f32,
        seam_weight: f64,
    ) -> Vec<f64> {
        let tok = IdctToken::summon();
        let orig: Vec<[f32; 64]> = raw_dct.iter().map(|b| idct_raw_block(tok, b)).collect();
        let rec: Vec<[f32; 64]> = quant_blocks
            .iter()
            .map(|q| idct_quantized_zigzag(tok, q, qvals))
            .collect();
        let vis: Option<Vec<[f32; 64]>> = vis_luma.map(|(lr, ls1)| {
            lr.iter()
                .map(|b| block_visibility(&idct_raw_block(tok, b), ls1, 2.0))
                .collect()
        });
        let inv_s1_sq = 1.0 / ((s1 as f64) * (s1 as f64)).max(1e-12);
        let inv_seam_tau = 1.0 / ((1.5 * s1) as f64).max(1e-6);
        let mut err = vec![0.0f64; bw * bh];

        for by in 0..bh {
            for bx in 0..bw {
                let idx = by * bw + bx;
                let ob = &orig[idx];
                let rb = &rec[idx];
                let v = vis.as_ref().map(|vv| &vv[idx]);
                let mut acc = 0.0f64;
                for k in 0..64 {
                    let e = (ob[k] - rb[k]) as f64;
                    let e2 = (e * e * inv_s1_sq).min(64.0);
                    acc += v.map_or(1.0, |va| va[k] as f64) * (e2 + 0.15 * e2 * e2);
                }
                if bx > 0 {
                    let mut hs = 0.0f64;
                    for y in 0..8 {
                        let oa = orig[idx - 1][y * 8 + 7];
                        let obp = ob[y * 8];
                        hs += seam_pair_cost(
                            oa,
                            obp,
                            oa - rec[idx - 1][y * 8 + 7],
                            obp - rb[y * 8],
                            inv_seam_tau,
                            inv_s1_sq,
                        );
                    }
                    let term = seam_weight * hs;
                    acc += term;
                    err[idx - 1] += term;
                }
                err[idx] += acc;
            }
        }
        for by in 1..bh {
            for bx in 0..bw {
                let ia = (by - 1) * bw + bx;
                let ib = by * bw + bx;
                let mut vs = 0.0f64;
                for x in 0..8 {
                    let oa = orig[ia][7 * 8 + x];
                    let ob = orig[ib][x];
                    vs += seam_pair_cost(
                        oa,
                        ob,
                        oa - rec[ia][7 * 8 + x],
                        ob - rec[ib][x],
                        inv_seam_tau,
                        inv_s1_sq,
                    );
                }
                let term = seam_weight * vs;
                err[ia] += term;
                err[ib] += term;
            }
        }
        err
    }

    #[test]
    fn streaming_t1_and_t2_match_full_plane_bit_exact() {
        // Cross multiple T1_STRIPE_ROWS (16) boundaries with odd width/height
        // across 4:2:0, 4:4:4, 4:2:2, and XYB BQuarter geometries!
        let make_blocks = |n: usize, seed: u32| -> (Vec<Block8x8f>, Vec<[i16; 64]>) {
            let mut raw = Vec::with_capacity(n);
            let mut quant = Vec::with_capacity(n);
            for i in 0..n {
                let mut rows = [[0.0f32; 8]; 8];
                let mut q = [0i16; 64];
                for r in 0..8 {
                    for c in 0..8 {
                        let h = ((i as u32).wrapping_mul(2654435761)
                            ^ ((r * 8 + c) as u32).wrapping_mul(2246822519)
                            ^ seed)
                            >> 20;
                        let val = ((h as i32) - 2048) as f32 / (12.0 + (r + c) as f32 * 8.0);
                        rows[r][c] = val;
                        let zz = JPEG_ZIGZAG_ORDER[r * 8 + c] as usize;
                        q[zz] = (val * 8.0 / 14.0).round() as i16;
                    }
                }
                raw.push(Block8x8f { rows });
                quant.push(q);
            }
            (raw, quant)
        };

        let qt_y = QuantTable {
            values: [14; 64],
            precision: 0,
        };
        let qt_c = QuantTable {
            values: [18; 64],
            precision: 0,
        };

        let configs = [
            // (label, use_xyb, y_bw, y_bh, cb_bw, cb_bh, cr_bw, cr_bh)
            ("ycbcr_420_odd_multistripe", false, 19, 37, 10, 19, 10, 19),
            ("ycbcr_444_multistripe", false, 17, 35, 17, 35, 17, 35),
            ("ycbcr_422_multistripe", false, 18, 33, 9, 33, 9, 33),
            ("xyb_bquarter_multistripe", true, 19, 37, 19, 37, 10, 19),
        ];

        for (label, use_xyb, y_bw, y_bh, cb_bw, cb_bh, cr_bw, cr_bh) in configs {
            let (y_raw, y_q) = make_blocks(y_bw * y_bh, 0x1111);
            let (cb_raw, cb_q) = make_blocks(cb_bw * cb_bh, 0x2222);
            let (cr_raw, cr_q) = make_blocks(cr_bw * cr_bh, 0x3333);

            // 1. T2 streaming vs full-plane (1T and MT)
            let ref_t2 = reference_full_plane_t2(
                use_xyb, y_bw, y_bh, cb_bw, cb_bh, cr_bw, cr_bh, &y_raw, &cb_raw, &cr_raw, &qt_y,
                &qt_c, &qt_c,
            );
            for par in [false, true] {
                let st_t2 = build_t2_imij_lambda_map_streaming(
                    use_xyb,
                    true,
                    y_bw,
                    y_bh,
                    cb_bw,
                    cb_bh,
                    cr_bw,
                    cr_bh,
                    &y_raw,
                    &cb_raw,
                    &cr_raw,
                    &qt_y,
                    &qt_c,
                    &qt_c,
                    T2_STRENGTH,
                    T2_SAT_WEIGHT,
                    true,
                    par,
                )
                .unwrap();
                assert_eq!(st_t2.len(), ref_t2.len());
                for (i, (&a, &b)) in st_t2.iter().zip(ref_t2.iter()).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "{label} T2 par={par} mismatch at block {i}: {a} vs {b}"
                    );
                }
            }

            // 2. T1 streaming vs full-plane (SelfLuma, ExternalLuma, None; 1T and MT)
            let ref_t1_y = reference_full_plane_t1_errors(
                &y_raw,
                &y_q,
                &qt_y.values,
                Some((&y_raw, 14.0)),
                y_bw,
                y_bh,
                14.0,
                T1_SEAM_WEIGHT,
            );
            let ref_t1_c = reference_full_plane_t1_errors(
                &cb_raw,
                &cb_q,
                &qt_c.values,
                if cb_bw == y_bw && cb_bh == y_bh {
                    Some((&y_raw, 14.0))
                } else {
                    None
                },
                cb_bw,
                cb_bh,
                18.0,
                T1_SEAM_WEIGHT,
            );
            let _ = &cr_q;
            for par in [false, true] {
                let st_y = plane_closed_loop_errors_streaming(
                    &y_raw,
                    &y_q,
                    &qt_y.values,
                    VisMode::SelfLuma { luma_s1: 14.0 },
                    y_bw,
                    y_bh,
                    14.0,
                    T1_SEAM_WEIGHT,
                    par,
                )
                .unwrap();
                for (i, (&a, &b)) in st_y.iter().zip(ref_t1_y.iter()).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "{label} T1 Y par={par} mismatch at block {i}: {a} vs {b}"
                    );
                }

                let c_vis = if cb_bw == y_bw && cb_bh == y_bh {
                    VisMode::ExternalLuma {
                        luma_raw: &y_raw,
                        luma_s1: 14.0,
                    }
                } else {
                    VisMode::None
                };
                let st_c = plane_closed_loop_errors_streaming(
                    &cb_raw,
                    &cb_q,
                    &qt_c.values,
                    c_vis,
                    cb_bw,
                    cb_bh,
                    18.0,
                    T1_SEAM_WEIGHT,
                    par,
                )
                .unwrap();
                for (i, (&a, &b)) in st_c.iter().zip(ref_t1_c.iter()).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "{label} T1 Cb par={par} mismatch at block {i}: {a} vs {b}"
                    );
                }
            }
        }
    }

    #[cfg(feature = "__diagnostics")]
    #[test]
    fn diagnostics_capture_matches_hybrid_presets() {
        let (w, h) = (64u32, 48u32);
        let rgb = make_patterned_rgb(w as usize, h as usize);
        for preset in [
            OptimizationPreset::HybridBaseline,
            OptimizationPreset::HybridProgressive,
            OptimizationPreset::HybridMaxCompression,
        ] {
            let cfg = EncoderConfig::ycbcr(75.0, ChromaSubsampling::Quarter).optimization(preset);
            let mut enc = cfg.encode_from_bytes(w, h, PixelLayout::Rgb8Srgb).unwrap();
            let handle = enc.capture_diagnostics().unwrap();
            enc.push_packed(&rgb, enough::Unstoppable).unwrap();
            let jpg = enc.finish().unwrap();
            let diag = handle.try_take().unwrap();
            let decoded = crate::decode::Decoder::new()
                .decode_coefficients(&jpg, enough::Unstoppable)
                .unwrap();
            assert_eq!(diag.components.len(), decoded.components.len());
            for (comp, ref_comp) in diag.components.iter().zip(&decoded.components) {
                let (bw, bh) = comp.block_grid;
                for y in 0..bh as usize {
                    for x in 0..bw as usize {
                        assert_eq!(
                            &comp.blocks[y * bw as usize + x].coef_levels[..],
                            ref_comp.block_at(x, y),
                            "{preset:?} comp {} block ({x},{y})",
                            comp.component_id
                        );
                    }
                }
            }
        }
    }
}
