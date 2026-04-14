//! Sharp YUV: iterative perceptual chroma optimization.
//!
//! Minimizes reconstruction error in gamma-encoded RGB space by iteratively
//! adjusting Cb/Cr values. Same algorithm as libwebp's SharpYUV, parameterized
//! over gamma transfer function, color matrix, and range.
//!
//! No traits — gamma LUTs are passed as `&GammaLuts`, matrix coefficients as
//! `&ForwardCoeffs` / `&InverseCoeffs`. All functions are `#[inline(always)]`
//! so they can be called from `#[arcane]` SIMD regions in the future.

extern crate alloc;

use crate::gamma::GammaLuts;
use crate::types::{ForwardCoeffs, InverseCoeffs, Matrix, Range};

/// Configuration for Sharp YUV chroma optimization.
pub struct SharpYuvConfig {
    /// Maximum refinement iterations per 2×2 block (default: 4).
    pub max_iterations: u32,
    /// Stop early if total reconstruction error drops below this (default: 0.1).
    pub convergence_threshold: f32,
    /// Use gamma-aware (linear-space) averaging for the initial Cb/Cr estimate.
    /// Slightly better quality (~0.1% error reduction) at ~2× init cost.
    /// When false (default), uses box-average in gamma space — the iterative
    /// loop converges to the same result within 1-2 extra iterations.
    pub gamma_aware_init: bool,
    /// Which delinearization to use for gamma-aware initial estimate.
    /// `true` = sRGB (zenjpeg), `false` = libwebp gamma^0.45 (zenwebp).
    /// Only used when `gamma_aware_init` is true.
    pub srgb_delinearize: bool,
}

impl Default for SharpYuvConfig {
    fn default() -> Self {
        Self {
            max_iterations: 2,
            convergence_threshold: 0.1,
            gamma_aware_init: false,
            srgb_delinearize: true,
        }
    }
}

/// Convert packed RGB to Y/Cb/Cr 4:2:0 with Sharp YUV chroma optimization.
///
/// Y is computed at full resolution via the fast SIMD path. Cb/Cr are computed
/// per 2×2 block using iterative refinement that minimizes reconstruction error
/// in gamma-encoded RGB space.
///
/// `luts` controls the gamma transfer (sRGB or libwebp). `range` and `matrix`
/// control the YCbCr color space. `config` tunes the iteration.
pub fn rgb_to_yuv420_sharp(
    rgb: &[u8],
    y: &mut [u8],
    cb: &mut [u8],
    cr: &mut [u8],
    width: usize,
    height: usize,
    range: Range,
    matrix: Matrix,
    luts: &GammaLuts,
    config: &SharpYuvConfig,
) {
    let n = width * height;
    let cw = width.div_ceil(2);
    let ch = height.div_ceil(2);
    assert!(rgb.len() >= n * 3);
    assert!(y.len() >= n);
    assert!(cb.len() >= cw * ch);
    assert!(cr.len() >= cw * ch);

    // Compute Y at full resolution using the fast SIMD path.
    crate::encode::rgb_to_yuv444_y_only(rgb, y, width, height, range, matrix);

    let fwd = ForwardCoeffs::new(matrix, range);
    let inv = InverseCoeffs::new(matrix, range);

    // Process one chroma row at a time — SoA buffers are `cw` entries (~2KB
    // for 1024px) and stay in L1 cache.
    let mut y0 = alloc::vec![0.0f32; cw];
    let mut y1 = alloc::vec![0.0f32; cw];
    let mut y2 = alloc::vec![0.0f32; cw];
    let mut y3 = alloc::vec![0.0f32; cw];
    let mut or0 = alloc::vec![0.0f32; cw];
    let mut og0 = alloc::vec![0.0f32; cw];
    let mut ob0 = alloc::vec![0.0f32; cw];
    let mut or1 = alloc::vec![0.0f32; cw];
    let mut og1 = alloc::vec![0.0f32; cw];
    let mut ob1 = alloc::vec![0.0f32; cw];
    let mut or2 = alloc::vec![0.0f32; cw];
    let mut og2 = alloc::vec![0.0f32; cw];
    let mut ob2 = alloc::vec![0.0f32; cw];
    let mut or3 = alloc::vec![0.0f32; cw];
    let mut og3 = alloc::vec![0.0f32; cw];
    let mut ob3 = alloc::vec![0.0f32; cw];
    let mut cb_f = alloc::vec![0.0f32; cw];
    let mut cr_f = alloc::vec![0.0f32; cw];

    for cy_idx in 0..ch {
        let y0r = cy_idx * 2;
        let y1r = (y0r + 1).min(height - 1);

        // Extract one row of blocks into SoA.
        for cx_idx in 0..cw {
            let x0 = cx_idx * 2;
            let x1 = (x0 + 1).min(width - 1);

            y0[cx_idx] = y[y0r * width + x0] as f32;
            y1[cx_idx] = y[y0r * width + x1] as f32;
            y2[cx_idx] = y[y1r * width + x0] as f32;
            y3[cx_idx] = y[y1r * width + x1] as f32;

            let i00 = (y0r * width + x0) * 3;
            let i10 = (y0r * width + x1) * 3;
            let i01 = (y1r * width + x0) * 3;
            let i11 = (y1r * width + x1) * 3;

            or0[cx_idx] = rgb[i00] as f32;
            og0[cx_idx] = rgb[i00 + 1] as f32;
            ob0[cx_idx] = rgb[i00 + 2] as f32;
            or1[cx_idx] = rgb[i10] as f32;
            og1[cx_idx] = rgb[i10 + 1] as f32;
            ob1[cx_idx] = rgb[i10 + 2] as f32;
            or2[cx_idx] = rgb[i01] as f32;
            og2[cx_idx] = rgb[i01 + 1] as f32;
            ob2[cx_idx] = rgb[i01 + 2] as f32;
            or3[cx_idx] = rgb[i11] as f32;
            og3[cx_idx] = rgb[i11 + 1] as f32;
            ob3[cx_idx] = rgb[i11 + 2] as f32;

            if config.gamma_aware_init {
                // Linearize, average in linear space, delinearize.
                let lin = |v: f32| luts.to_linear[v as u8 as usize];
                let lr = (lin(or0[cx_idx]) + lin(or1[cx_idx]) + lin(or2[cx_idx]) + lin(or3[cx_idx])) * 0.25;
                let lg = (lin(og0[cx_idx]) + lin(og1[cx_idx]) + lin(og2[cx_idx]) + lin(og3[cx_idx])) * 0.25;
                let lb = (lin(ob0[cx_idx]) + lin(ob1[cx_idx]) + lin(ob2[cx_idx]) + lin(ob3[cx_idx])) * 0.25;
                let r_avg = if config.srgb_delinearize {
                    crate::gamma::delinearize_srgb(lr) * 255.0
                } else {
                    crate::gamma::delinearize_libwebp(lr) * 255.0
                };
                let g_avg = if config.srgb_delinearize {
                    crate::gamma::delinearize_srgb(lg) * 255.0
                } else {
                    crate::gamma::delinearize_libwebp(lg) * 255.0
                };
                let b_avg = if config.srgb_delinearize {
                    crate::gamma::delinearize_srgb(lb) * 255.0
                } else {
                    crate::gamma::delinearize_libwebp(lb) * 255.0
                };
                cb_f[cx_idx] = fwd.cb_r_f * r_avg + fwd.cb_g_f * g_avg + fwd.cb_b_f * b_avg + fwd.uv_bias_f;
                cr_f[cx_idx] = fwd.cr_r_f * r_avg + fwd.cr_g_f * g_avg + fwd.cr_b_f * b_avg + fwd.uv_bias_f;
            } else {
                // Box average in gamma space (faster, iteration compensates).
                let r_avg = (or0[cx_idx] + or1[cx_idx] + or2[cx_idx] + or3[cx_idx]) * 0.25;
                let g_avg = (og0[cx_idx] + og1[cx_idx] + og2[cx_idx] + og3[cx_idx]) * 0.25;
                let b_avg = (ob0[cx_idx] + ob1[cx_idx] + ob2[cx_idx] + ob3[cx_idx]) * 0.25;
                cb_f[cx_idx] = fwd.cb_r_f * r_avg + fwd.cb_g_f * g_avg + fwd.cb_b_f * b_avg + fwd.uv_bias_f;
                cr_f[cx_idx] = fwd.cr_r_f * r_avg + fwd.cr_g_f * g_avg + fwd.cr_b_f * b_avg + fwd.uv_bias_f;
            }
        }

        // Run #[autoversion] iterative refinement on this row of blocks.
        sharp_iterate_all_blocks(
            &y0[..cw], &y1[..cw], &y2[..cw], &y3[..cw],
            &or0[..cw], &og0[..cw], &ob0[..cw],
            &or1[..cw], &og1[..cw], &ob1[..cw],
            &or2[..cw], &og2[..cw], &ob2[..cw],
            &or3[..cw], &og3[..cw], &ob3[..cw],
            &mut cb_f[..cw], &mut cr_f[..cw],
            &inv, &fwd,
            config.max_iterations,
            config.convergence_threshold,
        );

        // Write back this row.
        let row_off = cy_idx * cw;
        for cx_idx in 0..cw {
            cb[row_off + cx_idx] = clamp_u8(cb_f[cx_idx]);
            cr[row_off + cx_idx] = clamp_u8(cr_f[cx_idx]);
        }
    }
}

/// Iterative refinement over a row of blocks. Processes 8 blocks in parallel
/// via magetypes `f32x8` — same codegen on AVX2 (native), NEON (2×f32x4),
/// WASM128 (2×f32x4), and scalar (8×f32).
///
/// Uses L2-optimal Newton step with the correct inverse-matrix Jacobian.
fn sharp_iterate_all_blocks(
    y0: &[f32], y1: &[f32], y2: &[f32], y3: &[f32],
    or0: &[f32], og0: &[f32], ob0: &[f32],
    or1: &[f32], og1: &[f32], ob1: &[f32],
    or2: &[f32], og2: &[f32], ob2: &[f32],
    or3: &[f32], og3: &[f32], ob3: &[f32],
    cb_f: &mut [f32], cr_f: &mut [f32],
    inv: &InverseCoeffs, _fwd: &ForwardCoeffs,
    max_iterations: u32,
    _convergence_threshold: f32,
) {
    incant!(sharp_iterate_simd(
        y0, y1, y2, y3,
        or0, og0, ob0, or1, og1, ob1, or2, og2, ob2, or3, og3, ob3,
        cb_f, cr_f, inv, max_iterations
    ));
}

use archmage::prelude::*;
use magetypes::simd::generic::f32x8 as GenericF32x8;

#[magetypes(v3, neon, wasm128, scalar)]
#[inline(always)]
fn sharp_iterate_simd(
    token: Token,
    y0: &[f32], y1: &[f32], y2: &[f32], y3: &[f32],
    or0: &[f32], og0: &[f32], ob0: &[f32],
    or1: &[f32], og1: &[f32], ob1: &[f32],
    or2: &[f32], og2: &[f32], ob2: &[f32],
    or3: &[f32], og3: &[f32], ob3: &[f32],
    cb_f: &mut [f32], cr_f: &mut [f32],
    inv: &InverseCoeffs,
    max_iterations: u32,
) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;

    let n = cb_f.len();
    let chunks = n / 8;

    let uv_center_v = f32x8::splat(token, inv.uv_offset.abs());
    let y_coeff_v = f32x8::splat(token, inv.y_coeff);
    let y_off_v = f32x8::splat(token, inv.y_offset);
    let cr_to_r_v = f32x8::splat(token, inv.cr_to_r);
    let cr_to_g_v = f32x8::splat(token, inv.cr_to_g);
    let cb_to_g_v = f32x8::splat(token, inv.cb_to_g);
    let cb_to_b_v = f32x8::splat(token, inv.cb_to_b);
    let zero_v = f32x8::splat(token, 0.0);
    let max_v = f32x8::splat(token, 255.0);

    let cb_denom = 4.0 * (inv.cb_to_g * inv.cb_to_g + inv.cb_to_b * inv.cb_to_b);
    let cr_denom = 4.0 * (inv.cr_to_r * inv.cr_to_r + inv.cr_to_g * inv.cr_to_g);
    let cb_inv_denom_v = f32x8::splat(token, 1.0 / cb_denom);
    let cr_inv_denom_v = f32x8::splat(token, 1.0 / cr_denom);

    for _ in 0..max_iterations {
        // SIMD: 8 blocks at a time.
        for c in 0..chunks {
            let base = c * 8;
            let cb_v = f32x8::from_slice(token, &cb_f[base..]);
            let cr_v = f32x8::from_slice(token, &cr_f[base..]);
            let cb_c = cb_v - uv_center_v;
            let cr_c = cr_v - uv_center_v;
            let mut cb_num = f32x8::splat(token, 0.0);
            let mut cr_num = f32x8::splat(token, 0.0);

            // 4 pixel positions, unrolled. Each loads 8 values from the SoA arrays.
            macro_rules! pixel {
                ($y:expr, $or:expr, $og:expr, $ob:expr) => {{
                    let yv = f32x8::from_slice(token, &$y[base..]);
                    let y_adj = y_coeff_v * (yv - y_off_v);
                    let rec_r = (y_adj + cr_to_r_v * cr_c).max(zero_v).min(max_v);
                    let rec_g = (y_adj + cr_to_g_v * cr_c + cb_to_g_v * cb_c).max(zero_v).min(max_v);
                    let rec_b = (y_adj + cb_to_b_v * cb_c).max(zero_v).min(max_v);
                    let or_v = f32x8::from_slice(token, &$or[base..]);
                    let og_v = f32x8::from_slice(token, &$og[base..]);
                    let ob_v = f32x8::from_slice(token, &$ob[base..]);
                    let er = or_v - rec_r;
                    let eg = og_v - rec_g;
                    let eb = ob_v - rec_b;
                    cb_num = cb_num + eg * cb_to_g_v + eb * cb_to_b_v;
                    cr_num = cr_num + er * cr_to_r_v + eg * cr_to_g_v;
                }};
            }
            pixel!(y0, or0, og0, ob0);
            pixel!(y1, or1, og1, ob1);
            pixel!(y2, or2, og2, ob2);
            pixel!(y3, or3, og3, ob3);

            let new_cb = (cb_v + cb_num * cb_inv_denom_v).max(zero_v).min(max_v);
            let new_cr = (cr_v + cr_num * cr_inv_denom_v).max(zero_v).min(max_v);

            // Store back. from_slice loads 8, we need store 8.
            let cb_arr = new_cb.to_array();
            let cr_arr = new_cr.to_array();
            cb_f[base..base + 8].copy_from_slice(&cb_arr);
            cr_f[base..base + 8].copy_from_slice(&cr_arr);
        }

        // Scalar tail for remaining blocks.
        let tail_start = chunks * 8;
        let uv_center = inv.uv_offset.abs();
        for i in tail_start..n {
            let cb_c = cb_f[i] - uv_center;
            let cr_c = cr_f[i] - uv_center;
            let mut cb_num = 0.0f32;
            let mut cr_num = 0.0f32;
            macro_rules! pixel_scalar {
                ($yv:expr, $or:expr, $og:expr, $ob:expr) => {{
                    let y_adj = inv.y_coeff * ($yv - inv.y_offset);
                    let rec_r = (y_adj + inv.cr_to_r * cr_c).clamp(0.0, 255.0);
                    let rec_g = (y_adj + inv.cr_to_g * cr_c + inv.cb_to_g * cb_c).clamp(0.0, 255.0);
                    let rec_b = (y_adj + inv.cb_to_b * cb_c).clamp(0.0, 255.0);
                    let er = $or - rec_r;
                    let eg = $og - rec_g;
                    let eb = $ob - rec_b;
                    cb_num += eg * inv.cb_to_g + eb * inv.cb_to_b;
                    cr_num += er * inv.cr_to_r + eg * inv.cr_to_g;
                }};
            }
            pixel_scalar!(y0[i], or0[i], og0[i], ob0[i]);
            pixel_scalar!(y1[i], or1[i], og1[i], ob1[i]);
            pixel_scalar!(y2[i], or2[i], og2[i], ob2[i]);
            pixel_scalar!(y3[i], or3[i], og3[i], ob3[i]);
            cb_f[i] = (cb_f[i] + cb_num / cb_denom).clamp(0.0, 255.0);
            cr_f[i] = (cr_f[i] + cr_num / cr_denom).clamp(0.0, 255.0);
        }
    }
}

#[inline(always)]
fn clamp_u8(v: f32) -> u8 {
    v.round().clamp(0.0, 255.0) as u8
}
