//! Sharp YUV: iterative perceptual chroma optimization.
//!
//! Minimizes reconstruction error in gamma-encoded RGB space by iteratively
//! adjusting Cb/Cr values. Same algorithm as libwebp's SharpYUV, parameterized
//! over gamma transfer function, color matrix, and range.
//!
//! No traits — gamma LUTs are passed as `&GammaLuts`, matrix coefficients as
//! `&ForwardCoeffs` / `&InverseCoeffs`. All functions are `#[inline(always)]`
//! so they can be called from `#[arcane]` SIMD regions in the future.

use crate::gamma::{self, GammaLuts};
use crate::types::{ForwardCoeffs, InverseCoeffs, Matrix, Range};

/// Configuration for Sharp YUV chroma optimization.
pub struct SharpYuvConfig {
    /// Maximum refinement iterations per 2×2 block (default: 4).
    pub max_iterations: u32,
    /// Stop early if total reconstruction error drops below this (default: 0.1).
    pub convergence_threshold: f32,
    /// Which delinearization to use for gamma-aware initial estimate.
    /// `true` = sRGB (zenjpeg), `false` = libwebp gamma^0.45 (zenwebp).
    pub srgb_delinearize: bool,
}

impl Default for SharpYuvConfig {
    fn default() -> Self {
        Self {
            max_iterations: 4,
            convergence_threshold: 0.1,
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

    // Compute Cb/Cr via iterative optimization per 2×2 block.
    let fwd = ForwardCoeffs::new(matrix, range);
    let inv = InverseCoeffs::new(matrix, range);

    for cy_idx in 0..ch {
        for cx_idx in 0..cw {
            let (cb_val, cr_val) = iterative_chroma_2x2(
                rgb, y, width, height, cx_idx, cy_idx, &fwd, &inv, luts, config,
            );
            cb[cy_idx * cw + cx_idx] = clamp_u8(cb_val);
            cr[cy_idx * cw + cx_idx] = clamp_u8(cr_val);
        }
    }
}

/// Gamma-aware (non-iterative) chroma for a single 2×2 block.
///
/// Linearize 4 RGB pixels via LUT, average in linear space, delinearize,
/// then apply forward matrix. This is the initial estimate that the iterative
/// kernel refines.
#[inline(always)]
fn gamma_aware_chroma_2x2(
    rgb: &[u8],
    width: usize,
    height: usize,
    cx: usize,
    cy: usize,
    fwd: &ForwardCoeffs,
    luts: &GammaLuts,
    srgb_delin: bool,
) -> (f32, f32) {
    let x0 = cx * 2;
    let y0 = cy * 2;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);

    let lin = |x: usize, y: usize| -> (f32, f32, f32) {
        let i = (y * width + x) * 3;
        (
            gamma::linearize(luts, rgb[i]),
            gamma::linearize(luts, rgb[i + 1]),
            gamma::linearize(luts, rgb[i + 2]),
        )
    };

    let (lr00, lg00, lb00) = lin(x0, y0);
    let (lr10, lg10, lb10) = lin(x1, y0);
    let (lr01, lg01, lb01) = lin(x0, y1);
    let (lr11, lg11, lb11) = lin(x1, y1);

    let lr = (lr00 + lr10 + lr01 + lr11) * 0.25;
    let lg = (lg00 + lg10 + lg01 + lg11) * 0.25;
    let lb = (lb00 + lb10 + lb01 + lb11) * 0.25;

    // Delinearize back to gamma space.
    let (r, g, b) = if srgb_delin {
        (
            gamma::delinearize_srgb(lr) * 255.0,
            gamma::delinearize_srgb(lg) * 255.0,
            gamma::delinearize_srgb(lb) * 255.0,
        )
    } else {
        (
            gamma::delinearize_libwebp(lr) * 255.0,
            gamma::delinearize_libwebp(lg) * 255.0,
            gamma::delinearize_libwebp(lb) * 255.0,
        )
    };

    // Forward matrix: RGB → (Y, Cb, Cr). We only need Cb/Cr.
    let cb = fwd.cb_r_f * r + fwd.cb_g_f * g + fwd.cb_b_f * b + fwd.uv_bias_f;
    let cr = fwd.cr_r_f * r + fwd.cr_g_f * g + fwd.cr_b_f * b + fwd.uv_bias_f;
    (cb, cr)
}

/// Iteratively optimize chroma for a 2×2 block to minimize reconstruction error.
///
/// 1. Start with gamma-aware averaged chroma (single-pass estimate)
/// 2. For each iteration: reconstruct RGB from Y + Cb/Cr, compute error,
///    adjust Cb/Cr to reduce error
/// 3. Stop on convergence or max iterations
#[inline(always)]
fn iterative_chroma_2x2(
    rgb: &[u8],
    y_plane: &[u8],
    width: usize,
    height: usize,
    cx: usize,
    cy: usize,
    fwd: &ForwardCoeffs,
    inv: &InverseCoeffs,
    luts: &GammaLuts,
    config: &SharpYuvConfig,
) -> (f32, f32) {
    let x0 = cx * 2;
    let y0 = cy * 2;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);

    // Read 4 Y values as f32.
    let y_vals = [
        y_plane[y0 * width + x0] as f32,
        y_plane[y0 * width + x1] as f32,
        y_plane[y1 * width + x0] as f32,
        y_plane[y1 * width + x1] as f32,
    ];

    // Read 4 original RGB as f32.
    let get_rgb = |x: usize, y: usize| -> (f32, f32, f32) {
        let i = (y * width + x) * 3;
        (rgb[i] as f32, rgb[i + 1] as f32, rgb[i + 2] as f32)
    };
    let orig = [
        get_rgb(x0, y0),
        get_rgb(x1, y0),
        get_rgb(x0, y1),
        get_rgb(x1, y1),
    ];

    // Initial estimate via gamma-aware averaging.
    let (mut cb, mut cr) =
        gamma_aware_chroma_2x2(rgb, width, height, cx, cy, fwd, luts, config.srgb_delinearize);

    // Adjustment weights derived from the forward Cb/Cr matrix rows.
    let adj_cb_r = fwd.cb_r_f;
    let adj_cb_g = fwd.cb_g_f;
    let adj_cb_b = fwd.cb_b_f;
    let adj_cr_r = fwd.cr_r_f;
    let adj_cr_g = fwd.cr_g_f;
    let adj_cr_b = fwd.cr_b_f;

    for _ in 0..config.max_iterations {
        let mut total_error = 0.0f32;
        let mut cb_adj = 0.0f32;
        let mut cr_adj = 0.0f32;

        for i in 0..4 {
            let (orig_r, orig_g, orig_b) = orig[i];
            let yv = y_vals[i];

            // Reconstruct RGB from Y + current Cb/Cr via inverse matrix.
            let cb_centered = cb - inv.uv_offset.abs(); // uv_offset is -128
            let cr_centered = cr - inv.uv_offset.abs();
            let y_adj = inv.y_coeff * (yv - inv.y_offset);
            let rec_r = (y_adj + inv.cr_to_r * cr_centered).clamp(0.0, 255.0);
            let rec_g =
                (y_adj + inv.cr_to_g * cr_centered + inv.cb_to_g * cb_centered).clamp(0.0, 255.0);
            let rec_b = (y_adj + inv.cb_to_b * cb_centered).clamp(0.0, 255.0);

            let err_r = orig_r - rec_r;
            let err_g = orig_g - rec_g;
            let err_b = orig_b - rec_b;

            total_error += err_r.abs() + err_g.abs() + err_b.abs();

            cb_adj += adj_cb_b * err_b + adj_cb_r * err_r + adj_cb_g * err_g;
            cr_adj += adj_cr_r * err_r + adj_cr_g * err_g + adj_cr_b * err_b;
        }

        if total_error < config.convergence_threshold {
            break;
        }

        // Damped update: average over 4 pixels, scale by 0.5 to prevent oscillation.
        let scale = 0.25 * 0.5;
        cb = (cb + cb_adj * scale).clamp(0.0, 255.0);
        cr = (cr + cr_adj * scale).clamp(0.0, 255.0);
    }

    (cb, cr)
}

#[inline(always)]
fn clamp_u8(v: f32) -> u8 {
    v.round().clamp(0.0, 255.0) as u8
}
