//! Chroma downsampling dispatch for JPEG encoding.
//!
//! Routes to `zenyuv` for 4:2:0 (both GammaAware and GammaAwareIterative).
//! Scalar fallback for 4:2:2 and 4:4:0 gamma-aware paths (zenyuv doesn't
//! support those subsamplings yet).
//!
//! # Methods
//!
//! | DownsamplingMethod    | 4:2:0 path     | 4:2:2 / 4:4:0 path |
//! |----------------------|----------------|---------------------|
//! | Box (default)        | fast_yuv/zenyuv| 4:4:4 then separate downsample |
//! | GammaAware           | zenyuv (iter=0)| scalar (this module) |
//! | GammaAwareIterative  | zenyuv sharp   | scalar (this module) |

// Dead-code analysis note: several items here are reachable only through
// the `__test-utils` pub surface (benches, examples, debugging tools) or
// through target-dependent SIMD dispatch tiers, so the default build
// cannot see their consumers. Suppress dead-code noise for the default
// build; keep the crate warning-clean so REAL warnings stay visible.
#![cfg_attr(not(feature = "__test-utils"), allow(dead_code))]

use crate::color::xyb::{linear_to_srgb_fast, srgb_u8_to_linear};
use crate::foundation::consts::{YCBCR_B_TO_Y, YCBCR_G_TO_Y, YCBCR_R_TO_Y};

// ── 4:2:0 strip dispatch (delegates to zenyuv) ─────────────────────────────

/// Gamma-aware chroma for a strip of image data (4:2:0 mode).
///
/// Both `GammaAware` and `GammaAwareIterative` route through zenyuv:
/// - `use_iterative=true`: Newton-step Sharp YUV (2 iterations, f32x8 SIMD)
/// - `use_iterative=false`: gamma-aware initial estimate only (iterations=0)
pub fn gamma_aware_strip_420(
    rgb_strip: &[u8],
    y_strip: &mut [f32],
    cb_down: &mut [f32],
    cr_down: &mut [f32],
    width: usize,
    strip_height: usize,
    _strip_y: usize,
    _image_height: usize,
    bpp: usize,
    use_iterative: bool,
) {
    zenyuv_strip_420(
        rgb_strip,
        y_strip,
        cb_down,
        cr_down,
        width,
        strip_height,
        bpp,
        use_iterative,
    );
}

/// Delegate to zenyuv for 4:2:0 chroma. Uses the native f32 output path —
/// no u8 intermediate, no u8→f32 conversion pass.
fn zenyuv_strip_420(
    rgb_strip: &[u8],
    y_strip: &mut [f32],
    cb_down: &mut [f32],
    cr_down: &mut [f32],
    width: usize,
    strip_height: usize,
    bpp: usize,
    use_iterative: bool,
) {
    let num_pixels = width * strip_height;
    let cw = (width + 1) / 2;

    // Strip alpha if RGBA.
    let rgb_only: alloc::vec::Vec<u8>;
    let rgb_input = if bpp == 4 {
        rgb_only = rgb_strip
            .chunks_exact(4)
            .take(num_pixels)
            .flat_map(|chunk| [chunk[0], chunk[1], chunk[2]])
            .collect();
        &rgb_only
    } else {
        rgb_strip
    };

    let config = if use_iterative {
        zenyuv::SharpYuvConfig::default() // Newton iter=2
    } else {
        zenyuv::SharpYuvConfig {
            max_iterations: 0,
            ..Default::default()
        }
    };

    let mut ws = zenyuv::sharp::SharpYuvWorkspace::new(cw);
    zenyuv::sharp::rgb_to_yuv420_sharp_f32(
        rgb_input,
        y_strip,
        cb_down,
        cr_down,
        width,
        strip_height,
        zenyuv::Range::Full,
        zenyuv::Matrix::Bt601,
        &config,
        &mut ws,
    );
}

// ── 4:2:2 / 4:4:0 strip paths (scalar, not yet in zenyuv) ──────────────────

/// Gamma-aware chroma for a strip (4:2:2 horizontal-only downsampling).
pub fn gamma_aware_strip_422(
    rgb_strip: &[u8],
    y_strip: &mut [f32],
    cb_down: &mut [f32],
    cr_down: &mut [f32],
    width: usize,
    strip_height: usize,
    bpp: usize,
    use_iterative: bool,
) {
    compute_y_plane_from_rgb(rgb_strip, width, strip_height, bpp, y_strip);
    let c_width = (width + 1) / 2;
    for y in 0..strip_height {
        for cx in 0..c_width {
            let (cb, cr) = if use_iterative {
                iterative_chroma_2x1_strip(rgb_strip, y_strip, width, strip_height, bpp, cx, y)
            } else {
                gamma_aware_chroma_2x1_strip(rgb_strip, width, strip_height, bpp, cx, y)
            };
            cb_down[y * c_width + cx] = cb;
            cr_down[y * c_width + cx] = cr;
        }
    }
}

/// Gamma-aware chroma for a strip (4:4:0 vertical-only downsampling).
pub fn gamma_aware_strip_440(
    rgb_strip: &[u8],
    y_strip: &mut [f32],
    cb_down: &mut [f32],
    cr_down: &mut [f32],
    width: usize,
    strip_height: usize,
    bpp: usize,
    use_iterative: bool,
) {
    compute_y_plane_from_rgb(rgb_strip, width, strip_height, bpp, y_strip);
    let c_height = (strip_height + 1) / 2;
    for cy in 0..c_height {
        for x in 0..width {
            let (cb, cr) = if use_iterative {
                iterative_chroma_1x2_strip(rgb_strip, y_strip, width, strip_height, bpp, x, cy)
            } else {
                gamma_aware_chroma_1x2_strip(rgb_strip, width, strip_height, bpp, x, cy)
            };
            cb_down[cy * width + x] = cb;
            cr_down[cy * width + x] = cr;
        }
    }
}

// ── Helpers (scalar, for 4:2:2 / 4:4:0 only) ───────────────────────────────

/// Compute Y plane from interleaved RGB.
fn compute_y_plane_from_rgb(
    data: &[u8],
    width: usize,
    height: usize,
    bpp: usize,
    y_plane: &mut [f32],
) {
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * bpp;
            let r = data[idx] as f32;
            let g = data[idx + 1] as f32;
            let b = data[idx + 2] as f32;
            y_plane[y * width + x] = YCBCR_R_TO_Y * r + YCBCR_G_TO_Y * g + YCBCR_B_TO_Y * b;
        }
    }
}

/// Gamma-aware chroma for a 2x1 block (4:2:2).
fn gamma_aware_chroma_2x1_strip(
    data: &[u8],
    width: usize,
    _height: usize,
    bpp: usize,
    cx: usize,
    y: usize,
) -> (f32, f32) {
    let x0 = cx * 2;
    let x1 = (x0 + 1).min(width - 1);
    let get = |x: usize| -> (f32, f32, f32) {
        let i = (y * width + x) * bpp;
        (
            srgb_u8_to_linear(data[i]),
            srgb_u8_to_linear(data[i + 1]),
            srgb_u8_to_linear(data[i + 2]),
        )
    };
    let (lr0, lg0, lb0) = get(x0);
    let (lr1, lg1, lb1) = get(x1);
    let r = linear_to_srgb_fast((lr0 + lr1) * 0.5) * 255.0;
    let g = linear_to_srgb_fast((lg0 + lg1) * 0.5) * 255.0;
    let b = linear_to_srgb_fast((lb0 + lb1) * 0.5) * 255.0;
    {
        let (_, cb, cr) = crate::color::rgb_to_ycbcr_f32(r, g, b);
        (cb, cr)
    }
}

/// Gamma-aware chroma for a 1x2 block (4:4:0).
fn gamma_aware_chroma_1x2_strip(
    data: &[u8],
    width: usize,
    height: usize,
    bpp: usize,
    x: usize,
    cy: usize,
) -> (f32, f32) {
    let y0 = cy * 2;
    let y1 = (y0 + 1).min(height - 1);
    let get = |y: usize| -> (f32, f32, f32) {
        let i = (y * width + x) * bpp;
        (
            srgb_u8_to_linear(data[i]),
            srgb_u8_to_linear(data[i + 1]),
            srgb_u8_to_linear(data[i + 2]),
        )
    };
    let (lr0, lg0, lb0) = get(y0);
    let (lr1, lg1, lb1) = get(y1);
    let r = linear_to_srgb_fast((lr0 + lr1) * 0.5) * 255.0;
    let g = linear_to_srgb_fast((lg0 + lg1) * 0.5) * 255.0;
    let b = linear_to_srgb_fast((lb0 + lb1) * 0.5) * 255.0;
    {
        let (_, cb, cr) = crate::color::rgb_to_ycbcr_f32(r, g, b);
        (cb, cr)
    }
}

/// Iterative chroma for a 2x1 block (4:2:2). Scalar Newton step.
fn iterative_chroma_2x1_strip(
    data: &[u8],
    y_plane: &[f32],
    width: usize,
    _height: usize,
    bpp: usize,
    cx: usize,
    y: usize,
) -> (f32, f32) {
    let x0 = cx * 2;
    let x1 = (x0 + 1).min(width - 1);
    let y_vals = [y_plane[y * width + x0], y_plane[y * width + x1]];
    let get_rgb = |x: usize| -> (f32, f32, f32) {
        let i = (y * width + x) * bpp;
        (data[i] as f32, data[i + 1] as f32, data[i + 2] as f32)
    };
    let orig = [get_rgb(x0), get_rgb(x1)];
    let (mut cb, mut cr) = gamma_aware_chroma_2x1_strip(data, width, _height, bpp, cx, y);
    iterative_refine_n(&y_vals, &orig, &mut cb, &mut cr, 2);
    (cb, cr)
}

/// Iterative chroma for a 1x2 block (4:4:0). Scalar Newton step.
fn iterative_chroma_1x2_strip(
    data: &[u8],
    y_plane: &[f32],
    width: usize,
    height: usize,
    bpp: usize,
    x: usize,
    cy: usize,
) -> (f32, f32) {
    let y0 = cy * 2;
    let y1 = (y0 + 1).min(height - 1);
    let y_vals = [y_plane[y0 * width + x], y_plane[y1 * width + x]];
    let get_rgb = |y: usize| -> (f32, f32, f32) {
        let i = (y * width + x) * bpp;
        (data[i] as f32, data[i + 1] as f32, data[i + 2] as f32)
    };
    let orig = [get_rgb(y0), get_rgb(y1)];
    let (mut cb, mut cr) = gamma_aware_chroma_1x2_strip(data, width, height, bpp, x, cy);
    iterative_refine_n(&y_vals, &orig, &mut cb, &mut cr, 2);
    (cb, cr)
}

/// Newton-step iterative refinement for N pixels sharing one Cb/Cr.
/// Uses the correct inverse-matrix Jacobian (same math as zenyuv's sharp kernel).
fn iterative_refine_n(
    y_vals: &[f32],
    orig: &[(f32, f32, f32)],
    cb: &mut f32,
    cr: &mut f32,
    n: usize,
) {
    use crate::foundation::consts::*;
    // Inverse matrix coefficients for BT.601 full range.
    let cb_to_g: f32 = YCBCR_CB_TO_G;
    let cb_to_b: f32 = YCBCR_CB_TO_B;
    let cr_to_r: f32 = YCBCR_CR_TO_R;
    let cr_to_g: f32 = YCBCR_CR_TO_G;
    let cb_denom = n as f32 * (cb_to_g * cb_to_g + cb_to_b * cb_to_b);
    let cr_denom = n as f32 * (cr_to_r * cr_to_r + cr_to_g * cr_to_g);

    for _ in 0..2 {
        let cb_c = *cb - 128.0;
        let cr_c = *cr - 128.0;
        let mut cb_num = 0.0f32;
        let mut cr_num = 0.0f32;
        for i in 0..n {
            let yv = y_vals[i];
            let (or, og, ob) = orig[i];
            let rec_r = (yv + cr_to_r * cr_c).clamp(0.0, 255.0);
            let rec_g = (yv + cr_to_g * cr_c + cb_to_g * cb_c).clamp(0.0, 255.0);
            let rec_b = (yv + cb_to_b * cb_c).clamp(0.0, 255.0);
            cb_num += (og - rec_g) * cb_to_g + (ob - rec_b) * cb_to_b;
            cr_num += (or - rec_r) * cr_to_r + (og - rec_g) * cr_to_g;
        }
        *cb = (*cb + cb_num / cb_denom).clamp(0.0, 255.0);
        *cr = (*cr + cr_num / cr_denom).clamp(0.0, 255.0);
    }
}

// ── Whole-image entry points (used by tests/examples, not strip encoder) ────

use crate::error::{Error, Result};
use crate::foundation::alloc::{checked_size_2d, try_alloc_zeroed_f32};
use crate::types::PixelFormat;

fn get_bpp(pixel_format: PixelFormat) -> Result<usize> {
    match pixel_format {
        PixelFormat::Rgb => Ok(3),
        PixelFormat::Rgba => Ok(4),
        _ => Err(Error::unsupported_feature(
            "only RGB/RGBA supported for gamma-aware chroma",
        )),
    }
}

/// Convert a full image with gamma-aware 4:2:0 downsampling.
pub fn convert_gamma_aware_420(
    data: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, usize, usize)> {
    let bpp = get_bpp(pixel_format)?;
    let num_pixels = checked_size_2d(width, height)?;
    let c_width = (width + 1) / 2;
    let c_height = (height + 1) / 2;
    let c_size = checked_size_2d(c_width, c_height)?;
    let mut y_plane = try_alloc_zeroed_f32(num_pixels, "chroma")?;
    let mut cb_plane = try_alloc_zeroed_f32(c_size, "chroma")?;
    let mut cr_plane = try_alloc_zeroed_f32(c_size, "chroma")?;
    gamma_aware_strip_420(
        data,
        &mut y_plane,
        &mut cb_plane,
        &mut cr_plane,
        width,
        height,
        0,
        height,
        bpp,
        false,
    );
    Ok((y_plane, cb_plane, cr_plane, c_width, c_height))
}

/// Convert a full image with iterative 4:2:0 downsampling (Sharp YUV).
pub fn convert_gamma_aware_iterative_420(
    data: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, usize, usize)> {
    let bpp = get_bpp(pixel_format)?;
    let num_pixels = checked_size_2d(width, height)?;
    let c_width = (width + 1) / 2;
    let c_height = (height + 1) / 2;
    let c_size = checked_size_2d(c_width, c_height)?;
    let mut y_plane = try_alloc_zeroed_f32(num_pixels, "chroma")?;
    let mut cb_plane = try_alloc_zeroed_f32(c_size, "chroma")?;
    let mut cr_plane = try_alloc_zeroed_f32(c_size, "chroma")?;
    gamma_aware_strip_420(
        data,
        &mut y_plane,
        &mut cb_plane,
        &mut cr_plane,
        width,
        height,
        0,
        height,
        bpp,
        true,
    );
    Ok((y_plane, cb_plane, cr_plane, c_width, c_height))
}

/// Convert a full image with gamma-aware 4:2:2 downsampling.
pub fn convert_gamma_aware_422(
    data: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, usize, usize)> {
    let bpp = get_bpp(pixel_format)?;
    let num_pixels = checked_size_2d(width, height)?;
    let c_width = (width + 1) / 2;
    let mut y_plane = try_alloc_zeroed_f32(num_pixels, "chroma")?;
    let mut cb_plane = try_alloc_zeroed_f32(checked_size_2d(c_width, height)?, "chroma")?;
    let mut cr_plane = try_alloc_zeroed_f32(checked_size_2d(c_width, height)?, "chroma")?;
    gamma_aware_strip_422(
        data,
        &mut y_plane,
        &mut cb_plane,
        &mut cr_plane,
        width,
        height,
        bpp,
        false,
    );
    Ok((y_plane, cb_plane, cr_plane, c_width, height))
}

/// Convert a full image with iterative 4:2:2 downsampling.
pub fn convert_gamma_aware_iterative_422(
    data: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, usize, usize)> {
    let bpp = get_bpp(pixel_format)?;
    let num_pixels = checked_size_2d(width, height)?;
    let c_width = (width + 1) / 2;
    let mut y_plane = try_alloc_zeroed_f32(num_pixels, "chroma")?;
    let mut cb_plane = try_alloc_zeroed_f32(checked_size_2d(c_width, height)?, "chroma")?;
    let mut cr_plane = try_alloc_zeroed_f32(checked_size_2d(c_width, height)?, "chroma")?;
    gamma_aware_strip_422(
        data,
        &mut y_plane,
        &mut cb_plane,
        &mut cr_plane,
        width,
        height,
        bpp,
        true,
    );
    Ok((y_plane, cb_plane, cr_plane, c_width, height))
}

/// Convert a full image with gamma-aware 4:4:0 downsampling.
pub fn convert_gamma_aware_440(
    data: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, usize, usize)> {
    let bpp = get_bpp(pixel_format)?;
    let num_pixels = checked_size_2d(width, height)?;
    let c_height = (height + 1) / 2;
    let mut y_plane = try_alloc_zeroed_f32(num_pixels, "chroma")?;
    let mut cb_plane = try_alloc_zeroed_f32(checked_size_2d(width, c_height)?, "chroma")?;
    let mut cr_plane = try_alloc_zeroed_f32(checked_size_2d(width, c_height)?, "chroma")?;
    gamma_aware_strip_440(
        data,
        &mut y_plane,
        &mut cb_plane,
        &mut cr_plane,
        width,
        height,
        bpp,
        false,
    );
    Ok((y_plane, cb_plane, cr_plane, width, c_height))
}

/// Convert a full image with iterative 4:4:0 downsampling.
pub fn convert_gamma_aware_iterative_440(
    data: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, usize, usize)> {
    let bpp = get_bpp(pixel_format)?;
    let num_pixels = checked_size_2d(width, height)?;
    let c_height = (height + 1) / 2;
    let mut y_plane = try_alloc_zeroed_f32(num_pixels, "chroma")?;
    let mut cb_plane = try_alloc_zeroed_f32(checked_size_2d(width, c_height)?, "chroma")?;
    let mut cr_plane = try_alloc_zeroed_f32(checked_size_2d(width, c_height)?, "chroma")?;
    gamma_aware_strip_440(
        data,
        &mut y_plane,
        &mut cb_plane,
        &mut cr_plane,
        width,
        height,
        bpp,
        true,
    );
    Ok((y_plane, cb_plane, cr_plane, width, c_height))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gamma_aware_420_basic() {
        let w = 16;
        let h = 8;
        let rgb = alloc::vec![128u8; w * h * 3];
        let (y, cb, cr, cw, ch) = convert_gamma_aware_420(&rgb, w, h, PixelFormat::Rgb).unwrap();
        assert_eq!(cw, 8);
        assert_eq!(ch, 4);
        // Gray input → Y≈128, Cb≈128, Cr≈128
        assert!((y[0] - 128.0).abs() < 2.0, "Y={}", y[0]);
        assert!((cb[0] - 128.0).abs() < 2.0, "Cb={}", cb[0]);
        assert!((cr[0] - 128.0).abs() < 2.0, "Cr={}", cr[0]);
    }

    #[test]
    fn test_iterative_420_basic() {
        let w = 16;
        let h = 8;
        let rgb = alloc::vec![128u8; w * h * 3];
        let (y, cb, cr, cw, ch) =
            convert_gamma_aware_iterative_420(&rgb, w, h, PixelFormat::Rgb).unwrap();
        assert_eq!(cw, 8);
        assert_eq!(ch, 4);
        assert!((y[0] - 128.0).abs() < 2.0);
        assert!((cb[0] - 128.0).abs() < 2.0);
        assert!((cr[0] - 128.0).abs() < 2.0);
    }
}
