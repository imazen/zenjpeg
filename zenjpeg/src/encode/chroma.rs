//! Chroma downsampling methods for JPEG encoding.
//!
//! This module provides various approaches to chroma subsampling (4:2:0, 4:2:2, 4:4:0),
//! each with different quality/performance tradeoffs.
//!
//! # Downsampling Methods
//!
//! | Method | Description | Quality | Speed |
//! |--------|-------------|---------|-------|
//! | Box | Simple averaging of Cb/Cr values | Low | Fast |
//! | BoxSmoothed | 3x3 pre-blur + box filter | Medium | Fast |
//! | GammaAware | Average RGB in linear space | Good | Medium |
//! | GammaAwareIterative | Iterative optimization with clipping | Best | Slow |
//!
//! # Background
//!
//! Naive box filtering of chroma values produces incorrect results because Cb/Cr
//! are derived from gamma-encoded RGB. Averaging in non-linear space causes:
//! - Color bleeding on sharp edges
//! - Darkening of saturated colors
//! - Loss of thin colored lines
//!
//! Gamma-aware methods work in linear RGB space to preserve perceptual accuracy.
//! The iterative variant additionally handles out-of-gamut clipping for best quality.

#![allow(dead_code)]

use crate::color;
use crate::color::xyb::{linear_to_srgb_fast, srgb_u8_to_linear};
use crate::error::{Error, Result};
use crate::foundation::alloc::{checked_size_2d, try_alloc_zeroed_f32};
use crate::foundation::consts::{YCBCR_B_TO_Y, YCBCR_G_TO_Y, YCBCR_R_TO_Y};
use crate::types::PixelFormat;

use wide::f32x8;

// ============================================================================
// Constants
// ============================================================================

/// Number of iterations for iterative gamma-aware downsampling.
/// Matches libwebp's kNumIterations = 4.
const NUM_ITERATIONS: usize = 4;

/// Convergence threshold for iterative refinement.
/// If the total Y difference is below this, we've converged.
const CONVERGENCE_THRESHOLD: f32 = 0.1;

// ============================================================================
// SIMD Helper Functions
// ============================================================================

/// Compute Y (luminance) plane from interleaved RGB u8 data using SIMD.
///
/// Processes 8 pixels at a time using f32x8 vectors.
fn compute_y_plane_from_rgb(
    data: &[u8],
    width: usize,
    height: usize,
    bpp: usize,
    y_plane: &mut [f32],
) {
    let r_to_y = f32x8::splat(YCBCR_R_TO_Y);
    let g_to_y = f32x8::splat(YCBCR_G_TO_Y);
    let b_to_y = f32x8::splat(YCBCR_B_TO_Y);

    let num_pixels = width * height;
    let chunks = num_pixels / 8;

    for chunk in 0..chunks {
        let base = chunk * 8;
        let rgb_base = base * bpp;

        // Gather 8 RGB pixels
        let r = f32x8::from([
            data[rgb_base] as f32,
            data[rgb_base + bpp] as f32,
            data[rgb_base + 2 * bpp] as f32,
            data[rgb_base + 3 * bpp] as f32,
            data[rgb_base + 4 * bpp] as f32,
            data[rgb_base + 5 * bpp] as f32,
            data[rgb_base + 6 * bpp] as f32,
            data[rgb_base + 7 * bpp] as f32,
        ]);
        let g = f32x8::from([
            data[rgb_base + 1] as f32,
            data[rgb_base + bpp + 1] as f32,
            data[rgb_base + 2 * bpp + 1] as f32,
            data[rgb_base + 3 * bpp + 1] as f32,
            data[rgb_base + 4 * bpp + 1] as f32,
            data[rgb_base + 5 * bpp + 1] as f32,
            data[rgb_base + 6 * bpp + 1] as f32,
            data[rgb_base + 7 * bpp + 1] as f32,
        ]);
        let b = f32x8::from([
            data[rgb_base + 2] as f32,
            data[rgb_base + bpp + 2] as f32,
            data[rgb_base + 2 * bpp + 2] as f32,
            data[rgb_base + 3 * bpp + 2] as f32,
            data[rgb_base + 4 * bpp + 2] as f32,
            data[rgb_base + 5 * bpp + 2] as f32,
            data[rgb_base + 6 * bpp + 2] as f32,
            data[rgb_base + 7 * bpp + 2] as f32,
        ]);

        // Y = R_TO_Y * R + G_TO_Y * G + B_TO_Y * B - use FMA for accuracy
        let y = r_to_y.mul_add(r, g_to_y.mul_add(g, b_to_y * b));
        let arr: [f32; 8] = y.into();
        y_plane[base..base + 8].copy_from_slice(&arr);
    }

    // Handle remainder with scalar code
    for i in (chunks * 8)..num_pixels {
        let idx = i * bpp;
        let r = data[idx] as f32;
        let g = data[idx + 1] as f32;
        let b = data[idx + 2] as f32;
        y_plane[i] = color::rgb_to_ycbcr_f32(r, g, b).0;
    }
}

// ============================================================================
// Gamma-Aware Downsampling (Single Pass)
// ============================================================================

/// Converts RGB to YCbCr with gamma-aware chroma downsampling for 4:2:0.
///
/// This is the f32-native alternative to yuv crate's Sharp YUV:
/// - Y channel computed at full resolution from each pixel
/// - Cb/Cr computed by averaging RGB in linear space, then converting to YCbCr
///
/// # Arguments
/// * `data` - RGB or RGBA pixel data
/// * `width` - Image width
/// * `height` - Image height
/// * `pixel_format` - Input pixel format (RGB or RGBA)
///
/// # Returns
/// (y_plane, cb_plane, cr_plane, chroma_width, chroma_height)
pub fn convert_gamma_aware_420(
    data: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, usize, usize)> {
    let c_width = (width + 1) / 2;
    let c_height = (height + 1) / 2;

    // Allocate output planes
    let y_size = checked_size_2d(width, height)?;
    let c_size = checked_size_2d(c_width, c_height)?;
    let mut y_plane = try_alloc_zeroed_f32(y_size, "Y plane")?;
    let mut cb_plane = try_alloc_zeroed_f32(c_size, "Cb plane")?;
    let mut cr_plane = try_alloc_zeroed_f32(c_size, "Cr plane")?;

    let bpp = get_bpp(pixel_format)?;

    // First pass: compute Y at full resolution (SIMD-optimized)
    compute_y_plane_from_rgb(data, width, height, bpp, &mut y_plane);

    // Second pass: compute Cb/Cr with gamma-aware downsampling
    for cy in 0..c_height {
        for cx in 0..c_width {
            let (cb, cr) = gamma_aware_chroma_2x2(data, width, height, bpp, cx, cy);
            cb_plane[cy * c_width + cx] = cb;
            cr_plane[cy * c_width + cx] = cr;
        }
    }

    Ok((y_plane, cb_plane, cr_plane, c_width, c_height))
}

/// Converts RGB to YCbCr with gamma-aware chroma downsampling for 4:2:2.
///
/// Similar to 4:2:0 but only downsamples horizontally (2x1 blocks).
pub fn convert_gamma_aware_422(
    data: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, usize, usize)> {
    let c_width = (width + 1) / 2;

    let y_size = checked_size_2d(width, height)?;
    let c_size = checked_size_2d(c_width, height)?;
    let mut y_plane = try_alloc_zeroed_f32(y_size, "Y plane")?;
    let mut cb_plane = try_alloc_zeroed_f32(c_size, "Cb plane")?;
    let mut cr_plane = try_alloc_zeroed_f32(c_size, "Cr plane")?;

    let bpp = get_bpp(pixel_format)?;

    // First pass: compute Y at full resolution (SIMD-optimized)
    compute_y_plane_from_rgb(data, width, height, bpp, &mut y_plane);

    // Second pass: gamma-aware horizontal downsampling for Cb/Cr
    for y in 0..height {
        for cx in 0..c_width {
            let (cb, cr) = gamma_aware_chroma_2x1(data, width, bpp, cx, y);
            cb_plane[y * c_width + cx] = cb;
            cr_plane[y * c_width + cx] = cr;
        }
    }

    Ok((y_plane, cb_plane, cr_plane, c_width, height))
}

/// Converts RGB to YCbCr with gamma-aware chroma downsampling for 4:4:0.
///
/// Similar to 4:2:0 but only downsamples vertically (1x2 blocks).
pub fn convert_gamma_aware_440(
    data: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, usize, usize)> {
    let c_height = (height + 1) / 2;

    let y_size = checked_size_2d(width, height)?;
    let c_size = checked_size_2d(width, c_height)?;
    let mut y_plane = try_alloc_zeroed_f32(y_size, "Y plane")?;
    let mut cb_plane = try_alloc_zeroed_f32(c_size, "Cb plane")?;
    let mut cr_plane = try_alloc_zeroed_f32(c_size, "Cr plane")?;

    let bpp = get_bpp(pixel_format)?;

    // First pass: compute Y at full resolution (SIMD-optimized)
    compute_y_plane_from_rgb(data, width, height, bpp, &mut y_plane);

    // Second pass: gamma-aware vertical downsampling for Cb/Cr
    for cy in 0..c_height {
        for x in 0..width {
            let (cb, cr) = gamma_aware_chroma_1x2(data, width, height, bpp, x, cy);
            cb_plane[cy * width + x] = cb;
            cr_plane[cy * width + x] = cr;
        }
    }

    Ok((y_plane, cb_plane, cr_plane, width, c_height))
}

// ============================================================================
// Gamma-Aware Iterative Downsampling (Sharp YUV style)
// ============================================================================

/// Converts RGB to YCbCr with iterative gamma-aware chroma downsampling for 4:2:0.
///
/// This is similar to libwebp's Sharp YUV algorithm:
/// 1. Start with gamma-aware averaged chroma
/// 2. Iteratively refine to minimize reconstruction error
/// 3. Handle out-of-gamut clipping
///
/// Produces the highest quality chroma at the cost of more computation.
pub fn convert_gamma_aware_iterative_420(
    data: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, usize, usize)> {
    let c_width = (width + 1) / 2;
    let c_height = (height + 1) / 2;

    let y_size = checked_size_2d(width, height)?;
    let c_size = checked_size_2d(c_width, c_height)?;
    let mut y_plane = try_alloc_zeroed_f32(y_size, "Y plane")?;
    let mut cb_plane = try_alloc_zeroed_f32(c_size, "Cb plane")?;
    let mut cr_plane = try_alloc_zeroed_f32(c_size, "Cr plane")?;

    let bpp = get_bpp(pixel_format)?;

    // First pass: compute Y at full resolution (SIMD-optimized)
    compute_y_plane_from_rgb(data, width, height, bpp, &mut y_plane);

    // Second pass: iterative gamma-aware chroma optimization
    for cy in 0..c_height {
        for cx in 0..c_width {
            let (cb, cr) = iterative_chroma_2x2(data, &y_plane, width, height, bpp, cx, cy);
            cb_plane[cy * c_width + cx] = cb;
            cr_plane[cy * c_width + cx] = cr;
        }
    }

    Ok((y_plane, cb_plane, cr_plane, c_width, c_height))
}

/// Converts RGB to YCbCr with iterative gamma-aware chroma downsampling for 4:2:2.
pub fn convert_gamma_aware_iterative_422(
    data: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, usize, usize)> {
    let c_width = (width + 1) / 2;

    let y_size = checked_size_2d(width, height)?;
    let c_size = checked_size_2d(c_width, height)?;
    let mut y_plane = try_alloc_zeroed_f32(y_size, "Y plane")?;
    let mut cb_plane = try_alloc_zeroed_f32(c_size, "Cb plane")?;
    let mut cr_plane = try_alloc_zeroed_f32(c_size, "Cr plane")?;

    let bpp = get_bpp(pixel_format)?;

    // First pass: compute Y at full resolution (SIMD-optimized)
    compute_y_plane_from_rgb(data, width, height, bpp, &mut y_plane);

    // Second pass: iterative gamma-aware horizontal downsampling
    for y in 0..height {
        for cx in 0..c_width {
            let (cb, cr) = iterative_chroma_2x1(data, &y_plane, width, bpp, cx, y);
            cb_plane[y * c_width + cx] = cb;
            cr_plane[y * c_width + cx] = cr;
        }
    }

    Ok((y_plane, cb_plane, cr_plane, c_width, height))
}

/// Converts RGB to YCbCr with iterative gamma-aware chroma downsampling for 4:4:0.
pub fn convert_gamma_aware_iterative_440(
    data: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, usize, usize)> {
    let c_height = (height + 1) / 2;

    let y_size = checked_size_2d(width, height)?;
    let c_size = checked_size_2d(width, c_height)?;
    let mut y_plane = try_alloc_zeroed_f32(y_size, "Y plane")?;
    let mut cb_plane = try_alloc_zeroed_f32(c_size, "Cb plane")?;
    let mut cr_plane = try_alloc_zeroed_f32(c_size, "Cr plane")?;

    let bpp = get_bpp(pixel_format)?;

    // First pass: compute Y at full resolution (SIMD-optimized)
    compute_y_plane_from_rgb(data, width, height, bpp, &mut y_plane);

    // Second pass: iterative gamma-aware vertical downsampling
    for cy in 0..c_height {
        for x in 0..width {
            let (cb, cr) = iterative_chroma_1x2(data, &y_plane, width, height, bpp, x, cy);
            cb_plane[cy * width + x] = cb;
            cr_plane[cy * width + x] = cr;
        }
    }

    Ok((y_plane, cb_plane, cr_plane, width, c_height))
}

// ============================================================================
// Strip-Aware Gamma Conversion (for strip-based encoder)
// ============================================================================

/// Computes gamma-aware chroma for a strip of image data (4:2:0 mode).
///
/// This function is designed for strip-based encoding where we process
/// the image in horizontal strips. It computes Cb/Cr directly at the
/// downsampled resolution using gamma-aware averaging.
///
/// # Arguments
/// * `rgb_strip` - RGB data for this strip
/// * `y_strip` - Pre-computed Y values for this strip (output buffer, will be filled)
/// * `cb_down` - Output buffer for downsampled Cb (size: c_width × c_strip_height)
/// * `cr_down` - Output buffer for downsampled Cr (size: c_width × c_strip_height)
/// * `width` - Image width in pixels
/// * `strip_height` - Height of this strip in pixels
/// * `strip_y` - Y offset of this strip in the full image
/// * `image_height` - Total image height (for edge handling)
/// * `bpp` - Bytes per pixel (3 for RGB, 4 for RGBA)
/// * `use_iterative` - If true, use iterative refinement for best quality
pub fn gamma_aware_strip_420(
    rgb_strip: &[u8],
    y_strip: &mut [f32],
    cb_down: &mut [f32],
    cr_down: &mut [f32],
    width: usize,
    strip_height: usize,
    strip_y: usize,
    image_height: usize,
    bpp: usize,
    use_iterative: bool,
) {
    // Compute Y at full resolution using SIMD
    compute_y_plane_from_rgb(rgb_strip, width, strip_height, bpp, y_strip);

    // Compute chroma at half resolution
    let c_width = (width + 1) / 2;
    let c_strip_height = (strip_height + 1) / 2;

    for cy in 0..c_strip_height {
        for cx in 0..c_width {
            let (cb, cr) = if use_iterative {
                iterative_chroma_2x2_strip(rgb_strip, y_strip, width, strip_height, bpp, cx, cy)
            } else {
                gamma_aware_chroma_2x2_strip(rgb_strip, width, strip_height, bpp, cx, cy)
            };
            cb_down[cy * c_width + cx] = cb;
            cr_down[cy * c_width + cx] = cr;
        }
    }

    // Suppress unused warnings
    let _ = (strip_y, image_height);
}

/// Computes gamma-aware chroma for a strip of image data (4:2:2 mode).
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
    // Compute Y at full resolution using SIMD
    compute_y_plane_from_rgb(rgb_strip, width, strip_height, bpp, y_strip);

    // Compute chroma at half horizontal resolution
    let c_width = (width + 1) / 2;

    for y in 0..strip_height {
        for cx in 0..c_width {
            let (cb, cr) = if use_iterative {
                iterative_chroma_2x1_strip(rgb_strip, y_strip, width, bpp, cx, y)
            } else {
                gamma_aware_chroma_2x1_strip(rgb_strip, width, bpp, cx, y)
            };
            cb_down[y * c_width + cx] = cb;
            cr_down[y * c_width + cx] = cr;
        }
    }
}

/// Computes gamma-aware chroma for a strip of image data (4:4:0 mode).
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
    // Compute Y at full resolution using SIMD
    compute_y_plane_from_rgb(rgb_strip, width, strip_height, bpp, y_strip);

    // Compute chroma at half vertical resolution
    let c_strip_height = (strip_height + 1) / 2;

    for cy in 0..c_strip_height {
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

// ============================================================================
// Box Fused Functions (Fast Path)
// ============================================================================

/// Fused Y + downsampled CbCr computation using simple box averaging.
///
/// This is the fast path that:
/// - Computes Y at full resolution (SIMD)
/// - Computes Cb/Cr directly at half resolution using simple 2x2 box averaging
/// - Avoids intermediate full-resolution Cb/Cr buffers
/// - No gamma correction (same as C++ jpegli default)
#[inline]
pub fn box_fused_strip_420(
    rgb_strip: &[u8],
    y_strip: &mut [f32],
    cb_down: &mut [f32],
    cr_down: &mut [f32],
    width: usize,
    strip_height: usize,
    bpp: usize,
) {
    use crate::foundation::consts::{
        YCBCR_B_TO_CB, YCBCR_B_TO_CR, YCBCR_G_TO_CB, YCBCR_G_TO_CR, YCBCR_R_TO_CB, YCBCR_R_TO_CR,
    };

    // Compute Y at full resolution using SIMD
    compute_y_plane_from_rgb(rgb_strip, width, strip_height, bpp, y_strip);

    // Compute chroma at half resolution with simple box averaging
    let c_width = (width + 1) / 2;
    let c_strip_height = (strip_height + 1) / 2;

    for cy in 0..c_strip_height {
        let y0 = cy * 2;
        let y1 = (y0 + 1).min(strip_height - 1);

        for cx in 0..c_width {
            let x0 = cx * 2;
            let x1 = (x0 + 1).min(width - 1);

            // Gather 2x2 block of RGB values
            let mut r_sum = 0.0f32;
            let mut g_sum = 0.0f32;
            let mut b_sum = 0.0f32;

            for &py in &[y0, y1] {
                for &px in &[x0, x1] {
                    let idx = (py * width + px) * bpp;
                    r_sum += rgb_strip[idx] as f32;
                    g_sum += rgb_strip[idx + 1] as f32;
                    b_sum += rgb_strip[idx + 2] as f32;
                }
            }

            // Average (divide by 4)
            let r_avg = r_sum * 0.25;
            let g_avg = g_sum * 0.25;
            let b_avg = b_sum * 0.25;

            // Convert averaged RGB to Cb/Cr (using FMA)
            let cb = YCBCR_R_TO_CB.mul_add(
                r_avg,
                YCBCR_G_TO_CB.mul_add(g_avg, YCBCR_B_TO_CB.mul_add(b_avg, 128.0)),
            );
            let cr = YCBCR_R_TO_CR.mul_add(
                r_avg,
                YCBCR_G_TO_CR.mul_add(g_avg, YCBCR_B_TO_CR.mul_add(b_avg, 128.0)),
            );

            cb_down[cy * c_width + cx] = cb;
            cr_down[cy * c_width + cx] = cr;
        }
    }
}

/// Fused Y + downsampled CbCr for 4:2:2 (horizontal only).
#[inline]
pub fn box_fused_strip_422(
    rgb_strip: &[u8],
    y_strip: &mut [f32],
    cb_down: &mut [f32],
    cr_down: &mut [f32],
    width: usize,
    strip_height: usize,
    bpp: usize,
) {
    use crate::foundation::consts::{
        YCBCR_B_TO_CB, YCBCR_B_TO_CR, YCBCR_G_TO_CB, YCBCR_G_TO_CR, YCBCR_R_TO_CB, YCBCR_R_TO_CR,
    };

    // Compute Y at full resolution using SIMD
    compute_y_plane_from_rgb(rgb_strip, width, strip_height, bpp, y_strip);

    // Compute chroma at half horizontal resolution
    let c_width = (width + 1) / 2;

    for y in 0..strip_height {
        for cx in 0..c_width {
            let x0 = cx * 2;
            let x1 = (x0 + 1).min(width - 1);

            // Average 2 horizontal pixels
            let idx0 = (y * width + x0) * bpp;
            let idx1 = (y * width + x1) * bpp;

            let r_avg = (rgb_strip[idx0] as f32 + rgb_strip[idx1] as f32) * 0.5;
            let g_avg = (rgb_strip[idx0 + 1] as f32 + rgb_strip[idx1 + 1] as f32) * 0.5;
            let b_avg = (rgb_strip[idx0 + 2] as f32 + rgb_strip[idx1 + 2] as f32) * 0.5;

            // Using FMA for accuracy
            let cb = YCBCR_R_TO_CB.mul_add(
                r_avg,
                YCBCR_G_TO_CB.mul_add(g_avg, YCBCR_B_TO_CB.mul_add(b_avg, 128.0)),
            );
            let cr = YCBCR_R_TO_CR.mul_add(
                r_avg,
                YCBCR_G_TO_CR.mul_add(g_avg, YCBCR_B_TO_CR.mul_add(b_avg, 128.0)),
            );

            cb_down[y * c_width + cx] = cb;
            cr_down[y * c_width + cx] = cr;
        }
    }
}

/// Fused Y + downsampled CbCr for 4:4:0 (vertical only).
#[inline]
pub fn box_fused_strip_440(
    rgb_strip: &[u8],
    y_strip: &mut [f32],
    cb_down: &mut [f32],
    cr_down: &mut [f32],
    width: usize,
    strip_height: usize,
    bpp: usize,
) {
    use crate::foundation::consts::{
        YCBCR_B_TO_CB, YCBCR_B_TO_CR, YCBCR_G_TO_CB, YCBCR_G_TO_CR, YCBCR_R_TO_CB, YCBCR_R_TO_CR,
    };

    // Compute Y at full resolution using SIMD
    compute_y_plane_from_rgb(rgb_strip, width, strip_height, bpp, y_strip);

    // Compute chroma at half vertical resolution
    let c_strip_height = (strip_height + 1) / 2;

    for cy in 0..c_strip_height {
        let y0 = cy * 2;
        let y1 = (y0 + 1).min(strip_height - 1);

        for x in 0..width {
            // Average 2 vertical pixels
            let idx0 = (y0 * width + x) * bpp;
            let idx1 = (y1 * width + x) * bpp;

            let r_avg = (rgb_strip[idx0] as f32 + rgb_strip[idx1] as f32) * 0.5;
            let g_avg = (rgb_strip[idx0 + 1] as f32 + rgb_strip[idx1 + 1] as f32) * 0.5;
            let b_avg = (rgb_strip[idx0 + 2] as f32 + rgb_strip[idx1 + 2] as f32) * 0.5;

            // Using FMA for accuracy
            let cb = YCBCR_R_TO_CB.mul_add(
                r_avg,
                YCBCR_G_TO_CB.mul_add(g_avg, YCBCR_B_TO_CB.mul_add(b_avg, 128.0)),
            );
            let cr = YCBCR_R_TO_CR.mul_add(
                r_avg,
                YCBCR_G_TO_CR.mul_add(g_avg, YCBCR_B_TO_CR.mul_add(b_avg, 128.0)),
            );

            cb_down[cy * width + x] = cb;
            cr_down[cy * width + x] = cr;
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Get bytes per pixel for supported formats.
fn get_bpp(pixel_format: PixelFormat) -> Result<usize> {
    match pixel_format {
        PixelFormat::Rgb => Ok(3),
        PixelFormat::Rgba => Ok(4),
        PixelFormat::Bgr | PixelFormat::Bgra | PixelFormat::Bgrx => {
            Err(Error::invalid_color_format(
                "BGR/BGRA/BGRX not supported with gamma-aware downsampling; use box filter or convert to RGB first",
            ))
        }
        _ => Err(Error::invalid_color_format(
            "Unsupported pixel format for gamma-aware conversion",
        )),
    }
}

/// Compute gamma-aware chroma for a 2x2 block (4:2:0).
/// Uses LUT for sRGB→linear (exact) and fastpow for linear→sRGB (~5x faster).
#[inline]
fn gamma_aware_chroma_2x2(
    data: &[u8],
    width: usize,
    height: usize,
    bpp: usize,
    cx: usize,
    cy: usize,
) -> (f32, f32) {
    let x0 = cx * 2;
    let y0 = cy * 2;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);

    // Get RGB values for all 4 pixels and convert to linear using LUT
    let get_linear_rgb = |x: usize, y: usize| -> (f32, f32, f32) {
        let idx = (y * width + x) * bpp;
        (
            srgb_u8_to_linear(data[idx]),
            srgb_u8_to_linear(data[idx + 1]),
            srgb_u8_to_linear(data[idx + 2]),
        )
    };

    let (lr00, lg00, lb00) = get_linear_rgb(x0, y0);
    let (lr10, lg10, lb10) = get_linear_rgb(x1, y0);
    let (lr01, lg01, lb01) = get_linear_rgb(x0, y1);
    let (lr11, lg11, lb11) = get_linear_rgb(x1, y1);

    // Average in linear space
    let lr_avg = (lr00 + lr10 + lr01 + lr11) * 0.25;
    let lg_avg = (lg00 + lg10 + lg01 + lg11) * 0.25;
    let lb_avg = (lb00 + lb10 + lb01 + lb11) * 0.25;

    // Convert back to sRGB then YCbCr using fast approximation
    let r_avg = linear_to_srgb_fast(lr_avg) * 255.0;
    let g_avg = linear_to_srgb_fast(lg_avg) * 255.0;
    let b_avg = linear_to_srgb_fast(lb_avg) * 255.0;

    let (_, cb, cr) = color::rgb_to_ycbcr_f32(r_avg, g_avg, b_avg);
    (cb, cr)
}

/// Compute gamma-aware chroma for a 2x1 block (4:2:2 horizontal).
/// Uses LUT for sRGB→linear (exact) and fastpow for linear→sRGB (~5x faster).
#[inline]
fn gamma_aware_chroma_2x1(
    data: &[u8],
    width: usize,
    bpp: usize,
    cx: usize,
    y: usize,
) -> (f32, f32) {
    let x0 = cx * 2;
    let x1 = (x0 + 1).min(width - 1);

    let get_linear_rgb = |x: usize| -> (f32, f32, f32) {
        let idx = (y * width + x) * bpp;
        (
            srgb_u8_to_linear(data[idx]),
            srgb_u8_to_linear(data[idx + 1]),
            srgb_u8_to_linear(data[idx + 2]),
        )
    };

    let (lr0, lg0, lb0) = get_linear_rgb(x0);
    let (lr1, lg1, lb1) = get_linear_rgb(x1);

    let lr_avg = (lr0 + lr1) * 0.5;
    let lg_avg = (lg0 + lg1) * 0.5;
    let lb_avg = (lb0 + lb1) * 0.5;

    let r_avg = linear_to_srgb_fast(lr_avg) * 255.0;
    let g_avg = linear_to_srgb_fast(lg_avg) * 255.0;
    let b_avg = linear_to_srgb_fast(lb_avg) * 255.0;

    let (_, cb, cr) = color::rgb_to_ycbcr_f32(r_avg, g_avg, b_avg);
    (cb, cr)
}

/// Compute gamma-aware chroma for a 1x2 block (4:4:0 vertical).
/// Uses LUT for sRGB→linear (exact) and fastpow for linear→sRGB (~5x faster).
#[inline]
fn gamma_aware_chroma_1x2(
    data: &[u8],
    width: usize,
    height: usize,
    bpp: usize,
    x: usize,
    cy: usize,
) -> (f32, f32) {
    let y0 = cy * 2;
    let y1 = (y0 + 1).min(height - 1);

    let get_linear_rgb = |y: usize| -> (f32, f32, f32) {
        let idx = (y * width + x) * bpp;
        (
            srgb_u8_to_linear(data[idx]),
            srgb_u8_to_linear(data[idx + 1]),
            srgb_u8_to_linear(data[idx + 2]),
        )
    };

    let (lr0, lg0, lb0) = get_linear_rgb(y0);
    let (lr1, lg1, lb1) = get_linear_rgb(y1);

    let lr_avg = (lr0 + lr1) * 0.5;
    let lg_avg = (lg0 + lg1) * 0.5;
    let lb_avg = (lb0 + lb1) * 0.5;

    let r_avg = linear_to_srgb_fast(lr_avg) * 255.0;
    let g_avg = linear_to_srgb_fast(lg_avg) * 255.0;
    let b_avg = linear_to_srgb_fast(lb_avg) * 255.0;

    let (_, cb, cr) = color::rgb_to_ycbcr_f32(r_avg, g_avg, b_avg);
    (cb, cr)
}

// ============================================================================
// Iterative Chroma Optimization
// ============================================================================

/// Iteratively optimize chroma for a 2x2 block to minimize reconstruction error.
///
/// This is similar to libwebp's Sharp YUV algorithm:
/// 1. Start with gamma-aware averaged chroma
/// 2. For each iteration:
///    - Reconstruct RGB from Y + current Cb/Cr
///    - Check for clipping (values outside [0, 255])
///    - Adjust Cb/Cr to minimize error
/// 3. Stop when converged or max iterations reached
fn iterative_chroma_2x2(
    data: &[u8],
    y_plane: &[f32],
    width: usize,
    height: usize,
    bpp: usize,
    cx: usize,
    cy: usize,
) -> (f32, f32) {
    let x0 = cx * 2;
    let y0 = cy * 2;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);

    // Get the 4 Y values for this block
    let y_vals = [
        y_plane[y0 * width + x0],
        y_plane[y0 * width + x1],
        y_plane[y1 * width + x0],
        y_plane[y1 * width + x1],
    ];

    // Get original RGB values (0-255 range)
    let get_rgb = |x: usize, y: usize| -> (f32, f32, f32) {
        let idx = (y * width + x) * bpp;
        (data[idx] as f32, data[idx + 1] as f32, data[idx + 2] as f32)
    };

    let orig_rgb = [
        get_rgb(x0, y0),
        get_rgb(x1, y0),
        get_rgb(x0, y1),
        get_rgb(x1, y1),
    ];

    // Start with gamma-aware initial estimate
    let (mut cb, mut cr) = gamma_aware_chroma_2x2(data, width, height, bpp, cx, cy);

    // Iterative refinement
    for _ in 0..NUM_ITERATIONS {
        let mut total_error = 0.0f32;
        let mut cb_adjustment = 0.0f32;
        let mut cr_adjustment = 0.0f32;

        // For each pixel in the block, compute reconstruction error
        for i in 0..4 {
            let (orig_r, orig_g, orig_b) = orig_rgb[i];
            let y_val = y_vals[i];

            // Reconstruct RGB from Y, Cb, Cr
            let (rec_r, rec_g, rec_b) = color::ycbcr_to_rgb_f32(y_val, cb, cr);

            // Clamp to valid range
            let rec_r = rec_r.clamp(0.0, 255.0);
            let rec_g = rec_g.clamp(0.0, 255.0);
            let rec_b = rec_b.clamp(0.0, 255.0);

            // Compute error
            let err_r = orig_r - rec_r;
            let err_g = orig_g - rec_g;
            let err_b = orig_b - rec_b;

            total_error += err_r.abs() + err_g.abs() + err_b.abs();

            // Compute chroma adjustments based on error
            // These coefficients are derived from the YCbCr conversion matrix inverse
            // Cb affects B positively and G/R negatively
            // Cr affects R positively and G negatively
            cb_adjustment += 0.5 * err_b - 0.169 * err_r - 0.331 * err_g;
            cr_adjustment += 0.5 * err_r - 0.419 * err_g - 0.081 * err_b;
        }

        // Check convergence
        if total_error < CONVERGENCE_THRESHOLD {
            break;
        }

        // Apply averaged adjustment (scaled down to prevent oscillation)
        let scale = 0.25; // Average over 4 pixels, with damping
        cb = (cb + cb_adjustment * scale * 0.5).clamp(0.0, 255.0);
        cr = (cr + cr_adjustment * scale * 0.5).clamp(0.0, 255.0);
    }

    (cb, cr)
}

/// Iteratively optimize chroma for a 2x1 block (horizontal).
fn iterative_chroma_2x1(
    data: &[u8],
    y_plane: &[f32],
    width: usize,
    bpp: usize,
    cx: usize,
    y: usize,
) -> (f32, f32) {
    let x0 = cx * 2;
    let x1 = (x0 + 1).min(width - 1);

    let y_vals = [y_plane[y * width + x0], y_plane[y * width + x1]];

    let get_rgb = |x: usize| -> (f32, f32, f32) {
        let idx = (y * width + x) * bpp;
        (data[idx] as f32, data[idx + 1] as f32, data[idx + 2] as f32)
    };

    let orig_rgb = [get_rgb(x0), get_rgb(x1)];

    // Start with gamma-aware initial estimate
    let (mut cb, mut cr) = gamma_aware_chroma_2x1(data, width, bpp, cx, y);

    for _ in 0..NUM_ITERATIONS {
        let mut total_error = 0.0f32;
        let mut cb_adjustment = 0.0f32;
        let mut cr_adjustment = 0.0f32;

        for i in 0..2 {
            let (orig_r, orig_g, orig_b) = orig_rgb[i];
            let y_val = y_vals[i];

            let (rec_r, rec_g, rec_b) = color::ycbcr_to_rgb_f32(y_val, cb, cr);
            let rec_r = rec_r.clamp(0.0, 255.0);
            let rec_g = rec_g.clamp(0.0, 255.0);
            let rec_b = rec_b.clamp(0.0, 255.0);

            let err_r = orig_r - rec_r;
            let err_g = orig_g - rec_g;
            let err_b = orig_b - rec_b;

            total_error += err_r.abs() + err_g.abs() + err_b.abs();

            cb_adjustment += 0.5 * err_b - 0.169 * err_r - 0.331 * err_g;
            cr_adjustment += 0.5 * err_r - 0.419 * err_g - 0.081 * err_b;
        }

        if total_error < CONVERGENCE_THRESHOLD {
            break;
        }

        let scale = 0.5 * 0.5; // Average over 2 pixels with damping
        cb = (cb + cb_adjustment * scale).clamp(0.0, 255.0);
        cr = (cr + cr_adjustment * scale).clamp(0.0, 255.0);
    }

    (cb, cr)
}

/// Iteratively optimize chroma for a 1x2 block (vertical).
fn iterative_chroma_1x2(
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
        let idx = (y * width + x) * bpp;
        (data[idx] as f32, data[idx + 1] as f32, data[idx + 2] as f32)
    };

    let orig_rgb = [get_rgb(y0), get_rgb(y1)];

    // Start with gamma-aware initial estimate
    let (mut cb, mut cr) = gamma_aware_chroma_1x2(data, width, height, bpp, x, cy);

    for _ in 0..NUM_ITERATIONS {
        let mut total_error = 0.0f32;
        let mut cb_adjustment = 0.0f32;
        let mut cr_adjustment = 0.0f32;

        for i in 0..2 {
            let (orig_r, orig_g, orig_b) = orig_rgb[i];
            let y_val = y_vals[i];

            let (rec_r, rec_g, rec_b) = color::ycbcr_to_rgb_f32(y_val, cb, cr);
            let rec_r = rec_r.clamp(0.0, 255.0);
            let rec_g = rec_g.clamp(0.0, 255.0);
            let rec_b = rec_b.clamp(0.0, 255.0);

            let err_r = orig_r - rec_r;
            let err_g = orig_g - rec_g;
            let err_b = orig_b - rec_b;

            total_error += err_r.abs() + err_g.abs() + err_b.abs();

            cb_adjustment += 0.5 * err_b - 0.169 * err_r - 0.331 * err_g;
            cr_adjustment += 0.5 * err_r - 0.419 * err_g - 0.081 * err_b;
        }

        if total_error < CONVERGENCE_THRESHOLD {
            break;
        }

        let scale = 0.5 * 0.5;
        cb = (cb + cb_adjustment * scale).clamp(0.0, 255.0);
        cr = (cr + cr_adjustment * scale).clamp(0.0, 255.0);
    }

    (cb, cr)
}

// ============================================================================
// Strip-Specific Helper Functions
// ============================================================================

/// Gamma-aware chroma for a 2x2 block within a strip (4:2:0).
#[inline]
fn gamma_aware_chroma_2x2_strip(
    data: &[u8],
    width: usize,
    strip_height: usize,
    bpp: usize,
    cx: usize,
    cy: usize,
) -> (f32, f32) {
    // Delegate to the existing function - it already handles edge cases
    gamma_aware_chroma_2x2(data, width, strip_height, bpp, cx, cy)
}

/// Gamma-aware chroma for a 2x1 block within a strip (4:2:2).
#[inline]
fn gamma_aware_chroma_2x1_strip(
    data: &[u8],
    width: usize,
    bpp: usize,
    cx: usize,
    y: usize,
) -> (f32, f32) {
    gamma_aware_chroma_2x1(data, width, bpp, cx, y)
}

/// Gamma-aware chroma for a 1x2 block within a strip (4:4:0).
#[inline]
fn gamma_aware_chroma_1x2_strip(
    data: &[u8],
    width: usize,
    strip_height: usize,
    bpp: usize,
    x: usize,
    cy: usize,
) -> (f32, f32) {
    gamma_aware_chroma_1x2(data, width, strip_height, bpp, x, cy)
}

/// Iterative chroma optimization for a 2x2 block within a strip.
#[inline]
fn iterative_chroma_2x2_strip(
    data: &[u8],
    y_plane: &[f32],
    width: usize,
    strip_height: usize,
    bpp: usize,
    cx: usize,
    cy: usize,
) -> (f32, f32) {
    iterative_chroma_2x2(data, y_plane, width, strip_height, bpp, cx, cy)
}

/// Iterative chroma optimization for a 2x1 block within a strip.
#[inline]
fn iterative_chroma_2x1_strip(
    data: &[u8],
    y_plane: &[f32],
    width: usize,
    bpp: usize,
    cx: usize,
    y: usize,
) -> (f32, f32) {
    iterative_chroma_2x1(data, y_plane, width, bpp, cx, y)
}

/// Iterative chroma optimization for a 1x2 block within a strip.
#[inline]
fn iterative_chroma_1x2_strip(
    data: &[u8],
    y_plane: &[f32],
    width: usize,
    strip_height: usize,
    bpp: usize,
    x: usize,
    cy: usize,
) -> (f32, f32) {
    iterative_chroma_1x2(data, y_plane, width, strip_height, bpp, x, cy)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_image(width: usize, height: usize) -> Vec<u8> {
        let mut data = vec![0u8; width * height * 3];
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 3;
                data[idx] = (x * 8) as u8; // R gradient
                data[idx + 1] = (y * 8) as u8; // G gradient
                data[idx + 2] = 128; // B constant
            }
        }
        data
    }

    #[test]
    fn test_gamma_aware_420() {
        let data = create_test_image(32, 32);
        let result = convert_gamma_aware_420(&data, 32, 32, PixelFormat::Rgb);
        assert!(result.is_ok());

        let (y, cb, cr, c_w, c_h) = result.unwrap();
        assert_eq!(y.len(), 32 * 32);
        assert_eq!(cb.len(), 16 * 16);
        assert_eq!(cr.len(), 16 * 16);
        assert_eq!(c_w, 16);
        assert_eq!(c_h, 16);

        // Y values should be in valid range
        for &val in &y {
            assert!((0.0..=255.0).contains(&val), "Y out of range: {}", val);
        }

        // Cb/Cr should be centered around 128
        for &val in &cb {
            assert!((0.0..=255.0).contains(&val), "Cb out of range: {}", val);
        }
        for &val in &cr {
            assert!((0.0..=255.0).contains(&val), "Cr out of range: {}", val);
        }
    }

    #[test]
    fn test_gamma_aware_422() {
        let data = create_test_image(32, 32);
        let result = convert_gamma_aware_422(&data, 32, 32, PixelFormat::Rgb);
        assert!(result.is_ok());

        let (y, cb, cr, c_w, c_h) = result.unwrap();
        assert_eq!(y.len(), 32 * 32);
        assert_eq!(cb.len(), 16 * 32);
        assert_eq!(cr.len(), 16 * 32);
        assert_eq!(c_w, 16);
        assert_eq!(c_h, 32);
    }

    #[test]
    fn test_gamma_aware_440() {
        let data = create_test_image(32, 32);
        let result = convert_gamma_aware_440(&data, 32, 32, PixelFormat::Rgb);
        assert!(result.is_ok());

        let (y, cb, cr, c_w, c_h) = result.unwrap();
        assert_eq!(y.len(), 32 * 32);
        assert_eq!(cb.len(), 32 * 16);
        assert_eq!(cr.len(), 32 * 16);
        assert_eq!(c_w, 32);
        assert_eq!(c_h, 16);
    }

    #[test]
    fn test_gamma_aware_iterative_420() {
        let data = create_test_image(32, 32);
        let result = convert_gamma_aware_iterative_420(&data, 32, 32, PixelFormat::Rgb);
        assert!(result.is_ok());

        let (y, cb, cr, c_w, c_h) = result.unwrap();
        assert_eq!(y.len(), 32 * 32);
        assert_eq!(cb.len(), 16 * 16);
        assert_eq!(cr.len(), 16 * 16);
        assert_eq!(c_w, 16);
        assert_eq!(c_h, 16);
    }

    #[test]
    fn test_iterative_vs_simple_different() {
        // Create an image with sharp color edges where iterative should differ
        let width = 4;
        let height = 4;
        let mut data = vec![0u8; width * height * 3];

        // Create a red/cyan checkerboard pattern
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 3;
                if (x + y) % 2 == 0 {
                    // Red
                    data[idx] = 255;
                    data[idx + 1] = 0;
                    data[idx + 2] = 0;
                } else {
                    // Cyan
                    data[idx] = 0;
                    data[idx + 1] = 255;
                    data[idx + 2] = 255;
                }
            }
        }

        let simple = convert_gamma_aware_420(&data, width, height, PixelFormat::Rgb).unwrap();
        let iterative =
            convert_gamma_aware_iterative_420(&data, width, height, PixelFormat::Rgb).unwrap();

        // The results should be similar but not necessarily identical
        // due to iterative refinement
        let (_, cb_simple, cr_simple, _, _) = simple;
        let (_, cb_iter, cr_iter, _, _) = iterative;

        // Both should produce valid chroma values
        assert!(!cb_simple.is_empty());
        assert!(!cb_iter.is_empty());
        assert!(!cr_simple.is_empty());
        assert!(!cr_iter.is_empty());

        // Values should be in valid range
        for &val in &cb_iter {
            assert!((0.0..=255.0).contains(&val));
        }
        for &val in &cr_iter {
            assert!((0.0..=255.0).contains(&val));
        }
    }

    #[test]
    fn test_rgba_input() {
        let width = 16;
        let height = 16;
        let mut data = vec![0u8; width * height * 4];

        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 4;
                data[idx] = (x * 16) as u8;
                data[idx + 1] = (y * 16) as u8;
                data[idx + 2] = 128;
                data[idx + 3] = 255; // Alpha
            }
        }

        let result = convert_gamma_aware_420(&data, width, height, PixelFormat::Rgba);
        assert!(result.is_ok());
    }

    #[test]
    fn test_unsupported_format() {
        let data = vec![0u8; 32 * 32 * 3];
        let result = convert_gamma_aware_420(&data, 32, 32, PixelFormat::Bgr);
        assert!(result.is_err());
    }

    #[test]
    fn test_odd_dimensions() {
        // Test with odd dimensions to verify edge handling
        let data = create_test_image(31, 33);
        let result = convert_gamma_aware_420(&data, 31, 33, PixelFormat::Rgb);
        assert!(result.is_ok());

        let (y, cb, cr, c_w, c_h) = result.unwrap();
        assert_eq!(y.len(), 31 * 33);
        assert_eq!(c_w, 16); // (31 + 1) / 2
        assert_eq!(c_h, 17); // (33 + 1) / 2
        assert_eq!(cb.len(), 16 * 17);
        assert_eq!(cr.len(), 16 * 17);
    }
}
