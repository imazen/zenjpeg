//! SIMD-optimized encoding functions.
//!
//! Contains SIMD implementations for encoder hot paths:
//! - Chroma downsampling (2x2, 2x1, 1x2)
//! - RGB to YCbCr color conversion
//!
//! All functions have tests to verify parity with scalar implementations.

use wide::f32x8;

use crate::consts::{
    YCBCR_B_TO_CB, YCBCR_B_TO_CR, YCBCR_B_TO_Y, YCBCR_G_TO_CB, YCBCR_G_TO_CR, YCBCR_G_TO_Y,
    YCBCR_R_TO_CB, YCBCR_R_TO_CR, YCBCR_R_TO_Y,
};

// ============================================================================
// Memory Allocation Helpers
// ============================================================================

/// Allocate a Vec<f32> without zeroing (all elements will be written before use).
/// # Safety
/// The caller MUST write to all elements before reading any.
#[inline(always)]
fn alloc_uninit_f32(len: usize) -> Vec<f32> {
    let mut vec = Vec::with_capacity(len);
    // SAFETY: We will write to all elements before reading.
    // This is safe because f32 has no drop logic and no invalid bit patterns.
    unsafe { vec.set_len(len) };
    vec
}

// ============================================================================
// Chroma Downsampling (2x2 box filter)
// ============================================================================

/// SIMD-optimized 2x2 box filter downsampling.
///
/// Processes 8 output pixels at a time (requires 16 input pixels per row).
///
/// # Arguments
/// * `plane` - Input plane (f32, full resolution)
/// * `width` - Input width
/// * `height` - Input height
///
/// # Returns
/// Downsampled plane at half resolution
pub fn downsample_2x2_simd(plane: &[f32], width: usize, height: usize) -> Vec<f32> {
    let new_width = (width + 1) / 2;
    let new_height = (height + 1) / 2;
    // Use uninit allocation since we write to every element
    let mut result = alloc_uninit_f32(new_width * new_height);

    let scale = f32x8::splat(0.25);
    let chunks = new_width / 8;

    for y in 0..new_height {
        let y0 = y * 2;
        let y1 = (y0 + 1).min(height - 1);
        let out_row_start = y * new_width;

        // SIMD path: process 8 output pixels at a time
        for chunk in 0..chunks {
            let out_x = chunk * 8;
            let in_x = out_x * 2;

            // Load 16 consecutive pixels from row y0 (evens and odds)
            // p00 = [in_x+0, in_x+2, in_x+4, ...] (even columns)
            // p10 = [in_x+1, in_x+3, in_x+5, ...] (odd columns)
            let row0_idx = y0 * width + in_x;
            let row1_idx = y1 * width + in_x;

            // Gather even/odd from row 0
            let (p00, p10) = gather_even_odd_x8(plane, row0_idx, width);
            // Gather even/odd from row 1
            let (p01, p11) = gather_even_odd_x8(plane, row1_idx, width);

            // Box filter: (p00 + p10 + p01 + p11) * 0.25
            let sum = p00 + p10 + p01 + p11;
            let avg = sum * scale;

            // Store result
            let avg_arr: [f32; 8] = avg.into();
            result[out_row_start + out_x..out_row_start + out_x + 8].copy_from_slice(&avg_arr);
        }

        // Scalar remainder
        for out_x in (chunks * 8)..new_width {
            let x0 = out_x * 2;
            let x1 = (x0 + 1).min(width - 1);

            let p00 = plane[y0 * width + x0];
            let p10 = plane[y0 * width + x1];
            let p01 = plane[y1 * width + x0];
            let p11 = plane[y1 * width + x1];

            result[out_row_start + out_x] = (p00 + p10 + p01 + p11) * 0.25;
        }
    }

    result
}

/// Gather even and odd indexed elements from a row into two f32x8 vectors.
///
/// Given input [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, ...]:
/// - evens = [a, c, e, g, i, k, m, o]
/// - odds = [b, d, f, h, j, l, n, p]
#[inline(always)]
fn gather_even_odd_x8(plane: &[f32], start_idx: usize, width: usize) -> (f32x8, f32x8) {
    // Boundary-safe gather
    let get = |offset: usize| -> f32 {
        let idx = start_idx + offset;
        if idx < plane.len() {
            plane[idx]
        } else {
            plane[plane.len() - 1]
        }
    };

    let evens = f32x8::from([
        get(0),
        get(2),
        get(4),
        get(6),
        get(8),
        get(10),
        get(12),
        get(14),
    ]);

    let odds = f32x8::from([
        get(1.min(width - 1)),
        get(3.min(width - 1)),
        get(5.min(width - 1)),
        get(7.min(width - 1)),
        get(9.min(width - 1)),
        get(11.min(width - 1)),
        get(13.min(width - 1)),
        get(15.min(width - 1)),
    ]);

    (evens, odds)
}

// ============================================================================
// RGB to YCbCr Color Conversion
// ============================================================================

/// SIMD-optimized RGB to YCbCr conversion for entire image.
///
/// Processes 8 pixels at a time, converting from interleaved RGB to planar YCbCr.
///
/// # Arguments
/// * `rgb_data` - Input RGB data (3 bytes per pixel, interleaved)
/// * `num_pixels` - Number of pixels
///
/// # Returns
/// Tuple of (Y plane, Cb plane, Cr plane) as f32 vectors
pub fn rgb_to_ycbcr_planes_simd(rgb_data: &[u8], num_pixels: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    // Use uninit allocation since we write to every element
    let mut y_plane = alloc_uninit_f32(num_pixels);
    let mut cb_plane = alloc_uninit_f32(num_pixels);
    let mut cr_plane = alloc_uninit_f32(num_pixels);

    // Coefficients as SIMD vectors
    let r_to_y = f32x8::splat(YCBCR_R_TO_Y);
    let g_to_y = f32x8::splat(YCBCR_G_TO_Y);
    let b_to_y = f32x8::splat(YCBCR_B_TO_Y);

    let r_to_cb = f32x8::splat(YCBCR_R_TO_CB);
    let g_to_cb = f32x8::splat(YCBCR_G_TO_CB);
    let b_to_cb = f32x8::splat(YCBCR_B_TO_CB);

    let r_to_cr = f32x8::splat(YCBCR_R_TO_CR);
    let g_to_cr = f32x8::splat(YCBCR_G_TO_CR);
    let b_to_cr = f32x8::splat(YCBCR_B_TO_CR);

    let offset_128 = f32x8::splat(128.0);

    let chunks = num_pixels / 8;

    for chunk in 0..chunks {
        let pixel_idx = chunk * 8;
        let rgb_idx = pixel_idx * 3;

        // Gather 8 R, G, B values from interleaved data
        let r = f32x8::from([
            rgb_data[rgb_idx] as f32,
            rgb_data[rgb_idx + 3] as f32,
            rgb_data[rgb_idx + 6] as f32,
            rgb_data[rgb_idx + 9] as f32,
            rgb_data[rgb_idx + 12] as f32,
            rgb_data[rgb_idx + 15] as f32,
            rgb_data[rgb_idx + 18] as f32,
            rgb_data[rgb_idx + 21] as f32,
        ]);

        let g = f32x8::from([
            rgb_data[rgb_idx + 1] as f32,
            rgb_data[rgb_idx + 4] as f32,
            rgb_data[rgb_idx + 7] as f32,
            rgb_data[rgb_idx + 10] as f32,
            rgb_data[rgb_idx + 13] as f32,
            rgb_data[rgb_idx + 16] as f32,
            rgb_data[rgb_idx + 19] as f32,
            rgb_data[rgb_idx + 22] as f32,
        ]);

        let b = f32x8::from([
            rgb_data[rgb_idx + 2] as f32,
            rgb_data[rgb_idx + 5] as f32,
            rgb_data[rgb_idx + 8] as f32,
            rgb_data[rgb_idx + 11] as f32,
            rgb_data[rgb_idx + 14] as f32,
            rgb_data[rgb_idx + 17] as f32,
            rgb_data[rgb_idx + 20] as f32,
            rgb_data[rgb_idx + 23] as f32,
        ]);

        // Compute Y, Cb, Cr
        let y = r * r_to_y + g * g_to_y + b * b_to_y;
        let cb = offset_128 + r * r_to_cb + g * g_to_cb + b * b_to_cb;
        let cr = offset_128 + r * r_to_cr + g * g_to_cr + b * b_to_cr;

        // Store results
        let y_arr: [f32; 8] = y.into();
        let cb_arr: [f32; 8] = cb.into();
        let cr_arr: [f32; 8] = cr.into();

        y_plane[pixel_idx..pixel_idx + 8].copy_from_slice(&y_arr);
        cb_plane[pixel_idx..pixel_idx + 8].copy_from_slice(&cb_arr);
        cr_plane[pixel_idx..pixel_idx + 8].copy_from_slice(&cr_arr);
    }

    // Scalar remainder
    for i in (chunks * 8)..num_pixels {
        let rgb_idx = i * 3;
        let r = rgb_data[rgb_idx] as f32;
        let g = rgb_data[rgb_idx + 1] as f32;
        let b = rgb_data[rgb_idx + 2] as f32;

        y_plane[i] = YCBCR_R_TO_Y * r + YCBCR_G_TO_Y * g + YCBCR_B_TO_Y * b;
        cb_plane[i] = 128.0 + YCBCR_R_TO_CB * r + YCBCR_G_TO_CB * g + YCBCR_B_TO_CB * b;
        cr_plane[i] = 128.0 + YCBCR_R_TO_CR * r + YCBCR_G_TO_CR * g + YCBCR_B_TO_CR * b;
    }

    (y_plane, cb_plane, cr_plane)
}

/// SIMD-optimized RGBA to YCbCr conversion for entire image (ignores alpha).
pub fn rgba_to_ycbcr_planes_simd(rgba_data: &[u8], num_pixels: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    // Use uninit allocation since we write to every element
    let mut y_plane = alloc_uninit_f32(num_pixels);
    let mut cb_plane = alloc_uninit_f32(num_pixels);
    let mut cr_plane = alloc_uninit_f32(num_pixels);

    let r_to_y = f32x8::splat(YCBCR_R_TO_Y);
    let g_to_y = f32x8::splat(YCBCR_G_TO_Y);
    let b_to_y = f32x8::splat(YCBCR_B_TO_Y);

    let r_to_cb = f32x8::splat(YCBCR_R_TO_CB);
    let g_to_cb = f32x8::splat(YCBCR_G_TO_CB);
    let b_to_cb = f32x8::splat(YCBCR_B_TO_CB);

    let r_to_cr = f32x8::splat(YCBCR_R_TO_CR);
    let g_to_cr = f32x8::splat(YCBCR_G_TO_CR);
    let b_to_cr = f32x8::splat(YCBCR_B_TO_CR);

    let offset_128 = f32x8::splat(128.0);

    let chunks = num_pixels / 8;

    for chunk in 0..chunks {
        let pixel_idx = chunk * 8;
        let rgba_idx = pixel_idx * 4;

        // Gather 8 R, G, B values from interleaved RGBA data (stride 4)
        let r = f32x8::from([
            rgba_data[rgba_idx] as f32,
            rgba_data[rgba_idx + 4] as f32,
            rgba_data[rgba_idx + 8] as f32,
            rgba_data[rgba_idx + 12] as f32,
            rgba_data[rgba_idx + 16] as f32,
            rgba_data[rgba_idx + 20] as f32,
            rgba_data[rgba_idx + 24] as f32,
            rgba_data[rgba_idx + 28] as f32,
        ]);

        let g = f32x8::from([
            rgba_data[rgba_idx + 1] as f32,
            rgba_data[rgba_idx + 5] as f32,
            rgba_data[rgba_idx + 9] as f32,
            rgba_data[rgba_idx + 13] as f32,
            rgba_data[rgba_idx + 17] as f32,
            rgba_data[rgba_idx + 21] as f32,
            rgba_data[rgba_idx + 25] as f32,
            rgba_data[rgba_idx + 29] as f32,
        ]);

        let b = f32x8::from([
            rgba_data[rgba_idx + 2] as f32,
            rgba_data[rgba_idx + 6] as f32,
            rgba_data[rgba_idx + 10] as f32,
            rgba_data[rgba_idx + 14] as f32,
            rgba_data[rgba_idx + 18] as f32,
            rgba_data[rgba_idx + 22] as f32,
            rgba_data[rgba_idx + 26] as f32,
            rgba_data[rgba_idx + 30] as f32,
        ]);

        let y = r * r_to_y + g * g_to_y + b * b_to_y;
        let cb = offset_128 + r * r_to_cb + g * g_to_cb + b * b_to_cb;
        let cr = offset_128 + r * r_to_cr + g * g_to_cr + b * b_to_cr;

        let y_arr: [f32; 8] = y.into();
        let cb_arr: [f32; 8] = cb.into();
        let cr_arr: [f32; 8] = cr.into();

        y_plane[pixel_idx..pixel_idx + 8].copy_from_slice(&y_arr);
        cb_plane[pixel_idx..pixel_idx + 8].copy_from_slice(&cb_arr);
        cr_plane[pixel_idx..pixel_idx + 8].copy_from_slice(&cr_arr);
    }

    // Scalar remainder
    for i in (chunks * 8)..num_pixels {
        let rgba_idx = i * 4;
        let r = rgba_data[rgba_idx] as f32;
        let g = rgba_data[rgba_idx + 1] as f32;
        let b = rgba_data[rgba_idx + 2] as f32;

        y_plane[i] = YCBCR_R_TO_Y * r + YCBCR_G_TO_Y * g + YCBCR_B_TO_Y * b;
        cb_plane[i] = 128.0 + YCBCR_R_TO_CB * r + YCBCR_G_TO_CB * g + YCBCR_B_TO_CB * b;
        cr_plane[i] = 128.0 + YCBCR_R_TO_CR * r + YCBCR_G_TO_CR * g + YCBCR_B_TO_CR * b;
    }

    (y_plane, cb_plane, cr_plane)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-5;

    #[test]
    fn test_downsample_2x2_simd_matches_scalar() {
        // Create test data (16x16 gradient)
        let width = 16;
        let height = 16;
        let plane: Vec<f32> = (0..width * height)
            .map(|i| {
                let x = i % width;
                let y = i / width;
                (x + y * 10) as f32
            })
            .collect();

        // SIMD version
        let simd_result = downsample_2x2_simd(&plane, width, height);

        // Scalar reference
        let new_width = (width + 1) / 2;
        let new_height = (height + 1) / 2;
        let mut scalar_result = vec![0.0f32; new_width * new_height];

        for y in 0..new_height {
            for x in 0..new_width {
                let x0 = x * 2;
                let y0 = y * 2;
                let x1 = (x0 + 1).min(width - 1);
                let y1 = (y0 + 1).min(height - 1);

                let p00 = plane[y0 * width + x0];
                let p10 = plane[y0 * width + x1];
                let p01 = plane[y1 * width + x0];
                let p11 = plane[y1 * width + x1];

                scalar_result[y * new_width + x] = (p00 + p10 + p01 + p11) * 0.25;
            }
        }

        // Compare
        assert_eq!(simd_result.len(), scalar_result.len());
        for i in 0..simd_result.len() {
            let diff = (simd_result[i] - scalar_result[i]).abs();
            assert!(
                diff < EPSILON,
                "Downsample mismatch at {}: SIMD={}, scalar={}, diff={}",
                i, simd_result[i], scalar_result[i], diff
            );
        }
    }

    #[test]
    fn test_rgb_to_ycbcr_simd_matches_scalar() {
        // Create test RGB data (24 pixels for good SIMD coverage)
        let num_pixels = 24;
        let rgb_data: Vec<u8> = (0..num_pixels * 3)
            .map(|i| ((i * 11 + 7) % 256) as u8)
            .collect();

        // SIMD version
        let (y_simd, cb_simd, cr_simd) = rgb_to_ycbcr_planes_simd(&rgb_data, num_pixels);

        // Scalar reference
        let mut y_scalar = vec![0.0f32; num_pixels];
        let mut cb_scalar = vec![0.0f32; num_pixels];
        let mut cr_scalar = vec![0.0f32; num_pixels];

        for i in 0..num_pixels {
            let r = rgb_data[i * 3] as f32;
            let g = rgb_data[i * 3 + 1] as f32;
            let b = rgb_data[i * 3 + 2] as f32;

            y_scalar[i] = YCBCR_R_TO_Y * r + YCBCR_G_TO_Y * g + YCBCR_B_TO_Y * b;
            cb_scalar[i] = 128.0 + YCBCR_R_TO_CB * r + YCBCR_G_TO_CB * g + YCBCR_B_TO_CB * b;
            cr_scalar[i] = 128.0 + YCBCR_R_TO_CR * r + YCBCR_G_TO_CR * g + YCBCR_B_TO_CR * b;
        }

        // Compare
        for i in 0..num_pixels {
            assert!(
                (y_simd[i] - y_scalar[i]).abs() < EPSILON,
                "Y mismatch at {}: SIMD={}, scalar={}",
                i, y_simd[i], y_scalar[i]
            );
            assert!(
                (cb_simd[i] - cb_scalar[i]).abs() < EPSILON,
                "Cb mismatch at {}: SIMD={}, scalar={}",
                i, cb_simd[i], cb_scalar[i]
            );
            assert!(
                (cr_simd[i] - cr_scalar[i]).abs() < EPSILON,
                "Cr mismatch at {}: SIMD={}, scalar={}",
                i, cr_simd[i], cr_scalar[i]
            );
        }
    }

    #[test]
    fn test_rgba_to_ycbcr_simd_matches_scalar() {
        let num_pixels = 24;
        let rgba_data: Vec<u8> = (0..num_pixels * 4)
            .map(|i| ((i * 13 + 3) % 256) as u8)
            .collect();

        let (y_simd, cb_simd, cr_simd) = rgba_to_ycbcr_planes_simd(&rgba_data, num_pixels);

        for i in 0..num_pixels {
            let r = rgba_data[i * 4] as f32;
            let g = rgba_data[i * 4 + 1] as f32;
            let b = rgba_data[i * 4 + 2] as f32;

            let y_scalar = YCBCR_R_TO_Y * r + YCBCR_G_TO_Y * g + YCBCR_B_TO_Y * b;
            let cb_scalar = 128.0 + YCBCR_R_TO_CB * r + YCBCR_G_TO_CB * g + YCBCR_B_TO_CB * b;
            let cr_scalar = 128.0 + YCBCR_R_TO_CR * r + YCBCR_G_TO_CR * g + YCBCR_B_TO_CR * b;

            assert!(
                (y_simd[i] - y_scalar).abs() < EPSILON,
                "Y mismatch at {}: SIMD={}, scalar={}",
                i, y_simd[i], y_scalar
            );
            assert!(
                (cb_simd[i] - cb_scalar).abs() < EPSILON,
                "Cb mismatch at {}: SIMD={}, scalar={}",
                i, cb_simd[i], cb_scalar
            );
            assert!(
                (cr_simd[i] - cr_scalar).abs() < EPSILON,
                "Cr mismatch at {}: SIMD={}, scalar={}",
                i, cr_simd[i], cr_scalar
            );
        }
    }
}
