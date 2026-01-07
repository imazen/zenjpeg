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
use crate::error::Result;
use crate::foundation::alloc::try_alloc_zeroed_f32;

// ============================================================================
// Memory Allocation Helpers
// ============================================================================

/// Allocate a Vec<f32> for writing with fallible allocation.
///
/// Uses zeroed allocation which may be faster than uninitialized because:
/// - OS can provide pre-zeroed pages from zero-page pool
/// - Avoids lazy page fault overhead on first write
/// - Pages are mapped immediately rather than on first touch
#[inline(always)]
fn try_alloc_f32(len: usize, context: &'static str) -> Result<Vec<f32>> {
    try_alloc_zeroed_f32(len, context)
}

// ============================================================================
// Type Conversion Helpers
// ============================================================================

/// SIMD-optimized conversion from u8 slice to f32 Vec.
///
/// Processes 8 elements at a time, converting each u8 to f32.
/// Used for converting YUV planes from external crates to internal f32 representation.
///
/// # Arguments
/// * `input` - Input slice of u8 values
///
/// # Returns
/// Vec of f32 values, or error on allocation failure
#[inline]
pub fn u8_slice_to_f32_simd(input: &[u8]) -> Result<Vec<f32>> {
    let len = input.len();
    let mut result = try_alloc_f32(len, "u8_slice_to_f32_simd")?;

    let chunks = len / 8;

    // Process 8 elements at a time
    for i in 0..chunks {
        let k = i * 8;
        // Load 8 u8 values and convert to f32
        let vals = f32x8::from([
            input[k] as f32,
            input[k + 1] as f32,
            input[k + 2] as f32,
            input[k + 3] as f32,
            input[k + 4] as f32,
            input[k + 5] as f32,
            input[k + 6] as f32,
            input[k + 7] as f32,
        ]);
        let arr: [f32; 8] = vals.into();
        result[k..k + 8].copy_from_slice(&arr);
    }

    // Handle remaining elements (scalar)
    let remainder_start = chunks * 8;
    for i in remainder_start..len {
        result[i] = input[i] as f32;
    }

    Ok(result)
}

/// SIMD-optimized conversion from u8 iterator to f32 Vec.
///
/// Same as u8_slice_to_f32_simd but works with an iterator that has a known length.
#[inline]
pub fn u8_iter_to_f32_simd(input: impl Iterator<Item = u8>, len: usize) -> Result<Vec<f32>> {
    let mut result = try_alloc_f32(len, "u8_iter_to_f32_simd")?;
    let mut i = 0;

    // Collect into buffer for SIMD processing
    let mut buf = [0u8; 8];
    let mut buf_idx = 0;

    for val in input {
        buf[buf_idx] = val;
        buf_idx += 1;

        if buf_idx == 8 {
            let vals = f32x8::from([
                buf[0] as f32,
                buf[1] as f32,
                buf[2] as f32,
                buf[3] as f32,
                buf[4] as f32,
                buf[5] as f32,
                buf[6] as f32,
                buf[7] as f32,
            ]);
            let arr: [f32; 8] = vals.into();
            result[i..i + 8].copy_from_slice(&arr);
            i += 8;
            buf_idx = 0;
        }
    }

    // Handle remaining elements
    for j in 0..buf_idx {
        result[i + j] = buf[j] as f32;
    }

    Ok(result)
}

/// SIMD-optimized scaling of f32 slice by a constant factor.
///
/// Multiplies all elements by the given scale factor.
/// Used for scaling Y plane from [0,1] to [0,255] for AQ computation.
///
/// # Arguments
/// * `input` - Input slice of f32 values
/// * `scale` - Scale factor to multiply by
///
/// # Returns
/// Vec of scaled f32 values, or error on allocation failure
#[inline]
pub fn scale_f32_slice_simd(input: &[f32], scale: f32) -> Result<Vec<f32>> {
    let len = input.len();
    let mut result = try_alloc_f32(len, "scale_f32_slice_simd")?;

    let chunks = len / 8;
    let scale_vec = f32x8::splat(scale);

    // Process 8 elements at a time
    for i in 0..chunks {
        let k = i * 8;
        let vals = f32x8::from([
            input[k],
            input[k + 1],
            input[k + 2],
            input[k + 3],
            input[k + 4],
            input[k + 5],
            input[k + 6],
            input[k + 7],
        ]);
        let scaled = vals * scale_vec;
        let arr: [f32; 8] = scaled.into();
        result[k..k + 8].copy_from_slice(&arr);
    }

    // Handle remaining elements (scalar)
    let remainder_start = chunks * 8;
    for i in remainder_start..len {
        result[i] = input[i] * scale;
    }

    Ok(result)
}

// ============================================================================
// Chroma Downsampling (2x2 box filter)
// ============================================================================

/// SIMD-optimized 2x2 box filter downsampling, writing to pre-allocated buffer.
///
/// This is the **zero-allocation** version for hot paths.
///
/// # Arguments
/// * `plane` - Input plane (f32, full resolution)
/// * `width` - Input width
/// * `height` - Input height
/// * `result` - Output buffer (must be at least `((width+1)/2) * ((height+1)/2)` elements)
#[inline]
pub fn downsample_2x2_simd_inplace(plane: &[f32], width: usize, height: usize, result: &mut [f32]) {
    let new_width = (width + 1) / 2;
    let new_height = (height + 1) / 2;
    debug_assert!(result.len() >= new_width * new_height);

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

            let row0_idx = y0 * width + in_x;
            let row1_idx = y1 * width + in_x;

            // Gather even/odd from row 0 and row 1
            let (p00, p10) = gather_even_odd_x8(plane, row0_idx, width);
            let (p01, p11) = gather_even_odd_x8(plane, row1_idx, width);

            // Box filter: (p00 + p10 + p01 + p11) * 0.25
            let sum = p00 + p10 + p01 + p11;
            let avg = sum * scale;

            // Store result using direct SIMD store
            unsafe {
                let out_ptr = result.as_mut_ptr().add(out_row_start + out_x);
                *(out_ptr as *mut [f32; 8]) = avg.into();
            }
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
}

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
/// Downsampled plane at half resolution, or error on allocation failure
pub fn downsample_2x2_simd(plane: &[f32], width: usize, height: usize) -> Result<Vec<f32>> {
    let new_width = (width + 1) / 2;
    let new_height = (height + 1) / 2;
    let mut result = try_alloc_f32(new_width * new_height, "downsample_2x2_simd")?;

    downsample_2x2_simd_inplace(plane, width, height, &mut result);

    Ok(result)
}

/// SIMD-optimized 2x1 (horizontal) box filter downsampling.
pub fn downsample_2x1_simd(plane: &[f32], width: usize, height: usize) -> Result<Vec<f32>> {
    let new_width = (width + 1) / 2;
    let mut result = try_alloc_f32(new_width * height, "downsample_2x1_simd")?;

    let scale = f32x8::splat(0.5);
    let chunks = new_width / 8;

    for y in 0..height {
        let out_row_start = y * new_width;
        let in_row_start = y * width;

        // SIMD path: process 8 output pixels at a time
        for chunk in 0..chunks {
            let out_x = chunk * 8;
            let in_x = out_x * 2;

            // Gather even/odd pixels from the row
            let (p0, p1) = gather_even_odd_x8(plane, in_row_start + in_x, width);

            // Box filter: (p0 + p1) * 0.5
            let avg = (p0 + p1) * scale;

            // Store result using direct SIMD store
            unsafe {
                let out_ptr = result.as_mut_ptr().add(out_row_start + out_x);
                *(out_ptr as *mut [f32; 8]) = avg.into();
            }
        }

        // Scalar remainder
        for out_x in (chunks * 8)..new_width {
            let x0 = out_x * 2;
            let x1 = (x0 + 1).min(width - 1);

            let p0 = plane[in_row_start + x0];
            let p1 = plane[in_row_start + x1];

            result[out_row_start + out_x] = (p0 + p1) * 0.5;
        }
    }

    Ok(result)
}

/// SIMD-optimized 2x1 (horizontal) box filter downsampling (in-place).
///
/// Writes to pre-allocated result buffer.
pub fn downsample_2x1_simd_inplace(plane: &[f32], width: usize, height: usize, result: &mut [f32]) {
    let new_width = (width + 1) / 2;
    debug_assert!(result.len() >= new_width * height);

    let scale = f32x8::splat(0.5);
    let chunks = new_width / 8;

    for y in 0..height {
        let out_row_start = y * new_width;
        let in_row_start = y * width;

        // SIMD path: process 8 output pixels at a time
        for chunk in 0..chunks {
            let out_x = chunk * 8;
            let in_x = out_x * 2;

            // Gather even/odd pixels from the row
            let (p0, p1) = gather_even_odd_x8(plane, in_row_start + in_x, width);

            // Box filter: (p0 + p1) * 0.5
            let avg = (p0 + p1) * scale;

            // Store result using direct SIMD store
            unsafe {
                let out_ptr = result.as_mut_ptr().add(out_row_start + out_x);
                *(out_ptr as *mut [f32; 8]) = avg.into();
            }
        }

        // Scalar remainder
        for out_x in (chunks * 8)..new_width {
            let x0 = out_x * 2;
            let x1 = (x0 + 1).min(width - 1);

            let p0 = plane[in_row_start + x0];
            let p1 = plane[in_row_start + x1];

            result[out_row_start + out_x] = (p0 + p1) * 0.5;
        }
    }
}

/// SIMD-optimized 1x2 (vertical) box filter downsampling.
pub fn downsample_1x2_simd(plane: &[f32], width: usize, height: usize) -> Result<Vec<f32>> {
    let new_height = (height + 1) / 2;
    let mut result = try_alloc_f32(width * new_height, "downsample_1x2_simd")?;

    let scale = f32x8::splat(0.5);
    let chunks = width / 8;

    for y in 0..new_height {
        let y0 = y * 2;
        let y1 = (y0 + 1).min(height - 1);
        let out_row_start = y * width;

        // SIMD path: process 8 pixels at a time
        for chunk in 0..chunks {
            let x = chunk * 8;

            // Load 8 consecutive pixels from row y0 and y1 using direct SIMD loads
            let row0_idx = y0 * width + x;
            let row1_idx = y1 * width + x;

            // SAFETY: chunks calculation ensures x + 8 <= width
            let (p0, p1) = unsafe {
                let ptr0 = plane.as_ptr().add(row0_idx);
                let ptr1 = plane.as_ptr().add(row1_idx);
                (
                    f32x8::from(*(ptr0 as *const [f32; 8])),
                    f32x8::from(*(ptr1 as *const [f32; 8])),
                )
            };

            let avg = (p0 + p1) * scale;

            // Store result using direct SIMD store
            unsafe {
                let out_ptr = result.as_mut_ptr().add(out_row_start + x);
                *(out_ptr as *mut [f32; 8]) = avg.into();
            }
        }

        // Scalar remainder
        for x in (chunks * 8)..width {
            let p0 = plane[y0 * width + x];
            let p1 = plane[y1 * width + x];
            result[out_row_start + x] = (p0 + p1) * 0.5;
        }
    }

    Ok(result)
}

/// SIMD-optimized 1x2 (vertical) box filter downsampling (in-place).
///
/// Writes to pre-allocated result buffer.
pub fn downsample_1x2_simd_inplace(plane: &[f32], width: usize, height: usize, result: &mut [f32]) {
    let new_height = (height + 1) / 2;
    debug_assert!(result.len() >= width * new_height);

    let scale = f32x8::splat(0.5);
    let chunks = width / 8;

    for y in 0..new_height {
        let y0 = y * 2;
        let y1 = (y0 + 1).min(height - 1);
        let out_row_start = y * width;

        // SIMD path: process 8 pixels at a time
        for chunk in 0..chunks {
            let x = chunk * 8;

            // Load 8 consecutive pixels from row y0 and y1 using direct SIMD loads
            let row0_idx = y0 * width + x;
            let row1_idx = y1 * width + x;

            // SAFETY: chunks calculation ensures x + 8 <= width
            let (p0, p1) = unsafe {
                let ptr0 = plane.as_ptr().add(row0_idx);
                let ptr1 = plane.as_ptr().add(row1_idx);
                (
                    f32x8::from(*(ptr0 as *const [f32; 8])),
                    f32x8::from(*(ptr1 as *const [f32; 8])),
                )
            };

            let avg = (p0 + p1) * scale;

            // Store result using direct SIMD store
            unsafe {
                let out_ptr = result.as_mut_ptr().add(out_row_start + x);
                *(out_ptr as *mut [f32; 8]) = avg.into();
            }
        }

        // Scalar remainder
        for x in (chunks * 8)..width {
            let p0 = plane[y0 * width + x];
            let p1 = plane[y1 * width + x];
            result[out_row_start + x] = (p0 + p1) * 0.5;
        }
    }
}

/// Gather even and odd indexed elements from a row into two f32x8 vectors.
///
/// Given input [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, ...]:
/// - evens = [a, c, e, g, i, k, m, o]
/// - odds = [b, d, f, h, j, l, n, p]
#[inline(always)]
fn gather_even_odd_x8(plane: &[f32], start_idx: usize, width: usize) -> (f32x8, f32x8) {
    // Fast path: when we have at least 16 elements available, use direct loads
    if start_idx + 16 <= plane.len() {
        // SAFETY: We just checked that start_idx + 16 <= plane.len()
        unsafe {
            let ptr = plane.as_ptr().add(start_idx);
            // Load first 8 floats [0-7]
            let a: [f32; 8] = *(ptr as *const [f32; 8]);
            // Load second 8 floats [8-15]
            let b: [f32; 8] = *(ptr.add(8) as *const [f32; 8]);

            let evens = f32x8::from([a[0], a[2], a[4], a[6], b[0], b[2], b[4], b[6]]);
            let odds = f32x8::from([a[1], a[3], a[5], a[7], b[1], b[3], b[5], b[7]]);

            return (evens, odds);
        }
    }

    // Slow path: boundary-safe gather with clamping
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

/// SIMD-optimized RGB to YCbCr conversion, writing to pre-allocated buffers.
///
/// This is the **zero-allocation** version for hot paths. Use this when encoding
/// multiple images or when performance is critical.
///
/// # Arguments
/// * `rgb_data` - Input RGB data (3 bytes per pixel, interleaved)
/// * `y_plane` - Output Y plane (must be at least `num_pixels` elements)
/// * `cb_plane` - Output Cb plane (must be at least `num_pixels` elements)
/// * `cr_plane` - Output Cr plane (must be at least `num_pixels` elements)
/// * `num_pixels` - Number of pixels to process
#[inline]
pub fn rgb_to_ycbcr_planes_simd_inplace(
    rgb_data: &[u8],
    y_plane: &mut [f32],
    cb_plane: &mut [f32],
    cr_plane: &mut [f32],
    num_pixels: usize,
) {
    debug_assert!(rgb_data.len() >= num_pixels * 3);
    debug_assert!(y_plane.len() >= num_pixels);
    debug_assert!(cb_plane.len() >= num_pixels);
    debug_assert!(cr_plane.len() >= num_pixels);

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

        // Store results using direct SIMD stores for better performance
        // SAFETY: We verified buffer lengths in debug_assert above
        unsafe {
            let y_ptr = y_plane.as_mut_ptr().add(pixel_idx);
            let cb_ptr = cb_plane.as_mut_ptr().add(pixel_idx);
            let cr_ptr = cr_plane.as_mut_ptr().add(pixel_idx);
            *(y_ptr as *mut [f32; 8]) = y.into();
            *(cb_ptr as *mut [f32; 8]) = cb.into();
            *(cr_ptr as *mut [f32; 8]) = cr.into();
        }
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
}

/// SIMD-optimized RGB to YCbCr conversion for entire image.
///
/// Processes 8 pixels at a time, converting from interleaved RGB to planar YCbCr.
///
/// # Arguments
/// * `rgb_data` - Input RGB data (3 bytes per pixel, interleaved)
/// * `num_pixels` - Number of pixels
///
/// # Returns
/// Tuple of (Y plane, Cb plane, Cr plane) as f32 vectors, or error on allocation failure
pub fn rgb_to_ycbcr_planes_simd(
    rgb_data: &[u8],
    num_pixels: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    // Use uninit allocation since we write to every element
    let mut y_plane = try_alloc_f32(num_pixels, "rgb_to_ycbcr Y plane")?;
    let mut cb_plane = try_alloc_f32(num_pixels, "rgb_to_ycbcr Cb plane")?;
    let mut cr_plane = try_alloc_f32(num_pixels, "rgb_to_ycbcr Cr plane")?;

    rgb_to_ycbcr_planes_simd_inplace(
        rgb_data,
        &mut y_plane,
        &mut cb_plane,
        &mut cr_plane,
        num_pixels,
    );

    Ok((y_plane, cb_plane, cr_plane))
}

/// SIMD-optimized RGBA to YCbCr conversion, writing to pre-allocated buffers.
#[inline]
pub fn rgba_to_ycbcr_planes_simd_inplace(
    rgba_data: &[u8],
    y_plane: &mut [f32],
    cb_plane: &mut [f32],
    cr_plane: &mut [f32],
    num_pixels: usize,
) {
    debug_assert!(rgba_data.len() >= num_pixels * 4);
    debug_assert!(y_plane.len() >= num_pixels);
    debug_assert!(cb_plane.len() >= num_pixels);
    debug_assert!(cr_plane.len() >= num_pixels);

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

        unsafe {
            let y_ptr = y_plane.as_mut_ptr().add(pixel_idx);
            let cb_ptr = cb_plane.as_mut_ptr().add(pixel_idx);
            let cr_ptr = cr_plane.as_mut_ptr().add(pixel_idx);
            *(y_ptr as *mut [f32; 8]) = y.into();
            *(cb_ptr as *mut [f32; 8]) = cb.into();
            *(cr_ptr as *mut [f32; 8]) = cr.into();
        }
    }

    for i in (chunks * 8)..num_pixels {
        let rgba_idx = i * 4;
        let r = rgba_data[rgba_idx] as f32;
        let g = rgba_data[rgba_idx + 1] as f32;
        let b = rgba_data[rgba_idx + 2] as f32;

        y_plane[i] = YCBCR_R_TO_Y * r + YCBCR_G_TO_Y * g + YCBCR_B_TO_Y * b;
        cb_plane[i] = 128.0 + YCBCR_R_TO_CB * r + YCBCR_G_TO_CB * g + YCBCR_B_TO_CB * b;
        cr_plane[i] = 128.0 + YCBCR_R_TO_CR * r + YCBCR_G_TO_CR * g + YCBCR_B_TO_CR * b;
    }
}

/// SIMD-optimized RGBA to YCbCr conversion for entire image (ignores alpha).
pub fn rgba_to_ycbcr_planes_simd(
    rgba_data: &[u8],
    num_pixels: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    // Use uninit allocation since we write to every element
    let mut y_plane = try_alloc_f32(num_pixels, "rgba_to_ycbcr Y plane")?;
    let mut cb_plane = try_alloc_f32(num_pixels, "rgba_to_ycbcr Cb plane")?;
    let mut cr_plane = try_alloc_f32(num_pixels, "rgba_to_ycbcr Cr plane")?;

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

    Ok((y_plane, cb_plane, cr_plane))
}

/// SIMD-optimized grayscale to YCbCr conversion, writing to pre-allocated buffers.
#[inline]
pub fn gray_to_ycbcr_planes_simd_inplace(
    gray_data: &[u8],
    y_plane: &mut [f32],
    cb_plane: &mut [f32],
    cr_plane: &mut [f32],
    num_pixels: usize,
) {
    debug_assert!(gray_data.len() >= num_pixels);
    debug_assert!(y_plane.len() >= num_pixels);
    debug_assert!(cb_plane.len() >= num_pixels);
    debug_assert!(cr_plane.len() >= num_pixels);

    let offset_128 = f32x8::splat(128.0);
    let chunks = num_pixels / 8;

    for chunk in 0..chunks {
        let idx = chunk * 8;

        let y = f32x8::from([
            gray_data[idx] as f32,
            gray_data[idx + 1] as f32,
            gray_data[idx + 2] as f32,
            gray_data[idx + 3] as f32,
            gray_data[idx + 4] as f32,
            gray_data[idx + 5] as f32,
            gray_data[idx + 6] as f32,
            gray_data[idx + 7] as f32,
        ]);

        unsafe {
            let y_ptr = y_plane.as_mut_ptr().add(idx);
            let cb_ptr = cb_plane.as_mut_ptr().add(idx);
            let cr_ptr = cr_plane.as_mut_ptr().add(idx);
            *(y_ptr as *mut [f32; 8]) = y.into();
            *(cb_ptr as *mut [f32; 8]) = offset_128.into();
            *(cr_ptr as *mut [f32; 8]) = offset_128.into();
        }
    }

    for i in (chunks * 8)..num_pixels {
        y_plane[i] = gray_data[i] as f32;
        cb_plane[i] = 128.0;
        cr_plane[i] = 128.0;
    }
}

/// SIMD-optimized grayscale to YCbCr conversion.
/// Y = gray value, Cb = Cr = 128.0
pub fn gray_to_ycbcr_planes_simd(
    gray_data: &[u8],
    num_pixels: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let mut y_plane = try_alloc_f32(num_pixels, "gray_to_ycbcr Y plane")?;
    let mut cb_plane = try_alloc_f32(num_pixels, "gray_to_ycbcr Cb plane")?;
    let mut cr_plane = try_alloc_f32(num_pixels, "gray_to_ycbcr Cr plane")?;

    gray_to_ycbcr_planes_simd_inplace(
        gray_data,
        &mut y_plane,
        &mut cb_plane,
        &mut cr_plane,
        num_pixels,
    );

    Ok((y_plane, cb_plane, cr_plane))
}

/// SIMD-optimized BGR to YCbCr conversion, writing to pre-allocated buffers.
#[inline]
pub fn bgr_to_ycbcr_planes_simd_inplace(
    bgr_data: &[u8],
    y_plane: &mut [f32],
    cb_plane: &mut [f32],
    cr_plane: &mut [f32],
    num_pixels: usize,
) {
    debug_assert!(bgr_data.len() >= num_pixels * 3);
    debug_assert!(y_plane.len() >= num_pixels);
    debug_assert!(cb_plane.len() >= num_pixels);
    debug_assert!(cr_plane.len() >= num_pixels);

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
        let bgr_idx = pixel_idx * 3;

        let b = f32x8::from([
            bgr_data[bgr_idx] as f32,
            bgr_data[bgr_idx + 3] as f32,
            bgr_data[bgr_idx + 6] as f32,
            bgr_data[bgr_idx + 9] as f32,
            bgr_data[bgr_idx + 12] as f32,
            bgr_data[bgr_idx + 15] as f32,
            bgr_data[bgr_idx + 18] as f32,
            bgr_data[bgr_idx + 21] as f32,
        ]);

        let g = f32x8::from([
            bgr_data[bgr_idx + 1] as f32,
            bgr_data[bgr_idx + 4] as f32,
            bgr_data[bgr_idx + 7] as f32,
            bgr_data[bgr_idx + 10] as f32,
            bgr_data[bgr_idx + 13] as f32,
            bgr_data[bgr_idx + 16] as f32,
            bgr_data[bgr_idx + 19] as f32,
            bgr_data[bgr_idx + 22] as f32,
        ]);

        let r = f32x8::from([
            bgr_data[bgr_idx + 2] as f32,
            bgr_data[bgr_idx + 5] as f32,
            bgr_data[bgr_idx + 8] as f32,
            bgr_data[bgr_idx + 11] as f32,
            bgr_data[bgr_idx + 14] as f32,
            bgr_data[bgr_idx + 17] as f32,
            bgr_data[bgr_idx + 20] as f32,
            bgr_data[bgr_idx + 23] as f32,
        ]);

        let y = r * r_to_y + g * g_to_y + b * b_to_y;
        let cb = offset_128 + r * r_to_cb + g * g_to_cb + b * b_to_cb;
        let cr = offset_128 + r * r_to_cr + g * g_to_cr + b * b_to_cr;

        unsafe {
            let y_ptr = y_plane.as_mut_ptr().add(pixel_idx);
            let cb_ptr = cb_plane.as_mut_ptr().add(pixel_idx);
            let cr_ptr = cr_plane.as_mut_ptr().add(pixel_idx);
            *(y_ptr as *mut [f32; 8]) = y.into();
            *(cb_ptr as *mut [f32; 8]) = cb.into();
            *(cr_ptr as *mut [f32; 8]) = cr.into();
        }
    }

    for i in (chunks * 8)..num_pixels {
        let bgr_idx = i * 3;
        let b = bgr_data[bgr_idx] as f32;
        let g = bgr_data[bgr_idx + 1] as f32;
        let r = bgr_data[bgr_idx + 2] as f32;

        y_plane[i] = YCBCR_R_TO_Y * r + YCBCR_G_TO_Y * g + YCBCR_B_TO_Y * b;
        cb_plane[i] = 128.0 + YCBCR_R_TO_CB * r + YCBCR_G_TO_CB * g + YCBCR_B_TO_CB * b;
        cr_plane[i] = 128.0 + YCBCR_R_TO_CR * r + YCBCR_G_TO_CR * g + YCBCR_B_TO_CR * b;
    }
}

/// SIMD-optimized BGR to YCbCr conversion for entire image.
pub fn bgr_to_ycbcr_planes_simd(
    bgr_data: &[u8],
    num_pixels: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let mut y_plane = try_alloc_f32(num_pixels, "bgr_to_ycbcr Y plane")?;
    let mut cb_plane = try_alloc_f32(num_pixels, "bgr_to_ycbcr Cb plane")?;
    let mut cr_plane = try_alloc_f32(num_pixels, "bgr_to_ycbcr Cr plane")?;

    bgr_to_ycbcr_planes_simd_inplace(
        bgr_data,
        &mut y_plane,
        &mut cb_plane,
        &mut cr_plane,
        num_pixels,
    );

    Ok((y_plane, cb_plane, cr_plane))
}

/// SIMD-optimized BGRA to YCbCr conversion, writing to pre-allocated buffers.
#[inline]
pub fn bgra_to_ycbcr_planes_simd_inplace(
    bgra_data: &[u8],
    y_plane: &mut [f32],
    cb_plane: &mut [f32],
    cr_plane: &mut [f32],
    num_pixels: usize,
) {
    debug_assert!(bgra_data.len() >= num_pixels * 4);
    debug_assert!(y_plane.len() >= num_pixels);
    debug_assert!(cb_plane.len() >= num_pixels);
    debug_assert!(cr_plane.len() >= num_pixels);

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
        let bgra_idx = pixel_idx * 4;

        let b = f32x8::from([
            bgra_data[bgra_idx] as f32,
            bgra_data[bgra_idx + 4] as f32,
            bgra_data[bgra_idx + 8] as f32,
            bgra_data[bgra_idx + 12] as f32,
            bgra_data[bgra_idx + 16] as f32,
            bgra_data[bgra_idx + 20] as f32,
            bgra_data[bgra_idx + 24] as f32,
            bgra_data[bgra_idx + 28] as f32,
        ]);

        let g = f32x8::from([
            bgra_data[bgra_idx + 1] as f32,
            bgra_data[bgra_idx + 5] as f32,
            bgra_data[bgra_idx + 9] as f32,
            bgra_data[bgra_idx + 13] as f32,
            bgra_data[bgra_idx + 17] as f32,
            bgra_data[bgra_idx + 21] as f32,
            bgra_data[bgra_idx + 25] as f32,
            bgra_data[bgra_idx + 29] as f32,
        ]);

        let r = f32x8::from([
            bgra_data[bgra_idx + 2] as f32,
            bgra_data[bgra_idx + 6] as f32,
            bgra_data[bgra_idx + 10] as f32,
            bgra_data[bgra_idx + 14] as f32,
            bgra_data[bgra_idx + 18] as f32,
            bgra_data[bgra_idx + 22] as f32,
            bgra_data[bgra_idx + 26] as f32,
            bgra_data[bgra_idx + 30] as f32,
        ]);

        let y = r * r_to_y + g * g_to_y + b * b_to_y;
        let cb = offset_128 + r * r_to_cb + g * g_to_cb + b * b_to_cb;
        let cr = offset_128 + r * r_to_cr + g * g_to_cr + b * b_to_cr;

        unsafe {
            let y_ptr = y_plane.as_mut_ptr().add(pixel_idx);
            let cb_ptr = cb_plane.as_mut_ptr().add(pixel_idx);
            let cr_ptr = cr_plane.as_mut_ptr().add(pixel_idx);
            *(y_ptr as *mut [f32; 8]) = y.into();
            *(cb_ptr as *mut [f32; 8]) = cb.into();
            *(cr_ptr as *mut [f32; 8]) = cr.into();
        }
    }

    for i in (chunks * 8)..num_pixels {
        let bgra_idx = i * 4;
        let b = bgra_data[bgra_idx] as f32;
        let g = bgra_data[bgra_idx + 1] as f32;
        let r = bgra_data[bgra_idx + 2] as f32;

        y_plane[i] = YCBCR_R_TO_Y * r + YCBCR_G_TO_Y * g + YCBCR_B_TO_Y * b;
        cb_plane[i] = 128.0 + YCBCR_R_TO_CB * r + YCBCR_G_TO_CB * g + YCBCR_B_TO_CB * b;
        cr_plane[i] = 128.0 + YCBCR_R_TO_CR * r + YCBCR_G_TO_CR * g + YCBCR_B_TO_CR * b;
    }
}

/// SIMD-optimized BGRA to YCbCr conversion for entire image (ignores alpha).
pub fn bgra_to_ycbcr_planes_simd(
    bgra_data: &[u8],
    num_pixels: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let mut y_plane = try_alloc_f32(num_pixels, "bgra_to_ycbcr Y plane")?;
    let mut cb_plane = try_alloc_f32(num_pixels, "bgra_to_ycbcr Cb plane")?;
    let mut cr_plane = try_alloc_f32(num_pixels, "bgra_to_ycbcr Cr plane")?;

    bgra_to_ycbcr_planes_simd_inplace(
        bgra_data,
        &mut y_plane,
        &mut cb_plane,
        &mut cr_plane,
        num_pixels,
    );

    Ok((y_plane, cb_plane, cr_plane))
}

// ============================================================================
// Block Extraction
// ============================================================================

/// SIMD-optimized block extraction from a plane with level shift.
///
/// Extracts an 8x8 block from the given position, subtracting 128.0 (level shift).
/// Uses fast path for interior blocks (no bounds checking needed).
///
/// # Arguments
/// * `plane` - Input plane data
/// * `width` - Plane width
/// * `height` - Plane height
/// * `bx` - Block x coordinate (in blocks, not pixels)
/// * `by` - Block y coordinate (in blocks, not pixels)
#[inline]
pub fn extract_block_simd(
    plane: &[f32],
    width: usize,
    height: usize,
    bx: usize,
    by: usize,
) -> [f32; 64] {
    let px_start = bx * 8;
    let py_start = by * 8;

    // Check if block is entirely within bounds
    let is_interior = px_start + 8 <= width && py_start + 8 <= height;

    let level_shift = f32x8::splat(128.0);
    let mut block = [0.0f32; 64];

    if is_interior {
        // Fast path: no bounds checking needed
        for y in 0..8 {
            let row_start = (py_start + y) * width + px_start;

            // Load 8 consecutive f32 values
            let row_slice = &plane[row_start..row_start + 8];
            let row = f32x8::from([
                row_slice[0],
                row_slice[1],
                row_slice[2],
                row_slice[3],
                row_slice[4],
                row_slice[5],
                row_slice[6],
                row_slice[7],
            ]);

            // Subtract level shift
            let shifted = row - level_shift;

            // Store to block
            let arr: [f32; 8] = shifted.into();
            block[y * 8..y * 8 + 8].copy_from_slice(&arr);
        }
    } else {
        // Edge case: need bounds checking
        for y in 0..8 {
            let py = (py_start + y).min(height - 1);
            for x in 0..8 {
                let px = (px_start + x).min(width - 1);
                block[y * 8 + x] = plane[py * width + px] - 128.0;
            }
        }
    }

    block
}

/// SIMD-optimized block extraction for XYB planes with scaling.
///
/// XYB values are in range ~[-2.1, 7.3], need to scale by 255 and level shift by -128.
///
/// # Arguments
/// * `plane` - Input XYB plane data
/// * `width` - Plane width
/// * `height` - Plane height
/// * `bx` - Block x coordinate (in blocks)
/// * `by` - Block y coordinate (in blocks)
#[inline]
pub fn extract_block_xyb_simd(
    plane: &[f32],
    width: usize,
    height: usize,
    bx: usize,
    by: usize,
) -> [f32; 64] {
    let px_start = bx * 8;
    let py_start = by * 8;

    let is_interior = px_start + 8 <= width && py_start + 8 <= height;

    let scale = f32x8::splat(255.0);
    let level_shift = f32x8::splat(128.0);
    let mut block = [0.0f32; 64];

    if is_interior {
        for y in 0..8 {
            let row_start = (py_start + y) * width + px_start;
            let row_slice = &plane[row_start..row_start + 8];
            let row = f32x8::from([
                row_slice[0],
                row_slice[1],
                row_slice[2],
                row_slice[3],
                row_slice[4],
                row_slice[5],
                row_slice[6],
                row_slice[7],
            ]);

            // XYB: val * 255.0 - 128.0
            let scaled = row * scale - level_shift;

            let arr: [f32; 8] = scaled.into();
            block[y * 8..y * 8 + 8].copy_from_slice(&arr);
        }
    } else {
        for y in 0..8 {
            let py = (py_start + y).min(height - 1);
            for x in 0..8 {
                let px = (px_start + x).min(width - 1);
                block[y * 8 + x] = plane[py * width + px] * 255.0 - 128.0;
            }
        }
    }

    block
}

// ============================================================================
// Input Smoothing (3x3 convolution)
// ============================================================================

/// SIMD-optimized 3x3 smoothing filter for chroma planes.
///
/// Applies weighted average: center * kw0 + neighbors * kw1
/// where kw1 = factor/1024 and kw0 = 1 - 8*kw1.
///
/// # Arguments
/// * `plane` - Input plane data
/// * `width` - Plane width
/// * `height` - Plane height
/// * `factor` - Smoothing factor (0-127, 0 = no smoothing)
#[inline]
pub fn apply_smoothing_simd(
    plane: &[f32],
    width: usize,
    height: usize,
    factor: u8,
) -> Result<Vec<f32>> {
    if factor == 0 {
        return Ok(plane.to_vec());
    }

    let mut result = try_alloc_f32(width * height, "apply_smoothing_simd")?;

    let kw1 = factor as f32 / 1024.0;
    let kw0 = 1.0 - 8.0 * kw1;

    let kw0_simd = f32x8::splat(kw0);
    let kw1_simd = f32x8::splat(kw1);

    // Process interior rows (can use SIMD for horizontal)
    for y in 0..height {
        let y_t = y.saturating_sub(1);
        let y_b = (y + 1).min(height - 1);

        // Process 8 pixels at a time for interior
        let chunks = if width >= 10 { (width - 2) / 8 } else { 0 };

        // Handle left edge (x=0)
        {
            let x = 0;
            let x_r = 1;

            let val_tl = plane[y_t * width + x];
            let val_tm = plane[y_t * width + x];
            let val_tr = plane[y_t * width + x_r];
            let val_ml = plane[y * width + x];
            let val_mm = plane[y * width + x];
            let val_mr = plane[y * width + x_r];
            let val_bl = plane[y_b * width + x];
            let val_bm = plane[y_b * width + x];
            let val_br = plane[y_b * width + x_r];

            let neighbors = val_tl + val_tm + val_tr + val_ml + val_mr + val_bl + val_bm + val_br;
            result[y * width + x] = val_mm * kw0 + neighbors * kw1;
        }

        // SIMD interior
        for chunk in 0..chunks {
            let x_start = 1 + chunk * 8;

            // Load center row
            let mm = f32x8::from([
                plane[y * width + x_start],
                plane[y * width + x_start + 1],
                plane[y * width + x_start + 2],
                plane[y * width + x_start + 3],
                plane[y * width + x_start + 4],
                plane[y * width + x_start + 5],
                plane[y * width + x_start + 6],
                plane[y * width + x_start + 7],
            ]);

            // Load left neighbors (offset -1)
            let ml = f32x8::from([
                plane[y * width + x_start - 1],
                plane[y * width + x_start],
                plane[y * width + x_start + 1],
                plane[y * width + x_start + 2],
                plane[y * width + x_start + 3],
                plane[y * width + x_start + 4],
                plane[y * width + x_start + 5],
                plane[y * width + x_start + 6],
            ]);

            // Load right neighbors (offset +1)
            let mr = f32x8::from([
                plane[y * width + x_start + 1],
                plane[y * width + x_start + 2],
                plane[y * width + x_start + 3],
                plane[y * width + x_start + 4],
                plane[y * width + x_start + 5],
                plane[y * width + x_start + 6],
                plane[y * width + x_start + 7],
                plane[y * width + x_start + 8],
            ]);

            // Load top row
            let tm = f32x8::from([
                plane[y_t * width + x_start],
                plane[y_t * width + x_start + 1],
                plane[y_t * width + x_start + 2],
                plane[y_t * width + x_start + 3],
                plane[y_t * width + x_start + 4],
                plane[y_t * width + x_start + 5],
                plane[y_t * width + x_start + 6],
                plane[y_t * width + x_start + 7],
            ]);
            let tl = f32x8::from([
                plane[y_t * width + x_start - 1],
                plane[y_t * width + x_start],
                plane[y_t * width + x_start + 1],
                plane[y_t * width + x_start + 2],
                plane[y_t * width + x_start + 3],
                plane[y_t * width + x_start + 4],
                plane[y_t * width + x_start + 5],
                plane[y_t * width + x_start + 6],
            ]);
            let tr = f32x8::from([
                plane[y_t * width + x_start + 1],
                plane[y_t * width + x_start + 2],
                plane[y_t * width + x_start + 3],
                plane[y_t * width + x_start + 4],
                plane[y_t * width + x_start + 5],
                plane[y_t * width + x_start + 6],
                plane[y_t * width + x_start + 7],
                plane[y_t * width + x_start + 8],
            ]);

            // Load bottom row
            let bm = f32x8::from([
                plane[y_b * width + x_start],
                plane[y_b * width + x_start + 1],
                plane[y_b * width + x_start + 2],
                plane[y_b * width + x_start + 3],
                plane[y_b * width + x_start + 4],
                plane[y_b * width + x_start + 5],
                plane[y_b * width + x_start + 6],
                plane[y_b * width + x_start + 7],
            ]);
            let bl = f32x8::from([
                plane[y_b * width + x_start - 1],
                plane[y_b * width + x_start],
                plane[y_b * width + x_start + 1],
                plane[y_b * width + x_start + 2],
                plane[y_b * width + x_start + 3],
                plane[y_b * width + x_start + 4],
                plane[y_b * width + x_start + 5],
                plane[y_b * width + x_start + 6],
            ]);
            let br = f32x8::from([
                plane[y_b * width + x_start + 1],
                plane[y_b * width + x_start + 2],
                plane[y_b * width + x_start + 3],
                plane[y_b * width + x_start + 4],
                plane[y_b * width + x_start + 5],
                plane[y_b * width + x_start + 6],
                plane[y_b * width + x_start + 7],
                plane[y_b * width + x_start + 8],
            ]);

            // Sum neighbors
            let neighbors = tl + tm + tr + ml + mr + bl + bm + br;

            // Apply weights
            let out = mm * kw0_simd + neighbors * kw1_simd;

            // Store
            let arr: [f32; 8] = out.into();
            result[y * width + x_start..y * width + x_start + 8].copy_from_slice(&arr);
        }

        // Handle remainder (right edge and any pixels after last full chunk)
        let start = if chunks > 0 { 1 + chunks * 8 } else { 1 };
        for x in start..width {
            let x_l = x.saturating_sub(1);
            let x_r = (x + 1).min(width - 1);

            let val_tl = plane[y_t * width + x_l];
            let val_tm = plane[y_t * width + x];
            let val_tr = plane[y_t * width + x_r];
            let val_ml = plane[y * width + x_l];
            let val_mm = plane[y * width + x];
            let val_mr = plane[y * width + x_r];
            let val_bl = plane[y_b * width + x_l];
            let val_bm = plane[y_b * width + x];
            let val_br = plane[y_b * width + x_r];

            let neighbors = val_tl + val_tm + val_tr + val_ml + val_mr + val_bl + val_bm + val_br;
            result[y * width + x] = val_mm * kw0 + neighbors * kw1;
        }
    }

    Ok(result)
}

/// SIMD-optimized input smoothing (in-place to separate buffer).
///
/// Writes to pre-allocated result buffer. Input and output must NOT overlap.
pub fn apply_smoothing_simd_inplace(
    plane: &[f32],
    width: usize,
    height: usize,
    factor: u8,
    result: &mut [f32],
) {
    if factor == 0 {
        result[..plane.len()].copy_from_slice(plane);
        return;
    }

    debug_assert!(result.len() >= width * height);

    let kw1 = factor as f32 / 1024.0;
    let kw0 = 1.0 - 8.0 * kw1;

    let kw0_simd = f32x8::splat(kw0);
    let kw1_simd = f32x8::splat(kw1);

    // Process interior rows (can use SIMD for horizontal)
    for y in 0..height {
        let y_t = y.saturating_sub(1);
        let y_b = (y + 1).min(height - 1);

        // Process 8 pixels at a time for interior
        let chunks = if width >= 10 { (width - 2) / 8 } else { 0 };

        // Handle left edge (x=0)
        {
            let x = 0;
            let x_r = 1;

            let val_tl = plane[y_t * width + x];
            let val_tm = plane[y_t * width + x];
            let val_tr = plane[y_t * width + x_r];
            let val_ml = plane[y * width + x];
            let val_mm = plane[y * width + x];
            let val_mr = plane[y * width + x_r];
            let val_bl = plane[y_b * width + x];
            let val_bm = plane[y_b * width + x];
            let val_br = plane[y_b * width + x_r];

            let neighbors = val_tl + val_tm + val_tr + val_ml + val_mr + val_bl + val_bm + val_br;
            result[y * width + x] = val_mm * kw0 + neighbors * kw1;
        }

        // SIMD interior
        for chunk in 0..chunks {
            let x_start = 1 + chunk * 8;

            // Load center row
            let mm = f32x8::from([
                plane[y * width + x_start],
                plane[y * width + x_start + 1],
                plane[y * width + x_start + 2],
                plane[y * width + x_start + 3],
                plane[y * width + x_start + 4],
                plane[y * width + x_start + 5],
                plane[y * width + x_start + 6],
                plane[y * width + x_start + 7],
            ]);

            // Load left neighbors (offset -1)
            let ml = f32x8::from([
                plane[y * width + x_start - 1],
                plane[y * width + x_start],
                plane[y * width + x_start + 1],
                plane[y * width + x_start + 2],
                plane[y * width + x_start + 3],
                plane[y * width + x_start + 4],
                plane[y * width + x_start + 5],
                plane[y * width + x_start + 6],
            ]);

            // Load right neighbors (offset +1)
            let mr = f32x8::from([
                plane[y * width + x_start + 1],
                plane[y * width + x_start + 2],
                plane[y * width + x_start + 3],
                plane[y * width + x_start + 4],
                plane[y * width + x_start + 5],
                plane[y * width + x_start + 6],
                plane[y * width + x_start + 7],
                plane[y * width + x_start + 8],
            ]);

            // Load top row
            let tm = f32x8::from([
                plane[y_t * width + x_start],
                plane[y_t * width + x_start + 1],
                plane[y_t * width + x_start + 2],
                plane[y_t * width + x_start + 3],
                plane[y_t * width + x_start + 4],
                plane[y_t * width + x_start + 5],
                plane[y_t * width + x_start + 6],
                plane[y_t * width + x_start + 7],
            ]);
            let tl = f32x8::from([
                plane[y_t * width + x_start - 1],
                plane[y_t * width + x_start],
                plane[y_t * width + x_start + 1],
                plane[y_t * width + x_start + 2],
                plane[y_t * width + x_start + 3],
                plane[y_t * width + x_start + 4],
                plane[y_t * width + x_start + 5],
                plane[y_t * width + x_start + 6],
            ]);
            let tr = f32x8::from([
                plane[y_t * width + x_start + 1],
                plane[y_t * width + x_start + 2],
                plane[y_t * width + x_start + 3],
                plane[y_t * width + x_start + 4],
                plane[y_t * width + x_start + 5],
                plane[y_t * width + x_start + 6],
                plane[y_t * width + x_start + 7],
                plane[y_t * width + x_start + 8],
            ]);

            // Load bottom row
            let bm = f32x8::from([
                plane[y_b * width + x_start],
                plane[y_b * width + x_start + 1],
                plane[y_b * width + x_start + 2],
                plane[y_b * width + x_start + 3],
                plane[y_b * width + x_start + 4],
                plane[y_b * width + x_start + 5],
                plane[y_b * width + x_start + 6],
                plane[y_b * width + x_start + 7],
            ]);
            let bl = f32x8::from([
                plane[y_b * width + x_start - 1],
                plane[y_b * width + x_start],
                plane[y_b * width + x_start + 1],
                plane[y_b * width + x_start + 2],
                plane[y_b * width + x_start + 3],
                plane[y_b * width + x_start + 4],
                plane[y_b * width + x_start + 5],
                plane[y_b * width + x_start + 6],
            ]);
            let br = f32x8::from([
                plane[y_b * width + x_start + 1],
                plane[y_b * width + x_start + 2],
                plane[y_b * width + x_start + 3],
                plane[y_b * width + x_start + 4],
                plane[y_b * width + x_start + 5],
                plane[y_b * width + x_start + 6],
                plane[y_b * width + x_start + 7],
                plane[y_b * width + x_start + 8],
            ]);

            // Sum neighbors
            let neighbors = tl + tm + tr + ml + mr + bl + bm + br;

            // Apply weights
            let out = mm * kw0_simd + neighbors * kw1_simd;

            // Store
            let arr: [f32; 8] = out.into();
            result[y * width + x_start..y * width + x_start + 8].copy_from_slice(&arr);
        }

        // Handle remainder (right edge and any pixels after last full chunk)
        let start = if chunks > 0 { 1 + chunks * 8 } else { 1 };
        for x in start..width {
            let x_l = x.saturating_sub(1);
            let x_r = (x + 1).min(width - 1);

            let val_tl = plane[y_t * width + x_l];
            let val_tm = plane[y_t * width + x];
            let val_tr = plane[y_t * width + x_r];
            let val_ml = plane[y * width + x_l];
            let val_mm = plane[y * width + x];
            let val_mr = plane[y * width + x_r];
            let val_bl = plane[y_b * width + x_l];
            let val_bm = plane[y_b * width + x];
            let val_br = plane[y_b * width + x_r];

            let neighbors = val_tl + val_tm + val_tr + val_ml + val_mr + val_bl + val_bm + val_br;
            result[y * width + x] = val_mm * kw0 + neighbors * kw1;
        }
    }
}

// ============================================================================
// XYB/Level Shift Conversion Helpers (for decoder)
// ============================================================================

/// SIMD-optimized XYB plane level shift to interleaved RGB u8.
///
/// Converts 3 XYB f32 planes to interleaved RGB u8, applying:
/// - Level shift (+128)
/// - Clamp to [0, 255]
/// - Convert to u8
///
/// This is used for XYB decode path where no YCbCr→RGB conversion is needed.
#[inline]
pub fn xyb_planes_to_rgb_u8_simd(plane0: &[f32], plane1: &[f32], plane2: &[f32], rgb: &mut [u8]) {
    debug_assert_eq!(plane0.len(), plane1.len());
    debug_assert_eq!(plane0.len(), plane2.len());
    debug_assert_eq!(rgb.len(), plane0.len() * 3);

    let num_pixels = plane0.len();
    let offset = f32x8::splat(128.0);
    let zero = f32x8::splat(0.0);
    let max_val = f32x8::splat(255.0);

    let chunks = num_pixels / 8;
    for chunk in 0..chunks {
        let base = chunk * 8;

        // Load 8 values from each plane
        let p0 = f32x8::from([
            plane0[base],
            plane0[base + 1],
            plane0[base + 2],
            plane0[base + 3],
            plane0[base + 4],
            plane0[base + 5],
            plane0[base + 6],
            plane0[base + 7],
        ]);
        let p1 = f32x8::from([
            plane1[base],
            plane1[base + 1],
            plane1[base + 2],
            plane1[base + 3],
            plane1[base + 4],
            plane1[base + 5],
            plane1[base + 6],
            plane1[base + 7],
        ]);
        let p2 = f32x8::from([
            plane2[base],
            plane2[base + 1],
            plane2[base + 2],
            plane2[base + 3],
            plane2[base + 4],
            plane2[base + 5],
            plane2[base + 6],
            plane2[base + 7],
        ]);

        // Level shift, clamp
        let r = (p0 + offset).max(zero).min(max_val);
        let g = (p1 + offset).max(zero).min(max_val);
        let b = (p2 + offset).max(zero).min(max_val);

        let r_arr: [f32; 8] = r.into();
        let g_arr: [f32; 8] = g.into();
        let b_arr: [f32; 8] = b.into();

        // Store interleaved RGB
        for j in 0..8 {
            let idx = (base + j) * 3;
            rgb[idx] = r_arr[j] as u8;
            rgb[idx + 1] = g_arr[j] as u8;
            rgb[idx + 2] = b_arr[j] as u8;
        }
    }

    // Scalar remainder
    for i in (chunks * 8)..num_pixels {
        let idx = i * 3;
        rgb[idx] = (plane0[i] + 128.0).clamp(0.0, 255.0) as u8;
        rgb[idx + 1] = (plane1[i] + 128.0).clamp(0.0, 255.0) as u8;
        rgb[idx + 2] = (plane2[i] + 128.0).clamp(0.0, 255.0) as u8;
    }
}

/// SIMD-optimized XYB plane level shift to interleaved RGB f32 (normalized 0-1).
#[inline]
pub fn xyb_planes_to_rgb_f32_simd(plane0: &[f32], plane1: &[f32], plane2: &[f32], rgb: &mut [f32]) {
    debug_assert_eq!(plane0.len(), plane1.len());
    debug_assert_eq!(plane0.len(), plane2.len());
    debug_assert_eq!(rgb.len(), plane0.len() * 3);

    let num_pixels = plane0.len();
    let offset = f32x8::splat(128.0);
    let scale = f32x8::splat(1.0 / 255.0);
    let zero = f32x8::splat(0.0);
    let one = f32x8::splat(1.0);

    let chunks = num_pixels / 8;
    for chunk in 0..chunks {
        let base = chunk * 8;

        let p0 = f32x8::from([
            plane0[base],
            plane0[base + 1],
            plane0[base + 2],
            plane0[base + 3],
            plane0[base + 4],
            plane0[base + 5],
            plane0[base + 6],
            plane0[base + 7],
        ]);
        let p1 = f32x8::from([
            plane1[base],
            plane1[base + 1],
            plane1[base + 2],
            plane1[base + 3],
            plane1[base + 4],
            plane1[base + 5],
            plane1[base + 6],
            plane1[base + 7],
        ]);
        let p2 = f32x8::from([
            plane2[base],
            plane2[base + 1],
            plane2[base + 2],
            plane2[base + 3],
            plane2[base + 4],
            plane2[base + 5],
            plane2[base + 6],
            plane2[base + 7],
        ]);

        // Level shift, scale to 0-1, clamp
        let r = ((p0 + offset) * scale).max(zero).min(one);
        let g = ((p1 + offset) * scale).max(zero).min(one);
        let b = ((p2 + offset) * scale).max(zero).min(one);

        let r_arr: [f32; 8] = r.into();
        let g_arr: [f32; 8] = g.into();
        let b_arr: [f32; 8] = b.into();

        for j in 0..8 {
            let idx = (base + j) * 3;
            rgb[idx] = r_arr[j];
            rgb[idx + 1] = g_arr[j];
            rgb[idx + 2] = b_arr[j];
        }
    }

    // Scalar remainder
    for i in (chunks * 8)..num_pixels {
        let idx = i * 3;
        rgb[idx] = ((plane0[i] + 128.0) / 255.0).clamp(0.0, 1.0);
        rgb[idx + 1] = ((plane1[i] + 128.0) / 255.0).clamp(0.0, 1.0);
        rgb[idx + 2] = ((plane2[i] + 128.0) / 255.0).clamp(0.0, 1.0);
    }
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
        let simd_result = downsample_2x2_simd(&plane, width, height).unwrap();

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
                i,
                simd_result[i],
                scalar_result[i],
                diff
            );
        }
    }

    #[test]
    fn test_downsample_2x1_simd_matches_scalar() {
        let width = 16;
        let height = 8;
        let plane: Vec<f32> = (0..width * height)
            .map(|i| {
                let x = i % width;
                let y = i / width;
                (x + y * 10) as f32
            })
            .collect();

        let simd_result = downsample_2x1_simd(&plane, width, height).unwrap();

        let new_width = (width + 1) / 2;
        let mut scalar_result = vec![0.0f32; new_width * height];
        for y in 0..height {
            for x in 0..new_width {
                let x0 = x * 2;
                let x1 = (x0 + 1).min(width - 1);
                let p0 = plane[y * width + x0];
                let p1 = plane[y * width + x1];
                scalar_result[y * new_width + x] = (p0 + p1) * 0.5;
            }
        }

        assert_eq!(simd_result.len(), scalar_result.len());
        for i in 0..simd_result.len() {
            let diff = (simd_result[i] - scalar_result[i]).abs();
            assert!(
                diff < EPSILON,
                "Downsample 2x1 mismatch at {}: SIMD={}, scalar={}, diff={}",
                i,
                simd_result[i],
                scalar_result[i],
                diff
            );
        }
    }

    #[test]
    fn test_downsample_1x2_simd_matches_scalar() {
        let width = 8;
        let height = 16;
        let plane: Vec<f32> = (0..width * height)
            .map(|i| {
                let x = i % width;
                let y = i / width;
                (x + y * 10) as f32
            })
            .collect();

        let simd_result = downsample_1x2_simd(&plane, width, height).unwrap();

        let new_height = (height + 1) / 2;
        let mut scalar_result = vec![0.0f32; width * new_height];
        for y in 0..new_height {
            for x in 0..width {
                let y0 = y * 2;
                let y1 = (y0 + 1).min(height - 1);
                let p0 = plane[y0 * width + x];
                let p1 = plane[y1 * width + x];
                scalar_result[y * width + x] = (p0 + p1) * 0.5;
            }
        }

        assert_eq!(simd_result.len(), scalar_result.len());
        for i in 0..simd_result.len() {
            let diff = (simd_result[i] - scalar_result[i]).abs();
            assert!(
                diff < EPSILON,
                "Downsample 1x2 mismatch at {}: SIMD={}, scalar={}, diff={}",
                i,
                simd_result[i],
                scalar_result[i],
                diff
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
        let (y_simd, cb_simd, cr_simd) = rgb_to_ycbcr_planes_simd(&rgb_data, num_pixels).unwrap();

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
                i,
                y_simd[i],
                y_scalar[i]
            );
            assert!(
                (cb_simd[i] - cb_scalar[i]).abs() < EPSILON,
                "Cb mismatch at {}: SIMD={}, scalar={}",
                i,
                cb_simd[i],
                cb_scalar[i]
            );
            assert!(
                (cr_simd[i] - cr_scalar[i]).abs() < EPSILON,
                "Cr mismatch at {}: SIMD={}, scalar={}",
                i,
                cr_simd[i],
                cr_scalar[i]
            );
        }
    }

    #[test]
    fn test_rgba_to_ycbcr_simd_matches_scalar() {
        let num_pixels = 24;
        let rgba_data: Vec<u8> = (0..num_pixels * 4)
            .map(|i| ((i * 13 + 3) % 256) as u8)
            .collect();

        let (y_simd, cb_simd, cr_simd) = rgba_to_ycbcr_planes_simd(&rgba_data, num_pixels).unwrap();

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
                i,
                y_simd[i],
                y_scalar
            );
            assert!(
                (cb_simd[i] - cb_scalar).abs() < EPSILON,
                "Cb mismatch at {}: SIMD={}, scalar={}",
                i,
                cb_simd[i],
                cb_scalar
            );
            assert!(
                (cr_simd[i] - cr_scalar).abs() < EPSILON,
                "Cr mismatch at {}: SIMD={}, scalar={}",
                i,
                cr_simd[i],
                cr_scalar
            );
        }
    }

    #[test]
    fn test_bgr_to_ycbcr_simd_matches_scalar() {
        let num_pixels = 24;
        let bgr_data: Vec<u8> = (0..num_pixels * 3)
            .map(|i| ((i * 17 + 5) % 256) as u8)
            .collect();

        let (y_simd, cb_simd, cr_simd) = bgr_to_ycbcr_planes_simd(&bgr_data, num_pixels).unwrap();

        for i in 0..num_pixels {
            // BGR: B at offset 0, G at offset 1, R at offset 2
            let b = bgr_data[i * 3] as f32;
            let g = bgr_data[i * 3 + 1] as f32;
            let r = bgr_data[i * 3 + 2] as f32;

            let y_scalar = YCBCR_R_TO_Y * r + YCBCR_G_TO_Y * g + YCBCR_B_TO_Y * b;
            let cb_scalar = 128.0 + YCBCR_R_TO_CB * r + YCBCR_G_TO_CB * g + YCBCR_B_TO_CB * b;
            let cr_scalar = 128.0 + YCBCR_R_TO_CR * r + YCBCR_G_TO_CR * g + YCBCR_B_TO_CR * b;

            assert!(
                (y_simd[i] - y_scalar).abs() < EPSILON,
                "Y mismatch at {}: SIMD={}, scalar={}",
                i,
                y_simd[i],
                y_scalar
            );
            assert!(
                (cb_simd[i] - cb_scalar).abs() < EPSILON,
                "Cb mismatch at {}: SIMD={}, scalar={}",
                i,
                cb_simd[i],
                cb_scalar
            );
            assert!(
                (cr_simd[i] - cr_scalar).abs() < EPSILON,
                "Cr mismatch at {}: SIMD={}, scalar={}",
                i,
                cr_simd[i],
                cr_scalar
            );
        }
    }

    #[test]
    fn test_bgra_to_ycbcr_simd_matches_scalar() {
        let num_pixels = 24;
        let bgra_data: Vec<u8> = (0..num_pixels * 4)
            .map(|i| ((i * 19 + 7) % 256) as u8)
            .collect();

        let (y_simd, cb_simd, cr_simd) = bgra_to_ycbcr_planes_simd(&bgra_data, num_pixels).unwrap();

        for i in 0..num_pixels {
            // BGRA: B at offset 0, G at offset 1, R at offset 2, A at offset 3
            let b = bgra_data[i * 4] as f32;
            let g = bgra_data[i * 4 + 1] as f32;
            let r = bgra_data[i * 4 + 2] as f32;

            let y_scalar = YCBCR_R_TO_Y * r + YCBCR_G_TO_Y * g + YCBCR_B_TO_Y * b;
            let cb_scalar = 128.0 + YCBCR_R_TO_CB * r + YCBCR_G_TO_CB * g + YCBCR_B_TO_CB * b;
            let cr_scalar = 128.0 + YCBCR_R_TO_CR * r + YCBCR_G_TO_CR * g + YCBCR_B_TO_CR * b;

            assert!(
                (y_simd[i] - y_scalar).abs() < EPSILON,
                "Y mismatch at {}: SIMD={}, scalar={}",
                i,
                y_simd[i],
                y_scalar
            );
            assert!(
                (cb_simd[i] - cb_scalar).abs() < EPSILON,
                "Cb mismatch at {}: SIMD={}, scalar={}",
                i,
                cb_simd[i],
                cb_scalar
            );
            assert!(
                (cr_simd[i] - cr_scalar).abs() < EPSILON,
                "Cr mismatch at {}: SIMD={}, scalar={}",
                i,
                cr_simd[i],
                cr_scalar
            );
        }
    }

    #[test]
    fn test_gray_to_ycbcr_simd_matches_scalar() {
        let num_pixels = 24;
        let gray_data: Vec<u8> = (0..num_pixels)
            .map(|i| ((i * 11 + 3) % 256) as u8)
            .collect();

        let (y_simd, cb_simd, cr_simd) = gray_to_ycbcr_planes_simd(&gray_data, num_pixels).unwrap();

        for i in 0..num_pixels {
            let y_scalar = gray_data[i] as f32;
            let cb_scalar = 128.0f32;
            let cr_scalar = 128.0f32;

            assert!(
                (y_simd[i] - y_scalar).abs() < EPSILON,
                "Y mismatch at {}: SIMD={}, scalar={}",
                i,
                y_simd[i],
                y_scalar
            );
            assert!(
                (cb_simd[i] - cb_scalar).abs() < EPSILON,
                "Cb mismatch at {}: SIMD={}, scalar={}",
                i,
                cb_simd[i],
                cb_scalar
            );
            assert!(
                (cr_simd[i] - cr_scalar).abs() < EPSILON,
                "Cr mismatch at {}: SIMD={}, scalar={}",
                i,
                cr_simd[i],
                cr_scalar
            );
        }
    }

    #[test]
    fn test_extract_block_simd_interior() {
        // Create a 32x32 plane (4x4 blocks)
        let width = 32usize;
        let height = 32usize;
        let plane: Vec<f32> = (0..(width * height)).map(|i| (i % 256) as f32).collect();

        // Test interior block at (1, 1) - no edge handling needed
        let block = extract_block_simd(&plane, width, height, 1, 1);

        // Verify values manually
        for y in 0..8 {
            for x in 0..8 {
                let px = 1 * 8 + x;
                let py = 1 * 8 + y;
                let expected = plane[py * width + px] - 128.0;
                assert!(
                    (block[y * 8 + x] - expected).abs() < EPSILON,
                    "Mismatch at ({}, {}): got={}, expected={}",
                    x,
                    y,
                    block[y * 8 + x],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_extract_block_simd_edge() {
        // Create a 20x20 plane (blocks at edge need clamping)
        let width = 20usize;
        let height = 20usize;
        let plane: Vec<f32> = (0..(width * height)).map(|i| (i % 256) as f32).collect();

        // Test edge block at (2, 2) - partially outside bounds
        let block = extract_block_simd(&plane, width, height, 2, 2);

        // Verify values with clamping
        for y in 0..8 {
            for x in 0..8 {
                let px = (2 * 8 + x).min(width - 1);
                let py = (2 * 8 + y).min(height - 1);
                let expected = plane[py * width + px] - 128.0;
                assert!(
                    (block[y * 8 + x] - expected).abs() < EPSILON,
                    "Mismatch at ({}, {}): got={}, expected={}",
                    x,
                    y,
                    block[y * 8 + x],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_apply_smoothing_simd() {
        let width = 32usize;
        let height = 32usize;
        let plane: Vec<f32> = (0..(width * height)).map(|i| (i % 256) as f32).collect();

        let factor = 64u8; // Moderate smoothing
        let result = apply_smoothing_simd(&plane, width, height, factor).unwrap();

        let kw1 = factor as f32 / 1024.0;
        let kw0 = 1.0 - 8.0 * kw1;

        // Verify some interior pixels
        for y in 2..height - 2 {
            for x in 2..width - 2 {
                let val_tl = plane[(y - 1) * width + x - 1];
                let val_tm = plane[(y - 1) * width + x];
                let val_tr = plane[(y - 1) * width + x + 1];
                let val_ml = plane[y * width + x - 1];
                let val_mm = plane[y * width + x];
                let val_mr = plane[y * width + x + 1];
                let val_bl = plane[(y + 1) * width + x - 1];
                let val_bm = plane[(y + 1) * width + x];
                let val_br = plane[(y + 1) * width + x + 1];

                let neighbors =
                    val_tl + val_tm + val_tr + val_ml + val_mr + val_bl + val_bm + val_br;
                let expected = val_mm * kw0 + neighbors * kw1;

                assert!(
                    (result[y * width + x] - expected).abs() < EPSILON,
                    "Smoothing mismatch at ({}, {}): got={}, expected={}",
                    x,
                    y,
                    result[y * width + x],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_apply_smoothing_simd_factor_zero() {
        let width = 16usize;
        let height = 16usize;
        let plane: Vec<f32> = (0..(width * height)).map(|i| (i % 256) as f32).collect();

        // Factor 0 should return a copy
        let result = apply_smoothing_simd(&plane, width, height, 0).unwrap();
        assert_eq!(plane, result);
    }
}
