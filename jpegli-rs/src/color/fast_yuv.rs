//! Fast SIMD RGB→YCbCr conversion using the `yuv` crate.
//!
//! This module provides 10-150× faster color conversion compared to scalar f32 math
//! by using the `yuv` crate's optimized SIMD implementations (AVX-512/AVX2/SSE/NEON).
//!
//! ## Precision
//!
//! The yuv crate uses 15-bit fixed-point integer math (Professional mode):
//! - Max Y error: ~0.5 levels (out of 255)
//! - Avg Y error: ~0.25 levels
//! - This is invisible after JPEG DCT quantization (which loses 2-4 levels at Q85)
//!
//! ## Usage
//!
//! This module is used automatically when the `fast-yuv` feature is enabled (default).
//! Falls back to scalar f32 conversion when disabled.

use yuv::{
    rgb_to_yuv444, YuvChromaSubsampling, YuvConversionMode, YuvPlanarImageMut, YuvRange,
    YuvStandardMatrix,
};

/// Convert RGB to YCbCr using fast SIMD integer math.
///
/// Outputs f32 planes for compatibility with the rest of the encoder pipeline.
/// The u8→f32 conversion overhead is negligible compared to the 10-150× speedup.
///
/// # Arguments
/// * `rgb_data` - Input RGB data (3 bytes per pixel, row-major)
/// * `y_plane` - Output Y plane (f32, [0, 255] range)
/// * `cb_plane` - Output Cb plane (f32, [0, 255] range)
/// * `cr_plane` - Output Cr plane (f32, [0, 255] range)
/// * `width` - Image width in pixels
/// * `height` - Number of rows
///
/// # Panics
/// Panics if output planes are too small.
#[allow(dead_code)] // Utility function, currently only strided version is used
pub fn rgb_to_ycbcr_fast(
    rgb_data: &[u8],
    y_plane: &mut [f32],
    cb_plane: &mut [f32],
    cr_plane: &mut [f32],
    width: usize,
    height: usize,
) {
    let num_pixels = width * height;
    debug_assert!(rgb_data.len() >= num_pixels * 3);
    debug_assert!(y_plane.len() >= num_pixels);
    debug_assert!(cb_plane.len() >= num_pixels);
    debug_assert!(cr_plane.len() >= num_pixels);

    // Use yuv crate for fast SIMD conversion (outputs u8)
    let mut yuv_image =
        YuvPlanarImageMut::alloc(width as u32, height as u32, YuvChromaSubsampling::Yuv444);

    rgb_to_yuv444(
        &mut yuv_image,
        rgb_data,
        width as u32 * 3,
        YuvRange::Full,
        YuvStandardMatrix::Bt601,
        YuvConversionMode::Professional,
    )
    .expect("yuv conversion failed");

    // Convert u8 planes to f32 (fast - just a cast per pixel)
    let y_u8 = yuv_image.y_plane.borrow();
    let cb_u8 = yuv_image.u_plane.borrow();
    let cr_u8 = yuv_image.v_plane.borrow();

    for i in 0..num_pixels {
        y_plane[i] = y_u8[i] as f32;
        cb_plane[i] = cb_u8[i] as f32;
        cr_plane[i] = cr_u8[i] as f32;
    }
}

/// Convert RGB to YCbCr with strided Y output (for padded strips).
///
/// Y is written with `y_stride` spacing between rows, while Cb/Cr use `width` stride.
/// This matches the strip processor's buffer layout.
///
/// # Arguments
/// * `rgb_data` - Input RGB data (3 or 4 bytes per pixel depending on bpp)
/// * `y_plane` - Output Y plane with y_stride spacing
/// * `cb_plane` - Output Cb plane with width spacing
/// * `cr_plane` - Output Cr plane with width spacing
/// * `width` - Image width in pixels
/// * `height` - Number of rows
/// * `y_stride` - Y output stride (typically padded_width)
/// * `bpp` - Bytes per pixel (3 for RGB, 4 for RGBA)
pub fn rgb_to_ycbcr_strided_fast(
    rgb_data: &[u8],
    y_plane: &mut [f32],
    cb_plane: &mut [f32],
    cr_plane: &mut [f32],
    width: usize,
    height: usize,
    y_stride: usize,
    bpp: usize,
) {
    debug_assert!(rgb_data.len() >= width * height * bpp);
    debug_assert!(y_plane.len() >= y_stride * height);
    debug_assert!(cb_plane.len() >= width * height);
    debug_assert!(cr_plane.len() >= width * height);

    // Handle RGBA by stripping alpha channel
    let rgb_only: Vec<u8>;
    let rgb_input = if bpp == 4 {
        rgb_only = rgb_data
            .chunks_exact(4)
            .take(width * height)
            .flat_map(|chunk| [chunk[0], chunk[1], chunk[2]])
            .collect();
        &rgb_only
    } else {
        rgb_data
    };

    // Use yuv crate for fast SIMD conversion
    let mut yuv_image =
        YuvPlanarImageMut::alloc(width as u32, height as u32, YuvChromaSubsampling::Yuv444);

    rgb_to_yuv444(
        &mut yuv_image,
        rgb_input,
        width as u32 * 3,
        YuvRange::Full,
        YuvStandardMatrix::Bt601,
        YuvConversionMode::Professional,
    )
    .expect("yuv conversion failed");

    let y_u8 = yuv_image.y_plane.borrow();
    let cb_u8 = yuv_image.u_plane.borrow();
    let cr_u8 = yuv_image.v_plane.borrow();

    // Copy with appropriate strides
    if y_stride == width {
        // Fast path: contiguous output
        let num_pixels = width * height;
        for i in 0..num_pixels {
            y_plane[i] = y_u8[i] as f32;
            cb_plane[i] = cb_u8[i] as f32;
            cr_plane[i] = cr_u8[i] as f32;
        }
    } else {
        // Strided path: Y has different stride than Cb/Cr
        for row in 0..height {
            let src_start = row * width;
            let y_dst_start = row * y_stride;
            let cbcr_dst_start = row * width;

            for x in 0..width {
                y_plane[y_dst_start + x] = y_u8[src_start + x] as f32;
                cb_plane[cbcr_dst_start + x] = cb_u8[src_start + x] as f32;
                cr_plane[cbcr_dst_start + x] = cr_u8[src_start + x] as f32;
            }
        }
    }
}

/// Convert BGR to YCbCr with strided Y output.
///
/// Same as `rgb_to_ycbcr_strided_fast` but for BGR/BGRA input.
pub fn bgr_to_ycbcr_strided_fast(
    bgr_data: &[u8],
    y_plane: &mut [f32],
    cb_plane: &mut [f32],
    cr_plane: &mut [f32],
    width: usize,
    height: usize,
    y_stride: usize,
    bpp: usize,
) {
    debug_assert!(bgr_data.len() >= width * height * bpp);
    debug_assert!(y_plane.len() >= y_stride * height);
    debug_assert!(cb_plane.len() >= width * height);
    debug_assert!(cr_plane.len() >= width * height);

    // Convert BGR(A) to RGB for yuv crate
    let rgb_data: Vec<u8> = if bpp == 4 {
        bgr_data
            .chunks_exact(4)
            .take(width * height)
            .flat_map(|chunk| [chunk[2], chunk[1], chunk[0]]) // BGR(A) -> RGB
            .collect()
    } else {
        bgr_data
            .chunks_exact(3)
            .take(width * height)
            .flat_map(|chunk| [chunk[2], chunk[1], chunk[0]]) // BGR -> RGB
            .collect()
    };

    // Use the RGB path
    rgb_to_ycbcr_strided_fast(
        &rgb_data, y_plane, cb_plane, cr_plane, width, height, y_stride, 3,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::rgb_to_ycbcr_f32;

    #[test]
    fn test_fast_yuv_matches_f32() {
        let width = 64;
        let height = 64;
        let num_pixels = width * height;

        // Generate test image
        let mut rgb = vec![0u8; num_pixels * 3];
        for i in 0..num_pixels {
            rgb[i * 3] = (i % 256) as u8;
            rgb[i * 3 + 1] = ((i * 7) % 256) as u8;
            rgb[i * 3 + 2] = ((i * 13) % 256) as u8;
        }

        // Fast conversion
        let mut y_fast = vec![0.0f32; num_pixels];
        let mut cb_fast = vec![0.0f32; num_pixels];
        let mut cr_fast = vec![0.0f32; num_pixels];
        rgb_to_ycbcr_fast(&rgb, &mut y_fast, &mut cb_fast, &mut cr_fast, width, height);

        // Reference f32 conversion
        let mut max_y_diff = 0.0f32;
        let mut max_cb_diff = 0.0f32;
        let mut max_cr_diff = 0.0f32;

        for i in 0..num_pixels {
            let (y_ref, cb_ref, cr_ref) = rgb_to_ycbcr_f32(
                rgb[i * 3] as f32,
                rgb[i * 3 + 1] as f32,
                rgb[i * 3 + 2] as f32,
            );

            max_y_diff = max_y_diff.max((y_fast[i] - y_ref).abs());
            max_cb_diff = max_cb_diff.max((cb_fast[i] - cb_ref).abs());
            max_cr_diff = max_cr_diff.max((cr_fast[i] - cr_ref).abs());
        }

        // Allow up to 1.5 levels difference (integer rounding)
        assert!(max_y_diff < 1.5, "Y diff {} exceeds threshold", max_y_diff);
        assert!(
            max_cb_diff < 1.5,
            "Cb diff {} exceeds threshold",
            max_cb_diff
        );
        assert!(
            max_cr_diff < 1.5,
            "Cr diff {} exceeds threshold",
            max_cr_diff
        );
    }

    #[test]
    fn test_fast_yuv_strided() {
        let width = 60; // Not aligned to 8
        let height = 4;
        let y_stride = 64; // Padded
        let num_pixels = width * height;

        let rgb = vec![128u8; num_pixels * 3];
        let mut y_plane = vec![0.0f32; y_stride * height];
        let mut cb_plane = vec![0.0f32; num_pixels];
        let mut cr_plane = vec![0.0f32; num_pixels];

        rgb_to_ycbcr_strided_fast(
            &rgb,
            &mut y_plane,
            &mut cb_plane,
            &mut cr_plane,
            width,
            height,
            y_stride,
            3,
        );

        // Check gray produces correct Y (should be ~128)
        for row in 0..height {
            for x in 0..width {
                let y = y_plane[row * y_stride + x];
                assert!((y - 128.0).abs() < 2.0, "Gray Y={} at ({}, {})", y, x, row);
            }
        }
    }
}
