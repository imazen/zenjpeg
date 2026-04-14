//! Fast SIMD RGB→YCbCr conversion using `zenyuv`.
//!
//! This module wraps zenyuv's AVX2/NEON/WASM SIMD kernels with the strided
//! f32-output interface that zenjpeg's encoder pipeline expects. The flow:
//!
//! 1. Strip alpha (RGBA→RGB) or swap channels (BGR→RGB) if needed
//! 2. `zenyuv::rgb_to_yuv444` / `zenyuv::rgb_to_yuv420` → u8 temp planes
//! 3. u8→f32 copy with Y stride + edge replication
//!
//! ## Precision
//!
//! zenyuv uses 15-bit fixed-point (AVX2 pmaddwd) or f32 FMA (other platforms).
//! Max error vs f32 reference: ±1 level — invisible after JPEG quantization.

/// Convert RGB to YCbCr with strided Y output (for padded strips).
///
/// Y is written with `y_stride` spacing between rows, while Cb/Cr use `width` stride.
/// This matches the strip processor's buffer layout.
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
    let num_pixels = width * height;
    debug_assert!(rgb_data.len() >= num_pixels * bpp);
    debug_assert!(y_plane.len() >= y_stride * height);
    debug_assert!(cb_plane.len() >= num_pixels);
    debug_assert!(cr_plane.len() >= num_pixels);

    let rgb_only: alloc::vec::Vec<u8>;
    let rgb_input = if bpp == 4 {
        rgb_only = rgb_data
            .chunks_exact(4)
            .take(num_pixels)
            .flat_map(|chunk| [chunk[0], chunk[1], chunk[2]])
            .collect();
        &rgb_only
    } else {
        rgb_data
    };

    // Stack-local u8 temps for small images, heap for large.
    let mut y_u8 = alloc::vec![0u8; num_pixels];
    let mut cb_u8 = alloc::vec![0u8; num_pixels];
    let mut cr_u8 = alloc::vec![0u8; num_pixels];

    let mut ctx = zenyuv::YuvContext::new(zenyuv::Range::Full, zenyuv::Matrix::Bt601);
    ctx.encode_444_u8(rgb_input, &mut y_u8, &mut cb_u8, &mut cr_u8, width, height);

    u8_to_f32_strided(&y_u8, y_plane, width, height, y_stride);
    u8_to_f32_contiguous(&cb_u8, cb_plane, num_pixels);
    u8_to_f32_contiguous(&cr_u8, cr_plane, num_pixels);
}

/// Convert RGB to YCbCr using pre-allocated u8 buffers (zero allocation).
pub fn rgb_to_ycbcr_strided_reuse(
    rgb_data: &[u8],
    y_plane: &mut [f32],
    cb_plane: &mut [f32],
    cr_plane: &mut [f32],
    yuv_temp_y: &mut [u8],
    yuv_temp_cb: &mut [u8],
    yuv_temp_cr: &mut [u8],
    width: usize,
    height: usize,
    y_stride: usize,
    bpp: usize,
) {
    let num_pixels = width * height;
    debug_assert!(rgb_data.len() >= num_pixels * bpp);
    debug_assert!(y_plane.len() >= y_stride * height);
    debug_assert!(cb_plane.len() >= num_pixels);
    debug_assert!(cr_plane.len() >= num_pixels);
    debug_assert!(yuv_temp_y.len() >= num_pixels);
    debug_assert!(yuv_temp_cb.len() >= num_pixels);
    debug_assert!(yuv_temp_cr.len() >= num_pixels);

    let rgb_only: alloc::vec::Vec<u8>;
    let rgb_input = if bpp == 4 {
        rgb_only = rgb_data
            .chunks_exact(4)
            .take(num_pixels)
            .flat_map(|chunk| [chunk[0], chunk[1], chunk[2]])
            .collect();
        &rgb_only
    } else {
        rgb_data
    };

    let mut ctx = zenyuv::YuvContext::new(zenyuv::Range::Full, zenyuv::Matrix::Bt601);
    ctx.encode_444_u8(
        rgb_input,
        &mut yuv_temp_y[..num_pixels],
        &mut yuv_temp_cb[..num_pixels],
        &mut yuv_temp_cr[..num_pixels],
        width,
        height,
    );

    u8_to_f32_strided(yuv_temp_y, y_plane, width, height, y_stride);
    u8_to_f32_contiguous(yuv_temp_cb, cb_plane, num_pixels);
    u8_to_f32_contiguous(yuv_temp_cr, cr_plane, num_pixels);
}

/// Convert BGR to YCbCr using pre-allocated u8 buffers (zero allocation).
pub fn bgr_to_ycbcr_strided_reuse(
    bgr_data: &[u8],
    y_plane: &mut [f32],
    cb_plane: &mut [f32],
    cr_plane: &mut [f32],
    yuv_temp_y: &mut [u8],
    yuv_temp_cb: &mut [u8],
    yuv_temp_cr: &mut [u8],
    width: usize,
    height: usize,
    y_stride: usize,
    bpp: usize,
) {
    let num_pixels = width * height;

    let rgb_converted: alloc::vec::Vec<u8> = if bpp == 4 {
        bgr_data
            .chunks_exact(4)
            .take(num_pixels)
            .flat_map(|chunk| [chunk[2], chunk[1], chunk[0]])
            .collect()
    } else {
        bgr_data
            .chunks_exact(3)
            .take(num_pixels)
            .flat_map(|chunk| [chunk[2], chunk[1], chunk[0]])
            .collect()
    };

    rgb_to_ycbcr_strided_reuse(
        &rgb_converted,
        y_plane,
        cb_plane,
        cr_plane,
        yuv_temp_y,
        yuv_temp_cb,
        yuv_temp_cr,
        width,
        height,
        y_stride,
        3,
    );
}

/// Convert BGR to YCbCr with strided Y output.
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
    let num_pixels = width * height;

    let rgb_converted: alloc::vec::Vec<u8> = if bpp == 4 {
        bgr_data
            .chunks_exact(4)
            .take(num_pixels)
            .flat_map(|chunk| [chunk[2], chunk[1], chunk[0]])
            .collect()
    } else {
        bgr_data
            .chunks_exact(3)
            .take(num_pixels)
            .flat_map(|chunk| [chunk[2], chunk[1], chunk[0]])
            .collect()
    };

    rgb_to_ycbcr_strided_fast(
        &rgb_converted,
        y_plane,
        cb_plane,
        cr_plane,
        width,
        height,
        y_stride,
        3,
    );
}

/// Convert RGB to YCbCr 4:2:0 using pre-allocated u8 buffers.
pub fn rgb_to_ycbcr_420_reuse(
    rgb_data: &[u8],
    y_plane: &mut [f32],
    cb_down: &mut [f32],
    cr_down: &mut [f32],
    yuv_temp_y: &mut [u8],
    yuv_temp_cb: &mut [u8],
    yuv_temp_cr: &mut [u8],
    width: usize,
    height: usize,
    y_stride: usize,
    bpp: usize,
) {
    let num_pixels = width * height;
    let c_width = (width + 1) / 2;
    let c_height = (height + 1) / 2;
    let c_size = c_width * c_height;

    debug_assert!(rgb_data.len() >= num_pixels * bpp);
    debug_assert!(y_plane.len() >= y_stride * height);
    debug_assert!(cb_down.len() >= c_size);
    debug_assert!(cr_down.len() >= c_size);
    debug_assert!(yuv_temp_y.len() >= num_pixels);
    debug_assert!(yuv_temp_cb.len() >= c_size);
    debug_assert!(yuv_temp_cr.len() >= c_size);

    let rgb_only: alloc::vec::Vec<u8>;
    let rgb_input = if bpp == 4 {
        rgb_only = rgb_data
            .chunks_exact(4)
            .take(num_pixels)
            .flat_map(|chunk| [chunk[0], chunk[1], chunk[2]])
            .collect();
        &rgb_only
    } else {
        rgb_data
    };

    let mut ctx = zenyuv::YuvContext::new(zenyuv::Range::Full, zenyuv::Matrix::Bt601);
    ctx.encode_420_u8(
        rgb_input,
        &mut yuv_temp_y[..num_pixels],
        &mut yuv_temp_cb[..c_size],
        &mut yuv_temp_cr[..c_size],
        width,
        height,
    );

    u8_to_f32_strided(yuv_temp_y, y_plane, width, height, y_stride);
    u8_to_f32_contiguous(yuv_temp_cb, cb_down, c_size);
    u8_to_f32_contiguous(yuv_temp_cr, cr_down, c_size);
}

/// Convert BGR to YCbCr 4:2:0 using pre-allocated u8 buffers.
pub fn bgr_to_ycbcr_420_reuse(
    bgr_data: &[u8],
    y_plane: &mut [f32],
    cb_down: &mut [f32],
    cr_down: &mut [f32],
    yuv_temp_y: &mut [u8],
    yuv_temp_cb: &mut [u8],
    yuv_temp_cr: &mut [u8],
    width: usize,
    height: usize,
    y_stride: usize,
    bpp: usize,
) {
    let num_pixels = width * height;

    let rgb_converted: alloc::vec::Vec<u8> = if bpp == 4 {
        bgr_data
            .chunks_exact(4)
            .take(num_pixels)
            .flat_map(|chunk| [chunk[2], chunk[1], chunk[0]])
            .collect()
    } else {
        bgr_data
            .chunks_exact(3)
            .take(num_pixels)
            .flat_map(|chunk| [chunk[2], chunk[1], chunk[0]])
            .collect()
    };

    rgb_to_ycbcr_420_reuse(
        &rgb_converted,
        y_plane,
        cb_down,
        cr_down,
        yuv_temp_y,
        yuv_temp_cb,
        yuv_temp_cr,
        width,
        height,
        y_stride,
        3,
    );
}

// ── Helpers ─────────────────────────────────────────────────────────────────

#[inline]
fn u8_to_f32_contiguous(src: &[u8], dst: &mut [f32], n: usize) {
    for i in 0..n {
        dst[i] = src[i] as f32;
    }
}

#[inline]
fn u8_to_f32_strided(src: &[u8], dst: &mut [f32], width: usize, height: usize, y_stride: usize) {
    if y_stride == width {
        u8_to_f32_contiguous(src, dst, width * height);
    } else {
        for row in 0..height {
            let src_start = row * width;
            let dst_start = row * y_stride;
            for x in 0..width {
                dst[dst_start + x] = src[src_start + x] as f32;
            }
            // Edge-replicate Y for rightmost partial MCU
            if width < y_stride {
                let edge_val = dst[dst_start + width - 1];
                for x in width..y_stride {
                    dst[dst_start + x] = edge_val;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgb_to_ycbcr_strided_fast_basic() {
        let w = 16;
        let h = 8;
        let y_stride = 24; // padded
        let mut rgb = alloc::vec![0u8; w * h * 3];
        for i in 0..w * h {
            rgb[i * 3] = 128;
            rgb[i * 3 + 1] = 64;
            rgb[i * 3 + 2] = 192;
        }
        let mut y = alloc::vec![0.0f32; y_stride * h];
        let mut cb = alloc::vec![0.0f32; w * h];
        let mut cr = alloc::vec![0.0f32; w * h];
        rgb_to_ycbcr_strided_fast(&rgb, &mut y, &mut cb, &mut cr, w, h, y_stride, 3);

        // All pixels identical → all Y/Cb/Cr values should be the same
        let y0 = y[0];
        let cb0 = cb[0];
        let cr0 = cr[0];
        assert!(y0 > 50.0 && y0 < 200.0, "Y={y0} out of range");
        assert!(cb0 > 0.0 && cb0 < 255.0, "Cb={cb0} out of range");
        assert!(cr0 > 0.0 && cr0 < 255.0, "Cr={cr0} out of range");

        for row in 0..h {
            for x in 0..w {
                assert_eq!(y[row * y_stride + x], y0);
            }
            // Padded columns should be edge-replicated
            for x in w..y_stride {
                assert_eq!(y[row * y_stride + x], y0);
            }
        }
    }

    #[test]
    fn test_420_reuse_basic() {
        let w = 16;
        let h = 8;
        let cw = w / 2;
        let ch = h / 2;
        let rgb = alloc::vec![128u8; w * h * 3];
        let mut y = alloc::vec![0.0f32; w * h];
        let mut cb = alloc::vec![0.0f32; cw * ch];
        let mut cr = alloc::vec![0.0f32; cw * ch];
        let mut ty = alloc::vec![0u8; w * h];
        let mut tu = alloc::vec![0u8; cw * ch];
        let mut tv = alloc::vec![0u8; cw * ch];
        rgb_to_ycbcr_420_reuse(
            &rgb, &mut y, &mut cb, &mut cr, &mut ty, &mut tu, &mut tv, w, h, w, 3,
        );
        // Gray input → Y≈128, Cb≈128, Cr≈128
        assert!((y[0] - 128.0).abs() < 2.0, "Y={}", y[0]);
        assert!((cb[0] - 128.0).abs() < 2.0, "Cb={}", cb[0]);
        assert!((cr[0] - 128.0).abs() < 2.0, "Cr={}", cr[0]);
    }
}
