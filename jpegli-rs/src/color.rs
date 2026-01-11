//! Color space conversion functions.
//!
//! This module provides conversions between:
//! - RGB and YCbCr (BT.601 standard JPEG color space)
//! - RGB and CMYK
//! - Various pixel format conversions
//!
//! SIMD optimization is available via the `simd` feature (enabled by default).

use crate::alloc::{checked_size, checked_size_2d, try_alloc_zeroed};
use crate::consts::{
    YCBCR_B_TO_CB, YCBCR_B_TO_CR, YCBCR_B_TO_Y, YCBCR_CB_TO_B, YCBCR_CB_TO_G, YCBCR_CB_TO_R,
    YCBCR_CR_TO_B, YCBCR_CR_TO_G, YCBCR_CR_TO_R, YCBCR_G_TO_CB, YCBCR_G_TO_CR, YCBCR_G_TO_Y,
    YCBCR_R_TO_CB, YCBCR_R_TO_CR, YCBCR_R_TO_Y, YCBCR_Y_TO_B, YCBCR_Y_TO_G, YCBCR_Y_TO_R,
};
use crate::error::Result;
use crate::types::PixelFormat;

use multiversion::multiversion;
use wide::{f32x4, f32x8};

/// Converts a single RGB pixel to YCbCr.
///
/// Uses BT.601 coefficients (standard JPEG).
/// Y is in range [0, 255], Cb and Cr are in range [0, 255] (centered at 128).
#[inline]
#[must_use]
pub fn rgb_to_ycbcr(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
    let rf = r as f32;
    let gf = g as f32;
    let bf = b as f32;

    // Y = 0.299*R + 0.587*G + 0.114*B - use FMA for accuracy
    let y = YCBCR_R_TO_Y.mul_add(rf, YCBCR_G_TO_Y.mul_add(gf, YCBCR_B_TO_Y * bf));

    // Cb = 128 - 0.168736*R - 0.331264*G + 0.5*B
    let cb = YCBCR_R_TO_CB.mul_add(rf, YCBCR_G_TO_CB.mul_add(gf, YCBCR_B_TO_CB.mul_add(bf, 128.0)));

    // Cr = 128 + 0.5*R - 0.418688*G - 0.081312*B
    let cr = YCBCR_R_TO_CR.mul_add(rf, YCBCR_G_TO_CR.mul_add(gf, YCBCR_B_TO_CR.mul_add(bf, 128.0)));

    (
        y.round().clamp(0.0, 255.0) as u8,
        cb.round().clamp(0.0, 255.0) as u8,
        cr.round().clamp(0.0, 255.0) as u8,
    )
}

/// Converts a single YCbCr pixel to RGB.
#[inline]
#[must_use]
pub fn ycbcr_to_rgb(y: u8, cb: u8, cr: u8) -> (u8, u8, u8) {
    let yf = y as f32;
    let cbf = cb as f32 - 128.0;
    let crf = cr as f32 - 128.0;

    // R = Y + 1.402*Cr - use FMA for accuracy
    let r = YCBCR_Y_TO_R.mul_add(yf, YCBCR_CB_TO_R.mul_add(cbf, YCBCR_CR_TO_R * crf));

    // G = Y - 0.344136*Cb - 0.714136*Cr
    let g = YCBCR_Y_TO_G.mul_add(yf, YCBCR_CB_TO_G.mul_add(cbf, YCBCR_CR_TO_G * crf));

    // B = Y + 1.772*Cb
    let b = YCBCR_Y_TO_B.mul_add(yf, YCBCR_CB_TO_B.mul_add(cbf, YCBCR_CR_TO_B * crf));

    (
        r.round().clamp(0.0, 255.0) as u8,
        g.round().clamp(0.0, 255.0) as u8,
        b.round().clamp(0.0, 255.0) as u8,
    )
}

/// Converts RGB float values to YCbCr float values.
///
/// Input/output range is [0.0, 255.0].
#[inline]
#[must_use]
pub fn rgb_to_ycbcr_f32(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    // Use FMA for accuracy (single rounding)
    let y = YCBCR_R_TO_Y.mul_add(r, YCBCR_G_TO_Y.mul_add(g, YCBCR_B_TO_Y * b));
    let cb = YCBCR_R_TO_CB.mul_add(r, YCBCR_G_TO_CB.mul_add(g, YCBCR_B_TO_CB.mul_add(b, 128.0)));
    let cr = YCBCR_R_TO_CR.mul_add(r, YCBCR_G_TO_CR.mul_add(g, YCBCR_B_TO_CR.mul_add(b, 128.0)));
    (y, cb, cr)
}

/// Converts YCbCr float values to RGB float values.
#[inline]
#[must_use]
pub fn ycbcr_to_rgb_f32(y: f32, cb: f32, cr: f32) -> (f32, f32, f32) {
    let cbf = cb - 128.0;
    let crf = cr - 128.0;

    let r = YCBCR_Y_TO_R * y + YCBCR_CB_TO_R * cbf + YCBCR_CR_TO_R * crf;
    let g = YCBCR_Y_TO_G * y + YCBCR_CB_TO_G * cbf + YCBCR_CR_TO_G * crf;
    let b = YCBCR_Y_TO_B * y + YCBCR_CB_TO_B * cbf + YCBCR_CR_TO_B * crf;

    (r, g, b)
}

/// Converts an RGB image buffer to YCbCr in-place.
///
/// The buffer is assumed to be in RGB order (3 bytes per pixel).
pub fn convert_rgb_to_ycbcr_buffer(buffer: &mut [u8]) {
    assert!(buffer.len() % 3 == 0, "Buffer length must be multiple of 3");

    for chunk in buffer.chunks_exact_mut(3) {
        let (y, cb, cr) = rgb_to_ycbcr(chunk[0], chunk[1], chunk[2]);
        chunk[0] = y;
        chunk[1] = cb;
        chunk[2] = cr;
    }
}

/// Converts a YCbCr image buffer to RGB in-place.
pub fn convert_ycbcr_to_rgb_buffer(buffer: &mut [u8]) {
    assert!(buffer.len() % 3 == 0, "Buffer length must be multiple of 3");

    for chunk in buffer.chunks_exact_mut(3) {
        let (r, g, b) = ycbcr_to_rgb(chunk[0], chunk[1], chunk[2]);
        chunk[0] = r;
        chunk[1] = g;
        chunk[2] = b;
    }
}

// SIMD-optimized color conversion
#[cfg(feature = "simd")]
mod simd {
    use super::*;

    /// Process 4 RGB pixels to YCbCr using SIMD.
    /// Returns (Y[4], Cb[4], Cr[4]) as u8 arrays.
    #[inline]
    pub fn rgb_to_ycbcr_x4(r: [u8; 4], g: [u8; 4], b: [u8; 4]) -> ([u8; 4], [u8; 4], [u8; 4]) {
        // Convert to f32 vectors
        let rf = f32x4::from([r[0] as f32, r[1] as f32, r[2] as f32, r[3] as f32]);
        let gf = f32x4::from([g[0] as f32, g[1] as f32, g[2] as f32, g[3] as f32]);
        let bf = f32x4::from([b[0] as f32, b[1] as f32, b[2] as f32, b[3] as f32]);

        // YCbCr coefficients as vectors
        let r_to_y = f32x4::splat(YCBCR_R_TO_Y);
        let g_to_y = f32x4::splat(YCBCR_G_TO_Y);
        let b_to_y = f32x4::splat(YCBCR_B_TO_Y);

        let r_to_cb = f32x4::splat(YCBCR_R_TO_CB);
        let g_to_cb = f32x4::splat(YCBCR_G_TO_CB);
        let b_to_cb = f32x4::splat(YCBCR_B_TO_CB);

        let r_to_cr = f32x4::splat(YCBCR_R_TO_CR);
        let g_to_cr = f32x4::splat(YCBCR_G_TO_CR);
        let b_to_cr = f32x4::splat(YCBCR_B_TO_CR);

        let offset_128 = f32x4::splat(128.0);

        // Compute Y, Cb, Cr (using FMA)
        let y = r_to_y.mul_add(rf, g_to_y.mul_add(gf, b_to_y * bf));
        let cb = r_to_cb.mul_add(rf, g_to_cb.mul_add(gf, b_to_cb.mul_add(bf, offset_128)));
        let cr = r_to_cr.mul_add(rf, g_to_cr.mul_add(gf, b_to_cr.mul_add(bf, offset_128)));

        // Round and clamp to u8
        let y_arr = y.to_array();
        let cb_arr = cb.to_array();
        let cr_arr = cr.to_array();

        let clamp = |v: f32| v.round().clamp(0.0, 255.0) as u8;

        (
            [
                clamp(y_arr[0]),
                clamp(y_arr[1]),
                clamp(y_arr[2]),
                clamp(y_arr[3]),
            ],
            [
                clamp(cb_arr[0]),
                clamp(cb_arr[1]),
                clamp(cb_arr[2]),
                clamp(cb_arr[3]),
            ],
            [
                clamp(cr_arr[0]),
                clamp(cr_arr[1]),
                clamp(cr_arr[2]),
                clamp(cr_arr[3]),
            ],
        )
    }

    /// Process 4 YCbCr pixels to RGB using SIMD.
    #[inline]
    pub fn ycbcr_to_rgb_x4(y: [u8; 4], cb: [u8; 4], cr: [u8; 4]) -> ([u8; 4], [u8; 4], [u8; 4]) {
        // Convert to f32 vectors
        let yf = f32x4::from([y[0] as f32, y[1] as f32, y[2] as f32, y[3] as f32]);
        let cbf = f32x4::from([cb[0] as f32, cb[1] as f32, cb[2] as f32, cb[3] as f32])
            - f32x4::splat(128.0);
        let crf = f32x4::from([cr[0] as f32, cr[1] as f32, cr[2] as f32, cr[3] as f32])
            - f32x4::splat(128.0);

        // RGB coefficients as vectors
        let y_to_r = f32x4::splat(YCBCR_Y_TO_R);
        let cb_to_r = f32x4::splat(YCBCR_CB_TO_R);
        let cr_to_r = f32x4::splat(YCBCR_CR_TO_R);

        let y_to_g = f32x4::splat(YCBCR_Y_TO_G);
        let cb_to_g = f32x4::splat(YCBCR_CB_TO_G);
        let cr_to_g = f32x4::splat(YCBCR_CR_TO_G);

        let y_to_b = f32x4::splat(YCBCR_Y_TO_B);
        let cb_to_b = f32x4::splat(YCBCR_CB_TO_B);
        let cr_to_b = f32x4::splat(YCBCR_CR_TO_B);

        // Compute R, G, B (using FMA)
        let r = y_to_r.mul_add(yf, cb_to_r.mul_add(cbf, cr_to_r * crf));
        let g = y_to_g.mul_add(yf, cb_to_g.mul_add(cbf, cr_to_g * crf));
        let b = y_to_b.mul_add(yf, cb_to_b.mul_add(cbf, cr_to_b * crf));

        // Round and clamp to u8
        let r_arr = r.to_array();
        let g_arr = g.to_array();
        let b_arr = b.to_array();

        let clamp = |v: f32| v.round().clamp(0.0, 255.0) as u8;

        (
            [
                clamp(r_arr[0]),
                clamp(r_arr[1]),
                clamp(r_arr[2]),
                clamp(r_arr[3]),
            ],
            [
                clamp(g_arr[0]),
                clamp(g_arr[1]),
                clamp(g_arr[2]),
                clamp(g_arr[3]),
            ],
            [
                clamp(b_arr[0]),
                clamp(b_arr[1]),
                clamp(b_arr[2]),
                clamp(b_arr[3]),
            ],
        )
    }
}

/// Converts RGB to separate Y, Cb, Cr planes.
///
/// Uses SIMD optimization when the `simd` feature is enabled.
///
/// # Errors
///
/// Returns an error if memory allocation fails.
#[cfg(feature = "simd")]
pub fn rgb_to_ycbcr_planes(
    rgb: &[u8],
    width: usize,
    height: usize,
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let num_pixels = checked_size_2d(width, height)?;
    let expected_len = checked_size(width, height, 3)?;
    assert_eq!(rgb.len(), expected_len);

    let mut y_plane = try_alloc_zeroed(num_pixels, "YCbCr Y plane")?;
    let mut cb_plane = try_alloc_zeroed(num_pixels, "YCbCr Cb plane")?;
    let mut cr_plane = try_alloc_zeroed(num_pixels, "YCbCr Cr plane")?;

    // Process 4 pixels at a time with SIMD
    let chunks = num_pixels / 4;
    for chunk in 0..chunks {
        let base = chunk * 4;
        let rgb_base = base * 3;

        let r = [
            rgb[rgb_base],
            rgb[rgb_base + 3],
            rgb[rgb_base + 6],
            rgb[rgb_base + 9],
        ];
        let g = [
            rgb[rgb_base + 1],
            rgb[rgb_base + 4],
            rgb[rgb_base + 7],
            rgb[rgb_base + 10],
        ];
        let b = [
            rgb[rgb_base + 2],
            rgb[rgb_base + 5],
            rgb[rgb_base + 8],
            rgb[rgb_base + 11],
        ];

        let (y, cb, cr) = simd::rgb_to_ycbcr_x4(r, g, b);

        y_plane[base..base + 4].copy_from_slice(&y);
        cb_plane[base..base + 4].copy_from_slice(&cb);
        cr_plane[base..base + 4].copy_from_slice(&cr);
    }

    // Handle remaining pixels with scalar code
    for i in (chunks * 4)..num_pixels {
        let (y, cb, cr) = rgb_to_ycbcr(rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]);
        y_plane[i] = y;
        cb_plane[i] = cb;
        cr_plane[i] = cr;
    }

    Ok((y_plane, cb_plane, cr_plane))
}

/// Converts RGB to separate Y, Cb, Cr planes (scalar version).
///
/// # Errors
///
/// Returns an error if memory allocation fails.
#[cfg(not(feature = "simd"))]
pub fn rgb_to_ycbcr_planes(
    rgb: &[u8],
    width: usize,
    height: usize,
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let num_pixels = checked_size_2d(width, height)?;
    let expected_len = checked_size(width, height, 3)?;
    assert_eq!(rgb.len(), expected_len);

    let mut y_plane = try_alloc_zeroed(num_pixels, "YCbCr Y plane")?;
    let mut cb_plane = try_alloc_zeroed(num_pixels, "YCbCr Cb plane")?;
    let mut cr_plane = try_alloc_zeroed(num_pixels, "YCbCr Cr plane")?;

    for i in 0..num_pixels {
        let (y, cb, cr) = rgb_to_ycbcr(rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]);
        y_plane[i] = y;
        cb_plane[i] = cb;
        cr_plane[i] = cr;
    }

    Ok((y_plane, cb_plane, cr_plane))
}

/// Converts separate Y, Cb, Cr planes to RGB.
///
/// Uses SIMD optimization when the `simd` feature is enabled.
///
/// # Errors
///
/// Returns an error if memory allocation fails.
#[cfg(feature = "simd")]
pub fn ycbcr_planes_to_rgb(
    y_plane: &[u8],
    cb_plane: &[u8],
    cr_plane: &[u8],
    width: usize,
    height: usize,
) -> Result<Vec<u8>> {
    let num_pixels = checked_size_2d(width, height)?;
    assert_eq!(y_plane.len(), num_pixels);
    assert_eq!(cb_plane.len(), num_pixels);
    assert_eq!(cr_plane.len(), num_pixels);

    let rgb_size = checked_size(width, height, 3)?;
    let mut rgb = try_alloc_zeroed(rgb_size, "RGB output buffer")?;

    // Process 4 pixels at a time with SIMD
    let chunks = num_pixels / 4;
    for chunk in 0..chunks {
        let base = chunk * 4;
        let rgb_base = base * 3;

        let y = [
            y_plane[base],
            y_plane[base + 1],
            y_plane[base + 2],
            y_plane[base + 3],
        ];
        let cb = [
            cb_plane[base],
            cb_plane[base + 1],
            cb_plane[base + 2],
            cb_plane[base + 3],
        ];
        let cr = [
            cr_plane[base],
            cr_plane[base + 1],
            cr_plane[base + 2],
            cr_plane[base + 3],
        ];

        let (r, g, b) = simd::ycbcr_to_rgb_x4(y, cb, cr);

        // Store in interleaved RGB format
        rgb[rgb_base] = r[0];
        rgb[rgb_base + 1] = g[0];
        rgb[rgb_base + 2] = b[0];
        rgb[rgb_base + 3] = r[1];
        rgb[rgb_base + 4] = g[1];
        rgb[rgb_base + 5] = b[1];
        rgb[rgb_base + 6] = r[2];
        rgb[rgb_base + 7] = g[2];
        rgb[rgb_base + 8] = b[2];
        rgb[rgb_base + 9] = r[3];
        rgb[rgb_base + 10] = g[3];
        rgb[rgb_base + 11] = b[3];
    }

    // Handle remaining pixels with scalar code
    for i in (chunks * 4)..num_pixels {
        let (r, g, b) = ycbcr_to_rgb(y_plane[i], cb_plane[i], cr_plane[i]);
        rgb[i * 3] = r;
        rgb[i * 3 + 1] = g;
        rgb[i * 3 + 2] = b;
    }

    Ok(rgb)
}

/// Converts separate Y, Cb, Cr planes to RGB (scalar version).
///
/// # Errors
///
/// Returns an error if memory allocation fails.
#[cfg(not(feature = "simd"))]
pub fn ycbcr_planes_to_rgb(
    y_plane: &[u8],
    cb_plane: &[u8],
    cr_plane: &[u8],
    width: usize,
    height: usize,
) -> Result<Vec<u8>> {
    let num_pixels = checked_size_2d(width, height)?;
    assert_eq!(y_plane.len(), num_pixels);
    assert_eq!(cb_plane.len(), num_pixels);
    assert_eq!(cr_plane.len(), num_pixels);

    let rgb_size = checked_size(width, height, 3)?;
    let mut rgb = try_alloc_zeroed(rgb_size, "RGB output buffer")?;

    for i in 0..num_pixels {
        let (r, g, b) = ycbcr_to_rgb(y_plane[i], cb_plane[i], cr_plane[i]);
        rgb[i * 3] = r;
        rgb[i * 3 + 1] = g;
        rgb[i * 3 + 2] = b;
    }

    Ok(rgb)
}

// =============================================================================
// Batch f32 color conversion for decoder
// =============================================================================

/// Batch YCbCr to RGB conversion for f32 planes.
///
/// Converts separate Y, Cb, Cr f32 planes to interleaved RGB u8.
/// Input values are in IDCT output range (centered around 0).
/// Applies level shift (+128) and clamps to 0-255.
///
/// This is optimized for the decoder which processes planes separately.
#[multiversion(targets("x86_64+avx2+fma", "x86_64+sse2", "aarch64+neon"))]
pub fn ycbcr_planes_f32_to_rgb_u8(
    y_plane: &[f32],
    cb_plane: &[f32],
    cr_plane: &[f32],
    rgb: &mut [u8],
) {
    debug_assert_eq!(y_plane.len(), cb_plane.len());
    debug_assert_eq!(y_plane.len(), cr_plane.len());
    debug_assert_eq!(rgb.len(), y_plane.len() * 3);

    let num_pixels = y_plane.len();

    // BT.601 coefficients
    const CR_TO_R: f32 = 1.402;
    const CB_TO_G: f32 = -0.344136;
    const CR_TO_G: f32 = -0.714136;
    const CB_TO_B: f32 = 1.772;

    let cr_to_r = f32x8::splat(CR_TO_R);
    let cb_to_g = f32x8::splat(CB_TO_G);
    let cr_to_g = f32x8::splat(CR_TO_G);
    let cb_to_b = f32x8::splat(CB_TO_B);
    let offset = f32x8::splat(128.0);
    let zero = f32x8::splat(0.0);
    let max_val = f32x8::splat(255.0);

    let chunks = num_pixels / 8;
    for chunk in 0..chunks {
        let base = chunk * 8;

        // Load planes directly from slices
        let y = f32x8::from(&y_plane[base..base + 8]);
        let cb = f32x8::from(&cb_plane[base..base + 8]);
        let cr = f32x8::from(&cr_plane[base..base + 8]);

        // YCbCr to RGB (using FMA)
        let r = cr_to_r.mul_add(cr, y + offset).max(zero).min(max_val);
        let g = cb_to_g.mul_add(cb, cr_to_g.mul_add(cr, y + offset))
            .max(zero)
            .min(max_val);
        let b = cb_to_b.mul_add(cb, y + offset).max(zero).min(max_val);

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

    // Handle remaining pixels with scalar code
    for i in (chunks * 8)..num_pixels {
        let y = y_plane[i];
        let cb = cb_plane[i];
        let cr = cr_plane[i];

        // Use FMA for scalar remainder
        let r = CR_TO_R.mul_add(cr, y);
        let g = CB_TO_G.mul_add(cb, CR_TO_G.mul_add(cr, y));
        let b = CB_TO_B.mul_add(cb, y);

        let idx = i * 3;
        rgb[idx] = (r + 128.0).clamp(0.0, 255.0) as u8;
        rgb[idx + 1] = (g + 128.0).clamp(0.0, 255.0) as u8;
        rgb[idx + 2] = (b + 128.0).clamp(0.0, 255.0) as u8;
    }
}

/// Batch YCbCr to RGB conversion for f32 planes to f32 output.
///
/// Output values are normalized to 0.0-1.0 range.
#[inline(never)]
pub fn ycbcr_planes_f32_to_rgb_f32(
    y_plane: &[f32],
    cb_plane: &[f32],
    cr_plane: &[f32],
    rgb: &mut [f32],
) {
    debug_assert_eq!(y_plane.len(), cb_plane.len());
    debug_assert_eq!(y_plane.len(), cr_plane.len());
    debug_assert_eq!(rgb.len(), y_plane.len() * 3);

    let num_pixels = y_plane.len();

    const CR_TO_R: f32 = 1.402;
    const CB_TO_G: f32 = -0.344136;
    const CR_TO_G: f32 = -0.714136;
    const CB_TO_B: f32 = 1.772;

    #[cfg(feature = "simd")]
    {
        let cr_to_r = f32x8::splat(CR_TO_R);
        let cb_to_g = f32x8::splat(CB_TO_G);
        let cr_to_g = f32x8::splat(CR_TO_G);
        let cb_to_b = f32x8::splat(CB_TO_B);
        let offset = f32x8::splat(128.0);
        let scale = f32x8::splat(1.0 / 255.0);
        let zero = f32x8::splat(0.0);
        let one = f32x8::splat(1.0);

        let chunks = num_pixels / 8;
        for chunk in 0..chunks {
            let base = chunk * 8;
            // Use slice loads instead of manual gather
            let y = f32x8::from(&y_plane[base..base + 8]);
            let cb = f32x8::from(&cb_plane[base..base + 8]);
            let cr = f32x8::from(&cr_plane[base..base + 8]);

            // YCbCr to RGB, level shift, normalize to 0-1 (using FMA)
            let r = (cr_to_r.mul_add(cr, y + offset) * scale).max(zero).min(one);
            let g = (cb_to_g.mul_add(cb, cr_to_g.mul_add(cr, y + offset)) * scale)
                .max(zero)
                .min(one);
            let b = (cb_to_b.mul_add(cb, y + offset) * scale).max(zero).min(one);

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

        // Scalar remainder (using FMA)
        for i in (chunks * 8)..num_pixels {
            let y = y_plane[i];
            let cb = cb_plane[i];
            let cr = cr_plane[i];

            let r = CR_TO_R.mul_add(cr, y);
            let g = CB_TO_G.mul_add(cb, CR_TO_G.mul_add(cr, y));
            let b = CB_TO_B.mul_add(cb, y);

            let idx = i * 3;
            rgb[idx] = ((r + 128.0) / 255.0).clamp(0.0, 1.0);
            rgb[idx + 1] = ((g + 128.0) / 255.0).clamp(0.0, 1.0);
            rgb[idx + 2] = ((b + 128.0) / 255.0).clamp(0.0, 1.0);
        }
    }

    #[cfg(not(feature = "simd"))]
    {
        for i in 0..num_pixels {
            let y = y_plane[i];
            let cb = cb_plane[i];
            let cr = cr_plane[i];

            let r = CR_TO_R.mul_add(cr, y);
            let g = CB_TO_G.mul_add(cb, CR_TO_G.mul_add(cr, y));
            let b = CB_TO_B.mul_add(cb, y);

            let idx = i * 3;
            rgb[idx] = ((r + 128.0) / 255.0).clamp(0.0, 1.0);
            rgb[idx + 1] = ((g + 128.0) / 255.0).clamp(0.0, 1.0);
            rgb[idx + 2] = ((b + 128.0) / 255.0).clamp(0.0, 1.0);
        }
    }
}

/// Batch grayscale to RGB conversion for f32 to u8.
#[inline(never)]
pub fn gray_f32_to_rgb_u8(y_plane: &[f32], rgb: &mut [u8]) {
    debug_assert_eq!(rgb.len(), y_plane.len() * 3);

    let num_pixels = y_plane.len();

    #[cfg(feature = "simd")]
    {
        let offset = f32x8::splat(128.0);
        let zero = f32x8::splat(0.0);
        let max_val = f32x8::splat(255.0);

        let chunks = num_pixels / 8;
        for chunk in 0..chunks {
            let base = chunk * 8;
            let y = f32x8::from(&y_plane[base..base + 8]);

            let val = (y + offset).max(zero).min(max_val);
            let arr: [f32; 8] = val.into();

            for j in 0..8 {
                let idx = (base + j) * 3;
                let v = arr[j] as u8;
                rgb[idx] = v;
                rgb[idx + 1] = v;
                rgb[idx + 2] = v;
            }
        }

        // Remainder
        for i in (chunks * 8)..num_pixels {
            let val = (y_plane[i] + 128.0).clamp(0.0, 255.0) as u8;
            let idx = i * 3;
            rgb[idx] = val;
            rgb[idx + 1] = val;
            rgb[idx + 2] = val;
        }
    }

    #[cfg(not(feature = "simd"))]
    {
        for (i, &y) in y_plane.iter().enumerate() {
            let val = (y + 128.0).clamp(0.0, 255.0) as u8;
            let idx = i * 3;
            rgb[idx] = val;
            rgb[idx + 1] = val;
            rgb[idx + 2] = val;
        }
    }
}

/// Batch grayscale to RGB conversion for f32 to f32.
#[inline(never)]
pub fn gray_f32_to_rgb_f32(y_plane: &[f32], rgb: &mut [f32]) {
    debug_assert_eq!(rgb.len(), y_plane.len() * 3);

    let num_pixels = y_plane.len();

    #[cfg(feature = "simd")]
    {
        let offset = f32x8::splat(128.0);
        let scale = f32x8::splat(1.0 / 255.0);
        let zero = f32x8::splat(0.0);
        let one = f32x8::splat(1.0);

        let chunks = num_pixels / 8;
        for chunk in 0..chunks {
            let base = chunk * 8;
            let y = f32x8::from(&y_plane[base..base + 8]);

            let val = ((y + offset) * scale).max(zero).min(one);
            let arr: [f32; 8] = val.into();

            for j in 0..8 {
                let idx = (base + j) * 3;
                rgb[idx] = arr[j];
                rgb[idx + 1] = arr[j];
                rgb[idx + 2] = arr[j];
            }
        }

        // Remainder
        for i in (chunks * 8)..num_pixels {
            let val = ((y_plane[i] + 128.0) / 255.0).clamp(0.0, 1.0);
            let idx = i * 3;
            rgb[idx] = val;
            rgb[idx + 1] = val;
            rgb[idx + 2] = val;
        }
    }

    #[cfg(not(feature = "simd"))]
    {
        for (i, &y) in y_plane.iter().enumerate() {
            let val = ((y + 128.0) / 255.0).clamp(0.0, 1.0);
            let idx = i * 3;
            rgb[idx] = val;
            rgb[idx + 1] = val;
            rgb[idx + 2] = val;
        }
    }
}

/// Batch level shift for grayscale f32 to u8.
#[inline(never)]
pub fn gray_f32_to_gray_u8(y_plane: &[f32], output: &mut [u8]) {
    debug_assert_eq!(y_plane.len(), output.len());

    let num_pixels = y_plane.len();

    #[cfg(feature = "simd")]
    {
        let offset = f32x8::splat(128.0);
        let zero = f32x8::splat(0.0);
        let max_val = f32x8::splat(255.0);

        let chunks = num_pixels / 8;
        for chunk in 0..chunks {
            let base = chunk * 8;
            let y = f32x8::from(&y_plane[base..base + 8]);

            let val = (y + offset).max(zero).min(max_val);
            let arr: [f32; 8] = val.into();

            for j in 0..8 {
                output[base + j] = arr[j] as u8;
            }
        }

        // Remainder
        for i in (chunks * 8)..num_pixels {
            output[i] = (y_plane[i] + 128.0).clamp(0.0, 255.0) as u8;
        }
    }

    #[cfg(not(feature = "simd"))]
    {
        for (y, out) in y_plane.iter().zip(output.iter_mut()) {
            *out = (*y + 128.0).clamp(0.0, 255.0) as u8;
        }
    }
}

/// Batch level shift for grayscale f32 to f32 (0.0-1.0).
#[inline(never)]
pub fn gray_f32_to_gray_f32(y_plane: &[f32], output: &mut [f32]) {
    debug_assert_eq!(y_plane.len(), output.len());

    let num_pixels = y_plane.len();

    #[cfg(feature = "simd")]
    {
        let offset = f32x8::splat(128.0);
        let scale = f32x8::splat(1.0 / 255.0);
        let zero = f32x8::splat(0.0);
        let one = f32x8::splat(1.0);

        let chunks = num_pixels / 8;
        for chunk in 0..chunks {
            let base = chunk * 8;
            let y = f32x8::from(&y_plane[base..base + 8]);

            let val = ((y + offset) * scale).max(zero).min(one);
            let arr: [f32; 8] = val.into();
            output[base..base + 8].copy_from_slice(&arr);
        }

        // Remainder
        for i in (chunks * 8)..num_pixels {
            output[i] = ((y_plane[i] + 128.0) / 255.0).clamp(0.0, 1.0);
        }
    }

    #[cfg(not(feature = "simd"))]
    {
        for (y, out) in y_plane.iter().zip(output.iter_mut()) {
            *out = ((*y + 128.0) / 255.0).clamp(0.0, 1.0);
        }
    }
}

/// Converts BGR to RGB.
#[inline]
pub fn bgr_to_rgb(bgr: &[u8; 3]) -> [u8; 3] {
    [bgr[2], bgr[1], bgr[0]]
}

/// Converts BGRA to RGBA.
#[inline]
pub fn bgra_to_rgba(bgra: &[u8; 4]) -> [u8; 4] {
    [bgra[2], bgra[1], bgra[0], bgra[3]]
}

/// Converts CMYK to RGB.
///
/// Note: This is a simple conversion without ICC profile.
/// For accurate CMYK conversion, use the CMS feature.
#[inline]
#[must_use]
pub fn cmyk_to_rgb(c: u8, m: u8, y: u8, k: u8) -> (u8, u8, u8) {
    // CMYK values are often inverted in JPEG (0 = full ink)
    let c = c as f32 / 255.0;
    let m = m as f32 / 255.0;
    let y = y as f32 / 255.0;
    let k = k as f32 / 255.0;

    let r = 255.0 * (1.0 - c) * (1.0 - k);
    let g = 255.0 * (1.0 - m) * (1.0 - k);
    let b = 255.0 * (1.0 - y) * (1.0 - k);

    (
        r.round().clamp(0.0, 255.0) as u8,
        g.round().clamp(0.0, 255.0) as u8,
        b.round().clamp(0.0, 255.0) as u8,
    )
}

/// Converts RGB to CMYK.
#[inline]
#[must_use]
pub fn rgb_to_cmyk(r: u8, g: u8, b: u8) -> (u8, u8, u8, u8) {
    let r = r as f32 / 255.0;
    let g = g as f32 / 255.0;
    let b = b as f32 / 255.0;

    let k = 1.0 - r.max(g).max(b);

    if k >= 1.0 {
        return (0, 0, 0, 255);
    }

    let c = (1.0 - r - k) / (1.0 - k);
    let m = (1.0 - g - k) / (1.0 - k);
    let y = (1.0 - b - k) / (1.0 - k);

    (
        (c * 255.0).round() as u8,
        (m * 255.0).round() as u8,
        (y * 255.0).round() as u8,
        (k * 255.0).round() as u8,
    )
}

/// Extracts a single channel from a pixel buffer.
///
/// # Errors
///
/// Returns an error if memory allocation fails.
pub fn extract_channel(data: &[u8], format: PixelFormat, channel: usize) -> Result<Vec<u8>> {
    let bpp = format.bytes_per_pixel();
    let num_pixels = data.len() / bpp;
    let mut result = try_alloc_zeroed(num_pixels, "channel extraction buffer")?;

    for i in 0..num_pixels {
        result[i] = data[i * bpp + channel];
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgb_ycbcr_roundtrip() {
        // Test with various colors
        let test_colors = [
            (0u8, 0u8, 0u8),       // Black
            (255u8, 255u8, 255u8), // White
            (255u8, 0u8, 0u8),     // Red
            (0u8, 255u8, 0u8),     // Green
            (0u8, 0u8, 255u8),     // Blue
            (128u8, 128u8, 128u8), // Gray
        ];

        for (r, g, b) in test_colors {
            let (y, cb, cr) = rgb_to_ycbcr(r, g, b);
            let (r2, g2, b2) = ycbcr_to_rgb(y, cb, cr);

            // Allow small rounding errors
            assert!(
                (r as i16 - r2 as i16).abs() <= 1,
                "R mismatch for ({},{},{})",
                r,
                g,
                b
            );
            assert!(
                (g as i16 - g2 as i16).abs() <= 1,
                "G mismatch for ({},{},{})",
                r,
                g,
                b
            );
            assert!(
                (b as i16 - b2 as i16).abs() <= 1,
                "B mismatch for ({},{},{})",
                r,
                g,
                b
            );
        }
    }

    #[test]
    fn test_gray_ycbcr() {
        // Gray values should have Cb=Cr=128
        for gray in [0u8, 64, 128, 192, 255] {
            let (y, cb, cr) = rgb_to_ycbcr(gray, gray, gray);
            assert_eq!(y, gray);
            assert!((cb as i16 - 128).abs() <= 1);
            assert!((cr as i16 - 128).abs() <= 1);
        }
    }

    #[test]
    fn test_cmyk_rgb_roundtrip() {
        let (r, g, b) = cmyk_to_rgb(0, 0, 0, 0);
        assert_eq!((r, g, b), (255, 255, 255)); // White

        let (r, g, b) = cmyk_to_rgb(255, 255, 255, 255);
        assert_eq!((r, g, b), (0, 0, 0)); // Black
    }

    #[test]
    fn test_bgr_conversion() {
        assert_eq!(bgr_to_rgb(&[1, 2, 3]), [3, 2, 1]);
        assert_eq!(bgra_to_rgba(&[1, 2, 3, 4]), [3, 2, 1, 4]);
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_simd_rgb_to_ycbcr_matches_scalar() {
        // Test that SIMD version produces same results as scalar
        let test_colors = [
            (0u8, 0u8, 0u8),
            (255u8, 255u8, 255u8),
            (255u8, 0u8, 0u8),
            (0u8, 255u8, 0u8),
            (0u8, 0u8, 255u8),
            (128u8, 128u8, 128u8),
            (100u8, 150u8, 200u8),
            (33u8, 66u8, 99u8),
        ];

        // Test 4 pixels at a time
        for chunk in test_colors.chunks(4) {
            if chunk.len() < 4 {
                continue;
            }

            let r = [chunk[0].0, chunk[1].0, chunk[2].0, chunk[3].0];
            let g = [chunk[0].1, chunk[1].1, chunk[2].1, chunk[3].1];
            let b = [chunk[0].2, chunk[1].2, chunk[2].2, chunk[3].2];

            let (y_simd, cb_simd, cr_simd) = simd::rgb_to_ycbcr_x4(r, g, b);

            for i in 0..4 {
                let (y_scalar, cb_scalar, cr_scalar) = rgb_to_ycbcr(r[i], g[i], b[i]);
                assert_eq!(y_simd[i], y_scalar, "Y mismatch at {}", i);
                assert_eq!(cb_simd[i], cb_scalar, "Cb mismatch at {}", i);
                assert_eq!(cr_simd[i], cr_scalar, "Cr mismatch at {}", i);
            }
        }
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_simd_ycbcr_to_rgb_matches_scalar() {
        // Test that SIMD version produces same results as scalar
        let test_ycbcr = [
            (0u8, 128u8, 128u8),   // Black
            (255u8, 128u8, 128u8), // White
            (76u8, 85u8, 255u8),   // Red
            (150u8, 44u8, 21u8),   // Green
            (29u8, 255u8, 107u8),  // Blue
            (128u8, 128u8, 128u8), // Gray
        ];

        // Test 4 pixels at a time
        for chunk in test_ycbcr.chunks(4) {
            if chunk.len() < 4 {
                continue;
            }

            let y = [chunk[0].0, chunk[1].0, chunk[2].0, chunk[3].0];
            let cb = [chunk[0].1, chunk[1].1, chunk[2].1, chunk[3].1];
            let cr = [chunk[0].2, chunk[1].2, chunk[2].2, chunk[3].2];

            let (r_simd, g_simd, b_simd) = simd::ycbcr_to_rgb_x4(y, cb, cr);

            for i in 0..4 {
                let (r_scalar, g_scalar, b_scalar) = ycbcr_to_rgb(y[i], cb[i], cr[i]);
                assert_eq!(r_simd[i], r_scalar, "R mismatch at {}", i);
                assert_eq!(g_simd[i], g_scalar, "G mismatch at {}", i);
                assert_eq!(b_simd[i], b_scalar, "B mismatch at {}", i);
            }
        }
    }

    #[test]
    fn test_rgb_to_ycbcr_f32() {
        // Test f32 version matches u8 version
        let (y, cb, cr) = rgb_to_ycbcr_f32(255.0, 0.0, 0.0); // Red
        assert!((y - 76.0).abs() < 1.0);
        assert!((cb - 85.0).abs() < 1.0);
        assert!((cr - 255.0).abs() < 1.0);

        let (y, cb, cr) = rgb_to_ycbcr_f32(0.0, 255.0, 0.0); // Green
        assert!((y - 150.0).abs() < 1.0);

        let (y, cb, cr) = rgb_to_ycbcr_f32(0.0, 0.0, 255.0); // Blue
        assert!((y - 29.0).abs() < 1.0);
    }

    #[test]
    fn test_ycbcr_to_rgb_f32() {
        // Test f32 conversion
        let (r, g, b) = ycbcr_to_rgb_f32(128.0, 128.0, 128.0); // Gray
        assert!((r - 128.0).abs() < 1.0);
        assert!((g - 128.0).abs() < 1.0);
        assert!((b - 128.0).abs() < 1.0);
    }

    #[test]
    fn test_convert_rgb_to_ycbcr_buffer() {
        let mut buffer = [255, 0, 0, 0, 255, 0, 0, 0, 255]; // RGB: red, green, blue
        convert_rgb_to_ycbcr_buffer(&mut buffer);
        // After conversion, first pixel should have Y ~ 76 (red)
        assert!((buffer[0] as i16 - 76).abs() <= 1);
    }

    #[test]
    fn test_convert_ycbcr_to_rgb_buffer() {
        let mut buffer = [128, 128, 128, 128, 128, 128]; // Gray YCbCr
        convert_ycbcr_to_rgb_buffer(&mut buffer);
        // Should convert back to gray RGB
        assert!((buffer[0] as i16 - 128).abs() <= 1);
        assert!((buffer[1] as i16 - 128).abs() <= 1);
        assert!((buffer[2] as i16 - 128).abs() <= 1);
    }

    #[test]
    fn test_rgb_to_ycbcr_planes() {
        let rgb = vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 128, 128, 128]; // 4 pixels
        let (y, cb, cr) = rgb_to_ycbcr_planes(&rgb, 2, 2).unwrap();
        assert_eq!(y.len(), 4);
        assert_eq!(cb.len(), 4);
        assert_eq!(cr.len(), 4);
        // Red pixel should have Y ~ 76
        assert!((y[0] as i16 - 76).abs() <= 1);
    }

    #[test]
    fn test_ycbcr_planes_to_rgb() {
        let y = vec![128u8, 128, 128, 128];
        let cb = vec![128u8, 128, 128, 128];
        let cr = vec![128u8, 128, 128, 128];
        let rgb = ycbcr_planes_to_rgb(&y, &cb, &cr, 2, 2).unwrap();
        assert_eq!(rgb.len(), 12); // 4 pixels * 3 channels
                                   // All pixels should be gray
        for i in 0..4 {
            assert!((rgb[i * 3] as i16 - 128).abs() <= 1);
        }
    }

    #[test]
    fn test_ycbcr_planes_f32_to_rgb_u8() {
        // Create f32 planes (centered around 0 for IDCT output)
        let y = vec![0.0f32; 4];
        let cb = vec![0.0f32; 4];
        let cr = vec![0.0f32; 4];
        let mut rgb = vec![0u8; 12];
        ycbcr_planes_f32_to_rgb_u8(&y, &cb, &cr, &mut rgb);
        // All should be gray (128 after level shift)
        for i in 0..4 {
            assert_eq!(rgb[i * 3], 128);
            assert_eq!(rgb[i * 3 + 1], 128);
            assert_eq!(rgb[i * 3 + 2], 128);
        }
    }

    #[test]
    fn test_ycbcr_planes_f32_to_rgb_f32() {
        let y = vec![0.0f32; 4];
        let cb = vec![0.0f32; 4];
        let cr = vec![0.0f32; 4];
        let mut rgb = vec![0.0f32; 12];
        ycbcr_planes_f32_to_rgb_f32(&y, &cb, &cr, &mut rgb);
        // All should be ~0.5 (128/255)
        for i in 0..4 {
            assert!((rgb[i * 3] - 0.502).abs() < 0.01);
        }
    }

    #[test]
    fn test_gray_f32_to_rgb_u8() {
        let y = vec![0.0f32, 127.0, -128.0]; // 0+128=128, 127+128=255, -128+128=0
        let mut rgb = vec![0u8; 9];
        gray_f32_to_rgb_u8(&y, &mut rgb);
        assert_eq!(rgb[0], 128); // R
        assert_eq!(rgb[1], 128); // G
        assert_eq!(rgb[2], 128); // B
        assert_eq!(rgb[3], 255); // Second pixel R
        assert_eq!(rgb[6], 0); // Third pixel R
    }

    #[test]
    fn test_gray_f32_to_rgb_f32() {
        let y = vec![0.0f32; 2];
        let mut rgb = vec![0.0f32; 6];
        gray_f32_to_rgb_f32(&y, &mut rgb);
        // Should be ~0.5
        for v in &rgb {
            assert!((*v - 0.502).abs() < 0.01);
        }
    }

    #[test]
    fn test_gray_f32_to_gray_u8() {
        let y = vec![0.0f32, 127.0, -128.0];
        let mut output = vec![0u8; 3];
        gray_f32_to_gray_u8(&y, &mut output);
        assert_eq!(output[0], 128);
        assert_eq!(output[1], 255);
        assert_eq!(output[2], 0);
    }

    #[test]
    fn test_gray_f32_to_gray_f32() {
        let y = vec![0.0f32, 127.0];
        let mut output = vec![0.0f32; 2];
        gray_f32_to_gray_f32(&y, &mut output);
        assert!((output[0] - 0.502).abs() < 0.01);
        assert!((output[1] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_rgb_to_cmyk() {
        // White -> CMYK(0,0,0,0)
        let (c, m, y, k) = rgb_to_cmyk(255, 255, 255);
        assert_eq!((c, m, y, k), (0, 0, 0, 0));

        // Black -> CMYK(0,0,0,255)
        let (c, m, y, k) = rgb_to_cmyk(0, 0, 0);
        assert_eq!(k, 255);

        // Red -> Cyan=0
        let (c, _, _, _) = rgb_to_cmyk(255, 0, 0);
        assert_eq!(c, 0);
    }

    #[test]
    fn test_extract_channel() {
        let data = vec![10, 20, 30, 40, 50, 60]; // 2 RGB pixels
        let red = extract_channel(&data, PixelFormat::Rgb, 0).unwrap();
        assert_eq!(red, vec![10, 40]);
        let green = extract_channel(&data, PixelFormat::Rgb, 1).unwrap();
        assert_eq!(green, vec![20, 50]);
        let blue = extract_channel(&data, PixelFormat::Rgb, 2).unwrap();
        assert_eq!(blue, vec![30, 60]);
    }

    #[test]
    fn test_extract_channel_rgba() {
        let data = vec![10, 20, 30, 255, 40, 50, 60, 128]; // 2 RGBA pixels
        let alpha = extract_channel(&data, PixelFormat::Rgba, 3).unwrap();
        assert_eq!(alpha, vec![255, 128]);
    }
}
