//! Color space conversion functions.
//!
//! This module provides conversions between:
//! - RGB and YCbCr (BT.601 standard JPEG color space)
//! - RGB and CMYK
//! - Various pixel format conversions
//!
//! SIMD optimization via the `wide` crate is always enabled.

#![allow(dead_code)] // Multiple conversion variants for different pipelines

use crate::error::Result;
use crate::foundation::alloc::{checked_size, checked_size_2d, try_alloc_zeroed};
use crate::foundation::consts::{
    YCBCR_B_TO_CB, YCBCR_B_TO_CR, YCBCR_B_TO_Y, YCBCR_CB_TO_B, YCBCR_CB_TO_G, YCBCR_CB_TO_R,
    YCBCR_CR_TO_B, YCBCR_CR_TO_G, YCBCR_CR_TO_R, YCBCR_G_TO_CB, YCBCR_G_TO_CR, YCBCR_G_TO_Y,
    YCBCR_R_TO_CB, YCBCR_R_TO_CR, YCBCR_R_TO_Y, YCBCR_Y_TO_B, YCBCR_Y_TO_G, YCBCR_Y_TO_R,
};
use crate::types::PixelFormat;

use wide::{f32x4, f32x8};

#[cfg(target_arch = "x86_64")]
use archmage::{SimdToken, arcane};

#[cfg(target_arch = "x86_64")]
use safe_unaligned_simd::x86_64 as safe_simd;

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
    let cb = YCBCR_R_TO_CB.mul_add(
        rf,
        YCBCR_G_TO_CB.mul_add(gf, YCBCR_B_TO_CB.mul_add(bf, 128.0)),
    );

    // Cr = 128 + 0.5*R - 0.418688*G - 0.081312*B
    let cr = YCBCR_R_TO_CR.mul_add(
        rf,
        YCBCR_G_TO_CR.mul_add(gf, YCBCR_B_TO_CR.mul_add(bf, 128.0)),
    );

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

// SIMD-optimized color conversion (always available via `wide` crate)
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

/// Converts separate Y, Cb, Cr planes to RGB.
///
/// Uses SIMD optimization when the `simd` feature is enabled.
///
/// # Errors
///
/// Returns an error if memory allocation fails.
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
/// Dispatches to AVX2+FMA on x86_64, falls back to wide f32x8 (2× SSE2).
pub fn ycbcr_planes_f32_to_rgb_u8(
    y_plane: &[f32],
    cb_plane: &[f32],
    cr_plane: &[f32],
    rgb: &mut [u8],
) {
    #[cfg(target_arch = "x86_64")]
    if let Some(token) = archmage::X64V3Token::summon() {
        return mage_ycbcr_planes_f32_to_rgb_u8(token, y_plane, cb_plane, cr_plane, rgb);
    }

    ycbcr_planes_f32_to_rgb_u8_wide(y_plane, cb_plane, cr_plane, rgb);
}

/// AVX2+FMA implementation of YCbCr f32 planes → interleaved RGB u8.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn mage_ycbcr_planes_f32_to_rgb_u8(
    token: archmage::X64V3Token,
    y_plane: &[f32],
    cb_plane: &[f32],
    cr_plane: &[f32],
    rgb: &mut [u8],
) {
    use magetypes::simd::f32x8 as mf32x8;

    debug_assert_eq!(y_plane.len(), cb_plane.len());
    debug_assert_eq!(y_plane.len(), cr_plane.len());
    debug_assert_eq!(rgb.len(), y_plane.len() * 3);

    let num_pixels = y_plane.len();

    // BT.601 coefficients
    let cr_to_r = mf32x8::splat(token, 1.402);
    let cb_to_g = mf32x8::splat(token, -0.344136);
    let cr_to_g = mf32x8::splat(token, -0.714136);
    let cb_to_b = mf32x8::splat(token, 1.772);
    let offset = mf32x8::splat(token, 128.0);
    let zero = mf32x8::splat(token, 0.0);
    let max_val = mf32x8::splat(token, 255.0);

    let chunks = num_pixels / 8;

    for chunk in 0..chunks {
        let i = chunk * 8;

        // Load 8 values from each plane
        let y = mf32x8::from_array(token, <[f32; 8]>::try_from(&y_plane[i..i + 8]).unwrap());
        let cb = mf32x8::from_array(token, <[f32; 8]>::try_from(&cb_plane[i..i + 8]).unwrap());
        let cr = mf32x8::from_array(token, <[f32; 8]>::try_from(&cr_plane[i..i + 8]).unwrap());

        let y_off = y + offset;

        // YCbCr to RGB with real FMA (vfmadd instructions)
        let r = cr_to_r.mul_add(cr, y_off).max(zero).min(max_val);
        let g = cb_to_g
            .mul_add(cb, cr_to_g.mul_add(cr, y_off))
            .max(zero)
            .min(max_val);
        let b = cb_to_b.mul_add(cb, y_off).max(zero).min(max_val);

        // Extract to arrays for interleaved store
        let r_arr = r.to_array();
        let g_arr = g.to_array();
        let b_arr = b.to_array();

        // Store interleaved RGB
        let rgb_chunk = &mut rgb[i * 3..(i + 8) * 3];
        for j in 0..8 {
            rgb_chunk[j * 3] = r_arr[j] as u8;
            rgb_chunk[j * 3 + 1] = g_arr[j] as u8;
            rgb_chunk[j * 3 + 2] = b_arr[j] as u8;
        }
    }

    // Scalar remainder
    let start = chunks * 8;
    for i in start..num_pixels {
        let y = y_plane[i];
        let cb = cb_plane[i];
        let cr = cr_plane[i];

        let r = 1.402f32.mul_add(cr, y);
        let g = (-0.344136f32).mul_add(cb, (-0.714136f32).mul_add(cr, y));
        let b_val = 1.772f32.mul_add(cb, y);

        rgb[i * 3] = (r + 128.0).clamp(0.0, 255.0) as u8;
        rgb[i * 3 + 1] = (g + 128.0).clamp(0.0, 255.0) as u8;
        rgb[i * 3 + 2] = (b_val + 128.0).clamp(0.0, 255.0) as u8;
    }
}

/// Wide-based fallback for YCbCr f32 planes → interleaved RGB u8.
/// Uses wide::f32x8 (2× SSE2 on x86_64, portable on other targets).
fn ycbcr_planes_f32_to_rgb_u8_wide(
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

    let y_chunks = y_plane.chunks_exact(8);
    let cb_chunks = cb_plane.chunks_exact(8);
    let cr_chunks = cr_plane.chunks_exact(8);
    let rgb_chunks = rgb.chunks_exact_mut(24);

    let y_remainder = y_chunks.remainder();
    let cb_remainder = cb_chunks.remainder();
    let cr_remainder = cr_chunks.remainder();

    for (((y_chunk, cb_chunk), cr_chunk), rgb_chunk) in
        y_chunks.zip(cb_chunks).zip(cr_chunks).zip(rgb_chunks)
    {
        let y = f32x8::from(<[f32; 8]>::try_from(y_chunk).unwrap());
        let cb = f32x8::from(<[f32; 8]>::try_from(cb_chunk).unwrap());
        let cr = f32x8::from(<[f32; 8]>::try_from(cr_chunk).unwrap());

        let r = cr_to_r.mul_add(cr, y + offset).max(zero).min(max_val);
        let g = cb_to_g
            .mul_add(cb, cr_to_g.mul_add(cr, y + offset))
            .max(zero)
            .min(max_val);
        let b = cb_to_b.mul_add(cb, y + offset).max(zero).min(max_val);

        let r_arr: [f32; 8] = r.into();
        let g_arr: [f32; 8] = g.into();
        let b_arr: [f32; 8] = b.into();

        for j in 0..8 {
            rgb_chunk[j * 3] = r_arr[j] as u8;
            rgb_chunk[j * 3 + 1] = g_arr[j] as u8;
            rgb_chunk[j * 3 + 2] = b_arr[j] as u8;
        }
    }

    // Scalar remainder
    let chunks_processed = (num_pixels / 8) * 8;
    let rgb_start = chunks_processed * 3;
    for (i, ((y, cb), cr)) in y_remainder
        .iter()
        .zip(cb_remainder.iter())
        .zip(cr_remainder.iter())
        .enumerate()
    {
        let r = CR_TO_R.mul_add(*cr, *y);
        let g = CB_TO_G.mul_add(*cb, CR_TO_G.mul_add(*cr, *y));
        let b_val = CB_TO_B.mul_add(*cb, *y);

        let idx = rgb_start + i * 3;
        rgb[idx] = (r + 128.0).clamp(0.0, 255.0) as u8;
        rgb[idx + 1] = (g + 128.0).clamp(0.0, 255.0) as u8;
        rgb[idx + 2] = (b_val + 128.0).clamp(0.0, 255.0) as u8;
    }
}

/// Batch YCbCr to RGB conversion for f32 planes to f32 output.
///
/// Input: centered YCbCr (Y, Cb, Cr all centered around 0 from f32 IDCT).
/// Output: RGB normalized to approximately 0.0-1.0 range. Values may slightly
/// exceed [0, 1] due to YCbCr→RGB color matrix expansion — this is intentional
/// to preserve full precision. Callers should clamp only at final output if needed.
///
/// Dispatches to AVX2+FMA on x86_64, falls back to wide f32x8 (2× SSE2).
pub fn ycbcr_planes_f32_to_rgb_f32(
    y_plane: &[f32],
    cb_plane: &[f32],
    cr_plane: &[f32],
    rgb: &mut [f32],
) {
    #[cfg(target_arch = "x86_64")]
    if let Some(token) = archmage::X64V3Token::summon() {
        return mage_ycbcr_planes_f32_to_rgb_f32(token, y_plane, cb_plane, cr_plane, rgb);
    }

    ycbcr_planes_f32_to_rgb_f32_wide(y_plane, cb_plane, cr_plane, rgb);
}

/// AVX2+FMA implementation of YCbCr f32 planes → interleaved RGB f32.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn mage_ycbcr_planes_f32_to_rgb_f32(
    token: archmage::X64V3Token,
    y_plane: &[f32],
    cb_plane: &[f32],
    cr_plane: &[f32],
    rgb: &mut [f32],
) {
    use magetypes::simd::f32x8 as mf32x8;

    debug_assert_eq!(y_plane.len(), cb_plane.len());
    debug_assert_eq!(y_plane.len(), cr_plane.len());
    debug_assert_eq!(rgb.len(), y_plane.len() * 3);

    let num_pixels = y_plane.len();

    let cr_to_r = mf32x8::splat(token, 1.402);
    let cb_to_g = mf32x8::splat(token, -0.344136);
    let cr_to_g = mf32x8::splat(token, -0.714136);
    let cb_to_b = mf32x8::splat(token, 1.772);
    let offset = mf32x8::splat(token, 128.0);
    let scale = mf32x8::splat(token, 1.0 / 255.0);

    let chunks = num_pixels / 8;
    for chunk in 0..chunks {
        let base = chunk * 8;

        let y = mf32x8::from_array(
            token,
            <[f32; 8]>::try_from(&y_plane[base..base + 8]).unwrap(),
        );
        let cb = mf32x8::from_array(
            token,
            <[f32; 8]>::try_from(&cb_plane[base..base + 8]).unwrap(),
        );
        let cr = mf32x8::from_array(
            token,
            <[f32; 8]>::try_from(&cr_plane[base..base + 8]).unwrap(),
        );

        let y_off = y + offset;

        // YCbCr to RGB with real FMA, level shift, normalize — no clamping
        let r = cr_to_r.mul_add(cr, y_off) * scale;
        let g = cb_to_g.mul_add(cb, cr_to_g.mul_add(cr, y_off)) * scale;
        let b = cb_to_b.mul_add(cb, y_off) * scale;

        let r_arr = r.to_array();
        let g_arr = g.to_array();
        let b_arr = b.to_array();

        for j in 0..8 {
            let idx = (base + j) * 3;
            rgb[idx] = r_arr[j];
            rgb[idx + 1] = g_arr[j];
            rgb[idx + 2] = b_arr[j];
        }
    }

    // Scalar remainder
    for i in (chunks * 8)..num_pixels {
        let y = y_plane[i];
        let cb = cb_plane[i];
        let cr = cr_plane[i];

        let r = 1.402f32.mul_add(cr, y);
        let g = (-0.344136f32).mul_add(cb, (-0.714136f32).mul_add(cr, y));
        let b = 1.772f32.mul_add(cb, y);

        let idx = i * 3;
        rgb[idx] = (r + 128.0) / 255.0;
        rgb[idx + 1] = (g + 128.0) / 255.0;
        rgb[idx + 2] = (b + 128.0) / 255.0;
    }
}

/// Wide-based fallback for YCbCr f32 planes → interleaved RGB f32.
fn ycbcr_planes_f32_to_rgb_f32_wide(
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

    let cr_to_r = f32x8::splat(CR_TO_R);
    let cb_to_g = f32x8::splat(CB_TO_G);
    let cr_to_g = f32x8::splat(CR_TO_G);
    let cb_to_b = f32x8::splat(CB_TO_B);
    let offset = f32x8::splat(128.0);
    let scale = f32x8::splat(1.0 / 255.0);

    let chunks = num_pixels / 8;
    for chunk in 0..chunks {
        let base = chunk * 8;
        let y = f32x8::from(<[f32; 8]>::try_from(&y_plane[base..base + 8]).unwrap());
        let cb = f32x8::from(<[f32; 8]>::try_from(&cb_plane[base..base + 8]).unwrap());
        let cr = f32x8::from(<[f32; 8]>::try_from(&cr_plane[base..base + 8]).unwrap());

        let r = cr_to_r.mul_add(cr, y + offset) * scale;
        let g = cb_to_g.mul_add(cb, cr_to_g.mul_add(cr, y + offset)) * scale;
        let b = cb_to_b.mul_add(cb, y + offset) * scale;

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
        let y = y_plane[i];
        let cb = cb_plane[i];
        let cr = cr_plane[i];

        let r = CR_TO_R.mul_add(cr, y);
        let g = CB_TO_G.mul_add(cb, CR_TO_G.mul_add(cr, y));
        let b = CB_TO_B.mul_add(cb, y);

        let idx = i * 3;
        rgb[idx] = (r + 128.0) / 255.0;
        rgb[idx + 1] = (g + 128.0) / 255.0;
        rgb[idx + 2] = (b + 128.0) / 255.0;
    }
}

/// Batch grayscale to RGB conversion for f32 to u8.
///
/// Dispatches to AVX2 on x86_64, falls back to wide f32x8 (2× SSE2).
pub fn gray_f32_to_rgb_u8(y_plane: &[f32], rgb: &mut [u8]) {
    #[cfg(target_arch = "x86_64")]
    if let Some(token) = archmage::X64V3Token::summon() {
        return mage_gray_f32_to_rgb_u8(token, y_plane, rgb);
    }

    gray_f32_to_rgb_u8_wide(y_plane, rgb);
}

/// AVX2 implementation of grayscale f32 → interleaved RGB u8.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn mage_gray_f32_to_rgb_u8(token: archmage::X64V3Token, y_plane: &[f32], rgb: &mut [u8]) {
    use magetypes::simd::f32x8 as mf32x8;

    debug_assert_eq!(rgb.len(), y_plane.len() * 3);

    let num_pixels = y_plane.len();
    let offset = mf32x8::splat(token, 128.0);
    let zero = mf32x8::splat(token, 0.0);
    let max_val = mf32x8::splat(token, 255.0);

    let chunks = num_pixels / 8;
    for chunk in 0..chunks {
        let base = chunk * 8;
        let y = mf32x8::from_array(
            token,
            <[f32; 8]>::try_from(&y_plane[base..base + 8]).unwrap(),
        );

        let val = (y + offset).max(zero).min(max_val);
        let arr = val.to_array();

        for j in 0..8 {
            let idx = (base + j) * 3;
            let v = arr[j] as u8;
            rgb[idx] = v;
            rgb[idx + 1] = v;
            rgb[idx + 2] = v;
        }
    }

    for i in (chunks * 8)..num_pixels {
        let val = (y_plane[i] + 128.0).clamp(0.0, 255.0) as u8;
        let idx = i * 3;
        rgb[idx] = val;
        rgb[idx + 1] = val;
        rgb[idx + 2] = val;
    }
}

/// Wide-based fallback for grayscale f32 → interleaved RGB u8.
fn gray_f32_to_rgb_u8_wide(y_plane: &[f32], rgb: &mut [u8]) {
    debug_assert_eq!(rgb.len(), y_plane.len() * 3);

    let num_pixels = y_plane.len();
    let offset = f32x8::splat(128.0);
    let zero = f32x8::splat(0.0);
    let max_val = f32x8::splat(255.0);

    let chunks = num_pixels / 8;
    for chunk in 0..chunks {
        let base = chunk * 8;
        let y = f32x8::from(<[f32; 8]>::try_from(&y_plane[base..base + 8]).unwrap());

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

    for i in (chunks * 8)..num_pixels {
        let val = (y_plane[i] + 128.0).clamp(0.0, 255.0) as u8;
        let idx = i * 3;
        rgb[idx] = val;
        rgb[idx + 1] = val;
        rgb[idx + 2] = val;
    }
}

/// Batch grayscale to RGB conversion for f32 to f32.
///
/// Input: centered grayscale (Y centered around 0 from f32 IDCT).
/// Output: normalized to approximately 0.0-1.0 range without clamping.
///
/// Dispatches to AVX2 on x86_64, falls back to wide f32x8 (2× SSE2).
pub fn gray_f32_to_rgb_f32(y_plane: &[f32], rgb: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if let Some(token) = archmage::X64V3Token::summon() {
        return mage_gray_f32_to_rgb_f32(token, y_plane, rgb);
    }

    gray_f32_to_rgb_f32_wide(y_plane, rgb);
}

/// AVX2 implementation of grayscale f32 → interleaved RGB f32.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn mage_gray_f32_to_rgb_f32(token: archmage::X64V3Token, y_plane: &[f32], rgb: &mut [f32]) {
    use magetypes::simd::f32x8 as mf32x8;

    debug_assert_eq!(rgb.len(), y_plane.len() * 3);

    let num_pixels = y_plane.len();
    let offset = mf32x8::splat(token, 128.0);
    let scale = mf32x8::splat(token, 1.0 / 255.0);

    let chunks = num_pixels / 8;
    for chunk in 0..chunks {
        let base = chunk * 8;
        let y = mf32x8::from_array(
            token,
            <[f32; 8]>::try_from(&y_plane[base..base + 8]).unwrap(),
        );

        let val = (y + offset) * scale;
        let arr = val.to_array();

        for j in 0..8 {
            let idx = (base + j) * 3;
            rgb[idx] = arr[j];
            rgb[idx + 1] = arr[j];
            rgb[idx + 2] = arr[j];
        }
    }

    for i in (chunks * 8)..num_pixels {
        let val = (y_plane[i] + 128.0) / 255.0;
        let idx = i * 3;
        rgb[idx] = val;
        rgb[idx + 1] = val;
        rgb[idx + 2] = val;
    }
}

/// Wide-based fallback for grayscale f32 → interleaved RGB f32.
fn gray_f32_to_rgb_f32_wide(y_plane: &[f32], rgb: &mut [f32]) {
    debug_assert_eq!(rgb.len(), y_plane.len() * 3);

    let num_pixels = y_plane.len();
    let offset = f32x8::splat(128.0);
    let scale = f32x8::splat(1.0 / 255.0);

    let chunks = num_pixels / 8;
    for chunk in 0..chunks {
        let base = chunk * 8;
        let y = f32x8::from(<[f32; 8]>::try_from(&y_plane[base..base + 8]).unwrap());

        let val = (y + offset) * scale;
        let arr: [f32; 8] = val.into();

        for j in 0..8 {
            let idx = (base + j) * 3;
            rgb[idx] = arr[j];
            rgb[idx + 1] = arr[j];
            rgb[idx + 2] = arr[j];
        }
    }

    for i in (chunks * 8)..num_pixels {
        let val = (y_plane[i] + 128.0) / 255.0;
        let idx = i * 3;
        rgb[idx] = val;
        rgb[idx + 1] = val;
        rgb[idx + 2] = val;
    }
}

/// Batch level shift for grayscale f32 to u8.
///
/// Dispatches to AVX2 on x86_64, falls back to wide f32x8.
pub fn gray_f32_to_gray_u8(y_plane: &[f32], output: &mut [u8]) {
    #[cfg(target_arch = "x86_64")]
    if let Some(token) = archmage::X64V3Token::summon() {
        return mage_gray_f32_to_gray_u8(token, y_plane, output);
    }

    gray_f32_to_gray_u8_wide(y_plane, output);
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn mage_gray_f32_to_gray_u8(token: archmage::X64V3Token, y_plane: &[f32], output: &mut [u8]) {
    use magetypes::simd::f32x8 as mf32x8;

    debug_assert_eq!(y_plane.len(), output.len());

    let num_pixels = y_plane.len();
    let offset = mf32x8::splat(token, 128.0);
    let zero = mf32x8::splat(token, 0.0);
    let max_val = mf32x8::splat(token, 255.0);

    let chunks = num_pixels / 8;
    for chunk in 0..chunks {
        let base = chunk * 8;
        let y = mf32x8::from_array(
            token,
            <[f32; 8]>::try_from(&y_plane[base..base + 8]).unwrap(),
        );

        let val = (y + offset).max(zero).min(max_val);
        let arr = val.to_array();

        for j in 0..8 {
            output[base + j] = arr[j] as u8;
        }
    }

    for i in (chunks * 8)..num_pixels {
        output[i] = (y_plane[i] + 128.0).clamp(0.0, 255.0) as u8;
    }
}

fn gray_f32_to_gray_u8_wide(y_plane: &[f32], output: &mut [u8]) {
    debug_assert_eq!(y_plane.len(), output.len());

    let num_pixels = y_plane.len();
    let offset = f32x8::splat(128.0);
    let zero = f32x8::splat(0.0);
    let max_val = f32x8::splat(255.0);

    let chunks = num_pixels / 8;
    for chunk in 0..chunks {
        let base = chunk * 8;
        let y = f32x8::from(<[f32; 8]>::try_from(&y_plane[base..base + 8]).unwrap());

        let val = (y + offset).max(zero).min(max_val);
        let arr: [f32; 8] = val.into();

        for j in 0..8 {
            output[base + j] = arr[j] as u8;
        }
    }

    for i in (chunks * 8)..num_pixels {
        output[i] = (y_plane[i] + 128.0).clamp(0.0, 255.0) as u8;
    }
}

/// Batch level shift for grayscale f32 to f32 (approximately 0.0-1.0).
///
/// Input: centered grayscale (Y centered around 0 from f32 IDCT).
/// Output: normalized without clamping to preserve full precision.
///
/// Dispatches to AVX2 on x86_64, falls back to wide f32x8.
pub fn gray_f32_to_gray_f32(y_plane: &[f32], output: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if let Some(token) = archmage::X64V3Token::summon() {
        return mage_gray_f32_to_gray_f32(token, y_plane, output);
    }

    gray_f32_to_gray_f32_wide(y_plane, output);
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn mage_gray_f32_to_gray_f32(token: archmage::X64V3Token, y_plane: &[f32], output: &mut [f32]) {
    use magetypes::simd::f32x8 as mf32x8;

    debug_assert_eq!(y_plane.len(), output.len());

    let num_pixels = y_plane.len();
    let offset = mf32x8::splat(token, 128.0);
    let scale = mf32x8::splat(token, 1.0 / 255.0);

    let chunks = num_pixels / 8;
    for chunk in 0..chunks {
        let base = chunk * 8;
        let y = mf32x8::from_array(
            token,
            <[f32; 8]>::try_from(&y_plane[base..base + 8]).unwrap(),
        );

        let val = (y + offset) * scale;
        let arr = val.to_array();
        output[base..base + 8].copy_from_slice(&arr);
    }

    for i in (chunks * 8)..num_pixels {
        output[i] = (y_plane[i] + 128.0) / 255.0;
    }
}

fn gray_f32_to_gray_f32_wide(y_plane: &[f32], output: &mut [f32]) {
    debug_assert_eq!(y_plane.len(), output.len());

    let num_pixels = y_plane.len();
    let offset = f32x8::splat(128.0);
    let scale = f32x8::splat(1.0 / 255.0);

    let chunks = num_pixels / 8;
    for chunk in 0..chunks {
        let base = chunk * 8;
        let y = f32x8::from(<[f32; 8]>::try_from(&y_plane[base..base + 8]).unwrap());

        let val = (y + offset) * scale;
        let arr: [f32; 8] = val.into();
        output[base..base + 8].copy_from_slice(&arr);
    }

    for i in (chunks * 8)..num_pixels {
        output[i] = (y_plane[i] + 128.0) / 255.0;
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

/// Swaps R and B channels in-place for a packed RGB/BGR u8 buffer.
///
/// The buffer length must be a multiple of 3.
#[cfg(feature = "decoder")]
pub fn rgb_u8_swap_rb_inplace(data: &mut [u8]) {
    debug_assert_eq!(data.len() % 3, 0);
    for pixel in data.chunks_exact_mut(3) {
        pixel.swap(0, 2);
    }
}

/// Converts packed RGB u8 to packed RGBA u8 (alpha = 255).
///
/// `src.len()` must be a multiple of 3 and `dst.len() >= src.len() / 3 * 4`.
#[cfg(feature = "decoder")]
pub fn rgb_u8_to_rgba_u8(src: &[u8], dst: &mut [u8]) {
    debug_assert_eq!(src.len() % 3, 0);
    let npixels = src.len() / 3;
    debug_assert!(dst.len() >= npixels * 4);
    for (s, d) in src.chunks_exact(3).zip(dst.chunks_exact_mut(4)) {
        d[0] = s[0];
        d[1] = s[1];
        d[2] = s[2];
        d[3] = 255;
    }
}

/// Converts packed RGB u8 to packed BGRA u8 (alpha = 255, R/B swapped).
///
/// `src.len()` must be a multiple of 3 and `dst.len() >= src.len() / 3 * 4`.
#[cfg(feature = "decoder")]
pub fn rgb_u8_to_bgra_u8(src: &[u8], dst: &mut [u8]) {
    debug_assert_eq!(src.len() % 3, 0);
    let npixels = src.len() / 3;
    debug_assert!(dst.len() >= npixels * 4);
    for (s, d) in src.chunks_exact(3).zip(dst.chunks_exact_mut(4)) {
        d[0] = s[2]; // B
        d[1] = s[1]; // G
        d[2] = s[0]; // R
        d[3] = 255;
    }
}

/// Converts packed RGB u8 to packed BGRX u8 (pad = 255, R/B swapped).
///
/// Identical to [`rgb_u8_to_bgra_u8`] — the pad byte is set to 255.
#[cfg(feature = "decoder")]
#[inline]
pub fn rgb_u8_to_bgrx_u8(src: &[u8], dst: &mut [u8]) {
    rgb_u8_to_bgra_u8(src, dst);
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

/// Converts Adobe CMYK to RGB.
///
/// Adobe JPEG CMYK stores values inverted: 0 = full ink, 255 = no ink.
/// This function handles the inversion automatically.
#[inline]
#[must_use]
pub fn cmyk_adobe_to_rgb(c: u8, m: u8, y: u8, k: u8) -> (u8, u8, u8) {
    // Adobe stores inverted: 0 = full ink, 255 = no ink
    // The formula with inverted values becomes:
    // R = C * K / 255, G = M * K / 255, B = Y * K / 255
    // (This is equivalent to: R = (255-C') * (255-K') / 255 with non-inverted C', K')
    let c32 = c as u32;
    let m32 = m as u32;
    let y32 = y as u32;
    let k32 = k as u32;

    // Compute R = C * K / 255, etc.
    // Using integer math with rounding: (a * b + 127) / 255
    let r = ((c32 * k32 + 127) / 255) as u8;
    let g = ((m32 * k32 + 127) / 255) as u8;
    let b = ((y32 * k32 + 127) / 255) as u8;

    (r, g, b)
}

/// Converts YCCK to RGB.
///
/// YCCK stores YCbCr (representing CMY values directly) plus K (Adobe-inverted).
/// The YCbCr→RGB gives CMY values where 255=full ink (subtractive).
/// K is stored inverted: K_adobe=255 means no black, K_adobe=0 means full black.
///
/// Formula: R = (255 - C) * K_adobe / 255
#[inline]
#[must_use]
pub fn ycck_to_rgb(y: u8, cb: u8, cr: u8, k: u8) -> (u8, u8, u8) {
    // Convert YCbCr to RGB, which gives us the CMY values directly
    // (where 255 = full ink in subtractive model)
    let (c, m, yy) = ycbcr_to_rgb(y, cb, cr);

    // Convert CMY + K (Adobe-inverted) to RGB
    // R = (255 - C) * K_adobe / 255
    // G = (255 - M) * K_adobe / 255
    // B = (255 - Y) * K_adobe / 255
    let k32 = k as u32;
    let r = (((255 - c as u32) * k32 + 127) / 255) as u8;
    let g = (((255 - m as u32) * k32 + 127) / 255) as u8;
    let b = (((255 - yy as u32) * k32 + 127) / 255) as u8;

    (r, g, b)
}

/// Batch convert CMYK planes (Adobe format) to interleaved RGB.
///
/// Each plane contains values in Adobe inverted format (0 = full ink).
/// Output is interleaved RGB bytes.
pub fn cmyk_planes_to_rgb_u8(
    c_plane: &[f32],
    m_plane: &[f32],
    y_plane: &[f32],
    k_plane: &[f32],
    rgb: &mut [u8],
) {
    debug_assert_eq!(c_plane.len(), m_plane.len());
    debug_assert_eq!(c_plane.len(), y_plane.len());
    debug_assert_eq!(c_plane.len(), k_plane.len());
    debug_assert_eq!(rgb.len(), c_plane.len() * 3);

    for i in 0..c_plane.len() {
        // Level shift and clamp to 0-255 range
        let c = (c_plane[i] + 128.0).round().clamp(0.0, 255.0) as u8;
        let m = (m_plane[i] + 128.0).round().clamp(0.0, 255.0) as u8;
        let y = (y_plane[i] + 128.0).round().clamp(0.0, 255.0) as u8;
        let k = (k_plane[i] + 128.0).round().clamp(0.0, 255.0) as u8;

        let (r, g, b) = cmyk_adobe_to_rgb(c, m, y, k);
        rgb[i * 3] = r;
        rgb[i * 3 + 1] = g;
        rgb[i * 3 + 2] = b;
    }
}

/// Batch convert YCCK planes to interleaved RGB.
///
/// Takes Y, Cb, Cr, K planes (f32, centered at 0) and outputs RGB.
pub fn ycck_planes_to_rgb_u8(
    y_plane: &[f32],
    cb_plane: &[f32],
    cr_plane: &[f32],
    k_plane: &[f32],
    rgb: &mut [u8],
) {
    debug_assert_eq!(y_plane.len(), cb_plane.len());
    debug_assert_eq!(y_plane.len(), cr_plane.len());
    debug_assert_eq!(y_plane.len(), k_plane.len());
    debug_assert_eq!(rgb.len(), y_plane.len() * 3);

    for i in 0..y_plane.len() {
        // Level shift and clamp
        let y = (y_plane[i] + 128.0).round().clamp(0.0, 255.0) as u8;
        let cb = (cb_plane[i] + 128.0).round().clamp(0.0, 255.0) as u8;
        let cr = (cr_plane[i] + 128.0).round().clamp(0.0, 255.0) as u8;
        let k = (k_plane[i] + 128.0).round().clamp(0.0, 255.0) as u8;

        let (r, g, b) = ycck_to_rgb(y, cb, cr, k);
        rgb[i * 3] = r;
        rgb[i * 3 + 1] = g;
        rgb[i * 3 + 2] = b;
    }
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

// =============================================================================
// Integer color conversion for fast decode path
// =============================================================================

// Fixed-point coefficients (14-bit precision), matching zune-jpeg
// These are the BT.601 coefficients scaled by 16384 (1 << 14)
const Y_CF_INT: i32 = 16384; // 1.0 << 14
const CR_TO_R_INT: i32 = 22970; // 1.402 << 14
const CB_TO_B_INT: i32 = 29032; // 1.772 << 14
const CR_TO_G_INT: i32 = -11700; // -0.714136 << 14
const CB_TO_G_INT: i32 = -5638; // -0.344136 << 14
const YUV_ROUND: i32 = 8192; // 0.5 << 14 for rounding

/// Fast integer YCbCr to RGB conversion for 16 pixels.
///
/// This is the core conversion function for the fast decode path.
/// Takes i16 inputs (IDCT output with level shift already applied, range 0-255)
/// and writes interleaved RGB u8 output.
///
/// The conversion uses 14-bit fixed-point arithmetic for speed.
#[inline]
pub fn ycbcr_to_rgb_i16_x16(
    y: &[i16; 16],
    cb: &[i16; 16],
    cr: &[i16; 16],
    rgb: &mut [u8],
    offset: &mut usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(token) = archmage::X64V3Token::summon() {
            ycbcr_to_rgb_i16_x16_avx2(token, y, cb, cr, rgb, offset);
            return;
        }
    }
    // Scalar fallback
    ycbcr_to_rgb_i16_x16_scalar(y, cb, cr, rgb, offset);
}

/// Scalar implementation of integer YCbCr to RGB for 16 pixels.
#[inline]
fn ycbcr_to_rgb_i16_x16_scalar(
    y: &[i16; 16],
    cb: &[i16; 16],
    cr: &[i16; 16],
    rgb: &mut [u8],
    offset: &mut usize,
) {
    for i in 0..16 {
        let y_val = i32::from(y[i]);
        let cb_val = i32::from(cb[i]) - 128;
        let cr_val = i32::from(cr[i]) - 128;

        // Fixed-point conversion with 14-bit precision
        let y_scaled = y_val * Y_CF_INT + YUV_ROUND;

        let r = (y_scaled + cr_val * CR_TO_R_INT) >> 14;
        let g = (y_scaled + cr_val * CR_TO_G_INT + cb_val * CB_TO_G_INT) >> 14;
        let b = (y_scaled + cb_val * CB_TO_B_INT) >> 14;

        let idx = *offset + i * 3;
        rgb[idx] = r.clamp(0, 255) as u8;
        rgb[idx + 1] = g.clamp(0, 255) as u8;
        rgb[idx + 2] = b.clamp(0, 255) as u8;
    }

    *offset += 48;
}

/// AVX2 implementation of integer YCbCr to RGB for 16 pixels.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ycbcr_to_rgb_i16_x16_avx2(
    _token: archmage::X64V3Token,
    y: &[i16; 16],
    cb: &[i16; 16],
    cr: &[i16; 16],
    rgb: &mut [u8],
    offset: &mut usize,
) {
    use core::arch::x86_64::*;

    // Load Y, Cb, Cr (16 i16 values each)
    let (y_vec, cb_vec, cr_vec) = (
        safe_simd::_mm256_loadu_si256(y),
        safe_simd::_mm256_loadu_si256(cb),
        safe_simd::_mm256_loadu_si256(cr),
    );

    // Subtract 128 from Cb and Cr (bias removal)
    let bias = _mm256_set1_epi16(128);
    let cb_centered = _mm256_sub_epi16(cb_vec, bias);
    let cr_centered = _mm256_sub_epi16(cr_vec, bias);

    // Y coefficient and rounding
    let y_coeff = _mm256_set1_epi32(Y_CF_INT);
    let rounding = _mm256_set1_epi32(YUV_ROUND);

    // Zero-extend Y to 32-bit (Y is unsigned [0,255]).
    // unpacklo/hi gives lane ordering that works correctly with packs_epi32:
    // lo = [0,1,2,3 | 8,9,10,11], hi = [4,5,6,7 | 12,13,14,15]
    let zero = _mm256_setzero_si256();
    let y_lo = _mm256_unpacklo_epi16(y_vec, zero);
    let y_hi = _mm256_unpackhi_epi16(y_vec, zero);

    // y_scaled = y * Y_CF + rounding
    let y_scaled_lo = _mm256_add_epi32(_mm256_mullo_epi32(y_lo, y_coeff), rounding);
    let y_scaled_hi = _mm256_add_epi32(_mm256_mullo_epi32(y_hi, y_coeff), rounding);

    // Sign-extend Cb/Cr to 32-bit (they are signed [-128,127]).
    // Use arithmetic shift to get sign bits, then unpack with those for proper sign extension.
    // This maintains the same lane ordering as Y for correct packing.
    let cb_sign = _mm256_srai_epi16(cb_centered, 15); // All 1s for negative, all 0s for positive
    let cr_sign = _mm256_srai_epi16(cr_centered, 15);
    let cb_lo = _mm256_unpacklo_epi16(cb_centered, cb_sign);
    let cb_hi = _mm256_unpackhi_epi16(cb_centered, cb_sign);
    let cr_lo = _mm256_unpacklo_epi16(cr_centered, cr_sign);
    let cr_hi = _mm256_unpackhi_epi16(cr_centered, cr_sign);

    // R = (y_scaled + cr * CR_TO_R) >> 14
    let r_lo = _mm256_srai_epi32(
        _mm256_add_epi32(
            y_scaled_lo,
            _mm256_mullo_epi32(cr_lo, _mm256_set1_epi32(CR_TO_R_INT)),
        ),
        14,
    );
    let r_hi = _mm256_srai_epi32(
        _mm256_add_epi32(
            y_scaled_hi,
            _mm256_mullo_epi32(cr_hi, _mm256_set1_epi32(CR_TO_R_INT)),
        ),
        14,
    );

    // G = (y_scaled + cr * CR_TO_G + cb * CB_TO_G) >> 14
    let g_lo = _mm256_srai_epi32(
        _mm256_add_epi32(
            y_scaled_lo,
            _mm256_add_epi32(
                _mm256_mullo_epi32(cr_lo, _mm256_set1_epi32(CR_TO_G_INT)),
                _mm256_mullo_epi32(cb_lo, _mm256_set1_epi32(CB_TO_G_INT)),
            ),
        ),
        14,
    );
    let g_hi = _mm256_srai_epi32(
        _mm256_add_epi32(
            y_scaled_hi,
            _mm256_add_epi32(
                _mm256_mullo_epi32(cr_hi, _mm256_set1_epi32(CR_TO_G_INT)),
                _mm256_mullo_epi32(cb_hi, _mm256_set1_epi32(CB_TO_G_INT)),
            ),
        ),
        14,
    );

    // B = (y_scaled + cb * CB_TO_B) >> 14
    let b_lo = _mm256_srai_epi32(
        _mm256_add_epi32(
            y_scaled_lo,
            _mm256_mullo_epi32(cb_lo, _mm256_set1_epi32(CB_TO_B_INT)),
        ),
        14,
    );
    let b_hi = _mm256_srai_epi32(
        _mm256_add_epi32(
            y_scaled_hi,
            _mm256_mullo_epi32(cb_hi, _mm256_set1_epi32(CB_TO_B_INT)),
        ),
        14,
    );

    // Pack i32 -> i16 with saturation, then i16 -> u8 with unsigned saturation
    let r_16 = _mm256_packs_epi32(r_lo, r_hi);
    let g_16 = _mm256_packs_epi32(g_lo, g_hi);
    let b_16 = _mm256_packs_epi32(b_lo, b_hi);

    // packus saturates to 0-255
    let r_8 = _mm256_packus_epi16(r_16, _mm256_setzero_si256());
    let g_8 = _mm256_packus_epi16(g_16, _mm256_setzero_si256());
    let b_8 = _mm256_packus_epi16(b_16, _mm256_setzero_si256());

    // Reorder lanes for correct order after packing
    let r_8 = _mm256_permute4x64_epi64(r_8, 0b11_01_10_00);
    let g_8 = _mm256_permute4x64_epi64(g_8, 0b11_01_10_00);
    let b_8 = _mm256_permute4x64_epi64(b_8, 0b11_01_10_00);

    // Interleave RGB using shuffle and blend (from zune-jpeg)
    let sh_r = _mm256_setr_epi8(
        0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14,
        9, 4, 15, 10, 5,
    );
    let sh_g = _mm256_setr_epi8(
        5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3,
        14, 9, 4, 15, 10,
    );
    let sh_b = _mm256_setr_epi8(
        10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8,
        3, 14, 9, 4, 15,
    );

    let r0 = _mm256_shuffle_epi8(r_8, sh_r);
    let g0 = _mm256_shuffle_epi8(g_8, sh_g);
    let b0 = _mm256_shuffle_epi8(b_8, sh_b);

    let m0 = _mm256_setr_epi8(
        0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1,
        0, 0, -1, 0, 0,
    );
    let m1 = _mm256_setr_epi8(
        0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0,
        -1, 0, 0, -1, 0,
    );

    let p0 = _mm256_blendv_epi8(_mm256_blendv_epi8(r0, g0, m0), b0, m1);
    let p1 = _mm256_blendv_epi8(_mm256_blendv_epi8(g0, b0, m0), r0, m1);
    let p2 = _mm256_blendv_epi8(_mm256_blendv_epi8(b0, r0, m0), g0, m1);

    let rgb0 = _mm256_permute2x128_si256(p0, p1, 0x20);
    let rgb1 = _mm256_permute2x128_si256(p2, p0, 0x30);

    // Store 48 bytes (16 pixels * 3 channels)
    safe_simd::_mm256_storeu_si256(
        <&mut [u8; 32]>::try_from(&mut rgb[*offset..*offset + 32]).unwrap(),
        rgb0,
    );
    safe_simd::_mm_storeu_si128(
        <&mut [u8; 16]>::try_from(&mut rgb[*offset + 32..*offset + 48]).unwrap(),
        _mm256_castsi256_si128(rgb1),
    );

    *offset += 48;
}

/// Autovectorized YCbCr to separate R, G, B planes.
///
/// This function is decorated with `#[autoversion]` to generate optimized versions
/// for different SIMD instruction sets (AVX2, SSE4.1, NEON) with runtime dispatch.
/// Writing to separate planes allows better autovectorization than interleaved output.
#[archmage::autoversion]
fn ycbcr_to_rgb_planes_autovec(
    y_plane: &[i16],
    cb_plane: &[i16],
    cr_plane: &[i16],
    r_out: &mut [u8],
    g_out: &mut [u8],
    b_out: &mut [u8],
) {
    let len = y_plane.len();

    for i in 0..len {
        let y_val = i32::from(y_plane[i]);
        let cb_val = i32::from(cb_plane[i]) - 128;
        let cr_val = i32::from(cr_plane[i]) - 128;

        let y_scaled = y_val * Y_CF_INT + YUV_ROUND;

        let r_raw = (y_scaled + cr_val * CR_TO_R_INT) >> 14;
        let g_raw = (y_scaled + cr_val * CR_TO_G_INT + cb_val * CB_TO_G_INT) >> 14;
        let b_raw = (y_scaled + cb_val * CB_TO_B_INT) >> 14;

        // Clamp to [0, 255]
        r_out[i] = r_raw.clamp(0, 255) as u8;
        g_out[i] = g_raw.clamp(0, 255) as u8;
        b_out[i] = b_raw.clamp(0, 255) as u8;
    }
}

/// Interleave R, G, B planes into RGB buffer.
#[archmage::autoversion]
fn interleave_rgb_planes(r: &[u8], g: &[u8], b: &[u8], rgb: &mut [u8]) {
    let len = r.len();
    for i in 0..len {
        let out_idx = i * 3;
        rgb[out_idx] = r[i];
        rgb[out_idx + 1] = g[i];
        rgb[out_idx + 2] = b[i];
    }
}

/// Batch convert i16 YCbCr planes to interleaved RGB u8.
///
/// This is the fast path for standard JPEG decoding, avoiding f32 entirely.
/// Input planes should be i16 with values in [0, 255] range (level-shifted IDCT output).
pub fn ycbcr_planes_i16_to_rgb_u8(
    y_plane: &[i16],
    cb_plane: &[i16],
    cr_plane: &[i16],
    rgb: &mut [u8],
) {
    debug_assert_eq!(y_plane.len(), cb_plane.len());
    debug_assert_eq!(y_plane.len(), cr_plane.len());
    debug_assert_eq!(rgb.len(), y_plane.len() * 3);

    let len = y_plane.len();

    // Use AVX-512 path when available (16 pixels with wider intermediates, fewer instructions)
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(token) = archmage::X64V4Token::summon() {
            ycbcr_planes_i16_to_rgb_u8_avx512(token, y_plane, cb_plane, cr_plane, rgb);
            return;
        }
    }

    // Use AVX2 SIMD path when available (16 pixels at a time, direct interleaved output)
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(token) = archmage::X64V3Token::summon() {
            ycbcr_planes_i16_to_rgb_u8_avx2(token, y_plane, cb_plane, cr_plane, rgb);
            return;
        }
    }

    // Scalar fallback - process directly without temp allocations
    for i in 0..len {
        let y_val = i32::from(y_plane[i]);
        let cb_val = i32::from(cb_plane[i]) - 128;
        let cr_val = i32::from(cr_plane[i]) - 128;

        let y_scaled = y_val * Y_CF_INT + YUV_ROUND;

        let r = (y_scaled + cr_val * CR_TO_R_INT) >> 14;
        let g = (y_scaled + cr_val * CR_TO_G_INT + cb_val * CB_TO_G_INT) >> 14;
        let b = (y_scaled + cb_val * CB_TO_B_INT) >> 14;

        let idx = i * 3;
        rgb[idx] = r.clamp(0, 255) as u8;
        rgb[idx + 1] = g.clamp(0, 255) as u8;
        rgb[idx + 2] = b.clamp(0, 255) as u8;
    }
}

/// AVX2 batch conversion of YCbCr planes to interleaved RGB.
/// Processes 16 pixels at a time with direct pointer loads.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ycbcr_planes_i16_to_rgb_u8_avx2(
    _token: archmage::X64V3Token,
    y_plane: &[i16],
    cb_plane: &[i16],
    cr_plane: &[i16],
    rgb: &mut [u8],
) {
    use core::arch::x86_64::*;

    let len = y_plane.len();

    // Preload constants outside the loop
    let bias = _mm256_set1_epi16(128);
    let y_coeff = _mm256_set1_epi32(Y_CF_INT);
    let rounding = _mm256_set1_epi32(YUV_ROUND);
    let cr_to_r = _mm256_set1_epi32(CR_TO_R_INT);
    let cr_to_g = _mm256_set1_epi32(CR_TO_G_INT);
    let cb_to_g = _mm256_set1_epi32(CB_TO_G_INT);
    let cb_to_b = _mm256_set1_epi32(CB_TO_B_INT);
    let zero = _mm256_setzero_si256();

    // Shuffle masks for RGB interleaving
    let sh_r = _mm256_setr_epi8(
        0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14,
        9, 4, 15, 10, 5,
    );
    let sh_g = _mm256_setr_epi8(
        5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3,
        14, 9, 4, 15, 10,
    );
    let sh_b = _mm256_setr_epi8(
        10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8,
        3, 14, 9, 4, 15,
    );
    let m0 = _mm256_setr_epi8(
        0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1,
        0, 0, -1, 0, 0,
    );
    let m1 = _mm256_setr_epi8(
        0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0,
        -1, 0, 0, -1, 0,
    );

    // Use chunks_exact to let the compiler prove slice lengths, eliminating bounds checks.
    // Input: 3 planes of i16, chunked by 16. Output: interleaved RGB u8, chunked by 48.
    let y_chunks = y_plane.chunks_exact(16);
    let remainder_len = y_chunks.remainder().len();
    for ((y_chunk, cb_chunk), (cr_chunk, rgb_chunk)) in y_chunks
        .zip(cb_plane.chunks_exact(16))
        .zip(cr_plane.chunks_exact(16).zip(rgb.chunks_exact_mut(48)))
    {
        let (y_vec, cb_vec, cr_vec) = (
            safe_simd::_mm256_loadu_si256(<&[i16; 16]>::try_from(y_chunk).unwrap()),
            safe_simd::_mm256_loadu_si256(<&[i16; 16]>::try_from(cb_chunk).unwrap()),
            safe_simd::_mm256_loadu_si256(<&[i16; 16]>::try_from(cr_chunk).unwrap()),
        );

        // Subtract 128 from Cb and Cr
        let cb_centered = _mm256_sub_epi16(cb_vec, bias);
        let cr_centered = _mm256_sub_epi16(cr_vec, bias);

        // Zero-extend Y to 32-bit
        let y_lo = _mm256_unpacklo_epi16(y_vec, zero);
        let y_hi = _mm256_unpackhi_epi16(y_vec, zero);

        // y_scaled = y * Y_CF + rounding
        let y_scaled_lo = _mm256_add_epi32(_mm256_mullo_epi32(y_lo, y_coeff), rounding);
        let y_scaled_hi = _mm256_add_epi32(_mm256_mullo_epi32(y_hi, y_coeff), rounding);

        // Sign-extend Cb/Cr to 32-bit
        let cb_sign = _mm256_srai_epi16(cb_centered, 15);
        let cr_sign = _mm256_srai_epi16(cr_centered, 15);
        let cb_lo = _mm256_unpacklo_epi16(cb_centered, cb_sign);
        let cb_hi = _mm256_unpackhi_epi16(cb_centered, cb_sign);
        let cr_lo = _mm256_unpacklo_epi16(cr_centered, cr_sign);
        let cr_hi = _mm256_unpackhi_epi16(cr_centered, cr_sign);

        // R = (y_scaled + cr * CR_TO_R) >> 14
        let r_lo = _mm256_srai_epi32(
            _mm256_add_epi32(y_scaled_lo, _mm256_mullo_epi32(cr_lo, cr_to_r)),
            14,
        );
        let r_hi = _mm256_srai_epi32(
            _mm256_add_epi32(y_scaled_hi, _mm256_mullo_epi32(cr_hi, cr_to_r)),
            14,
        );

        // G = (y_scaled + cr * CR_TO_G + cb * CB_TO_G) >> 14
        let g_lo = _mm256_srai_epi32(
            _mm256_add_epi32(
                y_scaled_lo,
                _mm256_add_epi32(
                    _mm256_mullo_epi32(cr_lo, cr_to_g),
                    _mm256_mullo_epi32(cb_lo, cb_to_g),
                ),
            ),
            14,
        );
        let g_hi = _mm256_srai_epi32(
            _mm256_add_epi32(
                y_scaled_hi,
                _mm256_add_epi32(
                    _mm256_mullo_epi32(cr_hi, cr_to_g),
                    _mm256_mullo_epi32(cb_hi, cb_to_g),
                ),
            ),
            14,
        );

        // B = (y_scaled + cb * CB_TO_B) >> 14
        let b_lo = _mm256_srai_epi32(
            _mm256_add_epi32(y_scaled_lo, _mm256_mullo_epi32(cb_lo, cb_to_b)),
            14,
        );
        let b_hi = _mm256_srai_epi32(
            _mm256_add_epi32(y_scaled_hi, _mm256_mullo_epi32(cb_hi, cb_to_b)),
            14,
        );

        // Pack i32 -> i16 -> u8
        let r_16 = _mm256_packs_epi32(r_lo, r_hi);
        let g_16 = _mm256_packs_epi32(g_lo, g_hi);
        let b_16 = _mm256_packs_epi32(b_lo, b_hi);

        let r_8 = _mm256_permute4x64_epi64(_mm256_packus_epi16(r_16, zero), 0b11_01_10_00);
        let g_8 = _mm256_permute4x64_epi64(_mm256_packus_epi16(g_16, zero), 0b11_01_10_00);
        let b_8 = _mm256_permute4x64_epi64(_mm256_packus_epi16(b_16, zero), 0b11_01_10_00);

        // Interleave RGB
        let r0 = _mm256_shuffle_epi8(r_8, sh_r);
        let g0 = _mm256_shuffle_epi8(g_8, sh_g);
        let b0 = _mm256_shuffle_epi8(b_8, sh_b);

        let p0 = _mm256_blendv_epi8(_mm256_blendv_epi8(r0, g0, m0), b0, m1);
        let p1 = _mm256_blendv_epi8(_mm256_blendv_epi8(g0, b0, m0), r0, m1);
        let p2 = _mm256_blendv_epi8(_mm256_blendv_epi8(b0, r0, m0), g0, m1);

        let rgb0 = _mm256_permute2x128_si256(p0, p1, 0x20);
        let rgb1 = _mm256_permute2x128_si256(p2, p0, 0x30);

        // Store 48 bytes (16 pixels * 3 channels)
        let (rgb_lo, rgb_hi) = rgb_chunk.split_at_mut(32);
        safe_simd::_mm256_storeu_si256(<&mut [u8; 32]>::try_from(rgb_lo).unwrap(), rgb0);
        safe_simd::_mm_storeu_si128(
            <&mut [u8; 16]>::try_from(rgb_hi).unwrap(),
            _mm256_castsi256_si128(rgb1),
        );
    }

    // Handle remainder with scalar
    let remainder_start = len - remainder_len;
    for i in remainder_start..len {
        let y_val = i32::from(y_plane[i]);
        let cb_val = i32::from(cb_plane[i]) - 128;
        let cr_val = i32::from(cr_plane[i]) - 128;

        let y_scaled = y_val * Y_CF_INT + YUV_ROUND;

        let r = (y_scaled + cr_val * CR_TO_R_INT) >> 14;
        let g = (y_scaled + cr_val * CR_TO_G_INT + cb_val * CB_TO_G_INT) >> 14;
        let b = (y_scaled + cb_val * CB_TO_B_INT) >> 14;

        let idx = i * 3;
        rgb[idx] = r.clamp(0, 255) as u8;
        rgb[idx + 1] = g.clamp(0, 255) as u8;
        rgb[idx + 2] = b.clamp(0, 255) as u8;
    }
}

/// AVX-512 batch conversion of YCbCr planes to interleaved RGB.
///
/// Processes 16 pixels at a time using 512-bit intermediates for the i32 compute
/// phase. Key advantages over AVX2:
/// - `_mm512_cvtepi16_epi32` widens 16 i16→i32 in one instruction (vs manual unpack)
/// - Single `_mm512_mullo_epi32` per multiply (vs lo+hi halves)
/// - `_mm512_cvtsepi32_epi16` packs cleanly without permute fixup
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ycbcr_planes_i16_to_rgb_u8_avx512(
    _token: archmage::X64V4Token,
    y_plane: &[i16],
    cb_plane: &[i16],
    cr_plane: &[i16],
    rgb: &mut [u8],
) {
    use core::arch::x86_64::*;

    let len = y_plane.len();
    let chunks = len / 16;

    // Preload 512-bit constants
    let y_coeff = _mm512_set1_epi32(Y_CF_INT);
    let rounding = _mm512_set1_epi32(YUV_ROUND);
    let cr_to_r = _mm512_set1_epi32(CR_TO_R_INT);
    let cr_to_g = _mm512_set1_epi32(CR_TO_G_INT);
    let cb_to_g = _mm512_set1_epi32(CB_TO_G_INT);
    let cb_to_b = _mm512_set1_epi32(CB_TO_B_INT);
    let bias_16 = _mm256_set1_epi16(128);
    let zero_256 = _mm256_setzero_si256();

    // RGB interleave masks (same as AVX2 — we pack down to __m256i for interleave)
    let sh_r = _mm256_setr_epi8(
        0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14,
        9, 4, 15, 10, 5,
    );
    let sh_g = _mm256_setr_epi8(
        5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3,
        14, 9, 4, 15, 10,
    );
    let sh_b = _mm256_setr_epi8(
        10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8,
        3, 14, 9, 4, 15,
    );
    let m0 = _mm256_setr_epi8(
        0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1,
        0, 0, -1, 0, 0,
    );
    let m1 = _mm256_setr_epi8(
        0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0,
        -1, 0, 0, -1, 0,
    );

    for chunk in 0..chunks {
        let in_offset = chunk * 16;
        let out_offset = chunk * 48;

        // Load 16 i16 values per channel (256-bit loads)
        let y_vec = safe_simd::_mm256_loadu_si256(
            <&[i16; 16]>::try_from(&y_plane[in_offset..in_offset + 16]).unwrap(),
        );
        let cb_vec = safe_simd::_mm256_loadu_si256(
            <&[i16; 16]>::try_from(&cb_plane[in_offset..in_offset + 16]).unwrap(),
        );
        let cr_vec = safe_simd::_mm256_loadu_si256(
            <&[i16; 16]>::try_from(&cr_plane[in_offset..in_offset + 16]).unwrap(),
        );

        // Subtract 128 from Cb and Cr (still 256-bit i16)
        let cb_centered = _mm256_sub_epi16(cb_vec, bias_16);
        let cr_centered = _mm256_sub_epi16(cr_vec, bias_16);

        // Widen to 512-bit i32 in ONE instruction each (vs 4+ instructions in AVX2)
        let y_32 = _mm512_cvtepi16_epi32(y_vec);
        let cb_32 = _mm512_cvtepi16_epi32(cb_centered);
        let cr_32 = _mm512_cvtepi16_epi32(cr_centered);

        // y_scaled = y * Y_CF + rounding (one multiply + one add, vs two each in AVX2)
        let y_scaled = _mm512_add_epi32(_mm512_mullo_epi32(y_32, y_coeff), rounding);

        // R = (y_scaled + cr * CR_TO_R) >> 14
        let r_32 = _mm512_srai_epi32(
            _mm512_add_epi32(y_scaled, _mm512_mullo_epi32(cr_32, cr_to_r)),
            14,
        );

        // G = (y_scaled + cr * CR_TO_G + cb * CB_TO_G) >> 14
        let g_32 = _mm512_srai_epi32(
            _mm512_add_epi32(
                y_scaled,
                _mm512_add_epi32(
                    _mm512_mullo_epi32(cr_32, cr_to_g),
                    _mm512_mullo_epi32(cb_32, cb_to_g),
                ),
            ),
            14,
        );

        // B = (y_scaled + cb * CB_TO_B) >> 14
        let b_32 = _mm512_srai_epi32(
            _mm512_add_epi32(y_scaled, _mm512_mullo_epi32(cb_32, cb_to_b)),
            14,
        );

        // Pack i32 → i16 with saturation (512→256, naturally ordered!)
        let r_16 = _mm512_cvtsepi32_epi16(r_32);
        let g_16 = _mm512_cvtsepi32_epi16(g_32);
        let b_16 = _mm512_cvtsepi32_epi16(b_32);

        // Pack i16 → u8 with saturation (256→128, then fix lane ordering)
        let r_8 = _mm256_permute4x64_epi64(_mm256_packus_epi16(r_16, zero_256), 0b11_01_10_00);
        let g_8 = _mm256_permute4x64_epi64(_mm256_packus_epi16(g_16, zero_256), 0b11_01_10_00);
        let b_8 = _mm256_permute4x64_epi64(_mm256_packus_epi16(b_16, zero_256), 0b11_01_10_00);

        // Interleave RGB (same shuffle+blend as AVX2 path)
        let r0 = _mm256_shuffle_epi8(r_8, sh_r);
        let g0 = _mm256_shuffle_epi8(g_8, sh_g);
        let b0 = _mm256_shuffle_epi8(b_8, sh_b);

        let p0 = _mm256_blendv_epi8(_mm256_blendv_epi8(r0, g0, m0), b0, m1);
        let p1 = _mm256_blendv_epi8(_mm256_blendv_epi8(g0, b0, m0), r0, m1);
        let p2 = _mm256_blendv_epi8(_mm256_blendv_epi8(b0, r0, m0), g0, m1);

        let rgb0 = _mm256_permute2x128_si256(p0, p1, 0x20);
        let rgb1 = _mm256_permute2x128_si256(p2, p0, 0x30);

        // Store 48 bytes (16 pixels × 3 channels)
        safe_simd::_mm256_storeu_si256(
            <&mut [u8; 32]>::try_from(&mut rgb[out_offset..out_offset + 32]).unwrap(),
            rgb0,
        );
        safe_simd::_mm_storeu_si128(
            <&mut [u8; 16]>::try_from(&mut rgb[out_offset + 32..out_offset + 48]).unwrap(),
            _mm256_castsi256_si128(rgb1),
        );
    }

    // Handle remainder with scalar
    let remainder_start = chunks * 16;
    for i in remainder_start..len {
        let y_val = i32::from(y_plane[i]);
        let cb_val = i32::from(cb_plane[i]) - 128;
        let cr_val = i32::from(cr_plane[i]) - 128;

        let y_scaled = y_val * Y_CF_INT + YUV_ROUND;

        let r = (y_scaled + cr_val * CR_TO_R_INT) >> 14;
        let g = (y_scaled + cr_val * CR_TO_G_INT + cb_val * CB_TO_G_INT) >> 14;
        let b = (y_scaled + cb_val * CB_TO_B_INT) >> 14;

        let idx = i * 3;
        rgb[idx] = r.clamp(0, 255) as u8;
        rgb[idx + 1] = g.clamp(0, 255) as u8;
        rgb[idx + 2] = b.clamp(0, 255) as u8;
    }
}

/// Fused box-filter 4:2:0 horizontal upsample + YCbCr→RGB conversion.
///
/// Processes one output row. For 4:2:0 box filter, both output rows in a vertical
/// pair use the same chroma row (nearest-neighbor vertical upsampling).
///
/// Each chroma pixel maps to 2 horizontal output pixels. This eliminates
/// the intermediate upsampled chroma buffers entirely.
///
/// `y_row`: Y values for one output row (`width` elements)
/// `cb_row`: Cb values at half resolution (`width/2` elements)
/// `cr_row`: Cr values at half resolution (`width/2` elements)
/// `rgb`: Output RGB buffer (`width * 3` bytes)
/// `width`: Output width in pixels
pub fn fused_h2v2_box_ycbcr_to_rgb_u8(
    y_row: &[i16],
    cb_row: &[i16],
    cr_row: &[i16],
    rgb: &mut [u8],
    width: usize,
) {
    debug_assert!(y_row.len() >= width);
    debug_assert!(cb_row.len() >= (width + 1) / 2);
    debug_assert!(cr_row.len() >= (width + 1) / 2);
    debug_assert!(rgb.len() >= width * 3);

    #[cfg(target_arch = "x86_64")]
    {
        if let Some(token) = archmage::X64V3Token::summon() {
            fused_h2v2_box_ycbcr_to_rgb_u8_avx2(token, y_row, cb_row, cr_row, rgb, width);
            return;
        }
    }

    // Scalar fallback
    let chroma_width = (width + 1) / 2;
    for cx in 0..chroma_width {
        let cb_val = i32::from(cb_row[cx]) - 128;
        let cr_val = i32::from(cr_row[cx]) - 128;

        // Left output pixel
        let px0 = cx * 2;
        if px0 < width {
            let y_val = i32::from(y_row[px0]);
            let y_scaled = y_val * Y_CF_INT + YUV_ROUND;
            let r = (y_scaled + cr_val * CR_TO_R_INT) >> 14;
            let g = (y_scaled + cr_val * CR_TO_G_INT + cb_val * CB_TO_G_INT) >> 14;
            let b = (y_scaled + cb_val * CB_TO_B_INT) >> 14;
            let idx = px0 * 3;
            rgb[idx] = r.clamp(0, 255) as u8;
            rgb[idx + 1] = g.clamp(0, 255) as u8;
            rgb[idx + 2] = b.clamp(0, 255) as u8;
        }

        // Right output pixel (same chroma)
        let px1 = cx * 2 + 1;
        if px1 < width {
            let y_val = i32::from(y_row[px1]);
            let y_scaled = y_val * Y_CF_INT + YUV_ROUND;
            let r = (y_scaled + cr_val * CR_TO_R_INT) >> 14;
            let g = (y_scaled + cr_val * CR_TO_G_INT + cb_val * CB_TO_G_INT) >> 14;
            let b = (y_scaled + cb_val * CB_TO_B_INT) >> 14;
            let idx = px1 * 3;
            rgb[idx] = r.clamp(0, 255) as u8;
            rgb[idx + 1] = g.clamp(0, 255) as u8;
            rgb[idx + 2] = b.clamp(0, 255) as u8;
        }
    }
}

/// Fused horizontal-fancy + vertical-box 4:2:0 upsample + YCbCr→RGB.
///
/// Vertical: box (duplicate rows). Horizontal: triangle (3:1 bilinear).
/// This is the hybrid "h-fancy" mode — no vertical context needed, so
/// each MCU row is independent (trivially parallelizable), but horizontal
/// chroma transitions are smoothed instead of stairstepped.
///
/// Takes half-width chroma rows (already vertically duplicated by caller).
pub fn fused_h2v2_hfancy_ycbcr_to_rgb_u8(
    y_row: &[i16],
    cb_row: &[i16],
    cr_row: &[i16],
    rgb: &mut [u8],
    width: usize,
) {
    debug_assert!(y_row.len() >= width);
    debug_assert!(cb_row.len() >= (width + 1) / 2);
    debug_assert!(cr_row.len() >= (width + 1) / 2);
    debug_assert!(rgb.len() >= width * 3);

    #[cfg(target_arch = "x86_64")]
    {
        if let Some(token) = archmage::X64V3Token::summon() {
            fused_h2v2_hfancy_ycbcr_to_rgb_u8_avx2(token, y_row, cb_row, cr_row, rgb, width);
            return;
        }
    }

    let chroma_width = (width + 1) / 2;

    for cx in 0..chroma_width {
        let curr_cb = i32::from(cb_row[cx]);
        let curr_cr = i32::from(cr_row[cx]);

        // Left output pixel: (3*curr + left_neighbor + 2) >> 2
        let left_cb = if cx > 0 {
            i32::from(cb_row[cx - 1])
        } else {
            curr_cb
        };
        let left_cr = if cx > 0 {
            i32::from(cr_row[cx - 1])
        } else {
            curr_cr
        };
        let cb_l = ((3 * curr_cb + left_cb + 2) >> 2) - 128;
        let cr_l = ((3 * curr_cr + left_cr + 2) >> 2) - 128;

        let px0 = cx * 2;
        if px0 < width {
            let y_val = i32::from(y_row[px0]);
            let y_scaled = y_val * Y_CF_INT + YUV_ROUND;
            let r = (y_scaled + cr_l * CR_TO_R_INT) >> 14;
            let g = (y_scaled + cr_l * CR_TO_G_INT + cb_l * CB_TO_G_INT) >> 14;
            let b = (y_scaled + cb_l * CB_TO_B_INT) >> 14;
            let idx = px0 * 3;
            rgb[idx] = r.clamp(0, 255) as u8;
            rgb[idx + 1] = g.clamp(0, 255) as u8;
            rgb[idx + 2] = b.clamp(0, 255) as u8;
        }

        // Right output pixel: (3*curr + right_neighbor + 2) >> 2
        let right_cb = if cx + 1 < chroma_width {
            i32::from(cb_row[cx + 1])
        } else {
            curr_cb
        };
        let right_cr = if cx + 1 < chroma_width {
            i32::from(cr_row[cx + 1])
        } else {
            curr_cr
        };
        let cb_r = ((3 * curr_cb + right_cb + 2) >> 2) - 128;
        let cr_r = ((3 * curr_cr + right_cr + 2) >> 2) - 128;

        let px1 = cx * 2 + 1;
        if px1 < width {
            let y_val = i32::from(y_row[px1]);
            let y_scaled = y_val * Y_CF_INT + YUV_ROUND;
            let r = (y_scaled + cr_r * CR_TO_R_INT) >> 14;
            let g = (y_scaled + cr_r * CR_TO_G_INT + cb_r * CB_TO_G_INT) >> 14;
            let b = (y_scaled + cb_r * CB_TO_B_INT) >> 14;
            let idx = px1 * 3;
            rgb[idx] = r.clamp(0, 255) as u8;
            rgb[idx + 1] = g.clamp(0, 255) as u8;
            rgb[idx + 2] = b.clamp(0, 255) as u8;
        }
    }
}

/// AVX2 fused h-fancy 4:2:0 upsample + YCbCr→RGB.
/// Horizontal: triangle filter (3:1 bilinear). Vertical: box (caller duplicates).
/// Processes 16 output pixels per iteration (8 chroma pixels → 16 output pixels).
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn fused_h2v2_hfancy_ycbcr_to_rgb_u8_avx2(
    _token: archmage::X64V3Token,
    y_row: &[i16],
    cb_row: &[i16],
    cr_row: &[i16],
    rgb: &mut [u8],
    width: usize,
) {
    use core::arch::x86_64::*;

    let chroma_width = (width + 1) / 2;
    let chunks = width / 16; // 16 output pixels = 8 chroma samples per chunk

    // Preload constants
    let bias = _mm256_set1_epi16(128);
    let y_coeff = _mm256_set1_epi32(Y_CF_INT);
    let rounding = _mm256_set1_epi32(YUV_ROUND);
    let cr_to_r = _mm256_set1_epi32(CR_TO_R_INT);
    let cr_to_g = _mm256_set1_epi32(CR_TO_G_INT);
    let cb_to_g = _mm256_set1_epi32(CB_TO_G_INT);
    let cb_to_b = _mm256_set1_epi32(CB_TO_B_INT);
    let zero = _mm256_setzero_si256();
    let three = _mm_set1_epi16(3);
    let round2 = _mm_set1_epi16(2);

    // RGB interleave masks
    let sh_r = _mm256_setr_epi8(
        0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14,
        9, 4, 15, 10, 5,
    );
    let sh_g = _mm256_setr_epi8(
        5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3,
        14, 9, 4, 15, 10,
    );
    let sh_b = _mm256_setr_epi8(
        10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8,
        3, 14, 9, 4, 15,
    );
    let m0 = _mm256_setr_epi8(
        0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1,
        0, 0, -1, 0, 0,
    );
    let m1 = _mm256_setr_epi8(
        0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0,
        -1, 0, 0, -1, 0,
    );

    // Track the last chroma value from the previous chunk for left-neighbor lookback
    let mut prev_cb = cb_row[0]; // Edge replication for first chunk
    let mut prev_cr = cr_row[0];

    for chunk in 0..chunks {
        let c_offset = chunk * 8;
        let y_offset = chunk * 16;
        let out_offset = chunk * 48;

        // Load 16 Y values
        let y_vec = safe_simd::_mm256_loadu_si256(
            <&[i16; 16]>::try_from(&y_row[y_offset..y_offset + 16]).unwrap(),
        );

        // Load 8 chroma values
        let cb_curr = safe_simd::_mm_loadu_si128(
            <&[i16; 8]>::try_from(&cb_row[c_offset..c_offset + 8]).unwrap(),
        );
        let cr_curr = safe_simd::_mm_loadu_si128(
            <&[i16; 8]>::try_from(&cr_row[c_offset..c_offset + 8]).unwrap(),
        );

        // Create left-neighbor vector: [prev_last, c0, c1, c2, c3, c4, c5, c6]
        let cb_left = _mm_insert_epi16::<0>(_mm_slli_si128::<2>(cb_curr), prev_cb as i32);
        let cr_left = _mm_insert_epi16::<0>(_mm_slli_si128::<2>(cr_curr), prev_cr as i32);

        // Create right-neighbor vector: [c1, c2, c3, c4, c5, c6, c7, next_first]
        let next_cb = if c_offset + 8 < chroma_width {
            cb_row[c_offset + 8]
        } else {
            cb_row[c_offset + 7] // Edge replication
        };
        let next_cr = if c_offset + 8 < chroma_width {
            cr_row[c_offset + 8]
        } else {
            cr_row[c_offset + 7]
        };
        let cb_right = _mm_insert_epi16::<7>(_mm_srli_si128::<2>(cb_curr), next_cb as i32);
        let cr_right = _mm_insert_epi16::<7>(_mm_srli_si128::<2>(cr_curr), next_cr as i32);

        // Save last value for next chunk's left neighbor
        prev_cb = cb_row[c_offset + 7];
        prev_cr = cr_row[c_offset + 7];

        // Compute bilinear interpolation:
        // interp_left = (3*curr + left + 2) >> 2  (for even output pixels)
        // interp_right = (3*curr + right + 2) >> 2 (for odd output pixels)
        let three_cb = _mm_mullo_epi16(cb_curr, three);
        let three_cr = _mm_mullo_epi16(cr_curr, three);

        let cb_interp_l =
            _mm_srai_epi16::<2>(_mm_add_epi16(_mm_add_epi16(three_cb, cb_left), round2));
        let cr_interp_l =
            _mm_srai_epi16::<2>(_mm_add_epi16(_mm_add_epi16(three_cr, cr_left), round2));
        let cb_interp_r =
            _mm_srai_epi16::<2>(_mm_add_epi16(_mm_add_epi16(three_cb, cb_right), round2));
        let cr_interp_r =
            _mm_srai_epi16::<2>(_mm_add_epi16(_mm_add_epi16(three_cr, cr_right), round2));

        // Interleave left and right: [L0, R0, L1, R1, ...] → 16 chroma values
        let cb_lo = _mm_unpacklo_epi16(cb_interp_l, cb_interp_r);
        let cb_hi = _mm_unpackhi_epi16(cb_interp_l, cb_interp_r);
        let cb_vec = _mm256_set_m128i(cb_hi, cb_lo);

        let cr_lo = _mm_unpacklo_epi16(cr_interp_l, cr_interp_r);
        let cr_hi = _mm_unpackhi_epi16(cr_interp_l, cr_interp_r);
        let cr_vec = _mm256_set_m128i(cr_hi, cr_lo);

        // Subtract 128 from Cb and Cr
        let cb_centered = _mm256_sub_epi16(cb_vec, bias);
        let cr_centered = _mm256_sub_epi16(cr_vec, bias);

        // Zero-extend Y to 32-bit
        let y_lo = _mm256_unpacklo_epi16(y_vec, zero);
        let y_hi = _mm256_unpackhi_epi16(y_vec, zero);

        // y_scaled = y * Y_CF + rounding
        let y_scaled_lo = _mm256_add_epi32(_mm256_mullo_epi32(y_lo, y_coeff), rounding);
        let y_scaled_hi = _mm256_add_epi32(_mm256_mullo_epi32(y_hi, y_coeff), rounding);

        // Sign-extend Cb/Cr to 32-bit
        let cb_sign = _mm256_srai_epi16(cb_centered, 15);
        let cr_sign = _mm256_srai_epi16(cr_centered, 15);
        let cb_lo32 = _mm256_unpacklo_epi16(cb_centered, cb_sign);
        let cb_hi32 = _mm256_unpackhi_epi16(cb_centered, cb_sign);
        let cr_lo32 = _mm256_unpacklo_epi16(cr_centered, cr_sign);
        let cr_hi32 = _mm256_unpackhi_epi16(cr_centered, cr_sign);

        // R = (y_scaled + cr * CR_TO_R) >> 14
        let r_lo = _mm256_srai_epi32(
            _mm256_add_epi32(y_scaled_lo, _mm256_mullo_epi32(cr_lo32, cr_to_r)),
            14,
        );
        let r_hi = _mm256_srai_epi32(
            _mm256_add_epi32(y_scaled_hi, _mm256_mullo_epi32(cr_hi32, cr_to_r)),
            14,
        );

        // G = (y_scaled + cr * CR_TO_G + cb * CB_TO_G) >> 14
        let g_lo = _mm256_srai_epi32(
            _mm256_add_epi32(
                y_scaled_lo,
                _mm256_add_epi32(
                    _mm256_mullo_epi32(cr_lo32, cr_to_g),
                    _mm256_mullo_epi32(cb_lo32, cb_to_g),
                ),
            ),
            14,
        );
        let g_hi = _mm256_srai_epi32(
            _mm256_add_epi32(
                y_scaled_hi,
                _mm256_add_epi32(
                    _mm256_mullo_epi32(cr_hi32, cr_to_g),
                    _mm256_mullo_epi32(cb_hi32, cb_to_g),
                ),
            ),
            14,
        );

        // B = (y_scaled + cb * CB_TO_B) >> 14
        let b_lo = _mm256_srai_epi32(
            _mm256_add_epi32(y_scaled_lo, _mm256_mullo_epi32(cb_lo32, cb_to_b)),
            14,
        );
        let b_hi = _mm256_srai_epi32(
            _mm256_add_epi32(y_scaled_hi, _mm256_mullo_epi32(cb_hi32, cb_to_b)),
            14,
        );

        // Pack i32 -> i16 -> u8
        let r_16 = _mm256_packs_epi32(r_lo, r_hi);
        let g_16 = _mm256_packs_epi32(g_lo, g_hi);
        let b_16 = _mm256_packs_epi32(b_lo, b_hi);

        let r_8 = _mm256_permute4x64_epi64(_mm256_packus_epi16(r_16, zero), 0b11_01_10_00);
        let g_8 = _mm256_permute4x64_epi64(_mm256_packus_epi16(g_16, zero), 0b11_01_10_00);
        let b_8 = _mm256_permute4x64_epi64(_mm256_packus_epi16(b_16, zero), 0b11_01_10_00);

        // Interleave RGB
        let r0 = _mm256_shuffle_epi8(r_8, sh_r);
        let g0 = _mm256_shuffle_epi8(g_8, sh_g);
        let b0 = _mm256_shuffle_epi8(b_8, sh_b);

        let p0 = _mm256_blendv_epi8(_mm256_blendv_epi8(r0, g0, m0), b0, m1);
        let p1 = _mm256_blendv_epi8(_mm256_blendv_epi8(g0, b0, m0), r0, m1);
        let p2 = _mm256_blendv_epi8(_mm256_blendv_epi8(b0, r0, m0), g0, m1);

        let rgb0 = _mm256_permute2x128_si256(p0, p1, 0x20);
        let rgb1 = _mm256_permute2x128_si256(p2, p0, 0x30);

        // Store 48 bytes (16 pixels * 3 channels)
        safe_simd::_mm256_storeu_si256(
            <&mut [u8; 32]>::try_from(&mut rgb[out_offset..out_offset + 32]).unwrap(),
            rgb0,
        );
        safe_simd::_mm_storeu_si128(
            <&mut [u8; 16]>::try_from(&mut rgb[out_offset + 32..out_offset + 48]).unwrap(),
            _mm256_castsi256_si128(rgb1),
        );
    }

    // Handle remainder with scalar
    let c_remainder_start = chunks * 8;
    for cx in c_remainder_start..chroma_width {
        let curr_cb = i32::from(cb_row[cx]);
        let curr_cr = i32::from(cr_row[cx]);

        let left_cb = if cx > 0 {
            i32::from(cb_row[cx - 1])
        } else {
            curr_cb
        };
        let left_cr = if cx > 0 {
            i32::from(cr_row[cx - 1])
        } else {
            curr_cr
        };
        let cb_l = ((3 * curr_cb + left_cb + 2) >> 2) - 128;
        let cr_l = ((3 * curr_cr + left_cr + 2) >> 2) - 128;

        let px0 = cx * 2;
        if px0 < width {
            let y_val = i32::from(y_row[px0]);
            let y_scaled = y_val * Y_CF_INT + YUV_ROUND;
            let r = (y_scaled + cr_l * CR_TO_R_INT) >> 14;
            let g = (y_scaled + cr_l * CR_TO_G_INT + cb_l * CB_TO_G_INT) >> 14;
            let b = (y_scaled + cb_l * CB_TO_B_INT) >> 14;
            let idx = px0 * 3;
            rgb[idx] = r.clamp(0, 255) as u8;
            rgb[idx + 1] = g.clamp(0, 255) as u8;
            rgb[idx + 2] = b.clamp(0, 255) as u8;
        }

        let right_cb = if cx + 1 < chroma_width {
            i32::from(cb_row[cx + 1])
        } else {
            curr_cb
        };
        let right_cr = if cx + 1 < chroma_width {
            i32::from(cr_row[cx + 1])
        } else {
            curr_cr
        };
        let cb_r = ((3 * curr_cb + right_cb + 2) >> 2) - 128;
        let cr_r = ((3 * curr_cr + right_cr + 2) >> 2) - 128;

        let px1 = cx * 2 + 1;
        if px1 < width {
            let y_val = i32::from(y_row[px1]);
            let y_scaled = y_val * Y_CF_INT + YUV_ROUND;
            let r = (y_scaled + cr_r * CR_TO_R_INT) >> 14;
            let g = (y_scaled + cr_r * CR_TO_G_INT + cb_r * CB_TO_G_INT) >> 14;
            let b = (y_scaled + cb_r * CB_TO_B_INT) >> 14;
            let idx = px1 * 3;
            rgb[idx] = r.clamp(0, 255) as u8;
            rgb[idx + 1] = g.clamp(0, 255) as u8;
            rgb[idx + 2] = b.clamp(0, 255) as u8;
        }
    }
}

/// AVX2 fused box-filter 4:2:0 upsample + YCbCr→RGB.
/// Processes 16 output pixels per iteration (8 chroma pixels → 16 output pixels).
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn fused_h2v2_box_ycbcr_to_rgb_u8_avx2(
    _token: archmage::X64V3Token,
    y_row: &[i16],
    cb_row: &[i16],
    cr_row: &[i16],
    rgb: &mut [u8],
    width: usize,
) {
    use core::arch::x86_64::*;

    let chunks = width / 16;

    // Preload constants
    let bias = _mm256_set1_epi16(128);
    let y_coeff = _mm256_set1_epi32(Y_CF_INT);
    let rounding = _mm256_set1_epi32(YUV_ROUND);
    let cr_to_r = _mm256_set1_epi32(CR_TO_R_INT);
    let cr_to_g = _mm256_set1_epi32(CR_TO_G_INT);
    let cb_to_g = _mm256_set1_epi32(CB_TO_G_INT);
    let cb_to_b = _mm256_set1_epi32(CB_TO_B_INT);
    let zero = _mm256_setzero_si256();

    // RGB interleave masks (same as existing ycbcr_planes_i16_to_rgb_u8_avx2)
    let sh_r = _mm256_setr_epi8(
        0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14,
        9, 4, 15, 10, 5,
    );
    let sh_g = _mm256_setr_epi8(
        5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3,
        14, 9, 4, 15, 10,
    );
    let sh_b = _mm256_setr_epi8(
        10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8,
        3, 14, 9, 4, 15,
    );
    let m0 = _mm256_setr_epi8(
        0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1,
        0, 0, -1, 0, 0,
    );
    let m1 = _mm256_setr_epi8(
        0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0,
        -1, 0, 0, -1, 0,
    );

    for chunk in 0..chunks {
        let y_offset = chunk * 16;
        let c_offset = chunk * 8;
        let out_offset = chunk * 48;

        // Load 16 Y values
        let y_vec = safe_simd::_mm256_loadu_si256(
            <&[i16; 16]>::try_from(&y_row[y_offset..y_offset + 16]).unwrap(),
        );

        // Load 8 chroma values and duplicate each for box-filter horizontal upsampling:
        // [c0, c1, c2, c3, c4, c5, c6, c7] → [c0, c0, c1, c1, c2, c2, c3, c3, c4, c4, c5, c5, c6, c6, c7, c7]
        let cb_half = safe_simd::_mm_loadu_si128(
            <&[i16; 8]>::try_from(&cb_row[c_offset..c_offset + 8]).unwrap(),
        );
        let cr_half = safe_simd::_mm_loadu_si128(
            <&[i16; 8]>::try_from(&cr_row[c_offset..c_offset + 8]).unwrap(),
        );

        // Duplicate each i16: unpacklo/hi interleaves with itself
        let cb_lo = _mm_unpacklo_epi16(cb_half, cb_half); // [c0,c0, c1,c1, c2,c2, c3,c3]
        let cb_hi = _mm_unpackhi_epi16(cb_half, cb_half); // [c4,c4, c5,c5, c6,c6, c7,c7]
        let cb_vec = _mm256_set_m128i(cb_hi, cb_lo);

        let cr_lo = _mm_unpacklo_epi16(cr_half, cr_half);
        let cr_hi = _mm_unpackhi_epi16(cr_half, cr_half);
        let cr_vec = _mm256_set_m128i(cr_hi, cr_lo);

        // Subtract 128 from Cb and Cr
        let cb_centered = _mm256_sub_epi16(cb_vec, bias);
        let cr_centered = _mm256_sub_epi16(cr_vec, bias);

        // Zero-extend Y to 32-bit
        let y_lo = _mm256_unpacklo_epi16(y_vec, zero);
        let y_hi = _mm256_unpackhi_epi16(y_vec, zero);

        // y_scaled = y * Y_CF + rounding
        let y_scaled_lo = _mm256_add_epi32(_mm256_mullo_epi32(y_lo, y_coeff), rounding);
        let y_scaled_hi = _mm256_add_epi32(_mm256_mullo_epi32(y_hi, y_coeff), rounding);

        // Sign-extend Cb/Cr to 32-bit
        let cb_sign = _mm256_srai_epi16(cb_centered, 15);
        let cr_sign = _mm256_srai_epi16(cr_centered, 15);
        let cb_lo32 = _mm256_unpacklo_epi16(cb_centered, cb_sign);
        let cb_hi32 = _mm256_unpackhi_epi16(cb_centered, cb_sign);
        let cr_lo32 = _mm256_unpacklo_epi16(cr_centered, cr_sign);
        let cr_hi32 = _mm256_unpackhi_epi16(cr_centered, cr_sign);

        // R = (y_scaled + cr * CR_TO_R) >> 14
        let r_lo = _mm256_srai_epi32(
            _mm256_add_epi32(y_scaled_lo, _mm256_mullo_epi32(cr_lo32, cr_to_r)),
            14,
        );
        let r_hi = _mm256_srai_epi32(
            _mm256_add_epi32(y_scaled_hi, _mm256_mullo_epi32(cr_hi32, cr_to_r)),
            14,
        );

        // G = (y_scaled + cr * CR_TO_G + cb * CB_TO_G) >> 14
        let g_lo = _mm256_srai_epi32(
            _mm256_add_epi32(
                y_scaled_lo,
                _mm256_add_epi32(
                    _mm256_mullo_epi32(cr_lo32, cr_to_g),
                    _mm256_mullo_epi32(cb_lo32, cb_to_g),
                ),
            ),
            14,
        );
        let g_hi = _mm256_srai_epi32(
            _mm256_add_epi32(
                y_scaled_hi,
                _mm256_add_epi32(
                    _mm256_mullo_epi32(cr_hi32, cr_to_g),
                    _mm256_mullo_epi32(cb_hi32, cb_to_g),
                ),
            ),
            14,
        );

        // B = (y_scaled + cb * CB_TO_B) >> 14
        let b_lo = _mm256_srai_epi32(
            _mm256_add_epi32(y_scaled_lo, _mm256_mullo_epi32(cb_lo32, cb_to_b)),
            14,
        );
        let b_hi = _mm256_srai_epi32(
            _mm256_add_epi32(y_scaled_hi, _mm256_mullo_epi32(cb_hi32, cb_to_b)),
            14,
        );

        // Pack i32 -> i16 -> u8
        let r_16 = _mm256_packs_epi32(r_lo, r_hi);
        let g_16 = _mm256_packs_epi32(g_lo, g_hi);
        let b_16 = _mm256_packs_epi32(b_lo, b_hi);

        let r_8 = _mm256_permute4x64_epi64(_mm256_packus_epi16(r_16, zero), 0b11_01_10_00);
        let g_8 = _mm256_permute4x64_epi64(_mm256_packus_epi16(g_16, zero), 0b11_01_10_00);
        let b_8 = _mm256_permute4x64_epi64(_mm256_packus_epi16(b_16, zero), 0b11_01_10_00);

        // Interleave RGB
        let r0 = _mm256_shuffle_epi8(r_8, sh_r);
        let g0 = _mm256_shuffle_epi8(g_8, sh_g);
        let b0 = _mm256_shuffle_epi8(b_8, sh_b);

        let p0 = _mm256_blendv_epi8(_mm256_blendv_epi8(r0, g0, m0), b0, m1);
        let p1 = _mm256_blendv_epi8(_mm256_blendv_epi8(g0, b0, m0), r0, m1);
        let p2 = _mm256_blendv_epi8(_mm256_blendv_epi8(b0, r0, m0), g0, m1);

        let rgb0 = _mm256_permute2x128_si256(p0, p1, 0x20);
        let rgb1 = _mm256_permute2x128_si256(p2, p0, 0x30);

        // Store 48 bytes (16 pixels * 3 channels)
        safe_simd::_mm256_storeu_si256(
            <&mut [u8; 32]>::try_from(&mut rgb[out_offset..out_offset + 32]).unwrap(),
            rgb0,
        );
        safe_simd::_mm_storeu_si128(
            <&mut [u8; 16]>::try_from(&mut rgb[out_offset + 32..out_offset + 48]).unwrap(),
            _mm256_castsi256_si128(rgb1),
        );
    }

    // Handle remainder with scalar
    let c_remainder_start = chunks * 8;
    let chroma_width = (width + 1) / 2;
    for cx in c_remainder_start..chroma_width {
        let cb_val = i32::from(cb_row[cx]) - 128;
        let cr_val = i32::from(cr_row[cx]) - 128;

        let px0 = cx * 2;
        if px0 < width {
            let y_val = i32::from(y_row[px0]);
            let y_scaled = y_val * Y_CF_INT + YUV_ROUND;
            let r = (y_scaled + cr_val * CR_TO_R_INT) >> 14;
            let g = (y_scaled + cr_val * CR_TO_G_INT + cb_val * CB_TO_G_INT) >> 14;
            let b = (y_scaled + cb_val * CB_TO_B_INT) >> 14;
            let idx = px0 * 3;
            rgb[idx] = r.clamp(0, 255) as u8;
            rgb[idx + 1] = g.clamp(0, 255) as u8;
            rgb[idx + 2] = b.clamp(0, 255) as u8;
        }
        let px1 = cx * 2 + 1;
        if px1 < width {
            let y_val = i32::from(y_row[px1]);
            let y_scaled = y_val * Y_CF_INT + YUV_ROUND;
            let r = (y_scaled + cr_val * CR_TO_R_INT) >> 14;
            let g = (y_scaled + cr_val * CR_TO_G_INT + cb_val * CB_TO_G_INT) >> 14;
            let b = (y_scaled + cb_val * CB_TO_B_INT) >> 14;
            let idx = px1 * 3;
            rgb[idx] = r.clamp(0, 255) as u8;
            rgb[idx + 1] = g.clamp(0, 255) as u8;
            rgb[idx + 2] = b.clamp(0, 255) as u8;
        }
    }
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

    #[cfg(feature = "decoder")]
    #[test]
    fn test_rgb_u8_swap_rb_inplace() {
        let mut data = vec![10, 20, 30, 40, 50, 60]; // 2 pixels
        rgb_u8_swap_rb_inplace(&mut data);
        assert_eq!(data, [30, 20, 10, 60, 50, 40]);
    }

    #[cfg(feature = "decoder")]
    #[test]
    fn test_rgb_u8_to_rgba_u8() {
        let src = [10, 20, 30, 40, 50, 60]; // 2 pixels
        let mut dst = [0u8; 8];
        rgb_u8_to_rgba_u8(&src, &mut dst);
        assert_eq!(dst, [10, 20, 30, 255, 40, 50, 60, 255]);
    }

    #[cfg(feature = "decoder")]
    #[test]
    fn test_rgb_u8_to_bgra_u8() {
        let src = [10, 20, 30, 40, 50, 60]; // 2 RGB pixels
        let mut dst = [0u8; 8];
        rgb_u8_to_bgra_u8(&src, &mut dst);
        // R=10,G=20,B=30 → B=30,G=20,R=10,A=255
        assert_eq!(dst, [30, 20, 10, 255, 60, 50, 40, 255]);
    }

    #[cfg(feature = "decoder")]
    #[test]
    fn test_rgb_u8_to_bgrx_u8() {
        let src = [10, 20, 30];
        let mut dst = [0u8; 4];
        rgb_u8_to_bgrx_u8(&src, &mut dst);
        assert_eq!(dst, [30, 20, 10, 255]);
    }

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

        let (y, _cb, _cr) = rgb_to_ycbcr_f32(0.0, 255.0, 0.0); // Green
        assert!((y - 150.0).abs() < 1.0);

        let (y, _cb, _cr) = rgb_to_ycbcr_f32(0.0, 0.0, 255.0); // Blue
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
        let (_c, _m, _y, k) = rgb_to_cmyk(0, 0, 0);
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

    #[test]
    fn test_ycbcr_to_rgb_i16_scalar() {
        // Test that integer path matches f32 path (within tolerance)
        let test_cases = [
            (128i16, 128i16, 128i16), // Gray
            (76i16, 85i16, 255i16),   // Red
            (150i16, 44i16, 21i16),   // Green
            (29i16, 255i16, 107i16),  // Blue
        ];

        for (y, cb, cr) in test_cases {
            let (r_f32, g_f32, b_f32) = ycbcr_to_rgb(y as u8, cb as u8, cr as u8);

            // Test scalar integer path
            let y_arr = [y; 16];
            let cb_arr = [cb; 16];
            let cr_arr = [cr; 16];
            let mut rgb = vec![0u8; 48];
            let mut offset = 0;
            ycbcr_to_rgb_i16_x16_scalar(&y_arr, &cb_arr, &cr_arr, &mut rgb, &mut offset);

            // Allow ±2 difference due to rounding
            assert!(
                (rgb[0] as i16 - r_f32 as i16).abs() <= 2,
                "R mismatch: {} vs {} for Y={}, Cb={}, Cr={}",
                rgb[0],
                r_f32,
                y,
                cb,
                cr
            );
            assert!(
                (rgb[1] as i16 - g_f32 as i16).abs() <= 2,
                "G mismatch: {} vs {} for Y={}, Cb={}, Cr={}",
                rgb[1],
                g_f32,
                y,
                cb,
                cr
            );
            assert!(
                (rgb[2] as i16 - b_f32 as i16).abs() <= 2,
                "B mismatch: {} vs {} for Y={}, Cb={}, Cr={}",
                rgb[2],
                b_f32,
                y,
                cb,
                cr
            );
        }
    }

    #[test]
    fn test_ycbcr_planes_i16_to_rgb_u8() {
        // Test batch conversion matches scalar
        let y_plane: Vec<i16> = (0..32).map(|i| 128 + (i % 5) as i16).collect();
        let cb_plane: Vec<i16> = (0..32).map(|i| 128 + (i % 3) as i16).collect();
        let cr_plane: Vec<i16> = (0..32).map(|i| 128 + (i % 7) as i16).collect();

        let mut rgb = vec![0u8; 96];
        ycbcr_planes_i16_to_rgb_u8(&y_plane, &cb_plane, &cr_plane, &mut rgb);

        // Verify against scalar f32 conversion
        for i in 0..32 {
            let (r_ref, g_ref, b_ref) =
                ycbcr_to_rgb(y_plane[i] as u8, cb_plane[i] as u8, cr_plane[i] as u8);

            // Allow ±2 difference
            assert!(
                (rgb[i * 3] as i16 - r_ref as i16).abs() <= 2,
                "R mismatch at {}: {} vs {}",
                i,
                rgb[i * 3],
                r_ref
            );
            assert!(
                (rgb[i * 3 + 1] as i16 - g_ref as i16).abs() <= 2,
                "G mismatch at {}: {} vs {}",
                i,
                rgb[i * 3 + 1],
                g_ref
            );
            assert!(
                (rgb[i * 3 + 2] as i16 - b_ref as i16).abs() <= 2,
                "B mismatch at {}: {} vs {}",
                i,
                rgb[i * 3 + 2],
                b_ref
            );
        }
    }

    #[test]
    fn test_fused_hfancy_avx2_matches_scalar() {
        // Test that AVX2 h-fancy kernel produces identical output to scalar.
        // Use widths that exercise: full AVX2 chunks, remainder, edge cases.
        for width in [16, 32, 48, 64, 100, 128, 255, 256, 300, 512] {
            let chroma_width = (width + 1) / 2;
            // Varied chroma values to exercise interpolation
            let y_row: Vec<i16> = (0..width).map(|i| 16 + (i as i16 * 7) % 220).collect();
            let cb_row: Vec<i16> = (0..chroma_width)
                .map(|i| 30 + (i as i16 * 13) % 200)
                .collect();
            let cr_row: Vec<i16> = (0..chroma_width)
                .map(|i| 50 + (i as i16 * 11) % 180)
                .collect();

            // Scalar reference
            let mut rgb_scalar = vec![0u8; width * 3];
            {
                for cx in 0..chroma_width {
                    let curr_cb = i32::from(cb_row[cx]);
                    let curr_cr = i32::from(cr_row[cx]);
                    let left_cb = if cx > 0 {
                        i32::from(cb_row[cx - 1])
                    } else {
                        curr_cb
                    };
                    let left_cr = if cx > 0 {
                        i32::from(cr_row[cx - 1])
                    } else {
                        curr_cr
                    };
                    let cb_l = ((3 * curr_cb + left_cb + 2) >> 2) - 128;
                    let cr_l = ((3 * curr_cr + left_cr + 2) >> 2) - 128;
                    let px0 = cx * 2;
                    if px0 < width {
                        let y_val = i32::from(y_row[px0]);
                        let y_scaled = y_val * Y_CF_INT + YUV_ROUND;
                        let r = (y_scaled + cr_l * CR_TO_R_INT) >> 14;
                        let g = (y_scaled + cr_l * CR_TO_G_INT + cb_l * CB_TO_G_INT) >> 14;
                        let b = (y_scaled + cb_l * CB_TO_B_INT) >> 14;
                        let idx = px0 * 3;
                        rgb_scalar[idx] = r.clamp(0, 255) as u8;
                        rgb_scalar[idx + 1] = g.clamp(0, 255) as u8;
                        rgb_scalar[idx + 2] = b.clamp(0, 255) as u8;
                    }
                    let right_cb = if cx + 1 < chroma_width {
                        i32::from(cb_row[cx + 1])
                    } else {
                        curr_cb
                    };
                    let right_cr = if cx + 1 < chroma_width {
                        i32::from(cr_row[cx + 1])
                    } else {
                        curr_cr
                    };
                    let cb_r = ((3 * curr_cb + right_cb + 2) >> 2) - 128;
                    let cr_r = ((3 * curr_cr + right_cr + 2) >> 2) - 128;
                    let px1 = cx * 2 + 1;
                    if px1 < width {
                        let y_val = i32::from(y_row[px1]);
                        let y_scaled = y_val * Y_CF_INT + YUV_ROUND;
                        let r = (y_scaled + cr_r * CR_TO_R_INT) >> 14;
                        let g = (y_scaled + cr_r * CR_TO_G_INT + cb_r * CB_TO_G_INT) >> 14;
                        let b = (y_scaled + cb_r * CB_TO_B_INT) >> 14;
                        let idx = px1 * 3;
                        rgb_scalar[idx] = r.clamp(0, 255) as u8;
                        rgb_scalar[idx + 1] = g.clamp(0, 255) as u8;
                        rgb_scalar[idx + 2] = b.clamp(0, 255) as u8;
                    }
                }
            }

            // Fused function (dispatches to AVX2 on x86_64)
            let mut rgb_fused = vec![0u8; width * 3];
            fused_h2v2_hfancy_ycbcr_to_rgb_u8(&y_row, &cb_row, &cr_row, &mut rgb_fused, width);

            assert_eq!(
                rgb_scalar,
                rgb_fused,
                "Mismatch at width={width}: first diff at pixel {}",
                rgb_scalar
                    .iter()
                    .zip(rgb_fused.iter())
                    .position(|(a, b)| a != b)
                    .unwrap_or(0)
                    / 3
            );
        }
    }
}
