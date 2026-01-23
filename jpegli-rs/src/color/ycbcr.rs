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

use multiversed::multiversed;
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
/// Uses SIMD for YCbCr math with efficient interleaved RGB storage.
#[multiversed]
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

    // Process chunks of 8 pixels
    // Use chunks_exact for optimal iteration
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
        // Load planes - chunks_exact guarantees exactly 8 elements
        let y = f32x8::from(<[f32; 8]>::try_from(y_chunk).unwrap());
        let cb = f32x8::from(<[f32; 8]>::try_from(cb_chunk).unwrap());
        let cr = f32x8::from(<[f32; 8]>::try_from(cr_chunk).unwrap());

        // YCbCr to RGB (using FMA)
        let r = cr_to_r.mul_add(cr, y + offset).max(zero).min(max_val);
        let g = cb_to_g
            .mul_add(cb, cr_to_g.mul_add(cr, y + offset))
            .max(zero)
            .min(max_val);
        let b = cb_to_b.mul_add(cb, y + offset).max(zero).min(max_val);

        // Convert to arrays for interleaved store
        let r_arr: [f32; 8] = r.into();
        let g_arr: [f32; 8] = g.into();
        let b_arr: [f32; 8] = b.into();

        // Store interleaved RGB - slice is guaranteed to be exactly 24 bytes
        rgb_chunk[0] = r_arr[0] as u8;
        rgb_chunk[1] = g_arr[0] as u8;
        rgb_chunk[2] = b_arr[0] as u8;
        rgb_chunk[3] = r_arr[1] as u8;
        rgb_chunk[4] = g_arr[1] as u8;
        rgb_chunk[5] = b_arr[1] as u8;
        rgb_chunk[6] = r_arr[2] as u8;
        rgb_chunk[7] = g_arr[2] as u8;
        rgb_chunk[8] = b_arr[2] as u8;
        rgb_chunk[9] = r_arr[3] as u8;
        rgb_chunk[10] = g_arr[3] as u8;
        rgb_chunk[11] = b_arr[3] as u8;
        rgb_chunk[12] = r_arr[4] as u8;
        rgb_chunk[13] = g_arr[4] as u8;
        rgb_chunk[14] = b_arr[4] as u8;
        rgb_chunk[15] = r_arr[5] as u8;
        rgb_chunk[16] = g_arr[5] as u8;
        rgb_chunk[17] = b_arr[5] as u8;
        rgb_chunk[18] = r_arr[6] as u8;
        rgb_chunk[19] = g_arr[6] as u8;
        rgb_chunk[20] = b_arr[6] as u8;
        rgb_chunk[21] = r_arr[7] as u8;
        rgb_chunk[22] = g_arr[7] as u8;
        rgb_chunk[23] = b_arr[7] as u8;
    }

    // Handle remaining pixels with scalar code
    let chunks_processed = (num_pixels / 8) * 8;
    let rgb_start = chunks_processed * 3;
    for (i, ((y, cb), cr)) in y_remainder
        .iter()
        .zip(cb_remainder.iter())
        .zip(cr_remainder.iter())
        .enumerate()
    {
        // Use FMA for scalar remainder
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
            let y = f32x8::from(<[f32; 8]>::try_from(&y_plane[base..base + 8]).unwrap());
            let cb = f32x8::from(<[f32; 8]>::try_from(&cb_plane[base..base + 8]).unwrap());
            let cr = f32x8::from(<[f32; 8]>::try_from(&cr_plane[base..base + 8]).unwrap());

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

    {
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

        // Remainder
        for i in (chunks * 8)..num_pixels {
            let val = (y_plane[i] + 128.0).clamp(0.0, 255.0) as u8;
            let idx = i * 3;
            rgb[idx] = val;
            rgb[idx + 1] = val;
            rgb[idx + 2] = val;
        }
    }

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

    {
        let offset = f32x8::splat(128.0);
        let scale = f32x8::splat(1.0 / 255.0);
        let zero = f32x8::splat(0.0);
        let one = f32x8::splat(1.0);

        let chunks = num_pixels / 8;
        for chunk in 0..chunks {
            let base = chunk * 8;
            let y = f32x8::from(<[f32; 8]>::try_from(&y_plane[base..base + 8]).unwrap());

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

    {
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

        // Remainder
        for i in (chunks * 8)..num_pixels {
            output[i] = (y_plane[i] + 128.0).clamp(0.0, 255.0) as u8;
        }
    }

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

    {
        let offset = f32x8::splat(128.0);
        let scale = f32x8::splat(1.0 / 255.0);
        let zero = f32x8::splat(0.0);
        let one = f32x8::splat(1.0);

        let chunks = num_pixels / 8;
        for chunk in 0..chunks {
            let base = chunk * 8;
            let y = f32x8::from(<[f32; 8]>::try_from(&y_plane[base..base + 8]).unwrap());

            let val = ((y + offset) * scale).max(zero).min(one);
            let arr: [f32; 8] = val.into();
            output[base..base + 8].copy_from_slice(&arr);
        }

        // Remainder
        for i in (chunks * 8)..num_pixels {
            output[i] = ((y_plane[i] + 128.0) / 255.0).clamp(0.0, 1.0);
        }
    }

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
    #[cfg(all(
        feature = "unsafe_simd",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    {
        if is_x86_feature_detected!("avx2") {
            // Safety: we just checked for AVX2 support
            unsafe {
                ycbcr_to_rgb_i16_x16_avx2(y, cb, cr, rgb, offset);
            }
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
#[cfg(all(
    feature = "unsafe_simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx2")]
unsafe fn ycbcr_to_rgb_i16_x16_avx2(
    y: &[i16; 16],
    cb: &[i16; 16],
    cr: &[i16; 16],
    rgb: &mut [u8],
    offset: &mut usize,
) {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    // Load Y, Cb, Cr (16 i16 values each)
    let y_vec = _mm256_loadu_si256(y.as_ptr().cast());
    let cb_vec = _mm256_loadu_si256(cb.as_ptr().cast());
    let cr_vec = _mm256_loadu_si256(cr.as_ptr().cast());

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
    let out_ptr = rgb.as_mut_ptr().add(*offset);
    _mm256_storeu_si256(out_ptr.cast(), rgb0);
    _mm_storeu_si128(out_ptr.add(32).cast(), _mm256_castsi256_si128(rgb1));

    *offset += 48;
}

/// Autovectorized YCbCr to separate R, G, B planes.
///
/// This function is decorated with `#[multiversion]` to generate optimized versions
/// for different SIMD instruction sets (AVX2, SSE4.1, NEON) with runtime dispatch.
/// Writing to separate planes allows better autovectorization than interleaved output.
#[multiversion::multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "x86_64+sse4.1", "aarch64+neon"))]
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
        r_out[i] = r_raw.max(0).min(255) as u8;
        g_out[i] = g_raw.max(0).min(255) as u8;
        b_out[i] = b_raw.max(0).min(255) as u8;
    }
}

/// Interleave R, G, B planes into RGB buffer.
#[multiversion::multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "x86_64+sse4.1", "aarch64+neon"))]
fn interleave_rgb_planes(
    r: &[u8],
    g: &[u8],
    b: &[u8],
    rgb: &mut [u8],
) {
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

    // Use AVX2 SIMD path when available (16 pixels at a time, direct interleaved output)
    #[cfg(all(
        feature = "unsafe_simd",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    {
        if is_x86_feature_detected!("avx2") {
            // Safety: AVX2 feature detected, pointers are valid for the slice lengths
            unsafe {
                ycbcr_planes_i16_to_rgb_u8_avx2(y_plane, cb_plane, cr_plane, rgb);
            }
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
#[cfg(all(
    feature = "unsafe_simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx2")]
unsafe fn ycbcr_planes_i16_to_rgb_u8_avx2(
    y_plane: &[i16],
    cb_plane: &[i16],
    cr_plane: &[i16],
    rgb: &mut [u8],
) {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    let len = y_plane.len();
    let chunks = len / 16;

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
        0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5,
        0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5,
    );
    let sh_g = _mm256_setr_epi8(
        5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10,
        5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10,
    );
    let sh_b = _mm256_setr_epi8(
        10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15,
        10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15,
    );
    let m0 = _mm256_setr_epi8(
        0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0,
        0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0,
    );
    let m1 = _mm256_setr_epi8(
        0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0,
        0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0,
    );

    let y_ptr = y_plane.as_ptr();
    let cb_ptr = cb_plane.as_ptr();
    let cr_ptr = cr_plane.as_ptr();
    let rgb_ptr = rgb.as_mut_ptr();

    for chunk in 0..chunks {
        let in_offset = chunk * 16;
        let out_offset = chunk * 48;

        // Load directly from pointers
        let y_vec = _mm256_loadu_si256(y_ptr.add(in_offset).cast());
        let cb_vec = _mm256_loadu_si256(cb_ptr.add(in_offset).cast());
        let cr_vec = _mm256_loadu_si256(cr_ptr.add(in_offset).cast());

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

        // Store 48 bytes
        let out_ptr = rgb_ptr.add(out_offset);
        _mm256_storeu_si256(out_ptr.cast(), rgb0);
        _mm_storeu_si128(out_ptr.add(32).cast(), _mm256_castsi256_si128(rgb1));
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
}
