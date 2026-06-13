//! Color space conversion functions.
//!
//! This module provides conversions between:
//! - RGB and YCbCr (BT.601 standard JPEG color space)
//! - RGB and CMYK
//! - Various pixel format conversions
//!
//! SIMD optimization via archmage/magetypes generics with multi-tier dispatch.

use crate::error::Result;
use crate::foundation::alloc::{checked_size, checked_size_2d, try_alloc_zeroed};
use crate::foundation::consts::{
    YCBCR_B_TO_CB, YCBCR_B_TO_CR, YCBCR_B_TO_Y, YCBCR_CB_TO_B, YCBCR_CB_TO_G, YCBCR_CB_TO_R,
    YCBCR_CR_TO_B, YCBCR_CR_TO_G, YCBCR_CR_TO_R, YCBCR_G_TO_CB, YCBCR_G_TO_CR, YCBCR_G_TO_Y,
    YCBCR_R_TO_CB, YCBCR_R_TO_CR, YCBCR_R_TO_Y, YCBCR_Y_TO_B, YCBCR_Y_TO_G, YCBCR_Y_TO_R,
};
use crate::types::PixelFormat;

use archmage::prelude::*;
use magetypes::simd::generic::f32x8 as GenericF32x8;
use magetypes::simd::generic::i32x4 as GenericI32x4;

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

// Scalar color conversion for 4-pixel batches (used by plane conversion functions).
mod simd {
    use super::*;

    /// Process 4 RGB pixels to YCbCr using scalar FMA.
    /// Returns (Y[4], Cb[4], Cr[4]) as u8 arrays.
    #[inline]
    pub fn rgb_to_ycbcr_x4(r: [u8; 4], g: [u8; 4], b: [u8; 4]) -> ([u8; 4], [u8; 4], [u8; 4]) {
        let clamp = |v: f32| v.round().clamp(0.0, 255.0) as u8;

        let mut y_out = [0u8; 4];
        let mut cb_out = [0u8; 4];
        let mut cr_out = [0u8; 4];

        for i in 0..4 {
            let rf = r[i] as f32;
            let gf = g[i] as f32;
            let bf = b[i] as f32;

            let y = YCBCR_R_TO_Y.mul_add(rf, YCBCR_G_TO_Y.mul_add(gf, YCBCR_B_TO_Y * bf));
            let cb = YCBCR_R_TO_CB.mul_add(
                rf,
                YCBCR_G_TO_CB.mul_add(gf, YCBCR_B_TO_CB.mul_add(bf, 128.0)),
            );
            let cr = YCBCR_R_TO_CR.mul_add(
                rf,
                YCBCR_G_TO_CR.mul_add(gf, YCBCR_B_TO_CR.mul_add(bf, 128.0)),
            );

            y_out[i] = clamp(y);
            cb_out[i] = clamp(cb);
            cr_out[i] = clamp(cr);
        }

        (y_out, cb_out, cr_out)
    }

    /// Process 4 YCbCr pixels to RGB using scalar FMA.
    #[inline]
    pub fn ycbcr_to_rgb_x4(y: [u8; 4], cb: [u8; 4], cr: [u8; 4]) -> ([u8; 4], [u8; 4], [u8; 4]) {
        let clamp = |v: f32| v.round().clamp(0.0, 255.0) as u8;

        let mut r_out = [0u8; 4];
        let mut g_out = [0u8; 4];
        let mut b_out = [0u8; 4];

        for i in 0..4 {
            let yf = y[i] as f32;
            let cbf = cb[i] as f32 - 128.0;
            let crf = cr[i] as f32 - 128.0;

            let r = YCBCR_Y_TO_R.mul_add(yf, YCBCR_CB_TO_R.mul_add(cbf, YCBCR_CR_TO_R * crf));
            let g = YCBCR_Y_TO_G.mul_add(yf, YCBCR_CB_TO_G.mul_add(cbf, YCBCR_CR_TO_G * crf));
            let b = YCBCR_Y_TO_B.mul_add(yf, YCBCR_CB_TO_B.mul_add(cbf, YCBCR_CR_TO_B * crf));

            r_out[i] = clamp(r);
            g_out[i] = clamp(g);
            b_out[i] = clamp(b);
        }

        (r_out, g_out, b_out)
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
/// Uses `incant!` for multi-tier SIMD dispatch (AVX2+FMA, NEON, WASM128, scalar).
pub fn ycbcr_planes_f32_to_rgb_u8(
    y_plane: &[f32],
    cb_plane: &[f32],
    cr_plane: &[f32],
    rgb: &mut [u8],
) {
    incant!(ycbcr_planes_f32_to_rgb_u8_impl(
        y_plane, cb_plane, cr_plane, rgb
    ));
}

#[magetypes(v3, neon, wasm128, scalar)]
#[inline(always)]
fn ycbcr_planes_f32_to_rgb_u8_impl(
    token: Token,
    y_plane: &[f32],
    cb_plane: &[f32],
    cr_plane: &[f32],
    rgb: &mut [u8],
) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;

    debug_assert_eq!(y_plane.len(), cb_plane.len());
    debug_assert_eq!(y_plane.len(), cr_plane.len());
    debug_assert_eq!(rgb.len(), y_plane.len() * 3);

    let num_pixels = y_plane.len();

    // BT.601 coefficients
    let cr_to_r = f32x8::splat(token, 1.402);
    let cb_to_g = f32x8::splat(token, -0.344136);
    let cr_to_g = f32x8::splat(token, -0.714136);
    let cb_to_b = f32x8::splat(token, 1.772);
    let offset = f32x8::splat(token, 128.0);
    let zero = f32x8::splat(token, 0.0);
    let max_val = f32x8::splat(token, 255.0);

    let chunks = num_pixels / 8;

    for chunk in 0..chunks {
        let i = chunk * 8;

        // Load 8 values from each plane
        let y = f32x8::from_array(token, *<&[f32; 8]>::try_from(&y_plane[i..i + 8]).unwrap());
        let cb = f32x8::from_array(token, *<&[f32; 8]>::try_from(&cb_plane[i..i + 8]).unwrap());
        let cr = f32x8::from_array(token, *<&[f32; 8]>::try_from(&cr_plane[i..i + 8]).unwrap());

        let y_off = y + offset;

        // YCbCr to RGB with FMA
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

/// Batch YCbCr to RGB conversion for f32 planes to f32 output.
///
/// Input: centered YCbCr (Y, Cb, Cr all centered around 0 from f32 IDCT).
/// Output: RGB normalized to approximately 0.0-1.0 range. Values may slightly
/// exceed [0, 1] due to YCbCr→RGB color matrix expansion — this is intentional
/// to preserve full precision. Callers should clamp only at final output if needed.
///
/// Uses `incant!` for multi-tier SIMD dispatch (AVX2+FMA, NEON, WASM128, scalar).
pub fn ycbcr_planes_f32_to_rgb_f32(
    y_plane: &[f32],
    cb_plane: &[f32],
    cr_plane: &[f32],
    rgb: &mut [f32],
) {
    incant!(ycbcr_planes_f32_to_rgb_f32_impl(
        y_plane, cb_plane, cr_plane, rgb
    ));
}

#[magetypes(v3, neon, wasm128, scalar)]
#[inline(always)]
fn ycbcr_planes_f32_to_rgb_f32_impl(
    token: Token,
    y_plane: &[f32],
    cb_plane: &[f32],
    cr_plane: &[f32],
    rgb: &mut [f32],
) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;

    debug_assert_eq!(y_plane.len(), cb_plane.len());
    debug_assert_eq!(y_plane.len(), cr_plane.len());
    debug_assert_eq!(rgb.len(), y_plane.len() * 3);

    let num_pixels = y_plane.len();

    let cr_to_r = f32x8::splat(token, 1.402);
    let cb_to_g = f32x8::splat(token, -0.344136);
    let cr_to_g = f32x8::splat(token, -0.714136);
    let cb_to_b = f32x8::splat(token, 1.772);
    let offset = f32x8::splat(token, 128.0);
    let scale = f32x8::splat(token, 1.0 / 255.0);

    let chunks = num_pixels / 8;
    for chunk in 0..chunks {
        let base = chunk * 8;

        let y = f32x8::from_array(
            token,
            *<&[f32; 8]>::try_from(&y_plane[base..base + 8]).unwrap(),
        );
        let cb = f32x8::from_array(
            token,
            *<&[f32; 8]>::try_from(&cb_plane[base..base + 8]).unwrap(),
        );
        let cr = f32x8::from_array(
            token,
            *<&[f32; 8]>::try_from(&cr_plane[base..base + 8]).unwrap(),
        );

        let y_off = y + offset;

        // YCbCr to RGB with FMA, level shift, normalize — no clamping
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

/// Batch grayscale to RGB conversion for f32 to u8.
///
/// Uses `incant!` for multi-tier SIMD dispatch (AVX2, NEON, WASM128, scalar).
pub fn gray_f32_to_rgb_u8(y_plane: &[f32], rgb: &mut [u8]) {
    incant!(gray_f32_to_rgb_u8_impl(y_plane, rgb));
}

#[magetypes(v3, neon, wasm128, scalar)]
#[inline(always)]
fn gray_f32_to_rgb_u8_impl(token: Token, y_plane: &[f32], rgb: &mut [u8]) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;

    debug_assert_eq!(rgb.len(), y_plane.len() * 3);

    let num_pixels = y_plane.len();
    let offset = f32x8::splat(token, 128.0);
    let zero = f32x8::splat(token, 0.0);
    let max_val = f32x8::splat(token, 255.0);

    let chunks = num_pixels / 8;
    for chunk in 0..chunks {
        let base = chunk * 8;
        let y = f32x8::from_array(
            token,
            *<&[f32; 8]>::try_from(&y_plane[base..base + 8]).unwrap(),
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

/// Batch grayscale to RGB conversion for f32 to f32.
///
/// Input: centered grayscale (Y centered around 0 from f32 IDCT).
/// Output: normalized to approximately 0.0-1.0 range without clamping.
///
/// Uses `incant!` for multi-tier SIMD dispatch (AVX2, NEON, WASM128, scalar).
pub fn gray_f32_to_rgb_f32(y_plane: &[f32], rgb: &mut [f32]) {
    incant!(gray_f32_to_rgb_f32_impl(y_plane, rgb));
}

#[magetypes(v3, neon, wasm128, scalar)]
#[inline(always)]
fn gray_f32_to_rgb_f32_impl(token: Token, y_plane: &[f32], rgb: &mut [f32]) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;

    debug_assert_eq!(rgb.len(), y_plane.len() * 3);

    let num_pixels = y_plane.len();
    let offset = f32x8::splat(token, 128.0);
    let scale = f32x8::splat(token, 1.0 / 255.0);

    let chunks = num_pixels / 8;
    for chunk in 0..chunks {
        let base = chunk * 8;
        let y = f32x8::from_array(
            token,
            *<&[f32; 8]>::try_from(&y_plane[base..base + 8]).unwrap(),
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

/// Batch level shift for grayscale f32 to u8.
///
/// Uses `incant!` for multi-tier SIMD dispatch (AVX2, NEON, WASM128, scalar).
pub fn gray_f32_to_gray_u8(y_plane: &[f32], output: &mut [u8]) {
    incant!(gray_f32_to_gray_u8_impl(y_plane, output));
}

#[magetypes(v3, neon, wasm128, scalar)]
#[inline(always)]
fn gray_f32_to_gray_u8_impl(token: Token, y_plane: &[f32], output: &mut [u8]) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;

    debug_assert_eq!(y_plane.len(), output.len());

    let num_pixels = y_plane.len();
    let offset = f32x8::splat(token, 128.0);
    let zero = f32x8::splat(token, 0.0);
    let max_val = f32x8::splat(token, 255.0);

    let chunks = num_pixels / 8;
    for chunk in 0..chunks {
        let base = chunk * 8;
        let y = f32x8::from_array(
            token,
            *<&[f32; 8]>::try_from(&y_plane[base..base + 8]).unwrap(),
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

/// Batch level shift for grayscale f32 to f32 (approximately 0.0-1.0).
///
/// Input: centered grayscale (Y centered around 0 from f32 IDCT).
/// Output: normalized without clamping to preserve full precision.
///
/// Uses `incant!` for multi-tier SIMD dispatch (AVX2, NEON, WASM128, scalar).
pub fn gray_f32_to_gray_f32(y_plane: &[f32], output: &mut [f32]) {
    incant!(gray_f32_to_gray_f32_impl(y_plane, output));
}

#[magetypes(v3, neon, wasm128, scalar)]
#[inline(always)]
fn gray_f32_to_gray_f32_impl(token: Token, y_plane: &[f32], output: &mut [f32]) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;

    debug_assert_eq!(y_plane.len(), output.len());

    let num_pixels = y_plane.len();
    let offset = f32x8::splat(token, 128.0);
    let scale = f32x8::splat(token, 1.0 / 255.0);

    let chunks = num_pixels / 8;
    for chunk in 0..chunks {
        let base = chunk * 8;
        let y = f32x8::from_array(
            token,
            *<&[f32; 8]>::try_from(&y_plane[base..base + 8]).unwrap(),
        );

        let val = (y + offset) * scale;
        let arr = val.to_array();
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
pub fn rgb_u8_swap_rb_inplace(data: &mut [u8]) {
    debug_assert_eq!(data.len() % 3, 0);
    for pixel in data.chunks_exact_mut(3) {
        pixel.swap(0, 2);
    }
}

/// Converts packed RGB u8 to packed RGBA u8 (alpha = 255).
///
/// `src.len()` must be a multiple of 3 and `dst.len() >= src.len() / 3 * 4`.
pub fn rgb_u8_to_rgba_u8(src: &[u8], dst: &mut [u8]) {
    rgb_u8_to_xrgba_u8(src, dst, false);
}

/// Converts packed RGB u8 to packed BGRA u8 (alpha = 255, R/B swapped).
///
/// `src.len()` must be a multiple of 3 and `dst.len() >= src.len() / 3 * 4`.
pub fn rgb_u8_to_bgra_u8(src: &[u8], dst: &mut [u8]) {
    rgb_u8_to_xrgba_u8(src, dst, true);
}

/// Shared RGB→RGBA/BGRA implementation with optional R/B swap.
///
/// Delegates to `garb`'s SIMD kernels (AVX2 8 pixels/iter, NEON, WASM128, scalar).
pub fn rgb_u8_to_xrgba_u8(src: &[u8], dst: &mut [u8], swap_rb: bool) {
    debug_assert_eq!(src.len() % 3, 0);
    let npixels = src.len() / 3;
    debug_assert!(dst.len() >= npixels * 4);

    if swap_rb {
        garb::bytes::rgb_to_bgra(src, &mut dst[..npixels * 4]).expect("pre-validated sizes");
    } else {
        garb::bytes::rgb_to_rgba(src, &mut dst[..npixels * 4]).expect("pre-validated sizes");
    }
}

/// Converts packed RGB u8 to packed BGRX u8 (pad = 255, R/B swapped).
///
/// Identical to [`rgb_u8_to_bgra_u8`] — the pad byte is set to 255.
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

/// Batch convert CMYK planes (Adobe format) to interleaved raw CMYK.
///
/// Each plane contains values in Adobe inverted format (0 = full ink).
/// Output is interleaved CMYK bytes (4 bytes per pixel), preserving
/// the inverted byte values as-is for downstream ICC-based conversion.
pub fn cmyk_planes_to_cmyk_u8(
    c_plane: &[f32],
    m_plane: &[f32],
    y_plane: &[f32],
    k_plane: &[f32],
    cmyk: &mut [u8],
) {
    debug_assert_eq!(c_plane.len(), m_plane.len());
    debug_assert_eq!(c_plane.len(), y_plane.len());
    debug_assert_eq!(c_plane.len(), k_plane.len());
    debug_assert_eq!(cmyk.len(), c_plane.len() * 4);

    for i in 0..c_plane.len() {
        // Level shift and clamp to 0-255 range
        let c = (c_plane[i] + 128.0).round().clamp(0.0, 255.0) as u8;
        let m = (m_plane[i] + 128.0).round().clamp(0.0, 255.0) as u8;
        let y = (y_plane[i] + 128.0).round().clamp(0.0, 255.0) as u8;
        let k = (k_plane[i] + 128.0).round().clamp(0.0, 255.0) as u8;

        cmyk[i * 4] = c;
        cmyk[i * 4 + 1] = m;
        cmyk[i * 4 + 2] = y;
        cmyk[i * 4 + 3] = k;
    }
}

/// Batch convert YCCK planes to interleaved raw CMYK.
///
/// Takes Y, Cb, Cr, K planes (f32, centered at 0) and outputs raw CMYK bytes.
/// YCbCr channels are converted to CMY via standard YCbCr→RGB, producing
/// CMY values. K is level-shifted and passed through. Output is interleaved
/// CMYK bytes (4 bytes per pixel) in the same inverted format as
/// [`cmyk_planes_to_cmyk_u8`].
pub fn ycck_planes_to_cmyk_u8(
    y_plane: &[f32],
    cb_plane: &[f32],
    cr_plane: &[f32],
    k_plane: &[f32],
    cmyk: &mut [u8],
) {
    debug_assert_eq!(y_plane.len(), cb_plane.len());
    debug_assert_eq!(y_plane.len(), cr_plane.len());
    debug_assert_eq!(y_plane.len(), k_plane.len());
    debug_assert_eq!(cmyk.len(), y_plane.len() * 4);

    for i in 0..y_plane.len() {
        // Level shift and clamp
        let y = (y_plane[i] + 128.0).round().clamp(0.0, 255.0) as u8;
        let cb = (cb_plane[i] + 128.0).round().clamp(0.0, 255.0) as u8;
        let cr = (cr_plane[i] + 128.0).round().clamp(0.0, 255.0) as u8;
        let k = (k_plane[i] + 128.0).round().clamp(0.0, 255.0) as u8;

        // YCbCr→RGB converts the encoded channels back to "RGB-like" values
        // in direct format: 0 = no ink, 255 = full ink.
        // Adobe/libjpeg CMYK convention is inverted: 0 = full ink, 255 = no ink.
        // Invert each CMY channel to match. K is already inverted in YCCK.
        let (c, m, yy) = ycbcr_to_rgb(y, cb, cr);

        cmyk[i * 4] = 255 - c;
        cmyk[i * 4 + 1] = 255 - m;
        cmyk[i * 4 + 2] = 255 - yy;
        cmyk[i * 4 + 3] = k; // K is already in inverted format
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

// libjpeg-turbo 16-bit (SCALEBITS=16) YCbCr→RGB coefficients, matching
// jdcolor.c `build_ycc_rgb_table` exactly. Selected by IdctMethod::Libjpeg
// for byte-exact mozjpeg/djpeg parity (vs the default zune-style 14-bit set
// above, which differs by ±1 on ~0.15% of the G/B channels). The extra two
// bits of constant precision are the entire residual — chroma upsampling and
// the IDCT are already bit-exact under that method. FIX(x) = round(x·2^16);
// the +128 level-shifted Y enters at unit weight (×2^16) so it cancels the
// final >>16, matching turbo's `Y + ((const·chroma + ONE_HALF) >> 16)`.
const TURBO_Y_CF: i32 = 65536; // 1 << 16
const TURBO_CR_TO_R: i32 = 91881; // FIX(1.40200)
const TURBO_CB_TO_B: i32 = 116130; // FIX(1.77200)
const TURBO_CR_TO_G: i32 = -46802; // -FIX(0.71414)
const TURBO_CB_TO_G: i32 = -22554; // -FIX(0.34414)
const TURBO_ROUND: i32 = 32768; // ONE_HALF = 1 << 15

/// Default-vs-turbo YCbCr→RGB fixed-point constant set, returned as
/// `(y_coeff, round, cr_to_r, cr_to_g, cb_to_g, cb_to_b, shift)`. `turbo`
/// selects libjpeg-turbo's 16-bit table constants (byte-exact mozjpeg
/// parity, IdctMethod::Libjpeg); otherwise the default zune-style 14-bit
/// set. The scalar fallbacks call this; the SIMD kernels select inline on
/// their `const TURBO` parameter (so it const-folds — `TURBO == false`
/// compiles to byte-identical code to the pre-turbo kernels).
#[inline(always)]
const fn ycc_consts(turbo: bool) -> (i32, i32, i32, i32, i32, i32, u32) {
    if turbo {
        (
            TURBO_Y_CF,
            TURBO_ROUND,
            TURBO_CR_TO_R,
            TURBO_CR_TO_G,
            TURBO_CB_TO_G,
            TURBO_CB_TO_B,
            16,
        )
    } else {
        (
            Y_CF_INT,
            YUV_ROUND,
            CR_TO_R_INT,
            CR_TO_G_INT,
            CB_TO_G_INT,
            CB_TO_B_INT,
            14,
        )
    }
}

/// Scalar YCbCr→RGB for one pixel (default 14-bit or turbo 16-bit per
/// `turbo`). `cb_c`/`cr_c` are already centered (sample − 128).
#[inline(always)]
fn ycc_rgb_pixel(y: i32, cb_c: i32, cr_c: i32, turbo: bool) -> (u8, u8, u8) {
    let (yc, rnd, crr, crg, cbg, cbb, sh) = ycc_consts(turbo);
    let ys = y * yc + rnd;
    let r = ((ys + cr_c * crr) >> sh).clamp(0, 255) as u8;
    let g = ((ys + cr_c * crg + cb_c * cbg) >> sh).clamp(0, 255) as u8;
    let b = ((ys + cb_c * cbb) >> sh).clamp(0, 255) as u8;
    (r, g, b)
}

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
    // Magetypes generic: 2x 8-pixel passes via i32x4
    incant!(ycbcr_to_rgb_i16_x8_generic(y, cb, cr, rgb, *offset));
    incant!(ycbcr_to_rgb_i16_x8_generic(
        &y[8..],
        &cb[8..],
        &cr[8..],
        rgb,
        *offset + 24
    ));
    *offset += 48;
}

/// Scalar implementation of integer YCbCr to RGB for 16 pixels.
#[cfg(test)]
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

/// Magetypes-generic integer YCbCr to RGB for 8 pixels via i32x4.
///
/// Processes 8 pixels in two 4-wide i32x4 passes. Uses the same 14-bit
/// fixed-point math as the scalar and AVX2 versions. Generic across
/// all platforms (x86 AVX2, NEON, WASM128, scalar).
#[magetypes(v3, neon, wasm128, scalar)]
#[inline(always)]
fn ycbcr_to_rgb_i16_x8_generic(
    token: Token,
    y: &[i16],
    cb: &[i16],
    cr: &[i16],
    rgb: &mut [u8],
    base: usize,
) {
    #[allow(non_camel_case_types)]
    type i32x4 = GenericI32x4<Token>;

    let y_coeff = i32x4::splat(token, Y_CF_INT);
    let rounding = i32x4::splat(token, YUV_ROUND);
    let bias = i32x4::splat(token, 128);
    let zero = i32x4::zero(token);
    let max255 = i32x4::splat(token, 255);

    let cr_to_r = i32x4::splat(token, CR_TO_R_INT);
    let cr_to_g = i32x4::splat(token, CR_TO_G_INT);
    let cb_to_g = i32x4::splat(token, CB_TO_G_INT);
    let cb_to_b = i32x4::splat(token, CB_TO_B_INT);

    // Process 8 pixels in two 4-wide passes
    for half in 0..2 {
        let off = half * 4;
        let y4 = i32x4::from_array(
            token,
            [
                i32::from(y[off]),
                i32::from(y[off + 1]),
                i32::from(y[off + 2]),
                i32::from(y[off + 3]),
            ],
        );
        let cb4 = i32x4::from_array(
            token,
            [
                i32::from(cb[off]),
                i32::from(cb[off + 1]),
                i32::from(cb[off + 2]),
                i32::from(cb[off + 3]),
            ],
        ) - bias;
        let cr4 = i32x4::from_array(
            token,
            [
                i32::from(cr[off]),
                i32::from(cr[off + 1]),
                i32::from(cr[off + 2]),
                i32::from(cr[off + 3]),
            ],
        ) - bias;

        let y_scaled = y4 * y_coeff + rounding;

        let r = (y_scaled + cr4 * cr_to_r).shr_arithmetic::<14>();
        let g = (y_scaled + cr4 * cr_to_g + cb4 * cb_to_g).shr_arithmetic::<14>();
        let b = (y_scaled + cb4 * cb_to_b).shr_arithmetic::<14>();

        // Clamp to 0-255
        let r = r.max(zero).min(max255);
        let g = g.max(zero).min(max255);
        let b = b.max(zero).min(max255);

        // Extract and interleave RGB
        let ra = r.to_array();
        let ga = g.to_array();
        let ba = b.to_array();

        let idx = base + off * 3;
        for i in 0..4 {
            rgb[idx + i * 3] = ra[i] as u8;
            rgb[idx + i * 3 + 1] = ga[i] as u8;
            rgb[idx + i * 3 + 2] = ba[i] as u8;
        }
    }
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
#[allow(dead_code)] // Alternative codepath for plane-based output (not currently used)
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
#[allow(dead_code)] // Helper for ycbcr_to_rgb_planes_autovec (not currently used)
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
    turbo: bool,
) {
    debug_assert_eq!(y_plane.len(), cb_plane.len());
    debug_assert_eq!(y_plane.len(), cr_plane.len());
    debug_assert_eq!(rgb.len(), y_plane.len() * 3);

    let len = y_plane.len();

    // `turbo` selects libjpeg-turbo-exact 16-bit color (IdctMethod::Libjpeg)
    // vs the default zune-style 14-bit. The hand kernels are monomorphized on
    // it as `const TURBO`, so both modes share the same SIMD interleave and
    // `turbo == false` const-folds to the original code (byte-identical).
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(token) = archmage::X64V4Token::summon() {
            if turbo {
                ycbcr_planes_i16_to_rgb_u8_avx512::<true>(token, y_plane, cb_plane, cr_plane, rgb);
            } else {
                ycbcr_planes_i16_to_rgb_u8_avx512::<false>(token, y_plane, cb_plane, cr_plane, rgb);
            }
            return;
        }
        if let Some(token) = archmage::X64V3Token::summon() {
            if turbo {
                ycbcr_planes_i16_to_rgb_u8_avx2::<true>(token, y_plane, cb_plane, cr_plane, rgb);
            } else {
                ycbcr_planes_i16_to_rgb_u8_avx2::<false>(token, y_plane, cb_plane, cr_plane, rgb);
            }
            return;
        }
    }

    // Scalar fallback (non-x86, or x86 without AVX2).
    for i in 0..len {
        let (r, g, b) = ycc_rgb_pixel(
            i32::from(y_plane[i]),
            i32::from(cb_plane[i]) - 128,
            i32::from(cr_plane[i]) - 128,
            turbo,
        );
        let idx = i * 3;
        rgb[idx] = r;
        rgb[idx + 1] = g;
        rgb[idx + 2] = b;
    }
}

/// AVX2 batch conversion of YCbCr planes to interleaved RGB.
/// Processes 16 pixels at a time with direct pointer loads.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ycbcr_planes_i16_to_rgb_u8_avx2<const TURBO: bool>(
    _token: archmage::X64V3Token,
    y_plane: &[i16],
    cb_plane: &[i16],
    cr_plane: &[i16],
    rgb: &mut [u8],
) {
    use core::arch::x86_64::*;

    let len = y_plane.len();

    // `TURBO` selects libjpeg-turbo's 16-bit constants + a >>16 descale;
    // it const-folds, so `TURBO == false` is byte-identical to the 14-bit
    // original. `srai!` picks the shift the same way.
    macro_rules! srai {
        ($x:expr) => {{
            if TURBO {
                _mm256_srai_epi32($x, 16)
            } else {
                _mm256_srai_epi32($x, 14)
            }
        }};
    }

    // Preload constants outside the loop
    let bias = _mm256_set1_epi16(128);
    let y_coeff = _mm256_set1_epi32(if TURBO { TURBO_Y_CF } else { Y_CF_INT });
    let rounding = _mm256_set1_epi32(if TURBO { TURBO_ROUND } else { YUV_ROUND });
    let cr_to_r = _mm256_set1_epi32(if TURBO { TURBO_CR_TO_R } else { CR_TO_R_INT });
    let cr_to_g = _mm256_set1_epi32(if TURBO { TURBO_CR_TO_G } else { CR_TO_G_INT });
    let cb_to_g = _mm256_set1_epi32(if TURBO { TURBO_CB_TO_G } else { CB_TO_G_INT });
    let cb_to_b = _mm256_set1_epi32(if TURBO { TURBO_CB_TO_B } else { CB_TO_B_INT });
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
        let r_lo = srai!(_mm256_add_epi32(
            y_scaled_lo,
            _mm256_mullo_epi32(cr_lo, cr_to_r)
        ));
        let r_hi = srai!(_mm256_add_epi32(
            y_scaled_hi,
            _mm256_mullo_epi32(cr_hi, cr_to_r)
        ));

        // G = (y_scaled + cr * CR_TO_G + cb * CB_TO_G) >> 14
        let g_lo = srai!(_mm256_add_epi32(
            y_scaled_lo,
            _mm256_add_epi32(
                _mm256_mullo_epi32(cr_lo, cr_to_g),
                _mm256_mullo_epi32(cb_lo, cb_to_g),
            ),
        ));
        let g_hi = srai!(_mm256_add_epi32(
            y_scaled_hi,
            _mm256_add_epi32(
                _mm256_mullo_epi32(cr_hi, cr_to_g),
                _mm256_mullo_epi32(cb_hi, cb_to_g),
            ),
        ));

        // B = (y_scaled + cb * CB_TO_B) >> 14
        let b_lo = srai!(_mm256_add_epi32(
            y_scaled_lo,
            _mm256_mullo_epi32(cb_lo, cb_to_b)
        ));
        let b_hi = srai!(_mm256_add_epi32(
            y_scaled_hi,
            _mm256_mullo_epi32(cb_hi, cb_to_b)
        ));

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
        let (r, g, b) = ycc_rgb_pixel(
            i32::from(y_plane[i]),
            i32::from(cb_plane[i]) - 128,
            i32::from(cr_plane[i]) - 128,
            TURBO,
        );

        let idx = i * 3;
        rgb[idx] = r;
        rgb[idx + 1] = g;
        rgb[idx + 2] = b;
    }
}

/// Batch convert i16 YCbCr planes to interleaved RGBA/BGRA u8 with A=255.
///
/// Dispatches to AVX2 where available, scalar fallback otherwise.
/// `swap_rb=false` writes R,G,B,255; `swap_rb=true` writes B,G,R,255.
pub fn ycbcr_planes_i16_to_xrgba_u8(
    y_plane: &[i16],
    cb_plane: &[i16],
    cr_plane: &[i16],
    rgba: &mut [u8],
    swap_rb: bool,
    turbo: bool,
) {
    debug_assert_eq!(y_plane.len(), cb_plane.len());
    debug_assert_eq!(y_plane.len(), cr_plane.len());
    debug_assert_eq!(rgba.len(), y_plane.len() * 4);

    // `turbo` selects libjpeg-turbo 16-bit color; the avx2 kernel is
    // monomorphized on it (const-folds, default byte-identical).
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(token) = archmage::X64V3Token::summon() {
            if turbo {
                ycbcr_planes_i16_to_xrgba_u8_avx2::<true>(
                    token, y_plane, cb_plane, cr_plane, rgba, swap_rb,
                );
            } else {
                ycbcr_planes_i16_to_xrgba_u8_avx2::<false>(
                    token, y_plane, cb_plane, cr_plane, rgba, swap_rb,
                );
            }
            return;
        }
    }

    let len = y_plane.len();
    for i in 0..len {
        let (r, g, b) = ycc_rgb_pixel(
            i32::from(y_plane[i]),
            i32::from(cb_plane[i]) - 128,
            i32::from(cr_plane[i]) - 128,
            turbo,
        );
        let idx = i * 4;
        if swap_rb {
            rgba[idx] = b;
            rgba[idx + 1] = g;
            rgba[idx + 2] = r;
        } else {
            rgba[idx] = r;
            rgba[idx + 1] = g;
            rgba[idx + 2] = b;
        }
        rgba[idx + 3] = 255;
    }
}

/// AVX2 batch conversion of YCbCr planes to interleaved 4bpp (RGBA or BGRA).
///
/// Processes 16 pixels at a time. The YCbCr→RGB compute reuses the same lane
/// layout as [`ycbcr_planes_i16_to_rgb_u8_avx2`]; the final packing uses 128-bit
/// unpacks instead of the 3-byte shuffle sequence, which is cheaper for 4bpp.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ycbcr_planes_i16_to_xrgba_u8_avx2<const TURBO: bool>(
    _token: archmage::X64V3Token,
    y_plane: &[i16],
    cb_plane: &[i16],
    cr_plane: &[i16],
    rgba: &mut [u8],
    swap_rb: bool,
) {
    use core::arch::x86_64::*;

    macro_rules! srai {
        ($x:expr) => {{
            if TURBO {
                _mm256_srai_epi32($x, 16)
            } else {
                _mm256_srai_epi32($x, 14)
            }
        }};
    }

    let len = y_plane.len();

    let bias = _mm256_set1_epi16(128);
    let y_coeff = _mm256_set1_epi32(if TURBO { TURBO_Y_CF } else { Y_CF_INT });
    let rounding = _mm256_set1_epi32(if TURBO { TURBO_ROUND } else { YUV_ROUND });
    let cr_to_r = _mm256_set1_epi32(if TURBO { TURBO_CR_TO_R } else { CR_TO_R_INT });
    let cr_to_g = _mm256_set1_epi32(if TURBO { TURBO_CR_TO_G } else { CR_TO_G_INT });
    let cb_to_g = _mm256_set1_epi32(if TURBO { TURBO_CB_TO_G } else { CB_TO_G_INT });
    let cb_to_b = _mm256_set1_epi32(if TURBO { TURBO_CB_TO_B } else { CB_TO_B_INT });
    let zero = _mm256_setzero_si256();
    let alpha_sse = _mm_set1_epi8(-1_i8); // 0xFF

    // Process 16 pixels per iteration → 64 output bytes.
    let y_chunks = y_plane.chunks_exact(16);
    let remainder_len = y_chunks.remainder().len();
    for ((y_chunk, cb_chunk), (cr_chunk, out_chunk)) in y_chunks
        .zip(cb_plane.chunks_exact(16))
        .zip(cr_plane.chunks_exact(16).zip(rgba.chunks_exact_mut(64)))
    {
        let y_vec = safe_simd::_mm256_loadu_si256(<&[i16; 16]>::try_from(y_chunk).unwrap());
        let cb_vec = safe_simd::_mm256_loadu_si256(<&[i16; 16]>::try_from(cb_chunk).unwrap());
        let cr_vec = safe_simd::_mm256_loadu_si256(<&[i16; 16]>::try_from(cr_chunk).unwrap());

        let cb_centered = _mm256_sub_epi16(cb_vec, bias);
        let cr_centered = _mm256_sub_epi16(cr_vec, bias);

        // Widen Y to i32 (zero-extend; Y is non-negative after level shift).
        let y_lo = _mm256_unpacklo_epi16(y_vec, zero);
        let y_hi = _mm256_unpackhi_epi16(y_vec, zero);

        let y_scaled_lo = _mm256_add_epi32(_mm256_mullo_epi32(y_lo, y_coeff), rounding);
        let y_scaled_hi = _mm256_add_epi32(_mm256_mullo_epi32(y_hi, y_coeff), rounding);

        // Sign-extend Cb/Cr to i32.
        let cb_sign = _mm256_srai_epi16(cb_centered, 15);
        let cr_sign = _mm256_srai_epi16(cr_centered, 15);
        let cb_lo = _mm256_unpacklo_epi16(cb_centered, cb_sign);
        let cb_hi = _mm256_unpackhi_epi16(cb_centered, cb_sign);
        let cr_lo = _mm256_unpacklo_epi16(cr_centered, cr_sign);
        let cr_hi = _mm256_unpackhi_epi16(cr_centered, cr_sign);

        let r_lo = srai!(_mm256_add_epi32(
            y_scaled_lo,
            _mm256_mullo_epi32(cr_lo, cr_to_r)
        ));
        let r_hi = srai!(_mm256_add_epi32(
            y_scaled_hi,
            _mm256_mullo_epi32(cr_hi, cr_to_r)
        ));
        let g_lo = srai!(_mm256_add_epi32(
            y_scaled_lo,
            _mm256_add_epi32(
                _mm256_mullo_epi32(cr_lo, cr_to_g),
                _mm256_mullo_epi32(cb_lo, cb_to_g),
            ),
        ));
        let g_hi = srai!(_mm256_add_epi32(
            y_scaled_hi,
            _mm256_add_epi32(
                _mm256_mullo_epi32(cr_hi, cr_to_g),
                _mm256_mullo_epi32(cb_hi, cb_to_g),
            ),
        ));
        let b_lo = srai!(_mm256_add_epi32(
            y_scaled_lo,
            _mm256_mullo_epi32(cb_lo, cb_to_b)
        ));
        let b_hi = srai!(_mm256_add_epi32(
            y_scaled_hi,
            _mm256_mullo_epi32(cb_hi, cb_to_b)
        ));

        // Pack i32 → i16 (signed saturating). Per-lane packs give us:
        //   r_16 lane0 = [R0..R3 (from r_lo lo), R4..R7 (from r_hi lo)]  → [R0..R7]
        //   r_16 lane1 = [R8..R11 (from r_lo hi), R12..R15 (from r_hi hi)] → [R8..R15]
        // So r_16 is in natural order after packs_epi32.
        let r_16 = _mm256_packs_epi32(r_lo, r_hi);
        let g_16 = _mm256_packs_epi32(g_lo, g_hi);
        let b_16 = _mm256_packs_epi32(b_lo, b_hi);

        // Pack i16 → u8 (unsigned saturating). packus per-lane:
        //   lane0 of packus(x, zero) → [x.lane0 → 8 u8, 0..0]
        //   lane1 of packus(x, zero) → [x.lane1 → 8 u8, 0..0]
        // Then permute4x64 with [0, 2, 1, 3] puts the useful bytes in
        // the low 128 bits as [byte0..byte15] in natural order.
        let r_u8 = _mm256_permute4x64_epi64(_mm256_packus_epi16(r_16, zero), 0b11_01_10_00);
        let g_u8 = _mm256_permute4x64_epi64(_mm256_packus_epi16(g_16, zero), 0b11_01_10_00);
        let b_u8 = _mm256_permute4x64_epi64(_mm256_packus_epi16(b_16, zero), 0b11_01_10_00);

        // Now low 128 of r_u8 = [R0..R15]. Interleave via 128-bit unpacks.
        let r = _mm256_castsi256_si128(r_u8);
        let g = _mm256_castsi256_si128(g_u8);
        let b = _mm256_castsi256_si128(b_u8);

        // Choose first (R or B) and third (B or R) channels based on swap.
        let (first, third) = if swap_rb { (b, r) } else { (r, b) };

        // Interleave [first, G, third, 255] per pixel.
        let rg_lo = _mm_unpacklo_epi8(first, g); // F0 G0 F1 G1 ... F7 G7 (16 bytes)
        let rg_hi = _mm_unpackhi_epi8(first, g); // F8..F15
        let ba_lo = _mm_unpacklo_epi8(third, alpha_sse); // T0 A T1 A ... T7 A
        let ba_hi = _mm_unpackhi_epi8(third, alpha_sse); // T8 A ... T15 A

        let rgba0 = _mm_unpacklo_epi16(rg_lo, ba_lo); // 4 pixels (pixels 0-3)
        let rgba1 = _mm_unpackhi_epi16(rg_lo, ba_lo); // pixels 4-7
        let rgba2 = _mm_unpacklo_epi16(rg_hi, ba_hi); // pixels 8-11
        let rgba3 = _mm_unpackhi_epi16(rg_hi, ba_hi); // pixels 12-15

        let (p0, p1) = out_chunk.split_at_mut(32);
        let (p0a, p0b) = p0.split_at_mut(16);
        let (p1a, p1b) = p1.split_at_mut(16);
        safe_simd::_mm_storeu_si128(<&mut [u8; 16]>::try_from(p0a).unwrap(), rgba0);
        safe_simd::_mm_storeu_si128(<&mut [u8; 16]>::try_from(p0b).unwrap(), rgba1);
        safe_simd::_mm_storeu_si128(<&mut [u8; 16]>::try_from(p1a).unwrap(), rgba2);
        safe_simd::_mm_storeu_si128(<&mut [u8; 16]>::try_from(p1b).unwrap(), rgba3);
    }

    // Scalar remainder.
    let remainder_start = len - remainder_len;
    for i in remainder_start..len {
        let (r, g, b) = ycc_rgb_pixel(
            i32::from(y_plane[i]),
            i32::from(cb_plane[i]) - 128,
            i32::from(cr_plane[i]) - 128,
            TURBO,
        );
        let idx = i * 4;
        if swap_rb {
            rgba[idx] = b;
            rgba[idx + 1] = g;
            rgba[idx + 2] = r;
        } else {
            rgba[idx] = r;
            rgba[idx + 1] = g;
            rgba[idx + 2] = b;
        }
        rgba[idx + 3] = 255;
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
fn ycbcr_planes_i16_to_rgb_u8_avx512<const TURBO: bool>(
    _token: archmage::X64V4Token,
    y_plane: &[i16],
    cb_plane: &[i16],
    cr_plane: &[i16],
    rgb: &mut [u8],
) {
    use core::arch::x86_64::*;

    macro_rules! srai {
        ($x:expr) => {{
            if TURBO {
                _mm512_srai_epi32($x, 16)
            } else {
                _mm512_srai_epi32($x, 14)
            }
        }};
    }

    let len = y_plane.len();
    let chunks = len / 16;

    // Preload 512-bit constants (TURBO selects libjpeg 16-bit; const-folds).
    let y_coeff = _mm512_set1_epi32(if TURBO { TURBO_Y_CF } else { Y_CF_INT });
    let rounding = _mm512_set1_epi32(if TURBO { TURBO_ROUND } else { YUV_ROUND });
    let cr_to_r = _mm512_set1_epi32(if TURBO { TURBO_CR_TO_R } else { CR_TO_R_INT });
    let cr_to_g = _mm512_set1_epi32(if TURBO { TURBO_CR_TO_G } else { CR_TO_G_INT });
    let cb_to_g = _mm512_set1_epi32(if TURBO { TURBO_CB_TO_G } else { CB_TO_G_INT });
    let cb_to_b = _mm512_set1_epi32(if TURBO { TURBO_CB_TO_B } else { CB_TO_B_INT });
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
        let r_32 = srai!(_mm512_add_epi32(
            y_scaled,
            _mm512_mullo_epi32(cr_32, cr_to_r)
        ));

        // G = (y_scaled + cr * CR_TO_G + cb * CB_TO_G) >> 14
        let g_32 = srai!(_mm512_add_epi32(
            y_scaled,
            _mm512_add_epi32(
                _mm512_mullo_epi32(cr_32, cr_to_g),
                _mm512_mullo_epi32(cb_32, cb_to_g),
            ),
        ));

        // B = (y_scaled + cb * CB_TO_B) >> 14
        let b_32 = srai!(_mm512_add_epi32(
            y_scaled,
            _mm512_mullo_epi32(cb_32, cb_to_b)
        ));

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
        let (r, g, b) = ycc_rgb_pixel(
            i32::from(y_plane[i]),
            i32::from(cb_plane[i]) - 128,
            i32::from(cr_plane[i]) - 128,
            TURBO,
        );

        let idx = i * 3;
        rgb[idx] = r;
        rgb[idx + 1] = g;
        rgb[idx + 2] = b;
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
/// Magetypes-generic fused box upsample + YCbCr→RGB.
///
/// Processes 4 chroma pixels → 8 output pixels per i32x4 pass.
/// Each chroma value is duplicated horizontally (box filter).
#[magetypes(v3, neon, wasm128, scalar)]
#[inline(always)]
fn fused_h2v2_box_ycbcr_to_rgb_u8_generic(
    token: Token,
    y_row: &[i16],
    cb_row: &[i16],
    cr_row: &[i16],
    rgb: &mut [u8],
    width: usize,
) {
    #[allow(non_camel_case_types)]
    type i32x4 = GenericI32x4<Token>;

    let y_coeff = i32x4::splat(token, Y_CF_INT);
    let rounding = i32x4::splat(token, YUV_ROUND);
    let bias = i32x4::splat(token, 128);
    let zero = i32x4::zero(token);
    let max255 = i32x4::splat(token, 255);

    let cr_to_r = i32x4::splat(token, CR_TO_R_INT);
    let cr_to_g = i32x4::splat(token, CR_TO_G_INT);
    let cb_to_g = i32x4::splat(token, CB_TO_G_INT);
    let cb_to_b = i32x4::splat(token, CB_TO_B_INT);

    let chroma_width = (width + 1) / 2;
    // Process 4 chroma pixels (8 output pixels) at a time.
    // Last chunk needs px_base + 7 < width, so stop early for odd widths.
    let safe_chroma = if width >= 8 { (width - 7) / 2 } else { 0 };
    let chunks = safe_chroma / 4;
    for chunk in 0..chunks {
        let cx_base = chunk * 4;
        let cb4 = i32x4::from_array(
            token,
            [
                i32::from(cb_row[cx_base]),
                i32::from(cb_row[cx_base + 1]),
                i32::from(cb_row[cx_base + 2]),
                i32::from(cb_row[cx_base + 3]),
            ],
        ) - bias;
        let cr4 = i32x4::from_array(
            token,
            [
                i32::from(cr_row[cx_base]),
                i32::from(cr_row[cx_base + 1]),
                i32::from(cr_row[cx_base + 2]),
                i32::from(cr_row[cx_base + 3]),
            ],
        ) - bias;

        // Precompute color offsets (same for left and right output pixels)
        let cr_r = cr4 * cr_to_r;
        let cr_g = cr4 * cr_to_g;
        let cb_g = cb4 * cb_to_g;
        let cb_b = cb4 * cb_to_b;
        let chroma_g = cr_g + cb_g;

        // Left pixels (even indices)
        let px_base = cx_base * 2;
        let y_left = i32x4::from_array(
            token,
            [
                i32::from(y_row[px_base]),
                i32::from(y_row[px_base + 2]),
                i32::from(y_row[px_base + 4]),
                i32::from(y_row[px_base + 6]),
            ],
        );
        let ys_left = y_left * y_coeff + rounding;
        let rl = (ys_left + cr_r)
            .shr_arithmetic::<14>()
            .max(zero)
            .min(max255);
        let gl = (ys_left + chroma_g)
            .shr_arithmetic::<14>()
            .max(zero)
            .min(max255);
        let bl = (ys_left + cb_b)
            .shr_arithmetic::<14>()
            .max(zero)
            .min(max255);

        // Right pixels (odd indices)
        let y_right = i32x4::from_array(
            token,
            [
                i32::from(y_row[px_base + 1]),
                i32::from(y_row[px_base + 3]),
                i32::from(y_row[px_base + 5]),
                i32::from(y_row[px_base + 7]),
            ],
        );
        let ys_right = y_right * y_coeff + rounding;
        let rr = (ys_right + cr_r)
            .shr_arithmetic::<14>()
            .max(zero)
            .min(max255);
        let gr = (ys_right + chroma_g)
            .shr_arithmetic::<14>()
            .max(zero)
            .min(max255);
        let br = (ys_right + cb_b)
            .shr_arithmetic::<14>()
            .max(zero)
            .min(max255);

        // Interleave and store (left pixel, right pixel, left pixel, right pixel...)
        let rla = rl.to_array();
        let gla = gl.to_array();
        let bla = bl.to_array();
        let rra = rr.to_array();
        let gra = gr.to_array();
        let bra = br.to_array();

        for i in 0..4 {
            let idx = (px_base + i * 2) * 3;
            rgb[idx] = rla[i] as u8;
            rgb[idx + 1] = gla[i] as u8;
            rgb[idx + 2] = bla[i] as u8;
            rgb[idx + 3] = rra[i] as u8;
            rgb[idx + 4] = gra[i] as u8;
            rgb[idx + 5] = bra[i] as u8;
        }
    }

    // Scalar remainder
    for cx in (chunks * 4)..chroma_width {
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

/// `cr_row`: Cr values at half resolution (`width/2` elements)
/// `rgb`: Output RGB buffer (`width * 3` bytes)
/// `width`: Output width in pixels
pub fn fused_h2v2_box_ycbcr_to_rgb_u8(
    y_row: &[i16],
    cb_row: &[i16],
    cr_row: &[i16],
    rgb: &mut [u8],
    width: usize,
    turbo: bool,
) {
    debug_assert!(y_row.len() >= width);
    debug_assert!(cb_row.len() >= (width + 1) / 2);
    debug_assert!(cr_row.len() >= (width + 1) / 2);
    debug_assert!(rgb.len() >= width * 3);

    // x86: const-generic AVX2 for both modes (turbo const-folds, default
    // byte-identical). The hand kernel handles the box upsample + convert.
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(token) = archmage::X64V3Token::summon() {
            if turbo {
                fused_h2v2_box_ycbcr_to_rgb_u8_avx2::<true>(
                    token, y_row, cb_row, cr_row, rgb, width,
                );
            } else {
                fused_h2v2_box_ycbcr_to_rgb_u8_avx2::<false>(
                    token, y_row, cb_row, cr_row, rgb, width,
                );
            }
            return;
        }
    }

    if turbo {
        // Non-x86 turbo: scalar box upsample + convert. (Default non-x86 uses
        // the magetypes generic below; an ARM/WASM turbo SIMD path is a
        // follow-up — the broader non-x86 plane-converter cliff is tracked
        // separately.)
        for px in 0..width {
            let c = px / 2;
            let (r, g, b) = ycc_rgb_pixel(
                i32::from(y_row[px]),
                i32::from(cb_row[c]) - 128,
                i32::from(cr_row[c]) - 128,
                true,
            );
            let idx = px * 3;
            rgb[idx] = r;
            rgb[idx + 1] = g;
            rgb[idx + 2] = b;
        }
        return;
    }

    // Non-x86 default: magetypes generic (4 chroma px → 8 output px per pass).
    incant!(fused_h2v2_box_ycbcr_to_rgb_u8_generic(
        y_row, cb_row, cr_row, rgb, width
    ));
}

/// AVX2 fused box-filter 4:2:0 upsample + YCbCr→RGB.
/// Processes 16 output pixels per iteration (8 chroma pixels → 16 output pixels).
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn fused_h2v2_box_ycbcr_to_rgb_u8_avx2<const TURBO: bool>(
    _token: archmage::X64V3Token,
    y_row: &[i16],
    cb_row: &[i16],
    cr_row: &[i16],
    rgb: &mut [u8],
    width: usize,
) {
    use core::arch::x86_64::*;

    macro_rules! srai {
        ($x:expr) => {{
            if TURBO {
                _mm256_srai_epi32($x, 16)
            } else {
                _mm256_srai_epi32($x, 14)
            }
        }};
    }

    let chunks = width / 16;

    // Preload constants (TURBO selects libjpeg 16-bit; const-folds).
    let bias = _mm256_set1_epi16(128);
    let y_coeff = _mm256_set1_epi32(if TURBO { TURBO_Y_CF } else { Y_CF_INT });
    let rounding = _mm256_set1_epi32(if TURBO { TURBO_ROUND } else { YUV_ROUND });
    let cr_to_r = _mm256_set1_epi32(if TURBO { TURBO_CR_TO_R } else { CR_TO_R_INT });
    let cr_to_g = _mm256_set1_epi32(if TURBO { TURBO_CR_TO_G } else { CR_TO_G_INT });
    let cb_to_g = _mm256_set1_epi32(if TURBO { TURBO_CB_TO_G } else { CB_TO_G_INT });
    let cb_to_b = _mm256_set1_epi32(if TURBO { TURBO_CB_TO_B } else { CB_TO_B_INT });
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
        let r_lo = srai!(_mm256_add_epi32(
            y_scaled_lo,
            _mm256_mullo_epi32(cr_lo32, cr_to_r)
        ));
        let r_hi = srai!(_mm256_add_epi32(
            y_scaled_hi,
            _mm256_mullo_epi32(cr_hi32, cr_to_r)
        ));

        // G = (y_scaled + cr * CR_TO_G + cb * CB_TO_G) >> 14
        let g_lo = srai!(_mm256_add_epi32(
            y_scaled_lo,
            _mm256_add_epi32(
                _mm256_mullo_epi32(cr_lo32, cr_to_g),
                _mm256_mullo_epi32(cb_lo32, cb_to_g),
            ),
        ));
        let g_hi = srai!(_mm256_add_epi32(
            y_scaled_hi,
            _mm256_add_epi32(
                _mm256_mullo_epi32(cr_hi32, cr_to_g),
                _mm256_mullo_epi32(cb_hi32, cb_to_g),
            ),
        ));

        // B = (y_scaled + cb * CB_TO_B) >> 14
        let b_lo = srai!(_mm256_add_epi32(
            y_scaled_lo,
            _mm256_mullo_epi32(cb_lo32, cb_to_b)
        ));
        let b_hi = srai!(_mm256_add_epi32(
            y_scaled_hi,
            _mm256_mullo_epi32(cb_hi32, cb_to_b)
        ));

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
            let (r, g, b) = ycc_rgb_pixel(i32::from(y_row[px0]), cb_val, cr_val, TURBO);
            let idx = px0 * 3;
            rgb[idx] = r;
            rgb[idx + 1] = g;
            rgb[idx + 2] = b;
        }
        let px1 = cx * 2 + 1;
        if px1 < width {
            let (r, g, b) = ycc_rgb_pixel(i32::from(y_row[px1]), cb_val, cr_val, TURBO);
            let idx = px1 * 3;
            rgb[idx] = r;
            rgb[idx + 1] = g;
            rgb[idx + 2] = b;
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

    #[test]
    fn test_rgb_u8_swap_rb_inplace() {
        let mut data = vec![10, 20, 30, 40, 50, 60]; // 2 pixels
        rgb_u8_swap_rb_inplace(&mut data);
        assert_eq!(data, [30, 20, 10, 60, 50, 40]);
    }

    #[test]
    fn test_rgb_u8_to_rgba_u8() {
        let src = [10, 20, 30, 40, 50, 60]; // 2 pixels
        let mut dst = [0u8; 8];
        rgb_u8_to_rgba_u8(&src, &mut dst);
        assert_eq!(dst, [10, 20, 30, 255, 40, 50, 60, 255]);
    }

    #[test]
    fn test_rgb_u8_to_bgra_u8() {
        let src = [10, 20, 30, 40, 50, 60]; // 2 RGB pixels
        let mut dst = [0u8; 8];
        rgb_u8_to_bgra_u8(&src, &mut dst);
        // R=10,G=20,B=30 → B=30,G=20,R=10,A=255
        assert_eq!(dst, [30, 20, 10, 255, 60, 50, 40, 255]);
    }

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
        ycbcr_planes_i16_to_rgb_u8(&y_plane, &cb_plane, &cr_plane, &mut rgb, false);

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

    /// Pin the exact relationship between our 14-bit integer YCbCr→RGB and
    /// libjpeg-turbo's 16-bit table conversion (jdcolor.c
    /// `build_ycc_rgb_table`: per-table rounding at SCALEBITS=16) over the
    /// full 256³ input cube. Our 14-bit path (inherited from zune-jpeg)
    /// differs by at most ±1 per channel; the rate is reported. This — not
    /// the IDCT or the upsampler — is the residual ±1 in
    /// "box + IdctMethod::Libjpeg matches mozjpeg within max=1".
    #[test]
    fn int_ycbcr_vs_libjpeg_turbo_tables() {
        // turbo tables: FIX(x) = (x * 65536 + 0.5) as i32
        const FIX_1_40200: i64 = 91881;
        const FIX_1_77200: i64 = 116130;
        const FIX_0_71414: i64 = 46802;
        const FIX_0_34414: i64 = 22554;
        const ONE_HALF: i64 = 1 << 15;

        let mut cr_r = [0i32; 256];
        let mut cb_b = [0i32; 256];
        let mut cr_g = [0i64; 256];
        let mut cb_g = [0i64; 256];
        for i in 0..256 {
            let x = i as i64 - 128;
            cr_r[i] = ((FIX_1_40200 * x + ONE_HALF) >> 16) as i32;
            cb_b[i] = ((FIX_1_77200 * x + ONE_HALF) >> 16) as i32;
            cr_g[i] = -FIX_0_71414 * x;
            cb_g[i] = -FIX_0_34414 * x + ONE_HALF;
        }

        let mut diff_count = [0u64; 3];
        let mut max_diff = 0i32;
        let mut total = 0u64;

        for y in 0..256u16 {
            for cb in 0..256u16 {
                // Convert all 256 cr values in 16-wide batches.
                for cr_base in (0..256u16).step_by(16) {
                    let y_arr = [y as i16; 16];
                    let cb_arr = [cb as i16; 16];
                    let cr_arr: [i16; 16] = core::array::from_fn(|i| (cr_base as usize + i) as i16);
                    let mut rgb = [0u8; 48];
                    let mut offset = 0usize;
                    ycbcr_to_rgb_i16_x16(&y_arr, &cb_arr, &cr_arr, &mut rgb, &mut offset);

                    for i in 0..16 {
                        let cr = cr_base as usize + i;
                        let tr = (y as i32 + cr_r[cr]).clamp(0, 255);
                        let tg = (y as i32 + ((cb_g[cb as usize] + cr_g[cr]) >> 16) as i32)
                            .clamp(0, 255);
                        let tb = (y as i32 + cb_b[cb as usize]).clamp(0, 255);

                        for (ch, turbo_v) in [(0usize, tr), (1, tg), (2, tb)] {
                            let ours = rgb[i * 3 + ch] as i32;
                            let d = (ours - turbo_v).abs();
                            if d != 0 {
                                diff_count[ch] += 1;
                                max_diff = max_diff.max(d);
                            }
                        }
                        total += 1;
                    }
                }
            }
        }

        eprintln!(
            "int YCbCr→RGB vs libjpeg-turbo tables over {total} triples: \
             diff rate R={:.3}% G={:.3}% B={:.3}%, max diff {max_diff}",
            100.0 * diff_count[0] as f64 / total as f64,
            100.0 * diff_count[1] as f64 / total as f64,
            100.0 * diff_count[2] as f64 / total as f64
        );
        assert!(
            max_diff <= 1,
            "14-bit vs turbo 16-bit conversion must stay within ±1"
        );
    }

    /// libjpeg-turbo reference RGB for one centered triple (the exact table
    /// math from `int_ycbcr_vs_libjpeg_turbo_tables`).
    fn turbo_ref_rgb(y: i32, cb: i32, cr: i32) -> (u8, u8, u8) {
        const FIX_1_40200: i64 = 91881;
        const FIX_1_77200: i64 = 116130;
        const FIX_0_71414: i64 = 46802;
        const FIX_0_34414: i64 = 22554;
        const ONE_HALF: i64 = 1 << 15;
        let xr = cr as i64 - 128;
        let xb = cb as i64 - 128;
        let r = (y + ((FIX_1_40200 * xr + ONE_HALF) >> 16) as i32).clamp(0, 255) as u8;
        let g = (y + ((-FIX_0_71414 * xr + (-FIX_0_34414 * xb + ONE_HALF)) >> 16) as i32)
            .clamp(0, 255) as u8;
        let b = (y + ((FIX_1_77200 * xb + ONE_HALF) >> 16) as i32).clamp(0, 255) as u8;
        (r, g, b)
    }

    /// The turbo color converters (all three layout families, every SIMD
    /// tier) are bit-identical to libjpeg-turbo's table converter — 0 diffs,
    /// not ≤1. This is what makes IdctMethod::Libjpeg byte-exact with mozjpeg
    /// at the RGB level. Also pins SIMD dispatch parity (every tier ==
    /// scalar) via `for_each_token_permutation`.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn turbo_converters_match_libjpeg_tables() {
        use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

        // Pseudo-random planes spanning the full 0..=255 range, plus
        // structured edges, at a length that exercises SIMD body + remainder.
        let n = 67usize;
        let y: Vec<i16> = (0..n).map(|i| ((i * 37 + 11) % 256) as i16).collect();
        let cb: Vec<i16> = (0..n).map(|i| ((i * 53 + 200) % 256) as i16).collect();
        let cr: Vec<i16> = (0..n).map(|i| ((i * 29 + 7) % 256) as i16).collect();

        let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
            // Family 1: RGB plane.
            let mut rgb = vec![0u8; n * 3];
            ycbcr_planes_i16_to_rgb_u8(&y, &cb, &cr, &mut rgb, true);
            for i in 0..n {
                let (r, g, b) = turbo_ref_rgb(y[i] as i32, cb[i] as i32, cr[i] as i32);
                assert_eq!(
                    (rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]),
                    (r, g, b),
                    "turbo RGB plane != turbo tables at px {i} ({perm})"
                );
            }

            // Family 2: RGBA and BGRA.
            for swap in [false, true] {
                let mut rgba = vec![0u8; n * 4];
                ycbcr_planes_i16_to_xrgba_u8(&y, &cb, &cr, &mut rgba, swap, true);
                for i in 0..n {
                    let (r, g, b) = turbo_ref_rgb(y[i] as i32, cb[i] as i32, cr[i] as i32);
                    let (e0, e2) = if swap { (b, r) } else { (r, b) };
                    assert_eq!(
                        [
                            rgba[i * 4],
                            rgba[i * 4 + 1],
                            rgba[i * 4 + 2],
                            rgba[i * 4 + 3]
                        ],
                        [e0, g, e2, 255],
                        "turbo xrgba(swap={swap}) != turbo tables at px {i} ({perm})"
                    );
                }
            }

            // Family 3: fused box upsample (half-res chroma → 2× RGB).
            let cw = n.div_ceil(2);
            let cbh: Vec<i16> = (0..cw).map(|i| ((i * 53 + 200) % 256) as i16).collect();
            let crh: Vec<i16> = (0..cw).map(|i| ((i * 29 + 7) % 256) as i16).collect();
            let mut frgb = vec![0u8; n * 3];
            fused_h2v2_box_ycbcr_to_rgb_u8(&y, &cbh, &crh, &mut frgb, n, true);
            for px in 0..n {
                let c = px / 2;
                let (r, g, b) = turbo_ref_rgb(y[px] as i32, cbh[c] as i32, crh[c] as i32);
                assert_eq!(
                    (frgb[px * 3], frgb[px * 3 + 1], frgb[px * 3 + 2]),
                    (r, g, b),
                    "turbo fused-box != turbo tables at px {px} ({perm})"
                );
            }
        });
        eprintln!("turbo converter dispatch parity: {report}");
        assert!(report.permutations_run >= 2, "expected ≥2 SIMD tiers");
    }
}
