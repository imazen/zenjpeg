//! Color space conversion functions.
//!
//! This module provides conversions between:
//! - RGB and YCbCr (BT.601 standard JPEG color space)
//! - RGB and CMYK
//! - Various pixel format conversions
//!
//! SIMD optimization is available via the `simd` feature (enabled by default).

use crate::jpegli::alloc::{checked_size, checked_size_2d, try_alloc_zeroed};
use crate::jpegli::consts::{
    YCBCR_B_TO_CB, YCBCR_B_TO_CR, YCBCR_B_TO_Y, YCBCR_CB_TO_B, YCBCR_CB_TO_G, YCBCR_CB_TO_R,
    YCBCR_CR_TO_B, YCBCR_CR_TO_G, YCBCR_CR_TO_R, YCBCR_G_TO_CB, YCBCR_G_TO_CR, YCBCR_G_TO_Y,
    YCBCR_R_TO_CB, YCBCR_R_TO_CR, YCBCR_R_TO_Y, YCBCR_Y_TO_B, YCBCR_Y_TO_G, YCBCR_Y_TO_R,
};
use crate::jpegli::error::Result;
use crate::jpegli::types::PixelFormat;

#[cfg(feature = "simd")]
use wide::f32x4;

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

    // Y = 0.299*R + 0.587*G + 0.114*B
    let y = YCBCR_R_TO_Y * rf + YCBCR_G_TO_Y * gf + YCBCR_B_TO_Y * bf;

    // Cb = 128 - 0.168736*R - 0.331264*G + 0.5*B
    let cb = 128.0 + YCBCR_R_TO_CB * rf + YCBCR_G_TO_CB * gf + YCBCR_B_TO_CB * bf;

    // Cr = 128 + 0.5*R - 0.418688*G - 0.081312*B
    let cr = 128.0 + YCBCR_R_TO_CR * rf + YCBCR_G_TO_CR * gf + YCBCR_B_TO_CR * bf;

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

    // R = Y + 1.402*Cr
    let r = YCBCR_Y_TO_R * yf + YCBCR_CB_TO_R * cbf + YCBCR_CR_TO_R * crf;

    // G = Y - 0.344136*Cb - 0.714136*Cr
    let g = YCBCR_Y_TO_G * yf + YCBCR_CB_TO_G * cbf + YCBCR_CR_TO_G * crf;

    // B = Y + 1.772*Cb
    let b = YCBCR_Y_TO_B * yf + YCBCR_CB_TO_B * cbf + YCBCR_CR_TO_B * crf;

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
    let y = YCBCR_R_TO_Y * r + YCBCR_G_TO_Y * g + YCBCR_B_TO_Y * b;
    let cb = 128.0 + YCBCR_R_TO_CB * r + YCBCR_G_TO_CB * g + YCBCR_B_TO_CB * b;
    let cr = 128.0 + YCBCR_R_TO_CR * r + YCBCR_G_TO_CR * g + YCBCR_B_TO_CR * b;
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

        // Compute Y, Cb, Cr
        let y = rf * r_to_y + gf * g_to_y + bf * b_to_y;
        let cb = offset_128 + rf * r_to_cb + gf * g_to_cb + bf * b_to_cb;
        let cr = offset_128 + rf * r_to_cr + gf * g_to_cr + bf * b_to_cr;

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

        // Compute R, G, B
        let r = yf * y_to_r + cbf * cb_to_r + crf * cr_to_r;
        let g = yf * y_to_g + cbf * cb_to_g + crf * cr_to_g;
        let b = yf * y_to_b + cbf * cb_to_b + crf * cr_to_b;

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
}
