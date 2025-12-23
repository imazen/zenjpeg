//! Color space conversion functions.
//!
//! This module provides conversions between:
//! - RGB and YCbCr (BT.601 standard JPEG color space)
//! - RGB and CMYK
//! - Various pixel format conversions

use crate::consts::{
    YCBCR_B_TO_CB, YCBCR_B_TO_CR, YCBCR_B_TO_Y, YCBCR_CB_TO_B, YCBCR_CB_TO_G, YCBCR_CB_TO_R,
    YCBCR_CR_TO_B, YCBCR_CR_TO_G, YCBCR_CR_TO_R, YCBCR_G_TO_CB, YCBCR_G_TO_CR, YCBCR_G_TO_Y,
    YCBCR_R_TO_CB, YCBCR_R_TO_CR, YCBCR_R_TO_Y, YCBCR_Y_TO_B, YCBCR_Y_TO_G, YCBCR_Y_TO_R,
};
use crate::types::PixelFormat;

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

/// Converts RGB to separate Y, Cb, Cr planes.
pub fn rgb_to_ycbcr_planes(rgb: &[u8], width: usize, height: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let num_pixels = width * height;
    assert_eq!(rgb.len(), num_pixels * 3);

    let mut y_plane = vec![0u8; num_pixels];
    let mut cb_plane = vec![0u8; num_pixels];
    let mut cr_plane = vec![0u8; num_pixels];

    for i in 0..num_pixels {
        let (y, cb, cr) = rgb_to_ycbcr(rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]);
        y_plane[i] = y;
        cb_plane[i] = cb;
        cr_plane[i] = cr;
    }

    (y_plane, cb_plane, cr_plane)
}

/// Converts separate Y, Cb, Cr planes to RGB.
pub fn ycbcr_planes_to_rgb(
    y_plane: &[u8],
    cb_plane: &[u8],
    cr_plane: &[u8],
    width: usize,
    height: usize,
) -> Vec<u8> {
    let num_pixels = width * height;
    assert_eq!(y_plane.len(), num_pixels);
    assert_eq!(cb_plane.len(), num_pixels);
    assert_eq!(cr_plane.len(), num_pixels);

    let mut rgb = vec![0u8; num_pixels * 3];

    for i in 0..num_pixels {
        let (r, g, b) = ycbcr_to_rgb(y_plane[i], cb_plane[i], cr_plane[i]);
        rgb[i * 3] = r;
        rgb[i * 3 + 1] = g;
        rgb[i * 3 + 2] = b;
    }

    rgb
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
pub fn extract_channel(data: &[u8], format: PixelFormat, channel: usize) -> Vec<u8> {
    let bpp = format.bytes_per_pixel();
    let num_pixels = data.len() / bpp;
    let mut result = vec![0u8; num_pixels];

    for i in 0..num_pixels {
        result[i] = data[i * bpp + channel];
    }

    result
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
}
