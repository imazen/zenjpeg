//! Color space conversion for JPEG encoding
//!
//! Handles RGB to YCbCr conversion with proper JFIF/BT.601 coefficients.

/// Convert RGB pixels to YCbCr using JFIF/BT.601 coefficients
///
/// The conversion formula is:
/// - Y  =  0.299 * R + 0.587 * G + 0.114 * B
/// - Cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 128
/// - Cr =  0.5 * R - 0.418688 * G - 0.081312 * B + 128
#[inline]
pub fn rgb_to_ycbcr(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
    let r = r as f32;
    let g = g as f32;
    let b = b as f32;

    let y = 0.299 * r + 0.587 * g + 0.114 * b;
    let cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0;
    let cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0;

    (
        y.round().clamp(0.0, 255.0) as u8,
        cb.round().clamp(0.0, 255.0) as u8,
        cr.round().clamp(0.0, 255.0) as u8,
    )
}

/// Convert YCbCr to RGB (for verification/testing)
#[inline]
pub fn ycbcr_to_rgb(y: u8, cb: u8, cr: u8) -> (u8, u8, u8) {
    let y = y as f32;
    let cb = cb as f32 - 128.0;
    let cr = cr as f32 - 128.0;

    let r = y + 1.402 * cr;
    let g = y - 0.344136 * cb - 0.714136 * cr;
    let b = y + 1.772 * cb;

    (
        r.round().clamp(0.0, 255.0) as u8,
        g.round().clamp(0.0, 255.0) as u8,
        b.round().clamp(0.0, 255.0) as u8,
    )
}

/// Convert a buffer of RGB pixels to YCbCr
pub fn convert_rgb_to_ycbcr(rgb: &[u8], width: usize, height: usize) -> Vec<u8> {
    assert_eq!(rgb.len(), width * height * 3);

    let mut ycbcr = Vec::with_capacity(rgb.len());

    for chunk in rgb.chunks_exact(3) {
        let (y, cb, cr) = rgb_to_ycbcr(chunk[0], chunk[1], chunk[2]);
        ycbcr.push(y);
        ycbcr.push(cb);
        ycbcr.push(cr);
    }

    ycbcr
}

/// Extract Y, Cb, Cr planes from interleaved YCbCr data
pub fn deinterleave_ycbcr(
    ycbcr: &[u8],
    width: usize,
    height: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let pixel_count = width * height;
    let mut y_plane = Vec::with_capacity(pixel_count);
    let mut cb_plane = Vec::with_capacity(pixel_count);
    let mut cr_plane = Vec::with_capacity(pixel_count);

    for chunk in ycbcr.chunks_exact(3) {
        y_plane.push(chunk[0]);
        cb_plane.push(chunk[1]);
        cr_plane.push(chunk[2]);
    }

    (y_plane, cb_plane, cr_plane)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgb_ycbcr_roundtrip() {
        // Test several colors
        let colors = [
            (0, 0, 0),       // Black
            (255, 255, 255), // White
            (255, 0, 0),     // Red
            (0, 255, 0),     // Green
            (0, 0, 255),     // Blue
            (128, 128, 128), // Gray
        ];

        for (r, g, b) in colors {
            let (y, cb, cr) = rgb_to_ycbcr(r, g, b);
            let (r2, g2, b2) = ycbcr_to_rgb(y, cb, cr);

            // Allow ±1 due to rounding
            assert!((r as i16 - r2 as i16).abs() <= 1, "R: {} vs {}", r, r2);
            assert!((g as i16 - g2 as i16).abs() <= 1, "G: {} vs {}", g, g2);
            assert!((b as i16 - b2 as i16).abs() <= 1, "B: {} vs {}", b, b2);
        }
    }
}
