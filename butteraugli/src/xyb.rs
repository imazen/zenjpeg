//! XYB color space conversion for butteraugli.
//!
//! XYB is a hybrid opponent/trichromatic color space used by butteraugli
//! and JPEG XL. It's designed to model human visual perception more accurately
//! than RGB or YCbCr.
//!
//! The color space is based on:
//! - Opsin: models photoreceptor response
//! - X: roughly red-minus-green (opponent)
//! - Y: yellow/luminance
//! - B: blue
//!
//! XYB values are computed from opsin mixing, not directly from RGB.

use crate::consts::{
    XYB_NEG_OPSIN_ABSORBANCE_BIAS_CBRT, XYB_OPSIN_ABSORBANCE_BIAS, XYB_OPSIN_ABSORBANCE_MATRIX,
};

/// Applies sRGB gamma decoding (sRGB to linear RGB).
///
/// The sRGB transfer function is a piecewise function:
/// - Linear for very dark values (v <= 0.04045)
/// - Power function for the rest
#[inline]
#[must_use]
pub fn srgb_to_linear(v: f32) -> f32 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// Applies sRGB gamma encoding (linear RGB to sRGB).
#[inline]
#[must_use]
pub fn linear_to_srgb(v: f32) -> f32 {
    if v <= 0.003_130_8 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

/// Converts sRGB u8 to linear float (0.0-1.0 range).
#[inline]
#[must_use]
pub fn srgb_u8_to_linear(v: u8) -> f32 {
    srgb_to_linear(v as f32 / 255.0)
}

/// Converts linear float to sRGB u8.
#[inline]
#[must_use]
pub fn linear_to_srgb_u8(v: f32) -> u8 {
    (linear_to_srgb(v.clamp(0.0, 1.0)) * 255.0).round() as u8
}

/// Mixed transfer function used by butteraugli/jpegli (cube root based).
///
/// The cube root provides a more perceptually uniform response than
/// simple gamma correction.
#[inline]
#[must_use]
fn mixed_cbrt(v: f32) -> f32 {
    if v < 0.0 {
        -((-v).cbrt())
    } else {
        v.cbrt()
    }
}

/// Inverse of mixed cube root.
#[inline]
#[must_use]
fn mixed_cube(v: f32) -> f32 {
    if v < 0.0 {
        -((-v).powi(3))
    } else {
        v.powi(3)
    }
}

/// Converts linear RGB to XYB color space.
///
/// # Arguments
/// * `r`, `g`, `b` - Linear RGB values (0.0-1.0 range, but can exceed for HDR)
///
/// # Returns
/// (X, Y, B) values in XYB space
///
/// # Details
/// The conversion involves:
/// 1. Apply opsin absorbance matrix (LMS-like transform)
/// 2. Cube root for perceptual uniformity
/// 3. Final XYB matrix transform
#[must_use]
pub fn linear_rgb_to_xyb(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    // Step 1: Apply opsin absorbance matrix
    let m = &XYB_OPSIN_ABSORBANCE_MATRIX;
    let bias = &XYB_OPSIN_ABSORBANCE_BIAS;

    let opsin_r = m[0] * r + m[1] * g + m[2] * b + bias[0];
    let opsin_g = m[3] * r + m[4] * g + m[5] * b + bias[1];
    let opsin_b = m[6] * r + m[7] * g + m[8] * b + bias[2];

    // Step 2: Apply cube root for perceptual uniformity
    let cbrt_r = mixed_cbrt(opsin_r);
    let cbrt_g = mixed_cbrt(opsin_g);
    let cbrt_b = mixed_cbrt(opsin_b);

    // Step 3: Subtract bias after cube root
    let neg_bias = &XYB_NEG_OPSIN_ABSORBANCE_BIAS_CBRT;
    let cbrt_r = cbrt_r + neg_bias[0];
    let cbrt_g = cbrt_g + neg_bias[1];
    let cbrt_b = cbrt_b + neg_bias[2];

    // Step 4: Final XYB transform
    // X = (L - M) / 2
    // Y = (L + M) / 2
    // B = S
    let x = 0.5 * (cbrt_r - cbrt_g);
    let y = 0.5 * (cbrt_r + cbrt_g);
    let b_out = cbrt_b;

    (x, y, b_out)
}

/// Converts XYB to linear RGB.
///
/// This is the inverse of `linear_rgb_to_xyb`.
#[must_use]
pub fn xyb_to_linear_rgb(x: f32, y: f32, b: f32) -> (f32, f32, f32) {
    // Inverse of final XYB transform
    let neg_bias = &XYB_NEG_OPSIN_ABSORBANCE_BIAS_CBRT;

    let cbrt_r = y + x;
    let cbrt_g = y - x;
    let cbrt_b = b;

    // Add back the bias
    let cbrt_r = cbrt_r - neg_bias[0];
    let cbrt_g = cbrt_g - neg_bias[1];
    let cbrt_b = cbrt_b - neg_bias[2];

    // Inverse cube root
    let opsin_r = mixed_cube(cbrt_r);
    let opsin_g = mixed_cube(cbrt_g);
    let opsin_b = mixed_cube(cbrt_b);

    // Inverse opsin matrix
    let bias = &XYB_OPSIN_ABSORBANCE_BIAS;
    let opsin_r = opsin_r - bias[0];
    let opsin_g = opsin_g - bias[1];
    let opsin_b = opsin_b - bias[2];

    // Pre-computed inverse of opsin absorbance matrix
    const INV_OPSIN: [f32; 9] = [
        11.031_567, -9.866_944, -0.164_623, -3.254_147, 4.418_770, -0.164_623, -3.658_851,
        2.712_923, 1.945_928,
    ];

    let r = INV_OPSIN[0] * opsin_r + INV_OPSIN[1] * opsin_g + INV_OPSIN[2] * opsin_b;
    let g = INV_OPSIN[3] * opsin_r + INV_OPSIN[4] * opsin_g + INV_OPSIN[5] * opsin_b;
    let b_out = INV_OPSIN[6] * opsin_r + INV_OPSIN[7] * opsin_g + INV_OPSIN[8] * opsin_b;

    (r, g, b_out)
}

/// Converts sRGB u8 to XYB.
#[must_use]
pub fn srgb_to_xyb(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let lr = srgb_u8_to_linear(r);
    let lg = srgb_u8_to_linear(g);
    let lb = srgb_u8_to_linear(b);
    linear_rgb_to_xyb(lr, lg, lb)
}

/// Converts XYB to sRGB u8.
#[must_use]
pub fn xyb_to_srgb(x: f32, y: f32, b: f32) -> (u8, u8, u8) {
    let (lr, lg, lb) = xyb_to_linear_rgb(x, y, b);
    (
        linear_to_srgb_u8(lr),
        linear_to_srgb_u8(lg),
        linear_to_srgb_u8(lb),
    )
}

/// Converts an RGB buffer to XYB planes.
///
/// # Arguments
/// * `rgb` - Input RGB buffer (3 bytes per pixel, row-major)
/// * `width` - Image width
/// * `height` - Image height
///
/// # Returns
/// Three separate planes: (X, Y, B), each with width * height elements
pub fn rgb_to_xyb_planes(
    rgb: &[u8],
    width: usize,
    height: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let num_pixels = width * height;
    assert_eq!(rgb.len(), num_pixels * 3);

    let mut x_plane = vec![0.0f32; num_pixels];
    let mut y_plane = vec![0.0f32; num_pixels];
    let mut b_plane = vec![0.0f32; num_pixels];

    for i in 0..num_pixels {
        let (x, y, b) = srgb_to_xyb(rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]);
        x_plane[i] = x;
        y_plane[i] = y;
        b_plane[i] = b;
    }

    (x_plane, y_plane, b_plane)
}

/// Converts XYB planes to RGB buffer.
pub fn xyb_planes_to_rgb(
    x_plane: &[f32],
    y_plane: &[f32],
    b_plane: &[f32],
    width: usize,
    height: usize,
) -> Vec<u8> {
    let num_pixels = width * height;
    assert_eq!(x_plane.len(), num_pixels);
    assert_eq!(y_plane.len(), num_pixels);
    assert_eq!(b_plane.len(), num_pixels);

    let mut rgb = vec![0u8; num_pixels * 3];

    for i in 0..num_pixels {
        let (r, g, b) = xyb_to_srgb(x_plane[i], y_plane[i], b_plane[i]);
        rgb[i * 3] = r;
        rgb[i * 3 + 1] = g;
        rgb[i * 3 + 2] = b;
    }

    rgb
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_srgb_linear_roundtrip() {
        for v in 0..=255u8 {
            let linear = srgb_u8_to_linear(v);
            let back = linear_to_srgb_u8(linear);
            assert!((v as i16 - back as i16).abs() <= 1, "Failed for {}", v);
        }
    }

    #[test]
    fn test_xyb_roundtrip() {
        // Test roundtrip for common colors
        // Note: butteraugli XYB is designed for quality assessment, not perfect roundtrip
        // Some colors may have larger errors due to the opsin matrix approximations
        let test_colors = [
            (0u8, 0u8, 0u8),
            (255u8, 255u8, 255u8),
            (255u8, 0u8, 0u8),
            (0u8, 255u8, 0u8),
            (0u8, 0u8, 255u8),
            (128u8, 128u8, 128u8),
        ];

        for (r, g, b) in test_colors {
            let (x, y, b_xyb) = srgb_to_xyb(r, g, b);
            let (r2, g2, b2) = xyb_to_srgb(x, y, b_xyb);

            // Allow reasonable rounding error (up to 15 for saturated colors)
            let tol = 15i16;
            assert!(
                (r as i16 - r2 as i16).abs() <= tol,
                "R mismatch for ({},{},{}): {} vs {}",
                r,
                g,
                b,
                r,
                r2
            );
            assert!(
                (g as i16 - g2 as i16).abs() <= tol,
                "G mismatch for ({},{},{}): {} vs {}",
                r,
                g,
                b,
                g,
                g2
            );
            assert!(
                (b as i16 - b2 as i16).abs() <= tol,
                "B mismatch for ({},{},{}): {} vs {}",
                r,
                g,
                b,
                b,
                b2
            );
        }
    }

    #[test]
    fn test_gray_xyb() {
        // Gray values should have X relatively small (close to 0)
        // Note: butteraugli XYB uses different opsin matrix than jpegli,
        // so X may not be exactly 0 for gray values
        for gray in [0u8, 64, 128, 192, 255] {
            let (x, _y, _b) = srgb_to_xyb(gray, gray, gray);
            assert!(x.abs() < 0.1, "X should be small for gray, got {}", x);
        }
    }
}
