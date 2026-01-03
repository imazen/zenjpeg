//! XYB color space conversion.
//!
//! XYB is jpegli's perceptually optimized color space that provides better
//! compression quality compared to YCbCr for the same file size.
//!
//! The transform is:
//! 1. Linear RGB (gamma decoded)
//! 2. Apply opsin absorbance matrix (mix of LMS-like transform)
//! 3. Cube root for perceptual uniformity
//! 4. Final XYB matrix
//! 5. Scale for JPEG encoding (ScaleXYBRow)

use crate::consts::{
    XYB_NEG_OPSIN_ABSORBANCE_BIAS_CBRT, XYB_OPSIN_ABSORBANCE_BIAS, XYB_OPSIN_ABSORBANCE_MATRIX,
};

// Scaling constants for jpegli XYB encoding
// These are used after XYB conversion to map values to ranges suitable for JPEG quantization
pub const SCALED_XYB_OFFSET: [f32; 3] = [0.015_386_134, 0.0, 0.277_704_59];
pub const SCALED_XYB_SCALE: [f32; 3] = [22.995_788_804, 1.183_000_077, 1.502_141_333];

// ============================================================================
// Fast Gamma Conversion (from imageflow)
// ============================================================================
//
// For u8 input: use LUT (exact, fastest)
// For f32 input: use fastpow (approximate, ~5-10x faster than powf)

/// Pre-computed LUT for sRGB u8 → linear f32 conversion.
/// Exact values computed using the sRGB specification formula.
#[rustfmt::skip]
static SRGB_TO_LINEAR_LUT: [f32; 256] = [
    0.0, 0.000303526984, 0.000607053967, 0.000910580951, 0.00121410793, 0.00151763492, 0.0018211619, 0.00212468888,
    0.00242821587, 0.00273174285, 0.00303526984, 0.00334653576, 0.00367650732, 0.00402471702, 0.00439144204, 0.00477695348,
    0.0051815167, 0.00560539162, 0.00604883302, 0.00651209079, 0.00699541019, 0.00749903204, 0.00802319299, 0.00856812562,
    0.0091340587, 0.00972121732, 0.010329823, 0.010960094, 0.0116122452, 0.0122864884, 0.0129830323, 0.013702083,
    0.0144438436, 0.0152085144, 0.0159962934, 0.0168073758, 0.0176419545, 0.0185002201, 0.019382361, 0.0202885631,
    0.0212190104, 0.0221738848, 0.0231533662, 0.0241576324, 0.0251868596, 0.0262412219, 0.0273208916, 0.0284260395,
    0.0295568344, 0.0307134437, 0.0318960331, 0.0331047666, 0.0343398068, 0.0356013149, 0.0368894504, 0.0382043716,
    0.0395462353, 0.0409151969, 0.0423114106, 0.0437350293, 0.0451862044, 0.0466650863, 0.0481718242, 0.049706566,
    0.0512694584, 0.052860647, 0.0544802764, 0.05612849, 0.0578054302, 0.0595112382, 0.0612460542, 0.0630100177,
    0.0648032667, 0.0666259386, 0.0684781698, 0.0703600957, 0.0722718507, 0.0742135684, 0.0761853815, 0.0781874218,
    0.0802198203, 0.0822827071, 0.0843762115, 0.086500462, 0.0886555863, 0.0908417112, 0.0930589628, 0.0953074666,
    0.0975873471, 0.0998987282, 0.102241733, 0.104616484, 0.107023103, 0.109461711, 0.111932428, 0.114435374,
    0.116970668, 0.119538428, 0.122138772, 0.124771818, 0.12743768, 0.130136477, 0.132868322, 0.13563333,
    0.138431615, 0.141263291, 0.144128471, 0.147027266, 0.14995979, 0.152926152, 0.155926464, 0.158960835,
    0.162029376, 0.165132195, 0.1682694, 0.171441101, 0.174647404, 0.177888416, 0.181164244, 0.184474995,
    0.187820772, 0.191201683, 0.19461783, 0.19806932, 0.201556254, 0.205078736, 0.20863687, 0.212230757,
    0.2158605, 0.2195262, 0.223227957, 0.226965874, 0.230740049, 0.234550582, 0.238397574, 0.242281122,
    0.246201327, 0.250158285, 0.254152094, 0.258182853, 0.262250658, 0.266355605, 0.270497791, 0.274677312,
    0.278894263, 0.28314874, 0.287440838, 0.29177065, 0.296138271, 0.300543794, 0.304987314, 0.309468923,
    0.313988713, 0.318546778, 0.323143209, 0.327778098, 0.332451536, 0.337163615, 0.341914425, 0.346704056,
    0.3515326, 0.356400144, 0.36130678, 0.366252596, 0.37123768, 0.376262123, 0.381326011, 0.386429434,
    0.391572478, 0.396755231, 0.40197778, 0.407240212, 0.412542613, 0.417885071, 0.42326767, 0.428690497,
    0.434153636, 0.439657174, 0.445201195, 0.450785783, 0.456411023, 0.462077, 0.467783796, 0.473531496,
    0.479320183, 0.48514994, 0.49102085, 0.496932995, 0.502886458, 0.508881321, 0.514917665, 0.520995573,
    0.527115126, 0.533276404, 0.539479489, 0.545724461, 0.552011402, 0.55834039, 0.564711506, 0.571124829,
    0.57758044, 0.584078418, 0.590618841, 0.597201788, 0.603827339, 0.610495571, 0.617206562, 0.623960392,
    0.630757136, 0.637596874, 0.644479682, 0.651405637, 0.658374817, 0.665387298, 0.672443157, 0.67954247,
    0.686685312, 0.693871761, 0.701101892, 0.70837578, 0.715693501, 0.723055129, 0.73046074, 0.737910409,
    0.74540421, 0.752942217, 0.760524505, 0.768151147, 0.775822218, 0.783537792, 0.79129794, 0.799102738,
    0.806952258, 0.814846572, 0.822785754, 0.830769877, 0.838799012, 0.846873232, 0.854992608, 0.863157213,
    0.871367119, 0.879622397, 0.887923118, 0.896269353, 0.904661174, 0.913098652, 0.921581856, 0.930110858,
    0.938685728, 0.947306537, 0.955973353, 0.964686248, 0.97344529, 0.98225055, 0.991102097, 1.0,
];

/// Fast 2^x approximation using IEEE 754 bit manipulation.
/// From imageflow, originally from fast approximate functions.
#[inline]
fn fastpow2(p: f32) -> f32 {
    let offset: f32 = if p < 0.0 { 1.0 } else { 0.0 };
    let clipp: f32 = if p < -126.0 { -126.0 } else { p };
    let w: i32 = clipp as i32;
    let z: f32 = clipp - w as f32 + offset;
    let bits = ((1_i32 << 23) as f32
        * (clipp + 121.274_055 + 27.728_024 / (4.842_525_5 - z) - 1.490_129_1 * z))
        as u32;
    f32::from_bits(bits)
}

/// Fast log2(x) approximation using IEEE 754 bit manipulation.
#[inline]
fn fastlog2(x: f32) -> f32 {
    let bits = x.to_bits();
    let mx_bits = (bits & 0x007f_ffff) | 0x3f00_0000;
    let mx = f32::from_bits(mx_bits);
    let mut y = bits as f32;
    y *= 1.192_092_9e-7;
    y - 124.225_52 - 1.498_030_3 * mx - 1.725_88 / (0.352_088_72 + mx)
}

/// Fast x^p approximation (~5-10x faster than powf, ~0.1% error).
#[inline]
fn fastpow(x: f32, p: f32) -> f32 {
    fastpow2(p * fastlog2(x))
}

/// Applies sRGB gamma decoding (sRGB to linear RGB).
/// Uses exact formula with powf - use srgb_to_linear_fast for speed.
#[inline]
#[must_use]
pub fn srgb_to_linear(v: f32) -> f32 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// Fast sRGB gamma decoding using fastpow (~5-10x faster, ~0.1% error).
#[inline]
#[must_use]
pub fn srgb_to_linear_fast(v: f32) -> f32 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        fastpow((v + 0.055) / 1.055, 2.4)
    }
}

/// Applies sRGB gamma encoding (linear RGB to sRGB).
/// Uses exact formula with powf - use linear_to_srgb_fast for speed.
#[inline]
#[must_use]
pub fn linear_to_srgb(v: f32) -> f32 {
    if v <= 0.003_130_8 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

/// Fast sRGB gamma encoding using fastpow (~5-10x faster, ~0.1% error).
#[inline]
#[must_use]
pub fn linear_to_srgb_fast(v: f32) -> f32 {
    if v <= 0.003_130_8 {
        v * 12.92
    } else {
        1.055 * fastpow(v, 1.0 / 2.4) - 0.055
    }
}

/// Converts sRGB u8 to linear float (0.0-1.0 range).
/// Uses LUT for exact values and maximum speed.
#[inline]
#[must_use]
pub fn srgb_u8_to_linear(v: u8) -> f32 {
    SRGB_TO_LINEAR_LUT[v as usize]
}

/// Converts sRGB u8 to linear float using exact formula (for verification).
#[inline]
#[must_use]
pub fn srgb_u8_to_linear_exact(v: u8) -> f32 {
    srgb_to_linear(v as f32 / 255.0)
}

/// Converts linear float to sRGB u8.
#[inline]
#[must_use]
pub fn linear_to_srgb_u8(v: f32) -> u8 {
    (linear_to_srgb(v.clamp(0.0, 1.0)) * 255.0).round() as u8
}

/// Fast linear to sRGB u8 using fastpow.
#[inline]
#[must_use]
pub fn linear_to_srgb_u8_fast(v: f32) -> u8 {
    (linear_to_srgb_fast(v.clamp(0.0, 1.0)) * 255.0).round() as u8
}

/// Mixed transfer function used by jpegli (cube root based).
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
/// * `r`, `g`, `b` - Linear RGB values (0.0-1.0 range)
///
/// # Returns
/// (X, Y, B) values in XYB space
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

    // Inverse of opsin absorbance matrix
    // Pre-computed inverse matrix
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

/// Scales XYB values for JPEG encoding (matches C++ ScaleXYBRow).
///
/// This applies the final scaling step needed for jpegli encoding.
/// The scaled values are suitable for DCT and quantization.
#[inline]
#[must_use]
pub fn scale_xyb(x: f32, y: f32, b: f32) -> (f32, f32, f32) {
    // Note: row2 (B) uses row1 (Y) in the calculation
    let scaled_b = (b - y + SCALED_XYB_OFFSET[2]) * SCALED_XYB_SCALE[2];
    let scaled_x = (x + SCALED_XYB_OFFSET[0]) * SCALED_XYB_SCALE[0];
    let scaled_y = (y + SCALED_XYB_OFFSET[1]) * SCALED_XYB_SCALE[1];
    (scaled_x, scaled_y, scaled_b)
}

/// Inverse of scale_xyb for decoding.
#[inline]
#[must_use]
pub fn unscale_xyb(scaled_x: f32, scaled_y: f32, scaled_b: f32) -> (f32, f32, f32) {
    let y = scaled_y / SCALED_XYB_SCALE[1] - SCALED_XYB_OFFSET[1];
    let x = scaled_x / SCALED_XYB_SCALE[0] - SCALED_XYB_OFFSET[0];
    let b = scaled_b / SCALED_XYB_SCALE[2] - SCALED_XYB_OFFSET[2] + y;
    (x, y, b)
}

/// Full sRGB to scaled XYB conversion for jpegli encoding.
///
/// This performs the complete conversion chain:
/// sRGB u8 -> linear RGB -> XYB -> scaled XYB
#[must_use]
pub fn srgb_to_scaled_xyb(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let (x, y, b_xyb) = srgb_to_xyb(r, g, b);
    scale_xyb(x, y, b_xyb)
}

/// Inverse: scaled XYB to sRGB for decoding.
#[must_use]
pub fn scaled_xyb_to_srgb(scaled_x: f32, scaled_y: f32, scaled_b: f32) -> (u8, u8, u8) {
    let (x, y, b) = unscale_xyb(scaled_x, scaled_y, scaled_b);
    xyb_to_srgb(x, y, b)
}

/// Converts an RGB buffer to XYB planes.
pub fn rgb_buffer_to_xyb_planes(
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

/// Converts an RGB buffer to scaled XYB planes for jpegli encoding.
///
/// This is the full conversion chain needed for XYB mode encoding.
pub fn rgb_buffer_to_scaled_xyb_planes(
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
        let (x, y, b) = srgb_to_scaled_xyb(rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]);
        x_plane[i] = x;
        y_plane[i] = y;
        b_plane[i] = b;
    }

    (x_plane, y_plane, b_plane)
}

/// Converts XYB planes to RGB buffer.
pub fn xyb_planes_to_rgb_buffer(
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

            // Allow some rounding error
            assert!(
                (r as i16 - r2 as i16).abs() <= 2,
                "R mismatch for ({},{},{}): {} vs {}",
                r,
                g,
                b,
                r,
                r2
            );
            assert!(
                (g as i16 - g2 as i16).abs() <= 2,
                "G mismatch for ({},{},{}): {} vs {}",
                r,
                g,
                b,
                g,
                g2
            );
            assert!(
                (b as i16 - b2 as i16).abs() <= 2,
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
        // Gray values should have X near 0
        for gray in [0u8, 64, 128, 192, 255] {
            let (x, _y, _b) = srgb_to_xyb(gray, gray, gray);
            assert!(x.abs() < 0.01, "X should be ~0 for gray, got {}", x);
        }
    }

    #[test]
    fn test_scale_unscale_roundtrip() {
        let test_values = [
            (0.0f32, 0.0f32, 0.0f32),
            (0.1, 0.5, 0.3),
            (-0.1, 0.8, 0.6),
            (0.05, 0.3, 0.4),
        ];

        for (x, y, b) in test_values {
            let (sx, sy, sb) = scale_xyb(x, y, b);
            let (x2, y2, b2) = unscale_xyb(sx, sy, sb);

            assert!((x - x2).abs() < 1e-5, "X mismatch: {} vs {}", x, x2);
            assert!((y - y2).abs() < 1e-5, "Y mismatch: {} vs {}", y, y2);
            assert!((b - b2).abs() < 1e-5, "B mismatch: {} vs {}", b, b2);
        }
    }

    #[test]
    fn test_srgb_scaled_xyb_roundtrip() {
        let test_colors = [
            (0u8, 0u8, 0u8),
            (255u8, 255u8, 255u8),
            (255u8, 0u8, 0u8),
            (128u8, 128u8, 128u8),
        ];

        for (r, g, b) in test_colors {
            let (sx, sy, sb) = srgb_to_scaled_xyb(r, g, b);
            let (r2, g2, b2) = scaled_xyb_to_srgb(sx, sy, sb);

            assert!(
                (r as i16 - r2 as i16).abs() <= 2,
                "R mismatch for ({},{},{}): {} vs {}",
                r,
                g,
                b,
                r,
                r2
            );
            assert!(
                (g as i16 - g2 as i16).abs() <= 2,
                "G mismatch for ({},{},{}): {} vs {}",
                r,
                g,
                b,
                g,
                g2
            );
            assert!(
                (b as i16 - b2 as i16).abs() <= 2,
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
    fn test_linear_rgb_xyb_direct() {
        // Test linear RGB to XYB conversion directly
        let (x, y, b) = linear_rgb_to_xyb(0.5, 0.5, 0.5);
        // Gray should have X near 0
        assert!(x.abs() < 0.01, "X should be ~0 for gray, got {}", x);

        // Y should be positive for mid-gray
        assert!(y > 0.0, "Y should be positive, got {}", y);

        // Roundtrip
        let (r, g, b_out) = xyb_to_linear_rgb(x, y, b);
        assert!((r - 0.5).abs() < 0.01);
        assert!((g - 0.5).abs() < 0.01);
        assert!((b_out - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_srgb_linear_edge_cases() {
        // Test sRGB to linear at boundaries
        assert_eq!(srgb_to_linear(0.0), 0.0);
        assert!((srgb_to_linear(1.0) - 1.0).abs() < 1e-6);

        // Test near the 0.04045 threshold
        let below = srgb_to_linear(0.04);
        let above = srgb_to_linear(0.05);
        assert!(below < above);

        // Test linear to sRGB at boundaries
        assert_eq!(linear_to_srgb(0.0), 0.0);
        assert!((linear_to_srgb(1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_rgb_buffer_to_xyb_planes() {
        // 2x2 image with different colors
        let rgb = vec![
            255, 0, 0, // Red
            0, 255, 0, // Green
            0, 0, 255, // Blue
            128, 128, 128, // Gray
        ];

        let (x_plane, y_plane, b_plane) = rgb_buffer_to_xyb_planes(&rgb, 2, 2);

        assert_eq!(x_plane.len(), 4);
        assert_eq!(y_plane.len(), 4);
        assert_eq!(b_plane.len(), 4);

        // Gray should have X near 0
        assert!(x_plane[3].abs() < 0.01);
    }

    #[test]
    fn test_rgb_buffer_to_scaled_xyb_planes() {
        let rgb = vec![128, 128, 128, 255, 255, 255]; // 2 pixels

        let (x_plane, y_plane, b_plane) = rgb_buffer_to_scaled_xyb_planes(&rgb, 2, 1);

        assert_eq!(x_plane.len(), 2);
        assert_eq!(y_plane.len(), 2);
        assert_eq!(b_plane.len(), 2);

        // Values should be in reasonable ranges for JPEG
        for v in &x_plane {
            assert!(v.is_finite());
        }
        for v in &y_plane {
            assert!(v.is_finite());
        }
        for v in &b_plane {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_xyb_planes_to_rgb_buffer() {
        // Create XYB planes for a gray image
        let x_plane = vec![0.0f32; 4];
        let y_plane = vec![0.5f32; 4];
        let b_plane = vec![0.5f32; 4];

        let rgb = xyb_planes_to_rgb_buffer(&x_plane, &y_plane, &b_plane, 2, 2);

        assert_eq!(rgb.len(), 12); // 4 pixels * 3 channels
    }

    #[test]
    fn test_mixed_cbrt_cube() {
        // Test that mixed_cbrt and mixed_cube are inverses
        let test_values = [-1.0f32, -0.5, 0.0, 0.5, 1.0, 2.0, -2.0];

        for v in test_values {
            let cbrt = mixed_cbrt(v);
            let back = mixed_cube(cbrt);
            assert!(
                (v - back).abs() < 1e-6,
                "Roundtrip failed for {}: got {}",
                v,
                back
            );
        }
    }

    #[test]
    fn test_xyb_extreme_colors() {
        // Test with extreme colors to ensure no overflow/underflow
        let extreme_colors = [
            (0u8, 0u8, 0u8),       // Black
            (255u8, 255u8, 255u8), // White
            (255u8, 0u8, 0u8),     // Pure red
            (0u8, 255u8, 0u8),     // Pure green
            (0u8, 0u8, 255u8),     // Pure blue
            (255u8, 255u8, 0u8),   // Yellow
            (255u8, 0u8, 255u8),   // Magenta
            (0u8, 255u8, 255u8),   // Cyan
        ];

        for (r, g, b) in extreme_colors {
            let (x, y, b_xyb) = srgb_to_xyb(r, g, b);
            assert!(x.is_finite(), "X not finite for ({},{},{})", r, g, b);
            assert!(y.is_finite(), "Y not finite for ({},{},{})", r, g, b);
            assert!(b_xyb.is_finite(), "B not finite for ({},{},{})", r, g, b);

            let (sx, sy, sb) = scale_xyb(x, y, b_xyb);
            assert!(
                sx.is_finite(),
                "Scaled X not finite for ({},{},{})",
                r,
                g,
                b
            );
            assert!(
                sy.is_finite(),
                "Scaled Y not finite for ({},{},{})",
                r,
                g,
                b
            );
            assert!(
                sb.is_finite(),
                "Scaled B not finite for ({},{},{})",
                r,
                g,
                b
            );
        }
    }

    #[test]
    fn test_lut_vs_exact() {
        // Verify LUT values match exact computation within acceptable tolerance
        let mut max_error: f32 = 0.0;
        let mut worst_index = 0;
        for i in 0..=255u8 {
            let lut_val = srgb_u8_to_linear(i);
            let exact_val = srgb_u8_to_linear_exact(i);
            let error = (lut_val - exact_val).abs();
            if error > max_error {
                max_error = error;
                worst_index = i;
            }
        }
        // Allow up to 0.5% error for the polynomial approximation in const fn
        assert!(
            max_error < 0.005,
            "LUT error too large: {} at index {} (lut={}, exact={})",
            max_error,
            worst_index,
            srgb_u8_to_linear(worst_index),
            srgb_u8_to_linear_exact(worst_index)
        );
        println!("LUT max error: {:.6} at index {}", max_error, worst_index);
    }

    #[test]
    fn test_fastpow_accuracy() {
        // Test fastpow accuracy for the specific exponents we use
        let test_values = [0.01, 0.1, 0.2, 0.5, 0.8, 0.9, 0.99, 1.0];

        for &v in &test_values {
            // Test sRGB decode (exponent 2.4)
            let exact = ((v + 0.055) / 1.055_f32).powf(2.4);
            let fast = srgb_to_linear_fast(v);
            let error = (fast - srgb_to_linear(v)).abs();
            assert!(
                error < 0.002,
                "srgb_to_linear_fast error too large for {}: {} (exact={}, fast={})",
                v,
                error,
                srgb_to_linear(v),
                fast
            );
        }

        for &v in &test_values {
            // Test sRGB encode (exponent 1/2.4)
            let exact = linear_to_srgb(v);
            let fast = linear_to_srgb_fast(v);
            let error = (fast - exact).abs();
            assert!(
                error < 0.002,
                "linear_to_srgb_fast error too large for {}: {} (exact={}, fast={})",
                v,
                error,
                exact,
                fast
            );
        }
    }

    #[test]
    fn test_fast_roundtrip() {
        // Verify fast functions roundtrip correctly (±1 for u8)
        for v in 0..=255u8 {
            let linear = srgb_u8_to_linear(v); // LUT-based, exact
            let back = linear_to_srgb_u8_fast(linear);
            assert!(
                (v as i16 - back as i16).abs() <= 1,
                "Fast roundtrip failed for {}: got {}",
                v,
                back
            );
        }
    }

    #[test]
    fn test_fast_vs_exact_u8_output() {
        // For u8 output, fast should match exact within ±1
        for v in 0..=255u8 {
            let linear = srgb_u8_to_linear_exact(v);
            let exact_back = linear_to_srgb_u8(linear);
            let fast_back = linear_to_srgb_u8_fast(linear);
            assert!(
                (exact_back as i16 - fast_back as i16).abs() <= 1,
                "Fast vs exact mismatch for input {}: exact={}, fast={}",
                v,
                exact_back,
                fast_back
            );
        }
    }
}
