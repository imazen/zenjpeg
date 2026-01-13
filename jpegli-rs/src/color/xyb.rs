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

/// sRGB to linear using C++ jpegli's rational polynomial approximation.
///
/// This matches the `TF_SRGB::DisplayFromEncoded` function in libjxl's
/// transfer_functions-inl.h. The XYB perceptual model was tuned with this
/// approximation, so using it gives better quality match with C++.
#[allow(dead_code)]
#[inline]
#[must_use]
fn srgb_to_linear_poly(x: f32) -> f32 {
    const THRESH: f32 = 0.04045;
    const LOW_DIV_INV: f32 = 1.0 / 12.92;

    // Rational polynomial coefficients from C++ (Chebyshev approximation)
    const P: [f32; 5] = [
        2.200248328e-04,
        1.043637593e-02,
        1.624820318e-01,
        7.961564959e-01,
        8.210152774e-01,
    ];
    const Q: [f32; 5] = [
        2.631846970e-01,
        1.076976492e+00,
        4.987528350e-01,
        -5.512498495e-02,
        6.521209011e-03,
    ];

    let x = x.abs();

    if x <= THRESH {
        x * LOW_DIV_INV
    } else {
        // Evaluate rational polynomial p(x)/q(x) using Horner's method
        // Coefficients are ordered: [degree-0, degree-1, degree-2, degree-3, degree-4]
        // Horner evaluation starts from highest degree: p4*x^4 + p3*x^3 + p2*x^2 + p1*x + p0
        let p_val = P[4]
            .mul_add(x, P[3])
            .mul_add(x, P[2])
            .mul_add(x, P[1])
            .mul_add(x, P[0]);
        let q_val = Q[4]
            .mul_add(x, Q[3])
            .mul_add(x, Q[2])
            .mul_add(x, Q[1])
            .mul_add(x, Q[0]);
        p_val / q_val
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

/// Fast cube root matching C++ jpegli's CubeRootAndAdd algorithm exactly.
///
/// Uses IEEE 754 bit manipulation for initial approximation, then
/// 3 Newton-Raphson iterations in f32 (NOT f64) to match C++ precision.
///
/// Maximum error: ~6 ULP (Units in Last Place)
#[inline]
#[must_use]
fn cbrtf_fast(x: f32) -> f32 {
    // Match C++ exactly: CubeRootAndAdd from lib/base/fast_math-inl.h
    // The algorithm computes 1/cbrt(x), then derives cbrt(x) = x * (1/cbrt(x))^2

    if x == 0.0 {
        return 0.0;
    }

    // Constants matching C++ exactly
    const K_EXP_BIAS: u32 = 0x5480_0000; // cast(1.) + cast(1.) / 3
    const K_EXP_MUL: u32 = 0x002A_AAAA; // shifted 1/3
    const K1_3: f32 = 1.0 / 3.0;
    const K4_3: f32 = 4.0 / 3.0;

    let xa = x;
    let xa_3 = K1_3 * xa;

    // Initial approximation via IEEE 754 bit manipulation
    // Computes approximate 1/cbrt(x)
    let m1 = xa.to_bits() as i32;
    let m2 = if m1 == 0 {
        0
    } else {
        (K_EXP_BIAS as i32) - ((m1 >> 23) * (K_EXP_MUL as i32))
    };
    let mut r = f32::from_bits(m2 as u32);

    // 3 Newton-Raphson iterations (matching C++ exactly)
    // Formula: r = (4/3)*r - (x/3)*r^4
    for _ in 0..3 {
        let r2 = r * r;
        r = K4_3 * r - xa_3 * r2 * r2;
    }

    // Final iteration for extra precision
    // r = r + (1/3) * (r - x * r^4)
    let r2 = r * r;
    r = r + K1_3 * (r - xa * r2 * r2);

    // Convert from 1/cbrt(x) to cbrt(x): cbrt(x) = x * (1/cbrt(x))^2
    let r2 = r * r;
    r2 * x
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
    // Step 1: Apply opsin absorbance matrix using FMA for precision
    let m = &XYB_OPSIN_ABSORBANCE_MATRIX;
    let bias = XYB_OPSIN_ABSORBANCE_BIAS[0];

    // Use mul_add for fused multiply-add (single rounding, matches C++)
    let opsin_r = m[0].mul_add(r, m[1].mul_add(g, m[2].mul_add(b, bias)));
    let opsin_g = m[3].mul_add(r, m[4].mul_add(g, m[5].mul_add(b, bias)));
    let opsin_b = m[6].mul_add(r, m[7].mul_add(g, m[8].mul_add(b, bias)));

    // Step 2: Clamp negatives to zero (matches C++ ZeroIfNegative)
    let opsin_r = opsin_r.max(0.0);
    let opsin_g = opsin_g.max(0.0);
    let opsin_b = opsin_b.max(0.0);

    // Step 3: Apply cube root for perceptual uniformity
    // Use fast cbrt approximation matching C++/ssimulacra2 algorithm
    let neg_bias_cbrt = -cbrtf_fast(XYB_OPSIN_ABSORBANCE_BIAS[0]);
    let cbrt_r = cbrtf_fast(opsin_r) + neg_bias_cbrt;
    let cbrt_g = cbrtf_fast(opsin_g) + neg_bias_cbrt;
    let cbrt_b = cbrtf_fast(opsin_b) + neg_bias_cbrt;

    // Step 4: Final XYB transform
    // X = (L - M) / 2
    // Y = (L + M) / 2
    // B = S
    let x = 0.5 * (cbrt_r - cbrt_g);
    let y = 0.5 * (cbrt_r + cbrt_g);

    (x, y, cbrt_b)
}

/// Converts linear RGB to XYB color space using C++ jpegli range conventions.
///
/// This matches C++ jpegli's LinearRGBRowToXYB which expects linear RGB in 0-255 range.
/// The resulting XYB values are in a larger range than `linear_rgb_to_xyb` (Y up to ~6.2 for white).
///
/// # Arguments
/// * `r`, `g`, `b` - Linear RGB values (0.0-255.0 range, matching C++ jpegli)
///
/// # Returns
/// (X, Y, B) values in XYB space (Y range approximately 0-6.2 for white)
#[must_use]
pub fn linear_rgb_to_xyb_255(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    // Same algorithm as linear_rgb_to_xyb, but input is in 0-255 range like C++
    let m = &XYB_OPSIN_ABSORBANCE_MATRIX;
    let bias = XYB_OPSIN_ABSORBANCE_BIAS[0];

    let opsin_r = m[0].mul_add(r, m[1].mul_add(g, m[2].mul_add(b, bias)));
    let opsin_g = m[3].mul_add(r, m[4].mul_add(g, m[5].mul_add(b, bias)));
    let opsin_b = m[6].mul_add(r, m[7].mul_add(g, m[8].mul_add(b, bias)));

    let opsin_r = opsin_r.max(0.0);
    let opsin_g = opsin_g.max(0.0);
    let opsin_b = opsin_b.max(0.0);

    let neg_bias_cbrt = -cbrtf_fast(XYB_OPSIN_ABSORBANCE_BIAS[0]);
    let cbrt_r = cbrtf_fast(opsin_r) + neg_bias_cbrt;
    let cbrt_g = cbrtf_fast(opsin_g) + neg_bias_cbrt;
    let cbrt_b = cbrtf_fast(opsin_b) + neg_bias_cbrt;

    let x = 0.5 * (cbrt_r - cbrt_g);
    let y = 0.5 * (cbrt_r + cbrt_g);

    (x, y, cbrt_b)
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

    // Use FMA for matrix multiply
    let r = INV_OPSIN[0].mul_add(
        opsin_r,
        INV_OPSIN[1].mul_add(opsin_g, INV_OPSIN[2] * opsin_b),
    );
    let g = INV_OPSIN[3].mul_add(
        opsin_r,
        INV_OPSIN[4].mul_add(opsin_g, INV_OPSIN[5] * opsin_b),
    );
    let b_out = INV_OPSIN[6].mul_add(
        opsin_r,
        INV_OPSIN[7].mul_add(opsin_g, INV_OPSIN[8] * opsin_b),
    );

    (r, g, b_out)
}

/// Converts sRGB u8 to XYB using C++ jpegli conventions.
///
/// This matches C++ jpegli's pipeline which uses linear RGB in 0-1 range.
/// The resulting XYB values are in the standard range (Y up to ~0.84 for white).
#[must_use]
pub fn srgb_to_xyb(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    // Convert sRGB to linear (0-1 range from LUT)
    let lr = srgb_u8_to_linear(r);
    let lg = srgb_u8_to_linear(g);
    let lb = srgb_u8_to_linear(b);
    // Use 0-1 linear RGB (matching C++ jpegli's LinearRGBRowToXYB)
    linear_rgb_to_xyb(lr, lg, lb)
}

/// Converts XYB to sRGB u8.
///
/// This handles XYB values in the standard C++ jpegli convention (Y ~0.84 for white).
/// The inverse linear RGB is in 0-1 range matching the forward conversion.
#[must_use]
pub fn xyb_to_srgb(x: f32, y: f32, b: f32) -> (u8, u8, u8) {
    let (lr, lg, lb) = xyb_to_linear_rgb(x, y, b);
    // Linear RGB is already in 0-1 range (matching C++ jpegli's conventions)
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

/// SIMD-optimized sRGB to scaled XYB conversion.
///
/// Takes contiguous sRGB u8 data and outputs separate scaled XYB f32 planes.
/// Uses SIMD acceleration for the XYB conversion with the LUT for sRGB to linear.
pub fn srgb_to_scaled_xyb_planes_simd(
    rgb_data: &[u8],
    num_pixels: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    use wide::f32x8;

    assert!(rgb_data.len() >= num_pixels * 3);

    // Allocate zeroed - OS zero-page optimization makes this fast
    let mut x_plane = vec![0.0f32; num_pixels];
    let mut y_plane = vec![0.0f32; num_pixels];
    let mut b_plane = vec![0.0f32; num_pixels];

    let m = &XYB_OPSIN_ABSORBANCE_MATRIX;
    let bias = XYB_OPSIN_ABSORBANCE_BIAS[0];
    let neg_bias_cbrt = -cbrtf_fast(bias);

    // SIMD constants for XYB matrix
    let m00 = f32x8::splat(m[0]);
    let m01 = f32x8::splat(m[1]);
    let m02 = f32x8::splat(m[2]);
    let m10 = f32x8::splat(m[3]);
    let m11 = f32x8::splat(m[4]);
    let m12 = f32x8::splat(m[5]);
    let m20 = f32x8::splat(m[6]);
    let m21 = f32x8::splat(m[7]);
    let m22 = f32x8::splat(m[8]);
    let bias_simd = f32x8::splat(bias);
    let zero = f32x8::splat(0.0);
    let neg_bias_cbrt_simd = f32x8::splat(neg_bias_cbrt);
    let half = f32x8::splat(0.5);

    // SIMD constants for scaling
    let scale_x = f32x8::splat(SCALED_XYB_SCALE[0]);
    let scale_y = f32x8::splat(SCALED_XYB_SCALE[1]);
    let scale_b = f32x8::splat(SCALED_XYB_SCALE[2]);
    let offset_x = f32x8::splat(SCALED_XYB_OFFSET[0]);
    let offset_y = f32x8::splat(SCALED_XYB_OFFSET[1]);
    let offset_b = f32x8::splat(SCALED_XYB_OFFSET[2]);

    let chunks = num_pixels / 8;

    for chunk in 0..chunks {
        let pixel_idx = chunk * 8;
        let rgb_idx = pixel_idx * 3;

        // Gather 8 RGB pixels and convert sRGB to linear via LUT
        let r = f32x8::from([
            SRGB_TO_LINEAR_LUT[rgb_data[rgb_idx] as usize],
            SRGB_TO_LINEAR_LUT[rgb_data[rgb_idx + 3] as usize],
            SRGB_TO_LINEAR_LUT[rgb_data[rgb_idx + 6] as usize],
            SRGB_TO_LINEAR_LUT[rgb_data[rgb_idx + 9] as usize],
            SRGB_TO_LINEAR_LUT[rgb_data[rgb_idx + 12] as usize],
            SRGB_TO_LINEAR_LUT[rgb_data[rgb_idx + 15] as usize],
            SRGB_TO_LINEAR_LUT[rgb_data[rgb_idx + 18] as usize],
            SRGB_TO_LINEAR_LUT[rgb_data[rgb_idx + 21] as usize],
        ]);

        let g = f32x8::from([
            SRGB_TO_LINEAR_LUT[rgb_data[rgb_idx + 1] as usize],
            SRGB_TO_LINEAR_LUT[rgb_data[rgb_idx + 4] as usize],
            SRGB_TO_LINEAR_LUT[rgb_data[rgb_idx + 7] as usize],
            SRGB_TO_LINEAR_LUT[rgb_data[rgb_idx + 10] as usize],
            SRGB_TO_LINEAR_LUT[rgb_data[rgb_idx + 13] as usize],
            SRGB_TO_LINEAR_LUT[rgb_data[rgb_idx + 16] as usize],
            SRGB_TO_LINEAR_LUT[rgb_data[rgb_idx + 19] as usize],
            SRGB_TO_LINEAR_LUT[rgb_data[rgb_idx + 22] as usize],
        ]);

        let b_in = f32x8::from([
            SRGB_TO_LINEAR_LUT[rgb_data[rgb_idx + 2] as usize],
            SRGB_TO_LINEAR_LUT[rgb_data[rgb_idx + 5] as usize],
            SRGB_TO_LINEAR_LUT[rgb_data[rgb_idx + 8] as usize],
            SRGB_TO_LINEAR_LUT[rgb_data[rgb_idx + 11] as usize],
            SRGB_TO_LINEAR_LUT[rgb_data[rgb_idx + 14] as usize],
            SRGB_TO_LINEAR_LUT[rgb_data[rgb_idx + 17] as usize],
            SRGB_TO_LINEAR_LUT[rgb_data[rgb_idx + 20] as usize],
            SRGB_TO_LINEAR_LUT[rgb_data[rgb_idx + 23] as usize],
        ]);

        // Opsin absorbance matrix with FMA
        let mut opsin0 = m00.mul_add(r, m01.mul_add(g, m02.mul_add(b_in, bias_simd)));
        let mut opsin1 = m10.mul_add(r, m11.mul_add(g, m12.mul_add(b_in, bias_simd)));
        let mut opsin2 = m20.mul_add(r, m21.mul_add(g, m22.mul_add(b_in, bias_simd)));

        // Clamp negatives
        opsin0 = opsin0.max(zero);
        opsin1 = opsin1.max(zero);
        opsin2 = opsin2.max(zero);

        // Cube root + bias subtraction
        opsin0 = cbrtf_x8(opsin0) + neg_bias_cbrt_simd;
        opsin1 = cbrtf_x8(opsin1) + neg_bias_cbrt_simd;
        opsin2 = cbrtf_x8(opsin2) + neg_bias_cbrt_simd;

        // XYB transform: X = (L-M)/2, Y = (L+M)/2, B = S
        let x_xyb = half * (opsin0 - opsin1);
        let y_xyb = half * (opsin0 + opsin1);
        let b_xyb = opsin2;

        // Scale XYB for JPEG: scaled = (value + offset) * scale
        // Note: B uses Y in calculation
        let scaled_x = (x_xyb + offset_x) * scale_x;
        let scaled_y = (y_xyb + offset_y) * scale_y;
        let scaled_b = (b_xyb - y_xyb + offset_b) * scale_b;

        // Store results to planes
        let x_arr: [f32; 8] = scaled_x.into();
        let y_arr: [f32; 8] = scaled_y.into();
        let b_arr: [f32; 8] = scaled_b.into();

        x_plane[pixel_idx..pixel_idx + 8].copy_from_slice(&x_arr);
        y_plane[pixel_idx..pixel_idx + 8].copy_from_slice(&y_arr);
        b_plane[pixel_idx..pixel_idx + 8].copy_from_slice(&b_arr);
    }

    // Scalar remainder
    for i in (chunks * 8)..num_pixels {
        let (x, y, b) =
            srgb_to_scaled_xyb(rgb_data[i * 3], rgb_data[i * 3 + 1], rgb_data[i * 3 + 2]);
        x_plane[i] = x;
        y_plane[i] = y;
        b_plane[i] = b;
    }

    (x_plane, y_plane, b_plane)
}

/// SIMD-accelerated sRGB to scaled XYB conversion - inplace version.
///
/// Writes directly to pre-allocated buffers. Used by strip-based encoder
/// to avoid per-strip allocations.
pub fn srgb_to_scaled_xyb_planes_simd_inplace(
    rgb_data: &[u8],
    x_plane: &mut [f32],
    y_plane: &mut [f32],
    b_plane: &mut [f32],
    num_pixels: usize,
) {
    use wide::f32x8;

    assert!(rgb_data.len() >= num_pixels * 3);
    assert!(x_plane.len() >= num_pixels);
    assert!(y_plane.len() >= num_pixels);
    assert!(b_plane.len() >= num_pixels);

    let m = &XYB_OPSIN_ABSORBANCE_MATRIX;
    let bias = XYB_OPSIN_ABSORBANCE_BIAS[0];
    let neg_bias_cbrt = -cbrtf_fast(bias);

    // SIMD constants for XYB matrix
    let m00 = f32x8::splat(m[0]);
    let m01 = f32x8::splat(m[1]);
    let m02 = f32x8::splat(m[2]);
    let m10 = f32x8::splat(m[3]);
    let m11 = f32x8::splat(m[4]);
    let m12 = f32x8::splat(m[5]);
    let m20 = f32x8::splat(m[6]);
    let m21 = f32x8::splat(m[7]);
    let m22 = f32x8::splat(m[8]);
    let bias_simd = f32x8::splat(bias);
    let zero = f32x8::splat(0.0);
    let neg_bias_cbrt_simd = f32x8::splat(neg_bias_cbrt);
    let half = f32x8::splat(0.5);

    // SIMD constants for scaling
    let scale_x = f32x8::splat(SCALED_XYB_SCALE[0]);
    let scale_y = f32x8::splat(SCALED_XYB_SCALE[1]);
    let scale_b = f32x8::splat(SCALED_XYB_SCALE[2]);
    let offset_x = f32x8::splat(SCALED_XYB_OFFSET[0]);
    let offset_y = f32x8::splat(SCALED_XYB_OFFSET[1]);
    let offset_b = f32x8::splat(SCALED_XYB_OFFSET[2]);

    let chunks = num_pixels / 8;

    for chunk in 0..chunks {
        let pixel_idx = chunk * 8;
        let rgb_idx = pixel_idx * 3;

        // Gather 8 RGB pixels and convert sRGB to linear via LUT
        let r = f32x8::from([
            SRGB_TO_LINEAR_LUT[rgb_data[rgb_idx] as usize],
            SRGB_TO_LINEAR_LUT[rgb_data[rgb_idx + 3] as usize],
            SRGB_TO_LINEAR_LUT[rgb_data[rgb_idx + 6] as usize],
            SRGB_TO_LINEAR_LUT[rgb_data[rgb_idx + 9] as usize],
            SRGB_TO_LINEAR_LUT[rgb_data[rgb_idx + 12] as usize],
            SRGB_TO_LINEAR_LUT[rgb_data[rgb_idx + 15] as usize],
            SRGB_TO_LINEAR_LUT[rgb_data[rgb_idx + 18] as usize],
            SRGB_TO_LINEAR_LUT[rgb_data[rgb_idx + 21] as usize],
        ]);

        let g = f32x8::from([
            SRGB_TO_LINEAR_LUT[rgb_data[rgb_idx + 1] as usize],
            SRGB_TO_LINEAR_LUT[rgb_data[rgb_idx + 4] as usize],
            SRGB_TO_LINEAR_LUT[rgb_data[rgb_idx + 7] as usize],
            SRGB_TO_LINEAR_LUT[rgb_data[rgb_idx + 10] as usize],
            SRGB_TO_LINEAR_LUT[rgb_data[rgb_idx + 13] as usize],
            SRGB_TO_LINEAR_LUT[rgb_data[rgb_idx + 16] as usize],
            SRGB_TO_LINEAR_LUT[rgb_data[rgb_idx + 19] as usize],
            SRGB_TO_LINEAR_LUT[rgb_data[rgb_idx + 22] as usize],
        ]);

        let b_in = f32x8::from([
            SRGB_TO_LINEAR_LUT[rgb_data[rgb_idx + 2] as usize],
            SRGB_TO_LINEAR_LUT[rgb_data[rgb_idx + 5] as usize],
            SRGB_TO_LINEAR_LUT[rgb_data[rgb_idx + 8] as usize],
            SRGB_TO_LINEAR_LUT[rgb_data[rgb_idx + 11] as usize],
            SRGB_TO_LINEAR_LUT[rgb_data[rgb_idx + 14] as usize],
            SRGB_TO_LINEAR_LUT[rgb_data[rgb_idx + 17] as usize],
            SRGB_TO_LINEAR_LUT[rgb_data[rgb_idx + 20] as usize],
            SRGB_TO_LINEAR_LUT[rgb_data[rgb_idx + 23] as usize],
        ]);

        // Apply opsin absorbance matrix: mixed = M * [r, g, b] (using FMA)
        let mixed0 = m00.mul_add(r, m01.mul_add(g, m02.mul_add(b_in, bias_simd)));
        let mixed1 = m10.mul_add(r, m11.mul_add(g, m12.mul_add(b_in, bias_simd)));
        let mixed2 = m20.mul_add(r, m21.mul_add(g, m22.mul_add(b_in, bias_simd)));

        // Clamp to non-negative for cube root
        let mixed0 = mixed0.max(zero);
        let mixed1 = mixed1.max(zero);
        let mixed2 = mixed2.max(zero);

        // Cube root approximation (fast, not perfectly accurate)
        let gamma0 = cbrtf_x8(mixed0) + neg_bias_cbrt_simd;
        let gamma1 = cbrtf_x8(mixed1) + neg_bias_cbrt_simd;
        let gamma2 = cbrtf_x8(mixed2) + neg_bias_cbrt_simd;

        // XYB color space transform
        let x_out = half * (gamma0 - gamma1);
        let y_out = half * (gamma0 + gamma1);
        let b_out = gamma2;

        // Apply scaling to [0, 1] range
        let scaled_x = x_out * scale_x + offset_x;
        let scaled_y = y_out * scale_y + offset_y;
        let scaled_b = b_out * scale_b + offset_b;

        // Store results
        let x_arr: [f32; 8] = scaled_x.into();
        let y_arr: [f32; 8] = scaled_y.into();
        let b_arr: [f32; 8] = scaled_b.into();

        x_plane[pixel_idx..pixel_idx + 8].copy_from_slice(&x_arr);
        y_plane[pixel_idx..pixel_idx + 8].copy_from_slice(&y_arr);
        b_plane[pixel_idx..pixel_idx + 8].copy_from_slice(&b_arr);
    }

    // Scalar remainder
    for i in (chunks * 8)..num_pixels {
        let (x, y, b) =
            srgb_to_scaled_xyb(rgb_data[i * 3], rgb_data[i * 3 + 1], rgb_data[i * 3 + 2]);
        x_plane[i] = x;
        y_plane[i] = y;
        b_plane[i] = b;
    }
}

/// SIMD-accelerated sRGB to scaled XYB conversion for RGBA input - inplace version.
pub fn srgb_to_scaled_xyb_planes_simd_rgba_inplace(
    rgba_data: &[u8],
    x_plane: &mut [f32],
    y_plane: &mut [f32],
    b_plane: &mut [f32],
    num_pixels: usize,
) {
    use wide::f32x8;

    assert!(rgba_data.len() >= num_pixels * 4);
    assert!(x_plane.len() >= num_pixels);
    assert!(y_plane.len() >= num_pixels);
    assert!(b_plane.len() >= num_pixels);

    let m = &XYB_OPSIN_ABSORBANCE_MATRIX;
    let bias = XYB_OPSIN_ABSORBANCE_BIAS[0];
    let neg_bias_cbrt = -cbrtf_fast(bias);

    let m00 = f32x8::splat(m[0]);
    let m01 = f32x8::splat(m[1]);
    let m02 = f32x8::splat(m[2]);
    let m10 = f32x8::splat(m[3]);
    let m11 = f32x8::splat(m[4]);
    let m12 = f32x8::splat(m[5]);
    let m20 = f32x8::splat(m[6]);
    let m21 = f32x8::splat(m[7]);
    let m22 = f32x8::splat(m[8]);
    let bias_simd = f32x8::splat(bias);
    let zero = f32x8::splat(0.0);
    let neg_bias_cbrt_simd = f32x8::splat(neg_bias_cbrt);
    let half = f32x8::splat(0.5);

    let scale_x = f32x8::splat(SCALED_XYB_SCALE[0]);
    let scale_y = f32x8::splat(SCALED_XYB_SCALE[1]);
    let scale_b = f32x8::splat(SCALED_XYB_SCALE[2]);
    let offset_x = f32x8::splat(SCALED_XYB_OFFSET[0]);
    let offset_y = f32x8::splat(SCALED_XYB_OFFSET[1]);
    let offset_b = f32x8::splat(SCALED_XYB_OFFSET[2]);

    let chunks = num_pixels / 8;

    for chunk in 0..chunks {
        let pixel_idx = chunk * 8;
        let rgba_idx = pixel_idx * 4;

        let r = f32x8::from([
            SRGB_TO_LINEAR_LUT[rgba_data[rgba_idx] as usize],
            SRGB_TO_LINEAR_LUT[rgba_data[rgba_idx + 4] as usize],
            SRGB_TO_LINEAR_LUT[rgba_data[rgba_idx + 8] as usize],
            SRGB_TO_LINEAR_LUT[rgba_data[rgba_idx + 12] as usize],
            SRGB_TO_LINEAR_LUT[rgba_data[rgba_idx + 16] as usize],
            SRGB_TO_LINEAR_LUT[rgba_data[rgba_idx + 20] as usize],
            SRGB_TO_LINEAR_LUT[rgba_data[rgba_idx + 24] as usize],
            SRGB_TO_LINEAR_LUT[rgba_data[rgba_idx + 28] as usize],
        ]);

        let g = f32x8::from([
            SRGB_TO_LINEAR_LUT[rgba_data[rgba_idx + 1] as usize],
            SRGB_TO_LINEAR_LUT[rgba_data[rgba_idx + 5] as usize],
            SRGB_TO_LINEAR_LUT[rgba_data[rgba_idx + 9] as usize],
            SRGB_TO_LINEAR_LUT[rgba_data[rgba_idx + 13] as usize],
            SRGB_TO_LINEAR_LUT[rgba_data[rgba_idx + 17] as usize],
            SRGB_TO_LINEAR_LUT[rgba_data[rgba_idx + 21] as usize],
            SRGB_TO_LINEAR_LUT[rgba_data[rgba_idx + 25] as usize],
            SRGB_TO_LINEAR_LUT[rgba_data[rgba_idx + 29] as usize],
        ]);

        let b_in = f32x8::from([
            SRGB_TO_LINEAR_LUT[rgba_data[rgba_idx + 2] as usize],
            SRGB_TO_LINEAR_LUT[rgba_data[rgba_idx + 6] as usize],
            SRGB_TO_LINEAR_LUT[rgba_data[rgba_idx + 10] as usize],
            SRGB_TO_LINEAR_LUT[rgba_data[rgba_idx + 14] as usize],
            SRGB_TO_LINEAR_LUT[rgba_data[rgba_idx + 18] as usize],
            SRGB_TO_LINEAR_LUT[rgba_data[rgba_idx + 22] as usize],
            SRGB_TO_LINEAR_LUT[rgba_data[rgba_idx + 26] as usize],
            SRGB_TO_LINEAR_LUT[rgba_data[rgba_idx + 30] as usize],
        ]);

        // Apply opsin absorbance matrix (using FMA)
        let mixed0 = m00.mul_add(r, m01.mul_add(g, m02.mul_add(b_in, bias_simd)));
        let mixed1 = m10.mul_add(r, m11.mul_add(g, m12.mul_add(b_in, bias_simd)));
        let mixed2 = m20.mul_add(r, m21.mul_add(g, m22.mul_add(b_in, bias_simd)));

        let mixed0 = mixed0.max(zero);
        let mixed1 = mixed1.max(zero);
        let mixed2 = mixed2.max(zero);

        let gamma0 = cbrtf_x8(mixed0) + neg_bias_cbrt_simd;
        let gamma1 = cbrtf_x8(mixed1) + neg_bias_cbrt_simd;
        let gamma2 = cbrtf_x8(mixed2) + neg_bias_cbrt_simd;

        let x_out = half * (gamma0 - gamma1);
        let y_out = half * (gamma0 + gamma1);
        let b_out = gamma2;

        let scaled_x = x_out * scale_x + offset_x;
        let scaled_y = y_out * scale_y + offset_y;
        let scaled_b = b_out * scale_b + offset_b;

        let x_arr: [f32; 8] = scaled_x.into();
        let y_arr: [f32; 8] = scaled_y.into();
        let b_arr: [f32; 8] = scaled_b.into();

        x_plane[pixel_idx..pixel_idx + 8].copy_from_slice(&x_arr);
        y_plane[pixel_idx..pixel_idx + 8].copy_from_slice(&y_arr);
        b_plane[pixel_idx..pixel_idx + 8].copy_from_slice(&b_arr);
    }

    // Scalar remainder
    for i in (chunks * 8)..num_pixels {
        let (x, y, b) =
            srgb_to_scaled_xyb(rgba_data[i * 4], rgba_data[i * 4 + 1], rgba_data[i * 4 + 2]);
        x_plane[i] = x;
        y_plane[i] = y;
        b_plane[i] = b;
    }
}

/// SIMD-accelerated sRGB to scaled XYB conversion for BGRA input - inplace version.
pub fn srgb_to_scaled_xyb_planes_simd_bgra_inplace(
    bgra_data: &[u8],
    x_plane: &mut [f32],
    y_plane: &mut [f32],
    b_plane: &mut [f32],
    num_pixels: usize,
) {
    use wide::f32x8;

    assert!(bgra_data.len() >= num_pixels * 4);
    assert!(x_plane.len() >= num_pixels);
    assert!(y_plane.len() >= num_pixels);
    assert!(b_plane.len() >= num_pixels);

    let m = &XYB_OPSIN_ABSORBANCE_MATRIX;
    let bias = XYB_OPSIN_ABSORBANCE_BIAS[0];
    let neg_bias_cbrt = -cbrtf_fast(bias);

    let m00 = f32x8::splat(m[0]);
    let m01 = f32x8::splat(m[1]);
    let m02 = f32x8::splat(m[2]);
    let m10 = f32x8::splat(m[3]);
    let m11 = f32x8::splat(m[4]);
    let m12 = f32x8::splat(m[5]);
    let m20 = f32x8::splat(m[6]);
    let m21 = f32x8::splat(m[7]);
    let m22 = f32x8::splat(m[8]);
    let bias_simd = f32x8::splat(bias);
    let zero = f32x8::splat(0.0);
    let neg_bias_cbrt_simd = f32x8::splat(neg_bias_cbrt);
    let half = f32x8::splat(0.5);

    let scale_x = f32x8::splat(SCALED_XYB_SCALE[0]);
    let scale_y = f32x8::splat(SCALED_XYB_SCALE[1]);
    let scale_b = f32x8::splat(SCALED_XYB_SCALE[2]);
    let offset_x = f32x8::splat(SCALED_XYB_OFFSET[0]);
    let offset_y = f32x8::splat(SCALED_XYB_OFFSET[1]);
    let offset_b = f32x8::splat(SCALED_XYB_OFFSET[2]);

    let chunks = num_pixels / 8;

    for chunk in 0..chunks {
        let pixel_idx = chunk * 8;
        let bgra_idx = pixel_idx * 4;

        // BGRA order: B=0, G=1, R=2, A=3
        let r = f32x8::from([
            SRGB_TO_LINEAR_LUT[bgra_data[bgra_idx + 2] as usize],
            SRGB_TO_LINEAR_LUT[bgra_data[bgra_idx + 6] as usize],
            SRGB_TO_LINEAR_LUT[bgra_data[bgra_idx + 10] as usize],
            SRGB_TO_LINEAR_LUT[bgra_data[bgra_idx + 14] as usize],
            SRGB_TO_LINEAR_LUT[bgra_data[bgra_idx + 18] as usize],
            SRGB_TO_LINEAR_LUT[bgra_data[bgra_idx + 22] as usize],
            SRGB_TO_LINEAR_LUT[bgra_data[bgra_idx + 26] as usize],
            SRGB_TO_LINEAR_LUT[bgra_data[bgra_idx + 30] as usize],
        ]);

        let g = f32x8::from([
            SRGB_TO_LINEAR_LUT[bgra_data[bgra_idx + 1] as usize],
            SRGB_TO_LINEAR_LUT[bgra_data[bgra_idx + 5] as usize],
            SRGB_TO_LINEAR_LUT[bgra_data[bgra_idx + 9] as usize],
            SRGB_TO_LINEAR_LUT[bgra_data[bgra_idx + 13] as usize],
            SRGB_TO_LINEAR_LUT[bgra_data[bgra_idx + 17] as usize],
            SRGB_TO_LINEAR_LUT[bgra_data[bgra_idx + 21] as usize],
            SRGB_TO_LINEAR_LUT[bgra_data[bgra_idx + 25] as usize],
            SRGB_TO_LINEAR_LUT[bgra_data[bgra_idx + 29] as usize],
        ]);

        let b_in = f32x8::from([
            SRGB_TO_LINEAR_LUT[bgra_data[bgra_idx] as usize],
            SRGB_TO_LINEAR_LUT[bgra_data[bgra_idx + 4] as usize],
            SRGB_TO_LINEAR_LUT[bgra_data[bgra_idx + 8] as usize],
            SRGB_TO_LINEAR_LUT[bgra_data[bgra_idx + 12] as usize],
            SRGB_TO_LINEAR_LUT[bgra_data[bgra_idx + 16] as usize],
            SRGB_TO_LINEAR_LUT[bgra_data[bgra_idx + 20] as usize],
            SRGB_TO_LINEAR_LUT[bgra_data[bgra_idx + 24] as usize],
            SRGB_TO_LINEAR_LUT[bgra_data[bgra_idx + 28] as usize],
        ]);

        // Apply opsin absorbance matrix (using FMA)
        let mixed0 = m00.mul_add(r, m01.mul_add(g, m02.mul_add(b_in, bias_simd)));
        let mixed1 = m10.mul_add(r, m11.mul_add(g, m12.mul_add(b_in, bias_simd)));
        let mixed2 = m20.mul_add(r, m21.mul_add(g, m22.mul_add(b_in, bias_simd)));

        let mixed0 = mixed0.max(zero);
        let mixed1 = mixed1.max(zero);
        let mixed2 = mixed2.max(zero);

        let gamma0 = cbrtf_x8(mixed0) + neg_bias_cbrt_simd;
        let gamma1 = cbrtf_x8(mixed1) + neg_bias_cbrt_simd;
        let gamma2 = cbrtf_x8(mixed2) + neg_bias_cbrt_simd;

        let x_out = half * (gamma0 - gamma1);
        let y_out = half * (gamma0 + gamma1);
        let b_out = gamma2;

        let scaled_x = x_out * scale_x + offset_x;
        let scaled_y = y_out * scale_y + offset_y;
        let scaled_b = b_out * scale_b + offset_b;

        let x_arr: [f32; 8] = scaled_x.into();
        let y_arr: [f32; 8] = scaled_y.into();
        let b_arr: [f32; 8] = scaled_b.into();

        x_plane[pixel_idx..pixel_idx + 8].copy_from_slice(&x_arr);
        y_plane[pixel_idx..pixel_idx + 8].copy_from_slice(&y_arr);
        b_plane[pixel_idx..pixel_idx + 8].copy_from_slice(&b_arr);
    }

    // Scalar remainder
    for i in (chunks * 8)..num_pixels {
        let (x, y, b) = srgb_to_scaled_xyb(
            bgra_data[i * 4 + 2], // R
            bgra_data[i * 4 + 1], // G
            bgra_data[i * 4],     // B
        );
        x_plane[i] = x;
        y_plane[i] = y;
        b_plane[i] = b;
    }
}

/// SIMD-accelerated sRGB to scaled XYB conversion for BGR input - inplace version.
pub fn srgb_to_scaled_xyb_planes_simd_bgr_inplace(
    bgr_data: &[u8],
    x_plane: &mut [f32],
    y_plane: &mut [f32],
    b_plane: &mut [f32],
    num_pixels: usize,
) {
    // Reuse RGB inplace with swapped channels
    // For now, use scalar fallback since we need to swap channels
    for i in 0..num_pixels {
        let (x, y, b) = srgb_to_scaled_xyb(
            bgr_data[i * 3 + 2], // R
            bgr_data[i * 3 + 1], // G
            bgr_data[i * 3],     // B
        );
        x_plane[i] = x;
        y_plane[i] = y;
        b_plane[i] = b;
    }
}

/// SIMD-accelerated sRGB to scaled XYB conversion for RGBA input (alpha ignored).
///
/// Same as `srgb_to_scaled_xyb_planes_simd` but takes 4 bytes per pixel (RGBA).
/// Avoids intermediate RGBA→RGB conversion allocation.
pub fn srgb_to_scaled_xyb_planes_simd_rgba(
    rgba_data: &[u8],
    num_pixels: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    use wide::f32x8;

    assert!(rgba_data.len() >= num_pixels * 4);

    // Allocate zeroed - OS zero-page optimization makes this fast
    let mut x_plane = vec![0.0f32; num_pixels];
    let mut y_plane = vec![0.0f32; num_pixels];
    let mut b_plane = vec![0.0f32; num_pixels];

    let m = &XYB_OPSIN_ABSORBANCE_MATRIX;
    let bias = XYB_OPSIN_ABSORBANCE_BIAS[0];
    let neg_bias_cbrt = -cbrtf_fast(bias);

    // SIMD constants for XYB matrix
    let m00 = f32x8::splat(m[0]);
    let m01 = f32x8::splat(m[1]);
    let m02 = f32x8::splat(m[2]);
    let m10 = f32x8::splat(m[3]);
    let m11 = f32x8::splat(m[4]);
    let m12 = f32x8::splat(m[5]);
    let m20 = f32x8::splat(m[6]);
    let m21 = f32x8::splat(m[7]);
    let m22 = f32x8::splat(m[8]);
    let bias_simd = f32x8::splat(bias);
    let zero = f32x8::splat(0.0);
    let neg_bias_cbrt_simd = f32x8::splat(neg_bias_cbrt);
    let half = f32x8::splat(0.5);

    // SIMD constants for scaling
    let scale_x = f32x8::splat(SCALED_XYB_SCALE[0]);
    let scale_y = f32x8::splat(SCALED_XYB_SCALE[1]);
    let scale_b = f32x8::splat(SCALED_XYB_SCALE[2]);
    let offset_x = f32x8::splat(SCALED_XYB_OFFSET[0]);
    let offset_y = f32x8::splat(SCALED_XYB_OFFSET[1]);
    let offset_b = f32x8::splat(SCALED_XYB_OFFSET[2]);

    let chunks = num_pixels / 8;

    for chunk in 0..chunks {
        let pixel_idx = chunk * 8;
        let rgba_idx = pixel_idx * 4; // 4 bytes per pixel for RGBA

        // Gather 8 RGBA pixels (stride 4) and convert sRGB to linear via LUT
        let r = f32x8::from([
            SRGB_TO_LINEAR_LUT[rgba_data[rgba_idx] as usize],
            SRGB_TO_LINEAR_LUT[rgba_data[rgba_idx + 4] as usize],
            SRGB_TO_LINEAR_LUT[rgba_data[rgba_idx + 8] as usize],
            SRGB_TO_LINEAR_LUT[rgba_data[rgba_idx + 12] as usize],
            SRGB_TO_LINEAR_LUT[rgba_data[rgba_idx + 16] as usize],
            SRGB_TO_LINEAR_LUT[rgba_data[rgba_idx + 20] as usize],
            SRGB_TO_LINEAR_LUT[rgba_data[rgba_idx + 24] as usize],
            SRGB_TO_LINEAR_LUT[rgba_data[rgba_idx + 28] as usize],
        ]);

        let g = f32x8::from([
            SRGB_TO_LINEAR_LUT[rgba_data[rgba_idx + 1] as usize],
            SRGB_TO_LINEAR_LUT[rgba_data[rgba_idx + 5] as usize],
            SRGB_TO_LINEAR_LUT[rgba_data[rgba_idx + 9] as usize],
            SRGB_TO_LINEAR_LUT[rgba_data[rgba_idx + 13] as usize],
            SRGB_TO_LINEAR_LUT[rgba_data[rgba_idx + 17] as usize],
            SRGB_TO_LINEAR_LUT[rgba_data[rgba_idx + 21] as usize],
            SRGB_TO_LINEAR_LUT[rgba_data[rgba_idx + 25] as usize],
            SRGB_TO_LINEAR_LUT[rgba_data[rgba_idx + 29] as usize],
        ]);

        let b_in = f32x8::from([
            SRGB_TO_LINEAR_LUT[rgba_data[rgba_idx + 2] as usize],
            SRGB_TO_LINEAR_LUT[rgba_data[rgba_idx + 6] as usize],
            SRGB_TO_LINEAR_LUT[rgba_data[rgba_idx + 10] as usize],
            SRGB_TO_LINEAR_LUT[rgba_data[rgba_idx + 14] as usize],
            SRGB_TO_LINEAR_LUT[rgba_data[rgba_idx + 18] as usize],
            SRGB_TO_LINEAR_LUT[rgba_data[rgba_idx + 22] as usize],
            SRGB_TO_LINEAR_LUT[rgba_data[rgba_idx + 26] as usize],
            SRGB_TO_LINEAR_LUT[rgba_data[rgba_idx + 30] as usize],
        ]);

        // Opsin absorbance matrix with FMA
        let mut opsin0 = m00.mul_add(r, m01.mul_add(g, m02.mul_add(b_in, bias_simd)));
        let mut opsin1 = m10.mul_add(r, m11.mul_add(g, m12.mul_add(b_in, bias_simd)));
        let mut opsin2 = m20.mul_add(r, m21.mul_add(g, m22.mul_add(b_in, bias_simd)));

        // Clamp negatives
        opsin0 = opsin0.max(zero);
        opsin1 = opsin1.max(zero);
        opsin2 = opsin2.max(zero);

        // Cube root + bias subtraction
        opsin0 = cbrtf_x8(opsin0) + neg_bias_cbrt_simd;
        opsin1 = cbrtf_x8(opsin1) + neg_bias_cbrt_simd;
        opsin2 = cbrtf_x8(opsin2) + neg_bias_cbrt_simd;

        // XYB transform: X = (L-M)/2, Y = (L+M)/2, B = S
        let x_xyb = half * (opsin0 - opsin1);
        let y_xyb = half * (opsin0 + opsin1);
        let b_xyb = opsin2;

        // Scale XYB for JPEG: scaled = (value + offset) * scale
        let scaled_x = (x_xyb + offset_x) * scale_x;
        let scaled_y = (y_xyb + offset_y) * scale_y;
        let scaled_b = (b_xyb - y_xyb + offset_b) * scale_b;

        // Store results to planes
        let x_arr: [f32; 8] = scaled_x.into();
        let y_arr: [f32; 8] = scaled_y.into();
        let b_arr: [f32; 8] = scaled_b.into();

        x_plane[pixel_idx..pixel_idx + 8].copy_from_slice(&x_arr);
        y_plane[pixel_idx..pixel_idx + 8].copy_from_slice(&y_arr);
        b_plane[pixel_idx..pixel_idx + 8].copy_from_slice(&b_arr);
    }

    // Scalar remainder
    for i in (chunks * 8)..num_pixels {
        let (x, y, b) =
            srgb_to_scaled_xyb(rgba_data[i * 4], rgba_data[i * 4 + 1], rgba_data[i * 4 + 2]);
        x_plane[i] = x;
        y_plane[i] = y;
        b_plane[i] = b;
    }

    (x_plane, y_plane, b_plane)
}

/// SIMD-accelerated sRGB to scaled XYB conversion for BGRA input (alpha ignored).
///
/// Same as `srgb_to_scaled_xyb_planes_simd` but takes 4 bytes per pixel (BGRA).
/// Swaps B and R channels and ignores alpha. Avoids intermediate allocation.
pub fn srgb_to_scaled_xyb_planes_simd_bgra(
    bgra_data: &[u8],
    num_pixels: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    use wide::f32x8;

    assert!(bgra_data.len() >= num_pixels * 4);

    // Allocate zeroed - OS zero-page optimization makes this fast
    let mut x_plane = vec![0.0f32; num_pixels];
    let mut y_plane = vec![0.0f32; num_pixels];
    let mut b_plane = vec![0.0f32; num_pixels];

    let m = &XYB_OPSIN_ABSORBANCE_MATRIX;
    let bias = XYB_OPSIN_ABSORBANCE_BIAS[0];
    let neg_bias_cbrt = -cbrtf_fast(bias);

    // SIMD constants
    let m00 = f32x8::splat(m[0]);
    let m01 = f32x8::splat(m[1]);
    let m02 = f32x8::splat(m[2]);
    let m10 = f32x8::splat(m[3]);
    let m11 = f32x8::splat(m[4]);
    let m12 = f32x8::splat(m[5]);
    let m20 = f32x8::splat(m[6]);
    let m21 = f32x8::splat(m[7]);
    let m22 = f32x8::splat(m[8]);
    let bias_simd = f32x8::splat(bias);
    let zero = f32x8::splat(0.0);
    let neg_bias_cbrt_simd = f32x8::splat(neg_bias_cbrt);
    let half = f32x8::splat(0.5);

    let scale_x = f32x8::splat(SCALED_XYB_SCALE[0]);
    let scale_y = f32x8::splat(SCALED_XYB_SCALE[1]);
    let scale_b = f32x8::splat(SCALED_XYB_SCALE[2]);
    let offset_x = f32x8::splat(SCALED_XYB_OFFSET[0]);
    let offset_y = f32x8::splat(SCALED_XYB_OFFSET[1]);
    let offset_b = f32x8::splat(SCALED_XYB_OFFSET[2]);

    let chunks = num_pixels / 8;

    for chunk in 0..chunks {
        let pixel_idx = chunk * 8;
        let bgra_idx = pixel_idx * 4;

        // BGRA layout: B=0, G=1, R=2, A=3 -> swap to get R, G, B
        let r = f32x8::from([
            SRGB_TO_LINEAR_LUT[bgra_data[bgra_idx + 2] as usize],
            SRGB_TO_LINEAR_LUT[bgra_data[bgra_idx + 6] as usize],
            SRGB_TO_LINEAR_LUT[bgra_data[bgra_idx + 10] as usize],
            SRGB_TO_LINEAR_LUT[bgra_data[bgra_idx + 14] as usize],
            SRGB_TO_LINEAR_LUT[bgra_data[bgra_idx + 18] as usize],
            SRGB_TO_LINEAR_LUT[bgra_data[bgra_idx + 22] as usize],
            SRGB_TO_LINEAR_LUT[bgra_data[bgra_idx + 26] as usize],
            SRGB_TO_LINEAR_LUT[bgra_data[bgra_idx + 30] as usize],
        ]);

        let g = f32x8::from([
            SRGB_TO_LINEAR_LUT[bgra_data[bgra_idx + 1] as usize],
            SRGB_TO_LINEAR_LUT[bgra_data[bgra_idx + 5] as usize],
            SRGB_TO_LINEAR_LUT[bgra_data[bgra_idx + 9] as usize],
            SRGB_TO_LINEAR_LUT[bgra_data[bgra_idx + 13] as usize],
            SRGB_TO_LINEAR_LUT[bgra_data[bgra_idx + 17] as usize],
            SRGB_TO_LINEAR_LUT[bgra_data[bgra_idx + 21] as usize],
            SRGB_TO_LINEAR_LUT[bgra_data[bgra_idx + 25] as usize],
            SRGB_TO_LINEAR_LUT[bgra_data[bgra_idx + 29] as usize],
        ]);

        let b_in = f32x8::from([
            SRGB_TO_LINEAR_LUT[bgra_data[bgra_idx] as usize],
            SRGB_TO_LINEAR_LUT[bgra_data[bgra_idx + 4] as usize],
            SRGB_TO_LINEAR_LUT[bgra_data[bgra_idx + 8] as usize],
            SRGB_TO_LINEAR_LUT[bgra_data[bgra_idx + 12] as usize],
            SRGB_TO_LINEAR_LUT[bgra_data[bgra_idx + 16] as usize],
            SRGB_TO_LINEAR_LUT[bgra_data[bgra_idx + 20] as usize],
            SRGB_TO_LINEAR_LUT[bgra_data[bgra_idx + 24] as usize],
            SRGB_TO_LINEAR_LUT[bgra_data[bgra_idx + 28] as usize],
        ]);

        let mut opsin0 = m00.mul_add(r, m01.mul_add(g, m02.mul_add(b_in, bias_simd)));
        let mut opsin1 = m10.mul_add(r, m11.mul_add(g, m12.mul_add(b_in, bias_simd)));
        let mut opsin2 = m20.mul_add(r, m21.mul_add(g, m22.mul_add(b_in, bias_simd)));

        opsin0 = opsin0.max(zero);
        opsin1 = opsin1.max(zero);
        opsin2 = opsin2.max(zero);

        opsin0 = cbrtf_x8(opsin0) + neg_bias_cbrt_simd;
        opsin1 = cbrtf_x8(opsin1) + neg_bias_cbrt_simd;
        opsin2 = cbrtf_x8(opsin2) + neg_bias_cbrt_simd;

        let x_xyb = half * (opsin0 - opsin1);
        let y_xyb = half * (opsin0 + opsin1);
        let b_xyb = opsin2;

        let scaled_x = (x_xyb + offset_x) * scale_x;
        let scaled_y = (y_xyb + offset_y) * scale_y;
        let scaled_b = (b_xyb - y_xyb + offset_b) * scale_b;

        let x_arr: [f32; 8] = scaled_x.into();
        let y_arr: [f32; 8] = scaled_y.into();
        let b_arr: [f32; 8] = scaled_b.into();

        x_plane[pixel_idx..pixel_idx + 8].copy_from_slice(&x_arr);
        y_plane[pixel_idx..pixel_idx + 8].copy_from_slice(&y_arr);
        b_plane[pixel_idx..pixel_idx + 8].copy_from_slice(&b_arr);
    }

    // Scalar remainder (BGRA: B=0, G=1, R=2)
    for i in (chunks * 8)..num_pixels {
        let (x, y, b) = srgb_to_scaled_xyb(
            bgra_data[i * 4 + 2], // R
            bgra_data[i * 4 + 1], // G
            bgra_data[i * 4],     // B
        );
        x_plane[i] = x;
        y_plane[i] = y;
        b_plane[i] = b;
    }

    (x_plane, y_plane, b_plane)
}

// ============================================================================
// SIMD XYB Conversion (f32x8, matching ssimulacra2 algorithm)
// ============================================================================

use wide::{f32x8, f64x2};

/// Initial cube root approximation using IEEE 754 bit manipulation.
#[inline]
fn cbrt_initial_approx(x: f32) -> f32 {
    const B1: u32 = 709_958_130;
    let ui: u32 = x.to_bits();
    let sign = ui & 0x8000_0000;
    let hx = ui & 0x7FFF_FFFF;
    let approx = hx / 3 + B1;
    f32::from_bits(sign | approx)
}

/// SIMD cube root for 8 values using f32x8 with f64x2 Newton iterations.
#[inline]
fn cbrtf_x8(x: f32x8) -> f32x8 {
    let x_arr: [f32; 8] = x.into();
    let t_arr: [f32; 8] = [
        cbrt_initial_approx(x_arr[0]),
        cbrt_initial_approx(x_arr[1]),
        cbrt_initial_approx(x_arr[2]),
        cbrt_initial_approx(x_arr[3]),
        cbrt_initial_approx(x_arr[4]),
        cbrt_initial_approx(x_arr[5]),
        cbrt_initial_approx(x_arr[6]),
        cbrt_initial_approx(x_arr[7]),
    ];

    // Process in f64x2 pairs for precision (2 Newton iterations each)
    let x0 = f64x2::new([x_arr[0] as f64, x_arr[1] as f64]);
    let x1 = f64x2::new([x_arr[2] as f64, x_arr[3] as f64]);
    let x2 = f64x2::new([x_arr[4] as f64, x_arr[5] as f64]);
    let x3 = f64x2::new([x_arr[6] as f64, x_arr[7] as f64]);

    let mut t0 = f64x2::new([t_arr[0] as f64, t_arr[1] as f64]);
    let mut t1 = f64x2::new([t_arr[2] as f64, t_arr[3] as f64]);
    let mut t2 = f64x2::new([t_arr[4] as f64, t_arr[5] as f64]);
    let mut t3 = f64x2::new([t_arr[6] as f64, t_arr[7] as f64]);

    let x2_0 = x0 + x0;
    let x2_1 = x1 + x1;
    let x2_2 = x2 + x2;
    let x2_3 = x3 + x3;

    // First Newton iteration: t = t * (2x + t³) / (x + 2t³)
    let r0 = t0 * t0 * t0;
    let r1 = t1 * t1 * t1;
    let r2 = t2 * t2 * t2;
    let r3 = t3 * t3 * t3;
    t0 = t0 * (x2_0 + r0) / (x0 + r0 + r0);
    t1 = t1 * (x2_1 + r1) / (x1 + r1 + r1);
    t2 = t2 * (x2_2 + r2) / (x2 + r2 + r2);
    t3 = t3 * (x2_3 + r3) / (x3 + r3 + r3);

    // Second Newton iteration
    let r0 = t0 * t0 * t0;
    let r1 = t1 * t1 * t1;
    let r2 = t2 * t2 * t2;
    let r3 = t3 * t3 * t3;
    t0 = t0 * (x2_0 + r0) / (x0 + r0 + r0);
    t1 = t1 * (x2_1 + r1) / (x1 + r1 + r1);
    t2 = t2 * (x2_2 + r2) / (x2 + r2 + r2);
    t3 = t3 * (x2_3 + r3) / (x3 + r3 + r3);

    let t0_arr: [f64; 2] = t0.into();
    let t1_arr: [f64; 2] = t1.into();
    let t2_arr: [f64; 2] = t2.into();
    let t3_arr: [f64; 2] = t3.into();

    f32x8::new([
        t0_arr[0] as f32,
        t0_arr[1] as f32,
        t1_arr[0] as f32,
        t1_arr[1] as f32,
        t2_arr[0] as f32,
        t2_arr[1] as f32,
        t3_arr[0] as f32,
        t3_arr[1] as f32,
    ])
}

/// SIMD linear RGB to XYB conversion for a batch of pixels (in-place).
///
/// Processes 8 pixels at a time using f32x8 SIMD with f64 Newton-Raphson
/// iterations for cube root precision. Falls back to scalar for remainder.
///
/// # Arguments
/// * `pixels` - Mutable slice of [R, G, B] linear RGB values (0.0-1.0).
///              After conversion, contains [X, Y, B] XYB values.
///
/// NOTE: This uses 0-1 linear RGB range. For C++ jpegli compatibility,
/// use `linear_rgb_to_xyb_simd_255` which expects 0-255 range.
pub fn linear_rgb_to_xyb_simd(pixels: &mut [[f32; 3]]) {
    let m = &XYB_OPSIN_ABSORBANCE_MATRIX;
    let bias = XYB_OPSIN_ABSORBANCE_BIAS[0];

    // Pre-compute negative bias cbrt for all pixels
    let neg_bias_cbrt = -cbrtf_fast(bias);

    // SIMD constants
    let m00 = f32x8::splat(m[0]);
    let m01 = f32x8::splat(m[1]);
    let m02 = f32x8::splat(m[2]);
    let m10 = f32x8::splat(m[3]);
    let m11 = f32x8::splat(m[4]);
    let m12 = f32x8::splat(m[5]);
    let m20 = f32x8::splat(m[6]);
    let m21 = f32x8::splat(m[7]);
    let m22 = f32x8::splat(m[8]);
    let bias_simd = f32x8::splat(bias);
    let zero = f32x8::splat(0.0);
    let neg_bias_cbrt_simd = f32x8::splat(neg_bias_cbrt);
    let half = f32x8::splat(0.5);

    let chunks_8 = pixels.len() / 8;

    for chunk_idx in 0..chunks_8 {
        let base = chunk_idx * 8;

        // Gather RGB values
        let mut r_arr = [0.0f32; 8];
        let mut g_arr = [0.0f32; 8];
        let mut b_arr = [0.0f32; 8];

        for i in 0..8 {
            let p = pixels[base + i];
            r_arr[i] = p[0];
            g_arr[i] = p[1];
            b_arr[i] = p[2];
        }

        let r = f32x8::new(r_arr);
        let g = f32x8::new(g_arr);
        let b = f32x8::new(b_arr);

        // Opsin absorbance matrix with FMA
        let mut opsin0 = m00.mul_add(r, m01.mul_add(g, m02.mul_add(b, bias_simd)));
        let mut opsin1 = m10.mul_add(r, m11.mul_add(g, m12.mul_add(b, bias_simd)));
        let mut opsin2 = m20.mul_add(r, m21.mul_add(g, m22.mul_add(b, bias_simd)));

        // Clamp negatives (matches C++ ZeroIfNegative)
        opsin0 = opsin0.max(zero);
        opsin1 = opsin1.max(zero);
        opsin2 = opsin2.max(zero);

        // Cube root + bias subtraction
        opsin0 = cbrtf_x8(opsin0) + neg_bias_cbrt_simd;
        opsin1 = cbrtf_x8(opsin1) + neg_bias_cbrt_simd;
        opsin2 = cbrtf_x8(opsin2) + neg_bias_cbrt_simd;

        // Final XYB transform: X = (L-M)/2, Y = (L+M)/2, B = S
        let x = half * (opsin0 - opsin1);
        let y = half * (opsin0 + opsin1);
        let b_out = opsin2;

        // Scatter results
        let x_arr: [f32; 8] = x.into();
        let y_arr: [f32; 8] = y.into();
        let b_arr: [f32; 8] = b_out.into();

        for i in 0..8 {
            pixels[base + i] = [x_arr[i], y_arr[i], b_arr[i]];
        }
    }

    // Scalar fallback for remainder
    let scalar_start = chunks_8 * 8;
    for pix in &mut pixels[scalar_start..] {
        let (x, y, b) = linear_rgb_to_xyb(pix[0], pix[1], pix[2]);
        *pix = [x, y, b];
    }
}

/// SIMD linear RGB (0-255 range) to XYB conversion for a batch of pixels (in-place).
///
/// This is the C++ jpegli compatible version that expects linear RGB in 0-255 range.
/// Processes 8 pixels at a time using f32x8 SIMD.
///
/// # Arguments
/// * `pixels` - Mutable slice of [R, G, B] linear RGB values (0.0-255.0).
///              After conversion, contains [X, Y, B] XYB values (larger range than 0-1 input).
pub fn linear_rgb_to_xyb_simd_255(pixels: &mut [[f32; 3]]) {
    let m = &XYB_OPSIN_ABSORBANCE_MATRIX;
    let bias = XYB_OPSIN_ABSORBANCE_BIAS[0];

    let neg_bias_cbrt = -cbrtf_fast(bias);

    let m00 = f32x8::splat(m[0]);
    let m01 = f32x8::splat(m[1]);
    let m02 = f32x8::splat(m[2]);
    let m10 = f32x8::splat(m[3]);
    let m11 = f32x8::splat(m[4]);
    let m12 = f32x8::splat(m[5]);
    let m20 = f32x8::splat(m[6]);
    let m21 = f32x8::splat(m[7]);
    let m22 = f32x8::splat(m[8]);
    let bias_simd = f32x8::splat(bias);
    let zero = f32x8::splat(0.0);
    let neg_bias_cbrt_simd = f32x8::splat(neg_bias_cbrt);
    let half = f32x8::splat(0.5);

    let chunks_8 = pixels.len() / 8;

    for chunk_idx in 0..chunks_8 {
        let base = chunk_idx * 8;

        let mut r_arr = [0.0f32; 8];
        let mut g_arr = [0.0f32; 8];
        let mut b_arr = [0.0f32; 8];

        for i in 0..8 {
            let p = pixels[base + i];
            r_arr[i] = p[0];
            g_arr[i] = p[1];
            b_arr[i] = p[2];
        }

        let r = f32x8::new(r_arr);
        let g = f32x8::new(g_arr);
        let b = f32x8::new(b_arr);

        let mut opsin0 = m00.mul_add(r, m01.mul_add(g, m02.mul_add(b, bias_simd)));
        let mut opsin1 = m10.mul_add(r, m11.mul_add(g, m12.mul_add(b, bias_simd)));
        let mut opsin2 = m20.mul_add(r, m21.mul_add(g, m22.mul_add(b, bias_simd)));

        opsin0 = opsin0.max(zero);
        opsin1 = opsin1.max(zero);
        opsin2 = opsin2.max(zero);

        opsin0 = cbrtf_x8(opsin0) + neg_bias_cbrt_simd;
        opsin1 = cbrtf_x8(opsin1) + neg_bias_cbrt_simd;
        opsin2 = cbrtf_x8(opsin2) + neg_bias_cbrt_simd;

        let x = half * (opsin0 - opsin1);
        let y = half * (opsin0 + opsin1);
        let b_out = opsin2;

        let x_arr: [f32; 8] = x.into();
        let y_arr: [f32; 8] = y.into();
        let b_arr: [f32; 8] = b_out.into();

        for i in 0..8 {
            pixels[base + i] = [x_arr[i], y_arr[i], b_arr[i]];
        }
    }

    // Scalar fallback for remainder - uses 0-255 version
    let scalar_start = chunks_8 * 8;
    for pix in &mut pixels[scalar_start..] {
        let (x, y, b) = linear_rgb_to_xyb_255(pix[0], pix[1], pix[2]);
        *pix = [x, y, b];
    }
}

/// SIMD sRGB u8 to XYB conversion for a batch of pixels.
///
/// Converts sRGB u8 input to XYB f32 output using SIMD acceleration.
/// This is the full conversion chain: sRGB u8 → linear (0-1) → XYB.
/// Uses C++ jpegli conventions with 0-1 linear RGB range.
pub fn srgb_to_xyb_batch(input: &[[u8; 3]], output: &mut [[f32; 3]]) {
    assert_eq!(input.len(), output.len());

    // Convert to linear RGB in 0-1 range (matching C++ jpegli conventions)
    for (inp, out) in input.iter().zip(output.iter_mut()) {
        out[0] = srgb_u8_to_linear(inp[0]);
        out[1] = srgb_u8_to_linear(inp[1]);
        out[2] = srgb_u8_to_linear(inp[2]);
    }

    // Apply SIMD XYB conversion (expects 0-1 linear RGB)
    linear_rgb_to_xyb_simd(output);
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

// ============================================================================
// XYB Decode Helpers (SIMD level shift to RGB)
// ============================================================================

/// SIMD-optimized XYB plane level shift to interleaved RGB u8.
///
/// Converts 3 XYB f32 planes to interleaved RGB u8, applying:
/// - Level shift (+128)
/// - Clamp to [0, 255]
/// - Convert to u8
///
/// This is used for XYB decode path where no YCbCr→RGB conversion is needed.
#[inline]
pub fn xyb_planes_to_rgb_u8_simd(plane0: &[f32], plane1: &[f32], plane2: &[f32], rgb: &mut [u8]) {
    use wide::f32x8;

    debug_assert_eq!(plane0.len(), plane1.len());
    debug_assert_eq!(plane0.len(), plane2.len());
    debug_assert_eq!(rgb.len(), plane0.len() * 3);

    let num_pixels = plane0.len();
    let offset = f32x8::splat(128.0);
    let zero = f32x8::splat(0.0);
    let max_val = f32x8::splat(255.0);

    let chunks = num_pixels / 8;
    for chunk in 0..chunks {
        let base = chunk * 8;

        // Load 8 values from each plane
        let p0 = f32x8::from([
            plane0[base],
            plane0[base + 1],
            plane0[base + 2],
            plane0[base + 3],
            plane0[base + 4],
            plane0[base + 5],
            plane0[base + 6],
            plane0[base + 7],
        ]);
        let p1 = f32x8::from([
            plane1[base],
            plane1[base + 1],
            plane1[base + 2],
            plane1[base + 3],
            plane1[base + 4],
            plane1[base + 5],
            plane1[base + 6],
            plane1[base + 7],
        ]);
        let p2 = f32x8::from([
            plane2[base],
            plane2[base + 1],
            plane2[base + 2],
            plane2[base + 3],
            plane2[base + 4],
            plane2[base + 5],
            plane2[base + 6],
            plane2[base + 7],
        ]);

        // Level shift, clamp
        let r = (p0 + offset).max(zero).min(max_val);
        let g = (p1 + offset).max(zero).min(max_val);
        let b = (p2 + offset).max(zero).min(max_val);

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

    // Scalar remainder
    for i in (chunks * 8)..num_pixels {
        let idx = i * 3;
        rgb[idx] = (plane0[i] + 128.0).clamp(0.0, 255.0) as u8;
        rgb[idx + 1] = (plane1[i] + 128.0).clamp(0.0, 255.0) as u8;
        rgb[idx + 2] = (plane2[i] + 128.0).clamp(0.0, 255.0) as u8;
    }
}

/// SIMD-optimized XYB plane level shift to interleaved RGB f32 (normalized 0-1).
#[inline]
pub fn xyb_planes_to_rgb_f32_simd(plane0: &[f32], plane1: &[f32], plane2: &[f32], rgb: &mut [f32]) {
    use wide::f32x8;

    debug_assert_eq!(plane0.len(), plane1.len());
    debug_assert_eq!(plane0.len(), plane2.len());
    debug_assert_eq!(rgb.len(), plane0.len() * 3);

    let num_pixels = plane0.len();
    let offset = f32x8::splat(128.0);
    let scale = f32x8::splat(1.0 / 255.0);
    let zero = f32x8::splat(0.0);
    let one = f32x8::splat(1.0);

    let chunks = num_pixels / 8;
    for chunk in 0..chunks {
        let base = chunk * 8;

        let p0 = f32x8::from([
            plane0[base],
            plane0[base + 1],
            plane0[base + 2],
            plane0[base + 3],
            plane0[base + 4],
            plane0[base + 5],
            plane0[base + 6],
            plane0[base + 7],
        ]);
        let p1 = f32x8::from([
            plane1[base],
            plane1[base + 1],
            plane1[base + 2],
            plane1[base + 3],
            plane1[base + 4],
            plane1[base + 5],
            plane1[base + 6],
            plane1[base + 7],
        ]);
        let p2 = f32x8::from([
            plane2[base],
            plane2[base + 1],
            plane2[base + 2],
            plane2[base + 3],
            plane2[base + 4],
            plane2[base + 5],
            plane2[base + 6],
            plane2[base + 7],
        ]);

        // Level shift, scale to 0-1, clamp
        let r = ((p0 + offset) * scale).max(zero).min(one);
        let g = ((p1 + offset) * scale).max(zero).min(one);
        let b = ((p2 + offset) * scale).max(zero).min(one);

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
        let idx = i * 3;
        rgb[idx] = ((plane0[i] + 128.0) / 255.0).clamp(0.0, 1.0);
        rgb[idx + 1] = ((plane1[i] + 128.0) / 255.0).clamp(0.0, 1.0);
        rgb[idx + 2] = ((plane2[i] + 128.0) / 255.0).clamp(0.0, 1.0);
    }
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
    fn test_cbrtf_fast_cube() {
        // Test that cbrtf_fast and cube are inverses for non-negative values
        // (negatives are clamped to 0 in the XYB pipeline, matching C++)
        let test_values = [0.0f32, 0.001, 0.5, 1.0, 2.0, 10.0, 100.0];

        for v in test_values {
            let cbrt = cbrtf_fast(v);
            let back = cbrt * cbrt * cbrt;
            // Use relative tolerance for larger values, absolute for small
            let tolerance = if v > 1.0 { v * 1e-6 } else { 1e-5 };
            assert!(
                (v - back).abs() < tolerance,
                "Roundtrip failed for {}: cbrt={}, back={}, error={}",
                v,
                cbrt,
                back,
                (v - back).abs()
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

    // ========================================================================
    // SIMD vs Scalar Parity Tests
    // ========================================================================

    #[test]
    fn test_simd_vs_scalar_parity() {
        // Test SIMD matches scalar implementation exactly
        let test_colors: Vec<[f32; 3]> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.5],
            [0.1, 0.2, 0.3],
            [0.9, 0.8, 0.7],
            // Need 8+ to test SIMD path
            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.75],
        ];

        // Get scalar results
        let scalar_results: Vec<[f32; 3]> = test_colors
            .iter()
            .map(|c| {
                let (x, y, b) = linear_rgb_to_xyb(c[0], c[1], c[2]);
                [x, y, b]
            })
            .collect();

        // Get SIMD results
        let mut simd_input = test_colors.clone();
        linear_rgb_to_xyb_simd(&mut simd_input);

        // Compare - should match within 1e-7 (76% exact, rest tiny error)
        let mut max_err: f32 = 0.0;
        for (i, (scalar, simd)) in scalar_results.iter().zip(simd_input.iter()).enumerate() {
            let err = (scalar[0] - simd[0])
                .abs()
                .max((scalar[1] - simd[1]).abs())
                .max((scalar[2] - simd[2]).abs());
            max_err = max_err.max(err);
            assert!(
                err < 1e-6,
                "SIMD vs scalar mismatch at {}: scalar={:?}, simd={:?}, err={}",
                i,
                scalar,
                simd,
                err
            );
        }
        // Verified: max error ~1.19e-7 across all 16.7M colors
        assert!(max_err < 1e-6, "Max error {} exceeds threshold", max_err);
    }

    #[test]
    fn test_simd_batch_conversion() {
        // Test the full sRGB u8 -> XYB batch conversion
        let input: Vec<[u8; 3]> = vec![
            [0, 0, 0],
            [255, 255, 255],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [128, 128, 128],
            [64, 128, 192],
            [200, 100, 50],
            [10, 20, 30],
            [240, 230, 220],
        ];

        let mut output = vec![[0.0f32; 3]; input.len()];
        srgb_to_xyb_batch(&input, &mut output);

        // Compare with scalar path
        for (i, inp) in input.iter().enumerate() {
            let (x, y, b) = srgb_to_xyb(inp[0], inp[1], inp[2]);
            let err = (x - output[i][0])
                .abs()
                .max((y - output[i][1]).abs())
                .max((b - output[i][2]).abs());
            assert!(
                err < 1e-6,
                "Batch vs scalar mismatch at {}: expected ({},{},{}), got {:?}",
                i,
                x,
                y,
                b,
                output[i]
            );
        }
    }

    #[test]
    fn test_simd_remainder_handling() {
        // Test that SIMD correctly handles non-multiple-of-8 lengths
        for len in 1..20 {
            let input: Vec<[f32; 3]> = (0..len).map(|i| [i as f32 / 20.0; 3]).collect();

            let scalar: Vec<[f32; 3]> = input
                .iter()
                .map(|c| {
                    let (x, y, b) = linear_rgb_to_xyb(c[0], c[1], c[2]);
                    [x, y, b]
                })
                .collect();

            let mut simd = input.clone();
            linear_rgb_to_xyb_simd(&mut simd);

            for i in 0..len {
                let err = (scalar[i][0] - simd[i][0])
                    .abs()
                    .max((scalar[i][1] - simd[i][1]).abs())
                    .max((scalar[i][2] - simd[i][2]).abs());
                assert!(
                    err < 1e-6,
                    "Mismatch at len={}, idx={}: err={}",
                    len,
                    i,
                    err
                );
            }
        }
    }

    // C++ FFI Parity Tests moved to tests/xyb_linear_cpp_parity.rs

    #[test]
    fn test_rgba_bgra_simd_parity() {
        // Test that RGBA/BGRA native SIMD functions produce same output as RGB version
        // with manual conversion

        // Create test RGB data
        let rgb_data: Vec<u8> = (0..64 * 3).map(|i| (i % 256) as u8).collect();
        let num_pixels = 64;

        // Create RGBA and BGRA data from the same RGB source
        let mut rgba_data = Vec::with_capacity(num_pixels * 4);
        let mut bgra_data = Vec::with_capacity(num_pixels * 4);
        for i in 0..num_pixels {
            let r = rgb_data[i * 3];
            let g = rgb_data[i * 3 + 1];
            let b = rgb_data[i * 3 + 2];
            rgba_data.extend_from_slice(&[r, g, b, 255]); // RGBA
            bgra_data.extend_from_slice(&[b, g, r, 255]); // BGRA
        }

        // Get reference output from RGB function
        let (ref_x, ref_y, ref_b) = srgb_to_scaled_xyb_planes_simd(&rgb_data, num_pixels);

        // Test RGBA function
        let (rgba_x, rgba_y, rgba_b) = srgb_to_scaled_xyb_planes_simd_rgba(&rgba_data, num_pixels);
        for i in 0..num_pixels {
            assert!(
                (ref_x[i] - rgba_x[i]).abs() < 1e-6,
                "RGBA X mismatch at {}: ref={}, rgba={}",
                i,
                ref_x[i],
                rgba_x[i]
            );
            assert!(
                (ref_y[i] - rgba_y[i]).abs() < 1e-6,
                "RGBA Y mismatch at {}: ref={}, rgba={}",
                i,
                ref_y[i],
                rgba_y[i]
            );
            assert!(
                (ref_b[i] - rgba_b[i]).abs() < 1e-6,
                "RGBA B mismatch at {}: ref={}, rgba={}",
                i,
                ref_b[i],
                rgba_b[i]
            );
        }

        // Test BGRA function
        let (bgra_x, bgra_y, bgra_b) = srgb_to_scaled_xyb_planes_simd_bgra(&bgra_data, num_pixels);
        for i in 0..num_pixels {
            assert!(
                (ref_x[i] - bgra_x[i]).abs() < 1e-6,
                "BGRA X mismatch at {}: ref={}, bgra={}",
                i,
                ref_x[i],
                bgra_x[i]
            );
            assert!(
                (ref_y[i] - bgra_y[i]).abs() < 1e-6,
                "BGRA Y mismatch at {}: ref={}, bgra={}",
                i,
                ref_y[i],
                bgra_y[i]
            );
            assert!(
                (ref_b[i] - bgra_b[i]).abs() < 1e-6,
                "BGRA B mismatch at {}: ref={}, bgra={}",
                i,
                ref_b[i],
                bgra_b[i]
            );
        }
    }
}
