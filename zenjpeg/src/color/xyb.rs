//! XYB color space conversion.
//!
//! XYB is jpegli's perceptually optimized color space that provides better
//! compression quality compared to YCbCr for the same file size.
//!
//! ## Transform Pipeline
//!
//! The encoding transform is:
//! 1. sRGB u8 → Linear RGB (via LUT or gamma formula)
//! 2. Apply opsin absorbance matrix (LMS-like transform)
//! 3. Cube root for perceptual uniformity
//! 4. Final XYB matrix: X = (L-M)/2, Y = (L+M)/2, B = S
//! 5. Scale for JPEG encoding
//!
//! ## Module Organization
//!
//! - **Constants**: Scaling factors, LUTs, matrices
//! - **Gamma conversion**: sRGB ↔ linear RGB
//! - **Scalar XYB**: Reference implementations
//! - **SIMD helpers**: Cube root, matrix operations
//! - **SIMD XYB conversion**: High-performance pixel format converters

#![allow(dead_code)] // Reference implementations and alternative codepaths

use crate::foundation::consts::{
    XYB_NEG_OPSIN_ABSORBANCE_BIAS_CBRT, XYB_OPSIN_ABSORBANCE_BIAS, XYB_OPSIN_ABSORBANCE_MATRIX,
};
use wide::{f32x8, f64x2};

#[cfg(target_arch = "x86_64")]
use archmage::{SimdToken, arcane};

// ============================================================================
// CONSTANTS
// ============================================================================

/// Scaling offset for jpegli XYB encoding (applied before scale multiplication).
pub const SCALED_XYB_OFFSET: [f32; 3] = [0.015_386_134, 0.0, 0.277_704_59];

/// Scaling factor for jpegli XYB encoding (maps XYB to JPEG-suitable range).
pub const SCALED_XYB_SCALE: [f32; 3] = [22.995_788_804, 1.183_000_077, 1.502_141_333];

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

// ============================================================================
// GAMMA CONVERSION (sRGB ↔ Linear RGB)
// ============================================================================

/// Fast 2^x approximation using IEEE 754 bit manipulation.
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

/// Applies sRGB gamma decoding (sRGB → linear RGB).
/// Uses exact formula with powf.
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

/// sRGB to linear using C++ jpegli's rational polynomial approximation.
///
/// Matches `TF_SRGB::DisplayFromEncoded` in libjxl's transfer_functions-inl.h.
#[inline]
#[must_use]
fn srgb_to_linear_poly(x: f32) -> f32 {
    const THRESH: f32 = 0.04045;
    const LOW_DIV_INV: f32 = 1.0 / 12.92;

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

/// Applies sRGB gamma encoding (linear RGB → sRGB).
#[inline]
#[must_use]
pub fn linear_to_srgb(v: f32) -> f32 {
    if v <= 0.003_130_8 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

/// Fast sRGB gamma encoding using fastpow.
#[inline]
#[must_use]
pub fn linear_to_srgb_fast(v: f32) -> f32 {
    if v <= 0.003_130_8 {
        v * 12.92
    } else {
        1.055 * fastpow(v, 1.0 / 2.4) - 0.055
    }
}

/// Converts sRGB u8 to linear f32 using LUT (exact, fastest).
#[inline]
#[must_use]
pub fn srgb_u8_to_linear(v: u8) -> f32 {
    SRGB_TO_LINEAR_LUT[v as usize]
}

/// Converts sRGB u8 to linear f32 using exact formula (for verification).
#[inline]
#[must_use]
pub fn srgb_u8_to_linear_exact(v: u8) -> f32 {
    srgb_to_linear(v as f32 / 255.0)
}

/// Converts linear f32 to sRGB u8.
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

// ============================================================================
// SCALAR CUBE ROOT (matches C++ jpegli)
// ============================================================================

/// Fast cube root matching C++ jpegli's CubeRootAndAdd algorithm.
///
/// Uses IEEE 754 bit manipulation + 3 Newton-Raphson iterations in f32.
/// Maximum error: ~6 ULP.
#[inline]
#[must_use]
fn cbrtf_fast(x: f32) -> f32 {
    if x == 0.0 {
        return 0.0;
    }

    const K_EXP_BIAS: u32 = 0x5480_0000;
    const K_EXP_MUL: u32 = 0x002A_AAAA;
    const K1_3: f32 = 1.0 / 3.0;
    const K4_3: f32 = 4.0 / 3.0;

    let xa = x;
    let xa_3 = K1_3 * xa;

    let m1 = xa.to_bits() as i32;
    let m2 = if m1 == 0 {
        0
    } else {
        (K_EXP_BIAS as i32) - ((m1 >> 23) * (K_EXP_MUL as i32))
    };
    let mut r = f32::from_bits(m2 as u32);

    // 3 Newton-Raphson iterations
    for _ in 0..3 {
        let r2 = r * r;
        r = K4_3 * r - xa_3 * r2 * r2;
    }

    // Final iteration for extra precision
    let r2 = r * r;
    r = r + K1_3 * (r - xa * r2 * r2);

    // Convert from 1/cbrt(x) to cbrt(x)
    let r2 = r * r;
    r2 * x
}

// ============================================================================
// SCALAR XYB CONVERSION
// ============================================================================

/// Inverse of cube root for negative handling.
#[inline]
#[must_use]
fn mixed_cube(v: f32) -> f32 {
    if v < 0.0 { -((-v).powi(3)) } else { v.powi(3) }
}

/// Converts linear RGB (0.0-1.0) to XYB color space.
///
/// This is the core XYB transform used by all conversion functions.
#[must_use]
pub fn linear_rgb_to_xyb(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let m = &XYB_OPSIN_ABSORBANCE_MATRIX;
    let bias = XYB_OPSIN_ABSORBANCE_BIAS[0];

    // Step 1: Opsin absorbance matrix
    let opsin_r = m[0].mul_add(r, m[1].mul_add(g, m[2].mul_add(b, bias)));
    let opsin_g = m[3].mul_add(r, m[4].mul_add(g, m[5].mul_add(b, bias)));
    let opsin_b = m[6].mul_add(r, m[7].mul_add(g, m[8].mul_add(b, bias)));

    // Step 2: Clamp negatives
    let opsin_r = opsin_r.max(0.0);
    let opsin_g = opsin_g.max(0.0);
    let opsin_b = opsin_b.max(0.0);

    // Step 3: Cube root with bias subtraction
    let neg_bias_cbrt = -cbrtf_fast(bias);
    let cbrt_r = cbrtf_fast(opsin_r) + neg_bias_cbrt;
    let cbrt_g = cbrtf_fast(opsin_g) + neg_bias_cbrt;
    let cbrt_b = cbrtf_fast(opsin_b) + neg_bias_cbrt;

    // Step 4: Final XYB transform
    let x = 0.5 * (cbrt_r - cbrt_g);
    let y = 0.5 * (cbrt_r + cbrt_g);

    (x, y, cbrt_b)
}

/// Converts linear RGB (0.0-255.0) to XYB - matches C++ jpegli's LinearRGBRowToXYB.
#[must_use]
pub fn linear_rgb_to_xyb_255(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let m = &XYB_OPSIN_ABSORBANCE_MATRIX;
    let bias = XYB_OPSIN_ABSORBANCE_BIAS[0];

    let opsin_r = m[0].mul_add(r, m[1].mul_add(g, m[2].mul_add(b, bias)));
    let opsin_g = m[3].mul_add(r, m[4].mul_add(g, m[5].mul_add(b, bias)));
    let opsin_b = m[6].mul_add(r, m[7].mul_add(g, m[8].mul_add(b, bias)));

    let opsin_r = opsin_r.max(0.0);
    let opsin_g = opsin_g.max(0.0);
    let opsin_b = opsin_b.max(0.0);

    let neg_bias_cbrt = -cbrtf_fast(bias);
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
    let neg_bias = &XYB_NEG_OPSIN_ABSORBANCE_BIAS_CBRT;

    let cbrt_r = y + x - neg_bias[0];
    let cbrt_g = y - x - neg_bias[1];
    let cbrt_b = b - neg_bias[2];

    let opsin_r = mixed_cube(cbrt_r);
    let opsin_g = mixed_cube(cbrt_g);
    let opsin_b = mixed_cube(cbrt_b);

    let bias = &XYB_OPSIN_ABSORBANCE_BIAS;
    let opsin_r = opsin_r - bias[0];
    let opsin_g = opsin_g - bias[1];
    let opsin_b = opsin_b - bias[2];

    const INV_OPSIN: [f32; 9] = [
        11.031_567, -9.866_944, -0.164_623, -3.254_147, 4.418_770, -0.164_623, -3.658_851,
        2.712_923, 1.945_928,
    ];

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
/// **IMPORTANT**: B channel uses Y in the calculation: `scaled_b = (b - y + offset) * scale`
#[inline]
#[must_use]
pub fn scale_xyb(x: f32, y: f32, b: f32) -> (f32, f32, f32) {
    let scaled_x = (x + SCALED_XYB_OFFSET[0]) * SCALED_XYB_SCALE[0];
    let scaled_y = (y + SCALED_XYB_OFFSET[1]) * SCALED_XYB_SCALE[1];
    let scaled_b = (b - y + SCALED_XYB_OFFSET[2]) * SCALED_XYB_SCALE[2];
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

// ============================================================================
// SCALAR BUFFER CONVERSIONS
// ============================================================================

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

// ============================================================================
// SIMD CUBE ROOT
// ============================================================================

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

// ============================================================================
// SIMD XYB CORE TRANSFORM
// ============================================================================

/// Pre-computed SIMD constants for XYB conversion.
///
/// Creating this once and reusing it avoids redundant `f32x8::splat` calls
/// in inner loops, improving performance.
struct XybSimdConstants {
    // Opsin absorbance matrix coefficients
    m00: f32x8,
    m01: f32x8,
    m02: f32x8,
    m10: f32x8,
    m11: f32x8,
    m12: f32x8,
    m20: f32x8,
    m21: f32x8,
    m22: f32x8,
    // Other constants
    bias: f32x8,
    zero: f32x8,
    neg_bias_cbrt: f32x8,
    half: f32x8,
    // Scaling constants
    scale_x: f32x8,
    scale_y: f32x8,
    scale_b: f32x8,
    offset_x: f32x8,
    offset_y: f32x8,
    offset_b: f32x8,
}

impl XybSimdConstants {
    /// Create new SIMD constants from the global XYB parameters.
    fn new() -> Self {
        let m = &XYB_OPSIN_ABSORBANCE_MATRIX;
        let bias = XYB_OPSIN_ABSORBANCE_BIAS[0];
        let neg_bias_cbrt = -cbrtf_fast(bias);

        Self {
            m00: f32x8::splat(m[0]),
            m01: f32x8::splat(m[1]),
            m02: f32x8::splat(m[2]),
            m10: f32x8::splat(m[3]),
            m11: f32x8::splat(m[4]),
            m12: f32x8::splat(m[5]),
            m20: f32x8::splat(m[6]),
            m21: f32x8::splat(m[7]),
            m22: f32x8::splat(m[8]),
            bias: f32x8::splat(bias),
            zero: f32x8::splat(0.0),
            neg_bias_cbrt: f32x8::splat(neg_bias_cbrt),
            half: f32x8::splat(0.5),
            scale_x: f32x8::splat(SCALED_XYB_SCALE[0]),
            scale_y: f32x8::splat(SCALED_XYB_SCALE[1]),
            scale_b: f32x8::splat(SCALED_XYB_SCALE[2]),
            offset_x: f32x8::splat(SCALED_XYB_OFFSET[0]),
            offset_y: f32x8::splat(SCALED_XYB_OFFSET[1]),
            offset_b: f32x8::splat(SCALED_XYB_OFFSET[2]),
        }
    }

    /// Core SIMD XYB transform: linear RGB → scaled XYB.
    ///
    /// Takes 8 linear RGB values and produces 8 scaled XYB values.
    /// This is the shared core that all SIMD converters use.
    #[inline(always)]
    fn linear_rgb_to_scaled_xyb(&self, r: f32x8, g: f32x8, b_in: f32x8) -> (f32x8, f32x8, f32x8) {
        // Step 1: Opsin absorbance matrix
        let mixed0 = self
            .m00
            .mul_add(r, self.m01.mul_add(g, self.m02.mul_add(b_in, self.bias)));
        let mixed1 = self
            .m10
            .mul_add(r, self.m11.mul_add(g, self.m12.mul_add(b_in, self.bias)));
        let mixed2 = self
            .m20
            .mul_add(r, self.m21.mul_add(g, self.m22.mul_add(b_in, self.bias)));

        // Step 2: Clamp negatives
        let mixed0 = mixed0.max(self.zero);
        let mixed1 = mixed1.max(self.zero);
        let mixed2 = mixed2.max(self.zero);

        // Step 3: Cube root + bias subtraction
        let gamma0 = cbrtf_x8(mixed0) + self.neg_bias_cbrt;
        let gamma1 = cbrtf_x8(mixed1) + self.neg_bias_cbrt;
        let gamma2 = cbrtf_x8(mixed2) + self.neg_bias_cbrt;

        // Step 4: XYB transform
        let x_xyb = self.half * (gamma0 - gamma1);
        let y_xyb = self.half * (gamma0 + gamma1);
        let b_xyb = gamma2;

        // Step 5: Scale for JPEG
        // IMPORTANT: B channel formula is (b - y + offset) * scale, NOT b * scale + offset
        let scaled_x = (x_xyb + self.offset_x) * self.scale_x;
        let scaled_y = (y_xyb + self.offset_y) * self.scale_y;
        let scaled_b = (b_xyb - y_xyb + self.offset_b) * self.scale_b;

        (scaled_x, scaled_y, scaled_b)
    }

    /// Core SIMD XYB transform without scaling: linear RGB → XYB.
    #[inline(always)]
    fn linear_rgb_to_xyb(&self, r: f32x8, g: f32x8, b_in: f32x8) -> (f32x8, f32x8, f32x8) {
        let mixed0 = self
            .m00
            .mul_add(r, self.m01.mul_add(g, self.m02.mul_add(b_in, self.bias)));
        let mixed1 = self
            .m10
            .mul_add(r, self.m11.mul_add(g, self.m12.mul_add(b_in, self.bias)));
        let mixed2 = self
            .m20
            .mul_add(r, self.m21.mul_add(g, self.m22.mul_add(b_in, self.bias)));

        let mixed0 = mixed0.max(self.zero);
        let mixed1 = mixed1.max(self.zero);
        let mixed2 = mixed2.max(self.zero);

        let gamma0 = cbrtf_x8(mixed0) + self.neg_bias_cbrt;
        let gamma1 = cbrtf_x8(mixed1) + self.neg_bias_cbrt;
        let gamma2 = cbrtf_x8(mixed2) + self.neg_bias_cbrt;

        let x = self.half * (gamma0 - gamma1);
        let y = self.half * (gamma0 + gamma1);
        let b = gamma2;

        (x, y, b)
    }
}

// ============================================================================
// ARCHMAGE XYB CORE TRANSFORM (AVX2+FMA with hardware cbrt)
// ============================================================================

/// AVX2+FMA XYB core transform using magetypes::simd::f32x8 for real FMA
/// and cbrt_midp() for hardware-accelerated cube root (~3 ULP precision,
/// better than C++ jpegli's ~6 ULP).
#[cfg(target_arch = "x86_64")]
#[arcane]
fn mage_linear_rgb_to_scaled_xyb(
    token: archmage::X64V3Token,
    r_arr: [f32; 8],
    g_arr: [f32; 8],
    b_arr: [f32; 8],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
    base: usize,
) {
    use magetypes::simd::f32x8 as mf32x8;

    let m = &XYB_OPSIN_ABSORBANCE_MATRIX;
    let bias_val = XYB_OPSIN_ABSORBANCE_BIAS[0];
    let neg_bias_cbrt_val = -cbrtf_fast(bias_val);

    let r = mf32x8::from_array(token, r_arr);
    let g = mf32x8::from_array(token, g_arr);
    let b_in = mf32x8::from_array(token, b_arr);

    // Opsin absorbance matrix (real FMA: vfmadd instructions)
    let m00 = mf32x8::splat(token, m[0]);
    let m01 = mf32x8::splat(token, m[1]);
    let m02 = mf32x8::splat(token, m[2]);
    let m10 = mf32x8::splat(token, m[3]);
    let m11 = mf32x8::splat(token, m[4]);
    let m12 = mf32x8::splat(token, m[5]);
    let m20 = mf32x8::splat(token, m[6]);
    let m21 = mf32x8::splat(token, m[7]);
    let m22 = mf32x8::splat(token, m[8]);
    let bias = mf32x8::splat(token, bias_val);
    let zero = mf32x8::splat(token, 0.0);
    let neg_bias_cbrt = mf32x8::splat(token, neg_bias_cbrt_val);
    let half = mf32x8::splat(token, 0.5);

    let mixed0 = m00.mul_add(r, m01.mul_add(g, m02.mul_add(b_in, bias)));
    let mixed1 = m10.mul_add(r, m11.mul_add(g, m12.mul_add(b_in, bias)));
    let mixed2 = m20.mul_add(r, m21.mul_add(g, m22.mul_add(b_in, bias)));

    let mixed0 = mixed0.max(zero);
    let mixed1 = mixed1.max(zero);
    let mixed2 = mixed2.max(zero);

    // Hardware-accelerated cube root (~3 ULP via Halley iteration)
    let gamma0 = mixed0.cbrt_midp() + neg_bias_cbrt;
    let gamma1 = mixed1.cbrt_midp() + neg_bias_cbrt;
    let gamma2 = mixed2.cbrt_midp() + neg_bias_cbrt;

    // XYB transform
    let x_xyb = half * (gamma0 - gamma1);
    let y_xyb = half * (gamma0 + gamma1);
    let b_xyb = gamma2;

    // Scale for JPEG
    let scale_x = mf32x8::splat(token, SCALED_XYB_SCALE[0]);
    let scale_y = mf32x8::splat(token, SCALED_XYB_SCALE[1]);
    let scale_b = mf32x8::splat(token, SCALED_XYB_SCALE[2]);
    let offset_x = mf32x8::splat(token, SCALED_XYB_OFFSET[0]);
    let offset_y = mf32x8::splat(token, SCALED_XYB_OFFSET[1]);
    let offset_b = mf32x8::splat(token, SCALED_XYB_OFFSET[2]);

    let sx = (x_xyb + offset_x) * scale_x;
    let sy = (y_xyb + offset_y) * scale_y;
    let sb = (b_xyb - y_xyb + offset_b) * scale_b;

    // Store to planes
    x_out[base..base + 8].copy_from_slice(&sx.to_array());
    y_out[base..base + 8].copy_from_slice(&sy.to_array());
    b_out[base..base + 8].copy_from_slice(&sb.to_array());
}

/// AVX2+FMA XYB core transform without JPEG scaling (for non-encoded XYB output).
#[cfg(target_arch = "x86_64")]
#[arcane]
fn mage_linear_rgb_to_xyb_inplace(token: archmage::X64V3Token, pixels: &mut [[f32; 3]]) {
    use magetypes::simd::f32x8 as mf32x8;

    let m = &XYB_OPSIN_ABSORBANCE_MATRIX;
    let bias_val = XYB_OPSIN_ABSORBANCE_BIAS[0];
    let neg_bias_cbrt_val = -cbrtf_fast(bias_val);

    let m00 = mf32x8::splat(token, m[0]);
    let m01 = mf32x8::splat(token, m[1]);
    let m02 = mf32x8::splat(token, m[2]);
    let m10 = mf32x8::splat(token, m[3]);
    let m11 = mf32x8::splat(token, m[4]);
    let m12 = mf32x8::splat(token, m[5]);
    let m20 = mf32x8::splat(token, m[6]);
    let m21 = mf32x8::splat(token, m[7]);
    let m22 = mf32x8::splat(token, m[8]);
    let bias = mf32x8::splat(token, bias_val);
    let zero = mf32x8::splat(token, 0.0);
    let neg_bias_cbrt = mf32x8::splat(token, neg_bias_cbrt_val);
    let half = mf32x8::splat(token, 0.5);

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

        let r = mf32x8::from_array(token, r_arr);
        let g = mf32x8::from_array(token, g_arr);
        let b_in = mf32x8::from_array(token, b_arr);

        let mixed0 = m00.mul_add(r, m01.mul_add(g, m02.mul_add(b_in, bias)));
        let mixed1 = m10.mul_add(r, m11.mul_add(g, m12.mul_add(b_in, bias)));
        let mixed2 = m20.mul_add(r, m21.mul_add(g, m22.mul_add(b_in, bias)));

        let mixed0 = mixed0.max(zero);
        let mixed1 = mixed1.max(zero);
        let mixed2 = mixed2.max(zero);

        let gamma0 = mixed0.cbrt_midp() + neg_bias_cbrt;
        let gamma1 = mixed1.cbrt_midp() + neg_bias_cbrt;
        let gamma2 = mixed2.cbrt_midp() + neg_bias_cbrt;

        let x_xyb = half * (gamma0 - gamma1);
        let y_xyb = half * (gamma0 + gamma1);
        let b_xyb = gamma2;

        let x_arr = x_xyb.to_array();
        let y_arr = y_xyb.to_array();
        let b_out = b_xyb.to_array();

        for i in 0..8 {
            pixels[base + i] = [x_arr[i], y_arr[i], b_out[i]];
        }
    }
}

// ============================================================================
// SIMD PIXEL GATHERING
// ============================================================================
//
// These functions gather 8 pixels from various input formats and return
// linear RGB f32x8 values for the core transform.

/// Gather 8 RGB pixels from packed RGB u8 data into arrays (for archmage path).
#[inline(always)]
fn gather_rgb_8_arr(rgb_data: &[u8], base: usize) -> ([f32; 8], [f32; 8], [f32; 8]) {
    let r = [
        SRGB_TO_LINEAR_LUT[rgb_data[base] as usize],
        SRGB_TO_LINEAR_LUT[rgb_data[base + 3] as usize],
        SRGB_TO_LINEAR_LUT[rgb_data[base + 6] as usize],
        SRGB_TO_LINEAR_LUT[rgb_data[base + 9] as usize],
        SRGB_TO_LINEAR_LUT[rgb_data[base + 12] as usize],
        SRGB_TO_LINEAR_LUT[rgb_data[base + 15] as usize],
        SRGB_TO_LINEAR_LUT[rgb_data[base + 18] as usize],
        SRGB_TO_LINEAR_LUT[rgb_data[base + 21] as usize],
    ];
    let g = [
        SRGB_TO_LINEAR_LUT[rgb_data[base + 1] as usize],
        SRGB_TO_LINEAR_LUT[rgb_data[base + 4] as usize],
        SRGB_TO_LINEAR_LUT[rgb_data[base + 7] as usize],
        SRGB_TO_LINEAR_LUT[rgb_data[base + 10] as usize],
        SRGB_TO_LINEAR_LUT[rgb_data[base + 13] as usize],
        SRGB_TO_LINEAR_LUT[rgb_data[base + 16] as usize],
        SRGB_TO_LINEAR_LUT[rgb_data[base + 19] as usize],
        SRGB_TO_LINEAR_LUT[rgb_data[base + 22] as usize],
    ];
    let b = [
        SRGB_TO_LINEAR_LUT[rgb_data[base + 2] as usize],
        SRGB_TO_LINEAR_LUT[rgb_data[base + 5] as usize],
        SRGB_TO_LINEAR_LUT[rgb_data[base + 8] as usize],
        SRGB_TO_LINEAR_LUT[rgb_data[base + 11] as usize],
        SRGB_TO_LINEAR_LUT[rgb_data[base + 14] as usize],
        SRGB_TO_LINEAR_LUT[rgb_data[base + 17] as usize],
        SRGB_TO_LINEAR_LUT[rgb_data[base + 20] as usize],
        SRGB_TO_LINEAR_LUT[rgb_data[base + 23] as usize],
    ];
    (r, g, b)
}

/// Gather 8 RGBA pixels from packed RGBA u8 data into arrays (for archmage path, alpha ignored).
#[inline(always)]
fn gather_rgba_8_arr(rgba_data: &[u8], base: usize) -> ([f32; 8], [f32; 8], [f32; 8]) {
    let r = [
        SRGB_TO_LINEAR_LUT[rgba_data[base] as usize],
        SRGB_TO_LINEAR_LUT[rgba_data[base + 4] as usize],
        SRGB_TO_LINEAR_LUT[rgba_data[base + 8] as usize],
        SRGB_TO_LINEAR_LUT[rgba_data[base + 12] as usize],
        SRGB_TO_LINEAR_LUT[rgba_data[base + 16] as usize],
        SRGB_TO_LINEAR_LUT[rgba_data[base + 20] as usize],
        SRGB_TO_LINEAR_LUT[rgba_data[base + 24] as usize],
        SRGB_TO_LINEAR_LUT[rgba_data[base + 28] as usize],
    ];
    let g = [
        SRGB_TO_LINEAR_LUT[rgba_data[base + 1] as usize],
        SRGB_TO_LINEAR_LUT[rgba_data[base + 5] as usize],
        SRGB_TO_LINEAR_LUT[rgba_data[base + 9] as usize],
        SRGB_TO_LINEAR_LUT[rgba_data[base + 13] as usize],
        SRGB_TO_LINEAR_LUT[rgba_data[base + 17] as usize],
        SRGB_TO_LINEAR_LUT[rgba_data[base + 21] as usize],
        SRGB_TO_LINEAR_LUT[rgba_data[base + 25] as usize],
        SRGB_TO_LINEAR_LUT[rgba_data[base + 29] as usize],
    ];
    let b = [
        SRGB_TO_LINEAR_LUT[rgba_data[base + 2] as usize],
        SRGB_TO_LINEAR_LUT[rgba_data[base + 6] as usize],
        SRGB_TO_LINEAR_LUT[rgba_data[base + 10] as usize],
        SRGB_TO_LINEAR_LUT[rgba_data[base + 14] as usize],
        SRGB_TO_LINEAR_LUT[rgba_data[base + 18] as usize],
        SRGB_TO_LINEAR_LUT[rgba_data[base + 22] as usize],
        SRGB_TO_LINEAR_LUT[rgba_data[base + 26] as usize],
        SRGB_TO_LINEAR_LUT[rgba_data[base + 30] as usize],
    ];
    (r, g, b)
}

/// Gather 8 BGRA pixels from packed BGRA u8 data into arrays (for archmage path, alpha ignored).
#[inline(always)]
fn gather_bgra_8_arr(bgra_data: &[u8], base: usize) -> ([f32; 8], [f32; 8], [f32; 8]) {
    let r = [
        SRGB_TO_LINEAR_LUT[bgra_data[base + 2] as usize],
        SRGB_TO_LINEAR_LUT[bgra_data[base + 6] as usize],
        SRGB_TO_LINEAR_LUT[bgra_data[base + 10] as usize],
        SRGB_TO_LINEAR_LUT[bgra_data[base + 14] as usize],
        SRGB_TO_LINEAR_LUT[bgra_data[base + 18] as usize],
        SRGB_TO_LINEAR_LUT[bgra_data[base + 22] as usize],
        SRGB_TO_LINEAR_LUT[bgra_data[base + 26] as usize],
        SRGB_TO_LINEAR_LUT[bgra_data[base + 30] as usize],
    ];
    let g = [
        SRGB_TO_LINEAR_LUT[bgra_data[base + 1] as usize],
        SRGB_TO_LINEAR_LUT[bgra_data[base + 5] as usize],
        SRGB_TO_LINEAR_LUT[bgra_data[base + 9] as usize],
        SRGB_TO_LINEAR_LUT[bgra_data[base + 13] as usize],
        SRGB_TO_LINEAR_LUT[bgra_data[base + 17] as usize],
        SRGB_TO_LINEAR_LUT[bgra_data[base + 21] as usize],
        SRGB_TO_LINEAR_LUT[bgra_data[base + 25] as usize],
        SRGB_TO_LINEAR_LUT[bgra_data[base + 29] as usize],
    ];
    let b = [
        SRGB_TO_LINEAR_LUT[bgra_data[base] as usize],
        SRGB_TO_LINEAR_LUT[bgra_data[base + 4] as usize],
        SRGB_TO_LINEAR_LUT[bgra_data[base + 8] as usize],
        SRGB_TO_LINEAR_LUT[bgra_data[base + 12] as usize],
        SRGB_TO_LINEAR_LUT[bgra_data[base + 16] as usize],
        SRGB_TO_LINEAR_LUT[bgra_data[base + 20] as usize],
        SRGB_TO_LINEAR_LUT[bgra_data[base + 24] as usize],
        SRGB_TO_LINEAR_LUT[bgra_data[base + 28] as usize],
    ];
    (r, g, b)
}

/// Gather 8 RGB pixels from packed RGB u8 data.
#[inline(always)]
fn gather_rgb_8(rgb_data: &[u8], base: usize) -> (f32x8, f32x8, f32x8) {
    let r = f32x8::from([
        SRGB_TO_LINEAR_LUT[rgb_data[base] as usize],
        SRGB_TO_LINEAR_LUT[rgb_data[base + 3] as usize],
        SRGB_TO_LINEAR_LUT[rgb_data[base + 6] as usize],
        SRGB_TO_LINEAR_LUT[rgb_data[base + 9] as usize],
        SRGB_TO_LINEAR_LUT[rgb_data[base + 12] as usize],
        SRGB_TO_LINEAR_LUT[rgb_data[base + 15] as usize],
        SRGB_TO_LINEAR_LUT[rgb_data[base + 18] as usize],
        SRGB_TO_LINEAR_LUT[rgb_data[base + 21] as usize],
    ]);
    let g = f32x8::from([
        SRGB_TO_LINEAR_LUT[rgb_data[base + 1] as usize],
        SRGB_TO_LINEAR_LUT[rgb_data[base + 4] as usize],
        SRGB_TO_LINEAR_LUT[rgb_data[base + 7] as usize],
        SRGB_TO_LINEAR_LUT[rgb_data[base + 10] as usize],
        SRGB_TO_LINEAR_LUT[rgb_data[base + 13] as usize],
        SRGB_TO_LINEAR_LUT[rgb_data[base + 16] as usize],
        SRGB_TO_LINEAR_LUT[rgb_data[base + 19] as usize],
        SRGB_TO_LINEAR_LUT[rgb_data[base + 22] as usize],
    ]);
    let b = f32x8::from([
        SRGB_TO_LINEAR_LUT[rgb_data[base + 2] as usize],
        SRGB_TO_LINEAR_LUT[rgb_data[base + 5] as usize],
        SRGB_TO_LINEAR_LUT[rgb_data[base + 8] as usize],
        SRGB_TO_LINEAR_LUT[rgb_data[base + 11] as usize],
        SRGB_TO_LINEAR_LUT[rgb_data[base + 14] as usize],
        SRGB_TO_LINEAR_LUT[rgb_data[base + 17] as usize],
        SRGB_TO_LINEAR_LUT[rgb_data[base + 20] as usize],
        SRGB_TO_LINEAR_LUT[rgb_data[base + 23] as usize],
    ]);
    (r, g, b)
}

/// Gather 8 RGBA pixels from packed RGBA u8 data (alpha ignored).
#[inline(always)]
fn gather_rgba_8(rgba_data: &[u8], base: usize) -> (f32x8, f32x8, f32x8) {
    let r = f32x8::from([
        SRGB_TO_LINEAR_LUT[rgba_data[base] as usize],
        SRGB_TO_LINEAR_LUT[rgba_data[base + 4] as usize],
        SRGB_TO_LINEAR_LUT[rgba_data[base + 8] as usize],
        SRGB_TO_LINEAR_LUT[rgba_data[base + 12] as usize],
        SRGB_TO_LINEAR_LUT[rgba_data[base + 16] as usize],
        SRGB_TO_LINEAR_LUT[rgba_data[base + 20] as usize],
        SRGB_TO_LINEAR_LUT[rgba_data[base + 24] as usize],
        SRGB_TO_LINEAR_LUT[rgba_data[base + 28] as usize],
    ]);
    let g = f32x8::from([
        SRGB_TO_LINEAR_LUT[rgba_data[base + 1] as usize],
        SRGB_TO_LINEAR_LUT[rgba_data[base + 5] as usize],
        SRGB_TO_LINEAR_LUT[rgba_data[base + 9] as usize],
        SRGB_TO_LINEAR_LUT[rgba_data[base + 13] as usize],
        SRGB_TO_LINEAR_LUT[rgba_data[base + 17] as usize],
        SRGB_TO_LINEAR_LUT[rgba_data[base + 21] as usize],
        SRGB_TO_LINEAR_LUT[rgba_data[base + 25] as usize],
        SRGB_TO_LINEAR_LUT[rgba_data[base + 29] as usize],
    ]);
    let b = f32x8::from([
        SRGB_TO_LINEAR_LUT[rgba_data[base + 2] as usize],
        SRGB_TO_LINEAR_LUT[rgba_data[base + 6] as usize],
        SRGB_TO_LINEAR_LUT[rgba_data[base + 10] as usize],
        SRGB_TO_LINEAR_LUT[rgba_data[base + 14] as usize],
        SRGB_TO_LINEAR_LUT[rgba_data[base + 18] as usize],
        SRGB_TO_LINEAR_LUT[rgba_data[base + 22] as usize],
        SRGB_TO_LINEAR_LUT[rgba_data[base + 26] as usize],
        SRGB_TO_LINEAR_LUT[rgba_data[base + 30] as usize],
    ]);
    (r, g, b)
}

/// Gather 8 BGRA pixels from packed BGRA u8 data (alpha ignored).
#[inline(always)]
fn gather_bgra_8(bgra_data: &[u8], base: usize) -> (f32x8, f32x8, f32x8) {
    // BGRA layout: B=0, G=1, R=2, A=3
    let r = f32x8::from([
        SRGB_TO_LINEAR_LUT[bgra_data[base + 2] as usize],
        SRGB_TO_LINEAR_LUT[bgra_data[base + 6] as usize],
        SRGB_TO_LINEAR_LUT[bgra_data[base + 10] as usize],
        SRGB_TO_LINEAR_LUT[bgra_data[base + 14] as usize],
        SRGB_TO_LINEAR_LUT[bgra_data[base + 18] as usize],
        SRGB_TO_LINEAR_LUT[bgra_data[base + 22] as usize],
        SRGB_TO_LINEAR_LUT[bgra_data[base + 26] as usize],
        SRGB_TO_LINEAR_LUT[bgra_data[base + 30] as usize],
    ]);
    let g = f32x8::from([
        SRGB_TO_LINEAR_LUT[bgra_data[base + 1] as usize],
        SRGB_TO_LINEAR_LUT[bgra_data[base + 5] as usize],
        SRGB_TO_LINEAR_LUT[bgra_data[base + 9] as usize],
        SRGB_TO_LINEAR_LUT[bgra_data[base + 13] as usize],
        SRGB_TO_LINEAR_LUT[bgra_data[base + 17] as usize],
        SRGB_TO_LINEAR_LUT[bgra_data[base + 21] as usize],
        SRGB_TO_LINEAR_LUT[bgra_data[base + 25] as usize],
        SRGB_TO_LINEAR_LUT[bgra_data[base + 29] as usize],
    ]);
    let b = f32x8::from([
        SRGB_TO_LINEAR_LUT[bgra_data[base] as usize],
        SRGB_TO_LINEAR_LUT[bgra_data[base + 4] as usize],
        SRGB_TO_LINEAR_LUT[bgra_data[base + 8] as usize],
        SRGB_TO_LINEAR_LUT[bgra_data[base + 12] as usize],
        SRGB_TO_LINEAR_LUT[bgra_data[base + 16] as usize],
        SRGB_TO_LINEAR_LUT[bgra_data[base + 20] as usize],
        SRGB_TO_LINEAR_LUT[bgra_data[base + 24] as usize],
        SRGB_TO_LINEAR_LUT[bgra_data[base + 28] as usize],
    ]);
    (r, g, b)
}

/// Scatter 8 XYB values to separate planes.
#[inline(always)]
fn scatter_xyb_8(
    x: f32x8,
    y: f32x8,
    b: f32x8,
    x_plane: &mut [f32],
    y_plane: &mut [f32],
    b_plane: &mut [f32],
    base: usize,
) {
    let x_arr: [f32; 8] = x.into();
    let y_arr: [f32; 8] = y.into();
    let b_arr: [f32; 8] = b.into();
    x_plane[base..base + 8].copy_from_slice(&x_arr);
    y_plane[base..base + 8].copy_from_slice(&y_arr);
    b_plane[base..base + 8].copy_from_slice(&b_arr);
}

// ============================================================================
// SIMD PUBLIC API: sRGB → Scaled XYB Planes
// ============================================================================

/// SIMD sRGB to scaled XYB conversion (RGB format, allocating).
pub fn srgb_to_scaled_xyb_planes_simd(
    rgb_data: &[u8],
    num_pixels: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    assert!(rgb_data.len() >= num_pixels * 3);

    let mut x_plane = vec![0.0f32; num_pixels];
    let mut y_plane = vec![0.0f32; num_pixels];
    let mut b_plane = vec![0.0f32; num_pixels];

    srgb_to_scaled_xyb_planes_simd_inplace(
        rgb_data,
        &mut x_plane,
        &mut y_plane,
        &mut b_plane,
        num_pixels,
    );

    (x_plane, y_plane, b_plane)
}

/// SIMD sRGB to scaled XYB conversion (RGB format, inplace).
///
/// Dispatches to AVX2+FMA with hardware cbrt on x86_64, wide fallback otherwise.
pub fn srgb_to_scaled_xyb_planes_simd_inplace(
    rgb_data: &[u8],
    x_plane: &mut [f32],
    y_plane: &mut [f32],
    b_plane: &mut [f32],
    num_pixels: usize,
) {
    assert!(rgb_data.len() >= num_pixels * 3);
    assert!(x_plane.len() >= num_pixels);
    assert!(y_plane.len() >= num_pixels);
    assert!(b_plane.len() >= num_pixels);

    let chunks = num_pixels / 8;

    #[cfg(target_arch = "x86_64")]
    if let Some(token) = archmage::X64V3Token::summon() {
        for chunk in 0..chunks {
            let pixel_idx = chunk * 8;
            let rgb_idx = pixel_idx * 3;

            // LUT gather is scalar (index-dependent), then pack into archmage transform
            let (r_arr, g_arr, b_arr) = gather_rgb_8_arr(rgb_data, rgb_idx);
            mage_linear_rgb_to_scaled_xyb(
                token, r_arr, g_arr, b_arr, x_plane, y_plane, b_plane, pixel_idx,
            );
        }

        for i in (chunks * 8)..num_pixels {
            let (x, y, b) =
                srgb_to_scaled_xyb(rgb_data[i * 3], rgb_data[i * 3 + 1], rgb_data[i * 3 + 2]);
            x_plane[i] = x;
            y_plane[i] = y;
            b_plane[i] = b;
        }
        return;
    }

    let consts = XybSimdConstants::new();
    for chunk in 0..chunks {
        let pixel_idx = chunk * 8;
        let rgb_idx = pixel_idx * 3;

        let (r, g, b_in) = gather_rgb_8(rgb_data, rgb_idx);
        let (sx, sy, sb) = consts.linear_rgb_to_scaled_xyb(r, g, b_in);
        scatter_xyb_8(sx, sy, sb, x_plane, y_plane, b_plane, pixel_idx);
    }

    for i in (chunks * 8)..num_pixels {
        let (x, y, b) =
            srgb_to_scaled_xyb(rgb_data[i * 3], rgb_data[i * 3 + 1], rgb_data[i * 3 + 2]);
        x_plane[i] = x;
        y_plane[i] = y;
        b_plane[i] = b;
    }
}

/// SIMD sRGB to scaled XYB conversion (RGBA format, allocating).
pub fn srgb_to_scaled_xyb_planes_simd_rgba(
    rgba_data: &[u8],
    num_pixels: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    assert!(rgba_data.len() >= num_pixels * 4);

    let mut x_plane = vec![0.0f32; num_pixels];
    let mut y_plane = vec![0.0f32; num_pixels];
    let mut b_plane = vec![0.0f32; num_pixels];

    srgb_to_scaled_xyb_planes_simd_rgba_inplace(
        rgba_data,
        &mut x_plane,
        &mut y_plane,
        &mut b_plane,
        num_pixels,
    );

    (x_plane, y_plane, b_plane)
}

/// SIMD sRGB to scaled XYB conversion (RGBA format, inplace).
///
/// Dispatches to AVX2+FMA with hardware cbrt on x86_64, wide fallback otherwise.
pub fn srgb_to_scaled_xyb_planes_simd_rgba_inplace(
    rgba_data: &[u8],
    x_plane: &mut [f32],
    y_plane: &mut [f32],
    b_plane: &mut [f32],
    num_pixels: usize,
) {
    assert!(rgba_data.len() >= num_pixels * 4);
    assert!(x_plane.len() >= num_pixels);
    assert!(y_plane.len() >= num_pixels);
    assert!(b_plane.len() >= num_pixels);

    let chunks = num_pixels / 8;

    #[cfg(target_arch = "x86_64")]
    if let Some(token) = archmage::X64V3Token::summon() {
        for chunk in 0..chunks {
            let pixel_idx = chunk * 8;
            let rgba_idx = pixel_idx * 4;

            let (r_arr, g_arr, b_arr) = gather_rgba_8_arr(rgba_data, rgba_idx);
            mage_linear_rgb_to_scaled_xyb(
                token, r_arr, g_arr, b_arr, x_plane, y_plane, b_plane, pixel_idx,
            );
        }

        for i in (chunks * 8)..num_pixels {
            let (x, y, b) =
                srgb_to_scaled_xyb(rgba_data[i * 4], rgba_data[i * 4 + 1], rgba_data[i * 4 + 2]);
            x_plane[i] = x;
            y_plane[i] = y;
            b_plane[i] = b;
        }
        return;
    }

    let consts = XybSimdConstants::new();
    for chunk in 0..chunks {
        let pixel_idx = chunk * 8;
        let rgba_idx = pixel_idx * 4;

        let (r, g, b_in) = gather_rgba_8(rgba_data, rgba_idx);
        let (sx, sy, sb) = consts.linear_rgb_to_scaled_xyb(r, g, b_in);
        scatter_xyb_8(sx, sy, sb, x_plane, y_plane, b_plane, pixel_idx);
    }

    for i in (chunks * 8)..num_pixels {
        let (x, y, b) =
            srgb_to_scaled_xyb(rgba_data[i * 4], rgba_data[i * 4 + 1], rgba_data[i * 4 + 2]);
        x_plane[i] = x;
        y_plane[i] = y;
        b_plane[i] = b;
    }
}

/// SIMD sRGB to scaled XYB conversion (BGRA format, allocating).
pub fn srgb_to_scaled_xyb_planes_simd_bgra(
    bgra_data: &[u8],
    num_pixels: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    assert!(bgra_data.len() >= num_pixels * 4);

    let mut x_plane = vec![0.0f32; num_pixels];
    let mut y_plane = vec![0.0f32; num_pixels];
    let mut b_plane = vec![0.0f32; num_pixels];

    srgb_to_scaled_xyb_planes_simd_bgra_inplace(
        bgra_data,
        &mut x_plane,
        &mut y_plane,
        &mut b_plane,
        num_pixels,
    );

    (x_plane, y_plane, b_plane)
}

/// SIMD sRGB to scaled XYB conversion (BGRA format, inplace).
///
/// Dispatches to AVX2+FMA with hardware cbrt on x86_64, wide fallback otherwise.
pub fn srgb_to_scaled_xyb_planes_simd_bgra_inplace(
    bgra_data: &[u8],
    x_plane: &mut [f32],
    y_plane: &mut [f32],
    b_plane: &mut [f32],
    num_pixels: usize,
) {
    assert!(bgra_data.len() >= num_pixels * 4);
    assert!(x_plane.len() >= num_pixels);
    assert!(y_plane.len() >= num_pixels);
    assert!(b_plane.len() >= num_pixels);

    let chunks = num_pixels / 8;

    #[cfg(target_arch = "x86_64")]
    if let Some(token) = archmage::X64V3Token::summon() {
        for chunk in 0..chunks {
            let pixel_idx = chunk * 8;
            let bgra_idx = pixel_idx * 4;

            let (r_arr, g_arr, b_arr) = gather_bgra_8_arr(bgra_data, bgra_idx);
            mage_linear_rgb_to_scaled_xyb(
                token, r_arr, g_arr, b_arr, x_plane, y_plane, b_plane, pixel_idx,
            );
        }

        for i in (chunks * 8)..num_pixels {
            let (x, y, b) =
                srgb_to_scaled_xyb(bgra_data[i * 4 + 2], bgra_data[i * 4 + 1], bgra_data[i * 4]);
            x_plane[i] = x;
            y_plane[i] = y;
            b_plane[i] = b;
        }
        return;
    }

    let consts = XybSimdConstants::new();
    for chunk in 0..chunks {
        let pixel_idx = chunk * 8;
        let bgra_idx = pixel_idx * 4;

        let (r, g, b_in) = gather_bgra_8(bgra_data, bgra_idx);
        let (sx, sy, sb) = consts.linear_rgb_to_scaled_xyb(r, g, b_in);
        scatter_xyb_8(sx, sy, sb, x_plane, y_plane, b_plane, pixel_idx);
    }

    for i in (chunks * 8)..num_pixels {
        let (x, y, b) =
            srgb_to_scaled_xyb(bgra_data[i * 4 + 2], bgra_data[i * 4 + 1], bgra_data[i * 4]);
        x_plane[i] = x;
        y_plane[i] = y;
        b_plane[i] = b;
    }
}

/// SIMD sRGB to scaled XYB conversion (BGR format, inplace).
pub fn srgb_to_scaled_xyb_planes_simd_bgr_inplace(
    bgr_data: &[u8],
    x_plane: &mut [f32],
    y_plane: &mut [f32],
    b_plane: &mut [f32],
    num_pixels: usize,
) {
    // Scalar fallback - could add SIMD BGR gather if needed
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

// ============================================================================
// SIMD PUBLIC API: Linear RGB → XYB (in-place)
// ============================================================================

/// SIMD linear RGB (0-1) to XYB conversion for a batch of pixels (in-place).
///
/// Dispatches to AVX2+FMA with hardware cbrt on x86_64, wide fallback otherwise.
pub fn linear_rgb_to_xyb_simd(pixels: &mut [[f32; 3]]) {
    #[cfg(target_arch = "x86_64")]
    if let Some(token) = archmage::X64V3Token::summon() {
        mage_linear_rgb_to_xyb_inplace(token, pixels);

        // Scalar fallback for remainder (archmage function handles chunks internally)
        let scalar_start = (pixels.len() / 8) * 8;
        for pix in &mut pixels[scalar_start..] {
            let (x, y, b) = linear_rgb_to_xyb(pix[0], pix[1], pix[2]);
            *pix = [x, y, b];
        }
        return;
    }

    let consts = XybSimdConstants::new();
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

        let (x, y, b_out) = consts.linear_rgb_to_xyb(r, g, b);

        let x_arr: [f32; 8] = x.into();
        let y_arr: [f32; 8] = y.into();
        let b_arr: [f32; 8] = b_out.into();

        for i in 0..8 {
            pixels[base + i] = [x_arr[i], y_arr[i], b_arr[i]];
        }
    }

    let scalar_start = chunks_8 * 8;
    for pix in &mut pixels[scalar_start..] {
        let (x, y, b) = linear_rgb_to_xyb(pix[0], pix[1], pix[2]);
        *pix = [x, y, b];
    }
}

/// SIMD linear RGB (0-255) to XYB conversion for C++ jpegli compatibility (in-place).
///
/// Dispatches to AVX2+FMA with hardware cbrt on x86_64, wide fallback otherwise.
pub fn linear_rgb_to_xyb_simd_255(pixels: &mut [[f32; 3]]) {
    // The archmage path handles 0-255 range identically (just different input values)
    #[cfg(target_arch = "x86_64")]
    if let Some(token) = archmage::X64V3Token::summon() {
        mage_linear_rgb_to_xyb_inplace(token, pixels);

        let scalar_start = (pixels.len() / 8) * 8;
        for pix in &mut pixels[scalar_start..] {
            let (x, y, b) = linear_rgb_to_xyb_255(pix[0], pix[1], pix[2]);
            *pix = [x, y, b];
        }
        return;
    }

    let consts = XybSimdConstants::new();
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

        let (x, y, b_out) = consts.linear_rgb_to_xyb(r, g, b);

        let x_arr: [f32; 8] = x.into();
        let y_arr: [f32; 8] = y.into();
        let b_arr: [f32; 8] = b_out.into();

        for i in 0..8 {
            pixels[base + i] = [x_arr[i], y_arr[i], b_arr[i]];
        }
    }

    let scalar_start = chunks_8 * 8;
    for pix in &mut pixels[scalar_start..] {
        let (x, y, b) = linear_rgb_to_xyb_255(pix[0], pix[1], pix[2]);
        *pix = [x, y, b];
    }
}

/// SIMD sRGB u8 to XYB conversion for a batch of pixels.
pub fn srgb_to_xyb_batch(input: &[[u8; 3]], output: &mut [[f32; 3]]) {
    assert_eq!(input.len(), output.len());

    // Convert to linear RGB first
    for (inp, out) in input.iter().zip(output.iter_mut()) {
        out[0] = srgb_u8_to_linear(inp[0]);
        out[1] = srgb_u8_to_linear(inp[1]);
        out[2] = srgb_u8_to_linear(inp[2]);
    }

    // Apply SIMD XYB conversion
    linear_rgb_to_xyb_simd(output);
}

// ============================================================================
// SIMD XYB DECODE HELPERS
// ============================================================================

/// SIMD XYB plane level shift to interleaved RGB u8.
///
/// Converts 3 XYB f32 planes to interleaved RGB u8, applying:
/// - Level shift (+128)
/// - Clamp to [0, 255]
/// - Convert to u8
#[inline]
pub fn xyb_planes_to_rgb_u8_simd(plane0: &[f32], plane1: &[f32], plane2: &[f32], rgb: &mut [u8]) {
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

        let r = (p0 + offset).max(zero).min(max_val);
        let g = (p1 + offset).max(zero).min(max_val);
        let b = (p2 + offset).max(zero).min(max_val);

        let r_arr: [f32; 8] = r.into();
        let g_arr: [f32; 8] = g.into();
        let b_arr: [f32; 8] = b.into();

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

/// SIMD XYB plane level shift to interleaved RGB f32 (normalized 0-1).
#[inline]
pub fn xyb_planes_to_rgb_f32_simd(plane0: &[f32], plane1: &[f32], plane2: &[f32], rgb: &mut [f32]) {
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

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------------
    // Gamma Conversion Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_srgb_linear_roundtrip() {
        for v in 0..=255u8 {
            let linear = srgb_u8_to_linear(v);
            let back = linear_to_srgb_u8(linear);
            assert!((v as i16 - back as i16).abs() <= 1, "Failed for {}", v);
        }
    }

    #[test]
    fn test_srgb_linear_edge_cases() {
        assert_eq!(srgb_to_linear(0.0), 0.0);
        assert!((srgb_to_linear(1.0) - 1.0).abs() < 1e-6);

        let below = srgb_to_linear(0.04);
        let above = srgb_to_linear(0.05);
        assert!(below < above);

        assert_eq!(linear_to_srgb(0.0), 0.0);
        assert!((linear_to_srgb(1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_lut_vs_exact() {
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
        assert!(
            max_error < 0.005,
            "LUT error too large: {} at index {}",
            max_error,
            worst_index
        );
    }

    #[test]
    fn test_fastpow_accuracy() {
        let test_values = [0.01, 0.1, 0.2, 0.5, 0.8, 0.9, 0.99, 1.0];

        for &v in &test_values {
            let fast = srgb_to_linear_fast(v);
            let error = (fast - srgb_to_linear(v)).abs();
            assert!(
                error < 0.002,
                "srgb_to_linear_fast error for {}: {}",
                v,
                error
            );
        }

        for &v in &test_values {
            let exact = linear_to_srgb(v);
            let fast = linear_to_srgb_fast(v);
            let error = (fast - exact).abs();
            assert!(
                error < 0.002,
                "linear_to_srgb_fast error for {}: {}",
                v,
                error
            );
        }
    }

    #[test]
    fn test_fast_roundtrip() {
        for v in 0..=255u8 {
            let linear = srgb_u8_to_linear(v);
            let back = linear_to_srgb_u8_fast(linear);
            assert!(
                (v as i16 - back as i16).abs() <= 1,
                "Fast roundtrip failed for {}: got {}",
                v,
                back
            );
        }
    }

    // ------------------------------------------------------------------------
    // Scalar XYB Tests
    // ------------------------------------------------------------------------

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

            assert!(
                (r as i16 - r2 as i16).abs() <= 2,
                "R mismatch for ({},{},{})",
                r,
                g,
                b
            );
            assert!(
                (g as i16 - g2 as i16).abs() <= 2,
                "G mismatch for ({},{},{})",
                r,
                g,
                b
            );
            assert!(
                (b as i16 - b2 as i16).abs() <= 2,
                "B mismatch for ({},{},{})",
                r,
                g,
                b
            );
        }
    }

    #[test]
    fn test_gray_xyb() {
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
    fn test_cbrtf_fast_cube() {
        let test_values = [0.0f32, 0.001, 0.5, 1.0, 2.0, 10.0, 100.0];

        for v in test_values {
            let cbrt = cbrtf_fast(v);
            let back = cbrt * cbrt * cbrt;
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
    fn test_cbrtf_fast_zero_not_nan() {
        // Halley iterations for cube root can produce NaN for x=0 when
        // t*numerator underflows below f32 min subnormal, yielding 0/0
        // in the second iteration. This guards against that regression.
        let result = cbrtf_fast(0.0);
        assert!(
            result.is_finite(),
            "cbrtf_fast(0.0) = {result} (expected finite)"
        );
        assert_eq!(result, 0.0, "cbrtf_fast(0.0) must be exactly 0.0");
    }

    #[test]
    fn test_xyb_extreme_colors() {
        let extreme_colors = [
            (0u8, 0u8, 0u8),
            (255u8, 255u8, 255u8),
            (255u8, 0u8, 0u8),
            (0u8, 255u8, 0u8),
            (0u8, 0u8, 255u8),
            (255u8, 255u8, 0u8),
            (255u8, 0u8, 255u8),
            (0u8, 255u8, 255u8),
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

    // ------------------------------------------------------------------------
    // SIMD Parity Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_simd_vs_scalar_parity() {
        let test_colors: Vec<[f32; 3]> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.5],
            [0.1, 0.2, 0.3],
            [0.9, 0.8, 0.7],
            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.75],
        ];

        let scalar_results: Vec<[f32; 3]> = test_colors
            .iter()
            .map(|c| {
                let (x, y, b) = linear_rgb_to_xyb(c[0], c[1], c[2]);
                [x, y, b]
            })
            .collect();

        let mut simd_input = test_colors.clone();
        linear_rgb_to_xyb_simd(&mut simd_input);

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
        assert!(max_err < 1e-6, "Max error {} exceeds threshold", max_err);
    }

    #[test]
    fn test_simd_batch_conversion() {
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

        for (i, inp) in input.iter().enumerate() {
            let (x, y, b) = srgb_to_xyb(inp[0], inp[1], inp[2]);
            let err = (x - output[i][0])
                .abs()
                .max((y - output[i][1]).abs())
                .max((b - output[i][2]).abs());
            assert!(err < 1e-6, "Batch vs scalar mismatch at {}", i);
        }
    }

    #[test]
    fn test_simd_remainder_handling() {
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

    #[test]
    fn test_rgba_bgra_simd_parity() {
        let rgb_data: Vec<u8> = (0..64 * 3).map(|i| (i % 256) as u8).collect();
        let num_pixels = 64;

        let mut rgba_data = Vec::with_capacity(num_pixels * 4);
        let mut bgra_data = Vec::with_capacity(num_pixels * 4);
        for i in 0..num_pixels {
            let r = rgb_data[i * 3];
            let g = rgb_data[i * 3 + 1];
            let b = rgb_data[i * 3 + 2];
            rgba_data.extend_from_slice(&[r, g, b, 255]);
            bgra_data.extend_from_slice(&[b, g, r, 255]);
        }

        let (ref_x, ref_y, ref_b) = srgb_to_scaled_xyb_planes_simd(&rgb_data, num_pixels);
        let (rgba_x, rgba_y, rgba_b) = srgb_to_scaled_xyb_planes_simd_rgba(&rgba_data, num_pixels);
        let (bgra_x, bgra_y, bgra_b) = srgb_to_scaled_xyb_planes_simd_bgra(&bgra_data, num_pixels);

        for i in 0..num_pixels {
            assert!(
                (ref_x[i] - rgba_x[i]).abs() < 1e-6,
                "RGBA X mismatch at {}",
                i
            );
            assert!(
                (ref_y[i] - rgba_y[i]).abs() < 1e-6,
                "RGBA Y mismatch at {}",
                i
            );
            assert!(
                (ref_b[i] - rgba_b[i]).abs() < 1e-6,
                "RGBA B mismatch at {}",
                i
            );
            assert!(
                (ref_x[i] - bgra_x[i]).abs() < 1e-6,
                "BGRA X mismatch at {}",
                i
            );
            assert!(
                (ref_y[i] - bgra_y[i]).abs() < 1e-6,
                "BGRA Y mismatch at {}",
                i
            );
            assert!(
                (ref_b[i] - bgra_b[i]).abs() < 1e-6,
                "BGRA B mismatch at {}",
                i
            );
        }
    }

    // ------------------------------------------------------------------------
    // B Channel Scaling Tests (regression tests for the scaling formula bug)
    // ------------------------------------------------------------------------

    #[test]
    fn test_b_channel_scaling_formula() {
        // Test that B channel scaling uses: (b - y + offset) * scale
        // NOT: b * scale + offset (which was a bug)
        let test_cases = [
            (0.0f32, 0.5f32, 0.3f32),
            (0.0, 0.3, 0.5),
            (0.0, 0.5, 0.5),
            (0.0, 0.8, 0.2),
            (0.0, 0.1, 0.9),
        ];

        for (x, y, b) in test_cases {
            let (scaled_x, scaled_y, scaled_b) = scale_xyb(x, y, b);

            let expected_b = (b - y + SCALED_XYB_OFFSET[2]) * SCALED_XYB_SCALE[2];
            let expected_x = (x + SCALED_XYB_OFFSET[0]) * SCALED_XYB_SCALE[0];
            let expected_y = (y + SCALED_XYB_OFFSET[1]) * SCALED_XYB_SCALE[1];

            assert!((scaled_x - expected_x).abs() < 1e-6, "X mismatch");
            assert!((scaled_y - expected_y).abs() < 1e-6, "Y mismatch");
            assert!((scaled_b - expected_b).abs() < 1e-6, "B mismatch");
        }
    }

    #[test]
    fn test_b_channel_simd_inplace_vs_scalar() {
        let rgb_data: Vec<u8> = vec![
            255, 0, 0, 0, 255, 0, 0, 0, 255, 128, 128, 128, 255, 255, 255, 0, 0, 0, 200, 100, 50,
            50, 100, 200, 255, 128, 0, 0, 128, 255, 64, 64, 64, 192, 192, 192, 100, 200, 100, 200,
            100, 200, 50, 150, 250, 250, 150, 50,
        ];
        let num_pixels = 16;

        // Get reference from scalar
        let mut ref_x = vec![0.0f32; num_pixels];
        let mut ref_y = vec![0.0f32; num_pixels];
        let mut ref_b = vec![0.0f32; num_pixels];
        for i in 0..num_pixels {
            let (x, y, b) =
                srgb_to_scaled_xyb(rgb_data[i * 3], rgb_data[i * 3 + 1], rgb_data[i * 3 + 2]);
            ref_x[i] = x;
            ref_y[i] = y;
            ref_b[i] = b;
        }

        // Test SIMD inplace
        let mut x_plane = vec![0.0f32; num_pixels];
        let mut y_plane = vec![0.0f32; num_pixels];
        let mut b_plane = vec![0.0f32; num_pixels];
        srgb_to_scaled_xyb_planes_simd_inplace(
            &rgb_data,
            &mut x_plane,
            &mut y_plane,
            &mut b_plane,
            num_pixels,
        );

        for i in 0..num_pixels {
            assert!((ref_x[i] - x_plane[i]).abs() < 1e-5, "X mismatch at {}", i);
            assert!((ref_y[i] - y_plane[i]).abs() < 1e-5, "Y mismatch at {}", i);
            assert!((ref_b[i] - b_plane[i]).abs() < 1e-5, "B mismatch at {}", i);
        }
    }

    #[test]
    fn test_b_channel_rgba_bgra_inplace_vs_scalar() {
        let num_pixels = 16;
        let rgb_data: Vec<u8> = (0..num_pixels * 3)
            .map(|i| ((i * 17) % 256) as u8)
            .collect();

        let mut rgba_data = Vec::with_capacity(num_pixels * 4);
        let mut bgra_data = Vec::with_capacity(num_pixels * 4);
        for i in 0..num_pixels {
            let r = rgb_data[i * 3];
            let g = rgb_data[i * 3 + 1];
            let b = rgb_data[i * 3 + 2];
            rgba_data.extend_from_slice(&[r, g, b, 255]);
            bgra_data.extend_from_slice(&[b, g, r, 255]);
        }

        // Get reference from scalar
        let mut ref_b = vec![0.0f32; num_pixels];
        for i in 0..num_pixels {
            let (_, _, b) =
                srgb_to_scaled_xyb(rgb_data[i * 3], rgb_data[i * 3 + 1], rgb_data[i * 3 + 2]);
            ref_b[i] = b;
        }

        // Test RGBA inplace
        let mut x_plane = vec![0.0f32; num_pixels];
        let mut y_plane = vec![0.0f32; num_pixels];
        let mut b_plane = vec![0.0f32; num_pixels];
        srgb_to_scaled_xyb_planes_simd_rgba_inplace(
            &rgba_data,
            &mut x_plane,
            &mut y_plane,
            &mut b_plane,
            num_pixels,
        );
        for i in 0..num_pixels {
            assert!(
                (ref_b[i] - b_plane[i]).abs() < 1e-5,
                "RGBA B mismatch at {}",
                i
            );
        }

        // Test BGRA inplace
        let mut x_plane = vec![0.0f32; num_pixels];
        let mut y_plane = vec![0.0f32; num_pixels];
        let mut b_plane = vec![0.0f32; num_pixels];
        srgb_to_scaled_xyb_planes_simd_bgra_inplace(
            &bgra_data,
            &mut x_plane,
            &mut y_plane,
            &mut b_plane,
            num_pixels,
        );
        for i in 0..num_pixels {
            assert!(
                (ref_b[i] - b_plane[i]).abs() < 1e-5,
                "BGRA B mismatch at {}",
                i
            );
        }
    }

    #[test]
    fn test_b_channel_blue_heavy_colors() {
        let blue_heavy_colors = [[0u8, 0, 255], [50, 50, 200], [0, 100, 255], [100, 0, 255]];

        for [r, g, b] in blue_heavy_colors {
            let (_ref_x, _ref_y, ref_b) = srgb_to_scaled_xyb(r, g, b);
            let (_, y_xyb, b_xyb) = srgb_to_xyb(r, g, b);

            let wrong_b = b_xyb * SCALED_XYB_SCALE[2] + SCALED_XYB_OFFSET[2];
            let correct_b = (b_xyb - y_xyb + SCALED_XYB_OFFSET[2]) * SCALED_XYB_SCALE[2];

            assert!(
                (wrong_b - correct_b).abs() > 0.1,
                "Test case [{},{},{}] doesn't differentiate formulas",
                r,
                g,
                b
            );
            assert!(
                (ref_b - correct_b).abs() < 1e-5,
                "Scalar B mismatch for [{},{},{}]",
                r,
                g,
                b
            );
        }
    }

    // ------------------------------------------------------------------------
    // Buffer Conversion Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_rgb_buffer_to_xyb_planes() {
        let rgb = vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 128, 128, 128];

        let (x_plane, y_plane, b_plane) = rgb_buffer_to_xyb_planes(&rgb, 2, 2);

        assert_eq!(x_plane.len(), 4);
        assert_eq!(y_plane.len(), 4);
        assert_eq!(b_plane.len(), 4);
        assert!(x_plane[3].abs() < 0.01); // Gray should have X near 0
    }

    #[test]
    fn test_rgb_buffer_to_scaled_xyb_planes() {
        let rgb = vec![128, 128, 128, 255, 255, 255];

        let (x_plane, y_plane, b_plane) = rgb_buffer_to_scaled_xyb_planes(&rgb, 2, 1);

        assert_eq!(x_plane.len(), 2);
        assert_eq!(y_plane.len(), 2);
        assert_eq!(b_plane.len(), 2);

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
        let x_plane = vec![0.0f32; 4];
        let y_plane = vec![0.5f32; 4];
        let b_plane = vec![0.5f32; 4];

        let rgb = xyb_planes_to_rgb_buffer(&x_plane, &y_plane, &b_plane, 2, 2);

        assert_eq!(rgb.len(), 12);
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

            assert!((r as i16 - r2 as i16).abs() <= 2, "R mismatch");
            assert!((g as i16 - g2 as i16).abs() <= 2, "G mismatch");
            assert!((b as i16 - b2 as i16).abs() <= 2, "B mismatch");
        }
    }

    #[test]
    fn test_linear_rgb_xyb_direct() {
        let (x, y, b) = linear_rgb_to_xyb(0.5, 0.5, 0.5);
        assert!(x.abs() < 0.01, "X should be ~0 for gray, got {}", x);
        assert!(y > 0.0, "Y should be positive, got {}", y);

        let (r, g, b_out) = xyb_to_linear_rgb(x, y, b);
        assert!((r - 0.5).abs() < 0.01);
        assert!((g - 0.5).abs() < 0.01);
        assert!((b_out - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_fast_vs_exact_u8_output() {
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
