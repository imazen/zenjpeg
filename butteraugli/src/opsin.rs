//! Butteraugli OpsinDynamicsImage implementation.
//!
//! This module implements the correct butteraugli color space conversion,
//! which is DIFFERENT from jpegli's XYB color space.
//!
//! Key differences from jpegli XYB:
//! 1. Different OpsinAbsorbance matrix coefficients
//! 2. Uses Gamma function (FastLog2f based), not cube root
//! 3. Includes dynamic sensitivity based on blurred image

use crate::blur::gaussian_blur;
use crate::image::Image3F;

// ============================================================================
// OpsinAbsorbance coefficients (C++ butteraugli.cc lines 1428-1439)
// ============================================================================

const MIXI0: f64 = 0.299_565_503_400_583_19;
const MIXI1: f64 = 0.633_730_878_338_259_36;
const MIXI2: f64 = 0.077_705_617_820_981_968;
const MIXI3: f64 = 1.755_748_364_328_735_3; // bias for channel 0

const MIXI4: f64 = 0.221_586_911_045_747_74;
const MIXI5: f64 = 0.693_913_880_441_161_42;
const MIXI6: f64 = 0.098_731_358_842_2;
const MIXI7: f64 = 1.755_748_364_328_735_3; // bias for channel 1

const MIXI8: f64 = 0.02;
const MIXI9: f64 = 0.02;
const MIXI10: f64 = 0.204_801_290_410_261_29;
const MIXI11: f64 = 12.226_454_707_163_354; // bias for channel 2

/// Minimum value for opsin channels (bias values)
const MIN_01: f32 = 1.755_748_364_328_735_3;
const MIN_2: f32 = 12.226_454_707_163_354;

// ============================================================================
// Gamma function (C++ butteraugli.cc lines 1404-1420)
// ============================================================================

/// Inverse of log2(e) for Gamma function
const K_INV_LOG2E: f32 = 1.0 / std::f32::consts::LOG2_E;

/// Fast approximation of log2 for Gamma function.
///
/// This is a direct port of libjxl's FastLog2f from fast_math-inl.h.
/// Uses a (2,2) rational polynomial approximation of log1p(x) / log(2)
/// with range reduction to [-1/3, 1/3].
/// L1 error ~3.9E-6.
#[inline]
pub fn fast_log2f(x: f32) -> f32 {
    // (2,2) rational polynomial coefficients from C++
    const P0: f32 = -1.8503833400518310E-06;
    const P1: f32 = 1.4287160470083755;
    const P2: f32 = 7.4245873327820566E-01;

    const Q0: f32 = 9.9032814277590719E-01;
    const Q1: f32 = 1.0096718572241148;
    const Q2: f32 = 1.7409343003366853E-01;

    let x_bits = x.to_bits() as i32;

    // Range reduction to [-1/3, 1/3] - subtract 2/3 (0x3f2aaaab in float)
    let exp_bits = x_bits.wrapping_sub(0x3f2aaaab_u32 as i32);
    // Shifted exponent = log2; also used to clear mantissa
    let exp_shifted = exp_bits >> 23;
    // Reconstruct mantissa in [2/3, 4/3] range
    let mantissa_bits = (x_bits - (exp_shifted << 23)) as u32;
    let mantissa = f32::from_bits(mantissa_bits);
    let exp_val = exp_shifted as f32;

    // Evaluate rational polynomial on (mantissa - 1.0), which is in [-1/3, 1/3]
    let m = mantissa - 1.0;

    // Horner's scheme for numerator: p[2]*x^2 + p[1]*x + p[0]
    let yp = P2 * m + P1;
    let yp = yp * m + P0;

    // Horner's scheme for denominator: q[2]*x^2 + q[1]*x + q[0]
    let yq = Q2 * m + Q1;
    let yq = yq * m + Q0;

    yp / yq + exp_val
}

/// Butteraugli Gamma function.
///
/// This is NOT a simple gamma curve - it's designed to model
/// the human visual system's dynamic range adaptation.
///
/// C++ implementation:
/// ```cpp
/// const auto kRetMul = Set(df, 19.245013259874995f * kInvLog2e);
/// const auto kRetAdd = Set(df, -23.16046239805755);
/// const auto biased = Add(v, Set(df, 9.9710635769299145));
/// const auto log = FastLog2f(df, biased);
/// return MulAdd(kRetMul, log, kRetAdd);
/// ```
#[inline]
pub fn gamma(v: f32) -> f32 {
    const K_RET_MUL: f32 = 19.245_013_259_874_995 * K_INV_LOG2E;
    const K_RET_ADD: f32 = -23.160_462_398_057_55;
    const K_BIAS: f32 = 9.971_063_576_929_914_5;

    // Clamp negative values to avoid NaN
    let v = v.max(0.0);
    let biased = v + K_BIAS;
    let log = fast_log2f(biased);
    K_RET_MUL * log + K_RET_ADD
}

// ============================================================================
// OpsinAbsorbance (C++ butteraugli.cc lines 1422-1463)
// ============================================================================

/// Applies the OpsinAbsorbance matrix to RGB values.
///
/// # Arguments
/// * `r`, `g`, `b` - Linear RGB values (scaled by intensity_target)
/// * `clamp` - If true, clamp outputs to minimum bias values
///
/// # Returns
/// Three opsin absorbance values (pre-mixed channels)
#[inline]
pub fn opsin_absorbance(r: f32, g: f32, b: f32, clamp: bool) -> (f32, f32, f32) {
    let out0 = (MIXI0 as f32) * r + (MIXI1 as f32) * g + (MIXI2 as f32) * b + (MIXI3 as f32);
    let out1 = (MIXI4 as f32) * r + (MIXI5 as f32) * g + (MIXI6 as f32) * b + (MIXI7 as f32);
    let out2 = (MIXI8 as f32) * r + (MIXI9 as f32) * g + (MIXI10 as f32) * b + (MIXI11 as f32);

    if clamp {
        (out0.max(MIN_01), out1.max(MIN_01), out2.max(MIN_2))
    } else {
        (out0, out1, out2)
    }
}

/// Converts linear RGB to butteraugli XYB using OpsinDynamicsImage.
///
/// This is the CORRECT butteraugli color conversion, which includes:
/// 1. Blur the input RGB with sigma=1.2
/// 2. Compute sensitivity = Gamma(pre_mixed) / pre_mixed
/// 3. Apply sensitivity to original RGB
/// 4. Convert to XYB: X = mixed0 - mixed1, Y = mixed0 + mixed1, B = mixed2
///
/// # Arguments
/// * `rgb` - Linear RGB image (3 planes)
/// * `intensity_target` - Nits corresponding to 1.0 input value (default 80.0)
///
/// # Returns
/// XYB image (3 planes)
pub fn opsin_dynamics_image(rgb: &Image3F, intensity_target: f32) -> Image3F {
    let width = rgb.plane(0).width();
    let height = rgb.plane(0).height();

    // Step 1: Blur RGB with sigma=1.2
    let sigma = 1.2;
    let blurred_r = gaussian_blur(rgb.plane(0), sigma);
    let blurred_g = gaussian_blur(rgb.plane(1), sigma);
    let blurred_b = gaussian_blur(rgb.plane(2), sigma);

    // Create output XYB image
    let mut xyb = Image3F::new(width, height);
    let min_val = 1e-4_f32;

    for y in 0..height {
        for x in 0..width {
            // Get RGB values scaled by intensity target
            let r = rgb.plane(0).get(x, y) * intensity_target;
            let g = rgb.plane(1).get(x, y) * intensity_target;
            let b = rgb.plane(2).get(x, y) * intensity_target;

            let blurred_r_val = blurred_r.get(x, y) * intensity_target;
            let blurred_g_val = blurred_g.get(x, y) * intensity_target;
            let blurred_b_val = blurred_b.get(x, y) * intensity_target;

            // Step 2: Calculate sensitivity based on blurred image
            let (pre0, pre1, pre2) =
                opsin_absorbance(blurred_r_val, blurred_g_val, blurred_b_val, true);
            let pre0 = pre0.max(min_val);
            let pre1 = pre1.max(min_val);
            let pre2 = pre2.max(min_val);

            let sensitivity0 = (gamma(pre0) / pre0).max(min_val);
            let sensitivity1 = (gamma(pre1) / pre1).max(min_val);
            let sensitivity2 = (gamma(pre2) / pre2).max(min_val);

            // Step 3: Apply sensitivity to original RGB
            let (cur0, cur1, cur2) = opsin_absorbance(r, g, b, false);
            let cur0 = (cur0 * sensitivity0).max(MIN_01);
            let cur1 = (cur1 * sensitivity1).max(MIN_01);
            let cur2 = (cur2 * sensitivity2).max(MIN_2);

            // Step 4: Convert to XYB
            let out_x = cur0 - cur1;
            let out_y = cur0 + cur1;
            let out_b = cur2;

            xyb.plane_mut(0).set(x, y, out_x);
            xyb.plane_mut(1).set(x, y, out_y);
            xyb.plane_mut(2).set(x, y, out_b);
        }
    }

    xyb
}

/// Converts sRGB u8 image to butteraugli XYB.
///
/// # Arguments
/// * `rgb` - sRGB image data (3 bytes per pixel, row-major)
/// * `width` - Image width
/// * `height` - Image height
/// * `intensity_target` - Nits for 1.0 value (default 80.0)
///
/// # Returns
/// XYB image (3 planes)
pub fn srgb_to_xyb_butteraugli(
    rgb: &[u8],
    width: usize,
    height: usize,
    intensity_target: f32,
) -> Image3F {
    assert_eq!(rgb.len(), width * height * 3);

    // Convert sRGB u8 to linear RGB Image3F
    let mut linear = Image3F::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let r = srgb_to_linear(rgb[idx * 3]);
            let g = srgb_to_linear(rgb[idx * 3 + 1]);
            let b = srgb_to_linear(rgb[idx * 3 + 2]);

            linear.plane_mut(0).set(x, y, r);
            linear.plane_mut(1).set(x, y, g);
            linear.plane_mut(2).set(x, y, b);
        }
    }

    // Apply OpsinDynamicsImage
    opsin_dynamics_image(&linear, intensity_target)
}

/// sRGB transfer function (gamma decoding)
#[inline]
pub fn srgb_to_linear(v: u8) -> f32 {
    let v = v as f32 / 255.0;
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// Converts linear RGB f32 interleaved data to butteraugli XYB.
///
/// This matches the C++ butteraugli API which expects linear RGB float input.
///
/// # Arguments
/// * `rgb` - Linear RGB image data (f32, 3 values per pixel, row-major, 0.0-1.0 range)
/// * `width` - Image width
/// * `height` - Image height
/// * `intensity_target` - Nits for 1.0 value (default 80.0)
///
/// # Returns
/// XYB image (3 planes)
pub fn linear_rgb_to_xyb_butteraugli(
    rgb: &[f32],
    width: usize,
    height: usize,
    intensity_target: f32,
) -> Image3F {
    assert_eq!(rgb.len(), width * height * 3);

    // Convert interleaved linear RGB to planar Image3F
    let mut linear = Image3F::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            linear.plane_mut(0).set(x, y, rgb[idx]);
            linear.plane_mut(1).set(x, y, rgb[idx + 1]);
            linear.plane_mut(2).set(x, y, rgb[idx + 2]);
        }
    }

    // Apply OpsinDynamicsImage
    opsin_dynamics_image(&linear, intensity_target)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gamma_positive() {
        // Gamma should produce reasonable values for positive inputs
        let result = gamma(1.0);
        assert!(result.is_finite());
        assert!(result > -30.0 && result < 30.0);
    }

    #[test]
    fn test_gamma_zero() {
        // Gamma at 0 should be finite (due to bias)
        let result = gamma(0.0);
        assert!(result.is_finite());
    }

    #[test]
    fn test_opsin_absorbance_bias() {
        // With zero RGB, outputs should equal biases
        let (out0, out1, out2) = opsin_absorbance(0.0, 0.0, 0.0, false);
        assert!((out0 - MIXI3 as f32).abs() < 1e-6);
        assert!((out1 - MIXI7 as f32).abs() < 1e-6);
        assert!((out2 - MIXI11 as f32).abs() < 1e-6);
    }

    #[test]
    fn test_opsin_absorbance_clamped() {
        // Clamped version should never go below minimums
        let (out0, out1, out2) = opsin_absorbance(-100.0, -100.0, -100.0, true);
        assert!(out0 >= MIN_01);
        assert!(out1 >= MIN_01);
        assert!(out2 >= MIN_2);
    }

    #[test]
    fn test_fast_log2f() {
        // Test fast_log2f approximation accuracy
        for i in 1..100 {
            let x = i as f32 * 0.1;
            let fast = fast_log2f(x);
            let exact = x.log2();
            assert!(
                (fast - exact).abs() < 0.1,
                "fast_log2f({}) = {}, expected {}",
                x,
                fast,
                exact
            );
        }
    }
}
