//! Visual masking functions for butteraugli.
//!
//! Visual masking is the phenomenon where the visibility of one feature
//! is reduced by the presence of another feature. This module implements
//! the masking computations used in butteraugli.
//!
//! Key functions:
//! - `combine_channels_for_masking`: Combines HF and UHF channels
//! - `diff_precompute`: Applies sqrt-like transform for numerical stability
//! - `fuzzy_erosion`: Finds smooth areas using weighted minimum
//! - `mask_y`: Converts mask value to AC masking factor
//! - `mask_dc_y`: Converts mask value to DC masking factor

use crate::blur::gaussian_blur;
use crate::consts::{
    COMBINE_CHANNELS_MULS, GLOBAL_SCALE, MASK_BIAS, MASK_DC_Y_MUL, MASK_DC_Y_OFFSET,
    MASK_DC_Y_SCALER, MASK_MUL, MASK_RADIUS, MASK_TO_ERROR_MUL, MASK_Y_MUL, MASK_Y_OFFSET,
    MASK_Y_SCALER,
};
use crate::image::ImageF;

/// Combines HF and UHF channels for masking computation.
///
/// Only X and Y components are involved in masking. B's influence
/// is considered less important in the high frequency area.
///
/// Matches C++ CombineChannelsForMasking (butteraugli.cc lines 1108-1132).
pub fn combine_channels_for_masking(hf: &[ImageF; 2], uhf: &[ImageF; 2], out: &mut ImageF) {
    let width = hf[0].width();
    let height = hf[0].height();

    for y in 0..height {
        let row_y_hf = hf[1].row(y);
        let row_y_uhf = uhf[1].row(y);
        let row_x_hf = hf[0].row(y);
        let row_x_uhf = uhf[0].row(y);
        let row_out = out.row_mut(y);

        for x in 0..width {
            // C++: xdiff = (row_x_uhf[x] + row_x_hf[x]) * muls[0]
            let xdiff = (row_x_uhf[x] + row_x_hf[x]) * COMBINE_CHANNELS_MULS[0];
            // C++: ydiff = row_y_uhf[x] * muls[1] + row_y_hf[x] * muls[2]
            let ydiff =
                row_y_uhf[x] * COMBINE_CHANNELS_MULS[1] + row_y_hf[x] * COMBINE_CHANNELS_MULS[2];
            // C++: row[x] = sqrt(xdiff * xdiff + ydiff * ydiff)
            row_out[x] = (xdiff * xdiff + ydiff * ydiff).sqrt();
        }
    }
}

/// Precomputes difference values for masking.
///
/// Applies sqrt-like transformation to make values more perceptually uniform.
/// Matches C++ DiffPrecompute (butteraugli.cc lines 1134-1147).
pub fn diff_precompute(xyb: &ImageF, mul: f32, bias_arg: f32, out: &mut ImageF) {
    let width = xyb.width();
    let height = xyb.height();
    let bias = mul * bias_arg;
    let sqrt_bias = bias.sqrt();

    for y in 0..height {
        let row_in = xyb.row(y);
        let row_out = out.row_mut(y);
        for x in 0..width {
            // C++: sqrt(mul * abs(row_in[x]) + bias) - sqrt_bias
            row_out[x] = (mul * row_in[x].abs() + bias).sqrt() - sqrt_bias;
        }
    }
}

/// Stores the three smallest values encountered.
/// Matches C++ StoreMin3 (butteraugli.cc lines 1155-1168).
#[inline]
fn store_min3(v: f32, min0: &mut f32, min1: &mut f32, min2: &mut f32) {
    if v < *min2 {
        if v < *min0 {
            *min2 = *min1;
            *min1 = *min0;
            *min0 = v;
        } else if v < *min1 {
            *min2 = *min1;
            *min1 = v;
        } else {
            *min2 = v;
        }
    }
}

/// Performs fuzzy erosion to find smooth areas.
///
/// Look for smooth areas near the area of degradation.
/// If the areas are generally smooth, don't apply masking.
///
/// Matches C++ FuzzyErosion (butteraugli.cc lines 1170-1208).
pub fn fuzzy_erosion(from: &ImageF, to: &mut ImageF) {
    let width = from.width();
    let height = from.height();
    const K_STEP: usize = 3;

    for y in 0..height {
        for x in 0..width {
            let mut min0 = from.get(x, y);
            let mut min1 = 2.0 * min0;
            let mut min2 = min1;

            // Check neighbors at distance K_STEP (C++ exact order)
            if x >= K_STEP {
                store_min3(from.get(x - K_STEP, y), &mut min0, &mut min1, &mut min2);
                if y >= K_STEP {
                    store_min3(
                        from.get(x - K_STEP, y - K_STEP),
                        &mut min0,
                        &mut min1,
                        &mut min2,
                    );
                }
                if y + K_STEP < height {
                    store_min3(
                        from.get(x - K_STEP, y + K_STEP),
                        &mut min0,
                        &mut min1,
                        &mut min2,
                    );
                }
            }
            if x + K_STEP < width {
                store_min3(from.get(x + K_STEP, y), &mut min0, &mut min1, &mut min2);
                if y >= K_STEP {
                    store_min3(
                        from.get(x + K_STEP, y - K_STEP),
                        &mut min0,
                        &mut min1,
                        &mut min2,
                    );
                }
                if y + K_STEP < height {
                    store_min3(
                        from.get(x + K_STEP, y + K_STEP),
                        &mut min0,
                        &mut min1,
                        &mut min2,
                    );
                }
            }
            if y >= K_STEP {
                store_min3(from.get(x, y - K_STEP), &mut min0, &mut min1, &mut min2);
            }
            if y + K_STEP < height {
                store_min3(from.get(x, y + K_STEP), &mut min0, &mut min1, &mut min2);
            }

            // C++: 0.45f * min0 + 0.3f * min1 + 0.25f * min2
            to.set(x, y, 0.45 * min0 + 0.3 * min1 + 0.25 * min2);
        }
    }
}

/// Converts mask value to AC masking multiplier.
///
/// Matches C++ MaskY (butteraugli.cc lines 1266-1273).
#[inline]
pub fn mask_y(delta: f64) -> f64 {
    let c = MASK_Y_MUL / (MASK_Y_SCALER * delta + MASK_Y_OFFSET);
    let retval = GLOBAL_SCALE as f64 * (1.0 + c);
    retval * retval
}

/// Converts mask value to DC masking multiplier.
///
/// Matches C++ MaskDcY (butteraugli.cc lines 1275-1282).
#[inline]
pub fn mask_dc_y(delta: f64) -> f64 {
    let c = MASK_DC_Y_MUL / (MASK_DC_Y_SCALER * delta + MASK_DC_Y_OFFSET);
    let retval = GLOBAL_SCALE as f64 * (1.0 + c);
    retval * retval
}

/// Computes mask from both images' psychovisual representations.
///
/// Matches C++ Mask function (butteraugli.cc lines 1212-1247).
///
/// # Arguments
/// * `mask0` - Combined HF/UHF mask from image 0
/// * `mask1` - Combined HF/UHF mask from image 1
/// * `diff_ac` - Optional AC difference accumulator
///
/// # Returns
/// The computed mask image
pub fn compute_mask(mask0: &ImageF, mask1: &ImageF, mut diff_ac: Option<&mut ImageF>) -> ImageF {
    let width = mask0.width();
    let height = mask0.height();

    // DiffPrecompute for mask0
    let mut diff0 = ImageF::new(width, height);
    diff_precompute(mask0, MASK_MUL, MASK_BIAS, &mut diff0);

    // DiffPrecompute for mask1
    let mut diff1 = ImageF::new(width, height);
    diff_precompute(mask1, MASK_MUL, MASK_BIAS, &mut diff1);

    // Blur diff0 and diff1
    let blurred0 = gaussian_blur(&diff0, MASK_RADIUS);
    let blurred1 = gaussian_blur(&diff1, MASK_RADIUS);

    // FuzzyErosion on blurred0
    let mut eroded0 = ImageF::new(width, height);
    fuzzy_erosion(&blurred0, &mut eroded0);

    // Final mask computation
    let mut mask = ImageF::new(width, height);
    for y in 0..height {
        for x in 0..width {
            mask.set(x, y, eroded0.get(x, y));

            if let Some(ref mut ac) = diff_ac {
                // C++: kMaskToErrorMul * diff * diff
                let diff = blurred0.get(x, y) - blurred1.get(x, y);
                let prev = ac.get(x, y);
                ac.set(x, y, prev + MASK_TO_ERROR_MUL * diff * diff);
            }
        }
    }

    mask
}

/// Applies visual masking based on local contrast.
///
/// Higher local contrast means differences are less visible (masked).
pub fn apply_masking(diff: &ImageF, mask: &ImageF, out: &mut ImageF) {
    let width = diff.width();
    let height = diff.height();

    for y in 0..height {
        let row_diff = diff.row(y);
        let row_mask = mask.row(y);
        let row_out = out.row_mut(y);

        for x in 0..width {
            // Higher mask value means lower sensitivity
            let sensitivity = 1.0 / (1.0 + row_mask[x]);
            row_out[x] = row_diff[x] * sensitivity;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_store_min3() {
        let mut min0 = 10.0f32;
        let mut min1 = 20.0f32;
        let mut min2 = 30.0f32;

        store_min3(5.0, &mut min0, &mut min1, &mut min2);
        assert!((min0 - 5.0).abs() < 0.001);
        assert!((min1 - 10.0).abs() < 0.001);
        assert!((min2 - 20.0).abs() < 0.001);

        store_min3(15.0, &mut min0, &mut min1, &mut min2);
        assert!((min0 - 5.0).abs() < 0.001);
        assert!((min1 - 10.0).abs() < 0.001);
        assert!((min2 - 15.0).abs() < 0.001);
    }

    #[test]
    fn test_fuzzy_erosion() {
        let mut from = ImageF::new(16, 16);
        // Create a pattern with one bright spot
        from.set(8, 8, 10.0);
        from.set(5, 5, 5.0);

        let mut to = ImageF::new(16, 16);
        fuzzy_erosion(&from, &mut to);

        // The output should be smoother
        // The bright spot should be somewhat reduced
        assert!(to.get(8, 8) <= 10.0);
    }

    #[test]
    fn test_diff_precompute() {
        let input = ImageF::filled(16, 16, 1.0);
        let mut output = ImageF::new(16, 16);

        diff_precompute(&input, 1.0, 0.01, &mut output);

        // All values should be positive
        for y in 0..16 {
            for x in 0..16 {
                assert!(output.get(x, y) >= 0.0);
            }
        }
    }

    #[test]
    fn test_mask_y() {
        // Test MaskY function
        let result = mask_y(1.0);
        assert!(result > 0.0);
        assert!(result.is_finite());

        // Higher delta should result in lower masking
        let result_high = mask_y(10.0);
        assert!(result_high < result);
    }

    #[test]
    fn test_mask_dc_y() {
        // Test MaskDcY function
        let result = mask_dc_y(1.0);
        assert!(result > 0.0);
        assert!(result.is_finite());

        // Higher delta should result in lower masking
        let result_high = mask_dc_y(10.0);
        assert!(result_high < result);
    }

    #[test]
    fn test_fuzzy_erosion_weights() {
        // Test that weights sum to 1.0
        let weights_sum: f64 = 0.45 + 0.3 + 0.25;
        assert!((weights_sum - 1.0).abs() < 0.001);
    }
}

#[test]
fn test_mask_y_cpp_values() {
    // Verify MaskY matches C++ implementation
    // C++ calculation for delta=1.0:
    // c = 2.5485944793 / (0.451936922203 * 1.0 + 0.829591754942) = 1.989
    // retval = 0.0709 * (1.0 + 1.989) = 0.2119
    // return 0.2119^2 = 0.0449

    let result = mask_y(1.0);
    println!("MaskY(1.0) = {}", result);

    // Calculate expected value
    let offset = 0.829591754942;
    let scaler = 0.451936922203;
    let mul = 2.5485944793;
    let global_scale = 1.0 / (17.83 * 0.790799174);

    let c = mul / (scaler * 1.0 + offset);
    let retval = global_scale * (1.0 + c);
    let expected = retval * retval;

    println!(
        "Expected: {}, c={}, retval={}, global_scale={}",
        expected, c, retval, global_scale
    );

    assert!(
        (result - expected).abs() < 1e-6,
        "MaskY(1.0) = {}, expected {}",
        result,
        expected
    );
}
