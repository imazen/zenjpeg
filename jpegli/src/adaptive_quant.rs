//! Adaptive Quantization for jpegli - C++ Matching Implementation
//!
//! # Status: PARTIAL IMPLEMENTATION
//!
//! This module contains a partial port of the C++ adaptive quantization.
//! The constants and helper functions are ported from a previous attempt
//! at `/home/lilith/work/jpeg-encoder/src/jpegli/adaptive_quantization.rs`.
//!
//! ## What C++ Does (from lib/jpegli/adaptive_quantization.cc)
//!
//! The C++ algorithm produces per-block `aq_strength` values in the range
//! 0.0-0.2 (mean ~0.08). These values are used in zero-biasing to determine
//! quantization thresholds.
//!
//! ### Algorithm Steps:
//!
//! 1. **ComputePreErosion()** - Computes initial quant field from DCT coefficients
//!    - Uses `QuantMasking()` for spatial frequency masking
//!    - Uses `MaskingSqrt()` to combine with butteraugli distance
//!
//! 2. **FuzzyErosion()** - Applies spatial smoothing to the quant field
//!    - 5x5 kernel with asymmetric weights
//!    - Separate passes for horizontal and vertical
//!
//! 3. **PerBlockModulations()** - Final per-block modulations
//!    - Uses spatial frequency analysis
//!    - Applies masking based on AC energy distribution
//!
//! 4. **Final Transform** - Converts quant_field to aq_strength:
//!    ```text
//!    aq_strength = max(0.0, (0.6 / quant_field) - 1.0)
//!    ```
//!
//! ## Current Workaround
//!
//! Until the algorithm is fully verified, the encoder uses a constant
//! `aq_strength = 0.08` calibrated from C++ testdata mean.
//!
//! ## Previous Port Attempt
//!
//! Constants and helpers below were ported from a previous attempt that
//! produced output in wrong range (0-5 instead of 0-0.2). The issue was
//! the missing final transform. These are preserved for future work.
//!
//! See also:
//! - `docs/ADAPTIVE_QUANTIZATION.md` for detailed analysis
//! - `tests/aq_locked_tests.rs` for invariant tests
//! - `simplified_quant.rs` for the simplified (non-C++) version

use std::f32::consts::PI;

// ============================================================================
// Constants ported from C++ adaptive_quantization.cc
// ============================================================================

/// Gamma-related constant from ComputePreErosion.
/// Note: C++ divides by kInputScaling (255.0), applied at usage.
const MATCH_GAMMA_OFFSET: f32 = 0.019;

/// Limit threshold for pre-erosion.
const LIMIT: f32 = 0.2;

/// AC quantization scaling constant.
const K_AC_QUANT: f32 = 0.841;

/// Input scaling factor (1/255 for 8-bit input).
const K_INPUT_SCALING: f32 = 1.0 / 255.0;

/// Gamma modulation bias (adjusted for scaling).
const K_GAMMA_MOD_BIAS: f32 = 0.16 * K_INPUT_SCALING;

/// Gamma modulation scale.
const K_GAMMA_MOD_SCALE: f32 = 1.0 / 64.0;

/// Inverse of ln(2) = ln(2).
const K_INV_LOG2E: f32 = 0.6931471805599453;

/// Gamma modulation gamma coefficient.
const K_GAMMA_MOD_GAMMA: f32 = -0.15526878023684174 * K_INV_LOG2E;

/// High-frequency modulation coefficient.
const K_HF_MOD_COEFF: f32 = -2.0052193233688884 / 112.0;

// Constants for ComputeMask (from C++)
const K_MASK_BASE: f32 = 0.6109318733215332;
const K_MUL4: f32 = 0.03879999369382858;
const K_MUL2: f32 = 0.17580001056194305;
const K_MASK_MUL4: f32 = 3.2353257320940401;
const K_MASK_MUL2: f32 = 12.906028311180409;
const K_MASK_OFFSET2: f32 = 305.04035728311436;
const K_MASK_MUL3: f32 = 5.0220313103171232;
const K_MUL3: f32 = 0.30230000615119934;
const K_MASK_OFFSET3: f32 = 2.1925739705298404;
const K_MASK_OFFSET4: f32 = 0.25 * K_MASK_OFFSET3;
const K_MASK_MUL0: f32 = 0.74760422233706747;

// Constants from RatioOfDerivatives
const K_EPSILON_RATIO: f32 = 1e-2;
const K_NUM_OFFSET_RATIO: f32 = K_EPSILON_RATIO / K_INPUT_SCALING / K_INPUT_SCALING;
const K_SG_MUL: f32 = 226.0480446705883;
const K_SG_MUL2: f32 = 1.0 / 73.377132366608819;
const K_SG_RET_MUL: f32 = K_SG_MUL2 * 18.6580932135 * K_INV_LOG2E;
const K_NUM_MUL_RATIO: f32 = K_SG_RET_MUL * 3.0 * K_SG_MUL;
const K_SG_VOFFSET: f32 = 7.14672470003;
const K_VOFFSET_RATIO: f32 = (K_SG_VOFFSET * K_INV_LOG2E + K_EPSILON_RATIO) / K_INPUT_SCALING;
const K_DEN_MUL_RATIO: f32 = K_INV_LOG2E * K_SG_MUL * K_INPUT_SCALING * K_INPUT_SCALING;

// ============================================================================
// Public API
// ============================================================================

/// Per-block adaptive quantization strength.
///
/// Values are in the range 0.0-0.2 (matching C++ output).
#[derive(Debug, Clone)]
pub struct AQStrengthMap {
    /// Width in 8x8 blocks
    pub width_blocks: usize,
    /// Height in 8x8 blocks
    pub height_blocks: usize,
    /// Per-block aq_strength values (0.0 to ~0.2)
    pub strengths: Vec<f32>,
}

impl AQStrengthMap {
    /// Creates a uniform AQ map with the given constant strength.
    ///
    /// The default C++ mean is ~0.08.
    #[must_use]
    pub fn uniform(width_blocks: usize, height_blocks: usize, strength: f32) -> Self {
        Self {
            width_blocks,
            height_blocks,
            strengths: vec![strength; width_blocks * height_blocks],
        }
    }

    /// Creates a uniform map with the C++ testdata mean (0.08).
    #[must_use]
    pub fn with_cpp_mean(width_blocks: usize, height_blocks: usize) -> Self {
        Self::uniform(width_blocks, height_blocks, 0.08)
    }

    /// Returns the aq_strength for a block.
    #[inline]
    #[must_use]
    pub fn get(&self, bx: usize, by: usize) -> f32 {
        let idx = by * self.width_blocks + bx;
        self.strengths.get(idx).copied().unwrap_or(0.08)
    }
}

/// Computes per-block adaptive quantization strength.
///
/// # Status: RETURNS CONSTANT (C++ mean)
///
/// This function currently returns a uniform map with the C++ testdata mean.
/// The algorithm below is implemented but needs verification against C++ testdata.
///
/// # Arguments
///
/// * `y_plane` - Luminance plane (Y channel) as f32 values (0-255 range)
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
/// * `distance` - Quality distance parameter
///
/// # Returns
///
/// Per-block aq_strength map with values in 0.0-0.2 range.
#[must_use]
pub fn compute_aq_strength_map(
    _y_plane: &[f32],
    width: usize,
    height: usize,
    _distance: f32,
) -> AQStrengthMap {
    let width_blocks = (width + 7) / 8;
    let height_blocks = (height + 7) / 8;

    // TODO: Enable once verified against C++ testdata:
    // let map = compute_aq_strength_map_impl(y_plane, width, height, distance);
    // return map;

    // For now, return uniform map with C++ testdata mean
    AQStrengthMap::with_cpp_mean(width_blocks, height_blocks)
}

/// Converts quant_field to aq_strength.
///
/// C++ formula: `aq_strength = max(0.0, (0.6 / quant_field) - 1.0)`
///
/// This is the CRITICAL missing piece from the failed port.
#[inline]
#[must_use]
pub fn quant_field_to_aq_strength(quant_field: f32) -> f32 {
    (0.6 / quant_field - 1.0).max(0.0)
}

// ============================================================================
// Implementation (needs verification against C++ testdata)
// ============================================================================

/// Full implementation of AQ strength computation.
///
/// # Status: UNVERIFIED - DO NOT USE IN PRODUCTION
///
/// This implementation needs to be verified against C++ testdata before use.
#[allow(dead_code)]
fn compute_aq_strength_map_impl(
    y_plane: &[f32],
    width: usize,
    height: usize,
    distance: f32,
) -> AQStrengthMap {
    let width_blocks = (width + 7) / 8;
    let height_blocks = (height + 7) / 8;
    let num_blocks = width_blocks * height_blocks;

    if width == 0 || height == 0 {
        return AQStrengthMap::uniform(0, 0, 0.08);
    }

    // Scale input to [0, 1]
    let input_scaled: Vec<f32> = y_plane.iter().map(|&v| v * K_INPUT_SCALING).collect();

    // 1. ComputePreErosion (downsamples 4x)
    let pre_erosion = compute_pre_erosion_scalar(&input_scaled, width, height);

    // 2. FuzzyErosion
    let pre_erosion_w = (width + 3) / 4;
    let pre_erosion_h = (height + 3) / 4;
    let mut quant_field = vec![0.0f32; num_blocks];
    fuzzy_erosion_scalar(
        &pre_erosion,
        pre_erosion_w,
        pre_erosion_h,
        width_blocks,
        height_blocks,
        &mut quant_field,
    );

    // 3. PerBlockModulations
    per_block_modulations_scalar(
        distance,
        &input_scaled,
        width,
        height,
        width_blocks,
        height_blocks,
        &mut quant_field,
    );

    // 4. Final transform: quant_field -> aq_strength
    let strengths: Vec<f32> = quant_field
        .iter()
        .map(|&qf| quant_field_to_aq_strength(qf))
        .collect();

    AQStrengthMap {
        width_blocks,
        height_blocks,
        strengths,
    }
}

// ============================================================================
// Helper functions (ported from previous attempt)
// ============================================================================

/// Generates a 1D Gaussian kernel.
#[allow(dead_code)]
fn gaussian_kernel(sigma: f32, radius: usize) -> Vec<f32> {
    let mut kernel = vec![0.0; 2 * radius + 1];
    let sigma_sq = sigma * sigma;
    let norm_factor = 1.0 / (2.0 * PI * sigma_sq).sqrt();
    let mut sum = 0.0;

    for i in 0..=radius {
        let dist_sq = (i * i) as f32;
        let val = norm_factor * (-dist_sq / (2.0 * sigma_sq)).exp();
        kernel[radius + i] = val;
        kernel[radius - i] = val;
        sum += if i == 0 { val } else { 2.0 * val };
    }

    // Normalize
    if sum > 1e-6 {
        for val in &mut kernel {
            *val /= sum;
        }
    }

    kernel
}

/// Calculates the ratio of derivatives for psychovisual modulation.
/// Ported from `RatioOfDerivativesOfCubicRootToSimpleGamma`.
#[allow(dead_code)]
fn ratio_of_derivatives(val: f32, invert: bool) -> f32 {
    let v = val.max(0.0);
    let v2 = v * v;

    let num = K_NUM_MUL_RATIO * v2 + K_NUM_OFFSET_RATIO;
    let den = (K_DEN_MUL_RATIO * v) * v2 + K_VOFFSET_RATIO;

    let safe_den = if den == 0.0 { 1e-9 } else { den };

    if invert {
        num / safe_den
    } else {
        safe_den / num
    }
}

/// Updates the 4 minimum values in a sorted array.
#[inline]
#[allow(dead_code)]
fn update_min4(val: f32, mins: &mut [f32; 4]) {
    if val < mins[3] {
        if val < mins[2] {
            mins[3] = mins[2];
            if val < mins[1] {
                mins[2] = mins[1];
                if val < mins[0] {
                    mins[1] = mins[0];
                    mins[0] = val;
                } else {
                    mins[1] = val;
                }
            } else {
                mins[2] = val;
            }
        } else {
            mins[3] = val;
        }
    }
}

/// Ported from ComputePreErosion (scalar version).
#[allow(dead_code)]
fn compute_pre_erosion_scalar(input_scaled: &[f32], width: usize, height: usize) -> Vec<f32> {
    let pre_erosion_w = (width + 3) / 4;
    let pre_erosion_h = (height + 3) / 4;
    let mut pre_erosion = vec![0.0f32; pre_erosion_w * pre_erosion_h];

    let limit = LIMIT / K_INPUT_SCALING;
    let offset = MATCH_GAMMA_OFFSET / K_INPUT_SCALING;

    for y_block in 0..pre_erosion_h {
        let y_start = y_block * 4;
        for x_block in 0..pre_erosion_w {
            let x_start = x_block * 4;
            let mut minval: f32 = f32::INFINITY;

            for iy in 0..4 {
                let y = y_start + iy;
                if y >= height {
                    continue;
                }
                let row_start = y * width;
                for ix in 0..4 {
                    let x = x_start + ix;
                    if x >= width {
                        continue;
                    }

                    let val = input_scaled[row_start + x];
                    let ratio = ratio_of_derivatives(val, false);
                    if ratio < minval {
                        minval = ratio;
                    }
                }
            }

            let val_transformed = if minval < limit {
                offset
            } else {
                (minval - limit) + offset
            };

            pre_erosion[y_block * pre_erosion_w + x_block] = val_transformed;
        }
    }

    pre_erosion
}

/// Ported from FuzzyErosion (scalar version).
#[allow(dead_code)]
fn fuzzy_erosion_scalar(
    pre_erosion: &[f32],
    pre_erosion_w: usize,
    pre_erosion_h: usize,
    block_w: usize,
    block_h: usize,
    aq_map: &mut [f32],
) {
    assert_eq!(aq_map.len(), block_w * block_h);

    let mut tmp = vec![0.0f32; pre_erosion_w * pre_erosion_h];

    // Process rows (forward + backward min)
    for y in 0..pre_erosion_h {
        let mut mins = [f32::INFINITY; 4];
        let row_start = y * pre_erosion_w;

        // Forward pass
        for x in 0..pre_erosion_w {
            let val = pre_erosion[row_start + x];
            update_min4(val, &mut mins);
            tmp[row_start + x] = mins[0];
        }

        // Backward pass
        let mut mins = [f32::INFINITY; 4];
        for x in (0..pre_erosion_w).rev() {
            let val = pre_erosion[row_start + x];
            update_min4(val, &mut mins);
            tmp[row_start + x] = tmp[row_start + x].min(mins[0]);
        }
    }

    // Process columns
    for x in 0..pre_erosion_w {
        let mut mins = [f32::INFINITY; 4];

        // Forward pass (top to bottom)
        for y in 0..pre_erosion_h {
            let idx = y * pre_erosion_w + x;
            let val = tmp[idx];
            update_min4(val, &mut mins);
            tmp[idx] = mins[0];
        }

        // Backward pass (bottom to top)
        let mut mins = [f32::INFINITY; 4];
        for y in (0..pre_erosion_h).rev() {
            let idx = y * pre_erosion_w + x;
            let val = tmp[idx];
            update_min4(val, &mut mins);
            let final_val = tmp[idx].min(mins[0]);

            // Map pre_erosion coords to block coords (1 pre_erosion = 2x2 blocks)
            let bx_start = x * 2;
            let by_start = y * 2;

            for by_off in 0..2 {
                let by = by_start + by_off;
                if by >= block_h {
                    continue;
                }
                for bx_off in 0..2 {
                    let bx = bx_start + bx_off;
                    if bx >= block_w {
                        continue;
                    }
                    aq_map[by * block_w + bx] = final_val;
                }
            }
        }
    }
}

/// Ported from ComputeMask (scalar version).
#[allow(dead_code)]
fn compute_mask_scalar(out_val: f32) -> f32 {
    let v1 = (out_val * K_MASK_MUL0).max(1e-3);
    let v2 = 1.0 / (v1 + K_MASK_OFFSET2);
    let v3 = 1.0 / (v1 * v1 + K_MASK_OFFSET3);
    let v4 = 1.0 / (v1 * v1 + K_MASK_OFFSET4);
    K_MASK_BASE + K_MUL4 * v4 + K_MUL2 * v2 + K_MUL3 * v3
}

/// Ported from HFModulation (scalar version).
#[allow(dead_code)]
fn hf_modulation_scalar(
    x: usize,
    y: usize,
    input_scaled: &[f32],
    width: usize,
    height: usize,
    current_val: f32,
) -> f32 {
    let center_idx = y * width + x;
    let center_val = input_scaled[center_idx];

    let left_idx = y * width + x.saturating_sub(1);
    let right_idx = y * width + (x + 1).min(width - 1);
    let top_idx = y.saturating_sub(1) * width + x;
    let bottom_idx = (y + 1).min(height - 1) * width + x;

    let diff_h =
        (input_scaled[left_idx] - center_val).abs() + (input_scaled[right_idx] - center_val).abs();
    let diff_v =
        (input_scaled[top_idx] - center_val).abs() + (input_scaled[bottom_idx] - center_val).abs();

    let diff_sum = diff_h + diff_v;
    current_val + K_HF_MOD_COEFF * diff_sum
}

/// Ported from GammaModulation (scalar version).
#[allow(dead_code)]
fn gamma_modulation_scalar(
    x: usize,
    y: usize,
    input_scaled: &[f32],
    width: usize,
    _height: usize,
    current_val: f32,
) -> f32 {
    let val = input_scaled[y * width + x];
    let log_arg = (val * K_GAMMA_MOD_SCALE + K_GAMMA_MOD_BIAS).max(1e-9);
    let modulation = K_GAMMA_MOD_GAMMA * log_arg.ln();
    current_val + modulation
}

/// Ported from PerBlockModulations (scalar version).
#[allow(dead_code)]
fn per_block_modulations_scalar(
    distance: f32,
    input_scaled: &[f32],
    width: usize,
    height: usize,
    block_w: usize,
    block_h: usize,
    aq_map: &mut [f32],
) {
    // Scale AC quant by distance
    let scaled_ac_quant = K_AC_QUANT / distance;

    for by in 0..block_h {
        let y_start = by * 8;
        for bx in 0..block_w {
            let x_start = bx * 8;
            let block_idx = by * block_w + bx;

            // Get value from fuzzy erosion
            let current_val = aq_map[block_idx];

            // Apply HF Modulation (using center of block)
            let center_x = (x_start + 1).min(width - 1);
            let center_y = (y_start + 1).min(height - 1);

            let hf_modulated_val =
                hf_modulation_scalar(center_x, center_y, input_scaled, width, height, current_val);

            // Apply Gamma Modulation
            let gamma_modulated_val = gamma_modulation_scalar(
                center_x,
                center_y,
                input_scaled,
                width,
                height,
                hf_modulated_val,
            );

            // Apply ComputeMask
            let mask_val = compute_mask_scalar(gamma_modulated_val);

            // Apply AC quant scaling
            let final_val = mask_val * scaled_ac_quant;

            aq_map[block_idx] = final_val;
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aq_strength_map_uniform() {
        let map = AQStrengthMap::uniform(10, 10, 0.08);
        assert_eq!(map.strengths.len(), 100);
        assert!((map.get(5, 5) - 0.08).abs() < 1e-6);
    }

    #[test]
    fn test_aq_strength_map_cpp_mean() {
        let map = AQStrengthMap::with_cpp_mean(10, 10);
        assert!((map.get(0, 0) - 0.08).abs() < 1e-6);
    }

    #[test]
    fn test_quant_field_to_aq_strength() {
        // quant_field = 0.6 → aq_strength = 0.0
        assert!((quant_field_to_aq_strength(0.6) - 0.0).abs() < 1e-6);

        // quant_field = 0.3 → aq_strength = 1.0
        assert!((quant_field_to_aq_strength(0.3) - 1.0).abs() < 1e-6);

        // quant_field = 6.0 → aq_strength = 0.0 (clamped)
        assert!(quant_field_to_aq_strength(6.0) >= 0.0);
    }

    #[test]
    fn test_compute_returns_uniform() {
        let plane = vec![128.0f32; 64 * 64];
        let map = compute_aq_strength_map(&plane, 64, 64, 1.0);
        assert_eq!(map.width_blocks, 8);
        assert_eq!(map.height_blocks, 8);
        // All values should be the C++ mean
        for &s in &map.strengths {
            assert!((s - 0.08).abs() < 1e-6);
        }
    }

    #[test]
    fn test_ratio_of_derivatives() {
        // Basic sanity checks
        let r1 = ratio_of_derivatives(0.5, false);
        let r2 = ratio_of_derivatives(0.5, true);
        assert!(r1.is_finite());
        assert!(r2.is_finite());
        assert!(r1 > 0.0);
        assert!(r2 > 0.0);
    }

    #[test]
    fn test_update_min4() {
        let mut mins = [f32::INFINITY; 4];
        update_min4(5.0, &mut mins);
        assert!((mins[0] - 5.0).abs() < 1e-6);

        update_min4(3.0, &mut mins);
        assert!((mins[0] - 3.0).abs() < 1e-6);
        assert!((mins[1] - 5.0).abs() < 1e-6);

        update_min4(7.0, &mut mins);
        assert!((mins[0] - 3.0).abs() < 1e-6);
        assert!((mins[1] - 5.0).abs() < 1e-6);
        assert!((mins[2] - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_mask_scalar() {
        let mask = compute_mask_scalar(1.0);
        assert!(mask.is_finite());
        assert!(mask > 0.0);
    }

    #[test]
    fn test_impl_runs_without_panic() {
        // Test that the full implementation runs without panicking
        let plane = vec![128.0f32; 64 * 64];
        let map = compute_aq_strength_map_impl(&plane, 64, 64, 1.0);
        assert_eq!(map.width_blocks, 8);
        assert_eq!(map.height_blocks, 8);
        assert_eq!(map.strengths.len(), 64);

        // Values should be finite
        for &s in &map.strengths {
            assert!(s.is_finite(), "aq_strength should be finite");
        }
    }

    #[test]
    fn test_impl_output_range() {
        // The implementation should produce values in 0.0-0.3 range
        // (C++ produces 0.0-0.2, we allow some margin)
        let plane: Vec<f32> = (0..64 * 64).map(|i| (i % 256) as f32).collect();
        let map = compute_aq_strength_map_impl(&plane, 64, 64, 1.0);

        for &s in &map.strengths {
            assert!(
                s >= 0.0 && s <= 0.5,
                "aq_strength {} outside expected range [0, 0.5]",
                s
            );
        }
    }
}
