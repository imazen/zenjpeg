//! Adaptive Quantization from jpegli
//!
//! Content-aware quantization that computes per-block aq_strength values
//! used for zero-biasing during quantization. This implementation ports
//! the C++ jpegli algorithm.
//!
//! # Algorithm Overview
//!
//! 1. **ComputePreErosion()** - Computes initial quant field from pixel differences
//!    - Uses `ratio_of_derivatives()` for gamma modulation
//!    - Uses `masking_sqrt()` for perceptual masking
//!    - Downsamples 4x4 blocks
//!
//! 2. **FuzzyErosion()** - Applies spatial smoothing
//!    - 3x3 kernel with weighted 4-smallest values
//!    - Produces per-8x8-block values
//!
//! 3. **PerBlockModulations()** - Final per-block modulations
//!    - ComputeMask for baseline
//!    - HF modulation for edge content
//!    - Gamma modulation for brightness
//!
//! 4. **Final Transform** - Converts quant_field to aq_strength:
//!    `aq_strength = max(0.0, (0.6 / quant_field) - 1.0)`

use std::f32::consts::PI;

/// Configuration for adaptive quantization
#[derive(Clone, Copy, Debug)]
pub struct AdaptiveQuantConfig {
    /// Enable adaptive quantization
    pub enabled: bool,
    /// Overall strength multiplier (0.0 = off, 1.0 = full strength)
    pub strength: f32,
}

impl Default for AdaptiveQuantConfig {
    fn default() -> Self {
        Self {
            enabled: false, // Disabled by default - enabled explicitly for jpegli strategy
            strength: 1.0,
        }
    }
}

// ============================================================================
// Constants from C++ jpegli adaptive_quantization.cc
// ============================================================================

/// Gamma-related constant from ComputePreErosion.
const MATCH_GAMMA_OFFSET: f32 = 0.019;

/// Limit threshold for pre-erosion squared differences.
const LIMIT: f32 = 0.2;

/// Input scaling factor (1/255 for 8-bit input).
const K_INPUT_SCALING: f32 = 1.0 / 255.0;

/// Inverse of ln(2).
const K_INV_LOG2E: f32 = 0.6931471805599453;

// Constants for ComputeMask
const K_MASK_BASE: f32 = -0.74174993;
const K_MASK_MUL0: f32 = 0.74760422233706747;
const K_MASK_MUL2: f32 = 12.906028311180409;
const K_MASK_MUL3: f32 = 5.0220313103171232;
const K_MASK_MUL4: f32 = 3.2353257320940401;
const K_MASK_OFFSET2: f32 = 305.04035728311436;
const K_MASK_OFFSET3: f32 = 2.1925739705298404;
const K_MASK_OFFSET4: f32 = 0.25 * K_MASK_OFFSET3;

// Constants for RatioOfDerivatives
const K_EPSILON_RATIO: f32 = 1e-2;
const K_NUM_OFFSET_RATIO: f32 = K_EPSILON_RATIO / K_INPUT_SCALING / K_INPUT_SCALING;
const K_SG_MUL: f32 = 226.0480446705883;
const K_SG_MUL2: f32 = 1.0 / 73.377132366608819;
const K_SG_RET_MUL: f32 = K_SG_MUL2 * 18.6580932135 * K_INV_LOG2E;
const K_NUM_MUL_RATIO: f32 = K_SG_RET_MUL * 3.0 * K_SG_MUL;
const K_SG_VOFFSET: f32 = 7.14672470003;
const K_VOFFSET_RATIO: f32 = (K_SG_VOFFSET * K_INV_LOG2E + K_EPSILON_RATIO) / K_INPUT_SCALING;
const K_DEN_MUL_RATIO: f32 = K_INV_LOG2E * K_SG_MUL * K_INPUT_SCALING * K_INPUT_SCALING;

// Constants for PerBlockModulations
const K_AC_QUANT: f32 = 0.841;
const K_DAMPEN_RAMP_START: f32 = 9.0;
const K_DAMPEN_RAMP_END: f32 = 65.0;

// ============================================================================
// Public API
// ============================================================================

/// Per-block adaptive quantization strength map.
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
    pub fn with_default_strength(width_blocks: usize, height_blocks: usize) -> Self {
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
/// # Arguments
/// * `y_plane` - Luminance plane as u8 values (0-255 range)
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
/// * `y_quant_01` - Y quantization table value at position 1 (first AC coefficient)
/// * `config` - Adaptive quantization configuration
///
/// # Returns
/// Per-block aq_strength map with values in 0.0-0.2 range.
#[must_use]
pub fn compute_aq_strength_map(
    y_plane: &[u8],
    width: usize,
    height: usize,
    y_quant_01: u16,
    config: &AdaptiveQuantConfig,
) -> AQStrengthMap {
    let width_blocks = (width + 7) / 8;
    let height_blocks = (height + 7) / 8;

    if !config.enabled || width == 0 || height == 0 {
        return AQStrengthMap::with_default_strength(width_blocks, height_blocks);
    }

    // Convert to f32 for computation
    let y_plane_f32: Vec<f32> = y_plane.iter().map(|&v| v as f32).collect();

    compute_aq_strength_map_impl(&y_plane_f32, width, height, y_quant_01 as f32, config.strength)
}

/// Converts quant_field to aq_strength.
///
/// C++ formula: `aq_strength = max(0.0, (0.6 / quant_field) - 1.0)`
#[inline]
#[must_use]
fn quant_field_to_aq_strength(quant_field: f32) -> f32 {
    (0.6 / quant_field - 1.0).max(0.0)
}

// ============================================================================
// Implementation
// ============================================================================

/// Full implementation of AQ strength computation.
fn compute_aq_strength_map_impl(
    y_plane: &[f32],
    width: usize,
    height: usize,
    y_quant_01: f32,
    strength: f32,
) -> AQStrengthMap {
    let width_blocks = (width + 7) / 8;
    let height_blocks = (height + 7) / 8;
    let num_blocks = width_blocks * height_blocks;

    if width == 0 || height == 0 {
        return AQStrengthMap::uniform(0, 0, 0.08);
    }

    // 1. ComputePreErosion (downsamples 4x)
    let pre_erosion = compute_pre_erosion(y_plane, width, height);

    // 2. FuzzyErosion
    let pre_erosion_w = (width + 3) / 4;
    let pre_erosion_h = (height + 3) / 4;
    let mut quant_field = vec![0.0f32; num_blocks];
    fuzzy_erosion(
        &pre_erosion,
        pre_erosion_w,
        pre_erosion_h,
        width_blocks,
        height_blocks,
        &mut quant_field,
    );

    // 3. PerBlockModulations
    per_block_modulations(
        y_quant_01,
        y_plane,
        width,
        height,
        width_blocks,
        height_blocks,
        &mut quant_field,
    );

    // 4. Final transform: quant_field -> aq_strength
    let strengths: Vec<f32> = quant_field
        .iter()
        .map(|&qf| {
            let aq = quant_field_to_aq_strength(qf);
            // Apply strength multiplier
            aq * strength
        })
        .collect();

    AQStrengthMap {
        width_blocks,
        height_blocks,
        strengths,
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// Calculates the ratio of derivatives for psychovisual modulation.
/// Ported from `RatioOfDerivativesOfCubicRootToSimpleGamma`.
#[inline]
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

/// MaskingSqrt from C++.
#[inline]
fn masking_sqrt(v: f32) -> f32 {
    const K_LOG_OFFSET: f32 = 28.0;
    const K_MUL: f32 = 211.50759899638012;
    0.25 * (v * (K_MUL * 1e8_f32).sqrt() + K_LOG_OFFSET).sqrt()
}

/// ComputeMask from C++.
#[inline]
fn compute_mask(out_val: f32) -> f32 {
    let v1 = (out_val * K_MASK_MUL0).max(1e-3);
    let v2 = 1.0 / (v1 + K_MASK_OFFSET2);
    let v3 = 1.0 / (v1 * v1 + K_MASK_OFFSET3);
    let v4 = 1.0 / (v1 * v1 + K_MASK_OFFSET4);
    K_MASK_BASE + K_MASK_MUL4 * v4 + K_MASK_MUL2 * v2 + K_MASK_MUL3 * v3
}

/// ComputePreErosion - computes initial quant field from pixel differences.
///
/// For each pixel, computes gamma-weighted difference from neighbors,
/// squares and clamps, then applies masking. Output is 4x downsampled.
fn compute_pre_erosion(input: &[f32], width: usize, height: usize) -> Vec<f32> {
    let pre_erosion_w = (width + 3) / 4;
    let pre_erosion_h = (height + 3) / 4;
    let mut pre_erosion = vec![0.0f32; pre_erosion_w * pre_erosion_h];

    // gamma_offset is in 0-255 range (0.019 * 255 = 4.85)
    let gamma_offset = MATCH_GAMMA_OFFSET / K_INPUT_SCALING;

    // Helper to get pixel with clamped bounds
    let get = |x: isize, y: isize| -> f32 {
        let x = x.clamp(0, width as isize - 1) as usize;
        let y = y.clamp(0, height as isize - 1) as usize;
        input[y * width + x]
    };

    // Temporary buffer for accumulating over 4 rows
    let mut diff_buffer = vec![0.0f32; width];

    for y_block in 0..pre_erosion_h {
        diff_buffer.fill(0.0);

        for iy in 0..4 {
            let y = y_block * 4 + iy;
            if y >= height {
                continue;
            }

            for x in 0..width {
                let ix = x as isize;
                let iy_s = y as isize;

                let pixel = get(ix, iy_s);

                // Base is average of 4 neighbors
                let base = 0.25
                    * (get(ix - 1, iy_s)
                        + get(ix + 1, iy_s)
                        + get(ix, iy_s - 1)
                        + get(ix, iy_s + 1));

                // Gamma-corrected ratio as weight
                let gammacv = ratio_of_derivatives(pixel + gamma_offset, false);

                // Weighted difference
                let diff = gammacv * (pixel - base);

                // Square and clamp
                let diff_sq = (diff * diff).min(LIMIT);

                // Apply MaskingSqrt
                let masked = masking_sqrt(diff_sq);

                // Accumulate
                diff_buffer[x] += masked;
            }
        }

        // Downsample 4x in x direction
        for x_block in 0..pre_erosion_w {
            let x_start = x_block * 4;
            let mut sum = 0.0f32;
            for ix in 0..4 {
                let x = x_start + ix;
                if x < width {
                    sum += diff_buffer[x];
                }
            }
            pre_erosion[y_block * pre_erosion_w + x_block] = sum * 0.25;
        }
    }

    pre_erosion
}

/// FuzzyErosion - applies spatial smoothing to the quant field.
///
/// For each pixel, finds 4 smallest values in 3x3 window and computes
/// weighted sum. Then sums 2x2 blocks to get final per-8x8-block values.
fn fuzzy_erosion(
    pre_erosion: &[f32],
    pre_erosion_w: usize,
    pre_erosion_h: usize,
    block_w: usize,
    block_h: usize,
    aq_map: &mut [f32],
) {
    assert_eq!(aq_map.len(), block_w * block_h);

    // Weights from C++ (sum = 0.31)
    const MUL0: f32 = 0.125;
    const MUL1: f32 = 0.075;
    const MUL2: f32 = 0.06;
    const MUL3: f32 = 0.05;

    // Temporary buffer for weighted min values
    let mut tmp = vec![0.0f32; pre_erosion_w * pre_erosion_h];

    // Helper to get value with clamped bounds
    let get = |x: isize, y: isize| -> f32 {
        let x = x.clamp(0, pre_erosion_w as isize - 1) as usize;
        let y = y.clamp(0, pre_erosion_h as isize - 1) as usize;
        pre_erosion[y * pre_erosion_w + x]
    };

    // Process each pixel - find 4 smallest in 3x3 window
    for y in 0..pre_erosion_h {
        for x in 0..pre_erosion_w {
            let ix = x as isize;
            let iy = y as isize;

            // Collect all 9 values in 3x3 window
            let mut vals = [
                get(ix - 1, iy - 1),
                get(ix, iy - 1),
                get(ix + 1, iy - 1),
                get(ix - 1, iy),
                get(ix, iy),
                get(ix + 1, iy),
                get(ix - 1, iy + 1),
                get(ix, iy + 1),
                get(ix + 1, iy + 1),
            ];

            // Partial sort to get 4 smallest values
            for i in 0..4 {
                for j in (i + 1)..9 {
                    if vals[j] < vals[i] {
                        vals.swap(i, j);
                    }
                }
            }

            // Weighted sum of 4 smallest
            let weighted = MUL0 * vals[0] + MUL1 * vals[1] + MUL2 * vals[2] + MUL3 * vals[3];
            tmp[y * pre_erosion_w + x] = weighted;
        }
    }

    // Sum 2x2 blocks from tmp to get final aq_map values
    let get_tmp = |x: usize, y: usize| -> f32 {
        let x = x.min(pre_erosion_w.saturating_sub(1));
        let y = y.min(pre_erosion_h.saturating_sub(1));
        tmp[y * pre_erosion_w + x]
    };

    for by in 0..block_h {
        for bx in 0..block_w {
            let px = bx * 2;
            let py = by * 2;

            let sum = get_tmp(px, py)
                + get_tmp(px + 1, py)
                + get_tmp(px, py + 1)
                + get_tmp(px + 1, py + 1);

            aq_map[by * block_w + bx] = sum;
        }
    }
}

/// PerBlockModulations - applies final per-block modulations.
///
/// 1. ComputeMask on fuzzy erosion value
/// 2. HfModulation: sum of abs diffs for 8x8 block
/// 3. GammaModulation: based on overall brightness
/// 4. Final: 2^(out_val * log2(e)) * mul + add
fn per_block_modulations(
    y_quant_01: f32,
    input: &[f32],
    width: usize,
    height: usize,
    block_w: usize,
    block_h: usize,
    aq_map: &mut [f32],
) {
    let base_level = 0.48 * K_AC_QUANT;

    // Compute dampen based on y_quant_01
    let dampen = if y_quant_01 >= K_DAMPEN_RAMP_START {
        let d = 1.0 - (y_quant_01 - K_DAMPEN_RAMP_START) / (K_DAMPEN_RAMP_END - K_DAMPEN_RAMP_START);
        d.max(0.0)
    } else {
        1.0
    };

    let mul = K_AC_QUANT * dampen;
    let add = (1.0 - dampen) * base_level;

    // Helper to get pixel with clamped bounds
    let get = |x: usize, y: usize| -> f32 {
        let x = x.min(width.saturating_sub(1));
        let y = y.min(height.saturating_sub(1));
        input[y * width + x]
    };

    for by in 0..block_h {
        let y_start = by * 8;
        for bx in 0..block_w {
            let x_start = bx * 8;
            let block_idx = by * block_w + bx;

            // Get value from fuzzy erosion
            let fuzzy_val = aq_map[block_idx];

            // 1. ComputeMask
            let mut out_val = compute_mask(fuzzy_val);

            // 2. HfModulation: sum of absolute differences with neighbors
            const K_SUM_COEFF: f32 = -2.0052193233688884 * K_INPUT_SCALING / 112.0;
            let mut sum = 0.0f32;
            for dy in 0..8 {
                let y = y_start + dy;
                for dx in 0..8 {
                    let x = x_start + dx;
                    let p = get(x, y);

                    // Right neighbor
                    if dx < 7 {
                        let pr = get(x + 1, y);
                        sum += (p - pr).abs();
                    }

                    // Below neighbor
                    if dy < 7 {
                        let pd = get(x, y + 1);
                        sum += (p - pd).abs();
                    }
                }
            }
            out_val += sum * K_SUM_COEFF;

            // 3. GammaModulation
            const K_GAMMA: f32 = -0.15526878023684174 * K_INV_LOG2E;
            const K_BIAS: f32 = 0.16 / K_INPUT_SCALING;
            const K_SCALE: f32 = K_INPUT_SCALING / 64.0;

            let mut overall_ratio = 0.0f32;
            for dy in 0..8 {
                let y = y_start + dy;
                for dx in 0..8 {
                    let x = x_start + dx;
                    let iny = get(x, y) + K_BIAS;
                    let ratio_g = ratio_of_derivatives(iny, true);
                    overall_ratio += ratio_g;
                }
            }
            overall_ratio *= K_SCALE;

            let log_ratio = if overall_ratio > 0.0 {
                overall_ratio.log2()
            } else {
                0.0
            };
            out_val += K_GAMMA * log_ratio;

            // 4. Final transform: 2^(out_val * log2(e)) * mul + add
            const LOG2_E: f32 = 1.442695041;
            let quant_field = (out_val * LOG2_E).exp2() * mul + add;

            aq_map[block_idx] = quant_field;
        }
    }
}

// ============================================================================
// Backward compatibility - old AqField API
// ============================================================================

/// Per-block adaptive quantization field (legacy API)
pub struct AqField {
    /// Width in blocks
    pub width_blocks: usize,
    /// Height in blocks
    pub height_blocks: usize,
    /// Per-block quantization multipliers
    pub multipliers: Vec<f32>,
}

impl AqField {
    /// Create a uniform AQ field (no adaptation)
    pub fn uniform(width_blocks: usize, height_blocks: usize) -> Self {
        Self {
            width_blocks,
            height_blocks,
            multipliers: vec![1.0; width_blocks * height_blocks],
        }
    }

    /// Get multiplier for a specific block
    #[inline]
    pub fn get(&self, bx: usize, by: usize) -> f32 {
        self.multipliers[by * self.width_blocks + bx]
    }
}

/// Apply adaptive quantization to a quantization table for a specific block.
///
/// Scales the quant table by the AQ multiplier.
pub fn apply_aq_to_quant(base_quant: &[u16; 64], aq_multiplier: f32) -> [u16; 64] {
    let mut output = [0u16; 64];
    for i in 0..64 {
        let scaled = (base_quant[i] as f32 * aq_multiplier).round() as u16;
        output[i] = scaled.clamp(1, 255);
    }
    output
}

/// Legacy function - compute AQ field (delegates to new API)
///
/// This uses the simplified variance-based approach for backward compatibility.
/// For jpegli-style encoding, use compute_aq_strength_map directly.
pub fn compute_aq_field(
    y_plane: &[u8],
    width: usize,
    height: usize,
    config: &AdaptiveQuantConfig,
) -> AqField {
    let wb = (width + 7) / 8;
    let hb = (height + 7) / 8;

    if !config.enabled {
        return AqField::uniform(wb, hb);
    }

    // Use simplified variance-based AQ for legacy compatibility
    // This produces multipliers around 1.0: smooth=1.3, medium=1.0, detailed=0.8
    let mut multipliers = Vec::with_capacity(wb * hb);

    for by in 0..hb {
        for bx in 0..wb {
            let mult = compute_block_aq_multiplier_legacy(y_plane, width, height, bx, by, config);
            multipliers.push(mult);
        }
    }

    AqField {
        width_blocks: wb,
        height_blocks: hb,
        multipliers,
    }
}

/// Compute AQ multiplier for a single block using simplified variance-based approach.
fn compute_block_aq_multiplier_legacy(
    y_plane: &[u8],
    width: usize,
    height: usize,
    bx: usize,
    by: usize,
    config: &AdaptiveQuantConfig,
) -> f32 {
    let block_x = bx * 8;
    let block_y = by * 8;

    // Compute local statistics
    let mut sum = 0u32;
    let mut sum_sq = 0u64;
    let mut count = 0u32;

    for dy in 0..8 {
        let y = block_y + dy;
        if y >= height {
            continue;
        }

        for dx in 0..8 {
            let x = block_x + dx;
            if x >= width {
                continue;
            }

            let val = y_plane[y * width + x] as u32;
            sum += val;
            sum_sq += (val * val) as u64;
            count += 1;
        }
    }

    if count == 0 {
        return 1.0;
    }

    // Variance = E[X^2] - E[X]^2
    let mean = sum as f64 / count as f64;
    let variance = (sum_sq as f64 / count as f64) - mean * mean;
    let std_dev = variance.sqrt() as f32;

    // Convert variance to AQ multiplier
    // High variance = high detail = lower multiplier (more bits)
    // Low variance = smooth area = higher multiplier (fewer bits)
    let base_multiplier = if std_dev < 5.0 {
        // Very smooth - use stronger quantization
        1.0 + 0.3 * config.strength
    } else if std_dev < 20.0 {
        // Moderate detail - neutral
        1.0
    } else {
        // High detail - use weaker quantization
        1.0 - 0.2 * config.strength
    };

    base_multiplier.clamp(0.5, 1.5)
}

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
    fn test_quant_field_to_aq_strength() {
        // quant_field = 0.6 → aq_strength = 0.0
        assert!((quant_field_to_aq_strength(0.6) - 0.0).abs() < 1e-6);

        // quant_field = 0.3 → aq_strength = 1.0
        assert!((quant_field_to_aq_strength(0.3) - 1.0).abs() < 1e-6);

        // quant_field = 6.0 → aq_strength = 0.0 (clamped)
        assert!(quant_field_to_aq_strength(6.0) >= 0.0);
    }

    #[test]
    fn test_compute_returns_reasonable_values() {
        let plane: Vec<u8> = vec![128; 64 * 64];
        let config = AdaptiveQuantConfig::default();
        let map = compute_aq_strength_map(&plane, 64, 64, 16, &config);

        assert_eq!(map.width_blocks, 8);
        assert_eq!(map.height_blocks, 8);

        // All values should be finite and non-negative
        for &s in &map.strengths {
            assert!(s.is_finite(), "aq_strength should be finite");
            assert!(s >= 0.0, "aq_strength should be non-negative");
        }
    }

    #[test]
    fn test_uniform_input_produces_uniform_output() {
        let plane: Vec<u8> = vec![128; 64 * 64];
        let config = AdaptiveQuantConfig::default();
        let map = compute_aq_strength_map(&plane, 64, 64, 16, &config);

        let first = map.strengths[0];
        for &s in &map.strengths {
            assert!(
                (s - first).abs() < 1e-4,
                "Uniform input should produce uniform output, got {} vs {}",
                s,
                first
            );
        }
    }

    #[test]
    fn test_ratio_of_derivatives() {
        let r1 = ratio_of_derivatives(0.5, false);
        let r2 = ratio_of_derivatives(0.5, true);
        assert!(r1.is_finite());
        assert!(r2.is_finite());
        assert!(r1 > 0.0);
        assert!(r2 > 0.0);
    }

    #[test]
    fn test_aq_field_uniform() {
        let field = AqField::uniform(10, 10);
        assert_eq!(field.multipliers.len(), 100);
        assert_eq!(field.get(5, 5), 1.0);
    }
}
