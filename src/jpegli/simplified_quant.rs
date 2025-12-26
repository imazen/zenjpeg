//! Simplified Adaptive Quantization for jpegli.
//!
//! ⚠️ WARNING: THIS IS A SIMPLIFIED IMPLEMENTATION ⚠️
//!
//! This module implements a SIMPLIFIED content-aware quantization that
//! DOES NOT MATCH the C++ jpegli algorithm. It uses activity/edge detection
//! with arbitrary thresholds.
//!
//! The C++ jpegli algorithm uses:
//! 1. ComputePreErosion() - initial quant field from DCT
//! 2. FuzzyErosion() - spatial smoothing
//! 3. PerBlockModulations() - spatial frequency masking
//!
//! This simplified version uses:
//! 1. Block activity (variance)
//! 2. Edge detection
//! 3. Arbitrary thresholds (10.0, 5.0, 50.0)
//!
//! Use this only as a placeholder until the proper C++ implementation
//! is ported. For proper C++ matching, see `adaptive_quant.rs` (TODO).

use crate::jpegli::consts::DCT_BLOCK_SIZE;

/// Simplified adaptive quantization mode.
///
/// ⚠️ This is NOT the C++ algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SimplifiedAQMode {
    /// No adaptive quantization
    #[default]
    Off,
    /// Standard simplified AQ (arbitrary thresholds)
    Standard,
    /// Aggressive simplified AQ (smaller files, more artifacts)
    Aggressive,
}

/// Configuration for simplified adaptive quantization.
///
/// ⚠️ This configuration does NOT match C++ jpegli.
#[derive(Debug, Clone)]
pub struct SimplifiedAQConfig {
    /// Quantization mode
    pub mode: SimplifiedAQMode,
    /// Base butteraugli distance
    pub base_distance: f32,
    /// DC quant multiplier
    pub dc_quant_mul: f32,
    /// AC quant multiplier for low frequencies
    pub ac_quant_mul_low: f32,
    /// AC quant multiplier for high frequencies
    pub ac_quant_mul_high: f32,
}

impl Default for SimplifiedAQConfig {
    fn default() -> Self {
        Self {
            mode: SimplifiedAQMode::Off,
            base_distance: 1.0,
            dc_quant_mul: 1.0,
            ac_quant_mul_low: 1.0,
            ac_quant_mul_high: 1.0,
        }
    }
}

/// Simplified quantization map for an image.
///
/// Stores per-block quantization multipliers.
/// ⚠️ Multipliers range 0.5-2.0, NOT C++ aq_strength (0.0-0.2).
#[derive(Debug, Clone)]
pub struct SimplifiedQuantMap {
    /// Width in blocks
    pub width_blocks: usize,
    /// Height in blocks
    pub height_blocks: usize,
    /// Per-block multipliers (1.0 = no adjustment)
    pub multipliers: Vec<f32>,
}

impl SimplifiedQuantMap {
    /// Creates a uniform quantization map (no adaptation).
    #[must_use]
    pub fn uniform(width_blocks: usize, height_blocks: usize) -> Self {
        Self {
            width_blocks,
            height_blocks,
            multipliers: vec![1.0; width_blocks * height_blocks],
        }
    }

    /// Returns the multiplier for a block.
    #[inline]
    #[must_use]
    pub fn get(&self, bx: usize, by: usize) -> f32 {
        let idx = by * self.width_blocks + bx;
        self.multipliers.get(idx).copied().unwrap_or(1.0)
    }

    /// Sets the multiplier for a block.
    #[inline]
    pub fn set(&mut self, bx: usize, by: usize, value: f32) {
        let idx = by * self.width_blocks + bx;
        if idx < self.multipliers.len() {
            self.multipliers[idx] = value;
        }
    }
}

/// Computes simplified adaptive quantization map from image data.
///
/// ⚠️ THIS DOES NOT MATCH C++ JPEGLI.
///
/// This analyzes the image using activity/edge detection with
/// arbitrary thresholds. The C++ algorithm uses a completely
/// different approach (ComputePreErosion → FuzzyErosion → PerBlockModulations).
pub fn compute_simplified_quant_map(
    y_plane: &[f32],
    width: usize,
    height: usize,
    config: &SimplifiedAQConfig,
) -> SimplifiedQuantMap {
    let width_blocks = (width + 7) / 8;
    let height_blocks = (height + 7) / 8;

    if config.mode == SimplifiedAQMode::Off {
        return SimplifiedQuantMap::uniform(width_blocks, height_blocks);
    }

    let mut map = SimplifiedQuantMap {
        width_blocks,
        height_blocks,
        multipliers: Vec::with_capacity(width_blocks * height_blocks),
    };

    for by in 0..height_blocks {
        for bx in 0..width_blocks {
            let multiplier = compute_block_quant_multiplier(y_plane, width, height, bx, by, config);
            map.multipliers.push(multiplier);
        }
    }

    map
}

/// Computes the quantization multiplier for a single block.
fn compute_block_quant_multiplier(
    y_plane: &[f32],
    width: usize,
    height: usize,
    bx: usize,
    by: usize,
    config: &SimplifiedAQConfig,
) -> f32 {
    // Compute local variance (activity measure)
    let activity = compute_block_activity(y_plane, width, height, bx, by);

    // Low activity = can quantize more
    // High activity = need to preserve detail
    let activity_factor = activity_to_multiplier(activity, config);

    // Apply edge detection penalty
    let edge_factor = compute_edge_factor(y_plane, width, height, bx, by);

    // Combine factors
    let multiplier = activity_factor * edge_factor;

    // Clamp to reasonable range
    multiplier.clamp(0.5, 2.0)
}

/// Computes the activity (variance) of a block.
fn compute_block_activity(
    y_plane: &[f32],
    width: usize,
    height: usize,
    bx: usize,
    by: usize,
) -> f32 {
    let mut sum = 0.0f32;
    let mut sum_sq = 0.0f32;
    let mut count = 0;

    for y in 0..8 {
        for x in 0..8 {
            let px = bx * 8 + x;
            let py = by * 8 + y;

            if px < width && py < height {
                let val = y_plane[py * width + px];
                sum += val;
                sum_sq += val * val;
                count += 1;
            }
        }
    }

    if count == 0 {
        return 0.0;
    }

    let mean = sum / count as f32;
    let variance = (sum_sq / count as f32) - (mean * mean);
    variance.max(0.0).sqrt()
}

/// Converts activity to a quantization multiplier.
fn activity_to_multiplier(activity: f32, config: &SimplifiedAQConfig) -> f32 {
    // Higher activity = lower multiplier (preserve detail)
    // Lower activity = higher multiplier (can quantize more)

    let threshold = match config.mode {
        SimplifiedAQMode::Off => return 1.0,
        SimplifiedAQMode::Standard => 10.0,
        SimplifiedAQMode::Aggressive => 5.0,
    };

    if activity < threshold {
        // Low activity - can quantize more
        1.2
    } else if activity > threshold * 3.0 {
        // High activity - preserve detail
        0.8
    } else {
        // Linear interpolation
        let t = (activity - threshold) / (threshold * 2.0);
        1.2 - 0.4 * t
    }
}

/// Computes an edge detection factor for a block.
fn compute_edge_factor(y_plane: &[f32], width: usize, height: usize, bx: usize, by: usize) -> f32 {
    // Simple edge detection using horizontal and vertical gradients
    let mut max_gradient = 0.0f32;

    for y in 0..7 {
        for x in 0..7 {
            let px = bx * 8 + x;
            let py = by * 8 + y;

            if px + 1 < width && py + 1 < height {
                let center = y_plane[py * width + px];
                let right = y_plane[py * width + px + 1];
                let bottom = y_plane[(py + 1) * width + px];

                let grad_h = (right - center).abs();
                let grad_v = (bottom - center).abs();
                let grad = (grad_h * grad_h + grad_v * grad_v).sqrt();

                max_gradient = max_gradient.max(grad);
            }
        }
    }

    // Strong edges need preservation
    if max_gradient > 50.0 {
        0.9
    } else {
        1.0
    }
}

/// Applies simplified adaptive quantization to a quantization table.
///
/// ⚠️ This uses arbitrary frequency bands, NOT C++ logic.
#[must_use]
pub fn apply_simplified_adaptive_quant(
    base_quant: &[u16; DCT_BLOCK_SIZE],
    multiplier: f32,
    config: &SimplifiedAQConfig,
) -> [u16; DCT_BLOCK_SIZE] {
    let mut result = [0u16; DCT_BLOCK_SIZE];

    for i in 0..DCT_BLOCK_SIZE {
        let factor = if i == 0 {
            // DC coefficient
            multiplier * config.dc_quant_mul
        } else if i < 10 {
            // Low frequency AC
            multiplier * config.ac_quant_mul_low
        } else {
            // High frequency AC
            multiplier * config.ac_quant_mul_high
        };

        let q = (base_quant[i] as f32 * factor).round();
        result[i] = (q as u16).clamp(1, 255);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_quant_map() {
        let map = SimplifiedQuantMap::uniform(10, 10);
        assert_eq!(map.multipliers.len(), 100);
        assert!(map.multipliers.iter().all(|&m| (m - 1.0).abs() < 0.001));
    }

    #[test]
    fn test_quant_map_get_set() {
        let mut map = SimplifiedQuantMap::uniform(10, 10);
        map.set(5, 5, 1.5);
        assert!((map.get(5, 5) - 1.5).abs() < 0.001);
        assert!((map.get(0, 0) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_compute_block_activity() {
        // Constant block should have zero activity
        let plane = vec![128.0f32; 64];
        let activity = compute_block_activity(&plane, 8, 8, 0, 0);
        assert!(activity < 0.01);

        // Varying block should have non-zero activity
        let mut plane = vec![0.0f32; 64];
        for i in 0..64 {
            plane[i] = (i * 4) as f32;
        }
        let activity = compute_block_activity(&plane, 8, 8, 0, 0);
        assert!(activity > 1.0);
    }

    #[test]
    fn test_apply_adaptive_quant() {
        let base = [16u16; DCT_BLOCK_SIZE];
        let config = SimplifiedAQConfig::default();

        let result = apply_simplified_adaptive_quant(&base, 1.0, &config);
        assert_eq!(result, base);

        let result = apply_simplified_adaptive_quant(&base, 2.0, &config);
        assert!(result[0] > base[0]);
    }
}
