//! Butteraugli perceptual image quality metric.
//!
//! This is a port of the butteraugli algorithm from libjxl.
//! Butteraugli is a psychovisual image similarity metric developed by Google.
//!
//! The metric is based on:
//! - Opsin: dynamics of photosensitive chemicals in the retina
//! - XYB: hybrid opponent/trichromatic color space
//! - Visual masking: how features hide other features
//! - Multi-scale analysis: HF, MF, LF, UHF components
//!
//! Reference: https://github.com/google/butteraugli
//!
//! TODO: This is a partial port. Full implementation requires:
//! - Gaussian blur convolution
//! - Multi-scale decomposition
//! - Masking functions
//! - Color difference computation

use crate::xyb::linear_rgb_to_xyb;

/// Butteraugli comparison parameters.
#[derive(Debug, Clone)]
pub struct ButteraugliParams {
    /// Multiplier for penalizing new HF artifacts more than blurring.
    /// 1.0 = neutral.
    pub hf_asymmetry: f32,
    /// Multiplier for psychovisual difference in X channel.
    pub xmul: f32,
    /// Number of nits corresponding to 1.0 input values.
    pub intensity_target: f32,
}

impl Default for ButteraugliParams {
    fn default() -> Self {
        Self {
            hf_asymmetry: 1.0,
            xmul: 1.0,
            intensity_target: 80.0,
        }
    }
}

// Butteraugli constants from C++ implementation
const W_MF_MALTA: f64 = 37.0819870399;
const NORM1_MF: f64 = 130262059.556;
const W_MF_MALTA_X: f64 = 8246.75321353;
const NORM1_MF_X: f64 = 1009002.70582;
const W_HF_MALTA: f64 = 18.7237414387;
const NORM1_HF: f64 = 4498534.45232;
const W_HF_MALTA_X: f64 = 6923.99476109;
const NORM1_HF_X: f64 = 8051.15833247;
const W_UHF_MALTA: f64 = 1.10039032555;
const NORM1_UHF: f64 = 71.7800275169;
const W_UHF_MALTA_X: f64 = 173.5;
const NORM1_UHF_X: f64 = 5.0;

#[allow(dead_code)]
const WMUL: [f64; 9] = [
    400.0, 1.50815703118, 0.0,
    2150.0, 10.6195433239, 16.2176043152,
    29.2353797994, 0.844626970982, 0.703646627719,
];

/// Quality threshold for "good" (images look the same).
pub const BUTTERAUGLI_GOOD: f64 = 1.0;

/// Quality threshold for "bad" (visible difference).
pub const BUTTERAUGLI_BAD: f64 = 2.0;

/// Computes a Gaussian blur kernel for the given sigma.
pub fn compute_kernel(sigma: f32) -> Vec<f32> {
    let m = 2.25f32; // Accuracy increases when m is increased
    let scaler = -1.0 / (2.0 * sigma * sigma);
    let diff = (m * sigma.abs()).max(1.0) as i32;
    let mut kernel = vec![0.0f32; (2 * diff + 1) as usize];

    for i in -diff..=diff {
        kernel[(i + diff) as usize] = (scaler * (i * i) as f32).exp();
    }

    kernel
}

/// Converts sRGB to linear RGB.
fn srgb_to_linear(v: f32) -> f32 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// Converts an RGB image buffer to XYB planes.
pub fn rgb_to_xyb_planes(
    rgb: &[u8],
    width: usize,
    height: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let num_pixels = width * height;
    let mut x_plane = vec![0.0f32; num_pixels];
    let mut y_plane = vec![0.0f32; num_pixels];
    let mut b_plane = vec![0.0f32; num_pixels];

    for i in 0..num_pixels {
        let r = srgb_to_linear(rgb[i * 3] as f32 / 255.0);
        let g = srgb_to_linear(rgb[i * 3 + 1] as f32 / 255.0);
        let b = srgb_to_linear(rgb[i * 3 + 2] as f32 / 255.0);

        let (x, y, b_val) = linear_rgb_to_xyb(r, g, b);
        x_plane[i] = x;
        y_plane[i] = y;
        b_plane[i] = b_val;
    }

    (x_plane, y_plane, b_plane)
}

/// Butteraugli image comparison result.
#[derive(Debug, Clone)]
pub struct ButteraugliResult {
    /// Global difference score. < 1.0 is "good", > 2.0 is "bad".
    pub score: f64,
    /// Per-pixel difference map (optional).
    pub diffmap: Option<Vec<f32>>,
}

/// Computes butteraugli score between two RGB images.
///
/// # Arguments
/// * `rgb1` - First image (sRGB u8, 3 bytes per pixel)
/// * `rgb2` - Second image (sRGB u8, 3 bytes per pixel)
/// * `width` - Image width
/// * `height` - Image height
/// * `params` - Comparison parameters
///
/// # Returns
/// Butteraugli score and optional per-pixel difference map.
///
/// # Note
/// This is a simplified implementation. Full butteraugli requires:
/// - Multi-scale Gaussian blur
/// - Masking computation
/// - HF/MF/LF/UHF decomposition
///
/// For now, we use a simplified XYB difference metric.
pub fn compute_butteraugli(
    rgb1: &[u8],
    rgb2: &[u8],
    width: usize,
    height: usize,
    params: &ButteraugliParams,
) -> ButteraugliResult {
    assert_eq!(rgb1.len(), width * height * 3);
    assert_eq!(rgb2.len(), width * height * 3);

    // Convert to XYB
    let (x1, y1, b1) = rgb_to_xyb_planes(rgb1, width, height);
    let (x2, y2, b2) = rgb_to_xyb_planes(rgb2, width, height);

    // Compute per-pixel XYB difference (simplified)
    let num_pixels = width * height;
    let mut diffmap = vec![0.0f32; num_pixels];
    let mut total_diff = 0.0f64;

    for i in 0..num_pixels {
        let dx = (x1[i] - x2[i]) * params.xmul;
        let dy = y1[i] - y2[i];
        let db = b1[i] - b2[i];

        // Simplified perceptual difference (full butteraugli uses masking)
        let diff = (dx * dx + dy * dy * 4.0 + db * db * 0.5).sqrt();
        diffmap[i] = diff;
        total_diff += diff as f64;
    }

    // Normalize to butteraugli-like scale
    // This is a rough approximation - real butteraugli uses complex pooling
    let avg_diff = total_diff / num_pixels as f64;
    let score = avg_diff * 100.0; // Scale factor to match butteraugli range

    ButteraugliResult {
        score,
        diffmap: Some(diffmap),
    }
}

/// Converts butteraugli score to quality percentage (0-100).
pub fn score_to_quality(score: f64) -> f64 {
    // Approximate mapping: score < 1.0 = 100%, score > 4.0 = 0%
    (100.0 - score * 25.0).clamp(0.0, 100.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_images() {
        let width = 16;
        let height = 16;
        let rgb: Vec<u8> = (0..width * height * 3).map(|i| (i % 256) as u8).collect();

        let result = compute_butteraugli(&rgb, &rgb, width, height, &ButteraugliParams::default());

        // Identical images should have score 0
        assert!(result.score < 0.001, "Identical images should have score ~0, got {}", result.score);
    }

    #[test]
    fn test_different_images() {
        let width = 16;
        let height = 16;
        let rgb1: Vec<u8> = vec![0; width * height * 3];
        let rgb2: Vec<u8> = vec![255; width * height * 3];

        let result = compute_butteraugli(&rgb1, &rgb2, width, height, &ButteraugliParams::default());

        // Very different images should have high score
        assert!(result.score > 1.0, "Very different images should have score > 1, got {}", result.score);
    }

    #[test]
    fn test_kernel_generation() {
        let kernel = compute_kernel(1.0);
        assert!(!kernel.is_empty());
        assert_eq!(kernel.len() % 2, 1); // Should be odd

        // Center should be maximum
        let center = kernel.len() / 2;
        for (i, &v) in kernel.iter().enumerate() {
            if i != center {
                assert!(v < kernel[center]);
            }
        }
    }
}
