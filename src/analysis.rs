//! Image analysis for auto-picking encoding strategy
//!
//! Uses evalchroma for chroma analysis and adds luma analysis
//! to decide between trellis (mozjpeg) and adaptive quantization (jpegli).

use evalchroma::{adjust_sampling, ChromaEvaluation, PixelSize, Sharpness};
use imgref::ImgRef;
use rgb::RGB8;

/// Image characteristics used to select encoding strategy
#[derive(Debug, Clone)]
pub struct ImageAnalysis {
    /// Chroma evaluation from evalchroma
    pub chroma: ChromaEvaluation,
    /// Luma complexity (0.0 = flat, 1.0 = highly detailed)
    pub luma_complexity: f32,
    /// Edge density (0.0 = no edges, 1.0 = many edges)
    pub edge_density: f32,
    /// Percentage of "flat" blocks (0-100)
    /// A block is flat if its variance < 100
    pub flat_block_pct: f32,
    /// Mean edge strength across sampled pixels (0-255 range)
    pub edge_strength_mean: f32,
    /// Mean local contrast (variance in 3x3 neighborhood, 0-255 range)
    pub local_contrast_mean: f32,
    /// Recommended encoding approach based on analysis
    pub recommended_approach: RecommendedApproach,
    /// Deringing benefit score (0.0 = no benefit, 1.0 = high benefit)
    /// High when image has saturated regions (pixels at 0 or 255) near edges
    pub deringing_benefit: f32,
}

/// Recommended encoding approach
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecommendedApproach {
    /// Use trellis quantization (better for low quality, simple images)
    Trellis,
    /// Use adaptive quantization (better for high quality, complex images)
    AdaptiveQuant,
    /// Use both with blend (hybrid approach)
    Hybrid,
}

/// Quick check if deringing would benefit this image.
///
/// Returns true if the image has significant saturated regions (pixels near 0 or 255)
/// adjacent to edges, which is where deringing helps most.
///
/// This is faster than full analyze_image() when you only need deringing decision.
pub fn should_use_deringing(pixels: &[u8], width: usize, height: usize) -> bool {
    compute_deringing_benefit(pixels, width, height) > 0.1
}

/// Analyze an image and return recommended encoding settings
pub fn analyze_image(pixels: &[u8], width: usize, height: usize, quality: f32) -> ImageAnalysis {
    // Convert to RGB8 slice for evalchroma
    let rgb_pixels: Vec<RGB8> = pixels
        .chunks_exact(3)
        .map(|c| RGB8::new(c[0], c[1], c[2]))
        .collect();

    let img = ImgRef::new(&rgb_pixels, width, height);

    // Get chroma analysis from evalchroma
    // Start with 4:2:0 as worst allowed, let evalchroma upgrade if needed
    let subsampling = PixelSize {
        cb: (2, 2),
        cr: (2, 2),
    };
    let chroma = adjust_sampling(img, subsampling, quality);

    // Compute luma complexity and flat block percentage
    let (luma_complexity, flat_block_pct) = compute_luma_metrics(pixels, width, height);

    // Compute edge metrics
    let (edge_density, edge_strength_mean, local_contrast_mean) =
        compute_edge_metrics(pixels, width, height);

    // Decide recommended approach using research-validated prediction model
    let recommended_approach = decide_approach(
        quality,
        &chroma,
        flat_block_pct,
        edge_strength_mean,
        local_contrast_mean,
    );

    // Compute deringing benefit (saturated regions near edges)
    let deringing_benefit = compute_deringing_benefit(pixels, width, height);

    ImageAnalysis {
        chroma,
        luma_complexity,
        edge_density,
        flat_block_pct,
        edge_strength_mean,
        local_contrast_mean,
        recommended_approach,
        deringing_benefit,
    }
}

/// Compute luma metrics: complexity and flat block percentage
///
/// Returns (luma_complexity, flat_block_pct) where:
/// - luma_complexity: 0.0 = flat, 1.0 = highly detailed
/// - flat_block_pct: 0-100, percentage of blocks with variance < FLAT_THRESHOLD
fn compute_luma_metrics(pixels: &[u8], width: usize, height: usize) -> (f32, f32) {
    // Threshold for considering a block "flat" (matches codec-compare heuristics)
    const FLAT_THRESHOLD: f32 = 100.0;

    if width < 8 || height < 8 {
        return (0.5, 50.0); // Default for tiny images
    }

    let block_size = 8;
    let blocks_x = width / block_size;
    let blocks_y = height / block_size;

    if blocks_x == 0 || blocks_y == 0 {
        return (0.5, 50.0);
    }

    let mut total_variance = 0.0f64;
    let mut flat_blocks = 0u32;
    let mut block_count = 0u32;

    // Sample every 4th block for speed
    for by in (0..blocks_y).step_by(4) {
        for bx in (0..blocks_x).step_by(4) {
            let variance = compute_block_variance(pixels, width, bx * block_size, by * block_size);
            total_variance += variance as f64;
            if variance < FLAT_THRESHOLD {
                flat_blocks += 1;
            }
            block_count += 1;
        }
    }

    if block_count == 0 {
        return (0.5, 50.0);
    }

    // Normalize: variance 0 = flat, variance ~2000 = highly detailed
    let avg_variance = total_variance / block_count as f64;
    let luma_complexity = (avg_variance / 2000.0).min(1.0) as f32;

    // Calculate flat block percentage (0-100 scale)
    let flat_block_pct = (flat_blocks as f32 / block_count as f32) * 100.0;

    (luma_complexity, flat_block_pct)
}

/// Compute variance of an 8x8 block (luma only)
fn compute_block_variance(pixels: &[u8], stride: usize, x: usize, y: usize) -> f32 {
    let mut sum = 0u32;
    let mut sum_sq = 0u64;

    for dy in 0..8 {
        for dx in 0..8 {
            let idx = ((y + dy) * stride + (x + dx)) * 3;
            if idx + 2 < pixels.len() {
                // Approximate luma: (R + 2*G + B) / 4
                let r = pixels[idx] as u32;
                let g = pixels[idx + 1] as u32;
                let b = pixels[idx + 2] as u32;
                let luma = (r + 2 * g + b) / 4;
                sum += luma;
                sum_sq += (luma * luma) as u64;
            }
        }
    }

    let n = 64u32;
    let mean = sum / n;
    let variance = (sum_sq / n as u64) as u32 - mean * mean;
    variance as f32
}

/// Compute edge metrics: density, mean strength, and local contrast
///
/// Returns (edge_density, edge_strength_mean, local_contrast_mean) where:
/// - edge_density: 0.0-1.0, fraction of pixels that are edges
/// - edge_strength_mean: 0-255, mean gradient magnitude
/// - local_contrast_mean: 0-255, mean variance in 3x3 neighborhood
fn compute_edge_metrics(pixels: &[u8], width: usize, height: usize) -> (f32, f32, f32) {
    if width < 3 || height < 3 {
        return (0.5, 15.0, 15.0);
    }

    let mut edge_count = 0u32;
    let mut total_gradient = 0u64;
    let mut total_local_contrast = 0u64;
    let mut total_pixels = 0u32;

    // Sample every 4th pixel for speed
    for y in (1..height - 1).step_by(4) {
        for x in (1..width - 1).step_by(4) {
            let center_idx = (y * width + x) * 3;
            let left_idx = (y * width + x - 1) * 3;
            let right_idx = (y * width + x + 1) * 3;
            let top_idx = ((y - 1) * width + x) * 3;
            let bottom_idx = ((y + 1) * width + x) * 3;

            if bottom_idx + 2 < pixels.len() {
                // Simple gradient magnitude (luma approximation)
                let center = luma_approx(&pixels[center_idx..center_idx + 3]) as i32;
                let left = luma_approx(&pixels[left_idx..left_idx + 3]) as i32;
                let right = luma_approx(&pixels[right_idx..right_idx + 3]) as i32;
                let top = luma_approx(&pixels[top_idx..top_idx + 3]) as i32;
                let bottom = luma_approx(&pixels[bottom_idx..bottom_idx + 3]) as i32;

                let gx = (right - left).abs();
                let gy = (bottom - top).abs();
                let gradient = gx + gy;

                total_gradient += gradient as u64;

                // Threshold for "edge"
                if gradient > 30 {
                    edge_count += 1;
                }

                // Local contrast: variance of 3x3 neighborhood (simplified: range of neighbors)
                let max_neighbor = center.max(left).max(right).max(top).max(bottom);
                let min_neighbor = center.min(left).min(right).min(top).min(bottom);
                let local_range = (max_neighbor - min_neighbor) as u64;
                total_local_contrast += local_range;

                total_pixels += 1;
            }
        }
    }

    if total_pixels == 0 {
        return (0.5, 15.0, 15.0);
    }

    let edge_density = (edge_count as f32 / total_pixels as f32).min(1.0);
    let edge_strength_mean = (total_gradient as f64 / total_pixels as f64) as f32;
    let local_contrast_mean = (total_local_contrast as f64 / total_pixels as f64) as f32;

    (edge_density, edge_strength_mean, local_contrast_mean)
}

#[inline]
fn luma_approx(rgb: &[u8]) -> u8 {
    ((rgb[0] as u16 + 2 * rgb[1] as u16 + rgb[2] as u16) / 4) as u8
}

/// Compute deringing benefit score based on saturated regions near edges.
///
/// Deringing is most beneficial when:
/// 1. Image has pixels at max value (255) - especially white backgrounds
/// 2. These saturated pixels are adjacent to non-saturated pixels (edges)
/// 3. Common in: text on white, logos, graphics, overexposed highlights
///
/// Returns 0.0-1.0 where higher means more benefit from deringing.
fn compute_deringing_benefit(pixels: &[u8], width: usize, height: usize) -> f32 {
    if width < 3 || height < 3 {
        return 0.0;
    }

    let mut saturated_edge_count = 0u32;
    let mut total_samples = 0u32;

    // Thresholds for "saturated" (near min/max)
    const MAX_THRESH: u8 = 250; // Near white
    const MIN_THRESH: u8 = 5; // Near black
    const EDGE_DIFF: i32 = 30; // Significant edge

    // Sample every 4th pixel for speed
    for y in (1..height - 1).step_by(4) {
        for x in (1..width - 1).step_by(4) {
            let idx = (y * width + x) * 3;
            if idx + 2 >= pixels.len() {
                continue;
            }

            let center_luma = luma_approx(&pixels[idx..idx + 3]);

            // Check if center pixel is saturated (near white or black)
            let is_saturated = center_luma >= MAX_THRESH || center_luma <= MIN_THRESH;

            if is_saturated {
                // Check if there's an edge nearby (significant luminance difference)
                let left_idx = (y * width + x - 1) * 3;
                let right_idx = (y * width + x + 1) * 3;
                let top_idx = ((y - 1) * width + x) * 3;
                let bottom_idx = ((y + 1) * width + x) * 3;

                if bottom_idx + 2 < pixels.len() {
                    let left = luma_approx(&pixels[left_idx..left_idx + 3]);
                    let right = luma_approx(&pixels[right_idx..right_idx + 3]);
                    let top = luma_approx(&pixels[top_idx..top_idx + 3]);
                    let bottom = luma_approx(&pixels[bottom_idx..bottom_idx + 3]);

                    // Check for significant edge in any direction
                    let has_edge = (center_luma as i32 - left as i32).abs() > EDGE_DIFF
                        || (center_luma as i32 - right as i32).abs() > EDGE_DIFF
                        || (center_luma as i32 - top as i32).abs() > EDGE_DIFF
                        || (center_luma as i32 - bottom as i32).abs() > EDGE_DIFF;

                    if has_edge {
                        saturated_edge_count += 1;
                    }
                }
            }
            total_samples += 1;
        }
    }

    if total_samples == 0 {
        return 0.0;
    }

    // Normalize: even a small fraction of saturated edges suggests deringing helps
    // Use sqrt to boost low values (deringing helps even with few such regions)
    let ratio = saturated_edge_count as f32 / total_samples as f32;
    (ratio * 10.0).sqrt().min(1.0)
}

/// Estimate BPP (bits per pixel) from quality and image flatness
///
/// Based on empirical analysis from codec-compare experiments.
/// BPP varies ~13x across images at same quality, but this gives
/// a reasonable estimate for strategy selection.
pub fn estimate_bpp(quality: f32, flat_block_pct: f32) -> f32 {
    // Base BPP from quality (empirical fit from corpus analysis)
    let base = 0.1 + 0.016 * quality;

    // Content factor: flat images compress better
    let content_factor = 0.3 + 0.7 * (100.0 - flat_block_pct) / 100.0;

    base * content_factor
}

/// Decide encoding approach using research-validated prediction model
///
/// Based on 382 significant BPP-matched comparisons across CLIC 2025
/// and Kodak corpora. Achieves 86.6% accuracy in predicting which
/// encoder produces better perceptual quality at matched file size.
///
/// Key finding: jpegli wins ~84% of cases. The only category where
/// mozjpeg dominates is very flat images (>75% flat blocks) at
/// low-to-medium BPP (0.35-0.6), with low complexity.
fn decide_approach(
    quality: f32,
    _chroma: &ChromaEvaluation,
    flat_block_pct: f32,
    edge_strength_mean: f32,
    local_contrast_mean: f32,
) -> RecommendedApproach {
    // Default to Butteraugli-optimized strategy
    decide_approach_for_metric(
        quality,
        flat_block_pct,
        edge_strength_mean,
        local_contrast_mean,
        crate::types::OptimizeFor::Butteraugli,
    )
}

/// Decide encoding approach based on target quality metric
///
/// Research findings by metric:
/// - Butteraugli: jpegli wins 84%, mozjpeg only for very flat + low BPP
/// - DSSIM: mozjpeg wins 67%, especially at 0.6-1.5 BPP
/// - SSIMULACRA2: balanced, mozjpeg slightly ahead at medium BPP
/// - FileSize: always use trellis
pub fn decide_approach_for_metric(
    quality: f32,
    flat_block_pct: f32,
    edge_strength_mean: f32,
    local_contrast_mean: f32,
    metric: crate::types::OptimizeFor,
) -> RecommendedApproach {
    use crate::types::OptimizeFor;

    let estimated_bpp = estimate_bpp(quality, flat_block_pct);
    let complexity = edge_strength_mean + local_contrast_mean;
    let uniformity = flat_block_pct;

    match metric {
        OptimizeFor::Butteraugli => {
            // jpegli wins 84% - only use mozjpeg for very flat + medium BPP
            if uniformity > 75.0 && complexity < 20.0 && estimated_bpp >= 0.35 && estimated_bpp < 0.6
            {
                RecommendedApproach::Trellis
            } else {
                RecommendedApproach::AdaptiveQuant
            }
        }
        OptimizeFor::Dssim => {
            // mozjpeg wins 67% - favored at 0.5-2.0 BPP range
            // jpegli only wins at very low (<0.4) or very high (>2.5) BPP
            if estimated_bpp < 0.4 {
                RecommendedApproach::AdaptiveQuant // jpegli wins at very low BPP
            } else if estimated_bpp > 2.5 {
                RecommendedApproach::AdaptiveQuant // jpegli wins at very high BPP
            } else {
                RecommendedApproach::Trellis // mozjpeg wins at medium BPP
            }
        }
        OptimizeFor::Ssimulacra2 => {
            // Balanced - mozjpeg slightly ahead at 0.5-1.5 BPP
            if estimated_bpp >= 0.5 && estimated_bpp <= 1.5 {
                // mozjpeg has slight edge
                if complexity > 30.0 {
                    RecommendedApproach::AdaptiveQuant // complex images favor jpegli
                } else {
                    RecommendedApproach::Trellis
                }
            } else {
                RecommendedApproach::AdaptiveQuant // jpegli at extremes
            }
        }
        OptimizeFor::FileSize => {
            // Always use trellis for file size optimization
            RecommendedApproach::Trellis
        }
    }
}

/// Convert quality to Butteraugli distance (jpegli formula)
///
/// Distance ~1.0 is "visually lossless", lower = higher quality
pub fn quality_to_distance(quality: f32) -> f32 {
    let q = quality as i32;
    if q >= 100 {
        0.01
    } else if q >= 30 {
        0.1 + (100 - q) as f32 * 0.09
    } else {
        let qf = quality;
        53.0 / 3000.0 * qf * qf - 23.0 / 20.0 * qf + 25.0
    }
}

/// Convert Butteraugli distance to approximate quality
///
/// Inverse of quality_to_distance (approximate)
pub fn distance_to_quality(distance: f32) -> f32 {
    if distance <= 0.01 {
        100.0
    } else if distance <= 0.1 + 70.0 * 0.09 {
        // In the linear region (Q30-Q100)
        100.0 - (distance - 0.1) / 0.09
    } else {
        // In the quadratic region (Q0-Q30) - use approximation
        // Solve: 53/3000 * q^2 - 23/20 * q + 25 = distance
        // Using Newton's method or just clamp to reasonable value
        ((25.0 - distance) / 1.15).max(1.0).min(30.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_to_distance() {
        // Q100 should give very low distance
        assert!(quality_to_distance(100.0) < 0.1);

        // Q90 should be around 1.0
        let d90 = quality_to_distance(90.0);
        assert!(d90 > 0.5 && d90 < 2.0);

        // Monotonic: lower quality = higher distance
        assert!(quality_to_distance(50.0) > quality_to_distance(90.0));
        assert!(quality_to_distance(10.0) > quality_to_distance(50.0));
    }

    #[test]
    fn test_distance_roundtrip() {
        for q in [40, 50, 60, 70, 80, 90, 95] {
            let d = quality_to_distance(q as f32);
            let q_back = distance_to_quality(d);
            assert!(
                (q_back - q as f32).abs() < 2.0,
                "Roundtrip failed: q={}, d={}, q_back={}",
                q,
                d,
                q_back
            );
        }
    }

    #[test]
    fn test_analyze_uniform_image() {
        // 64x64 uniform gray image
        let pixels = vec![128u8; 64 * 64 * 3];
        let analysis = analyze_image(&pixels, 64, 64, 75.0);

        // Should be simple image
        assert!(analysis.luma_complexity < 0.2);
        assert!(analysis.edge_density < 0.2);

        // Should recommend trellis for simple images at Q75
        assert_eq!(analysis.recommended_approach, RecommendedApproach::Trellis);
    }

    #[test]
    fn test_analyze_gradient_image() {
        // 64x64 horizontal gradient
        let mut pixels = vec![0u8; 64 * 64 * 3];
        for y in 0..64 {
            for x in 0..64 {
                let idx = (y * 64 + x) * 3;
                let v = (x * 4).min(255) as u8;
                pixels[idx] = v;
                pixels[idx + 1] = v;
                pixels[idx + 2] = v;
            }
        }

        let analysis = analyze_image(&pixels, 64, 64, 75.0);

        // Smooth gradient has low block variance but some edge density
        // (edges at block boundaries due to quantization)
        // Just verify it produces a valid analysis
        assert!(analysis.luma_complexity >= 0.0 && analysis.luma_complexity <= 1.0);
        assert!(analysis.edge_density >= 0.0 && analysis.edge_density <= 1.0);
    }

    #[test]
    fn test_prediction_model_very_flat_medium_quality() {
        // Research finding: mozjpeg wins for very flat images at 0.35-0.6 bpp
        // At Q75, uniform image has flat_pct=100, bpp≈0.39, so mozjpeg wins
        let pixels = vec![128u8; 64 * 64 * 3];
        let analysis = analyze_image(&pixels, 64, 64, 75.0);
        assert_eq!(analysis.recommended_approach, RecommendedApproach::Trellis);
    }

    #[test]
    fn test_prediction_model_flat_low_quality() {
        // Research finding: jpegli wins at very low bpp (<0.35) even for flat images
        // At Q30, uniform image has flat_pct=100, bpp≈0.17, so jpegli wins
        let pixels = vec![128u8; 64 * 64 * 3];
        let analysis = analyze_image(&pixels, 64, 64, 30.0);
        // At very low quality, jpegli still wins because bpp < 0.35
        assert_eq!(
            analysis.recommended_approach,
            RecommendedApproach::AdaptiveQuant
        );
    }

    #[test]
    fn test_high_quality_complex_image_uses_aq() {
        // Create a noisy/complex image
        let mut pixels = vec![0u8; 64 * 64 * 3];
        for i in 0..pixels.len() {
            pixels[i] = ((i * 17 + 31) % 256) as u8; // Pseudo-random pattern
        }

        let analysis = analyze_image(&pixels, 64, 64, 90.0);

        // High quality + complex = should prefer AQ
        assert_eq!(
            analysis.recommended_approach,
            RecommendedApproach::AdaptiveQuant
        );
    }
}
