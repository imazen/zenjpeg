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
    /// Recommended encoding approach based on analysis
    pub recommended_approach: RecommendedApproach,
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

    // Compute luma complexity
    let luma_complexity = compute_luma_complexity(pixels, width, height);

    // Compute edge density
    let edge_density = compute_edge_density(pixels, width, height);

    // Decide recommended approach
    let recommended_approach = decide_approach(quality, &chroma, luma_complexity, edge_density);

    ImageAnalysis {
        chroma,
        luma_complexity,
        edge_density,
        recommended_approach,
    }
}

/// Compute luma complexity (variance-based)
fn compute_luma_complexity(pixels: &[u8], width: usize, height: usize) -> f32 {
    if width < 8 || height < 8 {
        return 0.5; // Default for tiny images
    }

    // Sample blocks across the image
    let block_size = 8;
    let blocks_x = width / block_size;
    let blocks_y = height / block_size;

    if blocks_x == 0 || blocks_y == 0 {
        return 0.5;
    }

    let mut total_variance = 0.0f64;
    let mut block_count = 0;

    // Sample every 4th block for speed
    for by in (0..blocks_y).step_by(4) {
        for bx in (0..blocks_x).step_by(4) {
            let variance = compute_block_variance(pixels, width, bx * block_size, by * block_size);
            total_variance += variance as f64;
            block_count += 1;
        }
    }

    if block_count == 0 {
        return 0.5;
    }

    // Normalize: variance 0 = flat, variance ~2000 = highly detailed
    let avg_variance = total_variance / block_count as f64;
    (avg_variance / 2000.0).min(1.0) as f32
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

/// Compute edge density using simple Sobel-like detection
fn compute_edge_density(pixels: &[u8], width: usize, height: usize) -> f32 {
    if width < 3 || height < 3 {
        return 0.5;
    }

    let mut edge_count = 0u32;
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
                let center = luma_approx(&pixels[center_idx..center_idx + 3]);
                let left = luma_approx(&pixels[left_idx..left_idx + 3]);
                let right = luma_approx(&pixels[right_idx..right_idx + 3]);
                let top = luma_approx(&pixels[top_idx..top_idx + 3]);
                let bottom = luma_approx(&pixels[bottom_idx..bottom_idx + 3]);

                let gx = (right as i32 - left as i32).abs();
                let gy = (bottom as i32 - top as i32).abs();
                let gradient = gx + gy;

                // Threshold for "edge"
                if gradient > 30 {
                    edge_count += 1;
                }
                total_pixels += 1;
            }
        }
    }

    if total_pixels == 0 {
        return 0.5;
    }

    (edge_count as f32 / total_pixels as f32).min(1.0)
}

#[inline]
fn luma_approx(rgb: &[u8]) -> u8 {
    ((rgb[0] as u16 + 2 * rgb[1] as u16 + rgb[2] as u16) / 4) as u8
}

/// Decide encoding approach based on analysis
fn decide_approach(
    quality: f32,
    chroma: &ChromaEvaluation,
    luma_complexity: f32,
    edge_density: f32,
) -> RecommendedApproach {
    // Get sharpness info
    let sharpness = chroma.sharpness.unwrap_or(Sharpness {
        horiz: 0,
        vert: 0,
        peak: 0,
    });
    let max_sharpness = sharpness.horiz.max(sharpness.vert);

    // Decision matrix:
    //
    // Quality < 50: Always trellis (best for low bitrate)
    // Quality >= 80: Usually AQ (best for high quality)
    // Quality 50-80: Depends on image characteristics

    if quality < 50.0 {
        // Low quality: trellis always wins
        return RecommendedApproach::Trellis;
    }

    if quality >= 80.0 {
        // High quality: AQ usually wins, unless image is very simple
        if luma_complexity < 0.2 && edge_density < 0.1 {
            // Very simple image (logos, text) - trellis still helps
            return RecommendedApproach::Trellis;
        }
        return RecommendedApproach::AdaptiveQuant;
    }

    // Medium quality (50-80): decide based on image characteristics

    // Complex images with many edges benefit from AQ
    if luma_complexity > 0.5 && edge_density > 0.3 {
        return RecommendedApproach::AdaptiveQuant;
    }

    // Simple images with few edges benefit from trellis
    if luma_complexity < 0.3 && edge_density < 0.2 {
        return RecommendedApproach::Trellis;
    }

    // Sharp chroma (red text, saturated colors) needs AQ to preserve detail
    if max_sharpness > 1000 {
        return RecommendedApproach::AdaptiveQuant;
    }

    // Default for medium zone: hybrid
    RecommendedApproach::Hybrid
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
    fn test_low_quality_always_trellis() {
        let pixels = vec![128u8; 64 * 64 * 3];
        let analysis = analyze_image(&pixels, 64, 64, 30.0);
        assert_eq!(analysis.recommended_approach, RecommendedApproach::Trellis);
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
