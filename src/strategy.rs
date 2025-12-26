//! Encoding strategy selection
//!
//! Automatically selects the best encoding approach based on target quality
//! and image characteristics.

use crate::analysis::{analyze_image, ImageAnalysis, RecommendedApproach};
use crate::types::{AdaptiveQuantConfig, EncodingStrategy, Quality, TrellisConfig};

/// Selected encoding configuration
#[derive(Debug, Clone)]
pub struct SelectedStrategy {
    /// Primary encoding approach
    pub approach: EncodingApproach,
    /// Trellis quantization settings
    pub trellis: TrellisConfig,
    /// Adaptive quantization settings
    pub adaptive_quant: AdaptiveQuantConfig,
    /// Whether to use progressive encoding
    pub progressive: bool,
    /// Whether to optimize Huffman tables
    pub optimize_huffman: bool,
}

/// Primary encoding approach
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodingApproach {
    /// mozjpeg-style: trellis + progressive + Huffman optimization
    Mozjpeg,
    /// jpegli-style: adaptive quantization + perceptual optimization
    Jpegli,
    /// Hybrid: use both and pick best result for each block/region
    Hybrid,
}

/// Select the best encoding strategy for given quality target
pub fn select_strategy(quality: &Quality, strategy: EncodingStrategy) -> SelectedStrategy {
    match strategy {
        EncodingStrategy::Mozjpeg => mozjpeg_strategy(quality),
        EncodingStrategy::Jpegli => jpegli_strategy(quality),
        EncodingStrategy::Hybrid => hybrid_strategy(quality),
        EncodingStrategy::Auto => auto_strategy(quality),
        EncodingStrategy::Simple => simple_strategy(quality),
    }
}

/// Select the best encoding strategy based on image analysis
///
/// This is the smart auto-pick that analyzes the image to decide
/// between trellis (mozjpeg) and adaptive quantization (jpegli).
pub fn select_strategy_for_image(
    quality: &Quality,
    strategy: EncodingStrategy,
    pixels: &[u8],
    width: usize,
    height: usize,
) -> (SelectedStrategy, ImageAnalysis) {
    let q = quality.value();
    let analysis = analyze_image(pixels, width, height, q);

    let selected = match strategy {
        EncodingStrategy::Mozjpeg => mozjpeg_strategy(quality),
        EncodingStrategy::Jpegli => jpegli_strategy(quality),
        EncodingStrategy::Hybrid => hybrid_strategy(quality),
        EncodingStrategy::Auto => strategy_from_analysis(quality, &analysis),
        EncodingStrategy::Simple => simple_strategy(quality),
    };

    (selected, analysis)
}

/// Build strategy from image analysis
fn strategy_from_analysis(quality: &Quality, analysis: &ImageAnalysis) -> SelectedStrategy {
    let q = quality.value();

    match analysis.recommended_approach {
        RecommendedApproach::Trellis => {
            // Use mozjpeg-style with analysis-tuned parameters
            SelectedStrategy {
                approach: EncodingApproach::Mozjpeg,
                trellis: compute_trellis_config_for_quality(q, true),
                adaptive_quant: AdaptiveQuantConfig {
                    enabled: false,
                    strength: 0.0,
                },
                progressive: q < 90.0,
                optimize_huffman: true,
            }
        }
        RecommendedApproach::AdaptiveQuant => {
            // Use jpegli-style
            SelectedStrategy {
                approach: EncodingApproach::Jpegli,
                trellis: TrellisConfig::disabled(),
                adaptive_quant: AdaptiveQuantConfig {
                    enabled: true,
                    strength: compute_aq_strength_from_analysis(analysis),
                },
                progressive: false,
                optimize_huffman: true,
            }
        }
        RecommendedApproach::Hybrid => {
            // Use both with balanced settings
            SelectedStrategy {
                approach: EncodingApproach::Hybrid,
                trellis: compute_trellis_config_for_quality(q, false),
                adaptive_quant: AdaptiveQuantConfig {
                    enabled: true,
                    strength: compute_aq_strength_from_analysis(analysis) * 0.7,
                },
                progressive: q < 80.0,
                optimize_huffman: true,
            }
        }
    }
}

/// Compute AQ strength based on image analysis
fn compute_aq_strength_from_analysis(analysis: &ImageAnalysis) -> f32 {
    // Higher complexity = higher AQ strength
    let base = 0.5 + analysis.luma_complexity * 0.5;

    // More edges = slightly higher strength
    let edge_boost = analysis.edge_density * 0.2;

    (base + edge_boost).min(1.0)
}

/// mozjpeg-style encoding (best for low quality)
fn mozjpeg_strategy(quality: &Quality) -> SelectedStrategy {
    let q = quality.value();
    SelectedStrategy {
        approach: EncodingApproach::Mozjpeg,
        trellis: compute_trellis_config_for_quality(q, true),
        adaptive_quant: AdaptiveQuantConfig {
            enabled: false,
            strength: 0.0,
        },
        progressive: q < 90.0,
        optimize_huffman: true,
    }
}

/// jpegli-style encoding (best for high quality)
fn jpegli_strategy(quality: &Quality) -> SelectedStrategy {
    let q = quality.value();
    SelectedStrategy {
        approach: EncodingApproach::Jpegli,
        trellis: compute_trellis_config_for_quality(q, false),
        // AQ disabled - the simplified variance-based implementation hurts quality
        // TODO: Port jpegli's perceptual AQ algorithm for proper high-quality optimization
        adaptive_quant: AdaptiveQuantConfig {
            enabled: false,
            strength: compute_aq_strength_for_quality(quality),
        },
        progressive: false, // jpegli uses sequential by default
        optimize_huffman: true,
    }
}

/// Hybrid encoding (try both, use best)
fn hybrid_strategy(quality: &Quality) -> SelectedStrategy {
    let q = quality.value();
    SelectedStrategy {
        approach: EncodingApproach::Hybrid,
        trellis: compute_trellis_config_for_quality(q, false),
        // TODO: Re-enable AQ once properly tuned
        adaptive_quant: AdaptiveQuantConfig {
            enabled: false,
            strength: compute_aq_strength_for_quality(quality) * 0.7,
        },
        progressive: q < 80.0,
        optimize_huffman: true,
    }
}

/// Auto-select based on quality
fn auto_strategy(quality: &Quality) -> SelectedStrategy {
    let q = quality.value();

    if q < 50.0 {
        // Low quality: mozjpeg wins
        mozjpeg_strategy(quality)
    } else if q < 70.0 {
        // Medium quality: hybrid is often best
        hybrid_strategy(quality)
    } else {
        // High quality: jpegli wins
        jpegli_strategy(quality)
    }
}

/// Simple baseline encoding (no trellis, no optimization)
fn simple_strategy(_quality: &Quality) -> SelectedStrategy {
    SelectedStrategy {
        approach: EncodingApproach::Mozjpeg,
        trellis: TrellisConfig::disabled(),
        adaptive_quant: AdaptiveQuantConfig {
            enabled: false,
            strength: 0.0,
        },
        progressive: false,
        optimize_huffman: false,
    }
}

/// Compute trellis config based on quality
///
/// The lambda parameters control the rate-distortion tradeoff:
/// - lambda_log_scale1: Base lambda (higher = PRESERVE quality, lower = favor smaller files)
/// - lambda_log_scale2: Normalization factor (higher = less adaptive to block content)
///
/// Cost = Rate + Lambda * Distortion
/// Higher lambda means distortion (quality loss) is penalized more -> preserve quality
/// Lower lambda means rate (file size) matters more -> aggressive compression
///
/// mozjpeg defaults are (14.75, 16.5), which we use for all quality levels.
fn compute_trellis_config_for_quality(q: f32, aggressive: bool) -> TrellisConfig {
    // Use mozjpeg defaults for all quality levels
    // The quantization table itself controls quality; trellis just optimizes at that level
    TrellisConfig {
        ac_enabled: true,
        dc_enabled: aggressive && q < 50.0, // DC trellis only at low quality
        eob_opt: true,
        lambda_log_scale1: 14.75, // mozjpeg default
        lambda_log_scale2: 16.5,  // mozjpeg default
    }
}

/// Compute AQ strength based on quality
fn compute_aq_strength_for_quality(quality: &Quality) -> f32 {
    let q = quality.value();

    // Higher quality = higher AQ strength (more content-aware)
    if q >= 90.0 {
        1.0
    } else if q >= 80.0 {
        0.8
    } else if q >= 70.0 {
        0.6
    } else {
        0.4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_strategy_low_quality() {
        let strategy = select_strategy(&Quality::Standard(30), EncodingStrategy::Auto);
        assert_eq!(strategy.approach, EncodingApproach::Mozjpeg);
        assert!(strategy.trellis.ac_enabled);
        assert!(strategy.trellis.dc_enabled); // DC trellis enabled for low quality mozjpeg
    }

    #[test]
    fn test_auto_strategy_high_quality() {
        let strategy = select_strategy(&Quality::Standard(90), EncodingStrategy::Auto);
        assert_eq!(strategy.approach, EncodingApproach::Jpegli);
        // AQ disabled - simplified variance-based version hurts quality
        assert!(!strategy.adaptive_quant.enabled);
        // Trellis still enabled at high quality but with conservative params
        assert!(strategy.trellis.ac_enabled);
        assert!(!strategy.trellis.dc_enabled); // DC trellis disabled at high quality
    }

    #[test]
    fn test_hybrid_strategy() {
        let strategy = select_strategy(&Quality::Standard(60), EncodingStrategy::Hybrid);
        assert_eq!(strategy.approach, EncodingApproach::Hybrid);
        assert!(strategy.trellis.ac_enabled);
        // AQ currently disabled until properly tuned
        assert!(!strategy.adaptive_quant.enabled);
    }

    #[test]
    fn test_lambda_matches_mozjpeg_defaults() {
        // All quality levels should use mozjpeg's default lambda values
        let low = compute_trellis_config_for_quality(30.0, true);
        let high = compute_trellis_config_for_quality(90.0, false);
        // Both should use the same mozjpeg defaults
        assert!((low.lambda_log_scale1 - 14.75).abs() < 0.01);
        assert!((low.lambda_log_scale2 - 16.5).abs() < 0.01);
        assert!((high.lambda_log_scale1 - 14.75).abs() < 0.01);
        assert!((high.lambda_log_scale2 - 16.5).abs() < 0.01);
    }
}
