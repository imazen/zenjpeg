//! Encoding strategy selection
//!
//! Automatically selects the best encoding approach based on target quality
//! and image characteristics.

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
    }
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

/// Compute trellis config based on quality
///
/// The lambda parameters control the rate-distortion tradeoff:
/// - lambda_log_scale1: Base lambda (higher = favor smaller files)
/// - lambda_log_scale2: Normalization factor (higher = less adaptive to block content)
///
/// Note: mozjpeg defaults are (14.75, 16.5) but zenjpeg needs more conservative
/// values because our quantization pipeline differs from mozjpeg's.
fn compute_trellis_config_for_quality(q: f32, aggressive: bool) -> TrellisConfig {
    if q < 50.0 {
        // Low quality: more aggressive trellis
        TrellisConfig {
            ac_enabled: true,
            dc_enabled: aggressive,
            eob_opt: true,
            lambda_log_scale1: 14.5, // Conservative: mozjpeg uses ~14.75
            lambda_log_scale2: 16.5,
        }
    } else if q < 70.0 {
        // Medium quality: balanced trellis
        TrellisConfig {
            ac_enabled: true,
            dc_enabled: false,
            eob_opt: true,
            lambda_log_scale1: 13.5,
            lambda_log_scale2: 16.5,
        }
    } else if q < 80.0 {
        // High quality: conservative trellis
        TrellisConfig {
            ac_enabled: true,
            dc_enabled: false,
            eob_opt: false,
            lambda_log_scale1: 12.5, // Lower lambda = preserve more detail
            lambda_log_scale2: 16.5,
        }
    } else {
        // Very high quality: disable trellis - it hurts more than it helps
        // At Q>=80, rely on Huffman optimization only
        TrellisConfig {
            ac_enabled: false,
            dc_enabled: false,
            eob_opt: false,
            lambda_log_scale1: 11.5,
            lambda_log_scale2: 16.5,
        }
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
        // Trellis disabled at very high quality (Q>=80) where it hurts
        assert!(!strategy.trellis.ac_enabled);
        assert!(!strategy.trellis.dc_enabled);
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
    fn test_lambda_decreases_with_quality() {
        let low = compute_trellis_config_for_quality(30.0, true);
        let high = compute_trellis_config_for_quality(90.0, false);
        // Lower quality should have higher lambda
        assert!(low.lambda_log_scale1 > high.lambda_log_scale1);
    }
}
