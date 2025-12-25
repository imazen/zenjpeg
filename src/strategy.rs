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
    SelectedStrategy {
        approach: EncodingApproach::Mozjpeg,
        trellis: TrellisConfig {
            ac_enabled: true,
            dc_enabled: true,
            lambda_base: compute_lambda_for_quality(quality),
        },
        adaptive_quant: AdaptiveQuantConfig {
            enabled: false,
            strength: 0.0,
        },
        progressive: quality.value() < 90.0,
        optimize_huffman: true,
    }
}

/// jpegli-style encoding (best for high quality)
fn jpegli_strategy(quality: &Quality) -> SelectedStrategy {
    let q = quality.value();
    SelectedStrategy {
        approach: EncodingApproach::Jpegli,
        trellis: TrellisConfig {
            // Only enable trellis at lower quality where it helps most
            // Our simplified trellis is too aggressive at high quality
            ac_enabled: q < 85.0,
            dc_enabled: false,
            lambda_base: compute_lambda_for_quality(quality),
        },
        // TODO: Re-enable AQ once properly tuned
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
    // For hybrid, we enable trellis (AQ disabled until properly tuned)
    SelectedStrategy {
        approach: EncodingApproach::Hybrid,
        trellis: TrellisConfig {
            ac_enabled: true,
            dc_enabled: false, // DC trellis less effective with AQ
            lambda_base: compute_lambda_for_quality(quality) * 0.5,
        },
        // TODO: Re-enable AQ once properly tuned
        adaptive_quant: AdaptiveQuantConfig {
            enabled: false,
            strength: compute_aq_strength_for_quality(quality) * 0.7,
        },
        progressive: quality.value() < 80.0,
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

/// Compute lambda for trellis based on quality
fn compute_lambda_for_quality(quality: &Quality) -> f32 {
    let q = quality.value();

    // Lower quality = higher lambda (favor smaller files)
    if q < 50.0 {
        2.0
    } else if q < 70.0 {
        1.5
    } else if q < 85.0 {
        1.0
    } else {
        0.5
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
    }

    #[test]
    fn test_auto_strategy_high_quality() {
        let strategy = select_strategy(&Quality::Standard(90), EncodingStrategy::Auto);
        assert_eq!(strategy.approach, EncodingApproach::Jpegli);
        // AQ currently disabled until properly tuned
        assert!(!strategy.adaptive_quant.enabled);
        // Trellis disabled at very high quality (simplified implementation is too aggressive)
        assert!(!strategy.trellis.ac_enabled);
    }

    #[test]
    fn test_hybrid_strategy() {
        let strategy = select_strategy(&Quality::Standard(60), EncodingStrategy::Hybrid);
        assert_eq!(strategy.approach, EncodingApproach::Hybrid);
        assert!(strategy.trellis.ac_enabled);
        // AQ currently disabled until properly tuned
        assert!(!strategy.adaptive_quant.enabled);
    }
}
