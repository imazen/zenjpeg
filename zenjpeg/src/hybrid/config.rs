//! Configurable hybrid quantization parameters.
//!
//! This module exposes all tunable knobs for the hybrid AQ+trellis approach,
//! enabling systematic parameter sweeps and optimization.
//!
//! # ⚠️ Experimental Status
//!
//! **All findings in this module are preliminary and not statistically validated.**
//! Testing was performed on ~5 images at a few quality levels - far too small a sample
//! for reliable conclusions. The correlation coefficients, improvement percentages,
//! and threshold values should be treated as rough starting points, not established facts.
//!
//! ## Adaptive Mode (Preliminary)
//!
//! Early testing suggested **AQ mean** (average per-block complexity) might predict
//! which images benefit from hybrid trellis. However, the sample size was too small
//! for statistical validity:
//!
//! - `aq_mean > 0.25`: Complex images appeared to benefit more in limited testing
//! - `aq_mean ≤ 0.25`: Simple images showed smaller improvements
//!
//! Use [`should_use_hybrid`] as a heuristic, but verify on your own data.
//!
//! ## Parameters That Appeared Unhelpful (Limited Testing)
//!
//! The following were tested but showed no clear benefit or made things worse:
//!
//! - **`dc_enabled`**: DC trellis optimization - disabled by default, no clear wins observed
//! - **`aq_exponent != 1.0`**: Non-linear AQ mapping (sqrt, squared) - no improvement seen
//! - **`quality_adaptive`**: Scaling lambda by quality dampen - not clearly beneficial
//! - **`chroma_scale != 1.0`**: Separate chroma scaling - not tuned, kept at 1.0
//! - **`num_loops > 1`**: Multiple trellis passes - slower without clear quality gain
//! - **`aq_threshold > 0`**: Minimum AQ cutoff - no benefit observed
//!
//! These remain configurable for further experimentation.

#[cfg(feature = "experimental-hybrid-trellis")]
use mozjpeg_rs::TrellisConfig;

/// Threshold for AQ mean above which hybrid trellis might be beneficial.
///
/// **Note:** This threshold is a rough heuristic from very limited testing (~5 images).
/// The claimed correlation was not statistically validated. Use as a starting point only.
pub const AQ_MEAN_THRESHOLD: f32 = 0.25;

/// Heuristic to predict whether hybrid trellis might benefit this image.
///
/// Returns `true` if the image complexity (AQ mean) exceeds the threshold.
///
/// **⚠️ Preliminary:** This heuristic is based on very limited testing and may not
/// generalize. Consider running your own benchmarks on representative images.
///
/// # Arguments
/// * `aq_mean` - Mean AQ strength across all blocks (from AQStrengthMap)
pub fn should_use_hybrid(aq_mean: f32) -> bool {
    aq_mean > AQ_MEAN_THRESHOLD
}

/// Rough estimate of DSSIM improvement from hybrid trellis.
///
/// **⚠️ Not validated:** This linear model was fit on ~5 images and should not be
/// trusted for production decisions. The coefficients (85, -5) are arbitrary
/// starting points that need validation on larger, more diverse datasets.
///
/// # Returns
/// Estimated percentage improvement in DSSIM (unreliable)
pub fn estimate_hybrid_improvement(aq_mean: f32) -> f32 {
    // Unvalidated linear model - treat with skepticism
    (85.0 * aq_mean - 5.0).max(0.0)
}

/// Configuration for hybrid AQ+trellis quantization.
///
/// All parameters that affect the hybrid encoding can be tuned here.
#[derive(Debug, Clone, Copy)]
pub struct HybridConfig {
    /// Enable hybrid trellis mode
    pub enabled: bool,

    /// How much lambda increases per unit of AQ strength.
    /// Default: 2.0 (aq=0.5 → +1.0 to scale1 → 2x lambda)
    /// Range: 0.0 (ignore AQ) to ~8.0 (very aggressive)
    pub aq_lambda_scale: f32,

    /// Base lambda_log_scale1 value (default: 14.75)
    /// Higher = more conservative (preserve quality)
    /// Lower = more aggressive (smaller files)
    pub base_lambda_scale1: f32,

    /// Base lambda_log_scale2 value (default: 16.5)
    /// Affects the denominator in lambda calculation
    pub base_lambda_scale2: f32,

    /// Enable DC coefficient trellis optimization
    pub dc_enabled: bool,

    /// Number of trellis optimization loops
    pub num_loops: i32,

    /// Use perceptual lambda weighting table
    pub use_lambda_weight_tbl: bool,

    /// AQ strength exponent for non-linear mapping.
    /// 1.0 = linear, 2.0 = squared, 0.5 = sqrt
    pub aq_exponent: f32,

    /// Minimum AQ strength to apply lambda adjustment.
    /// Below this threshold, use base lambda unchanged.
    pub aq_threshold: f32,

    /// Scale lambda adjustment by AQ dampen factor (quality-adaptive).
    /// When true, lambda sensitivity decreases at low quality.
    pub quality_adaptive: bool,

    /// Separate scaling for chroma components (Cb, Cr).
    /// 1.0 = same as luma, <1.0 = less aggressive on chroma
    pub chroma_scale: f32,
}

impl Default for HybridConfig {
    /// Default configuration - a reasonable starting point.
    ///
    /// **Note:** These defaults emerged from limited testing (~5 images) and may not
    /// be optimal for your use case. The efficiency claims below are preliminary:
    /// - aq_lambda_scale=0.0 appeared most efficient in limited testing
    /// - Your mileage may vary significantly on different image types
    fn default() -> Self {
        Self {
            enabled: true,
            // 0.0 = no AQ influence on lambda, best efficiency
            // Use favor_quality() preset for aq_lambda_scale > 0
            aq_lambda_scale: 0.0,
            base_lambda_scale1: 14.75,
            base_lambda_scale2: 16.5,
            dc_enabled: false,
            num_loops: 1,
            use_lambda_weight_tbl: true,
            aq_exponent: 1.0,
            aq_threshold: 0.0,
            quality_adaptive: false,
            chroma_scale: 1.0,
        }
    }
}

impl HybridConfig {
    /// Create a new config with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Disable hybrid mode entirely.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }

    /// Preset favoring smaller file sizes (unvalidated).
    ///
    /// Uses lower base_lambda_scale1 which appeared to reduce size in limited testing.
    /// The specific values are not rigorously validated.
    pub fn favor_size() -> Self {
        Self {
            enabled: true,
            aq_lambda_scale: 0.0,     // Most efficient setting
            base_lambda_scale1: 14.0, // Lower base = smaller files
            dc_enabled: false,
            ..Self::default()
        }
    }

    /// Preset favoring quality improvement (unvalidated).
    ///
    /// Uses higher aq_lambda_scale which appeared to improve quality in limited testing,
    /// potentially at the cost of larger files. Not rigorously validated.
    pub fn favor_quality() -> Self {
        Self {
            enabled: true,
            aq_lambda_scale: 4.0,     // Maximum quality improvement
            base_lambda_scale1: 15.5, // Higher base = more quality
            dc_enabled: false,
            ..Self::default()
        }
    }

    /// Balanced preset (unvalidated).
    ///
    /// Middle-ground settings between favor_size() and favor_quality().
    /// Not rigorously validated - use as a starting point for experimentation.
    pub fn balanced() -> Self {
        Self {
            enabled: true,
            aq_lambda_scale: 2.0,
            base_lambda_scale1: 14.75,
            dc_enabled: false,
            ..Self::default()
        }
    }

    /// Builder: set AQ lambda scale
    pub fn aq_lambda_scale(mut self, scale: f32) -> Self {
        self.aq_lambda_scale = scale;
        self
    }

    /// Builder: set base lambda_log_scale1
    pub fn base_scale1(mut self, scale: f32) -> Self {
        self.base_lambda_scale1 = scale;
        self
    }

    /// Builder: set base lambda_log_scale2
    pub fn base_scale2(mut self, scale: f32) -> Self {
        self.base_lambda_scale2 = scale;
        self
    }

    /// Builder: enable/disable DC trellis
    pub fn dc_trellis(mut self, enabled: bool) -> Self {
        self.dc_enabled = enabled;
        self
    }

    /// Builder: set number of trellis loops
    pub fn num_loops(mut self, loops: i32) -> Self {
        self.num_loops = loops;
        self
    }

    /// Builder: set AQ exponent for non-linear mapping
    pub fn aq_exponent(mut self, exp: f32) -> Self {
        self.aq_exponent = exp;
        self
    }

    /// Builder: set AQ threshold
    pub fn aq_threshold(mut self, threshold: f32) -> Self {
        self.aq_threshold = threshold;
        self
    }

    /// Builder: enable quality-adaptive scaling
    pub fn quality_adaptive(mut self, enabled: bool) -> Self {
        self.quality_adaptive = enabled;
        self
    }

    /// Builder: set chroma scaling factor
    pub fn chroma_scale(mut self, scale: f32) -> Self {
        self.chroma_scale = scale;
        self
    }

    /// Compute the effective lambda adjustment for a block.
    ///
    /// # Arguments
    /// * `aq_strength` - Per-block AQ strength (0.0 to ~0.5)
    /// * `dampen` - Quality-based dampen factor (0.0 to 1.0)
    /// * `is_chroma` - True for Cb/Cr components
    ///
    /// # Returns
    /// The adjustment to add to lambda_log_scale1
    pub fn compute_lambda_adjustment(&self, aq_strength: f32, dampen: f32, is_chroma: bool) -> f32 {
        if !self.enabled || aq_strength < self.aq_threshold {
            return 0.0;
        }

        // Apply non-linear mapping
        let effective_aq = if self.aq_exponent != 1.0 {
            aq_strength.powf(self.aq_exponent)
        } else {
            aq_strength
        };

        // Base adjustment
        let mut adjustment = effective_aq * self.aq_lambda_scale;

        // Quality-adaptive scaling
        if self.quality_adaptive {
            adjustment *= dampen;
        }

        // Chroma scaling
        if is_chroma {
            adjustment *= self.chroma_scale;
        }

        adjustment
    }

    /// Convert to mozjpeg TrellisConfig for a specific block.
    #[cfg(feature = "experimental-hybrid-trellis")]
    pub fn to_trellis_config(
        &self,
        aq_strength: f32,
        dampen: f32,
        is_chroma: bool,
    ) -> TrellisConfig {
        let adjustment = self.compute_lambda_adjustment(aq_strength, dampen, is_chroma);

        TrellisConfig {
            enabled: true,
            dc_enabled: self.dc_enabled,
            eob_opt: true,
            use_lambda_weight_tbl: self.use_lambda_weight_tbl,
            use_scans_in_trellis: false,
            q_opt: false,
            lambda_log_scale1: self.base_lambda_scale1 + adjustment,
            lambda_log_scale2: self.base_lambda_scale2,
            freq_split: 8,
            num_loops: self.num_loops,
            delta_dc_weight: 0.0,
            speed_level: 7, // Adaptive (default)
        }
    }

    /// Generate a short identifier string for this config (for logging/filenames).
    pub fn id(&self) -> String {
        format!(
            "aq{:.1}_s1_{:.1}_dc{}_exp{:.1}",
            self.aq_lambda_scale,
            self.base_lambda_scale1,
            if self.dc_enabled { 1 } else { 0 },
            self.aq_exponent
        )
    }
}

/// Parameter sweep configuration for systematic testing.
#[derive(Debug, Clone)]
pub struct SweepConfig {
    /// AQ lambda scale values to test
    pub aq_lambda_scales: Vec<f32>,
    /// Base lambda_log_scale1 values to test
    pub base_scale1_values: Vec<f32>,
    /// DC enabled states to test
    pub dc_enabled_values: Vec<bool>,
    /// AQ exponent values to test
    pub aq_exponents: Vec<f32>,
    /// Quality levels to test
    pub quality_levels: Vec<u8>,
}

impl Default for SweepConfig {
    fn default() -> Self {
        Self {
            aq_lambda_scales: vec![0.0, 1.0, 2.0, 3.0, 4.0],
            base_scale1_values: vec![14.0, 14.75, 15.5],
            dc_enabled_values: vec![false, true],
            aq_exponents: vec![1.0],
            quality_levels: vec![75],
        }
    }
}

impl SweepConfig {
    /// Quick sweep with fewer combinations for fast iteration.
    pub fn quick() -> Self {
        Self {
            aq_lambda_scales: vec![0.0, 2.0, 4.0],
            base_scale1_values: vec![14.75],
            dc_enabled_values: vec![false],
            aq_exponents: vec![1.0],
            quality_levels: vec![75],
        }
    }

    /// Comprehensive sweep for thorough analysis.
    pub fn comprehensive() -> Self {
        Self {
            aq_lambda_scales: vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0],
            base_scale1_values: vec![13.5, 14.0, 14.5, 14.75, 15.0, 15.5, 16.0],
            dc_enabled_values: vec![false, true],
            aq_exponents: vec![0.5, 1.0, 2.0],
            quality_levels: vec![50, 75, 90],
        }
    }

    /// Generate all HybridConfig combinations.
    pub fn generate_configs(&self) -> Vec<HybridConfig> {
        let mut configs = Vec::new();

        for &aq_scale in &self.aq_lambda_scales {
            for &base_s1 in &self.base_scale1_values {
                for &dc_en in &self.dc_enabled_values {
                    for &aq_exp in &self.aq_exponents {
                        configs.push(HybridConfig {
                            enabled: true,
                            aq_lambda_scale: aq_scale,
                            base_lambda_scale1: base_s1,
                            dc_enabled: dc_en,
                            aq_exponent: aq_exp,
                            ..HybridConfig::default()
                        });
                    }
                }
            }
        }

        configs
    }

    /// Total number of configurations (configs × quality levels).
    pub fn total_combinations(&self) -> usize {
        self.aq_lambda_scales.len()
            * self.base_scale1_values.len()
            * self.dc_enabled_values.len()
            * self.aq_exponents.len()
            * self.quality_levels.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = HybridConfig::default();
        assert!(config.enabled);
        // Default is 0.0 (appeared most efficient in limited testing)
        assert_eq!(config.aq_lambda_scale, 0.0);
        assert_eq!(config.base_lambda_scale1, 14.75);
    }

    #[test]
    fn test_presets() {
        let favor_size = HybridConfig::favor_size();
        assert_eq!(favor_size.base_lambda_scale1, 14.0);

        let favor_quality = HybridConfig::favor_quality();
        assert_eq!(favor_quality.aq_lambda_scale, 4.0);
        assert_eq!(favor_quality.base_lambda_scale1, 15.5);

        let balanced = HybridConfig::balanced();
        assert_eq!(balanced.aq_lambda_scale, 2.0);
    }

    #[test]
    fn test_lambda_adjustment() {
        // Default config has aq_lambda_scale=0.0, so all adjustments are 0
        let config = HybridConfig::default();
        assert_eq!(config.compute_lambda_adjustment(0.5, 1.0, false), 0.0);
        assert_eq!(config.compute_lambda_adjustment(1.0, 1.0, false), 0.0);

        // Use balanced preset which has aq_lambda_scale=2.0
        let balanced = HybridConfig::balanced();

        // Zero AQ = zero adjustment
        assert_eq!(balanced.compute_lambda_adjustment(0.0, 1.0, false), 0.0);

        // 0.5 AQ with scale 2.0 = 1.0 adjustment
        assert_eq!(balanced.compute_lambda_adjustment(0.5, 1.0, false), 1.0);

        // Full AQ (1.0) with scale 2.0 = 2.0 adjustment
        assert_eq!(balanced.compute_lambda_adjustment(1.0, 1.0, false), 2.0);
    }

    #[test]
    fn test_quality_adaptive() {
        // Use balanced preset to have non-zero aq_lambda_scale
        let config = HybridConfig::balanced().quality_adaptive(true);

        // With dampen=0.5, adjustment should be halved
        let adj_full = config.compute_lambda_adjustment(0.5, 1.0, false);
        let adj_half = config.compute_lambda_adjustment(0.5, 0.5, false);
        assert_eq!(adj_half, adj_full * 0.5);
    }

    #[test]
    fn test_aq_exponent() {
        // Use balanced preset to have non-zero aq_lambda_scale
        let config = HybridConfig::balanced().aq_exponent(2.0);

        // With exponent 2.0, aq=0.5 becomes 0.25
        let adj = config.compute_lambda_adjustment(0.5, 1.0, false);
        assert_eq!(adj, 0.25 * 2.0); // 0.5^2 * scale
    }

    #[test]
    fn test_sweep_config() {
        let sweep = SweepConfig::quick();
        let configs = sweep.generate_configs();
        assert_eq!(configs.len(), 3); // 3 aq_scales × 1 × 1 × 1
    }
}
