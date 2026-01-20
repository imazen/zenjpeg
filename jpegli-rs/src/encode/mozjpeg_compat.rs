//! mozjpeg-compatible API types.
//!
//! This module provides types that mirror mozjpeg-rs's API for easier migration
//! and familiarity. These types can be used with jpegli's encoder while providing
//! the same configuration interface as mozjpeg-rs.
//!
//! # Example
//!
//! ```rust,ignore
//! use jpegli::encode::{EncoderConfig, ChromaSubsampling, TrellisConfig};
//!
//! // Configure trellis like mozjpeg-rs
//! let trellis = TrellisConfig::default()
//!     .ac_trellis(true)
//!     .dc_trellis(true)
//!     .speed_level(7);
//!
//! let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
//!     .trellis(trellis);
//! ```

/// Configuration for trellis quantization.
///
/// Trellis quantization uses dynamic programming to find optimal quantization
/// decisions that minimize rate + lambda * distortion. This typically produces
/// 10-15% smaller files at the same quality compared to simple rounding.
///
/// This type mirrors mozjpeg-rs's `TrellisConfig` API for compatibility.
///
/// # Presets
///
/// - [`TrellisConfig::default()`] - Balanced settings (AC + DC trellis, speed_level=7)
/// - [`TrellisConfig::disabled()`] - No trellis (fastest encoding)
/// - [`TrellisConfig::favor_size()`] - More aggressive zeroing (smaller files)
/// - [`TrellisConfig::favor_quality()`] - More conservative (better quality)
/// - [`TrellisConfig::thorough()`] - Full search, no speed optimizations
///
/// # Speed Levels
///
/// The `speed_level` parameter (0-10) controls adaptive search limiting:
///
/// - **0** = Thorough: Full O(n²) search on all blocks. Slowest but optimal.
/// - **7** = Default: ~30% faster. Limits search on high-entropy blocks.
/// - **10** = Fast: ~50% faster. Most aggressive limiting.
///
/// Speed levels only affect high-entropy blocks (many non-zero coefficients
/// at high quality). At lower quality or on smooth images, most blocks have
/// few non-zero coefficients and the optimization rarely triggers.
///
/// Quality impact is negligible even at level 10.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrellisConfig {
    /// Enable trellis quantization for AC coefficients
    pub(crate) enabled: bool,
    /// Enable trellis quantization for DC coefficients
    pub(crate) dc_enabled: bool,
    /// Optimize for sequences of EOB (end-of-block)
    pub(crate) eob_opt: bool,
    /// Use perceptual lambda weighting table
    pub(crate) use_lambda_weight_tbl: bool,
    /// Lambda log scale parameter 1 (rate penalty)
    pub(crate) lambda_log_scale1: f32,
    /// Lambda log scale parameter 2 (distortion sensitivity)
    pub(crate) lambda_log_scale2: f32,
    /// Number of trellis optimization loops
    pub(crate) num_loops: i32,
    /// Speed optimization level (0-10)
    pub(crate) speed_level: u8,
}

/// Default lambda_log_scale1 value (matches mozjpeg)
const DEFAULT_LAMBDA_LOG_SCALE1: f32 = 14.75;
/// Default lambda_log_scale2 value (matches mozjpeg)
const DEFAULT_LAMBDA_LOG_SCALE2: f32 = 16.5;

impl Default for TrellisConfig {
    /// Default configuration: AC + DC trellis enabled, balanced speed.
    ///
    /// Matches mozjpeg's default trellis behavior.
    fn default() -> Self {
        Self {
            enabled: true,
            dc_enabled: true,
            // EOB optimization disabled by default to match C mozjpeg.
            // The cross-block EOB algorithm can be aggressive in some cases.
            eob_opt: false,
            use_lambda_weight_tbl: true,
            lambda_log_scale1: DEFAULT_LAMBDA_LOG_SCALE1,
            lambda_log_scale2: DEFAULT_LAMBDA_LOG_SCALE2,
            num_loops: 1,
            speed_level: 7, // Balanced speed/quality
        }
    }
}

impl TrellisConfig {
    /// Create a new trellis configuration with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Configuration with trellis disabled (fastest encoding).
    ///
    /// Use this when encoding speed is critical and file size is less important.
    /// Produces ~10-15% larger files compared to trellis-enabled modes.
    #[must_use]
    pub const fn disabled() -> Self {
        Self {
            enabled: false,
            dc_enabled: false,
            eob_opt: false,
            use_lambda_weight_tbl: false,
            lambda_log_scale1: DEFAULT_LAMBDA_LOG_SCALE1,
            lambda_log_scale2: DEFAULT_LAMBDA_LOG_SCALE2,
            num_loops: 1,
            speed_level: 7,
        }
    }

    /// Preset that favors smaller file sizes over quality.
    ///
    /// Uses lower lambda values which makes the trellis algorithm more aggressive
    /// about zeroing coefficients, resulting in smaller files at the cost of some
    /// quality loss.
    #[must_use]
    pub fn favor_size() -> Self {
        Self {
            lambda_log_scale1: 14.0, // Lower = less distortion penalty
            lambda_log_scale2: 17.0, // Higher = smaller lambda
            ..Self::default()
        }
    }

    /// Preset that favors quality over file size.
    ///
    /// Uses higher lambda values which makes the trellis algorithm more conservative,
    /// preserving more coefficients for better quality at the cost of larger files.
    #[must_use]
    pub fn favor_quality() -> Self {
        Self {
            lambda_log_scale1: 15.5, // Higher = more distortion penalty
            lambda_log_scale2: 16.0, // Lower = larger lambda
            ..Self::default()
        }
    }

    /// Preset for thorough encoding (speed_level=0).
    ///
    /// Full trellis search on all blocks with no speed optimizations.
    /// Slowest but produces optimal results. Use when encoding time is not a concern.
    #[must_use]
    pub fn thorough() -> Self {
        Self {
            speed_level: 0,
            ..Self::default()
        }
    }

    // === Builder Methods ===

    /// Enable or disable AC coefficient trellis optimization.
    ///
    /// AC trellis optimizes the 63 AC coefficients in each 8x8 block using
    /// rate-distortion optimization. This is the main source of file size savings.
    ///
    /// Default: `true`
    #[must_use]
    pub fn ac_trellis(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Enable or disable DC coefficient trellis optimization.
    ///
    /// DC trellis optimizes the DC coefficient across multiple blocks using
    /// dynamic programming. It considers the differential encoding of DC values
    /// to find the optimal path.
    ///
    /// Default: `true`
    #[must_use]
    pub fn dc_trellis(mut self, enabled: bool) -> Self {
        self.dc_enabled = enabled;
        self
    }

    /// Enable or disable EOB (end-of-block) run optimization.
    ///
    /// When enabled, the encoder optimizes sequences of all-zero blocks by
    /// considering EOBRUN codes that encode multiple consecutive EOBs efficiently.
    ///
    /// **Note:** Disabled by default to match C mozjpeg behavior. The cross-block
    /// EOB algorithm can be aggressive with coefficient zeroing in some cases
    /// (especially with chroma subsampling).
    ///
    /// Default: `false`
    #[must_use]
    pub fn eob_optimization(mut self, enabled: bool) -> Self {
        self.eob_opt = enabled;
        self
    }

    /// Set the lambda log scale parameters directly.
    ///
    /// These control the rate-distortion tradeoff in trellis quantization:
    /// - `scale1`: Controls rate penalty (higher = smaller files, default 14.75)
    /// - `scale2`: Controls distortion sensitivity (higher = better quality, default 16.5)
    ///
    /// The effective lambda is: `2^scale1 / (2^scale2 + block_norm)`
    ///
    /// For most use cases, prefer [`rd_factor()`](Self::rd_factor) which provides
    /// a simpler interface.
    #[must_use]
    pub fn lambda_scales(mut self, scale1: f32, scale2: f32) -> Self {
        self.lambda_log_scale1 = scale1;
        self.lambda_log_scale2 = scale2;
        self
    }

    /// Adjust rate-distortion balance with a simple factor.
    ///
    /// This provides a simpler interface than [`lambda_scales()`](Self::lambda_scales):
    ///
    /// - `factor > 1.0`: Favor quality (higher lambda, more conservative)
    /// - `factor < 1.0`: Favor smaller files (lower lambda, more aggressive)
    /// - `factor = 1.0`: Default behavior
    ///
    /// The factor multiplies the effective lambda value logarithmically.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use jpegli::encode::TrellisConfig;
    ///
    /// // Favor smaller files (more aggressive zeroing)
    /// let config = TrellisConfig::default().rd_factor(0.7);
    ///
    /// // Favor quality (preserve more coefficients)
    /// let config = TrellisConfig::default().rd_factor(1.5);
    /// ```
    #[must_use]
    pub fn rd_factor(mut self, factor: f32) -> Self {
        // Adjust scale1 by log2 of the factor
        // factor=2.0 adds 1.0 to scale1 (doubles lambda -> more quality)
        // factor=0.5 subtracts 1.0 from scale1 (halves lambda -> smaller files)
        self.lambda_log_scale1 = DEFAULT_LAMBDA_LOG_SCALE1 + factor.log2();
        self
    }

    /// Set the speed optimization level (0-10).
    ///
    /// Higher levels detect high-entropy blocks and limit the trellis search,
    /// trading a negligible quality loss for faster encoding.
    ///
    /// | Level | Speed | Quality | Notes |
    /// |-------|-------|---------|-------|
    /// | 0 | ~1x | Optimal | Full search on all blocks |
    /// | 7 | ~1.3x | Excellent | Default, balanced |
    /// | 10 | ~1.5x | Very good | Most aggressive limiting |
    ///
    /// Speed gains are most significant for Q80-100 on noisy/high-detail images.
    /// At lower quality or on smooth images, most blocks have few non-zero
    /// coefficients and the optimization rarely triggers.
    ///
    /// Default: `7`
    #[must_use]
    pub fn speed_level(mut self, level: u8) -> Self {
        self.speed_level = level.min(10);
        self
    }

    /// Set the number of trellis optimization loops.
    ///
    /// Multiple loops can improve results but with diminishing returns.
    /// Generally not worth increasing beyond 1.
    ///
    /// Default: `1`
    #[must_use]
    pub fn num_loops(mut self, loops: i32) -> Self {
        self.num_loops = loops.max(1);
        self
    }

    // === Accessors ===

    /// Check if AC trellis is enabled.
    #[must_use]
    pub fn is_ac_enabled(&self) -> bool {
        self.enabled
    }

    /// Check if DC trellis is enabled.
    #[must_use]
    pub fn is_dc_enabled(&self) -> bool {
        self.dc_enabled
    }

    /// Check if EOB optimization is enabled.
    #[must_use]
    pub fn is_eob_opt_enabled(&self) -> bool {
        self.eob_opt
    }

    /// Get the current speed level.
    #[must_use]
    pub fn get_speed_level(&self) -> u8 {
        self.speed_level
    }

    /// Check if any trellis optimization is enabled.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.enabled || self.dc_enabled
    }

    /// Convert to mozjpeg-rs TrellisConfig for actual trellis quantization.
    ///
    /// This requires the `experimental-hybrid-trellis` feature.
    #[cfg(feature = "experimental-hybrid-trellis")]
    #[must_use]
    pub fn to_mozjpeg_config(&self) -> mozjpeg_rs::TrellisConfig {
        mozjpeg_rs::TrellisConfig {
            enabled: self.enabled,
            dc_enabled: self.dc_enabled,
            eob_opt: self.eob_opt,
            use_lambda_weight_tbl: self.use_lambda_weight_tbl,
            use_scans_in_trellis: false, // Not exposed in simplified API
            q_opt: false,                // Not exposed in simplified API
            lambda_log_scale1: self.lambda_log_scale1,
            lambda_log_scale2: self.lambda_log_scale2,
            freq_split: 8, // Default, not exposed
            num_loops: self.num_loops,
            delta_dc_weight: 0.0, // Default, not exposed
            speed_level: self.speed_level,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default() {
        let config = TrellisConfig::default();
        assert!(config.enabled);
        assert!(config.dc_enabled);
        assert!(!config.eob_opt); // Disabled by default
        assert_eq!(config.speed_level, 7);
        assert!((config.lambda_log_scale1 - 14.75).abs() < 0.01);
    }

    #[test]
    fn test_disabled() {
        let config = TrellisConfig::disabled();
        assert!(!config.enabled);
        assert!(!config.dc_enabled);
        assert!(!config.is_enabled());
    }

    #[test]
    fn test_presets() {
        let favor_size = TrellisConfig::favor_size();
        assert!(favor_size.lambda_log_scale1 < DEFAULT_LAMBDA_LOG_SCALE1);

        let favor_quality = TrellisConfig::favor_quality();
        assert!(favor_quality.lambda_log_scale1 > DEFAULT_LAMBDA_LOG_SCALE1);

        let thorough = TrellisConfig::thorough();
        assert_eq!(thorough.speed_level, 0);
    }

    #[test]
    fn test_builder_chain() {
        let config = TrellisConfig::default()
            .ac_trellis(true)
            .dc_trellis(false)
            .eob_optimization(true)
            .speed_level(5)
            .lambda_scales(15.0, 17.0);

        assert!(config.enabled);
        assert!(!config.dc_enabled);
        assert!(config.eob_opt);
        assert_eq!(config.speed_level, 5);
        assert!((config.lambda_log_scale1 - 15.0).abs() < 0.01);
        assert!((config.lambda_log_scale2 - 17.0).abs() < 0.01);
    }

    #[test]
    fn test_rd_factor() {
        // factor=1.0 should give default scale1
        let config = TrellisConfig::default().rd_factor(1.0);
        assert!((config.lambda_log_scale1 - DEFAULT_LAMBDA_LOG_SCALE1).abs() < 0.01);

        // factor=2.0 should add 1.0 to scale1
        let config = TrellisConfig::default().rd_factor(2.0);
        assert!((config.lambda_log_scale1 - (DEFAULT_LAMBDA_LOG_SCALE1 + 1.0)).abs() < 0.01);

        // factor=0.5 should subtract 1.0 from scale1
        let config = TrellisConfig::default().rd_factor(0.5);
        assert!((config.lambda_log_scale1 - (DEFAULT_LAMBDA_LOG_SCALE1 - 1.0)).abs() < 0.01);
    }

    #[test]
    fn test_speed_level_clamping() {
        let config = TrellisConfig::default().speed_level(15);
        assert_eq!(config.speed_level, 10); // Clamped to max
    }

    #[test]
    fn test_num_loops_minimum() {
        let config = TrellisConfig::default().num_loops(0);
        assert_eq!(config.num_loops, 1); // Minimum is 1
    }
}
