//! mozjpeg-compatible API types.
//!
//! This module provides types that mirror mozjpeg-rs's API for easier migration
//! and familiarity. These types can be used with jpegli's encoder while providing
//! the same configuration interface as mozjpeg-rs.
//!
//! # Example
//!
//! ```rust,ignore
//! use zenjpeg::encode::{EncoderConfig, ChromaSubsampling, TrellisConfig, TrellisSpeedMode};
//!
//! // Configure trellis like mozjpeg-rs
//! let trellis = TrellisConfig::default()
//!     .ac_trellis(true)
//!     .dc_trellis(true)
//!     .speed_mode(TrellisSpeedMode::Adaptive);
//!
//! let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
//!     .trellis(trellis);
//! ```

/// Speed optimization mode for trellis quantization.
///
/// Trellis quantization has O(n²) complexity per block. For high-entropy
/// blocks (many non-zero coefficients at high quality), this can be slow.
/// These modes control how aggressively to limit the search space.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum TrellisSpeedMode {
    /// Full search on all blocks (slowest, optimal quality).
    /// Use when encoding time is not a concern.
    Thorough,

    /// Two-tier adaptive limits (zenjpeg heuristic, NOT from C mozjpeg).
    ///
    /// - nonzero > 55: lookback=8, candidates=3 (extreme entropy)
    /// - nonzero > 48: lookback=16, candidates=4 (high entropy)
    /// - otherwise: full search
    ///
    /// This is a zenjpeg-specific speed heuristic. C mozjpeg uses a formula-based
    /// approach (see [`Level`](Self::Level)). Use [`Thorough`](Self::Thorough) to
    /// match C mozjpeg's default full-search behavior.
    #[default]
    Adaptive,

    /// Formula-based level (0-10), matching C mozjpeg's `trellis_num_nbits` speed
    /// optimization.
    ///
    /// - threshold = 61 - level×3
    /// - lookback = 26 - level×2
    /// - candidates = 9 - (level+1)/2
    ///
    /// Level 0 = full search (C mozjpeg default), Level 10 = fastest.
    Level(u8),

    /// Custom two-tier thresholds for fine-tuning.
    ///
    /// When nonzero coefficient count exceeds a threshold, the search is limited.
    /// Tier 1 is for extreme entropy (higher threshold, more aggressive limits).
    /// Tier 2 is for high entropy (lower threshold, moderate limits).
    Custom {
        /// Tier 1: nonzero count threshold (e.g., 55)
        tier1_threshold: u8,
        /// Tier 1: maximum lookback positions (e.g., 8)
        tier1_lookback: u8,
        /// Tier 1: maximum quantization candidates (e.g., 3)
        tier1_candidates: u8,
        /// Tier 2: nonzero count threshold (e.g., 48, must be <= tier1)
        tier2_threshold: u8,
        /// Tier 2: maximum lookback positions (e.g., 16)
        tier2_lookback: u8,
        /// Tier 2: maximum quantization candidates (e.g., 4)
        tier2_candidates: u8,
    },
}

impl TrellisSpeedMode {
    /// Returns (max_lookback, max_candidates) for the given nonzero count.
    #[inline]
    pub fn get_limits(&self, nonzero_count: i32) -> (usize, usize) {
        match *self {
            Self::Thorough => (63, 16),
            Self::Adaptive => {
                if nonzero_count > 55 {
                    (8, 3)
                } else if nonzero_count > 48 {
                    (16, 4)
                } else {
                    (63, 16)
                }
            }
            Self::Level(level) => {
                let level = level.min(10) as i32;
                if level == 0 {
                    return (63, 16);
                }
                let threshold = 61 - level * 3;
                if nonzero_count > threshold {
                    let lookback = (26 - level * 2).max(4) as usize;
                    let candidates = (9 - (level + 1) / 2).max(2) as usize;
                    (lookback, candidates)
                } else {
                    (63, 16)
                }
            }
            Self::Custom {
                tier1_threshold,
                tier1_lookback,
                tier1_candidates,
                tier2_threshold,
                tier2_lookback,
                tier2_candidates,
            } => {
                if nonzero_count > tier1_threshold as i32 {
                    (tier1_lookback as usize, tier1_candidates as usize)
                } else if nonzero_count > tier2_threshold as i32 {
                    (tier2_lookback as usize, tier2_candidates as usize)
                } else {
                    (63, 16)
                }
            }
        }
    }
}

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
/// - [`TrellisConfig::default()`] - AC + DC trellis, Adaptive speed (zenjpeg heuristic)
/// - [`TrellisConfig::thorough()`] - Full search, matches C mozjpeg default
/// - [`TrellisConfig::disabled()`] - No trellis (fastest encoding)
/// - [`TrellisConfig::favor_size()`] - More aggressive zeroing (smaller files)
/// - [`TrellisConfig::favor_quality()`] - More conservative (better quality)
///
/// # Speed Modes
///
/// The [`TrellisSpeedMode`] controls search limiting for high-entropy blocks:
///
/// - **Thorough**: Full O(n²) search. C mozjpeg default. Optimal quality.
/// - **Adaptive** (zenjpeg default): Two-tier heuristic, ~30% faster on noisy images.
/// - **Level(0-10)**: C mozjpeg formula. Level 0 = full search, Level 10 = fastest.
///
/// Speed modes only affect high-entropy blocks (many non-zero coefficients
/// at high quality). At lower quality or on smooth images, most blocks have
/// few non-zero coefficients and the optimization rarely triggers.
///
/// Quality impact is negligible even at the fastest settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrellisConfig {
    /// Enable trellis quantization for AC coefficients.
    pub enabled: bool,
    /// Enable trellis quantization for DC coefficients.
    pub dc_enabled: bool,
    /// Use perceptual lambda weighting table.
    ///
    /// **Currently unused:** The implementation always uses flat 1/q² weights
    /// regardless of this flag (see `encode/trellis/ac.rs`). Retained for
    /// future implementation of perceptual weighting.
    pub use_lambda_weight_tbl: bool,
    /// Lambda log scale parameter 1 (rate penalty).
    pub lambda_log_scale1: f32,
    /// Lambda log scale parameter 2 (distortion sensitivity).
    pub lambda_log_scale2: f32,
    /// Number of trellis optimization loops.
    ///
    /// **Currently unused:** The implementation always performs a single pass.
    /// Multi-loop trellis (iterating until convergence) is a potential future
    /// optimization — each loop refines coefficient choices based on updated
    /// rate estimates from the previous loop.
    pub num_loops: i32,
    /// Speed optimization mode.
    pub speed_mode: TrellisSpeedMode,
    /// Weight for vertical DC gradient consideration in DC trellis.
    /// When > 0.0, DC trellis also considers the difference between
    /// the current block's DC and the block directly above it, penalizing
    /// large vertical DC jumps that create visible banding.
    /// Default: 0.0 (disabled, matching C mozjpeg default).
    pub delta_dc_weight: f32,
}

/// Default lambda_log_scale1 value (matches mozjpeg)
const DEFAULT_LAMBDA_LOG_SCALE1: f32 = 14.75;
/// Default lambda_log_scale2 value (matches mozjpeg)
const DEFAULT_LAMBDA_LOG_SCALE2: f32 = 16.5;

impl Default for TrellisConfig {
    /// Default configuration: AC + DC trellis enabled, Adaptive speed.
    ///
    /// Note: C mozjpeg defaults to full search (Thorough). This uses the
    /// faster zenjpeg Adaptive heuristic instead. Use [`TrellisConfig::thorough()`]
    /// or the `MozjpegBaseline`/`MozjpegProgressive` presets for C mozjpeg parity.
    fn default() -> Self {
        Self {
            enabled: true,
            dc_enabled: true,
            use_lambda_weight_tbl: true,
            lambda_log_scale1: DEFAULT_LAMBDA_LOG_SCALE1,
            lambda_log_scale2: DEFAULT_LAMBDA_LOG_SCALE2,
            num_loops: 1,
            speed_mode: TrellisSpeedMode::Adaptive,
            delta_dc_weight: 0.0,
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
            use_lambda_weight_tbl: false,
            lambda_log_scale1: DEFAULT_LAMBDA_LOG_SCALE1,
            lambda_log_scale2: DEFAULT_LAMBDA_LOG_SCALE2,
            num_loops: 1,
            speed_mode: TrellisSpeedMode::Adaptive,
            delta_dc_weight: 0.0,
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

    /// Preset for thorough encoding (full search).
    ///
    /// Full trellis search on all blocks with no speed optimizations.
    /// Slowest but produces optimal results. Use when encoding time is not a concern.
    #[must_use]
    pub fn thorough() -> Self {
        Self {
            speed_mode: TrellisSpeedMode::Thorough,
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
    /// use zenjpeg::encode::TrellisConfig;
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

    /// Set the speed optimization mode.
    ///
    /// See [`TrellisSpeedMode`] for available modes.
    #[must_use]
    pub fn speed_mode(mut self, mode: TrellisSpeedMode) -> Self {
        self.speed_mode = mode;
        self
    }

    /// Set the speed optimization level (0-10).
    ///
    /// This uses the formula-based [`TrellisSpeedMode::Level`] mode.
    /// For C mozjpeg compatibility, use [`TrellisSpeedMode::Adaptive`] instead.
    ///
    /// | Level | Speed | Quality | Notes |
    /// |-------|-------|---------|-------|
    /// | 0 | ~1x | Optimal | Full search on all blocks |
    /// | 7 | ~1.3x | Excellent | ≈ C mozjpeg adaptive |
    /// | 10 | ~1.5x | Very good | Most aggressive limiting |
    ///
    /// Speed gains are most significant for Q80-100 on noisy/high-detail images.
    /// At lower quality or on smooth images, most blocks have few non-zero
    /// coefficients and the optimization rarely triggers.
    #[deprecated(
        since = "0.7.0",
        note = "Use speed_mode(TrellisSpeedMode::Level(n)) instead"
    )]
    #[must_use]
    pub fn speed_level(mut self, level: u8) -> Self {
        self.speed_mode = TrellisSpeedMode::Level(level.min(10));
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

    /// Set the weight for vertical DC gradient consideration in DC trellis.
    ///
    /// When > 0.0, DC trellis considers the difference between the current
    /// block's DC value and the block directly above it, penalizing large
    /// vertical DC jumps that create visible banding artifacts.
    ///
    /// - `0.0`: Disabled (default, matching C mozjpeg default behavior)
    /// - `0.5`: Moderate vertical smoothing
    /// - `1.0`: Strong vertical smoothing (may soften horizontal edges)
    ///
    /// Only has effect when DC trellis is enabled.
    ///
    /// Default: `0.0`
    #[must_use]
    pub fn delta_dc_weight(mut self, weight: f32) -> Self {
        self.delta_dc_weight = weight.max(0.0);
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

    /// Get the current speed mode.
    #[must_use]
    pub fn get_speed_mode(&self) -> TrellisSpeedMode {
        self.speed_mode
    }

    /// Get the current speed level (deprecated).
    #[deprecated(since = "0.7.0", note = "Use get_speed_mode() instead")]
    #[must_use]
    pub fn get_speed_level(&self) -> u8 {
        match self.speed_mode {
            TrellisSpeedMode::Thorough => 0,
            TrellisSpeedMode::Adaptive => 7,
            TrellisSpeedMode::Level(l) => l,
            TrellisSpeedMode::Custom { .. } => 7, // Approximate
        }
    }

    /// Check if any trellis optimization is enabled.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.enabled || self.dc_enabled
    }

    /// Returns the delta DC weight for vertical gradient consideration.
    #[must_use]
    pub fn get_delta_dc_weight(&self) -> f32 {
        self.delta_dc_weight
    }

    /// Returns lambda log scale parameter 1 (rate penalty).
    #[must_use]
    pub fn lambda_log_scale1(&self) -> f32 {
        self.lambda_log_scale1
    }

    /// Returns lambda log scale parameter 2 (distortion sensitivity).
    #[must_use]
    pub fn lambda_log_scale2(&self) -> f32 {
        self.lambda_log_scale2
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
        assert_eq!(config.speed_mode, TrellisSpeedMode::Adaptive);
        assert!((config.lambda_log_scale1 - 14.75).abs() < 0.01);
        assert!((config.delta_dc_weight - 0.0).abs() < 0.001);
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
        assert_eq!(thorough.speed_mode, TrellisSpeedMode::Thorough);
    }

    #[test]
    fn test_builder_chain() {
        let config = TrellisConfig::default()
            .ac_trellis(true)
            .dc_trellis(false)
            .speed_mode(TrellisSpeedMode::Level(5))
            .lambda_scales(15.0, 17.0);

        assert!(config.enabled);
        assert!(!config.dc_enabled);
        assert_eq!(config.speed_mode, TrellisSpeedMode::Level(5));
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
    #[allow(deprecated)]
    fn test_speed_level_clamping() {
        let config = TrellisConfig::default().speed_level(15);
        assert_eq!(config.speed_mode, TrellisSpeedMode::Level(10)); // Clamped to max
    }

    #[test]
    fn test_speed_mode_variants() {
        // Test all variants
        let thorough = TrellisConfig::default().speed_mode(TrellisSpeedMode::Thorough);
        assert_eq!(thorough.speed_mode, TrellisSpeedMode::Thorough);

        let adaptive = TrellisConfig::default().speed_mode(TrellisSpeedMode::Adaptive);
        assert_eq!(adaptive.speed_mode, TrellisSpeedMode::Adaptive);

        let level = TrellisConfig::default().speed_mode(TrellisSpeedMode::Level(5));
        assert_eq!(level.speed_mode, TrellisSpeedMode::Level(5));

        let custom = TrellisConfig::default().speed_mode(TrellisSpeedMode::Custom {
            tier1_threshold: 55,
            tier1_lookback: 8,
            tier1_candidates: 3,
            tier2_threshold: 48,
            tier2_lookback: 16,
            tier2_candidates: 4,
        });
        assert!(matches!(custom.speed_mode, TrellisSpeedMode::Custom { .. }));
    }

    #[test]
    fn test_speed_mode_get_limits() {
        // Thorough always returns full search
        assert_eq!(TrellisSpeedMode::Thorough.get_limits(60), (63, 16));

        // Adaptive uses two-tier thresholds
        assert_eq!(TrellisSpeedMode::Adaptive.get_limits(56), (8, 3)); // > 55
        assert_eq!(TrellisSpeedMode::Adaptive.get_limits(50), (16, 4)); // > 48
        assert_eq!(TrellisSpeedMode::Adaptive.get_limits(40), (63, 16)); // <= 48

        // Level(0) is full search
        assert_eq!(TrellisSpeedMode::Level(0).get_limits(60), (63, 16));
    }

    #[test]
    fn test_num_loops_minimum() {
        let config = TrellisConfig::default().num_loops(0);
        assert_eq!(config.num_loops, 1); // Minimum is 1
    }

    #[test]
    fn test_delta_dc_weight() {
        let config = TrellisConfig::default().delta_dc_weight(0.5);
        assert!((config.get_delta_dc_weight() - 0.5).abs() < 0.001);

        // Negative weights are clamped to 0
        let config = TrellisConfig::default().delta_dc_weight(-1.0);
        assert!((config.get_delta_dc_weight() - 0.0).abs() < 0.001);
    }
}
