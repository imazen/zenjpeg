//! Hybrid quantization: jpegli AQ + mozjpeg trellis.
//!
//! This module combines:
//! - Configurable hybrid quantization parameters ([`HybridConfig`])
//! - Core hybrid quantization algorithm ([`hybrid_quantize_block`])
//! - Encoder integration ([`HybridQuantContext`])
//!
//! Merged from:
//! - `hybrid/config.rs` - HybridConfig, SweepConfig, adaptive detection
//! - `hybrid/core.rs` - StandardRateTables, hybrid_quantize_block
//! - `encode/hybrid.rs` - HybridQuantContext, encoder integration

use super::compat::TrellisConfig;
use super::{RateTable, trellis_quantize_block};
use crate::encode::config::ComputedConfig;
use crate::encode::dct::forward_dct_8x8;
use crate::encode::natural_to_zigzag_into;
use crate::error::Result;
use crate::foundation::consts::DCT_BLOCK_SIZE;
use crate::quant::aq::AQStrengthMap;
use crate::quant::{self, QuantTable, ZeroBiasParams};

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

// ============================================================================
// Image Type Detection (Experimental)
// ============================================================================

/// Detected image type based on AQ statistics.
///
/// **⚠️ Experimental:** These classifications are based on limited testing with
/// ~5 images. The thresholds may not generalize. Always test on your own data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageType {
    /// Photographic content: high texture, moderate variance.
    /// Benefits from aggressive negative coupling without protection.
    Photo,
    /// Screenshot/UI/text: low mean AQ with sharp edges (high local variance).
    /// Requires max_adjustment protection or positive coupling.
    Screenshot,
    /// Mixed content or unclassifiable.
    /// Use safe settings with moderate protection.
    Mixed,
}

/// Thresholds for image type detection.
///
/// **⚠️ Experimental:** Derived from testing on ~5 images.
/// - flower_small (photo): mean=0.090, std=0.065, cv=0.72
/// - apple.com (screenshot with images): mean=0.084, std=0.198, cv=2.35
pub mod detection_thresholds {
    /// Coefficient of variation (std/mean) threshold for screenshot detection.
    /// Screenshots have very high CV due to sharp edges between flat and textured regions.
    /// Photos typically have CV < 1.0 (consistent texture throughout).
    pub const SCREENSHOT_CV_THRESHOLD: f32 = 1.5;

    /// Minimum mean AQ for photo classification.
    /// Photos have higher texture = higher mean AQ.
    pub const PHOTO_MEAN_THRESHOLD: f32 = 0.06;

    /// Very low mean indicates flat/UI content.
    pub const FLAT_MEAN_THRESHOLD: f32 = 0.03;
}

/// Detect image type based on AQ statistics.
///
/// Uses coefficient of variation (std/mean) and absolute mean to classify:
/// - **Photo**: Higher mean, moderate CV → natural textures throughout
/// - **Screenshot**: Lower mean, high CV → flat areas with sharp edges
/// - **Mixed**: Ambiguous statistics
///
/// **⚠️ Experimental:** Thresholds from ~5 images. Test on your own data.
///
/// # Arguments
/// * `aq_mean` - Mean AQ strength from `AQStrengthMap::mean()`
/// * `aq_std` - Standard deviation from `AQStrengthMap::std()`
///
/// # Example
/// ```
/// use zenjpeg::encode::trellis::hybrid::{detect_image_type, ImageType};
///
/// // Typical photo statistics
/// assert_eq!(detect_image_type(0.068, 0.039), ImageType::Photo);
///
/// // Typical screenshot statistics (high CV = std/mean > 1.5)
/// assert_eq!(detect_image_type(0.084, 0.198), ImageType::Screenshot);
/// ```
pub fn detect_image_type(aq_mean: f32, aq_std: f32) -> ImageType {
    use detection_thresholds::*;

    // Avoid division by zero
    if aq_mean < 0.001 {
        return ImageType::Screenshot; // Very flat = likely UI
    }

    let cv = aq_std / aq_mean; // Coefficient of variation

    // Very low mean regardless of CV = flat content (UI, solid colors)
    if aq_mean < FLAT_MEAN_THRESHOLD {
        return ImageType::Screenshot;
    }

    // Very high CV = highly variable content (screenshots with embedded images,
    // web pages, UI with photos). Even if mean is moderate, the extreme variance
    // indicates mixed content that needs protection.
    if cv > SCREENSHOT_CV_THRESHOLD {
        return ImageType::Screenshot;
    }

    // Moderate mean with reasonable CV = natural photo
    if aq_mean >= PHOTO_MEAN_THRESHOLD && cv <= SCREENSHOT_CV_THRESHOLD {
        return ImageType::Photo;
    }

    // Ambiguous cases (low mean, low CV)
    ImageType::Mixed
}

/// Get an adaptive HybridConfig based on detected image type and texture level.
///
/// **Texture-adaptive coupling:** For photos, the coupling strength is scaled
/// based on AQ mean to avoid over-quantizing high-texture images:
/// - Low texture (mean ≤ 0.15): coupling = -4.0 (aggressive)
/// - High texture (mean > 0.15): coupling scales down proportionally
///   - mean=0.30 → coupling=-2.0
///   - mean=0.45 → coupling=-1.3
///   - mean=0.60 → coupling=-1.0
///
/// This prevents the +20-45% butteraugli degradation seen on highly textured
/// CID22 images while maintaining good compression on simpler photos.
///
/// **⚠️ Experimental:** Based on CID22 testing. Validate on your own data.
///
/// # Arguments
/// * `aq_mean` - Mean AQ strength from `AQStrengthMap::mean()`
/// * `aq_std` - Standard deviation from `AQStrengthMap::std()`
///
/// # Example
/// ```
/// use zenjpeg::encode::trellis::hybrid::adaptive_config;
///
/// // Low-texture photo: aggressive compression
/// let config = adaptive_config(0.10, 0.05);
/// assert!(config.aq_lambda_scale <= -3.0);
///
/// // High-texture photo: gentler compression
/// let config = adaptive_config(0.50, 0.20);
/// assert!(config.aq_lambda_scale >= -2.0);
///
/// // Screenshot: safe compression with protection
/// let config = adaptive_config(0.05, 0.10);
/// assert!(config.max_adjustment > 0.0);
/// ```
pub fn adaptive_config(aq_mean: f32, aq_std: f32) -> HybridConfig {
    match detect_image_type(aq_mean, aq_std) {
        ImageType::Photo => {
            // Scale coupling based on texture level
            // High-texture images (high mean) get gentler coupling
            let texture_scale = (0.15 / aq_mean.max(0.15)).min(1.0);
            let coupling = -4.0 * texture_scale;

            HybridConfig {
                enabled: true,
                aq_lambda_scale: coupling,
                base_lambda_scale1: 14.75,
                dc_enabled: false,
                max_adjustment: 0.0, // No cap for photos
                ..HybridConfig::default()
            }
        }
        ImageType::Screenshot | ImageType::Mixed => HybridConfig::safe_compression(),
    }
}

/// Compute texture-adaptive coupling for a given AQ mean.
///
/// Returns a coupling value that's aggressive for low-texture images
/// and gentler for high-texture images.
///
/// Formula: coupling = -4.0 * (0.15 / max(aq_mean, 0.15))
///
/// | AQ Mean | Coupling |
/// |---------|----------|
/// | 0.10    | -4.0     |
/// | 0.15    | -4.0     |
/// | 0.30    | -2.0     |
/// | 0.45    | -1.33    |
/// | 0.60    | -1.0     |
/// | 0.75    | -0.8     |
pub fn texture_adaptive_coupling(aq_mean: f32) -> f32 {
    let texture_scale = (0.15 / aq_mean.max(0.15)).min(1.0);
    -4.0 * texture_scale
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

    /// Number of trellis optimization loops.
    ///
    /// **Currently unused:** The implementation always performs a single pass.
    /// Multi-loop trellis (iterating until convergence) is a potential future
    /// optimization.
    pub num_loops: i32,

    /// Use perceptual lambda weighting table.
    ///
    /// **Currently unused:** The implementation always uses flat 1/q² weights.
    /// Retained for future implementation of perceptual weighting.
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

    /// Use multiplicative coupling instead of additive.
    /// Additive (default): scale1 = base_scale1 + aq * coupling
    /// Multiplicative: scale1 = base_scale1 * (1 + aq * coupling)
    pub multiplicative: bool,

    /// Maximum absolute lambda adjustment (clamps to [-max, +max]).
    /// 0.0 = no limit. Useful for limiting quality degradation on sensitive images.
    pub max_adjustment: f32,
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
            multiplicative: false,
            max_adjustment: 0.0,
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

    /// Aggressive size reduction for photographic images.
    ///
    /// Uses negative coupling to compress textured areas more aggressively.
    /// Results: ~5-6% smaller files, ~8-9% worse DSSIM on photos.
    ///
    /// **WARNING**: Do NOT use on screenshots or text - causes severe degradation.
    /// Use `safe_compression()` for mixed or unknown content.
    pub fn aggressive_compression() -> Self {
        Self {
            enabled: true,
            aq_lambda_scale: -4.0, // Negative = smaller files
            base_lambda_scale1: 14.75,
            dc_enabled: false,
            max_adjustment: 0.0, // No cap (photos only!)
            ..Self::default()
        }
    }

    /// Safe compression with screenshot protection.
    ///
    /// Uses aggressive negative coupling but caps the adjustment to prevent
    /// quality destruction on screenshots and text images.
    ///
    /// Results on photos: ~4% smaller files, ~7-8% worse DSSIM
    /// Results on screenshots: +5% larger files (trellis overhead), <6% worse DSSIM
    pub fn safe_compression() -> Self {
        Self {
            enabled: true,
            aq_lambda_scale: -8.0, // Aggressive coupling
            base_lambda_scale1: 14.75,
            dc_enabled: false,
            max_adjustment: 1.0, // Cap protects screenshots
            ..Self::default()
        }
    }

    /// Quality boost for perceptually better encoding.
    ///
    /// Uses positive coupling to preserve more detail in textured areas.
    /// Results: ~3-4% larger files, ~2-3% better DSSIM.
    pub fn quality_boost() -> Self {
        Self {
            enabled: true,
            aq_lambda_scale: 4.0, // Positive = better quality
            base_lambda_scale1: 14.75,
            dc_enabled: false,
            max_adjustment: 0.0,
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

    /// Builder: enable multiplicative coupling (vs additive)
    pub fn multiplicative(mut self, enabled: bool) -> Self {
        self.multiplicative = enabled;
        self
    }

    /// Builder: set maximum absolute lambda adjustment
    pub fn max_adjustment(mut self, max: f32) -> Self {
        self.max_adjustment = max;
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

        // Clamp to max_adjustment if set
        if self.max_adjustment > 0.0 {
            adjustment = adjustment.clamp(-self.max_adjustment, self.max_adjustment);
        }

        adjustment
    }

    /// Convert to TrellisConfig for a specific block.
    pub fn to_trellis_config(
        self,
        aq_strength: f32,
        dampen: f32,
        is_chroma: bool,
    ) -> TrellisConfig {
        let adjustment = self.compute_lambda_adjustment(aq_strength, dampen, is_chroma);

        // Compute effective lambda_log_scale1
        let scale1 = if self.multiplicative {
            // Multiplicative: scale1 = base * (1 + aq * coupling)
            // Use smaller coupling values (e.g., 0.1 instead of 4.0) for similar effect
            // This provides proportional scaling: high-base-lambda blocks get larger adjustments
            self.base_lambda_scale1 * (1.0 + adjustment)
        } else {
            // Additive: scale1 = base + aq * coupling
            // Original behavior - absolute adjustment regardless of base value
            self.base_lambda_scale1 + adjustment
        };

        TrellisConfig::default()
            .ac_trellis(true)
            .dc_trellis(self.dc_enabled)
            .lambda_scales(scale1, self.base_lambda_scale2)
            .num_loops(self.num_loops)
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

// ============================================================================
// Core hybrid functions (from hybrid/core.rs)
// ============================================================================

/// Standard rate tables for trellis rate estimation.
///
/// Trellis quantization needs Huffman code lengths to estimate bit costs.
/// Using standard JPEG tables is a reasonable approximation when
/// optimized tables aren't available yet.
pub struct StandardRateTables {
    pub luma_ac: RateTable,
    pub chroma_ac: RateTable,
    pub luma_dc: RateTable,
    pub chroma_dc: RateTable,
}

impl StandardRateTables {
    /// Create standard rate tables for trellis.
    pub fn new() -> Self {
        Self {
            luma_ac: RateTable::standard_luma_ac(),
            chroma_ac: RateTable::standard_chroma_ac(),
            luma_dc: RateTable::standard_luma_dc(),
            chroma_dc: RateTable::standard_chroma_dc(),
        }
    }
}

impl Default for StandardRateTables {
    fn default() -> Self {
        Self::new()
    }
}

/// Scale a quantization table by adaptive quantization strength.
///
/// Higher AQ strength means the block is "less important" (busy/textured area),
/// so we can use coarser quantization (higher quant values = fewer bits).
///
/// # Arguments
/// * `base_quant` - Base quantization table (u16\[64\])
/// * `aq_strength` - Per-block AQ strength from jpegli (typically 0.0 to ~0.5)
///
/// # Returns
/// Scaled quantization table where higher AQ strength = higher quant values
pub fn scale_quant_by_aq(
    base_quant: &[u16; DCT_BLOCK_SIZE],
    aq_strength: f32,
) -> [u16; DCT_BLOCK_SIZE] {
    let mut scaled = [0u16; DCT_BLOCK_SIZE];
    // aq_strength typically ranges from 0.0 to ~0.5
    // strength=0 → multiplier=1.0 (no change)
    // strength=0.5 → multiplier=1.5 (50% coarser)
    let multiplier = 1.0 + aq_strength;

    use wide::f32x8;
    let mul = f32x8::splat(multiplier);
    let one = f32x8::splat(1.0);
    let max_val = f32x8::splat(255.0);

    for chunk in 0..8 {
        let k = chunk * 8;
        let base_f = f32x8::from([
            base_quant[k] as f32,
            base_quant[k + 1] as f32,
            base_quant[k + 2] as f32,
            base_quant[k + 3] as f32,
            base_quant[k + 4] as f32,
            base_quant[k + 5] as f32,
            base_quant[k + 6] as f32,
            base_quant[k + 7] as f32,
        ]);
        let val = (base_f * mul).round().max(one).min(max_val);
        let arr: [f32; 8] = val.into();
        for j in 0..8 {
            scaled[k + j] = arr[j] as u16;
        }
    }

    scaled
}

/// Convert f32 DCT coefficients to i32 for trellis quantization.
///
/// jpegli and mozjpeg use different quantization formulas:
/// - jpegli: quantized = round(DCT * 8 / quantval) (DCT at 1/64 scale)
/// - mozjpeg trellis: quantized = round(DCT / (8 * quantval))
///
/// To make trellis produce the same quantized values as jpegli:
/// - We multiply DCT by 64: trellis sees round((64*DCT) / (8*quantval)) = round(DCT*8 / quantval)
/// - This compensates for both the 1/64 DCT scaling and the trellis's 8× divisor
pub fn dct_f32_to_i32(coeffs: &[f32; DCT_BLOCK_SIZE]) -> [i32; DCT_BLOCK_SIZE] {
    let mut result = [0i32; DCT_BLOCK_SIZE];

    use wide::f32x8;
    let scale = f32x8::splat(64.0);

    for chunk in 0..8 {
        let k = chunk * 8;
        let v = f32x8::from([
            coeffs[k],
            coeffs[k + 1],
            coeffs[k + 2],
            coeffs[k + 3],
            coeffs[k + 4],
            coeffs[k + 5],
            coeffs[k + 6],
            coeffs[k + 7],
        ]);
        let scaled = (v * scale).round();
        let arr: [f32; 8] = scaled.into();
        for j in 0..8 {
            result[k + j] = arr[j] as i32;
        }
    }

    result
}

/// Hybrid quantization: jpegli AQ + mozjpeg trellis.
///
/// Runs trellis quantization with a pre-configured lambda. The caller
/// (typically [`HybridQuantContext::quantize_block`]) is responsible for
/// computing the AQ-adjusted `TrellisConfig` via
/// [`HybridConfig::to_trellis_config`] before calling this function.
///
/// # Arguments
/// * `dct_coeffs` - DCT coefficients in f32 (jpegli format)
/// * `base_quant` - Base quantization table
/// * `ac_table` - Huffman table for rate estimation
/// * `config` - Trellis configuration (already AQ-adjusted by caller)
///
/// # Returns
/// Quantized coefficients ready for entropy coding
pub fn hybrid_quantize_block(
    dct_coeffs: &[f32; DCT_BLOCK_SIZE],
    base_quant: &[u16; DCT_BLOCK_SIZE],
    ac_table: &RateTable,
    config: &TrellisConfig,
) -> [i16; DCT_BLOCK_SIZE] {
    // Convert f32 DCT to i32 (with 8x scaling to match trellis's 8x quant divisor)
    let dct_i32 = dct_f32_to_i32(dct_coeffs);

    // Run trellis quantization with caller-provided lambda config
    let mut quantized = [0i16; DCT_BLOCK_SIZE];
    trellis_quantize_block(&dct_i32, &mut quantized, base_quant, ac_table, config);

    quantized
}

/// Hybrid quantization without trellis (for comparison/testing).
///
/// Scales quant table by AQ but uses simple rounding instead of trellis.
/// This isolates the AQ scaling effect from trellis optimization.
pub fn hybrid_quantize_block_simple(
    dct_coeffs: &[f32; DCT_BLOCK_SIZE],
    base_quant: &[u16; DCT_BLOCK_SIZE],
    aq_strength: f32,
) -> [i16; DCT_BLOCK_SIZE] {
    // 1. Scale quant table by AQ strength
    let scaled_quant = scale_quant_by_aq(base_quant, aq_strength);

    // 2. Simple quantization (divide and round)
    let mut quantized = [0i16; DCT_BLOCK_SIZE];

    use wide::f32x8;
    for chunk in 0..8 {
        let k = chunk * 8;
        let dct = f32x8::from([
            dct_coeffs[k],
            dct_coeffs[k + 1],
            dct_coeffs[k + 2],
            dct_coeffs[k + 3],
            dct_coeffs[k + 4],
            dct_coeffs[k + 5],
            dct_coeffs[k + 6],
            dct_coeffs[k + 7],
        ]);
        let q = f32x8::from([
            scaled_quant[k] as f32,
            scaled_quant[k + 1] as f32,
            scaled_quant[k + 2] as f32,
            scaled_quant[k + 3] as f32,
            scaled_quant[k + 4] as f32,
            scaled_quant[k + 5] as f32,
            scaled_quant[k + 6] as f32,
            scaled_quant[k + 7] as f32,
        ]);
        // DCT uses 1/64 scaling (matching C++), so multiply by 8/quant
        let eight = f32x8::splat(8.0);
        let val = (dct * eight / q).round();
        let arr: [f32; 8] = val.into();
        for j in 0..8 {
            quantized[k + j] = arr[j] as i16;
        }
    }

    quantized
}

// ============================================================================
// Encoder integration (from encode/hybrid.rs)
// ============================================================================

// ============================================================================
// Setup Helpers
// ============================================================================

/// Get the AQ map, using custom if provided or computing from Y plane.
#[inline]
pub(crate) fn get_aq_map_or_compute(
    config: &ComputedConfig,
    y_plane: &[f32],
    width: usize,
    height: usize,
    y_quant_01: u16,
) -> Result<AQStrengthMap> {
    if let Some(ref custom) = config.custom_aq_map {
        Ok(custom.clone())
    } else {
        Ok(crate::quant::aq::compute_aq_strength_map(
            y_plane, width, height, y_quant_01,
        )?)
    }
}

/// Create hybrid quantization context if enabled in config.
///
/// Priority:
/// 1. If `trellis` is set (mozjpeg-compat API), use it directly
/// 2. Else if `hybrid_config.enabled`, use hybrid AQ+trellis mode
/// 3. Else return None (no trellis quantization)
#[inline]
pub(crate) fn create_hybrid_ctx(config: &ComputedConfig) -> Option<HybridQuantContext> {
    // First check for explicit TrellisConfig (mozjpeg-compat API)
    if let Some(ref trellis) = config.trellis
        && trellis.is_enabled()
    {
        return Some(HybridQuantContext::from_trellis_config(*trellis));
    }

    // Fall back to HybridConfig
    if config.hybrid_config.enabled {
        Some(HybridQuantContext::new(config.hybrid_config))
    } else {
        None
    }
}

// ============================================================================
// Quantization Dispatch Helper
// ============================================================================

/// Quantize a block, dispatching to hybrid trellis or standard quantization.
///
/// This inline helper centralizes the hybrid vs non-hybrid dispatch logic.
/// When `hybrid_ctx` is Some, uses trellis quantization; otherwise uses
/// standard zero-bias quantization.
#[inline]
pub(crate) fn quantize_block_dispatch(
    dct: &[f32; DCT_BLOCK_SIZE],
    quant_values: &[u16; DCT_BLOCK_SIZE],
    zero_bias: &ZeroBiasParams,
    aq_strength: f32,
    is_luma: bool,
    hybrid_ctx: Option<&HybridQuantContext>,
) -> [i16; DCT_BLOCK_SIZE] {
    if let Some(ctx) = hybrid_ctx {
        ctx.quantize_block(dct, quant_values, aq_strength, 1.0, is_luma)
    } else {
        quant::quantize_block_with_zero_bias_simd(dct, quant_values, zero_bias, aq_strength)
    }
}

// ============================================================================
// Hybrid Quantization Context
// ============================================================================

/// Mode for trellis quantization.
enum TrellisMode {
    /// Hybrid mode: jpegli AQ + mozjpeg trellis with AQ-adjusted lambda
    Hybrid(HybridConfig),
    /// Standalone mode: pure mozjpeg-style trellis (no AQ lambda adjustment)
    Standalone(TrellisConfig),
}

/// Quantization context for trellis mode.
///
/// This struct holds pre-built Huffman tables and trellis config for use
/// during trellis quantization. Supports two modes:
///
/// - **Hybrid**: Combines jpegli's AQ with mozjpeg's trellis, adjusting lambda per-block
/// - **Standalone**: Pure mozjpeg-style trellis with fixed lambda (no AQ adjustment)
pub(crate) struct HybridQuantContext {
    rate_tables: StandardRateTables,
    mode: TrellisMode,
}

impl std::fmt::Debug for HybridQuantContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mode_str = match &self.mode {
            TrellisMode::Hybrid(_) => "Hybrid",
            TrellisMode::Standalone(_) => "Standalone",
        };
        f.debug_struct("HybridQuantContext")
            .field("mode", &mode_str)
            .finish_non_exhaustive()
    }
}

impl HybridQuantContext {
    /// Creates a new hybrid quantization context (AQ + trellis).
    pub(crate) fn new(config: HybridConfig) -> Self {
        Self {
            rate_tables: StandardRateTables::new(),
            mode: TrellisMode::Hybrid(config),
        }
    }

    /// Creates a standalone trellis context (mozjpeg-compatible, no AQ adjustment).
    pub(crate) fn from_trellis_config(config: TrellisConfig) -> Self {
        Self {
            rate_tables: StandardRateTables::new(),
            mode: TrellisMode::Standalone(config),
        }
    }

    /// Quantize a block using trellis quantization.
    ///
    /// # Arguments
    /// * `dct_coeffs` - DCT coefficients
    /// * `quant` - Quantization table
    /// * `aq_strength` - Per-block AQ strength (used in hybrid mode)
    /// * `dampen` - Quality-based AQ dampen factor (0-1, used in hybrid mode)
    /// * `is_luma` - True for Y component, false for Cb/Cr
    pub(crate) fn quantize_block(
        &self,
        dct_coeffs: &[f32; DCT_BLOCK_SIZE],
        quant: &[u16; DCT_BLOCK_SIZE],
        aq_strength: f32,
        dampen: f32,
        is_luma: bool,
    ) -> [i16; DCT_BLOCK_SIZE] {
        let ac_table = if is_luma {
            &self.rate_tables.luma_ac
        } else {
            &self.rate_tables.chroma_ac
        };

        // Generate trellis config based on mode
        let trellis_config = match &self.mode {
            TrellisMode::Hybrid(hybrid_config) => {
                // Hybrid mode: lambda adjusted by AQ strength in to_trellis_config()
                hybrid_config.to_trellis_config(aq_strength, dampen, !is_luma)
            }
            TrellisMode::Standalone(trellis_config) => {
                // Standalone mode: pure mozjpeg-compatible trellis, no AQ influence
                *trellis_config
            }
        };

        hybrid_quantize_block(dct_coeffs, quant, ac_table, &trellis_config)
    }

    /// Returns true if DC trellis optimization is enabled.
    pub(crate) fn is_dc_trellis_enabled(&self) -> bool {
        match &self.mode {
            TrellisMode::Hybrid(config) => config.dc_enabled,
            TrellisMode::Standalone(config) => config.is_dc_enabled(),
        }
    }

    /// Returns the base trellis configuration (for DC trellis lambda parameters).
    pub(crate) fn trellis_config(&self) -> TrellisConfig {
        match &self.mode {
            TrellisMode::Hybrid(config) => config.to_trellis_config(0.0, 1.0, false),
            TrellisMode::Standalone(config) => *config,
        }
    }

    /// Returns the luma DC rate table for DC trellis optimization.
    pub(crate) fn luma_dc_rate_table(&self) -> &RateTable {
        &self.rate_tables.luma_dc
    }

    /// Returns the chroma DC rate table for DC trellis optimization.
    pub(crate) fn chroma_dc_rate_table(&self) -> &RateTable {
        &self.rate_tables.chroma_dc
    }
}

// ============================================================================
// XYB Block Quantization with Hybrid Trellis
// ============================================================================

/// Quantizes all XYB blocks with adaptive quantization and optional hybrid trellis.
///
/// This version uses the AQ map for per-block modulation and applies
/// hybrid trellis quantization when enabled via the HybridQuantContext.
///
/// For XYB mode:
/// - X and Y use luma tables (both are full-resolution "luma-like" channels)
/// - B uses chroma tables (downsampled blue channel)
#[allow(clippy::too_many_arguments)]
pub(crate) fn quantize_all_blocks_xyb_with_aq(
    x_plane: &[f32],
    y_plane: &[f32],
    b_plane: &[f32], // Already downsampled
    width: usize,
    height: usize,
    b_width: usize,
    b_height: usize,
    x_quant: &QuantTable,
    y_quant: &QuantTable,
    b_quant: &QuantTable,
    aq_map: &AQStrengthMap,
    hybrid_ctx: Option<&HybridQuantContext>,
) -> crate::error::Result<(
    Vec<[i16; DCT_BLOCK_SIZE]>,
    Vec<[i16; DCT_BLOCK_SIZE]>,
    Vec<[i16; DCT_BLOCK_SIZE]>,
)> {
    // MCU size for 2×2, 2×2, 1×1 sampling: 16×16 pixels
    let mcu_cols = (width + 15) / 16;
    let mcu_rows = (height + 15) / 16;
    let num_xy_blocks = mcu_cols * mcu_rows * 4; // 4 blocks per MCU for X and Y
    let num_b_blocks = mcu_cols * mcu_rows; // 1 block per MCU for B

    // Pre-allocate block arrays to avoid push() overhead
    let mut x_blocks = crate::foundation::alloc::try_alloc_dct_blocks(num_xy_blocks, "x_blocks")?;
    let mut y_blocks = crate::foundation::alloc::try_alloc_dct_blocks(num_xy_blocks, "y_blocks")?;
    let mut b_blocks = crate::foundation::alloc::try_alloc_dct_blocks(num_b_blocks, "b_blocks")?;

    for mcu_y in 0..mcu_rows {
        for mcu_x in 0..mcu_cols {
            let mcu_idx = mcu_y * mcu_cols + mcu_x;
            let xy_base = mcu_idx * 4; // 4 blocks per MCU for X and Y

            // Process 4 X blocks (2×2 arrangement within 16×16 MCU)
            for block_y in 0..2 {
                for block_x in 0..2 {
                    let bx = mcu_x * 2 + block_x;
                    let by = mcu_y * 2 + block_y;
                    let block_offset = block_y * 2 + block_x;
                    let aq_strength = aq_map.get(bx, by);

                    let x_block =
                        crate::encode_simd::extract_block_xyb_simd(x_plane, width, height, bx, by);
                    let x_dct = forward_dct_8x8(&x_block);

                    // X is luma-like in XYB, dampen=1.0
                    let x_quant_coeffs = if let Some(ctx) = hybrid_ctx {
                        ctx.quantize_block(&x_dct, &x_quant.values, aq_strength, 1.0, true)
                    } else {
                        quant::quantize_block(&x_dct, &x_quant.values)
                    };
                    natural_to_zigzag_into(&x_quant_coeffs, &mut x_blocks[xy_base + block_offset]);
                }
            }

            // Process 4 Y blocks (2×2 arrangement within 16×16 MCU)
            for block_y in 0..2 {
                for block_x in 0..2 {
                    let bx = mcu_x * 2 + block_x;
                    let by = mcu_y * 2 + block_y;
                    let block_offset = block_y * 2 + block_x;
                    let aq_strength = aq_map.get(bx, by);

                    let y_block =
                        crate::encode_simd::extract_block_xyb_simd(y_plane, width, height, bx, by);
                    let y_dct = forward_dct_8x8(&y_block);

                    // Y is the primary luma channel in XYB, dampen=1.0
                    let y_quant_coeffs = if let Some(ctx) = hybrid_ctx {
                        ctx.quantize_block(&y_dct, &y_quant.values, aq_strength, 1.0, true)
                    } else {
                        quant::quantize_block(&y_dct, &y_quant.values)
                    };
                    natural_to_zigzag_into(&y_quant_coeffs, &mut y_blocks[xy_base + block_offset]);
                }
            }

            // Process 1 B block (from downsampled plane)
            // Average AQ from the 4 corresponding full-res blocks
            let b_aq_strength = {
                let mut sum = 0.0f32;
                for dy in 0..2 {
                    for dx in 0..2 {
                        let bx = mcu_x * 2 + dx;
                        let by = mcu_y * 2 + dy;
                        sum += aq_map.get(bx, by);
                    }
                }
                sum / 4.0
            };

            let b_block = crate::encode_simd::extract_block_xyb_simd(
                b_plane, b_width, b_height, mcu_x, mcu_y,
            );
            let b_dct = forward_dct_8x8(&b_block);

            // B is chroma-like (blue channel), is_luma=false
            let b_quant_coeffs = if let Some(ctx) = hybrid_ctx {
                ctx.quantize_block(&b_dct, &b_quant.values, b_aq_strength, 1.0, false)
            } else {
                quant::quantize_block(&b_dct, &b_quant.values)
            };
            natural_to_zigzag_into(&b_quant_coeffs, &mut b_blocks[mcu_idx]);
        }
    }

    Ok((x_blocks, y_blocks, b_blocks))
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

    #[test]
    fn test_aggressive_compression_preset() {
        let config = HybridConfig::aggressive_compression();
        assert!(config.enabled);
        assert_eq!(config.aq_lambda_scale, -4.0); // Negative for smaller files
        assert_eq!(config.max_adjustment, 0.0); // No cap

        // Negative coupling produces negative adjustment
        let adj = config.compute_lambda_adjustment(0.5, 1.0, false);
        assert_eq!(adj, -2.0); // 0.5 * -4.0 = -2.0
    }

    #[test]
    fn test_safe_compression_preset() {
        let config = HybridConfig::safe_compression();
        assert!(config.enabled);
        assert_eq!(config.aq_lambda_scale, -8.0); // More aggressive coupling
        assert_eq!(config.max_adjustment, 1.0); // Cap at ±1.0

        // High AQ would produce -4.0 but capped to -1.0
        let adj = config.compute_lambda_adjustment(0.5, 1.0, false);
        assert_eq!(adj, -1.0); // Clamped from 0.5 * -8.0 = -4.0

        // Low AQ produces smaller adjustment (not capped)
        let adj_low = config.compute_lambda_adjustment(0.1, 1.0, false);
        assert_eq!(adj_low, -0.8); // 0.1 * -8.0 = -0.8, within ±1.0
    }

    #[test]
    fn test_quality_boost_preset() {
        let config = HybridConfig::quality_boost();
        assert!(config.enabled);
        assert_eq!(config.aq_lambda_scale, 4.0); // Positive for better quality

        // Positive coupling produces positive adjustment (larger files, better quality)
        let adj = config.compute_lambda_adjustment(0.5, 1.0, false);
        assert_eq!(adj, 2.0); // 0.5 * 4.0 = 2.0
    }

    #[test]
    fn test_max_adjustment_clamping() {
        let config = HybridConfig::new()
            .aq_lambda_scale(-10.0)
            .max_adjustment(2.0);

        // Very negative coupling clamped to -2.0
        let adj = config.compute_lambda_adjustment(0.5, 1.0, false);
        assert_eq!(adj, -2.0); // Clamped from 0.5 * -10.0 = -5.0

        // Also clamps positive
        let config_pos = HybridConfig::new()
            .aq_lambda_scale(10.0)
            .max_adjustment(2.0);
        let adj_pos = config_pos.compute_lambda_adjustment(0.5, 1.0, false);
        assert_eq!(adj_pos, 2.0); // Clamped from 0.5 * 10.0 = 5.0
    }

    #[test]
    fn test_multiplicative_coupling() {
        let config = HybridConfig::new()
            .aq_lambda_scale(0.1) // Use small value for multiplicative
            .multiplicative(true);

        // Multiplicative: scale1 = base * (1 + aq * coupling)
        // With aq=0.5 and coupling=0.1: adjustment = 0.5 * 0.1 = 0.05
        // scale1 = 14.75 * (1 + 0.05) = 14.75 * 1.05 = 15.4875
        let trellis = config.to_trellis_config(0.5, 1.0, false);
        let expected = 14.75 * 1.05;
        assert!((trellis.lambda_log_scale1 - expected).abs() < 0.001);

        // Compare to additive
        let config_add = HybridConfig::new().aq_lambda_scale(0.1);
        let trellis_add = config_add.to_trellis_config(0.5, 1.0, false);
        // Additive: scale1 = 14.75 + 0.05 = 14.80
        assert!((trellis_add.lambda_log_scale1 - 14.80).abs() < 0.001);
    }

    #[test]
    fn test_negative_coupling_with_threshold() {
        let config = HybridConfig::new().aq_lambda_scale(-4.0).aq_threshold(0.2);

        // Below threshold: no adjustment
        let adj_low = config.compute_lambda_adjustment(0.15, 1.0, false);
        assert_eq!(adj_low, 0.0);

        // Above threshold: full adjustment
        let adj_high = config.compute_lambda_adjustment(0.3, 1.0, false);
        assert_eq!(adj_high, -1.2); // 0.3 * -4.0
    }

    #[test]
    fn test_chroma_scale_with_negative_coupling() {
        let config = HybridConfig::new().aq_lambda_scale(-4.0).chroma_scale(0.5);

        // Luma: full adjustment
        let adj_luma = config.compute_lambda_adjustment(0.5, 1.0, false);
        assert_eq!(adj_luma, -2.0);

        // Chroma: half adjustment
        let adj_chroma = config.compute_lambda_adjustment(0.5, 1.0, true);
        assert_eq!(adj_chroma, -1.0);
    }

    // ========================================================================
    // Image Type Detection Tests
    // ========================================================================

    #[test]
    fn test_detect_photo() {
        // flower_small stats: mean=0.090, std=0.065, cv=0.72
        assert_eq!(detect_image_type(0.090, 0.065), ImageType::Photo);

        // Higher texture photo
        assert_eq!(detect_image_type(0.10, 0.05), ImageType::Photo);
    }

    #[test]
    fn test_detect_screenshot() {
        // apple.com stats: mean=0.084, std=0.198, cv=2.35 (very high CV)
        assert_eq!(detect_image_type(0.084, 0.198), ImageType::Screenshot);

        // Very flat UI
        assert_eq!(detect_image_type(0.02, 0.01), ImageType::Screenshot);

        // Zero mean (edge case)
        assert_eq!(detect_image_type(0.0, 0.0), ImageType::Screenshot);

        // High CV even with moderate mean = screenshot
        assert_eq!(detect_image_type(0.08, 0.16), ImageType::Screenshot); // CV=2.0
    }

    #[test]
    fn test_detect_mixed() {
        // Ambiguous: low mean, low CV (neither clearly photo nor screenshot)
        assert_eq!(detect_image_type(0.04, 0.02), ImageType::Mixed); // mean<0.06, CV=0.5
    }

    #[test]
    fn test_adaptive_config_low_texture_photo() {
        // Low texture photo (mean=0.10): aggressive coupling
        let config = adaptive_config(0.10, 0.05);
        assert_eq!(config.aq_lambda_scale, -4.0); // Full aggressive
        assert_eq!(config.max_adjustment, 0.0);
    }

    #[test]
    fn test_adaptive_config_medium_texture_photo() {
        // Medium texture photo (mean=0.30): moderate coupling
        let config = adaptive_config(0.30, 0.15);
        assert!((config.aq_lambda_scale - (-2.0)).abs() < 0.1); // -4.0 * (0.15/0.30) = -2.0
        assert_eq!(config.max_adjustment, 0.0);
    }

    #[test]
    fn test_adaptive_config_high_texture_photo() {
        // High texture photo (mean=0.60): gentle coupling
        let config = adaptive_config(0.60, 0.25);
        assert!((config.aq_lambda_scale - (-1.0)).abs() < 0.1); // -4.0 * (0.15/0.60) = -1.0
        assert_eq!(config.max_adjustment, 0.0);
    }

    #[test]
    fn test_adaptive_config_screenshot() {
        let config = adaptive_config(0.084, 0.198);
        // Screenshots get safe compression (with protection)
        assert_eq!(config.aq_lambda_scale, -8.0);
        assert_eq!(config.max_adjustment, 1.0);
    }

    #[test]
    fn test_adaptive_config_mixed() {
        let config = adaptive_config(0.04, 0.02);
        // Mixed gets safe settings (same as screenshot)
        assert!(config.max_adjustment > 0.0);
    }

    #[test]
    fn test_texture_adaptive_coupling() {
        // Low texture: full aggressive
        assert_eq!(texture_adaptive_coupling(0.10), -4.0);
        assert_eq!(texture_adaptive_coupling(0.15), -4.0);

        // Medium texture: scaled down
        assert!((texture_adaptive_coupling(0.30) - (-2.0)).abs() < 0.01);

        // High texture: gentler
        assert!((texture_adaptive_coupling(0.60) - (-1.0)).abs() < 0.01);
        assert!((texture_adaptive_coupling(0.75) - (-0.8)).abs() < 0.01);
    }

    #[test]
    fn test_scale_quant_by_aq() {
        let base = [16u16; 64];

        // Zero AQ strength = no change
        let scaled = scale_quant_by_aq(&base, 0.0);
        assert_eq!(scaled[0], 16);

        // 0.5 AQ strength = 1.5x
        let scaled = scale_quant_by_aq(&base, 0.5);
        assert_eq!(scaled[0], 24);

        // 1.0 AQ strength = 2x
        let scaled = scale_quant_by_aq(&base, 1.0);
        assert_eq!(scaled[0], 32);
    }

    #[test]
    fn test_scale_quant_clamping() {
        let base = [200u16; 64];

        // Should clamp at 255
        let scaled = scale_quant_by_aq(&base, 0.5);
        assert_eq!(scaled[0], 255);

        let base_low = [1u16; 64];
        // Should never go below 1
        let scaled = scale_quant_by_aq(&base_low, 0.0);
        assert_eq!(scaled[0], 1);
    }

    #[test]
    fn test_dct_f32_to_i32() {
        // Function multiplies by 64 for trellis compatibility (see docstring)
        // jpegli DCT is at 1/64 scale, trellis divides by 8*quant
        // So multiply by 64 to compensate: trellis sees round(64*DCT / (8*q)) = round(DCT*8/q)
        // 127.4 * 64 = 8153.6, rounds to 8154
        let f32_coeffs = [127.4f32; 64];
        let i32_coeffs = dct_f32_to_i32(&f32_coeffs);
        assert_eq!(i32_coeffs[0], 8154);

        // -127.6 * 64 = -8166.4, rounds to -8166
        let f32_coeffs = [-127.6f32; 64];
        let i32_coeffs = dct_f32_to_i32(&f32_coeffs);
        assert_eq!(i32_coeffs[0], -8166);
    }

    #[test]
    fn test_hybrid_quantize_simple() {
        // hybrid_quantize_block_simple uses jpegli's formula: round(DCT * 8 / quant)
        // DC coefficient of 1024 with quant=16 and no AQ
        let mut dct = [0.0f32; 64];
        dct[0] = 1024.0;
        let base_quant = [16u16; 64];

        let quantized = hybrid_quantize_block_simple(&dct, &base_quant, 0.0);
        assert_eq!(quantized[0], 512); // 1024 * 8 / 16 = 512

        // With AQ strength 0.5, quant becomes 24
        let quantized = hybrid_quantize_block_simple(&dct, &base_quant, 0.5);
        assert_eq!(quantized[0], 341); // 1024 * 8 / 24 = 341.33 -> 341
    }
}
