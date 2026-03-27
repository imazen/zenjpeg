//! Expert configuration for external optimization.
//!
//! [`ExpertConfig`] flattens all quality/size-affecting encoder parameters into a
//! single struct with no overlapping fields. External optimizers (simulated annealing,
//! Bayesian search, etc.) can mutate fields directly and call
//! [`to_encoder_config()`](ExpertConfig::to_encoder_config) to encode.
//!
//! # Design
//!
//! The encoder has 4 separate config types ([`EncodingTables`], [`TrellisConfig`],
//! [`HybridConfig`], [`EncoderConfig`]) with overlapping fields and different
//! visibility. `ExpertConfig` eliminates this overlap:
//!
//! - **One set of trellis parameters** (not duplicated between `TrellisConfig` and
//!   `HybridConfig`). The `aq_trellis_coupling` field controls the transition:
//!   `0.0` = standalone mozjpeg trellis, `> 0.0` = AQ-coupled hybrid mode.
//! - **All fields `pub`** for direct mutation by optimizers.
//! - **No mode booleans** for hybrid vs standalone — coupling strength is the control.
//!
//! # Standalone vs Hybrid Trellis
//!
//! When `trellis_enabled == true`, the mode depends on `aq_trellis_coupling`:
//!
//! - **`== 0.0` (standalone):** Produces a [`TrellisConfig`]. All `trellis_*` fields
//!   are forwarded. This matches C mozjpeg behavior.
//! - **`> 0.0` (hybrid):** Produces a [`HybridConfig`]. The hybrid path creates
//!   per-block trellis configs internally via [`HybridConfig::to_trellis_config()`],
//!   which does **not** forward all fields. See the "Hybrid-Mode Limitations" section
//!   on [`ExpertConfig`] for which fields are ignored.
//!
//! # Parameter Count
//!
//! ~30 direct fields, controlling ~481 f32-representable parameters:
//! - 192 quant table values (3 components x 64 coefficients) — in `tables.quant`
//! - 192 zero-bias multipliers (blended HQ/LQ) — in `tables.zero_bias_mul`
//! - 6 zero-bias offsets (3 DC + 3 AC) — in `tables.zero_bias_offset_*`
//! - 65 scaling params (1 global_scale + 64 frequency_exponents) — in `tables.scaling`
//!   (only when `ScalingParams::Scaled`; absent for `Exact`)
//! - 384 zero-bias blend endpoints (192 HQ + 192 LQ) — in `zero_bias_hq/lq`
//! - 2 zero-bias blend distances
//! - 9 trellis params
//! - 5 AQ-trellis coupling params
//! - 5 encoder flags (scan_mode, deringing, allow_16bit, downsampling, quality)
//!
//! # Usage
//!
//! ```rust,ignore
//! use zenjpeg::encode::{ExpertConfig, OptimizationPreset, ColorMode, ChromaSubsampling};
//!
//! // Start from a preset
//! let mut expert = ExpertConfig::from_preset(OptimizationPreset::HybridProgressive, 85.0);
//!
//! // Mutate for optimization
//! expert.trellis_lambda_log_scale1 = 15.0;
//! expert.aq_trellis_coupling = 2.0;
//! expert.tables.quant.scale_component(0, 1.05); // bump luma quant 5%
//!
//! // Must re-blend if quality or zero-bias endpoints changed
//! // (not needed here since we only changed trellis/quant params)
//!
//! // Convert to encoder config
//! let color = ColorMode::YCbCr { subsampling: ChromaSubsampling::Quarter };
//! let enc_config = expert.to_encoder_config(color);
//! ```

use super::encoder_config::EncoderConfig;
use super::encoder_types::{
    ColorMode, DownsamplingMethod, HuffmanStrategy, OptimizationPreset, ProgressiveScanMode,
    Quality, QuantTableConfig,
};
use super::trellis::HybridConfig;
use super::trellis::{TrellisConfig, TrellisSpeedMode};
use super::tuning::{EncodingTables, PerComponent};

/// All tunable encoder parameters for external optimization.
///
/// Every field is `pub` for direct mutation. Use [`from_preset()`](Self::from_preset)
/// for known-good starting points, mutate freely, then call
/// [`to_encoder_config()`](Self::to_encoder_config) to encode.
///
/// # Measured Parameter Impact (256x256 noise+patches, MozjpegBaseline Q85)
///
/// Parameters are grouped by measured file size impact. Parameters marked DEAD
/// have zero effect on output.
///
/// ## Parameters that affect file size
///
/// | Parameter | Range tested | Impact |
/// |-----------|-------------|--------|
/// | `tables.quant` (64×3 values) | 0.5x – 2.0x | -54% to +65% |
/// | `trellis_enabled` | on/off | ~15% savings when on |
/// | `trellis_lambda_log_scale1` | 12.0 – 17.0 | -46% to +12% (exponential) |
/// | `trellis_lambda_log_scale2` | 14.0 – 18.0 | -19% to +11% (inverse of scale1) |
/// | `scan_mode` | 4 variants | up to -2% (ProgressiveSearch best) |
/// | `zero_bias_mul` (jpegli only) | 0.0 – 1.0 | -14% to +31% |
/// | `trellis_delta_dc_weight` | 0.0 – 5.0 | 0% to +1% |
/// | `trellis_dc_enabled` | on/off | ~0.1% |
/// | `downsampling_method` | 3 variants | ±0.2% |
/// | `quality` (Scaled tables only) | 50 – 95 | -81% to +112% |
///
/// ## DEAD parameters (zero file size effect)
///
/// | Parameter | Why dead |
/// |-----------|----------|
/// | `trellis_use_lambda_weight_tbl` | Always uses flat 1/q² weights regardless of flag |
/// | `trellis_speed_mode` | Only limits search candidates; DP finds same optimum |
/// | `trellis_num_loops` | Stored but never read — single-pass only |
/// | `quality` (Exact tables) | Tables pre-scaled; zero-bias all zeros for mozjpeg |
/// | `allow_16bit_quant_tables` | No effect unless quant values > 255 |
/// | `deringing` | Only affects images with saturated (255) pixels near edges |
///
/// ## Hybrid mode (`aq_trellis_coupling != 0`)
///
/// Hybrid mode adjusts trellis lambda per-block based on AQ strength:
/// - **Positive coupling**: Higher lambda for textured blocks → better quality, larger files
/// - **Negative coupling**: Lower lambda for textured blocks → smaller files, worse quality
///
/// Recommended settings:
/// - For size optimization: `aq_trellis_coupling = -4.0` (additive) gives ~2% size reduction
/// - For quality: `aq_trellis_coupling = +4.0` gives ~2-3% better DSSIM
/// - For screenshots/text: use conservative values (-1 to +1) as they're more sensitive
///
/// The `aq_trellis_multiplicative` field switches between:
/// - Additive (default): `scale1 = base + aq * coupling`
/// - Multiplicative: `scale1 = base * (1 + aq * coupling)`
///
/// Note: Multiplicative is MORE aggressive on high-AQ images, not safer.
///
/// # Parameters for optimizers
///
/// The only parameters worth tuning for file size optimization are:
/// 1. `tables.quant` — 192 values, by far the largest effect
/// 2. `trellis_lambda_log_scale1` — single float, exponential effect
/// 3. `trellis_lambda_log_scale2` — single float, inverse relationship to scale1
/// 4. `zero_bias_mul` (jpegli presets) — 192 values, large effect
/// 5. `scan_mode` — categorical, up to 2% savings
/// 6. `trellis_enabled` — binary, ~15% savings
/// 7. `quality` (Scaled tables) — controls both quant scaling and zero-bias blend
#[derive(Clone, Debug)]
pub struct ExpertConfig {
    // === Quantization Tables ===
    /// Base quantization tables, zero-bias multipliers/offsets, and quality scaling.
    ///
    /// **Largest impact on file size.** Halving quant values = +65% size;
    /// doubling = -54% size. The 192 values in `quant` (3 × 64) are the primary
    /// optimization target.
    ///
    /// The `zero_bias_mul` field is also significant for jpegli presets: all-zeros
    /// = +31% size, all-ones = -14% (more aggressive rounding toward zero).
    /// For mozjpeg presets, `zero_bias_mul` is always all-zeros (standard rounding).
    ///
    /// When `tables.scaling == ScalingParams::Exact` (mozjpeg presets), quant values
    /// are pre-scaled to quality and used as-is. The `quality` field has **no effect**
    /// — not even on zero-bias, since mozjpeg zero-bias is all-zeros.
    ///
    /// When `tables.scaling == ScalingParams::Scaled` (jpegli/hybrid presets),
    /// the encoder applies per-frequency non-linear scaling and `quality` controls
    /// both quant scaling and zero-bias blend.
    pub tables: EncodingTables,

    // === Zero-Bias Blend Control ===
    // Only active for jpegli/hybrid presets (which have non-zero zero_bias_mul).
    // Mozjpeg presets use all-zeros zero-bias, making this entire section no-op.
    /// HQ zero-bias multiplier tables (endpoint at high quality / low distance).
    ///
    /// Blended with LQ tables based on quality distance. **Only matters for
    /// jpegli/hybrid presets** — mozjpeg presets have zero-bias = all zeros.
    ///
    /// Default: C++ jpegli's `kZeroBiasMulYCbCrHQ` tables.
    /// Ignored in XYB mode (uses uniform 0.5).
    pub zero_bias_hq: PerComponent<[f32; 64]>,

    /// LQ zero-bias multiplier tables (endpoint at low quality / high distance).
    ///
    /// Default: C++ jpegli's `kZeroBiasMulYCbCrLQ` tables.
    /// Ignored in XYB mode.
    pub zero_bias_lq: PerComponent<[f32; 64]>,

    /// Distance at or below which zero-bias is fully HQ.
    ///
    /// **Small impact** (~0.1% when widened). Default: `1.0`.
    /// Ignored in XYB mode and mozjpeg presets.
    pub zero_bias_hq_distance: f32,

    /// Distance at or above which zero-bias is fully LQ.
    ///
    /// **Up to ~2% impact** when set = hq_distance (forces one endpoint).
    /// Default: `3.0`. Ignored in XYB mode and mozjpeg presets.
    pub zero_bias_lq_distance: f32,

    // === Trellis Quantization ===
    /// Master switch for trellis quantization.
    ///
    /// **~15% file size savings** when enabled (measured on 256x256 noise+patches).
    /// When `false`, all other `trellis_*` and `aq_trellis_*` fields are ignored
    /// and the encoder uses simple rounding (jpegli default).
    pub trellis_enabled: bool,

    /// Enable DC coefficient trellis (cross-block DC optimization).
    ///
    /// **Measured impact: ~0.1%.** Tiny but real. Forwarded in standalone mode.
    ///
    /// Default: `true`.
    pub trellis_dc_enabled: bool,

    /// Use perceptual lambda weighting table.
    ///
    /// **DEAD PARAMETER.** The trellis always uses flat 1/q² weights regardless
    /// of this flag. The CSF-based weighting from C mozjpeg mode=1 was never
    /// implemented; mode=1 hardcodes flat weights. Toggling has zero effect.
    ///
    /// Default: `true`.
    pub trellis_use_lambda_weight_tbl: bool,

    /// Lambda log scale 1 (rate penalty). Higher = more aggressive quantization.
    ///
    /// **Huge impact: -46% to +12%.** The effective lambda is
    /// `2^scale1 / (2^scale2 + block_norm)`. This is the most impactful single
    /// float for trellis optimization.
    ///
    /// Default: `14.75`. Useful range: `12.0`–`17.0`.
    /// Below 12.0, trellis zeroes out nearly all AC coefficients.
    /// Above 17.0, approaches no-trellis behavior.
    pub trellis_lambda_log_scale1: f32,

    /// Lambda log scale 2 (distortion sensitivity). Higher = more quality.
    ///
    /// **Large impact: -19% to +11%.** Controls the denominator in the lambda
    /// formula. Relationship with scale1: reducing scale2 has a similar effect
    /// to increasing scale1 (both make lambda larger = more aggressive zeroing).
    ///
    /// Default: `16.5`. Useful range: `14.0`–`18.0`.
    pub trellis_lambda_log_scale2: f32,

    /// Number of trellis optimization loops.
    ///
    /// **DEAD PARAMETER.** Stored but never read by the trellis engine.
    /// The encoder always performs a single pass. Values 1–5 all produce
    /// identical output.
    ///
    /// Default: `1`.
    pub trellis_num_loops: i32,

    /// Speed optimization mode for high-entropy blocks.
    ///
    /// **No effect on file size** — only affects encoding speed. All modes
    /// (Adaptive, Thorough, Level(1)–Level(8)) produce identical output bytes.
    /// The dynamic programming finds the same optimum regardless of search
    /// bounds; tighter bounds just find it faster.
    ///
    /// Default: `TrellisSpeedMode::Adaptive`.
    pub trellis_speed_mode: TrellisSpeedMode,

    /// Weight for vertical DC gradient penalty.
    ///
    /// **Tiny impact: 0% to +1%.** When > 0.0, DC trellis penalizes large
    /// vertical DC jumps between blocks, reducing visible banding artifacts
    /// at the cost of slightly larger files.
    ///
    /// Default: `0.0` (disabled, matching C mozjpeg default).
    /// Useful range: `0.0`–`5.0` (diminishing returns above 2.0).
    pub trellis_delta_dc_weight: f32,

    // === AQ->Trellis Coupling (Hybrid Mode) ===
    // These fields control hybrid mode: AQ-adjusted per-block lambda for trellis.
    // Set `aq_trellis_coupling != 0` to enable. See struct-level docs for details.
    /// Per-unit AQ strength to lambda adjustment.
    ///
    /// `0.0` = standalone trellis (fixed lambda)
    /// `!= 0.0` = hybrid mode (lambda adjusted per-block based on AQ)
    ///
    /// Positive values: increase lambda for high-AQ blocks → better quality, larger files
    /// Negative values: decrease lambda for high-AQ blocks → smaller files, worse quality
    ///
    /// Default: `0.0`. Recommended range: `-8.0` to `+8.0`.
    pub aq_trellis_coupling: f32,

    /// Non-linear AQ mapping exponent.
    ///
    /// `1.0` = linear, `2.0` = emphasize high-AQ, `0.5` = compress AQ range.
    /// Default: `1.0`.
    pub aq_trellis_exponent: f32,

    /// Minimum AQ strength before coupling kicks in.
    ///
    /// Blocks with AQ below this use base lambda unchanged.
    /// Default: `0.0`.
    pub aq_trellis_threshold: f32,

    /// Scale coupling for chroma components (Cb, Cr).
    ///
    /// `1.0` = same as luma, `<1.0` = less aggressive on chroma.
    /// Default: `1.0`.
    pub aq_trellis_chroma_scale: f32,

    /// Scale coupling by quality-derived dampen factor.
    ///
    /// Default: `false`.
    pub aq_trellis_quality_adaptive: bool,

    /// Use multiplicative coupling instead of additive.
    ///
    /// Additive (default): `scale1 = base_scale1 + aq * coupling`
    /// Multiplicative: `scale1 = base_scale1 * (1 + aq * coupling)`
    ///
    /// Multiplicative provides proportional scaling (use smaller coupling values).
    /// Default: `false`.
    pub aq_trellis_multiplicative: bool,

    /// Maximum absolute lambda adjustment (clamps to [-max, +max]).
    ///
    /// `0.0` = no limit (default).
    /// Use to limit quality degradation on sensitive images (screenshots, text).
    /// Recommended: `1.0-2.0` for conservative size optimization.
    pub aq_trellis_max_adjustment: f32,

    // === Encoder Strategy ===
    /// Scan mode (baseline vs progressive variants).
    ///
    /// **Up to 2% savings.** Progressive modes outperform baseline:
    /// - `Baseline`: reference size
    /// - `Progressive`: -0.25%
    /// - `ProgressiveMozjpeg`: -1.6%
    /// - `ProgressiveSearch`: -2.0% (tries 64 candidate scan configs)
    ///
    /// Progressive modes automatically enable optimized Huffman tables.
    pub scan_mode: ProgressiveScanMode,

    /// Enable overshoot deringing.
    ///
    /// **Content-dependent.** Zero effect on images without saturated pixels
    /// (value 255) near edges. On photographic content with blown highlights
    /// or white backgrounds, reduces ringing artifacts with negligible size cost.
    ///
    /// Default: `true` for jpegli/hybrid, `false` for mozjpeg baseline/progressive.
    pub deringing: bool,

    /// Allow 16-bit quantization tables (SOF1 extended JPEG).
    ///
    /// **No effect at Q85+** (all quant values fit in 8 bits). Only matters
    /// below ~Q86 where chroma quant values exceed 255. When `false`, values
    /// are clamped to 255 for baseline compatibility (no quality impact — those
    /// high-frequency chroma coefficients quantize to zero regardless).
    ///
    /// Default: `false` (matching cjpegli CLI and C mozjpeg).
    pub allow_16bit_quant_tables: bool,

    /// Quality parameter.
    ///
    /// **Impact depends on scaling mode:**
    /// - `ScalingParams::Scaled` (jpegli/hybrid): Q50→Q95 = -81% to +112% size.
    ///   Controls both quant table scaling and zero-bias blend.
    /// - `ScalingParams::Exact` (mozjpeg): **zero effect.** Tables are pre-scaled
    ///   during `from_preset()`, and mozjpeg zero-bias is all-zeros so blend is no-op.
    ///
    /// Accepts `f32`, `u8`, `i32`, or explicit `Quality::*` variants.
    pub quality: Quality,

    /// Chroma downsampling method.
    ///
    /// **±0.2% impact.** Only affects RGB input with chroma subsampling (4:2:0, etc.).
    /// Ignored for 4:4:4, grayscale, and pre-subsampled YCbCr input.
    /// `GammaAware` and `GammaAwareIterative` may produce slightly different
    /// chroma, affecting compressibility marginally.
    pub downsampling_method: DownsamplingMethod,

    /// Enable adaptive quantization (jpegli AQ).
    ///
    /// When `true` (default for jpegli/hybrid presets), the encoder computes
    /// per-block AQ strengths from luminance data. When `false` (default for
    /// mozjpeg presets), AQ is skipped entirely — all blocks get neutral AQ (0.0).
    ///
    /// For mozjpeg presets where `zero_bias_mul` is all-zeros, disabling AQ
    /// produces identical output (AQ values are never applied). Disabling saves
    /// ~600KB-2.5MB of buffer allocations.
    pub aq_enabled: bool,
}

impl ExpertConfig {
    /// Default YCbCr config (jpegli defaults, no trellis).
    ///
    /// Uses jpegli perceptual tables with distance-based scaling, quality-adaptive
    /// zero-bias blend, progressive scan mode, deringing enabled, no trellis.
    ///
    /// Zero-bias is pre-blended for the given quality level.
    #[must_use]
    pub fn default_ycbcr(quality: impl Into<Quality>) -> Self {
        let quality = quality.into();
        let tables = EncodingTables::default_ycbcr();
        let zero_bias_hq = EncodingTables::ycbcr_hq_zero_bias_mul();
        let zero_bias_lq = EncodingTables::ycbcr_lq_zero_bias_mul();

        let mut config = Self {
            tables,
            zero_bias_hq,
            zero_bias_lq,
            zero_bias_hq_distance: 1.0,
            zero_bias_lq_distance: 3.0,

            trellis_enabled: false,
            trellis_dc_enabled: true,

            trellis_use_lambda_weight_tbl: true,
            trellis_lambda_log_scale1: 14.75,
            trellis_lambda_log_scale2: 16.5,
            trellis_num_loops: 1,
            trellis_speed_mode: TrellisSpeedMode::Adaptive,
            trellis_delta_dc_weight: 0.0,

            aq_trellis_coupling: 0.0,
            aq_trellis_exponent: 1.0,
            aq_trellis_threshold: 0.0,
            aq_trellis_chroma_scale: 1.0,
            aq_trellis_quality_adaptive: false,
            aq_trellis_multiplicative: false,
            aq_trellis_max_adjustment: 0.0,

            scan_mode: ProgressiveScanMode::Progressive,
            deringing: true,
            allow_16bit_quant_tables: false,
            quality,
            downsampling_method: DownsamplingMethod::default(),
            aq_enabled: true,
        };
        config.blend_zero_bias();
        config
    }

    /// Initialize from an [`OptimizationPreset`] with all tables pre-computed.
    ///
    /// Each preset maps to concrete field values matching the encoder profile.
    /// All presets start with `aq_trellis_coupling=0.0` (standalone trellis when
    /// enabled). The optimizer can increase coupling to explore hybrid territory.
    ///
    /// For mozjpeg presets, tables are pre-scaled to the given quality using
    /// libjpeg's quality scaling formula (`ScalingParams::Exact`). For jpegli/hybrid
    /// presets, tables use distance-based scaling (`ScalingParams::Scaled`).
    ///
    /// Zero-bias is pre-blended for the given quality level.
    #[must_use]
    pub fn from_preset(preset: OptimizationPreset, quality: impl Into<Quality>) -> Self {
        use OptimizationPreset::*;

        let quality = quality.into();

        // Determine base tables and zero-bias endpoints
        let (tables, zero_bias_hq, zero_bias_lq) = match preset {
            MozjpegBaseline | MozjpegProgressive | MozjpegMaxCompression => {
                // Use for_mozjpeg_tables() to preserve the original mozjpeg quality.
                // to_internal() remaps for jpegli's distance system, producing wrong tables.
                let mozjpeg_tables = super::tables::robidoux::generate_mozjpeg_default_tables(
                    quality.for_mozjpeg_tables(),
                    false,
                );
                // Mozjpeg uses neutral zero-bias (mul=0, offset=0.5), so
                // HQ/LQ blend is irrelevant — both endpoints are zero.
                let neutral = PerComponent::new([0.0f32; 64], [0.0f32; 64], [0.0f32; 64]);
                (*mozjpeg_tables, neutral.clone(), neutral)
            }
            _ => {
                // Jpegli perceptual tables for jpegli and hybrid presets
                let tables = EncodingTables::default_ycbcr();
                let hq = EncodingTables::ycbcr_hq_zero_bias_mul();
                let lq = EncodingTables::ycbcr_lq_zero_bias_mul();
                (tables, hq, lq)
            }
        };

        // Trellis: disabled for jpegli, Thorough for mozjpeg, Adaptive for hybrid
        let (trellis_enabled, trellis_speed_mode) = match preset {
            JpegliBaseline | JpegliProgressive => (false, TrellisSpeedMode::Adaptive),
            MozjpegBaseline | MozjpegProgressive | MozjpegMaxCompression => {
                (true, TrellisSpeedMode::Thorough)
            }
            HybridBaseline | HybridProgressive => (true, TrellisSpeedMode::Adaptive),
            HybridMaxCompression => (true, TrellisSpeedMode::Thorough),
        };

        // Scan mode: baseline, progressive, mozjpeg script, or search
        let scan_mode = match preset {
            JpegliBaseline | MozjpegBaseline | HybridBaseline => ProgressiveScanMode::Baseline,
            JpegliProgressive | HybridProgressive => ProgressiveScanMode::Progressive,
            MozjpegProgressive => ProgressiveScanMode::ProgressiveMozjpeg,
            MozjpegMaxCompression | HybridMaxCompression => ProgressiveScanMode::ProgressiveSearch,
        };

        // Deringing: enabled for all except mozjpeg baseline/progressive
        // (C mozjpeg only enables it for JCP_MAX_COMPRESSION)
        let deringing = !matches!(preset, MozjpegBaseline | MozjpegProgressive);

        let mut config = Self {
            tables,
            zero_bias_hq,
            zero_bias_lq,
            zero_bias_hq_distance: 1.0,
            zero_bias_lq_distance: 3.0,

            trellis_enabled,
            trellis_dc_enabled: true,

            trellis_use_lambda_weight_tbl: true,
            trellis_lambda_log_scale1: 14.75,
            trellis_lambda_log_scale2: 16.5,
            trellis_num_loops: 1,
            trellis_speed_mode,
            trellis_delta_dc_weight: 0.0,

            // All presets start uncoupled (standalone trellis when enabled).
            aq_trellis_coupling: 0.0,
            aq_trellis_exponent: 1.0,
            aq_trellis_threshold: 0.0,
            aq_trellis_chroma_scale: 1.0,
            aq_trellis_quality_adaptive: false,
            aq_trellis_multiplicative: false,
            aq_trellis_max_adjustment: 0.0,

            scan_mode,
            deringing,
            allow_16bit_quant_tables: false,
            quality,
            downsampling_method: DownsamplingMethod::default(),
            aq_enabled: preset.uses_aq(),
        };
        config.blend_zero_bias();
        config
    }

    /// Recompute `tables.zero_bias_mul` by blending HQ/LQ endpoints at the
    /// current quality's Butteraugli distance.
    ///
    /// The blend is linear between `zero_bias_hq_distance` (fully HQ, t=1.0)
    /// and `zero_bias_lq_distance` (fully LQ, t=0.0).
    ///
    /// **Must be called** after changing any of: `quality`, `zero_bias_hq`,
    /// `zero_bias_lq`, `zero_bias_hq_distance`, or `zero_bias_lq_distance`.
    /// The constructors ([`default_ycbcr`](Self::default_ycbcr),
    /// [`from_preset`](Self::from_preset)) call this automatically.
    ///
    /// [`to_encoder_config()`](Self::to_encoder_config) does **not** call this
    /// automatically (it takes `&self`). If you changed zero-bias-affecting
    /// fields after construction, call this before `to_encoder_config()`.
    pub fn blend_zero_bias(&mut self) {
        let distance = self.quality.to_distance();

        // Compute blend factor: 1.0 = fully HQ, 0.0 = fully LQ
        let t = if distance <= self.zero_bias_hq_distance {
            1.0
        } else if distance >= self.zero_bias_lq_distance {
            0.0
        } else {
            let range = self.zero_bias_lq_distance - self.zero_bias_hq_distance;
            if range <= 0.0 {
                0.0
            } else {
                1.0 - (distance - self.zero_bias_hq_distance) / range
            }
        };

        // lq.blend(hq, t) = lq*(1-t) + hq*t
        //   t=1.0 → hq (high quality, low distance)
        //   t=0.0 → lq (low quality, high distance)
        self.tables.zero_bias_mul = self.zero_bias_lq.blend(&self.zero_bias_hq, t);
    }

    /// Returns true if `quality` affects quant table values.
    ///
    /// When `true`, `tables.scaling == Scaled` and the encoder applies
    /// per-frequency non-linear scaling based on quality/distance.
    ///
    /// When `false`, `tables.scaling == Exact` and quant values are used as-is
    /// (e.g., mozjpeg presets where tables are pre-scaled to quality).
    #[must_use]
    pub fn uses_quality_scaling(&self) -> bool {
        !self.tables.is_exact()
    }

    /// Build an [`EncoderConfig`] for actual encoding.
    ///
    /// `color_mode` is separate because it's image-dependent (subsampling choice),
    /// not a tuning parameter for optimization.
    ///
    /// The tables are packaged as [`QuantTableConfig::Custom`] to bypass the
    /// encoder's default table generation. Zero-bias values from `tables.zero_bias_mul`
    /// are used as-is — call [`blend_zero_bias()`](Self::blend_zero_bias) first if
    /// you changed quality or zero-bias fields after construction.
    ///
    /// The trellis/hybrid dispatch depends on `trellis_enabled` and
    /// `aq_trellis_coupling`. See the struct-level "Hybrid-Mode Limitations"
    /// docs for which fields are ignored in each mode.
    #[must_use]
    pub fn to_encoder_config(&self, color_mode: ColorMode) -> EncoderConfig {
        let mut config = match color_mode {
            ColorMode::YCbCr { subsampling } => EncoderConfig::ycbcr(self.quality, subsampling),
            ColorMode::Xyb { subsampling } => EncoderConfig::xyb(self.quality, subsampling),
            ColorMode::Grayscale => EncoderConfig::grayscale(self.quality),
        };

        // Package tables as Custom to bypass default table generation.
        // The clone copies all fields including zero_bias_mul (which should
        // already be blended by the caller or constructor).
        config.quant_table_config = QuantTableConfig::Custom(Box::new(self.tables.clone()));

        // Scan mode — progressive modes automatically get optimized Huffman
        config.scan_mode = self.scan_mode;
        if self.scan_mode.is_progressive() {
            config.huffman = HuffmanStrategy::Optimize;
        }

        // Encoder flags
        config.deringing = self.deringing;
        config.aq_enabled = self.aq_enabled;
        config.allow_16bit_quant_tables = self.allow_16bit_quant_tables;
        config.downsampling_method = self.downsampling_method;

        // Trellis / hybrid dispatch
        let (trellis, hybrid) = self.build_trellis_or_hybrid();
        config.trellis = trellis;
        config.hybrid_config = hybrid;

        config
    }

    /// Pack trellis + coupling fields into `TrellisConfig` (standalone) or
    /// `HybridConfig` (coupled).
    ///
    /// Returns `(Some(TrellisConfig), disabled HybridConfig)` for standalone mode,
    /// or `(None, enabled HybridConfig)` for hybrid mode.
    ///
    /// In hybrid mode, the following fields are NOT forwarded to per-block trellis
    /// configs (see struct-level docs): `trellis_speed_mode`,
    /// `trellis_delta_dc_weight`, `trellis_use_lambda_weight_tbl`.
    fn build_trellis_or_hybrid(&self) -> (Option<TrellisConfig>, HybridConfig) {
        if !self.trellis_enabled {
            return (None, HybridConfig::disabled());
        }

        if self.aq_trellis_coupling != 0.0 {
            // Hybrid mode: AQ-coupled trellis (positive or negative coupling).
            // Note: trellis_speed_mode, trellis_delta_dc_weight,
            // and trellis_use_lambda_weight_tbl are stored in HybridConfig but
            // NOT forwarded to per-block TrellisConfig by to_trellis_config().
            // See struct-level "Hybrid-Mode Limitations" docs.
            let hybrid = HybridConfig {
                enabled: true,
                aq_lambda_scale: self.aq_trellis_coupling,
                base_lambda_scale1: self.trellis_lambda_log_scale1,
                base_lambda_scale2: self.trellis_lambda_log_scale2,
                dc_enabled: self.trellis_dc_enabled,
                num_loops: self.trellis_num_loops,
                use_lambda_weight_tbl: self.trellis_use_lambda_weight_tbl,
                aq_exponent: self.aq_trellis_exponent,
                aq_threshold: self.aq_trellis_threshold,
                quality_adaptive: self.aq_trellis_quality_adaptive,
                chroma_scale: self.aq_trellis_chroma_scale,
                multiplicative: self.aq_trellis_multiplicative,
                max_adjustment: self.aq_trellis_max_adjustment,
            };
            (None, hybrid)
        } else {
            // Standalone trellis: all fields forwarded directly.
            let trellis = TrellisConfig {
                enabled: true,
                dc_enabled: self.trellis_dc_enabled,
                use_lambda_weight_tbl: self.trellis_use_lambda_weight_tbl,
                lambda_log_scale1: self.trellis_lambda_log_scale1,
                lambda_log_scale2: self.trellis_lambda_log_scale2,
                num_loops: self.trellis_num_loops,
                speed_mode: self.trellis_speed_mode,
                delta_dc_weight: self.trellis_delta_dc_weight,
            };
            (Some(trellis), HybridConfig::disabled())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::encoder_types::ChromaSubsampling;
    use super::*;

    #[test]
    fn test_default_ycbcr_fields() {
        let config = ExpertConfig::default_ycbcr(90.0);
        assert!(!config.trellis_enabled);
        assert!(config.deringing);
        assert_eq!(config.scan_mode, ProgressiveScanMode::Progressive);
        assert!(!config.allow_16bit_quant_tables);
        // Default uses jpegli scaling, not exact
        assert!(config.uses_quality_scaling());
    }

    #[test]
    fn test_from_preset_jpegli_baseline() {
        let config = ExpertConfig::from_preset(OptimizationPreset::JpegliBaseline, 85.0);
        assert!(!config.trellis_enabled);
        assert!(config.deringing);
        assert_eq!(config.scan_mode, ProgressiveScanMode::Baseline);
        assert!(config.uses_quality_scaling());
    }

    #[test]
    fn test_from_preset_jpegli_progressive() {
        let config = ExpertConfig::from_preset(OptimizationPreset::JpegliProgressive, 85.0);
        assert!(!config.trellis_enabled);
        assert!(config.deringing);
        assert_eq!(config.scan_mode, ProgressiveScanMode::Progressive);
    }

    #[test]
    fn test_from_preset_mozjpeg_baseline() {
        let config = ExpertConfig::from_preset(OptimizationPreset::MozjpegBaseline, 85.0);
        assert!(config.trellis_enabled);
        assert!(!config.deringing);
        assert_eq!(config.scan_mode, ProgressiveScanMode::Baseline);
        assert_eq!(config.trellis_speed_mode, TrellisSpeedMode::Thorough);
        // Mozjpeg uses exact (pre-scaled) tables
        assert!(!config.uses_quality_scaling());
    }

    #[test]
    fn test_from_preset_mozjpeg_progressive() {
        let config = ExpertConfig::from_preset(OptimizationPreset::MozjpegProgressive, 85.0);
        assert!(config.trellis_enabled);
        assert!(!config.deringing);
        assert_eq!(config.scan_mode, ProgressiveScanMode::ProgressiveMozjpeg);
    }

    #[test]
    fn test_from_preset_mozjpeg_max_compression() {
        let config = ExpertConfig::from_preset(OptimizationPreset::MozjpegMaxCompression, 85.0);
        assert!(config.trellis_enabled);
        assert!(config.deringing);
        assert_eq!(config.scan_mode, ProgressiveScanMode::ProgressiveSearch);
        assert_eq!(config.trellis_speed_mode, TrellisSpeedMode::Thorough);
    }

    #[test]
    fn test_from_preset_hybrid_baseline() {
        let config = ExpertConfig::from_preset(OptimizationPreset::HybridBaseline, 85.0);
        assert!(config.trellis_enabled);
        assert!(config.deringing);
        assert_eq!(config.scan_mode, ProgressiveScanMode::Baseline);
        assert_eq!(config.trellis_speed_mode, TrellisSpeedMode::Adaptive);
        // Hybrid starts uncoupled
        assert_eq!(config.aq_trellis_coupling, 0.0);
    }

    #[test]
    fn test_from_preset_hybrid_progressive() {
        let config = ExpertConfig::from_preset(OptimizationPreset::HybridProgressive, 85.0);
        assert!(config.trellis_enabled);
        assert!(config.deringing);
        assert_eq!(config.scan_mode, ProgressiveScanMode::Progressive);
        assert_eq!(config.trellis_speed_mode, TrellisSpeedMode::Adaptive);
    }

    #[test]
    fn test_from_preset_hybrid_max_compression() {
        let config = ExpertConfig::from_preset(OptimizationPreset::HybridMaxCompression, 85.0);
        assert!(config.trellis_enabled);
        assert!(config.deringing);
        assert_eq!(config.scan_mode, ProgressiveScanMode::ProgressiveSearch);
        assert_eq!(config.trellis_speed_mode, TrellisSpeedMode::Thorough);
    }

    #[test]
    fn test_to_encoder_config_no_trellis() {
        let expert = ExpertConfig::default_ycbcr(90.0);
        let enc = expert.to_encoder_config(ColorMode::YCbCr {
            subsampling: ChromaSubsampling::Quarter,
        });

        assert!(enc.trellis.is_none());
        assert!(!enc.hybrid_config.enabled);
        assert!(enc.deringing);
        assert_eq!(enc.scan_mode, ProgressiveScanMode::Progressive);
    }

    #[test]
    fn test_to_encoder_config_standalone_trellis() {
        let mut expert = ExpertConfig::from_preset(OptimizationPreset::MozjpegBaseline, 85.0);
        expert.aq_trellis_coupling = 0.0;

        let enc = expert.to_encoder_config(ColorMode::YCbCr {
            subsampling: ChromaSubsampling::Quarter,
        });

        assert!(enc.trellis.is_some());
        assert!(!enc.hybrid_config.enabled);

        let trellis = enc.trellis.unwrap();
        assert!(trellis.enabled);
        assert!(trellis.dc_enabled);
        assert_eq!(trellis.speed_mode, TrellisSpeedMode::Thorough);
    }

    #[test]
    fn test_to_encoder_config_hybrid_mode() {
        let mut expert = ExpertConfig::from_preset(OptimizationPreset::HybridProgressive, 85.0);
        expert.aq_trellis_coupling = 2.0;
        expert.aq_trellis_exponent = 0.5;
        expert.aq_trellis_chroma_scale = 0.8;

        let enc = expert.to_encoder_config(ColorMode::YCbCr {
            subsampling: ChromaSubsampling::Quarter,
        });

        // Hybrid mode: trellis is None, hybrid_config is enabled
        assert!(enc.trellis.is_none());
        assert!(enc.hybrid_config.enabled);
        assert_eq!(enc.hybrid_config.aq_lambda_scale, 2.0);
        assert_eq!(enc.hybrid_config.aq_exponent, 0.5);
        assert_eq!(enc.hybrid_config.chroma_scale, 0.8);
    }

    #[test]
    fn test_blend_zero_bias_high_quality() {
        let config = ExpertConfig::default_ycbcr(Quality::ApproxButteraugli(0.5));
        // Distance 0.5 <= hq_distance (1.0), so should be fully HQ
        let hq = EncodingTables::ycbcr_hq_zero_bias_mul();
        assert!(
            (config.tables.zero_bias_mul.c0[5] - hq.c0[5]).abs() < 1e-6,
            "At high quality, zero-bias should match HQ tables"
        );
    }

    #[test]
    fn test_blend_zero_bias_low_quality() {
        let config = ExpertConfig::default_ycbcr(Quality::ApproxButteraugli(5.0));
        // Distance 5.0 >= lq_distance (3.0), so should be fully LQ
        let lq = EncodingTables::ycbcr_lq_zero_bias_mul();
        assert!(
            (config.tables.zero_bias_mul.c0[5] - lq.c0[5]).abs() < 1e-6,
            "At low quality, zero-bias should match LQ tables"
        );
    }

    #[test]
    fn test_blend_zero_bias_mid_quality() {
        // Distance 2.0 is midpoint of [1.0, 3.0] range -> t=0.5
        let config = ExpertConfig::default_ycbcr(Quality::ApproxButteraugli(2.0));
        let hq = EncodingTables::ycbcr_hq_zero_bias_mul();
        let lq = EncodingTables::ycbcr_lq_zero_bias_mul();
        let expected = (hq.c0[5] + lq.c0[5]) / 2.0;
        assert!(
            (config.tables.zero_bias_mul.c0[5] - expected).abs() < 1e-5,
            "At mid quality, zero-bias should be midpoint of HQ/LQ: got {} expected {}",
            config.tables.zero_bias_mul.c0[5],
            expected
        );
    }

    #[test]
    fn test_all_presets_round_trip() {
        for preset in OptimizationPreset::all() {
            let expert = ExpertConfig::from_preset(preset, 85.0);
            let _enc = expert.to_encoder_config(ColorMode::YCbCr {
                subsampling: ChromaSubsampling::Quarter,
            });
        }
    }

    /// Verify all 9 trellis fields pass through in standalone mode (coupling=0).
    #[test]
    fn test_trellis_fields_pass_through_standalone() {
        let mut expert = ExpertConfig::default_ycbcr(85.0);
        expert.trellis_enabled = true;
        expert.trellis_dc_enabled = false;
        expert.trellis_use_lambda_weight_tbl = false;
        expert.trellis_lambda_log_scale1 = 15.0;
        expert.trellis_lambda_log_scale2 = 17.0;
        expert.trellis_num_loops = 2;
        expert.trellis_speed_mode = TrellisSpeedMode::Level(5);
        expert.trellis_delta_dc_weight = 0.5;

        let enc = expert.to_encoder_config(ColorMode::YCbCr {
            subsampling: ChromaSubsampling::None,
        });

        let trellis = enc.trellis.unwrap();
        assert!(trellis.enabled);
        assert!(!trellis.dc_enabled);
        assert!(!trellis.use_lambda_weight_tbl);
        assert!((trellis.lambda_log_scale1 - 15.0).abs() < 1e-6);
        assert!((trellis.lambda_log_scale2 - 17.0).abs() < 1e-6);
        assert_eq!(trellis.num_loops, 2);
        assert_eq!(trellis.speed_mode, TrellisSpeedMode::Level(5));
        assert!((trellis.delta_dc_weight - 0.5).abs() < 1e-6);
    }

    /// Verify hybrid-mode fields pass through to HybridConfig.
    /// Note: some trellis fields are NOT forwarded to per-block TrellisConfig
    /// by HybridConfig::to_trellis_config() — see struct-level docs.
    #[test]
    fn test_hybrid_fields_pass_through() {
        let mut expert = ExpertConfig::default_ycbcr(85.0);
        expert.trellis_enabled = true;
        expert.aq_trellis_coupling = 3.5;
        expert.aq_trellis_exponent = 2.0;
        expert.aq_trellis_threshold = 0.1;
        expert.aq_trellis_chroma_scale = 0.7;
        expert.aq_trellis_quality_adaptive = true;
        expert.trellis_lambda_log_scale1 = 15.0;
        expert.trellis_lambda_log_scale2 = 17.0;
        expert.trellis_dc_enabled = false;
        expert.trellis_num_loops = 2;

        let enc = expert.to_encoder_config(ColorMode::YCbCr {
            subsampling: ChromaSubsampling::None,
        });

        assert!(enc.trellis.is_none());
        assert!(enc.hybrid_config.enabled);
        assert_eq!(enc.hybrid_config.aq_lambda_scale, 3.5);
        assert_eq!(enc.hybrid_config.aq_exponent, 2.0);
        assert_eq!(enc.hybrid_config.aq_threshold, 0.1);
        assert_eq!(enc.hybrid_config.chroma_scale, 0.7);
        assert!(enc.hybrid_config.quality_adaptive);
        assert!((enc.hybrid_config.base_lambda_scale1 - 15.0).abs() < 1e-6);
        assert!((enc.hybrid_config.base_lambda_scale2 - 17.0).abs() < 1e-6);
        assert!(!enc.hybrid_config.dc_enabled);
        assert_eq!(enc.hybrid_config.num_loops, 2);
    }

    #[test]
    fn test_custom_tables_preserved() {
        let mut expert = ExpertConfig::default_ycbcr(85.0);
        expert.tables.quant.c0[0] = 42.0;
        expert.tables.quant.c1[63] = 99.0;

        let enc = expert.to_encoder_config(ColorMode::YCbCr {
            subsampling: ChromaSubsampling::Quarter,
        });

        let custom = enc.quant_table_config.custom_tables().unwrap();
        assert!((custom.quant.c0[0] - 42.0).abs() < 1e-6);
        assert!((custom.quant.c1[63] - 99.0).abs() < 1e-6);
    }

    #[test]
    fn test_quality_types_accepted() {
        let _ = ExpertConfig::default_ycbcr(85.0f32);
        let _ = ExpertConfig::default_ycbcr(85u8);
        let _ = ExpertConfig::default_ycbcr(85i32);
        let _ = ExpertConfig::default_ycbcr(Quality::ApproxMozjpeg(80));
        let _ = ExpertConfig::default_ycbcr(Quality::ApproxSsim2(90.0));
        let _ = ExpertConfig::default_ycbcr(Quality::ApproxButteraugli(1.0));
    }

    #[test]
    fn test_scan_mode_progressive_enables_optimize() {
        let mut expert = ExpertConfig::default_ycbcr(85.0);
        expert.scan_mode = ProgressiveScanMode::ProgressiveSearch;

        let enc = expert.to_encoder_config(ColorMode::YCbCr {
            subsampling: ChromaSubsampling::Quarter,
        });

        assert!(matches!(enc.huffman, HuffmanStrategy::Optimize));
    }

    /// Verify that blend_zero_bias is idempotent and doesn't accumulate.
    #[test]
    fn test_blend_zero_bias_idempotent() {
        let mut config = ExpertConfig::default_ycbcr(85.0);
        let first = config.tables.zero_bias_mul.c0[5];
        config.blend_zero_bias();
        config.blend_zero_bias();
        config.blend_zero_bias();
        assert!(
            (config.tables.zero_bias_mul.c0[5] - first).abs() < 1e-6,
            "blend_zero_bias should be idempotent"
        );
    }

    // ====================================================================
    // Parameter sensitivity: encode with permutations, measure file size
    // ====================================================================

    /// Deterministic noise+patches test image (NOT gradient — see CLAUDE.md).
    /// Mixed content exercises both low-freq (patches) and high-freq (noise) paths.
    fn make_test_image(width: u32, height: u32) -> Vec<u8> {
        let mut pixels = vec![0u8; (width * height * 3) as usize];
        let mut state: u64 = 0xDEAD_BEEF;
        let next = |s: &mut u64| -> u8 {
            *s = s.wrapping_mul(1103515245).wrapping_add(12345) & 0x7FFF_FFFF;
            ((*s >> 16) & 0xFF) as u8
        };
        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * 3) as usize;
                // Patches: 32x32 blocks of distinct colors, noise within
                let patch_x = x / 32;
                let patch_y = y / 32;
                let base_r = ((patch_x * 73 + 50) % 256) as u8;
                let base_g = ((patch_y * 97 + 80) % 256) as u8;
                let base_b = (((patch_x + patch_y) * 131 + 30) % 256) as u8;
                // Add noise ±30
                let noise_r = (next(&mut state) % 61) as i16 - 30;
                let noise_g = (next(&mut state) % 61) as i16 - 30;
                let noise_b = (next(&mut state) % 61) as i16 - 30;
                pixels[idx] = (base_r as i16 + noise_r).clamp(0, 255) as u8;
                pixels[idx + 1] = (base_g as i16 + noise_g).clamp(0, 255) as u8;
                pixels[idx + 2] = (base_b as i16 + noise_b).clamp(0, 255) as u8;
            }
        }
        pixels
    }

    fn encode_expert(expert: &ExpertConfig, pixels: &[u8], w: u32, h: u32) -> usize {
        use super::super::encoder_types::PixelLayout;
        let enc_config = expert.to_encoder_config(ColorMode::YCbCr {
            subsampling: ChromaSubsampling::Quarter,
        });
        let mut enc = enc_config
            .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
            .expect("encoder creation failed");
        enc.push_packed(pixels, enough::Unstoppable)
            .expect("push failed");
        let jpeg = enc.finish().expect("finish failed");
        jpeg.len()
    }

    /// Test every ExpertConfig field for file-size sensitivity.
    ///
    /// Uses MozjpegBaseline as base (trellis enabled, standalone mode) so
    /// trellis fields are active. Then tests each field individually.
    #[test]
    fn test_parameter_sensitivity() {
        let w = 256u32;
        let h = 256u32;
        let pixels = make_test_image(w, h);

        // ---- Baselines across presets ----
        println!("\n=== Preset baseline sizes (Q85, 256x256, 4:2:0) ===");
        for preset in OptimizationPreset::all() {
            let expert = ExpertConfig::from_preset(preset, 85.0);
            let size = encode_expert(&expert, &pixels, w, h);
            println!("  {:?}: {} bytes", preset, size);
        }

        // ---- Per-field sensitivity from MozjpegBaseline base ----
        let base = ExpertConfig::from_preset(OptimizationPreset::MozjpegBaseline, 85.0);
        let base_size = encode_expert(&base, &pixels, w, h);
        println!("\n=== Base: MozjpegBaseline Q85 = {} bytes ===", base_size);

        // Helper: mutate one field, encode, report delta
        let mut results: Vec<(&str, i64, String)> = Vec::new();
        let mut test_field = |name: &'static str, config: &ExpertConfig, note: &str| {
            let size = encode_expert(config, &pixels, w, h);
            let delta = size as i64 - base_size as i64;
            let pct = (delta as f64 / base_size as f64) * 100.0;
            results.push((name, delta, note.to_string()));
            println!("  {:<45} {:>7} bytes  {:>+7.2}%  {}", name, size, pct, note);
        };

        println!("\n--- Trellis on/off ---");
        {
            let mut c = base.clone();
            c.trellis_enabled = false;
            test_field("trellis_enabled=false", &c, "(was true)");
        }

        println!("\n--- Trellis DC ---");
        {
            let mut c = base.clone();
            c.trellis_dc_enabled = false;
            test_field("trellis_dc_enabled=false", &c, "(was true)");
        }

        println!("\n--- Trellis lambda weight table ---");
        {
            let mut c = base.clone();
            c.trellis_use_lambda_weight_tbl = false;
            test_field("trellis_use_lambda_weight_tbl=false", &c, "(was true)");
        }

        println!("\n--- Trellis lambda_log_scale1 (rate penalty) ---");
        for val in [12.0, 13.0, 14.0, 14.75, 15.5, 16.0, 17.0] {
            let mut c = base.clone();
            c.trellis_lambda_log_scale1 = val;
            test_field(
                Box::leak(format!("trellis_lambda_log_scale1={}", val).into_boxed_str()),
                &c,
                if (val - 14.75).abs() < 0.01 {
                    "(default)"
                } else {
                    ""
                },
            );
        }

        println!("\n--- Trellis lambda_log_scale2 (distortion sensitivity) ---");
        for val in [14.0, 15.0, 16.0, 16.5, 17.0, 18.0] {
            let mut c = base.clone();
            c.trellis_lambda_log_scale2 = val;
            test_field(
                Box::leak(format!("trellis_lambda_log_scale2={}", val).into_boxed_str()),
                &c,
                if (val - 16.5).abs() < 0.01 {
                    "(default)"
                } else {
                    ""
                },
            );
        }

        println!("\n--- Trellis num_loops ---");
        for val in [1, 2, 3, 5] {
            let mut c = base.clone();
            c.trellis_num_loops = val;
            test_field(
                Box::leak(format!("trellis_num_loops={}", val).into_boxed_str()),
                &c,
                if val == 1 { "(default)" } else { "" },
            );
        }

        println!("\n--- Trellis speed_mode ---");
        {
            let mut c = base.clone();
            c.trellis_speed_mode = TrellisSpeedMode::Adaptive;
            test_field("trellis_speed_mode=Adaptive", &c, "(was Thorough)");
        }
        for level in [1, 3, 5, 8] {
            let mut c = base.clone();
            c.trellis_speed_mode = TrellisSpeedMode::Level(level);
            test_field(
                Box::leak(format!("trellis_speed_mode=Level({})", level).into_boxed_str()),
                &c,
                "",
            );
        }

        println!("\n--- Trellis delta_dc_weight ---");
        for val in [0.0, 0.1, 0.5, 1.0, 2.0, 5.0] {
            let mut c = base.clone();
            c.trellis_delta_dc_weight = val;
            test_field(
                Box::leak(format!("trellis_delta_dc_weight={}", val).into_boxed_str()),
                &c,
                if val == 0.0 { "(default)" } else { "" },
            );
        }

        println!("\n--- Deringing ---");
        {
            let mut c = base.clone();
            c.deringing = true;
            test_field("deringing=true", &c, "(was false for mozjpeg)");
        }

        println!("\n--- Scan mode ---");
        for mode in [
            ProgressiveScanMode::Baseline,
            ProgressiveScanMode::Progressive,
            ProgressiveScanMode::ProgressiveMozjpeg,
            ProgressiveScanMode::ProgressiveSearch,
        ] {
            let mut c = base.clone();
            c.scan_mode = mode;
            test_field(
                Box::leak(format!("scan_mode={:?}", mode).into_boxed_str()),
                &c,
                if mode == ProgressiveScanMode::Baseline {
                    "(default for this preset)"
                } else {
                    ""
                },
            );
        }

        println!("\n--- Quality (with Exact scaling, changes zero-bias only) ---");
        for q in [50.0, 70.0, 85.0, 90.0, 95.0] {
            let mut c = base.clone();
            c.quality = Quality::from(q);
            c.blend_zero_bias();
            test_field(
                Box::leak(format!("quality={} (exact tables)", q).into_boxed_str()),
                &c,
                if (q - 85.0).abs() < 0.01 {
                    "(default)"
                } else {
                    ""
                },
            );
        }

        // Test quality with Scaled tables (jpegli preset)
        println!("\n--- Quality (with Scaled tables, jpegli preset) ---");
        let jpegli_base = ExpertConfig::from_preset(OptimizationPreset::JpegliBaseline, 85.0);
        let jpegli_base_size = encode_expert(&jpegli_base, &pixels, w, h);
        println!("  JpegliBaseline Q85 base = {} bytes", jpegli_base_size);
        for q in [50.0, 70.0, 85.0, 90.0, 95.0] {
            let c = ExpertConfig::from_preset(OptimizationPreset::JpegliBaseline, q);
            let size = encode_expert(&c, &pixels, w, h);
            let delta = size as i64 - jpegli_base_size as i64;
            let pct = (delta as f64 / jpegli_base_size as f64) * 100.0;
            println!(
                "  {:<45} {:>7} bytes  {:>+7.2}%",
                format!("quality={} (scaled tables)", q),
                size,
                pct
            );
        }

        println!("\n--- Zero-bias blend range ---");
        {
            let mut c = ExpertConfig::from_preset(OptimizationPreset::JpegliBaseline, 85.0);
            // Widen range: fully HQ at distance 0.5, fully LQ at distance 5.0
            c.zero_bias_hq_distance = 0.5;
            c.zero_bias_lq_distance = 5.0;
            c.blend_zero_bias();
            let size = encode_expert(&c, &pixels, w, h);
            let delta = size as i64 - jpegli_base_size as i64;
            let pct = (delta as f64 / jpegli_base_size as f64) * 100.0;
            println!(
                "  {:<45} {:>7} bytes  {:>+7.2}%  (was 1.0-3.0)",
                "zero_bias range 0.5-5.0", size, pct
            );
        }
        {
            let mut c = ExpertConfig::from_preset(OptimizationPreset::JpegliBaseline, 85.0);
            // Narrow range: HQ=LQ=2.0 (no blend, snap to nearest)
            c.zero_bias_hq_distance = 2.0;
            c.zero_bias_lq_distance = 2.0;
            c.blend_zero_bias();
            let size = encode_expert(&c, &pixels, w, h);
            let delta = size as i64 - jpegli_base_size as i64;
            let pct = (delta as f64 / jpegli_base_size as f64) * 100.0;
            println!(
                "  {:<45} {:>7} bytes  {:>+7.2}%  (forced LQ)",
                "zero_bias range 2.0-2.0", size, pct
            );
        }

        // Zero out all zero-bias (make them 0.0 = no zero-biasing)
        {
            let mut c = ExpertConfig::from_preset(OptimizationPreset::JpegliBaseline, 85.0);
            c.tables.zero_bias_mul = PerComponent::new([0.0f32; 64], [0.0f32; 64], [0.0f32; 64]);
            let size = encode_expert(&c, &pixels, w, h);
            let delta = size as i64 - jpegli_base_size as i64;
            let pct = (delta as f64 / jpegli_base_size as f64) * 100.0;
            println!(
                "  {:<45} {:>7} bytes  {:>+7.2}%",
                "zero_bias_mul=all zeros (disabled)", size, pct
            );
        }
        // Max out all zero-bias (1.0 = maximum rounding toward zero)
        {
            let mut c = ExpertConfig::from_preset(OptimizationPreset::JpegliBaseline, 85.0);
            c.tables.zero_bias_mul = PerComponent::new([1.0f32; 64], [1.0f32; 64], [1.0f32; 64]);
            let size = encode_expert(&c, &pixels, w, h);
            let delta = size as i64 - jpegli_base_size as i64;
            let pct = (delta as f64 / jpegli_base_size as f64) * 100.0;
            println!(
                "  {:<45} {:>7} bytes  {:>+7.2}%",
                "zero_bias_mul=all ones (maximum)", size, pct
            );
        }

        println!("\n--- allow_16bit_quant_tables ---");
        {
            let mut c = base.clone();
            c.allow_16bit_quant_tables = true;
            test_field("allow_16bit_quant_tables=true", &c, "(was false)");
        }

        println!("\n--- Downsampling method ---");
        {
            let mut c = base.clone();
            c.downsampling_method = DownsamplingMethod::GammaAware;
            test_field("downsampling_method=GammaAware", &c, "(was Box)");
        }
        {
            let mut c = base.clone();
            c.downsampling_method = DownsamplingMethod::GammaAwareIterative;
            test_field("downsampling_method=GammaAwareIterative", &c, "(SharpYUV)");
        }

        // ---- Hybrid mode fields ----
        println!("\n--- Hybrid mode: aq_trellis_coupling sweep ---");
        for coupling in [0.0, 0.5, 1.0, 2.0, 4.0, 8.0] {
            let mut c = base.clone();
            c.aq_trellis_coupling = coupling;
            test_field(
                Box::leak(format!("aq_trellis_coupling={}", coupling).into_boxed_str()),
                &c,
                if coupling == 0.0 {
                    "(standalone)"
                } else {
                    "(hybrid)"
                },
            );
        }

        println!("\n--- Hybrid mode internals (coupling=2.0) ---");
        {
            let mut c = base.clone();
            c.aq_trellis_coupling = 2.0;
            let hybrid_base_size = encode_expert(&c, &pixels, w, h);
            println!("  Hybrid base (coupling=2.0) = {} bytes", hybrid_base_size);

            for exp in [0.5, 1.0, 2.0] {
                let mut c2 = c.clone();
                c2.aq_trellis_exponent = exp;
                let size = encode_expert(&c2, &pixels, w, h);
                let delta = size as i64 - hybrid_base_size as i64;
                let pct = (delta as f64 / hybrid_base_size as f64) * 100.0;
                println!(
                    "  {:<45} {:>7} bytes  {:>+7.2}%",
                    format!("aq_trellis_exponent={}", exp),
                    size,
                    pct
                );
            }

            for thresh in [0.0, 0.5, 1.0, 2.0] {
                let mut c2 = c.clone();
                c2.aq_trellis_threshold = thresh;
                let size = encode_expert(&c2, &pixels, w, h);
                let delta = size as i64 - hybrid_base_size as i64;
                let pct = (delta as f64 / hybrid_base_size as f64) * 100.0;
                println!(
                    "  {:<45} {:>7} bytes  {:>+7.2}%",
                    format!("aq_trellis_threshold={}", thresh),
                    size,
                    pct
                );
            }

            for cs in [0.0, 0.5, 1.0, 2.0] {
                let mut c2 = c.clone();
                c2.aq_trellis_chroma_scale = cs;
                let size = encode_expert(&c2, &pixels, w, h);
                let delta = size as i64 - hybrid_base_size as i64;
                let pct = (delta as f64 / hybrid_base_size as f64) * 100.0;
                println!(
                    "  {:<45} {:>7} bytes  {:>+7.2}%",
                    format!("aq_trellis_chroma_scale={}", cs),
                    size,
                    pct
                );
            }

            {
                let mut c2 = c.clone();
                c2.aq_trellis_quality_adaptive = true;
                let size = encode_expert(&c2, &pixels, w, h);
                let delta = size as i64 - hybrid_base_size as i64;
                let pct = (delta as f64 / hybrid_base_size as f64) * 100.0;
                println!(
                    "  {:<45} {:>7} bytes  {:>+7.2}%",
                    "aq_trellis_quality_adaptive=true", size, pct
                );
            }
        }

        // ---- Quant table scaling ----
        println!("\n--- Quant table values (direct manipulation) ---");
        {
            // Halve all quant values (= higher quality, bigger file)
            let mut c = base.clone();
            for v in c.tables.quant.c0.iter_mut() {
                *v *= 0.5;
            }
            for v in c.tables.quant.c1.iter_mut() {
                *v *= 0.5;
            }
            for v in c.tables.quant.c2.iter_mut() {
                *v *= 0.5;
            }
            test_field("quant_tables * 0.5 (finer quant)", &c, "");
        }
        {
            // Double all quant values (= lower quality, smaller file)
            let mut c = base.clone();
            for v in c.tables.quant.c0.iter_mut() {
                *v *= 2.0;
            }
            for v in c.tables.quant.c1.iter_mut() {
                *v *= 2.0;
            }
            for v in c.tables.quant.c2.iter_mut() {
                *v *= 2.0;
            }
            test_field("quant_tables * 2.0 (coarser quant)", &c, "");
        }

        // ---- Summary: identify dead fields ----
        println!("\n=== SENSITIVITY SUMMARY ===");
        println!("Fields with |delta| == 0 are effectively dead for this config:");
        for (name, delta, note) in &results {
            if *delta == 0 {
                println!("  DEAD: {} {}", name, note);
            }
        }
        println!("\nFields with |delta| > 0:");
        let mut active: Vec<_> = results.iter().filter(|(_, d, _)| *d != 0).collect();
        active.sort_by_key(|(_, d, _)| d.unsigned_abs());
        for (name, delta, note) in active.iter().rev() {
            let pct = (*delta as f64 / base_size as f64) * 100.0;
            println!("  {:>+7} bytes ({:>+6.2}%): {} {}", delta, pct, name, note);
        }
    }

    #[test]
    fn test_expert_aq_enabled_default() {
        let config = ExpertConfig::default_ycbcr(90.0);
        assert!(config.aq_enabled, "default_ycbcr should enable AQ");
    }

    #[test]
    fn test_expert_aq_enabled_mozjpeg() {
        let config = ExpertConfig::from_preset(OptimizationPreset::MozjpegBaseline, 85.0);
        assert!(!config.aq_enabled, "Mozjpeg preset should disable AQ");

        let config = ExpertConfig::from_preset(OptimizationPreset::MozjpegProgressive, 85.0);
        assert!(!config.aq_enabled);

        let config = ExpertConfig::from_preset(OptimizationPreset::MozjpegMaxCompression, 85.0);
        assert!(!config.aq_enabled);
    }

    #[test]
    fn test_expert_aq_enabled_jpegli() {
        let config = ExpertConfig::from_preset(OptimizationPreset::JpegliBaseline, 85.0);
        assert!(config.aq_enabled, "Jpegli preset should enable AQ");

        let config = ExpertConfig::from_preset(OptimizationPreset::JpegliProgressive, 85.0);
        assert!(config.aq_enabled);
    }

    #[test]
    fn test_expert_aq_enabled_hybrid() {
        let config = ExpertConfig::from_preset(OptimizationPreset::HybridProgressive, 85.0);
        assert!(config.aq_enabled, "Hybrid preset should enable AQ");
    }

    #[test]
    fn test_expert_aq_passthrough() {
        // Verify aq_enabled passes through to EncoderConfig
        let mut config = ExpertConfig::from_preset(OptimizationPreset::MozjpegBaseline, 85.0);
        assert!(!config.aq_enabled);

        let enc = config.to_encoder_config(ColorMode::YCbCr {
            subsampling: ChromaSubsampling::Quarter,
        });
        assert!(
            !enc.is_aq_enabled(),
            "aq_enabled should pass through to EncoderConfig"
        );

        // Now flip it and verify
        config.aq_enabled = true;
        let enc = config.to_encoder_config(ColorMode::YCbCr {
            subsampling: ChromaSubsampling::Quarter,
        });
        assert!(
            enc.is_aq_enabled(),
            "flipped aq_enabled should pass through"
        );
    }
}
