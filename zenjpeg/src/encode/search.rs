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
    ColorMode, DownsamplingMethod, HuffmanStrategy, OptimizationPreset, Quality, QuantTableConfig,
    ScanMode,
};
use super::mozjpeg_compat::{TrellisConfig, TrellisSpeedMode};
use super::tuning::{EncodingTables, PerComponent};
use crate::hybrid::config::HybridConfig;

/// All tunable encoder parameters for external optimization.
///
/// Every field is `pub` for direct mutation. Use [`from_preset()`](Self::from_preset)
/// for known-good starting points, mutate freely, then call
/// [`to_encoder_config()`](Self::to_encoder_config) to encode.
///
/// # Ignored Parameters
///
/// Some parameters are ignored depending on mode:
///
/// | Parameter | Ignored when... |
/// |-----------|-----------------|
/// | `quality` (quant scaling) | `tables.scaling == Exact` (quality still affects zero-bias blend) |
/// | `zero_bias_hq/lq/distances` | XYB mode (uses uniform 0.5) |
/// | `downsampling_method` | 4:4:4 or grayscale (no downsampling) |
/// | `trellis_*` fields | `trellis_enabled == false` |
/// | `aq_trellis_*` fields | `aq_trellis_coupling == 0.0` or `trellis_enabled == false` |
/// | `allow_16bit_quant_tables` | All quant values <= 255 at current quality |
///
/// # Hybrid-Mode Limitations
///
/// When `aq_trellis_coupling > 0.0`, the hybrid path creates per-block trellis
/// configs via [`HybridConfig::to_trellis_config()`]. That method builds from
/// `TrellisConfig::default()` and only forwards a subset of fields. These
/// `ExpertConfig` trellis fields are **ignored in hybrid mode**:
///
/// | Field | Hybrid behavior | Standalone behavior |
/// |-------|----------------|---------------------|
/// | `trellis_eob_opt` | Always `true` (hardcoded) | Forwarded |
/// | `trellis_speed_mode` | Always `Adaptive` (default) | Forwarded |
/// | `trellis_delta_dc_weight` | Always `0.0` (default) | Forwarded |
/// | `trellis_use_lambda_weight_tbl` | Stored but not forwarded to per-block config | Forwarded |
///
/// This is a limitation of the current `HybridConfig::to_trellis_config()` API
/// (pre-existing, not introduced by `ExpertConfig`). Fields that ARE forwarded
/// in both modes: `trellis_dc_enabled`, `trellis_lambda_log_scale1/2`,
/// `trellis_num_loops`.
#[derive(Clone, Debug)]
pub struct ExpertConfig {
    // === Quantization Tables ===
    /// Base quantization tables, zero-bias multipliers/offsets, and quality scaling.
    ///
    /// This is the primary data that controls quantization. The `quant` field holds
    /// base quantization matrices (3 components x 64 coefficients). The `zero_bias_mul`
    /// field holds the blended zero-bias multipliers (computed by
    /// [`blend_zero_bias()`](Self::blend_zero_bias) from the HQ/LQ endpoints below).
    ///
    /// When `tables.scaling == ScalingParams::Exact`, the `quality` field has no
    /// effect on quant table values (they're used as-is). Quality still affects
    /// zero-bias blend and AQ dampen factor.
    ///
    /// When `tables.scaling == ScalingParams::Scaled`, the encoder applies
    /// per-frequency non-linear scaling controlled by `global_scale` and
    /// `frequency_exponents`.
    pub tables: EncodingTables,

    // === Zero-Bias Blend Control ===
    /// HQ zero-bias multiplier tables (endpoint at high quality / low distance).
    ///
    /// These are the zero-bias values used when distance <= `zero_bias_hq_distance`.
    /// At intermediate distances, the encoder linearly blends between HQ and LQ.
    ///
    /// Default: C++ jpegli's `kZeroBiasMulYCbCrHQ` tables.
    ///
    /// Ignored in XYB mode (XYB uses uniform 0.5 for all AC coefficients).
    pub zero_bias_hq: PerComponent<[f32; 64]>,

    /// LQ zero-bias multiplier tables (endpoint at low quality / high distance).
    ///
    /// These are the zero-bias values used when distance >= `zero_bias_lq_distance`.
    ///
    /// Default: C++ jpegli's `kZeroBiasMulYCbCrLQ` tables.
    ///
    /// Ignored in XYB mode.
    pub zero_bias_lq: PerComponent<[f32; 64]>,

    /// Distance at or below which zero-bias is fully HQ.
    ///
    /// Default: `1.0`. Must be less than `zero_bias_lq_distance`.
    ///
    /// Ignored in XYB mode.
    pub zero_bias_hq_distance: f32,

    /// Distance at or above which zero-bias is fully LQ.
    ///
    /// Default: `3.0`. Must be greater than `zero_bias_hq_distance`.
    ///
    /// Ignored in XYB mode.
    pub zero_bias_lq_distance: f32,

    // === Trellis Quantization ===
    /// Master switch for trellis quantization.
    ///
    /// When `false`, all other `trellis_*` and `aq_trellis_*` fields are ignored.
    /// The encoder uses simple rounding (jpegli default).
    ///
    /// When `true`, the trellis mode depends on `aq_trellis_coupling`:
    /// - `== 0.0`: Standalone trellis (mozjpeg-style). All `trellis_*` fields forwarded.
    /// - `> 0.0`: Hybrid AQ-coupled trellis. See "Hybrid-Mode Limitations" above.
    pub trellis_enabled: bool,

    /// Enable DC coefficient trellis (cross-block DC optimization).
    ///
    /// Forwarded in both standalone and hybrid modes.
    ///
    /// Default: `true`.
    pub trellis_dc_enabled: bool,

    /// Enable EOB run optimization (cross-block zero runs).
    ///
    /// **Standalone mode only.** In hybrid mode, EOB optimization is always enabled
    /// (hardcoded in `HybridConfig::to_trellis_config()`).
    ///
    /// Default: `false` (matching C mozjpeg default).
    pub trellis_eob_opt: bool,

    /// Use perceptual lambda weighting table.
    ///
    /// **Standalone mode only.** In hybrid mode, this field is stored in
    /// `HybridConfig` but not forwarded to per-block trellis configs
    /// (they default to `true`).
    ///
    /// Default: `true`.
    pub trellis_use_lambda_weight_tbl: bool,

    /// Lambda log scale 1 (rate penalty). Higher = smaller files.
    ///
    /// The effective lambda is: `2^scale1 / (2^scale2 + block_norm)`.
    /// In hybrid mode, this is the base value before per-block AQ adjustment.
    ///
    /// Forwarded in both standalone and hybrid modes.
    ///
    /// Default: `14.75`. Typical range: `~12.0`-`17.0`.
    pub trellis_lambda_log_scale1: f32,

    /// Lambda log scale 2 (distortion sensitivity). Higher = more quality.
    ///
    /// Controls the denominator in the lambda formula.
    ///
    /// Forwarded in both standalone and hybrid modes.
    ///
    /// Default: `16.5`. Typical range: `~14.0`-`18.0`.
    pub trellis_lambda_log_scale2: f32,

    /// Number of trellis optimization loops.
    ///
    /// Multiple loops can improve results but with diminishing returns.
    ///
    /// Forwarded in both standalone and hybrid modes.
    ///
    /// Default: `1`.
    pub trellis_num_loops: i32,

    /// Speed optimization mode for high-entropy blocks.
    ///
    /// **Standalone mode only.** In hybrid mode, the per-block trellis config
    /// defaults to `TrellisSpeedMode::Adaptive`.
    ///
    /// Default: `TrellisSpeedMode::Adaptive`.
    pub trellis_speed_mode: TrellisSpeedMode,

    /// Weight for vertical DC gradient penalty.
    ///
    /// When > 0.0, DC trellis penalizes large vertical DC jumps between blocks,
    /// reducing visible banding artifacts.
    ///
    /// **Standalone mode only.** In hybrid mode, the per-block trellis config
    /// defaults to `0.0`.
    ///
    /// Default: `0.0` (disabled, matching C mozjpeg default).
    pub trellis_delta_dc_weight: f32,

    // === AQ->Trellis Coupling ===
    /// Per-unit AQ strength to lambda adjustment.
    ///
    /// Controls the transition between standalone and hybrid trellis:
    /// - `0.0`: Pure standalone trellis (no AQ influence on lambda).
    /// - `> 0.0`: Hybrid mode. Trellis lambda is adjusted per-block:
    ///   `effective_scale1 = trellis_lambda_log_scale1 + aq_strength^aq_trellis_exponent * coupling`
    ///
    /// All `aq_trellis_*` fields below are ignored when this is `0.0` or when
    /// `trellis_enabled == false`.
    ///
    /// Default: `0.0`. Range: `0.0`-`8.0`.
    pub aq_trellis_coupling: f32,

    /// Non-linear AQ mapping exponent.
    ///
    /// Applied to AQ strength before coupling: `aq_strength.powf(exponent)`.
    /// - `1.0` = linear
    /// - `0.5` = sqrt (compress high AQ, expand low AQ)
    /// - `2.0` = squared (expand high AQ, compress low AQ)
    ///
    /// Default: `1.0`.
    pub aq_trellis_exponent: f32,

    /// Minimum AQ strength before coupling kicks in.
    ///
    /// Blocks with AQ strength below this threshold use the base lambda unchanged.
    ///
    /// Default: `0.0`.
    pub aq_trellis_threshold: f32,

    /// Scale coupling for chroma (Cb/Cr) components.
    ///
    /// `1.0` = same coupling as luma. `< 1.0` = less aggressive trellis on chroma.
    ///
    /// Default: `1.0`.
    pub aq_trellis_chroma_scale: f32,

    /// Scale coupling by quality-derived dampen factor.
    ///
    /// When `true`, the AQ lambda adjustment is multiplied by a dampen factor
    /// that decreases at lower quality levels, making coupling less aggressive.
    ///
    /// Default: `false`.
    pub aq_trellis_quality_adaptive: bool,

    // === Encoder Strategy ===
    /// Scan mode (baseline vs progressive variants).
    ///
    /// Progressive modes automatically enable optimized Huffman tables in
    /// [`to_encoder_config()`](Self::to_encoder_config).
    pub scan_mode: ScanMode,

    /// Enable overshoot deringing.
    ///
    /// Smooths hard edges to reduce visible ringing artifacts, especially on
    /// white backgrounds. Negligible quality/speed cost for photographic content.
    ///
    /// Default: `true` for jpegli/hybrid, `false` for mozjpeg baseline/progressive.
    pub deringing: bool,

    /// Allow 16-bit quantization tables (SOF1 extended JPEG).
    ///
    /// When `false`, quant values are clamped to 255 for baseline compatibility.
    /// When `true`, tables can use values up to 32767 for higher precision at
    /// low quality levels (below ~Q86 for chroma).
    ///
    /// Default: `false` (matching both cjpegli CLI and C mozjpeg).
    pub allow_16bit_quant_tables: bool,

    /// Quality parameter.
    ///
    /// Affects quant table scaling (when `tables.scaling == Scaled`), zero-bias
    /// blend distance computation, and AQ dampen factor. Accepts any type that
    /// converts to [`Quality`]: `f32`, `u8`, `i32`, or explicit `Quality::*` variants.
    ///
    /// When using [`blend_zero_bias()`](Self::blend_zero_bias), quality is converted
    /// to Butteraugli distance internally to determine the HQ/LQ blend position.
    pub quality: Quality,

    /// Chroma downsampling method.
    ///
    /// Only affects RGB input with chroma subsampling (4:2:0, 4:2:2, etc.).
    /// Ignored for 4:4:4, grayscale, and pre-subsampled YCbCr input.
    pub downsampling_method: DownsamplingMethod,
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
            trellis_eob_opt: false,
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

            scan_mode: ScanMode::Progressive,
            deringing: true,
            allow_16bit_quant_tables: false,
            quality,
            downsampling_method: DownsamplingMethod::default(),
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
                let q_internal = quality.to_internal();
                let mozjpeg_tables = super::mozjpeg_table_data::generate_mozjpeg_default_tables(
                    q_internal as u8,
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
            JpegliBaseline | MozjpegBaseline | HybridBaseline => ScanMode::Baseline,
            JpegliProgressive | HybridProgressive => ScanMode::Progressive,
            MozjpegProgressive => ScanMode::ProgressiveMozjpeg,
            MozjpegMaxCompression | HybridMaxCompression => ScanMode::ProgressiveSearch,
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
            trellis_eob_opt: false,
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

            scan_mode,
            deringing,
            allow_16bit_quant_tables: false,
            quality,
            downsampling_method: DownsamplingMethod::default(),
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
    /// configs (see struct-level docs): `trellis_eob_opt`, `trellis_speed_mode`,
    /// `trellis_delta_dc_weight`, `trellis_use_lambda_weight_tbl`.
    fn build_trellis_or_hybrid(&self) -> (Option<TrellisConfig>, HybridConfig) {
        if !self.trellis_enabled {
            return (None, HybridConfig::disabled());
        }

        if self.aq_trellis_coupling > 0.0 {
            // Hybrid mode: AQ-coupled trellis.
            // Note: trellis_eob_opt, trellis_speed_mode, trellis_delta_dc_weight,
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
            };
            (None, hybrid)
        } else {
            // Standalone trellis: all fields forwarded directly.
            let trellis = TrellisConfig {
                enabled: true,
                dc_enabled: self.trellis_dc_enabled,
                eob_opt: self.trellis_eob_opt,
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
        assert_eq!(config.scan_mode, ScanMode::Progressive);
        assert!(!config.allow_16bit_quant_tables);
        // Default uses jpegli scaling, not exact
        assert!(config.uses_quality_scaling());
    }

    #[test]
    fn test_from_preset_jpegli_baseline() {
        let config = ExpertConfig::from_preset(OptimizationPreset::JpegliBaseline, 85.0);
        assert!(!config.trellis_enabled);
        assert!(config.deringing);
        assert_eq!(config.scan_mode, ScanMode::Baseline);
        assert!(config.uses_quality_scaling());
    }

    #[test]
    fn test_from_preset_jpegli_progressive() {
        let config = ExpertConfig::from_preset(OptimizationPreset::JpegliProgressive, 85.0);
        assert!(!config.trellis_enabled);
        assert!(config.deringing);
        assert_eq!(config.scan_mode, ScanMode::Progressive);
    }

    #[test]
    fn test_from_preset_mozjpeg_baseline() {
        let config = ExpertConfig::from_preset(OptimizationPreset::MozjpegBaseline, 85.0);
        assert!(config.trellis_enabled);
        assert!(!config.deringing);
        assert_eq!(config.scan_mode, ScanMode::Baseline);
        assert_eq!(config.trellis_speed_mode, TrellisSpeedMode::Thorough);
        // Mozjpeg uses exact (pre-scaled) tables
        assert!(!config.uses_quality_scaling());
    }

    #[test]
    fn test_from_preset_mozjpeg_progressive() {
        let config = ExpertConfig::from_preset(OptimizationPreset::MozjpegProgressive, 85.0);
        assert!(config.trellis_enabled);
        assert!(!config.deringing);
        assert_eq!(config.scan_mode, ScanMode::ProgressiveMozjpeg);
    }

    #[test]
    fn test_from_preset_mozjpeg_max_compression() {
        let config = ExpertConfig::from_preset(OptimizationPreset::MozjpegMaxCompression, 85.0);
        assert!(config.trellis_enabled);
        assert!(config.deringing);
        assert_eq!(config.scan_mode, ScanMode::ProgressiveSearch);
        assert_eq!(config.trellis_speed_mode, TrellisSpeedMode::Thorough);
    }

    #[test]
    fn test_from_preset_hybrid_baseline() {
        let config = ExpertConfig::from_preset(OptimizationPreset::HybridBaseline, 85.0);
        assert!(config.trellis_enabled);
        assert!(config.deringing);
        assert_eq!(config.scan_mode, ScanMode::Baseline);
        assert_eq!(config.trellis_speed_mode, TrellisSpeedMode::Adaptive);
        // Hybrid starts uncoupled
        assert_eq!(config.aq_trellis_coupling, 0.0);
    }

    #[test]
    fn test_from_preset_hybrid_progressive() {
        let config = ExpertConfig::from_preset(OptimizationPreset::HybridProgressive, 85.0);
        assert!(config.trellis_enabled);
        assert!(config.deringing);
        assert_eq!(config.scan_mode, ScanMode::Progressive);
        assert_eq!(config.trellis_speed_mode, TrellisSpeedMode::Adaptive);
    }

    #[test]
    fn test_from_preset_hybrid_max_compression() {
        let config = ExpertConfig::from_preset(OptimizationPreset::HybridMaxCompression, 85.0);
        assert!(config.trellis_enabled);
        assert!(config.deringing);
        assert_eq!(config.scan_mode, ScanMode::ProgressiveSearch);
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
        assert_eq!(enc.scan_mode, ScanMode::Progressive);
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
        expert.trellis_eob_opt = true;
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
        assert!(trellis.eob_opt);
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
        expert.scan_mode = ScanMode::ProgressiveSearch;

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
}
