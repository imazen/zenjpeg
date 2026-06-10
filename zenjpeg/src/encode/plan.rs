//! Encode-plan resolution: every knob resolved to its encode-time value.
//!
//! Two layers live here:
//!
//! - [`resolve_quant_tables`] — the single source of truth for turning
//!   quality + table-family + chroma knobs into the three quantization
//!   tables and zero-bias parameters. The streaming encoder calls this at
//!   construction; [`EncoderConfig::resolve_plan`] calls the same function,
//!   so the plan can never drift from what the encoder actually does.
//! - [`EncodePlan`] — a pure, inspectable snapshot answering "what will
//!   this builder chain actually configure?" for humans, tests, and sweep
//!   provenance logs.
//!
//! [`EncoderConfig::resolve_plan`]: super::encoder_config::EncoderConfig::resolve_plan

use crate::quant::{self, ZeroBiasParams};
use crate::types::{ColorSpace, QuantTable, Subsampling};

use super::encoder_config::EncoderConfig;
use super::encoder_types::{
    ColorMode, DownsamplingMethod, HuffmanStrategy, ProgressiveScanMode, Quality, QuantTableSource,
    TinyFileMode,
};
use super::trellis::TrellisConfig;
use super::tuning::EncodingTables;

// ============================================================================
// Shared quant-table resolution (used by StreamingEncoder + EncodePlan)
// ============================================================================

/// Inputs for [`resolve_quant_tables`]. Mirrors the knobs the streaming
/// encoder builder carries; both `StreamingEncoder::new` and
/// [`EncoderConfig::resolve_plan`] construct this from their own state.
pub(crate) struct TableResolveInputs<'a> {
    pub quality: Quality,
    pub chroma_distance_scale: f32,
    pub chroma_quality: Option<u8>,
    pub quant_source: QuantTableSource,
    pub separate_chroma_tables: bool,
    pub encoding_tables: Option<&'a EncodingTables>,
    pub use_xyb: bool,
    pub is_420: bool,
    pub allow_16bit: bool,
}

/// Output of [`resolve_quant_tables`].
pub(crate) struct ResolvedTables {
    pub quant: (QuantTable, QuantTable, QuantTable),
    pub zero_bias: (ZeroBiasParams, ZeroBiasParams, ZeroBiasParams),
    /// Per-component butteraugli distances `[Y, Cb, Cr]` (or `[X, Y, B]`
    /// in XYB mode — note components 1, 2 receive the chroma scale there).
    pub distances: [f32; 3],
}

/// Resolve quality + table family + chroma knobs into quantization tables
/// and zero-bias parameters.
///
/// This is the exact logic the streaming encoder runs at construction
/// (moved verbatim from `StreamingEncoder::new`); keeping it in one place
/// guarantees [`EncodePlan`] reports the tables that will actually be used.
pub(crate) fn resolve_quant_tables(inputs: TableResolveInputs<'_>) -> ResolvedTables {
    let TableResolveInputs {
        quality,
        chroma_distance_scale,
        chroma_quality,
        quant_source,
        separate_chroma_tables,
        encoding_tables,
        use_xyb,
        is_420,
        allow_16bit,
    } = inputs;

    let distance = quality.to_distance();
    // Per-component distance: [Y, Cb, Cr]. The scalar
    // `chroma_distance_scale` multiplies the chroma distances
    // identically. `scale == 1.0` reproduces the single-distance
    // path bit-for-bit (verified by `chroma_scale_default_identity`).
    let chroma_distance = distance * chroma_distance_scale;
    let distances_per_component = [distance, chroma_distance, chroma_distance];
    let color_space = if use_xyb {
        ColorSpace::Xyb
    } else {
        ColorSpace::YCbCr
    };

    let (quant, zero_bias) = if let Some(tables) = encoding_tables {
        // Branch 1: Custom encoding tables provided explicitly
        let quant = tables.generate_quant_tables(distances_per_component, is_420);
        let zero_bias = tables.generate_zero_bias_all();
        // Apply allow_16bit clamping if needed
        let quant = if allow_16bit {
            quant
        } else {
            (
                quant.0.clamp_to_baseline(),
                quant.1.clamp_to_baseline(),
                quant.2.clamp_to_baseline(),
            )
        };
        (quant, zero_bias)
    } else if quant_source == QuantTableSource::MozjpegDefault {
        // Branch 2: Mozjpeg Robidoux tables with quality scaling
        // Use for_mozjpeg_tables() to preserve the original mozjpeg quality.
        // to_internal() remaps for jpegli's distance system, producing wrong tables.
        let quality_u8 = quality.for_mozjpeg_tables();
        let force_baseline = !allow_16bit;
        // Optional independent chroma quality. `None` → chroma
        // tables scaled with the same quality as luma (historical
        // behaviour; bit-identical to old callers). `Some(cq)`
        // → chroma table scaled with `cq` instead.
        let tables = super::tables::robidoux::generate_mozjpeg_default_tables_with_chroma(
            quality_u8,
            chroma_quality,
            force_baseline,
        );
        let quant = tables.generate_quant_tables(distances_per_component, is_420);
        let zero_bias = tables.generate_zero_bias_all();
        (quant, zero_bias)
    } else {
        // Branch 3: Jpegli perceptual defaults (original path)
        //
        // When separate_chroma_tables is false (2-table mode, jpeg_set_quality),
        // use the Cr base matrix for both Cb and Cr tables. This matches C++
        // jpegli behavior where the single chroma table uses the Cr matrix.
        let cb_component = if separate_chroma_tables { 1 } else { 2 };

        // Luma uses the user's quality verbatim; chroma gets the
        // scaled distance. When chroma_scale == 1.0 the
        // `with_distance` call produces bit-identical tables to
        // the old `ex`-variant (see quant_table_identity test).
        let quant = (
            quant::generate_quant_table_ex(quality, 0, color_space, use_xyb, is_420, allow_16bit),
            quant::generate_quant_table_with_distance(
                distances_per_component[1],
                cb_component,
                color_space,
                use_xyb,
                is_420,
                allow_16bit,
            ),
            quant::generate_quant_table_with_distance(
                distances_per_component[2],
                2,
                color_space,
                use_xyb,
                is_420,
                allow_16bit,
            ),
        );

        // Compute effective distance for quality-adaptive zero bias.
        // Color-space aware: XYB tables invert against the XYB base
        // matrix + GLOBAL_SCALE_XYB; YCbCr tables invert against
        // their own matrix + GLOBAL_SCALE_YCBCR.
        let effective_distance =
            quant::quant_vals_to_distance(&quant.0, &quant.1, &quant.2, use_xyb);

        // Auto-select zero bias based on color mode
        let zero_bias = if use_xyb {
            (
                ZeroBiasParams::for_xyb(effective_distance, 0),
                ZeroBiasParams::for_xyb(effective_distance, 1),
                ZeroBiasParams::for_xyb(effective_distance, 2),
            )
        } else {
            (
                ZeroBiasParams::for_ycbcr(effective_distance, 0),
                ZeroBiasParams::for_ycbcr(effective_distance, 1),
                ZeroBiasParams::for_ycbcr(effective_distance, 2),
            )
        };

        (quant, zero_bias)
    };

    ResolvedTables {
        quant,
        zero_bias,
        distances: distances_per_component,
    }
}

// ============================================================================
// EncodePlan — public introspection
// ============================================================================

/// Which SOF (start-of-frame) marker the encoder will emit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SofMarker {
    /// SOF0 — baseline sequential DCT.
    Baseline,
    /// SOF1 — extended sequential DCT (XYB color mode or 16-bit DQT).
    Extended,
    /// SOF2 — progressive DCT.
    Progressive,
}

/// Fully resolved encode-time plan: what the encoder will actually do for
/// an image of the given dimensions.
///
/// Produced by [`EncoderConfig::resolve_plan`]. Pure — nothing is encoded.
/// `Debug` for structured dumps, `Display` for a compact human summary.
/// Useful for audits ("what did `auto_optimize` actually set?"), golden
/// tests, and sweep provenance columns.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct EncodePlan {
    /// Internal jpegli-scale quality (0–100) after unit conversion.
    pub internal_quality: f32,
    /// Per-component butteraugli distances `[Y, Cb, Cr]` (`[X, Y, B]` for
    /// XYB — components 1, 2 carry the chroma scale there).
    pub distances: [f32; 3],
    /// Color path (YCbCr subsampling / XYB B-subsampling / grayscale).
    pub color_mode: ColorMode,
    /// Table family driving quantization.
    pub table_source: QuantTableSource,
    /// Whether custom `EncodingTables` override the family.
    pub custom_tables: bool,
    /// 3-table (separate Cb/Cr) vs 2-table (shared chroma) layout.
    pub separate_chroma_tables: bool,
    /// Maximum quantization value per component table (after clamping).
    pub quant_max: [u16; 3],
    /// Whether each DQT table needs 16-bit precision.
    pub dqt_16bit: [bool; 3],
    /// Chroma distance multiplier (jpegli path).
    pub chroma_distance_scale: f32,
    /// Independent chroma quality override (mozjpeg path).
    pub chroma_quality: Option<u8>,
    /// Adaptive quantization enabled.
    pub aq_enabled: bool,
    /// Overshoot deringing enabled.
    pub deringing: bool,
    /// Coefficient optimization: `None` = plain zero-bias rounding;
    /// `Some` = trellis, with [`TrellisConfig::aq_coupling`] describing any
    /// per-block AQ→lambda coupling.
    pub trellis: Option<TrellisConfig>,
    /// Scan mode (baseline / progressive variant / scan search).
    pub scan_mode: ProgressiveScanMode,
    /// Whether Huffman tables are optimized in a second pass.
    pub optimized_huffman: bool,
    /// Chroma downsampling algorithm (RGB input only).
    pub downsampling: DownsamplingMethod,
    /// Pre-encode Gaussian blur sigma (0.0 = off).
    pub pre_blur_sigma: f32,
    /// Restart interval in MCUs (0 = no restart markers).
    pub restart_interval_mcus: u16,
    /// Tiny-file optimizations active for this image size.
    pub tiny_file_active: bool,
    /// SOF marker that will be emitted.
    pub sof: SofMarker,
}

impl core::fmt::Display for EncodePlan {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        writeln!(
            f,
            "EncodePlan: q={:.1} d=[{:.3}, {:.3}, {:.3}] {:?}",
            self.internal_quality,
            self.distances[0],
            self.distances[1],
            self.distances[2],
            self.color_mode,
        )?;
        writeln!(
            f,
            "  tables: {:?}{} ({}-table) quant_max=[{}, {}, {}]{}",
            self.table_source,
            if self.custom_tables { "+custom" } else { "" },
            if self.separate_chroma_tables { 3 } else { 2 },
            self.quant_max[0],
            self.quant_max[1],
            self.quant_max[2],
            if self.dqt_16bit.iter().any(|&b| b) {
                " (16-bit DQT)"
            } else {
                ""
            },
        )?;
        writeln!(
            f,
            "  chroma: scale={:.2} quality={:?}",
            self.chroma_distance_scale, self.chroma_quality
        )?;
        match &self.trellis {
            None => writeln!(f, "  coeff-opt: none (zero-bias rounding)")?,
            Some(t) => writeln!(
                f,
                "  coeff-opt: trellis λ1={:.2} λ2={:.2} dc={} coupling={}",
                t.lambda_log_scale1,
                t.lambda_log_scale2,
                t.dc_enabled,
                if t.aq_coupling.is_active() {
                    format!(
                        "scale {:+.2} exp {:.2} max ±{:.2}",
                        t.aq_coupling.scale, t.aq_coupling.exponent, t.aq_coupling.max_adjustment
                    )
                } else {
                    "off".to_string()
                },
            )?,
        }
        writeln!(
            f,
            "  scan: {:?} ({:?}) huffman-optimized={} aq={} deringing={}",
            self.scan_mode, self.sof, self.optimized_huffman, self.aq_enabled, self.deringing,
        )?;
        write!(
            f,
            "  misc: downsample={:?} pre_blur={:.2} restart={} MCUs tiny_file={}",
            self.downsampling,
            self.pre_blur_sigma,
            self.restart_interval_mcus,
            self.tiny_file_active,
        )
    }
}

impl EncoderConfig {
    /// Resolve every knob to its encode-time value for an image of the
    /// given dimensions, without encoding anything.
    ///
    /// The quantization tables are produced by the same code path the
    /// encoder runs at construction, so the plan cannot drift from
    /// reality. Use this to audit builder chains:
    ///
    /// ```
    /// use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig};
    ///
    /// let plan = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
    ///     .auto_optimize(true)
    ///     .resolve_plan(1920, 1080);
    /// assert!(plan.trellis.is_some()); // auto_optimize enabled trellis
    /// println!("{plan}");
    /// ```
    #[must_use]
    pub fn resolve_plan(&self, width: u32, height: u32) -> EncodePlan {
        let (use_xyb, subsampling) = match self.color_mode {
            ColorMode::YCbCr { subsampling } => (false, subsampling.into()),
            ColorMode::Xyb { .. } => (true, Subsampling::S444),
            ColorMode::Grayscale => (false, Subsampling::S444),
        };
        let is_420 = subsampling == Subsampling::S420;

        let custom = self.quant_table_config.custom_tables();
        let resolved = resolve_quant_tables(TableResolveInputs {
            quality: self.quality,
            chroma_distance_scale: self.chroma_distance_scale,
            chroma_quality: self.chroma_quality,
            quant_source: self.quant_table_config.quant_source(),
            separate_chroma_tables: self.quant_table_config.separate_chroma_tables(),
            encoding_tables: custom.as_ref(),
            use_xyb,
            is_420,
            allow_16bit: self.allow_16bit_quant_tables,
        });

        let quant_max = [
            resolved.quant.0.values.iter().copied().max().unwrap_or(1),
            resolved.quant.1.values.iter().copied().max().unwrap_or(1),
            resolved.quant.2.values.iter().copied().max().unwrap_or(1),
        ];
        let dqt_16bit = [
            resolved.quant.0.precision > 0,
            resolved.quant.1.precision > 0,
            resolved.quant.2.precision > 0,
        ];

        // Mirror byte_encoders: progressive suppresses restart markers
        // unless explicitly forced.
        let restart_interval_mcus = if self.scan_mode.is_progressive()
            && !self.force_restart_markers
        {
            0
        } else {
            super::config::resolve_restart_rows(self.restart_mcu_rows, width, height, subsampling)
        };

        let is_color = !matches!(self.color_mode, ColorMode::Grayscale);
        // Mirror StreamingEncoder: tiny-file requires non-XYB sequential
        // encoding with optimized Huffman tables.
        let tiny_file_eligible = !use_xyb
            && !self.scan_mode.is_progressive()
            && matches!(self.huffman, HuffmanStrategy::Optimize);
        let tiny_file_active = if !tiny_file_eligible {
            false
        } else {
            match self.tiny_file_mode {
                TinyFileMode::Off => false,
                TinyFileMode::Force => true,
                TinyFileMode::Auto => {
                    super::encoder_types::should_activate_tiny_file_mode_for_subsampling(
                        width,
                        height,
                        is_color,
                        subsampling,
                    )
                }
            }
        };

        // SOF resolution mirrors the serializer: progressive → SOF2; XYB
        // forces SOF1; 16-bit DQT requires SOF1; otherwise SOF0.
        let sof = if self.scan_mode.is_progressive() {
            SofMarker::Progressive
        } else if use_xyb || dqt_16bit.iter().any(|&b| b) {
            SofMarker::Extended
        } else {
            SofMarker::Baseline
        };

        let optimized_huffman =
            matches!(self.huffman, HuffmanStrategy::Optimize) || self.scan_mode.is_progressive();

        EncodePlan {
            internal_quality: self.quality.to_internal(),
            distances: resolved.distances,
            color_mode: self.color_mode,
            table_source: self.quant_table_config.quant_source(),
            custom_tables: custom.is_some(),
            separate_chroma_tables: self.quant_table_config.separate_chroma_tables(),
            quant_max,
            dqt_16bit,
            chroma_distance_scale: self.chroma_distance_scale,
            chroma_quality: self.chroma_quality,
            aq_enabled: self.aq_enabled,
            deringing: self.deringing,
            trellis: self.trellis,
            scan_mode: self.scan_mode,
            optimized_huffman,
            downsampling: self.downsampling_method,
            pre_blur_sigma: self.pre_blur,
            restart_interval_mcus,
            tiny_file_active,
            sof,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::encoder_types::ChromaSubsampling;

    #[test]
    fn auto_optimize_visible_in_plan() {
        let plan = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
            .auto_optimize(true)
            .resolve_plan(512, 512);
        let t = plan.trellis.expect("auto_optimize enables trellis at q85");
        assert!((t.lambda_log_scale1 - 14.5).abs() < 1e-6);
        assert!(!t.dc_enabled);
        assert!(!t.aq_coupling.is_active());
        assert_eq!(plan.scan_mode, ProgressiveScanMode::Progressive);
        assert_eq!(plan.sof, SofMarker::Progressive);
    }

    #[test]
    fn auto_optimize_below_gate_keeps_plain_quantization() {
        let plan = EncoderConfig::ycbcr(30, ChromaSubsampling::Quarter)
            .auto_optimize(true)
            .resolve_plan(512, 512);
        assert!(plan.trellis.is_none(), "q30 is below the d<5.0 gate");
        // ... but the scan-mode side effect still applies.
        assert_eq!(plan.scan_mode, ProgressiveScanMode::Progressive);
    }

    #[test]
    fn default_plan_is_progressive_no_trellis() {
        let plan = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter).resolve_plan(512, 512);
        assert!(plan.trellis.is_none());
        assert!(plan.optimized_huffman);
        assert_eq!(plan.distances[1], plan.distances[0]); // chroma scale 1.0
        assert!(plan.separate_chroma_tables);
        assert_eq!(plan.restart_interval_mcus, 0); // progressive suppresses RST
    }

    #[test]
    fn chroma_distance_scale_moves_chroma_distances() {
        let plan = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
            .chroma_distance_scale(2.0)
            .resolve_plan(512, 512);
        assert!((plan.distances[1] - plan.distances[0] * 2.0).abs() < 1e-6);
        assert_eq!(plan.distances[1], plan.distances[2]);
    }

    #[test]
    fn low_quality_clamps_to_8bit_by_default() {
        let plan = EncoderConfig::ycbcr(50, ChromaSubsampling::Quarter)
            .progressive(false)
            .resolve_plan(512, 512);
        // allow_16bit defaults to false: everything clamped to 255.
        assert!(plan.quant_max.iter().all(|&m| m <= 255));
        assert!(!plan.dqt_16bit.iter().any(|&b| b));
        assert_eq!(plan.sof, SofMarker::Baseline);
    }

    #[test]
    fn sixteen_bit_tables_resolve_to_extended_sof() {
        let plan = EncoderConfig::ycbcr(50, ChromaSubsampling::Quarter)
            .progressive(false)
            .allow_16bit_quant_tables(true)
            .resolve_plan(512, 512);
        // Q50 chroma exceeds 255 (quality-dependent, per docs).
        assert!(plan.dqt_16bit.iter().any(|&b| b));
        assert_eq!(plan.sof, SofMarker::Extended);
    }

    #[test]
    fn display_is_compact_and_total() {
        let plan = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
            .auto_optimize(true)
            .resolve_plan(512, 512);
        let s = format!("{plan}");
        assert!(s.contains("trellis"));
        assert!(s.contains("λ1=14.50"));
    }
}
