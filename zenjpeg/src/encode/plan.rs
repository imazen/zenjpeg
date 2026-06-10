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
    ColorMode, DownsamplingMethod, HuffmanStrategy, ProgressiveScanMode, Quality, QuantTableConfig,
    TinyFileMode,
};
use super::trellis::TrellisConfig;

// ============================================================================
// Shared quant-table resolution (used by StreamingEncoder + EncodePlan)
// ============================================================================

/// Per-component butteraugli distances — the internal quality currency.
///
/// Every quality front-end (the six [`Quality`] unit systems, the jpegli
/// families' `chroma_distance_scales`) resolves into this vector before
/// any table or zero-bias math runs. Component order follows the JPEG
/// component order: `[Y, Cb, Cr]` in YCbCr mode, `[X, Y, B]` in XYB mode.
pub(crate) struct ResolvedQuality {
    /// Per-component distances feeding table scaling and (when divergent)
    /// per-channel zero-bias.
    pub distances: [f32; 3],
    /// All components share the base distance (neutral scales). The
    /// uniform path uses the joint three-component table inversion for
    /// zero-bias — bit-identical to C++ jpegli parity behaviour.
    pub uniform: bool,
}

/// Resolve `Quality` + the table family's distance knobs into the
/// per-component distance vector.
pub(crate) fn resolve_quality(
    quality: Quality,
    table_config: &QuantTableConfig,
    use_xyb: bool,
) -> ResolvedQuality {
    let d = quality.to_distance();
    match table_config.chroma_distance_scales() {
        Some(scales) => {
            let s = [scales[0].clamp(0.1, 5.0), scales[1].clamp(0.1, 5.0)];
            // The scales apply to the chroma-like channels in ascending
            // component order. YCbCr: Cb, Cr (components 1, 2). XYB:
            // X, B (components 0, 2); Y keeps the base distance.
            let distances = if use_xyb {
                [d * s[0], d, d * s[1]]
            } else {
                [d, d * s[0], d * s[1]]
            };
            ResolvedQuality {
                distances,
                uniform: s[0] == 1.0 && s[1] == 1.0,
            }
        }
        None => ResolvedQuality {
            distances: [d; 3],
            uniform: true,
        },
    }
}

/// Inputs for [`resolve_quant_tables`]. Both `StreamingEncoder::new` and
/// [`EncoderConfig::resolve_plan`] construct this from their own state.
pub(crate) struct TableResolveInputs<'a> {
    pub quality: Quality,
    pub table_config: &'a QuantTableConfig,
    pub use_xyb: bool,
    pub is_420: bool,
    pub allow_16bit: bool,
}

/// Output of [`resolve_quant_tables`].
pub(crate) struct ResolvedTables {
    pub quant: (QuantTable, QuantTable, QuantTable),
    pub zero_bias: (ZeroBiasParams, ZeroBiasParams, ZeroBiasParams),
    /// Per-component distance inputs to table scaling. `[Y, Cb, Cr]` in
    /// YCbCr mode; `[X, Y, B]` in XYB mode (chroma-like X and B carry the
    /// scale, Y keeps the base distance). Informational for `Exact`
    /// families, which ignore distances.
    pub distances: [f32; 3],
    /// Whether the config's `Quality` affects the produced tables
    /// (false only for `Custom` tables with `ScalingParams::Exact`).
    pub quality_drives_tables: bool,
}

/// Resolve the table family + its live knobs into quantization tables and
/// zero-bias parameters.
///
/// This is the logic the streaming encoder runs at construction; keeping
/// it in one place guarantees [`EncodePlan`] reports the tables that will
/// actually be used. The `match` IS the family discrimination — each
/// variant consumes exactly its own knobs.
pub(crate) fn resolve_quant_tables(inputs: TableResolveInputs<'_>) -> ResolvedTables {
    let TableResolveInputs {
        quality,
        table_config,
        use_xyb,
        is_420,
        allow_16bit,
    } = inputs;

    let rq = resolve_quality(quality, table_config, use_xyb);
    let distance = quality.to_distance();
    let color_space = if use_xyb {
        ColorSpace::Xyb
    } else {
        ColorSpace::YCbCr
    };

    let clamp_baseline = |quant: (QuantTable, QuantTable, QuantTable)| {
        if allow_16bit {
            quant
        } else {
            (
                quant.0.clamp_to_baseline(),
                quant.1.clamp_to_baseline(),
                quant.2.clamp_to_baseline(),
            )
        }
    };

    match table_config {
        QuantTableConfig::Custom(tables) => {
            // Caller-provided tables own the chroma policy through their
            // per-component base matrices, so distances are uniform.
            // `Scaled` tables consume them; `Exact` tables ignore quality
            // entirely.
            let distances = [distance; 3];
            let quant = clamp_baseline(tables.generate_quant_tables(distances, is_420));
            let zero_bias = tables.generate_zero_bias_all();
            ResolvedTables {
                quant,
                zero_bias,
                distances,
                quality_drives_tables: !tables.is_exact(),
            }
        }
        QuantTableConfig::PiecewiseV4 => {
            // SA-piecewise anchors are indexed by quality; derive it from
            // the config's one quality knob (internal jpegli scale). The
            // generator clamps to its trained Q5-Q100 anchor range.
            let q = quality.to_internal().round().clamp(1.0, 100.0) as u8;
            let tables = super::tables::sa_piecewise_v4::tables_for_quality(q);
            let distances = [distance; 3];
            let quant = clamp_baseline(tables.generate_quant_tables(distances, is_420));
            let zero_bias = tables.generate_zero_bias_all();
            ResolvedTables {
                quant,
                zero_bias,
                distances,
                quality_drives_tables: true,
            }
        }
        QuantTableConfig::GlassaLowBpp => {
            // Glassa anchors are indexed by quality; derive it from the
            // config's one quality knob (internal jpegli scale, clamped
            // to the trained 3–25 anchor range).
            let glassa_q = quality.to_internal().round().clamp(3.0, 25.0) as u8;
            let tables = super::tables::glassa::tables_for_quality(glassa_q);
            let distances = [distance; 3];
            let quant = clamp_baseline(tables.generate_quant_tables(distances, is_420));
            let zero_bias = tables.generate_zero_bias_all();
            ResolvedTables {
                quant,
                zero_bias,
                distances,
                quality_drives_tables: true,
            }
        }
        QuantTableConfig::MozjpegRobidoux { chroma_quality } => {
            // Robidoux tables are ScalingParams::Exact — pre-scaled by
            // libjpeg's quality formula; the distances are informational.
            // Use for_mozjpeg_tables() to preserve the original mozjpeg
            // quality (to_internal() remaps for jpegli's distance system,
            // producing wrong tables).
            let quality_u8 = quality.for_mozjpeg_tables();
            let force_baseline = !allow_16bit;
            let tables = super::tables::robidoux::generate_mozjpeg_default_tables_with_chroma(
                quality_u8,
                chroma_quality.map(|q| q.clamp(1, 100)),
                force_baseline,
            );
            let distances = [distance; 3];
            let quant = tables.generate_quant_tables(distances, is_420);
            let zero_bias = tables.generate_zero_bias_all();
            ResolvedTables {
                quant,
                zero_bias,
                distances,
                quality_drives_tables: true,
            }
        }
        QuantTableConfig::Jpegli { .. } | QuantTableConfig::JpegliSharedChroma { .. } => {
            // Per-component distances come from ResolvedQuality (base
            // distance + the family's per-channel chroma scales).
            let distances = rq.distances;
            // 2-table mode uses the Cr base matrix (component 2) for both
            // chroma tables, matching jpeg_set_quality(). XYB always uses
            // per-component matrices (shared-chroma is rejected for XYB
            // by EncoderConfig::validate()).
            let component1 = if table_config.separate_chroma_tables() || use_xyb {
                1
            } else {
                2
            };
            let quant = (
                quant::generate_quant_table_with_distance(
                    distances[0],
                    0,
                    color_space,
                    use_xyb,
                    is_420,
                    allow_16bit,
                ),
                quant::generate_quant_table_with_distance(
                    distances[1],
                    component1,
                    color_space,
                    use_xyb,
                    is_420,
                    allow_16bit,
                ),
                quant::generate_quant_table_with_distance(
                    distances[2],
                    2,
                    color_space,
                    use_xyb,
                    is_420,
                    allow_16bit,
                ),
            );

            // Zero-bias derives from the effective distance of the
            // ACTUALLY-WRITTEN tables (post clamping/rounding), matching
            // C++ jpegli. Two modes:
            //
            // - Uniform distances (neutral scales): the joint
            //   three-component inversion — bit-identical to the
            //   historical / C++-parity output.
            // - Divergent distances: each channel inverts its own table,
            //   so zero-bias follows that channel's effective distance
            //   (per-channel zero-bias, §9.6.5).
            let eff = if rq.uniform {
                let e = quant::quant_vals_to_distance(&quant.0, &quant.1, &quant.2, use_xyb);
                [e; 3]
            } else {
                [
                    quant::quant_table_to_distance_component(&quant.0, 0, use_xyb),
                    quant::quant_table_to_distance_component(&quant.1, 1, use_xyb),
                    quant::quant_table_to_distance_component(&quant.2, 2, use_xyb),
                ]
            };

            // Auto-select zero bias based on color mode
            let zero_bias = if use_xyb {
                (
                    ZeroBiasParams::for_xyb(eff[0], 0),
                    ZeroBiasParams::for_xyb(eff[1], 1),
                    ZeroBiasParams::for_xyb(eff[2], 2),
                )
            } else {
                (
                    ZeroBiasParams::for_ycbcr(eff[0], 0),
                    ZeroBiasParams::for_ycbcr(eff[1], 1),
                    ZeroBiasParams::for_ycbcr(eff[2], 2),
                )
            };

            ResolvedTables {
                quant,
                zero_bias,
                distances,
                quality_drives_tables: true,
            }
        }
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
    /// Table family with its live knobs (the chroma policy lives inside
    /// the variant — see [`QuantTableConfig`]).
    pub table_family: QuantTableConfig,
    /// Whether the config's `Quality` affects the produced tables.
    /// `false` only for `Custom` tables with `ScalingParams::Exact` —
    /// changing `Quality` then changes gates (e.g. `auto_optimize`), not
    /// bytes.
    pub quality_drives_tables: bool,
    /// 3-table (separate Cb/Cr) vs 2-table (shared chroma) layout.
    pub separate_chroma_tables: bool,
    /// Maximum quantization value per component table (after clamping).
    pub quant_max: [u16; 3],
    /// Whether each DQT table needs 16-bit precision.
    pub dqt_16bit: [bool; 3],
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
    /// Tiny-file shared tables STRUCTURALLY guaranteed active
    /// (`Force` on an eligible path). Under `Auto` the decision is an
    /// exact byte-gated trial at emission time and is reported `false`
    /// here — the plan is static and cannot know the trial outcome.
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
        let family = match &self.table_family {
            QuantTableConfig::Jpegli {
                chroma_distance_scales: [a, b],
            } => format!("Jpegli(chroma_scales=[{a:.2}, {b:.2}])"),
            QuantTableConfig::JpegliSharedChroma {
                chroma_distance_scales: [a, b],
            } => format!("JpegliSharedChroma(chroma_scales=[{a:.2}, {b:.2}])"),
            QuantTableConfig::MozjpegRobidoux { chroma_quality } => {
                format!("MozjpegRobidoux(chroma_q={chroma_quality:?})")
            }
            QuantTableConfig::Custom(_) => "Custom".to_string(),
            QuantTableConfig::PiecewiseV4 => "PiecewiseV4".to_string(),
            QuantTableConfig::GlassaLowBpp => "GlassaLowBpp".to_string(),
        };
        writeln!(
            f,
            "  tables: {family} ({}-table) quant_max=[{}, {}, {}]{}{}",
            if self.separate_chroma_tables { 3 } else { 2 },
            self.quant_max[0],
            self.quant_max[1],
            self.quant_max[2],
            if self.dqt_16bit.iter().any(|&b| b) {
                " (16-bit DQT)"
            } else {
                ""
            },
            if self.quality_drives_tables {
                ""
            } else {
                " [quality does NOT drive these tables]"
            },
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

        let resolved = resolve_quant_tables(TableResolveInputs {
            quality: self.quality,
            table_config: &self.quant_table_config,
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

        // Tiny-file participation under Auto is decided by an exact
        // byte-gated trial at emission time (content-dependent), so a
        // static plan reports only the structural cases: Force when
        // possible (non-XYB explicit-Baseline with optimized Huffman).
        // Smallest's sequential candidates trial tiny tables at emission.
        let tiny_possible = !use_xyb
            && matches!(self.scan_mode, ProgressiveScanMode::Baseline)
            && matches!(self.huffman, HuffmanStrategy::Optimize);
        let tiny_file_active = tiny_possible && matches!(self.tiny_file_mode, TinyFileMode::Force);

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
            table_family: self.quant_table_config.clone(),
            quality_drives_tables: resolved.quality_drives_tables,
            separate_chroma_tables: self.quant_table_config.separate_chroma_tables(),
            quant_max,
            dqt_16bit,
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
            .quant_table_config(QuantTableConfig::Jpegli {
                chroma_distance_scales: [2.0, 2.0],
            })
            .resolve_plan(512, 512);
        assert!((plan.distances[1] - plan.distances[0] * 2.0).abs() < 1e-6);
        assert_eq!(plan.distances[1], plan.distances[2]);
    }

    #[test]
    fn xyb_chroma_scale_applies_to_x_and_b_not_y() {
        use crate::encode::encoder_types::XybSubsampling;
        let plan = EncoderConfig::xyb(85, XybSubsampling::BQuarter)
            .quant_table_config(QuantTableConfig::Jpegli {
                chroma_distance_scales: [2.0, 2.0],
            })
            .resolve_plan(512, 512);
        // XYB component order is X, Y, B: the chroma-like X and B carry
        // the scale; Y (the luma-like channel) keeps the base distance.
        assert!((plan.distances[0] - plan.distances[1] * 2.0).abs() < 1e-6);
        assert!((plan.distances[2] - plan.distances[1] * 2.0).abs() < 1e-6);
    }

    #[test]
    fn xyb_rejects_ycbcr_only_table_families() {
        use crate::encode::encoder_types::XybSubsampling;
        let shared = EncoderConfig::xyb(85, XybSubsampling::BQuarter).quant_table_config(
            QuantTableConfig::JpegliSharedChroma {
                chroma_distance_scales: [1.0, 1.0],
            },
        );
        assert!(shared.validate().is_err());

        let moz = EncoderConfig::xyb(85, XybSubsampling::BQuarter).quant_table_config(
            QuantTableConfig::MozjpegRobidoux {
                chroma_quality: None,
            },
        );
        assert!(moz.validate().is_err());

        let ok = EncoderConfig::xyb(85, XybSubsampling::BQuarter);
        assert!(ok.validate().is_ok());
    }

    #[test]
    fn glassa_quality_derives_from_quality_knob() {
        let plan = |q: u8| {
            EncoderConfig::ycbcr(q, ChromaSubsampling::Quarter)
                .quant_table_config(QuantTableConfig::GlassaLowBpp)
                .resolve_plan(64, 64)
        };
        let p10 = plan(10);
        let p20 = plan(20);
        assert!(p10.quality_drives_tables);
        assert_ne!(
            p10.quant_max, p20.quant_max,
            "Glassa anchors must follow the one quality knob"
        );
        // Above the trained range the anchor clamps at 25.
        assert_eq!(plan(40).quant_max, plan(25).quant_max);
    }

    #[test]
    fn exact_custom_tables_report_quality_inert() {
        let tables = crate::encode::tables::robidoux::generate_mozjpeg_default_tables_with_chroma(
            85, None, true,
        );
        let plan = |q: f32| {
            EncoderConfig::ycbcr(q, ChromaSubsampling::Quarter)
                .quant_table_config(QuantTableConfig::Custom(tables.clone()))
                .resolve_plan(64, 64)
        };
        let p = plan(85.0);
        assert!(!p.quality_drives_tables);
        // Same tables at any quality — the plan says so and the maxima agree.
        assert_eq!(p.quant_max, plan(30.0).quant_max);
    }

    #[test]
    fn per_channel_chroma_scales_move_independently() {
        let plan = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
            .quant_table_config(QuantTableConfig::Jpegli {
                chroma_distance_scales: [0.5, 3.0],
            })
            .resolve_plan(512, 512);
        assert!((plan.distances[1] - plan.distances[0] * 0.5).abs() < 1e-6);
        assert!((plan.distances[2] - plan.distances[0] * 3.0).abs() < 1e-6);
        // Independent channels: Cb table finer than uniform, Cr coarser.
        assert_ne!(plan.quant_max[1], plan.quant_max[2]);
    }

    #[test]
    fn per_channel_scales_xyb_x_and_b_independent() {
        use crate::encode::encoder_types::XybSubsampling;
        let plan = EncoderConfig::xyb(85, XybSubsampling::BQuarter)
            .quant_table_config(QuantTableConfig::Jpegli {
                chroma_distance_scales: [0.5, 3.0],
            })
            .resolve_plan(512, 512);
        // [X, Y, B]: X gets scales[0], B gets scales[1], Y stays base.
        assert!((plan.distances[0] - plan.distances[1] * 0.5).abs() < 1e-6);
        assert!((plan.distances[2] - plan.distances[1] * 3.0).abs() < 1e-6);
    }

    #[test]
    fn uniform_scales_use_joint_inversion_for_zero_bias() {
        // The uniform path must stay bit-identical to the legacy joint
        // three-component inversion (C++ parity behaviour).
        let resolved = resolve_quant_tables(TableResolveInputs {
            quality: crate::encode::encoder_types::Quality::ApproxJpegli(75.0),
            table_config: &QuantTableConfig::default(),
            use_xyb: false,
            is_420: true,
            allow_16bit: false,
        });
        let eff = crate::quant::quant_vals_to_distance(
            &resolved.quant.0,
            &resolved.quant.1,
            &resolved.quant.2,
            false,
        );
        let expected = crate::quant::ZeroBiasParams::for_ycbcr(eff, 2);
        assert_eq!(resolved.zero_bias.2.mul, expected.mul);
        assert_eq!(resolved.zero_bias.2.offset, expected.offset);
    }

    #[test]
    fn divergent_scales_give_per_channel_zero_bias() {
        let uniform = resolve_quant_tables(TableResolveInputs {
            quality: crate::encode::encoder_types::Quality::ApproxJpegli(75.0),
            table_config: &QuantTableConfig::default(),
            use_xyb: false,
            is_420: true,
            allow_16bit: false,
        });
        let divergent = resolve_quant_tables(TableResolveInputs {
            quality: crate::encode::encoder_types::Quality::ApproxJpegli(75.0),
            table_config: &QuantTableConfig::Jpegli {
                chroma_distance_scales: [1.0, 3.0],
            },
            use_xyb: false,
            is_420: true,
            allow_16bit: false,
        });
        // Cr's zero-bias now follows Cr's own (coarser) effective distance.
        let eff_cr = crate::quant::quant_table_to_distance_component(&divergent.quant.2, 2, false);
        let expected_cr = crate::quant::ZeroBiasParams::for_ycbcr(eff_cr, 2);
        assert_eq!(divergent.zero_bias.2.mul, expected_cr.mul);
        assert_ne!(
            divergent.zero_bias.2.mul, uniform.zero_bias.2.mul,
            "divergent chroma distance must move that channel's zero-bias"
        );
    }

    #[test]
    fn piecewise_v4_resolves_three_quality_tracked_tables() {
        let plan = |q: f32| {
            EncoderConfig::ycbcr(q, ChromaSubsampling::Quarter)
                .quant_table_config(QuantTableConfig::PiecewiseV4)
                .resolve_plan(512, 512)
        };
        let p50 = plan(50.0);
        assert!(p50.separate_chroma_tables, "piecewise anchors are 3-table");
        assert!(p50.quality_drives_tables);
        assert_ne!(
            p50.quant_max,
            plan(90.0).quant_max,
            "anchors must follow the quality knob"
        );
        // Lower quality must not produce finer tables than higher quality.
        assert!(p50.quant_max[0] >= plan(90.0).quant_max[0]);
    }

    #[test]
    fn piecewise_v4_rejected_for_xyb() {
        use crate::encode::encoder_types::XybSubsampling;
        let cfg = EncoderConfig::xyb(85, XybSubsampling::BQuarter)
            .quant_table_config(QuantTableConfig::PiecewiseV4);
        assert!(cfg.validate().is_err());
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
