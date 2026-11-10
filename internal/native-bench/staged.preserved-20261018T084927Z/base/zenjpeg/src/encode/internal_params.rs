//! Internal-params bundle for cross-codec uniformity (`__expert` feature).
//!
//! [`InternalParams`] collects the encoder knobs that codec-calibration
//! sweeps and the picker training pipeline want to drive externally,
//! mirroring `zenwebp::InternalParams` so a single picker model can
//! emit the same bundle shape for every codec in the zen family.
//!
//! Production callers should use [`EncoderConfig::ycbcr`] / [`xyb`] /
//! [`grayscale`] + [`OptimizationPreset`] and let zenjpeg pick
//! reasonable defaults for the rest. Reach for [`InternalParams`] when
//! you specifically need to vary calibration axes from outside the
//! codec — e.g., from a Pareto sweep harness or a learned picker that
//! emits per-image axis values.
//!
//! Each field is `Option<_>`. `None` means "leave the
//! [`EncoderConfig`]'s existing value alone." This is partial-merge,
//! same shape zenwebp uses, so callers can override one axis at a time
//! without spelling out the rest.
//!
//! [`EncoderConfig::ycbcr`]: super::encoder_config::EncoderConfig::ycbcr
//! [`xyb`]: super::encoder_config::EncoderConfig::xyb
//! [`grayscale`]: super::encoder_config::EncoderConfig::grayscale
//! [`OptimizationPreset`]: super::encoder_types::OptimizationPreset
//! [`EncoderConfig`]: super::encoder_config::EncoderConfig

#![cfg(feature = "__expert")]

use super::encoder_config::EncoderConfig;
#[cfg(feature = "boundary-rd")]
use super::encoder_config::{BoundaryRd, BoundaryRdConfig};
use super::encoder_types::{
    ChromaSubsampling, DownsamplingMethod, HuffmanStrategy, OptimizationPreset,
    ProgressiveScanMode, Quality, QuantTableConfig, QuantTableSource, ScanStrategy, TinyFileMode,
    XybSubsampling,
};
use super::trellis::TrellisConfig;

/// Color-path selection for [`InternalParams::color_path`].
///
/// Mirrors [`EncoderConfig`]'s constructors (`ycbcr` / `xyb` /
/// `grayscale` / `rgb`) as a single tagged enum so picker integrations
/// can emit one uniform field for codec color routing.
/// `#[non_exhaustive]` because additional color paths (e.g., a future
/// high-bit-depth route) may land here without bumping the major
/// version.
///
/// [`EncoderConfig`]: super::encoder_config::EncoderConfig
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ColorPath {
    /// Standard JPEG YCbCr with the supplied chroma subsampling.
    Ycbcr {
        /// Chroma subsampling mode.
        subsampling: ChromaSubsampling,
    },
    /// XYB perceptual color space with the supplied B-channel
    /// subsampling.
    Xyb {
        /// B-channel subsampling.
        b_subsampling: XybSubsampling,
    },
    /// Single-channel grayscale.
    Grayscale,
    /// RGB passthrough — channels stored without color transformation
    /// (issue #185). Always 4:4:4.
    Rgb,
}

/// Bundle of advanced encoder tuning knobs. Expert-only.
///
/// Intended for codec calibration sweeps and the picker training
/// pipeline. Production callers should rely on
/// [`EncoderConfig::ycbcr`] / [`xyb`] / [`grayscale`] +
/// [`OptimizationPreset`] and the per-axis builder methods on
/// [`EncoderConfig`].
///
/// Every field is `Option<_>`. `None` means "leave the
/// [`EncoderConfig`]'s existing value alone." Apply with
/// [`EncoderConfig::with_internal_params`].
///
/// `#[non_exhaustive]` so adding a new axis is a non-breaking change.
///
/// ```ignore
/// # #[cfg(feature = "__expert")]
/// # {
/// use zenjpeg::encoder::{EncoderConfig, ChromaSubsampling};
/// use zenjpeg::encode::internal_params::InternalParams;
///
/// let cfg = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
///     .with_internal_params(InternalParams {
///         optimize_huffman: Some(true),
///         pre_blur: Some(0.4),
///         ..Default::default()
///     });
/// # }
/// ```
///
/// [`EncoderConfig`]: super::encoder_config::EncoderConfig
/// [`EncoderConfig::ycbcr`]: super::encoder_config::EncoderConfig::ycbcr
/// [`xyb`]: super::encoder_config::EncoderConfig::xyb
/// [`grayscale`]: super::encoder_config::EncoderConfig::grayscale
/// [`EncoderConfig::with_internal_params`]: super::encoder_config::EncoderConfig::with_internal_params
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub struct InternalParams {
    /// Override the encoding quality (any [`Quality`] variant).
    pub quality: Option<Quality>,

    /// Color path: YCbCr / XYB / Grayscale, with the relevant
    /// subsampling. Maps to the three [`EncoderConfig`] constructors.
    ///
    /// Applied via [`EncoderConfig::color_mode`] (no constructor swap),
    /// so other axes set on the underlying config are preserved.
    ///
    /// [`EncoderConfig`]: super::encoder_config::EncoderConfig
    /// [`EncoderConfig::color_mode`]: super::encoder_config::EncoderConfig::color_mode
    pub color_path: Option<ColorPath>,

    /// Progressive / baseline scan mode (preferred over
    /// [`Self::scan_strategy`] / [`Self::optimize_scans`]).
    pub progressive: Option<ProgressiveScanMode>,

    /// Legacy `ScanStrategy` axis, applied before [`Self::progressive`]
    /// so the latter wins when both are set. Kept for picker
    /// configurations that already train against the strategy axis.
    pub scan_strategy: Option<ScanStrategy>,

    /// `true` switches to [`ProgressiveScanMode::ProgressiveSearch`]
    /// (search ~2 % smaller). `false` is a no-op (legacy behaviour
    /// of `optimize_scans(false)`).
    pub optimize_scans: Option<bool>,

    /// Quantization table configuration (preferred — bundles source
    /// + chroma layout + custom tables).
    pub quant_table_config: Option<QuantTableConfig>,

    /// Convenience axis: maps `Jpegli` → keep current jpegli config,
    /// `MozjpegDefault` → `MozjpegRobidoux`. Applied **after**
    /// [`Self::quant_table_config`], so the latter wins when both are
    /// set.
    pub quant_source: Option<QuantTableSource>,

    /// Apply an optimization preset (sets scan mode, quant tables,
    /// trellis mode, deringing, AQ in one shot).
    ///
    /// Applied **first**, before all other field-level overrides, so
    /// per-axis fields can fine-tune the preset's defaults.
    pub optimization: Option<OptimizationPreset>,

    /// Convenience: `true` → [`HuffmanStrategy::Optimize`], `false` →
    /// fixed Annex K tables. Applied before [`Self::huffman`] so the
    /// latter wins when both are set.
    pub optimize_huffman: Option<bool>,

    /// Huffman table strategy (preferred over
    /// [`Self::optimize_huffman`]).
    pub huffman: Option<HuffmanStrategy>,

    /// Allow 16-bit DQT entries (extended JPEG / SOF1 when needed).
    pub allow_16bit_quant_tables: Option<bool>,

    /// Boundary-continuity refinement (`#91` / PR `#102`). Requires
    /// the `boundary-rd` cargo feature; without it this field has no
    /// effect.
    #[cfg(feature = "boundary-rd")]
    pub boundary_rd: Option<BoundaryRdConfig>,

    /// Trellis quantization configuration. Set
    /// [`TrellisConfig::aq_coupling`] for AQ-coupled (hybrid) lambda.
    pub trellis: Option<TrellisConfig>,

    /// Chroma distance scale (jpegli table families only; clamped to
    /// `[0.1, 5.0]`). Applied onto [`Self::quant_table_config`]'s variant —
    /// no effect when the active family is mozjpeg/custom/Glassa.
    pub chroma_distance_scale: Option<f32>,

    /// Independent chroma quality ([`QuantTableConfig::MozjpegRobidoux`]
    /// only). `Some(Some(q))` sets `q`; `Some(None)` clears the override
    /// (revert to using luma quality for chroma); `None` leaves the
    /// existing value alone. No effect on other table families.
    pub chroma_quality: Option<Option<u8>>,

    /// Enable [`EncoderConfig::auto_optimize`].
    ///
    /// [`EncoderConfig::auto_optimize`]: super::encoder_config::EncoderConfig::auto_optimize
    pub auto_optimize: Option<bool>,

    /// Toggle overshoot deringing.
    pub deringing: Option<bool>,

    /// Toggle adaptive quantization (jpegli AQ).
    pub aq_enabled: Option<bool>,

    /// Pre-encode Gaussian blur sigma (0.0 = disabled).
    pub pre_blur: Option<f32>,

    /// Tiny-file optimization mode.
    pub tiny_file_mode: Option<TinyFileMode>,

    /// Restart marker interval in MCU rows (0 = disabled).
    pub restart_mcu_rows: Option<u16>,

    /// Force RST markers in progressive mode (default off — they are
    /// useless in progressive and cost ~10 % overhead).
    pub force_restart_markers: Option<bool>,

    /// Chroma downsampling method (only honoured for RGB/RGBX input
    /// with chroma subsampling).
    pub downsampling_method: Option<DownsamplingMethod>,
}

impl EncoderConfig {
    /// Apply a bundle of expert encoder knobs at once. Expert-only.
    ///
    /// [`InternalParams`] collects the calibration axes that codec
    /// consumers don't usually touch. Each field is `Option<_>` so
    /// callers override only the axes they care about; `None` keeps
    /// the existing (default) value.
    ///
    /// This is the recommended entry point for the picker training
    /// pipeline and codec-calibration sweeps — pass an
    /// [`InternalParams`] produced by the picker runtime instead of
    /// chaining individual `with_*` setters per axis. The bundle
    /// shape mirrors zenwebp's `InternalParams`, so a single
    /// cross-codec picker model can drive every zen codec the same
    /// way.
    ///
    /// **Field application order** (later fields override earlier ones
    /// when their axes overlap):
    /// 1. `optimization` (preset — sets many fields at once)
    /// 2. `quality`
    /// 3. `color_path`
    /// 4. `quant_source` then `quant_table_config`
    /// 5. `scan_strategy` / `optimize_scans` then `progressive`
    /// 6. `optimize_huffman` then `huffman`
    /// 7. `allow_16bit_quant_tables`
    /// 8. `auto_optimize`
    /// 9. `trellis`
    /// 10. `boundary_rd` (boundary-rd feature)
    /// 11. `chroma_distance_scale`, `chroma_quality`
    /// 12. `deringing`, `aq_enabled`, `pre_blur`, `tiny_file_mode`,
    ///     `restart_mcu_rows`, `force_restart_markers`,
    ///     `downsampling_method`
    #[must_use]
    pub fn with_internal_params(mut self, params: InternalParams) -> Self {
        if let Some(opt) = params.optimization {
            self = self.optimization(opt);
        }
        if let Some(q) = params.quality {
            self = self.quality(q);
        }
        if let Some(cp) = params.color_path {
            self = match cp {
                ColorPath::Ycbcr { subsampling } => {
                    self.color_mode(super::encoder_types::ColorMode::YCbCr { subsampling })
                }
                ColorPath::Xyb { b_subsampling } => {
                    self.color_mode(super::encoder_types::ColorMode::Xyb {
                        subsampling: b_subsampling,
                    })
                }
                ColorPath::Grayscale => self.color_mode(super::encoder_types::ColorMode::Grayscale),
                ColorPath::Rgb => self.color_mode(super::encoder_types::ColorMode::Rgb),
            };
        }
        if let Some(qs) = params.quant_source {
            self = self.quant_source(qs);
        }
        if let Some(qtc) = params.quant_table_config {
            self = self.quant_table_config(qtc);
        }
        if let Some(strategy) = params.scan_strategy {
            self = self.scan_strategy(strategy);
        }
        if let Some(enable) = params.optimize_scans {
            self = self.optimize_scans(enable);
        }
        if let Some(mode) = params.progressive {
            self = self.progressive(mode);
        }
        if let Some(opt) = params.optimize_huffman {
            self = self.optimize_huffman(opt);
        }
        if let Some(strategy) = params.huffman {
            self = self.huffman(strategy);
        }
        if let Some(b) = params.allow_16bit_quant_tables {
            self = self.allow_16bit_quant_tables(b);
        }
        if let Some(b) = params.auto_optimize {
            self = self.auto_optimize(b);
        }
        if let Some(t) = params.trellis {
            self = self.trellis(t);
        }
        #[cfg(feature = "boundary-rd")]
        if let Some(brd) = params.boundary_rd {
            self = self.boundary_rd(BoundaryRd::On(brd));
        }
        if let Some(scale) = params.chroma_distance_scale {
            // The knob lives on the jpegli table families; for other
            // families the axis has no meaning (set `quant_table_config`
            // first — it is applied before this field).
            // The scalar cross-codec axis applies uniformly to both
            // chroma-like channels; per-channel control goes through the
            // quant_table_config axis directly.
            let s = scale.clamp(0.1, 5.0);
            self.quant_table_config = match self.quant_table_config {
                QuantTableConfig::Jpegli { .. } => QuantTableConfig::Jpegli {
                    chroma_distance_scales: [s, s],
                },
                QuantTableConfig::JpegliSharedChroma { .. } => {
                    QuantTableConfig::JpegliSharedChroma {
                        chroma_distance_scales: [s, s],
                    }
                }
                other => other,
            };
        }
        if let Some(cq) = params.chroma_quality {
            // Lives on the MozjpegRobidoux family only.
            self.quant_table_config = match self.quant_table_config {
                QuantTableConfig::MozjpegRobidoux { .. } => QuantTableConfig::MozjpegRobidoux {
                    chroma_quality: cq.map(|q| q.clamp(1, 100)),
                },
                other => other,
            };
        }
        if let Some(b) = params.deringing {
            self = self.deringing(b);
        }
        if let Some(b) = params.aq_enabled {
            self = self.aq_enabled(b);
        }
        if let Some(sigma) = params.pre_blur {
            self = self.pre_blur(sigma);
        }
        if let Some(mode) = params.tiny_file_mode {
            self = self.tiny_file_mode(mode);
        }
        if let Some(rows) = params.restart_mcu_rows {
            self = self.restart_mcu_rows(rows);
        }
        if let Some(b) = params.force_restart_markers {
            self = self.force_restart_markers(b);
        }
        if let Some(method) = params.downsampling_method {
            self = self.downsampling_method(method);
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::encoder_types::{ChromaSubsampling, ColorMode};

    fn baseline() -> EncoderConfig {
        EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
    }

    /// Empty `InternalParams` (all `None`) leaves the config bytewise
    /// equivalent to the constructor default — debug-format equality
    /// is a coarse but reliable check that no field flipped.
    #[test]
    fn default_internal_params_is_noop() {
        let cfg = baseline();
        let cfg2 = baseline().with_internal_params(InternalParams::default());
        assert_eq!(format!("{cfg:?}"), format!("{cfg2:?}"));
    }

    #[test]
    fn quality_field_applies() {
        let cfg = baseline().with_internal_params(InternalParams {
            quality: Some(Quality::ApproxJpegli(50.0)),
            ..Default::default()
        });
        let dist50 = cfg.get_quality().to_distance();
        let dist85 = baseline().get_quality().to_distance();
        assert!(
            (dist50 - dist85).abs() > 0.1,
            "quality override should change effective distance: {dist50} vs {dist85}"
        );
    }

    #[test]
    fn progressive_field_applies() {
        let cfg = baseline().with_internal_params(InternalParams {
            progressive: Some(ProgressiveScanMode::Baseline),
            ..Default::default()
        });
        assert!(!cfg.is_progressive());
    }

    #[test]
    fn color_path_grayscale_switches_color_mode() {
        let cfg = baseline().with_internal_params(InternalParams {
            color_path: Some(ColorPath::Grayscale),
            ..Default::default()
        });
        assert!(matches!(cfg.get_color_mode(), ColorMode::Grayscale));
    }

    #[test]
    fn color_path_xyb_switches_color_mode() {
        let cfg = baseline().with_internal_params(InternalParams {
            color_path: Some(ColorPath::Xyb {
                b_subsampling: XybSubsampling::BQuarter,
            }),
            ..Default::default()
        });
        assert!(matches!(cfg.get_color_mode(), ColorMode::Xyb { .. }));
    }

    #[test]
    fn deringing_and_aq_toggle() {
        let cfg = baseline().with_internal_params(InternalParams {
            deringing: Some(false),
            aq_enabled: Some(false),
            ..Default::default()
        });
        // pub(crate) field access is fine inside the crate-internal test module.
        assert!(!cfg.deringing);
        assert!(!cfg.is_aq_enabled());
    }

    #[test]
    fn pre_blur_field_applies() {
        let cfg = baseline().with_internal_params(InternalParams {
            pre_blur: Some(0.4),
            ..Default::default()
        });
        assert!((cfg.pre_blur - 0.4).abs() < 1e-6);
    }

    #[test]
    fn restart_mcu_rows_applies() {
        let cfg = baseline().with_internal_params(InternalParams {
            restart_mcu_rows: Some(8),
            ..Default::default()
        });
        assert_eq!(cfg.restart_mcu_rows, 8);
    }

    #[test]
    fn allow_16bit_quant_tables_applies() {
        let cfg = baseline().with_internal_params(InternalParams {
            allow_16bit_quant_tables: Some(true),
            ..Default::default()
        });
        assert!(cfg.is_allow_16bit_quant_tables());
    }

    #[test]
    fn chroma_distance_scale_applies_on_jpegli_family() {
        let cfg = baseline().with_internal_params(InternalParams {
            chroma_distance_scale: Some(2.0),
            ..Default::default()
        });
        assert_eq!(
            cfg.get_quant_table_config().chroma_distance_scales(),
            Some([2.0, 2.0])
        );
    }

    #[test]
    fn chroma_distance_scale_inert_on_mozjpeg_family() {
        let cfg = baseline().with_internal_params(InternalParams {
            quant_table_config: Some(QuantTableConfig::MozjpegRobidoux {
                chroma_quality: None,
            }),
            chroma_distance_scale: Some(2.0),
            ..Default::default()
        });
        // The axis has no meaning for this family; the bundle's
        // quant_table_config wins.
        assert_eq!(cfg.get_quant_table_config().chroma_distance_scales(), None);
    }

    #[test]
    fn chroma_quality_applies_on_mozjpeg_family() {
        let cfg = baseline().with_internal_params(InternalParams {
            quant_table_config: Some(QuantTableConfig::MozjpegRobidoux {
                chroma_quality: None,
            }),
            chroma_quality: Some(Some(70)),
            ..Default::default()
        });
        assert_eq!(cfg.get_quant_table_config().chroma_quality(), Some(70));
    }

    #[test]
    fn chroma_quality_inert_on_jpegli_family() {
        let cfg = baseline().with_internal_params(InternalParams {
            chroma_quality: Some(Some(70)),
            ..Default::default()
        });
        assert_eq!(cfg.get_quant_table_config().chroma_quality(), None);
    }

    #[test]
    fn chroma_quality_some_none_clears_override() {
        let with_value = baseline().with_internal_params(InternalParams {
            quant_table_config: Some(QuantTableConfig::MozjpegRobidoux {
                chroma_quality: None,
            }),
            chroma_quality: Some(Some(70)),
            ..Default::default()
        });
        let cleared = with_value.with_internal_params(InternalParams {
            chroma_quality: Some(None),
            ..Default::default()
        });
        assert_eq!(cleared.get_quant_table_config().chroma_quality(), None);
    }

    #[test]
    fn force_restart_markers_applies() {
        let cfg = baseline().with_internal_params(InternalParams {
            force_restart_markers: Some(true),
            ..Default::default()
        });
        assert!(cfg.force_restart_markers);
    }

    #[test]
    fn downsampling_method_applies() {
        let cfg = baseline().with_internal_params(InternalParams {
            downsampling_method: Some(DownsamplingMethod::GammaAware),
            ..Default::default()
        });
        assert!(matches!(
            cfg.downsampling_method,
            DownsamplingMethod::GammaAware
        ));
    }

    #[test]
    fn quant_table_config_applies() {
        let cfg = baseline().with_internal_params(InternalParams {
            quant_table_config: Some(QuantTableConfig::JpegliSharedChroma {
                chroma_distance_scales: [1.0, 1.0],
            }),
            ..Default::default()
        });
        assert!(matches!(
            cfg.get_quant_table_config(),
            QuantTableConfig::JpegliSharedChroma { .. }
        ));
    }

    #[test]
    fn huffman_strategy_applies() {
        let cfg = baseline().with_internal_params(InternalParams {
            huffman: Some(HuffmanStrategy::FixedAnnexK),
            ..Default::default()
        });
        // is_optimize_huffman is only true for HuffmanStrategy::Optimize
        assert!(!cfg.is_optimize_huffman());
    }

    #[test]
    fn tiny_file_mode_applies() {
        let cfg = baseline().with_internal_params(InternalParams {
            tiny_file_mode: Some(TinyFileMode::Off),
            ..Default::default()
        });
        assert!(matches!(cfg.get_tiny_file_mode(), TinyFileMode::Off));
    }
    #[test]
    fn auto_optimize_applies() {
        // auto_optimize(true) only takes effect within the q≥50 band
        // for YCbCr; baseline q=85 is in-band, so it should turn on
        // hybrid trellis.
        let cfg = baseline().with_internal_params(InternalParams {
            auto_optimize: Some(true),
            ..Default::default()
        });
        assert!(cfg.trellis.is_some());
    }
    #[test]
    fn trellis_field_applies() {
        let cfg = baseline().with_internal_params(InternalParams {
            trellis: Some(TrellisConfig::default()),
            ..Default::default()
        });
        assert!(cfg.trellis.is_some());
    }

    /// Per-field permutation: every non-default field should produce
    /// observable state changes vs the baseline. This is the single
    /// "everything together" test the brief requires.
    #[test]
    fn full_permutation_round_trip() {
        let mut params = InternalParams {
            quality: Some(Quality::ApproxJpegli(70.0)),
            color_path: Some(ColorPath::Ycbcr {
                subsampling: ChromaSubsampling::None,
            }),
            progressive: Some(ProgressiveScanMode::Baseline),
            scan_strategy: None,
            optimize_scans: None,
            quant_table_config: Some(QuantTableConfig::JpegliSharedChroma {
                chroma_distance_scales: [1.0, 1.0],
            }),
            quant_source: None,
            optimization: None,
            optimize_huffman: Some(true),
            huffman: None,
            allow_16bit_quant_tables: Some(true),
            chroma_distance_scale: Some(1.5),
            chroma_quality: Some(Some(60)),
            deringing: Some(false),
            aq_enabled: Some(false),
            pre_blur: Some(0.3),
            tiny_file_mode: Some(TinyFileMode::Off),
            restart_mcu_rows: Some(2),
            force_restart_markers: Some(true),
            downsampling_method: Some(DownsamplingMethod::GammaAware),
            ..Default::default()
        };
        params.auto_optimize = Some(false);
        // suppress unused-mut when all conditional cfgs are off.
        let _ = &mut params;

        let cfg = baseline().with_internal_params(params);

        // Spot-check a representative subset.
        assert!(matches!(
            cfg.get_color_mode(),
            ColorMode::YCbCr {
                subsampling: ChromaSubsampling::None
            }
        ));
        assert!(!cfg.is_progressive());
        assert!(cfg.is_allow_16bit_quant_tables());
        // chroma_distance_scale lands on the SharedChroma family set above;
        // chroma_quality is inert there (mozjpeg-only knob).
        assert_eq!(
            cfg.get_quant_table_config().chroma_distance_scales(),
            Some([1.5, 1.5])
        );
        assert_eq!(cfg.get_quant_table_config().chroma_quality(), None);
        assert!(!cfg.deringing);
        assert!(!cfg.is_aq_enabled());
        assert!((cfg.pre_blur - 0.3).abs() < 1e-6);
        assert_eq!(cfg.restart_mcu_rows, 2);
        assert!(cfg.force_restart_markers);
    }
}
