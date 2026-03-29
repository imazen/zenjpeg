//! Encoder configuration for v2 API.

use super::byte_encoders::{BytesEncoder, RgbEncoder, YCbCrPlanarEncoder};
use super::encoder_types::{
    ChromaSubsampling, ColorMode, DownsamplingMethod, HuffmanStrategy, PixelLayout,
    ProgressiveScanMode, Quality, QuantTableConfig, QuantTableSource, ScanStrategy, XybSubsampling,
};
#[cfg(feature = "trellis")]
use super::trellis::TrellisConfig;
use crate::error::Result;
use crate::types::EdgePaddingConfig;

/// JPEG encoder configuration. Dimension-independent, reusable across images.
#[derive(Clone, Debug)]
pub struct EncoderConfig {
    pub(crate) quality: Quality,
    /// Quantization table configuration (source, chroma layout, custom tables).
    /// Replaces the old `tables` + `separate_chroma_tables` + `quant_source` triple.
    pub(crate) quant_table_config: QuantTableConfig,
    /// Scan mode (baseline vs progressive, with script strategy).
    /// Replaces the old `progressive` + `scan_strategy` pair.
    pub(crate) scan_mode: ProgressiveScanMode,
    pub(crate) huffman: HuffmanStrategy,
    pub(crate) color_mode: ColorMode,
    pub(crate) downsampling_method: DownsamplingMethod,
    /// Restart marker interval in MCU rows (0 = disabled, default = 4).
    ///
    /// An MCU row is one row of Minimum Coded Units: 8 pixels tall for 4:4:4,
    /// 16 pixels tall for 4:2:0/4:2:2. Resolved to an exact MCU count at
    /// encode time when image dimensions are known.
    ///
    /// Restart markers enable parallel decoding and error recovery with
    /// negligible compression overhead when row-aligned (+0.04% at 4 rows).
    pub(crate) restart_mcu_rows: u16,
    pub(crate) edge_padding: EdgePaddingConfig,
    /// Parallel encoding configuration (requires `parallel` feature)
    #[cfg(feature = "parallel")]
    pub(crate) parallel: Option<super::encoder_types::ParallelEncoding>,
    /// Hybrid quantization configuration (requires `trellis` feature).
    #[cfg(feature = "trellis")]
    pub(crate) hybrid_config: super::trellis::HybridConfig,
    /// Enable overshoot deringing (on by default).
    pub(crate) deringing: bool,
    /// Enable adaptive quantization (jpegli AQ). On by default.
    /// When disabled, AQ computation is skipped entirely and all blocks
    /// receive neutral AQ strength (0.0).
    pub(crate) aq_enabled: bool,
    /// Allow 16-bit quantization tables (extended JPEG, SOF1).
    /// When false, quant values are clamped to 255 for baseline compatibility.
    pub(crate) allow_16bit_quant_tables: bool,
    /// Trellis quantization configuration (mozjpeg-compatible API).
    /// When Some, enables trellis quantization for rate-distortion optimization.
    #[cfg(feature = "trellis")]
    pub(crate) trellis: Option<TrellisConfig>,
    /// Prepared segments for injection (EXIF, XMP, ICC, etc.) and MPF secondary images.
    pub(crate) segments: Option<super::extras::EncoderSegments>,
    /// Gaussian blur sigma applied before encoding (0.0 = disabled).
    ///
    /// A mild blur (σ=0.4) before JPEG encoding reduces file size ~5% with
    /// negligible perceptual quality loss. Only applies to u8 RGB/RGBA input.
    pub(crate) pre_blur: f32,
}

// Note: No Default impl - quality and color mode are required via constructors

impl EncoderConfig {
    /// Create a YCbCr encoder configuration.
    ///
    /// YCbCr is the standard JPEG color space, compatible with all decoders.
    ///
    /// # Arguments
    /// - `quality`: Quality level (0-100 for jpegli scale, or use `Quality::*` variants)
    /// - `subsampling`: Chroma subsampling mode
    ///   - `ChromaSubsampling::None` (4:4:4) - best quality, larger files
    ///   - `ChromaSubsampling::Quarter` (4:2:0) - good compression, smaller files
    ///   - `ChromaSubsampling::HalfHorizontal` (4:2:2) - horizontal only
    ///   - `ChromaSubsampling::HalfVertical` (4:4:0) - vertical only
    ///
    /// # Example
    /// ```ignore
    /// use zenjpeg::encoder::{EncoderConfig, ChromaSubsampling};
    ///
    /// let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
    ///     .progressive(true);
    /// ```
    #[must_use]
    pub fn ycbcr(quality: impl Into<Quality>, subsampling: ChromaSubsampling) -> Self {
        Self {
            quality: quality.into(),
            color_mode: ColorMode::YCbCr { subsampling },
            ..Self::default_internal()
        }
    }

    /// Create an XYB encoder configuration.
    ///
    /// XYB is a perceptual color space that can achieve better quality at the same
    /// file size for some images. The B (blue-yellow) channel can optionally be
    /// subsampled since it's less perceptually important.
    ///
    /// # Arguments
    /// - `quality`: Quality level (0-100 for jpegli scale, or use `Quality::*` variants)
    /// - `b_subsampling`: B channel subsampling
    ///   - `XybSubsampling::Full` - all channels at full resolution
    ///   - `XybSubsampling::BQuarter` - B channel at quarter resolution (default, recommended)
    ///
    /// # Notes
    /// - Requires linear RGB input (f32 or u16 pixel formats)
    /// - Embeds an ICC profile for proper color reproduction
    /// - Not all decoders support XYB JPEGs correctly
    ///
    /// # Example
    /// ```ignore
    /// use zenjpeg::encoder::{EncoderConfig, XybSubsampling};
    ///
    /// let config = EncoderConfig::xyb(85, XybSubsampling::BQuarter)
    ///     .progressive(true);
    /// ```
    #[must_use]
    pub fn xyb(quality: impl Into<Quality>, b_subsampling: XybSubsampling) -> Self {
        Self {
            quality: quality.into(),
            color_mode: ColorMode::Xyb {
                subsampling: b_subsampling,
            },
            // XYB doesn't need 16-bit quant tables (values >255 quantize to zero
            // anyway). SOF1 is forced separately via force_sof1 for DC categories.
            allow_16bit_quant_tables: false,
            ..Self::default_internal()
        }
    }

    /// Create a grayscale encoder configuration.
    ///
    /// Only the luminance channel is encoded. Works with any input format;
    /// color inputs are converted to grayscale.
    ///
    /// # Arguments
    /// - `quality`: Quality level (0-100 for jpegli scale, or use `Quality::*` variants)
    ///
    /// # Example
    /// ```ignore
    /// use zenjpeg::encoder::EncoderConfig;
    ///
    /// let config = EncoderConfig::grayscale(85)
    ///     .progressive(true);
    /// ```
    #[must_use]
    pub fn grayscale(quality: impl Into<Quality>) -> Self {
        Self {
            quality: quality.into(),
            color_mode: ColorMode::Grayscale,
            ..Self::default_internal()
        }
    }

    /// Create a YCbCr encoder with effort-based defaults.
    ///
    /// Combines quality, subsampling, and an [`Effort`] level into a single call.
    /// The effort level maps to an [`OptimizationPreset`] that configures
    /// progressive mode, trellis, AQ, scan strategy, and deringing.
    ///
    #[doc(hidden)]
    #[must_use]
    pub fn ycbcr_effort(
        quality: impl Into<Quality>,
        subsampling: ChromaSubsampling,
        effort: super::encoder_types::Effort,
    ) -> Self {
        Self::ycbcr(quality, subsampling).optimization(effort.to_preset())
    }

    #[doc(hidden)]
    #[must_use]
    pub fn xyb_effort(
        quality: impl Into<Quality>,
        b_subsampling: XybSubsampling,
        effort: super::encoder_types::Effort,
    ) -> Self {
        Self::xyb(quality, b_subsampling).optimization(effort.to_preset())
    }

    #[doc(hidden)]
    #[must_use]
    pub fn grayscale_effort(
        quality: impl Into<Quality>,
        effort: super::encoder_types::Effort,
    ) -> Self {
        Self::grayscale(quality).optimization(effort.to_preset())
    }

    /// Internal default for non-required fields only.
    fn default_internal() -> Self {
        Self {
            quality: Quality::default(),
            quant_table_config: QuantTableConfig::default(), // Jpegli, 3 tables
            scan_mode: ProgressiveScanMode::Progressive,     // Progressive gives 3-7% smaller
            huffman: HuffmanStrategy::Optimize,
            color_mode: ColorMode::default(),
            downsampling_method: DownsamplingMethod::default(),
            restart_mcu_rows: 4,
            edge_padding: EdgePaddingConfig::default(),
            #[cfg(feature = "parallel")]
            parallel: None,
            #[cfg(feature = "trellis")]
            hybrid_config: super::trellis::HybridConfig::disabled(),
            deringing: true,
            aq_enabled: true,
            allow_16bit_quant_tables: false,
            #[cfg(feature = "trellis")]
            trellis: None,
            segments: None,
            pre_blur: 0.0,
        }
    }

    // === Quality & Quantization ===

    /// Override the quality level.
    ///
    /// Accepts any type that converts to `Quality`:
    /// - `f32` or `u8` for ApproxJpegli scale
    /// - `Quality::ApproxMozjpeg(u8)` for mozjpeg-like quality
    /// - `Quality::ApproxSsim2(f32)` for SSIMULACRA2 target
    /// - `Quality::ApproxButteraugli(f32)` for Butteraugli target
    #[must_use]
    pub fn quality(mut self, q: impl Into<Quality>) -> Self {
        self.quality = q.into();
        self
    }

    // === Encoding Mode ===

    /// Set the scan mode (baseline vs progressive, with script strategy).
    ///
    /// This is the preferred way to configure progressive encoding.
    /// It bundles the progressive flag and scan script strategy into a
    /// single type-safe enum, preventing invalid combinations.
    ///
    /// Progressive modes automatically enable optimized Huffman tables.
    ///
    /// # Example
    ///
    /// Set the progressive/baseline scan mode.
    ///
    /// Accepts `bool`, `ProgressiveScanMode`, or any type that converts to it.
    ///
    /// | Input | Result |
    /// |-------|--------|
    /// | `true` | `Progressive` (jpegli default script) |
    /// | `false` | `Baseline` (sequential JPEG) |
    /// | `ProgressiveScanMode::Progressive` | jpegli progressive script |
    /// | `ProgressiveScanMode::ProgressiveMozjpeg` | mozjpeg progressive script |
    /// | `ProgressiveScanMode::ProgressiveSearch` | search 64 candidates (~2% smaller) |
    ///
    /// All progressive modes automatically enable `HuffmanStrategy::Optimize`.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use zenjpeg::encoder::{EncoderConfig, ProgressiveScanMode, ChromaSubsampling};
    ///
    /// // Boolean
    /// let config = EncoderConfig::ycbcr(85, sub).progressive(true);
    ///
    /// // Explicit mode for best compression
    /// let config = EncoderConfig::ycbcr(85, sub)
    ///     .progressive(ProgressiveScanMode::ProgressiveSearch);
    /// ```
    #[must_use]
    pub fn progressive(mut self, mode: impl Into<ProgressiveScanMode>) -> Self {
        self.scan_mode = mode.into();
        if self.scan_mode.is_progressive() {
            self.huffman = HuffmanStrategy::Optimize;
        }
        self
    }

    #[doc(hidden)]
    #[must_use]
    pub fn scan_mode(self, mode: ProgressiveScanMode) -> Self {
        self.progressive(mode)
    }

    #[doc(hidden)]
    #[must_use]
    pub fn scan_strategy(self, strategy: ScanStrategy) -> Self {
        self.progressive(match strategy {
            ScanStrategy::Default => ProgressiveScanMode::Progressive,
            ScanStrategy::Search => ProgressiveScanMode::ProgressiveSearch,
            ScanStrategy::Mozjpeg => ProgressiveScanMode::ProgressiveMozjpeg,
        })
    }

    #[doc(hidden)]
    #[must_use]
    pub fn optimize_scans(self, enable: bool) -> Self {
        if enable {
            self.progressive(ProgressiveScanMode::ProgressiveSearch)
        } else {
            self
        }
    }

    /// Set the quantization table configuration.
    ///
    /// This is the preferred way to configure quantization tables.
    /// It bundles table source, chroma layout, and custom tables into
    /// a single type-safe enum, preventing invalid combinations.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use zenjpeg::encode::{EncoderConfig, ChromaSubsampling, QuantTableConfig};
    ///
    /// let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
    ///     .quant_table_config(QuantTableConfig::MozjpegRobidoux);
    /// ```
    #[must_use]
    pub fn quant_table_config(mut self, config: QuantTableConfig) -> Self {
        self.quant_table_config = config;
        self
    }

    /// Set the quantization table source.
    ///
    /// Convenience method. Prefer [`quant_table_config()`](Self::quant_table_config)
    /// for full control — it bundles table source, chroma layout, and custom
    /// tables into one type-safe enum.
    ///
    /// - `QuantTableSource::Jpegli` → preserves current chroma table layout
    /// - `QuantTableSource::MozjpegDefault` → sets `MozjpegRobidoux` (always 2 tables)
    #[must_use]
    pub fn quant_source(mut self, source: QuantTableSource) -> Self {
        match source {
            QuantTableSource::Jpegli => {
                // Preserve current config if already jpegli; otherwise default to Jpegli
                if matches!(
                    self.quant_table_config,
                    QuantTableConfig::MozjpegRobidoux | QuantTableConfig::Custom(_)
                ) {
                    self.quant_table_config = QuantTableConfig::Jpegli;
                }
            }
            QuantTableSource::MozjpegDefault => {
                self.quant_table_config = QuantTableConfig::MozjpegRobidoux;
            }
        }
        self
    }

    #[doc(hidden)]
    #[must_use]
    pub fn optimization(self, preset: super::encoder_types::OptimizationPreset) -> Self {
        use super::encoder_types::OptimizationPreset::*;

        // Scan mode: bundles progressive + script strategy
        let scan_mode = match preset {
            JpegliBaseline => ProgressiveScanMode::Baseline,
            JpegliProgressive => ProgressiveScanMode::Progressive,
            #[cfg(feature = "trellis")]
            MozjpegBaseline | HybridBaseline => ProgressiveScanMode::Baseline,
            #[cfg(feature = "trellis")]
            HybridProgressive => ProgressiveScanMode::Progressive,
            #[cfg(feature = "trellis")]
            MozjpegProgressive => ProgressiveScanMode::ProgressiveMozjpeg,
            #[cfg(feature = "trellis")]
            MozjpegMaxCompression | HybridMaxCompression => ProgressiveScanMode::ProgressiveSearch,
        };

        // Quant table config: bundles source + chroma layout
        let quant_table_config = match preset {
            JpegliBaseline | JpegliProgressive => QuantTableConfig::Jpegli,
            #[cfg(feature = "trellis")]
            MozjpegBaseline | MozjpegProgressive | MozjpegMaxCompression => {
                QuantTableConfig::MozjpegRobidoux
            }
            #[cfg(feature = "trellis")]
            HybridBaseline | HybridProgressive | HybridMaxCompression => QuantTableConfig::Jpegli,
        };

        // Trellis configuration depends on preset lineage:
        // - Jpegli: no trellis (AQ-driven quality, no rate-distortion opt)
        // - Mozjpeg: Thorough (full search, matching C mozjpeg default)
        // - Hybrid: Adaptive (zenjpeg heuristic, good speed/quality balance)
        #[cfg(feature = "trellis")]
        let trellis = {
            use super::trellis::{TrellisConfig, TrellisSpeedMode};
            match preset {
                JpegliBaseline | JpegliProgressive => None,
                MozjpegBaseline | MozjpegProgressive | MozjpegMaxCompression => {
                    Some(TrellisConfig::default().speed_mode(TrellisSpeedMode::Thorough))
                }
                HybridBaseline | HybridProgressive => Some(TrellisConfig::default()),
                HybridMaxCompression => {
                    Some(TrellisConfig::default().speed_mode(TrellisSpeedMode::Thorough))
                }
            }
        };

        // Deringing: independent of AQ. C mozjpeg enables overshoot deringing
        // only for JCP_MAX_COMPRESSION profile. All jpegli/hybrid presets use it
        // (quality win, negligible cost). Mozjpeg baseline/progressive skip it
        // to match C mozjpeg's default profile.
        let deringing = match preset {
            JpegliBaseline | JpegliProgressive => true,
            #[cfg(feature = "trellis")]
            MozjpegBaseline | MozjpegProgressive => false,
            #[cfg(feature = "trellis")]
            MozjpegMaxCompression => true,
            #[cfg(feature = "trellis")]
            HybridBaseline | HybridProgressive | HybridMaxCompression => true,
        };

        Self {
            scan_mode,
            quant_table_config,
            huffman: HuffmanStrategy::Optimize,
            deringing,
            aq_enabled: preset.uses_aq(),
            #[cfg(feature = "trellis")]
            trellis,
            // Presets force baseline quant tables (matching cjpegli CLI and C
            // mozjpeg behavior). XYB SOF1 is handled by force_sof1, not this flag.
            allow_16bit_quant_tables: false,
            ..self
        }
    }

    /// Enable or disable Huffman table optimization.
    ///
    /// When enabled (default), a two-pass encode computes optimal Huffman tables
    /// from the image data. This produces the smallest files.
    ///
    #[doc(hidden)]
    #[must_use]
    pub fn optimize_huffman(self, enable: bool) -> Self {
        self.huffman(enable)
    }

    /// Set the Huffman table strategy.
    ///
    /// Accepts `bool`, `HuffmanStrategy`, or `HuffmanTableSet`.
    ///
    /// - `true` → `HuffmanStrategy::Optimize` (two-pass, smallest files)
    /// - `false` → `HuffmanStrategy::Fixed` (single-pass, ~2.5% larger)
    /// - `HuffmanStrategy::*` → explicit strategy selection
    /// - `HuffmanTableSet` → custom tables (single-pass)
    ///
    /// Progressive mode requires `HuffmanStrategy::Optimize`.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use zenjpeg::encoder::{EncoderConfig, HuffmanStrategy, HuffmanTableSet};
    ///
    /// // Boolean (same as old optimize_huffman)
    /// let config = EncoderConfig::ycbcr(85, sub).huffman(true);
    ///
    /// // Explicit strategy
    /// let config = EncoderConfig::ycbcr(85, sub).huffman(HuffmanStrategy::FixedAnnexK);
    ///
    /// // Custom tables
    /// let tables = HuffmanTableSet::annex_k()?;
    /// let config = EncoderConfig::ycbcr(85, sub).huffman(tables);
    /// ```
    #[must_use]
    pub fn huffman(mut self, strategy: impl Into<HuffmanStrategy>) -> Self {
        self.huffman = strategy.into();
        self
    }

    /// Allow 16-bit quantization tables.
    ///
    /// When enabled, quantization values can exceed 255, using 16-bit DQT
    /// markers and SOF1 (extended sequential) when needed.
    ///
    /// When disabled (default), quantization values are clamped to 255,
    /// using 8-bit DQT markers. This saves ~128 bytes at low quality.
    ///
    /// Note: XYB always uses SOF1 regardless of this setting because its
    /// wider dynamic range produces DC categories exceeding the baseline
    /// limit of 11. This flag only controls quant value precision, not
    /// the frame type.
    #[must_use]
    pub fn allow_16bit_quant_tables(mut self, enable: bool) -> Self {
        self.allow_16bit_quant_tables = enable;
        self
    }

    /// Use separate quantization tables for Cb and Cr components.
    ///
    /// When enabled (default), uses 3 quantization tables:
    /// - Table 0: Y (luma)
    /// - Table 1: Cb (blue chroma)
    /// - Table 2: Cr (red chroma)
    ///
    /// When disabled, uses 2 quantization tables:
    /// - Table 0: Y (luma)
    /// - Table 1: Cb and Cr (shared chroma)
    ///
    /// # Compatibility
    ///
    /// - 3 tables (default): Matches C++ jpegli's `jpegli_set_distance()` behavior
    /// - 2 tables: Matches C++ jpegli's `jpeg_set_quality()` behavior
    ///
    /// Use 2 tables when you need exact output parity with tools that use
    /// `jpeg_set_quality()` (most libjpeg-based encoders).
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Match jpeg_set_quality() behavior (2 tables)
    /// let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
    ///     .separate_chroma_tables(false);
    /// ```
    #[must_use]
    pub fn separate_chroma_tables(mut self, enable: bool) -> Self {
        // Map the bool to the appropriate QuantTableConfig variant,
        // preserving mozjpeg vs jpegli distinction.
        match &self.quant_table_config {
            QuantTableConfig::Custom(_) | QuantTableConfig::GlassaLowBpp(_) => {
                // Don't touch custom or Glassa tables (Glassa always uses shared chroma)
            }
            QuantTableConfig::MozjpegRobidoux => {
                // MozjpegRobidoux is always shared chroma; can't separate
                if enable {
                    // User is asking for separate chroma with mozjpeg tables,
                    // which isn't a valid combo — switch to Jpegli
                    self.quant_table_config = QuantTableConfig::Jpegli;
                }
            }
            QuantTableConfig::Jpegli | QuantTableConfig::JpegliSharedChroma => {
                self.quant_table_config = if enable {
                    QuantTableConfig::Jpegli
                } else {
                    QuantTableConfig::JpegliSharedChroma
                };
            }
        }
        self
    }

    /// Force baseline-compatible quantization and sequential scan mode.
    ///
    /// Equivalent to `.progressive(false).allow_16bit_quant_tables(false)`.
    ///
    /// Disables progressive encoding and clamps quant values to 255.
    /// For YCbCr, this produces true baseline JPEGs (SOF0).
    /// For XYB, this still uses SOF1 (required for DC categories >11),
    /// but with 8-bit quant tables for maximum decoder compatibility.
    #[must_use]
    pub fn force_baseline(self) -> Self {
        self.progressive(false).allow_16bit_quant_tables(false)
    }

    /// Set the restart marker interval in MCU rows (default: 4).
    ///
    /// An MCU (Minimum Coded Unit) row is one row of coding units in the
    /// JPEG grid. Its height in pixels depends on chroma subsampling:
    /// - 4:4:4 / 4:2:2: 8 pixels tall (1 block row)
    /// - 4:2:0 / 4:4:0: 16 pixels tall (2 block rows)
    ///
    /// The value is resolved to an exact MCU count at encode time when
    /// image dimensions are known (interval = rows × mcu_cols). This
    /// guarantees MCU-row-aligned restart boundaries, which is required
    /// for the fused chroma upsample + color conversion decode path.
    ///
    /// # Why restart markers matter
    ///
    /// Restart markers reset the DC prediction state and byte-align the
    /// bitstream, enabling:
    /// - **Parallel decoding**: independent segments can be decoded on
    ///   separate threads
    /// - **Error recovery**: corruption in one segment doesn't propagate
    /// - **Random access**: decoders can seek to any restart boundary
    ///
    /// # Compression overhead
    ///
    /// Row-aligned restart markers have negligible overhead because the
    /// DC prediction already makes a large jump at row boundaries (the
    /// last MCU of one row is spatially distant from the first MCU of
    /// the next row). Measured across 80 images at Q85:
    ///
    /// | MCU rows | Overhead | Pixel height (4:2:0) |
    /// |----------|----------|---------------------|
    /// | 1        | +0.16%   | 16 px               |
    /// | 4        | +0.04%   | 64 px               |
    /// | 8        | +0.02%   | 128 px              |
    ///
    /// Non-row-aligned intervals (e.g., DRI=64 MCUs) cost 3-15× more
    /// because they break DC prediction between adjacent MCUs mid-row.
    ///
    /// # Special values
    ///
    /// - `0`: disable restart markers entirely
    /// - `4` (default): good balance of parallelism and overhead
    #[must_use]
    pub fn restart_mcu_rows(mut self, rows: u16) -> Self {
        self.restart_mcu_rows = rows;
        self
    }

    /// Enable parallel encoding for improved throughput on multi-core systems.
    ///
    /// When enabled, the encoder uses multiple threads for:
    /// - DCT computation (block transforms)
    /// - Entropy/Huffman encoding (via restart markers)
    ///
    /// # Restart Marker Behavior
    ///
    /// Parallel entropy encoding requires restart markers between segments.
    /// The default of 4 MCU rows works well. If `restart_mcu_rows` is 0,
    /// the encoder auto-selects an optimal interval (4-16 rows) targeting
    /// 8+ segments for good parallelism.
    ///
    /// # Performance
    ///
    /// - 2 threads: ~1.2-1.6x speedup
    /// - 4 threads: ~1.3-1.7x speedup
    /// - Minimum useful size: ~512x512 (smaller images have too much overhead)
    ///
    /// # Example
    ///
    /// ```ignore
    /// use zenjpeg::{EncoderConfig, ChromaSubsampling, ParallelEncoding};
    ///
    /// let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter)
    ///     .parallel(ParallelEncoding::Auto);
    /// ```
    ///
    /// Requires the `parallel` feature flag.
    #[cfg(feature = "parallel")]
    #[must_use]
    pub fn parallel(mut self, mode: super::encoder_types::ParallelEncoding) -> Self {
        self.parallel = Some(mode);
        self
    }

    /// Configure hybrid quantization (jpegli AQ + mozjpeg trellis).
    ///
    /// Set hybrid AQ+trellis configuration directly.
    ///
    /// **Expert API.** Prefer using [`.expert()`](Self::expert) with
    /// [`ExpertConfig`](super::search::ExpertConfig) for full control.
    ///
    /// When a `HybridConfig` with `enabled = true` is set, it takes
    /// priority over any `TrellisConfig`.
    #[doc(hidden)]
    #[cfg(feature = "trellis")]
    #[must_use]
    pub fn hybrid_config(mut self, config: super::trellis::HybridConfig) -> Self {
        self.hybrid_config = config;
        if config.enabled {
            self.trellis = None;
        }
        self
    }

    // === Trellis Quantization ===

    /// Set trellis quantization configuration directly.
    ///
    /// **Expert API.** Prefer using [`.expert()`](Self::expert) with
    /// [`ExpertConfig`](super::search::ExpertConfig) for full control,
    /// or [`.optimization()`](Self::optimization) with presets.
    #[doc(hidden)]
    #[cfg(feature = "trellis")]
    #[must_use]
    pub fn trellis(mut self, config: TrellisConfig) -> Self {
        self.trellis = Some(config);
        self
    }

    /// Get the trellis configuration, if set.
    #[cfg(feature = "trellis")]
    #[must_use]
    pub fn get_trellis(&self) -> Option<&TrellisConfig> {
        self.trellis.as_ref()
    }

    /// Apply expert configuration overlay.
    ///
    /// Customizes quantization tables and trellis/hybrid settings on top of
    /// the current configuration. Only specified fields are overridden.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use zenjpeg::encode::{EncoderConfig, ExpertConfig, QuantTableConfig, ChromaSubsampling};
    /// use zenjpeg::encode::trellis::TrellisConfig;
    ///
    /// let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
    ///     .expert(ExpertConfig::default()
    ///         .tables(QuantTableConfig::MozjpegRobidoux)
    ///         .trellis(TrellisConfig::default()));
    /// ```
    #[cfg(feature = "trellis")]
    #[must_use]
    pub fn expert(mut self, expert: super::encoder_types::ExpertConfig) -> Self {
        // Apply tables if specified
        if let Some(tables) = expert.tables {
            self.quant_table_config = tables;
        }

        // Apply trellis/hybrid - hybrid takes priority
        if let Some(ref hybrid) = expert.hybrid
            && hybrid.enabled
        {
            self.hybrid_config = *hybrid;
            self.trellis = None;
        }
        if (expert.hybrid.is_none() || !expert.hybrid.as_ref().is_some_and(|h| h.enabled))
            && let Some(trellis) = expert.trellis
        {
            self.trellis = Some(trellis);
        }

        self
    }

    // === Color Mode ===

    /// Set the output color mode.
    #[must_use]
    pub fn color_mode(mut self, mode: ColorMode) -> Self {
        self.color_mode = mode;
        self
    }

    /// Set the chroma downsampling method.
    ///
    /// Only affects RGB/RGBX input with chroma subsampling enabled.
    /// Ignored for grayscale, YCbCr input, or 4:4:4 subsampling.
    #[must_use]
    pub fn downsampling_method(mut self, method: DownsamplingMethod) -> Self {
        self.downsampling_method = method;
        self
    }

    /// Internal: Set edge padding strategy for partial MCU blocks.
    #[doc(hidden)]
    #[must_use]
    pub fn edge_padding_internal(mut self, config: EdgePaddingConfig) -> Self {
        self.edge_padding = config;
        self
    }

    // === Tuning API (doc hidden) ===

    /// Apply custom encoding tables for experimentation.
    ///
    /// This replaces both quantization tables and zero-bias configuration
    /// with values from the provided `EncodingTables`.
    ///
    /// Takes `Box<EncodingTables>` since custom tables are rarely used and
    /// the struct is ~1.5KB. This keeps `EncoderConfig` small by default.
    ///
    /// # Notes
    /// - Tables must match the color mode (YCbCr or XYB)
    /// - When using `ScalingParams::Exact`, quality scaling is bypassed
    /// - When using `ScalingParams::Scaled`, tables are scaled by quality
    ///
    /// # Example
    /// ```
    /// use zenjpeg::encode::{EncoderConfig, ChromaSubsampling};
    /// use zenjpeg::encode::tuning::EncodingTables;
    ///
    /// let mut tables = EncodingTables::default_ycbcr();
    /// tables.scale_quant(0, 0, 0.8);  // Reduce DC quantization
    ///
    /// let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
    ///     .tables(Box::new(tables));
    /// ```
    #[must_use]
    pub fn tables(mut self, tables: Box<super::tuning::EncodingTables>) -> Self {
        self.quant_table_config = QuantTableConfig::Custom(tables);
        self
    }

    /// Enable automatic optimization for best quality/size tradeoff.
    ///
    /// When enabled, applies hybrid trellis quantization (AQ + rate-distortion
    /// optimization) that beats both jpegli and mozjpeg across most quality levels.
    ///
    /// Benchmark results vs alternatives at matched file size:
    /// - vs JpegliProg: **+1.5 SSIM2** points average
    /// - vs cjpegli-444: **+1.6 SSIM2** and **-0.3 Butteraugli**
    ///
    /// Uses jpegli quant tables with hybrid trellis λ=14.5 and progressive encoding.
    /// Requires the `trellis` feature. Without it, this method is not available.
    ///
    /// Quality thresholds (below these, falls back to defaults):
    /// - 4:2:0: q50+ (distance < 5.0)
    /// - 4:4:4: q50+ (distance < 5.0)
    ///
    /// # Example
    /// ```
    /// use zenjpeg::encoder::{EncoderConfig, ChromaSubsampling};
    ///
    /// let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
    ///     .auto_optimize(true);
    /// ```
    #[cfg(feature = "trellis")]
    #[must_use]
    pub fn auto_optimize(mut self, enable: bool) -> Self {
        if !enable {
            return self;
        }

        let distance = self.quality.to_distance();

        // Determine if we're in the quality range where hybrid optimization wins
        // (q50+ for both 4:2:0 and 4:4:4 based on R-D benchmarks)
        let should_use_hybrid = match self.color_mode {
            ColorMode::YCbCr { .. } => distance < 5.0, // q50+
            _ => false,
        };

        // Enable hybrid trellis with λ=14.5 (best R-D tradeoff from benchmarks)
        // Uses default jpegli quant tables - NOT CMA-ES scaling (incompatible)
        if should_use_hybrid {
            self.hybrid_config = super::trellis::HybridConfig {
                enabled: true,
                base_lambda_scale1: 14.5,
                ..Default::default()
            };
            // Clear standalone trellis (hybrid supersedes it)
            self.trellis = None;
        }

        // Enable progressive for better compression
        self.scan_mode = ProgressiveScanMode::Progressive;

        self
    }

    /// Sets custom Huffman tables for single-pass encoding.
    #[doc(hidden)]
    #[must_use]
    pub fn custom_huffman_tables(self, tables: crate::huffman::optimize::HuffmanTableSet) -> Self {
        self.huffman(tables)
    }

    /// Enable or disable SharpYUV (GammaAwareIterative) downsampling.
    ///
    /// SharpYUV produces better color preservation on edges and thin lines,
    /// at the cost of ~3x slower encoding.
    #[must_use]
    pub fn sharp_yuv(self, enable: bool) -> Self {
        self.downsampling_method(if enable {
            DownsamplingMethod::GammaAwareIterative
        } else {
            DownsamplingMethod::Box
        })
    }

    /// Enable or disable overshoot deringing (enabled by default).
    ///
    /// Deringing reduces ringing artifacts on white backgrounds by smoothing hard
    /// edges. It allows pixel values to "overshoot" beyond the displayable range.
    /// Since JPEG decoders clamp values to 0-255, the overshoot is invisible but
    /// the smoother curve compresses better with fewer artifacts.
    ///
    /// This technique was pioneered by [@kornel](https://github.com/kornelski) in
    /// [mozjpeg](https://github.com/mozilla/mozjpeg) and significantly improves
    /// quality for documents, graphics, and text without degrading photographic
    /// content.
    ///
    /// Particularly effective for:
    /// - Documents and screenshots with white backgrounds
    /// - Text and graphics with hard edges
    /// - Any image with saturated regions (pixels at 0 or 255)
    ///
    /// There is no quality downside to leaving this enabled for photos.
    #[must_use]
    pub fn deringing(mut self, enable: bool) -> Self {
        self.deringing = enable;
        self
    }

    /// Enable or disable adaptive quantization (jpegli AQ).
    ///
    /// When enabled (default), the encoder computes per-block AQ strengths from
    /// luminance data, adjusting quantization to allocate more bits to smooth
    /// areas and fewer to textured areas.
    ///
    /// When disabled, AQ computation is skipped entirely and all blocks receive
    /// neutral AQ strength (0.0). This saves memory (~600KB-2.5MB depending on
    /// image size) and computation.
    ///
    /// Mozjpeg presets disable AQ automatically via
    /// [`optimization()`](Self::optimization). For mozjpeg presets where
    /// `zero_bias_mul` is all-zeros, disabling AQ produces identical output
    /// since AQ values are never applied.
    #[must_use]
    pub fn aq_enabled(mut self, enable: bool) -> Self {
        self.aq_enabled = enable;
        self
    }

    /// Set Gaussian blur sigma applied before encoding (0.0 = disabled).
    ///
    /// A mild blur (σ ≈ 0.4) before JPEG encoding reduces file size ~5% with
    /// negligible perceptual quality loss (butteraugli delta < 0.2).
    ///
    /// Only applies to packed u8 sRGB input (Rgb8Srgb, Rgba8Srgb, etc.).
    /// Has no effect on f32/u16 linear input or YCbCr input.
    ///
    /// # Example
    /// ```ignore
    /// let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
    ///     .pre_blur(0.4);
    /// ```
    #[must_use]
    pub fn pre_blur(mut self, sigma: f32) -> Self {
        self.pre_blur = sigma;
        self
    }

    // === Validation ===

    /// Validate the configuration, returning an error for invalid combinations.
    ///
    /// Invalid combinations:
    /// - Progressive mode with disabled Huffman optimization
    pub fn validate(&self) -> Result<()> {
        if self.scan_mode.is_progressive() && !matches!(self.huffman, HuffmanStrategy::Optimize) {
            return Err(crate::error::Error::invalid_config(
                "progressive mode requires optimized Huffman tables".into(),
            ));
        }
        Ok(())
    }

    // === Request Builder ===

    /// Create a per-image encode request from this config.
    ///
    /// Returns an [`EncodeRequest`](super::request::EncodeRequest) that can bind per-image metadata (ICC, EXIF, XMP)
    /// and controls (stop token, limits) without modifying the reusable config.
    ///
    /// # Example
    /// ```ignore
    /// use zenjpeg::encoder::{EncoderConfig, ChromaSubsampling, Exif, Orientation};
    ///
    /// let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);
    ///
    /// // Reuse config, vary metadata per image
    /// let jpeg = config.request()
    ///     .icc_profile(&srgb_bytes)
    ///     .exif(Exif::build().orientation(Orientation::Rotate90))
    ///     .encode(&pixels, 1920, 1080)?;
    /// ```
    #[must_use]
    pub fn request(&self) -> super::request::EncodeRequest<'_> {
        super::request::EncodeRequest::new(self)
    }

    // === Encoder Creation ===

    /// Create an encoder from raw bytes with explicit pixel layout.
    ///
    /// Use this when working with raw byte buffers and you know the pixel layout.
    ///
    /// # Arguments
    /// - `width`: Image width in pixels
    /// - `height`: Image height in pixels
    /// - `layout`: Pixel data layout (channel order, depth, color space)
    ///
    /// # Example
    /// ```ignore
    /// use zenjpeg::{EncoderConfig, ChromaSubsampling, PixelLayout, Unstoppable};
    /// let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);
    /// let mut enc = config.encode_from_bytes(1920, 1080, PixelLayout::Rgb8Srgb)?;
    /// enc.push_packed(&rgb_bytes, Unstoppable)?;
    /// let jpeg = enc.finish()?;
    /// ```
    pub fn encode_from_bytes(
        &self,
        width: u32,
        height: u32,
        layout: PixelLayout,
    ) -> Result<BytesEncoder> {
        self.validate()?;
        BytesEncoder::new(self.clone(), width, height, layout, None, None, None)
    }

    /// Create an encoder from rgb crate pixel types.
    ///
    /// Layout is inferred from the type parameter. For RGBA/BGRA types,
    /// the 4th channel is ignored.
    ///
    /// # Type Parameter
    /// - `P`: Pixel type from the `rgb` crate (e.g., `RGB<u8>`, `RGBA<f32>`)
    ///
    /// # Example
    /// ```ignore
    /// use rgb::RGB;
    /// use zenjpeg::{EncoderConfig, ChromaSubsampling, Unstoppable};
    ///
    /// let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);
    /// let mut enc = config.encode_from_rgb::<RGB<u8>>(1920, 1080)?;
    /// enc.push_packed(&pixels, Unstoppable)?;
    /// let jpeg = enc.finish()?;
    /// ```
    pub fn encode_from_rgb<P: super::byte_encoders::Pixel>(
        &self,
        width: u32,
        height: u32,
    ) -> Result<RgbEncoder<P>> {
        self.validate()?;
        RgbEncoder::new(self.clone(), width, height, None, None, None)
    }

    /// Create an encoder from planar YCbCr data.
    ///
    /// Use this when you have pre-converted YCbCr from video decoders, etc.
    /// Skips RGB->YCbCr conversion entirely.
    ///
    /// Only valid with `ColorMode::YCbCr`. XYB mode requires RGB input.
    ///
    /// # Example
    /// ```ignore
    /// use zenjpeg::{EncoderConfig, ChromaSubsampling, Unstoppable};
    ///
    /// let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);
    /// let mut enc = config.encode_from_ycbcr_planar(1920, 1080)?;
    /// enc.push(&planes, height, Unstoppable)?;
    /// let jpeg = enc.finish()?;
    /// ```
    pub fn encode_from_ycbcr_planar(&self, width: u32, height: u32) -> Result<YCbCrPlanarEncoder> {
        self.validate()?;

        // Validate color mode
        if !matches!(self.color_mode, ColorMode::YCbCr { .. }) {
            return Err(crate::error::Error::invalid_config(
                "planar YCbCr input requires YCbCr color mode".into(),
            ));
        }

        YCbCrPlanarEncoder::new(self.clone(), width, height, None, None, None)
    }

    // === One-shot Convenience ===

    /// Encode a complete image from rgb crate pixel types in one call.
    ///
    /// This is a convenience wrapper around `encode_from_rgb` + `push_packed` + `finish`.
    /// For streaming or partial-image encoding, use [`encode_from_rgb`](Self::encode_from_rgb).
    ///
    /// # Example
    /// ```ignore
    /// use rgb::RGB;
    /// use zenjpeg::encoder::{EncoderConfig, ChromaSubsampling};
    ///
    /// let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);
    /// let jpeg = config.encode(&pixels, 1920, 1080)?;
    /// ```
    pub fn encode<P: super::byte_encoders::Pixel>(
        &self,
        pixels: &[P],
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>> {
        let mut enc = self.encode_from_rgb::<P>(width, height)?;
        enc.push_packed(pixels, enough::Unstoppable)?;
        enc.finish()
    }

    /// Encode a complete image into a caller-provided buffer.
    ///
    /// Like [`encode`](Self::encode) but writes into an existing `Vec<u8>` instead
    /// of allocating a new one.
    pub fn encode_into<P: super::byte_encoders::Pixel>(
        &self,
        pixels: &[P],
        width: u32,
        height: u32,
        output: &mut Vec<u8>,
    ) -> Result<()> {
        let mut enc = self.encode_from_rgb::<P>(width, height)?;
        enc.push_packed(pixels, enough::Unstoppable)?;
        enc.finish_into(output)
    }

    /// Encode a complete image from raw byte data in one call.
    ///
    /// This is a convenience wrapper around `encode_from_bytes` + `push_packed` + `finish`.
    pub fn encode_bytes(
        &self,
        data: &[u8],
        width: u32,
        height: u32,
        layout: PixelLayout,
    ) -> Result<Vec<u8>> {
        let mut enc = self.encode_from_bytes(width, height, layout)?;
        enc.push_packed(data, enough::Unstoppable)?;
        enc.finish()
    }

    /// Encode a complete image from raw byte data into a caller-provided buffer.
    pub fn encode_bytes_into(
        &self,
        data: &[u8],
        width: u32,
        height: u32,
        layout: PixelLayout,
        output: &mut Vec<u8>,
    ) -> Result<()> {
        let mut enc = self.encode_from_bytes(width, height, layout)?;
        enc.push_packed(data, enough::Unstoppable)?;
        enc.finish_into(output)
    }

    // === Resource Estimation ===

    /// Estimate peak memory usage for encoding an image of the given dimensions.
    ///
    /// Returns estimated bytes based on color mode, subsampling, and dimensions.
    /// Delegates to the streaming encoder's estimate which accounts for all
    /// internal buffers.
    #[must_use]
    pub fn estimate_memory(&self, width: u32, height: u32) -> usize {
        use crate::encode::streaming::StreamingEncoder;

        let subsampling = match self.color_mode {
            ColorMode::YCbCr { subsampling } => subsampling.into(),
            ColorMode::Xyb { .. } => crate::types::Subsampling::S444,
            ColorMode::Grayscale => crate::types::Subsampling::S444,
        };

        StreamingEncoder::new(width, height)
            .subsampling(subsampling)
            .huffman(self.huffman.clone())
            .estimate_memory_usage()
    }

    /// Returns an absolute ceiling on memory usage.
    ///
    /// Unlike `estimate_memory`, this returns a **guaranteed upper bound**
    /// that actual peak memory will never exceed. Use this for resource reservation
    /// when you need certainty rather than a close estimate.
    ///
    /// The ceiling accounts for:
    /// - Worst-case token counts per block (high-frequency content)
    /// - Maximum output buffer size (incompressible images)
    /// - Vec capacity overhead (allocator rounding)
    /// - All intermediate buffers at their maximum sizes
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use zenjpeg::encoder::{EncoderConfig, ChromaSubsampling};
    ///
    /// let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);
    /// let ceiling = config.estimate_memory_ceiling(1920, 1080);
    ///
    /// // Reserve this much memory - actual usage guaranteed to be less
    /// let buffer = Vec::with_capacity(ceiling);
    /// ```
    #[must_use]
    pub fn estimate_memory_ceiling(&self, width: u32, height: u32) -> usize {
        use crate::encode::streaming::StreamingEncoder;

        let subsampling = match self.color_mode {
            ColorMode::YCbCr { subsampling } => subsampling.into(),
            ColorMode::Xyb { .. } => crate::types::Subsampling::S444,
            ColorMode::Grayscale => crate::types::Subsampling::S444,
        };

        StreamingEncoder::new(width, height)
            .subsampling(subsampling)
            .estimate_memory_ceiling()
    }

    // === Accessors ===

    /// Get the configured quality.
    #[must_use]
    pub fn get_quality(&self) -> Quality {
        self.quality
    }

    /// Get the configured color mode.
    #[must_use]
    pub fn get_color_mode(&self) -> ColorMode {
        self.color_mode
    }

    /// Check if progressive mode is enabled.
    #[must_use]
    pub fn is_progressive(&self) -> bool {
        self.scan_mode.is_progressive()
    }

    /// Get the current scan mode.
    #[must_use]
    pub fn get_scan_mode(&self) -> ProgressiveScanMode {
        self.scan_mode
    }

    /// Get the current quantization table configuration.
    #[must_use]
    pub fn get_quant_table_config(&self) -> &QuantTableConfig {
        &self.quant_table_config
    }

    /// Check if Huffman optimization is enabled.
    #[must_use]
    pub fn is_optimize_huffman(&self) -> bool {
        matches!(self.huffman, HuffmanStrategy::Optimize)
    }

    /// Check if 16-bit quantization tables are allowed.
    #[must_use]
    pub fn is_allow_16bit_quant_tables(&self) -> bool {
        self.allow_16bit_quant_tables
    }

    /// Check if adaptive quantization (AQ) is enabled.
    #[must_use]
    pub fn is_aq_enabled(&self) -> bool {
        self.aq_enabled
    }

    /// Check if separate chroma tables are enabled (3 tables vs 2).
    #[must_use]
    pub fn is_separate_chroma_tables(&self) -> bool {
        self.quant_table_config.separate_chroma_tables()
    }

    /// Internal: Get the configured edge padding.
    #[doc(hidden)]
    #[must_use]
    pub fn get_edge_padding(&self) -> EdgePaddingConfig {
        self.edge_padding
    }

    // === Segment Injection ===

    /// Add prepared segments for injection into output.
    ///
    /// Use this to preserve metadata during round-trip encoding or to inject
    /// custom metadata and MPF secondary images.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use zenjpeg::decoder::Decoder;
    /// use zenjpeg::encoder::{EncoderConfig, ChromaSubsampling};
    ///
    /// // Decode with metadata preservation
    /// let decoded = Decoder::new().decode(&original)?;
    /// let extras = decoded.extras().unwrap();
    ///
    /// // Re-encode with same metadata
    /// let config = EncoderConfig::ycbcr(90.0, ChromaSubsampling::Quarter)
    ///     .with_segments(extras.to_encoder_segments());
    /// ```
    #[must_use]
    pub fn with_segments(mut self, segments: super::extras::EncoderSegments) -> Self {
        self.segments = Some(segments);
        self
    }

    /// Add a single segment (convenience method).
    ///
    /// The segment type is inferred from the marker and data.
    #[must_use]
    pub fn add_segment(mut self, marker: u8, data: Vec<u8>) -> Self {
        use super::extras::EncoderSegments;
        self.segments
            .get_or_insert_with(EncoderSegments::new)
            .add_raw_mut(marker, data);
        self
    }

    /// Add an MPF secondary image (gain map, depth map, etc.).
    ///
    /// The image data must be a complete JPEG file. An MPF directory
    /// will be automatically generated during encoding.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use zenjpeg::encoder::{EncoderConfig, ChromaSubsampling, MpfImageType};
    ///
    /// let config = EncoderConfig::ycbcr(90.0, ChromaSubsampling::Quarter)
    ///     .add_mpf_image(gainmap_jpeg, MpfImageType::Undefined);
    /// ```
    #[must_use]
    pub fn add_mpf_image(mut self, jpeg: Vec<u8>, typ: super::extras::MpfImageType) -> Self {
        use super::extras::EncoderSegments;
        self.segments
            .get_or_insert_with(EncoderSegments::new)
            .add_mpf_image_mut(jpeg, typ);
        self
    }

    /// Add a gain map (convenience for `MpfImageType::Undefined`).
    ///
    /// Gain maps are used by UltraHDR for HDR rendering. The image data
    /// must be a complete JPEG file (typically grayscale).
    #[must_use]
    pub fn add_gainmap(self, jpeg: Vec<u8>) -> Self {
        self.add_mpf_image(jpeg, super::extras::MpfImageType::Undefined)
    }

    /// Get the configured segments, if any.
    #[must_use]
    pub fn get_segments(&self) -> Option<&super::extras::EncoderSegments> {
        self.segments.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "trellis")]
    use crate::encode::trellis::TrellisSpeedMode;

    #[test]
    fn test_ycbcr_config() {
        let config = EncoderConfig::ycbcr(90.0, ChromaSubsampling::None);
        assert!(matches!(config.quality, Quality::ApproxJpegli(90.0)));
        assert!(config.scan_mode.is_progressive()); // Progressive is now the default
        assert!(matches!(config.huffman, HuffmanStrategy::Optimize));
        assert!(matches!(
            config.color_mode,
            ColorMode::YCbCr {
                subsampling: ChromaSubsampling::None
            }
        ));
    }

    #[test]
    fn test_xyb_config() {
        let config = EncoderConfig::xyb(90.0, XybSubsampling::BQuarter);
        assert!(matches!(config.quality, Quality::ApproxJpegli(90.0)));
        assert!(matches!(
            config.color_mode,
            ColorMode::Xyb {
                subsampling: XybSubsampling::BQuarter
            }
        ));

        let config = EncoderConfig::xyb(90.0, XybSubsampling::Full);
        assert!(matches!(
            config.color_mode,
            ColorMode::Xyb {
                subsampling: XybSubsampling::Full
            }
        ));
    }

    #[test]
    fn test_grayscale_config() {
        let config = EncoderConfig::grayscale(85);
        assert!(matches!(config.quality, Quality::ApproxJpegli(85.0)));
        assert!(matches!(config.color_mode, ColorMode::Grayscale));
    }

    #[test]
    fn test_builder_pattern() {
        let config = EncoderConfig::ycbcr(85, ChromaSubsampling::None)
            .progressive(true)
            .sharp_yuv(true);

        assert!(matches!(config.quality, Quality::ApproxJpegli(85.0)));
        assert!(config.scan_mode.is_progressive());
        assert!(matches!(config.huffman, HuffmanStrategy::Optimize)); // auto-enabled by progressive
        assert!(matches!(
            config.color_mode,
            ColorMode::YCbCr {
                subsampling: ChromaSubsampling::None
            }
        ));
        assert!(matches!(
            config.downsampling_method,
            DownsamplingMethod::GammaAwareIterative
        ));
    }

    #[test]
    fn test_progressive_enables_huffman() {
        let config = EncoderConfig::ycbcr(90.0, ChromaSubsampling::None)
            .optimize_huffman(false)
            .progressive(true);

        assert!(matches!(config.huffman, HuffmanStrategy::Optimize));
    }

    #[test]
    fn test_validation_progressive_huffman() {
        let mut config = EncoderConfig::ycbcr(90.0, ChromaSubsampling::None);
        config.scan_mode = ProgressiveScanMode::Progressive;
        config.huffman = HuffmanStrategy::Fixed;

        assert!(config.validate().is_err());
    }

    #[test]
    fn test_deprecated_new_still_works() {
        // Ensure backward compatibility during migration
        let config = EncoderConfig::ycbcr(90.0, ChromaSubsampling::Quarter);
        assert!(matches!(config.quality, Quality::ApproxJpegli(90.0)));
        assert!(matches!(
            config.color_mode,
            ColorMode::YCbCr {
                subsampling: ChromaSubsampling::Quarter
            }
        ));
    }

    #[cfg(feature = "trellis")]
    #[test]
    fn test_trellis_config() {
        // Default config has no trellis
        let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter);
        assert!(config.trellis.is_none());
        assert!(config.get_trellis().is_none());

        // Enable trellis with defaults
        let config =
            EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter).trellis(TrellisConfig::default());
        assert!(config.trellis.is_some());
        let trellis = config.get_trellis().unwrap();
        assert!(trellis.is_ac_enabled());
        assert!(trellis.is_dc_enabled());
        assert_eq!(trellis.get_speed_mode(), TrellisSpeedMode::Adaptive);
    }

    #[cfg(feature = "trellis")]
    #[test]
    fn test_trellis_config_builder() {
        let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter).trellis(
            TrellisConfig::default()
                .ac_trellis(true)
                .dc_trellis(false)
                .speed_mode(TrellisSpeedMode::Level(5))
                .rd_factor(0.8),
        );

        let trellis = config.get_trellis().unwrap();
        assert!(trellis.is_ac_enabled());
        assert!(!trellis.is_dc_enabled());
        assert_eq!(trellis.get_speed_mode(), TrellisSpeedMode::Level(5));
    }

    #[cfg(feature = "trellis")]
    #[test]
    fn test_trellis_disabled() {
        let config =
            EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter).trellis(TrellisConfig::disabled());

        let trellis = config.get_trellis().unwrap();
        assert!(!trellis.is_enabled());
        assert!(!trellis.is_ac_enabled());
        assert!(!trellis.is_dc_enabled());
    }

    #[test]
    fn test_optimization_preset_jpegli_baseline() {
        use crate::encode::encoder_types::{
            OptimizationPreset, ProgressiveScanMode, QuantTableConfig,
        };
        let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
            .optimization(OptimizationPreset::JpegliBaseline);
        assert_eq!(config.scan_mode, ProgressiveScanMode::Baseline);
        assert!(config.deringing);
        assert_eq!(config.quant_table_config, QuantTableConfig::Jpegli);
        #[cfg(feature = "trellis")]
        assert!(config.trellis.is_none());
        assert!(!config.allow_16bit_quant_tables);
    }

    #[cfg(feature = "trellis")]
    #[test]
    fn test_optimization_preset_mozjpeg_baseline() {
        use crate::encode::encoder_types::{
            OptimizationPreset, ProgressiveScanMode, QuantTableConfig,
        };
        let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
            .optimization(OptimizationPreset::MozjpegBaseline);
        assert_eq!(config.scan_mode, ProgressiveScanMode::Baseline);
        assert!(!config.deringing); // C mozjpeg default profile: no overshoot
        assert_eq!(config.quant_table_config, QuantTableConfig::MozjpegRobidoux);
        assert!(config.trellis.is_some());
        let trellis = config.trellis.unwrap();
        assert_eq!(trellis.get_speed_mode(), TrellisSpeedMode::Thorough); // C mozjpeg = full search
        assert!(!config.allow_16bit_quant_tables);
    }

    #[cfg(feature = "trellis")]
    #[test]
    fn test_optimization_preset_mozjpeg_progressive() {
        use crate::encode::encoder_types::{
            OptimizationPreset, ProgressiveScanMode, QuantTableConfig,
        };
        let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
            .optimization(OptimizationPreset::MozjpegProgressive);
        assert_eq!(config.scan_mode, ProgressiveScanMode::ProgressiveMozjpeg);
        assert!(!config.deringing); // C mozjpeg default profile: no overshoot
        assert_eq!(config.quant_table_config, QuantTableConfig::MozjpegRobidoux);
        assert!(config.trellis.is_some());
        let trellis = config.trellis.unwrap();
        assert_eq!(trellis.get_speed_mode(), TrellisSpeedMode::Thorough); // C mozjpeg = full search
        assert!(!config.allow_16bit_quant_tables);
    }

    #[cfg(feature = "trellis")]
    #[test]
    fn test_optimization_preset_mozjpeg_max() {
        use crate::encode::encoder_types::{
            OptimizationPreset, ProgressiveScanMode, QuantTableConfig,
        };
        let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
            .optimization(OptimizationPreset::MozjpegMaxCompression);
        assert_eq!(config.scan_mode, ProgressiveScanMode::ProgressiveSearch);
        assert!(config.deringing); // JCP_MAX_COMPRESSION enables overshoot
        assert_eq!(config.quant_table_config, QuantTableConfig::MozjpegRobidoux);
        assert!(config.trellis.is_some());
        let trellis = config.trellis.unwrap();
        assert_eq!(trellis.get_speed_mode(), TrellisSpeedMode::Thorough);
        assert!(!config.allow_16bit_quant_tables);
    }

    #[cfg(feature = "trellis")]
    #[test]
    fn test_optimization_preset_hybrid_progressive() {
        use crate::encode::encoder_types::{
            OptimizationPreset, ProgressiveScanMode, QuantTableConfig,
        };
        let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
            .optimization(OptimizationPreset::HybridProgressive);
        assert_eq!(config.scan_mode, ProgressiveScanMode::Progressive);
        assert!(config.deringing);
        assert_eq!(config.quant_table_config, QuantTableConfig::Jpegli);
        assert!(config.trellis.is_some());
        assert!(!config.allow_16bit_quant_tables);
    }

    #[test]
    fn test_optimization_preset_preserves_quality() {
        use crate::encode::encoder_types::OptimizationPreset;
        let config = EncoderConfig::ycbcr(42.0, ChromaSubsampling::None)
            .optimization(OptimizationPreset::JpegliBaseline);
        assert!(matches!(config.quality, Quality::ApproxJpegli(q) if (q - 42.0).abs() < 0.01));
        assert!(matches!(
            config.color_mode,
            ColorMode::YCbCr {
                subsampling: ChromaSubsampling::None
            }
        ));
    }

    #[cfg(feature = "trellis")]
    #[test]
    fn test_optimization_preset_overridable() {
        use crate::encode::encoder_types::{OptimizationPreset, ProgressiveScanMode};
        // Apply preset then override progressive
        let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
            .optimization(OptimizationPreset::MozjpegProgressive)
            .progressive(false);
        assert_eq!(config.scan_mode, ProgressiveScanMode::Baseline);
        // Trellis should still be set from the preset
        assert!(config.trellis.is_some());
    }

    #[test]
    fn test_progressive_accepts_bool_and_enum() {
        use crate::encode::encoder_types::ProgressiveScanMode;

        // Bool: true → Progressive
        let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter).progressive(true);
        assert_eq!(config.scan_mode, ProgressiveScanMode::Progressive);

        // Bool: false → Baseline
        let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter).progressive(false);
        assert_eq!(config.scan_mode, ProgressiveScanMode::Baseline);

        // Enum: explicit ProgressiveSearch
        let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
            .progressive(ProgressiveScanMode::ProgressiveSearch);
        assert_eq!(config.scan_mode, ProgressiveScanMode::ProgressiveSearch);

        // Enum: ProgressiveMozjpeg
        let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
            .progressive(ProgressiveScanMode::ProgressiveMozjpeg);
        assert_eq!(config.scan_mode, ProgressiveScanMode::ProgressiveMozjpeg);
    }

    #[test]
    fn test_quant_table_config_custom() {
        use crate::encode::encoder_types::QuantTableConfig;
        let tables = crate::encode::tuning::EncodingTables::default_ycbcr();
        let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter).tables(Box::new(tables));
        assert!(matches!(
            config.quant_table_config,
            QuantTableConfig::Custom(_)
        ));
    }

    #[test]
    fn test_separate_chroma_mozjpeg_switches_to_jpegli() {
        use crate::encode::encoder_types::QuantTableConfig;
        // MozjpegRobidoux is always shared chroma; requesting separate
        // should switch to Jpegli tables
        let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
            .quant_table_config(QuantTableConfig::MozjpegRobidoux)
            .separate_chroma_tables(true);
        assert_eq!(config.quant_table_config, QuantTableConfig::Jpegli);
    }

    #[test]
    fn test_aq_enabled_default() {
        let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter);
        assert!(config.is_aq_enabled(), "AQ should be enabled by default");
    }

    #[cfg(feature = "trellis")]
    #[test]
    fn test_aq_enabled_mozjpeg_preset() {
        use crate::encode::encoder_types::OptimizationPreset;
        let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
            .optimization(OptimizationPreset::MozjpegBaseline);
        assert!(!config.is_aq_enabled(), "Mozjpeg presets should disable AQ");

        let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
            .optimization(OptimizationPreset::MozjpegProgressive);
        assert!(!config.is_aq_enabled());

        let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
            .optimization(OptimizationPreset::MozjpegMaxCompression);
        assert!(!config.is_aq_enabled());
    }

    #[test]
    fn test_aq_enabled_jpegli_preset() {
        use crate::encode::encoder_types::OptimizationPreset;
        let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
            .optimization(OptimizationPreset::JpegliBaseline);
        assert!(config.is_aq_enabled(), "Jpegli presets should enable AQ");

        let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
            .optimization(OptimizationPreset::JpegliProgressive);
        assert!(config.is_aq_enabled());
    }

    #[cfg(feature = "trellis")]
    #[test]
    fn test_aq_enabled_hybrid_preset() {
        use crate::encode::encoder_types::OptimizationPreset;
        let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
            .optimization(OptimizationPreset::HybridProgressive);
        assert!(config.is_aq_enabled(), "Hybrid presets should enable AQ");
    }

    #[test]
    fn test_aq_enabled_override() {
        let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter).aq_enabled(false);
        assert!(!config.is_aq_enabled(), "Builder should override default");

        use crate::encode::encoder_types::OptimizationPreset;
        // Override after preset
        let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
            .optimization(OptimizationPreset::JpegliBaseline)
            .aq_enabled(false);
        assert!(!config.is_aq_enabled(), "Builder should override preset");
    }

    #[test]
    fn test_effort_constructors_fast() {
        use crate::encode::encoder_types::Effort;

        let config = EncoderConfig::ycbcr_effort(85, ChromaSubsampling::Quarter, Effort::Fast);
        assert!(config.is_aq_enabled()); // JpegliBaseline uses AQ
        assert!(!config.scan_mode.is_progressive()); // Baseline
    }

    #[cfg(feature = "trellis")]
    #[test]
    fn test_effort_constructors_trellis() {
        use crate::encode::encoder_types::Effort;

        let config = EncoderConfig::ycbcr_effort(85, ChromaSubsampling::Quarter, Effort::Balanced);
        assert!(config.is_aq_enabled()); // HybridProgressive uses AQ
        assert!(config.scan_mode.is_progressive());

        let config = EncoderConfig::ycbcr_effort(85, ChromaSubsampling::Quarter, Effort::Max);
        assert!(config.is_aq_enabled()); // HybridMaxCompression uses AQ
        assert!(config.scan_mode.is_progressive());
    }

    /// Helper: encode a 64x64 test image with the given config.
    fn encode_test_image(config: &EncoderConfig) -> Vec<u8> {
        use crate::encode::encoder_types::PixelLayout;
        // Create a simple 64x64 noise-like pattern (not a gradient!)
        let w = 64u32;
        let h = 64u32;
        let mut pixels = vec![0u8; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let idx = ((y * w + x) * 3) as usize;
                // Simple hash-based pattern for reproducibility
                let v = ((x.wrapping_mul(31).wrapping_add(y.wrapping_mul(67))) % 256) as u8;
                pixels[idx] = v;
                pixels[idx + 1] = v.wrapping_add(50);
                pixels[idx + 2] = v.wrapping_add(100);
            }
        }
        let stride = (w * 3) as usize;
        let mut enc = config
            .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
            .unwrap();
        enc.push(&pixels, h as usize, stride, enough::Unstoppable)
            .unwrap();
        enc.finish().unwrap()
    }

    #[cfg(feature = "trellis")]
    #[test]
    fn test_mozjpeg_aq_disabled_identical_output() {
        use crate::encode::encoder_types::OptimizationPreset;

        // Mozjpeg baseline with AQ enabled (default before this change)
        let config_with_aq = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
            .optimization(OptimizationPreset::MozjpegBaseline)
            .aq_enabled(true);

        // Mozjpeg baseline with AQ disabled (new default from preset)
        let config_without_aq = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
            .optimization(OptimizationPreset::MozjpegBaseline);
        assert!(!config_without_aq.is_aq_enabled());

        let jpeg_with = encode_test_image(&config_with_aq);
        let jpeg_without = encode_test_image(&config_without_aq);

        // Should be byte-identical: mozjpeg presets have zero_bias_mul = 0,
        // so AQ values are never applied to quantization
        assert_eq!(
            jpeg_with.len(),
            jpeg_without.len(),
            "Mozjpeg preset: AQ on vs off should produce same size (zero_bias_mul = 0)"
        );
        assert_eq!(
            jpeg_with, jpeg_without,
            "Mozjpeg preset: AQ on vs off should be byte-identical"
        );
    }

    #[test]
    fn test_jpegli_aq_disabled_different_output() {
        // Jpegli baseline with AQ enabled (default)
        let config_with_aq =
            EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter).progressive(false);
        assert!(config_with_aq.is_aq_enabled());

        // Jpegli baseline with AQ disabled
        let config_without_aq = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
            .progressive(false)
            .aq_enabled(false);
        assert!(!config_without_aq.is_aq_enabled());

        let jpeg_with = encode_test_image(&config_with_aq);
        let jpeg_without = encode_test_image(&config_without_aq);

        // Should differ: jpegli uses non-zero zero_bias_mul, so AQ affects output
        assert_ne!(
            jpeg_with, jpeg_without,
            "Jpegli preset: AQ on vs off should produce different output"
        );
    }
}
