//! Encoder configuration for v2 API.

use super::byte_encoders::{BytesEncoder, RgbEncoder, YCbCrPlanarEncoder};
use super::encoder_types::{
    ChromaSubsampling, ColorMode, DownsamplingMethod, HuffmanStrategy, PixelLayout, Quality,
    ScanStrategy, XybSubsampling,
};
use super::mozjpeg_compat::TrellisConfig;
use super::tuning::EncodingTables;
use crate::error::Result;
use crate::types::EdgePaddingConfig;

/// JPEG encoder configuration. Dimension-independent, reusable across images.
#[derive(Clone, Debug)]
pub struct EncoderConfig {
    pub(crate) quality: Quality,
    /// Custom encoding tables (quantization + zero-bias).
    /// `None` means use perceptual defaults based on color mode and quality.
    pub(crate) tables: Option<Box<EncodingTables>>,
    pub(crate) progressive: bool,
    pub(crate) huffman: HuffmanStrategy,
    pub(crate) color_mode: ColorMode,
    pub(crate) downsampling_method: DownsamplingMethod,
    pub(crate) restart_interval: u16,
    pub(crate) icc_profile: Option<Vec<u8>>,
    pub(crate) exif_data: Option<super::exif::Exif>,
    pub(crate) xmp_data: Option<Vec<u8>>,
    pub(crate) edge_padding: EdgePaddingConfig,
    /// Parallel encoding configuration (requires `parallel` feature)
    #[cfg(feature = "parallel")]
    pub(crate) parallel: Option<super::encoder_types::ParallelEncoding>,
    /// Hybrid quantization configuration.
    pub(crate) hybrid_config: crate::hybrid::config::HybridConfig,
    /// Enable overshoot deringing (on by default).
    pub(crate) deringing: bool,
    /// Allow 16-bit quantization tables (extended JPEG, SOF1).
    /// When false, quant values are clamped to 255 for baseline compatibility.
    pub(crate) allow_16bit_quant_tables: bool,
    /// Use separate quantization tables for Cb and Cr (3 tables total).
    /// When false, Cb and Cr share the same table (2 tables total).
    /// Default is true (3 tables), matching C++ jpegli's `jpegli_set_distance()`.
    /// Set to false for compatibility with `jpeg_set_quality()` behavior.
    pub(crate) separate_chroma_tables: bool,
    /// Trellis quantization configuration (mozjpeg-compatible API).
    /// When Some, enables trellis quantization for rate-distortion optimization.
    pub(crate) trellis: Option<TrellisConfig>,
    /// Prepared segments for injection (EXIF, XMP, ICC, etc.) and MPF secondary images.
    pub(crate) segments: Option<super::extras::EncoderSegments>,
    /// Progressive scan script strategy.
    /// Controls how scans are structured for progressive JPEGs.
    pub(crate) scan_strategy: ScanStrategy,
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

    /// Internal default for non-required fields only.
    fn default_internal() -> Self {
        Self {
            quality: Quality::default(),
            tables: None,      // Use perceptual defaults
            progressive: true, // Progressive gives 3-7% smaller files
            huffman: HuffmanStrategy::Optimize,
            color_mode: ColorMode::default(),
            downsampling_method: DownsamplingMethod::default(),
            restart_interval: 0,
            icc_profile: None,
            exif_data: None,
            xmp_data: None,
            edge_padding: EdgePaddingConfig::default(),
            #[cfg(feature = "parallel")]
            parallel: None,
            hybrid_config: crate::hybrid::config::HybridConfig::default(),
            deringing: true,
            allow_16bit_quant_tables: false,
            separate_chroma_tables: true, // 3 tables (matches jpegli_set_distance)
            trellis: None,
            segments: None,
            scan_strategy: ScanStrategy::Default,
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

    /// Enable or disable progressive encoding.
    ///
    /// Progressive encoding produces multiple scans for incremental display.
    /// Automatically enables optimized Huffman tables (required for progressive).
    #[must_use]
    pub fn progressive(mut self, enable: bool) -> Self {
        self.progressive = enable;
        if enable {
            self.huffman = HuffmanStrategy::Optimize;
        }
        self
    }

    /// Set the progressive scan script strategy.
    ///
    /// Controls how progressive JPEG scan scripts are generated:
    ///
    /// - [`ScanStrategy::Default`]: jpegli-style script with frequency split at 2/3
    ///   and successive approximation for all components.
    /// - [`ScanStrategy::Search`]: Tries 64 candidate configurations and picks the
    ///   smallest (mozjpeg-style `optimize_scans`). 1-3% smaller, ~2x slower.
    /// - [`ScanStrategy::Mozjpeg`]: mozjpeg's default script with frequency split
    ///   at 8/9 and no chroma successive approximation.
    ///
    /// All strategies produce the same decoded pixels—this is a **lossless**
    /// optimization affecting only the progressive scan structure.
    ///
    /// Automatically enables progressive mode and optimized Huffman tables
    /// for non-Default strategies.
    ///
    /// Skipped for XYB mode (XYB uses a fixed scan structure).
    #[must_use]
    pub fn scan_strategy(mut self, strategy: ScanStrategy) -> Self {
        self.scan_strategy = strategy;
        if strategy != ScanStrategy::Default {
            self.progressive = true;
            self.huffman = HuffmanStrategy::Optimize;
        }
        self
    }

    /// Enable progressive scan optimization (mozjpeg-style `optimize_scans`).
    ///
    /// This is equivalent to:
    /// - `scan_strategy(ScanStrategy::Search)` when `enable` is true
    /// - `scan_strategy(ScanStrategy::Default)` when `enable` is false
    ///
    /// Prefer using [`scan_strategy()`](Self::scan_strategy) directly for more control.
    #[must_use]
    pub fn optimize_scans(self, enable: bool) -> Self {
        self.scan_strategy(if enable {
            ScanStrategy::Search
        } else {
            ScanStrategy::Default
        })
    }

    /// Enable or disable Huffman table optimization.
    ///
    /// When enabled (default), a two-pass encode computes optimal Huffman tables
    /// from the image data. This produces the smallest files.
    ///
    /// When disabled, uses general-purpose trained tables (~2.5% larger than optimal).
    /// To use the original JPEG Annex K tables instead, use
    /// [`custom_huffman_tables(HuffmanTableSet::annex_k()?)`](Self::custom_huffman_tables).
    ///
    /// Note: Progressive mode requires optimized Huffman tables.
    #[must_use]
    pub fn optimize_huffman(mut self, enable: bool) -> Self {
        self.huffman = if enable {
            HuffmanStrategy::Optimize
        } else {
            HuffmanStrategy::Fixed
        };
        self
    }

    /// Allow 16-bit quantization tables (extended sequential JPEG, SOF1).
    ///
    /// When enabled (default), quantization values can exceed 255, producing
    /// extended sequential JPEGs (SOF1 marker) for better low-quality precision.
    ///
    /// When disabled, quantization values are clamped to 255, producing
    /// baseline-compatible JPEGs (SOF0 marker) that work with all decoders.
    ///
    /// Most modern decoders support 16-bit quant tables. Only disable this
    /// for maximum compatibility with legacy software.
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
        self.separate_chroma_tables = enable;
        self
    }

    /// Force baseline JPEG compatibility.
    ///
    /// This is a convenience method equivalent to:
    /// ```ignore
    /// config.progressive(false).allow_16bit_quant_tables(false)
    /// ```
    ///
    /// Baseline JPEGs (SOF0) are the most compatible format, supported by
    /// all JPEG decoders. Use this when targeting legacy software or when
    /// maximum compatibility is required.
    #[must_use]
    pub fn force_baseline(self) -> Self {
        self.progressive(false).allow_16bit_quant_tables(false)
    }

    /// Set the restart interval (MCUs between restart markers).
    ///
    /// Restart markers allow partial decoding and error recovery.
    /// Set to 0 to disable restart markers (default).
    #[must_use]
    pub fn restart_interval(mut self, interval: u16) -> Self {
        self.restart_interval = interval;
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
    /// When parallel encoding is enabled:
    /// - If `restart_interval` is 0 or too small, it will be **increased** to an
    ///   optimal value based on thread count and image size
    /// - User-specified `restart_interval` values are respected as a minimum
    ///   (the encoder may increase but will not decrease them)
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
    /// Allows fine-tuning all hybrid AQ+trellis parameters.
    /// See [`HybridConfig`](crate::hybrid::config::HybridConfig) for available options.
    #[must_use]
    pub fn hybrid_config(mut self, config: crate::hybrid::config::HybridConfig) -> Self {
        self.hybrid_config = config;
        self
    }

    // === Trellis Quantization ===

    /// Configure trellis quantization (mozjpeg-compatible API).
    ///
    /// Trellis quantization uses rate-distortion optimization to find the best
    /// quantization decisions, typically producing 10-15% smaller files at the
    /// same visual quality.
    ///
    /// This uses the same algorithm as mozjpeg and provides a compatible API.
    /// For advanced users who want to combine trellis with jpegli's adaptive
    /// quantization, see the `hybrid_config()` method.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use zenjpeg::encode::{EncoderConfig, ChromaSubsampling, TrellisConfig};
    ///
    /// // Enable trellis with default settings
    /// let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
    ///     .trellis(TrellisConfig::default());
    ///
    /// // Fine-tune trellis parameters
    /// let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
    ///     .trellis(TrellisConfig::default()
    ///         .ac_trellis(true)
    ///         .dc_trellis(true)
    ///         .speed_mode(TrellisSpeedMode::Level(5))
    ///         .rd_factor(0.8));
    ///
    /// // Disable trellis for fastest encoding
    /// let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
    ///     .trellis(TrellisConfig::disabled());
    /// ```
    #[must_use]
    pub fn trellis(mut self, config: TrellisConfig) -> Self {
        self.trellis = Some(config);
        self
    }

    /// Get the trellis configuration, if set.
    #[must_use]
    pub fn get_trellis(&self) -> Option<&TrellisConfig> {
        self.trellis.as_ref()
    }

    // === ICC Profile ===

    /// Attach an ICC color profile to the output JPEG.
    ///
    /// The profile will be written as APP2 marker segments with the standard
    /// "ICC_PROFILE" signature. Large profiles are automatically chunked
    /// (max 65519 bytes per segment) as required by the ICC profile embedding spec.
    ///
    /// Common profiles:
    /// - sRGB IEC61966-2.1 (~3KB)
    /// - Display P3 (~0.5KB)
    /// - Adobe RGB 1998 (~0.5KB)
    ///
    /// # Example
    /// ```ignore
    /// use zenjpeg::{EncoderConfig, ChromaSubsampling};
    /// let srgb_profile = std::fs::read("sRGB.icc")?;
    /// let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter)
    ///     .icc_profile(srgb_profile);
    /// ```
    #[must_use]
    pub fn icc_profile(mut self, profile: impl Into<Vec<u8>>) -> Self {
        self.icc_profile = Some(profile.into());
        self
    }

    // === EXIF/XMP Metadata ===

    /// Attach EXIF metadata to the output JPEG.
    ///
    /// Use [`Exif::raw`][super::exif::Exif::raw] for raw EXIF bytes, or
    /// [`Exif::build`][super::exif::Exif::build] to construct from common fields.
    ///
    /// The two modes are mutually exclusive at compile time - you cannot
    /// mix raw bytes with field-based building.
    ///
    /// # Examples
    ///
    /// Build from fields (orientation and copyright):
    /// ```ignore
    /// use zenjpeg::encoder::{EncoderConfig, ChromaSubsampling, Exif, Orientation};
    ///
    /// let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
    ///     .exif(Exif::build()
    ///         .orientation(Orientation::Rotate90)
    ///         .copyright("© 2024 Example Corp"));
    /// ```
    ///
    /// Use raw EXIF bytes:
    /// ```ignore
    /// use zenjpeg::encoder::{EncoderConfig, ChromaSubsampling, Exif};
    ///
    /// let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
    ///     .exif(Exif::raw(my_exif_bytes));
    /// ```
    ///
    /// # Notes
    ///
    /// - EXIF is placed immediately after SOI, before any other markers
    /// - Raw bytes should be TIFF data without the "Exif\0\0" prefix (added automatically)
    /// - Maximum size: 65527 bytes (larger data will be truncated)
    #[must_use]
    pub fn exif(mut self, exif: impl Into<super::exif::Exif>) -> Self {
        self.exif_data = Some(exif.into());
        self
    }

    /// Attach XMP metadata to the output JPEG.
    ///
    /// The data will be written as an APP1 marker segment with the standard
    /// Adobe XMP namespace signature. The provided bytes should be the raw XMP
    /// XML data without the APP1 marker or namespace prefix.
    ///
    /// XMP is placed after EXIF (if present) but before ICC profile.
    ///
    /// # Maximum Size
    /// Standard XMP is limited to 65502 bytes (65535 - 2 length - 29 namespace - 2 padding).
    /// For larger XMP data, use Extended XMP (not yet supported).
    #[must_use]
    pub fn xmp(mut self, data: impl Into<Vec<u8>>) -> Self {
        self.xmp_data = Some(data.into());
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
    pub fn tables(mut self, tables: Box<EncodingTables>) -> Self {
        self.tables = Some(tables);
        self
    }

    /// Sets custom Huffman tables for single-pass encoding.
    ///
    /// Blocks are entropy-encoded immediately using these tables instead of
    /// buffering for a two-pass Huffman optimization.
    ///
    /// Use [`HuffmanTableSet::annex_k()`] for the original JPEG standard tables,
    /// or provide tables from a previous encoding pass or external source.
    #[must_use]
    pub fn custom_huffman_tables(
        mut self,
        tables: crate::huffman::optimize::HuffmanTableSet,
    ) -> Self {
        self.huffman = HuffmanStrategy::Custom(Box::new(tables));
        self
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

    // === Validation ===

    /// Validate the configuration, returning an error for invalid combinations.
    ///
    /// Invalid combinations:
    /// - Progressive mode with disabled Huffman optimization
    pub fn validate(&self) -> Result<()> {
        if self.progressive && !matches!(self.huffman, HuffmanStrategy::Optimize) {
            return Err(crate::error::Error::invalid_config(
                "progressive mode requires optimized Huffman tables".into(),
            ));
        }
        Ok(())
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
        BytesEncoder::new(self.clone(), width, height, layout)
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
        RgbEncoder::new(self.clone(), width, height)
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

        YCbCrPlanarEncoder::new(self.clone(), width, height)
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
        self.progressive
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

    /// Check if separate chroma tables are enabled (3 tables vs 2).
    #[must_use]
    pub fn is_separate_chroma_tables(&self) -> bool {
        self.separate_chroma_tables
    }

    /// Get the ICC profile, if set.
    #[must_use]
    pub fn get_icc_profile(&self) -> Option<&[u8]> {
        self.icc_profile.as_deref()
    }

    /// Get the EXIF data, if set.
    #[must_use]
    pub fn get_exif(&self) -> Option<&super::exif::Exif> {
        self.exif_data.as_ref()
    }

    /// Get the XMP data, if set.
    #[must_use]
    pub fn get_xmp(&self) -> Option<&[u8]> {
        self.xmp_data.as_deref()
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
    use crate::encode::mozjpeg_compat::TrellisSpeedMode;

    #[test]
    fn test_ycbcr_config() {
        let config = EncoderConfig::ycbcr(90.0, ChromaSubsampling::None);
        assert!(matches!(config.quality, Quality::ApproxJpegli(90.0)));
        assert!(config.progressive); // Progressive is now the default
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
        assert!(config.progressive);
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
        config.progressive = true;
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

    #[test]
    fn test_trellis_disabled() {
        let config =
            EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter).trellis(TrellisConfig::disabled());

        let trellis = config.get_trellis().unwrap();
        assert!(!trellis.is_enabled());
        assert!(!trellis.is_ac_enabled());
        assert!(!trellis.is_dc_enabled());
    }
}
