//! Encoder configuration for v2 API.

use super::byte_encoders::{BytesEncoder, RgbEncoder, YCbCrPlanarEncoder};
use super::encoder_types::{
    ChromaSubsampling, ColorMode, DownsamplingMethod, PixelLayout, Quality, QuantTableConfig,
    XybSubsampling, ZeroBiasConfig,
};
use crate::error::Result;
use crate::types::EdgePaddingConfig;

/// JPEG encoder configuration. Dimension-independent, reusable across images.
#[derive(Clone, Debug)]
pub struct EncoderConfig {
    pub(crate) quality: Quality,
    pub(crate) quant_tables: QuantTableConfig,
    pub(crate) zero_bias: ZeroBiasConfig,
    pub(crate) progressive: bool,
    pub(crate) optimize_huffman: bool,
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
    /// Hybrid quantization configuration (requires `experimental-hybrid-trellis` feature)
    #[cfg(feature = "experimental-hybrid-trellis")]
    pub(crate) hybrid_config: crate::hybrid::config::HybridConfig,
    /// Enable overshoot deringing (requires `mozjpeg-deringing` feature)
    #[cfg(feature = "mozjpeg-deringing")]
    pub(crate) deringing: bool,
}

// Note: No Default impl - quality and subsampling are required via new()

impl EncoderConfig {
    /// Create a new encoder configuration with required quality and chroma subsampling.
    ///
    /// # Arguments
    /// - `quality`: Quality level (0-100 for jpegli scale, or use `Quality::*` variants)
    /// - `subsampling`: Chroma subsampling mode
    ///   - `ChromaSubsampling::Quarter` (4:2:0) - good compression, smaller files
    ///   - `ChromaSubsampling::None` (4:4:4) - best quality, larger files
    ///
    /// # Example
    /// ```ignore
    /// use jpegli::encoder::{EncoderConfig, ChromaSubsampling};
    ///
    /// let config = EncoderConfig::new(85.0, ChromaSubsampling::Quarter)
    ///     .progressive(true);
    /// ```
    #[must_use]
    pub fn new(quality: impl Into<Quality>, subsampling: ChromaSubsampling) -> Self {
        Self {
            quality: quality.into(),
            color_mode: ColorMode::YCbCr { subsampling },
            ..Self::default_internal()
        }
    }

    /// Internal default for non-required fields only.
    fn default_internal() -> Self {
        Self {
            quality: Quality::default(),
            quant_tables: QuantTableConfig::default(),
            zero_bias: ZeroBiasConfig::default(),
            progressive: false,
            optimize_huffman: true,
            color_mode: ColorMode::default(),
            downsampling_method: DownsamplingMethod::default(),
            restart_interval: 0,
            icc_profile: None,
            exif_data: None,
            xmp_data: None,
            edge_padding: EdgePaddingConfig::default(),
            #[cfg(feature = "parallel")]
            parallel: None,
            #[cfg(feature = "experimental-hybrid-trellis")]
            hybrid_config: crate::hybrid::config::HybridConfig::default(),
            #[cfg(feature = "mozjpeg-deringing")]
            deringing: false,
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

    /// Set custom quantization tables.
    #[must_use]
    pub fn quant_tables(mut self, config: QuantTableConfig) -> Self {
        self.quant_tables = config;
        self
    }

    /// Set zero-bias configuration.
    ///
    /// Zero-bias controls how DCT coefficients are rounded toward zero during
    /// quantization. The default mode auto-selects between YCbCr and XYB tables
    /// based on the color mode.
    ///
    /// # Options
    ///
    /// - `ZeroBiasConfig::Default` (default) - auto-select based on color mode
    /// - `ZeroBiasConfig::YCbCr` - force YCbCr quality-adaptive tables
    /// - `ZeroBiasConfig::Xyb` - force XYB 0.5 tables
    /// - `ZeroBiasConfig::Disabled` - no zero-bias (standard JPEG behavior)
    /// - `ZeroBiasConfig::Custom { .. }` - provide custom per-component tables
    ///
    /// # Example
    ///
    /// ```ignore
    /// use jpegli::encoder::{EncoderConfig, ZeroBiasConfig};
    ///
    /// // Disable zero-bias for standard JPEG behavior
    /// let config = EncoderConfig::new(85, ChromaSubsampling::None)
    ///     .zero_bias(ZeroBiasConfig::Disabled);
    /// ```
    #[must_use]
    pub fn zero_bias(mut self, config: ZeroBiasConfig) -> Self {
        self.zero_bias = config;
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
            self.optimize_huffman = true;
        }
        self
    }

    /// Enable or disable Huffman table optimization.
    ///
    /// When enabled (default), computes optimal Huffman tables from image data.
    /// When disabled, uses standard JPEG Huffman tables (faster but larger files).
    ///
    /// Note: Progressive mode requires optimized Huffman tables.
    #[must_use]
    pub fn optimize_huffman(mut self, enable: bool) -> Self {
        self.optimize_huffman = enable;
        self
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
    /// use jpegli::{EncoderConfig, ParallelEncoding};
    ///
    /// let config = EncoderConfig::new()
    ///     .quality(85)
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
    ///
    /// Requires the `experimental-hybrid-trellis` feature.
    #[cfg(feature = "experimental-hybrid-trellis")]
    #[must_use]
    pub fn hybrid_config(mut self, config: crate::hybrid::config::HybridConfig) -> Self {
        self.hybrid_config = config;
        self
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
    /// let srgb_profile = std::fs::read("sRGB.icc")?;
    /// let config = EncoderConfig::new()
    ///     .quality(85)
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
    /// use jpegli::encoder::{EncoderConfig, ChromaSubsampling, Exif, Orientation};
    ///
    /// let config = EncoderConfig::new(85, ChromaSubsampling::Quarter)
    ///     .exif(Exif::build()
    ///         .orientation(Orientation::Rotate90)
    ///         .copyright("© 2024 Example Corp"));
    /// ```
    ///
    /// Use raw EXIF bytes:
    /// ```ignore
    /// use jpegli::encoder::{EncoderConfig, ChromaSubsampling, Exif};
    ///
    /// let config = EncoderConfig::new(85, ChromaSubsampling::Quarter)
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

    // === Convenience Shortcuts ===

    /// Set YCbCr color mode with specified chroma subsampling.
    ///
    /// Common values:
    /// - `ChromaSubsampling::None` (4:4:4) - default, best quality
    /// - `ChromaSubsampling::Quarter` (4:2:0) - good compression, smaller files
    /// - `ChromaSubsampling::HalfHorizontal` (4:2:2) - horizontal subsampling only
    #[must_use]
    pub fn ycbcr(self, subsampling: ChromaSubsampling) -> Self {
        self.color_mode(ColorMode::YCbCr { subsampling })
    }

    /// Set XYB color mode with B-quarter subsampling (default, perceptually optimized).
    ///
    /// XYB is a perceptual color space that can achieve better quality at the same
    /// file size for some images. Requires linear RGB input (f32 or u16).
    #[must_use]
    pub fn xyb(self) -> Self {
        self.color_mode(ColorMode::Xyb {
            subsampling: XybSubsampling::BQuarter,
        })
    }

    /// Set XYB color mode with full resolution (no subsampling).
    #[must_use]
    pub fn xyb_full(self) -> Self {
        self.color_mode(ColorMode::Xyb {
            subsampling: XybSubsampling::Full,
        })
    }

    /// Set grayscale output mode.
    ///
    /// Only the luminance channel is encoded. Works with any input format.
    #[must_use]
    pub fn grayscale(self) -> Self {
        self.color_mode(ColorMode::Grayscale)
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

    /// Enable overshoot deringing to reduce ringing artifacts on white backgrounds.
    ///
    /// Deringing smooths hard edges (like text on white) by allowing values to
    /// "overshoot" beyond the maximum. Since JPEG decoders clamp values to 0-255,
    /// the overshoot is invisible but the smoother curve compresses better.
    ///
    /// This is particularly effective for:
    /// - Images with white backgrounds
    /// - Text and graphics with hard edges
    /// - Any image with saturated regions (pixels at 0 or 255)
    ///
    /// Requires the `mozjpeg-deringing` feature.
    #[cfg(feature = "mozjpeg-deringing")]
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
        if self.progressive && !self.optimize_huffman {
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
    /// let config = EncoderConfig::new().quality(85);
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
    ///
    /// let config = EncoderConfig::new().quality(85);
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
    /// let config = EncoderConfig::new()
    ///     .quality(85)
    ///     .ycbcr(ChromaSubsampling::Quarter);
    ///
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
    #[allow(deprecated)]
    pub fn estimate_memory(&self, width: u32, height: u32) -> usize {
        use crate::encode::streaming::StreamingEncoder;

        let subsampling = match self.color_mode {
            ColorMode::YCbCr { subsampling } => subsampling.to_legacy(),
            ColorMode::Xyb { .. } => crate::types::Subsampling::S444,
            ColorMode::Grayscale => crate::types::Subsampling::S444,
        };

        StreamingEncoder::new(width, height)
            .subsampling(subsampling)
            .optimize_huffman(self.optimize_huffman)
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
    /// use jpegli::encoder::EncoderConfig;
    ///
    /// let config = EncoderConfig::new().quality(85);
    /// let ceiling = config.estimate_memory_ceiling(1920, 1080);
    ///
    /// // Reserve this much memory - actual usage guaranteed to be less
    /// let buffer = Vec::with_capacity(ceiling);
    /// ```
    #[must_use]
    #[allow(deprecated)]
    pub fn estimate_memory_ceiling(&self, width: u32, height: u32) -> usize {
        use crate::encode::streaming::StreamingEncoder;

        let subsampling = match self.color_mode {
            ColorMode::YCbCr { subsampling } => subsampling.to_legacy(),
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
        self.optimize_huffman
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = EncoderConfig::new(90.0, ChromaSubsampling::None);
        assert!(matches!(config.quality, Quality::ApproxJpegli(90.0)));
        assert!(!config.progressive);
        assert!(config.optimize_huffman);
        assert!(matches!(
            config.color_mode,
            ColorMode::YCbCr {
                subsampling: ChromaSubsampling::None
            }
        ));
    }

    #[test]
    fn test_builder_pattern() {
        let config = EncoderConfig::new(85, ChromaSubsampling::None)
            .progressive(true)
            .sharp_yuv(true);

        assert!(matches!(config.quality, Quality::ApproxJpegli(85.0)));
        assert!(config.progressive);
        assert!(config.optimize_huffman); // auto-enabled by progressive
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
        let config = EncoderConfig::new(90.0, ChromaSubsampling::None)
            .optimize_huffman(false)
            .progressive(true);

        assert!(config.optimize_huffman);
    }

    #[test]
    fn test_validation_progressive_huffman() {
        let mut config = EncoderConfig::new(90.0, ChromaSubsampling::None);
        config.progressive = true;
        config.optimize_huffman = false;

        assert!(config.validate().is_err());
    }

    #[test]
    fn test_xyb_shortcuts() {
        let config = EncoderConfig::new(90.0, ChromaSubsampling::None).xyb();
        assert!(matches!(
            config.color_mode,
            ColorMode::Xyb {
                subsampling: XybSubsampling::BQuarter
            }
        ));

        let config = EncoderConfig::new(90.0, ChromaSubsampling::None).xyb_full();
        assert!(matches!(
            config.color_mode,
            ColorMode::Xyb {
                subsampling: XybSubsampling::Full
            }
        ));
    }
}
