//! Encoder configuration for v2 API.

use super::types::{
    ChromaSubsampling, ColorMode, DownsamplingMethod, PixelLayout, Quality, QuantTableConfig,
    XybSubsampling,
};
use super::{BytesEncoder, RgbEncoder, YCbCrPlanarEncoder};
use crate::error::Result;

/// JPEG encoder configuration. Dimension-independent, reusable across images.
#[derive(Clone, Debug)]
pub struct EncoderConfig {
    pub(crate) quality: Quality,
    pub(crate) quant_tables: QuantTableConfig,
    pub(crate) progressive: bool,
    pub(crate) optimize_huffman: bool,
    pub(crate) color_mode: ColorMode,
    pub(crate) downsampling_method: DownsamplingMethod,
    pub(crate) restart_interval: u16,
    pub(crate) icc_profile: Option<Vec<u8>>,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            quality: Quality::default(),
            quant_tables: QuantTableConfig::default(),
            progressive: false,
            optimize_huffman: true,
            color_mode: ColorMode::default(),
            downsampling_method: DownsamplingMethod::default(),
            restart_interval: 0,
            icc_profile: None,
        }
    }
}

impl EncoderConfig {
    /// Create a new encoder configuration with default settings.
    ///
    /// Defaults:
    /// - Quality: 90 (ApproxJpegli)
    /// - Color mode: YCbCr 4:2:0
    /// - Optimized Huffman: enabled
    /// - Progressive: disabled
    /// - Downsampling: Box
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    // === Quality & Quantization ===

    /// Set the quality level.
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

    // === Convenience Shortcuts ===

    /// Set YCbCr color mode with specified chroma subsampling.
    ///
    /// Common values:
    /// - `ChromaSubsampling::Quarter` (4:2:0) - default, good compression
    /// - `ChromaSubsampling::Full` (4:4:4) - best quality, larger files
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

    // === Validation ===

    /// Validate the configuration, returning an error for invalid combinations.
    ///
    /// Invalid combinations:
    /// - Progressive mode with disabled Huffman optimization
    pub fn validate(&self) -> Result<()> {
        if self.progressive && !self.optimize_huffman {
            return Err(crate::error::Error::InvalidConfig(
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
    /// enc.push_packed(&rgb_bytes, Never)?;
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
    /// enc.push_packed(&pixels, Never)?;
    /// let jpeg = enc.finish()?;
    /// ```
    pub fn encode_from_rgb<P: super::Pixel>(&self, width: u32, height: u32) -> Result<RgbEncoder<P>> {
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
    /// enc.push(&planes, height, Never)?;
    /// let jpeg = enc.finish()?;
    /// ```
    pub fn encode_from_ycbcr_planar(&self, width: u32, height: u32) -> Result<YCbCrPlanarEncoder> {
        self.validate()?;

        // Validate color mode
        if !matches!(self.color_mode, ColorMode::YCbCr { .. }) {
            return Err(crate::error::Error::InvalidConfig(
                "planar YCbCr input requires YCbCr color mode".into(),
            ));
        }

        YCbCrPlanarEncoder::new(self.clone(), width, height)
    }

    // === Resource Estimation ===

    /// Estimate peak memory usage for encoding an image of the given dimensions.
    ///
    /// This is a rough estimate based on:
    /// - Strip buffer size
    /// - Block storage
    /// - Huffman tables
    /// - Output buffer
    #[must_use]
    pub fn estimate_memory(&self, width: u32, height: u32) -> usize {
        let w = width as usize;
        let h = height as usize;

        // Strip height (16 rows for 4:2:0)
        let strip_height = 16;

        // Strip buffers (YCbCr f32 planes)
        let strip_buffer = w * strip_height * 4 * 3;

        // Block storage (worst case: all blocks in memory for progressive)
        let blocks_y = ((w + 7) / 8) * ((h + 7) / 8) * 64 * 2; // i16
        let blocks_c = blocks_y / 4; // for 4:2:0
        let blocks = blocks_y + blocks_c * 2;

        // Huffman tables and other overhead
        let overhead = 64 * 1024;

        // Output buffer estimate (quality-dependent, rough)
        let output_estimate = w * h / 4;

        strip_buffer + blocks + overhead + output_estimate
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = EncoderConfig::new();
        assert!(matches!(config.quality, Quality::ApproxJpegli(90.0)));
        assert!(!config.progressive);
        assert!(config.optimize_huffman);
        assert!(matches!(
            config.color_mode,
            ColorMode::YCbCr {
                subsampling: ChromaSubsampling::Quarter
            }
        ));
    }

    #[test]
    fn test_builder_pattern() {
        let config = EncoderConfig::new()
            .quality(85)
            .progressive(true)
            .ycbcr(ChromaSubsampling::Full)
            .sharp_yuv(true);

        assert!(matches!(config.quality, Quality::ApproxJpegli(85.0)));
        assert!(config.progressive);
        assert!(config.optimize_huffman); // auto-enabled by progressive
        assert!(matches!(
            config.color_mode,
            ColorMode::YCbCr {
                subsampling: ChromaSubsampling::Full
            }
        ));
        assert!(matches!(
            config.downsampling_method,
            DownsamplingMethod::GammaAwareIterative
        ));
    }

    #[test]
    fn test_progressive_enables_huffman() {
        let config = EncoderConfig::new().optimize_huffman(false).progressive(true);

        assert!(config.optimize_huffman);
    }

    #[test]
    fn test_validation_progressive_huffman() {
        let mut config = EncoderConfig::new();
        config.progressive = true;
        config.optimize_huffman = false;

        assert!(config.validate().is_err());
    }

    #[test]
    fn test_xyb_shortcuts() {
        let config = EncoderConfig::new().xyb();
        assert!(matches!(
            config.color_mode,
            ColorMode::Xyb {
                subsampling: XybSubsampling::BQuarter
            }
        ));

        let config = EncoderConfig::new().xyb_full();
        assert!(matches!(
            config.color_mode,
            ColorMode::Xyb {
                subsampling: XybSubsampling::Full
            }
        ));
    }
}
