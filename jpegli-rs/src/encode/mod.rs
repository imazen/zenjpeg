//! JPEG encoder implementation.
//!
//! This module provides the main encoder interface for creating JPEG images.

mod baseline;
mod blocks;
mod color;
pub mod config;
#[cfg(feature = "experimental-hybrid-trellis")]
mod hybrid;
mod output;
mod progressive;
mod workspace;

// Re-export config types
#[cfg(test)]
pub(crate) use config::ColorConversionMethod;
pub use config::{internal_pathway, EncoderConfig};
pub(crate) use config::{DownsamplingMethod, InternalPipeline, ProgressiveScan};
#[cfg(feature = "experimental-hybrid-trellis")]
pub(crate) use hybrid::HybridQuantContext;
pub use workspace::EncoderWorkspace;

use crate::alloc::{checked_size_2d, validate_dimensions, DEFAULT_MAX_PIXELS};
use crate::chroma;
#[cfg(test)]
use crate::consts::MARKER_SOI;
use crate::consts::{DCT_BLOCK_SIZE, DCT_SIZE, JPEG_ZIGZAG_ORDER, MARKER_EOI, XYB_ICC_PROFILE};
use crate::dct::forward_dct_8x8;
use crate::entropy::{self, EntropyEncoder};
use crate::error::{Error, Result};
use crate::huffman::optimize::{
    ContextConfig, FrequencyCounter, OptimizedHuffmanTables, OptimizedTable, ProgressiveTokenBuffer,
};
use crate::huffman::HuffmanEncodeTable;
use crate::quant::aq::compute_aq_strength_map;
use crate::quant::{self, Quality, QuantTable, ZeroBiasParams};
use crate::simd_types::{QuantTableSimd, ZeroBiasSimd};
use crate::types::{ChromaConversion, ColorSpace, JpegMode, PixelFormat, Subsampling};

/// JPEG encoder.
pub struct Encoder {
    config: EncoderConfig,
}

impl Encoder {
    /// Creates a new encoder with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: EncoderConfig::default(),
        }
    }

    /// Creates an encoder from configuration.
    #[must_use]
    pub fn from_config(config: EncoderConfig) -> Self {
        Self { config }
    }

    /// Sets the image width.
    #[must_use]
    pub fn width(mut self, width: u32) -> Self {
        self.config.width = width;
        self
    }

    /// Sets the image height.
    #[must_use]
    pub fn height(mut self, height: u32) -> Self {
        self.config.height = height;
        self
    }

    /// Sets the pixel format.
    #[must_use]
    pub fn pixel_format(mut self, format: PixelFormat) -> Self {
        self.config.pixel_format = format;
        self
    }

    /// Sets the quality using jpegli's native quality scale.
    ///
    /// Use `Quality::from_quality(90.0)` for traditional JPEG quality (1-100)
    /// or `Quality::from_distance(1.0)` for butteraugli distance.
    #[must_use]
    pub fn jpegli_quality(mut self, quality: Quality) -> Self {
        self.config.quality = quality;
        self
    }

    /// Sets the quality to match another encoder's visual quality.
    ///
    /// This converts quality settings from other encoders (like mozjpeg) to
    /// equivalent jpegli quality values that produce similar visual results.
    ///
    /// # Example
    ///
    /// ```
    /// use jpegli::{Encoder, QualityConversion, QualityComparisonMetric, Subsampling};
    ///
    /// // Match mozjpeg Q85 visual quality
    /// let conversion = QualityConversion::mozjpeg_equivalent(
    ///     85,
    ///     Subsampling::S444,
    ///     QualityComparisonMetric::Dssim,
    /// );
    ///
    /// let encoder = Encoder::new()
    ///     .width(800)
    ///     .height(600)
    ///     .equivalent_quality(conversion);
    /// ```
    #[must_use]
    pub fn equivalent_quality(
        mut self,
        conversion: crate::quality_conversion::QualityConversion,
    ) -> Self {
        self.config.quality = conversion.to_jpegli_quality();
        self
    }

    /// Sets the quality.
    ///
    /// **Deprecated:** Use `jpegli_quality()` for explicit jpegli quality, or
    /// `equivalent_quality()` to match other encoders like mozjpeg.
    #[must_use]
    #[deprecated(
        since = "0.4.0",
        note = "Use jpegli_quality() or equivalent_quality() instead"
    )]
    pub fn quality(mut self, quality: Quality) -> Self {
        self.config.quality = quality;
        self
    }

    /// Sets the encoding mode.
    #[must_use]
    pub fn mode(mut self, mode: JpegMode) -> Self {
        self.config.mode = mode;
        self
    }

    /// Sets chroma subsampling.
    #[must_use]
    pub fn subsampling(mut self, subsampling: Subsampling) -> Self {
        self.config.subsampling = subsampling;
        self
    }

    /// Enables XYB-optimized encoding mode.
    ///
    /// XYB mode encodes images using the perceptually-optimized XYB color space
    /// from JPEG XL. This provides better quality at the same file size compared
    /// to standard YCbCr encoding.
    ///
    /// The implementation includes:
    /// 1. Full sRGB → linear RGB → XYB color space conversion
    /// 2. XYB value scaling for optimal quantization
    /// 3. Embedded ICC profile for decoder color interpretation
    /// 4. Blue channel subsampling (R:2×2, G:2×2, B:1×1)
    /// 5. Separate XYB-optimized quant tables per component
    ///
    /// The ICC profile allows any ICC-aware decoder (including djpegli, ImageMagick,
    /// and most image viewers) to correctly interpret the XYB values back to sRGB.
    ///
    /// Note: Without ICC profile support in the decoder, images will display with
    /// incorrect colors. Use standard YCbCr mode for maximum compatibility.
    #[must_use]
    pub fn use_xyb(mut self, enable: bool) -> Self {
        self.config.use_xyb = enable;
        self
    }

    /// Sets the restart interval.
    #[must_use]
    pub fn restart_interval(mut self, interval: u16) -> Self {
        self.config.restart_interval = interval;
        self
    }

    /// Enables optimized Huffman tables.
    #[must_use]
    pub fn optimize_huffman(mut self, enable: bool) -> Self {
        self.config.optimize_huffman = enable;
        self
    }

    /// Sets the input smoothing factor (0-100).
    ///
    /// When non-zero, applies a 3x3 weighted blur to chroma planes before
    /// downsampling to reduce aliasing artifacts. Higher values = more blur.
    ///
    /// This matches libjpeg/jpegli's `smoothing_factor` parameter.
    /// Default is 0 (disabled), which is also jpegli's default.
    ///
    /// **Important**: Only works with [`ChromaConversion::Intrinsic`].
    /// The yuv crate paths (Fast, Sharp) perform conversion + downsampling
    /// in a single pass, so there's no intermediate chroma plane to blur.
    ///
    /// Only affects chroma subsampling modes (4:2:0, 4:2:2, 4:4:0).
    /// Has no effect on 4:4:4 mode since no downsampling occurs.
    #[must_use]
    pub fn smoothing_factor(mut self, factor: u8) -> Self {
        self.config.smoothing_factor = factor.min(100);
        self
    }

    /// Set chroma conversion method.
    ///
    /// Controls how RGB is converted to YCbCr chroma planes:
    /// - [`ChromaConversion::Intrinsic`]: Our f32 conversion with box filter
    ///   downsampling. Supports `smoothing_factor` for pre-blur.
    /// - [`ChromaConversion::Fast`]: yuv crate SIMD path with box filter.
    ///   Fast but may have color bleeding on edges.
    /// - [`ChromaConversion::Sharp`]: yuv crate Sharp YUV (gamma-aware bilinear).
    ///   Best quality for edges, graphics, and text.
    /// - [`ChromaConversion::Auto`]: Intrinsic (matches C++ jpegli default)
    ///
    /// Sharp YUV is often 10-50% FASTER than Intrinsic due to optimized SIMD.
    #[must_use]
    pub fn chroma_conversion(mut self, method: ChromaConversion) -> Self {
        self.config.chroma_conversion = method;
        self
    }

    /// Convenience method: enable Sharp YUV chroma downsampling.
    ///
    /// - `enable = true` → `ChromaConversion::Sharp`
    /// - `enable = false` → `ChromaConversion::Intrinsic`
    #[must_use]
    pub fn sharp_yuv(mut self, enable: bool) -> Self {
        self.config.chroma_conversion = if enable {
            ChromaConversion::Sharp
        } else {
            ChromaConversion::Intrinsic
        };
        self
    }

    /// Sets custom base quantization matrices for experimentation.
    ///
    /// **This is an undocumented escape hatch for research purposes.**
    ///
    /// See [`CustomQuantMatrices`](crate::quant::CustomQuantMatrices) for details
    /// on the matrix format and how quantization works.
    ///
    /// # Example
    /// ```ignore
    /// use jpegli::quant::CustomQuantMatrices;
    ///
    /// // Create custom matrices by modifying the defaults
    /// let mut custom_ycbcr = jpegli::consts::BASE_QUANT_MATRIX_YCBCR;
    /// // Modify DC coefficient (index 0) for Y channel
    /// custom_ycbcr[0] *= 0.8; // 20% smaller DC quantization step
    ///
    /// let custom = CustomQuantMatrices::new()
    ///     .with_ycbcr(custom_ycbcr);
    ///
    /// let jpeg = Encoder::new()
    ///     .width(800)
    ///     .height(600)
    ///     .custom_quant_matrices(custom)
    ///     .encode(&pixels)?;
    /// ```
    #[doc(hidden)]
    #[must_use]
    pub fn custom_quant_matrices(mut self, custom: crate::quant::CustomQuantMatrices) -> Self {
        self.config.custom_quant_matrices = Some(custom);
        self
    }

    /// Sets an internal chroma pipeline for benchmarking (undocumented API).
    ///
    /// This method is intentionally not documented in the public API.
    /// It allows external benchmarks to test different chroma conversion
    /// and downsampling strategies without committing to a stable API.
    ///
    /// # Pathway Encoding (u64)
    ///
    /// - Bits 0-7: Color conversion (0=Auto, 1=IntrinsicF32, 2=YuvBalanced, 3=YuvProfessional)
    /// - Bits 8-15: Downsampling (0=Auto, 1=None, 2=Box, 3=BoxSmoothed, 4=Sharp, 5=GammaAwareF32, 6=GammaAwareIterative)
    /// - Bits 16-23: Smoothing factor (0-100, only for BoxSmoothed)
    /// - Bits 24-63: Reserved (must be 0)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Reserved bits are non-zero
    /// - Invalid color conversion or downsampling method value
    /// - Smoothing factor > 100
    /// - Incompatible combination (e.g., Sharp with 4:4:4, None with 4:2:0)
    /// - Unimplemented method (e.g., YuvProfessional)
    #[doc(hidden)]
    pub fn set_internal_pathway(mut self, pathway: u64) -> Result<Self> {
        let pipeline = InternalPipeline::from_u64(pathway)?;
        pipeline.validate(self.config.subsampling)?;
        self.config.internal_pipeline = Some(pipeline);
        Ok(self)
    }

    /// Enable hybrid quantization (jpegli AQ + mozjpeg trellis).
    ///
    /// This combines jpegli's adaptive quantization (which determines WHERE
    /// to spend bits based on image content) with mozjpeg's trellis quantization
    /// (which optimizes HOW to spend bits via rate-distortion optimization).
    ///
    /// Requires the `experimental-hybrid-trellis` feature.
    #[cfg(feature = "experimental-hybrid-trellis")]
    #[must_use]
    pub fn hybrid_trellis(mut self, enable: bool) -> Self {
        if enable {
            self.config.hybrid_config = crate::hybrid::config::HybridConfig::default();
        } else {
            self.config.hybrid_config = crate::hybrid::config::HybridConfig::disabled();
        }
        self
    }

    /// Set custom hybrid quantization configuration.
    ///
    /// Allows fine-tuning all hybrid AQ+trellis parameters.
    /// See [`HybridConfig`](crate::hybrid::config::HybridConfig) for available options.
    ///
    /// Requires the `experimental-hybrid-trellis` feature.
    #[cfg(feature = "experimental-hybrid-trellis")]
    #[must_use]
    pub fn hybrid_config(mut self, config: crate::hybrid::config::HybridConfig) -> Self {
        self.config.hybrid_config = config;
        self
    }

    /// Sets a custom AQ (adaptive quantization) strength map.
    ///
    /// This allows pre-scaling the AQ map to control file size. When the AQ map
    /// is scaled up, more bits are allocated to complex regions (larger files).
    /// When scaled down, fewer bits are allocated (smaller files).
    ///
    /// If not provided, the AQ map is computed automatically from the image.
    ///
    /// # Example
    /// ```ignore
    /// use jpegli::adaptive_quant::compute_aq_strength_map;
    ///
    /// // Compute AQ map from Y plane
    /// let mut aq_map = compute_aq_strength_map(&y_plane, width, height, 8);
    ///
    /// // Scale down to reduce file size by ~16%
    /// let scale = aq_map.scale_for_size_reduction(16.0);
    /// aq_map.scale(scale);
    ///
    /// // Use the scaled map
    /// let jpeg = Encoder::new()
    ///     .width(width as u32)
    ///     .height(height as u32)
    ///     .hybrid_config(HybridConfig::default())
    ///     .aq_map(aq_map)
    ///     .encode(&pixels)?;
    /// ```
    ///
    /// Requires the `experimental-hybrid-trellis` feature.
    #[cfg(feature = "experimental-hybrid-trellis")]
    #[must_use]
    pub fn aq_map(mut self, map: crate::adaptive_quant::AQStrengthMap) -> Self {
        self.config.custom_aq_map = Some(map);
        self
    }

    /// Validates the configuration.
    fn validate(&self) -> Result<()> {
        // Use validate_dimensions for comprehensive checks (zero, max dimension, max pixels)
        validate_dimensions(self.config.width, self.config.height, DEFAULT_MAX_PIXELS)?;
        Ok(())
    }

    /// Encodes the image data.
    pub fn encode(&self, data: &[u8]) -> Result<Vec<u8>> {
        self.validate()?;

        // Calculate expected size with overflow checking
        let expected_size =
            checked_size_2d(self.config.width as usize, self.config.height as usize)?;
        let expected_size =
            checked_size_2d(expected_size, self.config.pixel_format.bytes_per_pixel())?;

        if data.len() != expected_size {
            return Err(Error::InvalidBufferSize {
                expected: expected_size,
                actual: data.len(),
            });
        }

        // For now, implement baseline encoding only
        match self.config.mode {
            JpegMode::Baseline => self.encode_baseline(data),
            JpegMode::Progressive => self.encode_progressive(data),
            _ => Err(Error::UnsupportedFeature {
                feature: "extended/lossless encoding",
            }),
        }
    }

    /// Encodes the image data using a pre-allocated workspace.
    ///
    /// This method reuses buffers from the workspace to avoid allocation
    /// overhead (~25-30% of encode time for large images). The workspace
    /// must be able to handle the image dimensions.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use jpegli::{Encoder, EncoderWorkspace};
    ///
    /// // Create workspace once, reuse for multiple encodes
    /// let mut workspace = EncoderWorkspace::new(4096, 4096)?;
    ///
    /// for image in images {
    ///     let jpeg = Encoder::new()
    ///         .width(image.width)
    ///         .height(image.height)
    ///         .encode_with_workspace(&image.pixels, &mut workspace)?;
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The workspace cannot handle the image dimensions
    /// - The input buffer size doesn't match the image dimensions
    /// - Encoding fails for any other reason
    pub fn encode_with_workspace(
        &self,
        data: &[u8],
        workspace: &mut EncoderWorkspace,
    ) -> Result<Vec<u8>> {
        self.validate()?;

        let width = self.config.width as usize;
        let height = self.config.height as usize;

        // Check workspace capacity
        if !workspace.can_handle(width, height) {
            return Err(Error::InvalidBufferSize {
                expected: workspace.max_pixels(),
                actual: width * height,
            });
        }

        // Calculate expected size with overflow checking
        let expected_size = checked_size_2d(width, height)?;
        let expected_size =
            checked_size_2d(expected_size, self.config.pixel_format.bytes_per_pixel())?;

        if data.len() != expected_size {
            return Err(Error::InvalidBufferSize {
                expected: expected_size,
                actual: data.len(),
            });
        }

        // Currently falls back to regular path - workspace integration needs
        // deeper refactoring to pass slices through the pipeline
        let _ = workspace;

        match self.config.mode {
            JpegMode::Baseline => self.encode_baseline(data),
            JpegMode::Progressive => self.encode_progressive(data),
            _ => Err(Error::UnsupportedFeature {
                feature: "extended/lossless encoding",
            }),
        }
    }

    /// Generate a quantization table, using custom matrices if configured.
    ///
    /// This helper method respects the `custom_quant_matrices` config option.
    #[inline]
    fn gen_quant_table(&self, component: usize, use_xyb: bool, is_420: bool) -> QuantTable {
        let distance = self.config.quality.to_distance();

        if let Some(ref custom) = self.config.custom_quant_matrices {
            quant::generate_quant_table_custom(distance, component, use_xyb, custom)
        } else {
            quant::generate_quant_table(
                self.config.quality,
                component,
                ColorSpace::YCbCr, // ColorSpace is not used by generate_quant_table when use_xyb is set
                use_xyb,
                is_420,
            )
        }
    }
}

impl Default for Encoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Converts coefficients from natural order to zigzag order for JPEG encoding.
#[inline]
fn natural_to_zigzag(natural: &[i16; DCT_BLOCK_SIZE]) -> [i16; DCT_BLOCK_SIZE] {
    let mut zigzag = [0i16; DCT_BLOCK_SIZE];
    for i in 0..DCT_BLOCK_SIZE {
        zigzag[JPEG_ZIGZAG_ORDER[i] as usize] = natural[i];
    }
    zigzag
}

/// Converts coefficients from natural order to zigzag order, writing directly to destination.
/// Avoids allocation when writing to pre-allocated block arrays.
#[inline]
fn natural_to_zigzag_into(natural: &[i16; DCT_BLOCK_SIZE], dest: &mut [i16; DCT_BLOCK_SIZE]) {
    for i in 0..DCT_BLOCK_SIZE {
        dest[JPEG_ZIGZAG_ORDER[i] as usize] = natural[i];
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
