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
pub mod strip;

// Re-export config types
pub use config::EncoderConfig;
pub(crate) use config::ProgressiveScan;

use crate::alloc::{checked_size_2d, try_with_capacity, validate_dimensions, DEFAULT_MAX_PIXELS};
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
use crate::types::{ChromaDownsampling, ColorSpace, JpegMode, PixelFormat, Subsampling};

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

    /// Set chroma downsampling method for subsampled modes.
    ///
    /// Controls how chroma planes are downsampled:
    /// - [`ChromaDownsampling::Box`]: Simple box filter (default, matches C++ jpegli)
    /// - [`ChromaDownsampling::GammaAware`]: Gamma-aware averaging (better edges)
    /// - [`ChromaDownsampling::GammaAwareIterative`]: Sharp YUV-style optimization (best quality)
    ///
    /// Has no effect for 4:4:4 subsampling (no downsampling needed).
    #[must_use]
    pub fn chroma_downsampling(mut self, method: ChromaDownsampling) -> Self {
        self.config.chroma_downsampling = method;
        self
    }

    /// Convenience method: enable Sharp YUV-style chroma downsampling.
    ///
    /// - `enable = true` → `ChromaDownsampling::GammaAwareIterative`
    /// - `enable = false` → `ChromaDownsampling::Box`
    #[must_use]
    pub fn sharp_yuv(mut self, enable: bool) -> Self {
        self.config.chroma_downsampling = if enable {
            ChromaDownsampling::GammaAwareIterative
        } else {
            ChromaDownsampling::Box
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

    /// Force full-plane encoding even for large images (benchmarking only).
    ///
    /// When enabled, disables auto-dispatch to strip-based encoding for images > 2MP.
    /// This is useful for benchmarking to compare full-plane vs strip performance.
    #[doc(hidden)]
    pub fn force_full_plane(mut self, force: bool) -> Self {
        self.config.force_full_plane = force;
        self
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
    /// let mut aq_map = compute_aq_strength_map(&y_plane, width, height, 8)?;
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

    /// Encodes the image using strip-based processing for reduced memory usage.
    ///
    /// This method processes the image in horizontal strips (MCU rows) instead
    /// of materializing full f32 planes, reducing peak memory by ~5x for large
    /// images (e.g., 230 MB → 40 MB for 12MP).
    ///
    /// Supports YCbCr baseline and progressive encoding with optimized Huffman.
    /// XYB color space is not yet supported in strip mode.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use jpegli::Encoder;
    ///
    /// let jpeg = Encoder::new()
    ///     .width(4000)
    ///     .height(3000)
    ///     .encode_strip_based(&rgb_data)?;
    /// ```
    pub fn encode_strip_based(&self, data: &[u8]) -> Result<Vec<u8>> {
        self.validate()?;

        let width = self.config.width as usize;
        let height = self.config.height as usize;

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

        // Supports baseline and progressive modes
        if self.config.mode != JpegMode::Baseline && self.config.mode != JpegMode::Progressive {
            return Err(Error::UnsupportedFeature {
                feature: "strip-based encoding only supports baseline and progressive modes",
            });
        }

        // Only supports YCbCr color space (not XYB)
        if self.config.use_xyb {
            return Err(Error::UnsupportedFeature {
                feature: "strip-based encoding does not support XYB color space",
            });
        }

        // Create strip processor
        let mut processor = strip::StripProcessor::new(
            width,
            height,
            self.config.subsampling,
            self.config.pixel_format,
        )?;

        // Generate quantization tables
        let is_420 = self.config.subsampling == Subsampling::S420;
        let y_quant = self.gen_quant_table(0, false, is_420);
        let cb_quant = self.gen_quant_table(1, false, is_420);
        let cr_quant = self.gen_quant_table(2, false, is_420);

        // Compute zero bias params
        let effective_distance = quant::quant_vals_to_distance(&y_quant, &cb_quant, &cr_quant);
        let y_zero_bias = ZeroBiasParams::for_ycbcr(effective_distance, 0);
        let cb_zero_bias = ZeroBiasParams::for_ycbcr(effective_distance, 1);
        let cr_zero_bias = ZeroBiasParams::for_ycbcr(effective_distance, 2);

        processor.set_quant_tables(
            y_quant.clone(),
            cb_quant.clone(),
            cr_quant.clone(),
            y_zero_bias,
            cb_zero_bias,
            cr_zero_bias,
        )?;

        // Process all strips
        let strip_height = processor.strip_height();
        let bpp = self.config.pixel_format.bytes_per_pixel();
        for strip_y in (0..height).step_by(strip_height) {
            let strip_end = (strip_y + strip_height).min(height);
            let strip_start = strip_y * width * bpp;
            let strip_end_idx = strip_end * width * bpp;
            let rgb_strip = &data[strip_start..strip_end_idx];

            processor.process_strip(rgb_strip, strip_y)?;
        }

        // Finalize strip processing to get blocks
        let strip_output = processor.finalize()?;

        // Branch based on encoding mode
        match self.config.mode {
            JpegMode::Progressive => {
                // Use progressive encoding path
                self.encode_progressive_from_blocks(
                    &strip_output.y_blocks,
                    &strip_output.cb_blocks,
                    &strip_output.cr_blocks,
                    &y_quant,
                    &cb_quant,
                    &cr_quant,
                )
            }
            _ => {
                // Baseline encoding path
                let is_color = self.config.pixel_format != PixelFormat::Gray;

                // Build output JPEG
                let mut output = try_with_capacity(width * height / 4, "jpeg output")?; // Rough estimate

                // Write JPEG headers
                self.write_header(&mut output)?;
                self.write_quant_tables(&mut output, &y_quant, &cb_quant, &cr_quant)?;
                self.write_frame_header(&mut output)?;

                // Build optimized Huffman tables from blocks (uses MCU order)
                let tables = self.build_optimized_tables(
                    &strip_output.y_blocks,
                    &strip_output.cb_blocks,
                    &strip_output.cr_blocks,
                    is_color,
                )?;

                self.write_huffman_tables_optimized(&mut output, &tables)?;

                if self.config.restart_interval > 0 {
                    self.write_restart_interval(&mut output)?;
                }
                self.write_scan_header(&mut output)?;

                // Encode blocks with optimized tables
                let scan_data = self.encode_with_tables(
                    &strip_output.y_blocks,
                    &strip_output.cb_blocks,
                    &strip_output.cr_blocks,
                    is_color,
                    &tables,
                )?;

                output.extend_from_slice(&scan_data);

                // Write EOI
                output.push(0xFF);
                output.push(MARKER_EOI);

                Ok(output)
            }
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
