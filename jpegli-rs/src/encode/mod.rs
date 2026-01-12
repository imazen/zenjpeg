//! JPEG encoder implementation.
//!
//! This module provides JPEG encoding capabilities with two APIs:
//!
//! ## Recommended: StreamingEncoder (new)
//!
//! The streaming encoder accepts rows incrementally, reducing peak memory
//! and providing better performance:
//!
//! ```rust,ignore
//! use jpegli::{StreamingEncoder, Quality, Subsampling};
//!
//! // Simple all-at-once encoding
//! let jpeg = StreamingEncoder::new(640, 480)
//!     .quality(Quality::from_quality(85.0))
//!     .subsampling(Subsampling::S420)
//!     .encode_all(&pixels)?;
//!
//! // Or stream rows for large images
//! let mut enc = StreamingEncoder::new(640, 480)
//!     .quality(Quality::from_quality(85.0))
//!     .build()?;
//! for row in image_rows {
//!     enc.push_row(row)?;
//! }
//! let jpeg = enc.finish()?;
//! ```
//!
//! ## Legacy: Encoder (deprecated)
//!
//! The original encoder API is still available but deprecated. It requires
//! the full image in memory and has higher peak memory usage.
//!
//! ```rust,ignore
//! use jpegli::Encoder;
//!
//! #[allow(deprecated)]
//! let jpeg = Encoder::new()
//!     .width(640)
//!     .height(480)
//!     .encode(&pixels)?;
//! ```

// Legacy encoder implementation (deprecated, kept for backwards compatibility)
mod old;

pub mod config;
#[cfg(feature = "experimental-hybrid-trellis")]
mod hybrid;
pub(crate) mod linear_lut;
#[cfg(feature = "parallel")]
pub mod parallel;
pub mod streaming;
pub mod strip;

// Re-export config types
pub use config::EncoderConfig;
pub(crate) use config::ProgressiveScan;

use crate::alloc::{
    checked_size_2d, try_alloc_zeroed_f32, try_clone_slice, try_with_capacity, validate_dimensions,
    DEFAULT_MAX_PIXELS,
};
use crate::consts::{DCT_BLOCK_SIZE, JPEG_ZIGZAG_ORDER, MARKER_EOI, XYB_ICC_PROFILE};
use crate::error::{Error, Result};
use crate::quant::{self, Quality, QuantTable, ZeroBiasParams};
use crate::types::{
    ChromaDownsampling, ColorSpace, EdgePadding, EdgePaddingConfig, JpegMode, PixelFormat,
    Subsampling,
};
use enough::{Never, Stop};

/// JPEG encoder.
///
/// **Deprecated:** Use [`StreamingEncoder`] instead, which provides better
/// performance and lower memory usage. The streaming API is now the
/// recommended way to encode JPEG images.
///
/// # Migration
///
/// ```rust,ignore
/// // Old API (deprecated):
/// #[allow(deprecated)]
/// let jpeg = Encoder::new()
///     .width(640)
///     .height(480)
///     .encode(&pixels)?;
///
/// // New API (recommended):
/// let jpeg = StreamingEncoder::new(640, 480)
///     .encode_all(&pixels)?;
/// ```
#[deprecated(
    since = "0.4.0",
    note = "Use StreamingEncoder instead, which provides better performance and lower memory usage"
)]
pub struct Encoder {
    /// Encoder configuration (accessible within crate for streaming encoder).
    pub(crate) config: EncoderConfig,
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

    // encoding_backend method removed - strip-based encoding is now the only backend

    /// Sets the edge padding strategy for partial MCU blocks.
    ///
    /// When image dimensions are not multiples of the MCU size (8 or 16 pixels),
    /// the encoder must pad edge blocks. This setting controls how that padding
    /// is performed, with separate strategies for luma and chroma channels.
    ///
    /// # Presets
    ///
    /// - [`EdgePaddingConfig::cpp_compat()`]: Match C++ jpegli behavior (Replicate all)
    /// - [`EdgePaddingConfig::recommended()`]: Mirror for luma, Replicate for chroma
    /// - [`EdgePaddingConfig::uniform(strategy)`]: Same strategy for all channels
    ///
    /// # Example
    ///
    /// ```
    /// use jpegli::{Encoder, EdgePaddingConfig, EdgePadding};
    ///
    /// // Match C++ jpegli behavior
    /// let encoder = Encoder::new()
    ///     .edge_padding(EdgePaddingConfig::cpp_compat());
    ///
    /// // Use recommended settings (better gradients, safe chroma)
    /// let encoder = Encoder::new()
    ///     .edge_padding(EdgePaddingConfig::recommended());
    ///
    /// // Custom per-channel configuration
    /// let encoder = Encoder::new()
    ///     .edge_padding(EdgePaddingConfig {
    ///         luma: EdgePadding::Mirror,
    ///         chroma: EdgePadding::Replicate,
    ///     });
    /// ```
    #[must_use]
    pub fn edge_padding(mut self, config: EdgePaddingConfig) -> Self {
        self.config.edge_padding = config;
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
    ///
    /// This is equivalent to calling `encode_with_stop(data, Never)`.
    pub fn encode(&self, data: &[u8]) -> Result<Vec<u8>> {
        self.encode_with_stop(data, Never)
    }

    /// Encodes the image data with cooperative cancellation support.
    ///
    /// The encoding can be cancelled at MCU row boundaries by signalling the `stop` source.
    /// Returns `Error::Cancelled` if cancellation is requested.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use jpegli::{Encoder, Stopper};
    /// use std::time::Duration;
    ///
    /// let stop = Stopper::new();
    /// let timed = stop.clone().with_timeout(Duration::from_secs(30));
    ///
    /// // In another thread: stop.cancel();
    /// let result = encoder.encode_with_stop(&data, timed);
    /// ```
    pub fn encode_with_stop(&self, data: &[u8], stop: impl Stop) -> Result<Vec<u8>> {
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

        // Validate mode is supported
        if self.config.mode != JpegMode::Baseline && self.config.mode != JpegMode::Progressive {
            return Err(Error::UnsupportedFeature {
                feature: "only baseline and progressive modes are supported",
            });
        }

        // Both YCbCr and XYB use strip-based encoding (low memory)
        self.encode_strip_based_with_stop(data, stop)
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
        self.encode_strip_based_with_stop(data, Never)
    }

    /// Encodes using strip-based processing with cancellation support.
    ///
    /// The encoding can be cancelled at strip boundaries by signalling the `stop` source.
    fn encode_strip_based_with_stop(&self, data: &[u8], stop: impl Stop) -> Result<Vec<u8>> {
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

        // Create strip processor with chroma downsampling, restart interval, and XYB mode
        let mut processor = strip::StripProcessor::with_xyb(
            width,
            height,
            self.config.subsampling,
            self.config.pixel_format,
            self.config.chroma_downsampling,
            self.config.restart_interval,
            self.config.use_xyb,
        )?;

        // Generate quantization tables
        let is_420 = self.config.subsampling == Subsampling::S420;
        let y_quant = self.gen_quant_table(0, self.config.use_xyb, is_420);
        let cb_quant = self.gen_quant_table(1, self.config.use_xyb, is_420);
        let cr_quant = self.gen_quant_table(2, self.config.use_xyb, is_420);

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
            // Check for cancellation at each strip boundary
            stop.check()?;

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
                // Progressive mode requires optimized Huffman tables
                if !self.config.optimize_huffman {
                    return Err(Error::UnsupportedFeature {
                        feature:
                            "Progressive mode with fixed Huffman codes (use optimize_huffman=true)",
                    });
                }
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

                // Branch based on XYB vs YCbCr mode
                let scan_data = if self.config.use_xyb {
                    // XYB mode: uses different headers, tables, and encoding
                    // strip_output contains: y_blocks = X, cb_blocks = Y, cr_blocks = B (2x2 downsampled)
                    self.write_header_xyb(&mut output)?;
                    // Write APP14 Adobe marker for RGB colorspace (required by decoders)
                    self.write_app14_adobe(&mut output, 0)?; // 0 = RGB (no transform)
                                                             // Write XYB ICC profile so decoders can interpret the colors correctly
                    self.write_icc_profile(&mut output, &XYB_ICC_PROFILE)?;
                    self.write_quant_tables_xyb(&mut output, &y_quant, &cb_quant, &cr_quant)?;
                    self.write_frame_header_xyb(&mut output)?;

                    if self.config.optimize_huffman {
                        // Use raster-ordered XYB encoding (strip encoder produces raster order)
                        let (dc_table, ac_table) = self.build_optimized_tables_xyb_raster(
                            &strip_output.y_blocks,  // X component
                            &strip_output.cb_blocks, // Y component
                            &strip_output.cr_blocks, // B component (2x2 downsampled)
                        )?;

                        self.write_huffman_tables_xyb_optimized(&mut output, &dc_table, &ac_table);

                        if self.config.restart_interval > 0 {
                            self.write_restart_interval(&mut output)?;
                        }
                        self.write_scan_header_xyb(&mut output)?;

                        self.encode_with_tables_xyb_raster(
                            &strip_output.y_blocks,
                            &strip_output.cb_blocks,
                            &strip_output.cr_blocks,
                            &dc_table,
                            &ac_table,
                        )?
                    } else {
                        self.write_huffman_tables(&mut output)?;

                        if self.config.restart_interval > 0 {
                            self.write_restart_interval(&mut output)?;
                        }
                        self.write_scan_header_xyb(&mut output)?;

                        // XYB without optimized tables - use raster-ordered standard encoding
                        self.encode_with_tables_xyb_standard_raster(
                            &strip_output.y_blocks,
                            &strip_output.cb_blocks,
                            &strip_output.cr_blocks,
                        )?
                    }
                } else {
                    // YCbCr mode: standard JPEG encoding
                    self.write_header(&mut output)?;
                    self.write_quant_tables(&mut output, &y_quant, &cb_quant, &cr_quant)?;
                    self.write_frame_header(&mut output)?;

                    // Respect optimize_huffman config (must match full-plane encoder behavior)
                    if self.config.optimize_huffman {
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
                        self.encode_with_tables(
                            &strip_output.y_blocks,
                            &strip_output.cb_blocks,
                            &strip_output.cr_blocks,
                            is_color,
                            Some(&tables),
                        )?
                    } else {
                        // Use standard (fixed) Huffman tables
                        self.write_huffman_tables(&mut output)?;

                        if self.config.restart_interval > 0 {
                            self.write_restart_interval(&mut output)?;
                        }
                        self.write_scan_header(&mut output)?;

                        // Encode blocks with standard tables
                        self.encode_with_tables(
                            &strip_output.y_blocks,
                            &strip_output.cb_blocks,
                            &strip_output.cr_blocks,
                            is_color,
                            None,
                        )?
                    }
                };

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

// ============================================================================
// Edge Padding Helpers
// ============================================================================

/// Compute the source coordinate for a padded pixel using the specified strategy.
///
/// For coordinates within the original image, returns the coordinate unchanged.
/// For coordinates beyond the edge, applies the padding strategy.
#[inline]
fn get_padded_coord(coord: usize, size: usize, strategy: EdgePadding) -> usize {
    if coord < size {
        return coord;
    }

    match strategy {
        EdgePadding::Replicate => size - 1,
        EdgePadding::Mirror => {
            // Reflect: coord beyond edge mirrors back
            // For coord = size + d, return size - 1 - d
            let d = coord - size;
            size.saturating_sub(1).saturating_sub(d)
        }
        EdgePadding::Wrap => coord % size,
    }
}

/// Pad a single-channel f32 plane to MCU-aligned dimensions.
///
/// Returns (padded_plane, padded_width, padded_height).
/// If no padding is needed, returns a clone of the input.
pub(crate) fn pad_plane_f32(
    plane: &[f32],
    width: usize,
    height: usize,
    mcu_size: usize,
    strategy: EdgePadding,
) -> Result<(Vec<f32>, usize, usize)> {
    let padded_w = (width + mcu_size - 1) / mcu_size * mcu_size;
    let padded_h = (height + mcu_size - 1) / mcu_size * mcu_size;

    // No padding needed
    if padded_w == width && padded_h == height {
        return Ok((
            try_clone_slice(plane, "pad_plane_f32 clone")?,
            width,
            height,
        ));
    }

    let mut out = try_alloc_zeroed_f32(padded_w * padded_h, "pad_plane_f32 output")?;

    for y in 0..padded_h {
        let src_y = get_padded_coord(y, height, strategy);
        for x in 0..padded_w {
            let src_x = get_padded_coord(x, width, strategy);
            out[y * padded_w + x] = plane[src_y * width + src_x];
        }
    }

    Ok((out, padded_w, padded_h))
}

/// Pad YCbCr f32 planes to MCU-aligned dimensions with per-channel strategies.
///
/// Y plane uses the luma strategy, Cb/Cr planes use the chroma strategy.
/// Handles subsampled chroma planes correctly (cb/cr may have different dimensions than y).
///
/// Returns ((y, cb, cr), padded_luma_w, padded_luma_h, padded_chroma_w, padded_chroma_h).
#[allow(clippy::type_complexity)]
pub(crate) fn pad_ycbcr_planes_subsampled(
    y: &[f32],
    width: usize,
    height: usize,
    cb: &[f32],
    cr: &[f32],
    c_width: usize,
    c_height: usize,
    mcu_size: usize,
    config: EdgePaddingConfig,
) -> Result<((Vec<f32>, Vec<f32>, Vec<f32>), usize, usize, usize, usize)> {
    // Pad luma to MCU-aligned dimensions
    let (y_padded, padded_w, padded_h) = pad_plane_f32(y, width, height, mcu_size, config.luma)?;

    // Chroma blocks are always 8x8. Padding chroma to multiples of 8 aligns with
    // the MCU grid because c_width = ceil(width / h_factor) and:
    // ceil(ceil(width / h_factor) / 8) * 8 == ceil(width / mcu_size) * (mcu_size / h_factor)
    let (cb_padded, padded_cw, padded_ch) = pad_plane_f32(cb, c_width, c_height, 8, config.chroma)?;
    let (cr_padded, _, _) = pad_plane_f32(cr, c_width, c_height, 8, config.chroma)?;

    Ok((
        (y_padded, cb_padded, cr_padded),
        padded_w,
        padded_h,
        padded_cw,
        padded_ch,
    ))
}

/// Pad grayscale f32 plane to MCU-aligned dimensions.
///
/// Returns (padded_plane, padded_width, padded_height).
#[allow(dead_code)] // Kept for future grayscale encoding support
pub(crate) fn pad_gray_plane(
    y: &[f32],
    width: usize,
    height: usize,
    mcu_size: usize,
    config: EdgePaddingConfig,
) -> Result<(Vec<f32>, usize, usize)> {
    pad_plane_f32(y, width, height, mcu_size, config.luma)
}

// Tests are in the old module (old/tests.rs)
