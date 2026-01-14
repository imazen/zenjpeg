//! JPEG encoder implementation.
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use jpegli::{EncoderConfig, PixelLayout};
//!
//! // Create reusable config
//! let config = EncoderConfig::new()
//!     .quality(85)
//!     .progressive(true);
//!
//! // Encode from raw bytes
//! let jpeg = config.encode_one(1920, 1080, PixelLayout::Rgb8Srgb, &rgb_bytes)?;
//! ```
//!
//! # Streaming Encoding
//!
//! For large images or when you want to process rows incrementally:
//!
//! ```rust,ignore
//! use jpegli::{EncoderConfig, PixelLayout};
//! use enough::Unstoppable;
//!
//! let config = EncoderConfig::new().quality(85);
//! let mut enc = config.encode_from_bytes(1920, 1080, PixelLayout::Rgb8Srgb)?;
//!
//! // Push rows (or use push_packed for all at once)
//! enc.push_packed(&rgb_bytes, Unstoppable)?;
//! let jpeg = enc.finish()?;
//! ```

#![allow(dead_code)]

#![allow(deprecated)]

// Internal implementation modules (pub for internal crate re-exports)
mod blocks;
#[doc(hidden)]
pub mod chroma;
#[doc(hidden)]
pub mod dct;
mod progressive;
#[doc(hidden)]
pub mod scan_script;
mod serialize;

#[doc(hidden)]
pub mod config;
#[cfg(feature = "experimental-hybrid-trellis")]
mod hybrid;
pub(crate) mod linear_lut;
#[cfg(feature = "parallel")]
#[doc(hidden)]
pub mod parallel;
#[doc(hidden)]
pub mod streaming;
#[doc(hidden)]
pub mod strip;

// v2 is the primary public API (types re-exported below)
#[doc(hidden)]
pub mod v2;

// Re-export v2 types at encode:: level for cleaner imports
#[cfg(feature = "parallel")]
#[allow(unused_imports)] // Public API re-export
pub use v2::ParallelEncoding;
pub use v2::Stop;

use crate::error::{Error, Result};

// Internal config types (v2::EncoderConfig is re-exported above, legacy one is used internally)
use config::EncoderConfig as LegacyEncoderConfig;
pub(crate) use config::ProgressiveScan;

use crate::foundation::alloc::{
    checked_size_2d, try_alloc_zeroed_f32, try_clone_slice, validate_dimensions, DEFAULT_MAX_PIXELS,
};
#[cfg(feature = "experimental-hybrid-trellis")]
use crate::foundation::consts::{DCT_BLOCK_SIZE, JPEG_ZIGZAG_ORDER};
use crate::quant::{self, Quality as LegacyQuality, QuantTable};
use crate::types::{
    ChromaDownsampling as LegacyChromaDownsampling, ColorSpace, EdgePadding, EdgePaddingConfig,
    JpegMode, PixelFormat as LegacyPixelFormat, Subsampling as LegacySubsampling,
};
use enough::Unstoppable;

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
///     .encode(&pixels)?;
/// ```
#[deprecated(
    since = "0.4.0",
    note = "Use StreamingEncoder instead, which provides better performance and lower memory usage"
)]
pub struct Encoder {
    /// Encoder configuration (accessible within crate for streaming encoder).
    pub(crate) config: LegacyEncoderConfig,
}

impl Encoder {
    /// Creates a new encoder with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: LegacyEncoderConfig::default(),
        }
    }

    /// Creates an encoder from configuration.
    #[must_use]
    pub fn from_config(config: LegacyEncoderConfig) -> Self {
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
    pub fn pixel_format(mut self, format: LegacyPixelFormat) -> Self {
        self.config.pixel_format = format;
        self
    }

    /// Sets the quality using jpegli's native quality scale.
    ///
    /// Use `Quality::from_quality(90.0)` for traditional JPEG quality (1-100)
    /// or `Quality::from_distance(1.0)` for butteraugli distance.
    #[must_use]
    pub fn jpegli_quality(mut self, quality: LegacyQuality) -> Self {
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
        conversion: crate::quant::quality_conversion::QualityConversion,
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
    pub fn quality(mut self, quality: LegacyQuality) -> Self {
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
    pub fn subsampling(mut self, subsampling: LegacySubsampling) -> Self {
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

    /// Allow 16-bit quantization tables for better low-quality precision.
    ///
    /// When `true` (default), quantization values can go up to 32767,
    /// using 16-bit DQT tables and extended sequential JPEGs (SOF1) when
    /// values exceed 255. This provides better precision at low quality.
    ///
    /// When `false`, quantization values are clamped to 255, producing
    /// baseline-compatible JPEGs (SOF0) that work with all decoders.
    ///
    /// Most images at quality >= 20 will have all quant values <= 255 anyway,
    /// so this only matters for very low quality settings. Most modern
    /// decoders support 16-bit quant tables without issue.
    #[must_use]
    pub fn allow_16bit_quant_tables(mut self, enable: bool) -> Self {
        self.config.allow_16bit_quant_tables = enable;
        self
    }

    /// Enables parallel encoding for improved throughput on multi-core systems.
    ///
    /// When enabled and `restart_interval > 0`, the encoder will use multiple
    /// threads for entropy encoding.
    #[cfg(feature = "parallel")]
    #[must_use]
    pub fn parallel(mut self, enable: bool) -> Self {
        self.config.parallel = enable;
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
    /// - [`LegacyChromaDownsampling::Box`]: Simple box filter (default, matches C++ jpegli)
    /// - [`LegacyChromaDownsampling::GammaAware`]: Gamma-aware averaging (better edges)
    /// - [`LegacyChromaDownsampling::GammaAwareIterative`]: Sharp YUV-style optimization (best quality)
    ///
    /// Has no effect for 4:4:4 subsampling (no downsampling needed).
    #[must_use]
    pub fn chroma_downsampling(mut self, method: LegacyChromaDownsampling) -> Self {
        self.config.chroma_downsampling = method;
        self
    }

    /// Convenience method: enable Sharp YUV-style chroma downsampling.
    ///
    /// - `enable = true` → `LegacyChromaDownsampling::GammaAwareIterative`
    /// - `enable = false` → `LegacyChromaDownsampling::Box`
    #[must_use]
    pub fn sharp_yuv(mut self, enable: bool) -> Self {
        self.config.chroma_downsampling = if enable {
            LegacyChromaDownsampling::GammaAwareIterative
        } else {
            LegacyChromaDownsampling::Box
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
    /// use jpegli::quant::aq::compute_aq_strength_map;
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
    pub fn aq_map(mut self, map: crate::quant::aq::AQStrengthMap) -> Self {
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
    /// This is equivalent to calling `encode_with_stop(data, Unstoppable)`.
    pub fn encode(&self, data: &[u8]) -> Result<Vec<u8>> {
        self.encode_with_stop(data, Unstoppable)
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
        self.encode_strip_based_with_stop(data, Unstoppable)
    }

    /// Encodes using strip-based processing with cancellation support.
    ///
    /// Delegates to StreamingEncoder which is the canonical implementation.
    fn encode_strip_based_with_stop(&self, data: &[u8], stop: impl Stop) -> Result<Vec<u8>> {
        // Build a StreamingEncoderBuilder with our config
        let mut builder = streaming::StreamingEncoder::new(self.config.width, self.config.height)
            .quality(self.config.quality)
            .subsampling(self.config.subsampling)
            .pixel_format(self.config.pixel_format)
            .mode(self.config.mode)
            .optimize_huffman(self.config.optimize_huffman)
            .chroma_downsampling(self.config.chroma_downsampling)
            .restart_interval(self.config.restart_interval)
            .use_xyb(self.config.use_xyb);

        if let Some(ref custom) = self.config.custom_quant_matrices {
            builder = builder.custom_quant_matrices(custom.clone());
        }

        builder.encode_with_stop(data, stop)}

    /// Generate a quantization table, using custom matrices if configured.
    ///
    /// This helper method respects the `custom_quant_matrices` config option.
    #[inline]
    #[allow(dead_code)]
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

/// Converts coefficients from natural order to zigzag order, writing directly to destination.
/// Avoids allocation when writing to pre-allocated block arrays.
#[cfg(feature = "experimental-hybrid-trellis")]
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
#[allow(dead_code)]
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
