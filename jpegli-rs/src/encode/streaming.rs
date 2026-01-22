//! Streaming input encoder API.
//!
//! This module provides a streaming encoder that accepts rows incrementally,
//! reducing peak memory by not requiring the full input image in memory.
//!
//! # Memory Savings
//!
//! For a 4K (3840x2160) RGB image:
//! - Standard encoder: ~50 MB peak (input buffer + internal)
//! - Streaming encoder: ~26 MB peak (~50% reduction)
//!
//! # Example
//!
//! ```rust,ignore
//! use jpegli::{StreamingEncoder, Quality, Subsampling};
//!
//! let mut encoder = StreamingEncoder::new(1920, 1080)
//!     .quality(Quality::ApproxJpegli(85.0))
//!     .subsampling(Subsampling::S420)
//!     .start()?;
//!
//! // Push rows one at a time (e.g., from a decoder or generator)
//! for row in image_rows {
//!     encoder.push_row(row)?;
//! }
//!
//! // Or push chunks of rows
//! // encoder.push_rows(chunk, 4)?;
//!
//! let jpeg = encoder.finish()?;
//! ```

#![allow(dead_code)]

use super::encoder_types::DownsamplingMethod;
use super::encoder_types::Quality;
use crate::encode::config::ComputedConfig;
use crate::encode::strip::StripProcessor;
use crate::encode::tuning::EncodingTables;
use crate::error::{Error, Result};
use crate::quant::{self, QuantTable, ZeroBiasParams};
use crate::types::{ColorSpace, JpegMode, PixelFormat, Subsampling};
use enough::{Stop, Unstoppable};

/// Builder for creating a streaming encoder.
///
/// Use [`StreamingEncoder::new()`] to start building.
#[derive(Debug, Clone)]
pub(crate) struct StreamingEncoderBuilder {
    width: u32,
    height: u32,
    quality: Quality,
    subsampling: Subsampling,
    pixel_format: PixelFormat,
    mode: JpegMode,
    optimize_huffman: bool,
    chroma_downsampling: DownsamplingMethod,
    restart_interval: u16,
    /// Custom encoding tables (quantization + zero-bias).
    /// `None` means use perceptual defaults based on color mode and quality.
    encoding_tables: Option<Box<EncodingTables>>,
    use_xyb: bool,
    /// Enable mozjpeg-style overshoot deringing (on by default)
    deringing: bool,
    /// Allow 16-bit quantization tables (default: true)
    allow_16bit_quant_tables: bool,
    /// Use separate Cb and Cr quantization tables (default: true = 3 tables)
    separate_chroma_tables: bool,
    /// Enable parallel encoding (requires `parallel` feature)
    #[cfg(feature = "parallel")]
    parallel: bool,
    /// Hybrid quantization configuration (requires `experimental-hybrid-trellis` feature)
    #[cfg(feature = "experimental-hybrid-trellis")]
    hybrid_config: crate::hybrid::config::HybridConfig,
    /// Custom AQ map (requires `experimental-hybrid-trellis` feature)
    #[cfg(feature = "experimental-hybrid-trellis")]
    custom_aq_map: Option<crate::quant::aq::AQStrengthMap>,
    /// Trellis quantization config (mozjpeg-compat API, requires `experimental-hybrid-trellis`)
    #[cfg(feature = "experimental-hybrid-trellis")]
    trellis: Option<super::mozjpeg_compat::TrellisConfig>,
}

impl StreamingEncoderBuilder {
    /// Creates a new streaming encoder builder with default settings.
    fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            quality: Quality::default(),
            subsampling: Subsampling::S444,
            pixel_format: PixelFormat::Rgb,
            mode: JpegMode::Baseline,
            optimize_huffman: true,
            chroma_downsampling: DownsamplingMethod::Box,
            restart_interval: 0,
            encoding_tables: None,
            use_xyb: false,
            deringing: true,
            allow_16bit_quant_tables: true,
            separate_chroma_tables: true,
            #[cfg(feature = "parallel")]
            parallel: false,
            #[cfg(feature = "experimental-hybrid-trellis")]
            hybrid_config: crate::hybrid::config::HybridConfig::disabled(),
            #[cfg(feature = "experimental-hybrid-trellis")]
            custom_aq_map: None,
            #[cfg(feature = "experimental-hybrid-trellis")]
            trellis: None,
        }
    }

    /// Sets the quality using jpegli's native quality scale.
    ///
    /// Accepts either:
    /// - An integer (1-100) for traditional JPEG quality
    /// - A `Quality` enum for advanced options including butteraugli distance
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// // Simple integer quality (most common)
    /// let enc = JpegEncoder::new(640, 480).quality(85);
    ///
    /// // Quality enum for explicit control
    /// let enc = JpegEncoder::new(640, 480).quality(Quality::ApproxJpegli(85.0));
    ///
    /// // Butteraugli distance (advanced)
    /// let enc = JpegEncoder::new(640, 480).quality(Quality::ApproxButteraugli(1.0));
    /// ```
    #[must_use]
    pub(crate) fn quality(mut self, quality: impl Into<Quality>) -> Self {
        self.quality = quality.into();
        self
    }

    /// Sets the quality using butteraugli distance.
    ///
    /// Butteraugli distance is a perceptual quality metric where:
    /// - 0.0 = lossless (not achievable with JPEG)
    /// - 0.5 = very high quality
    /// - 1.0 = high quality (default)
    /// - 2.0 = medium quality
    /// - 3.0+ = low quality
    ///
    /// This is the native quality metric used by jpegli internally.
    /// For most users, `.quality(85)` with traditional 1-100 scale is easier.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let enc = JpegEncoder::new(640, 480).distance(1.0);
    /// ```
    #[must_use]
    pub(crate) fn distance(mut self, distance: f32) -> Self {
        self.quality = Quality::ApproxButteraugli(distance);
        self
    }

    /// Enables or disables progressive JPEG encoding.
    ///
    /// Progressive JPEGs display a low-quality version first, then progressively
    /// improve as more data loads. They're slightly smaller but require optimized
    /// Huffman tables.
    ///
    /// When enabled, `optimize_huffman` is automatically enabled as well.
    ///
    /// When disabled, if the current mode is Progressive, it switches to Baseline.
    /// Otherwise, the current mode (e.g., Extended) is preserved.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let enc = JpegEncoder::new(640, 480).progressive(true);
    /// ```
    #[must_use]
    pub(crate) fn progressive(mut self, enable: bool) -> Self {
        if enable {
            self.mode = JpegMode::Progressive;
            self.optimize_huffman = true;
        } else if self.mode == JpegMode::Progressive {
            // Only change from Progressive to Baseline; preserve other modes like Extended
            self.mode = JpegMode::Baseline;
        }
        self
    }

    /// Sets chroma subsampling.
    #[must_use]
    pub(crate) fn subsampling(mut self, subsampling: Subsampling) -> Self {
        self.subsampling = subsampling;
        self
    }

    /// Sets the pixel format of input data.
    #[must_use]
    pub(crate) fn pixel_format(mut self, format: PixelFormat) -> Self {
        self.pixel_format = format;
        self
    }

    /// Sets the JPEG encoding mode.
    #[must_use]
    pub(crate) fn mode(mut self, mode: JpegMode) -> Self {
        self.mode = mode;
        self
    }

    /// Enables optimized Huffman tables.
    #[must_use]
    pub(crate) fn optimize_huffman(mut self, enable: bool) -> Self {
        self.optimize_huffman = enable;
        self
    }

    /// Sets chroma downsampling method for subsampled modes.
    #[must_use]
    pub(crate) fn chroma_downsampling(mut self, method: DownsamplingMethod) -> Self {
        self.chroma_downsampling = method;
        self
    }

    /// Enables Sharp YUV chroma downsampling for better edge quality.
    ///
    /// Sharp YUV uses iterative optimization to preserve edges during chroma
    /// subsampling (4:2:0, 4:2:2). This produces noticeably better quality
    /// on images with sharp color transitions at the cost of slower encoding.
    ///
    /// Equivalent to `.chroma_downsampling(DownsamplingMethod::GammaAwareIterative)`.
    ///
    /// Has no effect for 4:4:4 subsampling (no downsampling needed).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let jpeg = JpegEncoder::new(640, 480)
    ///     .quality(85)
    ///     .subsampling(Subsampling::S420)
    ///     .sharp_yuv(true)
    ///     .encode(&pixels)?;
    /// ```
    #[must_use]
    pub(crate) fn sharp_yuv(mut self, enable: bool) -> Self {
        self.chroma_downsampling = if enable {
            DownsamplingMethod::GammaAwareIterative
        } else {
            DownsamplingMethod::Box
        };
        self
    }

    /// Sets the restart interval (MCUs between restart markers).
    #[must_use]
    pub(crate) fn restart_interval(mut self, interval: u16) -> Self {
        self.restart_interval = interval;
        self
    }

    /// Enables parallel encoding for improved throughput on multi-core systems.
    ///
    /// When enabled, the encoder will use multiple threads for entropy encoding
    /// (and optionally DCT). This requires restart markers, so if `restart_interval`
    /// is 0, it will be automatically set to 64 MCUs.
    ///
    /// Performance characteristics (2048x2048 image):
    /// - 2 threads: 1.2-1.6x speedup, 60-80% efficiency
    /// - 4 threads: 1.3-1.7x speedup, 30-40% efficiency
    ///
    /// Parallel encoding is most beneficial for images >= 512x512.
    /// For smaller images, the overhead may negate the benefits.
    ///
    /// Requires the `parallel` feature to be enabled.
    #[cfg(feature = "parallel")]
    #[must_use]
    pub(crate) fn parallel(mut self, enable: bool) -> Self {
        self.parallel = enable;
        self
    }

    /// Sets custom encoding tables (quantization + zero-bias).
    ///
    /// This replaces both quantization tables and zero-bias configuration
    /// with values from the provided `EncodingTables`.
    ///
    /// Takes `Box<EncodingTables>` since custom tables are rarely used and
    /// the struct is ~1.5KB.
    #[must_use]
    pub(crate) fn encoding_tables(mut self, tables: Box<EncodingTables>) -> Self {
        self.encoding_tables = Some(tables);
        self
    }

    /// Enables XYB color space encoding.
    ///
    /// XYB is a perceptual color space used by JPEG XL that better models human
    /// vision than YCbCr. When enabled, the output JPEG uses XYB-encoded data
    /// with an ICC profile that allows compatible decoders to render correctly.
    ///
    /// Linear input formats (Rgb16, Rgba16, RgbF32, RgbaF32) are ideal for XYB
    /// since XYB is defined in linear light space.
    ///
    /// Note: Without ICC profile support in the decoder, images will display with
    /// incorrect colors. Use standard YCbCr mode for maximum compatibility.
    #[must_use]
    pub(crate) fn use_xyb(mut self, enable: bool) -> Self {
        self.use_xyb = enable;
        self
    }

    /// Enables mozjpeg-style overshoot deringing.
    ///
    /// This reduces visible ringing artifacts near sharp edges, particularly
    /// on white backgrounds. Works by allowing pixel values to "overshoot"
    /// beyond the displayable range, which gets clamped on decode but produces
    /// smoother DCT coefficients.
    ///
    /// Enabled by default. This technique was pioneered by @kornel in mozjpeg.
    #[must_use]
    pub(crate) fn deringing(mut self, enable: bool) -> Self {
        self.deringing = enable;
        self
    }

    /// Allow 16-bit quantization tables for better low-quality precision.
    ///
    /// When enabled (default), quantization values can exceed 255, producing
    /// extended sequential JPEGs (SOF1 marker).
    ///
    /// When disabled, quantization values are clamped to 255, producing
    /// baseline-compatible JPEGs (SOF0 marker) that work with all decoders.
    #[must_use]
    pub(crate) fn allow_16bit_quant_tables(mut self, enable: bool) -> Self {
        self.allow_16bit_quant_tables = enable;
        self
    }

    /// Use separate Cb and Cr quantization tables.
    ///
    /// When enabled (default), uses 3 tables: Y, Cb, Cr.
    /// When disabled, uses 2 tables: Y, shared chroma.
    #[must_use]
    pub(crate) fn separate_chroma_tables(mut self, enable: bool) -> Self {
        self.separate_chroma_tables = enable;
        self
    }

    /// Enables hybrid quantization (jpegli AQ + mozjpeg trellis).
    ///
    /// This combines jpegli's adaptive quantization (which determines WHERE
    /// to spend bits based on image content) with mozjpeg's trellis quantization
    /// (which optimizes HOW to spend bits via rate-distortion optimization).
    ///
    /// Requires the `experimental-hybrid-trellis` feature.
    #[cfg(feature = "experimental-hybrid-trellis")]
    #[must_use]
    pub(crate) fn hybrid_trellis(mut self, enable: bool) -> Self {
        self.hybrid_config = if enable {
            crate::hybrid::config::HybridConfig::default()
        } else {
            crate::hybrid::config::HybridConfig::disabled()
        };
        self
    }

    /// Sets custom hybrid quantization configuration.
    ///
    /// Allows fine-tuning all hybrid AQ+trellis parameters.
    /// See `HybridConfig` for available options.
    ///
    /// Requires the `experimental-hybrid-trellis` feature.
    #[cfg(feature = "experimental-hybrid-trellis")]
    #[must_use]
    pub(crate) fn hybrid_config(mut self, config: crate::hybrid::config::HybridConfig) -> Self {
        self.hybrid_config = config;
        self
    }

    /// Sets trellis quantization configuration (mozjpeg-compatible API).
    ///
    /// This enables trellis quantization for rate-distortion optimization,
    /// using the mozjpeg-rs compatible `TrellisConfig` type.
    ///
    /// Requires the `experimental-hybrid-trellis` feature.
    #[cfg(feature = "experimental-hybrid-trellis")]
    #[must_use]
    pub fn trellis(mut self, config: super::mozjpeg_compat::TrellisConfig) -> Self {
        self.trellis = Some(config);
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
    /// Requires the `experimental-hybrid-trellis` feature.
    #[cfg(feature = "experimental-hybrid-trellis")]
    #[must_use]
    pub(crate) fn aq_map(mut self, map: crate::quant::aq::AQStrengthMap) -> Self {
        self.custom_aq_map = Some(map);
        self
    }

    /// Starts a streaming encoder for row-by-row input.
    ///
    /// Use this when you want to push rows incrementally (e.g., from a decoder
    /// or generator). For encoding a complete buffer at once, use `.encode()`
    /// instead.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use jpegli::JpegEncoder;
    ///
    /// let mut encoder = JpegEncoder::new(640, 480)
    ///     .quality(85)
    ///     .start()?;
    ///
    /// for row in image_rows {
    ///     encoder.push_row(row)?;
    /// }
    ///
    /// let jpeg = encoder.finish()?;
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Dimensions are zero or exceed maximum
    /// - Memory allocation fails
    pub(crate) fn start(self) -> Result<StreamingEncoder> {
        StreamingEncoder::from_builder(self)
    }

    /// Encodes a complete image buffer in one call.
    ///
    /// This is the simplest way to encode an image. For streaming scenarios
    /// where you want to push rows incrementally, use `.start()` instead.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use jpegli::{JpegEncoder, Subsampling};
    ///
    /// let pixels: Vec<u8> = vec![128; 640 * 480 * 3];
    /// let jpeg = JpegEncoder::new(640, 480)
    ///     .quality(85)
    ///     .subsampling(Subsampling::S420)
    ///     .encode(&pixels)?;
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Buffer size doesn't match width × height × bytes_per_pixel
    /// - Encoding fails
    pub(crate) fn encode(self, data: &[u8]) -> Result<Vec<u8>> {
        let width = self.width as usize;
        let height = self.height as usize;
        let bpp = self.pixel_format.bytes_per_pixel();
        let expected_size = width * height * bpp;

        if data.len() != expected_size {
            return Err(Error::invalid_buffer_size(expected_size, data.len()));
        }

        let mut encoder = self.start()?;
        let row_size = width * bpp;

        for y in 0..height {
            let start = y * row_size;
            encoder.push_row(&data[start..start + row_size])?;
        }

        encoder.finish()
    }

    /// Encodes a complete image buffer with cancellation support.
    ///
    /// Like `encode()`, but checks for cancellation between strips.
    pub(crate) fn encode_with_stop(self, data: &[u8], stop: impl Stop) -> Result<Vec<u8>> {
        let width = self.width as usize;
        let height = self.height as usize;
        let bpp = self.pixel_format.bytes_per_pixel();
        let expected_size = width * height * bpp;

        if data.len() != expected_size {
            return Err(Error::invalid_buffer_size(expected_size, data.len()));
        }

        let mut encoder = self.start()?;
        let row_size = width * bpp;

        for y in 0..height {
            let start = y * row_size;
            encoder.push_row_with_stop(&data[start..start + row_size], &stop)?;
        }

        encoder.finish_with_stop(stop)
    }

    /// Estimates the peak memory usage for this configuration.
    ///
    /// Returns the estimated peak memory in bytes based on image dimensions,
    /// subsampling mode, and pixel format. This estimate includes:
    /// - Row buffer (one strip's worth of input data)
    /// - Strip processing buffers (f32 YCbCr planes)
    /// - Pending DCT blocks (double-buffered)
    /// - Final i16 block storage
    /// - AQ strength storage
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use jpegli::{StreamingEncoder, Subsampling};
    ///
    /// let estimated = StreamingEncoder::new(3840, 2160)
    ///     .subsampling(Subsampling::S420)
    ///     .estimate_memory_usage();
    ///
    /// println!("Estimated peak memory: {} MB", estimated / 1024 / 1024);
    /// ```
    #[must_use]
    pub(crate) fn estimate_memory_usage(&self) -> usize {
        let width = self.width as usize;
        let height = self.height as usize;

        // Strip height based on subsampling
        let strip_height = match self.subsampling {
            Subsampling::S420 | Subsampling::S440 => 16,
            _ => 8,
        };

        // MCU size for padding
        let mcu_size = self.subsampling.mcu_size();
        let padded_width = (width + mcu_size - 1) / mcu_size * mcu_size;

        // Chroma dimensions
        let (c_width, c_strip_height) = match self.subsampling {
            Subsampling::S420 => ((width + 1) / 2, strip_height / 2),
            Subsampling::S422 => ((width + 1) / 2, strip_height),
            Subsampling::S440 => (width, strip_height / 2),
            Subsampling::S444 => (width, strip_height),
        };
        let padded_c_width = (c_width + 7) / 8 * 8;

        // Block counts
        let y_blocks_w = (width + 7) / 8;
        let y_blocks_h = (height + 7) / 8;
        let y_block_count = y_blocks_w * y_blocks_h;

        let c_block_count = match self.subsampling {
            Subsampling::S420 => ((width + 15) / 16) * ((height + 15) / 16),
            Subsampling::S422 => ((width + 15) / 16) * y_blocks_h,
            Subsampling::S440 => y_blocks_w * ((height + 15) / 16),
            Subsampling::S444 => y_block_count,
        };

        // 1. Row buffer for input (one strip's worth)
        let bpp = self.pixel_format.bytes_per_pixel();
        let row_buffer = width * strip_height * bpp;

        // 2. Strip f32 buffers (Y, Cb, Cr at full resolution before downsampling)
        let strip_y = padded_width * strip_height * 4; // f32 = 4 bytes
        let strip_cb = padded_width * strip_height * 4;
        let strip_cr = padded_width * strip_height * 4;

        // 3. Downsampled chroma temp buffers
        let strip_cb_down = padded_c_width * c_strip_height * 4;
        let strip_cr_down = padded_c_width * c_strip_height * 4;

        // 4. Pending f32 DCT blocks (double-buffered, 2 iMCU rows)
        let padded_y_blocks_h = padded_width / 8;
        let v_samp = match self.subsampling {
            Subsampling::S420 | Subsampling::S440 => 2,
            _ => 1,
        };
        let pending_y_capacity = padded_y_blocks_h * v_samp;
        let padded_c_blocks_h = padded_c_width / 8;
        let pending_c_capacity = padded_c_blocks_h;

        // 256 bytes per f32 block, 2 buffers (double-buffered)
        let pending_y_f32 = 2 * pending_y_capacity * 256;
        let pending_cb_f32 = 2 * pending_c_capacity * 256;
        let pending_cr_f32 = 2 * pending_c_capacity * 256;

        // 5. Final i16 blocks (128 bytes per block)
        let y_blocks_i16 = y_block_count * 128;
        let c_blocks_i16 = c_block_count * 2 * 128; // Cb + Cr

        // 6. AQ strengths (one f32 per Y block)
        let aq_strengths = y_block_count * 4;

        // 7. Token buffer (always allocated to store encoded coefficients)
        // Each block produces ~50 symbols on average, each symbol is ~4 bytes.
        // This buffer is used regardless of Huffman optimization setting because
        // the encoder must buffer all coefficients before final output.
        // Note: Huffman optimization may use slightly more memory for table building,
        // but the token buffer dominates and is always needed.
        let total_blocks = y_block_count + c_block_count * 2;
        let token_buffer = total_blocks * 50 * 4; // ~50 symbols per block, 4 bytes each

        // 8. Output buffer estimate (grows during encoding)
        let output_estimate = width * height / 8; // ~1 bit per pixel rough estimate

        // Total estimate
        row_buffer
            + strip_y
            + strip_cb
            + strip_cr
            + strip_cb_down
            + strip_cr_down
            + pending_y_f32
            + pending_cb_f32
            + pending_cr_f32
            + y_blocks_i16
            + c_blocks_i16
            + aq_strengths
            + token_buffer
            + output_estimate
    }

    /// Returns an absolute ceiling on memory usage.
    ///
    /// Unlike [`estimate_memory_usage`], this returns a **guaranteed upper bound**
    /// that actual peak memory will never exceed. Use this for resource reservation
    /// when you need certainty rather than accuracy.
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
    /// use jpegli::{StreamingEncoder, Subsampling};
    ///
    /// let ceiling = StreamingEncoder::new(3840, 2160)
    ///     .subsampling(Subsampling::S420)
    ///     .estimate_memory_ceiling();
    ///
    /// // Reserve this much memory before encoding
    /// assert!(actual_peak <= ceiling);
    /// ```
    #[must_use]
    pub(crate) fn estimate_memory_ceiling(&self) -> usize {
        let width = self.width as usize;
        let height = self.height as usize;

        // Strip height based on subsampling
        let strip_height = match self.subsampling {
            Subsampling::S420 | Subsampling::S440 => 16,
            _ => 8,
        };

        // MCU size for padding (worst case alignment)
        let mcu_size = self.subsampling.mcu_size();
        let padded_width = (width + mcu_size - 1) / mcu_size * mcu_size;
        let padded_height = (height + mcu_size - 1) / mcu_size * mcu_size;

        // Chroma dimensions (padded)
        let (c_width, c_strip_height) = match self.subsampling {
            Subsampling::S420 => ((padded_width + 1) / 2, strip_height / 2),
            Subsampling::S422 => ((padded_width + 1) / 2, strip_height),
            Subsampling::S440 => (padded_width, strip_height / 2),
            Subsampling::S444 => (padded_width, strip_height),
        };
        let padded_c_width = (c_width + 7) / 8 * 8;

        // Block counts (use padded dimensions for ceiling)
        let y_blocks_w = padded_width / 8;
        let y_blocks_h = padded_height / 8;
        let y_block_count = y_blocks_w * y_blocks_h;

        let c_block_count = match self.subsampling {
            Subsampling::S420 => (padded_width / 16) * (padded_height / 16),
            Subsampling::S422 => (padded_width / 16) * y_blocks_h,
            Subsampling::S440 => y_blocks_w * (padded_height / 16),
            Subsampling::S444 => y_block_count,
        };

        // 1. Row buffer for input (one strip's worth, use max bpp=4 for RGBA)
        let max_bpp = 4;
        let row_buffer = padded_width * strip_height * max_bpp;

        // 2. Strip f32 buffers (Y, Cb, Cr at full resolution)
        // Account for potential 2x capacity for Vec growth
        let strip_y = padded_width * strip_height * 4;
        let strip_cb = padded_width * strip_height * 4;
        let strip_cr = padded_width * strip_height * 4;

        // 3. Downsampled chroma temp buffers
        let strip_cb_down = padded_c_width * c_strip_height * 4;
        let strip_cr_down = padded_c_width * c_strip_height * 4;

        // 4. Pending f32 DCT blocks (double-buffered)
        let padded_y_blocks_per_row = padded_width / 8;
        let v_samp = match self.subsampling {
            Subsampling::S420 | Subsampling::S440 => 2,
            _ => 1,
        };
        let pending_y_capacity = padded_y_blocks_per_row * v_samp;
        let padded_c_blocks_per_row = padded_c_width / 8;
        let pending_c_capacity = padded_c_blocks_per_row;

        // 256 bytes per f32 block (64 floats), 2 buffers each
        let pending_y_f32 = 2 * pending_y_capacity * 256;
        let pending_cb_f32 = 2 * pending_c_capacity * 256;
        let pending_cr_f32 = 2 * pending_c_capacity * 256;

        // 5. Final i16 blocks (128 bytes per block = 64 * i16)
        let y_blocks_i16 = y_block_count * 128;
        let c_blocks_i16 = c_block_count * 2 * 128;

        // 6. AQ strengths (one f32 per Y block)
        let aq_strengths = y_block_count * 4;

        // 7. Token buffer - CEILING: worst-case ~90 tokens per block
        // High-frequency blocks (noise, text, fine detail) produce more tokens.
        // Each token is typically 4 bytes (symbol + context).
        let total_blocks = y_block_count + c_block_count * 2;
        let token_buffer = total_blocks * 90 * 4;

        // 8. Output buffer - CEILING: worst-case is when image is incompressible
        // Quality 100 with noise can produce output larger than input.
        // Absolute ceiling: 1 byte per pixel (8 bits vs ~1-2 bits typical).
        let output_ceiling = padded_width * padded_height;

        // 9. Huffman table overhead (frequency tables, code tables)
        // 4 tables (2 DC + 2 AC) × 256 entries × 8 bytes = 8 KB
        let huffman_tables = 4 * 256 * 8;

        // 10. Progressive scan overhead (if enabled, stores all coefficients)
        // This is already covered by the i16 blocks, but add scan metadata
        let scan_overhead = 64 * 8; // ~512 bytes for scan definitions

        // 11. Vec capacity overhead - allocators round up
        // Add 5% for allocator overhead (power-of-2 rounding, headers)
        let subtotal = row_buffer
            + strip_y
            + strip_cb
            + strip_cr
            + strip_cb_down
            + strip_cr_down
            + pending_y_f32
            + pending_cb_f32
            + pending_cr_f32
            + y_blocks_i16
            + c_blocks_i16
            + aq_strengths
            + token_buffer
            + output_ceiling
            + huffman_tables
            + scan_overhead;

        // Add 5% allocator overhead ceiling
        subtotal + subtotal / 20
    }
}

/// Streaming input JPEG encoder.
///
/// Accepts rows incrementally and outputs JPEG at the end.
/// Uses strip-based processing internally for low peak memory usage.
pub(crate) struct StreamingEncoder {
    /// Image width in pixels
    width: usize,
    /// Image height in pixels
    height: usize,
    /// Bytes per row of input data
    bytes_per_row: usize,
    /// Strip height (rows to buffer before processing)
    strip_height: usize,

    /// Row buffer (accumulates rows until strip is ready)
    row_buffer: Vec<u8>,
    /// Number of rows currently buffered
    rows_buffered: usize,
    /// Current Y position (rows processed so far)
    current_y: usize,

    /// Underlying strip processor
    processor: StripProcessor,

    /// Configuration for JPEG output generation
    config: ComputedConfig,

    /// Quantization tables (generated from quality)
    y_quant: QuantTable,
    cb_quant: QuantTable,
    cr_quant: QuantTable,
}

impl StreamingEncoder {
    /// Creates a new streaming encoder builder with the given dimensions.
    ///
    /// Use the builder methods to configure quality, subsampling, etc.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use jpegli::{StreamingEncoder, Quality, Subsampling};
    ///
    /// let encoder = StreamingEncoder::new(1920, 1080)
    ///     .quality(Quality::ApproxJpegli(85.0))
    ///     .subsampling(Subsampling::S420)
    ///     .start()?;
    /// ```
    #[must_use]
    #[allow(clippy::new_ret_no_self)] // Builder pattern: new() returns builder
    pub(crate) fn new(width: u32, height: u32) -> StreamingEncoderBuilder {
        StreamingEncoderBuilder::new(width, height)
    }

    /// Creates a streaming encoder from builder configuration.
    fn from_builder(builder: StreamingEncoderBuilder) -> Result<Self> {
        let width = builder.width as usize;
        let height = builder.height as usize;

        if width == 0 || height == 0 {
            return Err(Error::invalid_dimensions(
                builder.width,
                builder.height,
                "dimensions must be non-zero",
            ));
        }

        // Create strip processor
        let mut processor = StripProcessor::with_options(
            width,
            height,
            builder.subsampling,
            builder.pixel_format,
            builder.chroma_downsampling,
            builder.restart_interval,
        )?;

        // Enable XYB mode if requested
        if builder.use_xyb {
            processor.set_xyb_mode(true);
        }

        // Set deringing (on by default in both builder and processor)
        processor.set_deringing(builder.deringing);

        // Enable trellis quantization if configured
        #[cfg(feature = "experimental-hybrid-trellis")]
        if let Some(ref trellis) = builder.trellis {
            processor.set_trellis(*trellis);
        }

        // Generate quantization tables and zero-bias params
        let is_420 = builder.subsampling == Subsampling::S420;
        let distance = builder.quality.to_distance();
        let color_space = if builder.use_xyb {
            ColorSpace::Xyb
        } else {
            ColorSpace::YCbCr
        };

        let allow_16bit = builder.allow_16bit_quant_tables;
        let ((y_quant, cb_quant, cr_quant), (y_zero_bias, cb_zero_bias, cr_zero_bias)) =
            if let Some(ref tables) = builder.encoding_tables {
                // Use custom encoding tables
                let quant = tables.generate_quant_tables(distance, is_420);
                let zero_bias = tables.generate_zero_bias_all();
                // Apply allow_16bit clamping if needed
                let quant = if allow_16bit {
                    quant
                } else {
                    (
                        quant.0.clamp_to_baseline(),
                        quant.1.clamp_to_baseline(),
                        quant.2.clamp_to_baseline(),
                    )
                };
                (quant, zero_bias)
            } else {
                // Use perceptual defaults with allow_16bit support
                //
                // When separate_chroma_tables is false (2-table mode, jpeg_set_quality),
                // use the Cr base matrix for both Cb and Cr tables. This matches C++
                // jpegli behavior where the single chroma table uses the Cr matrix.
                let cb_component = if builder.separate_chroma_tables { 1 } else { 2 };

                let quant = (
                    quant::generate_quant_table_ex(
                        builder.quality,
                        0,
                        color_space,
                        builder.use_xyb,
                        is_420,
                        allow_16bit,
                    ),
                    quant::generate_quant_table_ex(
                        builder.quality,
                        cb_component,
                        color_space,
                        builder.use_xyb,
                        is_420,
                        allow_16bit,
                    ),
                    quant::generate_quant_table_ex(
                        builder.quality,
                        2,
                        color_space,
                        builder.use_xyb,
                        is_420,
                        allow_16bit,
                    ),
                );

                // Compute effective distance for quality-adaptive zero bias
                let effective_distance =
                    quant::quant_vals_to_distance(&quant.0, &quant.1, &quant.2);

                // Auto-select zero bias based on color mode (matches C++ jpegli behavior)
                let zero_bias = if builder.use_xyb {
                    (
                        ZeroBiasParams::for_xyb(),
                        ZeroBiasParams::for_xyb(),
                        ZeroBiasParams::for_xyb(),
                    )
                } else {
                    (
                        ZeroBiasParams::for_ycbcr(effective_distance, 0),
                        ZeroBiasParams::for_ycbcr(effective_distance, 1),
                        ZeroBiasParams::for_ycbcr(effective_distance, 2),
                    )
                };

                (quant, zero_bias)
            };

        processor.set_quant_tables(
            y_quant.clone(),
            cb_quant.clone(),
            cr_quant.clone(),
            y_zero_bias,
            cb_zero_bias,
            cr_zero_bias,
        )?;

        let strip_height = processor.strip_height();
        let bytes_per_row = width * builder.pixel_format.bytes_per_pixel();

        // Allocate row buffer for one strip
        let row_buffer = vec![0u8; bytes_per_row * strip_height];

        // Create config for final JPEG output
        let config = ComputedConfig {
            width: builder.width,
            height: builder.height,
            pixel_format: builder.pixel_format,
            quality: builder.quality,
            subsampling: builder.subsampling,
            mode: builder.mode,
            optimize_huffman: builder.optimize_huffman,
            chroma_downsampling: builder.chroma_downsampling,
            restart_interval: builder.restart_interval,
            use_xyb: builder.use_xyb,
            #[cfg(feature = "parallel")]
            parallel: builder.parallel,
            #[cfg(feature = "experimental-hybrid-trellis")]
            hybrid_config: builder.hybrid_config,
            #[cfg(feature = "experimental-hybrid-trellis")]
            custom_aq_map: builder.custom_aq_map,
            #[cfg(feature = "experimental-hybrid-trellis")]
            trellis: builder.trellis,
            encoding_tables: builder.encoding_tables,
            edge_padding: crate::types::EdgePaddingConfig::default(),
            original_width: None,
            original_height: None,
            allow_16bit_quant_tables: builder.allow_16bit_quant_tables,
            separate_chroma_tables: builder.separate_chroma_tables,
        };

        Ok(Self {
            width,
            height,
            bytes_per_row,
            strip_height,
            row_buffer,
            rows_buffered: 0,
            current_y: 0,
            processor,
            config,
            y_quant,
            cb_quant,
            cr_quant,
        })
    }

    /// Returns the number of rows pushed so far.
    #[must_use]
    pub(crate) fn rows_pushed(&self) -> usize {
        self.current_y + self.rows_buffered
    }

    /// Returns the expected number of bytes per row.
    #[must_use]
    pub(crate) fn bytes_per_row(&self) -> usize {
        self.bytes_per_row
    }

    /// Returns the total height of the image.
    #[must_use]
    pub(crate) fn height(&self) -> usize {
        self.height
    }

    /// Returns the strip height (internal processing unit).
    #[must_use]
    pub(crate) fn strip_height(&self) -> usize {
        self.strip_height
    }

    /// Pushes a single row of pixel data.
    ///
    /// The row must be exactly `bytes_per_row()` bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Row length doesn't match expected bytes per row
    /// - All rows have already been pushed
    /// - Internal processing fails
    pub(crate) fn push_row(&mut self, row: &[u8]) -> Result<()> {
        self.push_row_with_stop(row, Unstoppable)
    }

    /// Pushes a single row with cancellation support.
    ///
    /// The `stop` source is checked before processing each strip.
    /// Returns `Error::cancelled()` if cancellation is requested.
    pub(crate) fn push_row_with_stop(&mut self, row: &[u8], stop: impl Stop) -> Result<()> {
        // Check cancellation
        stop.check()?;

        // Validate row size
        if row.len() != self.bytes_per_row {
            return Err(Error::invalid_buffer_size(self.bytes_per_row, row.len()));
        }

        // Check if we've already received all rows
        if self.current_y + self.rows_buffered >= self.height {
            return Err(Error::io_error(format!(
                "already received all {} rows",
                self.height
            )));
        }

        // Copy row into buffer
        let offset = self.rows_buffered * self.bytes_per_row;
        self.row_buffer[offset..offset + self.bytes_per_row].copy_from_slice(row);
        self.rows_buffered += 1;

        // Check if we should flush the strip
        let remaining = self.height - self.current_y;
        if self.rows_buffered >= self.strip_height || self.rows_buffered >= remaining {
            self.flush_strip_with_stop(&stop)?;
        }

        Ok(())
    }

    /// Pushes multiple rows at once.
    ///
    /// The data must be exactly `num_rows * bytes_per_row()` bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Data length doesn't match expected size
    /// - Too many rows would be pushed
    /// - Internal processing fails
    pub(crate) fn push_rows(&mut self, data: &[u8], num_rows: usize) -> Result<()> {
        self.push_rows_with_stop(data, num_rows, Unstoppable)
    }

    /// Pushes multiple rows with cancellation support.
    ///
    /// This method is optimized to process complete strips directly from the input
    /// buffer without intermediate copies. Only partial strips at the beginning
    /// and end require buffering.
    pub(crate) fn push_rows_with_stop(
        &mut self,
        data: &[u8],
        num_rows: usize,
        stop: impl Stop,
    ) -> Result<()> {
        let expected_len = num_rows * self.bytes_per_row;
        if data.len() != expected_len {
            return Err(Error::invalid_buffer_size(expected_len, data.len()));
        }

        if num_rows == 0 {
            return Ok(());
        }

        // Check if we've already received all rows
        if self.current_y + self.rows_buffered >= self.height {
            return Err(Error::io_error(format!(
                "already received all {} rows",
                self.height
            )));
        }

        let mut data_offset = 0usize;
        let mut rows_remaining = num_rows;

        // Step 1: Complete any partial strip in buffer
        if self.rows_buffered > 0 {
            let rows_to_complete = (self.strip_height - self.rows_buffered).min(rows_remaining);
            let rows_to_complete =
                rows_to_complete.min(self.height - self.current_y - self.rows_buffered);

            // Copy rows to buffer to complete the strip
            let buf_offset = self.rows_buffered * self.bytes_per_row;
            let src_bytes = rows_to_complete * self.bytes_per_row;
            self.row_buffer[buf_offset..buf_offset + src_bytes]
                .copy_from_slice(&data[data_offset..data_offset + src_bytes]);

            self.rows_buffered += rows_to_complete;
            data_offset += src_bytes;
            rows_remaining -= rows_to_complete;

            // Flush if strip is complete
            let remaining_height = self.height - self.current_y;
            if self.rows_buffered >= self.strip_height || self.rows_buffered >= remaining_height {
                self.flush_strip_with_stop(&stop)?;
            }
        }

        // Step 2: Process complete strips directly from input (no copy!)
        while rows_remaining >= self.strip_height {
            stop.check()?;

            let remaining_height = self.height - self.current_y;
            let strip_rows = self.strip_height.min(remaining_height);

            if strip_rows == 0 {
                break;
            }

            let strip_bytes = strip_rows * self.bytes_per_row;
            let strip_data = &data[data_offset..data_offset + strip_bytes];

            // Process directly from input buffer
            self.processor.process_strip(strip_data, self.current_y)?;
            self.current_y += strip_rows;

            data_offset += strip_bytes;
            rows_remaining -= strip_rows;
        }

        // Step 3: Buffer any remaining partial rows
        if rows_remaining > 0 {
            let remaining_height = self.height - self.current_y;
            let rows_to_buffer = rows_remaining.min(remaining_height);

            if rows_to_buffer > 0 {
                let src_bytes = rows_to_buffer * self.bytes_per_row;
                self.row_buffer[..src_bytes]
                    .copy_from_slice(&data[data_offset..data_offset + src_bytes]);
                self.rows_buffered = rows_to_buffer;

                // Check if this is the final partial strip
                if rows_to_buffer >= remaining_height {
                    self.flush_strip_with_stop(&stop)?;
                }
            }
        }

        Ok(())
    }

    /// Pushes a strip of YCbCr f32 planar data.
    ///
    /// This bypasses RGB→YCbCr conversion, accepting YCbCr data directly.
    /// Values should be in centered range [-128, 127].
    ///
    /// # Arguments
    /// * `y` - Y plane data (width × num_rows floats)
    /// * `cb` - Cb plane data (width × num_rows floats, full resolution)
    /// * `cr` - Cr plane data (width × num_rows floats, full resolution)
    /// * `num_rows` - Number of rows in this strip
    ///
    /// # Note
    ///
    /// Unlike `push_row` which buffers internally, this method processes
    /// the strip immediately. For optimal performance, push `strip_height()`
    /// rows at a time.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - RGB rows are already buffered (can't mix RGB and YCbCr input)
    /// - Plane sizes don't match expected dimensions
    /// - XYB mode is enabled (requires RGB input)
    pub(crate) fn push_ycbcr_strip_f32(
        &mut self,
        y: &[f32],
        cb: &[f32],
        cr: &[f32],
        num_rows: usize,
    ) -> Result<()> {
        // Can't mix RGB and YCbCr input
        if self.rows_buffered > 0 {
            return Err(Error::internal(
                "cannot mix RGB and YCbCr input (RGB rows buffered)",
            ));
        }

        // Validate we haven't received all rows yet
        if self.current_y >= self.height {
            return Err(Error::io_error(format!(
                "already received all {} rows",
                self.height
            )));
        }

        // Clamp to remaining rows
        let actual_rows = num_rows.min(self.height - self.current_y);

        // Validate plane sizes
        let expected_size = self.width * actual_rows;
        if y.len() < expected_size {
            return Err(Error::invalid_buffer_size(expected_size, y.len()));
        }
        if cb.len() < expected_size || cr.len() < expected_size {
            return Err(Error::invalid_buffer_size(
                expected_size,
                cb.len().min(cr.len()),
            ));
        }

        // Process in chunks of strip_height rows
        let mut processed = 0;
        while processed < actual_rows {
            let remaining = self.height - self.current_y;
            let strip_rows = self
                .strip_height
                .min(actual_rows - processed)
                .min(remaining);

            let start = processed * self.width;
            let end = start + strip_rows * self.width;

            self.processor.process_strip_ycbcr_f32(
                &y[start..end],
                &cb[start..end],
                &cr[start..end],
                self.current_y,
            )?;

            self.current_y += strip_rows;
            processed += strip_rows;
        }

        Ok(())
    }

    /// Pushes a strip of pre-downsampled YCbCr f32 planar data.
    ///
    /// This accepts chroma data that is already downsampled according to the
    /// subsampling mode. Skips the internal chroma downsampling step.
    ///
    /// # Arguments
    /// * `y` - Y plane data (width × num_rows floats)
    /// * `cb` - Cb plane data (chroma_width × chroma_rows floats)
    /// * `cr` - Cr plane data (chroma_width × chroma_rows floats)
    /// * `num_rows` - Number of Y rows in this strip
    ///
    /// # Chroma Dimensions
    /// - 4:4:4: cb/cr at full width × full height
    /// - 4:2:2: cb/cr at width/2 × full height
    /// - 4:2:0: cb/cr at width/2 × height/2
    pub(crate) fn push_ycbcr_strip_f32_subsampled(
        &mut self,
        y: &[f32],
        cb: &[f32],
        cr: &[f32],
        num_rows: usize,
    ) -> Result<()> {
        // Can't mix RGB and YCbCr input
        if self.rows_buffered > 0 {
            return Err(Error::internal(
                "cannot mix RGB and YCbCr input (RGB rows buffered)",
            ));
        }

        // Validate we haven't received all rows yet
        if self.current_y >= self.height {
            return Err(Error::io_error(format!(
                "already received all {} rows",
                self.height
            )));
        }

        // Clamp to remaining rows
        let actual_rows = num_rows.min(self.height - self.current_y);

        // Validate Y plane size
        let expected_y_size = self.width * actual_rows;
        if y.len() < expected_y_size {
            return Err(Error::invalid_buffer_size(expected_y_size, y.len()));
        }

        // Get subsampling info for chroma slicing
        let subsampling = self.processor.subsampling();
        let chroma_width = match subsampling {
            Subsampling::S444 | Subsampling::S440 => self.width,
            Subsampling::S422 | Subsampling::S420 => (self.width + 1) / 2,
        };
        let chroma_h_factor = match subsampling {
            Subsampling::S444 | Subsampling::S422 => 1,
            Subsampling::S420 | Subsampling::S440 => 2,
        };

        // Process in chunks of strip_height rows
        let mut y_processed = 0;
        let mut chroma_processed = 0;
        while y_processed < actual_rows {
            let remaining = self.height - self.current_y;
            let strip_rows = self
                .strip_height
                .min(actual_rows - y_processed)
                .min(remaining);

            let y_start = y_processed * self.width;
            let y_end = y_start + strip_rows * self.width;

            // Calculate chroma rows for this strip
            let chroma_rows = (strip_rows + chroma_h_factor - 1) / chroma_h_factor;
            let c_start = chroma_processed * chroma_width;
            let c_end = c_start + chroma_rows * chroma_width;

            self.processor.process_strip_ycbcr_f32_subsampled(
                &y[y_start..y_end],
                &cb[c_start..c_end.min(cb.len())],
                &cr[c_start..c_end.min(cr.len())],
                self.current_y,
            )?;

            self.current_y += strip_rows;
            y_processed += strip_rows;
            chroma_processed += chroma_rows;
        }

        Ok(())
    }

    /// Flushes the current strip buffer to the processor.
    fn flush_strip_with_stop(&mut self, stop: &impl Stop) -> Result<()> {
        stop.check()?;

        if self.rows_buffered == 0 {
            return Ok(());
        }

        let strip_data = &self.row_buffer[..self.rows_buffered * self.bytes_per_row];
        self.processor.process_strip(strip_data, self.current_y)?;

        self.current_y += self.rows_buffered;
        self.rows_buffered = 0;

        Ok(())
    }

    /// Finishes encoding and returns the JPEG data.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Not all rows have been pushed
    /// - JPEG generation fails
    pub(crate) fn finish(self) -> Result<Vec<u8>> {
        self.finish_with_stop(Unstoppable)
    }

    /// Finishes encoding with cancellation support.
    pub(crate) fn finish_with_stop(mut self, stop: impl Stop) -> Result<Vec<u8>> {
        stop.check()?;

        // Calculate total rows received
        let total_rows = self.current_y + self.rows_buffered;

        // Validate all rows were pushed before trying to process
        if total_rows < self.height {
            return Err(Error::io_error(format!(
                "only {} of {} rows were pushed",
                total_rows, self.height
            )));
        }

        // Flush any remaining rows
        if self.rows_buffered > 0 {
            self.flush_strip_with_stop(&stop)?;
        }

        // Extract needed data before consuming processor
        let config = self.config;
        let y_quant = self.y_quant;
        let cb_quant = self.cb_quant;
        let cr_quant = self.cr_quant;
        let width = self.width;
        let height = self.height;

        // Finalize strip processing
        let strip_output = self.processor.finalize()?;

        // Build JPEG output using the config's internal methods
        Self::build_jpeg_from_blocks(
            &config,
            &y_quant,
            &cb_quant,
            &cr_quant,
            width,
            height,
            strip_output,
            stop,
        )
    }

    /// Builds JPEG output from processed blocks.
    fn build_jpeg_from_blocks(
        config: &ComputedConfig,
        y_quant: &QuantTable,
        cb_quant: &QuantTable,
        cr_quant: &QuantTable,
        _width: usize,
        _height: usize,
        strip_output: crate::encode::strip::StripProcessorOutput,
        stop: impl Stop,
    ) -> Result<Vec<u8>> {
        stop.check()?;

        // Branch based on encoding mode (mirrors encode_strip_based in encode/mod.rs)
        match config.mode {
            JpegMode::Progressive => {
                // Progressive mode requires optimized Huffman tables
                if !config.optimize_huffman {
                    return Err(Error::unsupported_feature(
                        "Progressive mode with fixed Huffman codes (use optimize_huffman=true)",
                    ));
                }
                // Use progressive encoding path
                config.encode_progressive_from_blocks(
                    &strip_output.y_blocks,
                    &strip_output.cb_blocks,
                    &strip_output.cr_blocks,
                    y_quant,
                    cb_quant,
                    cr_quant,
                )
            }
            _ => {
                // Baseline encoding
                Self::build_jpeg_baseline(config, y_quant, cb_quant, cr_quant, strip_output)
            }
        }
    }

    /// Builds baseline JPEG output from processed blocks.
    fn build_jpeg_baseline(
        config: &ComputedConfig,
        y_quant: &QuantTable,
        cb_quant: &QuantTable,
        cr_quant: &QuantTable,
        strip_output: crate::encode::strip::StripProcessorOutput,
    ) -> Result<Vec<u8>> {
        let is_color = !config.pixel_format.is_grayscale();
        let width = config.width as usize;
        let height = config.height as usize;

        let mut output = Vec::with_capacity(width * height / 4);

        // Branch based on XYB vs YCbCr mode
        let scan_data = if config.use_xyb {
            // XYB mode: uses different headers, tables, and encoding
            config.write_header_xyb(&mut output)?;
            config.write_app14_adobe(&mut output, 0)?;
            config.write_icc_profile(&mut output, &crate::foundation::consts::XYB_ICC_PROFILE)?;
            config.write_quant_tables_xyb(&mut output, y_quant, cb_quant, cr_quant)?;
            // Use SOF1 if any quant table needs 16-bit precision
            let is_extended =
                y_quant.precision > 0 || cb_quant.precision > 0 || cr_quant.precision > 0;
            config.write_frame_header_xyb_ex(&mut output, is_extended)?;

            if config.optimize_huffman {
                let (dc_table, ac_table) = config.build_optimized_tables_xyb_raster(
                    &strip_output.y_blocks,
                    &strip_output.cb_blocks,
                    &strip_output.cr_blocks,
                )?;

                config.write_huffman_tables_xyb_optimized(&mut output, &dc_table, &ac_table);

                if config.restart_interval > 0 {
                    config.write_restart_interval(&mut output)?;
                }
                config.write_scan_header_xyb(&mut output)?;

                config.encode_with_tables_xyb_raster(
                    &strip_output.y_blocks,
                    &strip_output.cb_blocks,
                    &strip_output.cr_blocks,
                    &dc_table,
                    &ac_table,
                )?
            } else {
                config.write_huffman_tables(&mut output)?;

                if config.restart_interval > 0 {
                    config.write_restart_interval(&mut output)?;
                }
                config.write_scan_header_xyb(&mut output)?;

                config.encode_with_tables_xyb_standard_raster(
                    &strip_output.y_blocks,
                    &strip_output.cb_blocks,
                    &strip_output.cr_blocks,
                )?
            }
        } else {
            // YCbCr mode: standard JPEG encoding
            config.write_header(&mut output)?;
            config.write_quant_tables(&mut output, y_quant, cb_quant, cr_quant)?;
            // Use SOF1 if any quant table needs 16-bit precision
            let is_extended =
                y_quant.precision > 0 || cb_quant.precision > 0 || cr_quant.precision > 0;
            config.write_frame_header_ex(&mut output, is_extended)?;

            if config.optimize_huffman {
                let tables = config.build_optimized_tables(
                    &strip_output.y_blocks,
                    &strip_output.cb_blocks,
                    &strip_output.cr_blocks,
                    is_color,
                )?;

                config.write_huffman_tables_optimized(&mut output, &tables)?;

                if config.restart_interval > 0 {
                    config.write_restart_interval(&mut output)?;
                }
                config.write_scan_header(&mut output)?;

                config.encode_with_tables(
                    &strip_output.y_blocks,
                    &strip_output.cb_blocks,
                    &strip_output.cr_blocks,
                    is_color,
                    Some(&tables),
                )?
            } else {
                config.write_huffman_tables(&mut output)?;

                if config.restart_interval > 0 {
                    config.write_restart_interval(&mut output)?;
                }
                config.write_scan_header(&mut output)?;

                config.encode_with_tables(
                    &strip_output.y_blocks,
                    &strip_output.cb_blocks,
                    &strip_output.cr_blocks,
                    is_color,
                    None,
                )?
            }
        };

        output.extend_from_slice(&scan_data);

        // Write EOI marker
        output.push(0xFF);
        output.push(crate::foundation::consts::MARKER_EOI);

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_encoder_creation() {
        let encoder = StreamingEncoder::new(640, 480).start();
        assert!(encoder.is_ok());
        let encoder = encoder.unwrap();
        assert_eq!(encoder.height(), 480);
        assert_eq!(encoder.bytes_per_row(), 640 * 3); // RGB default
    }

    #[test]
    fn test_streaming_encoder_420_strip_height() {
        let encoder = StreamingEncoder::new(640, 480)
            .subsampling(Subsampling::S420)
            .start()
            .unwrap();
        assert_eq!(encoder.strip_height(), 16);
    }

    #[test]
    fn test_streaming_encoder_444_strip_height() {
        let encoder = StreamingEncoder::new(640, 480)
            .subsampling(Subsampling::S444)
            .start()
            .unwrap();
        assert_eq!(encoder.strip_height(), 8);
    }

    #[test]
    fn test_streaming_encoder_wrong_row_size() {
        let mut encoder = StreamingEncoder::new(640, 480).start().unwrap();
        let wrong_row = vec![0u8; 100]; // Wrong size
        let result = encoder.push_row(&wrong_row);
        assert!(result.is_err());
    }

    #[test]
    fn test_streaming_encoder_too_many_rows() {
        let mut encoder = StreamingEncoder::new(4, 2).start().unwrap();
        let row = vec![128u8; 4 * 3]; // 4 pixels * 3 channels

        // Push first 2 rows (all of them)
        encoder.push_row(&row).unwrap();
        encoder.push_row(&row).unwrap();

        // Third row should fail
        let result = encoder.push_row(&row);
        assert!(result.is_err());
    }

    #[test]
    fn test_streaming_encoder_incomplete() {
        let mut encoder = StreamingEncoder::new(4, 4).start().unwrap();
        let row = vec![128u8; 4 * 3];

        // Push only 2 of 4 rows
        encoder.push_row(&row).unwrap();
        encoder.push_row(&row).unwrap();

        // finish() should fail
        let result = encoder.finish();
        assert!(result.is_err());
    }

    #[test]
    fn test_memory_estimate() {
        let estimate = StreamingEncoder::new(3840, 2160)
            .subsampling(Subsampling::S420)
            .estimate_memory_usage();

        // Should be around 67 MB for 4K with 4:2:0 (includes token buffer)
        // Allow some tolerance for implementation details
        assert!(estimate > 60_000_000, "estimate {} too low", estimate);
        assert!(estimate < 80_000_000, "estimate {} too high", estimate);
    }

    #[test]
    fn test_streaming_matches_oneshot() {
        // Create a small test image
        let width = 32u32;
        let height = 32u32;
        let pixels: Vec<u8> = (0..width * height * 3)
            .map(|i| ((i * 17) % 256) as u8)
            .collect();

        // Encode with one-shot method
        let oneshot_result = StreamingEncoder::new(width, height)
            .quality(Quality::ApproxJpegli(85.0))
            .subsampling(Subsampling::S444)
            .encode(&pixels)
            .unwrap();

        // Encode with streaming encoder (row by row)
        let mut streaming = StreamingEncoder::new(width, height)
            .quality(Quality::ApproxJpegli(85.0))
            .subsampling(Subsampling::S444)
            .start()
            .unwrap();

        let row_size = width as usize * 3;
        for y in 0..height as usize {
            let start = y * row_size;
            let end = start + row_size;
            streaming.push_row(&pixels[start..end]).unwrap();
        }
        let streaming_result = streaming.finish().unwrap();

        // Results should be identical
        assert_eq!(
            oneshot_result.len(),
            streaming_result.len(),
            "output lengths differ"
        );
        assert_eq!(oneshot_result, streaming_result, "outputs differ");
    }
}
