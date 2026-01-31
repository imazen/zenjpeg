//! Builder for creating a streaming encoder.
//!
//! Split from `streaming.rs` for readability. The builder configures encoding
//! parameters; the actual encoder lives in [`super::streaming::StreamingEncoder`].

#![allow(dead_code)]

use super::encoder_types::DownsamplingMethod;
use super::encoder_types::Quality;
use super::streaming::StreamingEncoder;
use crate::encode::tuning::EncodingTables;
use crate::error::Result;
use crate::types::{JpegMode, PixelFormat, Subsampling};

/// Builder for creating a streaming encoder.
///
/// Use [`StreamingEncoder::new()`] to start building.
#[derive(Debug, Clone)]
pub(crate) struct StreamingEncoderBuilder {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) quality: Quality,
    pub(crate) subsampling: Subsampling,
    pub(crate) pixel_format: PixelFormat,
    pub(crate) mode: JpegMode,
    pub(crate) optimize_huffman: bool,
    pub(crate) chroma_downsampling: DownsamplingMethod,
    pub(crate) restart_interval: u16,
    /// Custom encoding tables (quantization + zero-bias).
    /// `None` means use perceptual defaults based on color mode and quality.
    pub(crate) encoding_tables: Option<Box<EncodingTables>>,
    pub(crate) use_xyb: bool,
    /// Enable mozjpeg-style overshoot deringing (on by default)
    pub(crate) deringing: bool,
    /// Allow 16-bit quantization tables (default: true)
    pub(crate) allow_16bit_quant_tables: bool,
    /// Use separate Cb and Cr quantization tables (default: true = 3 tables)
    pub(crate) separate_chroma_tables: bool,
    /// Custom Huffman tables for streaming-through encoding.
    /// When set, blocks are encoded immediately on each strip flush
    /// instead of buffering all blocks for optimized table generation.
    pub(crate) custom_huffman_tables: Option<crate::huffman::optimize::OptimizedHuffmanTables>,
    /// Enable parallel encoding (requires `parallel` feature)
    #[cfg(feature = "parallel")]
    pub(crate) parallel: bool,
    /// Hybrid quantization configuration (requires `experimental-hybrid-trellis` feature)
    #[cfg(feature = "experimental-hybrid-trellis")]
    pub(crate) hybrid_config: crate::hybrid::config::HybridConfig,
    /// Custom AQ map (requires `experimental-hybrid-trellis` feature)
    #[cfg(feature = "experimental-hybrid-trellis")]
    pub(crate) custom_aq_map: Option<crate::quant::aq::AQStrengthMap>,
    /// Trellis quantization config (mozjpeg-compat API, requires `experimental-hybrid-trellis`)
    #[cfg(feature = "experimental-hybrid-trellis")]
    pub(crate) trellis: Option<super::mozjpeg_compat::TrellisConfig>,
}

impl StreamingEncoderBuilder {
    /// Creates a new streaming encoder builder with default settings.
    pub(crate) fn new(width: u32, height: u32) -> Self {
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
            custom_huffman_tables: None,
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
    /// Has no effect for 4:4:4 subsampling (no downsampling needed).
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

    /// Sets custom Huffman tables for streaming-through encoding.
    ///
    /// When provided, blocks are entropy-encoded immediately on each strip flush
    /// using these tables, instead of buffering all blocks for a two-pass optimized
    /// table generation. This enables true single-pass encoding with bounded memory.
    ///
    /// Custom tables can come from [`crate::huffman::trained`] (pre-trained on image
    /// corpora) or from a previous encoding pass via [`crate::huffman::optimize::FrequencyCounter`].
    #[must_use]
    pub(crate) fn custom_huffman_tables(
        mut self,
        tables: crate::huffman::optimize::OptimizedHuffmanTables,
    ) -> Self {
        self.custom_huffman_tables = Some(tables);
        self
    }

    /// Enables XYB color space encoding.
    ///
    /// XYB is a perceptual color space used by JPEG XL that better models human
    /// vision than YCbCr. When enabled, the output JPEG uses XYB-encoded data
    /// with an ICC profile that allows compatible decoders to render correctly.
    #[must_use]
    pub(crate) fn use_xyb(mut self, enable: bool) -> Self {
        self.use_xyb = enable;
        self
    }

    /// Enables mozjpeg-style overshoot deringing.
    ///
    /// This reduces visible ringing artifacts near sharp edges, particularly
    /// on white backgrounds.
    ///
    /// Enabled by default.
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
    /// Requires the `experimental-hybrid-trellis` feature.
    #[cfg(feature = "experimental-hybrid-trellis")]
    #[must_use]
    pub(crate) fn hybrid_config(mut self, config: crate::hybrid::config::HybridConfig) -> Self {
        self.hybrid_config = config;
        self
    }

    /// Sets trellis quantization configuration (mozjpeg-compatible API).
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
    pub(crate) fn start(self) -> Result<StreamingEncoder> {
        StreamingEncoder::from_builder(self)
    }

    /// Encodes a complete image buffer in one call.
    ///
    /// This is the simplest way to encode an image. For streaming scenarios
    /// where you want to push rows incrementally, use `.start()` instead.
    pub(crate) fn encode(self, data: &[u8]) -> Result<Vec<u8>> {
        let width = self.width as usize;
        let height = self.height as usize;
        let bpp = self.pixel_format.bytes_per_pixel();
        let expected_size = width * height * bpp;

        if data.len() != expected_size {
            return Err(crate::error::Error::invalid_buffer_size(
                expected_size,
                data.len(),
            ));
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
    pub(crate) fn encode_with_stop(
        self,
        data: &[u8],
        stop: impl enough::Stop,
    ) -> Result<Vec<u8>> {
        let width = self.width as usize;
        let height = self.height as usize;
        let bpp = self.pixel_format.bytes_per_pixel();
        let expected_size = width * height * bpp;

        if data.len() != expected_size {
            return Err(crate::error::Error::invalid_buffer_size(
                expected_size,
                data.len(),
            ));
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
    /// subsampling mode, and pixel format.
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

        // 7. Entropy encoder output buffer (baseline mode)
        let total_blocks = y_block_count + c_block_count * 2;
        let entropy_output = total_blocks * 3;

        // 8. Output buffer estimate (grows during encoding)
        let output_estimate = width * height / 8;

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
            + entropy_output
            + output_estimate
    }

    /// Returns an absolute ceiling on memory usage.
    ///
    /// Unlike [`estimate_memory_usage`], this returns a **guaranteed upper bound**
    /// that actual peak memory will never exceed.
    #[must_use]
    pub(crate) fn estimate_memory_ceiling(&self) -> usize {
        let width = self.width as usize;
        let height = self.height as usize;

        let strip_height = match self.subsampling {
            Subsampling::S420 | Subsampling::S440 => 16,
            _ => 8,
        };

        let mcu_size = self.subsampling.mcu_size();
        let padded_width = (width + mcu_size - 1) / mcu_size * mcu_size;
        let padded_height = (height + mcu_size - 1) / mcu_size * mcu_size;

        let (c_width, c_strip_height) = match self.subsampling {
            Subsampling::S420 => ((padded_width + 1) / 2, strip_height / 2),
            Subsampling::S422 => ((padded_width + 1) / 2, strip_height),
            Subsampling::S440 => (padded_width, strip_height / 2),
            Subsampling::S444 => (padded_width, strip_height),
        };
        let padded_c_width = (c_width + 7) / 8 * 8;

        let y_blocks_w = padded_width / 8;
        let y_blocks_h = padded_height / 8;
        let y_block_count = y_blocks_w * y_blocks_h;

        let c_block_count = match self.subsampling {
            Subsampling::S420 => (padded_width / 16) * (padded_height / 16),
            Subsampling::S422 => (padded_width / 16) * y_blocks_h,
            Subsampling::S440 => y_blocks_w * (padded_height / 16),
            Subsampling::S444 => y_block_count,
        };

        let max_bpp = 4;
        let row_buffer = padded_width * strip_height * max_bpp;

        let strip_y = padded_width * strip_height * 4;
        let strip_cb = padded_width * strip_height * 4;
        let strip_cr = padded_width * strip_height * 4;

        let strip_cb_down = padded_c_width * c_strip_height * 4;
        let strip_cr_down = padded_c_width * c_strip_height * 4;

        let padded_y_blocks_per_row = padded_width / 8;
        let v_samp = match self.subsampling {
            Subsampling::S420 | Subsampling::S440 => 2,
            _ => 1,
        };
        let pending_y_capacity = padded_y_blocks_per_row * v_samp;
        let padded_c_blocks_per_row = padded_c_width / 8;
        let pending_c_capacity = padded_c_blocks_per_row;

        let pending_y_f32 = 2 * pending_y_capacity * 256;
        let pending_cb_f32 = 2 * pending_c_capacity * 256;
        let pending_cr_f32 = 2 * pending_c_capacity * 256;

        let y_blocks_i16 = y_block_count * 128;
        let c_blocks_i16 = c_block_count * 2 * 128;

        let aq_strengths = y_block_count * 4;

        let total_blocks = y_block_count + c_block_count * 2;
        let entropy_output = total_blocks * 10;

        let output_ceiling = padded_width * padded_height;

        let huffman_tables = 4 * 256 * 8;
        let scan_overhead = 64 * 8;

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
            + entropy_output
            + output_ceiling
            + huffman_tables
            + scan_overhead;

        // Add 5% allocator overhead ceiling
        subtotal + subtotal / 20
    }
}
