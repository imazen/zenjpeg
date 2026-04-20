//! Builder for creating a streaming encoder.
//!
//! Split from `streaming.rs` for readability. The builder configures encoding
//! parameters; the actual encoder lives in `super::streaming::StreamingEncoder`.

use super::encoder_types::DownsamplingMethod;
use super::encoder_types::HuffmanStrategy;
use super::encoder_types::Quality;
use super::encoder_types::QuantTableSource;
use super::encoder_types::ScanStrategy;
use super::layout::LayoutParams;
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
    pub(crate) huffman: HuffmanStrategy,
    pub(crate) chroma_downsampling: DownsamplingMethod,
    pub(crate) restart_interval: u16,
    /// Custom encoding tables (quantization + zero-bias).
    /// `None` means use perceptual defaults based on color mode and quality.
    pub(crate) encoding_tables: Option<Box<EncodingTables>>,
    pub(crate) use_xyb: bool,
    /// XYB B-channel layout (BQuarter or Full). Ignored when `use_xyb` is false.
    pub(crate) xyb_subsampling: super::encoder_types::XybSubsampling,
    /// Enable mozjpeg-style overshoot deringing (on by default)
    pub(crate) deringing: bool,
    /// Enable adaptive quantization (jpegli AQ). On by default.
    pub(crate) aq_enabled: bool,
    /// Enable boundary-continuity refinement (Phase 2 of #91). Off by default.
    pub(crate) boundary_rd: bool,
    /// Seam-jump weight (α) for boundary-continuity D_b term.
    pub(crate) boundary_rd_alpha: f32,
    /// D_b trigger threshold as multiplier of per-block AC DCT energy.
    pub(crate) boundary_rd_threshold: f32,
    /// AQ-strength multiplier applied on each boundary-RD retry.
    pub(crate) boundary_rd_shrink: f32,
    /// Max number of boundary-RD refinement retries per triggered block.
    pub(crate) boundary_rd_max_retries: u8,
    /// Enable above-neighbor (top-edge) boundary-RD (Phase 4 of #91). Off
    /// by default. A no-op unless `boundary_rd` is also true.
    pub(crate) boundary_rd_above: bool,
    /// Allow 16-bit quantization tables (default: false)
    pub(crate) allow_16bit_quant_tables: bool,
    /// Force SOF1 (extended sequential) regardless of quant table precision.
    /// Required for XYB (DC categories can exceed baseline limit of 11).
    pub(crate) force_sof1: bool,
    /// Use separate Cb and Cr quantization tables (default: true = 3 tables)
    pub(crate) separate_chroma_tables: bool,
    /// Progressive scan script strategy
    pub(crate) scan_strategy: ScanStrategy,
    /// Enable parallel encoding (requires `parallel` feature)
    #[cfg(feature = "parallel")]
    pub(crate) parallel: bool,
    /// Hybrid quantization configuration
    #[cfg(feature = "trellis")]
    pub(crate) hybrid_config: super::trellis::HybridConfig,
    /// Custom AQ map
    pub(crate) custom_aq_map: Option<crate::quant::aq::AQStrengthMap>,
    /// Trellis quantization config (mozjpeg-compat API)
    #[cfg(feature = "trellis")]
    pub(crate) trellis: Option<super::trellis::TrellisConfig>,
    /// Source of quantization tables (jpegli perceptual vs mozjpeg Robidoux).
    /// Only used when `encoding_tables` is `None` (no custom tables).
    pub(crate) quant_source: QuantTableSource,
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
            huffman: HuffmanStrategy::Optimize,
            chroma_downsampling: DownsamplingMethod::Box,
            restart_interval: 0,
            encoding_tables: None,
            use_xyb: false,
            xyb_subsampling: super::encoder_types::XybSubsampling::BQuarter,
            deringing: true,
            aq_enabled: true,
            boundary_rd: false,
            // Phase 5 tuned defaults — see EncoderConfig::default_internal.
            boundary_rd_alpha: 1.0,
            boundary_rd_threshold: 0.05,
            boundary_rd_shrink: 0.5,
            boundary_rd_max_retries: 2,
            boundary_rd_above: false,
            allow_16bit_quant_tables: false,
            force_sof1: false,
            separate_chroma_tables: true,
            scan_strategy: ScanStrategy::Default,
            #[cfg(feature = "parallel")]
            parallel: false,
            #[cfg(feature = "trellis")]
            hybrid_config: super::trellis::HybridConfig::disabled(),
            custom_aq_map: None,
            #[cfg(feature = "trellis")]
            trellis: None,
            quant_source: QuantTableSource::default(),
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
            self.huffman = HuffmanStrategy::Optimize;
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

    /// Sets the Huffman table strategy.
    #[must_use]
    pub(crate) fn huffman(mut self, strategy: HuffmanStrategy) -> Self {
        self.huffman = strategy;
        self
    }

    /// Sets chroma downsampling method for subsampled modes.
    #[must_use]
    pub(crate) fn chroma_downsampling(mut self, method: DownsamplingMethod) -> Self {
        self.chroma_downsampling = method;
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

    /// Sets the XYB B-channel subsampling layout. Ignored unless `use_xyb` is set.
    #[must_use]
    pub(crate) fn xyb_subsampling(mut self, sub: super::encoder_types::XybSubsampling) -> Self {
        self.xyb_subsampling = sub;
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

    /// Enables or disables adaptive quantization (jpegli AQ).
    ///
    /// When disabled, AQ computation is skipped entirely and all blocks
    /// receive neutral AQ strength (0.0).
    #[must_use]
    pub(crate) fn aq_enabled(mut self, enable: bool) -> Self {
        self.aq_enabled = enable;
        self
    }

    /// Enables boundary-continuity refinement (Phase 2 of #91). Off by default.
    #[must_use]
    pub(crate) fn boundary_rd(mut self, enable: bool) -> Self {
        self.boundary_rd = enable;
        self
    }

    /// Set the α (seam-jump weight) for boundary-continuity D_b.
    #[must_use]
    pub(crate) fn boundary_rd_alpha(mut self, alpha: f32) -> Self {
        self.boundary_rd_alpha = alpha;
        self
    }

    /// Set the D_b trigger threshold (multiplier of per-block AC DCT energy).
    #[must_use]
    pub(crate) fn boundary_rd_threshold(mut self, threshold: f32) -> Self {
        self.boundary_rd_threshold = threshold;
        self
    }

    /// Set the AQ-strength multiplier applied on each boundary-RD retry.
    #[must_use]
    pub(crate) fn boundary_rd_shrink(mut self, shrink: f32) -> Self {
        self.boundary_rd_shrink = shrink;
        self
    }

    /// Set the max number of boundary-RD refinement retries per triggered block.
    #[must_use]
    pub(crate) fn boundary_rd_max_retries(mut self, retries: u8) -> Self {
        self.boundary_rd_max_retries = retries;
        self
    }

    /// Enable above-neighbor (top-edge) boundary-RD (Phase 4 of #91).
    #[must_use]
    pub(crate) fn boundary_rd_above(mut self, enable: bool) -> Self {
        self.boundary_rd_above = enable;
        self
    }

    /// Allow 16-bit quantization tables for better low-quality precision.
    ///
    /// When enabled, quantization values can exceed 255, using 16-bit DQT
    /// markers. When any table exceeds 255, SOF1 is used automatically.
    ///
    /// When disabled, quantization values are clamped to 255 (8-bit DQT).
    /// SOF0 is used unless `force_sof1` is set.
    #[must_use]
    pub(crate) fn allow_16bit_quant_tables(mut self, enable: bool) -> Self {
        self.allow_16bit_quant_tables = enable;
        self
    }

    /// Force SOF1 (extended sequential) regardless of quant table precision.
    ///
    /// Required for XYB color space, where DC categories can exceed the
    /// baseline limit of 11 due to the wider dynamic range.
    #[must_use]
    pub(crate) fn force_sof1(mut self, enable: bool) -> Self {
        self.force_sof1 = enable;
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

    /// Sets the progressive scan script strategy.
    #[must_use]
    pub(crate) fn scan_strategy(mut self, strategy: ScanStrategy) -> Self {
        self.scan_strategy = strategy;
        self
    }

    /// Sets custom hybrid quantization configuration.
    #[cfg(feature = "trellis")]
    #[must_use]
    pub(crate) fn hybrid_config(mut self, config: super::trellis::HybridConfig) -> Self {
        self.hybrid_config = config;
        self
    }

    /// Sets trellis quantization configuration (mozjpeg-compatible API).
    #[cfg(feature = "trellis")]
    #[must_use]
    pub fn trellis(mut self, config: super::trellis::TrellisConfig) -> Self {
        self.trellis = Some(config);
        self
    }

    /// Sets the quantization table source.
    #[must_use]
    pub(crate) fn quant_source(mut self, source: QuantTableSource) -> Self {
        self.quant_source = source;
        self
    }

    /// Sets a custom AQ (adaptive quantization) strength map.
    #[must_use]
    #[allow(dead_code)] // Internal API; EncoderConfig has its own wrapper
    pub(crate) fn aq_map(mut self, map: crate::quant::aq::AQStrengthMap) -> Self {
        self.custom_aq_map = Some(map);
        self
    }

    /// Starts a streaming encoder for row-by-row input.
    pub(crate) fn start(self) -> Result<StreamingEncoder> {
        StreamingEncoder::from_builder(self)
    }

    /// Convenience: set `HuffmanStrategy::Optimize` or `HuffmanStrategy::Fixed`.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn optimize_huffman(mut self, enable: bool) -> Self {
        self.huffman = if enable {
            HuffmanStrategy::Optimize
        } else {
            HuffmanStrategy::Fixed
        };
        self
    }

    /// Convenience: wrap a `HuffmanTableSet` in `HuffmanStrategy::Custom`.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn custom_huffman_tables(
        mut self,
        tables: crate::huffman::optimize::HuffmanTableSet,
    ) -> Self {
        self.huffman = HuffmanStrategy::Custom(Box::new(tables));
        self
    }

    /// One-shot encode: push the whole buffer and finish.
    #[cfg(test)]
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

    /// Estimates the peak memory usage for this configuration.
    ///
    /// Returns the estimated peak memory in bytes based on image dimensions,
    /// subsampling mode, and pixel format. Uses `LayoutParams` for all geometry
    /// to correctly handle XYB mode (strip_height=16, v_samp=2).
    #[must_use]
    pub(crate) fn estimate_memory_usage(&self) -> usize {
        let lp = LayoutParams::new_xyb(
            self.width as usize,
            self.height as usize,
            self.subsampling,
            self.use_xyb,
            self.xyb_subsampling,
        );

        let y_block_count = lp.total_y_blocks;
        let c_block_count = lp.total_c_blocks;

        // 1. Row buffer for input (one strip's worth)
        let bpp = self.pixel_format.bytes_per_pixel();
        let row_buffer = lp.width * lp.strip_height * bpp;

        // 2. Strip f32 buffers (Y, Cb, Cr at full resolution before downsampling)
        let strip_y = lp.padded_width * lp.strip_height * 4; // f32 = 4 bytes
        let strip_cb = lp.padded_width * lp.strip_height * 4;
        let strip_cr = lp.padded_width * lp.strip_height * 4;

        // 3. Downsampled chroma temp buffers
        let strip_cb_down = lp.padded_c_width * lp.c_strip_height * 4;
        let strip_cr_down = lp.padded_c_width * lp.c_strip_height * 4;

        // 4. Pending f32 DCT blocks (double-buffered, 2 iMCU rows)
        // 256 bytes per f32 block, 2 buffers (double-buffered)
        let pending_y_f32 = 2 * lp.pending_y_capacity * 256;
        let pending_cb_f32 = 2 * lp.pending_c_capacity * 256;
        let pending_cr_f32 = 2 * lp.pending_c_capacity * 256;

        // 5. Final i16 blocks (128 bytes per block)
        let y_blocks_i16 = y_block_count * 128;
        let c_blocks_i16 = c_block_count * 2 * 128; // Cb + Cr

        // 6. AQ strengths (one f32 per Y block)
        let aq_strengths = y_block_count * 4;

        // 7. Entropy encoder output buffer (baseline mode)
        let total_blocks = y_block_count + c_block_count * 2;
        let entropy_output = total_blocks * 3;

        // 8. Output buffer estimate (grows during encoding)
        let output_estimate = lp.width * lp.height / 8;

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
    /// that actual peak memory will never exceed. Uses `LayoutParams` for all geometry
    /// to correctly handle XYB mode.
    #[must_use]
    pub(crate) fn estimate_memory_ceiling(&self) -> usize {
        let lp = LayoutParams::new_xyb(
            self.width as usize,
            self.height as usize,
            self.subsampling,
            self.use_xyb,
            self.xyb_subsampling,
        );

        // Use padded dimensions for ceiling (worst case)
        let padded_height = (lp.height + lp.mcu_size - 1) / lp.mcu_size * lp.mcu_size;
        let y_blocks_w_padded = lp.padded_width / 8;
        let y_blocks_h_padded = padded_height / 8;
        let y_block_count = y_blocks_w_padded * y_blocks_h_padded;

        let c_block_count = match lp.subsampling {
            Subsampling::S420 => (lp.padded_width / 16) * (padded_height / 16),
            Subsampling::S422 => (lp.padded_width / 16) * y_blocks_h_padded,
            Subsampling::S440 => y_blocks_w_padded * (padded_height / 16),
            Subsampling::S444 => y_block_count,
        };

        let max_bpp = 4;
        let row_buffer = lp.padded_width * lp.strip_height * max_bpp;

        let strip_y = lp.padded_width * lp.strip_height * 4;
        let strip_cb = lp.padded_width * lp.strip_height * 4;
        let strip_cr = lp.padded_width * lp.strip_height * 4;

        let strip_cb_down = lp.padded_c_width * lp.c_strip_height * 4;
        let strip_cr_down = lp.padded_c_width * lp.c_strip_height * 4;

        let pending_y_f32 = 2 * lp.pending_y_capacity * 256;
        let pending_cb_f32 = 2 * lp.pending_c_capacity * 256;
        let pending_cr_f32 = 2 * lp.pending_c_capacity * 256;

        let y_blocks_i16 = y_block_count * 128;
        let c_blocks_i16 = c_block_count * 2 * 128;

        let aq_strengths = y_block_count * 4;

        let total_blocks = y_block_count + c_block_count * 2;
        let entropy_output = total_blocks * 10;

        let output_ceiling = lp.padded_width * padded_height;

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
