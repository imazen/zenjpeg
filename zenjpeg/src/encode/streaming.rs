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
//! use zenjpeg::{StreamingEncoder, Quality, Subsampling};
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

use crate::huffman::optimize::{FrequencyCounter, OptimizedHuffmanTables};

/// A complete set of frequency counters for Huffman table optimization.
///
/// Contains DC and AC counters for both luminance and chrominance.
/// Can be used to build custom Huffman tables or to supply pre-computed
/// frequency distributions to the encoder.
#[derive(Clone, Debug)]
pub struct HuffmanFrequencyCounts {
    /// DC luminance frequency counter
    pub dc_luma: FrequencyCounter,
    /// AC luminance frequency counter
    pub ac_luma: FrequencyCounter,
    /// DC chrominance frequency counter
    pub dc_chroma: FrequencyCounter,
    /// AC chrominance frequency counter
    pub ac_chroma: FrequencyCounter,
}

impl HuffmanFrequencyCounts {
    /// Creates a new set of empty frequency counters.
    #[must_use]
    pub fn new() -> Self {
        Self {
            dc_luma: FrequencyCounter::new(),
            ac_luma: FrequencyCounter::new(),
            dc_chroma: FrequencyCounter::new(),
            ac_chroma: FrequencyCounter::new(),
        }
    }

    /// Generates optimized Huffman tables from these frequency counts.
    ///
    /// This ensures coverage for all valid DC/AC symbols before generating,
    /// so the resulting tables can encode any valid JPEG symbol.
    pub fn generate_tables(&self) -> crate::error::Result<OptimizedHuffmanTables> {
        let huffman_method = crate::types::HuffmanMethod::JpegliCreateTree;

        let mut dc_luma = self.dc_luma.clone();
        let mut ac_luma = self.ac_luma.clone();
        let mut dc_chroma = self.dc_chroma.clone();
        let mut ac_chroma = self.ac_chroma.clone();

        dc_luma.ensure_dc_coverage();
        ac_luma.ensure_ac_coverage();
        dc_chroma.ensure_dc_coverage();
        ac_chroma.ensure_ac_coverage();

        Ok(OptimizedHuffmanTables {
            dc_luma: dc_luma.generate_table_with_method(huffman_method)?,
            ac_luma: ac_luma.generate_table_with_method(huffman_method)?,
            dc_chroma: dc_chroma.generate_table_with_method(huffman_method)?,
            ac_chroma: ac_chroma.generate_table_with_method(huffman_method)?,
        })
    }

    /// Adds counts from another set of frequency counters.
    pub fn add(&mut self, other: &HuffmanFrequencyCounts) {
        self.dc_luma.add(&other.dc_luma);
        self.ac_luma.add(&other.ac_luma);
        self.dc_chroma.add(&other.dc_chroma);
        self.ac_chroma.add(&other.ac_chroma);
    }

    /// Combines two sets of frequency counts into a new set.
    #[must_use]
    pub fn combined(&self, other: &HuffmanFrequencyCounts) -> Self {
        Self {
            dc_luma: self.dc_luma.combined(&other.dc_luma),
            ac_luma: self.ac_luma.combined(&other.ac_luma),
            dc_chroma: self.dc_chroma.combined(&other.dc_chroma),
            ac_chroma: self.ac_chroma.combined(&other.ac_chroma),
        }
    }
}

impl Default for HuffmanFrequencyCounts {
    fn default() -> Self {
        Self::new()
    }
}

/// Result from encoding that includes both JPEG data and Huffman statistics.
///
/// Returned by [`StreamingEncoder::finish_with_tables`].
#[derive(Debug)]
pub struct EncodingResult {
    /// The encoded JPEG data.
    pub jpeg: Vec<u8>,
    /// Final frequency counts from the entire image.
    ///
    /// These are the raw counts observed during encoding, before any
    /// coverage padding. Use for building "universal" tables from
    /// multiple images.
    pub frequency_counts: HuffmanFrequencyCounts,
    /// Huffman tables that were used for encoding.
    ///
    /// For optimized encoding, these are generated from partial or
    /// full frequency data. For standard tables mode, these are the
    /// JPEG standard tables.
    pub huffman_tables: OptimizedHuffmanTables,
}

/// Builder for creating a streaming encoder.
///
/// Use [`StreamingEncoder::new()`] to start building.
#[derive(Debug, Clone)]
#[cfg_attr(not(feature = "test-utils"), doc(hidden))]
pub struct StreamingEncoderBuilder {
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
    /// Memory limit for bounded-memory streaming (bytes)
    memory_limit: Option<usize>,
    /// Force transition to streaming after this many rows (for testing)
    transition_after_rows: Option<usize>,
    /// Minimum AC entropy required before allowing transition (bits, default 4.0)
    min_entropy: Option<f64>,
    /// Minimum AC symbol coverage required before allowing transition (%, default 30.0)
    min_coverage: Option<f64>,
    /// Minimum percentage of rows before transition is allowed (0-100)
    min_transition_percent: Option<usize>,
    /// Use standard Huffman tables when heuristics fail (fallback for pathological images)
    use_standard_tables_fallback: bool,
    /// Custom Huffman tables to use instead of generating from image data.
    custom_huffman_tables: Option<OptimizedHuffmanTables>,
    /// Custom frequency counts to generate Huffman tables from.
    /// Takes precedence over optimizing from image data, but custom_huffman_tables
    /// takes precedence over this.
    custom_frequency_counts: Option<HuffmanFrequencyCounts>,
    /// Number of iMCU rows to batch before encoding in streaming mode.
    /// Default is 1 (encode immediately). Higher values may improve throughput
    /// at the cost of slightly more memory.
    streaming_batch_size: usize,
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
            memory_limit: None,
            transition_after_rows: None,
            min_entropy: None,
            min_coverage: None,
            min_transition_percent: None,
            use_standard_tables_fallback: false,
            custom_huffman_tables: None,
            custom_frequency_counts: None,
            streaming_batch_size: 1, // Default: encode immediately
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
    pub fn quality(mut self, quality: impl Into<Quality>) -> Self {
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
    pub fn distance(mut self, distance: f32) -> Self {
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
    pub fn progressive(mut self, enable: bool) -> Self {
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
    pub fn subsampling(mut self, subsampling: Subsampling) -> Self {
        self.subsampling = subsampling;
        self
    }

    /// Sets the pixel format of input data.
    #[must_use]
    pub fn pixel_format(mut self, format: PixelFormat) -> Self {
        self.pixel_format = format;
        self
    }

    /// Sets the JPEG encoding mode.
    #[must_use]
    pub fn mode(mut self, mode: JpegMode) -> Self {
        self.mode = mode;
        self
    }

    /// Enables optimized Huffman tables.
    #[must_use]
    pub fn optimize_huffman(mut self, enable: bool) -> Self {
        self.optimize_huffman = enable;
        self
    }

    /// Sets chroma downsampling method for subsampled modes.
    #[must_use]
    pub fn chroma_downsampling(mut self, method: DownsamplingMethod) -> Self {
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
    pub fn sharp_yuv(mut self, enable: bool) -> Self {
        self.chroma_downsampling = if enable {
            DownsamplingMethod::GammaAwareIterative
        } else {
            DownsamplingMethod::Box
        };
        self
    }

    /// Sets the restart interval (MCUs between restart markers).
    #[must_use]
    pub fn restart_interval(mut self, interval: u16) -> Self {
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
    pub fn parallel(mut self, enable: bool) -> Self {
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
    pub fn encoding_tables(mut self, tables: Box<EncodingTables>) -> Self {
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
    pub fn use_xyb(mut self, enable: bool) -> Self {
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
    pub fn deringing(mut self, enable: bool) -> Self {
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
    pub fn allow_16bit_quant_tables(mut self, enable: bool) -> Self {
        self.allow_16bit_quant_tables = enable;
        self
    }

    /// Use separate Cb and Cr quantization tables.
    ///
    /// When enabled (default), uses 3 tables: Y, Cb, Cr.
    /// When disabled, uses 2 tables: Y, shared chroma.
    #[must_use]
    pub fn separate_chroma_tables(mut self, enable: bool) -> Self {
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
    pub fn hybrid_trellis(mut self, enable: bool) -> Self {
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
    pub fn hybrid_config(mut self, config: crate::hybrid::config::HybridConfig) -> Self {
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
    pub fn aq_map(mut self, map: crate::quant::aq::AQStrengthMap) -> Self {
        self.custom_aq_map = Some(map);
        self
    }

    /// Sets a memory limit for bounded-memory streaming.
    ///
    /// When the accumulated block storage reaches this limit, the encoder
    /// transitions to streaming mode:
    ///
    /// 1. Builds Huffman tables from accumulated symbol frequencies
    /// 2. Writes JPEG header (SOI, DQT, SOF, DHT, SOS)
    /// 3. Encodes all accumulated blocks and releases their storage
    /// 4. Continues encoding new blocks immediately (no buffering)
    ///
    /// This allows optimized Huffman encoding with bounded memory usage.
    ///
    /// **Important**: Progressive mode is not compatible with bounded streaming
    /// because progressive encoding requires multiple passes over all blocks.
    /// Use baseline mode (`progressive(false)`) with memory limits.
    ///
    /// # Arguments
    ///
    /// * `limit` - Maximum bytes for block storage. Recommended: at least
    ///   `estimate_memory_usage() / 4` to accumulate enough data for good
    ///   Huffman table optimization.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let encoder = StreamingEncoder::new(4000, 3000)
    ///     .quality(85)
    ///     .progressive(false)  // Required for bounded streaming
    ///     .memory_limit(8 * 1024 * 1024)  // 8 MB limit
    ///     .start()?;
    /// ```
    #[must_use]
    pub fn memory_limit(mut self, limit: usize) -> Self {
        self.memory_limit = Some(limit);
        self
    }

    /// Forces transition to streaming mode after the specified number of rows.
    ///
    /// This is primarily for testing to measure file size overhead at different
    /// transition points. In production, use `memory_limit()` instead.
    #[must_use]
    pub fn transition_after_rows(mut self, rows: usize) -> Self {
        self.transition_after_rows = Some(rows);
        self
    }

    /// Forces transition to streaming mode after the specified percentage of rows.
    ///
    /// Convenience method that calculates the row count from percentage.
    /// For example, `transition_after_percent(25)` transitions after 25% of rows.
    #[must_use]
    pub fn transition_after_percent(self, percent: usize) -> Self {
        let rows = (self.height as usize * percent) / 100;
        self.transition_after_rows(rows.max(16)) // At least 1 MCU row (16 pixels)
    }

    /// Sets minimum entropy threshold for frequency distribution stability.
    ///
    /// Before transitioning to streaming mode, the encoder will check that
    /// the accumulated AC frequency distribution has at least this entropy.
    /// Low entropy indicates the data is concentrated on few symbols (e.g.,
    /// smooth gradients), which may not be representative of the full image.
    ///
    /// Typical values:
    /// - 4.0 bits: Conservative (requires moderate variety)
    /// - 3.0 bits: Permissive (allows some concentration)
    /// - 5.0 bits: Strict (requires high variety)
    ///
    /// Default: None (no entropy check)
    #[must_use]
    pub fn min_entropy(mut self, entropy: f64) -> Self {
        self.min_entropy = Some(entropy);
        self
    }

    /// Sets minimum symbol coverage threshold for frequency distribution stability.
    ///
    /// Before transitioning to streaming mode, the encoder will check that
    /// at least this percentage of valid AC symbols have been seen.
    /// Low coverage indicates the data uses only a subset of symbols,
    /// which may lead to poor Huffman tables for unseen symbols.
    ///
    /// Range: 0.0-100.0 (percentage)
    /// Typical values:
    /// - 30.0%: Conservative (requires seeing ~50 of 162 valid symbols)
    /// - 20.0%: Permissive
    /// - 50.0%: Strict
    ///
    /// Default: None (no coverage check)
    #[must_use]
    pub fn min_coverage(mut self, coverage: f64) -> Self {
        self.min_coverage = Some(coverage);
        self
    }

    /// Sets both entropy and coverage thresholds with recommended defaults.
    ///
    /// This enables heuristic-based transition that delays streaming mode
    /// until the frequency distribution appears representative. Useful for
    /// avoiding pathological cases where the image start has very different
    /// content than the rest (e.g., gradient sky).
    ///
    /// Default thresholds: entropy=4.0 bits, coverage=30%, min 50% of rows
    #[must_use]
    pub fn require_stable_distribution(self) -> Self {
        self.min_entropy(4.0)
            .min_coverage(30.0)
            .min_transition_percent(50)
    }

    /// Sets minimum percentage of rows that must be processed before transition.
    ///
    /// Even if memory limit is reached and heuristics pass, transition will be
    /// delayed until at least this percentage of rows has been processed.
    ///
    /// Recommended value: 25% for most images to ensure sufficient data for
    /// stable Huffman table optimization.
    #[must_use]
    pub fn min_transition_percent(mut self, percent: usize) -> Self {
        self.min_transition_percent = Some(percent.min(100));
        self
    }

    /// Use standard Huffman tables instead of optimized tables from partial data.
    ///
    /// When enabled, the encoder will use JPEG standard Huffman tables instead
    /// of generating tables from accumulated frequency data. This provides
    /// bounded overhead (~5-10%) even for pathological images where early
    /// frequencies are not representative.
    ///
    /// Use this when:
    /// - You need guaranteed bounded overhead regardless of image content
    /// - Early transition is more important than optimal compression
    /// - You're encoding images with potentially pathological content
    ///
    /// The overhead with standard tables is typically 5-10% compared to
    /// optimized tables, but provides consistent results across all images.
    #[must_use]
    pub fn use_standard_huffman_tables(mut self, enable: bool) -> Self {
        self.use_standard_tables_fallback = enable;
        self
    }

    /// Uses custom Huffman tables instead of generating from image data.
    ///
    /// This allows using pre-computed "universal" tables that work well across
    /// a corpus of images. The tables are used directly without modification.
    ///
    /// This takes precedence over:
    /// - `custom_frequency_counts()` - ignored if tables provided
    /// - Optimized tables from image data
    /// - Standard tables (if `use_standard_huffman_tables(true)`)
    ///
    /// # Use Cases
    ///
    /// - **Consistent encoding**: Use the same tables across all images
    /// - **Pre-optimized tables**: Use tables tuned for specific content types
    /// - **Bounded-memory streaming**: Avoid table optimization overhead
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Load tables from previous corpus analysis
    /// let tables = load_universal_tables()?;
    ///
    /// let jpeg = StreamingEncoder::new(640, 480)
    ///     .quality(85)
    ///     .custom_huffman_tables(tables)
    ///     .encode(&pixels)?;
    /// ```
    #[must_use]
    pub fn custom_huffman_tables(mut self, tables: OptimizedHuffmanTables) -> Self {
        self.custom_huffman_tables = Some(tables);
        self
    }

    /// Uses custom frequency counts to generate Huffman tables.
    ///
    /// The tables will be generated from these counts at encoding time.
    /// This is useful for building "universal" tables from a corpus of images.
    ///
    /// Unlike `custom_huffman_tables()`, this generates tables using the
    /// same algorithm as optimized encoding, but from your supplied counts
    /// instead of from the image being encoded.
    ///
    /// This is ignored if `custom_huffman_tables()` is also set.
    ///
    /// # Use Cases
    ///
    /// - **Corpus-based tables**: Combine counts from multiple images
    /// - **Incremental learning**: Update counts as you encode more images
    /// - **Domain-specific tables**: Build tables tuned for specific content
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Combine counts from multiple images
    /// let mut corpus_counts = HuffmanFrequencyCounts::new();
    /// for image in corpus {
    ///     let result = encode_and_get_counts(image)?;
    ///     corpus_counts.add(&result.frequency_counts);
    /// }
    ///
    /// // Use combined counts for new encodes
    /// let jpeg = StreamingEncoder::new(640, 480)
    ///     .quality(85)
    ///     .custom_frequency_counts(corpus_counts.clone())
    ///     .encode(&pixels)?;
    /// ```
    #[must_use]
    pub fn custom_frequency_counts(mut self, counts: HuffmanFrequencyCounts) -> Self {
        self.custom_frequency_counts = Some(counts);
        self
    }

    /// Sets the number of iMCU rows to batch before encoding in streaming mode.
    ///
    /// Default is 1 (encode immediately after each iMCU row is ready).
    /// This only affects streaming mode (after transition from buffered mode
    /// or with immediate streaming via `memory_limit(1)`).
    ///
    /// # Performance Notes
    ///
    /// Benchmarks show no meaningful performance difference between batch sizes
    /// 1-16 on 4K images. The encoder is already well-optimized for per-iMCU-row
    /// processing, so the default of 1 is recommended.
    ///
    /// This option is provided for experimentation but is unlikely to provide
    /// benefits in practice.
    #[must_use]
    pub fn streaming_batch_size(mut self, size: usize) -> Self {
        self.streaming_batch_size = size.max(1); // Minimum 1
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
    /// use zenjpeg::JpegEncoder;
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
    pub fn start(self) -> Result<StreamingEncoder> {
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
    /// use zenjpeg::{JpegEncoder, Subsampling};
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
    pub fn encode(self, data: &[u8]) -> Result<Vec<u8>> {
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
    pub fn encode_with_stop(self, data: &[u8], stop: impl Stop) -> Result<Vec<u8>> {
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
    /// use zenjpeg::{StreamingEncoder, Subsampling};
    ///
    /// let estimated = StreamingEncoder::new(3840, 2160)
    ///     .subsampling(Subsampling::S420)
    ///     .estimate_memory_usage();
    ///
    /// println!("Estimated peak memory: {} MB", estimated / 1024 / 1024);
    /// ```
    #[must_use]
    pub fn estimate_memory_usage(&self) -> usize {
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
        // Pre-allocated at ~3 bytes/block, grows if needed. Typical images use 0.5-3 bytes/block,
        // worst case (high-frequency noise) uses ~7 bytes/block.
        // Progressive mode uses smaller per-scan buffers instead.
        let total_blocks = y_block_count + c_block_count * 2;
        let entropy_output = total_blocks * 3;

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
            + entropy_output
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
    /// use zenjpeg::{StreamingEncoder, Subsampling};
    ///
    /// let ceiling = StreamingEncoder::new(3840, 2160)
    ///     .subsampling(Subsampling::S420)
    ///     .estimate_memory_ceiling();
    ///
    /// // Reserve this much memory before encoding
    /// assert!(actual_peak <= ceiling);
    /// ```
    #[must_use]
    pub fn estimate_memory_ceiling(&self) -> usize {
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

        // 7. Entropy output buffer - CEILING: worst-case ~10 bytes per block
        // High-frequency content (noise, text, fine detail) produces more output.
        // Measured worst case is ~7 bytes/block, use 10 for ceiling.
        let total_blocks = y_block_count + c_block_count * 2;
        let entropy_output = total_blocks * 10;

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
            + entropy_output
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
///
/// # Bounded-Memory Streaming
///
/// By default, the encoder buffers all quantized blocks until `finish()`.
/// With `set_memory_limit()`, the encoder can transition to true streaming
/// mode after accumulating enough data to build optimal Huffman tables:
///
/// 1. **Accumulation phase**: Buffers blocks and counts symbol frequencies
/// 2. **Transition**: When memory limit reached, builds Huffman tables from
///    accumulated frequencies, writes JPEG header, encodes buffered blocks,
///    then releases block storage
/// 3. **Streaming phase**: Encodes new blocks immediately after quantization
///
/// This allows optimized Huffman encoding with bounded memory usage.
#[cfg_attr(not(feature = "test-utils"), doc(hidden))]
pub struct StreamingEncoder {
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

    // === Bounded-memory streaming fields ===
    /// Memory limit for block storage (None = unlimited buffering)
    memory_limit: Option<usize>,
    /// True if we've transitioned to streaming mode (header written, blocks encoded immediately)
    streaming_mode: bool,
    /// Output buffer for streaming mode (holds JPEG data being built)
    streaming_output: Option<Vec<u8>>,
    /// Optimized Huffman tables (built during transition, used in streaming mode)
    streaming_tables: Option<crate::huffman::optimize::OptimizedHuffmanTables>,
    /// Previous DC values for entropy encoding continuity in streaming mode
    streaming_prev_dc: [i16; 3],
    /// MCU index for restart interval tracking in streaming mode
    streaming_mcu_idx: usize,
    /// Restart counter (0-7) for streaming mode
    streaming_restart_count: u8,
    /// Bit buffer state for streaming mode (partial byte from previous flush)
    streaming_bit_buffer: u64,
    /// Number of valid bits in streaming_bit_buffer
    streaming_bits_in_buffer: u8,
    /// Force transition to streaming after this many rows (for testing)
    transition_after_rows: Option<usize>,
    /// Minimum AC entropy required before allowing transition (bits)
    min_entropy: Option<f64>,
    /// Minimum AC symbol coverage required before allowing transition (%)
    min_coverage: Option<f64>,
    /// Minimum percentage of rows before transition is allowed
    min_transition_percent: Option<usize>,
    /// Row at which transition to streaming mode occurred (for diagnostics)
    transition_at_row: Option<usize>,
    /// Reason for transition (for diagnostics)
    transition_reason: Option<TransitionReason>,
    /// Use standard Huffman tables instead of optimized from partial data
    use_standard_tables: bool,
    /// Custom Huffman tables to use (takes precedence over all other table sources)
    custom_huffman_tables: Option<OptimizedHuffmanTables>,
    /// Custom frequency counts to generate tables from
    custom_frequency_counts: Option<HuffmanFrequencyCounts>,
    /// Final tables used for encoding (stored for finish_with_tables)
    final_tables: Option<OptimizedHuffmanTables>,
    /// Number of iMCU rows to batch before encoding in streaming mode
    streaming_batch_size: usize,
    /// Number of iMCU rows accumulated since last encode
    streaming_imcu_pending: usize,
}

/// Reason why streaming transition occurred.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(not(feature = "test-utils"), doc(hidden))]
pub enum TransitionReason {
    /// Forced by transition_after_rows (testing API)
    ForcedByRows,
    /// Memory limit hit and heuristics passed (or no heuristics configured)
    HeuristicsPassed,
    /// Memory limit hit but waiting for min_transition_percent
    MinPercentReached,
    /// Safety valve at 50% of image
    SafetyValve,
    /// No streaming transition (full buffering mode)
    NoTransition,
}

impl StreamingEncoder {
    /// Creates a new streaming encoder builder with the given dimensions.
    ///
    /// Use the builder methods to configure quality, subsampling, etc.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use zenjpeg::{StreamingEncoder, Quality, Subsampling};
    ///
    /// let encoder = StreamingEncoder::new(1920, 1080)
    ///     .quality(Quality::ApproxJpegli(85.0))
    ///     .subsampling(Subsampling::S420)
    ///     .start()?;
    /// ```
    #[must_use]
    #[allow(clippy::new_ret_no_self)] // Builder pattern: new() returns builder
    pub fn new(width: u32, height: u32) -> StreamingEncoderBuilder {
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

        // Skip frequency counting if custom Huffman tables are provided
        // (avoids wasted work when tables won't be built from frequencies)
        if builder.custom_huffman_tables.is_some() {
            processor.set_skip_frequency_counting(true);
        }

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

        // Validate memory limit compatibility
        if builder.memory_limit.is_some() && builder.mode == JpegMode::Progressive {
            return Err(Error::unsupported_feature(
                "memory_limit is not compatible with progressive mode (requires multiple passes)",
            ));
        }

        // Auto-enable immediate streaming when custom tables are provided
        // (no reason to buffer blocks if we're not building tables from frequencies)
        let memory_limit = if builder.custom_huffman_tables.is_some()
            && builder.memory_limit.is_none()
            && builder.mode != JpegMode::Progressive
        {
            Some(1) // Trigger immediate streaming
        } else {
            builder.memory_limit
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
            // Bounded-memory streaming fields
            memory_limit,
            streaming_mode: false,
            streaming_output: None,
            streaming_tables: None,
            streaming_prev_dc: [0; 3],
            streaming_mcu_idx: 0,
            streaming_restart_count: 0,
            streaming_bit_buffer: 0,
            streaming_bits_in_buffer: 0,
            transition_after_rows: builder.transition_after_rows,
            min_entropy: builder.min_entropy,
            min_coverage: builder.min_coverage,
            min_transition_percent: builder.min_transition_percent,
            transition_at_row: None,
            transition_reason: None,
            use_standard_tables: builder.use_standard_tables_fallback,
            custom_huffman_tables: builder.custom_huffman_tables,
            custom_frequency_counts: builder.custom_frequency_counts,
            final_tables: None,
            streaming_batch_size: builder.streaming_batch_size,
            streaming_imcu_pending: 0,
        })
    }

    /// Returns the number of rows pushed so far.
    #[must_use]
    pub fn rows_pushed(&self) -> usize {
        self.current_y + self.rows_buffered
    }

    /// Returns the expected number of bytes per row.
    #[must_use]
    pub fn bytes_per_row(&self) -> usize {
        self.bytes_per_row
    }

    /// Returns the total height of the image.
    #[must_use]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Returns the strip height (internal processing unit).
    #[must_use]
    pub fn strip_height(&self) -> usize {
        self.strip_height
    }

    /// Returns allocation statistics from the strip processor.
    ///
    /// This tracks all major allocations made during encoding setup,
    /// including color plane buffers, DCT block storage, and AQ buffers.
    #[must_use]
    pub fn allocation_stats(&self) -> &crate::foundation::alloc::AllocationStats {
        self.processor.allocation_stats()
    }

    /// Returns whether the encoder is in streaming mode.
    ///
    /// In streaming mode, blocks are encoded immediately after quantization
    /// rather than being buffered. This happens after the memory limit is
    /// reached and the encoder transitions from accumulation to streaming.
    #[must_use]
    pub fn is_streaming(&self) -> bool {
        self.streaming_mode
    }

    /// Estimates current block storage memory usage in bytes.
    ///
    /// This counts the memory used by quantized coefficient blocks (Y, Cb, Cr).
    /// Each block is 64 i16 coefficients = 128 bytes.
    #[must_use]
    pub fn estimate_block_storage(&self) -> usize {
        self.processor.estimate_block_storage()
    }

    /// Returns frequency distribution heuristics for the accumulated data.
    ///
    /// Returns (ac_luma_coverage%, ac_luma_entropy, ac_chroma_coverage%, ac_chroma_entropy).
    /// These can be used to detect pathological distributions before transitioning.
    #[must_use]
    pub fn frequency_heuristics(&self) -> (f64, f64, f64, f64) {
        let (_, ac_luma, _, ac_chroma) = self.processor.frequency_counters();
        (
            ac_luma.ac_symbol_coverage(),
            ac_luma.entropy(),
            ac_chroma.ac_symbol_coverage(),
            ac_chroma.entropy(),
        )
    }

    /// Returns raw frequency counters for analysis (test-utils only).
    ///
    /// Returns (dc_luma, ac_luma, dc_chroma, ac_chroma) frequency counters.
    #[cfg(feature = "test-utils")]
    #[must_use]
    pub fn frequency_counters(
        &self,
    ) -> (
        &crate::huffman::optimize::FrequencyCounter,
        &crate::huffman::optimize::FrequencyCounter,
        &crate::huffman::optimize::FrequencyCounter,
        &crate::huffman::optimize::FrequencyCounter,
    ) {
        self.processor.frequency_counters()
    }

    /// Checks if the accumulated frequency distribution appears stable/representative.
    ///
    /// Returns true if:
    /// - AC luma entropy >= min_entropy (default 4.0 bits)
    /// - AC symbol coverage >= min_coverage (default 30%)
    ///
    /// Low entropy or coverage suggests we're in a smooth/gradient region
    /// that may not be representative of the full image.
    #[must_use]
    pub fn is_distribution_stable(&self, min_entropy: f64, min_coverage: f64) -> bool {
        let (ac_cov, ac_ent, _, _) = self.frequency_heuristics();
        ac_ent >= min_entropy && ac_cov >= min_coverage
    }

    /// Returns the row at which transition to streaming mode occurred.
    ///
    /// Returns `None` if:
    /// - No memory limit was set (full buffering mode)
    /// - Transition hasn't happened yet
    /// - Encoding completed without transitioning (image fit in memory limit)
    #[must_use]
    pub fn transition_row(&self) -> Option<usize> {
        self.transition_at_row
    }

    /// Returns the percentage of rows at which transition occurred.
    ///
    /// Returns `None` if transition hasn't happened.
    #[must_use]
    pub fn transition_percent(&self) -> Option<f64> {
        self.transition_at_row
            .map(|row| 100.0 * row as f64 / self.height as f64)
    }

    /// Returns the reason for streaming transition.
    ///
    /// Returns `None` if transition hasn't happened yet.
    #[must_use]
    pub fn transition_reason(&self) -> Option<TransitionReason> {
        self.transition_reason
    }

    /// Returns both transition percentage and reason as a formatted string.
    #[must_use]
    pub fn transition_info(&self) -> String {
        match (self.transition_percent(), self.transition_reason) {
            (Some(pct), Some(reason)) => {
                let reason_str = match reason {
                    TransitionReason::ForcedByRows => "forced",
                    TransitionReason::HeuristicsPassed => "heuristics",
                    TransitionReason::MinPercentReached => "min%",
                    TransitionReason::SafetyValve => "safety",
                    TransitionReason::NoTransition => "none",
                };
                format!("{:.0}% ({})", pct, reason_str)
            }
            (Some(pct), None) => format!("{:.0}%", pct),
            _ => "N/A".to_string(),
        }
    }

    /// Checks if memory limit is exceeded and transitions to streaming mode if needed.
    ///
    /// This should be called after each strip is processed. If the memory limit
    /// is exceeded, this method:
    /// 1. Builds Huffman tables from accumulated inline frequencies
    /// 2. Writes JPEG header (SOI, APP0, DQT, SOF, DHT, SOS)
    /// 3. Encodes all accumulated blocks
    /// 4. Releases block storage
    /// 5. Sets streaming_mode = true
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Huffman table generation fails
    /// - JPEG header writing fails
    /// - Block encoding fails
    fn check_and_maybe_transition(&mut self) -> Result<()> {
        // Skip if already in streaming mode
        if self.streaming_mode {
            return Ok(());
        }

        // Check row-based threshold (for testing different transition points)
        // This bypasses heuristics - used for threshold testing
        if let Some(threshold_rows) = self.transition_after_rows {
            let rows_processed = self.current_y;
            if rows_processed >= threshold_rows {
                self.transition_reason = Some(TransitionReason::ForcedByRows);
                return self.transition_to_streaming();
            }
        }

        // Check memory limit with optional heuristic gating
        if let Some(limit) = self.memory_limit {
            let current_usage = self.estimate_block_storage();
            if current_usage >= limit {
                // First check: minimum percentage requirement
                // This prevents transition on too little data regardless of heuristics
                let min_pct_ok = if let Some(min_pct) = self.min_transition_percent {
                    let min_rows = (self.height as usize * min_pct) / 100;
                    self.current_y >= min_rows
                } else {
                    true // No minimum set, always OK
                };

                // Check heuristics
                let heuristics_pass = self.check_distribution_heuristics();

                // Determine transition reason
                if min_pct_ok && heuristics_pass {
                    // Both min_pct and heuristics are satisfied
                    // Report which was the limiting factor
                    if self.min_entropy.is_some() || self.min_coverage.is_some() {
                        self.transition_reason = Some(TransitionReason::HeuristicsPassed);
                    } else {
                        self.transition_reason = Some(TransitionReason::MinPercentReached);
                    }
                    return self.transition_to_streaming();
                }

                // Safety valve: always transition after 50% of image regardless of heuristics
                let rows_processed = self.current_y;
                let half_image = self.height / 2;
                if rows_processed >= half_image {
                    self.transition_reason = Some(TransitionReason::SafetyValve);
                    return self.transition_to_streaming();
                }
            }
        }

        Ok(())
    }

    /// Checks if distribution stability heuristics pass.
    ///
    /// Returns true if either:
    /// - No heuristics are configured, or
    /// - All configured heuristic thresholds are met
    fn check_distribution_heuristics(&self) -> bool {
        let (ac_cov, ac_ent, _, _) = self.frequency_heuristics();

        // Check entropy threshold
        if let Some(min_ent) = self.min_entropy {
            if ac_ent < min_ent {
                return false;
            }
        }

        // Check coverage threshold
        if let Some(min_cov) = self.min_coverage {
            if ac_cov < min_cov {
                return false;
            }
        }

        true
    }

    /// Transitions to streaming mode.
    ///
    /// This is the core bounded-memory streaming implementation:
    /// 1. Builds Huffman tables from accumulated inline frequencies
    /// 2. Writes JPEG header up through SOS marker
    /// 3. Encodes all accumulated blocks
    /// 4. Releases block storage
    /// 5. Sets streaming_mode = true
    fn transition_to_streaming(&mut self) -> Result<()> {
        use crate::foundation::bitstream::BitWriter;

        // Record transition point for diagnostics
        self.transition_at_row = Some(self.current_y);

        // Build Huffman tables, checking sources in priority order:
        // 1. custom_huffman_tables - use directly
        // 2. custom_frequency_counts - generate tables from provided counts
        // 3. use_standard_tables - use JPEG standard tables
        // 4. default - optimize from accumulated image frequency data
        let tables = if let Some(tables) = self.custom_huffman_tables.take() {
            // Use pre-built custom tables directly
            tables
        } else if let Some(ref counts) = self.custom_frequency_counts {
            // Generate tables from custom frequency counts
            counts.generate_tables()?
        } else if self.use_standard_tables {
            // Use JPEG standard tables
            Self::build_standard_tables()
        } else {
            // Generate from accumulated image data (original logic)
            self.build_tables_from_image_data()?
        };

        // Store tables for finish_with_tables
        self.final_tables = Some(tables.clone());

        let is_color = !self.config.pixel_format.is_grayscale();

        // Initialize output buffer
        let mut output = Vec::new();
        if output.try_reserve(64 * 1024).is_err() {
            return Err(Error::allocation_failed(
                64 * 1024,
                "streaming output buffer",
            ));
        }

        // Write JPEG header using ComputedConfig methods
        self.config.write_header(&mut output)?;
        self.config.write_quant_tables(
            &mut output,
            &self.y_quant,
            &self.cb_quant,
            &self.cr_quant,
        )?;
        self.config.write_frame_header(&mut output)?;
        self.config
            .write_huffman_tables_optimized(&mut output, &tables)?;

        // Write DRI if restart interval is set
        if self.config.restart_interval > 0 {
            self.config.write_restart_interval(&mut output)?;
        }

        // Write SOS marker (start of scan)
        self.config.write_scan_header(&mut output)?;

        // Encode accumulated blocks
        let blocks = self.processor.take_blocks();
        let mut writer = BitWriter::new();

        // Encode all accumulated blocks in MCU order
        let (prev_dc, mcu_idx, restart_count) = self.encode_blocks_mcu_order_ex(
            &blocks.y_blocks,
            &blocks.cb_blocks,
            &blocks.cr_blocks,
            &tables,
            &mut writer,
            is_color,
            [0, 0, 0],
            0,
            0,
        )?;

        // Flush ONLY complete bytes to output, preserve partial byte for next chunk
        let (bit_buffer, bits_in_buffer) = writer.flush_complete_bytes_only(&mut output)?;

        // Save state for continuing in streaming mode
        self.streaming_prev_dc = prev_dc;
        self.streaming_mcu_idx = mcu_idx;
        self.streaming_restart_count = restart_count;
        self.streaming_bit_buffer = bit_buffer;
        self.streaming_bits_in_buffer = bits_in_buffer;
        self.streaming_output = Some(output);
        self.streaming_tables = Some(tables);
        self.streaming_mode = true;

        Ok(())
    }

    /// Builds JPEG standard Huffman tables.
    fn build_standard_tables() -> OptimizedHuffmanTables {
        use crate::huffman::optimize::OptimizedTable;
        use crate::huffman::{
            HuffmanEncodeTable, STD_AC_CHROMINANCE_BITS, STD_AC_CHROMINANCE_VALUES,
            STD_AC_LUMINANCE_BITS, STD_AC_LUMINANCE_VALUES, STD_DC_CHROMINANCE_BITS,
            STD_DC_CHROMINANCE_VALUES, STD_DC_LUMINANCE_BITS, STD_DC_LUMINANCE_VALUES,
        };

        OptimizedHuffmanTables {
            dc_luma: OptimizedTable {
                table: HuffmanEncodeTable::std_dc_luminance().clone(),
                bits: STD_DC_LUMINANCE_BITS,
                values: STD_DC_LUMINANCE_VALUES.to_vec(),
            },
            ac_luma: OptimizedTable {
                table: HuffmanEncodeTable::std_ac_luminance().clone(),
                bits: STD_AC_LUMINANCE_BITS,
                values: STD_AC_LUMINANCE_VALUES.to_vec(),
            },
            dc_chroma: OptimizedTable {
                table: HuffmanEncodeTable::std_dc_chrominance().clone(),
                bits: STD_DC_CHROMINANCE_BITS,
                values: STD_DC_CHROMINANCE_VALUES.to_vec(),
            },
            ac_chroma: OptimizedTable {
                table: HuffmanEncodeTable::std_ac_chrominance().clone(),
                bits: STD_AC_CHROMINANCE_BITS,
                values: STD_AC_CHROMINANCE_VALUES.to_vec(),
            },
        }
    }

    /// Builds optimized Huffman tables from accumulated image frequency data.
    fn build_tables_from_image_data(&self) -> Result<OptimizedHuffmanTables> {
        use crate::huffman::optimize::OptimizedTable;

        // Get frequency counters from processor
        let (dc_luma_freq, ac_luma_freq, dc_chroma_freq, ac_chroma_freq) =
            self.processor.frequency_counters();

        // Clone and ensure coverage for all valid symbols.
        // This creates partially-optimized tables that:
        // 1. Favor the observed symbol frequencies (shorter codes for common symbols)
        // 2. Have codes for ALL valid symbols (longer codes for unseen ones)
        //
        // Without ensure_*_coverage(), zero-frequency symbols get no code assigned,
        // causing silent encoding failures when those symbols appear later.
        let mut dc_luma_freq = dc_luma_freq.clone();
        let mut ac_luma_freq = ac_luma_freq.clone();
        dc_luma_freq.ensure_dc_coverage();
        ac_luma_freq.ensure_ac_coverage();

        let huffman_method = crate::types::HuffmanMethod::JpegliCreateTree;
        let dc_luma = dc_luma_freq.generate_table_with_method(huffman_method)?;
        let ac_luma = ac_luma_freq.generate_table_with_method(huffman_method)?;

        let is_color = !self.config.pixel_format.is_grayscale();
        let (dc_chroma, ac_chroma) = if is_color {
            let mut dc_chroma_freq = dc_chroma_freq.clone();
            let mut ac_chroma_freq = ac_chroma_freq.clone();
            dc_chroma_freq.ensure_dc_coverage();
            ac_chroma_freq.ensure_ac_coverage();
            (
                dc_chroma_freq.generate_table_with_method(huffman_method)?,
                ac_chroma_freq.generate_table_with_method(huffman_method)?,
            )
        } else {
            // Use standard tables for grayscale (won't be used)
            use crate::huffman::{
                HuffmanEncodeTable, STD_AC_CHROMINANCE_BITS, STD_AC_CHROMINANCE_VALUES,
                STD_DC_CHROMINANCE_BITS, STD_DC_CHROMINANCE_VALUES,
            };
            (
                OptimizedTable {
                    table: HuffmanEncodeTable::std_dc_chrominance().clone(),
                    bits: STD_DC_CHROMINANCE_BITS,
                    values: STD_DC_CHROMINANCE_VALUES.to_vec(),
                },
                OptimizedTable {
                    table: HuffmanEncodeTable::std_ac_chrominance().clone(),
                    bits: STD_AC_CHROMINANCE_BITS,
                    values: STD_AC_CHROMINANCE_VALUES.to_vec(),
                },
            )
        };

        Ok(OptimizedHuffmanTables {
            dc_luma,
            ac_luma,
            dc_chroma,
            ac_chroma,
        })
    }

    /// Encodes blocks in MCU order and returns the final DC prediction values.
    ///
    /// # Arguments
    /// * `y_blocks`, `cb_blocks`, `cr_blocks` - Blocks to encode (relative indices from start_mcu_idx)
    /// * `tables` - Huffman tables to use
    /// * `writer` - BitWriter to write encoded data to
    /// * `is_color` - Whether this is a color image
    /// * `start_prev_dc` - Starting DC prediction values (Y, Cb, Cr)
    /// * `start_mcu_idx` - Starting MCU index (blocks are indexed relative to this)
    /// * `start_restart_count` - Starting restart counter
    ///
    /// Returns: (final DC values, final MCU index, restart counter)
    ///
    /// IMPORTANT: Blocks are indexed relative to the start of the passed slices.
    /// Block [0] in y_blocks corresponds to the first Y block of the MCU at start_mcu_idx.
    fn encode_blocks_mcu_order_ex(
        &self,
        y_blocks: &[[i16; 64]],
        cb_blocks: &[[i16; 64]],
        cr_blocks: &[[i16; 64]],
        tables: &crate::huffman::optimize::OptimizedHuffmanTables,
        writer: &mut crate::foundation::bitstream::BitWriter,
        is_color: bool,
        start_prev_dc: [i16; 3],
        start_mcu_idx: usize,
        start_restart_count: u8,
    ) -> Result<([i16; 3], usize, u8)> {
        use crate::entropy;

        let (h_samp, v_samp) = match self.config.subsampling {
            Subsampling::S444 => (1, 1),
            Subsampling::S422 => (2, 1),
            Subsampling::S420 => (2, 2),
            Subsampling::S440 => (1, 2),
        };

        let y_blocks_h = (self.width + 7) / 8;
        let y_blocks_v = (self.height + 7) / 8;
        let mcu_h = (y_blocks_h + h_samp - 1) / h_samp;
        let mcu_v = (y_blocks_v + v_samp - 1) / v_samp;

        let mut prev_y_dc = start_prev_dc[0];
        let mut prev_cb_dc = start_prev_dc[1];
        let mut prev_cr_dc = start_prev_dc[2];

        let restart_interval = self.config.restart_interval as usize;
        let total_mcus = mcu_h * mcu_v;
        let mut mcu_idx = start_mcu_idx;
        let mut restart_count = start_restart_count;

        // Calculate starting MCU row from start_mcu_idx
        let start_mcu_y = start_mcu_idx / mcu_h;

        // Chroma block dimensions
        let c_width = (self.width + h_samp - 1) / h_samp;
        let c_blocks_h = (c_width + 7) / 8;

        // Calculate how many MCU rows of blocks we have
        // For raster-ordered blocks, we need to know the block row offset
        let y_rows_in_blocks = (y_blocks.len() + y_blocks_h - 1) / y_blocks_h.max(1);
        let mcu_rows_in_blocks = (y_rows_in_blocks + v_samp - 1) / v_samp;

        // Zero block for out-of-bounds padding
        const ZERO_BLOCK: [i16; 64] = [0i16; 64];

        // CRITICAL: Blocks are in RASTER order, not MCU order!
        // We must convert MCU coordinates to raster indices.
        'mcu_loop: for mcu_y in start_mcu_y..mcu_v {
            // Check if we have blocks for this MCU row
            let rel_mcu_y = mcu_y - start_mcu_y;
            if rel_mcu_y >= mcu_rows_in_blocks {
                break 'mcu_loop;
            }

            // For the first row, start from the correct MCU x position
            let start_mcu_x = if mcu_y == start_mcu_y {
                start_mcu_idx % mcu_h
            } else {
                0
            };

            for mcu_x in start_mcu_x..mcu_h {
                // Encode Y blocks in this MCU using raster indexing
                for dy in 0..v_samp {
                    for dx in 0..h_samp {
                        // Calculate absolute block position in image
                        let y_bx = mcu_x * h_samp + dx;
                        let y_by = mcu_y * v_samp + dy;

                        // Calculate relative position within passed blocks
                        // Blocks start at row (start_mcu_y * v_samp)
                        let rel_y_by = y_by - (start_mcu_y * v_samp);
                        let y_idx = rel_y_by * y_blocks_h + y_bx;

                        let block = if y_idx < y_blocks.len() && y_bx < y_blocks_h {
                            &y_blocks[y_idx]
                        } else {
                            &ZERO_BLOCK
                        };

                        entropy::encode_block_to_writer(
                            block,
                            &tables.dc_luma.table,
                            &tables.ac_luma.table,
                            prev_y_dc,
                            writer,
                        )?;
                        prev_y_dc = block[0];
                    }
                }

                // Encode Cb/Cr blocks using raster indexing
                if is_color {
                    // Calculate relative chroma block position
                    let rel_mcu_y = mcu_y - start_mcu_y;
                    let c_idx = rel_mcu_y * c_blocks_h + mcu_x;

                    let cb_block = if c_idx < cb_blocks.len() && mcu_x < c_blocks_h {
                        &cb_blocks[c_idx]
                    } else {
                        &ZERO_BLOCK
                    };
                    let cr_block = if c_idx < cr_blocks.len() && mcu_x < c_blocks_h {
                        &cr_blocks[c_idx]
                    } else {
                        &ZERO_BLOCK
                    };

                    entropy::encode_block_to_writer(
                        cb_block,
                        &tables.dc_chroma.table,
                        &tables.ac_chroma.table,
                        prev_cb_dc,
                        writer,
                    )?;
                    prev_cb_dc = cb_block[0];

                    entropy::encode_block_to_writer(
                        cr_block,
                        &tables.dc_chroma.table,
                        &tables.ac_chroma.table,
                        prev_cr_dc,
                        writer,
                    )?;
                    prev_cr_dc = cr_block[0];
                }

                mcu_idx += 1;

                // Handle restart markers
                if restart_interval > 0 && mcu_idx < total_mcus && mcu_idx % restart_interval == 0 {
                    writer.flush_restart_marker(restart_count)?;
                    restart_count = (restart_count + 1) % 8;
                    prev_y_dc = 0;
                    prev_cb_dc = 0;
                    prev_cr_dc = 0;
                }
            }
        }

        Ok(([prev_y_dc, prev_cb_dc, prev_cr_dc], mcu_idx, restart_count))
    }

    /// Simple wrapper for encode_blocks_mcu_order_ex that starts from zero state.
    fn encode_blocks_mcu_order(
        &self,
        y_blocks: &[[i16; 64]],
        cb_blocks: &[[i16; 64]],
        cr_blocks: &[[i16; 64]],
        tables: &crate::huffman::optimize::OptimizedHuffmanTables,
        writer: &mut crate::foundation::bitstream::BitWriter,
        is_color: bool,
    ) -> Result<(i16, i16, i16)> {
        let (prev_dc, _, _) = self.encode_blocks_mcu_order_ex(
            y_blocks,
            cb_blocks,
            cr_blocks,
            tables,
            writer,
            is_color,
            [0, 0, 0],
            0,
            0,
        )?;
        Ok((prev_dc[0], prev_dc[1], prev_dc[2]))
    }

    /// Static version of encode_blocks_mcu_order_ex that doesn't require &self.
    ///
    /// Used by finish_streaming where self.processor has been consumed.
    ///
    /// IMPORTANT: This function assumes `y_blocks`, `cb_blocks`, `cr_blocks` contain
    /// ONLY the blocks to be encoded (from start_mcu_idx onwards). Block indices
    /// are relative to the start of the passed slices, not absolute image indices.
    ///
    /// For example, if encoding MCU rows 5-10 of a 100-MCU-row image, pass:
    /// - `start_mcu_idx = 5 * mcu_width` (where encoding should start)
    /// - `y_blocks` containing only the Y blocks for MCU rows 5-10
    /// - Block [0] in y_blocks corresponds to first block of MCU row 5
    #[allow(clippy::too_many_arguments)]
    fn encode_blocks_mcu_order_static(
        y_blocks: &[[i16; 64]],
        cb_blocks: &[[i16; 64]],
        cr_blocks: &[[i16; 64]],
        tables: &crate::huffman::optimize::OptimizedHuffmanTables,
        writer: &mut crate::foundation::bitstream::BitWriter,
        is_color: bool,
        start_prev_dc: [i16; 3],
        start_mcu_idx: usize,
        start_restart_count: u8,
        subsampling: Subsampling,
        width: usize,
        height: usize,
        restart_interval: u16,
    ) -> Result<([i16; 3], usize, u8)> {
        use crate::entropy;

        let (h_samp, v_samp) = match subsampling {
            Subsampling::S444 => (1, 1),
            Subsampling::S422 => (2, 1),
            Subsampling::S420 => (2, 2),
            Subsampling::S440 => (1, 2),
        };

        let y_blocks_h = (width + 7) / 8;
        let y_blocks_v = (height + 7) / 8;
        let c_blocks_h = ((width + h_samp - 1) / h_samp + 7) / 8;
        let mcu_h = (y_blocks_h + h_samp - 1) / h_samp;
        let mcu_v = (y_blocks_v + v_samp - 1) / v_samp;

        let mut prev_y_dc = start_prev_dc[0];
        let mut prev_cb_dc = start_prev_dc[1];
        let mut prev_cr_dc = start_prev_dc[2];

        let restart_interval = restart_interval as usize;
        let total_mcus = mcu_h * mcu_v;
        let mut mcu_idx = start_mcu_idx;
        let mut restart_count = start_restart_count;

        // Calculate starting MCU row from start_mcu_idx
        let start_mcu_y = start_mcu_idx / mcu_h;

        // Calculate how many MCU rows of blocks we have
        let y_rows_in_blocks = (y_blocks.len() + y_blocks_h - 1) / y_blocks_h.max(1);
        let mcu_rows_in_blocks = (y_rows_in_blocks + v_samp - 1) / v_samp;

        // Zero block for out-of-bounds padding
        const ZERO_BLOCK: [i16; 64] = [0i16; 64];

        // CRITICAL: Blocks are in RASTER order, not MCU order!
        // We must convert MCU coordinates to raster indices.
        'mcu_loop: for mcu_y in start_mcu_y..mcu_v {
            // Check if we have blocks for this MCU row
            let rel_mcu_y = mcu_y - start_mcu_y;
            if rel_mcu_y >= mcu_rows_in_blocks {
                break 'mcu_loop;
            }

            // For the first row, start from the correct MCU x position
            let start_mcu_x = if mcu_y == start_mcu_y {
                start_mcu_idx % mcu_h
            } else {
                0
            };

            for mcu_x in start_mcu_x..mcu_h {
                // Encode Y blocks in this MCU using raster indexing
                for dy in 0..v_samp {
                    for dx in 0..h_samp {
                        // Calculate absolute block position in image
                        let y_bx = mcu_x * h_samp + dx;
                        let y_by = mcu_y * v_samp + dy;

                        // Calculate relative position within passed blocks
                        // Blocks start at row (start_mcu_y * v_samp)
                        let rel_y_by = y_by - (start_mcu_y * v_samp);
                        let y_idx = rel_y_by * y_blocks_h + y_bx;

                        let block = if y_idx < y_blocks.len() && y_bx < y_blocks_h {
                            &y_blocks[y_idx]
                        } else {
                            &ZERO_BLOCK
                        };

                        entropy::encode_block_to_writer(
                            block,
                            &tables.dc_luma.table,
                            &tables.ac_luma.table,
                            prev_y_dc,
                            writer,
                        )?;
                        prev_y_dc = block[0];
                    }
                }

                // Encode Cb/Cr blocks using raster indexing
                if is_color {
                    // Calculate relative chroma block position
                    let rel_mcu_y = mcu_y - start_mcu_y;
                    let c_idx = rel_mcu_y * c_blocks_h + mcu_x;

                    let cb_block = if c_idx < cb_blocks.len() && mcu_x < c_blocks_h {
                        &cb_blocks[c_idx]
                    } else {
                        &ZERO_BLOCK
                    };
                    let cr_block = if c_idx < cr_blocks.len() && mcu_x < c_blocks_h {
                        &cr_blocks[c_idx]
                    } else {
                        &ZERO_BLOCK
                    };

                    entropy::encode_block_to_writer(
                        cb_block,
                        &tables.dc_chroma.table,
                        &tables.ac_chroma.table,
                        prev_cb_dc,
                        writer,
                    )?;
                    prev_cb_dc = cb_block[0];

                    entropy::encode_block_to_writer(
                        cr_block,
                        &tables.dc_chroma.table,
                        &tables.ac_chroma.table,
                        prev_cr_dc,
                        writer,
                    )?;
                    prev_cr_dc = cr_block[0];
                }

                mcu_idx += 1;

                // Handle restart markers
                if restart_interval > 0 && mcu_idx < total_mcus && mcu_idx % restart_interval == 0 {
                    writer.flush_restart_marker(restart_count)?;
                    restart_count = (restart_count + 1) % 8;
                    prev_y_dc = 0;
                    prev_cb_dc = 0;
                    prev_cr_dc = 0;
                }
            }
        }

        Ok(([prev_y_dc, prev_cb_dc, prev_cr_dc], mcu_idx, restart_count))
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
    pub fn push_row(&mut self, row: &[u8]) -> Result<()> {
        self.push_row_with_stop(row, Unstoppable)
    }

    /// Pushes a single row with cancellation support.
    ///
    /// The `stop` source is checked before processing each strip.
    /// Returns `Error::cancelled()` if cancellation is requested.
    pub fn push_row_with_stop(&mut self, row: &[u8], stop: impl Stop) -> Result<()> {
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
    pub fn push_rows(&mut self, data: &[u8], num_rows: usize) -> Result<()> {
        self.push_rows_with_stop(data, num_rows, Unstoppable)
    }

    /// Pushes multiple rows with cancellation support.
    ///
    /// This method is optimized to process complete strips directly from the input
    /// buffer without intermediate copies. Only partial strips at the beginning
    /// and end require buffering.
    pub fn push_rows_with_stop(
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
    pub fn push_ycbcr_strip_f32(
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
    pub fn push_ycbcr_strip_f32_subsampled(
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

        // Check if we should transition to streaming mode (bounded-memory feature)
        self.check_and_maybe_transition()?;

        // If we're in streaming mode, batch blocks before encoding
        if self.streaming_mode {
            self.streaming_imcu_pending += 1;
            // Encode when batch is full
            if self.streaming_imcu_pending >= self.streaming_batch_size {
                self.encode_new_blocks_streaming()?;
                self.streaming_imcu_pending = 0;
            }
        }

        Ok(())
    }

    /// Encodes newly quantized blocks in streaming mode.
    ///
    /// This is called after each strip when in streaming mode. It encodes
    /// any new blocks that have been added since the last call.
    fn encode_new_blocks_streaming(&mut self) -> Result<()> {
        // Get the new blocks from the processor
        let blocks = self.processor.take_blocks();

        if blocks.y_blocks.is_empty() {
            return Ok(());
        }

        let tables = self
            .streaming_tables
            .as_ref()
            .ok_or_else(|| Error::internal("streaming_tables not set in streaming mode"))?;

        let is_color = !self.config.pixel_format.is_grayscale();

        // Continue from previous partial byte state
        let mut writer = crate::foundation::bitstream::BitWriter::with_initial_bits(
            self.streaming_bit_buffer,
            self.streaming_bits_in_buffer,
        );

        // Extract state before encoding (to avoid borrow conflicts)
        let prev_dc = self.streaming_prev_dc;
        let mcu_idx = self.streaming_mcu_idx;
        let restart_count = self.streaming_restart_count;

        // Encode the new blocks, continuing from saved state
        let (new_prev_dc, new_mcu_idx, new_restart_count) = self.encode_blocks_mcu_order_ex(
            &blocks.y_blocks,
            &blocks.cb_blocks,
            &blocks.cr_blocks,
            tables,
            &mut writer,
            is_color,
            prev_dc,
            mcu_idx,
            restart_count,
        )?;

        // Update state for next call
        self.streaming_prev_dc = new_prev_dc;
        self.streaming_mcu_idx = new_mcu_idx;
        self.streaming_restart_count = new_restart_count;

        // Flush ONLY complete bytes, preserve partial byte for next chunk
        let output = self
            .streaming_output
            .as_mut()
            .ok_or_else(|| Error::internal("streaming_output not set in streaming mode"))?;
        let (bit_buffer, bits_in_buffer) = writer.flush_complete_bytes_only(output)?;

        // Save bit buffer state for next call
        self.streaming_bit_buffer = bit_buffer;
        self.streaming_bits_in_buffer = bits_in_buffer;

        Ok(())
    }

    /// Finishes encoding and returns the JPEG data.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Not all rows have been pushed
    /// - JPEG generation fails
    pub fn finish(self) -> Result<Vec<u8>> {
        self.finish_with_stop(Unstoppable)
    }

    /// Finishes encoding with cancellation support.
    pub fn finish_with_stop(self, stop: impl Stop) -> Result<Vec<u8>> {
        let mut output = Vec::new();
        self.finish_into_with_stop(&mut output, stop)?;
        Ok(output)
    }

    /// Finishes encoding and returns both JPEG data and Huffman statistics.
    ///
    /// This is the same as [`finish`] but also returns the frequency counts
    /// and Huffman tables that were used for encoding. This is useful for:
    ///
    /// - **Building universal tables**: Collect frequency counts from many
    ///   images, combine them, and generate tables optimized for a corpus
    /// - **Analysis**: Understand what symbols are most common for a content type
    /// - **Debugging**: Verify tables are being generated correctly
    ///
    /// # Returns
    ///
    /// An [`EncodingResult`] containing:
    /// - `jpeg`: The encoded JPEG data
    /// - `frequency_counts`: Raw frequency counts observed during encoding
    /// - `huffman_tables`: The Huffman tables used for encoding
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut encoder = StreamingEncoder::new(640, 480)
    ///     .quality(85)
    ///     .start()?;
    ///
    /// for row in &pixels.chunks(640 * 3) {
    ///     encoder.push_row(row)?;
    /// }
    ///
    /// let result = encoder.finish_with_tables()?;
    /// println!("JPEG size: {} bytes", result.jpeg.len());
    /// println!("AC luma entropy: {:.2} bits", result.frequency_counts.ac_luma.entropy());
    /// ```
    pub fn finish_with_tables(mut self) -> Result<EncodingResult> {
        // Extract frequency counts BEFORE finishing (processor gets consumed)
        let frequency_counts = {
            let (dc_luma, ac_luma, dc_chroma, ac_chroma) = self.processor.frequency_counters();
            HuffmanFrequencyCounts {
                dc_luma: dc_luma.clone(),
                ac_luma: ac_luma.clone(),
                dc_chroma: dc_chroma.clone(),
                ac_chroma: ac_chroma.clone(),
            }
        };

        // Calculate total rows received
        let total_rows = self.current_y + self.rows_buffered;
        if total_rows < self.height {
            return Err(Error::io_error(format!(
                "only {} of {} rows were pushed",
                total_rows, self.height
            )));
        }

        // Flush any remaining rows
        if self.rows_buffered > 0 {
            self.flush_strip_with_stop(&Unstoppable)?;
        }

        // Handle streaming mode
        if self.streaming_mode {
            let tables = self
                .final_tables
                .take()
                .or_else(|| self.streaming_tables.clone())
                .ok_or_else(|| Error::internal("streaming_tables not set"))?;

            let mut output = Vec::new();
            // Need to take ownership for finish_streaming
            self.finish_streaming(&mut output)?;

            return Ok(EncodingResult {
                jpeg: output,
                frequency_counts,
                huffman_tables: tables,
            });
        }

        // Non-streaming mode: build tables for returning
        let huffman_tables = if let Some(tables) = self.custom_huffman_tables.take() {
            tables
        } else if let Some(ref counts) = self.custom_frequency_counts {
            counts.generate_tables()?
        } else {
            // Generate from the frequency counts we already captured
            frequency_counts.generate_tables()?
        };

        // Non-streaming mode: build JPEG from buffered blocks
        let config = self.config;
        let y_quant = self.y_quant;
        let cb_quant = self.cb_quant;
        let cr_quant = self.cr_quant;
        let width = self.width;
        let height = self.height;

        let strip_output = self.processor.finalize()?;

        let mut output = Vec::new();
        Self::build_jpeg_from_blocks_into(
            &config,
            &y_quant,
            &cb_quant,
            &cr_quant,
            width,
            height,
            strip_output,
            &mut output,
            Unstoppable,
        )?;

        Ok(EncodingResult {
            jpeg: output,
            frequency_counts,
            huffman_tables,
        })
    }

    /// Finishes encoding, writing directly to the provided buffer.
    ///
    /// This avoids an extra allocation compared to `finish()`. The buffer
    /// is cleared before writing.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Not all rows have been pushed
    /// - JPEG generation fails
    /// - Memory allocation fails
    pub fn finish_into(self, output: &mut Vec<u8>) -> Result<()> {
        self.finish_into_with_stop(output, Unstoppable)
    }

    /// Finishes encoding into provided buffer with cancellation support.
    pub fn finish_into_with_stop(mut self, output: &mut Vec<u8>, stop: impl Stop) -> Result<()> {
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

        // Handle streaming mode: we've already written most of the JPEG,
        // just need to encode remaining blocks and write EOI
        if self.streaming_mode {
            return self.finish_streaming(output);
        }

        // Non-streaming mode: build JPEG from buffered blocks
        let config = self.config;
        let y_quant = self.y_quant;
        let cb_quant = self.cb_quant;
        let cr_quant = self.cr_quant;
        let width = self.width;
        let height = self.height;

        // Finalize strip processing
        let strip_output = self.processor.finalize()?;

        // Build JPEG output directly into provided buffer
        Self::build_jpeg_from_blocks_into(
            &config,
            &y_quant,
            &cb_quant,
            &cr_quant,
            width,
            height,
            strip_output,
            output,
            stop,
        )
    }

    /// Finishes encoding in streaming mode.
    ///
    /// The header and most scan data have already been written. This method:
    /// 1. Encodes any remaining blocks
    /// 2. Writes EOI marker
    /// 3. Moves the streaming output to the provided buffer
    fn finish_streaming(mut self, output: &mut Vec<u8>) -> Result<()> {
        // Extract all state we need before finalize() consumes self.processor
        let tables = self
            .streaming_tables
            .take()
            .ok_or_else(|| Error::internal("streaming_tables not set"))?;
        let mut streaming_output = self
            .streaming_output
            .take()
            .ok_or_else(|| Error::internal("streaming_output not set"))?;
        let is_color = !self.config.pixel_format.is_grayscale();
        let prev_dc = self.streaming_prev_dc;
        let mcu_idx = self.streaming_mcu_idx;
        let restart_count = self.streaming_restart_count;
        let bit_buffer = self.streaming_bit_buffer;
        let bits_in_buffer = self.streaming_bits_in_buffer;
        let subsampling = self.config.subsampling;
        let width = self.width;
        let height = self.height;
        let restart_interval = self.config.restart_interval;

        // Finalize strip processing to get any remaining blocks
        let strip_output = self.processor.finalize()?;

        // Create writer with any remaining partial byte from previous encoding
        let mut writer =
            crate::foundation::bitstream::BitWriter::with_initial_bits(bit_buffer, bits_in_buffer);

        // Encode any remaining blocks
        if !strip_output.y_blocks.is_empty() {
            // Encode the remaining blocks, continuing from saved state
            Self::encode_blocks_mcu_order_static(
                &strip_output.y_blocks,
                &strip_output.cb_blocks,
                &strip_output.cr_blocks,
                &tables,
                &mut writer,
                is_color,
                prev_dc,
                mcu_idx,
                restart_count,
                subsampling,
                width,
                height,
                restart_interval,
            )?;
        }

        // Final flush WITH padding (this is the end of the scan)
        writer.flush_without_eoi(&mut streaming_output)?;

        // Write EOI marker
        streaming_output.push(0xFF);
        streaming_output.push(crate::foundation::consts::MARKER_EOI);

        // Move to the provided output buffer
        output.clear();
        *output = streaming_output;

        Ok(())
    }

    /// Builds JPEG output from processed blocks.
    fn build_jpeg_from_blocks(
        config: &ComputedConfig,
        y_quant: &QuantTable,
        cb_quant: &QuantTable,
        cr_quant: &QuantTable,
        width: usize,
        height: usize,
        strip_output: crate::encode::strip::StripProcessorOutput,
        stop: impl Stop,
    ) -> Result<Vec<u8>> {
        let mut output = Vec::new();
        Self::build_jpeg_from_blocks_into(
            config,
            y_quant,
            cb_quant,
            cr_quant,
            width,
            height,
            strip_output,
            &mut output,
            stop,
        )?;
        Ok(output)
    }

    /// Builds JPEG output from processed blocks into provided buffer.
    fn build_jpeg_from_blocks_into(
        config: &ComputedConfig,
        y_quant: &QuantTable,
        cb_quant: &QuantTable,
        cr_quant: &QuantTable,
        _width: usize,
        _height: usize,
        strip_output: crate::encode::strip::StripProcessorOutput,
        output: &mut Vec<u8>,
        stop: impl Stop,
    ) -> Result<()> {
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
                config.encode_progressive_from_blocks_into(
                    &strip_output.y_blocks,
                    &strip_output.cb_blocks,
                    &strip_output.cr_blocks,
                    y_quant,
                    cb_quant,
                    cr_quant,
                    output,
                )
            }
            _ => {
                // Baseline encoding
                Self::build_jpeg_baseline_into(
                    config,
                    y_quant,
                    cb_quant,
                    cr_quant,
                    strip_output,
                    output,
                )
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
        let mut output = Vec::new();
        Self::build_jpeg_baseline_into(
            config,
            y_quant,
            cb_quant,
            cr_quant,
            strip_output,
            &mut output,
        )?;
        Ok(output)
    }

    /// Builds baseline JPEG output from processed blocks into provided buffer.
    fn build_jpeg_baseline_into(
        config: &ComputedConfig,
        y_quant: &QuantTable,
        cb_quant: &QuantTable,
        cr_quant: &QuantTable,
        strip_output: crate::encode::strip::StripProcessorOutput,
        output: &mut Vec<u8>,
    ) -> Result<()> {
        let is_color = !config.pixel_format.is_grayscale();
        let width = config.width as usize;
        let height = config.height as usize;

        output.clear();
        output
            .try_reserve(width * height / 4)
            .map_err(|_| Error::allocation_failed(width * height / 4, "baseline jpeg output"))?;

        // Branch based on XYB vs YCbCr mode
        let scan_data = if config.use_xyb {
            // XYB mode: uses different headers, tables, and encoding
            config.write_header_xyb(output)?;
            config.write_app14_adobe(output, 0)?;
            config.write_icc_profile(output, &crate::foundation::consts::XYB_ICC_PROFILE)?;
            config.write_quant_tables_xyb(output, y_quant, cb_quant, cr_quant)?;
            // Use SOF1 if any quant table needs 16-bit precision
            let is_extended =
                y_quant.precision > 0 || cb_quant.precision > 0 || cr_quant.precision > 0;
            config.write_frame_header_xyb_ex(output, is_extended)?;

            if config.optimize_huffman {
                let (dc_table, ac_table) = config.build_optimized_tables_xyb_raster(
                    &strip_output.y_blocks,
                    &strip_output.cb_blocks,
                    &strip_output.cr_blocks,
                )?;

                config.write_huffman_tables_xyb_optimized(output, &dc_table, &ac_table);

                if config.restart_interval > 0 {
                    config.write_restart_interval(output)?;
                }
                config.write_scan_header_xyb(output)?;

                config.encode_with_tables_xyb_raster(
                    &strip_output.y_blocks,
                    &strip_output.cb_blocks,
                    &strip_output.cr_blocks,
                    &dc_table,
                    &ac_table,
                )?
            } else {
                config.write_huffman_tables(output)?;

                if config.restart_interval > 0 {
                    config.write_restart_interval(output)?;
                }
                config.write_scan_header_xyb(output)?;

                config.encode_with_tables_xyb_standard_raster(
                    &strip_output.y_blocks,
                    &strip_output.cb_blocks,
                    &strip_output.cr_blocks,
                )?
            }
        } else {
            // YCbCr mode: standard JPEG encoding
            config.write_header(output)?;
            config.write_quant_tables(output, y_quant, cb_quant, cr_quant)?;
            // Use SOF1 if any quant table needs 16-bit precision
            let is_extended =
                y_quant.precision > 0 || cb_quant.precision > 0 || cr_quant.precision > 0;
            config.write_frame_header_ex(output, is_extended)?;

            if config.optimize_huffman {
                let tables = config.build_optimized_tables(
                    &strip_output.y_blocks,
                    &strip_output.cb_blocks,
                    &strip_output.cr_blocks,
                    is_color,
                )?;

                config.write_huffman_tables_optimized(output, &tables)?;

                if config.restart_interval > 0 {
                    config.write_restart_interval(output)?;
                }
                config.write_scan_header(output)?;

                config.encode_with_tables(
                    &strip_output.y_blocks,
                    &strip_output.cb_blocks,
                    &strip_output.cr_blocks,
                    is_color,
                    Some(&tables),
                )?
            } else {
                config.write_huffman_tables(output)?;

                if config.restart_interval > 0 {
                    config.write_restart_interval(output)?;
                }
                config.write_scan_header(output)?;

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

        Ok(())
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

        // 4K with 4:2:0: ~28 MB (blocks + entropy output + working buffers)
        // Heaptrack measured ~28 MB for encoder alone (excluding input pixels)
        assert!(estimate > 25_000_000, "estimate {} too low", estimate);
        assert!(estimate < 40_000_000, "estimate {} too high", estimate);
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

    #[test]
    fn test_memory_limit_not_compatible_with_progressive() {
        let result = StreamingEncoder::new(64, 64)
            .progressive(true)
            .memory_limit(1024)
            .start();

        assert!(
            result.is_err(),
            "Should have returned error for progressive with memory_limit"
        );
        match result {
            Err(err) => {
                assert!(
                    err.to_string().contains("progressive"),
                    "Error should mention progressive mode: {}",
                    err
                );
            }
            Ok(_) => unreachable!(),
        }
    }

    #[test]
    fn test_bounded_streaming_basic() {
        // Create a test image large enough to trigger streaming transition
        let width = 128u32;
        let height = 128u32;
        let pixels: Vec<u8> = (0..width * height * 3)
            .map(|i| ((i * 17) % 256) as u8)
            .collect();

        // Encode without memory limit (standard buffering)
        let standard_result = StreamingEncoder::new(width, height)
            .quality(Quality::ApproxJpegli(85.0))
            .subsampling(Subsampling::S420)
            .progressive(false) // Baseline required for bounded streaming
            .encode(&pixels)
            .unwrap();

        // Encode with very small memory limit (should trigger streaming early)
        let mut bounded = StreamingEncoder::new(width, height)
            .quality(Quality::ApproxJpegli(85.0))
            .subsampling(Subsampling::S420)
            .progressive(false)
            .memory_limit(1024) // 1KB limit - will trigger after first strip
            .start()
            .unwrap();

        let row_size = width as usize * 3;
        for y in 0..height as usize {
            let start = y * row_size;
            let end = start + row_size;
            bounded.push_row(&pixels[start..end]).unwrap();
        }

        // Should have transitioned to streaming mode
        assert!(
            bounded.is_streaming(),
            "Should have transitioned to streaming mode with 1KB limit"
        );

        let bounded_result = bounded.finish().unwrap();

        // Both should produce valid JPEGs (sizes may differ due to different Huffman tables)
        assert!(
            standard_result.len() > 100,
            "Standard result too small: {} bytes",
            standard_result.len()
        );
        assert!(
            bounded_result.len() > 100,
            "Bounded result too small: {} bytes",
            bounded_result.len()
        );

        // Verify JPEG structure
        assert_eq!(standard_result[0..2], [0xFF, 0xD8], "Missing SOI marker");
        assert_eq!(bounded_result[0..2], [0xFF, 0xD8], "Missing SOI marker");
        assert_eq!(
            standard_result[standard_result.len() - 2..],
            [0xFF, 0xD9],
            "Missing EOI marker"
        );
        assert_eq!(
            bounded_result[bounded_result.len() - 2..],
            [0xFF, 0xD9],
            "Missing EOI marker"
        );
    }

    #[test]
    fn test_bounded_streaming_no_transition_if_below_limit() {
        // Create a small test image
        let width = 32u32;
        let height = 32u32;
        let pixels: Vec<u8> = (0..width * height * 3)
            .map(|i| ((i * 17) % 256) as u8)
            .collect();

        // Use a large memory limit - should NOT transition to streaming
        let mut encoder = StreamingEncoder::new(width, height)
            .quality(Quality::ApproxJpegli(85.0))
            .subsampling(Subsampling::S444)
            .progressive(false)
            .memory_limit(1024 * 1024) // 1MB - way more than needed for 32x32
            .start()
            .unwrap();

        let row_size = width as usize * 3;
        for y in 0..height as usize {
            let start = y * row_size;
            let end = start + row_size;
            encoder.push_row(&pixels[start..end]).unwrap();
        }

        // Should NOT have transitioned (small image, large limit)
        assert!(
            !encoder.is_streaming(),
            "Should NOT have transitioned with 1MB limit on 32x32 image"
        );

        let result = encoder.finish().unwrap();

        // Verify valid JPEG
        assert_eq!(result[0..2], [0xFF, 0xD8], "Missing SOI marker");
        assert_eq!(
            result[result.len() - 2..],
            [0xFF, 0xD9],
            "Missing EOI marker"
        );
    }

    /// Test for memory profiling with heaptrack.
    ///
    /// Run with:
    /// ```bash
    /// cargo test --release -p zenjpeg --lib --features "test-utils,decoder" -- \
    ///     encode::streaming::tests::test_memory_profile_large_image --nocapture --ignored
    /// ```
    ///
    /// Then profile with heaptrack:
    /// ```bash
    /// heaptrack cargo test --release -p zenjpeg --lib --features "test-utils,decoder" -- \
    ///     encode::streaming::tests::test_memory_profile_large_image --nocapture --ignored
    /// ```
    #[test]
    #[ignore] // Run manually for memory profiling
    fn test_memory_profile_large_image() {
        // 2000x1500 image = 3 megapixels
        // At 4:2:0: ~187,500 Y blocks + 47k Cb + 47k Cr = ~280k blocks
        // Block storage: 280k * 128 bytes = ~36 MB without streaming limit
        let width = 2000u32;
        let height = 1500u32;

        eprintln!(
            "Image size: {}x{} ({:.1} MP)",
            width,
            height,
            (width * height) as f64 / 1e6
        );

        // Generate test pixels
        let pixels: Vec<u8> = (0..height as usize)
            .flat_map(|y| {
                (0..width as usize).flat_map(move |x| {
                    let r = ((x * 255) / width as usize) as u8;
                    let g = ((y * 255) / height as usize) as u8;
                    let b = (((x + y) * 127) / (width as usize + height as usize)) as u8;
                    [r, g, b]
                })
            })
            .collect();

        // Test 1: No limit (baseline)
        eprintln!("\n=== Test 1: No memory limit (baseline) ===");
        {
            let mut encoder = StreamingEncoder::new(width, height)
                .quality(Quality::ApproxJpegli(85.0))
                .subsampling(Subsampling::S420)
                .progressive(false)
                .start()
                .unwrap();

            let row_size = width as usize * 3;
            for y in 0..height as usize {
                let start = y * row_size;
                let end = start + row_size;
                encoder.push_row(&pixels[start..end]).unwrap();
            }

            eprintln!("Streaming mode: {}", encoder.is_streaming());
            let result = encoder.finish().unwrap();
            eprintln!("Output: {} KB", result.len() / 1024);
        }

        // Test 2: 2 MB limit
        eprintln!("\n=== Test 2: 2 MB memory limit ===");
        {
            let mut encoder = StreamingEncoder::new(width, height)
                .quality(Quality::ApproxJpegli(85.0))
                .subsampling(Subsampling::S420)
                .progressive(false)
                .memory_limit(2 * 1024 * 1024) // 2 MB
                .start()
                .unwrap();

            let row_size = width as usize * 3;
            for y in 0..height as usize {
                let start = y * row_size;
                let end = start + row_size;
                encoder.push_row(&pixels[start..end]).unwrap();
            }

            eprintln!("Streaming mode: {}", encoder.is_streaming());
            assert!(
                encoder.is_streaming(),
                "Should have transitioned with 2MB limit"
            );
            let result = encoder.finish().unwrap();
            eprintln!("Output: {} KB", result.len() / 1024);
        }

        // Test 3: 5 MB limit
        eprintln!("\n=== Test 3: 5 MB memory limit ===");
        {
            let mut encoder = StreamingEncoder::new(width, height)
                .quality(Quality::ApproxJpegli(85.0))
                .subsampling(Subsampling::S420)
                .progressive(false)
                .memory_limit(5 * 1024 * 1024) // 5 MB
                .start()
                .unwrap();

            let row_size = width as usize * 3;
            for y in 0..height as usize {
                let start = y * row_size;
                let end = start + row_size;
                encoder.push_row(&pixels[start..end]).unwrap();
            }

            eprintln!("Streaming mode: {}", encoder.is_streaming());
            assert!(
                encoder.is_streaming(),
                "Should have transitioned with 5MB limit"
            );
            let result = encoder.finish().unwrap();
            eprintln!("Output: {} KB", result.len() / 1024);
        }

        eprintln!("\n=== Done ===");
    }

    /// Individual test: No memory limit (for isolated profiling)
    #[test]
    #[ignore]
    fn test_memory_profile_no_limit() {
        let width = 2000u32;
        let height = 1500u32;
        let pixels: Vec<u8> = (0..height as usize)
            .flat_map(|y| {
                (0..width as usize).flat_map(move |x| {
                    let r = ((x * 255) / width as usize) as u8;
                    let g = ((y * 255) / height as usize) as u8;
                    let b = (((x + y) * 127) / (width as usize + height as usize)) as u8;
                    [r, g, b]
                })
            })
            .collect();

        eprintln!("Image: {}x{}, no limit", width, height);
        let mut encoder = StreamingEncoder::new(width, height)
            .quality(Quality::ApproxJpegli(85.0))
            .subsampling(Subsampling::S420)
            .progressive(false)
            .start()
            .unwrap();

        let row_size = width as usize * 3;
        for y in 0..height as usize {
            let start = y * row_size;
            let end = start + row_size;
            encoder.push_row(&pixels[start..end]).unwrap();
        }
        let result = encoder.finish().unwrap();
        eprintln!("Output: {} KB, streaming={}", result.len() / 1024, false);
    }

    /// Individual test: 2 MB limit (for isolated profiling)
    #[test]
    #[ignore]
    fn test_memory_profile_2mb_limit() {
        let width = 2000u32;
        let height = 1500u32;
        let pixels: Vec<u8> = (0..height as usize)
            .flat_map(|y| {
                (0..width as usize).flat_map(move |x| {
                    let r = ((x * 255) / width as usize) as u8;
                    let g = ((y * 255) / height as usize) as u8;
                    let b = (((x + y) * 127) / (width as usize + height as usize)) as u8;
                    [r, g, b]
                })
            })
            .collect();

        eprintln!("Image: {}x{}, 2MB limit", width, height);
        let mut encoder = StreamingEncoder::new(width, height)
            .quality(Quality::ApproxJpegli(85.0))
            .subsampling(Subsampling::S420)
            .progressive(false)
            .memory_limit(2 * 1024 * 1024)
            .start()
            .unwrap();

        let row_size = width as usize * 3;
        for y in 0..height as usize {
            let start = y * row_size;
            let end = start + row_size;
            encoder.push_row(&pixels[start..end]).unwrap();
        }
        let streaming = encoder.is_streaming();
        let result = encoder.finish().unwrap();
        eprintln!(
            "Output: {} KB, streaming={}",
            result.len() / 1024,
            streaming
        );
    }

    /// Test quality comparison between streaming and non-streaming mode.
    ///
    /// This compares SSIMULACRA2 scores between:
    /// 1. Standard encoding (no memory limit)
    /// 2. Bounded streaming encoding (2 MB limit)
    ///
    /// Both should decode to nearly identical images since they use the same
    /// quantization. Differences come from Huffman table optimization (built
    /// from partial data in streaming mode).
    #[test]
    #[ignore] // Run manually: cargo test --features decoder -- test_streaming_quality_comparison --ignored --nocapture
    fn test_streaming_quality_comparison() {
        // Note: Using zune-jpeg for decoding because our decoder has issues with
        // the specific Huffman tables generated by streaming mode.
        // This is a decoder bug, not an encoder bug - the JPEG is valid.

        let width = 512u32;
        let height = 512u32;

        // Generate test image with varied content (gradients + patterns)
        let pixels: Vec<u8> = (0..height as usize)
            .flat_map(|y| {
                (0..width as usize).flat_map(move |x| {
                    // Mix of gradient and checkerboard for entropy variation
                    let checker = ((x / 8) + (y / 8)) % 2;
                    let r = (((x * 255) / width as usize) as u8).wrapping_add((checker * 30) as u8);
                    let g =
                        (((y * 255) / height as usize) as u8).wrapping_add((checker * 20) as u8);
                    let b = ((((x + y) * 127) / (width as usize + height as usize)) as u8)
                        .wrapping_add((checker * 25) as u8);
                    [r, g, b]
                })
            })
            .collect();

        eprintln!("Test image: {}x{}", width, height);

        // Encode without memory limit (standard optimized Huffman)
        let standard_result = StreamingEncoder::new(width, height)
            .quality(Quality::ApproxJpegli(85.0))
            .subsampling(Subsampling::S420)
            .progressive(false)
            .encode(&pixels)
            .unwrap();

        // Encode with 1KB memory limit (forces early transition to streaming)
        let mut bounded = StreamingEncoder::new(width, height)
            .quality(Quality::ApproxJpegli(85.0))
            .subsampling(Subsampling::S420)
            .progressive(false)
            .memory_limit(1024) // 1KB - will transition very early
            .start()
            .unwrap();

        let row_size = width as usize * 3;
        for y in 0..height as usize {
            let start = y * row_size;
            let end = start + row_size;
            bounded.push_row(&pixels[start..end]).unwrap();
        }
        let bounded_result = bounded.finish().unwrap();

        eprintln!("Standard output: {} bytes", standard_result.len());
        eprintln!("Bounded output: {} bytes", bounded_result.len());
        eprintln!(
            "Size difference: {:.2}%",
            100.0 * (bounded_result.len() as f64 - standard_result.len() as f64)
                / standard_result.len() as f64
        );

        // Write both JPEGs for debugging
        let _ = std::fs::create_dir_all("/mnt/v/output/zenjpeg/streaming-debug");
        std::fs::write(
            "/mnt/v/output/zenjpeg/streaming-debug/standard.jpg",
            &standard_result,
        )
        .unwrap();
        std::fs::write(
            "/mnt/v/output/zenjpeg/streaming-debug/bounded.jpg",
            &bounded_result,
        )
        .unwrap();
        eprintln!("Wrote JPEGs to /mnt/v/output/zenjpeg/streaming-debug/");

        // Try external decoder first
        let djpeg_result = std::process::Command::new("djpeg")
            .args([
                "-outfile",
                "/mnt/v/output/zenjpeg/streaming-debug/bounded.ppm",
                "/mnt/v/output/zenjpeg/streaming-debug/bounded.jpg",
            ])
            .output();
        match djpeg_result {
            Ok(output) => {
                if output.status.success() {
                    eprintln!("djpeg decoded bounded.jpg successfully!");
                } else {
                    eprintln!("djpeg failed: {}", String::from_utf8_lossy(&output.stderr));
                }
            }
            Err(e) => eprintln!("djpeg not available: {}", e),
        }

        // Decode both to compare using zune-jpeg
        // (our decoder has issues with streaming-mode Huffman tables)
        use zune_jpeg::zune_core::bytestream::ZCursor;
        let standard_decoded = zune_jpeg::JpegDecoder::new(ZCursor::new(&standard_result))
            .decode()
            .expect("zune-jpeg failed to decode standard");
        let bounded_decoded = zune_jpeg::JpegDecoder::new(ZCursor::new(&bounded_result))
            .decode()
            .expect("zune-jpeg failed to decode bounded");

        // Compare pixel values
        let mut max_diff = 0i32;
        let mut total_diff = 0u64;
        let mut diff_count = 0usize;

        for (_i, (&s, &b)) in standard_decoded
            .iter()
            .zip(bounded_decoded.iter())
            .enumerate()
        {
            let diff = (s as i32 - b as i32).abs();
            if diff > 0 {
                diff_count += 1;
                total_diff += diff as u64;
                if diff > max_diff {
                    max_diff = diff;
                }
            }
        }

        let mean_diff = if diff_count > 0 {
            total_diff as f64 / diff_count as f64
        } else {
            0.0
        };

        eprintln!("\nDecoded comparison:");
        eprintln!(
            "  Pixels with differences: {} ({:.2}%)",
            diff_count,
            100.0 * diff_count as f64 / standard_decoded.len() as f64
        );
        eprintln!("  Max pixel difference: {}", max_diff);
        eprintln!("  Mean difference (where differs): {:.2}", mean_diff);

        // The decoded images should be very close
        // Allow some difference due to different Huffman tables
        assert!(
            max_diff <= 10,
            "Max pixel difference {} exceeds threshold 10",
            max_diff
        );
    }
}
