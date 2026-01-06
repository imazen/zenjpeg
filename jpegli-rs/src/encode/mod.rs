//! JPEG encoder implementation.
//!
//! This module provides the main encoder interface for creating JPEG images.

pub mod config;
mod baseline;
mod color;
#[cfg(feature = "experimental-hybrid-trellis")]
mod hybrid;
mod output;
mod progressive;

// Re-export config types
pub use config::{internal_pathway, EncoderConfig};
pub(crate) use config::{DownsamplingMethod, InternalPipeline, ProgressiveScan};
#[cfg(test)]
pub(crate) use config::ColorConversionMethod;
#[cfg(feature = "experimental-hybrid-trellis")]
pub(crate) use hybrid::HybridQuantContext;

use crate::adaptive_quant::compute_aq_strength_map;
use crate::alloc::{checked_size_2d, validate_dimensions, DEFAULT_MAX_PIXELS};
use crate::chroma;
use crate::consts::{DCT_BLOCK_SIZE, DCT_SIZE, JPEG_ZIGZAG_ORDER, MARKER_EOI, XYB_ICC_PROFILE};
#[cfg(test)]
use crate::consts::MARKER_SOI;
use crate::dct::forward_dct_8x8;
use crate::entropy::{self, EntropyEncoder};
use crate::error::{Error, Result};
use crate::huffman::HuffmanEncodeTable;
use crate::huffman_opt::{
    ContextConfig, FrequencyCounter, OptimizedHuffmanTables, OptimizedTable, ProgressiveTokenBuffer,
};
use crate::quant::{self, Quality, QuantTable, ZeroBiasParams};
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
            self.config.hybrid_config = crate::hybrid_config::HybridConfig::default();
        } else {
            self.config.hybrid_config = crate::hybrid_config::HybridConfig::disabled();
        }
        self
    }

    /// Set custom hybrid quantization configuration.
    ///
    /// Allows fine-tuning all hybrid AQ+trellis parameters.
    /// See [`HybridConfig`](crate::hybrid_config::HybridConfig) for available options.
    ///
    /// Requires the `experimental-hybrid-trellis` feature.
    #[cfg(feature = "experimental-hybrid-trellis")]
    #[must_use]
    pub fn hybrid_config(mut self, config: crate::hybrid_config::HybridConfig) -> Self {
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

    // Baseline encoding functions are in baseline.rs



    /// Encodes the scan data (u8 version - legacy).
    #[allow(dead_code)]
    fn encode_scan(
        &self,
        y_plane: &[u8],
        cb_plane: &[u8],
        cr_plane: &[u8],
        y_quant: &QuantTable,
        c_quant: &QuantTable,
    ) -> Result<Vec<u8>> {
        let mut encoder = EntropyEncoder::new();

        // Set up Huffman tables
        encoder.set_dc_table(0, HuffmanEncodeTable::std_dc_luminance());
        encoder.set_ac_table(0, HuffmanEncodeTable::std_ac_luminance());
        encoder.set_dc_table(1, HuffmanEncodeTable::std_dc_chrominance());
        encoder.set_ac_table(1, HuffmanEncodeTable::std_ac_chrominance());

        if self.config.restart_interval > 0 {
            encoder.set_restart_interval(self.config.restart_interval);
        }

        let width = self.config.width as usize;
        let height = self.config.height as usize;

        // For 4:2:0, process MCUs
        let _mcu_width = ((width + 15) / 16) * 16;
        let _mcu_height = ((height + 15) / 16) * 16;

        // TODO: Implement full MCU processing with subsampling
        // For now, simplified 4:4:4 encoding
        let blocks_h = (width + 7) / 8;
        let blocks_v = (height + 7) / 8;

        // Zero-bias parameters for each component
        // Use effective distance inferred from quant tables (like C++ QuantValsToDistance)
        // For YCbCr mode, Cb and Cr share the same quant table (c_quant)
        let _input_distance = self.config.quality.to_distance();
        let effective_distance = quant::quant_vals_to_distance(y_quant, c_quant, c_quant);
        let y_zero_bias = ZeroBiasParams::for_ycbcr(effective_distance, 0);
        let cb_zero_bias = ZeroBiasParams::for_ycbcr(effective_distance, 1);
        let cr_zero_bias = ZeroBiasParams::for_ycbcr(effective_distance, 2);

        // Convert Y plane to f32 for AQ computation (SIMD)
        let y_plane_f32 = crate::encode_simd::u8_slice_to_f32_simd(y_plane);

        // Compute per-block adaptive quantization strength from Y plane
        // C++ uses y_quant_01 = quant_table[1] for dampen calculation
        let y_quant_01 = y_quant.values[1];
        #[cfg(feature = "experimental-hybrid-trellis")]
        let aq_map = hybrid::get_aq_map_or_compute(
            &self.config, &y_plane_f32, width, height, y_quant_01);
        #[cfg(not(feature = "experimental-hybrid-trellis"))]
        let aq_map = compute_aq_strength_map(&y_plane_f32, width, height, y_quant_01);

        // Create hybrid quantization context if enabled
        #[cfg(feature = "experimental-hybrid-trellis")]
        let hybrid_ctx = hybrid::create_hybrid_ctx(&self.config);

        for by in 0..blocks_v {
            for bx in 0..blocks_h {
                // Get per-block aq_strength (C++ AQ produces 0.0-0.2, mean ~0.08)
                let aq_strength = aq_map.get(bx, by);

                // Extract and encode Y block
                let y_block = self.extract_block(y_plane, width, height, bx, by);
                let y_dct = forward_dct_8x8(&y_block);

                #[cfg(feature = "experimental-hybrid-trellis")]
                let y_quant_coeffs = hybrid::quantize_block_dispatch(
                    &y_dct, &y_quant.values, &y_zero_bias, aq_strength, true, hybrid_ctx.as_ref());
                #[cfg(not(feature = "experimental-hybrid-trellis"))]
                let y_quant_coeffs = quant::quantize_block_with_zero_bias_simd(
                    &y_dct, &y_quant.values, &y_zero_bias, aq_strength);

                let y_zigzag = natural_to_zigzag(&y_quant_coeffs);
                encoder.encode_block(&y_zigzag, 0, 0, 0)?;

                if self.config.pixel_format != PixelFormat::Gray {
                    // Cb block
                    let cb_block = self.extract_block(cb_plane, width, height, bx, by);
                    let cb_dct = forward_dct_8x8(&cb_block);

                    #[cfg(feature = "experimental-hybrid-trellis")]
                    let cb_quant_coeffs = hybrid::quantize_block_dispatch(
                        &cb_dct, &c_quant.values, &cb_zero_bias, aq_strength, false, hybrid_ctx.as_ref());
                    #[cfg(not(feature = "experimental-hybrid-trellis"))]
                    let cb_quant_coeffs = quant::quantize_block_with_zero_bias_simd(
                        &cb_dct, &c_quant.values, &cb_zero_bias, aq_strength);

                    let cb_zigzag = natural_to_zigzag(&cb_quant_coeffs);
                    encoder.encode_block(&cb_zigzag, 1, 1, 1)?;

                    // Cr block
                    let cr_block = self.extract_block(cr_plane, width, height, bx, by);
                    let cr_dct = forward_dct_8x8(&cr_block);

                    #[cfg(feature = "experimental-hybrid-trellis")]
                    let cr_quant_coeffs = hybrid::quantize_block_dispatch(
                        &cr_dct, &c_quant.values, &cr_zero_bias, aq_strength, false, hybrid_ctx.as_ref());
                    #[cfg(not(feature = "experimental-hybrid-trellis"))]
                    let cr_quant_coeffs = quant::quantize_block_with_zero_bias_simd(
                        &cr_dct, &c_quant.values, &cr_zero_bias, aq_strength);

                    let cr_zigzag = natural_to_zigzag(&cr_quant_coeffs);
                    encoder.encode_block(&cr_zigzag, 2, 1, 1)?;
                }

                encoder.check_restart();
            }
        }

        Ok(encoder.finish())
    }

    /// Quantizes all blocks in the image.
    ///
    /// This is separated from encoding to allow Huffman optimization:
    /// 1. Quantize all blocks
    /// 2. Collect frequencies to build optimal tables
    /// 3. Encode with optimal tables
    fn quantize_all_blocks(
        &self,
        y_plane: &[f32],
        cb_plane: &[f32],
        cr_plane: &[f32],
        y_quant: &QuantTable,
        cb_quant: &QuantTable,
        cr_quant: &QuantTable,
    ) -> Result<(
        Vec<[i16; DCT_BLOCK_SIZE]>,
        Vec<[i16; DCT_BLOCK_SIZE]>,
        Vec<[i16; DCT_BLOCK_SIZE]>,
    )> {
        let width = self.config.width as usize;
        let height = self.config.height as usize;
        let blocks_h = (width + 7) / 8;
        let blocks_v = (height + 7) / 8;
        let is_color = self.config.pixel_format != PixelFormat::Gray;

        // Zero-bias parameters for each component
        // Use effective distance inferred from quant tables (like C++ QuantValsToDistance)
        // This is important at Q100 where quant values are all 1s but input distance is 0.01
        let _input_distance = self.config.quality.to_distance();
        let effective_distance = quant::quant_vals_to_distance(y_quant, cb_quant, cr_quant);
        let y_zero_bias = ZeroBiasParams::for_ycbcr(effective_distance, 0);
        let cb_zero_bias = ZeroBiasParams::for_ycbcr(effective_distance, 1);
        let cr_zero_bias = ZeroBiasParams::for_ycbcr(effective_distance, 2);

        // Compute per-block adaptive quantization strength from Y plane
        // C++ uses y_quant_01 = quant_table[1] for dampen calculation
        let y_quant_01 = y_quant.values[1];
        #[cfg(feature = "experimental-hybrid-trellis")]
        let aq_map = hybrid::get_aq_map_or_compute(
            &self.config, y_plane, width, height, y_quant_01);
        #[cfg(not(feature = "experimental-hybrid-trellis"))]
        let aq_map = compute_aq_strength_map(y_plane, width, height, y_quant_01);

        // Create hybrid quantization context if enabled
        #[cfg(feature = "experimental-hybrid-trellis")]
        let hybrid_ctx = hybrid::create_hybrid_ctx(&self.config);

        let mut y_blocks = Vec::with_capacity(blocks_h * blocks_v);
        let mut cb_blocks = Vec::with_capacity(if is_color { blocks_h * blocks_v } else { 0 });
        let mut cr_blocks = Vec::with_capacity(if is_color { blocks_h * blocks_v } else { 0 });

        for by in 0..blocks_v {
            for bx in 0..blocks_h {
                // Get per-block aq_strength
                let aq_strength = aq_map.get(bx, by);

                let y_block =
                    crate::encode_simd::extract_block_simd(y_plane, width, height, bx, by);
                let y_dct = forward_dct_8x8(&y_block);

                #[cfg(feature = "experimental-hybrid-trellis")]
                let y_quant_coeffs = hybrid::quantize_block_dispatch(
                    &y_dct, &y_quant.values, &y_zero_bias, aq_strength, true, hybrid_ctx.as_ref());
                #[cfg(not(feature = "experimental-hybrid-trellis"))]
                let y_quant_coeffs = quant::quantize_block_with_zero_bias_simd(
                    &y_dct, &y_quant.values, &y_zero_bias, aq_strength);

                y_blocks.push(natural_to_zigzag(&y_quant_coeffs));

                if is_color {
                    let cb_block =
                        crate::encode_simd::extract_block_simd(cb_plane, width, height, bx, by);
                    let cb_dct = forward_dct_8x8(&cb_block);

                    #[cfg(feature = "experimental-hybrid-trellis")]
                    let cb_quant_coeffs = hybrid::quantize_block_dispatch(
                        &cb_dct, &cb_quant.values, &cb_zero_bias, aq_strength, false, hybrid_ctx.as_ref());
                    #[cfg(not(feature = "experimental-hybrid-trellis"))]
                    let cb_quant_coeffs = quant::quantize_block_with_zero_bias_simd(
                        &cb_dct, &cb_quant.values, &cb_zero_bias, aq_strength);

                    cb_blocks.push(natural_to_zigzag(&cb_quant_coeffs));

                    let cr_block =
                        crate::encode_simd::extract_block_simd(cr_plane, width, height, bx, by);
                    let cr_dct = forward_dct_8x8(&cr_block);

                    #[cfg(feature = "experimental-hybrid-trellis")]
                    let cr_quant_coeffs = hybrid::quantize_block_dispatch(
                        &cr_dct, &cr_quant.values, &cr_zero_bias, aq_strength, false, hybrid_ctx.as_ref());
                    #[cfg(not(feature = "experimental-hybrid-trellis"))]
                    let cr_quant_coeffs = quant::quantize_block_with_zero_bias_simd(
                        &cr_dct, &cr_quant.values, &cr_zero_bias, aq_strength);

                    cr_blocks.push(natural_to_zigzag(&cr_quant_coeffs));
                }
            }
        }

        Ok((y_blocks, cb_blocks, cr_blocks))
    }

    /// Quantizes all blocks with subsampling support.
    ///
    /// Unlike `quantize_all_blocks`, this version handles different dimensions
    /// for Y and chroma planes (needed for 4:2:0, 4:2:2, 4:4:0 subsampling).
    #[allow(clippy::too_many_arguments)]
    fn quantize_all_blocks_subsampled(
        &self,
        y_plane: &[f32],
        y_width: usize,
        y_height: usize,
        cb_plane: &[f32],
        cr_plane: &[f32],
        c_width: usize,
        c_height: usize,
        y_quant: &QuantTable,
        cb_quant: &QuantTable,
        cr_quant: &QuantTable,
    ) -> Result<(
        Vec<[i16; DCT_BLOCK_SIZE]>,
        Vec<[i16; DCT_BLOCK_SIZE]>,
        Vec<[i16; DCT_BLOCK_SIZE]>,
    )> {
        let y_blocks_h = (y_width + 7) / 8;
        let y_blocks_v = (y_height + 7) / 8;
        let c_blocks_h = (c_width + 7) / 8;
        let c_blocks_v = (c_height + 7) / 8;
        let is_color = self.config.pixel_format != PixelFormat::Gray;

        // Zero-bias parameters for each component
        let effective_distance = quant::quant_vals_to_distance(y_quant, cb_quant, cr_quant);
        let y_zero_bias = ZeroBiasParams::for_ycbcr(effective_distance, 0);
        let cb_zero_bias = ZeroBiasParams::for_ycbcr(effective_distance, 1);
        let cr_zero_bias = ZeroBiasParams::for_ycbcr(effective_distance, 2);

        // Compute per-block adaptive quantization strength from Y plane
        let y_quant_01 = y_quant.values[1];
        #[cfg(feature = "experimental-hybrid-trellis")]
        let aq_map = hybrid::get_aq_map_or_compute(
            &self.config, y_plane, y_width, y_height, y_quant_01);
        #[cfg(not(feature = "experimental-hybrid-trellis"))]
        let aq_map = compute_aq_strength_map(y_plane, y_width, y_height, y_quant_01);

        // Create hybrid quantization context if enabled
        #[cfg(feature = "experimental-hybrid-trellis")]
        let hybrid_ctx = hybrid::create_hybrid_ctx(&self.config);

        let mut y_blocks = Vec::with_capacity(y_blocks_h * y_blocks_v);
        let mut cb_blocks = Vec::with_capacity(if is_color { c_blocks_h * c_blocks_v } else { 0 });
        let mut cr_blocks = Vec::with_capacity(if is_color { c_blocks_h * c_blocks_v } else { 0 });

        // Quantize Y blocks
        for by in 0..y_blocks_v {
            for bx in 0..y_blocks_h {
                let aq_strength = aq_map.get(bx, by);
                let y_block =
                    crate::encode_simd::extract_block_simd(y_plane, y_width, y_height, bx, by);
                let y_dct = forward_dct_8x8(&y_block);

                #[cfg(feature = "experimental-hybrid-trellis")]
                let y_quant_coeffs = hybrid::quantize_block_dispatch(
                    &y_dct, &y_quant.values, &y_zero_bias, aq_strength, true, hybrid_ctx.as_ref());
                #[cfg(not(feature = "experimental-hybrid-trellis"))]
                let y_quant_coeffs = quant::quantize_block_with_zero_bias_simd(
                    &y_dct, &y_quant.values, &y_zero_bias, aq_strength);

                y_blocks.push(natural_to_zigzag(&y_quant_coeffs));
            }
        }

        // Quantize chroma blocks (from possibly downsampled planes)
        if is_color {
            for by in 0..c_blocks_v {
                for bx in 0..c_blocks_h {
                    // For chroma, use average AQ strength from corresponding Y region
                    // For 4:2:0, each chroma block corresponds to 2x2 Y blocks
                    let y_bx = (bx * y_blocks_h) / c_blocks_h;
                    let y_by = (by * y_blocks_v) / c_blocks_v;
                    let aq_strength =
                        aq_map.get(y_bx.min(y_blocks_h - 1), y_by.min(y_blocks_v - 1));

                    let cb_block =
                        crate::encode_simd::extract_block_simd(cb_plane, c_width, c_height, bx, by);
                    let cb_dct = forward_dct_8x8(&cb_block);

                    #[cfg(feature = "experimental-hybrid-trellis")]
                    let cb_quant_coeffs = hybrid::quantize_block_dispatch(
                        &cb_dct, &cb_quant.values, &cb_zero_bias, aq_strength, false, hybrid_ctx.as_ref());
                    #[cfg(not(feature = "experimental-hybrid-trellis"))]
                    let cb_quant_coeffs = quant::quantize_block_with_zero_bias_simd(
                        &cb_dct, &cb_quant.values, &cb_zero_bias, aq_strength);

                    cb_blocks.push(natural_to_zigzag(&cb_quant_coeffs));

                    let cr_block =
                        crate::encode_simd::extract_block_simd(cr_plane, c_width, c_height, bx, by);
                    let cr_dct = forward_dct_8x8(&cr_block);

                    #[cfg(feature = "experimental-hybrid-trellis")]
                    let cr_quant_coeffs = hybrid::quantize_block_dispatch(
                        &cr_dct, &cr_quant.values, &cr_zero_bias, aq_strength, false, hybrid_ctx.as_ref());
                    #[cfg(not(feature = "experimental-hybrid-trellis"))]
                    let cr_quant_coeffs = quant::quantize_block_with_zero_bias_simd(
                        &cr_dct, &cr_quant.values, &cr_zero_bias, aq_strength);

                    cr_blocks.push(natural_to_zigzag(&cr_quant_coeffs));
                }
            }
        }

        Ok((y_blocks, cb_blocks, cr_blocks))
    }

    /// Builds optimized Huffman tables from quantized blocks.
    ///
    /// Collects symbol frequencies from all blocks and generates optimal
    /// Huffman tables with their DHT marker representations.
    ///
    /// For subsampled modes, this iterates blocks in MCU order to correctly
    /// account for padding blocks.
    fn build_optimized_tables(
        &self,
        y_blocks: &[[i16; DCT_BLOCK_SIZE]],
        cb_blocks: &[[i16; DCT_BLOCK_SIZE]],
        cr_blocks: &[[i16; DCT_BLOCK_SIZE]],
        is_color: bool,
    ) -> Result<OptimizedHuffmanTables> {
        let mut dc_luma_freq = FrequencyCounter::new();
        let mut dc_chroma_freq = FrequencyCounter::new();
        let mut ac_luma_freq = FrequencyCounter::new();
        let mut ac_chroma_freq = FrequencyCounter::new();

        let width = self.config.width as usize;
        let height = self.config.height as usize;
        let (h_samp, v_samp) = match self.config.subsampling {
            Subsampling::S444 => (1, 1),
            Subsampling::S422 => (2, 1),
            Subsampling::S420 => (2, 2),
            Subsampling::S440 => (1, 2),
        };

        // Zero block for padding
        const ZERO_BLOCK: [i16; DCT_BLOCK_SIZE] = [0i16; DCT_BLOCK_SIZE];

        if h_samp == 1 && v_samp == 1 {
            // 4:4:4 mode - simple iteration, no padding needed
            let mut prev_y_dc: i16 = 0;
            let mut prev_cb_dc: i16 = 0;
            let mut prev_cr_dc: i16 = 0;

            for (i, y_block) in y_blocks.iter().enumerate() {
                Self::collect_block_frequencies(
                    y_block,
                    prev_y_dc,
                    &mut dc_luma_freq,
                    &mut ac_luma_freq,
                );
                prev_y_dc = y_block[0];

                if is_color {
                    Self::collect_block_frequencies(
                        &cb_blocks[i],
                        prev_cb_dc,
                        &mut dc_chroma_freq,
                        &mut ac_chroma_freq,
                    );
                    prev_cb_dc = cb_blocks[i][0];

                    Self::collect_block_frequencies(
                        &cr_blocks[i],
                        prev_cr_dc,
                        &mut dc_chroma_freq,
                        &mut ac_chroma_freq,
                    );
                    prev_cr_dc = cr_blocks[i][0];
                }
            }
        } else {
            // Subsampled mode - iterate in MCU order with padding
            let y_blocks_h = (width + 7) / 8;
            let y_blocks_v = (height + 7) / 8;
            // Use ceiling division for chroma dimensions: (n + d - 1) / d
            let c_width = (width + h_samp - 1) / h_samp;
            let c_height = (height + v_samp - 1) / v_samp;
            let c_blocks_h = (c_width + 7) / 8;
            let c_blocks_v = (c_height + 7) / 8;
            let mcu_h = (y_blocks_h + h_samp - 1) / h_samp;
            let mcu_v = (y_blocks_v + v_samp - 1) / v_samp;

            let mut prev_y_dc: i16 = 0;
            let mut prev_cb_dc: i16 = 0;
            let mut prev_cr_dc: i16 = 0;

            for mcu_y in 0..mcu_v {
                for mcu_x in 0..mcu_h {
                    // Y blocks in this MCU
                    for dy in 0..v_samp {
                        for dx in 0..h_samp {
                            let y_bx = mcu_x * h_samp + dx;
                            let y_by = mcu_y * v_samp + dy;
                            let block = if y_bx < y_blocks_h && y_by < y_blocks_v {
                                let y_idx = y_by * y_blocks_h + y_bx;
                                &y_blocks[y_idx]
                            } else {
                                &ZERO_BLOCK
                            };
                            Self::collect_block_frequencies(
                                block,
                                prev_y_dc,
                                &mut dc_luma_freq,
                                &mut ac_luma_freq,
                            );
                            prev_y_dc = block[0];
                        }
                    }

                    // Chroma blocks
                    if is_color {
                        let (cb_block, cr_block) = if mcu_x < c_blocks_h && mcu_y < c_blocks_v {
                            let c_idx = mcu_y * c_blocks_h + mcu_x;
                            (&cb_blocks[c_idx], &cr_blocks[c_idx])
                        } else {
                            (&ZERO_BLOCK, &ZERO_BLOCK)
                        };

                        Self::collect_block_frequencies(
                            cb_block,
                            prev_cb_dc,
                            &mut dc_chroma_freq,
                            &mut ac_chroma_freq,
                        );
                        prev_cb_dc = cb_block[0];

                        Self::collect_block_frequencies(
                            cr_block,
                            prev_cr_dc,
                            &mut dc_chroma_freq,
                            &mut ac_chroma_freq,
                        );
                        prev_cr_dc = cr_block[0];
                    }
                }
            }
        }

        // Determine which Huffman algorithm to use
        let huffman_method = self
            .config
            .internal_pipeline
            .map(|p| p.huffman_method)
            .unwrap_or(crate::types::HuffmanMethod::JpegliCreateTree);

        // Build optimized tables with DHT data using selected algorithm
        let dc_luma = dc_luma_freq.generate_table_with_method(huffman_method)?;
        let ac_luma = ac_luma_freq.generate_table_with_method(huffman_method)?;

        let (dc_chroma, ac_chroma) = if is_color {
            (
                dc_chroma_freq.generate_table_with_method(huffman_method)?,
                ac_chroma_freq.generate_table_with_method(huffman_method)?,
            )
        } else {
            // Use standard tables for grayscale (won't be used but needed for structure)
            use crate::huffman::{
                STD_AC_CHROMINANCE_BITS, STD_AC_CHROMINANCE_VALUES, STD_DC_CHROMINANCE_BITS,
                STD_DC_CHROMINANCE_VALUES,
            };
            use crate::huffman_opt::OptimizedTable;

            (
                OptimizedTable {
                    table: HuffmanEncodeTable::std_dc_chrominance(),
                    bits: STD_DC_CHROMINANCE_BITS,
                    values: STD_DC_CHROMINANCE_VALUES.to_vec(),
                },
                OptimizedTable {
                    table: HuffmanEncodeTable::std_ac_chrominance(),
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

    /// Encodes blocks using optimized Huffman tables.
    ///
    /// Handles MCU interleaving for subsampled modes (4:2:0, 4:2:2, 4:4:0).
    fn encode_with_tables(
        &self,
        y_blocks: &[[i16; DCT_BLOCK_SIZE]],
        cb_blocks: &[[i16; DCT_BLOCK_SIZE]],
        cr_blocks: &[[i16; DCT_BLOCK_SIZE]],
        is_color: bool,
        tables: &OptimizedHuffmanTables,
    ) -> Result<Vec<u8>> {
        let mut encoder = EntropyEncoder::new();

        encoder.set_dc_table(0, tables.dc_luma.table.clone());
        encoder.set_ac_table(0, tables.ac_luma.table.clone());
        encoder.set_dc_table(1, tables.dc_chroma.table.clone());
        encoder.set_ac_table(1, tables.ac_chroma.table.clone());

        if self.config.restart_interval > 0 {
            encoder.set_restart_interval(self.config.restart_interval);
        }

        let width = self.config.width as usize;
        let height = self.config.height as usize;
        let (h_samp, v_samp) = match self.config.subsampling {
            Subsampling::S444 => (1, 1),
            Subsampling::S422 => (2, 1),
            Subsampling::S420 => (2, 2),
            Subsampling::S440 => (1, 2),
        };

        if h_samp == 1 && v_samp == 1 {
            // 4:4:4 mode - simple 1:1 interleaving
            for (i, y_block) in y_blocks.iter().enumerate() {
                encoder.encode_block(y_block, 0, 0, 0)?;

                if is_color {
                    encoder.encode_block(&cb_blocks[i], 1, 1, 1)?;
                    encoder.encode_block(&cr_blocks[i], 2, 1, 1)?;
                }

                encoder.check_restart();
            }
        } else {
            // Subsampled mode - MCU interleaving
            let y_blocks_h = (width + 7) / 8;
            let y_blocks_v = (height + 7) / 8;
            // Use ceiling division for chroma dimensions: (n + d - 1) / d
            let c_width = (width + h_samp - 1) / h_samp;
            let c_height = (height + v_samp - 1) / v_samp;
            let c_blocks_h = (c_width + 7) / 8;
            let c_blocks_v = (c_height + 7) / 8;

            let mcu_h = (y_blocks_h + h_samp - 1) / h_samp;
            let mcu_v = (y_blocks_v + v_samp - 1) / v_samp;

            // Zero block for padding out-of-bounds MCU positions
            const ZERO_BLOCK: [i16; DCT_BLOCK_SIZE] = [0i16; DCT_BLOCK_SIZE];

            for mcu_y in 0..mcu_v {
                for mcu_x in 0..mcu_h {
                    // Encode Y blocks in this MCU (must encode all 4 even if out of bounds)
                    for dy in 0..v_samp {
                        for dx in 0..h_samp {
                            let y_bx = mcu_x * h_samp + dx;
                            let y_by = mcu_y * v_samp + dy;
                            if y_bx < y_blocks_h && y_by < y_blocks_v {
                                let y_idx = y_by * y_blocks_h + y_bx;
                                encoder.encode_block(&y_blocks[y_idx], 0, 0, 0)?;
                            } else {
                                // Out of bounds - encode zero block (padding)
                                encoder.encode_block(&ZERO_BLOCK, 0, 0, 0)?;
                            }
                        }
                    }

                    // Encode Cb and Cr blocks (always, even if out of bounds)
                    if is_color {
                        if mcu_x < c_blocks_h && mcu_y < c_blocks_v {
                            let c_idx = mcu_y * c_blocks_h + mcu_x;
                            encoder.encode_block(&cb_blocks[c_idx], 1, 1, 1)?;
                            encoder.encode_block(&cr_blocks[c_idx], 2, 1, 1)?;
                        } else {
                            // Out of bounds - encode zero blocks (padding)
                            encoder.encode_block(&ZERO_BLOCK, 1, 1, 1)?;
                            encoder.encode_block(&ZERO_BLOCK, 2, 1, 1)?;
                        }
                    }

                    encoder.check_restart();
                }
            }
        }

        Ok(encoder.finish())
    }

    /// Encodes blocks using standard (fixed) Huffman tables - single pass.
    ///
    /// Handles MCU interleaving for subsampled modes (4:2:0, 4:2:2, 4:4:0).
    fn encode_blocks_standard(
        &self,
        y_blocks: &[[i16; DCT_BLOCK_SIZE]],
        cb_blocks: &[[i16; DCT_BLOCK_SIZE]],
        cr_blocks: &[[i16; DCT_BLOCK_SIZE]],
        is_color: bool,
    ) -> Result<Vec<u8>> {
        let mut encoder = EntropyEncoder::new();

        encoder.set_dc_table(0, HuffmanEncodeTable::std_dc_luminance());
        encoder.set_ac_table(0, HuffmanEncodeTable::std_ac_luminance());
        encoder.set_dc_table(1, HuffmanEncodeTable::std_dc_chrominance());
        encoder.set_ac_table(1, HuffmanEncodeTable::std_ac_chrominance());

        if self.config.restart_interval > 0 {
            encoder.set_restart_interval(self.config.restart_interval);
        }

        let width = self.config.width as usize;
        let height = self.config.height as usize;
        let (h_samp, v_samp) = match self.config.subsampling {
            Subsampling::S444 => (1, 1),
            Subsampling::S422 => (2, 1),
            Subsampling::S420 => (2, 2),
            Subsampling::S440 => (1, 2),
        };

        if h_samp == 1 && v_samp == 1 {
            // 4:4:4 mode - simple 1:1 interleaving
            for (i, y_block) in y_blocks.iter().enumerate() {
                encoder.encode_block(y_block, 0, 0, 0)?;

                if is_color {
                    encoder.encode_block(&cb_blocks[i], 1, 1, 1)?;
                    encoder.encode_block(&cr_blocks[i], 2, 1, 1)?;
                }

                encoder.check_restart();
            }
        } else {
            // Subsampled mode - MCU interleaving
            let y_blocks_h = (width + 7) / 8;
            let y_blocks_v = (height + 7) / 8;
            // Use ceiling division for chroma dimensions: (n + d - 1) / d
            let c_width = (width + h_samp - 1) / h_samp;
            let c_height = (height + v_samp - 1) / v_samp;
            let c_blocks_h = (c_width + 7) / 8;
            let c_blocks_v = (c_height + 7) / 8;

            // MCU dimensions in terms of Y blocks
            let mcu_h = (y_blocks_h + h_samp - 1) / h_samp;
            let mcu_v = (y_blocks_v + v_samp - 1) / v_samp;

            // Zero block for padding out-of-bounds MCU positions
            const ZERO_BLOCK: [i16; DCT_BLOCK_SIZE] = [0i16; DCT_BLOCK_SIZE];

            for mcu_y in 0..mcu_v {
                for mcu_x in 0..mcu_h {
                    // Encode Y blocks in this MCU (must encode all even if out of bounds)
                    for dy in 0..v_samp {
                        for dx in 0..h_samp {
                            let y_bx = mcu_x * h_samp + dx;
                            let y_by = mcu_y * v_samp + dy;
                            if y_bx < y_blocks_h && y_by < y_blocks_v {
                                let y_idx = y_by * y_blocks_h + y_bx;
                                encoder.encode_block(&y_blocks[y_idx], 0, 0, 0)?;
                            } else {
                                // Out of bounds - encode zero block (padding)
                                encoder.encode_block(&ZERO_BLOCK, 0, 0, 0)?;
                            }
                        }
                    }

                    // Encode Cb and Cr blocks (always, even if out of bounds)
                    if is_color {
                        if mcu_x < c_blocks_h && mcu_y < c_blocks_v {
                            let c_idx = mcu_y * c_blocks_h + mcu_x;
                            encoder.encode_block(&cb_blocks[c_idx], 1, 1, 1)?;
                            encoder.encode_block(&cr_blocks[c_idx], 2, 1, 1)?;
                        } else {
                            // Out of bounds - encode zero blocks (padding)
                            encoder.encode_block(&ZERO_BLOCK, 1, 1, 1)?;
                            encoder.encode_block(&ZERO_BLOCK, 2, 1, 1)?;
                        }
                    }

                    encoder.check_restart();
                }
            }
        }

        Ok(encoder.finish())
    }

    /// Reorders blocks from MCU order to raster order for XYB progressive encoding.
    ///
    /// For non-interleaved progressive scans, the JPEG decoder expects blocks
    /// in raster order (row by row), not MCU order.
    ///
    /// XYB quantization produces blocks in MCU order:
    /// - MCU 0: (0,0), (1,0), (0,1), (1,1) at indices 0,1,2,3
    /// - MCU 1: (2,0), (3,0), (2,1), (3,1) at indices 4,5,6,7
    ///
    /// But progressive scans need raster order:
    /// - Row 0: (0,0), (1,0), (2,0), (3,0), ... at indices 0,1,2,3,...
    /// - Row 1: (0,1), (1,1), (2,1), (3,1), ... at indices 8,9,10,11,...
    fn reorder_mcu_to_raster(
        mcu_blocks: &[[i16; DCT_BLOCK_SIZE]],
        blocks_x: usize,
        blocks_y: usize,
    ) -> Vec<[i16; DCT_BLOCK_SIZE]> {
        let total_blocks = blocks_x * blocks_y;
        let mut raster = vec![[0i16; DCT_BLOCK_SIZE]; total_blocks];

        let mcu_cols = (blocks_x + 1) / 2;

        // Iterate through MCU-ordered blocks and place in raster order
        for (mcu_idx, chunk) in mcu_blocks.chunks(4).enumerate() {
            let mcu_x = mcu_idx % mcu_cols;
            let mcu_y = mcu_idx / mcu_cols;

            // Within each MCU, blocks are in order: (0,0), (1,0), (0,1), (1,1)
            // which corresponds to positions:
            // [0]: (mcu_x*2 + 0, mcu_y*2 + 0) = top-left
            // [1]: (mcu_x*2 + 1, mcu_y*2 + 0) = top-right
            // [2]: (mcu_x*2 + 0, mcu_y*2 + 1) = bottom-left
            // [3]: (mcu_x*2 + 1, mcu_y*2 + 1) = bottom-right
            for (i, block) in chunk.iter().enumerate() {
                let dx = i % 2;
                let dy = i / 2;
                let bx = mcu_x * 2 + dx;
                let by = mcu_y * 2 + dy;

                if bx < blocks_x && by < blocks_y {
                    let raster_idx = by * blocks_x + bx;
                    raster[raster_idx] = *block;
                }
            }
        }

        raster
    }

    /// Collects symbol frequencies from a block for Huffman optimization.
    fn collect_block_frequencies(
        coeffs: &[i16; DCT_BLOCK_SIZE],
        prev_dc: i16,
        dc_freq: &mut FrequencyCounter,
        ac_freq: &mut FrequencyCounter,
    ) {
        // DC coefficient - limit category to 11 for 8-bit JPEG compatibility
        let dc_diff = coeffs[0] - prev_dc;
        let dc_category = entropy::category(dc_diff).min(11);
        dc_freq.count(dc_category);

        // AC coefficients
        let mut run = 0u8;
        for i in 1..DCT_BLOCK_SIZE {
            let ac = coeffs[i];

            if ac == 0 {
                run += 1;
            } else {
                // Encode runs of 16 zeros (ZRL)
                while run >= 16 {
                    ac_freq.count(0xF0);
                    run -= 16;
                }

                // Encode run/size symbol
                let ac_category = entropy::category(ac);
                let symbol = (run << 4) | ac_category;
                ac_freq.count(symbol);
                run = 0;
            }
        }

        // EOB if trailing zeros
        if run > 0 {
            ac_freq.count(0x00);
        }
    }

    /// Quantizes all XYB blocks for Huffman optimization.
    ///
    /// Returns quantized blocks for X, Y, and B components.
    /// B component is already downsampled (half resolution).
    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)] // Reserved for future XYB encoding improvements
    fn quantize_all_blocks_xyb(
        &self,
        x_plane: &[f32],
        y_plane: &[f32],
        b_plane: &[f32], // Already downsampled
        width: usize,
        height: usize,
        b_width: usize,
        b_height: usize,
        x_quant: &QuantTable,
        y_quant: &QuantTable,
        b_quant: &QuantTable,
    ) -> (
        Vec<[i16; DCT_BLOCK_SIZE]>,
        Vec<[i16; DCT_BLOCK_SIZE]>,
        Vec<[i16; DCT_BLOCK_SIZE]>,
    ) {
        // MCU size for 2×2, 2×2, 1×1 sampling: 16×16 pixels
        let mcu_cols = (width + 15) / 16;
        let mcu_rows = (height + 15) / 16;
        let num_xy_blocks = mcu_cols * mcu_rows * 4; // 4 blocks per MCU for X and Y
        let num_b_blocks = mcu_cols * mcu_rows; // 1 block per MCU for B

        let mut x_blocks = Vec::with_capacity(num_xy_blocks);
        let mut y_blocks = Vec::with_capacity(num_xy_blocks);
        let mut b_blocks = Vec::with_capacity(num_b_blocks);

        for mcu_y in 0..mcu_rows {
            for mcu_x in 0..mcu_cols {
                // Process 4 X blocks (2×2 arrangement within 16×16 MCU)
                for block_y in 0..2 {
                    for block_x in 0..2 {
                        let bx = mcu_x * 2 + block_x;
                        let by = mcu_y * 2 + block_y;
                        let x_block = crate::encode_simd::extract_block_xyb_simd(
                            x_plane, width, height, bx, by,
                        );
                        let x_dct = forward_dct_8x8(&x_block);
                        let x_quant_coeffs = quant::quantize_block(&x_dct, &x_quant.values);
                        x_blocks.push(natural_to_zigzag(&x_quant_coeffs));
                    }
                }

                // Process 4 Y blocks (2×2 arrangement within 16×16 MCU)
                for block_y in 0..2 {
                    for block_x in 0..2 {
                        let bx = mcu_x * 2 + block_x;
                        let by = mcu_y * 2 + block_y;
                        let y_block = crate::encode_simd::extract_block_xyb_simd(
                            y_plane, width, height, bx, by,
                        );
                        let y_dct = forward_dct_8x8(&y_block);
                        let y_quant_coeffs = quant::quantize_block(&y_dct, &y_quant.values);
                        y_blocks.push(natural_to_zigzag(&y_quant_coeffs));
                    }
                }

                // Process 1 B block (from downsampled plane)
                let b_block = crate::encode_simd::extract_block_xyb_simd(
                    b_plane, b_width, b_height, mcu_x, mcu_y,
                );
                let b_dct = forward_dct_8x8(&b_block);
                let b_quant_coeffs = quant::quantize_block(&b_dct, &b_quant.values);
                b_blocks.push(natural_to_zigzag(&b_quant_coeffs));
            }
        }

        (x_blocks, y_blocks, b_blocks)
    }

    /// Quantizes all XYB blocks with jpegli-style adaptive quantization (no trellis).
    ///
    /// This version uses the AQ map for per-block modulation with zero-bias,
    /// matching jpegli's default AQ behavior without hybrid trellis.
    ///
    /// For XYB mode:
    /// - X and Y use luma tables (both are full-resolution "luma-like" channels)
    /// - B uses chroma tables (downsampled blue channel)
    #[allow(clippy::too_many_arguments)]
    fn quantize_all_blocks_xyb_with_aq_simple(
        &self,
        x_plane: &[f32],
        y_plane: &[f32],
        b_plane: &[f32], // Already downsampled
        width: usize,
        height: usize,
        b_width: usize,
        b_height: usize,
        x_quant: &QuantTable,
        y_quant: &QuantTable,
        b_quant: &QuantTable,
        aq_map: &crate::adaptive_quant::AQStrengthMap,
        x_zero_bias: &ZeroBiasParams,
        y_zero_bias: &ZeroBiasParams,
        b_zero_bias: &ZeroBiasParams,
    ) -> (
        Vec<[i16; DCT_BLOCK_SIZE]>,
        Vec<[i16; DCT_BLOCK_SIZE]>,
        Vec<[i16; DCT_BLOCK_SIZE]>,
    ) {
        // MCU size for 2×2, 2×2, 1×1 sampling: 16×16 pixels
        let mcu_cols = (width + 15) / 16;
        let mcu_rows = (height + 15) / 16;
        let num_xy_blocks = mcu_cols * mcu_rows * 4; // 4 blocks per MCU for X and Y
        let num_b_blocks = mcu_cols * mcu_rows; // 1 block per MCU for B

        let mut x_blocks = Vec::with_capacity(num_xy_blocks);
        let mut y_blocks = Vec::with_capacity(num_xy_blocks);
        let mut b_blocks = Vec::with_capacity(num_b_blocks);

        for mcu_y in 0..mcu_rows {
            for mcu_x in 0..mcu_cols {
                // Process 4 X blocks (2×2 arrangement within 16×16 MCU)
                for block_y in 0..2 {
                    for block_x in 0..2 {
                        let bx = mcu_x * 2 + block_x;
                        let by = mcu_y * 2 + block_y;
                        let aq_strength = aq_map.get(bx, by);

                        let x_block = crate::encode_simd::extract_block_xyb_simd(
                            x_plane, width, height, bx, by,
                        );
                        let x_dct = forward_dct_8x8(&x_block);
                        let x_quant_coeffs = quant::quantize_block_with_zero_bias_simd(
                            &x_dct,
                            &x_quant.values,
                            x_zero_bias,
                            aq_strength,
                        );
                        x_blocks.push(natural_to_zigzag(&x_quant_coeffs));
                    }
                }

                // Process 4 Y blocks (2×2 arrangement within 16×16 MCU)
                for block_y in 0..2 {
                    for block_x in 0..2 {
                        let bx = mcu_x * 2 + block_x;
                        let by = mcu_y * 2 + block_y;
                        let aq_strength = aq_map.get(bx, by);

                        let y_block = crate::encode_simd::extract_block_xyb_simd(
                            y_plane, width, height, bx, by,
                        );
                        let y_dct = forward_dct_8x8(&y_block);
                        let y_quant_coeffs = quant::quantize_block_with_zero_bias_simd(
                            &y_dct,
                            &y_quant.values,
                            y_zero_bias,
                            aq_strength,
                        );
                        y_blocks.push(natural_to_zigzag(&y_quant_coeffs));
                    }
                }

                // Process 1 B block (from downsampled plane)
                // For B channel: Average AQ from 4 parent full-res blocks
                let b_aq_strength = {
                    let mut sum = 0.0f32;
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let bx = mcu_x * 2 + dx;
                            let by = mcu_y * 2 + dy;
                            sum += aq_map.get(bx, by);
                        }
                    }
                    sum / 4.0
                };

                let b_block = crate::encode_simd::extract_block_xyb_simd(
                    b_plane, b_width, b_height, mcu_x, mcu_y,
                );
                let b_dct = forward_dct_8x8(&b_block);
                let b_quant_coeffs = quant::quantize_block_with_zero_bias_simd(
                    &b_dct,
                    &b_quant.values,
                    b_zero_bias,
                    b_aq_strength,
                );
                b_blocks.push(natural_to_zigzag(&b_quant_coeffs));
            }
        }

        (x_blocks, y_blocks, b_blocks)
    }

    /// Builds optimized Huffman tables for XYB mode.
    ///
    /// XYB uses a single shared table for all components (luminance tables).
    /// Returns the optimized DC and AC tables.
    fn build_optimized_tables_xyb(
        &self,
        x_blocks: &[[i16; DCT_BLOCK_SIZE]],
        y_blocks: &[[i16; DCT_BLOCK_SIZE]],
        b_blocks: &[[i16; DCT_BLOCK_SIZE]],
    ) -> Result<(
        crate::huffman_opt::OptimizedTable,
        crate::huffman_opt::OptimizedTable,
    )> {
        let mut dc_freq = FrequencyCounter::new();
        let mut ac_freq = FrequencyCounter::new();

        // Collect frequencies from all components
        // Note: XYB MCU order is 4 X blocks, 4 Y blocks, 1 B block per MCU
        // But since all share the same table, we just iterate through them

        // In XYB mode, we have interleaved blocks per MCU:
        // [X0, X1, X2, X3, Y0, Y1, Y2, Y3, B0] per MCU
        // DC prediction carries across MCUs for each component (standard JPEG behavior)

        let mcu_count = b_blocks.len();

        // Each component maintains its own DC prediction across all MCUs
        let mut prev_dc_x: i16 = 0;
        let mut prev_dc_y: i16 = 0;
        let mut prev_dc_b: i16 = 0;

        for mcu_idx in 0..mcu_count {
            // X blocks (4 per MCU)
            let x_start = mcu_idx * 4;
            for i in 0..4 {
                let block = &x_blocks[x_start + i];
                Self::collect_block_frequencies(block, prev_dc_x, &mut dc_freq, &mut ac_freq);
                prev_dc_x = block[0];
            }

            // Y blocks (4 per MCU)
            let y_start = mcu_idx * 4;
            for i in 0..4 {
                let block = &y_blocks[y_start + i];
                Self::collect_block_frequencies(block, prev_dc_y, &mut dc_freq, &mut ac_freq);
                prev_dc_y = block[0];
            }

            // B block (1 per MCU)
            Self::collect_block_frequencies(
                &b_blocks[mcu_idx],
                prev_dc_b,
                &mut dc_freq,
                &mut ac_freq,
            );
            prev_dc_b = b_blocks[mcu_idx][0];
        }

        // Determine which Huffman algorithm to use
        let huffman_method = self
            .config
            .internal_pipeline
            .map(|p| p.huffman_method)
            .unwrap_or(crate::types::HuffmanMethod::JpegliCreateTree);

        // Generate optimized tables using selected algorithm
        let dc_table = dc_freq.generate_table_with_method(huffman_method)?;
        let ac_table = ac_freq.generate_table_with_method(huffman_method)?;

        Ok((dc_table, ac_table))
    }

    /// Encodes XYB blocks using optimized Huffman tables.
    #[allow(clippy::too_many_arguments)]
    fn encode_with_tables_xyb(
        &self,
        x_blocks: &[[i16; DCT_BLOCK_SIZE]],
        y_blocks: &[[i16; DCT_BLOCK_SIZE]],
        b_blocks: &[[i16; DCT_BLOCK_SIZE]],
        dc_table: &crate::huffman_opt::OptimizedTable,
        ac_table: &crate::huffman_opt::OptimizedTable,
    ) -> Result<Vec<u8>> {
        let mut encoder = EntropyEncoder::new();

        // Use the same optimized table for all components
        encoder.set_dc_table(0, dc_table.table.clone());
        encoder.set_ac_table(0, ac_table.table.clone());

        if self.config.restart_interval > 0 {
            encoder.set_restart_interval(self.config.restart_interval);
        }

        let mcu_count = b_blocks.len();
        for mcu_idx in 0..mcu_count {
            // X blocks (4 per MCU)
            let x_start = mcu_idx * 4;
            for i in 0..4 {
                encoder.encode_block(&x_blocks[x_start + i], 0, 0, 0)?;
            }

            // Y blocks (4 per MCU)
            let y_start = mcu_idx * 4;
            for i in 0..4 {
                encoder.encode_block(&y_blocks[y_start + i], 1, 0, 0)?;
            }

            // B block (1 per MCU)
            encoder.encode_block(&b_blocks[mcu_idx], 2, 0, 0)?;

            encoder.check_restart();
        }

        Ok(encoder.finish())
    }


    /// Encodes scan data for XYB mode with float planes.
    ///
    /// Uses scaled XYB values (in [0, 1] range), converts to [0, 255],
    /// then level shifts by subtracting 128 before DCT.
    #[allow(clippy::too_many_arguments)]
    fn encode_scan_xyb_float(
        &self,
        x_plane: &[f32],
        y_plane: &[f32],
        b_plane: &[f32], // Already downsampled
        width: usize,
        height: usize,
        b_width: usize,
        b_height: usize,
        x_quant: &QuantTable,
        y_quant: &QuantTable,
        b_quant: &QuantTable,
    ) -> Result<Vec<u8>> {
        let mut encoder = EntropyEncoder::new();

        // Set up Huffman tables - use luminance tables for all components in XYB mode
        encoder.set_dc_table(0, HuffmanEncodeTable::std_dc_luminance());
        encoder.set_ac_table(0, HuffmanEncodeTable::std_ac_luminance());

        if self.config.restart_interval > 0 {
            encoder.set_restart_interval(self.config.restart_interval);
        }

        // MCU size for 2×2, 2×2, 1×1 sampling: 16×16 pixels
        // Each MCU contains: 4 X blocks + 4 Y blocks + 1 B block = 9 blocks
        let mcu_cols = (width + 15) / 16;
        let mcu_rows = (height + 15) / 16;

        for mcu_y in 0..mcu_rows {
            for mcu_x in 0..mcu_cols {
                // Process 4 X blocks (2×2 arrangement within 16×16 MCU)
                for block_y in 0..2 {
                    for block_x in 0..2 {
                        let bx = mcu_x * 2 + block_x;
                        let by = mcu_y * 2 + block_y;
                        let x_block = crate::encode_simd::extract_block_xyb_simd(
                            x_plane, width, height, bx, by,
                        );
                        let x_dct = forward_dct_8x8(&x_block);
                        let x_quant_coeffs = quant::quantize_block(&x_dct, &x_quant.values);
                        let x_zigzag = natural_to_zigzag(&x_quant_coeffs);
                        encoder.encode_block(&x_zigzag, 0, 0, 0)?;
                    }
                }

                // Process 4 Y blocks (2×2 arrangement within 16×16 MCU)
                for block_y in 0..2 {
                    for block_x in 0..2 {
                        let bx = mcu_x * 2 + block_x;
                        let by = mcu_y * 2 + block_y;
                        let y_block = crate::encode_simd::extract_block_xyb_simd(
                            y_plane, width, height, bx, by,
                        );
                        let y_dct = forward_dct_8x8(&y_block);
                        let y_quant_coeffs = quant::quantize_block(&y_dct, &y_quant.values);
                        let y_zigzag = natural_to_zigzag(&y_quant_coeffs);
                        encoder.encode_block(&y_zigzag, 1, 0, 0)?;
                    }
                }

                // Process 1 B block (from downsampled plane)
                let b_block = crate::encode_simd::extract_block_xyb_simd(
                    b_plane, b_width, b_height, mcu_x, mcu_y,
                );
                let b_dct = forward_dct_8x8(&b_block);
                let b_quant_coeffs = quant::quantize_block(&b_dct, &b_quant.values);
                let b_zigzag = natural_to_zigzag(&b_quant_coeffs);
                encoder.encode_block(&b_zigzag, 2, 0, 0)?;

                encoder.check_restart();
            }
        }

        Ok(encoder.finish())
    }

    /// Extracts an 8x8 block from a float plane (scaled XYB values).
    ///
    /// Scaled XYB values are in [0, 1] range. This method:
    /// 1. Multiplies by 255 to get to [0, 255] range
    /// 2. Subtracts 128 for level shifting (DCT input is [-128, 127])
    #[allow(dead_code)]
    fn extract_block_f32(
        &self,
        plane: &[f32],
        width: usize,
        height: usize,
        bx: usize,
        by: usize,
    ) -> [f32; DCT_BLOCK_SIZE] {
        let mut block = [0.0f32; DCT_BLOCK_SIZE];

        for y in 0..DCT_SIZE {
            for x in 0..DCT_SIZE {
                let px = (bx * DCT_SIZE + x).min(width - 1);
                let py = (by * DCT_SIZE + y).min(height - 1);
                let idx = py * width + px;
                let val = plane[idx];
                // XYB scaled values are in range approximately [-2.1, 7.3] after our fix
                // to use C++ jpegli's 0-255 linear RGB convention.
                // After ×255: [-536, 1862]. After -128: [-664, 1734].
                // This is correct for XYB mode - the larger range is expected.
                debug_assert!(
                    val >= -3.0 && val <= 10.0,
                    "extract_block_f32: value {} at ({}, {}) outside expected XYB range [-3, 10]",
                    val,
                    px,
                    py
                );
                // Scale from XYB range to DCT input range, then level shift by -128
                block[y * DCT_SIZE + x] = val * 255.0 - 128.0;
            }
        }

        block
    }

    /// Extracts an 8x8 block from a u8 plane with level shift.
    #[allow(dead_code)]
    fn extract_block(
        &self,
        plane: &[u8],
        width: usize,
        height: usize,
        bx: usize,
        by: usize,
    ) -> [f32; DCT_BLOCK_SIZE] {
        let mut block = [0.0f32; DCT_BLOCK_SIZE];

        for y in 0..DCT_SIZE {
            for x in 0..DCT_SIZE {
                let px = (bx * DCT_SIZE + x).min(width - 1);
                let py = (by * DCT_SIZE + y).min(height - 1);
                let idx = py * width + px;
                // Level shift: subtract 128
                block[y * DCT_SIZE + x] = plane[idx] as f32 - 128.0;
            }
        }

        block
    }

    /// Extracts an 8x8 block from a YCbCr f32 plane with level shift.
    /// Input values are in [0, 255] range, output is level-shifted by -128.
    #[allow(dead_code)]
    fn extract_block_ycbcr_f32(
        &self,
        plane: &[f32],
        width: usize,
        height: usize,
        bx: usize,
        by: usize,
    ) -> [f32; DCT_BLOCK_SIZE] {
        let mut block = [0.0f32; DCT_BLOCK_SIZE];

        for y in 0..DCT_SIZE {
            for x in 0..DCT_SIZE {
                let px = (bx * DCT_SIZE + x).min(width - 1);
                let py = (by * DCT_SIZE + y).min(height - 1);
                let idx = py * width + px;
                // Level shift: subtract 128 (values are already in [0, 255])
                block[y * DCT_SIZE + x] = plane[idx] - 128.0;
            }
        }

        block
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

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
