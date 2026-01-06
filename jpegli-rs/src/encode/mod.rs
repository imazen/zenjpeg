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


    /// Encodes progressive JPEG using XYB color space.
    ///
    /// This uses the same progressive scan structure as YCbCr encoding
    /// but with XYB color conversion and appropriate headers (ICC profile, APP14).
    fn encode_progressive_xyb(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Progressive mode requires Huffman optimization (already validated in encode_progressive)
        // But adding explicit check here for safety in case this is called directly
        if !self.config.optimize_huffman {
            return Err(Error::UnsupportedFeature {
                feature: "Progressive mode with fixed Huffman codes (use optimize_huffman=true)",
            });
        }

        // Use optimized Huffman tables if enabled (2-pass encoding)
        if self.config.optimize_huffman {
            return self.encode_progressive_xyb_optimized(data);
        }

        let mut output = Vec::with_capacity(data.len() / 4);
        let width = self.config.width as usize;
        let height = self.config.height as usize;

        // Convert sRGB to scaled XYB
        let (x_plane, y_plane, b_plane) = self.convert_to_scaled_xyb(data)?;

        // Downsample B channel to match C++ XYB behavior (2x2,2x2,1x1 subsampling)
        // Apply input smoothing before downsampling (matches C++ jpegli behavior)
        let b_smooth = self.apply_input_smoothing(&b_plane, width, height)?;
        let b_downsampled = self.downsample_2x2_f32(&b_smooth, width, height)?;
        let b_width = (width + 1) / 2;
        let b_height = (height + 1) / 2;

        // Generate XYB quantization tables
        // Use separate quantization tables for each component (matches C++ jpegli XYB mode)
        let x_quant = self.gen_quant_table(0, true, false);
        let y_quant = self.gen_quant_table(1, true, false);
        let b_quant = self.gen_quant_table(2, true, false);

        // Quantize all blocks for progressive encoding
        // Use X, Y, B as if they were Y, Cb, Cr for the progressive structure
        // Note: B is downsampled, so use b_downsampled instead of b_plane
        let (x_blocks, y_blocks, b_blocks) = self.quantize_all_blocks_xyb(
            &x_plane,
            &y_plane,
            &b_downsampled,
            width,
            height,
            b_width,
            b_height,
            &x_quant,
            &y_quant,
            &b_quant,
        );
        let is_color = self.config.pixel_format != PixelFormat::Gray;

        // Write XYB-specific headers
        self.write_header_xyb(&mut output)?;
        // Write APP14 Adobe marker for RGB (required by some decoders)
        self.write_app14_adobe(&mut output, 0)?; // 0 = RGB (no transform)
                                                 // Write XYB ICC profile
        self.write_icc_profile(&mut output, &XYB_ICC_PROFILE)?;
        // Write quantization tables
        self.write_quant_tables(&mut output, &x_quant, &y_quant, &b_quant)?;
        // Write SOF2 frame header for progressive XYB (with correct component IDs and subsampling)
        self.write_frame_header_xyb_progressive(&mut output)?;

        // Use standard Huffman tables (optimized tables could be added later)
        self.write_huffman_tables(&mut output)?;
        let tables: Option<OptimizedHuffmanTables> = None;

        if self.config.restart_interval > 0 {
            self.write_restart_interval(&mut output)?;
        }

        // Get progressive scan script
        let scans = self.get_progressive_scan_script(is_color);

        // Encode each scan (reusing the YCbCr progressive scan logic)
        for scan in &scans {
            self.write_progressive_scan_header(&mut output, scan, is_color)?;
            let scan_data = self.encode_progressive_scan(
                &x_blocks, &y_blocks, &b_blocks, scan, is_color, &tables,
            )?;
            output.extend_from_slice(&scan_data);
        }

        // Write EOI
        output.push(0xFF);
        output.push(MARKER_EOI);

        Ok(output)
    }

    /// Encodes progressive XYB JPEG with optimized Huffman tables (2-pass).
    ///
    /// This is similar to encode_progressive_optimized() but for XYB color space.
    /// It performs 2-pass encoding: first tokenizes all scans to collect statistics,
    /// then builds optimized Huffman tables and replays tokens.
    fn encode_progressive_xyb_optimized(&self, data: &[u8]) -> Result<Vec<u8>> {
        let mut output = Vec::with_capacity(data.len() / 4);
        let width = self.config.width as usize;
        let height = self.config.height as usize;

        // Convert sRGB to scaled XYB
        let (x_plane, y_plane, b_plane) = self.convert_to_scaled_xyb(data)?;

        // Downsample B channel (2x2,2x2,1x1 subsampling for XYB)
        let b_smooth = self.apply_input_smoothing(&b_plane, width, height)?;
        let b_downsampled = self.downsample_2x2_f32(&b_smooth, width, height)?;
        let b_width = (width + 1) / 2;
        let b_height = (height + 1) / 2;

        // Generate XYB quantization tables
        let x_quant = self.gen_quant_table(0, true, false);
        let y_quant = self.gen_quant_table(1, true, false);
        let b_quant = self.gen_quant_table(2, true, false);

        // Compute AQ map from Y plane (same as baseline XYB, using SIMD scaling)
        let y_plane_scaled = crate::encode_simd::scale_f32_slice_simd(&y_plane, 255.0);
        let y_quant_01 = y_quant.values[1];
        let aq_map = compute_aq_strength_map(&y_plane_scaled, width, height, y_quant_01);

        // Generate zero-bias parameters (same as baseline XYB)
        let effective_distance = quant::quant_vals_to_distance(&x_quant, &y_quant, &b_quant);
        let x_zero_bias = ZeroBiasParams::for_ycbcr(effective_distance, 0); // X uses luma params
        let y_zero_bias = ZeroBiasParams::for_ycbcr(effective_distance, 0); // Y uses luma params
        let b_zero_bias = ZeroBiasParams::for_ycbcr(effective_distance, 1); // B uses chroma params

        // Quantize all blocks WITH adaptive quantization (same as baseline)
        let (x_blocks, y_blocks, b_blocks) = self.quantize_all_blocks_xyb_with_aq_simple(
            &x_plane,
            &y_plane,
            &b_downsampled,
            width,
            height,
            b_width,
            b_height,
            &x_quant,
            &y_quant,
            &b_quant,
            &aq_map,
            &x_zero_bias,
            &y_zero_bias,
            &b_zero_bias,
        );

        // quantize_all_blocks_xyb_with_aq_simple produces blocks in MCU order:
        // - x_blocks[mcu_idx*4..mcu_idx*4+4] = 4 X blocks for mcu_idx
        // - y_blocks[mcu_idx*4..mcu_idx*4+4] = 4 Y blocks for mcu_idx
        // - b_blocks[mcu_idx] = 1 B block for mcu_idx
        //
        // But for non-interleaved progressive scans, the JPEG decoder expects
        // blocks in RASTER order (row by row), not MCU order.
        // So we must reorder X and Y blocks from MCU order to raster order.
        // B blocks don't need reordering since B has 1×1 sampling (1 block per MCU).
        let blocks_x = (width + 7) / 8;
        let blocks_y = (height + 7) / 8;
        let x_blocks_raster = Self::reorder_mcu_to_raster(&x_blocks, blocks_x, blocks_y);
        let y_blocks_raster = Self::reorder_mcu_to_raster(&y_blocks, blocks_x, blocks_y);

        let is_color = self.config.pixel_format != PixelFormat::Gray;
        let num_components = if is_color { 3 } else { 1 };

        // Define progressive scan script
        let scans = self.get_progressive_scan_script(is_color);

        // ========== PASS 1: TOKENIZATION ==========
        let mut token_buffer = ProgressiveTokenBuffer::new(num_components, scans.len());

        for scan in scans.iter() {
            // Calculate context for this scan
            // XYB: all components share same Huffman table (context 0 for DC, 3 for AC)
            // YCbCr: component-specific contexts for luma/chroma split
            let context = if scan.ss == 0 && scan.se == 0 {
                // DC scan: XYB uses context 0 for all, YCbCr uses component index
                if self.config.use_xyb {
                    0 // All XYB components use DC context 0
                } else {
                    scan.components[0] // YCbCr: component-specific DC context
                }
            } else {
                // AC scan: XYB uses context 3 for all, YCbCr uses component-specific
                if self.config.use_xyb {
                    num_components as u8 // All XYB components use AC context 3
                } else {
                    (num_components as u8) + scan.components[0] // YCbCr: offset contexts
                }
            };

            if scan.ss == 0 && scan.se == 0 {
                // DC scan - for XYB, DC scans are non-interleaved (one component per scan)
                // Use raster-ordered blocks for X and Y (decoder expects raster order)
                let blocks: Vec<&[[i16; DCT_BLOCK_SIZE]]> = scan
                    .components
                    .iter()
                    .map(|&c| match c {
                        0 => x_blocks_raster.as_slice(),
                        1 => y_blocks_raster.as_slice(),
                        2 => b_blocks.as_slice(), // B has 1×1 sampling, already in raster order
                        _ => &[][..],
                    })
                    .collect();
                let component_indices: Vec<usize> =
                    scan.components.iter().map(|&c| c as usize).collect();
                token_buffer.tokenize_dc_scan(&blocks, &component_indices, scan.al, scan.ah);
            } else if scan.ah == 0 {
                // AC first scan - use raster-ordered blocks for X and Y
                let blocks: &[[i16; DCT_BLOCK_SIZE]] = match scan.components[0] {
                    0 => &x_blocks_raster,
                    1 => &y_blocks_raster,
                    2 => &b_blocks, // B has 1×1 sampling, already in raster order
                    _ => {
                        return Err(Error::InternalError {
                            reason: "Invalid component",
                        })
                    }
                };
                token_buffer.tokenize_ac_first_scan(blocks, context, scan.ss, scan.se, scan.al);
            } else {
                // AC refinement scan - use raster-ordered blocks for X and Y
                let blocks: &[[i16; DCT_BLOCK_SIZE]] = match scan.components[0] {
                    0 => &x_blocks_raster,
                    1 => &y_blocks_raster,
                    2 => &b_blocks, // B has 1×1 sampling, already in raster order
                    _ => {
                        return Err(Error::InternalError {
                            reason: "Invalid component",
                        })
                    }
                };
                token_buffer.tokenize_ac_refinement_scan(
                    blocks, context, scan.ss, scan.se, scan.ah, scan.al,
                );
            }
        }

        // ========== GENERATE OPTIMIZED TABLES ==========
        // XYB mode uses merged tables (all components share one DC and one AC table)
        let opt_tables = token_buffer.generate_xyb_tables(num_components)?;

        // Create context config for XYB (sequential-style, ac_offset=4)
        let xyb_context_config = ContextConfig::for_sequential(num_components);

        // For XYB, all contexts map to table 0 (DC contexts 0..4 -> 0, AC contexts 4.. -> 1)
        let xyb_context_map: Vec<usize> = (0..xyb_context_config.num_contexts)
            .map(|c| {
                if c < xyb_context_config.ac_offset {
                    0
                } else {
                    1
                }
            })
            .collect();

        // Convert OptimizedHuffmanTables to vec format: [DC table 0, AC table 0]
        let xyb_tables_vec = vec![opt_tables.dc_luma.clone(), opt_tables.ac_luma.clone()];

        // ========== WRITE JPEG STRUCTURE ==========
        self.write_header_xyb(&mut output)?;
        self.write_app14_adobe(&mut output, 0)?; // 0 = RGB (no transform)
        self.write_icc_profile(&mut output, &XYB_ICC_PROFILE)?;
        self.write_quant_tables(&mut output, &x_quant, &y_quant, &b_quant)?;
        self.write_frame_header_xyb_progressive(&mut output)?;

        // Write optimized Huffman tables (XYB uses only table 0)
        self.write_huffman_tables_optimized(&mut output, &opt_tables)?;

        if self.config.restart_interval > 0 {
            self.write_restart_interval(&mut output)?;
        }

        // ========== PASS 2: REPLAY TOKENS ==========
        // XYB uses 1 DC table and 1 AC table (all components share)
        let xyb_num_dc_tables = 1;
        let xyb_ac_slot_ids = vec![0]; // Single AC table uses slot 0
        let xyb_tables_emitted = 2; // All tables emitted upfront (1 DC + 1 AC)
        for (scan_idx, scan) in scans.iter().enumerate() {
            self.write_progressive_scan_header_with_context(
                &mut output,
                scan_idx,
                scan,
                is_color,
                &xyb_context_config,
                &xyb_context_map,
                xyb_num_dc_tables,
            )?;
            let scan_data = self.replay_progressive_scan(
                &token_buffer,
                scan_idx,
                scan,
                is_color,
                &xyb_context_config,
                &xyb_tables_vec,
                xyb_num_dc_tables,
                &xyb_context_map,
                &xyb_ac_slot_ids,
                xyb_tables_emitted,
            )?;
            output.extend_from_slice(&scan_data);
        }

        // Write EOI
        output.push(0xFF);
        output.push(MARKER_EOI);

        Ok(output)
    }

    /// Encodes as progressive JPEG (level 2, matching cjpegli default).
    ///
    /// Progressive level 2 uses the following scan script:
    /// 1. DC first: Ss=0, Se=0, Ah=0, Al=0 (DC only, full precision)
    /// 2. AC 1-2: Ss=1, Se=2, Ah=0, Al=0 (low AC, full precision)
    /// 3. AC 3-63 first: Ss=3, Se=63, Ah=0, Al=2 (high AC, top bits)
    /// 4. AC 3-63 refine: Ss=3, Se=63, Ah=2, Al=1 (bit 1 refinement)
    /// 5. AC 3-63 refine: Ss=3, Se=63, Ah=1, Al=0 (bit 0 refinement)
    fn encode_progressive(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Progressive mode requires Huffman optimization because standard JPEG Huffman tables
        // are designed for baseline/sequential encoding and produce massive bloat (10-100×)
        // when used with progressive AC refinement scans. This matches C++ cjpegli behavior.
        if !self.config.optimize_huffman {
            return Err(Error::UnsupportedFeature {
                feature: "Progressive mode with fixed Huffman codes (use optimize_huffman=true)",
            });
        }

        // XYB progressive mode - route to specialized encoder
        if self.config.use_xyb {
            return self.encode_progressive_xyb(data);
        }

        // Use tokenization-based approach when optimizing Huffman tables
        if self.config.optimize_huffman {
            return self.encode_progressive_optimized(data);
        }

        let mut output = Vec::with_capacity(data.len() / 4);

        // Convert to YCbCr using f32 precision
        let (y_plane, cb_plane, cr_plane) = self.convert_to_ycbcr_f32(data)?;

        // Generate quantization tables (3 separate tables like C++ cjpegli)
        // Progressive mode uses 4:4:4, so is_420 = false
        let y_quant = self.gen_quant_table(0, false, false);
        let cb_quant = self.gen_quant_table(1, false, false);
        let cr_quant = self.gen_quant_table(2, false, false);

        // Quantize all blocks to get full-precision coefficients
        let (y_blocks, cb_blocks, cr_blocks) = self.quantize_all_blocks(
            &y_plane, &cb_plane, &cr_plane, &y_quant, &cb_quant, &cr_quant,
        )?;
        let is_color = self.config.pixel_format != PixelFormat::Gray;

        // Write JPEG structure
        self.write_header(&mut output)?;
        self.write_quant_tables(&mut output, &y_quant, &cb_quant, &cr_quant)?;
        self.write_frame_header(&mut output)?; // Uses SOF2 for progressive

        // For non-optimized progressive, use standard Huffman tables
        self.write_huffman_tables(&mut output)?;
        let tables: Option<OptimizedHuffmanTables> = None;

        if self.config.restart_interval > 0 {
            self.write_restart_interval(&mut output)?;
        }

        // Define progressive scan script (level 2)
        // For 4:4:4 (no subsampling), DC can be interleaved
        let scans = self.get_progressive_scan_script(is_color);

        // Encode each scan
        for scan in &scans {
            // Write SOS header for this scan
            self.write_progressive_scan_header(&mut output, scan, is_color)?;

            // Encode the scan data
            let scan_data = self.encode_progressive_scan(
                &y_blocks, &cb_blocks, &cr_blocks, scan, is_color, &tables,
            )?;
            output.extend_from_slice(&scan_data);
        }

        // Write EOI
        output.push(0xFF);
        output.push(MARKER_EOI);

        Ok(output)
    }

    /// Encodes progressive JPEG with optimized Huffman tables using two-pass tokenization.
    ///
    /// This approach:
    /// 1. Tokenizes all scans first to collect actual symbol usage
    /// 2. Builds histograms from actual tokens (not estimated baseline statistics)
    /// 3. Clusters similar histograms to minimize table overhead
    /// 4. Generates optimal Huffman tables from clustered histograms
    /// 5. Replays tokens with optimized tables
    fn encode_progressive_optimized(&self, data: &[u8]) -> Result<Vec<u8>> {
        let mut output = Vec::with_capacity(data.len() / 4);
        let width = self.config.width as usize;
        let height = self.config.height as usize;

        // Resolve Auto to concrete method based on subsampling
        let chroma_method = self
            .config
            .chroma_conversion
            .resolve(self.config.subsampling);

        // Get YCbCr planes with appropriate chroma handling
        // Sharp/Fast path: yuv crate performs color conversion + downsampling together
        // Intrinsic path: f32 conversion then separate downsampling
        let (y_plane, cb_plane_final, cr_plane_final, c_width, c_height) = if matches!(
            chroma_method,
            ChromaConversion::Sharp | ChromaConversion::Fast
        ) {
            let use_sharp = matches!(chroma_method, ChromaConversion::Sharp);
            match self.config.subsampling {
                Subsampling::S420 => self.convert_yuv_crate_420(data, use_sharp)?,
                Subsampling::S422 => self.convert_yuv_crate_422(data, use_sharp)?,
                // yuv crate doesn't support S440/S444, fall through to Intrinsic
                _ => self.convert_intrinsic_with_subsampling(data)?,
            }
        } else {
            self.convert_intrinsic_with_subsampling(data)?
        };

        // Generate quantization tables (3 separate tables like C++ cjpegli)
        // Apply 4:2:0 quality compensation if using 4:2:0 subsampling
        let is_420 = self.config.subsampling == Subsampling::S420;
        let y_quant = self.gen_quant_table(0, false, is_420);
        let cb_quant = self.gen_quant_table(1, false, is_420);
        let cr_quant = self.gen_quant_table(2, false, is_420);

        // Quantize all blocks with proper subsampling support
        let (y_blocks, cb_blocks, cr_blocks) = self.quantize_all_blocks_subsampled(
            &y_plane,
            width,
            height,
            &cb_plane_final,
            &cr_plane_final,
            c_width,
            c_height,
            &y_quant,
            &cb_quant,
            &cr_quant,
        )?;
        let is_color = self.config.pixel_format != PixelFormat::Gray;
        let num_components = if is_color { 3 } else { 1 };

        // Define progressive scan script
        let scans = self.get_progressive_scan_script(is_color);

        // ========== CREATE CONTEXT CONFIG ==========
        // Per C++ design (encode.cc:340-383): DC contexts are 0..num_components,
        // AC contexts start at 4 with one per component per AC scan.
        let context_config = ContextConfig::for_progressive(
            num_components,
            scans.iter().map(|s| (s.ss, s.se, s.components.len())),
        );

        // ========== PASS 1: TOKENIZATION ==========
        // Tokenize all scans to collect symbol statistics.
        // Use context_config.num_contexts to allocate proper histogram count.
        let mut token_buffer =
            ProgressiveTokenBuffer::new(num_components, context_config.num_contexts);

        for (scan_idx, scan) in scans.iter().enumerate() {
            // Calculate context for this scan using C++ context assignment:
            // - DC: component index (0-3)
            // - AC: context_config.ac_context(scan_idx, comp_in_scan)
            let context = if scan.ss == 0 && scan.se == 0 {
                // DC scan: use component index as context
                context_config.dc_context(scan.components[0] as usize) as u8
            } else {
                // AC scan: use per-scan context from config
                context_config.ac_context(scan_idx, 0) as u8
            };

            if scan.ss == 0 && scan.se == 0 {
                // DC scan
                let blocks: Vec<&[[i16; DCT_BLOCK_SIZE]]> = scan
                    .components
                    .iter()
                    .map(|&c| match c {
                        0 => y_blocks.as_slice(),
                        1 => cb_blocks.as_slice(),
                        2 => cr_blocks.as_slice(),
                        _ => &[][..],
                    })
                    .collect();
                let component_indices: Vec<usize> =
                    scan.components.iter().map(|&c| c as usize).collect();
                token_buffer.tokenize_dc_scan(&blocks, &component_indices, scan.al, scan.ah);
            } else if scan.ah == 0 {
                // AC first scan
                let blocks: &[[i16; DCT_BLOCK_SIZE]] = match scan.components[0] {
                    0 => &y_blocks,
                    1 => &cb_blocks,
                    2 => &cr_blocks,
                    _ => {
                        return Err(Error::InternalError {
                            reason: "Invalid component",
                        })
                    }
                };
                token_buffer.tokenize_ac_first_scan(blocks, context, scan.ss, scan.se, scan.al);
            } else {
                // AC refinement scan
                let blocks: &[[i16; DCT_BLOCK_SIZE]] = match scan.components[0] {
                    0 => &y_blocks,
                    1 => &cb_blocks,
                    2 => &cr_blocks,
                    _ => {
                        return Err(Error::InternalError {
                            reason: "Invalid component",
                        })
                    }
                };
                token_buffer.tokenize_ac_refinement_scan(
                    blocks, context, scan.ss, scan.se, scan.ah, scan.al,
                );
            }
        }

        // ========== GENERATE OPTIMIZED TABLES ==========
        // Use histogram clustering to find optimal table assignments.
        // Per C++ design: progressive mode can have more than 4 AC tables,
        // with slot IDs cycling through 0-3 and redefinition via DHT markers.
        // We allow up to 12 AC clusters (one per AC scan) to enable this.
        let (context_map, num_dc_tables, tables, ac_slot_ids) = token_buffer
            .generate_optimized_tables(
                4,                        // max DC clusters
                12,                       // max AC clusters (allows per-scan specialization)
                context_config.ac_offset, // num_dc_contexts (always 4 per C++ design, but clamped to actual)
                false,                    // force_baseline
            )?;

        // Debug: print context map and scan-to-table assignment
        if std::env::var("DUMP_CONTEXT_MAP").is_ok() {
            eprintln!("=== Context Map Debug ===");
            eprintln!("num_dc_tables: {}", num_dc_tables);
            eprintln!("context_map: {:?}", context_map);
            eprintln!("ac_slot_ids: {:?}", ac_slot_ids);
            for (scan_idx, scan) in scans.iter().enumerate() {
                let scan_type = if scan.ss == 0 && scan.se == 0 {
                    "DC"
                } else if scan.ah == 0 {
                    "AC_first"
                } else {
                    "AC_refine"
                };
                let ac_context = context_config.ac_context(scan_idx, 0);
                let table_idx = if scan.ss == 0 {
                    0
                } else {
                    context_map.get(ac_context).copied().unwrap_or(0)
                };
                let ac_table_idx = table_idx.saturating_sub(num_dc_tables);
                let slot = ac_slot_ids.get(ac_table_idx).copied().unwrap_or(0);
                eprintln!(
                    "Scan {}: {} ss={} se={} ah={} al={} comp={} -> ctx={} table={} slot={}",
                    scan_idx,
                    scan_type,
                    scan.ss,
                    scan.se,
                    scan.ah,
                    scan.al,
                    scan.components[0],
                    ac_context,
                    table_idx,
                    slot
                );
            }
            // Dump histogram symbol counts for AC contexts
            for (i, scan) in scans.iter().enumerate() {
                if scan.ss > 0 {
                    let ac_context = context_config.ac_context(i, 0);
                    if let Some(counter) = token_buffer.counter(ac_context) {
                        let mut syms: Vec<u8> = Vec::new();
                        for s in 0u8..=255 {
                            if counter.get_count(s) > 0 {
                                syms.push(s);
                            }
                        }
                        eprintln!("  Scan {} context {} symbols: {:02x?}", i, ac_context, syms);
                    }
                }
            }
            eprintln!("=========================");
        }

        // ========== WRITE JPEG STRUCTURE ==========
        self.write_header(&mut output)?;
        self.write_quant_tables(&mut output, &y_quant, &cb_quant, &cr_quant)?;
        self.write_frame_header(&mut output)?; // Uses SOF2 for progressive

        // Write initial Huffman tables (all DC + up to 4 AC)
        // Like C++ jpegli, additional AC tables are emitted on-demand before scans that need them.
        let mut next_dht_index = self.write_huffman_tables_progressive_initial(
            &mut output,
            &tables,
            num_dc_tables,
            4, // max_initial_ac: emit up to 4 AC tables initially
        )?;

        if self.config.restart_interval > 0 {
            self.write_restart_interval(&mut output)?;
        }

        // ========== PASS 2: REPLAY TOKENS ==========
        // Encode each scan by replaying tokens with optimized tables
        for (scan_idx, scan) in scans.iter().enumerate() {
            // For AC scans (Ss > 0), check if we need to emit a new Huffman table
            // This matches C++ jpegli behavior: emit tables on-demand for progressive AC scans
            if scan.ss > 0 {
                // Get the AC context for this scan
                let ac_context = context_config.ac_context(scan_idx, 0);
                // Get the table index from context_map
                if let Some(&table_idx) = context_map.get(ac_context) {
                    // If this scan needs the "next" table, emit it now
                    if table_idx == next_dht_index && table_idx < tables.len() {
                        // Get the AC table slot ID from ac_slot_ids
                        let cluster_idx = table_idx.saturating_sub(num_dc_tables);
                        let ac_slot = ac_slot_ids
                            .get(cluster_idx)
                            .copied()
                            .unwrap_or(cluster_idx % 4);
                        self.write_single_ac_table(&mut output, &tables[table_idx], ac_slot)?;
                        next_dht_index += 1;
                    }
                }
            }

            // Write SOS header with context-based table selection
            self.write_progressive_scan_header_with_slot_ids(
                &mut output,
                scan_idx,
                scan,
                is_color,
                &context_config,
                &context_map,
                num_dc_tables,
                &ac_slot_ids,
            )?;

            // Replay tokens for this scan
            let scan_data = self.replay_progressive_scan(
                &token_buffer,
                scan_idx,
                scan,
                is_color,
                &context_config,
                &tables,
                num_dc_tables,
                &context_map,
                &ac_slot_ids,
                next_dht_index, // tables_emitted so far
            )?;
            output.extend_from_slice(&scan_data);
        }

        // Write EOI
        output.push(0xFF);
        output.push(MARKER_EOI);

        Ok(output)
    }

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
