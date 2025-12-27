//! Main encoder implementation
//!
//! Provides the public Encoder API that combines mozjpeg and jpegli approaches
//! for Pareto-optimal JPEG compression.

use crate::adaptive_quant::{compute_aq_field, compute_aq_strength_map, AdaptiveQuantConfig, AQStrengthMap};
use crate::color::{convert_rgb_to_ycbcr, deinterleave_ycbcr};
use crate::consts::DCTSIZE2;
use crate::dct::{forward_dct_8x8, forward_dct_8x8_int, level_shift, descale_for_quant, scale_for_trellis, quantize_block_int};
use crate::entropy::{EntropyEncoder, ProgressiveEncoder, SymbolCounter};
use crate::error::Error;
use crate::huffman::{std_ac_luma, std_dc_luma, DerivedTable, FrequencyCounter, HuffTable};
use crate::progressive::{
    generate_minimal_progressive_scans, generate_simple_progressive_scans,
    generate_standard_progressive_scans, ScanInfo,
};
use crate::types::ScanScript;
use crate::quant::{QuantTableSet, ZeroBiasParams, quantize_block_with_zero_bias, quant_vals_to_distance};
use crate::strategy::{select_strategy, EncodingApproach, SelectedStrategy};
use crate::trellis::{dc_trellis_optimize_indexed, trellis_quantize_block, TrellisConfig};
use crate::types::{EncodingStrategy, OptimizeFor, PixelFormat, Quality, Subsampling};
use crate::Result;

/// Standard AC luma derived table for trellis rate estimation.
/// Created once and reused - mozjpeg uses standard tables for trellis.
fn get_std_ac_derived() -> DerivedTable {
    let htbl = std_ac_luma();
    DerivedTable::from_huff_table(&htbl, false).expect("Standard AC table should be valid")
}

/// Standard DC luma derived table for DC trellis rate estimation.
fn get_std_dc_derived() -> DerivedTable {
    let htbl = std_dc_luma();
    DerivedTable::from_huff_table(&htbl, true).expect("Standard DC table should be valid")
}

/// Run DC trellis optimization row by row (matching C mozjpeg behavior).
///
/// C mozjpeg processes DC trellis one block row at a time, with each row
/// forming an independent chain. This matches the actual JPEG encoding order.
fn run_dc_trellis_by_row(
    raw_blocks: &[[i32; DCTSIZE2]],
    quantized_blocks: &mut [[i16; DCTSIZE2]],
    dc_quantval: u16,
    dc_table: &DerivedTable,
    lambda_log_scale1: f32,
    lambda_log_scale2: f32,
    block_rows: usize,
    block_cols: usize,
) {
    // Process each block row independently
    for row in 0..block_rows {
        // Generate indices for this row (simple row-major order for 4:4:4)
        let start_idx = row * block_cols;
        let indices: Vec<usize> = (0..block_cols).map(|col| start_idx + col).collect();

        // Each row starts with last_dc = 0 (C mozjpeg behavior for trellis pass)
        dc_trellis_optimize_indexed(
            raw_blocks,
            quantized_blocks,
            &indices,
            dc_quantval,
            dc_table,
            0, // last_dc = 0 for each row
            lambda_log_scale1,
            lambda_log_scale2,
        );
    }
}

/// JPEG encoder with configurable quality and encoding strategy
#[derive(Clone)]
pub struct Encoder {
    quality: Quality,
    strategy: EncodingStrategy,
    subsampling: Subsampling,
    progressive: Option<bool>,
    scan_script: ScanScript,
    optimize_huffman: bool,
    optimize_for: OptimizeFor,
}

impl Default for Encoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Encoder {
    /// Create a new encoder with default settings
    pub fn new() -> Self {
        Self {
            quality: Quality::Standard(85),
            strategy: EncodingStrategy::Auto,
            subsampling: Subsampling::S444,
            progressive: None, // Auto-select based on strategy
            scan_script: ScanScript::default(),
            optimize_huffman: true,
            optimize_for: OptimizeFor::default(),
        }
    }

    /// Create encoder optimized for maximum compression
    pub fn max_compression() -> Self {
        Self {
            quality: Quality::Low(75),
            strategy: EncodingStrategy::Mozjpeg,
            subsampling: Subsampling::S420,
            progressive: Some(true),
            scan_script: ScanScript::Minimal, // Simple progressive, lowest overhead
            optimize_huffman: true,
            optimize_for: OptimizeFor::FileSize,
        }
    }

    /// Create encoder optimized for maximum quality
    pub fn max_quality() -> Self {
        Self {
            quality: Quality::High(95),
            strategy: EncodingStrategy::Jpegli,
            subsampling: Subsampling::S444,
            progressive: Some(false),
            scan_script: ScanScript::default(),
            optimize_huffman: true,
            optimize_for: OptimizeFor::Butteraugli,
        }
    }

    /// Create fastest encoder (sacrifices some compression)
    pub fn fastest() -> Self {
        Self {
            quality: Quality::Standard(85),
            strategy: EncodingStrategy::Simple,
            subsampling: Subsampling::S420,
            progressive: Some(false),
            scan_script: ScanScript::Minimal,
            optimize_huffman: false,
            optimize_for: OptimizeFor::default(),
        }
    }

    /// Set the quality level
    pub fn quality(mut self, quality: Quality) -> Self {
        self.quality = quality;
        self
    }

    /// Set the encoding strategy
    pub fn strategy(mut self, strategy: EncodingStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set chroma subsampling mode
    pub fn subsampling(mut self, subsampling: Subsampling) -> Self {
        self.subsampling = subsampling;
        self
    }

    /// Enable or disable progressive encoding
    pub fn progressive(mut self, progressive: bool) -> Self {
        self.progressive = Some(progressive);
        self
    }

    /// Set the progressive scan script.
    ///
    /// This only applies when progressive mode is enabled.
    ///
    /// # Arguments
    /// * `script` - The scan script to use:
    ///   - `Minimal`: DC + full AC scans (fast, good for web)
    ///   - `Simple`: DC + AC bands 1-5 and 6-63 (better progressive display)
    ///   - `Standard`: With successive approximation (best compression)
    ///   - `Custom(scans)`: User-defined scan script
    pub fn scan_script(mut self, script: ScanScript) -> Self {
        self.scan_script = script;
        self
    }

    /// Enable or disable Huffman table optimization
    pub fn optimize_huffman(mut self, optimize: bool) -> Self {
        self.optimize_huffman = optimize;
        self
    }

    /// Set the target quality metric to optimize for.
    ///
    /// Different metrics favor different encoding strategies:
    /// - `Butteraugli`: Favors jpegli (84% win rate) - best for web images
    /// - `Dssim`: Favors mozjpeg at medium bitrates (67% win rate) - best for archival
    /// - `Ssimulacra2`: Balanced between encoders - good general choice
    /// - `FileSize`: Aggressive trellis for smallest files
    ///
    /// # Example
    /// ```
    /// use zenjpeg::{Encoder, OptimizeFor};
    ///
    /// let encoder = Encoder::new()
    ///     .optimize_for(OptimizeFor::Dssim);  // Optimize for structural similarity
    /// ```
    pub fn optimize_for(mut self, metric: OptimizeFor) -> Self {
        self.optimize_for = metric;
        self
    }

    /// Encode RGB image data to JPEG
    pub fn encode_rgb(&self, pixels: &[u8], width: usize, height: usize) -> Result<Vec<u8>> {
        self.validate_dimensions(width, height)?;
        self.validate_pixel_data(pixels, width, height, PixelFormat::Rgb8)?;

        // Handle special quality modes
        match &self.quality {
            Quality::Perceptual(target) => {
                return self.encode_rgb_targeting_quality(pixels, width, height, *target);
            }
            Quality::TargetSize(target_bytes) => {
                return self.encode_rgb_targeting_size(pixels, width, height, *target_bytes);
            }
            _ => {}
        }

        // Select encoding strategy
        let selected = select_strategy(&self.quality, self.strategy);

        // For Jpegli strategy, delegate to the actual jpegli encoder
        // This gives us full perceptual quality (XYB, proper AQ, etc.)
        if selected.approach == EncodingApproach::Jpegli {
            return self.encode_rgb_with_jpegli(pixels, width, height);
        }

        // Convert to YCbCr
        let ycbcr = convert_rgb_to_ycbcr(pixels, width, height);
        let (y_plane, cb_plane, cr_plane) = deinterleave_ycbcr(&ycbcr, width, height);

        // Encode with selected strategy
        self.encode_ycbcr_planes(&y_plane, &cb_plane, &cr_plane, width, height, &selected)
    }

    /// Encode targeting a specific perceptual quality (Butteraugli distance).
    ///
    /// Uses binary search to find the quality value that produces
    /// approximately the target Butteraugli distance.
    ///
    /// # Arguments
    /// * `target_distance` - Target Butteraugli distance (0.5 = excellent, 1.0 = good, 2.0 = acceptable)
    fn encode_rgb_targeting_quality(
        &self,
        pixels: &[u8],
        width: usize,
        height: usize,
        target_distance: f32,
    ) -> Result<Vec<u8>> {
        // Use the inverse quality-to-distance formula to estimate starting quality
        let estimated_q = crate::analysis::distance_to_quality(target_distance);

        // Binary search around the estimate
        let mut low_q = (estimated_q - 15.0).max(10.0);
        let mut high_q = (estimated_q + 15.0).min(100.0);
        let mut best_jpeg = Vec::new();
        let mut best_q = estimated_q;

        // 4 iterations of binary search gives ~1 quality unit precision
        for _ in 0..4 {
            let mid_q = (low_q + high_q) / 2.0;
            let encoder = self.clone().quality(Quality::Standard(mid_q as u8));
            let jpeg = encoder.encode_rgb_internal(pixels, width, height)?;

            // Estimate distance from quality (not actual measurement)
            let est_distance = crate::analysis::quality_to_distance(mid_q);

            if est_distance > target_distance {
                // Distance too high (quality too low) - increase quality
                low_q = mid_q;
            } else {
                // Distance low enough - this is good, try lower quality for smaller file
                high_q = mid_q;
            }

            best_jpeg = jpeg;
            best_q = mid_q;
        }

        // Final encode at best quality
        let encoder = self.clone().quality(Quality::Standard(best_q as u8));
        encoder.encode_rgb_internal(pixels, width, height)
    }

    /// Encode targeting a specific file size in bytes.
    ///
    /// Uses binary search on quality to find the highest quality
    /// that fits within the target size.
    fn encode_rgb_targeting_size(
        &self,
        pixels: &[u8],
        width: usize,
        height: usize,
        target_bytes: usize,
    ) -> Result<Vec<u8>> {
        let mut low_q = 10.0f32;
        let mut high_q = 100.0f32;
        let mut best_jpeg = Vec::new();

        // 7 iterations gives ~1 quality unit precision
        for _ in 0..7 {
            let mid_q = (low_q + high_q) / 2.0;
            let encoder = self.clone().quality(Quality::Standard(mid_q as u8));
            let jpeg = encoder.encode_rgb_internal(pixels, width, height)?;

            if jpeg.len() <= target_bytes {
                // Fits! Try higher quality
                low_q = mid_q;
                best_jpeg = jpeg;
            } else {
                // Too big - try lower quality
                high_q = mid_q;
            }
        }

        if best_jpeg.is_empty() {
            // Even lowest quality doesn't fit, return it anyway
            let encoder = self.clone().quality(Quality::Standard(10));
            best_jpeg = encoder.encode_rgb_internal(pixels, width, height)?;
        }

        Ok(best_jpeg)
    }

    /// Internal encode without special quality mode handling
    fn encode_rgb_internal(&self, pixels: &[u8], width: usize, height: usize) -> Result<Vec<u8>> {
        let selected = select_strategy(&self.quality, self.strategy);

        if selected.approach == EncodingApproach::Jpegli {
            return self.encode_rgb_with_jpegli(pixels, width, height);
        }

        let ycbcr = convert_rgb_to_ycbcr(pixels, width, height);
        let (y_plane, cb_plane, cr_plane) = deinterleave_ycbcr(&ycbcr, width, height);
        self.encode_ycbcr_planes(&y_plane, &cb_plane, &cr_plane, width, height, &selected)
    }

    /// Encode RGB image using our forked jpegli encoder.
    ///
    /// Uses the forked jpegli module for full perceptual quality:
    /// - XYB color space processing
    /// - Butteraugli-based adaptive quantization
    /// - Perceptual coefficient optimization
    ///
    /// This is a fork that we can modify to improve upon jpegli.
    fn encode_rgb_with_jpegli(&self, pixels: &[u8], width: usize, height: usize) -> Result<Vec<u8>> {
        let q = self.quality.value();

        let result = crate::jpegli::Encoder::new()
            .width(width as u32)
            .height(height as u32)
            .pixel_format(crate::jpegli::PixelFormat::Rgb)
            .quality(crate::jpegli::Quality::Traditional(q as f32))
            .encode(pixels);

        match result {
            Ok(data) => Ok(data),
            Err(e) => Err(Error::EncodingFailed {
                stage: "jpegli",
                reason: format!("{:?}", e),
            }),
        }
    }

    /// Encode grayscale image data to JPEG
    pub fn encode_gray(&self, pixels: &[u8], width: usize, height: usize) -> Result<Vec<u8>> {
        self.validate_dimensions(width, height)?;
        self.validate_pixel_data(pixels, width, height, PixelFormat::Gray8)?;

        let selected = select_strategy(&self.quality, self.strategy);

        // For Jpegli strategy, delegate to the actual jpegli encoder
        if selected.approach == EncodingApproach::Jpegli {
            return self.encode_gray_with_jpegli(pixels, width, height);
        }

        // Grayscale uses only Y component
        self.encode_gray_plane(pixels, width, height, &selected)
    }

    /// Encode grayscale image using our forked jpegli encoder.
    fn encode_gray_with_jpegli(&self, pixels: &[u8], width: usize, height: usize) -> Result<Vec<u8>> {
        let q = self.quality.value();

        let result = crate::jpegli::Encoder::new()
            .width(width as u32)
            .height(height as u32)
            .pixel_format(crate::jpegli::PixelFormat::Gray)
            .quality(crate::jpegli::Quality::Traditional(q as f32))
            .encode(pixels);

        match result {
            Ok(data) => Ok(data),
            Err(e) => Err(Error::EncodingFailed {
                stage: "jpegli",
                reason: format!("{:?}", e),
            }),
        }
    }

    /// Validate image dimensions
    fn validate_dimensions(&self, width: usize, height: usize) -> Result<()> {
        if width == 0 || height == 0 {
            return Err(Error::InvalidDimensions {
                width,
                height,
                reason: "dimensions must be non-zero",
            });
        }

        if width > 65535 || height > 65535 {
            return Err(Error::InvalidDimensions {
                width,
                height,
                reason: "dimensions exceed JPEG maximum (65535)",
            });
        }

        Ok(())
    }

    /// Validate pixel data length
    fn validate_pixel_data(
        &self,
        pixels: &[u8],
        width: usize,
        height: usize,
        format: PixelFormat,
    ) -> Result<()> {
        let expected = width * height * format.bytes_per_pixel();
        if pixels.len() != expected {
            return Err(Error::InvalidPixelData {
                expected,
                actual: pixels.len(),
            });
        }
        Ok(())
    }

    /// Encode YCbCr planes to JPEG
    fn encode_ycbcr_planes(
        &self,
        y_plane: &[u8],
        cb_plane: &[u8],
        cr_plane: &[u8],
        width: usize,
        height: usize,
        strategy: &SelectedStrategy,
    ) -> Result<Vec<u8>> {
        let q = self.quality.value() as u8;

        // Select quantization tables based on strategy
        // - Jpegli: perceptually-optimized tables with non-linear frequency scaling
        // - Mozjpeg/other: mozjpeg's ImageMagick variant tables
        let use_jpegli_style = strategy.approach == EncodingApproach::Jpegli;
        let quant_tables = if use_jpegli_style {
            QuantTableSet::jpegli(q)
        } else {
            QuantTableSet::mozjpeg(q)
        };

        // Create standard derived tables for trellis rate estimation
        let ac_derived = get_std_ac_derived();
        let dc_derived = get_std_dc_derived();

        let width_blocks = (width + 7) / 8;
        let height_blocks = (height + 7) / 8;
        let total_blocks = width_blocks * height_blocks;

        // Separate storage for each component
        let mut y_coeffs: Vec<[i16; DCTSIZE2]> = Vec::with_capacity(total_blocks);
        let mut cb_coeffs: Vec<[i16; DCTSIZE2]> = Vec::with_capacity(total_blocks);
        let mut cr_coeffs: Vec<[i16; DCTSIZE2]> = Vec::with_capacity(total_blocks);

        if use_jpegli_style {
            // Jpegli-style encoding: use zero-bias quantization with AQ strength
            // Use effective distance from actual quant table values (accounts for clamping)
            let effective_distance = quant_vals_to_distance(&quant_tables.luma, &quant_tables.chroma);

            // Create zero-bias params for each component using effective distance
            let y_zero_bias = ZeroBiasParams::for_ycbcr(effective_distance, 0);
            let cb_zero_bias = ZeroBiasParams::for_ycbcr(effective_distance, 1);
            let cr_zero_bias = ZeroBiasParams::for_ycbcr(effective_distance, 2);

            // Compute AQ strength map from Y plane
            let aq_config = AdaptiveQuantConfig {
                enabled: true,
                strength: 1.0,
            };
            let aq_map = compute_aq_strength_map(y_plane, width, height, quant_tables.luma.values[1], &aq_config);

            for by in 0..height_blocks {
                for bx in 0..width_blocks {
                    let aq_strength = aq_map.get(bx, by);

                    // Y block: DCT + zero-bias quantization
                    let y_block = extract_block(y_plane, width, height, bx, by);
                    let y_shifted = level_shift(&y_block);
                    let y_dct = forward_dct_8x8(&y_shifted);
                    let y_quant = quantize_block_with_zero_bias(
                        &y_dct,
                        &quant_tables.luma.values,
                        &y_zero_bias,
                        aq_strength,
                    );
                    y_coeffs.push(y_quant);

                    // Cb block
                    let cb_block = extract_block(cb_plane, width, height, bx, by);
                    let cb_shifted = level_shift(&cb_block);
                    let cb_dct = forward_dct_8x8(&cb_shifted);
                    let cb_quant = quantize_block_with_zero_bias(
                        &cb_dct,
                        &quant_tables.chroma.values,
                        &cb_zero_bias,
                        aq_strength,
                    );
                    cb_coeffs.push(cb_quant);

                    // Cr block
                    let cr_block = extract_block(cr_plane, width, height, bx, by);
                    let cr_shifted = level_shift(&cr_block);
                    let cr_dct = forward_dct_8x8(&cr_shifted);
                    let cr_quant = quantize_block_with_zero_bias(
                        &cr_dct,
                        &quant_tables.chroma.values,
                        &cr_zero_bias,
                        aq_strength,
                    );
                    cr_coeffs.push(cr_quant);
                }
            }
        } else {
            // Mozjpeg-style encoding: trellis quantization with AQ multiplier
            let aq_field = if strategy.adaptive_quant.enabled {
                compute_aq_field(y_plane, width, height, &strategy.adaptive_quant)
            } else {
                crate::adaptive_quant::AqField::uniform(width_blocks, height_blocks)
            };

            // Check if DC trellis is enabled
            let dc_trellis_enabled = strategy.trellis.ac_enabled && strategy.trellis.dc_enabled;

            // Storage for raw DCT coefficients (needed for DC trellis)
            let mut y_raw_dct: Option<Vec<[i32; DCTSIZE2]>> = if dc_trellis_enabled {
                Some(vec![[0i32; DCTSIZE2]; total_blocks])
            } else {
                None
            };
            let mut cb_raw_dct: Option<Vec<[i32; DCTSIZE2]>> = if dc_trellis_enabled {
                Some(vec![[0i32; DCTSIZE2]; total_blocks])
            } else {
                None
            };
            let mut cr_raw_dct: Option<Vec<[i32; DCTSIZE2]>> = if dc_trellis_enabled {
                Some(vec![[0i32; DCTSIZE2]; total_blocks])
            } else {
                None
            };

            for by in 0..height_blocks {
                for bx in 0..width_blocks {
                    let block_idx = by * width_blocks + bx;

                    // Y block
                    let y_block = extract_block(y_plane, width, height, bx, by);
                    let (y_quant, y_raw) = self.quantize_block_with_raw(
                        &y_block,
                        &quant_tables.luma.values,
                        aq_field.get(bx, by),
                        strategy,
                        &ac_derived,
                    );
                    y_coeffs.push(y_quant);
                    if let Some(ref mut raw) = y_raw_dct {
                        raw[block_idx] = y_raw;
                    }

                    // Cb block
                    let cb_block = extract_block(cb_plane, width, height, bx, by);
                    let (cb_quant, cb_raw) = self.quantize_block_with_raw(
                        &cb_block,
                        &quant_tables.chroma.values,
                        1.0,
                        strategy,
                        &ac_derived,
                    );
                    cb_coeffs.push(cb_quant);
                    if let Some(ref mut raw) = cb_raw_dct {
                        raw[block_idx] = cb_raw;
                    }

                    // Cr block
                    let cr_block = extract_block(cr_plane, width, height, bx, by);
                    let (cr_quant, cr_raw) = self.quantize_block_with_raw(
                        &cr_block,
                        &quant_tables.chroma.values,
                        1.0,
                        strategy,
                        &ac_derived,
                    );
                    cr_coeffs.push(cr_quant);
                    if let Some(ref mut raw) = cr_raw_dct {
                        raw[block_idx] = cr_raw;
                    }
                }
            }

            // Run DC trellis optimization if enabled
            if dc_trellis_enabled {
                if let Some(ref y_raw) = y_raw_dct {
                    run_dc_trellis_by_row(
                        y_raw,
                        &mut y_coeffs,
                        quant_tables.luma.values[0],
                        &dc_derived,
                        strategy.trellis.lambda_log_scale1,
                        strategy.trellis.lambda_log_scale2,
                        height_blocks,
                        width_blocks,
                    );
                }
                if let Some(ref cb_raw) = cb_raw_dct {
                    run_dc_trellis_by_row(
                        cb_raw,
                        &mut cb_coeffs,
                        quant_tables.chroma.values[0],
                        &dc_derived,
                        strategy.trellis.lambda_log_scale1,
                        strategy.trellis.lambda_log_scale2,
                        height_blocks,
                        width_blocks,
                    );
                }
                if let Some(ref cr_raw) = cr_raw_dct {
                    run_dc_trellis_by_row(
                        cr_raw,
                        &mut cr_coeffs,
                        quant_tables.chroma.values[0],
                        &dc_derived,
                        strategy.trellis.lambda_log_scale1,
                        strategy.trellis.lambda_log_scale2,
                        height_blocks,
                        width_blocks,
                    );
                }
            }
        }

        // Interleave coefficients back into MCU order (Y, Cb, Cr per block)
        let mut all_coeffs: Vec<[i16; 64]> = Vec::with_capacity(total_blocks * 3);
        for i in 0..total_blocks {
            all_coeffs.push(y_coeffs[i]);
            all_coeffs.push(cb_coeffs[i]);
            all_coeffs.push(cr_coeffs[i]);
        }

        // Check if progressive mode is enabled
        // Use explicit setting if provided, otherwise use strategy's recommendation
        let use_progressive = self.progressive.unwrap_or(strategy.progressive);

        // Check if using band-split scan script (requires different symbol frequencies)
        let uses_band_split = matches!(
            self.scan_script,
            ScanScript::Simple | ScanScript::Standard | ScanScript::Custom(_)
        );

        // Get Huffman tables (optimized or standard)
        // Note: Progressive mode with band-split scans (Simple, Standard) requires
        // symbol frequencies that differ from baseline encoding. Use standard tables
        // for band-split progressive. Minimal scan (full AC 1-63) is safe to optimize.
        let can_optimize = self.optimize_huffman && !(use_progressive && uses_band_split);
        let (dc_htbl, ac_htbl) = if can_optimize {
            // Two-pass encoding: count symbol frequencies, then generate optimal tables
            let mut dc_counter = FrequencyCounter::new();
            let mut ac_counter = FrequencyCounter::new();
            let mut symbol_counter = SymbolCounter::new();

            // Count symbols in all blocks
            for (i, coeffs) in all_coeffs.iter().enumerate() {
                let component = i % 3; // Y=0, Cb=1, Cr=2
                symbol_counter.count_block(coeffs, component, &mut dc_counter, &mut ac_counter);
            }

            // Generate optimal tables from frequency counts
            let dc_table = dc_counter.generate_table().unwrap_or_else(|_| std_dc_luma());
            let ac_table = ac_counter.generate_table().unwrap_or_else(|_| std_ac_luma());
            (dc_table, ac_table)
        } else {
            // Use standard tables (for band-split progressive or when optimization is disabled)
            (std_dc_luma(), std_ac_luma())
        };

        // Convert to derived tables for entropy encoding
        let dc_dtbl = DerivedTable::from_huff_table(&dc_htbl, true)
            .expect("Failed to build DC derived table");
        let ac_dtbl = DerivedTable::from_huff_table(&ac_htbl, false)
            .expect("Failed to build AC derived table");

        // Build output buffer with JPEG markers
        let mut output = Vec::with_capacity(width * height);

        // SOI marker
        output.extend_from_slice(&[0xFF, 0xD8]);

        // APP0 (JFIF) marker
        self.write_app0(&mut output);

        // DQT markers
        self.write_dqt(&mut output, &quant_tables.luma);
        self.write_dqt(&mut output, &quant_tables.chroma);

        if use_progressive {
            // SOF2 marker (progressive DCT)
            self.write_sof2(&mut output, width as u16, height as u16);

            // DHT markers
            self.write_dht(&mut output, 0x00, &dc_htbl); // DC table 0
            self.write_dht(&mut output, 0x10, &ac_htbl); // AC table 0

            // Encode progressive scans
            self.encode_progressive_ycbcr(
                &mut output,
                &y_coeffs,
                &cb_coeffs,
                &cr_coeffs,
                &dc_htbl,
                &ac_htbl,
            );
        } else {
            // SOF0 marker (baseline DCT)
            self.write_sof0(&mut output, width as u16, height as u16);

            // DHT markers
            self.write_dht(&mut output, 0x00, &dc_htbl); // DC table 0
            self.write_dht(&mut output, 0x10, &ac_htbl); // AC table 0

            // SOS marker
            self.write_sos(&mut output);

            // Encode blocks with derived tables
            let mut encoder = EntropyEncoder::new(dc_dtbl, ac_dtbl);
            for (i, coeffs) in all_coeffs.iter().enumerate() {
                let component = i % 3;
                encoder.encode_block(coeffs, component);
            }
            output.extend_from_slice(&encoder.finish());
        }

        // EOI marker
        output.extend_from_slice(&[0xFF, 0xD9]);

        Ok(output)
    }

    /// Write DHT marker for a Huffman table
    fn write_dht(&self, output: &mut Vec<u8>, table_info: u8, table: &HuffTable) {
        // Count actual symbols
        let num_symbols: usize = table.bits[1..=16].iter().map(|&x| x as usize).sum();
        let len = 2 + 1 + 16 + num_symbols;
        output.extend_from_slice(&[0xFF, 0xC4]); // DHT marker
        output.extend_from_slice(&(len as u16).to_be_bytes());
        output.push(table_info); // table class and ID

        // Write bits[1..=16] (skip bits[0])
        output.extend_from_slice(&table.bits[1..=16]);
        // Write symbol values
        output.extend_from_slice(&table.huffval[..num_symbols]);
    }

    /// Encode grayscale plane to JPEG
    fn encode_gray_plane(
        &self,
        y_plane: &[u8],
        width: usize,
        height: usize,
        strategy: &SelectedStrategy,
    ) -> Result<Vec<u8>> {
        let q = self.quality.value() as u8;

        // Select quantization table based on strategy
        let use_jpegli_style = strategy.approach == EncodingApproach::Jpegli;
        let quant_table = if use_jpegli_style {
            crate::quant::QuantTable::luma_jpegli(q)
        } else {
            crate::quant::QuantTable::luma_mozjpeg(q)
        };

        // Create standard derived tables for trellis rate estimation
        let ac_derived = get_std_ac_derived();
        let dc_derived = get_std_dc_derived();

        let width_blocks = (width + 7) / 8;
        let height_blocks = (height + 7) / 8;
        let total_blocks = width_blocks * height_blocks;

        // Check if DC trellis is enabled
        let dc_trellis_enabled = strategy.trellis.ac_enabled && strategy.trellis.dc_enabled;

        // Storage for raw DCT coefficients (needed for DC trellis)
        let mut raw_dct: Option<Vec<[i32; DCTSIZE2]>> = if dc_trellis_enabled {
            Some(vec![[0i32; DCTSIZE2]; total_blocks])
        } else {
            None
        };

        // Quantize all blocks first
        let mut all_coeffs: Vec<[i16; 64]> = Vec::with_capacity(total_blocks);
        for by in 0..height_blocks {
            for bx in 0..width_blocks {
                let block_idx = by * width_blocks + bx;
                let block = extract_block(y_plane, width, height, bx, by);
                let (coeffs, raw) =
                    self.quantize_block_with_raw(&block, &quant_table.values, 1.0, strategy, &ac_derived);
                all_coeffs.push(coeffs);
                if let Some(ref mut raw_storage) = raw_dct {
                    raw_storage[block_idx] = raw;
                }
            }
        }

        // Run DC trellis optimization if enabled
        if dc_trellis_enabled {
            if let Some(ref raw) = raw_dct {
                run_dc_trellis_by_row(
                    raw,
                    &mut all_coeffs,
                    quant_table.values[0],
                    &dc_derived,
                    strategy.trellis.lambda_log_scale1,
                    strategy.trellis.lambda_log_scale2,
                    height_blocks,
                    width_blocks,
                );
            }
        }

        // Check if progressive mode is enabled
        // Use explicit setting if provided, otherwise use strategy's recommendation
        let use_progressive = self.progressive.unwrap_or(strategy.progressive);

        // Check if using band-split scan script
        let uses_band_split = matches!(
            self.scan_script,
            ScanScript::Simple | ScanScript::Standard | ScanScript::Custom(_)
        );

        // Get Huffman tables (optimized or standard)
        // Use standard tables for band-split progressive (see RGB encoder comment)
        let can_optimize = self.optimize_huffman && !(use_progressive && uses_band_split);
        let (dc_htbl, ac_htbl) = if can_optimize {
            let mut dc_counter = FrequencyCounter::new();
            let mut ac_counter = FrequencyCounter::new();
            let mut symbol_counter = SymbolCounter::new();

            for coeffs in &all_coeffs {
                symbol_counter.count_block(coeffs, 0, &mut dc_counter, &mut ac_counter);
            }

            let dc_table = dc_counter.generate_table().unwrap_or_else(|_| std_dc_luma());
            let ac_table = ac_counter.generate_table().unwrap_or_else(|_| std_ac_luma());
            (dc_table, ac_table)
        } else {
            (std_dc_luma(), std_ac_luma())
        };

        // Convert to derived tables for entropy encoding
        let dc_dtbl = DerivedTable::from_huff_table(&dc_htbl, true)
            .expect("Failed to build DC derived table");
        let ac_dtbl = DerivedTable::from_huff_table(&ac_htbl, false)
            .expect("Failed to build AC derived table");

        // Build output
        let mut output = Vec::with_capacity(width * height);

        output.extend_from_slice(&[0xFF, 0xD8]); // SOI
        self.write_app0(&mut output);
        self.write_dqt(&mut output, &quant_table);

        if use_progressive {
            self.write_sof2_gray(&mut output, width as u16, height as u16);
            self.write_dht(&mut output, 0x00, &dc_htbl);
            self.write_dht(&mut output, 0x10, &ac_htbl);
            self.encode_progressive_gray(&mut output, &all_coeffs, &dc_htbl, &ac_htbl);
        } else {
            self.write_sof0_gray(&mut output, width as u16, height as u16);
            self.write_dht(&mut output, 0x00, &dc_htbl);
            self.write_dht(&mut output, 0x10, &ac_htbl);
            self.write_sos_gray(&mut output);

            // Encode all blocks
            let mut encoder = EntropyEncoder::new(dc_dtbl, ac_dtbl);
            for coeffs in &all_coeffs {
                encoder.encode_block(coeffs, 0);
            }
            output.extend_from_slice(&encoder.finish());
        }

        output.extend_from_slice(&[0xFF, 0xD9]); // EOI

        Ok(output)
    }

    /// Write APP0 (JFIF) marker
    fn write_app0(&self, output: &mut Vec<u8>) {
        output.extend_from_slice(&[0xFF, 0xE0]); // APP0
        output.extend_from_slice(&[0x00, 0x10]); // Length = 16
        output.extend_from_slice(b"JFIF\0"); // Identifier
        output.extend_from_slice(&[0x01, 0x01]); // Version 1.1
        output.push(0x00); // Units: none
        output.extend_from_slice(&[0x00, 0x01]); // X density
        output.extend_from_slice(&[0x00, 0x01]); // Y density
        output.extend_from_slice(&[0x00, 0x00]); // No thumbnail
    }

    /// Write DQT marker
    fn write_dqt(&self, output: &mut Vec<u8>, table: &crate::quant::QuantTable) {
        output.extend_from_slice(&[0xFF, 0xDB]); // DQT
        output.extend_from_slice(&[0x00, 0x43]); // Length = 67
        output.push(table.slot); // Table ID (8-bit precision)

        // Write values in zigzag order
        for i in 0..64 {
            let idx = crate::consts::ZIGZAG[i];
            output.push(table.values[idx] as u8);
        }
    }

    /// Write SOF0 marker (baseline, 3 components)
    fn write_sof0(&self, output: &mut Vec<u8>, width: u16, height: u16) {
        output.extend_from_slice(&[0xFF, 0xC0]); // SOF0
        output.extend_from_slice(&[0x00, 0x11]); // Length = 17
        output.push(0x08); // 8-bit precision
        output.extend_from_slice(&height.to_be_bytes());
        output.extend_from_slice(&width.to_be_bytes());
        output.push(0x03); // 3 components

        // Y component
        output.push(0x01); // Component ID
        output.push(0x11); // Sampling factors (1x1)
        output.push(0x00); // Quant table 0

        // Cb component
        output.push(0x02);
        output.push(0x11);
        output.push(0x01);

        // Cr component
        output.push(0x03);
        output.push(0x11);
        output.push(0x01);
    }

    /// Write SOF0 marker (baseline, 1 component for grayscale)
    fn write_sof0_gray(&self, output: &mut Vec<u8>, width: u16, height: u16) {
        output.extend_from_slice(&[0xFF, 0xC0]); // SOF0
        output.extend_from_slice(&[0x00, 0x0B]); // Length = 11
        output.push(0x08); // 8-bit precision
        output.extend_from_slice(&height.to_be_bytes());
        output.extend_from_slice(&width.to_be_bytes());
        output.push(0x01); // 1 component

        output.push(0x01); // Component ID
        output.push(0x11); // Sampling factors
        output.push(0x00); // Quant table 0
    }

    /// Write SOS marker (3 components)
    fn write_sos(&self, output: &mut Vec<u8>) {
        output.extend_from_slice(&[0xFF, 0xDA]); // SOS
        output.extend_from_slice(&[0x00, 0x0C]); // Length = 12
        output.push(0x03); // 3 components

        output.push(0x01); // Y: component 1
        output.push(0x00); // Y: DC table 0, AC table 0

        output.push(0x02); // Cb: component 2
        output.push(0x00); // Cb: DC table 0, AC table 0

        output.push(0x03); // Cr: component 3
        output.push(0x00); // Cr: DC table 0, AC table 0

        output.push(0x00); // Ss
        output.push(0x3F); // Se
        output.push(0x00); // Ah/Al
    }

    /// Write SOS marker (1 component for grayscale)
    fn write_sos_gray(&self, output: &mut Vec<u8>) {
        output.extend_from_slice(&[0xFF, 0xDA]);
        output.extend_from_slice(&[0x00, 0x08]); // Length = 8
        output.push(0x01); // 1 component

        output.push(0x01); // Y component
        output.push(0x00); // DC/AC table 0

        output.push(0x00); // Ss
        output.push(0x3F); // Se
        output.push(0x00); // Ah/Al
    }

    /// Write SOF2 marker (progressive DCT, 3 components)
    fn write_sof2(&self, output: &mut Vec<u8>, width: u16, height: u16) {
        output.extend_from_slice(&[0xFF, 0xC2]); // SOF2
        output.extend_from_slice(&[0x00, 0x11]); // Length = 17
        output.push(0x08); // 8-bit precision
        output.extend_from_slice(&height.to_be_bytes());
        output.extend_from_slice(&width.to_be_bytes());
        output.push(0x03); // 3 components

        // Y component
        output.push(0x01); // Component ID
        output.push(0x11); // Sampling factors (1x1)
        output.push(0x00); // Quant table 0

        // Cb component
        output.push(0x02);
        output.push(0x11);
        output.push(0x01);

        // Cr component
        output.push(0x03);
        output.push(0x11);
        output.push(0x01);
    }

    /// Write SOF2 marker (progressive DCT, 1 component for grayscale)
    fn write_sof2_gray(&self, output: &mut Vec<u8>, width: u16, height: u16) {
        output.extend_from_slice(&[0xFF, 0xC2]); // SOF2
        output.extend_from_slice(&[0x00, 0x0B]); // Length = 11
        output.push(0x08); // 8-bit precision
        output.extend_from_slice(&height.to_be_bytes());
        output.extend_from_slice(&width.to_be_bytes());
        output.push(0x01); // 1 component

        output.push(0x01); // Component ID
        output.push(0x11); // Sampling factors
        output.push(0x00); // Quant table 0
    }

    /// Write SOS marker for a progressive scan
    fn write_sos_progressive(&self, output: &mut Vec<u8>, scan: &ScanInfo) {
        let num_comps = scan.comps_in_scan as usize;
        let length = 6 + 2 * num_comps;

        output.extend_from_slice(&[0xFF, 0xDA]); // SOS
        output.extend_from_slice(&(length as u16).to_be_bytes());
        output.push(scan.comps_in_scan);

        // Write component selectors and table assignments
        for i in 0..num_comps {
            let comp_id = scan.component_index[i] + 1; // JPEG component IDs start at 1
            output.push(comp_id);
            // For simplicity, use table 0 for all (DC table 0, AC table 0)
            output.push(0x00);
        }

        output.push(scan.ss); // Spectral selection start
        output.push(scan.se); // Spectral selection end
        output.push((scan.ah << 4) | scan.al); // Successive approximation
    }

    /// Quantize a single 8x8 block (DCT + quantization)
    ///
    /// Uses integer Loeffler DCT for better precision and compression.
    fn quantize_block_impl(
        &self,
        block: &[u8; 64],
        quant: &[u16; 64],
        aq_mult: f32,
        strategy: &SelectedStrategy,
        ac_derived: &DerivedTable,
    ) -> [i16; 64] {
        let (quantized, _) = self.quantize_block_with_raw(block, quant, aq_mult, strategy, ac_derived);
        quantized
    }

    /// Quantize a single 8x8 block and return both quantized and raw DCT coefficients.
    ///
    /// The raw DCT coefficients are needed for DC trellis optimization.
    /// They are stored scaled by 8 (matching mozjpeg's convention).
    fn quantize_block_with_raw(
        &self,
        block: &[u8; 64],
        quant: &[u16; 64],
        aq_mult: f32,
        strategy: &SelectedStrategy,
        ac_derived: &DerivedTable,
    ) -> ([i16; 64], [i32; DCTSIZE2]) {
        // Apply AQ to quantization table
        let effective_quant = if aq_mult != 1.0 {
            crate::adaptive_quant::apply_aq_to_quant(quant, aq_mult)
        } else {
            *quant
        };

        // Level shift and integer DCT (output is scaled by 8)
        let shifted = level_shift(block);
        let mut dct_coeffs = [0i16; 64];
        forward_dct_8x8_int(&shifted, &mut dct_coeffs);

        // Convert to i32 for trellis/quantization and store raw DCT
        let mut dct_i32 = [0i32; DCTSIZE2];
        scale_for_trellis(&dct_coeffs, &mut dct_i32);
        let raw_dct = dct_i32; // Copy for DC trellis

        // Quantize (with optional trellis)
        let quantized = if strategy.trellis.ac_enabled {
            // Trellis path: pass 8x-scaled DCT directly
            // Trellis internally multiplies qtable by 8 to match
            let mut quantized = [0i16; 64];
            trellis_quantize_block(
                &dct_i32,
                &mut quantized,
                &effective_quant,
                ac_derived,
                &strategy.trellis,
            );
            quantized
        } else {
            // Non-trellis path: descale DCT first, then quantize
            let mut dct_descaled = [0i32; 64];
            descale_for_quant(&dct_coeffs, &mut dct_descaled);

            let mut quantized = [0i16; 64];
            quantize_block_int(&dct_descaled, &effective_quant, &mut quantized);
            quantized
        };

        (quantized, raw_dct)
    }
}

impl Encoder {
    /// Encode YCbCr coefficients using progressive mode.
    ///
    /// This writes multiple scans with progressive encoding:
    /// - DC scan for all components (first pass or refinement)
    /// - AC scans for each component (first pass or refinement)
    fn encode_progressive_ycbcr(
        &self,
        output: &mut Vec<u8>,
        y_coeffs: &[[i16; 64]],
        cb_coeffs: &[[i16; 64]],
        cr_coeffs: &[[i16; 64]],
        dc_htbl: &HuffTable,
        ac_htbl: &HuffTable,
    ) {
        let scans = match &self.scan_script {
            ScanScript::Minimal => generate_minimal_progressive_scans(3),
            ScanScript::Simple => generate_simple_progressive_scans(3),
            ScanScript::Standard => generate_standard_progressive_scans(3),
            ScanScript::Custom(custom_scans) => custom_scans.clone(),
        };

        // Convert to derived tables
        let dc_dtbl = DerivedTable::from_huff_table(dc_htbl, true)
            .expect("Failed to build DC derived table");
        let ac_dtbl = DerivedTable::from_huff_table(ac_htbl, false)
            .expect("Failed to build AC derived table");

        for scan in &scans {
            // Write SOS marker for this scan
            self.write_sos_progressive(output, scan);

            // Create encoder for this scan
            let mut encoder = ProgressiveEncoder::new_standard_tables(dc_dtbl.clone(), ac_dtbl.clone());

            // Get coefficients for the components in this scan
            let num_blocks = y_coeffs.len();

            if scan.is_dc_scan() {
                // DC scan
                if scan.is_refinement() {
                    // DC refinement: output single bit per block
                    for block_idx in 0..num_blocks {
                        for comp_idx in 0..scan.comps_in_scan as usize {
                            let coeffs = match scan.component_index[comp_idx] {
                                0 => &y_coeffs[block_idx],
                                1 => &cb_coeffs[block_idx],
                                2 => &cr_coeffs[block_idx],
                                _ => continue,
                            };
                            encoder.encode_dc_refine(coeffs, scan.al);
                        }
                    }
                } else {
                    // DC first pass
                    for block_idx in 0..num_blocks {
                        for comp_idx in 0..scan.comps_in_scan as usize {
                            let coeffs = match scan.component_index[comp_idx] {
                                0 => &y_coeffs[block_idx],
                                1 => &cb_coeffs[block_idx],
                                2 => &cr_coeffs[block_idx],
                                _ => continue,
                            };
                            encoder.encode_dc_first(coeffs, comp_idx, scan.al);
                        }
                    }
                }
            } else {
                // AC scan
                let comp_idx = scan.component_index[0] as usize;
                let coeffs_slice = match comp_idx {
                    0 => y_coeffs,
                    1 => cb_coeffs,
                    2 => cr_coeffs,
                    _ => y_coeffs,
                };

                if scan.is_refinement() {
                    // AC refinement
                    for coeffs in coeffs_slice {
                        encoder.encode_ac_refine(coeffs, scan.ss, scan.se, scan.ah, scan.al);
                    }
                } else {
                    // AC first pass
                    for coeffs in coeffs_slice {
                        encoder.encode_ac_first(coeffs, scan.ss, scan.se, scan.al);
                    }
                }
            }

            // Finish scan and append data
            encoder.finish_scan();
            output.extend_from_slice(&encoder.finish());
        }
    }

    /// Encode grayscale coefficients using progressive mode.
    fn encode_progressive_gray(
        &self,
        output: &mut Vec<u8>,
        coeffs: &[[i16; 64]],
        dc_htbl: &HuffTable,
        ac_htbl: &HuffTable,
    ) {
        let scans = match &self.scan_script {
            ScanScript::Minimal => generate_minimal_progressive_scans(1),
            ScanScript::Simple => generate_simple_progressive_scans(1),
            ScanScript::Standard => generate_standard_progressive_scans(1),
            ScanScript::Custom(custom_scans) => custom_scans.clone(),
        };

        let dc_dtbl = DerivedTable::from_huff_table(dc_htbl, true)
            .expect("Failed to build DC derived table");
        let ac_dtbl = DerivedTable::from_huff_table(ac_htbl, false)
            .expect("Failed to build AC derived table");

        for scan in &scans {
            self.write_sos_progressive(output, scan);

            let mut encoder = ProgressiveEncoder::new_standard_tables(dc_dtbl.clone(), ac_dtbl.clone());

            if scan.is_dc_scan() {
                if scan.is_refinement() {
                    // DC refinement
                    for block_coeffs in coeffs {
                        encoder.encode_dc_refine(block_coeffs, scan.al);
                    }
                } else {
                    // DC first pass
                    for block_coeffs in coeffs {
                        encoder.encode_dc_first(block_coeffs, 0, scan.al);
                    }
                }
            } else {
                if scan.is_refinement() {
                    // AC refinement
                    for block_coeffs in coeffs {
                        encoder.encode_ac_refine(block_coeffs, scan.ss, scan.se, scan.ah, scan.al);
                    }
                } else {
                    // AC first pass
                    for block_coeffs in coeffs {
                        encoder.encode_ac_first(block_coeffs, scan.ss, scan.se, scan.al);
                    }
                }
            }

            encoder.finish_scan();
            output.extend_from_slice(&encoder.finish());
        }
    }
}

/// Extract an 8x8 block from a plane, with edge padding
fn extract_block(plane: &[u8], width: usize, height: usize, bx: usize, by: usize) -> [u8; 64] {
    let mut block = [0u8; 64];
    let start_x = bx * 8;
    let start_y = by * 8;

    for dy in 0..8 {
        let y = (start_y + dy).min(height - 1);
        for dx in 0..8 {
            let x = (start_x + dx).min(width - 1);
            block[dy * 8 + dx] = plane[y * width + x];
        }
    }

    block
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_creation() {
        let encoder = Encoder::new();
        assert!(matches!(encoder.quality, Quality::Standard(85)));
    }

    #[test]
    fn test_encode_small_gray() {
        let encoder = Encoder::new().quality(Quality::Standard(75));
        let pixels = vec![128u8; 16 * 16]; // 16x16 gray image
        let result = encoder.encode_gray(&pixels, 16, 16);
        assert!(result.is_ok());

        let jpeg = result.unwrap();
        // Should start with SOI
        assert_eq!(jpeg[0], 0xFF);
        assert_eq!(jpeg[1], 0xD8);
        // Should end with EOI
        assert_eq!(jpeg[jpeg.len() - 2], 0xFF);
        assert_eq!(jpeg[jpeg.len() - 1], 0xD9);
    }

    #[test]
    fn test_encode_small_rgb() {
        let encoder = Encoder::new().quality(Quality::Standard(75));
        let pixels = vec![128u8; 16 * 16 * 3]; // 16x16 RGB
        let result = encoder.encode_rgb(&pixels, 16, 16);
        assert!(result.is_ok());
    }

    #[test]
    fn test_invalid_dimensions() {
        let encoder = Encoder::new();
        let pixels = vec![0u8; 0];
        let result = encoder.encode_rgb(&pixels, 0, 0);
        assert!(result.is_err());
    }
}
