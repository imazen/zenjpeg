//! JPEG encoder implementation.
//!
//! This module provides the main encoder interface for creating JPEG images.

use crate::jpegli::adaptive_quant::compute_aq_strength_map;
use crate::jpegli::alloc::{
    checked_size_2d, try_alloc_filled, try_alloc_zeroed_f32, validate_dimensions,
    DEFAULT_MAX_PIXELS,
};
use crate::jpegli::color;
use crate::jpegli::consts::{
    DCT_BLOCK_SIZE, DCT_SIZE, ICC_PROFILE_SIGNATURE, JPEG_NATURAL_ORDER, JPEG_ZIGZAG_ORDER,
    MARKER_APP0, MARKER_APP2, MARKER_DHT, MARKER_DQT, MARKER_DRI, MARKER_EOI, MARKER_SOF0,
    MARKER_SOF2, MARKER_SOI, MARKER_SOS, MAX_ICC_BYTES_PER_MARKER, XYB_ICC_PROFILE,
};
use crate::jpegli::dct::forward_dct_8x8;
use crate::jpegli::entropy::{self, EntropyEncoder};
use crate::jpegli::error::{Error, Result};
use crate::jpegli::huffman::HuffmanEncodeTable;
use crate::jpegli::huffman_opt::{
    FrequencyCounter, OptimizedHuffmanTables, OptimizedTable, ProgressiveTokenBuffer,
};
use crate::jpegli::quant::{self, Quality, QuantTable, ZeroBiasParams};
use crate::jpegli::types::{ColorSpace, JpegMode, PixelFormat, Subsampling};
use crate::jpegli::xyb::srgb_to_scaled_xyb;

/// Progressive scan parameters.
#[derive(Debug, Clone)]
struct ProgressiveScan {
    /// Component indices in this scan (0=Y, 1=Cb, 2=Cr)
    components: Vec<u8>,
    /// Spectral selection start (0=DC, 1-63=AC)
    ss: u8,
    /// Spectral selection end (0-63)
    se: u8,
    /// Successive approximation high bit (previous pass)
    ah: u8,
    /// Successive approximation low bit (current pass)
    al: u8,
}

/// Encoder configuration.
#[derive(Debug, Clone)]
pub struct EncoderConfig {
    /// Image width
    pub width: u32,
    /// Image height
    pub height: u32,
    /// Input pixel format
    pub pixel_format: PixelFormat,
    /// Quality setting
    pub quality: Quality,
    /// Encoding mode
    pub mode: JpegMode,
    /// Chroma subsampling
    pub subsampling: Subsampling,
    /// Use XYB color space
    pub use_xyb: bool,
    /// Restart interval (0 = disabled)
    pub restart_interval: u16,
    /// Use optimized Huffman tables
    pub optimize_huffman: bool,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            width: 0,
            height: 0,
            pixel_format: PixelFormat::Rgb,
            quality: Quality::default(),
            mode: JpegMode::Baseline,
            // Use 4:4:4 - this is what the encoder actually supports currently
            subsampling: Subsampling::S444,
            use_xyb: false,
            restart_interval: 0,
            // Match C++ jpegli default: optimize_coding = true
            optimize_huffman: true,
        }
    }
}

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

    /// Sets the quality.
    #[must_use]
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

    /// Encodes as baseline JPEG.
    fn encode_baseline(&self, data: &[u8]) -> Result<Vec<u8>> {
        let mut output = Vec::with_capacity(data.len() / 4);

        if self.config.use_xyb {
            self.encode_baseline_xyb(data, &mut output)
        } else {
            self.encode_baseline_ycbcr(data, &mut output)
        }
    }

    /// Encodes using standard YCbCr color space.
    fn encode_baseline_ycbcr(&self, data: &[u8], output: &mut Vec<u8>) -> Result<Vec<u8>> {
        // Convert to YCbCr using f32 precision throughout (matches C++ jpegli)
        let (y_plane, cb_plane, cr_plane) = self.convert_to_ycbcr_f32(data)?;

        // Generate quantization tables (3 separate tables like C++ cjpegli)
        let y_quant = quant::generate_quant_table(self.config.quality, 0, ColorSpace::YCbCr, false);
        let cb_quant =
            quant::generate_quant_table(self.config.quality, 1, ColorSpace::YCbCr, false);
        let cr_quant =
            quant::generate_quant_table(self.config.quality, 2, ColorSpace::YCbCr, false);

        // Quantize all blocks first (needed for both standard and optimized encoding)
        let (y_blocks, cb_blocks, cr_blocks) = self.quantize_all_blocks(
            &y_plane, &cb_plane, &cr_plane, &y_quant, &cb_quant, &cr_quant,
        )?;
        let is_color = self.config.pixel_format != PixelFormat::Gray;

        // Write JPEG structure
        self.write_header(output)?;
        self.write_quant_tables(output, &y_quant, &cb_quant, &cr_quant)?;
        self.write_frame_header(output)?;

        // For optimized Huffman, build tables from block frequencies before writing DHT
        let scan_data = if self.config.optimize_huffman {
            let tables =
                self.build_optimized_tables(&y_blocks, &cb_blocks, &cr_blocks, is_color)?;
            self.write_huffman_tables_optimized(output, &tables)?;

            if self.config.restart_interval > 0 {
                self.write_restart_interval(output)?;
            }
            self.write_scan_header(output)?;

            // Encode with optimized tables
            self.encode_with_tables(&y_blocks, &cb_blocks, &cr_blocks, is_color, &tables)?
        } else {
            self.write_huffman_tables(output)?;

            if self.config.restart_interval > 0 {
                self.write_restart_interval(output)?;
            }
            self.write_scan_header(output)?;

            // Encode with standard tables
            self.encode_blocks_standard(&y_blocks, &cb_blocks, &cr_blocks, is_color)?
        };

        output.extend_from_slice(&scan_data);

        // Write EOI
        output.push(0xFF);
        output.push(MARKER_EOI);

        Ok(std::mem::take(output))
    }

    /// Encodes using XYB mode (perceptually optimized color space).
    ///
    /// XYB encoding pipeline:
    /// 1. sRGB → linear RGB → XYB → scaled XYB (values in [0, 1])
    /// 2. Multiply by 255 for JPEG sample range
    /// 3. Level shift by subtracting 128 for DCT
    fn encode_baseline_xyb(&self, data: &[u8], output: &mut Vec<u8>) -> Result<Vec<u8>> {
        let width = self.config.width as usize;
        let height = self.config.height as usize;

        // Convert sRGB to scaled XYB (full color conversion pipeline)
        let (x_plane, y_plane, b_plane) = self.convert_to_scaled_xyb(data)?;

        // Downsample B channel (XYB subsamples B to 1/4 resolution)
        let b_downsampled = self.downsample_2x2_f32(&b_plane, width, height)?;
        let b_width = (width + 1) / 2;
        let b_height = (height + 1) / 2;

        // Generate XYB quantization tables (one per component)
        let x_quant = quant::generate_quant_table(
            self.config.quality,
            0, // X component
            ColorSpace::Rgb,
            true,
        );
        let y_quant = quant::generate_quant_table(
            self.config.quality,
            1, // Y component (luma-like)
            ColorSpace::Rgb,
            true,
        );
        let b_quant = quant::generate_quant_table(
            self.config.quality,
            2, // B component
            ColorSpace::Rgb,
            true,
        );

        // Write JPEG structure for XYB mode (no JFIF, just ICC profile)
        self.write_header_xyb(output)?;
        // Write XYB ICC profile so decoders can interpret the colors correctly
        self.write_icc_profile(output, &XYB_ICC_PROFILE)?;
        self.write_quant_tables_xyb(output, &x_quant, &y_quant, &b_quant)?;
        self.write_frame_header_xyb(output)?;

        // For optimized Huffman, quantize all blocks first to collect frequencies
        let scan_data = if self.config.optimize_huffman {
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
            let (dc_table, ac_table) =
                self.build_optimized_tables_xyb(&x_blocks, &y_blocks, &b_blocks)?;
            self.write_huffman_tables_xyb_optimized(output, &dc_table, &ac_table);

            if self.config.restart_interval > 0 {
                self.write_restart_interval(output)?;
            }
            self.write_scan_header_xyb(output)?;

            // Encode with optimized tables
            self.encode_with_tables_xyb(&x_blocks, &y_blocks, &b_blocks, &dc_table, &ac_table)?
        } else {
            self.write_huffman_tables(output)?;

            if self.config.restart_interval > 0 {
                self.write_restart_interval(output)?;
            }
            self.write_scan_header_xyb(output)?;

            // Encode with standard tables
            self.encode_scan_xyb_float(
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
            )?
        };

        output.extend_from_slice(&scan_data);

        // Write EOI
        output.push(0xFF);
        output.push(MARKER_EOI);

        Ok(std::mem::take(output))
    }

    /// Converts input data to scaled XYB planes.
    ///
    /// Performs the full conversion: sRGB u8 → linear RGB → XYB → scaled XYB
    /// Output values are in [0, 1] range, ready to be scaled to [0, 255] for JPEG.
    fn convert_to_scaled_xyb(&self, data: &[u8]) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let width = self.config.width as usize;
        let height = self.config.height as usize;
        let num_pixels = checked_size_2d(width, height)?;

        let mut x_plane = try_alloc_zeroed_f32(num_pixels, "allocating XYB X plane")?;
        let mut y_plane = try_alloc_zeroed_f32(num_pixels, "allocating XYB Y plane")?;
        let mut b_plane = try_alloc_zeroed_f32(num_pixels, "allocating XYB B plane")?;

        match self.config.pixel_format {
            PixelFormat::Rgb => {
                for i in 0..num_pixels {
                    let (x, y, b) =
                        srgb_to_scaled_xyb(data[i * 3], data[i * 3 + 1], data[i * 3 + 2]);
                    x_plane[i] = x;
                    y_plane[i] = y;
                    b_plane[i] = b;
                }
            }
            PixelFormat::Rgba => {
                for i in 0..num_pixels {
                    let (x, y, b) =
                        srgb_to_scaled_xyb(data[i * 4], data[i * 4 + 1], data[i * 4 + 2]);
                    x_plane[i] = x;
                    y_plane[i] = y;
                    b_plane[i] = b;
                }
            }
            PixelFormat::Gray => {
                // Grayscale: R=G=B
                for i in 0..num_pixels {
                    let (x, y, b) = srgb_to_scaled_xyb(data[i], data[i], data[i]);
                    x_plane[i] = x;
                    y_plane[i] = y;
                    b_plane[i] = b;
                }
            }
            PixelFormat::Bgr => {
                for i in 0..num_pixels {
                    let (x, y, b) =
                        srgb_to_scaled_xyb(data[i * 3 + 2], data[i * 3 + 1], data[i * 3]);
                    x_plane[i] = x;
                    y_plane[i] = y;
                    b_plane[i] = b;
                }
            }
            PixelFormat::Bgra => {
                for i in 0..num_pixels {
                    let (x, y, b) =
                        srgb_to_scaled_xyb(data[i * 4 + 2], data[i * 4 + 1], data[i * 4]);
                    x_plane[i] = x;
                    y_plane[i] = y;
                    b_plane[i] = b;
                }
            }
            PixelFormat::Cmyk => {
                return Err(Error::UnsupportedFeature {
                    feature: "CMYK with XYB mode",
                });
            }
        }

        Ok((x_plane, y_plane, b_plane))
    }

    /// Downsamples a float plane by 2x2 (box filter averaging).
    fn downsample_2x2_f32(&self, plane: &[f32], width: usize, height: usize) -> Result<Vec<f32>> {
        let new_width = (width + 1) / 2;
        let new_height = (height + 1) / 2;
        let result_size = checked_size_2d(new_width, new_height)?;
        let mut result = try_alloc_zeroed_f32(result_size, "allocating downsampled plane")?;

        for y in 0..new_height {
            for x in 0..new_width {
                let x0 = x * 2;
                let y0 = y * 2;
                let x1 = (x0 + 1).min(width - 1);
                let y1 = (y0 + 1).min(height - 1);

                let p00 = plane[y0 * width + x0];
                let p10 = plane[y0 * width + x1];
                let p01 = plane[y1 * width + x0];
                let p11 = plane[y1 * width + x1];

                result[y * new_width + x] = (p00 + p10 + p01 + p11) * 0.25;
            }
        }

        Ok(result)
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
        // Use tokenization-based approach when optimizing Huffman tables
        if self.config.optimize_huffman {
            return self.encode_progressive_optimized(data);
        }

        let mut output = Vec::with_capacity(data.len() / 4);

        // Convert to YCbCr using f32 precision
        let (y_plane, cb_plane, cr_plane) = self.convert_to_ycbcr_f32(data)?;

        // Generate quantization tables (3 separate tables like C++ cjpegli)
        let y_quant = quant::generate_quant_table(self.config.quality, 0, ColorSpace::YCbCr, false);
        let cb_quant =
            quant::generate_quant_table(self.config.quality, 1, ColorSpace::YCbCr, false);
        let cr_quant =
            quant::generate_quant_table(self.config.quality, 2, ColorSpace::YCbCr, false);

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

        // Convert to YCbCr using f32 precision
        let (y_plane, cb_plane, cr_plane) = self.convert_to_ycbcr_f32(data)?;

        // Generate quantization tables (3 separate tables like C++ cjpegli)
        let y_quant = quant::generate_quant_table(self.config.quality, 0, ColorSpace::YCbCr, false);
        let cb_quant =
            quant::generate_quant_table(self.config.quality, 1, ColorSpace::YCbCr, false);
        let cr_quant =
            quant::generate_quant_table(self.config.quality, 2, ColorSpace::YCbCr, false);

        // Quantize all blocks to get full-precision coefficients
        let (y_blocks, cb_blocks, cr_blocks) = self.quantize_all_blocks(
            &y_plane, &cb_plane, &cr_plane, &y_quant, &cb_quant, &cr_quant,
        )?;
        let is_color = self.config.pixel_format != PixelFormat::Gray;
        let num_components = if is_color { 3 } else { 1 };

        // Define progressive scan script
        let scans = self.get_progressive_scan_script(is_color);

        // ========== PASS 1: TOKENIZATION ==========
        // Tokenize all scans to collect symbol statistics
        let mut token_buffer = ProgressiveTokenBuffer::new(num_components, scans.len());

        for (_scan_idx, scan) in scans.iter().enumerate() {
            // Calculate context for this scan
            // Context determines which Huffman table histogram to use
            let context = if scan.ss == 0 && scan.se == 0 {
                // DC scan: use component index as context (0=Y, 1=Cb, 2=Cr)
                scan.components[0]
            } else {
                // AC scan: use num_components + component_index as context
                // This ensures Y always uses luma table, Cb/Cr use chroma table
                // regardless of scan order (which varies with subsampling mode)
                (num_components as u8) + scan.components[0]
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
        // Use explicit luma/chroma grouping to ensure table assignment matches
        // what the replay code expects (luma=0, chroma=1)
        let (num_dc_tables, tables) = token_buffer.generate_luma_chroma_tables(num_components)?;

        // Convert to OptimizedHuffmanTables format for compatibility
        let opt_tables =
            self.build_progressive_huffman_tables(&tables, num_components, num_dc_tables)?;

        // ========== WRITE JPEG STRUCTURE ==========
        self.write_header(&mut output)?;
        self.write_quant_tables(&mut output, &y_quant, &cb_quant, &cr_quant)?;
        self.write_frame_header(&mut output)?; // Uses SOF2 for progressive

        // Write optimized Huffman tables
        self.write_huffman_tables_optimized(&mut output, &opt_tables)?;

        if self.config.restart_interval > 0 {
            self.write_restart_interval(&mut output)?;
        }

        // ========== PASS 2: REPLAY TOKENS ==========
        // Encode each scan by replaying tokens with optimized tables
        for (scan_idx, scan) in scans.iter().enumerate() {
            // Write SOS header
            self.write_progressive_scan_header(&mut output, scan, is_color)?;

            // Replay tokens for this scan
            let scan_data =
                self.replay_progressive_scan(&token_buffer, scan_idx, scan, is_color, &opt_tables)?;
            output.extend_from_slice(&scan_data);
        }

        // Write EOI
        output.push(0xFF);
        output.push(MARKER_EOI);

        Ok(output)
    }

    /// Builds OptimizedHuffmanTables from the clustered tables.
    fn build_progressive_huffman_tables(
        &self,
        tables: &[OptimizedTable],
        num_components: usize,
        num_dc_tables: usize,
    ) -> Result<OptimizedHuffmanTables> {
        // Tables are arranged: DC clusters first, then AC clusters
        // num_dc_tables tells us where DC ends and AC begins

        let dc_luma = tables.first().cloned().unwrap_or_else(|| {
            // Create a minimal default table
            let mut counter = FrequencyCounter::new();
            counter.count(0);
            counter.generate_table_with_dht().unwrap()
        });

        // DC chroma is the second DC table if it exists
        let dc_chroma = if num_components > 1 && num_dc_tables > 1 {
            tables.get(1).cloned().unwrap_or_else(|| dc_luma.clone())
        } else {
            dc_luma.clone()
        };

        // AC tables start after DC tables
        let ac_luma = tables.get(num_dc_tables).cloned().unwrap_or_else(|| {
            let mut counter = FrequencyCounter::new();
            counter.count(0);
            counter.generate_table_with_dht().unwrap()
        });

        // AC chroma is the second AC table if it exists
        let ac_chroma = if num_components > 1 && tables.len() > num_dc_tables + 1 {
            tables
                .get(num_dc_tables + 1)
                .cloned()
                .unwrap_or_else(|| ac_luma.clone())
        } else {
            ac_luma.clone()
        };

        Ok(OptimizedHuffmanTables {
            dc_luma,
            ac_luma,
            dc_chroma,
            ac_chroma,
        })
    }

    /// Replays tokens for a progressive scan with optimized tables.
    fn replay_progressive_scan(
        &self,
        token_buffer: &ProgressiveTokenBuffer,
        scan_idx: usize,
        scan: &ProgressiveScan,
        is_color: bool,
        tables: &OptimizedHuffmanTables,
    ) -> Result<Vec<u8>> {
        let mut encoder = EntropyEncoder::new();

        // Set up Huffman tables
        encoder.set_dc_table(0, tables.dc_luma.table.clone());
        encoder.set_ac_table(0, tables.ac_luma.table.clone());
        if is_color {
            encoder.set_dc_table(1, tables.dc_chroma.table.clone());
            encoder.set_ac_table(1, tables.ac_chroma.table.clone());
        }

        if self.config.restart_interval > 0 {
            encoder.set_restart_interval(self.config.restart_interval);
        }

        // Get scan info
        let scan_info = token_buffer
            .scan_info
            .get(scan_idx)
            .ok_or(Error::InternalError {
                reason: "Scan info not found",
            })?;

        if scan.ss == 0 && scan.se == 0 {
            // DC scan: replay DC tokens
            let tokens = token_buffer.scan_tokens(scan_idx);
            // Create context map for DC (component index -> table index)
            let context_to_table: Vec<usize> = (0..4)
                .map(|c| if is_color && c > 0 { 1 } else { 0 })
                .collect();
            encoder.write_dc_tokens(tokens, &context_to_table)?;
        } else if scan.ah == 0 {
            // AC first scan: replay AC tokens
            let tokens = token_buffer.scan_tokens(scan_idx);
            let table_idx = if is_color && scan.components[0] > 0 {
                1
            } else {
                0
            };
            encoder.write_ac_first_tokens(tokens, table_idx)?;
        } else {
            // AC refinement scan: replay refinement tokens
            let table_idx = if is_color && scan.components[0] > 0 {
                1
            } else {
                0
            };
            encoder.write_ac_refinement_tokens(scan_info, table_idx)?;
        }

        Ok(encoder.finish())
    }

    /// Returns the progressive scan script for level 2.
    fn get_progressive_scan_script(&self, is_color: bool) -> Vec<ProgressiveScan> {
        let num_components = if is_color { 3 } else { 1 };
        let mut scans = Vec::new();

        // For 4:4:4 subsampling, DC can be interleaved
        let dc_interleaved = matches!(self.config.subsampling, Subsampling::S444);

        // DC first scan
        if dc_interleaved && is_color {
            // Interleaved DC for all components
            scans.push(ProgressiveScan {
                components: vec![0, 1, 2],
                ss: 0,
                se: 0,
                ah: 0,
                al: 0,
            });
        } else {
            // Non-interleaved DC
            for c in 0..num_components {
                scans.push(ProgressiveScan {
                    components: vec![c],
                    ss: 0,
                    se: 0,
                    ah: 0,
                    al: 0,
                });
            }
        }

        // AC scans are always non-interleaved
        // For now, use simple progressive without successive approximation
        // TODO: Add progressive level 2 with SA once basic progressive works
        for c in 0..num_components {
            // AC 1-63, full precision (no successive approximation)
            scans.push(ProgressiveScan {
                components: vec![c],
                ss: 1,
                se: 63,
                ah: 0,
                al: 0,
            });
        }

        scans
    }

    /// Writes SOS header for a progressive scan.
    fn write_progressive_scan_header(
        &self,
        output: &mut Vec<u8>,
        scan: &ProgressiveScan,
        is_color: bool,
    ) -> Result<()> {
        output.push(0xFF);
        output.push(MARKER_SOS);

        let num_components = scan.components.len() as u8;
        let length = 6u16 + num_components as u16 * 2;
        output.push((length >> 8) as u8);
        output.push(length as u8);

        output.push(num_components);

        for &comp_idx in &scan.components {
            // Component ID (1-based for YCbCr)
            let comp_id = comp_idx + 1;
            output.push(comp_id);

            // DC/AC table selectors
            // For DC scans (ss=0): use DC table for the component
            // For AC scans (ss>0): use AC table for the component
            let table_selector = if is_color && comp_idx > 0 {
                0x11 // DC table 1, AC table 1 for chroma
            } else {
                0x00 // DC table 0, AC table 0 for luma
            };
            output.push(table_selector);
        }

        output.push(scan.ss); // Spectral selection start
        output.push(scan.se); // Spectral selection end
        output.push((scan.ah << 4) | scan.al); // Successive approximation

        Ok(())
    }

    /// Encodes a single progressive scan.
    fn encode_progressive_scan(
        &self,
        y_blocks: &[[i16; DCT_BLOCK_SIZE]],
        cb_blocks: &[[i16; DCT_BLOCK_SIZE]],
        cr_blocks: &[[i16; DCT_BLOCK_SIZE]],
        scan: &ProgressiveScan,
        is_color: bool,
        tables: &Option<OptimizedHuffmanTables>,
    ) -> Result<Vec<u8>> {
        let mut encoder = EntropyEncoder::new();

        // Set up Huffman tables
        if let Some(ref opt_tables) = tables {
            encoder.set_dc_table(0, opt_tables.dc_luma.table.clone());
            encoder.set_ac_table(0, opt_tables.ac_luma.table.clone());
            if is_color {
                encoder.set_dc_table(1, opt_tables.dc_chroma.table.clone());
                encoder.set_ac_table(1, opt_tables.ac_chroma.table.clone());
            }
        } else {
            encoder.set_dc_table(0, HuffmanEncodeTable::std_dc_luminance());
            encoder.set_ac_table(0, HuffmanEncodeTable::std_ac_luminance());
            if is_color {
                encoder.set_dc_table(1, HuffmanEncodeTable::std_dc_chrominance());
                encoder.set_ac_table(1, HuffmanEncodeTable::std_ac_chrominance());
            }
        }

        if self.config.restart_interval > 0 {
            encoder.set_restart_interval(self.config.restart_interval);
        }

        let width = self.config.width as usize;
        let height = self.config.height as usize;
        let blocks_h = (width + DCT_SIZE - 1) / DCT_SIZE;
        let blocks_v = (height + DCT_SIZE - 1) / DCT_SIZE;

        // Determine scan type and encode accordingly
        if scan.ss == 0 && scan.se == 0 {
            // DC scan (first or refinement)
            self.encode_dc_scan(
                &mut encoder,
                y_blocks,
                cb_blocks,
                cr_blocks,
                scan,
                blocks_h,
                blocks_v,
                is_color,
            )?;
        } else if scan.ah == 0 {
            // AC first scan
            self.encode_ac_first_scan(
                &mut encoder,
                y_blocks,
                cb_blocks,
                cr_blocks,
                scan,
                blocks_h,
                blocks_v,
                is_color,
            )?;
        } else {
            // AC refinement scan
            self.encode_ac_refine_scan(
                &mut encoder,
                y_blocks,
                cb_blocks,
                cr_blocks,
                scan,
                blocks_h,
                blocks_v,
                is_color,
            )?;
        }

        Ok(encoder.finish())
    }

    /// Encodes DC scan (first or refinement).
    fn encode_dc_scan(
        &self,
        encoder: &mut EntropyEncoder,
        y_blocks: &[[i16; DCT_BLOCK_SIZE]],
        cb_blocks: &[[i16; DCT_BLOCK_SIZE]],
        cr_blocks: &[[i16; DCT_BLOCK_SIZE]],
        scan: &ProgressiveScan,
        blocks_h: usize,
        blocks_v: usize,
        is_color: bool,
    ) -> Result<()> {
        for by in 0..blocks_v {
            for bx in 0..blocks_h {
                let block_idx = by * blocks_h + bx;

                for (comp_num, &comp_idx) in scan.components.iter().enumerate() {
                    let blocks: &[[i16; DCT_BLOCK_SIZE]] = match comp_idx {
                        0 => y_blocks,
                        1 => cb_blocks,
                        2 => cr_blocks,
                        _ => {
                            return Err(Error::InternalError {
                                reason: "Invalid component index",
                            })
                        }
                    };

                    if block_idx >= blocks.len() {
                        continue;
                    }

                    let dc = blocks[block_idx][0];
                    let table = if is_color && comp_idx > 0 { 1 } else { 0 };

                    encoder.encode_dc_progressive(dc, comp_num, table, scan.al, scan.ah)?;
                }
            }
        }

        Ok(())
    }

    /// Encodes AC first scan (Ah=0, ss>0).
    fn encode_ac_first_scan(
        &self,
        encoder: &mut EntropyEncoder,
        y_blocks: &[[i16; DCT_BLOCK_SIZE]],
        cb_blocks: &[[i16; DCT_BLOCK_SIZE]],
        cr_blocks: &[[i16; DCT_BLOCK_SIZE]],
        scan: &ProgressiveScan,
        blocks_h: usize,
        blocks_v: usize,
        is_color: bool,
    ) -> Result<()> {
        // AC first scan is always non-interleaved (single component)
        assert_eq!(scan.components.len(), 1);
        let comp_idx = scan.components[0];

        let blocks: &[[i16; DCT_BLOCK_SIZE]] = match comp_idx {
            0 => y_blocks,
            1 => cb_blocks,
            2 => cr_blocks,
            _ => {
                return Err(Error::InternalError {
                    reason: "Invalid component index",
                })
            }
        };

        let table_idx = if is_color && comp_idx > 0 { 1 } else { 0 };

        let mut eob_run = 0u16;

        for by in 0..blocks_v {
            for bx in 0..blocks_h {
                let block_idx = by * blocks_h + bx;

                if block_idx >= blocks.len() {
                    continue;
                }

                encoder.encode_ac_progressive_first(
                    &blocks[block_idx],
                    table_idx,
                    scan.ss,
                    scan.se,
                    scan.al,
                    &mut eob_run,
                )?;
            }
        }

        // Flush remaining EOB run
        encoder.flush_eob_run(table_idx, eob_run)?;

        Ok(())
    }

    /// Encodes AC refinement scan (Ah>0, ss>0).
    fn encode_ac_refine_scan(
        &self,
        encoder: &mut EntropyEncoder,
        y_blocks: &[[i16; DCT_BLOCK_SIZE]],
        cb_blocks: &[[i16; DCT_BLOCK_SIZE]],
        cr_blocks: &[[i16; DCT_BLOCK_SIZE]],
        scan: &ProgressiveScan,
        blocks_h: usize,
        blocks_v: usize,
        is_color: bool,
    ) -> Result<()> {
        // AC refinement scan is always non-interleaved
        assert_eq!(scan.components.len(), 1);
        let comp_idx = scan.components[0];

        let blocks: &[[i16; DCT_BLOCK_SIZE]] = match comp_idx {
            0 => y_blocks,
            1 => cb_blocks,
            2 => cr_blocks,
            _ => {
                return Err(Error::InternalError {
                    reason: "Invalid component index",
                })
            }
        };

        let table_idx = if is_color && comp_idx > 0 { 1 } else { 0 };

        let mut eob_run = 0u16;
        let mut pending_bits: Vec<u8> = Vec::new();

        for by in 0..blocks_v {
            for bx in 0..blocks_h {
                let block_idx = by * blocks_h + bx;

                if block_idx >= blocks.len() {
                    continue;
                }

                encoder.encode_ac_progressive_refine(
                    &blocks[block_idx],
                    table_idx,
                    scan.ss,
                    scan.se,
                    scan.al,
                    scan.ah,
                    &mut eob_run,
                    &mut pending_bits,
                )?;
            }
        }

        // Flush remaining EOB run
        encoder.flush_refine_eob(table_idx, eob_run)?;

        Ok(())
    }

    /// Converts input data to YCbCr planes (u8 version - legacy).
    #[allow(dead_code)]
    fn convert_to_ycbcr(&self, data: &[u8]) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
        let width = self.config.width as usize;
        let height = self.config.height as usize;
        let num_pixels = checked_size_2d(width, height)?;

        match self.config.pixel_format {
            PixelFormat::Gray => {
                let y = data.to_vec();
                let cb = try_alloc_filled(num_pixels, 128u8, "YCbCr Cb plane")?;
                let cr = try_alloc_filled(num_pixels, 128u8, "YCbCr Cr plane")?;
                Ok((y, cb, cr))
            }
            PixelFormat::Rgb => color::rgb_to_ycbcr_planes(data, width, height),
            PixelFormat::Rgba => {
                // Strip alpha and convert
                let rgb: Vec<u8> = data
                    .chunks(4)
                    .flat_map(|chunk| [chunk[0], chunk[1], chunk[2]])
                    .collect();
                color::rgb_to_ycbcr_planes(&rgb, width, height)
            }
            PixelFormat::Bgr => {
                let rgb: Vec<u8> = data
                    .chunks(3)
                    .flat_map(|chunk| [chunk[2], chunk[1], chunk[0]])
                    .collect();
                color::rgb_to_ycbcr_planes(&rgb, width, height)
            }
            PixelFormat::Bgra => {
                let rgb: Vec<u8> = data
                    .chunks(4)
                    .flat_map(|chunk| [chunk[2], chunk[1], chunk[0]])
                    .collect();
                color::rgb_to_ycbcr_planes(&rgb, width, height)
            }
            PixelFormat::Cmyk => Err(Error::UnsupportedFeature {
                feature: "CMYK encoding",
            }),
        }
    }

    /// Converts input data to YCbCr planes using full f32 precision.
    /// This matches C++ jpegli which uses float throughout the pipeline.
    /// Output values are in [0, 255] range (not level-shifted).
    fn convert_to_ycbcr_f32(&self, data: &[u8]) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let width = self.config.width as usize;
        let height = self.config.height as usize;
        let num_pixels = checked_size_2d(width, height)?;

        let mut y_plane = try_alloc_zeroed_f32(num_pixels, "YCbCr Y plane f32")?;
        let mut cb_plane = try_alloc_zeroed_f32(num_pixels, "YCbCr Cb plane f32")?;
        let mut cr_plane = try_alloc_zeroed_f32(num_pixels, "YCbCr Cr plane f32")?;

        match self.config.pixel_format {
            PixelFormat::Gray => {
                for i in 0..num_pixels {
                    y_plane[i] = data[i] as f32;
                    cb_plane[i] = 128.0;
                    cr_plane[i] = 128.0;
                }
            }
            PixelFormat::Rgb => {
                for i in 0..num_pixels {
                    let (y, cb, cr) = color::rgb_to_ycbcr_f32(
                        data[i * 3] as f32,
                        data[i * 3 + 1] as f32,
                        data[i * 3 + 2] as f32,
                    );
                    y_plane[i] = y;
                    cb_plane[i] = cb;
                    cr_plane[i] = cr;
                }
            }
            PixelFormat::Rgba => {
                for i in 0..num_pixels {
                    let (y, cb, cr) = color::rgb_to_ycbcr_f32(
                        data[i * 4] as f32,
                        data[i * 4 + 1] as f32,
                        data[i * 4 + 2] as f32,
                    );
                    y_plane[i] = y;
                    cb_plane[i] = cb;
                    cr_plane[i] = cr;
                }
            }
            PixelFormat::Bgr => {
                for i in 0..num_pixels {
                    let (y, cb, cr) = color::rgb_to_ycbcr_f32(
                        data[i * 3 + 2] as f32,
                        data[i * 3 + 1] as f32,
                        data[i * 3] as f32,
                    );
                    y_plane[i] = y;
                    cb_plane[i] = cb;
                    cr_plane[i] = cr;
                }
            }
            PixelFormat::Bgra => {
                for i in 0..num_pixels {
                    let (y, cb, cr) = color::rgb_to_ycbcr_f32(
                        data[i * 4 + 2] as f32,
                        data[i * 4 + 1] as f32,
                        data[i * 4] as f32,
                    );
                    y_plane[i] = y;
                    cb_plane[i] = cb;
                    cr_plane[i] = cr;
                }
            }
            PixelFormat::Cmyk => {
                return Err(Error::UnsupportedFeature {
                    feature: "CMYK encoding",
                });
            }
        }

        Ok((y_plane, cb_plane, cr_plane))
    }

    /// Writes the JPEG header (SOI + APP0).
    fn write_header(&self, output: &mut Vec<u8>) -> Result<()> {
        // SOI
        output.push(0xFF);
        output.push(MARKER_SOI);

        // APP0 (JFIF header)
        output.push(0xFF);
        output.push(MARKER_APP0);

        let app0_data = [
            0x00, 0x10, // Length
            b'J', b'F', b'I', b'F', 0x00, // Identifier
            0x01, 0x01, // Version 1.01
            0x00, // Units: no units
            0x00, 0x01, // X density
            0x00, 0x01, // Y density
            0x00, 0x00, // No thumbnail
        ];
        output.extend_from_slice(&app0_data);

        Ok(())
    }

    /// Writes the JPEG header for XYB mode (SOI only, no JFIF).
    ///
    /// XYB mode uses RGB component IDs and an ICC profile for color interpretation.
    /// JFIF APP0 is not appropriate because it implies YCbCr colorspace.
    fn write_header_xyb(&self, output: &mut Vec<u8>) -> Result<()> {
        // SOI only - no JFIF marker for XYB mode
        output.push(0xFF);
        output.push(MARKER_SOI);
        Ok(())
    }

    /// Writes an ICC profile to the JPEG output.
    ///
    /// ICC profiles are stored in APP2 marker segments with the signature "ICC_PROFILE\0".
    /// Large profiles are split into multiple segments (max ~65519 bytes per segment).
    fn write_icc_profile(&self, output: &mut Vec<u8>, icc_data: &[u8]) -> Result<()> {
        if icc_data.is_empty() {
            return Ok(());
        }

        // Calculate number of chunks needed
        let num_chunks = (icc_data.len() + MAX_ICC_BYTES_PER_MARKER - 1) / MAX_ICC_BYTES_PER_MARKER;

        let mut offset = 0;
        for chunk_num in 0..num_chunks {
            let chunk_size = (icc_data.len() - offset).min(MAX_ICC_BYTES_PER_MARKER);

            // APP2 marker
            output.push(0xFF);
            output.push(MARKER_APP2);

            // Length: 2 (length field) + 12 (signature) + 2 (chunk info) + data
            let segment_length = 2 + 12 + 2 + chunk_size;
            output.push((segment_length >> 8) as u8);
            output.push(segment_length as u8);

            // ICC_PROFILE signature
            output.extend_from_slice(&ICC_PROFILE_SIGNATURE);

            // Chunk number (1-based) and total chunks
            output.push((chunk_num + 1) as u8);
            output.push(num_chunks as u8);

            // ICC data chunk
            output.extend_from_slice(&icc_data[offset..offset + chunk_size]);

            offset += chunk_size;
        }

        Ok(())
    }

    /// Writes quantization tables (3 separate tables for Y, Cb, Cr).
    /// This matches C++ jpegli behavior with add_two_chroma_tables=true.
    fn write_quant_tables(
        &self,
        output: &mut Vec<u8>,
        y_quant: &QuantTable,
        cb_quant: &QuantTable,
        cr_quant: &QuantTable,
    ) -> Result<()> {
        // Write all 3 tables in one DQT segment
        // Length = 2 + 3 * (1 + 64) = 197 bytes
        output.push(0xFF);
        output.push(MARKER_DQT);
        output.push(0x00);
        output.push(0xC5); // Length: 197 bytes

        // Table 0 (Y) - values must be written in zigzag order
        output.push(0x00); // 8-bit precision, table 0
        for i in 0..DCT_BLOCK_SIZE {
            output.push(y_quant.values[JPEG_NATURAL_ORDER[i] as usize] as u8);
        }

        // Table 1 (Cb)
        output.push(0x01); // 8-bit precision, table 1
        for i in 0..DCT_BLOCK_SIZE {
            output.push(cb_quant.values[JPEG_NATURAL_ORDER[i] as usize] as u8);
        }

        // Table 2 (Cr)
        output.push(0x02); // 8-bit precision, table 2
        for i in 0..DCT_BLOCK_SIZE {
            output.push(cr_quant.values[JPEG_NATURAL_ORDER[i] as usize] as u8);
        }

        Ok(())
    }

    /// Writes quantization tables for XYB mode (3 separate tables).
    fn write_quant_tables_xyb(
        &self,
        output: &mut Vec<u8>,
        r_quant: &QuantTable,
        g_quant: &QuantTable,
        b_quant: &QuantTable,
    ) -> Result<()> {
        // Write all 3 tables in one DQT segment
        // Length = 2 + 3 * (1 + 64) = 197 bytes
        output.push(0xFF);
        output.push(MARKER_DQT);
        output.push(0x00);
        output.push(0xC5); // Length: 197 bytes

        // Table 0 (Red)
        output.push(0x00); // 8-bit precision, table 0
        for i in 0..DCT_BLOCK_SIZE {
            output.push(r_quant.values[JPEG_NATURAL_ORDER[i] as usize] as u8);
        }

        // Table 1 (Green)
        output.push(0x01); // 8-bit precision, table 1
        for i in 0..DCT_BLOCK_SIZE {
            output.push(g_quant.values[JPEG_NATURAL_ORDER[i] as usize] as u8);
        }

        // Table 2 (Blue)
        output.push(0x02); // 8-bit precision, table 2
        for i in 0..DCT_BLOCK_SIZE {
            output.push(b_quant.values[JPEG_NATURAL_ORDER[i] as usize] as u8);
        }

        Ok(())
    }

    /// Writes the frame header (SOF0).
    fn write_frame_header(&self, output: &mut Vec<u8>) -> Result<()> {
        let marker = if self.config.mode == JpegMode::Progressive {
            MARKER_SOF2
        } else {
            MARKER_SOF0
        };

        output.push(0xFF);
        output.push(marker);

        let num_components = if self.config.pixel_format == PixelFormat::Gray {
            1u8
        } else {
            3u8
        };

        let length = 8u16 + num_components as u16 * 3;
        output.push((length >> 8) as u8);
        output.push(length as u8);

        output.push(8); // Sample precision
        output.push((self.config.height >> 8) as u8);
        output.push(self.config.height as u8);
        output.push((self.config.width >> 8) as u8);
        output.push(self.config.width as u8);
        output.push(num_components);

        if num_components == 1 {
            // Grayscale
            output.push(1); // Component ID
            output.push(0x11); // 1x1 sampling
            output.push(0); // Quant table 0
        } else {
            // Y component
            let (h_samp, v_samp) = match self.config.subsampling {
                Subsampling::S444 => (1, 1),
                Subsampling::S422 => (2, 1),
                Subsampling::S420 => (2, 2),
                Subsampling::S440 => (1, 2),
            };

            output.push(1); // Component ID = 1 (Y)
            output.push((h_samp << 4) | v_samp);
            output.push(0); // Quant table 0

            output.push(2); // Component ID = 2 (Cb)
            output.push(0x11); // 1x1 sampling
            output.push(1); // Quant table 1

            output.push(3); // Component ID = 3 (Cr)
            output.push(0x11); // 1x1 sampling
            output.push(2); // Quant table 2 (separate Cr table like C++ cjpegli)
        }

        Ok(())
    }

    /// Writes the frame header for XYB mode (RGB with B subsampling).
    fn write_frame_header_xyb(&self, output: &mut Vec<u8>) -> Result<()> {
        output.push(0xFF);
        output.push(MARKER_SOF0); // Baseline DCT

        // 3 components: R, G, B
        let length = 8u16 + 3 * 3; // 17 bytes
        output.push((length >> 8) as u8);
        output.push(length as u8);

        output.push(8); // Sample precision
        output.push((self.config.height >> 8) as u8);
        output.push(self.config.height as u8);
        output.push((self.config.width >> 8) as u8);
        output.push(self.config.width as u8);
        output.push(3); // Number of components

        // XYB sampling: R:2×2, G:2×2, B:1×1
        // This means R and G are full resolution, B is 1/4 resolution
        output.push(b'R'); // Component ID = 'R' (82)
        output.push(0x22); // 2x2 sampling
        output.push(0); // Quant table 0

        output.push(b'G'); // Component ID = 'G' (71)
        output.push(0x22); // 2x2 sampling
        output.push(1); // Quant table 1

        output.push(b'B'); // Component ID = 'B' (66)
        output.push(0x11); // 1x1 sampling (subsampled)
        output.push(2); // Quant table 2

        Ok(())
    }

    /// Writes Huffman tables.
    fn write_huffman_tables(&self, output: &mut Vec<u8>) -> Result<()> {
        use crate::jpegli::huffman::{
            STD_AC_CHROMINANCE_BITS, STD_AC_CHROMINANCE_VALUES, STD_AC_LUMINANCE_BITS,
            STD_AC_LUMINANCE_VALUES, STD_DC_CHROMINANCE_BITS, STD_DC_CHROMINANCE_VALUES,
            STD_DC_LUMINANCE_BITS, STD_DC_LUMINANCE_VALUES,
        };

        // Helper to write one table
        let write_table = |out: &mut Vec<u8>, class: u8, id: u8, bits: &[u8; 16], values: &[u8]| {
            out.push(0xFF);
            out.push(MARKER_DHT);

            let length = 2 + 1 + 16 + values.len();
            out.push((length >> 8) as u8);
            out.push(length as u8);

            out.push((class << 4) | id);
            out.extend_from_slice(bits);
            out.extend_from_slice(values);
        };

        // DC luminance (class 0, id 0)
        write_table(
            output,
            0,
            0,
            &STD_DC_LUMINANCE_BITS,
            &STD_DC_LUMINANCE_VALUES,
        );

        // AC luminance (class 1, id 0)
        write_table(
            output,
            1,
            0,
            &STD_AC_LUMINANCE_BITS,
            &STD_AC_LUMINANCE_VALUES,
        );

        // DC chrominance (class 0, id 1)
        write_table(
            output,
            0,
            1,
            &STD_DC_CHROMINANCE_BITS,
            &STD_DC_CHROMINANCE_VALUES,
        );

        // AC chrominance (class 1, id 1)
        write_table(
            output,
            1,
            1,
            &STD_AC_CHROMINANCE_BITS,
            &STD_AC_CHROMINANCE_VALUES,
        );

        Ok(())
    }

    /// Writes optimized Huffman tables.
    ///
    /// This is used when `optimize_huffman` is enabled to write the
    /// image-specific optimized tables to the DHT markers.
    fn write_huffman_tables_optimized(
        &self,
        output: &mut Vec<u8>,
        tables: &OptimizedHuffmanTables,
    ) -> Result<()> {
        // Helper to write one table
        let write_table = |out: &mut Vec<u8>, class: u8, id: u8, bits: &[u8; 16], values: &[u8]| {
            out.push(0xFF);
            out.push(MARKER_DHT);

            let length = 2 + 1 + 16 + values.len();
            out.push((length >> 8) as u8);
            out.push(length as u8);

            out.push((class << 4) | id);
            out.extend_from_slice(bits);
            out.extend_from_slice(values);
        };

        // DC luminance (class 0, id 0)
        write_table(output, 0, 0, &tables.dc_luma.bits, &tables.dc_luma.values);

        // AC luminance (class 1, id 0)
        write_table(output, 1, 0, &tables.ac_luma.bits, &tables.ac_luma.values);

        // DC chrominance (class 0, id 1)
        write_table(
            output,
            0,
            1,
            &tables.dc_chroma.bits,
            &tables.dc_chroma.values,
        );

        // AC chrominance (class 1, id 1)
        write_table(
            output,
            1,
            1,
            &tables.ac_chroma.bits,
            &tables.ac_chroma.values,
        );

        Ok(())
    }

    /// Writes restart interval.
    fn write_restart_interval(&self, output: &mut Vec<u8>) -> Result<()> {
        output.push(0xFF);
        output.push(MARKER_DRI);
        output.push(0x00);
        output.push(0x04); // Length
        output.push((self.config.restart_interval >> 8) as u8);
        output.push(self.config.restart_interval as u8);
        Ok(())
    }

    /// Writes scan header.
    fn write_scan_header(&self, output: &mut Vec<u8>) -> Result<()> {
        output.push(0xFF);
        output.push(MARKER_SOS);

        let num_components = if self.config.pixel_format == PixelFormat::Gray {
            1u8
        } else {
            3u8
        };

        let length = 6u16 + num_components as u16 * 2;
        output.push((length >> 8) as u8);
        output.push(length as u8);

        output.push(num_components);

        if num_components == 1 {
            output.push(1); // Component selector
            output.push(0x00); // DC/AC table selectors
        } else {
            output.push(1); // Y component
            output.push(0x00); // DC table 0, AC table 0

            output.push(2); // Cb component
            output.push(0x11); // DC table 1, AC table 1

            output.push(3); // Cr component
            output.push(0x11); // DC table 1, AC table 1
        }

        output.push(0x00); // Ss (spectral selection start)
        output.push(0x3F); // Se (spectral selection end = 63)
        output.push(0x00); // Ah/Al (successive approximation)

        Ok(())
    }

    /// Writes scan header for XYB mode.
    fn write_scan_header_xyb(&self, output: &mut Vec<u8>) -> Result<()> {
        output.push(0xFF);
        output.push(MARKER_SOS);

        // 3 components: R, G, B
        let length = 6u16 + 3 * 2; // 12 bytes
        output.push((length >> 8) as u8);
        output.push(length as u8);

        output.push(3); // Number of components

        // R component: DC table 0, AC table 0
        output.push(b'R');
        output.push(0x00);

        // G component: DC table 0, AC table 0
        output.push(b'G');
        output.push(0x00);

        // B component: DC table 0, AC table 0
        output.push(b'B');
        output.push(0x00);

        output.push(0x00); // Ss (spectral selection start)
        output.push(0x3F); // Se (spectral selection end = 63)
        output.push(0x00); // Ah/Al (successive approximation)

        Ok(())
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
        let input_distance = self.config.quality.to_distance();
        let effective_distance = quant::quant_vals_to_distance(y_quant, c_quant, c_quant);
        let y_zero_bias = ZeroBiasParams::for_ycbcr(effective_distance, 0);
        let cb_zero_bias = ZeroBiasParams::for_ycbcr(effective_distance, 1);
        let cr_zero_bias = ZeroBiasParams::for_ycbcr(effective_distance, 2);

        // Convert Y plane to f32 for AQ computation
        let y_plane_f32: Vec<f32> = y_plane.iter().map(|&v| v as f32).collect();

        // Compute per-block adaptive quantization strength from Y plane
        // C++ uses y_quant_01 = quant_table[1] for dampen calculation
        let y_quant_01 = y_quant.values[1];
        let aq_map = compute_aq_strength_map(&y_plane_f32, width, height, y_quant_01);

        for by in 0..blocks_v {
            for bx in 0..blocks_h {
                // Get per-block aq_strength (C++ AQ produces 0.0-0.2, mean ~0.08)
                let aq_strength = aq_map.get(bx, by);

                // Extract and encode Y block
                let y_block = self.extract_block(y_plane, width, height, bx, by);
                let y_dct = forward_dct_8x8(&y_block);
                let y_quant_coeffs = quant::quantize_block_with_zero_bias(
                    &y_dct,
                    &y_quant.values,
                    &y_zero_bias,
                    aq_strength,
                );
                let y_zigzag = natural_to_zigzag(&y_quant_coeffs);
                encoder.encode_block(&y_zigzag, 0, 0, 0)?;

                if self.config.pixel_format != PixelFormat::Gray {
                    // Cb block
                    let cb_block = self.extract_block(cb_plane, width, height, bx, by);
                    let cb_dct = forward_dct_8x8(&cb_block);
                    let cb_quant_coeffs = quant::quantize_block_with_zero_bias(
                        &cb_dct,
                        &c_quant.values,
                        &cb_zero_bias,
                        aq_strength,
                    );
                    let cb_zigzag = natural_to_zigzag(&cb_quant_coeffs);
                    encoder.encode_block(&cb_zigzag, 1, 1, 1)?;

                    // Cr block
                    let cr_block = self.extract_block(cr_plane, width, height, bx, by);
                    let cr_dct = forward_dct_8x8(&cr_block);
                    let cr_quant_coeffs = quant::quantize_block_with_zero_bias(
                        &cr_dct,
                        &c_quant.values,
                        &cr_zero_bias,
                        aq_strength,
                    );
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
        let input_distance = self.config.quality.to_distance();
        let effective_distance = quant::quant_vals_to_distance(y_quant, cb_quant, cr_quant);
        let y_zero_bias = ZeroBiasParams::for_ycbcr(effective_distance, 0);
        let cb_zero_bias = ZeroBiasParams::for_ycbcr(effective_distance, 1);
        let cr_zero_bias = ZeroBiasParams::for_ycbcr(effective_distance, 2);

        // Compute per-block adaptive quantization strength from Y plane
        // C++ uses y_quant_01 = quant_table[1] for dampen calculation
        let y_quant_01 = y_quant.values[1];
        let aq_map = compute_aq_strength_map(y_plane, width, height, y_quant_01);

        let mut y_blocks = Vec::with_capacity(blocks_h * blocks_v);
        let mut cb_blocks = Vec::with_capacity(if is_color { blocks_h * blocks_v } else { 0 });
        let mut cr_blocks = Vec::with_capacity(if is_color { blocks_h * blocks_v } else { 0 });

        for by in 0..blocks_v {
            for bx in 0..blocks_h {
                // Get per-block aq_strength
                let aq_strength = aq_map.get(bx, by);

                let y_block = self.extract_block_ycbcr_f32(y_plane, width, height, bx, by);
                let y_dct = forward_dct_8x8(&y_block);
                let y_quant_coeffs = quant::quantize_block_with_zero_bias(
                    &y_dct,
                    &y_quant.values,
                    &y_zero_bias,
                    aq_strength,
                );
                y_blocks.push(natural_to_zigzag(&y_quant_coeffs));

                if is_color {
                    let cb_block = self.extract_block_ycbcr_f32(cb_plane, width, height, bx, by);
                    let cb_dct = forward_dct_8x8(&cb_block);
                    let cb_quant_coeffs = quant::quantize_block_with_zero_bias(
                        &cb_dct,
                        &cb_quant.values,
                        &cb_zero_bias,
                        aq_strength,
                    );
                    cb_blocks.push(natural_to_zigzag(&cb_quant_coeffs));

                    let cr_block = self.extract_block_ycbcr_f32(cr_plane, width, height, bx, by);
                    let cr_dct = forward_dct_8x8(&cr_block);
                    let cr_quant_coeffs = quant::quantize_block_with_zero_bias(
                        &cr_dct,
                        &cr_quant.values,
                        &cr_zero_bias,
                        aq_strength,
                    );
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

        // Collect frequencies from all blocks
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

        // Build optimized tables with DHT data
        let dc_luma = dc_luma_freq.generate_table_with_dht()?;
        let ac_luma = ac_luma_freq.generate_table_with_dht()?;

        let (dc_chroma, ac_chroma) = if is_color {
            (
                dc_chroma_freq.generate_table_with_dht()?,
                ac_chroma_freq.generate_table_with_dht()?,
            )
        } else {
            // Use standard tables for grayscale (won't be used but needed for structure)
            use crate::jpegli::huffman::{
                STD_AC_CHROMINANCE_BITS, STD_AC_CHROMINANCE_VALUES, STD_DC_CHROMINANCE_BITS,
                STD_DC_CHROMINANCE_VALUES,
            };
            use crate::jpegli::huffman_opt::OptimizedTable;

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

    /// Encodes blocks using the provided Huffman tables.
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

        for (i, y_block) in y_blocks.iter().enumerate() {
            encoder.encode_block(y_block, 0, 0, 0)?;

            if is_color {
                encoder.encode_block(&cb_blocks[i], 1, 1, 1)?;
                encoder.encode_block(&cr_blocks[i], 2, 1, 1)?;
            }

            encoder.check_restart();
        }

        Ok(encoder.finish())
    }

    /// Encodes blocks using standard (fixed) Huffman tables - single pass.
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

        for (i, y_block) in y_blocks.iter().enumerate() {
            encoder.encode_block(y_block, 0, 0, 0)?;

            if is_color {
                encoder.encode_block(&cb_blocks[i], 1, 1, 1)?;
                encoder.encode_block(&cr_blocks[i], 2, 1, 1)?;
            }

            encoder.check_restart();
        }

        Ok(encoder.finish())
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
                        let x_block = self.extract_block_f32(x_plane, width, height, bx, by);
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
                        let y_block = self.extract_block_f32(y_plane, width, height, bx, by);
                        let y_dct = forward_dct_8x8(&y_block);
                        let y_quant_coeffs = quant::quantize_block(&y_dct, &y_quant.values);
                        y_blocks.push(natural_to_zigzag(&y_quant_coeffs));
                    }
                }

                // Process 1 B block (from downsampled plane)
                let b_block = self.extract_block_f32(b_plane, b_width, b_height, mcu_x, mcu_y);
                let b_dct = forward_dct_8x8(&b_block);
                let b_quant_coeffs = quant::quantize_block(&b_dct, &b_quant.values);
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
        crate::jpegli::huffman_opt::OptimizedTable,
        crate::jpegli::huffman_opt::OptimizedTable,
    )> {
        let mut dc_freq = FrequencyCounter::new();
        let mut ac_freq = FrequencyCounter::new();

        // Collect frequencies from all components
        // Note: XYB MCU order is 4 X blocks, 4 Y blocks, 1 B block per MCU
        // But since all share the same table, we just iterate through them

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

        // Generate optimized tables
        let dc_table = dc_freq.generate_table_with_dht()?;
        let ac_table = ac_freq.generate_table_with_dht()?;

        Ok((dc_table, ac_table))
    }

    /// Encodes XYB blocks using optimized Huffman tables.
    #[allow(clippy::too_many_arguments)]
    fn encode_with_tables_xyb(
        &self,
        x_blocks: &[[i16; DCT_BLOCK_SIZE]],
        y_blocks: &[[i16; DCT_BLOCK_SIZE]],
        b_blocks: &[[i16; DCT_BLOCK_SIZE]],
        dc_table: &crate::jpegli::huffman_opt::OptimizedTable,
        ac_table: &crate::jpegli::huffman_opt::OptimizedTable,
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

    /// Writes DHT markers for XYB optimized tables.
    fn write_huffman_tables_xyb_optimized(
        &self,
        output: &mut Vec<u8>,
        dc_table: &crate::jpegli::huffman_opt::OptimizedTable,
        ac_table: &crate::jpegli::huffman_opt::OptimizedTable,
    ) {
        let write_table = |out: &mut Vec<u8>, class: u8, id: u8, bits: &[u8; 16], values: &[u8]| {
            out.push(0xFF);
            out.push(MARKER_DHT);
            let length = 2 + 1 + 16 + values.len();
            out.push((length >> 8) as u8);
            out.push(length as u8);
            out.push((class << 4) | id);
            out.extend_from_slice(bits);
            out.extend_from_slice(values);
        };

        // DC table (class=0, id=0)
        write_table(output, 0, 0, &dc_table.bits, &dc_table.values);
        // AC table (class=1, id=0)
        write_table(output, 1, 0, &ac_table.bits, &ac_table.values);
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
                        let x_block = self.extract_block_f32(x_plane, width, height, bx, by);
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
                        let y_block = self.extract_block_f32(y_plane, width, height, bx, by);
                        let y_dct = forward_dct_8x8(&y_block);
                        let y_quant_coeffs = quant::quantize_block(&y_dct, &y_quant.values);
                        let y_zigzag = natural_to_zigzag(&y_quant_coeffs);
                        encoder.encode_block(&y_zigzag, 1, 0, 0)?;
                    }
                }

                // Process 1 B block (from downsampled plane)
                let b_block = self.extract_block_f32(b_plane, b_width, b_height, mcu_x, mcu_y);
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
                // Scale from [0, 1] to [0, 255], then level shift by -128
                block[y * DCT_SIZE + x] = plane[idx] * 255.0 - 128.0;
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
fn natural_to_zigzag(natural: &[i16; DCT_BLOCK_SIZE]) -> [i16; DCT_BLOCK_SIZE] {
    let mut zigzag = [0i16; DCT_BLOCK_SIZE];
    for i in 0..DCT_BLOCK_SIZE {
        zigzag[JPEG_ZIGZAG_ORDER[i] as usize] = natural[i];
    }
    zigzag
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_creation() {
        let encoder = Encoder::new()
            .width(640)
            .height(480)
            .quality(Quality::from_quality(90.0));

        assert_eq!(encoder.config.width, 640);
        assert_eq!(encoder.config.height, 480);
    }

    #[test]
    fn test_encoder_validation() {
        let encoder = Encoder::new();
        assert!(encoder.validate().is_err());

        let encoder = Encoder::new().width(100).height(100);
        assert!(encoder.validate().is_ok());
    }

    #[test]
    fn test_encode_small_gray() {
        let encoder = Encoder::new()
            .width(8)
            .height(8)
            .pixel_format(PixelFormat::Gray)
            .quality(Quality::from_quality(90.0));

        let data = vec![128u8; 64];
        let result = encoder.encode(&data);
        assert!(result.is_ok());

        let jpeg = result.unwrap();
        // Should start with SOI
        assert_eq!(jpeg[0], 0xFF);
        assert_eq!(jpeg[1], MARKER_SOI);
        // Should end with EOI
        assert_eq!(jpeg[jpeg.len() - 2], 0xFF);
        assert_eq!(jpeg[jpeg.len() - 1], MARKER_EOI);
    }

    #[test]
    fn test_encode_rgb_xyb_mode() {
        // Test XYB mode encoding with a 16x16 RGB image
        let encoder = Encoder::new()
            .width(16)
            .height(16)
            .pixel_format(PixelFormat::Rgb)
            .quality(Quality::from_quality(90.0))
            .use_xyb(true);

        // Create a simple gradient test image
        let mut data = vec![0u8; 16 * 16 * 3];
        for y in 0..16 {
            for x in 0..16 {
                let idx = (y * 16 + x) * 3;
                data[idx] = (x * 16) as u8; // Red gradient
                data[idx + 1] = (y * 16) as u8; // Green gradient
                data[idx + 2] = 128; // Constant blue
            }
        }

        let result = encoder.encode(&data);
        assert!(result.is_ok(), "XYB encoding failed: {:?}", result.err());

        let jpeg = result.unwrap();
        // Should start with SOI
        assert_eq!(jpeg[0], 0xFF);
        assert_eq!(jpeg[1], MARKER_SOI);
        // Should end with EOI
        assert_eq!(jpeg[jpeg.len() - 2], 0xFF);
        assert_eq!(jpeg[jpeg.len() - 1], MARKER_EOI);

        // Should be a valid size (not too small)
        assert!(jpeg.len() > 100, "JPEG too small: {} bytes", jpeg.len());
        println!("XYB encoded JPEG size: {} bytes", jpeg.len());
    }

    #[test]
    fn test_encode_rgb_xyb_larger() {
        // Test XYB mode with a larger image (32x32)
        let encoder = Encoder::new()
            .width(32)
            .height(32)
            .pixel_format(PixelFormat::Rgb)
            .quality(Quality::from_quality(75.0))
            .use_xyb(true);

        // Create a test pattern
        let mut data = vec![0u8; 32 * 32 * 3];
        for y in 0..32 {
            for x in 0..32 {
                let idx = (y * 32 + x) * 3;
                // Checkerboard pattern
                let checker = ((x / 4) + (y / 4)) % 2 == 0;
                data[idx] = if checker { 255 } else { 0 }; // Red
                data[idx + 1] = if checker { 0 } else { 255 }; // Green
                data[idx + 2] = 128; // Blue
            }
        }

        let result = encoder.encode(&data);
        assert!(result.is_ok(), "XYB encoding failed: {:?}", result.err());

        let jpeg = result.unwrap();
        assert_eq!(jpeg[0], 0xFF);
        assert_eq!(jpeg[1], MARKER_SOI);
        assert_eq!(jpeg[jpeg.len() - 2], 0xFF);
        assert_eq!(jpeg[jpeg.len() - 1], MARKER_EOI);
        println!("XYB encoded 32x32 JPEG size: {} bytes", jpeg.len());
    }

    #[test]
    fn test_huffman_optimization_produces_valid_jpeg() {
        // Create a gradient test image
        let width = 64u32;
        let height = 64u32;
        let mut data = vec![0u8; (width * height * 3) as usize];

        for y in 0..height as usize {
            for x in 0..width as usize {
                let idx = (y * width as usize + x) * 3;
                data[idx] = (x * 4) as u8; // R
                data[idx + 1] = (y * 4) as u8; // G
                data[idx + 2] = ((x + y) * 2) as u8; // B
            }
        }

        let encoder = Encoder::new()
            .width(width)
            .height(height)
            .quality(Quality::from_quality(75.0))
            .optimize_huffman(true);

        let result = encoder.encode(&data);
        assert!(
            result.is_ok(),
            "Optimized Huffman encoding failed: {:?}",
            result.err()
        );

        let jpeg = result.unwrap();
        assert_eq!(jpeg[0], 0xFF);
        assert_eq!(jpeg[1], MARKER_SOI);
        assert_eq!(jpeg[jpeg.len() - 2], 0xFF);
        assert_eq!(jpeg[jpeg.len() - 1], MARKER_EOI);

        // Verify it's decodable
        let decoded = jpeg_decoder::Decoder::new(&jpeg[..]).decode();
        assert!(
            decoded.is_ok(),
            "Optimized JPEG not decodable: {:?}",
            decoded.err()
        );
    }

    #[test]
    fn test_huffman_optimization_reduces_file_size() {
        // Create a more complex test image that benefits from optimization
        let width = 128u32;
        let height = 128u32;
        let mut data = vec![0u8; (width * height * 3) as usize];

        // Create a pattern that will have non-uniform symbol frequencies
        for y in 0..height as usize {
            for x in 0..width as usize {
                let idx = (y * width as usize + x) * 3;
                // Create blocks with varying content
                let block_type = ((x / 16) + (y / 16)) % 4;
                match block_type {
                    0 => {
                        // Solid color
                        data[idx] = 180;
                        data[idx + 1] = 180;
                        data[idx + 2] = 180;
                    }
                    1 => {
                        // Gradient
                        data[idx] = (x * 2) as u8;
                        data[idx + 1] = (y * 2) as u8;
                        data[idx + 2] = 100;
                    }
                    2 => {
                        // Checkerboard
                        let checker = ((x + y) % 2) as u8 * 255;
                        data[idx] = checker;
                        data[idx + 1] = checker;
                        data[idx + 2] = checker;
                    }
                    _ => {
                        // Texture
                        data[idx] = ((x * 5 + y * 3) % 256) as u8;
                        data[idx + 1] = ((x * 3 + y * 7) % 256) as u8;
                        data[idx + 2] = ((x * 2 + y * 2) % 256) as u8;
                    }
                }
            }
        }

        // Encode without optimization
        let jpeg_standard = Encoder::new()
            .width(width)
            .height(height)
            .quality(Quality::from_quality(75.0))
            .optimize_huffman(false)
            .encode(&data)
            .expect("Standard encoding failed");

        // Encode with optimization
        let jpeg_optimized = Encoder::new()
            .width(width)
            .height(height)
            .quality(Quality::from_quality(75.0))
            .optimize_huffman(true)
            .encode(&data)
            .expect("Optimized encoding failed");

        println!(
            "Standard size: {} bytes, Optimized size: {} bytes, Savings: {:.1}%",
            jpeg_standard.len(),
            jpeg_optimized.len(),
            (1.0 - jpeg_optimized.len() as f64 / jpeg_standard.len() as f64) * 100.0
        );

        // Optimized should be smaller or equal (never larger)
        assert!(
            jpeg_optimized.len() <= jpeg_standard.len(),
            "Optimized ({}) should not be larger than standard ({})",
            jpeg_optimized.len(),
            jpeg_standard.len()
        );

        // Verify both are decodable
        let decoded_std = jpeg_decoder::Decoder::new(&jpeg_standard[..]).decode();
        let decoded_opt = jpeg_decoder::Decoder::new(&jpeg_optimized[..]).decode();
        assert!(decoded_std.is_ok(), "Standard JPEG not decodable");
        assert!(decoded_opt.is_ok(), "Optimized JPEG not decodable");
    }

    #[test]
    fn test_xyb_huffman_optimization() {
        // Create test image for XYB mode
        let width = 64u32;
        let height = 64u32;
        let mut data = vec![0u8; (width * height * 3) as usize];

        for y in 0..height as usize {
            for x in 0..width as usize {
                let idx = (y * width as usize + x) * 3;
                data[idx] = (x * 4) as u8;
                data[idx + 1] = (y * 4) as u8;
                data[idx + 2] = ((x + y) * 2) as u8;
            }
        }

        // Encode XYB without optimization
        let jpeg_standard = Encoder::new()
            .width(width)
            .height(height)
            .quality(Quality::from_quality(75.0))
            .use_xyb(true)
            .optimize_huffman(false)
            .encode(&data)
            .expect("Standard XYB encoding failed");

        // Encode XYB with optimization
        let jpeg_optimized = Encoder::new()
            .width(width)
            .height(height)
            .quality(Quality::from_quality(75.0))
            .use_xyb(true)
            .optimize_huffman(true)
            .encode(&data)
            .expect("Optimized XYB encoding failed");

        println!(
            "XYB Standard: {} bytes, Optimized: {} bytes, Savings: {:.1}%",
            jpeg_standard.len(),
            jpeg_optimized.len(),
            (1.0 - jpeg_optimized.len() as f64 / jpeg_standard.len() as f64) * 100.0
        );

        // Verify both have valid JPEG structure
        assert_eq!(jpeg_standard[0], 0xFF);
        assert_eq!(jpeg_standard[1], MARKER_SOI);
        assert_eq!(jpeg_optimized[0], 0xFF);
        assert_eq!(jpeg_optimized[1], MARKER_SOI);

        // Optimized should be smaller or equal
        assert!(
            jpeg_optimized.len() <= jpeg_standard.len(),
            "XYB Optimized ({}) should not be larger than standard ({})",
            jpeg_optimized.len(),
            jpeg_standard.len()
        );
    }
}
