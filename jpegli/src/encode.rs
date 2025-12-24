//! JPEG encoder implementation.
//!
//! This module provides the main encoder interface for creating JPEG images.

use crate::color;
use crate::consts::{
    DCT_BLOCK_SIZE, DCT_SIZE, ICC_PROFILE_SIGNATURE, JPEG_NATURAL_ORDER, JPEG_ZIGZAG_ORDER,
    MARKER_APP0, MARKER_APP2, MARKER_DHT, MARKER_DQT, MARKER_DRI, MARKER_EOI, MARKER_SOF0,
    MARKER_SOF2, MARKER_SOI, MARKER_SOS, MAX_ICC_BYTES_PER_MARKER, XYB_ICC_PROFILE,
};
use crate::dct::forward_dct_8x8;
use crate::entropy::EntropyEncoder;
use crate::error::{Error, Result};
use crate::huffman::HuffmanEncodeTable;
use crate::quant::{self, Quality, QuantTable, ZeroBiasParams};
use crate::types::{ColorSpace, JpegMode, PixelFormat, Subsampling};
use crate::xyb::srgb_to_scaled_xyb;

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
            optimize_huffman: false,
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
        if self.config.width == 0 || self.config.height == 0 {
            return Err(Error::InvalidDimensions {
                width: self.config.width,
                height: self.config.height,
                reason: "dimensions cannot be zero",
            });
        }

        if self.config.width > 65535 || self.config.height > 65535 {
            return Err(Error::InvalidDimensions {
                width: self.config.width,
                height: self.config.height,
                reason: "dimensions exceed maximum (65535)",
            });
        }

        Ok(())
    }

    /// Encodes the image data.
    pub fn encode(&self, data: &[u8]) -> Result<Vec<u8>> {
        self.validate()?;

        let expected_size = self.config.width as usize
            * self.config.height as usize
            * self.config.pixel_format.bytes_per_pixel();

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

        // Generate quantization tables
        let y_quant = quant::generate_quant_table(self.config.quality, 0, ColorSpace::YCbCr, false);
        let c_quant = quant::generate_quant_table(self.config.quality, 1, ColorSpace::YCbCr, false);

        // Write JPEG structure
        self.write_header(output)?;
        self.write_quant_tables(output, &y_quant, &c_quant)?;
        self.write_frame_header(output)?;
        self.write_huffman_tables(output)?;

        if self.config.restart_interval > 0 {
            self.write_restart_interval(output)?;
        }

        self.write_scan_header(output)?;

        // Encode image data using f32 pipeline for full precision
        let scan_data = self.encode_scan_f32(&y_plane, &cb_plane, &cr_plane, &y_quant, &c_quant)?;
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
        let b_downsampled = self.downsample_2x2_f32(&b_plane, width, height);
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
        self.write_huffman_tables(output)?;

        if self.config.restart_interval > 0 {
            self.write_restart_interval(output)?;
        }

        self.write_scan_header_xyb(output)?;

        // Encode image data with XYB MCU structure (float-based)
        let scan_data = self.encode_scan_xyb_float(
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
        )?;
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
        let num_pixels = width * height;

        let mut x_plane = vec![0.0f32; num_pixels];
        let mut y_plane = vec![0.0f32; num_pixels];
        let mut b_plane = vec![0.0f32; num_pixels];

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
    fn downsample_2x2_f32(&self, plane: &[f32], width: usize, height: usize) -> Vec<f32> {
        let new_width = (width + 1) / 2;
        let new_height = (height + 1) / 2;
        let mut result = vec![0.0f32; new_width * new_height];

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

        result
    }

    /// Encodes as progressive JPEG.
    fn encode_progressive(&self, _data: &[u8]) -> Result<Vec<u8>> {
        // TODO: Implement progressive encoding
        Err(Error::UnsupportedFeature {
            feature: "progressive encoding (not yet implemented)",
        })
    }

    /// Converts input data to YCbCr planes (u8 version - legacy).
    #[allow(dead_code)]
    fn convert_to_ycbcr(&self, data: &[u8]) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
        let width = self.config.width as usize;
        let height = self.config.height as usize;

        match self.config.pixel_format {
            PixelFormat::Gray => {
                let y = data.to_vec();
                let cb = vec![128u8; width * height];
                let cr = vec![128u8; width * height];
                Ok((y, cb, cr))
            }
            PixelFormat::Rgb => Ok(color::rgb_to_ycbcr_planes(data, width, height)),
            PixelFormat::Rgba => {
                // Strip alpha and convert
                let rgb: Vec<u8> = data
                    .chunks(4)
                    .flat_map(|chunk| [chunk[0], chunk[1], chunk[2]])
                    .collect();
                Ok(color::rgb_to_ycbcr_planes(&rgb, width, height))
            }
            PixelFormat::Bgr => {
                let rgb: Vec<u8> = data
                    .chunks(3)
                    .flat_map(|chunk| [chunk[2], chunk[1], chunk[0]])
                    .collect();
                Ok(color::rgb_to_ycbcr_planes(&rgb, width, height))
            }
            PixelFormat::Bgra => {
                let rgb: Vec<u8> = data
                    .chunks(4)
                    .flat_map(|chunk| [chunk[2], chunk[1], chunk[0]])
                    .collect();
                Ok(color::rgb_to_ycbcr_planes(&rgb, width, height))
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
        let num_pixels = width * height;

        let mut y_plane = vec![0.0f32; num_pixels];
        let mut cb_plane = vec![0.0f32; num_pixels];
        let mut cr_plane = vec![0.0f32; num_pixels];

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

    /// Writes quantization tables.
    fn write_quant_tables(
        &self,
        output: &mut Vec<u8>,
        y_quant: &QuantTable,
        c_quant: &QuantTable,
    ) -> Result<()> {
        // DQT for Y (table 0) - values must be written in zigzag order
        output.push(0xFF);
        output.push(MARKER_DQT);
        output.push(0x00);
        output.push(0x43); // Length: 67 bytes
        output.push(0x00); // 8-bit precision, table 0
        for i in 0..DCT_BLOCK_SIZE {
            // For zigzag position i, output the quant value for natural position JPEG_NATURAL_ORDER[i]
            output.push(y_quant.values[JPEG_NATURAL_ORDER[i] as usize] as u8);
        }

        // DQT for C (table 1) - values must be written in zigzag order
        output.push(0xFF);
        output.push(MARKER_DQT);
        output.push(0x00);
        output.push(0x43);
        output.push(0x01); // 8-bit precision, table 1
        for i in 0..DCT_BLOCK_SIZE {
            output.push(c_quant.values[JPEG_NATURAL_ORDER[i] as usize] as u8);
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
            output.push(1); // Quant table 1
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
        use crate::huffman::{
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

        // Zero-bias parameters for each component (matches C++ jpegli)
        // Without adaptive quantization, aq_strength = 0.0
        let y_zero_bias = ZeroBiasParams::default();
        let cb_zero_bias = ZeroBiasParams::default();
        let cr_zero_bias = ZeroBiasParams::default();
        let aq_strength = 0.0f32;

        for by in 0..blocks_v {
            for bx in 0..blocks_h {
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

    /// Encodes the scan data using f32 planes for full precision.
    /// This matches C++ jpegli which uses float throughout the pipeline.
    fn encode_scan_f32(
        &self,
        y_plane: &[f32],
        cb_plane: &[f32],
        cr_plane: &[f32],
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

        // 4:4:4 encoding (no subsampling)
        let blocks_h = (width + 7) / 8;
        let blocks_v = (height + 7) / 8;

        // Zero-bias parameters for each component (matches C++ jpegli)
        // Without adaptive quantization, aq_strength = 0.0
        let y_zero_bias = ZeroBiasParams::default();
        let cb_zero_bias = ZeroBiasParams::default();
        let cr_zero_bias = ZeroBiasParams::default();
        let aq_strength = 0.0f32;

        for by in 0..blocks_v {
            for bx in 0..blocks_h {
                // Extract and encode Y block
                let y_block = self.extract_block_ycbcr_f32(y_plane, width, height, bx, by);
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
                    let cb_block = self.extract_block_ycbcr_f32(cb_plane, width, height, bx, by);
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
                    let cr_block = self.extract_block_ycbcr_f32(cr_plane, width, height, bx, by);
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
}
