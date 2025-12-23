//! JPEG encoder implementation.
//!
//! This module provides the main encoder interface for creating JPEG images.

use crate::color;
use crate::consts::{
    DCT_BLOCK_SIZE, DCT_SIZE, JPEG_NATURAL_ORDER, JPEG_ZIGZAG_ORDER, MARKER_APP0, MARKER_DHT,
    MARKER_DQT, MARKER_DRI, MARKER_EOI, MARKER_SOF0, MARKER_SOF2, MARKER_SOI, MARKER_SOS,
};
use crate::dct::forward_dct_8x8;
use crate::entropy::EntropyEncoder;
use crate::error::{Error, Result};
use crate::huffman::HuffmanEncodeTable;
use crate::quant::{self, Quality, QuantTable};
use crate::types::{ColorSpace, JpegMode, PixelFormat, Subsampling};

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
            subsampling: Subsampling::S420,
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

    /// Enables XYB color space.
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

        // Convert to YCbCr
        let (y_plane, cb_plane, cr_plane) = self.convert_to_ycbcr(data)?;

        // Generate quantization tables
        let y_quant = quant::generate_quant_table(
            self.config.quality,
            0,
            ColorSpace::YCbCr,
            self.config.use_xyb,
        );
        let c_quant = quant::generate_quant_table(
            self.config.quality,
            1,
            ColorSpace::YCbCr,
            self.config.use_xyb,
        );

        // Write JPEG structure
        self.write_header(&mut output)?;
        self.write_quant_tables(&mut output, &y_quant, &c_quant)?;
        self.write_frame_header(&mut output)?;
        self.write_huffman_tables(&mut output)?;

        if self.config.restart_interval > 0 {
            self.write_restart_interval(&mut output)?;
        }

        self.write_scan_header(&mut output)?;

        // Encode image data
        let scan_data = self.encode_scan(&y_plane, &cb_plane, &cr_plane, &y_quant, &c_quant)?;
        output.extend_from_slice(&scan_data);

        // Write EOI
        output.push(0xFF);
        output.push(MARKER_EOI);

        Ok(output)
    }

    /// Encodes as progressive JPEG.
    fn encode_progressive(&self, _data: &[u8]) -> Result<Vec<u8>> {
        // TODO: Implement progressive encoding
        Err(Error::UnsupportedFeature {
            feature: "progressive encoding (not yet implemented)",
        })
    }

    /// Converts input data to YCbCr planes.
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

    /// Encodes the scan data.
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

        for by in 0..blocks_v {
            for bx in 0..blocks_h {
                // Extract and encode Y block
                let y_block = self.extract_block(y_plane, width, height, bx, by);
                let y_dct = forward_dct_8x8(&y_block);
                let y_quant_coeffs = quant::quantize_block(&y_dct, &y_quant.values);
                let y_zigzag = natural_to_zigzag(&y_quant_coeffs);
                encoder.encode_block(&y_zigzag, 0, 0, 0)?;

                if self.config.pixel_format != PixelFormat::Gray {
                    // Cb block
                    let cb_block = self.extract_block(cb_plane, width, height, bx, by);
                    let cb_dct = forward_dct_8x8(&cb_block);
                    let cb_quant_coeffs = quant::quantize_block(&cb_dct, &c_quant.values);
                    let cb_zigzag = natural_to_zigzag(&cb_quant_coeffs);
                    encoder.encode_block(&cb_zigzag, 1, 1, 1)?;

                    // Cr block
                    let cr_block = self.extract_block(cr_plane, width, height, bx, by);
                    let cr_dct = forward_dct_8x8(&cr_block);
                    let cr_quant_coeffs = quant::quantize_block(&cr_dct, &c_quant.values);
                    let cr_zigzag = natural_to_zigzag(&cr_quant_coeffs);
                    encoder.encode_block(&cr_zigzag, 2, 1, 1)?;
                }

                encoder.check_restart();
            }
        }

        Ok(encoder.finish())
    }

    /// Extracts an 8x8 block from a plane with level shift.
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
}
