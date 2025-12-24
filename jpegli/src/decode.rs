//! JPEG decoder implementation.
//!
//! This module provides the main decoder interface for reading JPEG images.
//!
//! # ICC Profile Support
//!
//! The decoder can extract and apply embedded ICC profiles, including XYB profiles
//! used by jpegli. ICC profile support requires enabling `cms-lcms2` or `cms-moxcms` feature.
//!
//! ```ignore
//! use jpegli::decode::Decoder;
//!
//! let decoder = Decoder::new().apply_icc(true);
//! let decoded = decoder.decode(&jpeg_data)?;
//! ```

use crate::color;
use crate::consts::{
    DCT_BLOCK_SIZE, DCT_SIZE, JPEG_NATURAL_ORDER, MARKER_APP0, MARKER_COM, MARKER_DHT, MARKER_DQT,
    MARKER_DRI, MARKER_EOI, MARKER_SOF0, MARKER_SOF1, MARKER_SOF2, MARKER_SOI, MARKER_SOS,
    MAX_COMPONENTS, MAX_HUFFMAN_TABLES, MAX_QUANT_TABLES,
};
use crate::entropy::EntropyDecoder;
use crate::error::{Error, Result};
use crate::huffman::HuffmanDecodeTable;
use crate::icc::{extract_icc_profile, is_xyb_profile};
#[cfg(any(feature = "cms-lcms2", feature = "cms-moxcms"))]
use crate::icc::apply_icc_transform;
use crate::idct::inverse_dct_8x8;
use crate::quant::dequantize_block;
use crate::types::{ColorSpace, Component, Dimensions, JpegMode, PixelFormat};

/// Decoder configuration.
#[derive(Debug, Clone)]
pub struct DecoderConfig {
    /// Output pixel format (None = use source format)
    pub output_format: Option<PixelFormat>,
    /// Whether to apply fancy upsampling
    pub fancy_upsampling: bool,
    /// Whether to apply block smoothing
    pub block_smoothing: bool,
    /// Whether to apply embedded ICC profile (requires cms feature)
    pub apply_icc: bool,
}

impl Default for DecoderConfig {
    fn default() -> Self {
        Self {
            output_format: None,
            fancy_upsampling: false,
            block_smoothing: false,
            // Apply ICC by default when CMS is available
            apply_icc: cfg!(any(feature = "cms-lcms2", feature = "cms-moxcms")),
        }
    }
}

/// Information about a decoded JPEG.
#[derive(Debug, Clone)]
pub struct JpegInfo {
    /// Image dimensions
    pub dimensions: Dimensions,
    /// Color space
    pub color_space: ColorSpace,
    /// Sample precision (8 or 12 bits)
    pub precision: u8,
    /// Number of components
    pub num_components: u8,
    /// Encoding mode
    pub mode: JpegMode,
    /// Whether an ICC profile is embedded
    pub has_icc_profile: bool,
    /// Whether the ICC profile is an XYB profile
    pub is_xyb: bool,
}

/// JPEG decoder.
pub struct Decoder {
    config: DecoderConfig,
}

impl Decoder {
    /// Creates a new decoder with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: DecoderConfig::default(),
        }
    }

    /// Creates a decoder from configuration.
    #[must_use]
    pub fn from_config(config: DecoderConfig) -> Self {
        Self { config }
    }

    /// Sets the output pixel format.
    #[must_use]
    pub fn output_format(mut self, format: PixelFormat) -> Self {
        self.config.output_format = Some(format);
        self
    }

    /// Enables fancy upsampling.
    #[must_use]
    pub fn fancy_upsampling(mut self, enable: bool) -> Self {
        self.config.fancy_upsampling = enable;
        self
    }

    /// Enables block smoothing.
    #[must_use]
    pub fn block_smoothing(mut self, enable: bool) -> Self {
        self.config.block_smoothing = enable;
        self
    }

    /// Enables ICC profile application.
    ///
    /// When enabled, embedded ICC profiles will be applied to convert
    /// the image to sRGB. This is required for correct display of
    /// XYB-encoded images.
    ///
    /// Note: Requires `cms-lcms2` or `cms-moxcms` feature to be enabled.
    /// Without a CMS feature, this setting has no effect.
    #[must_use]
    pub fn apply_icc(mut self, enable: bool) -> Self {
        self.config.apply_icc = enable;
        self
    }

    /// Reads JPEG info without decoding.
    pub fn read_info(&self, data: &[u8]) -> Result<JpegInfo> {
        let mut parser = JpegParser::new(data)?;
        parser.read_header()?;
        Ok(parser.info())
    }

    /// Decodes a JPEG image.
    pub fn decode(&self, data: &[u8]) -> Result<DecodedImage> {
        let mut parser = JpegParser::new(data)?;
        parser.decode()?;

        let info = parser.info();
        let output_format = self.config.output_format.unwrap_or(PixelFormat::Rgb);

        // Convert to output format
        let mut pixels = parser.to_pixels(output_format)?;

        // Apply ICC profile if enabled and present
        #[cfg(any(feature = "cms-lcms2", feature = "cms-moxcms"))]
        if self.config.apply_icc && output_format == PixelFormat::Rgb {
            if let Some(ref icc_profile) = parser.icc_profile {
                pixels = apply_icc_transform(
                    &pixels,
                    info.dimensions.width as usize,
                    info.dimensions.height as usize,
                    icc_profile,
                )?;
            }
        }

        Ok(DecodedImage {
            width: info.dimensions.width,
            height: info.dimensions.height,
            format: output_format,
            data: pixels,
        })
    }
}

impl Default for Decoder {
    fn default() -> Self {
        Self::new()
    }
}

/// A decoded image.
#[derive(Debug, Clone)]
pub struct DecodedImage {
    /// Image width
    pub width: u32,
    /// Image height
    pub height: u32,
    /// Pixel format
    pub format: PixelFormat,
    /// Pixel data
    pub data: Vec<u8>,
}

/// Internal JPEG parser state.
struct JpegParser<'a> {
    data: &'a [u8],
    position: usize,

    // Frame info
    width: u32,
    height: u32,
    precision: u8,
    num_components: u8,
    mode: JpegMode,

    // Component info
    components: [Component; MAX_COMPONENTS],

    // Tables
    quant_tables: [Option<[u16; DCT_BLOCK_SIZE]>; MAX_QUANT_TABLES],
    dc_tables: [Option<HuffmanDecodeTable>; MAX_HUFFMAN_TABLES],
    ac_tables: [Option<HuffmanDecodeTable>; MAX_HUFFMAN_TABLES],

    // Restart
    restart_interval: u16,

    // Decoded coefficient data
    coeffs: Vec<Vec<[i16; DCT_BLOCK_SIZE]>>, // Per component

    // ICC profile (extracted from raw data, not during parsing)
    icc_profile: Option<Vec<u8>>,
}

impl<'a> JpegParser<'a> {
    fn new(data: &'a [u8]) -> Result<Self> {
        // Check for SOI
        if data.len() < 2 || data[0] != 0xFF || data[1] != MARKER_SOI {
            return Err(Error::InvalidJpegData {
                reason: "missing SOI marker",
            });
        }

        // Extract ICC profile from raw data upfront
        let icc_profile = extract_icc_profile(data);

        Ok(Self {
            data,
            position: 2,
            width: 0,
            height: 0,
            precision: 8,
            num_components: 0,
            mode: JpegMode::Baseline,
            components: std::array::from_fn(|_| Component::default()),
            quant_tables: [None, None, None, None],
            dc_tables: [None, None, None, None],
            ac_tables: [None, None, None, None],
            restart_interval: 0,
            coeffs: Vec::new(),
            icc_profile,
        })
    }

    fn read_u8(&mut self) -> Result<u8> {
        if self.position >= self.data.len() {
            return Err(Error::UnexpectedEof {
                context: "reading byte",
            });
        }
        let byte = self.data[self.position];
        self.position += 1;
        Ok(byte)
    }

    fn read_u16(&mut self) -> Result<u16> {
        let high = self.read_u8()? as u16;
        let low = self.read_u8()? as u16;
        Ok((high << 8) | low)
    }

    fn read_marker(&mut self) -> Result<u8> {
        loop {
            let byte = self.read_u8()?;
            if byte != 0xFF {
                continue;
            }

            let marker = self.read_u8()?;
            if marker != 0x00 && marker != 0xFF {
                return Ok(marker);
            }
        }
    }

    fn read_header(&mut self) -> Result<()> {
        loop {
            let marker = self.read_marker()?;

            match marker {
                MARKER_SOF0 | MARKER_SOF1 => {
                    self.mode = JpegMode::Baseline;
                    self.parse_frame_header()?;
                    return Ok(());
                }
                MARKER_SOF2 => {
                    self.mode = JpegMode::Progressive;
                    self.parse_frame_header()?;
                    return Ok(());
                }
                MARKER_DQT => self.parse_quant_table()?,
                MARKER_DHT => self.parse_huffman_table()?,
                MARKER_DRI => self.parse_restart_interval()?,
                MARKER_APP0..=0xEF | MARKER_COM => self.skip_segment()?,
                MARKER_EOI => {
                    return Err(Error::InvalidJpegData {
                        reason: "unexpected EOI before frame header",
                    });
                }
                _ => self.skip_segment()?,
            }
        }
    }

    fn parse_frame_header(&mut self) -> Result<()> {
        let length = self.read_u16()?;
        if length < 8 {
            return Err(Error::InvalidJpegData {
                reason: "frame header too short",
            });
        }

        self.precision = self.read_u8()?;
        self.height = self.read_u16()? as u32;
        self.width = self.read_u16()? as u32;
        self.num_components = self.read_u8()?;

        if self.num_components > MAX_COMPONENTS as u8 {
            return Err(Error::UnsupportedFeature {
                feature: "more than 4 components",
            });
        }

        for i in 0..self.num_components as usize {
            self.components[i].id = self.read_u8()?;
            let sampling = self.read_u8()?;
            self.components[i].h_samp_factor = sampling >> 4;
            self.components[i].v_samp_factor = sampling & 0x0F;
            self.components[i].quant_table_idx = self.read_u8()?;
        }

        Ok(())
    }

    fn parse_quant_table(&mut self) -> Result<()> {
        let mut length = self.read_u16()? as i32 - 2;

        while length > 0 {
            let info = self.read_u8()?;
            let precision = info >> 4;
            let table_idx = (info & 0x0F) as usize;

            if table_idx >= MAX_QUANT_TABLES {
                return Err(Error::InvalidQuantTable {
                    table_idx: table_idx as u8,
                    reason: "table index out of range",
                });
            }

            // Read values in zigzag order (as stored in JPEG)
            let mut zigzag_values = [0u16; DCT_BLOCK_SIZE];

            if precision == 0 {
                // 8-bit values
                for i in 0..DCT_BLOCK_SIZE {
                    zigzag_values[i] = self.read_u8()? as u16;
                }
                length -= 65;
            } else {
                // 16-bit values
                for i in 0..DCT_BLOCK_SIZE {
                    zigzag_values[i] = self.read_u16()?;
                }
                length -= 129;
            }

            // Convert from zigzag order to natural order for dequantization
            let mut natural_values = [0u16; DCT_BLOCK_SIZE];
            for i in 0..DCT_BLOCK_SIZE {
                natural_values[JPEG_NATURAL_ORDER[i] as usize] = zigzag_values[i];
            }

            self.quant_tables[table_idx] = Some(natural_values);
        }

        Ok(())
    }

    fn parse_huffman_table(&mut self) -> Result<()> {
        let mut length = self.read_u16()? as i32 - 2;

        while length > 0 {
            let info = self.read_u8()?;
            let table_class = info >> 4; // 0 = DC, 1 = AC
            let table_idx = (info & 0x0F) as usize;

            if table_idx >= MAX_HUFFMAN_TABLES {
                return Err(Error::InvalidHuffmanTable {
                    table_idx: table_idx as u8,
                    reason: "table index out of range",
                });
            }

            let mut bits = [0u8; 16];
            for i in 0..16 {
                bits[i] = self.read_u8()?;
            }

            let num_values: usize = bits.iter().map(|&b| b as usize).sum();
            let mut values = vec![0u8; num_values];
            for i in 0..num_values {
                values[i] = self.read_u8()?;
            }

            length -= 17 + num_values as i32;

            let table = HuffmanDecodeTable::from_bits_values(&bits, &values)?;

            if table_class == 0 {
                self.dc_tables[table_idx] = Some(table);
            } else {
                self.ac_tables[table_idx] = Some(table);
            }
        }

        Ok(())
    }

    fn parse_restart_interval(&mut self) -> Result<()> {
        let _length = self.read_u16()?;
        self.restart_interval = self.read_u16()?;
        Ok(())
    }

    fn skip_segment(&mut self) -> Result<()> {
        let length = self.read_u16()? as usize;
        if length < 2 {
            return Err(Error::InvalidJpegData {
                reason: "segment length too short",
            });
        }
        self.position += length - 2;
        Ok(())
    }

    fn decode(&mut self) -> Result<()> {
        // First read header
        self.position = 2; // Skip SOI
        self.read_header()?;

        // Continue parsing until we hit SOS
        loop {
            let marker = self.read_marker()?;

            match marker {
                MARKER_SOS => {
                    self.parse_scan()?;
                    // After scan, look for more markers
                }
                MARKER_DQT => self.parse_quant_table()?,
                MARKER_DHT => self.parse_huffman_table()?,
                MARKER_DRI => self.parse_restart_interval()?,
                MARKER_EOI => break,
                MARKER_APP0..=0xEF | MARKER_COM => self.skip_segment()?,
                _ => self.skip_segment()?,
            }
        }

        Ok(())
    }

    fn parse_scan(&mut self) -> Result<()> {
        let _length = self.read_u16()?;
        let num_components = self.read_u8()?;

        let mut scan_components = Vec::with_capacity(num_components as usize);

        for _ in 0..num_components {
            let component_id = self.read_u8()?;
            let tables = self.read_u8()?;
            let dc_table = tables >> 4;
            let ac_table = tables & 0x0F;

            // Find component index
            let comp_idx = self.components[..self.num_components as usize]
                .iter()
                .position(|c| c.id == component_id)
                .ok_or(Error::InvalidJpegData {
                    reason: "unknown component in scan",
                })?;

            scan_components.push((comp_idx, dc_table, ac_table));
        }

        let _ss = self.read_u8()?; // Spectral selection start
        let _se = self.read_u8()?; // Spectral selection end
        let ah_al = self.read_u8()?;
        let _ah = ah_al >> 4;
        let _al = ah_al & 0x0F;

        // Decode entropy-coded segment
        self.decode_scan(&scan_components)?;

        Ok(())
    }

    fn decode_scan(&mut self, scan_components: &[(usize, u8, u8)]) -> Result<()> {
        // Initialize coefficient storage
        let blocks_h = ((self.width + 7) / 8) as usize;
        let blocks_v = ((self.height + 7) / 8) as usize;

        if self.coeffs.is_empty() {
            for _ in 0..self.num_components {
                self.coeffs
                    .push(vec![[0i16; DCT_BLOCK_SIZE]; blocks_h * blocks_v]);
            }
        }

        // Set up entropy decoder
        let scan_data = &self.data[self.position..];
        let mut decoder = EntropyDecoder::new(scan_data);

        for (comp_idx, dc_table, ac_table) in scan_components {
            if let Some(table) = &self.dc_tables[*dc_table as usize] {
                decoder.set_dc_table(*dc_table as usize, table.clone());
            }
            if let Some(table) = &self.ac_tables[*ac_table as usize] {
                decoder.set_ac_table(*ac_table as usize, table.clone());
            }
        }

        // Simplified decoding (no MCU interleaving)
        for by in 0..blocks_v {
            for bx in 0..blocks_h {
                let block_idx = by * blocks_h + bx;

                for (comp_idx, dc_table, ac_table) in scan_components {
                    let coeffs =
                        decoder.decode_block(*comp_idx, *dc_table as usize, *ac_table as usize)?;
                    self.coeffs[*comp_idx][block_idx] = coeffs;
                }
            }
        }

        self.position += decoder.position();
        Ok(())
    }

    fn info(&self) -> JpegInfo {
        let has_icc = self.icc_profile.is_some();
        let is_xyb = self.icc_profile.as_ref().is_some_and(|p| is_xyb_profile(p));

        // Determine color space, considering XYB profile
        let color_space = if is_xyb {
            ColorSpace::Xyb
        } else {
            match self.num_components {
                1 => ColorSpace::Grayscale,
                3 => ColorSpace::YCbCr,
                4 => ColorSpace::Cmyk,
                _ => ColorSpace::Unknown,
            }
        };

        JpegInfo {
            dimensions: Dimensions::new(self.width, self.height),
            color_space,
            precision: self.precision,
            num_components: self.num_components,
            mode: self.mode,
            has_icc_profile: has_icc,
            is_xyb,
        }
    }

    fn to_pixels(&self, format: PixelFormat) -> Result<Vec<u8>> {
        if self.coeffs.is_empty() {
            return Err(Error::InternalError {
                reason: "no decoded data",
            });
        }

        let width = self.width as usize;
        let height = self.height as usize;
        let blocks_h = (width + 7) / 8;
        let blocks_v = (height + 7) / 8;

        // Dequantize and IDCT all blocks
        let mut planes: Vec<Vec<u8>> = Vec::new();

        for comp_idx in 0..self.num_components as usize {
            let quant_idx = self.components[comp_idx].quant_table_idx as usize;
            let quant = self.quant_tables[quant_idx]
                .as_ref()
                .ok_or(Error::InternalError {
                    reason: "missing quantization table",
                })?;

            let mut plane = vec![0u8; width * height];

            for by in 0..blocks_v {
                for bx in 0..blocks_h {
                    let block_idx = by * blocks_h + bx;
                    let coeffs = &self.coeffs[comp_idx][block_idx];

                    // Convert to natural order and dequantize
                    let mut natural_coeffs = [0i16; DCT_BLOCK_SIZE];
                    for (i, &zi) in JPEG_NATURAL_ORDER[..DCT_BLOCK_SIZE].iter().enumerate() {
                        natural_coeffs[zi as usize] = coeffs[i];
                    }

                    let dequant = dequantize_block(&natural_coeffs, quant);
                    let pixels = inverse_dct_8x8(&dequant);

                    // Copy to plane with level shift
                    for y in 0..DCT_SIZE {
                        for x in 0..DCT_SIZE {
                            let px = bx * DCT_SIZE + x;
                            let py = by * DCT_SIZE + y;
                            if px < width && py < height {
                                let val = (pixels[y * DCT_SIZE + x] + 128.0).round();
                                plane[py * width + px] = val.clamp(0.0, 255.0) as u8;
                            }
                        }
                    }
                }
            }

            planes.push(plane);
        }

        // Convert to output format
        match (self.num_components, format) {
            (1, PixelFormat::Gray) => Ok(planes[0].clone()),
            (1, PixelFormat::Rgb) => {
                let mut rgb = vec![0u8; width * height * 3];
                for (i, &y) in planes[0].iter().enumerate() {
                    rgb[i * 3] = y;
                    rgb[i * 3 + 1] = y;
                    rgb[i * 3 + 2] = y;
                }
                Ok(rgb)
            }
            (3, PixelFormat::Rgb) => Ok(color::ycbcr_planes_to_rgb(
                &planes[0], &planes[1], &planes[2], width, height,
            )),
            _ => Err(Error::UnsupportedFeature {
                feature: "unsupported color conversion",
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::Encoder;
    use crate::quant::Quality;

    #[test]
    fn test_decoder_creation() {
        let decoder = Decoder::new()
            .output_format(PixelFormat::Rgb)
            .fancy_upsampling(true);

        assert_eq!(decoder.config.output_format, Some(PixelFormat::Rgb));
        assert!(decoder.config.fancy_upsampling);
    }

    #[test]
    fn test_encode_decode_roundtrip_gray() {
        // Create a simple 8x8 grayscale image
        let width = 8;
        let height = 8;
        let mut input = vec![0u8; width * height];
        for y in 0..height {
            for x in 0..width {
                input[y * width + x] = ((x + y) * 16) as u8;
            }
        }

        // Encode
        let encoder = Encoder::new()
            .width(width as u32)
            .height(height as u32)
            .pixel_format(PixelFormat::Gray)
            .quality(Quality::from_quality(95.0));

        let jpeg = encoder.encode(&input).expect("encoding should succeed");

        // Verify JPEG structure
        assert_eq!(jpeg[0], 0xFF);
        assert_eq!(jpeg[1], 0xD8); // SOI
        assert_eq!(jpeg[jpeg.len() - 2], 0xFF);
        assert_eq!(jpeg[jpeg.len() - 1], 0xD9); // EOI

        // Decode
        let decoder = Decoder::new().output_format(PixelFormat::Gray);
        let decoded = decoder.decode(&jpeg).expect("decoding should succeed");

        assert_eq!(decoded.width, width as u32);
        assert_eq!(decoded.height, height as u32);
        assert_eq!(decoded.data.len(), width * height);

        // Check pixel values are reasonably close (JPEG is lossy)
        let mut max_diff = 0i32;
        for i in 0..input.len() {
            let diff = (input[i] as i32 - decoded.data[i] as i32).abs();
            max_diff = max_diff.max(diff);
        }
        // At quality 95, differences should be small
        assert!(max_diff < 20, "max_diff {} too large", max_diff);
    }

    #[test]
    fn test_encode_decode_roundtrip_rgb() {
        // Create a simple 16x16 RGB image
        let width = 16;
        let height = 16;
        let mut input = vec![0u8; width * height * 3];
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 3;
                input[idx] = (x * 16) as u8; // R
                input[idx + 1] = (y * 16) as u8; // G
                input[idx + 2] = 128; // B
            }
        }

        // Encode
        let encoder = Encoder::new()
            .width(width as u32)
            .height(height as u32)
            .pixel_format(PixelFormat::Rgb)
            .quality(Quality::from_quality(95.0));

        let jpeg = encoder.encode(&input).expect("encoding should succeed");

        // Decode
        let decoder = Decoder::new().output_format(PixelFormat::Rgb);
        let decoded = decoder.decode(&jpeg).expect("decoding should succeed");

        assert_eq!(decoded.width, width as u32);
        assert_eq!(decoded.height, height as u32);
        assert_eq!(decoded.data.len(), width * height * 3);

        // Check pixel values are reasonably close
        let mut max_diff = 0i32;
        for i in 0..input.len() {
            let diff = (input[i] as i32 - decoded.data[i] as i32).abs();
            max_diff = max_diff.max(diff);
        }
        // At quality 95, differences should be small
        assert!(max_diff < 30, "max_diff {} too large", max_diff);
    }
}
