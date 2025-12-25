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

use crate::alloc::{
    checked_size_2d, try_alloc_dct_blocks, try_alloc_vec, try_alloc_zeroed, validate_dimensions,
    MemoryTracker, DEFAULT_MAX_MEMORY, DEFAULT_MAX_PIXELS, JPEG_MAX_DIMENSION,
};
use crate::color;
use crate::consts::{
    DCT_BLOCK_SIZE, DCT_SIZE, JPEG_NATURAL_ORDER, MARKER_APP0, MARKER_COM, MARKER_DHT, MARKER_DQT,
    MARKER_DRI, MARKER_EOI, MARKER_SOF0, MARKER_SOF1, MARKER_SOF2, MARKER_SOI, MARKER_SOS,
    MAX_COMPONENTS, MAX_HUFFMAN_TABLES, MAX_QUANT_TABLES,
};
use crate::entropy::EntropyDecoder;
use crate::error::{Error, Result};
use crate::huffman::HuffmanDecodeTable;
#[cfg(any(feature = "cms-lcms2", feature = "cms-moxcms"))]
use crate::icc::apply_icc_transform;
use crate::icc::{extract_icc_profile, is_xyb_profile};
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
    /// Maximum pixels allowed (for DoS protection).
    /// Default is 100 megapixels. Set to 0 for unlimited.
    pub max_pixels: u64,
    /// Maximum total memory for allocations (for DoS protection).
    /// Default is 512 MB. Set to 0 for unlimited.
    pub max_memory: usize,
}

impl Default for DecoderConfig {
    fn default() -> Self {
        Self {
            output_format: None,
            fancy_upsampling: false,
            block_smoothing: false,
            // Apply ICC by default when CMS is available
            apply_icc: cfg!(any(feature = "cms-lcms2", feature = "cms-moxcms")),
            max_pixels: DEFAULT_MAX_PIXELS,
            max_memory: DEFAULT_MAX_MEMORY,
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

    /// Sets the maximum number of pixels allowed (for DoS protection).
    ///
    /// Default is 100 megapixels. Set to 0 for unlimited.
    #[must_use]
    pub fn max_pixels(mut self, pixels: u64) -> Self {
        self.config.max_pixels = pixels;
        self
    }

    /// Sets the maximum memory allowed for allocations during decoding.
    ///
    /// Default is 512 MB. Set to `usize::MAX` for unlimited.
    /// This prevents memory exhaustion attacks from malicious images.
    #[must_use]
    pub fn max_memory(mut self, bytes: usize) -> Self {
        self.config.max_memory = bytes;
        self
    }

    /// Reads JPEG info without decoding.
    pub fn read_info(&self, data: &[u8]) -> Result<JpegInfo> {
        let mut parser = JpegParser::new(data, self.config.max_pixels)?;
        parser.read_header()?;
        Ok(parser.info())
    }

    /// Decodes a JPEG image.
    pub fn decode(&self, data: &[u8]) -> Result<DecodedImage> {
        let mut parser = JpegParser::new(data, self.config.max_pixels)?;
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

    // Security limits
    max_pixels: u64,
}

impl<'a> JpegParser<'a> {
    fn new(data: &'a [u8], max_pixels: u64) -> Result<Self> {
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
            max_pixels,
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
        // Validate precision: must be 8 for baseline JPEG, 8 or 12 for extended
        if self.precision != 8 && self.precision != 12 {
            return Err(Error::InvalidJpegData {
                reason: "invalid data precision (must be 8 or 12)",
            });
        }

        self.height = self.read_u16()? as u32;
        self.width = self.read_u16()? as u32;

        // Validate dimensions against security limits
        // max_pixels == 0 means unlimited
        let effective_max = if self.max_pixels == 0 {
            u64::MAX
        } else {
            self.max_pixels
        };
        validate_dimensions(self.width, self.height, effective_max)?;

        self.num_components = self.read_u8()?;

        // Validate num_components
        if self.num_components == 0 {
            return Err(Error::InvalidJpegData {
                reason: "number of components is zero",
            });
        }
        if self.num_components > MAX_COMPONENTS as u8 {
            return Err(Error::UnsupportedFeature {
                feature: "more than 4 components",
            });
        }

        // Validate marker length matches expected size
        let expected_length = 8 + 3 * self.num_components as u16;
        if length != expected_length {
            return Err(Error::InvalidJpegData {
                reason: "SOF marker length mismatch",
            });
        }

        for i in 0..self.num_components as usize {
            self.components[i].id = self.read_u8()?;
            let sampling = self.read_u8()?;
            let h_samp = sampling >> 4;
            let v_samp = sampling & 0x0F;

            // Validate sampling factors are non-zero and <= 4
            if h_samp == 0 || v_samp == 0 {
                return Err(Error::InvalidJpegData {
                    reason: "sampling factor is zero",
                });
            }
            if h_samp > 4 || v_samp > 4 {
                return Err(Error::InvalidJpegData {
                    reason: "sampling factor exceeds maximum (4)",
                });
            }

            self.components[i].h_samp_factor = h_samp;
            self.components[i].v_samp_factor = v_samp;

            let quant_idx = self.read_u8()?;
            // Validate quant table index
            if quant_idx as usize >= MAX_QUANT_TABLES {
                return Err(Error::InvalidJpegData {
                    reason: "quantization table index out of range",
                });
            }
            self.components[i].quant_table_idx = quant_idx;
        }

        Ok(())
    }

    fn parse_quant_table(&mut self) -> Result<()> {
        let mut length = self.read_u16()? as i32 - 2;

        while length > 0 {
            let info = self.read_u8()?;
            let precision = info >> 4;
            let table_idx = (info & 0x0F) as usize;

            // Validate precision (0 = 8-bit, 1 = 16-bit)
            if precision > 1 {
                return Err(Error::InvalidQuantTable {
                    table_idx: table_idx as u8,
                    reason: "invalid precision (must be 0 or 1)",
                });
            }

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
                    let val = self.read_u8()? as u16;
                    if val == 0 {
                        return Err(Error::InvalidQuantTable {
                            table_idx: table_idx as u8,
                            reason: "quantization value is zero",
                        });
                    }
                    zigzag_values[i] = val;
                }
                length -= 65;
            } else {
                // 16-bit values
                for i in 0..DCT_BLOCK_SIZE {
                    let val = self.read_u16()?;
                    if val == 0 {
                        return Err(Error::InvalidQuantTable {
                            table_idx: table_idx as u8,
                            reason: "quantization value is zero",
                        });
                    }
                    zigzag_values[i] = val;
                }
                length -= 129;
            }

            // Validate DQT marker length consistency
            if length < 0 {
                return Err(Error::InvalidJpegData {
                    reason: "DQT marker length mismatch",
                });
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

            // Validate table class (must be 0 for DC or 1 for AC)
            if table_class > 1 {
                return Err(Error::InvalidHuffmanTable {
                    table_idx: table_idx as u8,
                    reason: "invalid table class (must be 0 or 1)",
                });
            }

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

            // Validate that we didn't read past the marker length
            if length < 0 {
                return Err(Error::InvalidJpegData {
                    reason: "DHT marker length mismatch",
                });
            }

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

        // Validate num_components in scan
        if num_components == 0 {
            return Err(Error::InvalidJpegData {
                reason: "SOS num_components is zero",
            });
        }
        if num_components > self.num_components {
            return Err(Error::InvalidJpegData {
                reason: "SOS num_components exceeds frame components",
            });
        }
        if num_components > MAX_COMPONENTS as u8 {
            return Err(Error::InvalidJpegData {
                reason: "SOS num_components too large",
            });
        }

        let mut scan_components = Vec::with_capacity(num_components as usize);

        for _ in 0..num_components {
            let component_id = self.read_u8()?;
            let tables = self.read_u8()?;
            let dc_table = tables >> 4;
            let ac_table = tables & 0x0F;

            // Validate Huffman table indexes
            if dc_table as usize >= MAX_HUFFMAN_TABLES {
                return Err(Error::InvalidJpegData {
                    reason: "SOS DC Huffman table index out of range",
                });
            }
            if ac_table as usize >= MAX_HUFFMAN_TABLES {
                return Err(Error::InvalidJpegData {
                    reason: "SOS AC Huffman table index out of range",
                });
            }

            // Find component index
            let comp_idx = self.components[..self.num_components as usize]
                .iter()
                .position(|c| c.id == component_id)
                .ok_or(Error::InvalidJpegData {
                    reason: "unknown component in scan",
                })?;

            scan_components.push((comp_idx, dc_table, ac_table));
        }

        let ss = self.read_u8()?; // Spectral selection start
        let se = self.read_u8()?; // Spectral selection end
        let ah_al = self.read_u8()?;
        let ah = ah_al >> 4;
        let al = ah_al & 0x0F;

        // Validate spectral selection (must be 0-63)
        if ss > 63 {
            return Err(Error::InvalidJpegData {
                reason: "SOS Ss (spectral start) out of range",
            });
        }
        if se > 63 {
            return Err(Error::InvalidJpegData {
                reason: "SOS Se (spectral end) out of range",
            });
        }

        // Decode entropy-coded segment based on mode
        if self.mode == JpegMode::Progressive {
            self.decode_progressive_scan(&scan_components, ss, se, ah, al)?;
        } else {
            self.decode_scan(&scan_components)?;
        }

        Ok(())
    }

    fn decode_scan(&mut self, scan_components: &[(usize, u8, u8)]) -> Result<()> {
        // Calculate max sampling factors to determine MCU structure
        let mut max_h_samp = 1u8;
        let mut max_v_samp = 1u8;
        for i in 0..self.num_components as usize {
            max_h_samp = max_h_samp.max(self.components[i].h_samp_factor);
            max_v_samp = max_v_samp.max(self.components[i].v_samp_factor);
        }

        // MCU dimensions in pixels
        let mcu_width = (max_h_samp as usize) * 8;
        let mcu_height = (max_v_samp as usize) * 8;

        // Number of MCUs
        let mcu_cols = (self.width as usize + mcu_width - 1) / mcu_width;
        let mcu_rows = (self.height as usize + mcu_height - 1) / mcu_height;

        // Initialize coefficient storage - size depends on component's sampling factor
        if self.coeffs.is_empty() {
            for i in 0..self.num_components as usize {
                let h_samp = self.components[i].h_samp_factor as usize;
                let v_samp = self.components[i].v_samp_factor as usize;
                let comp_blocks_h = checked_size_2d(mcu_cols, h_samp)?;
                let comp_blocks_v = checked_size_2d(mcu_rows, v_samp)?;
                let num_blocks = checked_size_2d(comp_blocks_h, comp_blocks_v)?;
                self.coeffs.push(try_alloc_dct_blocks(
                    num_blocks,
                    "allocating DCT coefficients",
                )?);
            }
        }

        // Set up entropy decoder
        let scan_data = &self.data[self.position..];
        let mut decoder = EntropyDecoder::new(scan_data);

        for (_comp_idx, dc_table, ac_table) in scan_components {
            let dc_idx = (*dc_table as usize).min(MAX_HUFFMAN_TABLES - 1);
            let ac_idx = (*ac_table as usize).min(MAX_HUFFMAN_TABLES - 1);
            if let Some(table) = &self.dc_tables[dc_idx] {
                decoder.set_dc_table(dc_idx, table.clone());
            }
            if let Some(table) = &self.ac_tables[ac_idx] {
                decoder.set_ac_table(ac_idx, table.clone());
            }
        }

        // Decode MCUs with proper interleaving
        for mcu_y in 0..mcu_rows {
            for mcu_x in 0..mcu_cols {
                // For each component in the scan
                for (comp_idx, dc_table, ac_table) in scan_components {
                    let h_samp = self.components[*comp_idx].h_samp_factor as usize;
                    let v_samp = self.components[*comp_idx].v_samp_factor as usize;
                    let comp_blocks_h = mcu_cols * h_samp;

                    // Decode all blocks for this component in this MCU
                    for v in 0..v_samp {
                        for h in 0..h_samp {
                            let block_x = mcu_x * h_samp + h;
                            let block_y = mcu_y * v_samp + v;
                            let block_idx = block_y * comp_blocks_h + block_x;

                            let coeffs = decoder.decode_block(
                                *comp_idx,
                                *dc_table as usize,
                                *ac_table as usize,
                            )?;
                            self.coeffs[*comp_idx][block_idx] = coeffs;
                        }
                    }
                }
            }
        }

        self.position += decoder.position();
        Ok(())
    }

    fn decode_progressive_scan(
        &mut self,
        scan_components: &[(usize, u8, u8)],
        ss: u8,
        se: u8,
        ah: u8,
        al: u8,
    ) -> Result<()> {
        // Calculate max sampling factors to determine MCU structure
        let mut max_h_samp = 1u8;
        let mut max_v_samp = 1u8;
        for i in 0..self.num_components as usize {
            max_h_samp = max_h_samp.max(self.components[i].h_samp_factor);
            max_v_samp = max_v_samp.max(self.components[i].v_samp_factor);
        }

        // MCU dimensions in pixels
        let mcu_width = (max_h_samp as usize) * 8;
        let mcu_height = (max_v_samp as usize) * 8;

        // Number of MCUs
        let mcu_cols = (self.width as usize + mcu_width - 1) / mcu_width;
        let mcu_rows = (self.height as usize + mcu_height - 1) / mcu_height;

        // Initialize coefficient storage if not already done
        if self.coeffs.is_empty() {
            for i in 0..self.num_components as usize {
                let h_samp = self.components[i].h_samp_factor as usize;
                let v_samp = self.components[i].v_samp_factor as usize;
                let comp_blocks_h = checked_size_2d(mcu_cols, h_samp)?;
                let comp_blocks_v = checked_size_2d(mcu_rows, v_samp)?;
                let num_blocks = checked_size_2d(comp_blocks_h, comp_blocks_v)?;
                self.coeffs.push(try_alloc_dct_blocks(
                    num_blocks,
                    "allocating DCT coefficients",
                )?);
            }
        }

        // Set up entropy decoder
        let scan_data = &self.data[self.position..];
        let mut decoder = EntropyDecoder::new(scan_data);

        for (_comp_idx, dc_table, ac_table) in scan_components {
            let dc_idx = (*dc_table as usize).min(MAX_HUFFMAN_TABLES - 1);
            let ac_idx = (*ac_table as usize).min(MAX_HUFFMAN_TABLES - 1);
            if let Some(table) = &self.dc_tables[dc_idx] {
                decoder.set_dc_table(dc_idx, table.clone());
            }
            if let Some(table) = &self.ac_tables[ac_idx] {
                decoder.set_ac_table(ac_idx, table.clone());
            }
        }

        // Determine scan type
        let is_dc_scan = ss == 0 && se == 0;
        let is_first_scan = ah == 0;

        // EOB run tracking for AC scans
        let mut eob_run = 0u16;

        if is_dc_scan {
            // DC scan (interleaved or single component)
            for mcu_y in 0..mcu_rows {
                for mcu_x in 0..mcu_cols {
                    for (comp_idx, dc_table, _ac_table) in scan_components {
                        let h_samp = self.components[*comp_idx].h_samp_factor as usize;
                        let v_samp = self.components[*comp_idx].v_samp_factor as usize;
                        let comp_blocks_h = mcu_cols * h_samp;

                        for v in 0..v_samp {
                            for h in 0..h_samp {
                                let block_x = mcu_x * h_samp + h;
                                let block_y = mcu_y * v_samp + v;
                                let block_idx = block_y * comp_blocks_h + block_x;

                                if is_first_scan {
                                    // DC first scan
                                    let dc = decoder.decode_dc_first(
                                        *comp_idx,
                                        *dc_table as usize,
                                        al,
                                    )?;
                                    self.coeffs[*comp_idx][block_idx][0] = dc;
                                } else {
                                    // DC refinement scan
                                    let bit = decoder.decode_dc_refine(al)?;
                                    self.coeffs[*comp_idx][block_idx][0] |= bit;
                                }
                            }
                        }
                    }
                }
            }
        } else {
            // AC scan (single component only for progressive)
            // Progressive AC scans can only have one component
            if scan_components.len() != 1 {
                return Err(Error::InvalidJpegData {
                    reason: "progressive AC scan must have single component",
                });
            }

            let (comp_idx, _dc_table, ac_table) = scan_components[0];
            let h_samp = self.components[comp_idx].h_samp_factor as usize;
            let v_samp = self.components[comp_idx].v_samp_factor as usize;
            let comp_blocks_h = mcu_cols * h_samp;

            for mcu_y in 0..mcu_rows {
                for mcu_x in 0..mcu_cols {
                    for v in 0..v_samp {
                        for h in 0..h_samp {
                            let block_x = mcu_x * h_samp + h;
                            let block_y = mcu_y * v_samp + v;
                            let block_idx = block_y * comp_blocks_h + block_x;

                            if is_first_scan {
                                // AC first scan
                                decoder.decode_ac_first(
                                    &mut self.coeffs[comp_idx][block_idx],
                                    ac_table as usize,
                                    ss,
                                    se,
                                    al,
                                    &mut eob_run,
                                )?;
                            } else {
                                // AC refinement scan
                                decoder.decode_ac_refine(
                                    &mut self.coeffs[comp_idx][block_idx],
                                    ac_table as usize,
                                    ss,
                                    se,
                                    al,
                                    &mut eob_run,
                                )?;
                            }
                        }
                    }
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

        // Calculate max sampling factors
        let mut max_h_samp = 1u8;
        let mut max_v_samp = 1u8;
        for i in 0..self.num_components as usize {
            max_h_samp = max_h_samp.max(self.components[i].h_samp_factor);
            max_v_samp = max_v_samp.max(self.components[i].v_samp_factor);
        }

        // MCU dimensions
        let mcu_width = (max_h_samp as usize) * 8;
        let mcu_height = (max_v_samp as usize) * 8;
        let mcu_cols = (width + mcu_width - 1) / mcu_width;
        let mcu_rows = (height + mcu_height - 1) / mcu_height;

        // Dequantize and IDCT all blocks, then upsample if needed
        let mut planes: Vec<Vec<u8>> = Vec::new();

        for comp_idx in 0..self.num_components as usize {
            let quant_idx = self.components[comp_idx].quant_table_idx as usize;
            let quant = self.quant_tables[quant_idx]
                .as_ref()
                .ok_or(Error::InternalError {
                    reason: "missing quantization table",
                })?;

            let h_samp = self.components[comp_idx].h_samp_factor as usize;
            let v_samp = self.components[comp_idx].v_samp_factor as usize;
            let comp_blocks_h = mcu_cols * h_samp;
            let comp_blocks_v = mcu_rows * v_samp;

            // Component plane dimensions (may be smaller than full image for subsampled)
            let comp_width = checked_size_2d(comp_blocks_h, 8)?;
            let comp_height = checked_size_2d(comp_blocks_v, 8)?;

            let comp_plane_size = checked_size_2d(comp_width, comp_height)?;
            let mut comp_plane = try_alloc_zeroed(comp_plane_size, "allocating component plane")?;

            for by in 0..comp_blocks_v {
                for bx in 0..comp_blocks_h {
                    let block_idx = by * comp_blocks_h + bx;
                    if block_idx >= self.coeffs[comp_idx].len() {
                        continue;
                    }
                    let coeffs = &self.coeffs[comp_idx][block_idx];

                    // Convert to natural order and dequantize
                    let mut natural_coeffs = [0i16; DCT_BLOCK_SIZE];
                    for (i, &zi) in JPEG_NATURAL_ORDER[..DCT_BLOCK_SIZE].iter().enumerate() {
                        natural_coeffs[zi as usize] = coeffs[i];
                    }

                    let dequant = dequantize_block(&natural_coeffs, quant);
                    let pixels = inverse_dct_8x8(&dequant);

                    // Copy to component plane with level shift
                    for y in 0..DCT_SIZE {
                        for x in 0..DCT_SIZE {
                            let px = bx * DCT_SIZE + x;
                            let py = by * DCT_SIZE + y;
                            if px < comp_width && py < comp_height {
                                let val = (pixels[y * DCT_SIZE + x] + 128.0).round();
                                comp_plane[py * comp_width + px] = val.clamp(0.0, 255.0) as u8;
                            }
                        }
                    }
                }
            }

            // Upsample if this component has lower sampling than max
            let output_size = checked_size_2d(width, height)?;
            let plane = if h_samp < max_h_samp as usize || v_samp < max_v_samp as usize {
                let scale_x = max_h_samp as usize / h_samp;
                let scale_y = max_v_samp as usize / v_samp;
                let mut upsampled = try_alloc_zeroed(output_size, "allocating upsampled plane")?;
                for py in 0..height {
                    for px in 0..width {
                        let sx = (px / scale_x).min(comp_width - 1);
                        let sy = (py / scale_y).min(comp_height - 1);
                        upsampled[py * width + px] = comp_plane[sy * comp_width + sx];
                    }
                }
                upsampled
            } else {
                // Full resolution - just clip to image dimensions
                let mut plane = try_alloc_zeroed(output_size, "allocating output plane")?;
                for py in 0..height {
                    for px in 0..width {
                        plane[py * width + px] = comp_plane[py * comp_width + px];
                    }
                }
                plane
            };

            planes.push(plane);
        }

        // Convert to output format
        match (self.num_components, format) {
            (1, PixelFormat::Gray) => Ok(planes[0].clone()),
            (1, PixelFormat::Rgb) => {
                let rgb_size =
                    checked_size_2d(width, height).and_then(|s| checked_size_2d(s, 3))?;
                let mut rgb = try_alloc_zeroed(rgb_size, "allocating RGB output")?;
                for (i, &y) in planes[0].iter().enumerate() {
                    rgb[i * 3] = y;
                    rgb[i * 3 + 1] = y;
                    rgb[i * 3 + 2] = y;
                }
                Ok(rgb)
            }
            (3, PixelFormat::Rgb) => {
                color::ycbcr_planes_to_rgb(&planes[0], &planes[1], &planes[2], width, height)
            }
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
