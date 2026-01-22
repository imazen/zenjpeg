//! JPEG parser implementation.
//!
//! Internal parser for reading and decoding JPEG data.

use super::idct::inverse_dct_8x8;
use super::idct_int::{idct_int_auto, idct_int_tiered};
use super::upsample::upsample_fancy;
use super::{JpegInfo, ScanInfo};
use crate::color::icc::{extract_icc_profile, is_xyb_profile};
use crate::color::{
    gray_f32_to_gray_f32, gray_f32_to_gray_u8, gray_f32_to_rgb_f32, gray_f32_to_rgb_u8,
    ycbcr_planes_f32_to_rgb_f32, ycbcr_planes_f32_to_rgb_u8, ycbcr_planes_i16_to_rgb_u8,
};
use crate::entropy::EntropyDecoder;
use crate::error::{Error, Result, ScanRead};
use crate::foundation::alloc::{
    checked_size_2d, try_alloc_dct_blocks, try_alloc_maybeuninit, validate_dimensions,
};
use crate::foundation::consts::{
    DCT_BLOCK_SIZE, DCT_SIZE, JPEG_NATURAL_ORDER, MARKER_APP0, MARKER_COM, MARKER_DHT, MARKER_DQT,
    MARKER_DRI, MARKER_EOI, MARKER_SOF0, MARKER_SOF1, MARKER_SOF2, MARKER_SOI, MARKER_SOS,
    MAX_COMPONENTS, MAX_HUFFMAN_TABLES, MAX_QUANT_TABLES,
};
use crate::huffman::HuffmanDecodeTable;
use crate::quant::{
    dequantize_block, dequantize_block_i32, dequantize_block_with_bias, dequantize_unzigzag_i32,
    dequantize_unzigzag_i32_into, DequantBiasStats,
};
use crate::types::{ColorSpace, Component, Dimensions, JpegMode, PixelFormat};

/// Pre-computed component info for decoding efficiency.
///
/// Computed once per decode, reused across multiple methods.
struct CompInfo {
    quant_idx: usize,
    h_samp: usize,
    v_samp: usize,
    comp_blocks_h: usize,
    comp_blocks_v: usize,
    /// Component width in pixels (comp_blocks_h * 8)
    comp_width: usize,
    /// Component height in pixels (comp_blocks_v * 8)
    comp_height: usize,
    /// True if this component has full resolution (no subsampling)
    is_full_res: bool,
}

/// Internal JPEG parser state.
pub(super) struct JpegParser<'a> {
    data: &'a [u8],
    position: usize,

    // Frame info
    pub(super) width: u32,
    pub(super) height: u32,
    precision: u8,
    pub(super) num_components: u8,
    pub(super) mode: JpegMode,

    // Component info
    pub(super) components: [Component; MAX_COMPONENTS],

    // Tables
    pub(super) quant_tables: [Option<[u16; DCT_BLOCK_SIZE]>; MAX_QUANT_TABLES],
    pub(super) dc_tables: [Option<HuffmanDecodeTable>; MAX_HUFFMAN_TABLES],
    pub(super) ac_tables: [Option<HuffmanDecodeTable>; MAX_HUFFMAN_TABLES],

    // Restart
    pub(super) restart_interval: u16,

    // Decoded coefficient data (used for progressive and non-streaming baseline)
    coeffs: Vec<Vec<[i16; DCT_BLOCK_SIZE]>>, // Per component
    coeff_counts: Vec<Vec<u8>>,              // Coefficient count per block (for tiered IDCT)

    // Streaming decode result (used for baseline 4:4:4 JPEGs)
    streaming_rgb: Option<Vec<u8>>,
    // Whether to prefer streaming decode (set false for f32 output which needs coefficients)
    pub(super) prefer_streaming: bool,

    // ICC profile (extracted from raw data, not during parsing)
    pub(super) icc_profile: Option<Vec<u8>>,

    // Security limits
    max_pixels: u64,
}

impl<'a> JpegParser<'a> {
    pub(super) fn new(data: &'a [u8], max_pixels: u64) -> Result<Self> {
        // Check for SOI
        if data.len() < 2 || data[0] != 0xFF || data[1] != MARKER_SOI {
            return Err(Error::invalid_jpeg_data("missing SOI marker"));
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
            coeff_counts: Vec::new(),
            streaming_rgb: None,
            prefer_streaming: true, // Default to streaming for RGB decode
            icc_profile,
            max_pixels,
        })
    }

    /// Build component info for all components.
    ///
    /// `num_comps` allows overriding for XYB which always uses 3 components.
    fn build_comp_infos(
        &self,
        mcu_cols: usize,
        mcu_rows: usize,
        max_h_samp: usize,
        max_v_samp: usize,
        num_comps: usize,
    ) -> Result<Vec<CompInfo>> {
        let mut comp_infos = Vec::with_capacity(num_comps);
        for comp_idx in 0..num_comps {
            let h_samp = self.components[comp_idx].h_samp_factor as usize;
            let v_samp = self.components[comp_idx].v_samp_factor as usize;
            let comp_blocks_h = mcu_cols * h_samp;
            let comp_blocks_v = mcu_rows * v_samp;
            let comp_width = checked_size_2d(comp_blocks_h, 8)?;
            let comp_height = checked_size_2d(comp_blocks_v, 8)?;
            comp_infos.push(CompInfo {
                quant_idx: self.components[comp_idx].quant_table_idx as usize,
                h_samp,
                v_samp,
                comp_blocks_h,
                comp_blocks_v,
                comp_width,
                comp_height,
                is_full_res: h_samp == max_h_samp && v_samp == max_v_samp,
            });
        }
        Ok(comp_infos)
    }

    fn read_u8(&mut self) -> Result<u8> {
        if self.position >= self.data.len() {
            return Err(Error::truncated_data("reading marker data"));
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
            // Skip until we find 0xFF
            let byte = self.read_u8()?;
            if byte != 0xFF {
                continue;
            }

            // Skip fill bytes (consecutive 0xFF)
            loop {
                let marker = self.read_u8()?;
                if marker == 0xFF {
                    // Fill byte, keep looking
                    continue;
                }
                if marker == 0x00 {
                    // Byte stuffing (0xFF 0x00 = literal 0xFF in data)
                    // This shouldn't happen in marker parsing, but skip it
                    break;
                }
                // Found a real marker
                return Ok(marker);
            }
        }
    }

    pub(super) fn read_header(&mut self) -> Result<()> {
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
                    return Err(Error::invalid_jpeg_data(
                        "unexpected EOI before frame header",
                    ));
                }
                _ => self.skip_segment()?,
            }
        }
    }

    /// Finds the SOS marker and extracts scan info without decoding.
    /// Used by scanline reader to get table mapping and data start position.
    pub(super) fn find_scan_info(&mut self) -> Result<ScanInfo> {
        // Continue from current position to find SOS
        loop {
            let marker = self.read_marker()?;

            match marker {
                MARKER_SOS => {
                    let _length = self.read_u16()?;
                    let num_components = self.read_u8()?;

                    if num_components != 3 {
                        return Err(Error::unsupported_feature(
                            "scanline reader requires 3 components in scan",
                        ));
                    }

                    let mut table_mapping = [(0usize, 0usize); 3];

                    for _i in 0..num_components as usize {
                        let component_id = self.read_u8()?;
                        let tables = self.read_u8()?;
                        let dc_table = (tables >> 4) as usize;
                        let ac_table = (tables & 0x0F) as usize;

                        // Find component index
                        let comp_idx = self.components[..self.num_components as usize]
                            .iter()
                            .position(|c| c.id == component_id)
                            .ok_or(Error::invalid_jpeg_data("unknown component in scan"))?;

                        table_mapping[comp_idx] = (dc_table, ac_table);
                    }

                    // Skip spectral selection bytes (Ss, Se, Ah/Al)
                    let _ss = self.read_u8()?;
                    let _se = self.read_u8()?;
                    let _ah_al = self.read_u8()?;

                    return Ok(ScanInfo {
                        table_mapping,
                        data_start: self.position,
                    });
                }
                MARKER_DQT => self.parse_quant_table()?,
                MARKER_DHT => self.parse_huffman_table()?,
                MARKER_DRI => self.parse_restart_interval()?,
                MARKER_APP0..=0xEF | MARKER_COM => self.skip_segment()?,
                MARKER_EOI => {
                    return Err(Error::invalid_jpeg_data("unexpected EOI before SOS"));
                }
                _ => self.skip_segment()?,
            }
        }
    }

    fn parse_frame_header(&mut self) -> Result<()> {
        let length = self.read_u16()?;
        if length < 8 {
            return Err(Error::invalid_jpeg_data("frame header too short"));
        }

        self.precision = self.read_u8()?;
        // Validate precision: must be 8 for baseline JPEG, 8 or 12 for extended
        if self.precision != 8 && self.precision != 12 {
            return Err(Error::invalid_jpeg_data(
                "invalid data precision (must be 8 or 12)",
            ));
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
            return Err(Error::invalid_jpeg_data("number of components is zero"));
        }
        if self.num_components > MAX_COMPONENTS as u8 {
            return Err(Error::unsupported_feature("more than 4 components"));
        }

        // Validate marker length matches expected size
        let expected_length = 8 + 3 * self.num_components as u16;
        if length != expected_length {
            return Err(Error::invalid_jpeg_data("SOF marker length mismatch"));
        }

        for i in 0..self.num_components as usize {
            self.components[i].id = self.read_u8()?;
            let sampling = self.read_u8()?;
            let h_samp = sampling >> 4;
            let v_samp = sampling & 0x0F;

            // Validate sampling factors are non-zero and <= 4
            if h_samp == 0 || v_samp == 0 {
                return Err(Error::invalid_jpeg_data("sampling factor is zero"));
            }
            if h_samp > 4 || v_samp > 4 {
                return Err(Error::invalid_jpeg_data(
                    "sampling factor exceeds maximum (4)",
                ));
            }

            self.components[i].h_samp_factor = h_samp;
            self.components[i].v_samp_factor = v_samp;

            let quant_idx = self.read_u8()?;
            // Validate quant table index
            if quant_idx as usize >= MAX_QUANT_TABLES {
                return Err(Error::invalid_jpeg_data(
                    "quantization table index out of range",
                ));
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
                return Err(Error::invalid_quant_table(
                    table_idx as u8,
                    "invalid precision (must be 0 or 1)",
                ));
            }

            if table_idx >= MAX_QUANT_TABLES {
                return Err(Error::invalid_quant_table(
                    table_idx as u8,
                    "table index out of range",
                ));
            }

            // Read values in zigzag order (as stored in JPEG)
            let mut zigzag_values = [0u16; DCT_BLOCK_SIZE];

            if precision == 0 {
                // 8-bit values
                for i in 0..DCT_BLOCK_SIZE {
                    let val = self.read_u8()? as u16;
                    if val == 0 {
                        return Err(Error::invalid_quant_table(
                            table_idx as u8,
                            "quantization value is zero",
                        ));
                    }
                    zigzag_values[i] = val;
                }
                length -= 65;
            } else {
                // 16-bit values
                for i in 0..DCT_BLOCK_SIZE {
                    let val = self.read_u16()?;
                    if val == 0 {
                        return Err(Error::invalid_quant_table(
                            table_idx as u8,
                            "quantization value is zero",
                        ));
                    }
                    zigzag_values[i] = val;
                }
                length -= 129;
            }

            // Validate DQT marker length consistency
            if length < 0 {
                return Err(Error::invalid_jpeg_data("DQT marker length mismatch"));
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
                return Err(Error::invalid_huffman_table(
                    table_idx as u8,
                    "invalid table class (must be 0 or 1)",
                ));
            }

            if table_idx >= MAX_HUFFMAN_TABLES {
                return Err(Error::invalid_huffman_table(
                    table_idx as u8,
                    "table index out of range",
                ));
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
                return Err(Error::invalid_jpeg_data("DHT marker length mismatch"));
            }

            if table_class == 0 {
                // DC table - use standard lookup
                let table = HuffmanDecodeTable::from_bits_values(&bits, &values)?;
                self.dc_tables[table_idx] = Some(table);
            } else {
                // AC table - use fast AC lookup for combined decode + sign extend
                let table = HuffmanDecodeTable::from_bits_values_ac(&bits, &values)?;
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
            return Err(Error::invalid_jpeg_data("segment length too short"));
        }
        self.position += length - 2;
        Ok(())
    }

    pub(super) fn decode(&mut self) -> Result<()> {
        // First read header
        self.position = 2; // Skip SOI
        self.read_header()?;

        // Continue parsing until we hit EOI
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
            return Err(Error::invalid_jpeg_data("SOS num_components is zero"));
        }
        if num_components > self.num_components {
            return Err(Error::invalid_jpeg_data(
                "SOS num_components exceeds frame components",
            ));
        }
        if num_components > MAX_COMPONENTS as u8 {
            return Err(Error::invalid_jpeg_data("SOS num_components too large"));
        }

        let mut scan_components = Vec::with_capacity(num_components as usize);

        for _ in 0..num_components {
            let component_id = self.read_u8()?;
            let tables = self.read_u8()?;
            let dc_table = tables >> 4;
            let ac_table = tables & 0x0F;

            // Validate Huffman table indexes
            if dc_table as usize >= MAX_HUFFMAN_TABLES {
                return Err(Error::invalid_jpeg_data(
                    "SOS DC Huffman table index out of range",
                ));
            }
            if ac_table as usize >= MAX_HUFFMAN_TABLES {
                return Err(Error::invalid_jpeg_data(
                    "SOS AC Huffman table index out of range",
                ));
            }

            // Find component index
            let comp_idx = self.components[..self.num_components as usize]
                .iter()
                .position(|c| c.id == component_id)
                .ok_or(Error::invalid_jpeg_data("unknown component in scan"))?;

            scan_components.push((comp_idx, dc_table, ac_table));
        }

        let ss = self.read_u8()?; // Spectral selection start
        let se = self.read_u8()?; // Spectral selection end
        let ah_al = self.read_u8()?;
        let ah = ah_al >> 4;
        let al = ah_al & 0x0F;

        // Validate spectral selection (must be 0-63)
        if ss > 63 {
            return Err(Error::invalid_jpeg_data(
                "SOS Ss (spectral start) out of range",
            ));
        }
        if se > 63 {
            return Err(Error::invalid_jpeg_data(
                "SOS Se (spectral end) out of range",
            ));
        }

        // Decode entropy-coded segment based on mode
        if self.mode == JpegMode::Progressive {
            self.decode_progressive_scan(&scan_components, ss, se, ah, al)?;
        } else if self.prefer_streaming && self.can_use_streaming() && self.streaming_rgb.is_none()
        {
            // Use streaming decode for baseline 4:4:4 - fuses decode + IDCT + color
            let rgb = self.decode_baseline_streaming_rgb(&scan_components)?;
            self.streaming_rgb = Some(rgb);
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
                // Allocate parallel storage for coefficient counts (tiered IDCT)
                self.coeff_counts.push(vec![64u8; num_blocks]);
            }
        }

        // Set up entropy decoder
        let scan_data = &self.data[self.position..];
        let mut decoder = EntropyDecoder::new(scan_data);

        for (_comp_idx, dc_table, ac_table) in scan_components {
            let dc_idx = (*dc_table as usize).min(MAX_HUFFMAN_TABLES - 1);
            let ac_idx = (*ac_table as usize).min(MAX_HUFFMAN_TABLES - 1);

            // Use explicit table if provided, otherwise use standard JPEG tables.
            // MJPEG files often omit DHT markers and expect standard tables.
            // Tables are borrowed, not cloned (~1.5KB savings per table).
            let dc_table_ref: &HuffmanDecodeTable = match &self.dc_tables[dc_idx] {
                Some(table) => table,
                None => {
                    if dc_idx == 0 {
                        HuffmanDecodeTable::std_dc_luminance()
                    } else {
                        HuffmanDecodeTable::std_dc_chrominance()
                    }
                }
            };
            decoder.set_dc_table(dc_idx, dc_table_ref);

            let ac_table_ref: &HuffmanDecodeTable = match &self.ac_tables[ac_idx] {
                Some(table) => table,
                None => {
                    if ac_idx == 0 {
                        HuffmanDecodeTable::std_ac_luminance()
                    } else {
                        HuffmanDecodeTable::std_ac_chrominance()
                    }
                }
            };
            decoder.set_ac_table(ac_idx, ac_table_ref);
        }

        // Decode MCUs with proper interleaving
        let mut mcu_count = 0u32;
        let restart_interval = self.restart_interval as u32;
        let mut next_restart_num = 0u8;

        for mcu_y in 0..mcu_rows {
            for mcu_x in 0..mcu_cols {
                // Check for restart marker
                if restart_interval > 0 && mcu_count > 0 && mcu_count % restart_interval == 0 {
                    // Align to byte boundary (discard padding bits)
                    decoder.align_to_byte();
                    // Read and verify restart marker
                    decoder.read_restart_marker(next_restart_num)?;
                    // Update expected marker number (cycles 0-7)
                    next_restart_num = (next_restart_num + 1) & 7;
                    // Reset DC predictors
                    decoder.reset_dc();
                }

                // For each component in the scan
                for (comp_idx, dc_table, ac_table) in scan_components {
                    let h_samp = self.components[*comp_idx].h_samp_factor as usize;
                    let v_samp = self.components[*comp_idx].v_samp_factor as usize;
                    let comp_blocks_h = mcu_cols * h_samp;

                    // Calculate actual content dimensions for this component
                    // Some encoders omit padding blocks beyond the image bounds
                    let comp_width = (self.width as usize * h_samp + max_h_samp as usize - 1)
                        / max_h_samp as usize;
                    let comp_height = (self.height as usize * v_samp + max_v_samp as usize - 1)
                        / max_v_samp as usize;
                    let actual_blocks_h = (comp_width + 7) / 8;
                    let actual_blocks_v = (comp_height + 7) / 8;

                    // For single-component images with unusual sampling (grayscale with h/v > 1),
                    // some encoders omit padding blocks entirely. Detect this case.
                    let is_single_component_oversample =
                        scan_components.len() == 1 && (h_samp > 1 || v_samp > 1);

                    // Decode all blocks for this component in this MCU
                    for v in 0..v_samp {
                        for h in 0..h_samp {
                            let block_x = mcu_x * h_samp + h;
                            let block_y = mcu_y * v_samp + v;
                            let block_idx = block_y * comp_blocks_h + block_x;

                            // Check if this block is beyond actual image bounds (padding)
                            let is_padding =
                                block_x >= actual_blocks_h || block_y >= actual_blocks_v;

                            if is_padding && is_single_component_oversample {
                                // Single-component with oversampling: skip padding blocks
                                // These encoders typically omit them
                                self.coeffs[*comp_idx][block_idx] = [0i16; 64];
                                self.coeff_counts[*comp_idx][block_idx] = 1; // DC-only (zeros)
                                continue;
                            }

                            if is_padding {
                                // For padding blocks in multi-component images, use speculative decoding
                                // Most encoders include them, but some might not
                                let saved_state = decoder.save_state();
                                match decoder.decode_block_with_count(
                                    *comp_idx,
                                    *dc_table as usize,
                                    *ac_table as usize,
                                ) {
                                    Ok(ScanRead::Value((coeffs, count))) => {
                                        // Encoder included padding block
                                        self.coeffs[*comp_idx][block_idx] = coeffs;
                                        self.coeff_counts[*comp_idx][block_idx] = count;
                                    }
                                    Ok(ScanRead::EndOfScan | ScanRead::Truncated) => {
                                        // Encoder omitted padding block - restore state and fill zeros
                                        decoder.restore_state(saved_state);
                                        self.coeffs[*comp_idx][block_idx] = [0i16; 64];
                                        self.coeff_counts[*comp_idx][block_idx] = 1;
                                    }
                                    Err(_e) => {
                                        // Other error - also restore and skip
                                        decoder.restore_state(saved_state);
                                        self.coeffs[*comp_idx][block_idx] = [0i16; 64];
                                        self.coeff_counts[*comp_idx][block_idx] = 1;
                                        // Log but don't fail on padding block errors
                                        #[cfg(debug_assertions)]
                                        eprintln!(
                                            "DEBUG: Padding block ({},{}) error: {:?}",
                                            block_x, block_y, _e
                                        );
                                    }
                                }
                            } else {
                                let (coeffs, count) = match decoder.decode_block_with_count(
                                    *comp_idx,
                                    *dc_table as usize,
                                    *ac_table as usize,
                                )? {
                                    ScanRead::Value(v) => v,
                                    // EndOfScan/Truncated mid-decode is unusual but not fatal - fill with zeros
                                    ScanRead::EndOfScan | ScanRead::Truncated => {
                                        self.coeffs[*comp_idx][block_idx] = [0i16; 64];
                                        self.coeff_counts[*comp_idx][block_idx] = 1;
                                        continue;
                                    }
                                };
                                self.coeffs[*comp_idx][block_idx] = coeffs;
                                self.coeff_counts[*comp_idx][block_idx] = count;
                            }
                        }
                    }
                }

                mcu_count += 1;
            }
        }

        self.position += decoder.position();
        Ok(())
    }

    /// Check if streaming decode can be used.
    /// Streaming is only possible for baseline 4:4:4 YCbCr images.
    fn can_use_streaming(&self) -> bool {
        // Must be baseline (not progressive)
        if self.mode != JpegMode::Baseline {
            return false;
        }
        // Must have 3 components (YCbCr)
        if self.num_components != 3 {
            return false;
        }
        // Must be 4:4:4 (all components have same sampling factors)
        let h0 = self.components[0].h_samp_factor;
        let v0 = self.components[0].v_samp_factor;
        for i in 1..3 {
            if self.components[i].h_samp_factor != h0 || self.components[i].v_samp_factor != v0 {
                return false;
            }
        }
        // Must have 1x1 sampling (no subsampling)
        if h0 != 1 || v0 != 1 {
            return false;
        }
        true
    }

    /// Streaming decode for baseline 4:4:4 YCbCr images.
    /// Combines Huffman decode + dequantize + IDCT + color convert in one pass.
    /// No coefficient storage - processes MCU row by row directly to RGB output.
    fn decode_baseline_streaming_rgb(
        &mut self,
        scan_components: &[(usize, u8, u8)],
    ) -> Result<Vec<u8>> {
        let width = self.width as usize;
        let height = self.height as usize;

        // For 4:4:4, MCU = 8x8 pixels (single block per component)
        let mcu_cols = (width + 7) / 8;
        let mcu_rows = (height + 7) / 8;
        let strip_width = mcu_cols * 8;

        // Get quantization tables
        let quant_y = self.quant_tables[self.components[0].quant_table_idx as usize]
            .as_ref()
            .ok_or(Error::internal("missing Y quantization table"))?;
        let quant_cb = self.quant_tables[self.components[1].quant_table_idx as usize]
            .as_ref()
            .ok_or(Error::internal("missing Cb quantization table"))?;
        let quant_cr = self.quant_tables[self.components[2].quant_table_idx as usize]
            .as_ref()
            .ok_or(Error::internal("missing Cr quantization table"))?;

        // Set up entropy decoder
        let scan_data = &self.data[self.position..];
        let mut decoder = EntropyDecoder::new(scan_data);

        // Set up Huffman tables
        for (comp_idx, dc_table, ac_table) in scan_components {
            let dc_idx = (*dc_table as usize).min(MAX_HUFFMAN_TABLES - 1);
            let ac_idx = (*ac_table as usize).min(MAX_HUFFMAN_TABLES - 1);

            let dc_table_ref: &HuffmanDecodeTable = match &self.dc_tables[dc_idx] {
                Some(table) => table,
                None => {
                    if dc_idx == 0 {
                        HuffmanDecodeTable::std_dc_luminance()
                    } else {
                        HuffmanDecodeTable::std_dc_chrominance()
                    }
                }
            };
            decoder.set_dc_table(*comp_idx, dc_table_ref);

            let ac_table_ref: &HuffmanDecodeTable = match &self.ac_tables[ac_idx] {
                Some(table) => table,
                None => {
                    if ac_idx == 0 {
                        HuffmanDecodeTable::std_ac_luminance()
                    } else {
                        HuffmanDecodeTable::std_ac_chrominance()
                    }
                }
            };
            decoder.set_ac_table(*comp_idx, ac_table_ref);
        }

        // Allocate strip buffers for one MCU row (8 rows of pixels)
        // Note: All elements are written by IDCT before color conversion reads them
        let strip_size = strip_width * 8;
        let mut y_strip: Vec<i16> = try_alloc_maybeuninit(strip_size, "Y strip buffer")?;
        let mut cb_strip: Vec<i16> = try_alloc_maybeuninit(strip_size, "Cb strip buffer")?;
        let mut cr_strip: Vec<i16> = try_alloc_maybeuninit(strip_size, "Cr strip buffer")?;

        // Allocate output RGB buffer
        // Note: All pixels are written by color conversion before return
        let rgb_size = checked_size_2d(width, height).and_then(|s| checked_size_2d(s, 3))?;
        let mut rgb: Vec<u8> = try_alloc_maybeuninit(rgb_size, "RGB output buffer")?;

        let mut mcu_count = 0u32;
        let restart_interval = self.restart_interval as u32;
        let mut next_restart_num = 0u8;

        // Reusable dequantization buffer - avoids allocation per block
        let mut dequant_buf = [0i32; DCT_BLOCK_SIZE];

        // Process MCU row by row
        for mcu_y in 0..mcu_rows {
            // Decode one MCU row's worth of blocks
            for mcu_x in 0..mcu_cols {
                // Check for restart marker
                if restart_interval > 0 && mcu_count > 0 && mcu_count % restart_interval == 0 {
                    decoder.align_to_byte();
                    decoder.read_restart_marker(next_restart_num)?;
                    next_restart_num = (next_restart_num + 1) & 7;
                    decoder.reset_dc();
                }

                // Decode, dequantize, and IDCT each component's block directly to strip
                for (comp_idx, dc_table, ac_table) in scan_components {
                    let (coeffs, coeff_count) = match decoder.decode_block_with_count(
                        *comp_idx,
                        *dc_table as usize,
                        *ac_table as usize,
                    )? {
                        ScanRead::Value(v) => v,
                        ScanRead::EndOfScan | ScanRead::Truncated => continue, // End of scan mid-block, skip remaining
                    };

                    let quant = match *comp_idx {
                        0 => quant_y,
                        1 => quant_cb,
                        _ => quant_cr,
                    };
                    let strip = match *comp_idx {
                        0 => &mut y_strip,
                        1 => &mut cb_strip,
                        _ => &mut cr_strip,
                    };

                    // Fused dequantize + unzigzag into reusable buffer
                    dequantize_unzigzag_i32_into(&coeffs, quant, &mut dequant_buf);

                    // IDCT directly to strip buffer
                    let dst_offset = mcu_x * 8;
                    idct_int_tiered(
                        &mut dequant_buf,
                        &mut strip[dst_offset..],
                        strip_width,
                        coeff_count,
                    );
                }

                mcu_count += 1;
            }

            // Color convert this MCU row directly to RGB output
            let y_start = mcu_y * 8;
            let rows_this_mcu = 8.min(height.saturating_sub(y_start));
            let cols_this_mcu = width.min(strip_width);

            for row in 0..rows_this_mcu {
                let strip_offset = row * strip_width;
                let rgb_offset = (y_start + row) * width * 3;

                ycbcr_planes_i16_to_rgb_u8(
                    &y_strip[strip_offset..strip_offset + cols_this_mcu],
                    &cb_strip[strip_offset..strip_offset + cols_this_mcu],
                    &cr_strip[strip_offset..strip_offset + cols_this_mcu],
                    &mut rgb[rgb_offset..rgb_offset + cols_this_mcu * 3],
                );
            }
        }

        self.position += decoder.position();
        Ok(rgb)
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
                // For progressive, we don't know coeff counts until all scans are done
                // Default to 64 (full IDCT) - tiered IDCT is mainly for baseline
                self.coeff_counts.push(vec![64u8; num_blocks]);
            }
        }

        // Set up entropy decoder
        let scan_data = &self.data[self.position..];

        // Debug: print progressive scan info
        if std::env::var("DEBUG_PROGRESSIVE").is_ok() {
            let first_bytes: Vec<u8> = scan_data.iter().take(8).copied().collect();
            eprintln!(
                "DEBUG prog: ss={} se={} ah={} al={} comps={:?} pos={} data={:02x?}",
                ss, se, ah, al, scan_components, self.position, first_bytes
            );
        }

        let mut decoder = EntropyDecoder::new(scan_data);

        for (_comp_idx, dc_table, ac_table) in scan_components {
            let dc_idx = (*dc_table as usize).min(MAX_HUFFMAN_TABLES - 1);
            let ac_idx = (*ac_table as usize).min(MAX_HUFFMAN_TABLES - 1);

            // Use explicit table if provided, otherwise use standard JPEG tables.
            // MJPEG files often omit DHT markers and expect standard tables.
            // Tables are borrowed, not cloned (~1.5KB savings per table).
            let dc_table_ref: &HuffmanDecodeTable = match &self.dc_tables[dc_idx] {
                Some(table) => table,
                None => {
                    if dc_idx == 0 {
                        HuffmanDecodeTable::std_dc_luminance()
                    } else {
                        HuffmanDecodeTable::std_dc_chrominance()
                    }
                }
            };
            decoder.set_dc_table(dc_idx, dc_table_ref);

            let ac_table_ref: &HuffmanDecodeTable = match &self.ac_tables[ac_idx] {
                Some(table) => table,
                None => {
                    if ac_idx == 0 {
                        HuffmanDecodeTable::std_ac_luminance()
                    } else {
                        HuffmanDecodeTable::std_ac_chrominance()
                    }
                }
            };
            decoder.set_ac_table(ac_idx, ac_table_ref);
        }

        // Determine scan type
        let is_dc_scan = ss == 0 && se == 0;
        let is_first_scan = ah == 0;

        // EOB run tracking for AC scans
        let mut eob_run = 0u16;

        // Restart marker handling
        let mut mcu_count = 0u32;
        let restart_interval = self.restart_interval as u32;
        let mut next_restart_num = 0u8;

        if is_dc_scan {
            // DC scan - can be interleaved (multiple components) or non-interleaved (single component)
            // For non-interleaved scans, blocks are in raster order (like AC scans)
            // For interleaved scans, blocks follow MCU order

            if scan_components.len() == 1 {
                // Non-interleaved DC scan: blocks in raster order (like AC scans)
                let (comp_idx, dc_table, _ac_table) = scan_components[0];
                let h_samp = self.components[comp_idx].h_samp_factor as usize;
                let v_samp = self.components[comp_idx].v_samp_factor as usize;
                let comp_blocks_h = mcu_cols * h_samp;
                let comp_blocks_v = mcu_rows * v_samp;
                let total_blocks = comp_blocks_h * comp_blocks_v;

                for block_idx in 0..total_blocks {
                    // Check for restart marker
                    if restart_interval > 0 && mcu_count > 0 && mcu_count % restart_interval == 0 {
                        decoder.align_to_byte();
                        decoder.read_restart_marker(next_restart_num)?;
                        next_restart_num = (next_restart_num + 1) & 7;
                        decoder.reset_dc();
                    }

                    if is_first_scan {
                        match decoder.decode_dc_first(comp_idx, dc_table as usize, al)? {
                            ScanRead::Value(dc) => self.coeffs[comp_idx][block_idx][0] = dc,
                            ScanRead::EndOfScan | ScanRead::Truncated => {
                                // End of scan data - remaining blocks have DC=0
                                break;
                            }
                        }
                    } else {
                        match decoder.decode_dc_refine(al)? {
                            ScanRead::Value(bit) => self.coeffs[comp_idx][block_idx][0] |= bit,
                            ScanRead::EndOfScan | ScanRead::Truncated => {
                                // End of scan data - remaining blocks unchanged
                                break;
                            }
                        }
                    }

                    mcu_count += 1;
                }
            } else {
                // Interleaved DC scan: blocks in MCU order
                'dc_scan: for mcu_y in 0..mcu_rows {
                    for mcu_x in 0..mcu_cols {
                        // Check for restart marker
                        if restart_interval > 0
                            && mcu_count > 0
                            && mcu_count % restart_interval == 0
                        {
                            // Align to byte boundary (discard padding bits)
                            decoder.align_to_byte();
                            // Read and verify restart marker
                            decoder.read_restart_marker(next_restart_num)?;
                            // Update expected marker number (cycles 0-7)
                            next_restart_num = (next_restart_num + 1) & 7;
                            // Reset DC predictors
                            decoder.reset_dc();
                        }

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
                                        match decoder.decode_dc_first(
                                            *comp_idx,
                                            *dc_table as usize,
                                            al,
                                        )? {
                                            ScanRead::Value(dc) => {
                                                self.coeffs[*comp_idx][block_idx][0] = dc;
                                            }
                                            ScanRead::EndOfScan | ScanRead::Truncated => {
                                                // End of scan data - remaining blocks have DC=0
                                                break 'dc_scan;
                                            }
                                        }
                                    } else {
                                        // DC refinement scan
                                        match decoder.decode_dc_refine(al)? {
                                            ScanRead::Value(bit) => {
                                                self.coeffs[*comp_idx][block_idx][0] |= bit;
                                            }
                                            ScanRead::EndOfScan | ScanRead::Truncated => {
                                                // End of scan data - remaining blocks unchanged
                                                break 'dc_scan;
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        mcu_count += 1;
                    }
                }
            }
        } else {
            // AC scan (single component only for progressive)
            // Progressive AC scans can only have one component
            if scan_components.len() != 1 {
                return Err(Error::invalid_jpeg_data(
                    "progressive AC scan must have single component",
                ));
            }

            let (comp_idx, _dc_table, ac_table) = scan_components[0];
            let h_samp = self.components[comp_idx].h_samp_factor as usize;
            let v_samp = self.components[comp_idx].v_samp_factor as usize;

            // For non-interleaved AC scans, blocks are encoded in raster order
            // NOT in interleaved MCU order. Each MCU contains exactly 1 block.
            let comp_blocks_h = mcu_cols * h_samp;
            let comp_blocks_v = mcu_rows * v_samp;
            let total_blocks = comp_blocks_h * comp_blocks_v;

            // Reset MCU count and restart number for AC scan (each scan has its own restart sequence)
            mcu_count = 0;
            next_restart_num = 0;

            for block_idx in 0..total_blocks {
                // Check for restart marker
                if restart_interval > 0 && mcu_count > 0 && mcu_count % restart_interval == 0 {
                    // Align to byte boundary (discard padding bits)
                    decoder.align_to_byte();
                    // Read and verify restart marker
                    decoder.read_restart_marker(next_restart_num)?;
                    // Update expected marker number (cycles 0-7)
                    next_restart_num = (next_restart_num + 1) & 7;
                    // Reset DC predictors and EOB run
                    decoder.reset_dc();
                    eob_run = 0;
                }

                if is_first_scan {
                    // AC first scan
                    match decoder.decode_ac_first(
                        &mut self.coeffs[comp_idx][block_idx],
                        ac_table as usize,
                        ss,
                        se,
                        al,
                        &mut eob_run,
                    )? {
                        ScanRead::Value(()) => {}
                        ScanRead::EndOfScan | ScanRead::Truncated => {
                            // End of scan data - remaining blocks have zeros (implicit EOB)
                            // This is normal in progressive JPEG when encoder uses
                            // implicit EOB at end of scan
                            break;
                        }
                    }
                } else {
                    // AC refinement scan
                    match decoder.decode_ac_refine(
                        &mut self.coeffs[comp_idx][block_idx],
                        ac_table as usize,
                        ss,
                        se,
                        al,
                        &mut eob_run,
                    )? {
                        ScanRead::Value(()) => {}
                        ScanRead::EndOfScan | ScanRead::Truncated => {
                            // End of scan data - remaining blocks unchanged
                            break;
                        }
                    }
                }

                mcu_count += 1;
            }
        }

        // Debug: print position after scan
        if std::env::var("DEBUG_PROGRESSIVE").is_ok() {
            eprintln!(
                "DEBUG prog end: decoder.position()={} new self.position={}",
                decoder.position(),
                self.position + decoder.position()
            );
        }

        self.position += decoder.position();
        Ok(())
    }

    pub(super) fn info(&self) -> JpegInfo {
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

    /// Extracts raw quantized DCT coefficients for analysis.
    ///
    /// Must be called after `decode()` with streaming disabled.
    pub(super) fn extract_coefficients(
        &self,
    ) -> Result<super::image::DecodedCoefficients> {
        use super::image::{ComponentCoefficients, DecodedCoefficients};

        if self.coeffs.is_empty() {
            return Err(Error::internal("no coefficients available - was streaming used?"));
        }

        // Calculate MCU dimensions
        let mut max_h_samp = 1u8;
        let mut max_v_samp = 1u8;
        for i in 0..self.num_components as usize {
            max_h_samp = max_h_samp.max(self.components[i].h_samp_factor);
            max_v_samp = max_v_samp.max(self.components[i].v_samp_factor);
        }
        let mcu_width = (max_h_samp as usize) * 8;
        let mcu_height = (max_v_samp as usize) * 8;
        let mcu_cols = (self.width as usize + mcu_width - 1) / mcu_width;
        let mcu_rows = (self.height as usize + mcu_height - 1) / mcu_height;

        // Extract component coefficients
        let mut components = Vec::with_capacity(self.num_components as usize);
        for comp_idx in 0..self.num_components as usize {
            let comp = &self.components[comp_idx];
            let h_samp = comp.h_samp_factor as usize;
            let v_samp = comp.v_samp_factor as usize;
            let blocks_wide = mcu_cols * h_samp;
            let blocks_high = mcu_rows * v_samp;

            // Flatten blocks into contiguous coefficient array
            let mut coeffs = Vec::with_capacity(blocks_wide * blocks_high * 64);
            for block in &self.coeffs[comp_idx] {
                coeffs.extend_from_slice(block);
            }

            components.push(ComponentCoefficients {
                id: comp.id,
                coeffs,
                blocks_wide,
                blocks_high,
                h_samp: comp.h_samp_factor,
                v_samp: comp.v_samp_factor,
            });
        }

        // Copy quant tables
        let quant_tables = self.quant_tables.to_vec();

        Ok(DecodedCoefficients {
            width: self.width,
            height: self.height,
            components,
            quant_tables,
        })
    }

    /// Check if we can use the fast integer decode path.
    ///
    /// Fast path requirements:
    /// - Non-XYB (standard JPEG)
    /// - 4:4:4 subsampling (no chroma downsampling to avoid f32 upsampling)
    /// - RGB output format
    fn can_use_fast_i16_path(&self, format: PixelFormat, is_xyb: bool) -> bool {
        if is_xyb {
            return false;
        }
        if format != PixelFormat::Rgb {
            return false;
        }
        if self.num_components != 3 {
            return false;
        }

        // Check for 4:4:4 (all components have same sampling factors)
        let h_samp_0 = self.components[0].h_samp_factor;
        let v_samp_0 = self.components[0].v_samp_factor;
        for i in 1..3 {
            if self.components[i].h_samp_factor != h_samp_0
                || self.components[i].v_samp_factor != v_samp_0
            {
                return false;
            }
        }

        true
    }

    /// Fast decode path using integer arithmetic throughout.
    ///
    /// This path avoids f32 entirely by using:
    /// - Integer IDCT (outputs i16 [0, 255])
    /// - Integer color conversion (i16 YCbCr → u8 RGB)
    ///
    /// Streams MCU row by row to keep data in L2 cache.
    /// Only works for non-XYB 4:4:4 RGB output.
    fn to_pixels_fast_i16(&self, _fancy_upsampling: bool) -> Result<Vec<u8>> {
        let width = self.width as usize;
        let height = self.height as usize;

        // Calculate max sampling factors (should all be the same for 4:4:4)
        let max_h_samp = self.components[0].h_samp_factor as usize;
        let max_v_samp = self.components[0].v_samp_factor as usize;

        // MCU dimensions
        let mcu_height = max_v_samp * 8;
        let mcu_cols = (width + max_h_samp * 8 - 1) / (max_h_samp * 8);
        let mcu_rows = (height + mcu_height - 1) / mcu_height;

        // Component info
        let comp_infos = self.build_comp_infos(mcu_cols, mcu_rows, max_h_samp, max_v_samp, 3)?;

        // Allocate strip buffers for one MCU row (reused each iteration)
        // Strip height = max_v_samp * 8 pixels
        let strip_height = mcu_height;
        let strip_width = comp_infos[0].comp_width;
        let strip_size = strip_width * strip_height;

        // Allocate strip buffers - values will be fully overwritten by IDCT
        // Note: Strips are fully written by IDCT before color conversion reads them
        let mut y_strip: Vec<i16> = try_alloc_maybeuninit(strip_size, "Y strip buffer")?;
        let mut cb_strip: Vec<i16> = try_alloc_maybeuninit(strip_size, "Cb strip buffer")?;
        let mut cr_strip: Vec<i16> = try_alloc_maybeuninit(strip_size, "Cr strip buffer")?;

        // Allocate output RGB buffer
        // Note: All pixels are written by color conversion before the buffer is returned
        let rgb_size = checked_size_2d(width, height).and_then(|s| checked_size_2d(s, 3))?;
        let mut rgb: Vec<u8> = try_alloc_maybeuninit(rgb_size, "RGB output buffer")?;

        // Process MCU row by row
        for imcu_row in 0..mcu_rows {
            // No need to clear strips - we write all pixels we'll read

            // IDCT all blocks in this MCU row for all 3 components
            for comp_idx in 0..3 {
                let info = &comp_infos[comp_idx];
                let quant = self.quant_tables[info.quant_idx]
                    .as_ref()
                    .ok_or(Error::internal("missing quantization table"))?;

                let strip = match comp_idx {
                    0 => &mut y_strip,
                    1 => &mut cb_strip,
                    _ => &mut cr_strip,
                };

                for iy in 0..info.v_samp {
                    let by = imcu_row * info.v_samp + iy;
                    if by >= info.comp_blocks_v {
                        continue;
                    }

                    let strip_row = iy * DCT_SIZE; // Row within the strip

                    for bx in 0..info.comp_blocks_h {
                        let block_idx = by * info.comp_blocks_h + bx;
                        if block_idx >= self.coeffs[comp_idx].len() {
                            continue;
                        }
                        let coeffs = &self.coeffs[comp_idx][block_idx];
                        let coeff_count = self.coeff_counts[comp_idx][block_idx];

                        // Fused dequantize + unzigzag (single pass)
                        let mut dequant_i32 = dequantize_unzigzag_i32(coeffs, quant);

                        // IDCT writes directly to strip buffer (no intermediate copy)
                        // Use tiered IDCT based on coefficient count for speed
                        let base_px = bx * DCT_SIZE;
                        let dst_offset = strip_row * strip_width + base_px;
                        idct_int_tiered(
                            &mut dequant_i32,
                            &mut strip[dst_offset..],
                            strip_width,
                            coeff_count,
                        );
                    }
                }
            }

            // Color convert this MCU row's strips directly to RGB output
            let y_start = imcu_row * mcu_height;
            let rows_this_mcu = mcu_height.min(height.saturating_sub(y_start));
            let cols_this_mcu = width.min(strip_width);

            for row in 0..rows_this_mcu {
                let strip_offset = row * strip_width;
                let rgb_offset = (y_start + row) * width * 3;

                // Convert one row at a time for cache efficiency
                ycbcr_planes_i16_to_rgb_u8(
                    &y_strip[strip_offset..strip_offset + cols_this_mcu],
                    &cb_strip[strip_offset..strip_offset + cols_this_mcu],
                    &cr_strip[strip_offset..strip_offset + cols_this_mcu],
                    &mut rgb[rgb_offset..rgb_offset + cols_this_mcu * 3],
                );
            }
        }

        Ok(rgb)
    }

    #[allow(clippy::wrong_self_convention)] // Takes &mut self to take() internal buffer
    pub(super) fn to_pixels(
        &mut self,
        format: PixelFormat,
        is_xyb: bool,
        fancy_upsampling: bool,
    ) -> Result<Vec<u8>> {
        // If streaming decode was used, return its result directly (zero-copy)
        if format == PixelFormat::Rgb && !is_xyb {
            if let Some(rgb) = self.streaming_rgb.take() {
                return Ok(rgb);
            }
        }

        if self.coeffs.is_empty() {
            return Err(Error::internal("no decoded data"));
        }

        // Try fast integer path for non-XYB 4:4:4 RGB images
        if self.can_use_fast_i16_path(format, is_xyb) {
            return self.to_pixels_fast_i16(fancy_upsampling);
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

        // Pre-compute component info for efficiency
        let comp_infos = self.build_comp_infos(
            mcu_cols,
            mcu_rows,
            max_h_samp as usize,
            max_v_samp as usize,
            self.num_components as usize,
        )?;

        // Initialize bias stats and biases (C++ initializes to 0 via memset)
        let mut bias_stats = DequantBiasStats::new(self.num_components as usize);
        let mut component_biases: Vec<[f32; DCT_BLOCK_SIZE]> =
            vec![[0.0f32; DCT_BLOCK_SIZE]; self.num_components as usize];

        // Allocate component planes as f32 (C++ jpegli keeps f32 until final output)
        let mut comp_planes_f32: Vec<Vec<f32>> = Vec::new();
        for info in &comp_infos {
            let comp_plane_size = checked_size_2d(info.comp_width, info.comp_height)?;
            comp_planes_f32.push(vec![0.0f32; comp_plane_size]);
        }

        // Process MCU row by MCU row (matching C++ incremental bias recomputation)
        for imcu_row in 0..mcu_rows {
            // For each component in this MCU row
            for comp_idx in 0..self.num_components as usize {
                let info = &comp_infos[comp_idx];
                let quant = self.quant_tables[info.quant_idx]
                    .as_ref()
                    .ok_or(Error::internal("missing quantization table"))?;

                // Phase 1: Gather stats for full-res components
                if info.is_full_res {
                    for iy in 0..info.v_samp {
                        let by = imcu_row * info.v_samp + iy;
                        if by >= info.comp_blocks_v {
                            continue;
                        }
                        for bx in 0..info.comp_blocks_h {
                            let block_idx = by * info.comp_blocks_h + bx;
                            if block_idx >= self.coeffs[comp_idx].len() {
                                continue;
                            }
                            let coeffs = &self.coeffs[comp_idx][block_idx];
                            let mut natural_coeffs = [0i16; DCT_BLOCK_SIZE];
                            for (i, &zi) in JPEG_NATURAL_ORDER[..DCT_BLOCK_SIZE].iter().enumerate()
                            {
                                natural_coeffs[zi as usize] = coeffs[i];
                            }
                            bias_stats.gather_block(comp_idx, &natural_coeffs);
                        }
                    }

                    // Phase 2: Recompute biases every 4 MCU rows (matching C++ behavior)
                    if imcu_row % 4 == 3 {
                        component_biases[comp_idx] = bias_stats.compute_biases(comp_idx);
                    }
                }

                // Phase 3: IDCT for this component in this MCU row
                // Store as f32 (C++ jpegli keeps f32 until final output for precision)
                let _biases = &component_biases[comp_idx];
                let comp_plane_f32 = &mut comp_planes_f32[comp_idx];

                for iy in 0..info.v_samp {
                    let by = imcu_row * info.v_samp + iy;
                    if by >= info.comp_blocks_v {
                        continue;
                    }

                    // Pre-compute base y position and check row bounds once
                    let base_py = by * DCT_SIZE;
                    let rows_to_copy = DCT_SIZE.min(info.comp_height.saturating_sub(base_py));

                    for bx in 0..info.comp_blocks_h {
                        let block_idx = by * info.comp_blocks_h + bx;
                        if block_idx >= self.coeffs[comp_idx].len() {
                            continue;
                        }
                        let coeffs = &self.coeffs[comp_idx][block_idx];

                        // Zigzag reorder
                        let mut natural_coeffs = [0i16; DCT_BLOCK_SIZE];
                        for (i, &zi) in JPEG_NATURAL_ORDER[..DCT_BLOCK_SIZE].iter().enumerate() {
                            natural_coeffs[zi as usize] = coeffs[i];
                        }

                        // Store pixels - use row-based copy for efficiency
                        let base_px = bx * DCT_SIZE;
                        let cols_to_copy = DCT_SIZE.min(info.comp_width.saturating_sub(base_px));

                        if is_xyb {
                            // XYB mode: use f32 IDCT for extended gamut precision
                            let dequant = dequantize_block(&natural_coeffs, quant);
                            let pixels = inverse_dct_8x8(&dequant);

                            if cols_to_copy == DCT_SIZE {
                                for y in 0..rows_to_copy {
                                    let dst_offset = (base_py + y) * info.comp_width + base_px;
                                    let src_offset = y * DCT_SIZE;
                                    comp_plane_f32[dst_offset..dst_offset + DCT_SIZE]
                                        .copy_from_slice(
                                            &pixels[src_offset..src_offset + DCT_SIZE],
                                        );
                                }
                            } else {
                                for y in 0..rows_to_copy {
                                    for x in 0..cols_to_copy {
                                        comp_plane_f32
                                            [(base_py + y) * info.comp_width + base_px + x] =
                                            pixels[y * DCT_SIZE + x];
                                    }
                                }
                            }
                        } else {
                            // Standard JPEG: use fast integer IDCT
                            let mut dequant_i32 = dequantize_block_i32(&natural_coeffs, quant);
                            let mut pixels_i16 = [0i16; DCT_BLOCK_SIZE];
                            idct_int_auto(&mut dequant_i32, &mut pixels_i16, 8);

                            // Convert i16 [0,255] to f32 centered [-128,127]
                            if cols_to_copy == DCT_SIZE {
                                for y in 0..rows_to_copy {
                                    let dst_offset = (base_py + y) * info.comp_width + base_px;
                                    let src_offset = y * DCT_SIZE;
                                    for x in 0..DCT_SIZE {
                                        comp_plane_f32[dst_offset + x] =
                                            pixels_i16[src_offset + x] as f32 - 128.0;
                                    }
                                }
                            } else {
                                for y in 0..rows_to_copy {
                                    for x in 0..cols_to_copy {
                                        comp_plane_f32
                                            [(base_py + y) * info.comp_width + base_px + x] =
                                            pixels_i16[y * DCT_SIZE + x] as f32 - 128.0;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Upsample if needed - keep as f32 for precision
        let output_size = checked_size_2d(width, height)?;
        let mut planes_f32: Vec<Vec<f32>> = Vec::new();

        for comp_idx in 0..self.num_components as usize {
            let info = &comp_infos[comp_idx];
            let comp_plane_f32 = &comp_planes_f32[comp_idx];

            let plane_f32 = if info.h_samp < max_h_samp as usize
                || info.v_samp < max_v_samp as usize
            {
                let scale_x = max_h_samp as usize / info.h_samp;
                let scale_y = max_v_samp as usize / info.v_samp;

                if fancy_upsampling {
                    // Triangle filter (3:1 weights) - separable implementation
                    // First upsample horizontally, then vertically
                    upsample_fancy(
                        comp_plane_f32,
                        info.comp_width,
                        info.comp_height,
                        width,
                        height,
                        scale_x,
                        scale_y,
                    )
                } else {
                    // Box filter (nearest neighbor)
                    let mut upsampled = vec![0.0f32; output_size];
                    for py in 0..height {
                        for px in 0..width {
                            let sx = (px / scale_x).min(info.comp_width - 1);
                            let sy = (py / scale_y).min(info.comp_height - 1);
                            upsampled[py * width + px] = comp_plane_f32[sy * info.comp_width + sx];
                        }
                    }
                    upsampled
                }
            } else {
                // Full resolution - just clip to image dimensions
                let mut plane = vec![0.0f32; output_size];
                for py in 0..height {
                    for px in 0..width {
                        plane[py * width + px] = comp_plane_f32[py * info.comp_width + px];
                    }
                }
                plane
            };

            planes_f32.push(plane_f32);
        }

        // Convert to output format using batch conversion functions
        match (self.num_components, format) {
            (1, PixelFormat::Gray) => {
                // Grayscale: level shift and convert to u8
                let mut output = vec![0u8; output_size];
                gray_f32_to_gray_u8(&planes_f32[0], &mut output);
                Ok(output)
            }
            (1, PixelFormat::Rgb) => {
                let rgb_size =
                    checked_size_2d(width, height).and_then(|s| checked_size_2d(s, 3))?;
                let mut rgb = vec![0u8; rgb_size];
                gray_f32_to_rgb_u8(&planes_f32[0], &mut rgb);
                Ok(rgb)
            }
            (3, PixelFormat::Rgb) => {
                let rgb_size =
                    checked_size_2d(width, height).and_then(|s| checked_size_2d(s, 3))?;
                let mut rgb = vec![0u8; rgb_size];

                if is_xyb {
                    // XYB mode: Output raw level-shifted values, NO YCbCr→RGB conversion.
                    // The XYB values are stored in YCbCr positions but are NOT YCbCr.
                    // The ICC profile transforms these directly to sRGB.
                    crate::color::xyb::xyb_planes_to_rgb_u8_simd(
                        &planes_f32[0],
                        &planes_f32[1],
                        &planes_f32[2],
                        &mut rgb,
                    );
                } else {
                    // YCbCr to RGB conversion using batch function
                    ycbcr_planes_f32_to_rgb_u8(
                        &planes_f32[0],
                        &planes_f32[1],
                        &planes_f32[2],
                        &mut rgb,
                    );
                }
                Ok(rgb)
            }
            _ => Err(Error::unsupported_feature("unsupported color conversion")),
        }
    }

    /// Convert decoded coefficients to f32 pixels.
    /// Values are normalized to range 0.0-1.0.
    pub(super) fn to_pixels_f32(
        &self,
        format: PixelFormat,
        is_xyb: bool,
        fancy_upsampling: bool,
    ) -> Result<Vec<f32>> {
        if self.coeffs.is_empty() {
            return Err(Error::internal("no decoded data"));
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

        // Pre-compute component info
        let comp_infos = self.build_comp_infos(
            mcu_cols,
            mcu_rows,
            max_h_samp as usize,
            max_v_samp as usize,
            self.num_components as usize,
        )?;

        // Initialize bias stats and biases
        let mut bias_stats = DequantBiasStats::new(self.num_components as usize);
        let mut component_biases: Vec<[f32; DCT_BLOCK_SIZE]> =
            vec![[0.0f32; DCT_BLOCK_SIZE]; self.num_components as usize];

        // Allocate component planes as f32
        let mut comp_planes_f32: Vec<Vec<f32>> = Vec::new();
        for info in &comp_infos {
            let comp_plane_size = checked_size_2d(info.comp_width, info.comp_height)?;
            comp_planes_f32.push(vec![0.0f32; comp_plane_size]);
        }

        // Process MCU row by MCU row
        for imcu_row in 0..mcu_rows {
            for comp_idx in 0..self.num_components as usize {
                let info = &comp_infos[comp_idx];
                let quant = self.quant_tables[info.quant_idx]
                    .as_ref()
                    .ok_or(Error::internal("missing quantization table"))?;

                // Gather stats for full-res components
                if info.is_full_res {
                    for iy in 0..info.v_samp {
                        let by = imcu_row * info.v_samp + iy;
                        if by >= info.comp_blocks_v {
                            continue;
                        }
                        for bx in 0..info.comp_blocks_h {
                            let block_idx = by * info.comp_blocks_h + bx;
                            if block_idx >= self.coeffs[comp_idx].len() {
                                continue;
                            }
                            let coeffs = &self.coeffs[comp_idx][block_idx];
                            let mut natural_coeffs = [0i16; DCT_BLOCK_SIZE];
                            for (i, &zi) in JPEG_NATURAL_ORDER[..DCT_BLOCK_SIZE].iter().enumerate()
                            {
                                natural_coeffs[zi as usize] = coeffs[i];
                            }
                            bias_stats.gather_block(comp_idx, &natural_coeffs);
                        }
                    }

                    // Recompute biases every 4 MCU rows
                    if imcu_row % 4 == 3 {
                        component_biases[comp_idx] = bias_stats.compute_biases(comp_idx);
                    }
                }

                // IDCT for this component
                let biases = &component_biases[comp_idx];
                let comp_plane_f32 = &mut comp_planes_f32[comp_idx];

                for iy in 0..info.v_samp {
                    let by = imcu_row * info.v_samp + iy;
                    if by >= info.comp_blocks_v {
                        continue;
                    }

                    for bx in 0..info.comp_blocks_h {
                        let block_idx = by * info.comp_blocks_h + bx;
                        if block_idx >= self.coeffs[comp_idx].len() {
                            continue;
                        }
                        let coeffs = &self.coeffs[comp_idx][block_idx];

                        let mut natural_coeffs = [0i16; DCT_BLOCK_SIZE];
                        for (i, &zi) in JPEG_NATURAL_ORDER[..DCT_BLOCK_SIZE].iter().enumerate() {
                            natural_coeffs[zi as usize] = coeffs[i];
                        }

                        // Always use f32 IDCT for f32 output - preserves fractional precision
                        let dequant = if is_xyb {
                            dequantize_block(&natural_coeffs, quant)
                        } else {
                            dequantize_block_with_bias(&natural_coeffs, quant, biases)
                        };
                        let pixels = inverse_dct_8x8(&dequant);

                        for y in 0..DCT_SIZE {
                            for x in 0..DCT_SIZE {
                                let px = bx * DCT_SIZE + x;
                                let py = by * DCT_SIZE + y;
                                if px < info.comp_width && py < info.comp_height {
                                    comp_plane_f32[py * info.comp_width + px] =
                                        pixels[y * DCT_SIZE + x];
                                }
                            }
                        }
                    }
                }
            }
        }

        // Upsample if needed
        let output_size = checked_size_2d(width, height)?;
        let mut planes_f32: Vec<Vec<f32>> = Vec::new();

        for comp_idx in 0..self.num_components as usize {
            let info = &comp_infos[comp_idx];
            let comp_plane_f32 = &comp_planes_f32[comp_idx];

            let plane_f32 = if info.h_samp < max_h_samp as usize
                || info.v_samp < max_v_samp as usize
            {
                let scale_x = max_h_samp as usize / info.h_samp;
                let scale_y = max_v_samp as usize / info.v_samp;

                if fancy_upsampling {
                    // Triangle filter (3:1 weights) - separable implementation
                    upsample_fancy(
                        comp_plane_f32,
                        info.comp_width,
                        info.comp_height,
                        width,
                        height,
                        scale_x,
                        scale_y,
                    )
                } else {
                    // Box filter (nearest neighbor)
                    let mut upsampled = vec![0.0f32; output_size];
                    for py in 0..height {
                        for px in 0..width {
                            let sx = (px / scale_x).min(info.comp_width - 1);
                            let sy = (py / scale_y).min(info.comp_height - 1);
                            upsampled[py * width + px] = comp_plane_f32[sy * info.comp_width + sx];
                        }
                    }
                    upsampled
                }
            } else {
                let mut plane = vec![0.0f32; output_size];
                for py in 0..height {
                    for px in 0..width {
                        plane[py * width + px] = comp_plane_f32[py * info.comp_width + px];
                    }
                }
                plane
            };

            planes_f32.push(plane_f32);
        }

        // Convert to output format as f32 (values normalized to 0.0-1.0)
        match (self.num_components, format) {
            (1, PixelFormat::Gray) => {
                // Grayscale: level shift and normalize to 0.0-1.0
                let mut output = vec![0.0f32; output_size];
                gray_f32_to_gray_f32(&planes_f32[0], &mut output);
                Ok(output)
            }
            (1, PixelFormat::Rgb) => {
                let rgb_size =
                    checked_size_2d(width, height).and_then(|s| checked_size_2d(s, 3))?;
                let mut rgb = vec![0.0f32; rgb_size];
                gray_f32_to_rgb_f32(&planes_f32[0], &mut rgb);
                Ok(rgb)
            }
            (3, PixelFormat::Rgb) => {
                let rgb_size =
                    checked_size_2d(width, height).and_then(|s| checked_size_2d(s, 3))?;
                let mut rgb = vec![0.0f32; rgb_size];

                if is_xyb {
                    // XYB mode: Output raw level-shifted values, normalized to 0.0-1.0
                    crate::color::xyb::xyb_planes_to_rgb_f32_simd(
                        &planes_f32[0],
                        &planes_f32[1],
                        &planes_f32[2],
                        &mut rgb,
                    );
                } else {
                    // YCbCr to RGB conversion using batch function
                    ycbcr_planes_f32_to_rgb_f32(
                        &planes_f32[0],
                        &planes_f32[1],
                        &planes_f32[2],
                        &mut rgb,
                    );
                }
                Ok(rgb)
            }
            _ => Err(Error::unsupported_feature("unsupported color conversion")),
        }
    }

    /// Convert decoded coefficients to YCbCr f32 planes.
    ///
    /// Returns (Y, Cb, Cr) planes, each width×height in size.
    /// Values are in centered range [-128, 127] (raw DCT output).
    /// Chroma planes are upsampled to full resolution.
    pub(super) fn to_ycbcr_planes_f32(
        &self,
        fancy_upsampling: bool,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        if self.coeffs.is_empty() {
            return Err(Error::internal("no decoded data"));
        }

        if self.num_components != 3 {
            return Err(Error::unsupported_feature(
                "YCbCr planes require 3-component image",
            ));
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

        // Pre-compute component info
        let comp_infos = self.build_comp_infos(
            mcu_cols,
            mcu_rows,
            max_h_samp as usize,
            max_v_samp as usize,
            self.num_components as usize,
        )?;

        // Initialize bias stats and biases
        let mut bias_stats = DequantBiasStats::new(self.num_components as usize);
        let mut component_biases: Vec<[f32; DCT_BLOCK_SIZE]> =
            vec![[0.0f32; DCT_BLOCK_SIZE]; self.num_components as usize];

        // Allocate component planes as f32
        let mut comp_planes_f32: Vec<Vec<f32>> = Vec::new();
        for info in &comp_infos {
            let comp_plane_size = checked_size_2d(info.comp_width, info.comp_height)?;
            comp_planes_f32.push(vec![0.0f32; comp_plane_size]);
        }

        // Process MCU row by MCU row
        for imcu_row in 0..mcu_rows {
            for comp_idx in 0..self.num_components as usize {
                let info = &comp_infos[comp_idx];
                let quant = self.quant_tables[info.quant_idx]
                    .as_ref()
                    .ok_or(Error::internal("missing quantization table"))?;

                // Phase 1: Gather stats for full-res components
                if info.is_full_res {
                    for iy in 0..info.v_samp {
                        let by = imcu_row * info.v_samp + iy;
                        if by >= info.comp_blocks_v {
                            continue;
                        }
                        for bx in 0..info.comp_blocks_h {
                            let block_idx = by * info.comp_blocks_h + bx;
                            if block_idx >= self.coeffs[comp_idx].len() {
                                continue;
                            }
                            let coeffs = &self.coeffs[comp_idx][block_idx];
                            let mut natural_coeffs = [0i16; DCT_BLOCK_SIZE];
                            for (i, &zi) in JPEG_NATURAL_ORDER[..DCT_BLOCK_SIZE].iter().enumerate()
                            {
                                natural_coeffs[zi as usize] = coeffs[i];
                            }
                            bias_stats.gather_block(comp_idx, &natural_coeffs);
                        }
                    }

                    // Recompute biases every 4 MCU rows
                    if imcu_row % 4 == 3 {
                        component_biases[comp_idx] = bias_stats.compute_biases(comp_idx);
                    }
                }

                // Phase 2: IDCT
                let _biases = &component_biases[comp_idx];
                let comp_plane_f32 = &mut comp_planes_f32[comp_idx];

                for iy in 0..info.v_samp {
                    let by = imcu_row * info.v_samp + iy;
                    if by >= info.comp_blocks_v {
                        continue;
                    }

                    let base_py = by * DCT_SIZE;
                    let rows_to_copy = DCT_SIZE.min(info.comp_height.saturating_sub(base_py));

                    for bx in 0..info.comp_blocks_h {
                        let block_idx = by * info.comp_blocks_h + bx;
                        if block_idx >= self.coeffs[comp_idx].len() {
                            continue;
                        }
                        let coeffs = &self.coeffs[comp_idx][block_idx];

                        // Zigzag reorder
                        let mut natural_coeffs = [0i16; DCT_BLOCK_SIZE];
                        for (i, &zi) in JPEG_NATURAL_ORDER[..DCT_BLOCK_SIZE].iter().enumerate() {
                            natural_coeffs[zi as usize] = coeffs[i];
                        }

                        // Use fast integer IDCT (always non-XYB for YCbCr output)
                        let mut dequant_i32 = dequantize_block_i32(&natural_coeffs, quant);
                        let mut pixels_i16 = [0i16; DCT_BLOCK_SIZE];
                        idct_int_auto(&mut dequant_i32, &mut pixels_i16, 8);

                        // Store pixels
                        let base_px = bx * DCT_SIZE;
                        let cols_to_copy = DCT_SIZE.min(info.comp_width.saturating_sub(base_px));

                        // Convert i16 [0,255] to f32 centered [-128,127]
                        if cols_to_copy == DCT_SIZE {
                            for y in 0..rows_to_copy {
                                let dst_offset = (base_py + y) * info.comp_width + base_px;
                                let src_offset = y * DCT_SIZE;
                                for x in 0..DCT_SIZE {
                                    comp_plane_f32[dst_offset + x] =
                                        pixels_i16[src_offset + x] as f32 - 128.0;
                                }
                            }
                        } else {
                            for y in 0..rows_to_copy {
                                for x in 0..cols_to_copy {
                                    comp_plane_f32[(base_py + y) * info.comp_width + base_px + x] =
                                        pixels_i16[y * DCT_SIZE + x] as f32 - 128.0;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Upsample chroma and clip to image dimensions
        let output_size = checked_size_2d(width, height)?;
        let mut planes_f32: Vec<Vec<f32>> = Vec::with_capacity(3);

        for comp_idx in 0..3 {
            let info = &comp_infos[comp_idx];
            let comp_plane_f32 = &comp_planes_f32[comp_idx];

            let plane_f32 = if info.h_samp < max_h_samp as usize
                || info.v_samp < max_v_samp as usize
            {
                let scale_x = max_h_samp as usize / info.h_samp;
                let scale_y = max_v_samp as usize / info.v_samp;

                if fancy_upsampling {
                    upsample_fancy(
                        comp_plane_f32,
                        info.comp_width,
                        info.comp_height,
                        width,
                        height,
                        scale_x,
                        scale_y,
                    )
                } else {
                    // Box filter (nearest neighbor)
                    let mut upsampled = vec![0.0f32; output_size];
                    for py in 0..height {
                        for px in 0..width {
                            let sx = (px / scale_x).min(info.comp_width - 1);
                            let sy = (py / scale_y).min(info.comp_height - 1);
                            upsampled[py * width + px] = comp_plane_f32[sy * info.comp_width + sx];
                        }
                    }
                    upsampled
                }
            } else {
                // Full resolution - just clip to image dimensions
                let mut plane = vec![0.0f32; output_size];
                for py in 0..height {
                    for px in 0..width {
                        plane[py * width + px] = comp_plane_f32[py * info.comp_width + px];
                    }
                }
                plane
            };

            planes_f32.push(plane_f32);
        }

        Ok((
            core::mem::take(&mut planes_f32[0]),
            core::mem::take(&mut planes_f32[1]),
            core::mem::take(&mut planes_f32[2]),
        ))
    }
}
