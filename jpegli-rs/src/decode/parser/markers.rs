//! JPEG marker parsing (SOF, DHT, DQT, DRI, APP segments).
//!
//! This module handles parsing of JPEG marker segments during header reading.

use crate::error::{Error, Result};
use crate::foundation::alloc::validate_dimensions;
use crate::foundation::consts::{
    DCT_BLOCK_SIZE, JPEG_NATURAL_ORDER, MARKER_APP0, MARKER_COM, MARKER_DHT, MARKER_DQT,
    MARKER_DRI, MARKER_EOI, MARKER_SOF0, MARKER_SOF1, MARKER_SOF2, MAX_COMPONENTS,
    MAX_HUFFMAN_TABLES, MAX_QUANT_TABLES,
};
use crate::huffman::HuffmanDecodeTable;
use crate::types::JpegMode;

use super::JpegParser;

/// Marker parsing methods for JpegParser.
impl<'a> JpegParser<'a> {
    /// Read and parse the JPEG header up to (but not including) SOS.
    pub(crate) fn read_header(&mut self) -> Result<()> {
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

    /// Parse SOF (Start of Frame) marker - frame dimensions and components.
    pub(super) fn parse_frame_header(&mut self) -> Result<()> {
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

    /// Parse DQT (Define Quantization Table) marker.
    pub(super) fn parse_quant_table(&mut self) -> Result<()> {
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

    /// Parse DHT (Define Huffman Table) marker.
    pub(super) fn parse_huffman_table(&mut self) -> Result<()> {
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

    /// Parse DRI (Define Restart Interval) marker.
    pub(super) fn parse_restart_interval(&mut self) -> Result<()> {
        let _length = self.read_u16()?;
        self.restart_interval = self.read_u16()?;
        Ok(())
    }

    /// Skip an unknown or unneeded marker segment.
    pub(super) fn skip_segment(&mut self) -> Result<()> {
        let length = self.read_u16()? as usize;
        if length < 2 {
            return Err(Error::invalid_jpeg_data("segment length too short"));
        }
        self.position += length - 2;
        Ok(())
    }
}
