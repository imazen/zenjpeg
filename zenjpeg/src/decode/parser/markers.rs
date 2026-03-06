//! JPEG marker parsing (SOF, DHT, DQT, DRI, APP segments).
//!
//! This module handles parsing of JPEG marker segments during header reading.

use crate::decode::extras::{
    AdobeColorTransform, SegmentType, detect_segment_type, should_preserve_segment,
};
use crate::error::{Error, Result};
use crate::foundation::alloc::validate_dimensions;
use crate::foundation::consts::{
    DCT_BLOCK_SIZE, JPEG_NATURAL_ORDER, MARKER_APP0, MARKER_APP14, MARKER_COM, MARKER_DAC,
    MARKER_DHT, MARKER_DQT, MARKER_DRI, MARKER_EOI, MARKER_SOF0, MARKER_SOF1, MARKER_SOF2,
    MARKER_SOF9, MARKER_SOF10, MAX_COMPONENTS, MAX_HUFFMAN_TABLES, MAX_QUANT_TABLES,
};
use crate::huffman::HuffmanDecodeTable;
use crate::types::JpegMode;

use super::super::{DecodeWarning, Strictness};
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
                MARKER_SOF9 => {
                    self.mode = JpegMode::ArithmeticSequential;
                    self.parse_frame_header()?;
                    return Ok(());
                }
                MARKER_SOF10 => {
                    self.mode = JpegMode::ArithmeticProgressive;
                    self.parse_frame_header()?;
                    return Ok(());
                }
                MARKER_DQT => self.parse_header_marker(|s| s.parse_quant_table())?,
                MARKER_DHT => self.parse_header_marker(|s| s.parse_huffman_table())?,
                MARKER_DAC => self.parse_header_marker(|s| s.parse_dac())?,
                MARKER_DRI => self.parse_header_marker(|s| s.parse_restart_interval())?,
                MARKER_APP0..=0xEF | MARKER_COM => {
                    self.parse_header_marker(|s| s.process_app_or_com(marker))?;
                }
                MARKER_EOI => {
                    return Err(Error::invalid_jpeg_data(
                        "unexpected EOI before frame header",
                    ));
                }
                // Standalone markers (no length field) — skip silently.
                // RST0-RST7 and TEM can appear as stray bytes in corrupted files.
                0xD0..=0xD7 | 0x01 => {}
                _ => self.parse_header_marker(|s| s.skip_segment())?,
            }
        }
    }

    /// Parse a header marker, recovering from errors in Permissive mode.
    ///
    /// In Permissive mode, if a marker's content is corrupted (wrong lengths,
    /// invalid values, etc.), we skip it and continue to the next marker.
    /// This mimics libjpeg-turbo's tolerance of fuzz-mutated files.
    fn parse_header_marker(&mut self, f: impl FnOnce(&mut Self) -> Result<()>) -> Result<()> {
        if self.strictness != Strictness::Permissive {
            return f(self);
        }
        // Save position so we can recover on error
        let saved_pos = self.position;
        match f(self) {
            Ok(()) => Ok(()),
            Err(_) => {
                // Try to skip past the corrupted marker segment.
                // Read the length from the original position and skip.
                self.position = saved_pos;
                if self.position + 2 <= self.data.len() {
                    let len = ((self.data[self.position] as usize) << 8)
                        | self.data[self.position + 1] as usize;
                    if len >= 2 {
                        self.position += len;
                    } else {
                        self.position += 2; // Skip at least the length field
                    }
                }
                self.warn(DecodeWarning::MalformedSegmentSkipped)?;
                Ok(())
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
        // Note: height=0 is valid here - it means DNL marker will define height after first scan
        let effective_max = if self.max_pixels == 0 {
            u64::MAX
        } else {
            self.max_pixels
        };
        if self.height == 0 {
            // DNL mode: validate width only, height will be validated after DNL marker
            if self.width == 0 {
                return Err(Error::invalid_dimensions(
                    self.width,
                    self.height,
                    "width cannot be zero",
                ));
            }
        } else {
            validate_dimensions(self.width, self.height, effective_max)?;
        }

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

            let permissive = self.strictness == Strictness::Permissive;
            let mut had_zero = false;

            if precision == 0 {
                // 8-bit values
                for i in 0..DCT_BLOCK_SIZE {
                    let val = self.read_u8()? as u16;
                    if val == 0 {
                        if permissive {
                            zigzag_values[i] = 1; // Clamp to 1
                            had_zero = true;
                            continue;
                        }
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
                        if permissive {
                            zigzag_values[i] = 1; // Clamp to 1
                            had_zero = true;
                            continue;
                        }
                        return Err(Error::invalid_quant_table(
                            table_idx as u8,
                            "quantization value is zero",
                        ));
                    }
                    zigzag_values[i] = val;
                }
                length -= 129;
            }

            if had_zero {
                self.warn(DecodeWarning::ZeroQuantValue {
                    table_idx: table_idx as u8,
                })?;
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

    /// Parse DAC (Define Arithmetic Coding) marker.
    ///
    /// DAC defines conditioning parameters for arithmetic coding:
    /// - For DC tables: L and U values that classify DC differences
    /// - For AC tables: Kx value that selects context for AC magnitudes
    ///
    /// Format per table: Tc/Th (1 byte) + Cs (1 byte)
    /// - Tc: 0=DC, 1=AC
    /// - Th: table destination (0-3)
    /// - Cs: For DC: L in low 4 bits, U in high 4 bits
    ///       For AC: Kx value (0-63)
    pub(super) fn parse_dac(&mut self) -> Result<()> {
        let mut length = self.read_u16()? as i32 - 2;

        while length >= 2 {
            let info = self.read_u8()?;
            let table_class = info >> 4; // 0=DC, 1=AC
            let table_idx = (info & 0x0F) as usize;

            if table_idx >= 4 {
                return Err(Error::invalid_jpeg_data("DAC table index out of range"));
            }

            let cs = self.read_u8()?;
            length -= 2;

            if table_class == 0 {
                // DC conditioning: L in low 4 bits, U in high 4 bits
                let l = cs & 0x0F;
                let u = cs >> 4;
                // Validate: L <= U (per spec)
                if l > u {
                    return Err(Error::invalid_jpeg_data("DAC DC conditioning: L > U"));
                }
                self.arith_dc_cond[table_idx] = (l, u);
            } else {
                // AC conditioning: Kx value
                let kx = cs & 0x3F; // Kx is 0-63
                self.arith_ac_kx[table_idx] = kx;
            }
        }

        Ok(())
    }

    /// Parse DNL (Define Number of Lines) marker.
    ///
    /// The DNL marker allows the height to be specified after the first scan,
    /// which is useful for Motion JPEG and streaming encoders that don't know
    /// the final height until encoding is complete.
    ///
    /// Per ITU-T T.81 section B.2.5:
    /// - DNL marker appears after the first scan's entropy-coded data
    /// - Only valid if height was 0 in the SOF marker
    /// - Contains a 2-byte length followed by 2-byte number of lines
    pub(super) fn parse_dnl(&mut self) -> Result<()> {
        let length = self.read_u16()?;
        if length != 4 {
            if self.strictness == Strictness::Permissive {
                // Skip malformed DNL marker using declared length
                if length >= 2 {
                    self.position += (length as usize).saturating_sub(2);
                }
                self.warn(DecodeWarning::MalformedSegmentSkipped)?;
                return Ok(());
            }
            return Err(Error::invalid_jpeg_data("DNL marker must have length 4"));
        }

        let num_lines = self.read_u16()? as u32;

        // DNL is only valid if height was 0 in SOF
        if self.height == 0 {
            self.height = num_lines;
        } else if self.height != num_lines {
            // Height was already specified - this is technically invalid.
            // mozjpeg ignores DNL entirely (skip_variable), so Balanced matches that.
            // Strict errors via warn(), Balanced/Lenient: ignore DNL, keep SOF height.
            self.warn(DecodeWarning::DnlHeightConflict {
                sof_height: self.height,
                dnl_height: num_lines,
            })?;
        }

        Ok(())
    }

    /// Skip an unknown or unneeded marker segment.
    pub(super) fn skip_segment(&mut self) -> Result<()> {
        let length = self.read_u16()? as usize;
        if length < 2 {
            if self.strictness == Strictness::Permissive {
                self.warn(DecodeWarning::MalformedSegmentSkipped)?;
                return Ok(());
            }
            return Err(Error::invalid_jpeg_data("segment length too short"));
        }
        self.position += length - 2;
        Ok(())
    }

    /// Process an APP or COM marker, optionally preserving its data.
    pub(super) fn process_app_or_com(&mut self, marker: u8) -> Result<()> {
        let length = self.read_u16()? as usize;
        if length < 2 {
            if self.strictness == Strictness::Permissive {
                self.warn(DecodeWarning::MalformedSegmentSkipped)?;
                return Ok(());
            }
            return Err(Error::invalid_jpeg_data("segment length too short"));
        }
        let data_len = length - 2;

        // Always check for APP14 Adobe marker (needed for CMYK/YCCK detection)
        if marker == MARKER_APP14 && self.position + data_len <= self.data.len() {
            let data = &self.data[self.position..self.position + data_len];
            if let Some(transform) = parse_adobe_app14(data) {
                self.adobe_transform = Some(transform);
            }
        }

        // Check if we should preserve this segment
        if let (Some(config), Some(extras)) = (&self.preserve_config, &mut self.extras)
            && self.position + data_len <= self.data.len()
        {
            let data = &self.data[self.position..self.position + data_len];
            let segment_type = detect_segment_type(marker, data);

            // Record MPF header position for secondary image extraction
            // MPF offsets are relative to the TIFF header (after "MPF\0")
            if segment_type == SegmentType::Mpf && self.mpf_header_pos == 0 {
                self.mpf_header_pos = self.position + 4; // Skip "MPF\0" to get TIFF header pos
            }

            if should_preserve_segment(config, segment_type) {
                extras.add_segment(marker, data.to_vec(), segment_type);
            }
        }

        self.position += data_len;
        Ok(())
    }
}

/// Parse Adobe APP14 segment to extract color transform.
/// Format: "Adobe\0" + version (2 bytes) + flags0 (2) + flags1 (2) + color_transform (1)
fn parse_adobe_app14(data: &[u8]) -> Option<AdobeColorTransform> {
    const ADOBE_SIG: &[u8] = b"Adobe";
    if data.len() < ADOBE_SIG.len() + 7 {
        return None;
    }
    if !data.starts_with(ADOBE_SIG) {
        return None;
    }

    let offset = ADOBE_SIG.len();
    let color_transform_byte = data[offset + 6];

    Some(match color_transform_byte {
        0 => AdobeColorTransform::Unknown, // CMYK (raw, values often inverted)
        1 => AdobeColorTransform::YCbCr,   // Standard YCbCr
        2 => AdobeColorTransform::Ycck,    // YCCK
        _ => AdobeColorTransform::Unknown,
    })
}
