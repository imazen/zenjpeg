//! JPEG output writing methods.
//!
//! This module contains methods for writing JPEG structure:
//! - Markers (SOI, SOF, DQT, DHT, DRI, SOS, EOI)
//! - Quantization tables
//! - Huffman tables
//! - Frame and scan headers

use crate::error::Result;
use crate::foundation::consts::{
    DCT_BLOCK_SIZE, ICC_PROFILE_SIGNATURE, JPEG_NATURAL_ORDER, MARKER_APP2, MARKER_APP14,
    MARKER_DHT, MARKER_DQT, MARKER_DRI, MARKER_SOF0, MARKER_SOF1, MARKER_SOF2, MARKER_SOI,
    MARKER_SOS, MAX_ICC_BYTES_PER_MARKER,
};
use crate::huffman::optimize::{ContextConfig, HuffmanTableSet, OptimizedTable};
use crate::quant::QuantTable;
use crate::types::{JpegMode, Subsampling};

use super::ProgressiveScan;
use super::config::ComputedConfig;

impl ComputedConfig {
    /// Writes the JPEG header (SOI only, no JFIF APP0).
    ///
    /// Note: C++ jpegli does not write JFIF APP0, so we skip it for parity.
    /// The JFIF marker is optional and many modern decoders don't require it.
    pub(crate) fn write_header(&self, output: &mut Vec<u8>) -> Result<()> {
        // SOI only - no JFIF marker for C++ parity
        output.push(0xFF);
        output.push(MARKER_SOI);
        Ok(())
    }

    /// Writes the JPEG header for XYB mode (SOI only, no JFIF).
    ///
    /// XYB mode uses RGB component IDs and an ICC profile for color interpretation.
    /// JFIF APP0 is not appropriate because it implies YCbCr colorspace.
    pub(crate) fn write_header_xyb(&self, output: &mut Vec<u8>) -> Result<()> {
        // SOI only - no JFIF marker for XYB mode
        output.push(0xFF);
        output.push(MARKER_SOI);
        Ok(())
    }

    /// Writes an APP14 Adobe marker for RGB/CMYK/YCCK colorspaces.
    ///
    /// The APP14 marker is required by some decoders to properly interpret
    /// RGB (including XYB), CMYK, and YCCK colorspaces.
    ///
    /// See: https://github.com/google/jpegli/pull/135
    ///
    /// # Arguments
    /// * `transform` - Color transform type:
    ///   - 0 = RGB or CMYK (no transform)
    ///   - 1 = YCbCr
    ///   - 2 = YCCK
    pub(crate) fn write_app14_adobe(&self, output: &mut Vec<u8>, transform: u8) -> Result<()> {
        output.push(0xFF);
        output.push(MARKER_APP14);
        output.extend_from_slice(&[
            0x00, 0x0E, // Length: 14 bytes (includes length field)
            b'A', b'd', b'o', b'b', b'e', // Signature
            0x00, 0x64, // DCTEncodeVersion (100)
            0x00, 0x00, // APP14Flags0
            0x00, 0x00,      // APP14Flags1
            transform, // Color transform
        ]);
        Ok(())
    }

    /// Writes an ICC profile to the JPEG output.
    ///
    /// ICC profiles are stored in APP2 marker segments with the signature "ICC_PROFILE\0".
    /// Large profiles are split into multiple segments (max ~65519 bytes per segment).
    pub(crate) fn write_icc_profile(&self, output: &mut Vec<u8>, icc_data: &[u8]) -> Result<()> {
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

    /// Writes quantization tables for YCbCr mode.
    ///
    /// When `separate_chroma_tables` is true (default):
    /// - Writes 3 tables (Y, Cb, Cr) matching `jpegli_set_distance()` behavior
    ///
    /// When `separate_chroma_tables` is false:
    /// - Writes 2 tables (Y, shared chroma) matching `jpeg_set_quality()` behavior
    /// - The `cr_quant` parameter is ignored; Cb table is used for both
    ///
    /// Supports both 8-bit (baseline) and 16-bit (extended) precision based on
    /// the `precision` field of each QuantTable.
    pub(crate) fn write_quant_tables(
        &self,
        output: &mut Vec<u8>,
        y_quant: &QuantTable,
        cb_quant: &QuantTable,
        cr_quant: &QuantTable,
    ) -> Result<()> {
        if self.separate_chroma_tables {
            // 3 tables: Y, Cb, Cr (matches jpegli_set_distance)
            let tables = [(0u8, y_quant), (1u8, cb_quant), (2u8, cr_quant)];
            Self::write_dqt_segment(output, &tables)
        } else {
            // 2 tables: Y, shared chroma (matches jpeg_set_quality)
            let tables = [(0u8, y_quant), (1u8, cb_quant)];
            Self::write_dqt_segment(output, &tables)
        }
    }

    /// Writes quantization tables for XYB mode (3 separate tables).
    ///
    /// Supports both 8-bit (baseline) and 16-bit (extended) precision based on
    /// the `precision` field of each QuantTable.
    pub(crate) fn write_quant_tables_xyb(
        &self,
        output: &mut Vec<u8>,
        r_quant: &QuantTable,
        g_quant: &QuantTable,
        b_quant: &QuantTable,
    ) -> Result<()> {
        let tables = [(0u8, r_quant), (1u8, g_quant), (2u8, b_quant)];
        Self::write_dqt_segment(output, &tables)
    }

    /// Writes a DQT segment containing one or more quantization tables.
    ///
    /// Each table can independently use 8-bit (precision=0) or 16-bit (precision=1)
    /// values based on its `precision` field.
    fn write_dqt_segment(output: &mut Vec<u8>, tables: &[(u8, &QuantTable)]) -> Result<()> {
        // Calculate total segment length
        // Length field = 2 bytes
        // Per table: 1 byte (precision + table_id) + 64 or 128 bytes (values)
        let mut length: usize = 2;
        for (_, qt) in tables {
            length += 1 + if qt.precision == 0 { 64 } else { 128 };
        }

        // Write marker
        output.push(0xFF);
        output.push(MARKER_DQT);

        // Write length (big-endian)
        output.push((length >> 8) as u8);
        output.push((length & 0xFF) as u8);

        // Write each table
        for &(table_id, qt) in tables {
            // Precision in high nibble, table ID in low nibble
            let info_byte = (qt.precision << 4) | (table_id & 0x0F);
            output.push(info_byte);

            // Write values in zigzag order
            if qt.precision == 0 {
                // 8-bit values
                for i in 0..DCT_BLOCK_SIZE {
                    let val = qt.values[JPEG_NATURAL_ORDER[i] as usize];
                    output.push(val as u8);
                }
            } else {
                // 16-bit values (big-endian)
                for i in 0..DCT_BLOCK_SIZE {
                    let val = qt.values[JPEG_NATURAL_ORDER[i] as usize];
                    output.push((val >> 8) as u8);
                    output.push((val & 0xFF) as u8);
                }
            }
        }

        Ok(())
    }

    /// Writes the frame header (SOF0, SOF1, or SOF2).
    ///
    /// Uses original dimensions (before MCU padding) so decoders can crop correctly.
    ///
    /// Marker selection:
    /// - SOF0 (0xC0): Baseline DCT - 8-bit quant tables only
    /// - SOF1 (0xC1): Extended sequential DCT - allows 16-bit quant tables
    /// - SOF2 (0xC2): Progressive DCT
    pub(crate) fn write_frame_header(&self, output: &mut Vec<u8>) -> Result<()> {
        self.write_frame_header_ex(output, false)
    }

    /// Writes the frame header with explicit extended mode control.
    ///
    /// # Arguments
    /// * `is_extended` - If true and mode is not progressive, use SOF1 instead of SOF0.
    ///   This is needed when any quantization table uses 16-bit precision.
    pub(crate) fn write_frame_header_ex(
        &self,
        output: &mut Vec<u8>,
        is_extended: bool,
    ) -> Result<()> {
        let marker = if self.mode == JpegMode::Progressive {
            MARKER_SOF2
        } else if is_extended {
            MARKER_SOF1 // Extended sequential DCT (allows 16-bit quant tables)
        } else {
            MARKER_SOF0 // Baseline DCT
        };

        output.push(0xFF);
        output.push(marker);

        let num_components = if self.pixel_format.is_grayscale() {
            1u8
        } else {
            3u8
        };

        let length = 8u16 + num_components as u16 * 3;
        output.push((length >> 8) as u8);
        output.push(length as u8);

        // Use original dimensions (before MCU padding) for the header.
        // Decoders will decode full MCUs but crop to these dimensions.
        let header_width = self.original_width.unwrap_or(self.width);
        let header_height = self.original_height.unwrap_or(self.height);

        output.push(8); // Sample precision
        output.push((header_height >> 8) as u8);
        output.push(header_height as u8);
        output.push((header_width >> 8) as u8);
        output.push(header_width as u8);
        output.push(num_components);

        if num_components == 1 {
            // Grayscale
            output.push(1); // Component ID
            output.push(0x11); // 1x1 sampling
            output.push(0); // Quant table 0
        } else {
            // Y component
            let (h_samp, v_samp) = match self.subsampling {
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
            // Cr uses table 2 when separate_chroma_tables=true (jpegli_set_distance)
            // Cr uses table 1 when separate_chroma_tables=false (jpeg_set_quality)
            output.push(if self.separate_chroma_tables { 2 } else { 1 });
        }

        Ok(())
    }

    /// Writes the frame header for XYB mode (RGB with B subsampling).
    #[allow(dead_code)] // Wrapper for callers that don't need extended mode
    pub(crate) fn write_frame_header_xyb(&self, output: &mut Vec<u8>) -> Result<()> {
        self.write_frame_header_xyb_ex(output, false)
    }

    /// Writes the frame header for XYB mode with explicit extended mode control.
    ///
    /// # Arguments
    /// * `is_extended` - If true and mode is not progressive, use SOF1 instead of SOF0.
    pub(crate) fn write_frame_header_xyb_ex(
        &self,
        output: &mut Vec<u8>,
        is_extended: bool,
    ) -> Result<()> {
        let marker = if self.mode == JpegMode::Progressive {
            MARKER_SOF2
        } else if is_extended {
            MARKER_SOF1 // Extended sequential DCT (allows 16-bit quant tables)
        } else {
            MARKER_SOF0 // Baseline DCT
        };

        output.push(0xFF);
        output.push(marker);

        // 3 components: R, G, B
        let length = 8u16 + 3 * 3; // 17 bytes
        output.push((length >> 8) as u8);
        output.push(length as u8);

        // Use original dimensions (before MCU padding) for the header
        let header_width = self.original_width.unwrap_or(self.width);
        let header_height = self.original_height.unwrap_or(self.height);

        output.push(8); // Sample precision
        output.push((header_height >> 8) as u8);
        output.push(header_height as u8);
        output.push((header_width >> 8) as u8);
        output.push(header_width as u8);
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

    /// Writes the frame header for XYB progressive mode.
    #[allow(dead_code)] // May be used in future progressive XYB support
    pub(crate) fn write_frame_header_xyb_progressive(&self, output: &mut Vec<u8>) -> Result<()> {
        // Progressive mode always uses SOF2, no need for is_extended
        self.write_frame_header_xyb_ex(output, false)
    }

    /// Writes standard Huffman tables in a single DHT segment.
    pub(crate) fn write_huffman_tables(&self, output: &mut Vec<u8>) -> Result<()> {
        use crate::huffman::{
            STD_AC_CHROMINANCE_BITS, STD_AC_CHROMINANCE_VALUES, STD_AC_LUMINANCE_BITS,
            STD_AC_LUMINANCE_VALUES, STD_DC_CHROMINANCE_BITS, STD_DC_CHROMINANCE_VALUES,
            STD_DC_LUMINANCE_BITS, STD_DC_LUMINANCE_VALUES,
        };

        // Write all 4 Huffman tables in a single DHT segment (like C++ jpegli)
        output.push(0xFF);
        output.push(MARKER_DHT);

        // Calculate total length
        let total_len = 2
            + (1 + 16 + STD_DC_LUMINANCE_VALUES.len())
            + (1 + 16 + STD_AC_LUMINANCE_VALUES.len())
            + (1 + 16 + STD_DC_CHROMINANCE_VALUES.len())
            + (1 + 16 + STD_AC_CHROMINANCE_VALUES.len());

        output.push((total_len >> 8) as u8);
        output.push(total_len as u8);

        // DC luminance (class 0, id 0)
        output.push(0x00);
        output.extend_from_slice(&STD_DC_LUMINANCE_BITS);
        output.extend_from_slice(&STD_DC_LUMINANCE_VALUES);

        // AC luminance (class 1, id 0)
        output.push(0x10);
        output.extend_from_slice(&STD_AC_LUMINANCE_BITS);
        output.extend_from_slice(&STD_AC_LUMINANCE_VALUES);

        // DC chrominance (class 0, id 1)
        output.push(0x01);
        output.extend_from_slice(&STD_DC_CHROMINANCE_BITS);
        output.extend_from_slice(&STD_DC_CHROMINANCE_VALUES);

        // AC chrominance (class 1, id 1)
        output.push(0x11);
        output.extend_from_slice(&STD_AC_CHROMINANCE_BITS);
        output.extend_from_slice(&STD_AC_CHROMINANCE_VALUES);

        Ok(())
    }

    /// Writes optimized Huffman tables.
    ///
    /// This is used when `optimize_huffman` is enabled to write the
    /// image-specific optimized tables to the DHT markers.
    pub(crate) fn write_huffman_tables_optimized(
        &self,
        output: &mut Vec<u8>,
        tables: &HuffmanTableSet,
    ) -> Result<()> {
        // Write all 4 Huffman tables in a single DHT segment (like C++ jpegli)
        // This saves 12 bytes compared to 4 separate segments
        output.push(0xFF);
        output.push(MARKER_DHT);

        // Calculate total length: 2 (length field) + 4 tables × (1 + 16 + values.len())
        let total_len = 2
            + (1 + 16 + tables.dc_luma.values.len())
            + (1 + 16 + tables.ac_luma.values.len())
            + (1 + 16 + tables.dc_chroma.values.len())
            + (1 + 16 + tables.ac_chroma.values.len());

        output.push((total_len >> 8) as u8);
        output.push(total_len as u8);

        // Write DC tables first (class 0), then AC tables (class 1)
        // This matches C++ jpegli and is expected by zune-jpeg

        // DC luminance (class 0, id 0)
        output.push(0x00);
        output.extend_from_slice(&tables.dc_luma.bits);
        output.extend_from_slice(&tables.dc_luma.values);

        // DC chrominance (class 0, id 1)
        output.push(0x01);
        output.extend_from_slice(&tables.dc_chroma.bits);
        output.extend_from_slice(&tables.dc_chroma.values);

        // AC luminance (class 1, id 0)
        output.push(0x10);
        output.extend_from_slice(&tables.ac_luma.bits);
        output.extend_from_slice(&tables.ac_luma.values);

        // AC chrominance (class 1, id 1)
        output.push(0x11);
        output.extend_from_slice(&tables.ac_chroma.bits);
        output.extend_from_slice(&tables.ac_chroma.values);

        Ok(())
    }

    /// Writes initial Huffman tables for progressive mode.
    ///
    /// Like C++ jpegli, this writes all DC tables plus up to `max_initial_ac` AC tables.
    /// Additional AC tables are emitted on-demand before the scans that need them.
    ///
    /// Returns the number of tables written (next_dht_index).
    pub(crate) fn write_huffman_tables_progressive_initial(
        &self,
        output: &mut Vec<u8>,
        tables: &[OptimizedTable],
        num_dc_tables: usize,
        max_initial_ac: usize,
    ) -> Result<usize> {
        // Count how many AC tables to include initially
        let num_ac_tables = tables.len().saturating_sub(num_dc_tables);
        let num_initial_ac = num_ac_tables.min(max_initial_ac);
        let num_initial_tables = num_dc_tables + num_initial_ac;

        if num_initial_tables == 0 {
            return Ok(0);
        }

        output.push(0xFF);
        output.push(MARKER_DHT);

        // Calculate total length
        let mut total_len = 2; // Length field itself
        for table in tables.iter().take(num_initial_tables) {
            total_len += 1 + 16 + table.values.len(); // class/id + bits + values
        }

        output.push((total_len >> 8) as u8);
        output.push(total_len as u8);

        // Write DC tables first (class 0)
        for (i, table) in tables.iter().take(num_dc_tables).enumerate() {
            let class_id = i as u8; // class 0, id = i
            output.push(class_id);
            output.extend_from_slice(&table.bits);
            output.extend_from_slice(&table.values);
        }

        // Write initial AC tables (class 1)
        for (i, table) in tables
            .iter()
            .skip(num_dc_tables)
            .take(num_initial_ac)
            .enumerate()
        {
            let class_id = 0x10 | (i as u8); // class 1, id = i
            output.push(class_id);
            output.extend_from_slice(&table.bits);
            output.extend_from_slice(&table.values);
        }

        Ok(num_initial_tables)
    }

    /// Writes a single AC Huffman table as a DHT marker.
    ///
    /// This is used for on-demand emission of AC tables in progressive mode.
    /// The table is written with class 1 and the specified slot ID.
    pub(crate) fn write_single_ac_table(
        &self,
        output: &mut Vec<u8>,
        table: &OptimizedTable,
        slot_id: usize,
    ) -> Result<()> {
        output.push(0xFF);
        output.push(MARKER_DHT);

        let total_len = 2 + 1 + 16 + table.values.len();
        output.push((total_len >> 8) as u8);
        output.push(total_len as u8);

        let class_id = 0x10 | (slot_id as u8); // class 1 (AC), id = slot_id
        output.push(class_id);
        output.extend_from_slice(&table.bits);
        output.extend_from_slice(&table.values);

        Ok(())
    }

    /// Writes restart interval.
    pub(crate) fn write_restart_interval(&self, output: &mut Vec<u8>) -> Result<()> {
        output.push(0xFF);
        output.push(MARKER_DRI);
        output.push(0x00);
        output.push(0x04); // Length
        output.push((self.restart_interval >> 8) as u8);
        output.push(self.restart_interval as u8);
        Ok(())
    }

    /// Writes scan header.
    pub(crate) fn write_scan_header(&self, output: &mut Vec<u8>) -> Result<()> {
        output.push(0xFF);
        output.push(MARKER_SOS);

        let num_components = if self.pixel_format.is_grayscale() {
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
    pub(crate) fn write_scan_header_xyb(&self, output: &mut Vec<u8>) -> Result<()> {
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

    /// Writes DHT markers for XYB optimized tables.
    pub(crate) fn write_huffman_tables_xyb_optimized(
        &self,
        output: &mut Vec<u8>,
        dc_table: &OptimizedTable,
        ac_table: &OptimizedTable,
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

    /// Writes SOS header for a progressive scan with slot ID support.
    ///
    /// This version uses `ac_slot_ids` to get the correct JPEG DHT slot for each AC table,
    /// which is needed when more than 4 AC tables are used (slot IDs cycle through 0-3).
    pub(crate) fn write_progressive_scan_header_with_slot_ids(
        &self,
        output: &mut Vec<u8>,
        scan_idx: usize,
        scan: &ProgressiveScan,
        _is_color: bool,
        context_config: &ContextConfig,
        context_map: &[usize],
        num_dc_tables: usize,
        ac_slot_ids: &[usize],
    ) -> Result<()> {
        output.push(0xFF);
        output.push(MARKER_SOS);

        let num_components = scan.components.len() as u8;
        let length = 6u16 + num_components as u16 * 2;
        output.push((length >> 8) as u8);
        output.push(length as u8);

        output.push(num_components);

        for (comp_in_scan, &comp_idx) in scan.components.iter().enumerate() {
            // Component ID: 1-based for YCbCr, or 'R','G','B' for XYB
            let comp_id = if self.use_xyb {
                match comp_idx {
                    0 => b'R', // 82
                    1 => b'G', // 71
                    2 => b'B', // 66
                    _ => comp_idx + 1,
                }
            } else {
                comp_idx + 1
            };
            output.push(comp_id);

            // DC table selector: use DC context (component index)
            let dc_context = context_config.dc_context(comp_idx as usize);
            let dc_table = context_map.get(dc_context).copied().unwrap_or(0);

            // AC table selector: use per-scan AC context and slot IDs
            let ac_context = context_config.ac_context(scan_idx, comp_in_scan);
            let cluster_idx = context_map
                .get(ac_context)
                .map(|&t| t.saturating_sub(num_dc_tables))
                .unwrap_or(0);
            // Get the actual JPEG slot ID from ac_slot_ids
            let ac_table = ac_slot_ids
                .get(cluster_idx)
                .copied()
                .unwrap_or(cluster_idx % 4);

            let table_selector = ((dc_table as u8) << 4) | (ac_table as u8);
            output.push(table_selector);
        }

        output.push(scan.ss); // Spectral selection start
        output.push(scan.se); // Spectral selection end
        output.push((scan.ah << 4) | scan.al); // Successive approximation

        Ok(())
    }
}
