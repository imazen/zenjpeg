//! Pull-based scanline decoder for streaming JPEG decoding.
//!
//! This module provides a scanline-by-scanline decoder that allows reading
//! JPEG images row by row without loading the entire image into memory.
//!
//! # Example
//! ```ignore
//! use jpegli::{Decoder, ImgRefMut};
//!
//! let mut reader = Decoder::new().scanline_reader(&jpeg_data)?;
//! let width = reader.width() as usize;
//! let height = reader.height() as usize;
//!
//! // Allocate output buffer
//! let mut pixels = vec![0u8; width * height * 3];
//!
//! // Read in chunks
//! let mut rows_read = 0;
//! while rows_read < height {
//!     let remaining = height - rows_read;
//!     let output = ImgRefMut::new(&mut pixels[rows_read * width * 3..], width, remaining);
//!     let count = reader.read_rows_rgb8(output)?;
//!     rows_read += count;
//! }
//! ```

use crate::alloc::try_alloc_uninitialized;
use crate::color::{ycbcr_planes_i16_to_rgb_u8, ycbcr_to_rgb};
use crate::consts::{DCT_BLOCK_SIZE, MAX_HUFFMAN_TABLES};
use crate::entropy::{EntropyDecoder, EntropyDecoderState};
use crate::error::{Error, Result};
use crate::huffman::HuffmanDecodeTable;
use crate::idct_int::idct_int_tiered;
use crate::quant::dequantize_unzigzag_i32_into;
use crate::types::{ColorSpace, Dimensions};
use imgref::ImgRefMut;

/// Information about the JPEG being decoded.
#[derive(Debug, Clone)]
pub struct ScanlineInfo {
    /// Image dimensions
    pub dimensions: Dimensions,
    /// Color space
    pub color_space: ColorSpace,
    /// Whether this is an XYB image
    pub is_xyb: bool,
}

/// Pull-based scanline reader for JPEG decoding.
///
/// Decodes JPEG images row by row, only decoding MCU rows as needed.
/// This minimizes memory usage and allows early processing of image data.
pub struct ScanlineReader<'a> {
    // Raw JPEG data
    data: &'a [u8],

    // Image dimensions
    width: u32,
    height: u32,
    num_components: u8,

    // MCU structure
    mcu_rows: usize,
    mcu_cols: usize,
    strip_width: usize,

    // Current position
    current_row: usize,     // Current output row (0 to height-1)
    current_mcu_row: usize, // Current MCU row being processed
    row_in_mcu: usize,      // Row within current MCU (0-7)
    mcu_row_decoded: bool,  // Whether current MCU row has been decoded

    // Strip buffers (one MCU row = 8 pixel rows)
    y_strip: Vec<i16>,
    cb_strip: Vec<i16>,
    cr_strip: Vec<i16>,

    // Quantization tables (copied, since we outlive the parser)
    quant_tables: [Option<[u16; DCT_BLOCK_SIZE]>; 4],
    quant_indices: [usize; 3], // Which quant table each component uses

    // Huffman tables (copied)
    dc_tables: [Option<HuffmanDecodeTable>; MAX_HUFFMAN_TABLES],
    ac_tables: [Option<HuffmanDecodeTable>; MAX_HUFFMAN_TABLES],
    table_mapping: [(usize, usize); 3], // (dc_table, ac_table) for each component

    // Entropy decoder state
    scan_data_start: usize, // Position where scan data begins
    decoder_state: Option<EntropyDecoderState>, // Saved state for resuming (None = start of scan)

    // Restart markers
    restart_interval: u16,
    mcu_count: u32,
    next_restart_num: u8,

    // Reusable buffer
    dequant_buf: [i32; DCT_BLOCK_SIZE],

    // Info
    is_xyb: bool,
}

impl<'a> ScanlineReader<'a> {
    /// Creates a new scanline reader from parsed JPEG data.
    ///
    /// This is called internally by `Decoder::scanline_reader()`.
    pub(crate) fn new(
        data: &'a [u8],
        width: u32,
        height: u32,
        num_components: u8,
        quant_tables: [Option<[u16; DCT_BLOCK_SIZE]>; 4],
        quant_indices: [usize; 3],
        dc_tables: [Option<HuffmanDecodeTable>; MAX_HUFFMAN_TABLES],
        ac_tables: [Option<HuffmanDecodeTable>; MAX_HUFFMAN_TABLES],
        table_mapping: [(usize, usize); 3],
        scan_data_start: usize,
        restart_interval: u16,
        is_xyb: bool,
    ) -> Result<Self> {
        let mcu_cols = (width as usize + 7) / 8;
        let mcu_rows = (height as usize + 7) / 8;
        let strip_width = mcu_cols * 8;
        let strip_size = strip_width * 8;

        // Allocate strip buffers
        let y_strip = unsafe { try_alloc_uninitialized(strip_size, "Y strip buffer")? };
        let cb_strip = unsafe { try_alloc_uninitialized(strip_size, "Cb strip buffer")? };
        let cr_strip = unsafe { try_alloc_uninitialized(strip_size, "Cr strip buffer")? };

        Ok(Self {
            data,
            width,
            height,
            num_components,
            mcu_rows,
            mcu_cols,
            strip_width,
            current_row: 0,
            current_mcu_row: 0,
            row_in_mcu: 0,
            mcu_row_decoded: false,
            y_strip,
            cb_strip,
            cr_strip,
            quant_tables,
            quant_indices,
            dc_tables,
            ac_tables,
            table_mapping,
            scan_data_start,
            decoder_state: None,
            restart_interval,
            mcu_count: 0,
            next_restart_num: 0,
            dequant_buf: [0i32; DCT_BLOCK_SIZE],
            is_xyb,
        })
    }

    /// Returns the image width.
    #[inline]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Returns the image height.
    #[inline]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Returns image info.
    pub fn info(&self) -> ScanlineInfo {
        ScanlineInfo {
            dimensions: Dimensions {
                width: self.width,
                height: self.height,
            },
            color_space: if self.num_components == 1 {
                ColorSpace::Grayscale
            } else {
                ColorSpace::YCbCr
            },
            is_xyb: self.is_xyb,
        }
    }

    /// Returns the current row position (0 to height-1).
    #[inline]
    pub fn current_row(&self) -> usize {
        self.current_row
    }

    /// Returns true if all rows have been read.
    #[inline]
    pub fn is_finished(&self) -> bool {
        self.current_row >= self.height as usize
    }

    /// Decodes the current MCU row into strip buffers.
    fn decode_mcu_row(&mut self) -> Result<()> {
        if self.mcu_row_decoded {
            return Ok(());
        }

        // Always create decoder from the full scan data slice
        let scan_data = &self.data[self.scan_data_start..];
        let mut decoder = EntropyDecoder::new(scan_data);

        // Set up Huffman tables first (before restoring state)
        for comp_idx in 0..self.num_components as usize {
            let (dc_idx, ac_idx) = self.table_mapping[comp_idx];

            if let Some(ref table) = self.dc_tables[dc_idx] {
                decoder.set_dc_table(dc_idx, table);
            }
            if let Some(ref table) = self.ac_tables[ac_idx] {
                decoder.set_ac_table(ac_idx, table);
            }
        }

        // Restore full decoder state if we have one (includes bit buffer position)
        if let Some(ref state) = self.decoder_state {
            decoder.restore_state(state.clone());
        }

        // Decode one MCU row
        for mcu_x in 0..self.mcu_cols {
            // Check for restart marker
            if self.restart_interval > 0
                && self.mcu_count > 0
                && self.mcu_count % self.restart_interval as u32 == 0
            {
                decoder.align_to_byte();
                decoder.read_restart_marker(self.next_restart_num)?;
                self.next_restart_num = (self.next_restart_num + 1) & 7;
                decoder.reset_dc();
            }

            // Decode each component's block
            for comp_idx in 0..self.num_components as usize {
                let (dc_idx, ac_idx) = self.table_mapping[comp_idx];
                let (coeffs, coeff_count) =
                    decoder.decode_block_with_count(comp_idx, dc_idx, ac_idx)?;

                let quant_idx = self.quant_indices[comp_idx];
                let quant = self.quant_tables[quant_idx]
                    .as_ref()
                    .ok_or(Error::InternalError {
                        reason: "missing quantization table",
                    })?;

                let strip = match comp_idx {
                    0 => &mut self.y_strip,
                    1 => &mut self.cb_strip,
                    _ => &mut self.cr_strip,
                };

                // Dequantize and IDCT
                dequantize_unzigzag_i32_into(&coeffs, quant, &mut self.dequant_buf);
                let dst_offset = mcu_x * 8;
                idct_int_tiered(
                    &mut self.dequant_buf,
                    &mut strip[dst_offset..],
                    self.strip_width,
                    coeff_count,
                );
            }

            self.mcu_count += 1;
        }

        // Save full state for next MCU row (includes bit buffer position)
        self.decoder_state = Some(decoder.save_state());
        self.mcu_row_decoded = true;

        Ok(())
    }

    /// Advances to the next MCU row.
    fn advance_mcu_row(&mut self) {
        self.current_mcu_row += 1;
        self.row_in_mcu = 0;
        self.mcu_row_decoded = false;
    }

    /// Read rows into an RGB8 buffer.
    ///
    /// Returns the number of rows actually written (may be less than requested
    /// if end of image is reached).
    pub fn read_rows_rgb8(&mut self, mut output: ImgRefMut<'_, u8>) -> Result<usize> {
        let max_rows = output.height();
        let width = self.width as usize;

        if output.width() < width * 3 {
            return Err(Error::InternalError {
                reason: "output buffer too narrow for RGB8",
            });
        }

        let mut rows_written = 0;

        while rows_written < max_rows && self.current_row < self.height as usize {
            // Ensure current MCU row is decoded
            self.decode_mcu_row()?;

            // Copy rows from strip to output
            let strip_row = self.row_in_mcu;
            let strip_offset = strip_row * self.strip_width;
            let cols = width.min(self.strip_width);

            let out_row = output.rows_mut().nth(rows_written).unwrap();

            // Convert YCbCr to RGB using the same function as the main decoder
            ycbcr_planes_i16_to_rgb_u8(
                &self.y_strip[strip_offset..strip_offset + cols],
                &self.cb_strip[strip_offset..strip_offset + cols],
                &self.cr_strip[strip_offset..strip_offset + cols],
                out_row,
            );

            rows_written += 1;
            self.current_row += 1;
            self.row_in_mcu += 1;

            // Move to next MCU row if needed
            if self.row_in_mcu >= 8 {
                self.advance_mcu_row();
            }
        }

        Ok(rows_written)
    }

    /// Read rows into an RGBX8 buffer (RGB with padding byte).
    ///
    /// Returns the number of rows actually written.
    pub fn read_rows_rgbx8(&mut self, mut output: ImgRefMut<'_, u8>) -> Result<usize> {
        let max_rows = output.height();
        let width = self.width as usize;

        if output.width() < width * 4 {
            return Err(Error::InternalError {
                reason: "output buffer too narrow for RGBX8",
            });
        }

        let mut rows_written = 0;

        while rows_written < max_rows && self.current_row < self.height as usize {
            self.decode_mcu_row()?;

            let strip_row = self.row_in_mcu;
            let strip_offset = strip_row * self.strip_width;
            let cols = width.min(self.strip_width);

            let out_row = output.rows_mut().nth(rows_written).unwrap();

            for x in 0..cols {
                let y = self.y_strip[strip_offset + x];
                let cb = self.cb_strip[strip_offset + x];
                let cr = self.cr_strip[strip_offset + x];
                let (r, g, b) = ycbcr_to_rgb(
                    y.clamp(0, 255) as u8,
                    cb.clamp(0, 255) as u8,
                    cr.clamp(0, 255) as u8,
                );
                out_row[x * 4] = r;
                out_row[x * 4 + 1] = g;
                out_row[x * 4 + 2] = b;
                out_row[x * 4 + 3] = 255; // Alpha/padding
            }

            rows_written += 1;
            self.current_row += 1;
            self.row_in_mcu += 1;

            if self.row_in_mcu >= 8 {
                self.advance_mcu_row();
            }
        }

        Ok(rows_written)
    }

    /// Read rows into a linear f32 RGBA buffer.
    ///
    /// Output is in linear light (not sRGB gamma).
    /// Returns the number of rows actually written.
    pub fn read_rows_rgba_f32(&mut self, mut output: ImgRefMut<'_, f32>) -> Result<usize> {
        let max_rows = output.height();
        let width = self.width as usize;

        if output.width() < width * 4 {
            return Err(Error::InternalError {
                reason: "output buffer too narrow for RGBA f32",
            });
        }

        let mut rows_written = 0;

        while rows_written < max_rows && self.current_row < self.height as usize {
            self.decode_mcu_row()?;

            let strip_row = self.row_in_mcu;
            let strip_offset = strip_row * self.strip_width;
            let cols = width.min(self.strip_width);

            let out_row = output.rows_mut().nth(rows_written).unwrap();

            for x in 0..cols {
                let y = self.y_strip[strip_offset + x];
                let cb = self.cb_strip[strip_offset + x];
                let cr = self.cr_strip[strip_offset + x];
                let (r, g, b) = ycbcr_to_rgb(
                    y.clamp(0, 255) as u8,
                    cb.clamp(0, 255) as u8,
                    cr.clamp(0, 255) as u8,
                );

                // Convert sRGB u8 to linear f32
                out_row[x * 4] = srgb_to_linear(r);
                out_row[x * 4 + 1] = srgb_to_linear(g);
                out_row[x * 4 + 2] = srgb_to_linear(b);
                out_row[x * 4 + 3] = 1.0; // Alpha
            }

            rows_written += 1;
            self.current_row += 1;
            self.row_in_mcu += 1;

            if self.row_in_mcu >= 8 {
                self.advance_mcu_row();
            }
        }

        Ok(rows_written)
    }

    /// Read rows into separate YCbCr f32 planes.
    ///
    /// Each plane receives normalized values in range [0, 1] for Y, [-0.5, 0.5] for Cb/Cr.
    /// Returns the number of rows actually written.
    pub fn read_rows_ycbcr_planes(
        &mut self,
        y_plane: &mut [f32],
        cb_plane: &mut [f32],
        cr_plane: &mut [f32],
        stride: usize,
        max_rows: usize,
    ) -> Result<usize> {
        let width = self.width as usize;

        if stride < width {
            return Err(Error::InternalError {
                reason: "stride too small for image width",
            });
        }

        let mut rows_written = 0;

        while rows_written < max_rows && self.current_row < self.height as usize {
            self.decode_mcu_row()?;

            let strip_row = self.row_in_mcu;
            let strip_offset = strip_row * self.strip_width;
            let cols = width.min(self.strip_width);

            let out_offset = rows_written * stride;

            for x in 0..cols {
                // Normalize: Y from [0, 255] to [0, 1]
                // Cb/Cr from [0, 255] (centered at 128) to [-0.5, 0.5]
                y_plane[out_offset + x] = self.y_strip[strip_offset + x] as f32 / 255.0;
                cb_plane[out_offset + x] = (self.cb_strip[strip_offset + x] as f32 - 128.0) / 255.0;
                cr_plane[out_offset + x] = (self.cr_strip[strip_offset + x] as f32 - 128.0) / 255.0;
            }

            rows_written += 1;
            self.current_row += 1;
            self.row_in_mcu += 1;

            if self.row_in_mcu >= 8 {
                self.advance_mcu_row();
            }
        }

        Ok(rows_written)
    }
}

/// Convert sRGB u8 to linear f32.
#[inline]
fn srgb_to_linear(srgb: u8) -> f32 {
    let s = srgb as f32 / 255.0;
    if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_srgb_to_linear() {
        // Black
        assert!((srgb_to_linear(0) - 0.0).abs() < 1e-6);
        // White
        assert!((srgb_to_linear(255) - 1.0).abs() < 1e-6);
        // Mid-gray (sRGB 128 ≈ linear 0.2159)
        assert!((srgb_to_linear(128) - 0.2159).abs() < 0.01);
    }

    #[test]
    fn test_scanline_reader_rgb8() {
        use crate::{Decoder, Quality, StreamingEncoder};

        // Create test image - 64x48 for multiple MCU rows
        let width = 64u32;
        let height = 48u32;
        let mut pixels = vec![0u8; (width * height * 3) as usize];
        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * 3) as usize;
                pixels[idx] = (x * 4) as u8; // R gradient
                pixels[idx + 1] = (y * 5) as u8; // G gradient
                pixels[idx + 2] = 128; // B constant
            }
        }

        // Encode as baseline 4:4:4 (default)
        let encoder = StreamingEncoder::new(width, height).quality(Quality::from_quality(95.0));
        let jpeg = encoder.encode_all(&pixels).expect("encode failed");

        // Decode normally for comparison
        let decoder = Decoder::new();
        let decoded = decoder.decode(&jpeg).expect("decode failed");

        // Decode via scanline reader
        let mut reader = decoder
            .scanline_reader(&jpeg)
            .expect("scanline_reader failed");
        assert_eq!(reader.width(), width);
        assert_eq!(reader.height(), height);

        let mut scanline_pixels = vec![0u8; (width * height * 3) as usize];

        // Read all rows
        let mut total_rows = 0;
        while !reader.is_finished() {
            let remaining = height as usize - total_rows;
            let stride = (width * 3) as usize;
            let buf_start = total_rows * stride;
            let output =
                imgref::ImgRefMut::new(&mut scanline_pixels[buf_start..], stride, remaining);
            let rows = reader
                .read_rows_rgb8(output)
                .expect("read_rows_rgb8 failed");
            total_rows += rows;
        }

        assert_eq!(total_rows, height as usize);

        // Compare outputs - should be identical
        assert_eq!(
            scanline_pixels.len(),
            decoded.data.len(),
            "output size mismatch"
        );
        assert_eq!(
            scanline_pixels, decoded.data,
            "scanline reader output differs from regular decode"
        );
    }

    #[test]
    fn test_scanline_reader_partial_reads() {
        use crate::{Decoder, Quality, StreamingEncoder};

        // Create test image - 32x32
        let width = 32u32;
        let height = 32u32;
        let mut pixels = vec![0u8; (width * height * 3) as usize];
        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * 3) as usize;
                pixels[idx] = ((x + y) * 4) as u8;
                pixels[idx + 1] = ((x * 2 + y) % 256) as u8;
                pixels[idx + 2] = ((y * 2 + x) % 256) as u8;
            }
        }

        let encoder = StreamingEncoder::new(width, height).quality(Quality::from_quality(90.0));
        let jpeg = encoder.encode_all(&pixels).expect("encode failed");

        let decoder = Decoder::new();
        let decoded = decoder.decode(&jpeg).expect("decode failed");

        // Read in small chunks (3 rows at a time)
        let mut reader = decoder
            .scanline_reader(&jpeg)
            .expect("scanline_reader failed");
        let mut scanline_pixels = vec![0u8; (width * height * 3) as usize];
        let stride = (width * 3) as usize;

        let mut total_rows = 0;
        while !reader.is_finished() {
            let chunk_size = 3; // Read 3 rows at a time
            let rows_to_read = chunk_size.min(height as usize - total_rows);
            let buf_start = total_rows * stride;
            let output =
                imgref::ImgRefMut::new(&mut scanline_pixels[buf_start..], stride, rows_to_read);
            let rows = reader.read_rows_rgb8(output).expect("read failed");
            assert!(rows > 0 || reader.is_finished());
            total_rows += rows;
        }

        assert_eq!(total_rows, height as usize);
        assert_eq!(scanline_pixels, decoded.data);
    }

    #[test]
    fn test_scanline_reader_rgbx8() {
        use crate::{Decoder, Quality, StreamingEncoder};

        let width = 24u32;
        let height = 24u32;
        let mut pixels = vec![0u8; (width * height * 3) as usize];
        for i in 0..pixels.len() {
            pixels[i] = ((i * 7) % 256) as u8;
        }

        let encoder = StreamingEncoder::new(width, height).quality(Quality::from_quality(85.0));
        let jpeg = encoder.encode_all(&pixels).expect("encode failed");

        let decoder = Decoder::new();
        let decoded = decoder.decode(&jpeg).expect("decode failed");

        let mut reader = decoder
            .scanline_reader(&jpeg)
            .expect("scanline_reader failed");
        let mut rgbx_pixels = vec![0u8; (width * height * 4) as usize];
        let stride = (width * 4) as usize;

        let mut total_rows = 0;
        while !reader.is_finished() {
            let remaining = height as usize - total_rows;
            let buf_start = total_rows * stride;
            let output = imgref::ImgRefMut::new(&mut rgbx_pixels[buf_start..], stride, remaining);
            let rows = reader.read_rows_rgbx8(output).expect("read failed");
            total_rows += rows;
        }

        // Verify RGBX matches RGB with alpha=255
        for y in 0..height as usize {
            for x in 0..width as usize {
                let rgb_idx = (y * width as usize + x) * 3;
                let rgbx_idx = (y * width as usize + x) * 4;
                assert_eq!(rgbx_pixels[rgbx_idx], decoded.data[rgb_idx], "R mismatch");
                assert_eq!(
                    rgbx_pixels[rgbx_idx + 1],
                    decoded.data[rgb_idx + 1],
                    "G mismatch"
                );
                assert_eq!(
                    rgbx_pixels[rgbx_idx + 2],
                    decoded.data[rgb_idx + 2],
                    "B mismatch"
                );
                assert_eq!(rgbx_pixels[rgbx_idx + 3], 255, "Alpha should be 255");
            }
        }
    }

    #[test]
    fn test_scanline_reader_rgba_f32() {
        use crate::{Decoder, Quality, StreamingEncoder};

        let width = 16u32;
        let height = 16u32;
        let mut pixels = vec![0u8; (width * height * 3) as usize];
        for i in 0..pixels.len() {
            pixels[i] = ((i * 11) % 256) as u8;
        }

        let encoder = StreamingEncoder::new(width, height).quality(Quality::from_quality(90.0));
        let jpeg = encoder.encode_all(&pixels).expect("encode failed");

        let decoder = Decoder::new();
        let decoded = decoder.decode(&jpeg).expect("decode failed");

        let mut reader = decoder
            .scanline_reader(&jpeg)
            .expect("scanline_reader failed");
        let mut rgba_pixels = vec![0.0f32; (width * height * 4) as usize];
        let stride = (width * 4) as usize;

        let mut total_rows = 0;
        while !reader.is_finished() {
            let remaining = height as usize - total_rows;
            let buf_start = total_rows * stride;
            let output = imgref::ImgRefMut::new(&mut rgba_pixels[buf_start..], stride, remaining);
            let rows = reader.read_rows_rgba_f32(output).expect("read failed");
            total_rows += rows;
        }

        // Verify values are in valid range
        for (i, &val) in rgba_pixels.iter().enumerate() {
            if i % 4 == 3 {
                // Alpha channel
                assert!(
                    (val - 1.0).abs() < 1e-6,
                    "Alpha at {} should be 1.0, got {}",
                    i,
                    val
                );
            } else {
                // RGB channels should be in [0, 1] range
                assert!(
                    (0.0..=1.0).contains(&val),
                    "Value at {} should be in [0,1], got {}",
                    i,
                    val
                );
            }
        }

        // Verify RGB matches (converting back from linear)
        for y in 0..height as usize {
            for x in 0..width as usize {
                let rgb_idx = (y * width as usize + x) * 3;
                let rgba_idx = (y * width as usize + x) * 4;

                // Compare each channel (allow some tolerance due to linear conversion)
                for c in 0..3 {
                    let expected_linear = srgb_to_linear(decoded.data[rgb_idx + c]);
                    let actual_linear = rgba_pixels[rgba_idx + c];
                    assert!(
                        (expected_linear - actual_linear).abs() < 0.01,
                        "Linear mismatch at ({},{}) channel {}: expected {}, got {}",
                        x,
                        y,
                        c,
                        expected_linear,
                        actual_linear
                    );
                }
            }
        }
    }

    #[test]
    fn test_scanline_reader_ycbcr_planes() {
        use crate::{Decoder, Quality, StreamingEncoder};

        let width = 32u32;
        let height = 24u32;
        let mut pixels = vec![0u8; (width * height * 3) as usize];
        for i in 0..pixels.len() {
            pixels[i] = ((i * 13) % 256) as u8;
        }

        let encoder = StreamingEncoder::new(width, height).quality(Quality::from_quality(90.0));
        let jpeg = encoder.encode_all(&pixels).expect("encode failed");

        let decoder = Decoder::new();

        let mut reader = decoder
            .scanline_reader(&jpeg)
            .expect("scanline_reader failed");
        let plane_size = (width * height) as usize;
        let mut y_plane = vec![0.0f32; plane_size];
        let mut cb_plane = vec![0.0f32; plane_size];
        let mut cr_plane = vec![0.0f32; plane_size];

        let mut total_rows = 0;
        while !reader.is_finished() {
            let remaining = height as usize - total_rows;
            let offset = total_rows * width as usize;
            let rows = reader
                .read_rows_ycbcr_planes(
                    &mut y_plane[offset..],
                    &mut cb_plane[offset..],
                    &mut cr_plane[offset..],
                    width as usize,
                    remaining,
                )
                .expect("read failed");
            total_rows += rows;
        }

        // Verify Y values are in [0, 1] and Cb/Cr in [-0.5, 0.5]
        for i in 0..plane_size {
            assert!(
                (0.0..=1.0).contains(&y_plane[i]),
                "Y[{}] = {} out of range",
                i,
                y_plane[i]
            );
            assert!(
                (-0.6..=0.6).contains(&cb_plane[i]),
                "Cb[{}] = {} out of range",
                i,
                cb_plane[i]
            );
            assert!(
                (-0.6..=0.6).contains(&cr_plane[i]),
                "Cr[{}] = {} out of range",
                i,
                cr_plane[i]
            );
        }
    }

    #[test]
    fn test_scanline_reader_non_mcu_aligned() {
        use crate::{Decoder, Quality, StreamingEncoder};

        // Non-MCU-aligned dimensions (not multiples of 8)
        let width = 37u32;
        let height = 29u32;
        let mut pixels = vec![0u8; (width * height * 3) as usize];
        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * 3) as usize;
                pixels[idx] = (x * 7) as u8;
                pixels[idx + 1] = (y * 9) as u8;
                pixels[idx + 2] = ((x + y) * 3) as u8;
            }
        }

        let encoder = StreamingEncoder::new(width, height).quality(Quality::from_quality(90.0));
        let jpeg = encoder.encode_all(&pixels).expect("encode failed");

        let decoder = Decoder::new();
        let decoded = decoder.decode(&jpeg).expect("decode failed");

        let mut reader = decoder
            .scanline_reader(&jpeg)
            .expect("scanline_reader failed");
        let mut scanline_pixels = vec![0u8; (width * height * 3) as usize];
        let stride = (width * 3) as usize;

        let mut total_rows = 0;
        while !reader.is_finished() {
            let remaining = height as usize - total_rows;
            let buf_start = total_rows * stride;
            let output =
                imgref::ImgRefMut::new(&mut scanline_pixels[buf_start..], stride, remaining);
            let rows = reader.read_rows_rgb8(output).expect("read failed");
            total_rows += rows;
        }

        assert_eq!(total_rows, height as usize);
        assert_eq!(scanline_pixels, decoded.data);
    }
}
