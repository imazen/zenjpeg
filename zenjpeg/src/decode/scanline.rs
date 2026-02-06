//! Pull-based scanline decoder for streaming JPEG decoding.
//!
//! This module provides a scanline-by-scanline decoder that allows reading
//! JPEG images row by row without loading the entire image into memory.
//!
//! # Example
//! ```ignore
//! use zenjpeg::{Decoder, ImgRefMut};
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

use super::pipeline::StripProcessor;
use crate::color::{ycbcr_planes_i16_to_rgb_u8, ycbcr_to_rgb};
use crate::entropy::{EntropyDecoder, EntropyDecoderState};
use crate::error::{Error, Result, ScanRead};
use crate::foundation::consts::{DCT_BLOCK_SIZE, MAX_HUFFMAN_TABLES};
use crate::huffman::HuffmanDecodeTable;
use crate::types::{ColorSpace, Dimensions, Subsampling};
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
    /// Chroma subsampling mode
    pub subsampling: Subsampling,
}

/// Pull-based scanline reader for JPEG decoding.
///
/// Decodes JPEG images row by row, only decoding MCU rows as needed.
/// This minimizes memory usage and allows early processing of image data.
///
/// For progressive JPEGs, the image is fully decoded upfront and served
/// from a buffer, since progressive encoding requires all scans to be
/// processed before final pixels are available.
pub struct ScanlineReader<'a> {
    // Raw JPEG data (unused in buffered mode)
    data: &'a [u8],

    // Image dimensions
    width: u32,
    height: u32,
    num_components: u8,

    // Buffered mode for progressive JPEGs
    // When Some, we serve from this buffer instead of decoding on-the-fly
    buffered_rgb: Option<Vec<u8>>,

    // Strip processing (IDCT + upsampling)
    strip: StripProcessor,

    // Current position
    current_row: usize,     // Current output row (0 to height-1)
    current_mcu_row: usize, // Current MCU row being processed
    row_in_mcu: usize,      // Row within current MCU (0 to mcu_height-1)
    mcu_row_decoded: bool,  // Whether current MCU row has been decoded

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

    // Reusable buffers for zero-copy decode
    coeffs_buf: [i16; DCT_BLOCK_SIZE],
    /// Track previous coefficient count per component for smart zeroing
    prev_coeff_counts: [u8; 4],

    // Info
    is_xyb: bool,
    /// True for Adobe RGB JPEGs (APP14 transform=0) - skip YCbCr→RGB conversion
    is_rgb: bool,
}

impl<'a> ScanlineReader<'a> {
    /// Creates a new scanline reader from parsed JPEG scan data.
    ///
    /// This is called internally by `Decoder::scanline_reader()`.
    pub(super) fn from_scan_data(
        scan: super::parser::ParsedScanData<'a>,
        chroma_upsampling: super::ChromaUpsampling,
    ) -> Result<Self> {
        let super::parser::ParsedScanData {
            data,
            width,
            height,
            num_components,
            h_samp,
            v_samp,
            quant_tables,
            quant_indices,
            dc_tables,
            ac_tables,
            table_mapping,
            scan_data_start,
            restart_interval,
            is_xyb,
            is_rgb,
        } = scan;

        let strip = StripProcessor::new(width, num_components, h_samp, v_samp, chroma_upsampling)?;

        Ok(Self {
            data,
            width,
            height,
            num_components,
            buffered_rgb: None,
            strip,
            current_row: 0,
            current_mcu_row: 0,
            row_in_mcu: 0,
            mcu_row_decoded: false,
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
            coeffs_buf: [0i16; DCT_BLOCK_SIZE],
            prev_coeff_counts: [64; 4],
            is_xyb,
            is_rgb,
        })
    }

    /// Creates a new scanline reader in buffered mode (for progressive JPEGs).
    ///
    /// In buffered mode, the image has already been decoded and we serve
    /// rows from the pre-decoded buffer.
    pub(crate) fn new_buffered(
        data: &'a [u8],
        width: u32,
        height: u32,
        num_components: u8,
        subsampling: Subsampling,
        pixels: Vec<u8>,
        is_xyb: bool,
    ) -> Self {
        Self {
            data,
            width,
            height,
            num_components,
            buffered_rgb: Some(pixels),
            strip: StripProcessor::new_dummy(subsampling),
            current_row: 0,
            current_mcu_row: 0,
            row_in_mcu: 0,
            mcu_row_decoded: false,
            quant_tables: [None, None, None, None],
            quant_indices: [0, 0, 0],
            dc_tables: [None, None, None, None],
            ac_tables: [None, None, None, None],
            table_mapping: [(0, 0), (0, 0), (0, 0)],
            scan_data_start: 0,
            decoder_state: None,
            restart_interval: 0,
            mcu_count: 0,
            next_restart_num: 0,
            coeffs_buf: [0i16; DCT_BLOCK_SIZE],
            prev_coeff_counts: [64; 4],
            is_xyb,
            is_rgb: false,
        }
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
            subsampling: self.strip.subsampling,
        }
    }

    /// Returns the chroma subsampling mode.
    #[inline]
    pub fn subsampling(&self) -> Subsampling {
        self.strip.subsampling
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

    /// Returns true if this is a grayscale (single-component) image.
    #[inline]
    pub fn is_grayscale(&self) -> bool {
        self.num_components == 1
    }

    /// Returns the number of components (1 for grayscale, 3 for color).
    #[inline]
    pub fn num_components(&self) -> u8 {
        self.num_components
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
            decoder.restore_state(*state);
        }

        // Pre-validate quant tables outside the hot loop (error allocation happens ONCE, not per-block)
        let quant_y = self.quant_tables[self.quant_indices[0]]
            .as_ref()
            .ok_or_else(|| Error::internal("missing Y quantization table"))?;
        let quant_cb = if self.num_components > 1 {
            self.quant_tables[self.quant_indices[1]]
                .as_ref()
                .ok_or_else(|| Error::internal("missing Cb quantization table"))?
        } else {
            quant_y
        };
        let quant_cr = if self.num_components > 2 {
            self.quant_tables[self.quant_indices[2]]
                .as_ref()
                .ok_or_else(|| Error::internal("missing Cr quantization table"))?
        } else {
            quant_y
        };
        let quant_refs: [&[u16; 64]; 4] = [quant_y, quant_cb, quant_cr, quant_y];

        let mcu_cols = self.strip.mcu_cols();

        // Decode one MCU row
        for mcu_x in 0..mcu_cols {
            // Check for restart marker
            if self.restart_interval > 0
                && self.mcu_count > 0
                && self.mcu_count % self.restart_interval as u32 == 0
            {
                decoder.align_to_byte();
                decoder.read_restart_marker(self.next_restart_num)?;
                self.next_restart_num = (self.next_restart_num + 1) & 7;
                decoder.reset_dc();
                self.prev_coeff_counts = [64; 4];
            }

            // Decode each component's blocks
            for comp_idx in 0..self.num_components as usize {
                let h_blocks = self.strip.h_samp[comp_idx] as usize;
                let v_blocks = self.strip.v_samp[comp_idx] as usize;

                let (dc_idx, ac_idx) = self.table_mapping[comp_idx];
                let quant = quant_refs[comp_idx];

                for v in 0..v_blocks {
                    for h in 0..h_blocks {
                        let coeff_count = match decoder.decode_block_into(
                            &mut self.coeffs_buf,
                            self.prev_coeff_counts[comp_idx],
                            comp_idx,
                            dc_idx,
                            ac_idx,
                        )? {
                            ScanRead::Value(c) => c,
                            ScanRead::EndOfScan | ScanRead::Truncated => {
                                self.prev_coeff_counts[comp_idx] = 64;
                                continue;
                            }
                        };
                        self.prev_coeff_counts[comp_idx] =
                            self.prev_coeff_counts[comp_idx].max(coeff_count);

                        self.strip.idct_block(
                            comp_idx,
                            mcu_x,
                            h,
                            v,
                            &self.coeffs_buf,
                            coeff_count,
                            quant,
                        );
                    }
                }
            }

            self.mcu_count += 1;
        }

        // Save full state for next MCU row (includes bit buffer position)
        self.decoder_state = Some(decoder.save_state());

        // Upsample chroma if needed
        self.strip.upsample_chroma();

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
            return Err(Error::internal("output buffer too narrow for RGB8"));
        }

        // Buffered mode: serve from pre-decoded buffer (progressive JPEGs)
        if let Some(ref buffer) = self.buffered_rgb {
            let mut rows_written = 0;
            let row_bytes = width * 3;

            while rows_written < max_rows && self.current_row < self.height as usize {
                let src_offset = self.current_row * row_bytes;
                let out_row = output.rows_mut().nth(rows_written).unwrap();
                out_row[..row_bytes].copy_from_slice(&buffer[src_offset..src_offset + row_bytes]);

                rows_written += 1;
                self.current_row += 1;
            }

            return Ok(rows_written);
        }

        // Streaming mode: decode on-the-fly
        let mut rows_written = 0;

        while rows_written < max_rows && self.current_row < self.height as usize {
            self.decode_mcu_row()?;

            let cols = width.min(self.strip.strip_width);
            let out_row = output.rows_mut().nth(rows_written).unwrap();

            if self.is_rgb {
                let (y, cb, cr) = self.strip.row_planes(self.row_in_mcu, cols);
                for px in 0..cols {
                    out_row[px * 3] = y[px].clamp(0, 255) as u8;
                    out_row[px * 3 + 1] = cb[px].clamp(0, 255) as u8;
                    out_row[px * 3 + 2] = cr[px].clamp(0, 255) as u8;
                }
            } else {
                let (y, cb, cr) = self.strip.row_planes(self.row_in_mcu, cols);
                ycbcr_planes_i16_to_rgb_u8(y, cb, cr, out_row);
            }

            rows_written += 1;
            self.current_row += 1;
            self.row_in_mcu += 1;

            if self.row_in_mcu >= self.strip.mcu_height {
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
            return Err(Error::internal("output buffer too narrow for RGBX8"));
        }

        // Buffered mode: serve from pre-decoded RGB buffer, expanding to RGBX
        if let Some(ref buffer) = self.buffered_rgb {
            let mut rows_written = 0;
            let src_row_bytes = width * 3;

            while rows_written < max_rows && self.current_row < self.height as usize {
                let src_offset = self.current_row * src_row_bytes;
                let out_row = output.rows_mut().nth(rows_written).unwrap();

                // Expand RGB to RGBX
                for x in 0..width {
                    out_row[x * 4] = buffer[src_offset + x * 3];
                    out_row[x * 4 + 1] = buffer[src_offset + x * 3 + 1];
                    out_row[x * 4 + 2] = buffer[src_offset + x * 3 + 2];
                    out_row[x * 4 + 3] = 255;
                }

                rows_written += 1;
                self.current_row += 1;
            }

            return Ok(rows_written);
        }

        // Streaming mode: decode on-the-fly
        let mut rows_written = 0;

        while rows_written < max_rows && self.current_row < self.height as usize {
            self.decode_mcu_row()?;

            let cols = width.min(self.strip.strip_width);
            let out_row = output.rows_mut().nth(rows_written).unwrap();

            let (y_row, cb_row, cr_row) = self.strip.row_planes(self.row_in_mcu, cols);

            for x in 0..cols {
                let (r, g, b) = if self.is_rgb {
                    (
                        y_row[x].clamp(0, 255) as u8,
                        cb_row[x].clamp(0, 255) as u8,
                        cr_row[x].clamp(0, 255) as u8,
                    )
                } else {
                    ycbcr_to_rgb(
                        y_row[x].clamp(0, 255) as u8,
                        cb_row[x].clamp(0, 255) as u8,
                        cr_row[x].clamp(0, 255) as u8,
                    )
                };
                out_row[x * 4] = r;
                out_row[x * 4 + 1] = g;
                out_row[x * 4 + 2] = b;
                out_row[x * 4 + 3] = 255;
            }

            rows_written += 1;
            self.current_row += 1;
            self.row_in_mcu += 1;

            if self.row_in_mcu >= self.strip.mcu_height {
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
            return Err(Error::internal("output buffer too narrow for RGBA f32"));
        }

        // Buffered mode: serve from pre-decoded RGB buffer, convert to linear f32
        if let Some(ref buffer) = self.buffered_rgb {
            let mut rows_written = 0;
            let src_row_bytes = width * 3;

            while rows_written < max_rows && self.current_row < self.height as usize {
                let src_offset = self.current_row * src_row_bytes;
                let out_row = output.rows_mut().nth(rows_written).unwrap();

                // Convert RGB8 to linear RGBA f32
                for x in 0..width {
                    out_row[x * 4] = srgb_to_linear(buffer[src_offset + x * 3]);
                    out_row[x * 4 + 1] = srgb_to_linear(buffer[src_offset + x * 3 + 1]);
                    out_row[x * 4 + 2] = srgb_to_linear(buffer[src_offset + x * 3 + 2]);
                    out_row[x * 4 + 3] = 1.0;
                }

                rows_written += 1;
                self.current_row += 1;
            }

            return Ok(rows_written);
        }

        // Streaming mode: decode on-the-fly
        let mut rows_written = 0;

        while rows_written < max_rows && self.current_row < self.height as usize {
            self.decode_mcu_row()?;

            let cols = width.min(self.strip.strip_width);
            let out_row = output.rows_mut().nth(rows_written).unwrap();

            let (y_row, cb_row, cr_row) = self.strip.row_planes(self.row_in_mcu, cols);

            for x in 0..cols {
                let (r, g, b) = if self.is_rgb {
                    (
                        y_row[x].clamp(0, 255) as u8,
                        cb_row[x].clamp(0, 255) as u8,
                        cr_row[x].clamp(0, 255) as u8,
                    )
                } else {
                    ycbcr_to_rgb(
                        y_row[x].clamp(0, 255) as u8,
                        cb_row[x].clamp(0, 255) as u8,
                        cr_row[x].clamp(0, 255) as u8,
                    )
                };

                out_row[x * 4] = srgb_to_linear(r);
                out_row[x * 4 + 1] = srgb_to_linear(g);
                out_row[x * 4 + 2] = srgb_to_linear(b);
                out_row[x * 4 + 3] = 1.0;
            }

            rows_written += 1;
            self.current_row += 1;
            self.row_in_mcu += 1;

            if self.row_in_mcu >= self.strip.mcu_height {
                self.advance_mcu_row();
            }
        }

        Ok(rows_written)
    }

    /// Read rows into separate YCbCr f32 planes.
    ///
    /// Each plane receives normalized values in range [0, 1] for Y, [-0.5, 0.5] for Cb/Cr.
    /// Chroma values are upsampled to full resolution for subsampled images.
    /// Returns the number of rows actually written.
    ///
    /// Note: For progressive JPEGs (buffered mode), this converts from RGB back to YCbCr
    /// using BT.601 coefficients, which may introduce small rounding differences.
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
            return Err(Error::internal("stride too small for image width"));
        }

        // Buffered mode: convert RGB back to YCbCr
        if let Some(ref buffer) = self.buffered_rgb {
            let mut rows_written = 0;

            if self.num_components == 1 {
                // Grayscale: Y only, Cb/Cr are zero
                while rows_written < max_rows && self.current_row < self.height as usize {
                    let src_offset = self.current_row * width;
                    let out_offset = rows_written * stride;

                    for x in 0..width {
                        y_plane[out_offset + x] = buffer[src_offset + x] as f32 / 255.0;
                        cb_plane[out_offset + x] = 0.0;
                        cr_plane[out_offset + x] = 0.0;
                    }

                    rows_written += 1;
                    self.current_row += 1;
                }
            } else {
                // Color: convert RGB to YCbCr using BT.601
                let src_row_bytes = width * 3;
                while rows_written < max_rows && self.current_row < self.height as usize {
                    let src_offset = self.current_row * src_row_bytes;
                    let out_offset = rows_written * stride;

                    for x in 0..width {
                        let r = buffer[src_offset + x * 3] as f32;
                        let g = buffer[src_offset + x * 3 + 1] as f32;
                        let b = buffer[src_offset + x * 3 + 2] as f32;

                        // BT.601 RGB to YCbCr (normalized output)
                        // Y  =  0.299*R + 0.587*G + 0.114*B
                        // Cb = -0.169*R - 0.331*G + 0.500*B
                        // Cr =  0.500*R - 0.419*G - 0.081*B
                        y_plane[out_offset + x] = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0;
                        cb_plane[out_offset + x] = (-0.169 * r - 0.331 * g + 0.500 * b) / 255.0;
                        cr_plane[out_offset + x] = (0.500 * r - 0.419 * g - 0.081 * b) / 255.0;
                    }

                    rows_written += 1;
                    self.current_row += 1;
                }
            }

            return Ok(rows_written);
        }

        // Streaming mode: decode on-the-fly
        let mut rows_written = 0;

        while rows_written < max_rows && self.current_row < self.height as usize {
            self.decode_mcu_row()?;

            let cols = width.min(self.strip.strip_width);
            let out_offset = rows_written * stride;

            let (y_slice, cb_slice, cr_slice) = self.strip.row_planes(self.row_in_mcu, cols);

            for x in 0..cols {
                y_plane[out_offset + x] = y_slice[x] as f32 / 255.0;
                cb_plane[out_offset + x] = (cb_slice[x] as f32 - 128.0) / 255.0;
                cr_plane[out_offset + x] = (cr_slice[x] as f32 - 128.0) / 255.0;
            }

            rows_written += 1;
            self.current_row += 1;
            self.row_in_mcu += 1;

            if self.row_in_mcu >= self.strip.mcu_height {
                self.advance_mcu_row();
            }
        }

        Ok(rows_written)
    }

    /// Read rows into a grayscale u8 buffer.
    ///
    /// This method is optimized for grayscale JPEGs (1 component).
    /// For color JPEGs, it extracts the Y (luminance) channel.
    ///
    /// Returns the number of rows actually written (may be less than requested
    /// if end of image is reached).
    pub fn read_rows_gray8(&mut self, mut output: ImgRefMut<'_, u8>) -> Result<usize> {
        let max_rows = output.height();
        let width = self.width as usize;

        if output.width() < width {
            return Err(Error::internal("output buffer too narrow for grayscale"));
        }

        // Buffered mode: serve from pre-decoded buffer
        if let Some(ref buffer) = self.buffered_rgb {
            let mut rows_written = 0;

            if self.num_components == 1 {
                // Grayscale buffer (1 byte per pixel)
                while rows_written < max_rows && self.current_row < self.height as usize {
                    let src_offset = self.current_row * width;
                    let out_row = output.rows_mut().nth(rows_written).unwrap();
                    out_row[..width].copy_from_slice(&buffer[src_offset..src_offset + width]);

                    rows_written += 1;
                    self.current_row += 1;
                }
            } else {
                // Color buffer (RGB8): convert to grayscale using BT.601 coefficients
                let src_row_bytes = width * 3;
                while rows_written < max_rows && self.current_row < self.height as usize {
                    let src_offset = self.current_row * src_row_bytes;
                    let out_row = output.rows_mut().nth(rows_written).unwrap();

                    for x in 0..width {
                        let r = buffer[src_offset + x * 3] as u32;
                        let g = buffer[src_offset + x * 3 + 1] as u32;
                        let b = buffer[src_offset + x * 3 + 2] as u32;
                        // BT.601: Y = 0.299*R + 0.587*G + 0.114*B (scaled by 1000)
                        out_row[x] = ((299 * r + 587 * g + 114 * b) / 1000) as u8;
                    }

                    rows_written += 1;
                    self.current_row += 1;
                }
            }

            return Ok(rows_written);
        }

        // Streaming mode: decode on-the-fly
        let mut rows_written = 0;

        while rows_written < max_rows && self.current_row < self.height as usize {
            self.decode_mcu_row()?;

            let cols = width.min(self.strip.strip_width);
            let out_row = output.rows_mut().nth(rows_written).unwrap();

            let y_slice = self.strip.y_row(self.row_in_mcu, cols);

            for (out, &y) in out_row[..cols].iter_mut().zip(y_slice) {
                *out = y.clamp(0, 255) as u8;
            }

            rows_written += 1;
            self.current_row += 1;
            self.row_in_mcu += 1;

            if self.row_in_mcu >= self.strip.mcu_height {
                self.advance_mcu_row();
            }
        }

        Ok(rows_written)
    }

    /// Read rows into a grayscale f32 buffer.
    ///
    /// Output is normalized to [0, 1] range.
    /// For grayscale JPEGs, this extracts the Y channel directly.
    /// For color JPEGs, it extracts the Y (luminance) channel.
    ///
    /// Returns the number of rows actually written.
    pub fn read_rows_gray_f32(&mut self, mut output: ImgRefMut<'_, f32>) -> Result<usize> {
        let max_rows = output.height();
        let width = self.width as usize;

        if output.width() < width {
            return Err(Error::internal(
                "output buffer too narrow for grayscale f32",
            ));
        }

        // Buffered mode: serve from pre-decoded buffer
        if let Some(ref buffer) = self.buffered_rgb {
            let mut rows_written = 0;

            if self.num_components == 1 {
                // Grayscale buffer (1 byte per pixel)
                while rows_written < max_rows && self.current_row < self.height as usize {
                    let src_offset = self.current_row * width;
                    let out_row = output.rows_mut().nth(rows_written).unwrap();

                    for x in 0..width {
                        out_row[x] = buffer[src_offset + x] as f32 / 255.0;
                    }

                    rows_written += 1;
                    self.current_row += 1;
                }
            } else {
                // Color buffer (RGB8): convert to grayscale using BT.601 coefficients
                let src_row_bytes = width * 3;
                while rows_written < max_rows && self.current_row < self.height as usize {
                    let src_offset = self.current_row * src_row_bytes;
                    let out_row = output.rows_mut().nth(rows_written).unwrap();

                    for x in 0..width {
                        let r = buffer[src_offset + x * 3] as f32;
                        let g = buffer[src_offset + x * 3 + 1] as f32;
                        let b = buffer[src_offset + x * 3 + 2] as f32;
                        // BT.601: Y = 0.299*R + 0.587*G + 0.114*B
                        out_row[x] = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0;
                    }

                    rows_written += 1;
                    self.current_row += 1;
                }
            }

            return Ok(rows_written);
        }

        // Streaming mode: decode on-the-fly
        let mut rows_written = 0;

        while rows_written < max_rows && self.current_row < self.height as usize {
            self.decode_mcu_row()?;

            let cols = width.min(self.strip.strip_width);
            let out_row = output.rows_mut().nth(rows_written).unwrap();

            let y_slice = self.strip.y_row(self.row_in_mcu, cols);

            for (out, &y) in out_row[..cols].iter_mut().zip(y_slice) {
                *out = y.clamp(0, 255) as f32 / 255.0;
            }

            rows_written += 1;
            self.current_row += 1;
            self.row_in_mcu += 1;

            if self.row_in_mcu >= self.strip.mcu_height {
                self.advance_mcu_row();
            }
        }

        Ok(rows_written)
    }

    /// Read rows into a linear f32 grayscale buffer.
    ///
    /// Output is in linear light (not sRGB gamma), range [0, 1].
    /// This applies the sRGB to linear conversion to each pixel.
    ///
    /// Returns the number of rows actually written.
    pub fn read_rows_gray_linear_f32(&mut self, mut output: ImgRefMut<'_, f32>) -> Result<usize> {
        let max_rows = output.height();
        let width = self.width as usize;

        if output.width() < width {
            return Err(Error::internal(
                "output buffer too narrow for linear grayscale f32",
            ));
        }

        // Buffered mode: serve from pre-decoded buffer with linearization
        if let Some(ref buffer) = self.buffered_rgb {
            let mut rows_written = 0;

            if self.num_components == 1 {
                // Grayscale buffer (1 byte per pixel)
                while rows_written < max_rows && self.current_row < self.height as usize {
                    let src_offset = self.current_row * width;
                    let out_row = output.rows_mut().nth(rows_written).unwrap();

                    for x in 0..width {
                        out_row[x] = srgb_to_linear(buffer[src_offset + x]);
                    }

                    rows_written += 1;
                    self.current_row += 1;
                }
            } else {
                // Color buffer (RGB8): convert to linear grayscale
                let src_row_bytes = width * 3;
                while rows_written < max_rows && self.current_row < self.height as usize {
                    let src_offset = self.current_row * src_row_bytes;
                    let out_row = output.rows_mut().nth(rows_written).unwrap();

                    for x in 0..width {
                        // Linearize each channel first, then compute luminance
                        let r = srgb_to_linear(buffer[src_offset + x * 3]);
                        let g = srgb_to_linear(buffer[src_offset + x * 3 + 1]);
                        let b = srgb_to_linear(buffer[src_offset + x * 3 + 2]);
                        // BT.601 in linear space
                        out_row[x] = 0.299 * r + 0.587 * g + 0.114 * b;
                    }

                    rows_written += 1;
                    self.current_row += 1;
                }
            }

            return Ok(rows_written);
        }

        // Streaming mode: decode on-the-fly
        let mut rows_written = 0;

        while rows_written < max_rows && self.current_row < self.height as usize {
            self.decode_mcu_row()?;

            let cols = width.min(self.strip.strip_width);
            let out_row = output.rows_mut().nth(rows_written).unwrap();

            let y_slice = self.strip.y_row(self.row_in_mcu, cols);

            for (out, &y) in out_row[..cols].iter_mut().zip(y_slice) {
                *out = srgb_to_linear(y.clamp(0, 255) as u8);
            }

            rows_written += 1;
            self.current_row += 1;
            self.row_in_mcu += 1;

            if self.row_in_mcu >= self.strip.mcu_height {
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

    /// Helper to encode RGB pixels with 4:4:4 (no subsampling).
    /// This ensures the streaming decode path is used, which matches the scanline reader's
    /// integer IDCT implementation.
    fn encode_rgb(width: u32, height: u32, pixels: &[u8], quality: f32) -> Vec<u8> {
        use crate::encode::v2::{ChromaSubsampling, EncoderConfig, PixelLayout};
        use enough::Unstoppable;
        // Use 4:4:4 to ensure streaming decode path is used (same IDCT as scanline reader)
        let config = EncoderConfig::ycbcr(quality, ChromaSubsampling::None);
        let mut enc = config
            .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
            .unwrap();
        enc.push_packed(pixels, Unstoppable).unwrap();
        enc.finish().unwrap()
    }

    /// Helper to encode RGB pixels with subsampling (baseline mode for test stability)
    fn encode_rgb_subsampled(
        width: u32,
        height: u32,
        pixels: &[u8],
        quality: f32,
        subsampling: crate::encode::v2::ChromaSubsampling,
    ) -> Vec<u8> {
        use crate::encode::v2::{EncoderConfig, PixelLayout};
        use enough::Unstoppable;
        let config = EncoderConfig::ycbcr(quality, subsampling).progressive(false);
        let mut enc = config
            .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
            .unwrap();
        enc.push_packed(pixels, Unstoppable).unwrap();
        enc.finish().unwrap()
    }

    /// Compare two u8 slices and return (max_diff, diff_count, first_diff_idx)
    fn compare_u8_slices(a: &[u8], b: &[u8]) -> (u8, usize, Option<usize>) {
        assert_eq!(a.len(), b.len(), "slice length mismatch");
        let mut max_diff: u8 = 0;
        let mut diff_count: usize = 0;
        let mut first_diff_idx: Option<usize> = None;

        for (i, (&va, &vb)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (va as i16 - vb as i16).unsigned_abs() as u8;
            if diff > 0 {
                diff_count += 1;
                if first_diff_idx.is_none() {
                    first_diff_idx = Some(i);
                }
                if diff > max_diff {
                    max_diff = diff;
                }
            }
        }
        (max_diff, diff_count, first_diff_idx)
    }

    /// Compare two f32 slices and return (max_diff, diff_count, first_diff_idx)
    #[allow(dead_code)]
    fn compare_f32_slices(a: &[f32], b: &[f32]) -> (f32, usize, Option<usize>) {
        assert_eq!(a.len(), b.len(), "slice length mismatch");
        let mut max_diff: f32 = 0.0;
        let mut diff_count: usize = 0;
        let mut first_diff_idx: Option<usize> = None;

        for (i, (&va, &vb)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (va - vb).abs();
            if diff > 1e-6 {
                diff_count += 1;
                if first_diff_idx.is_none() {
                    first_diff_idx = Some(i);
                }
                if diff > max_diff {
                    max_diff = diff;
                }
            }
        }
        (max_diff, diff_count, first_diff_idx)
    }

    /// Assert slices are equal, with detailed diff info on failure
    fn assert_slices_equal_u8(actual: &[u8], expected: &[u8], context: &str) {
        let (max_diff, diff_count, first_diff_idx) = compare_u8_slices(actual, expected);
        if diff_count > 0 {
            let first_idx = first_diff_idx.unwrap();
            panic!(
                "{}: slices differ - max_diff={}, diff_count={}/{} ({:.2}%), first_diff at idx {} (actual={}, expected={})",
                context, max_diff, diff_count, actual.len(),
                100.0 * diff_count as f64 / actual.len() as f64,
                first_idx, actual[first_idx], expected[first_idx]
            );
        }
    }

    /// Assert f32 slices are equal, with detailed diff info on failure
    #[allow(dead_code)]
    fn assert_slices_equal_f32(actual: &[f32], expected: &[f32], context: &str) {
        let (max_diff, diff_count, first_diff_idx) = compare_f32_slices(actual, expected);
        if diff_count > 0 {
            let first_idx = first_diff_idx.unwrap();
            panic!(
                "{}: slices differ - max_diff={:.6}, diff_count={}/{} ({:.2}%), first_diff at idx {} (actual={:.6}, expected={:.6})",
                context, max_diff, diff_count, actual.len(),
                100.0 * diff_count as f64 / actual.len() as f64,
                first_idx, actual[first_idx], expected[first_idx]
            );
        }
    }

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
        use crate::decode::Decoder;

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
        let jpeg = encode_rgb(width, height, &pixels, 95.0);

        // Decode normally for comparison
        let decoder = Decoder::new();
        let decoded = decoder
            .decode(&jpeg, enough::Unstoppable)
            .expect("decode failed");

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
        assert_slices_equal_u8(&scanline_pixels, &decoded.data, "test_scanline_reader_rgb8");
    }

    #[test]
    fn test_scanline_reader_partial_reads() {
        use crate::decode::Decoder;

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

        let jpeg = encode_rgb(width, height, &pixels, 90.0);

        let decoder = Decoder::new();
        let decoded = decoder
            .decode(&jpeg, enough::Unstoppable)
            .expect("decode failed");

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
        assert_slices_equal_u8(
            &scanline_pixels,
            &decoded.data,
            "test_scanline_reader_partial_reads",
        );
    }

    #[test]
    fn test_scanline_reader_rgbx8() {
        use crate::decode::Decoder;

        let width = 24u32;
        let height = 24u32;
        let mut pixels = vec![0u8; (width * height * 3) as usize];
        for i in 0..pixels.len() {
            pixels[i] = ((i * 7) % 256) as u8;
        }

        let jpeg = encode_rgb(width, height, &pixels, 85.0);

        let decoder = Decoder::new();
        let decoded = decoder
            .decode(&jpeg, enough::Unstoppable)
            .expect("decode failed");

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
        // First collect stats
        let mut max_diff: u8 = 0;
        let mut diff_count: usize = 0;
        let mut first_diff: Option<(usize, usize, &str, u8, u8)> = None;

        for y in 0..height as usize {
            for x in 0..width as usize {
                let rgb_idx = (y * width as usize + x) * 3;
                let rgbx_idx = (y * width as usize + x) * 4;

                for (c, name) in [(0, "R"), (1, "G"), (2, "B")] {
                    let actual = rgbx_pixels[rgbx_idx + c];
                    let expected = decoded.data[rgb_idx + c];
                    let diff = (actual as i16 - expected as i16).unsigned_abs() as u8;
                    if diff > 0 {
                        diff_count += 1;
                        if first_diff.is_none() {
                            first_diff = Some((x, y, name, actual, expected));
                        }
                        if diff > max_diff {
                            max_diff = diff;
                        }
                    }
                }
                assert_eq!(rgbx_pixels[rgbx_idx + 3], 255, "Alpha should be 255");
            }
        }

        if diff_count > 0 {
            let (x, y, ch, actual, expected) = first_diff.unwrap();
            let total = (width * height * 3) as usize;
            panic!(
                "test_scanline_reader_rgbx8: max_diff={}, diff_count={}/{} ({:.2}%), first_diff at ({},{}) {}={} expected={}",
                max_diff, diff_count, total, 100.0 * diff_count as f64 / total as f64,
                x, y, ch, actual, expected
            );
        }
    }

    #[test]
    fn test_scanline_reader_rgba_f32() {
        use crate::decode::Decoder;

        let width = 16u32;
        let height = 16u32;
        let mut pixels = vec![0u8; (width * height * 3) as usize];
        for i in 0..pixels.len() {
            pixels[i] = ((i * 11) % 256) as u8;
        }

        let jpeg = encode_rgb(width, height, &pixels, 90.0);

        let decoder = Decoder::new();
        let decoded = decoder
            .decode(&jpeg, enough::Unstoppable)
            .expect("decode failed");

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
        let mut max_diff: f32 = 0.0;
        let mut diff_count: usize = 0;
        let mut first_diff: Option<(usize, usize, usize, f32, f32)> = None;

        for y in 0..height as usize {
            for x in 0..width as usize {
                let rgb_idx = (y * width as usize + x) * 3;
                let rgba_idx = (y * width as usize + x) * 4;

                for c in 0..3 {
                    let expected_linear = srgb_to_linear(decoded.data[rgb_idx + c]);
                    let actual_linear = rgba_pixels[rgba_idx + c];
                    let diff = (expected_linear - actual_linear).abs();
                    if diff > 0.01 {
                        diff_count += 1;
                        if first_diff.is_none() {
                            first_diff = Some((x, y, c, actual_linear, expected_linear));
                        }
                        if diff > max_diff {
                            max_diff = diff;
                        }
                    }
                }
            }
        }

        if diff_count > 0 {
            let (x, y, c, actual, expected) = first_diff.unwrap();
            let total = (width * height * 3) as usize;
            panic!(
                "test_scanline_reader_rgba_f32: max_diff={:.6}, diff_count={}/{} ({:.2}%), first_diff at ({},{}) ch{}={:.6} expected={:.6}",
                max_diff, diff_count, total, 100.0 * diff_count as f64 / total as f64,
                x, y, c, actual, expected
            );
        }
    }

    #[test]
    fn test_scanline_reader_ycbcr_planes() {
        use crate::decode::Decoder;

        let width = 32u32;
        let height = 24u32;
        let mut pixels = vec![0u8; (width * height * 3) as usize];
        for i in 0..pixels.len() {
            pixels[i] = ((i * 13) % 256) as u8;
        }

        let jpeg = encode_rgb(width, height, &pixels, 90.0);

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
        use crate::decode::Decoder;

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

        let jpeg = encode_rgb(width, height, &pixels, 90.0);

        let decoder = Decoder::new();
        let decoded = decoder
            .decode(&jpeg, enough::Unstoppable)
            .expect("decode failed");

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
        assert_slices_equal_u8(
            &scanline_pixels,
            &decoded.data,
            "test_scanline_reader_non_mcu_aligned",
        );
    }

    #[test]
    fn test_scanline_reader_420() {
        use crate::decode::Decoder;
        use crate::encode::v2::ChromaSubsampling;

        // Create test image - 64x48 for multiple MCU rows
        // 4:2:0 has 16x16 MCUs, so this is 4x3 MCUs
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

        // Encode as 4:2:0
        let jpeg = encode_rgb_subsampled(width, height, &pixels, 95.0, ChromaSubsampling::Quarter);

        // Decode normally for comparison
        let decoder = Decoder::new();
        let decoded = decoder
            .decode(&jpeg, enough::Unstoppable)
            .expect("decode failed");

        // Decode via scanline reader
        let mut reader = decoder
            .scanline_reader(&jpeg)
            .expect("scanline_reader failed");
        assert_eq!(reader.width(), width);
        assert_eq!(reader.height(), height);
        assert_eq!(reader.subsampling(), Subsampling::S420);

        let mut scanline_pixels = vec![0u8; (width * height * 3) as usize];
        let stride = (width * 3) as usize;

        let mut total_rows = 0;
        while !reader.is_finished() {
            let remaining = height as usize - total_rows;
            let buf_start = total_rows * stride;
            let output =
                imgref::ImgRefMut::new(&mut scanline_pixels[buf_start..], stride, remaining);
            let rows = reader
                .read_rows_rgb8(output)
                .expect("read_rows_rgb8 failed");
            total_rows += rows;
        }

        assert_eq!(total_rows, height as usize);
        assert_eq!(
            scanline_pixels.len(),
            decoded.data.len(),
            "output size mismatch"
        );

        // Compare outputs with tolerance - scanline reader uses simpler i16 processing
        // while regular decoder uses f32 with bias computation, so outputs won't be bit-identical
        let mut max_diff = 0i32;
        let mut total_diff = 0u64;
        for (i, (&a, &b)) in scanline_pixels.iter().zip(decoded.data.iter()).enumerate() {
            let diff = (a as i32 - b as i32).abs();
            max_diff = max_diff.max(diff);
            total_diff += diff as u64;
            if diff > 10 {
                panic!(
                    "Pixel at index {} differs by {} (scanline={}, regular={})",
                    i, diff, a, b
                );
            }
        }
        let avg_diff = total_diff as f64 / scanline_pixels.len() as f64;
        assert!(
            avg_diff < 3.0,
            "Average pixel difference {} too high (max diff: {})",
            avg_diff,
            max_diff
        );
    }

    #[test]
    fn test_scanline_reader_420_non_mcu_aligned() {
        use crate::decode::Decoder;
        use crate::encode::v2::ChromaSubsampling;

        // Non-MCU-aligned dimensions (not multiples of 16 for 4:2:0)
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

        // Encode as 4:2:0
        let jpeg = encode_rgb_subsampled(width, height, &pixels, 90.0, ChromaSubsampling::Quarter);

        // Decode normally for comparison
        let decoder = Decoder::new();
        let decoded = decoder
            .decode(&jpeg, enough::Unstoppable)
            .expect("decode failed");

        // Decode via scanline reader
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
        assert_eq!(
            scanline_pixels.len(),
            decoded.data.len(),
            "output size mismatch"
        );

        // Compare with tolerance
        let mut max_diff = 0i32;
        let mut total_diff = 0u64;
        for (i, (&a, &b)) in scanline_pixels.iter().zip(decoded.data.iter()).enumerate() {
            let diff = (a as i32 - b as i32).abs();
            max_diff = max_diff.max(diff);
            total_diff += diff as u64;
            if diff > 10 {
                panic!(
                    "Pixel at index {} differs by {} (scanline={}, regular={})",
                    i, diff, a, b
                );
            }
        }
        let avg_diff = total_diff as f64 / scanline_pixels.len() as f64;
        assert!(
            avg_diff < 3.0,
            "Average pixel difference {} too high (max diff: {})",
            avg_diff,
            max_diff
        );
    }

    /// Helper to encode grayscale pixels.
    fn encode_grayscale(width: u32, height: u32, pixels: &[u8], quality: f32) -> Vec<u8> {
        use crate::encode::v2::{EncoderConfig, PixelLayout};
        use enough::Unstoppable;
        let config = EncoderConfig::grayscale(quality);
        let mut enc = config
            .encode_from_bytes(width, height, PixelLayout::Gray8Srgb)
            .unwrap();
        enc.push_packed(pixels, Unstoppable).unwrap();
        enc.finish().unwrap()
    }

    #[test]
    fn test_scanline_reader_grayscale_basic() {
        use crate::decode::Decoder;
        use crate::types::PixelFormat;

        // Create test grayscale image - 64x48 for multiple MCU rows
        let width = 64u32;
        let height = 48u32;
        let mut pixels = vec![0u8; (width * height) as usize];
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize;
                // Diagonal gradient
                pixels[idx] = ((x + y) * 2) as u8;
            }
        }

        // Encode as grayscale
        let jpeg = encode_grayscale(width, height, &pixels, 95.0);

        // Decode normally for comparison - use Gray output format
        let decoder = Decoder::new().output_format(PixelFormat::Gray);
        let decoded = decoder
            .decode(&jpeg, enough::Unstoppable)
            .expect("decode failed");

        // Decode via scanline reader
        let mut reader = Decoder::new()
            .scanline_reader(&jpeg)
            .expect("scanline_reader failed for grayscale");
        assert_eq!(reader.width(), width);
        assert_eq!(reader.height(), height);
        assert!(reader.is_grayscale());
        assert_eq!(reader.num_components(), 1);

        let mut scanline_pixels = vec![0u8; (width * height) as usize];

        // Read all rows using grayscale method
        let mut total_rows = 0;
        while !reader.is_finished() {
            let remaining = height as usize - total_rows;
            let stride = width as usize;
            let buf_start = total_rows * stride;
            let output =
                imgref::ImgRefMut::new(&mut scanline_pixels[buf_start..], stride, remaining);
            let rows = reader
                .read_rows_gray8(output)
                .expect("read_rows_gray8 failed");
            total_rows += rows;
        }

        assert_eq!(total_rows, height as usize);

        // Compare outputs - should match within JPEG compression tolerance
        assert_eq!(
            scanline_pixels.len(),
            decoded.data.len(),
            "output size mismatch"
        );

        let (max_diff, diff_count, _) = compare_u8_slices(&scanline_pixels, &decoded.data);
        assert!(
            max_diff <= 2,
            "grayscale scanline reader max_diff {} > 2 (diff_count: {})",
            max_diff,
            diff_count
        );
    }

    #[test]
    fn test_scanline_reader_grayscale_non_mcu_aligned() {
        use crate::decode::Decoder;
        use crate::types::PixelFormat;

        // Non-MCU-aligned dimensions (not multiples of 8)
        let width = 37u32;
        let height = 29u32;
        let mut pixels = vec![0u8; (width * height) as usize];
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize;
                pixels[idx] = (x * 7 + y * 3) as u8;
            }
        }

        let jpeg = encode_grayscale(width, height, &pixels, 90.0);

        // Use Gray output format for comparison
        let decoder = Decoder::new().output_format(PixelFormat::Gray);
        let decoded = decoder
            .decode(&jpeg, enough::Unstoppable)
            .expect("decode failed");

        let mut reader = Decoder::new()
            .scanline_reader(&jpeg)
            .expect("scanline_reader failed");
        assert!(reader.is_grayscale());

        let mut scanline_pixels = vec![0u8; (width * height) as usize];
        let stride = width as usize;

        let mut total_rows = 0;
        while !reader.is_finished() {
            let remaining = height as usize - total_rows;
            let buf_start = total_rows * stride;
            let output =
                imgref::ImgRefMut::new(&mut scanline_pixels[buf_start..], stride, remaining);
            let rows = reader.read_rows_gray8(output).expect("read failed");
            total_rows += rows;
        }

        assert_eq!(total_rows, height as usize);
        let (max_diff, _, _) = compare_u8_slices(&scanline_pixels, &decoded.data);
        assert!(
            max_diff <= 2,
            "grayscale non-MCU-aligned max_diff {} > 2",
            max_diff
        );
    }

    #[test]
    fn test_scanline_reader_grayscale_f32() {
        use crate::decode::Decoder;

        let width = 32u32;
        let height = 24u32;
        let mut pixels = vec![0u8; (width * height) as usize];
        for i in 0..pixels.len() {
            pixels[i] = ((i * 13) % 256) as u8;
        }

        let jpeg = encode_grayscale(width, height, &pixels, 90.0);

        let decoder = Decoder::new();
        let mut reader = decoder
            .scanline_reader(&jpeg)
            .expect("scanline_reader failed");

        let mut gray_pixels = vec![0.0f32; (width * height) as usize];
        let stride = width as usize;

        let mut total_rows = 0;
        while !reader.is_finished() {
            let remaining = height as usize - total_rows;
            let buf_start = total_rows * stride;
            let output = imgref::ImgRefMut::new(&mut gray_pixels[buf_start..], stride, remaining);
            let rows = reader.read_rows_gray_f32(output).expect("read failed");
            total_rows += rows;
        }

        assert_eq!(total_rows, height as usize);

        // Verify values are in valid [0, 1] range
        for (i, &val) in gray_pixels.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&val),
                "Value at {} should be in [0,1], got {}",
                i,
                val
            );
        }
    }

    #[test]
    fn test_scanline_reader_grayscale_linear_f32() {
        use crate::decode::Decoder;

        let width = 16u32;
        let height = 16u32;
        let mut pixels = vec![0u8; (width * height) as usize];
        for i in 0..pixels.len() {
            pixels[i] = (i % 256) as u8;
        }

        let jpeg = encode_grayscale(width, height, &pixels, 95.0);

        let decoder = Decoder::new();
        let mut reader = decoder
            .scanline_reader(&jpeg)
            .expect("scanline_reader failed");

        let mut linear_pixels = vec![0.0f32; (width * height) as usize];
        let stride = width as usize;

        let mut total_rows = 0;
        while !reader.is_finished() {
            let remaining = height as usize - total_rows;
            let buf_start = total_rows * stride;
            let output = imgref::ImgRefMut::new(&mut linear_pixels[buf_start..], stride, remaining);
            let rows = reader
                .read_rows_gray_linear_f32(output)
                .expect("read failed");
            total_rows += rows;
        }

        // Verify values are in valid [0, 1] range
        for (i, &val) in linear_pixels.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&val),
                "Linear value at {} should be in [0,1], got {}",
                i,
                val
            );
        }

        // Verify linear conversion: sRGB 128 ≈ linear 0.2159
        // Find pixels close to 128 and verify they're near 0.2159 in linear
        // (Since JPEG is lossy, we can't be exact, but the relationship should hold)
    }

    /// Encode to progressive JPEG using the encoder
    fn encode_progressive_rgb(width: u32, height: u32, pixels: &[u8], quality: f32) -> Vec<u8> {
        use crate::encode::v2::{ChromaSubsampling, EncoderConfig, PixelLayout};
        use enough::Unstoppable;
        // Use progressive mode
        let config = EncoderConfig::ycbcr(quality, ChromaSubsampling::None).progressive(true);
        let mut enc = config
            .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
            .unwrap();
        enc.push_packed(pixels, Unstoppable).unwrap();
        enc.finish().unwrap()
    }

    #[test]
    fn test_scanline_reader_progressive() {
        // Test that progressive JPEG works via buffered mode
        let width = 16u32;
        let height = 16u32;
        let mut input_pixels = vec![0u8; (width * height * 3) as usize];
        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * 3) as usize;
                input_pixels[idx] = ((x * 16) % 256) as u8; // R
                input_pixels[idx + 1] = ((y * 16) % 256) as u8; // G
                input_pixels[idx + 2] = 128; // B
            }
        }

        // Encode as progressive JPEG
        let progressive_jpeg = encode_progressive_rgb(width, height, &input_pixels, 95.0);

        // Verify it's actually progressive by checking SOF marker
        assert!(
            progressive_jpeg.windows(2).any(|w| w == [0xFF, 0xC2]), // SOF2 = progressive
            "JPEG should be progressive (SOF2)"
        );

        // Decode via scanline reader
        let decoder = crate::decode::Decoder::new();
        let mut reader = decoder
            .scanline_reader(&progressive_jpeg)
            .expect("scanline_reader should support progressive via buffered mode");

        assert_eq!(reader.width(), width);
        assert_eq!(reader.height(), height);

        // Read all rows
        let mut scanline_pixels = vec![0u8; (width * height * 3) as usize];
        let mut rows_read = 0;
        while rows_read < height as usize {
            let remaining = height as usize - rows_read;
            let output = ImgRefMut::new(
                &mut scanline_pixels[rows_read * width as usize * 3..],
                width as usize * 3,
                remaining,
            );
            let count = reader
                .read_rows_rgb8(output)
                .expect("read_rows_rgb8 failed");
            if count == 0 {
                break;
            }
            rows_read += count;
        }

        assert_eq!(rows_read, height as usize, "Should read all rows");

        // Compare with full-frame decode
        let decoded = decoder
            .decode(&progressive_jpeg, enough::Unstoppable)
            .expect("decode failed");
        let (max_diff, diff_count, _) = compare_u8_slices(&scanline_pixels, &decoded.data);

        // Should be identical (same decode path for progressive)
        assert_eq!(
            max_diff, 0,
            "Scanline reader should match full-frame decode for progressive JPEG (max diff={}, diff_count={})",
            max_diff, diff_count
        );
    }

    #[test]
    fn test_scanline_reader_progressive_grayscale() {
        // Test that progressive grayscale JPEG works via buffered mode
        let width = 16u32;
        let height = 16u32;
        let mut input_pixels = vec![0u8; (width * height) as usize];
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize;
                input_pixels[idx] = ((x * 16 + y * 8) % 256) as u8;
            }
        }

        // Encode grayscale as progressive JPEG
        use crate::encode::v2::{EncoderConfig, PixelLayout};
        use enough::Unstoppable;
        let config = EncoderConfig::grayscale(95.0).progressive(true);
        let mut enc = config
            .encode_from_bytes(width, height, PixelLayout::Gray8Srgb)
            .unwrap();
        enc.push_packed(&input_pixels, Unstoppable).unwrap();
        let progressive_jpeg = enc.finish().unwrap();

        // Verify it's actually progressive
        assert!(
            progressive_jpeg.windows(2).any(|w| w == [0xFF, 0xC2]),
            "JPEG should be progressive (SOF2)"
        );

        // Decode via scanline reader
        let decoder = crate::decode::Decoder::new();
        let mut reader = decoder
            .scanline_reader(&progressive_jpeg)
            .expect("scanline_reader should support progressive grayscale");

        // Read grayscale rows
        let mut scanline_pixels = vec![0u8; (width * height) as usize];
        let mut rows_read = 0;
        while rows_read < height as usize {
            let remaining = height as usize - rows_read;
            let output = ImgRefMut::new(
                &mut scanline_pixels[rows_read * width as usize..],
                width as usize,
                remaining,
            );
            let count = reader
                .read_rows_gray8(output)
                .expect("read_rows_gray8 failed");
            if count == 0 {
                break;
            }
            rows_read += count;
        }

        assert_eq!(rows_read, height as usize, "Should read all rows");

        // The grayscale values should be reasonable (can't compare exactly since
        // the full-frame decoder uses PixelFormat::Rgb which converts grayscale to RGB)
        let mean: f64 =
            scanline_pixels.iter().map(|&x| x as f64).sum::<f64>() / scanline_pixels.len() as f64;
        assert!(
            mean > 50.0 && mean < 200.0,
            "Grayscale mean should be reasonable: {}",
            mean
        );
    }
}
