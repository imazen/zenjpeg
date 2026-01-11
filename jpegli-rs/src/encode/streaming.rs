//! Streaming input encoder API.
//!
//! This module provides a streaming encoder that accepts rows incrementally,
//! reducing peak memory by not requiring the full input image in memory.
//!
//! # Memory Savings
//!
//! For a 4K (3840x2160) RGB image:
//! - Standard encoder: ~50 MB peak (input buffer + internal)
//! - Streaming encoder: ~26 MB peak (~50% reduction)
//!
//! # Example
//!
//! ```rust,ignore
//! use jpegli::{StreamingEncoder, Quality, Subsampling};
//!
//! let mut encoder = StreamingEncoder::new(1920, 1080)
//!     .quality(Quality::from_quality(85.0))
//!     .subsampling(Subsampling::S420)
//!     .build()?;
//!
//! // Push rows one at a time (e.g., from a decoder or generator)
//! for row in image_rows {
//!     encoder.push_row(row)?;
//! }
//!
//! // Or push chunks of rows
//! // encoder.push_rows(chunk, 4)?;
//!
//! let jpeg = encoder.finish()?;
//! ```

use crate::encode::strip::StripProcessor;
use crate::encode::Encoder;
use crate::error::{Error, Result};
use crate::quant::{self, Quality, QuantTable, ZeroBiasParams};
use crate::types::{
    ChromaDownsampling, ColorSpace, EncodingBackend, JpegMode, PixelFormat, Subsampling,
};
use enough::{Never, Stop};

/// Builder for creating a streaming encoder.
///
/// Use [`StreamingEncoder::new()`] to start building.
#[derive(Debug, Clone)]
pub struct StreamingEncoderBuilder {
    width: u32,
    height: u32,
    quality: Quality,
    subsampling: Subsampling,
    pixel_format: PixelFormat,
    mode: JpegMode,
    optimize_huffman: bool,
    chroma_downsampling: ChromaDownsampling,
    restart_interval: u16,
}

impl StreamingEncoderBuilder {
    /// Creates a new streaming encoder builder with default settings.
    fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            quality: Quality::default(),
            subsampling: Subsampling::S444,
            pixel_format: PixelFormat::Rgb,
            mode: JpegMode::Baseline,
            optimize_huffman: true,
            chroma_downsampling: ChromaDownsampling::Box,
            restart_interval: 0,
        }
    }

    /// Sets the quality using jpegli's native quality scale.
    ///
    /// Use `Quality::from_quality(90.0)` for traditional JPEG quality (1-100)
    /// or `Quality::from_distance(1.0)` for butteraugli distance.
    #[must_use]
    pub fn quality(mut self, quality: Quality) -> Self {
        self.quality = quality;
        self
    }

    /// Sets chroma subsampling.
    #[must_use]
    pub fn subsampling(mut self, subsampling: Subsampling) -> Self {
        self.subsampling = subsampling;
        self
    }

    /// Sets the pixel format of input data.
    #[must_use]
    pub fn pixel_format(mut self, format: PixelFormat) -> Self {
        self.pixel_format = format;
        self
    }

    /// Sets the JPEG encoding mode.
    #[must_use]
    pub fn mode(mut self, mode: JpegMode) -> Self {
        self.mode = mode;
        self
    }

    /// Enables optimized Huffman tables.
    #[must_use]
    pub fn optimize_huffman(mut self, enable: bool) -> Self {
        self.optimize_huffman = enable;
        self
    }

    /// Sets chroma downsampling method for subsampled modes.
    #[must_use]
    pub fn chroma_downsampling(mut self, method: ChromaDownsampling) -> Self {
        self.chroma_downsampling = method;
        self
    }

    /// Sets the restart interval (MCUs between restart markers).
    #[must_use]
    pub fn restart_interval(mut self, interval: u16) -> Self {
        self.restart_interval = interval;
        self
    }

    /// Builds the streaming encoder.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Dimensions are zero or exceed maximum
    /// - Memory allocation fails
    pub fn build(self) -> Result<StreamingEncoder> {
        StreamingEncoder::from_builder(self)
    }

    /// Encodes a complete image buffer in one call.
    ///
    /// This is a convenience method that builds the encoder, pushes all rows,
    /// and finishes in a single call. For large images or streaming scenarios,
    /// use `.build()` and push rows incrementally instead.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use jpegli::{StreamingEncoder, Quality, Subsampling};
    ///
    /// let pixels: Vec<u8> = vec![128; 640 * 480 * 3];
    /// let jpeg = StreamingEncoder::new(640, 480)
    ///     .quality(Quality::from_quality(85.0))
    ///     .subsampling(Subsampling::S420)
    ///     .encode_all(&pixels)?;
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Buffer size doesn't match width × height × bytes_per_pixel
    /// - Encoding fails
    pub fn encode_all(self, data: &[u8]) -> Result<Vec<u8>> {
        let width = self.width as usize;
        let height = self.height as usize;
        let bpp = self.pixel_format.bytes_per_pixel();
        let expected_size = width * height * bpp;

        if data.len() != expected_size {
            return Err(Error::InvalidBufferSize {
                expected: expected_size,
                actual: data.len(),
            });
        }

        let mut encoder = self.build()?;
        let row_size = width * bpp;

        for y in 0..height {
            let start = y * row_size;
            encoder.push_row(&data[start..start + row_size])?;
        }

        encoder.finish()
    }

    /// Encodes a complete image buffer with cancellation support.
    ///
    /// Like `encode_all()`, but checks for cancellation between strips.
    pub fn encode_all_with_stop(self, data: &[u8], stop: impl Stop) -> Result<Vec<u8>> {
        let width = self.width as usize;
        let height = self.height as usize;
        let bpp = self.pixel_format.bytes_per_pixel();
        let expected_size = width * height * bpp;

        if data.len() != expected_size {
            return Err(Error::InvalidBufferSize {
                expected: expected_size,
                actual: data.len(),
            });
        }

        let mut encoder = self.build()?;
        let row_size = width * bpp;

        for y in 0..height {
            let start = y * row_size;
            encoder.push_row_with_stop(&data[start..start + row_size], &stop)?;
        }

        encoder.finish_with_stop(stop)
    }

    /// Estimates the peak memory usage for this configuration.
    ///
    /// Returns the estimated peak memory in bytes based on image dimensions,
    /// subsampling mode, and pixel format. This estimate includes:
    /// - Row buffer (one strip's worth of input data)
    /// - Strip processing buffers (f32 YCbCr planes)
    /// - Pending DCT blocks (double-buffered)
    /// - Final i16 block storage
    /// - AQ strength storage
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use jpegli::{StreamingEncoder, Subsampling};
    ///
    /// let estimated = StreamingEncoder::new(3840, 2160)
    ///     .subsampling(Subsampling::S420)
    ///     .estimate_memory_usage();
    ///
    /// println!("Estimated peak memory: {} MB", estimated / 1024 / 1024);
    /// ```
    #[must_use]
    pub fn estimate_memory_usage(&self) -> usize {
        let width = self.width as usize;
        let height = self.height as usize;

        // Strip height based on subsampling
        let strip_height = match self.subsampling {
            Subsampling::S420 | Subsampling::S440 => 16,
            _ => 8,
        };

        // MCU size for padding
        let mcu_size = self.subsampling.mcu_size();
        let padded_width = (width + mcu_size - 1) / mcu_size * mcu_size;

        // Chroma dimensions
        let (c_width, c_strip_height) = match self.subsampling {
            Subsampling::S420 => ((width + 1) / 2, strip_height / 2),
            Subsampling::S422 => ((width + 1) / 2, strip_height),
            Subsampling::S440 => (width, strip_height / 2),
            Subsampling::S444 => (width, strip_height),
        };
        let padded_c_width = (c_width + 7) / 8 * 8;

        // Block counts
        let y_blocks_w = (width + 7) / 8;
        let y_blocks_h = (height + 7) / 8;
        let y_block_count = y_blocks_w * y_blocks_h;

        let c_block_count = match self.subsampling {
            Subsampling::S420 => ((width + 15) / 16) * ((height + 15) / 16),
            Subsampling::S422 => ((width + 15) / 16) * y_blocks_h,
            Subsampling::S440 => y_blocks_w * ((height + 15) / 16),
            Subsampling::S444 => y_block_count,
        };

        // 1. Row buffer for input (one strip's worth)
        let bpp = self.pixel_format.bytes_per_pixel();
        let row_buffer = width * strip_height * bpp;

        // 2. Strip f32 buffers (Y, Cb, Cr at full resolution before downsampling)
        let strip_y = padded_width * strip_height * 4; // f32 = 4 bytes
        let strip_cb = padded_width * strip_height * 4;
        let strip_cr = padded_width * strip_height * 4;

        // 3. Downsampled chroma temp buffers
        let strip_cb_down = padded_c_width * c_strip_height * 4;
        let strip_cr_down = padded_c_width * c_strip_height * 4;

        // 4. Pending f32 DCT blocks (double-buffered, 2 iMCU rows)
        let padded_y_blocks_h = padded_width / 8;
        let v_samp = match self.subsampling {
            Subsampling::S420 | Subsampling::S440 => 2,
            _ => 1,
        };
        let pending_y_capacity = padded_y_blocks_h * v_samp;
        let padded_c_blocks_h = padded_c_width / 8;
        let pending_c_capacity = padded_c_blocks_h;

        // 256 bytes per f32 block, 2 buffers (double-buffered)
        let pending_y_f32 = 2 * pending_y_capacity * 256;
        let pending_cb_f32 = 2 * pending_c_capacity * 256;
        let pending_cr_f32 = 2 * pending_c_capacity * 256;

        // 5. Final i16 blocks (128 bytes per block)
        let y_blocks_i16 = y_block_count * 128;
        let c_blocks_i16 = c_block_count * 2 * 128; // Cb + Cr

        // 6. AQ strengths (one f32 per Y block)
        let aq_strengths = y_block_count * 4;

        // Total estimate
        row_buffer
            + strip_y
            + strip_cb
            + strip_cr
            + strip_cb_down
            + strip_cr_down
            + pending_y_f32
            + pending_cb_f32
            + pending_cr_f32
            + y_blocks_i16
            + c_blocks_i16
            + aq_strengths
    }
}

/// Streaming input JPEG encoder.
///
/// Accepts rows incrementally and outputs JPEG at the end.
/// Uses strip-based processing internally for low peak memory usage.
///
/// # Example
///
/// ```rust,ignore
/// use jpegli::StreamingEncoder;
///
/// let mut encoder = StreamingEncoder::new(1920, 1080).build()?;
///
/// // Push rows from a decoder or generator
/// for row in image_rows {
///     encoder.push_row(row)?;
/// }
///
/// let jpeg = encoder.finish()?;
/// ```
pub struct StreamingEncoder {
    /// Image width in pixels
    width: usize,
    /// Image height in pixels
    height: usize,
    /// Bytes per row of input data
    bytes_per_row: usize,
    /// Strip height (rows to buffer before processing)
    strip_height: usize,

    /// Row buffer (accumulates rows until strip is ready)
    row_buffer: Vec<u8>,
    /// Number of rows currently buffered
    rows_buffered: usize,
    /// Current Y position (rows processed so far)
    current_y: usize,

    /// Underlying strip processor
    processor: StripProcessor,

    /// Encoder for JPEG output generation
    encoder: Encoder,

    /// Quantization tables (generated from quality)
    y_quant: QuantTable,
    cb_quant: QuantTable,
    cr_quant: QuantTable,
}

impl StreamingEncoder {
    /// Creates a new streaming encoder builder with the given dimensions.
    ///
    /// Use the builder methods to configure quality, subsampling, etc.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use jpegli::{StreamingEncoder, Quality, Subsampling};
    ///
    /// let encoder = StreamingEncoder::new(1920, 1080)
    ///     .quality(Quality::from_quality(85.0))
    ///     .subsampling(Subsampling::S420)
    ///     .build()?;
    /// ```
    #[must_use]
    pub fn new(width: u32, height: u32) -> StreamingEncoderBuilder {
        StreamingEncoderBuilder::new(width, height)
    }

    /// Creates a streaming encoder from builder configuration.
    fn from_builder(builder: StreamingEncoderBuilder) -> Result<Self> {
        let width = builder.width as usize;
        let height = builder.height as usize;

        if width == 0 || height == 0 {
            return Err(Error::InvalidDimensions {
                width: builder.width,
                height: builder.height,
                reason: "dimensions must be non-zero",
            });
        }

        // Create strip processor
        let mut processor = StripProcessor::with_options(
            width,
            height,
            builder.subsampling,
            builder.pixel_format,
            builder.chroma_downsampling,
            builder.restart_interval,
        )?;

        // Generate quantization tables
        let is_420 = builder.subsampling == Subsampling::S420;
        let y_quant = quant::generate_quant_table(
            builder.quality,
            0,
            ColorSpace::YCbCr,
            false, // not XYB
            is_420,
        );
        let cb_quant = quant::generate_quant_table(
            builder.quality,
            1,
            ColorSpace::YCbCr,
            false,
            is_420,
        );
        let cr_quant = quant::generate_quant_table(
            builder.quality,
            2,
            ColorSpace::YCbCr,
            false,
            is_420,
        );

        // Compute zero bias params
        let effective_distance =
            quant::quant_vals_to_distance(&y_quant, &cb_quant, &cr_quant);
        let y_zero_bias = ZeroBiasParams::for_ycbcr(effective_distance, 0);
        let cb_zero_bias = ZeroBiasParams::for_ycbcr(effective_distance, 1);
        let cr_zero_bias = ZeroBiasParams::for_ycbcr(effective_distance, 2);

        processor.set_quant_tables(
            y_quant.clone(),
            cb_quant.clone(),
            cr_quant.clone(),
            y_zero_bias,
            cb_zero_bias,
            cr_zero_bias,
        )?;

        let strip_height = processor.strip_height();
        let bytes_per_row = width * builder.pixel_format.bytes_per_pixel();

        // Allocate row buffer for one strip
        let row_buffer = vec![0u8; bytes_per_row * strip_height];

        // Create encoder for final JPEG output
        #[allow(deprecated)]
        let encoder = Encoder::new()
            .width(builder.width)
            .height(builder.height)
            .pixel_format(builder.pixel_format)
            .quality(builder.quality)
            .subsampling(builder.subsampling)
            .mode(builder.mode)
            .optimize_huffman(builder.optimize_huffman)
            .chroma_downsampling(builder.chroma_downsampling)
            .restart_interval(builder.restart_interval)
            .encoding_backend(EncodingBackend::Strip);

        Ok(Self {
            width,
            height,
            bytes_per_row,
            strip_height,
            row_buffer,
            rows_buffered: 0,
            current_y: 0,
            processor,
            encoder,
            y_quant,
            cb_quant,
            cr_quant,
        })
    }

    /// Returns the number of rows pushed so far.
    #[must_use]
    pub fn rows_pushed(&self) -> usize {
        self.current_y + self.rows_buffered
    }

    /// Returns the expected number of bytes per row.
    #[must_use]
    pub fn bytes_per_row(&self) -> usize {
        self.bytes_per_row
    }

    /// Returns the total height of the image.
    #[must_use]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Returns the strip height (internal processing unit).
    #[must_use]
    pub fn strip_height(&self) -> usize {
        self.strip_height
    }

    /// Pushes a single row of pixel data.
    ///
    /// The row must be exactly `bytes_per_row()` bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Row length doesn't match expected bytes per row
    /// - All rows have already been pushed
    /// - Internal processing fails
    pub fn push_row(&mut self, row: &[u8]) -> Result<()> {
        self.push_row_with_stop(row, Never)
    }

    /// Pushes a single row with cancellation support.
    ///
    /// The `stop` source is checked before processing each strip.
    /// Returns `Error::Cancelled` if cancellation is requested.
    pub fn push_row_with_stop(&mut self, row: &[u8], stop: impl Stop) -> Result<()> {
        // Check cancellation
        stop.check()?;

        // Validate row size
        if row.len() != self.bytes_per_row {
            return Err(Error::InvalidBufferSize {
                expected: self.bytes_per_row,
                actual: row.len(),
            });
        }

        // Check if we've already received all rows
        if self.current_y + self.rows_buffered >= self.height {
            return Err(Error::IoError {
                reason: format!(
                    "already received all {} rows",
                    self.height
                ),
            });
        }

        // Copy row into buffer
        let offset = self.rows_buffered * self.bytes_per_row;
        self.row_buffer[offset..offset + self.bytes_per_row].copy_from_slice(row);
        self.rows_buffered += 1;

        // Check if we should flush the strip
        let remaining = self.height - self.current_y;
        if self.rows_buffered >= self.strip_height || self.rows_buffered >= remaining {
            self.flush_strip_with_stop(&stop)?;
        }

        Ok(())
    }

    /// Pushes multiple rows at once.
    ///
    /// The data must be exactly `num_rows * bytes_per_row()` bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Data length doesn't match expected size
    /// - Too many rows would be pushed
    /// - Internal processing fails
    pub fn push_rows(&mut self, data: &[u8], num_rows: usize) -> Result<()> {
        self.push_rows_with_stop(data, num_rows, Never)
    }

    /// Pushes multiple rows with cancellation support.
    pub fn push_rows_with_stop(
        &mut self,
        data: &[u8],
        num_rows: usize,
        stop: impl Stop,
    ) -> Result<()> {
        let expected_len = num_rows * self.bytes_per_row;
        if data.len() != expected_len {
            return Err(Error::InvalidBufferSize {
                expected: expected_len,
                actual: data.len(),
            });
        }

        // Push rows one at a time (this handles strip flushing correctly)
        for i in 0..num_rows {
            let start = i * self.bytes_per_row;
            let end = start + self.bytes_per_row;
            self.push_row_with_stop(&data[start..end], &stop)?;
        }

        Ok(())
    }

    /// Flushes the current strip buffer to the processor.
    fn flush_strip_with_stop(&mut self, stop: &impl Stop) -> Result<()> {
        stop.check()?;

        if self.rows_buffered == 0 {
            return Ok(());
        }

        let strip_data = &self.row_buffer[..self.rows_buffered * self.bytes_per_row];
        self.processor.process_strip(strip_data, self.current_y)?;

        self.current_y += self.rows_buffered;
        self.rows_buffered = 0;

        Ok(())
    }

    /// Finishes encoding and returns the JPEG data.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Not all rows have been pushed
    /// - JPEG generation fails
    pub fn finish(self) -> Result<Vec<u8>> {
        self.finish_with_stop(Never)
    }

    /// Finishes encoding with cancellation support.
    pub fn finish_with_stop(mut self, stop: impl Stop) -> Result<Vec<u8>> {
        stop.check()?;

        // Calculate total rows received
        let total_rows = self.current_y + self.rows_buffered;

        // Validate all rows were pushed before trying to process
        if total_rows < self.height {
            return Err(Error::IoError {
                reason: format!(
                    "only {} of {} rows were pushed",
                    total_rows,
                    self.height
                ),
            });
        }

        // Flush any remaining rows
        if self.rows_buffered > 0 {
            self.flush_strip_with_stop(&stop)?;
        }

        // Extract needed data before consuming processor
        let encoder = self.encoder;
        let y_quant = self.y_quant;
        let cb_quant = self.cb_quant;
        let cr_quant = self.cr_quant;
        let width = self.width;
        let height = self.height;

        // Finalize strip processing
        let strip_output = self.processor.finalize()?;

        // Build JPEG output using the encoder's internal methods
        Self::build_jpeg_from_blocks(
            &encoder,
            &y_quant,
            &cb_quant,
            &cr_quant,
            width,
            height,
            strip_output,
            stop,
        )
    }

    /// Builds JPEG output from processed blocks.
    fn build_jpeg_from_blocks(
        encoder: &Encoder,
        y_quant: &QuantTable,
        cb_quant: &QuantTable,
        cr_quant: &QuantTable,
        _width: usize,
        _height: usize,
        strip_output: crate::encode::strip::StripProcessorOutput,
        stop: impl Stop,
    ) -> Result<Vec<u8>> {
        stop.check()?;

        // Branch based on encoding mode (mirrors encode_strip_based in encode/mod.rs)
        match encoder.config.mode {
            JpegMode::Progressive => {
                // Use progressive encoding path
                encoder.encode_progressive_from_blocks(
                    &strip_output.y_blocks,
                    &strip_output.cb_blocks,
                    &strip_output.cr_blocks,
                    y_quant,
                    cb_quant,
                    cr_quant,
                )
            }
            _ => {
                // Baseline encoding path
                Self::build_jpeg_baseline(encoder, y_quant, cb_quant, cr_quant, strip_output)
            }
        }
    }

    /// Builds baseline JPEG output from processed blocks.
    fn build_jpeg_baseline(
        encoder: &Encoder,
        y_quant: &QuantTable,
        cb_quant: &QuantTable,
        cr_quant: &QuantTable,
        strip_output: crate::encode::strip::StripProcessorOutput,
    ) -> Result<Vec<u8>> {
        let is_color = encoder.config.pixel_format != PixelFormat::Gray;
        let width = encoder.config.width as usize;
        let height = encoder.config.height as usize;

        let mut output = Vec::with_capacity(width * height / 4);

        // Write JPEG headers
        encoder.write_header(&mut output)?;
        encoder.write_quant_tables(&mut output, y_quant, cb_quant, cr_quant)?;
        encoder.write_frame_header(&mut output)?;

        // Generate scan data with optimized or standard tables
        let scan_data = if encoder.config.optimize_huffman {
            let tables = encoder.build_optimized_tables(
                &strip_output.y_blocks,
                &strip_output.cb_blocks,
                &strip_output.cr_blocks,
                is_color,
            )?;

            encoder.write_huffman_tables_optimized(&mut output, &tables)?;

            if encoder.config.restart_interval > 0 {
                encoder.write_restart_interval(&mut output)?;
            }
            encoder.write_scan_header(&mut output)?;

            encoder.encode_with_tables(
                &strip_output.y_blocks,
                &strip_output.cb_blocks,
                &strip_output.cr_blocks,
                is_color,
                Some(&tables),
            )?
        } else {
            encoder.write_huffman_tables(&mut output)?;

            if encoder.config.restart_interval > 0 {
                encoder.write_restart_interval(&mut output)?;
            }
            encoder.write_scan_header(&mut output)?;

            encoder.encode_with_tables(
                &strip_output.y_blocks,
                &strip_output.cb_blocks,
                &strip_output.cr_blocks,
                is_color,
                None,
            )?
        };

        output.extend_from_slice(&scan_data);

        // Write EOI marker
        output.push(0xFF);
        output.push(crate::consts::MARKER_EOI);

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_encoder_creation() {
        let encoder = StreamingEncoder::new(640, 480).build();
        assert!(encoder.is_ok());
        let encoder = encoder.unwrap();
        assert_eq!(encoder.height(), 480);
        assert_eq!(encoder.bytes_per_row(), 640 * 3); // RGB default
    }

    #[test]
    fn test_streaming_encoder_420_strip_height() {
        let encoder = StreamingEncoder::new(640, 480)
            .subsampling(Subsampling::S420)
            .build()
            .unwrap();
        assert_eq!(encoder.strip_height(), 16);
    }

    #[test]
    fn test_streaming_encoder_444_strip_height() {
        let encoder = StreamingEncoder::new(640, 480)
            .subsampling(Subsampling::S444)
            .build()
            .unwrap();
        assert_eq!(encoder.strip_height(), 8);
    }

    #[test]
    fn test_streaming_encoder_wrong_row_size() {
        let mut encoder = StreamingEncoder::new(640, 480).build().unwrap();
        let wrong_row = vec![0u8; 100]; // Wrong size
        let result = encoder.push_row(&wrong_row);
        assert!(result.is_err());
    }

    #[test]
    fn test_streaming_encoder_too_many_rows() {
        let mut encoder = StreamingEncoder::new(4, 2).build().unwrap();
        let row = vec![128u8; 4 * 3]; // 4 pixels * 3 channels

        // Push first 2 rows (all of them)
        encoder.push_row(&row).unwrap();
        encoder.push_row(&row).unwrap();

        // Third row should fail
        let result = encoder.push_row(&row);
        assert!(result.is_err());
    }

    #[test]
    fn test_streaming_encoder_incomplete() {
        let mut encoder = StreamingEncoder::new(4, 4).build().unwrap();
        let row = vec![128u8; 4 * 3];

        // Push only 2 of 4 rows
        encoder.push_row(&row).unwrap();
        encoder.push_row(&row).unwrap();

        // finish() should fail
        let result = encoder.finish();
        assert!(result.is_err());
    }

    #[test]
    fn test_memory_estimate() {
        let estimate = StreamingEncoder::new(3840, 2160)
            .subsampling(Subsampling::S420)
            .estimate_memory_usage();

        // Should be around 26 MB for 4K with 4:2:0
        // Allow some tolerance for implementation details
        assert!(estimate > 20_000_000, "estimate {} too low", estimate);
        assert!(estimate < 40_000_000, "estimate {} too high", estimate);
    }

    #[test]
    fn test_streaming_matches_standard_small() {
        // Create a small test image
        let width = 32u32;
        let height = 32u32;
        let pixels: Vec<u8> = (0..width * height * 3)
            .map(|i| ((i * 17) % 256) as u8)
            .collect();

        // Encode with standard encoder
        #[allow(deprecated)]
        let standard_result = Encoder::new()
            .width(width)
            .height(height)
            .quality(Quality::from_quality(85.0))
            .subsampling(Subsampling::S444)
            .encoding_backend(EncodingBackend::Strip) // Use same backend for comparison
            .encode(&pixels)
            .unwrap();

        // Encode with streaming encoder
        let mut streaming = StreamingEncoder::new(width, height)
            .quality(Quality::from_quality(85.0))
            .subsampling(Subsampling::S444)
            .build()
            .unwrap();

        let row_size = width as usize * 3;
        for y in 0..height as usize {
            let start = y * row_size;
            let end = start + row_size;
            streaming.push_row(&pixels[start..end]).unwrap();
        }
        let streaming_result = streaming.finish().unwrap();

        // Results should be identical
        assert_eq!(
            standard_result.len(),
            streaming_result.len(),
            "output lengths differ"
        );
        assert_eq!(
            standard_result, streaming_result,
            "outputs differ"
        );
    }
}
