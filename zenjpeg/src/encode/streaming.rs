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
//! use zenjpeg::{StreamingEncoder, Quality, Subsampling};
//!
//! let mut encoder = StreamingEncoder::new(1920, 1080)
//!     .quality(Quality::ApproxJpegli(85.0))
//!     .subsampling(Subsampling::S420)
//!     .start()?;
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

#![allow(dead_code)]

use crate::encode::config::ComputedConfig;
use crate::encode::encoder_types::HuffmanStrategy;
use crate::encode::strip::StripProcessor;
use crate::error::{Error, Result};
use crate::quant::{self, QuantTable, ZeroBiasParams};
use crate::types::{ColorSpace, JpegMode, Subsampling};
use enough::{Stop, Unstoppable};

pub(crate) use super::streaming_builder::StreamingEncoderBuilder;

/// State for streaming-through encoding mode.
///
/// When present, blocks are entropy-encoded immediately on each strip flush
/// rather than buffered for a later two-pass Huffman optimization.
struct StreamingOutputState {
    /// BitWriter accumulates encoded scan data across strip flushes.
    writer: crate::foundation::bitstream::BitWriter,
    /// Huffman tables used for encoding (boxed: ~5.7 KB, uncommon path).
    tables: Box<crate::huffman::optimize::HuffmanTableSet>,
    /// Entropy encoding state (DC prediction, restart markers).
    entropy_state: crate::entropy::StreamingEntropyState,
    /// Total MCUs in the full image (for restart marker logic).
    total_mcus: usize,
    /// JPEG header bytes (SOI through SOS), written at construction time.
    header: Vec<u8>,
}

/// Streaming input JPEG encoder.
///
/// Accepts rows incrementally and outputs JPEG at the end.
/// Uses strip-based processing internally for low peak memory usage.
///
/// Two encoding modes:
/// - **Buffered** (default with `HuffmanStrategy::Optimize`): buffers all blocks,
///   builds optimal Huffman tables at `finish()`.
/// - **Streaming-through** (with `HuffmanStrategy::Custom` or `Fixed`, sequential only):
///   writes JPEG header at construction, encodes blocks immediately on each
///   strip flush. At `finish()`, just appends EOI.
pub(crate) struct StreamingEncoder {
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

    /// Configuration for JPEG output generation
    config: ComputedConfig,

    /// Quantization tables (generated from quality)
    y_quant: QuantTable,
    cb_quant: QuantTable,
    cr_quant: QuantTable,

    /// Streaming-through state. None = buffered mode (default).
    streaming: Option<StreamingOutputState>,
}

impl StreamingEncoder {
    /// Creates a new streaming encoder builder with the given dimensions.
    ///
    /// Use the builder methods to configure quality, subsampling, etc.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use zenjpeg::{StreamingEncoder, Quality, Subsampling};
    ///
    /// let encoder = StreamingEncoder::new(1920, 1080)
    ///     .quality(Quality::ApproxJpegli(85.0))
    ///     .subsampling(Subsampling::S420)
    ///     .start()?;
    /// ```
    #[must_use]
    #[allow(clippy::new_ret_no_self)] // Builder pattern: new() returns builder
    pub(crate) fn new(width: u32, height: u32) -> StreamingEncoderBuilder {
        StreamingEncoderBuilder::new(width, height)
    }

    /// Creates a streaming encoder from builder configuration.
    pub(crate) fn from_builder(builder: StreamingEncoderBuilder) -> Result<Self> {
        let width = builder.width as usize;
        let height = builder.height as usize;

        if width == 0 || height == 0 {
            return Err(Error::invalid_dimensions(
                builder.width,
                builder.height,
                "dimensions must be non-zero",
            ));
        }

        // Generate quantization tables and zero-bias params
        let is_420 = builder.subsampling == Subsampling::S420;
        let distance = builder.quality.to_distance();
        let color_space = if builder.use_xyb {
            ColorSpace::Xyb
        } else {
            ColorSpace::YCbCr
        };

        let allow_16bit = builder.allow_16bit_quant_tables;
        let ((y_quant, cb_quant, cr_quant), (y_zero_bias, cb_zero_bias, cr_zero_bias)) =
            if let Some(ref tables) = builder.encoding_tables {
                // Branch 1: Custom encoding tables provided explicitly
                let quant = tables.generate_quant_tables(distance, is_420);
                let zero_bias = tables.generate_zero_bias_all();
                // Apply allow_16bit clamping if needed
                let quant = if allow_16bit {
                    quant
                } else {
                    (
                        quant.0.clamp_to_baseline(),
                        quant.1.clamp_to_baseline(),
                        quant.2.clamp_to_baseline(),
                    )
                };
                (quant, zero_bias)
            } else if builder.quant_source == super::encoder_types::QuantTableSource::MozjpegDefault
            {
                // Branch 2: Mozjpeg Robidoux tables with quality scaling
                let quality_u8 = builder.quality.to_internal().round().clamp(1.0, 100.0) as u8;
                let force_baseline = !allow_16bit;
                let tables = super::tables::robidoux::generate_mozjpeg_default_tables(
                    quality_u8,
                    force_baseline,
                );
                let quant = tables.generate_quant_tables(distance, is_420);
                let zero_bias = tables.generate_zero_bias_all();
                (quant, zero_bias)
            } else {
                // Branch 3: Jpegli perceptual defaults (original path)
                //
                // When separate_chroma_tables is false (2-table mode, jpeg_set_quality),
                // use the Cr base matrix for both Cb and Cr tables. This matches C++
                // jpegli behavior where the single chroma table uses the Cr matrix.
                let cb_component = if builder.separate_chroma_tables { 1 } else { 2 };

                let quant = (
                    quant::generate_quant_table_ex(
                        builder.quality,
                        0,
                        color_space,
                        builder.use_xyb,
                        is_420,
                        allow_16bit,
                    ),
                    quant::generate_quant_table_ex(
                        builder.quality,
                        cb_component,
                        color_space,
                        builder.use_xyb,
                        is_420,
                        allow_16bit,
                    ),
                    quant::generate_quant_table_ex(
                        builder.quality,
                        2,
                        color_space,
                        builder.use_xyb,
                        is_420,
                        allow_16bit,
                    ),
                );

                // Compute effective distance for quality-adaptive zero bias
                let effective_distance =
                    quant::quant_vals_to_distance(&quant.0, &quant.1, &quant.2);

                // Auto-select zero bias based on color mode (matches C++ jpegli behavior)
                let zero_bias = if builder.use_xyb {
                    (
                        ZeroBiasParams::for_xyb(),
                        ZeroBiasParams::for_xyb(),
                        ZeroBiasParams::for_xyb(),
                    )
                } else {
                    (
                        ZeroBiasParams::for_ycbcr(effective_distance, 0),
                        ZeroBiasParams::for_ycbcr(effective_distance, 1),
                        ZeroBiasParams::for_ycbcr(effective_distance, 2),
                    )
                };

                (quant, zero_bias)
            };

        // Build quantization context (all tables + SIMD variants)
        let quant_ctx = crate::encode::strip::QuantContext::new(
            y_quant.clone(),
            cb_quant.clone(),
            cr_quant.clone(),
            y_zero_bias,
            cb_zero_bias,
            cr_zero_bias,
        );

        // Create strip processor with quant tables provided at construction
        let mut processor = StripProcessor::with_xyb(
            width,
            height,
            builder.subsampling,
            builder.pixel_format,
            builder.chroma_downsampling,
            builder.restart_interval,
            builder.use_xyb,
            quant_ctx,
            builder.aq_enabled,
        )?;

        // Set deringing (on by default in both builder and processor)
        processor.set_deringing(builder.deringing);

        // Set KLT decorrelation matrix if configured
        if let Some((matrix, mean_rgb)) = builder.klt_matrix {
            processor.set_klt_matrix(matrix, mean_rgb);
        }

        // Enable trellis quantization if configured
        #[cfg(feature = "trellis")]
        {
            if let Some(ref trellis) = builder.trellis {
                processor.set_trellis(*trellis);
            } else if builder.hybrid_config.enabled {
                processor.set_hybrid(builder.hybrid_config);
            }
        }

        let strip_height = processor.strip_height();
        let bytes_per_row = width * builder.pixel_format.bytes_per_pixel();

        // Allocate row buffer for one strip
        let row_buffer = vec![0u8; bytes_per_row * strip_height];

        // Create config for final JPEG output
        let config = ComputedConfig {
            width: builder.width,
            height: builder.height,
            pixel_format: builder.pixel_format,
            quality: builder.quality,
            subsampling: builder.subsampling,
            mode: builder.mode,
            huffman: builder.huffman.clone(),
            chroma_downsampling: builder.chroma_downsampling,
            restart_interval: builder.restart_interval,
            use_xyb: builder.use_xyb,
            klt_matrix: builder.klt_matrix,
            #[cfg(feature = "parallel")]
            parallel: builder.parallel,
            #[cfg(feature = "trellis")]
            hybrid_config: builder.hybrid_config,
            custom_aq_map: builder.custom_aq_map,
            #[cfg(feature = "trellis")]
            trellis: builder.trellis,
            encoding_tables: builder.encoding_tables,
            edge_padding: crate::types::EdgePaddingConfig::default(),
            original_width: None,
            original_height: None,
            allow_16bit_quant_tables: builder.allow_16bit_quant_tables,
            separate_chroma_tables: builder.separate_chroma_tables,
            scan_strategy: builder.scan_strategy,
        };

        // Determine if we can use streaming-through encoding:
        // - Need known Huffman tables (Custom or Fixed)
        // - Must be sequential (progressive needs multi-pass)
        // - Not XYB (different header/table structure)
        let enable_streaming = !matches!(builder.huffman, HuffmanStrategy::Optimize)
            && builder.mode != JpegMode::Progressive
            && !builder.use_xyb
            && builder.klt_matrix.is_none();

        let streaming = if enable_streaming {
            // Get tables: custom if provided, otherwise standard JPEG tables
            let tables = match builder.huffman {
                HuffmanStrategy::Custom(tables) => tables,
                HuffmanStrategy::Fixed => Box::new(crate::huffman::builtin_tables::select_tables(
                    &builder.quality,
                    builder.use_xyb,
                    builder.subsampling,
                )),
                HuffmanStrategy::FixedAnnexK => {
                    // Annex K tables are well-defined constants, cannot fail
                    Box::new(
                        crate::huffman::optimize::HuffmanTableSet::annex_k()
                            .expect("JPEG Annex K tables are constant and valid"),
                    )
                }
                HuffmanStrategy::Optimize => unreachable!(),
            };

            // Write JPEG header (SOI through SOS) into buffer
            let mut header = Vec::new();
            config.write_header(&mut header)?;
            config.write_quant_tables(&mut header, &y_quant, &cb_quant, &cr_quant)?;
            let is_extended =
                y_quant.precision > 0 || cb_quant.precision > 0 || cr_quant.precision > 0;
            config.write_frame_header_ex(&mut header, is_extended)?;
            config.write_huffman_tables_optimized(&mut header, &tables)?;
            if config.restart_interval > 0 {
                config.write_restart_interval(&mut header)?;
            }
            config.write_scan_header(&mut header)?;

            Some(StreamingOutputState {
                writer: crate::foundation::bitstream::BitWriter::new(),
                tables,
                entropy_state: crate::entropy::StreamingEntropyState::new(),
                total_mcus: processor.layout.total_mcus,
                header,
            })
        } else {
            None
        };

        Ok(Self {
            width,
            height,
            bytes_per_row,
            strip_height,
            row_buffer,
            rows_buffered: 0,
            current_y: 0,
            processor,
            config,
            y_quant,
            cb_quant,
            cr_quant,
            streaming,
        })
    }

    /// Returns the number of rows pushed so far.
    #[must_use]
    pub(crate) fn rows_pushed(&self) -> usize {
        self.current_y + self.rows_buffered
    }

    /// Returns the expected number of bytes per row.
    #[must_use]
    pub(crate) fn bytes_per_row(&self) -> usize {
        self.bytes_per_row
    }

    /// Returns the total height of the image.
    #[must_use]
    pub(crate) fn height(&self) -> usize {
        self.height
    }

    /// Returns the strip height (internal processing unit).
    #[must_use]
    pub(crate) fn strip_height(&self) -> usize {
        self.strip_height
    }

    /// Returns allocation statistics from the strip processor.
    ///
    /// This tracks all major allocations made during encoding setup,
    /// including color plane buffers, DCT block storage, and AQ buffers.
    #[must_use]
    pub(crate) fn encode_stats(&self) -> &crate::foundation::alloc::EncodeStats {
        self.processor.encode_stats()
    }

    /// Returns whether this encoder is in streaming-through mode.
    ///
    /// In streaming mode, blocks are encoded immediately on each strip flush.
    /// In buffered mode (default), all blocks are buffered and encoded at `finish()`.
    #[must_use]
    pub(crate) fn is_streaming(&self) -> bool {
        self.streaming.is_some()
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
    pub(crate) fn push_row(&mut self, row: &[u8]) -> Result<()> {
        self.push_row_with_stop(row, Unstoppable)
    }

    /// Pushes a single row with cancellation support.
    ///
    /// The `stop` source is checked before processing each strip.
    /// Returns `Error::cancelled()` if cancellation is requested.
    pub(crate) fn push_row_with_stop(&mut self, row: &[u8], stop: impl Stop) -> Result<()> {
        // Check cancellation
        stop.check()?;

        // Validate row size
        if row.len() != self.bytes_per_row {
            return Err(Error::invalid_buffer_size(self.bytes_per_row, row.len()));
        }

        // Check if we've already received all rows
        if self.current_y + self.rows_buffered >= self.height {
            return Err(Error::io_error(format!(
                "already received all {} rows",
                self.height
            )));
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
    pub(crate) fn push_rows(&mut self, data: &[u8], num_rows: usize) -> Result<()> {
        self.push_rows_with_stop(data, num_rows, Unstoppable)
    }

    /// Pushes multiple rows with cancellation support.
    ///
    /// This method is optimized to process complete strips directly from the input
    /// buffer without intermediate copies. Only partial strips at the beginning
    /// and end require buffering.
    pub(crate) fn push_rows_with_stop(
        &mut self,
        data: &[u8],
        num_rows: usize,
        stop: impl Stop,
    ) -> Result<()> {
        let expected_len = num_rows * self.bytes_per_row;
        if data.len() != expected_len {
            return Err(Error::invalid_buffer_size(expected_len, data.len()));
        }

        if num_rows == 0 {
            return Ok(());
        }

        // Check if we've already received all rows
        if self.current_y + self.rows_buffered >= self.height {
            return Err(Error::io_error(format!(
                "already received all {} rows",
                self.height
            )));
        }

        let mut data_offset = 0usize;
        let mut rows_remaining = num_rows;

        // Step 1: Complete any partial strip in buffer
        if self.rows_buffered > 0 {
            let rows_to_complete = (self.strip_height - self.rows_buffered).min(rows_remaining);
            let rows_to_complete =
                rows_to_complete.min(self.height - self.current_y - self.rows_buffered);

            // Copy rows to buffer to complete the strip
            let buf_offset = self.rows_buffered * self.bytes_per_row;
            let src_bytes = rows_to_complete * self.bytes_per_row;
            self.row_buffer[buf_offset..buf_offset + src_bytes]
                .copy_from_slice(&data[data_offset..data_offset + src_bytes]);

            self.rows_buffered += rows_to_complete;
            data_offset += src_bytes;
            rows_remaining -= rows_to_complete;

            // Flush if strip is complete
            let remaining_height = self.height - self.current_y;
            if self.rows_buffered >= self.strip_height || self.rows_buffered >= remaining_height {
                self.flush_strip_with_stop(&stop)?;
            }
        }

        // Step 2: Process complete strips directly from input (no copy!)
        while rows_remaining >= self.strip_height {
            stop.check()?;

            let remaining_height = self.height - self.current_y;
            let strip_rows = self.strip_height.min(remaining_height);

            if strip_rows == 0 {
                break;
            }

            let strip_bytes = strip_rows * self.bytes_per_row;
            let strip_data = &data[data_offset..data_offset + strip_bytes];

            // Process directly from input buffer
            self.processor.process_strip(strip_data, self.current_y)?;
            self.current_y += strip_rows;

            data_offset += strip_bytes;
            rows_remaining -= strip_rows;
        }

        // Step 3: Buffer any remaining partial rows
        if rows_remaining > 0 {
            let remaining_height = self.height - self.current_y;
            let rows_to_buffer = rows_remaining.min(remaining_height);

            if rows_to_buffer > 0 {
                let src_bytes = rows_to_buffer * self.bytes_per_row;
                self.row_buffer[..src_bytes]
                    .copy_from_slice(&data[data_offset..data_offset + src_bytes]);
                self.rows_buffered = rows_to_buffer;

                // Check if this is the final partial strip
                if rows_to_buffer >= remaining_height {
                    self.flush_strip_with_stop(&stop)?;
                }
            }
        }

        Ok(())
    }

    /// Pushes a strip of YCbCr f32 planar data.
    ///
    /// This bypasses RGB→YCbCr conversion, accepting YCbCr data directly.
    /// Values should be in centered range [-128, 127].
    ///
    /// # Arguments
    /// * `y` - Y plane data (width × num_rows floats)
    /// * `cb` - Cb plane data (width × num_rows floats, full resolution)
    /// * `cr` - Cr plane data (width × num_rows floats, full resolution)
    /// * `num_rows` - Number of rows in this strip
    ///
    /// # Note
    ///
    /// Unlike `push_row` which buffers internally, this method processes
    /// the strip immediately. For optimal performance, push `strip_height()`
    /// rows at a time.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - RGB rows are already buffered (can't mix RGB and YCbCr input)
    /// - Plane sizes don't match expected dimensions
    /// - XYB mode is enabled (requires RGB input)
    pub(crate) fn push_ycbcr_strip_f32(
        &mut self,
        y: &[f32],
        cb: &[f32],
        cr: &[f32],
        num_rows: usize,
    ) -> Result<()> {
        // Can't mix RGB and YCbCr input
        if self.rows_buffered > 0 {
            return Err(Error::internal(
                "cannot mix RGB and YCbCr input (RGB rows buffered)",
            ));
        }

        // Validate we haven't received all rows yet
        if self.current_y >= self.height {
            return Err(Error::io_error(format!(
                "already received all {} rows",
                self.height
            )));
        }

        // Clamp to remaining rows
        let actual_rows = num_rows.min(self.height - self.current_y);

        // Validate plane sizes
        let expected_size = self.width * actual_rows;
        if y.len() < expected_size {
            return Err(Error::invalid_buffer_size(expected_size, y.len()));
        }
        if cb.len() < expected_size || cr.len() < expected_size {
            return Err(Error::invalid_buffer_size(
                expected_size,
                cb.len().min(cr.len()),
            ));
        }

        // Process in chunks of strip_height rows
        let mut processed = 0;
        while processed < actual_rows {
            let remaining = self.height - self.current_y;
            let strip_rows = self
                .strip_height
                .min(actual_rows - processed)
                .min(remaining);

            let start = processed * self.width;
            let end = start + strip_rows * self.width;

            self.processor.process_strip_ycbcr_f32(
                &y[start..end],
                &cb[start..end],
                &cr[start..end],
                self.current_y,
            )?;

            self.current_y += strip_rows;
            processed += strip_rows;
        }

        Ok(())
    }

    /// Pushes a strip of pre-downsampled YCbCr f32 planar data.
    ///
    /// This accepts chroma data that is already downsampled according to the
    /// subsampling mode. Skips the internal chroma downsampling step.
    ///
    /// # Arguments
    /// * `y` - Y plane data (width × num_rows floats)
    /// * `cb` - Cb plane data (chroma_width × chroma_rows floats)
    /// * `cr` - Cr plane data (chroma_width × chroma_rows floats)
    /// * `num_rows` - Number of Y rows in this strip
    ///
    /// # Chroma Dimensions
    /// - 4:4:4: cb/cr at full width × full height
    /// - 4:2:2: cb/cr at width/2 × full height
    /// - 4:2:0: cb/cr at width/2 × height/2
    pub(crate) fn push_ycbcr_strip_f32_subsampled(
        &mut self,
        y: &[f32],
        cb: &[f32],
        cr: &[f32],
        num_rows: usize,
    ) -> Result<()> {
        // Can't mix RGB and YCbCr input
        if self.rows_buffered > 0 {
            return Err(Error::internal(
                "cannot mix RGB and YCbCr input (RGB rows buffered)",
            ));
        }

        // Validate we haven't received all rows yet
        if self.current_y >= self.height {
            return Err(Error::io_error(format!(
                "already received all {} rows",
                self.height
            )));
        }

        // Clamp to remaining rows
        let actual_rows = num_rows.min(self.height - self.current_y);

        // Validate Y plane size
        let expected_y_size = self.width * actual_rows;
        if y.len() < expected_y_size {
            return Err(Error::invalid_buffer_size(expected_y_size, y.len()));
        }

        // Get subsampling info for chroma slicing
        let subsampling = self.processor.subsampling();
        let chroma_width = match subsampling {
            Subsampling::S444 | Subsampling::S440 => self.width,
            Subsampling::S422 | Subsampling::S420 => (self.width + 1) / 2,
        };
        let chroma_h_factor = match subsampling {
            Subsampling::S444 | Subsampling::S422 => 1,
            Subsampling::S420 | Subsampling::S440 => 2,
        };

        // Process in chunks of strip_height rows
        let mut y_processed = 0;
        let mut chroma_processed = 0;
        while y_processed < actual_rows {
            let remaining = self.height - self.current_y;
            let strip_rows = self
                .strip_height
                .min(actual_rows - y_processed)
                .min(remaining);

            let y_start = y_processed * self.width;
            let y_end = y_start + strip_rows * self.width;

            // Calculate chroma rows for this strip
            let chroma_rows = (strip_rows + chroma_h_factor - 1) / chroma_h_factor;
            let c_start = chroma_processed * chroma_width;
            let c_end = c_start + chroma_rows * chroma_width;

            self.processor.process_strip_ycbcr_f32_subsampled(
                &y[y_start..y_end],
                &cb[c_start..c_end.min(cb.len())],
                &cr[c_start..c_end.min(cr.len())],
                self.current_y,
            )?;

            self.current_y += strip_rows;
            y_processed += strip_rows;
            chroma_processed += chroma_rows;
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

        // In streaming mode, encode the new blocks immediately and free them
        if self.streaming.is_some() {
            self.encode_new_blocks_streaming()?;
        }

        self.current_y += self.rows_buffered;
        self.rows_buffered = 0;

        Ok(())
    }

    /// Encodes newly-produced blocks in streaming mode.
    ///
    /// Takes all blocks from the strip processor and encodes them to the
    /// BitWriter, freeing the block memory immediately.
    fn encode_new_blocks_streaming(&mut self) -> Result<()> {
        let blocks = self.processor.take_blocks();
        let is_color = !self.config.pixel_format.is_grayscale();
        let width = self.width;
        let subsampling = self.config.subsampling;
        let restart_interval = self.config.restart_interval;

        let state = self.streaming.as_mut().unwrap();
        crate::entropy::encode_blocks_mcu_order(
            &blocks.y_blocks,
            &blocks.cb_blocks,
            &blocks.cr_blocks,
            &state.tables,
            &mut state.writer,
            is_color,
            &mut state.entropy_state,
            subsampling,
            width,
            restart_interval,
            state.total_mcus,
        )?;

        Ok(())
    }

    /// Finishes encoding and returns the JPEG data.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Not all rows have been pushed
    /// - JPEG generation fails
    pub(crate) fn finish(self) -> Result<Vec<u8>> {
        self.finish_with_stop(Unstoppable)
    }

    /// Finishes encoding and returns both the JPEG and the frequency counts.
    ///
    /// For sequential mode with `HuffmanStrategy::Optimize`, the counts come from
    /// the optimization pass at no extra cost. For progressive mode, the blocks
    /// are counted separately (same quantized blocks, same symbol distribution).
    ///
    /// Returns `None` for counts if:
    /// - Streaming-through mode was used (no buffered blocks to count)
    pub(crate) fn finish_with_huffman_frequencies(
        self,
    ) -> Result<(
        Vec<u8>,
        Option<Box<super::blocks::HuffmanSymbolFrequencies>>,
    )> {
        let mut output = Vec::new();
        let counts = self.finish_into_with_huffman_frequencies(&mut output, Unstoppable)?;
        Ok((output, counts))
    }

    /// Finishes encoding with cancellation support.
    pub(crate) fn finish_with_stop(self, stop: impl Stop) -> Result<Vec<u8>> {
        let mut output = Vec::new();
        self.finish_into_with_stop(&mut output, stop)?;
        Ok(output)
    }

    /// Finishes encoding, writing directly to the provided buffer.
    ///
    /// This avoids an extra allocation compared to `finish()`. The buffer
    /// is cleared before writing.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Not all rows have been pushed
    /// - JPEG generation fails
    /// - Memory allocation fails
    pub(crate) fn finish_into(self, output: &mut Vec<u8>) -> Result<()> {
        self.finish_into_with_stop(output, Unstoppable)
    }

    /// Finishes encoding into provided buffer with cancellation support.
    pub(crate) fn finish_into_with_stop(
        mut self,
        output: &mut Vec<u8>,
        stop: impl Stop,
    ) -> Result<()> {
        stop.check()?;

        // Calculate total rows received
        let total_rows = self.current_y + self.rows_buffered;

        // Validate all rows were pushed before trying to process
        if total_rows < self.height {
            return Err(Error::io_error(format!(
                "only {} of {} rows were pushed",
                total_rows, self.height
            )));
        }

        // Flush any remaining rows (in streaming mode, this also encodes them)
        if self.rows_buffered > 0 {
            self.flush_strip_with_stop(&stop)?;
        }

        if let Some(mut streaming) = self.streaming.take() {
            // Streaming mode: finalize the strip processor to flush the last
            // pending iMCU (AQ delays blocks by one strip), then encode them
            let is_color = !self.config.pixel_format.is_grayscale();
            let width = self.width;
            let subsampling = self.config.subsampling;
            let restart_interval = self.config.restart_interval;

            let final_output = self.processor.finalize()?;
            if !final_output.y_blocks.is_empty() {
                crate::entropy::encode_blocks_mcu_order(
                    &final_output.y_blocks,
                    &final_output.cb_blocks,
                    &final_output.cr_blocks,
                    &streaming.tables,
                    &mut streaming.writer,
                    is_color,
                    &mut streaming.entropy_state,
                    subsampling,
                    width,
                    restart_interval,
                    streaming.total_mcus,
                )?;
            }

            // header + scan data + EOI
            Self::finish_streaming_static(streaming, output)
        } else {
            // Buffered mode: build complete JPEG from all blocks
            let config = self.config;
            let y_quant = self.y_quant;
            let cb_quant = self.cb_quant;
            let cr_quant = self.cr_quant;
            let width = self.width;
            let height = self.height;

            let strip_output = self.processor.finalize()?;

            Self::build_jpeg_from_blocks_into(
                &config,
                &y_quant,
                &cb_quant,
                &cr_quant,
                width,
                height,
                strip_output,
                output,
                stop,
            )
        }
    }

    /// Like `finish_into_with_stop`, but also returns frequency counts from
    /// the Huffman optimization pass when in buffered + `HuffmanStrategy::Optimize` mode.
    pub(crate) fn finish_into_with_huffman_frequencies(
        mut self,
        output: &mut Vec<u8>,
        stop: impl Stop,
    ) -> Result<Option<Box<super::blocks::HuffmanSymbolFrequencies>>> {
        stop.check()?;

        let total_rows = self.current_y + self.rows_buffered;
        if total_rows < self.height {
            return Err(Error::io_error(format!(
                "only {} of {} rows were pushed",
                total_rows, self.height
            )));
        }

        if self.rows_buffered > 0 {
            self.flush_strip_with_stop(&stop)?;
        }

        if let Some(mut streaming) = self.streaming.take() {
            // Streaming mode - no buffered blocks, no counts available
            let is_color = !self.config.pixel_format.is_grayscale();
            let width = self.width;
            let subsampling = self.config.subsampling;
            let restart_interval = self.config.restart_interval;

            let final_output = self.processor.finalize()?;
            if !final_output.y_blocks.is_empty() {
                crate::entropy::encode_blocks_mcu_order(
                    &final_output.y_blocks,
                    &final_output.cb_blocks,
                    &final_output.cr_blocks,
                    &streaming.tables,
                    &mut streaming.writer,
                    is_color,
                    &mut streaming.entropy_state,
                    subsampling,
                    width,
                    restart_interval,
                    streaming.total_mcus,
                )?;
            }
            Self::finish_streaming_static(streaming, output)?;
            Ok(None)
        } else {
            // Buffered mode: extract frequency counts from the optimize pass
            let config = self.config;
            let y_quant = self.y_quant;
            let cb_quant = self.cb_quant;
            let cr_quant = self.cr_quant;
            let strip_output = self.processor.finalize()?;

            match config.mode {
                JpegMode::Progressive => {
                    if !matches!(config.huffman, HuffmanStrategy::Optimize) {
                        return Err(Error::unsupported_feature(
                            "Progressive mode requires optimized Huffman tables",
                        ));
                    }

                    // Count frequencies from the buffered blocks. The symbol
                    // distribution is the same regardless of scan structure,
                    // so single-scan counts are useful for general-purpose table training.
                    let is_color = !config.pixel_format.is_grayscale();
                    let counts = Box::new(config.count_block_frequencies(
                        &strip_output.y_blocks,
                        &strip_output.cb_blocks,
                        &strip_output.cr_blocks,
                        is_color,
                    ));

                    config.encode_progressive_from_blocks_into(
                        &strip_output.y_blocks,
                        &strip_output.cb_blocks,
                        &strip_output.cr_blocks,
                        &y_quant,
                        &cb_quant,
                        &cr_quant,
                        output,
                    )?;
                    Ok(Some(counts))
                }
                _ => {
                    // Sequential: collect_frequencies=true gets the counts
                    // from the optimize pass at no extra cost.
                    Self::build_jpeg_sequential_into(
                        &config,
                        &y_quant,
                        &cb_quant,
                        &cr_quant,
                        strip_output,
                        output,
                        true,
                    )
                }
            }
        }
    }

    /// Finishes streaming-through encoding.
    ///
    /// Combines the pre-written header with the scan data from the BitWriter
    /// and appends the EOI marker.
    fn finish_streaming_static(
        streaming: StreamingOutputState,
        output: &mut Vec<u8>,
    ) -> Result<()> {
        // Flush the BitWriter's remaining bits and scan data
        let scan_data = streaming.writer.into_bytes();

        // Assemble: header + scan data + EOI
        let total_size = streaming.header.len() + scan_data.len() + 2;
        output.clear();
        output
            .try_reserve(total_size)
            .map_err(|_| Error::allocation_failed(total_size, "streaming finish output"))?;

        output.extend_from_slice(&streaming.header);
        output.extend_from_slice(&scan_data);
        output.push(0xFF);
        output.push(crate::foundation::consts::MARKER_EOI);

        Ok(())
    }

    /// Builds JPEG output from processed blocks.
    fn build_jpeg_from_blocks(
        config: &ComputedConfig,
        y_quant: &QuantTable,
        cb_quant: &QuantTable,
        cr_quant: &QuantTable,
        width: usize,
        height: usize,
        strip_output: crate::encode::strip::StripProcessorOutput,
        stop: impl Stop,
    ) -> Result<Vec<u8>> {
        let mut output = Vec::new();
        Self::build_jpeg_from_blocks_into(
            config,
            y_quant,
            cb_quant,
            cr_quant,
            width,
            height,
            strip_output,
            &mut output,
            stop,
        )?;
        Ok(output)
    }

    /// Builds JPEG output from processed blocks into provided buffer.
    fn build_jpeg_from_blocks_into(
        config: &ComputedConfig,
        y_quant: &QuantTable,
        cb_quant: &QuantTable,
        cr_quant: &QuantTable,
        _width: usize,
        _height: usize,
        strip_output: crate::encode::strip::StripProcessorOutput,
        output: &mut Vec<u8>,
        stop: impl Stop,
    ) -> Result<()> {
        stop.check()?;

        // Branch based on encoding mode (mirrors encode_strip_based in encode/mod.rs)
        match config.mode {
            JpegMode::Progressive => {
                // Progressive mode requires optimized Huffman tables
                if !matches!(config.huffman, HuffmanStrategy::Optimize) {
                    return Err(Error::unsupported_feature(
                        "Progressive mode requires optimized Huffman tables",
                    ));
                }
                // Use progressive encoding path
                config.encode_progressive_from_blocks_into(
                    &strip_output.y_blocks,
                    &strip_output.cb_blocks,
                    &strip_output.cr_blocks,
                    y_quant,
                    cb_quant,
                    cr_quant,
                    output,
                )
            }
            _ => {
                // Sequential encoding
                Self::build_jpeg_sequential_into(
                    config,
                    y_quant,
                    cb_quant,
                    cr_quant,
                    strip_output,
                    output,
                    false,
                )?;
                Ok(())
            }
        }
    }

    /// Builds sequential JPEG output from processed blocks.
    fn build_jpeg_sequential(
        config: &ComputedConfig,
        y_quant: &QuantTable,
        cb_quant: &QuantTable,
        cr_quant: &QuantTable,
        strip_output: crate::encode::strip::StripProcessorOutput,
    ) -> Result<Vec<u8>> {
        let mut output = Vec::new();
        Self::build_jpeg_sequential_into(
            config,
            y_quant,
            cb_quant,
            cr_quant,
            strip_output,
            &mut output,
            false,
        )?;
        Ok(output)
    }

    /// Builds sequential JPEG output from processed blocks into provided buffer.
    ///
    /// When `collect_frequencies` is true, the YCbCr optimized Huffman path
    /// returns the symbol frequencies used to build the tables (at no extra
    /// cost—they are produced during the normal optimization pass).
    fn build_jpeg_sequential_into(
        config: &ComputedConfig,
        y_quant: &QuantTable,
        cb_quant: &QuantTable,
        cr_quant: &QuantTable,
        strip_output: crate::encode::strip::StripProcessorOutput,
        output: &mut Vec<u8>,
        collect_frequencies: bool,
    ) -> Result<Option<Box<super::blocks::HuffmanSymbolFrequencies>>> {
        let width = config.width as usize;
        let height = config.height as usize;

        output.clear();
        output
            .try_reserve(width * height / 4)
            .map_err(|_| Error::allocation_failed(width * height / 4, "sequential jpeg output"))?;

        let (scan_data, frequencies) = if config.klt_matrix.is_some() {
            Self::encode_sequential_klt(
                config,
                y_quant,
                cb_quant,
                cr_quant,
                &strip_output,
                output,
                collect_frequencies,
            )?
        } else if config.use_xyb {
            Self::encode_sequential_xyb(
                config,
                y_quant,
                cb_quant,
                cr_quant,
                &strip_output,
                output,
                collect_frequencies,
            )?
        } else {
            Self::encode_sequential_ycbcr(
                config,
                y_quant,
                cb_quant,
                cr_quant,
                &strip_output,
                output,
                collect_frequencies,
            )?
        };

        output.extend_from_slice(&scan_data);

        // Write EOI marker
        output.push(0xFF);
        output.push(crate::foundation::consts::MARKER_EOI);

        Ok(frequencies)
    }

    /// Encodes sequential JPEG in XYB color mode.
    ///
    /// Writes XYB-specific headers (Adobe APP14, XYB ICC profile, XYB frame/scan
    /// headers) and encodes using XYB-specific table building and entropy coding.
    /// XYB mode never collects frequency tables.
    fn encode_sequential_xyb(
        config: &ComputedConfig,
        y_quant: &QuantTable,
        cb_quant: &QuantTable,
        cr_quant: &QuantTable,
        strip_output: &crate::encode::strip::StripProcessorOutput,
        output: &mut Vec<u8>,
        collect_frequencies: bool,
    ) -> Result<(
        Vec<u8>,
        Option<Box<super::blocks::HuffmanSymbolFrequencies>>,
    )> {
        config.write_header_xyb(output)?;
        config.write_app14_adobe(output, 0)?;
        config.write_icc_profile(output, &crate::foundation::consts::XYB_ICC_PROFILE)?;
        config.write_quant_tables_xyb(output, y_quant, cb_quant, cr_quant)?;

        // Use SOF1 if any quant table needs 16-bit precision
        let is_extended = y_quant.precision > 0 || cb_quant.precision > 0 || cr_quant.precision > 0;
        config.write_frame_header_xyb_ex(output, is_extended)?;

        if matches!(config.huffman, HuffmanStrategy::Optimize) {
            let (dc_table, ac_table, frequencies) = if collect_frequencies {
                let (dc, ac, f) = config.build_optimized_tables_xyb_raster_with_counts(
                    &strip_output.y_blocks,
                    &strip_output.cb_blocks,
                    &strip_output.cr_blocks,
                )?;
                (dc, ac, Some(f))
            } else {
                let (dc, ac) = config.build_optimized_tables_xyb_raster(
                    &strip_output.y_blocks,
                    &strip_output.cb_blocks,
                    &strip_output.cr_blocks,
                )?;
                (dc, ac, None)
            };

            config.write_huffman_tables_xyb_optimized(output, &dc_table, &ac_table);

            if config.restart_interval > 0 {
                config.write_restart_interval(output)?;
            }
            config.write_scan_header_xyb(output)?;

            let scan_data = config.encode_with_tables_xyb_raster(
                &strip_output.y_blocks,
                &strip_output.cb_blocks,
                &strip_output.cr_blocks,
                &dc_table,
                &ac_table,
            )?;
            Ok((scan_data, frequencies))
        } else if let HuffmanStrategy::Custom(ref tables) = config.huffman {
            // Custom tables: XYB uses dc_luma/ac_luma as the shared pair.
            config.write_huffman_tables_xyb_optimized(output, &tables.dc_luma, &tables.ac_luma);

            if config.restart_interval > 0 {
                config.write_restart_interval(output)?;
            }
            config.write_scan_header_xyb(output)?;

            let scan_data = config.encode_with_tables_xyb_raster(
                &strip_output.y_blocks,
                &strip_output.cb_blocks,
                &strip_output.cr_blocks,
                &tables.dc_luma,
                &tables.ac_luma,
            )?;
            Ok((scan_data, None))
        } else {
            // Fixed: use general-purpose trained tables for XYB
            let tables = crate::huffman::builtin_tables::select_tables(
                &config.quality,
                true,
                config.subsampling,
            );
            config.write_huffman_tables_xyb_optimized(output, &tables.dc_luma, &tables.ac_luma);

            if config.restart_interval > 0 {
                config.write_restart_interval(output)?;
            }
            config.write_scan_header_xyb(output)?;

            let scan_data = config.encode_with_tables_xyb_raster(
                &strip_output.y_blocks,
                &strip_output.cb_blocks,
                &strip_output.cr_blocks,
                &tables.dc_luma,
                &tables.ac_luma,
            )?;
            Ok((scan_data, None))
        }
    }

    /// Encodes sequential JPEG in KLT custom color transform mode.
    ///
    /// Writes RGB-style headers (APP14 Adobe transform=0, KLT ICC profile, RGB
    /// component IDs) but uses the standard YCbCr entropy coding path since the
    /// KLT output channels have similar luma/chroma statistical properties.
    fn encode_sequential_klt(
        config: &ComputedConfig,
        y_quant: &QuantTable,
        cb_quant: &QuantTable,
        cr_quant: &QuantTable,
        strip_output: &crate::encode::strip::StripProcessorOutput,
        output: &mut Vec<u8>,
        collect_frequencies: bool,
    ) -> Result<(
        Vec<u8>,
        Option<Box<super::blocks::HuffmanSymbolFrequencies>>,
    )> {
        let is_color = true; // KLT always produces 3 components

        // Build the KLT ICC profile with per-channel scaling
        let (klt_matrix, mean_rgb) = config
            .klt_matrix
            .as_ref()
            .expect("KLT path requires klt_matrix");
        let encode_params = crate::color::klt::KltEncodeParams::from_forward_with_center(*klt_matrix, *mean_rgb);
        let icc_profile =
            crate::color::icc_builder::build_klt_icc_profile(&encode_params, "zenjpeg KLT");

        // Write headers: SOI, APP14 Adobe (transform=0), ICC profile, quant tables, SOF, DHT, SOS
        config.write_header(output)?;
        config.write_app14_adobe(output, 0)?;
        config.write_icc_profile(output, &icc_profile)?;
        config.write_quant_tables(output, y_quant, cb_quant, cr_quant)?;

        let is_extended = y_quant.precision > 0 || cb_quant.precision > 0 || cr_quant.precision > 0;
        config.write_frame_header_klt_ex(output, is_extended)?;

        // Huffman tables and entropy coding reuse the YCbCr path
        if matches!(config.huffman, HuffmanStrategy::Optimize) {
            let (tables, frequencies) = if collect_frequencies {
                let (t, f) = config.build_optimized_tables_with_counts(
                    &strip_output.y_blocks,
                    &strip_output.cb_blocks,
                    &strip_output.cr_blocks,
                    is_color,
                )?;
                (t, Some(f))
            } else {
                let t = config.build_optimized_tables(
                    &strip_output.y_blocks,
                    &strip_output.cb_blocks,
                    &strip_output.cr_blocks,
                    is_color,
                )?;
                (t, None)
            };

            config.write_huffman_tables_optimized(output, &tables)?;

            if config.restart_interval > 0 {
                config.write_restart_interval(output)?;
            }
            config.write_scan_header_klt(output)?;

            let scan_data = config.encode_with_tables(
                &strip_output.y_blocks,
                &strip_output.cb_blocks,
                &strip_output.cr_blocks,
                is_color,
                Some(&tables),
            )?;
            Ok((scan_data, frequencies))
        } else if let HuffmanStrategy::Custom(ref tables) = config.huffman {
            config.write_huffman_tables_optimized(output, tables)?;

            if config.restart_interval > 0 {
                config.write_restart_interval(output)?;
            }
            config.write_scan_header_klt(output)?;

            let scan_data = config.encode_with_tables(
                &strip_output.y_blocks,
                &strip_output.cb_blocks,
                &strip_output.cr_blocks,
                is_color,
                Some(tables),
            )?;
            Ok((scan_data, None))
        } else {
            let tables = crate::huffman::builtin_tables::select_tables(
                &config.quality,
                false,
                config.subsampling,
            );
            config.write_huffman_tables_optimized(output, &tables)?;

            if config.restart_interval > 0 {
                config.write_restart_interval(output)?;
            }
            config.write_scan_header_klt(output)?;

            let scan_data = config.encode_with_tables(
                &strip_output.y_blocks,
                &strip_output.cb_blocks,
                &strip_output.cr_blocks,
                is_color,
                Some(&tables),
            )?;
            Ok((scan_data, None))
        }
    }

    /// Encodes sequential JPEG in YCbCr color mode.
    ///
    /// Writes standard JPEG headers and encodes using standard table building
    /// and entropy coding. When `collect_frequencies` is true, returns the
    /// symbol frequencies from the Huffman optimization pass at no extra cost.
    fn encode_sequential_ycbcr(
        config: &ComputedConfig,
        y_quant: &QuantTable,
        cb_quant: &QuantTable,
        cr_quant: &QuantTable,
        strip_output: &crate::encode::strip::StripProcessorOutput,
        output: &mut Vec<u8>,
        collect_frequencies: bool,
    ) -> Result<(
        Vec<u8>,
        Option<Box<super::blocks::HuffmanSymbolFrequencies>>,
    )> {
        let is_color = !config.pixel_format.is_grayscale();

        config.write_header(output)?;
        config.write_quant_tables(output, y_quant, cb_quant, cr_quant)?;

        // Use SOF1 if any quant table needs 16-bit precision
        let is_extended = y_quant.precision > 0 || cb_quant.precision > 0 || cr_quant.precision > 0;
        config.write_frame_header_ex(output, is_extended)?;

        if matches!(config.huffman, HuffmanStrategy::Optimize) {
            let (tables, frequencies) = if collect_frequencies {
                let (t, f) = config.build_optimized_tables_with_counts(
                    &strip_output.y_blocks,
                    &strip_output.cb_blocks,
                    &strip_output.cr_blocks,
                    is_color,
                )?;
                (t, Some(f))
            } else {
                let t = config.build_optimized_tables(
                    &strip_output.y_blocks,
                    &strip_output.cb_blocks,
                    &strip_output.cr_blocks,
                    is_color,
                )?;
                (t, None)
            };

            config.write_huffman_tables_optimized(output, &tables)?;

            if config.restart_interval > 0 {
                config.write_restart_interval(output)?;
            }
            config.write_scan_header(output)?;

            let scan_data = config.encode_with_tables(
                &strip_output.y_blocks,
                &strip_output.cb_blocks,
                &strip_output.cr_blocks,
                is_color,
                Some(&tables),
            )?;
            Ok((scan_data, frequencies))
        } else if let HuffmanStrategy::Custom(ref tables) = config.huffman {
            config.write_huffman_tables_optimized(output, tables)?;

            if config.restart_interval > 0 {
                config.write_restart_interval(output)?;
            }
            config.write_scan_header(output)?;

            let scan_data = config.encode_with_tables(
                &strip_output.y_blocks,
                &strip_output.cb_blocks,
                &strip_output.cr_blocks,
                is_color,
                Some(tables),
            )?;
            Ok((scan_data, None))
        } else {
            // Fixed: use general-purpose trained tables
            let tables = crate::huffman::builtin_tables::select_tables(
                &config.quality,
                false,
                config.subsampling,
            );
            config.write_huffman_tables_optimized(output, &tables)?;

            if config.restart_interval > 0 {
                config.write_restart_interval(output)?;
            }
            config.write_scan_header(output)?;

            let scan_data = config.encode_with_tables(
                &strip_output.y_blocks,
                &strip_output.cb_blocks,
                &strip_output.cr_blocks,
                is_color,
                Some(&tables),
            )?;
            Ok((scan_data, None))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::encoder_types::Quality;

    #[test]
    fn test_streaming_encoder_creation() {
        let encoder = StreamingEncoder::new(640, 480).start();
        assert!(encoder.is_ok());
        let encoder = encoder.unwrap();
        assert_eq!(encoder.height(), 480);
        assert_eq!(encoder.bytes_per_row(), 640 * 3); // RGB default
    }

    #[test]
    fn test_streaming_encoder_420_strip_height() {
        let encoder = StreamingEncoder::new(640, 480)
            .subsampling(Subsampling::S420)
            .start()
            .unwrap();
        assert_eq!(encoder.strip_height(), 16);
    }

    #[test]
    fn test_streaming_encoder_444_strip_height() {
        let encoder = StreamingEncoder::new(640, 480)
            .subsampling(Subsampling::S444)
            .start()
            .unwrap();
        assert_eq!(encoder.strip_height(), 8);
    }

    #[test]
    fn test_streaming_encoder_wrong_row_size() {
        let mut encoder = StreamingEncoder::new(640, 480).start().unwrap();
        let wrong_row = vec![0u8; 100]; // Wrong size
        let result = encoder.push_row(&wrong_row);
        assert!(result.is_err());
    }

    #[test]
    fn test_streaming_encoder_too_many_rows() {
        let mut encoder = StreamingEncoder::new(4, 2).start().unwrap();
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
        let mut encoder = StreamingEncoder::new(4, 4).start().unwrap();
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

        // 4K with 4:2:0: ~28 MB (blocks + entropy output + working buffers)
        // Heaptrack measured ~28 MB for encoder alone (excluding input pixels)
        assert!(estimate > 25_000_000, "estimate {} too low", estimate);
        assert!(estimate < 40_000_000, "estimate {} too high", estimate);
    }

    #[test]
    fn test_streaming_matches_oneshot() {
        // Create a small test image
        let width = 32u32;
        let height = 32u32;
        let pixels: Vec<u8> = (0..width * height * 3)
            .map(|i| ((i * 17) % 256) as u8)
            .collect();

        // Encode with one-shot method
        let oneshot_result = StreamingEncoder::new(width, height)
            .quality(Quality::ApproxJpegli(85.0))
            .subsampling(Subsampling::S444)
            .encode(&pixels)
            .unwrap();

        // Encode with streaming encoder (row by row)
        let mut streaming = StreamingEncoder::new(width, height)
            .quality(Quality::ApproxJpegli(85.0))
            .subsampling(Subsampling::S444)
            .start()
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
            oneshot_result.len(),
            streaming_result.len(),
            "output lengths differ"
        );
        assert_eq!(oneshot_result, streaming_result, "outputs differ");
    }

    // === Streaming-through tests ===

    fn make_test_image(width: usize, height: usize) -> Vec<u8> {
        let mut data = vec![0u8; width * height * 3];
        for y in 0..height {
            for x in 0..width {
                let i = (y * width + x) * 3;
                data[i] = (x * 255 / width.max(1)) as u8; // R gradient
                data[i + 1] = (y * 255 / height.max(1)) as u8; // G gradient
                data[i + 2] = 128; // B constant
            }
        }
        data
    }

    #[test]
    fn test_custom_tables_enables_streaming() {
        let tables = crate::huffman::optimize::HuffmanTableSet::from_standard().unwrap();
        let encoder = StreamingEncoder::new(64, 64)
            .custom_huffman_tables(tables)
            .start()
            .unwrap();
        assert!(encoder.is_streaming());
    }

    #[test]
    fn test_fixed_tables_enables_streaming() {
        let encoder = StreamingEncoder::new(64, 64)
            .optimize_huffman(false)
            .start()
            .unwrap();
        assert!(encoder.is_streaming());
    }

    #[test]
    fn test_progressive_does_not_stream() {
        let tables = crate::huffman::optimize::HuffmanTableSet::from_standard().unwrap();
        let encoder = StreamingEncoder::new(64, 64)
            .custom_huffman_tables(tables)
            .progressive(true)
            .start()
            .unwrap();
        assert!(!encoder.is_streaming());
    }

    #[test]
    fn test_default_is_buffered() {
        let encoder = StreamingEncoder::new(64, 64).start().unwrap();
        assert!(!encoder.is_streaming());
    }

    #[test]
    fn test_custom_tables_produces_valid_jpeg() {
        let width = 64;
        let height = 64;
        let data = make_test_image(width, height);
        let tables = crate::huffman::optimize::HuffmanTableSet::from_standard().unwrap();

        let jpeg = StreamingEncoder::new(width as u32, height as u32)
            .custom_huffman_tables(tables)
            .encode(&data)
            .unwrap();

        // Valid JPEG starts with FFD8 and ends with FFD9
        assert!(jpeg.len() > 4, "JPEG too small: {} bytes", jpeg.len());
        assert_eq!(jpeg[0], 0xFF);
        assert_eq!(jpeg[1], 0xD8);
        assert_eq!(jpeg[jpeg.len() - 2], 0xFF);
        assert_eq!(jpeg[jpeg.len() - 1], 0xD9);
    }

    #[test]
    fn test_fixed_tables_produces_valid_jpeg() {
        let width = 64;
        let height = 64;
        let data = make_test_image(width, height);

        let jpeg = StreamingEncoder::new(width as u32, height as u32)
            .optimize_huffman(false)
            .encode(&data)
            .unwrap();

        assert!(jpeg.len() > 4, "JPEG too small: {} bytes", jpeg.len());
        assert_eq!(jpeg[0], 0xFF);
        assert_eq!(jpeg[1], 0xD8);
        assert_eq!(jpeg[jpeg.len() - 2], 0xFF);
        assert_eq!(jpeg[jpeg.len() - 1], 0xD9);
    }

    #[test]
    fn test_streaming_vs_buffered_both_valid() {
        let width = 128;
        let height = 128;
        let data = make_test_image(width, height);

        // Streaming path (fixed tables)
        let streaming_jpeg = StreamingEncoder::new(width as u32, height as u32)
            .optimize_huffman(false)
            .encode(&data)
            .unwrap();

        // Buffered path (optimized tables)
        let buffered_jpeg = StreamingEncoder::new(width as u32, height as u32)
            .optimize_huffman(true)
            .encode(&data)
            .unwrap();

        // Both should be valid JPEGs
        for (name, jpeg) in [("streaming", &streaming_jpeg), ("buffered", &buffered_jpeg)] {
            assert!(
                jpeg.len() > 4,
                "{name} JPEG too small: {} bytes",
                jpeg.len()
            );
            assert_eq!(jpeg[0], 0xFF, "{name} missing SOI");
            assert_eq!(jpeg[1], 0xD8, "{name} missing SOI");
            assert_eq!(jpeg[jpeg.len() - 2], 0xFF, "{name} missing EOI");
            assert_eq!(jpeg[jpeg.len() - 1], 0xD9, "{name} missing EOI");
        }

        // Standard tables produce larger output than optimized tables.
        // The difference shouldn't be extreme for natural-ish content.
        let ratio = streaming_jpeg.len() as f64 / buffered_jpeg.len() as f64;
        assert!(
            ratio < 1.7,
            "streaming is too much larger than buffered: {ratio:.2}x ({} vs {} bytes)",
            streaming_jpeg.len(),
            buffered_jpeg.len(),
        );
    }

    #[test]
    fn test_streaming_420_produces_valid_jpeg() {
        let width = 128;
        let height = 128;
        let data = make_test_image(width, height);
        let tables = crate::huffman::optimize::HuffmanTableSet::from_standard().unwrap();

        let jpeg = StreamingEncoder::new(width as u32, height as u32)
            .custom_huffman_tables(tables)
            .subsampling(Subsampling::S420)
            .encode(&data)
            .unwrap();

        assert!(jpeg.len() > 4);
        assert_eq!(jpeg[0], 0xFF);
        assert_eq!(jpeg[1], 0xD8);
        assert_eq!(jpeg[jpeg.len() - 2], 0xFF);
        assert_eq!(jpeg[jpeg.len() - 1], 0xD9);
    }

    #[test]
    fn test_streaming_non_mcu_aligned_dimensions() {
        // Non-8-aligned dimensions to test edge handling
        let width = 67;
        let height = 53;
        let data = make_test_image(width, height);
        let tables = crate::huffman::optimize::HuffmanTableSet::from_standard().unwrap();

        let jpeg = StreamingEncoder::new(width as u32, height as u32)
            .custom_huffman_tables(tables)
            .encode(&data)
            .unwrap();

        assert!(jpeg.len() > 4);
        assert_eq!(jpeg[0], 0xFF);
        assert_eq!(jpeg[1], 0xD8);
        assert_eq!(jpeg[jpeg.len() - 2], 0xFF);
        assert_eq!(jpeg[jpeg.len() - 1], 0xD9);
    }

    #[test]
    fn test_streaming_422_produces_valid_jpeg() {
        let width = 128;
        let height = 128;
        let data = make_test_image(width, height);
        let tables = crate::huffman::optimize::HuffmanTableSet::from_standard().unwrap();

        let jpeg = StreamingEncoder::new(width as u32, height as u32)
            .custom_huffman_tables(tables)
            .subsampling(Subsampling::S422)
            .encode(&data)
            .unwrap();

        assert_valid_jpeg(&jpeg, "422");
    }

    #[test]
    fn test_streaming_440_produces_valid_jpeg() {
        let width = 128;
        let height = 128;
        let data = make_test_image(width, height);
        let tables = crate::huffman::optimize::HuffmanTableSet::from_standard().unwrap();

        let jpeg = StreamingEncoder::new(width as u32, height as u32)
            .custom_huffman_tables(tables)
            .subsampling(Subsampling::S440)
            .encode(&data)
            .unwrap();

        assert_valid_jpeg(&jpeg, "440");
    }

    #[test]
    fn test_streaming_with_restart_markers() {
        let width = 128;
        let height = 128;
        let data = make_test_image(width, height);
        let tables = crate::huffman::optimize::HuffmanTableSet::from_standard().unwrap();

        let jpeg = StreamingEncoder::new(width as u32, height as u32)
            .custom_huffman_tables(tables)
            .restart_interval(10)
            .encode(&data)
            .unwrap();

        assert_valid_jpeg(&jpeg, "restart");

        // Verify restart markers are present in the scan data
        // Restart markers are FFD0-FFD7
        let mut restart_count = 0;
        for i in 0..jpeg.len() - 1 {
            if jpeg[i] == 0xFF && (0xD0..=0xD7).contains(&jpeg[i + 1]) {
                restart_count += 1;
            }
        }
        assert!(
            restart_count > 0,
            "Expected restart markers in output, found none"
        );
    }

    #[test]
    fn test_streaming_420_non_aligned() {
        // Non-16-aligned height with 4:2:0 (strip height = 16)
        let width = 100;
        let height = 75;
        let data = make_test_image(width, height);
        let tables = crate::huffman::optimize::HuffmanTableSet::from_standard().unwrap();

        let jpeg = StreamingEncoder::new(width as u32, height as u32)
            .custom_huffman_tables(tables)
            .subsampling(Subsampling::S420)
            .encode(&data)
            .unwrap();

        assert_valid_jpeg(&jpeg, "420-non-aligned");
    }

    #[test]
    fn test_streaming_larger_image() {
        // 512×512 exercises multiple strip flushes and DC prediction across strips
        let width = 512;
        let height = 512;
        let data = make_test_image(width, height);
        let tables = crate::huffman::optimize::HuffmanTableSet::from_standard().unwrap();

        let jpeg = StreamingEncoder::new(width as u32, height as u32)
            .custom_huffman_tables(tables)
            .encode(&data)
            .unwrap();

        assert_valid_jpeg(&jpeg, "512x512");
        // Sanity: compressed size should be smaller than raw RGB
        assert!(
            jpeg.len() < data.len(),
            "JPEG ({}) should be smaller than raw ({})",
            jpeg.len(),
            data.len()
        );
    }

    #[test]
    fn test_streaming_row_by_row() {
        // Verify row-by-row push works in streaming mode
        let width = 64;
        let height = 64;
        let data = make_test_image(width, height);
        let tables = crate::huffman::optimize::HuffmanTableSet::from_standard().unwrap();

        let mut encoder = StreamingEncoder::new(width as u32, height as u32)
            .custom_huffman_tables(tables)
            .start()
            .unwrap();

        assert!(encoder.is_streaming());

        let row_bytes = width * 3;
        for y in 0..height {
            encoder
                .push_row(&data[y * row_bytes..(y + 1) * row_bytes])
                .unwrap();
        }

        let jpeg = encoder.finish().unwrap();
        assert_valid_jpeg(&jpeg, "row-by-row");
    }

    #[test]
    fn test_streaming_finish_into() {
        // Verify finish_into writes to caller's buffer
        let width = 64;
        let height = 64;
        let data = make_test_image(width, height);

        let mut encoder = StreamingEncoder::new(width as u32, height as u32)
            .optimize_huffman(false)
            .start()
            .unwrap();

        let row_bytes = width * 3;
        for y in 0..height {
            encoder
                .push_row(&data[y * row_bytes..(y + 1) * row_bytes])
                .unwrap();
        }

        let mut output = Vec::new();
        encoder.finish_into(&mut output).unwrap();

        assert_valid_jpeg(&output, "finish_into");
    }

    #[test]
    fn test_streaming_row_by_row_matches_encode() {
        // Row-by-row and encode() convenience should produce identical output
        let width = 64;
        let height = 64;
        let data = make_test_image(width, height);
        let tables = crate::huffman::optimize::HuffmanTableSet::from_standard().unwrap();

        // Path 1: encode() convenience
        let jpeg_oneshot = StreamingEncoder::new(width as u32, height as u32)
            .custom_huffman_tables(tables.clone())
            .encode(&data)
            .unwrap();

        // Path 2: row-by-row
        let mut encoder = StreamingEncoder::new(width as u32, height as u32)
            .custom_huffman_tables(tables)
            .start()
            .unwrap();
        let row_bytes = width * 3;
        for y in 0..height {
            encoder
                .push_row(&data[y * row_bytes..(y + 1) * row_bytes])
                .unwrap();
        }
        let jpeg_manual = encoder.finish().unwrap();

        assert_eq!(
            jpeg_oneshot, jpeg_manual,
            "encode() and row-by-row should produce identical output"
        );
    }

    #[test]
    fn test_streaming_multiple_qualities() {
        // Verify streaming works across a range of quality levels
        let width = 64;
        let height = 64;
        let data = make_test_image(width, height);
        let tables = crate::huffman::optimize::HuffmanTableSet::from_standard().unwrap();

        let mut prev_size = usize::MAX;
        for &q in &[95, 80, 50, 20] {
            let jpeg = StreamingEncoder::new(width as u32, height as u32)
                .custom_huffman_tables(tables.clone())
                .quality(Quality::from(q))
                .encode(&data)
                .unwrap();

            assert_valid_jpeg(&jpeg, &format!("q{q}"));

            // Lower quality should generally produce smaller files
            // (not strictly monotonic for all content, but true for gradients)
            if q < 95 {
                assert!(
                    jpeg.len() < prev_size,
                    "q{q} ({} bytes) should be smaller than previous ({} bytes)",
                    jpeg.len(),
                    prev_size,
                );
            }
            prev_size = jpeg.len();
        }
    }

    /// Decode streaming output and verify dimensions and pixel count.
    #[test]
    #[cfg(feature = "decoder")]
    fn test_streaming_round_trip_decode() {
        let width = 128;
        let height = 128;
        let data = make_test_image(width, height);
        let tables = crate::huffman::optimize::HuffmanTableSet::from_standard().unwrap();

        let jpeg = StreamingEncoder::new(width as u32, height as u32)
            .custom_huffman_tables(tables)
            .encode(&data)
            .unwrap();

        #[allow(deprecated)]
        let decoded = crate::decode::Decoder::new()
            .decode(&jpeg, enough::Unstoppable)
            .unwrap();
        assert_eq!(decoded.width, width as u32);
        assert_eq!(decoded.height, height as u32);
        assert_eq!(decoded.pixels_u8().unwrap().len(), width * height * 3);
    }

    /// Decode 4:2:0 streaming output and verify dimensions.
    #[test]
    #[cfg(feature = "decoder")]
    fn test_streaming_420_round_trip_decode() {
        let width = 128;
        let height = 128;
        let data = make_test_image(width, height);
        let tables = crate::huffman::optimize::HuffmanTableSet::from_standard().unwrap();

        let jpeg = StreamingEncoder::new(width as u32, height as u32)
            .custom_huffman_tables(tables)
            .subsampling(Subsampling::S420)
            .encode(&data)
            .unwrap();

        #[allow(deprecated)]
        let decoded = crate::decode::Decoder::new()
            .decode(&jpeg, enough::Unstoppable)
            .unwrap();
        assert_eq!(decoded.width, width as u32);
        assert_eq!(decoded.height, height as u32);
    }

    /// Decode streaming output with restart markers and verify it decodes correctly.
    #[test]
    #[cfg(feature = "decoder")]
    fn test_streaming_restart_round_trip_decode() {
        let width = 128;
        let height = 128;
        let data = make_test_image(width, height);
        let tables = crate::huffman::optimize::HuffmanTableSet::from_standard().unwrap();

        let jpeg = StreamingEncoder::new(width as u32, height as u32)
            .custom_huffman_tables(tables)
            .restart_interval(5)
            .encode(&data)
            .unwrap();

        #[allow(deprecated)]
        let decoded = crate::decode::Decoder::new()
            .decode(&jpeg, enough::Unstoppable)
            .unwrap();
        assert_eq!(decoded.width, width as u32);
        assert_eq!(decoded.height, height as u32);
        assert_eq!(decoded.pixels_u8().unwrap().len(), width * height * 3);
    }

    /// Decode non-aligned streaming output with 4:2:0.
    #[test]
    #[cfg(feature = "decoder")]
    fn test_streaming_non_aligned_420_round_trip() {
        let width = 100;
        let height = 75;
        let data = make_test_image(width, height);
        let tables = crate::huffman::optimize::HuffmanTableSet::from_standard().unwrap();

        let jpeg = StreamingEncoder::new(width as u32, height as u32)
            .custom_huffman_tables(tables)
            .subsampling(Subsampling::S420)
            .encode(&data)
            .unwrap();

        #[allow(deprecated)]
        let decoded = crate::decode::Decoder::new()
            .decode(&jpeg, enough::Unstoppable)
            .unwrap();
        assert_eq!(decoded.width, width as u32);
        assert_eq!(decoded.height, height as u32);
    }

    fn assert_valid_jpeg(jpeg: &[u8], label: &str) {
        assert!(
            jpeg.len() > 4,
            "{label}: JPEG too small: {} bytes",
            jpeg.len()
        );
        assert_eq!(jpeg[0], 0xFF, "{label}: missing SOI");
        assert_eq!(jpeg[1], 0xD8, "{label}: missing SOI");
        assert_eq!(jpeg[jpeg.len() - 2], 0xFF, "{label}: missing EOI");
        assert_eq!(jpeg[jpeg.len() - 1], 0xD9, "{label}: missing EOI");
    }

    /// Test KLT encoding produces valid JPEG with expected markers.
    #[test]
    fn test_klt_encoding_produces_valid_jpeg() {
        use crate::color::klt::CovarianceAccumulator;

        let width = 64;
        let height = 64;
        let data = make_test_image(width, height);

        // Compute KLT from image statistics
        let mut acc = CovarianceAccumulator::new();
        acc.accumulate_rgb_u8(&data, width, 3);
        let cov = acc.covariance().expect("covariance from test image");
        let mean = acc.mean().expect("mean from test image");
        let klt = crate::color::klt::compute_klt(cov, mean);

        let jpeg = StreamingEncoder::new(width as u32, height as u32)
            .klt_matrix(klt.forward, klt.mean)
            .subsampling(Subsampling::S444)
            .encode(&data)
            .unwrap();

        assert_valid_jpeg(&jpeg, "KLT 444");

        // Verify APP14 Adobe marker is present (transform=0)
        let app14_sig = b"Adobe";
        let app14_pos = jpeg
            .windows(5)
            .position(|w| w == app14_sig)
            .expect("APP14 Adobe marker not found");
        // Transform byte is 11 bytes after the 'A' of "Adobe"
        assert_eq!(
            jpeg[app14_pos + 11],
            0,
            "APP14 transform should be 0 (RGB)"
        );

        // Verify ICC profile is present
        let icc_sig = b"ICC_PROFILE\0";
        assert!(
            jpeg.windows(12).any(|w| w == icc_sig),
            "ICC_PROFILE marker not found"
        );

        // Verify 'R','G','B' component IDs in SOF marker
        // SOF0 = 0xC0, SOF1 = 0xC1
        let sof_pos = jpeg
            .windows(2)
            .position(|w| w[0] == 0xFF && (w[1] == 0xC0 || w[1] == 0xC1))
            .expect("SOF marker not found");
        // SOF structure: marker(2) + length(2) + precision(1) + height(2) + width(2) + ncomp(1) + comp_data
        // First component ID is at offset 10 from the marker start
        assert_eq!(jpeg[sof_pos + 10], b'R', "First component should be 'R'");
        assert_eq!(jpeg[sof_pos + 13], b'G', "Second component should be 'G'");
        assert_eq!(jpeg[sof_pos + 16], b'B', "Third component should be 'B'");
    }

    /// Test KLT roundtrip: encode with KLT, decode, apply inverse matrix, compare.
    #[test]
    #[cfg(feature = "decoder")]
    fn test_klt_roundtrip_with_inverse() {
        use crate::color::klt::CovarianceAccumulator;

        // Use real image (not synthetic gradient which has degenerate covariance)
        let png_path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/images/1.png");
        let png_data = std::fs::read(png_path).expect("read test png");
        let decoder = png::Decoder::new(std::io::Cursor::new(&png_data));
        let mut reader = decoder.read_info().expect("png decode info");
        let mut buf = vec![0u8; reader.output_buffer_size()];
        let info = reader.next_frame(&mut buf).expect("png decode frame");
        let width = info.width as usize;
        let height = info.height as usize;
        let bpp = info.color_type.samples();
        let data = &buf[..width * height * bpp];

        // Compute KLT
        let mut acc = CovarianceAccumulator::new();
        acc.accumulate_rgb_u8(data, width, bpp);
        let cov = acc.covariance().expect("covariance");
        let mean = acc.mean().expect("mean");
        let klt = crate::color::klt::compute_klt(cov, mean);

        let pf = if bpp == 4 { crate::types::PixelFormat::Rgba } else { crate::types::PixelFormat::Rgb };

        // Encode with KLT (high quality to minimize lossy error)
        let jpeg = StreamingEncoder::new(width as u32, height as u32)
            .klt_matrix(klt.forward, klt.mean)
            .subsampling(Subsampling::S444)
            .quality(crate::encode::encoder_types::Quality::ApproxJpegli(98.0))
            .pixel_format(pf)
            .encode(data)
            .unwrap();

        assert_valid_jpeg(&jpeg, "KLT roundtrip");

        // Decode — returns raw decorrelated channels as "RGB"
        #[allow(deprecated)]
        let decoded = crate::decode::Decoder::new()
            .apply_icc(false)
            .decode(&jpeg, enough::Unstoppable)
            .unwrap();
        assert_eq!(decoded.width, width as u32);
        assert_eq!(decoded.height, height as u32);
        let decoded_pixels = decoded.pixels_u8().expect("decoded pixels");

        // Apply inverse KLT (with unscaling) to recover original RGB
        let inverse = klt.inverse;
        let encode_params = crate::color::klt::KltEncodeParams::from_forward_with_center(klt.forward, klt.mean);
        let (inv_scale, inv_offset) = encode_params.inverse_scale_offset();

        let mut max_error = 0u8;
        let mut total_error = 0u64;
        for i in 0..(width * height) {
            // Undo the per-channel scale+offset to get raw KLT values
            let c0 = decoded_pixels[i * 3] as f32 * inv_scale[0] + inv_offset[0];
            let c1 = decoded_pixels[i * 3 + 1] as f32 * inv_scale[1] + inv_offset[1];
            let c2 = decoded_pixels[i * 3 + 2] as f32 * inv_scale[2] + inv_offset[2];

            let [r, g, b] = inverse.transform([c0, c1, c2]);

            let orig_r = data[i * bpp];
            let orig_g = data[i * bpp + 1];
            let orig_b = data[i * bpp + 2];

            let err_r = (r.round().clamp(0.0, 255.0) as u8).abs_diff(orig_r);
            let err_g = (g.round().clamp(0.0, 255.0) as u8).abs_diff(orig_g);
            let err_b = (b.round().clamp(0.0, 255.0) as u8).abs_diff(orig_b);

            max_error = max_error.max(err_r).max(err_g).max(err_b);
            total_error += err_r as u64 + err_g as u64 + err_b as u64;
        }

        let avg_error = total_error as f64 / (width * height * 3) as f64;
        eprintln!("KLT roundtrip: max_error={}, avg_error={:.2}", max_error, avg_error);
        // At q98 with 4:4:4, roundtrip error should be very low
        assert!(
            max_error < 50,
            "KLT roundtrip max error too high: {max_error}"
        );
        assert!(
            avg_error < 2.0,
            "KLT roundtrip avg error too high: {avg_error:.2}"
        );
    }

    /// Write KLT-encoded JPEG to disk for visual inspection with ICC-aware tools.
    #[test]
    fn test_klt_write_to_disk() {
        use crate::color::klt::CovarianceAccumulator;

        // Load test image
        let png_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/images/1.png"
        );
        let png_data = std::fs::read(png_path).expect("read test png");
        let decoder = png::Decoder::new(std::io::Cursor::new(&png_data));
        let mut reader = decoder.read_info().expect("png decode info");
        let mut buf = vec![0u8; reader.output_buffer_size()];
        let info = reader.next_frame(&mut buf).expect("png decode frame");
        let width = info.width as usize;
        let height = info.height as usize;
        let bpp = info.color_type.samples();
        let rgb_data = &buf[..width * height * bpp];

        // Compute KLT
        let mut acc = CovarianceAccumulator::new();
        acc.accumulate_rgb_u8(rgb_data, width, bpp);
        let cov = acc.covariance().expect("covariance");
        let mean = acc.mean().expect("mean");
        let klt = crate::color::klt::compute_klt(cov, mean);

        // Encode with KLT at various qualities
        for (quality, subsampling, label) in [
            (85.0, Subsampling::S444, "q85_444"),
            (85.0, Subsampling::S420, "q85_420"),
            (95.0, Subsampling::S444, "q95_444"),
        ] {
            let jpeg = StreamingEncoder::new(width as u32, height as u32)
                .klt_matrix(klt.forward, klt.mean)
                .subsampling(subsampling)
                .quality(crate::encode::encoder_types::Quality::ApproxJpegli(quality))
                .pixel_format(if bpp == 4 {
                    crate::types::PixelFormat::Rgba
                } else {
                    crate::types::PixelFormat::Rgb
                })
                .encode(rgb_data)
                .unwrap();

            let out_path = format!("/mnt/v/output/zenjpeg/klt-test/klt_{label}.jpg");
            std::fs::write(&out_path, &jpeg).expect("write KLT jpeg");

            // Also encode standard YCbCr for comparison
            let ycbcr_jpeg = StreamingEncoder::new(width as u32, height as u32)
                .subsampling(subsampling)
                .quality(crate::encode::encoder_types::Quality::ApproxJpegli(quality))
                .pixel_format(if bpp == 4 {
                    crate::types::PixelFormat::Rgba
                } else {
                    crate::types::PixelFormat::Rgb
                })
                .encode(rgb_data)
                .unwrap();

            let ycbcr_path = format!("/mnt/v/output/zenjpeg/klt-test/ycbcr_{label}.jpg");
            std::fs::write(&ycbcr_path, &ycbcr_jpeg).expect("write YCbCr jpeg");

            eprintln!("{label}: KLT={} bytes, YCbCr={} bytes, ratio={:.3}",
                jpeg.len(), ycbcr_jpeg.len(),
                jpeg.len() as f64 / ycbcr_jpeg.len() as f64);
        }

        eprintln!("KLT energy concentration: {:.4}", klt.energy_concentration);
        eprintln!("KLT eigenvalues: {:?}", klt.eigenvalues);
        eprintln!("KLT beneficial: {}", crate::color::klt::klt_is_beneficial(&klt));
    }

    /// Test KLT encoding with 4:2:0 subsampling.
    #[test]
    fn test_klt_encoding_420() {
        use crate::color::klt::CovarianceAccumulator;

        let width = 64;
        let height = 64;
        let data = make_test_image(width, height);

        let mut acc = CovarianceAccumulator::new();
        acc.accumulate_rgb_u8(&data, width, 3);
        let cov = acc.covariance().expect("covariance from test image");
        let mean = acc.mean().expect("mean from test image");
        let klt = crate::color::klt::compute_klt(cov, mean);

        let jpeg = StreamingEncoder::new(width as u32, height as u32)
            .klt_matrix(klt.forward, klt.mean)
            .subsampling(Subsampling::S420)
            .encode(&data)
            .unwrap();

        assert_valid_jpeg(&jpeg, "KLT 420");

        // Find SOF and check first component has 2x2 sampling
        let sof_pos = jpeg
            .windows(2)
            .position(|w| w[0] == 0xFF && (w[1] == 0xC0 || w[1] == 0xC1))
            .expect("SOF marker not found");
        // Component 0 ('R') sampling factor should be 0x22 (2x2) for 4:2:0
        assert_eq!(jpeg[sof_pos + 11], 0x22, "R component should have 2x2 sampling for 4:2:0");
        // Components 1,2 ('G','B') should have 0x11 (1x1)
        assert_eq!(jpeg[sof_pos + 14], 0x11, "G component should have 1x1 sampling");
        assert_eq!(jpeg[sof_pos + 17], 0x11, "B component should have 1x1 sampling");
    }

    /// Compare DCT coefficient distributions between KLT and YCbCr encoding.
    ///
    /// This is the core evaluation tool for the CCTX feature. It measures:
    /// - Zero coefficient counts (more zeros = better entropy coding)
    /// - Sum of absolute coefficients (lower = fewer bits needed)
    /// - Per-component energy distribution
    /// - File size comparison at matched quality
    #[test]
    fn test_klt_coefficient_evaluation() {
        use crate::color::klt::CovarianceAccumulator;
        use crate::decode::Decoder;

        // Load test image
        let png_path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/images/1.png");
        let png_data = std::fs::read(png_path).expect("read test png");
        let decoder = png::Decoder::new(std::io::Cursor::new(&png_data));
        let mut reader = decoder.read_info().expect("png decode info");
        let mut buf = vec![0u8; reader.output_buffer_size()];
        let info = reader.next_frame(&mut buf).expect("png decode frame");
        let width = info.width as usize;
        let height = info.height as usize;
        let bpp = info.color_type.samples();
        let rgb_data = &buf[..width * height * bpp];
        let pixels = width * height;

        // Compute KLT
        let mut acc = CovarianceAccumulator::new();
        acc.accumulate_rgb_u8(rgb_data, width, bpp);
        let cov = acc.covariance().expect("covariance");
        let mean = acc.mean().expect("mean");
        let klt = crate::color::klt::compute_klt(cov, mean);

        let pf = if bpp == 4 {
            crate::types::PixelFormat::Rgba
        } else {
            crate::types::PixelFormat::Rgb
        };

        let jpeg_decoder = Decoder::new();

        eprintln!("\n=== KLT vs YCbCr DCT Coefficient Evaluation ===");
        eprintln!("Image: {}x{} ({} pixels)", width, height, pixels);
        eprintln!("KLT energy concentration: {:.4}", klt.energy_concentration);
        eprintln!("KLT eigenvalues: [{:.1}, {:.1}, {:.1}]",
            klt.eigenvalues[0], klt.eigenvalues[1], klt.eigenvalues[2]);
        eprintln!("KLT beneficial: {}", crate::color::klt::klt_is_beneficial(&klt));
        eprintln!();

        for (quality, subsampling, label) in [
            (75.0, Subsampling::S444, "q75_444"),
            (85.0, Subsampling::S444, "q85_444"),
            (90.0, Subsampling::S444, "q90_444"),
            (95.0, Subsampling::S444, "q95_444"),
            (85.0, Subsampling::S420, "q85_420"),
        ] {
            let q = crate::encode::encoder_types::Quality::ApproxJpegli(quality);

            // Encode KLT
            let klt_jpeg = StreamingEncoder::new(width as u32, height as u32)
                .klt_matrix(klt.forward, klt.mean)
                .subsampling(subsampling)
                .quality(q)
                .pixel_format(pf)
                .encode(rgb_data)
                .unwrap();

            // Encode YCbCr
            let ycbcr_jpeg = StreamingEncoder::new(width as u32, height as u32)
                .subsampling(subsampling)
                .quality(q)
                .pixel_format(pf)
                .encode(rgb_data)
                .unwrap();

            // Decode coefficients
            let klt_coeffs = jpeg_decoder
                .decode_coefficients(&klt_jpeg, enough::Unstoppable)
                .expect("decode KLT coefficients");
            let ycbcr_coeffs = jpeg_decoder
                .decode_coefficients(&ycbcr_jpeg, enough::Unstoppable)
                .expect("decode YCbCr coefficients");

            eprintln!("--- {} ---", label);
            eprintln!("File size: KLT={} bytes, YCbCr={} bytes, delta={:+.2}%",
                klt_jpeg.len(), ycbcr_jpeg.len(),
                (klt_jpeg.len() as f64 / ycbcr_jpeg.len() as f64 - 1.0) * 100.0);
            eprintln!("BPP: KLT={:.3}, YCbCr={:.3}",
                klt_jpeg.len() as f64 * 8.0 / pixels as f64,
                ycbcr_jpeg.len() as f64 * 8.0 / pixels as f64);

            // Per-component coefficient analysis
            let comp_names_klt = ["C0(principal)", "C1", "C2"];
            let comp_names_ycbcr = ["Y", "Cb", "Cr"];

            for (label_set, coeffs, names) in [
                ("KLT", &klt_coeffs, &comp_names_klt[..]),
                ("YCbCr", &ycbcr_coeffs, &comp_names_ycbcr[..]),
            ] {
                eprintln!("  {} components:", label_set);
                let mut total_zeros = 0u64;
                let mut total_coeffs = 0u64;
                let mut total_abs_sum = 0u64;

                for (i, comp) in coeffs.components.iter().enumerate() {
                    let num_blocks = comp.num_blocks();
                    let n_coeffs = num_blocks * 64;
                    let mut zeros = 0u64;
                    let mut abs_sum = 0u64;
                    let mut dc_abs_sum = 0u64;
                    let mut ac_abs_sum = 0u64;

                    for block_idx in 0..num_blocks {
                        let block = comp.block(block_idx);
                        dc_abs_sum += block[0].unsigned_abs() as u64;
                        for &coeff in &block[1..] {
                            ac_abs_sum += coeff.unsigned_abs() as u64;
                            if coeff == 0 {
                                zeros += 1;
                            }
                        }
                        if block[0] == 0 {
                            zeros += 1;
                        }
                        abs_sum += block.iter().map(|c| c.unsigned_abs() as u64).sum::<u64>();
                    }

                    total_zeros += zeros;
                    total_coeffs += n_coeffs as u64;
                    total_abs_sum += abs_sum;

                    let name = names.get(i).unwrap_or(&"?");
                    eprintln!(
                        "    {}: {} blocks, zeros={}/{} ({:.1}%), |DC|_sum={}, |AC|_sum={}, |all|_sum={}",
                        name, num_blocks, zeros, n_coeffs,
                        100.0 * zeros as f64 / n_coeffs as f64,
                        dc_abs_sum, ac_abs_sum, abs_sum
                    );
                }
                eprintln!(
                    "    TOTAL: zeros={}/{} ({:.1}%), |coeff|_sum={}",
                    total_zeros, total_coeffs,
                    100.0 * total_zeros as f64 / total_coeffs as f64,
                    total_abs_sum
                );
            }

            // Quant table comparison
            eprintln!("  Quant tables:");
            for (label_q, coeffs) in [("KLT", &klt_coeffs), ("YCbCr", &ycbcr_coeffs)] {
                for (i, qt) in coeffs.quant_tables.iter().enumerate() {
                    if let Some(table) = qt {
                        eprintln!("    {} table {}: DC={}, AC[1]={}, AC[63]={}",
                            label_q, i, table[0], table[1], table[63]);
                    }
                }
            }
            eprintln!();
        }

        // Write output files for visual inspection
        let out_dir = "/mnt/v/output/zenjpeg/klt-eval/";
        std::fs::create_dir_all(out_dir).ok();
    }

    /// Corpus-level KLT evaluation across 41 CID22-512 images.
    ///
    /// Measures per-image KLT benefit: file size delta, zero count delta,
    /// coefficient magnitude delta, and energy concentration.
    /// Outputs CSV to /mnt/v/output/zenjpeg/klt-eval/corpus_results.csv
    #[test]
    #[ignore] // Requires CID22 corpus at ~/work/codec-corpus
    fn test_klt_corpus_evaluation() {
        use crate::color::klt::{self, CovarianceAccumulator};
        use crate::decode::Decoder;

        let corpus_dir = "/home/lilith/work/codec-corpus/CID22/CID22-512/validation";
        let out_dir = "/mnt/v/output/zenjpeg/klt-eval";
        std::fs::create_dir_all(out_dir).ok();

        let mut images: Vec<_> = std::fs::read_dir(corpus_dir)
            .expect("open corpus dir")
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "png"))
            .collect();
        images.sort_by_key(|e| e.file_name());

        assert!(!images.is_empty(), "no images found in {}", corpus_dir);

        let jpeg_decoder = Decoder::new();

        // CSV header
        let mut csv = String::from(
            "image,width,height,energy_conc,eigenval_ratio,beneficial,\
             klt_bytes_q85,ycbcr_bytes_q85,delta_pct_q85,\
             klt_zeros_q85,ycbcr_zeros_q85,\
             klt_abssum_q85,ycbcr_abssum_q85,\
             klt_bytes_q95,ycbcr_bytes_q95,delta_pct_q95\n"
        );

        let mut total_klt_85 = 0u64;
        let mut total_ycbcr_85 = 0u64;
        let mut total_klt_95 = 0u64;
        let mut total_ycbcr_95 = 0u64;
        let mut beneficial_count = 0;
        let mut klt_wins_85 = 0;
        let mut klt_wins_95 = 0;

        eprintln!("\n=== KLT Corpus Evaluation ({} images) ===\n", images.len());
        eprintln!("{:<15} {:>6} {:>6} {:>8} {:>8} {:>7} {:>8} {:>8} {:>7}",
            "Image", "EnConc", "EigR", "KLT_85", "YCbCr85", "Δ%_85", "KLT_95", "YCbCr95", "Δ%_95");
        eprintln!("{}", "-".repeat(95));

        for entry in &images {
            let path = entry.path();
            let name = path.file_stem().unwrap().to_string_lossy().to_string();

            // Load PNG
            let png_data = std::fs::read(&path).expect("read png");
            let decoder_png = png::Decoder::new(std::io::Cursor::new(&png_data));
            let mut reader = decoder_png.read_info().expect("png info");
            let mut buf = vec![0u8; reader.output_buffer_size()];
            let info = reader.next_frame(&mut buf).expect("png frame");
            let width = info.width as usize;
            let height = info.height as usize;
            let bpp = info.color_type.samples();
            let rgb_data = &buf[..width * height * bpp];

            let pf = if bpp == 4 {
                crate::types::PixelFormat::Rgba
            } else {
                crate::types::PixelFormat::Rgb
            };

            // Compute KLT
            let mut acc = CovarianceAccumulator::new();
            acc.accumulate_rgb_u8(rgb_data, width, bpp);
            let cov = acc.covariance().expect("covariance");
            let mean = acc.mean().expect("mean");
            let klt_result = klt::compute_klt(cov, mean);
            let beneficial = klt::klt_is_beneficial(&klt_result);
            if beneficial { beneficial_count += 1; }

            let eigenval_ratio = klt_result.eigenvalues[0] / klt_result.eigenvalues[2].max(0.001);

            let mut row_data = Vec::new();

            for (quality, label) in [(85.0, "q85"), (95.0, "q95")] {
                let q = crate::encode::encoder_types::Quality::ApproxJpegli(quality);

                let klt_jpeg = StreamingEncoder::new(width as u32, height as u32)
                    .klt_matrix(klt_result.forward, klt_result.mean)
                    .subsampling(Subsampling::S444)
                    .quality(q)
                    .pixel_format(pf)
                    .encode(rgb_data)
                    .unwrap();

                let ycbcr_jpeg = StreamingEncoder::new(width as u32, height as u32)
                    .subsampling(Subsampling::S444)
                    .quality(q)
                    .pixel_format(pf)
                    .encode(rgb_data)
                    .unwrap();

                let delta = (klt_jpeg.len() as f64 / ycbcr_jpeg.len() as f64 - 1.0) * 100.0;

                // Coefficient analysis for q85 only
                let (klt_zeros, klt_abssum, ycbcr_zeros, ycbcr_abssum) = if label == "q85" {
                    let klt_coeffs = jpeg_decoder
                        .decode_coefficients(&klt_jpeg, enough::Unstoppable)
                        .expect("decode KLT coefficients");
                    let ycbcr_coeffs = jpeg_decoder
                        .decode_coefficients(&ycbcr_jpeg, enough::Unstoppable)
                        .expect("decode YCbCr coefficients");

                    let (kz, ka) = coeff_stats(&klt_coeffs.components);
                    let (yz, ya) = coeff_stats(&ycbcr_coeffs.components);
                    (kz, ka, yz, ya)
                } else {
                    (0, 0, 0, 0)
                };

                row_data.push((klt_jpeg.len(), ycbcr_jpeg.len(), delta, klt_zeros, klt_abssum, ycbcr_zeros, ycbcr_abssum));

                match label {
                    "q85" => {
                        total_klt_85 += klt_jpeg.len() as u64;
                        total_ycbcr_85 += ycbcr_jpeg.len() as u64;
                        if klt_jpeg.len() < ycbcr_jpeg.len() { klt_wins_85 += 1; }
                    }
                    "q95" => {
                        total_klt_95 += klt_jpeg.len() as u64;
                        total_ycbcr_95 += ycbcr_jpeg.len() as u64;
                        if klt_jpeg.len() < ycbcr_jpeg.len() { klt_wins_95 += 1; }
                    }
                    _ => {}
                }
            }

            let d85 = &row_data[0];
            let d95 = &row_data[1];

            eprintln!("{:<15} {:>5.3} {:>6.0} {:>8} {:>8} {:>+6.2}% {:>8} {:>8} {:>+6.2}%",
                &name[..name.len().min(15)],
                klt_result.energy_concentration, eigenval_ratio,
                d85.0, d85.1, d85.2,
                d95.0, d95.1, d95.2);

            csv.push_str(&format!(
                "{},{},{},{:.4},{:.1},{},{},{},{:.3},{},{},{},{},{},{},{:.3}\n",
                name, width, height,
                klt_result.energy_concentration, eigenval_ratio, beneficial,
                d85.0, d85.1, d85.2,
                d85.3, d85.5, // klt_zeros, ycbcr_zeros
                d85.4, d85.6, // klt_abssum, ycbcr_abssum
                d95.0, d95.1, d95.2,
            ));
        }

        eprintln!("{}", "-".repeat(95));
        eprintln!("TOTALS: q85: KLT={} YCbCr={} Δ={:+.2}%  q95: KLT={} YCbCr={} Δ={:+.2}%",
            total_klt_85, total_ycbcr_85,
            (total_klt_85 as f64 / total_ycbcr_85 as f64 - 1.0) * 100.0,
            total_klt_95, total_ycbcr_95,
            (total_klt_95 as f64 / total_ycbcr_95 as f64 - 1.0) * 100.0);
        eprintln!("KLT beneficial: {}/{}", beneficial_count, images.len());
        eprintln!("KLT wins: q85={}/{}, q95={}/{}", klt_wins_85, images.len(), klt_wins_95, images.len());

        let csv_path = format!("{}/corpus_results.csv", out_dir);
        std::fs::write(&csv_path, &csv).expect("write CSV");
        eprintln!("\nCSV written to {}", csv_path);
    }

    fn coeff_stats(components: &[crate::decode::ComponentCoefficients]) -> (u64, u64) {
        let mut zeros = 0u64;
        let mut abs_sum = 0u64;
        for comp in components {
            for block_idx in 0..comp.num_blocks() {
                let block = comp.block(block_idx);
                for &c in block {
                    if c == 0 { zeros += 1; }
                    abs_sum += c.unsigned_abs() as u64;
                }
            }
        }
        (zeros, abs_sum)
    }

    /// Debug: detailed coefficient analysis for problem vs winning images.
    #[test]
    #[ignore]
    fn debug_klt_problem_images() {
        use crate::color::klt::{self, CovarianceAccumulator};
        use crate::decode::Decoder;

        let corpus_dir = "/home/lilith/work/codec-corpus/CID22/CID22-512/validation";
        // Top losers and winners from corpus eval
        let images = [
            ("1475938.png", "+64.6%"),
            ("792079.png", "+23.2%"),
            ("1418519.png", "+15.9%"),
            ("1044329.png", "-8.5%"),
            ("2775196.png", "-7.0%"),
            ("2936831.png", "-5.8%"),
        ];

        let jpeg_decoder = Decoder::new();

        for (img_name, expected_delta) in images {
            let path = format!("{}/{}", corpus_dir, img_name);
            let png_data = std::fs::read(&path).expect("read png");
            let dec = png::Decoder::new(std::io::Cursor::new(&png_data));
            let mut reader = dec.read_info().expect("png info");
            let mut buf = vec![0u8; reader.output_buffer_size()];
            let info = reader.next_frame(&mut buf).expect("png frame");
            let w = info.width as usize;
            let h = info.height as usize;
            let bpp = info.color_type.samples();
            let rgb = &buf[..w * h * bpp];
            let pf = if bpp == 4 { crate::types::PixelFormat::Rgba } else { crate::types::PixelFormat::Rgb };

            let mut acc = CovarianceAccumulator::new();
            acc.accumulate_rgb_u8(rgb, w, bpp);
            let cov = acc.covariance().expect("cov");
            let mean = acc.mean().expect("mean");
            let klt_result = klt::compute_klt(cov, mean);

            let q = crate::encode::encoder_types::Quality::ApproxJpegli(85.0);

            let klt_jpeg = StreamingEncoder::new(w as u32, h as u32)
                .klt_matrix(klt_result.forward, klt_result.mean)
                .subsampling(Subsampling::S444)
                .quality(q)
                .pixel_format(pf)
                .encode(rgb)
                .unwrap();

            let ycbcr_jpeg = StreamingEncoder::new(w as u32, h as u32)
                .subsampling(Subsampling::S444)
                .quality(q)
                .pixel_format(pf)
                .encode(rgb)
                .unwrap();

            let klt_coeffs = jpeg_decoder.decode_coefficients(&klt_jpeg, enough::Unstoppable).unwrap();
            let ycbcr_coeffs = jpeg_decoder.decode_coefficients(&ycbcr_jpeg, enough::Unstoppable).unwrap();

            eprintln!("\n=== {} (expected {}) ===", img_name, expected_delta);
            eprintln!("Sizes: KLT={}, YCbCr={}", klt_jpeg.len(), ycbcr_jpeg.len());
            eprintln!("Energy conc: {:.4}, eigenvals: [{:.1}, {:.1}, {:.1}]",
                klt_result.energy_concentration,
                klt_result.eigenvalues[0], klt_result.eigenvalues[1], klt_result.eigenvalues[2]);

            for (label, coeffs) in [("KLT", &klt_coeffs), ("YCbCr", &ycbcr_coeffs)] {
                eprintln!("  {} coefficients:", label);
                for (i, comp) in coeffs.components.iter().enumerate() {
                    let nb = comp.num_blocks();
                    let mut zeros = 0u64;
                    let mut dc_abs_sum = 0u64;
                    let mut ac_abs_sum = 0u64;
                    let mut dc_min = i16::MAX;
                    let mut dc_max = i16::MIN;
                    let mut ac_nonzero = 0u64;

                    for bi in 0..nb {
                        let block = comp.block(bi);
                        let dc = block[0];
                        dc_abs_sum += dc.unsigned_abs() as u64;
                        dc_min = dc_min.min(dc);
                        dc_max = dc_max.max(dc);
                        if dc == 0 { zeros += 1; }
                        for &c in &block[1..] {
                            if c == 0 { zeros += 1; }
                            else { ac_nonzero += 1; }
                            ac_abs_sum += c.unsigned_abs() as u64;
                        }
                    }

                    let name = match (label, i) {
                        ("KLT", 0) => "C0",
                        ("KLT", 1) => "C1",
                        ("KLT", 2) => "C2",
                        ("YCbCr", 0) => "Y ",
                        ("YCbCr", 1) => "Cb",
                        ("YCbCr", 2) => "Cr",
                        _ => "??",
                    };
                    eprintln!(
                        "    {}: {} blocks, DC range=[{},{}], |DC|={}, |AC|={}, AC_nonzero={}, zeros={}/{}",
                        name, nb, dc_min, dc_max, dc_abs_sum, ac_abs_sum, ac_nonzero,
                        zeros, nb * 64
                    );
                }
            }
        }
    }

    /// KLT vs YCbCr quality comparison using SSIMULACRA2 and Butteraugli.
    ///
    /// At the same quality setting, compares file size AND perceptual quality
    /// to verify KLT isn't trading quality for size.
    ///
    /// Run: cargo test test_klt_quality_metrics --lib --features cms -- --ignored --nocapture
    #[test]
    #[ignore]
    fn test_klt_quality_metrics() {
        use crate::color::klt::{self, CovarianceAccumulator};
        use butteraugli::{compute_butteraugli, ButteraugliParams};
        use fast_ssim2::{compute_frame_ssimulacra2, ColorPrimaries, Rgb, TransferCharacteristic};

        let corpus_dir = "/home/lilith/work/codec-corpus/CID22/CID22-512/validation";
        let entries: Vec<_> = std::fs::read_dir(corpus_dir)
            .expect("read corpus dir")
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|x| x == "png"))
            .collect();
        assert!(!entries.is_empty(), "no PNG files in corpus");

        fn to_ssim2_frame(
            pixels: &[u8],
            w: usize,
            h: usize,
        ) -> Rgb {
            let rgb: Vec<[f32; 3]> = pixels
                .chunks(3)
                .map(|c| [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0])
                .collect();
            Rgb::new(rgb, w, h, TransferCharacteristic::SRGB, ColorPrimaries::BT709).unwrap()
        }

        fn decode_klt_to_srgb(
            jpeg: &[u8],
            klt_result: &klt::KltAnalysis,
            w: usize,
            h: usize,
        ) -> Vec<u8> {
            // Decode without ICC — get raw KLT channel values
            #[allow(deprecated)]
            let decoded = crate::decode::Decoder::new()
                .apply_icc(false)
                .decode(jpeg, enough::Unstoppable)
                .unwrap();
            let pixels = decoded.pixels_u8().expect("decoded pixels");

            // Apply inverse KLT to recover sRGB
            let encode_params =
                klt::KltEncodeParams::from_forward_with_center(klt_result.forward, klt_result.mean);
            let (inv_scale, inv_offset) = encode_params.inverse_scale_offset();
            let inverse = klt_result.inverse;

            let mut out = vec![0u8; w * h * 3];
            for i in 0..(w * h) {
                let c0 = pixels[i * 3] as f32 * inv_scale[0] + inv_offset[0];
                let c1 = pixels[i * 3 + 1] as f32 * inv_scale[1] + inv_offset[1];
                let c2 = pixels[i * 3 + 2] as f32 * inv_scale[2] + inv_offset[2];
                let [r, g, b] = inverse.transform([c0, c1, c2]);
                out[i * 3] = r.round().clamp(0.0, 255.0) as u8;
                out[i * 3 + 1] = g.round().clamp(0.0, 255.0) as u8;
                out[i * 3 + 2] = b.round().clamp(0.0, 255.0) as u8;
            }
            out
        }

        let qualities = [75.0, 85.0, 95.0];
        let ba_params = ButteraugliParams::default();

        eprintln!("\n=== KLT vs YCbCr Quality Metrics (CID22-512 corpus) ===\n");
        eprintln!(
            "{:<12} {:>5} {:>8} {:>8} {:>7} {:>8} {:>8} {:>8} {:>8}",
            "Image", "Q", "KLT_sz", "YCbCr_sz", "Δ%", "KLT_ss2", "YCb_ss2", "KLT_ba", "YCb_ba"
        );
        eprintln!("{}", "-".repeat(95));

        // Accumulators for summary
        let mut total_klt_bytes = vec![0u64; qualities.len()];
        let mut total_ycbcr_bytes = vec![0u64; qualities.len()];
        let mut sum_klt_ssim2 = vec![0.0f64; qualities.len()];
        let mut sum_ycbcr_ssim2 = vec![0.0f64; qualities.len()];
        let mut sum_klt_ba = vec![0.0f64; qualities.len()];
        let mut sum_ycbcr_ba = vec![0.0f64; qualities.len()];
        let mut count = 0usize;
        let mut klt_ssim2_better = vec![0usize; qualities.len()];
        let mut klt_ba_better = vec![0usize; qualities.len()];

        let mut sorted_entries: Vec<_> = entries.iter().collect();
        sorted_entries.sort_by_key(|e| e.file_name());

        for entry in &sorted_entries {
            let png_data = std::fs::read(entry.path()).expect("read png");
            let dec = png::Decoder::new(std::io::Cursor::new(&png_data));
            let mut reader = dec.read_info().expect("png info");
            let mut buf = vec![0u8; reader.output_buffer_size()];
            let info = reader.next_frame(&mut buf).expect("png frame");
            let w = info.width as usize;
            let h = info.height as usize;
            let bpp = info.color_type.samples();
            let rgb = &buf[..w * h * bpp];
            let pf = if bpp == 4 {
                crate::types::PixelFormat::Rgba
            } else {
                crate::types::PixelFormat::Rgb
            };

            // Compute KLT
            let mut acc = CovarianceAccumulator::new();
            acc.accumulate_rgb_u8(rgb, w, bpp);
            let cov = acc.covariance().expect("cov");
            let mean = acc.mean().expect("mean");
            let klt_result = klt::compute_klt(cov, mean);

            let img_name = entry.path().file_stem().unwrap().to_string_lossy().to_string();

            // Need RGB-only for metrics
            let orig_rgb3: Vec<u8> = if bpp == 3 {
                rgb.to_vec()
            } else {
                rgb.chunks(bpp).flat_map(|c| &c[..3]).copied().collect()
            };

            for (qi, &quality) in qualities.iter().enumerate() {
                let q = crate::encode::encoder_types::Quality::ApproxJpegli(quality);

                // Encode KLT
                let klt_jpeg = StreamingEncoder::new(w as u32, h as u32)
                    .klt_matrix(klt_result.forward, klt_result.mean)
                    .subsampling(Subsampling::S444)
                    .quality(q)
                    .pixel_format(pf)
                    .encode(rgb)
                    .unwrap();

                // Encode YCbCr
                let ycbcr_jpeg = StreamingEncoder::new(w as u32, h as u32)
                    .subsampling(Subsampling::S444)
                    .quality(q)
                    .pixel_format(pf)
                    .encode(rgb)
                    .unwrap();

                // Decode KLT to sRGB via manual inverse
                let klt_decoded = decode_klt_to_srgb(&klt_jpeg, &klt_result, w, h);

                // Decode YCbCr (straightforward)
                #[allow(deprecated)]
                let ycbcr_dec = crate::decode::Decoder::new()
                    .decode(&ycbcr_jpeg, enough::Unstoppable)
                    .unwrap();
                let ycbcr_decoded = ycbcr_dec.pixels_u8().expect("ycbcr pixels");

                // SSIMULACRA2
                let orig_frame = to_ssim2_frame(&orig_rgb3, w, h);
                let klt_frame = to_ssim2_frame(&klt_decoded, w, h);
                let ycbcr_frame = to_ssim2_frame(&ycbcr_decoded, w, h);

                let klt_ssim2 =
                    compute_frame_ssimulacra2(orig_frame.clone(), klt_frame).unwrap_or(-999.0);
                let ycbcr_ssim2 =
                    compute_frame_ssimulacra2(orig_frame, ycbcr_frame).unwrap_or(-999.0);

                // Butteraugli
                let klt_ba = compute_butteraugli(&orig_rgb3, &klt_decoded, w, h, &ba_params)
                    .map(|r| r.score)
                    .unwrap_or(f64::NAN);
                let ycbcr_ba = compute_butteraugli(&orig_rgb3, &ycbcr_decoded, w, h, &ba_params)
                    .map(|r| r.score)
                    .unwrap_or(f64::NAN);

                let size_delta =
                    (klt_jpeg.len() as f64 - ycbcr_jpeg.len() as f64) / ycbcr_jpeg.len() as f64
                        * 100.0;

                eprintln!(
                    "{:<12} {:>5.0} {:>8} {:>8} {:>+6.1}% {:>8.2} {:>8.2} {:>8.3} {:>8.3}",
                    img_name,
                    quality,
                    klt_jpeg.len(),
                    ycbcr_jpeg.len(),
                    size_delta,
                    klt_ssim2,
                    ycbcr_ssim2,
                    klt_ba,
                    ycbcr_ba
                );

                total_klt_bytes[qi] += klt_jpeg.len() as u64;
                total_ycbcr_bytes[qi] += ycbcr_jpeg.len() as u64;
                sum_klt_ssim2[qi] += klt_ssim2;
                sum_ycbcr_ssim2[qi] += ycbcr_ssim2;
                sum_klt_ba[qi] += klt_ba;
                sum_ycbcr_ba[qi] += ycbcr_ba;
                if klt_ssim2 > ycbcr_ssim2 {
                    klt_ssim2_better[qi] += 1;
                }
                if klt_ba < ycbcr_ba {
                    klt_ba_better[qi] += 1;
                }
            }
            count += 1;
        }

        // Summary
        eprintln!("\n=== Summary ({} images) ===", count);
        for (qi, &quality) in qualities.iter().enumerate() {
            let n = count as f64;
            let size_delta = (total_klt_bytes[qi] as f64 - total_ycbcr_bytes[qi] as f64)
                / total_ycbcr_bytes[qi] as f64
                * 100.0;
            eprintln!(
                "Q{:.0}: size Δ={:+.2}%, avg SSIM2: KLT={:.3} YCbCr={:.3} (KLT better {}/{}), avg BA: KLT={:.4} YCbCr={:.4} (KLT better {}/{})",
                quality, size_delta,
                sum_klt_ssim2[qi] / n, sum_ycbcr_ssim2[qi] / n,
                klt_ssim2_better[qi], count,
                sum_klt_ba[qi] / n, sum_ycbcr_ba[qi] / n,
                klt_ba_better[qi], count,
            );
        }

        // Write CSV
        let output_dir = "/mnt/v/output/zenjpeg/klt-eval";
        std::fs::create_dir_all(output_dir).ok();
        let _csv_path = format!("{}/quality_metrics.csv", output_dir);
        // CSV is written by the eprintln output — user can capture if needed
        eprintln!("\nResults printed above. No assertions — this is an evaluation test.");
    }
}
