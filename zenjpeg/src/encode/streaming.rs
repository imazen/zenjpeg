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

use crate::encode::config::ComputedConfig;
use crate::encode::encoder_types::HuffmanStrategy;
use crate::encode::strip::StripProcessor;
use crate::error::{Error, Result};
use crate::quant::QuantTable;
use crate::types::{JpegMode, Subsampling};
use enough::{Stop, Unstoppable};

/// Output size at or below which entropy-stage trials run (the
/// `Smallest` sequential candidates; `TinyFileMode::Auto`'s exact
/// tiny-table trial). For `Smallest`: progressive output at or
/// below which the
/// sequential candidates. Above it, the progressive stream is
/// emitted directly (one serialization).
///
/// The gate lives in the BYTE domain, not pixels: the crossover is
/// driven by progressive's per-scan overhead versus how much
/// entropy-coded content amortizes it, and pixel count is only a
/// proxy that degenerate content breaks (a near-flat large image
/// produces few bytes). The trigger is self-measuring — the
/// progressive candidate is serialized first, so its size is in
/// hand — and self-cost-limiting: small progressive output implies
/// little entropy content, which bounds the cost of the extra
/// sequential passes regardless of dimensions.
///
/// Provenance: the original 16 KiB gate was set at ~7× the largest
/// crossover then observed (sequential wins at ≤ ~2.4 KB of
/// progressive output; zero wins at larger outputs in photographic
/// RD sweeps). The 2026-06-10 `sweep_validate` run produced a
/// counterexample ABOVE that gate: CID22 1044329 at q10 emits
/// 19.8 KB progressive while plain baseline is 2.0 % smaller
/// (19.4 KB) — low-q photographic content, not synthetic. Gate
/// raised to 32 KiB (~1.6× the new largest observed crossover);
/// regret above the gate shrinks with size (−2.0 % at 19.8 KB is
/// the worst seen) and the trial cost stays bounded by
/// construction. Empirical bound, not a theorem (successive
/// approximation + EOBn restructure the symbol stream, so no
/// partitioned-coding dominance argument closes); revisit again if
/// a larger counterexample appears.
pub(crate) const ENTROPY_TRIAL_MAX_BYTES: usize = 32 * 1024;

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
    /// Access computed config (for fused parallel encoder).
    #[cfg(feature = "parallel")]
    pub(crate) fn config(&self) -> &super::config::ComputedConfig {
        &self.config
    }

    /// Access quant context (for fused parallel encoder).
    #[cfg(feature = "parallel")]
    pub(crate) fn quant_context(&self) -> &super::strip::QuantContext {
        &self.processor.quant
    }

    /// Install a per-iMCU [`super::aq_controller::AqController`] on the
    /// underlying strip processor. Used by the target-zq iteration loop
    /// in [`super::zq`].
    pub(crate) fn set_aq_controller(
        &mut self,
        ctrl: Option<Box<dyn super::aq_controller::AqController>>,
    ) {
        self.processor.set_aq_controller(ctrl);
    }

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

        if builder.use_rgb {
            if builder.use_xyb {
                return Err(Error::invalid_config(
                    "RGB passthrough mode and XYB mode are mutually exclusive".into(),
                ));
            }
            if builder.subsampling != Subsampling::S444 {
                return Err(Error::invalid_config(
                    "RGB passthrough mode requires 4:4:4 (no chroma subsampling)".into(),
                ));
            }
            match builder.pixel_format {
                crate::types::PixelFormat::Rgb
                | crate::types::PixelFormat::Rgba
                | crate::types::PixelFormat::Bgr
                | crate::types::PixelFormat::Bgra
                | crate::types::PixelFormat::Bgrx => {}
                _ => {
                    return Err(Error::unsupported_feature(
                        "RGB passthrough mode requires an 8-bit RGB-family pixel format",
                    ));
                }
            }
        }

        // Generate quantization tables and zero-bias params. Shared with
        // `EncoderConfig::resolve_plan` (encode::plan::resolve_quant_tables)
        // so introspection can never drift from the encoder.
        let is_420 = builder.subsampling == Subsampling::S420;
        let resolved = super::plan::resolve_quant_tables(super::plan::TableResolveInputs {
            quality: builder.quality,
            table_config: &builder.quant_table_config,
            use_xyb: builder.use_xyb,
            use_rgb: builder.use_rgb,
            is_420,
            allow_16bit: builder.allow_16bit_quant_tables,
        });
        let (y_quant, cb_quant, cr_quant) = resolved.quant;
        let (y_zero_bias, cb_zero_bias, cr_zero_bias) = resolved.zero_bias;

        // Build quantization context (all tables + SIMD variants)
        let quant_ctx = crate::encode::strip::QuantContext::new(
            y_quant.clone(),
            cb_quant.clone(),
            cr_quant.clone(),
            y_zero_bias,
            cb_zero_bias,
            cr_zero_bias,
        );

        // Determine if we can use streaming-through encoding early,
        // so StripProcessor can allocate minimal block storage.
        let enable_streaming = !matches!(builder.huffman, HuffmanStrategy::Optimize)
            && builder.mode != JpegMode::Progressive
            && !builder.use_xyb
            && !builder.use_rgb;

        // Create strip processor with quant tables provided at construction.
        // In streaming-through mode, only allocate one strip of block storage.
        let mut processor = if enable_streaming {
            StripProcessor::with_color_modes_streaming(
                width,
                height,
                builder.subsampling,
                builder.pixel_format,
                builder.chroma_downsampling,
                builder.restart_interval,
                builder.use_xyb,
                builder.xyb_subsampling,
                builder.use_rgb,
                quant_ctx,
                builder.aq_enabled,
            )?
        } else {
            StripProcessor::with_color_modes(
                width,
                height,
                builder.subsampling,
                builder.pixel_format,
                builder.chroma_downsampling,
                builder.restart_interval,
                builder.use_xyb,
                builder.xyb_subsampling,
                builder.use_rgb,
                quant_ctx,
                builder.aq_enabled,
            )?
        };

        // Set deringing (on by default in both builder and processor)
        processor.set_deringing(builder.deringing);

        // Set boundary-RD config (off by default; #91). `None` keeps the
        // refinement module out of the hot path entirely.
        // Feature-gated: only wire the flat knobs through when
        // `--features boundary-rd` is enabled.
        #[cfg(feature = "boundary-rd")]
        processor.set_boundary_rd(builder.boundary_rd_flat);

        // Enable trellis quantization if configured
        if let Some(ref trellis) = builder.trellis {
            processor.set_trellis(*trellis);
        }

        let strip_height = processor.strip_height();
        let bytes_per_row = width * builder.pixel_format.bytes_per_pixel();

        // Allocate row buffer for one strip
        let row_buffer = vec![0u8; bytes_per_row * strip_height];

        // Resolve tiny-file mode to a concrete active/inactive flag.
        //
        // Caveats applied here:
        //
        // - XYB mode already uses a shared single-pair Huffman layout (see
        //   `write_scan_header_rgb_ids` and `write_huffman_tables_shared_pair`)
        //   and is not the heuristic's intended target, so tiny-file mode is
        //   a no-op there.
        // - The shared-table path is currently implemented for the baseline
        //   (sequential) writer only. Progressive mode has its own per-scan
        //   AC slot cycling that would need separate work, so we leave it
        //   off there for now.
        // - Optimized Huffman tables must be requested; the shared-table
        //   path builds tables from symbol frequencies.
        // Tiny-file participation is resolved at EMISSION time now:
        // `TinyFileMode::Auto` runs an exact byte-gated trial (see
        // `build_sequential_with_tiny_trial`), `Force` activates when
        // structurally possible, `Off` never. Nothing to resolve here.
        let tiny_file_active = false;

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
            xyb_subsampling: builder.xyb_subsampling,
            use_rgb: builder.use_rgb,
            #[cfg(feature = "parallel")]
            parallel: builder.parallel,
            trellis: builder.trellis,
            edge_padding: crate::types::EdgePaddingConfig::default(),
            original_width: None,
            original_height: None,
            allow_16bit_quant_tables: builder.allow_16bit_quant_tables,
            force_sof1: builder.force_sof1,
            separate_chroma_tables: builder.quant_table_config.separate_chroma_tables(),
            scan_strategy: builder.scan_strategy,
            smallest_scan: builder.smallest_scan,
            smallest_seq_restart_interval: builder.smallest_seq_restart_interval,
            tiny_file_mode: builder.tiny_file_mode,
            tiny_file_active,
        };

        #[allow(unused_mut)]
        let mut config = config;

        // Enforce MCU row alignment for any nonzero restart interval.
        // Non-row-aligned restarts break the fused chroma upsample + color
        // conversion decode path, which processes complete MCU rows.
        if config.restart_interval > 0 {
            config.restart_interval = config.align_restart_to_row(config.restart_interval);
        }

        // Parallel entropy encoding joins segments at restart markers, so it
        // needs a nonzero interval. Resolve the documented auto-selection
        // ("if restart_mcu_rows is 0, the encoder auto-selects") HERE — once,
        // before frequency counting, DRI emission, and entropy encoding — so
        // all three agree. The pre-fix code substituted 64 inside the emitter
        // only, producing RST markers with no DRI header and DC-prediction
        // resets the histogram never saw (sweep issue #197).
        #[cfg(feature = "parallel")]
        if config.parallel && config.restart_interval == 0 {
            config.restart_interval = super::config::resolve_restart_rows(
                4,
                config.width,
                config.height,
                config.subsampling,
            );
        }

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
    #[allow(dead_code)] // Internal API; BytesEncoder has its own wrapper
    #[must_use]
    pub(crate) fn bytes_per_row(&self) -> usize {
        self.bytes_per_row
    }

    /// Returns the total height of the image.
    #[must_use]
    #[allow(dead_code)] // Internal API; BytesEncoder has its own wrapper
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
    #[allow(dead_code)] // Internal API; BytesEncoder has its own wrapper
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
    #[allow(dead_code)] // Internal API; BytesEncoder has its own wrapper
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
    #[allow(dead_code)] // Internal API; BytesEncoder has its own wrapper
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

            // In streaming mode, encode blocks immediately to avoid accumulation
            if self.streaming.is_some() {
                self.encode_new_blocks_streaming()?;
            }

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
    /// Borrows blocks from the strip processor, encodes them to the
    /// BitWriter, then clears them (keeping allocation for reuse).
    fn encode_new_blocks_streaming(&mut self) -> Result<()> {
        let is_color = !self.config.pixel_format.is_grayscale();
        let width = self.width;
        let subsampling = self.config.subsampling;
        let restart_interval = self.config.restart_interval;

        let state = self.streaming.as_mut().unwrap();
        crate::entropy::encode_blocks_mcu_order(
            self.processor.y_blocks(),
            self.processor.cb_blocks(),
            self.processor.cr_blocks(),
            &state.tables,
            &mut state.writer,
            is_color,
            &mut state.entropy_state,
            subsampling,
            width,
            restart_interval,
            state.total_mcus,
        )?;

        self.processor.clear_blocks();

        Ok(())
    }

    /// Finishes encoding and returns the JPEG data.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Not all rows have been pushed
    #[allow(dead_code)] // Internal API; BytesEncoder has its own wrapper
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

    #[allow(dead_code)] // Internal API; BytesEncoder has its own wrapper
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

            if config.smallest_scan {
                let is_color = !config.pixel_format.is_grayscale();
                let counts = Box::new(config.count_block_frequencies(
                    &strip_output.y_blocks,
                    &strip_output.cb_blocks,
                    &strip_output.cr_blocks,
                    is_color,
                ));
                Self::build_smallest_into(
                    &config,
                    &y_quant,
                    &cb_quant,
                    &cr_quant,
                    &strip_output,
                    output,
                )?;
                return Ok(Some(counts));
            }

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
                    Self::build_sequential_with_tiny_trial(
                        &config,
                        &y_quant,
                        &cb_quant,
                        &cr_quant,
                        &strip_output,
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

    #[allow(dead_code)] // Internal API; BytesEncoder has its own wrapper
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

        if config.smallest_scan {
            return Self::build_smallest_into(
                config,
                y_quant,
                cb_quant,
                cr_quant,
                &strip_output,
                output,
            );
        }

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
                Self::build_sequential_with_tiny_trial(
                    config,
                    y_quant,
                    cb_quant,
                    cr_quant,
                    &strip_output,
                    output,
                    false,
                )?;
                Ok(())
            }
        }
    }

    #[allow(dead_code)] // Internal API; BytesEncoder has its own wrapper
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
            &strip_output,
            &mut output,
            false,
        )?;
        Ok(output)
    }

    /// Sequential emission with exact tiny-file resolution.
    ///
    /// `TinyFileMode::Auto` no longer uses pixel-count crossover
    /// heuristics: the plain sequential stream is emitted, and when it
    /// lands at or below [`ENTROPY_TRIAL_MAX_BYTES`](ENTROPY_TRIAL_MAX_BYTES)
    /// the tiny-file shared-table variant is also serialized and the
    /// smaller wins — identical pixels, exact `min(bytes)`. `Force`
    /// activates tiny tables whenever structurally possible; `Off`
    /// never does.
    fn build_sequential_with_tiny_trial(
        config: &ComputedConfig,
        y_quant: &QuantTable,
        cb_quant: &QuantTable,
        cr_quant: &QuantTable,
        strip_output: &crate::encode::strip::StripProcessorOutput,
        output: &mut Vec<u8>,
        collect_frequencies: bool,
    ) -> Result<Option<Box<super::blocks::HuffmanSymbolFrequencies>>> {
        use super::encoder_types::TinyFileMode;
        // Tiny shared tables need optimized Huffman and are not defined
        // for the XYB/RGB serialization paths (both already use a single
        // shared table pair, so there is nothing for tiny mode to merge).
        let tiny_possible = !config.use_xyb
            && !config.use_rgb
            && matches!(config.huffman, HuffmanStrategy::Optimize);
        // Grayscale tiny mode is a DOMINANCE case, not a trial case:
        // with a single component there is no shared-table cost to
        // weigh — it is pure header pruning (~208 B), smaller at every
        // size. Color tiny mode trades shared-table scan bits against
        // header savings, so it gets the exact byte-gated trial.
        let is_grayscale = config.pixel_format.is_grayscale();
        let (plain_active, trial) = match config.tiny_file_mode {
            TinyFileMode::Off => (false, false),
            TinyFileMode::Force => (tiny_possible, false),
            TinyFileMode::Auto if is_grayscale => (tiny_possible, false),
            TinyFileMode::Auto => (false, tiny_possible),
        };

        let mut plain_cfg = config.clone();
        plain_cfg.tiny_file_active = plain_active;
        let counts = Self::build_jpeg_sequential_into(
            &plain_cfg,
            y_quant,
            cb_quant,
            cr_quant,
            strip_output,
            output,
            collect_frequencies,
        )?;

        if trial && output.len() <= ENTROPY_TRIAL_MAX_BYTES {
            let mut tiny_cfg = config.clone();
            tiny_cfg.tiny_file_active = true;
            let mut tiny = Vec::new();
            Self::build_jpeg_sequential_into(
                &tiny_cfg,
                y_quant,
                cb_quant,
                cr_quant,
                strip_output,
                &mut tiny,
                false,
            )?;
            if tiny.len() < output.len() {
                output.clear();
                output.extend_from_slice(&tiny);
            }
        }

        Ok(counts)
    }

    /// Smallest-entropy selection: emit the smallest byte stream for the
    /// SAME quantized coefficients.
    ///
    /// At or below [`SMALLEST_TRIAL_MAX_PIXELS`](Self::SMALLEST_TRIAL_MAX_PIXELS):
    /// exact `min` over {sequential, sequential+tiny (when eligible),
    /// progressive} — every candidate decodes to identical pixels, so
    /// this is a pure rate decision. Above: progressive is emitted
    /// directly (measured-universal winner at those sizes; one
    /// serialization).
    fn build_smallest_into(
        config: &ComputedConfig,
        y_quant: &QuantTable,
        cb_quant: &QuantTable,
        cr_quant: &QuantTable,
        strip_output: &crate::encode::strip::StripProcessorOutput,
        output: &mut Vec<u8>,
    ) -> Result<()> {
        if !matches!(config.huffman, HuffmanStrategy::Optimize) {
            return Err(Error::unsupported_feature(
                "Smallest scan selection requires optimized Huffman tables",
            ));
        }

        // Candidate 1: progressive. The clone inherits the config's
        // ScanStrategy, so under SmallestSearch this candidate is the
        // scan-script search winner (which always includes the default
        // jpegli script in its candidate set).
        let mut prog_cfg = config.clone();
        prog_cfg.smallest_scan = false;
        prog_cfg.mode = JpegMode::Progressive;
        prog_cfg.tiny_file_active = false;
        let mut prog = Vec::new();
        prog_cfg.encode_progressive_from_blocks_into(
            &strip_output.y_blocks,
            &strip_output.cb_blocks,
            &strip_output.cr_blocks,
            y_quant,
            cb_quant,
            cr_quant,
            &mut prog,
        )?;

        // Gate the sequential trials on the measured progressive size:
        // above the threshold, emit progressive directly (one
        // serialization, zero trial overhead).
        let run_trials = prog.len() <= ENTROPY_TRIAL_MAX_BYTES;
        if !run_trials {
            output.clear();
            output
                .try_reserve(prog.len())
                .map_err(|_| Error::allocation_failed(prog.len(), "smallest-scan output"))?;
            output.extend_from_slice(&prog);
            return Ok(());
        }

        // Candidate 2: plain sequential.
        let mut seq_cfg = config.clone();
        seq_cfg.smallest_scan = false;
        seq_cfg.mode = JpegMode::Baseline;
        seq_cfg.tiny_file_active = false;
        // Pure rate minimizer: restart-free, and the progressive
        // alternative cannot restart-parallel-decode either, so 0
        // trades essentially nothing away. NOT a strict dominance:
        // restart markers also re-base DC prediction, which on rare
        // content nets out cheaper than the marker overhead
        // (sweep_validate 2026-06-10: a 4-MCU-row-interval baseline
        // beat this restart-free candidate by 8 bytes / 0.04 % on
        // CID22 1044329 q10). Smallest deliberately does not sweep
        // restart-interval space for that tail.
        // `force_restart_markers` overrides via
        // smallest_seq_restart_interval (resolved in byte_encoders,
        // where that knob is in scope).
        seq_cfg.restart_interval = config.smallest_seq_restart_interval;
        let mut seq = Vec::new();
        Self::build_jpeg_sequential_into(
            &seq_cfg,
            y_quant,
            cb_quant,
            cr_quant,
            strip_output,
            &mut seq,
            false,
        )?;

        let mut best = if seq.len() < prog.len() { seq } else { prog };

        // Candidate 3: sequential with tiny-file shared tables. Replaces the
        // pixel-count activation heuristic with the exact answer; skipped
        // only when structurally ineligible (XYB) or explicitly Off.
        let tiny_eligible = !config.use_xyb
            && !config.use_rgb
            && !matches!(
                config.tiny_file_mode,
                super::encoder_types::TinyFileMode::Off
            );
        if tiny_eligible {
            let mut tiny_cfg = config.clone();
            tiny_cfg.smallest_scan = false;
            tiny_cfg.mode = JpegMode::Baseline;
            tiny_cfg.tiny_file_active = true;
            tiny_cfg.restart_interval = config.smallest_seq_restart_interval;
            let mut tiny = Vec::new();
            Self::build_jpeg_sequential_into(
                &tiny_cfg,
                y_quant,
                cb_quant,
                cr_quant,
                strip_output,
                &mut tiny,
                false,
            )?;
            if tiny.len() < best.len() {
                best = tiny;
            }
        }

        output.clear();
        output
            .try_reserve(best.len())
            .map_err(|_| Error::allocation_failed(best.len(), "smallest-scan output"))?;
        output.extend_from_slice(&best);
        Ok(())
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
        strip_output: &crate::encode::strip::StripProcessorOutput,
        output: &mut Vec<u8>,
        collect_frequencies: bool,
    ) -> Result<Option<Box<super::blocks::HuffmanSymbolFrequencies>>> {
        let width = config.width as usize;
        let height = config.height as usize;

        output.clear();
        output
            .try_reserve(width * height / 4)
            .map_err(|_| Error::allocation_failed(width * height / 4, "sequential jpeg output"))?;

        let (scan_data, frequencies) = if config.use_xyb {
            Self::encode_sequential_xyb(
                config,
                y_quant,
                cb_quant,
                cr_quant,
                strip_output,
                output,
                collect_frequencies,
            )?
        } else if config.use_rgb {
            Self::encode_sequential_rgb(config, y_quant, strip_output, output)?
        } else {
            Self::encode_sequential_ycbcr(
                config,
                y_quant,
                cb_quant,
                cr_quant,
                strip_output,
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

        // Use SOF1 if any quant table needs 16-bit precision, or if forced (XYB DC categories)
        let is_extended = config.force_sof1
            || y_quant.precision > 0
            || cb_quant.precision > 0
            || cr_quant.precision > 0;
        config.write_frame_header_xyb_ex(output, is_extended)?;

        // Branch on XYB B-channel layout. Full uses 1+1+1 blocks per MCU
        // (matches its R:1×1 G:1×1 B:1×1 SOF); BQuarter uses 4+4+1.
        let xyb_full = config.xyb_subsampling == crate::encode::encoder_types::XybSubsampling::Full;

        if matches!(config.huffman, HuffmanStrategy::Optimize) {
            let (dc_table, ac_table, frequencies) = if xyb_full {
                let (dc, ac, f) = config.build_optimized_tables_shared_full(
                    &strip_output.y_blocks,
                    &strip_output.cb_blocks,
                    &strip_output.cr_blocks,
                )?;
                (dc, ac, if collect_frequencies { Some(f) } else { None })
            } else if collect_frequencies {
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

            config.write_huffman_tables_shared_pair(output, &dc_table, &ac_table);

            if config.restart_interval > 0 {
                config.write_restart_interval(output)?;
            }
            config.write_scan_header_rgb_ids(output)?;

            let scan_data = if xyb_full {
                config.encode_with_tables_shared_full(
                    &strip_output.y_blocks,
                    &strip_output.cb_blocks,
                    &strip_output.cr_blocks,
                    &dc_table,
                    &ac_table,
                )?
            } else {
                config.encode_with_tables_xyb_raster(
                    &strip_output.y_blocks,
                    &strip_output.cb_blocks,
                    &strip_output.cr_blocks,
                    &dc_table,
                    &ac_table,
                )?
            };
            Ok((scan_data, frequencies))
        } else if let HuffmanStrategy::Custom(ref tables) = config.huffman {
            // Custom tables: XYB uses dc_luma/ac_luma as the shared pair.
            config.write_huffman_tables_shared_pair(output, &tables.dc_luma, &tables.ac_luma);

            if config.restart_interval > 0 {
                config.write_restart_interval(output)?;
            }
            config.write_scan_header_rgb_ids(output)?;

            let scan_data = if xyb_full {
                config.encode_with_tables_shared_full(
                    &strip_output.y_blocks,
                    &strip_output.cb_blocks,
                    &strip_output.cr_blocks,
                    &tables.dc_luma,
                    &tables.ac_luma,
                )?
            } else {
                config.encode_with_tables_xyb_raster(
                    &strip_output.y_blocks,
                    &strip_output.cb_blocks,
                    &strip_output.cr_blocks,
                    &tables.dc_luma,
                    &tables.ac_luma,
                )?
            };
            Ok((scan_data, None))
        } else {
            // Fixed: use general-purpose trained tables for XYB
            let tables = crate::huffman::builtin_tables::select_tables(
                &config.quality,
                true,
                config.subsampling,
            );
            config.write_huffman_tables_shared_pair(output, &tables.dc_luma, &tables.ac_luma);

            if config.restart_interval > 0 {
                config.write_restart_interval(output)?;
            }
            config.write_scan_header_rgb_ids(output)?;

            let scan_data = if xyb_full {
                config.encode_with_tables_shared_full(
                    &strip_output.y_blocks,
                    &strip_output.cb_blocks,
                    &strip_output.cr_blocks,
                    &tables.dc_luma,
                    &tables.ac_luma,
                )?
            } else {
                config.encode_with_tables_xyb_raster(
                    &strip_output.y_blocks,
                    &strip_output.cb_blocks,
                    &strip_output.cr_blocks,
                    &tables.dc_luma,
                    &tables.ac_luma,
                )?
            };
            Ok((scan_data, None))
        }
    }

    /// Encodes sequential JPEG in RGB passthrough mode (issue #185).
    ///
    /// Mirrors C++ jpegli's non-XYB `JCS_RGB` serialization: component IDs
    /// 'R','G','B' at 1×1 sampling, a single shared quantization table
    /// (slot 0) and a single shared Huffman pair (slot 0), plus an Adobe
    /// APP14 marker with transform=0 (libjpeg also writes one for
    /// `JCS_RGB`; jpegli relies on component IDs alone). The 1+1+1 MCU
    /// layout is identical to XYB-Full, so the XYB-Full table builders
    /// and entropy encoder are reused directly.
    ///
    /// RGB mode never collects Huffman symbol frequencies (like XYB).
    fn encode_sequential_rgb(
        config: &ComputedConfig,
        quant: &QuantTable,
        strip_output: &crate::encode::strip::StripProcessorOutput,
        output: &mut Vec<u8>,
    ) -> Result<(
        Vec<u8>,
        Option<Box<super::blocks::HuffmanSymbolFrequencies>>,
    )> {
        config.write_header(output)?;
        config.write_app14_adobe(output, 0)?;
        config.write_quant_tables_rgb(output, quant)?;

        // SOF1 only when the shared table needs 16-bit precision. Unlike
        // XYB there is no forced SOF1: passthrough DC stays in baseline
        // category range.
        let is_extended = quant.precision > 0;
        config.write_frame_header_rgb(output, is_extended)?;

        if matches!(config.huffman, HuffmanStrategy::Optimize) {
            let (dc_table, ac_table, _f) = config.build_optimized_tables_shared_full(
                &strip_output.y_blocks,
                &strip_output.cb_blocks,
                &strip_output.cr_blocks,
            )?;

            config.write_huffman_tables_shared_pair(output, &dc_table, &ac_table);

            if config.restart_interval > 0 {
                config.write_restart_interval(output)?;
            }
            config.write_scan_header_rgb_ids(output)?;

            let scan_data = config.encode_with_tables_shared_full(
                &strip_output.y_blocks,
                &strip_output.cb_blocks,
                &strip_output.cr_blocks,
                &dc_table,
                &ac_table,
            )?;
            Ok((scan_data, None))
        } else if let HuffmanStrategy::Custom(ref tables) = config.huffman {
            // Custom tables: like XYB, dc_luma/ac_luma serve as the shared pair.
            config.write_huffman_tables_shared_pair(output, &tables.dc_luma, &tables.ac_luma);

            if config.restart_interval > 0 {
                config.write_restart_interval(output)?;
            }
            config.write_scan_header_rgb_ids(output)?;

            let scan_data = config.encode_with_tables_shared_full(
                &strip_output.y_blocks,
                &strip_output.cb_blocks,
                &strip_output.cr_blocks,
                &tables.dc_luma,
                &tables.ac_luma,
            )?;
            Ok((scan_data, None))
        } else {
            // Fixed: general-purpose trained tables; the luma pair is shared.
            let tables = crate::huffman::builtin_tables::select_tables(
                &config.quality,
                false,
                config.subsampling,
            );
            config.write_huffman_tables_shared_pair(output, &tables.dc_luma, &tables.ac_luma);

            if config.restart_interval > 0 {
                config.write_restart_interval(output)?;
            }
            config.write_scan_header_rgb_ids(output)?;

            let scan_data = config.encode_with_tables_shared_full(
                &strip_output.y_blocks,
                &strip_output.cb_blocks,
                &strip_output.cr_blocks,
                &tables.dc_luma,
                &tables.ac_luma,
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

        // Use SOF1 if any quant table needs 16-bit precision, or if forced (XYB DC categories)
        let is_extended = config.force_sof1
            || y_quant.precision > 0
            || cb_quant.precision > 0
            || cr_quant.precision > 0;
        config.write_frame_header_ex(output, is_extended)?;

        if matches!(config.huffman, HuffmanStrategy::Optimize) {
            if config.tiny_file_active {
                // Tiny-file path: merge luma+chroma frequencies, emit a
                // single DC+AC DHT pair, and write SOS with `Td=0, Ta=0` for
                // every component. The entropy encoder still indexes tables
                // by channel (slot 0 for luma, slot 1 for chroma), but both
                // slots point at the same merged tables — the bitstream is
                // therefore unchanged relative to a hypothetical "slot 0
                // everywhere" variant, and the file shrinks by exactly the
                // two table headers (~210 bytes).
                let tables = config.build_shared_tables(
                    &strip_output.y_blocks,
                    &strip_output.cb_blocks,
                    &strip_output.cr_blocks,
                    is_color,
                )?;

                config.write_huffman_tables_shared(output, &tables.dc_luma, &tables.ac_luma)?;

                if config.restart_interval > 0 {
                    config.write_restart_interval(output)?;
                }
                config.write_scan_header_shared(output)?;

                let scan_data = config.encode_with_tables(
                    &strip_output.y_blocks,
                    &strip_output.cb_blocks,
                    &strip_output.cr_blocks,
                    is_color,
                    Some(&tables),
                )?;
                return Ok((scan_data, None));
            }

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
        use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

        // Create a small test image
        let width = 32u32;
        let height = 32u32;
        let pixels: Vec<u8> = (0..width * height * 3)
            .map(|i| ((i * 17) % 256) as u8)
            .collect();

        let _ = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
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
                "output lengths differ at {perm}"
            );
            assert_eq!(oneshot_result, streaming_result, "outputs differ at {perm}");
        });
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

        // Use row-aligned interval: 16 MCU cols for 128px 4:4:4
        let jpeg = StreamingEncoder::new(width as u32, height as u32)
            .custom_huffman_tables(tables)
            .restart_interval(16)
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
        use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

        // Row-by-row and encode() convenience should produce identical output
        let width = 64;
        let height = 64;
        let data = make_test_image(width, height);

        let _ = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
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
                "encode() and row-by-row should produce identical output at {perm}"
            );
        });
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
    fn test_streaming_round_trip_decode() {
        let width = 128;
        let height = 128;
        let data = make_test_image(width, height);
        let tables = crate::huffman::optimize::HuffmanTableSet::from_standard().unwrap();

        let jpeg = StreamingEncoder::new(width as u32, height as u32)
            .custom_huffman_tables(tables)
            .encode(&data)
            .unwrap();

        let decoded = crate::decode::Decoder::new()
            .decode(&jpeg, enough::Unstoppable)
            .unwrap();
        assert_eq!(decoded.width, width as u32);
        assert_eq!(decoded.height, height as u32);
        assert_eq!(decoded.pixels_u8().unwrap().len(), width * height * 3);
    }

    /// Decode 4:2:0 streaming output and verify dimensions.
    #[test]
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

        let decoded = crate::decode::Decoder::new()
            .decode(&jpeg, enough::Unstoppable)
            .unwrap();
        assert_eq!(decoded.width, width as u32);
        assert_eq!(decoded.height, height as u32);
    }

    /// Decode streaming output with restart markers and verify it decodes correctly.
    #[test]
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

        let decoded = crate::decode::Decoder::new()
            .decode(&jpeg, enough::Unstoppable)
            .unwrap();
        assert_eq!(decoded.width, width as u32);
        assert_eq!(decoded.height, height as u32);
        assert_eq!(decoded.pixels_u8().unwrap().len(), width * height * 3);
    }

    /// Decode non-aligned streaming output with 4:2:0.
    #[test]
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
}
