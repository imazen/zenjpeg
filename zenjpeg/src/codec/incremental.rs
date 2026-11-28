//! zencodec incremental decode impl: `JpegIncrementalDecoder`.

use alloc::borrow::Cow;
use alloc::vec::Vec;

use enough::Stop;
use imgref::ImgRefMut;
use whereat::At;
use zencodec::Pulse;
use zencodec::decode::{
    BatchMeta, DecodeRowSink, Event, IncrementalDecode, Input, InputRequest, Progress, RowStrip,
    StepError,
};
use zencodec::{CodecError, ImageInfo, ResourceLimits};
use zenpixels::{ChannelOrder, PixelBuf, PixelFormat, SampleType};

use crate::decode::parser::JpegParser;
use crate::decode::{ChromaUpsampling, DeblockMode, DecodeConfig, OutputTarget, ScanlineReader};
use crate::error::Error;

use super::decode::{JpegDecoderConfig, build_decode_config, will_auto_orient};
use super::info::{decode_descriptor, to_image_info};

/// Incremental JPEG decoder implementing [`zencodec::decode::IncrementalDecode`].
///
/// Accepts compressed input bytes in arbitrary chunks without performing any I/O,
/// decoding and yielding scanlines/batches as soon as sufficient data has been pushed.
///
/// Interleaved 8-bit sequential (baseline) YCbCr/Grayscale JPEGs stream MCU row by MCU row
/// directly from incoming chunks when configured with [`ChromaUpsampling::NearestNeighbor`],
/// preserved orientation, and standard 8-bit output. Progressive, arithmetic, non-interleaved,
/// 12-bit, CMYK, XYB, auto-oriented, cropped, gain-map, and fancy-upsampled decodes buffer
/// within effective resource limits and execute the exact one-shot [`JpegDecodeJob`] decoder
/// at EOF, delivering strips directly from the decoded buffer without a second full-frame copy.
pub struct JpegIncrementalDecoder {
    config: JpegDecoderConfig,
    limits: ResourceLimits,
    effective_max_pixels: u64,
    effective_max_memory: u64,
    crop_hint: Option<(u32, u32, u32, u32)>,
    orientation: zencodec::OrientationHint,
    policy: Option<zencodec::decode::DecodePolicy>,
    gain_map_render: zencodec::GainMapRender,
    stop: Option<zencodec::SharedPulse>,
    preferred: Vec<PixelFormat>,
    format: Option<PixelFormat>,
    info: Option<ImageInfo>,

    // Header / whole-file accumulation & mode selection
    header_buf: Vec<u8>,
    header_parsed: bool,
    buffered_whole_file: bool,
    buffered_output: Option<PixelBuf>,
    buffered_y: u32,

    // Streaming scanline reader (baseline single-scan MCU-row mode)
    reader: Option<ScanlineReader<'static>>,
    mcu_height: usize,
    current_row: u32,
    row_buf: Vec<u8>,

    // Pending compressed scan data (for baseline streaming)
    pending_scan: Vec<u8>,
    pending_scan_pos: usize,
    mcu_row_ready: bool,

    // Status
    is_eof: bool,
    is_complete: bool,
    total_input_bytes: u64,
    request: [InputRequest; 1],
    begun: bool,
    failed: bool,
    done: Option<zencodec::decode::OutputInfo>,
    pending_rows: u32,
    pending_y: u32,
    pending_offset_rows: usize,
    metadata: crate::container::marker::MetadataCapture,
}

fn normalize_unlimited_limit(value: u64) -> u64 {
    if value == 0 { u64::MAX } else { value }
}

/// Admit actual backing-capacity growth for `vec` before allocating, keeping
/// `other_retained + vec.capacity() <= max_memory`.
fn try_admit_vec_growth(
    vec: &mut Vec<u8>,
    additional: usize,
    other_retained: u64,
    max_memory: u64,
    context: &'static str,
) -> Result<(), At<CodecError>> {
    let needed_len = vec
        .len()
        .checked_add(additional)
        .ok_or_else(|| Error::allocation_failed(usize::MAX, context))?;
    if needed_len <= vec.capacity() {
        return Ok(());
    }
    let projected_min = other_retained.saturating_add(needed_len as u64);
    if projected_min > max_memory {
        return Err(Error::resource_limit_exceeded(
            zencodec::LimitKind::Memory,
            projected_min,
            max_memory,
        )
        .into());
    }
    let max_self = max_memory
        .saturating_sub(other_retained)
        .min(usize::MAX as u64) as usize;
    let doubled = vec.capacity().saturating_mul(2).max(needed_len);
    let target_cap = doubled.min(max_self).max(needed_len);
    let extra = target_cap.saturating_sub(vec.len());
    vec.try_reserve_exact(extra)
        .map_err(|_| Error::allocation_failed(extra, context))?;
    let actual = other_retained.saturating_add(vec.capacity() as u64);
    if actual > max_memory {
        return Err(Error::resource_limit_exceeded(
            zencodec::LimitKind::Memory,
            actual,
            max_memory,
        )
        .into());
    }
    Ok(())
}

impl JpegIncrementalDecoder {
    /// Create a new incremental decoder from job parameters.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        config: JpegDecoderConfig,
        limits: ResourceLimits,
        crop_hint: Option<(u32, u32, u32, u32)>,
        orientation: zencodec::OrientationHint,
        policy: Option<zencodec::decode::DecodePolicy>,
        gain_map_render: zencodec::GainMapRender,
        stop: Option<zencodec::SharedPulse>,
        preferred: &[PixelFormat],
        force_buffered: bool,
    ) -> Self {
        // Preserve explicit zencodec job caps (including `Some(0)`), while normalizing
        // native `DecodeConfig`'s `0 == unlimited` sentinel when the job did not set a cap.
        let effective_max_pixels = match limits.max_pixels {
            Some(n) => n,
            None => normalize_unlimited_limit(config.inner.get_max_pixels()),
        };
        let effective_max_memory = match limits.max_memory_bytes {
            Some(n) => n,
            None => normalize_unlimited_limit(config.inner.max_memory),
        };
        Self {
            config,
            limits,
            effective_max_pixels,
            effective_max_memory,
            crop_hint,
            orientation,
            policy,
            gain_map_render,
            stop,
            preferred: preferred.to_vec(),
            format: None,
            info: None,
            header_buf: Vec::new(),
            header_parsed: force_buffered,
            buffered_whole_file: force_buffered,
            buffered_output: None,
            buffered_y: 0,
            reader: None,
            mcu_height: 8,
            current_row: 0,
            row_buf: Vec::new(),
            pending_scan: Vec::new(),
            pending_scan_pos: 0,
            mcu_row_ready: false,
            is_eof: false,
            is_complete: false,
            total_input_bytes: 0,
            request: [InputRequest::AtLeast {
                offset: 0,
                min_len: 1,
            }],
            begun: false,
            failed: false,
            done: None,
            pending_rows: 0,
            pending_y: 0,
            pending_offset_rows: 0,
            metadata: crate::container::marker::MetadataCapture::new(),
        }
    }

    fn retained_except_header(&self) -> u64 {
        (self.pending_scan.capacity() as u64)
            .saturating_add(self.row_buf.capacity() as u64)
            .saturating_add(self.metadata.capacity() as u64)
    }

    fn retained_except_scan(&self) -> u64 {
        (self.header_buf.capacity() as u64)
            .saturating_add(self.row_buf.capacity() as u64)
            .saturating_add(self.metadata.capacity() as u64)
    }

    fn retained_except_metadata(&self) -> u64 {
        (self.header_buf.capacity() as u64)
            .saturating_add(self.pending_scan.capacity() as u64)
            .saturating_add(self.row_buf.capacity() as u64)
    }

    /// Try parsing the header and initializing the streaming scanline reader,
    /// or switch cleanly to whole-file buffered decode for non-streamable JPEGs.
    fn try_parse_header(&mut self, is_eof: bool) -> Result<bool, At<CodecError>> {
        let mut parser = match JpegParser::with_strictness(
            &self.header_buf,
            self.effective_max_pixels,
            self.effective_max_memory,
            None,
            self.config.inner.strictness,
            self.config.inner.alloc_pref,
        ) {
            Ok(p) => p,
            Err(e) if !is_eof && (self.header_buf.len() < 2 || e.is_truncated()) => {
                return Ok(false);
            }
            Err(e) => return Err(e.into()),
        };

        match parser.read_header() {
            Ok(()) => {
                let header = parser.info();
                self.limits
                    .check_dimensions(header.dimensions.width, header.dimensions.height)
                    .map_err(Error::from)?;
                let pixels = u64::from(header.dimensions.width)
                    .saturating_mul(u64::from(header.dimensions.height));
                if pixels > self.effective_max_pixels {
                    return Err(Error::resource_limit_exceeded(
                        zencodec::LimitKind::Pixels,
                        pixels,
                        self.effective_max_pixels,
                    )
                    .into());
                }
                let info = to_image_info(&header);
                let format = decode_descriptor(
                    &self.preferred,
                    &header,
                    self.config.inner.correct_color.as_ref(),
                )?;
                let format_streamable = format.layout.sample == SampleType::U8
                    && matches!(
                        format.layout.order,
                        ChannelOrder::Gray
                            | ChannelOrder::Rgb
                            | ChannelOrder::Rgba
                            | ChannelOrder::Rgbx
                            | ChannelOrder::Bgra
                            | ChannelOrder::Bgrx
                    );
                let can_stream_baseline =
                    DecodeConfig::can_stream_incremental(&parser) && format_streamable;

                if !can_stream_baseline {
                    self.metadata = crate::container::marker::MetadataCapture::new();
                    self.buffered_whole_file = true;
                    self.header_parsed = true;
                    return Ok(true);
                }

                let is_grayscale = parser.num_components == 1;
                match parser.into_scan_data(is_grayscale) {
                    Ok(scan_data) => {
                        if !is_grayscale && !scan_data.interleaved {
                            self.metadata = crate::container::marker::MetadataCapture::new();
                            self.buffered_whole_file = true;
                            self.header_parsed = true;
                            return Ok(true);
                        }
                        let scan_data_start = scan_data.scan_data_start;
                        let width = scan_data.width;
                        let height = scan_data.height;
                        let mut reader = ScanlineReader::from_scan_data_cow(
                            scan_data,
                            Cow::Borrowed(&[]),
                            ChromaUpsampling::NearestNeighbor,
                            self.config.inner.effective_idct_method(),
                            OutputTarget::Srgb8,
                            DeblockMode::Off,
                            self.config.inner.alloc_pref,
                        )?;

                        if self.header_buf.len() > scan_data_start {
                            let tail_len = self.header_buf.len() - scan_data_start;
                            let other = self.retained_except_scan();
                            try_admit_vec_growth(
                                &mut self.pending_scan,
                                tail_len,
                                other,
                                self.effective_max_memory,
                                "incremental JPEG scan tail",
                            )?;
                            self.pending_scan
                                .extend_from_slice(&self.header_buf[scan_data_start..]);
                        }
                        self.header_buf = Vec::new();

                        let mut cfg = build_decode_config(
                            &self.config.inner,
                            &self.limits,
                            self.crop_hint,
                            self.orientation,
                            self.policy.as_ref(),
                        );
                        cfg = cfg.chroma_upsampling(ChromaUpsampling::NearestNeighbor);

                        let mcu_height = reader.luma_rows_per_mcu();
                        cfg.apply_crop(&mut reader, width, height, mcu_height)?;

                        self.info = Some(info);
                        self.format = Some(format);
                        self.mcu_height = mcu_height;
                        self.reader = Some(reader);
                        self.header_parsed = true;
                        Ok(true)
                    }
                    Err(e) if !is_eof && e.is_truncated() => Ok(false),
                    Err(e) => Err(e.into()),
                }
            }
            Err(e) if !is_eof && e.is_truncated() => Ok(false),
            Err(e) => Err(e.into()),
        }
    }

    fn push_chunk(
        &mut self,
        chunk: &[u8],
        is_eof: bool,
        stop: &dyn Stop,
    ) -> Result<(), At<CodecError>> {
        stop.check().map_err(Error::from)?;

        if self.is_complete {
            return Ok(());
        }

        self.total_input_bytes = self.total_input_bytes.saturating_add(chunk.len() as u64);
        self.limits
            .check_input_size(self.total_input_bytes)
            .map_err(|_| {
                Error::resource_limit_exceeded(
                    zencodec::LimitKind::InputSize,
                    self.total_input_bytes,
                    self.limits.max_input_bytes.unwrap_or(0),
                )
            })?;

        if is_eof {
            self.is_eof = true;
        }

        if !self.buffered_whole_file {
            let other = self.retained_except_metadata();
            self.metadata
                .push_bounded(chunk, other, self.effective_max_memory)?;
        }

        // Phase 1: Header or Whole-File Accumulation
        if !self.header_parsed {
            let other = self.retained_except_header();
            try_admit_vec_growth(
                &mut self.header_buf,
                chunk.len(),
                other,
                self.effective_max_memory,
                "incremental JPEG header",
            )?;
            self.header_buf.extend_from_slice(chunk);
            if !self.try_parse_header(is_eof)? {
                return Ok(());
            }
        } else if !self.buffered_whole_file {
            let other = self.retained_except_scan();
            try_admit_vec_growth(
                &mut self.pending_scan,
                chunk.len(),
                other,
                self.effective_max_memory,
                "incremental JPEG scan",
            )?;
            self.pending_scan.extend_from_slice(chunk);
        } else {
            let other = self.retained_except_header();
            try_admit_vec_growth(
                &mut self.header_buf,
                chunk.len(),
                other,
                self.effective_max_memory,
                "incremental JPEG whole-file buffer",
            )?;
            self.header_buf.extend_from_slice(chunk);
        }

        // Phase 2: Whole-File Buffered Fallback (exact one-shot JpegDecodeJob parity)
        if self.buffered_whole_file {
            if !self.is_eof {
                return Ok(());
            }
            if self.buffered_output.is_none() {
                use zencodec::decode::{DecodeJob as _, DecoderConfig as _};
                let mut job = self
                    .config
                    .clone()
                    .job()
                    .with_limits(self.limits)
                    .with_orientation(self.orientation)
                    .with_gain_map_render(self.gain_map_render);
                if let Some((x, y, w, h)) = self.crop_hint {
                    job = job.with_crop_hint(zencodec::Rect::new(x, y, w, h));
                }
                if let Some(policy) = self.policy {
                    job = job.try_with_policy(policy).map_err(Error::from)?;
                }
                if let Some(stop_tok) = self.stop.clone() {
                    job = job.with_pulse(stop_tok);
                }
                let data = core::mem::take(&mut self.header_buf);
                let out = job
                    .decoder(Cow::Owned(data), &self.preferred)?
                    .decode_with_stop(Some(stop))?;
                let fmt = out.pixels().format().clone();
                let info = out.info().clone();
                self.info = Some(info);
                self.format = Some(fmt);
                self.buffered_y = 0;
                self.buffered_output = Some(out.into_buffer());
            }
            return Ok(());
        }

        // Phase 3: Baseline Streaming
        if !self.mcu_row_ready
            && let Some(ref mut reader) = self.reader
        {
            let slice = &self.pending_scan[self.pending_scan_pos..];
            if let Some(bytes) = reader.decode_mcu_row_slice(slice, self.is_eof, stop)? {
                self.pending_scan_pos += bytes;
                self.mcu_row_ready = true;
            }
        }

        if self
            .reader
            .as_ref()
            .is_some_and(|reader| reader.is_finished())
        {
            self.is_complete = true;
        }
        Ok(())
    }

    /// Decode the next streaming MCU row into `self.row_buf`, returning `(y, row_count)`.
    fn pull_streaming_batch(
        &mut self,
        stop: &dyn Stop,
    ) -> Result<Option<(u32, u32)>, At<CodecError>> {
        stop.check().map_err(Error::from)?;

        if self.is_complete || self.buffered_whole_file {
            return Ok(None);
        }

        if !self.mcu_row_ready
            && let Some(ref mut reader) = self.reader
            && !reader.is_finished()
        {
            let slice = &self.pending_scan[self.pending_scan_pos..];
            if let Some(bytes) = reader.decode_mcu_row_slice(slice, self.is_eof, stop)? {
                self.pending_scan_pos += bytes;
                self.mcu_row_ready = true;
            }
        }

        if !self.mcu_row_ready {
            if let Some(ref reader) = self.reader
                && reader.is_finished()
            {
                self.is_complete = true;
            }
            return Ok(None);
        }

        let Some(reader) = self.reader.as_mut() else {
            return Ok(None);
        };

        if reader.is_finished() {
            self.is_complete = true;
            return Ok(None);
        }

        let width = reader.width() as usize;
        let format = self
            .format
            .as_ref()
            .ok_or_else(|| Error::internal("incremental decode: format missing"))?;

        let bpp = format.layout.bytes_per_pixel();
        let row_bytes = width
            .checked_mul(bpp)
            .ok_or_else(|| Error::allocation_failed(usize::MAX, "incremental JPEG row stride"))?;
        let batch_rows = self.mcu_height;
        let batch_bytes = row_bytes
            .checked_mul(batch_rows)
            .ok_or_else(|| Error::allocation_failed(usize::MAX, "incremental JPEG MCU strip"))?;

        if self.row_buf.len() < batch_bytes {
            let grow = batch_bytes - self.row_buf.len();
            let other = (self.header_buf.capacity() as u64)
                .saturating_add(self.pending_scan.capacity() as u64)
                .saturating_add(self.metadata.capacity() as u64);
            try_admit_vec_growth(
                &mut self.row_buf,
                grow,
                other,
                self.effective_max_memory,
                "incremental JPEG MCU strip",
            )?;
            self.row_buf.resize(batch_bytes, 0);
        }

        let sample = format.layout.sample;
        let order = format.layout.order;
        let out = ImgRefMut::new(&mut self.row_buf[..batch_bytes], row_bytes, batch_rows);
        let count = match (sample, order) {
            (SampleType::U8, ChannelOrder::Gray) => reader.read_rows_gray8(out)?,
            (SampleType::U8, ChannelOrder::Rgb) => reader.read_rows_rgb8(out)?,
            (SampleType::U8, ChannelOrder::Rgba) => reader.read_rows_rgba8(out)?,
            (SampleType::U8, ChannelOrder::Rgbx) => reader.read_rows_rgbx8(out)?,
            (SampleType::U8, ChannelOrder::Bgra) => reader.read_rows_bgra8(out)?,
            (SampleType::U8, ChannelOrder::Bgrx) => reader.read_rows_bgrx8(out)?,
            _ => {
                return Err(Error::from(zencodec::UnsupportedOperation::PixelFormat).into());
            }
        };

        self.mcu_row_ready = false;

        if count == 0 {
            self.is_complete = true;
            return Ok(None);
        }

        if self.pending_scan_pos >= self.pending_scan.len() {
            self.pending_scan.clear();
            self.pending_scan_pos = 0;
        } else if self.pending_scan_pos > 32768 {
            self.pending_scan.drain(..self.pending_scan_pos);
            self.pending_scan_pos = 0;
        }

        let y = self.current_row;
        self.current_row += count as u32;

        if reader.is_finished() {
            self.is_complete = true;
        }

        Ok(Some((y, count as u32)))
    }
}

impl IncrementalDecode for JpegIncrementalDecoder {
    type Error = At<CodecError>;
    fn requested(&self) -> &[InputRequest] {
        &self.request
    }
    fn info(&self) -> Option<&ImageInfo> {
        self.info.as_ref()
    }
    fn output_pixel_format(&self) -> Option<PixelFormat> {
        self.format.clone()
    }
    fn step(
        &mut self,
        input: Input<'_>,
        sink: &mut dyn DecodeRowSink,
        pulse: Option<&dyn Pulse>,
    ) -> Result<Progress, StepError<Self::Error>> {
        if self.failed {
            return Err(StepError::Finished);
        }
        if let Some(done) = self.done.clone() {
            return Ok(Progress::new(0, Event::Done(done)));
        }
        self.failed = true; // Poison before native or sink code can unwind.
        let result = self.step_inner(input, sink, pulse);
        if result.is_ok() {
            self.failed = false;
        }
        if result.is_err() {
            self.failed = true;
            self.reader = None;
            self.buffered_output = None;
            self.metadata = crate::container::marker::MetadataCapture::new();
            self.header_buf = Vec::new();
            self.pending_scan = Vec::new();
            self.row_buf = Vec::new();
        }
        result
    }
}

impl JpegIncrementalDecoder {
    fn check(&self, pulse: Option<&dyn Pulse>) -> Result<(), StepError<At<CodecError>>> {
        if let Some(owned) = &self.stop {
            owned.check().map_err(StepError::Cancelled)?;
        }
        if let Some(pulse) = pulse {
            pulse.check().map_err(StepError::Cancelled)?;
        }
        Ok(())
    }

    fn next_accept_len(&self, available: usize) -> usize {
        let mut cap = available.min(4096);
        if let Some(max_in) = self.limits.max_input_bytes {
            // Allow at most remaining + 1 so exceeding max_input_bytes triggers
            // the typed InputSize limit error without copying an unbounded chunk.
            let remaining = max_in.saturating_sub(self.total_input_bytes);
            let allowed = remaining.saturating_add(1).min(cap as u64) as usize;
            cap = allowed;
        }
        cap
    }

    fn step_inner(
        &mut self,
        input: Input<'_>,
        sink: &mut dyn DecodeRowSink,
        pulse: Option<&dyn Pulse>,
    ) -> Result<Progress, StepError<At<CodecError>>> {
        self.check(pulse)?;
        let owned = self.stop.clone();
        let combined = CombinedStop {
            owned: owned.as_ref(),
            call: pulse,
        };
        let mut consumed = 0usize;

        // Buffered whole-file strip delivery directly from `buffered_output` (zero second copy).
        if self.buffered_output.is_none() && self.pending_rows == 0 && !self.is_complete {
            if !self.buffered_whole_file
                && let Some((y, h)) = self
                    .pull_streaming_batch(&combined)
                    .map_err(native_step_error)?
            {
                self.pending_y = y;
                self.pending_rows = h;
                self.pending_offset_rows = 0;
            }
            if self.buffered_output.is_none() && self.pending_rows == 0 && !self.is_complete {
                if input.offset != self.total_input_bytes {
                    return Ok(Progress::new(0, Event::Need));
                }
                let accepted = self.next_accept_len(input.bytes.len());
                let eof = input.total_len == Some(input.offset + accepted as u64);
                self.push_chunk(&input.bytes[..accepted], eof, &combined)
                    .map_err(native_step_error)?;
                consumed = accepted;
                self.request = [InputRequest::AtLeast {
                    offset: self.total_input_bytes,
                    min_len: 1,
                }];
                if !self.buffered_whole_file
                    && let Some((y, h)) = self
                        .pull_streaming_batch(&combined)
                        .map_err(native_step_error)?
                {
                    self.pending_y = y;
                    self.pending_rows = h;
                    self.pending_offset_rows = 0;
                }
                if eof
                    && self.buffered_output.is_none()
                    && self.pending_rows == 0
                    && !self.is_complete
                {
                    return Err(StepError::Codec(
                        Error::internal("native input resumption made no progress at EOF").into(),
                    ));
                }
            }
        }
        self.check(pulse)?;

        // Direct strip delivery from buffered whole-file `PixelBuf`.
        if let Some(ref buf) = self.buffered_output {
            let view = buf.as_view();
            let width = view.width();
            let height = view.height();
            let format = view.format().clone();
            if !self.begun {
                sink.begin(width, height, format.clone())
                    .map_err(StepError::Sink)?;
                self.begun = true;
                self.check(pulse)?;
            }
            let remaining = height.saturating_sub(self.buffered_y);
            if remaining > 0 {
                let rows = remaining.min(sink.max_buffer_rows().get());
                let row_bytes = width as usize * format.layout.bytes_per_pixel();
                let y = self.buffered_y;
                {
                    let mut dst = sink
                        .provide_next_buffer(RowStrip::new(y, width, rows), format)
                        .map_err(StepError::Sink)?;
                    for r in 0..rows {
                        let src_row = view.row(y + r);
                        dst.row_mut(r).copy_from_slice(&src_row[..row_bytes]);
                    }
                }
                self.check(pulse)?;
                self.buffered_y += rows;
                if self.buffered_y >= height {
                    self.buffered_output = None;
                    self.is_complete = true;
                }
                return Ok(Progress::new(
                    consumed,
                    Event::Rows {
                        meta: BatchMeta::final_rows(y),
                        rows,
                    },
                ));
            }
            self.buffered_output = None;
            self.is_complete = true;
        }

        // Streaming MCU-row strip delivery from `self.row_buf`.
        if self.pending_rows > 0 {
            let info = self.info.as_ref().unwrap();
            let (width, height) = (info.width, info.height);
            let format = self.format.as_ref().unwrap().clone();
            if !self.begun {
                sink.begin(width, height, format.clone())
                    .map_err(StepError::Sink)?;
                self.begun = true;
                self.check(pulse)?;
            }
            let rows = self.pending_rows.min(sink.max_buffer_rows().get());
            let stride = width as usize * format.layout.bytes_per_pixel();
            let start = self.pending_offset_rows * stride;
            {
                let mut dst = sink
                    .provide_next_buffer(RowStrip::new(self.pending_y, width, rows), format)
                    .map_err(StepError::Sink)?;
                for r in 0..rows as usize {
                    dst.row_mut(r as u32).copy_from_slice(
                        &self.row_buf[start + r * stride..start + (r + 1) * stride],
                    );
                }
            }
            self.check(pulse)?;
            let y = self.pending_y;
            self.pending_y += rows;
            self.pending_rows -= rows;
            self.pending_offset_rows += rows as usize;
            return Ok(Progress::new(
                consumed,
                Event::Rows {
                    meta: BatchMeta::final_rows(y),
                    rows,
                },
            ));
        }

        if self.is_complete {
            if !self.buffered_whole_file {
                if !self.metadata.done() && !self.is_eof {
                    if consumed == 0 && input.offset == self.total_input_bytes {
                        let accepted = self.next_accept_len(input.bytes.len());
                        self.limits
                            .check_input_size(self.total_input_bytes + accepted as u64)
                            .map_err(|e| StepError::Codec(Error::from(e).into()))?;
                        let other = self.retained_except_metadata();
                        self.metadata
                            .push_bounded(
                                &input.bytes[..accepted],
                                other,
                                self.effective_max_memory,
                            )
                            .map_err(|e| StepError::Codec(e.into()))?;
                        self.total_input_bytes += accepted as u64;
                        consumed = accepted;
                        self.is_eof = input.total_len == Some(self.total_input_bytes);
                        self.request = [InputRequest::at_least(self.total_input_bytes, 1)];
                    }
                    if !self.metadata.done() && !self.is_eof {
                        return Ok(Progress::need(consumed));
                    }
                }
                if self.metadata.done() {
                    let meta_cfg = build_decode_config(
                        &self.config.inner,
                        &self.limits,
                        self.crop_hint,
                        self.orientation,
                        self.policy.as_ref(),
                    );
                    let full = meta_cfg
                        .read_full_info_preserving(
                            self.metadata.bytes(),
                            &crate::decode::PreserveConfig::all(),
                            &combined,
                        )
                        .map_err(|e| native_step_error(e.into()))?;
                    let mut info = to_image_info(&full);
                    if will_auto_orient(self.orientation) {
                        info.orientation = zencodec::Orientation::Identity;
                    }
                    self.info = Some(info);
                }
            }
            sink.finish().map_err(StepError::Sink)?;
            self.check(pulse)?;
            let info = self.info.as_ref().unwrap();
            let format = self.format.as_ref().unwrap().clone();
            let done = zencodec::decode::OutputInfo::full_decode(info.width, info.height, format);
            self.done = Some(done.clone());
            return Ok(Progress::new(consumed, Event::Done(done)));
        }
        Ok(Progress::need(consumed))
    }
}

struct CombinedStop<'a> {
    owned: Option<&'a zencodec::SharedPulse>,
    call: Option<&'a dyn Pulse>,
}

impl enough::Stop for CombinedStop<'_> {
    fn check(&self) -> Result<(), enough::StopReason> {
        if let Some(owned) = self.owned {
            owned.check()?;
        }
        if let Some(call) = self.call {
            call.check()?;
        }
        Ok(())
    }

    fn may_stop(&self) -> bool {
        self.owned.is_some_and(|o| o.may_stop()) || self.call.is_some_and(|c| c.may_stop())
    }
}

fn native_step_error(error: At<CodecError>) -> StepError<At<CodecError>> {
    use zencodec::CodecErrorExt;
    if let Some(zencodec::ErrorCategory::Stopped(reason)) = error.error_category() {
        StepError::Cancelled(reason)
    } else {
        StepError::Codec(error)
    }
}
