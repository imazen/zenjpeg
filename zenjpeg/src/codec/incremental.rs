//! zencodec incremental decode impl: `JpegIncrementalDecoder`.

use alloc::borrow::Cow;
use alloc::vec::Vec;

use enough::Stop;
use imgref::ImgRefMut;
use whereat::At;
use zencodec::Pulse;
use zencodec::decode::{
    BatchMeta, DecodeRowSink, Event, IncrementalDecode, Input, InputRequest, Progress, StepError,
};

use zencodec::{CodecError, ImageInfo, ResourceLimits};
use zenpixels::{ChannelLayout, ChannelType, PixelDescriptor, PixelSlice};

use crate::decode::parser::JpegParser;
use crate::decode::{ChromaUpsampling, DeblockMode, OutputTarget, ScanlineReader};
use crate::error::Error;
use crate::types::JpegMode;

use super::decode::{JpegDecoderConfig, build_decode_config, will_auto_orient};
use super::info::{decode_descriptor, to_image_info};

/// Incremental JPEG decoder implementing [`zencodec::decode::IncrementalDecode`].
///
/// Accepts compressed input bytes in arbitrary chunks without performing any I/O,
/// decoding and yielding scanlines/batches as soon as sufficient data has been pushed.
///
/// Sequential (baseline) JPEGs stream MCU row by MCU row directly from incoming chunks
/// for the explicitly selected nearest-neighbor configuration. Retention is not bounded
/// by this adapter. Progressive JPEGs buffer scans until EOF, then yield
/// MCU row batches.
pub struct JpegIncrementalDecoder {
    config: JpegDecoderConfig,
    limits: ResourceLimits,
    crop_hint: Option<(u32, u32, u32, u32)>,
    orientation: zencodec::OrientationHint,
    policy: Option<zencodec::decode::DecodePolicy>,
    stop: Option<zencodec::SharedPulse>,
    preferred: Vec<PixelDescriptor>,
    descriptor: Option<PixelDescriptor>,
    info: Option<ImageInfo>,

    // Header buffer & parser state
    header_buf: Vec<u8>,
    header_parsed: bool,
    is_progressive: bool,

    // Streaming scanline reader (baseline mode) or buffered reader (progressive mode)
    reader: Option<ScanlineReader<'static>>,
    mcu_height: usize,
    current_row: u32,
    row_buf: aligned_vec::AVec<u8, aligned_vec::ConstAlign<4>>,

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
    metadata: crate::container::marker::MetadataCapture,
    fallback: Option<
        zencodec::decode::Fused<
            zencodec::helpers::WholeFileIncremental<super::decode::JpegDecodeJob>,
        >,
    >,
}

impl JpegIncrementalDecoder {
    /// Create a new incremental decoder from job parameters.
    pub(super) fn new(
        config: JpegDecoderConfig,
        limits: ResourceLimits,
        crop_hint: Option<(u32, u32, u32, u32)>,
        orientation: zencodec::OrientationHint,
        policy: Option<zencodec::decode::DecodePolicy>,
        stop: Option<zencodec::SharedPulse>,
        preferred: &[PixelDescriptor],
    ) -> Self {
        Self {
            config,
            limits,
            crop_hint,
            orientation,
            policy,
            stop,
            preferred: preferred.to_vec(),
            descriptor: None,
            info: None,
            header_buf: Vec::new(),
            header_parsed: false,
            is_progressive: false,
            reader: None,
            mcu_height: 8,
            current_row: 0,
            row_buf: aligned_vec::AVec::new(4),
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
            metadata: crate::container::marker::MetadataCapture::new(),
            fallback: None,
        }
    }

    /// Try parsing the header and initializing the scanline reader.
    fn try_parse_header(&mut self, is_eof: bool) -> Result<bool, At<CodecError>> {
        let max_pixels = self.limits.max_pixels.unwrap_or(u64::MAX);
        let max_memory = self.limits.max_memory_bytes.unwrap_or(u64::MAX);

        let mut parser = match JpegParser::with_strictness(
            &self.header_buf,
            max_pixels,
            max_memory,
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
                let mut info = to_image_info(&header);
                if !matches!(header.mode, JpegMode::Baseline | JpegMode::Progressive)
                    || header.precision != 8
                    || header.num_components > 3
                    || (will_auto_orient(self.orientation)
                        && info.orientation != zencodec::Orientation::Identity)
                {
                    return Err(Error::from(zencodec::UnsupportedOperation::PixelFormat).into());
                }
                if will_auto_orient(self.orientation) {
                    info = info.with_orientation(zencodec::Orientation::Identity);
                }
                let descriptor = decode_descriptor(
                    &self.preferred,
                    &header,
                    self.config.inner.correct_color.as_ref(),
                );
                self.info = Some(info);
                self.descriptor = Some(descriptor);

                if matches!(
                    header.mode,
                    JpegMode::Progressive | JpegMode::ArithmeticProgressive
                ) {
                    self.is_progressive = true;
                    self.header_parsed = true;
                    Ok(true)
                } else {
                    let is_grayscale = parser.num_components == 1;
                    match parser.into_scan_data(is_grayscale) {
                        Ok(scan_data) => {
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
                                self.pending_scan
                                    .extend_from_slice(&self.header_buf[scan_data_start..]);
                            }
                            self.header_buf.clear();
                            self.header_buf.shrink_to_fit();

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

                            self.mcu_height = mcu_height;
                            self.reader = Some(reader);
                            self.header_parsed = true;
                            Ok(true)
                        }
                        Err(e) if !is_eof && e.is_truncated() => Ok(false),
                        Err(e) => Err(e.into()),
                    }
                }
            }
            Err(e) if !is_eof && e.is_truncated() => Ok(false),
            Err(e) => Err(e.into()),
        }
    }
}

impl JpegIncrementalDecoder {
    fn push_chunk(
        &mut self,
        chunk: &[u8],
        is_eof: bool,
        stop: Option<&dyn Stop>,
    ) -> Result<(), At<CodecError>> {
        let stop = stop.unwrap_or(&enough::Unstoppable);
        stop.check().map_err(Error::from)?;
        if let Some(ref stop) = self.stop {
            stop.check().map_err(Error::from)?;
        }

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

        self.metadata.push(chunk)?;

        // Phase 1: Header Accumulation
        if !self.header_parsed {
            self.header_buf
                .try_reserve(chunk.len())
                .map_err(|_| Error::allocation_failed(chunk.len(), "incremental JPEG header"))?;
            self.header_buf.extend_from_slice(chunk);
            if !self.try_parse_header(is_eof)? {
                return Ok(());
            }
        } else if !self.is_progressive {
            self.pending_scan
                .try_reserve(chunk.len())
                .map_err(|_| Error::allocation_failed(chunk.len(), "incremental JPEG scan"))?;
            self.pending_scan.extend_from_slice(chunk);
        } else {
            self.header_buf
                .try_reserve(chunk.len())
                .map_err(|_| Error::allocation_failed(chunk.len(), "incremental JPEG header"))?;
            self.header_buf.extend_from_slice(chunk);
        }

        // Phase 2: Progressive Handling
        if self.is_progressive {
            if !self.is_eof {
                return Ok(());
            }
            if self.reader.is_none() {
                let cfg = build_decode_config(
                    &self.config.inner,
                    &self.limits,
                    self.crop_hint,
                    self.orientation,
                    self.policy.as_ref(),
                );
                let data = core::mem::take(&mut self.header_buf);
                let reader = cfg.scanline_reader_cow(Cow::Owned(data))?;
                self.mcu_height = reader.luma_rows_per_mcu();
                self.reader = Some(reader);
                self.mcu_row_ready = true;
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

    fn pull_batch(
        &mut self,
        stop: &dyn enough::Stop,
    ) -> Result<Option<(u32, PixelSlice<'_>)>, At<CodecError>> {
        if let Some(ref stop) = self.stop {
            stop.check().map_err(Error::from)?;
        }

        if self.is_complete {
            return Ok(None);
        }

        // Try decoding next MCU row if not already ready
        if !self.mcu_row_ready
            && !self.is_progressive
            && let Some(ref mut reader) = self.reader
            && !reader.is_finished()
        {
            let slice = &self.pending_scan[self.pending_scan_pos..];
            if let Some(bytes) = reader.decode_mcu_row_slice(slice, self.is_eof, stop)? {
                self.pending_scan_pos += bytes;
                self.mcu_row_ready = true;
            }
        }

        let is_ready = self.mcu_row_ready || (self.is_progressive && self.reader.is_some());
        if !is_ready {
            if let Some(ref reader) = self.reader
                && reader.is_finished()
            {
                self.is_complete = true;
            }
            return Ok(None);
        }

        let reader = match self.reader.as_mut() {
            Some(r) => r,
            None => return Ok(None),
        };

        if reader.is_finished() {
            self.is_complete = true;
            return Ok(None);
        }

        let width = reader.width() as usize;
        let descriptor = match self.descriptor {
            Some(d) => d,
            None => return Err(Error::internal("incremental decode: descriptor missing").into()),
        };

        let bpp = descriptor.bytes_per_pixel();
        let row_bytes = width * bpp;
        let batch_rows = self.mcu_height;
        let batch_bytes = row_bytes * batch_rows;
        if self.row_buf.len() < batch_bytes {
            self.row_buf.resize(batch_bytes, 0);
        }

        let ch_type = descriptor.channel_type();
        let ch_layout = descriptor.layout();

        let count = match (ch_type, ch_layout) {
            (ChannelType::U8, ChannelLayout::Gray) => {
                let out = ImgRefMut::new(&mut self.row_buf[..batch_bytes], row_bytes, batch_rows);
                reader.read_rows_gray8(out)?
            }
            (ChannelType::U8, ChannelLayout::Rgb) => {
                let out = ImgRefMut::new(&mut self.row_buf[..batch_bytes], row_bytes, batch_rows);
                reader.read_rows_rgb8(out)?
            }
            (ChannelType::U8, ChannelLayout::Rgba) => {
                let out = ImgRefMut::new(&mut self.row_buf[..batch_bytes], row_bytes, batch_rows);
                if descriptor.alpha() == Some(zenpixels::AlphaMode::Undefined) {
                    reader.read_rows_rgbx8(out)?
                } else {
                    reader.read_rows_rgba8(out)?
                }
            }
            (ChannelType::U8, ChannelLayout::Bgra) => {
                let out = ImgRefMut::new(&mut self.row_buf[..batch_bytes], row_bytes, batch_rows);
                if descriptor.alpha() == Some(zenpixels::AlphaMode::Undefined) {
                    reader.read_rows_bgrx8(out)?
                } else {
                    reader.read_rows_bgra8(out)?
                }
            }
            (ChannelType::F32, ChannelLayout::Gray) => {
                let float_count = width * batch_rows;
                let float_bytes = float_count * 4;
                self.row_buf.resize(float_bytes, 0);
                let float_slice: &mut [f32] = bytemuck::cast_slice_mut(&mut self.row_buf);
                let f_out = ImgRefMut::new(float_slice, width, batch_rows);
                reader.read_rows_gray_f32(f_out)?
            }
            (ChannelType::F32, ChannelLayout::Rgb | ChannelLayout::Rgba) => {
                let channels = 4;
                let float_count = width * channels * batch_rows;
                let float_bytes = float_count * 4;
                self.row_buf.resize(float_bytes, 0);
                let float_slice: &mut [f32] = bytemuck::cast_slice_mut(&mut self.row_buf);
                let f_out = ImgRefMut::new(float_slice, width * channels, batch_rows);
                reader.read_rows_rgba_f32(f_out)?
            }
            _ => {
                return Err(Error::from(zencodec::UnsupportedOperation::PixelFormat).into());
            }
        };

        self.mcu_row_ready = false;

        if count == 0 {
            self.is_complete = true;
            return Ok(None);
        }

        // Compact pending_scan buffer
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

        let actual_bytes = row_bytes * count;
        let slice = PixelSlice::new(
            &self.row_buf[..actual_bytes],
            width as u32,
            count as u32,
            row_bytes,
            descriptor,
        )
        .map_err(|_| Error::internal("incremental decode: pixel slice construction failed"))?;

        Ok(Some((y, slice)))
    }
}

impl IncrementalDecode for JpegIncrementalDecoder {
    type Error = At<CodecError>;
    fn requested(&self) -> &[InputRequest] {
        self.fallback
            .as_ref()
            .map_or(&self.request, |d| d.requested())
    }
    fn info(&self) -> Option<&ImageInfo> {
        self.fallback
            .as_ref()
            .map_or_else(|| self.info.as_ref(), |d| d.info())
    }
    fn output_descriptor(&self) -> Option<PixelDescriptor> {
        self.fallback
            .as_ref()
            .map_or(self.descriptor, |d| d.output_descriptor())
    }
    fn step(
        &mut self,
        input: Input<'_>,
        sink: &mut dyn DecodeRowSink,
        pulse: Option<&dyn Pulse>,
    ) -> Result<Progress, StepError<Self::Error>> {
        if let Some(fallback) = &mut self.fallback {
            return fallback.step(input, sink, pulse);
        }
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
            self.metadata = crate::container::marker::MetadataCapture::new();
            self.header_buf = Vec::new();
            self.pending_scan = Vec::new();
            self.row_buf = aligned_vec::AVec::new(4);
        }
        result
    }
}
impl JpegIncrementalDecoder {
    pub(super) fn buffered(
        job: super::decode::JpegDecodeJob,
        preferred: &[PixelDescriptor],
    ) -> Self {
        let mut decoder = Self::new(
            JpegDecoderConfig::new(),
            ResourceLimits::none(),
            None,
            zencodec::OrientationHint::Preserve,
            None,
            None,
            preferred,
        );
        decoder.fallback = Some(zencodec::decode::Fused::new(
            zencodec::helpers::WholeFileIncremental::new(job, preferred),
        ));
        decoder
    }
    fn check(&self, pulse: Option<&dyn Pulse>) -> Result<(), StepError<At<CodecError>>> {
        if let Some(owned) = &self.stop {
            owned.check().map_err(StepError::Cancelled)?;
        }
        if let Some(pulse) = pulse {
            pulse.check().map_err(StepError::Cancelled)?;
        }
        Ok(())
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
        let mut consumed = 0;
        if self.pending_rows == 0 && !self.is_complete {
            if let Some((y, rows)) = self.pull_batch(&combined).map_err(native_step_error)? {
                let height = rows.rows();
                self.pending_y = y;
                self.pending_rows = height;
            }
            if self.pending_rows == 0 && !self.is_complete {
                if input.offset != self.total_input_bytes {
                    return Ok(Progress::new(0, Event::Need));
                }
                let accepted = input.bytes.len().min(4096);
                let eof = input.total_len == Some(input.offset + accepted as u64);
                self.push_chunk(&input.bytes[..accepted], eof, Some(&combined))
                    .map_err(native_step_error)?;
                consumed = accepted;
                self.request = [InputRequest::AtLeast {
                    offset: self.total_input_bytes,
                    min_len: 1,
                }];
                if let Some((y, rows)) = self.pull_batch(&combined).map_err(native_step_error)? {
                    let height = rows.rows();
                    self.pending_y = y;
                    self.pending_rows = height;
                }
                if eof && self.pending_rows == 0 && !self.is_complete {
                    return Err(StepError::Codec(
                        Error::internal("native input resumption made no progress at EOF").into(),
                    ));
                }
            }
        }
        self.check(pulse)?;
        if self.pending_rows > 0 {
            let info = self.info.as_ref().unwrap();
            let (width, height, desc) = (info.width, info.height, self.descriptor.unwrap());
            if !self.begun {
                sink.begin(width, height, desc).map_err(StepError::Sink)?;
                self.begun = true;
                self.check(pulse)?;
            }
            let rows = self.pending_rows.min(sink.max_buffer_rows().get());
            let stride = width as usize * desc.bytes_per_pixel();
            let start = (self.pending_y % self.mcu_height as u32) as usize * stride;
            {
                let mut dst = sink
                    .provide_next_buffer(self.pending_y, rows, width, desc)
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
            return Ok(Progress::new(
                consumed,
                Event::Rows {
                    meta: BatchMeta::final_rows(y),
                    rows,
                },
            ));
        }
        if self.is_complete {
            if !self.metadata.done() && !self.is_eof {
                if consumed == 0 && input.offset == self.total_input_bytes {
                    let accepted = input.bytes.len().min(4096);
                    self.limits
                        .check_input_size(self.total_input_bytes + accepted as u64)
                        .map_err(|e| StepError::Codec(Error::from(e).into()))?;
                    self.metadata
                        .push(&input.bytes[..accepted])
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
                let full = self
                    .config
                    .inner
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
            sink.finish().map_err(StepError::Sink)?;
            self.check(pulse)?;
            let info = self.info.as_ref().unwrap();
            let done = zencodec::decode::OutputInfo::full_decode(
                info.width,
                info.height,
                self.descriptor.unwrap(),
            );
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
}
fn native_step_error(error: At<CodecError>) -> StepError<At<CodecError>> {
    use zencodec::CodecErrorExt;
    if let Some(zencodec::ErrorCategory::Stopped(reason)) = error.error_category() {
        StepError::Cancelled(reason)
    } else {
        StepError::Codec(error)
    }
}
