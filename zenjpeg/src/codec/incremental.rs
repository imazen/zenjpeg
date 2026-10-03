//! zencodec incremental decode impl: `JpegIncrementalDecoder`.

use alloc::borrow::Cow;
use alloc::vec::Vec;

use enough::Stop;
use imgref::ImgRefMut;
use whereat::At;
use zencodec::decode::{IncrementalDecode, PushOutcome, PushStatus};
use zencodec::{CodecError, ImageInfo, ResourceLimits};
use zenpixels::{ChannelLayout, ChannelType, PixelDescriptor, PixelSlice};

use crate::decode::parser::JpegParser;
use crate::decode::{ChromaUpsampling, DeblockMode, IdctMethod, OutputTarget, ScanlineReader};
use crate::error::Error;
use crate::types::JpegMode;

use super::decode::{build_decode_config, will_auto_orient, JpegDecoderConfig};
use super::info::{decode_descriptor, to_image_info};

/// Incremental JPEG decoder implementing [`zencodec::decode::IncrementalDecode`].
///
/// Accepts compressed input bytes in arbitrary chunks without performing any I/O,
/// decoding and yielding scanlines/batches as soon as sufficient data has been pushed.
///
/// Sequential (baseline) JPEGs stream MCU row by MCU row directly from incoming chunks
/// with minimal memory overhead. Progressive JPEGs buffer scans until EOF, then yield
/// MCU row batches.
pub struct JpegIncrementalDecoder {
    config: JpegDecoderConfig,
    limits: ResourceLimits,
    crop_hint: Option<(u32, u32, u32, u32)>,
    orientation: zencodec::OrientationHint,
    policy: Option<zencodec::decode::DecodePolicy>,
    stop: Option<zencodec::StopToken>,
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
}

impl JpegIncrementalDecoder {
    /// Create a new incremental decoder from job parameters.
    pub(super) fn new(
        config: JpegDecoderConfig,
        limits: ResourceLimits,
        crop_hint: Option<(u32, u32, u32, u32)>,
        orientation: zencodec::OrientationHint,
        policy: Option<zencodec::decode::DecodePolicy>,
        stop: Option<zencodec::StopToken>,
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
                                IdctMethod::Libjpeg,
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

impl IncrementalDecode for JpegIncrementalDecoder {
    type Error = At<CodecError>;

    fn push_chunk(
        &mut self,
        chunk: &[u8],
        is_eof: bool,
        stop: Option<&dyn Stop>,
    ) -> Result<PushOutcome, Self::Error> {
        if let Some(stop) = stop {
            stop.check().map_err(Error::from)?;
        }
        if let Some(ref stop) = self.stop {
            stop.check().map_err(Error::from)?;
        }

        if self.is_complete {
            return Ok(PushOutcome::new(0, PushStatus::Complete));
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

        // Phase 1: Header Accumulation
        if !self.header_parsed {
            self.header_buf.extend_from_slice(chunk);
            if !self.try_parse_header(is_eof)? {
                return Ok(PushOutcome::new(chunk.len(), PushStatus::NeedMoreInput));
            }
        } else if !self.is_progressive {
            self.pending_scan.extend_from_slice(chunk);
        } else {
            self.header_buf.extend_from_slice(chunk);
        }

        // Phase 2: Progressive Handling
        if self.is_progressive {
            if !self.is_eof {
                return Ok(PushOutcome::new(chunk.len(), PushStatus::NeedMoreInput));
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
            return Ok(PushOutcome::new(chunk.len(), PushStatus::OutputReady));
        }

        // Phase 3: Baseline Streaming
        if !self.mcu_row_ready {
            if let Some(ref mut reader) = self.reader {
                let slice = &self.pending_scan[self.pending_scan_pos..];
                if let Some(bytes) = reader.decode_mcu_row_slice(slice, self.is_eof)? {
                    self.pending_scan_pos += bytes;
                    self.mcu_row_ready = true;
                }
            }
        }

        let status = if self.mcu_row_ready {
            PushStatus::OutputReady
        } else if let Some(ref reader) = self.reader {
            if reader.is_finished() {
                self.is_complete = true;
                PushStatus::Complete
            } else {
                PushStatus::NeedMoreInput
            }
        } else {
            PushStatus::NeedMoreInput
        };

        Ok(PushOutcome::new(chunk.len(), status))
    }

    fn pull_batch(&mut self) -> Result<Option<(u32, PixelSlice<'_>)>, Self::Error> {
        if let Some(ref stop) = self.stop {
            stop.check().map_err(Error::from)?;
        }

        if self.is_complete {
            return Ok(None);
        }

        // Try decoding next MCU row if not already ready
        if !self.mcu_row_ready && !self.is_progressive {
            if let Some(ref mut reader) = self.reader {
                if !reader.is_finished() {
                    let slice = &self.pending_scan[self.pending_scan_pos..];
                    if let Some(bytes) = reader.decode_mcu_row_slice(slice, self.is_eof)? {
                        self.pending_scan_pos += bytes;
                        self.mcu_row_ready = true;
                    }
                }
            }
        }

        let is_ready = self.mcu_row_ready || (self.is_progressive && self.reader.is_some());
        if !is_ready {
            if let Some(ref reader) = self.reader {
                if reader.is_finished() {
                    self.is_complete = true;
                }
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

    fn info(&self) -> Option<&ImageInfo> {
        self.info.as_ref()
    }

    fn is_complete(&self) -> bool {
        self.is_complete
    }
}
