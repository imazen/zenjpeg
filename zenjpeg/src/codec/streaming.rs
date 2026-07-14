//! zencodec streaming decode impl: `JpegStreamingDecoder`.

use whereat::At;
use zencodec::{CodecError, ImageInfo};
use zenpixels::{PixelDescriptor, PixelSlice};

use crate::error::Error;

// ── StreamingDecode ─────────────────────────────────────────────────────────

/// Streaming JPEG decoder implementing [`zencodec::decode::StreamingDecode`].
///
/// Wraps zenjpeg's `ScanlineReader` to yield scanline batches via `next_batch()`.
/// Each batch contains one MCU-row worth of decoded pixels (8 or 16 rows).
pub struct JpegStreamingDecoder<'a> {
    pub(super) reader: crate::decode::ScanlineReader<'a>,
    pub(super) info: ImageInfo,
    pub(super) descriptor: PixelDescriptor,
    /// Reusable row buffer for decoded pixel data (sized for MCU-row batches).
    /// 4-byte aligned so bytemuck casts to &mut [f32] are safe.
    pub(super) row_buf: aligned_vec::AVec<u8, aligned_vec::ConstAlign<4>>,
    pub(super) current_row: u32,
    /// MCU height in pixels (8 or 16 depending on subsampling).
    pub(super) mcu_height: u32,
    /// Cooperative cancellation token.
    pub(super) stop: Option<zencodec::StopToken>,
}

impl zencodec::decode::StreamingDecode for JpegStreamingDecoder<'_> {
    type Error = At<CodecError>;

    fn next_batch(&mut self) -> Result<Option<(u32, PixelSlice<'_>)>, Self::Error> {
        {
            use imgref::ImgRefMut;
            use zenpixels::{ChannelLayout, ChannelType};

            // Check cooperative cancellation before doing work. `StopReason` is
            // not a `core::error::Error`, so route it through the native `Error`
            // (→ `ErrorKind::Cancelled`, which categorizes as Cancelled/TimedOut)
            // before the bridge lifts it into the envelope.
            if let Some(ref stop) = self.stop {
                use enough::Stop;
                stop.check().map_err(Error::from)?;
            }

            if self.reader.is_finished() {
                return Ok(None);
            }

            let width = self.reader.width() as usize;
            let bpp = self.descriptor.bytes_per_pixel();
            let row_bytes = width * bpp;
            // Allocate for MCU-row batch instead of single row
            let batch_rows = self.mcu_height as usize;
            let batch_bytes = row_bytes * batch_rows;
            self.row_buf.resize(batch_bytes, 0);

            let ch_type = self.descriptor.channel_type();
            let ch_layout = self.descriptor.layout();

            let count = match (ch_type, ch_layout) {
                (ChannelType::U8, ChannelLayout::Gray) => {
                    let out =
                        ImgRefMut::new(&mut self.row_buf[..batch_bytes], row_bytes, batch_rows);
                    self.reader.read_rows_gray8(out)?
                }
                (ChannelType::U8, ChannelLayout::Rgb) => {
                    let out =
                        ImgRefMut::new(&mut self.row_buf[..batch_bytes], row_bytes, batch_rows);
                    self.reader.read_rows_rgb8(out)?
                }
                (ChannelType::U8, ChannelLayout::Rgba) => {
                    if self.descriptor.alpha() == Some(zenpixels::AlphaMode::Undefined) {
                        let out =
                            ImgRefMut::new(&mut self.row_buf[..batch_bytes], row_bytes, batch_rows);
                        self.reader.read_rows_rgbx8(out)?
                    } else {
                        let out =
                            ImgRefMut::new(&mut self.row_buf[..batch_bytes], row_bytes, batch_rows);
                        self.reader.read_rows_rgba8(out)?
                    }
                }
                (ChannelType::U8, ChannelLayout::Bgra) => {
                    if self.descriptor.alpha() == Some(zenpixels::AlphaMode::Undefined) {
                        let out =
                            ImgRefMut::new(&mut self.row_buf[..batch_bytes], row_bytes, batch_rows);
                        self.reader.read_rows_bgrx8(out)?
                    } else {
                        let out =
                            ImgRefMut::new(&mut self.row_buf[..batch_bytes], row_bytes, batch_rows);
                        self.reader.read_rows_bgra8(out)?
                    }
                }
                (ChannelType::F32, ChannelLayout::Gray) => {
                    let float_count = width * batch_rows;
                    let float_bytes = float_count * 4;
                    self.row_buf.resize(float_bytes, 0);
                    let float_slice: &mut [f32] = bytemuck::cast_slice_mut(&mut self.row_buf);
                    let f_out = ImgRefMut::new(float_slice, width, batch_rows);
                    self.reader.read_rows_gray_f32(f_out)?
                }
                (ChannelType::F32, ChannelLayout::Rgb | ChannelLayout::Rgba) => {
                    // read_rows_rgba_f32 always writes 4 channels
                    let channels = 4;
                    let float_count = width * channels * batch_rows;
                    let float_bytes = float_count * 4;
                    self.row_buf.resize(float_bytes, 0);
                    let float_slice: &mut [f32] = bytemuck::cast_slice_mut(&mut self.row_buf);
                    let f_out = ImgRefMut::new(float_slice, width * channels, batch_rows);
                    self.reader.read_rows_rgba_f32(f_out)?
                }
                _ => {
                    // A caller-requested pixel format/descriptor this codec
                    // doesn't negotiate — the dedicated
                    // `UnsupportedOperation::PixelFormat` axis, not the
                    // generic string-payload `unsupported_feature` (caterr
                    // Pattern-B follow-up finding #1 investigation). Two
                    // explicit hops: `UnsupportedOperation` → `Error` (via
                    // `From<zencodec::UnsupportedOperation>`) → `At<CodecError>`
                    // (via `From<Error>`) — `Into` doesn't chain transitively.
                    return Err(Error::from(zencodec::UnsupportedOperation::PixelFormat).into());
                }
            };

            if count == 0 {
                return Ok(None);
            }

            let y = self.current_row;
            self.current_row += count as u32;

            let actual_bytes = row_bytes * count;
            let stride = row_bytes;
            let slice = PixelSlice::new(
                &self.row_buf[..actual_bytes],
                width as u32,
                count as u32,
                stride,
                self.descriptor,
            )
            .map_err(|_| Error::internal("streaming decode: pixel slice construction failed"))?;

            Ok(Some((y, slice)))
        }
    }

    fn info(&self) -> &ImageInfo {
        &self.info
    }
}
