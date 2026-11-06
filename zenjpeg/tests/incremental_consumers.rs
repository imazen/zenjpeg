#![cfg(feature = "zencodec")]

use std::borrow::Cow;
use std::io::Cursor;

use zencodec::decode::{
    Decode, DecodeJob, DecodeRowSink, DecoderConfig, IncrementalDecode, OutputInfo, SinkError,
};
use zencodec::encode::{EncodeJob, Encoder, EncoderConfig};
use zencodec::helpers::{PumpError, pump_read_to_sink};
use zenjpeg::{JpegDecoderConfig, JpegEncoderConfig};
use zenpixels::{PixelDescriptor, PixelSlice, PixelSliceMut};

struct CollectSink {
    width: u32,
    height: u32,
    buf: Vec<u8>,
    began: bool,
    finished: bool,
}

impl CollectSink {
    fn new() -> Self {
        Self {
            width: 0,
            height: 0,
            buf: Vec::new(),
            began: false,
            finished: false,
        }
    }
}

impl DecodeRowSink for CollectSink {
    fn begin(
        &mut self,
        width: u32,
        height: u32,
        descriptor: PixelDescriptor,
    ) -> Result<(), SinkError> {
        self.width = width;
        self.height = height;
        let bpp = descriptor.bytes_per_pixel();
        self.buf.resize(width as usize * height as usize * bpp, 0);
        self.began = true;
        Ok(())
    }

    fn provide_next_buffer(
        &mut self,
        y: u32,
        height: u32,
        width: u32,
        descriptor: PixelDescriptor,
    ) -> Result<PixelSliceMut<'_>, SinkError> {
        assert!(self.began);
        let bpp = descriptor.bytes_per_pixel();
        let stride = width as usize * bpp;
        let start = y as usize * stride;
        let end = start + height as usize * stride;
        let slice = &mut self.buf[start..end];
        PixelSliceMut::new(slice, width, height, stride, descriptor)
            .map_err(|e| format!("{e:?}").into())
    }

    fn finish(&mut self) -> Result<(), SinkError> {
        self.finished = true;
        Ok(())
    }
}

fn generate_test_jpeg(width: u32, height: u32) -> Vec<u8> {
    let mut raw_pixels = vec![0u8; (width * height * 3) as usize];
    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 3) as usize;
            raw_pixels[idx] = (x ^ y) as u8;
            raw_pixels[idx + 1] = ((x * 2 + y) % 256) as u8;
            raw_pixels[idx + 2] = ((y * 3 + x) % 256) as u8;
        }
    }

    let slice = PixelSlice::new(
        &raw_pixels,
        width,
        height,
        width as usize * 3,
        PixelDescriptor::RGB8_SRGB,
    )
    .expect("valid slice");

    let encoder = JpegEncoderConfig::new();
    let job = encoder.job();
    let enc = job.encoder().expect("encoder init");
    enc.encode(slice).expect("encode succeeds").into_vec()
}

#[test]
fn test_zenjpeg_stdio_consumer_pixel_parity() {
    let width = 64u32;
    let height = 64u32;
    let jpeg_bytes = generate_test_jpeg(width, height);

    // 1. Monolithic decode reference
    let dec_cfg = JpegDecoderConfig::new();
    let dec = dec_cfg
        .job()
        .decoder(Cow::Borrowed(&jpeg_bytes), &[])
        .expect("decoder init");
    let mono_output = dec.decode().expect("monolithic decode");
    let mono_pixels = mono_output.pixels().as_strided_bytes().to_vec();

    // 2. Stdio incremental consumer via pump_read_to_sink
    let mut inc_decoder = JpegDecoderConfig::new()
        .job()
        .into_incremental_decoder(&[])
        .expect("incremental decoder init");
    let mut sink = CollectSink::new();
    let reader = Cursor::new(&jpeg_bytes);

    let info = pump_read_to_sink(reader, &mut inc_decoder, &mut sink, None, 256)
        .expect("stdio pump succeeds");

    assert_eq!(info.width, width);
    assert_eq!(info.height, height);
    assert!(sink.finished);
    assert_eq!(sink.buf.len(), mono_pixels.len());

    // 3. Pixel parity: exactly 0 bits difference
    assert_eq!(
        sink.buf, mono_pixels,
        "Stdio incremental consumer produced different pixels than monolithic decode!"
    );
}

// ── Tokio Async Consumer Test ───────────────────────────────────────────────

async fn pump_async_read_to_sink<R, D>(
    mut reader: R,
    decoder: &mut D,
    sink: &mut (dyn DecodeRowSink + '_),
    buffer_size: usize,
) -> Result<OutputInfo, PumpError<D::Error>>
where
    R: tokio::io::AsyncRead + Unpin,
    D: IncrementalDecode,
{
    use tokio::io::AsyncReadExt;

    let buffer_size = buffer_size.max(512);
    let mut buf = vec![0u8; buffer_size];
    let mut sink_begun = false;
    let mut chosen_desc: Option<PixelDescriptor> = None;
    let mut total_w = 0u32;
    let mut total_h = 0u32;

    let mut drain_batches =
        |dec: &mut D, dst_sink: &mut (dyn DecodeRowSink + '_)| -> Result<(), PumpError<D::Error>> {
            if !sink_begun {
                let info_h = dec.info().map(|i| i.height);
                if let Some((meta, slice)) = dec.pull_batch_meta().map_err(PumpError::Codec)? {
                    let desc = slice.descriptor();
                    chosen_desc = Some(desc);
                    let w = slice.width();
                    let h = info_h.unwrap_or(slice.rows() as u32);
                    total_w = w;
                    total_h = h;
                    dst_sink
                        .begin(w, h, desc)
                        .map_err(|e| PumpError::Codec(D::wrap_sink_error(e)))?;
                    sink_begun = true;

                    let batch_rows = slice.rows() as u32;
                    let mut dst = dst_sink
                        .provide_next_buffer(meta.y, batch_rows, w, desc)
                        .map_err(|e| PumpError::Codec(D::wrap_sink_error(e)))?;
                    for row_idx in 0..batch_rows {
                        dst.row_mut(row_idx).copy_from_slice(slice.row(row_idx));
                    }
                }
            }

            if sink_begun {
                while let Some(_rows) = dec.pull_batch_into(dst_sink).map_err(PumpError::Codec)? {}
            }
            Ok(())
        };

    loop {
        drain_batches(decoder, sink)?;

        if decoder.is_complete() {
            if sink_begun {
                sink.finish()
                    .map_err(|e| PumpError::Codec(D::wrap_sink_error(e)))?;
            }
            let info = decoder.info().ok_or(PumpError::UnexpectedEof)?;
            let desc = chosen_desc.unwrap_or(PixelDescriptor::RGB8_SRGB);
            return Ok(OutputInfo::full_decode(info.width, info.height, desc));
        }

        let n = reader.read(&mut buf).await.map_err(PumpError::Io)?;
        let is_eof = n == 0;

        let _outcome = decoder
            .push_chunk(&buf[..n], is_eof, None)
            .map_err(PumpError::Codec)?;

        drain_batches(decoder, sink)?;

        if is_eof {
            drain_batches(decoder, sink)?;
            if !decoder.is_complete() {
                return Err(PumpError::UnexpectedEof);
            }
            if sink_begun {
                sink.finish()
                    .map_err(|e| PumpError::Codec(D::wrap_sink_error(e)))?;
            }
            let desc = chosen_desc.unwrap_or(PixelDescriptor::RGB8_SRGB);
            let w = if total_w > 0 {
                total_w
            } else {
                decoder.info().map(|i| i.width).unwrap_or(0)
            };
            let h = if total_h > 0 {
                total_h
            } else {
                decoder.info().map(|i| i.height).unwrap_or(0)
            };
            return Ok(OutputInfo::full_decode(w, h, desc));
        }
    }
}

#[tokio::test]
async fn test_zenjpeg_tokio_async_consumer_pixel_parity() {
    let width = 64u32;
    let height = 64u32;
    let jpeg_bytes = generate_test_jpeg(width, height);

    // 1. Monolithic decode reference
    let dec_cfg = JpegDecoderConfig::new();
    let dec = dec_cfg
        .job()
        .decoder(Cow::Borrowed(&jpeg_bytes), &[])
        .expect("decoder init");
    let mono_output = dec.decode().expect("monolithic decode");
    let mono_pixels = mono_output.pixels().as_strided_bytes().to_vec();

    // 2. Tokio async incremental consumer
    let mut inc_decoder = JpegDecoderConfig::new()
        .job()
        .into_incremental_decoder(&[])
        .expect("incremental decoder init");
    let mut sink = CollectSink::new();
    let reader = tokio::io::BufReader::new(&jpeg_bytes[..]);

    let info = pump_async_read_to_sink(reader, &mut inc_decoder, &mut sink, 128)
        .await
        .expect("tokio async pump succeeds");

    assert_eq!(info.width, width);
    assert_eq!(info.height, height);
    assert!(sink.finished);
    assert_eq!(sink.buf.len(), mono_pixels.len());

    // 3. Exact pixel match
    assert_eq!(
        sink.buf, mono_pixels,
        "Tokio async incremental consumer produced different pixels than monolithic decode!"
    );
}
