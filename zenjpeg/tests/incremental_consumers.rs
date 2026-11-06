#![cfg(feature = "zencodec")]

use zencodec::{
    decode::{Decode, DecodeJob, DecoderConfig},
    encode::EncoderConfig,
};
use zenjpeg::{JpegDecoderConfig, JpegEncoderConfig};

fn generate_test_jpeg(width: u32, height: u32) -> Vec<u8> {
    let pixels: Vec<u8> = (0..width * height * 3).map(|i| (i % 256) as u8).collect();
    let slice = zenpixels::PixelSlice::new(
        &pixels,
        width,
        height,
        width as usize * 3,
        zenpixels::PixelDescriptor::RGB8_SRGB,
    )
    .unwrap();
    JpegEncoderConfig::new().encode(slice).unwrap().into_vec()
}
fn expected(bytes: &[u8]) -> Vec<u8> {
    let output = JpegDecoderConfig::new()
        .job()
        .decoder(std::borrow::Cow::Borrowed(bytes), &[])
        .unwrap()
        .decode()
        .unwrap();
    let pixels = output.pixels();
    (0..pixels.rows())
        .flat_map(|y| pixels.row(y).to_vec())
        .collect()
}
#[test]
fn test_zenjpeg_stdio_consumer_pixel_parity() {
    let bytes = generate_test_jpeg(64, 64);
    let mut decoder = JpegDecoderConfig::new()
        .job()
        .into_incremental_decoder(&[])
        .unwrap();
    let mut sink = zencodec_io::CollectSink::new();
    let info = zencodec_io::pump_read_to_sink(
        std::io::Cursor::new(&bytes),
        &mut decoder,
        &mut sink,
        None,
        256,
        bytes.len().max(4096),
    )
    .unwrap();
    assert_eq!((info.width, info.height), (64, 64));
    assert!(sink.finished());
    let reference = expected(&bytes);
    assert_eq!(sink.bytes().len(), reference.len());
    assert_eq!(sink.bytes(), reference);
}
#[tokio::test]
async fn test_zenjpeg_tokio_async_consumer_pixel_parity() {
    let bytes = generate_test_jpeg(64, 64);
    let reference = expected(&bytes);
    let decoder = JpegDecoderConfig::new()
        .job()
        .into_incremental_decoder(&[])
        .unwrap();
    let mut source = zencodec_io::tokio_io::AsyncReadSource::new(bytes.as_slice());
    let (info, sink) = zencodec_io::tokio_io::decode_async(
        decoder,
        zencodec_io::CollectSink::new(),
        &mut source,
        zencodec_io::tokio_io::Inline,
        zencodec_io::DriveOptions::new(0, 512),
        None,
    )
    .await
    .unwrap();
    assert_eq!((info.width, info.height), (64, 64));
    assert!(sink.finished());
    assert_eq!(sink.bytes().len(), reference.len());
    assert_eq!(sink.bytes(), reference);
}
