//! Reader construction fits an ordinary thread stack in unoptimized builds.
//!
//! Consumers compile zenjpeg at opt-level 0 in their debug and test profiles
//! and decode on 2 MiB threads: test harness threads, tokio and rayon pools.
//! Construction once took about 2.1 MB of stack there (two inline Huffman
//! table sets moved by value through every `Result` from the controlled
//! decoder down to the scanline reader), so controlled, streaming and
//! one-shot decodes overflowed. Each decode here runs on a thread whose stack
//! is `ZENJPEG_DECODE_STACK` bytes (default 1 MiB, half the platform default).
//! `just stack-demand-check` runs this file unoptimized; the regular suites
//! run it optimized.
#![cfg(all(feature = "zencodec", feature = "controlled-decode"))]

use std::borrow::Cow;
use std::sync::Arc;

use zencodec::decode::{Decode, DecodeJob, DecoderConfig, StreamingDecode};
use zencodec::encode::{EncodeJob, Encoder, EncoderConfig};
use zencodec::operation::PolicySet;
use zencodec::storage::StorageBudget;
use zencodec_io::{CollectSink, PolicyDecoder, drive_from_slice};
use zenjpeg::encoder::ChromaSubsampling;
use zenjpeg::{JpegDecoderConfig, JpegEncoderConfig};
use zenpixels::{PixelDescriptor, PixelSlice};

fn stack_bytes() -> usize {
    std::env::var("ZENJPEG_DECODE_STACK")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1 << 20)
}

/// Runs `f` on a thread with the test stack. Overflowing it aborts the process.
fn on_test_stack<T: Send + 'static>(what: String, f: impl FnOnce() -> T + Send + 'static) -> T {
    std::thread::Builder::new()
        .name(what.clone())
        .stack_size(stack_bytes())
        .spawn(f)
        .unwrap()
        .join()
        .unwrap_or_else(|_| panic!("{what} panicked"))
}

fn packed_rows(pixels: &PixelSlice<'_>) -> Vec<u8> {
    let row = pixels.width() as usize * pixels.descriptor().bytes_per_pixel();
    (0..pixels.rows())
        .flat_map(|y| pixels.row(y)[..row].to_vec())
        .collect()
}

fn fixtures() -> Vec<(String, Arc<Vec<u8>>)> {
    let (w, h) = (67u32, 45u32);
    let rgb: Vec<u8> = (0..w * h * 3)
        .map(|i| ((i * 37 + i / 7) % 251) as u8)
        .collect();
    let gray: Vec<u8> = (0..w * h).map(|i| ((i * 13 + i / 5) % 253) as u8).collect();
    let mut out = Vec::new();
    for (name, config) in [
        (
            "baseline 4:2:0",
            JpegEncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter),
        ),
        (
            "baseline 4:4:4",
            JpegEncoderConfig::ycbcr(85.0, ChromaSubsampling::None),
        ),
        (
            "progressive 4:2:0",
            JpegEncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter).with_progressive(true),
        ),
    ] {
        let pixels =
            PixelSlice::new(&rgb, w, h, w as usize * 3, PixelDescriptor::RGB8_SRGB).unwrap();
        let encoded = config.job().encoder().unwrap().encode(pixels).unwrap();
        out.push((format!("{name} RGB"), Arc::new(encoded.into_vec())));
    }
    let pixels = PixelSlice::new(&gray, w, h, w as usize, PixelDescriptor::GRAY8_SRGB).unwrap();
    let encoded = JpegEncoderConfig::new()
        .job()
        .encoder()
        .unwrap()
        .encode(pixels)
        .unwrap();
    out.push(("baseline gray".to_owned(), Arc::new(encoded.into_vec())));
    out
}

#[test]
fn every_decode_route_fits_the_test_stack_and_agrees() {
    for (name, bytes) in fixtures() {
        let source = bytes.clone();
        let one_shot = on_test_stack(format!("{name} one-shot"), move || {
            let output = JpegDecoderConfig::new()
                .job()
                .decoder(Cow::Borrowed(&source[..]), &[])
                .unwrap()
                .decode()
                .unwrap();
            packed_rows(&output.pixels())
        });
        let source = bytes.clone();
        let streamed = on_test_stack(format!("{name} streaming"), move || {
            let mut decoder = JpegDecoderConfig::new()
                .job()
                .streaming_decoder(Cow::Borrowed(&source[..]), &[])
                .unwrap();
            let mut rows = Vec::new();
            while let Some((_, strip)) = decoder.next_batch().unwrap() {
                rows.extend(packed_rows(&strip));
            }
            rows
        });
        let source = bytes.clone();
        let controlled = on_test_stack(format!("{name} controlled"), move || {
            let native = JpegDecoderConfig::new()
                .job()
                .dyn_controlled_decoder(&[], StorageBudget::new(1 << 20, 4))
                .unwrap();
            let mut session = PolicyDecoder::new(native, PolicySet::new());
            let mut sink = CollectSink::new();
            drive_from_slice(&mut session, &source[..], &mut sink, None, 11).unwrap();
            packed_rows(&sink.as_pixel_slice().unwrap())
        });
        assert!(!one_shot.is_empty(), "{name}");
        assert_eq!(streamed, one_shot, "{name}: streaming vs one-shot pixels");
        assert_eq!(
            controlled, one_shot,
            "{name}: controlled vs one-shot pixels"
        );
    }
}
