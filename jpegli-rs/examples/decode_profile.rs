//! Benchmark decoder performance vs zune-jpeg
//!
//! Usage: cargo run --release --example decode_profile

use jpegli::{Decoder, Encoder, PixelFormat, Quality};
use std::time::Instant;
use zune_jpeg::zune_core::bytestream::ZCursor;
use zune_jpeg::zune_core::colorspace::ColorSpace;
use zune_jpeg::zune_core::options::DecoderOptions;
use zune_jpeg::JpegDecoder;

fn create_test_jpeg(width: u32, height: u32) -> Vec<u8> {
    let mut data = vec![0u8; (width * height * 3) as usize];
    for y in 0..height as usize {
        for x in 0..width as usize {
            let idx = (y * width as usize + x) * 3;
            data[idx] = ((x * 255) / width as usize) as u8;
            data[idx + 1] = ((y * 255) / height as usize) as u8;
            data[idx + 2] = 128;
        }
    }
    #[allow(deprecated)]
    Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(Quality::from_quality(90.0))
        .encode(&data)
        .unwrap()
}

fn bench_jpegli(jpeg_data: &[u8], iterations: usize) -> f64 {
    let decoder = Decoder::new().output_format(PixelFormat::Rgb);

    // Warmup
    for _ in 0..3 {
        let _ = decoder.decode(jpeg_data);
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = decoder.decode(jpeg_data);
    }
    let elapsed = start.elapsed();

    let pixels = 2048.0 * 2048.0 * iterations as f64;
    pixels / elapsed.as_secs_f64() / 1_000_000.0
}

fn bench_zune(jpeg_data: &[u8], iterations: usize) -> f64 {
    let options = DecoderOptions::new_fast().jpeg_set_out_colorspace(ColorSpace::RGB);

    // Warmup
    for _ in 0..3 {
        let cursor = ZCursor::new(jpeg_data);
        let mut decoder = JpegDecoder::new_with_options(cursor, options);
        let _ = decoder.decode().unwrap();
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let cursor = ZCursor::new(jpeg_data);
        let mut decoder = JpegDecoder::new_with_options(cursor, options);
        let _ = decoder.decode().unwrap();
    }
    let elapsed = start.elapsed();

    let pixels = 2048.0 * 2048.0 * iterations as f64;
    pixels / elapsed.as_secs_f64() / 1_000_000.0
}

fn main() {
    let jpeg_data = create_test_jpeg(2048, 2048);
    let iterations = 10;

    println!("Benchmarking 2048x2048 decode ({} iterations)...\n", iterations);

    let jpegli_mpps = bench_jpegli(&jpeg_data, iterations);
    let zune_mpps = bench_zune(&jpeg_data, iterations);

    println!("jpegli-rs: {:.1} MP/s", jpegli_mpps);
    println!("zune-jpeg: {:.1} MP/s", zune_mpps);
    println!(
        "\nRatio: {:.2}x (zune-jpeg is {:.1}x faster)",
        zune_mpps / jpegli_mpps,
        zune_mpps / jpegli_mpps
    );
}
