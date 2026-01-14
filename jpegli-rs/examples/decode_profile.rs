//! Benchmark decoder performance vs zune-jpeg
//!
//! Usage: cargo run --release --example decode_profile
//!        cargo run --release --example decode_profile -- --jpegli-only

use enough::Unstoppable;
use jpegli::{decoder::{Decoder, PixelFormat}, encoder::{EncoderConfig, PixelLayout}};
use std::env;
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
    let config = EncoderConfig::new().quality(90.0);
    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .unwrap();
    enc.push_packed(&data, Unstoppable).unwrap();
    enc.finish().unwrap()
}

fn bench_jpegli(jpeg_data: &[u8], iterations: usize) -> f64 {
    let decoder = Decoder::new().output_format(PixelFormat::Rgb);

    // Warmup - more iterations to ensure CPU turbo is active
    for _ in 0..10 {
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
    for _ in 0..10 {
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
    let args: Vec<String> = env::args().collect();
    let jpegli_only = args.iter().any(|a| a == "--jpegli-only");
    let iterations = 20;

    // Create JPEG once before benchmarking
    eprintln!("Creating test JPEG...");
    let jpeg_data = create_test_jpeg(2048, 2048);
    eprintln!("JPEG size: {} bytes\n", jpeg_data.len());

    println!(
        "Benchmarking 2048x2048 decode ({} iterations)...\n",
        iterations
    );

    let jpegli_mpps = bench_jpegli(&jpeg_data, iterations);
    println!("jpegli-rs: {:.1} MP/s", jpegli_mpps);

    if !jpegli_only {
        let zune_mpps = bench_zune(&jpeg_data, iterations);
        println!("zune-jpeg: {:.1} MP/s", zune_mpps);
        println!(
            "\nRatio: {:.2}x (zune-jpeg is {:.1}x faster)",
            zune_mpps / jpegli_mpps,
            zune_mpps / jpegli_mpps
        );
    }
}
