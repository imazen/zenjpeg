//! Decoder performance benchmark
//!
//! Compares jpegli-rs decoder vs jpeg-decoder vs zune-jpeg
//!
//! Usage: cargo run --release --example decode_benchmark

use std::time::Instant;
use zune_jpeg::JpegDecoder;

fn main() {
    // Generate test image
    let width = 1024u32;
    let height = 768u32;
    let pixels: Vec<u8> = (0..width * height * 3)
        .map(|i| ((i % 256) ^ ((i / 256) % 256)) as u8)
        .collect();

    // Encode with jpegli
    let jpeg = jpegli::Encoder::new()
        .width(width)
        .height(height)
        .jpegli_quality(jpegli::Quality::from_quality(85.0))
        .encode(&pixels)
        .expect("encoding failed");

    println!(
        "Image: {}x{} ({:.2} MP)",
        width,
        height,
        (width * height) as f64 / 1_000_000.0
    );
    println!(
        "JPEG size: {} bytes ({:.2} bpp)",
        jpeg.len(),
        jpeg.len() as f64 * 8.0 / (width * height) as f64
    );
    println!();

    // Warmup
    let _ = jpegli::Decoder::new().decode(&jpeg);
    let _ = jpeg_decoder::Decoder::new(std::io::Cursor::new(&jpeg)).decode();
    let mut zune = JpegDecoder::new(&jpeg);
    let _ = zune.decode();

    let iterations = 20;
    let mpixels = (width * height) as f64 / 1_000_000.0;

    // Benchmark jpegli-rs decoder (u8 output)
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = jpegli::Decoder::new().decode(&jpeg).unwrap();
    }
    let jpegli_time = start.elapsed() / iterations;
    println!(
        "jpegli-rs u8:   {:?} ({:.1} MP/s)",
        jpegli_time,
        mpixels / jpegli_time.as_secs_f64()
    );

    // Benchmark jpegli-rs decoder (f32 output)
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = jpegli::Decoder::new().decode_f32(&jpeg).unwrap();
    }
    let jpegli_f32_time = start.elapsed() / iterations;
    println!(
        "jpegli-rs f32:  {:?} ({:.1} MP/s)",
        jpegli_f32_time,
        mpixels / jpegli_f32_time.as_secs_f64()
    );

    // Benchmark jpeg-decoder
    let start = Instant::now();
    for _ in 0..iterations {
        let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&jpeg));
        let _ = decoder.decode().unwrap();
    }
    let jpeg_decoder_time = start.elapsed() / iterations;
    println!(
        "jpeg-decoder:   {:?} ({:.1} MP/s)",
        jpeg_decoder_time,
        mpixels / jpeg_decoder_time.as_secs_f64()
    );

    // Benchmark zune-jpeg
    let start = Instant::now();
    for _ in 0..iterations {
        let mut decoder = JpegDecoder::new(&jpeg);
        let _ = decoder.decode().unwrap();
    }
    let zune_time = start.elapsed() / iterations;
    println!(
        "zune-jpeg:      {:?} ({:.1} MP/s)",
        zune_time,
        mpixels / zune_time.as_secs_f64()
    );

    println!();
    println!("Speed comparison (vs jpegli-rs u8):");
    println!(
        "  jpeg-decoder: {:.2}x faster",
        jpegli_time.as_secs_f64() / jpeg_decoder_time.as_secs_f64()
    );
    println!(
        "  zune-jpeg:    {:.2}x faster",
        jpegli_time.as_secs_f64() / zune_time.as_secs_f64()
    );

    // Also test with larger image
    println!("\n--- 4K Test ---");
    let width_4k = 3840u32;
    let height_4k = 2160u32;
    let pixels_4k: Vec<u8> = (0..width_4k * height_4k * 3)
        .map(|i| ((i % 256) ^ ((i / 256) % 256)) as u8)
        .collect();

    let jpeg_4k = jpegli::Encoder::new()
        .width(width_4k)
        .height(height_4k)
        .jpegli_quality(jpegli::Quality::from_quality(85.0))
        .encode(&pixels_4k)
        .expect("encoding 4K failed");

    println!(
        "Image: {}x{} ({:.2} MP)",
        width_4k,
        height_4k,
        (width_4k * height_4k) as f64 / 1_000_000.0
    );

    let iterations_4k = 5;
    let mpixels_4k = (width_4k * height_4k) as f64 / 1_000_000.0;

    let start = Instant::now();
    for _ in 0..iterations_4k {
        let _ = jpegli::Decoder::new().decode(&jpeg_4k).unwrap();
    }
    let jpegli_4k_time = start.elapsed() / iterations_4k;
    println!(
        "jpegli-rs:      {:?} ({:.1} MP/s)",
        jpegli_4k_time,
        mpixels_4k / jpegli_4k_time.as_secs_f64()
    );

    let start = Instant::now();
    for _ in 0..iterations_4k {
        let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&jpeg_4k));
        let _ = decoder.decode().unwrap();
    }
    let jpeg_decoder_4k_time = start.elapsed() / iterations_4k;
    println!(
        "jpeg-decoder:   {:?} ({:.1} MP/s)",
        jpeg_decoder_4k_time,
        mpixels_4k / jpeg_decoder_4k_time.as_secs_f64()
    );

    let start = Instant::now();
    for _ in 0..iterations_4k {
        let mut decoder = JpegDecoder::new(&jpeg_4k);
        let _ = decoder.decode().unwrap();
    }
    let zune_4k_time = start.elapsed() / iterations_4k;
    println!(
        "zune-jpeg:      {:?} ({:.1} MP/s)",
        zune_4k_time,
        mpixels_4k / zune_4k_time.as_secs_f64()
    );
}
