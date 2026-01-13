//! Benchmark comparing YCbCr f32 decode path vs RGB decode path.
//!
//! Run with: cargo run --release --example ycbcr_benchmark

use jpegli::{Decoder, JpegEncoder, PixelFormat, Quality, Subsampling};
use std::time::{Duration, Instant};

fn create_test_jpeg(width: u32, height: u32, quality: f32) -> Vec<u8> {
    let mut data = vec![0u8; (width * height * 3) as usize];
    for y in 0..height as usize {
        for x in 0..width as usize {
            let idx = (y * width as usize + x) * 3;
            data[idx] = ((x * 255) / width as usize) as u8;
            data[idx + 1] = ((y * 255) / height as usize) as u8;
            data[idx + 2] = 128;
        }
    }
    JpegEncoder::new(width, height)
        .pixel_format(PixelFormat::Rgb)
        .quality(Quality::from_quality(quality))
        .subsampling(Subsampling::S420)
        .encode(&data)
        .unwrap()
}

fn bench_decode_rgb(jpeg_data: &[u8], iterations: usize) -> Duration {
    let decoder = Decoder::new().output_format(PixelFormat::Rgb);

    // Warmup
    for _ in 0..3 {
        let _ = decoder.decode(jpeg_data);
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = decoder.decode(jpeg_data).unwrap();
    }
    start.elapsed()
}

fn bench_decode_ycbcr(jpeg_data: &[u8], iterations: usize) -> Duration {
    let decoder = Decoder::new();

    // Warmup
    for _ in 0..3 {
        let _ = decoder.decode_to_ycbcr_f32(jpeg_data);
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = decoder.decode_to_ycbcr_f32(jpeg_data).unwrap();
    }
    start.elapsed()
}

fn bench_zune_jpeg(jpeg_data: &[u8], iterations: usize) -> Duration {
    use std::io::Cursor;
    use zune_jpeg::JpegDecoder;

    // Warmup
    for _ in 0..3 {
        let cursor = Cursor::new(jpeg_data);
        let mut decoder = JpegDecoder::new(cursor);
        let _ = decoder.decode();
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let cursor = Cursor::new(jpeg_data);
        let mut decoder = JpegDecoder::new(cursor);
        let _ = decoder.decode().unwrap();
    }
    start.elapsed()
}

fn main() {
    let sizes = [(512, 512), (1024, 1024), (2048, 2048)];
    let iterations = 10;

    println!("YCbCr f32 Decode Performance Comparison");
    println!("========================================\n");
    println!(
        "{:>10} {:>12} {:>12} {:>12} {:>15}",
        "Size", "RGB (ms)", "YCbCr (ms)", "zune (ms)", "YCbCr speedup"
    );
    println!("{:-<67}", "");

    for (width, height) in sizes {
        let jpeg_data = create_test_jpeg(width, height, 90.0);
        let pixels = width as f64 * height as f64;

        let rgb_time = bench_decode_rgb(&jpeg_data, iterations);
        let ycbcr_time = bench_decode_ycbcr(&jpeg_data, iterations);
        let zune_time = bench_zune_jpeg(&jpeg_data, iterations);

        let rgb_ms = rgb_time.as_secs_f64() * 1000.0 / iterations as f64;
        let ycbcr_ms = ycbcr_time.as_secs_f64() * 1000.0 / iterations as f64;
        let zune_ms = zune_time.as_secs_f64() * 1000.0 / iterations as f64;

        let speedup = rgb_ms / ycbcr_ms;

        println!(
            "{:>10} {:>12.2} {:>12.2} {:>12.2} {:>14.1}x",
            format!("{}x{}", width, height),
            rgb_ms,
            ycbcr_ms,
            zune_ms,
            speedup
        );

        // Also print MP/s
        let rgb_mpps = pixels * iterations as f64 / rgb_time.as_secs_f64() / 1_000_000.0;
        let ycbcr_mpps = pixels * iterations as f64 / ycbcr_time.as_secs_f64() / 1_000_000.0;
        let zune_mpps = pixels * iterations as f64 / zune_time.as_secs_f64() / 1_000_000.0;

        println!(
            "{:>10} {:>12.1} {:>12.1} {:>12.1} (MP/s)",
            "", rgb_mpps, ycbcr_mpps, zune_mpps
        );
        println!();
    }

    println!("\nNotes:");
    println!("- YCbCr path bypasses YCbCr->RGB color conversion (the main bottleneck)");
    println!("- zune-jpeg is a pure Rust SIMD-optimized decoder");
    println!("- All values in milliseconds, lower is better");
    println!("- MP/s = megapixels per second");
}
