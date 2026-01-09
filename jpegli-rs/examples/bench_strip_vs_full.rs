//! Benchmark strip-based vs full-plane encoding
use jpegli::{Encoder, PixelFormat, Quality};
use std::time::Instant;

fn create_gradient(width: usize, height: usize) -> Vec<u8> {
    let mut data = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            data[idx] = ((x * 255) / width) as u8;
            data[idx + 1] = ((y * 255) / height) as u8;
            data[idx + 2] = (((x + y) * 17) % 256) as u8;
        }
    }
    data
}

fn bench_full(data: &[u8], width: usize, height: usize) -> (f64, usize) {
    // Warmup
    for _ in 0..2 {
        let _ = Encoder::new()
            .width(width as u32)
            .height(height as u32)
            .pixel_format(PixelFormat::Rgb)
            .jpegli_quality(Quality::from_quality(90.0))
            .encode(data);
    }

    let mut times = Vec::new();
    let mut size = 0;
    for _ in 0..5 {
        let start = Instant::now();
        let result = Encoder::new()
            .width(width as u32)
            .height(height as u32)
            .pixel_format(PixelFormat::Rgb)
            .jpegli_quality(Quality::from_quality(90.0))
            .encode(data)
            .unwrap();
        times.push(start.elapsed().as_secs_f64() * 1000.0);
        size = result.len();
    }
    (times.iter().sum::<f64>() / 5.0, size)
}

fn bench_strip(data: &[u8], width: usize, height: usize) -> (f64, usize) {
    // Warmup
    for _ in 0..2 {
        let _ = Encoder::new()
            .width(width as u32)
            .height(height as u32)
            .pixel_format(PixelFormat::Rgb)
            .jpegli_quality(Quality::from_quality(90.0))
            .encode_strip_based(data);
    }

    let mut times = Vec::new();
    let mut size = 0;
    for _ in 0..5 {
        let start = Instant::now();
        let result = Encoder::new()
            .width(width as u32)
            .height(height as u32)
            .pixel_format(PixelFormat::Rgb)
            .jpegli_quality(Quality::from_quality(90.0))
            .encode_strip_based(data)
            .unwrap();
        times.push(start.elapsed().as_secs_f64() * 1000.0);
        size = result.len();
    }
    (times.iter().sum::<f64>() / 5.0, size)
}

fn main() {
    for (name, width, height) in [("2K", 2048, 2048), ("4K", 3840, 2160)] {
        println!("\n=== {} ({}x{}) ===", name, width, height);
        let data = create_gradient(width, height);
        let pixels = (width * height) as f64;

        let (full_ms, full_size) = bench_full(&data, width, height);
        let full_mpps = pixels / (full_ms / 1000.0) / 1_000_000.0;
        println!(
            "Full-plane: {:.1} ms, {:.1} MP/s, {} KB",
            full_ms,
            full_mpps,
            full_size / 1024
        );

        let (strip_ms, strip_size) = bench_strip(&data, width, height);
        let strip_mpps = pixels / (strip_ms / 1000.0) / 1_000_000.0;
        println!(
            "Strip-based: {:.1} ms, {:.1} MP/s, {} KB",
            strip_ms,
            strip_mpps,
            strip_size / 1024
        );

        let speedup = full_ms / strip_ms;
        println!("Speedup: {:.2}x", speedup);
    }
}
