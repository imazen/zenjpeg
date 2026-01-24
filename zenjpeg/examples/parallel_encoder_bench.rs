//! Benchmark parallel vs sequential full encoder performance.
//!
//! Run with: cargo run --release --features parallel --example parallel_encoder_bench

use enough::Unstoppable;
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};
use std::time::Instant;

fn main() {
    println!("Full Encoder Benchmark (Parallel Y DCT)\n");
    println!(
        "{:>12} {:>10} {:>10} {:>8} {:>10}",
        "Size", "Time", "MP/s", "Speedup", "Output KB"
    );
    println!("{}", "-".repeat(56));

    for &(width, height) in &[(512, 512), (1024, 1024), (2048, 2048), (4096, 4096)] {
        let result = benchmark_encode(width, height);
        let megapixels = (width * height) as f64 / 1_000_000.0;
        let mp_per_sec = megapixels / (result.time_ms / 1000.0);

        println!(
            "{:>5}x{:<5} {:>8.2}ms {:>8.1} {:>8} {:>10.1}",
            width,
            height,
            result.time_ms,
            mp_per_sec,
            "-", // No comparison baseline in this simple bench
            result.output_kb
        );
    }

    println!("\nConfiguration:");
    println!("- Feature: parallel (Y channel DCT parallelized)");
    println!("- Threads: {}", rayon::current_num_threads());
    println!("- Quality: 85");
}

struct BenchResult {
    time_ms: f64,
    output_kb: f64,
}

fn benchmark_encode(width: usize, height: usize) -> BenchResult {
    // Create pseudo-random test image
    let mut pixels = vec![0u8; width * height * 3];
    let mut seed = 12345u64;
    for p in pixels.iter_mut() {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        *p = (seed >> 33) as u8;
    }

    let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);

    // Warm up
    {
        let mut enc = config
            .encode_from_bytes(width as u32, height as u32, PixelLayout::Rgb8Srgb)
            .unwrap();
        enc.push_packed(&pixels, Unstoppable).unwrap();
        let _ = enc.finish().unwrap();
    }

    // Benchmark
    let iterations = if width * height < 4_000_000 { 20 } else { 5 };

    let start = Instant::now();
    let mut output = Vec::new();
    for _ in 0..iterations {
        let mut enc = config
            .encode_from_bytes(width as u32, height as u32, PixelLayout::Rgb8Srgb)
            .unwrap();
        enc.push_packed(&pixels, Unstoppable).unwrap();
        output = enc.finish().unwrap();
    }
    let elapsed = start.elapsed();

    BenchResult {
        time_ms: elapsed.as_secs_f64() * 1000.0 / iterations as f64,
        output_kb: output.len() as f64 / 1024.0,
    }
}
