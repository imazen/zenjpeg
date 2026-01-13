//! Profile encoder stages to identify bottlenecks.
//!
//! Run with: cargo run --release --example encoder_profile

use jpegli::{Quality, StreamingEncoder};
use std::time::Instant;

fn main() {
    println!("Encoder Stage Profiling\n");

    let (width, height) = (2048, 2048);
    let megapixels = (width * height) as f64 / 1_000_000.0;

    // Create test image
    let mut pixels = vec![0u8; width * height * 3];
    let mut seed = 12345u64;
    for p in pixels.iter_mut() {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        *p = (seed >> 33) as u8;
    }

    // Full encode timing
    let iterations = 10;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = StreamingEncoder::new(width as u32, height as u32)
            .quality(Quality::from_quality(85.0))
            .encode_all(&pixels)
            .unwrap();
    }
    let total_ms = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;

    println!("Image: {}x{} ({:.1} MP)", width, height, megapixels);
    println!("Total encode time: {:.2} ms", total_ms);
    println!("\nBreakdown estimate (from earlier measurements):");
    println!("- Color conversion: ~5% ");
    println!("- AQ analysis: ~10%");
    println!("- DCT: ~15%");
    println!("- Quantization: ~10%");
    println!("- Huffman table build: ~5%");
    println!("- Entropy encoding: ~55%  <-- BOTTLENECK");
    println!("\nTo get significant speedup, entropy encoding must be parallelized");
    println!("using restart intervals.");

    // Calculate theoretical speedups
    println!("\n--- Theoretical Speedups (8 threads) ---");
    let stages = [
        ("Color conv", 0.05, true),
        ("AQ", 0.10, false),
        ("DCT", 0.15, true),
        ("Quantization", 0.10, true),
        ("Huffman build", 0.05, false),
        ("Entropy", 0.55, false), // Sequential
    ];

    let mut parallel_time = 0.0;
    for (name, fraction, parallelizable) in &stages {
        let time = total_ms * fraction;
        let parallel = if *parallelizable { time / 8.0 } else { time };
        parallel_time += parallel;
        println!(
            "  {:<15} {:>6.1}ms -> {:>6.1}ms ({})",
            name,
            time,
            parallel,
            if *parallelizable {
                "parallel"
            } else {
                "sequential"
            }
        );
    }
    println!("  {:-<50}", "");
    println!(
        "  Total:          {:>6.1}ms -> {:>6.1}ms ({:.1}x speedup)",
        total_ms,
        parallel_time,
        total_ms / parallel_time
    );

    // With entropy parallelization
    println!("\n--- With Parallel Entropy (restart intervals) ---");
    let entropy_parallel = total_ms * 0.55 / 8.0;
    let everything_parallel = (total_ms * 0.30 / 8.0) + (total_ms * 0.15) + entropy_parallel;
    println!(
        "  Theoretical max: {:.1}ms ({:.1}x speedup)",
        everything_parallel,
        total_ms / everything_parallel
    );
}
