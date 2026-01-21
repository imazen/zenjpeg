//! Benchmark: wide crate vs multiversion autovectorization for AQ
//!
//! This compares two approaches to SIMD:
//! 1. `wide` crate with explicit f32x8 types (compile-time feature detection)
//! 2. Pure scalar code with `#[multiversion]` (runtime dispatch, autovectorization)
//!
//! Run: cargo run --release -p jpegli-rs --example bench_autovec_aq

use std::time::Instant;

fn main() {
    const WIDTH: usize = 4096;
    const ITERS: usize = 1000;

    // Create padded test data (width + 2 for left/right padding)
    let padded_len = WIDTH + 2;
    let mut row: Vec<f32> = (0..padded_len).map(|i| (i as f32 * 1.7) % 255.0).collect();
    let mut row_above: Vec<f32> = (0..padded_len).map(|i| (i as f32 * 2.3 + 10.0) % 255.0).collect();
    let mut row_below: Vec<f32> = (0..padded_len).map(|i| (i as f32 * 3.1 + 20.0) % 255.0).collect();

    // Edge replication for padding
    row[0] = row[1];
    row[padded_len - 1] = row[padded_len - 2];
    row_above[0] = row_above[1];
    row_above[padded_len - 1] = row_above[padded_len - 2];
    row_below[0] = row_below[1];
    row_below[padded_len - 1] = row_below[padded_len - 2];

    let mut output_wide = vec![0.0f32; WIDTH];
    let mut output_autovec = vec![0.0f32; WIDTH];

    // Warmup
    for _ in 0..10 {
        jpegli::quant::aq::simd::pre_erosion_row_padded(
            &row, &row_above, &row_below, WIDTH, &mut output_wide
        );
        jpegli::quant::aq::autovec::pre_erosion_row_autovec(
            &row, &row_above, &row_below, WIDTH, &mut output_autovec
        );
    }

    // Benchmark wide-based version (reset between iterations for fair comparison)
    let start = Instant::now();
    for _ in 0..ITERS {
        output_wide.fill(0.0);
        jpegli::quant::aq::simd::pre_erosion_row_padded(
            &row, &row_above, &row_below, WIDTH, &mut output_wide
        );
        std::hint::black_box(&output_wide);
    }
    let wide_time = start.elapsed();

    // Benchmark autovec version
    let start = Instant::now();
    for _ in 0..ITERS {
        output_autovec.fill(0.0);
        jpegli::quant::aq::autovec::pre_erosion_row_autovec(
            &row, &row_above, &row_below, WIDTH, &mut output_autovec
        );
        std::hint::black_box(&output_autovec);
    }
    let autovec_time = start.elapsed();

    // Benchmark autovec iter version
    let mut output_autovec_iter = vec![0.0f32; WIDTH];
    let start = Instant::now();
    for _ in 0..ITERS {
        output_autovec_iter.fill(0.0);
        jpegli::quant::aq::autovec::pre_erosion_row_autovec_iter(
            &row, &row_above, &row_below, WIDTH, &mut output_autovec_iter
        );
        std::hint::black_box(&output_autovec_iter);
    }
    let autovec_iter_time = start.elapsed();

    // Verify correctness (run once with fresh buffers)
    output_wide.fill(0.0);
    output_autovec.fill(0.0);
    jpegli::quant::aq::simd::pre_erosion_row_padded(
        &row, &row_above, &row_below, WIDTH, &mut output_wide
    );
    jpegli::quant::aq::autovec::pre_erosion_row_autovec_iter(
        &row, &row_above, &row_below, WIDTH, &mut output_autovec
    );

    let mut max_diff = 0.0f32;
    for i in 0..WIDTH {
        let diff = (output_wide[i] - output_autovec[i]).abs();
        max_diff = max_diff.max(diff);
    }

    println!("Pre-erosion row benchmark (width={}, {} iters)", WIDTH, ITERS);
    println!("=========================================================");
    println!();
    println!("wide (f32x8):      {:>8.2} ms ({:.2} µs/row)",
             wide_time.as_secs_f64() * 1000.0,
             wide_time.as_secs_f64() * 1e6 / ITERS as f64);
    println!("autovec (chunked): {:>8.2} ms ({:.2} µs/row)",
             autovec_time.as_secs_f64() * 1000.0,
             autovec_time.as_secs_f64() * 1e6 / ITERS as f64);
    println!("autovec (iter):    {:>8.2} ms ({:.2} µs/row)",
             autovec_iter_time.as_secs_f64() * 1000.0,
             autovec_iter_time.as_secs_f64() * 1e6 / ITERS as f64);
    println!();

    let speedup_chunked = wide_time.as_secs_f64() / autovec_time.as_secs_f64();
    let speedup_iter = wide_time.as_secs_f64() / autovec_iter_time.as_secs_f64();

    if speedup_chunked > 1.0 {
        println!("autovec (chunked) is {:.2}x FASTER than wide", speedup_chunked);
    } else {
        println!("autovec (chunked) is {:.2}x SLOWER than wide", 1.0 / speedup_chunked);
    }

    if speedup_iter > 1.0 {
        println!("autovec (iter) is {:.2}x FASTER than wide", speedup_iter);
    } else {
        println!("autovec (iter) is {:.2}x SLOWER than wide", 1.0 / speedup_iter);
    }

    println!();
    println!("Max difference: {:.2e} (should be ~0)", max_diff);

    if max_diff > 1e-4 {
        println!("WARNING: Results differ significantly!");
    } else {
        println!("✓ Results match");
    }
}
