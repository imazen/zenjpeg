//! Benchmark: wide crate vs multiversion autovectorization for AQ
//!
//! This compares two approaches to SIMD:
//! 1. `wide` crate with explicit f32x8 types (compile-time feature detection)
//! 2. Pure scalar code with `#[multiversion]` (runtime dispatch, autovectorization)
//!
//! Run: cargo run --release -p jpegli-rs --example bench_autovec_aq

use std::time::Instant;

fn main() {
    bench_pre_erosion();
    println!();
    bench_gamma_hf_modulation();
    println!();
    bench_per_block_modulations();
}

fn bench_pre_erosion() {
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
        jpegli::quant::aq::autovec::pre_erosion_row_autovec_iter(
            &row, &row_above, &row_below, WIDTH, &mut output_autovec
        );
    }

    // Benchmark wide-based version
    let start = Instant::now();
    for _ in 0..ITERS {
        output_wide.fill(0.0);
        jpegli::quant::aq::simd::pre_erosion_row_padded(
            &row, &row_above, &row_below, WIDTH, &mut output_wide
        );
        std::hint::black_box(&output_wide);
    }
    let wide_time = start.elapsed();

    // Benchmark autovec iter version
    let start = Instant::now();
    for _ in 0..ITERS {
        output_autovec.fill(0.0);
        jpegli::quant::aq::autovec::pre_erosion_row_autovec_iter(
            &row, &row_above, &row_below, WIDTH, &mut output_autovec
        );
        std::hint::black_box(&output_autovec);
    }
    let autovec_time = start.elapsed();

    // Verify correctness
    output_wide.fill(0.0);
    output_autovec.fill(0.0);
    jpegli::quant::aq::simd::pre_erosion_row_padded(
        &row, &row_above, &row_below, WIDTH, &mut output_wide
    );
    jpegli::quant::aq::autovec::pre_erosion_row_autovec_iter(
        &row, &row_above, &row_below, WIDTH, &mut output_autovec
    );

    let max_diff = output_wide.iter().zip(&output_autovec)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!("Pre-erosion row (width={}, {} iters)", WIDTH, ITERS);
    println!("=========================================================");
    println!("wide (f32x8): {:>8.2} µs/row", wide_time.as_secs_f64() * 1e6 / ITERS as f64);
    println!("autovec:      {:>8.2} µs/row", autovec_time.as_secs_f64() * 1e6 / ITERS as f64);

    let speedup = wide_time.as_secs_f64() / autovec_time.as_secs_f64();
    if speedup > 1.0 {
        println!("autovec is {:.2}x FASTER than wide", speedup);
    } else {
        println!("autovec is {:.2}x SLOWER than wide", 1.0 / speedup);
    }
    println!("Max diff: {:.2e} {}", max_diff, if max_diff < 1e-4 { "✓" } else { "⚠" });
}

fn bench_gamma_hf_modulation() {
    const BLOCK_W: usize = 512;  // 512 blocks = 4096 pixels
    const HEIGHT: usize = 8;
    const ITERS: usize = 1000;

    let stride = BLOCK_W * 8 + 1;  // +1 for horizontal neighbor access
    let input: Vec<f32> = (0..stride * HEIGHT)
        .map(|i| ((i % 256) as f32 * 1.7) % 255.0)
        .collect();

    // Warmup
    for bx in 0..10.min(BLOCK_W) {
        let x_start = bx * 8;
        let block = &input[x_start..];
        std::hint::black_box(jpegli::quant::aq::simd::gamma_modulation_sum_8x8(
            block, stride, x_start, 0, BLOCK_W * 8, HEIGHT
        ));
        std::hint::black_box(jpegli::quant::aq::autovec::gamma_modulation_sum_8x8_autovec(
            block, stride, 0, HEIGHT
        ));
    }

    // Benchmark wide gamma
    let start = Instant::now();
    for _ in 0..ITERS {
        for bx in 0..BLOCK_W {
            let x_start = bx * 8;
            let block = &input[x_start..];
            std::hint::black_box(jpegli::quant::aq::simd::gamma_modulation_sum_8x8(
                block, stride, x_start, 0, BLOCK_W * 8, HEIGHT
            ));
        }
    }
    let wide_gamma_time = start.elapsed();

    // Benchmark autovec gamma
    let start = Instant::now();
    for _ in 0..ITERS {
        for bx in 0..BLOCK_W {
            let x_start = bx * 8;
            let block = &input[x_start..];
            std::hint::black_box(jpegli::quant::aq::autovec::gamma_modulation_sum_8x8_autovec(
                block, stride, 0, HEIGHT
            ));
        }
    }
    let autovec_gamma_time = start.elapsed();

    // Benchmark wide hf
    let start = Instant::now();
    for _ in 0..ITERS {
        for bx in 0..BLOCK_W {
            let x_start = bx * 8;
            let block = &input[x_start..];
            std::hint::black_box(jpegli::quant::aq::simd::hf_modulation_sum_8x8(
                block, stride, x_start, 0, BLOCK_W * 8, HEIGHT
            ));
        }
    }
    let wide_hf_time = start.elapsed();

    // Benchmark autovec hf
    let start = Instant::now();
    for _ in 0..ITERS {
        for bx in 0..BLOCK_W {
            let x_start = bx * 8;
            let block = &input[x_start..];
            std::hint::black_box(jpegli::quant::aq::autovec::hf_modulation_sum_8x8_autovec(
                block, stride, 0, HEIGHT
            ));
        }
    }
    let autovec_hf_time = start.elapsed();

    println!("Gamma/HF modulation ({} blocks, {} iters)", BLOCK_W, ITERS);
    println!("=========================================================");
    println!("gamma_modulation_sum_8x8:");
    println!("  wide:    {:>6.1} ns/block", wide_gamma_time.as_nanos() as f64 / (ITERS * BLOCK_W) as f64);
    println!("  autovec: {:>6.1} ns/block", autovec_gamma_time.as_nanos() as f64 / (ITERS * BLOCK_W) as f64);
    let speedup = wide_gamma_time.as_secs_f64() / autovec_gamma_time.as_secs_f64();
    if speedup > 1.0 {
        println!("  autovec is {:.2}x FASTER", speedup);
    } else {
        println!("  autovec is {:.2}x SLOWER", 1.0 / speedup);
    }

    println!("hf_modulation_sum_8x8:");
    println!("  wide:    {:>6.1} ns/block", wide_hf_time.as_nanos() as f64 / (ITERS * BLOCK_W) as f64);
    println!("  autovec: {:>6.1} ns/block", autovec_hf_time.as_nanos() as f64 / (ITERS * BLOCK_W) as f64);
    let speedup = wide_hf_time.as_secs_f64() / autovec_hf_time.as_secs_f64();
    if speedup > 1.0 {
        println!("  autovec is {:.2}x FASTER", speedup);
    } else {
        println!("  autovec is {:.2}x SLOWER", 1.0 / speedup);
    }
}

fn bench_per_block_modulations() {
    const BLOCK_W: usize = 512;  // 512 blocks = 4096 pixels
    const HEIGHT: usize = 8;
    const ITERS: usize = 1000;

    let stride = BLOCK_W * 8 + 1;
    let input: Vec<f32> = (0..stride * HEIGHT)
        .map(|i| ((i % 256) as f32 * 1.7) % 255.0)
        .collect();

    let mut aq_row_wide = vec![0.5f32; BLOCK_W];
    let mut aq_row_autovec = vec![0.5f32; BLOCK_W];

    let mul = 0.841;
    let add = 0.1;

    // Warmup
    for _ in 0..10 {
        aq_row_wide.fill(0.5);
        jpegli::quant::aq::simd::per_block_modulations_row(
            &input, stride, BLOCK_W * 8, HEIGHT, 0, BLOCK_W, &mut aq_row_wide, mul, add
        );
        aq_row_autovec.fill(0.5);
        jpegli::quant::aq::autovec::per_block_modulations_row_autovec(
            &input, stride, BLOCK_W * 8, HEIGHT, 0, BLOCK_W, &mut aq_row_autovec, mul, add
        );
    }

    // Benchmark wide
    let start = Instant::now();
    for _ in 0..ITERS {
        aq_row_wide.fill(0.5);
        jpegli::quant::aq::simd::per_block_modulations_row(
            &input, stride, BLOCK_W * 8, HEIGHT, 0, BLOCK_W, &mut aq_row_wide, mul, add
        );
        std::hint::black_box(&aq_row_wide);
    }
    let wide_time = start.elapsed();

    // Benchmark autovec
    let start = Instant::now();
    for _ in 0..ITERS {
        aq_row_autovec.fill(0.5);
        jpegli::quant::aq::autovec::per_block_modulations_row_autovec(
            &input, stride, BLOCK_W * 8, HEIGHT, 0, BLOCK_W, &mut aq_row_autovec, mul, add
        );
        std::hint::black_box(&aq_row_autovec);
    }
    let autovec_time = start.elapsed();

    // Verify correctness
    aq_row_wide.fill(0.5);
    aq_row_autovec.fill(0.5);
    jpegli::quant::aq::simd::per_block_modulations_row(
        &input, stride, BLOCK_W * 8, HEIGHT, 0, BLOCK_W, &mut aq_row_wide, mul, add
    );
    jpegli::quant::aq::autovec::per_block_modulations_row_autovec(
        &input, stride, BLOCK_W * 8, HEIGHT, 0, BLOCK_W, &mut aq_row_autovec, mul, add
    );

    let max_diff = aq_row_wide.iter().zip(&aq_row_autovec)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!("per_block_modulations_row ({} blocks, {} iters)", BLOCK_W, ITERS);
    println!("=========================================================");
    println!("wide:    {:>6.2} µs/row ({:.1} ns/block)",
             wide_time.as_secs_f64() * 1e6 / ITERS as f64,
             wide_time.as_nanos() as f64 / (ITERS * BLOCK_W) as f64);
    println!("autovec: {:>6.2} µs/row ({:.1} ns/block)",
             autovec_time.as_secs_f64() * 1e6 / ITERS as f64,
             autovec_time.as_nanos() as f64 / (ITERS * BLOCK_W) as f64);

    let speedup = wide_time.as_secs_f64() / autovec_time.as_secs_f64();
    if speedup > 1.0 {
        println!("autovec is {:.2}x FASTER than wide", speedup);
    } else {
        println!("autovec is {:.2}x SLOWER than wide", 1.0 / speedup);
    }
    println!("Max diff: {:.2e} {}", max_diff, if max_diff < 1e-3 { "✓" } else { "⚠" });
}
