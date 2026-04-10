//! Benchmark parallel vs sequential DCT performance.
//!
//! Run with: cargo run --release --features parallel --example parallel_bench

#[cfg(feature = "parallel")]
use std::time::Instant;
#[cfg(feature = "parallel")]
use zenjpeg::encode::parallel::parallel_dct_y_blocks;

fn main() {
    println!("Parallel vs Sequential DCT Benchmark\n");

    #[cfg(not(feature = "parallel"))]
    {
        println!("ERROR: Run with --features parallel");
    }

    #[cfg(feature = "parallel")]
    run_benchmark();
}

#[cfg(feature = "parallel")]
fn run_benchmark() {
    println!(
        "{:>12} {:>12} {:>12} {:>12} {:>8}",
        "Size", "Blocks", "Sequential", "Parallel", "Speedup"
    );
    println!("{}", "-".repeat(60));

    for &(width, height) in &[
        (256, 256),
        (512, 512),
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
    ] {
        let result = benchmark_dct(width, height);
        println!(
            "{:>5}x{:<5} {:>12} {:>10.2}ms {:>10.2}ms {:>7.2}x",
            width, height, result.blocks, result.sequential_ms, result.parallel_ms, result.speedup
        );
    }

    println!("\nNotes:");
    println!("- Parallel threshold: 16384 blocks (1024x1024)");
    println!("- Chunk size: 4096 blocks per task");
    println!("- Threads: {}", rayon::current_num_threads());
    println!("- Sequential uses public API (slower), parallel uses internal SIMD");
}

#[cfg(feature = "parallel")]
struct BenchResult {
    blocks: usize,
    sequential_ms: f64,
    parallel_ms: f64,
    speedup: f64,
}

#[cfg(feature = "parallel")]
fn benchmark_dct(width: usize, height: usize) -> BenchResult {
    let padded_width = ((width + 7) / 8) * 8;
    let blocks_w = (width + 7) / 8;
    let blocks_h = (height + 7) / 8;
    let total_blocks = blocks_w * blocks_h;

    // Create test data (simulated Y strip covering full image)
    let strip: Vec<f32> = (0..height * padded_width)
        .map(|i| ((i * 17 + i / 3 * 31) % 256) as f32)
        .collect();

    // Warm up
    let mut warmup_output = Vec::new();
    parallel_dct_y_blocks(
        &strip,
        blocks_w,
        blocks_h,
        padded_width,
        None,
        &mut warmup_output,
    );
    drop(warmup_output);

    // Benchmark sequential
    let iterations = if total_blocks < 10000 { 50 } else { 10 };

    let seq_start = Instant::now();
    for _ in 0..iterations {
        let mut output = Vec::with_capacity(total_blocks);
        for local_by in 0..blocks_h {
            for bx in 0..blocks_w {
                let block = extract_block_sequential(&strip, bx, local_by, padded_width);
                let dct = dct_block(&block);
                output.push(dct);
            }
        }
        std::hint::black_box(&output);
    }
    let seq_elapsed = seq_start.elapsed();
    let sequential_ms = seq_elapsed.as_secs_f64() * 1000.0 / iterations as f64;

    // Benchmark parallel
    let par_start = Instant::now();
    for _ in 0..iterations {
        let mut output = Vec::new();
        parallel_dct_y_blocks(&strip, blocks_w, blocks_h, padded_width, None, &mut output);
        std::hint::black_box(&output);
    }
    let par_elapsed = par_start.elapsed();
    let parallel_ms = par_elapsed.as_secs_f64() * 1000.0 / iterations as f64;

    BenchResult {
        blocks: total_blocks,
        sequential_ms,
        parallel_ms,
        speedup: sequential_ms / parallel_ms,
    }
}

#[cfg(feature = "parallel")]
fn extract_block_sequential(
    strip: &[f32],
    block_x: usize,
    block_y: usize,
    padded_width: usize,
) -> zenjpeg::foundation::simd_types::Block8x8f {
    use zenjpeg::foundation::simd_types::Block8x8f;

    let start_x = block_x * 8;
    let start_y = block_y * 8;

    let mut block = Block8x8f::default();

    for row in 0..8 {
        let y = start_y + row;
        let row_start = y * padded_width + start_x;

        if row_start + 8 <= strip.len() {
            let arr: [f32; 8] = strip[row_start..row_start + 8].try_into().unwrap();
            block.rows[row] = arr;
        }
    }

    block
}

/// Simple DCT implementation for benchmarking (using public API)
#[cfg(feature = "parallel")]
fn dct_block(
    block: &zenjpeg::foundation::simd_types::Block8x8f,
) -> zenjpeg::foundation::simd_types::Block8x8f {
    // Use the dct module's public function
    let arr = block.to_array();
    let output = zenjpeg::encode::dct::forward_dct_8x8(&arr);
    zenjpeg::foundation::simd_types::Block8x8f::from_array(&output)
}
