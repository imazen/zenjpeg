//! WASM SIMD128 DCT benchmark
//!
//! Profiles the forward DCT to identify bottlenecks on WASM.
//!
//! Run with:
//! ```sh
//! CARGO_TARGET_WASM32_WASIP1_RUNNER="wasmtime --wasm simd" \
//! RUSTFLAGS="-C target-feature=+simd128" \
//! cargo run --release -p zenjpeg --example wasm_dct_bench \
//!     --target wasm32-wasip1 --no-default-features --features std
//! ```

use std::time::Instant;
use wide::f32x8;

/// 8x8 block of f32 values, stored as 8 SIMD vectors
#[derive(Clone, Copy)]
#[repr(C, align(32))]
struct Block8x8f {
    rows: [f32x8; 8],
}

impl Default for Block8x8f {
    fn default() -> Self {
        Self { rows: [f32x8::ZERO; 8] }
    }
}

// AAN DCT constants
const C4: f32 = 0.707106781;      // cos(π/4) = √2/2
const C6: f32 = 0.382683433;      // sin(π/8)
const C2_M_C6: f32 = 0.541196100; // cos(π/8) - cos(3π/8)
const C2_P_C6: f32 = 1.306562965; // cos(π/8) + cos(3π/8)

/// AAN 1D DCT on 8 f32 values (SIMD version using f32x8)
#[inline(always)]
fn aan_dct_1d_simd(input: f32x8) -> f32x8 {
    let arr = input.to_array();

    // Stage 1: butterfly
    let tmp0 = arr[0] + arr[7];
    let tmp7 = arr[0] - arr[7];
    let tmp1 = arr[1] + arr[6];
    let tmp6 = arr[1] - arr[6];
    let tmp2 = arr[2] + arr[5];
    let tmp5 = arr[2] - arr[5];
    let tmp3 = arr[3] + arr[4];
    let tmp4 = arr[3] - arr[4];

    // Stage 2
    let tmp10 = tmp0 + tmp3;
    let tmp13 = tmp0 - tmp3;
    let tmp11 = tmp1 + tmp2;
    let tmp12 = tmp1 - tmp2;

    // Output for positions 0, 4
    let out0 = tmp10 + tmp11;
    let out4 = tmp10 - tmp11;

    // Output for positions 2, 6
    let z1 = (tmp12 + tmp13) * C4;
    let out2 = tmp13 + z1;
    let out6 = tmp13 - z1;

    // Odd part
    let z1_odd = tmp4 + tmp7;
    let z2 = tmp5 + tmp6;
    let z3 = tmp4 + tmp6;
    let z4 = tmp5 + tmp7;
    let z5 = (z3 + z4) * 1.175875602;

    let tmp4_s = tmp4 * 0.298631336;
    let tmp5_s = tmp5 * 2.053119869;
    let tmp6_s = tmp6 * 3.072711026;
    let tmp7_s = tmp7 * 1.501321110;
    let z1_s = z1_odd * (-0.899976223);
    let z2_s = z2 * (-2.562915447);
    let z3_s = z3 * (-1.961570560);
    let z4_s = z4 * (-0.390180644);

    let z3_final = z3_s + z5;
    let z4_final = z4_s + z5;

    let out7 = tmp4_s + z1_s + z3_final;
    let out5 = tmp5_s + z2_s + z4_final;
    let out3 = tmp6_s + z2_s + z3_final;
    let out1 = tmp7_s + z1_s + z4_final;

    f32x8::new([out0, out1, out2, out3, out4, out5, out6, out7])
}

/// Full 8x8 DCT using wide crate
fn forward_dct_8x8_wide(block: &Block8x8f) -> Block8x8f {
    // Apply 1D DCT to rows
    let mut temp = Block8x8f::default();
    for i in 0..8 {
        temp.rows[i] = aan_dct_1d_simd(block.rows[i]);
    }

    // Transpose
    let transposed_rows = f32x8::transpose(temp.rows);

    // Apply 1D DCT to columns (which are now rows after transpose)
    let mut result = Block8x8f::default();
    for i in 0..8 {
        result.rows[i] = aan_dct_1d_simd(transposed_rows[i]);
    }

    // Transpose back
    result.rows = f32x8::transpose(result.rows);

    // Scale by 1/8 (per pass, so 1/64 total)
    let scale = f32x8::splat(1.0 / 8.0);
    for row in result.rows.iter_mut() {
        *row = *row * scale;
    }

    result
}

/// Benchmark transpose alone
fn bench_transpose(iterations: usize) -> std::time::Duration {
    let block = Block8x8f::default();
    let start = Instant::now();
    for _ in 0..iterations {
        let transposed = f32x8::transpose(block.rows);
        std::hint::black_box(&transposed);
    }
    start.elapsed()
}

/// Benchmark 1D DCT alone
fn bench_dct_1d(iterations: usize) -> std::time::Duration {
    let row = f32x8::splat(1.0);
    let start = Instant::now();
    for _ in 0..iterations {
        let result = aan_dct_1d_simd(row);
        std::hint::black_box(&result);
    }
    start.elapsed()
}

/// Benchmark full 8x8 DCT
fn bench_dct_8x8(iterations: usize) -> std::time::Duration {
    let mut block = Block8x8f::default();
    for i in 0..8 {
        block.rows[i] = f32x8::new([
            (i * 8) as f32, (i * 8 + 1) as f32, (i * 8 + 2) as f32, (i * 8 + 3) as f32,
            (i * 8 + 4) as f32, (i * 8 + 5) as f32, (i * 8 + 6) as f32, (i * 8 + 7) as f32,
        ]);
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let result = forward_dct_8x8_wide(&block);
        std::hint::black_box(&result);
    }
    start.elapsed()
}

fn main() {
    println!("WASM SIMD128 DCT Benchmark");
    println!("==========================\n");

    #[cfg(target_feature = "simd128")]
    println!("Mode: WASM SIMD128 enabled");
    #[cfg(not(target_feature = "simd128"))]
    println!("Mode: WASM scalar (no SIMD)");

    let iterations = 100_000;

    // Benchmark transpose
    let transpose_time = bench_transpose(iterations);
    println!(
        "Transpose (8x8):  {:?} for {} iterations ({:.2} ns/op)",
        transpose_time,
        iterations,
        transpose_time.as_nanos() as f64 / iterations as f64
    );

    // Benchmark 1D DCT
    let dct_1d_time = bench_dct_1d(iterations);
    println!(
        "1D DCT (8 vals):  {:?} for {} iterations ({:.2} ns/op)",
        dct_1d_time,
        iterations,
        dct_1d_time.as_nanos() as f64 / iterations as f64
    );

    // Benchmark full 8x8 DCT
    let dct_8x8_time = bench_dct_8x8(iterations);
    println!(
        "Full 8x8 DCT:     {:?} for {} iterations ({:.2} ns/op)",
        dct_8x8_time,
        iterations,
        dct_8x8_time.as_nanos() as f64 / iterations as f64
    );

    // Calculate breakdown
    println!("\n--- Breakdown ---");
    let transpose_pct = (transpose_time.as_nanos() as f64 * 2.0) / dct_8x8_time.as_nanos() as f64 * 100.0;
    let dct_1d_pct = (dct_1d_time.as_nanos() as f64 * 16.0) / dct_8x8_time.as_nanos() as f64 * 100.0;
    println!("Transpose contribution (2x): ~{:.1}%", transpose_pct);
    println!("1D DCT contribution (16x):   ~{:.1}%", dct_1d_pct);

    // Throughput
    let blocks_per_sec = iterations as f64 / dct_8x8_time.as_secs_f64();
    let pixels_per_sec = blocks_per_sec * 64.0;
    println!("\nThroughput:");
    println!("  {:.2} M blocks/sec", blocks_per_sec / 1_000_000.0);
    println!("  {:.2} MP/s (DCT only)", pixels_per_sec / 1_000_000.0);
}
