//! WASM SIMD128 transpose benchmark and optimization
//!
//! Compares wide crate's f32x8::transpose (scalar fallback on WASM)
//! against explicit WASM SIMD128 intrinsics.
//!
//! Run with:
//! ```sh
//! CARGO_TARGET_WASM32_WASIP1_RUNNER="wasmtime --wasm simd" \
//! RUSTFLAGS="-C target-feature=+simd128" \
//! cargo run --release -p zenjpeg --example wasm_simd_transpose \
//!     --target wasm32-wasip1 --no-default-features --features std
//! ```

use std::time::Instant;

/// Wide crate transpose (uses scalar fallback on WASM)
fn transpose_wide(data: [wide::f32x8; 8]) -> [wide::f32x8; 8] {
    wide::f32x8::transpose(data)
}

/// WASM SIMD128 optimized 4x4 f32 transpose using shuffles
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn transpose_4x4_wasm(
    r0: core::arch::wasm32::v128,
    r1: core::arch::wasm32::v128,
    r2: core::arch::wasm32::v128,
    r3: core::arch::wasm32::v128,
) -> (
    core::arch::wasm32::v128,
    core::arch::wasm32::v128,
    core::arch::wasm32::v128,
    core::arch::wasm32::v128,
) {
    use core::arch::wasm32::*;

    // Phase 1: Interleave pairs
    // unpacklo(r0, r1) = [r0[0], r1[0], r0[1], r1[1]]
    // unpackhi(r0, r1) = [r0[2], r1[2], r0[3], r1[3]]
    let a0 = i32x4_shuffle::<0, 4, 1, 5>(r0, r1); // [r0[0], r1[0], r0[1], r1[1]]
    let a1 = i32x4_shuffle::<2, 6, 3, 7>(r0, r1); // [r0[2], r1[2], r0[3], r1[3]]
    let a2 = i32x4_shuffle::<0, 4, 1, 5>(r2, r3); // [r2[0], r3[0], r2[1], r3[1]]
    let a3 = i32x4_shuffle::<2, 6, 3, 7>(r2, r3); // [r2[2], r3[2], r2[3], r3[3]]

    // Phase 2: Interleave quads
    let t0 = i32x4_shuffle::<0, 1, 4, 5>(a0, a2); // [r0[0], r1[0], r2[0], r3[0]] = col 0
    let t1 = i32x4_shuffle::<2, 3, 6, 7>(a0, a2); // [r0[1], r1[1], r2[1], r3[1]] = col 1
    let t2 = i32x4_shuffle::<0, 1, 4, 5>(a1, a3); // [r0[2], r1[2], r2[2], r3[2]] = col 2
    let t3 = i32x4_shuffle::<2, 3, 6, 7>(a1, a3); // [r0[3], r1[3], r2[3], r3[3]] = col 3

    (t0, t1, t2, t3)
}

/// WASM SIMD128 optimized 8x8 f32 transpose
///
/// Since WASM only has 128-bit vectors (4 floats), we process as 4 quadrants:
/// ```text
/// [A B]    [A^T C^T]
/// [C D] -> [B^T D^T]
/// ```
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn transpose_8x8_wasm(input: &[f32; 64], output: &mut [f32; 64]) {
    use core::arch::wasm32::*;

    // Load 8 rows as pairs of v128 (low 4, high 4)
    let r0_lo = unsafe { v128_load(input.as_ptr().add(0) as *const v128) };
    let r0_hi = unsafe { v128_load(input.as_ptr().add(4) as *const v128) };
    let r1_lo = unsafe { v128_load(input.as_ptr().add(8) as *const v128) };
    let r1_hi = unsafe { v128_load(input.as_ptr().add(12) as *const v128) };
    let r2_lo = unsafe { v128_load(input.as_ptr().add(16) as *const v128) };
    let r2_hi = unsafe { v128_load(input.as_ptr().add(20) as *const v128) };
    let r3_lo = unsafe { v128_load(input.as_ptr().add(24) as *const v128) };
    let r3_hi = unsafe { v128_load(input.as_ptr().add(28) as *const v128) };
    let r4_lo = unsafe { v128_load(input.as_ptr().add(32) as *const v128) };
    let r4_hi = unsafe { v128_load(input.as_ptr().add(36) as *const v128) };
    let r5_lo = unsafe { v128_load(input.as_ptr().add(40) as *const v128) };
    let r5_hi = unsafe { v128_load(input.as_ptr().add(44) as *const v128) };
    let r6_lo = unsafe { v128_load(input.as_ptr().add(48) as *const v128) };
    let r6_hi = unsafe { v128_load(input.as_ptr().add(52) as *const v128) };
    let r7_lo = unsafe { v128_load(input.as_ptr().add(56) as *const v128) };
    let r7_hi = unsafe { v128_load(input.as_ptr().add(60) as *const v128) };

    // Transpose quadrant A (top-left 4x4 from rows 0-3, cols 0-3)
    let (a0, a1, a2, a3) = transpose_4x4_wasm(r0_lo, r1_lo, r2_lo, r3_lo);

    // Transpose quadrant B (top-right 4x4 from rows 0-3, cols 4-7)
    let (b0, b1, b2, b3) = transpose_4x4_wasm(r0_hi, r1_hi, r2_hi, r3_hi);

    // Transpose quadrant C (bottom-left 4x4 from rows 4-7, cols 0-3)
    let (c0, c1, c2, c3) = transpose_4x4_wasm(r4_lo, r5_lo, r6_lo, r7_lo);

    // Transpose quadrant D (bottom-right 4x4 from rows 4-7, cols 4-7)
    let (d0, d1, d2, d3) = transpose_4x4_wasm(r4_hi, r5_hi, r6_hi, r7_hi);

    // Store transposed: A^T goes to out rows 0-3 cols 0-3, C^T goes to out rows 0-3 cols 4-7
    // etc.
    unsafe {
        // Row 0: [A^T row 0] [C^T row 0]
        v128_store(output.as_mut_ptr().add(0) as *mut v128, a0);
        v128_store(output.as_mut_ptr().add(4) as *mut v128, c0);
        // Row 1
        v128_store(output.as_mut_ptr().add(8) as *mut v128, a1);
        v128_store(output.as_mut_ptr().add(12) as *mut v128, c1);
        // Row 2
        v128_store(output.as_mut_ptr().add(16) as *mut v128, a2);
        v128_store(output.as_mut_ptr().add(20) as *mut v128, c2);
        // Row 3
        v128_store(output.as_mut_ptr().add(24) as *mut v128, a3);
        v128_store(output.as_mut_ptr().add(28) as *mut v128, c3);
        // Row 4: [B^T row 0] [D^T row 0]
        v128_store(output.as_mut_ptr().add(32) as *mut v128, b0);
        v128_store(output.as_mut_ptr().add(36) as *mut v128, d0);
        // Row 5
        v128_store(output.as_mut_ptr().add(40) as *mut v128, b1);
        v128_store(output.as_mut_ptr().add(44) as *mut v128, d1);
        // Row 6
        v128_store(output.as_mut_ptr().add(48) as *mut v128, b2);
        v128_store(output.as_mut_ptr().add(52) as *mut v128, d2);
        // Row 7
        v128_store(output.as_mut_ptr().add(56) as *mut v128, b3);
        v128_store(output.as_mut_ptr().add(60) as *mut v128, d3);
    }
}

/// Scalar transpose for comparison
fn transpose_scalar(input: &[f32; 64], output: &mut [f32; 64]) {
    for row in 0..8 {
        for col in 0..8 {
            output[col * 8 + row] = input[row * 8 + col];
        }
    }
}

fn main() {
    println!("WASM SIMD128 Transpose Benchmark");
    println!("=================================\n");

    #[cfg(target_feature = "simd128")]
    println!("Mode: WASM SIMD128 enabled");
    #[cfg(not(target_feature = "simd128"))]
    println!("Mode: WASM scalar (no SIMD)");

    // Create test data
    let mut input = [0.0f32; 64];
    for i in 0..64 {
        input[i] = i as f32;
    }

    let iterations = 100_000;

    // Benchmark scalar
    let mut output_scalar = [0.0f32; 64];
    let start = Instant::now();
    for _ in 0..iterations {
        transpose_scalar(&input, &mut output_scalar);
        std::hint::black_box(&output_scalar);
    }
    let scalar_time = start.elapsed();
    println!(
        "Scalar:     {:?} for {} iterations ({:.2} ns/op)",
        scalar_time,
        iterations,
        scalar_time.as_nanos() as f64 / iterations as f64
    );

    // Benchmark wide crate
    let start = Instant::now();
    for _ in 0..iterations {
        let rows = [
            wide::f32x8::from(<[f32; 8]>::try_from(&input[0..8]).unwrap()),
            wide::f32x8::from(<[f32; 8]>::try_from(&input[8..16]).unwrap()),
            wide::f32x8::from(<[f32; 8]>::try_from(&input[16..24]).unwrap()),
            wide::f32x8::from(<[f32; 8]>::try_from(&input[24..32]).unwrap()),
            wide::f32x8::from(<[f32; 8]>::try_from(&input[32..40]).unwrap()),
            wide::f32x8::from(<[f32; 8]>::try_from(&input[40..48]).unwrap()),
            wide::f32x8::from(<[f32; 8]>::try_from(&input[48..56]).unwrap()),
            wide::f32x8::from(<[f32; 8]>::try_from(&input[56..64]).unwrap()),
        ];
        let transposed = transpose_wide(rows);
        std::hint::black_box(&transposed);
    }
    let wide_time = start.elapsed();
    println!(
        "Wide crate: {:?} for {} iterations ({:.2} ns/op)",
        wide_time,
        iterations,
        wide_time.as_nanos() as f64 / iterations as f64
    );

    // Benchmark WASM SIMD intrinsics
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        let mut output_wasm = [0.0f32; 64];
        let start = Instant::now();
        for _ in 0..iterations {
            transpose_8x8_wasm(&input, &mut output_wasm);
            std::hint::black_box(&output_wasm);
        }
        let wasm_time = start.elapsed();
        println!(
            "WASM SIMD:  {:?} for {} iterations ({:.2} ns/op)",
            wasm_time,
            iterations,
            wasm_time.as_nanos() as f64 / iterations as f64
        );

        // Verify correctness
        transpose_8x8_wasm(&input, &mut output_wasm);
        for row in 0..8 {
            for col in 0..8 {
                let expected = input[row * 8 + col];
                let got = output_wasm[col * 8 + row];
                assert!(
                    (expected - got).abs() < 1e-6,
                    "Mismatch at [{},{}]: expected {} got {}",
                    row,
                    col,
                    expected,
                    got
                );
            }
        }
        println!("\nCorrectness: PASSED");

        println!("\nSpeedup vs scalar: {:.2}x", scalar_time.as_nanos() as f64 / wasm_time.as_nanos() as f64);
        println!("Speedup vs wide:   {:.2}x", wide_time.as_nanos() as f64 / wasm_time.as_nanos() as f64);
    }

    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        println!("\nNote: WASM SIMD benchmark not available (not WASM or simd128 not enabled)");
    }
}
