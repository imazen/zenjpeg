//! Experiment: Auto-vectorization patterns for 8x8 transpose
//!
//! Testing if scalar code written in specific patterns can be auto-vectorized
//! by the compiler to match explicit SIMD assembly.
//!
//! Key finding: With `-C target-cpu=x86-64-v3`, the compiler autovectorizes
//! even the naive transpose to use `vunpcklps`, `vinsertf128`, `vshufps` - the
//! exact same instructions as manual SIMD!
//!
//! The `multiversion` crate enables this at runtime without global target flags.
//!
//! Run with: cargo asm -p zenjpeg --example autovec_transpose --release transpose_naive
//! Compare:  cargo asm -p zenjpeg --example autovec_transpose --release transpose_multiversion

#![allow(dead_code)]

use multiversion::multiversion;

/// Naive scalar transpose - baseline (won't vectorize)
#[inline(never)]
pub fn transpose_naive(input: &[f32; 64], output: &mut [f32; 64]) {
    for row in 0..8 {
        for col in 0..8 {
            output[col * 8 + row] = input[row * 8 + col];
        }
    }
}

/// Pattern 1: Explicit row-by-row with struct-like access
/// Uses array-of-arrays to hint at memory layout
#[repr(C)]
#[derive(Copy, Clone)]
pub struct Row8([f32; 8]);

#[repr(C)]
pub struct Matrix8x8 {
    rows: [Row8; 8],
}

impl Matrix8x8 {
    #[inline(always)]
    fn from_slice(data: &[f32; 64]) -> &Self {
        // SAFETY: [f32; 64] has same layout as [[f32; 8]; 8] with #[repr(C)]
        unsafe { &*(data.as_ptr() as *const Self) }
    }

    #[inline(always)]
    fn from_slice_mut(data: &mut [f32; 64]) -> &mut Self {
        unsafe { &mut *(data.as_mut_ptr() as *mut Self) }
    }
}

/// Pattern 2: Structured types with clear memory access
#[inline(never)]
pub fn transpose_structured(input: &[f32; 64], output: &mut [f32; 64]) {
    let src = Matrix8x8::from_slice(input);
    let dst = Matrix8x8::from_slice_mut(output);

    // Process each output column (= input row) as a unit
    for row in 0..8 {
        for col in 0..8 {
            dst.rows[col].0[row] = src.rows[row].0[col];
        }
    }
}

/// Pattern 3: Chunk-based with explicit 8-element groups
/// This matches AVX2 register width
#[inline(never)]
pub fn transpose_chunked(input: &[f32; 64], output: &mut [f32; 64]) {
    // Process 8 elements at a time (one row → one column)
    let input_rows: &[[f32; 8]; 8] = bytemuck::cast_ref(input);
    let output_rows: &mut [[f32; 8]; 8] = bytemuck::cast_mut(output);

    // Each iteration writes one column of output
    for src_row_idx in 0..8 {
        let src_row = &input_rows[src_row_idx];
        // Scatter src_row[i] to output_rows[i][src_row_idx]
        for i in 0..8 {
            output_rows[i][src_row_idx] = src_row[i];
        }
    }
}

/// Pattern 4: Explicit 4x4 blocks (like AVX2 128-bit lanes)
/// This is how the actual AVX2 transpose works internally
#[inline(never)]
pub fn transpose_4x4_blocks(input: &[f32; 64], output: &mut [f32; 64]) {
    let src: &[[f32; 8]; 8] = bytemuck::cast_ref(input);
    let dst: &mut [[f32; 8]; 8] = bytemuck::cast_mut(output);

    // Transpose as four 4x4 sub-matrices
    // Top-left 4x4: src[0..4][0..4] → dst[0..4][0..4]
    for i in 0..4 {
        for j in 0..4 {
            dst[j][i] = src[i][j];
        }
    }

    // Top-right 4x4: src[0..4][4..8] → dst[4..8][0..4]
    for i in 0..4 {
        for j in 0..4 {
            dst[j + 4][i] = src[i][j + 4];
        }
    }

    // Bottom-left 4x4: src[4..8][0..4] → dst[0..4][4..8]
    for i in 0..4 {
        for j in 0..4 {
            dst[j][i + 4] = src[i + 4][j];
        }
    }

    // Bottom-right 4x4: src[4..8][4..8] → dst[4..8][4..8]
    for i in 0..4 {
        for j in 0..4 {
            dst[j + 4][i + 4] = src[i + 4][j + 4];
        }
    }
}

/// Pattern 5: Fully unrolled with explicit indices
/// Compiler might recognize this as a permutation pattern
#[inline(never)]
pub fn transpose_unrolled(input: &[f32; 64], output: &mut [f32; 64]) {
    // Row 0 → Column 0
    output[0] = input[0];
    output[8] = input[1];
    output[16] = input[2];
    output[24] = input[3];
    output[32] = input[4];
    output[40] = input[5];
    output[48] = input[6];
    output[56] = input[7];

    // Row 1 → Column 1
    output[1] = input[8];
    output[9] = input[9];
    output[17] = input[10];
    output[25] = input[11];
    output[33] = input[12];
    output[41] = input[13];
    output[49] = input[14];
    output[57] = input[15];

    // Row 2 → Column 2
    output[2] = input[16];
    output[10] = input[17];
    output[18] = input[18];
    output[26] = input[19];
    output[34] = input[20];
    output[42] = input[21];
    output[50] = input[22];
    output[58] = input[23];

    // Row 3 → Column 3
    output[3] = input[24];
    output[11] = input[25];
    output[19] = input[26];
    output[27] = input[27];
    output[35] = input[28];
    output[43] = input[29];
    output[51] = input[30];
    output[59] = input[31];

    // Row 4 → Column 4
    output[4] = input[32];
    output[12] = input[33];
    output[20] = input[34];
    output[28] = input[35];
    output[36] = input[36];
    output[44] = input[37];
    output[52] = input[38];
    output[60] = input[39];

    // Row 5 → Column 5
    output[5] = input[40];
    output[13] = input[41];
    output[21] = input[42];
    output[29] = input[43];
    output[37] = input[44];
    output[45] = input[45];
    output[53] = input[46];
    output[61] = input[47];

    // Row 6 → Column 6
    output[6] = input[48];
    output[14] = input[49];
    output[22] = input[50];
    output[30] = input[51];
    output[38] = input[52];
    output[46] = input[53];
    output[54] = input[54];
    output[62] = input[55];

    // Row 7 → Column 7
    output[7] = input[56];
    output[15] = input[57];
    output[23] = input[58];
    output[31] = input[59];
    output[39] = input[60];
    output[47] = input[61];
    output[55] = input[62];
    output[63] = input[63];
}

/// Pattern 6: In-place transpose using swaps (like the register-based version)
/// Process as pairs that need to be exchanged
#[inline(never)]
pub fn transpose_inplace(data: &mut [f32; 64]) {
    let m: &mut [[f32; 8]; 8] = bytemuck::cast_mut(data);

    // Swap elements above/below diagonal
    for i in 0..8 {
        for j in (i + 1)..8 {
            // Swap m[i][j] with m[j][i]
            let tmp = m[i][j];
            m[i][j] = m[j][i];
            m[j][i] = tmp;
        }
    }
}

/// Pattern 7: Using std::mem::swap (compiler might optimize better)
#[inline(never)]
pub fn transpose_memswap(data: &mut [f32; 64]) {
    let m: &mut [[f32; 8]; 8] = bytemuck::cast_mut(data);

    for i in 0..8 {
        for j in (i + 1)..8 {
            // Use a tuple swap which the compiler can often optimize well
            let (row_i, row_j) = m.split_at_mut(j);
            std::mem::swap(&mut row_i[i][j], &mut row_j[0][i]);
        }
    }
}

/// Pattern 8: Multiversion - compiles multiple versions, picks best at runtime
/// This is the KEY technique: same scalar code, but compiled for multiple targets.
#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "x86_64+sse4.1", "aarch64+neon"))]
#[inline(never)]
pub fn transpose_multiversion(input: &[f32; 64], output: &mut [f32; 64]) {
    // Same naive code, but multiversion compiles separate AVX2/AVX/SSE versions
    for row in 0..8 {
        for col in 0..8 {
            output[col * 8 + row] = input[row * 8 + col];
        }
    }
}

/// Pattern 9: Multiversion with explicit chunking hint
#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "x86_64+sse4.1", "aarch64+neon"))]
#[inline(never)]
pub fn transpose_multiversion_chunked(input: &[f32; 64], output: &mut [f32; 64]) {
    let input_rows: &[[f32; 8]; 8] = bytemuck::cast_ref(input);
    let output_rows: &mut [[f32; 8]; 8] = bytemuck::cast_mut(output);

    for src_row_idx in 0..8 {
        let src_row = &input_rows[src_row_idx];
        for i in 0..8 {
            output_rows[i][src_row_idx] = src_row[i];
        }
    }
}

fn main() {
    use std::time::Instant;

    const ITERS: usize = 10_000_000;

    let input: [f32; 64] = core::array::from_fn(|i| i as f32);
    let mut output = [0.0f32; 64];

    // Verify all implementations produce the same result
    let mut expected = [0.0f32; 64];
    transpose_naive(&input, &mut expected);

    transpose_structured(&input, &mut output);
    assert_eq!(output, expected, "structured mismatch");

    transpose_chunked(&input, &mut output);
    assert_eq!(output, expected, "chunked mismatch");

    transpose_4x4_blocks(&input, &mut output);
    assert_eq!(output, expected, "4x4 blocks mismatch");

    transpose_unrolled(&input, &mut output);
    assert_eq!(output, expected, "unrolled mismatch");

    let mut inplace_data = input;
    transpose_inplace(&mut inplace_data);
    assert_eq!(inplace_data, expected, "inplace mismatch");

    let mut memswap_data = input;
    transpose_memswap(&mut memswap_data);
    assert_eq!(memswap_data, expected, "memswap mismatch");

    println!("All implementations verified correct.\n");

    // Verify multiversion
    transpose_multiversion(&input, &mut output);
    assert_eq!(output, expected, "multiversion mismatch");

    transpose_multiversion_chunked(&input, &mut output);
    assert_eq!(output, expected, "multiversion_chunked mismatch");

    // Benchmark each
    let benches: &[(&str, fn(&[f32; 64], &mut [f32; 64]))] = &[
        ("naive", transpose_naive),
        ("structured", transpose_structured),
        ("chunked", transpose_chunked),
        ("4x4_blocks", transpose_4x4_blocks),
        ("unrolled", transpose_unrolled),
        ("multiversion", transpose_multiversion),
        ("mv_chunked", transpose_multiversion_chunked),
    ];

    for (name, func) in benches {
        let start = Instant::now();
        for _ in 0..ITERS {
            func(&input, &mut output);
            std::hint::black_box(&output);
        }
        let elapsed = start.elapsed();
        let ns_per_op = elapsed.as_nanos() as f64 / ITERS as f64;
        println!("{:15}: {:.2} ns/transpose", name, ns_per_op);
    }

    // In-place benchmarks
    println!();
    let start = Instant::now();
    for _ in 0..ITERS {
        let mut data = input;
        transpose_inplace(&mut data);
        std::hint::black_box(&data);
    }
    let elapsed = start.elapsed();
    println!(
        "{:15}: {:.2} ns/transpose",
        "inplace",
        elapsed.as_nanos() as f64 / ITERS as f64
    );

    let start = Instant::now();
    for _ in 0..ITERS {
        let mut data = input;
        transpose_memswap(&mut data);
        std::hint::black_box(&data);
    }
    let elapsed = start.elapsed();
    println!(
        "{:15}: {:.2} ns/transpose",
        "memswap",
        elapsed.as_nanos() as f64 / ITERS as f64
    );

    println!("\nTo compare assembly:");
    println!("  cargo asm -p zenjpeg --example autovec_transpose --release transpose_naive");
    println!("  cargo asm -p zenjpeg --example autovec_transpose --release transpose_chunked");
}
