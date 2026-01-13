//! Minimal standalone SIMD parity test for cross-platform benchmarking.
//! No external dependencies except `wide` and `multiversion`.

use std::time::Instant;
use wide::i32x8;
use multiversion::multiversion;

// ============================================================================
// Scalar IDCT (reference implementation)
// ============================================================================

mod scalar {
    #[inline(always)]
    const fn fsh(x: i32) -> i32 { x << 12 }

    #[inline(always)]
    fn clamp(a: i32) -> i16 { a.clamp(0, 255) as i16 }

    const SCALE_BITS: i32 = 512 + 65536 + (128 << 17);

    pub fn idct_int(input: &[i32; 64], output: &mut [i16; 64]) {
        let mut tmp = *input;

        // Vertical pass (columns)
        for ptr in 0..8 {
            let p2 = tmp[ptr + 16];
            let p3 = tmp[ptr + 48];
            let p1 = (p2 + p3).wrapping_mul(2217);
            let t2 = p1.wrapping_add(p3.wrapping_mul(-7567));
            let t3 = p1.wrapping_add(p2.wrapping_mul(3135));

            let p2 = tmp[ptr];
            let p3 = tmp[32 + ptr];
            let t0 = fsh(p2.wrapping_add(p3));
            let t1 = fsh(p2.wrapping_sub(p3));

            let x0 = t0.wrapping_add(t3).wrapping_add(512);
            let x3 = t0.wrapping_sub(t3).wrapping_add(512);
            let x1 = t1.wrapping_add(t2).wrapping_add(512);
            let x2 = t1.wrapping_sub(t2).wrapping_add(512);

            let mut t0 = tmp[ptr + 56];
            let mut t1 = tmp[ptr + 40];
            let mut t2 = tmp[ptr + 24];
            let mut t3 = tmp[ptr + 8];

            let p3 = t0.wrapping_add(t2);
            let p4 = t1.wrapping_add(t3);
            let p1 = t0.wrapping_add(t3);
            let p2 = t1.wrapping_add(t2);
            let p5 = (p3.wrapping_add(p4)).wrapping_mul(4816);

            t0 = t0.wrapping_mul(1223);
            t1 = t1.wrapping_mul(8410);
            t2 = t2.wrapping_mul(12586);
            t3 = t3.wrapping_mul(6149);

            let p1 = p5.wrapping_add(p1.wrapping_mul(-3685));
            let p2 = p5.wrapping_add(p2.wrapping_mul(-10497));
            let p3 = p3.wrapping_mul(-8034);
            let p4 = p4.wrapping_mul(-1597);

            t3 = t3.wrapping_add(p1).wrapping_add(p4);
            t2 = t2.wrapping_add(p2).wrapping_add(p3);
            t1 = t1.wrapping_add(p2).wrapping_add(p4);
            t0 = t0.wrapping_add(p1).wrapping_add(p3);

            tmp[ptr] = x0.wrapping_add(t3) >> 10;
            tmp[ptr + 8] = x1.wrapping_add(t2) >> 10;
            tmp[ptr + 16] = x2.wrapping_add(t1) >> 10;
            tmp[ptr + 24] = x3.wrapping_add(t0) >> 10;
            tmp[ptr + 32] = x3.wrapping_sub(t0) >> 10;
            tmp[ptr + 40] = x2.wrapping_sub(t1) >> 10;
            tmp[ptr + 48] = x1.wrapping_sub(t2) >> 10;
            tmp[ptr + 56] = x0.wrapping_sub(t3) >> 10;
        }

        // Horizontal pass (rows)
        for i in 0..8 {
            let base = i * 8;
            let p2 = tmp[base + 2];
            let p3 = tmp[base + 6];
            let p1 = (p2 + p3).wrapping_mul(2217);
            let t2 = p1.wrapping_add(p3.wrapping_mul(-7567));
            let t3 = p1.wrapping_add(p2.wrapping_mul(3135));

            let p2 = tmp[base];
            let p3 = tmp[base + 4];
            let t0 = fsh(p2.wrapping_add(p3));
            let t1 = fsh(p2.wrapping_sub(p3));

            let x0 = t0.wrapping_add(t3).wrapping_add(SCALE_BITS);
            let x3 = t0.wrapping_sub(t3).wrapping_add(SCALE_BITS);
            let x1 = t1.wrapping_add(t2).wrapping_add(SCALE_BITS);
            let x2 = t1.wrapping_sub(t2).wrapping_add(SCALE_BITS);

            let mut t0 = tmp[base + 7];
            let mut t1 = tmp[base + 5];
            let mut t2 = tmp[base + 3];
            let mut t3 = tmp[base + 1];

            let p3 = t0.wrapping_add(t2);
            let p4 = t1.wrapping_add(t3);
            let p1 = t0.wrapping_add(t3);
            let p2 = t1.wrapping_add(t2);
            let p5 = (p3.wrapping_add(p4)).wrapping_mul(4816);

            t0 = t0.wrapping_mul(1223);
            t1 = t1.wrapping_mul(8410);
            t2 = t2.wrapping_mul(12586);
            t3 = t3.wrapping_mul(6149);

            let p1 = p5.wrapping_add(p1.wrapping_mul(-3685));
            let p2 = p5.wrapping_add(p2.wrapping_mul(-10497));
            let p3 = p3.wrapping_mul(-8034);
            let p4 = p4.wrapping_mul(-1597);

            t3 = t3.wrapping_add(p1).wrapping_add(p4);
            t2 = t2.wrapping_add(p2).wrapping_add(p3);
            t1 = t1.wrapping_add(p2).wrapping_add(p4);
            t0 = t0.wrapping_add(p1).wrapping_add(p3);

            output[base] = clamp(x0.wrapping_add(t3) >> 17);
            output[base + 1] = clamp(x1.wrapping_add(t2) >> 17);
            output[base + 2] = clamp(x2.wrapping_add(t1) >> 17);
            output[base + 3] = clamp(x3.wrapping_add(t0) >> 17);
            output[base + 4] = clamp(x3.wrapping_sub(t0) >> 17);
            output[base + 5] = clamp(x2.wrapping_sub(t1) >> 17);
            output[base + 6] = clamp(x1.wrapping_sub(t2) >> 17);
            output[base + 7] = clamp(x0.wrapping_sub(t3) >> 17);
        }
    }
}

// ============================================================================
// Wide IDCT (portable SIMD using wide crate)
// ============================================================================

const SCALE_BITS: i32 = 512 + 65536 + (128 << 17);

#[multiversion(targets("x86_64+avx2", "aarch64+neon"))]
pub fn idct_wide(input: &[i32; 64], output: &mut [i16; 64]) {
    // Load 8 rows
    let mut rows: [i32x8; 8] = std::array::from_fn(|i| {
        i32x8::from(<[i32; 8]>::try_from(&input[i * 8..(i + 1) * 8]).unwrap())
    });

    // Vertical pass
    idct_pass(&mut rows, i32x8::splat(512), 10);

    // Transpose
    rows = i32x8::transpose(rows);

    // Horizontal pass
    idct_pass(&mut rows, i32x8::splat(SCALE_BITS), 17);

    // Transpose back
    rows = i32x8::transpose(rows);

    // Extract with clamping
    for (i, row) in rows.iter().enumerate() {
        let arr = row.to_array();
        for (j, &val) in arr.iter().enumerate() {
            output[i * 8 + j] = val.clamp(0, 255) as i16;
        }
    }
}

#[inline(always)]
fn idct_pass(rows: &mut [i32x8; 8], scale: i32x8, shift: i32) {
    // Even part
    let p1 = (rows[2] + rows[6]) * i32x8::splat(2217);
    let t2 = p1 + rows[6] * i32x8::splat(-7567);
    let t3 = p1 + rows[2] * i32x8::splat(3135);

    let t0 = (rows[0] + rows[4]) << 12;
    let t1 = (rows[0] - rows[4]) << 12;

    let x0 = t0 + t3 + scale;
    let x3 = t0 - t3 + scale;
    let x1 = t1 + t2 + scale;
    let x2 = t1 - t2 + scale;

    // Odd part
    let p3 = rows[7] + rows[3];
    let p4 = rows[5] + rows[1];
    let p1_odd = rows[7] + rows[1];
    let p2_odd = rows[5] + rows[3];
    let p5 = (p3 + p4) * i32x8::splat(4816);

    let mut t0 = rows[7] * i32x8::splat(1223);
    let mut t1 = rows[5] * i32x8::splat(8410);
    let mut t2 = rows[3] * i32x8::splat(12586);
    let mut t3 = rows[1] * i32x8::splat(6149);

    let p1_f = p5 + p1_odd * i32x8::splat(-3685);
    let p2_f = p5 + p2_odd * i32x8::splat(-10497);
    let p3_f = p3 * i32x8::splat(-8034);
    let p4_f = p4 * i32x8::splat(-1597);

    t3 = t3 + p1_f + p4_f;
    t2 = t2 + p2_f + p3_f;
    t1 = t1 + p2_f + p4_f;
    t0 = t0 + p1_f + p3_f;

    rows[0] = (x0 + t3) >> shift;
    rows[1] = (x1 + t2) >> shift;
    rows[2] = (x2 + t1) >> shift;
    rows[3] = (x3 + t0) >> shift;
    rows[4] = (x3 - t0) >> shift;
    rows[5] = (x2 - t1) >> shift;
    rows[6] = (x1 - t2) >> shift;
    rows[7] = (x0 - t3) >> shift;
}

fn main() {
    println!("SIMD IDCT Parity Test");
    println!("=====================");

    #[cfg(target_arch = "x86_64")]
    {
        println!("Platform: x86_64");
        println!("AVX2: {}", is_x86_feature_detected!("avx2"));
    }

    #[cfg(target_arch = "aarch64")]
    {
        println!("Platform: aarch64 (NEON assumed available)");
    }

    #[cfg(target_arch = "wasm32")]
    println!("Platform: wasm32");

    // Generate test input
    let input: [i32; 64] = std::array::from_fn(|i| {
        let v = ((i as i32 * 17 + 31) % 256) - 128;
        v * 8
    });

    let mut out_scalar = [0i16; 64];
    let mut out_wide = [0i16; 64];

    scalar::idct_int(&input, &mut out_scalar);
    idct_wide(&input, &mut out_wide);

    // Compare
    let mut diffs = 0;
    let mut max_diff = 0i16;
    for i in 0..64 {
        let d = (out_scalar[i] - out_wide[i]).abs();
        if d > 0 {
            diffs += 1;
            max_diff = max_diff.max(d);
        }
    }

    if diffs == 0 {
        println!("\n✓ Scalar and Wide MATCH exactly");
    } else {
        println!("\n✗ {}/64 values differ, max diff = {}", diffs, max_diff);
        println!("Scalar: {:?}", &out_scalar[0..16]);
        println!("Wide:   {:?}", &out_wide[0..16]);
    }

    // Benchmark (not available on WASM - no std::time::Instant)
    #[cfg(not(target_arch = "wasm32"))]
    {
        let iterations = 100_000;

        let start = Instant::now();
        for _ in 0..iterations {
            let mut out = [0i16; 64];
            scalar::idct_int(&input, &mut out);
            std::hint::black_box(&out);
        }
        let scalar_ns = start.elapsed().as_nanos() as f64 / iterations as f64;

        let start = Instant::now();
        for _ in 0..iterations {
            let mut out = [0i16; 64];
            idct_wide(&input, &mut out);
            std::hint::black_box(&out);
        }
        let wide_ns = start.elapsed().as_nanos() as f64 / iterations as f64;

        println!("\nBenchmark ({} iterations):", iterations);
        println!("  Scalar: {:.1} ns/block ({:.3} µs)", scalar_ns, scalar_ns / 1000.0);
        println!("  Wide:   {:.1} ns/block ({:.3} µs)", wide_ns, wide_ns / 1000.0);
        println!("  Speedup: {:.2}x", scalar_ns / wide_ns);
    }

    #[cfg(target_arch = "wasm32")]
    println!("\n(Benchmark skipped on WASM - no std::time::Instant)");
}
