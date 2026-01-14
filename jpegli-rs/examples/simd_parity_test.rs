//! Side-by-side testing of Intel-specific SIMD vs portable implementations.
//!
//! This example tests parity between:
//! 1. AVX2/SSE intrinsics implementations
//! 2. Scalar fallback implementations
//! 3. `wide` crate portable SIMD implementations (future)
//!
//! Run with: cargo run --release --example simd_parity_test
//!
//! Tests are run OUTSIDE multiversioned functions for direct comparison.

use std::time::Instant;

// ============================================================================
// Test 1: Integer IDCT (decode hot path - 40% of decode time)
// ============================================================================

/// Scalar IDCT reference (copied from idct_int.rs for standalone testing)
mod idct_scalar {
    const SCALE_BITS: i32 = 512 + 65536 + (128 << 17);

    #[inline]
    const fn fsh(x: i32) -> i32 {
        x << 12
    }

    #[inline]
    fn clamp(a: i32) -> i16 {
        a.clamp(0, 255) as i16
    }

    #[inline(always)]
    const fn wa(a: i32, b: i32) -> i32 {
        a.wrapping_add(b)
    }

    #[inline(always)]
    const fn ws(a: i32, b: i32) -> i32 {
        a.wrapping_sub(b)
    }

    #[inline(always)]
    const fn wm(a: i32, b: i32) -> i32 {
        a.wrapping_mul(b)
    }

    pub fn idct_int(in_vector: &mut [i32; 64], out_vector: &mut [i16; 64]) {
        // Vertical pass (columns)
        for ptr in 0..8 {
            let p2 = in_vector[ptr + 16];
            let p3 = in_vector[ptr + 48];

            let p1 = wm(wa(p2, p3), 2217);

            let t2 = wa(p1, wm(p3, -7567));
            let t3 = wa(p1, wm(p2, 3135));

            let p2 = in_vector[ptr];
            let p3 = in_vector[32 + ptr];

            let t0 = fsh(wa(p2, p3));
            let t1 = fsh(ws(p2, p3));

            let x0 = wa(wa(t0, t3), 512);
            let x3 = wa(ws(t0, t3), 512);
            let x1 = wa(wa(t1, t2), 512);
            let x2 = wa(ws(t1, t2), 512);

            let mut t0 = in_vector[ptr + 56];
            let mut t1 = in_vector[ptr + 40];
            let mut t2 = in_vector[ptr + 24];
            let mut t3 = in_vector[ptr + 8];

            let p3 = wa(t0, t2);
            let p4 = wa(t1, t3);
            let p1 = wa(t0, t3);
            let p2 = wa(t1, t2);
            let p5 = wm(wa(p3, p4), 4816);

            t0 = wm(t0, 1223);
            t1 = wm(t1, 8410);
            t2 = wm(t2, 12586);
            t3 = wm(t3, 6149);

            let p1 = wa(p5, wm(p1, -3685));
            let p2 = wa(p5, wm(p2, -10497));
            let p3 = wm(p3, -8034);
            let p4 = wm(p4, -1597);

            t3 = wa(t3, wa(p1, p4));
            t2 = wa(t2, wa(p2, p3));
            t1 = wa(t1, wa(p2, p4));
            t0 = wa(t0, wa(p1, p3));

            in_vector[ptr] = wa(x0, t3) >> 10;
            in_vector[ptr + 8] = wa(x1, t2) >> 10;
            in_vector[ptr + 16] = wa(x2, t1) >> 10;
            in_vector[ptr + 24] = wa(x3, t0) >> 10;
            in_vector[ptr + 32] = ws(x3, t0) >> 10;
            in_vector[ptr + 40] = ws(x2, t1) >> 10;
            in_vector[ptr + 48] = ws(x1, t2) >> 10;
            in_vector[ptr + 56] = ws(x0, t3) >> 10;
        }

        // Horizontal pass (rows)
        let mut pos = 0;
        for i in (0..64).step_by(8) {
            let p2 = in_vector[i + 2];
            let p3 = in_vector[i + 6];

            let p1 = wm(wa(p2, p3), 2217);
            let t2 = wa(p1, wm(p3, -7567));
            let t3 = wa(p1, wm(p2, 3135));

            let p2 = in_vector[i];
            let p3 = in_vector[i + 4];

            let t0 = fsh(wa(p2, p3));
            let t1 = fsh(ws(p2, p3));

            let x0 = wa(wa(t0, t3), SCALE_BITS);
            let x3 = wa(ws(t0, t3), SCALE_BITS);
            let x1 = wa(wa(t1, t2), SCALE_BITS);
            let x2 = wa(ws(t1, t2), SCALE_BITS);

            let mut t0 = in_vector[i + 7];
            let mut t1 = in_vector[i + 5];
            let mut t2 = in_vector[i + 3];
            let mut t3 = in_vector[i + 1];

            let p3 = wa(t0, t2);
            let p4 = wa(t1, t3);
            let p1 = wa(t0, t3);
            let p2 = wa(t1, t2);
            let p5 = wm(wa(p3, p4), 4816);

            t0 = wm(t0, 1223);
            t1 = wm(t1, 8410);
            t2 = wm(t2, 12586);
            t3 = wm(t3, 6149);

            let p1 = wa(p5, wm(p1, -3685));
            let p2 = wa(p5, wm(p2, -10497));
            let p3 = wm(p3, -8034);
            let p4 = wm(p4, -1597);

            t3 = wa(t3, wa(p1, p4));
            t2 = wa(t2, wa(p2, p3));
            t1 = wa(t1, wa(p2, p4));
            t0 = wa(t0, wa(p1, p3));

            out_vector[pos] = clamp(wa(x0, t3) >> 17);
            out_vector[pos + 1] = clamp(wa(x1, t2) >> 17);
            out_vector[pos + 2] = clamp(wa(x2, t1) >> 17);
            out_vector[pos + 3] = clamp(wa(x3, t0) >> 17);
            out_vector[pos + 4] = clamp(ws(x3, t0) >> 17);
            out_vector[pos + 5] = clamp(ws(x2, t1) >> 17);
            out_vector[pos + 6] = clamp(ws(x1, t2) >> 17);
            out_vector[pos + 7] = clamp(ws(x0, t3) >> 17);

            pos += 8;
        }
    }
}

/// Wide crate portable IDCT using wide::i32x8::transpose
mod idct_wide {
    use multiversion::multiversion;
    use wide::i32x8;

    const SCALE_BITS: i32 = 512 + 65536 + (128 << 17);

    /// DCT constants (same as scalar)
    const C2217: i32 = 2217;
    const C3135: i32 = 3135;
    const CN7567: i32 = -7567;
    const C4816: i32 = 4816;
    const C1223: i32 = 1223;
    const C8410: i32 = 8410;
    const C12586: i32 = 12586;
    const C6149: i32 = 6149;
    const CN3685: i32 = -3685;
    const CN10497: i32 = -10497;
    const CN8034: i32 = -8034;
    const CN1597: i32 = -1597;

    /// Portable IDCT using wide crate's i32x8
    /// Uses wide::i32x8::transpose which is AVX2-accelerated but has scalar fallback
    #[multiversion(targets("x86_64+avx2", "aarch64+neon"))]
    pub fn idct_int_wide(in_vector: &mut [i32; 64], out_vector: &mut [i16; 64]) {
        // Load 8 rows as i32x8 vectors
        let mut rows: [i32x8; 8] = std::array::from_fn(|i| {
            i32x8::from(<[i32; 8]>::try_from(&in_vector[i * 8..(i + 1) * 8]).unwrap())
        });

        // First pass (columns) - process all 8 columns in parallel
        idct_pass_wide(&mut rows, i32x8::splat(512), 10);

        // Transpose using wide's built-in (AVX2-accelerated, scalar fallback)
        rows = i32x8::transpose(rows);

        // Second pass (rows)
        idct_pass_wide(&mut rows, i32x8::splat(SCALE_BITS), 17);

        // Transpose back
        rows = i32x8::transpose(rows);

        // Extract and clamp to output
        for (row_idx, row) in rows.iter().enumerate() {
            let arr = row.to_array();
            for (col_idx, &val) in arr.iter().enumerate() {
                out_vector[row_idx * 8 + col_idx] = val.clamp(0, 255) as i16;
            }
        }
    }

    /// One pass of IDCT butterfly using i32x8 SIMD
    #[inline(always)]
    fn idct_pass_wide(rows: &mut [i32x8; 8], scale_bits: i32x8, shift: i32) {
        // Even part
        let p1 = (rows[2] + rows[6]) * i32x8::splat(C2217);
        let t2 = p1 + rows[6] * i32x8::splat(CN7567);
        let t3 = p1 + rows[2] * i32x8::splat(C3135);

        let t0 = (rows[0] + rows[4]) << 12;
        let t1 = (rows[0] - rows[4]) << 12;

        let x0 = t0 + t3 + scale_bits;
        let x3 = t0 - t3 + scale_bits;
        let x1 = t1 + t2 + scale_bits;
        let x2 = t1 - t2 + scale_bits;

        // Odd part
        let p3 = rows[7] + rows[3];
        let p4 = rows[5] + rows[1];
        let p1_odd = rows[7] + rows[1];
        let p2_odd = rows[5] + rows[3];
        let p5 = (p3 + p4) * i32x8::splat(C4816);

        let mut t0 = rows[7] * i32x8::splat(C1223);
        let mut t1 = rows[5] * i32x8::splat(C8410);
        let mut t2 = rows[3] * i32x8::splat(C12586);
        let mut t3 = rows[1] * i32x8::splat(C6149);

        let p1_final = p5 + p1_odd * i32x8::splat(CN3685);
        let p2_final = p5 + p2_odd * i32x8::splat(CN10497);
        let p3_final = p3 * i32x8::splat(CN8034);
        let p4_final = p4 * i32x8::splat(CN1597);

        t3 = t3 + p1_final + p4_final;
        t2 = t2 + p2_final + p3_final;
        t1 = t1 + p2_final + p4_final;
        t0 = t0 + p1_final + p3_final;

        // Combine and shift
        rows[0] = (x0 + t3) >> shift;
        rows[1] = (x1 + t2) >> shift;
        rows[2] = (x2 + t1) >> shift;
        rows[3] = (x3 + t0) >> shift;
        rows[4] = (x3 - t0) >> shift;
        rows[5] = (x2 - t1) >> shift;
        rows[6] = (x1 - t2) >> shift;
        rows[7] = (x0 - t3) >> shift;
    }
}

// ============================================================================
// Test 2: YCbCr→RGB i16 batch conversion (decode color conversion)
// ============================================================================

mod ycbcr_scalar {
    const Y_CF_INT: i32 = 16384;
    const CR_TO_R_INT: i32 = 22970;
    const CB_TO_B_INT: i32 = 29032;
    const CR_TO_G_INT: i32 = -11700;
    const CB_TO_G_INT: i32 = -5638;
    const YUV_ROUND: i32 = 8192;

    pub fn ycbcr_to_rgb_i16_x16(y: &[i16; 16], cb: &[i16; 16], cr: &[i16; 16], rgb: &mut [u8; 48]) {
        for i in 0..16 {
            let y_val = i32::from(y[i]);
            let cb_val = i32::from(cb[i]) - 128;
            let cr_val = i32::from(cr[i]) - 128;

            let y_scaled = y_val * Y_CF_INT + YUV_ROUND;

            let r = (y_scaled + cr_val * CR_TO_R_INT) >> 14;
            let g = (y_scaled + cr_val * CR_TO_G_INT + cb_val * CB_TO_G_INT) >> 14;
            let b = (y_scaled + cb_val * CB_TO_B_INT) >> 14;

            rgb[i * 3] = r.clamp(0, 255) as u8;
            rgb[i * 3 + 1] = g.clamp(0, 255) as u8;
            rgb[i * 3 + 2] = b.clamp(0, 255) as u8;
        }
    }
}

mod ycbcr_wide {
    use wide::i32x8;

    const Y_CF_INT: i32 = 16384;
    const CR_TO_R_INT: i32 = 22970;
    const CB_TO_B_INT: i32 = 29032;
    const CR_TO_G_INT: i32 = -11700;
    const CB_TO_G_INT: i32 = -5638;
    const YUV_ROUND: i32 = 8192;

    /// Portable YCbCr to RGB using wide crate i32x8
    /// Processes 8 pixels at a time (vs AVX2's 16)
    pub fn ycbcr_to_rgb_i16_x8(y: &[i16; 8], cb: &[i16; 8], cr: &[i16; 8], rgb: &mut [u8; 24]) {
        // Convert i16 to i32 (sign extension)
        let y_i32: [i32; 8] = std::array::from_fn(|i| y[i] as i32);
        let cb_i32: [i32; 8] = std::array::from_fn(|i| (cb[i] as i32) - 128);
        let cr_i32: [i32; 8] = std::array::from_fn(|i| (cr[i] as i32) - 128);

        let y_vec = i32x8::from(y_i32);
        let cb_vec = i32x8::from(cb_i32);
        let cr_vec = i32x8::from(cr_i32);

        // Coefficients
        let y_coeff = i32x8::splat(Y_CF_INT);
        let cr_to_r = i32x8::splat(CR_TO_R_INT);
        let cb_to_b = i32x8::splat(CB_TO_B_INT);
        let cr_to_g = i32x8::splat(CR_TO_G_INT);
        let cb_to_g = i32x8::splat(CB_TO_G_INT);
        let rounding = i32x8::splat(YUV_ROUND);

        // y_scaled = y * Y_CF + rounding
        let y_scaled = y_vec * y_coeff + rounding;

        // R = (y_scaled + cr * CR_TO_R) >> 14
        let r_raw = y_scaled + cr_vec * cr_to_r;
        let r_arr = r_raw.to_array();
        let r_arr: [i32; 8] = std::array::from_fn(|i| r_arr[i] >> 14);

        // G = (y_scaled + cr * CR_TO_G + cb * CB_TO_G) >> 14
        let g_raw = y_scaled + cr_vec * cr_to_g + cb_vec * cb_to_g;
        let g_arr = g_raw.to_array();
        let g_arr: [i32; 8] = std::array::from_fn(|i| g_arr[i] >> 14);

        // B = (y_scaled + cb * CB_TO_B) >> 14
        let b_raw = y_scaled + cb_vec * cb_to_b;
        let b_arr = b_raw.to_array();
        let b_arr: [i32; 8] = std::array::from_fn(|i| b_arr[i] >> 14);

        for i in 0..8 {
            rgb[i * 3] = r_arr[i].clamp(0, 255) as u8;
            rgb[i * 3 + 1] = g_arr[i].clamp(0, 255) as u8;
            rgb[i * 3 + 2] = b_arr[i].clamp(0, 255) as u8;
        }
    }

    /// Process 16 pixels by calling 8-pixel version twice
    pub fn ycbcr_to_rgb_i16_x16(y: &[i16; 16], cb: &[i16; 16], cr: &[i16; 16], rgb: &mut [u8; 48]) {
        // First 8 pixels
        let y0: [i16; 8] = y[0..8].try_into().unwrap();
        let cb0: [i16; 8] = cb[0..8].try_into().unwrap();
        let cr0: [i16; 8] = cr[0..8].try_into().unwrap();
        let mut rgb0 = [0u8; 24];
        ycbcr_to_rgb_i16_x8(&y0, &cb0, &cr0, &mut rgb0);
        rgb[0..24].copy_from_slice(&rgb0);

        // Second 8 pixels
        let y1: [i16; 8] = y[8..16].try_into().unwrap();
        let cb1: [i16; 8] = cb[8..16].try_into().unwrap();
        let cr1: [i16; 8] = cr[8..16].try_into().unwrap();
        let mut rgb1 = [0u8; 24];
        ycbcr_to_rgb_i16_x8(&y1, &cb1, &cr1, &mut rgb1);
        rgb[24..48].copy_from_slice(&rgb1);
    }
}

// ============================================================================
// Test 3: Even/Odd gather (chroma downsampling)
// ============================================================================

mod gather_scalar {
    /// Gather even and odd indexed elements from 16 consecutive floats
    pub fn gather_even_odd(data: &[f32; 16]) -> ([f32; 8], [f32; 8]) {
        let evens = [
            data[0], data[2], data[4], data[6], data[8], data[10], data[12], data[14],
        ];
        let odds = [
            data[1], data[3], data[5], data[7], data[9], data[11], data[13], data[15],
        ];
        (evens, odds)
    }
}

mod gather_wide {
    use wide::f32x8;

    /// Portable even/odd gather using wide crate
    /// AVX2 uses _mm256_shuffle_ps + _mm256_permute4x64_epi64
    /// Wide approach: construct from array indices
    pub fn gather_even_odd(data: &[f32; 16]) -> (f32x8, f32x8) {
        let evens = f32x8::from([
            data[0], data[2], data[4], data[6], data[8], data[10], data[12], data[14],
        ]);
        let odds = f32x8::from([
            data[1], data[3], data[5], data[7], data[9], data[11], data[13], data[15],
        ]);
        (evens, odds)
    }
}

// ============================================================================
// Test runners
// ============================================================================

fn test_idct_parity() {
    println!("\n=== IDCT Parity Test ===");

    // Generate test coefficients
    let original_coeffs: [i32; 64] = std::array::from_fn(|i| {
        let v = ((i as i32 * 17 + 31) % 256) - 128;
        v * 8
    });

    let mut coeffs_scalar = original_coeffs;
    let mut coeffs_wide = original_coeffs;

    let mut output_scalar = [0i16; 64];
    let mut output_wide = [0i16; 64];

    // Run both implementations
    idct_scalar::idct_int(&mut coeffs_scalar, &mut output_scalar);
    idct_wide::idct_int_wide(&mut coeffs_wide, &mut output_wide);

    // Compare outputs
    let mut max_diff = 0i16;
    let mut diff_count = 0;
    for i in 0..64 {
        let diff = (output_scalar[i] - output_wide[i]).abs();
        if diff > 0 {
            diff_count += 1;
            max_diff = max_diff.max(diff);
        }
    }

    if diff_count == 0 {
        println!("✓ Scalar and Wide IDCT MATCH exactly");
    } else {
        println!(
            "✗ Differences found: {}/64 values differ, max diff = {}",
            diff_count, max_diff
        );
    }

    println!("Scalar output (first 16): {:?}", &output_scalar[0..16]);
    println!("Wide output   (first 16): {:?}", &output_wide[0..16]);

    // Verify output range
    let min_scalar = output_scalar.iter().min().unwrap();
    let max_scalar = output_scalar.iter().max().unwrap();
    println!(
        "Output range: [{}, {}] (expected [0, 255])",
        min_scalar, max_scalar
    );

    // Benchmark both
    let iterations = 100_000;

    let start = Instant::now();
    for _ in 0..iterations {
        let mut coeffs = original_coeffs;
        let mut output = [0i16; 64];
        idct_scalar::idct_int(&mut coeffs, &mut output);
        std::hint::black_box(&output);
    }
    let scalar_time = start.elapsed();

    let start = Instant::now();
    for _ in 0..iterations {
        let mut coeffs = original_coeffs;
        let mut output = [0i16; 64];
        idct_wide::idct_int_wide(&mut coeffs, &mut output);
        std::hint::black_box(&output);
    }
    let wide_time = start.elapsed();

    println!(
        "Scalar IDCT: {:.3} µs/block",
        scalar_time.as_nanos() as f64 / iterations as f64 / 1000.0
    );
    println!(
        "Wide IDCT:   {:.3} µs/block",
        wide_time.as_nanos() as f64 / iterations as f64 / 1000.0
    );
    println!(
        "Wide speedup: {:.2}x",
        scalar_time.as_nanos() as f64 / wide_time.as_nanos() as f64
    );

    // Note about platform behavior
    println!("\nNote: wide::i32x8::transpose is AVX2-accelerated on x86_64,");
    println!("      scalar fallback on ARM/WASM (still benefits from SIMD butterfly)");
}

fn test_ycbcr_parity() {
    println!("\n=== YCbCr→RGB Parity Test ===");

    // Generate test data
    let y: [i16; 16] = std::array::from_fn(|i| 128 + (i as i16 % 50));
    let cb: [i16; 16] = std::array::from_fn(|i| 128 + ((i * 3) as i16 % 40));
    let cr: [i16; 16] = std::array::from_fn(|i| 128 + ((i * 7) as i16 % 60));

    let mut rgb_scalar = [0u8; 48];
    let mut rgb_wide = [0u8; 48];

    // Run both implementations
    ycbcr_scalar::ycbcr_to_rgb_i16_x16(&y, &cb, &cr, &mut rgb_scalar);
    ycbcr_wide::ycbcr_to_rgb_i16_x16(&y, &cb, &cr, &mut rgb_wide);

    // Compare
    let mut max_diff = 0i16;
    let mut diff_count = 0;
    for i in 0..48 {
        let diff = (rgb_scalar[i] as i16 - rgb_wide[i] as i16).abs();
        if diff > 0 {
            diff_count += 1;
            max_diff = max_diff.max(diff);
        }
    }

    if diff_count == 0 {
        println!("✓ Scalar and Wide implementations MATCH exactly");
    } else {
        println!(
            "✗ Differences found: {} values differ, max diff = {}",
            diff_count, max_diff
        );
    }

    // Print first 6 pixels (18 bytes)
    println!("First 6 pixels (scalar): {:?}", &rgb_scalar[0..18]);
    println!("First 6 pixels (wide):   {:?}", &rgb_wide[0..18]);

    // Benchmark
    let iterations = 1_000_000;

    let start = Instant::now();
    for _ in 0..iterations {
        let mut rgb = [0u8; 48];
        ycbcr_scalar::ycbcr_to_rgb_i16_x16(&y, &cb, &cr, &mut rgb);
        std::hint::black_box(&rgb);
    }
    let scalar_time = start.elapsed();

    let start = Instant::now();
    for _ in 0..iterations {
        let mut rgb = [0u8; 48];
        ycbcr_wide::ycbcr_to_rgb_i16_x16(&y, &cb, &cr, &mut rgb);
        std::hint::black_box(&rgb);
    }
    let wide_time = start.elapsed();

    println!(
        "Scalar: {:.2} ns/16px, Wide: {:.2} ns/16px",
        scalar_time.as_nanos() as f64 / iterations as f64,
        wide_time.as_nanos() as f64 / iterations as f64
    );
    println!(
        "Wide speedup: {:.2}x",
        scalar_time.as_nanos() as f64 / wide_time.as_nanos() as f64
    );
}

fn test_gather_parity() {
    println!("\n=== Even/Odd Gather Parity Test ===");

    // Generate test data
    let data: [f32; 16] = std::array::from_fn(|i| i as f32);

    let (evens_scalar, odds_scalar) = gather_scalar::gather_even_odd(&data);
    let (evens_wide, odds_wide) = gather_wide::gather_even_odd(&data);

    let evens_wide_arr = evens_wide.to_array();
    let odds_wide_arr = odds_wide.to_array();

    // Compare
    let mut match_count = 0;
    for i in 0..8 {
        if (evens_scalar[i] - evens_wide_arr[i]).abs() < 1e-6
            && (odds_scalar[i] - odds_wide_arr[i]).abs() < 1e-6
        {
            match_count += 1;
        }
    }

    if match_count == 8 {
        println!("✓ Scalar and Wide gather MATCH exactly");
    } else {
        println!("✗ Mismatch: {}/8 pairs match", match_count);
    }

    println!("Evens (scalar): {:?}", evens_scalar);
    println!("Evens (wide):   {:?}", evens_wide_arr);

    // Benchmark
    let iterations = 10_000_000;

    let start = Instant::now();
    for _ in 0..iterations {
        let (e, o) = gather_scalar::gather_even_odd(&data);
        std::hint::black_box((&e, &o));
    }
    let scalar_time = start.elapsed();

    let start = Instant::now();
    for _ in 0..iterations {
        let (e, o) = gather_wide::gather_even_odd(&data);
        std::hint::black_box((&e, &o));
    }
    let wide_time = start.elapsed();

    println!(
        "Scalar: {:.2} ns/op, Wide: {:.2} ns/op",
        scalar_time.as_nanos() as f64 / iterations as f64,
        wide_time.as_nanos() as f64 / iterations as f64
    );
}

fn main() {
    println!("SIMD Parity Testing - Intel vs Portable Implementations");
    println!("========================================================");

    #[cfg(target_arch = "x86_64")]
    {
        println!("\nCPU Features detected:");
        println!("  AVX2: {}", is_x86_feature_detected!("avx2"));
        println!("  FMA:  {}", is_x86_feature_detected!("fma"));
        println!("  SSE2: {}", is_x86_feature_detected!("sse2"));
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        println!("\nNon-x86_64 platform - testing portable implementations only");
    }

    test_idct_parity();
    test_ycbcr_parity();
    test_gather_parity();

    println!("\n========================================================");
    println!("Summary: These tests compare Intel-specific AVX2 code with");
    println!("portable `wide` crate implementations for WASM/ARM targets.");
    println!("\nNext steps:");
    println!("1. Implement i32x8 transpose for portable IDCT");
    println!("2. Add NEON-specific paths where beneficial");
    println!("3. Test on actual ARM/WASM targets");
}
