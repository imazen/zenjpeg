//! Integer IDCT implementation for standard JPEG decoding.
//!
//! This module provides a fast integer-only IDCT for non-XYB JPEGs.
//! Based on zune-jpeg's implementation (MIT/Apache/Zlib licensed).
//!
//! For XYB mode, use the f32 IDCT in `idct.rs` instead.
//!
//! # SIMD Implementations
//!
//! Three implementations are available:
//! - **wide**: Portable SIMD using `wide` crate with `#[autoversion]` (recommended)
//! - **avx2**: AVX2 intrinsics via archmage capability tokens (x86_64 only, kept for reference)
//! - **scalar**: Pure scalar fallback
//!
//! Benchmarks (8x8 IDCT block):
//! - x86_64 AVX2: wide 1.64x faster than scalar
//! - aarch64 NEON: wide 1.11x faster than scalar

#![allow(dead_code)]

#[cfg(target_arch = "x86_64")]
use archmage::SimdToken;
#[cfg(target_arch = "x86_64")]
use safe_unaligned_simd::x86_64 as safe_simd;

/// Rounding and level-shift constants.
/// SCALE_BITS = 512 + 65536 + (128 << 17)
/// - 512 = rounding for first pass (>> 10)
/// - 65536 = rounding for second pass (>> 17)
/// - (128 << 17) = level shift (+128) pre-scaled
const SCALE_BITS: i32 = 512 + 65536 + (128 << 17);

/// Fixed-point multiplication factor (4096 = 1 << 12).
#[inline]
const fn f2f(x: f32) -> i32 {
    (x * 4096.0 + 0.5) as i32
}

/// Shift left by 12 bits (multiply by 4096).
#[inline]
const fn fsh(x: i32) -> i32 {
    x << 12
}

/// Clamp to [0, 255] and convert to i16.
#[inline]
fn clamp(a: i32) -> i16 {
    a.clamp(0, 255) as i16
}

/// Wrapping add.
#[inline(always)]
const fn wa(a: i32, b: i32) -> i32 {
    a.wrapping_add(b)
}

/// Wrapping subtract.
#[inline(always)]
const fn ws(a: i32, b: i32) -> i32 {
    a.wrapping_sub(b)
}

/// Wrapping multiply.
#[inline(always)]
const fn wm(a: i32, b: i32) -> i32 {
    a.wrapping_mul(b)
}

/// Fast path: DC-only block (all AC coefficients are zero).
/// Fills entire output block with the scaled DC value.
#[inline]
pub fn idct_int_dc_only(dc_coeff: i32, out_vector: &mut [i16], stride: usize) {
    // DC value after IDCT = (DC + rounding + level_shift) >> 3
    // Rounding: add 4 (half of 8)
    // Level shift: add 1024 (128 << 3)
    let coeff = wa(wa(dc_coeff, 4), 1024).wrapping_shr(3).clamp(0, 255) as i16;

    // Single bounds check: ensure the buffer can hold 8 strided rows.
    // This lets the compiler prove all indexed accesses are in-bounds
    // and elide per-row bounds checks in the loop below.
    let min_len = stride * 7 + 8;
    assert!(out_vector.len() >= min_len);
    let out = &mut out_vector[..min_len];
    for i in 0..8 {
        let off = i * stride;
        out[off..off + 8].fill(coeff);
    }
}

/// Check if all AC coefficients are zero (DC-only block).
#[inline]
pub fn is_dc_only_int(coeffs: &[i32; 64]) -> bool {
    coeffs[1..].iter().all(|&x| x == 0)
}

// ============================================================================
// libjpeg-compatible IDCT (Loeffler algorithm, 13-bit precision)
// ============================================================================
//
// This is a direct port of libjpeg-turbo's jpeg_idct_islow (jidctint.c).
// Uses the Loeffler, Ligtenberg, Moschytz algorithm with 13-bit constants
// and PASS1_BITS=2. Produces output bit-identical to libjpeg-turbo for
// matching dequantized input coefficients.
//
// Reference: C. Loeffler, A. Ligtenberg and G. Moschytz,
//   "Practical Fast 1-D DCT Algorithms with 11 Multiplications",
//   Proc. ICASSP '89, pp. 988-991.

/// 13-bit fixed-point constants for the Loeffler IDCT.
///
/// These are i64 to match libjpeg-turbo's `JLONG` type (`long` = 64-bit on
/// LP64 systems). Note: on Windows (LLP64), `long` is 32-bit, so libjpeg-turbo
/// uses i32 there. Additionally, libjpeg-turbo's 8-bit path uses
/// `MULTIPLY16C16` which truncates intermediates to INT16 before multiplying,
/// so the JLONG workspace doesn't prevent overflow — both libjpeg-turbo and
/// our i32 Jpegli IDCT wrap for extreme dequantized values (coefficient × quant
/// \> 32767). Our i64 path is more mathematically correct but produces different
/// output from libjpeg-turbo on such inputs.
const LJ_FIX_0_298631336: i64 = 2446;
const LJ_FIX_0_390180644: i64 = 3196;
const LJ_FIX_0_541196100: i64 = 4433;
const LJ_FIX_0_765366865: i64 = 6270;
const LJ_FIX_0_899976223: i64 = 7373;
const LJ_FIX_1_175875602: i64 = 9633;
const LJ_FIX_1_501321110: i64 = 12299;
const LJ_FIX_1_847759065: i64 = 15137;
const LJ_FIX_1_961570560: i64 = 16069;
const LJ_FIX_2_053119869: i64 = 16819;
const LJ_FIX_2_562915447: i64 = 20995;
const LJ_FIX_3_072711026: i64 = 25172;

const LJ_CONST_BITS: u32 = 13;
const LJ_PASS1_BITS: u32 = 2;

/// Rounded right shift: (x + (1 << (n-1))) >> n
#[inline(always)]
const fn descale(x: i64, n: u32) -> i64 {
    (x + (1 << (n - 1))) >> n
}

/// libjpeg-turbo compatible integer IDCT (Loeffler algorithm).
///
/// Input: dequantized DCT coefficients in natural (row-major) order.
/// Output: pixel values level-shifted to [0, 255] as i16.
///
/// This produces output matching libjpeg-turbo's `jpeg_idct_islow` for
/// identical dequantized input.
#[allow(clippy::too_many_lines)]
pub fn idct_int_libjpeg(in_vector: &mut [i32; 64], out_vector: &mut [i16], stride: usize) {
    // DC-only fast path (identical to libjpeg)
    if is_dc_only_int(in_vector) {
        return idct_int_dc_only(in_vector[0], out_vector, stride);
    }

    // Single bounds check for all strided writes below
    let min_len = stride * 7 + 8;
    assert!(out_vector.len() >= min_len);
    let out_vector = &mut out_vector[..min_len];

    // i64 workspace. Note: libjpeg-turbo uses JLONG (long) which is 64-bit on
    // LP64 but 32-bit on Windows. In practice, libjpeg-turbo's 8-bit path also
    // truncates to INT16 in MULTIPLY16C16, so i64 vs i32 workspace rarely matters.
    let mut workspace = [0i64; 64];

    // Pass 1: process columns, store into workspace.
    // Results are scaled up by sqrt(8) * 2^PASS1_BITS.
    for col in 0..8 {
        // Short-circuit for columns with all AC terms zero
        if in_vector[col + 8] == 0
            && in_vector[col + 16] == 0
            && in_vector[col + 24] == 0
            && in_vector[col + 32] == 0
            && in_vector[col + 40] == 0
            && in_vector[col + 48] == 0
            && in_vector[col + 56] == 0
        {
            let dcval = (in_vector[col] as i64) << LJ_PASS1_BITS;
            workspace[col] = dcval;
            workspace[col + 8] = dcval;
            workspace[col + 16] = dcval;
            workspace[col + 24] = dcval;
            workspace[col + 32] = dcval;
            workspace[col + 40] = dcval;
            workspace[col + 48] = dcval;
            workspace[col + 56] = dcval;
            continue;
        }

        // Even part — widen to i64 at the boundary
        let z2 = in_vector[col + 16] as i64;
        let z3 = in_vector[col + 48] as i64;

        let z1 = (z2 + z3) * LJ_FIX_0_541196100;
        let tmp2 = z1 + z3 * (-LJ_FIX_1_847759065);
        let tmp3 = z1 + z2 * LJ_FIX_0_765366865;

        let z2 = in_vector[col] as i64;
        let z3 = in_vector[col + 32] as i64;

        let tmp0 = (z2 + z3) << LJ_CONST_BITS;
        let tmp1 = (z2 - z3) << LJ_CONST_BITS;

        let tmp10 = tmp0 + tmp3;
        let tmp13 = tmp0 - tmp3;
        let tmp11 = tmp1 + tmp2;
        let tmp12 = tmp1 - tmp2;

        // Odd part
        let mut tmp0 = in_vector[col + 56] as i64;
        let mut tmp1 = in_vector[col + 40] as i64;
        let mut tmp2 = in_vector[col + 24] as i64;
        let mut tmp3 = in_vector[col + 8] as i64;

        let z1 = tmp0 + tmp3;
        let z2 = tmp1 + tmp2;
        let z3 = tmp0 + tmp2;
        let z4 = tmp1 + tmp3;
        let z5 = (z3 + z4) * LJ_FIX_1_175875602;

        tmp0 *= LJ_FIX_0_298631336;
        tmp1 *= LJ_FIX_2_053119869;
        tmp2 *= LJ_FIX_3_072711026;
        tmp3 *= LJ_FIX_1_501321110;
        let z1 = z1 * (-LJ_FIX_0_899976223);
        let z2 = z2 * (-LJ_FIX_2_562915447);
        let z3 = z3 * (-LJ_FIX_1_961570560) + z5;
        let z4 = z4 * (-LJ_FIX_0_390180644) + z5;

        tmp0 += z1 + z3;
        tmp1 += z2 + z4;
        tmp2 += z2 + z3;
        tmp3 += z1 + z4;

        // Final output: descale by (CONST_BITS - PASS1_BITS)
        workspace[col] = descale(tmp10 + tmp3, LJ_CONST_BITS - LJ_PASS1_BITS);
        workspace[col + 56] = descale(tmp10 - tmp3, LJ_CONST_BITS - LJ_PASS1_BITS);
        workspace[col + 8] = descale(tmp11 + tmp2, LJ_CONST_BITS - LJ_PASS1_BITS);
        workspace[col + 48] = descale(tmp11 - tmp2, LJ_CONST_BITS - LJ_PASS1_BITS);
        workspace[col + 16] = descale(tmp12 + tmp1, LJ_CONST_BITS - LJ_PASS1_BITS);
        workspace[col + 40] = descale(tmp12 - tmp1, LJ_CONST_BITS - LJ_PASS1_BITS);
        workspace[col + 24] = descale(tmp13 + tmp0, LJ_CONST_BITS - LJ_PASS1_BITS);
        workspace[col + 32] = descale(tmp13 - tmp0, LJ_CONST_BITS - LJ_PASS1_BITS);
    }

    // Pass 2: process rows from workspace, store into output.
    // Descale by factor of 8 (2^3) plus PASS1_BITS.
    let total_shift = LJ_CONST_BITS + LJ_PASS1_BITS + 3;

    for row in 0..8 {
        let base = row * 8;

        // Row DC-only short-circuit
        if workspace[base + 1] == 0
            && workspace[base + 2] == 0
            && workspace[base + 3] == 0
            && workspace[base + 4] == 0
            && workspace[base + 5] == 0
            && workspace[base + 6] == 0
            && workspace[base + 7] == 0
        {
            let dcval = (descale(workspace[base], LJ_PASS1_BITS + 3) + 128).clamp(0, 255) as i16;
            let out_base = row * stride;
            out_vector[out_base..out_base + 8].fill(dcval);
            continue;
        }

        // Even part (all i64 — workspace is already i64)
        let z2 = workspace[base + 2];
        let z3 = workspace[base + 6];

        let z1 = (z2 + z3) * LJ_FIX_0_541196100;
        let tmp2 = z1 + z3 * (-LJ_FIX_1_847759065);
        let tmp3 = z1 + z2 * LJ_FIX_0_765366865;

        let tmp0 = (workspace[base] + workspace[base + 4]) << LJ_CONST_BITS;
        let tmp1 = (workspace[base] - workspace[base + 4]) << LJ_CONST_BITS;

        let tmp10 = tmp0 + tmp3;
        let tmp13 = tmp0 - tmp3;
        let tmp11 = tmp1 + tmp2;
        let tmp12 = tmp1 - tmp2;

        // Odd part
        let mut tmp0 = workspace[base + 7];
        let mut tmp1 = workspace[base + 5];
        let mut tmp2 = workspace[base + 3];
        let mut tmp3 = workspace[base + 1];

        let z1 = tmp0 + tmp3;
        let z2 = tmp1 + tmp2;
        let z3 = tmp0 + tmp2;
        let z4 = tmp1 + tmp3;
        let z5 = (z3 + z4) * LJ_FIX_1_175875602;

        tmp0 *= LJ_FIX_0_298631336;
        tmp1 *= LJ_FIX_2_053119869;
        tmp2 *= LJ_FIX_3_072711026;
        tmp3 *= LJ_FIX_1_501321110;
        let z1 = z1 * (-LJ_FIX_0_899976223);
        let z2 = z2 * (-LJ_FIX_2_562915447);
        let z3 = z3 * (-LJ_FIX_1_961570560) + z5;
        let z4 = z4 * (-LJ_FIX_0_390180644) + z5;

        tmp0 += z1 + z3;
        tmp1 += z2 + z4;
        tmp2 += z2 + z3;
        tmp3 += z1 + z4;

        // Final output: descale + level shift (+128) + clamp
        let out_base = row * stride;
        out_vector[out_base] = (descale(tmp10 + tmp3, total_shift) + 128).clamp(0, 255) as i16;
        out_vector[out_base + 7] = (descale(tmp10 - tmp3, total_shift) + 128).clamp(0, 255) as i16;
        out_vector[out_base + 1] = (descale(tmp11 + tmp2, total_shift) + 128).clamp(0, 255) as i16;
        out_vector[out_base + 6] = (descale(tmp11 - tmp2, total_shift) + 128).clamp(0, 255) as i16;
        out_vector[out_base + 2] = (descale(tmp12 + tmp1, total_shift) + 128).clamp(0, 255) as i16;
        out_vector[out_base + 5] = (descale(tmp12 - tmp1, total_shift) + 128).clamp(0, 255) as i16;
        out_vector[out_base + 3] = (descale(tmp13 + tmp0, total_shift) + 128).clamp(0, 255) as i16;
        out_vector[out_base + 4] = (descale(tmp13 - tmp0, total_shift) + 128).clamp(0, 255) as i16;
    }
}

/// Integer IDCT for 8x8 block.
///
/// # Arguments
/// * `in_vector` - Input dequantized DCT coefficients (modified in place during computation)
/// * `out_vector` - Output pixel values (i16 in range [0, 255])
/// * `stride` - Stride between output rows
///
/// # Notes
/// - Uses fixed-point arithmetic with 12-bit precision
/// - Output is already level-shifted (+128) and clamped to [0, 255]
#[allow(clippy::too_many_lines)]
pub fn idct_int(in_vector: &mut [i32; 64], out_vector: &mut [i16], stride: usize) {
    // DC-only fast path
    if is_dc_only_int(in_vector) {
        return idct_int_dc_only(in_vector[0], out_vector, stride);
    }

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
        let p5 = wm(wa(p3, p4), f2f(1.175_875_6));

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

        pos += stride;
    }
}

/// Integer IDCT for blocks with only top-left 4x4 non-zero.
/// Faster than full 8x8 IDCT when AC coefficients are sparse.
#[allow(clippy::too_many_lines)]
pub fn idct_int_4x4(in_vector: &mut [i32; 64], out_vector: &mut [i16], stride: usize) {
    // Vertical pass (only first 4 columns matter)
    for ptr in 0..4 {
        let i0 = wa(fsh(in_vector[ptr]), 512);
        let i2 = in_vector[ptr + 16];

        let p1 = wm(i2, 2217);
        let p3 = wm(i2, 5352);

        let x0 = wa(i0, p3);
        let x1 = wa(i0, p1);
        let x2 = ws(i0, p1);
        let x3 = ws(i0, p3);

        // Odd part
        let i4 = in_vector[ptr + 24];
        let i3 = in_vector[ptr + 8];

        let p5 = wm(wa(i4, i3), 4816);

        let p1 = wa(p5, wm(i3, -3685));
        let p2 = wa(p5, wm(i4, -10497));

        let t3 = wa(p5, wm(i3, 867));
        let t2 = wa(p5, wm(i4, -5945));

        let t1 = wa(p2, wm(i3, -1597));
        let t0 = wa(p1, wm(i4, -8034));

        in_vector[ptr] = wa(x0, t3) >> 10;
        in_vector[ptr + 8] = wa(x1, t2) >> 10;
        in_vector[ptr + 16] = wa(x2, t1) >> 10;
        in_vector[ptr + 24] = wa(x3, t0) >> 10;
        in_vector[ptr + 32] = ws(x3, t0) >> 10;
        in_vector[ptr + 40] = ws(x2, t1) >> 10;
        in_vector[ptr + 48] = ws(x1, t2) >> 10;
        in_vector[ptr + 56] = ws(x0, t3) >> 10;
    }

    // Horizontal pass (full 8 rows)
    let mut pos = 0;
    for i in (0..64).step_by(8) {
        let i2 = in_vector[i + 2];
        let i0 = in_vector[i];

        let t0 = wa(fsh(i0), SCALE_BITS);
        let t2 = wm(i2, 2217);
        let t3 = wm(i2, 5352);

        let x0 = wa(t0, t3);
        let x3 = ws(t0, t3);
        let x1 = wa(t0, t2);
        let x2 = ws(t0, t2);

        // Odd part
        let i3 = in_vector[i + 3];
        let i1 = in_vector[i + 1];

        let p5 = wm(wa(i3, i1), f2f(1.175_875_6));

        let p1 = wa(p5, wm(i1, -3685));
        let p2 = wa(p5, wm(i3, -10497));

        let t3 = wa(p5, wm(i1, 867));
        let t2 = wa(p5, wm(i3, -5945));

        let t1 = wa(p2, wm(i1, -1597));
        let t0 = wa(p1, wm(i3, -8034));

        out_vector[pos] = clamp(wa(x0, t3) >> 17);
        out_vector[pos + 1] = clamp(wa(x1, t2) >> 17);
        out_vector[pos + 2] = clamp(wa(x2, t1) >> 17);
        out_vector[pos + 3] = clamp(wa(x3, t0) >> 17);
        out_vector[pos + 4] = clamp(ws(x3, t0) >> 17);
        out_vector[pos + 5] = clamp(ws(x2, t1) >> 17);
        out_vector[pos + 6] = clamp(ws(x1, t2) >> 17);
        out_vector[pos + 7] = clamp(ws(x0, t3) >> 17);

        pos += stride;
    }

    // Clear the parts we used (for next block reuse)
    in_vector[32..36].fill(0);
    in_vector[40..44].fill(0);
    in_vector[48..52].fill(0);
    in_vector[56..60].fill(0);
}

// =============================================================================
// AVX2 SIMD Implementation
// =============================================================================

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx2 {
    use super::*;
    use archmage::{arcane, rite};

    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    /// Shuffle constant helper (equivalent to _MM_SHUFFLE).
    #[inline]
    const fn shuffle(z: i32, y: i32, x: i32, w: i32) -> i32 {
        (z << 6) | (y << 4) | (x << 2) | w
    }

    /// Clamp i16 values to [0, 255] range.
    #[rite]
    fn clamp_avx(_token: archmage::X64V3Token, reg: __m256i) -> __m256i {
        let min_s = _mm256_set1_epi16(0);
        let max_s = _mm256_set1_epi16(255);
        let max_v = _mm256_max_epi16(reg, min_s);
        _mm256_min_epi16(max_v, max_s)
    }

    /// In-register 8x8 transpose for i32 values.
    #[rite]
    fn transpose_8x8_i32(
        _token: archmage::X64V3Token,
        v0: &mut __m256i,
        v1: &mut __m256i,
        v2: &mut __m256i,
        v3: &mut __m256i,
        v4: &mut __m256i,
        v5: &mut __m256i,
        v6: &mut __m256i,
        v7: &mut __m256i,
    ) {
        // Stage 1: interleave 32-bit values
        let va0 = _mm256_permute4x64_epi64(*v0, shuffle(3, 1, 2, 0));
        let vb0 = _mm256_permute4x64_epi64(*v1, shuffle(3, 1, 2, 0));
        let w0 = _mm256_unpacklo_epi32(va0, vb0);
        let w1 = _mm256_unpackhi_epi32(va0, vb0);

        let va2 = _mm256_permute4x64_epi64(*v2, shuffle(3, 1, 2, 0));
        let vb2 = _mm256_permute4x64_epi64(*v3, shuffle(3, 1, 2, 0));
        let w2 = _mm256_unpacklo_epi32(va2, vb2);
        let w3 = _mm256_unpackhi_epi32(va2, vb2);

        let va4 = _mm256_permute4x64_epi64(*v4, shuffle(3, 1, 2, 0));
        let vb4 = _mm256_permute4x64_epi64(*v5, shuffle(3, 1, 2, 0));
        let w4 = _mm256_unpacklo_epi32(va4, vb4);
        let w5 = _mm256_unpackhi_epi32(va4, vb4);

        let va6 = _mm256_permute4x64_epi64(*v6, shuffle(3, 1, 2, 0));
        let vb6 = _mm256_permute4x64_epi64(*v7, shuffle(3, 1, 2, 0));
        let w6 = _mm256_unpacklo_epi32(va6, vb6);
        let w7 = _mm256_unpackhi_epi32(va6, vb6);

        // Stage 2: interleave 64-bit values
        let xa0 = _mm256_permute4x64_epi64(w0, shuffle(3, 1, 2, 0));
        let xb0 = _mm256_permute4x64_epi64(w2, shuffle(3, 1, 2, 0));
        let x0 = _mm256_unpacklo_epi64(xa0, xb0);
        let x1 = _mm256_unpackhi_epi64(xa0, xb0);

        let xa1 = _mm256_permute4x64_epi64(w1, shuffle(3, 1, 2, 0));
        let xb1 = _mm256_permute4x64_epi64(w3, shuffle(3, 1, 2, 0));
        let x2 = _mm256_unpacklo_epi64(xa1, xb1);
        let x3 = _mm256_unpackhi_epi64(xa1, xb1);

        let xa4 = _mm256_permute4x64_epi64(w4, shuffle(3, 1, 2, 0));
        let xb4 = _mm256_permute4x64_epi64(w6, shuffle(3, 1, 2, 0));
        let x4 = _mm256_unpacklo_epi64(xa4, xb4);
        let x5 = _mm256_unpackhi_epi64(xa4, xb4);

        let xa5 = _mm256_permute4x64_epi64(w5, shuffle(3, 1, 2, 0));
        let xb5 = _mm256_permute4x64_epi64(w7, shuffle(3, 1, 2, 0));
        let x6 = _mm256_unpacklo_epi64(xa5, xb5);
        let x7 = _mm256_unpackhi_epi64(xa5, xb5);

        // Stage 3: interleave 128-bit lanes
        *v0 = _mm256_permute2x128_si256(x0, x4, shuffle(0, 2, 0, 0));
        *v1 = _mm256_permute2x128_si256(x0, x4, shuffle(0, 3, 0, 1));
        *v2 = _mm256_permute2x128_si256(x1, x5, shuffle(0, 2, 0, 0));
        *v3 = _mm256_permute2x128_si256(x1, x5, shuffle(0, 3, 0, 1));
        *v4 = _mm256_permute2x128_si256(x2, x6, shuffle(0, 2, 0, 0));
        *v5 = _mm256_permute2x128_si256(x2, x6, shuffle(0, 3, 0, 1));
        *v6 = _mm256_permute2x128_si256(x3, x7, shuffle(0, 2, 0, 0));
        *v7 = _mm256_permute2x128_si256(x3, x7, shuffle(0, 3, 0, 1));
    }

    /// AVX2 integer IDCT.
    ///
    /// Uses archmage capability token for safe SIMD dispatch.
    /// Load/store operations use safe_unaligned_simd wrappers.
    #[arcane]
    #[allow(unused_assignments)] // pos is incremented in macro but last value is unused
    pub fn idct_int_avx2(
        _token: archmage::X64V3Token,
        in_vector: &mut [i32; 64],
        out_vector: &mut [i16],
        stride: usize,
    ) {
        // Single bounds check for all strided writes (DC-only and full IDCT paths)
        assert!(out_vector.len() >= stride * 7 + 8);

        // Load all 8 rows
        let mut row0 =
            safe_simd::_mm256_loadu_si256(<&[i32; 8]>::try_from(&in_vector[0..8]).unwrap());
        let mut row1 =
            safe_simd::_mm256_loadu_si256(<&[i32; 8]>::try_from(&in_vector[8..16]).unwrap());
        let mut row2 =
            safe_simd::_mm256_loadu_si256(<&[i32; 8]>::try_from(&in_vector[16..24]).unwrap());
        let mut row3 =
            safe_simd::_mm256_loadu_si256(<&[i32; 8]>::try_from(&in_vector[24..32]).unwrap());
        let mut row4 =
            safe_simd::_mm256_loadu_si256(<&[i32; 8]>::try_from(&in_vector[32..40]).unwrap());
        let mut row5 =
            safe_simd::_mm256_loadu_si256(<&[i32; 8]>::try_from(&in_vector[40..48]).unwrap());
        let mut row6 =
            safe_simd::_mm256_loadu_si256(<&[i32; 8]>::try_from(&in_vector[48..56]).unwrap());
        let mut row7 =
            safe_simd::_mm256_loadu_si256(<&[i32; 8]>::try_from(&in_vector[56..64]).unwrap());

        // Check for DC-only (all AC = 0)
        let ac_check =
            safe_simd::_mm256_loadu_si256(<&[i32; 8]>::try_from(&in_vector[1..9]).unwrap());
        let mut bitmap = _mm256_or_si256(row1, row2);
        bitmap = _mm256_or_si256(bitmap, row3);
        bitmap = _mm256_or_si256(bitmap, row4);
        bitmap = _mm256_or_si256(bitmap, row5);
        bitmap = _mm256_or_si256(bitmap, row6);
        bitmap = _mm256_or_si256(bitmap, row7);
        bitmap = _mm256_or_si256(bitmap, ac_check);

        if _mm256_testz_si256(bitmap, bitmap) == 1 {
            // DC-only fast path
            let coeff = ((in_vector[0] + 4 + 1024) >> 3).clamp(0, 255) as i16;
            let idct_value = _mm_set1_epi16(coeff);

            let mut pos = 0;
            for _ in 0..8 {
                safe_simd::_mm_storeu_si128(
                    <&mut [i16; 8]>::try_from(&mut out_vector[pos..pos + 8]).unwrap(),
                    idct_value,
                );
                pos += stride;
            }
            return;
        }

        // Constants
        let c2217 = _mm256_set1_epi32(2217);
        let c3135 = _mm256_set1_epi32(3135);
        let cn7567 = _mm256_set1_epi32(-7567);
        let c4816 = _mm256_set1_epi32(4816);
        let c1223 = _mm256_set1_epi32(1223);
        let c8410 = _mm256_set1_epi32(8410);
        let c12586 = _mm256_set1_epi32(12586);
        let c6149 = _mm256_set1_epi32(6149);
        let cn3685 = _mm256_set1_epi32(-3685);
        let cn10497 = _mm256_set1_epi32(-10497);
        let cn8034 = _mm256_set1_epi32(-8034);
        let cn1597 = _mm256_set1_epi32(-1597);
        let c512 = _mm256_set1_epi32(512);
        let cscale = _mm256_set1_epi32(SCALE_BITS);

        // Macro for DCT pass
        macro_rules! dct_pass {
            ($scale_bits:expr, $shift:expr) => {
                // Even part
                let p1 = _mm256_mullo_epi32(_mm256_add_epi32(row2, row6), c2217);
                let t2 = _mm256_add_epi32(p1, _mm256_mullo_epi32(row6, cn7567));
                let t3 = _mm256_add_epi32(p1, _mm256_mullo_epi32(row2, c3135));

                let t0 = _mm256_slli_epi32(_mm256_add_epi32(row0, row4), 12);
                let t1 = _mm256_slli_epi32(_mm256_sub_epi32(row0, row4), 12);

                let x0 = _mm256_add_epi32(_mm256_add_epi32(t0, t3), $scale_bits);
                let x3 = _mm256_add_epi32(_mm256_sub_epi32(t0, t3), $scale_bits);
                let x1 = _mm256_add_epi32(_mm256_add_epi32(t1, t2), $scale_bits);
                let x2 = _mm256_add_epi32(_mm256_sub_epi32(t1, t2), $scale_bits);

                // Odd part
                let p3 = _mm256_add_epi32(row7, row3);
                let p4 = _mm256_add_epi32(row5, row1);
                let p1 = _mm256_add_epi32(row7, row1);
                let p2 = _mm256_add_epi32(row5, row3);
                let p5 = _mm256_mullo_epi32(_mm256_add_epi32(p3, p4), c4816);

                let mut t0 = _mm256_mullo_epi32(row7, c1223);
                let mut t1 = _mm256_mullo_epi32(row5, c8410);
                let mut t2 = _mm256_mullo_epi32(row3, c12586);
                let mut t3 = _mm256_mullo_epi32(row1, c6149);

                let p1 = _mm256_add_epi32(p5, _mm256_mullo_epi32(p1, cn3685));
                let p2 = _mm256_add_epi32(p5, _mm256_mullo_epi32(p2, cn10497));
                let p3 = _mm256_mullo_epi32(p3, cn8034);
                let p4 = _mm256_mullo_epi32(p4, cn1597);

                t3 = _mm256_add_epi32(t3, _mm256_add_epi32(p1, p4));
                t2 = _mm256_add_epi32(t2, _mm256_add_epi32(p2, p3));
                t1 = _mm256_add_epi32(t1, _mm256_add_epi32(p2, p4));
                t0 = _mm256_add_epi32(t0, _mm256_add_epi32(p1, p3));

                row0 = _mm256_srai_epi32(_mm256_add_epi32(x0, t3), $shift);
                row1 = _mm256_srai_epi32(_mm256_add_epi32(x1, t2), $shift);
                row2 = _mm256_srai_epi32(_mm256_add_epi32(x2, t1), $shift);
                row3 = _mm256_srai_epi32(_mm256_add_epi32(x3, t0), $shift);
                row4 = _mm256_srai_epi32(_mm256_sub_epi32(x3, t0), $shift);
                row5 = _mm256_srai_epi32(_mm256_sub_epi32(x2, t1), $shift);
                row6 = _mm256_srai_epi32(_mm256_sub_epi32(x1, t2), $shift);
                row7 = _mm256_srai_epi32(_mm256_sub_epi32(x0, t3), $shift);
            };
        }

        // First pass (columns)
        dct_pass!(c512, 10);

        // Transpose
        transpose_8x8_i32(
            _token, &mut row0, &mut row1, &mut row2, &mut row3, &mut row4, &mut row5, &mut row6,
            &mut row7,
        );

        // Second pass (rows)
        dct_pass!(cscale, 17);

        // Transpose back
        transpose_8x8_i32(
            _token, &mut row0, &mut row1, &mut row2, &mut row3, &mut row4, &mut row5, &mut row6,
            &mut row7,
        );

        // Pack and store
        let mut pos = 0;

        macro_rules! pack_store {
            ($r0:expr, $r1:expr) => {
                let packed = _mm256_packs_epi32($r0, $r1);
                let clamped = clamp_avx(_token, packed);
                let reordered = _mm256_permute4x64_epi64(clamped, shuffle(3, 1, 2, 0));

                safe_simd::_mm_storeu_si128(
                    <&mut [i16; 8]>::try_from(&mut out_vector[pos..pos + 8]).unwrap(),
                    _mm256_extracti128_si256::<0>(reordered),
                );
                pos += stride;
                safe_simd::_mm_storeu_si128(
                    <&mut [i16; 8]>::try_from(&mut out_vector[pos..pos + 8]).unwrap(),
                    _mm256_extracti128_si256::<1>(reordered),
                );
                pos += stride;
            };
        }

        pack_store!(row0, row1);
        pack_store!(row2, row3);
        pack_store!(row4, row5);
        pack_store!(row6, row7);
        let _ = pos;
    }

    /// Unclamped AVX2 integer IDCT.
    ///
    /// Same butterfly as `idct_int_avx2` but outputs i16 values WITHOUT
    /// clamping to [0, 255]. Values are level-shifted (+128) and saturated
    /// to i16 range by `_mm256_packs_epi32`, but NOT clamped to [0, 255].
    ///
    /// This is critical for correct YCbCr→RGB conversion of wide-gamut images
    /// where Cb/Cr values can legitimately exceed [0, 255] after IDCT.
    #[arcane]
    #[allow(unused_assignments)]
    pub fn idct_int_avx2_unclamped(
        _token: archmage::X64V3Token,
        in_vector: &mut [i32; 64],
        out_vector: &mut [i16],
        stride: usize,
    ) {
        assert!(out_vector.len() >= stride * 7 + 8);

        // Load all 8 rows
        let mut row0 =
            safe_simd::_mm256_loadu_si256(<&[i32; 8]>::try_from(&in_vector[0..8]).unwrap());
        let mut row1 =
            safe_simd::_mm256_loadu_si256(<&[i32; 8]>::try_from(&in_vector[8..16]).unwrap());
        let mut row2 =
            safe_simd::_mm256_loadu_si256(<&[i32; 8]>::try_from(&in_vector[16..24]).unwrap());
        let mut row3 =
            safe_simd::_mm256_loadu_si256(<&[i32; 8]>::try_from(&in_vector[24..32]).unwrap());
        let mut row4 =
            safe_simd::_mm256_loadu_si256(<&[i32; 8]>::try_from(&in_vector[32..40]).unwrap());
        let mut row5 =
            safe_simd::_mm256_loadu_si256(<&[i32; 8]>::try_from(&in_vector[40..48]).unwrap());
        let mut row6 =
            safe_simd::_mm256_loadu_si256(<&[i32; 8]>::try_from(&in_vector[48..56]).unwrap());
        let mut row7 =
            safe_simd::_mm256_loadu_si256(<&[i32; 8]>::try_from(&in_vector[56..64]).unwrap());

        // DC-only check
        let ac_check =
            safe_simd::_mm256_loadu_si256(<&[i32; 8]>::try_from(&in_vector[1..9]).unwrap());
        let mut bitmap = _mm256_or_si256(row1, row2);
        bitmap = _mm256_or_si256(bitmap, row3);
        bitmap = _mm256_or_si256(bitmap, row4);
        bitmap = _mm256_or_si256(bitmap, row5);
        bitmap = _mm256_or_si256(bitmap, row6);
        bitmap = _mm256_or_si256(bitmap, row7);
        bitmap = _mm256_or_si256(bitmap, ac_check);

        if _mm256_testz_si256(bitmap, bitmap) == 1 {
            // DC-only: unclamped
            let coeff = ((in_vector[0] + 4 + 1024) >> 3) as i16;
            let idct_value = _mm_set1_epi16(coeff);
            let mut pos = 0;
            for _ in 0..8 {
                safe_simd::_mm_storeu_si128(
                    <&mut [i16; 8]>::try_from(&mut out_vector[pos..pos + 8]).unwrap(),
                    idct_value,
                );
                pos += stride;
            }
            return;
        }

        // Constants
        let c2217 = _mm256_set1_epi32(2217);
        let c3135 = _mm256_set1_epi32(3135);
        let cn7567 = _mm256_set1_epi32(-7567);
        let c4816 = _mm256_set1_epi32(4816);
        let c1223 = _mm256_set1_epi32(1223);
        let c8410 = _mm256_set1_epi32(8410);
        let c12586 = _mm256_set1_epi32(12586);
        let c6149 = _mm256_set1_epi32(6149);
        let cn3685 = _mm256_set1_epi32(-3685);
        let cn10497 = _mm256_set1_epi32(-10497);
        let cn8034 = _mm256_set1_epi32(-8034);
        let cn1597 = _mm256_set1_epi32(-1597);
        let c512 = _mm256_set1_epi32(512);
        let cscale = _mm256_set1_epi32(SCALE_BITS);

        macro_rules! dct_pass {
            ($scale_bits:expr, $shift:expr) => {
                let p1 = _mm256_mullo_epi32(_mm256_add_epi32(row2, row6), c2217);
                let t2 = _mm256_add_epi32(p1, _mm256_mullo_epi32(row6, cn7567));
                let t3 = _mm256_add_epi32(p1, _mm256_mullo_epi32(row2, c3135));

                let t0 = _mm256_slli_epi32(_mm256_add_epi32(row0, row4), 12);
                let t1 = _mm256_slli_epi32(_mm256_sub_epi32(row0, row4), 12);

                let x0 = _mm256_add_epi32(_mm256_add_epi32(t0, t3), $scale_bits);
                let x3 = _mm256_add_epi32(_mm256_sub_epi32(t0, t3), $scale_bits);
                let x1 = _mm256_add_epi32(_mm256_add_epi32(t1, t2), $scale_bits);
                let x2 = _mm256_add_epi32(_mm256_sub_epi32(t1, t2), $scale_bits);

                let p3 = _mm256_add_epi32(row7, row3);
                let p4 = _mm256_add_epi32(row5, row1);
                let p1 = _mm256_add_epi32(row7, row1);
                let p2 = _mm256_add_epi32(row5, row3);
                let p5 = _mm256_mullo_epi32(_mm256_add_epi32(p3, p4), c4816);

                let mut t0 = _mm256_mullo_epi32(row7, c1223);
                let mut t1 = _mm256_mullo_epi32(row5, c8410);
                let mut t2 = _mm256_mullo_epi32(row3, c12586);
                let mut t3 = _mm256_mullo_epi32(row1, c6149);

                let p1 = _mm256_add_epi32(p5, _mm256_mullo_epi32(p1, cn3685));
                let p2 = _mm256_add_epi32(p5, _mm256_mullo_epi32(p2, cn10497));
                let p3 = _mm256_mullo_epi32(p3, cn8034);
                let p4 = _mm256_mullo_epi32(p4, cn1597);

                t3 = _mm256_add_epi32(t3, _mm256_add_epi32(p1, p4));
                t2 = _mm256_add_epi32(t2, _mm256_add_epi32(p2, p3));
                t1 = _mm256_add_epi32(t1, _mm256_add_epi32(p2, p4));
                t0 = _mm256_add_epi32(t0, _mm256_add_epi32(p1, p3));

                row0 = _mm256_srai_epi32(_mm256_add_epi32(x0, t3), $shift);
                row1 = _mm256_srai_epi32(_mm256_add_epi32(x1, t2), $shift);
                row2 = _mm256_srai_epi32(_mm256_add_epi32(x2, t1), $shift);
                row3 = _mm256_srai_epi32(_mm256_add_epi32(x3, t0), $shift);
                row4 = _mm256_srai_epi32(_mm256_sub_epi32(x3, t0), $shift);
                row5 = _mm256_srai_epi32(_mm256_sub_epi32(x2, t1), $shift);
                row6 = _mm256_srai_epi32(_mm256_sub_epi32(x1, t2), $shift);
                row7 = _mm256_srai_epi32(_mm256_sub_epi32(x0, t3), $shift);
            };
        }

        // Column pass
        dct_pass!(c512, 10);

        // Transpose
        transpose_8x8_i32(
            _token, &mut row0, &mut row1, &mut row2, &mut row3, &mut row4, &mut row5, &mut row6,
            &mut row7,
        );

        // Row pass
        dct_pass!(cscale, 17);

        // Transpose back
        transpose_8x8_i32(
            _token, &mut row0, &mut row1, &mut row2, &mut row3, &mut row4, &mut row5, &mut row6,
            &mut row7,
        );

        // Pack and store WITHOUT clamping: packs_epi32 saturates to i16 range
        // [-32768, 32767] which is sufficient for YCbCr→RGB.
        let mut pos = 0;
        macro_rules! pack_store_unclamped {
            ($r0:expr, $r1:expr) => {
                let packed = _mm256_packs_epi32($r0, $r1);
                let reordered = _mm256_permute4x64_epi64(packed, shuffle(3, 1, 2, 0));

                safe_simd::_mm_storeu_si128(
                    <&mut [i16; 8]>::try_from(&mut out_vector[pos..pos + 8]).unwrap(),
                    _mm256_extracti128_si256::<0>(reordered),
                );
                pos += stride;
                safe_simd::_mm_storeu_si128(
                    <&mut [i16; 8]>::try_from(&mut out_vector[pos..pos + 8]).unwrap(),
                    _mm256_extracti128_si256::<1>(reordered),
                );
                pos += stride;
            };
        }

        pack_store_unclamped!(row0, row1);
        pack_store_unclamped!(row2, row3);
        pack_store_unclamped!(row4, row5);
        pack_store_unclamped!(row6, row7);
        let _ = pos;
    }
}

// =============================================================================
// Portable SIMD Implementation using `wide` crate
// =============================================================================

/// Portable SIMD IDCT using `wide` crate with `#[autoversion]` for cross-platform support.
///
/// This implementation uses `wide::i32x8` for the butterfly operations and
/// `wide::i32x8::transpose` for the 8x8 matrix transpose.
///
/// Performance vs scalar (standalone benchmark):
/// - x86_64 AVX2: 1.64x faster (26.9 ns vs 44.2 ns per block)
/// - aarch64 NEON: 1.11x faster (583.9 ns vs 646.9 ns per block via qemu)
mod wide_simd {
    use super::SCALE_BITS;
    use archmage::autoversion;
    use wide::i32x8;

    /// IDCT constants (fixed-point, 12-bit precision)
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

    /// Portable SIMD IDCT using `wide` crate.
    ///
    /// IMPORTANT: Uses `#[autoversion]` to enable SIMD on each target.
    /// Without this, `wide` falls back to scalar and is slower!
    ///
    /// Targets (in priority order):
    /// - x86_64+avx2: Uses AVX2 for i32x8 ops and transpose
    /// - x86_64+sse4.1: Uses SSE4.1 (i32x8 = 2x __m128i)
    /// - aarch64+neon: Uses NEON for i32x8 ops
    /// - default: Scalar fallback
    #[autoversion]
    pub fn idct_int_wide(in_vector: &[i32; 64], out_vector: &mut [i16], stride: usize) {
        // Load 8 rows as i32x8 vectors
        let mut rows: [i32x8; 8] = [
            i32x8::from(*<&[i32; 8]>::try_from(&in_vector[0..8]).unwrap()),
            i32x8::from(*<&[i32; 8]>::try_from(&in_vector[8..16]).unwrap()),
            i32x8::from(*<&[i32; 8]>::try_from(&in_vector[16..24]).unwrap()),
            i32x8::from(*<&[i32; 8]>::try_from(&in_vector[24..32]).unwrap()),
            i32x8::from(*<&[i32; 8]>::try_from(&in_vector[32..40]).unwrap()),
            i32x8::from(*<&[i32; 8]>::try_from(&in_vector[40..48]).unwrap()),
            i32x8::from(*<&[i32; 8]>::try_from(&in_vector[48..56]).unwrap()),
            i32x8::from(*<&[i32; 8]>::try_from(&in_vector[56..64]).unwrap()),
        ];

        // First pass (columns) - process all 8 columns in parallel
        idct_pass(&mut rows, i32x8::splat(512), 10);

        // Transpose using wide's built-in (uses SIMD on supported targets)
        rows = i32x8::transpose(rows);

        // Second pass (rows)
        idct_pass(&mut rows, i32x8::splat(SCALE_BITS), 17);

        // Transpose back to row-major order
        rows = i32x8::transpose(rows);

        // Extract and clamp to output with stride.
        // Single bounds check proves all strided writes are in-bounds.
        let min_len = stride * 7 + 8;
        assert!(out_vector.len() >= min_len);
        let out = &mut out_vector[..min_len];
        let mut out_pos = 0;
        for row in &rows {
            let arr = row.to_array();
            for (j, &val) in arr.iter().enumerate() {
                out[out_pos + j] = val.clamp(0, 255) as i16;
            }
            out_pos += stride;
        }
    }

    /// One pass of IDCT butterfly using i32x8 SIMD.
    ///
    /// This is the core IDCT computation, called twice (columns then rows).
    #[inline(always)]
    pub(super) fn idct_pass(rows: &mut [i32x8; 8], scale_bits: i32x8, shift: i32) {
        // Even part (rows 0, 2, 4, 6)
        let p1 = (rows[2] + rows[6]) * i32x8::splat(C2217);
        let t2 = p1 + rows[6] * i32x8::splat(CN7567);
        let t3 = p1 + rows[2] * i32x8::splat(C3135);

        let t0 = (rows[0] + rows[4]) << 12;
        let t1 = (rows[0] - rows[4]) << 12;

        let x0 = t0 + t3 + scale_bits;
        let x3 = t0 - t3 + scale_bits;
        let x1 = t1 + t2 + scale_bits;
        let x2 = t1 - t2 + scale_bits;

        // Odd part (rows 1, 3, 5, 7)
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

        // Combine even and odd parts, then shift
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

// =============================================================================
// Public API with runtime dispatch
// =============================================================================

/// Perform integer IDCT with automatic SIMD dispatch.
///
/// Uses AVX2 intrinsics on x86_64 with runtime detection, or falls back
/// to the portable `wide` crate implementation for other architectures.
///
/// # Arguments
/// * `coeffs` - Input dequantized DCT coefficients (not modified)
/// * `output` - Output pixel buffer (i16 in range [0, 255])
/// * `stride` - Stride between output rows
#[inline]
pub fn idct_int_auto(coeffs: &mut [i32; 64], output: &mut [i16], stride: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(token) = archmage::X64V3Token::summon() {
            avx2::idct_int_avx2(token, coeffs, output, stride);
            return;
        }
    }
    // Fallback to portable wide implementation
    wide_simd::idct_int_wide(coeffs, output, stride);
}

/// Perform integer IDCT using AVX2 intrinsics via archmage capability token.
///
/// This is the legacy implementation kept for comparison. In most cases,
/// `idct_int_auto` (which uses `wide`) should be preferred as it's portable
/// and has similar performance.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
pub fn idct_int_avx2_raw(
    token: archmage::X64V3Token,
    coeffs: &mut [i32; 64],
    output: &mut [i16],
    stride: usize,
) {
    avx2::idct_int_avx2(token, coeffs, output, stride);
}

/// Tiered IDCT selection based on coefficient count.
///
/// Selects the optimal IDCT implementation based on how many non-zero
/// coefficients are in the block (in zigzag scan order):
/// - count <= 1: DC-only (just broadcast DC value)
/// - count > 1: Full 8x8 IDCT with AVX2 (x86_64) or portable SIMD
///
/// Note: The 4x4 IDCT optimization for sparse blocks was removed because
/// the scalar 4x4 path was slower than the SIMD 8x8 path on modern CPUs.
///
/// # Arguments
/// * `coeffs` - Input dequantized DCT coefficients (modified in place for 4x4)
/// * `output` - Output pixel buffer (i16 in range [0, 255])
/// * `stride` - Stride between output rows
/// * `coeff_count` - Number of non-zero coefficients (1 = DC only, up to 64)
#[inline]
pub fn idct_int_tiered(coeffs: &mut [i32; 64], output: &mut [i16], stride: usize, coeff_count: u8) {
    if coeff_count <= 1 {
        // DC-only fast path
        idct_int_dc_only(coeffs[0], output, stride);
    } else {
        // Full 8x8 IDCT with SIMD (AVX2 on x86_64, wide otherwise)
        // Note: AVX2 IDCT with DC-only check is faster than tiered 4x4 scalar
        #[cfg(target_arch = "x86_64")]
        {
            if let Some(token) = archmage::X64V3Token::summon() {
                avx2::idct_int_avx2(token, coeffs, output, stride);
                return;
            }
        }
        // Fallback to portable SIMD
        wide_simd::idct_int_wide(coeffs, output, stride);
    }
}

/// libjpeg-compatible tiered IDCT dispatch.
///
/// Uses DC-only fast path for single-coefficient blocks, otherwise
/// uses `idct_int_libjpeg` (Loeffler algorithm with i64 intermediates).
pub fn idct_int_tiered_libjpeg(
    coeffs: &mut [i32; 64],
    output: &mut [i16],
    stride: usize,
    coeff_count: u8,
) {
    if coeff_count <= 1 {
        idct_int_dc_only(coeffs[0], output, stride);
    } else {
        idct_int_libjpeg(coeffs, output, stride);
    }
}

/// Convert a block of dequantized i32 coefficients to an [f32; 64] array.
/// Used for compatibility with existing f32 code paths.
#[inline]
pub fn coeffs_i32_to_f32(coeffs: &[i32; 64]) -> [f32; 64] {
    let mut out = [0.0f32; 64];
    for (i, &c) in coeffs.iter().enumerate() {
        out[i] = c as f32;
    }
    out
}

/// Convert [i16; 64] pixel output to [f32; 64] for compatibility.
/// Subtracts 128 to convert from 0-255 to centered -128 to 127 range.
#[inline]
pub fn pixels_i16_to_f32_centered(pixels: &[i16; 64]) -> [f32; 64] {
    let mut out = [0.0f32; 64];
    for (i, &p) in pixels.iter().enumerate() {
        out[i] = p as f32 - 128.0;
    }
    out
}

// =============================================================================
// Unclamped IDCT variants (for f32 output targets)
//
// These skip the [0, 255] clamping at the end, preserving ringing outside
// the nominal range. The level shift (+128) is still applied. Values
// typically land in [-30, 285] which is safe for all downstream consumers
// (upsampling uses i32 intermediates, YCbCr→RGB clamps at output).
// =============================================================================

/// Unclamped DC-only IDCT. Level-shifted but not clamped to [0, 255].
#[inline]
pub fn idct_int_dc_only_unclamped(dc_coeff: i32, out_vector: &mut [i16], stride: usize) {
    let coeff = wa(wa(dc_coeff, 4), 1024).wrapping_shr(3) as i16;

    let min_len = stride * 7 + 8;
    assert!(out_vector.len() >= min_len);
    let out = &mut out_vector[..min_len];
    for i in 0..8 {
        let off = i * stride;
        out[off..off + 8].fill(coeff);
    }
}

/// Unclamped wide SIMD IDCT.
fn idct_int_wide_unclamped(in_vector: &[i32; 64], out_vector: &mut [i16], stride: usize) {
    use wide::i32x8;

    let mut rows: [i32x8; 8] = [
        i32x8::from(*<&[i32; 8]>::try_from(&in_vector[0..8]).unwrap()),
        i32x8::from(*<&[i32; 8]>::try_from(&in_vector[8..16]).unwrap()),
        i32x8::from(*<&[i32; 8]>::try_from(&in_vector[16..24]).unwrap()),
        i32x8::from(*<&[i32; 8]>::try_from(&in_vector[24..32]).unwrap()),
        i32x8::from(*<&[i32; 8]>::try_from(&in_vector[32..40]).unwrap()),
        i32x8::from(*<&[i32; 8]>::try_from(&in_vector[40..48]).unwrap()),
        i32x8::from(*<&[i32; 8]>::try_from(&in_vector[48..56]).unwrap()),
        i32x8::from(*<&[i32; 8]>::try_from(&in_vector[56..64]).unwrap()),
    ];

    wide_simd::idct_pass(&mut rows, i32x8::splat(512), 10);
    rows = i32x8::transpose(rows);
    wide_simd::idct_pass(&mut rows, i32x8::splat(SCALE_BITS), 17);
    rows = i32x8::transpose(rows);

    // Store WITHOUT clamping — single bounds check for all strided writes
    let min_len = stride * 7 + 8;
    assert!(out_vector.len() >= min_len);
    let out = &mut out_vector[..min_len];
    let mut out_pos = 0;
    for row in &rows {
        let arr = row.to_array();
        for (j, &val) in arr.iter().enumerate() {
            out[out_pos + j] = val as i16;
        }
        out_pos += stride;
    }
}

/// Unclamped libjpeg-compatible IDCT.
pub fn idct_int_libjpeg_unclamped(
    in_vector: &mut [i32; 64],
    out_vector: &mut [i16],
    stride: usize,
) {
    // Single bounds check for all strided writes below
    let min_len = stride * 7 + 8;
    assert!(out_vector.len() >= min_len);
    let out_vector = &mut out_vector[..min_len];

    // i64 workspace matches libjpeg-turbo's JLONG (see idct_int_libjpeg).
    let mut workspace = [0i64; 64];

    // Column pass
    for col in 0..8 {
        let base = col;

        // DC-only shortcut
        if in_vector[base + 8] == 0
            && in_vector[base + 16] == 0
            && in_vector[base + 24] == 0
            && in_vector[base + 32] == 0
            && in_vector[base + 40] == 0
            && in_vector[base + 48] == 0
            && in_vector[base + 56] == 0
        {
            let dcval = (in_vector[base] as i64) << LJ_PASS1_BITS;
            for r in 0..8 {
                workspace[r * 8 + col] = dcval;
            }
            continue;
        }

        let z2 = in_vector[base + 16] as i64;
        let z3 = in_vector[base + 48] as i64;

        let z1 = (z2 + z3) * LJ_FIX_0_541196100;
        let tmp2 = z1 + z3 * (-LJ_FIX_1_847759065);
        let tmp3 = z1 + z2 * LJ_FIX_0_765366865;

        let z2 = in_vector[base] as i64;
        let z3 = in_vector[base + 32] as i64;

        let tmp0 = (z2 + z3) << LJ_CONST_BITS;
        let tmp1 = (z2 - z3) << LJ_CONST_BITS;

        let tmp10 = tmp0 + tmp3;
        let tmp13 = tmp0 - tmp3;
        let tmp11 = tmp1 + tmp2;
        let tmp12 = tmp1 - tmp2;

        let tmp0 = in_vector[base + 56] as i64;
        let tmp1 = in_vector[base + 40] as i64;
        let tmp2 = in_vector[base + 24] as i64;
        let tmp3 = in_vector[base + 8] as i64;

        let z1 = tmp0 + tmp3;
        let z2 = tmp1 + tmp2;
        let z3 = tmp0 + tmp2;
        let z4 = tmp1 + tmp3;
        let z5 = (z3 + z4) * LJ_FIX_1_175875602;

        let tmp0 = tmp0 * LJ_FIX_0_298631336;
        let tmp1 = tmp1 * LJ_FIX_2_053119869;
        let tmp2 = tmp2 * LJ_FIX_3_072711026;
        let tmp3 = tmp3 * LJ_FIX_1_501321110;
        let z1 = z1 * (-LJ_FIX_0_899976223);
        let z2 = z2 * (-LJ_FIX_2_562915447);
        let z3 = z3 * (-LJ_FIX_1_961570560) + z5;
        let z4 = z4 * (-LJ_FIX_0_390180644) + z5;

        let tmp0 = tmp0 + z1 + z3;
        let tmp1 = tmp1 + z2 + z4;
        let tmp2 = tmp2 + z2 + z3;
        let tmp3 = tmp3 + z1 + z4;

        workspace[col] = descale(tmp10 + tmp3, LJ_CONST_BITS - LJ_PASS1_BITS);
        workspace[7 * 8 + col] = descale(tmp10 - tmp3, LJ_CONST_BITS - LJ_PASS1_BITS);
        workspace[8 + col] = descale(tmp11 + tmp2, LJ_CONST_BITS - LJ_PASS1_BITS);
        workspace[6 * 8 + col] = descale(tmp11 - tmp2, LJ_CONST_BITS - LJ_PASS1_BITS);
        workspace[2 * 8 + col] = descale(tmp12 + tmp1, LJ_CONST_BITS - LJ_PASS1_BITS);
        workspace[5 * 8 + col] = descale(tmp12 - tmp1, LJ_CONST_BITS - LJ_PASS1_BITS);
        workspace[3 * 8 + col] = descale(tmp13 + tmp0, LJ_CONST_BITS - LJ_PASS1_BITS);
        workspace[4 * 8 + col] = descale(tmp13 - tmp0, LJ_CONST_BITS - LJ_PASS1_BITS);
    }

    // Row pass (unclamped output)
    let total_shift = LJ_CONST_BITS + LJ_PASS1_BITS + 3;

    for row in 0..8 {
        let base = row * 8;

        // DC-only shortcut
        if workspace[base + 1] == 0
            && workspace[base + 2] == 0
            && workspace[base + 3] == 0
            && workspace[base + 4] == 0
            && workspace[base + 5] == 0
            && workspace[base + 6] == 0
            && workspace[base + 7] == 0
        {
            let dcval = (descale(workspace[base], LJ_PASS1_BITS + 3) + 128) as i16;
            let out_base = row * stride;
            out_vector[out_base..out_base + 8].fill(dcval);
            continue;
        }

        let z2 = workspace[base + 2];
        let z3 = workspace[base + 6];

        let z1 = (z2 + z3) * LJ_FIX_0_541196100;
        let tmp2 = z1 + z3 * (-LJ_FIX_1_847759065);
        let tmp3 = z1 + z2 * LJ_FIX_0_765366865;

        let z2 = workspace[base];
        let z3 = workspace[base + 4];

        let tmp0 = (z2 + z3) << LJ_CONST_BITS;
        let tmp1 = (z2 - z3) << LJ_CONST_BITS;

        let tmp10 = tmp0 + tmp3;
        let tmp13 = tmp0 - tmp3;
        let tmp11 = tmp1 + tmp2;
        let tmp12 = tmp1 - tmp2;

        let tmp0 = workspace[base + 7];
        let tmp1 = workspace[base + 5];
        let tmp2 = workspace[base + 3];
        let tmp3 = workspace[base + 1];

        let z1 = tmp0 + tmp3;
        let z2 = tmp1 + tmp2;
        let z3 = tmp0 + tmp2;
        let z4 = tmp1 + tmp3;
        let z5 = (z3 + z4) * LJ_FIX_1_175875602;

        let tmp0 = tmp0 * LJ_FIX_0_298631336;
        let tmp1 = tmp1 * LJ_FIX_2_053119869;
        let tmp2 = tmp2 * LJ_FIX_3_072711026;
        let tmp3 = tmp3 * LJ_FIX_1_501321110;
        let z1 = z1 * (-LJ_FIX_0_899976223);
        let z2 = z2 * (-LJ_FIX_2_562915447);
        let z3 = z3 * (-LJ_FIX_1_961570560) + z5;
        let z4 = z4 * (-LJ_FIX_0_390180644) + z5;

        let tmp0 = tmp0 + z1 + z3;
        let tmp1 = tmp1 + z2 + z4;
        let tmp2 = tmp2 + z2 + z3;
        let tmp3 = tmp3 + z1 + z4;

        // Unclamped output: level shift (+128) but NO clamp to [0, 255]
        let out_base = row * stride;
        out_vector[out_base] = (descale(tmp10 + tmp3, total_shift) + 128) as i16;
        out_vector[out_base + 7] = (descale(tmp10 - tmp3, total_shift) + 128) as i16;
        out_vector[out_base + 1] = (descale(tmp11 + tmp2, total_shift) + 128) as i16;
        out_vector[out_base + 6] = (descale(tmp11 - tmp2, total_shift) + 128) as i16;
        out_vector[out_base + 2] = (descale(tmp12 + tmp1, total_shift) + 128) as i16;
        out_vector[out_base + 5] = (descale(tmp12 - tmp1, total_shift) + 128) as i16;
        out_vector[out_base + 3] = (descale(tmp13 + tmp0, total_shift) + 128) as i16;
        out_vector[out_base + 4] = (descale(tmp13 - tmp0, total_shift) + 128) as i16;
    }
}

/// Unclamped full 8x8 IDCT dispatch (non-tiered, for f32 output paths).
pub fn idct_int_auto_unclamped(coeffs: &mut [i32; 64], output: &mut [i16], stride: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(token) = archmage::X64V3Token::summon() {
            avx2::idct_int_avx2_unclamped(token, coeffs, output, stride);
            return;
        }
    }
    idct_int_wide_unclamped(coeffs, output, stride);
}

/// Unclamped tiered IDCT dispatch (default upsampling mode).
pub fn idct_int_tiered_unclamped(
    coeffs: &mut [i32; 64],
    output: &mut [i16],
    stride: usize,
    coeff_count: u8,
) {
    if coeff_count <= 1 {
        idct_int_dc_only_unclamped(coeffs[0], output, stride);
    } else {
        #[cfg(target_arch = "x86_64")]
        {
            if let Some(token) = archmage::X64V3Token::summon() {
                avx2::idct_int_avx2_unclamped(token, coeffs, output, stride);
                return;
            }
        }
        idct_int_wide_unclamped(coeffs, output, stride);
    }
}

/// Unclamped libjpeg-compatible tiered IDCT dispatch.
///
/// Uses i64 intermediates (Loeffler algorithm). Output is NOT clamped to [0,255].
pub fn idct_int_tiered_libjpeg_unclamped(
    coeffs: &mut [i32; 64],
    output: &mut [i16],
    stride: usize,
    coeff_count: u8,
) {
    if coeff_count <= 1 {
        idct_int_dc_only_unclamped(coeffs[0], output, stride);
    } else {
        idct_int_libjpeg_unclamped(coeffs, output, stride);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dc_only() {
        let mut coeffs = [0i32; 64];
        coeffs[0] = 1024; // DC coefficient

        let mut output = [0i16; 64];
        idct_int(&mut coeffs, &mut output, 8);

        // All values should be the same
        let first = output[0];
        for &v in &output {
            assert_eq!(v, first, "DC-only should produce uniform output");
        }
    }

    #[test]
    fn test_is_dc_only_int() {
        let dc_only = [
            100i32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
        ];
        assert!(is_dc_only_int(&dc_only));

        let not_dc_only = [
            100i32, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
        ];
        assert!(!is_dc_only_int(&not_dc_only));
    }

    #[test]
    fn test_output_range() {
        // Test with various coefficient patterns
        for dc in [-2000i32, -500, 0, 500, 2000] {
            let mut coeffs = [0i32; 64];
            coeffs[0] = dc;

            let mut output = [0i16; 64];
            idct_int(&mut coeffs, &mut output, 8);

            for &v in &output {
                assert!((0..=255).contains(&v), "Output {} out of range [0,255]", v);
            }
        }
    }

    #[test]
    fn test_idct_int_auto() {
        let mut coeffs = [0i32; 64];
        coeffs[0] = 512;
        coeffs[1] = 100;
        coeffs[8] = -50;

        let mut output = [0i16; 64];
        idct_int_auto(&mut coeffs.clone(), &mut output, 8);

        // Verify output is in valid range
        for &v in &output {
            assert!((0..=255).contains(&v));
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn test_avx2_matches_scalar() {
        let Some(token) = archmage::X64V3Token::summon() else {
            return;
        };

        // Test with random-ish pattern
        let mut coeffs_scalar = [0i32; 64];
        let mut coeffs_avx2 = [0i32; 64];

        for i in 0..64 {
            let v = ((i as i32 * 17 + 31) % 256) - 128;
            coeffs_scalar[i] = v * 8;
            coeffs_avx2[i] = v * 8;
        }

        let mut output_scalar = [0i16; 64];
        let mut output_avx2 = [0i16; 64];

        idct_int(&mut coeffs_scalar, &mut output_scalar, 8);
        avx2::idct_int_avx2(token, &mut coeffs_avx2, &mut output_avx2, 8);

        for i in 0..64 {
            assert_eq!(
                output_scalar[i], output_avx2[i],
                "Mismatch at {}: scalar={}, avx2={}",
                i, output_scalar[i], output_avx2[i]
            );
        }
    }

    #[test]
    fn test_wide_matches_scalar() {
        // Test with random-ish pattern
        let mut coeffs_scalar = [0i32; 64];

        for i in 0..64 {
            let v = ((i as i32 * 17 + 31) % 256) - 128;
            coeffs_scalar[i] = v * 8;
        }
        let coeffs_wide: [i32; 64] = coeffs_scalar;

        let mut output_scalar = [0i16; 64];
        let mut output_wide = [0i16; 64];

        idct_int(&mut coeffs_scalar, &mut output_scalar, 8);
        wide_simd::idct_int_wide(&coeffs_wide, &mut output_wide, 8);

        for i in 0..64 {
            assert_eq!(
                output_scalar[i], output_wide[i],
                "Mismatch at {}: scalar={}, wide={}",
                i, output_scalar[i], output_wide[i]
            );
        }
    }

    #[test]
    fn test_wide_with_stride() {
        // Test wide implementation with non-8 stride
        let coeffs: [i32; 64] = std::array::from_fn(|i| {
            let v = ((i as i32 * 17 + 31) % 256) - 128;
            v * 8
        });

        // Test with stride 16 (typical for interleaved output)
        let mut output_stride8 = [0i16; 64];
        let mut output_stride16 = [0i16; 128];

        wide_simd::idct_int_wide(&coeffs, &mut output_stride8, 8);
        wide_simd::idct_int_wide(&coeffs, &mut output_stride16, 16);

        // Compare row by row
        for row in 0..8 {
            for col in 0..8 {
                assert_eq!(
                    output_stride8[row * 8 + col],
                    output_stride16[row * 16 + col],
                    "Stride mismatch at ({}, {})",
                    row,
                    col
                );
            }
        }
    }

    #[test]
    fn test_wide_dc_patterns() {
        // Test various DC-only patterns
        for dc in [-2000i32, -500, 0, 500, 1000, 2000] {
            let mut coeffs = [0i32; 64];
            coeffs[0] = dc;

            let mut output = [0i16; 64];
            wide_simd::idct_int_wide(&coeffs, &mut output, 8);

            // All values should be same and in range
            let first = output[0];
            for (i, &v) in output.iter().enumerate() {
                assert!(
                    (0..=255).contains(&v),
                    "DC {} produced out-of-range {} at {}",
                    dc,
                    v,
                    i
                );
                // DC-only should produce uniform output (within rounding)
                assert!(
                    (v - first).abs() <= 1,
                    "DC {} non-uniform: {} vs {} at {}",
                    dc,
                    first,
                    v,
                    i
                );
            }
        }
    }

    #[test]
    fn test_wide_exhaustive() {
        // Test many coefficient patterns
        for seed in 0..100 {
            let coeffs: [i32; 64] = std::array::from_fn(|i| {
                let v = ((i as i32 * 17 + seed * 7 + 31) % 512) - 256;
                v * 4
            });

            let mut coeffs_scalar = coeffs;
            let mut output_scalar = [0i16; 64];
            let mut output_wide = [0i16; 64];

            idct_int(&mut coeffs_scalar, &mut output_scalar, 8);
            wide_simd::idct_int_wide(&coeffs, &mut output_wide, 8);

            for i in 0..64 {
                assert_eq!(
                    output_scalar[i], output_wide[i],
                    "Seed {}: Mismatch at {}: scalar={}, wide={}",
                    seed, i, output_scalar[i], output_wide[i]
                );
            }
        }
    }

    /// Verify i64 intermediates handle large dequantized coefficients.
    /// These magnitudes can occur at low quality levels (large quant values ×
    /// max-category coefficients). The i32 Jpegli IDCT wraps at these magnitudes.
    #[test]
    fn test_libjpeg_idct_large_coefficients() {
        // Coefficients that exceed i16 range — can occur at low quality (Q50 and below)
        // where quant values are large. With i64 intermediates these produce valid output.
        let mut coeffs = [0i32; 64];
        coeffs[0] = 40000; // DC
        coeffs[1] = -35000; // AC[0,1]

        let mut output = [0i16; 64];
        idct_int_tiered_libjpeg(&mut coeffs, &mut output, 8, 2);

        // Direct call should produce identical results (no truncation step)
        let mut coeffs2 = [0i32; 64];
        coeffs2[0] = 40000;
        coeffs2[1] = -35000;
        let mut output2 = [0i16; 64];
        idct_int_libjpeg(&mut coeffs2, &mut output2, 8);

        assert_eq!(
            output, output2,
            "tiered and direct libjpeg IDCT should produce identical results"
        );

        // All pixel values should be in valid range [0, 255]
        for &v in &output {
            assert!(
                (0..=255).contains(&v),
                "IDCT output {v} out of [0, 255] range"
            );
        }
    }

    /// Reference f64 IDCT for cross-validation. Type-II DCT inverse using
    /// the textbook cos() formula. Slow but maximally precise.
    fn reference_idct_f64(coeffs: &[i32; 64]) -> [f64; 64] {
        use core::f64::consts::PI;
        let mut output = [0.0f64; 64];

        for y in 0..8 {
            for x in 0..8 {
                let mut sum = 0.0f64;
                for v in 0..8 {
                    for u in 0..8 {
                        let cu = if u == 0 {
                            1.0 / core::f64::consts::SQRT_2
                        } else {
                            1.0
                        };
                        let cv = if v == 0 {
                            1.0 / core::f64::consts::SQRT_2
                        } else {
                            1.0
                        };
                        let cos_x = ((2 * x + 1) as f64 * u as f64 * PI / 16.0).cos();
                        let cos_y = ((2 * y + 1) as f64 * v as f64 * PI / 16.0).cos();
                        sum += cu * cv * coeffs[v * 8 + u] as f64 * cos_x * cos_y;
                    }
                }
                output[y * 8 + x] = sum / 4.0 + 128.0; // level shift
            }
        }
        output
    }

    /// Exhaustive IDCT cross-validation harness.
    ///
    /// Tests all integer IDCT variants against an f64 reference across a wide
    /// range of coefficient magnitudes (normal JPEG through extreme values).
    /// Each variant must produce output within `max_err` of the reference.
    #[test]
    fn test_idct_cross_validation_harness() {
        // Coefficient magnitude ranges to test:
        // - Normal JPEG: ±512 (Q90 typical)
        // - Low quality: ±2048 (Q50 typical)
        // - Large dequant: ±8000 (low quality with large quant values)
        // - Extreme: ±16000 (worst-case extended sequential)
        let magnitudes = [512, 2048, 4000, 8000, 16000];

        // Simple LCG PRNG for reproducible test data
        let mut rng = 0x1234_5678_9ABC_DEF0u64;
        let mut next = || -> i32 {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (rng >> 33) as i32
        };

        let mut total_blocks = 0u64;
        let mut max_err_loeffler = 0.0f64;
        let mut max_err_zune = 0.0f64;
        let mut max_err_zune_simd = 0.0f64;

        for &mag in &magnitudes {
            for _trial in 0..200 {
                // Generate random coefficient block scaled to magnitude
                let coeffs: [i32; 64] = core::array::from_fn(|_| next() % (2 * mag + 1) - mag);

                // f64 reference (textbook precision)
                let ref_output = reference_idct_f64(&coeffs);

                // Loeffler i64 (our primary IDCT for libjpeg compat)
                let mut coeffs_lj = coeffs;
                let mut out_lj = [0i16; 64];
                idct_int_libjpeg(&mut coeffs_lj, &mut out_lj, 8);

                // Zune-based scalar (12-bit, wrapping i32)
                let mut coeffs_zune = coeffs;
                let mut out_zune = [0i16; 64];
                idct_int(&mut coeffs_zune, &mut out_zune, 8);

                // Zune-based wide SIMD
                let mut out_wide = [0i16; 64];
                wide_simd::idct_int_wide(&coeffs, &mut out_wide, 8);

                // Compare each against reference
                for i in 0..64 {
                    let ref_clamped = ref_output[i].round().clamp(0.0, 255.0);

                    let err_lj = (out_lj[i] as f64 - ref_clamped).abs();
                    max_err_loeffler = max_err_loeffler.max(err_lj);

                    let err_zune = (out_zune[i] as f64 - ref_clamped).abs();
                    max_err_zune = max_err_zune.max(err_zune);

                    let err_wide = (out_wide[i] as f64 - ref_clamped).abs();
                    max_err_zune_simd = max_err_zune_simd.max(err_wide);

                    // Loeffler i64 must be within ±2 of reference at all magnitudes
                    assert!(
                        err_lj <= 2.0,
                        "Loeffler i64 error {err_lj} at pos {i}, mag={mag}, \
                         ref={ref_clamped}, got={}",
                        out_lj[i]
                    );
                }

                // Zune scalar vs SIMD must match exactly (same algorithm)
                assert_eq!(out_zune, out_wide, "zune scalar/SIMD mismatch at mag={mag}");

                total_blocks += 1;
            }
        }

        // Zune-based (12-bit i32) WILL have large errors at high magnitudes
        // due to wrapping arithmetic — that's expected and documented.
        // We just verify it doesn't panic.

        eprintln!(
            "IDCT harness: {total_blocks} blocks tested, \
             max_err: loeffler={max_err_loeffler:.1}, \
             zune={max_err_zune:.1}, zune_simd={max_err_zune_simd:.1}"
        );
    }

    /// Test that the Loeffler IDCT handles extreme coefficients without panic
    /// or producing out-of-range output. This covers the boundary between
    /// baseline (max coeff ±1023 * qval 255 = ±260,865) and extended sequential
    /// (max coeff ±1023 * qval 32767 = ±33,520,641).
    #[test]
    fn test_loeffler_extreme_coefficients() {
        // Worst case: all 64 coefficients at maximum magnitude
        for &mag in &[1000, 5000, 10000, 50000, 100000] {
            let coeffs_pos: [i32; 64] = [mag; 64];
            let mut coeffs = coeffs_pos;
            let mut output = [0i16; 64];
            idct_int_libjpeg(&mut coeffs, &mut output, 8);

            for (i, &v) in output.iter().enumerate() {
                assert!(
                    (0..=255).contains(&v),
                    "mag={mag} pos {i}: output {v} out of range"
                );
            }

            // Alternating signs (worst for intermediate sums)
            let coeffs_alt: [i32; 64] =
                core::array::from_fn(|i| if i % 2 == 0 { mag } else { -mag });
            let mut coeffs = coeffs_alt;
            let mut output = [0i16; 64];
            idct_int_libjpeg(&mut coeffs, &mut output, 8);

            for (i, &v) in output.iter().enumerate() {
                assert!(
                    (0..=255).contains(&v),
                    "mag={mag} alt pos {i}: output {v} out of range"
                );
            }
        }
    }
}
