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
//! - **generic**: Portable SIMD using magetypes generics with multi-tier dispatch (recommended)
//! - **avx2**: AVX2 intrinsics via archmage capability tokens (x86_64 only, kept for reference)
//! - **scalar**: Pure scalar fallback
//!
//! Benchmarks (8x8 IDCT block):
//! - x86_64 AVX2: generic 1.64x faster than scalar
//! - aarch64 NEON: generic 1.11x faster than scalar

use archmage::prelude::*;
use magetypes::simd::generic::i32x8 as GenericI32x8;

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
/// These are i64 to match libjpeg-turbo's `JLONG` type (`typedef long JLONG`
/// in jpegint.h: 64-bit on LP64 Linux/macOS, 32-bit on Windows/LLP64).
/// libjpeg-turbo's 8-bit C path nominally uses `MULTIPLY16C16`, but its
/// default definition is a plain full-width multiply — the INT16-truncating
/// variants are opt-in legacy defines (`SHORTxSHORT_32`) that modern builds
/// don't set. The 16-bit-truncating arithmetic lives in libjpeg-turbo's
/// x86/NEON SIMD islow (pmaddwd-style), which wraps dequantized values to
/// i16 on load. So: our i64 path matches libjpeg-turbo's C islow on LP64
/// exactly; for extreme dequantized values (|coeff x quant| > 32767) the
/// turbo SIMD, turbo Windows C, and turbo LP64 C paths all diverge from one
/// another, and we match the LP64 C (widest, no wrap) behavior.
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
    #[allow(unused_imports)]
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

    /// AVX2 libjpeg-exact islow IDCT with the i32-exactness guards.
    ///
    /// Same Loeffler butterfly and descale rounding as `idct_int_libjpeg`,
    /// in i32 lanes with an in-register transpose. Returns `false` (output
    /// untouched) when a guard trips so the caller can fall back to the
    /// scalar i64 kernel — see the guard derivation at the
    /// "SIMD libjpeg-exact IDCT" section below.
    ///
    /// Within the guards every pre-clamp output fits i16 (|sum| < 2^31
    /// before the >>18 implies |pixel| < 8192), so `packs_epi32`
    /// saturation never engages and pack equals the scalar `as i16` cast.
    #[arcane]
    #[allow(unused_assignments)]
    pub fn idct_int_libjpeg_avx2(
        _token: archmage::X64V3Token,
        in_vector: &[i32; 64],
        out_vector: &mut [i16],
        stride: usize,
        clamp_255: bool,
    ) -> bool {
        assert!(out_vector.len() >= stride * 7 + 8);

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

        // Guard: all values must fit the i16 window [-32768, 32767].
        // abs() maps i32::MIN to itself (sign bit set), which the mask
        // test rejects, so the window check is exact on every input.
        let out_of_window = _mm256_set1_epi32(!0x7FFF);
        macro_rules! rows_fit_i16 {
            () => {{
                let mut acc = _mm256_abs_epi32(row0);
                acc = _mm256_or_si256(acc, _mm256_abs_epi32(row1));
                acc = _mm256_or_si256(acc, _mm256_abs_epi32(row2));
                acc = _mm256_or_si256(acc, _mm256_abs_epi32(row3));
                acc = _mm256_or_si256(acc, _mm256_abs_epi32(row4));
                acc = _mm256_or_si256(acc, _mm256_abs_epi32(row5));
                acc = _mm256_or_si256(acc, _mm256_abs_epi32(row6));
                acc = _mm256_or_si256(acc, _mm256_abs_epi32(row7));
                let over = _mm256_and_si256(acc, out_of_window);
                _mm256_testz_si256(over, over) == 1
            }};
        }

        if !rows_fit_i16!() {
            return false;
        }

        // 13-bit islow constants
        let c4433 = _mm256_set1_epi32(LJ32_0_541196100);
        let cn15137 = _mm256_set1_epi32(-LJ32_1_847759065);
        let c6270 = _mm256_set1_epi32(LJ32_0_765366865);
        let c9633 = _mm256_set1_epi32(LJ32_1_175875602);
        let c2446 = _mm256_set1_epi32(LJ32_0_298631336);
        let c16819 = _mm256_set1_epi32(LJ32_2_053119869);
        let c25172 = _mm256_set1_epi32(LJ32_3_072711026);
        let c12299 = _mm256_set1_epi32(LJ32_1_501321110);
        let cn7373 = _mm256_set1_epi32(-LJ32_0_899976223);
        let cn20995 = _mm256_set1_epi32(-LJ32_2_562915447);
        let cn16069 = _mm256_set1_epi32(-LJ32_1_961570560);
        let cn3196 = _mm256_set1_epi32(-LJ32_0_390180644);
        let bias1 = _mm256_set1_epi32(LJ32_PASS1_BIAS);
        let bias2 = _mm256_set1_epi32(LJ32_PASS2_BIAS);

        macro_rules! islow_pass {
            ($bias:expr, $shift:literal) => {
                // Even part
                let z1 = _mm256_mullo_epi32(_mm256_add_epi32(row2, row6), c4433);
                let tmp2 = _mm256_add_epi32(z1, _mm256_mullo_epi32(row6, cn15137));
                let tmp3 = _mm256_add_epi32(z1, _mm256_mullo_epi32(row2, c6270));

                let tmp0 = _mm256_slli_epi32(_mm256_add_epi32(row0, row4), 13);
                let tmp1 = _mm256_slli_epi32(_mm256_sub_epi32(row0, row4), 13);

                let tmp10 = _mm256_add_epi32(_mm256_add_epi32(tmp0, tmp3), $bias);
                let tmp13 = _mm256_add_epi32(_mm256_sub_epi32(tmp0, tmp3), $bias);
                let tmp11 = _mm256_add_epi32(_mm256_add_epi32(tmp1, tmp2), $bias);
                let tmp12 = _mm256_add_epi32(_mm256_sub_epi32(tmp1, tmp2), $bias);

                // Odd part
                let z1o = _mm256_add_epi32(row7, row1);
                let z2o = _mm256_add_epi32(row5, row3);
                let z3o = _mm256_add_epi32(row7, row3);
                let z4o = _mm256_add_epi32(row5, row1);
                let z5 = _mm256_mullo_epi32(_mm256_add_epi32(z3o, z4o), c9633);

                let t0 = _mm256_mullo_epi32(row7, c2446);
                let t1 = _mm256_mullo_epi32(row5, c16819);
                let t2 = _mm256_mullo_epi32(row3, c25172);
                let t3 = _mm256_mullo_epi32(row1, c12299);
                let z1o = _mm256_mullo_epi32(z1o, cn7373);
                let z2o = _mm256_mullo_epi32(z2o, cn20995);
                let z3o = _mm256_add_epi32(_mm256_mullo_epi32(z3o, cn16069), z5);
                let z4o = _mm256_add_epi32(_mm256_mullo_epi32(z4o, cn3196), z5);

                let t0 = _mm256_add_epi32(t0, _mm256_add_epi32(z1o, z3o));
                let t1 = _mm256_add_epi32(t1, _mm256_add_epi32(z2o, z4o));
                let t2 = _mm256_add_epi32(t2, _mm256_add_epi32(z2o, z3o));
                let t3 = _mm256_add_epi32(t3, _mm256_add_epi32(z1o, z4o));

                // islow output ordering
                row0 = _mm256_srai_epi32(_mm256_add_epi32(tmp10, t3), $shift);
                row7 = _mm256_srai_epi32(_mm256_sub_epi32(tmp10, t3), $shift);
                row1 = _mm256_srai_epi32(_mm256_add_epi32(tmp11, t2), $shift);
                row6 = _mm256_srai_epi32(_mm256_sub_epi32(tmp11, t2), $shift);
                row2 = _mm256_srai_epi32(_mm256_add_epi32(tmp12, t1), $shift);
                row5 = _mm256_srai_epi32(_mm256_sub_epi32(tmp12, t1), $shift);
                row3 = _mm256_srai_epi32(_mm256_add_epi32(tmp13, t0), $shift);
                row4 = _mm256_srai_epi32(_mm256_sub_epi32(tmp13, t0), $shift);
            };
        }

        // Pass 1 (columns)
        islow_pass!(bias1, 11);
        // Guard: pass-1 outputs must fit i16 or pass-2 products overflow.
        if !rows_fit_i16!() {
            return false;
        }
        transpose_8x8_i32(
            _token, &mut row0, &mut row1, &mut row2, &mut row3, &mut row4, &mut row5, &mut row6,
            &mut row7,
        );
        // Pass 2 (rows) with the level shift folded into the bias
        islow_pass!(bias2, 18);
        transpose_8x8_i32(
            _token, &mut row0, &mut row1, &mut row2, &mut row3, &mut row4, &mut row5, &mut row6,
            &mut row7,
        );

        let mut pos = 0;
        macro_rules! pack_store_islow {
            ($r0:expr, $r1:expr) => {
                let packed = _mm256_packs_epi32($r0, $r1);
                let result = if clamp_255 {
                    clamp_avx(_token, packed)
                } else {
                    packed
                };
                let reordered = _mm256_permute4x64_epi64(result, shuffle(3, 1, 2, 0));

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

        pack_store_islow!(row0, row1);
        pack_store_islow!(row2, row3);
        pack_store_islow!(row4, row5);
        pack_store_islow!(row6, row7);
        let _ = pos;
        true
    }
}

// =============================================================================
// Portable SIMD Implementation using magetypes generics
// =============================================================================

/// Portable SIMD IDCT using magetypes generics with multi-tier dispatch.
///
/// This implementation uses `GenericI32x8` for the butterfly operations and
/// a scalar transpose (magetypes i32x8 has no transpose_8x8).
///
/// Performance vs scalar (standalone benchmark):
/// - x86_64 AVX2: 1.64x faster (26.9 ns vs 44.2 ns per block)
/// - aarch64 NEON: 1.11x faster (583.9 ns vs 646.9 ns per block via qemu)
mod wide_simd {
    use super::SCALE_BITS;
    use archmage::prelude::*;
    use magetypes::simd::generic::i32x8 as GenericI32x8;

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

    /// Portable SIMD IDCT using magetypes generics.
    ///
    /// Uses `#[magetypes]` for multi-tier SIMD dispatch (AVX2, NEON, WASM128, scalar).
    pub fn idct_int_wide(in_vector: &[i32; 64], out_vector: &mut [i16], stride: usize) {
        incant!(idct_int_wide_impl(in_vector, out_vector, stride));
    }

    #[magetypes(v3, neon, wasm128, scalar)]
    fn idct_int_wide_impl(
        token: Token,
        in_vector: &[i32; 64],
        out_vector: &mut [i16],
        stride: usize,
    ) {
        #[allow(non_camel_case_types)]
        type i32x8 = GenericI32x8<Token>;

        // Load 8 rows as i32x8 vectors
        let mut rows: [i32x8; 8] = core::array::from_fn(|i| {
            i32x8::from_array(
                token,
                *<&[i32; 8]>::try_from(&in_vector[i * 8..(i + 1) * 8]).unwrap(),
            )
        });

        // First pass (columns) - process all 8 columns in parallel
        idct_pass_generic(token, &mut rows, i32x8::splat(token, 512), 10);

        // Transpose using to_array/from_array (no native i32x8 transpose)
        transpose_i32x8(token, &mut rows);

        // Second pass (rows)
        idct_pass_generic(token, &mut rows, i32x8::splat(token, SCALE_BITS), 17);

        // Transpose back to row-major order
        transpose_i32x8(token, &mut rows);

        // Extract and clamp to output with stride.
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

    /// Scalar 8x8 transpose for i32x8 vectors (no native transpose available).
    #[inline(always)]
    pub(super) fn transpose_i32x8<T: magetypes::simd::backends::I32x8Backend>(
        token: T,
        rows: &mut [GenericI32x8<T>; 8],
    ) {
        let r: [[i32; 8]; 8] = core::array::from_fn(|i| rows[i].to_array());
        for i in 0..8 {
            rows[i] = GenericI32x8::<T>::from_array(token, core::array::from_fn(|j| r[j][i]));
        }
    }

    /// One pass of IDCT butterfly using generic i32x8 SIMD.
    ///
    /// This is the core IDCT computation, called twice (columns then rows).
    /// Note: uses `shl_const` and `shr_arithmetic_const` instead of `<<`/`>>`
    /// operators (not available on GenericI32x8).
    #[inline(always)]
    pub(super) fn idct_pass_generic<T: magetypes::simd::backends::I32x8Backend>(
        token: T,
        rows: &mut [GenericI32x8<T>; 8],
        scale_bits: GenericI32x8<T>,
        shift: i32,
    ) {
        #[allow(non_camel_case_types)]
        type i32x8<U> = GenericI32x8<U>;

        // Even part (rows 0, 2, 4, 6)
        let p1 = (rows[2] + rows[6]) * i32x8::splat(token, C2217);
        let t2 = p1 + rows[6] * i32x8::splat(token, CN7567);
        let t3 = p1 + rows[2] * i32x8::splat(token, C3135);

        let t0 = (rows[0] + rows[4]).shl_const::<12>();
        let t1 = (rows[0] - rows[4]).shl_const::<12>();

        let x0 = t0 + t3 + scale_bits;
        let x3 = t0 - t3 + scale_bits;
        let x1 = t1 + t2 + scale_bits;
        let x2 = t1 - t2 + scale_bits;

        // Odd part (rows 1, 3, 5, 7)
        let p3 = rows[7] + rows[3];
        let p4 = rows[5] + rows[1];
        let p1_odd = rows[7] + rows[1];
        let p2_odd = rows[5] + rows[3];
        let p5 = (p3 + p4) * i32x8::splat(token, C4816);

        let mut t0 = rows[7] * i32x8::splat(token, C1223);
        let mut t1 = rows[5] * i32x8::splat(token, C8410);
        let mut t2 = rows[3] * i32x8::splat(token, C12586);
        let mut t3 = rows[1] * i32x8::splat(token, C6149);

        let p1_final = p5 + p1_odd * i32x8::splat(token, CN3685);
        let p2_final = p5 + p2_odd * i32x8::splat(token, CN10497);
        let p3_final = p3 * i32x8::splat(token, CN8034);
        let p4_final = p4 * i32x8::splat(token, CN1597);

        t3 = t3 + p1_final + p4_final;
        t2 = t2 + p2_final + p3_final;
        t1 = t1 + p2_final + p4_final;
        t0 = t0 + p1_final + p3_final;

        // Combine even and odd parts, then shift
        // Note: shift is always 10 or 17, known at call site but not const generic.
        // Use match to dispatch to const generic shifts.
        match shift {
            10 => {
                rows[0] = (x0 + t3).shr_arithmetic_const::<10>();
                rows[1] = (x1 + t2).shr_arithmetic_const::<10>();
                rows[2] = (x2 + t1).shr_arithmetic_const::<10>();
                rows[3] = (x3 + t0).shr_arithmetic_const::<10>();
                rows[4] = (x3 - t0).shr_arithmetic_const::<10>();
                rows[5] = (x2 - t1).shr_arithmetic_const::<10>();
                rows[6] = (x1 - t2).shr_arithmetic_const::<10>();
                rows[7] = (x0 - t3).shr_arithmetic_const::<10>();
            }
            17 => {
                rows[0] = (x0 + t3).shr_arithmetic_const::<17>();
                rows[1] = (x1 + t2).shr_arithmetic_const::<17>();
                rows[2] = (x2 + t1).shr_arithmetic_const::<17>();
                rows[3] = (x3 + t0).shr_arithmetic_const::<17>();
                rows[4] = (x3 - t0).shr_arithmetic_const::<17>();
                rows[5] = (x2 - t1).shr_arithmetic_const::<17>();
                rows[6] = (x1 - t2).shr_arithmetic_const::<17>();
                rows[7] = (x0 - t3).shr_arithmetic_const::<17>();
            }
            _ => unreachable!("idct_pass_generic only supports shift=10 or shift=17"),
        }
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
    idct_int_portable(coeffs, output, stride);
}

/// Non-x86 full 8x8 IDCT: the magetypes-generic SIMD kernel normally
/// (NEON on aarch64, wasm128 on wasm32+simd128), but the dedicated scalar
/// kernel on wasm32 WITHOUT simd128.
///
/// On a no-simd128 wasm build the magetypes generic can't select its
/// wasm128 tier and degrades to a lane-by-lane scalar *emulation* of a
/// transpose-heavy 8-wide algorithm — measured ~60-68% slower than the
/// purpose-written scalar i64 IDCT under wasmtime (2026-06-13). The two
/// produce bit-identical output (`test_wide_matches_scalar`), so this is a
/// pure speed routing. The production wasm config ships simd128, where the
/// wasm128 tier wins, so this branch only affects no-simd128 builds.
#[inline]
fn idct_int_portable(coeffs: &mut [i32; 64], output: &mut [i16], stride: usize) {
    #[cfg(all(target_arch = "wasm32", not(target_feature = "simd128")))]
    idct_int(coeffs, output, stride);
    #[cfg(not(all(target_arch = "wasm32", not(target_feature = "simd128"))))]
    wide_simd::idct_int_wide(coeffs, output, stride);
}

/// Perform integer IDCT using AVX2 intrinsics via archmage capability token.
///
/// This is the legacy implementation kept for comparison. In most cases,
/// `idct_int_auto` (which uses `wide`) should be preferred as it's portable
/// and has similar performance.
#[cfg(target_arch = "x86_64")]
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
        // Portable fallback (NEON / wasm128 / scalar — see idct_int_portable;
        // routes around the slow magetypes scalar emulation on no-simd128 wasm).
        idct_int_portable(coeffs, output, stride);
    }
}

/// libjpeg-compatible tiered IDCT dispatch.
///
/// Uses DC-only fast path for single-coefficient blocks, otherwise the
/// guarded SIMD islow kernel (bit-identical to `idct_int_libjpeg`, falling
/// back to the scalar i64 kernel outside the i32-exactness guard).
pub fn idct_int_tiered_libjpeg(
    coeffs: &mut [i32; 64],
    output: &mut [i16],
    stride: usize,
    coeff_count: u8,
) {
    if coeff_count <= 1 {
        idct_int_dc_only(coeffs[0], output, stride);
    } else {
        idct_int_libjpeg_auto(coeffs, output, stride);
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

/// Unclamped wide SIMD IDCT using magetypes generics.
fn idct_int_wide_unclamped(in_vector: &[i32; 64], out_vector: &mut [i16], stride: usize) {
    incant!(idct_int_wide_unclamped_impl(in_vector, out_vector, stride));
}

#[magetypes(v3, neon, wasm128, scalar)]
fn idct_int_wide_unclamped_impl(
    token: Token,
    in_vector: &[i32; 64],
    out_vector: &mut [i16],
    stride: usize,
) {
    #[allow(non_camel_case_types)]
    type i32x8 = GenericI32x8<Token>;

    let mut rows: [i32x8; 8] = core::array::from_fn(|i| {
        i32x8::from_array(
            token,
            *<&[i32; 8]>::try_from(&in_vector[i * 8..(i + 1) * 8]).unwrap(),
        )
    });

    wide_simd::idct_pass_generic(token, &mut rows, i32x8::splat(token, 512), 10);
    wide_simd::transpose_i32x8(token, &mut rows);
    wide_simd::idct_pass_generic(token, &mut rows, i32x8::splat(token, SCALE_BITS), 17);
    wide_simd::transpose_i32x8(token, &mut rows);

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

// =============================================================================
// SIMD libjpeg-exact IDCT (guarded i32 islow)
// =============================================================================
//
// Same Loeffler butterfly and descale rounding as `idct_int_libjpeg`, computed
// in i32 lanes (all 8 columns per pass, like the 12-bit kernel above). Bit
// equality with the i64 scalar is guaranteed by two range guards whose
// worst-case bound is derived in `test_islow_i32_guard_bound_analysis`:
//
// - guard 1 (inputs): every |dequantized coefficient| must fit the i16 window
//   [-32768, 32767]. The largest pass-1 intermediate is then
//   L1max * 32768 + 1024 < 2^31 (L1max = sum of |form coefficients| ~= 61213).
// - guard 2 (pass-1 outputs): every |workspace value| must fit the same
//   window, bounding pass-2 intermediates by L1max * 32768 + bias < 2^31.
//
// Honestly-encoded JPEGs sit far inside both guards (|coeff| <= ~4096 for
// 8-bit imagery, workspace ~4x pixel scale); only near-adversarial streams
// trip them and fall back to the scalar i64 kernel, so output never changes.

/// i32 copies of the 13-bit constants for SIMD lanes.
const LJ32_0_298631336: i32 = 2446;
const LJ32_0_390180644: i32 = 3196;
const LJ32_0_541196100: i32 = 4433;
const LJ32_0_765366865: i32 = 6270;
const LJ32_0_899976223: i32 = 7373;
const LJ32_1_175875602: i32 = 9633;
const LJ32_1_501321110: i32 = 12299;
const LJ32_1_847759065: i32 = 15137;
const LJ32_1_961570560: i32 = 16069;
const LJ32_2_053119869: i32 = 16819;
const LJ32_2_562915447: i32 = 20995;
const LJ32_3_072711026: i32 = 25172;

/// Pass-1 rounding bias: 1 << (CONST_BITS - PASS1_BITS - 1).
const LJ32_PASS1_BIAS: i32 = 1 << (LJ_CONST_BITS - LJ_PASS1_BITS - 1);
/// Pass-2 rounding bias with the +128 level shift folded in:
/// (1 << 17) + (128 << 18). Folding is exact for arithmetic shifts because
/// 128 << 18 is a multiple of 2^18.
const LJ32_PASS2_BIAS: i32 = (1 << 17) + (128 << 18);

/// True when every lane of every row fits the i16 window [-32768, 32767].
///
/// Uses ones-complement abs (`x ^ (x >> 31)`): exact magnitude for x >= 0,
/// magnitude minus one for x < 0, and i32::MAX for i32::MIN — so OR-ing and
/// testing bits 15..31 accepts exactly the [-32768, 32767] window per lane.
#[inline(always)]
fn islow_rows_fit_i16<T: magetypes::simd::backends::I32x8Backend>(
    token: T,
    rows: &[GenericI32x8<T>; 8],
) -> bool {
    let acc = rows
        .iter()
        .map(|r| *r ^ r.shr_arithmetic_const::<31>())
        .reduce(|a, b| a | b)
        .expect("rows is non-empty");
    let over = acc & GenericI32x8::<T>::splat(token, !0x7FFF);
    !over.simd_ne(GenericI32x8::<T>::splat(token, 0)).any_true()
}

/// One islow pass over 8 lanes (the exact `idct_int_libjpeg` butterfly).
///
/// `bias` is the descale rounding constant, added to the even-part
/// accumulators so each output receives it exactly once before the
/// arithmetic shift (descale). Pass 1 uses bias 1024 / shift 11; pass 2
/// uses the level-shift-folded bias / shift 18.
#[inline(always)]
fn islow_pass_generic<T: magetypes::simd::backends::I32x8Backend>(
    token: T,
    rows: &mut [GenericI32x8<T>; 8],
    bias: GenericI32x8<T>,
    shift: i32,
) {
    #[allow(non_camel_case_types)]
    type i32x8<U> = GenericI32x8<U>;

    // Even part
    let z2 = rows[2];
    let z3 = rows[6];
    let z1 = (z2 + z3) * i32x8::splat(token, LJ32_0_541196100);
    let tmp2 = z1 + z3 * i32x8::splat(token, -LJ32_1_847759065);
    let tmp3 = z1 + z2 * i32x8::splat(token, LJ32_0_765366865);

    let tmp0 = (rows[0] + rows[4]).shl_const::<13>();
    let tmp1 = (rows[0] - rows[4]).shl_const::<13>();

    let tmp10 = tmp0 + tmp3 + bias;
    let tmp13 = tmp0 - tmp3 + bias;
    let tmp11 = tmp1 + tmp2 + bias;
    let tmp12 = tmp1 - tmp2 + bias;

    // Odd part
    let t0 = rows[7];
    let t1 = rows[5];
    let t2 = rows[3];
    let t3 = rows[1];

    let z1 = t0 + t3;
    let z2 = t1 + t2;
    let z3 = t0 + t2;
    let z4 = t1 + t3;
    let z5 = (z3 + z4) * i32x8::splat(token, LJ32_1_175875602);

    let t0 = t0 * i32x8::splat(token, LJ32_0_298631336);
    let t1 = t1 * i32x8::splat(token, LJ32_2_053119869);
    let t2 = t2 * i32x8::splat(token, LJ32_3_072711026);
    let t3 = t3 * i32x8::splat(token, LJ32_1_501321110);
    let z1 = z1 * i32x8::splat(token, -LJ32_0_899976223);
    let z2 = z2 * i32x8::splat(token, -LJ32_2_562915447);
    let z3 = z3 * i32x8::splat(token, -LJ32_1_961570560) + z5;
    let z4 = z4 * i32x8::splat(token, -LJ32_0_390180644) + z5;

    let t0 = t0 + z1 + z3;
    let t1 = t1 + z2 + z4;
    let t2 = t2 + z2 + z3;
    let t3 = t3 + z1 + z4;

    match shift {
        11 => {
            rows[0] = (tmp10 + t3).shr_arithmetic_const::<11>();
            rows[7] = (tmp10 - t3).shr_arithmetic_const::<11>();
            rows[1] = (tmp11 + t2).shr_arithmetic_const::<11>();
            rows[6] = (tmp11 - t2).shr_arithmetic_const::<11>();
            rows[2] = (tmp12 + t1).shr_arithmetic_const::<11>();
            rows[5] = (tmp12 - t1).shr_arithmetic_const::<11>();
            rows[3] = (tmp13 + t0).shr_arithmetic_const::<11>();
            rows[4] = (tmp13 - t0).shr_arithmetic_const::<11>();
        }
        18 => {
            rows[0] = (tmp10 + t3).shr_arithmetic_const::<18>();
            rows[7] = (tmp10 - t3).shr_arithmetic_const::<18>();
            rows[1] = (tmp11 + t2).shr_arithmetic_const::<18>();
            rows[6] = (tmp11 - t2).shr_arithmetic_const::<18>();
            rows[2] = (tmp12 + t1).shr_arithmetic_const::<18>();
            rows[5] = (tmp12 - t1).shr_arithmetic_const::<18>();
            rows[3] = (tmp13 + t0).shr_arithmetic_const::<18>();
            rows[4] = (tmp13 - t0).shr_arithmetic_const::<18>();
        }
        _ => unreachable!("islow_pass_generic only supports shift=11 or shift=18"),
    }
}

/// SIMD islow over all 8 columns/rows. Returns `false` (output untouched)
/// when a range guard trips; the caller must fall back to the scalar kernel.
#[magetypes(v3, neon, wasm128, scalar)]
fn idct_libjpeg_wide_impl(
    token: Token,
    in_vector: &[i32; 64],
    out_vector: &mut [i16],
    stride: usize,
    clamp_255: bool,
) -> bool {
    #[allow(non_camel_case_types)]
    type i32x8 = GenericI32x8<Token>;

    let mut rows: [i32x8; 8] = core::array::from_fn(|i| {
        i32x8::from_array(
            token,
            *<&[i32; 8]>::try_from(&in_vector[i * 8..(i + 1) * 8]).unwrap(),
        )
    });

    // Guard 1: inputs must fit i16 or pass-1 products can exceed i32.
    if !islow_rows_fit_i16(token, &rows) {
        return false;
    }
    islow_pass_generic(token, &mut rows, i32x8::splat(token, LJ32_PASS1_BIAS), 11);
    // Guard 2: pass-1 outputs must fit i16 or pass-2 products can exceed i32.
    if !islow_rows_fit_i16(token, &rows) {
        return false;
    }
    wide_simd::transpose_i32x8(token, &mut rows);
    islow_pass_generic(token, &mut rows, i32x8::splat(token, LJ32_PASS2_BIAS), 18);
    wide_simd::transpose_i32x8(token, &mut rows);

    let min_len = stride * 7 + 8;
    assert!(out_vector.len() >= min_len);
    let out = &mut out_vector[..min_len];
    let mut out_pos = 0;
    if clamp_255 {
        for row in &rows {
            let arr = row.to_array();
            for (j, &val) in arr.iter().enumerate() {
                out[out_pos + j] = val.clamp(0, 255) as i16;
            }
            out_pos += stride;
        }
    } else {
        // Unclamped: level-shifted but not clamped; values stay well inside
        // i16 within the guards (|pre-shift| < 2^31 implies |pixel| < 7800).
        for row in &rows {
            let arr = row.to_array();
            for (j, &val) in arr.iter().enumerate() {
                out[out_pos + j] = val as i16;
            }
            out_pos += stride;
        }
    }
    true
}

/// libjpeg-exact IDCT with SIMD fast path.
///
/// Bit-identical to [`idct_int_libjpeg`] on every input: blocks whose
/// coefficients or pass-1 outputs exceed the i32-exactness guard fall back
/// to the scalar i64 kernel (see the guard derivation above).
pub fn idct_int_libjpeg_auto(coeffs: &mut [i32; 64], output: &mut [i16], stride: usize) {
    if is_dc_only_int(coeffs) {
        return idct_int_dc_only(coeffs[0], output, stride);
    }
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(token) = archmage::X64V3Token::summon() {
            if !avx2::idct_int_libjpeg_avx2(token, coeffs, output, stride, true) {
                idct_int_libjpeg(coeffs, output, stride);
            }
            return;
        }
    }
    // wasm32 without simd128: idct_libjpeg_wide_impl degrades to slow scalar
    // emulation; the dedicated scalar islow is ~60-68% faster there (measured,
    // bit-identical). simd128/x86/NEON builds keep the SIMD path.
    #[cfg(all(target_arch = "wasm32", not(target_feature = "simd128")))]
    idct_int_libjpeg(coeffs, output, stride);
    #[cfg(not(all(target_arch = "wasm32", not(target_feature = "simd128"))))]
    if !incant!(idct_libjpeg_wide_impl(coeffs, output, stride, true)) {
        idct_int_libjpeg(coeffs, output, stride);
    }
}

/// Unclamped libjpeg-exact IDCT with SIMD fast path.
///
/// Bit-identical to [`idct_int_libjpeg_unclamped`] on every input.
pub fn idct_int_libjpeg_auto_unclamped(coeffs: &mut [i32; 64], output: &mut [i16], stride: usize) {
    if is_dc_only_int(coeffs) {
        return idct_int_dc_only_unclamped(coeffs[0], output, stride);
    }
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(token) = archmage::X64V3Token::summon() {
            if !avx2::idct_int_libjpeg_avx2(token, coeffs, output, stride, false) {
                idct_int_libjpeg_unclamped(coeffs, output, stride);
            }
            return;
        }
    }
    if !incant!(idct_libjpeg_wide_impl(coeffs, output, stride, false)) {
        idct_int_libjpeg_unclamped(coeffs, output, stride);
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
/// Output is NOT clamped to \[0,255\]. Uses the guarded SIMD islow kernel
/// (bit-identical to `idct_int_libjpeg_unclamped` on every input).
pub fn idct_int_tiered_libjpeg_unclamped(
    coeffs: &mut [i32; 64],
    output: &mut [i16],
    stride: usize,
    coeff_count: u8,
) {
    if coeff_count <= 1 {
        idct_int_dc_only_unclamped(coeffs[0], output, stride);
    } else {
        idct_int_libjpeg_auto_unclamped(coeffs, output, stride);
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

    #[cfg(target_arch = "x86_64")]
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

    /// Simple LCG for reproducible randomized test blocks.
    struct Lcg(u64);
    impl Lcg {
        fn next_i32(&mut self) -> i32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 33) as i32
        }
        /// Uniform in [-mag, mag].
        fn coeff(&mut self, mag: i32) -> i32 {
            self.next_i32().rem_euclid(2 * mag + 1) - mag
        }
    }

    /// The guarded SIMD islow kernel must be bit-identical to the scalar i64
    /// kernel on EVERY input: inside the guards by exact i32 arithmetic,
    /// outside them by falling back to the scalar kernel.
    #[test]
    fn test_libjpeg_simd_bit_exact_vs_scalar() {
        let mut rng = Lcg(0xD1CE_5EED_0BAD_F00D);

        let check = |coeffs: &[i32; 64], stride: usize, what: &str| {
            let buf_len = stride * 7 + 8;

            let mut c1 = *coeffs;
            let mut scalar_clamped = vec![0i16; buf_len];
            idct_int_libjpeg(&mut c1, &mut scalar_clamped, stride);
            let mut c2 = *coeffs;
            let mut auto_clamped = vec![0i16; buf_len];
            idct_int_libjpeg_auto(&mut c2, &mut auto_clamped, stride);
            assert_eq!(scalar_clamped, auto_clamped, "clamped mismatch: {what}");

            let mut c3 = *coeffs;
            let mut scalar_raw = vec![0i16; buf_len];
            idct_int_libjpeg_unclamped(&mut c3, &mut scalar_raw, stride);
            let mut c4 = *coeffs;
            let mut auto_raw = vec![0i16; buf_len];
            idct_int_libjpeg_auto_unclamped(&mut c4, &mut auto_raw, stride);
            assert_eq!(scalar_raw, auto_raw, "unclamped mismatch: {what}");
        };

        // In-guard magnitudes: realistic JPEG through the full i16 window.
        for &mag in &[16, 256, 2047, 8192, 32767] {
            for trial in 0..300 {
                let stride = [8, 11, 16][trial % 3];

                // Dense random block
                let dense: [i32; 64] = core::array::from_fn(|_| rng.coeff(mag));
                check(&dense, stride, &format!("dense mag={mag} trial={trial}"));

                // Sparse block (8 random nonzero positions — typical JPEG)
                let mut sparse = [0i32; 64];
                for _ in 0..8 {
                    let pos = rng.next_i32().rem_euclid(64) as usize;
                    sparse[pos] = rng.coeff(mag);
                }
                check(&sparse, stride, &format!("sparse mag={mag} trial={trial}"));
            }

            // Structured patterns that exercise the per-column/row scalar
            // shortcuts the SIMD kernel doesn't have.
            let mut col0 = [0i32; 64];
            for r in 0..8 {
                col0[r * 8] = rng.coeff(mag);
            }
            check(&col0, 8, &format!("col0-only mag={mag}"));

            let mut row0 = [0i32; 64];
            for c in 0..8 {
                row0[c] = rng.coeff(mag);
            }
            check(&row0, 8, &format!("row0-only mag={mag}"));

            let mut dc_only = [0i32; 64];
            dc_only[0] = rng.coeff(mag);
            check(&dc_only, 8, &format!("dc-only mag={mag}"));

            let mut single_ac = [0i32; 64];
            single_ac[63] = rng.coeff(mag);
            check(&single_ac, 8, &format!("single-ac mag={mag}"));
        }

        // Guard-1 trips: inputs beyond the i16 window must fall back and
        // still match the scalar kernel exactly.
        for &mag in &[40_000, 200_000, 30_000_000] {
            for trial in 0..50 {
                let dense: [i32; 64] = core::array::from_fn(|_| rng.coeff(mag));
                check(&dense, 8, &format!("out-of-guard mag={mag} trial={trial}"));

                // Mixed: one huge coefficient among realistic ones
                let mut mixed: [i32; 64] = core::array::from_fn(|_| rng.coeff(300));
                let pos = rng.next_i32().rem_euclid(64) as usize;
                mixed[pos] = if trial % 2 == 0 { mag } else { -mag };
                check(&mixed, 8, &format!("mixed-spike mag={mag} trial={trial}"));
            }
        }

        // Guard-2 trip: all inputs inside the i16 window, but sign-aligned so
        // pass-1 outputs exceed it (|w| ~= 29.9 * 32000 >> 32767). The fallback
        // must engage and match scalar exactly.
        let aligned: [i32; 64] = [32_000; 64];
        check(&aligned, 8, "guard2 all-positive 32000");
        let neg_aligned: [i32; 64] = [-32_000; 64];
        check(&neg_aligned, 8, "guard2 all-negative 32000");

        // i32::MIN coefficients must not panic and must match scalar
        // (ones-complement abs maps MIN to i32::MAX, tripping guard 1).
        let mut min_block = [0i32; 64];
        min_block[0] = i32::MIN;
        min_block[9] = i32::MIN;
        check(&min_block, 8, "i32::MIN coefficients");
    }

    /// Pin the GENERIC magetypes islow tiers to the scalar kernel directly:
    /// on x86_64 with AVX2 the auto dispatch shortcuts to the intrinsics
    /// kernel, so `test_libjpeg_simd_bit_exact_vs_scalar` alone would leave
    /// the incant tiers (v3 generic / neon / wasm128 / scalar) unexercised
    /// on this machine.
    #[test]
    fn test_libjpeg_generic_simd_bit_exact_vs_scalar() {
        let mut rng = Lcg(0x9E3779B97F4A7C15);
        for &mag in &[300, 2047, 32767, 200_000] {
            for trial in 0..200 {
                let stride = [8, 13][trial % 2];
                let coeffs: [i32; 64] = core::array::from_fn(|_| rng.coeff(mag));
                let buf_len = stride * 7 + 8;

                let mut c1 = coeffs;
                let mut scalar_out = vec![0i16; buf_len];
                idct_int_libjpeg(&mut c1, &mut scalar_out, stride);

                let mut generic_out = vec![0i16; buf_len];
                if !incant!(idct_libjpeg_wide_impl(
                    &coeffs,
                    &mut generic_out,
                    stride,
                    true
                )) {
                    // Guard tripped (expected for the 200_000 population):
                    // the production fallback is the scalar kernel.
                    let mut c2 = coeffs;
                    idct_int_libjpeg(&mut c2, &mut generic_out, stride);
                }
                assert_eq!(
                    scalar_out, generic_out,
                    "generic islow mismatch at mag={mag} trial={trial}"
                );

                let mut c3 = coeffs;
                let mut scalar_raw = vec![0i16; buf_len];
                idct_int_libjpeg_unclamped(&mut c3, &mut scalar_raw, stride);
                let mut generic_raw = vec![0i16; buf_len];
                if !incant!(idct_libjpeg_wide_impl(
                    &coeffs,
                    &mut generic_raw,
                    stride,
                    false
                )) {
                    let mut c4 = coeffs;
                    idct_int_libjpeg_unclamped(&mut c4, &mut generic_raw, stride);
                }
                assert_eq!(
                    scalar_raw, generic_raw,
                    "generic islow unclamped mismatch at mag={mag} trial={trial}"
                );
            }
        }
    }

    /// Derive the worst-case islow intermediate magnitude by exact L1-norm
    /// propagation through the butterfly, and assert the SIMD guards keep
    /// every i32 intermediate below 2^31. This is the provenance of the
    /// `islow_rows_fit_i16` guard windows.
    #[test]
    fn test_islow_i32_guard_bound_analysis() {
        // Each value is tracked as its vector of coefficients over the 8
        // 1-D inputs; ops mirror `islow_pass_generic` exactly. f64 is exact
        // here (all magnitudes << 2^53).
        type Form = [f64; 8];
        fn unit(i: usize) -> Form {
            let mut f = [0.0; 8];
            f[i] = 1.0;
            f
        }
        fn add(a: Form, b: Form) -> Form {
            core::array::from_fn(|i| a[i] + b[i])
        }
        fn sub(a: Form, b: Form) -> Form {
            core::array::from_fn(|i| a[i] - b[i])
        }
        fn scale(a: Form, c: f64) -> Form {
            core::array::from_fn(|i| a[i] * c)
        }
        fn l1(a: &Form) -> f64 {
            a.iter().map(|x| x.abs()).sum()
        }

        let d: [Form; 8] = core::array::from_fn(unit);
        let mut worst: f64 = 0.0;
        let mut track = |f: Form| -> Form {
            worst = worst.max(l1(&f));
            f
        };

        // Even part
        let z1 = track(scale(add(d[2], d[6]), 4433.0));
        let tmp2 = track(add(z1, scale(d[6], -15137.0)));
        let tmp3 = track(add(z1, scale(d[2], 6270.0)));
        let tmp0 = track(scale(add(d[0], d[4]), 8192.0));
        let tmp1 = track(scale(sub(d[0], d[4]), 8192.0));
        let tmp10 = track(add(tmp0, tmp3));
        let tmp13 = track(sub(tmp0, tmp3));
        let tmp11 = track(add(tmp1, tmp2));
        let tmp12 = track(sub(tmp1, tmp2));

        // Odd part
        let z1 = track(add(d[7], d[1]));
        let z2 = track(add(d[5], d[3]));
        let z3 = track(add(d[7], d[3]));
        let z4 = track(add(d[5], d[1]));
        let z5 = track(scale(add(z3, z4), 9633.0));
        let t0 = track(scale(d[7], 2446.0));
        let t1 = track(scale(d[5], 16819.0));
        let t2 = track(scale(d[3], 25172.0));
        let t3 = track(scale(d[1], 12299.0));
        let z1 = track(scale(z1, -7373.0));
        let z2 = track(scale(z2, -20995.0));
        let z3 = track(add(scale(z3, -16069.0), z5));
        let z4 = track(add(scale(z4, -3196.0), z5));
        let t0 = track(add(add(t0, z1), z3));
        let t1 = track(add(add(t1, z2), z4));
        let t2 = track(add(add(t2, z2), z3));
        let t3 = track(add(add(t3, z1), z4));

        // Final pre-descale sums (the binding terms)
        let finals = [
            track(add(tmp10, t3)),
            track(sub(tmp10, t3)),
            track(add(tmp11, t2)),
            track(sub(tmp11, t2)),
            track(add(tmp12, t1)),
            track(sub(tmp12, t1)),
            track(add(tmp13, t0)),
            track(sub(tmp13, t0)),
        ];

        eprintln!("islow worst-case L1 over one pass: {worst}");

        // Guard window: inputs (pass 1) and workspace values (pass 2) are
        // both confined to [-32768, 32767] by `islow_rows_fit_i16`.
        const WINDOW: f64 = 32768.0;
        const I32_MAX: f64 = 2147483647.0;
        // Pass 1: worst intermediate + rounding bias must fit i32.
        assert!(
            worst * WINDOW + 1024.0 <= I32_MAX,
            "pass-1 worst-case {} overflows i32",
            worst * WINDOW + 1024.0
        );
        // Pass 2: worst intermediate + rounding bias + folded level shift.
        let pass2_bias = 131072.0 + 33554432.0;
        assert!(
            worst * WINDOW + pass2_bias <= I32_MAX,
            "pass-2 worst-case {} overflows i32",
            worst * WINDOW + pass2_bias
        );

        // Sanity: the workspace bound used in the doc comment (|w| within
        // ~30x input magnitude) holds, so honest images never trip guard 2.
        let w_gain = finals.iter().map(l1).fold(0.0, f64::max) / 2048.0;
        eprintln!("islow pass-1 output worst-case gain: {w_gain:.2}x input magnitude");
        assert!(w_gain < 32.0);
    }

    /// IEEE 1180-style accuracy comparison of both integer kernels against
    /// the f64 reference IDCT. Reports per-kernel max/mean/RMS error and the
    /// kernel-vs-kernel divergence rate. Loeffler must meet the IEEE 1180
    /// peak-pixel-error limit (<= 1); the 12-bit kernel's stats are the
    /// measured answer to "which kernel is more correct".
    #[test]
    fn test_idct_accuracy_stats_vs_reference() {
        struct Stats {
            n: u64,
            sum_err: f64,
            sum_abs: f64,
            sum_sq: f64,
            max_abs: f64,
        }
        impl Stats {
            fn new() -> Self {
                Stats {
                    n: 0,
                    sum_err: 0.0,
                    sum_abs: 0.0,
                    sum_sq: 0.0,
                    max_abs: 0.0,
                }
            }
            fn push(&mut self, got: i16, want: f64) {
                let e = got as f64 - want;
                self.n += 1;
                self.sum_err += e;
                self.sum_abs += e.abs();
                self.sum_sq += e * e;
                self.max_abs = self.max_abs.max(e.abs());
            }
            fn report(&self, name: &str) -> (f64, f64, f64) {
                let mean = self.sum_err / self.n as f64;
                let mean_abs = self.sum_abs / self.n as f64;
                let rms = (self.sum_sq / self.n as f64).sqrt();
                eprintln!(
                    "  {name:<14} ppe={:>4.1}  mean={mean:>+8.5}  mean_abs={mean_abs:.5}  rms={rms:.5}",
                    self.max_abs
                );
                (self.max_abs, mean_abs, rms)
            }
        }

        let mut rng = Lcg(0x1EEE_1180_CAFE_BABE);
        // (range, blocks, label): IEEE 1180 prescribes L=H=256/5/300 over
        // 10000 random blocks; plus a sparse "Q85 photo"-shaped population.
        let configs: [(i32, usize, &str); 4] = [
            (256, 10_000, "ieee [-256,255]"),
            (300, 10_000, "ieee [-300,300]"),
            (5, 10_000, "ieee [-5,5]"),
            (0, 10_000, "sparse-q85"),
        ];

        for (mag, blocks, label) in configs {
            let mut lj = Stats::new();
            let mut zune = Stats::new();
            let mut diverge = 0u64;
            let mut diverge_max = 0i32;
            let mut total = 0u64;

            for _ in 0..blocks {
                let coeffs: [i32; 64] = if mag > 0 {
                    core::array::from_fn(|_| rng.coeff(mag))
                } else {
                    // Sparse realistic block: DC plus a handful of low-freq
                    // ACs at photographic magnitudes.
                    let mut c = [0i32; 64];
                    c[0] = rng.coeff(1023);
                    for _ in 0..6 {
                        let pos = (rng.next_i32().rem_euclid(20) + 1) as usize;
                        c[pos] = rng.coeff(300);
                    }
                    c
                };

                let reference = reference_idct_f64(&coeffs);

                let mut c1 = coeffs;
                let mut out_lj = [0i16; 64];
                idct_int_libjpeg(&mut c1, &mut out_lj, 8);
                let mut c2 = coeffs;
                let mut out_zune = [0i16; 64];
                idct_int(&mut c2, &mut out_zune, 8);

                for i in 0..64 {
                    let want = reference[i].round().clamp(0.0, 255.0);
                    lj.push(out_lj[i], want);
                    zune.push(out_zune[i], want);
                    let d = (out_lj[i] as i32 - out_zune[i] as i32).abs();
                    if d != 0 {
                        diverge += 1;
                        diverge_max = diverge_max.max(d);
                    }
                    total += 1;
                }
            }

            eprintln!("IDCT accuracy, {label} ({blocks} blocks):");
            let (lj_ppe, _, _) = lj.report("loeffler-13bit");
            zune.report("zune-12bit");
            eprintln!(
                "  kernels diverge on {:.3}% of pixels (max diff {diverge_max})",
                100.0 * diverge as f64 / total as f64
            );

            // IEEE 1180 peak-pixel-error requirement for a compliant IDCT.
            assert!(
                lj_ppe <= 1.0,
                "loeffler kernel exceeds IEEE 1180 ppe=1 on {label}"
            );
        }
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
