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
//! - **wide**: Portable SIMD using `wide` crate with `multiversion` (recommended)
//! - **avx2**: Raw AVX2 intrinsics (x86_64 only, kept for reference)
//! - **scalar**: Pure scalar fallback
//!
//! Benchmarks (8x8 IDCT block):
//! - x86_64 AVX2: wide 1.64x faster than scalar
//! - aarch64 NEON: wide 1.11x faster than scalar

#![allow(dead_code)]

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

    // Fill first row, then advance 7 times (8 rows total)
    let mut out = out_vector;
    out[..8].fill(coeff);
    for _ in 0..7 {
        out = &mut out[stride..];
        out[..8].fill(coeff);
    }
}

/// Check if all AC coefficients are zero (DC-only block).
#[inline]
pub fn is_dc_only_int(coeffs: &[i32; 64]) -> bool {
    coeffs[1..].iter().all(|&x| x == 0)
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
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn clamp_avx(reg: __m256i) -> __m256i {
        let min_s = _mm256_set1_epi16(0);
        let max_s = _mm256_set1_epi16(255);
        let max_v = _mm256_max_epi16(reg, min_s);
        _mm256_min_epi16(max_v, max_s)
    }

    /// In-register 8x8 transpose for i32 values.
    #[target_feature(enable = "avx2")]
    unsafe fn transpose_8x8_i32(
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
    /// # Safety
    /// Requires AVX2 support.
    #[target_feature(enable = "avx2")]
    #[allow(unused_assignments)] // pos is incremented in macro but last value is unused
    pub unsafe fn idct_int_avx2(in_vector: &mut [i32; 64], out_vector: &mut [i16], stride: usize) {
        // Load all 8 rows
        let mut row0 = _mm256_loadu_si256(in_vector[0..].as_ptr().cast());
        let mut row1 = _mm256_loadu_si256(in_vector[8..].as_ptr().cast());
        let mut row2 = _mm256_loadu_si256(in_vector[16..].as_ptr().cast());
        let mut row3 = _mm256_loadu_si256(in_vector[24..].as_ptr().cast());
        let mut row4 = _mm256_loadu_si256(in_vector[32..].as_ptr().cast());
        let mut row5 = _mm256_loadu_si256(in_vector[40..].as_ptr().cast());
        let mut row6 = _mm256_loadu_si256(in_vector[48..].as_ptr().cast());
        let mut row7 = _mm256_loadu_si256(in_vector[56..].as_ptr().cast());

        // Check for DC-only (all AC = 0)
        let ac_check = _mm256_loadu_si256(in_vector[1..].as_ptr().cast());
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
                _mm_storeu_si128(out_vector[pos..pos + 8].as_mut_ptr().cast(), idct_value);
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
            &mut row0, &mut row1, &mut row2, &mut row3, &mut row4, &mut row5, &mut row6, &mut row7,
        );

        // Second pass (rows)
        dct_pass!(cscale, 17);

        // Transpose back
        transpose_8x8_i32(
            &mut row0, &mut row1, &mut row2, &mut row3, &mut row4, &mut row5, &mut row6, &mut row7,
        );

        // Pack and store
        let mut pos = 0;

        macro_rules! pack_store {
            ($r0:expr, $r1:expr) => {
                let packed = _mm256_packs_epi32($r0, $r1);
                let clamped = clamp_avx(packed);
                let reordered = _mm256_permute4x64_epi64(clamped, shuffle(3, 1, 2, 0));

                _mm_storeu_si128(
                    out_vector[pos..pos + 8].as_mut_ptr().cast(),
                    _mm256_extracti128_si256::<0>(reordered),
                );
                pos += stride;
                _mm_storeu_si128(
                    out_vector[pos..pos + 8].as_mut_ptr().cast(),
                    _mm256_extracti128_si256::<1>(reordered),
                );
                pos += stride;
            };
        }

        pack_store!(row0, row1);
        pack_store!(row2, row3);
        pack_store!(row4, row5);
        pack_store!(row6, row7);
    }
}

// =============================================================================
// Portable SIMD Implementation using `wide` crate
// =============================================================================

/// Portable SIMD IDCT using `wide` crate with `multiversion` for cross-platform support.
///
/// This implementation uses `wide::i32x8` for the butterfly operations and
/// `wide::i32x8::transpose` for the 8x8 matrix transpose.
///
/// Performance vs scalar (standalone benchmark):
/// - x86_64 AVX2: 1.64x faster (26.9 ns vs 44.2 ns per block)
/// - aarch64 NEON: 1.11x faster (583.9 ns vs 646.9 ns per block via qemu)
mod wide_simd {
    use super::SCALE_BITS;
    use multiversion::multiversion;
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
    /// IMPORTANT: Uses `#[multiversion]` to enable SIMD on each target.
    /// Without this, `wide` falls back to scalar and is slower!
    ///
    /// Targets (in priority order):
    /// - x86_64+avx2: Uses AVX2 for i32x8 ops and transpose
    /// - x86_64+sse4.1: Uses SSE4.1 (i32x8 = 2x __m128i)
    /// - aarch64+neon: Uses NEON for i32x8 ops
    /// - default: Scalar fallback
    #[multiversion(targets(
        "x86_64+avx2",
        "x86_64+sse4.1",
        "x86+avx2",
        "x86+sse4.1",
        "aarch64+neon",
        "arm+neon",
        "wasm32+simd128",
    ))]
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

        // Extract and clamp to output with stride
        let mut out_pos = 0;
        for row in &rows {
            let arr = row.to_array();
            for (j, &val) in arr.iter().enumerate() {
                out_vector[out_pos + j] = val.clamp(0, 255) as i16;
            }
            out_pos += stride;
        }
    }

    /// One pass of IDCT butterfly using i32x8 SIMD.
    ///
    /// This is the core IDCT computation, called twice (columns then rows).
    #[inline(always)]
    fn idct_pass(rows: &mut [i32x8; 8], scale_bits: i32x8, shift: i32) {
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
/// Uses the portable `wide` crate implementation with `multiversion` for
/// cross-platform SIMD support (AVX2, SSE4.1, NEON).
///
/// # Arguments
/// * `coeffs` - Input dequantized DCT coefficients (not modified)
/// * `output` - Output pixel buffer (i16 in range [0, 255])
/// * `stride` - Stride between output rows
#[inline]
pub fn idct_int_auto(coeffs: &mut [i32; 64], output: &mut [i16], stride: usize) {
    // Use portable wide implementation (multiversion handles dispatch)
    wide_simd::idct_int_wide(coeffs, output, stride);
}

/// Perform integer IDCT using raw AVX2 intrinsics.
///
/// This is the legacy implementation kept for comparison. In most cases,
/// `idct_int_auto` (which uses `wide`) should be preferred as it's portable
/// and has similar performance.
///
/// # Safety
/// Caller must ensure AVX2 is available.
#[cfg(all(
    feature = "unsafe_simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[inline]
pub unsafe fn idct_int_avx2_raw(coeffs: &mut [i32; 64], output: &mut [i16], stride: usize) {
    avx2::idct_int_avx2(coeffs, output, stride);
}

/// Tiered IDCT selection based on coefficient count.
///
/// Selects the optimal IDCT implementation based on how many non-zero
/// coefficients are in the block (in zigzag scan order):
/// - count <= 1: DC-only (just broadcast DC value)
/// - count <= 10: 4x4 IDCT (upper-left block only)
/// - count > 10: Full 8x8 IDCT with portable SIMD
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
    } else if coeff_count <= 10 {
        // 4x4 IDCT - coefficients only in upper-left 4x4 block
        idct_int_4x4(coeffs, output, stride);
    } else {
        // Full 8x8 IDCT with portable SIMD (multiversion handles dispatch)
        wide_simd::idct_int_wide(coeffs, output, stride);
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
/// Subtracts 128 to convert from [0,255] to centered [-128,127] range.
#[inline]
pub fn pixels_i16_to_f32_centered(pixels: &[i16; 64]) -> [f32; 64] {
    let mut out = [0.0f32; 64];
    for (i, &p) in pixels.iter().enumerate() {
        out[i] = p as f32 - 128.0;
    }
    out
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
                assert!(v >= 0 && v <= 255, "Output {} out of range [0,255]", v);
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
            assert!(v >= 0 && v <= 255);
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn test_avx2_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

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
        unsafe {
            avx2::idct_int_avx2(&mut coeffs_avx2, &mut output_avx2, 8);
        }

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
        let coeffs_wide: [i32; 64];

        for i in 0..64 {
            let v = ((i as i32 * 17 + 31) % 256) - 128;
            coeffs_scalar[i] = v * 8;
        }
        coeffs_wide = coeffs_scalar;

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
                    v >= 0 && v <= 255,
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
}
