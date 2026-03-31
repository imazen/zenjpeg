//! ARM NEON SIMD implementations using archmage capability tokens.
//!
//! These functions provide NEON-optimized implementations for ARM AArch64.
//! The `neon_` prefix distinguishes them from x86 AVX2 versions.
//!
//! # Token Model
//!
//! Each function requires a NeonToken that proves NEON is available.
//! NEON is baseline on AArch64, so `NeonToken::summon()` always succeeds.
//!
//! # Vector Width
//!
//! NEON is 128-bit (4-wide for f32), compared to AVX2's 256-bit (8-wide).
//! 8x8 DCT operations are implemented as two 4x4 blocks.

#![cfg(target_arch = "aarch64")]

use archmage::{NeonToken, SimdToken, arcane, rite};
use core::arch::aarch64::*;
use safe_unaligned_simd::aarch64 as safe_simd;

// ============================================================================
// DCT Constants (same as x86 version)
// ============================================================================

const WC4_0: f32 = 0.541196100146197;
const WC4_1: f32 = 1.3065629648763764;

const WC8_0: f32 = 0.5097955791041592;
const WC8_1: f32 = 0.6013448869350453;
const WC8_2: f32 = 0.8999762231364156;
const WC8_3: f32 = 2.5629154477415055;

const SQRT2: f32 = 1.41421356237;

// ============================================================================
// 4x4 Transpose (Foundation for 8x8)
// ============================================================================

/// In-place 4x4 transpose on 4 float32x4_t registers using NEON.
///
/// Uses the zip1/zip2 (interleave) pattern:
/// - Phase 1: Zip adjacent pairs (creates 2x4 blocks)
/// - Phase 2: Zip the results (creates transposed 4x4)
///
/// This is the building block for 8x8 transposes.
#[rite]
#[inline]
fn neon_transpose_4x4_inplace_inner(_token: NeonToken, r: &mut [float32x4_t; 4]) {
    // Phase 1: Interleave pairs
    // After this, q0/q1 have rows 0&1 interleaved, q2/q3 have rows 2&3 interleaved
    let q0 = vzip1q_f32(r[0], r[1]); // [r0[0], r1[0], r0[1], r1[1]]
    let q1 = vzip2q_f32(r[0], r[1]); // [r0[2], r1[2], r0[3], r1[3]]
    let q2 = vzip1q_f32(r[2], r[3]); // [r2[0], r3[0], r2[1], r3[1]]
    let q3 = vzip2q_f32(r[2], r[3]); // [r2[2], r3[2], r2[3], r3[3]]

    // Phase 2: Interleave 64-bit pairs (f32x2 treated as single 64-bit unit)
    // This completes the transpose
    r[0] = vreinterpretq_f32_f64(vzip1q_f64(
        vreinterpretq_f64_f32(q0),
        vreinterpretq_f64_f32(q2),
    )); // Column 0
    r[1] = vreinterpretq_f32_f64(vzip2q_f64(
        vreinterpretq_f64_f32(q0),
        vreinterpretq_f64_f32(q2),
    )); // Column 1
    r[2] = vreinterpretq_f32_f64(vzip1q_f64(
        vreinterpretq_f64_f32(q1),
        vreinterpretq_f64_f32(q3),
    )); // Column 2
    r[3] = vreinterpretq_f32_f64(vzip2q_f64(
        vreinterpretq_f64_f32(q1),
        vreinterpretq_f64_f32(q3),
    )); // Column 3
}

/// Public wrapper for 4x4 transpose.
#[arcane]
pub fn neon_transpose_4x4_inplace(token: NeonToken, r: &mut [float32x4_t; 4]) {
    neon_transpose_4x4_inplace_inner(token, r);
}

// ============================================================================
// 8x8 Transpose (Two 4x4 Blocks)
// ============================================================================

/// In-place 8x8 transpose implemented as two independent 4x4 transposes.
///
/// Input: 8 rows of 8 f32 values (64 total)
/// Output: 8 columns (transposed)
///
/// Since NEON is 4-wide, we split the 8x8 into:
/// - Top-left 4x4 block (rows 0-3, cols 0-3)
/// - Top-right 4x4 block (rows 0-3, cols 4-7)
/// - Bottom-left 4x4 block (rows 4-7, cols 0-3)
/// - Bottom-right 4x4 block (rows 4-7, cols 4-7)
///
/// Then transpose each 4x4 independently and reassemble.
#[rite]
#[inline]
fn neon_transpose_8x8_inplace_inner(token: NeonToken, data: &mut [f32; 64]) {
    // Load 8 rows as 16 float32x4_t registers (2 per row)
    let r0_lo = safe_simd::vld1q_f32(data[0..4].try_into().unwrap());
    let r0_hi = safe_simd::vld1q_f32(data[4..8].try_into().unwrap());
    let r1_lo = safe_simd::vld1q_f32(data[8..12].try_into().unwrap());
    let r1_hi = safe_simd::vld1q_f32(data[12..16].try_into().unwrap());
    let r2_lo = safe_simd::vld1q_f32(data[16..20].try_into().unwrap());
    let r2_hi = safe_simd::vld1q_f32(data[20..24].try_into().unwrap());
    let r3_lo = safe_simd::vld1q_f32(data[24..28].try_into().unwrap());
    let r3_hi = safe_simd::vld1q_f32(data[28..32].try_into().unwrap());
    let r4_lo = safe_simd::vld1q_f32(data[32..36].try_into().unwrap());
    let r4_hi = safe_simd::vld1q_f32(data[36..40].try_into().unwrap());
    let r5_lo = safe_simd::vld1q_f32(data[40..44].try_into().unwrap());
    let r5_hi = safe_simd::vld1q_f32(data[44..48].try_into().unwrap());
    let r6_lo = safe_simd::vld1q_f32(data[48..52].try_into().unwrap());
    let r6_hi = safe_simd::vld1q_f32(data[52..56].try_into().unwrap());
    let r7_lo = safe_simd::vld1q_f32(data[56..60].try_into().unwrap());
    let r7_hi = safe_simd::vld1q_f32(data[60..64].try_into().unwrap());

    // Transpose top-left 4x4
    let mut tl = [r0_lo, r1_lo, r2_lo, r3_lo];
    neon_transpose_4x4_inplace_inner(token, &mut tl);

    // Transpose top-right 4x4
    let mut tr = [r0_hi, r1_hi, r2_hi, r3_hi];
    neon_transpose_4x4_inplace_inner(token, &mut tr);

    // Transpose bottom-left 4x4
    let mut bl = [r4_lo, r5_lo, r6_lo, r7_lo];
    neon_transpose_4x4_inplace_inner(token, &mut bl);

    // Transpose bottom-right 4x4
    let mut br = [r4_hi, r5_hi, r6_hi, r7_hi];
    neon_transpose_4x4_inplace_inner(token, &mut br);

    // Store transposed blocks
    // After transpose, what were rows are now columns:
    // tl[0] = column 0, lanes 0-3 (from original rows 0-3)
    // bl[0] = column 0, lanes 4-7 (from original rows 4-7)
    safe_simd::vst1q_f32((&mut data[0..4]).try_into().unwrap(), tl[0]);
    safe_simd::vst1q_f32((&mut data[4..8]).try_into().unwrap(), bl[0]);
    safe_simd::vst1q_f32((&mut data[8..12]).try_into().unwrap(), tl[1]);
    safe_simd::vst1q_f32((&mut data[12..16]).try_into().unwrap(), bl[1]);
    safe_simd::vst1q_f32((&mut data[16..20]).try_into().unwrap(), tl[2]);
    safe_simd::vst1q_f32((&mut data[20..24]).try_into().unwrap(), bl[2]);
    safe_simd::vst1q_f32((&mut data[24..28]).try_into().unwrap(), tl[3]);
    safe_simd::vst1q_f32((&mut data[28..32]).try_into().unwrap(), bl[3]);
    safe_simd::vst1q_f32((&mut data[32..36]).try_into().unwrap(), tr[0]);
    safe_simd::vst1q_f32((&mut data[36..40]).try_into().unwrap(), br[0]);
    safe_simd::vst1q_f32((&mut data[40..44]).try_into().unwrap(), tr[1]);
    safe_simd::vst1q_f32((&mut data[44..48]).try_into().unwrap(), br[1]);
    safe_simd::vst1q_f32((&mut data[48..52]).try_into().unwrap(), tr[2]);
    safe_simd::vst1q_f32((&mut data[52..56]).try_into().unwrap(), br[2]);
    safe_simd::vst1q_f32((&mut data[56..60]).try_into().unwrap(), tr[3]);
    safe_simd::vst1q_f32((&mut data[60..64]).try_into().unwrap(), br[3]);
}

/// Public wrapper for 8x8 transpose.
#[arcane]
pub fn neon_transpose_8x8(token: NeonToken, data: &mut [f32; 64]) {
    neon_transpose_8x8_inplace_inner(token, data);
}

// ============================================================================
// DCT Butterfly Operations (4-wide)
// ============================================================================

/// 2-point DCT butterfly using NEON FMA.
///
/// Implements: (m0 + m1, m0 - m1)
#[rite]
#[inline]
fn neon_dct1d_2_inner(_token: NeonToken, m0: &mut float32x4_t, m1: &mut float32x4_t) {
    let sum = vaddq_f32(*m0, *m1);
    let diff = vsubq_f32(*m0, *m1);
    *m0 = sum;
    *m1 = diff;
}

/// 4-point DCT butterfly using NEON FMA.
///
/// Implements the standard 4-point DCT transform with FMA optimizations.
#[rite]
#[inline]
fn neon_dct1d_4_inner(token: NeonToken, m: &mut [float32x4_t; 4]) {
    // First layer: (m0+m3, m1+m2, m1-m2, m0-m3)
    let sum03 = vaddq_f32(m[0], m[3]);
    let sum12 = vaddq_f32(m[1], m[2]);
    let diff12 = vsubq_f32(m[1], m[2]);
    let diff03 = vsubq_f32(m[0], m[3]);

    // Second layer: apply 2-point DCT to (sum03, sum12)
    let mut t0 = sum03;
    let mut t1 = sum12;
    neon_dct1d_2_inner(token, &mut t0, &mut t1);

    // Apply WC4 coefficients to differences using FMA
    let wc4_0 = vdupq_n_f32(WC4_0);
    let wc4_1 = vdupq_n_f32(WC4_1);

    m[0] = t0;
    m[1] = vfmaq_f32(vmulq_f32(diff12, wc4_0), diff03, wc4_1); // diff12 * WC4_0 + diff03 * WC4_1
    m[2] = t1;
    m[3] = vfmsq_f32(vmulq_f32(diff12, wc4_1), diff03, wc4_0); // diff12 * WC4_1 - diff03 * WC4_0
}

/// 8-point DCT butterfly using NEON FMA (processes 4 blocks in parallel).
///
/// This is the core 1D DCT-II transform applied to 4 parallel streams.
#[rite]
#[inline]
fn neon_dct1d_8_inner(token: NeonToken, m: &mut [float32x4_t; 8]) {
    // First layer: butterfly on opposite ends
    let sum07 = vaddq_f32(m[0], m[7]);
    let sum16 = vaddq_f32(m[1], m[6]);
    let sum25 = vaddq_f32(m[2], m[5]);
    let sum34 = vaddq_f32(m[3], m[4]);
    let diff07 = vsubq_f32(m[0], m[7]);
    let diff16 = vsubq_f32(m[1], m[6]);
    let diff25 = vsubq_f32(m[2], m[5]);
    let diff34 = vsubq_f32(m[3], m[4]);

    // Apply 4-point DCT to sums
    let mut even = [sum07, sum16, sum25, sum34];
    neon_dct1d_4_inner(token, &mut even);

    // Apply WC8 coefficients to differences using FMA
    let wc8_0 = vdupq_n_f32(WC8_0);
    let wc8_1 = vdupq_n_f32(WC8_1);
    let wc8_2 = vdupq_n_f32(WC8_2);
    let wc8_3 = vdupq_n_f32(WC8_3);

    // Odd part (complex FMA chains)
    let t0 = vfmaq_f32(vmulq_f32(diff07, wc8_0), diff34, wc8_1);
    let t1 = vfmaq_f32(vmulq_f32(diff16, wc8_2), diff25, wc8_3);
    let t2 = vfmsq_f32(vmulq_f32(diff16, wc8_3), diff25, wc8_2);
    let t3 = vfmsq_f32(vmulq_f32(diff07, wc8_1), diff34, wc8_0);

    let odd0 = vaddq_f32(t0, t1);
    let odd1 = vsubq_f32(t0, t1);
    let odd2 = vaddq_f32(t2, t3);
    let odd3 = vsubq_f32(t2, t3);

    // Interleave even and odd results
    m[0] = even[0];
    m[1] = odd0;
    m[2] = even[1];
    m[3] = odd1;
    m[4] = even[2];
    m[5] = odd2;
    m[6] = even[3];
    m[7] = odd3;
}

// ============================================================================
// Forward DCT 8x8
// ============================================================================

/// Forward DCT 8x8 using NEON.
///
/// Processes the DCT as two 4x4 blocks due to NEON's 4-wide registers.
/// Each 4x4 block undergoes:
/// 1. Load 4 rows
/// 2. Apply 1D DCT to each row (4-point, extended to handle 8 points via splitting)
/// 3. Transpose
/// 4. Apply 1D DCT to each column
/// 5. Transpose back
///
/// Note: This is a simplified version. Full 8-point DCT needs special handling.
/// For now, this demonstrates the pattern.
#[arcane]
pub fn neon_forward_dct_8x8(token: NeonToken, input: &[f32; 64], output: &mut [f32; 64]) {
    // Load all 8 rows as float32x4_t pairs
    let mut rows_lo: [float32x4_t; 8] = [vdupq_n_f32(0.0); 8];
    let mut rows_hi: [float32x4_t; 8] = [vdupq_n_f32(0.0); 8];

    for i in 0..8 {
        rows_lo[i] = safe_simd::vld1q_f32(&input[i * 8..][..4].try_into().unwrap());
        rows_hi[i] = safe_simd::vld1q_f32(&input[i * 8 + 4..][..4].try_into().unwrap());

        // Apply 1D DCT to each row
        neon_dct1d_8_inner(token, &mut rows_lo);
        neon_dct1d_8_inner(token, &mut rows_hi);

        // Transpose
        let mut temp = [0.0f32; 64];
        for i in 0..8 {
            safe_simd::vst1q_f32(&mut temp[i * 8..][..4].try_into().unwrap(), rows_lo[i]);
            safe_simd::vst1q_f32(&mut temp[i * 8 + 4..][..4].try_into().unwrap(), rows_hi[i]);
        }
        neon_transpose_8x8_inplace_inner(token, &mut temp);

        // Reload transposed data
        for i in 0..8 {
            rows_lo[i] = safe_simd::vld1q_f32(&temp[i * 8..][..4].try_into().unwrap());
            rows_hi[i] = safe_simd::vld1q_f32(&temp[i * 8 + 4..][..4].try_into().unwrap());
        }

        // Apply 1D DCT to each column (now rows after transpose)
        neon_dct1d_8_inner(token, &mut rows_lo);
        neon_dct1d_8_inner(token, &mut rows_hi);

        // Store result
        for i in 0..8 {
            safe_simd::vst1q_f32(&mut output[i * 8..][..4].try_into().unwrap(), rows_lo[i]);
            safe_simd::vst1q_f32(
                &mut output[i * 8 + 4..][..4].try_into().unwrap(),
                rows_hi[i],
            );
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: run transpose test inside #[arcane] so NEON intrinsics are safe.
    #[arcane]
    fn transpose_4x4_test_inner(_token: NeonToken) {
        let input = [
            safe_simd::vld1q_f32(&[0.0, 1.0, 2.0, 3.0]),
            safe_simd::vld1q_f32(&[4.0, 5.0, 6.0, 7.0]),
            safe_simd::vld1q_f32(&[8.0, 9.0, 10.0, 11.0]),
            safe_simd::vld1q_f32(&[12.0, 13.0, 14.0, 15.0]),
        ];

        let mut r = input;
        neon_transpose_4x4_inplace(_token, &mut r);

        let mut col0 = [0.0f32; 4];
        let mut col1 = [0.0f32; 4];
        let mut col2 = [0.0f32; 4];
        let mut col3 = [0.0f32; 4];

        safe_simd::vst1q_f32((&mut col0[0..4]).try_into().unwrap(), r[0]);
        safe_simd::vst1q_f32((&mut col1[0..4]).try_into().unwrap(), r[1]);
        safe_simd::vst1q_f32((&mut col2[0..4]).try_into().unwrap(), r[2]);
        safe_simd::vst1q_f32((&mut col3[0..4]).try_into().unwrap(), r[3]);

        assert_eq!(col0, [0.0, 4.0, 8.0, 12.0]);
        assert_eq!(col1, [1.0, 5.0, 9.0, 13.0]);
        assert_eq!(col2, [2.0, 6.0, 10.0, 14.0]);
        assert_eq!(col3, [3.0, 7.0, 11.0, 15.0]);
    }

    #[test]
    fn test_neon_transpose_4x4() {
        if let Some(token) = NeonToken::summon() {
            transpose_4x4_test_inner(token);
        }
    }

    #[test]
    fn test_neon_transpose_8x8() {
        if let Some(token) = NeonToken::summon() {
            let mut input = [0.0f32; 64];
            for i in 0..64 {
                input[i] = i as f32;
            }

            neon_transpose_8x8(token, &mut input);

            // Check a few key positions
            assert_eq!(input[0], 0.0); // Was [0,0], still [0,0]
            assert_eq!(input[1], 8.0); // Was [1,0], now [0,1]
            assert_eq!(input[8], 1.0); // Was [0,1], now [1,0]
            assert_eq!(input[9], 9.0); // Was [1,1], still [1,1]
        }
    }
}

// ============================================================================
// Integer IDCT (for decoder)
// ============================================================================

/// Integer IDCT 8x8 using NEON.
///
/// Implements the libjpeg-turbo Loeffler algorithm with NEON intrinsics.
/// Processes two 4-wide columns in parallel using int32x4_t.
#[arcane]
pub fn neon_idct_int_8x8(_token: NeonToken, input: &[i32; 64], output: &mut [i16], stride: usize) {
    // Constants for Loeffler IDCT (13-bit fixed-point)
    let _fix_0_298631336 = vdupq_n_s32(2446);
    let _fix_0_390180644 = vdupq_n_s32(3196);
    let _fix_0_541196100 = vdupq_n_s32(4433);
    let _fix_0_765366865 = vdupq_n_s32(6270);
    let _fix_0_899976223 = vdupq_n_s32(7373);
    let _fix_1_175875602 = vdupq_n_s32(9633);
    let _fix_1_501321110 = vdupq_n_s32(12299);
    let _fix_1_847759065 = vdupq_n_s32(15137);
    let _fix_1_961570560 = vdupq_n_s32(16069);
    let _fix_2_053119869 = vdupq_n_s32(16819);
    let _fix_2_562915447 = vdupq_n_s32(20995);
    let _fix_3_072711026 = vdupq_n_s32(25172);

    const CONST_BITS: i32 = 13;
    const PASS1_BITS: i32 = 2;

    // DC-only fast path
    let all_ac_zero = true;
    for i in 1..64 {
        if input[i] != 0 {
            break;
        }

        if all_ac_zero {
            let dc = ((input[0] + 4 + 1024) >> 3).clamp(0, 255) as i16;
            let dc_vec = vdupq_n_s16(dc);
            let mut pos = 0;
            for _ in 0..8 {
                safe_simd::vst1q_s16((&mut output[pos..][..8]).try_into().unwrap(), dc_vec);
                pos += stride;
            }
            return;
        }

        // Full IDCT - load rows
        let mut rows: [int32x4x2_t; 8] = [int32x4x2_t(vdupq_n_s32(0), vdupq_n_s32(0)); 8];
        for i in 0..8 {
            rows[i] = int32x4x2_t(
                safe_simd::vld1q_s32(&input[i * 8..][..4].try_into().unwrap()),
                safe_simd::vld1q_s32(&input[i * 8 + 4..][..4].try_into().unwrap()),
            );
        }

        // Pass 1: process columns (simplified - would need full butterfly ops)
        // TODO: Implement full NEON column pass
        // TODO: Implement Loeffler IDCT algorithm (see zune-jpeg/src/idct/neon.rs)
        unimplemented!("NEON integer IDCT not yet implemented");
    }
}

// ============================================================================
// YCbCr Color Conversion (for decoder)
// ============================================================================

/// Convert YCbCr to RGB using NEON (16 pixels at once).
///
/// TODO: Implement using zune-jpeg/src/color_convert/neon64.rs as reference.
/// Requires vmlal_lane_s16 and proper lane selection patterns.
#[arcane]
pub fn neon_ycbcr_to_rgb(
    _token: NeonToken,
    _y: &[i16; 16],
    _cb: &[i16; 16],
    _cr: &[i16; 16],
    _rgb: &mut [u8; 48],
) {
    unimplemented!("NEON YCbCr→RGB not yet implemented - see zune-jpeg for reference");
}

// ============================================================================
// Chroma Upsampling (for decoder)
// ============================================================================

/// H2V1 upsampling (horizontal 2x) using triangle filter.
///
/// Implements the 3:1 weighting filter: output[2*i] = input[i],
/// output[2*i+1] = (3*input[i] + input[i+1] + 2) >> 2
#[arcane]
pub fn neon_upsample_h2v1(
    _token: NeonToken,
    input: &[f32],
    in_width: usize,
    output: &mut [f32],
    out_width: usize,
) {
    assert_eq!(out_width, in_width * 2);

    let v_three = vdupq_n_f32(3.0);
    let v_quarter = vdupq_n_f32(0.25);

    // First pixel
    output[0] = input[0];

    // Process 4 input pixels at a time → 8 output pixels
    let chunks = in_width / 4;
    for i in 0..chunks {
        let curr = safe_simd::vld1q_f32(&input[i * 4..][..4].try_into().unwrap());
        let next = safe_simd::vld1q_f32(&input[i * 4 + 1..][..4].try_into().unwrap());

        // even = curr, odd = (3*curr + next) * 0.25
        let odd = vmulq_f32(vfmaq_f32(next, curr, v_three), v_quarter);

        // Interleave even and odd
        let out0 = vzip1q_f32(curr, odd);
        let out1 = vzip2q_f32(curr, odd);

        safe_simd::vst1q_f32(&mut output[i * 8..][..4].try_into().unwrap(), out0);
        safe_simd::vst1q_f32(&mut output[i * 8 + 4..][..4].try_into().unwrap(), out1);

        // Last pixel
        output[out_width - 1] = input[in_width - 1];
    }
}
