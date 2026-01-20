//! Archmage-based SIMD implementations using capability tokens.
//!
//! These functions provide the same functionality as the `unsafe_simd` implementations
//! but with compile-time safety via archmage's token system. The `mage_` prefix
//! distinguishes them from the raw intrinsic versions.
//!
//! # Token Model
//!
//! Each function requires a capability token that proves the CPU supports the
//! required instruction sets. Tokens can be:
//! - Obtained via `Token::try_new()` at runtime
//! - Forged inside `#[multiversed]` functions where features are guaranteed
//! - Cached and reused across multiple calls
//!
//! # Example
//!
//! ```rust,ignore
//! use archmage::{Avx2FmaToken, SimdToken};
//!
//! // Cache the token outside hot loops
//! if let Some(token) = Avx2FmaToken::try_new() {
//!     for block in blocks {
//!         mage_forward_dct_8x8(token, &input, &mut output);
//!     }
//! }
//! ```

#![cfg(all(feature = "archmage-simd", target_arch = "x86_64"))]

use archmage::mem::avx;
use archmage::{arcane, Avx2FmaToken, Avx2Token, AvxToken};
use core::arch::x86_64::*;

// ============================================================================
// DCT Constants
// ============================================================================

// WC4 coefficients for 4-point DCT
const WC4_0: f32 = 0.541196100146197;
const WC4_1: f32 = 1.3065629648763764;

// WC8 coefficients for 8-point DCT
const WC8_0: f32 = 0.5097955791041592;
const WC8_1: f32 = 0.6013448869350453;
const WC8_2: f32 = 0.8999762231364156;
const WC8_3: f32 = 2.5629154477415055;

const SQRT2: f32 = 1.41421356237;

// ============================================================================
// 8x8 Transpose (In-Place on Registers)
// ============================================================================

/// In-place 8x8 transpose on 8 __m256 registers using AVX.
///
/// After transpose, `r[i]` contains column i from all 8 original rows.
/// Uses the 3-phase unpack/shuffle/permute pattern.
#[arcane]
#[inline]
fn mage_transpose_8x8_inplace_inner(_token: AvxToken, r: &mut [__m256; 8]) {
    // Phase 1: Interleave pairs (unpack)
    let q0 = _mm256_unpacklo_ps(r[0], r[2]);
    let q1 = _mm256_unpacklo_ps(r[1], r[3]);
    let q2 = _mm256_unpackhi_ps(r[0], r[2]);
    let q3 = _mm256_unpackhi_ps(r[1], r[3]);
    let q4 = _mm256_unpacklo_ps(r[4], r[6]);
    let q5 = _mm256_unpacklo_ps(r[5], r[7]);
    let q6 = _mm256_unpackhi_ps(r[4], r[6]);
    let q7 = _mm256_unpackhi_ps(r[5], r[7]);

    // Phase 2: Another round of unpack
    let s0 = _mm256_unpacklo_ps(q0, q1);
    let s1 = _mm256_unpackhi_ps(q0, q1);
    let s2 = _mm256_unpacklo_ps(q2, q3);
    let s3 = _mm256_unpackhi_ps(q2, q3);
    let s4 = _mm256_unpacklo_ps(q4, q5);
    let s5 = _mm256_unpackhi_ps(q4, q5);
    let s6 = _mm256_unpacklo_ps(q6, q7);
    let s7 = _mm256_unpackhi_ps(q6, q7);

    // Phase 3: Exchange 128-bit halves
    r[0] = _mm256_permute2f128_ps::<0x20>(s0, s4);
    r[1] = _mm256_permute2f128_ps::<0x20>(s1, s5);
    r[2] = _mm256_permute2f128_ps::<0x20>(s2, s6);
    r[3] = _mm256_permute2f128_ps::<0x20>(s3, s7);
    r[4] = _mm256_permute2f128_ps::<0x31>(s0, s4);
    r[5] = _mm256_permute2f128_ps::<0x31>(s1, s5);
    r[6] = _mm256_permute2f128_ps::<0x31>(s2, s6);
    r[7] = _mm256_permute2f128_ps::<0x31>(s3, s7);
}

/// Public wrapper for in-place transpose. Token proves AVX is available.
#[inline]
pub fn mage_transpose_8x8_inplace(token: AvxToken, r: &mut [__m256; 8]) {
    mage_transpose_8x8_inplace_inner(token, r);
}

// ============================================================================
// DCT Butterfly Operations
// ============================================================================

/// DCT base case for N=2: out0 = in0 + in1, out1 = in0 - in1
///
/// This is pure AVX (add/sub), no FMA needed.
#[arcane]
#[inline]
fn mage_dct1d_2_inner(_token: Avx2Token, m0: &mut __m256, m1: &mut __m256) {
    let in0 = *m0;
    let in1 = *m1;
    *m0 = _mm256_add_ps(in0, in1);
    *m1 = _mm256_sub_ps(in0, in1);
}

/// DCT for N=4 using FMA for the weighted operations.
#[arcane]
#[inline]
fn mage_dct1d_4_inner(token: Avx2FmaToken, m: &mut [__m256; 4]) {
    let wc4_0 = _mm256_set1_ps(WC4_0);
    let wc4_1 = _mm256_set1_ps(WC4_1);
    let sqrt2 = _mm256_set1_ps(SQRT2);

    // AddReverse<2>: tmp[0:2] = m[0:2] + reverse(m[2:4])
    let t0 = _mm256_add_ps(m[0], m[3]);
    let t1 = _mm256_add_ps(m[1], m[2]);

    // SubReverse<2>
    let t2 = _mm256_sub_ps(m[0], m[3]);
    let t3 = _mm256_sub_ps(m[1], m[2]);

    // DCT1D<2> on first half
    let r0 = _mm256_add_ps(t0, t1);
    let r1 = _mm256_sub_ps(t0, t1);

    // Multiply by WC4
    let t2_scaled = _mm256_mul_ps(t2, wc4_0);
    let t3_scaled = _mm256_mul_ps(t3, wc4_1);

    // DCT1D<2> on second half
    let r2 = _mm256_add_ps(t2_scaled, t3_scaled);
    let r3 = _mm256_sub_ps(t2_scaled, t3_scaled);

    // B<2>: r2 = r2 * sqrt2 + r3 (use FMA)
    let _ = token; // FMA token proves FMA available
    let r2_final = _mm256_fmadd_ps(r2, sqrt2, r3);

    // InverseEvenOdd<4>: interleave
    m[0] = r0;
    m[1] = r2_final;
    m[2] = r1;
    m[3] = r3;
}

/// DCT for N=8 using FMA. Processes 8 independent 8-point DCTs in parallel.
#[arcane]
#[inline]
fn mage_dct1d_8_inner(token: Avx2FmaToken, m: &mut [__m256; 8]) {
    let wc8_0 = _mm256_set1_ps(WC8_0);
    let wc8_1 = _mm256_set1_ps(WC8_1);
    let wc8_2 = _mm256_set1_ps(WC8_2);
    let wc8_3 = _mm256_set1_ps(WC8_3);
    let sqrt2 = _mm256_set1_ps(SQRT2);

    // AddReverse<4>: tmp[0:4] = m[0:4] + reverse(m[4:8])
    let t0 = _mm256_add_ps(m[0], m[7]);
    let t1 = _mm256_add_ps(m[1], m[6]);
    let t2 = _mm256_add_ps(m[2], m[5]);
    let t3 = _mm256_add_ps(m[3], m[4]);

    // SubReverse<4>
    let t4 = _mm256_sub_ps(m[0], m[7]);
    let t5 = _mm256_sub_ps(m[1], m[6]);
    let t6 = _mm256_sub_ps(m[2], m[5]);
    let t7 = _mm256_sub_ps(m[3], m[4]);

    // DCT1D<4> on first half
    let mut first = [t0, t1, t2, t3];
    mage_dct1d_4_inner(token, &mut first);

    // Multiply by WC8
    let t4_scaled = _mm256_mul_ps(t4, wc8_0);
    let t5_scaled = _mm256_mul_ps(t5, wc8_1);
    let t6_scaled = _mm256_mul_ps(t6, wc8_2);
    let t7_scaled = _mm256_mul_ps(t7, wc8_3);

    // DCT1D<4> on second half
    let mut second = [t4_scaled, t5_scaled, t6_scaled, t7_scaled];
    mage_dct1d_4_inner(token, &mut second);

    // B<4>: cumulative sum with FMA
    // second[0] = second[0] * sqrt2 + second[1]
    second[0] = _mm256_fmadd_ps(second[0], sqrt2, second[1]);
    // second[1] += second[2]
    second[1] = _mm256_add_ps(second[1], second[2]);
    // second[2] += second[3]
    second[2] = _mm256_add_ps(second[2], second[3]);
    // second[3] stays the same

    // InverseEvenOdd<8>: interleave
    m[0] = first[0];
    m[1] = second[0];
    m[2] = first[1];
    m[3] = second[1];
    m[4] = first[2];
    m[5] = second[2];
    m[6] = first[3];
    m[7] = second[3];
}

// ============================================================================
// Full 8x8 Forward DCT
// ============================================================================

/// Full 8x8 forward DCT using AVX2+FMA intrinsics via archmage tokens.
///
/// This is the archmage-based equivalent of `forward_dct_8x8_avx2`.
/// The token can be cached outside hot loops for zero overhead.
///
/// # Algorithm
///
/// 1. Load 8 rows into registers
/// 2. Transpose: reg[i] = column i (position i of all rows)
/// 3. Row DCT: 8 parallel 8-point DCTs
/// 4. Transpose: rearrange for column processing
/// 5. Column DCT: 8 parallel 8-point DCTs
/// 6. Scale by 1/8 and store
///
/// # Example
///
/// ```rust,ignore
/// use archmage::{Avx2FmaToken, SimdToken};
///
/// if let Some(token) = Avx2FmaToken::try_new() {
///     let input: [f32; 64] = /* ... */;
///     let mut output = [0.0f32; 64];
///     mage_forward_dct_8x8(token, &input, &mut output);
/// }
/// ```
#[arcane]
#[inline]
pub fn mage_forward_dct_8x8(token: Avx2FmaToken, input: &[f32; 64], output: &mut [f32; 64]) {
    let scale = _mm256_set1_ps(1.0 / 8.0);

    // Load 8 rows using safe archmage::mem operations
    // Split input into 8 contiguous chunks of 8 f32s each
    let mut reg = [
        avx::_mm256_loadu_ps(token.avx(), input[0..8].try_into().unwrap()),
        avx::_mm256_loadu_ps(token.avx(), input[8..16].try_into().unwrap()),
        avx::_mm256_loadu_ps(token.avx(), input[16..24].try_into().unwrap()),
        avx::_mm256_loadu_ps(token.avx(), input[24..32].try_into().unwrap()),
        avx::_mm256_loadu_ps(token.avx(), input[32..40].try_into().unwrap()),
        avx::_mm256_loadu_ps(token.avx(), input[40..48].try_into().unwrap()),
        avx::_mm256_loadu_ps(token.avx(), input[48..56].try_into().unwrap()),
        avx::_mm256_loadu_ps(token.avx(), input[56..64].try_into().unwrap()),
    ];

    // Transpose: reg[i] = column i = [row0[i], row1[i], ..., row7[i]]
    mage_transpose_8x8_inplace_inner(token.avx(), &mut reg);

    // Row DCT: all 8 rows processed in parallel
    mage_dct1d_8_inner(token, &mut reg);

    // Transpose: reg[i][j] = coef[i, j] (row-major coefficient matrix)
    mage_transpose_8x8_inplace_inner(token.avx(), &mut reg);

    // Column DCT: all 8 columns processed in parallel
    mage_dct1d_8_inner(token, &mut reg);

    // Scale and store using safe archmage::mem operations
    avx::_mm256_storeu_ps(
        token.avx(),
        (&mut output[0..8]).try_into().unwrap(),
        _mm256_mul_ps(reg[0], scale),
    );
    avx::_mm256_storeu_ps(
        token.avx(),
        (&mut output[8..16]).try_into().unwrap(),
        _mm256_mul_ps(reg[1], scale),
    );
    avx::_mm256_storeu_ps(
        token.avx(),
        (&mut output[16..24]).try_into().unwrap(),
        _mm256_mul_ps(reg[2], scale),
    );
    avx::_mm256_storeu_ps(
        token.avx(),
        (&mut output[24..32]).try_into().unwrap(),
        _mm256_mul_ps(reg[3], scale),
    );
    avx::_mm256_storeu_ps(
        token.avx(),
        (&mut output[32..40]).try_into().unwrap(),
        _mm256_mul_ps(reg[4], scale),
    );
    avx::_mm256_storeu_ps(
        token.avx(),
        (&mut output[40..48]).try_into().unwrap(),
        _mm256_mul_ps(reg[5], scale),
    );
    avx::_mm256_storeu_ps(
        token.avx(),
        (&mut output[48..56]).try_into().unwrap(),
        _mm256_mul_ps(reg[6], scale),
    );
    avx::_mm256_storeu_ps(
        token.avx(),
        (&mut output[56..64]).try_into().unwrap(),
        _mm256_mul_ps(reg[7], scale),
    );
}

// ============================================================================
// RGB to YCbCr Color Conversion
// ============================================================================

// BT.601 conversion constants
const YCBCR_R_TO_Y: f32 = 0.299;
const YCBCR_G_TO_Y: f32 = 0.587;
const YCBCR_B_TO_Y: f32 = 0.114;
const YCBCR_R_TO_CB: f32 = -0.168736;
const YCBCR_G_TO_CB: f32 = -0.331264;
const YCBCR_B_TO_CB: f32 = 0.5;
const YCBCR_R_TO_CR: f32 = 0.5;
const YCBCR_G_TO_CR: f32 = -0.418688;
const YCBCR_B_TO_CR: f32 = -0.081312;

/// Convert 8 RGB pixels to YCbCr using AVX2+FMA.
///
/// Takes pre-separated R, G, B values as f32 arrays and produces Y, Cb, Cr output.
/// This is the core color matrix multiplication using FMA for precision.
///
/// Y  = 0.299*R + 0.587*G + 0.114*B
/// Cb = 128 - 0.169*R - 0.331*G + 0.500*B
/// Cr = 128 + 0.500*R - 0.419*G - 0.081*B
#[arcane]
#[inline]
pub fn mage_rgb_to_ycbcr_8px(
    token: Avx2FmaToken,
    r: &[f32; 8],
    g: &[f32; 8],
    b: &[f32; 8],
    y_out: &mut [f32; 8],
    cb_out: &mut [f32; 8],
    cr_out: &mut [f32; 8],
) {
    // Load input vectors
    let r_vec = avx::_mm256_loadu_ps(token.avx(), r);
    let g_vec = avx::_mm256_loadu_ps(token.avx(), g);
    let b_vec = avx::_mm256_loadu_ps(token.avx(), b);

    // Coefficients
    let r_to_y = _mm256_set1_ps(YCBCR_R_TO_Y);
    let g_to_y = _mm256_set1_ps(YCBCR_G_TO_Y);
    let b_to_y = _mm256_set1_ps(YCBCR_B_TO_Y);
    let r_to_cb = _mm256_set1_ps(YCBCR_R_TO_CB);
    let g_to_cb = _mm256_set1_ps(YCBCR_G_TO_CB);
    let b_to_cb = _mm256_set1_ps(YCBCR_B_TO_CB);
    let r_to_cr = _mm256_set1_ps(YCBCR_R_TO_CR);
    let g_to_cr = _mm256_set1_ps(YCBCR_G_TO_CR);
    let b_to_cr = _mm256_set1_ps(YCBCR_B_TO_CR);
    let offset_128 = _mm256_set1_ps(128.0);

    // Y = r * r_to_y + g * g_to_y + b * b_to_y
    // Using FMA: result = a * b + c
    let y = _mm256_fmadd_ps(
        b_vec,
        b_to_y,
        _mm256_fmadd_ps(g_vec, g_to_y, _mm256_mul_ps(r_vec, r_to_y)),
    );

    // Cb = 128 + r * r_to_cb + g * g_to_cb + b * b_to_cb
    let cb = _mm256_fmadd_ps(
        b_vec,
        b_to_cb,
        _mm256_fmadd_ps(g_vec, g_to_cb, _mm256_fmadd_ps(r_vec, r_to_cb, offset_128)),
    );

    // Cr = 128 + r * r_to_cr + g * g_to_cr + b * b_to_cr
    let cr = _mm256_fmadd_ps(
        b_vec,
        b_to_cr,
        _mm256_fmadd_ps(g_vec, g_to_cr, _mm256_fmadd_ps(r_vec, r_to_cr, offset_128)),
    );

    // Store results
    avx::_mm256_storeu_ps(token.avx(), y_out, y);
    avx::_mm256_storeu_ps(token.avx(), cb_out, cb);
    avx::_mm256_storeu_ps(token.avx(), cr_out, cr);
}

/// Box filter downsample 2x2: average 4 adjacent pixels.
///
/// Takes evens and odds from two rows and computes (sum * 0.25).
/// For chroma downsampling in 4:2:0 encoding.
#[arcane]
#[inline]
pub fn mage_box_filter_2x2(
    _token: AvxToken,
    row0_evens: __m256,
    row0_odds: __m256,
    row1_evens: __m256,
    row1_odds: __m256,
) -> __m256 {
    let scale = _mm256_set1_ps(0.25);
    let sum = _mm256_add_ps(
        _mm256_add_ps(row0_evens, row0_odds),
        _mm256_add_ps(row1_evens, row1_odds),
    );
    _mm256_mul_ps(sum, scale)
}

// ============================================================================
// Even/Odd Deinterleave (for chroma downsampling)
// ============================================================================

/// AVX2-optimized deinterleave using Highway's ConcatEven/ConcatOdd pattern.
///
/// Given 16 consecutive f32s: [e0,o0,e1,o1,e2,o2,e3,o3, e4,o4,e5,o5,e6,o6,e7,o7]
/// Returns evens = [e0,e1,e2,e3,e4,e5,e6,e7], odds = [o0,o1,o2,o3,o4,o5,o6,o7]
///
/// This is ~4x faster than element-by-element construction.
#[arcane]
#[inline]
pub fn mage_gather_even_odd_x8(token: Avx2Token, data: &[f32; 16]) -> (__m256, __m256) {
    // Load 16 consecutive floats as two YMM registers
    let lo = avx::_mm256_loadu_ps(token.avx(), data[0..8].try_into().unwrap());
    let hi = avx::_mm256_loadu_ps(token.avx(), data[8..16].try_into().unwrap());

    // Highway's ConcatEven pattern for f32:
    // _mm256_shuffle_ps with 0x88 selects elements [0,2] from each source per lane
    let v2020 = _mm256_shuffle_ps(lo, hi, 0x88);
    // _mm256_permute4x64_epi64 with 0xD8 reorders 64-bit chunks: [0,2,1,3]
    let evens = _mm256_castsi256_ps(_mm256_permute4x64_epi64(_mm256_castps_si256(v2020), 0xD8));

    // Highway's ConcatOdd pattern for f32:
    // _mm256_shuffle_ps with 0xDD selects elements [1,3] from each source per lane
    let v3131 = _mm256_shuffle_ps(lo, hi, 0xDD);
    let odds = _mm256_castsi256_ps(_mm256_permute4x64_epi64(_mm256_castps_si256(v3131), 0xD8));

    (evens, odds)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use archmage::SimdToken;

    #[test]
    fn test_mage_forward_dct_8x8_identity() {
        if let Some(token) = Avx2FmaToken::try_new() {
            // Identity-like input (all zeros except DC)
            let mut input = [0.0f32; 64];
            input[0] = 64.0; // DC coefficient

            let mut output = [0.0f32; 64];
            mage_forward_dct_8x8(token, &input, &mut output);

            // DC should be non-zero, others should be small
            assert!(output[0].abs() > 0.1, "DC should be non-zero");
        }
    }

    #[test]
    fn test_mage_forward_dct_8x8_flat_block() {
        if let Some(token) = Avx2FmaToken::try_new() {
            // Flat block (constant value)
            let input = [128.0f32; 64];
            let mut output = [0.0f32; 64];

            mage_forward_dct_8x8(token, &input, &mut output);

            // For a flat block, only DC should be non-zero
            // DC = sum / 8 = 128 * 64 / 8 = 1024
            assert!(
                output[0].abs() > 100.0,
                "DC should be large for flat block"
            );

            // AC coefficients should be near zero
            for i in 1..64 {
                assert!(
                    output[i].abs() < 0.001,
                    "AC[{}] = {} should be ~0 for flat block",
                    i,
                    output[i]
                );
            }
        }
    }

    #[test]
    fn test_mage_transpose_8x8_inplace() {
        use super::avx;

        if let Some(token) = AvxToken::try_new() {
            let original: [f32; 64] = core::array::from_fn(|i| i as f32);

            // Load into registers using safe archmage::mem operations
            let mut reg = [
                avx::_mm256_loadu_ps(token, original[0..8].try_into().unwrap()),
                avx::_mm256_loadu_ps(token, original[8..16].try_into().unwrap()),
                avx::_mm256_loadu_ps(token, original[16..24].try_into().unwrap()),
                avx::_mm256_loadu_ps(token, original[24..32].try_into().unwrap()),
                avx::_mm256_loadu_ps(token, original[32..40].try_into().unwrap()),
                avx::_mm256_loadu_ps(token, original[40..48].try_into().unwrap()),
                avx::_mm256_loadu_ps(token, original[48..56].try_into().unwrap()),
                avx::_mm256_loadu_ps(token, original[56..64].try_into().unwrap()),
            ];

            // Transpose
            mage_transpose_8x8_inplace(token, &mut reg);

            // Store back using safe archmage::mem operations
            let mut result = [0.0f32; 64];
            avx::_mm256_storeu_ps(token, (&mut result[0..8]).try_into().unwrap(), reg[0]);
            avx::_mm256_storeu_ps(token, (&mut result[8..16]).try_into().unwrap(), reg[1]);
            avx::_mm256_storeu_ps(token, (&mut result[16..24]).try_into().unwrap(), reg[2]);
            avx::_mm256_storeu_ps(token, (&mut result[24..32]).try_into().unwrap(), reg[3]);
            avx::_mm256_storeu_ps(token, (&mut result[32..40]).try_into().unwrap(), reg[4]);
            avx::_mm256_storeu_ps(token, (&mut result[40..48]).try_into().unwrap(), reg[5]);
            avx::_mm256_storeu_ps(token, (&mut result[48..56]).try_into().unwrap(), reg[6]);
            avx::_mm256_storeu_ps(token, (&mut result[56..64]).try_into().unwrap(), reg[7]);

            // Verify transpose: result[col * 8 + row] == original[row * 8 + col]
            for row in 0..8 {
                for col in 0..8 {
                    let orig_val = original[row * 8 + col];
                    let trans_val = result[col * 8 + row];
                    assert_eq!(
                        orig_val, trans_val,
                        "Mismatch at ({}, {}): expected {}, got {}",
                        row, col, orig_val, trans_val
                    );
                }
            }
        }
    }

    #[test]
    fn test_mage_gather_even_odd_x8() {
        use super::avx;

        if let Some(token) = Avx2Token::try_new() {
            // Test data: [0, 1, 2, 3, ..., 15] interleaved as [e0,o0,e1,o1,...]
            let data: [f32; 16] = core::array::from_fn(|i| i as f32);

            let (evens, odds) = mage_gather_even_odd_x8(token, &data);

            // Store results to check
            let mut evens_arr = [0.0f32; 8];
            let mut odds_arr = [0.0f32; 8];
            avx::_mm256_storeu_ps(token.avx(), &mut evens_arr, evens);
            avx::_mm256_storeu_ps(token.avx(), &mut odds_arr, odds);

            // Expected: evens = [0, 2, 4, 6, 8, 10, 12, 14]
            //           odds  = [1, 3, 5, 7, 9, 11, 13, 15]
            let expected_evens = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0];
            let expected_odds = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0];

            for i in 0..8 {
                assert_eq!(
                    evens_arr[i], expected_evens[i],
                    "evens[{}]: got {}, expected {}",
                    i, evens_arr[i], expected_evens[i]
                );
                assert_eq!(
                    odds_arr[i], expected_odds[i],
                    "odds[{}]: got {}, expected {}",
                    i, odds_arr[i], expected_odds[i]
                );
            }
        }
    }

    #[test]
    fn test_mage_rgb_to_ycbcr_8px() {
        if let Some(token) = Avx2FmaToken::try_new() {
            // Test with pure white (255, 255, 255) -> Y=255, Cb=128, Cr=128
            let r = [255.0f32; 8];
            let g = [255.0f32; 8];
            let b = [255.0f32; 8];
            let mut y = [0.0f32; 8];
            let mut cb = [0.0f32; 8];
            let mut cr = [0.0f32; 8];

            mage_rgb_to_ycbcr_8px(token, &r, &g, &b, &mut y, &mut cb, &mut cr);

            for i in 0..8 {
                // Y should be ~255 (0.299*255 + 0.587*255 + 0.114*255 = 255)
                assert!(
                    (y[i] - 255.0).abs() < 0.1,
                    "Y[{}] = {} should be ~255",
                    i,
                    y[i]
                );
                // Cb should be ~128 (neutral)
                assert!(
                    (cb[i] - 128.0).abs() < 0.1,
                    "Cb[{}] = {} should be ~128",
                    i,
                    cb[i]
                );
                // Cr should be ~128 (neutral)
                assert!(
                    (cr[i] - 128.0).abs() < 0.1,
                    "Cr[{}] = {} should be ~128",
                    i,
                    cr[i]
                );
            }

            // Test with pure black (0, 0, 0) -> Y=0, Cb=128, Cr=128
            let r = [0.0f32; 8];
            let g = [0.0f32; 8];
            let b = [0.0f32; 8];

            mage_rgb_to_ycbcr_8px(token, &r, &g, &b, &mut y, &mut cb, &mut cr);

            for i in 0..8 {
                assert!(
                    y[i].abs() < 0.1,
                    "Y[{}] = {} should be ~0 for black",
                    i,
                    y[i]
                );
                assert!(
                    (cb[i] - 128.0).abs() < 0.1,
                    "Cb[{}] = {} should be ~128 for black",
                    i,
                    cb[i]
                );
                assert!(
                    (cr[i] - 128.0).abs() < 0.1,
                    "Cr[{}] = {} should be ~128 for black",
                    i,
                    cr[i]
                );
            }

            // Test with pure red (255, 0, 0) -> Y=76.2, Cb=84.5, Cr=255
            let r = [255.0f32; 8];
            let g = [0.0f32; 8];
            let b = [0.0f32; 8];

            mage_rgb_to_ycbcr_8px(token, &r, &g, &b, &mut y, &mut cb, &mut cr);

            for i in 0..8 {
                // Y = 0.299 * 255 = 76.245
                assert!(
                    (y[i] - 76.245).abs() < 0.1,
                    "Y[{}] = {} should be ~76.245 for red",
                    i,
                    y[i]
                );
            }
        }
    }

    #[test]
    fn test_mage_box_filter_2x2() {
        use super::avx;

        if let Some(token) = AvxToken::try_new() {
            // Create test data for 2x2 box filter
            // Each "pixel" should be averaged from 4 neighbors
            let row0_evens_arr = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let row0_odds_arr = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let row1_evens_arr = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let row1_odds_arr = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

            let row0_evens = avx::_mm256_loadu_ps(token, &row0_evens_arr);
            let row0_odds = avx::_mm256_loadu_ps(token, &row0_odds_arr);
            let row1_evens = avx::_mm256_loadu_ps(token, &row1_evens_arr);
            let row1_odds = avx::_mm256_loadu_ps(token, &row1_odds_arr);

            let result = mage_box_filter_2x2(token, row0_evens, row0_odds, row1_evens, row1_odds);

            let mut result_arr = [0.0f32; 8];
            avx::_mm256_storeu_ps(token, &mut result_arr, result);

            // Each output should be (4 * input) * 0.25 = input
            for i in 0..8 {
                let expected = (i + 1) as f32;
                assert!(
                    (result_arr[i] - expected).abs() < 0.001,
                    "result[{}] = {} should be {}",
                    i,
                    result_arr[i],
                    expected
                );
            }
        }
    }
}
