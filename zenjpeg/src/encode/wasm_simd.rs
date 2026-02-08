//! WASM SIMD128 implementations using archmage capability tokens.
//!
//! These functions provide SIMD128-optimized implementations for WebAssembly.
//! The `wasm_` prefix distinguishes them from x86 AVX2 and ARM NEON versions.
//!
//! # Token Model
//!
//! Each function requires a Wasm128Token that proves SIMD128 is available.
//! When compiled with `RUSTFLAGS="-C target-feature=+simd128"`, token summoning
//! is elided at compile time.
//!
//! # Vector Width
//!
//! WASM SIMD128 is 128-bit (4-wide for f32), same as NEON.
//! 8x8 DCT operations are implemented as two 4x4 blocks.

#![cfg(all(feature = "archmage-simd", target_arch = "wasm32"))]

use archmage::{arcane, Wasm128Token};
use core::arch::wasm32::*;

// Re-export Wasm128Token for callers
pub use archmage::Wasm128Token;

// ============================================================================
// DCT Constants (same as x86/ARM versions)
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

/// In-place 4x4 transpose on 4 v128 registers using WASM SIMD128.
///
/// Uses i32x4_shuffle for interleaving. WASM doesn't have typed float shuffle,
/// so we reinterpret as i32, shuffle, and reinterpret back.
///
/// This is the building block for 8x8 transposes.
#[arcane]
#[inline]
fn wasm_transpose_4x4_inplace_inner(_token: Wasm128Token, r: &mut [v128; 4]) {
    unsafe {
        // Phase 1: Interleave pairs (low and high 64-bit halves)
        // i32x4_shuffle takes two v128 and builds result from lanes selected by indices
        let q0 = i32x4_shuffle::<0, 4, 1, 5>(r[0], r[1]); // [r0[0], r1[0], r0[1], r1[1]]
        let q1 = i32x4_shuffle::<2, 6, 3, 7>(r[0], r[1]); // [r0[2], r1[2], r0[3], r1[3]]
        let q2 = i32x4_shuffle::<0, 4, 1, 5>(r[2], r[3]); // [r2[0], r3[0], r2[1], r3[1]]
        let q3 = i32x4_shuffle::<2, 6, 3, 7>(r[2], r[3]); // [r2[2], r3[2], r2[3], r3[3]]

        // Phase 2: Interleave 64-bit pairs
        r[0] = i64x2_shuffle::<0, 2>(q0, q2); // Column 0
        r[1] = i64x2_shuffle::<1, 3>(q0, q2); // Column 1
        r[2] = i64x2_shuffle::<0, 2>(q1, q3); // Column 2
        r[3] = i64x2_shuffle::<1, 3>(q1, q3); // Column 3
    }
}

/// Public wrapper for 4x4 transpose.
#[inline]
pub fn wasm_transpose_4x4_inplace(token: Wasm128Token, r: &mut [v128; 4]) {
    wasm_transpose_4x4_inplace_inner(token, r);
}

// ============================================================================
// 8x8 Transpose (Two 4x4 Blocks)
// ============================================================================

/// In-place 8x8 transpose implemented as two independent 4x4 transposes.
///
/// Input: 8 rows of 8 f32 values (64 total)
/// Output: 8 columns (transposed)
///
/// Since WASM SIMD128 is 4-wide, we split the 8x8 into four 4x4 blocks
/// and transpose each independently.
#[arcane]
#[inline]
fn wasm_transpose_8x8_inplace_inner(token: Wasm128Token, data: &mut [f32; 64]) {
    unsafe {
        // Load 8 rows as 16 v128 registers (2 per row)
        let mut r0_lo = v128_load(data.as_ptr() as *const v128);
        let mut r0_hi = v128_load(data.as_ptr().add(4) as *const v128);
        let mut r1_lo = v128_load(data.as_ptr().add(8) as *const v128);
        let mut r1_hi = v128_load(data.as_ptr().add(12) as *const v128);
        let mut r2_lo = v128_load(data.as_ptr().add(16) as *const v128);
        let mut r2_hi = v128_load(data.as_ptr().add(20) as *const v128);
        let mut r3_lo = v128_load(data.as_ptr().add(24) as *const v128);
        let mut r3_hi = v128_load(data.as_ptr().add(28) as *const v128);
        let mut r4_lo = v128_load(data.as_ptr().add(32) as *const v128);
        let mut r4_hi = v128_load(data.as_ptr().add(36) as *const v128);
        let mut r5_lo = v128_load(data.as_ptr().add(40) as *const v128);
        let mut r5_hi = v128_load(data.as_ptr().add(44) as *const v128);
        let mut r6_lo = v128_load(data.as_ptr().add(48) as *const v128);
        let mut r6_hi = v128_load(data.as_ptr().add(52) as *const v128);
        let mut r7_lo = v128_load(data.as_ptr().add(56) as *const v128);
        let mut r7_hi = v128_load(data.as_ptr().add(60) as *const v128);

        // Transpose top-left 4x4
        let mut tl = [r0_lo, r1_lo, r2_lo, r3_lo];
        wasm_transpose_4x4_inplace_inner(token, &mut tl);

        // Transpose top-right 4x4
        let mut tr = [r0_hi, r1_hi, r2_hi, r3_hi];
        wasm_transpose_4x4_inplace_inner(token, &mut tr);

        // Transpose bottom-left 4x4
        let mut bl = [r4_lo, r5_lo, r6_lo, r7_lo];
        wasm_transpose_4x4_inplace_inner(token, &mut bl);

        // Transpose bottom-right 4x4
        let mut br = [r4_hi, r5_hi, r6_hi, r7_hi];
        wasm_transpose_4x4_inplace_inner(token, &mut br);

        // Store transposed blocks
        v128_store(data.as_mut_ptr() as *mut v128, tl[0]);
        v128_store(data.as_mut_ptr().add(4) as *mut v128, bl[0]);
        v128_store(data.as_mut_ptr().add(8) as *mut v128, tl[1]);
        v128_store(data.as_mut_ptr().add(12) as *mut v128, bl[1]);
        v128_store(data.as_mut_ptr().add(16) as *mut v128, tl[2]);
        v128_store(data.as_mut_ptr().add(20) as *mut v128, bl[2]);
        v128_store(data.as_mut_ptr().add(24) as *mut v128, tl[3]);
        v128_store(data.as_mut_ptr().add(28) as *mut v128, bl[3]);
        v128_store(data.as_mut_ptr().add(32) as *mut v128, tr[0]);
        v128_store(data.as_mut_ptr().add(36) as *mut v128, br[0]);
        v128_store(data.as_mut_ptr().add(40) as *mut v128, tr[1]);
        v128_store(data.as_mut_ptr().add(44) as *mut v128, br[1]);
        v128_store(data.as_mut_ptr().add(48) as *mut v128, tr[2]);
        v128_store(data.as_mut_ptr().add(52) as *mut v128, br[2]);
        v128_store(data.as_mut_ptr().add(56) as *mut v128, tr[3]);
        v128_store(data.as_mut_ptr().add(60) as *mut v128, br[3]);
    }
}

/// Public wrapper for 8x8 transpose.
#[inline]
pub fn wasm_transpose_8x8(token: Wasm128Token, data: &mut [f32; 64]) {
    wasm_transpose_8x8_inplace_inner(token, data);
}

// ============================================================================
// DCT Butterfly Operations (4-wide)
// ============================================================================

/// 2-point DCT butterfly using WASM SIMD128.
///
/// Implements: (m0 + m1, m0 - m1)
/// Note: WASM doesn't have FMA, so we use separate add/sub operations.
#[arcane]
#[inline]
fn wasm_dct1d_2_inner(_token: Wasm128Token, m0: &mut v128, m1: &mut v128) {
    unsafe {
        let sum = f32x4_add(*m0, *m1);
        let diff = f32x4_sub(*m0, *m1);
        *m0 = sum;
        *m1 = diff;
    }
}

/// 4-point DCT butterfly using WASM SIMD128.
///
/// Implements the standard 4-point DCT transform.
/// No FMA available, so uses separate mul/add operations.
#[arcane]
#[inline]
fn wasm_dct1d_4_inner(token: Wasm128Token, m: &mut [v128; 4]) {
    unsafe {
        // First layer: (m0+m3, m1+m2, m1-m2, m0-m3)
        let sum03 = f32x4_add(m[0], m[3]);
        let sum12 = f32x4_add(m[1], m[2]);
        let diff12 = f32x4_sub(m[1], m[2]);
        let diff03 = f32x4_sub(m[0], m[3]);

        // Second layer: apply 2-point DCT to (sum03, sum12)
        let mut t0 = sum03;
        let mut t1 = sum12;
        wasm_dct1d_2_inner(token, &mut t0, &mut t1);

        // Apply WC4 coefficients to differences (no FMA, use mul + add)
        let wc4_0 = f32x4_splat(WC4_0);
        let wc4_1 = f32x4_splat(WC4_1);

        m[0] = t0;
        m[1] = f32x4_add(f32x4_mul(diff12, wc4_0), f32x4_mul(diff03, wc4_1));
        m[2] = t1;
        m[3] = f32x4_sub(f32x4_mul(diff12, wc4_1), f32x4_mul(diff03, wc4_0));
    }
}

/// 8-point DCT butterfly using WASM SIMD128 (processes 4 blocks in parallel).
///
/// This is the core 1D DCT-II transform applied to 4 parallel streams.
/// No FMA, so uses separate mul/add operations (2x the ops vs ARM NEON).
#[arcane]
#[inline]
fn wasm_dct1d_8_inner(token: Wasm128Token, m: &mut [v128; 8]) {
    unsafe {
        // First layer: butterfly on opposite ends
        let sum07 = f32x4_add(m[0], m[7]);
        let sum16 = f32x4_add(m[1], m[6]);
        let sum25 = f32x4_add(m[2], m[5]);
        let sum34 = f32x4_add(m[3], m[4]);
        let diff07 = f32x4_sub(m[0], m[7]);
        let diff16 = f32x4_sub(m[1], m[6]);
        let diff25 = f32x4_sub(m[2], m[5]);
        let diff34 = f32x4_sub(m[3], m[4]);

        // Apply 4-point DCT to sums
        let mut even = [sum07, sum16, sum25, sum34];
        wasm_dct1d_4_inner(token, &mut even);

        // Apply WC8 coefficients to differences (no FMA - separate mul/add)
        let wc8_0 = f32x4_splat(WC8_0);
        let wc8_1 = f32x4_splat(WC8_1);
        let wc8_2 = f32x4_splat(WC8_2);
        let wc8_3 = f32x4_splat(WC8_3);

        // Odd part (no FMA - each mul_add becomes mul + add)
        let t0 = f32x4_add(f32x4_mul(diff07, wc8_0), f32x4_mul(diff34, wc8_1));
        let t1 = f32x4_add(f32x4_mul(diff16, wc8_2), f32x4_mul(diff25, wc8_3));
        let t2 = f32x4_sub(f32x4_mul(diff16, wc8_3), f32x4_mul(diff25, wc8_2));
        let t3 = f32x4_sub(f32x4_mul(diff07, wc8_1), f32x4_mul(diff34, wc8_0));

        let odd0 = f32x4_add(t0, t1);
        let odd1 = f32x4_sub(t0, t1);
        let odd2 = f32x4_add(t2, t3);
        let odd3 = f32x4_sub(t2, t3);

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
}

// ============================================================================
// Forward DCT 8x8
// ============================================================================

/// Forward DCT 8x8 using WASM SIMD128.
///
/// Processes the DCT as two 4x4 blocks due to SIMD128's 4-wide registers.
#[arcane]
pub fn wasm_forward_dct_8x8(token: Wasm128Token, input: &[f32; 64], output: &mut [f32; 64]) {
    unsafe {
        // Load all 8 rows as v128 pairs
        let zero = f32x4_splat(0.0);
        let mut rows_lo: [v128; 8] = [zero; 8];
        let mut rows_hi: [v128; 8] = [zero; 8];

        for i in 0..8 {
            rows_lo[i] = v128_load(input.as_ptr().add(i * 8) as *const v128);
            rows_hi[i] = v128_load(input.as_ptr().add(i * 8 + 4) as *const v128);
        }

        // Apply 1D DCT to each row
        wasm_dct1d_8_inner(token, &mut rows_lo);
        wasm_dct1d_8_inner(token, &mut rows_hi);

        // Transpose
        let mut temp = [0.0f32; 64];
        for i in 0..8 {
            v128_store(temp.as_mut_ptr().add(i * 8) as *mut v128, rows_lo[i]);
            v128_store(temp.as_mut_ptr().add(i * 8 + 4) as *mut v128, rows_hi[i]);
        }
        wasm_transpose_8x8_inplace_inner(token, &mut temp);

        // Reload transposed data
        for i in 0..8 {
            rows_lo[i] = v128_load(temp.as_ptr().add(i * 8) as *const v128);
            rows_hi[i] = v128_load(temp.as_ptr().add(i * 8 + 4) as *const v128);
        }

        // Apply 1D DCT to each column (now rows after transpose)
        wasm_dct1d_8_inner(token, &mut rows_lo);
        wasm_dct1d_8_inner(token, &mut rows_hi);

        // Store result
        for i in 0..8 {
            v128_store(output.as_mut_ptr().add(i * 8) as *mut v128, rows_lo[i]);
            v128_store(output.as_mut_ptr().add(i * 8 + 4) as *mut v128, rows_hi[i]);
        }
    }
}

// Tests would go here (similar to ARM version)

// ============================================================================
// Integer IDCT (for decoder)
// ============================================================================

/// Integer IDCT 8x8 using WASM SIMD128.
///
/// Implements the libjpeg-turbo Loeffler algorithm with WASM intrinsics.
/// Processes 4-wide columns using v128 (i32x4).
#[arcane]
pub fn wasm_idct_int_8x8(
    token: Wasm128Token,
    input: &[i32; 64],
    output: &mut [i16],
    stride: usize,
) {
    unsafe {
        // DC-only fast path
        let mut all_ac_zero = true;
        for i in 1..64 {
            if input[i] != 0 {
                all_ac_zero = false;
                break;
            }
        }
        
        if all_ac_zero {
            let dc = ((input[0] + 4 + 1024) >> 3).clamp(0, 255) as i16;
            let dc_vec = i16x8_splat(dc);
            let mut pos = 0;
            for _ in 0..8 {
                v128_store(output[pos..].as_mut_ptr() as *mut v128, dc_vec);
                pos += stride;
            }
            return;
        }

        // Full IDCT - fallback to scalar for now
        // TODO: Implement full WASM column pass
        super::super::decode::idct_int::idct_int_libjpeg(
            &mut input.clone(),
            output,
            stride,
        );
    }
}

// ============================================================================
// YCbCr Color Conversion (for decoder)
// ============================================================================

/// Convert YCbCr to RGB using WASM SIMD128.
///
/// Note: WASM lacks FMA and multiply-accumulate, so uses separate mul+add.
#[arcane]
pub fn wasm_ycbcr_to_rgb(
    token: Wasm128Token,
    y: &[i16; 16],
    cb: &[i16; 16],
    cr: &[i16; 16],
    rgb: &mut [u8; 48],
) {
    unsafe {
        // Constants (14-bit fixed-point)
        let y_coeff = i16x8_splat(19595);
        let cr_to_r = i16x8_splat(22970);
        let cb_to_b = i16x8_splat(29032);
        let cr_to_g = i16x8_splat(-11698);
        let cb_to_g = i16x8_splat(-5636);
        
        let cb_cr_bias = i16x8_splat(128);

        // Process first 8 pixels
        let y0 = v128_load(y.as_ptr() as *const v128);
        let mut cb0 = v128_load(cb.as_ptr() as *const v128);
        let mut cr0 = v128_load(cr.as_ptr() as *const v128);

        // Unbias Cb/Cr
        cb0 = i16x8_sub(cb0, cb_cr_bias);
        cr0 = i16x8_sub(cr0, cb_cr_bias);

        // Widen to i32 for multiplication
        let y_lo = i32x4_extend_low_i16x8(y0);
        let y_hi = i32x4_extend_high_i16x8(y0);
        let cr_lo = i32x4_extend_low_i16x8(cr0);
        let cr_hi = i32x4_extend_high_i16x8(cr0);
        let cb_lo = i32x4_extend_low_i16x8(cb0);
        let cb_hi = i32x4_extend_high_i16x8(cb0);

        // Compute R = Y + Cr * coeff (no FMA, so mul + add)
        // TODO: Full implementation with proper shifting and saturation
        // For now, store planar placeholder
        let r_vec = u8x16_splat(128);
        let g_vec = u8x16_splat(128);
        let b_vec = u8x16_splat(128);

        v128_store(rgb.as_mut_ptr() as *mut v128, r_vec);
        v128_store(rgb[16..].as_mut_ptr() as *mut v128, g_vec);
        v128_store(rgb[32..].as_mut_ptr() as *mut v128, b_vec);
    }
}

// ============================================================================
// Chroma Upsampling (for decoder)
// ============================================================================

/// H2V1 upsampling (horizontal 2x) using triangle filter.
#[arcane]
pub fn wasm_upsample_h2v1(
    token: Wasm128Token,
    input: &[f32],
    in_width: usize,
    output: &mut [f32],
    out_width: usize,
) {
    unsafe {
        assert_eq!(out_width, in_width * 2);
        
        let v_three = f32x4_splat(3.0);
        let v_quarter = f32x4_splat(0.25);

        // First pixel
        output[0] = input[0];
        
        // Process 4 input pixels at a time → 8 output pixels
        let chunks = in_width / 4;
        for i in 0..chunks {
            let curr = v128_load(input[i * 4..].as_ptr() as *const v128);
            let next = v128_load(input[i * 4 + 1..].as_ptr() as *const v128);

            // even = curr, odd = (3*curr + next) * 0.25
            let odd = f32x4_mul(f32x4_add(f32x4_mul(curr, v_three), next), v_quarter);

            // Interleave even and odd using shuffle
            let out0 = i32x4_shuffle::<0, 4, 1, 5>(curr, odd);
            let out1 = i32x4_shuffle::<2, 6, 3, 7>(curr, odd);

            v128_store(output[i * 8..].as_mut_ptr() as *mut v128, out0);
            v128_store(output[i * 8 + 4..].as_mut_ptr() as *mut v128, out1);
        }

        // Last pixel
        output[out_width - 1] = input[in_width - 1];
    }
}

