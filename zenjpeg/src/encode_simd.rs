//! SIMD-optimized encoding functions.
//!
//! Contains SIMD implementations for encoder hot paths:
//! - Chroma downsampling (2x2, 2x1, 1x2)
//! - RGB to YCbCr color conversion
//!
//! # Safe SIMD Architecture
//!
//! This module follows a two-layer pattern:
//!
//! 1. **Safe public APIs** (e.g., `downsample_2x2_simd_inplace`, `rgb_to_ycbcr_planes_simd_inplace`):
//!    - Use magetypes generic SIMD types (`GenericF32x8<Token>`) via `#[magetypes]`
//!    - Safe load/store helpers with bounds checking
//!    - Multi-tier runtime dispatch via `incant!` (AVX2+FMA, NEON, WASM128, scalar)
//!
//! 2. **Safe internal functions via archmage** (e.g., `gather_even_odd_x8_avx2`, `rgb_to_ycbcr_8px_fma`):
//!    - Raw SSSE3/SSE4.1/AVX/AVX2/FMA intrinsics made safe via `#[arcane]` attribute
//!    - Capability tokens (`X64V3Token`) prove CPU support at the call site
//!    - Load/store operations use safe_unaligned_simd wrappers (no unsafe needed)
//!
//! All functions have tests to verify parity with scalar implementations.

use archmage::prelude::*;
use magetypes::simd::backends::F32x8Backend;
use magetypes::simd::generic::f32x8 as GenericF32x8;

// AVX2/SSE intrinsics - safe via archmage #[arcane] annotation
#[cfg(target_arch = "x86_64")]
use archmage::{SimdToken, arcane, rite};
#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
use core::arch::x86_64::{
    __m128, __m128i, __m256, _mm_cvtepu8_epi32, _mm_fmadd_ps, _mm_loadu_si128, _mm_mul_ps,
    _mm_set1_ps, _mm_setr_epi8, _mm_shuffle_epi8, _mm_storeu_ps,
};
#[cfg(target_arch = "x86_64")]
use safe_unaligned_simd::x86_64 as safe_simd;

use crate::foundation::consts::{
    YCBCR_B_TO_CB, YCBCR_B_TO_CR, YCBCR_B_TO_Y, YCBCR_G_TO_CB, YCBCR_G_TO_CR, YCBCR_G_TO_Y,
    YCBCR_R_TO_CB, YCBCR_R_TO_CR, YCBCR_R_TO_Y,
};
// ============================================================================
// Safe SIMD Load/Store Helpers
// ============================================================================

/// Load 8 f32s into GenericF32x8. Panics if slice is too short.
#[inline(always)]
fn load_f32x8<T: F32x8Backend>(token: T, slice: &[f32], offset: usize) -> GenericF32x8<T> {
    GenericF32x8::<T>::load(token, slice[offset..offset + 8].try_into().unwrap())
}

/// Store GenericF32x8 to slice. Panics if slice is too short.
#[inline(always)]
fn store_f32x8<T: F32x8Backend>(slice: &mut [f32], offset: usize, value: GenericF32x8<T>) {
    value.store(<&mut [f32; 8]>::try_from(&mut slice[offset..offset + 8]).unwrap())
}

// ============================================================================
// Chroma Downsampling (2x2 box filter)
// ============================================================================

/// SIMD-optimized 2x2 box filter downsampling, writing to pre-allocated buffer.
///
/// This is the **zero-allocation** version for hot paths.
///
/// # Arguments
/// * `plane` - Input plane (f32, full resolution)
/// * `width` - Input width
/// * `height` - Input height
/// * `result` - Output buffer (must be at least `((width+1)/2) * ((height+1)/2)` elements)
pub fn downsample_2x2_simd_inplace(plane: &[f32], width: usize, height: usize, result: &mut [f32]) {
    incant!(downsample_2x2_simd_inplace_impl(
        plane, width, height, result
    ));
}

#[magetypes(v3, neon, wasm128, scalar)]
fn downsample_2x2_simd_inplace_impl(
    token: Token,
    plane: &[f32],
    width: usize,
    height: usize,
    result: &mut [f32],
) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;

    let new_width = (width + 1) / 2;
    let new_height = (height + 1) / 2;
    debug_assert!(result.len() >= new_width * new_height);

    let scale = f32x8::splat(token, 0.25);
    // SIMD path needs 16 input elements per chunk. For odd widths, the last chunk
    // would read past the row boundary into the next row, so we use scalar path
    // for any columns where input x + 15 >= width (i.e., last 8 output columns when width % 16 >= 1).
    // Safe chunks: those where in_x + 16 <= width, i.e., out_x + 8 <= (width / 2)
    let safe_chunks = if width >= 16 { (width - 15) / 16 } else { 0 };

    for y in 0..new_height {
        let y0 = y * 2;
        let y1 = (y0 + 1).min(height - 1);
        let out_row_start = y * new_width;

        // SIMD path: process 8 output pixels at a time (only for chunks that don't cross row boundary)
        for chunk in 0..safe_chunks {
            let out_x = chunk * 8;
            let in_x = out_x * 2;

            let row0_idx = y0 * width + in_x;
            let row1_idx = y1 * width + in_x;

            // Gather even/odd from row 0 and row 1
            let (p00_arr, p10_arr) = gather_even_odd_x8(plane, row0_idx, width);
            let (p01_arr, p11_arr) = gather_even_odd_x8(plane, row1_idx, width);

            let p00 = f32x8::from_array(token, p00_arr);
            let p10 = f32x8::from_array(token, p10_arr);
            let p01 = f32x8::from_array(token, p01_arr);
            let p11 = f32x8::from_array(token, p11_arr);

            // Box filter: (p00 + p10 + p01 + p11) * 0.25
            let sum = p00 + p10 + p01 + p11;
            let avg = sum * scale;

            // Store result
            store_f32x8(&mut *result, out_row_start + out_x, avg);
        }

        // Scalar path for remaining columns (handles row boundary correctly)
        for out_x in (safe_chunks * 8)..new_width {
            let x0 = out_x * 2;
            let x1 = (x0 + 1).min(width - 1);

            let p00 = plane[y0 * width + x0];
            let p10 = plane[y0 * width + x1];
            let p01 = plane[y1 * width + x0];
            let p11 = plane[y1 * width + x1];

            result[out_row_start + out_x] = (p00 + p10 + p01 + p11) * 0.25;
        }
    }
}

/// SIMD-optimized 2x1 (horizontal) box filter downsampling (in-place).
///
/// Writes to pre-allocated result buffer.
pub fn downsample_2x1_simd_inplace(plane: &[f32], width: usize, height: usize, result: &mut [f32]) {
    incant!(downsample_2x1_simd_inplace_impl(
        plane, width, height, result
    ));
}

#[magetypes(v3, neon, wasm128, scalar)]
fn downsample_2x1_simd_inplace_impl(
    token: Token,
    plane: &[f32],
    width: usize,
    height: usize,
    result: &mut [f32],
) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;

    let new_width = (width + 1) / 2;
    debug_assert!(result.len() >= new_width * height);

    let scale = f32x8::splat(token, 0.5);
    // SIMD path needs 16 input elements per chunk. For odd widths, the last chunk
    // would read past the row boundary, so use scalar path for edge columns.
    let safe_chunks = if width >= 16 { (width - 15) / 16 } else { 0 };

    for y in 0..height {
        let out_row_start = y * new_width;
        let in_row_start = y * width;

        // SIMD path: process 8 output pixels at a time (only for safe chunks)
        for chunk in 0..safe_chunks {
            let out_x = chunk * 8;
            let in_x = out_x * 2;

            // Gather even/odd pixels from the row
            let (p0_arr, p1_arr) = gather_even_odd_x8(plane, in_row_start + in_x, width);
            let p0 = f32x8::from_array(token, p0_arr);
            let p1 = f32x8::from_array(token, p1_arr);

            // Box filter: (p0 + p1) * 0.5
            let avg = (p0 + p1) * scale;

            // Store result
            store_f32x8(&mut *result, out_row_start + out_x, avg);
        }

        // Scalar path for remaining columns (handles edge correctly)
        for out_x in (safe_chunks * 8)..new_width {
            let x0 = out_x * 2;
            let x1 = (x0 + 1).min(width - 1);

            let p0 = plane[in_row_start + x0];
            let p1 = plane[in_row_start + x1];

            result[out_row_start + out_x] = (p0 + p1) * 0.5;
        }
    }
}

/// SIMD-optimized 1x2 (vertical) box filter downsampling (in-place).
///
/// Writes to pre-allocated result buffer.
pub fn downsample_1x2_simd_inplace(plane: &[f32], width: usize, height: usize, result: &mut [f32]) {
    incant!(downsample_1x2_simd_inplace_impl(
        plane, width, height, result
    ));
}

#[magetypes(v3, neon, wasm128, scalar)]
fn downsample_1x2_simd_inplace_impl(
    token: Token,
    plane: &[f32],
    width: usize,
    height: usize,
    result: &mut [f32],
) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;

    let new_height = (height + 1) / 2;
    debug_assert!(result.len() >= width * new_height);

    let scale = f32x8::splat(token, 0.5);
    let chunks = width / 8;

    for y in 0..new_height {
        let y0 = y * 2;
        let y1 = (y0 + 1).min(height - 1);
        let out_row_start = y * width;

        // SIMD path: process 8 pixels at a time
        for chunk in 0..chunks {
            let x = chunk * 8;

            // Load 8 consecutive pixels from row y0 and y1
            let row0_idx = y0 * width + x;
            let row1_idx = y1 * width + x;

            let p0 = load_f32x8(token, plane, row0_idx);
            let p1 = load_f32x8(token, plane, row1_idx);

            let avg = (p0 + p1) * scale;

            // Store result
            store_f32x8(&mut *result, out_row_start + x, avg);
        }

        // Scalar remainder
        for x in (chunks * 8)..width {
            let p0 = plane[y0 * width + x];
            let p1 = plane[y1 * width + x];
            result[out_row_start + x] = (p0 + p1) * 0.5;
        }
    }
}

// ============================================================================
// AVX2 Intrinsics Implementations
// ============================================================================

/// Scalar reference implementation of gather_even_odd for testing.
///
/// This is the ground truth implementation that AVX2 versions are tested against.
#[cfg(test)]
#[allow(dead_code)] // Reference implementation for AVX2 validation
#[inline]
fn gather_even_odd_scalar(data: &[f32]) -> ([f32; 8], [f32; 8]) {
    debug_assert!(data.len() >= 16);
    let evens = [
        data[0], data[2], data[4], data[6], data[8], data[10], data[12], data[14],
    ];
    let odds = [
        data[1], data[3], data[5], data[7], data[9], data[11], data[13], data[15],
    ];
    (evens, odds)
}

/// AVX2-optimized deinterleave using Highway's ConcatEven/ConcatOdd pattern.
/// This is ~4x faster than element-by-element construction.
#[cfg(target_arch = "x86_64")]
#[arcane]
#[inline]
fn gather_even_odd_x8_avx2(_token: archmage::X64V3Token, data: &[f32; 16]) -> ([f32; 8], [f32; 8]) {
    use std::arch::x86_64::*;

    // Load 16 consecutive floats as two YMM registers
    // Memory: [e0,o0,e1,o1,e2,o2,e3,o3, e4,o4,e5,o5,e6,o6,e7,o7]
    let lo = safe_simd::_mm256_loadu_ps(<&[f32; 8]>::try_from(&data[..8]).unwrap());
    let hi = safe_simd::_mm256_loadu_ps(<&[f32; 8]>::try_from(&data[8..]).unwrap());

    // Highway's ConcatEven pattern for f32:
    // _mm256_shuffle_ps with 0x88 selects elements [0,2] from each source per lane
    // Lane0: [lo[0],lo[2],hi[0],hi[2]] = [e0,e1,e4,e5]
    // Lane1: [lo[4],lo[6],hi[4],hi[6]] = [e2,e3,e6,e7]
    let v2020 = _mm256_shuffle_ps(lo, hi, 0x88);
    // _mm256_permute4x64_epi64 with 0xD8 reorders 64-bit chunks: [0,2,1,3]
    // Final: [e0,e1,e2,e3,e4,e5,e6,e7]
    let evens_raw = _mm256_castsi256_ps(_mm256_permute4x64_epi64(_mm256_castps_si256(v2020), 0xD8));

    // Highway's ConcatOdd pattern for f32:
    // _mm256_shuffle_ps with 0xDD selects elements [1,3] from each source per lane
    let v3131 = _mm256_shuffle_ps(lo, hi, 0xDD);
    let odds_raw = _mm256_castsi256_ps(_mm256_permute4x64_epi64(_mm256_castps_si256(v3131), 0xD8));

    (
        bytemuck::cast::<__m256, [f32; 8]>(evens_raw),
        bytemuck::cast::<__m256, [f32; 8]>(odds_raw),
    )
}

/// Scalar fallback for gather_even_odd - used by non-AVX2 targets
#[inline(always)]
fn gather_even_odd_x8_scalar(slice: &[f32]) -> ([f32; 8], [f32; 8]) {
    // Caller guarantees at least 16 elements are available
    let evens = [
        slice[0], slice[2], slice[4], slice[6], slice[8], slice[10], slice[12], slice[14],
    ];
    let odds = [
        slice[1], slice[3], slice[5], slice[7], slice[9], slice[11], slice[13], slice[15],
    ];
    (evens, odds)
}

/// Boundary-safe gather with clamping for edge cases
#[inline(always)]
fn gather_even_odd_x8_boundary(plane: &[f32], start_idx: usize) -> ([f32; 8], [f32; 8]) {
    let get = |offset: usize| -> f32 {
        let idx = start_idx + offset;
        if idx < plane.len() {
            plane[idx]
        } else {
            plane[plane.len() - 1]
        }
    };

    let evens = [
        get(0),
        get(2),
        get(4),
        get(6),
        get(8),
        get(10),
        get(12),
        get(14),
    ];

    let odds = [
        get(1),
        get(3),
        get(5),
        get(7),
        get(9),
        get(11),
        get(13),
        get(15),
    ];

    (evens, odds)
}

/// Gather even and odd indexed elements from a row into two f32 arrays.
///
/// Given input [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, ...]:
/// - evens = [a, c, e, g, i, k, m, o]
/// - odds = [b, d, f, h, j, l, n, p]
///
/// Returns raw `[f32; 8]` arrays. Callers load into SIMD vectors as needed.
#[inline(always)]
fn gather_even_odd_x8(plane: &[f32], start_idx: usize, _width: usize) -> ([f32; 8], [f32; 8]) {
    // Fast path: when we have at least 16 elements available
    if start_idx + 16 <= plane.len() {
        let slice = &plane[start_idx..start_idx + 16];

        // Use runtime dispatch with inline function calls (no pointer indirection)
        // The branch is very predictable and intrinsics are inlined
        #[cfg(target_arch = "x86_64")]
        {
            if let Some(token) = archmage::X64V3Token::summon() {
                return gather_even_odd_x8_avx2(token, slice.try_into().unwrap());
            } else {
                return gather_even_odd_x8_scalar(slice);
            }
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            return gather_even_odd_x8_scalar(slice);
        }
    }

    // Slow path: boundary-safe gather with clamping
    gather_even_odd_x8_boundary(plane, start_idx)
}

// ============================================================================
// AVX2 RGB to YCbCr Intrinsics
// ============================================================================

/// Extract 4 R values from 16 bytes of RGB data using SSSE3 shuffle.
/// Input: [R0 G0 B0 R1 G1 B1 R2 G2 B2 R3 G3 B3 R4 G4 B4 R5]
/// Output: [R0 R1 R2 R3 0 0 0 0 0 0 0 0 0 0 0 0] (low 4 bytes valid)
#[cfg(target_arch = "x86_64")]
#[rite]
fn extract_r_ssse3(_token: archmage::X64V3Token, rgb: __m128i) -> __m128i {
    // Shuffle mask: extract bytes 0, 3, 6, 9 (R values)
    // Uses _mm_shuffle_epi8 which requires SSSE3
    let mask = _mm_setr_epi8(0, 3, 6, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    _mm_shuffle_epi8(rgb, mask)
}

/// Extract 4 G values from 16 bytes of RGB data using SSSE3 shuffle.
#[cfg(target_arch = "x86_64")]
#[rite]
fn extract_g_ssse3(_token: archmage::X64V3Token, rgb: __m128i) -> __m128i {
    // Shuffle mask: extract bytes 1, 4, 7, 10 (G values)
    // Uses _mm_shuffle_epi8 which requires SSSE3
    let mask = _mm_setr_epi8(1, 4, 7, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    _mm_shuffle_epi8(rgb, mask)
}

/// Extract 4 B values from 16 bytes of RGB data using SSSE3 shuffle.
#[cfg(target_arch = "x86_64")]
#[rite]
fn extract_b_ssse3(_token: archmage::X64V3Token, rgb: __m128i) -> __m128i {
    // Shuffle mask: extract bytes 2, 5, 8, 11 (B values)
    // Uses _mm_shuffle_epi8 which requires SSSE3
    let mask = _mm_setr_epi8(2, 5, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    _mm_shuffle_epi8(rgb, mask)
}

/// Convert 4 u8 values (in low bytes of __m128i) to __m128 f32.
/// Uses _mm_cvtepu8_epi32 which requires SSE4.1.
#[cfg(target_arch = "x86_64")]
#[rite]
fn u8x4_to_f32x4_sse41(_token: archmage::X64V3Token, v: __m128i) -> __m128 {
    use core::arch::x86_64::_mm_cvtepi32_ps;
    // Zero-extend u8 to i32, then convert to f32
    let i32_vec = _mm_cvtepu8_epi32(v);
    _mm_cvtepi32_ps(i32_vec)
}

/// FMA intrinsics implementation for RGB to YCbCr conversion.
///
/// Processes 8 pixels at a time using explicit SSSE3/SSE4.1/FMA intrinsics for
/// deinterleaving RGB data, which LLVM cannot auto-vectorize effectively.
/// Uses FMA (fused multiply-add) for better performance and precision.
///
/// This is a low-level function called by the safe `rgb_to_ycbcr_planes_simd_inplace` wrapper.
/// Production code should use the safe wrapper.
#[cfg(target_arch = "x86_64")]
#[arcane]
#[inline]
pub(crate) fn rgb_to_ycbcr_8px_fma(
    _token: archmage::X64V3Token,
    rgb_data: &[u8],
    y_out: &mut [f32; 8],
    cb_out: &mut [f32; 8],
    cr_out: &mut [f32; 8],
) {
    // Load 24 bytes as two overlapping 16-byte loads
    let rgb0 = safe_simd::_mm_loadu_si128(<&[u8; 16]>::try_from(&rgb_data[..16]).unwrap());
    let rgb1 = safe_simd::_mm_loadu_si128(<&[u8; 16]>::try_from(&rgb_data[12..28]).unwrap());

    // Extract R, G, B for first 4 pixels
    let r0_bytes = extract_r_ssse3(_token, rgb0);
    let g0_bytes = extract_g_ssse3(_token, rgb0);
    let b0_bytes = extract_b_ssse3(_token, rgb0);

    // Extract R, G, B for second 4 pixels
    let r1_bytes = extract_r_ssse3(_token, rgb1);
    let g1_bytes = extract_g_ssse3(_token, rgb1);
    let b1_bytes = extract_b_ssse3(_token, rgb1);

    // Convert to f32
    let r0: __m128 = u8x4_to_f32x4_sse41(_token, r0_bytes);
    let g0: __m128 = u8x4_to_f32x4_sse41(_token, g0_bytes);
    let b0: __m128 = u8x4_to_f32x4_sse41(_token, b0_bytes);
    let r1: __m128 = u8x4_to_f32x4_sse41(_token, r1_bytes);
    let g1: __m128 = u8x4_to_f32x4_sse41(_token, g1_bytes);
    let b1: __m128 = u8x4_to_f32x4_sse41(_token, b1_bytes);

    // Coefficients
    let r_to_y = _mm_set1_ps(YCBCR_R_TO_Y);
    let g_to_y = _mm_set1_ps(YCBCR_G_TO_Y);
    let b_to_y = _mm_set1_ps(YCBCR_B_TO_Y);
    let r_to_cb = _mm_set1_ps(YCBCR_R_TO_CB);
    let g_to_cb = _mm_set1_ps(YCBCR_G_TO_CB);
    let b_to_cb = _mm_set1_ps(YCBCR_B_TO_CB);
    let r_to_cr = _mm_set1_ps(YCBCR_R_TO_CR);
    let g_to_cr = _mm_set1_ps(YCBCR_G_TO_CR);
    let b_to_cr = _mm_set1_ps(YCBCR_B_TO_CR);
    let offset_128 = _mm_set1_ps(128.0);

    // Compute Y, Cb, Cr for first 4 pixels using FMA
    // y = r * r_to_y + g * g_to_y + b * b_to_y
    let y0 = _mm_fmadd_ps(b0, b_to_y, _mm_fmadd_ps(g0, g_to_y, _mm_mul_ps(r0, r_to_y)));
    // cb = 128 + r * r_to_cb + g * g_to_cb + b * b_to_cb
    let cb0 = _mm_fmadd_ps(
        b0,
        b_to_cb,
        _mm_fmadd_ps(g0, g_to_cb, _mm_fmadd_ps(r0, r_to_cb, offset_128)),
    );
    // cr = 128 + r * r_to_cr + g * g_to_cr + b * b_to_cr
    let cr0 = _mm_fmadd_ps(
        b0,
        b_to_cr,
        _mm_fmadd_ps(g0, g_to_cr, _mm_fmadd_ps(r0, r_to_cr, offset_128)),
    );

    // Compute Y, Cb, Cr for second 4 pixels using FMA
    let y1 = _mm_fmadd_ps(b1, b_to_y, _mm_fmadd_ps(g1, g_to_y, _mm_mul_ps(r1, r_to_y)));
    let cb1 = _mm_fmadd_ps(
        b1,
        b_to_cb,
        _mm_fmadd_ps(g1, g_to_cb, _mm_fmadd_ps(r1, r_to_cb, offset_128)),
    );
    let cr1 = _mm_fmadd_ps(
        b1,
        b_to_cr,
        _mm_fmadd_ps(g1, g_to_cr, _mm_fmadd_ps(r1, r_to_cr, offset_128)),
    );

    // Store results (two 4-element stores per plane)
    safe_simd::_mm_storeu_ps(<&mut [f32; 4]>::try_from(&mut y_out[..4]).unwrap(), y0);
    safe_simd::_mm_storeu_ps(<&mut [f32; 4]>::try_from(&mut y_out[4..]).unwrap(), y1);
    safe_simd::_mm_storeu_ps(<&mut [f32; 4]>::try_from(&mut cb_out[..4]).unwrap(), cb0);
    safe_simd::_mm_storeu_ps(<&mut [f32; 4]>::try_from(&mut cb_out[4..]).unwrap(), cb1);
    safe_simd::_mm_storeu_ps(<&mut [f32; 4]>::try_from(&mut cr_out[..4]).unwrap(), cr0);
    safe_simd::_mm_storeu_ps(<&mut [f32; 4]>::try_from(&mut cr_out[4..]).unwrap(), cr1);
}

/// Scalar reference implementation for RGB to YCbCr (for testing).
#[cfg(all(test, target_arch = "x86_64"))]
fn rgb_to_ycbcr_scalar(
    rgb_data: &[u8],
    y_plane: &mut [f32],
    cb_plane: &mut [f32],
    cr_plane: &mut [f32],
    num_pixels: usize,
) {
    for i in 0..num_pixels {
        let rgb_idx = i * 3;
        let r = rgb_data[rgb_idx] as f32;
        let g = rgb_data[rgb_idx + 1] as f32;
        let b = rgb_data[rgb_idx + 2] as f32;

        y_plane[i] = YCBCR_R_TO_Y.mul_add(r, YCBCR_G_TO_Y.mul_add(g, YCBCR_B_TO_Y * b));
        cb_plane[i] =
            YCBCR_R_TO_CB.mul_add(r, YCBCR_G_TO_CB.mul_add(g, YCBCR_B_TO_CB.mul_add(b, 128.0)));
        cr_plane[i] =
            YCBCR_R_TO_CR.mul_add(r, YCBCR_G_TO_CR.mul_add(g, YCBCR_B_TO_CR.mul_add(b, 128.0)));
    }
}

// ============================================================================
// Strided RGB→YCbCr Conversion (for strip encoder)
// ============================================================================

/// RGB to YCbCr with strided output using pre-allocated u8 buffers.
///
/// Dispatches to `zenyuv` via `fast_yuv` (15-bit fixed-point on AVX2,
/// f32 FMA on other platforms). The u8 temp planes are caller-owned so
/// no allocation occurs per strip.
///
/// # Arguments
/// * `rgb_data` - Input RGB data (width × height × bpp bytes)
/// * `y_plane` - Output Y plane (y_stride × height elements)
/// * `cb_plane` - Output Cb plane (width × height elements)
/// * `cr_plane` - Output Cr plane (width × height elements)
/// * `yuv_temp_y` - Reusable u8 buffer (width × height bytes)
/// * `yuv_temp_cb` - Reusable u8 buffer (width × height bytes)
/// * `yuv_temp_cr` - Reusable u8 buffer (width × height bytes)
/// * `width` - Image width in pixels
/// * `height` - Number of rows to process
/// * `y_stride` - Y output stride (typically padded_width)
/// * `bpp` - Bytes per pixel (3 for RGB, 4 for RGBA)
pub fn rgb_to_ycbcr_strided_reuse(
    rgb_data: &[u8],
    y_plane: &mut [f32],
    cb_plane: &mut [f32],
    cr_plane: &mut [f32],
    yuv_temp_y: &mut [u8],
    yuv_temp_cb: &mut [u8],
    yuv_temp_cr: &mut [u8],
    width: usize,
    height: usize,
    y_stride: usize,
    bpp: usize,
) {
    crate::color::fast_yuv::rgb_to_ycbcr_strided_reuse(
        rgb_data,
        y_plane,
        cb_plane,
        cr_plane,
        yuv_temp_y,
        yuv_temp_cb,
        yuv_temp_cr,
        width,
        height,
        y_stride,
        bpp,
    );
}

/// BGR to YCbCr with strided output using pre-allocated u8 buffers.
///
/// Same as `rgb_to_ycbcr_strided_reuse` but for BGR/BGRA input.
pub fn bgr_to_ycbcr_strided_reuse(
    bgr_data: &[u8],
    y_plane: &mut [f32],
    cb_plane: &mut [f32],
    cr_plane: &mut [f32],
    yuv_temp_y: &mut [u8],
    yuv_temp_cb: &mut [u8],
    yuv_temp_cr: &mut [u8],
    width: usize,
    height: usize,
    y_stride: usize,
    bpp: usize,
) {
    crate::color::fast_yuv::bgr_to_ycbcr_strided_reuse(
        bgr_data,
        y_plane,
        cb_plane,
        cr_plane,
        yuv_temp_y,
        yuv_temp_cb,
        yuv_temp_cr,
        width,
        height,
        y_stride,
        bpp,
    );
}

/// BGR variant of strided conversion (for BGR/BGRA input).
pub fn bgr_to_ycbcr_strided_inplace(
    bgr_data: &[u8],
    y_plane: &mut [f32],
    cb_plane: &mut [f32],
    cr_plane: &mut [f32],
    width: usize,
    height: usize,
    y_stride: usize,
    bpp: usize,
) {
    debug_assert!(bgr_data.len() >= width * height * bpp);
    debug_assert!(y_plane.len() >= y_stride * height);
    debug_assert!(cb_plane.len() >= width * height);
    debug_assert!(cr_plane.len() >= width * height);

    crate::color::fast_yuv::bgr_to_ycbcr_strided_fast(
        bgr_data, y_plane, cb_plane, cr_plane, width, height, y_stride, bpp,
    );
}

// ============================================================================
// Block Extraction
// ============================================================================

/// SIMD-optimized block extraction for XYB planes with scaling.
///
/// XYB values are in range ~[-2.1, 7.3], need to scale by 255 and level shift by -128.
///
/// # Arguments
/// * `plane` - Input XYB plane data
/// * `width` - Plane width
/// * `height` - Plane height
/// * `bx` - Block x coordinate (in blocks)
/// * `by` - Block y coordinate (in blocks)
#[allow(dead_code)] // Used by quantize_all_blocks_xyb_with_aq (XYB trellis path, not yet integrated)
#[inline]
pub fn extract_block_xyb_simd(
    plane: &[f32],
    width: usize,
    height: usize,
    bx: usize,
    by: usize,
) -> [f32; 64] {
    incant!(extract_block_xyb_simd_impl(plane, width, height, bx, by))
}

#[allow(dead_code)] // Used by extract_block_xyb_simd
#[magetypes(v3, neon, wasm128, scalar)]
fn extract_block_xyb_simd_impl(
    token: Token,
    plane: &[f32],
    width: usize,
    height: usize,
    bx: usize,
    by: usize,
) -> [f32; 64] {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;

    let px_start = bx * 8;
    let py_start = by * 8;

    let is_interior = px_start + 8 <= width && py_start + 8 <= height;

    let scale = f32x8::splat(token, 255.0);
    let level_shift = f32x8::splat(token, 128.0);
    let mut block = [0.0f32; 64];

    if is_interior {
        for y in 0..8 {
            let row_start = (py_start + y) * width + px_start;
            // Load 8 consecutive f32 values
            let row = load_f32x8(token, plane, row_start);

            // XYB: val * 255.0 - 128.0
            let scaled = row * scale - level_shift;

            scaled.store(<&mut [f32; 8]>::try_from(&mut block[y * 8..y * 8 + 8]).unwrap());
        }
    } else {
        for y in 0..8 {
            let py = (py_start + y).min(height - 1);
            for x in 0..8 {
                let px = (px_start + x).min(width - 1);
                block[y * 8 + x] = plane[py * width + px] * 255.0 - 128.0;
            }
        }
    }

    block
}
// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Tolerance for FMA vs non-FMA differences (FMA avoids intermediate rounding)
    const EPSILON: f32 = 1e-4;
    /// Slightly higher tolerance for accumulated operations (downsampling averages 4 values)
    #[cfg(target_arch = "x86_64")]
    const EPSILON_ACCUMULATED: f32 = 5e-4;

    #[test]
    fn test_gather_even_odd_x8_correctness() {
        // Create test data: 16 sequential floats
        let data: Vec<f32> = (0..32).map(|i| i as f32).collect();

        // Call the function
        let (evens_arr, odds_arr) = gather_even_odd_x8(&data, 0, 32);

        // Expected: evens = [0, 2, 4, 6, 8, 10, 12, 14]
        let expected_evens = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0];
        let expected_odds = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0];

        for i in 0..8 {
            assert!(
                (evens_arr[i] - expected_evens[i]).abs() < EPSILON,
                "evens[{}]: got {}, expected {}",
                i,
                evens_arr[i],
                expected_evens[i]
            );
            assert!(
                (odds_arr[i] - expected_odds[i]).abs() < EPSILON,
                "odds[{}]: got {}, expected {}",
                i,
                odds_arr[i],
                expected_odds[i]
            );
        }

        // Test with offset
        let (evens2_arr, odds2_arr) = gather_even_odd_x8(&data, 4, 32);

        // With offset 4: evens = [4, 6, 8, 10, 12, 14, 16, 18]
        let expected_evens2 = [4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0];
        let expected_odds2 = [5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0];

        for i in 0..8 {
            assert!(
                (evens2_arr[i] - expected_evens2[i]).abs() < EPSILON,
                "evens2[{}]: got {}, expected {}",
                i,
                evens2_arr[i],
                expected_evens2[i]
            );
            assert!(
                (odds2_arr[i] - expected_odds2[i]).abs() < EPSILON,
                "odds2[{}]: got {}, expected {}",
                i,
                odds2_arr[i],
                expected_odds2[i]
            );
        }
    }

    /// Test AVX2 intrinsics RGB to YCbCr against scalar reference.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_rgb_to_ycbcr_avx2_matches_scalar() {
        let Some(token) = archmage::X64V3Token::summon() else {
            return;
        };

        // Test with 8 pixels (one AVX2 batch)
        // Need 28 bytes: second overlapping 16-byte load reads [12..28], extra 4 bytes are discarded by shuffles
        let rgb_data: Vec<u8> = (0..28).map(|i| ((i * 17 + 5) % 256) as u8).collect();

        let mut y_avx2 = [0.0f32; 8];
        let mut cb_avx2 = [0.0f32; 8];
        let mut cr_avx2 = [0.0f32; 8];

        rgb_to_ycbcr_8px_fma(token, &rgb_data, &mut y_avx2, &mut cb_avx2, &mut cr_avx2);

        let mut y_scalar = vec![0.0f32; 8];
        let mut cb_scalar = vec![0.0f32; 8];
        let mut cr_scalar = vec![0.0f32; 8];
        rgb_to_ycbcr_scalar(&rgb_data, &mut y_scalar, &mut cb_scalar, &mut cr_scalar, 8);

        for i in 0..8 {
            let y_diff = (y_avx2[i] - y_scalar[i]).abs();
            let cb_diff = (cb_avx2[i] - cb_scalar[i]).abs();
            let cr_diff = (cr_avx2[i] - cr_scalar[i]).abs();
            assert!(
                y_diff < EPSILON_ACCUMULATED,
                "Y mismatch at {}: AVX2={}, scalar={}, diff={}",
                i,
                y_avx2[i],
                y_scalar[i],
                y_diff
            );
            assert!(
                cb_diff < EPSILON_ACCUMULATED,
                "Cb mismatch at {}: AVX2={}, scalar={}, diff={}",
                i,
                cb_avx2[i],
                cb_scalar[i],
                cb_diff
            );
            assert!(
                cr_diff < EPSILON_ACCUMULATED,
                "Cr mismatch at {}: AVX2={}, scalar={}, diff={}",
                i,
                cr_avx2[i],
                cr_scalar[i],
                cr_diff
            );
        }
    }

    /// Brute force test AVX2 RGB to YCbCr with all possible u8 values.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_rgb_to_ycbcr_avx2_brute_force() {
        let Some(token) = archmage::X64V3Token::summon() else {
            return;
        };

        // Test systematic patterns covering all u8 values
        let mut max_y_diff = 0.0f32;
        let mut max_cb_diff = 0.0f32;
        let mut max_cr_diff = 0.0f32;

        // Test every 16th R, G, B combination (256/16 = 16^3 = 4096 combinations)
        for r_base in (0u8..=255).step_by(16) {
            for g_base in (0u8..=255).step_by(64) {
                for b_base in (0u8..=255).step_by(64) {
                    // Create 8 pixels with slight variations
                    // Need 28 bytes: second overlapping 16-byte load reads [12..28], extra 4 bytes are discarded by shuffles
                    let mut rgb_data = vec![0u8; 28];
                    for p in 0..8 {
                        let r = r_base.wrapping_add((p * 2) as u8);
                        let g = g_base.wrapping_add((p * 3) as u8);
                        let b = b_base.wrapping_add((p * 5) as u8);
                        rgb_data[p * 3] = r;
                        rgb_data[p * 3 + 1] = g;
                        rgb_data[p * 3 + 2] = b;
                    }

                    let mut y_avx2 = [0.0f32; 8];
                    let mut cb_avx2 = [0.0f32; 8];
                    let mut cr_avx2 = [0.0f32; 8];

                    rgb_to_ycbcr_8px_fma(token, &rgb_data, &mut y_avx2, &mut cb_avx2, &mut cr_avx2);

                    let mut y_scalar = vec![0.0f32; 8];
                    let mut cb_scalar = vec![0.0f32; 8];
                    let mut cr_scalar = vec![0.0f32; 8];
                    rgb_to_ycbcr_scalar(
                        &rgb_data,
                        &mut y_scalar,
                        &mut cb_scalar,
                        &mut cr_scalar,
                        8,
                    );

                    for i in 0..8 {
                        max_y_diff = max_y_diff.max((y_avx2[i] - y_scalar[i]).abs());
                        max_cb_diff = max_cb_diff.max((cb_avx2[i] - cb_scalar[i]).abs());
                        max_cr_diff = max_cr_diff.max((cr_avx2[i] - cr_scalar[i]).abs());
                    }
                }
            }
        }

        // Allow tiny floating-point differences from different operation ordering
        assert!(
            max_y_diff < EPSILON_ACCUMULATED,
            "Max Y diff too large: {}",
            max_y_diff
        );
        assert!(
            max_cb_diff < EPSILON_ACCUMULATED,
            "Max Cb diff too large: {}",
            max_cb_diff
        );
        assert!(
            max_cr_diff < EPSILON_ACCUMULATED,
            "Max Cr diff too large: {}",
            max_cr_diff
        );
    }
}
