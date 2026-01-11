//! SIMD-optimized encoding functions.
//!
//! Contains SIMD implementations for encoder hot paths:
//! - Chroma downsampling (2x2, 2x1, 1x2)
//! - RGB to YCbCr color conversion
//!
//! # Safe/Unsafe Architecture
//!
//! This module follows a two-layer pattern:
//!
//! 1. **Safe public APIs** (e.g., `downsample_2x2_simd`, `rgb_to_ycbcr_planes_simd_inplace`):
//!    - Use `wide` crate's portable SIMD types (`f32x8`)
//!    - Safe load/store helpers with bounds checking
//!    - Runtime CPU feature detection via `multiversion` or `is_x86_feature_detected!`
//!
//! 2. **Unsafe internal functions** (e.g., `downsample_2x2_avx2`, `rgb_to_ycbcr_8px_avx2`):
//!    - Raw AVX2/SSE intrinsics for operations without `wide` equivalents
//!    - Permute, shuffle, and byte-level operations for de-interleaving
//!    - Marked `pub(crate)` for testing, but production code uses safe wrappers
//!
//! The unsafe intrinsics are necessary for operations like:
//! - `_mm256_permutevar8x32_ps`: Variable element permute (even/odd gather)
//! - `_mm_shuffle_epi8`: Byte shuffle (RGB channel extraction)
//! - `_mm256_permute2f128_ps`: 128-bit lane permute (matrix transpose)
//!
//! All functions have tests to verify parity with scalar implementations.

use multiversion::multiversion;
use wide::f32x8;

// Raw AVX2/SSE intrinsics - only available with `unsafe_simd` feature on x86_64
#[cfg(all(feature = "unsafe_simd", target_arch = "x86_64"))]
use core::arch::x86_64::{
    __m128, __m128i, __m256, _mm256_add_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_permute2f128_ps,
    _mm256_permutevar8x32_ps, _mm256_set1_ps, _mm256_setr_epi32, _mm256_storeu_ps,
    _mm_cvtepu8_epi32, _mm_fmadd_ps, _mm_loadu_si128, _mm_mul_ps, _mm_set1_ps, _mm_setr_epi8,
    _mm_shuffle_epi8, _mm_storeu_ps,
};

use crate::consts::{
    YCBCR_B_TO_CB, YCBCR_B_TO_CR, YCBCR_B_TO_Y, YCBCR_G_TO_CB, YCBCR_G_TO_CR, YCBCR_G_TO_Y,
    YCBCR_R_TO_CB, YCBCR_R_TO_CR, YCBCR_R_TO_Y,
};
use crate::error::Result;
use crate::foundation::alloc::try_alloc_zeroed_f32;

// ============================================================================
// Memory Allocation Helpers
// ============================================================================

/// Allocate a Vec<f32> for writing with fallible allocation.
///
/// Uses zeroed allocation which may be faster than uninitialized because:
/// - OS can provide pre-zeroed pages from zero-page pool
/// - Avoids lazy page fault overhead on first write
/// - Pages are mapped immediately rather than on first touch
#[inline(always)]
fn try_alloc_f32(len: usize, context: &'static str) -> Result<Vec<f32>> {
    try_alloc_zeroed_f32(len, context)
}

// ============================================================================
// Safe SIMD Load/Store Helpers
// ============================================================================

/// Safely load 8 f32s into f32x8. Panics if slice is too short.
/// LLVM optimizes this to identical code as unsafe pointer cast.
#[inline(always)]
fn load_f32x8(slice: &[f32], offset: usize) -> f32x8 {
    f32x8::from(<[f32; 8]>::try_from(&slice[offset..offset + 8]).unwrap())
}

/// Safely store f32x8 to slice. Panics if slice is too short.
#[inline(always)]
fn store_f32x8(slice: &mut [f32], offset: usize, value: f32x8) {
    let arr: [f32; 8] = value.into();
    slice[offset..offset + 8].copy_from_slice(&arr);
}

// ============================================================================
// Type Conversion Helpers
// ============================================================================

/// SIMD-optimized conversion from u8 slice to f32 Vec.
///
/// Processes 8 elements at a time, converting each u8 to f32.
/// Used for converting YUV planes from external crates to internal f32 representation.
///
/// # Arguments
/// * `input` - Input slice of u8 values
///
/// # Returns
/// Vec of f32 values, or error on allocation failure
#[inline]
pub fn u8_slice_to_f32_simd(input: &[u8]) -> Result<Vec<f32>> {
    let len = input.len();
    let mut result = try_alloc_f32(len, "u8_slice_to_f32_simd")?;

    let chunks = len / 8;

    // Process 8 elements at a time
    for i in 0..chunks {
        let k = i * 8;
        // Load 8 u8 values and convert to f32
        let vals = f32x8::from([
            input[k] as f32,
            input[k + 1] as f32,
            input[k + 2] as f32,
            input[k + 3] as f32,
            input[k + 4] as f32,
            input[k + 5] as f32,
            input[k + 6] as f32,
            input[k + 7] as f32,
        ]);
        let arr: [f32; 8] = vals.into();
        result[k..k + 8].copy_from_slice(&arr);
    }

    // Handle remaining elements (scalar)
    let remainder_start = chunks * 8;
    for i in remainder_start..len {
        result[i] = input[i] as f32;
    }

    Ok(result)
}

/// SIMD-optimized conversion from u8 iterator to f32 Vec.
///
/// Same as u8_slice_to_f32_simd but works with an iterator that has a known length.
#[inline]
pub fn u8_iter_to_f32_simd(input: impl Iterator<Item = u8>, len: usize) -> Result<Vec<f32>> {
    let mut result = try_alloc_f32(len, "u8_iter_to_f32_simd")?;
    let mut i = 0;

    // Collect into buffer for SIMD processing
    let mut buf = [0u8; 8];
    let mut buf_idx = 0;

    for val in input {
        buf[buf_idx] = val;
        buf_idx += 1;

        if buf_idx == 8 {
            let vals = f32x8::from([
                buf[0] as f32,
                buf[1] as f32,
                buf[2] as f32,
                buf[3] as f32,
                buf[4] as f32,
                buf[5] as f32,
                buf[6] as f32,
                buf[7] as f32,
            ]);
            let arr: [f32; 8] = vals.into();
            result[i..i + 8].copy_from_slice(&arr);
            i += 8;
            buf_idx = 0;
        }
    }

    // Handle remaining elements
    for j in 0..buf_idx {
        result[i + j] = buf[j] as f32;
    }

    Ok(result)
}

/// SIMD-optimized scaling of f32 slice by a constant factor.
///
/// Multiplies all elements by the given scale factor.
/// Used for scaling Y plane from [0,1] to [0,255] for AQ computation.
///
/// # Arguments
/// * `input` - Input slice of f32 values
/// * `scale` - Scale factor to multiply by
///
/// # Returns
/// Vec of scaled f32 values, or error on allocation failure
#[inline]
pub fn scale_f32_slice_simd(input: &[f32], scale: f32) -> Result<Vec<f32>> {
    let len = input.len();
    let mut result = try_alloc_f32(len, "scale_f32_slice_simd")?;

    let chunks = len / 8;
    let scale_vec = f32x8::splat(scale);

    // Process 8 elements at a time (zero-cost load from contiguous memory)
    for i in 0..chunks {
        let k = i * 8;
        let vals_arr: [f32; 8] = input[k..k + 8].try_into().unwrap();
        let vals = f32x8::from(vals_arr);
        let scaled = vals * scale_vec;
        let arr: [f32; 8] = scaled.into();
        result[k..k + 8].copy_from_slice(&arr);
    }

    // Handle remaining elements (scalar)
    let remainder_start = chunks * 8;
    for i in remainder_start..len {
        result[i] = input[i] * scale;
    }

    Ok(result)
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
#[multiversion(targets("x86_64+avx2+fma", "x86_64+sse2", "aarch64+neon"))]
pub fn downsample_2x2_simd_inplace(plane: &[f32], width: usize, height: usize, result: &mut [f32]) {
    let new_width = (width + 1) / 2;
    let new_height = (height + 1) / 2;
    debug_assert!(result.len() >= new_width * new_height);

    let scale = f32x8::splat(0.25);
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
            let (p00, p10) = gather_even_odd_x8(plane, row0_idx, width);
            let (p01, p11) = gather_even_odd_x8(plane, row1_idx, width);

            // Box filter: (p00 + p10 + p01 + p11) * 0.25
            let sum = p00 + p10 + p01 + p11;
            let avg = sum * scale;

            // Store result
            store_f32x8(result, out_row_start + out_x, avg);
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

/// SIMD-optimized 2x2 box filter downsampling.
///
/// Processes 8 output pixels at a time (requires 16 input pixels per row).
///
/// # Arguments
/// * `plane` - Input plane (f32, full resolution)
/// * `width` - Input width
/// * `height` - Input height
///
/// # Returns
/// Downsampled plane at half resolution, or error on allocation failure
pub fn downsample_2x2_simd(plane: &[f32], width: usize, height: usize) -> Result<Vec<f32>> {
    let new_width = (width + 1) / 2;
    let new_height = (height + 1) / 2;
    let mut result = try_alloc_f32(new_width * new_height, "downsample_2x2_simd")?;

    downsample_2x2_simd_inplace(plane, width, height, &mut result);

    Ok(result)
}

/// SIMD-optimized 2x1 (horizontal) box filter downsampling.
pub fn downsample_2x1_simd(plane: &[f32], width: usize, height: usize) -> Result<Vec<f32>> {
    let new_width = (width + 1) / 2;
    let mut result = try_alloc_f32(new_width * height, "downsample_2x1_simd")?;

    let scale = f32x8::splat(0.5);
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
            let (p0, p1) = gather_even_odd_x8(plane, in_row_start + in_x, width);

            // Box filter: (p0 + p1) * 0.5
            let avg = (p0 + p1) * scale;

            // Store result
            store_f32x8(&mut result, out_row_start + out_x, avg);
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

    Ok(result)
}

/// SIMD-optimized 2x1 (horizontal) box filter downsampling (in-place).
///
/// Writes to pre-allocated result buffer.
pub fn downsample_2x1_simd_inplace(plane: &[f32], width: usize, height: usize, result: &mut [f32]) {
    let new_width = (width + 1) / 2;
    debug_assert!(result.len() >= new_width * height);

    let scale = f32x8::splat(0.5);
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
            let (p0, p1) = gather_even_odd_x8(plane, in_row_start + in_x, width);

            // Box filter: (p0 + p1) * 0.5
            let avg = (p0 + p1) * scale;

            // Store result
            store_f32x8(result, out_row_start + out_x, avg);
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

/// SIMD-optimized 1x2 (vertical) box filter downsampling.
pub fn downsample_1x2_simd(plane: &[f32], width: usize, height: usize) -> Result<Vec<f32>> {
    let new_height = (height + 1) / 2;
    let mut result = try_alloc_f32(width * new_height, "downsample_1x2_simd")?;

    let scale = f32x8::splat(0.5);
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

            let p0 = load_f32x8(plane, row0_idx);
            let p1 = load_f32x8(plane, row1_idx);

            let avg = (p0 + p1) * scale;

            // Store result
            store_f32x8(&mut result, out_row_start + x, avg);
        }

        // Scalar remainder
        for x in (chunks * 8)..width {
            let p0 = plane[y0 * width + x];
            let p1 = plane[y1 * width + x];
            result[out_row_start + x] = (p0 + p1) * 0.5;
        }
    }

    Ok(result)
}

/// SIMD-optimized 1x2 (vertical) box filter downsampling (in-place).
///
/// Writes to pre-allocated result buffer.
pub fn downsample_1x2_simd_inplace(plane: &[f32], width: usize, height: usize, result: &mut [f32]) {
    let new_height = (height + 1) / 2;
    debug_assert!(result.len() >= width * new_height);

    let scale = f32x8::splat(0.5);
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

            let p0 = load_f32x8(plane, row0_idx);
            let p1 = load_f32x8(plane, row1_idx);

            let avg = (p0 + p1) * scale;

            // Store result
            store_f32x8(result, out_row_start + x, avg);
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

/// AVX2 intrinsics-based deinterleave: extract evens and odds from 16 consecutive floats.
///
/// Given input [a,b,c,d,e,f,g,h, i,j,k,l,m,n,o,p]:
/// - evens = [a,c,e,g,i,k,m,o]
/// - odds  = [b,d,f,h,j,l,n,p]
///
/// # Safety
/// - Requires AVX2 CPU feature
/// - `ptr` must point to at least 16 readable f32 values
#[cfg(all(feature = "unsafe_simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn gather_even_odd_avx2_raw(ptr: *const f32) -> (__m256, __m256) {
    // Load 16 consecutive floats
    let v0 = _mm256_loadu_ps(ptr);
    let v1 = _mm256_loadu_ps(ptr.add(8));

    // Permute indices: [0,2,4,6,1,3,5,7] puts evens in low 128, odds in high 128
    let perm_idx = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);

    let permuted0 = _mm256_permutevar8x32_ps(v0, perm_idx);
    let permuted1 = _mm256_permutevar8x32_ps(v1, perm_idx);

    // Combine: low128s together, high128s together
    // 0x20 = select low128 from both, 0x31 = select high128 from both
    let evens = _mm256_permute2f128_ps(permuted0, permuted1, 0x20);
    let odds = _mm256_permute2f128_ps(permuted0, permuted1, 0x31);

    (evens, odds)
}

/// AVX2 intrinsics-based 2x2 box filter downsample.
///
/// Processes 8 output pixels at a time using AVX2 permute instructions
/// for efficient deinterleaving.
///
/// This is a low-level function exposed for testing parity with the safe
/// `downsample_2x2_simd_inplace` wrapper. Production code should use the safe wrapper.
///
/// # Safety
/// - Requires AVX2 CPU feature
/// - All buffer bounds must be pre-validated
#[cfg(all(feature = "unsafe_simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) unsafe fn downsample_2x2_avx2(
    plane: &[f32],
    width: usize,
    height: usize,
    result: &mut [f32],
) {
    let new_width = (width + 1) / 2;
    let new_height = (height + 1) / 2;

    let scale = _mm256_set1_ps(0.25);

    // Calculate how many SIMD chunks we can process per row.
    // We need 16 consecutive input pixels (in_x to in_x+15) per chunk.
    // So max in_x for SIMD is width - 16, meaning max out_x is (width - 16) / 2.
    let simd_chunks_per_row = if width >= 16 {
        (width - 16) / 2 / 8 + 1
    } else {
        0
    };
    // But also cap based on output width
    let simd_chunks_per_row = simd_chunks_per_row.min(new_width / 8);

    for y in 0..new_height {
        let y0 = y * 2;
        let y1 = (y0 + 1).min(height - 1);
        let out_row_start = y * new_width;

        // AVX2 SIMD path: process 8 output pixels at a time
        for chunk in 0..simd_chunks_per_row {
            let out_x = chunk * 8;
            let in_x = out_x * 2;

            // Extra safety check: ensure we have 16 pixels within the row
            if in_x + 16 > width {
                break;
            }

            let row0_ptr = plane.as_ptr().add(y0 * width + in_x);
            let row1_ptr = plane.as_ptr().add(y1 * width + in_x);

            // Gather even/odd from row 0 and row 1 using AVX2 permutes
            let (p00, p10) = gather_even_odd_avx2_raw(row0_ptr);
            let (p01, p11) = gather_even_odd_avx2_raw(row1_ptr);

            // Box filter: (p00 + p10 + p01 + p11) * 0.25
            let sum01 = _mm256_add_ps(p00, p10);
            let sum23 = _mm256_add_ps(p01, p11);
            let sum = _mm256_add_ps(sum01, sum23);
            let avg = _mm256_mul_ps(sum, scale);

            // Store result
            let out_ptr = result.as_mut_ptr().add(out_row_start + out_x);
            _mm256_storeu_ps(out_ptr, avg);
        }

        // Scalar remainder - process all pixels not handled by SIMD
        let simd_processed = simd_chunks_per_row * 8;
        for out_x in simd_processed..new_width {
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

/// Scalar reference implementation of gather_even_odd for testing.
///
/// This is the ground truth implementation that AVX2 versions are tested against.
#[cfg(test)]
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

/// Scalar reference implementation of 2x2 downsample for testing.
///
/// This is the ground truth implementation that SIMD versions are tested against.
pub fn downsample_2x2_scalar(plane: &[f32], width: usize, height: usize) -> Vec<f32> {
    let new_width = (width + 1) / 2;
    let new_height = (height + 1) / 2;
    let mut result = vec![0.0f32; new_width * new_height];

    for y in 0..new_height {
        let y0 = y * 2;
        let y1 = (y0 + 1).min(height - 1);

        for x in 0..new_width {
            let x0 = x * 2;
            let x1 = (x0 + 1).min(width - 1);

            let p00 = plane[y0 * width + x0];
            let p10 = plane[y0 * width + x1];
            let p01 = plane[y1 * width + x0];
            let p11 = plane[y1 * width + x1];

            result[y * new_width + x] = (p00 + p10 + p01 + p11) * 0.25;
        }
    }

    result
}

/// AVX2-optimized deinterleave using Highway's ConcatEven/ConcatOdd pattern.
/// This is ~4x faster than element-by-element construction.
#[cfg(all(feature = "unsafe_simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn gather_even_odd_x8_avx2(ptr: *const f32) -> (f32x8, f32x8) {
    use std::arch::x86_64::*;

    // Load 16 consecutive floats as two YMM registers
    // Memory: [e0,o0,e1,o1,e2,o2,e3,o3, e4,o4,e5,o5,e6,o6,e7,o7]
    let lo = _mm256_loadu_ps(ptr); // [e0,o0,e1,o1 | e2,o2,e3,o3]
    let hi = _mm256_loadu_ps(ptr.add(8)); // [e4,o4,e5,o5 | e6,o6,e7,o7]

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
        std::mem::transmute::<__m256, f32x8>(evens_raw),
        std::mem::transmute::<__m256, f32x8>(odds_raw),
    )
}

/// Scalar fallback for gather_even_odd - used by non-AVX2 targets
#[inline(always)]
fn gather_even_odd_x8_scalar(ptr: *const f32) -> (f32x8, f32x8) {
    // SAFETY: Caller guarantees 16 elements are available
    unsafe {
        let a: [f32; 8] = *(ptr as *const [f32; 8]);
        let b: [f32; 8] = *(ptr.add(8) as *const [f32; 8]);
        let evens = f32x8::from([a[0], a[2], a[4], a[6], b[0], b[2], b[4], b[6]]);
        let odds = f32x8::from([a[1], a[3], a[5], a[7], b[1], b[3], b[5], b[7]]);
        (evens, odds)
    }
}

/// Boundary-safe gather with clamping for edge cases
#[inline(always)]
fn gather_even_odd_x8_boundary(plane: &[f32], start_idx: usize) -> (f32x8, f32x8) {
    let get = |offset: usize| -> f32 {
        let idx = start_idx + offset;
        if idx < plane.len() {
            plane[idx]
        } else {
            plane[plane.len() - 1]
        }
    };

    let evens = f32x8::from([
        get(0),
        get(2),
        get(4),
        get(6),
        get(8),
        get(10),
        get(12),
        get(14),
    ]);

    let odds = f32x8::from([
        get(1),
        get(3),
        get(5),
        get(7),
        get(9),
        get(11),
        get(13),
        get(15),
    ]);

    (evens, odds)
}

/// Gather even and odd indexed elements from a row into two f32x8 vectors.
///
/// Given input [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, ...]:
/// - evens = [a, c, e, g, i, k, m, o]
/// - odds = [b, d, f, h, j, l, n, p]
///
/// IMPORTANT: This is called from multiversioned functions. The caller
/// (downsample_2x2_simd_inplace) is compiled with AVX2 enabled via multiversion,
/// which means we can safely call AVX2 intrinsics here when the AVX2 version
/// of the caller is running.
///
/// We use `is_x86_feature_detected!` which is cheap (cached atomic load) to
/// select the right path. The branch predictor will quickly learn the pattern.
#[inline(always)]
fn gather_even_odd_x8(plane: &[f32], start_idx: usize, _width: usize) -> (f32x8, f32x8) {
    // Fast path: when we have at least 16 elements available
    if start_idx + 16 <= plane.len() {
        let ptr = unsafe { plane.as_ptr().add(start_idx) };

        // Use runtime dispatch with inline function calls (no pointer indirection)
        // The branch is very predictable and intrinsics are inlined
        #[cfg(all(feature = "unsafe_simd", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") {
                // SAFETY: AVX2 is detected
                return unsafe { gather_even_odd_x8_avx2(ptr) };
            } else {
                return gather_even_odd_x8_scalar(ptr);
            }
        }

        #[cfg(not(all(feature = "unsafe_simd", target_arch = "x86_64")))]
        {
            return gather_even_odd_x8_scalar(ptr);
        }
    }

    // Slow path: boundary-safe gather with clamping
    gather_even_odd_x8_boundary(plane, start_idx)
}

// ============================================================================
// AVX2 RGB to YCbCr Intrinsics
// ============================================================================

/// Extract 4 R values from 16 bytes of RGB data using SSE shuffle.
/// Input: [R0 G0 B0 R1 G1 B1 R2 G2 B2 R3 G3 B3 R4 G4 B4 R5]
/// Output: [R0 R1 R2 R3 0 0 0 0 0 0 0 0 0 0 0 0] (low 4 bytes valid)
#[cfg(all(feature = "unsafe_simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn extract_r_sse(rgb: __m128i) -> __m128i {
    // Shuffle mask: extract bytes 0, 3, 6, 9 (R values)
    let mask = _mm_setr_epi8(0, 3, 6, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    _mm_shuffle_epi8(rgb, mask)
}

/// Extract 4 G values from 16 bytes of RGB data using SSE shuffle.
#[cfg(all(feature = "unsafe_simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn extract_g_sse(rgb: __m128i) -> __m128i {
    // Shuffle mask: extract bytes 1, 4, 7, 10 (G values)
    let mask = _mm_setr_epi8(1, 4, 7, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    _mm_shuffle_epi8(rgb, mask)
}

/// Extract 4 B values from 16 bytes of RGB data using SSE shuffle.
#[cfg(all(feature = "unsafe_simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn extract_b_sse(rgb: __m128i) -> __m128i {
    // Shuffle mask: extract bytes 2, 5, 8, 11 (B values)
    let mask = _mm_setr_epi8(2, 5, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    _mm_shuffle_epi8(rgb, mask)
}

/// Convert 4 u8 values (in low bytes of __m128i) to __m128 f32.
#[cfg(all(feature = "unsafe_simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn u8x4_to_f32x4(v: __m128i) -> core::arch::x86_64::__m128 {
    use core::arch::x86_64::_mm_cvtepi32_ps;
    // Zero-extend u8 to i32, then convert to f32
    let i32_vec = _mm_cvtepu8_epi32(v);
    _mm_cvtepi32_ps(i32_vec)
}

/// AVX2 intrinsics implementation for RGB to YCbCr conversion.
///
/// Processes 8 pixels at a time using explicit SSE/AVX2+FMA intrinsics for
/// deinterleaving RGB data, which LLVM cannot auto-vectorize effectively.
/// Uses FMA (fused multiply-add) for better performance and precision.
///
/// This is a low-level function called by the safe `rgb_to_ycbcr_planes_simd_inplace` wrapper.
/// Production code should use the safe wrapper.
///
/// # Safety
/// Requires AVX2+FMA support. Caller must verify with `is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")`.
#[cfg(all(feature = "unsafe_simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub(crate) unsafe fn rgb_to_ycbcr_8px_avx2(
    rgb_ptr: *const u8,
    y_ptr: *mut f32,
    cb_ptr: *mut f32,
    cr_ptr: *mut f32,
) {
    // Intrinsics imported at module level

    // Load 24 bytes as two overlapping 16-byte loads
    // Load 0: bytes [0..15] for pixels 0-3 (plus partial pixel 4-5)
    // Load 1: bytes [12..27] for pixels 4-7 (overlaps by 4 bytes, but we only read 24)
    let rgb0 = _mm_loadu_si128(rgb_ptr as *const __m128i);
    let rgb1 = _mm_loadu_si128(rgb_ptr.add(12) as *const __m128i);

    // Extract R, G, B for first 4 pixels
    let r0_bytes = extract_r_sse(rgb0);
    let g0_bytes = extract_g_sse(rgb0);
    let b0_bytes = extract_b_sse(rgb0);

    // Extract R, G, B for second 4 pixels
    let r1_bytes = extract_r_sse(rgb1);
    let g1_bytes = extract_g_sse(rgb1);
    let b1_bytes = extract_b_sse(rgb1);

    // Convert to f32
    let r0: __m128 = u8x4_to_f32x4(r0_bytes);
    let g0: __m128 = u8x4_to_f32x4(g0_bytes);
    let b0: __m128 = u8x4_to_f32x4(b0_bytes);
    let r1: __m128 = u8x4_to_f32x4(r1_bytes);
    let g1: __m128 = u8x4_to_f32x4(g1_bytes);
    let b1: __m128 = u8x4_to_f32x4(b1_bytes);

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
    _mm_storeu_ps(y_ptr, y0);
    _mm_storeu_ps(y_ptr.add(4), y1);
    _mm_storeu_ps(cb_ptr, cb0);
    _mm_storeu_ps(cb_ptr.add(4), cb1);
    _mm_storeu_ps(cr_ptr, cr0);
    _mm_storeu_ps(cr_ptr.add(4), cr1);
}

/// Scalar reference implementation for RGB to YCbCr (for testing).
#[cfg(test)]
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

        y_plane[i] = YCBCR_R_TO_Y * r + YCBCR_G_TO_Y * g + YCBCR_B_TO_Y * b;
        cb_plane[i] = 128.0 + YCBCR_R_TO_CB * r + YCBCR_G_TO_CB * g + YCBCR_B_TO_CB * b;
        cr_plane[i] = 128.0 + YCBCR_R_TO_CR * r + YCBCR_G_TO_CR * g + YCBCR_B_TO_CR * b;
    }
}

// ============================================================================
// RGB to YCbCr Color Conversion
// ============================================================================

/// SIMD-optimized RGB to YCbCr conversion, writing to pre-allocated buffers.
///
/// This is the **zero-allocation** version for hot paths. Use this when encoding
/// multiple images or when performance is critical.
///
/// On x86_64 with AVX2+FMA, uses optimized intrinsics with shuffle-based RGB
/// deinterleaving and fused multiply-add operations.
///
/// # Arguments
/// * `rgb_data` - Input RGB data (3 bytes per pixel, interleaved)
/// * `y_plane` - Output Y plane (must be at least `num_pixels` elements)
/// * `cb_plane` - Output Cb plane (must be at least `num_pixels` elements)
/// * `cr_plane` - Output Cr plane (must be at least `num_pixels` elements)
/// * `num_pixels` - Number of pixels to process
#[inline]
pub fn rgb_to_ycbcr_planes_simd_inplace(
    rgb_data: &[u8],
    y_plane: &mut [f32],
    cb_plane: &mut [f32],
    cr_plane: &mut [f32],
    num_pixels: usize,
) {
    debug_assert!(rgb_data.len() >= num_pixels * 3);
    debug_assert!(y_plane.len() >= num_pixels);
    debug_assert!(cb_plane.len() >= num_pixels);
    debug_assert!(cr_plane.len() >= num_pixels);

    // Use AVX2+FMA intrinsics path when available (much faster due to shuffle-based
    // deinterleave instead of scalar gather, plus FMA operations)
    #[cfg(all(feature = "unsafe_simd", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            let chunks = num_pixels / 8;

            for chunk in 0..chunks {
                let pixel_idx = chunk * 8;
                let rgb_idx = pixel_idx * 3;

                // SAFETY: AVX2+FMA detected, and we've verified buffer lengths above
                unsafe {
                    let rgb_ptr = rgb_data.as_ptr().add(rgb_idx);
                    let y_ptr = y_plane.as_mut_ptr().add(pixel_idx);
                    let cb_ptr = cb_plane.as_mut_ptr().add(pixel_idx);
                    let cr_ptr = cr_plane.as_mut_ptr().add(pixel_idx);
                    rgb_to_ycbcr_8px_avx2(rgb_ptr, y_ptr, cb_ptr, cr_ptr);
                }
            }

            // Scalar remainder
            for i in (chunks * 8)..num_pixels {
                let rgb_idx = i * 3;
                let r = rgb_data[rgb_idx] as f32;
                let g = rgb_data[rgb_idx + 1] as f32;
                let b = rgb_data[rgb_idx + 2] as f32;

                y_plane[i] = YCBCR_R_TO_Y * r + YCBCR_G_TO_Y * g + YCBCR_B_TO_Y * b;
                cb_plane[i] = 128.0 + YCBCR_R_TO_CB * r + YCBCR_G_TO_CB * g + YCBCR_B_TO_CB * b;
                cr_plane[i] = 128.0 + YCBCR_R_TO_CR * r + YCBCR_G_TO_CR * g + YCBCR_B_TO_CR * b;
            }
            return;
        }
    }

    // Fallback path using wide crate's f32x8 (safe, portable SIMD)
    rgb_to_ycbcr_planes_simd_inplace_fallback(rgb_data, y_plane, cb_plane, cr_plane, num_pixels);
}

/// Fallback implementation using wide crate's f32x8 (for non-AVX2+FMA CPUs)
#[inline]
fn rgb_to_ycbcr_planes_simd_inplace_fallback(
    rgb_data: &[u8],
    y_plane: &mut [f32],
    cb_plane: &mut [f32],
    cr_plane: &mut [f32],
    num_pixels: usize,
) {
    // Coefficients as SIMD vectors
    let r_to_y = f32x8::splat(YCBCR_R_TO_Y);
    let g_to_y = f32x8::splat(YCBCR_G_TO_Y);
    let b_to_y = f32x8::splat(YCBCR_B_TO_Y);

    let r_to_cb = f32x8::splat(YCBCR_R_TO_CB);
    let g_to_cb = f32x8::splat(YCBCR_G_TO_CB);
    let b_to_cb = f32x8::splat(YCBCR_B_TO_CB);

    let r_to_cr = f32x8::splat(YCBCR_R_TO_CR);
    let g_to_cr = f32x8::splat(YCBCR_G_TO_CR);
    let b_to_cr = f32x8::splat(YCBCR_B_TO_CR);

    let offset_128 = f32x8::splat(128.0);

    let chunks = num_pixels / 8;

    for chunk in 0..chunks {
        let pixel_idx = chunk * 8;
        let rgb_idx = pixel_idx * 3;

        // Gather 8 R, G, B values from interleaved data
        let r = f32x8::from([
            rgb_data[rgb_idx] as f32,
            rgb_data[rgb_idx + 3] as f32,
            rgb_data[rgb_idx + 6] as f32,
            rgb_data[rgb_idx + 9] as f32,
            rgb_data[rgb_idx + 12] as f32,
            rgb_data[rgb_idx + 15] as f32,
            rgb_data[rgb_idx + 18] as f32,
            rgb_data[rgb_idx + 21] as f32,
        ]);

        let g = f32x8::from([
            rgb_data[rgb_idx + 1] as f32,
            rgb_data[rgb_idx + 4] as f32,
            rgb_data[rgb_idx + 7] as f32,
            rgb_data[rgb_idx + 10] as f32,
            rgb_data[rgb_idx + 13] as f32,
            rgb_data[rgb_idx + 16] as f32,
            rgb_data[rgb_idx + 19] as f32,
            rgb_data[rgb_idx + 22] as f32,
        ]);

        let b = f32x8::from([
            rgb_data[rgb_idx + 2] as f32,
            rgb_data[rgb_idx + 5] as f32,
            rgb_data[rgb_idx + 8] as f32,
            rgb_data[rgb_idx + 11] as f32,
            rgb_data[rgb_idx + 14] as f32,
            rgb_data[rgb_idx + 17] as f32,
            rgb_data[rgb_idx + 20] as f32,
            rgb_data[rgb_idx + 23] as f32,
        ]);

        // Compute Y, Cb, Cr
        let y = r * r_to_y + g * g_to_y + b * b_to_y;
        let cb = offset_128 + r * r_to_cb + g * g_to_cb + b * b_to_cb;
        let cr = offset_128 + r * r_to_cr + g * g_to_cr + b * b_to_cr;

        // Store results
        store_f32x8(y_plane, pixel_idx, y);
        store_f32x8(cb_plane, pixel_idx, cb);
        store_f32x8(cr_plane, pixel_idx, cr);
    }

    // Scalar remainder
    for i in (chunks * 8)..num_pixels {
        let rgb_idx = i * 3;
        let r = rgb_data[rgb_idx] as f32;
        let g = rgb_data[rgb_idx + 1] as f32;
        let b = rgb_data[rgb_idx + 2] as f32;

        y_plane[i] = YCBCR_R_TO_Y * r + YCBCR_G_TO_Y * g + YCBCR_B_TO_Y * b;
        cb_plane[i] = 128.0 + YCBCR_R_TO_CB * r + YCBCR_G_TO_CB * g + YCBCR_B_TO_CB * b;
        cr_plane[i] = 128.0 + YCBCR_R_TO_CR * r + YCBCR_G_TO_CR * g + YCBCR_B_TO_CR * b;
    }
}

/// SIMD-optimized RGB to YCbCr conversion for entire image.
///
/// Processes 8 pixels at a time, converting from interleaved RGB to planar YCbCr.
///
/// # Arguments
/// * `rgb_data` - Input RGB data (3 bytes per pixel, interleaved)
/// * `num_pixels` - Number of pixels
///
/// # Returns
/// Tuple of (Y plane, Cb plane, Cr plane) as f32 vectors, or error on allocation failure
pub fn rgb_to_ycbcr_planes_simd(
    rgb_data: &[u8],
    num_pixels: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    // Use uninit allocation since we write to every element
    let mut y_plane = try_alloc_f32(num_pixels, "rgb_to_ycbcr Y plane")?;
    let mut cb_plane = try_alloc_f32(num_pixels, "rgb_to_ycbcr Cb plane")?;
    let mut cr_plane = try_alloc_f32(num_pixels, "rgb_to_ycbcr Cr plane")?;

    rgb_to_ycbcr_planes_simd_inplace(
        rgb_data,
        &mut y_plane,
        &mut cb_plane,
        &mut cr_plane,
        num_pixels,
    );

    Ok((y_plane, cb_plane, cr_plane))
}

/// SIMD-optimized RGBA to YCbCr conversion, writing to pre-allocated buffers.
#[inline]
pub fn rgba_to_ycbcr_planes_simd_inplace(
    rgba_data: &[u8],
    y_plane: &mut [f32],
    cb_plane: &mut [f32],
    cr_plane: &mut [f32],
    num_pixels: usize,
) {
    debug_assert!(rgba_data.len() >= num_pixels * 4);
    debug_assert!(y_plane.len() >= num_pixels);
    debug_assert!(cb_plane.len() >= num_pixels);
    debug_assert!(cr_plane.len() >= num_pixels);

    let r_to_y = f32x8::splat(YCBCR_R_TO_Y);
    let g_to_y = f32x8::splat(YCBCR_G_TO_Y);
    let b_to_y = f32x8::splat(YCBCR_B_TO_Y);

    let r_to_cb = f32x8::splat(YCBCR_R_TO_CB);
    let g_to_cb = f32x8::splat(YCBCR_G_TO_CB);
    let b_to_cb = f32x8::splat(YCBCR_B_TO_CB);

    let r_to_cr = f32x8::splat(YCBCR_R_TO_CR);
    let g_to_cr = f32x8::splat(YCBCR_G_TO_CR);
    let b_to_cr = f32x8::splat(YCBCR_B_TO_CR);

    let offset_128 = f32x8::splat(128.0);
    let chunks = num_pixels / 8;

    for chunk in 0..chunks {
        let pixel_idx = chunk * 8;
        let rgba_idx = pixel_idx * 4;

        let r = f32x8::from([
            rgba_data[rgba_idx] as f32,
            rgba_data[rgba_idx + 4] as f32,
            rgba_data[rgba_idx + 8] as f32,
            rgba_data[rgba_idx + 12] as f32,
            rgba_data[rgba_idx + 16] as f32,
            rgba_data[rgba_idx + 20] as f32,
            rgba_data[rgba_idx + 24] as f32,
            rgba_data[rgba_idx + 28] as f32,
        ]);

        let g = f32x8::from([
            rgba_data[rgba_idx + 1] as f32,
            rgba_data[rgba_idx + 5] as f32,
            rgba_data[rgba_idx + 9] as f32,
            rgba_data[rgba_idx + 13] as f32,
            rgba_data[rgba_idx + 17] as f32,
            rgba_data[rgba_idx + 21] as f32,
            rgba_data[rgba_idx + 25] as f32,
            rgba_data[rgba_idx + 29] as f32,
        ]);

        let b = f32x8::from([
            rgba_data[rgba_idx + 2] as f32,
            rgba_data[rgba_idx + 6] as f32,
            rgba_data[rgba_idx + 10] as f32,
            rgba_data[rgba_idx + 14] as f32,
            rgba_data[rgba_idx + 18] as f32,
            rgba_data[rgba_idx + 22] as f32,
            rgba_data[rgba_idx + 26] as f32,
            rgba_data[rgba_idx + 30] as f32,
        ]);

        let y = r * r_to_y + g * g_to_y + b * b_to_y;
        let cb = offset_128 + r * r_to_cb + g * g_to_cb + b * b_to_cb;
        let cr = offset_128 + r * r_to_cr + g * g_to_cr + b * b_to_cr;

        store_f32x8(y_plane, pixel_idx, y);
        store_f32x8(cb_plane, pixel_idx, cb);
        store_f32x8(cr_plane, pixel_idx, cr);
    }

    for i in (chunks * 8)..num_pixels {
        let rgba_idx = i * 4;
        let r = rgba_data[rgba_idx] as f32;
        let g = rgba_data[rgba_idx + 1] as f32;
        let b = rgba_data[rgba_idx + 2] as f32;

        y_plane[i] = YCBCR_R_TO_Y * r + YCBCR_G_TO_Y * g + YCBCR_B_TO_Y * b;
        cb_plane[i] = 128.0 + YCBCR_R_TO_CB * r + YCBCR_G_TO_CB * g + YCBCR_B_TO_CB * b;
        cr_plane[i] = 128.0 + YCBCR_R_TO_CR * r + YCBCR_G_TO_CR * g + YCBCR_B_TO_CR * b;
    }
}

/// SIMD-optimized RGBA to YCbCr conversion for entire image (ignores alpha).
pub fn rgba_to_ycbcr_planes_simd(
    rgba_data: &[u8],
    num_pixels: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    // Use uninit allocation since we write to every element
    let mut y_plane = try_alloc_f32(num_pixels, "rgba_to_ycbcr Y plane")?;
    let mut cb_plane = try_alloc_f32(num_pixels, "rgba_to_ycbcr Cb plane")?;
    let mut cr_plane = try_alloc_f32(num_pixels, "rgba_to_ycbcr Cr plane")?;

    let r_to_y = f32x8::splat(YCBCR_R_TO_Y);
    let g_to_y = f32x8::splat(YCBCR_G_TO_Y);
    let b_to_y = f32x8::splat(YCBCR_B_TO_Y);

    let r_to_cb = f32x8::splat(YCBCR_R_TO_CB);
    let g_to_cb = f32x8::splat(YCBCR_G_TO_CB);
    let b_to_cb = f32x8::splat(YCBCR_B_TO_CB);

    let r_to_cr = f32x8::splat(YCBCR_R_TO_CR);
    let g_to_cr = f32x8::splat(YCBCR_G_TO_CR);
    let b_to_cr = f32x8::splat(YCBCR_B_TO_CR);

    let offset_128 = f32x8::splat(128.0);

    let chunks = num_pixels / 8;

    for chunk in 0..chunks {
        let pixel_idx = chunk * 8;
        let rgba_idx = pixel_idx * 4;

        // Gather 8 R, G, B values from interleaved RGBA data (stride 4)
        let r = f32x8::from([
            rgba_data[rgba_idx] as f32,
            rgba_data[rgba_idx + 4] as f32,
            rgba_data[rgba_idx + 8] as f32,
            rgba_data[rgba_idx + 12] as f32,
            rgba_data[rgba_idx + 16] as f32,
            rgba_data[rgba_idx + 20] as f32,
            rgba_data[rgba_idx + 24] as f32,
            rgba_data[rgba_idx + 28] as f32,
        ]);

        let g = f32x8::from([
            rgba_data[rgba_idx + 1] as f32,
            rgba_data[rgba_idx + 5] as f32,
            rgba_data[rgba_idx + 9] as f32,
            rgba_data[rgba_idx + 13] as f32,
            rgba_data[rgba_idx + 17] as f32,
            rgba_data[rgba_idx + 21] as f32,
            rgba_data[rgba_idx + 25] as f32,
            rgba_data[rgba_idx + 29] as f32,
        ]);

        let b = f32x8::from([
            rgba_data[rgba_idx + 2] as f32,
            rgba_data[rgba_idx + 6] as f32,
            rgba_data[rgba_idx + 10] as f32,
            rgba_data[rgba_idx + 14] as f32,
            rgba_data[rgba_idx + 18] as f32,
            rgba_data[rgba_idx + 22] as f32,
            rgba_data[rgba_idx + 26] as f32,
            rgba_data[rgba_idx + 30] as f32,
        ]);

        let y = r * r_to_y + g * g_to_y + b * b_to_y;
        let cb = offset_128 + r * r_to_cb + g * g_to_cb + b * b_to_cb;
        let cr = offset_128 + r * r_to_cr + g * g_to_cr + b * b_to_cr;

        let y_arr: [f32; 8] = y.into();
        let cb_arr: [f32; 8] = cb.into();
        let cr_arr: [f32; 8] = cr.into();

        y_plane[pixel_idx..pixel_idx + 8].copy_from_slice(&y_arr);
        cb_plane[pixel_idx..pixel_idx + 8].copy_from_slice(&cb_arr);
        cr_plane[pixel_idx..pixel_idx + 8].copy_from_slice(&cr_arr);
    }

    // Scalar remainder
    for i in (chunks * 8)..num_pixels {
        let rgba_idx = i * 4;
        let r = rgba_data[rgba_idx] as f32;
        let g = rgba_data[rgba_idx + 1] as f32;
        let b = rgba_data[rgba_idx + 2] as f32;

        y_plane[i] = YCBCR_R_TO_Y * r + YCBCR_G_TO_Y * g + YCBCR_B_TO_Y * b;
        cb_plane[i] = 128.0 + YCBCR_R_TO_CB * r + YCBCR_G_TO_CB * g + YCBCR_B_TO_CB * b;
        cr_plane[i] = 128.0 + YCBCR_R_TO_CR * r + YCBCR_G_TO_CR * g + YCBCR_B_TO_CR * b;
    }

    Ok((y_plane, cb_plane, cr_plane))
}

/// SIMD-optimized grayscale to YCbCr conversion, writing to pre-allocated buffers.
#[inline]
pub fn gray_to_ycbcr_planes_simd_inplace(
    gray_data: &[u8],
    y_plane: &mut [f32],
    cb_plane: &mut [f32],
    cr_plane: &mut [f32],
    num_pixels: usize,
) {
    debug_assert!(gray_data.len() >= num_pixels);
    debug_assert!(y_plane.len() >= num_pixels);
    debug_assert!(cb_plane.len() >= num_pixels);
    debug_assert!(cr_plane.len() >= num_pixels);

    let offset_128 = f32x8::splat(128.0);
    let chunks = num_pixels / 8;

    for chunk in 0..chunks {
        let idx = chunk * 8;

        let y = f32x8::from([
            gray_data[idx] as f32,
            gray_data[idx + 1] as f32,
            gray_data[idx + 2] as f32,
            gray_data[idx + 3] as f32,
            gray_data[idx + 4] as f32,
            gray_data[idx + 5] as f32,
            gray_data[idx + 6] as f32,
            gray_data[idx + 7] as f32,
        ]);

        store_f32x8(y_plane, idx, y);
        store_f32x8(cb_plane, idx, offset_128);
        store_f32x8(cr_plane, idx, offset_128);
    }

    for i in (chunks * 8)..num_pixels {
        y_plane[i] = gray_data[i] as f32;
        cb_plane[i] = 128.0;
        cr_plane[i] = 128.0;
    }
}

/// SIMD-optimized grayscale to YCbCr conversion.
/// Y = gray value, Cb = Cr = 128.0
pub fn gray_to_ycbcr_planes_simd(
    gray_data: &[u8],
    num_pixels: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let mut y_plane = try_alloc_f32(num_pixels, "gray_to_ycbcr Y plane")?;
    let mut cb_plane = try_alloc_f32(num_pixels, "gray_to_ycbcr Cb plane")?;
    let mut cr_plane = try_alloc_f32(num_pixels, "gray_to_ycbcr Cr plane")?;

    gray_to_ycbcr_planes_simd_inplace(
        gray_data,
        &mut y_plane,
        &mut cb_plane,
        &mut cr_plane,
        num_pixels,
    );

    Ok((y_plane, cb_plane, cr_plane))
}

/// SIMD-optimized BGR to YCbCr conversion, writing to pre-allocated buffers.
#[inline]
pub fn bgr_to_ycbcr_planes_simd_inplace(
    bgr_data: &[u8],
    y_plane: &mut [f32],
    cb_plane: &mut [f32],
    cr_plane: &mut [f32],
    num_pixels: usize,
) {
    debug_assert!(bgr_data.len() >= num_pixels * 3);
    debug_assert!(y_plane.len() >= num_pixels);
    debug_assert!(cb_plane.len() >= num_pixels);
    debug_assert!(cr_plane.len() >= num_pixels);

    let r_to_y = f32x8::splat(YCBCR_R_TO_Y);
    let g_to_y = f32x8::splat(YCBCR_G_TO_Y);
    let b_to_y = f32x8::splat(YCBCR_B_TO_Y);

    let r_to_cb = f32x8::splat(YCBCR_R_TO_CB);
    let g_to_cb = f32x8::splat(YCBCR_G_TO_CB);
    let b_to_cb = f32x8::splat(YCBCR_B_TO_CB);

    let r_to_cr = f32x8::splat(YCBCR_R_TO_CR);
    let g_to_cr = f32x8::splat(YCBCR_G_TO_CR);
    let b_to_cr = f32x8::splat(YCBCR_B_TO_CR);

    let offset_128 = f32x8::splat(128.0);
    let chunks = num_pixels / 8;

    for chunk in 0..chunks {
        let pixel_idx = chunk * 8;
        let bgr_idx = pixel_idx * 3;

        let b = f32x8::from([
            bgr_data[bgr_idx] as f32,
            bgr_data[bgr_idx + 3] as f32,
            bgr_data[bgr_idx + 6] as f32,
            bgr_data[bgr_idx + 9] as f32,
            bgr_data[bgr_idx + 12] as f32,
            bgr_data[bgr_idx + 15] as f32,
            bgr_data[bgr_idx + 18] as f32,
            bgr_data[bgr_idx + 21] as f32,
        ]);

        let g = f32x8::from([
            bgr_data[bgr_idx + 1] as f32,
            bgr_data[bgr_idx + 4] as f32,
            bgr_data[bgr_idx + 7] as f32,
            bgr_data[bgr_idx + 10] as f32,
            bgr_data[bgr_idx + 13] as f32,
            bgr_data[bgr_idx + 16] as f32,
            bgr_data[bgr_idx + 19] as f32,
            bgr_data[bgr_idx + 22] as f32,
        ]);

        let r = f32x8::from([
            bgr_data[bgr_idx + 2] as f32,
            bgr_data[bgr_idx + 5] as f32,
            bgr_data[bgr_idx + 8] as f32,
            bgr_data[bgr_idx + 11] as f32,
            bgr_data[bgr_idx + 14] as f32,
            bgr_data[bgr_idx + 17] as f32,
            bgr_data[bgr_idx + 20] as f32,
            bgr_data[bgr_idx + 23] as f32,
        ]);

        let y = r * r_to_y + g * g_to_y + b * b_to_y;
        let cb = offset_128 + r * r_to_cb + g * g_to_cb + b * b_to_cb;
        let cr = offset_128 + r * r_to_cr + g * g_to_cr + b * b_to_cr;

        store_f32x8(y_plane, pixel_idx, y);
        store_f32x8(cb_plane, pixel_idx, cb);
        store_f32x8(cr_plane, pixel_idx, cr);
    }

    for i in (chunks * 8)..num_pixels {
        let bgr_idx = i * 3;
        let b = bgr_data[bgr_idx] as f32;
        let g = bgr_data[bgr_idx + 1] as f32;
        let r = bgr_data[bgr_idx + 2] as f32;

        y_plane[i] = YCBCR_R_TO_Y * r + YCBCR_G_TO_Y * g + YCBCR_B_TO_Y * b;
        cb_plane[i] = 128.0 + YCBCR_R_TO_CB * r + YCBCR_G_TO_CB * g + YCBCR_B_TO_CB * b;
        cr_plane[i] = 128.0 + YCBCR_R_TO_CR * r + YCBCR_G_TO_CR * g + YCBCR_B_TO_CR * b;
    }
}

/// SIMD-optimized BGR to YCbCr conversion for entire image.
pub fn bgr_to_ycbcr_planes_simd(
    bgr_data: &[u8],
    num_pixels: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let mut y_plane = try_alloc_f32(num_pixels, "bgr_to_ycbcr Y plane")?;
    let mut cb_plane = try_alloc_f32(num_pixels, "bgr_to_ycbcr Cb plane")?;
    let mut cr_plane = try_alloc_f32(num_pixels, "bgr_to_ycbcr Cr plane")?;

    bgr_to_ycbcr_planes_simd_inplace(
        bgr_data,
        &mut y_plane,
        &mut cb_plane,
        &mut cr_plane,
        num_pixels,
    );

    Ok((y_plane, cb_plane, cr_plane))
}

/// SIMD-optimized BGRA to YCbCr conversion, writing to pre-allocated buffers.
#[inline]
pub fn bgra_to_ycbcr_planes_simd_inplace(
    bgra_data: &[u8],
    y_plane: &mut [f32],
    cb_plane: &mut [f32],
    cr_plane: &mut [f32],
    num_pixels: usize,
) {
    debug_assert!(bgra_data.len() >= num_pixels * 4);
    debug_assert!(y_plane.len() >= num_pixels);
    debug_assert!(cb_plane.len() >= num_pixels);
    debug_assert!(cr_plane.len() >= num_pixels);

    let r_to_y = f32x8::splat(YCBCR_R_TO_Y);
    let g_to_y = f32x8::splat(YCBCR_G_TO_Y);
    let b_to_y = f32x8::splat(YCBCR_B_TO_Y);

    let r_to_cb = f32x8::splat(YCBCR_R_TO_CB);
    let g_to_cb = f32x8::splat(YCBCR_G_TO_CB);
    let b_to_cb = f32x8::splat(YCBCR_B_TO_CB);

    let r_to_cr = f32x8::splat(YCBCR_R_TO_CR);
    let g_to_cr = f32x8::splat(YCBCR_G_TO_CR);
    let b_to_cr = f32x8::splat(YCBCR_B_TO_CR);

    let offset_128 = f32x8::splat(128.0);
    let chunks = num_pixels / 8;

    for chunk in 0..chunks {
        let pixel_idx = chunk * 8;
        let bgra_idx = pixel_idx * 4;

        let b = f32x8::from([
            bgra_data[bgra_idx] as f32,
            bgra_data[bgra_idx + 4] as f32,
            bgra_data[bgra_idx + 8] as f32,
            bgra_data[bgra_idx + 12] as f32,
            bgra_data[bgra_idx + 16] as f32,
            bgra_data[bgra_idx + 20] as f32,
            bgra_data[bgra_idx + 24] as f32,
            bgra_data[bgra_idx + 28] as f32,
        ]);

        let g = f32x8::from([
            bgra_data[bgra_idx + 1] as f32,
            bgra_data[bgra_idx + 5] as f32,
            bgra_data[bgra_idx + 9] as f32,
            bgra_data[bgra_idx + 13] as f32,
            bgra_data[bgra_idx + 17] as f32,
            bgra_data[bgra_idx + 21] as f32,
            bgra_data[bgra_idx + 25] as f32,
            bgra_data[bgra_idx + 29] as f32,
        ]);

        let r = f32x8::from([
            bgra_data[bgra_idx + 2] as f32,
            bgra_data[bgra_idx + 6] as f32,
            bgra_data[bgra_idx + 10] as f32,
            bgra_data[bgra_idx + 14] as f32,
            bgra_data[bgra_idx + 18] as f32,
            bgra_data[bgra_idx + 22] as f32,
            bgra_data[bgra_idx + 26] as f32,
            bgra_data[bgra_idx + 30] as f32,
        ]);

        let y = r * r_to_y + g * g_to_y + b * b_to_y;
        let cb = offset_128 + r * r_to_cb + g * g_to_cb + b * b_to_cb;
        let cr = offset_128 + r * r_to_cr + g * g_to_cr + b * b_to_cr;

        store_f32x8(y_plane, pixel_idx, y);
        store_f32x8(cb_plane, pixel_idx, cb);
        store_f32x8(cr_plane, pixel_idx, cr);
    }

    for i in (chunks * 8)..num_pixels {
        let bgra_idx = i * 4;
        let b = bgra_data[bgra_idx] as f32;
        let g = bgra_data[bgra_idx + 1] as f32;
        let r = bgra_data[bgra_idx + 2] as f32;

        y_plane[i] = YCBCR_R_TO_Y * r + YCBCR_G_TO_Y * g + YCBCR_B_TO_Y * b;
        cb_plane[i] = 128.0 + YCBCR_R_TO_CB * r + YCBCR_G_TO_CB * g + YCBCR_B_TO_CB * b;
        cr_plane[i] = 128.0 + YCBCR_R_TO_CR * r + YCBCR_G_TO_CR * g + YCBCR_B_TO_CR * b;
    }
}

/// SIMD-optimized BGRA to YCbCr conversion for entire image (ignores alpha).
pub fn bgra_to_ycbcr_planes_simd(
    bgra_data: &[u8],
    num_pixels: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let mut y_plane = try_alloc_f32(num_pixels, "bgra_to_ycbcr Y plane")?;
    let mut cb_plane = try_alloc_f32(num_pixels, "bgra_to_ycbcr Cb plane")?;
    let mut cr_plane = try_alloc_f32(num_pixels, "bgra_to_ycbcr Cr plane")?;

    bgra_to_ycbcr_planes_simd_inplace(
        bgra_data,
        &mut y_plane,
        &mut cb_plane,
        &mut cr_plane,
        num_pixels,
    );

    Ok((y_plane, cb_plane, cr_plane))
}

// ============================================================================
// Strided RGB→YCbCr Conversion (for strip encoder)
// ============================================================================

/// SIMD-optimized RGB to YCbCr conversion with strided Y output.
///
/// Writes Y plane with `y_stride` stride (for 8-aligned block extraction),
/// while Cb/Cr use packed stride (width). This eliminates the need for
/// a separate rearrange pass when Y needs padding.
///
/// # Arguments
/// * `rgb_data` - Input RGB data (3 bytes per pixel, interleaved)
/// * `y_plane` - Output Y plane (y_stride × height elements)
/// * `cb_plane` - Output Cb plane (width × height elements)
/// * `cr_plane` - Output Cr plane (width × height elements)
/// * `width` - Image width in pixels
/// * `height` - Number of rows to process
/// * `y_stride` - Y output stride (typically padded_width)
/// * `bpp` - Bytes per pixel (3 for RGB)
#[inline]
pub fn rgb_to_ycbcr_strided_inplace(
    rgb_data: &[u8],
    y_plane: &mut [f32],
    cb_plane: &mut [f32],
    cr_plane: &mut [f32],
    width: usize,
    height: usize,
    y_stride: usize,
    bpp: usize,
) {
    debug_assert!(rgb_data.len() >= width * height * bpp);
    debug_assert!(y_plane.len() >= y_stride * height);
    debug_assert!(cb_plane.len() >= width * height);
    debug_assert!(cr_plane.len() >= width * height);

    // Fast path: if Y stride matches width, use contiguous conversion
    if y_stride == width {
        let num_pixels = width * height;
        match bpp {
            3 => rgb_to_ycbcr_planes_simd_inplace(rgb_data, y_plane, cb_plane, cr_plane, num_pixels),
            4 => rgba_to_ycbcr_planes_simd_inplace(rgb_data, y_plane, cb_plane, cr_plane, num_pixels),
            _ => return, // Unsupported
        }
        return;
    }

    // Strided path: process row-by-row
    let r_to_y = f32x8::splat(YCBCR_R_TO_Y);
    let g_to_y = f32x8::splat(YCBCR_G_TO_Y);
    let b_to_y = f32x8::splat(YCBCR_B_TO_Y);
    let r_to_cb = f32x8::splat(YCBCR_R_TO_CB);
    let g_to_cb = f32x8::splat(YCBCR_G_TO_CB);
    let b_to_cb = f32x8::splat(YCBCR_B_TO_CB);
    let r_to_cr = f32x8::splat(YCBCR_R_TO_CR);
    let g_to_cr = f32x8::splat(YCBCR_G_TO_CR);
    let b_to_cr = f32x8::splat(YCBCR_B_TO_CR);
    let offset_128 = f32x8::splat(128.0);

    for row in 0..height {
        let rgb_row_start = row * width * bpp;
        let y_row_start = row * y_stride;
        let cbcr_row_start = row * width;

        let chunks = width / 8;

        // SIMD loop for 8-pixel chunks
        for chunk in 0..chunks {
            let px = chunk * 8;
            let rgb_idx = rgb_row_start + px * bpp;

            // Gather RGB (bpp=3 or 4)
            let (r, g, b) = if bpp == 3 {
                (
                    f32x8::from([
                        rgb_data[rgb_idx] as f32,
                        rgb_data[rgb_idx + 3] as f32,
                        rgb_data[rgb_idx + 6] as f32,
                        rgb_data[rgb_idx + 9] as f32,
                        rgb_data[rgb_idx + 12] as f32,
                        rgb_data[rgb_idx + 15] as f32,
                        rgb_data[rgb_idx + 18] as f32,
                        rgb_data[rgb_idx + 21] as f32,
                    ]),
                    f32x8::from([
                        rgb_data[rgb_idx + 1] as f32,
                        rgb_data[rgb_idx + 4] as f32,
                        rgb_data[rgb_idx + 7] as f32,
                        rgb_data[rgb_idx + 10] as f32,
                        rgb_data[rgb_idx + 13] as f32,
                        rgb_data[rgb_idx + 16] as f32,
                        rgb_data[rgb_idx + 19] as f32,
                        rgb_data[rgb_idx + 22] as f32,
                    ]),
                    f32x8::from([
                        rgb_data[rgb_idx + 2] as f32,
                        rgb_data[rgb_idx + 5] as f32,
                        rgb_data[rgb_idx + 8] as f32,
                        rgb_data[rgb_idx + 11] as f32,
                        rgb_data[rgb_idx + 14] as f32,
                        rgb_data[rgb_idx + 17] as f32,
                        rgb_data[rgb_idx + 20] as f32,
                        rgb_data[rgb_idx + 23] as f32,
                    ]),
                )
            } else {
                // bpp == 4 (RGBA)
                (
                    f32x8::from([
                        rgb_data[rgb_idx] as f32,
                        rgb_data[rgb_idx + 4] as f32,
                        rgb_data[rgb_idx + 8] as f32,
                        rgb_data[rgb_idx + 12] as f32,
                        rgb_data[rgb_idx + 16] as f32,
                        rgb_data[rgb_idx + 20] as f32,
                        rgb_data[rgb_idx + 24] as f32,
                        rgb_data[rgb_idx + 28] as f32,
                    ]),
                    f32x8::from([
                        rgb_data[rgb_idx + 1] as f32,
                        rgb_data[rgb_idx + 5] as f32,
                        rgb_data[rgb_idx + 9] as f32,
                        rgb_data[rgb_idx + 13] as f32,
                        rgb_data[rgb_idx + 17] as f32,
                        rgb_data[rgb_idx + 21] as f32,
                        rgb_data[rgb_idx + 25] as f32,
                        rgb_data[rgb_idx + 29] as f32,
                    ]),
                    f32x8::from([
                        rgb_data[rgb_idx + 2] as f32,
                        rgb_data[rgb_idx + 6] as f32,
                        rgb_data[rgb_idx + 10] as f32,
                        rgb_data[rgb_idx + 14] as f32,
                        rgb_data[rgb_idx + 18] as f32,
                        rgb_data[rgb_idx + 22] as f32,
                        rgb_data[rgb_idx + 26] as f32,
                        rgb_data[rgb_idx + 30] as f32,
                    ]),
                )
            };

            let y = r * r_to_y + g * g_to_y + b * b_to_y;
            let cb = offset_128 + r * r_to_cb + g * g_to_cb + b * b_to_cb;
            let cr = offset_128 + r * r_to_cr + g * g_to_cr + b * b_to_cr;

            // Write Y with strided offset, Cb/Cr with packed offset
            store_f32x8(y_plane, y_row_start + px, y);
            store_f32x8(cb_plane, cbcr_row_start + px, cb);
            store_f32x8(cr_plane, cbcr_row_start + px, cr);
        }

        // Scalar remainder for this row
        for px in (chunks * 8)..width {
            let rgb_idx = rgb_row_start + px * bpp;
            let r = rgb_data[rgb_idx] as f32;
            let g = rgb_data[rgb_idx + 1] as f32;
            let b = rgb_data[rgb_idx + 2] as f32;

            y_plane[y_row_start + px] = YCBCR_R_TO_Y * r + YCBCR_G_TO_Y * g + YCBCR_B_TO_Y * b;
            cb_plane[cbcr_row_start + px] = 128.0 + YCBCR_R_TO_CB * r + YCBCR_G_TO_CB * g + YCBCR_B_TO_CB * b;
            cr_plane[cbcr_row_start + px] = 128.0 + YCBCR_R_TO_CR * r + YCBCR_G_TO_CR * g + YCBCR_B_TO_CR * b;
        }

        // Edge-pad Y row to stride
        if width < y_stride {
            let edge_val = y_plane[y_row_start + width - 1];
            for px in width..y_stride {
                y_plane[y_row_start + px] = edge_val;
            }
        }
    }
}

/// BGR variant of strided conversion (for BGR/BGRA input).
#[inline]
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

    // Fast path: if Y stride matches width, use contiguous conversion
    if y_stride == width {
        let num_pixels = width * height;
        match bpp {
            3 => bgr_to_ycbcr_planes_simd_inplace(bgr_data, y_plane, cb_plane, cr_plane, num_pixels),
            4 => bgra_to_ycbcr_planes_simd_inplace(bgr_data, y_plane, cb_plane, cr_plane, num_pixels),
            _ => return,
        }
        return;
    }

    // Strided path: process row-by-row (swap R/B channels)
    let r_to_y = f32x8::splat(YCBCR_R_TO_Y);
    let g_to_y = f32x8::splat(YCBCR_G_TO_Y);
    let b_to_y = f32x8::splat(YCBCR_B_TO_Y);
    let r_to_cb = f32x8::splat(YCBCR_R_TO_CB);
    let g_to_cb = f32x8::splat(YCBCR_G_TO_CB);
    let b_to_cb = f32x8::splat(YCBCR_B_TO_CB);
    let r_to_cr = f32x8::splat(YCBCR_R_TO_CR);
    let g_to_cr = f32x8::splat(YCBCR_G_TO_CR);
    let b_to_cr = f32x8::splat(YCBCR_B_TO_CR);
    let offset_128 = f32x8::splat(128.0);

    for row in 0..height {
        let bgr_row_start = row * width * bpp;
        let y_row_start = row * y_stride;
        let cbcr_row_start = row * width;

        let chunks = width / 8;

        for chunk in 0..chunks {
            let px = chunk * 8;
            let bgr_idx = bgr_row_start + px * bpp;

            // Gather BGR (channels swapped vs RGB)
            let (r, g, b) = if bpp == 3 {
                (
                    f32x8::from([
                        bgr_data[bgr_idx + 2] as f32,
                        bgr_data[bgr_idx + 5] as f32,
                        bgr_data[bgr_idx + 8] as f32,
                        bgr_data[bgr_idx + 11] as f32,
                        bgr_data[bgr_idx + 14] as f32,
                        bgr_data[bgr_idx + 17] as f32,
                        bgr_data[bgr_idx + 20] as f32,
                        bgr_data[bgr_idx + 23] as f32,
                    ]),
                    f32x8::from([
                        bgr_data[bgr_idx + 1] as f32,
                        bgr_data[bgr_idx + 4] as f32,
                        bgr_data[bgr_idx + 7] as f32,
                        bgr_data[bgr_idx + 10] as f32,
                        bgr_data[bgr_idx + 13] as f32,
                        bgr_data[bgr_idx + 16] as f32,
                        bgr_data[bgr_idx + 19] as f32,
                        bgr_data[bgr_idx + 22] as f32,
                    ]),
                    f32x8::from([
                        bgr_data[bgr_idx] as f32,
                        bgr_data[bgr_idx + 3] as f32,
                        bgr_data[bgr_idx + 6] as f32,
                        bgr_data[bgr_idx + 9] as f32,
                        bgr_data[bgr_idx + 12] as f32,
                        bgr_data[bgr_idx + 15] as f32,
                        bgr_data[bgr_idx + 18] as f32,
                        bgr_data[bgr_idx + 21] as f32,
                    ]),
                )
            } else {
                // bpp == 4 (BGRA)
                (
                    f32x8::from([
                        bgr_data[bgr_idx + 2] as f32,
                        bgr_data[bgr_idx + 6] as f32,
                        bgr_data[bgr_idx + 10] as f32,
                        bgr_data[bgr_idx + 14] as f32,
                        bgr_data[bgr_idx + 18] as f32,
                        bgr_data[bgr_idx + 22] as f32,
                        bgr_data[bgr_idx + 26] as f32,
                        bgr_data[bgr_idx + 30] as f32,
                    ]),
                    f32x8::from([
                        bgr_data[bgr_idx + 1] as f32,
                        bgr_data[bgr_idx + 5] as f32,
                        bgr_data[bgr_idx + 9] as f32,
                        bgr_data[bgr_idx + 13] as f32,
                        bgr_data[bgr_idx + 17] as f32,
                        bgr_data[bgr_idx + 21] as f32,
                        bgr_data[bgr_idx + 25] as f32,
                        bgr_data[bgr_idx + 29] as f32,
                    ]),
                    f32x8::from([
                        bgr_data[bgr_idx] as f32,
                        bgr_data[bgr_idx + 4] as f32,
                        bgr_data[bgr_idx + 8] as f32,
                        bgr_data[bgr_idx + 12] as f32,
                        bgr_data[bgr_idx + 16] as f32,
                        bgr_data[bgr_idx + 20] as f32,
                        bgr_data[bgr_idx + 24] as f32,
                        bgr_data[bgr_idx + 28] as f32,
                    ]),
                )
            };

            let y = r * r_to_y + g * g_to_y + b * b_to_y;
            let cb = offset_128 + r * r_to_cb + g * g_to_cb + b * b_to_cb;
            let cr = offset_128 + r * r_to_cr + g * g_to_cr + b * b_to_cr;

            store_f32x8(y_plane, y_row_start + px, y);
            store_f32x8(cb_plane, cbcr_row_start + px, cb);
            store_f32x8(cr_plane, cbcr_row_start + px, cr);
        }

        for px in (chunks * 8)..width {
            let bgr_idx = bgr_row_start + px * bpp;
            let b = bgr_data[bgr_idx] as f32;
            let g = bgr_data[bgr_idx + 1] as f32;
            let r = bgr_data[bgr_idx + 2] as f32;

            y_plane[y_row_start + px] = YCBCR_R_TO_Y * r + YCBCR_G_TO_Y * g + YCBCR_B_TO_Y * b;
            cb_plane[cbcr_row_start + px] = 128.0 + YCBCR_R_TO_CB * r + YCBCR_G_TO_CB * g + YCBCR_B_TO_CB * b;
            cr_plane[cbcr_row_start + px] = 128.0 + YCBCR_R_TO_CR * r + YCBCR_G_TO_CR * g + YCBCR_B_TO_CR * b;
        }

        if width < y_stride {
            let edge_val = y_plane[y_row_start + width - 1];
            for px in width..y_stride {
                y_plane[y_row_start + px] = edge_val;
            }
        }
    }
}

// ============================================================================
// Block Extraction
// ============================================================================

/// SIMD-optimized block extraction from a plane with level shift.
///
/// Extracts an 8x8 block from the given position, subtracting 128.0 (level shift).
/// Uses fast path for interior blocks (no bounds checking needed).
///
/// # Arguments
/// * `plane` - Input plane data
/// * `width` - Plane width
/// * `height` - Plane height
/// * `bx` - Block x coordinate (in blocks, not pixels)
/// * `by` - Block y coordinate (in blocks, not pixels)
#[inline]
pub fn extract_block_simd(
    plane: &[f32],
    width: usize,
    height: usize,
    bx: usize,
    by: usize,
) -> [f32; 64] {
    let px_start = bx * 8;
    let py_start = by * 8;

    // Check if block is entirely within bounds
    let is_interior = px_start + 8 <= width && py_start + 8 <= height;

    let level_shift = f32x8::splat(128.0);
    let mut block = [0.0f32; 64];

    if is_interior {
        // Fast path: no bounds checking needed
        for y in 0..8 {
            let row_start = (py_start + y) * width + px_start;

            // Load 8 consecutive f32 values (zero-cost from contiguous memory)
            let row_arr: [f32; 8] = plane[row_start..row_start + 8].try_into().unwrap();
            let row = f32x8::from(row_arr);

            // Subtract level shift
            let shifted = row - level_shift;

            // Store to block
            let arr: [f32; 8] = shifted.into();
            block[y * 8..y * 8 + 8].copy_from_slice(&arr);
        }
    } else {
        // Edge case: need bounds checking
        for y in 0..8 {
            let py = (py_start + y).min(height - 1);
            for x in 0..8 {
                let px = (px_start + x).min(width - 1);
                block[y * 8 + x] = plane[py * width + px] - 128.0;
            }
        }
    }

    block
}

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
#[inline]
pub fn extract_block_xyb_simd(
    plane: &[f32],
    width: usize,
    height: usize,
    bx: usize,
    by: usize,
) -> [f32; 64] {
    let px_start = bx * 8;
    let py_start = by * 8;

    let is_interior = px_start + 8 <= width && py_start + 8 <= height;

    let scale = f32x8::splat(255.0);
    let level_shift = f32x8::splat(128.0);
    let mut block = [0.0f32; 64];

    if is_interior {
        for y in 0..8 {
            let row_start = (py_start + y) * width + px_start;
            // Load 8 consecutive f32 values (zero-cost from contiguous memory)
            let row_arr: [f32; 8] = plane[row_start..row_start + 8].try_into().unwrap();
            let row = f32x8::from(row_arr);

            // XYB: val * 255.0 - 128.0
            let scaled = row * scale - level_shift;

            let arr: [f32; 8] = scaled.into();
            block[y * 8..y * 8 + 8].copy_from_slice(&arr);
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
// XYB/Level Shift Conversion Helpers (for decoder)
// ============================================================================

/// SIMD-optimized XYB plane level shift to interleaved RGB u8.
///
/// Converts 3 XYB f32 planes to interleaved RGB u8, applying:
/// - Level shift (+128)
/// - Clamp to [0, 255]
/// - Convert to u8
///
/// This is used for XYB decode path where no YCbCr→RGB conversion is needed.
#[inline]
pub fn xyb_planes_to_rgb_u8_simd(plane0: &[f32], plane1: &[f32], plane2: &[f32], rgb: &mut [u8]) {
    debug_assert_eq!(plane0.len(), plane1.len());
    debug_assert_eq!(plane0.len(), plane2.len());
    debug_assert_eq!(rgb.len(), plane0.len() * 3);

    let num_pixels = plane0.len();
    let offset = f32x8::splat(128.0);
    let zero = f32x8::splat(0.0);
    let max_val = f32x8::splat(255.0);

    let chunks = num_pixels / 8;
    for chunk in 0..chunks {
        let base = chunk * 8;

        // Load 8 values from each plane
        let p0 = f32x8::from([
            plane0[base],
            plane0[base + 1],
            plane0[base + 2],
            plane0[base + 3],
            plane0[base + 4],
            plane0[base + 5],
            plane0[base + 6],
            plane0[base + 7],
        ]);
        let p1 = f32x8::from([
            plane1[base],
            plane1[base + 1],
            plane1[base + 2],
            plane1[base + 3],
            plane1[base + 4],
            plane1[base + 5],
            plane1[base + 6],
            plane1[base + 7],
        ]);
        let p2 = f32x8::from([
            plane2[base],
            plane2[base + 1],
            plane2[base + 2],
            plane2[base + 3],
            plane2[base + 4],
            plane2[base + 5],
            plane2[base + 6],
            plane2[base + 7],
        ]);

        // Level shift, clamp
        let r = (p0 + offset).max(zero).min(max_val);
        let g = (p1 + offset).max(zero).min(max_val);
        let b = (p2 + offset).max(zero).min(max_val);

        let r_arr: [f32; 8] = r.into();
        let g_arr: [f32; 8] = g.into();
        let b_arr: [f32; 8] = b.into();

        // Store interleaved RGB
        for j in 0..8 {
            let idx = (base + j) * 3;
            rgb[idx] = r_arr[j] as u8;
            rgb[idx + 1] = g_arr[j] as u8;
            rgb[idx + 2] = b_arr[j] as u8;
        }
    }

    // Scalar remainder
    for i in (chunks * 8)..num_pixels {
        let idx = i * 3;
        rgb[idx] = (plane0[i] + 128.0).clamp(0.0, 255.0) as u8;
        rgb[idx + 1] = (plane1[i] + 128.0).clamp(0.0, 255.0) as u8;
        rgb[idx + 2] = (plane2[i] + 128.0).clamp(0.0, 255.0) as u8;
    }
}

/// SIMD-optimized XYB plane level shift to interleaved RGB f32 (normalized 0-1).
#[inline]
pub fn xyb_planes_to_rgb_f32_simd(plane0: &[f32], plane1: &[f32], plane2: &[f32], rgb: &mut [f32]) {
    debug_assert_eq!(plane0.len(), plane1.len());
    debug_assert_eq!(plane0.len(), plane2.len());
    debug_assert_eq!(rgb.len(), plane0.len() * 3);

    let num_pixels = plane0.len();
    let offset = f32x8::splat(128.0);
    let scale = f32x8::splat(1.0 / 255.0);
    let zero = f32x8::splat(0.0);
    let one = f32x8::splat(1.0);

    let chunks = num_pixels / 8;
    for chunk in 0..chunks {
        let base = chunk * 8;

        let p0 = f32x8::from([
            plane0[base],
            plane0[base + 1],
            plane0[base + 2],
            plane0[base + 3],
            plane0[base + 4],
            plane0[base + 5],
            plane0[base + 6],
            plane0[base + 7],
        ]);
        let p1 = f32x8::from([
            plane1[base],
            plane1[base + 1],
            plane1[base + 2],
            plane1[base + 3],
            plane1[base + 4],
            plane1[base + 5],
            plane1[base + 6],
            plane1[base + 7],
        ]);
        let p2 = f32x8::from([
            plane2[base],
            plane2[base + 1],
            plane2[base + 2],
            plane2[base + 3],
            plane2[base + 4],
            plane2[base + 5],
            plane2[base + 6],
            plane2[base + 7],
        ]);

        // Level shift, scale to 0-1, clamp
        let r = ((p0 + offset) * scale).max(zero).min(one);
        let g = ((p1 + offset) * scale).max(zero).min(one);
        let b = ((p2 + offset) * scale).max(zero).min(one);

        let r_arr: [f32; 8] = r.into();
        let g_arr: [f32; 8] = g.into();
        let b_arr: [f32; 8] = b.into();

        for j in 0..8 {
            let idx = (base + j) * 3;
            rgb[idx] = r_arr[j];
            rgb[idx + 1] = g_arr[j];
            rgb[idx + 2] = b_arr[j];
        }
    }

    // Scalar remainder
    for i in (chunks * 8)..num_pixels {
        let idx = i * 3;
        rgb[idx] = ((plane0[i] + 128.0) / 255.0).clamp(0.0, 1.0);
        rgb[idx + 1] = ((plane1[i] + 128.0) / 255.0).clamp(0.0, 1.0);
        rgb[idx + 2] = ((plane2[i] + 128.0) / 255.0).clamp(0.0, 1.0);
    }
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
    const EPSILON_ACCUMULATED: f32 = 5e-4;

    #[test]
    fn test_gather_even_odd_x8_correctness() {
        // Create test data: 16 sequential floats
        let data: Vec<f32> = (0..32).map(|i| i as f32).collect();

        // Call the function
        let (evens, odds) = gather_even_odd_x8(&data, 0, 32);

        // Expected: evens = [0, 2, 4, 6, 8, 10, 12, 14]
        let evens_arr: [f32; 8] = evens.into();
        let odds_arr: [f32; 8] = odds.into();

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
        let (evens2, odds2) = gather_even_odd_x8(&data, 4, 32);
        let evens2_arr: [f32; 8] = evens2.into();
        let odds2_arr: [f32; 8] = odds2.into();

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

    #[test]
    fn test_downsample_2x2_simd_matches_scalar() {
        // Create test data (16x16 gradient)
        let width = 16;
        let height = 16;
        let plane: Vec<f32> = (0..width * height)
            .map(|i| {
                let x = i % width;
                let y = i / width;
                (x + y * 10) as f32
            })
            .collect();

        // SIMD version
        let simd_result = downsample_2x2_simd(&plane, width, height).unwrap();

        // Scalar reference
        let new_width = (width + 1) / 2;
        let new_height = (height + 1) / 2;
        let mut scalar_result = vec![0.0f32; new_width * new_height];

        for y in 0..new_height {
            for x in 0..new_width {
                let x0 = x * 2;
                let y0 = y * 2;
                let x1 = (x0 + 1).min(width - 1);
                let y1 = (y0 + 1).min(height - 1);

                let p00 = plane[y0 * width + x0];
                let p10 = plane[y0 * width + x1];
                let p01 = plane[y1 * width + x0];
                let p11 = plane[y1 * width + x1];

                scalar_result[y * new_width + x] = (p00 + p10 + p01 + p11) * 0.25;
            }
        }

        // Compare
        assert_eq!(simd_result.len(), scalar_result.len());
        for i in 0..simd_result.len() {
            let diff = (simd_result[i] - scalar_result[i]).abs();
            assert!(
                diff < EPSILON,
                "Downsample mismatch at {}: SIMD={}, scalar={}, diff={}",
                i,
                simd_result[i],
                scalar_result[i],
                diff
            );
        }
    }

    #[test]
    fn test_downsample_2x1_simd_matches_scalar() {
        let width = 16;
        let height = 8;
        let plane: Vec<f32> = (0..width * height)
            .map(|i| {
                let x = i % width;
                let y = i / width;
                (x + y * 10) as f32
            })
            .collect();

        let simd_result = downsample_2x1_simd(&plane, width, height).unwrap();

        let new_width = (width + 1) / 2;
        let mut scalar_result = vec![0.0f32; new_width * height];
        for y in 0..height {
            for x in 0..new_width {
                let x0 = x * 2;
                let x1 = (x0 + 1).min(width - 1);
                let p0 = plane[y * width + x0];
                let p1 = plane[y * width + x1];
                scalar_result[y * new_width + x] = (p0 + p1) * 0.5;
            }
        }

        assert_eq!(simd_result.len(), scalar_result.len());
        for i in 0..simd_result.len() {
            let diff = (simd_result[i] - scalar_result[i]).abs();
            assert!(
                diff < EPSILON,
                "Downsample 2x1 mismatch at {}: SIMD={}, scalar={}, diff={}",
                i,
                simd_result[i],
                scalar_result[i],
                diff
            );
        }
    }

    #[test]
    fn test_downsample_1x2_simd_matches_scalar() {
        let width = 8;
        let height = 16;
        let plane: Vec<f32> = (0..width * height)
            .map(|i| {
                let x = i % width;
                let y = i / width;
                (x + y * 10) as f32
            })
            .collect();

        let simd_result = downsample_1x2_simd(&plane, width, height).unwrap();

        let new_height = (height + 1) / 2;
        let mut scalar_result = vec![0.0f32; width * new_height];
        for y in 0..new_height {
            for x in 0..width {
                let y0 = y * 2;
                let y1 = (y0 + 1).min(height - 1);
                let p0 = plane[y0 * width + x];
                let p1 = plane[y1 * width + x];
                scalar_result[y * width + x] = (p0 + p1) * 0.5;
            }
        }

        assert_eq!(simd_result.len(), scalar_result.len());
        for i in 0..simd_result.len() {
            let diff = (simd_result[i] - scalar_result[i]).abs();
            assert!(
                diff < EPSILON,
                "Downsample 1x2 mismatch at {}: SIMD={}, scalar={}, diff={}",
                i,
                simd_result[i],
                scalar_result[i],
                diff
            );
        }
    }

    #[test]
    fn test_rgb_to_ycbcr_simd_matches_scalar() {
        // Create test RGB data (24 pixels for good SIMD coverage)
        let num_pixels = 24;
        let rgb_data: Vec<u8> = (0..num_pixels * 3)
            .map(|i| ((i * 11 + 7) % 256) as u8)
            .collect();

        // SIMD version
        let (y_simd, cb_simd, cr_simd) = rgb_to_ycbcr_planes_simd(&rgb_data, num_pixels).unwrap();

        // Scalar reference
        let mut y_scalar = vec![0.0f32; num_pixels];
        let mut cb_scalar = vec![0.0f32; num_pixels];
        let mut cr_scalar = vec![0.0f32; num_pixels];

        for i in 0..num_pixels {
            let r = rgb_data[i * 3] as f32;
            let g = rgb_data[i * 3 + 1] as f32;
            let b = rgb_data[i * 3 + 2] as f32;

            y_scalar[i] = YCBCR_R_TO_Y * r + YCBCR_G_TO_Y * g + YCBCR_B_TO_Y * b;
            cb_scalar[i] = 128.0 + YCBCR_R_TO_CB * r + YCBCR_G_TO_CB * g + YCBCR_B_TO_CB * b;
            cr_scalar[i] = 128.0 + YCBCR_R_TO_CR * r + YCBCR_G_TO_CR * g + YCBCR_B_TO_CR * b;
        }

        // Compare
        for i in 0..num_pixels {
            assert!(
                (y_simd[i] - y_scalar[i]).abs() < EPSILON,
                "Y mismatch at {}: SIMD={}, scalar={}",
                i,
                y_simd[i],
                y_scalar[i]
            );
            assert!(
                (cb_simd[i] - cb_scalar[i]).abs() < EPSILON,
                "Cb mismatch at {}: SIMD={}, scalar={}",
                i,
                cb_simd[i],
                cb_scalar[i]
            );
            assert!(
                (cr_simd[i] - cr_scalar[i]).abs() < EPSILON,
                "Cr mismatch at {}: SIMD={}, scalar={}",
                i,
                cr_simd[i],
                cr_scalar[i]
            );
        }
    }

    /// Test AVX2 intrinsics RGB to YCbCr against scalar reference.
    #[test]
    #[cfg(all(feature = "unsafe_simd", target_arch = "x86_64"))]
    fn test_rgb_to_ycbcr_avx2_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        // Test with 8 pixels (one AVX2 batch)
        let rgb_data: Vec<u8> = (0..24).map(|i| ((i * 17 + 5) % 256) as u8).collect();

        let mut y_avx2 = vec![0.0f32; 8];
        let mut cb_avx2 = vec![0.0f32; 8];
        let mut cr_avx2 = vec![0.0f32; 8];

        unsafe {
            rgb_to_ycbcr_8px_avx2(
                rgb_data.as_ptr(),
                y_avx2.as_mut_ptr(),
                cb_avx2.as_mut_ptr(),
                cr_avx2.as_mut_ptr(),
            );
        }

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
    #[cfg(all(feature = "unsafe_simd", target_arch = "x86_64"))]
    fn test_rgb_to_ycbcr_avx2_brute_force() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        // Test systematic patterns covering all u8 values
        let mut max_y_diff = 0.0f32;
        let mut max_cb_diff = 0.0f32;
        let mut max_cr_diff = 0.0f32;

        // Test every 16th R, G, B combination (256/16 = 16^3 = 4096 combinations)
        for r_base in (0u8..=255).step_by(16) {
            for g_base in (0u8..=255).step_by(64) {
                for b_base in (0u8..=255).step_by(64) {
                    // Create 8 pixels with slight variations
                    let mut rgb_data = vec![0u8; 24];
                    for p in 0..8 {
                        let r = r_base.wrapping_add((p * 2) as u8);
                        let g = g_base.wrapping_add((p * 3) as u8);
                        let b = b_base.wrapping_add((p * 5) as u8);
                        rgb_data[p * 3] = r;
                        rgb_data[p * 3 + 1] = g;
                        rgb_data[p * 3 + 2] = b;
                    }

                    let mut y_avx2 = vec![0.0f32; 8];
                    let mut cb_avx2 = vec![0.0f32; 8];
                    let mut cr_avx2 = vec![0.0f32; 8];

                    unsafe {
                        rgb_to_ycbcr_8px_avx2(
                            rgb_data.as_ptr(),
                            y_avx2.as_mut_ptr(),
                            cb_avx2.as_mut_ptr(),
                            cr_avx2.as_mut_ptr(),
                        );
                    }

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

    /// Test AVX2 RGB to YCbCr matches existing SIMD implementation.
    #[test]
    #[cfg(all(feature = "unsafe_simd", target_arch = "x86_64"))]
    fn test_rgb_to_ycbcr_avx2_matches_existing_simd() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        // Test multiple batches
        let num_pixels = 64;
        let rgb_data: Vec<u8> = (0..num_pixels * 3)
            .map(|i| ((i * 23 + 11) % 256) as u8)
            .collect();

        // Existing SIMD implementation
        let (y_simd, cb_simd, cr_simd) = rgb_to_ycbcr_planes_simd(&rgb_data, num_pixels).unwrap();

        // AVX2 intrinsics
        let mut y_avx2 = vec![0.0f32; num_pixels];
        let mut cb_avx2 = vec![0.0f32; num_pixels];
        let mut cr_avx2 = vec![0.0f32; num_pixels];

        for chunk in 0..(num_pixels / 8) {
            let rgb_offset = chunk * 24;
            let out_offset = chunk * 8;
            unsafe {
                rgb_to_ycbcr_8px_avx2(
                    rgb_data.as_ptr().add(rgb_offset),
                    y_avx2.as_mut_ptr().add(out_offset),
                    cb_avx2.as_mut_ptr().add(out_offset),
                    cr_avx2.as_mut_ptr().add(out_offset),
                );
            }
        }

        // Compare
        for i in 0..num_pixels {
            let y_diff = (y_avx2[i] - y_simd[i]).abs();
            let cb_diff = (cb_avx2[i] - cb_simd[i]).abs();
            let cr_diff = (cr_avx2[i] - cr_simd[i]).abs();
            assert!(
                y_diff < EPSILON_ACCUMULATED,
                "Y mismatch at {}: AVX2={}, existing SIMD={}, diff={}",
                i,
                y_avx2[i],
                y_simd[i],
                y_diff
            );
            assert!(
                cb_diff < EPSILON_ACCUMULATED,
                "Cb mismatch at {}: AVX2={}, existing SIMD={}, diff={}",
                i,
                cb_avx2[i],
                cb_simd[i],
                cb_diff
            );
            assert!(
                cr_diff < EPSILON_ACCUMULATED,
                "Cr mismatch at {}: AVX2={}, existing SIMD={}, diff={}",
                i,
                cr_avx2[i],
                cr_simd[i],
                cr_diff
            );
        }
    }

    #[test]
    fn test_rgba_to_ycbcr_simd_matches_scalar() {
        let num_pixels = 24;
        let rgba_data: Vec<u8> = (0..num_pixels * 4)
            .map(|i| ((i * 13 + 3) % 256) as u8)
            .collect();

        let (y_simd, cb_simd, cr_simd) = rgba_to_ycbcr_planes_simd(&rgba_data, num_pixels).unwrap();

        for i in 0..num_pixels {
            let r = rgba_data[i * 4] as f32;
            let g = rgba_data[i * 4 + 1] as f32;
            let b = rgba_data[i * 4 + 2] as f32;

            let y_scalar = YCBCR_R_TO_Y * r + YCBCR_G_TO_Y * g + YCBCR_B_TO_Y * b;
            let cb_scalar = 128.0 + YCBCR_R_TO_CB * r + YCBCR_G_TO_CB * g + YCBCR_B_TO_CB * b;
            let cr_scalar = 128.0 + YCBCR_R_TO_CR * r + YCBCR_G_TO_CR * g + YCBCR_B_TO_CR * b;

            assert!(
                (y_simd[i] - y_scalar).abs() < EPSILON,
                "Y mismatch at {}: SIMD={}, scalar={}",
                i,
                y_simd[i],
                y_scalar
            );
            assert!(
                (cb_simd[i] - cb_scalar).abs() < EPSILON,
                "Cb mismatch at {}: SIMD={}, scalar={}",
                i,
                cb_simd[i],
                cb_scalar
            );
            assert!(
                (cr_simd[i] - cr_scalar).abs() < EPSILON,
                "Cr mismatch at {}: SIMD={}, scalar={}",
                i,
                cr_simd[i],
                cr_scalar
            );
        }
    }

    #[test]
    fn test_bgr_to_ycbcr_simd_matches_scalar() {
        let num_pixels = 24;
        let bgr_data: Vec<u8> = (0..num_pixels * 3)
            .map(|i| ((i * 17 + 5) % 256) as u8)
            .collect();

        let (y_simd, cb_simd, cr_simd) = bgr_to_ycbcr_planes_simd(&bgr_data, num_pixels).unwrap();

        for i in 0..num_pixels {
            // BGR: B at offset 0, G at offset 1, R at offset 2
            let b = bgr_data[i * 3] as f32;
            let g = bgr_data[i * 3 + 1] as f32;
            let r = bgr_data[i * 3 + 2] as f32;

            let y_scalar = YCBCR_R_TO_Y * r + YCBCR_G_TO_Y * g + YCBCR_B_TO_Y * b;
            let cb_scalar = 128.0 + YCBCR_R_TO_CB * r + YCBCR_G_TO_CB * g + YCBCR_B_TO_CB * b;
            let cr_scalar = 128.0 + YCBCR_R_TO_CR * r + YCBCR_G_TO_CR * g + YCBCR_B_TO_CR * b;

            assert!(
                (y_simd[i] - y_scalar).abs() < EPSILON,
                "Y mismatch at {}: SIMD={}, scalar={}",
                i,
                y_simd[i],
                y_scalar
            );
            assert!(
                (cb_simd[i] - cb_scalar).abs() < EPSILON,
                "Cb mismatch at {}: SIMD={}, scalar={}",
                i,
                cb_simd[i],
                cb_scalar
            );
            assert!(
                (cr_simd[i] - cr_scalar).abs() < EPSILON,
                "Cr mismatch at {}: SIMD={}, scalar={}",
                i,
                cr_simd[i],
                cr_scalar
            );
        }
    }

    #[test]
    fn test_bgra_to_ycbcr_simd_matches_scalar() {
        let num_pixels = 24;
        let bgra_data: Vec<u8> = (0..num_pixels * 4)
            .map(|i| ((i * 19 + 7) % 256) as u8)
            .collect();

        let (y_simd, cb_simd, cr_simd) = bgra_to_ycbcr_planes_simd(&bgra_data, num_pixels).unwrap();

        for i in 0..num_pixels {
            // BGRA: B at offset 0, G at offset 1, R at offset 2, A at offset 3
            let b = bgra_data[i * 4] as f32;
            let g = bgra_data[i * 4 + 1] as f32;
            let r = bgra_data[i * 4 + 2] as f32;

            let y_scalar = YCBCR_R_TO_Y * r + YCBCR_G_TO_Y * g + YCBCR_B_TO_Y * b;
            let cb_scalar = 128.0 + YCBCR_R_TO_CB * r + YCBCR_G_TO_CB * g + YCBCR_B_TO_CB * b;
            let cr_scalar = 128.0 + YCBCR_R_TO_CR * r + YCBCR_G_TO_CR * g + YCBCR_B_TO_CR * b;

            assert!(
                (y_simd[i] - y_scalar).abs() < EPSILON,
                "Y mismatch at {}: SIMD={}, scalar={}",
                i,
                y_simd[i],
                y_scalar
            );
            assert!(
                (cb_simd[i] - cb_scalar).abs() < EPSILON,
                "Cb mismatch at {}: SIMD={}, scalar={}",
                i,
                cb_simd[i],
                cb_scalar
            );
            assert!(
                (cr_simd[i] - cr_scalar).abs() < EPSILON,
                "Cr mismatch at {}: SIMD={}, scalar={}",
                i,
                cr_simd[i],
                cr_scalar
            );
        }
    }

    #[test]
    fn test_gray_to_ycbcr_simd_matches_scalar() {
        let num_pixels = 24;
        let gray_data: Vec<u8> = (0..num_pixels)
            .map(|i| ((i * 11 + 3) % 256) as u8)
            .collect();

        let (y_simd, cb_simd, cr_simd) = gray_to_ycbcr_planes_simd(&gray_data, num_pixels).unwrap();

        for i in 0..num_pixels {
            let y_scalar = gray_data[i] as f32;
            let cb_scalar = 128.0f32;
            let cr_scalar = 128.0f32;

            assert!(
                (y_simd[i] - y_scalar).abs() < EPSILON,
                "Y mismatch at {}: SIMD={}, scalar={}",
                i,
                y_simd[i],
                y_scalar
            );
            assert!(
                (cb_simd[i] - cb_scalar).abs() < EPSILON,
                "Cb mismatch at {}: SIMD={}, scalar={}",
                i,
                cb_simd[i],
                cb_scalar
            );
            assert!(
                (cr_simd[i] - cr_scalar).abs() < EPSILON,
                "Cr mismatch at {}: SIMD={}, scalar={}",
                i,
                cr_simd[i],
                cr_scalar
            );
        }
    }

    #[test]
    fn test_extract_block_simd_interior() {
        // Create a 32x32 plane (4x4 blocks)
        let width = 32usize;
        let height = 32usize;
        let plane: Vec<f32> = (0..(width * height)).map(|i| (i % 256) as f32).collect();

        // Test interior block at (1, 1) - no edge handling needed
        let block = extract_block_simd(&plane, width, height, 1, 1);

        // Verify values manually
        for y in 0..8 {
            for x in 0..8 {
                let px = 1 * 8 + x;
                let py = 1 * 8 + y;
                let expected = plane[py * width + px] - 128.0;
                assert!(
                    (block[y * 8 + x] - expected).abs() < EPSILON,
                    "Mismatch at ({}, {}): got={}, expected={}",
                    x,
                    y,
                    block[y * 8 + x],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_extract_block_simd_edge() {
        // Create a 20x20 plane (blocks at edge need clamping)
        let width = 20usize;
        let height = 20usize;
        let plane: Vec<f32> = (0..(width * height)).map(|i| (i % 256) as f32).collect();

        // Test edge block at (2, 2) - partially outside bounds
        let block = extract_block_simd(&plane, width, height, 2, 2);

        // Verify values with clamping
        for y in 0..8 {
            for x in 0..8 {
                let px = (2 * 8 + x).min(width - 1);
                let py = (2 * 8 + y).min(height - 1);
                let expected = plane[py * width + px] - 128.0;
                assert!(
                    (block[y * 8 + x] - expected).abs() < EPSILON,
                    "Mismatch at ({}, {}): got={}, expected={}",
                    x,
                    y,
                    block[y * 8 + x],
                    expected
                );
            }
        }
    }

    // ========================================================================
    // AVX2 Intrinsics Brute Force Tests
    // ========================================================================

    /// Test gather_even_odd_avx2_raw against scalar reference with all possible byte patterns.
    #[test]
    #[cfg(all(feature = "unsafe_simd", target_arch = "x86_64"))]
    fn test_gather_even_odd_avx2_brute_force() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("Skipping AVX2 test - CPU doesn't support AVX2");
            return;
        }

        // Test with various patterns
        let test_patterns: Vec<Vec<f32>> = vec![
            // Sequential
            (0..16).map(|i| i as f32).collect(),
            // Reverse
            (0..16).rev().map(|i| i as f32).collect(),
            // All same
            vec![42.0; 16],
            // Alternating
            (0..16)
                .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
                .collect(),
            // Large values
            (0..16).map(|i| (i as f32) * 1000.0).collect(),
            // Small values
            (0..16).map(|i| (i as f32) * 0.001).collect(),
            // Random-ish pattern
            vec![
                3.14, 2.71, 1.41, 1.73, 2.23, 0.57, 1.61, 4.67, 9.81, 6.28, 0.69, 1.38, 2.30, 3.45,
                4.56, 5.67,
            ],
        ];

        for (pattern_idx, data) in test_patterns.iter().enumerate() {
            // Scalar reference
            let (scalar_evens, scalar_odds) = gather_even_odd_scalar(data);

            // AVX2 intrinsics
            unsafe {
                let (avx2_evens, avx2_odds) = gather_even_odd_avx2_raw(data.as_ptr());
                let avx2_evens_arr: [f32; 8] = core::mem::transmute(avx2_evens);
                let avx2_odds_arr: [f32; 8] = core::mem::transmute(avx2_odds);

                for i in 0..8 {
                    assert!(
                        (avx2_evens_arr[i] - scalar_evens[i]).abs() < EPSILON,
                        "Pattern {}: evens mismatch at {}: AVX2={}, scalar={}",
                        pattern_idx,
                        i,
                        avx2_evens_arr[i],
                        scalar_evens[i]
                    );
                    assert!(
                        (avx2_odds_arr[i] - scalar_odds[i]).abs() < EPSILON,
                        "Pattern {}: odds mismatch at {}: AVX2={}, scalar={}",
                        pattern_idx,
                        i,
                        avx2_odds_arr[i],
                        scalar_odds[i]
                    );
                }
            }
        }
    }

    /// Exhaustive test of gather_even_odd with sequential integers.
    #[test]
    #[cfg(all(feature = "unsafe_simd", target_arch = "x86_64"))]
    fn test_gather_even_odd_avx2_exhaustive() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        // Test all starting offsets from 0-255
        for offset in 0..256 {
            let data: Vec<f32> = (0..16).map(|i| (offset + i) as f32).collect();
            let (scalar_evens, scalar_odds) = gather_even_odd_scalar(&data);

            unsafe {
                let (avx2_evens, avx2_odds) = gather_even_odd_avx2_raw(data.as_ptr());
                let avx2_evens_arr: [f32; 8] = core::mem::transmute(avx2_evens);
                let avx2_odds_arr: [f32; 8] = core::mem::transmute(avx2_odds);

                for i in 0..8 {
                    assert_eq!(
                        avx2_evens_arr[i], scalar_evens[i],
                        "Offset {}: evens[{}] mismatch",
                        offset, i
                    );
                    assert_eq!(
                        avx2_odds_arr[i], scalar_odds[i],
                        "Offset {}: odds[{}] mismatch",
                        offset, i
                    );
                }
            }
        }
    }

    /// Test downsample_2x2_avx2 against scalar reference with various sizes.
    #[test]
    #[cfg(all(feature = "unsafe_simd", target_arch = "x86_64"))]
    fn test_downsample_2x2_avx2_brute_force() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("Skipping AVX2 test - CPU doesn't support AVX2");
            return;
        }

        // Test various image sizes
        let test_sizes = [
            (16, 16),
            (32, 32),
            (64, 64),
            (128, 128),
            (17, 17), // Odd sizes
            (31, 33),
            (100, 100),
            (256, 256),
        ];

        for (width, height) in test_sizes {
            // Create test data with gradient pattern
            let plane: Vec<f32> = (0..width * height)
                .map(|i| {
                    let x = i % width;
                    let y = i / width;
                    (x as f32 * 0.5 + y as f32 * 0.3) % 256.0
                })
                .collect();

            // Scalar reference
            let scalar_result = downsample_2x2_scalar(&plane, width, height);

            // AVX2 intrinsics
            let new_width = (width + 1) / 2;
            let new_height = (height + 1) / 2;
            let mut avx2_result = vec![0.0f32; new_width * new_height];

            unsafe {
                downsample_2x2_avx2(&plane, width, height, &mut avx2_result);
            }

            // Compare
            assert_eq!(
                avx2_result.len(),
                scalar_result.len(),
                "Size {}x{}: length mismatch",
                width,
                height
            );

            for i in 0..scalar_result.len() {
                let diff = (avx2_result[i] - scalar_result[i]).abs();
                assert!(
                    diff < EPSILON_ACCUMULATED,
                    "Size {}x{}: mismatch at {}: AVX2={}, scalar={}, diff={}",
                    width,
                    height,
                    i,
                    avx2_result[i],
                    scalar_result[i],
                    diff
                );
            }
        }
    }

    /// Test downsample_2x2_avx2 with random data.
    #[test]
    #[cfg(all(feature = "unsafe_simd", target_arch = "x86_64"))]
    fn test_downsample_2x2_avx2_random_patterns() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        let width = 128;
        let height = 128;

        // Pseudo-random LCG for deterministic "random" data
        let mut seed: u32 = 12345;
        let mut next_rand = || {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            (seed >> 16) as f32 / 65535.0 * 255.0
        };

        let plane: Vec<f32> = (0..width * height).map(|_| next_rand()).collect();

        let scalar_result = downsample_2x2_scalar(&plane, width, height);

        let new_width = (width + 1) / 2;
        let new_height = (height + 1) / 2;
        let mut avx2_result = vec![0.0f32; new_width * new_height];

        unsafe {
            downsample_2x2_avx2(&plane, width, height, &mut avx2_result);
        }

        for i in 0..scalar_result.len() {
            let diff = (avx2_result[i] - scalar_result[i]).abs();
            assert!(
                diff < EPSILON_ACCUMULATED,
                "Random test: mismatch at {}: AVX2={}, scalar={}, diff={}",
                i,
                avx2_result[i],
                scalar_result[i],
                diff
            );
        }
    }

    /// Test downsample_2x2_avx2 matches the existing SIMD implementation.
    #[test]
    #[cfg(all(feature = "unsafe_simd", target_arch = "x86_64"))]
    fn test_downsample_2x2_avx2_matches_existing_simd() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        let width = 256;
        let height = 256;
        let plane: Vec<f32> = (0..width * height)
            .map(|i| ((i * 17 + 23) % 256) as f32)
            .collect();

        // Existing SIMD implementation
        let existing_result = downsample_2x2_simd(&plane, width, height).unwrap();

        // New AVX2 intrinsics implementation
        let new_width = (width + 1) / 2;
        let new_height = (height + 1) / 2;
        let mut avx2_result = vec![0.0f32; new_width * new_height];

        unsafe {
            downsample_2x2_avx2(&plane, width, height, &mut avx2_result);
        }

        for i in 0..existing_result.len() {
            let diff = (avx2_result[i] - existing_result[i]).abs();
            assert!(
                diff < EPSILON,
                "Mismatch at {}: AVX2={}, existing SIMD={}, diff={}",
                i,
                avx2_result[i],
                existing_result[i],
                diff
            );
        }
    }

    /// Test edge cases: minimum sizes
    #[test]
    #[cfg(all(feature = "unsafe_simd", target_arch = "x86_64"))]
    fn test_downsample_2x2_avx2_edge_cases() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        // Test 2x2 (minimum)
        let plane_2x2 = vec![1.0, 2.0, 3.0, 4.0];
        let scalar = downsample_2x2_scalar(&plane_2x2, 2, 2);
        let mut avx2 = vec![0.0f32; 1];
        unsafe {
            downsample_2x2_avx2(&plane_2x2, 2, 2, &mut avx2);
        }
        assert!(
            (avx2[0] - scalar[0]).abs() < EPSILON,
            "2x2: AVX2={}, scalar={}",
            avx2[0],
            scalar[0]
        );

        // Test 1x1 (degenerate)
        let plane_1x1 = vec![42.0];
        let scalar = downsample_2x2_scalar(&plane_1x1, 1, 1);
        let mut avx2 = vec![0.0f32; 1];
        unsafe {
            downsample_2x2_avx2(&plane_1x1, 1, 1, &mut avx2);
        }
        assert!(
            (avx2[0] - scalar[0]).abs() < EPSILON,
            "1x1: AVX2={}, scalar={}",
            avx2[0],
            scalar[0]
        );
    }
}
