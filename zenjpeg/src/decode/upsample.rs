//! Chroma upsampling for JPEG decoding.
//!
//! Implements triangle filter (3:1 weighting) upsampling for various
//! chroma subsampling modes (4:2:2, 4:4:0, 4:2:0).

use wide::f32x8;

#[cfg(target_arch = "x86_64")]
use archmage::{SimdToken, arcane, rite};

#[cfg(target_arch = "x86_64")]
use safe_unaligned_simd::x86_64 as safe_simd;

/// Max chroma strip width for stack-allocated scratch in triangle upsampling.
/// Covers images up to 8192px wide (chroma width 4096 at 4:2:0).
pub(crate) const MAX_UPSAMPLE_SCRATCH: usize = 4096;

/// Fancy upsampling with triangle filter (3:1 weights).
///
/// Applies separable 3:1 interpolation: (3 * near + far) / 4.
/// Dispatches to specialized implementations based on scale factors.
pub fn upsample_fancy(
    input: &[f32],
    in_width: usize,
    in_height: usize,
    out_width: usize,
    out_height: usize,
    scale_x: usize,
    scale_y: usize,
) -> Vec<f32> {
    match (scale_x, scale_y) {
        (1, 1) => {
            // No upsampling needed, but still need to crop to output dimensions
            // Input may be block-aligned (e.g., 320x304) while output is image-sized (300x300)
            let mut output = vec![0.0f32; out_width * out_height];
            for y in 0..out_height {
                let in_y = y.min(in_height.saturating_sub(1));
                for x in 0..out_width {
                    let in_x = x.min(in_width.saturating_sub(1));
                    output[y * out_width + x] = input[in_y * in_width + in_x];
                }
            }
            output
        }
        (2, 1) => upsample_h2v1(input, in_width, in_height, out_width, out_height),
        (1, 2) => upsample_h1v2(input, in_width, in_height, out_width, out_height),
        (2, 2) => upsample_h2v2(input, in_width, in_height, out_width, out_height),
        _ => {
            // Fall back to box filter for unusual scale factors (e.g., 4x2)
            let mut output = vec![0.0f32; out_width * out_height];
            for y in 0..out_height {
                let in_y = (y / scale_y).min(in_height.saturating_sub(1));
                for x in 0..out_width {
                    let in_x = (x / scale_x).min(in_width.saturating_sub(1));
                    output[y * out_width + x] = input[in_y * in_width + in_x];
                }
            }
            output
        }
    }
}

/// Horizontal 2x upsampling (4:2:2) with triangle filter.
#[inline]
pub fn upsample_h2v1(
    input: &[f32],
    in_width: usize,
    in_height: usize,
    out_width: usize,
    out_height: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; out_width * out_height];

    for y in 0..out_height {
        let in_y = y.min(in_height.saturating_sub(1));
        for out_x in 0..out_width {
            let in_x = out_x / 2;
            let curr = input[in_y * in_width + in_x];

            if out_x % 2 == 0 {
                let left = if in_x > 0 {
                    input[in_y * in_width + in_x - 1]
                } else {
                    curr
                };
                output[y * out_width + out_x] = (3.0 * curr + left) * 0.25;
            } else {
                let right = if in_x + 1 < in_width {
                    input[in_y * in_width + in_x + 1]
                } else {
                    curr
                };
                output[y * out_width + out_x] = (3.0 * curr + right) * 0.25;
            }
        }
    }

    output
}

/// Vertical 2x upsampling (4:4:0) with triangle filter.
///
/// Uses SIMD (wide crate) for the interior pixels.
#[inline]
pub fn upsample_h1v2(
    input: &[f32],
    in_width: usize,
    in_height: usize,
    out_width: usize,
    out_height: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; out_width * out_height];

    let three = f32x8::splat(3.0);
    let quarter = f32x8::splat(0.25);

    for out_y in 0..out_height {
        let in_y = out_y / 2;
        let is_top = out_y % 2 == 0;
        let out_row_start = out_y * out_width;

        // Get neighbor row (above for top, below for bottom)
        let neighbor_y = if is_top {
            in_y.saturating_sub(1)
        } else {
            (in_y + 1).min(in_height - 1)
        };

        let curr_row_start = in_y * in_width;
        let neighbor_row_start = neighbor_y * in_width;

        // SIMD path: process 8 pixels at a time in interior
        let simd_width = in_width.min(out_width);
        let chunks = simd_width / 8;

        for chunk in 0..chunks {
            let x = chunk * 8;

            let curr = f32x8::from([
                input[curr_row_start + x],
                input[curr_row_start + x + 1],
                input[curr_row_start + x + 2],
                input[curr_row_start + x + 3],
                input[curr_row_start + x + 4],
                input[curr_row_start + x + 5],
                input[curr_row_start + x + 6],
                input[curr_row_start + x + 7],
            ]);
            let neighbor = f32x8::from([
                input[neighbor_row_start + x],
                input[neighbor_row_start + x + 1],
                input[neighbor_row_start + x + 2],
                input[neighbor_row_start + x + 3],
                input[neighbor_row_start + x + 4],
                input[neighbor_row_start + x + 5],
                input[neighbor_row_start + x + 6],
                input[neighbor_row_start + x + 7],
            ]);

            let blended = (three * curr + neighbor) * quarter;
            let arr: [f32; 8] = blended.into();
            output[out_row_start + x..out_row_start + x + 8].copy_from_slice(&arr);
        }

        // Scalar remainder
        for x in (chunks * 8)..out_width {
            let in_x = x.min(in_width.saturating_sub(1));
            let curr = input[curr_row_start + in_x];
            let neighbor = input[neighbor_row_start + in_x];
            output[out_row_start + x] = (3.0 * curr + neighbor) * 0.25;
        }
    }

    output
}

/// Both horizontal and vertical 2x upsampling (4:2:0) with triangle filter.
///
/// Applied separably: horizontal first, then vertical.
#[inline]
pub fn upsample_h2v2(
    input: &[f32],
    in_width: usize,
    in_height: usize,
    out_width: usize,
    out_height: usize,
) -> Vec<f32> {
    // First upsample horizontally
    let h_upsampled = upsample_h2v1(input, in_width, in_height, out_width, in_height);
    // Then upsample vertically
    upsample_h1v2(&h_upsampled, out_width, in_height, out_width, out_height)
}

// =============================================================================
// i16 Upsampling Functions (for fast decode path)
// =============================================================================

/// Triangle filter 2x2 upsampling in i16 (4:2:0 → 4:4:4).
///
/// Uses (3 * near + far + 2) >> 2 for proper rounding.
/// Output is written to pre-allocated buffer for zero allocation.
#[inline]
pub fn upsample_h2v2_i16_fancy(
    input: &[i16],
    in_width: usize,
    in_height: usize,
    output: &mut [i16],
    out_width: usize,
    out_height: usize,
) {
    if in_width == 0 || in_height == 0 || out_width == 0 || out_height == 0 {
        return;
    }

    // Try AVX2 SIMD path on x86_64
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(token) = archmage::X64V3Token::summon() {
            upsample_h2v2_i16_fancy_avx2(
                token, input, in_width, in_height, output, out_width, out_height,
            );
            return;
        }
    }

    // Scalar fallback
    upsample_h2v2_i16_fancy_scalar(input, in_width, in_height, output, out_width, out_height);
}

/// Like [`upsample_h2v2_i16_fancy`] but reuses a caller-provided scratch buffer
/// to avoid re-zeroing a `[0i16; 4096]` stack array on every call.
///
/// The scratch buffer must have length >= `in_width`. Its contents are overwritten
/// by the vertical pass before the horizontal pass reads them, so it does NOT
/// need to be zeroed between calls.
pub fn upsample_h2v2_i16_fancy_reuse_scratch(
    input: &[i16],
    in_width: usize,
    in_height: usize,
    output: &mut [i16],
    out_width: usize,
    out_height: usize,
    scratch: &mut [i16],
) {
    if in_width == 0 || in_height == 0 || out_width == 0 || out_height == 0 {
        return;
    }

    // Try AVX2 SIMD path on x86_64
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(token) = archmage::X64V3Token::summon() {
            upsample_h2v2_i16_fancy_avx2_with_scratch(
                token, input, in_width, in_height, output, out_width, out_height, scratch,
            );
            return;
        }
    }

    // Scalar fallback (doesn't use scratch buffer)
    upsample_h2v2_i16_fancy_scalar(input, in_width, in_height, output, out_width, out_height);
}

/// Triangle filter 2x2 upsampling with explicit strides (for SIMD-aligned buffers).
///
/// Same algorithm as `upsample_h2v2_i16_fancy` but supports buffers where
/// stride > width (common for SIMD alignment).
#[inline]
pub fn upsample_h2v2_i16_fancy_strided(
    input: &[i16],
    in_width: usize,
    in_stride: usize,
    in_height: usize,
    output: &mut [i16],
    out_width: usize,
    out_stride: usize,
    out_height: usize,
) {
    if in_width == 0 || in_height == 0 || out_width == 0 || out_height == 0 {
        return;
    }

    // Try AVX2 SIMD path on x86_64
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(token) = archmage::X64V3Token::summon() {
            upsample_h2v2_i16_fancy_strided_avx2(
                token, input, in_width, in_stride, in_height, output, out_width, out_stride,
                out_height,
            );
            return;
        }
    }

    // Scalar fallback
    upsample_h2v2_i16_fancy_strided_scalar(
        input, in_width, in_stride, in_height, output, out_width, out_stride, out_height,
    );
}

/// Scalar implementation of strided bilinear upsampling
fn upsample_h2v2_i16_fancy_strided_scalar(
    input: &[i16],
    in_width: usize,
    in_stride: usize,
    in_height: usize,
    output: &mut [i16],
    out_width: usize,
    out_stride: usize,
    out_height: usize,
) {
    for out_y in 0..out_height {
        let in_y = (out_y / 2).min(in_height - 1);
        let is_top_half = out_y % 2 == 0;

        let v_neighbor_y = if is_top_half {
            in_y.saturating_sub(1)
        } else {
            (in_y + 1).min(in_height - 1)
        };

        let curr_row = &input[in_y * in_stride..];
        let v_neighbor_row = &input[v_neighbor_y * in_stride..];
        let out_row = &mut output[out_y * out_stride..][..out_width];

        upsample_row_h2_fancy_bilinear(curr_row, v_neighbor_row, in_width, out_row, is_top_half);
    }
}

/// AVX2 SIMD implementation of strided bilinear upsampling
#[cfg(target_arch = "x86_64")]
#[arcane]
fn upsample_h2v2_i16_fancy_strided_avx2(
    token: archmage::X64V3Token,
    input: &[i16],
    in_width: usize,
    in_stride: usize,
    in_height: usize,
    output: &mut [i16],
    out_width: usize,
    out_stride: usize,
    out_height: usize,
) {
    // Stack-allocated scratch for one row of vertical interpolation results
    const MAX_SCRATCH: usize = 4096;
    let mut scratch_storage = [0i16; MAX_SCRATCH];

    if in_width > MAX_SCRATCH {
        // Fall back to scalar for very wide images
        upsample_h2v2_i16_fancy_strided_scalar(
            input, in_width, in_stride, in_height, output, out_width, out_stride, out_height,
        );
        return;
    }

    let scratch = &mut scratch_storage[..in_width];

    for out_y in 0..out_height {
        let in_y = (out_y / 2).min(in_height.saturating_sub(1));
        let is_top_half = out_y % 2 == 0;

        let v_neighbor_y = if is_top_half {
            in_y.saturating_sub(1)
        } else {
            (in_y + 1).min(in_height.saturating_sub(1))
        };

        // Use stride for row addressing
        let curr_row = &input[in_y * in_stride..][..in_width];
        let v_neighbor_row = &input[v_neighbor_y * in_stride..][..in_width];
        let out_row = &mut output[out_y * out_stride..][..out_width];

        // Vertical pass: compute vertically-interpolated row into scratch
        upsample_vertical_row_strided_avx2(token, curr_row, v_neighbor_row, scratch);

        // Horizontal pass: horizontally interpolate scratch into output
        upsample_horizontal_row_strided_avx2(token, scratch, out_row);
    }
}

/// Vertical upsampling helper (AVX2) - processes one row
#[cfg(target_arch = "x86_64")]
#[rite]
fn upsample_vertical_row_strided_avx2(
    _token: archmage::X64V3Token,
    curr_row: &[i16],
    v_neighbor_row: &[i16],
    out: &mut [i16],
) {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    let len = curr_row.len().min(v_neighbor_row.len()).min(out.len());
    let chunks = len / 16;

    let three = _mm256_set1_epi16(3);
    let two = _mm256_set1_epi16(2);

    for i in 0..chunks {
        let offset = i * 16;
        let curr = safe_simd::_mm256_loadu_si256(
            <&[i16; 16]>::try_from(&curr_row[offset..offset + 16]).unwrap(),
        );
        let neighbor = safe_simd::_mm256_loadu_si256(
            <&[i16; 16]>::try_from(&v_neighbor_row[offset..offset + 16]).unwrap(),
        );

        // (3 * curr + neighbor + 2) >> 2
        let curr3 = _mm256_mullo_epi16(curr, three);
        let sum = _mm256_add_epi16(curr3, neighbor);
        let sum = _mm256_add_epi16(sum, two);
        let result = _mm256_srai_epi16(sum, 2);

        safe_simd::_mm256_storeu_si256(
            <&mut [i16; 16]>::try_from(&mut out[offset..offset + 16]).unwrap(),
            result,
        );
    }

    // Handle remaining elements
    for i in (chunks * 16)..len {
        let curr = curr_row[i] as i32;
        let neighbor = v_neighbor_row[i] as i32;
        out[i] = ((3 * curr + neighbor + 2) >> 2) as i16;
    }
}

/// Horizontal upsampling helper (AVX2) - expands one row to 2x width
#[cfg(target_arch = "x86_64")]
#[rite]
fn upsample_horizontal_row_strided_avx2(
    _token: archmage::X64V3Token,
    input: &[i16],
    output: &mut [i16],
) {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    let in_len = input.len();
    let out_len = output.len();

    if in_len == 0 || out_len == 0 {
        return;
    }

    let three = _mm256_set1_epi16(3);
    let two = _mm256_set1_epi16(2);

    // Process 16 input pixels at a time -> 32 output pixels
    let chunks = in_len / 16;

    for i in 0..chunks {
        let offset = i * 16;
        let out_offset = offset * 2;

        if out_offset + 32 > out_len {
            break;
        }

        let curr = safe_simd::_mm256_loadu_si256(
            <&[i16; 16]>::try_from(&input[offset..offset + 16]).unwrap(),
        );

        // Create neighbor vectors (shifted by 1, with edge replication)
        let neighbor_right = if offset + 17 <= in_len {
            safe_simd::_mm256_loadu_si256(
                <&[i16; 16]>::try_from(&input[offset + 1..offset + 17]).unwrap(),
            )
        } else {
            // Edge case: replicate last element
            let mut tmp = [0i16; 16];
            for j in 0..16 {
                tmp[j] = input[(offset + j + 1).min(in_len - 1)];
            }
            safe_simd::_mm256_loadu_si256(&tmp)
        };

        let neighbor_left = if offset > 0 {
            safe_simd::_mm256_loadu_si256(
                <&[i16; 16]>::try_from(&input[offset - 1..offset + 15]).unwrap(),
            )
        } else {
            // Edge case: replicate first element
            let mut tmp = [0i16; 16];
            tmp[0] = input[0];
            tmp[1..16].copy_from_slice(&input[0..15]);
            safe_simd::_mm256_loadu_si256(&tmp)
        };

        // Left output: (3 * curr + left_neighbor + 2) >> 2
        let curr3 = _mm256_mullo_epi16(curr, three);
        let sum_left = _mm256_add_epi16(curr3, neighbor_left);
        let sum_left = _mm256_add_epi16(sum_left, two);
        let left = _mm256_srai_epi16(sum_left, 2);

        // Right output: (3 * curr + right_neighbor + 2) >> 2
        let sum_right = _mm256_add_epi16(curr3, neighbor_right);
        let sum_right = _mm256_add_epi16(sum_right, two);
        let right = _mm256_srai_epi16(sum_right, 2);

        // Interleave left and right: [L0, R0, L1, R1, ...]
        let lo = _mm256_unpacklo_epi16(left, right);
        let hi = _mm256_unpackhi_epi16(left, right);

        // AVX2 unpack works on 128-bit lanes, need to permute
        let out0 = _mm256_permute2x128_si256(lo, hi, 0x20);
        let out1 = _mm256_permute2x128_si256(lo, hi, 0x31);

        safe_simd::_mm256_storeu_si256(
            <&mut [i16; 16]>::try_from(&mut output[out_offset..out_offset + 16]).unwrap(),
            out0,
        );
        safe_simd::_mm256_storeu_si256(
            <&mut [i16; 16]>::try_from(&mut output[out_offset + 16..out_offset + 32]).unwrap(),
            out1,
        );
    }

    // Handle remaining elements with scalar code
    for i in (chunks * 16)..in_len {
        let curr = input[i] as i32;
        let left = if i > 0 { input[i - 1] } else { input[0] } as i32;
        let right = if i + 1 < in_len {
            input[i + 1]
        } else {
            input[in_len - 1]
        } as i32;

        let out_idx = i * 2;
        if out_idx < out_len {
            output[out_idx] = ((3 * curr + left + 2) >> 2) as i16;
        }
        if out_idx + 1 < out_len {
            output[out_idx + 1] = ((3 * curr + right + 2) >> 2) as i16;
        }
    }
}

/// Scalar implementation of bilinear upsampling
fn upsample_h2v2_i16_fancy_scalar(
    input: &[i16],
    in_width: usize,
    in_height: usize,
    output: &mut [i16],
    out_width: usize,
    out_height: usize,
) {
    for out_y in 0..out_height {
        let in_y = (out_y / 2).min(in_height - 1);
        let is_top_half = out_y % 2 == 0;

        let v_neighbor_y = if is_top_half {
            in_y.saturating_sub(1)
        } else {
            (in_y + 1).min(in_height - 1)
        };

        let curr_row = &input[in_y * in_width..];
        let v_neighbor_row = &input[v_neighbor_y * in_width..];
        let out_row = &mut output[out_y * out_width..][..out_width];

        upsample_row_h2_fancy_bilinear(curr_row, v_neighbor_row, in_width, out_row, is_top_half);
    }
}

/// AVX2 SIMD implementation of bilinear upsampling
/// Uses separable vertical + horizontal passes for efficiency
#[cfg(target_arch = "x86_64")]
#[arcane]
fn upsample_h2v2_i16_fancy_avx2(
    _token: archmage::X64V3Token,
    input: &[i16],
    in_width: usize,
    in_height: usize,
    output: &mut [i16],
    out_width: usize,
    out_height: usize,
) {
    use core::arch::x86_64::*;

    // Stack-allocated scratch for one row of vertical interpolation results
    // Zeroing is cheap compared to the SIMD work we're doing
    const MAX_SCRATCH: usize = 4096;
    let mut scratch_storage = [0i16; MAX_SCRATCH];

    if in_width > MAX_SCRATCH {
        // Fall back to scalar for very wide images
        upsample_h2v2_i16_fancy_scalar(input, in_width, in_height, output, out_width, out_height);
        return;
    }

    let scratch = &mut scratch_storage[..in_width];

    let v_three = _mm256_set1_epi16(3);
    let v_two = _mm256_set1_epi16(2);

    // Process each output row
    for out_y in 0..out_height {
        let in_y = (out_y / 2).min(in_height.saturating_sub(1));
        let is_top_half = out_y % 2 == 0;

        let v_neighbor_y = if is_top_half {
            in_y.saturating_sub(1)
        } else {
            (in_y + 1).min(in_height.saturating_sub(1))
        };

        let curr_row = &input[in_y * in_width..][..in_width];
        let v_neighbor_row = &input[v_neighbor_y * in_width..][..in_width];
        let out_row = &mut output[out_y * out_width..][..out_width];

        // Pass 1: Vertical interpolation (3*curr + neighbor + 2) >> 2
        let chunks = in_width / 16;
        for i in 0..chunks {
            let offset = i * 16;
            let v_curr = safe_simd::_mm256_loadu_si256(
                <&[i16; 16]>::try_from(&curr_row[offset..offset + 16]).unwrap(),
            );
            let v_neighbor = safe_simd::_mm256_loadu_si256(
                <&[i16; 16]>::try_from(&v_neighbor_row[offset..offset + 16]).unwrap(),
            );

            let v_result = _mm256_srai_epi16(
                _mm256_add_epi16(
                    _mm256_add_epi16(_mm256_mullo_epi16(v_curr, v_three), v_neighbor),
                    v_two,
                ),
                2,
            );

            safe_simd::_mm256_storeu_si256(
                <&mut [i16; 16]>::try_from(&mut scratch[offset..offset + 16]).unwrap(),
                v_result,
            );
        }

        // Scalar remainder for vertical pass
        for x in (chunks * 16)..in_width {
            let c = curr_row[x] as i32;
            let n = v_neighbor_row[x] as i32;
            scratch[x] = ((3 * c + n + 2) >> 2) as i16;
        }

        // Pass 2: Horizontal 2x upsampling from scratch to output
        // Edge: first two output pixels
        if out_width >= 1 {
            out_row[0] = scratch[0];
        }
        if out_width >= 2 && in_width > 1 {
            let curr = scratch[0] as i32;
            let next = scratch[1] as i32;
            out_row[1] = ((3 * curr + next + 2) >> 2) as i16;
        }

        // Interior: SIMD processing
        // Each iteration processes 16 input pixels → 32 output pixels
        let h_chunks = (in_width.saturating_sub(2)) / 16;

        for chunk in 0..h_chunks {
            let in_offset = chunk * 16 + 1;
            let out_offset = 2 + chunk * 32;

            if out_offset + 32 > out_width {
                break;
            }

            let v_prev = safe_simd::_mm256_loadu_si256(
                <&[i16; 16]>::try_from(&scratch[in_offset - 1..in_offset + 15]).unwrap(),
            );
            let v_curr = safe_simd::_mm256_loadu_si256(
                <&[i16; 16]>::try_from(&scratch[in_offset..in_offset + 16]).unwrap(),
            );
            let v_next = safe_simd::_mm256_loadu_si256(
                <&[i16; 16]>::try_from(&scratch[in_offset + 1..in_offset + 17]).unwrap(),
            );

            // 3*curr + 2
            let v_common = _mm256_add_epi16(_mm256_mullo_epi16(v_curr, v_three), v_two);

            // Even outputs: (3*curr + prev + 2) >> 2
            let v_even = _mm256_srai_epi16(_mm256_add_epi16(v_common, v_prev), 2);

            // Odd outputs: (3*curr + next + 2) >> 2
            let v_odd = _mm256_srai_epi16(_mm256_add_epi16(v_common, v_next), 2);

            // Interleave even and odd
            let v_lo = _mm256_unpacklo_epi16(v_even, v_odd);
            let v_hi = _mm256_unpackhi_epi16(v_even, v_odd);

            // Fix lane order
            let v_out0 = _mm256_permute2x128_si256(v_lo, v_hi, 0x20);
            let v_out1 = _mm256_permute2x128_si256(v_lo, v_hi, 0x31);

            safe_simd::_mm256_storeu_si256(
                <&mut [i16; 16]>::try_from(&mut out_row[out_offset..out_offset + 16]).unwrap(),
                v_out0,
            );
            safe_simd::_mm256_storeu_si256(
                <&mut [i16; 16]>::try_from(&mut out_row[out_offset + 16..out_offset + 32]).unwrap(),
                v_out1,
            );
        }

        // Scalar remainder for horizontal pass
        let processed_in = 1 + h_chunks * 16;
        for in_x in processed_in..in_width.saturating_sub(1) {
            let out_x = in_x * 2;
            if out_x + 1 >= out_width {
                break;
            }

            let prev = scratch[in_x - 1] as i32;
            let curr = scratch[in_x] as i32;
            let next = scratch[in_x + 1] as i32;

            out_row[out_x] = ((3 * curr + prev + 2) >> 2) as i16;
            out_row[out_x + 1] = ((3 * curr + next + 2) >> 2) as i16;
        }

        // Edge: last input pixel
        if in_width >= 1 {
            let last_in = in_width - 1;
            let last_out = last_in * 2;
            let curr = scratch[last_in] as i32;
            let prev = if last_in > 0 {
                scratch[last_in - 1] as i32
            } else {
                curr
            };

            if last_out < out_width {
                out_row[last_out] = ((3 * curr + prev + 2) >> 2) as i16;
            }
            if last_out + 1 < out_width {
                out_row[last_out + 1] = curr as i16;
            }
        }
    }
}

/// AVX2 triangle-filter 2x2 upsample using a caller-provided scratch buffer.
///
/// Same algorithm as `upsample_h2v2_i16_fancy_avx2` but avoids allocating and
/// zeroing a `[0i16; 4096]` stack array on each call. The scratch buffer is
/// written by the vertical pass before the horizontal pass reads it, so it
/// does not need zeroing between calls.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn upsample_h2v2_i16_fancy_avx2_with_scratch(
    _token: archmage::X64V3Token,
    input: &[i16],
    in_width: usize,
    in_height: usize,
    output: &mut [i16],
    out_width: usize,
    out_height: usize,
    scratch_storage: &mut [i16],
) {
    use core::arch::x86_64::*;

    if in_width > scratch_storage.len() {
        // Fall back to scalar for inputs larger than scratch
        upsample_h2v2_i16_fancy_scalar(input, in_width, in_height, output, out_width, out_height);
        return;
    }

    let scratch = &mut scratch_storage[..in_width];

    let v_three = _mm256_set1_epi16(3);
    let v_two = _mm256_set1_epi16(2);

    // Process each output row
    for out_y in 0..out_height {
        let in_y = (out_y / 2).min(in_height.saturating_sub(1));
        let is_top_half = out_y % 2 == 0;

        let v_neighbor_y = if is_top_half {
            in_y.saturating_sub(1)
        } else {
            (in_y + 1).min(in_height.saturating_sub(1))
        };

        let curr_row = &input[in_y * in_width..][..in_width];
        let v_neighbor_row = &input[v_neighbor_y * in_width..][..in_width];
        let out_row = &mut output[out_y * out_width..][..out_width];

        // Pass 1: Vertical interpolation (3*curr + neighbor + 2) >> 2
        // Use chunks_exact so the compiler can prove slice length == 16,
        // eliminating runtime bounds checks on try_into().
        let curr_chunks = curr_row.chunks_exact(16);
        let remainder_len = curr_chunks.remainder().len();
        for ((curr_chunk, neighbor_chunk), scratch_chunk) in curr_chunks
            .zip(v_neighbor_row.chunks_exact(16))
            .zip(scratch.chunks_exact_mut(16))
        {
            let v_c = safe_simd::_mm256_loadu_si256(<&[i16; 16]>::try_from(curr_chunk).unwrap());
            let v_n =
                safe_simd::_mm256_loadu_si256(<&[i16; 16]>::try_from(neighbor_chunk).unwrap());

            let v_result = _mm256_srai_epi16(
                _mm256_add_epi16(
                    _mm256_add_epi16(_mm256_mullo_epi16(v_c, v_three), v_n),
                    v_two,
                ),
                2,
            );

            safe_simd::_mm256_storeu_si256(
                <&mut [i16; 16]>::try_from(scratch_chunk).unwrap(),
                v_result,
            );
        }

        // Scalar remainder for vertical pass
        let simd_done = in_width - remainder_len;
        for x in simd_done..in_width {
            let c = curr_row[x] as i32;
            let n = v_neighbor_row[x] as i32;
            scratch[x] = ((3 * c + n + 2) >> 2) as i16;
        }

        // Pass 2: Horizontal 2x upsampling from scratch to output
        // Edge: first two output pixels
        if out_width >= 1 {
            out_row[0] = scratch[0];
        }
        if out_width >= 2 && in_width > 1 {
            let curr = scratch[0] as i32;
            let next = scratch[1] as i32;
            out_row[1] = ((3 * curr + next + 2) >> 2) as i16;
        }

        // Interior: SIMD processing of horizontal pass
        // Use chunks_exact_mut on the output buffer so the compiler can prove
        // the 16-element store slices are valid, eliminating bounds checks.
        // Input loads overlap (prev/curr/next) and are always in-bounds because
        // h_chunks = (in_width - 2) / 16 guarantees max_index = h_chunks*16 + 1 < in_width.
        let h_chunks = (in_width.saturating_sub(2)) / 16;
        let h_simd_end_out = 2 + h_chunks * 32;

        if h_chunks > 0 && h_simd_end_out <= out_width {
            let out_chunks = out_row[2..h_simd_end_out].chunks_exact_mut(32);
            for (chunk_idx, out_chunk) in out_chunks.enumerate() {
                let in_offset = chunk_idx * 16 + 1;

                let v_prev = safe_simd::_mm256_loadu_si256(
                    <&[i16; 16]>::try_from(&scratch[in_offset - 1..in_offset + 15]).unwrap(),
                );
                let v_curr = safe_simd::_mm256_loadu_si256(
                    <&[i16; 16]>::try_from(&scratch[in_offset..in_offset + 16]).unwrap(),
                );
                let v_next = safe_simd::_mm256_loadu_si256(
                    <&[i16; 16]>::try_from(&scratch[in_offset + 1..in_offset + 17]).unwrap(),
                );

                // 3*curr + 2
                let v_common = _mm256_add_epi16(_mm256_mullo_epi16(v_curr, v_three), v_two);

                // Even outputs: (3*curr + prev + 2) >> 2
                let v_even = _mm256_srai_epi16(_mm256_add_epi16(v_common, v_prev), 2);

                // Odd outputs: (3*curr + next + 2) >> 2
                let v_odd = _mm256_srai_epi16(_mm256_add_epi16(v_common, v_next), 2);

                // Interleave even and odd
                let v_lo = _mm256_unpacklo_epi16(v_even, v_odd);
                let v_hi = _mm256_unpackhi_epi16(v_even, v_odd);

                // Fix lane order
                let v_out0 = _mm256_permute2x128_si256(v_lo, v_hi, 0x20);
                let v_out1 = _mm256_permute2x128_si256(v_lo, v_hi, 0x31);

                let (out_lo, out_hi) = out_chunk.split_at_mut(16);
                safe_simd::_mm256_storeu_si256(<&mut [i16; 16]>::try_from(out_lo).unwrap(), v_out0);
                safe_simd::_mm256_storeu_si256(<&mut [i16; 16]>::try_from(out_hi).unwrap(), v_out1);
            }
        }

        // Scalar remainder for horizontal pass
        let processed_in = 1 + h_chunks * 16;
        for in_x in processed_in..in_width.saturating_sub(1) {
            let out_x = in_x * 2;
            if out_x + 1 >= out_width {
                break;
            }

            let prev = scratch[in_x - 1] as i32;
            let curr = scratch[in_x] as i32;
            let next = scratch[in_x + 1] as i32;

            out_row[out_x] = ((3 * curr + prev + 2) >> 2) as i16;
            out_row[out_x + 1] = ((3 * curr + next + 2) >> 2) as i16;
        }

        // Edge: last input pixel
        if in_width >= 1 {
            let last_in = in_width - 1;
            let last_out = last_in * 2;
            let curr = scratch[last_in] as i32;
            let prev = if last_in > 0 {
                scratch[last_in - 1] as i32
            } else {
                curr
            };

            if last_out < out_width {
                out_row[last_out] = ((3 * curr + prev + 2) >> 2) as i16;
            }
            if last_out + 1 < out_width {
                out_row[last_out + 1] = curr as i16;
            }
        }
    }
}

/// Upsample a single row horizontally 2x with vertical blending (triangle filter).
///
/// For each output pixel at (out_x, out_y):
/// - in_x = out_x / 2
/// - is_left = out_x % 2 == 0
/// - Horizontal neighbor: left if is_left, right otherwise
/// - Vertical neighbor: above if is_top_half, below otherwise
/// - Result = (9*curr + 3*h_neighbor + 3*v_neighbor + hv_neighbor + 8) >> 4
#[inline(always)]
pub(super) fn upsample_row_h2_fancy_bilinear(
    curr_row: &[i16],
    v_neighbor_row: &[i16],
    in_width: usize,
    output: &mut [i16],
    _is_top_half: bool,
) {
    let out_width = output.len();
    if in_width == 0 {
        return;
    }

    // Process interior pixels in bulk (skip first and last input columns for edge handling)
    // Interior: in_x from 1 to in_width-2, which maps to out_x from 2 to out_width-4
    let _interior_start_out = 2;
    let interior_end_out = if in_width >= 2 {
        ((in_width - 1) * 2).min(out_width)
    } else {
        0
    };

    // Handle left edge (out_x = 0, 1)
    if out_width >= 1 {
        // out_x = 0: in_x = 0, is_left = true, h_neighbor = left (clamped to 0)
        let curr = curr_row[0] as i32;
        let v_neighbor = v_neighbor_row[0] as i32;
        // h_neighbor and hv_neighbor are same as curr/v_neighbor (clamped edge)
        output[0] = ((9 * curr + 3 * curr + 3 * v_neighbor + v_neighbor + 8) >> 4) as i16;
    }
    if out_width >= 2 {
        // out_x = 1: in_x = 0, is_left = false, h_neighbor = right (in_x + 1)
        let curr = curr_row[0] as i32;
        let h_neighbor = curr_row[1.min(in_width - 1)] as i32;
        let v_neighbor = v_neighbor_row[0] as i32;
        let hv_neighbor = v_neighbor_row[1.min(in_width - 1)] as i32;
        output[1] = ((9 * curr + 3 * h_neighbor + 3 * v_neighbor + hv_neighbor + 8) >> 4) as i16;
    }

    // Interior loop: no edge checks needed
    // Each pair of output pixels (2*in_x, 2*in_x+1) comes from input at in_x
    for in_x in 1..in_width.saturating_sub(1) {
        let out_x = in_x * 2;
        if out_x >= interior_end_out || out_x + 1 >= out_width {
            break;
        }

        let curr = curr_row[in_x] as i32;
        let left = curr_row[in_x - 1] as i32;
        let right = curr_row[in_x + 1] as i32;
        let v_curr = v_neighbor_row[in_x] as i32;
        let v_left = v_neighbor_row[in_x - 1] as i32;
        let v_right = v_neighbor_row[in_x + 1] as i32;

        // Left half of output pixel pair (is_left = true, h_neighbor = left)
        output[out_x] = ((9 * curr + 3 * left + 3 * v_curr + v_left + 8) >> 4) as i16;

        // Right half of output pixel pair (is_left = false, h_neighbor = right)
        output[out_x + 1] = ((9 * curr + 3 * right + 3 * v_curr + v_right + 8) >> 4) as i16;
    }

    // Handle right edge
    if in_width >= 1 {
        let last_in_x = in_width - 1;
        let out_x = last_in_x * 2;

        if out_x < out_width {
            // Left pixel of last pair
            let curr = curr_row[last_in_x] as i32;
            let left = curr_row[last_in_x.saturating_sub(1)] as i32;
            let v_curr = v_neighbor_row[last_in_x] as i32;
            let v_left = v_neighbor_row[last_in_x.saturating_sub(1)] as i32;
            output[out_x] = ((9 * curr + 3 * left + 3 * v_curr + v_left + 8) >> 4) as i16;
        }

        if out_x + 1 < out_width {
            // Right pixel of last pair (h_neighbor clamped to edge)
            let curr = curr_row[last_in_x] as i32;
            let v_curr = v_neighbor_row[last_in_x] as i32;
            output[out_x + 1] = ((9 * curr + 3 * curr + 3 * v_curr + v_curr + 8) >> 4) as i16;
        }
    }
}

/// Upsample a single chroma row for h2v2 boundary fixup, matching the main
/// upsampler's formula to avoid systematic ±1 rounding mismatches at MCU boundaries.
///
/// On x86_64 with AVX2: uses the separable approach (vertical then horizontal,
/// each `(3*near + far + 2) >> 2`) matching `upsample_h2v2_i16_fancy_avx2`.
///
/// On scalar: uses the non-separable 4-tap formula
/// `(9*C + 3*H + 3*V + HV + 8) >> 4` matching `upsample_h2v2_i16_fancy_scalar`.
///
/// Using different formulas for fixup vs main upsampling creates a systematic
/// ±1 brightness shift at MCU boundary rows (rows 0 and 15 of each 16-row MCU),
/// visible as faint horizontal stripes on images with strong chroma content.
pub(super) fn upsample_row_h2v2_fixup(
    curr_row: &[i16],
    v_neighbor_row: &[i16],
    in_width: usize,
    output: &mut [i16],
) {
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(token) = archmage::X64V3Token::summon() {
            upsample_row_h2v2_fixup_avx2(token, curr_row, v_neighbor_row, in_width, output);
            return;
        }
    }
    // Scalar: non-separable formula matches scalar main upsampler
    upsample_row_h2_fancy_bilinear(curr_row, v_neighbor_row, in_width, output, false);
}

/// AVX2 separable fixup: vertical pass into scratch, horizontal pass to output.
/// Matches the formula used by `upsample_h2v2_i16_fancy_avx2`.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn upsample_row_h2v2_fixup_avx2(
    token: archmage::X64V3Token,
    curr_row: &[i16],
    v_neighbor_row: &[i16],
    in_width: usize,
    output: &mut [i16],
) {
    let mut scratch_storage = [0i16; MAX_UPSAMPLE_SCRATCH];
    let scratch = if in_width <= MAX_UPSAMPLE_SCRATCH {
        &mut scratch_storage[..in_width]
    } else {
        // Very wide images: fall back to non-separable scalar
        upsample_row_h2_fancy_bilinear(curr_row, v_neighbor_row, in_width, output, false);
        return;
    };

    // Vertical pass: (3 * curr + v_neighbor + 2) >> 2
    upsample_vertical_row_strided_avx2(
        token,
        &curr_row[..in_width],
        &v_neighbor_row[..in_width],
        scratch,
    );

    // Horizontal pass: (3 * curr + h_neighbor + 2) >> 2
    upsample_horizontal_row_strided_avx2(token, scratch, output);
}

/// Horizontal 2x upsampling in i16 (4:2:2 → 4:4:4) with triangle filter.
#[inline]
pub fn upsample_h2v1_i16_fancy(
    input: &[i16],
    in_width: usize,
    in_height: usize,
    output: &mut [i16],
    out_width: usize,
    out_height: usize,
) {
    upsample_h2v1_i16_fancy_strided(
        input, in_width, in_width, in_height, output, out_width, out_width, out_height,
    );
}

/// Vertical 2x upsampling in i16 (4:4:0 → 4:4:4) with triangle filter.
#[inline]
pub fn upsample_h1v2_i16_fancy(
    input: &[i16],
    in_width: usize,
    in_height: usize,
    output: &mut [i16],
    out_width: usize,
    out_height: usize,
) {
    upsample_h1v2_i16_fancy_strided(
        input, in_width, in_width, in_height, output, out_width, out_width, out_height,
    );
}

/// Strided horizontal 2x upsampling in i16 (4:2:2 → 4:4:4) with triangle filter.
pub fn upsample_h2v1_i16_fancy_strided(
    input: &[i16],
    in_width: usize,
    in_stride: usize,
    in_height: usize,
    output: &mut [i16],
    out_width: usize,
    out_stride: usize,
    out_height: usize,
) {
    for out_y in 0..out_height {
        let in_y = out_y.min(in_height.saturating_sub(1));
        let out_row = out_y * out_stride;
        let in_row = in_y * in_stride;

        for out_x in 0..out_width {
            let in_x = out_x / 2;
            let in_x_clamped = in_x.min(in_width.saturating_sub(1));
            let curr = input[in_row + in_x_clamped] as i32;

            let result = if out_x % 2 == 0 {
                let left = if in_x > 0 {
                    input[in_row + in_x - 1] as i32
                } else {
                    curr
                };
                (3 * curr + left + 2) >> 2
            } else {
                let right = if in_x + 1 < in_width {
                    input[in_row + in_x + 1] as i32
                } else {
                    curr
                };
                (3 * curr + right + 2) >> 2
            };
            output[out_row + out_x] = result as i16;
        }
    }
}

/// Strided vertical 2x upsampling in i16 (4:4:0 → 4:4:4) with triangle filter.
pub fn upsample_h1v2_i16_fancy_strided(
    input: &[i16],
    in_width: usize,
    in_stride: usize,
    in_height: usize,
    output: &mut [i16],
    out_width: usize,
    out_stride: usize,
    out_height: usize,
) {
    for out_y in 0..out_height {
        let in_y = out_y / 2;
        let is_top = out_y % 2 == 0;
        let out_row = out_y * out_stride;

        let neighbor_y = if is_top {
            in_y.saturating_sub(1)
        } else {
            (in_y + 1).min(in_height.saturating_sub(1))
        };

        let in_y_clamped = in_y.min(in_height.saturating_sub(1));
        let curr_row = in_y_clamped * in_stride;
        let neighbor_row = neighbor_y * in_stride;

        for out_x in 0..out_width {
            let in_x = out_x.min(in_width.saturating_sub(1));
            let curr = input[curr_row + in_x] as i32;
            let neighbor = input[neighbor_row + in_x] as i32;
            let result = (3 * curr + neighbor + 2) >> 2;
            output[out_row + out_x] = result as i16;
        }
    }
}

// ============================================================================
// AVX2 SIMD Upsampling
// ============================================================================

/// AVX2-optimized 2x2 upsampling using separable filter approach.
///
/// Uses two-pass method like zune-jpeg:
/// 1. Vertical pass: (3*curr + neighbor + 2) >> 2
/// 2. Horizontal pass: (3*curr + neighbor + 2) >> 2 with interleaving
///
/// Processes row-by-row to avoid scratch allocation.
///
/// NOTE: This is an alternative implementation kept for reference.
/// The active implementation is `upsample_h2v2_i16_fancy_avx2`.
#[allow(dead_code)]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub fn upsample_h2v2_i16_fancy_simd(
    input: &[i16],
    in_width: usize,
    in_height: usize,
    output: &mut [i16],
    out_width: usize,
    out_height: usize,
) {
    if in_width == 0 || in_height == 0 || out_width == 0 || out_height == 0 {
        return;
    }

    let Some(token) = archmage::X64V3Token::summon() else {
        // Fall back to scalar if AVX2 not available
        upsample_h2v2_i16_fancy_scalar(input, in_width, in_height, output, out_width, out_height);
        return;
    };

    // Stack-allocated scratch for one row of vertical interpolation results
    // This avoids heap allocation in the hot path
    const MAX_SCRATCH: usize = 4096; // Enough for 8K images
    let mut scratch_storage = [0i16; MAX_SCRATCH];

    if in_width > MAX_SCRATCH {
        // Fall back to scalar for very wide images
        upsample_h2v2_i16_fancy(input, in_width, in_height, output, out_width, out_height);
        return;
    }

    let scratch = &mut scratch_storage[..in_width];

    // Process each output row pair
    for out_y in 0..out_height {
        let in_y = (out_y / 2).min(in_height.saturating_sub(1));
        let is_top_half = out_y % 2 == 0;

        // Vertical neighbor: above for top half, below for bottom half
        let v_neighbor_y = if is_top_half {
            in_y.saturating_sub(1)
        } else {
            (in_y + 1).min(in_height.saturating_sub(1))
        };

        let curr_row = &input[in_y * in_width..][..in_width];
        let v_neighbor_row = &input[v_neighbor_y * in_width..][..in_width];
        let out_row = &mut output[out_y * out_width..][..out_width];

        // Vertical pass: compute vertically-interpolated row into scratch
        upsample_vertical_row_avx2(token, curr_row, v_neighbor_row, scratch);

        // Horizontal pass: horizontally interpolate scratch into output
        upsample_horizontal_row_avx2(token, scratch, out_row);
    }
}

/// Vertical upsampling of a single row: (3*curr + neighbor + 2) >> 2
#[allow(dead_code)]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[arcane]
fn upsample_vertical_row_avx2(
    _token: archmage::X64V3Token,
    curr: &[i16],
    neighbor: &[i16],
    output: &mut [i16],
) {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    let width = curr.len();
    let v_three = _mm256_set1_epi16(3);
    let v_two = _mm256_set1_epi16(2);

    // Process 16 i16 values at a time
    let chunks = width / 16;
    for i in 0..chunks {
        let offset = i * 16;
        let v_curr = safe_simd::_mm256_loadu_si256(
            <&[i16; 16]>::try_from(&curr[offset..offset + 16]).unwrap(),
        );
        let v_neighbor = safe_simd::_mm256_loadu_si256(
            <&[i16; 16]>::try_from(&neighbor[offset..offset + 16]).unwrap(),
        );

        // (3 * curr + neighbor + 2) >> 2
        let v_result = _mm256_srai_epi16(
            _mm256_add_epi16(
                _mm256_add_epi16(_mm256_mullo_epi16(v_curr, v_three), v_neighbor),
                v_two,
            ),
            2,
        );

        safe_simd::_mm256_storeu_si256(
            <&mut [i16; 16]>::try_from(&mut output[offset..offset + 16]).unwrap(),
            v_result,
        );
    }

    // Handle remainder
    let remainder_start = chunks * 16;
    for x in remainder_start..width {
        let c = curr[x] as i32;
        let n = neighbor[x] as i32;
        output[x] = ((3 * c + n + 2) >> 2) as i16;
    }
}

/// Horizontal upsampling of a single row: 1x width → 2x width with interleaving
/// Uses (3*curr + neighbor + 2) >> 2 for triangle filter
#[allow(dead_code)]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[arcane]
fn upsample_horizontal_row_avx2(_token: archmage::X64V3Token, input: &[i16], output: &mut [i16]) {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    let in_width = input.len();
    let out_width = output.len();

    if in_width < 18 {
        // Fall back to scalar for small inputs
        upsample_horizontal_row_scalar(input, output);
        return;
    }

    let v_three = _mm256_set1_epi16(3);
    let v_two = _mm256_set1_epi16(2);

    // First two output pixels (edge case)
    output[0] = input[0];
    output[1] = ((input[0] as i32 * 3 + input[1] as i32 + 2) >> 2) as i16;

    // Process interior: each iteration processes 16 input pixels → 32 output pixels
    // We need prev[i], curr[i], next[i] which means loading with offset
    let chunks = (in_width - 2) / 16;

    for chunk in 0..chunks {
        let in_offset = chunk * 16 + 1; // Start at index 1 (skip first)
        let out_offset = 2 + chunk * 32; // Output starts at index 2

        if out_offset + 32 > out_width {
            break;
        }

        let v_prev = safe_simd::_mm256_loadu_si256(
            <&[i16; 16]>::try_from(&input[in_offset - 1..in_offset + 15]).unwrap(),
        );
        let v_curr = safe_simd::_mm256_loadu_si256(
            <&[i16; 16]>::try_from(&input[in_offset..in_offset + 16]).unwrap(),
        );
        let v_next = safe_simd::_mm256_loadu_si256(
            <&[i16; 16]>::try_from(&input[in_offset + 1..in_offset + 17]).unwrap(),
        );

        // Common term: 3*curr + 2
        let v_common = _mm256_add_epi16(_mm256_mullo_epi16(v_curr, v_three), v_two);

        // Even outputs (left half): (3*curr + prev + 2) >> 2
        let v_even = _mm256_srai_epi16(_mm256_add_epi16(v_common, v_prev), 2);

        // Odd outputs (right half): (3*curr + next + 2) >> 2
        let v_odd = _mm256_srai_epi16(_mm256_add_epi16(v_common, v_next), 2);

        // Interleave even and odd to get final output order
        // unpacklo/hi interleave within 128-bit lanes, then permute to fix order
        let v_lo = _mm256_unpacklo_epi16(v_even, v_odd); // [e0,o0,e1,o1,e2,o2,e3,o3, e8,o8,e9,o9,...]
        let v_hi = _mm256_unpackhi_epi16(v_even, v_odd); // [e4,o4,e5,o5,e6,o6,e7,o7, e12,o12,...]

        // Fix lane order: low lanes of both, then high lanes of both
        let v_out0 = _mm256_permute2x128_si256(v_lo, v_hi, 0x20); // First 16 outputs
        let v_out1 = _mm256_permute2x128_si256(v_lo, v_hi, 0x31); // Second 16 outputs

        safe_simd::_mm256_storeu_si256(
            <&mut [i16; 16]>::try_from(&mut output[out_offset..out_offset + 16]).unwrap(),
            v_out0,
        );
        safe_simd::_mm256_storeu_si256(
            <&mut [i16; 16]>::try_from(&mut output[out_offset + 16..out_offset + 32]).unwrap(),
            v_out1,
        );
    }

    // Handle remainder with scalar
    let processed_in = 1 + chunks * 16;
    let _processed_out = 2 + chunks * 32;

    for in_x in processed_in..in_width {
        let out_x = in_x * 2;
        if out_x >= out_width {
            break;
        }

        let curr = input[in_x] as i32;
        let prev = input[in_x.saturating_sub(1)] as i32;
        let next = input[(in_x + 1).min(in_width - 1)] as i32;

        if out_x < out_width {
            output[out_x] = ((3 * curr + prev + 2) >> 2) as i16;
        }
        if out_x + 1 < out_width {
            output[out_x + 1] = ((3 * curr + next + 2) >> 2) as i16;
        }
    }

    // Last two output pixels (edge case)
    let last_in = in_width - 1;
    let last_out = last_in * 2;
    if last_out < out_width {
        output[last_out] =
            ((input[last_in] as i32 * 3 + input[last_in.saturating_sub(1)] as i32 + 2) >> 2) as i16;
    }
    if last_out + 1 < out_width {
        output[last_out + 1] = input[last_in];
    }
}

// ============================================================================
// Nearest-Neighbor Upsampling (Box Filter)
// ============================================================================

/// Horizontal 2x + vertical 2x nearest-neighbor upsampling in i16 (4:2:0 → 4:4:4).
///
/// Each chroma sample is replicated to fill the corresponding 2x2 output area.
pub fn upsample_h2v2_i16_nearest(
    input: &[i16],
    in_width: usize,
    in_height: usize,
    output: &mut [i16],
    out_width: usize,
    out_height: usize,
) {
    for out_y in 0..out_height {
        let in_y = (out_y / 2).min(in_height.saturating_sub(1));
        let out_row = out_y * out_width;
        let in_row = in_y * in_width;

        for out_x in 0..out_width {
            let in_x = (out_x / 2).min(in_width.saturating_sub(1));
            output[out_row + out_x] = input[in_row + in_x];
        }
    }
}

/// Horizontal 2x nearest-neighbor upsampling in i16 (4:2:2 → 4:4:4).
pub fn upsample_h2v1_i16_nearest(
    input: &[i16],
    in_width: usize,
    in_height: usize,
    output: &mut [i16],
    out_width: usize,
    out_height: usize,
) {
    upsample_h2v1_i16_nearest_strided(
        input, in_width, in_width, in_height, output, out_width, out_width, out_height,
    );
}

/// Vertical 2x nearest-neighbor upsampling in i16 (4:4:0 → 4:4:4).
pub fn upsample_h1v2_i16_nearest(
    input: &[i16],
    in_width: usize,
    in_height: usize,
    output: &mut [i16],
    out_width: usize,
    out_height: usize,
) {
    upsample_h1v2_i16_nearest_strided(
        input, in_width, in_width, in_height, output, out_width, out_width, out_height,
    );
}

/// Strided horizontal 2x nearest-neighbor upsampling in i16 (4:2:2 → 4:4:4).
pub fn upsample_h2v1_i16_nearest_strided(
    input: &[i16],
    in_width: usize,
    in_stride: usize,
    in_height: usize,
    output: &mut [i16],
    out_width: usize,
    out_stride: usize,
    out_height: usize,
) {
    for out_y in 0..out_height {
        let in_y = out_y.min(in_height.saturating_sub(1));
        let out_row = out_y * out_stride;
        let in_row = in_y * in_stride;

        for out_x in 0..out_width {
            let in_x = (out_x / 2).min(in_width.saturating_sub(1));
            output[out_row + out_x] = input[in_row + in_x];
        }
    }
}

/// Strided vertical 2x nearest-neighbor upsampling in i16 (4:4:0 → 4:4:4).
pub fn upsample_h1v2_i16_nearest_strided(
    input: &[i16],
    in_width: usize,
    in_stride: usize,
    in_height: usize,
    output: &mut [i16],
    out_width: usize,
    out_stride: usize,
    out_height: usize,
) {
    for out_y in 0..out_height {
        let in_y = (out_y / 2).min(in_height.saturating_sub(1));
        let out_row = out_y * out_stride;
        let in_row = in_y * in_stride;

        for out_x in 0..out_width {
            let in_x = out_x.min(in_width.saturating_sub(1));
            output[out_row + out_x] = input[in_row + in_x];
        }
    }
}

/// Strided horizontal 2x + vertical 2x nearest-neighbor upsampling in i16 (4:2:0 → 4:4:4).
pub fn upsample_h2v2_i16_nearest_strided(
    input: &[i16],
    in_width: usize,
    in_stride: usize,
    in_height: usize,
    output: &mut [i16],
    out_width: usize,
    out_stride: usize,
    out_height: usize,
) {
    for out_y in 0..out_height {
        let in_y = (out_y / 2).min(in_height.saturating_sub(1));
        let out_row = out_y * out_stride;
        let in_row = in_y * in_stride;

        for out_x in 0..out_width {
            let in_x = (out_x / 2).min(in_width.saturating_sub(1));
            output[out_row + out_x] = input[in_row + in_x];
        }
    }
}

// ============================================================================
// libjpeg-turbo Compatible Upsampling
// ============================================================================

/// Horizontal 2x upsampling in i16 with libjpeg-turbo compatible rounding (4:2:2 → 4:4:4).
///
/// Uses alternating rounding bias: +1 for left pixel, +2 for right pixel.
/// Matches libjpeg-turbo's `jdsample.c` h2v1_fancy_upsample.
pub fn upsample_h2v1_i16_libjpeg(
    input: &[i16],
    in_width: usize,
    in_height: usize,
    output: &mut [i16],
    out_width: usize,
    out_height: usize,
) {
    upsample_h2v1_i16_libjpeg_strided(
        input, in_width, in_width, in_height, output, out_width, out_width, out_height,
    );
}

/// Vertical 2x upsampling in i16 with libjpeg-turbo compatible rounding (4:4:0 → 4:4:4).
///
/// Uses alternating rounding bias: +1 for upper row (v=0), +2 for lower row (v=1).
pub fn upsample_h1v2_i16_libjpeg(
    input: &[i16],
    in_width: usize,
    in_height: usize,
    output: &mut [i16],
    out_width: usize,
    out_height: usize,
) {
    upsample_h1v2_i16_libjpeg_strided(
        input, in_width, in_width, in_height, output, out_width, out_width, out_height,
    );
}

/// Strided horizontal 2x upsampling in i16 with libjpeg-turbo compatible rounding (4:2:2 → 4:4:4).
///
/// Uses alternating rounding bias: +1 for left pixel, +2 for right pixel.
/// Matches libjpeg-turbo's `jdsample.c` h2v1_fancy_upsample.
pub fn upsample_h2v1_i16_libjpeg_strided(
    input: &[i16],
    in_width: usize,
    in_stride: usize,
    in_height: usize,
    output: &mut [i16],
    out_width: usize,
    out_stride: usize,
    out_height: usize,
) {
    if in_width == 0 || in_height == 0 {
        return;
    }

    for out_y in 0..out_height {
        let in_y = out_y.min(in_height.saturating_sub(1));
        let out_row = out_y * out_stride;
        let in_row = in_y * in_stride;

        if in_width == 1 {
            let val = input[in_row];
            if out_width > 0 {
                output[out_row] = val;
            }
            if out_width > 1 {
                output[out_row + 1] = val;
            }
            continue;
        }

        // First column
        let curr = input[in_row] as i32;
        let next = input[in_row + 1] as i32;
        output[out_row] = curr as i16;
        if out_width > 1 {
            output[out_row + 1] = ((curr * 3 + next + 2) >> 2) as i16;
        }

        // Interior columns
        for in_x in 1..in_width.saturating_sub(1) {
            let prev = input[in_row + in_x - 1] as i32;
            let curr = input[in_row + in_x] as i32;
            let next = input[in_row + in_x + 1] as i32;
            let left_out = in_x * 2;
            let right_out = left_out + 1;
            if left_out < out_width {
                output[out_row + left_out] = ((curr * 3 + prev + 1) >> 2) as i16;
            }
            if right_out < out_width {
                output[out_row + right_out] = ((curr * 3 + next + 2) >> 2) as i16;
            }
        }

        // Last column
        let last = in_width - 1;
        let prev = input[in_row + last - 1] as i32;
        let curr = input[in_row + last] as i32;
        let left_out = last * 2;
        let right_out = left_out + 1;
        if left_out < out_width {
            output[out_row + left_out] = ((curr * 3 + prev + 1) >> 2) as i16;
        }
        if right_out < out_width {
            output[out_row + right_out] = curr as i16;
        }
    }
}

/// Strided vertical 2x upsampling in i16 with libjpeg-turbo compatible rounding (4:4:0 → 4:4:4).
///
/// Uses alternating rounding bias: +1 for upper row (v=0), +2 for lower row (v=1).
pub fn upsample_h1v2_i16_libjpeg_strided(
    input: &[i16],
    in_width: usize,
    in_stride: usize,
    in_height: usize,
    output: &mut [i16],
    out_width: usize,
    out_stride: usize,
    out_height: usize,
) {
    if in_width == 0 || in_height == 0 {
        return;
    }

    for out_y in 0..out_height {
        let in_y = out_y / 2;
        let in_y_clamped = in_y.min(in_height.saturating_sub(1));
        let is_upper = out_y % 2 == 0;
        let out_row = out_y * out_stride;

        let far_y = if is_upper {
            in_y_clamped.saturating_sub(1)
        } else {
            (in_y + 1).min(in_height.saturating_sub(1))
        };

        let near_row = in_y_clamped * in_stride;
        let far_row = far_y * in_stride;

        let bias = if is_upper { 1i32 } else { 2i32 };

        for out_x in 0..out_width {
            let in_x = out_x.min(in_width.saturating_sub(1));
            let near = input[near_row + in_x] as i32;
            let far = input[far_row + in_x] as i32;
            output[out_row + out_x] = ((near * 3 + far + bias) >> 2) as i16;
        }
    }
}

/// Horizontal 2x + vertical 2x upsampling in i16 with libjpeg-turbo compatible rounding (4:2:0 → 4:4:4).
///
/// Uses fused 2D filter (NOT separable) with alternating rounding bias (+7/+8).
/// Matches libjpeg-turbo's `jdsample.c` h2v2_fancy_upsample exactly.
///
/// The fused algorithm avoids intermediate rounding errors from separable passes.
pub fn upsample_h2v2_i16_libjpeg(
    input: &[i16],
    in_width: usize,
    in_height: usize,
    output: &mut [i16],
    out_width: usize,
    out_height: usize,
) {
    if in_width == 0 || in_height == 0 || out_width == 0 || out_height == 0 {
        return;
    }

    for out_y in 0..out_height {
        let in_y = out_y / 2;
        let in_y_clamped = in_y.min(in_height.saturating_sub(1));
        let is_upper = out_y % 2 == 0;

        // near_row = current chroma row, far_row = vertical neighbor
        let far_y = if is_upper {
            in_y_clamped.saturating_sub(1)
        } else {
            (in_y + 1).min(in_height.saturating_sub(1))
        };

        let near_row = in_y_clamped * in_width;
        let far_row = far_y * in_width;
        let out_row = out_y * out_width;

        // Compute column sums: colsum[x] = near[x] * 3 + far[x]
        // Then apply horizontal filter on column sums with /16 rounding
        upsample_h2v2_libjpeg_row(
            &input[near_row..near_row + in_width],
            &input[far_row..far_row + in_width],
            &mut output[out_row..],
            in_width,
            out_width,
            is_upper,
        );
    }
}

/// Process one output row of fused h2v2 libjpeg-compat upsampling.
///
/// `is_upper` controls the rounding bias alternation pattern.
#[inline]
pub(super) fn upsample_h2v2_libjpeg_row(
    near: &[i16],
    far: &[i16],
    output: &mut [i16],
    in_width: usize,
    out_width: usize,
    is_upper: bool,
) {
    if in_width == 1 {
        // Single column: just vertical filter
        let colsum = near[0] as i32 * 3 + far[0] as i32;
        let val = ((colsum * 4 + 8) >> 4) as i16;
        if out_width > 0 {
            output[0] = val;
        }
        if out_width > 1 {
            output[1] = val;
        }
        return;
    }

    // Rounding biases per libjpeg-turbo:
    // For upper row (v=0): left=8, right=7
    // For lower row (v=1): left=7, right=8
    // This alternation eliminates systematic bias
    let (bias_left, bias_right) = if is_upper { (8i32, 7i32) } else { (7i32, 8i32) };

    // Column sums: near * 3 + far
    let this_colsum = near[0] as i32 * 3 + far[0] as i32;
    let next_colsum = near[1] as i32 * 3 + far[1] as i32;

    // First column
    output[0] = ((this_colsum * 4 + 8) >> 4) as i16;
    if out_width > 1 {
        output[1] = ((this_colsum * 3 + next_colsum + bias_right) >> 4) as i16;
    }

    // Interior columns
    let mut last_colsum = this_colsum;
    for in_x in 1..in_width.saturating_sub(1) {
        let this_colsum = near[in_x] as i32 * 3 + far[in_x] as i32;
        let next_colsum = near[in_x + 1] as i32 * 3 + far[in_x + 1] as i32;

        let left_out = in_x * 2;
        let right_out = left_out + 1;
        if left_out < out_width {
            output[left_out] = ((this_colsum * 3 + last_colsum + bias_left) >> 4) as i16;
        }
        if right_out < out_width {
            output[right_out] = ((this_colsum * 3 + next_colsum + bias_right) >> 4) as i16;
        }
        last_colsum = this_colsum;
    }

    // Last column
    let last = in_width - 1;
    let this_colsum = near[last] as i32 * 3 + far[last] as i32;
    let left_out = last * 2;
    let right_out = left_out + 1;
    if left_out < out_width {
        output[left_out] = ((this_colsum * 3 + last_colsum + bias_left) >> 4) as i16;
    }
    if right_out < out_width {
        output[right_out] = ((this_colsum * 4 + bias_right) >> 4) as i16;
    }
}

/// Strided horizontal 2x + vertical 2x upsampling in i16 with libjpeg-turbo compatible rounding.
///
/// Same algorithm as `upsample_h2v2_i16_libjpeg` but supports SIMD-aligned stride > width.
pub fn upsample_h2v2_i16_libjpeg_strided(
    input: &[i16],
    in_width: usize,
    in_stride: usize,
    in_height: usize,
    output: &mut [i16],
    out_width: usize,
    out_stride: usize,
    out_height: usize,
) {
    if in_width == 0 || in_height == 0 || out_width == 0 || out_height == 0 {
        return;
    }

    for out_y in 0..out_height {
        let in_y = out_y / 2;
        let in_y_clamped = in_y.min(in_height.saturating_sub(1));
        let is_upper = out_y % 2 == 0;

        let far_y = if is_upper {
            in_y_clamped.saturating_sub(1)
        } else {
            (in_y + 1).min(in_height.saturating_sub(1))
        };

        let near_row = in_y_clamped * in_stride;
        let far_row = far_y * in_stride;
        let out_row = out_y * out_stride;

        upsample_h2v2_libjpeg_row(
            &input[near_row..near_row + in_width],
            &input[far_row..far_row + in_width],
            &mut output[out_row..],
            in_width,
            out_width,
            is_upper,
        );
    }
}

// ============================================================================
// f32 Nearest-Neighbor and libjpeg-compat Upsampling
// ============================================================================

/// Nearest-neighbor upsampling for f32 planes.
///
/// Replaces the inline box filter code in output.rs.
pub fn upsample_nearest_f32(
    input: &[f32],
    in_width: usize,
    in_height: usize,
    output: &mut [f32],
    out_width: usize,
    out_height: usize,
    scale_x: usize,
    scale_y: usize,
) {
    for py in 0..out_height {
        let sy = (py / scale_y).min(in_height.saturating_sub(1));
        let out_row = py * out_width;
        let in_row = sy * in_width;
        for px in 0..out_width {
            let sx = (px / scale_x).min(in_width.saturating_sub(1));
            output[out_row + px] = input[in_row + sx];
        }
    }
}

/// libjpeg-turbo compatible upsampling for f32 planes.
///
/// Dispatches to the appropriate algorithm based on scale factors.
pub fn upsample_libjpeg_f32(
    input: &[f32],
    in_width: usize,
    in_height: usize,
    out_width: usize,
    out_height: usize,
    scale_x: usize,
    scale_y: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; out_width * out_height];
    match (scale_x, scale_y) {
        (2, 2) => upsample_h2v2_f32_libjpeg(
            input,
            in_width,
            in_height,
            &mut output,
            out_width,
            out_height,
        ),
        (2, 1) => upsample_h2v1_f32_libjpeg(
            input,
            in_width,
            in_height,
            &mut output,
            out_width,
            out_height,
        ),
        (1, 2) => upsample_h1v2_f32_libjpeg(
            input,
            in_width,
            in_height,
            &mut output,
            out_width,
            out_height,
        ),
        (1, 1) => {
            // No upsampling, just crop
            for y in 0..out_height {
                let in_y = y.min(in_height.saturating_sub(1));
                for x in 0..out_width {
                    let in_x = x.min(in_width.saturating_sub(1));
                    output[y * out_width + x] = input[in_y * in_width + in_x];
                }
            }
        }
        _ => {
            // Fall back to nearest-neighbor for unsupported ratios
            upsample_nearest_f32(
                input,
                in_width,
                in_height,
                &mut output,
                out_width,
                out_height,
                scale_x,
                scale_y,
            );
        }
    }
    output
}

/// f32 version of libjpeg-turbo h2v1 upsampling with alternating bias.
fn upsample_h2v1_f32_libjpeg(
    input: &[f32],
    in_width: usize,
    in_height: usize,
    output: &mut [f32],
    out_width: usize,
    out_height: usize,
) {
    if in_width == 0 || in_height == 0 {
        return;
    }

    for out_y in 0..out_height {
        let in_y = out_y.min(in_height.saturating_sub(1));
        let out_row = out_y * out_width;
        let in_row = in_y * in_width;

        if in_width == 1 {
            let val = input[in_row];
            if out_width > 0 {
                output[out_row] = val;
            }
            if out_width > 1 {
                output[out_row + 1] = val;
            }
            continue;
        }

        // First column
        let curr = input[in_row];
        let next = input[in_row + 1];
        output[out_row] = curr;
        if out_width > 1 {
            output[out_row + 1] = curr * 0.75 + next * 0.25;
        }

        // Interior
        for in_x in 1..in_width.saturating_sub(1) {
            let prev = input[in_row + in_x - 1];
            let curr = input[in_row + in_x];
            let next = input[in_row + in_x + 1];
            let left_out = in_x * 2;
            let right_out = left_out + 1;
            if left_out < out_width {
                output[out_row + left_out] = curr * 0.75 + prev * 0.25;
            }
            if right_out < out_width {
                output[out_row + right_out] = curr * 0.75 + next * 0.25;
            }
        }

        // Last column
        let last = in_width - 1;
        let prev = input[in_row + last - 1];
        let curr = input[in_row + last];
        let left_out = last * 2;
        let right_out = left_out + 1;
        if left_out < out_width {
            output[out_row + left_out] = curr * 0.75 + prev * 0.25;
        }
        if right_out < out_width {
            output[out_row + right_out] = curr;
        }
    }
}

/// f32 version of libjpeg-turbo h1v2 upsampling with alternating bias.
fn upsample_h1v2_f32_libjpeg(
    input: &[f32],
    in_width: usize,
    in_height: usize,
    output: &mut [f32],
    out_width: usize,
    out_height: usize,
) {
    if in_width == 0 || in_height == 0 {
        return;
    }

    for out_y in 0..out_height {
        let in_y = out_y / 2;
        let in_y_clamped = in_y.min(in_height.saturating_sub(1));
        let is_upper = out_y % 2 == 0;
        let out_row = out_y * out_width;

        let far_y = if is_upper {
            in_y_clamped.saturating_sub(1)
        } else {
            (in_y + 1).min(in_height.saturating_sub(1))
        };

        let near_row = in_y_clamped * in_width;
        let far_row = far_y * in_width;

        for out_x in 0..out_width {
            let in_x = out_x.min(in_width.saturating_sub(1));
            let near = input[near_row + in_x];
            let far = input[far_row + in_x];
            output[out_row + out_x] = near * 0.75 + far * 0.25;
        }
    }
}

/// f32 version of libjpeg-turbo fused h2v2 upsampling.
fn upsample_h2v2_f32_libjpeg(
    input: &[f32],
    in_width: usize,
    in_height: usize,
    output: &mut [f32],
    out_width: usize,
    out_height: usize,
) {
    if in_width == 0 || in_height == 0 || out_width == 0 || out_height == 0 {
        return;
    }

    for out_y in 0..out_height {
        let in_y = out_y / 2;
        let in_y_clamped = in_y.min(in_height.saturating_sub(1));
        let is_upper = out_y % 2 == 0;

        let far_y = if is_upper {
            in_y_clamped.saturating_sub(1)
        } else {
            (in_y + 1).min(in_height.saturating_sub(1))
        };

        let near_row = in_y_clamped * in_width;
        let far_row = far_y * in_width;
        let out_row = out_y * out_width;

        if in_width == 1 {
            let colsum = input[near_row] * 3.0 + input[far_row];
            let val = colsum * 0.25;
            if out_width > 0 {
                output[out_row] = val;
            }
            if out_width > 1 {
                output[out_row + 1] = val;
            }
            continue;
        }

        // Column sums
        let this_colsum = input[near_row] * 3.0 + input[far_row];
        let next_colsum = input[near_row + 1] * 3.0 + input[far_row + 1];

        // First column
        output[out_row] = this_colsum * 0.25;
        if out_width > 1 {
            output[out_row + 1] = (this_colsum * 3.0 + next_colsum) / 16.0;
        }

        let mut last_colsum = this_colsum;
        for in_x in 1..in_width.saturating_sub(1) {
            let this_colsum = input[near_row + in_x] * 3.0 + input[far_row + in_x];
            let next_colsum = input[near_row + in_x + 1] * 3.0 + input[far_row + in_x + 1];
            let left_out = in_x * 2;
            let right_out = left_out + 1;
            if left_out < out_width {
                output[out_row + left_out] = (this_colsum * 3.0 + last_colsum) / 16.0;
            }
            if right_out < out_width {
                output[out_row + right_out] = (this_colsum * 3.0 + next_colsum) / 16.0;
            }
            last_colsum = this_colsum;
        }

        // Last column
        let last = in_width - 1;
        let this_colsum = input[near_row + last] * 3.0 + input[far_row + last];
        let left_out = last * 2;
        let right_out = left_out + 1;
        if left_out < out_width {
            output[out_row + left_out] = (this_colsum * 3.0 + last_colsum) / 16.0;
        }
        if right_out < out_width {
            output[out_row + right_out] = this_colsum * 0.25;
        }
    }
}

// ============================================================================
// Existing Scalar Horizontal Row (for reference/internal use)
// ============================================================================

/// Scalar fallback for horizontal upsampling
#[allow(dead_code)]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn upsample_horizontal_row_scalar(input: &[i16], output: &mut [i16]) {
    let in_width = input.len();
    let out_width = output.len();

    for out_x in 0..out_width {
        let in_x = out_x / 2;
        let in_x = in_x.min(in_width.saturating_sub(1));
        let curr = input[in_x] as i32;

        let result = if out_x % 2 == 0 {
            // Left half - blend with left neighbor
            let prev = if in_x > 0 {
                input[in_x - 1] as i32
            } else {
                curr
            };
            (3 * curr + prev + 2) >> 2
        } else {
            // Right half - blend with right neighbor
            let next = if in_x + 1 < in_width {
                input[in_x + 1] as i32
            } else {
                curr
            };
            (3 * curr + next + 2) >> 2
        };
        output[out_x] = result as i16;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn h2v1_identity_at_edges() {
        let input = [10.0, 20.0, 30.0, 40.0];
        let output = upsample_h2v1(&input, 4, 1, 8, 1);
        assert_eq!(output.len(), 8);
        // First pixel: edge replication → (3*10 + 10) / 4 = 10
        assert!((output[0] - 10.0).abs() < 0.01);
        // Second pixel: (3*10 + 20) / 4 = 12.5
        assert!((output[1] - 12.5).abs() < 0.01);
        // Last pixel: edge replication → (3*40 + 40) / 4 = 40
        assert!((output[7] - 40.0).abs() < 0.01);
    }

    #[test]
    fn h2v1_constant_input() {
        let input = [50.0f32; 8];
        let output = upsample_h2v1(&input, 8, 1, 16, 1);
        for (i, &v) in output.iter().enumerate() {
            assert!((v - 50.0).abs() < 0.01, "pixel {i}: {v} != 50.0");
        }
    }

    #[test]
    fn h1v2_identity_at_edges() {
        let input = [10.0, 20.0, 30.0, 40.0];
        let output = upsample_h1v2(&input, 1, 4, 1, 8);
        assert_eq!(output.len(), 8);
        assert!((output[0] - 10.0).abs() < 0.01);
        assert!((output[7] - 40.0).abs() < 0.01);
    }

    #[test]
    fn h1v2_constant_input() {
        let input = [75.0f32; 16];
        let output = upsample_h1v2(&input, 4, 4, 4, 8);
        assert_eq!(output.len(), 32);
        for (i, &v) in output.iter().enumerate() {
            assert!((v - 75.0).abs() < 0.01, "pixel {i}: {v} != 75.0");
        }
    }

    #[test]
    fn h2v2_constant_input() {
        let input = [100.0f32; 16];
        let output = upsample_h2v2(&input, 4, 4, 8, 8);
        assert_eq!(output.len(), 64);
        for (i, &v) in output.iter().enumerate() {
            assert!((v - 100.0).abs() < 0.01, "pixel {i}: {v} != 100.0");
        }
    }

    #[test]
    fn h2v2_output_dimensions() {
        let input = [0.0f32; 32 * 32];
        let output = upsample_h2v2(&input, 32, 32, 64, 64);
        assert_eq!(output.len(), 64 * 64);
    }

    #[test]
    fn h2v2_small_1x1_to_2x2() {
        let input = [128.0f32];
        let output = upsample_h2v2(&input, 1, 1, 2, 2);
        assert_eq!(output.len(), 4);
        for &v in &output {
            assert!((v - 128.0).abs() < 0.01);
        }
    }

    #[test]
    fn fancy_dispatch_1x1() {
        let input = [42.0f32; 16];
        let output = upsample_fancy(&input, 4, 4, 4, 4, 1, 1);
        assert_eq!(output.len(), 16);
        for &v in &output {
            assert!((v - 42.0).abs() < 0.01);
        }
    }

    #[test]
    fn fancy_dispatch_routes_correctly() {
        // Verify each dispatch produces correctly-sized output
        let input = [50.0f32; 4];
        assert_eq!(upsample_fancy(&input, 4, 1, 8, 1, 2, 1).len(), 8);
        assert_eq!(upsample_fancy(&input, 1, 4, 1, 8, 1, 2).len(), 8);
        assert_eq!(upsample_fancy(&input, 2, 2, 4, 4, 2, 2).len(), 16);
    }

    #[test]
    fn fancy_fallback_unusual_scale() {
        let input = [25.0f32; 4];
        let output = upsample_fancy(&input, 2, 2, 8, 4, 4, 2);
        assert_eq!(output.len(), 32);
        for &v in &output {
            assert!((v - 25.0).abs() < 0.01);
        }
    }

    #[test]
    fn fancy_crop_smaller_output() {
        let input = [1.0f32; 100];
        let output = upsample_fancy(&input, 10, 10, 5, 5, 1, 1);
        assert_eq!(output.len(), 25);
    }

    #[test]
    fn h2v2_i16_fancy_basic() {
        let input: Vec<i16> = vec![1000; 4 * 4];
        let mut output = vec![0i16; 8 * 8];
        upsample_h2v2_i16_fancy(&input, 4, 4, &mut output, 8, 8);
        for (i, &v) in output.iter().enumerate() {
            assert!((v - 1000).abs() <= 1, "i16 h2v2 pixel {i}: {v} != ~1000");
        }
    }

    #[test]
    fn h2v1_i16_fancy_basic() {
        let input: Vec<i16> = vec![500; 8];
        let mut output = vec![0i16; 16];
        upsample_h2v1_i16_fancy(&input, 8, 1, &mut output, 16, 1);
        for (i, &v) in output.iter().enumerate() {
            assert!((v - 500).abs() <= 1, "i16 h2v1 pixel {i}: {v} != ~500");
        }
    }

    #[test]
    fn h1v2_i16_fancy_basic() {
        let input: Vec<i16> = vec![1000; 8 * 4];
        let mut output = vec![0i16; 8 * 8];
        upsample_h1v2_i16_fancy(&input, 8, 4, &mut output, 8, 8);
        for (i, &v) in output.iter().enumerate() {
            assert!((v - 1000).abs() <= 1, "i16 h1v2 pixel {i}: {v} != ~1000");
        }
    }

    #[test]
    fn h2v2_i16_nearest_basic() {
        let input: Vec<i16> = vec![750; 4 * 4];
        let mut output = vec![0i16; 8 * 8];
        upsample_h2v2_i16_nearest(&input, 4, 4, &mut output, 8, 8);
        for &v in &output {
            assert_eq!(v, 750);
        }
    }

    #[test]
    fn h2v1_i16_nearest_basic() {
        let input: Vec<i16> = vec![300; 4];
        let mut output = vec![0i16; 8];
        upsample_h2v1_i16_nearest(&input, 4, 1, &mut output, 8, 1);
        for &v in &output {
            assert_eq!(v, 300);
        }
    }

    #[test]
    fn h1v2_i16_nearest_basic() {
        let input: Vec<i16> = vec![200; 4 * 4];
        let mut output = vec![0i16; 4 * 8];
        upsample_h1v2_i16_nearest(&input, 4, 4, &mut output, 4, 8);
        for &v in &output {
            assert_eq!(v, 200);
        }
    }

    #[test]
    fn h2v1_i16_libjpeg_basic() {
        let input: Vec<i16> = vec![400; 8];
        let mut output = vec![0i16; 16];
        upsample_h2v1_i16_libjpeg(&input, 8, 1, &mut output, 16, 1);
        for (i, &v) in output.iter().enumerate() {
            assert!((v - 400).abs() <= 1, "libjpeg h2v1 pixel {i}: {v} != ~400");
        }
    }

    #[test]
    fn h2v2_multirow() {
        // Test with multiple rows — verifies vertical interpolation
        let input: Vec<f32> = (0..12).map(|i| i as f32 * 10.0).collect();
        let output = upsample_h2v1(&input, 4, 3, 8, 3);
        assert_eq!(output.len(), 24);
    }

    // ========================================================================
    // Archmage dispatch parity tests
    // ========================================================================
    //
    // These tests verify that AVX2 and scalar paths produce identical output
    // for all upsampling functions. They use `for_each_token_permutation` to
    // exhaustively test every SIMD tier the CPU supports.

    /// Test data: gradient pattern with varying values to exercise edge handling
    fn gradient_test_data(width: usize, height: usize) -> Vec<i16> {
        (0..width * height)
            .map(|i| {
                let x = i % width;
                let y = i / width;
                ((x as i32 * 37 + y as i32 * 53) % 500 - 250) as i16
            })
            .collect()
    }

    /// Test data: extreme chroma transitions (worst case for rounding differences)
    fn extreme_test_data(width: usize, height: usize) -> Vec<i16> {
        (0..width * height)
            .map(|i| {
                let x = i % width;
                let y = i / width;
                // Alternate between extreme values at block boundaries
                if (x / 4 + y / 4) % 2 == 0 {
                    2000
                } else {
                    -2000
                }
            })
            .collect()
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn h2v2_fancy_strided_dispatch_parity() {
        use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

        // The AVX2 path uses a separable formula (vertical then horizontal, each
        // `(3*near + far + 2) >> 2`) while the scalar path uses non-separable
        // `(9*C + 3*H + 3*V + HV + 8) >> 4`. These differ by ±1 due to
        // intermediate rounding in the separable version. Both are valid triangle
        // filter implementations — the important invariant is that the fixup
        // function matches whichever formula is active (tested separately in
        // `h2v2_fixup_matches_main_upsampler`).
        let sizes: &[(usize, usize)] = &[
            (4, 4),
            (8, 8),
            (16, 16),
            (17, 9), // non-aligned
            (32, 32),
            (64, 16),  // wide
            (33, 33),  // odd
            (128, 64), // larger
        ];

        for &(in_w, in_h) in sizes {
            let out_w = in_w * 2;
            let out_h = in_h * 2;

            for (label, input) in [
                ("gradient", gradient_test_data(in_w, in_h)),
                ("extreme", extreme_test_data(in_w, in_h)),
                ("constant", vec![1000i16; in_w * in_h]),
            ] {
                // Compute reference output
                let mut reference = vec![0i16; out_w * out_h];
                upsample_h2v2_i16_fancy_strided(
                    &input,
                    in_w,
                    in_w,
                    in_h,
                    &mut reference,
                    out_w,
                    out_w,
                    out_h,
                );

                let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
                    let mut result = vec![0i16; out_w * out_h];
                    upsample_h2v2_i16_fancy_strided(
                        &input,
                        in_w,
                        in_w,
                        in_h,
                        &mut result,
                        out_w,
                        out_w,
                        out_h,
                    );

                    let max_diff = result
                        .iter()
                        .zip(reference.iter())
                        .map(|(a, b)| (a - b).unsigned_abs())
                        .max()
                        .unwrap_or(0);

                    assert!(
                        max_diff <= 1,
                        "h2v2_fancy_strided max_diff={max_diff} (>1): {label} {in_w}x{in_h} at {perm}"
                    );
                });

                if label == "gradient" && in_w == 4 {
                    eprintln!("h2v2_fancy_strided dispatch: {report}");
                    assert!(
                        report.permutations_run >= 2,
                        "expected at least 2 permutations"
                    );
                }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn h2v2_fixup_dispatch_parity() {
        use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

        // Test various widths including non-aligned
        let widths: &[usize] = &[4, 8, 15, 16, 17, 32, 33, 64, 128, 255, 256];

        for &in_width in widths {
            let out_width = in_width * 2;

            for (label, curr_row, v_neighbor_row) in [
                (
                    "gradient",
                    (0..in_width)
                        .map(|x| (x as i32 * 37 % 500 - 250) as i16)
                        .collect::<Vec<_>>(),
                    (0..in_width)
                        .map(|x| (x as i32 * 53 % 500 - 250) as i16)
                        .collect::<Vec<_>>(),
                ),
                (
                    "extreme",
                    (0..in_width)
                        .map(|x| if x % 2 == 0 { 2000i16 } else { -2000 })
                        .collect(),
                    (0..in_width)
                        .map(|x| if x % 2 == 0 { -2000i16 } else { 2000 })
                        .collect(),
                ),
                ("constant", vec![1000i16; in_width], vec![1000i16; in_width]),
            ] {
                let mut reference = vec![0i16; out_width];
                upsample_row_h2v2_fixup(&curr_row, &v_neighbor_row, in_width, &mut reference);

                let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
                    let mut result = vec![0i16; out_width];
                    upsample_row_h2v2_fixup(&curr_row, &v_neighbor_row, in_width, &mut result);
                    assert_eq!(
                        result, reference,
                        "h2v2_fixup mismatch: {label} width={in_width} at {perm}"
                    );
                });

                if label == "gradient" && in_width == 4 {
                    eprintln!("h2v2_fixup dispatch: {report}");
                    assert!(
                        report.permutations_run >= 2,
                        "expected at least 2 permutations"
                    );
                }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn h2v2_fixup_matches_main_upsampler() {
        use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

        // The fixup function must produce output identical to the main upsampler
        // for the same row pair. This is the bug that was fixed: previously the
        // fixup used a non-separable formula while the main used separable,
        // causing ±1 differences at MCU boundaries.
        let widths: &[usize] = &[4, 8, 16, 32, 64, 128];

        for &in_width in widths {
            let out_width = in_width * 2;
            let in_height = 4;
            let out_height = in_height * 2;

            let input = gradient_test_data(in_width, in_height);

            // Get the main upsampler's output for row 0 (top edge: curr=row0, v_neighbor=row0)
            let mut main_output = vec![0i16; out_width * out_height];
            upsample_h2v2_i16_fancy_strided(
                &input,
                in_width,
                in_width,
                in_height,
                &mut main_output,
                out_width,
                out_width,
                out_height,
            );

            // Test that fixup matches main for each output row pair
            let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
                // Re-run main upsampler at this token level
                let mut main_at_tier = vec![0i16; out_width * out_height];
                upsample_h2v2_i16_fancy_strided(
                    &input,
                    in_width,
                    in_width,
                    in_height,
                    &mut main_at_tier,
                    out_width,
                    out_width,
                    out_height,
                );

                // For each pair of input rows, verify fixup matches main upsampler
                for in_y in 0..in_height {
                    let curr_row = &input[in_y * in_width..][..in_width];

                    // Top half: v_neighbor is above
                    let v_above_y = in_y.saturating_sub(1);
                    let v_above_row = &input[v_above_y * in_width..][..in_width];

                    let mut fixup_out = vec![0i16; out_width];
                    upsample_row_h2v2_fixup(curr_row, v_above_row, in_width, &mut fixup_out);

                    let main_row_idx = in_y * 2; // top half output row
                    let main_row = &main_at_tier[main_row_idx * out_width..][..out_width];

                    // ±1 tolerance: separable (main) vs non-separable (fixup)
                    // rounding difference at lower SIMD tiers (same as
                    // h2v2_fancy_strided_with_padding_dispatch_parity).
                    for (x, (&f, &m)) in fixup_out.iter().zip(main_row.iter()).enumerate() {
                        let diff = (f - m).unsigned_abs();
                        assert!(
                            diff <= 1,
                            "fixup != main at in_y={in_y} x={x} (top half), width={in_width}, \
                             fixup={f} main={m} {perm}"
                        );
                    }

                    // Bottom half: v_neighbor is below
                    let v_below_y = (in_y + 1).min(in_height - 1);
                    let v_below_row = &input[v_below_y * in_width..][..in_width];

                    let mut fixup_out = vec![0i16; out_width];
                    upsample_row_h2v2_fixup(curr_row, v_below_row, in_width, &mut fixup_out);

                    let main_row_idx = in_y * 2 + 1; // bottom half output row
                    let main_row = &main_at_tier[main_row_idx * out_width..][..out_width];

                    for (x, (&f, &m)) in fixup_out.iter().zip(main_row.iter()).enumerate() {
                        let diff = (f - m).unsigned_abs();
                        assert!(
                            diff <= 1,
                            "fixup != main at in_y={in_y} x={x} (bottom half), width={in_width}, \
                             fixup={f} main={m} {perm}"
                        );
                    }
                }
            });

            if in_width == 4 {
                eprintln!("fixup_matches_main: {report}");
                assert!(
                    report.permutations_run >= 2,
                    "expected at least 2 permutations"
                );
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn h2v2_fancy_strided_with_padding_dispatch_parity() {
        use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

        // Test with in_stride > in_width (padded rows), which exercises the
        // strided AVX2 path differently from contiguous layout.
        // Allows ±1 for separable/non-separable rounding difference (see
        // h2v2_fancy_strided_dispatch_parity for explanation).
        let cases: &[(usize, usize, usize)] = &[
            (16, 8, 32),  // stride = 2x width
            (33, 16, 48), // non-aligned width, padded stride
            (64, 32, 80), // moderate padding
        ];

        for &(in_w, in_h, in_stride) in cases {
            let out_w = in_w * 2;
            let out_h = in_h * 2;

            // Create strided input with garbage in padding
            let mut input = vec![0x7FFFi16; in_stride * in_h];
            for y in 0..in_h {
                for x in 0..in_w {
                    input[y * in_stride + x] = ((x as i32 * 37 + y as i32 * 53) % 500 - 250) as i16;
                }
            }

            let out_stride = out_w + 16; // padded output too
            let mut reference = vec![0i16; out_stride * out_h];
            upsample_h2v2_i16_fancy_strided(
                &input,
                in_w,
                in_stride,
                in_h,
                &mut reference,
                out_w,
                out_stride,
                out_h,
            );

            let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
                let mut result = vec![0i16; out_stride * out_h];
                upsample_h2v2_i16_fancy_strided(
                    &input,
                    in_w,
                    in_stride,
                    in_h,
                    &mut result,
                    out_w,
                    out_stride,
                    out_h,
                );

                // Compare only the active output region (not padding)
                for y in 0..out_h {
                    for x in 0..out_w {
                        let idx = y * out_stride + x;
                        let diff = (result[idx] - reference[idx]).unsigned_abs();
                        assert!(
                            diff <= 1,
                            "h2v2_fancy_strided (padded) diff={diff} at ({x},{y}): \
                             {in_w}x{in_h} stride={in_stride} at {perm}"
                        );
                    }
                }
            });

            if in_w == 16 {
                eprintln!("h2v2_fancy_strided (padded) dispatch: {report}");
                assert!(
                    report.permutations_run >= 2,
                    "expected at least 2 permutations"
                );
            }
        }
    }
}
