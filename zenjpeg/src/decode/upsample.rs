//! Chroma upsampling for JPEG decoding.
//!
//! Implements triangle filter (3:1 weighting) upsampling for various
//! chroma subsampling modes (4:2:2, 4:4:0, 4:2:0).

use wide::f32x8;

#[cfg(all(feature = "archmage-simd", target_arch = "x86_64"))]
use archmage::{arcane, rite, SimdToken};

#[cfg(all(feature = "archmage-simd", target_arch = "x86_64"))]
use safe_unaligned_simd::x86_64 as safe_simd;

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

    // Try AVX2 SIMD path on x86_64 (requires archmage-simd feature)
    #[cfg(all(feature = "archmage-simd", target_arch = "x86_64"))]
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

    // Try AVX2 SIMD path on x86_64 (requires archmage-simd feature)
    #[cfg(all(feature = "archmage-simd", target_arch = "x86_64"))]
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
#[cfg(all(feature = "archmage-simd", target_arch = "x86_64"))]
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
#[cfg(all(feature = "archmage-simd", target_arch = "x86_64"))]
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
#[cfg(all(feature = "archmage-simd", target_arch = "x86_64"))]
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
#[cfg(all(feature = "archmage-simd", target_arch = "x86_64"))]
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
#[cfg(all(
    feature = "archmage-simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
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
#[cfg(all(
    feature = "archmage-simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
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
#[cfg(all(
    feature = "archmage-simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
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
#[cfg(all(
    feature = "archmage-simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
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
