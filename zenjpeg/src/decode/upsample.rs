//! Chroma upsampling for JPEG decoding.
//!
//! Implements libjpeg-turbo compatible upsampling and nearest-neighbor (box filter)
//! upsampling for various chroma subsampling modes (4:2:2, 4:4:0, 4:2:0).

#[cfg(target_arch = "x86_64")]
use archmage::{SimdToken, arcane};

#[cfg(target_arch = "x86_64")]
use safe_unaligned_simd::x86_64 as safe_simd;

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
///
/// Dispatches to AVX2 SIMD on x86_64 when available, with scalar fallback.
#[inline]
pub(super) fn upsample_h2v2_libjpeg_row(
    near: &[i16],
    far: &[i16],
    output: &mut [i16],
    in_width: usize,
    out_width: usize,
    is_upper: bool,
) {
    // Try AVX2 SIMD path on x86_64
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(token) = archmage::X64V3Token::summon() {
            upsample_h2v2_libjpeg_row_avx2(token, near, far, output, in_width, out_width, is_upper);
            return;
        }
    }

    upsample_h2v2_libjpeg_row_scalar(near, far, output, in_width, out_width, is_upper);
}

/// Scalar implementation of one output row of fused h2v2 libjpeg-compat upsampling.
#[inline]
fn upsample_h2v2_libjpeg_row_scalar(
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

/// AVX2 SIMD implementation of one output row of fused h2v2 libjpeg-compat upsampling.
///
/// Processes 16 input chroma samples at a time → 32 output pixels.
/// Produces bit-exact output matching the scalar `upsample_h2v2_libjpeg_row_scalar`.
///
/// All arithmetic stays within i16 range: colsum max magnitude is 8192
/// (`2048*3 + 2048`), and `colsum*3 + colsum_neighbor + 8` max is 32760.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn upsample_h2v2_libjpeg_row_avx2(
    _token: archmage::X64V3Token,
    near: &[i16],
    far: &[i16],
    output: &mut [i16],
    in_width: usize,
    out_width: usize,
    is_upper: bool,
) {
    use core::arch::x86_64::*;

    // For very small widths, fall back to scalar (not worth SIMD overhead)
    if in_width < 18 {
        upsample_h2v2_libjpeg_row_scalar(near, far, output, in_width, out_width, is_upper);
        return;
    }

    let (bias_left, bias_right) = if is_upper { (8i16, 7i16) } else { (7i16, 8i16) };

    let v_three = _mm256_set1_epi16(3);
    let v_bias_left = _mm256_set1_epi16(bias_left);
    let v_bias_right = _mm256_set1_epi16(bias_right);

    // --- First column (scalar, special edge handling) ---
    let colsum_0 = near[0] as i32 * 3 + far[0] as i32;
    let colsum_1 = near[1] as i32 * 3 + far[1] as i32;
    output[0] = ((colsum_0 * 4 + 8) >> 4) as i16;
    if out_width > 1 {
        output[1] = ((colsum_0 * 3 + colsum_1 + bias_right as i32) >> 4) as i16;
    }

    // --- Interior: SIMD processing ---
    // Process chunks of 16 input pixels starting from position 1.
    // For each chunk we need colsum[x-1..x+16] and colsum[x..x+17],
    // so we need near/far[x-1..x+17] accessible. We process up to
    // the point where x+17 <= in_width (i.e., x <= in_width - 17).
    let simd_start = 1usize;
    let simd_end_exclusive = if in_width >= 17 {
        // Last chunk starts at x where x+16 <= in_width-1 (need next neighbor)
        // i.e., x <= in_width - 17
        let max_start = in_width - 17;
        // Round down to chunk boundary relative to simd_start
        let num_chunks = (max_start - simd_start + 16) / 16;
        simd_start + num_chunks * 16
    } else {
        simd_start
    };

    let mut x = simd_start;
    while x + 16 <= simd_end_exclusive {
        let out_base = x * 2;
        if out_base + 32 > out_width {
            break;
        }

        // Load near[x..x+16], far[x..x+16] → colsum[x..x+16]
        let v_near =
            safe_simd::_mm256_loadu_si256(<&[i16; 16]>::try_from(&near[x..x + 16]).unwrap());
        let v_far = safe_simd::_mm256_loadu_si256(<&[i16; 16]>::try_from(&far[x..x + 16]).unwrap());
        let v_colsum = _mm256_add_epi16(_mm256_mullo_epi16(v_near, v_three), v_far);

        // Load colsum for x-1 (prev neighbor)
        let v_near_prev =
            safe_simd::_mm256_loadu_si256(<&[i16; 16]>::try_from(&near[x - 1..x + 15]).unwrap());
        let v_far_prev =
            safe_simd::_mm256_loadu_si256(<&[i16; 16]>::try_from(&far[x - 1..x + 15]).unwrap());
        let v_colsum_prev = _mm256_add_epi16(_mm256_mullo_epi16(v_near_prev, v_three), v_far_prev);

        // Load colsum for x+1 (next neighbor)
        let v_near_next =
            safe_simd::_mm256_loadu_si256(<&[i16; 16]>::try_from(&near[x + 1..x + 17]).unwrap());
        let v_far_next =
            safe_simd::_mm256_loadu_si256(<&[i16; 16]>::try_from(&far[x + 1..x + 17]).unwrap());
        let v_colsum_next = _mm256_add_epi16(_mm256_mullo_epi16(v_near_next, v_three), v_far_next);

        // colsum * 3
        let v_colsum3 = _mm256_mullo_epi16(v_colsum, v_three);

        // Left output: (colsum*3 + colsum_prev + bias_left) >> 4
        let v_left = _mm256_srai_epi16(
            _mm256_add_epi16(_mm256_add_epi16(v_colsum3, v_colsum_prev), v_bias_left),
            4,
        );

        // Right output: (colsum*3 + colsum_next + bias_right) >> 4
        let v_right = _mm256_srai_epi16(
            _mm256_add_epi16(_mm256_add_epi16(v_colsum3, v_colsum_next), v_bias_right),
            4,
        );

        // Interleave left and right: [L0, R0, L1, R1, ...]
        // unpacklo/hi work on 128-bit lanes, so we need permute to fix order
        let lo = _mm256_unpacklo_epi16(v_left, v_right);
        let hi = _mm256_unpackhi_epi16(v_left, v_right);
        let out0 = _mm256_permute2x128_si256(lo, hi, 0x20);
        let out1 = _mm256_permute2x128_si256(lo, hi, 0x31);

        safe_simd::_mm256_storeu_si256(
            <&mut [i16; 16]>::try_from(&mut output[out_base..out_base + 16]).unwrap(),
            out0,
        );
        safe_simd::_mm256_storeu_si256(
            <&mut [i16; 16]>::try_from(&mut output[out_base + 16..out_base + 32]).unwrap(),
            out1,
        );

        x += 16;
    }

    // --- Scalar remainder for interior pixels not covered by SIMD ---
    let mut last_colsum_i32 = if x > 1 {
        near[x - 1] as i32 * 3 + far[x - 1] as i32
    } else {
        colsum_0
    };

    for in_x in x..in_width.saturating_sub(1) {
        let this_colsum = near[in_x] as i32 * 3 + far[in_x] as i32;
        let next_colsum = near[in_x + 1] as i32 * 3 + far[in_x + 1] as i32;

        let left_out = in_x * 2;
        let right_out = left_out + 1;
        if left_out < out_width {
            output[left_out] = ((this_colsum * 3 + last_colsum_i32 + bias_left as i32) >> 4) as i16;
        }
        if right_out < out_width {
            output[right_out] = ((this_colsum * 3 + next_colsum + bias_right as i32) >> 4) as i16;
        }
        last_colsum_i32 = this_colsum;
    }

    // --- Last column (scalar, special edge handling) ---
    let last = in_width - 1;
    if last >= x || x == simd_start {
        // Only emit last column if not already covered
        let this_colsum = near[last] as i32 * 3 + far[last] as i32;
        let prev_colsum = if last > 0 {
            near[last - 1] as i32 * 3 + far[last - 1] as i32
        } else {
            this_colsum
        };
        let left_out = last * 2;
        let right_out = left_out + 1;
        if left_out < out_width {
            output[left_out] = ((this_colsum * 3 + prev_colsum + bias_left as i32) >> 4) as i16;
        }
        if right_out < out_width {
            output[right_out] = ((this_colsum * 4 + bias_right as i32) >> 4) as i16;
        }
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

#[cfg(test)]
mod tests {
    use super::*;

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
    fn h2v2_libjpeg_row_dispatch_parity() {
        use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

        // Test that AVX2 and scalar paths produce identical output for
        // the libjpeg-compat h2v2 fused row upsampler.
        let widths: &[usize] = &[2, 4, 8, 15, 16, 17, 18, 32, 33, 64, 128, 255, 256, 512];

        for &in_width in widths {
            let out_width = in_width * 2;

            for (label, near_data, far_data) in [
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
                ("constant", vec![1000i16; in_width], vec![500i16; in_width]),
            ] {
                for is_upper in [true, false] {
                    // Compute reference using scalar path
                    let mut reference = vec![0i16; out_width];
                    upsample_h2v2_libjpeg_row_scalar(
                        &near_data,
                        &far_data,
                        &mut reference,
                        in_width,
                        out_width,
                        is_upper,
                    );

                    let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
                        let mut result = vec![0i16; out_width];
                        upsample_h2v2_libjpeg_row(
                            &near_data,
                            &far_data,
                            &mut result,
                            in_width,
                            out_width,
                            is_upper,
                        );

                        assert_eq!(
                            result, reference,
                            "h2v2_libjpeg_row mismatch: {label} width={in_width} \
                             is_upper={is_upper} at {perm}"
                        );
                    });

                    if label == "gradient" && in_width == 32 && is_upper {
                        eprintln!("h2v2_libjpeg_row dispatch: {report}");
                        assert!(
                            report.permutations_run >= 2,
                            "expected at least 2 permutations"
                        );
                    }
                }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn h2v2_libjpeg_full_dispatch_parity() {
        use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

        // Test the full h2v2 libjpeg upsampler (multiple rows) for SIMD parity.
        let sizes: &[(usize, usize)] = &[
            (4, 4),
            (8, 8),
            (16, 16),
            (17, 9),
            (32, 32),
            (64, 16),
            (33, 33),
            (128, 64),
        ];

        for &(in_w, in_h) in sizes {
            let out_w = in_w * 2;
            let out_h = in_h * 2;

            for (label, input) in [
                ("gradient", gradient_test_data(in_w, in_h)),
                ("extreme", extreme_test_data(in_w, in_h)),
                ("constant", vec![1000i16; in_w * in_h]),
            ] {
                // Compute reference using scalar path directly
                let mut reference = vec![0i16; out_w * out_h];
                for out_y in 0..out_h {
                    let in_y = out_y / 2;
                    let in_y_clamped = in_y.min(in_h.saturating_sub(1));
                    let is_upper = out_y % 2 == 0;

                    let far_y = if is_upper {
                        in_y_clamped.saturating_sub(1)
                    } else {
                        (in_y + 1).min(in_h.saturating_sub(1))
                    };

                    let near_row = in_y_clamped * in_w;
                    let far_row = far_y * in_w;
                    let out_row = out_y * out_w;

                    upsample_h2v2_libjpeg_row_scalar(
                        &input[near_row..near_row + in_w],
                        &input[far_row..far_row + in_w],
                        &mut reference[out_row..],
                        in_w,
                        out_w,
                        is_upper,
                    );
                }

                let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
                    let mut result = vec![0i16; out_w * out_h];
                    upsample_h2v2_i16_libjpeg(&input, in_w, in_h, &mut result, out_w, out_h);

                    assert_eq!(
                        result, reference,
                        "h2v2_libjpeg_full mismatch: {label} {in_w}x{in_h} at {perm}"
                    );
                });

                if label == "gradient" && in_w == 32 {
                    eprintln!("h2v2_libjpeg_full dispatch: {report}");
                    assert!(
                        report.permutations_run >= 2,
                        "expected at least 2 permutations"
                    );
                }
            }
        }
    }
}
