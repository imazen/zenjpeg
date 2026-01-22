//! Chroma upsampling for JPEG decoding.
//!
//! Implements triangle filter (3:1 weighting) upsampling for various
//! chroma subsampling modes (4:2:2, 4:4:0, 4:2:0).

use wide::f32x8;

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

use wide::i16x8;

/// Box filter 2x2 upsampling in i16 (4:2:0 → 4:4:4).
///
/// Simple pixel duplication - fastest possible upsampling.
/// Output is written to pre-allocated buffer for zero allocation.
#[inline]
pub fn upsample_h2v2_i16_box(
    input: &[i16],
    in_width: usize,
    in_height: usize,
    output: &mut [i16],
    out_width: usize,
    out_height: usize,
) {
    // Process two output rows at a time (both map to same input row)
    for out_y_pair in 0..(out_height + 1) / 2 {
        let in_y = out_y_pair.min(in_height.saturating_sub(1));
        let in_row = &input[in_y * in_width..];

        // First output row of pair
        let out_y0 = out_y_pair * 2;
        if out_y0 < out_height {
            let out_row0 = &mut output[out_y0 * out_width..][..out_width];
            upsample_row_h2_box(in_row, in_width, out_row0);
        }

        // Second output row of pair
        let out_y1 = out_y0 + 1;
        if out_y1 < out_height {
            let out_row1 = &mut output[out_y1 * out_width..][..out_width];
            upsample_row_h2_box(in_row, in_width, out_row1);
        }
    }
}

/// Upsample a single row horizontally 2x (box filter = pixel duplication).
#[inline(always)]
fn upsample_row_h2_box(input: &[i16], in_width: usize, output: &mut [i16]) {
    let out_width = output.len();

    // SIMD path: process 8 input pixels → 16 output pixels at a time
    let simd_in_chunks = in_width / 8;

    for chunk in 0..simd_in_chunks {
        let in_x = chunk * 8;
        let out_x = chunk * 16;

        if out_x + 16 > out_width {
            break;
        }

        // Load 8 input pixels
        let v = i16x8::from([
            input[in_x],
            input[in_x + 1],
            input[in_x + 2],
            input[in_x + 3],
            input[in_x + 4],
            input[in_x + 5],
            input[in_x + 6],
            input[in_x + 7],
        ]);

        // Duplicate each pixel: [a,b,c,d,e,f,g,h] → [a,a,b,b,c,c,d,d], [e,e,f,f,g,g,h,h]
        let arr: [i16; 8] = v.into();
        output[out_x] = arr[0];
        output[out_x + 1] = arr[0];
        output[out_x + 2] = arr[1];
        output[out_x + 3] = arr[1];
        output[out_x + 4] = arr[2];
        output[out_x + 5] = arr[2];
        output[out_x + 6] = arr[3];
        output[out_x + 7] = arr[3];
        output[out_x + 8] = arr[4];
        output[out_x + 9] = arr[4];
        output[out_x + 10] = arr[5];
        output[out_x + 11] = arr[5];
        output[out_x + 12] = arr[6];
        output[out_x + 13] = arr[6];
        output[out_x + 14] = arr[7];
        output[out_x + 15] = arr[7];
    }

    // Scalar remainder
    let processed_out = simd_in_chunks * 16;
    for out_x in processed_out..out_width {
        let in_x = (out_x / 2).min(in_width.saturating_sub(1));
        output[out_x] = input[in_x];
    }
}

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

    // Process each output row
    for out_y in 0..out_height {
        let in_y = (out_y / 2).min(in_height - 1);
        let is_top_half = out_y % 2 == 0;

        // Vertical neighbor: above for top half, below for bottom half
        let v_neighbor_y = if is_top_half {
            in_y.saturating_sub(1)
        } else {
            (in_y + 1).min(in_height - 1)
        };

        let curr_row = &input[in_y * in_width..];
        let v_neighbor_row = &input[v_neighbor_y * in_width..];
        let out_row = &mut output[out_y * out_width..][..out_width];

        // Process row with optimized interior loop
        upsample_row_h2_fancy_bilinear(curr_row, v_neighbor_row, in_width, out_row, is_top_half);
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
fn upsample_row_h2_fancy_bilinear(
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
    let interior_start_out = 2;
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
    for out_y in 0..out_height {
        let in_y = out_y.min(in_height.saturating_sub(1));
        let out_row = out_y * out_width;
        let in_row = in_y * in_width;

        for out_x in 0..out_width {
            let in_x = out_x / 2;
            let in_x_clamped = in_x.min(in_width.saturating_sub(1));
            let curr = input[in_row + in_x_clamped] as i32;

            let result = if out_x % 2 == 0 {
                // Left half - blend with left neighbor
                let left = if in_x > 0 {
                    input[in_row + in_x - 1] as i32
                } else {
                    curr
                };
                (3 * curr + left + 2) >> 2
            } else {
                // Right half - blend with right neighbor
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
    for out_y in 0..out_height {
        let in_y = out_y / 2;
        let is_top = out_y % 2 == 0;
        let out_row = out_y * out_width;

        let neighbor_y = if is_top {
            in_y.saturating_sub(1)
        } else {
            (in_y + 1).min(in_height.saturating_sub(1))
        };

        let in_y_clamped = in_y.min(in_height.saturating_sub(1));
        let curr_row = in_y_clamped * in_width;
        let neighbor_row = neighbor_y * in_width;

        for out_x in 0..out_width {
            let in_x = out_x.min(in_width.saturating_sub(1));
            let curr = input[curr_row + in_x] as i32;
            let neighbor = input[neighbor_row + in_x] as i32;
            let result = (3 * curr + neighbor + 2) >> 2;
            output[out_row + out_x] = result as i16;
        }
    }
}
