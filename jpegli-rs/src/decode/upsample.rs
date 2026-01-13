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
