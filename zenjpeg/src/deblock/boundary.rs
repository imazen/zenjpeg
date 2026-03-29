//! H.264-style 4-tap boundary deblocking filter.
//!
//! Operates in the pixel domain on f32 planes. At every 8-pixel block boundary,
//! applies a [1, 3, 3, 1]/8 weighted average to the two pixels straddling the
//! boundary, subject to a discontinuity threshold and maximum delta clamp.
//!
//! The filter runs two passes: vertical boundaries (columns at multiples of 8)
//! then horizontal boundaries (rows at multiples of 8).

/// Configuration for boundary deblocking strength.
#[derive(Debug, Clone, Copy)]
pub struct BoundaryStrength {
    /// Maximum pixel adjustment magnitude. Derived from DC quantization step.
    pub max_delta: f32,
    /// Discontinuity threshold below which no filtering occurs.
    pub threshold: f32,
}

impl BoundaryStrength {
    /// Compute strength from the DC quantization value of the component's quant table.
    ///
    /// Formula: `strength = dc_quant * 0.25`, capped at 12.0.
    /// Threshold: `strength * 0.4`.
    #[must_use]
    pub fn from_dc_quant(dc_quant: u16) -> Self {
        let strength = (dc_quant as f32 * 0.25).min(12.0);
        Self {
            max_delta: strength,
            threshold: strength * 0.4,
        }
    }
}

/// Apply 4-tap boundary deblocking to a single f32 plane in-place.
///
/// The plane is `width * height` contiguous f32 values in row-major order,
/// representing pixel values in the range [0, 255]. Block boundaries occur
/// at every 8th column and row.
///
/// # Arguments
/// * `plane` — Mutable pixel data, length must be >= `width * height`
/// * `width` — Plane width in pixels (must be >= 16 for any filtering)
/// * `height` — Plane height in pixels (must be >= 16 for any filtering)
/// * `strength` — Filter strength parameters
///
/// # Panics
/// Panics if `plane.len() < width * height`.
pub fn filter_plane_boundary_4tap(
    plane: &mut [f32],
    width: usize,
    height: usize,
    strength: BoundaryStrength,
) {
    debug_assert!(plane.len() >= width * height);

    if strength.max_delta < 0.5 || width < 16 || height < 16 {
        return;
    }

    filter_vertical_boundaries(plane, width, height, &strength);
    filter_horizontal_boundaries(plane, width, height, &strength);
}

/// Filter vertical boundaries (columns at multiples of 8).
///
/// For each boundary column, processes all rows. Vertical boundaries have strided
/// memory access (stride = width), so we process one row at a time with the
/// compiler's autovectorizer handling the simple arithmetic.
#[inline(never)]
fn filter_vertical_boundaries(
    plane: &mut [f32],
    width: usize,
    height: usize,
    strength: &BoundaryStrength,
) {
    let thresh = strength.threshold;
    let max_d = strength.max_delta;

    let num_boundaries = width / 8;

    for bx in 1..num_boundaries {
        let col = bx * 8;
        if col + 1 >= width || col < 2 {
            continue;
        }

        for y in 0..height {
            let base = y * width;
            let p1 = plane[base + col - 2];
            let p0 = plane[base + col - 1];
            let q0 = plane[base + col];
            let q1 = plane[base + col + 1];

            let disc = (p0 - q0).abs();
            if disc < thresh {
                continue;
            }

            // 4-tap: [1, 3, 3, 1] / 8 weighted average across boundary
            let avg = (p1 + 3.0 * p0 + 3.0 * q0 + q1) * 0.125;
            let delta_p = (avg - p0).clamp(-max_d, max_d);
            let delta_q = (avg - q0).clamp(-max_d, max_d);

            plane[base + col - 1] = (p0 + delta_p).clamp(0.0, 255.0);
            plane[base + col] = (q0 + delta_q).clamp(0.0, 255.0);
        }
    }
}

/// Filter horizontal boundaries (rows at multiples of 8).
///
/// For each boundary row, processes all columns. Horizontal boundaries have
/// contiguous memory access within each row, allowing efficient vectorization.
/// Uses `wide::f32x8` to process 8 contiguous columns at a time.
#[inline(never)]
fn filter_horizontal_boundaries(
    plane: &mut [f32],
    width: usize,
    height: usize,
    strength: &BoundaryStrength,
) {
    use wide::f32x8;

    let max_d = strength.max_delta;
    let thresh = strength.threshold;
    let max_d_v = f32x8::splat(max_d);
    let neg_max_d_v = f32x8::splat(-max_d);
    let zero_v = f32x8::splat(0.0);
    let val_255_v = f32x8::splat(255.0);
    let three_v = f32x8::splat(3.0);
    let eighth_v = f32x8::splat(0.125);

    let num_boundaries = height / 8;

    for by in 1..num_boundaries {
        let row = by * 8;
        if row + 1 >= height || row < 2 {
            continue;
        }

        let off_p1 = (row - 2) * width;
        let off_p0 = (row - 1) * width;
        let off_q0 = row * width;
        let off_q1 = (row + 1) * width;

        // SIMD: process 8 contiguous columns at a time
        let mut x = 0;
        while x + 8 <= width {
            // Contiguous loads — optimal memory access pattern
            let p1 = f32x8::new(plane[off_p1 + x..off_p1 + x + 8].try_into().unwrap());
            let p0 = f32x8::new(plane[off_p0 + x..off_p0 + x + 8].try_into().unwrap());
            let q0 = f32x8::new(plane[off_q0 + x..off_q0 + x + 8].try_into().unwrap());
            let q1 = f32x8::new(plane[off_q1 + x..off_q1 + x + 8].try_into().unwrap());

            // 4-tap average
            let avg = (p1 + three_v * p0 + three_v * q0 + q1) * eighth_v;

            // Deltas clamped to max strength
            let delta_p = (avg - p0).max(neg_max_d_v).min(max_d_v);
            let delta_q = (avg - q0).max(neg_max_d_v).min(max_d_v);

            // Apply with pixel range clamping
            let new_p0 = (p0 + delta_p).max(zero_v).min(val_255_v);
            let new_q0 = (q0 + delta_q).max(zero_v).min(val_255_v);

            // Threshold masking: only update lanes where |p0 - q0| >= threshold.
            // We do this per-element since wide doesn't have blend/select.
            let disc: [f32; 8] = (p0 - q0).abs().into();
            let new_p0_arr: [f32; 8] = new_p0.into();
            let new_q0_arr: [f32; 8] = new_q0.into();

            for i in 0..8 {
                if disc[i] >= thresh {
                    plane[off_p0 + x + i] = new_p0_arr[i];
                    plane[off_q0 + x + i] = new_q0_arr[i];
                }
            }

            x += 8;
        }

        // Scalar tail
        for x in (width & !7)..width {
            let p1 = plane[off_p1 + x];
            let p0 = plane[off_p0 + x];
            let q0 = plane[off_q0 + x];
            let q1 = plane[off_q1 + x];

            let disc = (p0 - q0).abs();
            if disc < thresh {
                continue;
            }

            let avg = (p1 + 3.0 * p0 + 3.0 * q0 + q1) * 0.125;
            let delta_p = (avg - p0).clamp(-max_d, max_d);
            let delta_q = (avg - q0).clamp(-max_d, max_d);

            plane[off_p0 + x] = (p0 + delta_p).clamp(0.0, 255.0);
            plane[off_q0 + x] = (q0 + delta_q).clamp(0.0, 255.0);
        }
    }
}

/// Apply 4-tap boundary deblocking to interleaved u8 pixel data in-place.
///
/// Operates on packed multi-channel data (e.g., RGB or RGBA) by filtering each
/// channel independently at 8-pixel block boundaries. This is the post-decode
/// variant for the streaming path where pixel data is already in interleaved u8
/// format rather than per-component f32 planes.
///
/// The filter runs two passes: vertical boundaries (columns at multiples of 8)
/// then horizontal boundaries (rows at multiples of 8). Within each pass, all
/// channels at a given boundary pixel are processed independently.
///
/// # Arguments
/// * `pixels` — Mutable pixel data in row-major, channel-interleaved order
///   (e.g., `[R,G,B, R,G,B, ...]`). Length must be >= `width * height * channels`.
/// * `width` — Image width in pixels (not bytes)
/// * `height` — Image height in pixels
/// * `channels` — Number of channels per pixel (e.g., 3 for RGB, 4 for RGBA)
/// * `strength` — Filter strength parameters (derived from DC quantization value)
///
/// # Panics
/// Panics if `pixels.len() < width * height * channels`.
pub fn filter_interleaved_u8_boundary_4tap(
    pixels: &mut [u8],
    width: usize,
    height: usize,
    channels: usize,
    strength: BoundaryStrength,
) {
    debug_assert!(pixels.len() >= width * height * channels);

    if strength.max_delta < 0.5 || width < 16 || height < 16 || channels == 0 {
        return;
    }

    filter_vertical_boundaries_u8(pixels, width, height, channels, &strength);
    filter_horizontal_boundaries_u8(pixels, width, height, channels, &strength);
}

/// Filter vertical boundaries (columns at multiples of 8) on interleaved u8 data.
#[inline(never)]
fn filter_vertical_boundaries_u8(
    pixels: &mut [u8],
    width: usize,
    height: usize,
    channels: usize,
    strength: &BoundaryStrength,
) {
    let thresh = strength.threshold;
    let max_d = strength.max_delta;
    let stride = width * channels;
    let num_boundaries = width / 8;

    for bx in 1..num_boundaries {
        let col = bx * 8;
        if col + 1 >= width || col < 2 {
            continue;
        }

        let base_p1 = (col - 2) * channels;
        let base_p0 = (col - 1) * channels;
        let base_q0 = col * channels;
        let base_q1 = (col + 1) * channels;

        for y in 0..height {
            let row_off = y * stride;
            for c in 0..channels {
                let p1 = pixels[row_off + base_p1 + c] as f32;
                let p0 = pixels[row_off + base_p0 + c] as f32;
                let q0 = pixels[row_off + base_q0 + c] as f32;
                let q1 = pixels[row_off + base_q1 + c] as f32;

                let disc = (p0 - q0).abs();
                if disc < thresh {
                    continue;
                }

                let avg = (p1 + 3.0 * p0 + 3.0 * q0 + q1) * 0.125;
                let delta_p = (avg - p0).clamp(-max_d, max_d);
                let delta_q = (avg - q0).clamp(-max_d, max_d);

                pixels[row_off + base_p0 + c] = (p0 + delta_p).clamp(0.0, 255.0) as u8;
                pixels[row_off + base_q0 + c] = (q0 + delta_q).clamp(0.0, 255.0) as u8;
            }
        }
    }
}

/// Filter horizontal boundaries (rows at multiples of 8) on interleaved u8 data.
#[inline(never)]
fn filter_horizontal_boundaries_u8(
    pixels: &mut [u8],
    width: usize,
    height: usize,
    channels: usize,
    strength: &BoundaryStrength,
) {
    let thresh = strength.threshold;
    let max_d = strength.max_delta;
    let stride = width * channels;
    let num_boundaries = height / 8;

    for by in 1..num_boundaries {
        let row = by * 8;
        if row + 1 >= height || row < 2 {
            continue;
        }

        let off_p1 = (row - 2) * stride;
        let off_p0 = (row - 1) * stride;
        let off_q0 = row * stride;
        let off_q1 = (row + 1) * stride;

        for x in 0..width {
            let px = x * channels;
            for c in 0..channels {
                let p1 = pixels[off_p1 + px + c] as f32;
                let p0 = pixels[off_p0 + px + c] as f32;
                let q0 = pixels[off_q0 + px + c] as f32;
                let q1 = pixels[off_q1 + px + c] as f32;

                let disc = (p0 - q0).abs();
                if disc < thresh {
                    continue;
                }

                let avg = (p1 + 3.0 * p0 + 3.0 * q0 + q1) * 0.125;
                let delta_p = (avg - p0).clamp(-max_d, max_d);
                let delta_q = (avg - q0).clamp(-max_d, max_d);

                pixels[off_p0 + px + c] = (p0 + delta_p).clamp(0.0, 255.0) as u8;
                pixels[off_q0 + px + c] = (q0 + delta_q).clamp(0.0, 255.0) as u8;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_filter_below_threshold() {
        // Smooth plane — no discontinuities
        let mut plane = vec![128.0f32; 32 * 32];
        let strength = BoundaryStrength::from_dc_quant(20);
        filter_plane_boundary_4tap(&mut plane, 32, 32, strength);
        assert!(plane.iter().all(|&v| (v - 128.0).abs() < 0.001));
    }

    #[test]
    fn test_filter_at_boundary() {
        // 32x32 plane with sharp step at column 8
        let mut plane = vec![0.0f32; 32 * 32];
        for y in 0..32 {
            for x in 0..32 {
                plane[y * 32 + x] = if x < 8 { 50.0 } else { 200.0 };
            }
        }

        let strength = BoundaryStrength::from_dc_quant(20);
        filter_plane_boundary_4tap(&mut plane, 32, 32, strength);

        // Pixels at boundary should be pulled toward each other
        let p0 = plane[2 * 32 + 7]; // col 7
        let q0 = plane[2 * 32 + 8]; // col 8
        assert!(p0 > 50.0, "p0 should increase: {p0}");
        assert!(q0 < 200.0, "q0 should decrease: {q0}");
    }

    #[test]
    fn test_horizontal_boundary() {
        // 32x32 plane with sharp step at row 8
        let mut plane = vec![0.0f32; 32 * 32];
        for y in 0..32 {
            for x in 0..32 {
                plane[y * 32 + x] = if y < 8 { 50.0 } else { 200.0 };
            }
        }

        let strength = BoundaryStrength::from_dc_quant(20);
        filter_plane_boundary_4tap(&mut plane, 32, 32, strength);

        // Pixels at boundary should be pulled toward each other
        let p0 = plane[7 * 32 + 5]; // row 7
        let q0 = plane[8 * 32 + 5]; // row 8
        assert!(p0 > 50.0, "p0 should increase: {p0}");
        assert!(q0 < 200.0, "q0 should decrease: {q0}");
    }

    #[test]
    fn test_strength_from_dc_quant() {
        let s = BoundaryStrength::from_dc_quant(40);
        assert!((s.max_delta - 10.0).abs() < 0.001);
        assert!((s.threshold - 4.0).abs() < 0.001);

        // Capped at 12
        let s = BoundaryStrength::from_dc_quant(100);
        assert!((s.max_delta - 12.0).abs() < 0.001);
    }

    #[test]
    fn test_small_plane_skipped() {
        let mut plane = vec![128.0f32; 8 * 8];
        let strength = BoundaryStrength::from_dc_quant(20);
        filter_plane_boundary_4tap(&mut plane, 8, 8, strength);
        assert!(plane.iter().all(|&v| (v - 128.0).abs() < 0.001));
    }
}
