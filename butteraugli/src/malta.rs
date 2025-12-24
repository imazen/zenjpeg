//! Malta edge-aware filter for butteraugli.
//!
//! The Malta filter applies 16 different line kernels at various orientations
//! and sums their squared responses. This captures edge information at
//! multiple orientations for perceptual quality measurement.
//!
//! There are two variants:
//! - `MaltaUnit`: Full 9x9 pattern with 9-sample line kernels
//! - `MaltaUnitLF`: Low-frequency variant with 5-sample line kernels

use crate::image::ImageF;

/// Malta filter for HF/UHF bands (9 samples per line, 16 orientations).
///
/// Applies 16 different line kernels in various orientations centered at (x,y)
/// and returns the sum of squared responses.
///
/// The kernel patterns span a 9×9 area (-4 to +4 in each direction).
pub fn malta_unit(data: &ImageF, x: usize, y: usize) -> f32 {
    let width = data.width();
    let height = data.height();

    // Helper to safely get pixel value with zero padding for out-of-bounds
    let get = |dx: isize, dy: isize| -> f32 {
        let nx = x as isize + dx;
        let ny = y as isize + dy;
        if nx < 0 || ny < 0 || nx >= width as isize || ny >= height as isize {
            0.0
        } else {
            data.get(nx as usize, ny as usize)
        }
    };

    let _center = get(0, 0);

    // Helper to sum 9 values along a line
    let sum9 = |offsets: &[(isize, isize); 9]| -> f32 {
        let mut s = 0.0;
        for &(dx, dy) in offsets {
            s += get(dx, dy);
        }
        s
    };

    // Helper to sum 7 values along a line
    let sum7 = |offsets: &[(isize, isize); 7]| -> f32 {
        let mut s = 0.0;
        for &(dx, dy) in offsets {
            s += get(dx, dy);
        }
        s
    };

    let mut retval = 0.0f32;

    // Pattern 1: x grows, y constant (horizontal line)
    let sum = sum9(&[
        (-4, 0),
        (-3, 0),
        (-2, 0),
        (-1, 0),
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (4, 0),
    ]);
    retval += sum * sum;

    // Pattern 2: y grows, x constant (vertical line)
    let sum = sum9(&[
        (0, -4),
        (0, -3),
        (0, -2),
        (0, -1),
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
    ]);
    retval += sum * sum;

    // Pattern 3: both grow (diagonal \)
    let sum = sum7(&[(-3, -3), (-2, -2), (-1, -1), (0, 0), (1, 1), (2, 2), (3, 3)]);
    retval += sum * sum;

    // Pattern 4: y grows, x shrinks (diagonal /)
    let sum = sum7(&[(3, -3), (2, -2), (1, -1), (0, 0), (-1, 1), (-2, 2), (-3, 3)]);
    retval += sum * sum;

    // Pattern 5: y grows -4 to 4, x shrinks 1 -> -1
    let sum = sum9(&[
        (1, -4),
        (1, -3),
        (1, -2),
        (0, -1),
        (0, 0),
        (0, 1),
        (-1, 2),
        (-1, 3),
        (-1, 4),
    ]);
    retval += sum * sum;

    // Pattern 6: y grows -4 to 4, x grows -1 -> 1
    let sum = sum9(&[
        (-1, -4),
        (-1, -3),
        (-1, -2),
        (0, -1),
        (0, 0),
        (0, 1),
        (1, 2),
        (1, 3),
        (1, 4),
    ]);
    retval += sum * sum;

    // Pattern 7: x grows -4 to 4, y grows -1 to 1
    let sum = sum9(&[
        (-4, -1),
        (-3, -1),
        (-2, -1),
        (-1, 0),
        (0, 0),
        (1, 0),
        (2, 1),
        (3, 1),
        (4, 1),
    ]);
    retval += sum * sum;

    // Pattern 8: x grows -4 to 4, y shrinks 1 to -1
    let sum = sum9(&[
        (-4, 1),
        (-3, 1),
        (-2, 1),
        (-1, 0),
        (0, 0),
        (1, 0),
        (2, -1),
        (3, -1),
        (4, -1),
    ]);
    retval += sum * sum;

    // Pattern 9: steep diagonal (2:1 slope)
    let sum = sum7(&[
        (-2, -3),
        (-1, -2),
        (-1, -1),
        (0, 0),
        (1, 1),
        (1, 2),
        (2, 3),
    ]);
    retval += sum * sum;

    // Pattern 10: steep diagonal other way
    let sum = sum7(&[
        (2, -3),
        (1, -2),
        (1, -1),
        (0, 0),
        (-1, 1),
        (-1, 2),
        (-2, 3),
    ]);
    retval += sum * sum;

    // Pattern 11: shallow diagonal (1:2 slope)
    let sum = sum7(&[
        (-3, -2),
        (-2, -1),
        (-1, -1),
        (0, 0),
        (1, 1),
        (2, 1),
        (3, 2),
    ]);
    retval += sum * sum;

    // Pattern 12: shallow diagonal other way
    let sum = sum7(&[
        (3, -2),
        (2, -1),
        (1, -1),
        (0, 0),
        (-1, 1),
        (-2, 1),
        (-3, 2),
    ]);
    retval += sum * sum;

    // Pattern 13: curved line pattern
    let sum = sum9(&[
        (-4, 1),
        (-3, 1),
        (-2, 1),
        (-1, 0),
        (0, 0),
        (1, 0),
        (2, -1),
        (3, -1),
        (4, -1),
    ]);
    retval += sum * sum;

    // Pattern 14: curved line other direction
    let sum = sum9(&[
        (-4, -1),
        (-3, -1),
        (-2, -1),
        (-1, 0),
        (0, 0),
        (1, 0),
        (2, 1),
        (3, 1),
        (4, 1),
    ]);
    retval += sum * sum;

    // Pattern 15: very shallow curve
    let sum = sum9(&[
        (-1, -4),
        (-1, -3),
        (-1, -2),
        (0, -1),
        (0, 0),
        (0, 1),
        (1, 2),
        (1, 3),
        (1, 4),
    ]);
    retval += sum * sum;

    // Pattern 16: very shallow curve other direction
    let sum = sum9(&[
        (1, -4),
        (1, -3),
        (1, -2),
        (0, -1),
        (0, 0),
        (0, 1),
        (-1, 2),
        (-1, 3),
        (-1, 4),
    ]);
    retval += sum * sum;

    retval
}

/// Malta filter for LF band (5 samples per line, 16 orientations).
///
/// Similar to `malta_unit` but with sparser sampling for low-frequency
/// content. Uses 5 samples per line instead of 9.
pub fn malta_unit_lf(data: &ImageF, x: usize, y: usize) -> f32 {
    let width = data.width();
    let height = data.height();

    // Helper to safely get pixel value with zero padding for out-of-bounds
    let get = |dx: isize, dy: isize| -> f32 {
        let nx = x as isize + dx;
        let ny = y as isize + dy;
        if nx < 0 || ny < 0 || nx >= width as isize || ny >= height as isize {
            0.0
        } else {
            data.get(nx as usize, ny as usize)
        }
    };

    let _center = get(0, 0);

    // Helper to sum 5 values along a line
    let sum5 = |offsets: &[(isize, isize); 5]| -> f32 {
        let mut s = 0.0;
        for &(dx, dy) in offsets {
            s += get(dx, dy);
        }
        s
    };

    let mut retval = 0.0f32;

    // Pattern 1: x grows, y constant (sparse horizontal)
    let sum = sum5(&[(-4, 0), (-2, 0), (0, 0), (2, 0), (4, 0)]);
    retval += sum * sum;

    // Pattern 2: y grows, x constant (sparse vertical)
    let sum = sum5(&[(0, -4), (0, -2), (0, 0), (0, 2), (0, 4)]);
    retval += sum * sum;

    // Pattern 3: both grow (diagonal)
    let sum = sum5(&[(-3, -3), (-2, -2), (0, 0), (2, 2), (3, 3)]);
    retval += sum * sum;

    // Pattern 4: y grows, x shrinks
    let sum = sum5(&[(3, -3), (2, -2), (0, 0), (-2, 2), (-3, 3)]);
    retval += sum * sum;

    // Pattern 5: y grows, x shifts 1 to -1
    let sum = sum5(&[(1, -4), (1, -2), (0, 0), (-1, 2), (-1, 4)]);
    retval += sum * sum;

    // Pattern 6: y grows, x shifts -1 to 1
    let sum = sum5(&[(-1, -4), (-1, -2), (0, 0), (1, 2), (1, 4)]);
    retval += sum * sum;

    // Pattern 7: x grows, y shifts -1 to 1
    let sum = sum5(&[(-4, -1), (-2, -1), (0, 0), (2, 1), (4, 1)]);
    retval += sum * sum;

    // Pattern 8: x grows, y shifts 1 to -1
    let sum = sum5(&[(-4, 1), (-2, 1), (0, 0), (2, -1), (4, -1)]);
    retval += sum * sum;

    // Pattern 9: steep slope
    let sum = sum5(&[(-2, -3), (-1, -2), (0, 0), (1, 2), (2, 3)]);
    retval += sum * sum;

    // Pattern 10: steep slope other way
    let sum = sum5(&[(2, -3), (1, -2), (0, 0), (-1, 2), (-2, 3)]);
    retval += sum * sum;

    // Pattern 11: shallow slope
    let sum = sum5(&[(-3, -2), (-2, -1), (0, 0), (2, 1), (3, 2)]);
    retval += sum * sum;

    // Pattern 12: shallow slope other way
    let sum = sum5(&[(3, -2), (2, -1), (0, 0), (-2, 1), (-3, 2)]);
    retval += sum * sum;

    // Pattern 13: curved path
    let sum = sum5(&[(-4, 2), (-2, 1), (0, 0), (2, -1), (4, -2)]);
    retval += sum * sum;

    // Pattern 14: curved other direction
    let sum = sum5(&[(-4, -2), (-2, -1), (0, 0), (2, 1), (4, 2)]);
    retval += sum * sum;

    // Pattern 15: vertical with shift
    let sum = sum5(&[(-2, -4), (-1, -2), (0, 0), (1, 2), (2, 4)]);
    retval += sum * sum;

    // Pattern 16: vertical other shift
    let sum = sum5(&[(2, -4), (1, -2), (0, 0), (-1, 2), (-2, 4)]);
    retval += sum * sum;

    retval
}

/// Computes the asymmetric Malta difference between two images.
///
/// This applies the Malta filter to the difference between two luminance images,
/// with asymmetric weighting to penalize artifacts differently than blur.
///
/// # Arguments
/// * `lum0` - First luminance image
/// * `lum1` - Second luminance image
/// * `w_0gt1` - Weight when original > reconstructed (penalize blurring)
/// * `w_0lt1` - Weight when original < reconstructed (penalize ringing)
/// * `norm1` - Normalization factor
/// * `use_lf` - If true, use LF variant of Malta filter
///
/// # Returns
/// Block difference AC map
pub fn malta_diff_map(
    lum0: &ImageF,
    lum1: &ImageF,
    w_0gt1: f64,
    w_0lt1: f64,
    norm1: f64,
    use_lf: bool,
) -> ImageF {
    let width = lum0.width();
    let height = lum0.height();

    // Constants from C++
    const K_WEIGHT0: f64 = 0.5;
    const K_WEIGHT1: f64 = 0.33;
    const LEN: f64 = 3.75;
    let mulli = if use_lf { 0.611612573796 } else { 0.39905817637 };

    let w_pre0gt1 = mulli * (K_WEIGHT0 * w_0gt1).sqrt() / (LEN * 2.0 + 1.0);
    let w_pre0lt1 = mulli * (K_WEIGHT1 * w_0lt1).sqrt() / (LEN * 2.0 + 1.0);
    let norm2_0gt1 = (w_pre0gt1 * norm1) as f32;
    let norm2_0lt1 = (w_pre0lt1 * norm1) as f32;

    // First pass: compute scaled differences
    let mut diffs = ImageF::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let v0 = lum0.get(x, y);
            let v1 = lum1.get(x, y);
            let absval = 0.5 * (v0.abs() + v1.abs());
            let diff = v0 - v1;
            let scaler = norm2_0gt1 / (norm1 as f32 + absval);

            // Primary symmetric quadratic objective
            let mut scaled_diff = scaler * diff;

            // Secondary half-open quadratic objectives
            let scaler2 = norm2_0lt1 / (norm1 as f32 + absval);
            let fabs0 = v0.abs();
            let too_small = 0.55 * fabs0;
            let too_big = 1.05 * fabs0;

            if v0 < 0.0 {
                if v1 > -too_small {
                    let impact = scaler2 * (v1 + too_small);
                    scaled_diff -= impact;
                } else if v1 < -too_big {
                    let impact = scaler2 * (-v1 - too_big);
                    scaled_diff += impact;
                }
            } else {
                if v1 < too_small {
                    let impact = scaler2 * (too_small - v1);
                    scaled_diff += impact;
                } else if v1 > too_big {
                    let impact = scaler2 * (v1 - too_big);
                    scaled_diff -= impact;
                }
            }

            diffs.set(x, y, scaled_diff);
        }
    }

    // Second pass: apply Malta filter
    let mut block_diff_ac = ImageF::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let malta_val = if use_lf {
                malta_unit_lf(&diffs, x, y)
            } else {
                malta_unit(&diffs, x, y)
            };
            block_diff_ac.set(x, y, malta_val);
        }
    }

    block_diff_ac
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_malta_uniform() {
        // Uniform image should have high Malta response (edge detection)
        let img = ImageF::filled(32, 32, 1.0);
        let center = malta_unit(&img, 16, 16);
        // 16 patterns × (9 samples × 1.0)² = 16 × 81 = 1296 for interior
        assert!(center > 0.0, "Malta should be positive for uniform image");
    }

    #[test]
    fn test_malta_edge() {
        // Image with strong vertical edge should have different response
        let mut img = ImageF::new(32, 32);
        for y in 0..32 {
            for x in 0..32 {
                if x < 16 {
                    img.set(x, y, 0.0);
                } else {
                    img.set(x, y, 1.0);
                }
            }
        }
        let edge = malta_unit(&img, 16, 16); // On the edge
        let uniform = malta_unit(&img, 8, 16); // In uniform region

        // Edge detection patterns should differ
        assert!(
            edge != uniform,
            "Malta should differ at edge vs uniform region"
        );
    }

    #[test]
    fn test_malta_diff_map_identical() {
        let img = ImageF::filled(32, 32, 0.5);
        let result = malta_diff_map(&img, &img, 1.0, 1.0, 1.0, false);

        // Identical images should have zero Malta diff
        let mut sum = 0.0;
        for y in 0..32 {
            for x in 0..32 {
                sum += result.get(x, y);
            }
        }
        assert!(sum.abs() < 1e-6, "Identical images should have zero diff");
    }

    #[test]
    fn test_malta_lf_smaller() {
        // LF variant uses fewer samples, should have different magnitude
        let img = ImageF::filled(32, 32, 1.0);
        let hf = malta_unit(&img, 16, 16);
        let lf = malta_unit_lf(&img, 16, 16);

        // Both should be positive
        assert!(hf > 0.0);
        assert!(lf > 0.0);
        // LF uses 5 samples instead of 9, so response should be different
        // 16 patterns × 25 for LF vs 16 × 81 for HF (for uniform image)
        assert!(lf < hf, "LF should have smaller response for uniform image");
    }
}
