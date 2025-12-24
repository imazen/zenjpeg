//! Visual masking functions for butteraugli.
//!
//! Visual masking is the phenomenon where the visibility of one feature
//! is reduced by the presence of another feature. This module implements
//! the masking computations used in butteraugli.

use crate::image::ImageF;

/// Combines HF and UHF channels for masking computation.
///
/// Only X and Y components are involved in masking. B's influence
/// is considered less important in the high frequency area.
pub fn combine_channels_for_masking(hf: &[ImageF; 2], uhf: &[ImageF; 2], out: &mut ImageF) {
    const MULS: [f32; 3] = [2.5, 0.4, 0.4];

    let width = hf[0].width();
    let height = hf[0].height();

    for y in 0..height {
        let row_y_hf = hf[1].row(y);
        let row_y_uhf = uhf[1].row(y);
        let row_x_hf = hf[0].row(y);
        let row_x_uhf = uhf[0].row(y);
        let row_out = out.row_mut(y);

        for x in 0..width {
            let xdiff = (row_x_uhf[x] + row_x_hf[x]) * MULS[0];
            let ydiff = row_y_uhf[x].mul_add(MULS[1], row_y_hf[x] * MULS[2]);
            row_out[x] = (xdiff.mul_add(xdiff, ydiff * ydiff)).sqrt();
        }
    }
}

/// Precomputes difference values for masking.
///
/// Applies sqrt-like transformation to make values more perceptually uniform.
pub fn diff_precompute(xyb: &ImageF, mul: f32, bias_arg: f32, out: &mut ImageF) {
    let width = xyb.width();
    let height = xyb.height();
    let bias = mul * bias_arg;
    let sqrt_bias = bias.sqrt();

    for y in 0..height {
        let row_in = xyb.row(y);
        let row_out = out.row_mut(y);
        for x in 0..width {
            // sqrt with bias for numerical stability
            row_out[x] = row_in[x].abs().mul_add(mul, bias).sqrt() - sqrt_bias;
        }
    }
}

/// Stores the three smallest values encountered.
#[inline]
fn store_min3(v: f32, min0: &mut f32, min1: &mut f32, min2: &mut f32) {
    if v < *min2 {
        if v < *min0 {
            *min2 = *min1;
            *min1 = *min0;
            *min0 = v;
        } else if v < *min1 {
            *min2 = *min1;
            *min1 = v;
        } else {
            *min2 = v;
        }
    }
}

/// Performs fuzzy erosion to find smooth areas.
///
/// Look for smooth areas near the area of degradation.
/// If the areas are generally smooth, don't apply masking.
pub fn fuzzy_erosion(from: &ImageF, to: &mut ImageF) {
    let width = from.width();
    let height = from.height();
    const STEP: i32 = 3;

    for y in 0..height {
        for x in 0..width {
            let mut min0 = from.get(x, y);
            let mut min1 = 2.0 * min0;
            let mut min2 = min1;

            // Check 8 neighbors at distance STEP
            let neighbors: [(i32, i32); 8] = [
                (-STEP, 0),
                (STEP, 0),
                (0, -STEP),
                (0, STEP),
                (-STEP, -STEP),
                (-STEP, STEP),
                (STEP, -STEP),
                (STEP, STEP),
            ];

            for (dx, dy) in neighbors {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                    store_min3(
                        from.get(nx as usize, ny as usize),
                        &mut min0,
                        &mut min1,
                        &mut min2,
                    );
                }
            }

            // Weighted average of the three smallest values
            to.set(x, y, (min0 + min1 + min2) / 3.0);
        }
    }
}

/// Applies visual masking based on local contrast.
///
/// Higher local contrast means differences are less visible (masked).
pub fn apply_masking(diff: &ImageF, mask: &ImageF, out: &mut ImageF) {
    let width = diff.width();
    let height = diff.height();

    for y in 0..height {
        let row_diff = diff.row(y);
        let row_mask = mask.row(y);
        let row_out = out.row_mut(y);

        for x in 0..width {
            // Higher mask value means lower sensitivity
            let sensitivity = 1.0 / (1.0 + row_mask[x]);
            row_out[x] = row_diff[x] * sensitivity;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_store_min3() {
        let mut min0 = 10.0f32;
        let mut min1 = 20.0f32;
        let mut min2 = 30.0f32;

        store_min3(5.0, &mut min0, &mut min1, &mut min2);
        assert!((min0 - 5.0).abs() < 0.001);
        assert!((min1 - 10.0).abs() < 0.001);
        assert!((min2 - 20.0).abs() < 0.001);

        store_min3(15.0, &mut min0, &mut min1, &mut min2);
        assert!((min0 - 5.0).abs() < 0.001);
        assert!((min1 - 10.0).abs() < 0.001);
        assert!((min2 - 15.0).abs() < 0.001);
    }

    #[test]
    fn test_fuzzy_erosion() {
        let mut from = ImageF::new(16, 16);
        // Create a pattern with one bright spot
        from.set(8, 8, 10.0);
        from.set(5, 5, 5.0);

        let mut to = ImageF::new(16, 16);
        fuzzy_erosion(&from, &mut to);

        // The output should be smoother
        // The bright spot should be somewhat reduced
        assert!(to.get(8, 8) <= 10.0);
    }

    #[test]
    fn test_diff_precompute() {
        let input = ImageF::filled(16, 16, 1.0);
        let mut output = ImageF::new(16, 16);

        diff_precompute(&input, 1.0, 0.01, &mut output);

        // All values should be positive
        for y in 0..16 {
            for x in 0..16 {
                assert!(output.get(x, y) >= 0.0);
            }
        }
    }
}
