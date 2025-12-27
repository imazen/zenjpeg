//! Multi-scale psychovisual decomposition for butteraugli.
//!
//! The PsychoImage struct holds frequency-decomposed versions of an image:
//! - UHF (Ultra High Frequency): Very fine details
//! - HF (High Frequency): Fine edges and textures
//! - MF (Medium Frequency): Larger-scale textures
//! - LF (Low Frequency): Smooth gradients and base colors
//!
//! This decomposition allows butteraugli to weight different spatial
//! frequencies according to human visual sensitivity.

use crate::blur::gaussian_blur;
use crate::consts::{
    ADD_HF_RANGE, ADD_MF_RANGE, BMUL_LF_TO_VALS, MAXCLAMP_HF, MAXCLAMP_UHF, MUL_Y_HF, MUL_Y_UHF,
    REMOVE_HF_RANGE, REMOVE_MF_RANGE, REMOVE_UHF_RANGE, SIGMA_HF, SIGMA_LF, SIGMA_UHF, SUPPRESS_S,
    SUPPRESS_XY, XMUL_LF_TO_VALS, YMUL_LF_TO_VALS, Y_TO_B_MUL_LF_TO_VALS,
};
use crate::image::{Image3F, ImageF};

/// Multi-scale psychovisual decomposition of an image.
///
/// Each frequency band captures different spatial features:
/// - `uhf`: Ultra high frequency (X, Y channels only)
/// - `hf`: High frequency (X, Y channels only)
/// - `mf`: Medium frequency (X, Y, B channels)
/// - `lf`: Low frequency (X, Y, B channels)
#[derive(Debug, Clone)]
pub struct PsychoImage {
    /// Ultra high frequency components (X, Y channels)
    pub uhf: [ImageF; 2],
    /// High frequency components (X, Y channels)
    pub hf: [ImageF; 2],
    /// Medium frequency components (X, Y, B channels)
    pub mf: Image3F,
    /// Low frequency components (X, Y, B channels)
    pub lf: Image3F,
}

impl PsychoImage {
    /// Creates a new PsychoImage with empty/zero images.
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            uhf: [ImageF::new(width, height), ImageF::new(width, height)],
            hf: [ImageF::new(width, height), ImageF::new(width, height)],
            mf: Image3F::new(width, height),
            lf: Image3F::new(width, height),
        }
    }

    /// Width of the images.
    #[must_use]
    pub fn width(&self) -> usize {
        self.lf.width()
    }

    /// Height of the images.
    #[must_use]
    pub fn height(&self) -> usize {
        self.lf.height()
    }
}

/// Removes values in a small range around zero.
///
/// Makes the area around zero less important by clamping small values to zero.
#[inline]
fn remove_range_around_zero(x: f32, range: f32) -> f32 {
    if x > range {
        x - range
    } else if x < -range {
        x + range
    } else {
        0.0
    }
}

/// Amplifies values in a small range around zero.
///
/// Makes the area around zero more important by doubling small values.
#[inline]
fn amplify_range_around_zero(x: f32, range: f32) -> f32 {
    if x > range {
        x + range
    } else if x < -range {
        x - range
    } else {
        x * 2.0
    }
}

/// Maximum clamp function from C++ butteraugli.
///
/// Compresses extreme values to prevent outliers from dominating.
#[inline]
fn maximum_clamp(v: f32, max_val: f32) -> f32 {
    const MUL: f32 = 0.724_216_146;
    if v >= max_val {
        (v - max_val).mul_add(MUL, max_val)
    } else if v <= -max_val {
        (v + max_val).mul_add(MUL, -max_val)
    } else {
        v
    }
}

/// Subtracts one image from another.
fn subtract(a: &ImageF, b: &ImageF, out: &mut ImageF) {
    for y in 0..a.height() {
        let row_a = a.row(y);
        let row_b = b.row(y);
        let row_out = out.row_mut(y);
        for x in 0..a.width() {
            row_out[x] = row_a[x] - row_b[x];
        }
    }
}

/// Converts low-frequency XYB to "vals" space for comparison.
///
/// Vals space can be converted to L2-norm space through visual masking.
fn xyb_low_freq_to_vals(lf: &mut Image3F) {
    let width = lf.width();
    let height = lf.height();

    // Collect all computed values first, then write back
    // This avoids borrow checker issues with multiple mutable borrows
    let mut new_vals = vec![(0.0f32, 0.0f32, 0.0f32); width * height];

    for y in 0..height {
        let row_x = lf.plane_row(0, y);
        let row_y = lf.plane_row(1, y);
        let row_b = lf.plane_row(2, y);

        for x in 0..width {
            let vx = row_x[x];
            let vy = row_y[x];
            let vb = row_b[x];

            let b = (Y_TO_B_MUL_LF_TO_VALS as f32).mul_add(vy, vb);
            let val_b = b * BMUL_LF_TO_VALS as f32;
            let val_x = vx * XMUL_LF_TO_VALS as f32;
            let val_y = vy * YMUL_LF_TO_VALS as f32;

            new_vals[y * width + x] = (val_x, val_y, val_b);
        }
    }

    // Write back the computed values
    for y in 0..height {
        for x in 0..width {
            let (vx, vy, vb) = new_vals[y * width + x];
            lf.plane_mut(0).set(x, y, vx);
            lf.plane_mut(1).set(x, y, vy);
            lf.plane_mut(2).set(x, y, vb);
        }
    }
}

/// Suppresses X channel based on Y channel values.
///
/// High Y (luminance) values reduce sensitivity to X (chroma) differences.
fn suppress_x_by_y(in_y: &ImageF, inout_x: &mut ImageF) {
    let width = in_y.width();
    let height = in_y.height();

    let s = SUPPRESS_S as f32;
    let one_minus_s = 1.0 - s;
    let yw = SUPPRESS_XY as f32;

    for y in 0..height {
        let row_y = in_y.row(y);
        let row_x = inout_x.row_mut(y);
        for x in 0..width {
            let vy = row_y[x];
            let vx = row_x[x];
            let scaler = (yw / vy.mul_add(vy, yw)).mul_add(one_minus_s, s);
            row_x[x] = scaler * vx;
        }
    }
}

/// Separates LF (low frequency) and MF (medium frequency) components.
fn separate_lf_and_mf(xyb: &Image3F, lf: &mut Image3F, mf: &mut Image3F) {
    let sigma = SIGMA_LF as f32;

    for i in 0..3 {
        // Extract LF via blur
        let blurred = gaussian_blur(xyb.plane(i), sigma);
        lf.plane_mut(i).copy_from(&blurred);

        // MF = original - LF
        subtract(xyb.plane(i), &blurred, mf.plane_mut(i));
    }

    // Convert LF to vals space
    xyb_low_freq_to_vals(lf);
}

/// Separates MF (medium frequency) and HF (high frequency) components.
fn separate_mf_and_hf(mf: &mut Image3F, hf: &mut [ImageF; 2]) {
    let width = mf.width();
    let height = mf.height();
    let sigma = SIGMA_HF as f32;

    // Process X and Y channels
    for i in 0..2 {
        // Copy to HF before blurring
        hf[i].copy_from(mf.plane(i));

        // Blur MF
        let blurred = gaussian_blur(mf.plane(i), sigma);
        mf.plane_mut(i).copy_from(&blurred);

        // HF = original - blurred
        let range = if i == 0 {
            REMOVE_MF_RANGE
        } else {
            ADD_MF_RANGE
        };

        for y in 0..height {
            let row_mf = mf.plane_row_mut(i, y);
            let row_hf = hf[i].row_mut(y);
            for x in 0..width {
                let hf_val = row_hf[x] - row_mf[x];
                row_hf[x] = hf_val;

                if i == 0 {
                    row_mf[x] = remove_range_around_zero(row_mf[x], range as f32);
                } else {
                    row_mf[x] = amplify_range_around_zero(row_mf[x], range as f32);
                }
            }
        }
    }

    // Blur B channel only (no HF/UHF for blue)
    let blurred_b = gaussian_blur(mf.plane(2), sigma);
    mf.plane_mut(2).copy_from(&blurred_b);

    // Suppress X by Y in HF
    let hf_y_copy = hf[1].clone();
    suppress_x_by_y(&hf_y_copy, &mut hf[0]);
}

/// Separates HF (high frequency) and UHF (ultra high frequency) components.
fn separate_hf_and_uhf(hf: &mut [ImageF; 2], uhf: &mut [ImageF; 2]) {
    let width = hf[0].width();
    let height = hf[0].height();
    let sigma = SIGMA_UHF as f32;

    for i in 0..2 {
        // Copy to UHF before blurring
        uhf[i].copy_from(&hf[i]);

        // Blur HF
        let blurred = gaussian_blur(&hf[i], sigma);
        hf[i].copy_from(&blurred);

        // UHF = original - blurred, with adjustments
        for y in 0..height {
            let row_hf = hf[i].row_mut(y);
            let row_uhf = uhf[i].row_mut(y);

            for x in 0..width {
                let hf_val = row_hf[x];

                if i == 0 {
                    // X channel: compute UHF before any clamping
                    let uhf_val = row_uhf[x] - hf_val;
                    row_hf[x] = remove_range_around_zero(hf_val, REMOVE_HF_RANGE as f32);
                    row_uhf[x] = remove_range_around_zero(uhf_val, REMOVE_UHF_RANGE as f32);
                } else {
                    // Y channel: C++ clamps HF BEFORE computing UHF
                    // This is critical - the subtraction uses the clamped value
                    let hf_clamped = maximum_clamp(hf_val, MAXCLAMP_HF as f32);
                    let uhf_val = row_uhf[x] - hf_clamped; // Use CLAMPED HF
                    let uhf_clamped = maximum_clamp(uhf_val, MAXCLAMP_UHF as f32);

                    row_uhf[x] = uhf_clamped * MUL_Y_UHF as f32;
                    row_hf[x] = amplify_range_around_zero(
                        hf_clamped * MUL_Y_HF as f32,
                        ADD_HF_RANGE as f32,
                    );
                }
            }
        }
    }
}

/// Performs the full frequency decomposition on an XYB image.
///
/// This is the main entry point for creating a PsychoImage from
/// an XYB color-space image.
pub fn separate_frequencies(xyb: &Image3F) -> PsychoImage {
    let width = xyb.width();
    let height = xyb.height();

    let mut ps = PsychoImage::new(width, height);

    // Separate into LF and MF
    separate_lf_and_mf(xyb, &mut ps.lf, &mut ps.mf);

    // Separate MF into MF and HF
    separate_mf_and_hf(&mut ps.mf, &mut ps.hf);

    // Separate HF into HF and UHF
    separate_hf_and_uhf(&mut ps.hf, &mut ps.uhf);

    ps
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_remove_range_around_zero() {
        assert!((remove_range_around_zero(0.5, 0.1) - 0.4).abs() < 0.001);
        assert!((remove_range_around_zero(-0.5, 0.1) - (-0.4)).abs() < 0.001);
        assert!((remove_range_around_zero(0.05, 0.1) - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_amplify_range_around_zero() {
        assert!((amplify_range_around_zero(0.5, 0.1) - 0.6).abs() < 0.001);
        assert!((amplify_range_around_zero(-0.5, 0.1) - (-0.6)).abs() < 0.001);
        assert!((amplify_range_around_zero(0.05, 0.1) - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_maximum_clamp() {
        assert!((maximum_clamp(5.0, 10.0) - 5.0).abs() < 0.001);
        assert!(maximum_clamp(15.0, 10.0) < 15.0);
        assert!(maximum_clamp(15.0, 10.0) > 10.0);
    }

    #[test]
    fn test_psycho_image_creation() {
        let ps = PsychoImage::new(100, 50);
        assert_eq!(ps.width(), 100);
        assert_eq!(ps.height(), 50);
    }

    #[test]
    fn test_frequency_separation() {
        // Create a simple XYB image
        let mut xyb = Image3F::new(32, 32);
        for y in 0..32 {
            for x in 0..32 {
                // Add some variation
                let val = (x + y) as f32 / 64.0;
                xyb.plane_mut(0).set(x, y, val * 0.1);
                xyb.plane_mut(1).set(x, y, val);
                xyb.plane_mut(2).set(x, y, val * 0.5);
            }
        }

        let ps = separate_frequencies(&xyb);

        // Just verify it runs and produces valid data
        assert_eq!(ps.width(), 32);
        assert_eq!(ps.height(), 32);
    }
}
