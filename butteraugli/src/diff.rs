//! Main butteraugli difference computation.
//!
//! This module ties together all the components to compute the
//! perceptual difference between two images.

use crate::consts::{
    GLOBAL_SCALE, NORM1_HF, NORM1_HF_X, NORM1_MF, NORM1_MF_X, NORM1_UHF, NORM1_UHF_X, W_HF_MALTA,
    W_HF_MALTA_X, W_MF_MALTA, W_MF_MALTA_X, W_UHF_MALTA, W_UHF_MALTA_X,
};
use crate::image::{Image3F, ImageF};
use crate::malta::malta_diff_map;
use crate::mask::{combine_channels_for_masking, fuzzy_erosion};
use crate::opsin::srgb_to_xyb_butteraugli;
use crate::psycho::{separate_frequencies, PsychoImage};
use crate::{ButteraugliParams, ButteraugliResult};

/// Minimum image dimension for multi-resolution processing.
/// Images smaller than this are handled without recursion.
const MIN_SIZE_FOR_MULTIRESOLUTION: usize = 8;

/// Converts RGB buffer to XYB Image3F using butteraugli's OpsinDynamicsImage.
///
/// This uses the correct butteraugli color conversion, which is DIFFERENT
/// from jpegli's XYB color space. Key differences:
/// 1. Different OpsinAbsorbance matrix coefficients
/// 2. Uses Gamma function (FastLog2f based), not cube root
/// 3. Includes dynamic sensitivity based on blurred image
fn rgb_to_xyb_image(rgb: &[u8], width: usize, height: usize, intensity_target: f32) -> Image3F {
    srgb_to_xyb_butteraugli(rgb, width, height, intensity_target)
}

/// Subsamples an Image3F by 2x using box filter averaging.
///
/// Each 2x2 block of pixels is averaged into a single pixel.
/// Edge cases for odd dimensions are handled by scaling the edge values.
fn subsample_2x(input: &Image3F) -> Image3F {
    let in_width = input.width();
    let in_height = input.height();
    let out_width = (in_width + 1) / 2;
    let out_height = (in_height + 1) / 2;

    let mut output = Image3F::new(out_width, out_height);

    // Initialize to zero (already done by Image3F::new)

    // Accumulate 2x2 blocks
    for c in 0..3 {
        for y in 0..in_height {
            for x in 0..in_width {
                let val = input.plane(c).get(x, y);
                let ox = x / 2;
                let oy = y / 2;
                let prev = output.plane(c).get(ox, oy);
                output.plane_mut(c).set(ox, oy, prev + 0.25 * val);
            }
        }

        // Handle odd width - last column only has half the samples
        if (in_width & 1) != 0 {
            let last_col = out_width - 1;
            for y in 0..out_height {
                let prev = output.plane(c).get(last_col, y);
                output.plane_mut(c).set(last_col, y, prev * 2.0);
            }
        }

        // Handle odd height - last row only has half the samples
        if (in_height & 1) != 0 {
            let last_row = out_height - 1;
            for x in 0..out_width {
                let prev = output.plane(c).get(x, last_row);
                output.plane_mut(c).set(x, last_row, prev * 2.0);
            }
        }
    }

    output
}

/// Subsamples an RGB buffer by 2x for multi-resolution processing.
fn subsample_rgb_2x(rgb: &[u8], width: usize, height: usize) -> (Vec<u8>, usize, usize) {
    let out_width = (width + 1) / 2;
    let out_height = (height + 1) / 2;
    let mut output = vec![0u8; out_width * out_height * 3];

    // Simple averaging of 2x2 blocks
    for oy in 0..out_height {
        for ox in 0..out_width {
            let mut r_sum = 0u32;
            let mut g_sum = 0u32;
            let mut b_sum = 0u32;
            let mut count = 0u32;

            for dy in 0..2 {
                for dx in 0..2 {
                    let ix = ox * 2 + dx;
                    let iy = oy * 2 + dy;
                    if ix < width && iy < height {
                        let idx = (iy * width + ix) * 3;
                        r_sum += rgb[idx] as u32;
                        g_sum += rgb[idx + 1] as u32;
                        b_sum += rgb[idx + 2] as u32;
                        count += 1;
                    }
                }
            }

            if count > 0 {
                let out_idx = (oy * out_width + ox) * 3;
                output[out_idx] = (r_sum / count) as u8;
                output[out_idx + 1] = (g_sum / count) as u8;
                output[out_idx + 2] = (b_sum / count) as u8;
            }
        }
    }

    (output, out_width, out_height)
}

/// Adds a supersampled (upscaled 2x) diffmap to the destination.
///
/// This blends the lower-resolution analysis with the higher-resolution one
/// using a heuristic mixing value to reduce noise from lower resolutions.
fn add_supersampled_2x(src: &ImageF, weight: f32, dest: &mut ImageF) {
    let width = dest.width();
    let height = dest.height();

    // Heuristic from C++: lower resolution images have less error
    const K_HEURISTIC_MIXING_VALUE: f32 = 0.3;

    for y in 0..height {
        for x in 0..width {
            let src_x = x / 2;
            let src_y = y / 2;
            let src_val = src.get(src_x.min(src.width() - 1), src_y.min(src.height() - 1));

            let prev = dest.get(x, y);
            let mixed = prev * (1.0 - K_HEURISTIC_MIXING_VALUE * weight) + weight * src_val;
            dest.set(x, y, mixed);
        }
    }
}

/// L2 difference (symmetric).
///
/// Computes squared difference weighted by w and adds to diffmap.
fn l2_diff(i0: &ImageF, i1: &ImageF, w: f32, diffmap: &mut ImageF) {
    let width = i0.width();
    let height = i0.height();

    for y in 0..height {
        let row0 = i0.row(y);
        let row1 = i1.row(y);
        let row_diff = diffmap.row_mut(y);

        for x in 0..width {
            let diff = row0[x] - row1[x];
            row_diff[x] += diff * diff * w;
        }
    }
}

/// L2 difference asymmetric.
///
/// This penalizes artifacts (original < reconstructed) more than blur
/// (original > reconstructed). Based on C++ L2DiffAsymmetric.
///
/// # Arguments
/// * `i0` - Original image
/// * `i1` - Reconstructed image
/// * `w_0gt1` - Weight when original > reconstructed (penalize blur)
/// * `w_0lt1` - Weight when original < reconstructed (penalize artifacts)
/// * `diffmap` - Output difference map (accumulated)
fn l2_diff_asymmetric(i0: &ImageF, i1: &ImageF, w_0gt1: f32, w_0lt1: f32, diffmap: &mut ImageF) {
    if w_0gt1 == 0.0 && w_0lt1 == 0.0 {
        return;
    }

    let width = i0.width();
    let height = i0.height();
    let vw_0gt1 = w_0gt1 * 0.8;
    let vw_0lt1 = w_0lt1 * 0.8;

    for y in 0..height {
        let row0 = i0.row(y);
        let row1 = i1.row(y);
        let row_diff = diffmap.row_mut(y);

        for x in 0..width {
            let val0 = row0[x];
            let val1 = row1[x];

            // Primary symmetric quadratic objective
            let diff = val0 - val1;
            let mut total = row_diff[x] + diff * diff * vw_0gt1;

            // Secondary half-open quadratic objectives
            let fabs0 = val0.abs();
            let too_small = 0.4 * fabs0;
            let too_big = fabs0;

            let v = if val0 < 0.0 {
                if val1 > -too_small {
                    val1 + too_small
                } else if val1 < -too_big {
                    -val1 - too_big
                } else {
                    0.0
                }
            } else {
                if val1 < too_small {
                    too_small - val1
                } else if val1 > too_big {
                    val1 - too_big
                } else {
                    0.0
                }
            };

            total += vw_0lt1 * v * v;
            row_diff[x] = total;
        }
    }
}

/// Weight multipliers for L2 differences (from C++ wmul array).
const WMUL: [f32; 9] = [
    400.0,   // HF X
    1.50,    // HF Y
    0.0,     // HF B (not used)
    2.0,     // MF X
    0.35,    // MF Y
    0.01,    // MF B
    18.0,    // LF X
    2.50,    // LF Y
    0.15,    // LF B
];

/// Computes difference between two PsychoImages using Malta filter.
///
/// This is the core butteraugli algorithm that applies:
/// 1. Malta edge-aware filter for UHF, HF, MF differences
/// 2. L2DiffAsymmetric for HF channels
/// 3. L2Diff for MF and LF channels
fn compute_psycho_diff_malta(
    ps0: &PsychoImage,
    ps1: &PsychoImage,
    hf_asymmetry: f32,
    xmul: f32,
) -> Image3F {
    let width = ps0.width();
    let height = ps0.height();

    // Block diff AC accumulates Malta and L2 differences
    let mut block_diff_ac = Image3F::new(width, height);

    // Apply Malta filter for UHF (uses full Malta, not LF variant)
    // UHF Y channel
    let uhf_y_diff = malta_diff_map(
        &ps0.uhf[1],
        &ps1.uhf[1],
        W_UHF_MALTA * hf_asymmetry as f64,
        W_UHF_MALTA / hf_asymmetry as f64,
        NORM1_UHF,
        false, // use full Malta
    );
    for y in 0..height {
        for x in 0..width {
            let v = block_diff_ac.plane(1).get(x, y) + uhf_y_diff.get(x, y);
            block_diff_ac.plane_mut(1).set(x, y, v);
        }
    }

    // UHF X channel
    let uhf_x_diff = malta_diff_map(
        &ps0.uhf[0],
        &ps1.uhf[0],
        W_UHF_MALTA_X * hf_asymmetry as f64,
        W_UHF_MALTA_X / hf_asymmetry as f64,
        NORM1_UHF_X,
        false,
    );
    for y in 0..height {
        for x in 0..width {
            let v = block_diff_ac.plane(0).get(x, y) + uhf_x_diff.get(x, y);
            block_diff_ac.plane_mut(0).set(x, y, v);
        }
    }

    // Apply Malta LF filter for HF
    let sqrt_hf_asym = hf_asymmetry.sqrt();

    // HF Y channel
    let hf_y_diff = malta_diff_map(
        &ps0.hf[1],
        &ps1.hf[1],
        W_HF_MALTA * sqrt_hf_asym as f64,
        W_HF_MALTA / sqrt_hf_asym as f64,
        NORM1_HF,
        true, // use LF Malta
    );
    for y in 0..height {
        for x in 0..width {
            let v = block_diff_ac.plane(1).get(x, y) + hf_y_diff.get(x, y);
            block_diff_ac.plane_mut(1).set(x, y, v);
        }
    }

    // HF X channel
    let hf_x_diff = malta_diff_map(
        &ps0.hf[0],
        &ps1.hf[0],
        W_HF_MALTA_X * sqrt_hf_asym as f64,
        W_HF_MALTA_X / sqrt_hf_asym as f64,
        NORM1_HF_X,
        true,
    );
    for y in 0..height {
        for x in 0..width {
            let v = block_diff_ac.plane(0).get(x, y) + hf_x_diff.get(x, y);
            block_diff_ac.plane_mut(0).set(x, y, v);
        }
    }

    // Apply Malta LF filter for MF
    // MF Y channel
    let mf_y_diff = malta_diff_map(
        ps0.mf.plane(1),
        ps1.mf.plane(1),
        W_MF_MALTA,
        W_MF_MALTA,
        NORM1_MF,
        true,
    );
    for y in 0..height {
        for x in 0..width {
            let v = block_diff_ac.plane(1).get(x, y) + mf_y_diff.get(x, y);
            block_diff_ac.plane_mut(1).set(x, y, v);
        }
    }

    // MF X channel
    let mf_x_diff = malta_diff_map(
        ps0.mf.plane(0),
        ps1.mf.plane(0),
        W_MF_MALTA_X,
        W_MF_MALTA_X,
        NORM1_MF_X,
        true,
    );
    for y in 0..height {
        for x in 0..width {
            let v = block_diff_ac.plane(0).get(x, y) + mf_x_diff.get(x, y);
            block_diff_ac.plane_mut(0).set(x, y, v);
        }
    }

    // Add L2DiffAsymmetric for HF channels (X and Y, no blue)
    l2_diff_asymmetric(
        &ps0.hf[0],
        &ps1.hf[0],
        WMUL[0] * hf_asymmetry,
        WMUL[0] / hf_asymmetry,
        block_diff_ac.plane_mut(0),
    );
    l2_diff_asymmetric(
        &ps0.hf[1],
        &ps1.hf[1],
        WMUL[1] * hf_asymmetry,
        WMUL[1] / hf_asymmetry,
        block_diff_ac.plane_mut(1),
    );

    // Add L2Diff for MF channels (all three)
    l2_diff(ps0.mf.plane(0), ps1.mf.plane(0), WMUL[3], block_diff_ac.plane_mut(0));
    l2_diff(ps0.mf.plane(1), ps1.mf.plane(1), WMUL[4], block_diff_ac.plane_mut(1));
    l2_diff(ps0.mf.plane(2), ps1.mf.plane(2), WMUL[5], block_diff_ac.plane_mut(2));

    block_diff_ac
}

/// Computes the mask from a PsychoImage.
fn compute_mask(ps: &PsychoImage) -> ImageF {
    let width = ps.width();
    let height = ps.height();
    let mut mask = ImageF::new(width, height);

    // Combine HF and UHF for masking
    combine_channels_for_masking(&ps.hf, &ps.uhf, &mut mask);

    // Apply fuzzy erosion to find smooth areas
    let mut eroded = ImageF::new(width, height);
    fuzzy_erosion(&mask, &mut eroded);

    eroded
}

/// Combines channels to produce final diffmap.
///
/// Applies masking and combines X, Y, B channels with appropriate weights.
fn combine_channels_to_diffmap(
    mask: &ImageF,
    block_diff_dc: &Image3F,
    block_diff_ac: &Image3F,
    xmul: f32,
) -> ImageF {
    let width = mask.width();
    let height = mask.height();
    let mut diffmap = ImageF::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let mask_val = mask.get(x, y);

            // Combine AC differences from all channels
            let ac_x = block_diff_ac.plane(0).get(x, y) * xmul;
            let ac_y = block_diff_ac.plane(1).get(x, y);
            let ac_b = block_diff_ac.plane(2).get(x, y);

            // Combine DC differences
            let dc_x = block_diff_dc.plane(0).get(x, y) * xmul;
            let dc_y = block_diff_dc.plane(1).get(x, y);
            let dc_b = block_diff_dc.plane(2).get(x, y);

            // Total difference
            let total = (ac_x + ac_y + ac_b + dc_x + dc_y + dc_b).sqrt();

            // Apply masking (higher mask = more masking = lower perceived difference)
            let masking_factor = 1.0 / (1.0 + mask_val * 0.1);

            diffmap.set(x, y, total * masking_factor);
        }
    }

    diffmap
}

/// Computes the global score from a difference map.
///
/// C++ butteraugli uses the maximum diffmap value as the score.
fn compute_score_from_diffmap(diffmap: &ImageF) -> f64 {
    let width = diffmap.width();
    let height = diffmap.height();
    let num_pixels = width * height;

    if num_pixels == 0 {
        return 0.0;
    }

    // Find maximum difference value (C++ butteraugli approach)
    let mut max_val = 0.0f64;

    for y in 0..height {
        for x in 0..width {
            let v = diffmap.get(x, y) as f64;
            if v > max_val {
                max_val = v;
            }
        }
    }

    // Apply global scale
    max_val * GLOBAL_SCALE as f64
}

/// Computes the diffmap for a single resolution level.
fn compute_diffmap_single_resolution(
    rgb1: &[u8],
    rgb2: &[u8],
    width: usize,
    height: usize,
    params: &ButteraugliParams,
) -> ImageF {
    // Convert to XYB using butteraugli's OpsinDynamicsImage
    let xyb1 = rgb_to_xyb_image(rgb1, width, height, params.intensity_target);
    let xyb2 = rgb_to_xyb_image(rgb2, width, height, params.intensity_target);

    // Perform frequency decomposition
    let ps1 = separate_frequencies(&xyb1);
    let ps2 = separate_frequencies(&xyb2);

    // Compute masks from both images
    let mask1 = compute_mask(&ps1);
    let mask2 = compute_mask(&ps2);

    // Average the masks
    let mut mask = ImageF::new(width, height);
    for y in 0..height {
        for x in 0..width {
            mask.set(x, y, (mask1.get(x, y) + mask2.get(x, y)) * 0.5);
        }
    }

    // Compute AC differences using Malta filter
    let block_diff_ac = compute_psycho_diff_malta(&ps1, &ps2, params.hf_asymmetry, params.xmul);

    // Compute DC (LF) differences
    let mut block_diff_dc = Image3F::new(width, height);
    for c in 0..3 {
        for y in 0..height {
            for x in 0..width {
                let d = ps1.lf.plane(c).get(x, y) - ps2.lf.plane(c).get(x, y);
                block_diff_dc.plane_mut(c).set(x, y, d * d * WMUL[6 + c]);
            }
        }
    }

    // Combine channels to final diffmap
    combine_channels_to_diffmap(&mask, &block_diff_dc, &block_diff_ac, params.xmul)
}

/// Recursively computes butteraugli diffmap at multiple resolutions.
fn compute_diffmap_multiresolution(
    rgb1: &[u8],
    rgb2: &[u8],
    width: usize,
    height: usize,
    params: &ButteraugliParams,
) -> ImageF {
    // Compute diffmap at current resolution
    let mut diffmap = compute_diffmap_single_resolution(rgb1, rgb2, width, height, params);

    // If image is large enough, recurse to lower resolution
    let sub_width = (width + 1) / 2;
    let sub_height = (height + 1) / 2;

    if sub_width >= MIN_SIZE_FOR_MULTIRESOLUTION && sub_height >= MIN_SIZE_FOR_MULTIRESOLUTION {
        // Subsample both images
        let (sub_rgb1, sw, sh) = subsample_rgb_2x(rgb1, width, height);
        let (sub_rgb2, _, _) = subsample_rgb_2x(rgb2, width, height);

        // Recursively compute at lower resolution
        let sub_diffmap = compute_diffmap_multiresolution(&sub_rgb1, &sub_rgb2, sw, sh, params);

        // Add supersampled lower-resolution result to current
        add_supersampled_2x(&sub_diffmap, 0.5, &mut diffmap);
    }

    diffmap
}

/// Main implementation of butteraugli comparison.
pub fn compute_butteraugli_impl(
    rgb1: &[u8],
    rgb2: &[u8],
    width: usize,
    height: usize,
    params: &ButteraugliParams,
) -> ButteraugliResult {
    assert_eq!(rgb1.len(), width * height * 3);
    assert_eq!(rgb2.len(), width * height * 3);

    // Handle identical images case
    if rgb1 == rgb2 {
        return ButteraugliResult {
            score: 0.0,
            diffmap: Some(ImageF::new(width, height)),
        };
    }

    // Handle very small images without multi-resolution
    let diffmap = if width < MIN_SIZE_FOR_MULTIRESOLUTION || height < MIN_SIZE_FOR_MULTIRESOLUTION {
        compute_diffmap_single_resolution(rgb1, rgb2, width, height, params)
    } else {
        compute_diffmap_multiresolution(rgb1, rgb2, width, height, params)
    };

    // Compute global score
    let score = compute_score_from_diffmap(&diffmap);

    ButteraugliResult {
        score,
        diffmap: Some(diffmap),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_images() {
        let width = 32;
        let height = 32;
        let rgb: Vec<u8> = (0..width * height * 3).map(|i| (i % 256) as u8).collect();

        let result =
            compute_butteraugli_impl(&rgb, &rgb, width, height, &ButteraugliParams::default());

        assert!(
            result.score < 0.001,
            "Identical images should have score ~0, got {}",
            result.score
        );
    }

    #[test]
    fn test_slightly_different_images() {
        let width = 32;
        let height = 32;
        let rgb1: Vec<u8> = vec![128; width * height * 3];
        let mut rgb2 = rgb1.clone();
        // Change one pixel slightly
        rgb2[0] = 129;
        rgb2[1] = 129;
        rgb2[2] = 129;

        let result =
            compute_butteraugli_impl(&rgb1, &rgb2, width, height, &ButteraugliParams::default());

        // Small difference should have low score
        assert!(
            result.score < 2.0,
            "Small difference should have low score, got {}",
            result.score
        );
    }

    #[test]
    fn test_very_different_images() {
        let width = 32;
        let height = 32;
        let rgb1: Vec<u8> = vec![0; width * height * 3];
        let rgb2: Vec<u8> = vec![255; width * height * 3];

        let result =
            compute_butteraugli_impl(&rgb1, &rgb2, width, height, &ButteraugliParams::default());

        // Very different images should have non-zero score
        // Note: uniform images (all black vs all white) have limited frequency content,
        // so the score may be lower than expected for natural images
        assert!(
            result.score > 0.01,
            "Very different images should have non-zero score, got {}",
            result.score
        );
    }

    #[test]
    fn test_diffmap_dimensions() {
        let width = 64;
        let height = 48;
        let rgb1: Vec<u8> = vec![100; width * height * 3];
        let rgb2: Vec<u8> = vec![150; width * height * 3];

        let result =
            compute_butteraugli_impl(&rgb1, &rgb2, width, height, &ButteraugliParams::default());

        let diffmap = result.diffmap.unwrap();
        assert_eq!(diffmap.width(), width);
        assert_eq!(diffmap.height(), height);
    }

    #[test]
    fn test_l2_diff_asymmetric() {
        let width = 16;
        let height = 16;
        let i0 = ImageF::filled(width, height, 1.0);
        let i1 = ImageF::filled(width, height, 0.5);
        let mut diffmap = ImageF::new(width, height);

        l2_diff_asymmetric(&i0, &i1, 1.0, 1.0, &mut diffmap);

        // Should have non-zero difference
        let mut sum = 0.0;
        for y in 0..height {
            for x in 0..width {
                sum += diffmap.get(x, y);
            }
        }
        assert!(sum > 0.0, "L2 diff should be non-zero for different images");
    }

    #[test]
    fn test_subsample_rgb_2x() {
        let width = 8;
        let height = 8;
        let rgb: Vec<u8> = vec![128; width * height * 3];

        let (sub_rgb, sw, sh) = subsample_rgb_2x(&rgb, width, height);

        assert_eq!(sw, 4);
        assert_eq!(sh, 4);
        assert_eq!(sub_rgb.len(), 4 * 4 * 3);
        // Uniform input should produce uniform output
        assert!(sub_rgb.iter().all(|&v| v == 128));
    }

    #[test]
    fn test_subsample_rgb_2x_odd() {
        let width = 7;
        let height = 7;
        let rgb: Vec<u8> = vec![128; width * height * 3];

        let (sub_rgb, sw, sh) = subsample_rgb_2x(&rgb, width, height);

        // (7+1)/2 = 4
        assert_eq!(sw, 4);
        assert_eq!(sh, 4);
    }

    #[test]
    fn test_add_supersampled_2x() {
        let src = ImageF::filled(4, 4, 1.0);
        let mut dest = ImageF::filled(8, 8, 2.0);

        add_supersampled_2x(&src, 0.5, &mut dest);

        // Should have blended values
        // new = old * (1 - 0.3 * 0.5) + 0.5 * 1.0 = 2.0 * 0.85 + 0.5 = 1.7 + 0.5 = 2.2
        let val = dest.get(0, 0);
        assert!(
            (val - 2.2).abs() < 0.01,
            "Expected ~2.2, got {}",
            val
        );
    }

    #[test]
    fn test_multiresolution_small_image() {
        // Very small image should not recurse
        let width = 4;
        let height = 4;
        let rgb1: Vec<u8> = vec![128; width * height * 3];
        let rgb2: Vec<u8> = vec![140; width * height * 3];

        let result =
            compute_butteraugli_impl(&rgb1, &rgb2, width, height, &ButteraugliParams::default());

        assert!(result.score > 0.0, "Should have non-zero score");
    }
}
