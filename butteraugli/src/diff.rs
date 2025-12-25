//! Main butteraugli difference computation.
//!
//! This module ties together all the components to compute the
//! perceptual difference between two images.

use crate::consts::{
    NORM1_HF, NORM1_HF_X, NORM1_MF, NORM1_MF_X, NORM1_UHF, NORM1_UHF_X, WMUL, W_HF_MALTA,
    W_HF_MALTA_X, W_MF_MALTA, W_MF_MALTA_X, W_UHF_MALTA, W_UHF_MALTA_X,
};
use crate::image::{Image3F, ImageF};
use crate::malta::malta_diff_map;
use crate::mask::{
    combine_channels_for_masking, compute_mask as compute_mask_from_images, mask_dc_y, mask_y,
};
use crate::opsin::{linear_rgb_to_xyb_butteraugli, srgb_to_xyb_butteraugli};
use crate::psycho::{separate_frequencies, PsychoImage};
use crate::{ButteraugliParams, ButteraugliResult};

/// Minimum image dimension for multi-resolution processing.
/// Images smaller than this are handled without recursion.
const MIN_SIZE_FOR_MULTIRESOLUTION: usize = 8;

/// Converts sRGB u8 buffer to XYB Image3F using butteraugli's OpsinDynamicsImage.
///
/// This uses the correct butteraugli color conversion, which is DIFFERENT
/// from jpegli's XYB color space. Key differences:
/// 1. Different OpsinAbsorbance matrix coefficients
/// 2. Uses Gamma function (FastLog2f based), not cube root
/// 3. Includes dynamic sensitivity based on blurred image
fn srgb_to_xyb_image(rgb: &[u8], width: usize, height: usize, intensity_target: f32) -> Image3F {
    srgb_to_xyb_butteraugli(rgb, width, height, intensity_target)
}

/// Converts linear RGB f32 buffer to XYB Image3F using butteraugli's OpsinDynamicsImage.
fn linear_rgb_to_xyb_image(
    rgb: &[f32],
    width: usize,
    height: usize,
    intensity_target: f32,
) -> Image3F {
    linear_rgb_to_xyb_butteraugli(rgb, width, height, intensity_target)
}

/// Subsamples an Image3F by 2x using box filter averaging.
///
/// Each 2x2 block of pixels is averaged into a single pixel.
/// Edge cases for odd dimensions are handled by scaling the edge values.
#[allow(dead_code)]
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

// WMUL weights are imported from crate::consts
// These match C++ butteraugli.cc:
// [HF_X, HF_Y, HF_B, MF_X, MF_Y, MF_B, LF_X, LF_Y, LF_B]
// Note: WMUL is f64 array, but we need f32 for pixel operations

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
    _xmul: f32,
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
        WMUL[0] as f32 * hf_asymmetry,
        WMUL[0] as f32 / hf_asymmetry,
        block_diff_ac.plane_mut(0),
    );
    l2_diff_asymmetric(
        &ps0.hf[1],
        &ps1.hf[1],
        WMUL[1] as f32 * hf_asymmetry,
        WMUL[1] as f32 / hf_asymmetry,
        block_diff_ac.plane_mut(1),
    );

    // Add L2Diff for MF channels (all three)
    l2_diff(
        ps0.mf.plane(0),
        ps1.mf.plane(0),
        WMUL[3] as f32,
        block_diff_ac.plane_mut(0),
    );
    l2_diff(
        ps0.mf.plane(1),
        ps1.mf.plane(1),
        WMUL[4] as f32,
        block_diff_ac.plane_mut(1),
    );
    l2_diff(
        ps0.mf.plane(2),
        ps1.mf.plane(2),
        WMUL[5] as f32,
        block_diff_ac.plane_mut(2),
    );

    block_diff_ac
}

/// Computes the mask from two PsychoImages.
///
/// Matches C++ MaskPsychoImage (butteraugli.cc lines 1250-1264).
/// Returns the computed mask and optionally accumulates AC differences.
fn mask_psycho_image(ps0: &PsychoImage, ps1: &PsychoImage, diff_ac: Option<&mut ImageF>) -> ImageF {
    let width = ps0.width();
    let height = ps0.height();

    // Combine HF and UHF channels for masking
    let mut mask0 = ImageF::new(width, height);
    let mut mask1 = ImageF::new(width, height);
    combine_channels_for_masking(&ps0.hf, &ps0.uhf, &mut mask0);
    combine_channels_for_masking(&ps1.hf, &ps1.uhf, &mut mask1);

    // Compute mask using DiffPrecompute, blur, and FuzzyErosion
    compute_mask_from_images(&mask0, &mask1, diff_ac)
}

/// Combines channels to produce final diffmap.
///
/// Matches C++ CombineChannelsToDiffmap (butteraugli.cc lines 1289-1315).
/// Applies MaskY for AC differences and MaskDcY for DC differences.
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
            let val = mask.get(x, y) as f64;

            // Compute masking factors from the mask value
            // MaskY is used for AC, MaskDcY is used for DC
            let maskval = mask_y(val) as f32;
            let dc_maskval = mask_dc_y(val) as f32;

            // Get difference values for each channel
            let diff_dc = [
                block_diff_dc.plane(0).get(x, y),
                block_diff_dc.plane(1).get(x, y),
                block_diff_dc.plane(2).get(x, y),
            ];
            let diff_ac = [
                block_diff_ac.plane(0).get(x, y),
                block_diff_ac.plane(1).get(x, y),
                block_diff_ac.plane(2).get(x, y),
            ];

            // Apply xmul to X channel (index 0)
            let diff_ac_scaled = [diff_ac[0] * xmul, diff_ac[1], diff_ac[2]];
            let diff_dc_scaled = [diff_dc[0] * xmul, diff_dc[1], diff_dc[2]];

            // MaskColor: sum of all channels multiplied by mask
            // C++: color[0] * mask + color[1] * mask + color[2] * mask
            let dc_masked = diff_dc_scaled[0] * dc_maskval
                + diff_dc_scaled[1] * dc_maskval
                + diff_dc_scaled[2] * dc_maskval;
            let ac_masked = diff_ac_scaled[0] * maskval
                + diff_ac_scaled[1] * maskval
                + diff_ac_scaled[2] * maskval;

            // Final diffmap value is sqrt of sum
            diffmap.set(x, y, (dc_masked + ac_masked).sqrt());
        }
    }

    diffmap
}

/// Computes the global score from a difference map.
///
/// C++ ButteraugliScoreFromDiffmap (butteraugli.cc lines 1952-1962)
/// returns the maximum value in the diffmap. The diffmap already has
/// the global scaling applied via MaskY/MaskDcY.
fn compute_score_from_diffmap(diffmap: &ImageF) -> f64 {
    let width = diffmap.width();
    let height = diffmap.height();
    let num_pixels = width * height;

    if num_pixels == 0 {
        return 0.0;
    }

    // Find maximum difference value (C++ butteraugli approach)
    let mut max_val = 0.0f32;

    for y in 0..height {
        for x in 0..width {
            let v = diffmap.get(x, y);
            if v > max_val {
                max_val = v;
            }
        }
    }

    // No additional scaling needed - MaskY/MaskDcY already include GLOBAL_SCALE
    max_val as f64
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
    let xyb1 = srgb_to_xyb_image(rgb1, width, height, params.intensity_target());
    let xyb2 = srgb_to_xyb_image(rgb2, width, height, params.intensity_target());

    // Perform frequency decomposition
    let ps1 = separate_frequencies(&xyb1);
    let ps2 = separate_frequencies(&xyb2);

    // Compute AC differences using Malta filter
    let mut block_diff_ac = compute_psycho_diff_malta(&ps1, &ps2, params.hf_asymmetry(), params.xmul());

    // Compute mask from both PsychoImages (also accumulates some AC differences)
    // This matches C++ MaskPsychoImage which calls CombineChannelsForMasking + Mask
    let mask = mask_psycho_image(&ps1, &ps2, Some(block_diff_ac.plane_mut(1)));

    // Compute DC (LF) differences
    let mut block_diff_dc = Image3F::new(width, height);
    for c in 0..3 {
        for y in 0..height {
            for x in 0..width {
                let d = ps1.lf.plane(c).get(x, y) - ps2.lf.plane(c).get(x, y);
                block_diff_dc
                    .plane_mut(c)
                    .set(x, y, d * d * WMUL[6 + c] as f32);
            }
        }
    }

    // Combine channels to final diffmap using MaskY/MaskDcY
    combine_channels_to_diffmap(&mask, &block_diff_dc, &block_diff_ac, params.xmul())
}

/// Computes butteraugli diffmap with optional single-level multiresolution.
///
/// Matches C++ ButteraugliInterfaceInPlace: only ONE level of subsampling,
/// not recursive. For images >= 15x15, compute at half resolution and add.
fn compute_diffmap_multiresolution(
    rgb1: &[u8],
    rgb2: &[u8],
    width: usize,
    height: usize,
    params: &ButteraugliParams,
) -> ImageF {
    // C++ uses 15 as threshold for multiresolution
    const MIN_SIZE_FOR_SUBSAMPLE: usize = 15;

    // First compute subdiffmap at half resolution (if image is large enough)
    let mut sub_diffmap = None;
    if width >= MIN_SIZE_FOR_SUBSAMPLE && height >= MIN_SIZE_FOR_SUBSAMPLE {
        let (sub_rgb1, sw, sh) = subsample_rgb_2x(rgb1, width, height);
        let (sub_rgb2, _, _) = subsample_rgb_2x(rgb2, width, height);

        // Single level only, not recursive (matches C++)
        sub_diffmap = Some(compute_diffmap_single_resolution(
            &sub_rgb1, &sub_rgb2, sw, sh, params,
        ));
    }

    // Compute diffmap at full resolution
    let mut diffmap = compute_diffmap_single_resolution(rgb1, rgb2, width, height, params);

    // Add supersampled subdiffmap if we computed one
    if let Some(sub) = sub_diffmap {
        add_supersampled_2x(&sub, 0.5, &mut diffmap);
    }

    diffmap
}

/// Subsamples linear RGB f32 buffer by 2x for multi-resolution processing.
fn subsample_linear_rgb_2x(rgb: &[f32], width: usize, height: usize) -> (Vec<f32>, usize, usize) {
    let out_width = (width + 1) / 2;
    let out_height = (height + 1) / 2;
    let mut output = vec![0.0f32; out_width * out_height * 3];

    // Simple averaging of 2x2 blocks
    for oy in 0..out_height {
        for ox in 0..out_width {
            let mut r_sum = 0.0f32;
            let mut g_sum = 0.0f32;
            let mut b_sum = 0.0f32;
            let mut count = 0.0f32;

            for dy in 0..2 {
                for dx in 0..2 {
                    let ix = ox * 2 + dx;
                    let iy = oy * 2 + dy;
                    if ix < width && iy < height {
                        let idx = (iy * width + ix) * 3;
                        r_sum += rgb[idx];
                        g_sum += rgb[idx + 1];
                        b_sum += rgb[idx + 2];
                        count += 1.0;
                    }
                }
            }

            if count > 0.0 {
                let out_idx = (oy * out_width + ox) * 3;
                output[out_idx] = r_sum / count;
                output[out_idx + 1] = g_sum / count;
                output[out_idx + 2] = b_sum / count;
            }
        }
    }

    (output, out_width, out_height)
}

/// Computes the diffmap for a single resolution level (linear RGB input).
fn compute_diffmap_single_resolution_linear(
    rgb1: &[f32],
    rgb2: &[f32],
    width: usize,
    height: usize,
    params: &ButteraugliParams,
) -> ImageF {
    // Convert to XYB using butteraugli's OpsinDynamicsImage
    let xyb1 = linear_rgb_to_xyb_image(rgb1, width, height, params.intensity_target());
    let xyb2 = linear_rgb_to_xyb_image(rgb2, width, height, params.intensity_target());

    // Perform frequency decomposition
    let ps1 = separate_frequencies(&xyb1);
    let ps2 = separate_frequencies(&xyb2);

    // Compute AC differences using Malta filter
    let mut block_diff_ac =
        compute_psycho_diff_malta(&ps1, &ps2, params.hf_asymmetry(), params.xmul());

    // Compute mask from both PsychoImages (also accumulates some AC differences)
    let mask = mask_psycho_image(&ps1, &ps2, Some(block_diff_ac.plane_mut(1)));

    // Compute DC (LF) differences
    let mut block_diff_dc = Image3F::new(width, height);
    for c in 0..3 {
        for y in 0..height {
            for x in 0..width {
                let d = ps1.lf.plane(c).get(x, y) - ps2.lf.plane(c).get(x, y);
                block_diff_dc
                    .plane_mut(c)
                    .set(x, y, d * d * WMUL[6 + c] as f32);
            }
        }
    }

    // Combine channels to final diffmap using MaskY/MaskDcY
    combine_channels_to_diffmap(&mask, &block_diff_dc, &block_diff_ac, params.xmul())
}

/// Computes butteraugli diffmap with multiresolution (linear RGB input).
fn compute_diffmap_multiresolution_linear(
    rgb1: &[f32],
    rgb2: &[f32],
    width: usize,
    height: usize,
    params: &ButteraugliParams,
) -> ImageF {
    const MIN_SIZE_FOR_SUBSAMPLE: usize = 15;

    // First compute subdiffmap at half resolution (if image is large enough)
    let mut sub_diffmap = None;
    if width >= MIN_SIZE_FOR_SUBSAMPLE && height >= MIN_SIZE_FOR_SUBSAMPLE {
        let (sub_rgb1, sw, sh) = subsample_linear_rgb_2x(rgb1, width, height);
        let (sub_rgb2, _, _) = subsample_linear_rgb_2x(rgb2, width, height);

        sub_diffmap = Some(compute_diffmap_single_resolution_linear(
            &sub_rgb1, &sub_rgb2, sw, sh, params,
        ));
    }

    // Compute diffmap at full resolution
    let mut diffmap = compute_diffmap_single_resolution_linear(rgb1, rgb2, width, height, params);

    // Add supersampled subdiffmap if we computed one
    if let Some(sub) = sub_diffmap {
        add_supersampled_2x(&sub, 0.5, &mut diffmap);
    }

    diffmap
}

/// Main implementation of butteraugli comparison (sRGB u8 input).
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

/// Main implementation of butteraugli comparison (linear RGB f32 input).
///
/// This matches the C++ butteraugli API which expects linear RGB float input.
pub fn compute_butteraugli_linear_impl(
    rgb1: &[f32],
    rgb2: &[f32],
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
        compute_diffmap_single_resolution_linear(rgb1, rgb2, width, height, params)
    } else {
        compute_diffmap_multiresolution_linear(rgb1, rgb2, width, height, params)
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
        assert!((val - 2.2).abs() < 0.01, "Expected ~2.2, got {}", val);
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
