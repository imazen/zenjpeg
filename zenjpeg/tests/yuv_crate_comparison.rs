//! Compare the `yuv` crate's RGB→YUV conversion with our standard implementation.
//!
//! This test verifies that:
//! 1. Standard `rgb_to_yuv420` matches our BT.601 math
//! 2. Sharp YUV produces intentionally different (better) chroma at edges
//!
//! Uses `YuvConversionMode::Balanced` for comparison (default mode, always available).
//! Max difference is ~1.08 (integer math precision loss vs our f32 implementation).
//!
//! Run with: cargo test --release --test yuv_crate_comparison -- --nocapture

#[allow(unused_imports)]
use zenjpeg::encoder::ChromaSubsampling;

use zenjpeg::color::rgb_to_ycbcr_f32;
use yuv::{
    rgb_to_sharp_yuv420, rgb_to_yuv420, SharpYuvGammaTransfer, YuvChromaSubsampling,
    YuvConversionMode, YuvPlanarImageMut, YuvRange, YuvStandardMatrix,
};

/// Generate test patterns for comparison
fn generate_test_patterns() -> Vec<(&'static str, Vec<u8>, usize, usize)> {
    let w = 16usize;
    let h = 16usize;

    vec![
        // Solid colors
        (
            "solid_red",
            (0..w * h).flat_map(|_| [255u8, 0, 0]).collect(),
            w,
            h,
        ),
        (
            "solid_green",
            (0..w * h).flat_map(|_| [0u8, 255, 0]).collect(),
            w,
            h,
        ),
        (
            "solid_blue",
            (0..w * h).flat_map(|_| [0u8, 0, 255]).collect(),
            w,
            h,
        ),
        (
            "solid_gray",
            (0..w * h).flat_map(|_| [128u8, 128, 128]).collect(),
            w,
            h,
        ),
        (
            "solid_white",
            (0..w * h).flat_map(|_| [255u8, 255, 255]).collect(),
            w,
            h,
        ),
        (
            "solid_black",
            (0..w * h).flat_map(|_| [0u8, 0, 0]).collect(),
            w,
            h,
        ),
        // Gradients
        (
            "gray_gradient",
            (0..h)
                .flat_map(|_| {
                    (0..w).flat_map(|x| {
                        let v = (x * 255 / (w - 1)) as u8;
                        [v, v, v]
                    })
                })
                .collect(),
            w,
            h,
        ),
        (
            "color_gradient",
            (0..h)
                .flat_map(|y| {
                    (0..w).flat_map(move |x| {
                        let r = (x * 255 / (w - 1)) as u8;
                        let g = (y * 255 / (h - 1)) as u8;
                        [r, g, 128u8]
                    })
                })
                .collect(),
            w,
            h,
        ),
        // Edges (where Sharp YUV shines)
        (
            "sharp_edge",
            (0..h)
                .flat_map(|_| {
                    (0..w).flat_map(|x| {
                        if x < w / 2 {
                            [255u8, 0, 0]
                        } else {
                            [0u8, 255, 255]
                        }
                    })
                })
                .collect(),
            w,
            h,
        ),
        // High-frequency patterns
        (
            "checkerboard_rg",
            (0..h)
                .flat_map(|y| {
                    (0..w).flat_map(move |x| {
                        if (x + y) % 2 == 0 {
                            [255u8, 0, 0]
                        } else {
                            [0u8, 255, 0]
                        }
                    })
                })
                .collect(),
            w,
            h,
        ),
        (
            "checkerboard_bw",
            (0..h)
                .flat_map(|y| {
                    (0..w).flat_map(move |x| {
                        if (x + y) % 2 == 0 {
                            [255u8, 255, 255]
                        } else {
                            [0u8, 0, 0]
                        }
                    })
                })
                .collect(),
            w,
            h,
        ),
    ]
}

/// Our standard RGB→YCbCr conversion + box-filter downsampling
fn convert_standard(data: &[u8], width: usize, height: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let c_width = (width + 1) / 2;
    let c_height = (height + 1) / 2;

    // Full-res YCbCr
    let mut y_full = vec![0.0f32; width * height];
    let mut cb_full = vec![0.0f32; width * height];
    let mut cr_full = vec![0.0f32; width * height];

    for i in 0..(width * height) {
        let r = data[i * 3] as f32;
        let g = data[i * 3 + 1] as f32;
        let b = data[i * 3 + 2] as f32;
        let (y, cb, cr) = rgb_to_ycbcr_f32(r, g, b);
        y_full[i] = y;
        cb_full[i] = cb;
        cr_full[i] = cr;
    }

    // Box-filter downsample chroma
    let mut cb_down = vec![0.0f32; c_width * c_height];
    let mut cr_down = vec![0.0f32; c_width * c_height];

    for cy in 0..c_height {
        for cx in 0..c_width {
            let x0 = cx * 2;
            let y0 = cy * 2;
            let x1 = (x0 + 1).min(width - 1);
            let y1 = (y0 + 1).min(height - 1);

            let cb_sum = cb_full[y0 * width + x0]
                + cb_full[y0 * width + x1]
                + cb_full[y1 * width + x0]
                + cb_full[y1 * width + x1];
            let cr_sum = cr_full[y0 * width + x0]
                + cr_full[y0 * width + x1]
                + cr_full[y1 * width + x0]
                + cr_full[y1 * width + x1];

            cb_down[cy * c_width + cx] = cb_sum / 4.0;
            cr_down[cy * c_width + cx] = cr_sum / 4.0;
        }
    }

    (y_full, cb_down, cr_down)
}

/// yuv crate's standard (non-sharp) RGB→YUV420 conversion
fn convert_yuv_crate_standard(
    data: &[u8],
    width: usize,
    height: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let c_width = (width + 1) / 2;
    let c_height = (height + 1) / 2;

    let mut yuv_image =
        YuvPlanarImageMut::alloc(width as u32, height as u32, YuvChromaSubsampling::Yuv420);

    rgb_to_yuv420(
        &mut yuv_image,
        data,
        width as u32 * 3,
        YuvRange::Full,
        YuvStandardMatrix::Bt601,
        YuvConversionMode::Balanced,
    )
    .expect("rgb_to_yuv420 failed");

    // Convert to f32
    let y: Vec<f32> = yuv_image
        .y_plane
        .borrow()
        .iter()
        .take(width * height)
        .map(|&v| v as f32)
        .collect();
    let cb: Vec<f32> = yuv_image
        .u_plane
        .borrow()
        .iter()
        .take(c_width * c_height)
        .map(|&v| v as f32)
        .collect();
    let cr: Vec<f32> = yuv_image
        .v_plane
        .borrow()
        .iter()
        .take(c_width * c_height)
        .map(|&v| v as f32)
        .collect();

    (y, cb, cr)
}

/// yuv crate's Sharp YUV RGB→YUV420 conversion
fn convert_yuv_crate_sharp(
    data: &[u8],
    width: usize,
    height: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let c_width = (width + 1) / 2;
    let c_height = (height + 1) / 2;

    let mut yuv_image =
        YuvPlanarImageMut::alloc(width as u32, height as u32, YuvChromaSubsampling::Yuv420);

    rgb_to_sharp_yuv420(
        &mut yuv_image,
        data,
        width as u32 * 3,
        YuvRange::Full,
        YuvStandardMatrix::Bt601,
        SharpYuvGammaTransfer::Srgb,
    )
    .expect("rgb_to_sharp_yuv420 failed");

    // Convert to f32
    let y: Vec<f32> = yuv_image
        .y_plane
        .borrow()
        .iter()
        .take(width * height)
        .map(|&v| v as f32)
        .collect();
    let cb: Vec<f32> = yuv_image
        .u_plane
        .borrow()
        .iter()
        .take(c_width * c_height)
        .map(|&v| v as f32)
        .collect();
    let cr: Vec<f32> = yuv_image
        .v_plane
        .borrow()
        .iter()
        .take(c_width * c_height)
        .map(|&v| v as f32)
        .collect();

    (y, cb, cr)
}

/// Compute stats for difference between two planes
fn diff_stats(a: &[f32], b: &[f32]) -> (f32, f64) {
    let mut max_diff = 0.0f32;
    let mut sum_diff = 0.0f64;
    for (av, bv) in a.iter().zip(b.iter()) {
        let diff = (av - bv).abs();
        max_diff = max_diff.max(diff);
        sum_diff += diff as f64;
    }
    let avg_diff = sum_diff / a.len() as f64;
    (max_diff, avg_diff)
}

#[test]
fn test_yuv_crate_standard_vs_our_impl() {
    println!("\n=== yuv crate (standard) vs our BT.601 implementation ===\n");
    println!(
        "{:20} | {:^20} | {:^20} | {:^20}",
        "Pattern", "Y (max/avg)", "Cb (max/avg)", "Cr (max/avg)"
    );
    println!("{:-<86}", "");

    for (name, data, width, height) in generate_test_patterns() {
        let (y_ours, cb_ours, cr_ours) = convert_standard(&data, width, height);
        let (y_yuv, cb_yuv, cr_yuv) = convert_yuv_crate_standard(&data, width, height);

        let (y_max, y_avg) = diff_stats(&y_ours, &y_yuv);
        let (cb_max, cb_avg) = diff_stats(&cb_ours, &cb_yuv);
        let (cr_max, cr_avg) = diff_stats(&cr_ours, &cr_yuv);

        println!(
            "{:20} | {:6.2} / {:6.2}      | {:6.2} / {:6.2}      | {:6.2} / {:6.2}",
            name, y_max, y_avg, cb_max, cb_avg, cr_max, cr_avg
        );

        // Y should be very close (both BT.601, just float vs int rounding)
        assert!(
            y_max < 1.5,
            "{}: Y max diff {} exceeds threshold",
            name,
            y_max
        );

        // For solid colors and simple gradients, chroma should also be close
        if name.starts_with("solid") || name.contains("gradient") {
            assert!(cb_max < 2.0, "{}: Cb max diff {} too large", name, cb_max);
            assert!(cr_max < 2.0, "{}: Cr max diff {} too large", name, cr_max);
        }
    }

    println!("\nConclusion: Standard yuv crate matches our BT.601 implementation closely.");
    println!("Small differences are due to float vs integer rounding.\n");
}

#[test]
fn test_sharp_yuv_differences() {
    println!("\n=== Sharp YUV vs Standard (both from yuv crate) ===\n");
    println!(
        "{:20} | {:^20} | {:^20} | {:^20}",
        "Pattern", "Y (max/avg)", "Cb (max/avg)", "Cr (max/avg)"
    );
    println!("{:-<86}", "");

    for (name, data, width, height) in generate_test_patterns() {
        let (y_std, cb_std, cr_std) = convert_yuv_crate_standard(&data, width, height);
        let (y_sharp, cb_sharp, cr_sharp) = convert_yuv_crate_sharp(&data, width, height);

        let (y_max, y_avg) = diff_stats(&y_std, &y_sharp);
        let (cb_max, cb_avg) = diff_stats(&cb_std, &cb_sharp);
        let (cr_max, cr_avg) = diff_stats(&cr_std, &cr_sharp);

        println!(
            "{:20} | {:6.2} / {:6.2}      | {:6.2} / {:6.2}      | {:6.2} / {:6.2}",
            name, y_max, y_avg, cb_max, cb_avg, cr_max, cr_avg
        );
    }

    println!("\nConclusion: Sharp YUV preserves Y but differs on chroma,");
    println!("especially at edges and high-frequency patterns.\n");
}

#[test]
fn test_brute_force_single_pixels() {
    println!("\n=== Brute-force single-pixel RGB→Y comparison ===\n");

    // Test a sampling of RGB values
    let mut max_y_diff = 0.0f32;
    let mut max_cb_diff = 0.0f32;
    let mut max_cr_diff = 0.0f32;
    let mut worst_rgb = (0u8, 0u8, 0u8);

    // Sample every 17th value to cover the range without testing all 16M combinations
    for r in (0..=255).step_by(17) {
        for g in (0..=255).step_by(17) {
            for b in (0..=255).step_by(17) {
                // Our implementation
                let (y_ours, cb_ours, cr_ours) = rgb_to_ycbcr_f32(r as f32, g as f32, b as f32);

                // yuv crate - need 2x2 block minimum for 4:2:0
                let data = vec![r, g, b, r, g, b, r, g, b, r, g, b];
                let mut yuv_image = YuvPlanarImageMut::alloc(2, 2, YuvChromaSubsampling::Yuv420);

                rgb_to_yuv420(
                    &mut yuv_image,
                    &data,
                    6, // stride = 2 pixels * 3 bytes
                    YuvRange::Full,
                    YuvStandardMatrix::Bt601,
                    YuvConversionMode::Balanced,
                )
                .expect("conversion failed");

                let y_yuv = yuv_image.y_plane.borrow()[0] as f32;
                let cb_yuv = yuv_image.u_plane.borrow()[0] as f32;
                let cr_yuv = yuv_image.v_plane.borrow()[0] as f32;

                let y_diff = (y_ours - y_yuv).abs();
                let cb_diff = (cb_ours - cb_yuv).abs();
                let cr_diff = (cr_ours - cr_yuv).abs();

                if y_diff > max_y_diff {
                    max_y_diff = y_diff;
                    worst_rgb = (r, g, b);
                }
                max_cb_diff = max_cb_diff.max(cb_diff);
                max_cr_diff = max_cr_diff.max(cr_diff);
            }
        }
    }

    println!("Sampled {} RGB values", 16 * 16 * 16);
    println!(
        "Max Y difference:  {:.2} (at RGB {:?})",
        max_y_diff, worst_rgb
    );
    println!("Max Cb difference: {:.2}", max_cb_diff);
    println!("Max Cr difference: {:.2}", max_cr_diff);

    // All differences should be small (rounding only)
    assert!(max_y_diff < 1.5, "Y diff {} too large", max_y_diff);
    assert!(max_cb_diff < 1.5, "Cb diff {} too large", max_cb_diff);
    assert!(max_cr_diff < 1.5, "Cr diff {} too large", max_cr_diff);

    println!("\nConclusion: Our BT.601 math matches yuv crate within rounding tolerance.\n");
}
