//! Step-by-step comparison of Rust vs C++ butteraugli pipeline.
//!
//! This test traces through each stage of the butteraugli algorithm
//! comparing intermediate values to locate where divergence occurs.

use butteraugli_oxide::blur::gaussian_blur;
use butteraugli_oxide::image::ImageF;
use butteraugli_oxide::opsin::srgb_to_xyb_butteraugli;
use butteraugli_oxide::psycho::separate_frequencies;
use jpegli_sys::{
    butteraugli_blur, butteraugli_compare_full, butteraugli_opsin_dynamics,
    butteraugli_separate_frequencies, BUTTERAUGLI_OK,
};

// ============================================================================
// Test image generators
// ============================================================================

/// 64x64 checkerboard - known to have ~15% divergence
fn generate_checkerboard_64x64() -> Vec<u8> {
    (0..64)
        .flat_map(|y| {
            (0..64).flat_map(move |x| {
                let v = if (x + y) % 2 == 0 { 50u8 } else { 200u8 };
                [v, v, v]
            })
        })
        .collect()
}

/// 64x64 uniform gray
fn generate_uniform_64x64(val: u8) -> Vec<u8> {
    vec![val; 64 * 64 * 3]
}

/// 64x64 horizontal gradient
fn generate_h_gradient_64x64() -> Vec<u8> {
    (0..64)
        .flat_map(|_y| {
            (0..64).flat_map(|x| {
                let v = (x * 255 / 63) as u8;
                [v, v, v]
            })
        })
        .collect()
}

// ============================================================================
// Helper functions
// ============================================================================

fn srgb_to_linear_rust(srgb: &[u8]) -> Vec<f32> {
    srgb.iter()
        .map(|&v| {
            let x = v as f32 / 255.0;
            if x <= 0.04045 {
                x / 12.92
            } else {
                ((x + 0.055) / 1.055).powf(2.4)
            }
        })
        .collect()
}

fn stats(data: &[f32]) -> (f32, f32, f32) {
    if data.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = data.iter().sum();
    let mean = sum / data.len() as f32;
    (min, max, mean)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn mean_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() {
        return 0.0;
    }
    let sum: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum();
    sum / a.len() as f32
}

// ============================================================================
// Stage 1: Blur comparison
// ============================================================================

#[test]
fn test_stage1_blur() {
    println!("\n=== Stage 1: Blur Comparison ===\n");

    let width = 64usize;
    let height = 64usize;

    // Create test pattern - a gradient
    let input: Vec<f32> = (0..height)
        .flat_map(|y| (0..width).map(move |x| (x + y) as f32 / (width + height) as f32))
        .collect();

    for sigma in [1.564f32, 3.225, 7.156] {
        // Rust blur
        let input_img = ImageF::from_vec(input.clone(), width, height);
        let rust_blurred = gaussian_blur(&input_img, sigma);
        let mut rust_flat = Vec::with_capacity(width * height);
        for y in 0..height {
            for x in 0..width {
                rust_flat.push(rust_blurred.get(x, y));
            }
        }

        // C++ blur
        let mut cpp_blurred = vec![0.0f32; width * height];
        let result = unsafe {
            butteraugli_blur(
                input.as_ptr(),
                width,
                height,
                sigma,
                cpp_blurred.as_mut_ptr(),
            )
        };

        if result != BUTTERAUGLI_OK {
            println!(
                "  sigma={:.3}: C++ butteraugli_blur failed with code {}",
                sigma, result
            );
            continue;
        }

        let max_diff = max_abs_diff(&rust_flat, &cpp_blurred);
        let mean_diff = mean_abs_diff(&rust_flat, &cpp_blurred);

        let (r_min, r_max, r_mean) = stats(&rust_flat);
        let (c_min, c_max, c_mean) = stats(&cpp_blurred);

        println!("  sigma={:.3}:", sigma);
        println!(
            "    Rust: min={:.4} max={:.4} mean={:.4}",
            r_min, r_max, r_mean
        );
        println!(
            "    C++:  min={:.4} max={:.4} mean={:.4}",
            c_min, c_max, c_mean
        );
        println!(
            "    max_diff={:.6} mean_diff={:.6} ({:.2}%)",
            max_diff,
            mean_diff,
            mean_diff / c_mean.abs().max(0.001) * 100.0
        );
    }
}

// ============================================================================
// Stage 2: XYB conversion comparison (via OpsinDynamicsImage)
// ============================================================================

#[test]
fn test_stage2_xyb_conversion() {
    println!("\n=== Stage 2: XYB Conversion (OpsinDynamicsImage) ===\n");

    let width = 64usize;
    let height = 64usize;
    let intensity_target = 80.0f32;

    for (name, srgb) in [
        ("Uniform Gray", generate_uniform_64x64(128)),
        ("H Gradient", generate_h_gradient_64x64()),
        ("Checkerboard", generate_checkerboard_64x64()),
    ] {
        let linear = srgb_to_linear_rust(&srgb);

        // Rust XYB
        let rust_xyb = srgb_to_xyb_butteraugli(&srgb, width, height, intensity_target);

        // C++ XYB (via opsin_dynamics wrapper)
        let mut cpp_xyb = vec![0.0f32; width * height * 3];
        let result = unsafe {
            butteraugli_opsin_dynamics(
                linear.as_ptr(),
                width,
                height,
                intensity_target,
                cpp_xyb.as_mut_ptr(),
            )
        };

        if result != BUTTERAUGLI_OK {
            println!("  {}: C++ opsin_dynamics failed", name);
            continue;
        }

        // Compare X, Y, B channels
        for (ch, ch_name) in [(0, "X"), (1, "Y"), (2, "B")] {
            let mut rust_flat = Vec::with_capacity(width * height);
            let mut cpp_flat = Vec::with_capacity(width * height);
            for y in 0..height {
                for x in 0..width {
                    rust_flat.push(rust_xyb.plane(ch).get(x, y));
                    cpp_flat.push(cpp_xyb[(y * width + x) * 3 + ch]);
                }
            }

            let max_diff = max_abs_diff(&rust_flat, &cpp_flat);
            let mean_diff = mean_abs_diff(&rust_flat, &cpp_flat);
            let (_, _, r_mean) = stats(&rust_flat);
            let (_, _, c_mean) = stats(&cpp_flat);

            println!(
                "  {} - {} channel: max_diff={:.6} mean_diff={:.6} (Rust mean={:.4} C++ mean={:.4})",
                name, ch_name, max_diff, mean_diff, r_mean, c_mean
            );
        }
    }
}

// ============================================================================
// Stage 3: Frequency separation comparison
// ============================================================================

#[test]
fn test_stage3_frequency_separation() {
    println!("\n=== Stage 3: Frequency Separation ===\n");

    let width = 64usize;
    let height = 64usize;
    let intensity_target = 80.0f32;

    for (name, srgb) in [
        ("Uniform Gray", generate_uniform_64x64(128)),
        ("H Gradient", generate_h_gradient_64x64()),
        ("Checkerboard", generate_checkerboard_64x64()),
    ] {
        println!("  {}:", name);

        // Convert sRGB to linear RGB for C++ (C++ does its own XYB conversion)
        let linear = srgb_to_linear_rust(&srgb);

        // Get Rust XYB and frequency separation
        let rust_xyb = srgb_to_xyb_butteraugli(&srgb, width, height, intensity_target);
        let rust_psycho = separate_frequencies(&rust_xyb);

        // C++ frequency separation
        let mut cpp_lf_x = vec![0.0f32; width * height];
        let mut cpp_lf_y = vec![0.0f32; width * height];
        let mut cpp_lf_b = vec![0.0f32; width * height];
        let mut cpp_mf_x = vec![0.0f32; width * height];
        let mut cpp_mf_y = vec![0.0f32; width * height];
        let mut cpp_mf_b = vec![0.0f32; width * height];
        let mut cpp_hf_x = vec![0.0f32; width * height];
        let mut cpp_hf_y = vec![0.0f32; width * height];
        let mut cpp_uhf_x = vec![0.0f32; width * height];
        let mut cpp_uhf_y = vec![0.0f32; width * height];

        // C++ takes linear RGB (interleaved) and does its own XYB conversion
        let result = unsafe {
            butteraugli_separate_frequencies(
                linear.as_ptr(),
                width,
                height,
                intensity_target,
                cpp_lf_x.as_mut_ptr(),
                cpp_lf_y.as_mut_ptr(),
                cpp_lf_b.as_mut_ptr(),
                cpp_mf_x.as_mut_ptr(),
                cpp_mf_y.as_mut_ptr(),
                cpp_mf_b.as_mut_ptr(),
                cpp_hf_x.as_mut_ptr(),
                cpp_hf_y.as_mut_ptr(),
                cpp_uhf_x.as_mut_ptr(),
                cpp_uhf_y.as_mut_ptr(),
            )
        };

        if result != BUTTERAUGLI_OK {
            println!("    C++ separate_frequencies failed with code {}", result);
            continue;
        }

        // Compare each frequency band
        let band_names = [
            "LF_X", "LF_Y", "LF_B", "MF_X", "MF_Y", "MF_B", "HF_X", "HF_Y", "UHF_X", "UHF_Y",
        ];
        let cpp_bands: [&Vec<f32>; 10] = [
            &cpp_lf_x, &cpp_lf_y, &cpp_lf_b, &cpp_mf_x, &cpp_mf_y, &cpp_mf_b, &cpp_hf_x, &cpp_hf_y,
            &cpp_uhf_x, &cpp_uhf_y,
        ];

        for (i, band_name) in band_names.iter().enumerate() {
            // Get rust plane based on index
            let mut rust_flat = Vec::with_capacity(width * height);
            for y in 0..height {
                for x in 0..width {
                    let val = match i {
                        0 => rust_psycho.lf.plane(0).get(x, y),
                        1 => rust_psycho.lf.plane(1).get(x, y),
                        2 => rust_psycho.lf.plane(2).get(x, y),
                        3 => rust_psycho.mf.plane(0).get(x, y),
                        4 => rust_psycho.mf.plane(1).get(x, y),
                        5 => rust_psycho.mf.plane(2).get(x, y),
                        6 => rust_psycho.hf[0].get(x, y),
                        7 => rust_psycho.hf[1].get(x, y),
                        8 => rust_psycho.uhf[0].get(x, y),
                        9 => rust_psycho.uhf[1].get(x, y),
                        _ => 0.0,
                    };
                    rust_flat.push(val);
                }
            }
            let cpp_data = cpp_bands[i];

            let max_diff = max_abs_diff(&rust_flat, cpp_data);
            let mean_diff = mean_abs_diff(&rust_flat, cpp_data);
            let (_, _, r_mean) = stats(&rust_flat);
            let (_, _, c_mean) = stats(cpp_data);
            let rel_diff = if c_mean.abs() > 0.001 {
                mean_diff / c_mean.abs() * 100.0
            } else {
                0.0
            };

            // Only print if there's significant difference
            if max_diff > 0.001 || rel_diff > 1.0 {
                println!(
                    "    {}: max={:.6} mean={:.6} ({:.1}%) R_mean={:.4} C_mean={:.4}",
                    band_name, max_diff, mean_diff, rel_diff, r_mean, c_mean
                );
            }
        }
    }
}

// ============================================================================
// Full pipeline comparison
// ============================================================================

#[test]
fn test_full_pipeline_divergence_location() {
    println!("\n=== Full Pipeline Divergence Location ===\n");
    println!("Testing checkerboard pattern (known 15% divergence case)\n");

    let width = 64usize;
    let height = 64usize;
    let intensity_target = 80.0f32;

    let img1 = generate_checkerboard_64x64();
    let img2: Vec<u8> = img1.iter().map(|&v| v.saturating_add(20)).collect();

    let linear1 = srgb_to_linear_rust(&img1);
    let linear2 = srgb_to_linear_rust(&img2);

    // Get final scores from C++
    let mut cpp_score = 0.0f64;
    let mut cpp_diffmap = vec![0.0f32; width * height];
    let result = unsafe {
        butteraugli_compare_full(
            linear1.as_ptr(),
            linear2.as_ptr(),
            width,
            height,
            1.0,
            1.0,
            intensity_target,
            &mut cpp_score,
            cpp_diffmap.as_mut_ptr(),
        )
    };
    assert_eq!(result, BUTTERAUGLI_OK);

    // Rust computation
    let rust_xyb1 = srgb_to_xyb_butteraugli(&img1, width, height, intensity_target);
    let rust_xyb2 = srgb_to_xyb_butteraugli(&img2, width, height, intensity_target);
    let rust_psycho1 = separate_frequencies(&rust_xyb1);
    let rust_psycho2 = separate_frequencies(&rust_xyb2);

    println!("Stage-by-stage analysis:");

    // Check frequency bands difference
    println!("\n  Frequency band differences (img1 vs img2):");
    let bands = [
        "LF_X", "LF_Y", "LF_B", "MF_X", "MF_Y", "MF_B", "HF_X", "HF_Y", "UHF_X", "UHF_Y",
    ];

    for (i, band_name) in bands.iter().enumerate() {
        let mut diff_vals = Vec::with_capacity(width * height);
        for y in 0..height {
            for x in 0..width {
                let v1 = match i {
                    0 => rust_psycho1.lf.plane(0).get(x, y),
                    1 => rust_psycho1.lf.plane(1).get(x, y),
                    2 => rust_psycho1.lf.plane(2).get(x, y),
                    3 => rust_psycho1.mf.plane(0).get(x, y),
                    4 => rust_psycho1.mf.plane(1).get(x, y),
                    5 => rust_psycho1.mf.plane(2).get(x, y),
                    6 => rust_psycho1.hf[0].get(x, y),
                    7 => rust_psycho1.hf[1].get(x, y),
                    8 => rust_psycho1.uhf[0].get(x, y),
                    9 => rust_psycho1.uhf[1].get(x, y),
                    _ => 0.0,
                };
                let v2 = match i {
                    0 => rust_psycho2.lf.plane(0).get(x, y),
                    1 => rust_psycho2.lf.plane(1).get(x, y),
                    2 => rust_psycho2.lf.plane(2).get(x, y),
                    3 => rust_psycho2.mf.plane(0).get(x, y),
                    4 => rust_psycho2.mf.plane(1).get(x, y),
                    5 => rust_psycho2.mf.plane(2).get(x, y),
                    6 => rust_psycho2.hf[0].get(x, y),
                    7 => rust_psycho2.hf[1].get(x, y),
                    8 => rust_psycho2.uhf[0].get(x, y),
                    9 => rust_psycho2.uhf[1].get(x, y),
                    _ => 0.0,
                };
                diff_vals.push((v1 - v2).abs());
            }
        }
        let (min, max, mean) = stats(&diff_vals);
        if max > 0.001 {
            println!(
                "    {}: min={:.6} max={:.6} mean={:.6}",
                band_name, min, max, mean
            );
        }
    }

    // Final score comparison
    println!("\n  Final scores:");
    println!("    C++ score: {:.6}", cpp_score);

    // Check diffmap stats
    let (c_min, c_max, c_mean) = stats(&cpp_diffmap);
    println!(
        "\n  C++ diffmap: min={:.6} max={:.6} mean={:.6}",
        c_min, c_max, c_mean
    );
}
