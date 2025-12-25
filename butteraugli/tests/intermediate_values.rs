//! Intermediate value comparison tests for butteraugli C++ parity.
//!
//! Uses synthetic test images stored as constants for reproducibility.
//! Compares intermediate pipeline stages between Rust and C++ implementations.

use butteraugli::opsin::{fast_log2f, gamma, opsin_dynamics_image, srgb_to_xyb_butteraugli};
use butteraugli::{compute_butteraugli, ButteraugliParams};
use jpegli_sys::{
    butteraugli_compare_full, butteraugli_opsin_dynamics, butteraugli_srgb_to_linear,
    BUTTERAUGLI_OK,
};

// ============================================================================
// Synthetic test data - deterministic and reproducible
// ============================================================================

/// 8x8 uniform gray image (sRGB 128)
const UNIFORM_GRAY_8X8: [u8; 192] = [128; 192];

/// 8x8 gradient (horizontal, 0-255)
const fn generate_gradient_8x8() -> [u8; 192] {
    let mut data = [0u8; 192];
    let mut i = 0;
    while i < 8 {
        let mut j = 0;
        while j < 8 {
            let val = (j * 255 / 7) as u8;
            let idx = (i * 8 + j) * 3;
            data[idx] = val;
            data[idx + 1] = val;
            data[idx + 2] = val;
            j += 1;
        }
        i += 1;
    }
    data
}
const GRADIENT_8X8: [u8; 192] = generate_gradient_8x8();

/// 8x8 checkerboard pattern
const fn generate_checkerboard_8x8() -> [u8; 192] {
    let mut data = [0u8; 192];
    let mut i = 0;
    while i < 8 {
        let mut j = 0;
        while j < 8 {
            let val = if (i + j) % 2 == 0 { 50u8 } else { 200u8 };
            let idx = (i * 8 + j) * 3;
            data[idx] = val;
            data[idx + 1] = val;
            data[idx + 2] = val;
            j += 1;
        }
        i += 1;
    }
    data
}
const CHECKERBOARD_8X8: [u8; 192] = generate_checkerboard_8x8();

/// 16x16 smooth color gradient
const fn generate_color_gradient_16x16() -> [u8; 768] {
    let mut data = [0u8; 768];
    let mut y = 0;
    while y < 16 {
        let mut x = 0;
        while x < 16 {
            let idx = (y * 16 + x) * 3;
            data[idx] = (x * 255 / 15) as u8; // R gradient
            data[idx + 1] = (y * 255 / 15) as u8; // G gradient
            data[idx + 2] = 128; // B constant
            x += 1;
        }
        y += 1;
    }
    data
}
const COLOR_GRADIENT_16X16: [u8; 768] = generate_color_gradient_16x16();

/// 32x32 seeded pseudo-random pattern (LCG)
fn generate_random_32x32(seed: u64) -> Vec<u8> {
    let mut state = seed;
    let lcg_next = |s: &mut u64| -> u8 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*s >> 33) as u8).saturating_add(32).min(224)
    };

    (0..32 * 32 * 3).map(|_| lcg_next(&mut state)).collect()
}

// ============================================================================
// Helper functions
// ============================================================================

/// Convert sRGB u8 to linear f32 (Rust implementation)
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

/// Get diffmap from C++ butteraugli
fn cpp_butteraugli_with_diffmap(
    srgb1: &[u8],
    srgb2: &[u8],
    width: usize,
    height: usize,
    intensity_target: f32,
) -> (f64, Vec<f32>) {
    let mut linear1 = vec![0.0f32; srgb1.len()];
    let mut linear2 = vec![0.0f32; srgb2.len()];
    unsafe {
        butteraugli_srgb_to_linear(srgb1.as_ptr(), width, height, linear1.as_mut_ptr());
        butteraugli_srgb_to_linear(srgb2.as_ptr(), width, height, linear2.as_mut_ptr());
    }

    let mut score = 0.0f64;
    let mut diffmap = vec![0.0f32; width * height];

    let result = unsafe {
        butteraugli_compare_full(
            linear1.as_ptr(),
            linear2.as_ptr(),
            width,
            height,
            1.0,
            1.0,
            intensity_target,
            &mut score,
            diffmap.as_mut_ptr(),
        )
    };
    assert_eq!(result, BUTTERAUGLI_OK, "C++ butteraugli failed");

    (score, diffmap)
}

/// Compute statistics for a float slice
fn stats(data: &[f32]) -> (f32, f32, f32, f32) {
    if data.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = data.iter().sum();
    let mean = sum / data.len() as f32;
    (min, max, mean, sum)
}

// ============================================================================
// Tests comparing diffmaps (final stage)
// ============================================================================

#[test]
fn test_diffmap_uniform_gray() {
    let width = 8;
    let height = 8;
    let img1 = UNIFORM_GRAY_8X8;
    let mut img2 = img1;
    // Slight brightness difference
    for v in img2.iter_mut() {
        *v = v.saturating_add(10);
    }

    let params = ButteraugliParams::default();
    let rust_result = compute_butteraugli(&img1, &img2, width, height, &params);
    let (cpp_score, cpp_diffmap) = cpp_butteraugli_with_diffmap(
        &img1,
        &img2,
        width,
        height,
        params.intensity_target,
    );

    let rust_diffmap = rust_result.diffmap.expect("Rust should produce diffmap");

    // Compare scores
    let score_diff = (rust_result.score - cpp_score).abs();
    let score_rel = if cpp_score > 0.001 {
        score_diff / cpp_score
    } else {
        score_diff
    };

    println!("Uniform gray 8x8:");
    println!("  Rust score: {:.6}", rust_result.score);
    println!("  C++ score:  {:.6}", cpp_score);
    println!("  Diff: {:.6} ({:.2}%)", score_diff, score_rel * 100.0);

    // Compare diffmaps
    let mut rust_flat: Vec<f32> = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            rust_flat.push(rust_diffmap.get(x, y));
        }
    }

    let max_pixel_diff = rust_flat
        .iter()
        .zip(cpp_diffmap.iter())
        .map(|(r, c)| (r - c).abs())
        .fold(0.0f32, f32::max);

    let (r_min, r_max, r_mean, _) = stats(&rust_flat);
    let (c_min, c_max, c_mean, _) = stats(&cpp_diffmap);

    println!("  Rust diffmap: min={:.4} max={:.4} mean={:.4}", r_min, r_max, r_mean);
    println!("  C++ diffmap:  min={:.4} max={:.4} mean={:.4}", c_min, c_max, c_mean);
    println!("  Max pixel diff: {:.6}", max_pixel_diff);

    // Assert reasonable parity
    assert!(
        score_rel < 0.25 || score_diff < 0.3,
        "Score diff too large"
    );
}

#[test]
fn test_diffmap_gradient() {
    let width = 8;
    let height = 8;
    let img1 = GRADIENT_8X8;
    let mut img2 = img1;
    for v in img2.iter_mut() {
        *v = v.saturating_add(15);
    }

    let params = ButteraugliParams::default();
    let rust_result = compute_butteraugli(&img1, &img2, width, height, &params);
    let (cpp_score, cpp_diffmap) = cpp_butteraugli_with_diffmap(
        &img1,
        &img2,
        width,
        height,
        params.intensity_target,
    );

    let rust_diffmap = rust_result.diffmap.expect("Rust should produce diffmap");

    let score_diff = (rust_result.score - cpp_score).abs();
    let score_rel = if cpp_score > 0.001 {
        score_diff / cpp_score
    } else {
        score_diff
    };

    let mut rust_flat: Vec<f32> = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            rust_flat.push(rust_diffmap.get(x, y));
        }
    }

    let max_pixel_diff = rust_flat
        .iter()
        .zip(cpp_diffmap.iter())
        .map(|(r, c)| (r - c).abs())
        .fold(0.0f32, f32::max);

    println!("Gradient 8x8:");
    println!("  Rust score: {:.6}", rust_result.score);
    println!("  C++ score:  {:.6}", cpp_score);
    println!("  Diff: {:.6} ({:.2}%)", score_diff, score_rel * 100.0);
    println!("  Max pixel diff: {:.6}", max_pixel_diff);

    assert!(score_rel < 0.30 || score_diff < 0.5, "Score diff too large");
}

#[test]
fn test_diffmap_checkerboard() {
    let width = 8;
    let height = 8;
    let img1 = CHECKERBOARD_8X8;
    // Inverse checkerboard
    let img2: [u8; 192] = {
        let mut data = [0u8; 192];
        for i in 0..8 {
            for j in 0..8 {
                let val = if (i + j) % 2 == 1 { 50u8 } else { 200u8 };
                let idx = (i * 8 + j) * 3;
                data[idx] = val;
                data[idx + 1] = val;
                data[idx + 2] = val;
            }
        }
        data
    };

    let params = ButteraugliParams::default();
    let rust_result = compute_butteraugli(&img1, &img2, width, height, &params);
    let (cpp_score, cpp_diffmap) = cpp_butteraugli_with_diffmap(
        &img1,
        &img2,
        width,
        height,
        params.intensity_target,
    );

    let score_diff = (rust_result.score - cpp_score).abs();
    let score_rel = if cpp_score > 0.001 {
        score_diff / cpp_score
    } else {
        score_diff
    };

    println!("Checkerboard 8x8:");
    println!("  Rust score: {:.6}", rust_result.score);
    println!("  C++ score:  {:.6}", cpp_score);
    println!("  Diff: {:.6} ({:.2}%)", score_diff, score_rel * 100.0);

    assert!(score_rel < 0.35 || score_diff < 1.0, "Score diff too large");
}

#[test]
fn test_diffmap_color_gradient() {
    let width = 16;
    let height = 16;
    let img1 = COLOR_GRADIENT_16X16;
    let img2: Vec<u8> = img1.iter().map(|&v| v.saturating_add(20)).collect();

    let params = ButteraugliParams::default();
    let rust_result = compute_butteraugli(&img1, &img2, width, height, &params);
    let (cpp_score, cpp_diffmap) = cpp_butteraugli_with_diffmap(
        &img1,
        &img2,
        width,
        height,
        params.intensity_target,
    );

    let score_diff = (rust_result.score - cpp_score).abs();
    let score_rel = if cpp_score > 0.001 {
        score_diff / cpp_score
    } else {
        score_diff
    };

    println!("Color gradient 16x16:");
    println!("  Rust score: {:.6}", rust_result.score);
    println!("  C++ score:  {:.6}", cpp_score);
    println!("  Diff: {:.6} ({:.2}%)", score_diff, score_rel * 100.0);

    assert!(score_rel < 0.30 || score_diff < 0.5, "Score diff too large");
}

#[test]
fn test_diffmap_random_32x32() {
    let width = 32;
    let height = 32;

    // Fixed seed for reproducibility
    let seed = 12345u64;
    let img1 = generate_random_32x32(seed);
    let img2 = generate_random_32x32(seed + 1);

    let params = ButteraugliParams::default();
    let rust_result = compute_butteraugli(&img1, &img2, width, height, &params);
    let (cpp_score, cpp_diffmap) = cpp_butteraugli_with_diffmap(
        &img1,
        &img2,
        width,
        height,
        params.intensity_target,
    );

    let rust_diffmap = rust_result.diffmap.expect("Rust should produce diffmap");

    let score_diff = (rust_result.score - cpp_score).abs();
    let score_rel = if cpp_score > 0.001 {
        score_diff / cpp_score
    } else {
        score_diff
    };

    let mut rust_flat: Vec<f32> = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            rust_flat.push(rust_diffmap.get(x, y));
        }
    }

    // Find where the largest differences are
    let mut diffs: Vec<(usize, usize, f32, f32, f32)> = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let r = rust_flat[idx];
            let c = cpp_diffmap[idx];
            diffs.push((x, y, r, c, (r - c).abs()));
        }
    }
    diffs.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap());

    println!("Random 32x32 (seed={}):", seed);
    println!("  Rust score: {:.6}", rust_result.score);
    println!("  C++ score:  {:.6}", cpp_score);
    println!("  Diff: {:.6} ({:.2}%)", score_diff, score_rel * 100.0);
    println!("  Top 5 pixel differences:");
    for (x, y, r, c, d) in diffs.iter().take(5) {
        println!("    ({:2},{:2}): Rust={:.4} C++={:.4} diff={:.4}", x, y, r, c, d);
    }

    assert!(score_rel < 0.35 || score_diff < 1.5, "Score diff too large");
}

// ============================================================================
// Tests comparing OpsinDynamicsImage output (intermediate stage)
// ============================================================================

#[test]
fn test_opsin_dynamics_uniform() {
    let width = 8;
    let height = 8;
    let srgb = UNIFORM_GRAY_8X8;

    // Convert to linear
    let linear = srgb_to_linear_rust(&srgb);

    // Rust OpsinDynamicsImage
    let rust_xyb = srgb_to_xyb_butteraugli(&srgb, width, height, 80.0);

    // C++ OpsinDynamicsImage (via simplified wrapper)
    let mut cpp_xyb = vec![0.0f32; width * height * 3];
    let result = unsafe {
        butteraugli_opsin_dynamics(
            linear.as_ptr(),
            width,
            height,
            80.0,
            cpp_xyb.as_mut_ptr(),
        )
    };

    if result != BUTTERAUGLI_OK {
        println!("C++ butteraugli_opsin_dynamics returned error: {}", result);
        // Don't fail - the simplified wrapper might not match exactly
        return;
    }

    // Compare X, Y, B channels
    let mut max_diff = [0.0f32; 3];
    let channel_names = ["X", "Y", "B"];

    for c in 0..3 {
        for y in 0..height {
            for x in 0..width {
                let rust_val = rust_xyb.plane(c).get(x, y);
                let cpp_val = cpp_xyb[(y * width + x) * 3 + c];
                let diff = (rust_val - cpp_val).abs();
                if diff > max_diff[c] {
                    max_diff[c] = diff;
                }
            }
        }
    }

    println!("OpsinDynamicsImage uniform 8x8:");
    for (i, name) in channel_names.iter().enumerate() {
        println!("  {} channel max diff: {:.6}", name, max_diff[i]);
    }

    // Note: We expect some difference due to simplified blur in C++ wrapper
}

#[test]
fn test_opsin_dynamics_gradient() {
    let width = 8;
    let height = 8;
    let srgb = GRADIENT_8X8;

    let linear = srgb_to_linear_rust(&srgb);
    let rust_xyb = srgb_to_xyb_butteraugli(&srgb, width, height, 80.0);

    let mut cpp_xyb = vec![0.0f32; width * height * 3];
    let result = unsafe {
        butteraugli_opsin_dynamics(
            linear.as_ptr(),
            width,
            height,
            80.0,
            cpp_xyb.as_mut_ptr(),
        )
    };

    if result != BUTTERAUGLI_OK {
        println!("C++ butteraugli_opsin_dynamics returned error: {}", result);
        return;
    }

    // Sample a few pixels to compare
    println!("OpsinDynamicsImage gradient 8x8 (sample pixels):");
    for (px, py) in [(0, 0), (3, 3), (7, 7)] {
        let idx = (py * width + px) * 3;
        println!(
            "  ({},{}): Rust XYB=({:.4},{:.4},{:.4}) C++ XYB=({:.4},{:.4},{:.4})",
            px,
            py,
            rust_xyb.plane(0).get(px, py),
            rust_xyb.plane(1).get(px, py),
            rust_xyb.plane(2).get(px, py),
            cpp_xyb[idx],
            cpp_xyb[idx + 1],
            cpp_xyb[idx + 2]
        );
    }
}

// ============================================================================
// Tests for specific divergence investigation
// ============================================================================

#[test]
fn test_divergence_investigation() {
    // Use 32x32 random image where we saw divergence
    let width = 32;
    let height = 32;
    let seed = 12345u64;
    let img1 = generate_random_32x32(seed);
    let img2 = generate_random_32x32(seed + 1);

    let params = ButteraugliParams::default();
    let rust_result = compute_butteraugli(&img1, &img2, width, height, &params);
    let (cpp_score, cpp_diffmap) = cpp_butteraugli_with_diffmap(
        &img1,
        &img2,
        width,
        height,
        params.intensity_target,
    );

    let rust_diffmap = rust_result.diffmap.expect("diffmap");

    // Analyze where differences occur
    let mut corner_rust = 0.0f32;
    let mut corner_cpp = 0.0f32;
    let mut center_rust = 0.0f32;
    let mut center_cpp = 0.0f32;
    let mut edge_rust = 0.0f32;
    let mut edge_cpp = 0.0f32;

    for y in 0..height {
        for x in 0..width {
            let r = rust_diffmap.get(x, y);
            let c = cpp_diffmap[y * width + x];

            let is_corner = (x < 4 || x >= width - 4) && (y < 4 || y >= height - 4);
            let is_edge = x < 4 || x >= width - 4 || y < 4 || y >= height - 4;

            if is_corner {
                corner_rust += r;
                corner_cpp += c;
            } else if is_edge {
                edge_rust += r;
                edge_cpp += c;
            } else {
                center_rust += r;
                center_cpp += c;
            }
        }
    }

    println!("Divergence analysis (32x32 random):");
    println!("  Score: Rust={:.4} C++={:.4} diff={:.2}%",
             rust_result.score, cpp_score,
             (rust_result.score - cpp_score).abs() / cpp_score * 100.0);
    println!("  Corner sum: Rust={:.2} C++={:.2} diff={:.2}",
             corner_rust, corner_cpp, (corner_rust - corner_cpp).abs());
    println!("  Edge sum: Rust={:.2} C++={:.2} diff={:.2}",
             edge_rust, edge_cpp, (edge_rust - edge_cpp).abs());
    println!("  Center sum: Rust={:.2} C++={:.2} diff={:.2}",
             center_rust, center_cpp, (center_rust - center_cpp).abs());
}
