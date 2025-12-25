//! Capture C++ butteraugli reference values for crates.io-ready tests.
//!
//! This script generates synthetic test images, runs them through both
//! Rust and C++ butteraugli, and outputs hard-coded reference data that
//! can be used in tests without requiring jpegli-sys at runtime.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example capture_cpp_reference > src/test_reference_data.rs
//! ```
//!
//! Or to just view the data:
//! ```bash
//! cargo run --example capture_cpp_reference
//! ```

use jpegli_sys::{butteraugli_compare_full, butteraugli_srgb_to_linear, BUTTERAUGLI_OK};

// ============================================================================
// Image Generation Functions
// ============================================================================

/// LCG pseudo-random number generator (deterministic)
struct Lcg {
    state: u64,
}

impl Lcg {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u8(&mut self) -> u8 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.state >> 33) & 0xFF) as u8
    }

    fn next_u8_range(&mut self, min: u8, max: u8) -> u8 {
        let range = (max - min) as u64 + 1;
        let val = self.next_u8() as u64;
        (min as u64 + (val * range / 256)) as u8
    }
}

/// Generate uniform color image
fn gen_uniform(width: usize, height: usize, r: u8, g: u8, b: u8) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height * 3);
    for _ in 0..width * height {
        data.push(r);
        data.push(g);
        data.push(b);
    }
    data
}

/// Generate horizontal gradient (grayscale)
fn gen_gradient_h(width: usize, height: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height * 3);
    for _y in 0..height {
        for x in 0..width {
            let val = if width > 1 {
                (x * 255 / (width - 1)) as u8
            } else {
                128
            };
            data.push(val);
            data.push(val);
            data.push(val);
        }
    }
    data
}

/// Generate vertical gradient (grayscale)
fn gen_gradient_v(width: usize, height: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        let val = if height > 1 {
            (y * 255 / (height - 1)) as u8
        } else {
            128
        };
        for _x in 0..width {
            data.push(val);
            data.push(val);
            data.push(val);
        }
    }
    data
}

/// Generate diagonal gradient (grayscale)
fn gen_gradient_diag(width: usize, height: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height * 3);
    let max_dist = width + height - 2;
    for y in 0..height {
        for x in 0..width {
            let val = if max_dist > 0 {
                ((x + y) * 255 / max_dist) as u8
            } else {
                128
            };
            data.push(val);
            data.push(val);
            data.push(val);
        }
    }
    data
}

/// Generate RGB color gradient
fn gen_color_gradient(width: usize, height: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let r = if width > 1 { (x * 255 / (width - 1)) as u8 } else { 128 };
            let g = if height > 1 { (y * 255 / (height - 1)) as u8 } else { 128 };
            let b = 128;
            data.push(r);
            data.push(g);
            data.push(b);
        }
    }
    data
}

/// Generate checkerboard pattern
fn gen_checkerboard(width: usize, height: usize, block_size: usize, lo: u8, hi: u8) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let checker = ((x / block_size) + (y / block_size)) % 2 == 0;
            let val = if checker { hi } else { lo };
            data.push(val);
            data.push(val);
            data.push(val);
        }
    }
    data
}

/// Generate inverse checkerboard
fn gen_checkerboard_inv(width: usize, height: usize, block_size: usize, lo: u8, hi: u8) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let checker = ((x / block_size) + (y / block_size)) % 2 == 1;
            let val = if checker { hi } else { lo };
            data.push(val);
            data.push(val);
            data.push(val);
        }
    }
    data
}

/// Generate horizontal stripes
fn gen_stripes_h(width: usize, height: usize, stripe_height: usize, lo: u8, hi: u8) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        let val = if (y / stripe_height) % 2 == 0 { hi } else { lo };
        for _x in 0..width {
            data.push(val);
            data.push(val);
            data.push(val);
        }
    }
    data
}

/// Generate vertical stripes
fn gen_stripes_v(width: usize, height: usize, stripe_width: usize, lo: u8, hi: u8) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height * 3);
    for _y in 0..height {
        for x in 0..width {
            let val = if (x / stripe_width) % 2 == 0 { hi } else { lo };
            data.push(val);
            data.push(val);
            data.push(val);
        }
    }
    data
}

/// Generate seeded random image
fn gen_random(width: usize, height: usize, seed: u64) -> Vec<u8> {
    let mut rng = Lcg::new(seed);
    let mut data = Vec::with_capacity(width * height * 3);
    for _ in 0..width * height * 3 {
        data.push(rng.next_u8());
    }
    data
}

/// Generate seeded random image with limited range (avoids extremes)
fn gen_random_midrange(width: usize, height: usize, seed: u64) -> Vec<u8> {
    let mut rng = Lcg::new(seed);
    let mut data = Vec::with_capacity(width * height * 3);
    for _ in 0..width * height * 3 {
        data.push(rng.next_u8_range(32, 224));
    }
    data
}

/// Generate smooth sine wave pattern
fn gen_sine_wave(width: usize, height: usize, freq_x: f32, freq_y: f32) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let fx = (x as f32 * freq_x * std::f32::consts::TAU / width as f32).sin();
            let fy = (y as f32 * freq_y * std::f32::consts::TAU / height as f32).sin();
            let val = ((fx + fy + 2.0) / 4.0 * 255.0) as u8;
            data.push(val);
            data.push(val);
            data.push(val);
        }
    }
    data
}

/// Generate radial gradient (distance from center)
fn gen_radial(width: usize, height: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height * 3);
    let cx = width as f32 / 2.0;
    let cy = height as f32 / 2.0;
    let max_dist = (cx * cx + cy * cy).sqrt();
    for y in 0..height {
        for x in 0..width {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let dist = (dx * dx + dy * dy).sqrt();
            let val = ((1.0 - dist / max_dist) * 255.0).clamp(0.0, 255.0) as u8;
            data.push(val);
            data.push(val);
            data.push(val);
        }
    }
    data
}

/// Generate edge pattern (sharp transition at center)
fn gen_edge_v(width: usize, height: usize, lo: u8, hi: u8) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height * 3);
    let mid = width / 2;
    for _y in 0..height {
        for x in 0..width {
            let val = if x < mid { lo } else { hi };
            data.push(val);
            data.push(val);
            data.push(val);
        }
    }
    data
}

/// Generate edge pattern (horizontal)
fn gen_edge_h(width: usize, height: usize, lo: u8, hi: u8) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height * 3);
    let mid = height / 2;
    for y in 0..height {
        let val = if y < mid { lo } else { hi };
        for _x in 0..width {
            data.push(val);
            data.push(val);
            data.push(val);
        }
    }
    data
}

// ============================================================================
// Distortion Functions
// ============================================================================

/// Add uniform brightness shift
fn distort_brightness(img: &[u8], delta: i16) -> Vec<u8> {
    img.iter()
        .map(|&v| (v as i16 + delta).clamp(0, 255) as u8)
        .collect()
}

/// Add per-pixel noise with fixed seed
fn distort_noise(img: &[u8], seed: u64, amplitude: u8) -> Vec<u8> {
    let mut rng = Lcg::new(seed);
    img.iter()
        .map(|&v| {
            let noise = rng.next_u8() as i16 - 128;
            let scaled = noise * amplitude as i16 / 128;
            (v as i16 + scaled).clamp(0, 255) as u8
        })
        .collect()
}

/// Contrast adjustment
fn distort_contrast(img: &[u8], factor: f32) -> Vec<u8> {
    img.iter()
        .map(|&v| {
            let centered = v as f32 - 128.0;
            let adjusted = centered * factor + 128.0;
            adjusted.clamp(0.0, 255.0) as u8
        })
        .collect()
}

/// Gamma adjustment
fn distort_gamma(img: &[u8], gamma: f32) -> Vec<u8> {
    img.iter()
        .map(|&v| {
            let normalized = v as f32 / 255.0;
            let adjusted = normalized.powf(gamma);
            (adjusted * 255.0).clamp(0.0, 255.0) as u8
        })
        .collect()
}

/// Simple box blur (3x3)
fn distort_blur(img: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut out = vec![0u8; img.len()];
    for y in 0..height {
        for x in 0..width {
            for c in 0..3 {
                let mut sum = 0u32;
                let mut count = 0u32;
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;
                        if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                            let idx = (ny as usize * width + nx as usize) * 3 + c;
                            sum += img[idx] as u32;
                            count += 1;
                        }
                    }
                }
                let idx = (y * width + x) * 3 + c;
                out[idx] = (sum / count) as u8;
            }
        }
    }
    out
}

/// Channel swap (R <-> B)
fn distort_channel_swap_rb(img: &[u8]) -> Vec<u8> {
    let mut out = img.to_vec();
    for chunk in out.chunks_mut(3) {
        chunk.swap(0, 2);
    }
    out
}

/// Hue shift (rotate RGB)
fn distort_hue_shift(img: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(img.len());
    for chunk in img.chunks(3) {
        out.push(chunk[1]); // G -> R
        out.push(chunk[2]); // B -> G
        out.push(chunk[0]); // R -> B
    }
    out
}

/// Quantize to fewer levels
fn distort_quantize(img: &[u8], levels: u8) -> Vec<u8> {
    let step = 256 / levels as u16;
    img.iter()
        .map(|&v| {
            let bucket = v as u16 / step;
            (bucket * step + step / 2).min(255) as u8
        })
        .collect()
}

// ============================================================================
// C++ Butteraugli Interface
// ============================================================================

fn srgb_to_linear(srgb: &[u8], width: usize, height: usize) -> Vec<f32> {
    let mut linear = vec![0.0f32; srgb.len()];
    unsafe {
        butteraugli_srgb_to_linear(srgb.as_ptr(), width, height, linear.as_mut_ptr());
    }
    linear
}

fn cpp_butteraugli(
    img1: &[u8],
    img2: &[u8],
    width: usize,
    height: usize,
    intensity_target: f32,
) -> Option<(f64, Vec<f32>)> {
    let linear1 = srgb_to_linear(img1, width, height);
    let linear2 = srgb_to_linear(img2, width, height);

    let mut score = 0.0f64;
    let mut diffmap = vec![0.0f32; width * height];

    let result = unsafe {
        butteraugli_compare_full(
            linear1.as_ptr(),
            linear2.as_ptr(),
            width,
            height,
            1.0, // hf_asymmetry
            1.0, // xmul
            intensity_target,
            &mut score,
            diffmap.as_mut_ptr(),
        )
    };

    if result == BUTTERAUGLI_OK {
        Some((score, diffmap))
    } else {
        None
    }
}

// ============================================================================
// Statistics
// ============================================================================

#[derive(Debug, Clone, Copy)]
struct DiffmapStats {
    min: f32,
    max: f32,
    mean: f32,
    std: f32,
}

impl DiffmapStats {
    fn compute(data: &[f32]) -> Self {
        if data.is_empty() {
            return Self {
                min: 0.0,
                max: 0.0,
                mean: 0.0,
                std: 0.0,
            };
        }

        let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = data.iter().sum();
        let mean = sum / data.len() as f32;
        let variance: f32 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
        let std = variance.sqrt();

        Self { min, max, mean, std }
    }
}

// ============================================================================
// Test Case Definition
// ============================================================================

struct TestCase {
    name: &'static str,
    width: usize,
    height: usize,
    gen_a: Box<dyn Fn() -> Vec<u8>>,
    gen_b: Box<dyn Fn() -> Vec<u8>>,
    /// Description of how to regenerate image A
    gen_a_code: &'static str,
    /// Description of how to regenerate image B
    gen_b_code: &'static str,
}

impl TestCase {
    fn new(
        name: &'static str,
        width: usize,
        height: usize,
        gen_a: impl Fn() -> Vec<u8> + 'static,
        gen_b: impl Fn() -> Vec<u8> + 'static,
        gen_a_code: &'static str,
        gen_b_code: &'static str,
    ) -> Self {
        Self {
            name,
            width,
            height,
            gen_a: Box::new(gen_a),
            gen_b: Box::new(gen_b),
            gen_a_code,
            gen_b_code,
        }
    }
}

// ============================================================================
// Generate All Test Cases
// ============================================================================

fn generate_test_cases() -> Vec<TestCase> {
    let mut cases = Vec::new();

    // Various sizes including non-mod-8
    let sizes: &[(usize, usize)] = &[
        (7, 7),
        (8, 8),
        (9, 9),
        (15, 15),
        (16, 16),
        (17, 17),
        (23, 23),
        (24, 24),
        (31, 31),
        (32, 32),
        (33, 33),
        (47, 47),
        (48, 48),
        (63, 63),
        (64, 64),
        (65, 65),
        // Non-square
        (7, 11),
        (11, 7),
        (15, 23),
        (23, 15),
        (17, 31),
        (31, 17),
        (33, 47),
        (47, 33),
    ];

    // ========== UNIFORM IMAGES ==========
    for &(w, h) in sizes.iter().take(10) {
        // Gray brightness shift +10
        let name = Box::leak(format!("uniform_gray_128_shift_10_{}x{}", w, h).into_boxed_str());
        cases.push(TestCase::new(
            name,
            w,
            h,
            move || gen_uniform(w, h, 128, 128, 128),
            move || gen_uniform(w, h, 138, 138, 138),
            "gen_uniform(W, H, 128, 128, 128)",
            "gen_uniform(W, H, 138, 138, 138)",
        ));

        // Gray brightness shift +50
        let name = Box::leak(format!("uniform_gray_128_shift_50_{}x{}", w, h).into_boxed_str());
        cases.push(TestCase::new(
            name,
            w,
            h,
            move || gen_uniform(w, h, 128, 128, 128),
            move || gen_uniform(w, h, 178, 178, 178),
            "gen_uniform(W, H, 128, 128, 128)",
            "gen_uniform(W, H, 178, 178, 178)",
        ));
    }

    // Color uniform shifts
    for &(w, h) in &[(16, 16), (23, 23), (32, 32)] {
        let name = Box::leak(format!("uniform_red_shift_20_{}x{}", w, h).into_boxed_str());
        cases.push(TestCase::new(
            name,
            w,
            h,
            move || gen_uniform(w, h, 128, 64, 64),
            move || gen_uniform(w, h, 148, 64, 64),
            "gen_uniform(W, H, 128, 64, 64)",
            "gen_uniform(W, H, 148, 64, 64)",
        ));

        let name = Box::leak(format!("uniform_green_shift_20_{}x{}", w, h).into_boxed_str());
        cases.push(TestCase::new(
            name,
            w,
            h,
            move || gen_uniform(w, h, 64, 128, 64),
            move || gen_uniform(w, h, 64, 148, 64),
            "gen_uniform(W, H, 64, 128, 64)",
            "gen_uniform(W, H, 64, 148, 64)",
        ));

        let name = Box::leak(format!("uniform_blue_shift_20_{}x{}", w, h).into_boxed_str());
        cases.push(TestCase::new(
            name,
            w,
            h,
            move || gen_uniform(w, h, 64, 64, 128),
            move || gen_uniform(w, h, 64, 64, 148),
            "gen_uniform(W, H, 64, 64, 128)",
            "gen_uniform(W, H, 64, 64, 148)",
        ));
    }

    // ========== GRADIENTS ==========
    for &(w, h) in sizes.iter().take(12) {
        // Horizontal gradient with brightness shift
        let name = Box::leak(format!("gradient_h_shift_15_{}x{}", w, h).into_boxed_str());
        cases.push(TestCase::new(
            name,
            w,
            h,
            move || gen_gradient_h(w, h),
            move || distort_brightness(&gen_gradient_h(w, h), 15),
            "gen_gradient_h(W, H)",
            "distort_brightness(&gen_gradient_h(W, H), 15)",
        ));

        // Vertical gradient with brightness shift
        let name = Box::leak(format!("gradient_v_shift_15_{}x{}", w, h).into_boxed_str());
        cases.push(TestCase::new(
            name,
            w,
            h,
            move || gen_gradient_v(w, h),
            move || distort_brightness(&gen_gradient_v(w, h), 15),
            "gen_gradient_v(W, H)",
            "distort_brightness(&gen_gradient_v(W, H), 15)",
        ));
    }

    // Diagonal gradient
    for &(w, h) in &[(16, 16), (23, 31), (32, 32), (47, 33)] {
        let name = Box::leak(format!("gradient_diag_shift_20_{}x{}", w, h).into_boxed_str());
        cases.push(TestCase::new(
            name,
            w,
            h,
            move || gen_gradient_diag(w, h),
            move || distort_brightness(&gen_gradient_diag(w, h), 20),
            "gen_gradient_diag(W, H)",
            "distort_brightness(&gen_gradient_diag(W, H), 20)",
        ));
    }

    // Color gradient
    for &(w, h) in &[(16, 16), (23, 23), (32, 32), (33, 47)] {
        let name = Box::leak(format!("color_gradient_shift_10_{}x{}", w, h).into_boxed_str());
        cases.push(TestCase::new(
            name,
            w,
            h,
            move || gen_color_gradient(w, h),
            move || distort_brightness(&gen_color_gradient(w, h), 10),
            "gen_color_gradient(W, H)",
            "distort_brightness(&gen_color_gradient(W, H), 10)",
        ));
    }

    // ========== CHECKERBOARD ==========
    for &(w, h) in &[(8, 8), (15, 15), (16, 16), (23, 23), (32, 32), (33, 33)] {
        // Checkerboard vs inverse
        let name = Box::leak(format!("checkerboard_vs_inverse_1px_{}x{}", w, h).into_boxed_str());
        cases.push(TestCase::new(
            name,
            w,
            h,
            move || gen_checkerboard(w, h, 1, 50, 200),
            move || gen_checkerboard_inv(w, h, 1, 50, 200),
            "gen_checkerboard(W, H, 1, 50, 200)",
            "gen_checkerboard_inv(W, H, 1, 50, 200)",
        ));

        // Checkerboard with 2px blocks
        let name = Box::leak(format!("checkerboard_vs_inverse_2px_{}x{}", w, h).into_boxed_str());
        cases.push(TestCase::new(
            name,
            w,
            h,
            move || gen_checkerboard(w, h, 2, 50, 200),
            move || gen_checkerboard_inv(w, h, 2, 50, 200),
            "gen_checkerboard(W, H, 2, 50, 200)",
            "gen_checkerboard_inv(W, H, 2, 50, 200)",
        ));

        // Checkerboard with brightness shift
        let name = Box::leak(format!("checkerboard_shift_10_{}x{}", w, h).into_boxed_str());
        cases.push(TestCase::new(
            name,
            w,
            h,
            move || gen_checkerboard(w, h, 2, 50, 200),
            move || distort_brightness(&gen_checkerboard(w, h, 2, 50, 200), 10),
            "gen_checkerboard(W, H, 2, 50, 200)",
            "distort_brightness(&gen_checkerboard(W, H, 2, 50, 200), 10)",
        ));
    }

    // Larger block checkerboard
    for &(w, h) in &[(32, 32), (47, 47), (64, 64)] {
        let name = Box::leak(format!("checkerboard_vs_inverse_4px_{}x{}", w, h).into_boxed_str());
        cases.push(TestCase::new(
            name,
            w,
            h,
            move || gen_checkerboard(w, h, 4, 30, 220),
            move || gen_checkerboard_inv(w, h, 4, 30, 220),
            "gen_checkerboard(W, H, 4, 30, 220)",
            "gen_checkerboard_inv(W, H, 4, 30, 220)",
        ));

        let name = Box::leak(format!("checkerboard_vs_inverse_8px_{}x{}", w, h).into_boxed_str());
        cases.push(TestCase::new(
            name,
            w,
            h,
            move || gen_checkerboard(w, h, 8, 30, 220),
            move || gen_checkerboard_inv(w, h, 8, 30, 220),
            "gen_checkerboard(W, H, 8, 30, 220)",
            "gen_checkerboard_inv(W, H, 8, 30, 220)",
        ));
    }

    // ========== STRIPES ==========
    for &(w, h) in &[(16, 16), (23, 23), (32, 32), (33, 47)] {
        let name = Box::leak(format!("stripes_h_2px_shift_15_{}x{}", w, h).into_boxed_str());
        cases.push(TestCase::new(
            name,
            w,
            h,
            move || gen_stripes_h(w, h, 2, 50, 200),
            move || distort_brightness(&gen_stripes_h(w, h, 2, 50, 200), 15),
            "gen_stripes_h(W, H, 2, 50, 200)",
            "distort_brightness(&gen_stripes_h(W, H, 2, 50, 200), 15)",
        ));

        let name = Box::leak(format!("stripes_v_2px_shift_15_{}x{}", w, h).into_boxed_str());
        cases.push(TestCase::new(
            name,
            w,
            h,
            move || gen_stripes_v(w, h, 2, 50, 200),
            move || distort_brightness(&gen_stripes_v(w, h, 2, 50, 200), 15),
            "gen_stripes_v(W, H, 2, 50, 200)",
            "distort_brightness(&gen_stripes_v(W, H, 2, 50, 200), 15)",
        ));
    }

    // ========== SINE WAVES ==========
    for &(w, h) in &[(32, 32), (33, 33), (47, 47), (64, 64)] {
        let name = Box::leak(format!("sine_1x1_shift_10_{}x{}", w, h).into_boxed_str());
        cases.push(TestCase::new(
            name,
            w,
            h,
            move || gen_sine_wave(w, h, 1.0, 1.0),
            move || distort_brightness(&gen_sine_wave(w, h, 1.0, 1.0), 10),
            "gen_sine_wave(W, H, 1.0, 1.0)",
            "distort_brightness(&gen_sine_wave(W, H, 1.0, 1.0), 10)",
        ));

        let name = Box::leak(format!("sine_2x2_shift_10_{}x{}", w, h).into_boxed_str());
        cases.push(TestCase::new(
            name,
            w,
            h,
            move || gen_sine_wave(w, h, 2.0, 2.0),
            move || distort_brightness(&gen_sine_wave(w, h, 2.0, 2.0), 10),
            "gen_sine_wave(W, H, 2.0, 2.0)",
            "distort_brightness(&gen_sine_wave(W, H, 2.0, 2.0), 10)",
        ));

        let name = Box::leak(format!("sine_4x4_shift_10_{}x{}", w, h).into_boxed_str());
        cases.push(TestCase::new(
            name,
            w,
            h,
            move || gen_sine_wave(w, h, 4.0, 4.0),
            move || distort_brightness(&gen_sine_wave(w, h, 4.0, 4.0), 10),
            "gen_sine_wave(W, H, 4.0, 4.0)",
            "distort_brightness(&gen_sine_wave(W, H, 4.0, 4.0), 10)",
        ));
    }

    // ========== RADIAL ==========
    for &(w, h) in &[(16, 16), (23, 23), (32, 32), (47, 47)] {
        let name = Box::leak(format!("radial_shift_15_{}x{}", w, h).into_boxed_str());
        cases.push(TestCase::new(
            name,
            w,
            h,
            move || gen_radial(w, h),
            move || distort_brightness(&gen_radial(w, h), 15),
            "gen_radial(W, H)",
            "distort_brightness(&gen_radial(W, H), 15)",
        ));
    }

    // ========== EDGES ==========
    for &(w, h) in &[(16, 16), (23, 31), (32, 32), (47, 33)] {
        let name = Box::leak(format!("edge_v_shift_10_{}x{}", w, h).into_boxed_str());
        cases.push(TestCase::new(
            name,
            w,
            h,
            move || gen_edge_v(w, h, 50, 200),
            move || distort_brightness(&gen_edge_v(w, h, 50, 200), 10),
            "gen_edge_v(W, H, 50, 200)",
            "distort_brightness(&gen_edge_v(W, H, 50, 200), 10)",
        ));

        let name = Box::leak(format!("edge_h_shift_10_{}x{}", w, h).into_boxed_str());
        cases.push(TestCase::new(
            name,
            w,
            h,
            move || gen_edge_h(w, h, 50, 200),
            move || distort_brightness(&gen_edge_h(w, h, 50, 200), 10),
            "gen_edge_h(W, H, 50, 200)",
            "distort_brightness(&gen_edge_h(W, H, 50, 200), 10)",
        ));

        // Edge vs blur
        let name = Box::leak(format!("edge_v_vs_blur_{}x{}", w, h).into_boxed_str());
        cases.push(TestCase::new(
            name,
            w,
            h,
            move || gen_edge_v(w, h, 50, 200),
            move || distort_blur(&gen_edge_v(w, h, 50, 200), w, h),
            "gen_edge_v(W, H, 50, 200)",
            "distort_blur(&gen_edge_v(W, H, 50, 200), W, H)",
        ));
    }

    // ========== RANDOM (FIXED SEED) ==========
    let seeds: &[u64] = &[
        0x12345678_9ABCDEF0,
        0xDEADBEEF_CAFEBABE,
        0x0BADC0DE_FEEDFACE,
        0x13371337_42424242,
        0xAAAAAAAA_55555555,
    ];

    for (seed_idx, &seed) in seeds.iter().enumerate() {
        for &(w, h) in &[(16, 16), (23, 23), (32, 32), (33, 47), (47, 33)] {
            // Random with brightness shift
            let name = Box::leak(
                format!("random_seed{}_shift_10_{}x{}", seed_idx, w, h).into_boxed_str(),
            );
            cases.push(TestCase::new(
                name,
                w,
                h,
                move || gen_random(w, h, seed),
                move || distort_brightness(&gen_random(w, h, seed), 10),
                "gen_random(W, H, SEED)",
                "distort_brightness(&gen_random(W, H, SEED), 10)",
            ));

            // Random with noise
            let name = Box::leak(
                format!("random_seed{}_noise_20_{}x{}", seed_idx, w, h).into_boxed_str(),
            );
            let noise_seed = seed.wrapping_add(1);
            cases.push(TestCase::new(
                name,
                w,
                h,
                move || gen_random(w, h, seed),
                move || distort_noise(&gen_random(w, h, seed), noise_seed, 20),
                "gen_random(W, H, SEED)",
                "distort_noise(&gen_random(W, H, SEED), NOISE_SEED, 20)",
            ));
        }
    }

    // Random midrange with various distortions
    for &(w, h) in &[(32, 32), (47, 47), (64, 64)] {
        let seed = 0xFEDCBA98_76543210u64;

        // Contrast
        let name = Box::leak(format!("random_mid_contrast_1.2_{}x{}", w, h).into_boxed_str());
        cases.push(TestCase::new(
            name,
            w,
            h,
            move || gen_random_midrange(w, h, seed),
            move || distort_contrast(&gen_random_midrange(w, h, seed), 1.2),
            "gen_random_midrange(W, H, 0xFEDCBA9876543210)",
            "distort_contrast(&gen_random_midrange(W, H, 0xFEDCBA9876543210), 1.2)",
        ));

        // Gamma
        let name = Box::leak(format!("random_mid_gamma_0.9_{}x{}", w, h).into_boxed_str());
        cases.push(TestCase::new(
            name,
            w,
            h,
            move || gen_random_midrange(w, h, seed),
            move || distort_gamma(&gen_random_midrange(w, h, seed), 0.9),
            "gen_random_midrange(W, H, 0xFEDCBA9876543210)",
            "distort_gamma(&gen_random_midrange(W, H, 0xFEDCBA9876543210), 0.9)",
        ));

        // Blur
        let name = Box::leak(format!("random_mid_blur_{}x{}", w, h).into_boxed_str());
        cases.push(TestCase::new(
            name,
            w,
            h,
            move || gen_random_midrange(w, h, seed),
            move || distort_blur(&gen_random_midrange(w, h, seed), w, h),
            "gen_random_midrange(W, H, 0xFEDCBA9876543210)",
            "distort_blur(&gen_random_midrange(W, H, 0xFEDCBA9876543210), W, H)",
        ));

        // Quantize
        let name = Box::leak(format!("random_mid_quantize_32_{}x{}", w, h).into_boxed_str());
        cases.push(TestCase::new(
            name,
            w,
            h,
            move || gen_random_midrange(w, h, seed),
            move || distort_quantize(&gen_random_midrange(w, h, seed), 32),
            "gen_random_midrange(W, H, 0xFEDCBA9876543210)",
            "distort_quantize(&gen_random_midrange(W, H, 0xFEDCBA9876543210), 32)",
        ));
    }

    // ========== COLOR DISTORTIONS ==========
    for &(w, h) in &[(16, 16), (23, 23), (32, 32), (47, 33)] {
        // Channel swap
        let name = Box::leak(format!("color_grad_channel_swap_{}x{}", w, h).into_boxed_str());
        cases.push(TestCase::new(
            name,
            w,
            h,
            move || gen_color_gradient(w, h),
            move || distort_channel_swap_rb(&gen_color_gradient(w, h)),
            "gen_color_gradient(W, H)",
            "distort_channel_swap_rb(&gen_color_gradient(W, H))",
        ));

        // Hue shift
        let name = Box::leak(format!("color_grad_hue_shift_{}x{}", w, h).into_boxed_str());
        cases.push(TestCase::new(
            name,
            w,
            h,
            move || gen_color_gradient(w, h),
            move || distort_hue_shift(&gen_color_gradient(w, h)),
            "gen_color_gradient(W, H)",
            "distort_hue_shift(&gen_color_gradient(W, H))",
        ));
    }

    // Random color images
    for &(w, h) in &[(32, 32), (47, 47)] {
        let seed = 0x1234567890ABCDEFu64;

        let name = Box::leak(format!("random_color_channel_swap_{}x{}", w, h).into_boxed_str());
        cases.push(TestCase::new(
            name,
            w,
            h,
            move || gen_random(w, h, seed),
            move || distort_channel_swap_rb(&gen_random(w, h, seed)),
            "gen_random(W, H, 0x1234567890ABCDEF)",
            "distort_channel_swap_rb(&gen_random(W, H, 0x1234567890ABCDEF))",
        ));

        let name = Box::leak(format!("random_color_hue_shift_{}x{}", w, h).into_boxed_str());
        cases.push(TestCase::new(
            name,
            w,
            h,
            move || gen_random(w, h, seed),
            move || distort_hue_shift(&gen_random(w, h, seed)),
            "gen_random(W, H, 0x1234567890ABCDEF)",
            "distort_hue_shift(&gen_random(W, H, 0x1234567890ABCDEF))",
        ));
    }

    cases
}

// ============================================================================
// Main: Run and Output
// ============================================================================

fn main() {
    let cases = generate_test_cases();
    let intensity_target = 80.0f32;

    eprintln!("Capturing C++ butteraugli reference data for {} test cases...", cases.len());
    eprintln!();

    // Track statistics
    let mut success_count = 0;
    let mut fail_count = 0;
    let mut results: Vec<(String, usize, usize, f64, DiffmapStats)> = Vec::new();

    for case in &cases {
        let img_a = (case.gen_a)();
        let img_b = (case.gen_b)();

        match cpp_butteraugli(&img_a, &img_b, case.width, case.height, intensity_target) {
            Some((score, diffmap)) => {
                let stats = DiffmapStats::compute(&diffmap);
                results.push((
                    case.name.to_string(),
                    case.width,
                    case.height,
                    score,
                    stats,
                ));
                success_count += 1;
                eprintln!(
                    "  ✓ {} ({}x{}): score={:.6}",
                    case.name, case.width, case.height, score
                );
            }
            None => {
                fail_count += 1;
                eprintln!(
                    "  ✗ {} ({}x{}): C++ butteraugli failed",
                    case.name, case.width, case.height
                );
            }
        }
    }

    eprintln!();
    eprintln!("Results: {} succeeded, {} failed", success_count, fail_count);
    eprintln!();

    // Output Rust code
    println!("//! Auto-generated C++ butteraugli reference data.");
    println!("//!");
    println!("//! Generated by: cargo run --example capture_cpp_reference");
    println!("//! Date: {}", chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC"));
    println!("//! Total test cases: {}", success_count);
    println!("//!");
    println!("//! This file contains reference values captured from the C++ butteraugli");
    println!("//! implementation. These values are used for regression testing without");
    println!("//! requiring jpegli-sys at runtime.");
    println!();
    println!("#![allow(clippy::excessive_precision)]");
    println!();
    println!("/// Diffmap statistics for a test case.");
    println!("#[derive(Debug, Clone, Copy)]");
    println!("pub struct DiffmapStats {{");
    println!("    pub min: f32,");
    println!("    pub max: f32,");
    println!("    pub mean: f32,");
    println!("    pub std: f32,");
    println!("}}");
    println!();
    println!("/// A reference test case with expected C++ butteraugli values.");
    println!("#[derive(Debug, Clone)]");
    println!("pub struct ReferenceCase {{");
    println!("    pub name: &'static str,");
    println!("    pub width: usize,");
    println!("    pub height: usize,");
    println!("    pub expected_score: f64,");
    println!("    pub expected_stats: DiffmapStats,");
    println!("}}");
    println!();
    println!("/// All reference test cases.");
    println!("pub const REFERENCE_CASES: &[ReferenceCase] = &[");

    for (name, width, height, score, stats) in &results {
        println!("    ReferenceCase {{");
        println!("        name: \"{}\",", name);
        println!("        width: {},", width);
        println!("        height: {},", height);
        println!("        expected_score: {:.15},", score);
        println!("        expected_stats: DiffmapStats {{");
        println!("            min: {:.10},", stats.min);
        println!("            max: {:.10},", stats.max);
        println!("            mean: {:.10},", stats.mean);
        println!("            std: {:.10},", stats.std);
        println!("        }},");
        println!("    }},");
    }

    println!("];");
    println!();

    // Output intensity target used
    println!("/// Intensity target used when capturing reference data.");
    println!("pub const REFERENCE_INTENSITY_TARGET: f32 = {:.1};", intensity_target);
    println!();

    // Output test count
    println!("/// Number of reference test cases.");
    println!("pub const REFERENCE_CASE_COUNT: usize = {};", success_count);
}
