//! XYB conversion parity test - absolute error vs C++ jpegli.
//!
//! Run with: cargo run --release --features "test-utils,ffi-tests" --example xyb_ulp_parity

use std::f32;

// ============================================================================
// Absolute error statistics (more meaningful than ULP for tiny differences)
// ============================================================================

#[derive(Default)]
struct AbsErrorStats {
    count: usize,
    max_abs_err: f32,
    max_err_values: (f32, f32, [f32; 3]), // (cpp, rust, input_rgb)
    total_abs_err: f64,
    // Histogram by error magnitude
    exact: usize,  // == 0.0
    tiny: usize,   // <= 1e-7 (epsilon)
    small: usize,  // <= 1e-6
    medium: usize, // <= 1e-5
    large: usize,  // <= 1e-4
    huge: usize,   // > 1e-4
}

impl AbsErrorStats {
    fn add(&mut self, cpp: f32, rust: f32, input_rgb: [f32; 3]) {
        let err = (cpp - rust).abs();
        self.count += 1;
        self.total_abs_err += err as f64;

        if err > self.max_abs_err {
            self.max_abs_err = err;
            self.max_err_values = (cpp, rust, input_rgb);
        }

        if err == 0.0 {
            self.exact += 1;
        } else if err <= 1e-7 {
            self.tiny += 1;
        } else if err <= 1e-6 {
            self.small += 1;
        } else if err <= 1e-5 {
            self.medium += 1;
        } else if err <= 1e-4 {
            self.large += 1;
        } else {
            self.huge += 1;
        }
    }

    fn mean_err(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.total_abs_err / self.count as f64
        }
    }

    fn print(&self, channel: &str) {
        let pct = |n: usize| 100.0 * n as f64 / self.count as f64;
        println!("\n  {channel} channel:");
        println!("    Max abs error: {:.2e}", self.max_abs_err);
        println!("    Mean abs error: {:.2e}", self.mean_err());
        println!(
            "    Worst case: C++={:.8} vs Rust={:.8}",
            self.max_err_values.0, self.max_err_values.1
        );
        println!(
            "    Input RGB: [{:.4}, {:.4}, {:.4}]",
            self.max_err_values.2[0], self.max_err_values.2[1], self.max_err_values.2[2]
        );
        println!("    Distribution:");
        println!(
            "      Exact (0.0):     {:6} ({:5.2}%)",
            self.exact,
            pct(self.exact)
        );
        println!(
            "      Tiny (≤1e-7):    {:6} ({:5.2}%)",
            self.tiny,
            pct(self.tiny)
        );
        println!(
            "      Small (≤1e-6):   {:6} ({:5.2}%)",
            self.small,
            pct(self.small)
        );
        println!(
            "      Medium (≤1e-5):  {:6} ({:5.2}%)",
            self.medium,
            pct(self.medium)
        );
        println!(
            "      Large (≤1e-4):   {:6} ({:5.2}%)",
            self.large,
            pct(self.large)
        );
        println!(
            "      HUGE (>1e-4):    {:6} ({:5.2}%) ⚠️",
            self.huge,
            pct(self.huge)
        );
    }
}

// ============================================================================
// XYB constants (matching C++ jpegli)
// ============================================================================

const K_M02: f32 = 0.078;
const K_M00: f32 = 0.30;
const K_M01: f32 = 1.0 - K_M02 - K_M00;
const K_M12: f32 = 0.078;
const K_M10: f32 = 0.23;
const K_M11: f32 = 1.0 - K_M12 - K_M10;
const K_M20: f32 = 0.243_422_69;
const K_M21: f32 = 0.204_767_45;
const K_M22: f32 = 1.0 - K_M20 - K_M21;
const K_B: f32 = 0.003_793_073_4;

// ============================================================================
// Fast cbrt - matching C++ jpegli's CubeRootAndAdd (3 Newton iterations)
// ============================================================================

/// C++ jpegli uses 3 Newton-Raphson iterations for cbrt
#[inline]
fn cbrtf_fast_3iter(x: f32) -> f32 {
    if x == 0.0 {
        return 0.0;
    }
    const B1: u32 = 709_958_130;
    let mut ui: u32 = x.to_bits();
    let mut hx: u32 = ui & 0x7FFF_FFFF;
    hx = hx / 3 + B1;
    ui &= 0x8000_0000;
    ui |= hx;
    let mut t: f64 = f64::from(f32::from_bits(ui));
    let xf64 = f64::from(x);
    // 3 Newton iterations (matching C++ jpegli)
    for _ in 0..3 {
        let r = t * t * t;
        t = t * (xf64 + xf64 + r) / (xf64 + r + r);
    }
    t as f32
}

/// 2 Newton iterations (ssimulacra2 style)
#[inline]
fn cbrtf_fast_2iter(x: f32) -> f32 {
    if x == 0.0 {
        return 0.0;
    }
    const B1: u32 = 709_958_130;
    let mut ui: u32 = x.to_bits();
    let mut hx: u32 = ui & 0x7FFF_FFFF;
    hx = hx / 3 + B1;
    ui &= 0x8000_0000;
    ui |= hx;
    let mut t: f64 = f64::from(f32::from_bits(ui));
    let xf64 = f64::from(x);
    // 2 Newton iterations
    let mut r = t * t * t;
    t = t * (xf64 + xf64 + r) / (xf64 + r + r);
    r = t * t * t;
    t = t * (xf64 + xf64 + r) / (xf64 + r + r);
    t as f32
}

// ============================================================================
// Rust XYB implementations to test
// ============================================================================

/// Current jpegli-rs implementation
fn rust_current(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    zenjpeg::color::xyb::linear_rgb_to_xyb(r, g, b)
}

/// Rust with 3 Newton iterations (matching C++ exactly)
fn rust_3iter(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    // FMA-style multiply-add
    let opsin_r = K_M00.mul_add(r, K_M01.mul_add(g, K_M02.mul_add(b, K_B)));
    let opsin_g = K_M10.mul_add(r, K_M11.mul_add(g, K_M12.mul_add(b, K_B)));
    let opsin_b = K_M20.mul_add(r, K_M21.mul_add(g, K_M22.mul_add(b, K_B)));

    // Clamp negatives (match C++ ZeroIfNegative)
    let opsin_r = opsin_r.max(0.0);
    let opsin_g = opsin_g.max(0.0);
    let opsin_b = opsin_b.max(0.0);

    // Fast cbrt with 3 iterations
    let neg_bias_cbrt = -cbrtf_fast_3iter(K_B);
    let cbrt_r = cbrtf_fast_3iter(opsin_r) + neg_bias_cbrt;
    let cbrt_g = cbrtf_fast_3iter(opsin_g) + neg_bias_cbrt;
    let cbrt_b = cbrtf_fast_3iter(opsin_b) + neg_bias_cbrt;

    (0.5 * (cbrt_r - cbrt_g), 0.5 * (cbrt_r + cbrt_g), cbrt_b)
}

/// Rust with 2 Newton iterations (ssimulacra2 style)
fn rust_2iter(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let opsin_r = K_M00.mul_add(r, K_M01.mul_add(g, K_M02.mul_add(b, K_B)));
    let opsin_g = K_M10.mul_add(r, K_M11.mul_add(g, K_M12.mul_add(b, K_B)));
    let opsin_b = K_M20.mul_add(r, K_M21.mul_add(g, K_M22.mul_add(b, K_B)));

    let opsin_r = opsin_r.max(0.0);
    let opsin_g = opsin_g.max(0.0);
    let opsin_b = opsin_b.max(0.0);

    let neg_bias_cbrt = -cbrtf_fast_2iter(K_B);
    let cbrt_r = cbrtf_fast_2iter(opsin_r) + neg_bias_cbrt;
    let cbrt_g = cbrtf_fast_2iter(opsin_g) + neg_bias_cbrt;
    let cbrt_b = cbrtf_fast_2iter(opsin_b) + neg_bias_cbrt;

    (0.5 * (cbrt_r - cbrt_g), 0.5 * (cbrt_r + cbrt_g), cbrt_b)
}

// ============================================================================
// C++ XYB via FFI
// ============================================================================

#[cfg(feature = "ffi-tests")]
fn cpp_linear_rgb_to_xyb(linear_rgb: &[[f32; 3]], intensity_target: f32) -> Vec<[f32; 3]> {
    use jpegli_internals_sys::jpegli_linear_to_xyb;

    let n = linear_rgb.len();
    let flat_input: Vec<f32> = linear_rgb.iter().flat_map(|p| p.iter().copied()).collect();
    let mut flat_output = vec![0.0f32; n * 3];

    unsafe {
        jpegli_linear_to_xyb(
            flat_input.as_ptr(),
            1,
            n,
            intensity_target,
            flat_output.as_mut_ptr(),
        );
    }

    flat_output.chunks(3).map(|c| [c[0], c[1], c[2]]).collect()
}

#[cfg(not(feature = "ffi-tests"))]
fn cpp_linear_rgb_to_xyb(_linear_rgb: &[[f32; 3]], _intensity_target: f32) -> Vec<[f32; 3]> {
    vec![]
}

// ============================================================================
// Test data generation
// ============================================================================

#[allow(clippy::vec_init_then_push)]
fn generate_test_linear_rgb() -> Vec<[f32; 3]> {
    let mut pixels = Vec::new();

    // Edge cases
    pixels.push([0.0, 0.0, 0.0]);
    pixels.push([1.0, 1.0, 1.0]);
    pixels.push([1.0, 0.0, 0.0]);
    pixels.push([0.0, 1.0, 0.0]);
    pixels.push([0.0, 0.0, 1.0]);
    pixels.push([0.5, 0.5, 0.5]);

    // Near-bias values
    pixels.push([0.001, 0.001, 0.001]);
    pixels.push([0.01, 0.01, 0.01]);

    // Single channel (may produce negative after matrix)
    pixels.push([0.0, 0.0, 0.5]);
    pixels.push([0.0, 0.5, 0.0]);

    // Dense grid
    for r in (0..=20).map(|x| x as f32 / 20.0) {
        for g in (0..=20).map(|x| x as f32 / 20.0) {
            for b in (0..=20).map(|x| x as f32 / 20.0) {
                pixels.push([r, g, b]);
            }
        }
    }

    // Random-ish values
    for i in 0..2000 {
        let r = ((i * 17) % 256) as f32 / 255.0;
        let g = ((i * 31) % 256) as f32 / 255.0;
        let b = ((i * 47) % 256) as f32 / 255.0;
        pixels.push([r, g, b]);
    }

    pixels
}

// ============================================================================
// Main comparison
// ============================================================================

fn main() {
    println!("=== XYB Parity Test vs C++ jpegli ===\n");

    let test_pixels = generate_test_linear_rgb();
    println!("Testing {} linear RGB values\n", test_pixels.len());

    // Get C++ reference results
    let cpp_results = cpp_linear_rgb_to_xyb(&test_pixels, 255.0);

    if cpp_results.is_empty() {
        println!("⚠️  C++ FFI not available. Run with: --features ffi-tests");
        println!("    Showing Rust implementation comparison only.\n");

        // Compare rust_current vs rust_3iter
        let mut x_diff = AbsErrorStats::default();
        let mut y_diff = AbsErrorStats::default();
        let mut b_diff = AbsErrorStats::default();

        for rgb in &test_pixels {
            let curr = rust_current(rgb[0], rgb[1], rgb[2]);
            let r3 = rust_3iter(rgb[0], rgb[1], rgb[2]);
            x_diff.add(r3.0, curr.0, *rgb);
            y_diff.add(r3.1, curr.1, *rgb);
            b_diff.add(r3.2, curr.2, *rgb);
        }

        println!("rust_current vs rust_3iter (should match if using 3 iterations):");
        x_diff.print("X");
        y_diff.print("Y");
        b_diff.print("B");
        return;
    }

    println!("C++ results available. Comparing implementations:\n");

    // Test each Rust implementation against C++
    let implementations: Vec<(&str, fn(f32, f32, f32) -> (f32, f32, f32))> = vec![
        ("rust_current (zenjpeg::color::xyb)", rust_current),
        ("rust_3iter (3 Newton iterations)", rust_3iter),
        ("rust_2iter (2 Newton iterations)", rust_2iter),
    ];

    for (name, func) in &implementations {
        println!("=== {} vs C++ ===", name);

        let mut x_stats = AbsErrorStats::default();
        let mut y_stats = AbsErrorStats::default();
        let mut b_stats = AbsErrorStats::default();

        for (i, rgb) in test_pixels.iter().enumerate() {
            let rust = func(rgb[0], rgb[1], rgb[2]);
            let cpp = cpp_results[i];

            x_stats.add(cpp[0], rust.0, *rgb);
            y_stats.add(cpp[1], rust.1, *rgb);
            b_stats.add(cpp[2], rust.2, *rgb);
        }

        x_stats.print("X");
        y_stats.print("Y");
        b_stats.print("B");

        // Summary verdict
        let max_err = x_stats
            .max_abs_err
            .max(y_stats.max_abs_err)
            .max(b_stats.max_abs_err);
        let huge_count = x_stats.huge + y_stats.huge + b_stats.huge;

        println!("\n  Summary:");
        println!("    Overall max error: {:.2e}", max_err);
        if huge_count > 0 {
            println!("    ❌ {} values with error > 1e-4", huge_count);
        } else if max_err <= 1e-6 {
            println!("    ✅ All errors ≤ 1e-6 (excellent parity)");
        } else if max_err <= 1e-5 {
            println!("    ✓ All errors ≤ 1e-5 (good parity)");
        } else {
            println!("    ⚠️ Some errors > 1e-5 (investigate)");
        }
        println!();
    }

    // Show sample values for verification
    println!("=== Sample Values (first 5) ===");
    println!(
        "{:>12} {:>12} {:>12} | {:>12} {:>12} {:>12} | {:>12} {:>12} {:>12}",
        "cpp_X", "cpp_Y", "cpp_B", "rust_X", "rust_Y", "rust_B", "err_X", "err_Y", "err_B"
    );

    for i in 0..5 {
        let rgb = &test_pixels[i];
        let cpp = cpp_results[i];
        let rust = rust_3iter(rgb[0], rgb[1], rgb[2]);
        println!("RGB({:.2},{:.2},{:.2}): {:12.8} {:12.8} {:12.8} | {:12.8} {:12.8} {:12.8} | {:12.2e} {:12.2e} {:12.2e}",
                 rgb[0], rgb[1], rgb[2],
                 cpp[0], cpp[1], cpp[2],
                 rust.0, rust.1, rust.2,
                 (cpp[0] - rust.0).abs(),
                 (cpp[1] - rust.1).abs(),
                 (cpp[2] - rust.2).abs());
    }
}
