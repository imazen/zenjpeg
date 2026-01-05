//! Brute-force XYB parity test - all 2^24 sRGB u8 values.
//!
//! Tests every possible sRGB color (16,777,216 values):
//! 1. Forward: Rust jpegli vs C++ jpegli
//! 2. Forward: Rust jpegli vs ssimulacra2 SIMD
//! 3. Roundtrip: Rust sRGB → XYB → sRGB
//!
//! Run with: cargo run --release --features "test-utils,ffi-tests" --example xyb_bruteforce_parity

use std::time::Instant;
use wide::{f32x8, f64x2};

// Import jpegli for the inverse function
use jpegli::xyb::xyb_to_srgb;

// ============================================================================
// sRGB to linear conversion (matching C++ jpegli)
// ============================================================================

#[inline]
fn srgb_u8_to_linear(v: u8) -> f32 {
    let v = v as f32 / 255.0;
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

// ============================================================================
// XYB constants (shared by jpegli-rs and ssimulacra2)
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

const OPSIN_ABSORBANCE_MATRIX: [f32; 9] = [
    K_M00, K_M01, K_M02, K_M10, K_M11, K_M12, K_M20, K_M21, K_M22,
];
const OPSIN_ABSORBANCE_BIAS: [f32; 3] = [K_B, K_B, K_B];

// ============================================================================
// Fast cbrt (shared scalar implementation)
// ============================================================================

#[inline]
fn cbrtf_fast(x: f32) -> f32 {
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
// jpegli-rs scalar implementation
// ============================================================================

#[inline]
fn rust_srgb_to_xyb(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let lr = srgb_u8_to_linear(r);
    let lg = srgb_u8_to_linear(g);
    let lb = srgb_u8_to_linear(b);

    let opsin_r = K_M00.mul_add(lr, K_M01.mul_add(lg, K_M02.mul_add(lb, K_B)));
    let opsin_g = K_M10.mul_add(lr, K_M11.mul_add(lg, K_M12.mul_add(lb, K_B)));
    let opsin_b = K_M20.mul_add(lr, K_M21.mul_add(lg, K_M22.mul_add(lb, K_B)));

    let opsin_r = opsin_r.max(0.0);
    let opsin_g = opsin_g.max(0.0);
    let opsin_b = opsin_b.max(0.0);

    let neg_bias_cbrt = -cbrtf_fast(K_B);
    let cbrt_r = cbrtf_fast(opsin_r) + neg_bias_cbrt;
    let cbrt_g = cbrtf_fast(opsin_g) + neg_bias_cbrt;
    let cbrt_b = cbrtf_fast(opsin_b) + neg_bias_cbrt;

    (0.5 * (cbrt_r - cbrt_g), 0.5 * (cbrt_r + cbrt_g), cbrt_b)
}

// ============================================================================
// ssimulacra2 SIMD implementation (copied from ssimulacra2/src/xyb_simd.rs)
// ============================================================================

#[inline]
fn initial_approx(x: f32) -> f32 {
    const B1: u32 = 709_958_130;
    let ui: u32 = x.to_bits();
    let sign = ui & 0x8000_0000;
    let hx = ui & 0x7FFF_FFFF;
    let approx = hx / 3 + B1;
    f32::from_bits(sign | approx)
}

#[inline]
fn cbrtf_x8(x: f32x8) -> f32x8 {
    let x_arr: [f32; 8] = x.into();
    let t_arr: [f32; 8] = [
        initial_approx(x_arr[0]),
        initial_approx(x_arr[1]),
        initial_approx(x_arr[2]),
        initial_approx(x_arr[3]),
        initial_approx(x_arr[4]),
        initial_approx(x_arr[5]),
        initial_approx(x_arr[6]),
        initial_approx(x_arr[7]),
    ];

    let x0 = f64x2::new([x_arr[0] as f64, x_arr[1] as f64]);
    let x1 = f64x2::new([x_arr[2] as f64, x_arr[3] as f64]);
    let x2 = f64x2::new([x_arr[4] as f64, x_arr[5] as f64]);
    let x3 = f64x2::new([x_arr[6] as f64, x_arr[7] as f64]);

    let mut t0 = f64x2::new([t_arr[0] as f64, t_arr[1] as f64]);
    let mut t1 = f64x2::new([t_arr[2] as f64, t_arr[3] as f64]);
    let mut t2 = f64x2::new([t_arr[4] as f64, t_arr[5] as f64]);
    let mut t3 = f64x2::new([t_arr[6] as f64, t_arr[7] as f64]);

    let x2_0 = x0 + x0;
    let x2_1 = x1 + x1;
    let x2_2 = x2 + x2;
    let x2_3 = x3 + x3;

    // First Newton iteration
    let r0 = t0 * t0 * t0;
    let r1 = t1 * t1 * t1;
    let r2 = t2 * t2 * t2;
    let r3 = t3 * t3 * t3;
    t0 = t0 * (x2_0 + r0) / (x0 + r0 + r0);
    t1 = t1 * (x2_1 + r1) / (x1 + r1 + r1);
    t2 = t2 * (x2_2 + r2) / (x2 + r2 + r2);
    t3 = t3 * (x2_3 + r3) / (x3 + r3 + r3);

    // Second Newton iteration
    let r0 = t0 * t0 * t0;
    let r1 = t1 * t1 * t1;
    let r2 = t2 * t2 * t2;
    let r3 = t3 * t3 * t3;
    t0 = t0 * (x2_0 + r0) / (x0 + r0 + r0);
    t1 = t1 * (x2_1 + r1) / (x1 + r1 + r1);
    t2 = t2 * (x2_2 + r2) / (x2 + r2 + r2);
    t3 = t3 * (x2_3 + r3) / (x3 + r3 + r3);

    let t0_arr: [f64; 2] = t0.into();
    let t1_arr: [f64; 2] = t1.into();
    let t2_arr: [f64; 2] = t2.into();
    let t3_arr: [f64; 2] = t3.into();
    f32x8::new([
        t0_arr[0] as f32,
        t0_arr[1] as f32,
        t1_arr[0] as f32,
        t1_arr[1] as f32,
        t2_arr[0] as f32,
        t2_arr[1] as f32,
        t3_arr[0] as f32,
        t3_arr[1] as f32,
    ])
}

/// ssimulacra2 SIMD implementation (f32x8 version) - processes batch in place
fn ssim2_linear_rgb_to_xyb_simd(input: &mut [[f32; 3]]) {
    let absorbance_bias: [f32; 3] = [
        -cbrtf_fast(OPSIN_ABSORBANCE_BIAS[0]),
        -cbrtf_fast(OPSIN_ABSORBANCE_BIAS[1]),
        -cbrtf_fast(OPSIN_ABSORBANCE_BIAS[2]),
    ];

    let m00 = f32x8::splat(OPSIN_ABSORBANCE_MATRIX[0]);
    let m01 = f32x8::splat(OPSIN_ABSORBANCE_MATRIX[1]);
    let m02 = f32x8::splat(OPSIN_ABSORBANCE_MATRIX[2]);
    let m10 = f32x8::splat(OPSIN_ABSORBANCE_MATRIX[3]);
    let m11 = f32x8::splat(OPSIN_ABSORBANCE_MATRIX[4]);
    let m12 = f32x8::splat(OPSIN_ABSORBANCE_MATRIX[5]);
    let m20 = f32x8::splat(OPSIN_ABSORBANCE_MATRIX[6]);
    let m21 = f32x8::splat(OPSIN_ABSORBANCE_MATRIX[7]);
    let m22 = f32x8::splat(OPSIN_ABSORBANCE_MATRIX[8]);
    let bias0 = f32x8::splat(OPSIN_ABSORBANCE_BIAS[0]);
    let bias1 = f32x8::splat(OPSIN_ABSORBANCE_BIAS[1]);
    let bias2 = f32x8::splat(OPSIN_ABSORBANCE_BIAS[2]);
    let zero = f32x8::splat(0.0);
    let absorb0 = f32x8::splat(absorbance_bias[0]);
    let absorb1 = f32x8::splat(absorbance_bias[1]);
    let absorb2 = f32x8::splat(absorbance_bias[2]);
    let half = f32x8::splat(0.5);

    let chunks_8 = input.len() / 8;

    for chunk_idx in 0..chunks_8 {
        let base = chunk_idx * 8;
        let mut r_arr = [0.0f32; 8];
        let mut g_arr = [0.0f32; 8];
        let mut b_arr = [0.0f32; 8];

        for i in 0..8 {
            let p = input[base + i];
            r_arr[i] = p[0];
            g_arr[i] = p[1];
            b_arr[i] = p[2];
        }

        let r = f32x8::new(r_arr);
        let g = f32x8::new(g_arr);
        let b = f32x8::new(b_arr);

        let mut mixed0 = m00.mul_add(r, m01.mul_add(g, m02.mul_add(b, bias0)));
        let mut mixed1 = m10.mul_add(r, m11.mul_add(g, m12.mul_add(b, bias1)));
        let mut mixed2 = m20.mul_add(r, m21.mul_add(g, m22.mul_add(b, bias2)));

        mixed0 = mixed0.max(zero);
        mixed1 = mixed1.max(zero);
        mixed2 = mixed2.max(zero);

        mixed0 = cbrtf_x8(mixed0) + absorb0;
        mixed1 = cbrtf_x8(mixed1) + absorb1;
        mixed2 = cbrtf_x8(mixed2) + absorb2;

        let x = half * (mixed0 - mixed1);
        let y = half * (mixed0 + mixed1);
        let b_out = mixed2;

        let x_arr: [f32; 8] = x.into();
        let y_arr: [f32; 8] = y.into();
        let b_arr: [f32; 8] = b_out.into();

        for i in 0..8 {
            input[base + i] = [x_arr[i], y_arr[i], b_arr[i]];
        }
    }

    // Scalar remainder
    let scalar_start = chunks_8 * 8;
    for pix in &mut input[scalar_start..] {
        let r = pix[0];
        let g = pix[1];
        let b = pix[2];
        let mut m0 = OPSIN_ABSORBANCE_MATRIX[0].mul_add(
            r,
            OPSIN_ABSORBANCE_MATRIX[1].mul_add(
                g,
                OPSIN_ABSORBANCE_MATRIX[2].mul_add(b, OPSIN_ABSORBANCE_BIAS[0]),
            ),
        );
        let mut m1 = OPSIN_ABSORBANCE_MATRIX[3].mul_add(
            r,
            OPSIN_ABSORBANCE_MATRIX[4].mul_add(
                g,
                OPSIN_ABSORBANCE_MATRIX[5].mul_add(b, OPSIN_ABSORBANCE_BIAS[1]),
            ),
        );
        let mut m2 = OPSIN_ABSORBANCE_MATRIX[6].mul_add(
            r,
            OPSIN_ABSORBANCE_MATRIX[7].mul_add(
                g,
                OPSIN_ABSORBANCE_MATRIX[8].mul_add(b, OPSIN_ABSORBANCE_BIAS[2]),
            ),
        );
        m0 = m0.max(0.0);
        m1 = m1.max(0.0);
        m2 = m2.max(0.0);
        m0 = cbrtf_fast(m0) + absorbance_bias[0];
        m1 = cbrtf_fast(m1) + absorbance_bias[1];
        m2 = cbrtf_fast(m2) + absorbance_bias[2];
        *pix = [0.5 * (m0 - m1), 0.5 * (m0 + m1), m2];
    }
}

// ============================================================================
// C++ XYB via FFI
// ============================================================================

#[cfg(feature = "ffi-tests")]
fn cpp_batch_srgb_to_xyb(srgb_pixels: &[[u8; 3]]) -> Vec<[f32; 3]> {
    use jpegli_internals_sys::jpegli_linear_to_xyb;

    let linear: Vec<[f32; 3]> = srgb_pixels
        .iter()
        .map(|p| {
            [
                srgb_u8_to_linear(p[0]),
                srgb_u8_to_linear(p[1]),
                srgb_u8_to_linear(p[2]),
            ]
        })
        .collect();

    let n = linear.len();
    let flat_input: Vec<f32> = linear.iter().flat_map(|p| p.iter().copied()).collect();
    let mut flat_output = vec![0.0f32; n * 3];

    unsafe {
        jpegli_linear_to_xyb(flat_input.as_ptr(), 1, n, 255.0, flat_output.as_mut_ptr());
    }

    flat_output.chunks(3).map(|c| [c[0], c[1], c[2]]).collect()
}

#[cfg(not(feature = "ffi-tests"))]
fn cpp_batch_srgb_to_xyb(_srgb_pixels: &[[u8; 3]]) -> Vec<[f32; 3]> {
    vec![]
}

// ============================================================================
// Statistics tracking
// ============================================================================

#[derive(Default)]
struct ErrorStats {
    name: String,
    total: u64,
    exact: u64,
    tiny: u64,   // <= 1e-7
    small: u64,  // <= 1e-6
    medium: u64, // <= 1e-5
    large: u64,  // <= 1e-4
    huge: u64,   // > 1e-4
    max_err: f32,
    max_err_rgb: [u8; 3],
    max_err_a: [f32; 3],
    max_err_b: [f32; 3],
    sum_err: f64,
}

impl ErrorStats {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            ..Default::default()
        }
    }

    fn add(&mut self, rgb: [u8; 3], a: [f32; 3], b: [f32; 3]) {
        let err = (a[0] - b[0])
            .abs()
            .max((a[1] - b[1]).abs())
            .max((a[2] - b[2]).abs());
        self.total += 1;
        self.sum_err += err as f64;

        if err > self.max_err {
            self.max_err = err;
            self.max_err_rgb = rgb;
            self.max_err_a = a;
            self.max_err_b = b;
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

    fn print(&self) {
        let pct = |n: u64| 100.0 * n as f64 / self.total as f64;
        println!("\n=== {} ({} colors) ===\n", self.name, self.total);
        println!("Max error: {:.2e}", self.max_err);
        println!("Mean error: {:.2e}", self.sum_err / self.total as f64);
        println!(
            "Worst RGB: ({}, {}, {})",
            self.max_err_rgb[0], self.max_err_rgb[1], self.max_err_rgb[2]
        );
        println!(
            "  A: X={:.8}, Y={:.8}, B={:.8}",
            self.max_err_a[0], self.max_err_a[1], self.max_err_a[2]
        );
        println!(
            "  B: X={:.8}, Y={:.8}, B={:.8}",
            self.max_err_b[0], self.max_err_b[1], self.max_err_b[2]
        );
        println!("\nDistribution:");
        println!(
            "  Exact (0.0):     {:>12} ({:6.3}%)",
            self.exact,
            pct(self.exact)
        );
        println!(
            "  Tiny (≤1e-7):    {:>12} ({:6.3}%)",
            self.tiny,
            pct(self.tiny)
        );
        println!(
            "  Small (≤1e-6):   {:>12} ({:6.3}%)",
            self.small,
            pct(self.small)
        );
        println!(
            "  Medium (≤1e-5):  {:>12} ({:6.3}%)",
            self.medium,
            pct(self.medium)
        );
        println!(
            "  Large (≤1e-4):   {:>12} ({:6.3}%)",
            self.large,
            pct(self.large)
        );
        println!(
            "  HUGE (>1e-4):    {:>12} ({:6.3}%) {}",
            self.huge,
            pct(self.huge),
            if self.huge > 0 { "⚠️" } else { "✅" }
        );

        if self.huge > 0 {
            println!("\n❌ {} colors have error > 1e-4", self.huge);
        } else if self.max_err <= 1e-6 {
            println!("\n✅ EXCELLENT: All within 1e-6");
        } else if self.max_err <= 1e-5 {
            println!("\n✓ GOOD: All within 1e-5");
        }
    }
}

#[derive(Default)]
struct RoundtripStats {
    total: u64,
    exact: u64,
    off_by_1: u64,
    off_by_2: u64,
    off_by_more: u64,
    max_diff: u8,
    max_diff_input: [u8; 3],
    max_diff_output: [u8; 3],
}

impl RoundtripStats {
    fn add(&mut self, input: [u8; 3], output: [u8; 3]) {
        let diff = (input[0] as i16 - output[0] as i16)
            .unsigned_abs()
            .max((input[1] as i16 - output[1] as i16).unsigned_abs())
            .max((input[2] as i16 - output[2] as i16).unsigned_abs()) as u8;
        self.total += 1;
        if diff > self.max_diff {
            self.max_diff = diff;
            self.max_diff_input = input;
            self.max_diff_output = output;
        }
        match diff {
            0 => self.exact += 1,
            1 => self.off_by_1 += 1,
            2 => self.off_by_2 += 1,
            _ => self.off_by_more += 1,
        }
    }

    fn print(&self) {
        let pct = |n: u64| 100.0 * n as f64 / self.total as f64;
        println!("\n=== Roundtrip (sRGB → XYB → sRGB) ===\n");
        println!("Max u8 diff: {}", self.max_diff);
        println!(
            "Worst: ({},{},{}) → ({},{},{})",
            self.max_diff_input[0],
            self.max_diff_input[1],
            self.max_diff_input[2],
            self.max_diff_output[0],
            self.max_diff_output[1],
            self.max_diff_output[2]
        );
        println!("\nDistribution:");
        println!(
            "  Exact (0):     {:>12} ({:6.3}%)",
            self.exact,
            pct(self.exact)
        );
        println!(
            "  Off by 1:      {:>12} ({:6.3}%)",
            self.off_by_1,
            pct(self.off_by_1)
        );
        println!(
            "  Off by 2:      {:>12} ({:6.3}%)",
            self.off_by_2,
            pct(self.off_by_2)
        );
        println!(
            "  Off by >2:     {:>12} ({:6.3}%) {}",
            self.off_by_more,
            pct(self.off_by_more),
            if self.off_by_more > 0 {
                "⚠️"
            } else {
                "✅"
            }
        );
        if self.max_diff == 0 {
            println!("\n✅ PERFECT: 100% exact roundtrip");
        } else if self.max_diff <= 1 {
            println!("\n✅ EXCELLENT: All within ±1");
        }
    }
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    println!("=== XYB Brute Force Parity Test ===");
    println!("Testing all 2^24 = 16,777,216 sRGB colors\n");

    let start = Instant::now();

    const BATCH_SIZE: usize = 65536;
    let mut rust_vs_cpp = ErrorStats::new("jpegli-rs vs C++ jpegli");
    let mut rust_vs_ssim2 = ErrorStats::new("jpegli-rs vs ssimulacra2 SIMD");
    let mut cpp_vs_ssim2 = ErrorStats::new("C++ jpegli vs ssimulacra2 SIMD");
    let mut roundtrip_stats = RoundtripStats::default();

    let mut batch_rgb: Vec<[u8; 3]> = Vec::with_capacity(BATCH_SIZE);
    let mut have_cpp = true;
    let mut last_progress = 0u8;

    for r in 0u8..=255 {
        let progress = (r as f32 / 255.0 * 100.0) as u8;
        if progress != last_progress && progress % 10 == 0 {
            println!("Progress: {}%...", progress);
            last_progress = progress;
        }

        for g in 0u8..=255 {
            for b in 0u8..=255 {
                batch_rgb.push([r, g, b]);

                if batch_rgb.len() == BATCH_SIZE {
                    // Get C++ results
                    let cpp_results = cpp_batch_srgb_to_xyb(&batch_rgb);
                    if cpp_results.is_empty() {
                        have_cpp = false;
                    }

                    // Prepare linear RGB for ssimulacra2 SIMD
                    let mut ssim2_input: Vec<[f32; 3]> = batch_rgb
                        .iter()
                        .map(|p| {
                            [
                                srgb_u8_to_linear(p[0]),
                                srgb_u8_to_linear(p[1]),
                                srgb_u8_to_linear(p[2]),
                            ]
                        })
                        .collect();
                    ssim2_linear_rgb_to_xyb_simd(&mut ssim2_input);

                    for (i, rgb) in batch_rgb.iter().enumerate() {
                        let rust_xyb = rust_srgb_to_xyb(rgb[0], rgb[1], rgb[2]);
                        let rust_arr = [rust_xyb.0, rust_xyb.1, rust_xyb.2];
                        let ssim2_arr = ssim2_input[i];

                        if have_cpp {
                            rust_vs_cpp.add(*rgb, cpp_results[i], rust_arr);
                            cpp_vs_ssim2.add(*rgb, cpp_results[i], ssim2_arr);
                        }
                        rust_vs_ssim2.add(*rgb, rust_arr, ssim2_arr);

                        let roundtrip = xyb_to_srgb(rust_xyb.0, rust_xyb.1, rust_xyb.2);
                        roundtrip_stats.add(*rgb, [roundtrip.0, roundtrip.1, roundtrip.2]);
                    }

                    batch_rgb.clear();
                }
            }
        }
    }

    // Process remaining
    if !batch_rgb.is_empty() {
        let cpp_results = cpp_batch_srgb_to_xyb(&batch_rgb);
        let mut ssim2_input: Vec<[f32; 3]> = batch_rgb
            .iter()
            .map(|p| {
                [
                    srgb_u8_to_linear(p[0]),
                    srgb_u8_to_linear(p[1]),
                    srgb_u8_to_linear(p[2]),
                ]
            })
            .collect();
        ssim2_linear_rgb_to_xyb_simd(&mut ssim2_input);

        for (i, rgb) in batch_rgb.iter().enumerate() {
            let rust_xyb = rust_srgb_to_xyb(rgb[0], rgb[1], rgb[2]);
            let rust_arr = [rust_xyb.0, rust_xyb.1, rust_xyb.2];
            let ssim2_arr = ssim2_input[i];

            if have_cpp && !cpp_results.is_empty() {
                rust_vs_cpp.add(*rgb, cpp_results[i], rust_arr);
                cpp_vs_ssim2.add(*rgb, cpp_results[i], ssim2_arr);
            }
            rust_vs_ssim2.add(*rgb, rust_arr, ssim2_arr);

            let roundtrip = xyb_to_srgb(rust_xyb.0, rust_xyb.1, rust_xyb.2);
            roundtrip_stats.add(*rgb, [roundtrip.0, roundtrip.1, roundtrip.2]);
        }
    }

    let elapsed = start.elapsed();
    println!(
        "\nCompleted in {:.2}s ({:.0} colors/sec)",
        elapsed.as_secs_f64(),
        roundtrip_stats.total as f64 / elapsed.as_secs_f64()
    );

    // Print results
    println!("\n{}", "=".repeat(60));
    println!("PART 1: FORWARD COMPARISONS");
    println!("{}", "=".repeat(60));

    if have_cpp {
        rust_vs_cpp.print();
        cpp_vs_ssim2.print();
    } else {
        println!("\n⚠️  C++ FFI not available. Run with: --features ffi-tests");
    }
    rust_vs_ssim2.print();

    println!("\n{}", "=".repeat(60));
    println!("PART 2: ROUNDTRIP");
    println!("{}", "=".repeat(60));
    roundtrip_stats.print();
}
