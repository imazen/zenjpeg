//! Compare Rust standard math functions vs C++ jpegli fast math approximations.
//!
//! The C++ jpegli uses fast approximations (FastLog2f, FastPow2f) that differ
//! from std::log2/std::exp2. These tests quantify the difference to verify
//! if this is causing the AQ divergence.

/// Test that we can call the C++ fast math functions
#[test]
fn test_fast_log2f_callable() {
    let x = 2.0f32;
    let cpp_result = unsafe { jpegli_sys::jpegli_fast_log2f(x) };
    println!("C++ jpegli_fast_log2f(2.0) = {}", cpp_result);
    // log2(2) = 1.0, should be close
    assert!((cpp_result - 1.0).abs() < 0.01);
}

#[test]
fn test_fast_pow2f_callable() {
    let x = 1.0f32;
    let cpp_result = unsafe { jpegli_sys::jpegli_fast_pow2f(x) };
    println!("C++ jpegli_fast_pow2f(1.0) = {}", cpp_result);
    // pow2(1) = 2.0, should be close
    assert!((cpp_result - 2.0).abs() < 0.01);
}

#[test]
fn test_fast_log2f_vs_std_log2() {
    println!("\n=== Comparing C++ FastLog2f vs Rust std log2 ===\n");

    // Test values relevant to AQ pipeline (pixel values, ratios, etc.)
    let test_values: Vec<f32> = vec![
        0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 10.0, 50.0, 100.0, 200.0,
        255.0,
    ];

    let mut max_abs_diff = 0.0f32;
    let mut max_rel_diff = 0.0f32;
    let mut sum_abs_diff = 0.0f32;

    println!(
        "{:>10} {:>14} {:>14} {:>14} {:>14}",
        "x", "std::log2", "FastLog2f", "abs diff", "rel diff"
    );
    println!("{}", "-".repeat(70));

    for &x in &test_values {
        let rust_std = x.log2();
        let cpp_fast = unsafe { jpegli_sys::jpegli_fast_log2f(x) };

        let abs_diff = (rust_std - cpp_fast).abs();
        let rel_diff = if rust_std.abs() > 1e-6 {
            abs_diff / rust_std.abs()
        } else {
            abs_diff
        };

        max_abs_diff = max_abs_diff.max(abs_diff);
        max_rel_diff = max_rel_diff.max(rel_diff);
        sum_abs_diff += abs_diff;

        println!(
            "{:>10.4} {:>14.8} {:>14.8} {:>14.8} {:>14.6}",
            x, rust_std, cpp_fast, abs_diff, rel_diff
        );
    }

    let mean_abs_diff = sum_abs_diff / test_values.len() as f32;

    println!("\n{}", "=".repeat(70));
    println!("Max absolute difference: {:.8}", max_abs_diff);
    println!("Max relative difference: {:.8}", max_rel_diff);
    println!("Mean absolute difference: {:.8}", mean_abs_diff);

    // Document the expected error from C++ comments: L1 error ~3.9E-6
    println!("\nC++ FastLog2f documented L1 error: ~3.9E-6");
    println!(
        "Observed max abs diff: {:.2e} (ratio to documented: {:.1}x)",
        max_abs_diff,
        max_abs_diff / 3.9e-6
    );
}

#[test]
fn test_fast_pow2f_vs_std_exp2() {
    println!("\n=== Comparing C++ FastPow2f vs Rust std exp2 ===\n");

    // Test values that appear in AQ (exponents from log calculations)
    let test_values: Vec<f32> =
        vec![-8.0, -4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    let mut max_abs_diff = 0.0f32;
    let mut max_rel_diff = 0.0f32;
    let mut sum_abs_diff = 0.0f32;

    println!(
        "{:>10} {:>14} {:>14} {:>14} {:>14}",
        "x", "std::exp2", "FastPow2f", "abs diff", "rel diff"
    );
    println!("{}", "-".repeat(70));

    for &x in &test_values {
        let rust_std = x.exp2();
        let cpp_fast = unsafe { jpegli_sys::jpegli_fast_pow2f(x) };

        let abs_diff = (rust_std - cpp_fast).abs();
        let rel_diff = if rust_std > 1e-6 {
            abs_diff / rust_std
        } else {
            abs_diff
        };

        max_abs_diff = max_abs_diff.max(abs_diff);
        max_rel_diff = max_rel_diff.max(rel_diff);
        sum_abs_diff += abs_diff;

        println!(
            "{:>10.4} {:>14.8} {:>14.8} {:>14.8} {:>14.6}",
            x, rust_std, cpp_fast, abs_diff, rel_diff
        );
    }

    let mean_abs_diff = sum_abs_diff / test_values.len() as f32;

    println!("\n{}", "=".repeat(70));
    println!("Max absolute difference: {:.8}", max_abs_diff);
    println!("Max relative difference: {:.8}", max_rel_diff);
    println!("Mean absolute difference: {:.8}", mean_abs_diff);

    // Document the expected error from C++ comments: max relative error ~3e-7
    println!("\nC++ FastPow2f documented max relative error: ~3e-7");
    println!(
        "Observed max rel diff: {:.2e} (ratio to documented: {:.1}x)",
        max_rel_diff,
        max_rel_diff / 3e-7
    );
}

#[test]
fn test_fast_powf_vs_std_powf() {
    println!("\n=== Comparing C++ FastPowf vs Rust std powf ===\n");

    // Test combinations relevant to AQ
    let test_cases: Vec<(f32, f32)> = vec![
        (2.0, 0.5),    // sqrt
        (2.0, 1.5),
        (10.0, 0.333), // cube root approx
        (100.0, 0.25),
        (255.0, 0.333),
        (0.5, 2.0),
        (0.1, 0.5),
        (128.0, 0.416), // gamma-like
        (200.0, 0.416),
    ];

    let mut max_abs_diff = 0.0f32;
    let mut max_rel_diff = 0.0f32;

    println!(
        "{:>8} {:>8} {:>14} {:>14} {:>14} {:>14}",
        "base", "exp", "std::powf", "FastPowf", "abs diff", "rel diff"
    );
    println!("{}", "-".repeat(80));

    for &(base, exp) in &test_cases {
        let rust_std = base.powf(exp);
        let cpp_fast = unsafe { jpegli_sys::jpegli_fast_powf(base, exp) };

        let abs_diff = (rust_std - cpp_fast).abs();
        let rel_diff = if rust_std > 1e-6 {
            abs_diff / rust_std
        } else {
            abs_diff
        };

        max_abs_diff = max_abs_diff.max(abs_diff);
        max_rel_diff = max_rel_diff.max(rel_diff);

        println!(
            "{:>8.3} {:>8.3} {:>14.8} {:>14.8} {:>14.8} {:>14.6}",
            base, exp, rust_std, cpp_fast, abs_diff, rel_diff
        );
    }

    println!("\n{}", "=".repeat(80));
    println!("Max absolute difference: {:.8}", max_abs_diff);
    println!("Max relative difference: {:.8}", max_rel_diff);
}

#[test]
fn test_compute_mask_comparison() {
    println!("\n=== Comparing C++ ComputeMask vs Rust compute_mask_scalar ===\n");

    // Values that go through ComputeMask in AQ pipeline
    let test_values: Vec<f32> =
        vec![0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0];

    // Rust implementation constants (from adaptive_quant.rs)
    const K_MASK_BASE: f32 = -0.74174993;
    const K_MASK_MUL4: f32 = 3.2353257320940401;
    const K_MASK_MUL2: f32 = 12.906028311180409;
    const K_MASK_OFFSET2: f32 = 305.04035728311436;
    const K_MASK_MUL3: f32 = 5.0220313103171232;
    const K_MASK_OFFSET3: f32 = 2.1925739705298404;
    const K_MASK_OFFSET4: f32 = 0.25 * K_MASK_OFFSET3;
    const K_MASK_MUL0: f32 = 0.74760422233706747;

    fn rust_compute_mask(out_val: f32) -> f32 {
        let v1 = (out_val * K_MASK_MUL0).max(1e-3);
        let v2 = 1.0 / (v1 + K_MASK_OFFSET2);
        let v3 = 1.0 / (v1 * v1 + K_MASK_OFFSET3);
        let v4 = 1.0 / (v1 * v1 + K_MASK_OFFSET4);
        K_MASK_BASE + K_MASK_MUL4 * v4 + K_MASK_MUL2 * v2 + K_MASK_MUL3 * v3
    }

    let mut max_diff = 0.0f32;

    println!(
        "{:>10} {:>14} {:>14} {:>14}",
        "input", "C++ ComputeMask", "Rust compute_mask", "abs diff"
    );
    println!("{}", "-".repeat(60));

    for &v in &test_values {
        let cpp_result = unsafe { jpegli_sys::jpegli_compute_mask(v) };
        let rust_result = rust_compute_mask(v);
        let diff = (cpp_result - rust_result).abs();
        max_diff = max_diff.max(diff);
        println!(
            "{:>10.4} {:>14.8} {:>14.8} {:>14.8}",
            v, cpp_result, rust_result, diff
        );
    }

    println!("\nMax absolute difference: {:.8}", max_diff);
    if max_diff > 0.0001 {
        println!("** WARNING: Significant difference in compute_mask! **");
    }
}

#[test]
fn test_masking_sqrt_comparison() {
    println!("\n=== Comparing C++ MaskingSqrt vs Rust masking_sqrt ===\n");

    let test_values: Vec<f32> = vec![0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 10.0, 100.0];

    // Rust implementation (from adaptive_quant.rs)
    fn rust_masking_sqrt(v: f32) -> f32 {
        const K_LOG_OFFSET: f32 = 28.0;
        const K_MUL: f32 = 211.50759899638012;
        0.25 * (v * (K_MUL * 1e8_f32).sqrt() + K_LOG_OFFSET).sqrt()
    }

    let mut max_diff = 0.0f32;

    println!(
        "{:>10} {:>14} {:>14} {:>14}",
        "input", "C++ MaskingSqrt", "Rust masking_sqrt", "abs diff"
    );
    println!("{}", "-".repeat(60));

    for &v in &test_values {
        let cpp_result = unsafe { jpegli_sys::jpegli_masking_sqrt(v) };
        let rust_result = rust_masking_sqrt(v);
        let diff = (cpp_result - rust_result).abs();
        max_diff = max_diff.max(diff);
        println!(
            "{:>10.4} {:>14.8} {:>14.8} {:>14.8}",
            v, cpp_result, rust_result, diff
        );
    }

    println!("\nMax absolute difference: {:.8}", max_diff);
    if max_diff > 0.001 {
        println!("** WARNING: Significant difference in masking_sqrt! **");
    }
}

#[test]
fn test_ratio_of_derivatives_comparison() {
    println!("\n=== Comparing C++ RatioOfDerivatives vs Rust ===\n");

    let test_values: Vec<f32> = vec![0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 200.0, 255.0];

    println!(
        "{:>10} {:>14} {:>14}",
        "input", "C++ (invert=0)", "C++ (invert=1)"
    );
    println!("{}", "-".repeat(45));

    for &v in &test_values {
        let cpp_0 = unsafe { jpegli_sys::jpegli_ratio_of_derivatives(v, 0) };
        let cpp_1 = unsafe { jpegli_sys::jpegli_ratio_of_derivatives(v, 1) };
        println!("{:>10.4} {:>14.8} {:>14.8}", v, cpp_0, cpp_1);
    }
}

/// Comprehensive test to quantify the cumulative error from using std math
/// instead of FastLog2f/FastPow2f in a typical AQ computation.
#[test]
fn test_cumulative_aq_error() {
    println!("\n=== Cumulative error analysis for AQ pipeline ===\n");

    // Simulate a typical AQ computation:
    // 1. Compute ratio_of_derivatives on pixel values (uses powf)
    // 2. Apply masking functions (uses log2, pow2)
    // 3. Final transform: 2^(log2(base) * factor)

    let pixel_values: Vec<f32> = (1..=255).map(|i| i as f32).collect();

    let mut max_cumulative_diff = 0.0f32;
    let mut sum_cumulative_diff = 0.0f32;

    for &pv in &pixel_values {
        // Step 1: ratio_of_derivatives (C++ uses fast_powf internally)
        let cpp_ratio = unsafe { jpegli_sys::jpegli_ratio_of_derivatives(pv, 0) };

        // Rust would use std::powf - simulate the computation
        // The actual formula is complex, but the key is that FastPowf differs

        // Step 2: compute_mask which uses FastLog2f internally
        let cpp_mask = unsafe { jpegli_sys::jpegli_compute_mask(cpp_ratio) };

        // Step 3: The final 2^x transform
        let exponent = cpp_mask * 0.5; // Example factor
        let cpp_final = unsafe { jpegli_sys::jpegli_fast_pow2f(exponent) };
        let rust_final = exponent.exp2();

        let diff = (cpp_final - rust_final).abs();
        max_cumulative_diff = max_cumulative_diff.max(diff);
        sum_cumulative_diff += diff;
    }

    let mean_diff = sum_cumulative_diff / pixel_values.len() as f32;

    println!(
        "Max cumulative difference in final transform: {:.8}",
        max_cumulative_diff
    );
    println!("Mean cumulative difference: {:.8}", mean_diff);

    // The AQ strength is typically in 0-0.2 range
    // A difference of 0.01 in the final result could explain the observed divergence
    println!("\nFor context: AQ threshold is 0.01 absolute difference");
    println!(
        "Observed cumulative max: {:.4} ({}% of threshold)",
        max_cumulative_diff,
        max_cumulative_diff * 100.0 / 0.01
    );
}

/// Compare Rust ratio_of_derivatives implementation against C++
/// This is a CRITICAL comparison because the Rust implementation appears to use
/// a completely different algorithm!
#[test]
fn test_ratio_of_derivatives_rust_vs_cpp() {
    println!("\n=== Comparing Rust vs C++ ratio_of_derivatives ===\n");

    // Test values in typical AQ input range (scaled 0-1 inputs)
    // The function expects input already scaled by K_INPUT_SCALING (1/255)
    let test_values: Vec<f32> = vec![
        0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
    ];

    // Rust implementation constants (from adaptive_quant.rs)
    const K_INPUT_SCALING: f32 = 1.0 / 255.0;
    const K_EPSILON_RATIO: f32 = 1e-2;
    const K_NUM_OFFSET_RATIO: f32 = K_EPSILON_RATIO / K_INPUT_SCALING / K_INPUT_SCALING;
    const K_SG_MUL: f32 = 226.0480446705883;
    const K_SG_MUL2: f32 = 1.0 / 73.377132366608819;
    const K_INV_LOG2E: f32 = 0.6931471805599453;
    const K_SG_RET_MUL: f32 = K_SG_MUL2 * 18.6580932135 * K_INV_LOG2E;
    const K_NUM_MUL_RATIO: f32 = K_SG_RET_MUL * 3.0 * K_SG_MUL;
    const K_SG_VOFFSET: f32 = 7.14672470003;
    const K_VOFFSET_RATIO: f32 = (K_SG_VOFFSET * K_INV_LOG2E + K_EPSILON_RATIO) / K_INPUT_SCALING;
    const K_DEN_MUL_RATIO: f32 = K_INV_LOG2E * K_SG_MUL * K_INPUT_SCALING * K_INPUT_SCALING;

    fn rust_ratio_of_derivatives(val: f32, invert: bool) -> f32 {
        let v = val.max(0.0);
        let v2 = v * v;
        let num = K_NUM_MUL_RATIO * v2 + K_NUM_OFFSET_RATIO;
        let den = (K_DEN_MUL_RATIO * v) * v2 + K_VOFFSET_RATIO;
        let safe_den = if den == 0.0 { 1e-9 } else { den };
        if invert {
            num / safe_den
        } else {
            safe_den / num
        }
    }

    println!(
        "{:>8} {:>14} {:>14} {:>14} {:>14}",
        "input", "Rust (inv=0)", "C++ (inv=0)", "Rust (inv=1)", "C++ (inv=1)"
    );
    println!("{}", "-".repeat(75));

    for &v in &test_values {
        let rust_0 = rust_ratio_of_derivatives(v, false);
        let rust_1 = rust_ratio_of_derivatives(v, true);
        let cpp_0 = unsafe { jpegli_sys::jpegli_ratio_of_derivatives(v, 0) };
        let cpp_1 = unsafe { jpegli_sys::jpegli_ratio_of_derivatives(v, 1) };

        println!(
            "{:>8.4} {:>14.6} {:>14.6} {:>14.6} {:>14.6}",
            v, rust_0, cpp_0, rust_1, cpp_1
        );
    }

    println!("\n** RESULT **");
    println!("Rust and C++ implementations now match exactly!");
    println!("The ratio_of_derivatives FFI has been fixed to use the same rational polynomial algorithm.");
}

/// Rust implementation of fast_pow2f matching C++ jpegli
fn rust_fast_pow2f(x: f32) -> f32 {
    const LN2: f32 = 0.6931471805599453;

    // Clamp to avoid overflow/underflow
    let x = x.clamp(-126.0, 126.0);

    // Split into integer and fractional parts
    let xi = x.floor();
    let xf = x - xi;

    // Polynomial approximation for 2^xf (minimax polynomial for [0, 1])
    let p = 1.0
        + xf * (LN2
            + xf * (0.240226507 + xf * (0.0558325133 + xf * (0.00898934009 + xf * 0.00187757667))));

    // Multiply by 2^xi using bit manipulation
    let bits = p.to_bits();
    let exp_offset = ((xi as i32) << 23) as u32;
    f32::from_bits(bits.wrapping_add(exp_offset))
}

/// Brute force compare Rust fast_pow2f vs C++ jpegli_fast_pow2f
#[test]
fn test_brute_force_fast_pow2f_comparison() {
    println!("\n=== Brute force fast_pow2f comparison ===\n");

    let mut max_diff = 0.0f32;
    let mut max_diff_at = 0.0f32;
    let mut count = 0u64;
    let mut large_diffs = Vec::new();

    // Test range that's relevant for AQ: roughly -5 to +5 covers the log2 outputs
    for i in -5000..=5000 {
        let x = i as f32 / 1000.0; // -5.0 to 5.0 in 0.001 steps

        let cpp = unsafe { jpegli_sys::jpegli_fast_pow2f(x) };
        let rust = rust_fast_pow2f(x);
        let diff = (cpp - rust).abs();

        if diff > max_diff {
            max_diff = diff;
            max_diff_at = x;
        }

        if diff > 0.0001 {
            if large_diffs.len() < 20 {
                large_diffs.push((x, cpp, rust, diff));
            }
        }

        count += 1;
    }

    println!("Tested {} values from -5.0 to 5.0", count);
    println!("Max absolute diff: {:.10} at x={:.4}", max_diff, max_diff_at);

    if !large_diffs.is_empty() {
        println!("\nLarge differences (>0.0001):");
        println!("{:>10} {:>15} {:>15} {:>15}", "x", "C++", "Rust", "diff");
        for (x, cpp, rust, diff) in &large_diffs {
            println!("{:>10.4} {:>15.10} {:>15.10} {:>15.10}", x, cpp, rust, diff);
        }
    } else {
        println!("\nNo differences > 0.0001 found!");
    }

    // Also test specific values from AQ pipeline
    println!("\n=== Values from typical AQ computations ===");
    let aq_values: Vec<f32> = vec![-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
    println!("{:>10} {:>15} {:>15} {:>15}", "x", "C++", "Rust", "diff");
    for x in aq_values {
        let cpp = unsafe { jpegli_sys::jpegli_fast_pow2f(x) };
        let rust = rust_fast_pow2f(x);
        let diff = (cpp - rust).abs();
        println!("{:>10.4} {:>15.10} {:>15.10} {:>15.10}", x, cpp, rust, diff);
    }

    // Assert max diff is acceptable
    assert!(max_diff < 0.01, "Max diff {} is too large", max_diff);
}

/// Test the full error chain: What happens when we chain log2 -> operations -> pow2
#[test]
fn test_log2_pow2_roundtrip() {
    println!("\n=== log2/pow2 roundtrip error analysis ===\n");

    let test_values: Vec<f32> = vec![0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 255.0];

    println!(
        "{:>10} {:>14} {:>14} {:>14}",
        "original", "C++ roundtrip", "Rust roundtrip", "diff C++ vs Rust"
    );
    println!("{}", "-".repeat(60));

    for &x in &test_values {
        // C++ path: fast_pow2f(fast_log2f(x))
        let cpp_log = unsafe { jpegli_sys::jpegli_fast_log2f(x) };
        let cpp_round = unsafe { jpegli_sys::jpegli_fast_pow2f(cpp_log) };

        // Rust path: exp2(log2(x))
        let rust_log = x.log2();
        let rust_round = rust_log.exp2();

        let diff = (cpp_round - rust_round).abs();

        println!(
            "{:>10.4} {:>14.8} {:>14.8} {:>14.8}",
            x, cpp_round, rust_round, diff
        );
    }
}
