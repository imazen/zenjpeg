//! Compare zune-based IDCT vs libjpeg-compatible IDCT to quantify differences.
//!
//! Run: cargo test --release -p zenjpeg --test idct_comparison -- --nocapture

use zenjpeg::decode::idct_int::{idct_int, idct_int_libjpeg};

/// Generate dequantized coefficients at a given magnitude.
fn make_test_coeffs(seed: u64, magnitude: i32) -> [i32; 64] {
    let mut coeffs = [0i32; 64];
    let mut state = seed;
    for c in coeffs.iter_mut() {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *c = ((state >> 33) as i32 % (magnitude * 2 + 1)) - magnitude;
    }
    coeffs
}

#[test]
fn compare_idct_algorithms() {
    // Test at different coefficient magnitudes to find the overflow threshold
    println!("\n=== IDCT comparison: idct_int (zune) vs idct_int_libjpeg ===\n");
    println!("{:<12} {:>8} {:>10}", "magnitude", "max_diff", "mean_diff");
    println!("{}", "-".repeat(32));

    for &mag in &[512, 1024, 2048, 3000, 4000, 5000, 6000, 7000, 8192] {
        let mut max_diff = 0i32;
        let mut sum_diff = 0u64;
        let mut total = 0u64;

        for seed in 0..5000u64 {
            let mut coeffs_a = make_test_coeffs(seed, mag);
            let mut coeffs_b = coeffs_a;

            let mut out_a = [0i16; 64];
            let mut out_b = [0i16; 64];

            idct_int(&mut coeffs_a, &mut out_a, 8);
            idct_int_libjpeg(&mut coeffs_b, &mut out_b, 8);

            for i in 0..64 {
                let diff = (out_a[i] as i32 - out_b[i] as i32).abs();
                max_diff = max_diff.max(diff);
                sum_diff += diff as u64;
                total += 1;
            }
        }

        let mean_diff = sum_diff as f64 / total as f64;
        println!("{mag:<12} {max_diff:>8} {mean_diff:>10.4}");
    }

    // Also check: what is the actual dequantized coefficient range for these files?
    println!("\n=== Note ===");
    println!("JPEG standard requires IDCT conformance for coefficients in [-2048, 2047].");
    println!("The zune IDCT uses 12-bit fixed-point (4096 scale), which overflows for larger values.");
    println!("libjpeg-turbo uses 13-bit fixed-point (8192 scale), which handles wider range.");
    println!("Wide-gamut images with high chroma saturation can produce larger dequantized values.");
}

