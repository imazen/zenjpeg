//! Property-based fuzzing tests for butteraugli C++ parity.
//!
//! These tests use proptest to generate random inputs and verify that
//! the Rust implementation matches the C++ implementation.

use butteraugli_oxide::opsin::{fast_log2f, gamma};
use butteraugli_oxide::{compute_butteraugli, ButteraugliParams};
use jpegli_sys::{
    butteraugli_compare, butteraugli_fast_log2f, butteraugli_gamma, butteraugli_srgb_to_linear,
    BUTTERAUGLI_OK,
};
use proptest::prelude::*;

// ============================================================================
// FastLog2f parity tests
// ============================================================================

proptest! {
    /// Test that FastLog2f matches C++ for arbitrary positive values.
    #[test]
    fn fuzz_fast_log2f_parity(x in 0.001f32..1000.0f32) {
        let rust_result = fast_log2f(x);
        let cpp_result = unsafe { butteraugli_fast_log2f(x) };

        let diff = (rust_result - cpp_result).abs();

        // Allow very small differences due to floating-point
        prop_assert!(
            diff < 1e-5,
            "FastLog2f mismatch at x={}: Rust={:.8} C++={:.8} diff={:.2e}",
            x, rust_result, cpp_result, diff
        );
    }

    /// Test FastLog2f at power-of-2 boundaries (regression for discontinuity bug).
    #[test]
    fn fuzz_fast_log2f_power_of_two_boundaries(exp in -10i32..10i32) {
        let base = 2.0f32.powi(exp);

        // Test slightly below and above power of 2
        for offset in [-0.01, -0.001, -0.0001, 0.0, 0.0001, 0.001, 0.01] {
            let x = base + offset * base;
            if x <= 0.0 { continue; }

            let rust_result = fast_log2f(x);
            let cpp_result = unsafe { butteraugli_fast_log2f(x) };

            let diff = (rust_result - cpp_result).abs();
            prop_assert!(
                diff < 1e-5,
                "FastLog2f power-of-2 boundary mismatch at x={}: Rust={:.8} C++={:.8}",
                x, rust_result, cpp_result
            );
        }
    }
}

// ============================================================================
// Gamma function parity tests
// ============================================================================

proptest! {
    /// Test that Gamma function matches C++ for arbitrary values.
    #[test]
    fn fuzz_gamma_parity(v in 0.0f32..100.0f32) {
        let rust_result = gamma(v);
        let cpp_result = unsafe { butteraugli_gamma(v) };

        let diff = (rust_result - cpp_result).abs();

        prop_assert!(
            diff < 1e-4,
            "Gamma mismatch at v={}: Rust={:.8} C++={:.8} diff={:.2e}",
            v, rust_result, cpp_result, diff
        );
    }
}

// ============================================================================
// sRGB to linear conversion parity tests
// ============================================================================

fn srgb_to_linear_rust(v: u8) -> f32 {
    let x = v as f32 / 255.0;
    if x <= 0.04045 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}

proptest! {
    /// Test sRGB to linear conversion matches C++ for all byte values.
    #[test]
    fn fuzz_srgb_to_linear_parity(v in 0u8..=255u8) {
        let rust_result = srgb_to_linear_rust(v);

        // Create small test image
        let srgb = [v, v, v];
        let mut linear = [0.0f32; 3];
        unsafe {
            butteraugli_srgb_to_linear(srgb.as_ptr(), 1, 1, linear.as_mut_ptr());
        }
        let cpp_result = linear[0]; // All channels same for grayscale

        let diff = (rust_result - cpp_result).abs();
        prop_assert!(
            diff < 1e-6,
            "sRGB to linear mismatch at v={}: Rust={:.8} C++={:.8}",
            v, rust_result, cpp_result
        );
    }
}

// ============================================================================
// Butteraugli score parity tests for small images
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(20))]

    /// Test butteraugli score parity for uniform color images.
    #[test]
    fn fuzz_uniform_image_score_parity(
        gray1 in 50u8..200u8,
        gray_diff in 1u8..50u8,
        size in 16usize..=64usize,
    ) {
        let gray2 = gray1.saturating_add(gray_diff);
        let width = size;
        let height = size;

        let srgb1: Vec<u8> = vec![gray1; width * height * 3];
        let srgb2: Vec<u8> = vec![gray2; width * height * 3];

        // Rust butteraugli
        let params = ButteraugliParams::default();
        let rust_result = compute_butteraugli(&srgb1, &srgb2, width, height, &params).expect("valid input");

        // C++ butteraugli
        let mut linear1 = vec![0.0f32; srgb1.len()];
        let mut linear2 = vec![0.0f32; srgb2.len()];
        unsafe {
            butteraugli_srgb_to_linear(srgb1.as_ptr(), width, height, linear1.as_mut_ptr());
            butteraugli_srgb_to_linear(srgb2.as_ptr(), width, height, linear2.as_mut_ptr());
        }

        let mut cpp_score = 0.0f64;
        let result = unsafe {
            butteraugli_compare(
                linear1.as_ptr(),
                linear2.as_ptr(),
                width,
                height,
                params.intensity_target,
                &mut cpp_score,
            )
        };
        prop_assert_eq!(result, BUTTERAUGLI_OK, "C++ butteraugli failed");

        let diff = (rust_result.score - cpp_score).abs();
        let relative_diff = if cpp_score > 0.001 {
            diff / cpp_score
        } else {
            diff
        };

        // Allow up to 25% relative difference (to be tightened)
        prop_assert!(
            relative_diff < 0.25 || diff < 0.5,
            "Uniform image score mismatch: gray1={} gray2={} size={}x{} Rust={:.4} C++={:.4} diff={:.1}%",
            gray1, gray2, width, height, rust_result.score, cpp_score, relative_diff * 100.0
        );
    }

    /// Test butteraugli score parity for gradient images.
    #[test]
    fn fuzz_gradient_image_score_parity(
        brightness_shift in 5u8..30u8,
        size in 32usize..=64usize,
    ) {
        let width = size;
        let height = size;

        // Horizontal gradient
        let srgb1: Vec<u8> = (0..height)
            .flat_map(|_| {
                (0..width).flat_map(|x| {
                    let v = ((x * 255) / (width - 1)) as u8;
                    [v, v, v]
                })
            })
            .collect();

        // Same gradient but shifted
        let srgb2: Vec<u8> = srgb1
            .iter()
            .map(|&v| v.saturating_add(brightness_shift))
            .collect();

        // Rust butteraugli
        let params = ButteraugliParams::default();
        let rust_result = compute_butteraugli(&srgb1, &srgb2, width, height, &params).expect("valid input");

        // C++ butteraugli
        let mut linear1 = vec![0.0f32; srgb1.len()];
        let mut linear2 = vec![0.0f32; srgb2.len()];
        unsafe {
            butteraugli_srgb_to_linear(srgb1.as_ptr(), width, height, linear1.as_mut_ptr());
            butteraugli_srgb_to_linear(srgb2.as_ptr(), width, height, linear2.as_mut_ptr());
        }

        let mut cpp_score = 0.0f64;
        let result = unsafe {
            butteraugli_compare(
                linear1.as_ptr(),
                linear2.as_ptr(),
                width,
                height,
                params.intensity_target,
                &mut cpp_score,
            )
        };
        prop_assert_eq!(result, BUTTERAUGLI_OK, "C++ butteraugli failed");

        let diff = (rust_result.score - cpp_score).abs();
        let relative_diff = if cpp_score > 0.001 {
            diff / cpp_score
        } else {
            diff
        };

        prop_assert!(
            relative_diff < 0.30 || diff < 1.0,
            "Gradient image score mismatch: shift={} size={}x{} Rust={:.4} C++={:.4} diff={:.1}%",
            brightness_shift, width, height, rust_result.score, cpp_score, relative_diff * 100.0
        );
    }

    /// Test butteraugli score parity for noise images.
    #[test]
    fn fuzz_noisy_image_score_parity(
        seed in 0u64..1000u64,
        noise_level in 5u8..30u8,
        size in 32usize..=64usize,
    ) {
        let width = size;
        let height = size;

        // Generate pseudo-random base image
        let mut rng_state = seed;
        let lcg_next = |state: &mut u64| -> u8 {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (*state >> 33) as u8
        };

        let srgb1: Vec<u8> = (0..width * height * 3)
            .map(|_| lcg_next(&mut rng_state).saturating_add(64).min(192))
            .collect();

        // Add noise to create second image
        let mut rng_state2 = seed.wrapping_add(1);
        let srgb2: Vec<u8> = srgb1
            .iter()
            .map(|&v| {
                let noise = (lcg_next(&mut rng_state2) as i16 - 128) * noise_level as i16 / 128;
                (v as i16 + noise).clamp(0, 255) as u8
            })
            .collect();

        // Rust butteraugli
        let params = ButteraugliParams::default();
        let rust_result = compute_butteraugli(&srgb1, &srgb2, width, height, &params).expect("valid input");

        // C++ butteraugli
        let mut linear1 = vec![0.0f32; srgb1.len()];
        let mut linear2 = vec![0.0f32; srgb2.len()];
        unsafe {
            butteraugli_srgb_to_linear(srgb1.as_ptr(), width, height, linear1.as_mut_ptr());
            butteraugli_srgb_to_linear(srgb2.as_ptr(), width, height, linear2.as_mut_ptr());
        }

        let mut cpp_score = 0.0f64;
        let result = unsafe {
            butteraugli_compare(
                linear1.as_ptr(),
                linear2.as_ptr(),
                width,
                height,
                params.intensity_target,
                &mut cpp_score,
            )
        };
        prop_assert_eq!(result, BUTTERAUGLI_OK, "C++ butteraugli failed");

        let diff = (rust_result.score - cpp_score).abs();
        let relative_diff = if cpp_score > 0.001 {
            diff / cpp_score
        } else {
            diff
        };

        prop_assert!(
            relative_diff < 0.35 || diff < 1.5,
            "Noisy image score mismatch: seed={} noise={} size={}x{} Rust={:.4} C++={:.4} diff={:.1}%",
            seed, noise_level, width, height, rust_result.score, cpp_score, relative_diff * 100.0
        );
    }
}
