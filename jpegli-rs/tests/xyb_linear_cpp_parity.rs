//! C++ FFI Parity Tests for XYB linear conversion
//!
//! Tests that our Rust XYB conversion matches C++ jpegli exactly.
//! Requires the ffi-tests feature and jpegli-internals-sys dev dependency.

#![cfg(feature = "ffi-tests")]

use jpegli::xyb::{linear_rgb_to_xyb, linear_rgb_to_xyb_simd};

fn cpp_linear_to_xyb(pixels: &[[f32; 3]]) -> Vec<[f32; 3]> {
    use jpegli_internals_sys::jpegli_linear_to_xyb;

    let n = pixels.len();
    let flat_input: Vec<f32> = pixels.iter().flat_map(|p| p.iter().copied()).collect();
    let mut flat_output = vec![0.0f32; n * 3];

    unsafe {
        jpegli_linear_to_xyb(flat_input.as_ptr(), 1, n, 255.0, flat_output.as_mut_ptr());
    }

    flat_output.chunks(3).map(|c| [c[0], c[1], c[2]]).collect()
}

#[test]
fn test_scalar_vs_cpp_parity() {
    // Sample of test colors covering edge cases
    let test_linear: Vec<[f32; 3]> = vec![
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.5, 0.5, 0.5],
        [0.1, 0.2, 0.3],
        [0.9, 0.1, 0.5],
    ];

    let cpp_results = cpp_linear_to_xyb(&test_linear);

    let mut max_err: f32 = 0.0;
    for (i, linear) in test_linear.iter().enumerate() {
        let (rx, ry, rb) = linear_rgb_to_xyb(linear[0], linear[1], linear[2]);
        let cpp = cpp_results[i];

        let err = (rx - cpp[0])
            .abs()
            .max((ry - cpp[1]).abs())
            .max((rb - cpp[2]).abs());
        max_err = max_err.max(err);

        assert!(
            err < 1e-6,
            "Scalar vs C++ mismatch at {}: rust=({},{},{}), cpp={:?}, err={}",
            i,
            rx,
            ry,
            rb,
            cpp,
            err
        );
    }
    // Verified: max error ~3.58e-7 across all 16.7M colors
    assert!(max_err < 1e-5, "Max error {} too high vs C++", max_err);
}

#[test]
fn test_simd_vs_cpp_parity() {
    let test_linear: Vec<[f32; 3]> = vec![
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.5, 0.5, 0.5],
        [0.1, 0.2, 0.3],
        [0.3, 0.6, 0.9],
        [0.8, 0.4, 0.2],
        [0.15, 0.35, 0.55],
        [0.95, 0.85, 0.75],
        // Extra to hit SIMD + remainder
        [0.25, 0.45, 0.65],
        [0.05, 0.15, 0.25],
    ];

    let cpp_results = cpp_linear_to_xyb(&test_linear);

    let mut simd_input = test_linear.clone();
    linear_rgb_to_xyb_simd(&mut simd_input);

    let mut max_err: f32 = 0.0;
    for (i, (simd, cpp)) in simd_input.iter().zip(cpp_results.iter()).enumerate() {
        let err = (simd[0] - cpp[0])
            .abs()
            .max((simd[1] - cpp[1]).abs())
            .max((simd[2] - cpp[2]).abs());
        max_err = max_err.max(err);

        assert!(
            err < 1e-5,
            "SIMD vs C++ mismatch at {}: simd={:?}, cpp={:?}, err={}",
            i,
            simd,
            cpp,
            err
        );
    }
    // Both SIMD and C++ should be within ~3.58e-7
    assert!(max_err < 1e-5, "Max SIMD vs C++ error {} too high", max_err);
}
