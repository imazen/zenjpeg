//! Butteraugli C++ parity tests.
//!
//! These tests compare the Rust butteraugli implementation against the C++
//! implementation via FFI bindings to verify mathematical parity.

use butteraugli_oxide::{compute_butteraugli, ButteraugliParams};
use jpegli_sys::{
    butteraugli_compare, butteraugli_fast_log2f, butteraugli_gamma, butteraugli_srgb_to_linear,
    BUTTERAUGLI_OK,
};
use std::fs;
use std::path::Path;

/// Load a PNG file and return RGB data.
fn load_png(path: &Path) -> Option<(Vec<u8>, usize, usize)> {
    let file = fs::File::open(path).ok()?;
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;

    let (width, height) = (info.width as usize, info.height as usize);

    let rgb = match info.color_type {
        png::ColorType::Rgb => buf[..width * height * 3].to_vec(),
        png::ColorType::Rgba => buf[..width * height * 4]
            .chunks(4)
            .flat_map(|c| [c[0], c[1], c[2]])
            .collect(),
        png::ColorType::Grayscale => buf[..width * height]
            .iter()
            .flat_map(|&g| [g, g, g])
            .collect(),
        png::ColorType::GrayscaleAlpha => buf[..width * height * 2]
            .chunks(2)
            .flat_map(|c| [c[0], c[0], c[0]])
            .collect(),
        _ => return None,
    };

    Some((rgb, width, height))
}

/// Convert sRGB u8 to linear RGB f32 (Rust implementation).
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

/// Test sRGB to linear conversion parity.
#[test]
fn test_srgb_to_linear_parity() {
    let test_values: Vec<u8> = (0..=255).collect();
    let width = 256;
    let height = 1;

    // Create RGB test data
    let srgb: Vec<u8> = test_values.iter().flat_map(|&v| [v, v, v]).collect();

    // Rust conversion
    let rust_linear = srgb_to_linear_rust(&srgb);

    // C++ conversion
    let mut cpp_linear = vec![0.0f32; srgb.len()];
    unsafe {
        butteraugli_srgb_to_linear(srgb.as_ptr(), width, height, cpp_linear.as_mut_ptr());
    }

    // Compare
    let mut max_diff = 0.0f32;
    for (i, (&rust, &cpp)) in rust_linear.iter().zip(cpp_linear.iter()).enumerate() {
        let diff = (rust - cpp).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        if diff > 1e-6 {
            println!(
                "sRGB={}: Rust={:.8} C++={:.8} diff={:.2e}",
                i / 3,
                rust,
                cpp,
                diff
            );
        }
    }

    println!("sRGB to linear max diff: {:.2e}", max_diff);
    assert!(
        max_diff < 1e-5,
        "sRGB to linear conversion differs by {:.2e}",
        max_diff
    );
}

/// Test FastLog2f parity.
#[test]
fn test_fast_log2f_parity() {
    use butteraugli_oxide::opsin::fast_log2f;

    // Test a range of values
    let test_values: Vec<f32> = (1..=1000)
        .map(|i| i as f32 * 0.1)
        .chain([0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
        .collect();

    let mut max_diff = 0.0f32;
    let mut max_diff_input = 0.0f32;

    for &v in &test_values {
        let rust_result = fast_log2f(v);
        let cpp_result = unsafe { butteraugli_fast_log2f(v) };

        let diff = (rust_result - cpp_result).abs();
        if diff > max_diff {
            max_diff = diff;
            max_diff_input = v;
        }
    }

    println!(
        "FastLog2f max diff: {:.2e} at input {:.4}",
        max_diff, max_diff_input
    );
    assert!(
        max_diff < 1e-5,
        "FastLog2f differs by {:.2e} at input {}",
        max_diff,
        max_diff_input
    );
}

/// Test Gamma function parity.
#[test]
fn test_gamma_parity() {
    use butteraugli_oxide::opsin::gamma;

    // Test a range of values
    let test_values: Vec<f32> = (0..=100)
        .map(|i| i as f32 * 0.01)
        .chain([0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
        .collect();

    let mut max_diff = 0.0f32;
    let mut max_diff_input = 0.0f32;

    for &v in &test_values {
        let rust_result = gamma(v);
        let cpp_result = unsafe { butteraugli_gamma(v) };

        let diff = (rust_result - cpp_result).abs();
        if diff > max_diff {
            max_diff = diff;
            max_diff_input = v;
        }
    }

    println!(
        "Gamma max diff: {:.2e} at input {:.4}",
        max_diff, max_diff_input
    );
    assert!(
        max_diff < 1e-4,
        "Gamma differs by {:.2e} at input {}",
        max_diff,
        max_diff_input
    );
}

/// Test butteraugli score parity with uniform gray images.
#[test]
fn test_uniform_gray_score_parity() {
    let width = 64;
    let height = 64;

    for gray_diff in [5, 10, 20, 50] {
        let gray1: u8 = 128;
        let gray2: u8 = (128_i32 + gray_diff).clamp(0, 255) as u8;

        let srgb1: Vec<u8> = vec![gray1; width * height * 3];
        let srgb2: Vec<u8> = vec![gray2; width * height * 3];

        // Rust butteraugli
        let params = ButteraugliParams::default();
        let rust_result = compute_butteraugli(&srgb1, &srgb2, width, height, &params);

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
        assert_eq!(result, BUTTERAUGLI_OK, "C++ butteraugli failed");

        let diff = (rust_result.score - cpp_score).abs();
        let relative_diff = if cpp_score > 0.001 {
            diff / cpp_score
        } else {
            diff
        };

        println!(
            "Gray {}→{}: Rust={:.4} C++={:.4} diff={:.4} ({:.1}%)",
            gray1,
            gray2,
            rust_result.score,
            cpp_score,
            diff,
            relative_diff * 100.0
        );

        // Allow up to 20% relative difference for now (to be tightened as we improve)
        assert!(
            relative_diff < 0.20 || diff < 0.5,
            "Score differs too much: Rust={:.4} C++={:.4}",
            rust_result.score,
            cpp_score
        );
    }
}

/// Test butteraugli score parity with gradient images.
#[test]
fn test_gradient_score_parity() {
    let width = 64;
    let height = 64;

    // Horizontal gradient
    let srgb1: Vec<u8> = (0..height)
        .flat_map(|_| {
            (0..width).flat_map(|x| {
                let v = ((x * 255) / (width - 1)) as u8;
                [v, v, v]
            })
        })
        .collect();

    // Same gradient but slightly brighter
    let srgb2: Vec<u8> = srgb1.iter().map(|&v| v.saturating_add(10)).collect();

    // Rust butteraugli
    let params = ButteraugliParams::default();
    let rust_result = compute_butteraugli(&srgb1, &srgb2, width, height, &params);

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
    assert_eq!(result, BUTTERAUGLI_OK, "C++ butteraugli failed");

    let diff = (rust_result.score - cpp_score).abs();
    let relative_diff = if cpp_score > 0.001 {
        diff / cpp_score
    } else {
        diff
    };

    println!(
        "Gradient: Rust={:.4} C++={:.4} diff={:.4} ({:.1}%)",
        rust_result.score,
        cpp_score,
        diff,
        relative_diff * 100.0
    );

    assert!(
        relative_diff < 0.25 || diff < 0.5,
        "Gradient score differs too much: Rust={:.4} C++={:.4}",
        rust_result.score,
        cpp_score
    );
}

/// Test butteraugli score parity with checkerboard patterns.
#[test]
fn test_checkerboard_score_parity() {
    let width = 64;
    let height = 64;

    // Checkerboard
    let srgb1: Vec<u8> = (0..height)
        .flat_map(|y| {
            (0..width).flat_map(move |x| {
                let v = if (x + y) % 2 == 0 { 50u8 } else { 200u8 };
                [v, v, v]
            })
        })
        .collect();

    // Inverse checkerboard
    let srgb2: Vec<u8> = (0..height)
        .flat_map(|y| {
            (0..width).flat_map(move |x| {
                let v = if (x + y) % 2 == 1 { 50u8 } else { 200u8 };
                [v, v, v]
            })
        })
        .collect();

    // Rust butteraugli
    let params = ButteraugliParams::default();
    let rust_result = compute_butteraugli(&srgb1, &srgb2, width, height, &params);

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
    assert_eq!(result, BUTTERAUGLI_OK, "C++ butteraugli failed");

    let diff = (rust_result.score - cpp_score).abs();
    let relative_diff = if cpp_score > 0.001 {
        diff / cpp_score
    } else {
        diff
    };

    println!(
        "Checkerboard: Rust={:.4} C++={:.4} diff={:.4} ({:.1}%)",
        rust_result.score,
        cpp_score,
        diff,
        relative_diff * 100.0
    );

    // Checkerboard patterns are more challenging due to high-frequency content
    assert!(
        relative_diff < 0.30 || diff < 1.0,
        "Checkerboard score differs too much: Rust={:.4} C++={:.4}",
        rust_result.score,
        cpp_score
    );
}

/// Test butteraugli score parity with real images.
#[test]
#[ignore] // Requires test image
fn test_real_image_score_parity() {
    let path = Path::new("/home/lilith/work/jpegli/testdata/jxl/flower/flower_small.rgb.png");
    if !path.exists() {
        eprintln!("Skipping: test image not found at {:?}", path);
        return;
    }

    let (original, width, height) = load_png(path).expect("Failed to load");

    // Encode with mozjpeg at various qualities
    for quality in [50, 70, 90] {
        let jpeg_data = encode_jpeg(&original, width as u32, height as u32, quality);
        let decoded = decode_jpeg(&jpeg_data);

        if decoded.len() != original.len() {
            continue;
        }

        // Rust butteraugli
        let params = ButteraugliParams::default();
        let rust_result = compute_butteraugli(&original, &decoded, width, height, &params);

        // C++ butteraugli
        let mut linear_orig = vec![0.0f32; original.len()];
        let mut linear_dec = vec![0.0f32; decoded.len()];
        unsafe {
            butteraugli_srgb_to_linear(original.as_ptr(), width, height, linear_orig.as_mut_ptr());
            butteraugli_srgb_to_linear(decoded.as_ptr(), width, height, linear_dec.as_mut_ptr());
        }

        let mut cpp_score = 0.0f64;
        let result = unsafe {
            butteraugli_compare(
                linear_orig.as_ptr(),
                linear_dec.as_ptr(),
                width,
                height,
                params.intensity_target,
                &mut cpp_score,
            )
        };
        assert_eq!(result, BUTTERAUGLI_OK, "C++ butteraugli failed");

        let diff = (rust_result.score - cpp_score).abs();
        let relative_diff = if cpp_score > 0.001 {
            diff / cpp_score
        } else {
            diff
        };

        println!(
            "Q{}: Rust={:.4} C++={:.4} diff={:.4} ({:.1}%)",
            quality,
            rust_result.score,
            cpp_score,
            diff,
            relative_diff * 100.0
        );

        // Real images should have reasonable parity
        assert!(
            relative_diff < 0.25 || diff < 0.5,
            "Q{} score differs too much: Rust={:.4} C++={:.4}",
            quality,
            rust_result.score,
            cpp_score
        );
    }
}

// Helper functions

fn encode_jpeg(rgb: &[u8], width: u32, height: u32, quality: u8) -> Vec<u8> {
    use std::io::Cursor;

    let mut output = Vec::new();

    let mut comp = mozjpeg::Compress::new(mozjpeg::ColorSpace::JCS_RGB);
    comp.set_size(width as usize, height as usize);
    comp.set_quality(quality as f32);

    let mut started = comp
        .start_compress(Cursor::new(&mut output))
        .expect("start compress");

    let row_stride = width as usize * 3;
    for row in rgb.chunks(row_stride) {
        started.write_scanlines(row).expect("write scanline");
    }

    started.finish().expect("finish compress");
    output
}

fn decode_jpeg(data: &[u8]) -> Vec<u8> {
    let mut decoder = jpeg_decoder::Decoder::new(data);
    decoder.decode().unwrap_or_default()
}
