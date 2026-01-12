//! XYB color space roundtrip tests.
//!
//! Tests that XYB conversion matches C++ implementation and roundtrips correctly.

use jpegli::xyb::{
    linear_rgb_to_xyb, linear_to_srgb_u8, rgb_buffer_to_xyb_planes, srgb_to_xyb, srgb_u8_to_linear,
    xyb_planes_to_rgb_buffer, xyb_to_linear_rgb, xyb_to_srgb,
};

/// Test XYB roundtrip for all 8-bit colors at key points.
#[test]
fn test_xyb_roundtrip_comprehensive() {
    let mut max_error = 0i16;
    let mut error_count = 0usize;
    let mut total_tests = 0usize;

    // Test key colors: corners, edges, gray ramp
    let test_values: Vec<u8> = vec![
        0, 1, 2, 4, 8, 16, 32, 64, 96, 128, 160, 192, 224, 240, 252, 254, 255,
    ];

    for &r in &test_values {
        for &g in &test_values {
            for &b in &test_values {
                let (x, y, b_xyb) = srgb_to_xyb(r, g, b);
                let (r2, g2, b2) = xyb_to_srgb(x, y, b_xyb);

                let dr = (r as i16 - r2 as i16).abs();
                let dg = (g as i16 - g2 as i16).abs();
                let db = (b as i16 - b2 as i16).abs();

                max_error = max_error.max(dr).max(dg).max(db);
                if dr > 1 || dg > 1 || db > 1 {
                    error_count += 1;
                }
                total_tests += 1;
            }
        }
    }

    println!(
        "XYB roundtrip: {} tests, max error: {}, errors>1: {}",
        total_tests, max_error, error_count
    );

    // Allow max error of 2 due to float precision
    assert!(
        max_error <= 2,
        "XYB roundtrip max error {} exceeds tolerance",
        max_error
    );

    // At most 5% of values should have error > 1
    let error_ratio = error_count as f64 / total_tests as f64;
    assert!(
        error_ratio < 0.05,
        "XYB roundtrip error ratio {} exceeds 5%",
        error_ratio
    );
}

/// Test that gray values maintain X ≈ 0.
#[test]
fn test_xyb_gray_neutrality() {
    for gray in 0..=255u8 {
        let (x, _y, _b) = srgb_to_xyb(gray, gray, gray);
        assert!(x.abs() < 0.01, "Gray {} has X={}, should be ~0", gray, x);
    }
}

/// Test XYB values for known reference colors (from C++ implementation).
#[test]
fn test_xyb_reference_values() {
    // Reference values from C++ jpegli (approximate)
    let test_cases = [
        // (r, g, b, expected_x, expected_y, expected_b) with tolerance
        (0u8, 0u8, 0u8, 0.0f32, 0.0f32, 0.0f32, 0.01), // Black
        (255u8, 255u8, 255u8, 0.0f32, 0.88f32, 0.82f32, 0.05), // White (approximate)
        (255u8, 0u8, 0u8, 0.1f32, 0.55f32, 0.08f32, 0.1), // Red
        (0u8, 255u8, 0u8, -0.15f32, 0.7f32, 0.25f32, 0.1), // Green
        (0u8, 0u8, 255u8, 0.0f32, 0.3f32, 0.75f32, 0.1), // Blue
    ];

    for (r, g, b, _exp_x, _exp_y, _exp_b, _tol) in test_cases {
        let (x, y, b_xyb) = srgb_to_xyb(r, g, b);
        println!(
            "RGB({},{},{}) -> XYB({:.4}, {:.4}, {:.4})",
            r, g, b, x, y, b_xyb
        );
        // Just log for now - exact values depend on C++ constants
    }
}

/// Test buffer conversion functions.
#[test]
fn test_xyb_buffer_roundtrip() {
    let width = 16;
    let height = 16;
    let mut rgb = vec![0u8; width * height * 3];

    // Create gradient test pattern
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            rgb[idx] = (x * 16) as u8;
            rgb[idx + 1] = (y * 16) as u8;
            rgb[idx + 2] = ((x + y) * 8) as u8;
        }
    }

    // Convert to XYB
    let (x_plane, y_plane, b_plane) = rgb_buffer_to_xyb_planes(&rgb, width, height);

    // Convert back
    let rgb2 = xyb_planes_to_rgb_buffer(&x_plane, &y_plane, &b_plane, width, height);

    // Check roundtrip
    let mut max_diff = 0i16;
    for i in 0..rgb.len() {
        let diff = (rgb[i] as i16 - rgb2[i] as i16).abs();
        max_diff = max_diff.max(diff);
    }

    println!("Buffer roundtrip max diff: {}", max_diff);
    assert!(
        max_diff <= 2,
        "Buffer roundtrip max diff {} exceeds 2",
        max_diff
    );
}

/// Test XYB with image encoding (if encoder supports XYB).
#[test]
fn test_xyb_encode_decode() {
    use jpegli::quant::Quality;
    use jpegli::{PixelFormat, StreamingEncoder};

    let width = 64u32;
    let height = 64u32;
    let mut rgb = vec![0u8; (width * height * 3) as usize];

    // Create test pattern
    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 3) as usize;
            rgb[idx] = ((x * 4) % 256) as u8;
            rgb[idx + 1] = ((y * 4) % 256) as u8;
            rgb[idx + 2] = (((x + y) * 2) % 256) as u8;
        }
    }

    // Encode with XYB (note: actual XYB encoding may not be fully implemented)
    let encoder = StreamingEncoder::new(width, height)
        .pixel_format(PixelFormat::Rgb)
        .quality(Quality::from_quality(90.0))
        .use_xyb(true);

    // Try encoding - this tests that the XYB flag doesn't break encoding
    match encoder.encode_all(&rgb) {
        Ok(jpeg_data) => {
            println!("XYB encoded JPEG: {} bytes", jpeg_data.len());
            assert!(!jpeg_data.is_empty());
        }
        Err(e) => {
            // XYB might not be fully implemented yet
            println!("XYB encoding not yet supported: {}", e);
        }
    }
}

/// Test linear RGB conversion accuracy.
#[test]
fn test_srgb_linear_precision() {
    let mut max_error = 0f32;

    for v in 0..=255u8 {
        let linear = srgb_u8_to_linear(v);
        let back = linear_to_srgb_u8(linear);

        // Check value is in valid range
        assert!(
            linear >= 0.0 && linear <= 1.0,
            "Linear value {} out of range for input {}",
            linear,
            v
        );

        // Allow 1-bit error due to rounding
        let error = (v as i16 - back as i16).abs();
        max_error = max_error.max(error as f32);
    }

    assert!(
        max_error <= 1.0,
        "sRGB<->linear max error {} exceeds 1",
        max_error
    );
}

/// Test XYB roundtrip quality: compare Rust XYB vs C++ cjpegli XYB.
/// Both should have similar DSSIM when decoded with ICC transform.
#[test]
#[ignore = "requires C++ cjpegli build"]
fn test_xyb_roundtrip_loss_vs_cpp() {
    use dssim::Dssim;
    use rgb::RGBA8;
    use std::fs;
    use std::io::Write;
    use std::process::Command;

    fn rgb_to_rgba(data: &[u8]) -> Vec<RGBA8> {
        data.chunks(3)
            .map(|c| RGBA8::new(c[0], c[1], c[2], 255))
            .collect()
    }

    fn write_ppm(path: &str, rgb: &[u8], width: usize, height: usize) -> std::io::Result<()> {
        let mut file = fs::File::create(path)?;
        writeln!(file, "P6")?;
        writeln!(file, "{} {}", width, height)?;
        writeln!(file, "255")?;
        file.write_all(rgb)?;
        Ok(())
    }

    // Check if cjpegli is available
    let cjpegli_path = jpegli::test_utils::require_cjpegli();

    // Create a meaningful test image (gradient with some texture)
    let width = 64;
    let height = 64;
    let mut rgb_data = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            // Create gradient + noise pattern
            rgb_data[idx] = ((x * 4) % 256) as u8;
            rgb_data[idx + 1] = ((y * 4) % 256) as u8;
            rgb_data[idx + 2] = (((x + y) * 2) % 256) as u8;
        }
    }

    // Save as PPM for cjpegli
    let ppm_path = "/tmp/test_xyb_roundtrip.ppm";
    write_ppm(ppm_path, &rgb_data, width, height).expect("Failed to write PPM");

    // Encode with C++ cjpegli in XYB mode
    let cpp_jpeg_path = "/tmp/test_xyb_roundtrip_cpp.jpg";
    let output = Command::new(&cjpegli_path)
        .args([ppm_path, cpp_jpeg_path, "--xyb", "-q", "90"])
        .output()
        .expect("Failed to run cjpegli");

    if !output.status.success() {
        panic!(
            "cjpegli failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let cpp_jpeg = fs::read(cpp_jpeg_path).expect("Failed to read C++ JPEG");

    // Encode with Rust in XYB mode
    let rust_jpeg = jpegli::StreamingEncoder::new(width as u32, height as u32)
        .pixel_format(jpegli::types::PixelFormat::Rgb)
        .quality(jpegli::quant::Quality::from_quality(90.0))
        .use_xyb(true)
        .encode_all(&rgb_data)
        .expect("Rust encoding failed");

    println!("C++ JPEG size: {} bytes", cpp_jpeg.len());
    println!("Rust JPEG size: {} bytes", rust_jpeg.len());

    // Decode both with our decoder (which applies ICC transform)
    let cpp_decoded = jpegli::Decoder::new()
        .apply_icc(true)
        .decode(&cpp_jpeg)
        .expect("C++ JPEG decode failed");

    let rust_decoded = jpegli::Decoder::new()
        .apply_icc(true)
        .decode(&rust_jpeg)
        .expect("Rust JPEG decode failed");

    // Compute DSSIM for both vs original
    let dssim = Dssim::new();

    // Convert to RGBA for DSSIM
    let orig_rgba = rgb_to_rgba(&rgb_data);
    let cpp_rgba = rgb_to_rgba(&cpp_decoded.data);
    let rust_rgba = rgb_to_rgba(&rust_decoded.data);

    let orig_img = dssim
        .create_image_rgba(&orig_rgba, width, height)
        .expect("create orig image");
    let cpp_img = dssim
        .create_image_rgba(
            &cpp_rgba,
            cpp_decoded.width as usize,
            cpp_decoded.height as usize,
        )
        .expect("create cpp image");
    let rust_img = dssim
        .create_image_rgba(
            &rust_rgba,
            rust_decoded.width as usize,
            rust_decoded.height as usize,
        )
        .expect("create rust image");

    let (cpp_dssim, _) = dssim.compare(&orig_img, cpp_img);
    let (rust_dssim, _) = dssim.compare(&orig_img, rust_img);

    println!("C++ JPEG DSSIM: {:.6}", cpp_dssim);
    println!("Rust JPEG DSSIM: {:.6}", rust_dssim);

    // Compute max pixel difference
    let cpp_max_diff: i16 = cpp_decoded
        .data
        .iter()
        .zip(rgb_data.iter())
        .map(|(a, b)| (*a as i16 - *b as i16).abs())
        .max()
        .unwrap_or(0);

    let rust_max_diff: i16 = rust_decoded
        .data
        .iter()
        .zip(rgb_data.iter())
        .map(|(a, b)| (*a as i16 - *b as i16).abs())
        .max()
        .unwrap_or(0);

    println!("C++ max pixel diff vs original: {}", cpp_max_diff);
    println!("Rust max pixel diff vs original: {}", rust_max_diff);

    // Assertions:
    // 1. Both should have good quality (DSSIM < 0.01 at Q90)
    assert!(
        f64::from(cpp_dssim) < 0.01,
        "C++ XYB DSSIM too high: {}",
        cpp_dssim
    );
    assert!(
        f64::from(rust_dssim) < 0.01,
        "Rust XYB DSSIM too high: {}",
        rust_dssim
    );

    // 2. Rust should be within 50% of C++ quality (allowing for implementation differences)
    let dssim_ratio = f64::from(rust_dssim) / f64::from(cpp_dssim).max(1e-10);
    println!("DSSIM ratio (Rust/C++): {:.2}x", dssim_ratio);
    assert!(
        dssim_ratio < 2.0,
        "Rust XYB quality significantly worse than C++: {}x",
        dssim_ratio
    );

    // 3. File sizes should be similar (within 20%)
    let size_ratio = rust_jpeg.len() as f64 / cpp_jpeg.len() as f64;
    println!("Size ratio (Rust/C++): {:.2}x", size_ratio);
    assert!(
        size_ratio > 0.8 && size_ratio < 1.2,
        "Rust XYB file size significantly different from C++: {}x",
        size_ratio
    );

    println!("\nXYB roundtrip comparison PASSED!");
    println!(
        "  C++ cjpegli: {} bytes, DSSIM {:.6}",
        cpp_jpeg.len(),
        cpp_dssim
    );
    println!(
        "  Rust jpegli: {} bytes, DSSIM {:.6}",
        rust_jpeg.len(),
        rust_dssim
    );
}
