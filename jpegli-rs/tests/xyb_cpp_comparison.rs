//! Test XYB mode output against C++ jpegli reference.
#![cfg(feature = "ffi-tests")]

use std::fs;
use std::process::Command;

/// Compare Rust XYB output with C++ cjpegli output.
#[test]
#[ignore = "requires C++ cjpegli build"]
fn test_xyb_cpp_comparison() {
    // Check if cjpegli is available
    let cjpegli_path = jpegli::test_utils::require_cjpegli();

    // Create test image (16x16 gradient)
    let width = 16;
    let height = 16;
    let mut rgb_data = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            rgb_data[idx] = (x * 16) as u8;
            rgb_data[idx + 1] = (y * 16) as u8;
            rgb_data[idx + 2] = 128;
        }
    }

    // Save as PPM for cjpegli
    let ppm_path = "/tmp/test_xyb_comparison.ppm";
    write_ppm(ppm_path, &rgb_data, width, height).expect("Failed to write PPM");

    // Encode with C++ cjpegli in XYB mode
    let cpp_jpeg_path = "/tmp/test_xyb_cpp.jpg";
    let output = Command::new(cjpegli_path)
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
    println!("C++ JPEG size: {} bytes", cpp_jpeg.len());

    // Encode with Rust in XYB mode
    let rust_jpeg = jpegli::JpegEncoder::new(width as u32, height as u32)
        .pixel_format(jpegli::types::PixelFormat::Rgb)
        .quality(jpegli::quant::Quality::from_quality(90.0))
        .use_xyb(true)
        .encode(&rgb_data)
        .expect("Rust encoding failed");

    println!("Rust JPEG size: {} bytes", rust_jpeg.len());

    // Save Rust output for inspection
    let rust_jpeg_path = "/tmp/test_xyb_rust.jpg";
    fs::write(rust_jpeg_path, &rust_jpeg).expect("Failed to write Rust JPEG");

    // Verify both have ICC profiles
    assert!(has_icc_profile(&cpp_jpeg), "C++ JPEG missing ICC profile");
    assert!(has_icc_profile(&rust_jpeg), "Rust JPEG missing ICC profile");

    // Extract and compare ICC profiles
    let cpp_icc = extract_icc_profile(&cpp_jpeg);
    let rust_icc = extract_icc_profile(&rust_jpeg);

    println!("C++ ICC profile size: {} bytes", cpp_icc.len());
    println!("Rust ICC profile size: {} bytes", rust_icc.len());

    // ICC profiles should be identical (we use the same embedded profile)
    assert_eq!(
        cpp_icc, rust_icc,
        "ICC profiles don't match between Rust and C++ output"
    );

    println!("SUCCESS: ICC profiles match!");
    println!("C++ JPEG saved to: {}", cpp_jpeg_path);
    println!("Rust JPEG saved to: {}", rust_jpeg_path);
}

/// Check if JPEG has ICC profile
fn has_icc_profile(data: &[u8]) -> bool {
    let sig = b"ICC_PROFILE\0";
    data.windows(12).any(|w| w == sig)
}

/// Extract ICC profile from JPEG
fn extract_icc_profile(data: &[u8]) -> Vec<u8> {
    let mut icc_chunks: Vec<(u8, Vec<u8>)> = Vec::new();

    let mut i = 0;
    while i < data.len().saturating_sub(4) {
        if data[i] == 0xFF && data[i + 1] == 0xE2 {
            // APP2 marker
            let length = ((data[i + 2] as usize) << 8) | (data[i + 3] as usize);
            if i + 2 + length <= data.len() {
                let marker_data = &data[i + 4..i + 2 + length];
                // Check for ICC_PROFILE signature
                if marker_data.len() >= 14 && &marker_data[..12] == b"ICC_PROFILE\0" {
                    let chunk_num = marker_data[12];
                    let _total = marker_data[13];
                    let chunk_data = marker_data[14..].to_vec();
                    icc_chunks.push((chunk_num, chunk_data));
                }
            }
            i += 2 + length;
        } else {
            i += 1;
        }
    }

    // Sort by chunk number and concatenate
    icc_chunks.sort_by_key(|(n, _)| *n);
    icc_chunks.into_iter().flat_map(|(_, d)| d).collect()
}

/// Write RGB data as PPM file
fn write_ppm(path: &str, rgb: &[u8], width: usize, height: usize) -> std::io::Result<()> {
    use std::io::Write;
    let mut file = fs::File::create(path)?;
    writeln!(file, "P6")?;
    writeln!(file, "{} {}", width, height)?;
    writeln!(file, "255")?;
    file.write_all(rgb)?;
    Ok(())
}

/// Test that XYB values match C++ implementation via FFI.
/// This test uses the actual C++ jpegli code to compute expected values.
#[test]
fn test_xyb_color_conversion_values() {
    use jpegli::xyb::{srgb_to_scaled_xyb, SCALED_XYB_OFFSET, SCALED_XYB_SCALE};

    // Test colors: sRGB values
    let test_colors: &[(u8, u8, u8)] = &[
        (0, 0, 0),       // Black
        (255, 255, 255), // White
        (255, 0, 0),     // Red
        (0, 255, 0),     // Green
        (0, 0, 255),     // Blue
        (128, 128, 128), // Gray
        (255, 128, 0),   // Orange
        (128, 0, 255),   // Purple
    ];

    // For each color, compute Rust result and C++ result via FFI
    for &(r, g, b) in test_colors {
        // Rust implementation
        let (rust_x, rust_y, rust_b) = srgb_to_scaled_xyb(r, g, b);

        // C++ implementation via FFI
        let srgb = [r, g, b];
        let mut cpp_xyb = [0.0f32; 3];
        unsafe {
            jpegli_internals_sys::jpegli_srgb_to_scaled_xyb(
                srgb.as_ptr(),
                1,
                1,
                255.0,
                cpp_xyb.as_mut_ptr(),
            );
        }
        let (cpp_x, cpp_y, cpp_b) = (cpp_xyb[0], cpp_xyb[1], cpp_xyb[2]);

        // Values should be in [0, 1] range (with small tolerance)
        let eps = 1e-5;
        assert!(
            rust_x >= -eps && rust_x <= 1.0 + eps,
            "X out of range: {}",
            rust_x
        );
        assert!(
            rust_y >= -eps && rust_y <= 1.0 + eps,
            "Y out of range: {}",
            rust_y
        );
        assert!(
            rust_b >= -eps && rust_b <= 1.0 + eps,
            "B out of range: {}",
            rust_b
        );

        // STRICT CHECK: Must match C++ within floating-point precision
        const XYB_TOLERANCE: f32 = 1e-5;
        assert!(
            (rust_x - cpp_x).abs() < XYB_TOLERANCE,
            "X mismatch for ({}, {}, {}): Rust={}, C++={} (diff: {})",
            r,
            g,
            b,
            rust_x,
            cpp_x,
            (rust_x - cpp_x).abs()
        );
        assert!(
            (rust_y - cpp_y).abs() < XYB_TOLERANCE,
            "Y mismatch for ({}, {}, {}): Rust={}, C++={} (diff: {})",
            r,
            g,
            b,
            rust_y,
            cpp_y,
            (rust_y - cpp_y).abs()
        );
        assert!(
            (rust_b - cpp_b).abs() < XYB_TOLERANCE,
            "B mismatch for ({}, {}, {}): Rust={}, C++={} (diff: {})",
            r,
            g,
            b,
            rust_b,
            cpp_b,
            (rust_b - cpp_b).abs()
        );

        println!(
            "({:3},{:3},{:3}) -> Rust: ({:.6}, {:.6}, {:.6}), C++: ({:.6}, {:.6}, {:.6})",
            r, g, b, rust_x, rust_y, rust_b, cpp_x, cpp_y, cpp_b
        );
    }

    println!("\nScaled XYB conversion matches C++ exactly!");
    println!(
        "Scale factors: {:?}, Offsets: {:?}",
        SCALED_XYB_SCALE, SCALED_XYB_OFFSET
    );
}

/// Test that XYB constants match C++.
#[test]
fn test_xyb_constants_match_cpp() {
    use jpegli::consts::{XYB_OPSIN_ABSORBANCE_BIAS, XYB_OPSIN_ABSORBANCE_MATRIX};
    use jpegli::xyb::{SCALED_XYB_OFFSET, SCALED_XYB_SCALE};

    // Get C++ constants via FFI
    let (cpp_matrix, cpp_bias, cpp_offset, cpp_scale) =
        jpegli_internals_sys::cpp_get_xyb_constants();

    println!("C++ Opsin Matrix:");
    for i in 0..3 {
        println!(
            "  [{:.6}, {:.6}, {:.6}]",
            cpp_matrix[i * 3],
            cpp_matrix[i * 3 + 1],
            cpp_matrix[i * 3 + 2]
        );
    }
    println!("C++ Opsin Bias: {:?}", cpp_bias);
    println!("C++ Scaled XYB Offset: {:?}", cpp_offset);
    println!("C++ Scaled XYB Scale: {:?}", cpp_scale);

    println!("\nRust Opsin Matrix:");
    for i in 0..3 {
        println!(
            "  [{:.6}, {:.6}, {:.6}]",
            XYB_OPSIN_ABSORBANCE_MATRIX[i * 3],
            XYB_OPSIN_ABSORBANCE_MATRIX[i * 3 + 1],
            XYB_OPSIN_ABSORBANCE_MATRIX[i * 3 + 2]
        );
    }
    println!("Rust Opsin Bias: {:?}", XYB_OPSIN_ABSORBANCE_BIAS);
    println!("Rust Scaled XYB Offset: {:?}", SCALED_XYB_OFFSET);
    println!("Rust Scaled XYB Scale: {:?}", SCALED_XYB_SCALE);

    // Compare
    const EPS: f32 = 1e-6;

    // Compare matrix
    for i in 0..9 {
        assert!(
            (cpp_matrix[i] - XYB_OPSIN_ABSORBANCE_MATRIX[i]).abs() < EPS,
            "Opsin matrix[{}] mismatch: C++={}, Rust={}",
            i,
            cpp_matrix[i],
            XYB_OPSIN_ABSORBANCE_MATRIX[i]
        );
    }

    // Compare bias
    for i in 0..3 {
        assert!(
            (cpp_bias[i] - XYB_OPSIN_ABSORBANCE_BIAS[i]).abs() < EPS,
            "Opsin bias[{}] mismatch: C++={}, Rust={}",
            i,
            cpp_bias[i],
            XYB_OPSIN_ABSORBANCE_BIAS[i]
        );
    }

    // Compare scaled XYB offset
    for i in 0..3 {
        assert!(
            (cpp_offset[i] - SCALED_XYB_OFFSET[i]).abs() < EPS,
            "Scaled XYB offset[{}] mismatch: C++={}, Rust={}",
            i,
            cpp_offset[i],
            SCALED_XYB_OFFSET[i]
        );
    }

    // Compare scaled XYB scale
    for i in 0..3 {
        assert!(
            (cpp_scale[i] - SCALED_XYB_SCALE[i]).abs() < EPS,
            "Scaled XYB scale[{}] mismatch: C++={}, Rust={}",
            i,
            cpp_scale[i],
            SCALED_XYB_SCALE[i]
        );
    }

    println!("\nAll XYB constants match C++ exactly!");
}

/// Test ICC profile embedding
#[test]
fn test_icc_profile_embedding() {
    let width = 8;
    let height = 8;
    let rgb_data = vec![128u8; width * height * 3];

    let jpeg = jpegli::JpegEncoder::new(width as u32, height as u32)
        .pixel_format(jpegli::types::PixelFormat::Rgb)
        .quality(jpegli::quant::Quality::from_quality(90.0))
        .use_xyb(true)
        .encode(&rgb_data)
        .expect("Encoding failed");

    // Check JPEG structure
    assert_eq!(jpeg[0], 0xFF, "Missing SOI marker");
    assert_eq!(jpeg[1], 0xD8, "Missing SOI marker");

    // Find ICC profile
    let icc_profile = extract_icc_profile(&jpeg);
    assert!(!icc_profile.is_empty(), "No ICC profile found");
    assert_eq!(
        icc_profile.len(),
        720,
        "ICC profile size mismatch (expected 720 bytes)"
    );

    // Verify ICC profile header
    // First 4 bytes should be profile size (big-endian)
    let profile_size = ((icc_profile[0] as u32) << 24)
        | ((icc_profile[1] as u32) << 16)
        | ((icc_profile[2] as u32) << 8)
        | (icc_profile[3] as u32);
    assert_eq!(profile_size, 720, "ICC profile header size mismatch");

    // Bytes 4-7 should be "jxl " (preferred CMM)
    assert_eq!(&icc_profile[4..8], b"jxl ", "ICC CMM signature mismatch");

    println!(
        "ICC profile embedding validated: {} bytes",
        icc_profile.len()
    );
}
