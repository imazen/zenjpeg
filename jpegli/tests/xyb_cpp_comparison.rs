//! Test XYB mode output against C++ jpegli reference.

use std::fs;
use std::process::Command;

/// Compare Rust XYB output with C++ cjpegli output.
#[test]
#[ignore = "requires C++ cjpegli build"]
fn test_xyb_cpp_comparison() {
    // Check if cjpegli is available
    let cjpegli_path = "/home/lilith/work/jpegli/build/tools/cjpegli";
    if !std::path::Path::new(cjpegli_path).exists() {
        panic!("cjpegli not found at {}. Build it first.", cjpegli_path);
    }

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
    #[allow(deprecated)]
    let rust_jpeg = jpegli::encode::Encoder::new()
        .width(width as u32)
        .height(height as u32)
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
    assert!(
        has_icc_profile(&cpp_jpeg),
        "C++ JPEG missing ICC profile"
    );
    assert!(
        has_icc_profile(&rust_jpeg),
        "Rust JPEG missing ICC profile"
    );

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

/// Test that XYB values match C++ implementation.
#[test]
fn test_xyb_color_conversion_values() {
    use jpegli::xyb::{srgb_to_scaled_xyb, SCALED_XYB_OFFSET, SCALED_XYB_SCALE};

    // Test known values - these should match C++ jpegli's XYB conversion
    let test_cases = [
        // (r, g, b) -> expected scaled (x, y, b) ranges
        ((0, 0, 0), (0.35, 0.0, 0.41)),
        ((255, 255, 255), (0.35, 1.0, 0.42)),
        ((255, 0, 0), (0.95, 0.55, 0.35)),
        ((0, 255, 0), (0.05, 0.80, 0.05)),
        ((0, 0, 255), (0.35, 0.30, 0.95)),
    ];

    for ((r, g, b), (exp_x, exp_y, exp_b)) in test_cases {
        let (x, y, b_out) = srgb_to_scaled_xyb(r, g, b);

        // Values should be approximately in [0, 1] range (allow small floating point tolerance)
        let eps = 1e-5;
        assert!(x >= -eps && x <= 1.0 + eps, "X out of range: {}", x);
        assert!(y >= -eps && y <= 1.0 + eps, "Y out of range: {}", y);
        assert!(b_out >= -eps && b_out <= 1.0 + eps, "B out of range: {}", b_out);

        // Rough range check (exact values depend on implementation details)
        assert!(
            (x - exp_x).abs() < 0.15,
            "X mismatch for ({}, {}, {}): got {}, expected ~{}",
            r,
            g,
            b,
            x,
            exp_x
        );
        assert!(
            (y - exp_y).abs() < 0.15,
            "Y mismatch for ({}, {}, {}): got {}, expected ~{}",
            r,
            g,
            b,
            y,
            exp_y
        );
        assert!(
            (b_out - exp_b).abs() < 0.15,
            "B mismatch for ({}, {}, {}): got {}, expected ~{}",
            r,
            g,
            b,
            b_out,
            exp_b
        );
    }

    println!("Scaled XYB conversion values validated");
    println!(
        "Scale factors: {:?}, Offsets: {:?}",
        SCALED_XYB_SCALE, SCALED_XYB_OFFSET
    );
}

/// Test ICC profile embedding
#[test]
fn test_icc_profile_embedding() {
    let width = 8;
    let height = 8;
    let rgb_data = vec![128u8; width * height * 3];

    #[allow(deprecated)]
    let jpeg = jpegli::encode::Encoder::new()
        .width(width as u32)
        .height(height as u32)
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

    println!("ICC profile embedding validated: {} bytes", icc_profile.len());
}
