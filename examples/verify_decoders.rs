//! Verify that generated JPEGs can be decoded by multiple decoders.
//!
//! Tests against:
//! - jpeg_decoder (Rust)
//! - djpeg (libjpeg-turbo)
//! - ImageMagick convert
//!
//! Run with: cargo run --example verify_decoders

use std::process::Command;
use zenjpeg::{Encoder, Quality, ScanScript, ScanInfo};

struct DecoderResult {
    name: &'static str,
    success: bool,
    error: Option<String>,
}

fn test_jpeg_decoder(data: &[u8]) -> DecoderResult {
    match jpeg_decoder::Decoder::new(data).decode() {
        Ok(_) => DecoderResult { name: "jpeg_decoder", success: true, error: None },
        Err(e) => DecoderResult { name: "jpeg_decoder", success: false, error: Some(format!("{:?}", e)) },
    }
}

fn test_djpeg(data: &[u8], temp_path: &str) -> DecoderResult {
    std::fs::write(temp_path, data).unwrap();
    let output = Command::new("djpeg")
        .args(["-grayscale", "-outfile", "/dev/null", temp_path])
        .output();

    match output {
        Ok(o) if o.status.success() => DecoderResult { name: "djpeg", success: true, error: None },
        Ok(o) => DecoderResult {
            name: "djpeg",
            success: false,
            error: Some(String::from_utf8_lossy(&o.stderr).to_string())
        },
        Err(e) => DecoderResult { name: "djpeg", success: false, error: Some(format!("{}", e)) },
    }
}

fn test_imagemagick(data: &[u8], temp_path: &str) -> DecoderResult {
    std::fs::write(temp_path, data).unwrap();
    let output = Command::new("convert")
        .args([temp_path, "-format", "%w", "info:"])
        .output();

    match output {
        Ok(o) if o.status.success() => DecoderResult { name: "ImageMagick", success: true, error: None },
        Ok(o) => DecoderResult {
            name: "ImageMagick",
            success: false,
            error: Some(String::from_utf8_lossy(&o.stderr).to_string())
        },
        Err(e) => DecoderResult { name: "ImageMagick", success: false, error: Some(format!("{}", e)) },
    }
}

fn verify_with_all_decoders(name: &str, data: &[u8], temp_path: &str) -> bool {
    let results = vec![
        test_jpeg_decoder(data),
        test_djpeg(data, temp_path),
        test_imagemagick(data, temp_path),
    ];

    let mut all_pass = true;
    print!("  {}: ", name);
    for r in &results {
        if r.success {
            print!("{} OK  ", r.name);
        } else {
            print!("{} FAIL  ", r.name);
            all_pass = false;
        }
    }
    println!();

    // Print errors for failures
    for r in &results {
        if !r.success {
            if let Some(ref e) = r.error {
                println!("    {} error: {}", r.name, e.trim());
            }
        }
    }

    all_pass
}

fn main() {
    let mut total = 0;
    let mut passed = 0;

    println!("=== Decoder Verification Tests ===\n");

    // Test 1: Simple grayscale
    println!("Test 1: Simple grayscale (baseline)");
    {
        let pixels: Vec<u8> = (0..64).map(|i| (i * 4) as u8).collect();
        let jpeg = Encoder::new()
            .quality(Quality::Standard(90))
            .encode_gray(&pixels, 8, 8)
            .expect("encode");
        total += 1;
        if verify_with_all_decoders("8x8 gradient", &jpeg, "/tmp/verify_test1.jpg") {
            passed += 1;
        }
    }

    // Test 2: Progressive with Simple script (no refinement)
    println!("\nTest 2: Progressive - Simple script (DC then AC)");
    {
        let pixels: Vec<u8> = (0..64).map(|i| (i * 4) as u8).collect();
        let jpeg = Encoder::new()
            .quality(Quality::Standard(90))
            .progressive(true)
            .scan_script(ScanScript::Simple)
            .encode_gray(&pixels, 8, 8)
            .expect("encode");
        total += 1;
        if verify_with_all_decoders("Simple script", &jpeg, "/tmp/verify_test2.jpg") {
            passed += 1;
        }
    }

    // Test 3: Progressive with Standard script (with refinement)
    println!("\nTest 3: Progressive - Standard script (with refinement)");
    {
        // Diagonal gradient - known to trigger the -1 coefficient edge case
        let mut pixels = vec![0u8; 64];
        for y in 0..8 {
            for x in 0..8 {
                pixels[y * 8 + x] = ((x * 32 + y * 16) % 256) as u8;
            }
        }
        let jpeg = Encoder::new()
            .quality(Quality::Standard(90))
            .progressive(true)
            .scan_script(ScanScript::Standard)
            .encode_gray(&pixels, 8, 8)
            .expect("encode");
        total += 1;
        if verify_with_all_decoders("Standard script", &jpeg, "/tmp/verify_test3.jpg") {
            passed += 1;
        }
    }

    // Test 4: Custom 3-scan refinement script
    println!("\nTest 4: Custom refinement script (DC, AC first Al=1, AC refine)");
    {
        let mut pixels = vec![0u8; 64];
        for y in 0..8 {
            for x in 0..8 {
                pixels[y * 8 + x] = ((x * 32 + y * 16) % 256) as u8;
            }
        }
        let scans = vec![
            ScanInfo { comps_in_scan: 1, component_index: [0,0,0,0], ss: 0, se: 0, ah: 0, al: 0 },
            ScanInfo { comps_in_scan: 1, component_index: [0,0,0,0], ss: 1, se: 63, ah: 0, al: 1 },
            ScanInfo { comps_in_scan: 1, component_index: [0,0,0,0], ss: 1, se: 63, ah: 1, al: 0 },
        ];
        let jpeg = Encoder::new()
            .quality(Quality::Standard(90))
            .progressive(true)
            .scan_script(ScanScript::Custom(scans))
            .encode_gray(&pixels, 8, 8)
            .expect("encode");
        total += 1;
        if verify_with_all_decoders("Custom refine", &jpeg, "/tmp/verify_test4.jpg") {
            passed += 1;
        }
    }

    // Test 5: Flat image with refinement (edge case: mostly zeros)
    println!("\nTest 5: Flat image with refinement (minimal AC coefficients)");
    {
        let pixels = vec![128u8; 64];
        let scans = vec![
            ScanInfo { comps_in_scan: 1, component_index: [0,0,0,0], ss: 0, se: 0, ah: 0, al: 0 },
            ScanInfo { comps_in_scan: 1, component_index: [0,0,0,0], ss: 1, se: 63, ah: 0, al: 1 },
            ScanInfo { comps_in_scan: 1, component_index: [0,0,0,0], ss: 1, se: 63, ah: 1, al: 0 },
        ];
        let jpeg = Encoder::new()
            .quality(Quality::Standard(90))
            .progressive(true)
            .scan_script(ScanScript::Custom(scans))
            .encode_gray(&pixels, 8, 8)
            .expect("encode");
        total += 1;
        if verify_with_all_decoders("Flat + refine", &jpeg, "/tmp/verify_test5.jpg") {
            passed += 1;
        }
    }

    // Test 6: Larger image with refinement
    println!("\nTest 6: Larger image (64x64) with refinement");
    {
        let mut pixels = vec![0u8; 64 * 64];
        for y in 0..64 {
            for x in 0..64 {
                pixels[y * 64 + x] = ((x * 4 + y * 2) % 256) as u8;
            }
        }
        let jpeg = Encoder::new()
            .quality(Quality::Standard(85))
            .progressive(true)
            .scan_script(ScanScript::Standard)
            .encode_gray(&pixels, 64, 64)
            .expect("encode");
        total += 1;
        if verify_with_all_decoders("64x64 Standard", &jpeg, "/tmp/verify_test6.jpg") {
            passed += 1;
        }
    }

    // Test 7: High contrast pattern (many non-zero coefficients)
    println!("\nTest 7: Checkerboard pattern with refinement");
    {
        let mut pixels = vec![0u8; 64];
        for y in 0..8 {
            for x in 0..8 {
                pixels[y * 8 + x] = if (x + y) % 2 == 0 { 255 } else { 0 };
            }
        }
        let jpeg = Encoder::new()
            .quality(Quality::Standard(90))
            .progressive(true)
            .scan_script(ScanScript::Standard)
            .encode_gray(&pixels, 8, 8)
            .expect("encode");
        total += 1;
        if verify_with_all_decoders("Checkerboard", &jpeg, "/tmp/verify_test7.jpg") {
            passed += 1;
        }
    }

    // Test 8: Different quality levels
    println!("\nTest 8: Various quality levels with refinement");
    for q in [50, 75, 95] {
        let mut pixels = vec![0u8; 64];
        for i in 0..64 {
            pixels[i] = (i * 4) as u8;
        }
        let jpeg = Encoder::new()
            .quality(Quality::Standard(q))
            .progressive(true)
            .scan_script(ScanScript::Standard)
            .encode_gray(&pixels, 8, 8)
            .expect("encode");
        total += 1;
        if verify_with_all_decoders(&format!("Q{}", q), &jpeg, "/tmp/verify_test8.jpg") {
            passed += 1;
        }
    }

    // Summary
    println!("\n=== Summary ===");
    println!("{}/{} tests passed", passed, total);

    if passed == total {
        println!("All decoders successfully parsed all generated JPEGs!");
    } else {
        println!("Some tests failed - review errors above.");
        std::process::exit(1);
    }
}
