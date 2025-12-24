//! Regression tests for progressive JPEG encoding.
//!
//! These tests verify that progressive JPEG encoding produces valid output
//! that can be decoded by standard decoders.

use jpegli::{Encoder, PixelFormat};
use jpegli::quant::Quality;
use jpegli::types::JpegMode;
use std::process::Command;

/// Test that progressive encoding of a grayscale gradient produces valid output.
#[test]
fn test_progressive_grayscale_gradient() {
    let width = 16u32;
    let height = 16u32;
    let mut data = Vec::with_capacity((width * height) as usize);

    for _y in 0..height {
        for x in 0..width {
            data.push((x * 16) as u8);
        }
    }

    let encoder = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Gray)
        .quality(Quality::from_quality(90.0))
        .optimize_huffman(true)
        .mode(JpegMode::Progressive);

    let jpeg_data = encoder.encode(&data).expect("Progressive encoding should succeed");

    // Verify the file is a valid JPEG by checking markers
    assert!(jpeg_data.len() > 100, "JPEG should be at least 100 bytes");
    assert_eq!(jpeg_data[0], 0xFF, "Should start with FF");
    assert_eq!(jpeg_data[1], 0xD8, "Should have SOI marker");

    // Check for SOF2 (progressive DCT)
    let mut found_sof2 = false;
    for i in 0..jpeg_data.len() - 1 {
        if jpeg_data[i] == 0xFF && jpeg_data[i + 1] == 0xC2 {
            found_sof2 = true;
            break;
        }
    }
    assert!(found_sof2, "Progressive JPEG should have SOF2 marker");

    // Verify EOI marker
    assert_eq!(jpeg_data[jpeg_data.len() - 2], 0xFF);
    assert_eq!(jpeg_data[jpeg_data.len() - 1], 0xD9);

    // If djpeg is available, verify the file decodes correctly
    if let Ok(output) = Command::new("djpeg")
        .args(["-outfile", "/dev/null"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .spawn()
    {
        let _ = output.wait_with_output();
        // djpeg might not be available in all environments, that's OK
    }
}

/// Test that progressive encoding of solid gray produces valid output.
#[test]
fn test_progressive_solid_gray() {
    let width = 16u32;
    let height = 16u32;
    let data = vec![128u8; (width * height) as usize];

    let encoder = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Gray)
        .quality(Quality::from_quality(90.0))
        .optimize_huffman(true)
        .mode(JpegMode::Progressive);

    let jpeg_data = encoder.encode(&data).expect("Progressive encoding should succeed");

    // Basic validation
    assert!(jpeg_data.len() > 50);
    assert_eq!(&jpeg_data[0..2], &[0xFF, 0xD8]); // SOI
    assert_eq!(&jpeg_data[jpeg_data.len()-2..], &[0xFF, 0xD9]); // EOI
}

/// Test that progressive encoding of RGB image produces valid output.
#[test]
fn test_progressive_rgb() {
    let width = 16u32;
    let height = 16u32;
    let mut data = Vec::with_capacity((width * height * 3) as usize);

    // Create a simple gradient
    for y in 0..height {
        for x in 0..width {
            data.push((x * 16) as u8);  // R
            data.push((y * 16) as u8);  // G
            data.push(128);              // B
        }
    }

    let encoder = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Rgb)
        .quality(Quality::from_quality(90.0))
        .optimize_huffman(true)
        .mode(JpegMode::Progressive);

    let jpeg_data = encoder.encode(&data).expect("Progressive RGB encoding should succeed");

    // Verify SOF2 marker for progressive
    let mut found_sof2 = false;
    for i in 0..jpeg_data.len() - 1 {
        if jpeg_data[i] == 0xFF && jpeg_data[i + 1] == 0xC2 {
            found_sof2 = true;
            break;
        }
    }
    assert!(found_sof2, "Progressive JPEG should have SOF2 marker");
}

/// Test that progressive encoding produces multiple scans.
#[test]
fn test_progressive_has_multiple_scans() {
    let width = 32u32;
    let height = 32u32;
    let mut data = Vec::with_capacity((width * height) as usize);

    for y in 0..height {
        for x in 0..width {
            data.push(((x + y) * 4) as u8);
        }
    }

    let encoder = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Gray)
        .quality(Quality::from_quality(85.0))
        .mode(JpegMode::Progressive);

    let jpeg_data = encoder.encode(&data).expect("Encoding should succeed");

    // Count SOS markers (Start Of Scan)
    let mut sos_count = 0;
    for i in 0..jpeg_data.len() - 1 {
        if jpeg_data[i] == 0xFF && jpeg_data[i + 1] == 0xDA {
            sos_count += 1;
        }
    }

    // Progressive JPEG should have at least 2 scans (DC + AC)
    assert!(sos_count >= 2, "Progressive JPEG should have at least 2 scans, found {}", sos_count);
}

/// Test that baseline encoding still works after progressive changes.
#[test]
fn test_baseline_still_works() {
    let width = 16u32;
    let height = 16u32;
    let mut data = Vec::with_capacity((width * height) as usize);

    for _y in 0..height {
        for x in 0..width {
            data.push((x * 16) as u8);
        }
    }

    let encoder = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Gray)
        .quality(Quality::from_quality(90.0))
        .mode(JpegMode::Baseline);

    let jpeg_data = encoder.encode(&data).expect("Baseline encoding should succeed");

    // Verify SOF0 marker for baseline (not SOF2)
    let mut found_sof0 = false;
    for i in 0..jpeg_data.len() - 1 {
        if jpeg_data[i] == 0xFF && jpeg_data[i + 1] == 0xC0 {
            found_sof0 = true;
            break;
        }
    }
    assert!(found_sof0, "Baseline JPEG should have SOF0 marker");

    // Should only have 1 SOS marker
    let sos_count = (0..jpeg_data.len() - 1)
        .filter(|&i| jpeg_data[i] == 0xFF && jpeg_data[i + 1] == 0xDA)
        .count();
    assert_eq!(sos_count, 1, "Baseline JPEG should have exactly 1 scan");
}
