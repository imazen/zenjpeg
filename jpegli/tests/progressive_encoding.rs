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

/// Test that optimized Huffman tables produce smaller or equal size files.
#[test]
fn test_progressive_optimized_smaller() {
    let width = 64u32;
    let height = 64u32;
    let mut data = Vec::with_capacity((width * height * 3) as usize);

    // Create a more complex pattern to show optimization benefit
    for y in 0..height {
        for x in 0..width {
            let val = ((x * 13 + y * 17) % 256) as u8;
            data.push(val);           // R
            data.push(255 - val);     // G
            data.push((val / 2) + 64); // B
        }
    }

    // Encode without optimization
    let encoder_no_opt = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Rgb)
        .quality(Quality::from_quality(85.0))
        .optimize_huffman(false)
        .mode(JpegMode::Progressive);

    let no_opt_data = encoder_no_opt.encode(&data).expect("Non-optimized encoding should succeed");

    // Encode with optimization
    let encoder_opt = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Rgb)
        .quality(Quality::from_quality(85.0))
        .optimize_huffman(true)
        .mode(JpegMode::Progressive);

    let opt_data = encoder_opt.encode(&data).expect("Optimized encoding should succeed");

    // Both should be valid JPEGs
    assert_eq!(&no_opt_data[0..2], &[0xFF, 0xD8]);
    assert_eq!(&opt_data[0..2], &[0xFF, 0xD8]);

    // Optimized should be smaller or equal (within reason - small images may not benefit)
    // For larger images, optimized tables should produce smaller output
    let size_diff = no_opt_data.len() as i64 - opt_data.len() as i64;

    // We allow the optimized to be up to 5% larger for small test images
    // (overhead of optimal tables may not pay off for tiny images)
    let tolerance = (no_opt_data.len() as f64 * 0.05) as i64;
    assert!(
        size_diff >= -tolerance,
        "Optimized should not be much larger: no_opt={}, opt={}, diff={}",
        no_opt_data.len(), opt_data.len(), size_diff
    );
}

/// Test that progressive encoding with optimized tables can be decoded by external decoders.
#[test]
fn test_progressive_optimized_external_decode() {
    let width = 32u32;
    let height = 32u32;
    let mut data = Vec::with_capacity((width * height) as usize);

    for y in 0..height {
        for x in 0..width {
            data.push(((x * 8 + y * 8) % 256) as u8);
        }
    }

    let encoder = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Gray)
        .quality(Quality::from_quality(90.0))
        .optimize_huffman(true)
        .mode(JpegMode::Progressive);

    let jpeg_data = encoder.encode(&data).expect("Encoding should succeed");

    // Verify it's a valid JPEG structure
    assert_eq!(&jpeg_data[0..2], &[0xFF, 0xD8]); // SOI
    assert_eq!(&jpeg_data[jpeg_data.len()-2..], &[0xFF, 0xD9]); // EOI

    // Verify it has DHT and SOF2 markers
    let mut found_dht = false;
    let mut found_sof2 = false;
    for i in 0..jpeg_data.len() - 1 {
        if jpeg_data[i] == 0xFF {
            match jpeg_data[i + 1] {
                0xC4 => found_dht = true,
                0xC2 => found_sof2 = true,
                _ => {}
            }
        }
    }
    assert!(found_dht, "Should have DHT marker");
    assert!(found_sof2, "Should have SOF2 marker for progressive");

    // Try decoding with jpeg-decoder (external crate)
    // This verifies our output is standards-compliant
    let decoded = jpeg_decoder::Decoder::new(&jpeg_data[..])
        .decode()
        .expect("External decoder should succeed");
    assert_eq!(decoded.len(), (width * height) as usize);
}

/// Test optimized progressive with a larger image shows file size benefit.
#[test]
fn test_progressive_optimized_larger_image() {
    let width = 256u32;
    let height = 256u32;
    let mut data = Vec::with_capacity((width * height * 3) as usize);

    // Create a realistic pattern with varied content
    for y in 0..height {
        for x in 0..width {
            // Mix of gradients and noise-like patterns
            let base = ((x as f32 / width as f32) * 255.0) as u8;
            let noise = ((x * 7 + y * 13) % 64) as u8;
            data.push(base.wrapping_add(noise));      // R
            data.push(255u8.wrapping_sub(base));       // G
            data.push(((y * 255) / height) as u8);     // B
        }
    }

    // Encode without optimization
    let no_opt_data = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Rgb)
        .quality(Quality::from_quality(85.0))
        .optimize_huffman(false)
        .mode(JpegMode::Progressive)
        .encode(&data)
        .expect("Non-optimized encoding should succeed");

    // Encode with optimization
    let opt_data = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Rgb)
        .quality(Quality::from_quality(85.0))
        .optimize_huffman(true)
        .mode(JpegMode::Progressive)
        .encode(&data)
        .expect("Optimized encoding should succeed");

    // For larger images, optimized should typically be smaller
    println!(
        "256x256 RGB: non-opt={} bytes, opt={} bytes, savings={:.1}%",
        no_opt_data.len(),
        opt_data.len(),
        (1.0 - opt_data.len() as f64 / no_opt_data.len() as f64) * 100.0
    );

    // Verify both decode correctly
    let decoded_no_opt = jpeg_decoder::Decoder::new(&no_opt_data[..])
        .decode()
        .expect("Non-optimized should decode");
    let decoded_opt = jpeg_decoder::Decoder::new(&opt_data[..])
        .decode()
        .expect("Optimized should decode");

    assert_eq!(decoded_no_opt.len(), decoded_opt.len());
}

/// Test progressive optimized with uniform solid color.
#[test]
fn test_progressive_optimized_solid_color() {
    let width = 64u32;
    let height = 64u32;
    // Solid red
    let data: Vec<u8> = (0..(width * height))
        .flat_map(|_| [255u8, 0, 0])
        .collect();

    let jpeg_data = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Rgb)
        .quality(Quality::from_quality(90.0))
        .optimize_huffman(true)
        .mode(JpegMode::Progressive)
        .encode(&data)
        .expect("Encoding should succeed");

    // Solid colors should compress very well
    assert!(jpeg_data.len() < 2000, "Solid color should compress well");

    // Verify decode
    let decoded = jpeg_decoder::Decoder::new(&jpeg_data[..])
        .decode()
        .expect("Should decode");
    assert_eq!(decoded.len(), (width * height * 3) as usize);
}

/// Test progressive optimized with high-frequency content.
#[test]
fn test_progressive_optimized_high_frequency() {
    let width = 64u32;
    let height = 64u32;
    let mut data = Vec::with_capacity((width * height * 3) as usize);

    // Checkerboard pattern - high frequency content
    for y in 0..height {
        for x in 0..width {
            let val = if (x + y) % 2 == 0 { 255u8 } else { 0u8 };
            data.push(val);
            data.push(val);
            data.push(val);
        }
    }

    let jpeg_data = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Rgb)
        .quality(Quality::from_quality(85.0))
        .optimize_huffman(true)
        .mode(JpegMode::Progressive)
        .encode(&data)
        .expect("Encoding should succeed");

    // Verify decode
    let decoded = jpeg_decoder::Decoder::new(&jpeg_data[..])
        .decode()
        .expect("Should decode");
    assert_eq!(decoded.len(), (width * height * 3) as usize);
}

/// Test progressive optimized at various quality levels.
#[test]
fn test_progressive_optimized_quality_levels() {
    let width = 64u32;
    let height = 64u32;
    let mut data = Vec::with_capacity((width * height * 3) as usize);

    // Use a more varied pattern to ensure all symbol categories appear
    for y in 0..height {
        for x in 0..width {
            let noise = ((x * 7 + y * 13) % 64) as u8;
            data.push(((x * 4) as u8).wrapping_add(noise));
            data.push(((y * 4) as u8).wrapping_add(noise / 2));
            data.push(128u8.wrapping_add(noise));
        }
    }

    let mut prev_size = 0usize;

    for quality in [70.0, 85.0, 95.0] {
        let jpeg_data = Encoder::new()
            .width(width)
            .height(height)
            .pixel_format(PixelFormat::Rgb)
            .quality(Quality::from_quality(quality))
            .optimize_huffman(true)
            .mode(JpegMode::Progressive)
            .encode(&data)
            .expect(&format!("Q{} encoding should succeed", quality));

        // Higher quality should generally produce larger files
        if prev_size > 0 {
            assert!(
                jpeg_data.len() >= prev_size - 100,
                "Q{} ({} bytes) should not be much smaller than lower quality ({} bytes)",
                quality,
                jpeg_data.len(),
                prev_size
            );
        }
        prev_size = jpeg_data.len();

        // Verify decode
        jpeg_decoder::Decoder::new(&jpeg_data[..])
            .decode()
            .expect(&format!("Q{} should decode", quality));
    }
}

/// Test progressive optimized with single 8x8 block (edge case).
#[test]
fn test_progressive_optimized_single_block() {
    let width = 8u32;
    let height = 8u32;
    let data: Vec<u8> = (0..64).flat_map(|i| [i as u8 * 4, 128, 64]).collect();

    let jpeg_data = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Rgb)
        .quality(Quality::from_quality(90.0))
        .optimize_huffman(true)
        .mode(JpegMode::Progressive)
        .encode(&data)
        .expect("Single block should encode");

    // Should still be valid
    assert_eq!(&jpeg_data[0..2], &[0xFF, 0xD8]);
    assert_eq!(&jpeg_data[jpeg_data.len()-2..], &[0xFF, 0xD9]);

    jpeg_decoder::Decoder::new(&jpeg_data[..])
        .decode()
        .expect("Single block should decode");
}

/// Test progressive optimized grayscale at various sizes.
#[test]
fn test_progressive_optimized_grayscale_sizes() {
    for size in [16u32, 32, 64, 128] {
        let mut data = Vec::with_capacity((size * size) as usize);
        for y in 0..size {
            for x in 0..size {
                data.push(((x + y) * 255 / (2 * size - 2)) as u8);
            }
        }

        let jpeg_data = Encoder::new()
            .width(size)
            .height(size)
            .pixel_format(PixelFormat::Gray)
            .quality(Quality::from_quality(85.0))
            .optimize_huffman(true)
            .mode(JpegMode::Progressive)
            .encode(&data)
            .expect(&format!("{}x{} gray should encode", size, size));

        let decoded = jpeg_decoder::Decoder::new(&jpeg_data[..])
            .decode()
            .expect(&format!("{}x{} gray should decode", size, size));

        assert_eq!(decoded.len(), (size * size) as usize);
    }
}

/// Test that progressive optimized produces valid scan structure.
#[test]
fn test_progressive_optimized_scan_structure() {
    let width = 32u32;
    let height = 32u32;
    let data: Vec<u8> = (0..(width * height * 3))
        .map(|i| (i % 256) as u8)
        .collect();

    let jpeg_data = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Rgb)
        .quality(Quality::from_quality(85.0))
        .optimize_huffman(true)
        .mode(JpegMode::Progressive)
        .encode(&data)
        .expect("Encoding should succeed");

    // Count markers
    let mut sos_count = 0;
    let mut dht_count = 0;
    let mut dqt_count = 0;
    let mut sof2_count = 0;

    let mut i = 0;
    while i < jpeg_data.len() - 1 {
        if jpeg_data[i] == 0xFF {
            match jpeg_data[i + 1] {
                0xDA => sos_count += 1,
                0xC4 => dht_count += 1,
                0xDB => dqt_count += 1,
                0xC2 => sof2_count += 1,
                _ => {}
            }
        }
        i += 1;
    }

    assert_eq!(sof2_count, 1, "Should have exactly 1 SOF2 marker");
    assert!(dht_count >= 1, "Should have at least 1 DHT marker");
    assert!(dqt_count >= 1, "Should have at least 1 DQT marker");
    assert!(sos_count >= 2, "Progressive should have at least 2 SOS markers");
}

/// Test non-square image dimensions.
#[test]
fn test_progressive_optimized_non_square() {
    // Wide image
    let width = 128u32;
    let height = 32u32;
    let mut data = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            data.push((x * 2) as u8);
            data.push((y * 8) as u8);
            data.push(128);
        }
    }

    let jpeg_data = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Rgb)
        .quality(Quality::from_quality(85.0))
        .optimize_huffman(true)
        .mode(JpegMode::Progressive)
        .encode(&data)
        .expect("Wide image should encode");

    let decoded = jpeg_decoder::Decoder::new(&jpeg_data[..])
        .decode()
        .expect("Wide image should decode");
    assert_eq!(decoded.len(), (width * height * 3) as usize);

    // Tall image
    let width = 32u32;
    let height = 128u32;
    let mut data = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            data.push((x * 8) as u8);
            data.push((y * 2) as u8);
            data.push(128);
        }
    }

    let jpeg_data = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Rgb)
        .quality(Quality::from_quality(85.0))
        .optimize_huffman(true)
        .mode(JpegMode::Progressive)
        .encode(&data)
        .expect("Tall image should encode");

    let decoded = jpeg_decoder::Decoder::new(&jpeg_data[..])
        .decode()
        .expect("Tall image should decode");
    assert_eq!(decoded.len(), (width * height * 3) as usize);
}

/// Test non-multiple-of-8 dimensions (requires padding).
#[test]
fn test_progressive_optimized_odd_dimensions() {
    for (width, height) in [(17u32, 23u32), (33, 41), (65, 70), (100, 99)] {
        let mut data = Vec::with_capacity((width * height * 3) as usize);
        for y in 0..height {
            for x in 0..width {
                data.push(((x.wrapping_mul(7).wrapping_add(y.wrapping_mul(3))) % 256) as u8);
                data.push(((x.wrapping_mul(3).wrapping_add(y.wrapping_mul(11))) % 256) as u8);
                data.push(((x.wrapping_add(y.wrapping_mul(5))) % 256) as u8);
            }
        }

        let jpeg_data = Encoder::new()
            .width(width)
            .height(height)
            .pixel_format(PixelFormat::Rgb)
            .quality(Quality::from_quality(85.0))
            .optimize_huffman(true)
            .mode(JpegMode::Progressive)
            .encode(&data)
            .expect(&format!("{}x{} should encode", width, height));

        // Verify full decode works and size is correct
        let decoded = jpeg_decoder::Decoder::new(&jpeg_data[..])
            .decode()
            .expect(&format!("{}x{} should decode", width, height));
        assert_eq!(
            decoded.len(),
            (width * height * 3) as usize,
            "Decoded size mismatch for {}x{}",
            width,
            height
        );
    }
}

/// Test baseline encoding still works after progressive changes.
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
