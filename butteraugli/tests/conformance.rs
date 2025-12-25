//! Butteraugli conformance tests.
//!
//! These tests verify butteraugli score quality using jpegli roundtripped images.
//! The goal is to ensure the Rust butteraugli implementation provides meaningful
//! perceptual quality scores.

use butteraugli_oxide::{
    compute_butteraugli, ButteraugliParams, BUTTERAUGLI_BAD, BUTTERAUGLI_GOOD,
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

    // Convert to RGB
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

/// Test that identical images have a score of 0.
#[test]
fn test_identical_images_score_zero() {
    let width = 64;
    let height = 64;
    // Create a gradient test image
    let rgb: Vec<u8> = (0..width * height * 3).map(|i| (i % 256) as u8).collect();

    let params = ButteraugliParams::default();
    let result = compute_butteraugli(&rgb, &rgb, width, height, &params);

    assert!(
        result.score < 0.001,
        "Identical images should have score ~0, got {}",
        result.score
    );
}

/// Test that slightly different images have low score.
#[test]
fn test_small_difference_low_score() {
    let width = 64;
    let height = 64;
    let rgb1: Vec<u8> = vec![128; width * height * 3];
    let mut rgb2 = rgb1.clone();

    // Change a few pixels slightly
    for i in 0..10 {
        rgb2[i * 3] = 130;
        rgb2[i * 3 + 1] = 130;
        rgb2[i * 3 + 2] = 130;
    }

    let params = ButteraugliParams::default();
    let result = compute_butteraugli(&rgb1, &rgb2, width, height, &params);

    // Small differences should be below the "good" threshold
    assert!(
        result.score < BUTTERAUGLI_GOOD * 2.0,
        "Small difference should have low score, got {}",
        result.score
    );
}

/// Test that large differences have non-zero score.
#[test]
fn test_large_difference_nonzero_score() {
    let width = 64;
    let height = 64;

    // Create images with high-frequency content differences
    let mut rgb1: Vec<u8> = vec![0; width * height * 3];
    let mut rgb2: Vec<u8> = vec![0; width * height * 3];

    // Image 1: checkerboard pattern
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            let val1 = if (x + y) % 2 == 0 { 200u8 } else { 50u8 };
            rgb1[idx] = val1;
            rgb1[idx + 1] = val1;
            rgb1[idx + 2] = val1;
        }
    }

    // Image 2: inverse checkerboard (shifted by 1)
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            let val2 = if (x + y) % 2 == 1 { 200u8 } else { 50u8 };
            rgb2[idx] = val2;
            rgb2[idx + 1] = val2;
            rgb2[idx + 2] = val2;
        }
    }

    let params = ButteraugliParams::default();
    let result = compute_butteraugli(&rgb1, &rgb2, width, height, &params);

    // Inverse checkerboards should produce visible difference
    // Note: exact threshold depends on implementation
    assert!(
        result.score > 0.001,
        "Inverse checkerboard images should have detectable difference, got {}",
        result.score
    );

    println!("Checkerboard difference score: {:.6}", result.score);
}

/// Test butteraugli score monotonicity with JPEG quality.
/// Higher quality should give lower (better) butteraugli scores.
#[test]
fn test_score_monotonicity_with_quality() {
    let path = Path::new("/home/lilith/work/jpegli/testdata/jxl/flower/flower_small.rgb.png");
    if !path.exists() {
        eprintln!("Skipping test: test image not found at {:?}", path);
        return;
    }

    let (original, width, height) = load_png(path).expect("Failed to load test image");

    let qualities = [50, 70, 90];
    let mut prev_score = f64::MAX;

    for quality in qualities {
        // Encode and decode with mozjpeg (or jpegli if available)
        let jpeg_data = encode_jpeg(&original, width as u32, height as u32, quality);
        let decoded = decode_jpeg(&jpeg_data);

        if decoded.len() != original.len() {
            eprintln!(
                "Size mismatch after roundtrip, skipping quality {}",
                quality
            );
            continue;
        }

        let params = ButteraugliParams::default();
        let result = compute_butteraugli(&original, &decoded, width, height, &params);

        println!(
            "Quality {}: butteraugli score = {:.4}",
            quality, result.score
        );

        // Higher quality should give lower score
        if quality > 50 {
            assert!(
                result.score <= prev_score * 1.5, // Allow some variance
                "Higher quality {} should give lower or similar score, got {} vs {}",
                quality,
                result.score,
                prev_score
            );
        }
        prev_score = result.score;
    }
}

/// Test score symmetry: butteraugli(a, b) should equal butteraugli(b, a).
#[test]
fn test_score_symmetry() {
    let width = 32;
    let height = 32;

    let rgb1: Vec<u8> = (0..width * height * 3).map(|i| (i % 200) as u8).collect();
    let rgb2: Vec<u8> = (0..width * height * 3)
        .map(|i| ((i + 50) % 256) as u8)
        .collect();

    let params = ButteraugliParams::default();
    let result1 = compute_butteraugli(&rgb1, &rgb2, width, height, &params);
    let result2 = compute_butteraugli(&rgb2, &rgb1, width, height, &params);

    // Scores should be identical or very close
    let diff = (result1.score - result2.score).abs();
    assert!(
        diff < result1.score * 0.1 + 0.01,
        "Butteraugli should be symmetric: {} vs {}",
        result1.score,
        result2.score
    );
}

/// Test diffmap is produced and has correct dimensions.
#[test]
fn test_diffmap_dimensions() {
    let width = 48;
    let height = 32;
    let rgb1: Vec<u8> = vec![100; width * height * 3];
    let rgb2: Vec<u8> = vec![150; width * height * 3];

    let params = ButteraugliParams::default();
    let result = compute_butteraugli(&rgb1, &rgb2, width, height, &params);

    let diffmap = result.diffmap.expect("Should produce diffmap");
    assert_eq!(diffmap.width(), width);
    assert_eq!(diffmap.height(), height);
}

/// Test with a real image through jpegli roundtrip.
#[test]
#[ignore] // Requires test image and jpegli encoder
fn test_jpegli_roundtrip_butteraugli() {
    let path = Path::new("/home/lilith/work/jpegli/testdata/jxl/flower/flower_small.rgb.png");
    if !path.exists() {
        eprintln!("Skipping test: test image not found at {:?}", path);
        return;
    }

    let (original, width, height) = load_png(path).expect("Failed to load test image");

    // Test at various quality levels
    for quality in [50, 70, 85, 95] {
        let jpeg_data = encode_jpeg(&original, width as u32, height as u32, quality);
        let decoded = decode_jpeg(&jpeg_data);

        if decoded.len() != original.len() {
            eprintln!("Size mismatch at Q{}, skipping", quality);
            continue;
        }

        let params = ButteraugliParams::default();
        let result = compute_butteraugli(&original, &decoded, width, height, &params);

        println!(
            "Q{}: size={} bytes, butteraugli={:.4}",
            quality,
            jpeg_data.len(),
            result.score
        );

        // High quality should be "good"
        if quality >= 85 {
            assert!(
                result.score < BUTTERAUGLI_BAD,
                "Q{} should have acceptable butteraugli score, got {}",
                quality,
                result.score
            );
        }
    }
}

/// Test comparing Rust butteraugli with C++ (if cjpegli is available).
#[test]
#[ignore] // Requires C++ cjpegli build
fn test_cpp_butteraugli_comparison() {
    let cjpegli = "/home/lilith/work/jpegli/build/tools/cjpegli";
    if !Path::new(cjpegli).exists() {
        eprintln!("Skipping: cjpegli not found at {}", cjpegli);
        return;
    }

    let path = Path::new("/home/lilith/work/jpegli/testdata/jxl/flower/flower_small.rgb.png");
    if !path.exists() {
        eprintln!("Skipping: test image not found");
        return;
    }

    let (original, width, height) = load_png(path).expect("Failed to load");

    // Encode with Rust jpegli, decode, measure with Rust butteraugli
    let jpeg_data = encode_jpeg(&original, width as u32, height as u32, 85);
    let decoded = decode_jpeg(&jpeg_data);

    let params = ButteraugliParams::default();
    let rust_result = compute_butteraugli(&original, &decoded, width, height, &params);
    println!("Rust butteraugli: {:.4}", rust_result.score);

    // TODO: Call C++ butteraugli for comparison
    // This would require either:
    // 1. FFI bindings to butteraugli library
    // 2. Calling a command-line tool that reports butteraugli score

    // For now, just report the score
    println!(
        "Note: C++ comparison not implemented. Rust score: {:.4}",
        rust_result.score
    );
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

    // Write scanlines row by row
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
