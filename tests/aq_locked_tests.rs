//! Adaptive Quantization Locked Tests
//!
//! THESE TESTS MUST NEVER BE:
//! - Marked as `#[ignore]`
//! - Deleted
//! - Have their assertions weakened
//!
//! If these tests fail, the AQ implementation is BROKEN.
//! Fix the implementation, not the tests.
//!
//! Adapted from jpegli-rs/tests/aq_locked_tests.rs

use zenjpeg::adaptive_quant::{AdaptiveQuantConfig, AqField};
use zenjpeg::{Encoder, Quality};

/// Test that AQ field values are in the expected range.
/// The multiplier values should be around 1.0 (0.5 to 1.5 range).
#[test]
fn test_aq_field_value_range() {
    // Create an AQ field
    let field = AqField::uniform(10, 10);

    // Uniform field should have all values at 1.0
    for &mult in &field.multipliers {
        assert!(
            mult >= 0.5 && mult <= 1.5,
            "AQ multiplier {} outside expected range [0.5, 1.5]",
            mult
        );
    }
}

/// Test that AQ config defaults are sensible.
#[test]
fn test_aq_config_defaults() {
    let config = AdaptiveQuantConfig::default();

    // AQ should be disabled by default until properly tuned
    assert!(!config.enabled, "AQ should be disabled by default");

    // Strength should be 1.0 (full strength when enabled)
    assert!(
        (config.strength - 1.0).abs() < 1e-6,
        "Default strength should be 1.0, got {}",
        config.strength
    );
}

/// Test that quality-to-distance conversion is monotonic.
/// Higher quality should produce smaller files at same distortion.
#[test]
fn test_quality_monotonic() {
    let width = 64;
    let height = 64;

    // Create a test image with content
    let mut rgb = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let r = ((x * 4) % 256) as u8;
            let g = ((y * 4) % 256) as u8;
            let b = (((x + y) * 2) % 256) as u8;
            rgb.push(r);
            rgb.push(g);
            rgb.push(b);
        }
    }

    let mut prev_size = 0usize;

    for quality in [30, 50, 70, 85, 95] {
        let jpeg = Encoder::new()
            .quality(Quality::Standard(quality))
            .encode_rgb(&rgb, width, height)
            .expect("encoding failed");

        if quality > 30 {
            // Higher quality should produce larger files
            assert!(
                jpeg.len() >= prev_size,
                "Q{} size ({}) should be >= Q{} size ({})",
                quality,
                jpeg.len(),
                quality - 20,
                prev_size
            );
        }

        prev_size = jpeg.len();
    }
}

/// Test that encoding with AQ enabled produces valid output.
/// This test ensures the encoder doesn't crash or produce garbage.
#[test]
fn test_encoding_with_aq_valid() {
    // Create a simple test image
    let width = 64;
    let height = 64;
    let rgb: Vec<u8> = (0..width * height * 3)
        .map(|i| ((i * 7) % 256) as u8)
        .collect();

    // Encode at high quality (where AQ would typically be active)
    let jpeg_data = Encoder::max_quality()
        .quality(Quality::High(90))
        .encode_rgb(&rgb, width, height)
        .expect("encoding failed");

    // Verify output is valid JPEG
    assert!(
        jpeg_data.len() > 100,
        "JPEG too small: {} bytes",
        jpeg_data.len()
    );
    assert_eq!(&jpeg_data[0..2], &[0xFF, 0xD8], "Missing JPEG SOI marker");
    assert_eq!(
        &jpeg_data[jpeg_data.len() - 2..],
        &[0xFF, 0xD9],
        "Missing JPEG EOI marker"
    );

    // Decode and verify pixels are reasonable
    let mut decoder = jpeg_decoder::Decoder::new(&jpeg_data[..]);
    let decoded = decoder.decode().expect("decode failed");
    let info = decoder.info().unwrap();

    assert_eq!(info.width, width as u16);
    assert_eq!(info.height, height as u16);

    // Check decoded pixels are not all zeros or all 255
    let sum: u64 = decoded.iter().map(|&v| v as u64).sum();
    let avg = sum as f64 / decoded.len() as f64;
    assert!(
        avg > 10.0 && avg < 245.0,
        "Decoded average {} suggests encoding failure",
        avg
    );
}

/// Test that encoding at different quality levels produces different sizes.
#[test]
fn test_quality_affects_size() {
    let width = 128;
    let height = 128;
    let rgb: Vec<u8> = (0..width * height * 3)
        .map(|i| ((i * 13 + i / 7) % 256) as u8)
        .collect();

    let encode_at_quality = |q: u8| -> usize {
        Encoder::new()
            .quality(Quality::Standard(q))
            .encode_rgb(&rgb, width, height)
            .expect("encoding failed")
            .len()
    };

    let size_q60 = encode_at_quality(60);
    let size_q75 = encode_at_quality(75);
    let size_q90 = encode_at_quality(90);
    let size_q95 = encode_at_quality(95);

    // Higher quality should produce larger files
    assert!(
        size_q60 < size_q75,
        "Q60 ({}) should be smaller than Q75 ({})",
        size_q60,
        size_q75
    );
    assert!(
        size_q75 < size_q90,
        "Q75 ({}) should be smaller than Q90 ({})",
        size_q75,
        size_q90
    );
    assert!(
        size_q90 < size_q95,
        "Q90 ({}) should be smaller than Q95 ({})",
        size_q90,
        size_q95
    );
}

/// Test smooth blocks get higher AQ multiplier (more compression).
#[test]
fn test_smooth_blocks_higher_multiplier() {
    use zenjpeg::adaptive_quant::compute_aq_field;

    // Create a uniform (smooth) image
    let smooth_image: Vec<u8> = vec![128; 64 * 64];
    let config = AdaptiveQuantConfig {
        enabled: true,
        strength: 1.0,
    };

    let field = compute_aq_field(&smooth_image, 64, 64, &config);

    // Smooth regions should have multiplier >= 1.0 (stronger compression)
    let avg_mult: f32 = field.multipliers.iter().sum::<f32>() / field.multipliers.len() as f32;

    assert!(
        avg_mult >= 1.0,
        "Smooth image should have avg multiplier >= 1.0, got {}",
        avg_mult
    );
}

/// Test high-detail blocks get lower AQ multiplier (less compression).
#[test]
fn test_detailed_blocks_lower_multiplier() {
    use zenjpeg::adaptive_quant::compute_aq_field;

    // Create a high-contrast checkerboard pattern
    let mut detailed_image = Vec::with_capacity(64 * 64);
    for y in 0..64 {
        for x in 0..64 {
            // 2x2 checkerboard for maximum local contrast
            let checker = ((x / 2) + (y / 2)) % 2;
            detailed_image.push(if checker == 0 { 50 } else { 200 });
        }
    }

    let config = AdaptiveQuantConfig {
        enabled: true,
        strength: 1.0,
    };

    let field = compute_aq_field(&detailed_image, 64, 64, &config);

    // Detailed regions should have multiplier <= 1.0 (less compression)
    let avg_mult: f32 = field.multipliers.iter().sum::<f32>() / field.multipliers.len() as f32;

    assert!(
        avg_mult <= 1.0,
        "Detailed image should have avg multiplier <= 1.0, got {}",
        avg_mult
    );
}

/// Test AQ field dimensions are correct.
#[test]
fn test_aq_field_dimensions() {
    use zenjpeg::adaptive_quant::compute_aq_field;

    let config = AdaptiveQuantConfig {
        enabled: true,
        strength: 1.0,
    };

    // Test various image sizes
    for (width, height) in [(64, 64), (100, 100), (123, 87), (256, 128)] {
        let image: Vec<u8> = vec![128; width * height];
        let field = compute_aq_field(&image, width, height, &config);

        let expected_w = (width + 7) / 8;
        let expected_h = (height + 7) / 8;

        assert_eq!(
            field.width_blocks, expected_w,
            "Width blocks mismatch for {}x{}",
            width, height
        );
        assert_eq!(
            field.height_blocks, expected_h,
            "Height blocks mismatch for {}x{}",
            width, height
        );
        assert_eq!(
            field.multipliers.len(),
            expected_w * expected_h,
            "Total multipliers mismatch for {}x{}",
            width,
            height
        );
    }
}

/// Test that AQ disabled returns uniform field.
#[test]
fn test_aq_disabled_uniform() {
    use zenjpeg::adaptive_quant::compute_aq_field;

    let config = AdaptiveQuantConfig {
        enabled: false,
        strength: 1.0,
    };

    // Create varied content
    let mut image = Vec::with_capacity(64 * 64);
    for i in 0..64 * 64 {
        image.push((i % 256) as u8);
    }

    let field = compute_aq_field(&image, 64, 64, &config);

    // When disabled, all multipliers should be exactly 1.0
    for &mult in &field.multipliers {
        let diff: f32 = (mult - 1.0).abs();
        assert!(
            diff < 1e-6,
            "Disabled AQ should have all multipliers = 1.0, got {}",
            mult
        );
    }
}

// ============================================================================
// Future Tests for Full jpegli AQ Implementation
// ============================================================================

/// Placeholder for C++ AQ strength range test.
/// When full AQ is implemented, aq_strength should be in 0.0-0.2 range (C++ documented).
#[test]
#[ignore] // Enable when full jpegli AQ is implemented
fn test_aq_strength_range_cpp() {
    // C++ produces values in 0.0-0.2 range with mean ~0.08
    // This test should verify our implementation matches

    // TODO: When per-block AQ is implemented:
    // 1. Compute AQ strength map using jpegli algorithm
    // 2. Assert all values are in [0.0, 0.2] range
    // 3. Assert mean is approximately 0.08
}

/// Placeholder for C++ testdata comparison.
#[test]
#[ignore] // Enable when full jpegli AQ is implemented
fn test_aq_vs_cpp_testdata() {
    // Check if testdata exists
    let testdata_path = "/home/lilith/work/jpegli/ComputeAdaptiveQuantField.testdata";
    if !std::path::Path::new(testdata_path).exists() {
        eprintln!("C++ testdata not found at {}. Skipping test.", testdata_path);
        return;
    }

    // TODO: When per-block AQ is implemented:
    // 1. Load ComputeAdaptiveQuantField.testdata
    // 2. Run Rust AQ on same input
    // 3. Compare output to expected_quant_field_slice
    // 4. Assert max difference < 0.025 (per jpegli tolerance)
}
