//! Comparison tests for deringing vs normal encoding.
//!
//! Evaluates the impact of overshoot deringing on compression and quality
//! using different image types.

use zenjpeg::{Encoder, Quality};

/// Test that deringing can be explicitly enabled/disabled
#[test]
fn test_deringing_toggle() {
    // Create a test image with a white background and sharp edges
    // (exactly the type of image where deringing helps most)
    let (width, height) = (64, 64);
    let mut pixels = vec![255u8; width * height * 3]; // White background

    // Add a dark rectangle in the center (creates hard edges)
    for y in 16..48 {
        for x in 16..48 {
            let idx = (y * width + x) * 3;
            pixels[idx] = 50; // R
            pixels[idx + 1] = 50; // G
            pixels[idx + 2] = 50; // B
        }
    }

    // Encode with deringing enabled
    let encoder_with = Encoder::new()
        .quality(Quality::Standard(75))
        .overshoot_deringing(true);
    let result_with = encoder_with.encode_rgb(&pixels, width, height);
    assert!(
        result_with.is_ok(),
        "Encoding with deringing should succeed"
    );
    let jpeg_with = result_with.unwrap();

    // Encode with deringing disabled
    let encoder_without = Encoder::new()
        .quality(Quality::Standard(75))
        .overshoot_deringing(false);
    let result_without = encoder_without.encode_rgb(&pixels, width, height);
    assert!(
        result_without.is_ok(),
        "Encoding without deringing should succeed"
    );
    let jpeg_without = result_without.unwrap();

    // Both should produce valid JPEGs
    assert!(
        jpeg_with.len() > 100,
        "JPEG with deringing should be reasonable size"
    );
    assert!(
        jpeg_without.len() > 100,
        "JPEG without deringing should be reasonable size"
    );

    // The sizes may differ (deringing can improve compression on white backgrounds)
    println!("With deringing: {} bytes", jpeg_with.len());
    println!("Without deringing: {} bytes", jpeg_without.len());
}

/// Test deringing on grayscale images
#[test]
fn test_deringing_grayscale() {
    let (width, height) = (32, 32);
    let mut pixels = vec![255u8; width * height]; // White background

    // Add a dark square
    for y in 8..24 {
        for x in 8..24 {
            pixels[y * width + x] = 50;
        }
    }

    // Encode with deringing
    let encoder = Encoder::new()
        .quality(Quality::Standard(75))
        .overshoot_deringing(true);
    let result = encoder.encode_gray(&pixels, width, height);
    assert!(
        result.is_ok(),
        "Grayscale encoding with deringing should succeed"
    );

    // Encode without deringing
    let encoder = Encoder::new()
        .quality(Quality::Standard(75))
        .overshoot_deringing(false);
    let result = encoder.encode_gray(&pixels, width, height);
    assert!(
        result.is_ok(),
        "Grayscale encoding without deringing should succeed"
    );
}

/// Test that auto-selection works correctly
/// (deringing should be enabled for mozjpeg strategy, disabled for jpegli)
#[test]
fn test_deringing_auto_selection() {
    let (width, height) = (32, 32);
    let pixels = vec![200u8; width * height * 3];

    // With mozjpeg strategy (should auto-enable deringing)
    let encoder = Encoder::new()
        .quality(Quality::Standard(60)) // Lower quality typically uses mozjpeg
        .strategy(zenjpeg::EncodingStrategy::Mozjpeg);
    let result = encoder.encode_rgb(&pixels, width, height);
    assert!(result.is_ok(), "Mozjpeg encoding should succeed");

    // With jpegli strategy (should auto-disable deringing)
    let encoder = Encoder::new()
        .quality(Quality::Standard(85))
        .strategy(zenjpeg::EncodingStrategy::Jpegli);
    let result = encoder.encode_rgb(&pixels, width, height);
    assert!(result.is_ok(), "Jpegli encoding should succeed");
}

/// Test that deringing works across quality levels
#[test]
fn test_deringing_quality_levels() {
    let (width, height) = (32, 32);
    let mut pixels = vec![255u8; width * height * 3]; // White

    // Add black text-like pattern
    for y in 10..22 {
        for x in 10..22 {
            if (x + y) % 4 < 2 {
                let idx = (y * width + x) * 3;
                pixels[idx] = 0;
                pixels[idx + 1] = 0;
                pixels[idx + 2] = 0;
            }
        }
    }

    for quality in [30, 50, 70, 85, 95] {
        let encoder = Encoder::new()
            .quality(Quality::Standard(quality))
            .overshoot_deringing(true);
        let result = encoder.encode_rgb(&pixels, width, height);
        assert!(
            result.is_ok(),
            "Quality {} with deringing should succeed",
            quality
        );

        let encoder = Encoder::new()
            .quality(Quality::Standard(quality))
            .overshoot_deringing(false);
        let result = encoder.encode_rgb(&pixels, width, height);
        assert!(
            result.is_ok(),
            "Quality {} without deringing should succeed",
            quality
        );
    }
}

/// Test that deringing doesn't break images without saturated regions
#[test]
fn test_deringing_no_saturated_pixels() {
    let (width, height) = (32, 32);

    // Mid-tone gradient (no pixels at 0 or 255)
    let mut pixels = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let val = 64 + ((x + y) * 128 / (width + height)) as u8;
            pixels.push(val);
            pixels.push(val);
            pixels.push(val);
        }
    }

    // Deringing should have no effect but shouldn't break anything
    let encoder = Encoder::new()
        .quality(Quality::Standard(75))
        .overshoot_deringing(true);
    let result = encoder.encode_rgb(&pixels, width, height);
    assert!(
        result.is_ok(),
        "Encoding mid-tone image with deringing should succeed"
    );
}

/// Test deringing with the max_compression preset
#[test]
fn test_deringing_max_compression_preset() {
    let (width, height) = (32, 32);
    let pixels = vec![240u8; width * height * 3];

    // max_compression enables deringing by default
    let encoder = Encoder::max_compression();
    let result = encoder.encode_rgb(&pixels, width, height);
    assert!(result.is_ok(), "max_compression preset should succeed");
}

/// Test deringing with the fastest preset
#[test]
fn test_deringing_fastest_preset() {
    let (width, height) = (32, 32);
    let pixels = vec![240u8; width * height * 3];

    // fastest disables deringing by default
    let encoder = Encoder::fastest();
    let result = encoder.encode_rgb(&pixels, width, height);
    assert!(result.is_ok(), "fastest preset should succeed");
}
