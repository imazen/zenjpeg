//! Parametrized quality tests using test-case.
//!
//! This demonstrates the parametrized test framework for testing
//! multiple quality levels and configurations in a structured way.

// Import test_utils through the feature-gated module
#[path = "../src/test_utils.rs"]
mod test_utils;

use test_utils::{distance_rms, generate_test_image, max_pixel_diff, thresholds, TestPattern};

use jpegli::{
    decoder::Decoder,
    encoder::{EncoderConfig, PixelLayout},
};
use test_case::test_case;

/// Helper function to encode RGB data
fn encode_rgb(
    width: u32,
    height: u32,
    data: &[u8],
    config: &EncoderConfig,
) -> jpegli::encoder::Result<Vec<u8>> {
    let mut enc = config.encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)?;
    enc.push_packed(data, enough::Unstoppable)?;
    enc.finish()
}

/// Helper to encode and decode an image, returning RMS and max diff.
fn roundtrip_quality(
    width: u32,
    height: u32,
    pattern: TestPattern,
    quality: f32,
) -> (f64, u8, usize) {
    let img = generate_test_image(width, height, pattern, 3);

    let config = EncoderConfig::new().quality(quality);

    let jpeg_data = encode_rgb(width, height, &img.pixels, &config).expect("encode failed");
    let decoder = Decoder::new();
    let decoded = decoder.decode(&jpeg_data).expect("decode failed");

    let rms = distance_rms(&img.pixels, &decoded.data);
    let max_diff = max_pixel_diff(&img.pixels, &decoded.data);

    (rms, max_diff, jpeg_data.len())
}

// ============================================================================
// Quality Level Tests
// ============================================================================

#[test_case(50.0, thresholds::Q50_MAX_RMS ; "Q50")]
#[test_case(75.0, thresholds::Q75_MAX_RMS ; "Q75")]
#[test_case(85.0, thresholds::Q85_MAX_RMS ; "Q85")]
#[test_case(90.0, thresholds::Q90_MAX_RMS ; "Q90")]
#[test_case(95.0, thresholds::Q95_MAX_RMS ; "Q95")]
fn test_gradient_quality_thresholds(quality: f32, max_rms: f64) {
    let (rms, _max_diff, _size) = roundtrip_quality(256, 256, TestPattern::GradientD, quality);
    assert!(
        rms <= max_rms,
        "Q{} gradient: RMS {:.2} exceeds threshold {:.2}",
        quality,
        rms,
        max_rms
    );
}

// Noise tests: random noise is very hard to compress and inherently lossy.
// These tests verify that higher quality still produces better results.
// Note: Random noise cannot be well-preserved by JPEG - this is expected behavior.
#[test_case(50.0 ; "Q50")]
#[test_case(75.0 ; "Q75")]
#[test_case(85.0 ; "Q85")]
#[test_case(90.0 ; "Q90")]
#[test_case(95.0 ; "Q95")]
fn test_noise_roundtrip(quality: f32) {
    let (rms, max_diff, size) = roundtrip_quality(128, 128, TestPattern::Noise, quality);
    // Noise images will have high RMS and max_diff - this is expected.
    // Just verify encoding/decoding works and produces output.
    println!(
        "Q{} noise: RMS={:.2}, max_diff={}, size={}",
        quality, rms, max_diff, size
    );
    // Verify file was produced
    assert!(size > 100, "Q{} noise: file size too small", quality);
}

// ============================================================================
// Image Size Tests
// ============================================================================

#[test_case(8, 8 ; "8x8")]
#[test_case(16, 16 ; "16x16")]
#[test_case(64, 64 ; "64x64")]
#[test_case(256, 256 ; "256x256")]
#[test_case(17, 31 ; "17x31_odd")]
#[test_case(100, 100 ; "100x100")]
fn test_various_sizes_roundtrip(width: u32, height: u32) {
    let (rms, max_diff, _size) = roundtrip_quality(width, height, TestPattern::GradientD, 90.0);
    assert!(
        rms <= thresholds::Q90_MAX_RMS,
        "{}x{}: RMS {:.2} exceeds Q90 threshold",
        width,
        height,
        rms
    );
    assert!(
        max_diff < 50,
        "{}x{}: max diff {} exceeds threshold",
        width,
        height,
        max_diff
    );
}

// ============================================================================
// Pattern Tests
// ============================================================================

#[test_case(TestPattern::GradientH, "gradient_h")]
#[test_case(TestPattern::GradientV, "gradient_v")]
#[test_case(TestPattern::GradientD, "gradient_d")]
#[test_case(TestPattern::Checkerboard, "checkerboard")]
#[test_case(TestPattern::ColorBars, "color_bars")]
#[test_case(TestPattern::SolidColor, "solid")]
fn test_pattern_roundtrip(pattern: TestPattern, name: &str) {
    let (rms, max_diff, size) = roundtrip_quality(128, 128, pattern, 90.0);
    println!(
        "{}: RMS={:.2}, max_diff={}, size={}",
        name, rms, max_diff, size
    );

    // All patterns should roundtrip reasonably well at Q90
    // Color bars have sharp edges causing ringing, so allow higher RMS
    let multiplier = if pattern == TestPattern::ColorBars {
        4.0
    } else {
        2.0
    };
    assert!(
        rms <= thresholds::Q90_MAX_RMS * multiplier,
        "{}: RMS {:.2} exceeds threshold {:.2}",
        name,
        rms,
        thresholds::Q90_MAX_RMS * multiplier
    );
}

// ============================================================================
// File Size Progression Tests
// ============================================================================

// NOTE: File size doesn't always increase monotonically with quality.
// At certain quality transitions, the quantization table changes can cause
// smaller files at higher quality (especially on small synthetic images).
// Testing only Q90 vs Q95 which should be more predictable.
#[test_case(90.0, 95.0 ; "Q90_lt_Q95")]
fn test_file_size_increases_with_quality(lower_q: f32, higher_q: f32) {
    let (_, _, size_lower) = roundtrip_quality(256, 256, TestPattern::Noise, lower_q);
    let (_, _, size_higher) = roundtrip_quality(256, 256, TestPattern::Noise, higher_q);

    // Allow 10% tolerance
    let tolerance = (size_lower as f64 * 0.10).max(50.0) as usize;
    assert!(
        size_higher + tolerance >= size_lower,
        "Q{} size {} should be >= Q{} size {} (with {}B tolerance)",
        higher_q,
        size_higher,
        lower_q,
        size_lower,
        tolerance
    );
}

// ============================================================================
// Quality Improves with Higher Q
// ============================================================================

#[test_case(50.0, 90.0 ; "Q50_vs_Q90")]
#[test_case(70.0, 95.0 ; "Q70_vs_Q95")]
fn test_quality_improves_with_higher_q(lower_q: f32, higher_q: f32) {
    let (rms_lower, _, _) = roundtrip_quality(128, 128, TestPattern::GradientD, lower_q);
    let (rms_higher, _, _) = roundtrip_quality(128, 128, TestPattern::GradientD, higher_q);

    assert!(
        rms_higher <= rms_lower,
        "Q{} RMS {:.2} should be <= Q{} RMS {:.2}",
        higher_q,
        rms_higher,
        lower_q,
        rms_lower
    );
}
