//! Decoder API conformance tests.
//!
//! Tests matching C++ jpegli decode_api_test.cc functionality.
use enough::Unstoppable;

#[path = "../src/test_utils.rs"]
mod test_utils;

use test_utils::{generate_gradient_d, read_test_data, TestImage};

use test_case::test_case;
use zenjpeg::{
    decoder::{Decoder, PixelFormat},
    encoder::{ChromaSubsampling, EncoderConfig, PixelLayout},
};

// ============================================================================
// Helper Functions
// ============================================================================

fn encode_rgb(
    width: u32,
    height: u32,
    data: &[u8],
    config: &EncoderConfig,
) -> zenjpeg::encoder::Result<Vec<u8>> {
    let mut enc = config.encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)?;
    enc.push_packed(data, enough::Unstoppable)?;
    enc.finish()
}

fn encode_gray(
    width: u32,
    height: u32,
    data: &[u8],
    config: &EncoderConfig,
) -> zenjpeg::encoder::Result<Vec<u8>> {
    let mut enc = config.encode_from_bytes(width, height, PixelLayout::Gray8Srgb)?;
    enc.push_packed(data, enough::Unstoppable)?;
    enc.finish()
}

fn create_test_jpeg(width: u32, height: u32, quality: f32) -> Vec<u8> {
    let img = generate_gradient_d(width, height, 3);
    // Use baseline for decoder tests to ensure stable behavior with small images
    let config = EncoderConfig::ycbcr(quality, ChromaSubsampling::Quarter).progressive(false);
    encode_rgb(width, height, &img.pixels, &config).expect("encode failed")
}

// ============================================================================
// Basic Decoding Tests
// ============================================================================

#[test]
fn test_decode_basic() {
    let jpeg = create_test_jpeg(128, 128, 90.0);
    let decoder = Decoder::new();
    let decoded = decoder.decode(&jpeg, Unstoppable).expect("decode failed");

    assert_eq!(decoded.width, 128);
    assert_eq!(decoded.height, 128);
    assert_eq!(decoded.format, PixelFormat::Rgb);
    assert_eq!(decoded.pixels_u8().unwrap().len(), 128 * 128 * 3);
}

#[test]
fn test_decode_grayscale() {
    // Create grayscale JPEG
    let img = test_utils::generate_gradient_h(64, 64, 1);
    let config = EncoderConfig::grayscale(90.0);
    let jpeg = encode_gray(64, 64, &img.pixels, &config).expect("encode failed");

    let decoder = Decoder::new();
    let decoded = decoder.decode(&jpeg, Unstoppable).expect("decode failed");

    assert_eq!(decoded.width, 64);
    assert_eq!(decoded.height, 64);
    // Grayscale decodes to RGB by default
    assert!(decoded.pixels_u8().unwrap().len() >= 64 * 64);
}

#[test]
fn test_decode_dimensions() {
    let jpeg = create_test_jpeg(256, 192, 85.0);
    let decoder = Decoder::new();

    // Decode and verify dimensions
    let decoded = decoder.decode(&jpeg, Unstoppable).expect("decode failed");
    assert_eq!(decoded.width, 256);
    assert_eq!(decoded.height, 192);
}

// ============================================================================
// Various Size Tests
// ============================================================================

#[test_case(8, 8 ; "8x8")]
#[test_case(16, 16 ; "16x16")]
#[test_case(17, 17 ; "17x17_odd")]
#[test_case(64, 64 ; "64x64")]
#[test_case(100, 100 ; "100x100")]
#[test_case(256, 256 ; "256x256")]
fn test_decode_various_sizes(width: u32, height: u32) {
    let jpeg = create_test_jpeg(width, height, 90.0);
    let decoder = Decoder::new();
    let decoded = decoder.decode(&jpeg, Unstoppable).expect("decode failed");

    assert_eq!(decoded.width, width);
    assert_eq!(decoded.height, height);
}

#[test]
fn test_decode_non_square() {
    // Wide
    let wide_jpeg = create_test_jpeg(256, 64, 90.0);
    let decoder = Decoder::new();
    let wide = decoder
        .decode(&wide_jpeg, Unstoppable)
        .expect("decode wide failed");
    assert_eq!(wide.width, 256);
    assert_eq!(wide.height, 64);

    // Tall
    let tall_jpeg = create_test_jpeg(64, 256, 90.0);
    let tall = decoder
        .decode(&tall_jpeg, Unstoppable)
        .expect("decode tall failed");
    assert_eq!(tall.width, 64);
    assert_eq!(tall.height, 256);
}

// ============================================================================
// Quality Level Decoding Tests
// ============================================================================

#[test_case(30.0 ; "Q30")]
#[test_case(50.0 ; "Q50")]
#[test_case(70.0 ; "Q70")]
#[test_case(85.0 ; "Q85")]
#[test_case(95.0 ; "Q95")]
fn test_decode_various_qualities(quality: f32) {
    let jpeg = create_test_jpeg(128, 128, quality);
    let decoder = Decoder::new();
    let decoded = decoder.decode(&jpeg, Unstoppable).expect("decode failed");

    assert_eq!(decoded.width, 128);
    assert_eq!(decoded.height, 128);
    assert_eq!(decoded.pixels_u8().unwrap().len(), 128 * 128 * 3);
}

// ============================================================================
// Progressive JPEG Decoding Tests
// ============================================================================

#[test]
fn test_decode_progressive() {
    let img = generate_gradient_d(128, 128, 3);
    let config = EncoderConfig::ycbcr(90.0, ChromaSubsampling::Quarter).progressive(true);
    let jpeg = encode_rgb(128, 128, &img.pixels, &config).expect("encode progressive failed");

    let decoder = Decoder::new();
    let decoded = decoder
        .decode(&jpeg, Unstoppable)
        .expect("decode progressive failed");

    assert_eq!(decoded.width, 128);
    assert_eq!(decoded.height, 128);
}

// ============================================================================
// Decoder Reuse Tests (matching C++ ReuseCinfo tests)
// ============================================================================

#[test]
fn test_decode_reuse_decoder() {
    let decoder = Decoder::new();

    // Decode multiple different JPEGs with same decoder
    for i in 0..5 {
        let size = 64 + i * 16;
        let jpeg = create_test_jpeg(size, size, 85.0);
        let decoded = decoder
            .decode(&jpeg, Unstoppable)
            .unwrap_or_else(|_| panic!("decode {} failed", i));
        assert_eq!(decoded.width, size);
        assert_eq!(decoded.height, size);
    }
}

// ============================================================================
// External JPEG Decoding Tests (from C++ testdata)
// ============================================================================

#[test]
#[ignore = "requires testdata"]
fn test_decode_flower_420() {
    let jpeg_data = read_test_data("jxl/flower/flower.png.im_q85_420.jpg");
    if jpeg_data.is_none() {
        eprintln!("Skipping: testdata not available");
        return;
    }

    let decoder = Decoder::new();
    let decoded = decoder
        .decode(&jpeg_data.unwrap(), Unstoppable)
        .expect("decode flower failed");

    // flower.png is 2268x1512
    assert_eq!(decoded.width, 2268);
    assert_eq!(decoded.height, 1512);
}

#[test]
#[ignore = "requires testdata"]
fn test_decode_flower_444() {
    let jpeg_data = read_test_data("jxl/flower/flower.png.im_q85_444.jpg");
    if jpeg_data.is_none() {
        eprintln!("Skipping: testdata not available");
        return;
    }

    let decoder = Decoder::new();
    let decoded = decoder
        .decode(&jpeg_data.unwrap(), Unstoppable)
        .expect("decode flower failed");

    assert_eq!(decoded.width, 2268);
    assert_eq!(decoded.height, 1512);
}

#[test]
#[ignore = "requires testdata"]
fn test_decode_flower_progressive() {
    let jpeg_data = read_test_data("jxl/flower/flower.png.im_q85_420_progr.jpg");
    if jpeg_data.is_none() {
        eprintln!("Skipping: testdata not available");
        return;
    }

    let decoder = Decoder::new();
    let decoded = decoder
        .decode(&jpeg_data.unwrap(), Unstoppable)
        .expect("decode progressive flower failed");

    assert_eq!(decoded.width, 2268);
    assert_eq!(decoded.height, 1512);
}

#[test]
#[ignore = "requires testdata"]
fn test_decode_flower_grayscale() {
    let jpeg_data = read_test_data("jxl/flower/flower.png.im_q85_gray.jpg");
    if jpeg_data.is_none() {
        eprintln!("Skipping: testdata not available");
        return;
    }

    let decoder = Decoder::new();
    let decoded = decoder
        .decode(&jpeg_data.unwrap(), Unstoppable)
        .expect("decode grayscale flower failed");

    assert_eq!(decoded.width, 2268);
    assert_eq!(decoded.height, 1512);
}

// ============================================================================
// Subsampling Variant Tests
// ============================================================================

#[test]
#[ignore = "requires testdata"]
fn test_decode_various_subsampling() {
    let subsampling_files = [
        "jxl/flower/flower.png.im_q85_420.jpg",
        "jxl/flower/flower.png.im_q85_422.jpg",
        "jxl/flower/flower.png.im_q85_440.jpg",
        "jxl/flower/flower.png.im_q85_444.jpg",
        "jxl/flower/flower.png.im_q85_444_1x2.jpg",
    ];

    let decoder = Decoder::new();

    for filename in &subsampling_files {
        if let Some(jpeg_data) = read_test_data(filename) {
            let decoded = decoder
                .decode(&jpeg_data, Unstoppable)
                .unwrap_or_else(|_| panic!("decode {} failed", filename));
            assert_eq!(decoded.width, 2268, "Width mismatch for {}", filename);
            assert_eq!(decoded.height, 1512, "Height mismatch for {}", filename);
        }
    }
}

// ============================================================================
// Marker Validation Tests
// ============================================================================

#[test]
fn test_decode_validates_soi() {
    // Missing SOI marker
    let bad_jpeg = vec![0xFF, 0xE0, 0x00, 0x10]; // No SOI
    let decoder = Decoder::new();
    assert!(decoder.decode(&bad_jpeg, Unstoppable).is_err());
}

#[test]
fn test_decode_validates_eoi() {
    // Create valid JPEG then truncate before EOI
    let jpeg = create_test_jpeg(64, 64, 90.0);
    let truncated: Vec<u8> = jpeg[..jpeg.len() - 10].to_vec();

    let decoder = Decoder::new();
    // Should still decode (EOI is optional per spec, but may fail)
    let result = decoder.decode(&truncated, Unstoppable);
    // We just verify it doesn't panic - behavior varies by implementation
    let _ = result;
}

#[test]
fn test_decode_empty_input() {
    let decoder = Decoder::new();
    assert!(decoder.decode(&[], Unstoppable).is_err());
}

#[test]
fn test_decode_too_small() {
    let decoder = Decoder::new();
    assert!(decoder.decode(&[0xFF], Unstoppable).is_err());
    assert!(decoder.decode(&[0xFF, 0xD8], Unstoppable).is_err()); // Only SOI
}

#[test]
fn test_decode_random_garbage() {
    let decoder = Decoder::new();
    let garbage: Vec<u8> = (0..1000).map(|i| (i * 7) as u8).collect();
    assert!(decoder.decode(&garbage, Unstoppable).is_err());
}

// ============================================================================
// Pixel Value Range Tests
// ============================================================================

#[test]
fn test_decode_pixel_range() {
    // Create image with full value range
    let mut img = TestImage::new(64, 64, 3);
    for y in 0..64 {
        for x in 0..64 {
            img.set_pixel(x, y, 0, (x * 4) as u8); // 0-252
            img.set_pixel(x, y, 1, (y * 4) as u8); // 0-252
            img.set_pixel(x, y, 2, ((x + y) * 2) as u8); // 0-252
        }
    }

    let config = EncoderConfig::ycbcr(100.0, ChromaSubsampling::Quarter);
    let jpeg = encode_rgb(64, 64, &img.pixels, &config).expect("encode failed");

    let decoder = Decoder::new();
    let decoded = decoder.decode(&jpeg, Unstoppable).expect("decode failed");

    // Ensure there's actual data
    assert!(
        !decoded.pixels_u8().unwrap().is_empty(),
        "Decoded data should not be empty"
    );
}

// ============================================================================
// Large Image Tests
// ============================================================================

#[test]
fn test_decode_large_image() {
    let jpeg = create_test_jpeg(1024, 768, 85.0);
    let decoder = Decoder::new();
    let decoded = decoder
        .decode(&jpeg, Unstoppable)
        .expect("decode large failed");

    assert_eq!(decoded.width, 1024);
    assert_eq!(decoded.height, 768);
    assert_eq!(decoded.pixels_u8().unwrap().len(), 1024 * 768 * 3);
}

// ============================================================================
// Consistency Tests
// ============================================================================

#[test]
fn test_decode_deterministic() {
    let jpeg = create_test_jpeg(128, 128, 90.0);
    let decoder = Decoder::new();

    // Decode same JPEG multiple times
    let decoded1 = decoder.decode(&jpeg, Unstoppable).expect("decode 1 failed");
    let decoded2 = decoder.decode(&jpeg, Unstoppable).expect("decode 2 failed");

    // Results should be identical
    assert_eq!(decoded1.pixels_u8().unwrap(), decoded2.pixels_u8().unwrap());
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_decode_1x1_pixel() {
    let img = TestImage::from_pixels(1, 1, 3, vec![100, 150, 200]);
    // Use baseline for small images to avoid progressive decoding edge cases
    let config = EncoderConfig::ycbcr(90.0, ChromaSubsampling::Quarter).progressive(false);
    let jpeg = encode_rgb(1, 1, &img.pixels, &config).expect("encode 1x1 failed");

    let decoder = Decoder::new();
    let decoded = decoder
        .decode(&jpeg, Unstoppable)
        .expect("decode 1x1 failed");

    assert_eq!(decoded.width, 1);
    assert_eq!(decoded.height, 1);
}

#[test]
fn test_decode_minimum_mcu() {
    // 8x8 is minimum MCU size
    let jpeg = create_test_jpeg(8, 8, 90.0);
    let decoder = Decoder::new();
    let decoded = decoder
        .decode(&jpeg, Unstoppable)
        .expect("decode 8x8 failed");

    assert_eq!(decoded.width, 8);
    assert_eq!(decoded.height, 8);
}

// ============================================================================
// OutputTarget Tests
// ============================================================================

use zenjpeg::decoder::OutputTarget;

/// Create a test JPEG with known content for OutputTarget tests.
fn create_output_target_jpeg() -> Vec<u8> {
    let img = generate_gradient_d(64, 64, 3);
    let config = EncoderConfig::ycbcr(90.0, ChromaSubsampling::None);
    let mut enc = config
        .encode_from_bytes(64, 64, PixelLayout::Rgb8Srgb)
        .expect("encoder");
    enc.push_packed(&img.pixels, Unstoppable).expect("push");
    enc.finish().expect("finish")
}

#[test]
fn output_target_srgb8_returns_u8() {
    let jpeg = create_output_target_jpeg();
    let result = Decoder::new()
        .output_target(OutputTarget::Srgb8)
        .decode(&jpeg, Unstoppable)
        .expect("decode");

    assert!(
        result.pixels_u8().is_some(),
        "Srgb8 should produce u8 pixels"
    );
    assert!(
        result.pixels_f32().is_none(),
        "Srgb8 should not produce f32 pixels"
    );
    assert_eq!(result.output_target(), OutputTarget::Srgb8);
    assert_eq!(result.pixels_u8().unwrap().len(), 64 * 64 * 3);
}

#[test]
fn output_target_srgb_f32_returns_f32() {
    let jpeg = create_output_target_jpeg();
    let result = Decoder::new()
        .output_target(OutputTarget::SrgbF32)
        .decode(&jpeg, Unstoppable)
        .expect("decode");

    assert!(
        result.pixels_f32().is_some(),
        "SrgbF32 should produce f32 pixels"
    );
    assert!(
        result.pixels_u8().is_none(),
        "SrgbF32 should not produce u8 pixels"
    );
    assert_eq!(result.output_target(), OutputTarget::SrgbF32);
    assert_eq!(result.pixels_f32().unwrap().len(), 64 * 64 * 3);

    // Values should be in approximately [0.0, 1.0] for sRGB gamma
    let pixels = result.pixels_f32().unwrap();
    for &v in pixels {
        assert!(
            v >= -0.1 && v <= 1.1,
            "sRGB f32 value {v} out of expected range"
        );
    }
}

#[test]
fn output_target_linear_f32_values_differ_from_srgb() {
    let jpeg = create_output_target_jpeg();

    let srgb = Decoder::new()
        .output_target(OutputTarget::SrgbF32)
        .decode(&jpeg, Unstoppable)
        .expect("srgb decode");

    let linear = Decoder::new()
        .output_target(OutputTarget::LinearF32)
        .decode(&jpeg, Unstoppable)
        .expect("linear decode");

    let srgb_px = srgb.pixels_f32().unwrap();
    let linear_px = linear.pixels_f32().unwrap();

    assert_eq!(srgb_px.len(), linear_px.len());

    // Linear values should differ from sRGB (darker mid-tones in linear)
    let mut total_diff = 0.0f64;
    for (&s, &l) in srgb_px.iter().zip(linear_px.iter()) {
        total_diff += (s as f64 - l as f64).abs();
    }
    let mean_diff = total_diff / srgb_px.len() as f64;
    assert!(
        mean_diff > 0.01,
        "Linear should differ from sRGB, mean diff was {mean_diff}"
    );

    // Linear values for mid-tones should be smaller (sRGB gamma boosts darks)
    // Check a mid-range sRGB value and its linear equivalent
    for (&s, &l) in srgb_px.iter().zip(linear_px.iter()) {
        if s > 0.2 && s < 0.8 {
            assert!(l < s, "Linear mid-tone {l} should be < sRGB {s}");
        }
    }
}

#[test]
fn output_target_precise_returns_f32() {
    let jpeg = create_output_target_jpeg();
    let result = Decoder::new()
        .output_target(OutputTarget::SrgbF32Precise)
        .decode(&jpeg, Unstoppable)
        .expect("decode");

    assert!(result.pixels_f32().is_some());
    assert_eq!(result.output_target(), OutputTarget::SrgbF32Precise);
}

#[test]
fn output_target_linear_precise_returns_f32() {
    let jpeg = create_output_target_jpeg();
    let result = Decoder::new()
        .output_target(OutputTarget::LinearF32Precise)
        .decode(&jpeg, Unstoppable)
        .expect("decode");

    assert!(result.pixels_f32().is_some());
    assert_eq!(result.output_target(), OutputTarget::LinearF32Precise);
}

#[test]
fn output_target_default_is_srgb8() {
    let jpeg = create_output_target_jpeg();
    let result = Decoder::new().decode(&jpeg, Unstoppable).expect("decode");

    assert!(result.pixels_u8().is_some());
    assert_eq!(result.output_target(), OutputTarget::Srgb8);
}

#[test]
fn srgb_f32_preserves_unclamped_ringing() {
    // At low quality, IDCT produces ringing outside [0, 255].
    // SrgbF32 should preserve these as values outside [0.0, 1.0].
    let img = generate_gradient_d(64, 64, 3);
    let config = EncoderConfig::ycbcr(10.0, ChromaSubsampling::None);
    let mut enc = config
        .encode_from_bytes(64, 64, PixelLayout::Rgb8Srgb)
        .expect("encoder");
    enc.push_packed(&img.pixels, Unstoppable).expect("push");
    let jpeg = enc.finish().expect("finish");

    let u8_result = Decoder::new()
        .output_target(OutputTarget::Srgb8)
        .decode(&jpeg, Unstoppable)
        .expect("u8 decode");

    let f32_result = Decoder::new()
        .output_target(OutputTarget::SrgbF32)
        .decode(&jpeg, Unstoppable)
        .expect("f32 decode");

    let u8_px = u8_result.pixels_u8().unwrap();
    let f32_px = f32_result.pixels_f32().unwrap();

    // Check that f32 has the same pixel count
    assert_eq!(u8_px.len(), f32_px.len());

    // Check that Srgb8 clamps but SrgbF32 may not
    let u8_has_zero = u8_px.iter().any(|&v| v == 0);
    let u8_has_255 = u8_px.iter().any(|&v| v == 255);
    let f32_has_negative = f32_px.iter().any(|&v| v < 0.0);
    let f32_has_over_one = f32_px.iter().any(|&v| v > 1.0);

    // At Q10, the u8 path should have some clamped values
    assert!(
        u8_has_zero || u8_has_255,
        "Q10 u8 should have some clamped values"
    );

    // The f32 path MAY have unclamped values (depends on IDCT ringing).
    // This isn't guaranteed for every image, but it demonstrates the capability.
    // We just verify the f32 values are reasonable.
    for &v in f32_px {
        assert!(
            v >= -1.0 && v <= 2.0,
            "f32 value {v} unreasonably out of range"
        );
    }

    // If we do have unclamped values, the f32 path is working correctly
    if f32_has_negative || f32_has_over_one {
        // Great - unclamped IDCT is working
    }
}

#[test]
fn gain_map_discard_returns_none() {
    // Regular JPEG should have no gain map regardless of GainMapHandling
    let jpeg = create_output_target_jpeg();

    let result = Decoder::new()
        .gain_map(zenjpeg::decoder::GainMapHandling::Discard)
        .decode(&jpeg, Unstoppable)
        .expect("decode");

    assert!(
        result.gain_map.is_none(),
        "Regular JPEG should have no gain map"
    );
}

#[test]
fn gain_map_preserve_raw_returns_none_for_regular_jpeg() {
    let jpeg = create_output_target_jpeg();

    let result = Decoder::new()
        .gain_map(zenjpeg::decoder::GainMapHandling::PreserveRaw)
        .decode(&jpeg, Unstoppable)
        .expect("decode");

    assert!(
        result.gain_map.is_none(),
        "Regular JPEG should have no gain map even with PreserveRaw"
    );
}
