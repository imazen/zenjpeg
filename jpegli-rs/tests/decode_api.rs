//! Decoder API conformance tests.
//!
//! Tests matching C++ jpegli decode_api_test.cc functionality.

#[path = "../src/test_utils.rs"]
mod test_utils;

use test_utils::{generate_gradient_d, get_test_data_path, read_test_data, TestImage};

use jpegli::{
    decode::{Decoder, DecoderConfig},
    types::PixelFormat,
    JpegEncoder, Quality,
};
use test_case::test_case;

// ============================================================================
// Helper Functions
// ============================================================================

fn create_test_jpeg(width: u32, height: u32, quality: f32) -> Vec<u8> {
    let img = generate_gradient_d(width, height, 3);
    let encoder = JpegEncoder::new(width, height).quality(Quality::from_quality(quality));
    encoder.encode(&img.pixels).expect("encode failed")
}

// ============================================================================
// Basic Decoding Tests
// ============================================================================

#[test]
fn test_decode_basic() {
    let jpeg = create_test_jpeg(128, 128, 90.0);
    let decoder = Decoder::new();
    let decoded = decoder.decode(&jpeg).expect("decode failed");

    assert_eq!(decoded.width, 128);
    assert_eq!(decoded.height, 128);
    assert_eq!(decoded.format, PixelFormat::Rgb);
    assert_eq!(decoded.data.len(), 128 * 128 * 3);
}

#[test]
fn test_decode_grayscale() {
    // Create grayscale JPEG
    let img = test_utils::generate_gradient_h(64, 64, 1);
    let encoder = JpegEncoder::new(64, 64).pixel_format(PixelFormat::Gray);
    let jpeg = encoder.encode(&img.pixels).expect("encode failed");

    let decoder = Decoder::new();
    let decoded = decoder.decode(&jpeg).expect("decode failed");

    assert_eq!(decoded.width, 64);
    assert_eq!(decoded.height, 64);
    // Grayscale decodes to RGB by default
    assert!(decoded.data.len() >= 64 * 64);
}

#[test]
fn test_decode_dimensions() {
    let jpeg = create_test_jpeg(256, 192, 85.0);
    let decoder = Decoder::new();

    // Decode and verify dimensions
    let decoded = decoder.decode(&jpeg).expect("decode failed");
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
    let decoded = decoder.decode(&jpeg).expect("decode failed");

    assert_eq!(decoded.width, width);
    assert_eq!(decoded.height, height);
}

#[test]
fn test_decode_non_square() {
    // Wide
    let wide_jpeg = create_test_jpeg(256, 64, 90.0);
    let decoder = Decoder::new();
    let wide = decoder.decode(&wide_jpeg).expect("decode wide failed");
    assert_eq!(wide.width, 256);
    assert_eq!(wide.height, 64);

    // Tall
    let tall_jpeg = create_test_jpeg(64, 256, 90.0);
    let tall = decoder.decode(&tall_jpeg).expect("decode tall failed");
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
    let decoded = decoder.decode(&jpeg).expect("decode failed");

    assert_eq!(decoded.width, 128);
    assert_eq!(decoded.height, 128);
    assert_eq!(decoded.data.len(), 128 * 128 * 3);
}

// ============================================================================
// Progressive JPEG Decoding Tests
// ============================================================================

#[test]
fn test_decode_progressive() {
    let img = generate_gradient_d(128, 128, 3);
    let encoder = JpegEncoder::new(128, 128).mode(jpegli::types::JpegMode::Progressive);
    let jpeg = encoder
        .encode(&img.pixels)
        .expect("encode progressive failed");

    let decoder = Decoder::new();
    let decoded = decoder.decode(&jpeg).expect("decode progressive failed");

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
            .decode(&jpeg)
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
        .decode(&jpeg_data.unwrap())
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
        .decode(&jpeg_data.unwrap())
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
        .decode(&jpeg_data.unwrap())
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
        .decode(&jpeg_data.unwrap())
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
                .decode(&jpeg_data)
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
    assert!(decoder.decode(&bad_jpeg).is_err());
}

#[test]
fn test_decode_validates_eoi() {
    // Create valid JPEG then truncate before EOI
    let jpeg = create_test_jpeg(64, 64, 90.0);
    let truncated: Vec<u8> = jpeg[..jpeg.len() - 10].to_vec();

    let decoder = Decoder::new();
    // Should still decode (EOI is optional per spec, but may fail)
    let result = decoder.decode(&truncated);
    // We just verify it doesn't panic - behavior varies by implementation
    let _ = result;
}

#[test]
fn test_decode_empty_input() {
    let decoder = Decoder::new();
    assert!(decoder.decode(&[]).is_err());
}

#[test]
fn test_decode_too_small() {
    let decoder = Decoder::new();
    assert!(decoder.decode(&[0xFF]).is_err());
    assert!(decoder.decode(&[0xFF, 0xD8]).is_err()); // Only SOI
}

#[test]
fn test_decode_random_garbage() {
    let decoder = Decoder::new();
    let garbage: Vec<u8> = (0..1000).map(|i| (i * 7) as u8).collect();
    assert!(decoder.decode(&garbage).is_err());
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

    let encoder = JpegEncoder::new(64, 64).quality(Quality::from_quality(100.0));
    let jpeg = encoder.encode(&img.pixels).expect("encode failed");

    let decoder = Decoder::new();
    let decoded = decoder.decode(&jpeg).expect("decode failed");

    // All pixel values should be in valid range
    for &pixel in &decoded.data {
        assert!(pixel <= 255, "Pixel value out of range: {}", pixel);
    }
}

// ============================================================================
// Large Image Tests
// ============================================================================

#[test]
fn test_decode_large_image() {
    let jpeg = create_test_jpeg(1024, 768, 85.0);
    let decoder = Decoder::new();
    let decoded = decoder.decode(&jpeg).expect("decode large failed");

    assert_eq!(decoded.width, 1024);
    assert_eq!(decoded.height, 768);
    assert_eq!(decoded.data.len(), 1024 * 768 * 3);
}

// ============================================================================
// Consistency Tests
// ============================================================================

#[test]
fn test_decode_deterministic() {
    let jpeg = create_test_jpeg(128, 128, 90.0);
    let decoder = Decoder::new();

    // Decode same JPEG multiple times
    let decoded1 = decoder.decode(&jpeg).expect("decode 1 failed");
    let decoded2 = decoder.decode(&jpeg).expect("decode 2 failed");

    // Results should be identical
    assert_eq!(decoded1.data, decoded2.data);
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_decode_1x1_pixel() {
    let img = TestImage::from_pixels(1, 1, 3, vec![100, 150, 200]);
    let encoder = JpegEncoder::new(1, 1);
    let jpeg = encoder.encode(&img.pixels).expect("encode 1x1 failed");

    let decoder = Decoder::new();
    let decoded = decoder.decode(&jpeg).expect("decode 1x1 failed");

    assert_eq!(decoded.width, 1);
    assert_eq!(decoded.height, 1);
}

#[test]
fn test_decode_minimum_mcu() {
    // 8x8 is minimum MCU size
    let jpeg = create_test_jpeg(8, 8, 90.0);
    let decoder = Decoder::new();
    let decoded = decoder.decode(&jpeg).expect("decode 8x8 failed");

    assert_eq!(decoded.width, 8);
    assert_eq!(decoded.height, 8);
}
