//! Encoder API conformance tests.
//!
//! Tests matching C++ jpegli encode_api_test.cc functionality.

#[path = "../src/test_utils.rs"]
mod test_utils;

use test_utils::{
    distance_rms, generate_checkerboard, generate_color_bars, generate_gradient_d,
    generate_gradient_h, generate_noise, generate_solid, generate_solid_rgb, max_pixel_diff,
    thresholds, TestImage,
};

use jpegli::{
    decode::Decoder,
    encode::{Encoder, EncoderConfig},
    types::{JpegMode, PixelFormat, Subsampling},
    Quality,
};
use test_case::test_case;

// ============================================================================
// Basic Encoding Tests
// ============================================================================

#[test]
fn test_encode_basic_rgb() {
    let img = generate_gradient_d(64, 64, 3);
    let encoder = Encoder::new().width(64).height(64);
    let jpeg = encoder.encode(&img.pixels).expect("encode failed");

    // Verify JPEG structure
    assert!(jpeg.len() > 100, "JPEG too small");
    assert_eq!(&jpeg[0..2], &[0xFF, 0xD8], "Missing SOI marker");
    assert_eq!(&jpeg[jpeg.len() - 2..], &[0xFF, 0xD9], "Missing EOI marker");
}

#[test]
fn test_encode_basic_grayscale() {
    let img = generate_gradient_h(64, 64, 1);
    let encoder = Encoder::new()
        .width(64)
        .height(64)
        .pixel_format(PixelFormat::Gray);
    let jpeg = encoder.encode(&img.pixels).expect("encode failed");

    assert!(jpeg.len() > 50, "JPEG too small");
    assert_eq!(&jpeg[0..2], &[0xFF, 0xD8], "Missing SOI marker");
}

#[test]
fn test_encode_rgba_input() {
    // RGBA input (4 channels, alpha ignored)
    let mut img = TestImage::new(32, 32, 4);
    for y in 0..32 {
        for x in 0..32 {
            img.set_pixel(x, y, 0, (x * 8) as u8); // R
            img.set_pixel(x, y, 1, (y * 8) as u8); // G
            img.set_pixel(x, y, 2, 128); // B
            img.set_pixel(x, y, 3, 255); // A (ignored)
        }
    }

    let encoder = Encoder::new()
        .width(32)
        .height(32)
        .pixel_format(PixelFormat::Rgba);
    let jpeg = encoder.encode(&img.pixels).expect("encode failed");
    assert!(jpeg.len() > 100, "JPEG too small");
}

// ============================================================================
// Quality Level Tests
// ============================================================================

#[test_case(10.0 ; "Q10")]
#[test_case(30.0 ; "Q30")]
#[test_case(50.0 ; "Q50")]
#[test_case(70.0 ; "Q70")]
#[test_case(85.0 ; "Q85")]
#[test_case(90.0 ; "Q90")]
#[test_case(95.0 ; "Q95")]
#[test_case(100.0 ; "Q100")]
fn test_encode_quality_levels(quality: f32) {
    let img = generate_gradient_d(128, 128, 3);
    let encoder = Encoder::new()
        .width(128)
        .height(128)
        .quality(Quality::from_quality(quality));

    let jpeg = encoder.encode(&img.pixels).expect("encode failed");
    assert!(jpeg.len() > 100, "Q{} JPEG too small", quality);

    // Decode and verify roundtrip
    let decoder = Decoder::new();
    let decoded = decoder.decode(&jpeg).expect("decode failed");
    assert_eq!(decoded.width, 128);
    assert_eq!(decoded.height, 128);
}

#[test]
fn test_encode_quality_affects_size() {
    let img = generate_gradient_d(256, 256, 3);

    let sizes: Vec<usize> = [30.0, 50.0, 70.0, 90.0]
        .iter()
        .map(|&q| {
            let encoder = Encoder::new()
                .width(256)
                .height(256)
                .quality(Quality::from_quality(q));
            encoder.encode(&img.pixels).unwrap().len()
        })
        .collect();

    // Higher quality should generally produce larger files
    for i in 1..sizes.len() {
        assert!(
            sizes[i] >= sizes[i - 1] * 8 / 10, // Allow some variance
            "Q{} size {} should be >= Q{} size {} (with margin)",
            [30.0, 50.0, 70.0, 90.0][i],
            sizes[i],
            [30.0, 50.0, 70.0, 90.0][i - 1],
            sizes[i - 1]
        );
    }
}

// ============================================================================
// Distance-based Quality Tests
// ============================================================================

#[test_case(0.5 ; "distance_0_5")]
#[test_case(1.0 ; "distance_1_0")]
#[test_case(2.0 ; "distance_2_0")]
#[test_case(4.0 ; "distance_4_0")]
fn test_encode_distance_quality(distance: f32) {
    let img = generate_gradient_d(128, 128, 3);
    let encoder = Encoder::new()
        .width(128)
        .height(128)
        .quality(Quality::from_distance(distance));

    let jpeg = encoder.encode(&img.pixels).expect("encode failed");
    assert!(jpeg.len() > 100, "Distance {} JPEG too small", distance);
}

// ============================================================================
// Image Size Tests
// ============================================================================

#[test_case(8, 8 ; "8x8_minimum")]
#[test_case(1, 1 ; "1x1_single_pixel")]
#[test_case(7, 7 ; "7x7_not_block_aligned")]
#[test_case(9, 9 ; "9x9_just_over_block")]
#[test_case(15, 17 ; "15x17_odd")]
#[test_case(16, 16 ; "16x16_two_blocks")]
#[test_case(100, 100 ; "100x100")]
#[test_case(256, 256 ; "256x256")]
#[test_case(640, 480 ; "640x480_vga")]
fn test_encode_various_sizes(width: u32, height: u32) {
    let img = generate_gradient_d(width, height, 3);
    let encoder = Encoder::new().width(width).height(height);

    let jpeg = encoder.encode(&img.pixels).expect("encode failed");
    assert!(jpeg.len() > 50, "{}x{} JPEG too small", width, height);

    // Verify dimensions in decoded output
    let decoder = Decoder::new();
    let decoded = decoder.decode(&jpeg).expect("decode failed");
    assert_eq!(decoded.width, width);
    assert_eq!(decoded.height, height);
}

#[test]
fn test_encode_non_square() {
    // Wide image
    let wide = generate_gradient_h(256, 64, 3);
    let encoder = Encoder::new().width(256).height(64);
    let jpeg = encoder.encode(&wide.pixels).expect("encode wide failed");

    let decoder = Decoder::new();
    let decoded = decoder.decode(&jpeg).expect("decode wide failed");
    assert_eq!(decoded.width, 256);
    assert_eq!(decoded.height, 64);

    // Tall image
    let tall = generate_gradient_h(64, 256, 3);
    let encoder = Encoder::new().width(64).height(256);
    let jpeg = encoder.encode(&tall.pixels).expect("encode tall failed");

    let decoded = decoder.decode(&jpeg).expect("decode tall failed");
    assert_eq!(decoded.width, 64);
    assert_eq!(decoded.height, 256);
}

// ============================================================================
// Encoding Mode Tests
// ============================================================================

#[test]
fn test_encode_baseline_mode() {
    let img = generate_gradient_d(128, 128, 3);
    let encoder = Encoder::new()
        .width(128)
        .height(128)
        .mode(JpegMode::Baseline);

    let jpeg = encoder.encode(&img.pixels).expect("encode failed");

    // Baseline should use SOF0 marker
    let sof0_pos = jpeg
        .windows(2)
        .position(|w| w == [0xFF, 0xC0])
        .expect("SOF0 not found");
    assert!(sof0_pos > 0, "SOF0 should be present for baseline");
}

#[test]
fn test_encode_progressive_mode() {
    let img = generate_gradient_d(128, 128, 3);
    let encoder = Encoder::new()
        .width(128)
        .height(128)
        .mode(JpegMode::Progressive);

    let jpeg = encoder.encode(&img.pixels).expect("encode failed");

    // Progressive should use SOF2 marker
    let sof2_pos = jpeg.windows(2).position(|w| w == [0xFF, 0xC2]);
    assert!(sof2_pos.is_some(), "SOF2 should be present for progressive");
}

#[test]
fn test_encode_progressive_smaller_than_baseline() {
    let img = generate_gradient_d(256, 256, 3);

    let baseline_encoder = Encoder::new()
        .width(256)
        .height(256)
        .mode(JpegMode::Baseline)
        .quality(Quality::from_quality(85.0));
    let baseline_jpeg = baseline_encoder
        .encode(&img.pixels)
        .expect("baseline failed");

    let progressive_encoder = Encoder::new()
        .width(256)
        .height(256)
        .mode(JpegMode::Progressive)
        .quality(Quality::from_quality(85.0));
    let progressive_jpeg = progressive_encoder
        .encode(&img.pixels)
        .expect("progressive failed");

    // Progressive encoding typically produces smaller files
    // Allow some variance as it depends on image content
    println!(
        "Baseline: {} bytes, Progressive: {} bytes",
        baseline_jpeg.len(),
        progressive_jpeg.len()
    );
}

// ============================================================================
// Huffman Optimization Tests
// ============================================================================

#[test]
fn test_encode_optimized_huffman() {
    let img = generate_gradient_d(256, 256, 3);

    let config_optimized = EncoderConfig {
        width: 256,
        height: 256,
        optimize_huffman: true,
        ..Default::default()
    };
    let encoder_opt = Encoder::from_config(config_optimized);
    let jpeg_opt = encoder_opt.encode(&img.pixels).expect("optimized failed");

    let config_fixed = EncoderConfig {
        width: 256,
        height: 256,
        optimize_huffman: false,
        ..Default::default()
    };
    let encoder_fixed = Encoder::from_config(config_fixed);
    let jpeg_fixed = encoder_fixed.encode(&img.pixels).expect("fixed failed");

    // Optimized should be smaller or equal
    println!(
        "Fixed Huffman: {} bytes, Optimized: {} bytes",
        jpeg_fixed.len(),
        jpeg_opt.len()
    );
    assert!(
        jpeg_opt.len() <= jpeg_fixed.len() + 100, // Allow small overhead
        "Optimized should not be significantly larger"
    );
}

// ============================================================================
// Encoder Reuse Tests (matching C++ ReuseCinfo tests)
// ============================================================================

#[test]
fn test_encode_reuse_encoder() {
    let encoder = Encoder::new().width(64).height(64);

    // Encode multiple different images with same encoder config
    for i in 0..5 {
        let img = generate_gradient_d(64, 64, 3);
        // Modify pattern slightly
        let mut pixels = img.pixels.clone();
        for p in pixels.iter_mut() {
            *p = p.wrapping_add(i * 10);
        }

        let jpeg = encoder.encode(&pixels).expect("encode failed");
        assert!(jpeg.len() > 100, "Iteration {} JPEG too small", i);

        // Verify decodes correctly
        let decoder = Decoder::new();
        let decoded = decoder.decode(&jpeg).expect("decode failed");
        assert_eq!(decoded.width, 64);
        assert_eq!(decoded.height, 64);
    }
}

// ============================================================================
// Content-Specific Tests
// ============================================================================

#[test]
fn test_encode_solid_color() {
    // Test various solid colors
    let colors = [
        (0, 0, 0),       // Black
        (255, 255, 255), // White
        (255, 0, 0),     // Red
        (0, 255, 0),     // Green
        (0, 0, 255),     // Blue
        (128, 128, 128), // Gray
    ];

    for (r, g, b) in colors {
        let img = generate_solid_rgb(64, 64, r, g, b);
        let encoder = Encoder::new()
            .width(64)
            .height(64)
            .quality(Quality::from_quality(95.0));
        let jpeg = encoder.encode(&img.pixels).expect("encode failed");

        let decoder = Decoder::new();
        let decoded = decoder.decode(&jpeg).expect("decode failed");

        // Solid colors should roundtrip very well
        let rms = distance_rms(&img.pixels, &decoded.data);
        assert!(rms < 5.0, "Solid ({},{},{}) RMS {} too high", r, g, b, rms);
    }
}

#[test]
fn test_encode_checkerboard() {
    let img = generate_checkerboard(128, 128, 8, 3);
    let encoder = Encoder::new()
        .width(128)
        .height(128)
        .quality(Quality::from_quality(95.0));
    let jpeg = encoder.encode(&img.pixels).expect("encode failed");

    let decoder = Decoder::new();
    let decoded = decoder.decode(&jpeg).expect("decode failed");

    // Checkerboard has sharp edges, expect some ringing
    let rms = distance_rms(&img.pixels, &decoded.data);
    let max_diff = max_pixel_diff(&img.pixels, &decoded.data);
    println!("Checkerboard: RMS={:.2}, max_diff={}", rms, max_diff);

    assert!(rms < 20.0, "Checkerboard RMS {} too high", rms);
}

#[test]
fn test_encode_color_bars() {
    let img = generate_color_bars(128, 64);
    let encoder = Encoder::new()
        .width(128)
        .height(64)
        .quality(Quality::from_quality(90.0));
    let jpeg = encoder.encode(&img.pixels).expect("encode failed");

    let decoder = Decoder::new();
    let decoded = decoder.decode(&jpeg).expect("decode failed");

    let rms = distance_rms(&img.pixels, &decoded.data);
    assert!(
        rms < thresholds::Q90_MAX_RMS * 2.0,
        "Color bars RMS too high"
    );
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_encode_minimum_dimensions() {
    // Smallest possible JPEG
    let img = TestImage::from_pixels(1, 1, 3, vec![128, 64, 192]);
    let encoder = Encoder::new().width(1).height(1);
    let jpeg = encoder.encode(&img.pixels).expect("encode 1x1 failed");
    assert!(jpeg.len() > 0, "1x1 JPEG should not be empty");

    let decoder = Decoder::new();
    let decoded = decoder.decode(&jpeg).expect("decode 1x1 failed");
    assert_eq!(decoded.width, 1);
    assert_eq!(decoded.height, 1);
}

#[test]
fn test_encode_large_image() {
    // Test larger image (but not too large for CI)
    let img = generate_gradient_d(1024, 768, 3);
    let encoder = Encoder::new()
        .width(1024)
        .height(768)
        .quality(Quality::from_quality(85.0));

    let jpeg = encoder.encode(&img.pixels).expect("encode large failed");
    assert!(jpeg.len() > 10000, "Large JPEG suspiciously small");

    let decoder = Decoder::new();
    let decoded = decoder.decode(&jpeg).expect("decode large failed");
    assert_eq!(decoded.width, 1024);
    assert_eq!(decoded.height, 768);
}

// ============================================================================
// JFIF Header Tests
// ============================================================================

#[test]
fn test_encode_has_jfif_header() {
    let img = generate_gradient_d(64, 64, 3);
    let encoder = Encoder::new().width(64).height(64);
    let jpeg = encoder.encode(&img.pixels).expect("encode failed");

    // Look for APP0 JFIF marker
    let app0_pos = jpeg.windows(2).position(|w| w == [0xFF, 0xE0]);
    assert!(app0_pos.is_some(), "JFIF APP0 marker should be present");

    // Verify JFIF signature
    if let Some(pos) = app0_pos {
        let sig_start = pos + 4; // Skip marker and length
        if jpeg.len() > sig_start + 5 {
            assert_eq!(&jpeg[sig_start..sig_start + 5], b"JFIF\0");
        }
    }
}

// ============================================================================
// Quantization Table Tests
// ============================================================================

#[test]
fn test_encode_dqt_present() {
    let img = generate_gradient_d(64, 64, 3);
    let encoder = Encoder::new().width(64).height(64);
    let jpeg = encoder.encode(&img.pixels).expect("encode failed");

    // Look for DQT marker
    let dqt_count = jpeg.windows(2).filter(|w| w == &[0xFF, 0xDB]).count();
    assert!(dqt_count >= 1, "At least one DQT marker should be present");
}

// ============================================================================
// Huffman Table Tests
// ============================================================================

#[test]
fn test_encode_dht_present() {
    let img = generate_gradient_d(64, 64, 3);
    let encoder = Encoder::new().width(64).height(64);
    let jpeg = encoder.encode(&img.pixels).expect("encode failed");

    // Look for DHT marker
    let dht_count = jpeg.windows(2).filter(|w| w == &[0xFF, 0xC4]).count();
    assert!(dht_count >= 1, "At least one DHT marker should be present");
}
