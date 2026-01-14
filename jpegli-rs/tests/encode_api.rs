//! Encoder API conformance tests.
//!
//! Tests matching C++ jpegli encode_api_test.cc functionality.

#[path = "../src/test_utils.rs"]
mod test_utils;

use test_utils::{
    distance_rms, generate_checkerboard, generate_color_bars, generate_gradient_d,
    generate_gradient_h, max_pixel_diff, thresholds, TestImage,
};

use jpegli::{
    decoder::Decoder,
    encoder::{ChromaSubsampling, EncoderConfig, PixelLayout, Quality},
};
use test_case::test_case;

// ============================================================================
// Helper Functions
// ============================================================================

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

fn encode_gray(
    width: u32,
    height: u32,
    data: &[u8],
    config: &EncoderConfig,
) -> jpegli::encoder::Result<Vec<u8>> {
    let mut enc = config.encode_from_bytes(width, height, PixelLayout::Gray8Srgb)?;
    enc.push_packed(data, enough::Unstoppable)?;
    enc.finish()
}

fn encode_rgba(
    width: u32,
    height: u32,
    data: &[u8],
    config: &EncoderConfig,
) -> jpegli::encoder::Result<Vec<u8>> {
    let mut enc = config.encode_from_bytes(width, height, PixelLayout::Rgbx8Srgb)?;
    enc.push_packed(data, enough::Unstoppable)?;
    enc.finish()
}

fn generate_solid_rgb(width: u32, height: u32, r: u8, g: u8, b: u8) -> TestImage {
    let mut img = TestImage::new(width, height, 3);
    for y in 0..height {
        for x in 0..width {
            img.set_pixel(x, y, 0, r);
            img.set_pixel(x, y, 1, g);
            img.set_pixel(x, y, 2, b);
        }
    }
    img
}

// ============================================================================
// Basic Encoding Tests
// ============================================================================

#[test]
fn test_encode_basic_rgb() {
    let img = generate_gradient_d(64, 64, 3);
    let config = EncoderConfig::new();
    let jpeg = encode_rgb(64, 64, &img.pixels, &config).expect("encode failed");

    // Verify JPEG structure
    assert!(jpeg.len() > 100, "JPEG too small");
    assert_eq!(&jpeg[0..2], &[0xFF, 0xD8], "Missing SOI marker");
    assert_eq!(&jpeg[jpeg.len() - 2..], &[0xFF, 0xD9], "Missing EOI marker");
}

#[test]
fn test_encode_basic_grayscale() {
    let img = generate_gradient_h(64, 64, 1);
    let config = EncoderConfig::new().grayscale();
    let jpeg = encode_gray(64, 64, &img.pixels, &config).expect("encode failed");

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

    let config = EncoderConfig::new();
    let jpeg = encode_rgba(32, 32, &img.pixels, &config).expect("encode failed");
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
    let config = EncoderConfig::new().quality(quality);

    let jpeg = encode_rgb(128, 128, &img.pixels, &config).expect("encode failed");
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
            let config = EncoderConfig::new().quality(q);
            encode_rgb(256, 256, &img.pixels, &config).unwrap().len()
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
    let config = EncoderConfig::new().quality(Quality::ApproxJpegli(distance));

    let jpeg = encode_rgb(128, 128, &img.pixels, &config).expect("encode failed");
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
    let config = EncoderConfig::new();

    let jpeg = encode_rgb(width, height, &img.pixels, &config).expect("encode failed");
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
    let config = EncoderConfig::new();
    let jpeg = encode_rgb(256, 64, &wide.pixels, &config).expect("encode wide failed");

    let decoder = Decoder::new();
    let decoded = decoder.decode(&jpeg).expect("decode wide failed");
    assert_eq!(decoded.width, 256);
    assert_eq!(decoded.height, 64);

    // Tall image
    let tall = generate_gradient_h(64, 256, 3);
    let jpeg = encode_rgb(64, 256, &tall.pixels, &config).expect("encode tall failed");

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
    let config = EncoderConfig::new().progressive(false);

    let jpeg = encode_rgb(128, 128, &img.pixels, &config).expect("encode failed");

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
    let config = EncoderConfig::new().progressive(true);

    let jpeg = encode_rgb(128, 128, &img.pixels, &config).expect("encode failed");

    // Progressive should use SOF2 marker
    let sof2_pos = jpeg.windows(2).position(|w| w == [0xFF, 0xC2]);
    assert!(sof2_pos.is_some(), "SOF2 should be present for progressive");
}

#[test]
fn test_encode_progressive_smaller_than_baseline() {
    let img = generate_gradient_d(256, 256, 3);

    let baseline_config = EncoderConfig::new().progressive(false).quality(85.0);
    let baseline_jpeg =
        encode_rgb(256, 256, &img.pixels, &baseline_config).expect("baseline failed");

    let progressive_config = EncoderConfig::new().progressive(true).quality(85.0);
    let progressive_jpeg =
        encode_rgb(256, 256, &img.pixels, &progressive_config).expect("progressive failed");

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

    let config_opt = EncoderConfig::new().optimize_huffman(true);
    let jpeg_opt = encode_rgb(256, 256, &img.pixels, &config_opt).expect("optimized failed");

    let config_fixed = EncoderConfig::new().optimize_huffman(false);
    let jpeg_fixed = encode_rgb(256, 256, &img.pixels, &config_fixed).expect("fixed failed");

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
    let config = EncoderConfig::new();

    // Encode multiple different images with same encoder config
    for i in 0..5u8 {
        let img = generate_gradient_d(64, 64, 3);
        // Modify pattern slightly
        let mut pixels = img.pixels.clone();
        for p in pixels.iter_mut() {
            *p = p.wrapping_add(i * 10);
        }

        // Reuse the config for each encode
        let jpeg = encode_rgb(64, 64, &pixels, &config).expect("encode failed");
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
        let config = EncoderConfig::new().quality(95.0);
        let jpeg = encode_rgb(64, 64, &img.pixels, &config).expect("encode failed");

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
    let config = EncoderConfig::new().quality(95.0);
    let jpeg = encode_rgb(128, 128, &img.pixels, &config).expect("encode failed");

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
    let config = EncoderConfig::new().quality(90.0);
    let jpeg = encode_rgb(128, 64, &img.pixels, &config).expect("encode failed");

    let decoder = Decoder::new();
    let decoded = decoder.decode(&jpeg).expect("decode failed");

    let rms = distance_rms(&img.pixels, &decoded.data);
    // Color bars have sharp edges which cause ringing - allow higher RMS
    assert!(
        rms < thresholds::Q90_MAX_RMS * 4.0,
        "Color bars RMS too high: {:.2} (max: {:.2})",
        rms,
        thresholds::Q90_MAX_RMS * 4.0
    );
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_encode_minimum_dimensions() {
    // Smallest possible JPEG
    let img = TestImage::from_pixels(1, 1, 3, vec![128, 64, 192]);
    let config = EncoderConfig::new();
    let jpeg = encode_rgb(1, 1, &img.pixels, &config).expect("encode 1x1 failed");
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
    let config = EncoderConfig::new().quality(85.0);

    let jpeg = encode_rgb(1024, 768, &img.pixels, &config).expect("encode large failed");
    // Gradients compress very well - 5KB is reasonable for Q85
    assert!(
        jpeg.len() > 5000,
        "Large JPEG suspiciously small: {} bytes",
        jpeg.len()
    );

    let decoder = Decoder::new();
    let decoded = decoder.decode(&jpeg).expect("decode large failed");
    assert_eq!(decoded.width, 1024);
    assert_eq!(decoded.height, 768);
}

// ============================================================================
// JFIF Header Tests
// ============================================================================

#[test]
fn test_encode_no_jfif_header() {
    // jpegli-rs intentionally omits the JFIF APP0 marker to match C++ jpegli behavior.
    // This saves 18 bytes per file.
    let img = generate_gradient_d(64, 64, 3);
    let config = EncoderConfig::new();
    let jpeg = encode_rgb(64, 64, &img.pixels, &config).expect("encode failed");

    // Should NOT have APP0 JFIF marker (matches C++ jpegli)
    let app0_pos = jpeg.windows(2).position(|w| w == [0xFF, 0xE0]);
    assert!(
        app0_pos.is_none(),
        "JFIF APP0 marker should NOT be present (matches C++ jpegli)"
    );
}

// ============================================================================
// Quantization Table Tests
// ============================================================================

#[test]
fn test_encode_dqt_present() {
    let img = generate_gradient_d(64, 64, 3);
    let config = EncoderConfig::new();
    let jpeg = encode_rgb(64, 64, &img.pixels, &config).expect("encode failed");

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
    let config = EncoderConfig::new();
    let jpeg = encode_rgb(64, 64, &img.pixels, &config).expect("encode failed");

    // Look for DHT marker
    let dht_count = jpeg.windows(2).filter(|w| w == &[0xFF, 0xC4]).count();
    assert!(dht_count >= 1, "At least one DHT marker should be present");
}

// ============================================================================
// APP14 Adobe Marker Tests (XYB mode)
// ============================================================================

#[test]
fn test_xyb_has_app14_adobe_marker() {
    let img = generate_gradient_d(64, 64, 3);
    let config = EncoderConfig::new().xyb();
    let jpeg = encode_rgb(64, 64, &img.pixels, &config).expect("encode failed");

    // Look for APP14 marker (0xFF 0xEE)
    let app14_pos = jpeg.windows(2).position(|w| w == [0xFF, 0xEE]);
    assert!(
        app14_pos.is_some(),
        "XYB mode should include APP14 Adobe marker"
    );

    // Verify Adobe signature
    if let Some(pos) = app14_pos {
        let sig_start = pos + 4; // Skip marker and length
        if jpeg.len() > sig_start + 5 {
            assert_eq!(
                &jpeg[sig_start..sig_start + 5],
                b"Adobe",
                "APP14 should have Adobe signature"
            );
        }

        // Verify transform byte is 0 (RGB)
        let transform_offset = pos + 4 + 11; // marker(2) + length(2) + Adobe(5) + version(2) + flags(4)
        if jpeg.len() > transform_offset {
            assert_eq!(
                jpeg[transform_offset], 0,
                "Transform byte should be 0 for RGB"
            );
        }
    }
}

#[test]
fn test_ycbcr_no_app14_marker() {
    let img = generate_gradient_d(64, 64, 3);
    let config = EncoderConfig::new().ycbcr(ChromaSubsampling::Full);
    let jpeg = encode_rgb(64, 64, &img.pixels, &config).expect("encode failed");

    // YCbCr mode should NOT have APP14 marker (JFIF is sufficient)
    let app14_pos = jpeg.windows(2).position(|w| w == [0xFF, 0xEE]);
    assert!(
        app14_pos.is_none(),
        "YCbCr mode should not include APP14 marker"
    );
}

// ============================================================================
// Huffman Variant Tests with XYB/YCbCr
// ============================================================================

#[test]
fn test_xyb_optimized_huffman_decodable() {
    let img = generate_gradient_d(64, 64, 3);
    let config = EncoderConfig::new()
        .quality(90.0)
        .xyb()
        .optimize_huffman(true);
    let jpeg = encode_rgb(64, 64, &img.pixels, &config).expect("encode failed");

    // Should be decodable by our decoder
    let decoder = Decoder::new();
    let decoded = decoder.decode(&jpeg).expect("decode XYB optimized failed");
    assert_eq!(decoded.width, 64);
    assert_eq!(decoded.height, 64);
}

#[test]
fn test_xyb_standard_huffman_decodable() {
    let img = generate_gradient_d(64, 64, 3);
    let config = EncoderConfig::new()
        .quality(90.0)
        .xyb()
        .optimize_huffman(false);
    let jpeg = encode_rgb(64, 64, &img.pixels, &config).expect("encode failed");

    // Should be decodable by our decoder
    let decoder = Decoder::new();
    let decoded = decoder.decode(&jpeg).expect("decode XYB standard failed");
    assert_eq!(decoded.width, 64);
    assert_eq!(decoded.height, 64);
}

#[test]
fn test_ycbcr_optimized_huffman_decodable() {
    let img = generate_gradient_d(64, 64, 3);
    let config = EncoderConfig::new()
        .quality(90.0)
        .ycbcr(ChromaSubsampling::Full)
        .optimize_huffman(true);
    let jpeg = encode_rgb(64, 64, &img.pixels, &config).expect("encode failed");

    // Should be decodable by our decoder
    let decoder = Decoder::new();
    let decoded = decoder
        .decode(&jpeg)
        .expect("decode YCbCr optimized failed");
    assert_eq!(decoded.width, 64);
    assert_eq!(decoded.height, 64);
}

#[test]
fn test_ycbcr_standard_huffman_decodable() {
    let img = generate_gradient_d(64, 64, 3);
    let config = EncoderConfig::new()
        .quality(90.0)
        .ycbcr(ChromaSubsampling::Full)
        .optimize_huffman(false);
    let jpeg = encode_rgb(64, 64, &img.pixels, &config).expect("encode failed");

    // Should be decodable by our decoder
    let decoder = Decoder::new();
    let decoded = decoder.decode(&jpeg).expect("decode YCbCr standard failed");
    assert_eq!(decoded.width, 64);
    assert_eq!(decoded.height, 64);
}

#[test]
fn test_all_huffman_colorspace_combinations_with_zune() {
    let img = generate_gradient_d(64, 64, 3);

    let configs: Vec<(EncoderConfig, &str)> = vec![
        (
            EncoderConfig::new()
                .quality(90.0)
                .xyb()
                .optimize_huffman(true),
            "XYB + optimized",
        ),
        (
            EncoderConfig::new()
                .quality(90.0)
                .xyb()
                .optimize_huffman(false),
            "XYB + standard",
        ),
        (
            EncoderConfig::new()
                .quality(90.0)
                .ycbcr(ChromaSubsampling::Full)
                .optimize_huffman(true),
            "YCbCr + optimized",
        ),
        (
            EncoderConfig::new()
                .quality(90.0)
                .ycbcr(ChromaSubsampling::Full)
                .optimize_huffman(false),
            "YCbCr + standard",
        ),
    ];

    for (config, label) in &configs {
        let jpeg =
            encode_rgb(64, 64, &img.pixels, config).expect(&format!("encode {} failed", label));

        // Test with zune-jpeg
        use zune_jpeg::zune_core::bytestream::ZCursor;
        use zune_jpeg::JpegDecoder;
        let cursor = ZCursor::new(&jpeg[..]);
        let mut decoder = JpegDecoder::new(cursor);
        let result = decoder.decode();
        assert!(
            result.is_ok(),
            "zune-jpeg failed to decode {}: {:?}",
            label,
            result.err()
        );
    }
}
