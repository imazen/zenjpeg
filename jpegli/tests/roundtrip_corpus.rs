//! Corpus-based roundtrip quality tests.
//!
//! Tests that verify roundtrip quality using test images from the corpus,
//! matching C++ jpegli quality verification patterns.

#[path = "../src/test_utils.rs"]
mod test_utils;

use test_utils::{
    distance_rms, generate_checkerboard, generate_color_bars, generate_gradient_d,
    generate_gradient_h, generate_gradient_v, generate_noise, generate_solid, generate_solid_rgb,
    max_pixel_diff, thresholds, TestImage, TestPattern,
};

use jpegli::{
    decode::Decoder,
    encode::Encoder,
    types::{JpegMode, PixelFormat},
    Quality,
};
use test_case::test_case;

// ============================================================================
// Helper Functions
// ============================================================================

/// Encode and decode an image, returning quality metrics.
fn roundtrip_metrics(img: &TestImage, quality: f32, mode: JpegMode) -> (f64, u8, usize, usize) {
    let encoder = Encoder::new()
        .width(img.width)
        .height(img.height)
        .quality(Quality::from_quality(quality))
        .mode(mode);

    let jpeg = encoder.encode(&img.pixels).expect("encode failed");
    let decoder = Decoder::new();
    let decoded = decoder.decode(&jpeg).expect("decode failed");

    let rms = distance_rms(&img.pixels, &decoded.data);
    let max_diff = max_pixel_diff(&img.pixels, &decoded.data);

    (rms, max_diff, jpeg.len(), img.pixels.len())
}

/// Calculate bits per pixel (bpp) from JPEG size and image dimensions.
fn calculate_bpp(jpeg_size: usize, width: u32, height: u32) -> f64 {
    (jpeg_size as f64 * 8.0) / (width as f64 * height as f64)
}

// ============================================================================
// Synthetic Image Corpus Tests
// ============================================================================

/// Test gradient images at various quality levels.
#[test_case(256, 256, 50.0, thresholds::Q50_MAX_RMS ; "256x256_Q50")]
#[test_case(256, 256, 75.0, thresholds::Q75_MAX_RMS ; "256x256_Q75")]
#[test_case(256, 256, 85.0, thresholds::Q85_MAX_RMS ; "256x256_Q85")]
#[test_case(256, 256, 90.0, thresholds::Q90_MAX_RMS ; "256x256_Q90")]
#[test_case(256, 256, 95.0, thresholds::Q95_MAX_RMS ; "256x256_Q95")]
fn test_gradient_quality(width: u32, height: u32, quality: f32, max_rms: f64) {
    let img = generate_gradient_d(width, height, 3);
    let (rms, max_diff, jpeg_size, _) = roundtrip_metrics(&img, quality, JpegMode::Baseline);
    let bpp = calculate_bpp(jpeg_size, width, height);

    println!(
        "Gradient {}x{} Q{}: RMS={:.2}, max_diff={}, size={}, bpp={:.2}",
        width, height, quality, rms, max_diff, jpeg_size, bpp
    );

    assert!(
        rms <= max_rms,
        "Gradient Q{} RMS {:.2} exceeds threshold {:.2}",
        quality,
        rms,
        max_rms
    );
}

/// Test solid color images - should compress very well.
#[test_case(0, 0, 0, "black")]
#[test_case(255, 255, 255, "white")]
#[test_case(255, 0, 0, "red")]
#[test_case(0, 255, 0, "green")]
#[test_case(0, 0, 255, "blue")]
#[test_case(128, 128, 128, "gray")]
#[test_case(255, 255, 0, "yellow")]
#[test_case(0, 255, 255, "cyan")]
#[test_case(255, 0, 255, "magenta")]
fn test_solid_color_roundtrip(r: u8, g: u8, b: u8, name: &str) {
    let img = generate_solid_rgb(128, 128, r, g, b);
    let (rms, max_diff, jpeg_size, _) = roundtrip_metrics(&img, 95.0, JpegMode::Baseline);

    println!(
        "Solid {} ({},{},{}): RMS={:.2}, max_diff={}, size={}",
        name, r, g, b, rms, max_diff, jpeg_size
    );

    // Solid colors should roundtrip very accurately
    assert!(rms < 3.0, "Solid {} RMS {:.2} too high", name, rms);
    assert!(
        max_diff < 15,
        "Solid {} max_diff {} too high",
        name,
        max_diff
    );
}

/// Test checkerboard patterns (high frequency content).
#[test_case(4, "4x4_blocks")]
#[test_case(8, "8x8_blocks")]
#[test_case(16, "16x16_blocks")]
#[test_case(32, "32x32_blocks")]
fn test_checkerboard_roundtrip(block_size: u32, name: &str) {
    let img = generate_checkerboard(256, 256, block_size, 3);
    let (rms, max_diff, jpeg_size, _) = roundtrip_metrics(&img, 95.0, JpegMode::Baseline);

    println!(
        "Checkerboard {}: RMS={:.2}, max_diff={}, size={}",
        name, rms, max_diff, jpeg_size
    );

    // Checkerboard has ringing artifacts at edges
    // Smaller blocks = more edges = more artifacts
    let expected_max_rms = if block_size <= 8 { 25.0 } else { 15.0 };
    assert!(
        rms < expected_max_rms,
        "Checkerboard {} RMS {:.2} exceeds threshold",
        name,
        rms
    );
}

/// Test color bars (TV test pattern).
#[test]
fn test_color_bars_roundtrip() {
    let img = generate_color_bars(256, 128);
    let (rms, max_diff, jpeg_size, _) = roundtrip_metrics(&img, 90.0, JpegMode::Baseline);

    println!(
        "Color bars: RMS={:.2}, max_diff={}, size={}",
        rms, max_diff, jpeg_size
    );

    assert!(
        rms < thresholds::Q90_MAX_RMS * 2.0,
        "Color bars RMS too high"
    );
}

// ============================================================================
// Image Size Corpus Tests
// ============================================================================

/// Test various image sizes to ensure encoder handles all dimensions.
#[test_case(8, 8 ; "8x8_minimum_mcu")]
#[test_case(16, 16 ; "16x16_two_mcu")]
#[test_case(17, 17 ; "17x17_partial_mcu")]
#[test_case(31, 33 ; "31x33_odd")]
#[test_case(64, 64 ; "64x64")]
#[test_case(100, 100 ; "100x100")]
#[test_case(128, 128 ; "128x128")]
#[test_case(255, 255 ; "255x255")]
#[test_case(256, 256 ; "256x256")]
#[test_case(320, 240 ; "320x240_qvga")]
#[test_case(640, 480 ; "640x480_vga")]
fn test_size_corpus(width: u32, height: u32) {
    let img = generate_gradient_d(width, height, 3);
    let (rms, max_diff, jpeg_size, _) = roundtrip_metrics(&img, 90.0, JpegMode::Baseline);

    println!(
        "Size {}x{}: RMS={:.2}, max_diff={}, size={}",
        width, height, rms, max_diff, jpeg_size
    );

    assert!(
        rms < thresholds::Q90_MAX_RMS,
        "{}x{} RMS {:.2} exceeds Q90 threshold",
        width,
        height,
        rms
    );
}

/// Test non-square aspect ratios.
#[test_case(256, 64, "4:1_wide")]
#[test_case(64, 256, "1:4_tall")]
#[test_case(320, 180, "16:9_wide")]
#[test_case(180, 320, "9:16_tall")]
#[test_case(400, 100, "4:1_banner")]
fn test_aspect_ratios(width: u32, height: u32, name: &str) {
    let img = generate_gradient_d(width, height, 3);
    let (rms, max_diff, jpeg_size, _) = roundtrip_metrics(&img, 90.0, JpegMode::Baseline);

    println!(
        "Aspect {} ({}x{}): RMS={:.2}, max_diff={}, size={}",
        name, width, height, rms, max_diff, jpeg_size
    );

    assert!(
        rms < thresholds::Q90_MAX_RMS,
        "{} RMS {:.2} exceeds threshold",
        name,
        rms
    );
}

// ============================================================================
// Quality vs File Size Tests
// ============================================================================

/// Verify quality-size tradeoff is reasonable.
#[test]
fn test_quality_size_tradeoff() {
    let img = generate_gradient_d(256, 256, 3);

    let qualities = [30.0, 50.0, 70.0, 85.0, 95.0];
    let mut prev_rms = f64::MAX;
    let mut prev_size = 0usize;

    println!("Quality vs Size tradeoff:");
    println!("Q\tRMS\tSize\tBPP");

    for &q in &qualities {
        let (rms, _max_diff, jpeg_size, _) = roundtrip_metrics(&img, q, JpegMode::Baseline);
        let bpp = calculate_bpp(jpeg_size, 256, 256);

        println!("{}\t{:.2}\t{}\t{:.2}", q, rms, jpeg_size, bpp);

        // Higher quality should have lower RMS
        assert!(
            rms < prev_rms + 1.0,
            "Q{} RMS {:.2} should be <= Q{} RMS {:.2}",
            q,
            rms,
            q - 20.0,
            prev_rms
        );

        // Higher quality should generally have larger file
        if prev_size > 0 {
            assert!(
                jpeg_size >= prev_size * 7 / 10,
                "Q{} size {} unexpectedly smaller than Q{} size {}",
                q,
                jpeg_size,
                q - 20.0,
                prev_size
            );
        }

        prev_rms = rms;
        prev_size = jpeg_size;
    }
}

// ============================================================================
// Progressive vs Baseline Comparison
// ============================================================================

#[test]
fn test_progressive_vs_baseline_quality() {
    let img = generate_gradient_d(256, 256, 3);

    let (baseline_rms, _, baseline_size, _) = roundtrip_metrics(&img, 85.0, JpegMode::Baseline);
    let (progressive_rms, _, progressive_size, _) =
        roundtrip_metrics(&img, 85.0, JpegMode::Progressive);

    println!(
        "Baseline:    RMS={:.2}, size={}",
        baseline_rms, baseline_size
    );
    println!(
        "Progressive: RMS={:.2}, size={}",
        progressive_rms, progressive_size
    );

    // Quality should be similar
    assert!(
        (baseline_rms - progressive_rms).abs() < 1.0,
        "Quality difference too large"
    );
}

// ============================================================================
// Grayscale Tests
// ============================================================================

#[test_case(64, 64 ; "64x64")]
#[test_case(128, 128 ; "128x128")]
#[test_case(256, 256 ; "256x256")]
fn test_grayscale_roundtrip(width: u32, height: u32) {
    let img = generate_gradient_h(width, height, 1);
    let encoder = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Gray)
        .quality(Quality::from_quality(90.0));

    let jpeg = encoder.encode(&img.pixels).expect("encode failed");
    let decoder = Decoder::new();
    let decoded = decoder.decode(&jpeg).expect("decode failed");

    // Note: Decoder may expand grayscale to RGB
    println!(
        "Grayscale {}x{}: JPEG size={}, decoded format={:?}",
        width,
        height,
        jpeg.len(),
        decoded.format
    );

    // Just verify dimensions are correct
    assert_eq!(decoded.width, width);
    assert_eq!(decoded.height, height);
}

// ============================================================================
// Consistency Tests
// ============================================================================

#[test]
fn test_encode_deterministic() {
    let img = generate_gradient_d(128, 128, 3);
    let encoder = Encoder::new()
        .width(128)
        .height(128)
        .quality(Quality::from_quality(85.0));

    let jpeg1 = encoder.encode(&img.pixels).expect("encode 1 failed");
    let jpeg2 = encoder.encode(&img.pixels).expect("encode 2 failed");

    assert_eq!(jpeg1, jpeg2, "Encoding should be deterministic");
}

#[test]
fn test_decode_deterministic() {
    let img = generate_gradient_d(128, 128, 3);
    let encoder = Encoder::new()
        .width(128)
        .height(128)
        .quality(Quality::from_quality(85.0));
    let jpeg = encoder.encode(&img.pixels).expect("encode failed");

    let decoder = Decoder::new();
    let decoded1 = decoder.decode(&jpeg).expect("decode 1 failed");
    let decoded2 = decoder.decode(&jpeg).expect("decode 2 failed");

    assert_eq!(
        decoded1.data, decoded2.data,
        "Decoding should be deterministic"
    );
}

// ============================================================================
// Compression Ratio Tests
// ============================================================================

#[test]
fn test_compression_ratio() {
    let test_cases = [
        ("Gradient", generate_gradient_d(256, 256, 3)),
        ("Solid", generate_solid(256, 256, 128, 3)),
        ("Checkerboard", generate_checkerboard(256, 256, 16, 3)),
        ("ColorBars", generate_color_bars(256, 128)),
    ];

    println!("Compression ratios at Q85:");
    println!("Image\t\tRaw Size\tJPEG Size\tRatio");

    for (name, img) in &test_cases {
        let raw_size = img.pixels.len();
        let encoder = Encoder::new()
            .width(img.width)
            .height(img.height)
            .quality(Quality::from_quality(85.0));
        let jpeg = encoder.encode(&img.pixels).expect("encode failed");

        let ratio = raw_size as f64 / jpeg.len() as f64;
        println!(
            "{}\t\t{}\t\t{}\t\t{:.1}:1",
            name,
            raw_size,
            jpeg.len(),
            ratio
        );

        // All should compress to some degree
        assert!(
            jpeg.len() < raw_size,
            "{} should compress (JPEG {} >= raw {})",
            name,
            jpeg.len(),
            raw_size
        );
    }
}

// ============================================================================
// Edge Case Patterns
// ============================================================================

#[test]
fn test_single_color_blocks() {
    // Image with distinct 8x8 blocks of solid colors
    let mut img = TestImage::new(64, 64, 3);
    for by in 0..8 {
        for bx in 0..8 {
            let r = (bx * 32) as u8;
            let g = (by * 32) as u8;
            let b = ((bx + by) * 16) as u8;

            for y in 0..8 {
                for x in 0..8 {
                    img.set_pixel(bx * 8 + x, by * 8 + y, 0, r);
                    img.set_pixel(bx * 8 + x, by * 8 + y, 1, g);
                    img.set_pixel(bx * 8 + x, by * 8 + y, 2, b);
                }
            }
        }
    }

    let (rms, max_diff, jpeg_size, _) = roundtrip_metrics(&img, 95.0, JpegMode::Baseline);
    println!(
        "Block colors: RMS={:.2}, max_diff={}, size={}",
        rms, max_diff, jpeg_size
    );

    // Should compress well since each 8x8 block is uniform
    assert!(rms < 5.0, "Block colors RMS too high");
}

#[test]
fn test_alternating_pixels() {
    // Maximum frequency content - alternating black/white pixels
    let mut img = TestImage::new(64, 64, 3);
    for y in 0..64 {
        for x in 0..64 {
            let val = if (x + y) % 2 == 0 { 255 } else { 0 };
            img.set_pixel(x, y, 0, val);
            img.set_pixel(x, y, 1, val);
            img.set_pixel(x, y, 2, val);
        }
    }

    let (rms, max_diff, jpeg_size, _) = roundtrip_metrics(&img, 95.0, JpegMode::Baseline);
    println!(
        "Alternating: RMS={:.2}, max_diff={}, size={}",
        rms, max_diff, jpeg_size
    );

    // High frequency content is poorly preserved by JPEG
    // Just verify it encodes/decodes without error
    assert!(jpeg_size > 0);
}
