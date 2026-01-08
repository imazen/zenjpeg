//! End-to-end test comparing strip-based vs standard encoder output.
//!
//! This test verifies that the strip-based encoder produces valid JPEGs
//! with quality comparable to the standard encoder.

use jpegli::types::{PixelFormat, Subsampling};
use jpegli::{Decoder, Encoder, Quality};

/// Generate a deterministic test image (gradient pattern).
fn generate_test_image(width: usize, height: usize) -> Vec<u8> {
    let mut rgb_data = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            rgb_data[idx] = (x * 255 / width.max(1)) as u8;
            rgb_data[idx + 1] = (y * 255 / height.max(1)) as u8;
            rgb_data[idx + 2] = 128;
        }
    }
    rgb_data
}

/// Generate a more complex test image with high-frequency content.
fn generate_checkerboard_image(width: usize, height: usize, block_size: usize) -> Vec<u8> {
    let mut rgb_data = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            let is_white = ((x / block_size) + (y / block_size)) % 2 == 0;
            let val = if is_white { 255 } else { 0 };
            rgb_data[idx] = val;
            rgb_data[idx + 1] = val;
            rgb_data[idx + 2] = val;
        }
    }
    rgb_data
}

#[test]
fn test_strip_produces_valid_jpeg_420() {
    let width = 256;
    let height = 256;
    let rgb_data = generate_test_image(width, height);

    let encoder = Encoder::new()
        .width(width as u32)
        .height(height as u32)
        .jpegli_quality(Quality::Traditional(85.0))
        .pixel_format(PixelFormat::Rgb)
        .subsampling(Subsampling::S420)
        .optimize_huffman(true);

    let jpeg_data = encoder
        .encode_strip_based(&rgb_data)
        .expect("encoding failed");

    // Verify JPEG markers
    assert!(
        jpeg_data.len() > 100,
        "JPEG too small: {} bytes",
        jpeg_data.len()
    );
    assert_eq!(&jpeg_data[0..2], &[0xFF, 0xD8], "Missing SOI marker");
    assert_eq!(
        &jpeg_data[jpeg_data.len() - 2..],
        &[0xFF, 0xD9],
        "Missing EOI marker"
    );

    // Decode the JPEG to verify it's valid
    let decoder = Decoder::new();
    let decoded = decoder.decode(&jpeg_data).expect("decoding failed");

    assert_eq!(decoded.width, width as u32);
    assert_eq!(decoded.height, height as u32);
}

#[test]
fn test_strip_produces_valid_jpeg_444() {
    let width = 256;
    let height = 256;
    let rgb_data = generate_test_image(width, height);

    let encoder = Encoder::new()
        .width(width as u32)
        .height(height as u32)
        .jpegli_quality(Quality::Traditional(85.0))
        .pixel_format(PixelFormat::Rgb)
        .subsampling(Subsampling::S444)
        .optimize_huffman(true);

    let jpeg_data = encoder
        .encode_strip_based(&rgb_data)
        .expect("encoding failed");

    // Verify JPEG markers
    assert!(jpeg_data.len() > 100, "JPEG too small");
    assert_eq!(&jpeg_data[0..2], &[0xFF, 0xD8], "Missing SOI marker");
    assert_eq!(
        &jpeg_data[jpeg_data.len() - 2..],
        &[0xFF, 0xD9],
        "Missing EOI marker"
    );

    // Decode the JPEG to verify it's valid
    let decoder = Decoder::new();
    let decoded = decoder.decode(&jpeg_data).expect("decoding failed");

    assert_eq!(decoded.width, width as u32);
    assert_eq!(decoded.height, height as u32);
}

#[test]
fn test_strip_vs_standard_similar_size_420() {
    let width = 512;
    let height = 512;
    let rgb_data = generate_test_image(width, height);

    let encoder = Encoder::new()
        .width(width as u32)
        .height(height as u32)
        .jpegli_quality(Quality::Traditional(85.0))
        .pixel_format(PixelFormat::Rgb)
        .subsampling(Subsampling::S420)
        .optimize_huffman(true);

    let standard_jpeg = encoder.encode(&rgb_data).expect("standard encoding failed");
    let strip_jpeg = encoder
        .encode_strip_based(&rgb_data)
        .expect("strip encoding failed");

    // Both should produce similarly-sized output
    let size_ratio = strip_jpeg.len() as f64 / standard_jpeg.len() as f64;
    assert!(
        size_ratio > 0.9 && size_ratio < 1.1,
        "Size difference too large: standard={}, strip={}, ratio={:.2}",
        standard_jpeg.len(),
        strip_jpeg.len(),
        size_ratio
    );
}

#[test]
fn test_strip_vs_standard_similar_size_444() {
    let width = 512;
    let height = 512;
    let rgb_data = generate_test_image(width, height);

    let encoder = Encoder::new()
        .width(width as u32)
        .height(height as u32)
        .jpegli_quality(Quality::Traditional(85.0))
        .pixel_format(PixelFormat::Rgb)
        .subsampling(Subsampling::S444)
        .optimize_huffman(true);

    let standard_jpeg = encoder.encode(&rgb_data).expect("standard encoding failed");
    let strip_jpeg = encoder
        .encode_strip_based(&rgb_data)
        .expect("strip encoding failed");

    // Both should produce similarly-sized output
    let size_ratio = strip_jpeg.len() as f64 / standard_jpeg.len() as f64;
    assert!(
        size_ratio > 0.9 && size_ratio < 1.1,
        "Size difference too large: standard={}, strip={}, ratio={:.2}",
        standard_jpeg.len(),
        strip_jpeg.len(),
        size_ratio
    );
}

#[test]
fn test_strip_various_sizes() {
    let test_cases = [
        (64, 64, "64x64"),
        (128, 128, "128x128"),
        (256, 256, "256x256"),
        (320, 240, "320x240"),
        (640, 480, "640x480"),
        (1920, 1080, "1920x1080"),
    ];

    for (width, height, name) in test_cases {
        let rgb_data = generate_test_image(width, height);

        let encoder = Encoder::new()
            .width(width as u32)
            .height(height as u32)
            .jpegli_quality(Quality::Traditional(85.0))
            .pixel_format(PixelFormat::Rgb)
            .subsampling(Subsampling::S420)
            .optimize_huffman(true);

        let jpeg_data = encoder
            .encode_strip_based(&rgb_data)
            .unwrap_or_else(|e| panic!("{}: encoding failed: {:?}", name, e));

        // Verify it can be decoded
        let decoder = Decoder::new();
        let decoded = decoder
            .decode(&jpeg_data)
            .unwrap_or_else(|e| panic!("{}: decoding failed: {:?}", name, e));

        assert_eq!(decoded.width, width as u32, "{}: width mismatch", name);
        assert_eq!(decoded.height, height as u32, "{}: height mismatch", name);
    }
}

#[test]
fn test_strip_various_qualities() {
    let width = 256;
    let height = 256;
    let rgb_data = generate_test_image(width, height);

    for quality in [50.0, 75.0, 85.0, 95.0] {
        let encoder = Encoder::new()
            .width(width as u32)
            .height(height as u32)
            .jpegli_quality(Quality::Traditional(quality))
            .pixel_format(PixelFormat::Rgb)
            .subsampling(Subsampling::S420)
            .optimize_huffman(true);

        let jpeg_data = encoder
            .encode_strip_based(&rgb_data)
            .unwrap_or_else(|e| panic!("q{}: encoding failed: {:?}", quality, e));

        // Decode to verify
        let decoder = Decoder::new();
        decoder
            .decode(&jpeg_data)
            .unwrap_or_else(|e| panic!("q{}: decoding failed: {:?}", quality, e));
    }
}

#[test]
fn test_strip_checkerboard_high_frequency() {
    // Checkerboard is a challenging pattern for DCT compression
    let width = 256;
    let height = 256;
    let rgb_data = generate_checkerboard_image(width, height, 8);

    let encoder = Encoder::new()
        .width(width as u32)
        .height(height as u32)
        .jpegli_quality(Quality::Traditional(85.0))
        .pixel_format(PixelFormat::Rgb)
        .subsampling(Subsampling::S420)
        .optimize_huffman(true);

    let jpeg_data = encoder
        .encode_strip_based(&rgb_data)
        .expect("encoding failed");

    // Should still produce valid JPEG
    let decoder = Decoder::new();
    let decoded = decoder.decode(&jpeg_data).expect("decoding failed");
    assert_eq!(decoded.width, width as u32);
    assert_eq!(decoded.height, height as u32);
}

#[test]
fn test_strip_422_subsampling() {
    let width = 256;
    let height = 256;
    let rgb_data = generate_test_image(width, height);

    let encoder = Encoder::new()
        .width(width as u32)
        .height(height as u32)
        .jpegli_quality(Quality::Traditional(85.0))
        .pixel_format(PixelFormat::Rgb)
        .subsampling(Subsampling::S422)
        .optimize_huffman(true);

    let jpeg_data = encoder
        .encode_strip_based(&rgb_data)
        .expect("encoding failed");

    // Verify JPEG markers
    assert!(jpeg_data.len() > 100, "JPEG too small");
    assert_eq!(&jpeg_data[0..2], &[0xFF, 0xD8], "Missing SOI marker");

    // Decode to verify
    let decoder = Decoder::new();
    let decoded = decoder.decode(&jpeg_data).expect("decoding failed");
    assert_eq!(decoded.width, width as u32);
    assert_eq!(decoded.height, height as u32);
}

#[test]
fn test_strip_440_subsampling() {
    let width = 256;
    let height = 256;
    let rgb_data = generate_test_image(width, height);

    let encoder = Encoder::new()
        .width(width as u32)
        .height(height as u32)
        .jpegli_quality(Quality::Traditional(85.0))
        .pixel_format(PixelFormat::Rgb)
        .subsampling(Subsampling::S440)
        .optimize_huffman(true);

    let jpeg_data = encoder
        .encode_strip_based(&rgb_data)
        .expect("encoding failed");

    // Verify JPEG markers
    assert!(jpeg_data.len() > 100, "JPEG too small");
    assert_eq!(&jpeg_data[0..2], &[0xFF, 0xD8], "Missing SOI marker");

    // Decode to verify
    let decoder = Decoder::new();
    let decoded = decoder.decode(&jpeg_data).expect("decoding failed");
    assert_eq!(decoded.width, width as u32);
    assert_eq!(decoded.height, height as u32);
}
