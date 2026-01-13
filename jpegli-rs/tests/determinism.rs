//! Encoder determinism tests.
//!
//! Verifies that encoding the same input multiple times produces identical output bytes.
//! This is critical for caching, reproducible builds, and debugging.

use jpegli::{Decoder, JpegEncoder, JpegMode, PixelFormat, Quality};

/// Generate a gradient test image
fn generate_gradient(width: usize, height: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            rgb[idx] = ((x * 255) / width.max(1)) as u8;
            rgb[idx + 1] = ((y * 255) / height.max(1)) as u8;
            rgb[idx + 2] = 128;
        }
    }
    rgb
}

#[test]
fn test_baseline_encoder_determinism() {
    let width = 128;
    let height = 128;
    let rgb = generate_gradient(width, height);

    let jpeg1 = JpegEncoder::new(width as u32, height as u32)
        .pixel_format(PixelFormat::Rgb)
        .quality(Quality::from_quality(90.0))
        .mode(JpegMode::Baseline)
        .encode(&rgb)
        .expect("encode 1 failed");

    let jpeg2 = JpegEncoder::new(width as u32, height as u32)
        .pixel_format(PixelFormat::Rgb)
        .quality(Quality::from_quality(90.0))
        .mode(JpegMode::Baseline)
        .encode(&rgb)
        .expect("encode 2 failed");

    let jpeg3 = JpegEncoder::new(width as u32, height as u32)
        .pixel_format(PixelFormat::Rgb)
        .quality(Quality::from_quality(90.0))
        .mode(JpegMode::Baseline)
        .encode(&rgb)
        .expect("encode 3 failed");

    assert_eq!(jpeg1.len(), jpeg2.len(), "JPEG sizes differ");
    assert_eq!(jpeg2.len(), jpeg3.len(), "JPEG sizes differ");
    assert_eq!(jpeg1, jpeg2, "Baseline encoder is non-deterministic");
    assert_eq!(jpeg2, jpeg3, "Baseline encoder is non-deterministic");
}

#[test]
fn test_progressive_encoder_determinism() {
    let width = 128;
    let height = 128;
    let rgb = generate_gradient(width, height);

    let jpeg1 = JpegEncoder::new(width as u32, height as u32)
        .pixel_format(PixelFormat::Rgb)
        .quality(Quality::from_quality(90.0))
        .mode(JpegMode::Progressive)
        .encode(&rgb)
        .expect("encode 1 failed");

    let jpeg2 = JpegEncoder::new(width as u32, height as u32)
        .pixel_format(PixelFormat::Rgb)
        .quality(Quality::from_quality(90.0))
        .mode(JpegMode::Progressive)
        .encode(&rgb)
        .expect("encode 2 failed");

    assert_eq!(jpeg1.len(), jpeg2.len(), "JPEG sizes differ");
    assert_eq!(jpeg1, jpeg2, "Progressive encoder is non-deterministic");
}

#[test]
fn test_optimized_huffman_determinism() {
    let width = 128;
    let height = 128;
    let rgb = generate_gradient(width, height);

    let jpeg1 = JpegEncoder::new(width as u32, height as u32)
        .pixel_format(PixelFormat::Rgb)
        .quality(Quality::from_quality(85.0))
        .optimize_huffman(true)
        .encode(&rgb)
        .expect("encode 1 failed");

    let jpeg2 = JpegEncoder::new(width as u32, height as u32)
        .pixel_format(PixelFormat::Rgb)
        .quality(Quality::from_quality(85.0))
        .optimize_huffman(true)
        .encode(&rgb)
        .expect("encode 2 failed");

    assert_eq!(
        jpeg1, jpeg2,
        "Optimized Huffman encoder is non-deterministic"
    );
}

#[test]
fn test_xyb_encoder_determinism() {
    let width = 128;
    let height = 128;
    let rgb = generate_gradient(width, height);

    let jpeg1 = JpegEncoder::new(width as u32, height as u32)
        .pixel_format(PixelFormat::Rgb)
        .quality(Quality::from_quality(90.0))
        .use_xyb(true)
        .encode(&rgb)
        .expect("encode 1 failed");

    let jpeg2 = JpegEncoder::new(width as u32, height as u32)
        .pixel_format(PixelFormat::Rgb)
        .quality(Quality::from_quality(90.0))
        .use_xyb(true)
        .encode(&rgb)
        .expect("encode 2 failed");

    assert_eq!(jpeg1, jpeg2, "XYB encoder is non-deterministic");
}

#[test]
fn test_grayscale_encoder_determinism() {
    let width = 64;
    let height = 64;
    let gray: Vec<u8> = (0..width * height).map(|i| (i % 256) as u8).collect();

    let jpeg1 = JpegEncoder::new(width as u32, height as u32)
        .pixel_format(PixelFormat::Gray)
        .quality(Quality::from_quality(90.0))
        .encode(&gray)
        .expect("encode 1 failed");

    let jpeg2 = JpegEncoder::new(width as u32, height as u32)
        .pixel_format(PixelFormat::Gray)
        .quality(Quality::from_quality(90.0))
        .encode(&gray)
        .expect("encode 2 failed");

    assert_eq!(jpeg1, jpeg2, "Grayscale encoder is non-deterministic");
}

#[test]
fn test_decoder_determinism() {
    let width = 128;
    let height = 128;
    let rgb = generate_gradient(width, height);

    // First encode
    let jpeg = JpegEncoder::new(width as u32, height as u32)
        .pixel_format(PixelFormat::Rgb)
        .quality(Quality::from_quality(90.0))
        .encode(&rgb)
        .expect("encode failed");

    // Decode multiple times
    let decoded1 = Decoder::new().decode(&jpeg).expect("decode 1 failed");
    let decoded2 = Decoder::new().decode(&jpeg).expect("decode 2 failed");
    let decoded3 = Decoder::new().decode(&jpeg).expect("decode 3 failed");

    assert_eq!(decoded1.data, decoded2.data, "Decoder is non-deterministic");
    assert_eq!(decoded2.data, decoded3.data, "Decoder is non-deterministic");
}
