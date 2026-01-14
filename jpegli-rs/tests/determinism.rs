//! Encoder determinism tests.
//!
//! Verifies that encoding the same input multiple times produces identical output bytes.
//! This is critical for caching, reproducible builds, and debugging.

use enough::Unstoppable;
use jpegli::{
    decoder::Decoder,
    encoder::{EncoderConfig, PixelLayout},
};

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

fn encode_rgb(width: u32, height: u32, data: &[u8], config: &EncoderConfig) -> Vec<u8> {
    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .expect("encoder creation failed");
    enc.push_packed(data, enough::Unstoppable)
        .expect("push failed");
    enc.finish().expect("finish failed")
}

fn encode_gray(width: u32, height: u32, data: &[u8], config: &EncoderConfig) -> Vec<u8> {
    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Gray8Srgb)
        .expect("encoder creation failed");
    enc.push_packed(data, enough::Unstoppable)
        .expect("push failed");
    enc.finish().expect("finish failed")
}

#[test]
fn test_baseline_encoder_determinism() {
    let width = 128;
    let height = 128;
    let rgb = generate_gradient(width, height);

    let config = EncoderConfig::new().quality(90.0);

    let jpeg1 = encode_rgb(width as u32, height as u32, &rgb, &config);
    let jpeg2 = encode_rgb(width as u32, height as u32, &rgb, &config);
    let jpeg3 = encode_rgb(width as u32, height as u32, &rgb, &config);

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

    let config = EncoderConfig::new().quality(90.0).progressive(true);

    let jpeg1 = encode_rgb(width as u32, height as u32, &rgb, &config);
    let jpeg2 = encode_rgb(width as u32, height as u32, &rgb, &config);

    assert_eq!(jpeg1.len(), jpeg2.len(), "JPEG sizes differ");
    assert_eq!(jpeg1, jpeg2, "Progressive encoder is non-deterministic");
}

#[test]
fn test_optimized_huffman_determinism() {
    let width = 128;
    let height = 128;
    let rgb = generate_gradient(width, height);

    let config = EncoderConfig::new().quality(85.0).optimize_huffman(true);

    let jpeg1 = encode_rgb(width as u32, height as u32, &rgb, &config);
    let jpeg2 = encode_rgb(width as u32, height as u32, &rgb, &config);

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

    let config = EncoderConfig::new().quality(90.0).xyb();

    let jpeg1 = encode_rgb(width as u32, height as u32, &rgb, &config);
    let jpeg2 = encode_rgb(width as u32, height as u32, &rgb, &config);

    assert_eq!(jpeg1, jpeg2, "XYB encoder is non-deterministic");
}

#[test]
fn test_grayscale_encoder_determinism() {
    let width = 64;
    let height = 64;
    let gray: Vec<u8> = (0..width * height).map(|i| (i % 256) as u8).collect();

    let config = EncoderConfig::new().quality(90.0).grayscale();

    let jpeg1 = encode_gray(width as u32, height as u32, &gray, &config);
    let jpeg2 = encode_gray(width as u32, height as u32, &gray, &config);

    assert_eq!(jpeg1, jpeg2, "Grayscale encoder is non-deterministic");
}

#[test]
fn test_decoder_determinism() {
    let width = 128;
    let height = 128;
    let rgb = generate_gradient(width, height);

    let config = EncoderConfig::new().quality(90.0);
    let jpeg = encode_rgb(width as u32, height as u32, &rgb, &config);

    // Decode multiple times
    let decoded1 = Decoder::new().decode(&jpeg).expect("decode 1 failed");
    let decoded2 = Decoder::new().decode(&jpeg).expect("decode 2 failed");
    let decoded3 = Decoder::new().decode(&jpeg).expect("decode 3 failed");

    assert_eq!(decoded1.data, decoded2.data, "Decoder is non-deterministic");
    assert_eq!(decoded2.data, decoded3.data, "Decoder is non-deterministic");
}
