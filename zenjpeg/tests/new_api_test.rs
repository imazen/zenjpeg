//! Test the encoder/decoder API.
use enough::Unstoppable;

use zenjpeg::{
    decoder::{Decoder, PixelFormat},
    encoder::{ChromaSubsampling, EncoderConfig, PixelLayout, Quality},
};

/// Helper function to encode RGB data
fn encode_rgb(
    width: u32,
    height: u32,
    data: &[u8],
    config: &EncoderConfig,
) -> zenjpeg::encoder::Result<Vec<u8>> {
    let mut enc = config.encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)?;
    enc.push_packed(data, Unstoppable)?;
    enc.finish()
}

/// Helper function to encode gray data
fn encode_gray(
    width: u32,
    height: u32,
    data: &[u8],
    config: &EncoderConfig,
) -> zenjpeg::encoder::Result<Vec<u8>> {
    let mut enc = config.encode_from_bytes(width, height, PixelLayout::Gray8Srgb)?;
    enc.push_packed(data, Unstoppable)?;
    enc.finish()
}

#[test]
fn test_encode_rgb() {
    let width = 64u32;
    let height = 64u32;
    let pixels: Vec<u8> = (0..width * height * 3)
        .map(|i| ((i * 17) % 256) as u8)
        .collect();

    let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);
    let jpeg = encode_rgb(width, height, &pixels, &config).expect("encode failed");
    assert!(jpeg.len() > 100, "JPEG too small");
    assert!(jpeg.starts_with(&[0xFF, 0xD8]), "Invalid JPEG header");
}

#[test]
fn test_encode_gray() {
    let width = 64u32;
    let height = 64u32;
    let pixels: Vec<u8> = (0..width * height).map(|i| (i % 256) as u8).collect();

    let config = EncoderConfig::ycbcr(90.0, ChromaSubsampling::Quarter);
    let jpeg = encode_gray(width, height, &pixels, &config).expect("encode failed");
    assert!(jpeg.len() > 50, "JPEG too small");
    assert!(jpeg.starts_with(&[0xFF, 0xD8]), "Invalid JPEG header");
}

#[test]
fn test_decode() {
    let width = 32u32;
    let height = 32u32;
    let pixels: Vec<u8> = vec![128; (width * height * 3) as usize];

    let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);
    let jpeg = encode_rgb(width, height, &pixels, &config).expect("encode failed");
    let decoded = Decoder::new()
        .decode(&jpeg, Unstoppable)
        .expect("decode failed");

    assert_eq!(decoded.width(), width);
    assert_eq!(decoded.height(), height);
}

#[test]
fn test_encoder_config_integer_quality() {
    let width = 32u32;
    let height = 32u32;
    let pixels: Vec<u8> = vec![128; (width * height * 3) as usize];

    let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);
    let jpeg = encode_rgb(width, height, &pixels, &config).expect("encode failed");
    assert!(jpeg.starts_with(&[0xFF, 0xD8]));
}

#[test]
fn test_encoder_config_distance() {
    let width = 32u32;
    let height = 32u32;
    let pixels: Vec<u8> = vec![128; (width * height * 3) as usize];

    // Use butteraugli distance (1.0 ~ quality 85)
    let config = EncoderConfig::ycbcr(Quality::ApproxButteraugli(1.0), ChromaSubsampling::Quarter);
    let jpeg = encode_rgb(width, height, &pixels, &config).expect("encode failed");
    assert!(jpeg.starts_with(&[0xFF, 0xD8]));
}

#[test]
fn test_encoder_config_progressive() {
    let width = 32u32;
    let height = 32u32;
    let pixels: Vec<u8> = vec![128; (width * height * 3) as usize];

    let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter).progressive(true);
    let jpeg = encode_rgb(width, height, &pixels, &config).expect("encode failed");

    assert!(jpeg.starts_with(&[0xFF, 0xD8]));
    // Check for progressive marker (SOF2)
    let has_sof2 = jpeg.windows(2).any(|w| w[0] == 0xFF && w[1] == 0xC2);
    assert!(has_sof2, "Progressive JPEG should have SOF2 marker");
}

#[test]
fn test_encoder_config_subsampling() {
    let width = 64u32;
    let height = 64u32;
    let pixels: Vec<u8> = (0..width * height * 3)
        .map(|i| ((i * 17) % 256) as u8)
        .collect();

    let config_444 = EncoderConfig::ycbcr(85.0, ChromaSubsampling::None);
    let jpeg_444 = encode_rgb(width, height, &pixels, &config_444).expect("encode failed");

    let config_420 = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);
    let jpeg_420 = encode_rgb(width, height, &pixels, &config_420).expect("encode failed");

    // 4:2:0 should be smaller due to chroma subsampling
    assert!(
        jpeg_420.len() < jpeg_444.len(),
        "4:2:0 ({}) should be smaller than 4:4:4 ({})",
        jpeg_420.len(),
        jpeg_444.len()
    );
}

#[test]
fn test_encoder_config_streaming() {
    let width = 32u32;
    let height = 32u32;
    let pixels: Vec<u8> = vec![128; (width * height * 3) as usize];

    let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);
    let mut encoder = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .expect("start failed");

    let row_size = width as usize * 3;
    for y in 0..height as usize {
        let start = y * row_size;
        encoder
            .push_packed(&pixels[start..start + row_size], Unstoppable)
            .expect("push_packed failed");
    }

    let jpeg = encoder.finish().expect("finish failed");
    assert!(jpeg.starts_with(&[0xFF, 0xD8]));
}

#[test]
fn test_quality_clamping() {
    let width = 8u32;
    let height = 8u32;
    let pixels: Vec<u8> = vec![128; (width * height * 3) as usize];

    // Quality 0 should be clamped to 1
    let config_low = EncoderConfig::ycbcr(0.0, ChromaSubsampling::Quarter);
    let jpeg_low = encode_rgb(width, height, &pixels, &config_low).expect("encode failed");
    assert!(jpeg_low.starts_with(&[0xFF, 0xD8]));

    // Quality > 100 should be clamped to 100
    let config_high = EncoderConfig::ycbcr(200.0, ChromaSubsampling::Quarter);
    let jpeg_high = encode_rgb(width, height, &pixels, &config_high).expect("encode failed");
    assert!(jpeg_high.starts_with(&[0xFF, 0xD8]));
}

// ============================================================================
// Decoder API Tests
// ============================================================================

#[test]
fn test_decode_f32() {
    let width = 32u32;
    let height = 32u32;
    let pixels: Vec<u8> = vec![128; (width * height * 3) as usize];

    let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);
    let jpeg = encode_rgb(width, height, &pixels, &config).expect("encode failed");
    let decoded = Decoder::new()
        .decode_f32(&jpeg, Unstoppable)
        .expect("decode_f32 failed");

    assert_eq!(decoded.width(), width);
    assert_eq!(decoded.height(), height);
    assert_eq!(
        decoded.pixels_f32().unwrap().len(),
        (width * height * 3) as usize
    );

    // Values should be in 0.0-1.0 range
    for &val in decoded.pixels_f32().unwrap() {
        assert!((0.0..=1.0).contains(&val), "f32 value {} out of range", val);
    }
}

#[test]
fn test_decode_to_format_rgb() {
    let width = 32u32;
    let height = 32u32;
    let pixels: Vec<u8> = vec![128; (width * height * 3) as usize];

    let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);
    let jpeg = encode_rgb(width, height, &pixels, &config).expect("encode failed");
    let decoded = Decoder::new()
        .output_format(PixelFormat::Rgb)
        .decode(&jpeg, Unstoppable)
        .expect("decode failed");

    assert_eq!(decoded.width(), width);
    assert_eq!(decoded.height(), height);
    // RGB has 3 bytes per pixel
    assert_eq!(
        decoded.pixels_u8().unwrap().len(),
        (width * height * 3) as usize
    );
    assert_eq!(decoded.bytes_per_pixel(), 3);
}

#[test]
fn test_decoder_new() {
    let width = 32u32;
    let height = 32u32;
    let pixels: Vec<u8> = vec![128; (width * height * 3) as usize];

    let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);
    let jpeg = encode_rgb(width, height, &pixels, &config).expect("encode failed");

    let decoded = Decoder::new()
        .decode(&jpeg, Unstoppable)
        .expect("decode failed");
    assert_eq!(decoded.width(), width);
    assert_eq!(decoded.height(), height);
}

#[test]
fn test_decoder_builder() {
    let width = 32u32;
    let height = 32u32;
    let pixels: Vec<u8> = vec![128; (width * height * 3) as usize];

    let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);
    let jpeg = encode_rgb(width, height, &pixels, &config).expect("encode failed");

    let decoded = Decoder::new()
        .output_format(PixelFormat::Rgb)
        .fancy_upsampling(true)
        .decode(&jpeg, Unstoppable)
        .expect("decode failed");

    assert_eq!(decoded.width(), width);
    assert_eq!(decoded.height(), height);
}
