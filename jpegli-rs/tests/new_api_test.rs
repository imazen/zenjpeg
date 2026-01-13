//! Test the new simplified API.

use jpegli::{
    decode, decode_f32, decode_to_format, encode_gray, encode_rgb, JpegDecoder, JpegEncoder,
    PixelFormat, Subsampling,
};

#[test]
fn test_encode_rgb_convenience() {
    let width = 64u32;
    let height = 64u32;
    let pixels: Vec<u8> = (0..width * height * 3)
        .map(|i| ((i * 17) % 256) as u8)
        .collect();

    let jpeg = encode_rgb(width, height, &pixels, 85).expect("encode_rgb failed");
    assert!(jpeg.len() > 100, "JPEG too small");
    assert!(jpeg.starts_with(&[0xFF, 0xD8]), "Invalid JPEG header");
}

#[test]
fn test_encode_gray_convenience() {
    let width = 64u32;
    let height = 64u32;
    let pixels: Vec<u8> = (0..width * height).map(|i| (i % 256) as u8).collect();

    let jpeg = encode_gray(width, height, &pixels, 90).expect("encode_gray failed");
    assert!(jpeg.len() > 50, "JPEG too small");
    assert!(jpeg.starts_with(&[0xFF, 0xD8]), "Invalid JPEG header");
}

#[test]
fn test_decode_convenience() {
    let width = 32u32;
    let height = 32u32;
    let pixels: Vec<u8> = vec![128; (width * height * 3) as usize];

    let jpeg = encode_rgb(width, height, &pixels, 85).expect("encode_rgb failed");
    let decoded = decode(&jpeg).expect("decode failed");

    assert_eq!(decoded.width(), width);
    assert_eq!(decoded.height(), height);
}

#[test]
fn test_jpeg_encoder_integer_quality() {
    let width = 32u32;
    let height = 32u32;
    let pixels: Vec<u8> = vec![128; (width * height * 3) as usize];

    // Test with u8 quality
    let jpeg = JpegEncoder::new(width, height)
        .quality(85u8)
        .encode(&pixels)
        .expect("encode failed");
    assert!(jpeg.starts_with(&[0xFF, 0xD8]));

    // Test with i32 quality
    let jpeg = JpegEncoder::new(width, height)
        .quality(85i32)
        .encode(&pixels)
        .expect("encode failed");
    assert!(jpeg.starts_with(&[0xFF, 0xD8]));

    // Test with f32 quality
    let jpeg = JpegEncoder::new(width, height)
        .quality(85.0f32)
        .encode(&pixels)
        .expect("encode failed");
    assert!(jpeg.starts_with(&[0xFF, 0xD8]));
}

#[test]
fn test_jpeg_encoder_distance() {
    let width = 32u32;
    let height = 32u32;
    let pixels: Vec<u8> = vec![128; (width * height * 3) as usize];

    let jpeg = JpegEncoder::new(width, height)
        .distance(1.0)
        .encode(&pixels)
        .expect("encode failed");
    assert!(jpeg.starts_with(&[0xFF, 0xD8]));
}

#[test]
fn test_jpeg_encoder_progressive() {
    let width = 32u32;
    let height = 32u32;
    let pixels: Vec<u8> = vec![128; (width * height * 3) as usize];

    let jpeg = JpegEncoder::new(width, height)
        .quality(85)
        .progressive(true)
        .encode(&pixels)
        .expect("encode failed");

    // Progressive JPEGs have SOF2 marker (0xFF 0xC2) instead of SOF0
    // Check that it's a valid JPEG
    assert!(jpeg.starts_with(&[0xFF, 0xD8]));
    // Check for progressive marker (SOF2)
    let has_sof2 = jpeg.windows(2).any(|w| w[0] == 0xFF && w[1] == 0xC2);
    assert!(has_sof2, "Progressive JPEG should have SOF2 marker");
}

#[test]
fn test_jpeg_encoder_subsampling() {
    let width = 64u32;
    let height = 64u32;
    let pixels: Vec<u8> = (0..width * height * 3)
        .map(|i| ((i * 17) % 256) as u8)
        .collect();

    let jpeg_444 = JpegEncoder::new(width, height)
        .quality(85)
        .subsampling(Subsampling::S444)
        .encode(&pixels)
        .expect("encode failed");

    let jpeg_420 = JpegEncoder::new(width, height)
        .quality(85)
        .subsampling(Subsampling::S420)
        .encode(&pixels)
        .expect("encode failed");

    // 4:2:0 should be smaller due to chroma subsampling
    assert!(
        jpeg_420.len() < jpeg_444.len(),
        "4:2:0 ({}) should be smaller than 4:4:4 ({})",
        jpeg_420.len(),
        jpeg_444.len()
    );
}

#[test]
fn test_jpeg_encoder_streaming_start() {
    let width = 32u32;
    let height = 32u32;
    let pixels: Vec<u8> = vec![128; (width * height * 3) as usize];

    let mut encoder = JpegEncoder::new(width, height)
        .quality(85)
        .start()
        .expect("start failed");

    let row_size = width as usize * 3;
    for y in 0..height as usize {
        let start = y * row_size;
        encoder
            .push_row(&pixels[start..start + row_size])
            .expect("push_row failed");
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
    let jpeg_low = JpegEncoder::new(width, height)
        .quality(0u8)
        .encode(&pixels)
        .expect("encode failed");
    assert!(jpeg_low.starts_with(&[0xFF, 0xD8]));

    // Quality > 100 should be clamped to 100
    let jpeg_high = JpegEncoder::new(width, height)
        .quality(200i32)
        .encode(&pixels)
        .expect("encode failed");
    assert!(jpeg_high.starts_with(&[0xFF, 0xD8]));
}

// ============================================================================
// Decoder API Tests
// ============================================================================

#[test]
fn test_decode_f32_convenience() {
    let width = 32u32;
    let height = 32u32;
    let pixels: Vec<u8> = vec![128; (width * height * 3) as usize];

    let jpeg = encode_rgb(width, height, &pixels, 85).expect("encode failed");
    let decoded = decode_f32(&jpeg).expect("decode_f32 failed");

    assert_eq!(decoded.width(), width);
    assert_eq!(decoded.height(), height);
    assert_eq!(decoded.pixels().len(), (width * height * 3) as usize);

    // Values should be in 0.0-1.0 range
    for &val in decoded.pixels() {
        assert!(val >= 0.0 && val <= 1.0, "f32 value {} out of range", val);
    }
}

#[test]
fn test_decode_to_format_rgb() {
    let width = 32u32;
    let height = 32u32;
    let pixels: Vec<u8> = vec![128; (width * height * 3) as usize];

    let jpeg = encode_rgb(width, height, &pixels, 85).expect("encode failed");
    let decoded = decode_to_format(&jpeg, PixelFormat::Rgb).expect("decode failed");

    assert_eq!(decoded.width(), width);
    assert_eq!(decoded.height(), height);
    // RGB has 3 bytes per pixel
    assert_eq!(decoded.pixels().len(), (width * height * 3) as usize);
    assert_eq!(decoded.bytes_per_pixel(), 3);
}

#[test]
fn test_jpeg_decoder_alias() {
    let width = 32u32;
    let height = 32u32;
    let pixels: Vec<u8> = vec![128; (width * height * 3) as usize];

    let jpeg = encode_rgb(width, height, &pixels, 85).expect("encode failed");

    // Test JpegDecoder (alias for Decoder)
    let decoded = JpegDecoder::new().decode(&jpeg).expect("decode failed");
    assert_eq!(decoded.width(), width);
    assert_eq!(decoded.height(), height);
}

#[test]
fn test_jpeg_decoder_builder() {
    let width = 32u32;
    let height = 32u32;
    let pixels: Vec<u8> = vec![128; (width * height * 3) as usize];

    let jpeg = encode_rgb(width, height, &pixels, 85).expect("encode failed");

    // Test builder pattern
    let decoded = JpegDecoder::new()
        .output_format(PixelFormat::Rgb)
        .fancy_upsampling(true)
        .decode(&jpeg)
        .expect("decode failed");

    assert_eq!(decoded.width(), width);
    assert_eq!(decoded.height(), height);
}
