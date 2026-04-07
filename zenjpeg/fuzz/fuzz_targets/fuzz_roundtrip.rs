//! Fuzz target for encode-then-decode roundtrip testing.
//!
//! Uses structured fuzzing via `arbitrary` to generate valid image parameters,
//! then tests that encoding followed by decoding produces consistent results.
//!
//! This tests the encoder's handling of edge cases in dimensions, quality,
//! and pixel data.

#![no_main]

use arbitrary::Arbitrary;
use enough::Unstoppable;
use zenjpeg::decoder::{Decoder, PixelFormat};
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};
use libfuzzer_sys::fuzz_target;

/// Structured input for roundtrip fuzzing.
#[derive(Debug, Arbitrary)]
struct RoundtripInput {
    /// Image width (clamped to reasonable range)
    width: u8,
    /// Image height (clamped to reasonable range)
    height: u8,
    /// Quality value (1-100)
    quality: u8,
    /// Subsampling mode
    subsampling: u8,
    /// Progressive mode
    progressive: bool,
    /// Pixel data (will be truncated/extended to fit dimensions)
    pixels: Vec<u8>,
}

fuzz_target!(|input: RoundtripInput| {
    // Clamp dimensions to reasonable range (1-256)
    // Avoid 0 which is invalid, and very large which is slow
    let width = (input.width as u32).max(1).min(256);
    let height = (input.height as u32).max(1).min(256);

    // Clamp quality to valid range
    let quality_val = (input.quality as f32).max(1.0).min(100.0);

    // Select subsampling
    let subsampling = match input.subsampling % 4 {
        0 => ChromaSubsampling::None,
        1 => ChromaSubsampling::HalfHorizontal,
        2 => ChromaSubsampling::Quarter,
        _ => ChromaSubsampling::HalfVertical,
    };

    // Generate pixel buffer (RGB)
    let pixel_count = (width * height * 3) as usize;
    let mut pixels = input.pixels;
    pixels.resize(pixel_count, 128); // Pad with gray if too short

    // Build encoder config
    let config = EncoderConfig::ycbcr(quality_val, subsampling)
        .progressive(input.progressive);

    // Create encoder and encode
    let mut enc = match config.encode_from_bytes(width, height, PixelLayout::Rgb8Srgb) {
        Ok(enc) => enc,
        Err(_) => return,
    };

    if enc.push_packed(&pixels, Unstoppable).is_err() {
        return;
    }

    let encoded = match enc.finish() {
        Ok(data) => data,
        Err(_) => return, // Encoding can fail for edge cases
    };

    // Decode
    let decoder = Decoder::new()
        .output_format(PixelFormat::Rgb)
        .max_pixels(4_000_000);
    let decoded = match decoder.decode(&encoded, Unstoppable) {
        Ok(img) => img,
        Err(_) => return,
    };

    let decoded_pixels = match decoded.pixels_u8() {
        Some(p) => p,
        None => return,
    };

    // Verify dimensions match
    assert_eq!(
        decoded.width, width,
        "Width mismatch: encoded {} but decoded {}",
        width, decoded.width
    );
    assert_eq!(
        decoded.height, height,
        "Height mismatch: encoded {} but decoded {}",
        height, decoded.height
    );

    // Verify pixel count matches
    let expected_size = (width * height * 3) as usize;
    assert_eq!(
        decoded_pixels.len(),
        expected_size,
        "Pixel buffer size mismatch"
    );
});
