//! Fuzz target for JPEG encoding.
//!
//! Tests the encoder's ability to handle various parameter combinations
//! without panicking. This doesn't validate output correctness, just crash safety.

#![no_main]

use arbitrary::Arbitrary;
use enough::Unstoppable;
use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, PixelLayout, XybSubsampling};
use libfuzzer_sys::fuzz_target;

/// Structured input for encoder fuzzing.
#[derive(Debug, Arbitrary)]
struct EncodeInput {
    /// Image width (1-512)
    width: u16,
    /// Image height (1-512)
    height: u16,
    /// Quality value (raw byte, will be clamped)
    quality: u8,
    /// Subsampling selector
    subsampling: u8,
    /// Mode selector (progressive or baseline)
    progressive: bool,
    /// Pixel format selector
    pixel_format: u8,
    /// Use XYB color space
    use_xyb: bool,
    /// Optimize Huffman tables
    optimize_huffman: bool,
    /// Pixel data
    pixels: Vec<u8>,
}

fuzz_target!(|input: EncodeInput| {
    // Clamp dimensions to reasonable range
    let width = (input.width as u32).max(1).min(512);
    let height = (input.height as u32).max(1).min(512);

    // Clamp quality
    let quality_val = (input.quality as f32).max(1.0).min(100.0);

    // Select subsampling
    let subsampling = match input.subsampling % 4 {
        0 => ChromaSubsampling::None,
        1 => ChromaSubsampling::HalfHorizontal,
        2 => ChromaSubsampling::Quarter,
        _ => ChromaSubsampling::HalfVertical,
    };

    // Select pixel format and layout
    let (pixel_layout, bytes_per_pixel) = match input.pixel_format % 5 {
        0 => (PixelLayout::Gray8Srgb, 1),
        1 => (PixelLayout::Rgb8Srgb, 3),
        2 => (PixelLayout::Rgbx8Srgb, 4),
        3 => (PixelLayout::Bgr8Srgb, 3),
        _ => (PixelLayout::Bgrx8Srgb, 4),
    };

    // Generate pixel buffer
    let pixel_count = (width * height) as usize * bytes_per_pixel;
    let mut pixels = input.pixels;
    pixels.resize(pixel_count, 128);

    // Build encoder config with various options
    // Grayscale has highest priority, then XYB (only for RGB), then YCbCr
    let config = if matches!(pixel_layout, PixelLayout::Gray8Srgb) {
        EncoderConfig::grayscale(quality_val)
    } else if input.use_xyb && matches!(pixel_layout, PixelLayout::Rgb8Srgb | PixelLayout::Rgbx8Srgb)
    {
        EncoderConfig::xyb(quality_val, XybSubsampling::BQuarter)
    } else {
        EncoderConfig::ycbcr(quality_val, subsampling)
    }
    .progressive(input.progressive)
    .optimize_huffman(input.optimize_huffman);

    // Create encoder and encode - we just want to ensure no panics
    let enc = config.encode_from_bytes(width, height, pixel_layout);
    if let Ok(mut enc) = enc {
        let _ = enc.push_packed(&pixels, Unstoppable);
        let _ = enc.finish();
    }
});
