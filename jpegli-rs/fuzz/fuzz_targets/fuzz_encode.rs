//! Fuzz target for JPEG encoding.
//!
//! Tests the encoder's ability to handle various parameter combinations
//! without panicking. This doesn't validate output correctness, just crash safety.

#![no_main]

use arbitrary::Arbitrary;
use jpegli::encode::Encoder;
use jpegli::types::{JpegMode, PixelFormat, Subsampling};
use jpegli::Quality;
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
    /// Mode selector
    mode: u8,
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
        0 => Subsampling::S444,
        1 => Subsampling::S422,
        2 => Subsampling::S420,
        _ => Subsampling::S440,
    };

    // Select mode
    let mode = match input.mode % 2 {
        0 => JpegMode::Baseline,
        _ => JpegMode::Progressive,
    };

    // Select pixel format
    let (pixel_format, bytes_per_pixel) = match input.pixel_format % 5 {
        0 => (PixelFormat::Gray, 1),
        1 => (PixelFormat::Rgb, 3),
        2 => (PixelFormat::Rgba, 4),
        3 => (PixelFormat::Bgr, 3),
        _ => (PixelFormat::Bgra, 4),
    };

    // Generate pixel buffer
    let pixel_count = (width * height) as usize * bytes_per_pixel;
    let mut pixels = input.pixels;
    pixels.resize(pixel_count, 128);

    // Build encoder with various options
    let mut encoder = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(pixel_format)
        .jpegli_quality(Quality::from_quality(quality_val))
        .subsampling(subsampling)
        .mode(mode)
        .optimize_huffman(input.optimize_huffman);

    // XYB only works with RGB
    if input.use_xyb && pixel_format == PixelFormat::Rgb {
        encoder = encoder.use_xyb(true);
    }

    // Encode - we just want to ensure no panics
    let _ = encoder.encode(&pixels);
});
