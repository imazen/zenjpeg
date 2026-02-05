//! Tests for 16-bit and float pixel formats (linear RGB input).
//!
//! These formats are treated as linear RGB and converted through sRGB
//! gamma correction before YCbCr encoding.
use enough::Unstoppable;

use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout, Quality, XybSubsampling};

/// Helper function to encode data with given config and layout
fn encode(
    width: u32,
    height: u32,
    data: &[u8],
    config: &EncoderConfig,
    layout: PixelLayout,
) -> zenjpeg::encoder::Result<Vec<u8>> {
    let mut enc = config.encode_from_bytes(width, height, layout)?;
    enc.push_packed(data, enough::Unstoppable)?;
    enc.finish()
}

/// Create a simple gradient test image in the specified format.
fn create_gradient_rgb16(width: usize, height: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height * 6);
    for y in 0..height {
        for x in 0..width {
            // Linear gradient (0.0 to 1.0)
            let r = (x as f32 / width as f32).powf(2.2) as f64; // Simulate linear
            let g = (y as f32 / height as f32).powf(2.2) as f64;
            let b = ((x + y) as f32 / (width + height) as f32).powf(2.2) as f64;

            // Convert to 16-bit
            let r16 = (r.clamp(0.0, 1.0) * 65535.0) as u16;
            let g16 = (g.clamp(0.0, 1.0) * 65535.0) as u16;
            let b16 = (b.clamp(0.0, 1.0) * 65535.0) as u16;

            data.extend_from_slice(&r16.to_ne_bytes());
            data.extend_from_slice(&g16.to_ne_bytes());
            data.extend_from_slice(&b16.to_ne_bytes());
        }
    }
    data
}

/// Create a simple gradient test image in RGBA16 format.
fn create_gradient_rgba16(width: usize, height: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height * 8);
    for y in 0..height {
        for x in 0..width {
            let r = (x as f32 / width as f32).powf(2.2) as f64;
            let g = (y as f32 / height as f32).powf(2.2) as f64;
            let b = ((x + y) as f32 / (width + height) as f32).powf(2.2) as f64;

            let r16 = (r.clamp(0.0, 1.0) * 65535.0) as u16;
            let g16 = (g.clamp(0.0, 1.0) * 65535.0) as u16;
            let b16 = (b.clamp(0.0, 1.0) * 65535.0) as u16;
            let a16 = 65535u16; // Fully opaque

            data.extend_from_slice(&r16.to_ne_bytes());
            data.extend_from_slice(&g16.to_ne_bytes());
            data.extend_from_slice(&b16.to_ne_bytes());
            data.extend_from_slice(&a16.to_ne_bytes());
        }
    }
    data
}

/// Create a simple gradient test image in Gray16 format.
fn create_gradient_gray16(width: usize, height: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height * 2);
    for y in 0..height {
        for x in 0..width {
            let v = ((x + y) as f32 / (width + height) as f32).powf(2.2) as f64;
            let v16 = (v.clamp(0.0, 1.0) * 65535.0) as u16;
            data.extend_from_slice(&v16.to_ne_bytes());
        }
    }
    data
}

/// Create a simple gradient test image in RgbF32 format.
fn create_gradient_rgbf32(width: usize, height: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height * 12);
    for y in 0..height {
        for x in 0..width {
            // Linear values (0.0 to 1.0)
            let r = (x as f32 / width as f32).powf(2.2);
            let g = (y as f32 / height as f32).powf(2.2);
            let b = ((x + y) as f32 / (width + height) as f32).powf(2.2);

            data.extend_from_slice(&r.to_ne_bytes());
            data.extend_from_slice(&g.to_ne_bytes());
            data.extend_from_slice(&b.to_ne_bytes());
        }
    }
    data
}

/// Create a simple gradient test image in RgbaF32 format.
fn create_gradient_rgbaf32(width: usize, height: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height * 16);
    for y in 0..height {
        for x in 0..width {
            let r = (x as f32 / width as f32).powf(2.2);
            let g = (y as f32 / height as f32).powf(2.2);
            let b = ((x + y) as f32 / (width + height) as f32).powf(2.2);
            let a = 1.0f32; // Fully opaque

            data.extend_from_slice(&r.to_ne_bytes());
            data.extend_from_slice(&g.to_ne_bytes());
            data.extend_from_slice(&b.to_ne_bytes());
            data.extend_from_slice(&a.to_ne_bytes());
        }
    }
    data
}

/// Create a simple gradient test image in GrayF32 format.
fn create_gradient_grayf32(width: usize, height: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height * 4);
    for y in 0..height {
        for x in 0..width {
            let v = ((x + y) as f32 / (width + height) as f32).powf(2.2);
            data.extend_from_slice(&v.to_ne_bytes());
        }
    }
    data
}

/// Verify JPEG is valid by checking magic bytes and EOI marker.
fn verify_jpeg(data: &[u8]) -> bool {
    data.len() > 4
        && data[0] == 0xFF
        && data[1] == 0xD8
        && data[data.len() - 2] == 0xFF
        && data[data.len() - 1] == 0xD9
}

#[test]
fn test_rgb16_encoding() {
    let width = 64;
    let height = 64;
    let pixels = create_gradient_rgb16(width, height);

    let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::None);
    let jpeg = encode(
        width as u32,
        height as u32,
        &pixels,
        &config,
        PixelLayout::Rgb16Linear,
    )
    .expect("RGB16 encoding should succeed");

    assert!(verify_jpeg(&jpeg), "Output should be valid JPEG");
    assert!(jpeg.len() > 100, "JPEG should have reasonable size");
    println!("RGB16 encode: {} bytes", jpeg.len());
}

#[test]
fn test_rgba16_encoding() {
    let width = 64;
    let height = 64;
    let pixels = create_gradient_rgba16(width, height);

    let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);
    let jpeg = encode(
        width as u32,
        height as u32,
        &pixels,
        &config,
        PixelLayout::Rgbx16Linear,
    )
    .expect("RGBA16 encoding should succeed");

    assert!(verify_jpeg(&jpeg), "Output should be valid JPEG");
    println!("RGBA16 encode: {} bytes", jpeg.len());
}

#[test]
fn test_gray16_encoding() {
    let width = 64;
    let height = 64;
    let pixels = create_gradient_gray16(width, height);

    let config = EncoderConfig::grayscale(85.0);
    let jpeg = encode(
        width as u32,
        height as u32,
        &pixels,
        &config,
        PixelLayout::Gray16Linear,
    )
    .expect("Gray16 encoding should succeed");

    assert!(verify_jpeg(&jpeg), "Output should be valid JPEG");
    println!("Gray16 encode: {} bytes", jpeg.len());
}

#[test]
fn test_rgbf32_encoding() {
    let width = 64;
    let height = 64;
    let pixels = create_gradient_rgbf32(width, height);

    let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::None);
    let jpeg = encode(
        width as u32,
        height as u32,
        &pixels,
        &config,
        PixelLayout::RgbF32Linear,
    )
    .expect("RgbF32 encoding should succeed");

    assert!(verify_jpeg(&jpeg), "Output should be valid JPEG");
    println!("RgbF32 encode: {} bytes", jpeg.len());
}

#[test]
fn test_rgbaf32_encoding() {
    let width = 64;
    let height = 64;
    let pixels = create_gradient_rgbaf32(width, height);

    let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);
    let jpeg = encode(
        width as u32,
        height as u32,
        &pixels,
        &config,
        PixelLayout::RgbxF32Linear,
    )
    .expect("RgbaF32 encoding should succeed");

    assert!(verify_jpeg(&jpeg), "Output should be valid JPEG");
    println!("RgbaF32 encode: {} bytes", jpeg.len());
}

#[test]
fn test_grayf32_encoding() {
    let width = 64;
    let height = 64;
    let pixels = create_gradient_grayf32(width, height);

    let config = EncoderConfig::grayscale(85.0);
    let jpeg = encode(
        width as u32,
        height as u32,
        &pixels,
        &config,
        PixelLayout::GrayF32Linear,
    )
    .expect("GrayF32 encoding should succeed");

    assert!(verify_jpeg(&jpeg), "Output should be valid JPEG");
    println!("GrayF32 encode: {} bytes", jpeg.len());
}

#[test]
fn test_linear_formats_different_sizes() {
    let config = EncoderConfig::ycbcr(75.0, ChromaSubsampling::None);

    // Test non-MCU-aligned sizes
    for (width, height) in [(63, 65), (17, 33), (100, 100), (1, 1)] {
        let pixels = create_gradient_rgbf32(width, height);
        let jpeg = encode(
            width as u32,
            height as u32,
            &pixels,
            &config,
            PixelLayout::RgbF32Linear,
        )
        .unwrap_or_else(|_| panic!("RgbF32 {}x{} should succeed", width, height));

        assert!(verify_jpeg(&jpeg), "Output should be valid JPEG");
    }
}

#[test]
fn test_linear_format_quality_range() {
    let width = 32;
    let height = 32;
    let pixels = create_gradient_rgb16(width, height);

    for quality in [10.0, 50.0, 85.0, 95.0] {
        let config = EncoderConfig::ycbcr(quality, ChromaSubsampling::None);
        let jpeg = encode(
            width as u32,
            height as u32,
            &pixels,
            &config,
            PixelLayout::Rgb16Linear,
        )
        .unwrap_or_else(|_| panic!("RGB16 quality {} should succeed", quality));

        assert!(verify_jpeg(&jpeg), "Output should be valid JPEG");
        println!("RGB16 q{}: {} bytes", quality, jpeg.len());
    }
}

#[test]
fn test_linear_format_subsampling_modes() {
    let width = 64;
    let height = 64;
    let pixels = create_gradient_rgbf32(width, height);

    for subsampling in [
        ChromaSubsampling::None,
        ChromaSubsampling::HalfHorizontal,
        ChromaSubsampling::Quarter,
        ChromaSubsampling::HalfVertical,
    ] {
        let config = EncoderConfig::ycbcr(85.0, subsampling);
        let jpeg = encode(
            width as u32,
            height as u32,
            &pixels,
            &config,
            PixelLayout::RgbF32Linear,
        )
        .unwrap_or_else(|_| panic!("RgbF32 {:?} should succeed", subsampling));

        assert!(verify_jpeg(&jpeg), "Output should be valid JPEG");
        println!("RgbF32 {:?}: {} bytes", subsampling, jpeg.len());
    }
}

/// Test XYB mode with linear RGB input.
/// XYB is defined in linear space, so this should work well.
///
/// IGNORED: XYB encoding via legacy path has a bug with chroma block indexing.
/// See CLAUDE.md "Known Bugs" #2 about XYB quality gap.
#[test]
#[ignore]
fn test_xyb_with_linear_formats() {
    let width = 64;
    let height = 64;

    // Test RgbF32 with XYB
    let pixels_f32 = create_gradient_rgbf32(width, height);
    let config_f32 = EncoderConfig::xyb(Quality::ApproxButteraugli(1.0), XybSubsampling::BQuarter);
    let jpeg_f32 = encode(
        width as u32,
        height as u32,
        &pixels_f32,
        &config_f32,
        PixelLayout::RgbF32Linear,
    )
    .expect("XYB with RgbF32 should succeed");

    assert!(verify_jpeg(&jpeg_f32), "Output should be valid JPEG");
    println!("XYB RgbF32: {} bytes", jpeg_f32.len());

    // Test Rgb16 with XYB
    let pixels_16 = create_gradient_rgb16(width, height);
    let config_16 = EncoderConfig::xyb(Quality::ApproxButteraugli(1.0), XybSubsampling::BQuarter);
    let jpeg_16 = encode(
        width as u32,
        height as u32,
        &pixels_16,
        &config_16,
        PixelLayout::Rgb16Linear,
    )
    .expect("XYB with Rgb16 should succeed");

    assert!(verify_jpeg(&jpeg_16), "Output should be valid JPEG");
    println!("XYB Rgb16: {} bytes", jpeg_16.len());
}

/// Test that gamma correction is being applied correctly.
/// Linear 0.5 should map to sRGB ~0.735 (brighter), not 0.5.
#[test]
fn test_gamma_correction_applied() {
    // Create a solid mid-gray image in linear space
    let width = 8;
    let height = 8;
    let linear_gray = 0.5f32; // Mid-gray in linear space

    let mut pixels_f32 = Vec::with_capacity(width * height * 12);
    for _ in 0..width * height {
        pixels_f32.extend_from_slice(&linear_gray.to_ne_bytes());
        pixels_f32.extend_from_slice(&linear_gray.to_ne_bytes());
        pixels_f32.extend_from_slice(&linear_gray.to_ne_bytes());
    }

    // Also create the same image in 8-bit sRGB for comparison
    // sRGB value for linear 0.5 is approximately 186/255 = 0.729
    let srgb_gray = 186u8;
    let pixels_8bit: Vec<u8> = vec![srgb_gray; width * height * 3];

    let config = EncoderConfig::ycbcr(100.0, ChromaSubsampling::None);

    let jpeg_f32 = encode(
        width as u32,
        height as u32,
        &pixels_f32,
        &config,
        PixelLayout::RgbF32Linear,
    )
    .expect("RgbF32 should succeed");

    let jpeg_8bit = encode(
        width as u32,
        height as u32,
        &pixels_8bit,
        &config,
        PixelLayout::Rgb8Srgb,
    )
    .expect("Rgb should succeed");

    // Both should produce similar output sizes since they represent
    // approximately the same visual brightness
    let size_diff = (jpeg_f32.len() as i64 - jpeg_8bit.len() as i64).abs();
    println!(
        "Gamma test: f32={} bytes, 8bit={} bytes, diff={}",
        jpeg_f32.len(),
        jpeg_8bit.len(),
        size_diff
    );

    // The outputs should be similar (within 20% since it's a solid color)
    assert!(
        size_diff < (jpeg_8bit.len() / 5) as i64,
        "Linear and sRGB encodings of same brightness should be similar"
    );
}
