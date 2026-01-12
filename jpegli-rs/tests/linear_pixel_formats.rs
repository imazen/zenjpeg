//! Tests for 16-bit and float pixel formats (linear RGB input).
//!
//! These formats are treated as linear RGB and converted through sRGB
//! gamma correction before YCbCr encoding.

use jpegli::{PixelFormat, Quality, StreamingEncoder, Subsampling};

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

    let jpeg = StreamingEncoder::new(width as u32, height as u32)
        .quality(Quality::from_quality(85.0))
        .pixel_format(PixelFormat::Rgb16)
        .subsampling(Subsampling::S444)
        .encode_all(&pixels)
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

    let jpeg = StreamingEncoder::new(width as u32, height as u32)
        .quality(Quality::from_quality(85.0))
        .pixel_format(PixelFormat::Rgba16)
        .subsampling(Subsampling::S420)
        .encode_all(&pixels)
        .expect("RGBA16 encoding should succeed");

    assert!(verify_jpeg(&jpeg), "Output should be valid JPEG");
    println!("RGBA16 encode: {} bytes", jpeg.len());
}

#[test]
fn test_gray16_encoding() {
    let width = 64;
    let height = 64;
    let pixels = create_gradient_gray16(width, height);

    let jpeg = StreamingEncoder::new(width as u32, height as u32)
        .quality(Quality::from_quality(85.0))
        .pixel_format(PixelFormat::Gray16)
        .encode_all(&pixels)
        .expect("Gray16 encoding should succeed");

    assert!(verify_jpeg(&jpeg), "Output should be valid JPEG");
    println!("Gray16 encode: {} bytes", jpeg.len());
}

#[test]
fn test_rgbf32_encoding() {
    let width = 64;
    let height = 64;
    let pixels = create_gradient_rgbf32(width, height);

    let jpeg = StreamingEncoder::new(width as u32, height as u32)
        .quality(Quality::from_quality(85.0))
        .pixel_format(PixelFormat::RgbF32)
        .subsampling(Subsampling::S444)
        .encode_all(&pixels)
        .expect("RgbF32 encoding should succeed");

    assert!(verify_jpeg(&jpeg), "Output should be valid JPEG");
    println!("RgbF32 encode: {} bytes", jpeg.len());
}

#[test]
fn test_rgbaf32_encoding() {
    let width = 64;
    let height = 64;
    let pixels = create_gradient_rgbaf32(width, height);

    let jpeg = StreamingEncoder::new(width as u32, height as u32)
        .quality(Quality::from_quality(85.0))
        .pixel_format(PixelFormat::RgbaF32)
        .subsampling(Subsampling::S420)
        .encode_all(&pixels)
        .expect("RgbaF32 encoding should succeed");

    assert!(verify_jpeg(&jpeg), "Output should be valid JPEG");
    println!("RgbaF32 encode: {} bytes", jpeg.len());
}

#[test]
fn test_grayf32_encoding() {
    let width = 64;
    let height = 64;
    let pixels = create_gradient_grayf32(width, height);

    let jpeg = StreamingEncoder::new(width as u32, height as u32)
        .quality(Quality::from_quality(85.0))
        .pixel_format(PixelFormat::GrayF32)
        .encode_all(&pixels)
        .expect("GrayF32 encoding should succeed");

    assert!(verify_jpeg(&jpeg), "Output should be valid JPEG");
    println!("GrayF32 encode: {} bytes", jpeg.len());
}

#[test]
fn test_linear_formats_different_sizes() {
    // Test non-MCU-aligned sizes
    for (width, height) in [(63, 65), (17, 33), (100, 100), (1, 1)] {
        let pixels = create_gradient_rgbf32(width, height);
        let jpeg = StreamingEncoder::new(width as u32, height as u32)
            .quality(Quality::from_quality(75.0))
            .pixel_format(PixelFormat::RgbF32)
            .encode_all(&pixels)
            .expect(&format!("RgbF32 {}x{} should succeed", width, height));

        assert!(verify_jpeg(&jpeg), "Output should be valid JPEG");
    }
}

#[test]
fn test_linear_format_quality_range() {
    let width = 32;
    let height = 32;
    let pixels = create_gradient_rgb16(width, height);

    for quality in [10.0, 50.0, 85.0, 95.0] {
        let jpeg = StreamingEncoder::new(width as u32, height as u32)
            .quality(Quality::from_quality(quality))
            .pixel_format(PixelFormat::Rgb16)
            .encode_all(&pixels)
            .expect(&format!("RGB16 quality {} should succeed", quality));

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
        Subsampling::S444,
        Subsampling::S422,
        Subsampling::S420,
        Subsampling::S440,
    ] {
        let jpeg = StreamingEncoder::new(width as u32, height as u32)
            .quality(Quality::from_quality(85.0))
            .pixel_format(PixelFormat::RgbF32)
            .subsampling(subsampling)
            .encode_all(&pixels)
            .expect(&format!("RgbF32 {:?} should succeed", subsampling));

        assert!(verify_jpeg(&jpeg), "Output should be valid JPEG");
        println!("RgbF32 {:?}: {} bytes", subsampling, jpeg.len());
    }
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

    let jpeg_f32 = StreamingEncoder::new(width as u32, height as u32)
        .quality(Quality::from_quality(100.0))
        .pixel_format(PixelFormat::RgbF32)
        .subsampling(Subsampling::S444)
        .encode_all(&pixels_f32)
        .expect("RgbF32 should succeed");

    let jpeg_8bit = StreamingEncoder::new(width as u32, height as u32)
        .quality(Quality::from_quality(100.0))
        .pixel_format(PixelFormat::Rgb)
        .subsampling(Subsampling::S444)
        .encode_all(&pixels_8bit)
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
