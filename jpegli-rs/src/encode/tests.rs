use super::*;

#[test]
fn test_encoder_creation() {
    let encoder = Encoder::new()
        .width(640)
        .height(480)
        .jpegli_quality(Quality::from_quality(90.0));

    assert_eq!(encoder.config.width, 640);
    assert_eq!(encoder.config.height, 480);
}

#[test]
fn test_encoder_validation() {
    let encoder = Encoder::new();
    assert!(encoder.validate().is_err());

    let encoder = Encoder::new().width(100).height(100);
    assert!(encoder.validate().is_ok());
}

#[test]
fn test_encode_small_gray() {
    let encoder = Encoder::new()
        .width(8)
        .height(8)
        .pixel_format(PixelFormat::Gray)
        .jpegli_quality(Quality::from_quality(90.0));

    let data = vec![128u8; 64];
    let result = encoder.encode(&data);
    assert!(result.is_ok());

    let jpeg = result.unwrap();
    // Should start with SOI
    assert_eq!(jpeg[0], 0xFF);
    assert_eq!(jpeg[1], MARKER_SOI);
    // Should end with EOI
    assert_eq!(jpeg[jpeg.len() - 2], 0xFF);
    assert_eq!(jpeg[jpeg.len() - 1], MARKER_EOI);
}

#[test]
fn test_encode_rgb_xyb_mode() {
    // Test XYB mode encoding with a 16x16 RGB image
    let encoder = Encoder::new()
        .width(16)
        .height(16)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(Quality::from_quality(90.0))
        .use_xyb(true);

    // Create a simple gradient test image
    let mut data = vec![0u8; 16 * 16 * 3];
    for y in 0..16 {
        for x in 0..16 {
            let idx = (y * 16 + x) * 3;
            data[idx] = (x * 16) as u8; // Red gradient
            data[idx + 1] = (y * 16) as u8; // Green gradient
            data[idx + 2] = 128; // Constant blue
        }
    }

    let result = encoder.encode(&data);
    assert!(result.is_ok(), "XYB encoding failed: {:?}", result.err());

    let jpeg = result.unwrap();
    // Should start with SOI
    assert_eq!(jpeg[0], 0xFF);
    assert_eq!(jpeg[1], MARKER_SOI);
    // Should end with EOI
    assert_eq!(jpeg[jpeg.len() - 2], 0xFF);
    assert_eq!(jpeg[jpeg.len() - 1], MARKER_EOI);

    // Should be a valid size (not too small)
    assert!(jpeg.len() > 100, "JPEG too small: {} bytes", jpeg.len());
    println!("XYB encoded JPEG size: {} bytes", jpeg.len());
}

#[test]
fn test_encode_rgb_xyb_larger() {
    // Test XYB mode with a larger image (32x32)
    let encoder = Encoder::new()
        .width(32)
        .height(32)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(Quality::from_quality(75.0))
        .use_xyb(true);

    // Create a test pattern
    let mut data = vec![0u8; 32 * 32 * 3];
    for y in 0..32 {
        for x in 0..32 {
            let idx = (y * 32 + x) * 3;
            // Checkerboard pattern
            let checker = ((x / 4) + (y / 4)) % 2 == 0;
            data[idx] = if checker { 255 } else { 0 }; // Red
            data[idx + 1] = if checker { 0 } else { 255 }; // Green
            data[idx + 2] = 128; // Blue
        }
    }

    let result = encoder.encode(&data);
    assert!(result.is_ok(), "XYB encoding failed: {:?}", result.err());

    let jpeg = result.unwrap();
    assert_eq!(jpeg[0], 0xFF);
    assert_eq!(jpeg[1], MARKER_SOI);
    assert_eq!(jpeg[jpeg.len() - 2], 0xFF);
    assert_eq!(jpeg[jpeg.len() - 1], MARKER_EOI);
    println!("XYB encoded 32x32 JPEG size: {} bytes", jpeg.len());
}

#[test]
fn test_huffman_optimization_produces_valid_jpeg() {
    // Create a gradient test image
    let width = 64u32;
    let height = 64u32;
    let mut data = vec![0u8; (width * height * 3) as usize];

    for y in 0..height as usize {
        for x in 0..width as usize {
            let idx = (y * width as usize + x) * 3;
            data[idx] = (x * 4) as u8; // R
            data[idx + 1] = (y * 4) as u8; // G
            data[idx + 2] = ((x + y) * 2) as u8; // B
        }
    }

    let encoder = Encoder::new()
        .width(width)
        .height(height)
        .jpegli_quality(Quality::from_quality(75.0))
        .optimize_huffman(true);

    let result = encoder.encode(&data);
    assert!(
        result.is_ok(),
        "Optimized Huffman encoding failed: {:?}",
        result.err()
    );

    let jpeg = result.unwrap();
    assert_eq!(jpeg[0], 0xFF);
    assert_eq!(jpeg[1], MARKER_SOI);
    assert_eq!(jpeg[jpeg.len() - 2], 0xFF);
    assert_eq!(jpeg[jpeg.len() - 1], MARKER_EOI);

    // Verify it's decodable
    let decoded = decode_zune(&jpeg[..]);
    assert!(
        decoded.is_ok(),
        "Optimized JPEG not decodable: {:?}",
        decoded.err()
    );
}

#[test]
fn test_huffman_optimization_reduces_file_size() {
    // Create a more complex test image that benefits from optimization
    let width = 128u32;
    let height = 128u32;
    let mut data = vec![0u8; (width * height * 3) as usize];

    // Create a pattern that will have non-uniform symbol frequencies
    for y in 0..height as usize {
        for x in 0..width as usize {
            let idx = (y * width as usize + x) * 3;
            // Create blocks with varying content
            let block_type = ((x / 16) + (y / 16)) % 4;
            match block_type {
                0 => {
                    // Solid color
                    data[idx] = 180;
                    data[idx + 1] = 180;
                    data[idx + 2] = 180;
                }
                1 => {
                    // Gradient
                    data[idx] = (x * 2) as u8;
                    data[idx + 1] = (y * 2) as u8;
                    data[idx + 2] = 100;
                }
                2 => {
                    // Checkerboard
                    let checker = ((x + y) % 2) as u8 * 255;
                    data[idx] = checker;
                    data[idx + 1] = checker;
                    data[idx + 2] = checker;
                }
                _ => {
                    // Texture
                    data[idx] = ((x * 5 + y * 3) % 256) as u8;
                    data[idx + 1] = ((x * 3 + y * 7) % 256) as u8;
                    data[idx + 2] = ((x * 2 + y * 2) % 256) as u8;
                }
            }
        }
    }

    // Encode without optimization
    let jpeg_standard = Encoder::new()
        .width(width)
        .height(height)
        .jpegli_quality(Quality::from_quality(75.0))
        .optimize_huffman(false)
        .encode(&data)
        .expect("Standard encoding failed");

    // Encode with optimization
    let jpeg_optimized = Encoder::new()
        .width(width)
        .height(height)
        .jpegli_quality(Quality::from_quality(75.0))
        .optimize_huffman(true)
        .encode(&data)
        .expect("Optimized encoding failed");

    println!(
        "Standard size: {} bytes, Optimized size: {} bytes, Savings: {:.1}%",
        jpeg_standard.len(),
        jpeg_optimized.len(),
        (1.0 - jpeg_optimized.len() as f64 / jpeg_standard.len() as f64) * 100.0
    );

    // Optimized should be smaller or equal (never larger)
    assert!(
        jpeg_optimized.len() <= jpeg_standard.len(),
        "Optimized ({}) should not be larger than standard ({})",
        jpeg_optimized.len(),
        jpeg_standard.len()
    );

    // Verify both are decodable
    let decoded_std = decode_zune(&jpeg_standard[..]);
    let decoded_opt = decode_zune(&jpeg_optimized[..]);
    assert!(decoded_std.is_ok(), "Standard JPEG not decodable");
    assert!(decoded_opt.is_ok(), "Optimized JPEG not decodable");
}

#[test]
fn test_xyb_huffman_optimization() {
    // Create test image for XYB mode
    let width = 64u32;
    let height = 64u32;
    let mut data = vec![0u8; (width * height * 3) as usize];

    for y in 0..height as usize {
        for x in 0..width as usize {
            let idx = (y * width as usize + x) * 3;
            data[idx] = (x * 4) as u8;
            data[idx + 1] = (y * 4) as u8;
            data[idx + 2] = ((x + y) * 2) as u8;
        }
    }

    // Encode XYB without optimization
    let jpeg_standard = Encoder::new()
        .width(width)
        .height(height)
        .jpegli_quality(Quality::from_quality(75.0))
        .use_xyb(true)
        .optimize_huffman(false)
        .encode(&data)
        .expect("Standard XYB encoding failed");

    // Encode XYB with optimization
    let jpeg_optimized = Encoder::new()
        .width(width)
        .height(height)
        .jpegli_quality(Quality::from_quality(75.0))
        .use_xyb(true)
        .optimize_huffman(true)
        .encode(&data)
        .expect("Optimized XYB encoding failed");

    println!(
        "XYB Standard: {} bytes, Optimized: {} bytes, Savings: {:.1}%",
        jpeg_standard.len(),
        jpeg_optimized.len(),
        (1.0 - jpeg_optimized.len() as f64 / jpeg_standard.len() as f64) * 100.0
    );

    // Verify both have valid JPEG structure
    assert_eq!(jpeg_standard[0], 0xFF);
    assert_eq!(jpeg_standard[1], MARKER_SOI);
    assert_eq!(jpeg_optimized[0], 0xFF);
    assert_eq!(jpeg_optimized[1], MARKER_SOI);

    // Optimized should be smaller or equal
    assert!(
        jpeg_optimized.len() <= jpeg_standard.len(),
        "XYB Optimized ({}) should not be larger than standard ({})",
        jpeg_optimized.len(),
        jpeg_standard.len()
    );
}

#[test]
fn test_sharp_yuv_420() {
    // Test Sharp YUV with 4:2:0 produces valid JPEG
    let width = 64u32;
    let height = 64u32;
    // Create a colorful gradient to test color edge preservation
    let mut data = vec![0u8; (width * height * 3) as usize];
    for y in 0..height as usize {
        for x in 0..width as usize {
            let idx = (y * width as usize + x) * 3;
            data[idx] = (x * 4) as u8; // R increases horizontally
            data[idx + 1] = (y * 4) as u8; // G increases vertically
            data[idx + 2] = 128; // B constant
        }
    }

    // Encode with Sharp YUV
    let jpeg_sharp = Encoder::new()
        .width(width)
        .height(height)
        .subsampling(Subsampling::S420)
        .sharp_yuv(true)
        .jpegli_quality(Quality::from_quality(90.0))
        .encode(&data)
        .expect("Sharp YUV 4:2:0 encoding failed");

    // Encode with standard downsampling (Sharp YUV disabled)
    let jpeg_standard = Encoder::new()
        .width(width)
        .height(height)
        .subsampling(Subsampling::S420)
        .sharp_yuv(false)
        .jpegli_quality(Quality::from_quality(90.0))
        .encode(&data)
        .expect("Standard 4:2:0 encoding failed");

    // Both should produce valid JPEGs
    assert!(jpeg_sharp.len() > 0, "Sharp YUV output should not be empty");
    assert!(
        jpeg_standard.len() > 0,
        "Standard output should not be empty"
    );

    // Sharp YUV should produce a valid JPEG (starts with SOI marker)
    assert_eq!(&jpeg_sharp[0..2], &[0xFF, 0xD8], "Should be valid JPEG");
    assert_eq!(
        &jpeg_sharp[jpeg_sharp.len() - 2..],
        &[0xFF, 0xD9],
        "Should end with EOI"
    );
}

#[test]
fn test_sharp_yuv_422() {
    // Test Sharp YUV with 4:2:2 produces valid JPEG
    let width = 64u32;
    let height = 64u32;
    let data: Vec<u8> = (0..width * height * 3).map(|i| (i % 256) as u8).collect();

    let jpeg = Encoder::new()
        .width(width)
        .height(height)
        .subsampling(Subsampling::S422)
        .sharp_yuv(true)
        .jpegli_quality(Quality::from_quality(90.0))
        .encode(&data)
        .expect("Sharp YUV 4:2:2 encoding failed");

    // Should produce valid JPEG
    assert!(jpeg.len() > 0);
    assert_eq!(&jpeg[0..2], &[0xFF, 0xD8]);
}

#[test]
fn test_sharp_yuv_falls_back_for_444() {
    // 4:4:4 should work with sharp_yuv=true (falls back to standard path)
    let width = 32u32;
    let height = 32u32;
    let data: Vec<u8> = (0..width * height * 3).map(|i| (i % 256) as u8).collect();

    let jpeg = Encoder::new()
        .width(width)
        .height(height)
        .subsampling(Subsampling::S444)
        .sharp_yuv(true) // Should still work, just uses standard path
        .jpegli_quality(Quality::from_quality(90.0))
        .encode(&data)
        .expect("Sharp YUV with 4:4:4 should fall back to standard");

    assert!(jpeg.len() > 0);
    assert_eq!(&jpeg[0..2], &[0xFF, 0xD8]);
}

// ========================================================================
// Chroma Downsampling Tests
// ========================================================================

fn decode_zune(data: &[u8]) -> std::result::Result<Vec<u8>, zune_jpeg::errors::DecodeErrors> {
    use zune_jpeg::zune_core::bytestream::ZCursor;
    use zune_jpeg::JpegDecoder;
    let cursor = ZCursor::new(data);
    let mut decoder = JpegDecoder::new(cursor);
    decoder.decode()
}

#[test]
fn test_chroma_downsampling_box_420() {
    let width = 32u32;
    let height = 32u32;
    let data: Vec<u8> = (0..width * height * 3).map(|i| (i % 256) as u8).collect();

    let jpeg = Encoder::new()
        .width(width)
        .height(height)
        .subsampling(Subsampling::S420)
        .chroma_downsampling(ChromaDownsampling::Box)
        .encode(&data)
        .expect("Box downsampling should work");

    assert_eq!(&jpeg[0..2], &[0xFF, 0xD8], "Should start with SOI");
    assert!(decode_zune(&jpeg).is_ok(), "Should be decodable");
}

#[test]
fn test_chroma_downsampling_gamma_aware_420() {
    let width = 32u32;
    let height = 32u32;
    let data: Vec<u8> = (0..width * height * 3).map(|i| (i % 256) as u8).collect();

    let jpeg = Encoder::new()
        .width(width)
        .height(height)
        .subsampling(Subsampling::S420)
        .chroma_downsampling(ChromaDownsampling::GammaAware)
        .encode(&data)
        .expect("GammaAware downsampling should work");

    assert_eq!(&jpeg[0..2], &[0xFF, 0xD8], "Should start with SOI");
    assert!(decode_zune(&jpeg).is_ok(), "Should be decodable");
}

#[test]
fn test_chroma_downsampling_gamma_aware_iterative_420() {
    let width = 32u32;
    let height = 32u32;
    let data: Vec<u8> = (0..width * height * 3).map(|i| (i % 256) as u8).collect();

    let jpeg = Encoder::new()
        .width(width)
        .height(height)
        .subsampling(Subsampling::S420)
        .chroma_downsampling(ChromaDownsampling::GammaAwareIterative)
        .encode(&data)
        .expect("GammaAwareIterative downsampling should work");

    assert_eq!(&jpeg[0..2], &[0xFF, 0xD8], "Should start with SOI");
    assert!(decode_zune(&jpeg).is_ok(), "Should be decodable");
}

#[test]
fn test_chroma_downsampling_gamma_aware_422() {
    let width = 32u32;
    let height = 32u32;
    let data: Vec<u8> = (0..width * height * 3).map(|i| (i % 256) as u8).collect();

    let jpeg = Encoder::new()
        .width(width)
        .height(height)
        .subsampling(Subsampling::S422)
        .chroma_downsampling(ChromaDownsampling::GammaAware)
        .encode(&data)
        .expect("GammaAware 4:2:2 should work");

    assert_eq!(&jpeg[0..2], &[0xFF, 0xD8]);
}

#[test]
fn test_chroma_downsampling_gamma_aware_440() {
    let width = 32u32;
    let height = 32u32;
    let data: Vec<u8> = (0..width * height * 3).map(|i| (i % 256) as u8).collect();

    let jpeg = Encoder::new()
        .width(width)
        .height(height)
        .subsampling(Subsampling::S440)
        .chroma_downsampling(ChromaDownsampling::GammaAware)
        .encode(&data)
        .expect("GammaAware 4:4:0 should work");

    assert_eq!(&jpeg[0..2], &[0xFF, 0xD8]);
}

#[test]
fn test_chroma_downsampling_444_ignores_setting() {
    // 4:4:4 has no downsampling, so the setting should be ignored
    let width = 32u32;
    let height = 32u32;
    let data: Vec<u8> = (0..width * height * 3).map(|i| (i % 256) as u8).collect();

    let jpeg = Encoder::new()
        .width(width)
        .height(height)
        .subsampling(Subsampling::S444)
        .chroma_downsampling(ChromaDownsampling::GammaAwareIterative) // Should be ignored
        .encode(&data)
        .expect("4:4:4 should work regardless of downsampling setting");

    assert_eq!(&jpeg[0..2], &[0xFF, 0xD8]);
}

#[test]
fn test_sharp_yuv_convenience_method() {
    let width = 32u32;
    let height = 32u32;
    let data: Vec<u8> = (0..width * height * 3).map(|i| (i % 256) as u8).collect();

    let jpeg = Encoder::new()
        .width(width)
        .height(height)
        .subsampling(Subsampling::S420)
        .sharp_yuv(true) // Should set GammaAwareIterative
        .encode(&data)
        .expect("sharp_yuv(true) should work");

    assert_eq!(&jpeg[0..2], &[0xFF, 0xD8]);
}

// ============================================================================
// Cancellation tests
// ============================================================================

#[test]
fn test_encode_with_stop_never() {
    // Test that encoding with Never (no cancellation) works normally
    let width = 32u32;
    let height = 32u32;
    let data: Vec<u8> = (0..width * height * 3).map(|i| (i % 256) as u8).collect();

    let encoder = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Rgb);

    // Using Never should be equivalent to encode()
    let result = encoder.encode_with_stop(&data, enough::Never);
    assert!(result.is_ok());
    let jpeg = result.unwrap();
    assert_eq!(&jpeg[0..2], &[0xFF, 0xD8]);
}

#[test]
fn test_encode_with_stop_cancelled() {
    // Test that encoding with a pre-cancelled stopper returns Cancelled error
    let width = 64u32;
    let height = 64u32;
    let data: Vec<u8> = (0..width * height * 3).map(|i| (i % 256) as u8).collect();

    let encoder = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Rgb);

    // Create a stopper and cancel it immediately
    let stopper = almost_enough::Stopper::new();
    stopper.cancel();

    let result = encoder.encode_with_stop(&data, &stopper);
    assert!(matches!(result, Err(crate::Error::Cancelled)));
}

#[test]
fn test_encode_strip_based_with_stop_cancelled() {
    // Test that strip-based encoding also respects cancellation
    let width = 64u32;
    let height = 64u32;
    let data: Vec<u8> = (0..width * height * 3).map(|i| (i % 256) as u8).collect();

    let encoder = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Rgb)
        .encoding_backend(EncodingBackend::Strip);

    // Create a stopper and cancel it immediately
    let stopper = almost_enough::Stopper::new();
    stopper.cancel();

    let result = encoder.encode_with_stop(&data, &stopper);
    assert!(matches!(result, Err(crate::Error::Cancelled)));
}
