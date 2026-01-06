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
fn test_smoothing_factor() {
    // Create a high-frequency COLOR pattern that will show smoothing effects
    // (black/white won't work - chroma is constant for grayscale)
    let width = 64u32;
    let height = 64u32;
    let mut data = vec![0u8; (width * height * 3) as usize];

    // Create colorful checkerboard pattern (red/cyan alternating)
    for y in 0..height as usize {
        for x in 0..width as usize {
            let idx = (y * width as usize + x) * 3;
            if (x + y) % 2 == 0 {
                // Red
                data[idx] = 255;
                data[idx + 1] = 0;
                data[idx + 2] = 0;
            } else {
                // Cyan
                data[idx] = 0;
                data[idx + 1] = 255;
                data[idx + 2] = 255;
            }
        }
    }

    // Encode with 4:2:0 subsampling, no smoothing
    let jpeg_no_smooth = Encoder::new()
        .width(width)
        .height(height)
        .subsampling(Subsampling::S420)
        .smoothing_factor(0)
        .jpegli_quality(Quality::from_quality(90.0))
        .encode(&data)
        .expect("Encoding without smoothing failed");

    // Encode with 4:2:0 subsampling, moderate smoothing
    let jpeg_smooth_50 = Encoder::new()
        .width(width)
        .height(height)
        .subsampling(Subsampling::S420)
        .smoothing_factor(50)
        .jpegli_quality(Quality::from_quality(90.0))
        .encode(&data)
        .expect("Encoding with smoothing=50 failed");

    // Encode with 4:2:0 subsampling, max smoothing
    let jpeg_smooth_100 = Encoder::new()
        .width(width)
        .height(height)
        .subsampling(Subsampling::S420)
        .smoothing_factor(100)
        .jpegli_quality(Quality::from_quality(90.0))
        .encode(&data)
        .expect("Encoding with smoothing=100 failed");

    // All should produce valid JPEGs
    assert_eq!(jpeg_no_smooth[0], 0xFF);
    assert_eq!(jpeg_no_smooth[1], MARKER_SOI);
    assert_eq!(jpeg_smooth_50[0], 0xFF);
    assert_eq!(jpeg_smooth_50[1], MARKER_SOI);
    assert_eq!(jpeg_smooth_100[0], 0xFF);
    assert_eq!(jpeg_smooth_100[1], MARKER_SOI);

    // All should be decodable
    assert!(decode_zune(&jpeg_no_smooth[..]).is_ok());
    assert!(decode_zune(&jpeg_smooth_50[..]).is_ok());
    assert!(decode_zune(&jpeg_smooth_100[..]).is_ok());

    // Smoothing should reduce file size for high-frequency content
    // (blurring reduces chroma complexity)
    println!(
        "No smooth: {} bytes, Smooth 50: {} bytes, Smooth 100: {} bytes",
        jpeg_no_smooth.len(),
        jpeg_smooth_50.len(),
        jpeg_smooth_100.len()
    );
}

#[test]
fn test_smoothing_factor_444_noop() {
    // With 4:4:4 subsampling, smoothing should have no effect
    let width = 32u32;
    let height = 32u32;
    let data: Vec<u8> = (0..width * height * 3).map(|i| (i % 256) as u8).collect();

    let jpeg_no_smooth = Encoder::new()
        .width(width)
        .height(height)
        .subsampling(Subsampling::S444)
        .smoothing_factor(0)
        .jpegli_quality(Quality::from_quality(90.0))
        .encode(&data)
        .expect("Encoding 444 without smoothing failed");

    let jpeg_smooth = Encoder::new()
        .width(width)
        .height(height)
        .subsampling(Subsampling::S444)
        .smoothing_factor(100)
        .jpegli_quality(Quality::from_quality(90.0))
        .encode(&data)
        .expect("Encoding 444 with smoothing failed");

    // With 4:4:4, smoothing shouldn't change anything (no downsampling)
    assert_eq!(
        jpeg_no_smooth.len(),
        jpeg_smooth.len(),
        "4:4:4 should not be affected by smoothing_factor"
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
// Internal Pathway Tests (for benchmarking infrastructure)
// ========================================================================

#[test]
fn test_internal_pathway_valid_f32_none_444() {
    use internal_pathway::*;

    // P_F32_NONE should work with 4:4:4
    let encoder = Encoder::new()
        .width(16)
        .height(16)
        .subsampling(Subsampling::S444)
        .set_internal_pathway(P_F32_NONE);

    assert!(encoder.is_ok(), "P_F32_NONE with 4:4:4 should be valid");
}

#[test]
fn test_internal_pathway_valid_yuv_sharp_420() {
    use internal_pathway::*;

    // P_YUV_SHARP should work with 4:2:0
    let encoder = Encoder::new()
        .width(16)
        .height(16)
        .subsampling(Subsampling::S420)
        .set_internal_pathway(P_YUV_SHARP);

    assert!(encoder.is_ok(), "P_YUV_SHARP with 4:2:0 should be valid");
}

#[test]
fn test_internal_pathway_valid_f32_box_420() {
    use internal_pathway::*;

    // P_F32_BOX should work with 4:2:0
    let encoder = Encoder::new()
        .width(16)
        .height(16)
        .subsampling(Subsampling::S420)
        .set_internal_pathway(P_F32_BOX);

    assert!(encoder.is_ok(), "P_F32_BOX with 4:2:0 should be valid");
}

#[test]
fn test_internal_pathway_valid_f32_box_smooth50() {
    use internal_pathway::*;

    // P_F32_BOX_SMOOTH50 should work with 4:2:0
    let encoder = Encoder::new()
        .width(16)
        .height(16)
        .subsampling(Subsampling::S420)
        .set_internal_pathway(P_F32_BOX_SMOOTH50);

    assert!(
        encoder.is_ok(),
        "P_F32_BOX_SMOOTH50 with 4:2:0 should be valid"
    );
}

#[test]
fn test_internal_pathway_invalid_none_with_420() {
    use internal_pathway::*;

    // DOWNSAMPLE_NONE with 4:2:0 should fail
    let encoder = Encoder::new()
        .width(16)
        .height(16)
        .subsampling(Subsampling::S420)
        .set_internal_pathway(COLOR_INTRINSIC_F32 | DOWNSAMPLE_NONE);

    assert!(encoder.is_err(), "DOWNSAMPLE_NONE with 4:2:0 should fail");
}

#[test]
fn test_internal_pathway_invalid_sharp_with_444() {
    use internal_pathway::*;

    // DOWNSAMPLE_SHARP with 4:4:4 should fail
    let encoder = Encoder::new()
        .width(16)
        .height(16)
        .subsampling(Subsampling::S444)
        .set_internal_pathway(COLOR_YUV_BALANCED | DOWNSAMPLE_SHARP);

    assert!(encoder.is_err(), "DOWNSAMPLE_SHARP with 4:4:4 should fail");
}

#[test]
fn test_internal_pathway_invalid_sharp_with_440() {
    use internal_pathway::*;

    // DOWNSAMPLE_SHARP with 4:4:0 should fail (yuv crate doesn't support it)
    let encoder = Encoder::new()
        .width(16)
        .height(16)
        .subsampling(Subsampling::S440)
        .set_internal_pathway(COLOR_YUV_BALANCED | DOWNSAMPLE_SHARP);

    assert!(encoder.is_err(), "DOWNSAMPLE_SHARP with 4:4:0 should fail");
}

#[test]
fn test_internal_pathway_invalid_yuv_balanced_with_440() {
    use internal_pathway::*;

    // COLOR_YUV_BALANCED with 4:4:0 should fail (yuv crate doesn't support 4:4:0)
    let encoder = Encoder::new()
        .width(16)
        .height(16)
        .subsampling(Subsampling::S440)
        .set_internal_pathway(COLOR_YUV_BALANCED | DOWNSAMPLE_BOX);

    assert!(
        encoder.is_err(),
        "COLOR_YUV_BALANCED with 4:4:0 should fail"
    );
}

#[test]
fn test_internal_pathway_gamma_aware_420() {
    use internal_pathway::*;

    // DOWNSAMPLE_GAMMA_AWARE_F32 should work with 4:2:0
    let encoder = Encoder::new()
        .width(16)
        .height(16)
        .subsampling(Subsampling::S420)
        .set_internal_pathway(P_F32_GAMMA_AWARE);

    assert!(
        encoder.is_ok(),
        "DOWNSAMPLE_GAMMA_AWARE_F32 should work with 4:2:0"
    );
}

#[test]
fn test_internal_pathway_gamma_aware_invalid_with_444() {
    use internal_pathway::*;

    // DOWNSAMPLE_GAMMA_AWARE_F32 should fail with 4:4:4 (no downsampling needed)
    let encoder = Encoder::new()
        .width(16)
        .height(16)
        .subsampling(Subsampling::S444)
        .set_internal_pathway(P_F32_GAMMA_AWARE);

    assert!(
        encoder.is_err(),
        "DOWNSAMPLE_GAMMA_AWARE_F32 should fail with 4:4:4"
    );
}

#[test]
fn test_internal_pathway_gamma_aware_encode_420() {
    use internal_pathway::*;

    // Create a simple gradient test image
    let width = 32u32;
    let height = 32u32;
    let mut data = vec![0u8; (width * height * 3) as usize];

    for y in 0..height as usize {
        for x in 0..width as usize {
            let idx = (y * width as usize + x) * 3;
            data[idx] = (x * 8) as u8; // R
            data[idx + 1] = (y * 8) as u8; // G
            data[idx + 2] = ((x + y) * 4) as u8; // B
        }
    }

    let encoder = Encoder::new()
        .width(width)
        .height(height)
        .subsampling(Subsampling::S420)
        .set_internal_pathway(P_F32_GAMMA_AWARE)
        .expect("Should create encoder");

    let result = encoder.encode(&data);
    assert!(
        result.is_ok(),
        "Gamma-aware 4:2:0 encoding failed: {:?}",
        result.err()
    );

    let jpeg = result.unwrap();
    // Verify it's a valid JPEG
    assert_eq!(&jpeg[0..2], &[0xFF, 0xD8], "Should start with SOI");
    assert_eq!(
        &jpeg[jpeg.len() - 2..],
        &[0xFF, 0xD9],
        "Should end with EOI"
    );
    assert!(jpeg.len() > 100, "JPEG should have reasonable size");
}

#[test]
fn test_internal_pathway_gamma_aware_encode_422() {
    use internal_pathway::*;

    let width = 32u32;
    let height = 32u32;
    let data: Vec<u8> = (0..width * height * 3).map(|i| (i % 256) as u8).collect();

    let encoder = Encoder::new()
        .width(width)
        .height(height)
        .subsampling(Subsampling::S422)
        .set_internal_pathway(COLOR_INTRINSIC_F32 | DOWNSAMPLE_GAMMA_AWARE_F32)
        .expect("Should create encoder");

    let result = encoder.encode(&data);
    assert!(
        result.is_ok(),
        "Gamma-aware 4:2:2 encoding failed: {:?}",
        result.err()
    );

    let jpeg = result.unwrap();
    assert_eq!(&jpeg[0..2], &[0xFF, 0xD8]);
}

#[test]
fn test_internal_pathway_gamma_aware_encode_440() {
    use internal_pathway::*;

    let width = 32u32;
    let height = 32u32;
    let data: Vec<u8> = (0..width * height * 3).map(|i| (i % 256) as u8).collect();

    let encoder = Encoder::new()
        .width(width)
        .height(height)
        .subsampling(Subsampling::S440)
        .set_internal_pathway(COLOR_INTRINSIC_F32 | DOWNSAMPLE_GAMMA_AWARE_F32)
        .expect("Should create encoder");

    let result = encoder.encode(&data);
    assert!(
        result.is_ok(),
        "Gamma-aware 4:4:0 encoding failed: {:?}",
        result.err()
    );

    let jpeg = result.unwrap();
    assert_eq!(&jpeg[0..2], &[0xFF, 0xD8]);
}

#[test]
fn test_internal_pathway_gamma_aware_iterative_420() {
    use internal_pathway::*;

    // DOWNSAMPLE_GAMMA_AWARE_ITERATIVE should work with 4:2:0
    let encoder = Encoder::new()
        .width(16)
        .height(16)
        .subsampling(Subsampling::S420)
        .set_internal_pathway(P_F32_GAMMA_AWARE_ITERATIVE);

    assert!(
        encoder.is_ok(),
        "DOWNSAMPLE_GAMMA_AWARE_ITERATIVE should work with 4:2:0"
    );
}

#[test]
fn test_internal_pathway_gamma_aware_iterative_encode_420() {
    use internal_pathway::*;

    // Create a simple gradient test image
    let width = 32u32;
    let height = 32u32;
    let mut data = vec![0u8; (width * height * 3) as usize];

    for y in 0..height as usize {
        for x in 0..width as usize {
            let idx = (y * width as usize + x) * 3;
            data[idx] = (x * 8) as u8; // R
            data[idx + 1] = (y * 8) as u8; // G
            data[idx + 2] = ((x + y) * 4) as u8; // B
        }
    }

    let encoder = Encoder::new()
        .width(width)
        .height(height)
        .subsampling(Subsampling::S420)
        .set_internal_pathway(P_F32_GAMMA_AWARE_ITERATIVE)
        .expect("Should create encoder");

    let result = encoder.encode(&data);
    assert!(
        result.is_ok(),
        "Gamma-aware iterative 4:2:0 encoding failed: {:?}",
        result.err()
    );

    let jpeg = result.unwrap();
    // Verify it's a valid JPEG
    assert_eq!(&jpeg[0..2], &[0xFF, 0xD8], "Should start with SOI");
    assert_eq!(
        &jpeg[jpeg.len() - 2..],
        &[0xFF, 0xD9],
        "Should end with EOI"
    );
    assert!(jpeg.len() > 100, "JPEG should have reasonable size");
}

#[test]
fn test_internal_pathway_gamma_aware_iterative_encode_422() {
    use internal_pathway::*;

    let width = 32u32;
    let height = 32u32;
    let data: Vec<u8> = (0..width * height * 3).map(|i| (i % 256) as u8).collect();

    let encoder = Encoder::new()
        .width(width)
        .height(height)
        .subsampling(Subsampling::S422)
        .set_internal_pathway(COLOR_INTRINSIC_F32 | DOWNSAMPLE_GAMMA_AWARE_ITERATIVE)
        .expect("Should create encoder");

    let result = encoder.encode(&data);
    assert!(
        result.is_ok(),
        "Gamma-aware iterative 4:2:2 encoding failed: {:?}",
        result.err()
    );

    let jpeg = result.unwrap();
    assert_eq!(&jpeg[0..2], &[0xFF, 0xD8]);
}

#[test]
fn test_internal_pathway_gamma_aware_iterative_encode_440() {
    use internal_pathway::*;

    let width = 32u32;
    let height = 32u32;
    let data: Vec<u8> = (0..width * height * 3).map(|i| (i % 256) as u8).collect();

    let encoder = Encoder::new()
        .width(width)
        .height(height)
        .subsampling(Subsampling::S440)
        .set_internal_pathway(COLOR_INTRINSIC_F32 | DOWNSAMPLE_GAMMA_AWARE_ITERATIVE)
        .expect("Should create encoder");

    let result = encoder.encode(&data);
    assert!(
        result.is_ok(),
        "Gamma-aware iterative 4:4:0 encoding failed: {:?}",
        result.err()
    );

    let jpeg = result.unwrap();
    assert_eq!(&jpeg[0..2], &[0xFF, 0xD8]);
}

#[test]
fn test_internal_pathway_unimplemented_yuv_professional() {
    use internal_pathway::*;

    // COLOR_YUV_PROFESSIONAL should fail (requires feature)
    let encoder = Encoder::new()
        .width(16)
        .height(16)
        .subsampling(Subsampling::S420)
        .set_internal_pathway(COLOR_YUV_PROFESSIONAL | DOWNSAMPLE_BOX);

    assert!(
        encoder.is_err(),
        "COLOR_YUV_PROFESSIONAL should fail (not implemented)"
    );
}

#[test]
fn test_internal_pathway_invalid_color_byte() {
    // Invalid color conversion byte (4+)
    let encoder = Encoder::new()
        .width(16)
        .height(16)
        .subsampling(Subsampling::S444)
        .set_internal_pathway(4); // Invalid color byte

    assert!(encoder.is_err(), "Color byte 4 should be invalid");
}

#[test]
fn test_internal_pathway_invalid_downsample_byte() {
    use internal_pathway::*;

    // Invalid downsampling byte (7+)
    let encoder = Encoder::new()
        .width(16)
        .height(16)
        .subsampling(Subsampling::S444)
        .set_internal_pathway(COLOR_INTRINSIC_F32 | (7 << 8));

    assert!(encoder.is_err(), "Downsample byte 7 should be invalid");
}

#[test]
fn test_internal_pathway_invalid_smoothing_over_100() {
    use internal_pathway::*;

    // Smoothing > 100 should fail
    let encoder = Encoder::new()
        .width(16)
        .height(16)
        .subsampling(Subsampling::S420)
        .set_internal_pathway(with_smoothing(
            COLOR_INTRINSIC_F32 | DOWNSAMPLE_BOX_SMOOTHED,
            101,
        ));

    assert!(encoder.is_err(), "Smoothing factor 101 should be invalid");
}

#[test]
fn test_internal_pathway_invalid_smoothing_without_box_smoothed() {
    use internal_pathway::*;

    // Smoothing with non-BoxSmoothed downsampling should fail
    let encoder = Encoder::new()
        .width(16)
        .height(16)
        .subsampling(Subsampling::S420)
        .set_internal_pathway(with_smoothing(COLOR_INTRINSIC_F32 | DOWNSAMPLE_BOX, 50));

    assert!(
        encoder.is_err(),
        "Smoothing with DOWNSAMPLE_BOX should fail"
    );
}

#[test]
fn test_internal_pathway_invalid_reserved_bits() {
    use internal_pathway::*;

    // Reserved bits (32-63) should cause failure
    // Note: bits 24-31 are the huffman method byte, not reserved
    let encoder = Encoder::new()
        .width(16)
        .height(16)
        .subsampling(Subsampling::S444)
        .set_internal_pathway(P_F32_NONE | (1u64 << 32));

    assert!(encoder.is_err(), "Reserved bit 32 should be invalid");
}

#[test]
fn test_internal_pathway_with_smoothing_helper() {
    use internal_pathway::*;

    // with_smoothing helper should work correctly
    let pathway = with_smoothing(COLOR_INTRINSIC_F32 | DOWNSAMPLE_BOX_SMOOTHED, 75);
    assert_eq!(pathway & 0xFF, COLOR_INTRINSIC_F32);
    assert_eq!((pathway >> 8) & 0xFF, 3); // DOWNSAMPLE_BOX_SMOOTHED = 3
    assert_eq!((pathway >> 16) & 0xFF, 75);
}

fn decode_zune(data: &[u8]) -> std::result::Result<Vec<u8>, zune_jpeg::errors::DecodeErrors> {
    use zune_jpeg::zune_core::bytestream::ZCursor;
    use zune_jpeg::JpegDecoder;
    let cursor = ZCursor::new(data);
    let mut decoder = JpegDecoder::new(cursor);
    decoder.decode()
}

#[test]
fn test_internal_pathway_pipeline_encode_decode() {
    use internal_pathway::*;

    // Test that InternalPipeline roundtrips correctly
    let pipeline = InternalPipeline::from_u64(P_F32_BOX_SMOOTH50).unwrap();
    assert_eq!(
        pipeline.color_conversion,
        ColorConversionMethod::IntrinsicF32
    );
    assert_eq!(pipeline.downsampling, DownsamplingMethod::BoxSmoothed);
    assert_eq!(pipeline.smoothing_factor, 50);

    // Test encode/decode roundtrip
    let encoded = pipeline.to_u64();
    let decoded = InternalPipeline::from_u64(encoded).unwrap();
    assert_eq!(decoded.color_conversion, pipeline.color_conversion);
    assert_eq!(decoded.downsampling, pipeline.downsampling);
    assert_eq!(decoded.smoothing_factor, pipeline.smoothing_factor);
}
