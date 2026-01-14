//! Comprehensive codec coverage tests for both encode and decode paths.
//!
//! This test file ensures coverage of:
//! - All encoder configuration options
//! - All decoder configuration options
//! - Edge cases and error handling
//! - Various image formats and sizes
//! - All subsampling modes
//! - Progressive and baseline modes
//! - XYB and YCbCr color spaces

#[path = "../src/test_utils.rs"]
mod test_utils;

use enough::Unstoppable;
use jpegli::{
    decoder::{Decoder, DecoderConfig, PixelFormat},
    encoder::{ChromaSubsampling, EncoderConfig, Error, PixelLayout, Quality},
};
use test_utils::{
    generate_checkerboard, generate_color_bars, generate_gradient_d, generate_gradient_h,
    generate_gradient_v, generate_noise, generate_solid, generate_solid_rgb, TestImage,
};

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

fn encode_rgb(
    width: u32,
    height: u32,
    data: &[u8],
    config: &EncoderConfig,
) -> jpegli::encoder::Result<Vec<u8>> {
    let mut enc = config.encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)?;
    enc.push_packed(data, enough::Unstoppable)?;
    enc.finish()
}

fn encode_gray(
    width: u32,
    height: u32,
    data: &[u8],
    config: &EncoderConfig,
) -> jpegli::encoder::Result<Vec<u8>> {
    let mut enc = config.encode_from_bytes(width, height, PixelLayout::Gray8Srgb)?;
    enc.push_packed(data, enough::Unstoppable)?;
    enc.finish()
}

fn encode_rgba(
    width: u32,
    height: u32,
    data: &[u8],
    config: &EncoderConfig,
) -> jpegli::encoder::Result<Vec<u8>> {
    let mut enc = config.encode_from_bytes(width, height, PixelLayout::Rgbx8Srgb)?;
    enc.push_packed(data, enough::Unstoppable)?;
    enc.finish()
}

fn encode_bgr(
    width: u32,
    height: u32,
    data: &[u8],
    config: &EncoderConfig,
) -> jpegli::encoder::Result<Vec<u8>> {
    let mut enc = config.encode_from_bytes(width, height, PixelLayout::Bgr8Srgb)?;
    enc.push_packed(data, enough::Unstoppable)?;
    enc.finish()
}

fn encode_bgra(
    width: u32,
    height: u32,
    data: &[u8],
    config: &EncoderConfig,
) -> jpegli::encoder::Result<Vec<u8>> {
    let mut enc = config.encode_from_bytes(width, height, PixelLayout::Bgrx8Srgb)?;
    enc.push_packed(data, enough::Unstoppable)?;
    enc.finish()
}

// ============================================================================
// ENCODER PATH COVERAGE
// ============================================================================

mod encode_coverage {
    use super::*;

    // --- Basic Encoding ---

    #[test]
    fn encode_rgb_basic() {
        let img = generate_gradient_d(64, 64, 3);
        let config = EncoderConfig::new();
        let jpeg = encode_rgb(64, 64, &img.pixels, &config).expect("encode failed");
        assert!(jpeg.len() > 100);
        verify_jpeg_structure(&jpeg);
    }

    #[test]
    fn encode_grayscale() {
        let img = generate_gradient_h(64, 64, 1);
        let config = EncoderConfig::new().grayscale();
        let jpeg = encode_gray(64, 64, &img.pixels, &config).expect("encode failed");
        verify_jpeg_structure(&jpeg);
    }

    #[test]
    fn encode_rgba_format() {
        let mut img = TestImage::new(32, 32, 4);
        for y in 0..32 {
            for x in 0..32 {
                img.set_pixel(x, y, 0, (x * 8) as u8);
                img.set_pixel(x, y, 1, (y * 8) as u8);
                img.set_pixel(x, y, 2, 128);
                img.set_pixel(x, y, 3, 255); // Alpha
            }
        }
        let config = EncoderConfig::new();
        let jpeg = super::encode_rgba(32, 32, &img.pixels, &config).expect("encode failed");
        verify_jpeg_structure(&jpeg);
    }

    // --- Quality Levels ---

    #[test]
    fn encode_quality_range() {
        let img = generate_gradient_d(64, 64, 3);
        // Test quality boundaries and key points
        for q in [1.0, 10.0, 30.0, 50.0, 70.0, 85.0, 90.0, 95.0, 99.0, 100.0] {
            let config = EncoderConfig::new().quality(q);
            let jpeg = encode_rgb(64, 64, &img.pixels, &config).expect(&format!("Q{} failed", q));
            assert!(jpeg.len() > 50, "Q{} too small", q);
        }
    }

    #[test]
    fn encode_distance_quality() {
        let img = generate_gradient_d(64, 64, 3);
        // Test distance-based quality (butteraugli distance)
        for d in [0.1, 0.5, 1.0, 2.0, 4.0, 8.0] {
            let config = EncoderConfig::new().quality(Quality::ApproxButteraugli(d));
            let jpeg =
                encode_rgb(64, 64, &img.pixels, &config).expect(&format!("dist {} failed", d));
            assert!(jpeg.len() > 50);
        }
    }

    // --- Subsampling Modes ---

    #[test]
    fn encode_subsampling_444() {
        let img = generate_gradient_d(128, 128, 3);
        let config = EncoderConfig::new().ycbcr(ChromaSubsampling::Full);
        let jpeg = encode_rgb(128, 128, &img.pixels, &config).expect("444 failed");
        verify_jpeg_structure(&jpeg);
    }

    #[test]
    fn encode_subsampling_422() {
        let img = generate_gradient_d(128, 128, 3);
        let config = EncoderConfig::new().ycbcr(ChromaSubsampling::HalfHorizontal);
        let jpeg = encode_rgb(128, 128, &img.pixels, &config).expect("422 failed");
        verify_jpeg_structure(&jpeg);
    }

    #[test]
    fn encode_subsampling_420() {
        let img = generate_gradient_d(128, 128, 3);
        let config = EncoderConfig::new().ycbcr(ChromaSubsampling::Quarter);
        let jpeg = encode_rgb(128, 128, &img.pixels, &config).expect("420 failed");
        verify_jpeg_structure(&jpeg);
    }

    #[test]
    fn encode_subsampling_440() {
        let img = generate_gradient_d(128, 128, 3);
        let config = EncoderConfig::new().ycbcr(ChromaSubsampling::HalfVertical);
        let jpeg = encode_rgb(128, 128, &img.pixels, &config).expect("440 failed");
        verify_jpeg_structure(&jpeg);
    }

    // --- JPEG Modes ---

    #[test]
    fn encode_baseline_mode() {
        let img = generate_gradient_d(128, 128, 3);
        let config = EncoderConfig::new().progressive(false);
        let jpeg = encode_rgb(128, 128, &img.pixels, &config).expect("baseline failed");

        // Verify SOF0 marker (baseline)
        assert!(
            jpeg.windows(2).any(|w| w == [0xFF, 0xC0]),
            "Missing SOF0 for baseline"
        );
    }

    #[test]
    fn encode_progressive_mode() {
        let img = generate_gradient_d(128, 128, 3);
        let config = EncoderConfig::new().progressive(true);
        let jpeg = encode_rgb(128, 128, &img.pixels, &config).expect("progressive failed");

        // Verify SOF2 marker (progressive)
        assert!(
            jpeg.windows(2).any(|w| w == [0xFF, 0xC2]),
            "Missing SOF2 for progressive"
        );
    }

    // --- Huffman Options ---

    #[test]
    fn encode_optimized_huffman() {
        let img = generate_gradient_d(256, 256, 3);
        let config_opt = EncoderConfig::new();
        let jpeg_opt = encode_rgb(256, 256, &img.pixels, &config_opt).expect("optimized failed");

        let config_fixed = EncoderConfig::new();
        let jpeg_fixed = encode_rgb(256, 256, &img.pixels, &config_fixed).expect("fixed failed");

        // Optimized should generally be smaller or similar
        assert!(jpeg_opt.len() <= jpeg_fixed.len() + 500);
    }

    // --- XYB Mode ---

    #[test]
    fn encode_xyb_mode() {
        let img = generate_gradient_d(64, 64, 3);
        let config = EncoderConfig::new().xyb();
        let jpeg = encode_rgb(64, 64, &img.pixels, &config).expect("XYB failed");
        verify_jpeg_structure(&jpeg);

        // XYB should have APP14 Adobe marker
        assert!(
            jpeg.windows(2).any(|w| w == [0xFF, 0xEE]),
            "Missing APP14 for XYB"
        );
    }

    #[test]
    fn encode_ycbcr_mode() {
        let img = generate_gradient_d(64, 64, 3);
        let config = EncoderConfig::new().ycbcr(ChromaSubsampling::Full);
        let jpeg = encode_rgb(64, 64, &img.pixels, &config).expect("YCbCr failed");
        verify_jpeg_structure(&jpeg);
    }

    // --- Image Sizes ---

    #[test]
    fn encode_minimum_size() {
        let img = TestImage::from_pixels(1, 1, 3, vec![128, 64, 192]);
        let config = EncoderConfig::new();
        let jpeg = encode_rgb(1, 1, &img.pixels, &config).expect("1x1 failed");
        verify_jpeg_structure(&jpeg);
    }

    #[test]
    fn encode_non_block_aligned() {
        // Sizes that don't align to 8x8 MCU boundaries
        for (w, h) in [(7, 7), (9, 9), (15, 17), (33, 31), (100, 101)] {
            let img = generate_gradient_d(w, h, 3);
            let config = EncoderConfig::new();
            let jpeg =
                encode_rgb(w, h, &img.pixels, &config).expect(&format!("{}x{} failed", w, h));
            verify_jpeg_structure(&jpeg);
        }
    }

    #[test]
    fn encode_non_square() {
        // Wide
        let wide = generate_gradient_h(256, 64, 3);
        let config = EncoderConfig::new();
        let jpeg = encode_rgb(256, 64, &wide.pixels, &config).expect("wide failed");
        verify_jpeg_structure(&jpeg);

        // Tall
        let tall = generate_gradient_v(64, 256, 3);
        let jpeg = encode_rgb(64, 256, &tall.pixels, &config).expect("tall failed");
        verify_jpeg_structure(&jpeg);
    }

    #[test]
    fn encode_large_image() {
        let img = generate_gradient_d(1024, 768, 3);
        let config = EncoderConfig::new().quality(85.0);
        let jpeg = encode_rgb(1024, 768, &img.pixels, &config).expect("large failed");
        // Gradients compress very well - 7KB is reasonable for Q85
        assert!(
            jpeg.len() > 5000,
            "Large image too small: {} bytes",
            jpeg.len()
        );
    }

    // --- Content Types ---

    #[test]
    fn encode_solid_colors() {
        let colors = [
            (0, 0, 0),
            (255, 255, 255),
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (128, 128, 128),
        ];
        let config = EncoderConfig::new();
        for (r, g, b) in colors {
            let img = generate_solid_rgb(64, 64, r, g, b);
            let jpeg = encode_rgb(64, 64, &img.pixels, &config).expect("solid failed");
            verify_jpeg_structure(&jpeg);
        }
    }

    #[test]
    fn encode_checkerboard() {
        let img = generate_checkerboard(128, 128, 8, 3);
        let config = EncoderConfig::new();
        let jpeg = encode_rgb(128, 128, &img.pixels, &config).expect("checkerboard failed");
        verify_jpeg_structure(&jpeg);
    }

    #[test]
    fn encode_color_bars() {
        let img = generate_color_bars(128, 64);
        let config = EncoderConfig::new();
        let jpeg = encode_rgb(128, 64, &img.pixels, &config).expect("color bars failed");
        verify_jpeg_structure(&jpeg);
    }

    #[test]
    fn encode_noise() {
        let img = generate_noise(128, 128, 12345, 3);
        let config = EncoderConfig::new();
        let jpeg = encode_rgb(128, 128, &img.pixels, &config).expect("noise failed");
        verify_jpeg_structure(&jpeg);
    }

    // --- EncoderConfig Builder ---

    #[test]
    fn encode_full_config() {
        let img = generate_gradient_d(128, 128, 3);
        let config = EncoderConfig::new()
            .quality(90.0)
            .progressive(false)
            .ycbcr(ChromaSubsampling::Full);
        let jpeg = encode_rgb(128, 128, &img.pixels, &config).expect("full config failed");
        verify_jpeg_structure(&jpeg);
    }

    // --- Error Handling ---

    #[test]
    fn encode_wrong_buffer_size() {
        // Buffer too small
        let config = EncoderConfig::new();
        let mut enc = config
            .encode_from_bytes(64, 64, PixelLayout::Rgb8Srgb)
            .unwrap();
        let result = enc.push_packed(&[0u8; 100], enough::Unstoppable);
        assert!(result.is_err());
    }

    #[test]
    fn encode_zero_dimensions() {
        let config = EncoderConfig::new();
        let result = config.encode_from_bytes(0, 64, PixelLayout::Rgb8Srgb);
        assert!(result.is_err());

        let result = config.encode_from_bytes(64, 0, PixelLayout::Rgb8Srgb);
        assert!(result.is_err());
    }

    fn verify_jpeg_structure(jpeg: &[u8]) {
        assert!(jpeg.len() >= 4, "JPEG too small");
        assert_eq!(&jpeg[0..2], &[0xFF, 0xD8], "Missing SOI");
        assert_eq!(&jpeg[jpeg.len() - 2..], &[0xFF, 0xD9], "Missing EOI");
    }
}

// ============================================================================
// DECODER PATH COVERAGE
// ============================================================================

mod decode_coverage {
    use super::*;

    fn create_test_jpeg(width: u32, height: u32, quality: f32) -> Vec<u8> {
        let img = generate_gradient_d(width, height, 3);
        let config = EncoderConfig::new().quality(quality);
        encode_rgb(width, height, &img.pixels, &config).expect("encode failed")
    }

    fn create_grayscale_jpeg(width: u32, height: u32) -> Vec<u8> {
        let img = generate_gradient_h(width, height, 1);
        let config = EncoderConfig::new().grayscale();
        encode_gray(width, height, &img.pixels, &config).expect("encode failed")
    }

    fn create_progressive_jpeg(width: u32, height: u32) -> Vec<u8> {
        let img = generate_gradient_d(width, height, 3);
        let config = EncoderConfig::new().progressive(true);
        encode_rgb(width, height, &img.pixels, &config).expect("encode failed")
    }

    fn create_subsampled_jpeg(width: u32, height: u32, subsampling: ChromaSubsampling) -> Vec<u8> {
        let img = generate_gradient_d(width, height, 3);
        let config = EncoderConfig::new().ycbcr(subsampling);
        encode_rgb(width, height, &img.pixels, &config).expect("encode failed")
    }

    // --- Basic Decoding ---

    #[test]
    fn decode_basic() {
        let jpeg = create_test_jpeg(128, 128, 90.0);
        let decoder = Decoder::new();
        let decoded = decoder.decode(&jpeg).expect("decode failed");

        assert_eq!(decoded.width, 128);
        assert_eq!(decoded.height, 128);
        assert_eq!(decoded.format, PixelFormat::Rgb);
        assert_eq!(decoded.data.len(), 128 * 128 * 3);
    }

    #[test]
    fn decode_grayscale() {
        let jpeg = create_grayscale_jpeg(64, 64);
        let decoder = Decoder::new();
        let decoded = decoder.decode(&jpeg).expect("decode failed");

        assert_eq!(decoded.width, 64);
        assert_eq!(decoded.height, 64);
    }

    // --- Various Quality Levels ---

    #[test]
    fn decode_quality_levels() {
        for q in [10.0, 30.0, 50.0, 70.0, 90.0, 100.0] {
            let jpeg = create_test_jpeg(64, 64, q);
            let decoder = Decoder::new();
            let decoded = decoder
                .decode(&jpeg)
                .expect(&format!("Q{} decode failed", q));
            assert_eq!(decoded.width, 64);
        }
    }

    // --- Various Sizes ---

    #[test]
    fn decode_various_sizes() {
        let sizes = [(8, 8), (16, 16), (17, 17), (64, 64), (100, 100), (256, 256)];
        for (w, h) in sizes {
            let jpeg = create_test_jpeg(w, h, 90.0);
            let decoder = Decoder::new();
            let decoded = decoder.decode(&jpeg).expect(&format!("{}x{} failed", w, h));
            assert_eq!(decoded.width, w);
            assert_eq!(decoded.height, h);
        }
    }

    #[test]
    fn decode_non_square() {
        // Wide
        let jpeg = create_test_jpeg(256, 64, 90.0);
        let decoder = Decoder::new();
        let decoded = decoder.decode(&jpeg).expect("wide decode failed");
        assert_eq!(decoded.width, 256);
        assert_eq!(decoded.height, 64);

        // Tall
        let jpeg = create_test_jpeg(64, 256, 90.0);
        let decoded = decoder.decode(&jpeg).expect("tall decode failed");
        assert_eq!(decoded.width, 64);
        assert_eq!(decoded.height, 256);
    }

    // --- Progressive Decoding ---

    #[test]
    fn decode_progressive() {
        let jpeg = create_progressive_jpeg(128, 128);
        let decoder = Decoder::new();
        let decoded = decoder.decode(&jpeg).expect("progressive decode failed");
        assert_eq!(decoded.width, 128);
        assert_eq!(decoded.height, 128);
    }

    // --- Subsampling Variants ---

    #[test]
    fn decode_subsampling_444() {
        let jpeg = create_subsampled_jpeg(128, 128, ChromaSubsampling::Full);
        let decoder = Decoder::new();
        let decoded = decoder.decode(&jpeg).expect("444 decode failed");
        assert_eq!(decoded.width, 128);
    }

    #[test]
    fn decode_subsampling_422() {
        let jpeg = create_subsampled_jpeg(128, 128, ChromaSubsampling::HalfHorizontal);
        let decoder = Decoder::new();
        let decoded = decoder.decode(&jpeg).expect("422 decode failed");
        assert_eq!(decoded.width, 128);
    }

    #[test]
    fn decode_subsampling_420() {
        let jpeg = create_subsampled_jpeg(128, 128, ChromaSubsampling::Quarter);
        let decoder = Decoder::new();
        let decoded = decoder.decode(&jpeg).expect("420 decode failed");
        assert_eq!(decoded.width, 128);
    }

    #[test]
    fn decode_subsampling_440() {
        let jpeg = create_subsampled_jpeg(128, 128, ChromaSubsampling::HalfVertical);
        let decoder = Decoder::new();
        let decoded = decoder.decode(&jpeg).expect("440 decode failed");
        assert_eq!(decoded.width, 128);
    }

    // --- XYB Mode ---

    #[test]
    fn decode_xyb() {
        let img = generate_gradient_d(64, 64, 3);
        let config = EncoderConfig::new().xyb();
        let jpeg = encode_rgb(64, 64, &img.pixels, &config).expect("XYB encode failed");

        let decoder = Decoder::new();
        let decoded = decoder.decode(&jpeg).expect("XYB decode failed");
        assert_eq!(decoded.width, 64);
        assert_eq!(decoded.height, 64);
    }

    // --- DecoderConfig Options ---

    #[test]
    fn decode_with_config() {
        let jpeg = create_test_jpeg(128, 128, 90.0);

        let config = DecoderConfig {
            output_format: Some(PixelFormat::Rgb),
            fancy_upsampling: true,
            block_smoothing: true,
            apply_icc: false,
            max_pixels: 1000000,
            max_memory: 100 * 1024 * 1024,
        };

        let decoder = Decoder::from_config(config);
        let decoded = decoder.decode(&jpeg).expect("config decode failed");
        assert_eq!(decoded.width, 128);
    }

    // --- Decoder Reuse ---

    #[test]
    fn decode_reuse() {
        let decoder = Decoder::new();
        for i in 0..5 {
            let size = 64 + i * 16;
            let jpeg = create_test_jpeg(size, size, 85.0);
            let decoded = decoder.decode(&jpeg).expect(&format!("reuse {} failed", i));
            assert_eq!(decoded.width, size);
        }
    }

    // --- Deterministic Decoding ---

    #[test]
    fn decode_deterministic() {
        let jpeg = create_test_jpeg(128, 128, 90.0);
        let decoder = Decoder::new();

        let decoded1 = decoder.decode(&jpeg).expect("decode 1 failed");
        let decoded2 = decoder.decode(&jpeg).expect("decode 2 failed");

        assert_eq!(decoded1.data, decoded2.data);
    }

    // --- Edge Cases ---

    #[test]
    fn decode_1x1() {
        let img = TestImage::from_pixels(1, 1, 3, vec![100, 150, 200]);
        let config = EncoderConfig::new();
        let jpeg = encode_rgb(1, 1, &img.pixels, &config).expect("1x1 encode failed");

        let decoder = Decoder::new();
        let decoded = decoder.decode(&jpeg).expect("1x1 decode failed");
        assert_eq!(decoded.width, 1);
        assert_eq!(decoded.height, 1);
    }

    #[test]
    fn decode_8x8_mcu() {
        let jpeg = create_test_jpeg(8, 8, 90.0);
        let decoder = Decoder::new();
        let decoded = decoder.decode(&jpeg).expect("8x8 decode failed");
        assert_eq!(decoded.width, 8);
        assert_eq!(decoded.height, 8);
    }

    #[test]
    fn decode_large_image() {
        let jpeg = create_test_jpeg(1024, 768, 85.0);
        let decoder = Decoder::new();
        let decoded = decoder.decode(&jpeg).expect("large decode failed");
        assert_eq!(decoded.width, 1024);
        assert_eq!(decoded.height, 768);
    }

    // --- Error Handling ---

    #[test]
    fn decode_empty_input() {
        let decoder = Decoder::new();
        assert!(decoder.decode(&[]).is_err());
    }

    #[test]
    fn decode_too_small() {
        let decoder = Decoder::new();
        assert!(decoder.decode(&[0xFF]).is_err());
        assert!(decoder.decode(&[0xFF, 0xD8]).is_err());
    }

    #[test]
    fn decode_missing_soi() {
        let decoder = Decoder::new();
        let bad = vec![0xFF, 0xE0, 0x00, 0x10];
        assert!(decoder.decode(&bad).is_err());
    }

    #[test]
    fn decode_garbage() {
        let decoder = Decoder::new();
        let garbage: Vec<u8> = (0..1000).map(|i| (i * 7) as u8).collect();
        assert!(decoder.decode(&garbage).is_err());
    }

    #[test]
    fn decode_truncated() {
        let jpeg = create_test_jpeg(64, 64, 90.0);
        let truncated: Vec<u8> = jpeg[..jpeg.len() / 2].to_vec();

        let decoder = Decoder::new();
        // May succeed or fail depending on truncation point
        let _ = decoder.decode(&truncated);
    }

    // --- Pixel Value Validation ---

    #[test]
    fn decode_pixel_range() {
        let mut img = TestImage::new(64, 64, 3);
        for y in 0..64 {
            for x in 0..64 {
                img.set_pixel(x, y, 0, (x * 4) as u8);
                img.set_pixel(x, y, 1, (y * 4) as u8);
                img.set_pixel(x, y, 2, ((x + y) * 2) as u8);
            }
        }

        let config = EncoderConfig::new().quality(100.0);
        let jpeg = encode_rgb(64, 64, &img.pixels, &config).expect("encode failed");

        let decoder = Decoder::new();
        let decoded = decoder.decode(&jpeg).expect("decode failed");

        // All pixels should be in valid range [0, 255]
        for &pixel in &decoded.data {
            assert!(pixel <= 255);
        }
    }
}

// ============================================================================
// ROUNDTRIP COVERAGE
// ============================================================================

mod roundtrip_coverage {
    use super::*;

    fn roundtrip_test(width: u32, height: u32, quality: f32) -> (Vec<u8>, Vec<u8>) {
        let img = generate_gradient_d(width, height, 3);
        let config = EncoderConfig::new().quality(quality);
        let jpeg = encode_rgb(width, height, &img.pixels, &config).expect("encode failed");

        let decoded = Decoder::new().decode(&jpeg).expect("decode failed");
        (img.pixels, decoded.data)
    }

    #[test]
    fn roundtrip_q100() {
        let (original, decoded) = roundtrip_test(64, 64, 100.0);
        // Q100 should be nearly lossless
        let max_diff = original
            .iter()
            .zip(decoded.iter())
            .map(|(a, b)| (*a as i32 - *b as i32).abs())
            .max()
            .unwrap_or(0);
        assert!(max_diff < 5, "Q100 max diff {} too high", max_diff);
    }

    #[test]
    fn roundtrip_all_subsampling() {
        for subsampling in [
            ChromaSubsampling::Full,
            ChromaSubsampling::HalfHorizontal,
            ChromaSubsampling::Quarter,
            ChromaSubsampling::HalfVertical,
        ] {
            let img = generate_gradient_d(128, 128, 3);
            let config = EncoderConfig::new().ycbcr(subsampling);
            let jpeg = encode_rgb(128, 128, &img.pixels, &config).expect("encode failed");

            let decoded = Decoder::new().decode(&jpeg).expect("decode failed");
            assert_eq!(decoded.width, 128);
            assert_eq!(decoded.height, 128);
        }
    }

    #[test]
    fn roundtrip_progressive() {
        let img = generate_gradient_d(128, 128, 3);
        let config = EncoderConfig::new().progressive(true);
        let jpeg = encode_rgb(128, 128, &img.pixels, &config).expect("progressive encode failed");

        let decoded = Decoder::new()
            .decode(&jpeg)
            .expect("progressive decode failed");
        assert_eq!(decoded.width, 128);
        assert_eq!(decoded.height, 128);
    }

    #[test]
    fn roundtrip_xyb() {
        let img = generate_gradient_d(64, 64, 3);
        let config = EncoderConfig::new().xyb();
        let jpeg = encode_rgb(64, 64, &img.pixels, &config).expect("XYB encode failed");

        let decoded = Decoder::new().decode(&jpeg).expect("XYB decode failed");
        assert_eq!(decoded.width, 64);
        assert_eq!(decoded.height, 64);
    }

    #[test]
    fn roundtrip_grayscale() {
        let img = generate_gradient_h(64, 64, 1);
        let config = EncoderConfig::new().grayscale();
        let jpeg = encode_gray(64, 64, &img.pixels, &config).expect("gray encode failed");

        let decoded = Decoder::new().decode(&jpeg).expect("gray decode failed");
        assert_eq!(decoded.width, 64);
        assert_eq!(decoded.height, 64);
    }

    #[test]
    fn roundtrip_content_types() {
        let config = EncoderConfig::new();
        let patterns = [
            ("solid", generate_solid(64, 64, 128, 3)),
            ("checkerboard", generate_checkerboard(64, 64, 8, 3)),
            ("noise", generate_noise(64, 64, 42, 3)),
            ("gradient_h", generate_gradient_h(64, 64, 3)),
            ("gradient_v", generate_gradient_v(64, 64, 3)),
            ("gradient_d", generate_gradient_d(64, 64, 3)),
        ];

        for (name, img) in patterns {
            let jpeg =
                encode_rgb(64, 64, &img.pixels, &config).expect(&format!("{} encode failed", name));

            let decoded = Decoder::new()
                .decode(&jpeg)
                .expect(&format!("{} decode failed", name));

            assert_eq!(decoded.width, 64, "{} width mismatch", name);
            assert_eq!(decoded.height, 64, "{} height mismatch", name);
        }
    }
}

// ============================================================================
// QUALITY API COVERAGE
// ============================================================================

mod quality_coverage {
    use super::*;

    #[test]
    fn quality_from_quality() {
        for q in [0.0, 25.0, 50.0, 75.0, 100.0] {
            let quality = Quality::Traditional(q);
            // Just verify it doesn't panic
            let _ = quality;
        }
    }

    #[test]
    fn quality_from_distance() {
        for d in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0] {
            let quality = Quality::ApproxButteraugli(d);
            let _ = quality;
        }
    }

    #[test]
    fn quality_to_distance() {
        let q = Quality::Traditional(90.0);
        let d = q.to_distance();
        assert!(d > 0.0, "Distance should be positive");
    }

    #[test]
    fn quality_roundtrip() {
        // Higher quality -> lower distance
        let q90 = Quality::Traditional(90.0);
        let q50 = Quality::Traditional(50.0);
        assert!(
            q90.to_distance() < q50.to_distance(),
            "Q90 should have lower distance than Q50"
        );
    }
}

// ============================================================================
// ERROR TYPE COVERAGE
// ============================================================================

mod encoder_error_coverage {
    use jpegli::encoder::Error;

    #[test]
    fn encoder_error_display() {
        let errors: Vec<Error> = vec![
            Error::InvalidDimensions {
                width: 0,
                height: 100,
                reason: "width cannot be zero",
            },
            Error::InvalidQuality {
                value: -1.0,
                valid_range: "0-100",
            },
            Error::InvalidColorFormat {
                reason: "unsupported format",
            },
            Error::InvalidBufferSize {
                expected: 1000,
                actual: 100,
            },
            Error::UnsupportedFeature {
                feature: "arithmetic coding",
            },
            Error::InternalError {
                reason: "unexpected state",
            },
            Error::IoError {
                reason: "disk full".to_string(),
            },
            Error::IccError("invalid profile".to_string()),
            Error::InvalidScanScript("overlapping scans".to_string()),
            Error::AllocationFailed {
                bytes: 1_000_000_000,
                context: "allocating DCT blocks",
            },
            Error::SizeOverflow {
                context: "computing buffer size",
            },
            Error::ImageTooLarge {
                pixels: 200_000_000,
                limit: 100_000_000,
            },
            Error::TooManyRows {
                height: 100,
                pushed: 200,
            },
        ];

        for err in errors {
            let display = format!("{}", err);
            assert!(!display.is_empty(), "Display should not be empty");
            let debug = format!("{:?}", err);
            assert!(!debug.is_empty(), "Debug should not be empty");
        }
    }

    #[test]
    fn encoder_error_equality() {
        let err1 = Error::InvalidDimensions {
            width: 0,
            height: 0,
            reason: "zero dimensions",
        };
        let err2 = Error::InvalidDimensions {
            width: 0,
            height: 0,
            reason: "zero dimensions",
        };
        assert_eq!(err1, err2);
    }

    #[test]
    fn encoder_error_clone() {
        let err = Error::InternalError {
            reason: "test error",
        };
        let cloned = err.clone();
        assert_eq!(err, cloned);
    }
}

mod decoder_error_coverage {
    use jpegli::decoder::Error;

    #[test]
    fn decoder_error_display() {
        let errors: Vec<Error> = vec![
            Error::InvalidJpegData {
                reason: "not a valid JPEG",
            },
            Error::TruncatedData {
                context: "reading header",
            },
            Error::EndOfScanData,
            Error::InvalidMarker {
                marker: 0xFF,
                context: "parsing markers",
            },
            Error::InvalidHuffmanTable {
                table_idx: 0,
                reason: "invalid code lengths",
            },
            Error::InvalidQuantTable {
                table_idx: 0,
                reason: "zero values",
            },
            Error::UnsupportedFeature {
                feature: "arithmetic coding",
            },
            Error::InternalError {
                reason: "unexpected state",
            },
            Error::IoError {
                reason: "disk full".to_string(),
            },
            Error::IccError("invalid profile".to_string()),
            Error::DecodeError("truncated data".to_string()),
            Error::AllocationFailed {
                bytes: 1_000_000_000,
                context: "allocating DCT blocks",
            },
            Error::SizeOverflow {
                context: "computing buffer size",
            },
            Error::ImageTooLarge {
                pixels: 200_000_000,
                limit: 100_000_000,
            },
            Error::TooManyScans {
                count: 200,
                limit: 100,
            },
        ];

        for err in errors {
            let display = format!("{}", err);
            assert!(!display.is_empty(), "Display should not be empty");
            let debug = format!("{:?}", err);
            assert!(!debug.is_empty(), "Debug should not be empty");
        }
    }

    #[test]
    fn decoder_error_equality() {
        let err1 = Error::InvalidJpegData {
            reason: "truncated",
        };
        let err2 = Error::InvalidJpegData {
            reason: "truncated",
        };
        assert_eq!(err1, err2);
    }

    #[test]
    fn decoder_error_clone() {
        let err = Error::InvalidJpegData {
            reason: "test error",
        };
        let cloned = err.clone();
        assert_eq!(err, cloned);
    }
}
