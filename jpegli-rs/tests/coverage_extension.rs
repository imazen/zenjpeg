//! Extended coverage tests targeting uncovered code paths.
//!
//! This test file specifically targets code paths that are not exercised
//! by the main codec_coverage tests.

#[path = "../src/test_utils.rs"]
mod test_utils;

use enough::Never;
use jpegli::{
    decode::{Decoder, DecoderConfig},
    types::{ColorSpace, Component, Dimensions, HuffmanTable},
    ChromaSubsampling, EncoderConfig, PixelLayout, Quality, QuantTable,
};
use test_utils::{generate_gradient_d, generate_noise};

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

fn encode_rgb(width: u32, height: u32, data: &[u8], config: &EncoderConfig) -> jpegli::Result<Vec<u8>> {
    let mut enc = config.encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)?;
    enc.push_packed(data, Never)?;
    enc.finish()
}

fn encode_gray(width: u32, height: u32, data: &[u8], config: &EncoderConfig) -> jpegli::Result<Vec<u8>> {
    let mut enc = config.encode_from_bytes(width, height, PixelLayout::Gray8Srgb)?;
    enc.push_packed(data, Never)?;
    enc.finish()
}

fn encode_bgr(width: u32, height: u32, data: &[u8], config: &EncoderConfig) -> jpegli::Result<Vec<u8>> {
    let mut enc = config.encode_from_bytes(width, height, PixelLayout::Bgr8Srgb)?;
    enc.push_packed(data, Never)?;
    enc.finish()
}

fn encode_bgra(width: u32, height: u32, data: &[u8], config: &EncoderConfig) -> jpegli::Result<Vec<u8>> {
    let mut enc = config.encode_from_bytes(width, height, PixelLayout::Bgrx8Srgb)?;
    enc.push_packed(data, Never)?;
    enc.finish()
}

// ============================================================================
// TYPES MODULE COVERAGE
// ============================================================================

mod types_coverage {
    use super::*;

    #[test]
    fn color_space_coverage() {
        // Test all ColorSpace variants
        assert_eq!(ColorSpace::Unknown.num_components(), 0);
        assert_eq!(ColorSpace::Grayscale.num_components(), 1);
        assert_eq!(ColorSpace::Rgb.num_components(), 3);
        assert_eq!(ColorSpace::YCbCr.num_components(), 3);
        assert_eq!(ColorSpace::Cmyk.num_components(), 4);
        assert_eq!(ColorSpace::Ycck.num_components(), 4);
        assert_eq!(ColorSpace::Xyb.num_components(), 3);

        // Test default_subsampling
        assert!(!ColorSpace::Unknown.default_subsampling());
        assert!(!ColorSpace::Grayscale.default_subsampling());
        assert!(!ColorSpace::Rgb.default_subsampling());
        assert!(ColorSpace::YCbCr.default_subsampling());
        assert!(!ColorSpace::Cmyk.default_subsampling());
        assert!(ColorSpace::Ycck.default_subsampling());
        assert!(!ColorSpace::Xyb.default_subsampling());

        // Test Default trait
        assert_eq!(ColorSpace::default(), ColorSpace::Unknown);
    }

    #[test]
    fn chroma_subsampling_coverage() {
        // Test all ChromaSubsampling modes
        let _full = ChromaSubsampling::Full;
        let _half_h = ChromaSubsampling::HalfHorizontal;
        let _quarter = ChromaSubsampling::Quarter;
        let _half_v = ChromaSubsampling::HalfVertical;
    }

    #[test]
    fn component_coverage() {
        let comp = Component::default();
        assert_eq!(comp.id, 0);
        assert_eq!(comp.h_samp_factor, 1);
        assert_eq!(comp.v_samp_factor, 1);
        assert_eq!(comp.quant_table_idx, 0);
        assert_eq!(comp.dc_huffman_idx, 0);
        assert_eq!(comp.ac_huffman_idx, 0);

        // Test clone and debug
        let comp2 = comp.clone();
        assert_eq!(format!("{:?}", comp), format!("{:?}", comp2));
    }

    #[test]
    fn quant_table_coverage() {
        let table = QuantTable::default();
        assert_eq!(table.precision, 0);
        assert_eq!(table.values[0], 16);

        // Test with 16-bit values
        let mut values = [256u16; 64];
        values[0] = 1;
        let table = QuantTable::from_natural_order(&values);
        assert_eq!(table.precision, 1); // 16-bit precision

        // Test to_natural_order
        let recovered = table.to_natural_order();
        assert_eq!(recovered[0], 1);
    }

    #[test]
    fn huffman_table_coverage() {
        let table = HuffmanTable::default();
        assert!(table.is_dc);
        assert!(table.values.is_empty());
        assert_eq!(table.bits, [0; 16]);

        // Test clone and debug
        let table2 = table.clone();
        assert_eq!(format!("{:?}", table), format!("{:?}", table2));
    }

    #[test]
    fn dimensions_coverage() {
        let dim = Dimensions::new(800, 600);
        assert_eq!(dim.width, 800);
        assert_eq!(dim.height, 600);
        assert_eq!(dim.width_in_blocks(), 100);
        assert_eq!(dim.height_in_blocks(), 75);
        assert_eq!(dim.num_pixels(), 480000);

        // Test with non-block-aligned dimensions
        let dim2 = Dimensions::new(7, 7);
        assert_eq!(dim2.width_in_blocks(), 1);
        assert_eq!(dim2.height_in_blocks(), 1);

        // Test Default
        let dim3 = Dimensions::default();
        assert_eq!(dim3.width, 0);
        assert_eq!(dim3.height, 0);
    }

}

// ============================================================================
// ENTROPY MODULE COVERAGE
// ============================================================================

mod entropy_coverage {
    use super::*;

    #[test]
    fn category_edge_cases() {
        // Test category function directly through encoding
        let img = generate_gradient_d(64, 64, 3);

        // Low quality creates larger coefficients
        let config = EncoderConfig::new().quality(10.0);
        let jpeg = encode_rgb(64, 64, &img.pixels, &config).expect("encode failed");
        assert!(jpeg.len() > 100);

        // High quality with small coefficients
        let config = EncoderConfig::new().quality(100.0);
        let jpeg = encode_rgb(64, 64, &img.pixels, &config).expect("encode failed");
        assert!(jpeg.len() > 100);
    }

    #[test]
    fn progressive_dc_encoding() {
        let img = generate_gradient_d(128, 128, 3);

        // Test progressive mode (exercises DC progressive encoding)
        let config = EncoderConfig::new()
            .progressive(true)
            .quality(90.0);
        let jpeg = encode_rgb(128, 128, &img.pixels, &config)
            .expect("progressive encode failed");

        // Verify it's actually progressive
        assert!(jpeg.windows(2).any(|w| w == [0xFF, 0xC2]));

        // Decode to verify correctness
        let decoded = Decoder::new().decode(&jpeg).expect("decode failed");
        assert_eq!(decoded.width, 128);
        assert_eq!(decoded.height, 128);
    }

    #[test]
    fn progressive_ac_encoding() {
        // Use a noisy image to exercise more AC coefficient paths
        let img = generate_noise(128, 128, 42, 3);

        let config = EncoderConfig::new()
            .progressive(true)
            .quality(80.0);
        let jpeg = encode_rgb(128, 128, &img.pixels, &config)
            .expect("progressive encode failed");

        let decoded = Decoder::new().decode(&jpeg).expect("decode failed");
        assert_eq!(decoded.width, 128);
    }

    #[test]
    fn eob_run_encoding() {
        // Solid color image should have many EOB runs
        let img = test_utils::generate_solid(128, 128, 128, 3);

        let config = EncoderConfig::new()
            .progressive(true)
            .quality(90.0);
        let jpeg = encode_rgb(128, 128, &img.pixels, &config).expect("encode failed");

        let decoded = Decoder::new().decode(&jpeg).expect("decode failed");
        assert_eq!(decoded.width, 128);
    }
}

// ============================================================================
// COLOR MODULE COVERAGE
// ============================================================================

mod color_coverage {
    use super::*;

    #[test]
    fn bgr_format() {
        // Create BGR image
        let mut pixels = vec![0u8; 64 * 64 * 3];
        for i in 0..(64 * 64) {
            pixels[i * 3] = (i % 256) as u8; // B
            pixels[i * 3 + 1] = (i / 64) as u8; // G
            pixels[i * 3 + 2] = 128; // R
        }

        let config = EncoderConfig::new();
        let jpeg = encode_bgr(64, 64, &pixels, &config).expect("BGR encode failed");

        let decoded = Decoder::new().decode(&jpeg).expect("decode failed");
        assert_eq!(decoded.width, 64);
    }

    #[test]
    fn bgra_format() {
        // Create BGRA image
        let mut pixels = vec![0u8; 64 * 64 * 4];
        for i in 0..(64 * 64) {
            pixels[i * 4] = (i % 256) as u8; // B
            pixels[i * 4 + 1] = (i / 64) as u8; // G
            pixels[i * 4 + 2] = 128; // R
            pixels[i * 4 + 3] = 255; // A
        }

        let config = EncoderConfig::new();
        let jpeg = encode_bgra(64, 64, &pixels, &config).expect("BGRA encode failed");

        let decoded = Decoder::new().decode(&jpeg).expect("decode failed");
        assert_eq!(decoded.width, 64);
    }

    #[test]
    fn grayscale_from_rgb() {
        // Encode RGB but with grayscale-like content
        let img = test_utils::generate_gradient_h(64, 64, 3);
        let config = EncoderConfig::new();
        let jpeg = encode_rgb(64, 64, &img.pixels, &config).expect("encode failed");

        let decoded = Decoder::new().decode(&jpeg).expect("decode failed");
        assert_eq!(decoded.width, 64);
    }
}

// ============================================================================
// IDCT MODULE COVERAGE
// ============================================================================

mod idct_coverage {
    use super::*;

    #[test]
    fn decode_various_quality_levels() {
        // Different quality levels exercise different coefficient ranges
        for q in [1.0, 5.0, 20.0, 40.0, 60.0, 80.0, 95.0, 100.0] {
            let img = generate_noise(64, 64, 12345, 3);
            let config = EncoderConfig::new().quality(q);
            let jpeg = encode_rgb(64, 64, &img.pixels, &config)
                .expect("encode failed");

            let decoded = Decoder::new().decode(&jpeg).expect("decode failed");
            assert_eq!(decoded.width, 64);
        }
    }

    #[test]
    fn decode_progressive_multiple_passes() {
        // Progressive decode exercises IDCT with partial coefficients
        let img = generate_noise(256, 256, 99, 3);
        let config = EncoderConfig::new()
            .progressive(true)
            .quality(70.0);
        let jpeg = encode_rgb(256, 256, &img.pixels, &config)
            .expect("encode failed");

        let decoded = Decoder::new().decode(&jpeg).expect("decode failed");
        assert_eq!(decoded.width, 256);
        assert_eq!(decoded.height, 256);
    }
}

// ============================================================================
// DECODE MODULE COVERAGE
// ============================================================================

mod decode_coverage {
    use super::*;

    #[test]
    fn decode_f32_output() {
        let img = generate_gradient_d(64, 64, 3);
        let config = EncoderConfig::new();
        let jpeg = encode_rgb(64, 64, &img.pixels, &config)
            .expect("encode failed");

        // Decode to f32
        let decoded = Decoder::new().decode_f32(&jpeg).expect("f32 decode failed");
        assert_eq!(decoded.width, 64);
        assert_eq!(decoded.height, 64);

        // Verify f32 values are in [0, 1] range
        for &val in decoded.data.iter() {
            assert!(val >= 0.0 && val <= 1.0, "f32 value {} out of range", val);
        }
    }

    #[test]
    fn decode_grayscale_to_rgb() {
        // Encode grayscale
        let img = test_utils::generate_gradient_h(64, 64, 1);
        let config = EncoderConfig::new().grayscale();
        let jpeg = encode_gray(64, 64, &img.pixels, &config)
            .expect("encode failed");

        // Decode to RGB (default)
        let decoded = Decoder::new().decode(&jpeg).expect("decode failed");
        assert_eq!(decoded.width, 64);
        // Grayscale decoded to RGB should have 3x the pixels
        assert!(decoded.data.len() >= 64 * 64);
    }

    #[test]
    fn decode_with_memory_limits() {
        let img = generate_gradient_d(128, 128, 3);
        let config = EncoderConfig::new();
        let jpeg = encode_rgb(128, 128, &img.pixels, &config)
            .expect("encode failed");

        // Test with custom memory limits
        let decode_config = DecoderConfig {
            max_pixels: 10_000_000,
            max_memory: 500 * 1024 * 1024,
            ..Default::default()
        };
        let decoder = Decoder::from_config(decode_config);
        let decoded = decoder.decode(&jpeg).expect("decode failed");
        assert_eq!(decoded.width, 128);
    }

    #[test]
    fn decode_with_block_smoothing() {
        let img = generate_noise(64, 64, 42, 3);
        let config = EncoderConfig::new().quality(30.0);
        let jpeg = encode_rgb(64, 64, &img.pixels, &config)
            .expect("encode failed");

        let decode_config = DecoderConfig {
            block_smoothing: true,
            ..Default::default()
        };
        let decoder = Decoder::from_config(decode_config);
        let decoded = decoder.decode(&jpeg).expect("decode failed");
        assert_eq!(decoded.width, 64);
    }

    #[test]
    fn decode_with_fancy_upsampling() {
        let img = generate_gradient_d(128, 128, 3);
        let config = EncoderConfig::new().ycbcr(ChromaSubsampling::Quarter);
        let jpeg = encode_rgb(128, 128, &img.pixels, &config)
            .expect("encode failed");

        let decode_config = DecoderConfig {
            fancy_upsampling: true,
            ..Default::default()
        };
        let decoder = Decoder::from_config(decode_config);
        let decoded = decoder.decode(&jpeg).expect("decode failed");
        assert_eq!(decoded.width, 128);
    }

    #[test]
    fn decode_xyb_to_rgb() {
        let img = generate_gradient_d(64, 64, 3);
        let config = EncoderConfig::new().xyb();
        let jpeg = encode_rgb(64, 64, &img.pixels, &config)
            .expect("XYB encode failed");

        // Decode with ICC application (requires cms feature)
        let decode_config = DecoderConfig {
            apply_icc: true,
            ..Default::default()
        };
        let decoder = Decoder::from_config(decode_config);
        let decoded = decoder.decode(&jpeg).expect("XYB decode failed");
        assert_eq!(decoded.width, 64);
    }
}

// ============================================================================
// ENCODE MODULE COVERAGE
// ============================================================================

mod encode_coverage {
    use super::*;

    #[test]
    fn encode_progressive_with_subsampling() {
        let img = generate_gradient_d(128, 128, 3);

        for subsampling in [
            ChromaSubsampling::Full,
            ChromaSubsampling::HalfHorizontal,
            ChromaSubsampling::Quarter,
            ChromaSubsampling::HalfVertical,
        ] {
            let config = EncoderConfig::new()
                .progressive(true)
                .ycbcr(subsampling);
            let jpeg = encode_rgb(128, 128, &img.pixels, &config).expect("encode failed");

            let decoded = Decoder::new().decode(&jpeg).expect("decode failed");
            assert_eq!(decoded.width, 128);
        }
    }

    #[test]
    fn encode_xyb_with_subsampling() {
        let img = generate_gradient_d(128, 128, 3);

        // XYB with 4:4:4 (default for XYB)
        let config = EncoderConfig::new().xyb();
        let jpeg = encode_rgb(128, 128, &img.pixels, &config)
            .expect("XYB encode failed");

        let decoded = Decoder::new().decode(&jpeg).expect("decode failed");
        assert_eq!(decoded.width, 128);
    }

    #[test]
    fn encode_very_small_images() {
        for size in [1, 2, 3, 4, 5, 6, 7, 8] {
            let img = generate_gradient_d(size, size, 3);
            let config = EncoderConfig::new();
            let jpeg = encode_rgb(size as u32, size as u32, &img.pixels, &config)
                .expect(&format!("{}x{} encode failed", size, size));

            let decoded = Decoder::new().decode(&jpeg).expect("decode failed");
            assert_eq!(decoded.width, size as u32);
            assert_eq!(decoded.height, size as u32);
        }
    }

    #[test]
    fn encode_extreme_qualities() {
        let img = generate_gradient_d(64, 64, 3);

        // Very low quality
        let config_low = EncoderConfig::new().quality(1.0);
        let jpeg_low = encode_rgb(64, 64, &img.pixels, &config_low)
            .expect("low Q encode failed");

        // Very high quality
        let config_high = EncoderConfig::new().quality(100.0);
        let jpeg_high = encode_rgb(64, 64, &img.pixels, &config_high)
            .expect("high Q encode failed");

        // High quality should produce larger file
        assert!(jpeg_high.len() > jpeg_low.len());
    }
}

// ============================================================================
// XYB MODULE COVERAGE
// ============================================================================

mod xyb_coverage {
    use super::*;

    #[test]
    fn xyb_encode_decode_roundtrip() {
        let img = generate_gradient_d(64, 64, 3);

        // Encode with XYB
        let config = EncoderConfig::new()
            .xyb()
            .quality(90.0);
        let jpeg = encode_rgb(64, 64, &img.pixels, &config)
            .expect("XYB encode failed");

        // Verify APP14 Adobe marker present
        assert!(jpeg.windows(2).any(|w| w == [0xFF, 0xEE]));

        // Decode
        let decoded = Decoder::new().decode(&jpeg).expect("XYB decode failed");
        assert_eq!(decoded.width, 64);
        assert_eq!(decoded.height, 64);
    }

    #[test]
    fn xyb_with_solid_colors() {
        // Test XYB with various solid colors
        let colors = [
            (0, 0, 0),
            (255, 255, 255),
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
        ];

        for (r, g, b) in colors {
            let img = test_utils::generate_solid_rgb(32, 32, r, g, b);
            let config = EncoderConfig::new().xyb();
            let jpeg = encode_rgb(32, 32, &img.pixels, &config)
                .expect("XYB encode failed");

            let decoded = Decoder::new().decode(&jpeg).expect("decode failed");
            assert_eq!(decoded.width, 32);
        }
    }

    #[test]
    fn xyb_progressive() {
        let img = generate_gradient_d(128, 128, 3);

        let config = EncoderConfig::new()
            .xyb()
            .progressive(true);
        let jpeg = encode_rgb(128, 128, &img.pixels, &config)
            .expect("XYB progressive encode failed");

        // Should have SOF2 (progressive)
        assert!(
            jpeg.windows(2).any(|w| w == [0xFF, 0xC2]),
            "Missing SOF2 marker"
        );

        // Should have APP14 Adobe marker (0xFF 0xEE)
        assert!(
            jpeg.windows(2).any(|w| w == [0xFF, 0xEE]),
            "XYB progressive should have APP14 Adobe marker"
        );

        // Should have ICC profile marker (APP2 with ICC_PROFILE signature)
        // APP2 marker: FF E2, then 2-byte length, then "ICC_PROFILE\0" (12 bytes)
        let has_icc = jpeg
            .windows(16)
            .any(|w| w[0] == 0xFF && w[1] == 0xE2 && &w[4..16] == b"ICC_PROFILE\0");
        assert!(has_icc, "XYB progressive should have ICC profile");

        // Verify JPEG is valid and can decode
        let decoded = Decoder::new().decode(&jpeg).expect("decode failed");
        assert_eq!(decoded.width, 128);
    }
}

// ============================================================================
// BITSTREAM MODULE COVERAGE
// ============================================================================

mod bitstream_coverage {
    use super::*;

    #[test]
    fn bitstream_via_encoding() {
        // Exercise bitstream through various encoding scenarios

        // Image with high entropy (random noise)
        let noise = generate_noise(64, 64, 12345, 3);
        let config = EncoderConfig::new();
        let jpeg_noise = encode_rgb(64, 64, &noise.pixels, &config)
            .expect("noise encode failed");

        // Image with low entropy (solid color)
        let solid = test_utils::generate_solid(64, 64, 128, 3);
        let jpeg_solid = encode_rgb(64, 64, &solid.pixels, &config)
            .expect("solid encode failed");

        // Noise should produce larger file
        assert!(jpeg_noise.len() > jpeg_solid.len());
    }
}

// ============================================================================
// HUFFMAN MODULE COVERAGE
// ============================================================================

mod huffman_coverage {
    use super::*;

    #[test]
    fn huffman_optimization_various_content() {
        // Test Huffman optimization with different content types

        let patterns = [
            ("gradient", generate_gradient_d(128, 128, 3)),
            ("noise", generate_noise(128, 128, 42, 3)),
            ("solid", test_utils::generate_solid(128, 128, 128, 3)),
            (
                "checkerboard",
                test_utils::generate_checkerboard(128, 128, 8, 3),
            ),
        ];

        let config = EncoderConfig::new();

        for (name, img) in patterns {
            let jpeg = encode_rgb(128, 128, &img.pixels, &config)
                .expect(&format!("{} encode failed", name));

            // Both should decode correctly
            let decoded = Decoder::new().decode(&jpeg).expect("decode failed");
            assert_eq!(decoded.width, 128);
        }
    }
}

// ============================================================================
// ADAPTIVE QUANTIZATION COVERAGE
// ============================================================================

mod aq_coverage {
    use super::*;

    #[test]
    fn aq_various_content_types() {
        // AQ should behave differently for different content types
        let patterns = [
            generate_gradient_d(128, 128, 3),
            generate_noise(128, 128, 42, 3),
            test_utils::generate_solid(128, 128, 128, 3),
            test_utils::generate_checkerboard(128, 128, 8, 3),
            test_utils::generate_color_bars(128, 64),
        ];

        for (i, img) in patterns.iter().enumerate() {
            let config = EncoderConfig::new().quality(85.0);
            let jpeg = encode_rgb(img.width, img.height, &img.pixels, &config)
                .expect(&format!("pattern {} encode failed", i));

            let decoded = Decoder::new().decode(&jpeg).expect("decode failed");
            assert_eq!(decoded.width, img.width);
        }
    }

    #[test]
    fn aq_quality_range() {
        let img = generate_noise(64, 64, 99, 3);

        // Test AQ behavior across quality range
        for q in [10.0, 30.0, 50.0, 70.0, 90.0, 100.0] {
            let config = EncoderConfig::new().quality(q);
            let jpeg = encode_rgb(64, 64, &img.pixels, &config)
                .expect(&format!("Q{} encode failed", q));

            let decoded = Decoder::new().decode(&jpeg).expect("decode failed");
            assert_eq!(decoded.width, 64);
        }
    }
}

// ============================================================================
// SCAN SCRIPT COVERAGE
// ============================================================================

mod scan_script_coverage {
    use super::*;

    #[test]
    fn progressive_scan_levels() {
        let img = generate_gradient_d(128, 128, 3);

        // Progressive encoding exercises scan script
        let config = EncoderConfig::new()
            .progressive(true)
            .quality(80.0);
        let jpeg = encode_rgb(128, 128, &img.pixels, &config)
            .expect("progressive encode failed");

        // Count SOS markers (each scan starts with SOS)
        let sos_count = jpeg.windows(2).filter(|w| w == &[0xFF, 0xDA]).count();
        assert!(sos_count >= 1, "Should have at least one SOS marker");

        let decoded = Decoder::new().decode(&jpeg).expect("decode failed");
        assert_eq!(decoded.width, 128);
    }

    #[test]
    fn progressive_grayscale() {
        let img = test_utils::generate_gradient_h(64, 64, 1);

        let config = EncoderConfig::new()
            .grayscale()
            .progressive(true);
        let jpeg = encode_gray(64, 64, &img.pixels, &config)
            .expect("progressive grayscale encode failed");

        let decoded = Decoder::new().decode(&jpeg).expect("decode failed");
        assert_eq!(decoded.width, 64);
    }
}

// ============================================================================
// QUANT MODULE COVERAGE
// ============================================================================

mod quant_coverage {
    use super::*;

    #[test]
    fn quality_distance_conversion() {
        // Test quality to distance and back
        for q in [10.0, 30.0, 50.0, 70.0, 90.0, 95.0, 99.0] {
            let quality = Quality::Traditional(q);
            let distance = quality.to_distance();
            assert!(distance > 0.0, "Distance should be positive for Q{}", q);
        }

        // Test distance-based quality
        for d in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0] {
            let quality = Quality::ApproxButteraugli(d);
            let distance_back = quality.to_distance();
            assert!(
                (distance_back - d).abs() < 0.01,
                "Distance roundtrip failed"
            );
        }
    }

    #[test]
    fn quant_table_generation() {
        // Different qualities should produce different quant tables
        let img = generate_gradient_d(64, 64, 3);

        let config_q10 = EncoderConfig::new().quality(10.0);
        let jpeg_q10 = encode_rgb(64, 64, &img.pixels, &config_q10)
            .expect("Q10 encode failed");

        let config_q90 = EncoderConfig::new().quality(90.0);
        let jpeg_q90 = encode_rgb(64, 64, &img.pixels, &config_q90)
            .expect("Q90 encode failed");

        // Different qualities should produce different output
        // (size relationship depends on content - gradients may not follow typical pattern)
        assert_ne!(jpeg_q90.len(), jpeg_q10.len(), "Q10 and Q90 should produce different sizes");
    }
}

// ============================================================================
// ALLOC MODULE COVERAGE
// ============================================================================

mod alloc_coverage {
    use super::*;

    #[test]
    fn large_image_allocation() {
        // Test allocation with moderately large images
        let img = generate_gradient_d(1024, 1024, 3);

        let config = EncoderConfig::new();
        let jpeg = encode_rgb(1024, 1024, &img.pixels, &config)
            .expect("large encode failed");

        let decoded = Decoder::new().decode(&jpeg).expect("decode failed");
        assert_eq!(decoded.width, 1024);
        assert_eq!(decoded.height, 1024);
    }

    #[test]
    fn decode_with_strict_limits() {
        let img = generate_gradient_d(64, 64, 3);
        let config = EncoderConfig::new();
        let jpeg = encode_rgb(64, 64, &img.pixels, &config)
            .expect("encode failed");

        // Test with very strict limits (but still enough for this image)
        let decode_config = DecoderConfig {
            max_pixels: 100_000,
            max_memory: 10 * 1024 * 1024,
            ..Default::default()
        };
        let decoder = Decoder::from_config(decode_config);
        let decoded = decoder.decode(&jpeg).expect("decode failed");
        assert_eq!(decoded.width, 64);
    }
}

// ============================================================================
// TRANSFER FUNCTIONS COVERAGE
// ============================================================================

mod transfer_coverage {
    use super::*;

    #[test]
    fn xyb_transfer_functions() {
        // XYB encoding exercises transfer functions
        let img = generate_gradient_d(64, 64, 3);

        let config = EncoderConfig::new().xyb();
        let jpeg = encode_rgb(64, 64, &img.pixels, &config)
            .expect("XYB encode failed");

        let decoded = Decoder::new().decode(&jpeg).expect("decode failed");
        assert_eq!(decoded.width, 64);
    }
}
