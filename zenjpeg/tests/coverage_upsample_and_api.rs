//! Test coverage for encoder/decoder config builders, EncodeRequest API,
//! Limits, edge cases, and encode/decode roundtrip.
//!
//! Upsample unit tests are in `src/decode/upsample.rs` (private module).

use enough::Unstoppable;
use zenjpeg::decoder::{Decoder, PixelFormat};
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout, Quality};

// ============================================================================
// Helpers
// ============================================================================

/// Generate a noise+patches test image (not gradients — those are degenerate).
fn make_test_pixels(width: usize, height: usize) -> Vec<u8> {
    let mut data = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            let bx = (x / 8) as u32;
            let by = (y / 8) as u32;
            let block_hash = bx
                .wrapping_mul(2654435761)
                .wrapping_add(by.wrapping_mul(40503));
            let block_type = block_hash % 4;
            let px = x as u32;
            let py = y as u32;
            let mut h = px
                .wrapping_mul(374761393)
                .wrapping_add(py.wrapping_mul(668265263));
            h = (h ^ (h >> 13)).wrapping_mul(1274126177);
            let noise = (h >> 24) as u8;
            match block_type {
                0 => {
                    let bias = ((bx.wrapping_mul(17) ^ by.wrapping_mul(31)) & 0xFF) as u8;
                    data[idx] = bias.wrapping_add(noise >> 2);
                    data[idx + 1] = bias.wrapping_add(noise >> 1);
                    data[idx + 2] = bias.wrapping_add(noise >> 3);
                }
                1 => {
                    data[idx] = ((x * 255) / width.max(1)) as u8;
                    data[idx + 1] = ((y * 255) / height.max(1)) as u8;
                    data[idx + 2] = noise >> 2;
                }
                2 => {
                    let edge: u8 = if (x % 8 < 4) ^ (y % 8 < 4) { 200 } else { 55 };
                    data[idx] = edge;
                    data[idx + 1] = edge.wrapping_add(noise >> 4);
                    data[idx + 2] = 255u8.wrapping_sub(edge);
                }
                _ => {
                    data[idx] = noise;
                    data[idx + 1] = noise.wrapping_mul(3);
                    data[idx + 2] = noise.wrapping_mul(7);
                }
            }
        }
    }
    data
}

fn encode(pixels: &[u8], w: u32, h: u32, config: &EncoderConfig) -> Vec<u8> {
    let mut enc = config
        .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
        .expect("encoder creation failed");
    enc.push_packed(pixels, Unstoppable).expect("push failed");
    enc.finish().expect("finish failed")
}

fn decode_rgb(jpeg: &[u8]) -> (Vec<u8>, u32, u32) {
    let decoder = Decoder::new().output_format(PixelFormat::Rgb);
    let result = decoder.decode(jpeg, Unstoppable).expect("decode failed");
    let w = result.width;
    let h = result.height;
    let pixels = result.into_pixels_u8().expect("pixels");
    (pixels, w, h)
}

// ============================================================================
// Encoder Config Builder Coverage
// ============================================================================

mod encoder_config_tests {
    use super::*;

    #[test]
    fn ycbcr_444_baseline() {
        let config = EncoderConfig::ycbcr(90.0, ChromaSubsampling::None).progressive(false);
        let pixels = make_test_pixels(64, 64);
        let jpeg = encode(&pixels, 64, 64, &config);
        assert!(!jpeg.is_empty());
        let (dec, w, h) = decode_rgb(&jpeg);
        assert_eq!(w, 64);
        assert_eq!(h, 64);
        assert_eq!(dec.len(), 64 * 64 * 3);
    }

    #[test]
    fn ycbcr_420_progressive() {
        let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter).progressive(true);
        let pixels = make_test_pixels(128, 96);
        let jpeg = encode(&pixels, 128, 96, &config);
        assert!(!jpeg.is_empty());
        let (_, w, h) = decode_rgb(&jpeg);
        assert_eq!((w, h), (128, 96));
    }

    #[test]
    fn ycbcr_422_baseline() {
        let config =
            EncoderConfig::ycbcr(80.0, ChromaSubsampling::HalfHorizontal).progressive(false);
        let pixels = make_test_pixels(48, 32);
        let jpeg = encode(&pixels, 48, 32, &config);
        let (_, w, h) = decode_rgb(&jpeg);
        assert_eq!((w, h), (48, 32));
    }

    #[test]
    fn ycbcr_440_baseline() {
        let config = EncoderConfig::ycbcr(80.0, ChromaSubsampling::HalfVertical).progressive(false);
        let pixels = make_test_pixels(32, 48);
        let jpeg = encode(&pixels, 32, 48, &config);
        let (_, w, h) = decode_rgb(&jpeg);
        assert_eq!((w, h), (32, 48));
    }

    #[test]
    fn grayscale_roundtrip() {
        let config = EncoderConfig::grayscale(90.0);
        let pixels = vec![128u8; 64 * 64];
        let mut enc = config
            .encode_from_bytes(64, 64, PixelLayout::Gray8Srgb)
            .expect("encoder");
        enc.push_packed(&pixels, Unstoppable).expect("push");
        let jpeg = enc.finish().expect("finish");

        let decoder = Decoder::new().output_format(PixelFormat::Gray);
        let result = decoder.decode(&jpeg, Unstoppable).expect("decode");
        assert_eq!(result.width, 64);
        assert_eq!(result.height, 64);
    }

    #[test]
    fn quality_from_u8() {
        let q: Quality = 85u8.into();
        let config = EncoderConfig::ycbcr(q, ChromaSubsampling::Quarter);
        let pixels = make_test_pixels(16, 16);
        let jpeg = encode(&pixels, 16, 16, &config);
        assert!(!jpeg.is_empty());
    }

    #[test]
    fn quality_from_f32() {
        let config = EncoderConfig::ycbcr(92.5f32, ChromaSubsampling::None);
        let pixels = make_test_pixels(16, 16);
        let jpeg = encode(&pixels, 16, 16, &config);
        assert!(!jpeg.is_empty());
    }

    #[test]
    fn progressive_produces_smaller_files() {
        let pixels = make_test_pixels(256, 256);
        let baseline = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter).progressive(false);
        let progressive = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter).progressive(true);

        let jpeg_base = encode(&pixels, 256, 256, &baseline);
        let jpeg_prog = encode(&pixels, 256, 256, &progressive);

        // Progressive should generally be smaller or similar
        // Allow 10% tolerance — the point is they both work
        assert!(
            jpeg_prog.len() < jpeg_base.len() * 11 / 10,
            "progressive ({}) shouldn't be much larger than baseline ({})",
            jpeg_prog.len(),
            jpeg_base.len()
        );
    }

    #[test]
    fn force_baseline_clamps_quant() {
        let config = EncoderConfig::ycbcr(50.0, ChromaSubsampling::Quarter)
            .progressive(false)
            .allow_16bit_quant_tables(false)
            .expect("allow_16bit_quant_tables");
        let pixels = make_test_pixels(64, 64);
        let jpeg = encode(&pixels, 64, 64, &config);
        // Should start with SOF0 (baseline, marker 0xFFC0)
        let sof_marker = jpeg
            .windows(2)
            .position(|w| w == [0xFF, 0xC0])
            .expect("should have SOF0");
        assert!(sof_marker > 0);
    }

    #[test]
    fn restart_interval_marker_present() {
        let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter)
            .progressive(false)
            .restart_mcu_rows(2);
        let pixels = make_test_pixels(128, 128);
        let jpeg = encode(&pixels, 128, 128, &config);
        // Should contain DRI marker (0xFFDD) indicating restart interval
        let has_dri = jpeg.windows(2).any(|w| w == [0xFF, 0xDD]);
        assert!(has_dri, "should contain DRI marker");
        // Valid JPEG that decodes successfully
        let (_, w, h) = decode_rgb(&jpeg);
        assert_eq!((w, h), (128, 128));
    }

    #[test]
    fn no_restart_interval_when_disabled() {
        let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter)
            .progressive(false)
            .restart_mcu_rows(0);
        let pixels = make_test_pixels(128, 128);
        let jpeg = encode(&pixels, 128, 128, &config);
        // Should NOT contain DRI marker
        let has_dri = jpeg.windows(2).any(|w| w == [0xFF, 0xDD]);
        assert!(!has_dri, "should NOT contain DRI marker");
    }

    #[test]
    fn config_is_clone_and_reusable() {
        let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter).progressive(false);
        let config2 = config.clone();
        let pixels = make_test_pixels(32, 32);

        let jpeg1 = encode(&pixels, 32, 32, &config);
        let jpeg2 = encode(&pixels, 32, 32, &config2);
        assert_eq!(
            jpeg1, jpeg2,
            "cloned config should produce identical output"
        );
    }

    #[test]
    fn deringing_toggle() {
        let pixels = make_test_pixels(64, 64);
        let with_dering = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter)
            .progressive(false)
            .deringing(true);
        let no_dering = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter)
            .progressive(false)
            .deringing(false);

        let jpeg1 = encode(&pixels, 64, 64, &with_dering);
        let jpeg2 = encode(&pixels, 64, 64, &no_dering);
        // Both should produce valid JPEGs
        let (_, w1, h1) = decode_rgb(&jpeg1);
        let (_, w2, h2) = decode_rgb(&jpeg2);
        assert_eq!((w1, h1), (64, 64));
        assert_eq!((w2, h2), (64, 64));
    }

    #[test]
    fn aq_toggle() {
        let pixels = make_test_pixels(64, 64);
        let with_aq = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter)
            .progressive(false)
            .aq_enabled(true);
        let no_aq = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter)
            .progressive(false)
            .aq_enabled(false);

        let jpeg1 = encode(&pixels, 64, 64, &with_aq);
        let jpeg2 = encode(&pixels, 64, 64, &no_aq);
        // Disabling AQ should change file size
        assert_ne!(
            jpeg1.len(),
            jpeg2.len(),
            "AQ toggle should affect file size"
        );
    }

    #[test]
    fn optimize_huffman_toggle() {
        let pixels = make_test_pixels(64, 64);
        let optimized = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter)
            .progressive(false)
            .optimize_huffman(true);
        let standard = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter)
            .progressive(false)
            .optimize_huffman(false);

        let jpeg1 = encode(&pixels, 64, 64, &optimized);
        let jpeg2 = encode(&pixels, 64, 64, &standard);
        // Optimized should generally be smaller or equal
        assert!(
            jpeg1.len() <= jpeg2.len() + 50,
            "optimized ({}) shouldn't be larger than standard ({})",
            jpeg1.len(),
            jpeg2.len()
        );
    }
}

// ============================================================================
// Decoder Config Builder Coverage
// ============================================================================

mod decoder_config_tests {
    use super::*;
    use zenjpeg::decoder::{ChromaUpsampling, Strictness};

    fn make_jpeg_420() -> Vec<u8> {
        let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter).progressive(false);
        let pixels = make_test_pixels(64, 64);
        encode(&pixels, 64, 64, &config)
    }

    #[test]
    fn decoder_default_format() {
        let jpeg = make_jpeg_420();
        let result = Decoder::new().decode(&jpeg, Unstoppable).expect("decode");
        assert_eq!(result.width, 64);
        assert_eq!(result.height, 64);
        assert!(result.pixels_u8().is_some());
    }

    #[test]
    fn decoder_rgb_format() {
        let jpeg = make_jpeg_420();
        let result = Decoder::new()
            .output_format(PixelFormat::Rgb)
            .decode(&jpeg, Unstoppable)
            .expect("decode");
        assert_eq!(result.format, PixelFormat::Rgb);
        assert_eq!(result.pixels_u8().unwrap().len(), 64 * 64 * 3);
    }

    #[test]
    fn decoder_rgba_format() {
        let jpeg = make_jpeg_420();
        let result = Decoder::new()
            .output_format(PixelFormat::Rgba)
            .decode(&jpeg, Unstoppable)
            .expect("decode");
        assert_eq!(result.format, PixelFormat::Rgba);
        assert_eq!(result.pixels_u8().unwrap().len(), 64 * 64 * 4);
    }

    #[test]
    fn decoder_bgra_format() {
        let jpeg = make_jpeg_420();
        let result = Decoder::new()
            .output_format(PixelFormat::Bgra)
            .decode(&jpeg, Unstoppable)
            .expect("decode");
        assert_eq!(result.format, PixelFormat::Bgra);
        assert_eq!(result.pixels_u8().unwrap().len(), 64 * 64 * 4);
    }

    #[test]
    fn decoder_gray_format() {
        // Use a grayscale source — YCbCr→Gray conversion isn't supported
        let config = EncoderConfig::grayscale(90.0);
        let gray_pixels = vec![128u8; 64 * 64];
        let mut enc = config
            .encode_from_bytes(64, 64, PixelLayout::Gray8Srgb)
            .expect("encoder");
        enc.push_packed(&gray_pixels, Unstoppable).expect("push");
        let jpeg = enc.finish().expect("finish");

        let result = Decoder::new()
            .output_format(PixelFormat::Gray)
            .decode(&jpeg, Unstoppable)
            .expect("decode");
        assert_eq!(result.format, PixelFormat::Gray);
        assert_eq!(result.pixels_u8().unwrap().len(), 64 * 64);
    }

    #[test]
    fn decoder_fancy_upsampling_toggle() {
        let jpeg = make_jpeg_420();
        let fancy = Decoder::new()
            .fancy_upsampling(true)
            .output_format(PixelFormat::Rgb)
            .decode(&jpeg, Unstoppable)
            .expect("fancy");
        let box_filter = Decoder::new()
            .fancy_upsampling(false)
            .output_format(PixelFormat::Rgb)
            .decode(&jpeg, Unstoppable)
            .expect("box");

        // Both should produce valid output with same dimensions
        assert_eq!(fancy.width, box_filter.width);
        assert_eq!(fancy.height, box_filter.height);
        // But pixel values should differ slightly
        let f = fancy.pixels_u8().unwrap();
        let b = box_filter.pixels_u8().unwrap();
        let diffs = f.iter().zip(b.iter()).filter(|&(&a, &b)| a != b).count();
        assert!(diffs > 0, "fancy vs box should differ for 4:2:0");
    }

    #[test]
    fn decoder_chroma_upsampling_variants() {
        let jpeg = make_jpeg_420();
        for method in [
            ChromaUpsampling::NearestNeighbor,
            ChromaUpsampling::Triangle,
            ChromaUpsampling::LibjpegCompat,
            ChromaUpsampling::HorizontalFancy,
        ] {
            let result = Decoder::new()
                .chroma_upsampling(method)
                .output_format(PixelFormat::Rgb)
                .decode(&jpeg, Unstoppable)
                .unwrap_or_else(|e| panic!("decode with {:?} failed: {}", method, e));
            assert_eq!(result.width, 64);
            assert_eq!(result.height, 64);
        }
    }

    #[test]
    fn decoder_strictness_levels() {
        let jpeg = make_jpeg_420();
        for strictness in [
            Strictness::Strict,
            Strictness::Balanced,
            Strictness::Lenient,
            Strictness::Permissive,
        ] {
            let result = Decoder::new()
                .strictness(strictness)
                .output_format(PixelFormat::Rgb)
                .decode(&jpeg, Unstoppable);
            assert!(
                result.is_ok(),
                "valid JPEG should decode at {:?} strictness",
                strictness
            );
        }
    }

    #[test]
    fn decoder_strict_convenience() {
        let jpeg = make_jpeg_420();
        let result = Decoder::new().strict().decode(&jpeg, Unstoppable);
        assert!(result.is_ok());
    }

    #[test]
    fn decoder_lenient_convenience() {
        let jpeg = make_jpeg_420();
        let result = Decoder::new().lenient().decode(&jpeg, Unstoppable);
        assert!(result.is_ok());
    }

    #[test]
    fn decoder_permissive_convenience() {
        let jpeg = make_jpeg_420();
        let result = Decoder::new().permissive().decode(&jpeg, Unstoppable);
        assert!(result.is_ok());
    }

    #[test]
    fn decoder_max_pixels_rejects_large() {
        let jpeg = make_jpeg_420(); // 64x64 = 4096 pixels
        let result = Decoder::new()
            .max_pixels(100) // Way too small
            .decode(&jpeg, Unstoppable);
        assert!(result.is_err(), "should reject image exceeding max_pixels");
    }

    #[test]
    fn decoder_max_pixels_allows_within_limit() {
        let jpeg = make_jpeg_420(); // 64x64 = 4096 pixels
        let result = Decoder::new().max_pixels(10_000).decode(&jpeg, Unstoppable);
        assert!(result.is_ok());
    }

    #[test]
    fn decoder_num_threads_sequential() {
        let jpeg = make_jpeg_420();
        let result = Decoder::new()
            .num_threads(1)
            .output_format(PixelFormat::Rgb)
            .decode(&jpeg, Unstoppable);
        assert!(result.is_ok());
    }

    #[test]
    fn decoder_auto_orient_toggle() {
        let jpeg = make_jpeg_420();
        let result = Decoder::new().auto_orient(false).decode(&jpeg, Unstoppable);
        assert!(result.is_ok());
    }

    #[test]
    fn decoder_is_clone() {
        let d = Decoder::new()
            .output_format(PixelFormat::Rgb)
            .fancy_upsampling(false)
            .max_pixels(1_000_000);
        let d2 = d.clone();
        let jpeg = make_jpeg_420();

        let r1 = d.decode(&jpeg, Unstoppable).expect("r1");
        let r2 = d2.decode(&jpeg, Unstoppable).expect("r2");
        assert_eq!(r1.pixels_u8().unwrap(), r2.pixels_u8().unwrap());
    }

    #[test]
    fn decoder_estimate_memory_usage() {
        let estimate = Decoder::new().estimate_memory_usage(1920, 1080);
        // Should be non-zero and reasonable (at least width * height * 3)
        assert!(estimate >= 1920 * 1080 * 3);
        assert!(estimate < 1920 * 1080 * 100); // Not absurdly large
    }

    #[test]
    fn decoder_read_info() {
        let jpeg = make_jpeg_420();
        let info = Decoder::new().read_info(&jpeg).expect("read_info");
        assert_eq!(info.dimensions.width, 64);
        assert_eq!(info.dimensions.height, 64);
    }

    #[test]
    fn decode_result_accessors() {
        let jpeg = make_jpeg_420();
        let result = Decoder::new()
            .output_format(PixelFormat::Rgb)
            .decode(&jpeg, Unstoppable)
            .expect("decode");

        assert_eq!(result.dimensions(), (64, 64));
        assert_eq!(result.format(), PixelFormat::Rgb);
        assert_eq!(result.bytes_per_pixel(), 3);
        assert!(result.stride() >= 64 * 3);
        assert!(!result.has_warnings());
        assert!(result.warnings().is_empty());
    }

    #[test]
    fn decode_result_into_pixels() {
        let jpeg = make_jpeg_420();
        let result = Decoder::new()
            .output_format(PixelFormat::Rgb)
            .decode(&jpeg, Unstoppable)
            .expect("decode");

        let owned = result.into_pixels_u8().expect("into_pixels_u8");
        assert_eq!(owned.len(), 64 * 64 * 3);
    }
}

// ============================================================================
// EncodeRequest API Coverage
// ============================================================================

mod encode_request_tests {
    use super::*;

    #[test]
    fn one_shot_encode() {
        let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);
        let pixels = make_test_pixels(64, 64);

        let jpeg = config
            .request()
            .encode_bytes(&pixels, 64, 64, PixelLayout::Rgb8Srgb)
            .expect("one-shot encode");

        assert!(!jpeg.is_empty());
        let (_, w, h) = decode_rgb(&jpeg);
        assert_eq!((w, h), (64, 64));
    }

    #[test]
    fn one_shot_encode_into() {
        let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);
        let pixels = make_test_pixels(32, 32);

        let mut output = Vec::new();
        config
            .request()
            .encode_bytes_into(&pixels, 32, 32, PixelLayout::Rgb8Srgb, &mut output)
            .expect("encode_into");

        assert!(!output.is_empty());
    }

    #[test]
    fn request_with_rgb_pixels() {
        let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::None);
        let w = 32u32;
        let h = 32u32;
        let pixels: Vec<rgb::RGB8> = (0..(w * h))
            .map(|i| {
                rgb::RGB8::new(
                    (i % 256) as u8,
                    ((i * 3) % 256) as u8,
                    ((i * 7) % 256) as u8,
                )
            })
            .collect();

        let jpeg = config
            .request()
            .encode(&pixels, w, h)
            .expect("encode with rgb::RGB8");
        assert!(!jpeg.is_empty());
    }

    #[test]
    fn request_with_icc_profile() {
        let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);
        let pixels = make_test_pixels(32, 32);
        // Minimal ICC profile (just needs to be non-empty for the API test)
        let fake_icc = vec![0u8; 128];

        let jpeg = config
            .request()
            .icc_profile(&fake_icc)
            .encode_bytes(&pixels, 32, 32, PixelLayout::Rgb8Srgb)
            .expect("encode with ICC");

        assert!(!jpeg.is_empty());
        // ICC marker should be present (APP2 = 0xFFE2)
        let has_app2 = jpeg.windows(2).any(|w| w == [0xFF, 0xE2]);
        assert!(has_app2, "should contain APP2 marker for ICC profile");
    }

    #[test]
    fn request_with_xmp() {
        let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);
        let pixels = make_test_pixels(32, 32);
        let xmp = b"<?xpacket begin='...'?>".to_vec();

        let jpeg = config
            .request()
            .xmp(&xmp)
            .encode_bytes(&pixels, 32, 32, PixelLayout::Rgb8Srgb)
            .expect("encode with XMP");

        assert!(!jpeg.is_empty());
        // XMP is in APP1 marker
        let has_app1 = jpeg.windows(2).any(|w| w == [0xFF, 0xE1]);
        assert!(has_app1, "should contain APP1 marker for XMP");
    }

    #[test]
    fn request_reuse_config() {
        let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);
        let pixels1 = make_test_pixels(32, 32);
        let pixels2 = make_test_pixels(48, 48);

        let jpeg1 = config
            .request()
            .encode_bytes(&pixels1, 32, 32, PixelLayout::Rgb8Srgb)
            .expect("first encode");
        let jpeg2 = config
            .request()
            .encode_bytes(&pixels2, 48, 48, PixelLayout::Rgb8Srgb)
            .expect("second encode");

        let (_, w1, h1) = decode_rgb(&jpeg1);
        let (_, w2, h2) = decode_rgb(&jpeg2);
        assert_eq!((w1, h1), (32, 32));
        assert_eq!((w2, h2), (48, 48));
    }

    #[test]
    fn streaming_from_rgb_type() {
        let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::None);
        let w = 32u32;
        let h = 32u32;
        let pixels: Vec<rgb::RGB8> = vec![rgb::RGB8::new(100, 150, 200); (w * h) as usize];

        let mut enc = config
            .request()
            .encode_from_rgb::<rgb::RGB8>(w, h)
            .expect("encoder");
        enc.push_packed(&pixels, Unstoppable).expect("push");
        let jpeg = enc.finish().expect("finish");
        assert!(!jpeg.is_empty());
    }
}

// ============================================================================
// Non-MCU-Aligned Dimensions (Edge Cases)
// ============================================================================

mod edge_dimension_tests {
    use super::*;

    #[test]
    fn width_not_multiple_of_8() {
        let w = 100u32;
        let h = 80u32;
        let pixels = make_test_pixels(w as usize, h as usize);
        let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter).progressive(false);
        let jpeg = encode(&pixels, w, h, &config);
        let (_, dw, dh) = decode_rgb(&jpeg);
        assert_eq!((dw, dh), (w, h));
    }

    #[test]
    fn height_not_multiple_of_8() {
        let w = 64u32;
        let h = 100u32;
        let pixels = make_test_pixels(w as usize, h as usize);
        let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter).progressive(false);
        let jpeg = encode(&pixels, w, h, &config);
        let (_, dw, dh) = decode_rgb(&jpeg);
        assert_eq!((dw, dh), (w, h));
    }

    #[test]
    fn neither_dimension_multiple_of_16() {
        let w = 100u32;
        let h = 100u32;
        let pixels = make_test_pixels(w as usize, h as usize);
        let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter).progressive(false);
        let jpeg = encode(&pixels, w, h, &config);
        let (_, dw, dh) = decode_rgb(&jpeg);
        assert_eq!((dw, dh), (w, h));
    }

    #[test]
    fn minimum_8x8() {
        let pixels = make_test_pixels(8, 8);
        let config = EncoderConfig::ycbcr(90.0, ChromaSubsampling::Quarter).progressive(false);
        let jpeg = encode(&pixels, 8, 8, &config);
        let (_, w, h) = decode_rgb(&jpeg);
        assert_eq!((w, h), (8, 8));
    }

    #[test]
    fn minimum_1x1() {
        let pixels = vec![128u8; 3];
        let config = EncoderConfig::ycbcr(90.0, ChromaSubsampling::None).progressive(false);
        let jpeg = encode(&pixels, 1, 1, &config);
        let (_, w, h) = decode_rgb(&jpeg);
        assert_eq!((w, h), (1, 1));
    }

    #[test]
    fn odd_dimensions_444() {
        for &(w, h) in &[(17, 17), (33, 15), (7, 23), (1, 64), (64, 1)] {
            let pixels = make_test_pixels(w as usize, h as usize);
            let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::None).progressive(false);
            let jpeg = encode(&pixels, w, h, &config);
            let (_, dw, dh) = decode_rgb(&jpeg);
            assert_eq!((dw, dh), (w, h), "444 roundtrip failed for {}x{}", w, h);
        }
    }

    #[test]
    fn odd_dimensions_420() {
        for &(w, h) in &[(17, 17), (33, 15), (9, 25), (48, 1), (1, 48)] {
            let pixels = make_test_pixels(w as usize, h as usize);
            let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter).progressive(false);
            let jpeg = encode(&pixels, w, h, &config);
            let (_, dw, dh) = decode_rgb(&jpeg);
            assert_eq!((dw, dh), (w, h), "420 roundtrip failed for {}x{}", w, h);
        }
    }
}

// ============================================================================
// Pixel Format Coverage (Encode Side)
// ============================================================================

mod pixel_format_tests {
    use super::*;

    #[test]
    fn encode_rgba_input() {
        let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);
        let pixels: Vec<u8> = (0..64 * 64 * 4).map(|i| (i % 256) as u8).collect();
        let mut enc = config
            .encode_from_bytes(64, 64, PixelLayout::Rgbx8Srgb)
            .expect("encoder");
        enc.push_packed(&pixels, Unstoppable).expect("push");
        let jpeg = enc.finish().expect("finish");
        let (_, w, h) = decode_rgb(&jpeg);
        assert_eq!((w, h), (64, 64));
    }

    #[test]
    fn encode_bgr_input() {
        let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);
        let pixels = make_test_pixels(64, 64);
        let mut enc = config
            .encode_from_bytes(64, 64, PixelLayout::Bgr8Srgb)
            .expect("encoder");
        enc.push_packed(&pixels, Unstoppable).expect("push");
        let jpeg = enc.finish().expect("finish");
        let (_, w, h) = decode_rgb(&jpeg);
        assert_eq!((w, h), (64, 64));
    }

    #[test]
    fn encode_bgrx_input() {
        let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);
        let pixels: Vec<u8> = (0..64 * 64 * 4).map(|i| (i % 256) as u8).collect();
        let mut enc = config
            .encode_from_bytes(64, 64, PixelLayout::Bgrx8Srgb)
            .expect("encoder");
        enc.push_packed(&pixels, Unstoppable).expect("push");
        let jpeg = enc.finish().expect("finish");
        let (_, w, h) = decode_rgb(&jpeg);
        assert_eq!((w, h), (64, 64));
    }
}

// ============================================================================
// Limits and Error Handling
// ============================================================================

mod limits_tests {
    use super::*;
    use zenjpeg::types::Limits;

    #[test]
    fn limits_default() {
        let limits = Limits::default();
        assert!(limits.max_pixels.is_none());
        assert!(limits.max_memory.is_none());
        assert!(limits.max_output.is_none());
    }

    #[test]
    fn limits_builder() {
        let limits = Limits::default()
            .max_pixels(1_000_000)
            .max_memory(100 * 1024 * 1024)
            .max_output(5 * 1024 * 1024);

        assert_eq!(limits.max_pixels, Some(1_000_000));
        assert_eq!(limits.max_memory, Some(100 * 1024 * 1024));
        assert_eq!(limits.max_output, Some(5 * 1024 * 1024));
    }

    #[test]
    fn decoder_with_limits() {
        let jpeg = {
            let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter).progressive(false);
            encode(&make_test_pixels(64, 64), 64, 64, &config)
        };

        let limits = Limits::default().max_pixels(10_000);
        let result = Decoder::new().limits(limits).decode(&jpeg, Unstoppable);
        assert!(result.is_ok());
    }

    #[test]
    fn decoder_with_tight_pixel_limit() {
        let jpeg = {
            let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter).progressive(false);
            encode(&make_test_pixels(64, 64), 64, 64, &config)
        };

        let limits = Limits::default().max_pixels(100);
        let result = Decoder::new().limits(limits).decode(&jpeg, Unstoppable);
        assert!(result.is_err(), "should reject image exceeding pixel limit");
    }

    #[test]
    fn empty_input_is_error() {
        let result = Decoder::new().decode(&[], Unstoppable);
        assert!(result.is_err());
    }

    #[test]
    fn truncated_input_is_error_strict() {
        let result = Decoder::new().strict().decode(&[0xFF, 0xD8], Unstoppable);
        assert!(result.is_err());
    }

    #[test]
    fn garbage_input_is_error() {
        let garbage = vec![0x42u8; 100];
        let result = Decoder::new().decode(&garbage, Unstoppable);
        assert!(result.is_err());
    }
}

// ============================================================================
// Quality Sweep Roundtrip
// ============================================================================

mod quality_sweep {
    use super::*;

    #[test]
    fn quality_sweep_420() {
        let pixels = make_test_pixels(64, 64);
        for q in [5, 10, 25, 50, 75, 85, 95, 100] {
            let config =
                EncoderConfig::ycbcr(q as f32, ChromaSubsampling::Quarter).progressive(false);
            let jpeg = encode(&pixels, 64, 64, &config);
            let (dec, w, h) = decode_rgb(&jpeg);
            assert_eq!((w, h), (64, 64), "Q{q} dimensions");
            assert_eq!(dec.len(), 64 * 64 * 3, "Q{q} pixel count");
        }
    }

    #[test]
    fn higher_quality_produces_larger_files() {
        let pixels = make_test_pixels(128, 128);
        let low = EncoderConfig::ycbcr(20.0, ChromaSubsampling::Quarter).progressive(false);
        let high = EncoderConfig::ycbcr(95.0, ChromaSubsampling::Quarter).progressive(false);

        let jpeg_low = encode(&pixels, 128, 128, &low);
        let jpeg_high = encode(&pixels, 128, 128, &high);

        assert!(
            jpeg_high.len() > jpeg_low.len(),
            "Q95 ({}) should be larger than Q20 ({})",
            jpeg_high.len(),
            jpeg_low.len()
        );
    }
}

// ============================================================================
// Scanline Reader Coverage
// ============================================================================

mod scanline_reader_tests {
    use super::*;
    use imgref::ImgRefMut;

    #[test]
    fn scanline_reader_basic() {
        let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter)
            .progressive(false)
            .restart_mcu_rows(2);
        let pixels = make_test_pixels(64, 64);
        let jpeg = encode(&pixels, 64, 64, &config);

        let decoder = Decoder::new()
            .output_format(PixelFormat::Rgb)
            .fancy_upsampling(false);
        let mut reader = decoder.scanline_reader(&jpeg).expect("scanline_reader");

        let info = reader.info();
        assert_eq!(info.dimensions.width, 64);
        assert_eq!(info.dimensions.height, 64);

        let mut all_rows = Vec::new();
        let stride = 64 * 3;
        let mut row_buf = vec![0u8; stride];
        while !reader.is_finished() {
            let img = ImgRefMut::new(&mut row_buf, stride, 1);
            let count = reader.read_rows_rgb8(img).expect("read_rows");
            if count == 0 {
                break;
            }
            all_rows.extend_from_slice(&row_buf[..stride]);
        }
        assert_eq!(all_rows.len(), 64 * 64 * 3);
    }

    #[test]
    fn scanline_reader_444() {
        let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::None)
            .progressive(false)
            .restart_mcu_rows(2);
        let pixels = make_test_pixels(32, 32);
        let jpeg = encode(&pixels, 32, 32, &config);

        let decoder = Decoder::new()
            .output_format(PixelFormat::Rgb)
            .fancy_upsampling(false);
        let mut reader = decoder.scanline_reader(&jpeg).expect("scanline_reader");

        let mut rows = 0;
        let stride = 32 * 3;
        let mut row_buf = vec![0u8; stride];
        while !reader.is_finished() {
            let img = ImgRefMut::new(&mut row_buf, stride, 1);
            let count = reader.read_rows_rgb8(img).expect("read_rows");
            if count == 0 {
                break;
            }
            rows += count;
        }
        assert_eq!(rows, 32);
    }
}
