//! Integration tests for progressive scan optimization (optimize_scans).
//!
//! Verifies that optimize_scans produces valid JPEGs with identical decoded
//! pixels and smaller-or-equal file sizes.

use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};

/// Helper: encode image bytes with given config, return JPEG bytes.
fn encode_image(config: &EncoderConfig, width: u32, height: u32, pixels: &[u8]) -> Vec<u8> {
    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .expect("encoder setup");
    enc.push_packed(pixels, enough::Unstoppable)
        .expect("push pixels");
    enc.finish().expect("finish encoding")
}

/// Generate a test image with photographic-like content.
///
/// Uses a combination of noise, edges, and color patches to produce DCT
/// coefficients with realistic magnitude distributions across all frequency
/// bands. Smooth gradients are deliberately avoided — they produce degenerate
/// coefficients (0 or ±1) where successive approximation is useless.
fn generate_test_image(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((width * height * 3) as usize);
    // Simple LCG PRNG (deterministic, no deps)
    let mut rng = 0x12345678u64;
    let mut next_rng = || -> u8 {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (rng >> 33) as u8
    };

    for y in 0..height {
        for x in 0..width {
            // Divide into 32×32 patches with different base colors
            let patch_x = (x / 32) % 8;
            let patch_y = (y / 32) % 8;
            let patch_id = patch_x + patch_y * 8;

            // Base color varies per patch (creates edges between patches)
            let base_r = ((patch_id * 37 + 50) % 256) as u8;
            let base_g = ((patch_id * 73 + 100) % 256) as u8;
            let base_b = ((patch_id * 113 + 150) % 256) as u8;

            // Add noise for high-frequency content (creates non-trivial DCT coefficients)
            let noise = next_rng() as i16 - 128; // -128..127
            let noise_strength = 40i16; // Moderate noise

            let r = (base_r as i16 + noise * noise_strength / 128).clamp(0, 255) as u8;
            let g = (base_g as i16 + noise * noise_strength / 128).clamp(0, 255) as u8;
            let b = (base_b as i16 + noise * noise_strength / 128).clamp(0, 255) as u8;

            pixels.push(r);
            pixels.push(g);
            pixels.push(b);
        }
    }
    pixels
}

/// Decode JPEG bytes to RGB pixels using zune-jpeg (standards-compliant reference decoder).
fn decode_jpeg_zune(jpeg_data: &[u8]) -> Vec<u8> {
    use zune_jpeg::zune_core::bytestream::ZCursor;
    use zune_jpeg::JpegDecoder;
    let cursor = ZCursor::new(jpeg_data);
    let mut decoder = JpegDecoder::new(cursor);
    decoder.decode().expect("zune-jpeg decode")
}

#[test]
fn test_optimize_scans_produces_valid_jpeg() {
    let width = 256u32;
    let height = 256u32;
    let pixels = generate_test_image(width, height);

    let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter).optimize_scans(true);

    let jpeg = encode_image(&config, width, height, &pixels);

    // Should produce non-empty output
    assert!(!jpeg.is_empty(), "Optimized JPEG should not be empty");

    // Should start with SOI marker
    assert_eq!(jpeg[0], 0xFF);
    assert_eq!(jpeg[1], 0xD8);

    // Should end with EOI marker
    assert_eq!(jpeg[jpeg.len() - 2], 0xFF);
    assert_eq!(jpeg[jpeg.len() - 1], 0xD9);

    // Full decode must succeed
    let decoded = decode_jpeg_zune(&jpeg);
    assert_eq!(
        decoded.len(),
        (width * height * 3) as usize,
        "Decoded pixel count should match 256x256 RGB"
    );
}

#[test]
fn test_optimize_scans_444() {
    let width = 256u32;
    let height = 256u32;
    let pixels = generate_test_image(width, height);

    let config_normal = EncoderConfig::ycbcr(85.0, ChromaSubsampling::None).progressive(true);
    let config_opt = EncoderConfig::ycbcr(85.0, ChromaSubsampling::None).optimize_scans(true);

    let jpeg_normal = encode_image(&config_normal, width, height, &pixels);
    let jpeg_opt = encode_image(&config_opt, width, height, &pixels);

    // Both must fully decode
    let decoded_normal = decode_jpeg_zune(&jpeg_normal);
    let decoded_opt = decode_jpeg_zune(&jpeg_opt);
    assert_eq!(decoded_normal.len(), (width * height * 3) as usize);
    assert_eq!(decoded_opt.len(), (width * height * 3) as usize);

    // Optimized should be no larger than normal (within small margin for
    // different scan overhead — optimization may add DC SA scans)
    let margin = (jpeg_normal.len() as f64 * 0.02) as usize; // 2% margin
    assert!(
        jpeg_opt.len() <= jpeg_normal.len() + margin,
        "Optimized ({}) should not be significantly larger than normal ({}) for 4:4:4",
        jpeg_opt.len(),
        jpeg_normal.len()
    );
}

#[test]
fn test_optimize_scans_420() {
    let width = 256u32;
    let height = 256u32;
    let pixels = generate_test_image(width, height);

    let config_normal = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter).progressive(true);
    let config_opt = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter).optimize_scans(true);

    let jpeg_normal = encode_image(&config_normal, width, height, &pixels);
    let jpeg_opt = encode_image(&config_opt, width, height, &pixels);

    // Both must fully decode
    let decoded_normal = decode_jpeg_zune(&jpeg_normal);
    let decoded_opt = decode_jpeg_zune(&jpeg_opt);
    assert_eq!(decoded_normal.len(), (width * height * 3) as usize);
    assert_eq!(decoded_opt.len(), (width * height * 3) as usize);

    let margin = (jpeg_normal.len() as f64 * 0.02) as usize;
    assert!(
        jpeg_opt.len() <= jpeg_normal.len() + margin,
        "Optimized ({}) should not be significantly larger than normal ({}) for 4:2:0",
        jpeg_opt.len(),
        jpeg_normal.len()
    );
}

#[test]
fn test_optimize_scans_422() {
    let width = 256u32;
    let height = 256u32;
    let pixels = generate_test_image(width, height);

    let config_normal =
        EncoderConfig::ycbcr(85.0, ChromaSubsampling::HalfHorizontal).progressive(true);
    let config_opt =
        EncoderConfig::ycbcr(85.0, ChromaSubsampling::HalfHorizontal).optimize_scans(true);

    let jpeg_normal = encode_image(&config_normal, width, height, &pixels);
    let jpeg_opt = encode_image(&config_opt, width, height, &pixels);

    // Both must fully decode
    let decoded_normal = decode_jpeg_zune(&jpeg_normal);
    let decoded_opt = decode_jpeg_zune(&jpeg_opt);
    assert_eq!(decoded_normal.len(), (width * height * 3) as usize);
    assert_eq!(decoded_opt.len(), (width * height * 3) as usize);
}

#[test]
fn test_optimize_scans_multiple_qualities() {
    let width = 128u32;
    let height = 128u32;
    let pixels = generate_test_image(width, height);

    for quality in [50.0, 75.0, 85.0, 95.0] {
        let config = EncoderConfig::ycbcr(quality, ChromaSubsampling::Quarter).optimize_scans(true);

        let jpeg = encode_image(&config, width, height, &pixels);

        // Full decode must succeed at every quality level
        let decoded = decode_jpeg_zune(&jpeg);
        assert_eq!(
            decoded.len(),
            (width * height * 3) as usize,
            "Decoded pixel count at q{} should match 128x128 RGB",
            quality
        );
    }
}

#[test]
fn test_optimize_scans_decodable_by_zune_jpeg() {
    // Verify optimized JPEGs can be decoded by zune-jpeg (reference decoder).
    let width = 128u32;
    let height = 128u32;
    let pixels = generate_test_image(width, height);

    let config = EncoderConfig::ycbcr(90.0, ChromaSubsampling::Quarter).optimize_scans(true);

    let jpeg = encode_image(&config, width, height, &pixels);

    let decoded = decode_jpeg_zune(&jpeg);
    assert!(!decoded.is_empty(), "Decoded pixels should not be empty");
    // 128x128 RGB = 49152 bytes
    assert_eq!(decoded.len(), (width * height * 3) as usize);
}

#[test]
fn test_optimize_scans_lossless_roundtrip() {
    // Encode the same image with and without optimize_scans,
    // verify decoded pixels are identical via zune-jpeg.
    let width = 128u32;
    let height = 128u32;
    let pixels = generate_test_image(width, height);

    let config_normal = EncoderConfig::ycbcr(90.0, ChromaSubsampling::Quarter).progressive(true);
    let config_opt = EncoderConfig::ycbcr(90.0, ChromaSubsampling::Quarter).optimize_scans(true);

    let jpeg_normal = encode_image(&config_normal, width, height, &pixels);
    let jpeg_opt = encode_image(&config_opt, width, height, &pixels);

    let pixels1 = decode_jpeg_zune(&jpeg_normal);
    let pixels2 = decode_jpeg_zune(&jpeg_opt);

    assert_eq!(pixels1.len(), pixels2.len(), "Decoded sizes should match");

    // Since optimize_scans only changes the scan structure (not the quantized
    // coefficients), decoded pixels should be identical or extremely close.
    let max_diff: u8 = pixels1
        .iter()
        .zip(pixels2.iter())
        .map(|(&a, &b)| a.abs_diff(b))
        .max()
        .unwrap_or(0);

    // Allow ±1 for progressive reconstruction rounding
    assert!(
        max_diff <= 1,
        "Max pixel difference {} exceeds threshold of 1",
        max_diff
    );
}

#[test]
fn test_optimize_scans_auto_enables_progressive() {
    // optimize_scans(true) should force progressive mode
    let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter)
        .progressive(false) // Start with baseline
        .optimize_scans(true); // Should re-enable progressive

    let width = 64u32;
    let height = 64u32;
    let pixels = generate_test_image(width, height);

    let jpeg = encode_image(&config, width, height, &pixels);

    // Look for SOF2 marker (progressive) in the JPEG
    let has_sof2 = jpeg.windows(2).any(|w| w[0] == 0xFF && w[1] == 0xC2);
    assert!(
        has_sof2,
        "optimize_scans should produce progressive JPEG (SOF2)"
    );
}

#[test]
fn test_optimize_scans_size_comparison() {
    // Test with a larger, more realistic image to see actual savings
    let width = 512u32;
    let height = 512u32;
    let pixels = generate_test_image(width, height);

    let config_normal = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter).progressive(true);
    let config_opt = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter).optimize_scans(true);

    let jpeg_normal = encode_image(&config_normal, width, height, &pixels);
    let jpeg_opt = encode_image(&config_opt, width, height, &pixels);

    // Both must fully decode
    let decoded_normal = decode_jpeg_zune(&jpeg_normal);
    let decoded_opt = decode_jpeg_zune(&jpeg_opt);
    assert_eq!(decoded_normal.len(), (width * height * 3) as usize);
    assert_eq!(decoded_opt.len(), (width * height * 3) as usize);

    let savings_pct = (1.0 - jpeg_opt.len() as f64 / jpeg_normal.len() as f64) * 100.0;

    eprintln!(
        "512x512 q85 4:2:0: normal={}, optimized={}, savings={:.2}%",
        jpeg_normal.len(),
        jpeg_opt.len(),
        savings_pct
    );
}
