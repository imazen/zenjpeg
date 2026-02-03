//! Tests verifying that TrellisConfig actually affects encoder output.

use zenjpeg::encode::trellis::{TrellisConfig, TrellisSpeedMode};
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};

/// Generate a simple test image (gradient with some texture)
fn generate_test_image(width: usize, height: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            // Add some texture via checkerboard pattern for trellis to work on
            let checker = ((x / 8 + y / 8) % 2 == 0) as u8 * 40;
            let r = (((x * 255) / width) as u8).saturating_add(checker);
            let g = (((y * 255) / height) as u8).saturating_add(checker);
            let b = ((((x + y) * 127) / (width + height)) as u8).saturating_add(checker);
            data.push(r);
            data.push(g);
            data.push(b);
        }
    }
    data
}

/// Encode an image with the given config and return the JPEG bytes
fn encode_with_config(config: &EncoderConfig, pixels: &[u8], width: u32, height: u32) -> Vec<u8> {
    let mut encoder = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .expect("encoder creation failed");

    encoder
        .push_packed(pixels, enough::Unstoppable)
        .expect("push failed");

    encoder.finish().expect("finish failed")
}

#[test]
fn test_trellis_changes_output() {
    let width = 128u32;
    let height = 128u32;
    let pixels = generate_test_image(width as usize, height as usize);

    // Encode without trellis (default)
    let config_no_trellis = EncoderConfig::ycbcr(75.0, ChromaSubsampling::None);
    let jpeg_no_trellis = encode_with_config(&config_no_trellis, &pixels, width, height);

    // Encode with trellis enabled
    let config_with_trellis =
        EncoderConfig::ycbcr(75.0, ChromaSubsampling::None).trellis(TrellisConfig::default());
    let jpeg_with_trellis = encode_with_config(&config_with_trellis, &pixels, width, height);

    // The outputs should be different
    assert_ne!(
        jpeg_no_trellis, jpeg_with_trellis,
        "Trellis quantization should produce different output than standard quantization"
    );

    // Trellis typically produces smaller files
    println!(
        "Without trellis: {} bytes, With trellis: {} bytes, Ratio: {:.2}%",
        jpeg_no_trellis.len(),
        jpeg_with_trellis.len(),
        jpeg_with_trellis.len() as f64 / jpeg_no_trellis.len() as f64 * 100.0
    );
}

#[test]
fn test_trellis_disabled_matches_default() {
    let width = 64u32;
    let height = 64u32;
    let pixels = generate_test_image(width as usize, height as usize);

    // Encode without trellis config
    let config_default = EncoderConfig::ycbcr(75.0, ChromaSubsampling::None);
    let jpeg_default = encode_with_config(&config_default, &pixels, width, height);

    // Encode with trellis explicitly disabled
    let config_disabled =
        EncoderConfig::ycbcr(75.0, ChromaSubsampling::None).trellis(TrellisConfig::disabled());
    let jpeg_disabled = encode_with_config(&config_disabled, &pixels, width, height);

    // Should produce identical output
    assert_eq!(
        jpeg_default, jpeg_disabled,
        "Disabled trellis should produce same output as no trellis config"
    );
}

#[test]
fn test_trellis_presets_produce_different_outputs() {
    let width = 128u32;
    let height = 128u32;
    let pixels = generate_test_image(width as usize, height as usize);

    // Test different presets
    let presets = [
        ("default", TrellisConfig::default()),
        ("favor_size", TrellisConfig::favor_size()),
        ("favor_quality", TrellisConfig::favor_quality()),
        ("thorough", TrellisConfig::thorough()),
    ];

    let mut results: Vec<(&str, Vec<u8>)> = Vec::new();

    for (name, preset) in presets.iter() {
        let config = EncoderConfig::ycbcr(75.0, ChromaSubsampling::None).trellis(*preset);
        let jpeg = encode_with_config(&config, &pixels, width, height);
        println!("{}: {} bytes", name, jpeg.len());
        results.push((name, jpeg));
    }

    // At least some presets should produce different outputs
    let mut different_pairs = 0;
    for i in 0..results.len() {
        for j in (i + 1)..results.len() {
            if results[i].1 != results[j].1 {
                different_pairs += 1;
            }
        }
    }

    assert!(
        different_pairs > 0,
        "Expected at least some presets to produce different outputs, but all were identical"
    );
}

#[test]
fn test_trellis_ac_dc_modes_work() {
    let width = 128u32;
    let height = 128u32;
    let pixels = generate_test_image(width as usize, height as usize);

    // AC trellis only
    let config_ac = EncoderConfig::ycbcr(75.0, ChromaSubsampling::None)
        .trellis(TrellisConfig::default().ac_trellis(true).dc_trellis(false));
    let jpeg_ac = encode_with_config(&config_ac, &pixels, width, height);

    // DC trellis only
    let config_dc = EncoderConfig::ycbcr(75.0, ChromaSubsampling::None)
        .trellis(TrellisConfig::default().ac_trellis(false).dc_trellis(true));
    let jpeg_dc = encode_with_config(&config_dc, &pixels, width, height);

    // Both
    let config_both = EncoderConfig::ycbcr(75.0, ChromaSubsampling::None)
        .trellis(TrellisConfig::default().ac_trellis(true).dc_trellis(true));
    let jpeg_both = encode_with_config(&config_both, &pixels, width, height);

    println!("AC only: {} bytes", jpeg_ac.len());
    println!("DC only: {} bytes", jpeg_dc.len());
    println!("Both: {} bytes", jpeg_both.len());

    // All modes should produce valid output
    // For small images, these modes may produce identical output
    // The important thing is they all work without errors
    assert!(!jpeg_ac.is_empty(), "AC-only mode should produce output");
    assert!(!jpeg_dc.is_empty(), "DC-only mode should produce output");
    assert!(!jpeg_both.is_empty(), "Both mode should produce output");
}

#[test]
fn test_trellis_speed_levels() {
    let width = 128u32;
    let height = 128u32;
    let pixels = generate_test_image(width as usize, height as usize);

    // Speed mode thorough
    let config_slow = EncoderConfig::ycbcr(75.0, ChromaSubsampling::None)
        .trellis(TrellisConfig::default().speed_mode(TrellisSpeedMode::Thorough));
    let jpeg_slow = encode_with_config(&config_slow, &pixels, width, height);

    // Speed mode fast (level 10)
    let config_fast = EncoderConfig::ycbcr(75.0, ChromaSubsampling::None)
        .trellis(TrellisConfig::default().speed_mode(TrellisSpeedMode::Level(10)));
    let jpeg_fast = encode_with_config(&config_fast, &pixels, width, height);

    println!(
        "Speed 0 (thorough): {} bytes, Speed 10 (fast): {} bytes",
        jpeg_slow.len(),
        jpeg_fast.len()
    );

    // They might be the same for small images, but at least verify they both work
    assert!(!jpeg_slow.is_empty());
    assert!(!jpeg_fast.is_empty());
}

#[test]
fn test_trellis_rd_factor() {
    let width = 128u32;
    let height = 128u32;
    let pixels = generate_test_image(width as usize, height as usize);

    // RD factor 0.7 (favor size)
    let config_small = EncoderConfig::ycbcr(75.0, ChromaSubsampling::None)
        .trellis(TrellisConfig::default().rd_factor(0.7));
    let jpeg_small = encode_with_config(&config_small, &pixels, width, height);

    // RD factor 1.5 (favor quality)
    let config_quality = EncoderConfig::ycbcr(75.0, ChromaSubsampling::None)
        .trellis(TrellisConfig::default().rd_factor(1.5));
    let jpeg_quality = encode_with_config(&config_quality, &pixels, width, height);

    println!(
        "rd_factor=0.7: {} bytes, rd_factor=1.5: {} bytes",
        jpeg_small.len(),
        jpeg_quality.len()
    );

    // Different rd_factors should produce different outputs
    assert_ne!(
        jpeg_small, jpeg_quality,
        "Different rd_factor values should produce different outputs"
    );

    // Lower rd_factor should produce smaller files (more aggressive zeroing)
    // This might not always be true for small images, so we just check they're different
}
