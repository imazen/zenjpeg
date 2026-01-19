//! Tests verifying that QuantTableConfig and ZeroBiasConfig actually affect encoder output.

use jpegli::encoder::{
    ChromaSubsampling, EncoderConfig, PixelLayout, QuantTableConfig, ZeroBiasConfig,
};

/// Generate a simple test image (gradient)
fn generate_test_image(width: usize, height: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let r = ((x * 255) / width) as u8;
            let g = ((y * 255) / height) as u8;
            let b = (((x + y) * 127) / (width + height)) as u8;
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
fn test_custom_quant_tables_change_output() {
    let width = 64u32;
    let height = 64u32;
    let pixels = generate_test_image(width as usize, height as usize);

    // Encode with default (Perceptual) tables
    let config_default = EncoderConfig::ycbcr(75.0, ChromaSubsampling::None);
    let jpeg_default = encode_with_config(&config_default, &pixels, width, height);

    // Encode with custom tables (all 16s - uniform quantization)
    let uniform_table = [16.0f32; 64];
    let config_custom = EncoderConfig::ycbcr(75.0, ChromaSubsampling::None).quant_tables(
        QuantTableConfig::CustomBase {
            luma: uniform_table,
            cb: uniform_table,
            cr: uniform_table,
        },
    );
    let jpeg_custom = encode_with_config(&config_custom, &pixels, width, height);

    // The outputs should be different
    assert_ne!(
        jpeg_default, jpeg_custom,
        "Custom quant tables should produce different output than defaults"
    );

    // Also verify sizes are different (uniform tables are not optimized)
    assert_ne!(
        jpeg_default.len(),
        jpeg_custom.len(),
        "File sizes should differ with different quant tables"
    );

    println!(
        "Default size: {} bytes, Custom size: {} bytes",
        jpeg_default.len(),
        jpeg_custom.len()
    );
}

#[test]
fn test_exact_quant_tables_change_output() {
    let width = 64u32;
    let height = 64u32;
    let pixels = generate_test_image(width as usize, height as usize);

    // Encode with default tables
    let config_default = EncoderConfig::ycbcr(75.0, ChromaSubsampling::None);
    let jpeg_default = encode_with_config(&config_default, &pixels, width, height);

    // Encode with exact tables (very coarse quantization)
    let coarse_table = [32u16; 64];
    let config_exact =
        EncoderConfig::ycbcr(75.0, ChromaSubsampling::None).quant_tables(QuantTableConfig::Exact {
            luma: coarse_table,
            cb: coarse_table,
            cr: coarse_table,
        });
    let jpeg_exact = encode_with_config(&config_exact, &pixels, width, height);

    // Outputs should differ
    assert_ne!(
        jpeg_default, jpeg_exact,
        "Exact quant tables should produce different output"
    );

    println!(
        "Default size: {} bytes, Exact coarse size: {} bytes",
        jpeg_default.len(),
        jpeg_exact.len()
    );
}

#[test]
fn test_separate_cb_cr_tables_differ_from_shared() {
    let width = 64u32;
    let height = 64u32;
    let pixels = generate_test_image(width as usize, height as usize);

    // Use same table for both Cb and Cr
    let shared_chroma = [20.0f32; 64];
    let luma = [16.0f32; 64];

    let config_shared = EncoderConfig::ycbcr(75.0, ChromaSubsampling::None).quant_tables(
        QuantTableConfig::CustomBase {
            luma,
            cb: shared_chroma,
            cr: shared_chroma,
        },
    );
    let jpeg_shared = encode_with_config(&config_shared, &pixels, width, height);

    // Use different tables for Cb and Cr
    let cb_table = [20.0f32; 64];
    let mut cr_table = [20.0f32; 64];
    // Make Cr more aggressive (larger values = more quantization)
    for i in 0..64 {
        cr_table[i] *= 2.0;
    }

    let config_separate = EncoderConfig::ycbcr(75.0, ChromaSubsampling::None).quant_tables(
        QuantTableConfig::CustomBase {
            luma,
            cb: cb_table,
            cr: cr_table,
        },
    );
    let jpeg_separate = encode_with_config(&config_separate, &pixels, width, height);

    // Outputs should differ because Cr is quantized more aggressively
    assert_ne!(
        jpeg_shared, jpeg_separate,
        "Separate Cb/Cr tables should produce different output than shared"
    );

    // Separate should be smaller (more aggressive Cr quantization)
    assert!(
        jpeg_separate.len() < jpeg_shared.len(),
        "More aggressive Cr quantization should produce smaller file: {} vs {}",
        jpeg_separate.len(),
        jpeg_shared.len()
    );
}

#[test]
fn test_zero_bias_disabled_changes_output() {
    let width = 64u32;
    let height = 64u32;
    let pixels = generate_test_image(width as usize, height as usize);

    // Encode with default (Perceptual) zero bias
    let config_perceptual = EncoderConfig::ycbcr(75.0, ChromaSubsampling::None);
    let jpeg_perceptual = encode_with_config(&config_perceptual, &pixels, width, height);

    // Encode with disabled zero bias
    let config_disabled =
        EncoderConfig::ycbcr(75.0, ChromaSubsampling::None).zero_bias(ZeroBiasConfig::Disabled);
    let jpeg_disabled = encode_with_config(&config_disabled, &pixels, width, height);

    // Outputs should differ
    assert_ne!(
        jpeg_perceptual, jpeg_disabled,
        "Disabled zero bias should produce different output than perceptual"
    );

    println!(
        "Perceptual zero-bias size: {} bytes, Disabled size: {} bytes",
        jpeg_perceptual.len(),
        jpeg_disabled.len()
    );
}

#[test]
fn test_custom_zero_bias_changes_output() {
    let width = 64u32;
    let height = 64u32;
    let pixels = generate_test_image(width as usize, height as usize);

    // Encode with default zero bias
    let config_default = EncoderConfig::ycbcr(75.0, ChromaSubsampling::None);
    let jpeg_default = encode_with_config(&config_default, &pixels, width, height);

    // Encode with aggressive custom zero bias (high multipliers = more zeroing)
    let aggressive_mul = [1.0f32; 64]; // High multiplier
    let zero_offset = [0.5f32; 64]; // Moderate offset

    let config_aggressive =
        EncoderConfig::ycbcr(75.0, ChromaSubsampling::None).zero_bias(ZeroBiasConfig::Custom {
            luma: (aggressive_mul, zero_offset),
            cb: (aggressive_mul, zero_offset),
            cr: (aggressive_mul, zero_offset),
        });
    let jpeg_aggressive = encode_with_config(&config_aggressive, &pixels, width, height);

    // Outputs should differ
    assert_ne!(
        jpeg_default, jpeg_aggressive,
        "Custom zero bias should produce different output"
    );

    println!(
        "Default zero-bias size: {} bytes, Aggressive size: {} bytes",
        jpeg_default.len(),
        jpeg_aggressive.len()
    );
}

#[test]
fn test_tables_module_provides_defaults() {
    use jpegli::encoder::tables;

    // Verify we can access the default tables
    let ycbcr = &tables::BASE_QUANT_YCBCR;
    assert_eq!(ycbcr.len(), 192);

    // Extract components
    let luma = tables::luma_from_192(ycbcr);
    let cb = tables::cb_from_192(ycbcr);
    let cr = tables::cr_from_192(ycbcr);

    assert_eq!(luma.len(), 64);
    assert_eq!(cb.len(), 64);
    assert_eq!(cr.len(), 64);

    // Verify they're not all zeros
    assert!(luma.iter().any(|&v| v != 0.0));
    assert!(cb.iter().any(|&v| v != 0.0));
    assert!(cr.iter().any(|&v| v != 0.0));

    // Verify zero bias tables are accessible
    assert_eq!(tables::ZERO_BIAS_MUL_YCBCR_LQ.len(), 192);
    assert_eq!(tables::ZERO_BIAS_MUL_YCBCR_HQ.len(), 192);
}

#[test]
fn test_modified_default_tables_change_output() {
    use jpegli::encoder::tables;

    let width = 64u32;
    let height = 64u32;
    let pixels = generate_test_image(width as usize, height as usize);

    // Encode with perceptual defaults
    let config_default = EncoderConfig::ycbcr(75.0, ChromaSubsampling::None);
    let jpeg_default = encode_with_config(&config_default, &pixels, width, height);

    // Start with defaults and modify
    let mut luma = tables::luma_from_192(&tables::BASE_QUANT_YCBCR);
    let cb = tables::cb_from_192(&tables::BASE_QUANT_YCBCR);
    let cr = tables::cr_from_192(&tables::BASE_QUANT_YCBCR);

    // Modify: double the DC quantization (makes image blockier but smaller)
    luma[0] *= 2.0;

    let config_modified = EncoderConfig::ycbcr(75.0, ChromaSubsampling::None)
        .quant_tables(QuantTableConfig::CustomBase { luma, cb, cr });
    let jpeg_modified = encode_with_config(&config_modified, &pixels, width, height);

    // Outputs should differ
    assert_ne!(
        jpeg_default, jpeg_modified,
        "Modified default tables should produce different output"
    );

    println!(
        "Original defaults: {} bytes, Modified DC: {} bytes",
        jpeg_default.len(),
        jpeg_modified.len()
    );
}
