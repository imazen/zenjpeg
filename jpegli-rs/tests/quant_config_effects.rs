//! Tests verifying that EncodingTables actually affect encoder output.

use jpegli::encode::tuning::{EncodingTables, PerComponent, ScalingParams};
use jpegli::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};

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
    let tables = EncodingTables {
        quant: PerComponent {
            c0: uniform_table,
            c1: uniform_table,
            c2: uniform_table,
        },
        scaling: ScalingParams::Exact, // Use exact values, no quality scaling
        ..EncodingTables::default_ycbcr()
    };
    let config_custom =
        EncoderConfig::ycbcr(75.0, ChromaSubsampling::None).tables(Box::new(tables));
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
    let coarse_table = [32.0f32; 64];
    let tables = EncodingTables {
        quant: PerComponent {
            c0: coarse_table,
            c1: coarse_table,
            c2: coarse_table,
        },
        scaling: ScalingParams::Exact, // Use exact values
        ..EncodingTables::default_ycbcr()
    };
    let config_exact = EncoderConfig::ycbcr(75.0, ChromaSubsampling::None).tables(Box::new(tables));
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

    let tables_shared = EncodingTables {
        quant: PerComponent {
            c0: luma,
            c1: shared_chroma,
            c2: shared_chroma,
        },
        scaling: ScalingParams::Exact,
        ..EncodingTables::default_ycbcr()
    };
    let config_shared =
        EncoderConfig::ycbcr(75.0, ChromaSubsampling::None).tables(Box::new(tables_shared));
    let jpeg_shared = encode_with_config(&config_shared, &pixels, width, height);

    // Use different tables for Cb and Cr
    let cb_table = [20.0f32; 64];
    let mut cr_table = [20.0f32; 64];
    // Make Cr more aggressive (larger values = more quantization)
    for i in 0..64 {
        cr_table[i] *= 2.0;
    }

    let tables_separate = EncodingTables {
        quant: PerComponent {
            c0: luma,
            c1: cb_table,
            c2: cr_table,
        },
        scaling: ScalingParams::Exact,
        ..EncodingTables::default_ycbcr()
    };
    let config_separate =
        EncoderConfig::ycbcr(75.0, ChromaSubsampling::None).tables(Box::new(tables_separate));
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
fn test_zero_bias_changes_output() {
    let width = 64u32;
    let height = 64u32;
    let pixels = generate_test_image(width as usize, height as usize);

    // Encode with default (Perceptual) zero bias
    let config_perceptual = EncoderConfig::ycbcr(75.0, ChromaSubsampling::None);
    let jpeg_perceptual = encode_with_config(&config_perceptual, &pixels, width, height);

    // Encode with disabled zero bias (all zeros)
    let no_bias_mul = [0.0f32; 64];
    let tables_no_bias = EncodingTables {
        zero_bias_mul: PerComponent {
            c0: no_bias_mul,
            c1: no_bias_mul,
            c2: no_bias_mul,
        },
        zero_bias_offset_dc: [0.0; 3],
        zero_bias_offset_ac: [0.0; 3],
        ..EncodingTables::default_ycbcr()
    };
    let config_disabled =
        EncoderConfig::ycbcr(75.0, ChromaSubsampling::None).tables(Box::new(tables_no_bias));
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
    let aggressive_mul = [0.9f32; 64];
    let tables_aggressive = EncodingTables {
        zero_bias_mul: PerComponent {
            c0: aggressive_mul,
            c1: aggressive_mul,
            c2: aggressive_mul,
        },
        zero_bias_offset_dc: [0.5; 3],
        zero_bias_offset_ac: [0.5; 3],
        ..EncodingTables::default_ycbcr()
    };
    let config_aggressive =
        EncoderConfig::ycbcr(75.0, ChromaSubsampling::None).tables(Box::new(tables_aggressive));
    let jpeg_aggressive = encode_with_config(&config_aggressive, &pixels, width, height);

    // Outputs should differ
    assert_ne!(
        jpeg_default, jpeg_aggressive,
        "Custom zero bias should produce different output"
    );

    // Note: Aggressive zero bias typically produces smaller files (more coefficients zeroed),
    // but this isn't guaranteed due to entropy coding effects on small images.
    // The important thing is that custom zero bias produces different output.

    println!(
        "Default zero-bias size: {} bytes, Aggressive size: {} bytes",
        jpeg_default.len(),
        jpeg_aggressive.len()
    );
}

#[test]
fn test_combined_quant_and_zero_bias_changes() {
    let width = 64u32;
    let height = 64u32;
    let pixels = generate_test_image(width as usize, height as usize);

    // Encode with default tables
    let config_default = EncoderConfig::ycbcr(75.0, ChromaSubsampling::None);
    let jpeg_default = encode_with_config(&config_default, &pixels, width, height);

    // Encode with both custom quant and zero bias
    let fine_table = [8.0f32; 64]; // Fine quantization
    let aggressive_mul = [0.8f32; 64]; // Aggressive zeroing
    let tables_custom = EncodingTables {
        quant: PerComponent {
            c0: fine_table,
            c1: fine_table,
            c2: fine_table,
        },
        zero_bias_mul: PerComponent {
            c0: aggressive_mul,
            c1: aggressive_mul,
            c2: aggressive_mul,
        },
        zero_bias_offset_dc: [0.5; 3],
        zero_bias_offset_ac: [0.5; 3],
        scaling: ScalingParams::Exact,
    };
    let config_custom =
        EncoderConfig::ycbcr(75.0, ChromaSubsampling::None).tables(Box::new(tables_custom));
    let jpeg_custom = encode_with_config(&config_custom, &pixels, width, height);

    // Outputs should differ
    assert_ne!(
        jpeg_default, jpeg_custom,
        "Combined custom tables should produce different output"
    );

    println!(
        "Default size: {} bytes, Custom combined size: {} bytes",
        jpeg_default.len(),
        jpeg_custom.len()
    );
}
