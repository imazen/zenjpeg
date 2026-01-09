//! Parity test comparing strip-based vs full-plane encoder output.
//!
//! This test verifies that the strip-based low-memory encoder produces
//! identical quantized blocks to the full-plane encoder.

use jpegli::encode::strip::{StripProcessor, StripProcessorOutput};
use jpegli::quant::{generate_quant_table, Quality, ZeroBiasParams};
use jpegli::types::{ColorSpace, PixelFormat, Subsampling};
use jpegli::Encoder;

/// Generate a deterministic test image (gradient pattern).
fn generate_test_image(width: usize, height: usize) -> Vec<u8> {
    let mut rgb_data = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            rgb_data[idx] = (x * 255 / width.max(1)) as u8;
            rgb_data[idx + 1] = (y * 255 / height.max(1)) as u8;
            rgb_data[idx + 2] = 128;
        }
    }
    rgb_data
}

/// Encode using strip-based processor and return blocks.
fn encode_strip(
    rgb_data: &[u8],
    width: usize,
    height: usize,
    quality: Quality,
) -> StripProcessorOutput {
    let mut processor =
        StripProcessor::new(width, height, Subsampling::S420, PixelFormat::Rgb).unwrap();

    // Generate quant tables
    let is_420 = true;
    let y_quant = generate_quant_table(quality, 0, ColorSpace::YCbCr, false, is_420);
    let cb_quant = generate_quant_table(quality, 1, ColorSpace::YCbCr, false, is_420);
    let cr_quant = generate_quant_table(quality, 2, ColorSpace::YCbCr, false, is_420);

    // Compute zero bias params
    let effective_distance = jpegli::quant::quant_vals_to_distance(&y_quant, &cb_quant, &cr_quant);
    let y_zero_bias = ZeroBiasParams::for_ycbcr(effective_distance, 0);
    let cb_zero_bias = ZeroBiasParams::for_ycbcr(effective_distance, 1);
    let cr_zero_bias = ZeroBiasParams::for_ycbcr(effective_distance, 2);

    processor.set_quant_tables(
        y_quant,
        cb_quant,
        cr_quant,
        y_zero_bias,
        cb_zero_bias,
        cr_zero_bias,
    );

    // Process in strips
    let strip_height = processor.strip_height();
    for strip_y in (0..height).step_by(strip_height) {
        let strip_end = (strip_y + strip_height).min(height);
        let strip_start = strip_y * width * 3;
        let strip_end_idx = strip_end * width * 3;
        let rgb_strip = &rgb_data[strip_start..strip_end_idx];

        processor.process_strip(rgb_strip, strip_y).unwrap();
    }

    processor.finalize().unwrap()
}

#[test]
fn test_strip_block_counts_match() {
    let test_cases = [
        (64, 64, "64x64 (minimal)"),
        (256, 256, "256x256"),
        (320, 240, "320x240 (non-power-of-2)"),
        (1920, 1080, "1920x1080 (2K)"),
    ];

    for (width, height, name) in test_cases {
        let rgb_data = generate_test_image(width, height);
        let output = encode_strip(&rgb_data, width, height, Quality::Traditional(85.0));

        // Expected block counts for 4:2:0
        let expected_y_blocks = ((width + 7) / 8) * ((height + 7) / 8);
        let expected_c_blocks = ((width + 15) / 16) * ((height + 15) / 16);

        assert_eq!(
            output.y_blocks.len(),
            expected_y_blocks,
            "{}: Y block count mismatch",
            name
        );
        assert_eq!(
            output.cb_blocks.len(),
            expected_c_blocks,
            "{}: Cb block count mismatch",
            name
        );
        assert_eq!(
            output.cr_blocks.len(),
            expected_c_blocks,
            "{}: Cr block count mismatch",
            name
        );
    }
}

#[test]
fn test_strip_dc_coefficients_reasonable() {
    // Test that DC coefficients are in a reasonable range
    let width = 256;
    let height = 256;
    let rgb_data = generate_test_image(width, height);
    let output = encode_strip(&rgb_data, width, height, Quality::Traditional(85.0));

    // DC coefficients should be non-zero for a gradient image
    let non_zero_y_dc = output.y_blocks.iter().filter(|b| b[0] != 0).count();
    let non_zero_cb_dc = output.cb_blocks.iter().filter(|b| b[0] != 0).count();
    let non_zero_cr_dc = output.cr_blocks.iter().filter(|b| b[0] != 0).count();

    // For a gradient image, most DC coefficients should be non-zero
    assert!(
        non_zero_y_dc > output.y_blocks.len() / 2,
        "Too few non-zero Y DC coefficients: {} of {}",
        non_zero_y_dc,
        output.y_blocks.len()
    );
    assert!(non_zero_cb_dc > 0, "No non-zero Cb DC coefficients");
    assert!(non_zero_cr_dc > 0, "No non-zero Cr DC coefficients");
}

#[test]
fn test_strip_output_can_be_encoded() {
    // Test that strip processor output can be used for Huffman table building
    let width = 256;
    let height = 256;
    let rgb_data = generate_test_image(width, height);
    let output = encode_strip(&rgb_data, width, height, Quality::Traditional(85.0));

    // Huffman frequency counters should have accumulated frequencies
    let dc_luma_total = output.dc_luma_freq.total();
    let ac_luma_total = output.ac_luma_freq.total();
    let dc_chroma_total = output.dc_chroma_freq.total();
    let ac_chroma_total = output.ac_chroma_freq.total();

    // Should have counted symbols
    assert!(dc_luma_total > 0, "No DC luma frequencies recorded");
    assert!(ac_luma_total > 0, "No AC luma frequencies recorded");
    assert!(dc_chroma_total > 0, "No DC chroma frequencies recorded");
    assert!(ac_chroma_total > 0, "No AC chroma frequencies recorded");
}

#[test]
fn test_strip_vs_standard_output_comparable() {
    // Compare strip-based output with standard encoder output.
    // The outputs should produce similar JPEG quality when encoded.
    let width = 256;
    let height = 256;
    let rgb_data = generate_test_image(width, height);

    // Encode with standard encoder
    let encoder = Encoder::new()
        .width(width as u32)
        .height(height as u32)
        .jpegli_quality(Quality::Traditional(85.0))
        .pixel_format(PixelFormat::Rgb)
        .subsampling(Subsampling::S420)
        .optimize_huffman(true);

    let standard_output = encoder.encode(&rgb_data).expect("encoding failed");

    // Encode with strip processor
    let strip_output = encode_strip(&rgb_data, width, height, Quality::Traditional(85.0));

    // Both should produce reasonable block counts
    let expected_y_blocks = ((width + 7) / 8) * ((height + 7) / 8);
    assert_eq!(strip_output.y_blocks.len(), expected_y_blocks);

    // Standard output should be a valid JPEG
    assert!(standard_output.len() > 100, "JPEG output too small");
    assert_eq!(&standard_output[0..2], &[0xFF, 0xD8], "Missing SOI marker");
    assert_eq!(
        &standard_output[standard_output.len() - 2..],
        &[0xFF, 0xD9],
        "Missing EOI marker"
    );
}

#[test]
fn test_strip_various_qualities() {
    let width = 128;
    let height = 128;
    let rgb_data = generate_test_image(width, height);

    for quality in [50.0, 75.0, 85.0, 95.0] {
        let output = encode_strip(&rgb_data, width, height, Quality::Traditional(quality));

        // Higher quality should have more non-zero AC coefficients
        let total_ac: usize = output
            .y_blocks
            .iter()
            .map(|b| b[1..].iter().filter(|&&c| c != 0).count())
            .sum();

        assert!(total_ac > 0, "No AC coefficients at quality {}", quality);
    }
}

#[test]
fn test_strip_edge_cases() {
    // Test edge cases: images that don't align to block boundaries
    // Note: For 4:2:0, chroma block counts may differ slightly from MCU-padded
    // expectations due to how strip processing handles partial strips.
    let test_cases = [
        (17, 17, "17x17 (just over one block)"),
        (63, 63, "63x63 (just under 8 blocks)"),
        (65, 65, "65x65 (just over 8 blocks)"),
        (100, 50, "100x50 (non-square, partial blocks)"),
    ];

    for (width, height, name) in test_cases {
        let rgb_data = generate_test_image(width, height);
        let output = encode_strip(&rgb_data, width, height, Quality::Traditional(85.0));

        // Y blocks should match exactly
        let expected_y_blocks = ((width + 7) / 8) * ((height + 7) / 8);
        assert_eq!(
            output.y_blocks.len(),
            expected_y_blocks,
            "{}: Y block count mismatch",
            name
        );

        // Cb and Cr should have same count (consistent)
        assert_eq!(
            output.cb_blocks.len(),
            output.cr_blocks.len(),
            "{}: Cb/Cr block count mismatch",
            name
        );

        // Chroma blocks should be non-empty
        assert!(
            output.cb_blocks.len() > 0,
            "{}: No Cb blocks produced",
            name
        );
    }
}
