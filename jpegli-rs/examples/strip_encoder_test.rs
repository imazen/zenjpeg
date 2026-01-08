//! Test the strip-based encoder for memory efficiency.
//!
//! This example demonstrates the strip-based encoding approach and
//! compares memory usage with the standard encoder.
//!
//! Run with: cargo run --release --example strip_encoder_test

use jpegli::encode::strip::{StripProcessor, StripProcessorOutput};
use jpegli::quant::{generate_quant_table, Quality, ZeroBiasParams};
use jpegli::types::{ColorSpace, PixelFormat, Subsampling};

fn main() {
    println!("=== Strip-Based Encoder Test ===\n");

    // Test with a small image first
    let width = 256usize;
    let height = 256usize;
    let quality = Quality::Traditional(85.0);

    println!("Image: {}×{}", width, height);
    println!("Quality: 85");
    println!();

    // Generate test image (gradient)
    let mut rgb_data = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            rgb_data[idx] = (x * 255 / width) as u8; // R: horizontal gradient
            rgb_data[idx + 1] = (y * 255 / height) as u8; // G: vertical gradient
            rgb_data[idx + 2] = 128; // B: constant
        }
    }

    // Create strip processor
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
    println!("Processing with {} row strips...", strip_height);

    let mut total_blocks = 0;
    for strip_y in (0..height).step_by(strip_height) {
        let strip_end = (strip_y + strip_height).min(height);
        let actual_height = strip_end - strip_y;
        let strip_start = strip_y * width * 3;
        let strip_end_idx = strip_end * width * 3;
        let rgb_strip = &rgb_data[strip_start..strip_end_idx];

        let blocks = processor.process_strip(rgb_strip, strip_y).unwrap();
        total_blocks += blocks;
        println!(
            "  Strip y={}-{}: {} Y blocks added",
            strip_y, strip_end, blocks
        );
    }

    // Finalize
    let output: StripProcessorOutput = processor.finalize();

    println!();
    println!("=== Results ===");
    println!("Total Y blocks: {}", output.y_blocks.len());
    println!("Total Cb blocks: {}", output.cb_blocks.len());
    println!("Total Cr blocks: {}", output.cr_blocks.len());
    println!("AQ strengths: {} values", output.aq_strengths.len());

    // Compute expected block counts
    let expected_y_blocks = ((width + 7) / 8) * ((height + 7) / 8);
    let expected_c_blocks = ((width + 15) / 16) * ((height + 15) / 16);
    println!();
    println!("Expected Y blocks: {}", expected_y_blocks);
    println!("Expected Cb/Cr blocks (4:2:0): {}", expected_c_blocks);

    // Check if counts match
    let y_ok = output.y_blocks.len() == expected_y_blocks;
    let cb_ok = output.cb_blocks.len() == expected_c_blocks;
    let cr_ok = output.cr_blocks.len() == expected_c_blocks;

    println!();
    if y_ok && cb_ok && cr_ok {
        println!("✓ Block counts match expected values!");
    } else {
        println!("✗ Block count mismatch:");
        if !y_ok {
            println!(
                "  Y: got {}, expected {}",
                output.y_blocks.len(),
                expected_y_blocks
            );
        }
        if !cb_ok {
            println!(
                "  Cb: got {}, expected {}",
                output.cb_blocks.len(),
                expected_c_blocks
            );
        }
        if !cr_ok {
            println!(
                "  Cr: got {}, expected {}",
                output.cr_blocks.len(),
                expected_c_blocks
            );
        }
    }

    // Print memory estimates
    println!();
    println!("=== Memory Estimates ===");
    let strip_buffer_size = width * strip_height * 4 * 3; // 3 planes, f32
    let block_storage =
        (output.y_blocks.len() + output.cb_blocks.len() + output.cr_blocks.len()) * 64 * 2; // i16
    let aq_storage = output.aq_strengths.len() * 4;

    println!("Strip buffers (reused): {} KB", strip_buffer_size / 1024);
    println!("Block storage: {} KB", block_storage / 1024);
    println!("AQ storage: {} KB", aq_storage / 1024);
    println!(
        "Total (excluding input): {} KB",
        (strip_buffer_size + block_storage + aq_storage) / 1024
    );
}
