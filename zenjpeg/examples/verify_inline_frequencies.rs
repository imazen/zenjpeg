//! Verify that inline frequency counting produces identical Huffman tables
//! to the batch approach.
//!
//! Run with: cargo run --release -p zenjpeg --example verify_inline_frequencies

use zenjpeg::encode::config::ComputedConfig;
use zenjpeg::encode::strip::StripProcessor;
use zenjpeg::quant::{generate_quant_table_ex, ZeroBiasParams};
use zenjpeg::types::{ColorSpace, JpegMode, PixelFormat, Quality, Subsampling};

fn main() {
    println!("Testing inline frequency counting vs batch...\n");

    // Test 4:2:0 subsampling (most complex case)
    test_subsampling("4:2:0", Subsampling::S420, true);

    // Test 4:4:4 subsampling
    test_subsampling("4:4:4", Subsampling::S444, false);

    println!("\nAll tests passed!");
}

fn test_subsampling(name: &str, subsampling: Subsampling, is_420: bool) {
    println!("Testing {} subsampling...", name);

    let width = 64usize;
    let height = 64usize;

    // Generate test image with gradients
    let mut pixels = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let r = ((x * 255) / width.max(1)) as u8;
            let g = ((y * 255) / height.max(1)) as u8;
            let b = (((x + y) * 127) / (width + height).max(1)) as u8;
            pixels.push(r);
            pixels.push(g);
            pixels.push(b);
        }
    }

    // Build processor and encode
    let mut processor = StripProcessor::new(width, height, subsampling, PixelFormat::Rgb).unwrap();

    // Set quant tables
    let quality = Quality::ApproxJpegli(85.0);
    let y_quant = generate_quant_table_ex(quality, 0, ColorSpace::YCbCr, false, is_420, true);
    let cb_quant = generate_quant_table_ex(quality, 1, ColorSpace::YCbCr, false, is_420, true);
    let cr_quant = generate_quant_table_ex(quality, 2, ColorSpace::YCbCr, false, is_420, true);
    let distance = quality.to_distance();
    processor
        .set_quant_tables(
            y_quant.clone(),
            cb_quant.clone(),
            cr_quant.clone(),
            ZeroBiasParams::for_ycbcr(distance, 0),
            ZeroBiasParams::for_ycbcr(distance, 1),
            ZeroBiasParams::for_ycbcr(distance, 2),
        )
        .unwrap();

    // Process all rows
    let strip_height = processor.strip_height();
    let row_bytes = width * 3;
    for y in (0..height).step_by(strip_height) {
        let strip_rows = strip_height.min(height - y);
        let strip_start = y * row_bytes;
        let strip_end = (y + strip_rows) * row_bytes;
        processor
            .process_strip(&pixels[strip_start..strip_end], y)
            .unwrap();
    }

    // Get inline frequencies
    let (dc_luma_inline, ac_luma_inline, dc_chroma_inline, ac_chroma_inline) =
        processor.frequency_counters();

    // Clone frequencies before finalize (which consumes processor)
    let dc_luma_inline = dc_luma_inline.clone();
    let ac_luma_inline = ac_luma_inline.clone();
    let dc_chroma_inline = dc_chroma_inline.clone();
    let ac_chroma_inline = ac_chroma_inline.clone();

    // Finalize and get blocks
    let output = processor.finalize().unwrap();

    println!("  Y blocks: {}", output.y_blocks.len());
    println!("  Cb blocks: {}", output.cb_blocks.len());
    println!("  Cr blocks: {}", output.cr_blocks.len());

    // Create config for batch frequency counting
    let config = ComputedConfig {
        width: width as u32,
        height: height as u32,
        pixel_format: PixelFormat::Rgb,
        quality,
        subsampling,
        mode: JpegMode::Baseline,
        optimize_huffman: true,
        chroma_downsampling: Default::default(),
        restart_interval: 0,
        use_xyb: false,
        #[cfg(feature = "parallel")]
        parallel: false,
        #[cfg(feature = "experimental-hybrid-trellis")]
        hybrid_config: Default::default(),
        #[cfg(feature = "experimental-hybrid-trellis")]
        custom_aq_map: None,
        #[cfg(feature = "experimental-hybrid-trellis")]
        trellis: None,
        encoding_tables: None,
        edge_padding: Default::default(),
        original_width: None,
        original_height: None,
        allow_16bit_quant_tables: true,
        separate_chroma_tables: true,
    };

    // Build optimized tables (this does batch frequency counting internally)
    let tables = config
        .build_optimized_tables(&output.y_blocks, &output.cb_blocks, &output.cr_blocks, true)
        .unwrap();

    // Generate tables from inline frequencies
    let huffman_method = zenjpeg::types::HuffmanMethod::JpegliCreateTree;
    let dc_luma_table = dc_luma_inline
        .generate_table_with_method(huffman_method)
        .unwrap();
    let ac_luma_table = ac_luma_inline
        .generate_table_with_method(huffman_method)
        .unwrap();
    let dc_chroma_table = dc_chroma_inline
        .generate_table_with_method(huffman_method)
        .unwrap();
    let ac_chroma_table = ac_chroma_inline
        .generate_table_with_method(huffman_method)
        .unwrap();

    // Compare bits arrays (the DHT representation)
    let mut all_match = true;

    if dc_luma_table.bits != tables.dc_luma.bits {
        println!("  MISMATCH: DC luma bits");
        println!("    Inline: {:?}", dc_luma_table.bits);
        println!("    Batch:  {:?}", tables.dc_luma.bits);
        all_match = false;
    }

    if ac_luma_table.bits != tables.ac_luma.bits {
        println!("  MISMATCH: AC luma bits");
        all_match = false;
    }

    if dc_chroma_table.bits != tables.dc_chroma.bits {
        println!("  MISMATCH: DC chroma bits");
        all_match = false;
    }

    if ac_chroma_table.bits != tables.ac_chroma.bits {
        println!("  MISMATCH: AC chroma bits");
        all_match = false;
    }

    if dc_luma_table.values != tables.dc_luma.values {
        println!("  MISMATCH: DC luma values");
        all_match = false;
    }

    if ac_luma_table.values != tables.ac_luma.values {
        println!("  MISMATCH: AC luma values");
        all_match = false;
    }

    if dc_chroma_table.values != tables.dc_chroma.values {
        println!("  MISMATCH: DC chroma values");
        all_match = false;
    }

    if ac_chroma_table.values != tables.ac_chroma.values {
        println!("  MISMATCH: AC chroma values");
        all_match = false;
    }

    if all_match {
        println!("  OK - All Huffman tables match!");
    } else {
        panic!("{} subsampling test FAILED!", name);
    }
}
