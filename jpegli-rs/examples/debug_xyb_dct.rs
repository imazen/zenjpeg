use jpegli::consts::{BASE_QUANT_MATRIX_XYB, DCT_BLOCK_SIZE, GLOBAL_SCALE_XYB};
use jpegli::dct::forward_dct_8x8;
use jpegli::quant::{distance_to_scale, generate_quant_table, Quality};
use jpegli::types::ColorSpace;

fn main() {
    // Simulate XYB Y channel for white: scaled Y = 7.3
    // After level shift (-128): -120.7
    let uniform_value = 7.3 - 128.0;

    let mut block = [0.0f32; DCT_BLOCK_SIZE];
    for i in 0..DCT_BLOCK_SIZE {
        block[i] = uniform_value;
    }

    println!("Input block (uniform -120.7):");
    println!("  block[0] = {}", block[0]);

    let dct = forward_dct_8x8(&block);
    println!("\nDCT output:");
    println!("  DC (dct[0]) = {:.4}", dct[0]);
    for i in 1..8 {
        println!("  AC[{}] = {:.4}", i, dct[i]);
    }

    // Check XYB Y quant table at Q90
    let quality = Quality::from_quality(90.0);
    let distance = quality.to_distance();
    println!("\nQ90 distance = {}", distance);

    let y_quant = generate_quant_table(quality, 1, ColorSpace::Xyb, false, false);
    println!("\nXYB Y quant table (Q90, component=1):");
    println!("  DC (quant[0]) = {}", y_quant.values[0]);
    for i in 1..8 {
        println!("  AC[{}] = {}", i, y_quant.values[i]);
    }

    // Also check X and B channels
    let x_quant = generate_quant_table(quality, 0, ColorSpace::Xyb, false, false);
    let b_quant = generate_quant_table(quality, 2, ColorSpace::Xyb, false, false);
    println!("\nXYB X quant (comp=0) DC = {}", x_quant.values[0]);
    println!("XYB B quant (comp=2) DC = {}", b_quant.values[0]);

    // Debug quant calculation
    println!("\n--- Debug quant calculation ---");
    let dist = 1.0_f32;
    let component = 1; // Y channel
    let base_idx = component * 64;
    let base_dc = BASE_QUANT_MATRIX_XYB[base_idx];
    let d2s = distance_to_scale(dist, 0);
    let scale = d2s * GLOBAL_SCALE_XYB;
    let q = (base_dc * scale).round() as u16;

    println!("distance = {}", dist);
    println!("BASE_QUANT_MATRIX_XYB[{}] (Y DC) = {}", base_idx, base_dc);
    println!("distance_to_scale(1.0, 0) = {}", d2s);
    println!("GLOBAL_SCALE_XYB = {}", GLOBAL_SCALE_XYB);
    println!("scale = {} * {} = {}", d2s, GLOBAL_SCALE_XYB, scale);
    println!("q = round({} * {}) = {}", base_dc, scale, q);

    // Quantize DC
    let quantized_dc = (dct[0] / y_quant.values[0] as f32).round() as i16;
    println!("\nQuantized DC = {}", quantized_dc);
}
