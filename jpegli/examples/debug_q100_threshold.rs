//! Debug Q100 threshold and coefficient values

use jpegli::quant::{generate_quant_table, Quality, ZeroBiasParams, quant_vals_to_distance};
use jpegli::types::ColorSpace;
use jpegli::dct::forward_dct_8x8;
use jpegli::consts::DCT_BLOCK_SIZE;

fn main() {
    let quality = Quality::from_quality(100.0);
    let input_distance = quality.to_distance();
    
    println!("=== Q100 Threshold Analysis ===\n");
    println!("Input distance from Q100: {:.6}", input_distance);
    
    // Generate quant tables
    let y_quant = generate_quant_table(quality, 0, ColorSpace::YCbCr, false);
    let cb_quant = generate_quant_table(quality, 1, ColorSpace::YCbCr, false);
    let cr_quant = generate_quant_table(quality, 2, ColorSpace::YCbCr, false);
    
    println!("\nY quant table (first 16 values):");
    for i in 0..16 {
        print!("{} ", y_quant.values[i]);
    }
    println!();
    
    // Compute effective distance
    let effective_distance = quant_vals_to_distance(&y_quant, &cb_quant, &cr_quant);
    println!("\nEffective distance from quant tables: {:.6}", effective_distance);
    
    // Get zero-bias parameters
    let y_zero_bias = ZeroBiasParams::for_ycbcr(effective_distance, 0);
    
    println!("\nY zero-bias parameters:");
    println!("  DC offset: {:.6}", y_zero_bias.offset[0]);
    println!("  DC mul: {:.6}", y_zero_bias.mul[0]);
    println!("  AC[1] offset: {:.6}", y_zero_bias.offset[1]);
    println!("  AC[1] mul: {:.6}", y_zero_bias.mul[1]);
    
    // Test with some typical aq_strength values
    println!("\nThreshold for AC[1] at different aq_strength values:");
    for aq in [0.0f32, 0.05, 0.08, 0.1, 0.2] {
        let threshold = y_zero_bias.offset[1] + y_zero_bias.mul[1] * aq;
        println!("  aq_strength={:.2}: threshold = {:.4}", aq, threshold);
    }
    
    // Create a test block with typical photo values
    let test_block: [f32; 64] = {
        let mut block = [0.0f32; 64];
        // Simulate level-shifted pixels (128 gives 0 after level shift)
        // A typical photo block might have gradual variation
        for y in 0..8 {
            for x in 0..8 {
                // Values around 0 (level-shifted from around 128)
                let val = ((y as f32 - 3.5) * 2.0 + (x as f32 - 3.5) * 1.5) as f32;
                block[y * 8 + x] = val;
            }
        }
        block
    };
    
    // Apply DCT
    let dct_coeffs = forward_dct_8x8(&test_block);
    
    println!("\nTest block DCT coefficients (first 16):");
    for i in 0..16 {
        print!("{:8.2} ", dct_coeffs[i]);
        if (i + 1) % 8 == 0 { println!(); }
    }
    
    // Compute qval for each coefficient at Q100 (quant=1)
    println!("\nqval (=dct/quant) for first 16 AC coefficients at Q100:");
    for i in 1..17 {
        let q = y_quant.values[i] as f32;
        let qval = dct_coeffs[i] / q;
        let threshold = y_zero_bias.offset[i] + y_zero_bias.mul[i] * 0.08; // typical aq
        let zeroed = qval.abs() < threshold;
        println!("  [{}] dct={:8.4}, q={}, qval={:8.4}, thresh={:.4}, zeroed={}",
                 i, dct_coeffs[i], y_quant.values[i], qval, threshold, zeroed);
    }
}
