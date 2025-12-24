use jpegli::xyb::{srgb_to_scaled_xyb, rgb_buffer_to_scaled_xyb_planes};
use jpegli::dct::forward_dct_8x8;
use jpegli::quant::{self, Quality, quantize_block};
use jpegli::types::ColorSpace;

fn main() {
    println!("=== XYB Encoding Full Trace ===\n");

    // Create 8x8 solid blue block (matches top-left of test image)
    let mut rgb_data = vec![0u8; 8 * 8 * 3];
    for i in 0..64 {
        rgb_data[i * 3] = 0;       // R = 0
        rgb_data[i * 3 + 1] = 0;   // G = 0
        rgb_data[i * 3 + 2] = 128; // B = 128
    }

    // Step 1: Convert to scaled XYB
    let (x_plane, y_plane, b_plane) = rgb_buffer_to_scaled_xyb_planes(&rgb_data, 8, 8);

    println!("1. Scaled XYB values (should be in [0,1]):");
    println!("   X[0] = {:.4}", x_plane[0]);
    println!("   Y[0] = {:.4}", y_plane[0]);
    println!("   B[0] = {:.4}", b_plane[0]);

    // Step 2: Level shift and create DCT input
    let mut x_block = [0.0f32; 64];
    let mut y_block = [0.0f32; 64];
    let mut b_block = [0.0f32; 64];

    for i in 0..64 {
        x_block[i] = x_plane[i] * 255.0 - 128.0;
        y_block[i] = y_plane[i] * 255.0 - 128.0;
        b_block[i] = b_plane[i] * 255.0 - 128.0;
    }

    println!("\n2. Level-shifted DCT input (should be in [-128, 127]):");
    println!("   X_block[0] = {:.2}", x_block[0]);
    println!("   Y_block[0] = {:.2}", y_block[0]);
    println!("   B_block[0] = {:.2}", b_block[0]);

    // Step 3: Forward DCT
    let x_dct = forward_dct_8x8(&x_block);
    let y_dct = forward_dct_8x8(&y_block);
    let b_dct = forward_dct_8x8(&b_block);

    println!("\n3. DCT output (DC coefficient @ index 0):");
    println!("   X_DC = {:.2}", x_dct[0]);
    println!("   Y_DC = {:.2}", y_dct[0]);
    println!("   B_DC = {:.2}", b_dct[0]);

    // For uniform block, DC = input_value * 8 (no AC components)
    println!("   Expected X_DC = {:.2}", x_block[0] * 8.0);
    println!("   Expected Y_DC = {:.2}", y_block[0] * 8.0);
    println!("   Expected B_DC = {:.2}", b_block[0] * 8.0);

    // Step 4: Quantization
    let quality = Quality::from_quality(90.0);
    let x_quant = quant::generate_quant_table(quality, 0, ColorSpace::Rgb, true);
    let y_quant = quant::generate_quant_table(quality, 1, ColorSpace::Rgb, true);
    let b_quant = quant::generate_quant_table(quality, 2, ColorSpace::Rgb, true);

    println!("\n4. Quantization tables (DC @ index 0):");
    println!("   X quant[0] = {}", x_quant.values[0]);
    println!("   Y quant[0] = {}", y_quant.values[0]);
    println!("   B quant[0] = {}", b_quant.values[0]);

    let x_qcoeffs = quantize_block(&x_dct, &x_quant.values);
    let y_qcoeffs = quantize_block(&y_dct, &y_quant.values);
    let b_qcoeffs = quantize_block(&b_dct, &b_quant.values);

    println!("\n5. Quantized DC coefficients:");
    println!("   X_DC_q = {}", x_qcoeffs[0]);
    println!("   Y_DC_q = {}", y_qcoeffs[0]);
    println!("   B_DC_q = {}", b_qcoeffs[0]);

    // What should decode to (approximation - just DC)
    let x_recon = (x_qcoeffs[0] as f32 * x_quant.values[0] as f32) / 8.0;
    let y_recon = (y_qcoeffs[0] as f32 * y_quant.values[0] as f32) / 8.0;
    let b_recon = (b_qcoeffs[0] as f32 * b_quant.values[0] as f32) / 8.0;

    println!("\n6. Reconstructed pixel values (DC only, before level unshift):");
    println!("   X_recon = {:.2}", x_recon);
    println!("   Y_recon = {:.2}", y_recon);
    println!("   B_recon = {:.2}", b_recon);

    println!("\n7. Reconstructed as 0-255 (add 128):");
    println!("   X_final = {:.1}", x_recon + 128.0);
    println!("   Y_final = {:.1}", y_recon + 128.0);
    println!("   B_final = {:.1}", b_recon + 128.0);

    // Original scaled XYB as 0-255
    println!("\n8. Original scaled XYB as 0-255 (for comparison):");
    println!("   X_orig = {:.1}", x_plane[0] * 255.0);
    println!("   Y_orig = {:.1}", y_plane[0] * 255.0);
    println!("   B_orig = {:.1}", b_plane[0] * 255.0);

    println!("\n=== What djpegli should decode (approximately) ===");
    println!("Raw XYB values in JPEG: X≈{:.0}, Y≈{:.0}, B≈{:.0}",
             x_recon + 128.0, y_recon + 128.0, b_recon + 128.0);
    println!("After ICC profile XYB→sRGB: should be close to (0, 0, 128)");
}
