//! Verify DCT scaling factor matches C++

use jpegli::dct;

fn main() {
    // Test: DC coefficient of all-ones block
    // For 8x8 block of all 1.0 values:
    // Standard DCT-II DC = sum of all values = 64
    // With JPEG normalization (orthonormal): DC = 64 / sqrt(64) = 8
    // With jpegli 1/64 scaling: DC = 64 / 64 = 1

    let ones_block = [1.0f32; 64];
    let dct_result = dct::forward_dct_blocks(&[ones_block])[0];

    println!("All-ones block:");
    println!("  DC coefficient = {:.6}", dct_result[0]);
    println!("  Expected with 1/8 scale: 8.0");
    println!("  Expected with 1/64 scale: 1.0");

    // What C++ produces for same input
    // C++ applies 1/8 twice (in each DCT1D), so total 1/64
    // DC = 64 * (1/64) = 1.0

    println!("\nC++ applies 1/8 scaling in EACH DCT1D pass (rows and columns)");
    println!("Total C++ scaling: 1/8 * 1/8 = 1/64");
    println!("Current Rust scaling: 1/8 (applied once at end)");

    if (dct_result[0] - 8.0).abs() < 0.001 {
        println!("\n❌ BUG CONFIRMED: Rust uses 1/8, should use 1/64!");
        println!("   DC=8.0 means Rust is 8x off from C++");
    } else if (dct_result[0] - 1.0).abs() < 0.001 {
        println!("\n✓ Scaling matches C++ (1/64)");
    }

    // Test AC coefficients too
    let mut gradient = [0.0f32; 64];
    for i in 0..64 {
        gradient[i] = (i % 8) as f32;  // 0,1,2,3,4,5,6,7 repeated
    }
    let gradient_dct = dct::forward_dct_blocks(&[gradient])[0];

    println!("\nGradient block (0-7 repeated):");
    println!("  DC = {:.4}", gradient_dct[0]);
    println!("  AC[1] = {:.4}", gradient_dct[1]);

    // With correct 1/64 scaling:
    // DC = sum/64 = (0+1+2+3+4+5+6+7)*8/64 = 28*8/64 = 3.5
    // With wrong 1/8 scaling: DC = 28*8/8 = 28

    if (gradient_dct[0] - 28.0).abs() < 0.1 {
        println!("  ❌ DC=28 confirms 1/8 scaling (should be 3.5 with 1/64)");
    } else if (gradient_dct[0] - 3.5).abs() < 0.1 {
        println!("  ✓ DC=3.5 confirms correct 1/64 scaling");
    }
}
