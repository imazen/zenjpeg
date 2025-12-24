//! Test DC coefficient handling: level shift before vs after DCT

fn main() {
    // Create a uniform 8x8 block with value 200
    let value = 200u8;

    println!("=== DC Bias Test ===\n");
    println!("Uniform block value: {}\n", value);

    // Method 1: Rust approach - level shift BEFORE DCT
    let mut block_rust = [0.0f32; 64];
    for i in 0..64 {
        block_rust[i] = value as f32 - 128.0; // Level shift first
    }
    let dct_rust = jpegli::dct::forward_dct_8x8(&block_rust);

    println!("Rust approach (level shift before DCT):");
    println!("  Input to DCT: {} (uniform)", value as f32 - 128.0);
    println!("  DC coefficient: {:.2}", dct_rust[0]);

    // Method 2: C++ approach - level shift AFTER DCT
    let mut block_cpp = [0.0f32; 64];
    for i in 0..64 {
        block_cpp[i] = value as f32; // NO level shift
    }
    let dct_cpp = jpegli::dct::forward_dct_8x8(&block_cpp);
    let dc_cpp_adjusted = dct_cpp[0] - 128.0; // Subtract 128 AFTER DCT

    println!("\nC++ approach (level shift after DCT):");
    println!("  Input to DCT: {} (uniform)", value as f32);
    println!("  Raw DC coefficient: {:.2}", dct_cpp[0]);
    println!("  DC after -128 adjustment: {:.2}", dc_cpp_adjusted);

    // Compare
    println!("\nDifference: {:.2}", (dct_rust[0] - dc_cpp_adjusted).abs());

    // AC coefficients should be identical (level shift only affects DC)
    println!("\nAC coefficients comparison (first 5):");
    for i in 1..6 {
        println!(
            "  AC[{}]: Rust={:.4}, C++={:.4}, diff={:.6}",
            i,
            dct_rust[i],
            dct_cpp[i],
            (dct_rust[i] - dct_cpp[i]).abs()
        );
    }

    // Test with a gradient block
    println!("\n=== Gradient Block Test ===\n");

    let mut grad_rust = [0.0f32; 64];
    let mut grad_cpp = [0.0f32; 64];
    for y in 0..8 {
        for x in 0..8 {
            let val = (y * 8 + x) as f32 * 4.0; // 0 to 252
            grad_rust[y * 8 + x] = val - 128.0;
            grad_cpp[y * 8 + x] = val;
        }
    }

    let dct_grad_rust = jpegli::dct::forward_dct_8x8(&grad_rust);
    let dct_grad_cpp = jpegli::dct::forward_dct_8x8(&grad_cpp);

    println!(
        "DC: Rust={:.2}, C++ raw={:.2}, C++ adjusted={:.2}",
        dct_grad_rust[0],
        dct_grad_cpp[0],
        dct_grad_cpp[0] - 128.0
    );
    println!(
        "Rust DC - C++ adjusted DC: {:.2}",
        dct_grad_rust[0] - (dct_grad_cpp[0] - 128.0)
    );

    println!("\nAC coefficients (first 10):");
    for i in 1..11 {
        let diff = (dct_grad_rust[i] - dct_grad_cpp[i]).abs();
        if diff > 0.001 {
            println!(
                "  AC[{}]: Rust={:.4}, C++={:.4}, DIFF={:.6} !!!",
                i, dct_grad_rust[i], dct_grad_cpp[i], diff
            );
        }
    }
    println!("(All AC coefficients should be identical - level shift only affects DC)");
}
