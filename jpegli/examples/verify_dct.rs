use jpegli::dct::forward_dct_8x8;

fn main() {
    // Test with uniform block of value 128 (will be level-shifted to 0)
    let mut block = [0.0f32; 64];
    for i in 0..64 {
        block[i] = 0.0; // Level-shifted 128
    }
    let dct = forward_dct_8x8(&block);
    println!("Uniform block (all 0 after level shift):");
    println!("  DC = {}", dct[0]);
    println!("  AC[1] = {}", dct[1]);
    
    // Test with uniform block of value 255 (level-shifted to 127)
    for i in 0..64 {
        block[i] = 127.0;
    }
    let dct = forward_dct_8x8(&block);
    println!("\nUniform block (all 127 after level shift):");
    println!("  DC = {} (expected: 127*8 = 1016)", dct[0]);
    
    // Test with a gradient block
    for y in 0..8 {
        for x in 0..8 {
            block[y * 8 + x] = ((y + x) as f32 * 10.0) - 35.0; // Range -35 to 105
        }
    }
    let dct = forward_dct_8x8(&block);
    println!("\nGradient block:");
    println!("  DC = {}", dct[0]);
    println!("  First 8 AC: {:?}", &dct[1..9]);
    
    // What threshold comparison looks like
    println!("\n--- Threshold Comparison ---");
    println!("For coefficient with dct=0.5, quant=1:");
    println!("  Rust qval = 0.5 / 1 = 0.5");
    println!("  Rust qval*8 = 4.0");
    println!("  threshold ≈ 0.59");
    println!("  With *8 fix: |4.0| >= 0.59? true -> NOT zeroed");
    println!("  Without fix: |0.5| >= 0.59? false -> zeroed");
    
    println!("\nFor coefficient with dct=0.07, quant=1:");
    println!("  Rust qval = 0.07 / 1 = 0.07");
    println!("  Rust qval*8 = 0.56");
    println!("  threshold ≈ 0.59");
    println!("  With *8 fix: |0.56| >= 0.59? false -> zeroed");
    println!("  Without fix: |0.07| >= 0.59? false -> zeroed");
}
