//! Compare Rust quantization tables against C++ exact values

use jpegli::quant::{self, Quality};
use jpegli::types::ColorSpace;

fn main() {
    // C++ tables for 4:4:4 at distance=1.0 (Traditional Q90)
    // Extracted from SetQuantMatrices.testdata
    let cpp_y_table: [u16; 64] = [
        2, 3, 5, 5, 6, 6, 7, 7, 3, 4, 5, 5, 6, 6, 7, 7, 5, 5, 5, 6, 6, 7, 7, 7, 5, 5, 6, 6, 7, 8,
        8, 8, 6, 6, 6, 7, 8, 9, 10, 10, 6, 6, 7, 8, 9, 9, 9, 11, 7, 7, 7, 8, 10, 9, 9, 9, 7, 7, 7,
        8, 10, 11, 9, 8,
    ];

    let cpp_cbcr_table: [u16; 64] = [
        5, 11, 16, 19, 19, 30, 32, 51, 11, 15, 16, 24, 29, 29, 47, 37, 16, 16, 19, 32, 34, 42, 179,
        43, 19, 24, 32, 32, 28, 44, 44, 62, 19, 29, 34, 28, 34, 26, 35, 89, 30, 29, 42, 44, 26, 54,
        81, 117, 32, 47, 179, 44, 35, 81, 107, 154, 51, 37, 43, 62, 89, 117, 154, 195,
    ];

    // Generate Rust tables - component 0=Y, 1=Cb, 2=Cr
    // This example compares 4:4:4 mode, so is_420 = false
    let quality = Quality::Traditional(90.0);
    let rust_y = quant::generate_quant_table(quality, 0, ColorSpace::YCbCr, false, false);
    let rust_cb = quant::generate_quant_table(quality, 1, ColorSpace::YCbCr, false, false);

    println!("=== Y Table Comparison (Q90, 4:4:4) ===");
    print_comparison("Y", &cpp_y_table, &rust_y.values);

    println!("\n=== CbCr Table Comparison (Q90, 4:4:4) ===");
    print_comparison("CbCr", &cpp_cbcr_table, &rust_cb.values);
}

fn print_comparison(name: &str, cpp: &[u16; 64], rust: &[u16; 64]) {
    println!("C++ first 16:  {:?}", &cpp[..16]);
    println!("Rust first 16: {:?}", &rust[..16]);

    let mut diffs = Vec::new();
    for i in 0..64 {
        if rust[i] != cpp[i] {
            diffs.push((i, cpp[i], rust[i]));
        }
    }

    if diffs.is_empty() {
        println!("✓ {} table matches exactly!", name);
    } else {
        println!("✗ {} differences found:", diffs.len());
        for (idx, cpp_val, rust_val) in &diffs {
            let row = idx / 8;
            let col = idx % 8;
            println!(
                "  [{}] ({},{}) C++={} Rust={} (diff={})",
                idx,
                row,
                col,
                cpp_val,
                rust_val,
                *rust_val as i32 - *cpp_val as i32
            );
        }
    }
}
