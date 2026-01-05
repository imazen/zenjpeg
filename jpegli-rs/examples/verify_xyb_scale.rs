//! Verify XYB scaling now matches C++ (0-255 linear RGB range)

use jpegli::xyb::{srgb_to_xyb, srgb_to_scaled_xyb};

fn main() {
    let test_colors: [(u8, u8, u8, &str); 6] = [
        (0, 0, 0, "black"),
        (255, 255, 255, "white"),
        (255, 0, 0, "red"),
        (0, 255, 0, "green"),
        (0, 0, 255, "blue"),
        (128, 128, 128, "gray"),
    ];

    println!("XYB Conversion (now uses C++ jpegli 0-255 linear RGB convention):");
    println!();
    println!("Scaled XYB values (srgb_to_scaled_xyb):");
    println!("{:<10} {:>12} {:>12} {:>12}", "Color", "X", "Y", "B");
    println!("{}", "-".repeat(50));

    for (r, g, b, name) in test_colors {
        let (sx, sy, sb) = srgb_to_scaled_xyb(r, g, b);
        println!("{:<10} {:>12.4} {:>12.4} {:>12.4}", name, sx, sy, sb);
    }

    println!();
    println!("Unscaled XYB values (srgb_to_xyb):");
    println!("{:<10} {:>12} {:>12} {:>12}", "Color", "X", "Y", "B");
    println!("{}", "-".repeat(50));

    for (r, g, b, name) in test_colors {
        let (x, y, b_val) = srgb_to_xyb(r, g, b);
        println!("{:<10} {:>12.6} {:>12.6} {:>12.6}", name, x, y, b_val);
    }

    println!();
    println!("Expected C++ values for white:");
    println!("  Unscaled Y: ~6.19 (matches C++ LinearRGBRowToXYB)");
    println!("  Scaled Y:   ~7.32 (matches C++ ScaleXYBRow)");

    let (_, y_white, _) = srgb_to_xyb(255, 255, 255);
    let (_, sy_white, _) = srgb_to_scaled_xyb(255, 255, 255);
    println!();
    println!("Our values for white:");
    println!("  Unscaled Y: {:.4}", y_white);
    println!("  Scaled Y:   {:.4}", sy_white);
    println!();

    if (y_white - 6.19).abs() < 0.1 && (sy_white - 7.32).abs() < 0.1 {
        println!("SUCCESS: XYB values match C++ jpegli convention!");
    } else {
        println!("WARNING: Values don't match expected C++ range");
    }
}
