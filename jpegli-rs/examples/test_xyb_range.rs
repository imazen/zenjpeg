//! Quick test to check XYB value ranges
use jpegli::xyb::{srgb_to_scaled_xyb, SCALED_XYB_SCALE, SCALED_XYB_OFFSET};

fn main() {
    let mut x_min = f32::MAX;
    let mut x_max = f32::MIN;
    let mut y_min = f32::MAX;
    let mut y_max = f32::MIN;
    let mut b_min = f32::MAX;
    let mut b_max = f32::MIN;

    // Test all possible RGB combinations (sampled)
    for r in (0..=255).step_by(16) {
        for g in (0..=255).step_by(16) {
            for b in (0..=255).step_by(16) {
                let (x, y, b_val) = srgb_to_scaled_xyb(r as u8, g as u8, b as u8);
                x_min = x_min.min(x);
                x_max = x_max.max(x);
                y_min = y_min.min(y);
                y_max = y_max.max(y);
                b_min = b_min.min(b_val);
                b_max = b_max.max(b_val);
            }
        }
    }

    println!("Scaled XYB value ranges:");
    println!("  X: [{:.4}, {:.4}]", x_min, x_max);
    println!("  Y: [{:.4}, {:.4}]", y_min, y_max);
    println!("  B: [{:.4}, {:.4}]", b_min, b_max);
    println!();
    println!("Scale constants: {:?}", SCALED_XYB_SCALE);
    println!("Offset constants: {:?}", SCALED_XYB_OFFSET);
    println!();
    
    // Check specific colors
    let test_colors = [
        (0, 0, 0, "black"),
        (255, 255, 255, "white"),
        (255, 0, 0, "red"),
        (0, 255, 0, "green"),
        (0, 0, 255, "blue"),
        (128, 128, 128, "gray"),
    ];
    
    println!("Specific colors:");
    for (r, g, b, name) in test_colors {
        let (x, y, bv) = srgb_to_scaled_xyb(r, g, b);
        println!("  {}: X={:.4}, Y={:.4}, B={:.4}", name, x, y, bv);
    }
}
