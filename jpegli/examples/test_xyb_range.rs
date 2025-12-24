use jpegli::xyb::{srgb_to_scaled_xyb, SCALED_XYB_OFFSET, SCALED_XYB_SCALE};

fn main() {
    // Test various sRGB values to see scaled XYB range
    let test_values = [
        (0, 0, 0),       // Black
        (255, 255, 255), // White
        (255, 0, 0),     // Red
        (0, 255, 0),     // Green
        (0, 0, 255),     // Blue
        (128, 128, 128), // Gray
    ];

    println!("sRGB to Scaled XYB conversion ranges:");
    println!("{:>20} -> {:>12} {:>12} {:>12}", "sRGB", "X", "Y", "B");

    let mut min_x = f32::MAX;
    let mut max_x = f32::MIN;
    let mut min_y = f32::MAX;
    let mut max_y = f32::MIN;
    let mut min_b = f32::MAX;
    let mut max_b = f32::MIN;

    for (r, g, b) in test_values {
        let (x, y, b_out) = srgb_to_scaled_xyb(r, g, b);
        println!(
            "({:3}, {:3}, {:3}) -> {:12.4} {:12.4} {:12.4}",
            r, g, b, x, y, b_out
        );

        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
        min_b = min_b.min(b_out);
        max_b = max_b.max(b_out);
    }

    println!("\nRanges:");
    println!("X: [{:.4}, {:.4}]", min_x, max_x);
    println!("Y: [{:.4}, {:.4}]", min_y, max_y);
    println!("B: [{:.4}, {:.4}]", min_b, max_b);

    // Test full range
    println!("\nFull 256 value scan:");
    min_x = f32::MAX;
    max_x = f32::MIN;
    min_y = f32::MAX;
    max_y = f32::MIN;
    min_b = f32::MAX;
    max_b = f32::MIN;

    for r in 0..=255u8 {
        for g in 0..=255u8 {
            for b in 0..=255u8 {
                let (x, y, b_out) = srgb_to_scaled_xyb(r, g, b);
                min_x = min_x.min(x);
                max_x = max_x.max(x);
                min_y = min_y.min(y);
                max_y = max_y.max(y);
                min_b = min_b.min(b_out);
                max_b = max_b.max(b_out);
            }
        }
    }

    println!(
        "X: [{:.4}, {:.4}] (range: {:.4})",
        min_x,
        max_x,
        max_x - min_x
    );
    println!(
        "Y: [{:.4}, {:.4}] (range: {:.4})",
        min_y,
        max_y,
        max_y - min_y
    );
    println!(
        "B: [{:.4}, {:.4}] (range: {:.4})",
        min_b,
        max_b,
        max_b - min_b
    );
}
