//! Compare DCT implementations between Rust and C++

fn main() {
    // Test with a real image block - simulating a smooth gradient
    let pixel_block: [u8; 64] = [
        180, 178, 176, 174, 172, 170, 168, 166,
        175, 173, 171, 169, 167, 165, 163, 161,
        170, 168, 166, 164, 162, 160, 158, 156,
        165, 163, 161, 159, 157, 155, 153, 151,
        160, 158, 156, 154, 152, 150, 148, 146,
        155, 153, 151, 149, 147, 145, 143, 141,
        150, 148, 146, 144, 142, 140, 138, 136,
        145, 143, 141, 139, 137, 135, 133, 131,
    ];

    // Test u8 DCT function directly
    let dct_result = jpegli::dct::forward_dct_8x8_u8(&pixel_block);

    println!("=== DCT Test ===\n");
    println!("Input pixel block:");
    for y in 0..8 {
        for x in 0..8 {
            print!("{:4}", pixel_block[y * 8 + x]);
        }
        println!();
    }

    println!("\nDCT coefficients (from u8 input):");
    for y in 0..8 {
        for x in 0..8 {
            print!("{:8.2}", dct_result[y * 8 + x]);
        }
        println!();
    }

    // Test with level-shifted f32 input
    let mut f32_input = [0.0f32; 64];
    for i in 0..64 {
        f32_input[i] = pixel_block[i] as f32 - 128.0;
    }

    let dct_f32 = jpegli::dct::forward_dct_8x8(&f32_input);

    println!("\nDCT coefficients (from f32 level-shifted input):");
    for y in 0..8 {
        for x in 0..8 {
            print!("{:8.2}", dct_f32[y * 8 + x]);
        }
        println!();
    }

    // Check if they match
    let mut max_diff = 0.0f32;
    for i in 0..64 {
        let diff = (dct_result[i] - dct_f32[i]).abs();
        if diff > max_diff { max_diff = diff; }
    }
    println!("\nMax difference between u8 and f32 paths: {:.6}", max_diff);
}
