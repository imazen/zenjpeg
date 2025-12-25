use jpegli::quant::{generate_quant_table, Quality};
use jpegli::types::ColorSpace;

fn main() {
    let quality = Quality::from_quality(100.0);
    let distance = quality.to_distance();

    println!("Q100 -> distance = {}", distance);

    // Generate Y table
    let y_table = generate_quant_table(quality, 0, ColorSpace::YCbCr, false);
    let cb_table = generate_quant_table(quality, 1, ColorSpace::YCbCr, false);
    let cr_table = generate_quant_table(quality, 2, ColorSpace::YCbCr, false);

    println!("\nY quant table (first 16 values):");
    for i in 0..16 {
        print!("{:3} ", y_table.values[i]);
    }
    println!("\n");

    println!("Cb quant table (first 16 values):");
    for i in 0..16 {
        print!("{:3} ", cb_table.values[i]);
    }
    println!("\n");

    println!("Cr quant table (first 16 values):");
    for i in 0..16 {
        print!("{:3} ", cr_table.values[i]);
    }
    println!("\n");

    // Sum all values
    let y_sum: u32 = y_table.values.iter().map(|&v| v as u32).sum();
    let cb_sum: u32 = cb_table.values.iter().map(|&v| v as u32).sum();
    let cr_sum: u32 = cr_table.values.iter().map(|&v| v as u32).sum();

    println!("Y sum: {}, Cb sum: {}, Cr sum: {}", y_sum, cb_sum, cr_sum);

    // Print full tables
    println!("\n\n=== Full Y table ===");
    for row in 0..8 {
        for col in 0..8 {
            print!("{:3} ", y_table.values[row * 8 + col]);
        }
        println!();
    }

    println!("\n=== Full Cb table ===");
    for row in 0..8 {
        for col in 0..8 {
            print!("{:3} ", cb_table.values[row * 8 + col]);
        }
        println!();
    }

    println!("\n=== Full Cr table ===");
    for row in 0..8 {
        for col in 0..8 {
            print!("{:3} ", cr_table.values[row * 8 + col]);
        }
        println!();
    }
}
