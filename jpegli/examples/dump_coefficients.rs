//! Dump exact quantized coefficients for a test block

use jpegli::color::rgb_to_ycbcr_f32;
use jpegli::dct;
use jpegli::quant::{self, Quality};
use jpegli::types::ColorSpace;

fn main() {
    // Create a simple 8x8 test block with known RGB values
    // This is the top-left block of a gradient image
    let mut rgb = Vec::new();
    for y in 0..8 {
        for x in 0..8 {
            rgb.push((x * 32) as u8); // R: 0-224
            rgb.push((y * 32) as u8); // G: 0-224
            rgb.push(128u8); // B: constant
        }
    }

    // Convert to YCbCr (using f32 precision)
    let mut y_vals = [0.0f32; 64];
    let mut cb_vals = [0.0f32; 64];
    let mut cr_vals = [0.0f32; 64];

    for i in 0..64 {
        let (y, cb, cr) = rgb_to_ycbcr_f32(
            rgb[i * 3] as f32,
            rgb[i * 3 + 1] as f32,
            rgb[i * 3 + 2] as f32,
        );
        y_vals[i] = y;
        cb_vals[i] = cb;
        cr_vals[i] = cr;
    }

    println!("=== RGB to YCbCr Conversion ===");
    println!(
        "Y values (first row): {:?}",
        y_vals[..8]
            .iter()
            .map(|x| format!("{:.2}", x))
            .collect::<Vec<_>>()
    );
    println!(
        "Cb values (first row): {:?}",
        cb_vals[..8]
            .iter()
            .map(|x| format!("{:.2}", x))
            .collect::<Vec<_>>()
    );

    // Level shift (subtract 128)
    let mut y_shifted = [0.0f32; 64];
    let mut cb_shifted = [0.0f32; 64];
    let mut cr_shifted = [0.0f32; 64];

    for i in 0..64 {
        y_shifted[i] = y_vals[i] - 128.0;
        cb_shifted[i] = cb_vals[i] - 128.0;
        cr_shifted[i] = cr_vals[i] - 128.0;
    }

    println!("\n=== Level Shifted (Y - 128) ===");
    println!(
        "Y shifted (first row): {:?}",
        y_shifted[..8]
            .iter()
            .map(|x| format!("{:.2}", x))
            .collect::<Vec<_>>()
    );

    // Forward DCT
    let y_dct = dct::forward_dct_blocks(&[y_shifted])[0];
    let cb_dct = dct::forward_dct_blocks(&[cb_shifted])[0];
    let cr_dct = dct::forward_dct_blocks(&[cr_shifted])[0];

    println!("\n=== DCT Coefficients ===");
    println!(
        "Y DCT (first row): {:?}",
        y_dct[..8]
            .iter()
            .map(|x| format!("{:.2}", x))
            .collect::<Vec<_>>()
    );
    println!(
        "Cb DCT (first row): {:?}",
        cb_dct[..8]
            .iter()
            .map(|x| format!("{:.2}", x))
            .collect::<Vec<_>>()
    );

    // Get quantization tables
    let quality = Quality::Traditional(90.0);
    let y_qtable = quant::generate_quant_table(quality, 0, ColorSpace::YCbCr, false);
    let cb_qtable = quant::generate_quant_table(quality, 1, ColorSpace::YCbCr, false);

    // Quantize
    let mut y_quant = [0i16; 64];
    let mut cb_quant = [0i16; 64];
    let mut cr_quant = [0i16; 64];

    for i in 0..64 {
        y_quant[i] = (y_dct[i] / y_qtable.values[i] as f32).round() as i16;
        cb_quant[i] = (cb_dct[i] / cb_qtable.values[i] as f32).round() as i16;
        cr_quant[i] = (cr_dct[i] / cb_qtable.values[i] as f32).round() as i16;
    }

    println!("\n=== Quantized Coefficients ===");
    println!("Y quantized (zigzag order, first 16):");
    let y_zz = zigzag(&y_quant);
    for i in 0..16 {
        print!("{:4} ", y_zz[i]);
    }
    println!();

    println!("Cb quantized (zigzag order, first 16):");
    let cb_zz = zigzag(&cb_quant);
    for i in 0..16 {
        print!("{:4} ", cb_zz[i]);
    }
    println!();

    // Count non-zero coefficients
    let y_nz = y_quant.iter().filter(|&&x| x != 0).count();
    let cb_nz = cb_quant.iter().filter(|&&x| x != 0).count();
    let cr_nz = cr_quant.iter().filter(|&&x| x != 0).count();
    println!("\nNon-zero counts: Y={}, Cb={}, Cr={}", y_nz, cb_nz, cr_nz);

    // Show the DC coefficient which is most important
    println!("\n=== DC Coefficients (most impactful) ===");
    println!("Y: raw={:.4}, quant={}", y_dct[0], y_quant[0]);
    println!("Cb: raw={:.4}, quant={}", cb_dct[0], cb_quant[0]);
    println!("Cr: raw={:.4}, quant={}", cr_dct[0], cr_quant[0]);
}

fn zigzag(block: &[i16; 64]) -> [i16; 64] {
    const ZIGZAG: [usize; 64] = [
        0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27,
        20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
    ];
    let mut result = [0i16; 64];
    for i in 0..64 {
        result[i] = block[ZIGZAG[i]];
    }
    result
}
