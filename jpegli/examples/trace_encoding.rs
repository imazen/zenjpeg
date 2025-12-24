//! Trace encoding pipeline step by step to find DCT/quant differences

use jpegli::dct;
use jpegli::quant::{self, Quality};
use jpegli::types::ColorSpace;

fn main() {
    // Simple 8x8 test block with known values
    let mut rgb = [0u8; 64 * 3];
    for i in 0..64 {
        let x = i % 8;
        let y = i / 8;
        rgb[i * 3] = (x * 32) as u8;      // R: 0,32,64...224
        rgb[i * 3 + 1] = (y * 32) as u8;  // G: 0,32,64...224
        rgb[i * 3 + 2] = 128;             // B: constant
    }

    println!("=== Step 1: RGB to YCbCr ===");
    let mut y_block = [0f32; 64];
    let mut cb_block = [0f32; 64];
    let mut cr_block = [0f32; 64];

    for i in 0..64 {
        let r = rgb[i * 3] as f32;
        let g = rgb[i * 3 + 1] as f32;
        let b = rgb[i * 3 + 2] as f32;

        // JFIF YCbCr conversion (BT.601)
        y_block[i] = 0.299 * r + 0.587 * g + 0.114 * b;
        cb_block[i] = 128.0 + (-0.168736 * r - 0.331264 * g + 0.5 * b);
        cr_block[i] = 128.0 + (0.5 * r - 0.418688 * g - 0.081312 * b);
    }

    println!("Y block (first 8):  {:?}", &y_block[..8].iter().map(|x| *x as i32).collect::<Vec<_>>());
    println!("Cb block (first 8): {:?}", &cb_block[..8].iter().map(|x| *x as i32).collect::<Vec<_>>());
    println!("Cr block (first 8): {:?}", &cr_block[..8].iter().map(|x| *x as i32).collect::<Vec<_>>());

    println!("\n=== Step 2: Level shift (subtract 128) ===");
    for i in 0..64 {
        y_block[i] -= 128.0;
        cb_block[i] -= 128.0;
        cr_block[i] -= 128.0;
    }
    println!("Y shifted (first 8):  {:?}", &y_block[..8].iter().map(|x| *x as i32).collect::<Vec<_>>());

    println!("\n=== Step 3: Forward DCT ===");
    let y_dct_result = dct::forward_dct_blocks(&[y_block]);
    let cb_dct_result = dct::forward_dct_blocks(&[cb_block]);
    let y_dct = y_dct_result[0];
    let cb_dct = cb_dct_result[0];

    println!("Y DCT (first row): {:?}", &y_dct[..8].iter().map(|x| *x as i32).collect::<Vec<_>>());
    println!("Cb DCT (first row): {:?}", &cb_dct[..8].iter().map(|x| *x as i32).collect::<Vec<_>>());

    println!("\n=== Step 4: Quantization ===");
    let quality = Quality::Traditional(90.0);
    let y_qtable = quant::generate_quant_table(quality, 0, ColorSpace::YCbCr, false);
    let cb_qtable = quant::generate_quant_table(quality, 1, ColorSpace::YCbCr, false);

    println!("Y quant table (first row): {:?}", &y_qtable.values[..8]);
    println!("Cb quant table (first row): {:?}", &cb_qtable.values[..8]);

    // Quantize Y
    let mut y_quantized = [0i16; 64];
    for i in 0..64 {
        let q = y_qtable.values[i] as f32;
        y_quantized[i] = (y_dct[i] / q).round() as i16;
    }

    // Quantize Cb
    let mut cb_quantized = [0i16; 64];
    for i in 0..64 {
        let q = cb_qtable.values[i] as f32;
        cb_quantized[i] = (cb_dct[i] / q).round() as i16;
    }

    println!("Y quantized (zigzag first 16): {:?}", &zigzag_order(&y_quantized)[..16]);
    println!("Cb quantized (zigzag first 16): {:?}", &zigzag_order(&cb_quantized)[..16]);

    // Count non-zero coefficients
    let y_nonzero = y_quantized.iter().filter(|&&x| x != 0).count();
    let cb_nonzero = cb_quantized.iter().filter(|&&x| x != 0).count();
    println!("\nNon-zero coefficients: Y={}, Cb={}", y_nonzero, cb_nonzero);

    // Print DC values (these are most important for visual quality)
    println!("\n=== DC Coefficients ===");
    println!("Y DC: raw={:.2}, quant={}", y_dct[0], y_quantized[0]);
    println!("Cb DC: raw={:.2}, quant={}", cb_dct[0], cb_quantized[0]);
}

fn zigzag_order(block: &[i16; 64]) -> [i16; 64] {
    const ZIGZAG: [usize; 64] = [
        0,  1,  8, 16,  9,  2,  3, 10,
       17, 24, 32, 25, 18, 11,  4,  5,
       12, 19, 26, 33, 40, 48, 41, 34,
       27, 20, 13,  6,  7, 14, 21, 28,
       35, 42, 49, 56, 57, 50, 43, 36,
       29, 22, 15, 23, 30, 37, 44, 51,
       58, 59, 52, 45, 38, 31, 39, 46,
       53, 60, 61, 54, 47, 55, 62, 63
    ];
    let mut result = [0i16; 64];
    for i in 0..64 {
        result[i] = block[ZIGZAG[i]];
    }
    result
}
