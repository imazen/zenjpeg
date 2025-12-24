//! Compare single block encoding between C++ and Rust

use std::fs;
use std::io::Write;
use std::process::Command;

fn main() {
    // Create an 8x8 test pattern - simple gradient
    let mut pixels = vec![0u8; 8 * 8 * 3];
    for y in 0..8 {
        for x in 0..8 {
            let idx = (y * 8 + x) * 3;
            let val = ((x + y) * 16) as u8; // 0 to 224
            pixels[idx] = val; // R
            pixels[idx + 1] = val; // G
            pixels[idx + 2] = val; // B
        }
    }

    // Convert to YCbCr (grayscale for simplicity)
    let mut y_block = [0.0f32; 64];
    for i in 0..64 {
        let r = pixels[i * 3] as f32;
        let g = pixels[i * 3 + 1] as f32;
        let b = pixels[i * 3 + 2] as f32;
        y_block[i] = 0.299 * r + 0.587 * g + 0.114 * b;
    }

    println!("=== Input Y values (8x8 block) ===");
    println!(
        "First row: {:?}",
        &y_block[0..8]
            .iter()
            .map(|x| format!("{:.1}", x))
            .collect::<Vec<_>>()
    );
    println!("Average: {:.2}", y_block.iter().sum::<f32>() / 64.0);

    // Level shift
    let mut y_shifted = y_block;
    for v in &mut y_shifted {
        *v -= 128.0;
    }

    println!("\n=== After level shift (Y - 128) ===");
    println!(
        "First row: {:?}",
        &y_shifted[0..8]
            .iter()
            .map(|x| format!("{:.1}", x))
            .collect::<Vec<_>>()
    );

    // DCT
    let y_dct = jpegli::dct::forward_dct_blocks(&[y_shifted])[0];

    println!("\n=== DCT coefficients ===");
    println!("DC = {:.4}", y_dct[0]);
    println!(
        "First row: {:?}",
        &y_dct[0..8]
            .iter()
            .map(|x| format!("{:.2}", x))
            .collect::<Vec<_>>()
    );

    // Quantization at Q90
    let qtable = jpegli::quant::generate_quant_table(
        jpegli::quant::Quality::Traditional(90.0),
        0,
        jpegli::types::ColorSpace::YCbCr,
        false,
    );

    println!("\n=== Quantization table (Q90 Y) ===");
    println!("First row: {:?}", &qtable.values[0..8]);

    // Quantize
    let mut quantized = [0i16; 64];
    for i in 0..64 {
        quantized[i] = (y_dct[i] / qtable.values[i] as f32).round() as i16;
    }

    println!("\n=== Quantized coefficients ===");
    println!("DC = {}", quantized[0]);
    println!("First row: {:?}", &quantized[0..8]);
    println!(
        "Non-zero count: {}",
        quantized.iter().filter(|&&x| x != 0).count()
    );

    // Now encode with C++ and extract its coefficients
    let ppm_path = "/tmp/test_8x8.ppm";
    {
        let mut f = fs::File::create(ppm_path).unwrap();
        writeln!(f, "P6\n8 8\n255").unwrap();
        f.write_all(&pixels).unwrap();
    }

    let cpp_path = "/tmp/cpp_8x8.jpg";
    Command::new("/home/lilith/work/jpegli/build/tools/cjpegli")
        .args([
            "--noadaptive_quantization",
            "--chroma_subsampling=444",
            "-p",
            "0",
            "--fixed_code",
            ppm_path,
            cpp_path,
            "-q",
            "90",
        ])
        .output()
        .expect("cjpegli failed");

    // Use djpeg to extract DCT coefficients (if possible) or just compare sizes
    let cpp_data = fs::read(cpp_path).unwrap();
    println!("\n=== File sizes ===");
    println!("C++ JPEG: {} bytes", cpp_data.len());

    // Rust encode
    let rust_jpeg = jpegli::encode::Encoder::new()
        .width(8)
        .height(8)
        .quality(jpegli::quant::Quality::Traditional(90.0))
        .encode(&pixels)
        .expect("Rust encoding failed");

    println!("Rust JPEG: {} bytes", rust_jpeg.len());
}
