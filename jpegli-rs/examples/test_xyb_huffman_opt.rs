use jpegli::{Encoder, PixelFormat};

fn main() {
    // Simple 64x64 gradient - same as debug_ac_refinement.rs
    let width = 64;
    let height = 64;
    let mut data = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            data[idx] = ((x * 255) / width) as u8; // R gradient
            data[idx + 1] = ((y * 255) / height) as u8; // G gradient
            data[idx + 2] = 128; // B constant
        }
    }

    println!("=== Testing XYB Progressive with Huffman Optimization ===\n");

    // WITHOUT optimization
    let jpeg_no_opt = Encoder::new()
        .width(width as u32)
        .height(height as u32)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(jpegli::quant::Quality::from_quality(90.0))
        .mode(jpegli::types::JpegMode::Progressive)
        .use_xyb(true)
        .optimize_huffman(false)
        .encode(&data)
        .unwrap();

    // WITH optimization
    let jpeg_opt = Encoder::new()
        .width(width as u32)
        .height(height as u32)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(jpegli::quant::Quality::from_quality(90.0))
        .mode(jpegli::types::JpegMode::Progressive)
        .use_xyb(true)
        .optimize_huffman(true)
        .encode(&data)
        .unwrap();

    println!("Without optimization: {} bytes", jpeg_no_opt.len());
    println!("With optimization:    {} bytes", jpeg_opt.len());
    println!(
        "Reduction:            {} bytes ({:.1}%)",
        jpeg_no_opt.len() as i32 - jpeg_opt.len() as i32,
        100.0 * (1.0 - jpeg_opt.len() as f64 / jpeg_no_opt.len() as f64)
    );

    println!("\nC++ reference (from earlier): 1537 bytes");
    println!(
        "Rust optimized gap:           {} bytes ({:.1}× larger)",
        jpeg_opt.len() as i32 - 1537,
        jpeg_opt.len() as f64 / 1537.0
    );
}
