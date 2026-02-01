//! Debug XYB B channel encoding to identify the row pattern issue.
//!
//! Compares Rust vs C++ jpegli XYB encoding at the coefficient level.

use enough::Unstoppable;
use std::process::Command;
use zenjpeg::encoder::{EncoderConfig, PixelLayout, XybSubsampling};

fn main() {
    // Use a small test image to make analysis tractable
    let width = 64usize;
    let height = 32usize;
    let quality = 90.0;

    // Create gradient test image
    let mut rgb = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            // Simple gradient for predictable XYB values
            rgb[idx] = (x * 4) as u8; // R
            rgb[idx + 1] = (y * 8) as u8; // G
            rgb[idx + 2] = 128; // B constant
        }
    }

    // Save as PNG for C++ encoding
    let png_path = "/tmp/xyb_debug_input.png";
    save_png(&rgb, width, height, png_path);

    println!("XYB B-Channel Debug Analysis");
    println!("============================");
    println!("Image: {}x{}", width, height);
    println!("B channel dimensions: {}x{}", (width + 1) / 2, (height + 1) / 2);
    println!();

    // 1. Encode with Rust
    println!("=== Rust XYB Encoding ===");
    let config = EncoderConfig::xyb(quality, XybSubsampling::BQuarter).optimize_huffman(true);
    let mut enc = config
        .encode_from_bytes(width as u32, height as u32, PixelLayout::Rgb8Srgb)
        .expect("encoder setup");
    enc.push_packed(&rgb, Unstoppable).expect("push");
    let rust_jpeg = enc.finish().expect("Rust XYB encode failed");

    let rust_path = "/tmp/xyb_debug_rust.jpg";
    std::fs::write(rust_path, &rust_jpeg).expect("write rust jpeg");
    println!("Rust JPEG: {} bytes", rust_jpeg.len());

    // 2. Encode with C++ jpegli
    println!("\n=== C++ jpegli Encoding ===");
    let cpp_path = "/tmp/xyb_debug_cpp.jpg";
    let cpp_result = Command::new("cjpegli")
        .args([png_path, cpp_path, "-q", &quality.to_string(), "--xyb"])
        .output()
        .expect("cjpegli");

    if !cpp_result.status.success() {
        eprintln!(
            "cjpegli failed: {}",
            String::from_utf8_lossy(&cpp_result.stderr)
        );
        return;
    }
    let cpp_jpeg = std::fs::read(cpp_path).expect("read cpp jpeg");
    println!("C++ JPEG: {} bytes", cpp_jpeg.len());

    // 3. Decode both and compare
    println!("\n=== Decoded RGB Comparison ===");

    // Use djpegli for consistent decoding
    let rust_png_path = "/tmp/xyb_debug_rust_decoded.png";
    let cpp_png_path = "/tmp/xyb_debug_cpp_decoded.png";

    decode_with_djpegli(rust_path, rust_png_path);
    decode_with_djpegli(cpp_path, cpp_png_path);

    let rust_decoded = load_png(rust_png_path);
    let cpp_decoded = load_png(cpp_png_path);

    if rust_decoded.len() != cpp_decoded.len() {
        println!("ERROR: Decoded size mismatch!");
        println!("  Rust: {} bytes", rust_decoded.len());
        println!("  C++:  {} bytes", cpp_decoded.len());
        return;
    }

    // Analyze per-channel, per-row differences
    println!("\nPer-row average absolute difference (R, G, B):");
    println!("{:>4} {:>8} {:>8} {:>8}", "Row", "R", "G", "B");
    println!("{}", "-".repeat(36));

    for y in 0..height {
        let mut r_diff = 0.0f64;
        let mut g_diff = 0.0f64;
        let mut b_diff = 0.0f64;

        for x in 0..width {
            let idx = (y * width + x) * 3;
            r_diff += (rust_decoded[idx] as i32 - cpp_decoded[idx] as i32).abs() as f64;
            g_diff += (rust_decoded[idx + 1] as i32 - cpp_decoded[idx + 1] as i32).abs() as f64;
            b_diff += (rust_decoded[idx + 2] as i32 - cpp_decoded[idx + 2] as i32).abs() as f64;
        }

        r_diff /= width as f64;
        g_diff /= width as f64;
        b_diff /= width as f64;

        let marker = if y % 2 == 0 { "EVEN" } else { "odd " };
        println!("{:>4} {:>8.2} {:>8.2} {:>8.2}  {}", y, r_diff, g_diff, b_diff, marker);
    }

    // Summary statistics
    println!("\n=== Summary ===");
    let mut total_r = 0.0f64;
    let mut total_g = 0.0f64;
    let mut total_b = 0.0f64;

    for i in 0..(width * height) {
        let idx = i * 3;
        total_r += (rust_decoded[idx] as i32 - cpp_decoded[idx] as i32).abs() as f64;
        total_g += (rust_decoded[idx + 1] as i32 - cpp_decoded[idx + 1] as i32).abs() as f64;
        total_b += (rust_decoded[idx + 2] as i32 - cpp_decoded[idx + 2] as i32).abs() as f64;
    }

    let pixels = (width * height) as f64;
    println!("Mean absolute error:");
    println!("  R: {:.3}", total_r / pixels);
    println!("  G: {:.3}", total_g / pixels);
    println!("  B: {:.3}", total_b / pixels);
}

fn save_png(rgb: &[u8], width: usize, height: usize, path: &str) {
    let file = std::fs::File::create(path).expect("create file");
    let ref mut w = std::io::BufWriter::new(file);
    let mut encoder = png::Encoder::new(w, width as u32, height as u32);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().expect("write header");
    writer.write_image_data(rgb).expect("write image");
}

fn load_png(path: &str) -> Vec<u8> {
    let file = std::fs::File::open(path).expect("open file");
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().expect("read info");
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).expect("decode");
    buf[..info.buffer_size()].to_vec()
}

fn decode_with_djpegli(jpeg_path: &str, png_path: &str) {
    let result = Command::new("djpegli")
        .args([jpeg_path, png_path])
        .output()
        .expect("djpegli");

    if !result.status.success() {
        eprintln!(
            "djpegli failed: {}",
            String::from_utf8_lossy(&result.stderr)
        );
    }
}
