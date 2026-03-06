//! Debug XYB encoding: compare Rust vs C++ jpegli per-channel.
//!
//! Usage: cargo run --release --example xyb_debug [image.png]
//! Default uses a 64x32 synthetic gradient if no image provided.

use enough::Unstoppable;
use std::process::Command;
use zenjpeg::encoder::{EncoderConfig, PixelLayout, XybSubsampling};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let (rgb, width, height, png_path) = if args.len() > 1 {
        // Load real image
        let path = &args[1];
        let (rgb, w, h) = load_png_rgb(path);
        (rgb, w, h, path.to_string())
    } else {
        // Synthetic gradient
        let width = 64usize;
        let height = 32usize;
        let mut rgb = vec![0u8; width * height * 3];
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 3;
                rgb[idx] = (x * 4) as u8;
                rgb[idx + 1] = (y * 8) as u8;
                rgb[idx + 2] = 128;
            }
        }
        let path = "/tmp/xyb_debug_input.png";
        save_png(&rgb, width, height, path);
        (rgb, width, height, path.to_string())
    };

    let quality = 90.0;

    println!("XYB Debug: {}x{}", width, height);
    println!("Input: {}", png_path);
    println!();

    // Encode with Rust
    let config = EncoderConfig::xyb(quality, XybSubsampling::BQuarter).optimize_huffman(true);
    let mut enc = config
        .encode_from_bytes(width as u32, height as u32, PixelLayout::Rgb8Srgb)
        .expect("encoder setup");
    enc.push_packed(&rgb, Unstoppable).expect("push");
    let rust_jpeg = enc.finish().expect("encode");

    let rust_path = "/tmp/xyb_debug_rust.jpg";
    std::fs::write(rust_path, &rust_jpeg).expect("write");
    println!("Rust: {} bytes", rust_jpeg.len());

    // Encode with C++ jpegli
    let cpp_path = "/tmp/xyb_debug_cpp.jpg";
    let cpp_result = Command::new("cjpegli")
        .args([&png_path, cpp_path, "-q", &quality.to_string(), "--xyb"])
        .output()
        .expect("cjpegli");

    if !cpp_result.status.success() {
        eprintln!(
            "cjpegli failed: {}",
            String::from_utf8_lossy(&cpp_result.stderr)
        );
        return;
    }
    let cpp_jpeg = std::fs::read(cpp_path).expect("read cpp");
    println!("C++:  {} bytes", cpp_jpeg.len());
    println!(
        "Size diff: {:+.1}%",
        (rust_jpeg.len() as f64 / cpp_jpeg.len() as f64 - 1.0) * 100.0
    );
    println!();

    // Decode both with djpegli
    let rust_decoded_path = "/tmp/xyb_debug_rust_decoded.png";
    let cpp_decoded_path = "/tmp/xyb_debug_cpp_decoded.png";
    decode_with_djpegli(rust_path, rust_decoded_path);
    decode_with_djpegli(cpp_path, cpp_decoded_path);

    let rust_decoded = load_png_rgb(rust_decoded_path).0;
    let cpp_decoded = load_png_rgb(cpp_decoded_path).0;

    // Per-row analysis
    println!("Per-row mean |diff| (R, G, B):");
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
        println!("{:>4} {:>8.2} {:>8.2} {:>8.2}", y, r_diff, g_diff, b_diff);
    }

    // Summary
    let pixels = (width * height) as f64;
    let mut total_r = 0.0f64;
    let mut total_g = 0.0f64;
    let mut total_b = 0.0f64;
    for i in 0..(width * height) {
        let idx = i * 3;
        total_r += (rust_decoded[idx] as i32 - cpp_decoded[idx] as i32).abs() as f64;
        total_g += (rust_decoded[idx + 1] as i32 - cpp_decoded[idx + 1] as i32).abs() as f64;
        total_b += (rust_decoded[idx + 2] as i32 - cpp_decoded[idx + 2] as i32).abs() as f64;
    }
    println!();
    println!(
        "Mean |diff|: R={:.3}, G={:.3}, B={:.3}",
        total_r / pixels,
        total_g / pixels,
        total_b / pixels
    );
}

fn save_png(rgb: &[u8], width: usize, height: usize, path: &str) {
    let file = std::fs::File::create(path).unwrap();
    let w = std::io::BufWriter::new(file);
    let mut encoder = png::Encoder::new(w, width as u32, height as u32);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().unwrap();
    writer.write_image_data(rgb).unwrap();
}

fn load_png_rgb(path: &str) -> (Vec<u8>, usize, usize) {
    let img =
        zenjpeg_bench_utils::load_png(std::path::Path::new(path)).expect("Failed to load PNG");
    let bytes: Vec<u8> = img.buf().iter().flat_map(|p| [p.r, p.g, p.b]).collect();
    (bytes, img.width(), img.height())
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
