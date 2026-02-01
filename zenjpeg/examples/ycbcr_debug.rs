//! Debug YCbCr encoding: compare Rust vs C++ jpegli per-channel.
//!
//! Usage: cargo run --release --example ycbcr_debug [image.png]

use enough::Unstoppable;
use std::process::Command;
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let (rgb, width, height, png_path) = if args.len() > 1 {
        let path = &args[1];
        let (rgb, w, h) = load_png(path);
        (rgb, w, h, path.to_string())
    } else {
        eprintln!("Usage: ycbcr_debug <image.png>");
        std::process::exit(1);
    };

    let quality = 90.0;

    println!("YCbCr Debug: {}x{}", width, height);
    println!("Input: {}", png_path);
    println!();

    // Encode with Rust (4:2:0)
    let config = EncoderConfig::ycbcr(quality, ChromaSubsampling::Quarter).optimize_huffman(true);
    let mut enc = config
        .encode_from_bytes(width as u32, height as u32, PixelLayout::Rgb8Srgb)
        .expect("encoder setup");
    enc.push_packed(&rgb, Unstoppable).expect("push");
    let rust_jpeg = enc.finish().expect("encode");

    let rust_path = "/tmp/ycbcr_debug_rust.jpg";
    std::fs::write(rust_path, &rust_jpeg).expect("write");
    println!("Rust: {} bytes", rust_jpeg.len());

    // Encode with C++ jpegli (4:2:0)
    let cpp_path = "/tmp/ycbcr_debug_cpp.jpg";
    let cpp_result = Command::new("cjpegli")
        .args([&png_path, cpp_path, "-q", &quality.to_string()])
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
    let rust_decoded_path = "/tmp/ycbcr_debug_rust_decoded.png";
    let cpp_decoded_path = "/tmp/ycbcr_debug_cpp_decoded.png";
    decode_with_djpegli(rust_path, rust_decoded_path);
    decode_with_djpegli(cpp_path, cpp_decoded_path);

    let rust_decoded = load_png(rust_decoded_path).0;
    let cpp_decoded = load_png(cpp_decoded_path).0;

    // Summary stats
    let pixels = (width * height) as f64;
    let mut total_r = 0.0f64;
    let mut total_g = 0.0f64;
    let mut total_b = 0.0f64;
    let mut max_r = 0i32;
    let mut max_g = 0i32;
    let mut max_b = 0i32;

    for i in 0..(width * height) {
        let idx = i * 3;
        let dr = (rust_decoded[idx] as i32 - cpp_decoded[idx] as i32).abs();
        let dg = (rust_decoded[idx + 1] as i32 - cpp_decoded[idx + 1] as i32).abs();
        let db = (rust_decoded[idx + 2] as i32 - cpp_decoded[idx + 2] as i32).abs();
        total_r += dr as f64;
        total_g += dg as f64;
        total_b += db as f64;
        max_r = max_r.max(dr);
        max_g = max_g.max(dg);
        max_b = max_b.max(db);
    }

    println!(
        "Mean |diff|: R={:.3}, G={:.3}, B={:.3}",
        total_r / pixels,
        total_g / pixels,
        total_b / pixels
    );
    println!("Max  |diff|: R={}, G={}, B={}", max_r, max_g, max_b);
}

fn load_png(path: &str) -> (Vec<u8>, usize, usize) {
    let file = std::fs::File::open(path).expect("open file");
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().expect("read info");
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).expect("decode");
    (
        buf[..info.buffer_size()].to_vec(),
        info.width as usize,
        info.height as usize,
    )
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
