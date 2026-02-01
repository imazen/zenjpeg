//! Compare progressive XYB output: Rust vs C jpegli.

use enough::Unstoppable;
use std::process::Command;
use zenjpeg::encoder::{EncoderConfig, PixelLayout, XybSubsampling};

fn main() {
    let images = [
        "/home/lilith/work/codec-eval/codec-corpus/kodak/1.png",
        "/home/lilith/work/codec-eval/codec-corpus/kodak/5.png",
        "/home/lilith/work/codec-eval/codec-corpus/kodak/13.png",
        "/home/lilith/work/codec-eval/codec-corpus/kodak/19.png",
    ];
    let qualities = [70, 80, 90];

    println!("Progressive XYB Comparison: Rust vs C jpegli");
    println!("{}", "=".repeat(80));
    println!();
    println!(
        "{:<15} {:<5} {:>12} {:>12} {:>10}",
        "Image", "Q", "C++ bytes", "Rust bytes", "Diff %"
    );
    println!("{}", "-".repeat(80));

    let cjpegli = "internal/jpegli-cpp/build/tools/cjpegli";
    
    for img_path in &images {
        let img_name = std::path::Path::new(img_path)
            .file_name()
            .unwrap()
            .to_string_lossy();

        // Load image once
        let file = std::fs::File::open(img_path).expect("open");
        let decoder = png::Decoder::new(file);
        let mut reader = decoder.read_info().expect("info");
        let mut buf = vec![0; reader.output_buffer_size()];
        let info = reader.next_frame(&mut buf).expect("decode");
        let pixels = &buf[..info.buffer_size()];
        let width = info.width;
        let height = info.height;

        for &q in &qualities {
            // C++ jpegli progressive XYB (default)
            let cpp_path = format!("/tmp/cpp_prog_xyb_{}_{}.jpg", img_name, q);
            let output = Command::new(cjpegli)
                .args(&[
                    *img_path,
                    &cpp_path,
                    "-q",
                    &q.to_string(),
                    "--xyb",
                ])
                .output()
                .expect("cjpegli");
            
            if !output.status.success() {
                eprintln!("cjpegli failed: {}", String::from_utf8_lossy(&output.stderr));
            }
            let cpp_bytes = std::fs::metadata(&cpp_path).map(|m| m.len()).unwrap_or(0);

            // Rust progressive XYB
            let rust_jpeg = {
                let config = EncoderConfig::xyb(q as f32, XybSubsampling::BQuarter)
                    .progressive(true);
                let mut enc = config
                    .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
                    .expect("encoder setup");
                enc.push_packed(pixels, Unstoppable).expect("push");
                enc.finish().expect("encode")
            };
            let rust_bytes = rust_jpeg.len();

            let diff = 100.0 * (rust_bytes as f64 - cpp_bytes as f64) / cpp_bytes as f64;

            println!(
                "{:<15} {:<5} {:>12} {:>12} {:>+9.2}%",
                img_name, q, cpp_bytes, rust_bytes, diff
            );
        }
    }
    
    println!("{}", "-".repeat(80));
    println!("Note: Both encoders using progressive mode with optimized Huffman");
}
