//! Debug Huffman decode failures

use jpegli::{Encoder, Quality};
use std::fs::File;

fn load_png(path: &str) -> (u32, u32, Vec<u8>) {
    let file = File::open(path).unwrap();
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    buf.truncate(info.buffer_size());

    let pixels = match info.color_type {
        png::ColorType::Rgb => buf,
        png::ColorType::Rgba => buf.chunks(4).flat_map(|c| [c[0], c[1], c[2]]).collect(),
        png::ColorType::Grayscale => buf.iter().flat_map(|&g| [g, g, g]).collect(),
        png::ColorType::GrayscaleAlpha => buf.chunks(2).flat_map(|c| [c[0], c[0], c[0]]).collect(),
        _ => panic!("Unsupported color type"),
    };
    (info.width, info.height, pixels)
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/mnt/v/work/corpus/CID22-512/1044329.png".to_string());

    println!("Loading: {}", path);
    let (w, h, pixels) = load_png(&path);
    println!("Image: {}x{}, {} bytes", w, h, pixels.len());

    // Test standard encoding
    println!("\n--- Standard Huffman ---");
    let jpeg_std = Encoder::new()
        .width(w)
        .height(h)
        .quality(Quality::from_quality(75.0))
        .optimize_huffman(false)
        .encode(&pixels)
        .unwrap();
    println!("JPEG size: {} bytes", jpeg_std.len());

    match jpeg_decoder::Decoder::new(&jpeg_std[..]).decode() {
        Ok(p) => println!("Decode OK: {} pixels", p.len()),
        Err(e) => println!("Decode FAILED: {:?}", e),
    }

    // Test optimized encoding
    println!("\n--- Optimized Huffman ---");
    let jpeg_opt = Encoder::new()
        .width(w)
        .height(h)
        .quality(Quality::from_quality(75.0))
        .optimize_huffman(true)
        .encode(&pixels)
        .unwrap();
    println!("JPEG size: {} bytes", jpeg_opt.len());

    // Save for external inspection
    std::fs::write("/tmp/test_optimized.jpg", &jpeg_opt).unwrap();
    println!("Saved to /tmp/test_optimized.jpg");

    match jpeg_decoder::Decoder::new(&jpeg_opt[..]).decode() {
        Ok(p) => println!("Decode OK: {} pixels", p.len()),
        Err(e) => {
            println!("Decode FAILED: {:?}", e);

            // Try with djpeg
            println!("\nTrying djpeg...");
            let output = std::process::Command::new("djpeg")
                .args(["-outfile", "/tmp/test_out.ppm", "/tmp/test_optimized.jpg"])
                .output();
            match output {
                Ok(o) => {
                    if o.status.success() {
                        println!("djpeg succeeded!");
                    } else {
                        println!("djpeg failed: {}", String::from_utf8_lossy(&o.stderr));
                    }
                }
                Err(e) => println!("djpeg not available: {}", e),
            }
        }
    }
}
