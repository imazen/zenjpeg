//! Heaptrack test for encoder memory usage.
//!
//! Run with:
//!   heaptrack cargo run --release --example heaptrack_encode -- baseline
//!   heaptrack cargo run --release --example heaptrack_encode -- progressive

use std::env;
use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, PixelLayout};

fn main() {
    let args: Vec<String> = env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("baseline");

    // 1920x1080 test image (2 megapixels)
    let width = 1920usize;
    let height = 1080usize;

    // Generate simple gradient test image
    let mut pixels = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            pixels[idx] = (x * 255 / width) as u8; // R
            pixels[idx + 1] = (y * 255 / height) as u8; // G
            pixels[idx + 2] = 128; // B
        }
    }

    eprintln!("Image: {}x{} = {} pixels", width, height, width * height);
    eprintln!("Mode: {}", mode);

    let config = match mode {
        "baseline" => {
            eprintln!("Using: Baseline + Optimized Huffman");
            EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
        }
        "baseline-fixed" => {
            eprintln!("Using: Baseline + Fixed Huffman (no optimization)");
            EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter).optimize_huffman(false)
        }
        "progressive" => {
            eprintln!("Using: Progressive + Optimized Huffman");
            EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter).progressive(true)
        }
        _ => {
            eprintln!("Usage: heaptrack_encode [baseline|baseline-fixed|progressive]");
            return;
        }
    };

    // Encode
    let mut encoder = config
        .encode_from_bytes(width as u32, height as u32, PixelLayout::Rgb8Srgb)
        .expect("Failed to create encoder");

    encoder
        .push_packed(&pixels, enough::Unstoppable)
        .expect("Failed to push pixels");

    let mut output = Vec::new();
    encoder.finish_into(&mut output).expect("Failed to finish");

    eprintln!("Output size: {} bytes", output.len());
}
