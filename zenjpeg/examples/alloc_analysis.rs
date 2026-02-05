//! Allocation analysis for encoder with real-world images.
//!
//! Run with:
//!   heaptrack cargo run --release --example alloc_analysis
//!   heaptrack_print heaptrack.alloc_analysis.*.zst
//!
//! Uses CID22 high-resolution images from codec-corpus for realistic allocation patterns.
use enough::Unstoppable;

use std::fs::File;
use std::io::BufReader;
use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, PixelLayout};

fn load_png(path: &str) -> (Vec<u8>, u32, u32) {
    let file = File::open(path).expect("Failed to open PNG");
    let decoder = png::Decoder::new(BufReader::new(file));
    let mut reader = decoder.read_info().expect("Failed to read PNG info");
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).expect("Failed to decode PNG");
    buf.truncate(info.buffer_size());
    (buf, info.width, info.height)
}

fn main() {
    // Try to load real image from codec-corpus - prefer high-res CID22
    let corpus_path = std::env::var("CODEC_CORPUS")
        .unwrap_or_else(|_| std::env::var("HOME").unwrap() + "/work/codec-eval/codec-corpus");

    // CID22 has 2268x1512 images (like flower.png) - better test case than 768x512 Kodak
    let cid22_path = format!("{}/cid22/flower.png", corpus_path);
    let kodak_path = format!("{}/kodak/1.png", corpus_path);

    let (pixels, width, height) = if std::path::Path::new(&cid22_path).exists() {
        eprintln!("Loading: {}", cid22_path);
        load_png(&cid22_path)
    } else if std::path::Path::new(&kodak_path).exists() {
        eprintln!("Loading: {}", kodak_path);
        load_png(&kodak_path)
    } else {
        eprintln!("Corpus not found, using synthetic 1080p image");
        let width = 1920u32;
        let height = 1080u32;
        let mut pixels = vec![0u8; (width * height * 3) as usize];
        // Mix of smooth gradients and some texture (simulates real photo)
        for y in 0..height as usize {
            for x in 0..width as usize {
                let idx = (y * width as usize + x) * 3;
                let base_r = (x * 200 / width as usize) as u8;
                let base_g = (y * 180 / height as usize) as u8;
                let base_b = 100u8;
                let noise = ((x * 7 + y * 13) % 30) as i16 - 15;
                pixels[idx] = (base_r as i16 + noise).clamp(0, 255) as u8;
                pixels[idx + 1] = (base_g as i16 + noise / 2).clamp(0, 255) as u8;
                pixels[idx + 2] = (base_b as i16 - noise / 3).clamp(0, 255) as u8;
            }
        }
        (pixels, width, height)
    };

    eprintln!("Image: {}x{} = {} pixels", width, height, width * height);
    eprintln!("Input size: {} bytes", pixels.len());

    let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter);

    let mut encoder = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .expect("Failed to create encoder");

    encoder
        .push_packed(&pixels, enough::Unstoppable)
        .expect("Failed to push pixels");

    let mut output = Vec::new();
    encoder.finish_into(&mut output).expect("Failed to finish");

    let y_blocks = ((width + 7) / 8) * ((height + 7) / 8);
    let chroma_blocks = 2 * ((width + 15) / 16) * ((height + 15) / 16);
    let blocks = y_blocks + chroma_blocks;
    eprintln!("Output size: {} bytes", output.len());
    eprintln!(
        "Blocks: {} (Y={}, Cb/Cr={})",
        blocks, y_blocks, chroma_blocks
    );
    eprintln!("Bytes/block: {:.2}", output.len() as f64 / blocks as f64)
}
