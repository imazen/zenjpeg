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

    // Default: 1920x1080. Override with WIDTH=7680 HEIGHT=4320 for 8K.
    let width: usize = env::var("WIDTH").ok().and_then(|s| s.parse().ok()).unwrap_or(1920);
    let height: usize = env::var("HEIGHT").ok().and_then(|s| s.parse().ok()).unwrap_or(1080);

    // Row buffer for streaming — only MCU-row height, not full image
    let mcu_rows = 16;
    let row_buf_size = width * mcu_rows * 3;

    // Also encode with all modes to compare sizes (using noise pattern)
    if mode == "baseline" {
        let mut all_pixels = vec![0u8; width * height * 3];
        let mut rng: u32 = 0xDEAD_BEEF;
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 3;
                // LCG noise + 8x8 block structure
                rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                let block_val = (((x / 64) * 37 + (y / 64) * 71) % 256) as u8;
                let noise = ((rng >> 16) & 0x1F) as u8;
                all_pixels[idx] = block_val.wrapping_add(noise);
                all_pixels[idx + 1] = block_val.wrapping_add(noise.wrapping_mul(2));
                all_pixels[idx + 2] = block_val.wrapping_add(noise.wrapping_mul(3));
            }
        }
        for (name, cfg) in [
            ("baseline-fixed", EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
                .progressive(false).optimize_huffman(false)),
            ("baseline-optimized", EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
                .progressive(false)),
            ("progressive", EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
                .progressive(true)),
        ] {
            let mut enc = cfg.encode_from_bytes(width as u32, height as u32, PixelLayout::Rgb8Srgb).unwrap();
            enc.push_packed(&all_pixels, enough::Unstoppable).unwrap();
            let out = enc.finish().unwrap();
            let overhead = if name != "baseline-fixed" { String::new() }
            else { String::new() };
            eprintln!("  {name}: {} bytes", out.len());
        }
        let _ = all_pixels; // drop before streaming
    }

    eprintln!("Image: {}x{} = {} pixels", width, height, width * height);
    eprintln!("Mode: {}", mode);

    let config = match mode {
        "baseline" => {
            eprintln!("Using: Baseline + Optimized Huffman");
            EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
                .progressive(false)
        }
        "baseline-fixed" => {
            eprintln!("Using: Baseline + Fixed Huffman (no optimization)");
            EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
                .progressive(false)
                .optimize_huffman(false)
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

    let estimate = config.estimate_memory(width as u32, height as u32);
    eprintln!("Estimated encoder memory: {} bytes ({:.1} KB)", estimate, estimate as f64 / 1024.0);

    eprintln!("Row buffer: {} bytes ({:.1} KB)", row_buf_size, row_buf_size as f64 / 1024.0);

    let mut encoder = config
        .encode_from_bytes(width as u32, height as u32, PixelLayout::Rgb8Srgb)
        .expect("Failed to create encoder");

    // Generate and push rows on the fly — no full-image allocation
    let mut row_buf = vec![0u8; row_buf_size];
    for chunk_start in (0..height).step_by(mcu_rows) {
        let chunk_end = (chunk_start + mcu_rows).min(height);
        let chunk_rows = chunk_end - chunk_start;
        for y in 0..chunk_rows {
            for x in 0..width {
                let idx = (y * width + x) * 3;
                row_buf[idx] = (x * 255 / width) as u8;
                row_buf[idx + 1] = ((chunk_start + y) * 255 / height) as u8;
                row_buf[idx + 2] = 128;
            }
        }
        encoder
            .push_packed(&row_buf[..chunk_rows * width * 3], enough::Unstoppable)
            .expect("Failed to push rows");
    }

    let mut output = Vec::new();
    encoder.finish_into(&mut output).expect("Failed to finish");

    eprintln!("Output size: {} bytes", output.len());
}
