//! Flamegraph profiling for 4K and 8K encoding
//!
//! Run with: cargo flamegraph --release --example flamegraph_profile -- [4k|8k]
//!
//! Default is 4K. Use "8k" argument for 8K profiling.

use enough::Unstoppable;
use jpegli::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};
use std::env;
use std::hint::black_box;

fn create_test_image(width: usize, height: usize) -> Vec<u8> {
    let mut data = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            // Create gradient + noise pattern (realistic frequency content)
            let noise = ((x * 7 + y * 13) % 256) as u8;
            data[idx] = ((x * 255) / width) as u8;
            data[idx + 1] = ((y * 255) / height) as u8;
            data[idx + 2] = noise;
        }
    }
    data
}

fn encode_image(data: &[u8], width: u32, height: u32, quality: f32) -> Vec<u8> {
    let config = EncoderConfig::ycbcr(quality, ChromaSubsampling::Quarter);
    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .unwrap();
    enc.push_packed(data, Unstoppable).unwrap();
    enc.finish().unwrap()
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("4k");

    let (width, height, iterations) = match mode {
        "8k" => (7680usize, 4320usize, 3),
        _ => (3840usize, 2160usize, 10),
    };

    eprintln!("Creating {}x{} test image...", width, height);
    let data = create_test_image(width, height);

    eprintln!("Warming up (1 iteration)...");
    let _ = black_box(encode_image(&data, width as u32, height as u32, 90.0));

    eprintln!(
        "Profiling {} iterations at q90 ({}x{})...",
        iterations, width, height
    );

    for i in 0..iterations {
        let result = black_box(encode_image(&data, width as u32, height as u32, 90.0));
        if i == 0 {
            eprintln!(
                "  Output size: {:.1} KB ({:.2} bpp)",
                result.len() as f64 / 1024.0,
                result.len() as f64 * 8.0 / (width * height) as f64
            );
        }
    }

    eprintln!("Done.");
}
