//! Decode-only profiling (no encode overhead).
//! Takes a pre-existing JPEG file and decodes it N times.
//!
//! Usage:
//!   cargo build --release --example profile_decode_only --features decoder
//!   # Generate test JPEG first:
//!   cargo run --release --example profile_decode_only --features decoder -- generate 1024
//!   # Then profile decode only:
//!   valgrind --tool=callgrind ./target/release/examples/profile_decode_only decode /tmp/zenjpeg_profile_1024x1024.jpg 1

use enough::Unstoppable;
use std::env;
use std::hint::black_box;
use zenjpeg::{
    decoder::{Decoder, PixelFormat},
    encoder::{ChromaSubsampling, EncoderConfig, PixelLayout},
};

fn create_test_jpeg(width: u32, height: u32) -> Vec<u8> {
    let mut data = vec![0u8; (width * height * 3) as usize];
    let mut rng: u32 = 0xDEADBEEF;
    for y in 0..height as usize {
        for x in 0..width as usize {
            let idx = (y * width as usize + x) * 3;
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            let noise = ((rng >> 16) & 0xFF) as u8;
            let patch_x = (x / 64) & 3;
            let patch_y = (y / 64) & 3;
            let base = ((patch_x * 64 + patch_y * 32) & 255) as u8;
            data[idx] = base.wrapping_add(noise >> 2);
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            data[idx + 1] = base.wrapping_add(((rng >> 16) & 0x3F) as u8);
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            data[idx + 2] = (255 - base).wrapping_add(((rng >> 16) & 0x1F) as u8);
        }
    }
    // Baseline (not progressive) for decode speed testing
    let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter).progressive(false);
    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .unwrap();
    enc.push_packed(&data, Unstoppable).unwrap();
    enc.finish().unwrap()
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage:");
        eprintln!("  {} generate <size>          # Generate test JPEG", args[0]);
        eprintln!(
            "  {} decode <file.jpg> [reps] # Decode JPEG N times",
            args[0]
        );
        std::process::exit(1);
    }

    match args[1].as_str() {
        "generate" => {
            let size: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1024);
            eprintln!("Creating {}x{} baseline test JPEG...", size, size);
            let jpeg_data = create_test_jpeg(size, size);
            let path = format!("/tmp/zenjpeg_profile_{}x{}.jpg", size, size);
            std::fs::write(&path, &jpeg_data).unwrap();
            eprintln!("Wrote {} bytes to {}", jpeg_data.len(), path);
        }
        "decode" => {
            let path = args
                .get(2)
                .expect("need JPEG file path")
                .clone();
            let reps: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);
            let jpeg_data = std::fs::read(&path).expect("failed to read JPEG");
            eprintln!(
                "Decoding {} ({} bytes) {} times...",
                path,
                jpeg_data.len(),
                reps
            );
            for _ in 0..reps {
                let decoder = Decoder::new().output_format(PixelFormat::Rgb);
                let result = decoder
                    .decode(black_box(&jpeg_data), Unstoppable)
                    .expect("decode failed");
                black_box(&result);
            }
            eprintln!("Done.");
        }
        _ => {
            eprintln!("Unknown command: {}. Use 'generate' or 'decode'", args[1]);
            std::process::exit(1);
        }
    }
}
