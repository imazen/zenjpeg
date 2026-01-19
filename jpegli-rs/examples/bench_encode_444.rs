//! Benchmark 4:4:4 baseline encoding path
use enough::Unstoppable;
use jpegli::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};
use std::hint::black_box;
use std::time::Instant;

/// Generate a photo-like synthetic test image
fn generate_test_image(width: usize, height: usize) -> Vec<u8> {
    let mut pixels = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            // Create gradients and patterns that stress DCT
            let r = ((x as f32 / width as f32) * 255.0) as u8;
            let g = ((y as f32 / height as f32) * 255.0) as u8;
            let b = (((x + y) as f32 / (width + height) as f32) * 255.0) as u8;
            pixels[idx] = r;
            pixels[idx + 1] = g;
            pixels[idx + 2] = b;
        }
    }
    pixels
}

fn main() {
    // Generate 2K test image (similar to flower.png)
    let width: u32 = 2048;
    let height: u32 = 1536;
    let pixels = generate_test_image(width as usize, height as usize);

    println!(
        "Image: {}x{} ({:.2} Mpx)",
        width,
        height,
        (width * height) as f64 / 1_000_000.0
    );

    let iterations = 20;

    let config = EncoderConfig::ycbcr(90.0, ChromaSubsampling::None) // 4:4:4
        .progressive(false) // Baseline
        .optimize_huffman(false); // Fixed tables for consistency

    // Warmup
    for _ in 0..3 {
        let mut enc = config
            .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
            .unwrap();
        enc.push_packed(black_box(&pixels), Unstoppable).unwrap();
        let _ = enc.finish().unwrap();
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let mut enc = config
            .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
            .unwrap();
        enc.push_packed(black_box(&pixels), Unstoppable).unwrap();
        let result = enc.finish().unwrap();
        black_box(&result);
    }
    let elapsed = start.elapsed();

    println!(
        "YUV/SEQ/FIX/444: {:.2} ms/encode ({} iterations)",
        elapsed.as_millis() as f64 / iterations as f64,
        iterations
    );
    println!(
        "Throughput: {:.2} Mpx/s",
        (width * height) as f64 * iterations as f64 / elapsed.as_secs_f64() / 1_000_000.0
    );
}
