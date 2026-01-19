//! Profile individual encoding stages

use enough::Unstoppable;
use std::hint::black_box;
use std::time::Instant;

use jpegli::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};

fn main() {
    let width = 2048u32;
    let height = 2048u32;
    let pixels = (width * height) as usize;

    let rgb: Vec<u8> = (0..pixels * 3)
        .map(|i| {
            let x = (i / 3) % width as usize;
            let y = (i / 3) / width as usize;
            let c = i % 3;
            match c {
                0 => ((x * 255) / width as usize) as u8,
                1 => ((y * 255) / height as usize) as u8,
                _ => 128u8,
            }
        })
        .collect();

    println!(
        "Image: {}x{} ({:.2} MP)\n",
        width,
        height,
        pixels as f64 / 1_000_000.0
    );

    let iterations = 5;

    for (name, progressive, subsampling, optimize) in [
        ("Baseline/Fixed/444", false, ChromaSubsampling::None, false),
        ("Baseline/Opt/444", false, ChromaSubsampling::None, true),
        ("Baseline/Opt/420", false, ChromaSubsampling::Quarter, true),
        ("Progressive/Opt/444", true, ChromaSubsampling::None, true),
    ] {
        let mut total_time = std::time::Duration::ZERO;
        let mut output_size = 0;

        for _ in 0..iterations {
            let start = Instant::now();

            let config = EncoderConfig::ycbcr(85.0, subsampling)
                .progressive(progressive)
                .optimize_huffman(optimize);
            let mut enc = config
                .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
                .expect("encoder setup");
            enc.push_packed(&rgb, Unstoppable).expect("push");
            let jpeg = enc.finish().expect("encode failed");

            total_time += start.elapsed();
            output_size = jpeg.len();
            black_box(jpeg);
        }

        let avg_ms = total_time.as_secs_f64() * 1000.0 / iterations as f64;
        let mp_per_sec = pixels as f64 / (avg_ms / 1000.0) / 1_000_000.0;

        println!(
            "{:20} {:7.1} ms  {:6} KB  {:.1} MP/s",
            name,
            avg_ms,
            output_size / 1024,
            mp_per_sec
        );
    }
}
