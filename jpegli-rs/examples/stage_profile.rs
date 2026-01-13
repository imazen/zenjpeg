//! Profile individual encoding stages

use std::hint::black_box;
use std::time::Instant;

use jpegli::types::{JpegMode, PixelFormat, Subsampling};
use jpegli::{Quality, JpegEncoder};

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

    for (name, mode, subsampling, optimize) in [
        (
            "Baseline/Fixed/444",
            JpegMode::Baseline,
            Subsampling::S444,
            false,
        ),
        (
            "Baseline/Opt/444",
            JpegMode::Baseline,
            Subsampling::S444,
            true,
        ),
        (
            "Baseline/Opt/420",
            JpegMode::Baseline,
            Subsampling::S420,
            true,
        ),
        (
            "Progressive/Opt/444",
            JpegMode::Progressive,
            Subsampling::S444,
            true,
        ),
    ] {
        let mut total_time = std::time::Duration::ZERO;
        let mut output_size = 0;

        for _ in 0..iterations {
            let start = Instant::now();

            let jpeg = JpegEncoder::new(width, height)
                .pixel_format(PixelFormat::Rgb)
                .quality(Quality::from_quality(85.0))
                .mode(mode)
                .subsampling(subsampling)
                .optimize_huffman(optimize)
                .encode(&rgb)
                .expect("encode failed");

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
