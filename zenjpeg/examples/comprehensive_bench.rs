//! Comprehensive encoder benchmark: sizes × subsampling × modes
//!
//! Tests 1K, 2K, 3K, 4K images with 4:4:4/4:2:0, progressive/baseline.
//!
//! Run with: cargo run --release --example comprehensive_bench [image.png]

use enough::Unstoppable;
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};
use png::Decoder;
use std::fs::File;
use std::io::BufReader;
use std::time::Instant;

/// Load a PNG and resize it to target dimensions using simple box filter
fn load_and_resize(path: &str, target_w: usize, target_h: usize) -> Vec<u8> {
    let file = File::open(path).expect("open file");
    let decoder = Decoder::new(BufReader::new(file));
    let mut reader = decoder.read_info().expect("read info");
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).expect("decode");

    let src_w = info.width as usize;
    let src_h = info.height as usize;
    let channels = match info.color_type {
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        _ => panic!("Unsupported color type: {:?}", info.color_type),
    };

    let mut result = vec![0u8; target_w * target_h * 3];
    for ty in 0..target_h {
        for tx in 0..target_w {
            let sx = (tx * src_w) / target_w;
            let sy = (ty * src_h) / target_h;
            let src_idx = (sy * src_w + sx) * channels;
            let dst_idx = (ty * target_w + tx) * 3;
            result[dst_idx] = buf[src_idx];
            result[dst_idx + 1] = buf[src_idx + 1];
            result[dst_idx + 2] = buf[src_idx + 2];
        }
    }
    result
}

struct BenchResult {
    time_ms: f64,
    mpps: f64,
    size_bytes: usize,
}

fn bench_encode(
    data: &[u8],
    width: usize,
    height: usize,
    subsampling: ChromaSubsampling,
    progressive: bool,
    quality: f32,
    iterations: usize,
) -> BenchResult {
    let config = EncoderConfig::ycbcr(quality, subsampling).progressive(progressive);

    // Warmup
    for _ in 0..2 {
        let mut enc = config
            .encode_from_bytes(width as u32, height as u32, PixelLayout::Rgb8Srgb)
            .unwrap();
        enc.push_packed(data, Unstoppable).unwrap();
        let _ = enc.finish();
    }

    let mut times = Vec::with_capacity(iterations);
    let mut size = 0;

    for _ in 0..iterations {
        let mut enc = config
            .encode_from_bytes(width as u32, height as u32, PixelLayout::Rgb8Srgb)
            .unwrap();

        let start = Instant::now();
        enc.push_packed(data, Unstoppable).unwrap();
        let result = enc.finish();
        times.push(start.elapsed().as_secs_f64() * 1000.0);

        if let Ok(r) = result {
            size = r.len();
        }
    }

    let avg_ms = times.iter().sum::<f64>() / iterations as f64;
    let pixels = (width * height) as f64;
    let mpps = pixels / (avg_ms / 1000.0) / 1_000_000.0;

    BenchResult {
        time_ms: avg_ms,
        mpps,
        size_bytes: size,
    }
}

fn main() {
    let source_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/lilith/work/codec-eval/codec-corpus/kodak/1.png".to_string());

    println!("=== Comprehensive Encoder Benchmark ===");
    println!("Source: {}", source_path);
    println!();

    let sizes = [
        (1024, 768, "1K"),  // 0.79M
        (1920, 1080, "2K"), // 2.07M
        (2560, 1440, "3K"), // 3.69M
        (3840, 2160, "4K"), // 8.29M
    ];

    let quality = 90.0;
    let iterations = 5;

    println!(
        "{:<6} {:>4} {:>6} {:>10} {:>8} {:>8}",
        "Size", "Sub", "Mode", "Time (ms)", "MP/s", "KB"
    );
    println!("{}", "-".repeat(52));

    for (w, h, size_name) in sizes {
        let data = load_and_resize(&source_path, w, h);

        for &(subsampling, sub_name) in &[
            (ChromaSubsampling::None, "444"),
            (ChromaSubsampling::Quarter, "420"),
        ] {
            // Baseline
            let baseline = bench_encode(&data, w, h, subsampling, false, quality, iterations);

            println!(
                "{:<6} {:>4} {:>6} {:>10.1} {:>8.1} {:>8}",
                size_name,
                sub_name,
                "base",
                baseline.time_ms,
                baseline.mpps,
                baseline.size_bytes / 1024
            );

            // Progressive
            let progressive = bench_encode(&data, w, h, subsampling, true, quality, iterations);

            println!(
                "{:<6} {:>4} {:>6} {:>10.1} {:>8.1} {:>8}",
                size_name,
                sub_name,
                "prog",
                progressive.time_ms,
                progressive.mpps,
                progressive.size_bytes / 1024
            );
        }
        println!();
    }

    println!("Quality: {:.0}", quality);
    println!("Iterations per test: {}", iterations);
}
