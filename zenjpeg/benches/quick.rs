//! Quick benchmark for development feedback.
//!
//! Fast, focused benchmark for iterating during development.
//! Tests core encoding path with minimal overhead.
//!
//! # Usage
//!
//! ```bash
//! # Quick check during development (~10 seconds)
//! cargo bench --bench quick
//!
//! # Compare against baseline
//! cargo bench --bench quick -- --baseline main
//! ```

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use enough::Unstoppable;
use std::time::Duration;
use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, PixelLayout, XybSubsampling};

fn generate_test_image(width: usize, height: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            let fx = x as f64 / width as f64;
            let fy = y as f64 / height as f64;
            rgb[idx] = ((fx * 255.0) + (fx * fy * 50.0).sin() * 30.0).clamp(0.0, 255.0) as u8;
            rgb[idx + 1] = ((fy * 255.0) + (fx * fy * 100.0).cos() * 40.0).clamp(0.0, 255.0) as u8;
            rgb[idx + 2] = (128.0 + ((fx + fy) * 50.0).sin() * 50.0).clamp(0.0, 255.0) as u8;
        }
    }
    rgb
}

fn quick_bench(c: &mut Criterion) {
    const WIDTH: u32 = 512;
    const HEIGHT: u32 = 512;
    let image = generate_test_image(WIDTH as usize, HEIGHT as usize);
    let pixels = WIDTH as u64 * HEIGHT as u64;

    let mut group = c.benchmark_group("quick");
    group.throughput(Throughput::Elements(pixels));
    group.warm_up_time(Duration::from_millis(200));
    group.measurement_time(Duration::from_secs(1));
    group.sample_size(50);

    // Core path: progressive + optimized huffman + 420 (most common)
    group.bench_function("prog-opt-420", |b| {
        b.iter(|| {
            let config = EncoderConfig::ycbcr(90.0, ChromaSubsampling::Quarter)
                .progressive(true)
                .optimize_huffman(true);
            let mut enc = config
                .encode_from_bytes(WIDTH, HEIGHT, PixelLayout::Rgb8Srgb)
                .unwrap();
            enc.push_packed(black_box(&image), Unstoppable).unwrap();
            enc.finish()
        });
    });

    // Baseline (simpler path)
    group.bench_function("base-opt-420", |b| {
        b.iter(|| {
            let config = EncoderConfig::ycbcr(90.0, ChromaSubsampling::Quarter)
                .progressive(false)
                .optimize_huffman(true);
            let mut enc = config
                .encode_from_bytes(WIDTH, HEIGHT, PixelLayout::Rgb8Srgb)
                .unwrap();
            enc.push_packed(black_box(&image), Unstoppable).unwrap();
            enc.finish()
        });
    });

    // 444 subsampling (no chroma downsampling)
    group.bench_function("prog-opt-444", |b| {
        b.iter(|| {
            let config = EncoderConfig::ycbcr(90.0, ChromaSubsampling::None)
                .progressive(true)
                .optimize_huffman(true);
            let mut enc = config
                .encode_from_bytes(WIDTH, HEIGHT, PixelLayout::Rgb8Srgb)
                .unwrap();
            enc.push_packed(black_box(&image), Unstoppable).unwrap();
            enc.finish()
        });
    });

    // XYB color space
    group.bench_function("prog-opt-444-xyb", |b| {
        b.iter(|| {
            let config = EncoderConfig::xyb(90.0, XybSubsampling::Full)
                .progressive(true)
                .optimize_huffman(true);
            let mut enc = config
                .encode_from_bytes(WIDTH, HEIGHT, PixelLayout::Rgb8Srgb)
                .unwrap();
            enc.push_packed(black_box(&image), Unstoppable).unwrap();
            enc.finish()
        });
    });

    group.finish();
}

criterion_group!(benches, quick_bench);
criterion_main!(benches);
