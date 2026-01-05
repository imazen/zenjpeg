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

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use jpegli::types::{JpegMode, PixelFormat, Subsampling};
use jpegli::{Encoder, Quality};
use std::time::Duration;

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
    let image = generate_test_image(512, 512);
    let pixels = 512u64 * 512;

    let mut group = c.benchmark_group("quick");
    group.throughput(Throughput::Elements(pixels));
    group.warm_up_time(Duration::from_millis(200));
    group.measurement_time(Duration::from_secs(1));
    group.sample_size(50);

    // Core path: progressive + optimized huffman + 420 (most common)
    group.bench_function("prog-opt-420", |b| {
        b.iter(|| {
            Encoder::new()
                .width(512)
                .height(512)
                .pixel_format(PixelFormat::Rgb)
                .jpegli_quality(Quality::from_quality(90.0))
                .mode(JpegMode::Progressive)
                .optimize_huffman(true)
                .subsampling(Subsampling::S420)
                .encode(black_box(&image))
        });
    });

    // Baseline (simpler path)
    group.bench_function("base-opt-420", |b| {
        b.iter(|| {
            Encoder::new()
                .width(512)
                .height(512)
                .pixel_format(PixelFormat::Rgb)
                .jpegli_quality(Quality::from_quality(90.0))
                .mode(JpegMode::Baseline)
                .optimize_huffman(true)
                .subsampling(Subsampling::S420)
                .encode(black_box(&image))
        });
    });

    // 444 subsampling (no chroma downsampling)
    group.bench_function("prog-opt-444", |b| {
        b.iter(|| {
            Encoder::new()
                .width(512)
                .height(512)
                .pixel_format(PixelFormat::Rgb)
                .jpegli_quality(Quality::from_quality(90.0))
                .mode(JpegMode::Progressive)
                .optimize_huffman(true)
                .subsampling(Subsampling::S444)
                .encode(black_box(&image))
        });
    });

    // XYB color space
    group.bench_function("prog-opt-444-xyb", |b| {
        b.iter(|| {
            Encoder::new()
                .width(512)
                .height(512)
                .pixel_format(PixelFormat::Rgb)
                .jpegli_quality(Quality::from_quality(90.0))
                .mode(JpegMode::Progressive)
                .optimize_huffman(true)
                .subsampling(Subsampling::S444)
                .use_xyb(true)
                .encode(black_box(&image))
        });
    });

    group.finish();
}

criterion_group!(benches, quick_bench);
criterion_main!(benches);
