//! Profiling benchmark for YCbCr encoding paths.
//!
//! Designed for use with flamegraph/samply. Uses larger images and longer
//! runs to get meaningful profiles of the encoding hot paths.
//!
//! # Usage
//!
//! ```bash
//! # Run benchmark normally
//! cargo bench --bench profile_ycbcr
//!
//! # Profile with flamegraph (10 second profile)
//! cargo flamegraph --bench profile_ycbcr -- --bench "ycbcr-420" --profile-time 10
//!
//! # Profile with samply
//! cargo build --profile profiling --bench profile_ycbcr
//! samply record ./target/profiling/deps/profile_ycbcr-* --bench "ycbcr-420" --profile-time 10
//! ```

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use enough::Unstoppable;
use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, PixelLayout};
use std::time::Duration;

/// Generate a test image with realistic-ish content for profiling.
/// Uses gradients and noise patterns that exercise color conversion paths.
fn generate_test_image(width: usize, height: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            let fx = x as f64 / width as f64;
            let fy = y as f64 / height as f64;

            // More complex pattern to stress color conversion
            let r = ((fx * 200.0) + (fx * fy * 80.0).sin() * 55.0).clamp(0.0, 255.0);
            let g = ((fy * 200.0) + ((fx + fy) * 60.0).cos() * 55.0).clamp(0.0, 255.0);
            let b =
                (100.0 + ((fx - fy).abs() * 100.0).sin() * 80.0 + (fx * fy * 200.0).cos() * 30.0)
                    .clamp(0.0, 255.0);

            rgb[idx] = r as u8;
            rgb[idx + 1] = g as u8;
            rgb[idx + 2] = b as u8;
        }
    }
    rgb
}

fn profile_ycbcr_bench(c: &mut Criterion) {
    // 1024x1024 for meaningful profile data
    let width: u32 = 1024;
    let height: u32 = 1024;
    let image = generate_test_image(width as usize, height as usize);
    let pixels = u64::from(width) * u64::from(height);

    let mut group = c.benchmark_group("profile");

    // Longer runs for profiling
    group.throughput(Throughput::Elements(pixels));
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(30);

    // YCbCr 4:2:0 - most common, has chroma downsampling
    group.bench_function("ycbcr-420", |b| {
        b.iter(|| {
            let config = EncoderConfig::ycbcr(90.0, ChromaSubsampling::Quarter)
                .progressive(true)
                .optimize_huffman(true);
            let mut enc = config
                .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
                .unwrap();
            enc.push_packed(black_box(&image), Unstoppable).unwrap();
            enc.finish()
        });
    });

    // YCbCr 4:4:4 - no chroma downsampling
    group.bench_function("ycbcr-444", |b| {
        b.iter(|| {
            let config = EncoderConfig::ycbcr(90.0, ChromaSubsampling::None)
                .progressive(true)
                .optimize_huffman(true);
            let mut enc = config
                .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
                .unwrap();
            enc.push_packed(black_box(&image), Unstoppable).unwrap();
            enc.finish()
        });
    });

    // Baseline mode variants (simpler scan structure)
    group.bench_function("ycbcr-420-baseline", |b| {
        b.iter(|| {
            let config = EncoderConfig::ycbcr(90.0, ChromaSubsampling::Quarter)
                .progressive(false)
                .optimize_huffman(true);
            let mut enc = config
                .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
                .unwrap();
            enc.push_packed(black_box(&image), Unstoppable).unwrap();
            enc.finish()
        });
    });

    group.bench_function("ycbcr-444-baseline", |b| {
        b.iter(|| {
            let config = EncoderConfig::ycbcr(90.0, ChromaSubsampling::None)
                .progressive(false)
                .optimize_huffman(true);
            let mut enc = config
                .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
                .unwrap();
            enc.push_packed(black_box(&image), Unstoppable).unwrap();
            enc.finish()
        });
    });

    group.finish();
}

criterion_group!(benches, profile_ycbcr_bench);
criterion_main!(benches);
