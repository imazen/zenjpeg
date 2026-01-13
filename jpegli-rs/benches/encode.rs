//! Encoding benchmarks for jpegli.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use jpegli::{JpegEncoder, PixelFormat, Quality};

fn create_test_image(width: usize, height: usize) -> Vec<u8> {
    let mut data = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            data[idx] = ((x * 255) / width) as u8; // R
            data[idx + 1] = ((y * 255) / height) as u8; // G
            data[idx + 2] = 128; // B
        }
    }
    data
}

fn bench_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode");

    for size in [64, 256, 512, 1024, 2048] {
        let data = create_test_image(size, size);

        group.bench_with_input(
            BenchmarkId::new("rgb", format!("{}x{}", size, size)),
            &data,
            |b, data| {
                b.iter(|| {
                    let encoder = JpegEncoder::new(width, height)
                        .pixel_format(PixelFormat::Rgb)
                        .quality(Quality::from_quality(90.0));
                    encoder.encode(black_box(data))
                });
            },
        );
    }

    group.finish();
}

fn bench_quality_levels(c: &mut Criterion) {
    let mut group = c.benchmark_group("quality");

    let data = create_test_image(512, 512);

    for quality in [50, 75, 90, 95] {
        group.bench_with_input(BenchmarkId::new("q", quality), &data, |b, data| {
            b.iter(|| {
                let encoder = JpegEncoder::new(width, height)
                    .pixel_format(PixelFormat::Rgb)
                    .quality(Quality::from_quality(quality as f32));
                encoder.encode(black_box(data))
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_encode, bench_quality_levels);
criterion_main!(benches);
