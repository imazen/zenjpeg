//! Encoding benchmarks for jpegli.

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout, Unstoppable};

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
                    let config = EncoderConfig::ycbcr(90, ChromaSubsampling::Quarter);
                    let mut enc = config
                        .encode_from_bytes(size as u32, size as u32, PixelLayout::Rgb8Srgb)
                        .unwrap();
                    enc.push_packed(black_box(data), Unstoppable).unwrap();
                    enc.finish()
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
                let config = EncoderConfig::ycbcr(quality, ChromaSubsampling::Quarter);
                let mut enc = config
                    .encode_from_bytes(512, 512, PixelLayout::Rgb8Srgb)
                    .unwrap();
                enc.push_packed(black_box(data), Unstoppable).unwrap();
                enc.finish()
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_encode, bench_quality_levels);
criterion_main!(benches);
