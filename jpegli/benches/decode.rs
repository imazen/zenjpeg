//! Decoding benchmarks for jpegli.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use jpegli::{Decoder, Encoder, PixelFormat, Quality};

fn create_test_jpeg(width: u32, height: u32, quality: f32) -> Vec<u8> {
    let mut data = vec![0u8; (width * height * 3) as usize];
    for y in 0..height as usize {
        for x in 0..width as usize {
            let idx = (y * width as usize + x) * 3;
            data[idx] = ((x * 255) / width as usize) as u8;
            data[idx + 1] = ((y * 255) / height as usize) as u8;
            data[idx + 2] = 128;
        }
    }

    let encoder = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Rgb)
        .quality(Quality::from_quality(quality));

    encoder.encode(&data).expect("encoding should succeed")
}

fn bench_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode");

    for size in [64, 256, 512] {
        let jpeg_data = create_test_jpeg(size, size, 90.0);

        group.bench_with_input(
            BenchmarkId::new("rgb", format!("{}x{}", size, size)),
            &jpeg_data,
            |b, data| {
                b.iter(|| {
                    let decoder = Decoder::new()
                        .output_format(PixelFormat::Rgb);
                    decoder.decode(black_box(data))
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_decode);
criterion_main!(benches);
