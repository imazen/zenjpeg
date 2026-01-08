//! Direct library comparison: jpegli-rs vs mozjpeg crate.
//!
//! Both use direct Rust library calls - no subprocess overhead.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use jpegli::{Encoder, PixelFormat, Quality};
use mozjpeg::{ColorSpace, Compress, ScanMode};

fn create_test_image(width: usize, height: usize) -> Vec<u8> {
    let mut data = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            // Gradient + pattern for realistic compression behavior
            data[idx] = ((x * 255) / width) as u8;
            data[idx + 1] = ((y * 255) / height) as u8;
            data[idx + 2] = (((x + y) * 128) / (width + height)) as u8;
        }
    }
    data
}

fn encode_jpegli(data: &[u8], width: usize, height: usize, quality: f32) -> Vec<u8> {
    Encoder::new()
        .width(width as u32)
        .height(height as u32)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(Quality::from_quality(quality))
        .encode(data)
        .unwrap()
}

fn encode_mozjpeg(data: &[u8], width: usize, height: usize, quality: f32) -> Vec<u8> {
    let mut comp = Compress::new(ColorSpace::JCS_RGB);
    comp.set_size(width, height);
    comp.set_quality(quality);
    comp.set_scan_optimization_mode(ScanMode::AllComponentsTogether);

    let mut comp = comp.start_compress(Vec::new()).unwrap();
    comp.write_scanlines(data).unwrap();
    comp.finish().unwrap()
}

fn bench_vs_mozjpeg(c: &mut Criterion) {
    let mut group = c.benchmark_group("jpegli_vs_mozjpeg");

    for size in [512, 1024, 2048] {
        let data = create_test_image(size, size);
        let quality = 90.0;

        group.bench_with_input(
            BenchmarkId::new("jpegli-rs", format!("{}x{}", size, size)),
            &data,
            |b, data| {
                b.iter(|| encode_jpegli(black_box(data), size, size, quality));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("mozjpeg", format!("{}x{}", size, size)),
            &data,
            |b, data| {
                b.iter(|| encode_mozjpeg(black_box(data), size, size, quality));
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_vs_mozjpeg);
criterion_main!(benches);
