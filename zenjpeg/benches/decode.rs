//! Decoding benchmarks for zenjpeg.

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use enough::Unstoppable;
use imgref::ImgRefMut;
use zenjpeg::decode::Decoder;
use zenjpeg::decoder::PixelFormat;
use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, PixelLayout};

fn create_test_jpeg(width: u32, height: u32, quality: f32, progressive: bool) -> Vec<u8> {
    let mut data = vec![0u8; (width * height * 3) as usize];
    for y in 0..height as usize {
        for x in 0..width as usize {
            let idx = (y * width as usize + x) * 3;
            data[idx] = ((x * 255) / width as usize) as u8;
            data[idx + 1] = ((y * 255) / height as usize) as u8;
            data[idx + 2] = 128;
        }
    }

    let mut config = EncoderConfig::ycbcr(quality, ChromaSubsampling::Quarter);
    if progressive {
        config = config.progressive(true);
    }
    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .expect("encoder creation should succeed");
    enc.push_packed(&data, Unstoppable)
        .expect("push should succeed");
    enc.finish().expect("encoding should succeed")
}

fn bench_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode");

    // Standard sizes
    for size in [512, 1024, 2048] {
        let jpeg_data = create_test_jpeg(size, size, 90.0, false);

        group.bench_with_input(
            BenchmarkId::new("sequential", format!("{}x{}", size, size)),
            &jpeg_data,
            |b, data| {
                let decoder = Decoder::new().output_format(PixelFormat::Rgb);
                b.iter(|| decoder.decode(black_box(data), Unstoppable));
            },
        );
    }

    // Progressive
    for size in [512, 1024, 2048] {
        let jpeg_data = create_test_jpeg(size, size, 90.0, true);

        group.bench_with_input(
            BenchmarkId::new("progressive", format!("{}x{}", size, size)),
            &jpeg_data,
            |b, data| {
                let decoder = Decoder::new().output_format(PixelFormat::Rgb);
                b.iter(|| decoder.decode(black_box(data), Unstoppable));
            },
        );
    }

    // 4K
    let jpeg_4k_seq = create_test_jpeg(3840, 2160, 90.0, false);
    let jpeg_4k_prog = create_test_jpeg(3840, 2160, 90.0, true);

    group.bench_with_input(
        BenchmarkId::new("sequential", "4K"),
        &jpeg_4k_seq,
        |b, data| {
            let decoder = Decoder::new().output_format(PixelFormat::Rgb);
            b.iter(|| decoder.decode(black_box(data), Unstoppable));
        },
    );

    group.bench_with_input(
        BenchmarkId::new("progressive", "4K"),
        &jpeg_4k_prog,
        |b, data| {
            let decoder = Decoder::new().output_format(PixelFormat::Rgb);
            b.iter(|| decoder.decode(black_box(data), Unstoppable));
        },
    );

    group.finish();
}

fn bench_scanline(c: &mut Criterion) {
    let mut group = c.benchmark_group("scanline");

    // 4K sequential (streaming mode)
    let jpeg_4k_seq = create_test_jpeg(3840, 2160, 90.0, false);
    let width = 3840usize;
    let height = 2160usize;

    group.bench_with_input(
        BenchmarkId::new("sequential", "4K"),
        &jpeg_4k_seq,
        |b, data| {
            let decoder = Decoder::new();
            b.iter(|| {
                let mut reader = decoder.scanline_reader(black_box(data)).unwrap();
                let mut out = vec![0u8; width * height * 3];
                let mut rows = 0;
                while rows < height {
                    let remaining = height - rows;
                    let output = ImgRefMut::new(&mut out[rows * width * 3..], width * 3, remaining);
                    rows += reader.read_rows_rgb8(output).unwrap();
                }
                out
            });
        },
    );

    // 4K progressive (buffered mode)
    let jpeg_4k_prog = create_test_jpeg(3840, 2160, 90.0, true);

    group.bench_with_input(
        BenchmarkId::new("progressive", "4K"),
        &jpeg_4k_prog,
        |b, data| {
            let decoder = Decoder::new();
            b.iter(|| {
                let mut reader = decoder.scanline_reader(black_box(data)).unwrap();
                let mut out = vec![0u8; width * height * 3];
                let mut rows = 0;
                while rows < height {
                    let remaining = height - rows;
                    let output = ImgRefMut::new(&mut out[rows * width * 3..], width * 3, remaining);
                    rows += reader.read_rows_rgb8(output).unwrap();
                }
                out
            });
        },
    );

    // 2K sizes too
    let jpeg_2k_seq = create_test_jpeg(2048, 2048, 90.0, false);
    let jpeg_2k_prog = create_test_jpeg(2048, 2048, 90.0, true);
    let w2k = 2048usize;
    let h2k = 2048usize;

    group.bench_with_input(
        BenchmarkId::new("sequential", "2K"),
        &jpeg_2k_seq,
        |b, data| {
            let decoder = Decoder::new();
            b.iter(|| {
                let mut reader = decoder.scanline_reader(black_box(data)).unwrap();
                let mut out = vec![0u8; w2k * h2k * 3];
                let mut rows = 0;
                while rows < h2k {
                    let remaining = h2k - rows;
                    let output = ImgRefMut::new(&mut out[rows * w2k * 3..], w2k * 3, remaining);
                    rows += reader.read_rows_rgb8(output).unwrap();
                }
                out
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("progressive", "2K"),
        &jpeg_2k_prog,
        |b, data| {
            let decoder = Decoder::new();
            b.iter(|| {
                let mut reader = decoder.scanline_reader(black_box(data)).unwrap();
                let mut out = vec![0u8; w2k * h2k * 3];
                let mut rows = 0;
                while rows < h2k {
                    let remaining = h2k - rows;
                    let output = ImgRefMut::new(&mut out[rows * w2k * 3..], w2k * 3, remaining);
                    rows += reader.read_rows_rgb8(output).unwrap();
                }
                out
            });
        },
    );

    group.finish();
}

criterion_group!(benches, bench_decode, bench_scanline);
criterion_main!(benches);
