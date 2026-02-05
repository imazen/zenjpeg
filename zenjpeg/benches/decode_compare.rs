//! Decoder comparison benchmark: zenjpeg vs zune-jpeg.
//!
//! Compares:
//! - zune-jpeg (baseline and progressive)
//! - zenjpeg full-frame decoder (baseline and progressive)
//! - zenjpeg scanline reader (baseline only, 4:2:0 and 4:4:4)
//!
//! Run with:
//! ```sh
//! cargo bench -p zenjpeg --bench decode_compare --features decoder
//! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use enough::Unstoppable;
use std::io::Cursor;
use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, PixelLayout};
use zune_jpeg::zune_core::colorspace::ColorSpace;
use zune_jpeg::zune_core::options::DecoderOptions;
use zune_jpeg::JpegDecoder;

fn create_test_jpeg_with_subsampling(
    width: u32,
    height: u32,
    quality: f32,
    progressive: bool,
    subsampling: ChromaSubsampling,
) -> Vec<u8> {
    // Create gradient test pattern
    let mut data = vec![0u8; (width * height * 3) as usize];
    for y in 0..height as usize {
        for x in 0..width as usize {
            let idx = (y * width as usize + x) * 3;
            data[idx] = ((x * 255) / width as usize) as u8;
            data[idx + 1] = ((y * 255) / height as usize) as u8;
            data[idx + 2] = (((x + y) * 128) / (width + height) as usize) as u8;
        }
    }

    let config = EncoderConfig::ycbcr(quality, subsampling).progressive(progressive);
    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .expect("encoder creation should succeed");
    enc.push_packed(&data, Unstoppable)
        .expect("push should succeed");
    enc.finish().expect("encoding should succeed")
}

fn create_test_jpeg(width: u32, height: u32, quality: f32, progressive: bool) -> Vec<u8> {
    create_test_jpeg_with_subsampling(
        width,
        height,
        quality,
        progressive,
        ChromaSubsampling::Quarter,
    )
}

fn bench_decode_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode_compare");

    // Test sizes from small to large
    for (width, height) in [(256, 256), (512, 512), (1024, 1024), (2048, 2048)] {
        let jpeg_baseline = create_test_jpeg(width, height, 85.0, false);
        let jpeg_progressive = create_test_jpeg(width, height, 85.0, true);
        let pixels = (width * height) as u64;

        // Baseline JPEG benchmarks
        group.throughput(Throughput::Elements(pixels));

        // zune-jpeg baseline
        group.bench_with_input(
            BenchmarkId::new("zune-baseline", format!("{}x{}", width, height)),
            &jpeg_baseline,
            |b, data| {
                b.iter(|| {
                    let options =
                        DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::RGB);
                    let cursor = Cursor::new(black_box(data.as_slice()));
                    let mut decoder = JpegDecoder::new_with_options(cursor, options);
                    decoder.decode().expect("decode failed")
                });
            },
        );

        // zenjpeg baseline
        #[cfg(feature = "decoder")]
        group.bench_with_input(
            BenchmarkId::new("jpegli-baseline", format!("{}x{}", width, height)),
            &jpeg_baseline,
            |b, data| {
                b.iter(|| {
                    use zenjpeg::decode::Decoder;
                    use zenjpeg::decoder::PixelFormat;
                    let decoder = Decoder::new().output_format(PixelFormat::Rgb);
                    decoder.decode(black_box(data), Unstoppable).expect("decode failed")
                });
            },
        );

        // zenjpeg baseline fast mode (box filter + fused upsample)
        #[cfg(feature = "decoder")]
        group.bench_with_input(
            BenchmarkId::new("jpegli-baseline-fast", format!("{}x{}", width, height)),
            &jpeg_baseline,
            |b, data| {
                b.iter(|| {
                    use zenjpeg::decode::Decoder;
                    use zenjpeg::decoder::PixelFormat;
                    let decoder = Decoder::new()
                        .output_format(PixelFormat::Rgb)
                        .fancy_upsampling(false);
                    decoder.decode(black_box(data), Unstoppable).expect("decode failed")
                });
            },
        );

        // Progressive JPEG benchmarks

        // zune-jpeg progressive
        group.bench_with_input(
            BenchmarkId::new("zune-progressive", format!("{}x{}", width, height)),
            &jpeg_progressive,
            |b, data| {
                b.iter(|| {
                    let options =
                        DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::RGB);
                    let cursor = Cursor::new(black_box(data.as_slice()));
                    let mut decoder = JpegDecoder::new_with_options(cursor, options);
                    decoder.decode().expect("decode failed")
                });
            },
        );

        // zenjpeg progressive
        #[cfg(feature = "decoder")]
        group.bench_with_input(
            BenchmarkId::new("jpegli-progressive", format!("{}x{}", width, height)),
            &jpeg_progressive,
            |b, data| {
                b.iter(|| {
                    use zenjpeg::decode::Decoder;
                    use zenjpeg::decoder::PixelFormat;
                    let decoder = Decoder::new().output_format(PixelFormat::Rgb);
                    decoder.decode(black_box(data), Unstoppable).expect("decode failed")
                });
            },
        );

        // zenjpeg progressive fast mode
        #[cfg(feature = "decoder")]
        group.bench_with_input(
            BenchmarkId::new("jpegli-progressive-fast", format!("{}x{}", width, height)),
            &jpeg_progressive,
            |b, data| {
                b.iter(|| {
                    use zenjpeg::decode::Decoder;
                    use zenjpeg::decoder::PixelFormat;
                    let decoder = Decoder::new()
                        .output_format(PixelFormat::Rgb)
                        .fancy_upsampling(false);
                    decoder.decode(black_box(data), Unstoppable).expect("decode failed")
                });
            },
        );

        // zenjpeg scanline reader (baseline 4:2:0)
        #[cfg(feature = "decoder")]
        group.bench_with_input(
            BenchmarkId::new("jpegli-scanline-420", format!("{}x{}", width, height)),
            &jpeg_baseline,
            |b, data| {
                b.iter(|| {
                    use imgref::ImgRefMut;
                    use zenjpeg::decode::Decoder;
                    let decoder = Decoder::new();
                    let mut reader = decoder
                        .scanline_reader(black_box(data))
                        .expect("scanline_reader failed");
                    let w = reader.width() as usize;
                    let h = reader.height() as usize;
                    let mut pixels = vec![0u8; w * h * 3];
                    let mut rows_read = 0;
                    while rows_read < h {
                        let remaining = h - rows_read;
                        let output =
                            ImgRefMut::new(&mut pixels[rows_read * w * 3..], w * 3, remaining);
                        rows_read += reader.read_rows_rgb8(output).expect("read failed");
                    }
                    pixels
                });
            },
        );
    }

    // Separate benchmark for 4:4:4 (scanline reader fast path)
    for (width, height) in [(512, 512), (1024, 1024), (2048, 2048)] {
        let jpeg_444 =
            create_test_jpeg_with_subsampling(width, height, 85.0, false, ChromaSubsampling::None);
        let pixels = (width * height) as u64;

        group.throughput(Throughput::Elements(pixels));

        // zune-jpeg 4:4:4
        group.bench_with_input(
            BenchmarkId::new("zune-baseline-444", format!("{}x{}", width, height)),
            &jpeg_444,
            |b, data| {
                b.iter(|| {
                    let options =
                        DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::RGB);
                    let cursor = Cursor::new(black_box(data.as_slice()));
                    let mut decoder = JpegDecoder::new_with_options(cursor, options);
                    decoder.decode().expect("decode failed")
                });
            },
        );

        // zenjpeg full-frame 4:4:4
        #[cfg(feature = "decoder")]
        group.bench_with_input(
            BenchmarkId::new("jpegli-baseline-444", format!("{}x{}", width, height)),
            &jpeg_444,
            |b, data| {
                b.iter(|| {
                    use zenjpeg::decode::Decoder;
                    use zenjpeg::decoder::PixelFormat;
                    let decoder = Decoder::new().output_format(PixelFormat::Rgb);
                    decoder.decode(black_box(data), Unstoppable).expect("decode failed")
                });
            },
        );

        // zenjpeg scanline reader 4:4:4 (fast path)
        #[cfg(feature = "decoder")]
        group.bench_with_input(
            BenchmarkId::new("jpegli-scanline-444", format!("{}x{}", width, height)),
            &jpeg_444,
            |b, data| {
                b.iter(|| {
                    use imgref::ImgRefMut;
                    use zenjpeg::decode::Decoder;
                    let decoder = Decoder::new();
                    let mut reader = decoder
                        .scanline_reader(black_box(data))
                        .expect("scanline_reader failed");
                    let w = reader.width() as usize;
                    let h = reader.height() as usize;
                    let mut pixels = vec![0u8; w * h * 3];
                    let mut rows_read = 0;
                    while rows_read < h {
                        let remaining = h - rows_read;
                        let output =
                            ImgRefMut::new(&mut pixels[rows_read * w * 3..], w * 3, remaining);
                        rows_read += reader.read_rows_rgb8(output).expect("read failed");
                    }
                    pixels
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_decode_comparison);
criterion_main!(benches);
