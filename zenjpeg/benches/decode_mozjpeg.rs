//! Decoder speed comparison: zenjpeg vs mozjpeg (libjpeg-turbo with NASM SIMD).
//!
//! This is a SEPARATE binary from decode_compare to avoid symbol conflicts
//! between mozjpeg-sys and jpegli-internals-sys (both provide jpeg_* symbols).
//!
//! Run with:
//! ```sh
//! cargo bench -p zenjpeg --bench decode_mozjpeg --features decoder
//! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use enough::Unstoppable;
use std::io::Cursor;
use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, PixelLayout};
use zune_jpeg::zune_core::colorspace::ColorSpace;
use zune_jpeg::zune_core::options::DecoderOptions;
use zune_jpeg::JpegDecoder;

/// Decode JPEG data using mozjpeg (libjpeg-turbo with NASM SIMD).
/// Returns RGB pixel data.
unsafe fn decode_with_mozjpeg(data: &[u8]) -> Vec<u8> {
    use mozjpeg_sys::*;
    use std::mem;

    let mut err: jpeg_error_mgr = mem::zeroed();
    jpeg_std_error(&mut err);

    let mut cinfo: jpeg_decompress_struct = mem::zeroed();
    cinfo.common.err = &mut err;
    jpeg_create_decompress(&mut cinfo);

    // Set memory source
    jpeg_mem_src(&mut cinfo, data.as_ptr(), data.len() as _);

    // Read header
    jpeg_read_header(&mut cinfo, true as boolean);

    // Request RGB output
    cinfo.out_color_space = J_COLOR_SPACE::JCS_RGB;

    // Start decompression
    jpeg_start_decompress(&mut cinfo);

    let width = cinfo.output_width as usize;
    let height = cinfo.output_height as usize;
    let components = cinfo.output_components as usize;
    let row_stride = width * components;

    let mut output = vec![0u8; height * row_stride];

    // Read scanlines
    while (cinfo.output_scanline as usize) < height {
        let offset = cinfo.output_scanline as usize * row_stride;
        let mut row_ptr = output[offset..].as_mut_ptr();
        jpeg_read_scanlines(&mut cinfo, &mut row_ptr, 1);
    }

    jpeg_finish_decompress(&mut cinfo);
    jpeg_destroy_decompress(&mut cinfo);

    output
}

fn create_test_jpeg(
    width: u32,
    height: u32,
    quality: f32,
    progressive: bool,
    subsampling: ChromaSubsampling,
) -> Vec<u8> {
    // Create noise+patches test pattern (not smooth gradients)
    let mut data = vec![0u8; (width * height * 3) as usize];
    let mut rng: u32 = 0xDEADBEEF;
    for y in 0..height as usize {
        for x in 0..width as usize {
            let idx = (y * width as usize + x) * 3;
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            let noise = ((rng >> 16) & 0xFF) as u8;
            let patch_x = (x / 64) & 3;
            let patch_y = (y / 64) & 3;
            let base = ((patch_x * 64 + patch_y * 32) & 255) as u8;
            data[idx] = base.wrapping_add(noise >> 2);
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            data[idx + 1] = base.wrapping_add(((rng >> 16) & 0x3F) as u8);
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            data[idx + 2] = (255 - base).wrapping_add(((rng >> 16) & 0x1F) as u8);
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

fn bench_decode_mozjpeg(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode_mozjpeg");

    for (width, height) in [
        (256, 256),
        (512, 512),
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
    ] {
        let jpeg_baseline =
            create_test_jpeg(width, height, 85.0, false, ChromaSubsampling::Quarter);
        let pixels = (width * height) as u64;
        let size_label = format!("{}x{}", width, height);

        group.throughput(Throughput::Elements(pixels));

        // mozjpeg (libjpeg-turbo + NASM SIMD) baseline
        group.bench_with_input(
            BenchmarkId::new("mozjpeg-baseline", &size_label),
            &jpeg_baseline,
            |b, data| {
                b.iter(|| unsafe { decode_with_mozjpeg(black_box(data)) });
            },
        );

        // zune-jpeg baseline (fastest pure Rust reference)
        group.bench_with_input(
            BenchmarkId::new("zune-baseline", &size_label),
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

        // zenjpeg buffered decoder baseline
        group.bench_with_input(
            BenchmarkId::new("zenjpeg-baseline", &size_label),
            &jpeg_baseline,
            |b, data| {
                b.iter(|| {
                    use zenjpeg::decode::Decoder;
                    use zenjpeg::decoder::PixelFormat;
                    let decoder = Decoder::new().output_format(PixelFormat::Rgb);
                    decoder
                        .decode(black_box(data), Unstoppable)
                        .expect("decode failed")
                });
            },
        );

        // zenjpeg fast mode (box filter)
        group.bench_with_input(
            BenchmarkId::new("zenjpeg-boxfilter", &size_label),
            &jpeg_baseline,
            |b, data| {
                b.iter(|| {
                    use zenjpeg::decode::Decoder;
                    use zenjpeg::decoder::PixelFormat;
                    let decoder = Decoder::new()
                        .output_format(PixelFormat::Rgb)
                        .fancy_upsampling(false);
                    decoder
                        .decode(black_box(data), Unstoppable)
                        .expect("decode failed")
                });
            },
        );

        // zenjpeg scanline reader
        group.bench_with_input(
            BenchmarkId::new("zenjpeg-scanline", &size_label),
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

    group.finish();
}

criterion_group!(benches, bench_decode_mozjpeg);
criterion_main!(benches);
