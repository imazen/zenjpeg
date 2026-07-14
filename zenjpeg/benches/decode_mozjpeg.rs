//! Decoder speed comparison: zenjpeg vs mozjpeg (libjpeg-turbo with NASM SIMD).
//!
//! This is a SEPARATE binary from decode_compare to avoid symbol conflicts
//! between mozjpeg-sys and jpegli-internals-sys (both provide jpeg_* symbols).
//!
//! Run with:
//! ```sh
//! cargo bench -p zenjpeg --bench decode_mozjpeg
//! ```

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use enough::Unstoppable;
use std::hint::black_box;
use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, PixelLayout};

/// Decode JPEG data using mozjpeg (libjpeg-turbo with NASM SIMD).
/// Returns RGB pixel data.
unsafe fn decode_with_mozjpeg(data: &[u8]) -> Vec<u8> {
    unsafe {
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
}

fn create_test_jpeg(
    width: u32,
    height: u32,
    quality: f32,
    progressive: bool,
    subsampling: ChromaSubsampling,
) -> Vec<u8> {
    // Deterministic noise+patches pattern — MUST match decode_compare.rs exactly.
    let mut data = vec![0u8; (width * height * 3) as usize];
    for y in 0..height as usize {
        for x in 0..width as usize {
            let idx = (y * width as usize + x) * 3;
            let bx = (x / 8) as u32;
            let by = (y / 8) as u32;
            let block_hash = bx
                .wrapping_mul(2654435761)
                .wrapping_add(by.wrapping_mul(40503));
            let block_type = block_hash % 4;
            let px = x as u32;
            let py = y as u32;
            let mut h = px
                .wrapping_mul(374761393)
                .wrapping_add(py.wrapping_mul(668265263));
            h = (h ^ (h >> 13)).wrapping_mul(1274126177);
            let noise = (h >> 24) as u8;
            match block_type {
                0 => {
                    let bias = ((bx.wrapping_mul(17) ^ by.wrapping_mul(31)) & 0xFF) as u8;
                    data[idx] = bias.wrapping_add(noise >> 2);
                    data[idx + 1] = bias.wrapping_add(noise >> 1);
                    data[idx + 2] = bias.wrapping_add(noise >> 3);
                }
                1 => {
                    data[idx] = ((x * 255) / width as usize) as u8;
                    data[idx + 1] = ((y * 255) / height as usize) as u8;
                    data[idx + 2] = noise >> 2;
                }
                2 => {
                    let edge = if (x % 8 < 4) ^ (y % 8 < 4) {
                        200u8
                    } else {
                        55u8
                    };
                    data[idx] = edge;
                    data[idx + 1] = edge.wrapping_add(noise >> 4);
                    data[idx + 2] = 255 - edge;
                }
                _ => {
                    data[idx] = noise;
                    data[idx + 1] = noise.wrapping_mul(3);
                    data[idx + 2] = noise.wrapping_mul(7);
                }
            }
        }
    }

    let mut config = EncoderConfig::ycbcr(quality, subsampling).progressive(progressive);
    if progressive {
        config = config.restart_mcu_rows(0);
    }
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
        let jpeg_progressive =
            create_test_jpeg(width, height, 85.0, true, ChromaSubsampling::Quarter);
        let pixels = (width * height) as u64;
        let size_label = format!("{}x{}", width, height);

        group.throughput(Throughput::Elements(pixels));

        // === Baseline 4:2:0 ===

        // mozjpeg (libjpeg-turbo + NASM SIMD) baseline
        group.bench_with_input(
            BenchmarkId::new("mozjpeg-baseline", &size_label),
            &jpeg_baseline,
            |b, data| {
                b.iter(|| unsafe { decode_with_mozjpeg(black_box(data)) });
            },
        );

        // zenjpeg baseline
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
                    use zenjpeg::decode::{ChromaUpsampling, Decoder};
                    use zenjpeg::decoder::PixelFormat;
                    let decoder = Decoder::new()
                        .output_format(PixelFormat::Rgb)
                        .chroma_upsampling(ChromaUpsampling::NearestNeighbor);
                    decoder
                        .decode(black_box(data), Unstoppable)
                        .expect("decode failed")
                });
            },
        );

        // === Progressive 4:2:0 (no DRI — zune-jpeg bug with DRI) ===

        // mozjpeg progressive
        group.bench_with_input(
            BenchmarkId::new("mozjpeg-progressive", &size_label),
            &jpeg_progressive,
            |b, data| {
                b.iter(|| unsafe { decode_with_mozjpeg(black_box(data)) });
            },
        );

        // zenjpeg progressive
        group.bench_with_input(
            BenchmarkId::new("zenjpeg-progressive", &size_label),
            &jpeg_progressive,
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
    }

    group.finish();
}

criterion_group!(benches, bench_decode_mozjpeg);
criterion_main!(benches);
