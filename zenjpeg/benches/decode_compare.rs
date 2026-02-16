//! Decoder comparison benchmark: zenjpeg vs zune-jpeg vs C++ jpegli.
//!
//! Compares:
//! - zune-jpeg (baseline and progressive)
//! - zenjpeg full-frame decoder (baseline and progressive)
//! - zenjpeg scanline reader (baseline only, 4:2:0 and 4:4:4)
//! - C++ jpegli decoder via FFI (baseline and progressive)
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

/// Decode JPEG data using C++ jpegli FFI (libjpeg-compatible API).
/// Returns RGB pixel data.
#[cfg(feature = "decoder")]
unsafe fn decode_with_cjpegli(data: &[u8]) -> Vec<u8> {
    use jpegli_internals_sys::*;
    use std::mem::MaybeUninit;

    // Set up error handler
    let mut err: MaybeUninit<jpeg_error_mgr> = MaybeUninit::zeroed();
    jpeg_std_error(err.as_mut_ptr());
    let mut err = err.assume_init();

    // Create decompressor
    let mut cinfo: MaybeUninit<jpeg_decompress_struct> = MaybeUninit::zeroed();
    let cinfo_ptr = cinfo.as_mut_ptr();
    (*cinfo_ptr).err = &mut err;
    jpeg_CreateDecompress(
        cinfo_ptr,
        JPEG_LIB_VERSION as i32,
        std::mem::size_of::<jpeg_decompress_struct>(),
    );
    let cinfo_ptr = cinfo.as_mut_ptr();

    // Set memory source
    jpeg_mem_src(cinfo_ptr, data.as_ptr(), data.len() as _);

    // Read header
    jpeg_read_header(cinfo_ptr, 1);

    // Request RGB output
    (*cinfo_ptr).out_color_space = JCS_EXT_RGB as u32;

    // Start decompression
    jpeg_start_decompress(cinfo_ptr);

    let width = (*cinfo_ptr).output_width as usize;
    let height = (*cinfo_ptr).output_height as usize;
    let components = (*cinfo_ptr).output_components as usize;
    let row_stride = width * components;

    let mut output = vec![0u8; height * row_stride];

    // Read scanlines in batches for efficiency
    let batch = 8u32;
    let mut row_ptrs = [std::ptr::null_mut::<u8>(); 8];
    #[allow(clippy::while_immutable_condition)] // output_scanline mutated by FFI call
    while ((*cinfo_ptr).output_scanline as usize) < height {
        let start = (*cinfo_ptr).output_scanline as usize;
        let remaining = height - start;
        let count = remaining.min(batch as usize);
        for i in 0..count {
            row_ptrs[i] = output[(start + i) * row_stride..].as_mut_ptr();
        }
        jpeg_read_scanlines(cinfo_ptr, row_ptrs.as_mut_ptr(), count as u32);
    }

    jpeg_finish_decompress(cinfo_ptr);
    jpeg_destroy_decompress(cinfo_ptr);

    output
}

fn create_test_jpeg_with_subsampling(
    width: u32,
    height: u32,
    quality: f32,
    progressive: bool,
    subsampling: ChromaSubsampling,
) -> Vec<u8> {
    // Deterministic noise+patches pattern that produces realistic DCT
    // coefficient distributions. Smooth gradients are degenerate (mostly
    // DC-only blocks) and don't represent real photographic content.
    let mut data = vec![0u8; (width * height * 3) as usize];
    for y in 0..height as usize {
        for x in 0..width as usize {
            let idx = (y * width as usize + x) * 3;
            // Block-level variation (8x8 aligned patches with different content)
            let bx = (x / 8) as u32;
            let by = (y / 8) as u32;
            let block_hash = bx
                .wrapping_mul(2654435761)
                .wrapping_add(by.wrapping_mul(40503));
            let block_type = block_hash % 4;

            // Pixel-level deterministic noise (xorshift-inspired)
            let px = x as u32;
            let py = y as u32;
            let mut h = px
                .wrapping_mul(374761393)
                .wrapping_add(py.wrapping_mul(668265263));
            h = (h ^ (h >> 13)).wrapping_mul(1274126177);
            let noise = (h >> 24) as u8;

            match block_type {
                0 => {
                    // Textured patch: noise with local bias
                    let bias = ((bx.wrapping_mul(17) ^ by.wrapping_mul(31)) & 0xFF) as u8;
                    data[idx] = bias.wrapping_add(noise >> 2);
                    data[idx + 1] = bias.wrapping_add(noise >> 1);
                    data[idx + 2] = bias.wrapping_add(noise >> 3);
                }
                1 => {
                    // Gradient region (some DC-heavy blocks are realistic)
                    data[idx] = ((x * 255) / width as usize) as u8;
                    data[idx + 1] = ((y * 255) / height as usize) as u8;
                    data[idx + 2] = noise >> 2;
                }
                2 => {
                    // Sharp edges: checkerboard within block
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
                    // High-frequency noise (exercises many AC coefficients)
                    data[idx] = noise;
                    data[idx + 1] = noise.wrapping_mul(3);
                    data[idx + 2] = noise.wrapping_mul(7);
                }
            }
        }
    }

    let mut config = EncoderConfig::ycbcr(quality, subsampling).progressive(progressive);
    if progressive {
        // Disable restart markers for progressive JPEGs in benchmarks.
        // zune-jpeg 0.5.12 has a bug where it silently skips AC refinement
        // scans when restart markers are present, producing incorrect output
        // (max_diff=224, 99.4% of pixels wrong). Without DRI, zune produces
        // correct output matching zenjpeg and cjpegli byte-for-byte.
        config = config.restart_mcu_rows(0);
    }
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
    for (width, height) in [
        (256, 256),
        (512, 512),
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
    ] {
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
            BenchmarkId::new("zenjpeg-baseline", format!("{}x{}", width, height)),
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

        // zenjpeg baseline fast mode (box filter + fused upsample)
        #[cfg(feature = "decoder")]
        group.bench_with_input(
            BenchmarkId::new("zenjpeg-baseline-fast", format!("{}x{}", width, height)),
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

        // C++ jpegli baseline
        #[cfg(feature = "decoder")]
        group.bench_with_input(
            BenchmarkId::new("cjpegli-baseline", format!("{}x{}", width, height)),
            &jpeg_baseline,
            |b, data| {
                b.iter(|| unsafe { decode_with_cjpegli(black_box(data)) });
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
            BenchmarkId::new("zenjpeg-progressive", format!("{}x{}", width, height)),
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

        // zenjpeg progressive fast mode
        #[cfg(feature = "decoder")]
        group.bench_with_input(
            BenchmarkId::new("zenjpeg-progressive-fast", format!("{}x{}", width, height)),
            &jpeg_progressive,
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

        // C++ jpegli progressive
        #[cfg(feature = "decoder")]
        group.bench_with_input(
            BenchmarkId::new("cjpegli-progressive", format!("{}x{}", width, height)),
            &jpeg_progressive,
            |b, data| {
                b.iter(|| unsafe { decode_with_cjpegli(black_box(data)) });
            },
        );

        // zenjpeg baseline parallel (requires parallel + decoder features)
        #[cfg(all(feature = "decoder", feature = "parallel"))]
        group.bench_with_input(
            BenchmarkId::new("zenjpeg-baseline-parallel", format!("{}x{}", width, height)),
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

        // zenjpeg progressive parallel (requires parallel + decoder features)
        #[cfg(all(feature = "decoder", feature = "parallel"))]
        group.bench_with_input(
            BenchmarkId::new(
                "zenjpeg-progressive-parallel",
                format!("{}x{}", width, height),
            ),
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

        // zenjpeg scanline reader (baseline 4:2:0)
        #[cfg(feature = "decoder")]
        group.bench_with_input(
            BenchmarkId::new("zenjpeg-scanline-420", format!("{}x{}", width, height)),
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

        // zenjpeg wave-parallel scanline reader (baseline 4:2:0, box filter)
        // Uses the scanline_reader() API with parallel feature — wave decode
        // activates automatically when DRI is present.
        #[cfg(all(feature = "decoder", feature = "parallel"))]
        group.bench_with_input(
            BenchmarkId::new("zenjpeg-wave-scanline-420", format!("{}x{}", width, height)),
            &jpeg_baseline,
            |b, data| {
                b.iter(|| {
                    use imgref::ImgRefMut;
                    use zenjpeg::decode::Decoder;
                    let decoder = Decoder::new().fancy_upsampling(false);
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

        // zenjpeg sequential scanline reader (baseline 4:2:0, box filter, forced sequential)
        // For comparing wave-parallel vs sequential scanline performance.
        #[cfg(all(feature = "decoder", feature = "parallel"))]
        group.bench_with_input(
            BenchmarkId::new(
                "zenjpeg-seq-scanline-box-420",
                format!("{}x{}", width, height),
            ),
            &jpeg_baseline,
            |b, data| {
                b.iter(|| {
                    use imgref::ImgRefMut;
                    use zenjpeg::decode::Decoder;
                    let decoder = Decoder::new()
                        .fancy_upsampling(false)
                        .num_threads(1); // Force sequential
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
            BenchmarkId::new("zenjpeg-baseline-444", format!("{}x{}", width, height)),
            &jpeg_444,
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

        // C++ jpegli 4:4:4
        #[cfg(feature = "decoder")]
        group.bench_with_input(
            BenchmarkId::new("cjpegli-baseline-444", format!("{}x{}", width, height)),
            &jpeg_444,
            |b, data| {
                b.iter(|| unsafe { decode_with_cjpegli(black_box(data)) });
            },
        );

        // zenjpeg scanline reader 4:4:4 (fast path)
        #[cfg(feature = "decoder")]
        group.bench_with_input(
            BenchmarkId::new("zenjpeg-scanline-444", format!("{}x{}", width, height)),
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
