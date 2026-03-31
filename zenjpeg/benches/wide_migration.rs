//! Comprehensive encode+decode benchmark for wide→magetypes migration.
//!
//! Covers every encode path (pixel formats, subsampling, progressive, trellis,
//! XYB, grayscale) and every decode path (IDCT methods, upsampling, deblock,
//! output formats, scanline, progressive) at 2k and 4k.
//!
//! Run:
//! ```bash
//! cargo bench -p zenjpeg --bench wide_migration --features "decoder,trellis"
//! ```
//!
//! Save baseline:
//! ```bash
//! cargo bench -p zenjpeg --bench wide_migration --features "decoder,trellis" -- --save-baseline=pre-migration
//! ```
//!
//! Compare after migration:
//! ```bash
//! cargo bench -p zenjpeg --bench wide_migration --features "decoder,trellis" -- --baseline=pre-migration
//! ```

use enough::Unstoppable;
use zenbench::prelude::*;
use zenjpeg::decode::Decoder;
use zenjpeg::decoder::PixelFormat;
#[cfg(feature = "trellis")]
use zenjpeg::encode::trellis::TrellisConfig;
use zenjpeg::encode::{
    ChromaSubsampling, EncoderConfig, ParallelEncoding, PixelLayout, XybSubsampling,
};

// ── Test image generation ──────────────────────────────────────────────────

/// Noise+patches test image (deterministic LCG). NOT gradients.
fn create_rgb8(width: usize, height: usize) -> Vec<u8> {
    let mut data = vec![0u8; width * height * 3];
    let mut rng: u64 = 0xDEAD_BEEF_CAFE_BABE;
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            let px = (x / 32) % 4;
            let py = (y / 32) % 4;
            for c in 0..3u8 {
                let base =
                    ((px * (60 + c as usize * 20) + py * (40 + c as usize * 30) + c as usize * 80)
                        % 256) as u8;
                rng = rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let noise = ((rng >> 33) % 40) as u8;
                data[idx + c as usize] = base.wrapping_add(noise);
            }
        }
    }
    data
}

fn create_rgba8(width: usize, height: usize) -> Vec<u8> {
    let rgb = create_rgb8(width, height);
    let mut rgba = Vec::with_capacity(width * height * 4);
    for chunk in rgb.chunks_exact(3) {
        rgba.extend_from_slice(chunk);
        rgba.push(255);
    }
    rgba
}

fn create_bgra8(width: usize, height: usize) -> Vec<u8> {
    let rgb = create_rgb8(width, height);
    let mut bgra = Vec::with_capacity(width * height * 4);
    for chunk in rgb.chunks_exact(3) {
        bgra.push(chunk[2]);
        bgra.push(chunk[1]);
        bgra.push(chunk[0]);
        bgra.push(255);
    }
    bgra
}

fn create_gray8(width: usize, height: usize) -> Vec<u8> {
    let rgb = create_rgb8(width, height);
    rgb.chunks_exact(3)
        .map(|c| ((c[0] as u16 * 77 + c[1] as u16 * 150 + c[2] as u16 * 29) >> 8) as u8)
        .collect()
}

fn create_rgb16(width: usize, height: usize) -> Vec<u16> {
    let rgb = create_rgb8(width, height);
    rgb.iter().map(|&v| (v as u16) << 8 | v as u16).collect()
}

fn create_rgbf32(width: usize, height: usize) -> Vec<f32> {
    let rgb = create_rgb8(width, height);
    rgb.iter().map(|&v| v as f32 / 255.0).collect()
}

fn encode_jpeg(rgb: &[u8], w: u32, h: u32, ss: ChromaSubsampling, progressive: bool) -> Vec<u8> {
    let mut config = EncoderConfig::ycbcr(90.0, ss);
    if progressive {
        config = config.progressive(true);
    }
    let mut enc = config
        .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
        .unwrap();
    enc.push_packed(rgb, Unstoppable).unwrap();
    enc.finish().unwrap()
}

fn encode_grayscale_jpeg(gray: &[u8], w: u32, h: u32) -> Vec<u8> {
    let config = EncoderConfig::grayscale(90.0);
    let mut enc = config
        .encode_from_bytes(w, h, PixelLayout::Gray8Srgb)
        .unwrap();
    enc.push_packed(gray, Unstoppable).unwrap();
    enc.finish().unwrap()
}

// ── Sizes ──────────────────────────────────────────────────────────────────

const S2K: (u32, u32) = (2048, 2048);
const S4K: (u32, u32) = (4096, 4096);

// ── Encode benchmarks ──────────────────────────────────────────────────────

fn bench_encode(suite: &mut Suite) {
    // Pre-generate test images (leaked for 'static)
    let rgb8_2k: &'static [u8] = Vec::leak(create_rgb8(S2K.0 as usize, S2K.1 as usize));
    let rgb8_4k: &'static [u8] = Vec::leak(create_rgb8(S4K.0 as usize, S4K.1 as usize));
    let rgba8_2k: &'static [u8] = Vec::leak(create_rgba8(S2K.0 as usize, S2K.1 as usize));
    let bgra8_2k: &'static [u8] = Vec::leak(create_bgra8(S2K.0 as usize, S2K.1 as usize));
    let gray8_2k: &'static [u8] = Vec::leak(create_gray8(S2K.0 as usize, S2K.1 as usize));
    let rgb16_2k: &'static [u16] = Vec::leak(create_rgb16(S2K.0 as usize, S2K.1 as usize));
    let rgbf32_2k: &'static [f32] = Vec::leak(create_rgbf32(S2K.0 as usize, S2K.1 as usize));

    // ── Subsampling × size × mode ─────────────────────────────────────

    let subsamplings = [
        ("444", ChromaSubsampling::None),
        ("422", ChromaSubsampling::HalfHorizontal),
        ("420", ChromaSubsampling::Quarter),
        ("440", ChromaSubsampling::HalfVertical),
    ];

    for &(ss_name, ss) in &subsamplings {
        // 2k baseline + progressive
        suite.group(&format!("enc/2k/{ss_name}"), |g| {
            g.throughput(Throughput::Bytes((S2K.0 * S2K.1 * 3) as u64));

            g.bench("base", move |b| {
                b.iter(|| {
                    let mut enc = EncoderConfig::ycbcr(90.0, ss)
                        .encode_from_bytes(S2K.0, S2K.1, PixelLayout::Rgb8Srgb)
                        .unwrap();
                    enc.push_packed(rgb8_2k, Unstoppable).unwrap();
                    enc.finish()
                })
            });

            g.bench("prog", move |b| {
                b.iter(|| {
                    let mut enc = EncoderConfig::ycbcr(90.0, ss)
                        .progressive(true)
                        .encode_from_bytes(S2K.0, S2K.1, PixelLayout::Rgb8Srgb)
                        .unwrap();
                    enc.push_packed(rgb8_2k, Unstoppable).unwrap();
                    enc.finish()
                })
            });
        });

        // 4k baseline + progressive
        suite.group(&format!("enc/4k/{ss_name}"), |g| {
            g.throughput(Throughput::Bytes((S4K.0 * S4K.1 * 3) as u64));

            g.bench("base", move |b| {
                b.iter(|| {
                    let mut enc = EncoderConfig::ycbcr(90.0, ss)
                        .encode_from_bytes(S4K.0, S4K.1, PixelLayout::Rgb8Srgb)
                        .unwrap();
                    enc.push_packed(rgb8_4k, Unstoppable).unwrap();
                    enc.finish()
                })
            });

            g.bench("prog", move |b| {
                b.iter(|| {
                    let mut enc = EncoderConfig::ycbcr(90.0, ss)
                        .progressive(true)
                        .encode_from_bytes(S4K.0, S4K.1, PixelLayout::Rgb8Srgb)
                        .unwrap();
                    enc.push_packed(rgb8_4k, Unstoppable).unwrap();
                    enc.finish()
                })
            });
        });
    }

    // ── Parallel encode: subsampling × size ─────────────────────────

    for &(ss_name, ss) in &subsamplings {
        suite.group(&format!("enc/2k/{ss_name}/par"), |g| {
            g.throughput(Throughput::Bytes((S2K.0 * S2K.1 * 3) as u64));

            g.bench("seq", move |b| {
                b.iter(|| {
                    let mut enc = EncoderConfig::ycbcr(90.0, ss)
                        .encode_from_bytes(S2K.0, S2K.1, PixelLayout::Rgb8Srgb)
                        .unwrap();
                    enc.push_packed(rgb8_2k, Unstoppable).unwrap();
                    enc.finish()
                })
            });

            g.bench("parallel", move |b| {
                b.iter(|| {
                    let mut enc = EncoderConfig::ycbcr(90.0, ss)
                        .parallel(ParallelEncoding::Auto)
                        .encode_from_bytes(S2K.0, S2K.1, PixelLayout::Rgb8Srgb)
                        .unwrap();
                    enc.push_packed(rgb8_2k, Unstoppable).unwrap();
                    enc.finish()
                })
            });
        });

        suite.group(&format!("enc/4k/{ss_name}/par"), |g| {
            g.throughput(Throughput::Bytes((S4K.0 * S4K.1 * 3) as u64));

            g.bench("seq", move |b| {
                b.iter(|| {
                    let mut enc = EncoderConfig::ycbcr(90.0, ss)
                        .encode_from_bytes(S4K.0, S4K.1, PixelLayout::Rgb8Srgb)
                        .unwrap();
                    enc.push_packed(rgb8_4k, Unstoppable).unwrap();
                    enc.finish()
                })
            });

            g.bench("parallel", move |b| {
                b.iter(|| {
                    let mut enc = EncoderConfig::ycbcr(90.0, ss)
                        .parallel(ParallelEncoding::Auto)
                        .encode_from_bytes(S4K.0, S4K.1, PixelLayout::Rgb8Srgb)
                        .unwrap();
                    enc.push_packed(rgb8_4k, Unstoppable).unwrap();
                    enc.finish()
                })
            });
        });
    }

    // ── Pixel format variants (2k, 420, baseline) ─────────────────────

    suite.group("enc/2k/pixfmt", |g| {
        g.throughput(Throughput::Bytes((S2K.0 * S2K.1 * 3) as u64));

        g.bench("rgb8", |b| {
            b.iter(|| {
                let mut enc = EncoderConfig::ycbcr(90.0, ChromaSubsampling::Quarter)
                    .encode_from_bytes(S2K.0, S2K.1, PixelLayout::Rgb8Srgb)
                    .unwrap();
                enc.push_packed(rgb8_2k, Unstoppable).unwrap();
                enc.finish()
            })
        });

        g.bench("rgba8", |b| {
            b.iter(|| {
                let mut enc = EncoderConfig::ycbcr(90.0, ChromaSubsampling::Quarter)
                    .encode_from_bytes(S2K.0, S2K.1, PixelLayout::Rgba8Srgb)
                    .unwrap();
                enc.push_packed(rgba8_2k, Unstoppable).unwrap();
                enc.finish()
            })
        });

        g.bench("bgra8", |b| {
            b.iter(|| {
                let mut enc = EncoderConfig::ycbcr(90.0, ChromaSubsampling::Quarter)
                    .encode_from_bytes(S2K.0, S2K.1, PixelLayout::Bgra8Srgb)
                    .unwrap();
                enc.push_packed(bgra8_2k, Unstoppable).unwrap();
                enc.finish()
            })
        });

        g.bench("rgb16", |b| {
            b.iter(|| {
                let mut enc = EncoderConfig::ycbcr(90.0, ChromaSubsampling::Quarter)
                    .encode_from_bytes(S2K.0, S2K.1, PixelLayout::Rgb16Linear)
                    .unwrap();
                enc.push_packed(bytemuck::cast_slice(rgb16_2k), Unstoppable)
                    .unwrap();
                enc.finish()
            })
        });

        g.bench("rgbf32", |b| {
            b.iter(|| {
                let mut enc = EncoderConfig::ycbcr(90.0, ChromaSubsampling::Quarter)
                    .encode_from_bytes(S2K.0, S2K.1, PixelLayout::RgbF32Linear)
                    .unwrap();
                enc.push_packed(bytemuck::cast_slice(rgbf32_2k), Unstoppable)
                    .unwrap();
                enc.finish()
            })
        });
    });

    // ── Grayscale ──────────────────────────────────────────────────────

    suite.group("enc/2k/gray", |g| {
        g.throughput(Throughput::Bytes((S2K.0 * S2K.1) as u64));

        g.bench("gray8_base", |b| {
            b.iter(|| {
                let mut enc = EncoderConfig::grayscale(90.0)
                    .encode_from_bytes(S2K.0, S2K.1, PixelLayout::Gray8Srgb)
                    .unwrap();
                enc.push_packed(gray8_2k, Unstoppable).unwrap();
                enc.finish()
            })
        });

        g.bench("gray8_prog", |b| {
            b.iter(|| {
                let mut enc = EncoderConfig::grayscale(90.0)
                    .progressive(true)
                    .encode_from_bytes(S2K.0, S2K.1, PixelLayout::Gray8Srgb)
                    .unwrap();
                enc.push_packed(gray8_2k, Unstoppable).unwrap();
                enc.finish()
            })
        });
    });

    // ── XYB color mode (2k, baseline) ─────────────────────────────────

    suite.group("enc/2k/xyb", |g| {
        g.throughput(Throughput::Bytes((S2K.0 * S2K.1 * 3) as u64));

        g.bench("xyb_base", |b| {
            b.iter(|| {
                let mut enc = EncoderConfig::xyb(90.0, XybSubsampling::BQuarter)
                    .encode_from_bytes(S2K.0, S2K.1, PixelLayout::Rgb8Srgb)
                    .unwrap();
                enc.push_packed(rgb8_2k, Unstoppable).unwrap();
                enc.finish()
            })
        });
    });

    // ── Trellis / auto_optimize (2k, 420, progressive) ────────────────

    #[cfg(feature = "trellis")]
    suite.group("enc/2k/optimize", |g| {
        g.throughput(Throughput::Bytes((S2K.0 * S2K.1 * 3) as u64));

        g.bench("trellis", |b| {
            b.iter(|| {
                let mut enc = EncoderConfig::ycbcr(90.0, ChromaSubsampling::Quarter)
                    .trellis(TrellisConfig::default())
                    .encode_from_bytes(S2K.0, S2K.1, PixelLayout::Rgb8Srgb)
                    .unwrap();
                enc.push_packed(rgb8_2k, Unstoppable).unwrap();
                enc.finish()
            })
        });

        g.bench("auto_optimize", |b| {
            b.iter(|| {
                let mut enc = EncoderConfig::ycbcr(90.0, ChromaSubsampling::Quarter)
                    .auto_optimize(true)
                    .encode_from_bytes(S2K.0, S2K.1, PixelLayout::Rgb8Srgb)
                    .unwrap();
                enc.push_packed(rgb8_2k, Unstoppable).unwrap();
                enc.finish()
            })
        });
    });
}

// ── Decode benchmarks ──────────────────────────────────────────────────────

fn bench_decode(suite: &mut Suite) {
    let rgb8_2k = create_rgb8(S2K.0 as usize, S2K.1 as usize);
    let rgb8_4k = create_rgb8(S4K.0 as usize, S4K.1 as usize);
    let gray8_2k = create_gray8(S2K.0 as usize, S2K.1 as usize);

    let subsamplings = [
        ("444", ChromaSubsampling::None),
        ("422", ChromaSubsampling::HalfHorizontal),
        ("420", ChromaSubsampling::Quarter),
        ("440", ChromaSubsampling::HalfVertical),
    ];

    // Pre-encode all test JPEGs (leaked for 'static)
    struct DecTestCase {
        label: String,
        jpeg: &'static [u8],
        pixels: u64,
    }

    let mut cases_2k = Vec::new();
    let mut cases_4k = Vec::new();

    for &(ss_name, ss) in &subsamplings {
        for progressive in [false, true] {
            let mode = if progressive { "prog" } else { "base" };
            let label = format!("{ss_name}_{mode}");

            let j2k = encode_jpeg(&rgb8_2k, S2K.0, S2K.1, ss, progressive);
            cases_2k.push(DecTestCase {
                label: label.clone(),
                jpeg: Vec::leak(j2k),
                pixels: (S2K.0 * S2K.1) as u64,
            });

            let j4k = encode_jpeg(&rgb8_4k, S4K.0, S4K.1, ss, progressive);
            cases_4k.push(DecTestCase {
                label,
                jpeg: Vec::leak(j4k),
                pixels: (S4K.0 * S4K.1) as u64,
            });
        }
    }

    // Grayscale
    let gray_jpeg_2k: &'static [u8] = Vec::leak(encode_grayscale_jpeg(&gray8_2k, S2K.0, S2K.1));

    // Leak the case vecs
    let cases_2k: &'static [DecTestCase] = Vec::leak(cases_2k);
    let cases_4k: &'static [DecTestCase] = Vec::leak(cases_4k);

    // ── Subsampling × size × mode (default decoder) ───────────────────

    for case in cases_2k.iter() {
        suite.group(&format!("dec/2k/{}", case.label), |g| {
            g.throughput(Throughput::Elements(case.pixels));

            g.bench("default", |b| {
                let dec = Decoder::new().output_format(PixelFormat::Rgb);
                b.iter(|| dec.decode(case.jpeg, Unstoppable))
            });
        });
    }

    for case in cases_4k.iter() {
        suite.group(&format!("dec/4k/{}", case.label), |g| {
            g.throughput(Throughput::Elements(case.pixels));

            g.bench("default", |b| {
                let dec = Decoder::new().output_format(PixelFormat::Rgb);
                b.iter(|| dec.decode(case.jpeg, Unstoppable))
            });
        });
    }

    // ── Parallel decode: subsampling × size (baseline only, needs DRI) ─

    for case in cases_2k.iter().filter(|c| c.label.ends_with("_base")) {
        suite.group(&format!("dec/2k/{}/par", case.label), |g| {
            g.throughput(Throughput::Elements(case.pixels));

            g.bench("seq", |b| {
                let dec = Decoder::new()
                    .num_threads(1)
                    .output_format(PixelFormat::Rgb);
                b.iter(|| dec.decode(case.jpeg, Unstoppable))
            });

            g.bench("parallel", |b| {
                let dec = Decoder::new()
                    .num_threads(0)
                    .output_format(PixelFormat::Rgb);
                b.iter(|| dec.decode(case.jpeg, Unstoppable))
            });
        });
    }

    for case in cases_4k.iter().filter(|c| c.label.ends_with("_base")) {
        suite.group(&format!("dec/4k/{}/par", case.label), |g| {
            g.throughput(Throughput::Elements(case.pixels));

            g.bench("seq", |b| {
                let dec = Decoder::new()
                    .num_threads(1)
                    .output_format(PixelFormat::Rgb);
                b.iter(|| dec.decode(case.jpeg, Unstoppable))
            });

            g.bench("parallel", |b| {
                let dec = Decoder::new()
                    .num_threads(0)
                    .output_format(PixelFormat::Rgb);
                b.iter(|| dec.decode(case.jpeg, Unstoppable))
            });
        });
    }

    // ── Wave-parallel scanline decode (2k, 420 baseline) ─────────────

    let baseline_420_2k_for_wave = cases_2k.iter().find(|c| c.label == "420_base").unwrap();
    let baseline_420_4k_for_wave = cases_4k.iter().find(|c| c.label == "420_base").unwrap();

    suite.group("dec/2k/wave", |g| {
        g.throughput(Throughput::Elements(baseline_420_2k_for_wave.pixels));

        g.bench("scanline_seq", |b| {
            b.iter(|| {
                let dec = Decoder::new()
                    .num_threads(1)
                    .chroma_upsampling(zenjpeg::decode::ChromaUpsampling::NearestNeighbor)
                    .output_format(PixelFormat::Rgb);
                let mut reader = dec.scanline_reader(baseline_420_2k_for_wave.jpeg).unwrap();
                let w = reader.width() as usize;
                let h = reader.height() as usize;
                let mut buf = vec![0u8; w * h * 3];
                reader
                    .read_rows_rgb8(imgref::ImgRefMut::new(&mut buf, w * 3, h))
                    .unwrap();
                buf
            })
        });

        g.bench("scanline_wave", |b| {
            b.iter(|| {
                let dec = Decoder::new()
                    .num_threads(0)
                    .chroma_upsampling(zenjpeg::decode::ChromaUpsampling::NearestNeighbor)
                    .output_format(PixelFormat::Rgb);
                let mut reader = dec.scanline_reader(baseline_420_2k_for_wave.jpeg).unwrap();
                let w = reader.width() as usize;
                let h = reader.height() as usize;
                let mut buf = vec![0u8; w * h * 3];
                reader
                    .read_rows_rgb8(imgref::ImgRefMut::new(&mut buf, w * 3, h))
                    .unwrap();
                buf
            })
        });
    });

    suite.group("dec/4k/wave", |g| {
        g.throughput(Throughput::Elements(baseline_420_4k_for_wave.pixels));

        g.bench("scanline_seq", |b| {
            b.iter(|| {
                let dec = Decoder::new()
                    .num_threads(1)
                    .chroma_upsampling(zenjpeg::decode::ChromaUpsampling::NearestNeighbor)
                    .output_format(PixelFormat::Rgb);
                let mut reader = dec.scanline_reader(baseline_420_4k_for_wave.jpeg).unwrap();
                let w = reader.width() as usize;
                let h = reader.height() as usize;
                let mut buf = vec![0u8; w * h * 3];
                reader
                    .read_rows_rgb8(imgref::ImgRefMut::new(&mut buf, w * 3, h))
                    .unwrap();
                buf
            })
        });

        g.bench("scanline_wave", |b| {
            b.iter(|| {
                let dec = Decoder::new()
                    .num_threads(0)
                    .chroma_upsampling(zenjpeg::decode::ChromaUpsampling::NearestNeighbor)
                    .output_format(PixelFormat::Rgb);
                let mut reader = dec.scanline_reader(baseline_420_4k_for_wave.jpeg).unwrap();
                let w = reader.width() as usize;
                let h = reader.height() as usize;
                let mut buf = vec![0u8; w * h * 3];
                reader
                    .read_rows_rgb8(imgref::ImgRefMut::new(&mut buf, w * 3, h))
                    .unwrap();
                buf
            })
        });
    });

    // ── IDCT methods (2k, 420 baseline) ───────────────────────────────

    // Find the 420 baseline case
    let baseline_420_2k = cases_2k.iter().find(|c| c.label == "420_base").unwrap();

    suite.group("dec/2k/idct", |g| {
        g.throughput(Throughput::Elements(baseline_420_2k.pixels));

        g.bench("jpegli", |b| {
            let dec = Decoder::new()
                .idct_method(zenjpeg::decode::IdctMethod::Jpegli)
                .output_format(PixelFormat::Rgb);
            b.iter(|| dec.decode(baseline_420_2k.jpeg, Unstoppable))
        });

        g.bench("libjpeg", |b| {
            let dec = Decoder::new()
                .idct_method(zenjpeg::decode::IdctMethod::Libjpeg)
                .output_format(PixelFormat::Rgb);
            b.iter(|| dec.decode(baseline_420_2k.jpeg, Unstoppable))
        });
    });

    // ── Chroma upsampling (2k, 420 baseline) ──────────────────────────

    suite.group("dec/2k/upsample", |g| {
        g.throughput(Throughput::Elements(baseline_420_2k.pixels));

        g.bench("triangle", |b| {
            let dec = Decoder::new()
                .chroma_upsampling(zenjpeg::decode::ChromaUpsampling::Triangle)
                .output_format(PixelFormat::Rgb);
            b.iter(|| dec.decode(baseline_420_2k.jpeg, Unstoppable))
        });

        g.bench("nearest", |b| {
            let dec = Decoder::new()
                .chroma_upsampling(zenjpeg::decode::ChromaUpsampling::NearestNeighbor)
                .output_format(PixelFormat::Rgb);
            b.iter(|| dec.decode(baseline_420_2k.jpeg, Unstoppable))
        });
    });

    // ── Deblock modes (2k, 420 baseline, low quality for deblock) ─────

    let lowq_jpeg_2k: &'static [u8] = Vec::leak(encode_jpeg(
        &rgb8_2k,
        S2K.0,
        S2K.1,
        ChromaSubsampling::Quarter,
        false,
    ));

    suite.group("dec/2k/deblock", |g| {
        g.throughput(Throughput::Elements((S2K.0 * S2K.1) as u64));

        g.bench("off", |b| {
            let dec = Decoder::new()
                .deblock(zenjpeg::decode::DeblockMode::Off)
                .output_format(PixelFormat::Rgb);
            b.iter(|| dec.decode(lowq_jpeg_2k, Unstoppable))
        });

        g.bench("boundary4tap", |b| {
            let dec = Decoder::new()
                .deblock(zenjpeg::decode::DeblockMode::Boundary4Tap)
                .output_format(PixelFormat::Rgb);
            b.iter(|| dec.decode(lowq_jpeg_2k, Unstoppable))
        });

        g.bench("knusperli", |b| {
            let dec = Decoder::new()
                .deblock(zenjpeg::decode::DeblockMode::Knusperli)
                .output_format(PixelFormat::Rgb);
            b.iter(|| dec.decode(lowq_jpeg_2k, Unstoppable))
        });
    });

    // ── Output pixel formats (2k, 420 baseline) ──────────────────────

    suite.group("dec/2k/outfmt", |g| {
        g.throughput(Throughput::Elements(baseline_420_2k.pixels));

        g.bench("rgb8", |b| {
            let dec = Decoder::new().output_format(PixelFormat::Rgb);
            b.iter(|| dec.decode(baseline_420_2k.jpeg, Unstoppable))
        });
        g.bench("rgba8", |b| {
            let dec = Decoder::new().output_format(PixelFormat::Rgba);
            b.iter(|| dec.decode(baseline_420_2k.jpeg, Unstoppable))
        });
        g.bench("bgr8", |b| {
            let dec = Decoder::new().output_format(PixelFormat::Bgr);
            b.iter(|| dec.decode(baseline_420_2k.jpeg, Unstoppable))
        });
        g.bench("bgra8", |b| {
            let dec = Decoder::new().output_format(PixelFormat::Bgra);
            b.iter(|| dec.decode(baseline_420_2k.jpeg, Unstoppable))
        });
        g.bench("gray8", |b| {
            let dec = Decoder::new().output_format(PixelFormat::Gray);
            b.iter(|| dec.decode(baseline_420_2k.jpeg, Unstoppable))
        });
        g.bench("rgb16", |b| {
            let dec = Decoder::new().output_format(PixelFormat::Rgb16);
            b.iter(|| dec.decode(baseline_420_2k.jpeg, Unstoppable))
        });
        g.bench("rgbf32", |b| {
            let dec = Decoder::new().output_format(PixelFormat::RgbF32);
            b.iter(|| dec.decode(baseline_420_2k.jpeg, Unstoppable))
        });
    });

    // ── Scanline decode (2k, 420 baseline) ────────────────────────────

    suite.group("dec/2k/scanline", |g| {
        g.throughput(Throughput::Elements(baseline_420_2k.pixels));

        g.bench("full_decode", |b| {
            let dec = Decoder::new().output_format(PixelFormat::Rgb);
            b.iter(|| dec.decode(baseline_420_2k.jpeg, Unstoppable))
        });

        g.bench("scanline_reader", |b| {
            b.iter(|| {
                let dec = Decoder::new().output_format(PixelFormat::Rgb);
                let mut reader = dec.scanline_reader(baseline_420_2k.jpeg).unwrap();
                let w = reader.width() as usize;
                let h = reader.height() as usize;
                let mut buf = vec![0u8; w * h * 3];
                reader
                    .read_rows_rgb8(imgref::ImgRefMut::new(&mut buf, w * 3, h))
                    .unwrap();
                buf
            })
        });
    });

    // ── Grayscale decode (2k) ─────────────────────────────────────────

    suite.group("dec/2k/gray", |g| {
        g.throughput(Throughput::Elements((S2K.0 * S2K.1) as u64));

        g.bench("gray_to_rgb", |b| {
            let dec = Decoder::new().output_format(PixelFormat::Rgb);
            b.iter(|| dec.decode(gray_jpeg_2k, Unstoppable))
        });

        g.bench("gray_to_gray", |b| {
            let dec = Decoder::new().output_format(PixelFormat::Gray);
            b.iter(|| dec.decode(gray_jpeg_2k, Unstoppable))
        });
    });

    // ── Dequant bias (2k, 420 baseline) ───────────────────────────────

    suite.group("dec/2k/dequant", |g| {
        g.throughput(Throughput::Elements(baseline_420_2k.pixels));

        g.bench("default", |b| {
            let dec = Decoder::new().output_format(PixelFormat::Rgb);
            b.iter(|| dec.decode(baseline_420_2k.jpeg, Unstoppable))
        });

        g.bench("dequant_bias", |b| {
            let dec = Decoder::new()
                .dequant_bias(true)
                .output_format(PixelFormat::Rgb);
            b.iter(|| dec.decode(baseline_420_2k.jpeg, Unstoppable))
        });
    });
}

fn bench_all(suite: &mut Suite) {
    bench_encode(suite);
    bench_decode(suite);
}

zenbench::main!(bench_all);
