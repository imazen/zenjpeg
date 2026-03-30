//! Decode benchmark: zenjpeg vs libjpeg-turbo/mozjpeg on real photos.
//!
//! Uses zenbench for interleaved measurement with paired statistics.
//! Tests all major decode paths on CID22 corpus images.
//!
//! Run:
//! ```bash
//! cargo bench -p zenjpeg --bench decode_zenbench --features "trellis decoder"
//! ```

use enough::Unstoppable;
use std::path::Path;
use zenbench::prelude::*;
use zenjpeg::decode::{ChromaUpsampling, DeblockMode};
use zenjpeg::decoder::Decoder;

// ── Image loading ───────────────────────────────────────────────────────────

fn load_png_rgb(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let file = std::fs::File::open(path).ok()?;
    let dec = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = dec.read_info().ok()?;
    let mut buf = vec![0u8; reader.output_buffer_size()?];
    let info = reader.next_frame(&mut buf).ok()?;
    let rgb = match info.color_type {
        png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgba => {
            let src = &buf[..info.buffer_size()];
            let mut rgb = Vec::with_capacity((info.width * info.height * 3) as usize);
            for c in src.chunks_exact(4) {
                rgb.extend_from_slice(&c[..3]);
            }
            rgb
        }
        _ => return None,
    };
    Some((rgb, info.width, info.height))
}

/// Encode a test image at given quality with mozjpeg (baseline, with DRI for parallel tests).
fn encode_mozjpeg_baseline(pixels: &[u8], w: u32, h: u32, q: u8) -> Vec<u8> {
    mozjpeg_rs::Encoder::new(mozjpeg_rs::Preset::BaselineBalanced)
        .quality(q)
        .subsampling(mozjpeg_rs::Subsampling::S420)
        .encode_rgb(pixels, w, h)
        .expect("mozjpeg encode")
}

/// Encode a test image as progressive (no DRI — standard mozjpeg progressive).
fn encode_mozjpeg_progressive(pixels: &[u8], w: u32, h: u32, q: u8) -> Vec<u8> {
    mozjpeg_rs::Encoder::new(mozjpeg_rs::Preset::ProgressiveSmallest)
        .quality(q)
        .subsampling(mozjpeg_rs::Subsampling::S420)
        .encode_rgb(pixels, w, h)
        .expect("mozjpeg encode")
}

/// Decode with libjpeg-turbo/mozjpeg via FFI (the C reference decoder).
fn decode_libjpeg_turbo(jpeg: &[u8]) -> Vec<u8> {
    use mozjpeg_sys::*;
    use std::mem;
    unsafe {
        let mut err: jpeg_error_mgr = mem::zeroed();
        jpeg_std_error(&mut err);
        let mut cinfo: jpeg_decompress_struct = mem::zeroed();
        cinfo.common.err = &mut err;
        jpeg_create_decompress(&mut cinfo);
        jpeg_mem_src(&mut cinfo, jpeg.as_ptr(), jpeg.len() as _);
        jpeg_read_header(&mut cinfo, true as boolean);
        cinfo.out_color_space = J_COLOR_SPACE::JCS_RGB;
        jpeg_start_decompress(&mut cinfo);
        let (w, h) = (cinfo.output_width, cinfo.output_height);
        let stride = w as usize * cinfo.output_components as usize;
        let mut out = vec![0u8; h as usize * stride];
        while cinfo.output_scanline < h {
            let off = cinfo.output_scanline as usize * stride;
            let mut p = out[off..].as_mut_ptr();
            jpeg_read_scanlines(&mut cinfo, &mut p, 1);
        }
        jpeg_finish_decompress(&mut cinfo);
        jpeg_destroy_decompress(&mut cinfo);
        out
    }
}

/// Decode with zune-jpeg.
fn decode_zune(jpeg: &[u8]) -> Vec<u8> {
    use zune_core::options::DecoderOptions;
    use zune_jpeg::JpegDecoder;
    let options =
        DecoderOptions::default().jpeg_set_out_colorspace(zune_core::colorspace::ColorSpace::RGB);
    let mut decoder = JpegDecoder::new_with_options(std::io::Cursor::new(jpeg), options);
    decoder.decode().expect("zune decode")
}

// ── Load corpus ─────────────────────────────────────────────────────────────

#[allow(dead_code)]
struct TestImage {
    name: String,
    baseline_q85: Vec<u8>,
    progressive_q85: Vec<u8>,
    baseline_q20: Vec<u8>,
    pixels: usize, // w * h
}

fn load_test_images() -> Vec<TestImage> {
    let corpus = match codec_corpus::Corpus::new() {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };
    let dir = match corpus.get("CID22/CID22-512/training") {
        Ok(d) => d,
        Err(_) => return Vec::new(),
    };

    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .is_some_and(|x| x.eq_ignore_ascii_case("png"))
        })
        .map(|e| e.path())
        .collect();
    paths.sort();
    paths.truncate(10); // 10 images for manageable bench time

    paths
        .iter()
        .filter_map(|p| {
            let (pixels, w, h) = load_png_rgb(p)?;
            let name = p.file_stem()?.to_string_lossy().into_owned();
            Some(TestImage {
                baseline_q85: encode_mozjpeg_baseline(&pixels, w, h, 85),
                progressive_q85: encode_mozjpeg_progressive(&pixels, w, h, 85),
                baseline_q20: encode_mozjpeg_baseline(&pixels, w, h, 20),
                pixels: (w * h) as usize,
                name,
            })
        })
        .collect()
}

// ── Benchmarks ──────────────────────────────────────────────────────────────

static IMAGES: std::sync::OnceLock<Vec<TestImage>> = std::sync::OnceLock::new();

fn get_images() -> &'static [TestImage] {
    IMAGES.get_or_init(load_test_images)
}

fn bench_decode(suite: &mut Suite) {
    let images = get_images();
    if images.is_empty() {
        eprintln!("No CID22 corpus found, skipping benchmarks");
        return;
    }

    let total_baseline_bytes: usize = images.iter().map(|i| i.baseline_q85.len()).sum();
    let total_prog_bytes: usize = images.iter().map(|i| i.progressive_q85.len()).sum();
    let total_lowq_bytes: usize = images.iter().map(|i| i.baseline_q20.len()).sum();

    eprintln!(
        "Loaded {} CID22 images (512x512), {:.1}MB baseline Q85, {:.1}MB progressive Q85",
        images.len(),
        total_baseline_bytes as f64 / 1024.0 / 1024.0,
        total_prog_bytes as f64 / 1024.0 / 1024.0,
    );

    // ── Baseline Q85 decode comparison ──────────────────────────────────

    suite.group("baseline_4:2:0_Q85", |g| {
        g.throughput(Throughput::Bytes(total_baseline_bytes as u64));

        g.bench("libjpeg-turbo/mozjpeg (C, NASM SIMD)", |b| {
            b.iter(|| {
                for img in get_images() {
                    decode_libjpeg_turbo(&img.baseline_q85);
                }
            })
        });

        g.bench("zune-jpeg", |b| {
            b.iter(|| {
                for img in get_images() {
                    decode_zune(&img.baseline_q85);
                }
            })
        });

        g.bench("zenjpeg default (Jpegli IDCT, Triangle)", |b| {
            let dec = Decoder::new();
            b.iter(|| {
                for img in get_images() {
                    dec.decode(&img.baseline_q85, Unstoppable).unwrap();
                }
            })
        });

        g.bench("zenjpeg LibjpegCompat (Libjpeg IDCT)", |b| {
            let dec = Decoder::new().chroma_upsampling(ChromaUpsampling::Triangle);
            b.iter(|| {
                for img in get_images() {
                    dec.decode(&img.baseline_q85, Unstoppable).unwrap();
                }
            })
        });

        // Isolate: LibjpegCompat upsample but Jpegli IDCT
        g.bench("zenjpeg LibjpegCompat + Jpegli IDCT", |b| {
            let dec = Decoder::new()
                .chroma_upsampling(ChromaUpsampling::Triangle)
                .idct_method(zenjpeg::decode::IdctMethod::Jpegli);
            b.iter(|| {
                for img in get_images() {
                    dec.decode(&img.baseline_q85, Unstoppable).unwrap();
                }
            })
        });

        // Isolate: Triangle upsample but Libjpeg IDCT
        g.bench("zenjpeg Triangle + Libjpeg IDCT", |b| {
            let dec = Decoder::new().idct_method(zenjpeg::decode::IdctMethod::Libjpeg);
            b.iter(|| {
                for img in get_images() {
                    dec.decode(&img.baseline_q85, Unstoppable).unwrap();
                }
            })
        });

        g.bench("zenjpeg NearestNeighbor (box filter)", |b| {
            let dec = Decoder::new().chroma_upsampling(ChromaUpsampling::NearestNeighbor);
            b.iter(|| {
                for img in get_images() {
                    dec.decode(&img.baseline_q85, Unstoppable).unwrap();
                }
            })
        });
    });

    // ── Progressive Q85 decode comparison ───────────────────────────────

    suite.group("progressive_4:2:0_Q85", |g| {
        g.throughput(Throughput::Bytes(total_prog_bytes as u64));

        g.bench("libjpeg-turbo/mozjpeg (C, NASM SIMD)", |b| {
            b.iter(|| {
                for img in get_images() {
                    decode_libjpeg_turbo(&img.progressive_q85);
                }
            })
        });

        g.bench("zune-jpeg", |b| {
            b.iter(|| {
                for img in get_images() {
                    decode_zune(&img.progressive_q85);
                }
            })
        });

        g.bench("zenjpeg default", |b| {
            let dec = Decoder::new();
            b.iter(|| {
                for img in get_images() {
                    dec.decode(&img.progressive_q85, Unstoppable).unwrap();
                }
            })
        });
    });

    // ── Deblock overhead (baseline Q20 — where deblock matters most) ────

    suite.group("deblock_baseline_4:2:0_Q20", |g| {
        g.throughput(Throughput::Bytes(total_lowq_bytes as u64));

        g.bench("zenjpeg Off (no deblock)", |b| {
            let dec = Decoder::new();
            b.iter(|| {
                for img in get_images() {
                    dec.decode(&img.baseline_q20, Unstoppable).unwrap();
                }
            })
        });

        g.bench("zenjpeg Boundary4Tap", |b| {
            let dec = Decoder::new().deblock(DeblockMode::Boundary4Tap);
            b.iter(|| {
                for img in get_images() {
                    dec.decode(&img.baseline_q20, Unstoppable).unwrap();
                }
            })
        });

        g.bench("zenjpeg Knusperli", |b| {
            let dec = Decoder::new().deblock(DeblockMode::Knusperli);
            b.iter(|| {
                for img in get_images() {
                    dec.decode(&img.baseline_q20, Unstoppable).unwrap();
                }
            })
        });

        g.bench("zenjpeg Auto", |b| {
            let dec = Decoder::new().deblock(DeblockMode::Auto);
            b.iter(|| {
                for img in get_images() {
                    dec.decode(&img.baseline_q20, Unstoppable).unwrap();
                }
            })
        });
    });

    // ── Scanline reader comparison (baseline Q85) ───────────────────────

    suite.group("scanline_baseline_4:2:0_Q85", |g| {
        g.throughput(Throughput::Bytes(total_baseline_bytes as u64));

        g.bench("zenjpeg scanline Off", |b| {
            b.iter(|| {
                for img in get_images() {
                    let mut reader = Decoder::new().scanline_reader(&img.baseline_q85).unwrap();
                    let mut buf = vec![0u8; 512 * 512 * 3];
                    reader
                        .read_rows_rgb8(imgref::ImgRefMut::new(&mut buf, 512 * 3, 512))
                        .unwrap();
                }
            })
        });

        g.bench("zenjpeg scanline Boundary4Tap", |b| {
            b.iter(|| {
                for img in get_images() {
                    let mut reader = Decoder::new()
                        .deblock(DeblockMode::Boundary4Tap)
                        .scanline_reader(&img.baseline_q85)
                        .unwrap();
                    let mut buf = vec![0u8; 512 * 512 * 3];
                    reader
                        .read_rows_rgb8(imgref::ImgRefMut::new(&mut buf, 512 * 3, 512))
                        .unwrap();
                }
            })
        });
    });

    // ── dequant_bias overhead (baseline Q85) ────────────────────────────

    suite.group("dequant_bias_baseline_Q85", |g| {
        g.throughput(Throughput::Bytes(total_baseline_bytes as u64));

        g.bench("zenjpeg default (no bias)", |b| {
            let dec = Decoder::new();
            b.iter(|| {
                for img in get_images() {
                    dec.decode(&img.baseline_q85, Unstoppable).unwrap();
                }
            })
        });

        g.bench("zenjpeg dequant_bias", |b| {
            let dec = Decoder::new().dequant_bias(true);
            b.iter(|| {
                for img in get_images() {
                    dec.decode(&img.baseline_q85, Unstoppable).unwrap();
                }
            })
        });
    });
}

// ── Size matrix benchmark (synthetic images, variant/size naming for matrix chart) ──

fn create_test_jpeg(width: u32, height: u32, quality: f32, progressive: bool) -> Vec<u8> {
    use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, PixelLayout};

    let mut data = vec![0u8; (width * height * 3) as usize];
    for y in 0..height as usize {
        for x in 0..width as usize {
            let idx = (y * width as usize + x) * 3;
            let bx = (x / 8) as u32;
            let by = (y / 8) as u32;
            let block_hash = bx.wrapping_mul(2654435761).wrapping_add(by.wrapping_mul(40503));
            let block_type = block_hash % 4;
            let px = x as u32;
            let py = y as u32;
            let mut h = px.wrapping_mul(374761393).wrapping_add(py.wrapping_mul(668265263));
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
                    let edge = if (x % 8 < 4) ^ (y % 8 < 4) { 200u8 } else { 55u8 };
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

    let mut config = EncoderConfig::ycbcr(quality, ChromaSubsampling::Quarter)
        .progressive(progressive);
    if progressive {
        config = config.restart_mcu_rows(0); // zune-jpeg bug with DRI + progressive
    }
    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .expect("encoder");
    enc.push_packed(&data, Unstoppable).expect("push");
    enc.finish().expect("finish")
}

fn bench_decode_matrix(suite: &mut Suite) {
    let sizes: &[(u32, u32)] = &[
        (256, 256),
        (512, 512),
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
    ];

    eprintln!("Generating synthetic test images...");

    // Pre-encode at all sizes
    let baseline_jpegs: Vec<(String, Vec<u8>)> = sizes
        .iter()
        .map(|&(w, h)| {
            let label = format!("{w}x{h}");
            let jpeg = create_test_jpeg(w, h, 85.0, false);
            eprintln!("  baseline {label}: {} bytes", jpeg.len());
            (label, jpeg)
        })
        .collect();

    let progressive_jpegs: Vec<(String, Vec<u8>)> = sizes
        .iter()
        .map(|&(w, h)| {
            let label = format!("{w}x{h}");
            let jpeg = create_test_jpeg(w, h, 85.0, true);
            eprintln!("  progressive {label}: {} bytes", jpeg.len());
            (label, jpeg)
        })
        .collect();

    // Leak the encoded data so closures can capture &'static references
    let baseline_jpegs: &'static [(String, Vec<u8>)] = baseline_jpegs.leak();
    let progressive_jpegs: &'static [(String, Vec<u8>)] = progressive_jpegs.leak();

    // ── Baseline 4:2:0 matrix ──────────────────────────────────────────

    suite.group("baseline_420_matrix", |g| {
        for (label, jpeg) in baseline_jpegs.iter() {
            g.bench(format!("mozjpeg/{label}"), {
                let jpeg = jpeg.clone();
                move |b| b.iter(|| decode_libjpeg_turbo(&jpeg))
            });
            g.bench(format!("zune/{label}"), {
                let jpeg = jpeg.clone();
                move |b| b.iter(|| decode_zune(&jpeg))
            });
            g.bench(format!("zenjpeg/{label}"), {
                let jpeg = jpeg.clone();
                move |b| {
                    let dec = Decoder::new();
                    b.iter(|| dec.decode(&jpeg, Unstoppable).unwrap())
                }
            });
            g.bench(format!("zenjpeg-box/{label}"), {
                let jpeg = jpeg.clone();
                move |b| {
                    let dec = Decoder::new()
                        .chroma_upsampling(ChromaUpsampling::NearestNeighbor);
                    b.iter(|| dec.decode(&jpeg, Unstoppable).unwrap())
                }
            });
        }
    });

    // ── Progressive 4:2:0 matrix ───────────────────────────────────────

    suite.group("progressive_420_matrix", |g| {
        for (label, jpeg) in progressive_jpegs.iter() {
            g.bench(format!("mozjpeg/{label}"), {
                let jpeg = jpeg.clone();
                move |b| b.iter(|| decode_libjpeg_turbo(&jpeg))
            });
            g.bench(format!("zune/{label}"), {
                let jpeg = jpeg.clone();
                move |b| b.iter(|| decode_zune(&jpeg))
            });
            g.bench(format!("zenjpeg/{label}"), {
                let jpeg = jpeg.clone();
                move |b| {
                    let dec = Decoder::new();
                    b.iter(|| dec.decode(&jpeg, Unstoppable).unwrap())
                }
            });
        }
    });

    // ── Parallel matrix (feature-gated) ────────────────────────────────

    #[cfg(feature = "parallel")]
    suite.group("parallel_420_matrix", |g| {
        // Skip 256 (too small for parallel)
        for (label, jpeg) in baseline_jpegs.iter().skip(1) {
            g.bench(format!("zenjpeg-seq/{label}"), {
                let jpeg = jpeg.clone();
                move |b| {
                    let dec = Decoder::new().num_threads(1);
                    b.iter(|| dec.decode(&jpeg, Unstoppable).unwrap())
                }
            });
            g.bench(format!("zenjpeg-par/{label}"), {
                let jpeg = jpeg.clone();
                move |b| {
                    let dec = Decoder::new();
                    b.iter(|| dec.decode(&jpeg, Unstoppable).unwrap())
                }
            });
            g.bench(format!("zenjpeg-box-par/{label}"), {
                let jpeg = jpeg.clone();
                move |b| {
                    let dec = Decoder::new()
                        .chroma_upsampling(ChromaUpsampling::NearestNeighbor);
                    b.iter(|| dec.decode(&jpeg, Unstoppable).unwrap())
                }
            });
        }
    });
}

fn bench_all(suite: &mut Suite) {
    bench_decode(suite);
    bench_decode_matrix(suite);
}

zenbench::main!(bench_all);
