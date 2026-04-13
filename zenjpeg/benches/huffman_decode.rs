//! Focused Huffman decode benchmark for measuring entropy decoding throughput.
//!
//! Uses CLIC 2025 photographs from codec-corpus for realistic coefficient
//! distributions. Falls back to noise+patches synthetic images if the corpus
//! is unavailable (no network / CI).
//!
//! Tests baseline and progressive at multiple quality levels to exercise
//! different Huffman code length distributions:
//! - Q85: typical web quality, moderate AC density
//! - Q50: lower quality, sparser AC coefficients (more EOBs, fast_ac hits)
//!
//! Run:
//! ```sh
//! cargo bench -p zenjpeg --bench huffman_decode --features decoder
//! ```

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use enough::Unstoppable;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use zenjpeg::decode::Decoder;
use zenjpeg::decoder::PixelFormat;
use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, PixelLayout};

// ── Image loading ──────────────────────────────────────────────────────────

/// Load a PNG file to flat RGB bytes.
fn load_png_rgb(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let file = std::fs::File::open(path).ok()?;
    let dec = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = dec.read_info().ok()?;
    let mut buf = vec![0u8; reader.output_buffer_size()?];
    let info = reader.next_frame(&mut buf).ok()?;
    let (w, h) = (info.width, info.height);

    let rgb = match info.color_type {
        png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgba => {
            let src = &buf[..info.buffer_size()];
            let mut rgb = Vec::with_capacity((w * h * 3) as usize);
            for chunk in src.chunks_exact(4) {
                rgb.extend_from_slice(&chunk[..3]);
            }
            rgb
        }
        png::ColorType::Grayscale => {
            let src = &buf[..info.buffer_size()];
            let mut rgb = Vec::with_capacity((w * h * 3) as usize);
            for &g in src {
                rgb.extend_from_slice(&[g, g, g]);
            }
            rgb
        }
        _ => return None,
    };

    Some((rgb, w, h))
}

/// Collect all PNG files from a directory, sorted by name.
fn collect_pngs(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .is_some_and(|x| x.eq_ignore_ascii_case("png"))
        })
        .map(|e| e.path())
        .collect();
    files.sort();
    files
}

/// Generate noise+patches pixel data (fallback when corpus is unavailable).
fn generate_photo_like_pixels(width: u32, height: u32) -> Vec<u8> {
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
    data
}

// ── JPEG encoding ──────────────────────────────────────────────────────────

fn encode_jpeg(
    pixels: &[u8],
    width: u32,
    height: u32,
    quality: f32,
    subsampling: ChromaSubsampling,
    progressive: bool,
) -> Vec<u8> {
    let mut config = EncoderConfig::ycbcr(quality, subsampling).progressive(progressive);
    if progressive {
        config = config.restart_mcu_rows(0);
    }
    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .expect("encoder creation");
    enc.push_packed(pixels, Unstoppable)
        .expect("push_packed");
    enc.finish().expect("finish")
}

// ── Test image set ─────────────────────────────────────────────────────────

struct SourceImage {
    name: String,
    pixels: Vec<u8>,
    width: u32,
    height: u32,
}

struct EncodedImage {
    label: String,
    jpeg: Vec<u8>,
    pixels: u64,
}

struct TestSet {
    baseline_q85: Vec<EncodedImage>,
    baseline_q50: Vec<EncodedImage>,
    progressive_q85: Vec<EncodedImage>,
    baseline_444_q85: Vec<EncodedImage>,
    #[allow(dead_code)]
    source_tag: &'static str,
}

/// Try loading CLIC 2025 images from codec-corpus. Returns up to `max` images.
fn load_clic_images(max: usize) -> Option<Vec<SourceImage>> {
    let corpus = codec_corpus::Corpus::new().ok()?;

    // Try CLIC 2025 final-test first, then training
    let dir = corpus
        .get("clic2025/final-test")
        .or_else(|_| corpus.get("clic2025/training"))
        .ok()?;

    let pngs = collect_pngs(&dir);
    if pngs.is_empty() {
        return None;
    }

    let images: Vec<SourceImage> = pngs
        .iter()
        .take(max)
        .filter_map(|p| {
            let (pixels, w, h) = load_png_rgb(p)?;
            let name = p.file_stem()?.to_string_lossy().into_owned();
            Some(SourceImage {
                name,
                pixels,
                width: w,
                height: h,
            })
        })
        .collect();

    if images.is_empty() {
        None
    } else {
        Some(images)
    }
}

/// Fall back to CID22 512px corpus.
fn load_cid22_images(max: usize) -> Option<Vec<SourceImage>> {
    let corpus = codec_corpus::Corpus::new().ok()?;
    let dir = corpus.get("CID22/CID22-512/training").ok()?;
    let pngs = collect_pngs(&dir);
    if pngs.is_empty() {
        return None;
    }

    let images: Vec<SourceImage> = pngs
        .iter()
        .take(max)
        .filter_map(|p| {
            let (pixels, w, h) = load_png_rgb(p)?;
            let name = p.file_stem()?.to_string_lossy().into_owned();
            Some(SourceImage {
                name,
                pixels,
                width: w,
                height: h,
            })
        })
        .collect();

    if images.is_empty() {
        None
    } else {
        Some(images)
    }
}

/// Generate synthetic fallback images at fixed sizes.
fn generate_synthetic_images() -> Vec<SourceImage> {
    [(1024, 1024, "synth_1024"), (2048, 2048, "synth_2048")]
        .iter()
        .map(|&(w, h, name)| SourceImage {
            name: name.to_string(),
            pixels: generate_photo_like_pixels(w, h),
            width: w,
            height: h,
        })
        .collect()
}

fn build_test_set() -> TestSet {
    // Try CLIC 2025 → CID22 → synthetic fallback
    let (sources, tag) = if let Some(images) = load_clic_images(8) {
        eprintln!(
            "Loaded {} CLIC 2025 images for Huffman decode benchmark",
            images.len()
        );
        for img in &images {
            eprintln!("  {} ({}x{})", img.name, img.width, img.height);
        }
        (images, "CLIC2025")
    } else if let Some(images) = load_cid22_images(10) {
        eprintln!(
            "CLIC unavailable, loaded {} CID22 images (512x512)",
            images.len()
        );
        (images, "CID22")
    } else {
        eprintln!("No corpus available, using synthetic noise+patches images");
        (generate_synthetic_images(), "synthetic")
    };

    let mut baseline_q85 = Vec::new();
    let mut baseline_q50 = Vec::new();
    let mut progressive_q85 = Vec::new();
    let mut baseline_444_q85 = Vec::new();

    for src in &sources {
        let pixels = (src.width as u64) * (src.height as u64);
        let label = format!("{}({}x{})", src.name, src.width, src.height);

        baseline_q85.push(EncodedImage {
            label: label.clone(),
            jpeg: encode_jpeg(
                &src.pixels,
                src.width,
                src.height,
                85.0,
                ChromaSubsampling::Quarter,
                false,
            ),
            pixels,
        });
        baseline_q50.push(EncodedImage {
            label: label.clone(),
            jpeg: encode_jpeg(
                &src.pixels,
                src.width,
                src.height,
                50.0,
                ChromaSubsampling::Quarter,
                false,
            ),
            pixels,
        });
        progressive_q85.push(EncodedImage {
            label: label.clone(),
            jpeg: encode_jpeg(
                &src.pixels,
                src.width,
                src.height,
                85.0,
                ChromaSubsampling::Quarter,
                true,
            ),
            pixels,
        });
        baseline_444_q85.push(EncodedImage {
            label,
            jpeg: encode_jpeg(
                &src.pixels,
                src.width,
                src.height,
                85.0,
                ChromaSubsampling::None,
                false,
            ),
            pixels,
        });
    }

    // Print stats
    let total_bytes: usize = baseline_q85.iter().map(|e| e.jpeg.len()).sum();
    let total_pixels: u64 = baseline_q85.iter().map(|e| e.pixels).sum();
    eprintln!(
        "Test set ({tag}): {} images, {:.1}MP total, {:.1}MB baseline Q85 JPEG",
        sources.len(),
        total_pixels as f64 / 1e6,
        total_bytes as f64 / (1024.0 * 1024.0),
    );

    TestSet {
        baseline_q85,
        baseline_q50,
        progressive_q85,
        baseline_444_q85,
        source_tag: match tag {
            "CLIC2025" => "CLIC2025",
            "CID22" => "CID22",
            _ => "synthetic",
        },
    }
}

static TEST_SET: std::sync::OnceLock<TestSet> = std::sync::OnceLock::new();

fn get_test_set() -> &'static TestSet {
    TEST_SET.get_or_init(build_test_set)
}

// ── Benchmark groups ───────────────────────────────────────────────────────

fn bench_group(
    c: &mut Criterion,
    group_name: &str,
    images: &[EncodedImage],
) {
    let mut group = c.benchmark_group(group_name);

    // Aggregate benchmark: decode all images in one iteration
    let total_bytes: u64 = images.iter().map(|e| e.jpeg.len() as u64).sum();
    let total_pixels: u64 = images.iter().map(|e| e.pixels).sum();
    group.throughput(Throughput::Bytes(total_bytes));

    group.bench_function("zenjpeg_all", |b| {
        let decoder = Decoder::new().output_format(PixelFormat::Rgb);
        b.iter(|| {
            for img in images {
                decoder.decode(black_box(&img.jpeg), Unstoppable).unwrap();
            }
        });
    });

    eprintln!(
        "  {group_name}: {:.1}MP total, {:.1}MB JPEG, {:.2} bpp",
        total_pixels as f64 / 1e6,
        total_bytes as f64 / (1024.0 * 1024.0),
        (total_bytes * 8) as f64 / total_pixels as f64,
    );

    // Per-image benchmarks for the largest images (skip tiny ones)
    for img in images {
        if img.pixels < 200_000 {
            continue; // Skip images too small for reliable measurement
        }
        group.throughput(Throughput::Bytes(img.jpeg.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("zenjpeg", &img.label),
            &img.jpeg,
            |b, data| {
                let decoder = Decoder::new().output_format(PixelFormat::Rgb);
                b.iter(|| decoder.decode(black_box(data), Unstoppable).unwrap());
            },
        );
    }

    group.finish();
}

fn bench_baseline_420_q85(c: &mut Criterion) {
    let ts = get_test_set();
    bench_group(c, "huffdec/baseline_420_Q85", &ts.baseline_q85);
}

fn bench_baseline_420_q50(c: &mut Criterion) {
    let ts = get_test_set();
    bench_group(c, "huffdec/baseline_420_Q50", &ts.baseline_q50);
}

fn bench_progressive_420_q85(c: &mut Criterion) {
    let ts = get_test_set();
    bench_group(c, "huffdec/progressive_420_Q85", &ts.progressive_q85);
}

fn bench_baseline_444_q85(c: &mut Criterion) {
    let ts = get_test_set();
    bench_group(c, "huffdec/baseline_444_Q85", &ts.baseline_444_q85);
}

criterion_group!(
    benches,
    bench_baseline_420_q85,
    bench_baseline_420_q50,
    bench_progressive_420_q85,
    bench_baseline_444_q85,
);
criterion_main!(benches);
