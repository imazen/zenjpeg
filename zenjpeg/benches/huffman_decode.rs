//! Focused Huffman decode benchmark for measuring entropy decoding throughput.
//!
//! Uses CLIC 2025 photographs from codec-corpus. Requires the corpus to be
//! available (auto-downloaded on first run). Panics if no images are found.
//!
//! Tests baseline and progressive at multiple quality levels to exercise
//! different Huffman code length distributions:
//! - Q85: typical web quality, moderate AC density
//! - Q50: lower quality, sparser AC coefficients (more EOBs, fast_ac hits)
//!
//! Run:
//! ```sh
//! cargo bench -p zenjpeg --bench huffman_decode
//! ```

use enough::Unstoppable;
use std::path::{Path, PathBuf};
use zenbench::prelude::*;
use zenjpeg::decode::Decoder;
use zenjpeg::decoder::PixelFormat;
use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, PixelLayout};

// ── Image loading ──────────────────────────────────────────────────────────

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
    enc.push_packed(pixels, Unstoppable).expect("push_packed");
    enc.finish().expect("finish")
}

// ── Corpus loading ─────────────────────────────────────────────────────────

struct SourceImage {
    name: String,
    pixels: Vec<u8>,
    width: u32,
    height: u32,
}

struct EncodedSet {
    label: &'static str,
    images: Vec<(String, Vec<u8>, u64)>, // (name, jpeg, pixel_count)
}

struct TestSet {
    baseline_q85: EncodedSet,
    baseline_q50: EncodedSet,
    progressive_q85: EncodedSet,
    baseline_444_q85: EncodedSet,
}

fn load_corpus_images(max: usize) -> Vec<SourceImage> {
    let corpus = codec_corpus::Corpus::new().expect("codec-corpus init failed");

    // Try CLIC 2025 final-test, then training, then CID22
    let dir = corpus
        .get("clic2025/final-test")
        .or_else(|_| corpus.get("clic2025/training"))
        .or_else(|_| corpus.get("CID22/CID22-512/training"))
        .expect("No corpus available (need clic2025 or CID22)");

    let pngs = collect_pngs(&dir);
    assert!(
        !pngs.is_empty(),
        "No PNG files found in corpus directory: {dir:?}"
    );

    let images: Vec<SourceImage> = pngs
        .iter()
        .take(max)
        .filter_map(|p| {
            let (pixels, w, h) = load_png_rgb(p)?;
            let name = p.file_stem()?.to_string_lossy().into_owned();
            // Truncate hash names for readable output
            let short_name = if name.len() > 12 {
                format!("{}..{}", &name[..6], &name[name.len() - 6..])
            } else {
                name
            };
            Some(SourceImage {
                name: short_name,
                pixels,
                width: w,
                height: h,
            })
        })
        .collect();

    assert!(!images.is_empty(), "Failed to load any images from corpus");
    images
}

fn encode_set(
    sources: &[SourceImage],
    quality: f32,
    subsampling: ChromaSubsampling,
    progressive: bool,
    label: &'static str,
) -> EncodedSet {
    let images: Vec<_> = sources
        .iter()
        .map(|src| {
            let pixels = (src.width as u64) * (src.height as u64);
            let jpeg = encode_jpeg(
                &src.pixels,
                src.width,
                src.height,
                quality,
                subsampling,
                progressive,
            );
            let name = format!("{}({}x{})", src.name, src.width, src.height);
            (name, jpeg, pixels)
        })
        .collect();
    EncodedSet { label, images }
}

fn build_test_set() -> TestSet {
    let sources = load_corpus_images(8);

    eprintln!(
        "Loaded {} corpus images for Huffman decode benchmark:",
        sources.len()
    );
    for img in &sources {
        eprintln!("  {} ({}x{})", img.name, img.width, img.height);
    }

    let baseline_q85 = encode_set(
        &sources,
        85.0,
        ChromaSubsampling::Quarter,
        false,
        "baseline_420_Q85",
    );
    let baseline_q50 = encode_set(
        &sources,
        50.0,
        ChromaSubsampling::Quarter,
        false,
        "baseline_420_Q50",
    );
    let progressive_q85 = encode_set(
        &sources,
        85.0,
        ChromaSubsampling::Quarter,
        true,
        "progressive_420_Q85",
    );
    let baseline_444_q85 = encode_set(
        &sources,
        85.0,
        ChromaSubsampling::None,
        false,
        "baseline_444_Q85",
    );

    let total_bytes: usize = baseline_q85.images.iter().map(|(_, j, _)| j.len()).sum();
    let total_pixels: u64 = baseline_q85.images.iter().map(|(_, _, p)| p).sum();
    eprintln!(
        "Test set: {:.1}MP total, {:.1}MB baseline Q85 JPEG, {:.2} bpp",
        total_pixels as f64 / 1e6,
        total_bytes as f64 / (1024.0 * 1024.0),
        (total_bytes as f64 * 8.0) / total_pixels as f64,
    );

    TestSet {
        baseline_q85,
        baseline_q50,
        progressive_q85,
        baseline_444_q85,
    }
}

static TEST_SET: std::sync::OnceLock<TestSet> = std::sync::OnceLock::new();

fn get_test_set() -> &'static TestSet {
    TEST_SET.get_or_init(build_test_set)
}

// ── Benchmark ──────────────────────────────────────────────────────────────

fn bench_set(suite: &mut Suite, set: &'static EncodedSet) {
    let total_bytes: u64 = set.images.iter().map(|(_, j, _)| j.len() as u64).sum();

    // Aggregate benchmark: decode all images per iteration
    suite.group(format!("huffdec/{}", set.label), |g| {
        g.throughput(Throughput::Bytes(total_bytes));

        g.bench("zenjpeg", |b| {
            let decoder = Decoder::new().output_format(PixelFormat::Rgb);
            b.iter(|| {
                for (_, jpeg, _) in &set.images {
                    decoder.decode(jpeg, Unstoppable).unwrap();
                }
            });
        });
    });

    // Per-image benchmarks
    for (name, jpeg, _pixels) in &set.images {
        suite.group(format!("huffdec/{}/{name}", set.label), |g| {
            g.throughput(Throughput::Bytes(jpeg.len() as u64));

            g.bench("zenjpeg", {
                let jpeg = jpeg.clone();
                move |b| {
                    let decoder = Decoder::new().output_format(PixelFormat::Rgb);
                    b.iter(|| {
                        decoder.decode(&jpeg, Unstoppable).unwrap();
                    });
                }
            });
        });
    }
}

fn bench_all(suite: &mut Suite) {
    let ts = get_test_set();
    bench_set(suite, &ts.baseline_q85);
    bench_set(suite, &ts.baseline_q50);
    bench_set(suite, &ts.progressive_q85);
    bench_set(suite, &ts.baseline_444_q85);
}

zenbench::main!(bench_all);
