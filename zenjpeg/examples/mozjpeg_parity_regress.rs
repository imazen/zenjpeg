//! Mozjpeg parity regression: measures how closely zenjpeg's MozjpegPreset
//! output matches mozjpeg-rs, and version-tracks the decoded pixels via
//! zensim-regress checksums.
//!
//! For each (image, quality, preset) triple this example:
//! 1. Encodes with mozjpeg-rs (reference)
//! 2. Encodes with zenjpeg MozjpegPreset (test)
//! 3. Decodes both with zenjpeg's decoder (constant decoder, isolating encoder diffs)
//! 4. Compares decoded outputs via zensim `check_regression`
//! 5. Tracks the zenjpeg decoded output via `ChecksumManager` for version-locking
//!
//! Usage:
//! ```bash
//! # Default: gb82 corpus, 10 images
//! cargo run --release -p zenjpeg --example mozjpeg_parity_regress \
//!     --features "trellis decoder"
//!
//! # First run / after intentional changes — create baselines:
//! UPDATE_CHECKSUMS=1 cargo run --release -p zenjpeg --example mozjpeg_parity_regress \
//!     --features "trellis decoder"
//!
//! # CID22 corpus
//! cargo run --release -p zenjpeg --example mozjpeg_parity_regress \
//!     --features "trellis decoder" -- \
//!     --corpus ~/work/codec-eval/codec-corpus/CID22/CID22-512/validation
//!
//! # Single quality, more images
//! cargo run --release -p zenjpeg --example mozjpeg_parity_regress \
//!     --features "trellis decoder" -- --quality 85 --images 25
//! ```

use enough::Unstoppable;
use std::path::PathBuf;
use zensim::{RgbSlice, Zensim, ZensimProfile};
use zensim_regress::Tolerance;
use zensim_regress::checksums::ChecksumManager;
use zensim_regress::testing::{RegressionTolerance, check_regression};

use std::collections::BTreeMap;
use zenjpeg::decoder::Decoder;
use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, OptimizationPreset, PixelLayout, Quality};
use zenjpeg_bench_utils::ImageData;

// ─── Configuration ──────────────────────────────────────────────────────────

const QUALITY_LEVELS: [u8; 8] = [50, 65, 75, 80, 85, 90, 95, 98];

struct Preset {
    name: &'static str,
    opt: OptimizationPreset,
    progressive: bool,
}

const PRESETS: [Preset; 3] = [
    Preset {
        name: "moz-base",
        opt: OptimizationPreset::MozjpegBaseline,
        progressive: false,
    },
    Preset {
        name: "moz-prog",
        opt: OptimizationPreset::MozjpegProgressive,
        progressive: true,
    },
    Preset {
        name: "moz-max",
        opt: OptimizationPreset::MozjpegMaxCompression,
        progressive: true,
    },
];

// ─── Args ───────────────────────────────────────────────────────────────────

struct Args {
    corpus: PathBuf,
    max_images: usize,
    quality_filter: Option<u8>,
    checksums_dir: PathBuf,
    no_checksums: bool,
}

fn default_corpus_dir() -> PathBuf {
    codec_corpus::Corpus::new()
        .expect("codec-corpus unavailable")
        .get("gb82")
        .expect("gb82 corpus not found")
}

fn parse_args() -> Args {
    let mut args = Args {
        corpus: default_corpus_dir(),
        max_images: 10,
        quality_filter: None,
        checksums_dir: PathBuf::from("/mnt/v/output/zenjpeg/mozjpeg_parity/checksums"),
        no_checksums: false,
    };
    let mut iter = std::env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--corpus" => {
                if let Some(s) = iter.next() {
                    let expanded = if s.starts_with('~') {
                        if let Some(home) = std::env::var_os("HOME") {
                            PathBuf::from(home).join(&s[2..])
                        } else {
                            PathBuf::from(s)
                        }
                    } else {
                        PathBuf::from(s)
                    };
                    args.corpus = expanded;
                }
            }
            "--images" => {
                args.max_images = iter.next().and_then(|s| s.parse().ok()).unwrap_or(10);
            }
            "--quality" => {
                args.quality_filter = iter.next().and_then(|s| s.parse().ok());
            }
            "--checksums" => {
                if let Some(s) = iter.next() {
                    args.checksums_dir = PathBuf::from(s);
                }
            }
            "--no-checksums" => args.no_checksums = true,
            "--help" | "-h" => {
                eprintln!("Usage: mozjpeg_parity_regress [OPTIONS]");
                eprintln!("  --corpus <dir>       Image directory (default: gb82)");
                eprintln!("  --images <N>         Max images (default: 10)");
                eprintln!("  --quality <Q>        Single quality level to test");
                eprintln!("  --checksums <dir>    Checksums dir (default: /mnt/v/output/...)");
                eprintln!("  --no-checksums       Skip checksum tracking");
                eprintln!();
                eprintln!("Environment:");
                eprintln!("  UPDATE_CHECKSUMS=1   Create/update checksum baselines");
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {other}");
                std::process::exit(1);
            }
        }
    }
    args
}

// ─── Encoding helpers ───────────────────────────────────────────────────────

fn encode_mozjpeg(pixels: &[u8], w: usize, h: usize, quality: u8, progressive: bool) -> Vec<u8> {
    let preset = if progressive {
        mozjpeg_rs::Preset::ProgressiveSmallest
    } else {
        mozjpeg_rs::Preset::BaselineBalanced
    };
    mozjpeg_rs::Encoder::new(preset)
        .quality(quality)
        .subsampling(mozjpeg_rs::Subsampling::S420)
        .encode_rgb(pixels, w as u32, h as u32)
        .expect("mozjpeg-rs encode failed")
}

fn encode_zenjpeg(pixels: &[u8], w: usize, h: usize, quality: u8, preset: &Preset) -> Vec<u8> {
    let config = EncoderConfig::ycbcr(Quality::ApproxMozjpeg(quality), ChromaSubsampling::Quarter)
        .optimization(preset.opt);

    let mut encoder = config
        .encode_from_bytes(w as u32, h as u32, PixelLayout::Rgb8Srgb)
        .expect("zenjpeg encoder creation failed");
    encoder
        .push_packed(pixels, Unstoppable)
        .expect("zenjpeg push_packed failed");
    encoder.finish().expect("zenjpeg finish failed")
}

fn decode_to_rgb(jpeg: &[u8]) -> (u32, u32, Vec<u8>) {
    let decoder = Decoder::new().apply_icc(false);
    let img = decoder.decode(jpeg, Unstoppable).expect("decode failed");
    let (w, h) = (img.width, img.height);
    (w, h, img.into_pixels_u8().unwrap())
}

fn rgb_to_rgba(rgb: &[u8]) -> Vec<u8> {
    let n = rgb.len() / 3;
    let mut rgba = Vec::with_capacity(n * 4);
    for i in 0..n {
        rgba.push(rgb[i * 3]);
        rgba.push(rgb[i * 3 + 1]);
        rgba.push(rgb[i * 3 + 2]);
        rgba.push(255);
    }
    rgba
}

/// Reinterpret flat `&[u8]` RGB as `&[[u8; 3]]` for zensim.
fn as_rgb_pixels(rgb: &[u8]) -> &[[u8; 3]] {
    assert_eq!(rgb.len() % 3, 0);
    // SAFETY: [u8; 3] has same layout as 3 consecutive u8s, no alignment requirement.
    // Using bytemuck for zero-copy reinterpret.
    bytemuck::cast_slice(rgb)
}

// ─── Result aggregation ─────────────────────────────────────────────────────

struct ComparisonResult {
    image_name: String,
    preset_name: String,
    quality: u8,
    moz_bytes: usize,
    zen_bytes: usize,
    size_ratio: f64,
    zensim_score: f64,
    max_delta: [u8; 3],
    pct_diff: f64,
    category: String,
}

// ─── Main ───────────────────────────────────────────────────────────────────

fn main() {
    let args = parse_args();

    // Load images
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&args.corpus)
        .unwrap_or_else(|e| {
            eprintln!("Cannot read {}: {e}", args.corpus.display());
            std::process::exit(1);
        })
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .is_some_and(|x| x.eq_ignore_ascii_case("png"))
        })
        .map(|e| e.path())
        .collect();
    paths.sort();
    paths.truncate(args.max_images);

    if paths.is_empty() {
        eprintln!("No PNG files found in {}", args.corpus.display());
        std::process::exit(1);
    }
    eprintln!(
        "Loading {} images from {}",
        paths.len(),
        args.corpus.display()
    );

    let images: Vec<(String, ImageData)> = paths
        .iter()
        .filter_map(|p| {
            let name = p.file_stem()?.to_string_lossy().into_owned();
            ImageData::from_path(p).map(|img| (name, img))
        })
        .collect();

    eprintln!("Loaded {} images", images.len());

    let qualities: Vec<u8> = match args.quality_filter {
        Some(q) => vec![q],
        None => QUALITY_LEVELS.to_vec(),
    };

    // Set up zensim and optional checksum manager
    let zensim = Zensim::new(ZensimProfile::latest());

    let mgr = if !args.no_checksums {
        std::fs::create_dir_all(&args.checksums_dir).ok();
        Some(ChecksumManager::new(&args.checksums_dir))
    } else {
        None
    };

    // Lossy cross-encoder tolerance: large deltas expected, track perceptual similarity
    let tolerance = RegressionTolerance::off_by_one()
        .with_max_delta(255)
        .with_max_pixels_different(1.0)
        .with_min_similarity(40.0)
        .ignore_alpha();

    let checksum_tol = Tolerance {
        max_delta: 255,
        min_similarity: 40.0,
        max_pixels_different: 1.0,
        max_alpha_delta: 0,
        ignore_alpha: true,
        overrides: BTreeMap::new(),
    };

    let mut results: Vec<ComparisonResult> = Vec::new();

    for (img_name, img) in &images {
        let pixels = &img.pixels;
        let (w, h) = (img.width, img.height);

        for &quality in &qualities {
            for preset in &PRESETS {
                // Encode with both
                let moz_jpeg = encode_mozjpeg(pixels, w, h, quality, preset.progressive);
                let zen_jpeg = encode_zenjpeg(pixels, w, h, quality, preset);

                // Decode both with the same decoder
                let (mw, mh, moz_rgb) = decode_to_rgb(&moz_jpeg);
                let (zw, zh, zen_rgb) = decode_to_rgb(&zen_jpeg);

                if mw != zw || mh != zh {
                    eprintln!(
                        "  SKIP {img_name} Q{quality} {}: dimension mismatch \
                         moz={mw}x{mh} zen={zw}x{zh}",
                        preset.name
                    );
                    continue;
                }

                // Compare decoded outputs with zensim
                let moz_src = RgbSlice::new(as_rgb_pixels(&moz_rgb), mw as usize, mh as usize);
                let zen_src = RgbSlice::new(as_rgb_pixels(&zen_rgb), zw as usize, zh as usize);
                let report = check_regression(&zensim, &moz_src, &zen_src, &tolerance)
                    .expect("zensim comparison failed");

                // Version-track the zenjpeg output via checksums
                if let Some(mgr) = &mgr {
                    let zen_rgba = rgb_to_rgba(&zen_rgb);
                    let detail = format!("q{quality}-420");
                    let check = mgr.check_pixels(
                        &format!("mozjpeg_parity_{}", preset.name),
                        img_name,
                        &detail,
                        &zen_rgba,
                        zw,
                        zh,
                        Some(&checksum_tol),
                    );
                    if let Err(e) = &check {
                        eprintln!(
                            "  checksum error for {img_name} Q{quality} {}: {e}",
                            preset.name
                        );
                    }
                }

                let cr = ComparisonResult {
                    image_name: img_name.clone(),
                    preset_name: preset.name.to_string(),
                    quality,
                    moz_bytes: moz_jpeg.len(),
                    zen_bytes: zen_jpeg.len(),
                    size_ratio: zen_jpeg.len() as f64 / moz_jpeg.len() as f64,
                    zensim_score: report.score(),
                    max_delta: report.max_channel_delta(),
                    pct_diff: report.pixels_differing() as f64 / report.pixel_count().max(1) as f64
                        * 100.0,
                    category: format!("{:?}", report.category()),
                };

                results.push(cr);
            }
        }
    }

    // ─── Print per-comparison table ─────────────────────────────────────

    println!();
    println!(
        "{:<20} {:<10} {:>3} {:>8} {:>8} {:>6} {:>7} {:>9} {:>6} {:<12}",
        "image",
        "preset",
        "Q",
        "moz_kb",
        "zen_kb",
        "ratio",
        "zensim",
        "max_delta",
        "%diff",
        "category"
    );
    println!("{}", "-".repeat(105));

    for r in &results {
        println!(
            "{:<20} {:<10} {:>3} {:>8.1} {:>8.1} {:>6.3} {:>7.2} {:>3},{:>2},{:>2} {:>6.1} {:<12}",
            truncate(&r.image_name, 20),
            r.preset_name,
            r.quality,
            r.moz_bytes as f64 / 1024.0,
            r.zen_bytes as f64 / 1024.0,
            r.size_ratio,
            r.zensim_score,
            r.max_delta[0],
            r.max_delta[1],
            r.max_delta[2],
            r.pct_diff,
            r.category,
        );
    }

    // ─── Print summary by preset × quality ──────────────────────────────

    println!();
    println!("=== Summary (mean across images) ===");
    println!();
    println!(
        "{:<10} {:>3} {:>6} {:>7} {:>7} {:>7}",
        "preset", "Q", "ratio", "zensim", "maxΔ", "%diff"
    );
    println!("{}", "-".repeat(50));

    for preset in &PRESETS {
        for &quality in &qualities {
            let matching: Vec<&ComparisonResult> = results
                .iter()
                .filter(|r| r.preset_name == preset.name && r.quality == quality)
                .collect();
            if matching.is_empty() {
                continue;
            }
            let n = matching.len() as f64;
            let avg_ratio = matching.iter().map(|r| r.size_ratio).sum::<f64>() / n;
            let avg_score = matching.iter().map(|r| r.zensim_score).sum::<f64>() / n;
            let max_delta = matching
                .iter()
                .map(|r| *r.max_delta.iter().max().unwrap())
                .max()
                .unwrap_or(0);
            let avg_pct = matching.iter().map(|r| r.pct_diff).sum::<f64>() / n;

            println!(
                "{:<10} {:>3} {:>6.3} {:>7.2} {:>5}   {:>6.1}",
                preset.name, quality, avg_ratio, avg_score, max_delta, avg_pct,
            );
        }
    }

    // ─── Final status ───────────────────────────────────────────────────

    let min_score = results
        .iter()
        .map(|r| r.zensim_score)
        .fold(f64::INFINITY, f64::min);
    let max_ratio = results.iter().map(|r| r.size_ratio).fold(0.0f64, f64::max);
    let worst_delta = results
        .iter()
        .flat_map(|r| r.max_delta.iter().copied())
        .max()
        .unwrap_or(0);

    println!();
    println!("--- Overall ---");
    println!("  Worst zensim score:  {min_score:.2}");
    println!("  Worst size ratio:    {max_ratio:.3}x");
    println!("  Worst max delta:     {worst_delta}");
    println!("  Total comparisons:   {}", results.len());

    if std::env::var("UPDATE_CHECKSUMS").is_ok_and(|v| v == "1") {
        println!();
        println!("Checksums updated in {}", args.checksums_dir.display());
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max - 1])
    }
}
