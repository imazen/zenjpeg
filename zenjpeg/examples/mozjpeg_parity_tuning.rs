//! Per-image mozjpeg parity tuning: finds the zensim ceiling for matching
//! mozjpeg-rs output by extracting exact quant tables and sweeping trellis/
//! zero-bias parameters.
//!
//! Tests several strategies to isolate where the divergence comes from:
//!
//! | Config | What it tests |
//! |--------|---------------|
//! | moz-ref | mozjpeg-rs reference output |
//! | zen-preset | zenjpeg MozjpegProgressive preset (baseline) |
//! | zen-exact | zenjpeg with mozjpeg's extracted quant tables |
//! | zen-exact-notrel | extracted tables, trellis disabled |
//! | moz-notrel | mozjpeg-rs with trellis disabled (DCT floor reference) |
//! | zen-notrel | zenjpeg with extracted tables, no trellis (DCT floor) |
//!
//! The "DCT floor" comparison (zen-notrel vs moz-notrel) isolates the
//! fundamental f32-vs-integer DCT/color-conversion precision gap.
//!
//! Usage:
//! ```bash
//! cargo run --release -p zenjpeg --example mozjpeg_parity_tuning \
//!     --features "trellis decoder"
//!
//! # Single quality, more images
//! cargo run --release -p zenjpeg --example mozjpeg_parity_tuning \
//!     --features "trellis decoder" -- --quality 85 --images 10
//! ```

use enough::Unstoppable;
use std::path::PathBuf;
use zensim::{RgbSlice, Zensim, ZensimProfile};
use zensim_regress::testing::{RegressionTolerance, check_regression};

use zenjpeg::decoder::Decoder;
use zenjpeg::detect;
use zenjpeg::encode::tuning::{EncodingTables, PerComponent, ScalingParams};
use zenjpeg::encode::{
    ChromaSubsampling, EncoderConfig, OptimizationPreset, PixelLayout, ProgressiveScanMode,
    Quality, QuantTableConfig,
};
use zenjpeg_bench_utils::ImageData;

// ─── Configuration ──────────────────────────────────────────────────────────

const QUALITY_LEVELS: [u8; 6] = [50, 65, 75, 85, 90, 95];

// ─── Args ───────────────────────────────────────────────────────────────────

struct Args {
    corpus: PathBuf,
    max_images: usize,
    quality_filter: Option<u8>,
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
        max_images: 5,
        quality_filter: None,
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
                args.max_images = iter.next().and_then(|s| s.parse().ok()).unwrap_or(5);
            }
            "--quality" => {
                args.quality_filter = iter.next().and_then(|s| s.parse().ok());
            }
            "--help" | "-h" => {
                eprintln!("Usage: mozjpeg_parity_tuning [OPTIONS]");
                eprintln!("  --corpus <dir>   Image directory (default: gb82)");
                eprintln!("  --images <N>     Max images (default: 5)");
                eprintln!("  --quality <Q>    Single quality level to test");
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

fn encode_mozjpeg(pixels: &[u8], w: u32, h: u32, quality: u8, trellis: bool) -> Vec<u8> {
    let preset = if trellis {
        mozjpeg_rs::Preset::ProgressiveSmallest
    } else {
        mozjpeg_rs::Preset::BaselineFastest
    };
    let mut enc = mozjpeg_rs::Encoder::new(preset)
        .quality(quality)
        .subsampling(mozjpeg_rs::Subsampling::S420);
    if !trellis {
        enc = enc.optimize_huffman(true);
    }
    enc.encode_rgb(pixels, w, h)
        .expect("mozjpeg-rs encode failed")
}

/// Extract quant tables from a JPEG and build EncodingTables with Exact scaling.
fn extract_tables(jpeg: &[u8]) -> Box<EncodingTables> {
    let probe = detect::probe(jpeg).expect("failed to probe JPEG");
    let dqt = &probe.dqt_tables;

    // mozjpeg 4:2:0 typically has 2 tables: [0]=luma, [1]=chroma
    let luma: [f32; 64] = if let Some(t) = dqt.iter().find(|t| t.index == 0) {
        std::array::from_fn(|i| t.values[i] as f32)
    } else {
        panic!("no luma quant table (index 0) found");
    };

    let chroma: [f32; 64] = if let Some(t) = dqt.iter().find(|t| t.index == 1) {
        std::array::from_fn(|i| t.values[i] as f32)
    } else {
        // Fallback: use luma for chroma too
        luma
    };

    Box::new(EncodingTables {
        quant: PerComponent {
            c0: luma,
            c1: chroma,
            c2: chroma,
        },
        // Neutral zero-bias: standard rounding (matches mozjpeg)
        zero_bias_mul: PerComponent {
            c0: [0.0f32; 64],
            c1: [0.0f32; 64],
            c2: [0.0f32; 64],
        },
        zero_bias_offset_dc: [0.0, 0.0, 0.0],
        zero_bias_offset_ac: [0.5, 0.5, 0.5],
        scaling: ScalingParams::Exact,
    })
}

fn encode_zenjpeg_with_config(pixels: &[u8], w: u32, h: u32, config: &EncoderConfig) -> Vec<u8> {
    let mut encoder = config
        .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
        .expect("zenjpeg encoder creation failed");
    encoder
        .push_packed(pixels, Unstoppable)
        .expect("zenjpeg push_packed failed");
    encoder.finish().expect("zenjpeg finish failed")
}

fn decode_to_rgb(jpeg: &[u8]) -> (u32, u32, Vec<u8>) {
    let decoder = Decoder::new();
    let img = decoder.decode(jpeg, Unstoppable).expect("decode failed");
    let (w, h) = (img.width, img.height);
    (w, h, img.into_pixels_u8().unwrap())
}

fn as_rgb_pixels(rgb: &[u8]) -> &[[u8; 3]] {
    bytemuck::cast_slice(rgb)
}

// ─── Comparison ─────────────────────────────────────────────────────────────

struct Score {
    zensim: f64,
    max_delta: u8,
    pct_diff: f64,
    size_bytes: usize,
}

fn compare_decoded(
    zensim: &Zensim,
    ref_rgb: &[u8],
    test_rgb: &[u8],
    w: u32,
    h: u32,
    test_bytes: usize,
) -> Score {
    let tolerance = RegressionTolerance::off_by_one()
        .with_max_delta(255)
        .with_max_pixels_different(1.0)
        .with_min_similarity(0.0)
        .ignore_alpha();

    let ref_src = RgbSlice::new(as_rgb_pixels(ref_rgb), w as usize, h as usize);
    let test_src = RgbSlice::new(as_rgb_pixels(test_rgb), w as usize, h as usize);
    let report = check_regression(zensim, &ref_src, &test_src, &tolerance)
        .expect("zensim comparison failed");

    Score {
        zensim: report.score(),
        max_delta: *report.max_channel_delta().iter().max().unwrap_or(&0),
        pct_diff: report.pixels_differing() as f64 / report.pixel_count().max(1) as f64 * 100.0,
        size_bytes: test_bytes,
    }
}

// ─── Configs ────────────────────────────────────────────────────────────────

fn make_zen_preset(quality: u8) -> EncoderConfig {
    EncoderConfig::ycbcr(Quality::ApproxMozjpeg(quality), ChromaSubsampling::Quarter)
        .optimization(OptimizationPreset::MozjpegProgressive)
}

fn make_zen_exact(quality: u8, tables: Box<EncodingTables>, trellis: bool) -> EncoderConfig {
    let mut config =
        EncoderConfig::ycbcr(Quality::ApproxMozjpeg(quality), ChromaSubsampling::Quarter)
            .optimization(if trellis {
                OptimizationPreset::MozjpegProgressive
            } else {
                OptimizationPreset::JpegliBaseline
            });

    // Override with extracted tables
    config = config.quant_table_config(QuantTableConfig::Custom(tables));

    // If no trellis, also make it baseline and disable AQ
    if !trellis {
        config = config.progressive(ProgressiveScanMode::Baseline);
        config = config.aq_enabled(false);
    }

    config
}

// ─── Main ───────────────────────────────────────────────────────────────────

fn main() {
    let args = parse_args();

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

    let images: Vec<(String, ImageData)> = paths
        .iter()
        .filter_map(|p| {
            let name = p.file_stem()?.to_string_lossy().into_owned();
            ImageData::from_path(p).map(|img| (name, img))
        })
        .collect();

    eprintln!(
        "Loaded {} images from {}",
        images.len(),
        args.corpus.display()
    );

    let qualities: Vec<u8> = match args.quality_filter {
        Some(q) => vec![q],
        None => QUALITY_LEVELS.to_vec(),
    };

    let zensim = Zensim::new(ZensimProfile::latest());

    // ─── Header ─────────────────────────────────────────────────────────

    println!();
    println!(
        "{:<16} {:>3}  {:<18} {:>7} {:>5} {:>6} {:>7}  {:<18} {:>7} {:>5} {:>6} {:>7}",
        "", "", "vs moz+trellis", "", "", "", "", "vs moz-notrellis", "", "", "", ""
    );
    println!(
        "{:<16} {:>3}  {:<18} {:>7} {:>5} {:>6} {:>7}  {:<18} {:>7} {:>5} {:>6} {:>7}",
        "image",
        "Q",
        "config",
        "zensim",
        "maxΔ",
        "%diff",
        "size",
        "config",
        "zensim",
        "maxΔ",
        "%diff",
        "size"
    );
    println!("{}", "-".repeat(130));

    // Aggregated scores per config
    let mut agg: std::collections::BTreeMap<(String, u8), Vec<f64>> =
        std::collections::BTreeMap::new();

    for (img_name, img) in &images {
        let pixels = &img.pixels;
        let (w, h) = (img.width as u32, img.height as u32);

        for &quality in &qualities {
            // ── Reference encodes ──────────────────────────────────────
            let moz_trel = encode_mozjpeg(pixels, w, h, quality, true);
            let moz_notrel = encode_mozjpeg(pixels, w, h, quality, false);

            let (_, _, moz_trel_rgb) = decode_to_rgb(&moz_trel);
            let (_, _, moz_notrel_rgb) = decode_to_rgb(&moz_notrel);

            // Extract quant tables from the trellis and no-trellis outputs
            let tables_trel = extract_tables(&moz_trel);
            let tables_notrel = extract_tables(&moz_notrel);

            // ── Test configs vs moz+trellis ────────────────────────────
            let configs_vs_trel: Vec<(&str, EncoderConfig)> = vec![
                ("zen-preset", make_zen_preset(quality)),
                (
                    "zen-exact",
                    make_zen_exact(quality, tables_trel.clone(), true),
                ),
                (
                    "zen-exact-notrel",
                    make_zen_exact(quality, tables_trel.clone(), false),
                ),
            ];

            for (name, cfg) in &configs_vs_trel {
                let zen_jpeg = encode_zenjpeg_with_config(pixels, w, h, cfg);
                let (_, _, zen_rgb) = decode_to_rgb(&zen_jpeg);
                let s = compare_decoded(&zensim, &moz_trel_rgb, &zen_rgb, w, h, zen_jpeg.len());

                agg.entry((name.to_string(), quality))
                    .or_default()
                    .push(s.zensim);

                // Also compute the DCT floor (no trellis on both sides)
                let floor_cfg = make_zen_exact(quality, tables_notrel.clone(), false);
                let floor_jpeg = encode_zenjpeg_with_config(pixels, w, h, &floor_cfg);
                let (_, _, floor_rgb) = decode_to_rgb(&floor_jpeg);
                let f =
                    compare_decoded(&zensim, &moz_notrel_rgb, &floor_rgb, w, h, floor_jpeg.len());

                if *name == "zen-exact-notrel" {
                    // Print side-by-side: vs trellis | vs no-trellis (DCT floor)
                    println!(
                        "{:<16} {:>3}  {:<18} {:>7.2} {:>5} {:>5.1} {:>6.1}k  {:<18} {:>7.2} {:>5} {:>5.1} {:>6.1}k",
                        truncate(img_name, 16),
                        quality,
                        name,
                        s.zensim,
                        s.max_delta,
                        s.pct_diff,
                        s.size_bytes as f64 / 1024.0,
                        "zen-dct-floor",
                        f.zensim,
                        f.max_delta,
                        f.pct_diff,
                        f.size_bytes as f64 / 1024.0,
                    );

                    agg.entry(("zen-dct-floor".to_string(), quality))
                        .or_default()
                        .push(f.zensim);
                } else {
                    println!(
                        "{:<16} {:>3}  {:<18} {:>7.2} {:>5} {:>5.1} {:>6.1}k",
                        truncate(img_name, 16),
                        quality,
                        name,
                        s.zensim,
                        s.max_delta,
                        s.pct_diff,
                        s.size_bytes as f64 / 1024.0,
                    );
                }
            }
            println!();
        }
    }

    // ─── Summary ────────────────────────────────────────────────────────

    println!("=== Mean zensim by config x quality ===");
    println!();
    println!("{:<20} {:>3} {:>7} {:>5}", "config", "Q", "zensim", "n");
    println!("{}", "-".repeat(40));

    let mut config_names: Vec<String> = agg.keys().map(|(n, _)| n.clone()).collect();
    config_names.sort();
    config_names.dedup();

    for name in &config_names {
        for &q in &qualities {
            if let Some(scores) = agg.get(&(name.clone(), q)) {
                let mean = scores.iter().sum::<f64>() / scores.len() as f64;
                println!("{:<20} {:>3} {:>7.2} {:>5}", name, q, mean, scores.len());
            }
        }
    }

    // Grand mean per config
    println!();
    println!("{:<20} {:>7}", "config", "mean");
    println!("{}", "-".repeat(30));
    for name in &config_names {
        let all: Vec<f64> = agg
            .iter()
            .filter(|((n, _), _)| n == name)
            .flat_map(|(_, v)| v.iter().copied())
            .collect();
        if !all.is_empty() {
            let mean = all.iter().sum::<f64>() / all.len() as f64;
            println!("{:<20} {:>7.2}", name, mean);
        }
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max - 1])
    }
}
