#![allow(clippy::len_zero, clippy::print_literal)]
//! Measure runtime and file size for each encoder configuration.
//!
//! Uses CID22 corpus (real photos). Reports wall-clock time and output
//! size for each (config, quality) combination.
//!
//! Run:
//! ```bash
//! cargo test --release -p zenjpeg --test effort_benchmark \
//!     --features "trellis decoder" -- --nocapture --ignored
//! ```

use enough::Unstoppable;
use std::path::{Path, PathBuf};
use std::time::Instant;
use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, OptimizationPreset, PixelLayout, Quality};

// ── Image loading ───────────────────────────────────────────────────────────

fn load_png_rgb(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let file = std::fs::File::open(path).ok()?;
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = decoder.read_info().ok()?;
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
        _ => return None,
    };
    Some((rgb, w, h))
}

fn collect_pngs(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
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

// ── Encoder configs ─────────────────────────────────────────────────────────

struct Mode {
    label: &'static str,
    desc: &'static str,
    make_config: fn(u8) -> EncoderConfig,
}

fn cfg_default(q: u8) -> EncoderConfig {
    EncoderConfig::ycbcr(q, ChromaSubsampling::Quarter)
}

fn cfg_fast(q: u8) -> EncoderConfig {
    EncoderConfig::ycbcr(q, ChromaSubsampling::Quarter)
        .optimization(OptimizationPreset::JpegliBaseline)
}

fn cfg_balanced(q: u8) -> EncoderConfig {
    EncoderConfig::ycbcr(q, ChromaSubsampling::Quarter)
        .optimization(OptimizationPreset::HybridProgressive)
}

fn cfg_max(q: u8) -> EncoderConfig {
    EncoderConfig::ycbcr(q, ChromaSubsampling::Quarter)
        .optimization(OptimizationPreset::HybridMaxCompression)
}

fn cfg_auto(q: u8) -> EncoderConfig {
    EncoderConfig::ycbcr(q, ChromaSubsampling::Quarter).auto_optimize(true)
}

fn cfg_mozjpeg_prog(q: u8) -> EncoderConfig {
    EncoderConfig::ycbcr(Quality::ApproxMozjpeg(q), ChromaSubsampling::Quarter)
        .optimization(OptimizationPreset::MozjpegProgressive)
}

const MODES: [Mode; 6] = [
    Mode {
        label: "default",
        desc: "ycbcr(Q) — progressive + AQ, no trellis, jpegli tables",
        make_config: cfg_default,
    },
    Mode {
        label: "fast",
        desc: "Effort::Fast — baseline + AQ, no trellis",
        make_config: cfg_fast,
    },
    Mode {
        label: "balanced",
        desc: "Effort::Balanced — progressive + AQ + adaptive trellis",
        make_config: cfg_balanced,
    },
    Mode {
        label: "max",
        desc: "Effort::Max — progressive + scan search + AQ + thorough trellis",
        make_config: cfg_max,
    },
    Mode {
        label: "auto",
        desc: "auto_optimize(true) — progressive + AQ + hybrid trellis λ=14.5",
        make_config: cfg_auto,
    },
    Mode {
        label: "moz-prog",
        desc: "ApproxMozjpeg(Q) + MozjpegProgressive — trellis, no AQ",
        make_config: cfg_mozjpeg_prog,
    },
];

// ── Encode + measure ────────────────────────────────────────────────────────

fn encode_once(
    pixels: &[u8],
    w: u32,
    h: u32,
    config: EncoderConfig,
) -> (usize, std::time::Duration) {
    let start = Instant::now();
    let mut enc = config
        .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
        .expect("encoder creation failed");
    enc.push_packed(pixels, Unstoppable).expect("push failed");
    let jpeg = enc.finish().expect("finish failed");
    (jpeg.len(), start.elapsed())
}

fn encode_mozjpeg_rs(pixels: &[u8], w: u32, h: u32, quality: u8) -> (usize, std::time::Duration) {
    let start = Instant::now();
    let jpeg = mozjpeg_rs::Encoder::new(mozjpeg_rs::Preset::ProgressiveSmallest)
        .quality(quality)
        .subsampling(mozjpeg_rs::Subsampling::S420)
        .encode_rgb(pixels, w, h)
        .expect("mozjpeg-rs encode failed");
    (jpeg.len(), start.elapsed())
}

// ── Test ────────────────────────────────────────────────────────────────────

#[test]
#[ignore = "requires CID22 corpus and trellis feature"]
fn effort_size_and_speed() {
    let corpus = match codec_corpus::Corpus::new() {
        Ok(c) => c,
        Err(e) => {
            println!("codec-corpus init failed: {e}");
            return;
        }
    };
    let dir = match corpus.get("CID22/CID22-512/training") {
        Ok(d) => d,
        Err(_) => {
            println!("CID22 not available");
            return;
        }
    };

    // Load 25 images (consistent subset for timing)
    let paths = collect_pngs(&dir);
    let images: Vec<(Vec<u8>, u32, u32)> = paths
        .iter()
        .take(25)
        .filter_map(|p| load_png_rgb(p))
        .collect();
    let n = images.len();
    println!("Loaded {n} images (512x512) from CID22\n");

    let qualities: [u8; 4] = [50, 75, 85, 95];

    // Header
    println!("  Modes:");
    for m in &MODES {
        println!("    {:<12} {}", m.label, m.desc);
    }
    println!(
        "    {:<12} {}",
        "mozjpeg-rs", "C mozjpeg via mozjpeg-rs | ProgressiveSmallest | 4:2:0"
    );
    println!();

    // Per-quality comparison
    for &q in &qualities {
        println!("=== Q{q} ({n} images, 512x512) ===\n");
        println!(
            "  {:<12} {:>8} {:>8} {:>8} {:>8}",
            "mode", "mean_kb", "vs_dflt", "mean_ms", "vs_dflt"
        );
        println!("  {}", "-".repeat(52));

        let mut default_kb = 0.0f64;
        let mut default_ms = 0.0f64;

        // zenjpeg modes
        for mode in &MODES {
            let mut total_bytes = 0usize;
            let mut total_us = 0u128;

            for (pixels, w, h) in &images {
                let config = (mode.make_config)(q);
                let (sz, dur) = encode_once(pixels, *w, *h, config);
                total_bytes += sz;
                total_us += dur.as_micros();
            }

            let mean_kb = total_bytes as f64 / n as f64 / 1024.0;
            let mean_ms = total_us as f64 / n as f64 / 1000.0;

            if mode.label == "default" {
                default_kb = mean_kb;
                default_ms = mean_ms;
                println!(
                    "  {:<12} {:>7.1}k {:>8} {:>7.1}ms {:>8}",
                    mode.label, mean_kb, "—", mean_ms, "—"
                );
            } else {
                let sz_pct = (mean_kb / default_kb - 1.0) * 100.0;
                let spd_x = mean_ms / default_ms;
                println!(
                    "  {:<12} {:>7.1}k {:>+7.1}% {:>7.1}ms {:>7.2}x",
                    mode.label, mean_kb, sz_pct, mean_ms, spd_x
                );
            }
        }

        // mozjpeg-rs baseline
        {
            let mut total_bytes = 0usize;
            let mut total_us = 0u128;
            for (pixels, w, h) in &images {
                let (sz, dur) = encode_mozjpeg_rs(pixels, *w, *h, q);
                total_bytes += sz;
                total_us += dur.as_micros();
            }
            let mean_kb = total_bytes as f64 / n as f64 / 1024.0;
            let mean_ms = total_us as f64 / n as f64 / 1000.0;
            let sz_pct = (mean_kb / default_kb - 1.0) * 100.0;
            let spd_x = mean_ms / default_ms;
            println!(
                "  {:<12} {:>7.1}k {:>+7.1}% {:>7.1}ms {:>7.2}x",
                "mozjpeg-rs", mean_kb, sz_pct, mean_ms, spd_x
            );
        }

        println!();
    }

    // Summary: what does auto_optimize cost?
    println!("=== auto_optimize overhead (mean across all Q levels) ===\n");

    let mut auto_size_sum = 0.0f64;
    let mut auto_speed_sum = 0.0f64;
    let mut count = 0u32;

    for &q in &qualities {
        let mut default_bytes = 0usize;
        let mut default_us = 0u128;
        let mut auto_bytes = 0usize;
        let mut auto_us = 0u128;

        for (pixels, w, h) in &images {
            let (sz, dur) = encode_once(pixels, *w, *h, cfg_default(q));
            default_bytes += sz;
            default_us += dur.as_micros();

            let (sz, dur) = encode_once(pixels, *w, *h, cfg_auto(q));
            auto_bytes += sz;
            auto_us += dur.as_micros();
        }

        let sz_ratio = auto_bytes as f64 / default_bytes as f64;
        let spd_ratio = auto_us as f64 / default_us as f64;
        auto_size_sum += sz_ratio;
        auto_speed_sum += spd_ratio;
        count += 1;

        println!(
            "  Q{q:<2}: auto size={sz_ratio:.4}x ({:+.1}%), speed={spd_ratio:.2}x",
            (sz_ratio - 1.0) * 100.0
        );
    }

    let avg_sz = auto_size_sum / count as f64;
    let avg_spd = auto_speed_sum / count as f64;
    println!(
        "\n  Mean: auto_optimize size={avg_sz:.4}x ({:+.1}%), speed={avg_spd:.2}x vs default",
        (avg_sz - 1.0) * 100.0
    );

    println!("\nDone.");
}
