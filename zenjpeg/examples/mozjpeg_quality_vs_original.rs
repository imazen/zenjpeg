//! Compare mozjpeg-rs and zenjpeg decoded quality against the uncompressed original.
//!
//! For each (image, quality) pair, encodes with both encoders, decodes both,
//! and measures zensim score of each against the original source pixels.
//! Shows that the quality difference between the two encoders is negligible
//! compared to the compression loss itself.
//!
//! Usage:
//! ```bash
//! cargo run --release -p zenjpeg --example mozjpeg_quality_vs_original \
//!     --features "trellis decoder"
//!
//! cargo run --release -p zenjpeg --example mozjpeg_quality_vs_original \
//!     --features "trellis decoder" -- --images 25
//! ```

use enough::Unstoppable;
use std::path::PathBuf;
use zensim::{RgbSlice, Zensim, ZensimProfile};

use zenjpeg::decoder::Decoder;
use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, OptimizationPreset, PixelLayout, Quality};
use zenjpeg_bench_utils::ImageData;

const QUALITY_LEVELS: [u8; 8] = [50, 65, 75, 80, 85, 90, 95, 98];

struct Args {
    corpus: PathBuf,
    max_images: usize,
    quality_filter: Option<u8>,
}

fn parse_args() -> Args {
    let mut args = Args {
        corpus: codec_corpus::Corpus::new()
            .expect("codec-corpus unavailable")
            .get("gb82")
            .expect("gb82 not found"),
        max_images: 10,
        quality_filter: None,
    };
    let mut iter = std::env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--corpus" => {
                if let Some(s) = iter.next() {
                    let p = if s.starts_with('~') {
                        if let Some(h) = std::env::var_os("HOME") {
                            PathBuf::from(h).join(&s[2..])
                        } else {
                            PathBuf::from(s)
                        }
                    } else {
                        PathBuf::from(s)
                    };
                    args.corpus = p;
                }
            }
            "--images" => {
                args.max_images = iter.next().and_then(|s| s.parse().ok()).unwrap_or(10);
            }
            "--quality" => {
                args.quality_filter = iter.next().and_then(|s| s.parse().ok());
            }
            "--help" | "-h" => {
                eprintln!("Usage: mozjpeg_quality_vs_original [OPTIONS]");
                eprintln!("  --corpus <dir>   Image directory (default: gb82)");
                eprintln!("  --images <N>     Max images (default: 10)");
                eprintln!("  --quality <Q>    Single quality level");
                std::process::exit(0);
            }
            _ => {
                eprintln!("Unknown argument: {arg}");
                std::process::exit(1);
            }
        }
    }
    args
}

fn as_rgb_pixels(rgb: &[u8]) -> &[[u8; 3]] {
    bytemuck::cast_slice(rgb)
}

fn encode_mozjpeg(pixels: &[u8], w: u32, h: u32, quality: u8) -> Vec<u8> {
    mozjpeg_rs::Encoder::new(mozjpeg_rs::Preset::ProgressiveSmallest)
        .quality(quality)
        .subsampling(mozjpeg_rs::Subsampling::S420)
        .encode_rgb(pixels, w, h)
        .expect("mozjpeg-rs encode failed")
}

fn encode_zenjpeg(pixels: &[u8], w: u32, h: u32, quality: u8) -> Vec<u8> {
    let config = EncoderConfig::ycbcr(Quality::ApproxMozjpeg(quality), ChromaSubsampling::Quarter)
        .optimization(OptimizationPreset::MozjpegProgressive);
    let mut enc = config
        .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
        .expect("encoder creation failed");
    enc.push_packed(pixels, Unstoppable).expect("push failed");
    enc.finish().expect("finish failed")
}

fn decode_to_rgb(jpeg: &[u8]) -> Vec<u8> {
    let dec = Decoder::new().apply_icc(false);
    let img = dec.decode(jpeg, Unstoppable).expect("decode failed");
    img.into_pixels_u8().unwrap()
}

fn zensim_score(z: &Zensim, a: &[u8], b: &[u8], w: usize, h: usize) -> f64 {
    let sa = RgbSlice::new(as_rgb_pixels(a), w, h);
    let sb = RgbSlice::new(as_rgb_pixels(b), w, h);
    z.compute(&sa, &sb).expect("zensim failed").score()
}

struct Row {
    image: String,
    quality: u8,
    moz_kb: f64,
    zen_kb: f64,
    moz_vs_orig: f64,
    zen_vs_orig: f64,
    delta: f64, // zen - moz (positive = zen better)
}

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
    let mut rows: Vec<Row> = Vec::new();

    for (name, img) in &images {
        let pixels = &img.pixels;
        let (w, h) = (img.width as u32, img.height as u32);

        for &q in &qualities {
            let moz_jpeg = encode_mozjpeg(pixels, w, h, q);
            let zen_jpeg = encode_zenjpeg(pixels, w, h, q);

            let moz_dec = decode_to_rgb(&moz_jpeg);
            let zen_dec = decode_to_rgb(&zen_jpeg);

            let moz_score = zensim_score(&zensim, pixels, &moz_dec, w as usize, h as usize);
            let zen_score = zensim_score(&zensim, pixels, &zen_dec, w as usize, h as usize);

            rows.push(Row {
                image: name.clone(),
                quality: q,
                moz_kb: moz_jpeg.len() as f64 / 1024.0,
                zen_kb: zen_jpeg.len() as f64 / 1024.0,
                moz_vs_orig: moz_score,
                zen_vs_orig: zen_score,
                delta: zen_score - moz_score,
            });
        }
    }

    // Per-image table
    println!();
    println!(
        "{:<18} {:>3} {:>7} {:>7} {:>8} {:>8} {:>7}",
        "image", "Q", "moz_kb", "zen_kb", "moz→orig", "zen→orig", "Δ(z-m)"
    );
    println!("{}", "-".repeat(65));

    for r in &rows {
        println!(
            "{:<18} {:>3} {:>7.1} {:>7.1} {:>8.2} {:>8.2} {:>+7.2}",
            truncate(&r.image, 18),
            r.quality,
            r.moz_kb,
            r.zen_kb,
            r.moz_vs_orig,
            r.zen_vs_orig,
            r.delta,
        );
    }

    // Summary by quality
    println!();
    println!("=== Mean across {} images ===", images.len());
    println!();
    println!(
        "{:>3} {:>7} {:>7} {:>8} {:>8} {:>7} {:>7}",
        "Q", "moz_kb", "zen_kb", "moz→orig", "zen→orig", "Δ(z-m)", "size_Δ%"
    );
    println!("{}", "-".repeat(60));

    for &q in &qualities {
        let qrows: Vec<&Row> = rows.iter().filter(|r| r.quality == q).collect();
        let n = qrows.len() as f64;
        let moz_kb = qrows.iter().map(|r| r.moz_kb).sum::<f64>() / n;
        let zen_kb = qrows.iter().map(|r| r.zen_kb).sum::<f64>() / n;
        let moz_s = qrows.iter().map(|r| r.moz_vs_orig).sum::<f64>() / n;
        let zen_s = qrows.iter().map(|r| r.zen_vs_orig).sum::<f64>() / n;
        let delta = qrows.iter().map(|r| r.delta).sum::<f64>() / n;
        let size_pct = (zen_kb / moz_kb - 1.0) * 100.0;
        println!(
            "{:>3} {:>7.1} {:>7.1} {:>8.2} {:>8.2} {:>+7.2} {:>+6.1}%",
            q, moz_kb, zen_kb, moz_s, zen_s, delta, size_pct,
        );
    }

    // Wins/ties/losses
    let wins = rows.iter().filter(|r| r.delta > 0.1).count();
    let losses = rows.iter().filter(|r| r.delta < -0.1).count();
    let ties = rows.len() - wins - losses;
    println!();
    println!(
        "zen wins: {} | ties (±0.1): {} | moz wins: {} | total: {}",
        wins,
        ties,
        losses,
        rows.len()
    );
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max - 1])
    }
}
