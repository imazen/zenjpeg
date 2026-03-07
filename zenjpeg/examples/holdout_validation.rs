//! Holdout validation: test corpus-trained Huffman tables on unseen images.
//!
//! Tests on images that were NOT part of the training set:
//! - CID22-validation (41 images, not in CID22-training)
//! - CLIC2025-final-test (30 images, not in clic2025-training)
//!
//! Also compares JpegliCreateTree vs MozjpegClassic algorithms
//! for generating corpus tables from the same aggregated frequencies.
//!
//! Run with:
//!   cargo run --release -p zenjpeg --example holdout_validation

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use enough::Unstoppable;
use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, HuffmanSymbolFrequencies, PixelLayout};
use zenjpeg::huffman::optimize::{FrequencyCounter, HuffmanTableSet};
use zenjpeg::types::HuffmanMethod;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn corpus_crate() -> std::result::Result<codec_corpus::Corpus, Box<dyn std::error::Error>> {
    Ok(codec_corpus::Corpus::new()?)
}
fn freq_dir() -> std::path::PathBuf {
    zenjpeg_bench_utils::zenjpeg_output_dir().join("huffman-freq")
}

const VALIDATION_QUALITIES: &[u8] = &[50, 75, 85, 90, 95];

struct ImageSet {
    name: &'static str,
    rel_path: &'static str,
    in_training: bool,
}

const IMAGE_SETS: &[ImageSet] = &[
    // HOLDOUT sets - never used for training
    ImageSet {
        name: "CID22-validation",
        rel_path: "CID22/CID22-512/validation",
        in_training: false,
    },
    ImageSet {
        name: "CLIC2025-final-test",
        rel_path: "clic2025/final-test",
        in_training: false,
    },
    // TRAINING sets - for comparison (expect slightly better fit)
    ImageSet {
        name: "CID22-training",
        rel_path: "CID22/CID22-512/training",
        in_training: true,
    },
    ImageSet {
        name: "kodak-legacy",
        rel_path: "kodak-legacy",
        in_training: true,
    },
];

#[derive(Clone, Copy)]
enum Mode {
    Ycbcr444,
    Ycbcr420,
}

impl Mode {
    fn dir_name(self) -> &'static str {
        match self {
            Mode::Ycbcr444 => "ycbcr-444",
            Mode::Ycbcr420 => "ycbcr-420",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Mode::Ycbcr444 => "YCbCr 4:4:4",
            Mode::Ycbcr420 => "YCbCr 4:2:0",
        }
    }

    fn base_config(self, quality: u8) -> EncoderConfig {
        match self {
            Mode::Ycbcr444 => {
                EncoderConfig::ycbcr(quality as f32, ChromaSubsampling::None).progressive(false)
            }
            Mode::Ycbcr420 => {
                EncoderConfig::ycbcr(quality as f32, ChromaSubsampling::Quarter).progressive(false)
            }
        }
    }
}

fn main() -> Result<()> {
    let start = Instant::now();

    // Load all image sets
    let cc = corpus_crate().expect("codec-corpus unavailable");
    let mut sets: Vec<(&ImageSet, Vec<PathBuf>)> = Vec::new();
    for set in IMAGE_SETS {
        let dir = match cc.get(set.rel_path) {
            Ok(p) => p,
            Err(_) => {
                eprintln!("SKIP: {} not available", set.name);
                continue;
            }
        };
        let images = load_image_list(&dir)?;
        if !images.is_empty() {
            println!(
                "  {} {} images ({})",
                set.name,
                images.len(),
                if set.in_training {
                    "TRAINING"
                } else {
                    "HOLDOUT"
                }
            );
            sets.push((set, images));
        }
    }

    let holdout_count: usize = sets
        .iter()
        .filter(|(s, _)| !s.in_training)
        .map(|(_, i)| i.len())
        .sum();
    let training_count: usize = sets
        .iter()
        .filter(|(s, _)| s.in_training)
        .map(|(_, i)| i.len())
        .sum();
    println!(
        "\n  Holdout: {} images, Training: {} images",
        holdout_count, training_count
    );
    println!("  Quality tiers: {:?}\n", VALIDATION_QUALITIES);

    for mode in [Mode::Ycbcr444, Mode::Ycbcr420] {
        println!("================================================================");
        println!("  {} — Holdout Validation", mode.label());
        println!("================================================================\n");

        // Load aggregated frequencies for both algorithms
        let agg_dir = freq_dir().join("aggregated").join(mode.dir_name());
        let agg_tables_jpegli = load_tables_for_mode(
            &agg_dir,
            VALIDATION_QUALITIES,
            HuffmanMethod::JpegliCreateTree,
        )?;
        let agg_tables_mozjpeg = load_tables_for_mode(
            &agg_dir,
            VALIDATION_QUALITIES,
            HuffmanMethod::MozjpegClassic,
        )?;

        // Header
        println!(
            "{:>5} | {:25} | {:>4} | {:>5} | {:>10} | {:>10} | {:>10} | {:>10}",
            "Q", "Image Set", "Type", "Imgs", "Jpegli %", "Mozjpeg %", "Annex K %", "Jpegli-Moz"
        );
        println!("{}", "-".repeat(105));

        for &quality in VALIDATION_QUALITIES {
            let jpegli_tbl = &agg_tables_jpegli[&quality];
            let mozjpeg_tbl = &agg_tables_mozjpeg[&quality];

            let mut holdout_opt = 0usize;
            let mut holdout_jpegli = 0usize;
            let mut holdout_mozjpeg = 0usize;
            let mut holdout_annex = 0usize;

            let mut training_opt = 0usize;
            let mut training_jpegli = 0usize;
            let mut training_mozjpeg = 0usize;
            let mut training_annex = 0usize;

            for (set, images) in &sets {
                let stats = validate_images(images, mode, quality, jpegli_tbl, mozjpeg_tbl)?;

                let jpegli_pct = pct_overhead(stats.jpegli_bytes, stats.optimal_bytes);
                let mozjpeg_pct = pct_overhead(stats.mozjpeg_bytes, stats.optimal_bytes);
                let annex_pct = pct_overhead(stats.annex_bytes, stats.optimal_bytes);
                let delta = jpegli_pct - mozjpeg_pct;

                let tag = if set.in_training { "TRAIN" } else { "HOLD" };

                println!(
                    "Q{:<3}  | {:25} | {:>4} | {:>5} | {:>9.3}% | {:>9.3}% | {:>9.3}% | {:>+9.3}%",
                    quality,
                    set.name,
                    tag,
                    images.len(),
                    jpegli_pct,
                    mozjpeg_pct,
                    annex_pct,
                    delta,
                );

                if set.in_training {
                    training_opt += stats.optimal_bytes;
                    training_jpegli += stats.jpegli_bytes;
                    training_mozjpeg += stats.mozjpeg_bytes;
                    training_annex += stats.annex_bytes;
                } else {
                    holdout_opt += stats.optimal_bytes;
                    holdout_jpegli += stats.jpegli_bytes;
                    holdout_mozjpeg += stats.mozjpeg_bytes;
                    holdout_annex += stats.annex_bytes;
                }
            }

            // Summary rows
            if holdout_opt > 0 {
                println!(
                    "Q{:<3}  | {:25} | {:>4} | {:>5} | {:>9.3}% | {:>9.3}% | {:>9.3}% | {:>+9.3}%",
                    quality,
                    "** HOLDOUT TOTAL **",
                    "HOLD",
                    holdout_count,
                    pct_overhead(holdout_jpegli, holdout_opt),
                    pct_overhead(holdout_mozjpeg, holdout_opt),
                    pct_overhead(holdout_annex, holdout_opt),
                    pct_overhead(holdout_jpegli, holdout_opt)
                        - pct_overhead(holdout_mozjpeg, holdout_opt),
                );
            }
            if training_opt > 0 {
                println!(
                    "Q{:<3}  | {:25} | {:>4} | {:>5} | {:>9.3}% | {:>9.3}% | {:>9.3}% | {:>+9.3}%",
                    quality,
                    "** TRAINING TOTAL **",
                    "TRAIN",
                    training_count,
                    pct_overhead(training_jpegli, training_opt),
                    pct_overhead(training_mozjpeg, training_opt),
                    pct_overhead(training_annex, training_opt),
                    pct_overhead(training_jpegli, training_opt)
                        - pct_overhead(training_mozjpeg, training_opt),
                );
            }
            println!("{}", "-".repeat(105));
        }
        println!();
    }

    println!("Done in {:.1}s", start.elapsed().as_secs_f64());
    Ok(())
}

struct ValidationStats {
    optimal_bytes: usize,
    jpegli_bytes: usize,
    mozjpeg_bytes: usize,
    annex_bytes: usize,
}

fn validate_images(
    images: &[PathBuf],
    mode: Mode,
    quality: u8,
    jpegli_tables: &HuffmanTableSet,
    mozjpeg_tables: &HuffmanTableSet,
) -> Result<ValidationStats> {
    let mut optimal_bytes = 0usize;
    let mut jpegli_bytes = 0usize;
    let mut mozjpeg_bytes = 0usize;
    let mut annex_bytes = 0usize;

    for path in images {
        let (w, h, pixels) = load_png(path)?;

        // Optimal (two-pass per-image)
        let opt_len = encode_optimal(w, h, &pixels, mode, quality)?;
        optimal_bytes += opt_len;

        // Corpus tables (JpegliCreateTree algorithm)
        jpegli_bytes += encode_with_tables(w, h, &pixels, mode, quality, jpegli_tables)?;

        // Corpus tables (MozjpegClassic algorithm)
        mozjpeg_bytes += encode_with_tables(w, h, &pixels, mode, quality, mozjpeg_tables)?;

        // Standard Annex K tables
        annex_bytes += encode_annex_k(w, h, &pixels, mode, quality)?;
    }

    Ok(ValidationStats {
        optimal_bytes,
        jpegli_bytes,
        mozjpeg_bytes,
        annex_bytes,
    })
}

fn pct_overhead(actual: usize, baseline: usize) -> f64 {
    100.0 * (actual as f64 - baseline as f64) / baseline as f64
}

fn encode_optimal(w: u32, h: u32, pixels: &[u8], mode: Mode, quality: u8) -> Result<usize> {
    let config = mode.base_config(quality);
    let mut enc = config.encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)?;
    enc.push_packed(pixels, Unstoppable)?;
    Ok(enc.finish()?.len())
}

fn encode_with_tables(
    w: u32,
    h: u32,
    pixels: &[u8],
    mode: Mode,
    quality: u8,
    tables: &HuffmanTableSet,
) -> Result<usize> {
    let config = mode
        .base_config(quality)
        .custom_huffman_tables(tables.clone());
    let mut enc = config.encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)?;
    enc.push_packed(pixels, Unstoppable)?;
    Ok(enc.finish()?.len())
}

fn encode_annex_k(w: u32, h: u32, pixels: &[u8], mode: Mode, quality: u8) -> Result<usize> {
    let tables = HuffmanTableSet::from_standard()?;
    let config = mode.base_config(quality).custom_huffman_tables(tables);
    let mut enc = config.encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)?;
    enc.push_packed(pixels, Unstoppable)?;
    Ok(enc.finish()?.len())
}

// --- Frequency / table loading ---

fn load_tables_for_mode(
    dir: &Path,
    qualities: &[u8],
    method: HuffmanMethod,
) -> Result<BTreeMap<u8, HuffmanTableSet>> {
    let json_path = dir.join("raw_frequencies.json");
    let data: serde_json::Value = serde_json::from_str(&fs::read_to_string(&json_path)?)?;

    let mut tables = BTreeMap::new();
    for &q in qualities {
        let key = format!("q{q}");
        let qdata = data
            .get(&key)
            .ok_or_else(|| format!("missing {key} in {}", json_path.display()))?;

        let freqs = json_to_frequencies(qdata)?;
        tables.insert(q, freqs.generate_tables_with_method(method)?);
    }
    Ok(tables)
}

fn json_to_frequencies(qdata: &serde_json::Value) -> Result<HuffmanSymbolFrequencies> {
    fn load_counter(arr: &serde_json::Value) -> Result<FrequencyCounter> {
        let vals: Vec<i64> = serde_json::from_value(arr.clone())?;
        if vals.len() != 256 {
            return Err(format!("expected 256 values, got {}", vals.len()).into());
        }
        let mut counter = FrequencyCounter::new();
        for (i, &v) in vals.iter().enumerate() {
            counter.set_count(i as u8, v);
        }
        Ok(counter)
    }

    Ok(HuffmanSymbolFrequencies {
        dc_luma: load_counter(&qdata["dc_luma"])?,
        ac_luma: load_counter(&qdata["ac_luma"])?,
        dc_chroma: load_counter(&qdata["dc_chroma"])?,
        ac_chroma: load_counter(&qdata["ac_chroma"])?,
    })
}

fn load_image_list(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut paths: Vec<PathBuf> = fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("png"))
        })
        .collect();
    paths.sort();
    Ok(paths)
}

fn load_png(path: &Path) -> Result<(u32, u32, Vec<u8>)> {
    let img = zenjpeg_bench_utils::load_png(path)
        .map_err(|e| -> Box<dyn std::error::Error> { format!("{e}").into() })?;
    let w = img.width() as u32;
    let h = img.height() as u32;
    let bytes: Vec<u8> = img.buf().iter().flat_map(|p| [p.r, p.g, p.b]).collect();
    Ok((w, h, bytes))
}
