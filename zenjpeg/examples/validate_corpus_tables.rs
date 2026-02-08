//! Validate corpus-trained Huffman tables against per-image optimal.
//!
//! Reads frequency JSON from a previous `gather_corpus_frequencies` run,
//! builds tables, and measures overhead vs per-image optimal encoding.
//!
//! For each YCbCr mode, compares:
//! - **Optimal**: Per-image two-pass (baseline, best possible)
//! - **Aggregated**: Tables from all corpora combined
//! - **Per-corpus**: Tables from this corpus only
//! - **Standard**: JPEG Annex K fixed tables
//!
//! Run with:
//!   cargo run --release -p zenjpeg --example validate_corpus_tables

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use enough::Unstoppable;
use zenjpeg::encode::{
    ChromaSubsampling, EncoderConfig, HuffmanSymbolFrequencies, PixelLayout, XybSubsampling,
};
use zenjpeg::huffman::optimize::{FrequencyCounter, HuffmanTableSet};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn corpus_crate() -> std::result::Result<codec_corpus::Corpus, Box<dyn std::error::Error>> {
    Ok(codec_corpus::Corpus::new()?)
}
const FREQ_DIR: &str = "/mnt/v/output/zenjpeg/huffman-freq";

/// Validation quality tiers (subset for speed).
const VALIDATION_QUALITIES: &[u8] = &[50, 75, 85, 90, 95];

struct Corpus {
    name: &'static str,
    rel_path: &'static str,
}

const CORPORA: &[Corpus] = &[
    Corpus {
        name: "clic2025-training",
        rel_path: "clic2025/training",
    },
    Corpus {
        name: "CID22-training",
        rel_path: "CID22/CID22-512/training",
    },
    Corpus {
        name: "gb82",
        rel_path: "gb82",
    },
    Corpus {
        name: "gb82-sc",
        rel_path: "gb82-sc",
    },
    Corpus {
        name: "kadid10k",
        rel_path: "kadid10k",
    },
    Corpus {
        name: "kodak-legacy",
        rel_path: "kodak-legacy",
    },
    Corpus {
        name: "qoi-screenshot-web",
        rel_path: "qoi-benchmark/screenshot_web",
    },
];

#[derive(Clone, Copy)]
enum Mode {
    Ycbcr444,
    Ycbcr422,
    Ycbcr420,
    XybFull,
    XybBquarter,
}

impl Mode {
    const ALL: &[Mode] = &[
        Mode::Ycbcr444,
        Mode::Ycbcr422,
        Mode::Ycbcr420,
        Mode::XybFull,
        Mode::XybBquarter,
    ];

    fn dir_name(self) -> &'static str {
        match self {
            Mode::Ycbcr444 => "ycbcr-444",
            Mode::Ycbcr422 => "ycbcr-422",
            Mode::Ycbcr420 => "ycbcr-420",
            Mode::XybFull => "xyb-full",
            Mode::XybBquarter => "xyb-bquarter",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Mode::Ycbcr444 => "YCbCr 4:4:4",
            Mode::Ycbcr422 => "YCbCr 4:2:2",
            Mode::Ycbcr420 => "YCbCr 4:2:0",
            Mode::XybFull => "XYB Full",
            Mode::XybBquarter => "XYB BQuarter",
        }
    }

    fn base_config(self, quality: u8) -> EncoderConfig {
        match self {
            Mode::Ycbcr444 => {
                EncoderConfig::ycbcr(quality as f32, ChromaSubsampling::None).progressive(false)
            }
            Mode::Ycbcr422 => {
                EncoderConfig::ycbcr(quality as f32, ChromaSubsampling::HalfHorizontal)
                    .progressive(false)
            }
            Mode::Ycbcr420 => {
                EncoderConfig::ycbcr(quality as f32, ChromaSubsampling::Quarter).progressive(false)
            }
            Mode::XybFull => {
                EncoderConfig::xyb(quality as f32, XybSubsampling::Full).progressive(false)
            }
            Mode::XybBquarter => {
                EncoderConfig::xyb(quality as f32, XybSubsampling::BQuarter).progressive(false)
            }
        }
    }
}

fn main() -> Result<()> {
    let start = Instant::now();

    // Load all corpus image lists
    let cc = corpus_crate().expect("codec-corpus unavailable");
    let mut corpus_images: Vec<(&Corpus, Vec<PathBuf>)> = Vec::new();
    for corpus in CORPORA {
        let dir = match cc.get(corpus.rel_path) {
            Ok(p) => p,
            Err(_) => {
                eprintln!("SKIP: {} not available", corpus.name);
                continue;
            }
        };
        let images = load_image_list(&dir)?;
        if !images.is_empty() {
            corpus_images.push((corpus, images));
        }
    }

    let total_images: usize = corpus_images.iter().map(|(_, i)| i.len()).sum();
    println!("=== Corpus Huffman Table Validation ===\n");
    println!("Corpora: {} ({} images)", corpus_images.len(), total_images);
    println!("Quality tiers: {:?}", VALIDATION_QUALITIES);
    println!(
        "Modes: {:?}\n",
        Mode::ALL.iter().map(|m| m.label()).collect::<Vec<_>>()
    );

    for mode in Mode::ALL {
        println!("==============================");
        println!("  YCbCr {}", mode.label());
        println!("==============================\n");

        // Load aggregated frequency tables for this mode
        let agg_tables = load_tables_for_mode(
            &PathBuf::from(FREQ_DIR)
                .join("aggregated")
                .join(mode.dir_name()),
            VALIDATION_QUALITIES,
        )?;

        // Load per-corpus frequency tables
        let mut corpus_tables: BTreeMap<String, BTreeMap<u8, HuffmanTableSet>> = BTreeMap::new();
        for (corpus, _) in &corpus_images {
            let dir = PathBuf::from(FREQ_DIR)
                .join(corpus.name)
                .join(mode.dir_name());
            if dir.exists() {
                corpus_tables.insert(
                    corpus.name.to_string(),
                    load_tables_for_mode(&dir, VALIDATION_QUALITIES)?,
                );
            }
        }

        // Print header
        println!(
            "{:>5} | {:25} | {:>5} | {:>10} | {:>10} | {:>10} | {:>10}",
            "Q", "Corpus", "Imgs", "Agg %", "Own %", "Std %", "Agg-Own"
        );
        println!("{}", "-".repeat(95));

        for &quality in VALIDATION_QUALITIES {
            let agg_tbl = &agg_tables[&quality];

            let mut total_opt = 0usize;
            let mut total_agg = 0usize;
            let mut total_own = 0usize;
            let mut total_std = 0usize;

            for (corpus, images) in &corpus_images {
                let own_tbl = corpus_tables.get(corpus.name).and_then(|m| m.get(&quality));

                let stats = validate_images(images, *mode, quality, agg_tbl, own_tbl)?;

                total_opt += stats.optimal_bytes;
                total_agg += stats.agg_bytes;
                total_own += stats.own_bytes;
                total_std += stats.std_bytes;

                let agg_pct = pct_overhead(stats.agg_bytes, stats.optimal_bytes);
                let own_pct = own_tbl
                    .map(|_| pct_overhead(stats.own_bytes, stats.optimal_bytes))
                    .unwrap_or(f64::NAN);
                let std_pct = pct_overhead(stats.std_bytes, stats.optimal_bytes);
                let delta = if own_pct.is_nan() {
                    f64::NAN
                } else {
                    agg_pct - own_pct
                };

                println!(
                    "Q{:<3}  | {:25} | {:>5} | {:>9.3}% | {:>9.3}% | {:>9.3}% | {:>+9.3}%",
                    quality,
                    corpus.name,
                    images.len(),
                    agg_pct,
                    own_pct,
                    std_pct,
                    delta,
                );
            }

            println!(
                "Q{:<3}  | {:25} | {:>5} | {:>9.3}% | {:>9.3}% | {:>9.3}% | {:>+9.3}%",
                quality,
                "** ALL **",
                total_images,
                pct_overhead(total_agg, total_opt),
                pct_overhead(total_own, total_opt),
                pct_overhead(total_std, total_opt),
                pct_overhead(total_agg, total_opt) - pct_overhead(total_own, total_opt),
            );
            println!("{}", "-".repeat(95));
        }
        println!();
    }

    println!("Done in {:.1}s", start.elapsed().as_secs_f64());
    Ok(())
}

struct ValidationStats {
    optimal_bytes: usize,
    agg_bytes: usize,
    own_bytes: usize,
    std_bytes: usize,
}

fn validate_images(
    images: &[PathBuf],
    mode: Mode,
    quality: u8,
    agg_tables: &HuffmanTableSet,
    own_tables: Option<&HuffmanTableSet>,
) -> Result<ValidationStats> {
    let mut optimal_bytes = 0usize;
    let mut agg_bytes = 0usize;
    let mut own_bytes = 0usize;
    let mut std_bytes = 0usize;

    for path in images {
        let (w, h, pixels) = load_png(path)?;

        // Optimal (two-pass)
        let (opt_len, _) = encode_optimal(w, h, &pixels, mode, quality)?;
        optimal_bytes += opt_len;

        // Aggregated corpus tables
        agg_bytes += encode_with_tables(w, h, &pixels, mode, quality, agg_tables)?;

        // Per-corpus tables (or aggregated as fallback)
        own_bytes += encode_with_tables(
            w,
            h,
            &pixels,
            mode,
            quality,
            own_tables.unwrap_or(agg_tables),
        )?;

        // Standard JPEG tables
        std_bytes += encode_standard(w, h, &pixels, mode, quality)?;
    }

    Ok(ValidationStats {
        optimal_bytes,
        agg_bytes,
        own_bytes,
        std_bytes,
    })
}

fn pct_overhead(actual: usize, baseline: usize) -> f64 {
    100.0 * (actual as f64 - baseline as f64) / baseline as f64
}

// --- Encoding helpers ---

fn encode_optimal(
    w: u32,
    h: u32,
    pixels: &[u8],
    mode: Mode,
    quality: u8,
) -> Result<(usize, Box<HuffmanSymbolFrequencies>)> {
    let config = mode.base_config(quality);
    let mut enc = config.encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)?;
    enc.push_packed(pixels, Unstoppable)?;
    let (jpeg, counts) = enc.finish_with_huffman_frequencies()?;
    let counts = counts.expect("optimize_huffman on by default");
    Ok((jpeg.len(), counts))
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

fn encode_standard(w: u32, h: u32, pixels: &[u8], mode: Mode, quality: u8) -> Result<usize> {
    let config = mode.base_config(quality).optimize_huffman(false);
    let mut enc = config.encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)?;
    enc.push_packed(pixels, Unstoppable)?;
    Ok(enc.finish()?.len())
}

// --- Frequency / table loading ---

fn load_tables_for_mode(dir: &Path, qualities: &[u8]) -> Result<BTreeMap<u8, HuffmanTableSet>> {
    let json_path = dir.join("raw_frequencies.json");
    let data: serde_json::Value = serde_json::from_str(&fs::read_to_string(&json_path)?)?;

    let mut tables = BTreeMap::new();
    for &q in qualities {
        let key = format!("q{q}");
        let qdata = data
            .get(&key)
            .ok_or_else(|| format!("missing {key} in {}", json_path.display()))?;

        let freqs = json_to_frequencies(qdata)?;
        tables.insert(q, freqs.generate_tables()?);
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

// --- I/O ---

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
    let mut decoder = png::Decoder::new(fs::File::open(path)?);
    decoder.set_transformations(png::Transformations::EXPAND);
    let mut reader = decoder.read_info()?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf)?;

    let (w, h) = (info.width, info.height);
    let pixels = match info.color_type {
        png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgba => buf[..info.buffer_size()]
            .chunks(4)
            .flat_map(|c| [c[0], c[1], c[2]])
            .collect(),
        png::ColorType::Grayscale => buf[..info.buffer_size()]
            .iter()
            .flat_map(|&g| [g, g, g])
            .collect(),
        png::ColorType::GrayscaleAlpha => buf[..info.buffer_size()]
            .chunks(2)
            .flat_map(|c| [c[0], c[0], c[0]])
            .collect(),
        other => return Err(format!("Unsupported color type: {other:?}").into()),
    };

    Ok((w, h, pixels))
}
