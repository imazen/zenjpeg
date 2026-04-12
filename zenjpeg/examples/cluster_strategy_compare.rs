//! Compare Huffman clustering slot replacement strategies across corpus images.
//!
//! Encodes each image with progressive mode using each of the 5 slot replacement
//! strategies, then compares file sizes. Only progressive mode triggers the
//! replacement code path (baseline has <=4 contexts, all fit in 4 slots).
//!
//! Usage:
//!   cargo run --release -p zenjpeg --example cluster_strategy_compare
//!
//! Requires codec-corpus crate (downloads images automatically).

use std::path::Path;

use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout, SlotReplacement};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

const STRATEGIES: &[(SlotReplacement, &str)] = &[
    (SlotReplacement::RoundRobin, "RoundRobin"),
    (SlotReplacement::SmallestCount, "SmallestCnt"),
    (SlotReplacement::LowestEvictionCost, "LowestEvict"),
    (SlotReplacement::OldestSlot, "OldestSlot"),
    (SlotReplacement::HighestSlotCost, "HighestCost"),
];

const QUALITIES: &[f32] = &[50.0, 75.0, 85.0, 95.0];

struct CorpusSet {
    name: &'static str,
    rel_path: &'static str,
}

const CORPORA: &[CorpusSet] = &[
    CorpusSet {
        name: "sc",
        rel_path: "gb82-sc",
    },
    CorpusSet {
        name: "cid",
        rel_path: "CID22/CID22-512/training",
    },
    CorpusSet {
        name: "clic",
        rel_path: "clic2025/training",
    },
];

fn load_png(path: &Path) -> Option<(u32, u32, Vec<u8>)> {
    let img = zenjpeg_bench_utils::load_png(path).ok()?;
    let w = img.width() as u32;
    let h = img.height() as u32;
    let bytes: Vec<u8> = img.buf().iter().flat_map(|p| [p.r, p.g, p.b]).collect();
    Some((w, h, bytes))
}

fn encode_with_strategy(
    width: u32,
    height: u32,
    pixels: &[u8],
    quality: f32,
    strategy: SlotReplacement,
) -> std::result::Result<Vec<u8>, String> {
    let config = EncoderConfig::ycbcr(quality, ChromaSubsampling::Quarter)
        .progressive(true)
        .slot_replacement(strategy);
    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .map_err(|e| format!("setup: {e}"))?;
    enc.push_packed(pixels, enough::Unstoppable)
        .map_err(|e| format!("push: {e}"))?;
    enc.finish().map_err(|e| format!("finish: {e}"))
}

/// Per-image result for all strategies at one quality level.
struct ImageResult {
    name: String,
    sizes: Vec<usize>, // one per strategy, indexed same as STRATEGIES
}

fn run_corpus(corpus_path: &Path, quality: f32) -> Vec<ImageResult> {
    let mut entries: Vec<_> = std::fs::read_dir(corpus_path)
        .expect("read dir")
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext == "png")
                .unwrap_or(false)
        })
        .collect();
    entries.sort_by_key(|e| e.file_name());

    let mut results = Vec::new();

    for entry in &entries {
        let path = entry.path();
        let name = path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .chars()
            .take(20)
            .collect::<String>();

        let (width, height, pixels) = match load_png(&path) {
            Some(v) => v,
            None => continue,
        };

        let mut sizes = Vec::with_capacity(STRATEGIES.len());
        let mut ok = true;

        for &(strategy, _) in STRATEGIES {
            match encode_with_strategy(width, height, &pixels, quality, strategy) {
                Ok(jpeg) => sizes.push(jpeg.len()),
                Err(e) => {
                    eprintln!("  SKIP {name}: {e}");
                    ok = false;
                    break;
                }
            }
        }

        if ok {
            results.push(ImageResult { name, sizes });
        }
    }

    results
}

fn main() -> Result<()> {
    let corpus = codec_corpus::Corpus::new()?;

    // Header
    print!("{:<8} {:<6}", "Corpus", "Q");
    for &(_, label) in STRATEGIES {
        print!(" {:>12}", label);
    }
    println!("  {:>8} {:>8}", "best_vs_rr", "best_name");
    println!("{}", "-".repeat(100));

    for corpus_set in CORPORA {
        let corpus_path = match corpus.get(corpus_set.rel_path) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Skipping {}: {}", corpus_set.name, e);
                continue;
            }
        };

        eprintln!(
            "Processing {} from {}...",
            corpus_set.name,
            corpus_path.display()
        );

        for &quality in QUALITIES {
            let results = run_corpus(&corpus_path, quality);
            if results.is_empty() {
                continue;
            }

            // Sum sizes across all images for each strategy
            let mut totals = vec![0usize; STRATEGIES.len()];
            for r in &results {
                for (i, &sz) in r.sizes.iter().enumerate() {
                    totals[i] += sz;
                }
            }

            let rr_total = totals[0] as f64;

            // Find best strategy (smallest total)
            let (best_idx, &best_total) = totals.iter().enumerate().min_by_key(|&(_, &t)| t).unwrap();
            let best_pct = (best_total as f64 - rr_total) / rr_total * 100.0;

            print!("{:<8} {:<6}", corpus_set.name, quality as u32);
            for (i, &total) in totals.iter().enumerate() {
                let pct = (total as f64 - rr_total) / rr_total * 100.0;
                if i == 0 {
                    // RoundRobin is baseline, show absolute
                    print!(" {:>12}", total);
                } else {
                    print!(" {:>+11.4}%", pct);
                }
            }
            println!(
                "  {:>+7.4}% {}",
                best_pct,
                STRATEGIES[best_idx].1,
            );
        }
    }

    // Per-image detail for images where strategies diverge
    println!("\n\n=== Per-image details (images where any strategy differs from RoundRobin) ===\n");
    print!("{:<8} {:<6} {:<22}", "Corpus", "Q", "Image");
    for &(_, label) in STRATEGIES {
        print!(" {:>12}", label);
    }
    println!();
    println!("{}", "-".repeat(110));

    for corpus_set in CORPORA {
        let corpus_path = match corpus.get(corpus_set.rel_path) {
            Ok(p) => p,
            Err(_) => continue,
        };

        for &quality in QUALITIES {
            let results = run_corpus(&corpus_path, quality);

            for r in &results {
                // Only show if any strategy differs from RoundRobin
                let rr = r.sizes[0];
                if r.sizes.iter().all(|&s| s == rr) {
                    continue;
                }

                print!("{:<8} {:<6} {:<22}", corpus_set.name, quality as u32, r.name);
                for &sz in &r.sizes {
                    let diff = sz as i64 - rr as i64;
                    if diff == 0 {
                        print!(" {:>12}", sz);
                    } else {
                        print!(" {:>+12}", diff);
                    }
                }
                println!();
            }
        }
    }

    Ok(())
}
