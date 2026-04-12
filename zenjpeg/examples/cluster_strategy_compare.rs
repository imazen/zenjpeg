//! Compare progressive encoding optimizations across corpus images.
//!
//! Tests:
//! 1. Huffman clustering refinement (1-opt post-greedy pass)
//! 2. Extended scan script search (13 split points vs 5)
//!
//! Usage:
//!   cargo run --release -p zenjpeg --example cluster_strategy_compare
//!
//! Requires codec-corpus crate (downloads images automatically).

use std::path::Path;

use zenjpeg::encoder::{
    ChromaSubsampling, EncoderConfig, PixelLayout, ScanStrategy, SlotReplacement,
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Each encoding configuration to compare.
struct EncodingVariant {
    name: &'static str,
    strategy: SlotReplacement,
    scan: ScanStrategy,
}

const VARIANTS: &[EncodingVariant] = &[
    // Baseline: default progressive (jpegli scan script, RoundRobin clustering)
    EncodingVariant {
        name: "progressive",
        strategy: SlotReplacement::RoundRobin,
        scan: ScanStrategy::Default,
    },
    // 1-opt refined clustering
    EncodingVariant {
        name: "prog+refine",
        strategy: SlotReplacement::Refined,
        scan: ScanStrategy::Default,
    },
    // optimize_scans (5 split points)
    EncodingVariant {
        name: "opt_scans",
        strategy: SlotReplacement::RoundRobin,
        scan: ScanStrategy::Search,
    },
    // optimize_scans + refined clustering
    EncodingVariant {
        name: "opt+refine",
        strategy: SlotReplacement::Refined,
        scan: ScanStrategy::Search,
    },
    // Extended scan search (13 split points)
    EncodingVariant {
        name: "opt_ext",
        strategy: SlotReplacement::RoundRobin,
        scan: ScanStrategy::SearchExtended,
    },
    // Extended scan search + refined clustering
    EncodingVariant {
        name: "ext+refine",
        strategy: SlotReplacement::Refined,
        scan: ScanStrategy::SearchExtended,
    },
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

fn encode_with(
    width: u32,
    height: u32,
    pixels: &[u8],
    quality: f32,
    variant: &EncodingVariant,
) -> std::result::Result<Vec<u8>, String> {
    let config = EncoderConfig::ycbcr(quality, ChromaSubsampling::Quarter)
        .scan_strategy(variant.scan)
        .slot_replacement(variant.strategy);
    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .map_err(|e| format!("setup: {e}"))?;
    enc.push_packed(pixels, enough::Unstoppable)
        .map_err(|e| format!("push: {e}"))?;
    enc.finish().map_err(|e| format!("finish: {e}"))
}

fn run_corpus(corpus_path: &Path, quality: f32) -> Vec<(String, Vec<usize>)> {
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

        let mut sizes = Vec::with_capacity(VARIANTS.len());
        let mut ok = true;

        for variant in VARIANTS {
            match encode_with(width, height, &pixels, quality, variant) {
                Ok(jpeg) => sizes.push(jpeg.len()),
                Err(e) => {
                    eprintln!("  SKIP {name} ({e})");
                    ok = false;
                    break;
                }
            }
        }

        if ok {
            results.push((name, sizes));
        }
    }

    results
}

fn main() -> Result<()> {
    let corpus = codec_corpus::Corpus::new()?;

    // Header
    print!("{:<6} {:>3}", "Corpus", "Q");
    for v in VARIANTS {
        print!(" {:>12}", v.name);
    }
    println!();
    println!("{}", "-".repeat(6 + 4 + VARIANTS.len() * 13));

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

            let mut totals = vec![0usize; VARIANTS.len()];
            for (_, sizes) in &results {
                for (i, &sz) in sizes.iter().enumerate() {
                    totals[i] += sz;
                }
            }

            let baseline = totals[0] as f64;

            print!("{:<6} {:>3}", corpus_set.name, quality as u32);
            for (i, &total) in totals.iter().enumerate() {
                if i == 0 {
                    print!(" {:>12}", total);
                } else {
                    let pct = (total as f64 - baseline) / baseline * 100.0;
                    print!(" {:>+11.3}%", pct);
                }
            }
            println!();
        }
    }

    // Per-image detail for images where opt_ext differs from opt_scans
    println!("\n=== Per-image: opt_ext vs opt_scans (images where they differ) ===\n");
    print!("{:<6} {:>3} {:<22}", "Corpus", "Q", "Image");
    for v in VARIANTS {
        print!(" {:>12}", v.name);
    }
    println!();
    println!("{}", "-".repeat(6 + 4 + 22 + VARIANTS.len() * 13));

    for corpus_set in CORPORA {
        let corpus_path = match corpus.get(corpus_set.rel_path) {
            Ok(p) => p,
            Err(_) => continue,
        };

        for &quality in QUALITIES {
            let results = run_corpus(&corpus_path, quality);
            for (name, sizes) in &results {
                // Show if any variant differs from baseline by more than 10 bytes
                let baseline = sizes[0];
                if sizes.iter().skip(1).all(|&s| (s as i64 - baseline as i64).unsigned_abs() < 10) {
                    continue;
                }

                print!("{:<6} {:>3} {:<22}", corpus_set.name, quality as u32, name);
                for (i, &sz) in sizes.iter().enumerate() {
                    let diff = sz as i64 - baseline as i64;
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
