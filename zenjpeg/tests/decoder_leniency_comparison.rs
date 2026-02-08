//! Compare decoder leniency across Rust JPEG decoders.
//!
//! Tests zenjpeg, zune-jpeg, and jpeg-decoder against the conformance corpus.
//!
//! Run with: cargo test --release --features decoder -p zenjpeg --test decoder_leniency_comparison -- --nocapture --ignored

use enough::Unstoppable;
use std::fs;
use std::path::Path;

fn corpus() -> Option<codec_corpus::Corpus> {
    codec_corpus::Corpus::new().ok()
}

fn collect_jpgs(dir: &Path) -> Vec<std::path::PathBuf> {
    let mut files = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                if let Some(ext) = path.extension() {
                    if ext == "jpg" || ext == "jpeg" {
                        files.push(path);
                    }
                }
            } else if path.is_dir() {
                files.extend(collect_jpgs(&path));
            }
        }
    }
    files.sort();
    files
}

fn decode_zenjpeg(data: &[u8]) -> Result<(), String> {
    use zenjpeg::decoder::Decoder;
    Decoder::new()
        .decode(data, Unstoppable)
        .map(|_| ())
        .map_err(|e| e.to_string())
}

fn decode_zenjpeg_lenient(data: &[u8]) -> Result<(), String> {
    use zenjpeg::decoder::{Decoder, Strictness};
    Decoder::new()
        .strictness(Strictness::Lenient)
        .decode(data, Unstoppable)
        .map(|_| ())
        .map_err(|e| e.to_string())
}

fn decode_zune(data: &[u8]) -> Result<(), String> {
    use zune_core::bytestream::ZCursor;
    use zune_jpeg::JpegDecoder;
    let mut decoder = JpegDecoder::new(ZCursor::new(data));
    decoder.decode().map(|_| ()).map_err(|e| e.to_string())
}

fn decode_jpeg_decoder(data: &[u8]) -> Result<(), String> {
    use jpeg_decoder::Decoder;
    let mut decoder = Decoder::new(data);
    decoder.decode().map(|_| ()).map_err(|e| e.to_string())
}

struct DecoderResult {
    name: &'static str,
    valid_passed: usize,
    valid_total: usize,
    valid_failed: Vec<String>,
    invalid_rejected: usize,
    invalid_total: usize,
    invalid_accepted: Vec<String>,
}

fn test_decoder(
    name: &'static str,
    decode_fn: impl Fn(&[u8]) -> Result<(), String>,
    valid_files: &[(std::path::PathBuf, Vec<u8>)],
    invalid_files: &[(std::path::PathBuf, Vec<u8>)],
) -> DecoderResult {
    let mut valid_passed = 0;
    let mut valid_failed = Vec::new();

    for (path, data) in valid_files {
        let fname = path.file_name().unwrap().to_string_lossy().to_string();
        if decode_fn(data).is_ok() {
            valid_passed += 1;
        } else {
            valid_failed.push(fname);
        }
    }

    let mut invalid_rejected = 0;
    let mut invalid_accepted = Vec::new();

    for (path, data) in invalid_files {
        let fname = path.file_name().unwrap().to_string_lossy().to_string();
        if decode_fn(data).is_err() {
            invalid_rejected += 1;
        } else {
            invalid_accepted.push(fname);
        }
    }

    DecoderResult {
        name,
        valid_passed,
        valid_total: valid_files.len(),
        valid_failed,
        invalid_rejected,
        invalid_total: invalid_files.len(),
        invalid_accepted,
    }
}

#[test]
#[ignore]
fn compare_decoder_leniency() {
    let corpus = match corpus() {
        Some(c) => c,
        None => { eprintln!("Skipping: corpus unavailable"); return; }
    };
    let valid_dir = match corpus.get("jpeg-conformance/valid") {
        Ok(p) => p,
        Err(e) => { eprintln!("Skipping: {e}"); return; }
    };
    let invalid_dir = match corpus.get("jpeg-conformance/invalid") {
        Ok(p) => p,
        Err(e) => { eprintln!("Skipping: {e}"); return; }
    };

    // Load all files into memory
    let valid_files: Vec<_> = collect_jpgs(&valid_dir)
        .into_iter()
        .filter_map(|p| fs::read(&p).ok().map(|d| (p, d)))
        .collect();

    let invalid_files: Vec<_> = collect_jpgs(&invalid_dir)
        .into_iter()
        .filter_map(|p| fs::read(&p).ok().map(|d| (p, d)))
        .collect();

    println!("\n=== DECODER LENIENCY COMPARISON ===");
    println!("Valid files: {}", valid_files.len());
    println!("Invalid files: {}", invalid_files.len());
    println!();

    let results = vec![
        test_decoder(
            "zenjpeg (Balanced)",
            decode_zenjpeg,
            &valid_files,
            &invalid_files,
        ),
        test_decoder(
            "zenjpeg (Lenient)",
            decode_zenjpeg_lenient,
            &valid_files,
            &invalid_files,
        ),
        test_decoder("zune-jpeg", decode_zune, &valid_files, &invalid_files),
        test_decoder(
            "jpeg-decoder",
            decode_jpeg_decoder,
            &valid_files,
            &invalid_files,
        ),
    ];

    // Summary table
    println!("| Decoder | Valid | Invalid Rejected | Leniency |");
    println!("|---------|-------|------------------|----------|");
    for r in &results {
        let leniency = r.invalid_total - r.invalid_rejected;
        println!(
            "| {} | {}/{} | {}/{} | {} accepted |",
            r.name, r.valid_passed, r.valid_total, r.invalid_rejected, r.invalid_total, leniency
        );
    }

    println!("\n=== VALID FILES FAILED ===");
    for r in &results {
        if !r.valid_failed.is_empty() {
            println!("\n{}:", r.name);
            for f in &r.valid_failed {
                println!("  - {}", f);
            }
        }
    }

    println!("\n=== INVALID FILES ACCEPTED (leniency) ===");
    for r in &results {
        if !r.invalid_accepted.is_empty() {
            println!("\n{} ({} accepted):", r.name, r.invalid_accepted.len());
            for f in &r.invalid_accepted {
                println!("  - {}", f);
            }
        }
    }

    // Find files where decoders disagree
    println!("\n=== DISAGREEMENTS ===");

    // Files accepted by some but not all
    let mut all_valid_failed: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut all_invalid_accepted: std::collections::HashSet<String> =
        std::collections::HashSet::new();

    for r in &results {
        for f in &r.valid_failed {
            all_valid_failed.insert(f.clone());
        }
        for f in &r.invalid_accepted {
            all_invalid_accepted.insert(f.clone());
        }
    }

    if !all_valid_failed.is_empty() {
        println!("\nValid files with disagreement:");
        for f in &all_valid_failed {
            let status: Vec<_> = results
                .iter()
                .map(|r| {
                    if r.valid_failed.contains(f) {
                        format!("{}: ✗", r.name)
                    } else {
                        format!("{}: ✓", r.name)
                    }
                })
                .collect();
            println!("  {}: {}", f, status.join(", "));
        }
    }

    if !all_invalid_accepted.is_empty() {
        println!("\nInvalid files with disagreement:");
        let mut disagree_count = 0;
        for f in &all_invalid_accepted {
            let acceptors: Vec<_> = results
                .iter()
                .filter(|r| r.invalid_accepted.contains(f))
                .map(|r| r.name)
                .collect();
            let rejecters: Vec<_> = results
                .iter()
                .filter(|r| !r.invalid_accepted.contains(f))
                .map(|r| r.name)
                .collect();

            if !rejecters.is_empty() {
                println!(
                    "  {}: accepted by [{}], rejected by [{}]",
                    f,
                    acceptors.join(", "),
                    rejecters.join(", ")
                );
                disagree_count += 1;
            }
        }
        if disagree_count == 0 {
            println!("  (all accepted by all lenient decoders)");
        }
    }
}
