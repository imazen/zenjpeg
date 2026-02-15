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

fn decode_zenjpeg_strict(data: &[u8]) -> Result<(), String> {
    use zenjpeg::decoder::{Decoder, Strictness};
    Decoder::new()
        .strictness(Strictness::Strict)
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
        None => {
            eprintln!("Skipping: corpus unavailable");
            return;
        }
    };
    let valid_dir = match corpus.get("jpeg-conformance/valid") {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Skipping: {e}");
            return;
        }
    };
    let invalid_dir = match corpus.get("jpeg-conformance/invalid") {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Skipping: {e}");
            return;
        }
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

/// Show specific error messages for truncated files in each mode.
///
/// Run with: cargo test --release --features decoder -p zenjpeg --test decoder_leniency_comparison -- truncated_error_detail --nocapture --ignored
#[test]
#[ignore]
fn truncated_error_detail() {
    use zenjpeg::decoder::{Decoder, Strictness};

    let corpus = match corpus() {
        Some(c) => c,
        None => {
            eprintln!("Skipping: corpus unavailable");
            return;
        }
    };

    let nonconf = match corpus.get("jpeg-conformance/non-conformant") {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Skipping: {e}");
            return;
        }
    };

    let test_files = [
        "truncated/missing_eoi.jpg",
        "truncated/scan_50pct.jpg",
        "truncated/scan_90pct.jpg",
        "truncated/progressive_50pct.jpg",
        "truncated/progressive_75pct.jpg",
    ];

    for relpath in test_files {
        let path = nonconf.join(relpath);
        let data = match fs::read(&path) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("Skipping {relpath}: file not found");
                continue;
            }
        };
        let name = path.file_name().unwrap().to_string_lossy();
        println!("\n=== {} ({} bytes) ===", name, data.len());

        for (mode_name, mode) in [
            ("Strict", Strictness::Strict),
            ("Balanced", Strictness::Balanced),
            ("Lenient", Strictness::Lenient),
        ] {
            match Decoder::new().strictness(mode).decode(&data, Unstoppable) {
                Ok(result) => {
                    let w = result.width();
                    let h = result.height();
                    let warns = result.warnings().len();
                    println!("  {:10}: OK ({}x{}, {} warnings)", mode_name, w, h, warns);
                    for w in result.warnings() {
                        println!("              warn: {}", w);
                    }
                }
                Err(e) => {
                    println!("  {:10}: ERR: {}", mode_name, e);
                }
            }
        }
    }
}

fn decode_djpeg(data: &[u8]) -> Result<(), String> {
    use std::io::Write;
    use std::process::Command;

    let mut tmp = tempfile::NamedTempFile::new().map_err(|e| e.to_string())?;
    tmp.write_all(data).map_err(|e| e.to_string())?;
    let path = tmp.path().to_owned();

    let output = Command::new("djpeg")
        .arg("-pnm")
        .arg(&path)
        .output()
        .map_err(|e| e.to_string())?;

    if output.status.success() {
        Ok(())
    } else {
        // djpeg exit code 2 = warning (produces output), exit code 1 = fatal
        // Check if output was produced — if so, it recovered
        if output.status.code() == Some(2) && !output.stdout.is_empty() {
            Ok(()) // Produced output despite warning — consider this accepted
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(stderr.trim().to_string())
        }
    }
}

/// Comprehensive strictness comparison across all three zenjpeg modes, libjpeg-turbo,
/// and Rust decoders. Tests valid, invalid, and non-conformant JPEG files.
///
/// Run with: cargo test --release --features decoder -p zenjpeg --test decoder_leniency_comparison -- compare_strictness --nocapture --ignored
#[test]
#[ignore]
fn compare_strictness_vs_libjpeg_turbo() {
    let corpus = match corpus() {
        Some(c) => c,
        None => {
            eprintln!("Skipping: corpus unavailable");
            return;
        }
    };

    let valid_dir = match corpus.get("jpeg-conformance/valid") {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Skipping: {e}");
            return;
        }
    };
    let invalid_dir = match corpus.get("jpeg-conformance/invalid") {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Skipping: {e}");
            return;
        }
    };

    // Load files
    let valid_files: Vec<_> = collect_jpgs(&valid_dir)
        .into_iter()
        .filter_map(|p| fs::read(&p).ok().map(|d| (p, d)))
        .collect();

    let invalid_files: Vec<_> = collect_jpgs(&invalid_dir)
        .into_iter()
        .filter_map(|p| fs::read(&p).ok().map(|d| (p, d)))
        .collect();

    // Load non-conformant files from all subdirectories
    let nonconf_base = match corpus.get("jpeg-conformance/non-conformant") {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Skipping non-conformant: {e}");
            return;
        }
    };
    let nonconf_files: Vec<_> = collect_jpgs(&nonconf_base)
        .into_iter()
        .filter_map(|p| fs::read(&p).ok().map(|d| (p, d)))
        .collect();

    type DecoderFn = dyn Fn(&[u8]) -> Result<(), String>;
    let decoders: Vec<(&str, Box<DecoderFn>)> = vec![
        ("zen-Strict", Box::new(decode_zenjpeg_strict)),
        ("zen-Balanced", Box::new(decode_zenjpeg)),
        ("zen-Lenient", Box::new(decode_zenjpeg_lenient)),
        ("libjpeg-turbo", Box::new(decode_djpeg)),
        ("zune-jpeg", Box::new(decode_zune)),
        ("jpeg-decoder", Box::new(decode_jpeg_decoder)),
    ];

    println!("\n===== COMPREHENSIVE STRICTNESS COMPARISON =====");
    println!(
        "Valid: {}, Invalid: {}, Non-conformant: {}",
        valid_files.len(),
        invalid_files.len(),
        nonconf_files.len()
    );

    // ========== Summary table ==========
    println!("\n## Summary");
    println!(
        "| {:16} | Valid  | Inv Rejected | Non-conf Accept |",
        "Decoder"
    );
    println!("|{:-<18}|{:-<7}|{:-<14}|{:-<17}|", "", "", "", "");

    for (name, decode_fn) in &decoders {
        let valid_ok = valid_files
            .iter()
            .filter(|(_, d)| decode_fn(d).is_ok())
            .count();
        let invalid_rejected = invalid_files
            .iter()
            .filter(|(_, d)| decode_fn(d).is_err())
            .count();
        let nonconf_ok = nonconf_files
            .iter()
            .filter(|(_, d)| decode_fn(d).is_ok())
            .count();

        println!(
            "| {:16} | {}/{:2} | {:3}/{:3}      | {:2}/{:2}            |",
            name,
            valid_ok,
            valid_files.len(),
            invalid_rejected,
            invalid_files.len(),
            nonconf_ok,
            nonconf_files.len()
        );
    }

    // ========== Non-conformant per-file detail ==========
    println!("\n## Non-conformant files (per-file behavior)");
    println!(
        "| {:40} | {:6} | {:6} | {:6} | {:6} | {:6} | {:6} |",
        "File", "Strict", "Balan", "Lennt", "ljt", "zune", "jpgdc"
    );
    println!(
        "|{:-<42}|{:-<8}|{:-<8}|{:-<8}|{:-<8}|{:-<8}|{:-<8}|",
        "", "", "", "", "", "", ""
    );

    for (path, data) in &nonconf_files {
        let fname = path.file_name().unwrap().to_string_lossy();
        let short = if fname.len() > 40 {
            format!("{}...", &fname[..37])
        } else {
            fname.to_string()
        };

        let results: Vec<&str> = decoders
            .iter()
            .map(|(_, decode_fn)| {
                if decode_fn(data).is_ok() {
                    "OK"
                } else {
                    "FAIL"
                }
            })
            .collect();

        println!(
            "| {:40} | {:6} | {:6} | {:6} | {:6} | {:6} | {:6} |",
            short, results[0], results[1], results[2], results[3], results[4], results[5]
        );
    }

    // ========== Valid file failures ==========
    println!("\n## Valid files rejected (should be 0)");
    for (name, decode_fn) in &decoders {
        let failures: Vec<_> = valid_files
            .iter()
            .filter(|(_, d)| decode_fn(d).is_err())
            .map(|(p, _)| p.file_name().unwrap().to_string_lossy().to_string())
            .collect();
        if !failures.is_empty() {
            println!("  {}: {}", name, failures.join(", "));
        }
    }

    // ========== Strict vs Balanced comparison ==========
    println!("\n## Strict rejects but Balanced accepts:");
    for (path, data) in invalid_files.iter().chain(nonconf_files.iter()) {
        let strict_ok = decode_zenjpeg_strict(data).is_ok();
        let balanced_ok = decode_zenjpeg(data).is_ok();
        if !strict_ok && balanced_ok {
            let fname = path.file_name().unwrap().to_string_lossy();
            println!("  {}", fname);
        }
    }

    // ========== Balanced vs Lenient comparison ==========
    println!("\n## Balanced rejects but Lenient accepts:");
    for (path, data) in invalid_files.iter().chain(nonconf_files.iter()) {
        let balanced_ok = decode_zenjpeg(data).is_ok();
        let lenient_ok = decode_zenjpeg_lenient(data).is_ok();
        if !balanced_ok && lenient_ok {
            let fname = path.file_name().unwrap().to_string_lossy();
            println!("  {}", fname);
        }
    }

    // ========== libjpeg-turbo accepts but zenjpeg-Balanced rejects ==========
    println!("\n## libjpeg-turbo accepts but zenjpeg-Balanced rejects:");
    for (path, data) in invalid_files.iter().chain(nonconf_files.iter()) {
        let ljt_ok = decode_djpeg(data).is_ok();
        let balanced_ok = decode_zenjpeg(data).is_ok();
        if ljt_ok && !balanced_ok {
            let fname = path.file_name().unwrap().to_string_lossy();
            println!("  {}", fname);
        }
    }

    // ========== zenjpeg-Balanced accepts but libjpeg-turbo rejects ==========
    println!("\n## zenjpeg-Balanced accepts but libjpeg-turbo rejects:");
    for (path, data) in invalid_files.iter().chain(nonconf_files.iter()) {
        let ljt_ok = decode_djpeg(data).is_ok();
        let balanced_ok = decode_zenjpeg(data).is_ok();
        if !ljt_ok && balanced_ok {
            let fname = path.file_name().unwrap().to_string_lossy();
            println!("  {}", fname);
        }
    }
}
