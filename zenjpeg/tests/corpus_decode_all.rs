//! Attempt to decode every file in the scraped corpus directory tree.
//!
//! This is a robustness test: try to feed every file to the decoder.
//! JPEG files should decode successfully. Non-JPEG files should be
//! gracefully rejected with an error (not panic, not crash).
//!
//! Uses rayon for parallel file IO and decoding (~16x faster on many-core).
//!
//! Run: cargo test --release -p zenjpeg --test corpus_decode_all -- --nocapture --ignored

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

use enough::Unstoppable;
use rayon::prelude::*;
use zenjpeg::decoder::Decoder;

const CORPUS_DIR: &str = "/mnt/v/output/corpus-builder";

/// Directories to skip (source code repos, not image files).
const SKIP_DIRS: &[&str] = &["repro-images", "cc-index"];

/// Max file size to attempt decoding (50 MB).
const MAX_FILE_SIZE: u64 = 50 * 1024 * 1024;

/// Collect all files recursively, skipping SKIP_DIRS.
fn collect_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    collect_files_inner(dir, &mut files);
    files.sort();
    files
}

fn collect_files_inner(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let name = path.file_name().unwrap_or_default().to_string_lossy();
            if SKIP_DIRS.iter().any(|&s| name == s) {
                continue;
            }
            collect_files_inner(&path, out);
        } else if path.is_file() {
            out.push(path);
        }
    }
}

/// Check if a file looks like a JPEG by magic bytes.
fn is_jpeg_by_magic(data: &[u8]) -> bool {
    data.len() >= 2 && data[0] == 0xFF && data[1] == 0xD8
}

/// Get a short relative path for display.
fn short_path(path: &Path) -> String {
    path.strip_prefix(CORPUS_DIR)
        .unwrap_or(path)
        .display()
        .to_string()
}

/// Result of attempting to decode one file.
enum DecodeResult {
    JpegOk,
    JpegErr(String, String, usize), // path, error, size
    NonJpegRejected,
    NonJpegDecoded(String),  // path
    Panicked(String, usize), // path, size
    ReadErr(String),         // path
    SkippedLarge,
}

/// Process a single file: read, detect format, attempt decode.
fn process_file(path: &Path) -> DecodeResult {
    // Check file size without reading
    let metadata = match std::fs::metadata(path) {
        Ok(m) => m,
        Err(e) => return DecodeResult::ReadErr(format!("{}: {e}", short_path(path))),
    };
    if metadata.len() > MAX_FILE_SIZE {
        return DecodeResult::SkippedLarge;
    }

    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(e) => return DecodeResult::ReadErr(format!("{}: {e}", short_path(path))),
    };

    let is_jpeg = is_jpeg_by_magic(&data);
    let decoder = Decoder::new();

    // Catch panics — decoder must never panic on any input
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        decoder.decode(&data, Unstoppable)
    }));

    match result {
        Err(_) => DecodeResult::Panicked(short_path(path), data.len()),
        Ok(Ok(decoded)) => {
            if is_jpeg {
                DecodeResult::JpegOk
            } else {
                DecodeResult::NonJpegDecoded(format!(
                    "{} ({}x{}, {} bytes)",
                    short_path(path),
                    decoded.width,
                    decoded.height,
                    data.len()
                ))
            }
        }
        Ok(Err(e)) => {
            if is_jpeg {
                DecodeResult::JpegErr(short_path(path), format!("{e}"), data.len())
            } else {
                DecodeResult::NonJpegRejected
            }
        }
    }
}

#[test]
#[ignore = "requires corpus at /mnt/v/output/corpus-builder"]
fn decode_all_corpus_files() {
    let corpus = Path::new(CORPUS_DIR);
    if !corpus.exists() {
        println!("Corpus not found at {CORPUS_DIR}, skipping");
        return;
    }

    println!(
        "Collecting files from {CORPUS_DIR} (skipping {:?})...",
        SKIP_DIRS
    );
    let files = collect_files(corpus);
    println!(
        "Found {} files, decoding with {} threads...",
        files.len(),
        rayon::current_num_threads()
    );

    let progress = AtomicU32::new(0);
    let total = files.len() as u32;

    // Parallel decode: rayon handles both IO and decode across all cores
    let results: Vec<DecodeResult> = files
        .par_iter()
        .map(|path| {
            let done = progress.fetch_add(1, Ordering::Relaxed);
            if done > 0 && done % 5000 == 0 {
                eprintln!("  ... {done}/{total} files processed");
            }
            process_file(path)
        })
        .collect();

    // Aggregate results
    let mut jpeg_ok = 0u32;
    let mut jpeg_err = 0u32;
    let mut non_jpeg_rejected = 0u32;
    let mut non_jpeg_decoded = 0u32;
    let mut read_err = 0u32;
    let mut panicked = 0u32;
    let mut skipped_large = 0u32;

    let mut jpeg_failures: Vec<(String, String, usize)> = Vec::new();
    let mut panic_files: Vec<(String, usize)> = Vec::new();

    for r in results {
        match r {
            DecodeResult::JpegOk => jpeg_ok += 1,
            DecodeResult::JpegErr(path, err, size) => {
                jpeg_err += 1;
                jpeg_failures.push((path, err, size));
            }
            DecodeResult::NonJpegRejected => non_jpeg_rejected += 1,
            DecodeResult::NonJpegDecoded(info) => {
                non_jpeg_decoded += 1;
                eprintln!("  NON-JPEG DECODED: {info}");
            }
            DecodeResult::Panicked(path, size) => {
                panicked += 1;
                panic_files.push((path, size));
            }
            DecodeResult::ReadErr(info) => {
                read_err += 1;
                eprintln!("  READ ERR: {info}");
            }
            DecodeResult::SkippedLarge => skipped_large += 1,
        }
    }

    // Summary
    println!("\n=== Corpus Decode Results ===");
    println!("Total files scanned:   {}", files.len());
    println!("Skipped (>50MB):       {skipped_large}");
    println!("Read errors:           {read_err}");
    println!();
    println!("JPEG decoded OK:       {jpeg_ok}");
    println!("JPEG decode errors:    {jpeg_err}");
    println!("Non-JPEG rejected:     {non_jpeg_rejected} (expected)");
    println!("Non-JPEG decoded:      {non_jpeg_decoded} (JPEG with wrong ext?)");
    println!("PANICKED:              {panicked}");

    if jpeg_ok + jpeg_err > 0 {
        println!(
            "\nJPEG success rate: {:.1}% ({jpeg_ok}/{})",
            jpeg_ok as f64 / (jpeg_ok + jpeg_err) as f64 * 100.0,
            jpeg_ok + jpeg_err
        );
    }

    if !jpeg_failures.is_empty() {
        println!("\n--- JPEG Decode Failures ({}) ---", jpeg_failures.len());
        let mut by_err: std::collections::BTreeMap<String, Vec<(String, usize)>> =
            std::collections::BTreeMap::new();
        for (path, err, size) in &jpeg_failures {
            by_err
                .entry(err.clone())
                .or_default()
                .push((path.clone(), *size));
        }
        for (err, paths) in &by_err {
            println!("\n  [{} files] {err}", paths.len());
            for (path, size) in paths.iter().take(5) {
                println!("    {path} ({size} bytes)");
            }
            if paths.len() > 5 {
                println!("    ... and {} more", paths.len() - 5);
            }
        }
    }

    if !panic_files.is_empty() {
        println!("\n--- PANICS ({}) ---", panic_files.len());
        for (path, size) in &panic_files {
            println!("  {path} ({size} bytes)");
        }
    }

    // Write full results to file
    let results_path = "/tmp/corpus_decode_all_results.txt";
    let mut report = String::new();
    report.push_str(&format!("Total files scanned: {}\n", files.len()));
    report.push_str(&format!("JPEG decoded OK: {jpeg_ok}\n"));
    report.push_str(&format!("JPEG decode errors: {jpeg_err}\n"));
    report.push_str(&format!("Non-JPEG rejected: {non_jpeg_rejected}\n"));
    report.push_str(&format!("Non-JPEG decoded: {non_jpeg_decoded}\n"));
    report.push_str(&format!("Read errors: {read_err}\n"));
    report.push_str(&format!("Skipped (>50MB): {skipped_large}\n"));
    report.push_str(&format!("PANICKED: {panicked}\n\n"));
    if !jpeg_failures.is_empty() {
        report.push_str(&format!(
            "JPEG Decode Failures ({}):\n",
            jpeg_failures.len()
        ));
        for (path, err, size) in &jpeg_failures {
            report.push_str(&format!("  {path} ({size} bytes): {err}\n"));
        }
    }
    if !panic_files.is_empty() {
        report.push_str(&format!("\nPANICS ({}):\n", panic_files.len()));
        for (path, size) in &panic_files {
            report.push_str(&format!("  {path} ({size} bytes)\n"));
        }
    }
    let _ = std::fs::write(results_path, &report);
    println!("\nFull results saved to {results_path}");

    // Hard failures
    assert_eq!(panicked, 0, "Decoder panicked on {panicked} files!");
}
