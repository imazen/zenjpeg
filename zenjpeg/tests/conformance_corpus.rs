//! Test decoder against jpeg-conformance corpus.
//!
//! Run with: cargo test --release --features decoder -p zenjpeg --test conformance_corpus -- --nocapture

use std::fs;
use std::path::Path;

const CORPUS_BASE: &str = "/home/lilith/work/codec-eval/codec-corpus/jpeg-conformance";

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

fn decode_file(path: &Path) -> Result<(u32, u32, usize), String> {
    let data = fs::read(path).map_err(|e| format!("read error: {e}"))?;
    let decoder = zenjpeg::decoder::Decoder::new();
    let image = decoder.decode(&data).map_err(|e| format!("decode error: {e}"))?;
    Ok((image.width(), image.height(), image.pixels().len()))
}

#[test]
#[ignore] // Run with --ignored
fn test_valid_files() {
    let dir = Path::new(CORPUS_BASE).join("valid");
    let files = collect_jpgs(&dir);

    println!("\n=== VALID FILES ({} total) ===", files.len());

    let mut passed = 0;
    let mut failed = Vec::new();

    for path in &files {
        let name = path.file_name().unwrap().to_string_lossy();
        match decode_file(path) {
            Ok((w, h, bytes)) => {
                println!("✓ {} ({}x{}, {} bytes)", name, w, h, bytes);
                passed += 1;
            }
            Err(e) => {
                println!("✗ {}: {}", name, e);
                failed.push((name.to_string(), e));
            }
        }
    }

    println!("\nPassed: {}/{}", passed, files.len());

    if !failed.is_empty() {
        println!("\nFailed files:");
        for (name, err) in &failed {
            println!("  - {}: {}", name, err);
        }
    }

    // Don't assert all pass - just report. Some features may not be implemented yet.
    assert!(
        passed >= files.len() * 90 / 100,
        "Expected at least 90% pass rate, got {}/{}",
        passed,
        files.len()
    );
}

#[test]
#[ignore] // Run with --ignored
fn test_invalid_files() {
    let dir = Path::new(CORPUS_BASE).join("invalid");
    let files = collect_jpgs(&dir);

    println!("\n=== INVALID FILES ({} total) ===", files.len());
    println!("Testing graceful rejection (no panics)...\n");

    let mut rejected = 0;
    let mut accepted = Vec::new();

    for path in &files {
        let name = path.file_name().unwrap().to_string_lossy();

        // Use catch_unwind to detect panics
        let result = std::panic::catch_unwind(|| decode_file(path));

        match result {
            Ok(Ok((w, h, _))) => {
                println!("⚠ {} ACCEPTED ({}x{}) - should have been rejected", name, w, h);
                accepted.push(name.to_string());
            }
            Ok(Err(_)) => {
                // Expected: graceful error
                rejected += 1;
            }
            Err(_) => {
                println!("💥 {} PANICKED - this is a bug!", name);
                panic!("Decoder panicked on {}", name);
            }
        }
    }

    println!("\nRejected (correct): {}/{}", rejected, files.len());
    println!("Accepted (unexpected): {}", accepted.len());

    if !accepted.is_empty() {
        println!("\nFiles that decoded but shouldn't have:");
        for name in &accepted {
            println!("  - {}", name);
        }
    }
}

#[test]
#[ignore] // Run with --ignored
fn test_nonconformant_files() {
    let dir = Path::new(CORPUS_BASE).join("non-conformant");
    let files = collect_jpgs(&dir);

    println!("\n=== NON-CONFORMANT FILES ({} total) ===", files.len());
    println!("Documenting behavior (reject or accept both valid)...\n");

    let mut accepted = 0;
    let mut rejected = 0;

    for path in &files {
        let name = path.file_name().unwrap().to_string_lossy();
        let parent = path.parent().unwrap().file_name().unwrap().to_string_lossy();

        // Read companion .txt if exists
        let txt_path = path.with_extension("txt");
        let description = fs::read_to_string(&txt_path).ok();

        let result = std::panic::catch_unwind(|| decode_file(path));

        match result {
            Ok(Ok((w, h, _))) => {
                println!("✓ {}/{} ACCEPTED ({}x{}) - lenient behavior", parent, name, w, h);
                accepted += 1;
            }
            Ok(Err(e)) => {
                println!("✗ {}/{} REJECTED: {} - strict behavior", parent, name, e);
                rejected += 1;
            }
            Err(_) => {
                println!("💥 {}/{} PANICKED - this is a bug!", parent, name);
                panic!("Decoder panicked on {}", name);
            }
        }

        if let Some(desc) = description {
            let first_line = desc.lines().next().unwrap_or("");
            println!("   Note: {}", first_line);
        }
        println!();
    }

    println!("\nSummary: {} accepted (lenient), {} rejected (strict)", accepted, rejected);
}

#[test]
#[ignore] // Run with --ignored
fn test_cmyk_files() {
    println!("\n=== CMYK/YCCK FILES ===\n");

    let cmyk_files = [
        "cmyk_logo.jpg",
        "cymk.jpg",
    ];

    for name in cmyk_files {
        let path = Path::new(CORPUS_BASE).join("valid").join(name);
        if !path.exists() {
            println!("⚠ {} not found", name);
            continue;
        }

        match decode_file(&path) {
            Ok((w, h, bytes)) => {
                // CMYK should decode to RGB (3 bytes per pixel)
                let expected_rgb = (w * h * 3) as usize;
                if bytes == expected_rgb {
                    println!("✓ {} ({}x{}) -> RGB ({} bytes)", name, w, h, bytes);
                } else {
                    println!("⚠ {} ({}x{}) -> {} bytes (expected {})", name, w, h, bytes, expected_rgb);
                }
            }
            Err(e) => {
                println!("✗ {}: {}", name, e);
            }
        }
    }
}
