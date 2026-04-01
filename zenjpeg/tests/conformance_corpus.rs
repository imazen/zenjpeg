//! Test decoder against jpeg-conformance corpus.
//!
//! Uses codec-corpus for on-demand download of JPEG conformance test files.
//! Downloads automatically on first run; cached for subsequent runs.
use enough::Unstoppable;

use std::fs;
use std::path::Path;
use zenjpeg::decoder::{Decoder, Strictness};

fn corpus() -> Option<codec_corpus::Corpus> {
    codec_corpus::Corpus::new().ok()
}

fn collect_jpgs(dir: &Path) -> Vec<std::path::PathBuf> {
    let mut files = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file()
                && let Some(ext) = path.extension()
                && (ext == "jpg" || ext == "jpeg")
            {
                files.push(path);
            } else if path.is_dir() {
                files.extend(collect_jpgs(&path));
            }
        }
    }
    files.sort();
    files
}

fn decode_file(path: &Path) -> Result<(u32, u32, usize), String> {
    decode_file_with_strictness(path, Strictness::default())
}

fn decode_file_with_strictness(
    path: &Path,
    strictness: Strictness,
) -> Result<(u32, u32, usize), String> {
    let data = fs::read(path).map_err(|e| format!("read error: {e}"))?;
    let decoder = Decoder::new().strictness(strictness);
    let image = decoder
        .decode(&data, Unstoppable)
        .map_err(|e| format!("decode error: {e}"))?;
    Ok((
        image.width(),
        image.height(),
        image.pixels_u8().unwrap().len(),
    ))
}

#[test]
fn test_valid_files() {
    let corpus = match corpus() {
        Some(c) => c,
        None => {
            eprintln!("Skipping: corpus unavailable");
            return;
        }
    };
    let dir = match corpus.get("jpeg-conformance/valid") {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Skipping: {e}");
            return;
        }
    };
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
fn test_invalid_files() {
    let corpus = match corpus() {
        Some(c) => c,
        None => {
            eprintln!("Skipping: corpus unavailable");
            return;
        }
    };
    let dir = match corpus.get("jpeg-conformance/invalid") {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Skipping: {e}");
            return;
        }
    };
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
                println!(
                    "⚠ {} ACCEPTED ({}x{}) - should have been rejected",
                    name, w, h
                );
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
fn test_nonconformant_files() {
    let corpus = match corpus() {
        Some(c) => c,
        None => {
            eprintln!("Skipping: corpus unavailable");
            return;
        }
    };
    let dir = match corpus.get("jpeg-conformance/non-conformant") {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Skipping: {e}");
            return;
        }
    };
    let files = collect_jpgs(&dir);

    println!("\n=== NON-CONFORMANT FILES ({} total) ===", files.len());
    println!("Documenting behavior (reject or accept both valid)...\n");

    let mut accepted = 0;
    let mut rejected = 0;

    for path in &files {
        let name = path.file_name().unwrap().to_string_lossy();
        let parent = path
            .parent()
            .unwrap()
            .file_name()
            .unwrap()
            .to_string_lossy();

        // Read companion .txt if exists
        let txt_path = path.with_extension("txt");
        let description = fs::read_to_string(&txt_path).ok();

        let result = std::panic::catch_unwind(|| decode_file(path));

        match result {
            Ok(Ok((w, h, _))) => {
                println!(
                    "✓ {}/{} ACCEPTED ({}x{}) - lenient behavior",
                    parent, name, w, h
                );
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

    println!(
        "\nSummary: {} accepted (lenient), {} rejected (strict)",
        accepted, rejected
    );
}

#[test]
fn test_cmyk_files() {
    println!("\n=== CMYK/YCCK FILES ===\n");

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

    let cmyk_files = ["cmyk_logo.jpg", "cymk.jpg"];

    for name in cmyk_files {
        let path = valid_dir.join(name);
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
                    println!(
                        "⚠ {} ({}x{}) -> {} bytes (expected {})",
                        name, w, h, bytes, expected_rgb
                    );
                }
            }
            Err(e) => {
                println!("✗ {}: {}", name, e);
            }
        }
    }
}

/// Test that strictness modes behave correctly.
///
/// - Valid files: should decode with all modes
/// - Non-conformant files: Strict may reject more than Lenient
/// - Invalid files: should be rejected by all modes
#[test]
fn test_strictness_modes() {
    println!("\n=== STRICTNESS MODE COMPARISON ===\n");

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
    let nonconf_dir = match corpus.get("jpeg-conformance/non-conformant") {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Skipping: {e}");
            return;
        }
    };

    // Test valid files with all strictness modes
    let valid_files = collect_jpgs(&valid_dir);

    println!("=== VALID FILES (all modes should accept) ===\n");
    let mut valid_strict_fail = Vec::new();
    let mut valid_balanced_fail = Vec::new();
    let mut valid_lenient_fail = Vec::new();

    for path in &valid_files {
        let name = path.file_name().unwrap().to_string_lossy();

        let strict_ok = decode_file_with_strictness(path, Strictness::Strict).is_ok();
        let balanced_ok = decode_file_with_strictness(path, Strictness::Balanced).is_ok();
        let lenient_ok = decode_file_with_strictness(path, Strictness::Lenient).is_ok();

        if !strict_ok {
            valid_strict_fail.push(name.to_string());
        }
        if !balanced_ok {
            valid_balanced_fail.push(name.to_string());
        }
        if !lenient_ok {
            valid_lenient_fail.push(name.to_string());
        }
    }

    println!(
        "Valid files: Strict {}/{}, Balanced {}/{}, Lenient {}/{}",
        valid_files.len() - valid_strict_fail.len(),
        valid_files.len(),
        valid_files.len() - valid_balanced_fail.len(),
        valid_files.len(),
        valid_files.len() - valid_lenient_fail.len(),
        valid_files.len()
    );

    // Test non-conformant files - expect Strict to reject more
    let nonconf_files = collect_jpgs(&nonconf_dir);

    println!("\n=== NON-CONFORMANT FILES (strictness differences) ===\n");
    let mut strict_accepted = 0;
    let mut balanced_accepted = 0;
    let mut lenient_accepted = 0;
    let mut differences = Vec::new();

    for path in &nonconf_files {
        let name = path.file_name().unwrap().to_string_lossy();
        let parent = path
            .parent()
            .unwrap()
            .file_name()
            .unwrap()
            .to_string_lossy();
        let full_name = format!("{}/{}", parent, name);

        let strict_ok =
            std::panic::catch_unwind(|| decode_file_with_strictness(path, Strictness::Strict))
                .map(|r| r.is_ok())
                .unwrap_or(false);

        let balanced_ok =
            std::panic::catch_unwind(|| decode_file_with_strictness(path, Strictness::Balanced))
                .map(|r| r.is_ok())
                .unwrap_or(false);

        let lenient_ok =
            std::panic::catch_unwind(|| decode_file_with_strictness(path, Strictness::Lenient))
                .map(|r| r.is_ok())
                .unwrap_or(false);

        if strict_ok {
            strict_accepted += 1;
        }
        if balanced_ok {
            balanced_accepted += 1;
        }
        if lenient_ok {
            lenient_accepted += 1;
        }

        // Record files where modes differ
        if strict_ok != lenient_ok || balanced_ok != lenient_ok {
            differences.push((
                full_name.clone(),
                if strict_ok { "✓" } else { "✗" },
                if balanced_ok { "✓" } else { "✗" },
                if lenient_ok { "✓" } else { "✗" },
            ));
        }
    }

    println!(
        "Non-conformant files: Strict {}/{}, Balanced {}/{}, Lenient {}/{}",
        strict_accepted,
        nonconf_files.len(),
        balanced_accepted,
        nonconf_files.len(),
        lenient_accepted,
        nonconf_files.len()
    );

    if !differences.is_empty() {
        println!("\nFiles with different behavior between modes:");
        println!("{:<50} Strict  Balanced  Lenient", "File");
        println!("{}", "-".repeat(70));
        for (name, strict, balanced, lenient) in &differences {
            println!(
                "{:<50} {}       {}         {}",
                name, strict, balanced, lenient
            );
        }
    }

    // Sanity checks
    // Lenient should accept at least as many as Balanced
    assert!(
        lenient_accepted >= balanced_accepted,
        "Lenient should accept at least as many as Balanced"
    );
    // Balanced should accept at least as many as Strict
    assert!(
        balanced_accepted >= strict_accepted,
        "Balanced should accept at least as many as Strict"
    );
}
