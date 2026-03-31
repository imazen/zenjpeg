//! Locked encoder output verification using CSV-based reference values.
//!
//! This test uses CSV files (`locked_values/values_{variant}.csv`) to store expected
//! encoder outputs. Each SIMD variant has its own file, protected by a SHA-256 hash
//! constant that must be updated when values change intentionally.
//!
//! ## Updating Values
//!
//! ```bash
//! # 1. Regenerate archmage values (fails, but writes new CSV)
//! REGENERATE_LOCKED_VALUES=1 cargo test --release -p zenjpeg --test locked_values -- regenerate --ignored --nocapture
//!
//! # 2. Regenerate wide values (fails, but writes new CSV)
//! REGENERATE_LOCKED_VALUES=1 cargo test --release -p zenjpeg --test locked_values --no-default-features --features "std,yuv" -- regenerate --ignored --nocapture
//!
//! # 3. Archive old values with justification
//! cp tests/locked_values/values_archmage.csv "tests/locked_values/history/$(date +%Y-%m-%d)_archmage_description.csv"
//!
//! # 4. Update the appropriate HASH constant in this file
//!
//! # 5. Run tests
//! cargo test --release -p zenjpeg --test locked_values
//! ```

use sha2::{Digest, Sha256};
use std::collections::HashMap;
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout, XybSubsampling};

// =============================================================================
// LOCKED FILE HASHES - Update when CSV files change intentionally
// =============================================================================

/// SHA-256 hash of normalized values_archmage.csv (LF line endings, no trailing whitespace).
#[cfg(target_arch = "x86_64")]
const VALUES_FILE_HASH: &str = "f2aa555e7fe8f4329b9dd195c98c8bca75818f8c247da5e39f45c70b32440e60";

/// SHA-256 hash of normalized values_wide.csv (LF line endings, no trailing whitespace).
#[cfg(not(target_arch = "x86_64"))]
const VALUES_FILE_HASH: &str = "1eaf3785274279445c1b2a5f21127d0dfcfdbdfa8138c2423ab798a37c872e78";

// =============================================================================
// CSV FILE (compile-time inclusion based on SIMD variant)
// =============================================================================

#[cfg(target_arch = "x86_64")]
const VALUES_CSV: &str = include_str!("locked_values/values_archmage.csv");

#[cfg(not(target_arch = "x86_64"))]
const VALUES_CSV: &str = include_str!("locked_values/values_wide.csv");

#[cfg(target_arch = "x86_64")]
const SIMD_VARIANT: &str = "archmage";

#[cfg(not(target_arch = "x86_64"))]
const SIMD_VARIANT: &str = "wide";

// =============================================================================
// TYPES
// =============================================================================

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ValueKey {
    mode: String,        // baseline, progressive
    subsampling: String, // 444, 422, 420, 440, xyb
    huffman: String,     // opt, fixed
    quality: u8,
}

#[derive(Debug, Clone)]
struct ExpectedValue {
    hash: String,
    size: usize,
}

// =============================================================================
// PARSING
// =============================================================================

fn normalize_csv(content: &str) -> String {
    content
        .lines()
        .map(|line| line.trim_end())
        .collect::<Vec<_>>()
        .join("\n")
}

fn hash_content(content: &str) -> String {
    let normalized = normalize_csv(content);
    let mut hasher = Sha256::new();
    hasher.update(normalized.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn parse_csv(content: &str) -> HashMap<ValueKey, ExpectedValue> {
    let mut map = HashMap::new();

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() != 7 {
            eprintln!("Warning: skipping malformed line: {}", line);
            continue;
        }

        // Skip entries for other SIMD variants (shouldn't happen with split files,
        // but keep for safety)
        if parts[4] != SIMD_VARIANT {
            continue;
        }

        let key = ValueKey {
            mode: parts[0].to_string(),
            subsampling: parts[1].to_string(),
            huffman: parts[2].to_string(),
            quality: parts[3].parse().expect("invalid quality"),
        };

        let value = ExpectedValue {
            hash: parts[5].to_string(),
            size: parts[6].parse().expect("invalid size"),
        };

        map.insert(key, value);
    }

    map
}

// =============================================================================
// TEST IMAGE
// =============================================================================

fn load_frymire() -> (Vec<u8>, u32, u32) {
    let png_path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/images/frymire.png");
    let img = zenjpeg_bench_utils::load_png(std::path::Path::new(png_path))
        .expect("Failed to load frymire.png");
    let width = img.width() as u32;
    let height = img.height() as u32;
    let rgb: Vec<u8> = img.buf().iter().flat_map(|p| [p.r, p.g, p.b]).collect();
    (rgb, width, height)
}

// =============================================================================
// ENCODER HELPERS
// =============================================================================

fn encode_config(
    pixels: &[u8],
    width: u32,
    height: u32,
    mode: &str,
    subsampling: &str,
    huffman: &str,
    quality: u8,
) -> Vec<u8> {
    let subsamp = match subsampling {
        "444" => ChromaSubsampling::None,
        "422" => ChromaSubsampling::HalfHorizontal,
        "420" => ChromaSubsampling::Quarter,
        "440" => ChromaSubsampling::HalfVertical,
        "xyb" => ChromaSubsampling::None, // XYB uses 444 internally
        _ => panic!("Unknown subsampling: {}", subsampling),
    };

    let progressive = mode == "progressive";
    let optimize = huffman == "opt";

    let config = if subsampling == "xyb" {
        EncoderConfig::xyb(quality as f32, XybSubsampling::BQuarter)
            .progressive(progressive)
            .optimize_huffman(optimize)
            .restart_mcu_rows(0) // Disable restart markers to match locked hashes
    } else {
        EncoderConfig::ycbcr(quality as f32, subsamp)
            .progressive(progressive)
            .optimize_huffman(optimize)
            .restart_mcu_rows(0) // Disable restart markers to match locked hashes
    };

    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .expect("encoder setup");
    enc.push_packed(pixels, enough::Unstoppable).expect("push");
    enc.finish().expect("encode")
}

fn hash_jpeg(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    format!("{:x}", hasher.finalize())
}

// =============================================================================
// TESTS
// =============================================================================

/// Verify that the CSV file hasn't been modified without updating the hash.
#[test]
fn test_values_file_integrity() {
    let actual_hash = hash_content(VALUES_CSV);

    if VALUES_FILE_HASH == "INITIAL_PLACEHOLDER" {
        panic!(
            "VALUES_FILE_HASH not set for {} variant. Run regenerate test first, then set it to:\n{}",
            SIMD_VARIANT, actual_hash
        );
    }

    assert_eq!(
        actual_hash, VALUES_FILE_HASH,
        "\nvalues_{}.csv was modified without updating VALUES_FILE_HASH.\n\
         If this change is intentional:\n\
         1. Archive the old file: cp tests/locked_values/values_{0}.csv tests/locked_values/history/DATE_reason.csv\n\
         2. Update VALUES_FILE_HASH for {} to: {}\n",
        SIMD_VARIANT, SIMD_VARIANT, actual_hash
    );
}

/// Verify all encoder outputs match expected values.
/// On non-x86_64 platforms: values_wide.csv needs regeneration after archmage 0.9.15 NEON changes.
/// Without yuv feature: YCbCr conversion path produces different output.
/// Run: REGENERATE_LOCKED_VALUES=1 cargo test --release -p zenjpeg --test locked_values
///   --no-default-features --features "std,yuv" -- regenerate --ignored --nocapture
#[cfg(feature = "yuv")]
#[cfg_attr(
    not(target_arch = "x86_64"),
    ignore = "values_wide.csv needs regeneration on this platform (archmage 0.9.15 changed NEON output)"
)]
#[test]
fn test_encoder_outputs() {
    // Skip if placeholder (let integrity test catch this)
    if VALUES_FILE_HASH == "INITIAL_PLACEHOLDER" {
        eprintln!("Skipping: VALUES_FILE_HASH not set for {}", SIMD_VARIANT);
        return;
    }

    let expected = parse_csv(VALUES_CSV);
    if expected.is_empty() {
        eprintln!("Warning: no values in CSV for {}", SIMD_VARIANT);
        return;
    }

    let (pixels, width, height) = load_frymire();

    let mut failures = Vec::new();

    // Test each configuration
    for (key, expected_val) in &expected {
        let jpeg = encode_config(
            &pixels,
            width,
            height,
            &key.mode,
            &key.subsampling,
            &key.huffman,
            key.quality,
        );

        let actual_hash = hash_jpeg(&jpeg);
        let actual_size = jpeg.len();

        if actual_hash != expected_val.hash {
            failures.push(format!(
                "{}_{}_{}_q{}: hash mismatch\n  expected: {}\n  actual:   {}",
                key.mode, key.subsampling, key.huffman, key.quality, expected_val.hash, actual_hash
            ));
        }

        if actual_size != expected_val.size {
            failures.push(format!(
                "{}_{}_{}_q{}: size {} != expected {} (diff: {:+})",
                key.mode,
                key.subsampling,
                key.huffman,
                key.quality,
                actual_size,
                expected_val.size,
                actual_size as i64 - expected_val.size as i64
            ));
        }
    }

    if !failures.is_empty() {
        panic!(
            "\n{} encoder output mismatches ({} variant):\n\n{}\n",
            failures.len(),
            SIMD_VARIANT,
            failures.join("\n\n")
        );
    }
}

/// Regenerate values CSV with current encoder output.
///
/// Only runs when REGENERATE_LOCKED_VALUES=1 is set.
/// Always fails after writing to force hash update.
#[test]
#[ignore]
fn regenerate_values() {
    if std::env::var("REGENERATE_LOCKED_VALUES").is_err() {
        panic!(
            "Set REGENERATE_LOCKED_VALUES=1 to regenerate.\n\
             This will update values_{}.csv and require updating VALUES_FILE_HASH.",
            SIMD_VARIANT
        );
    }

    let (pixels, width, height) = load_frymire();
    eprintln!("Loaded frymire.png: {}x{}", width, height);
    eprintln!("SIMD variant: {}", SIMD_VARIANT);

    // Generate for all configurations
    let modes = ["baseline", "progressive"];
    let subsamplings = ["444", "422", "420", "440", "xyb"];
    let huffmans = [("opt", true), ("fixed", false)];
    let qualities: [u8; 3] = [50, 75, 90];

    let mut entries = Vec::new();

    for mode in &modes {
        for subsamp in &subsamplings {
            // XYB doesn't support fixed huffman well, skip
            // Progressive mode requires optimized Huffman
            let huffman_opts: &[(&str, bool)] = if *subsamp == "xyb" || *mode == "progressive" {
                &[("opt", true)]
            } else {
                &huffmans
            };

            for (huffman_name, _) in huffman_opts {
                for quality in &qualities {
                    let jpeg = encode_config(
                        &pixels,
                        width,
                        height,
                        mode,
                        subsamp,
                        huffman_name,
                        *quality,
                    );

                    let hash = hash_jpeg(&jpeg);
                    let size = jpeg.len();

                    entries.push((mode, subsamp, huffman_name, *quality, hash.clone(), size));

                    eprintln!(
                        "  {}_{}_{}_q{}: {} bytes",
                        mode, subsamp, huffman_name, quality, size
                    );
                }
            }
        }
    }

    // Build output with header
    let reason = std::env::var("LOCKED_VALUES_REASON")
        .unwrap_or_else(|_| "[FILL IN JUSTIFICATION]".to_string());

    let mut lines = Vec::new();
    lines.push(format!(
        "# Locked encoder output values for frymire.png ({}x{})",
        width, height
    ));
    lines.push(format!("# Generated: {}", chrono_lite_date()));
    lines.push(format!("# Reason: {}", reason));
    lines.push(format!("# SIMD variant: {}", SIMD_VARIANT));
    lines.push("#".to_string());
    lines.push("# Fields: mode,subsampling,huffman,quality,simd,hash,size".to_string());

    for (mode, subsamp, huffman, quality, hash, size) in &entries {
        lines.push(format!(
            "{},{},{},{},{},{},{}",
            mode, subsamp, huffman, quality, SIMD_VARIANT, hash, size
        ));
    }

    let content = lines.join("\n") + "\n";
    let output_path = format!(
        "{}/tests/locked_values/values_{}.csv",
        env!("CARGO_MANIFEST_DIR"),
        SIMD_VARIANT
    );

    std::fs::write(&output_path, &content).expect("Failed to write CSV");
    eprintln!("\nWrote: {} ({} entries)", output_path, entries.len());

    let new_hash = hash_content(&content);
    eprintln!("\n=== UPDATE VALUES_FILE_HASH ({}) TO ===", SIMD_VARIANT);
    eprintln!("{}", new_hash);
    eprintln!("==========================================\n");

    panic!(
        "Regeneration complete. Update VALUES_FILE_HASH for {} in locked_values.rs to:\n{}\n\n\
         Then archive the old file if needed:\n\
         cp tests/locked_values/values_{0}.csv tests/locked_values/history/DATE_reason.csv",
        SIMD_VARIANT, new_hash
    );
}

// Simple date function using system time
fn chrono_lite_date() -> String {
    use std::process::Command;
    Command::new("date")
        .arg("+%Y-%m-%d")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "UNKNOWN".to_string())
}
