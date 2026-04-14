#![allow(dead_code, clippy::collapsible_if)]
//! Comprehensive ICC profile extraction tests against imageflow's test corpus.
//!
//! Walks all ICC-containing JPEGs from the imageflow image cache and verifies
//! that ICC profiles are reliably extracted at every level of the decode pipeline:
//!
//! 1. Low-level: `extract_icc_profile()` — raw APP2 marker scanner
//! 2. Mid-level: `Decoder::read_info()` — parser pipeline
//! 3. High-level: `JpegDecoderConfig::probe_header()` — zencodec trait (what imageflow uses)
//!
//! Run: cargo test --release -p zenjpeg --test icc_corpus_extraction --features "decoder,zencodec" -- --nocapture

use std::path::{Path, PathBuf};

use zenjpeg::color::icc::extract_icc_profile;

const IMAGEFLOW_CACHE: &str =
    "/home/lilith/work/imageflow/.image-cache/sources/imageflow-resources/test_inputs";

/// Discover all JPEG files under a directory tree.
fn find_jpegs(dir: &Path) -> Vec<PathBuf> {
    let mut jpegs = Vec::new();
    if !dir.exists() {
        return jpegs;
    }
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, out);
            } else if let Some(ext) = path.extension() {
                let ext = ext.to_string_lossy().to_lowercase();
                if ext == "jpg" || ext == "jpeg" {
                    out.push(path);
                }
            }
        }
    }
    walk(dir, &mut jpegs);
    jpegs.sort();
    jpegs
}

/// Check if a JPEG file contains an ICC profile at the byte level (APP2 marker scan).
fn has_icc_bytes(data: &[u8]) -> bool {
    // Look for the ICC_PROFILE signature in APP2 markers
    data.windows(12).any(|w| w == b"ICC_PROFILE\0")
}

/// Relative path from the cache root for readable test output.
fn rel_path(path: &Path) -> String {
    path.strip_prefix(IMAGEFLOW_CACHE)
        .unwrap_or(path)
        .display()
        .to_string()
}

/// Test result for a single file at a single extraction level.
#[derive(Debug)]
struct ExtractionResult {
    file: String,
    has_icc_bytes: bool,
    low_level: Option<usize>, // extract_icc_profile byte count
    mid_level: Result<Option<usize>, String>, // read_info ICC byte count
    #[cfg(feature = "zencodec")]
    high_level: Result<Option<usize>, String>, // probe_header ICC byte count
}

fn test_file(path: &Path) -> ExtractionResult {
    let data = std::fs::read(path).expect("read file");
    let name = rel_path(path);
    let has_bytes = has_icc_bytes(&data);

    // Level 1: raw extraction
    let low = extract_icc_profile(&data).map(|v| v.len());

    // Level 2: parser pipeline
    let mid = {
        let decoder = zenjpeg::decoder::Decoder::new();
        match decoder.read_info(&data) {
            Ok(info) => Ok(info.icc_profile.as_ref().map(|v| v.len())),
            Err(e) => Err(format!("{e}")),
        }
    };

    // Level 3: zencodec probe
    #[cfg(feature = "zencodec")]
    let high = {
        let config = zenjpeg::JpegDecoderConfig::new();
        match config.probe_header(&data) {
            Ok(info) => Ok(info.source_color.icc_profile.as_ref().map(|v| v.len())),
            Err(e) => Err(format!("{e}")),
        }
    };

    ExtractionResult {
        file: name,
        has_icc_bytes: has_bytes,
        low_level: low,
        mid_level: mid,
        #[cfg(feature = "zencodec")]
        high_level: high,
    }
}

#[test]
fn icc_extraction_all_levels_imageflow_corpus() {
    let cache = Path::new(IMAGEFLOW_CACHE);
    if !cache.exists() {
        eprintln!("SKIP: imageflow cache not found at {IMAGEFLOW_CACHE}");
        return;
    }

    let jpegs = find_jpegs(cache);
    if jpegs.is_empty() {
        eprintln!("SKIP: no JPEG files found in {IMAGEFLOW_CACHE}");
        return;
    }

    // Filter to only ICC-containing files
    let icc_jpegs: Vec<_> = jpegs
        .iter()
        .filter(|p| {
            let data = std::fs::read(p).unwrap_or_default();
            has_icc_bytes(&data)
        })
        .collect();

    eprintln!(
        "Found {} JPEG files, {} with ICC profiles",
        jpegs.len(),
        icc_jpegs.len()
    );

    let mut failures = Vec::new();
    let mut results = Vec::new();

    for path in &icc_jpegs {
        let r = test_file(path);

        // Check for regressions
        let mut file_failures = Vec::new();

        if r.low_level.is_none() {
            file_failures.push("extract_icc_profile returned None");
        }

        match &r.mid_level {
            Ok(None) => file_failures.push("read_info returned no ICC"),
            Err(e) => {
                file_failures.push(Box::leak(format!("read_info error: {e}").into_boxed_str()))
            }
            Ok(Some(_)) => {}
        }

        #[cfg(feature = "zencodec")]
        match &r.high_level {
            Ok(None) => file_failures.push("probe_header returned no ICC"),
            Err(e) => file_failures.push(Box::leak(
                format!("probe_header error: {e}").into_boxed_str(),
            )),
            Ok(Some(_)) => {}
        }

        // Check consistency: if low-level found ICC, mid/high should too
        if let Some(low_len) = r.low_level {
            if let Ok(Some(mid_len)) = r.mid_level {
                if low_len != mid_len {
                    file_failures.push(Box::leak(
                        format!("size mismatch: extract={low_len}, read_info={mid_len}")
                            .into_boxed_str(),
                    ));
                }
            }
        }

        if !file_failures.is_empty() {
            failures.push((r.file.clone(), file_failures));
        }

        results.push(r);
    }

    // Print summary
    eprintln!("\n=== ICC Extraction Results ===");
    for r in &results {
        let low = r
            .low_level
            .map(|n| format!("{n}B"))
            .unwrap_or_else(|| "NONE".to_string());
        let mid = match &r.mid_level {
            Ok(Some(n)) => format!("{n}B"),
            Ok(None) => "NONE".to_string(),
            Err(e) => format!("ERR({e})"),
        };
        #[cfg(feature = "zencodec")]
        let high = match &r.high_level {
            Ok(Some(n)) => format!("{n}B"),
            Ok(None) => "NONE".to_string(),
            Err(e) => format!("ERR({e})"),
        };
        #[cfg(not(feature = "zencodec"))]
        let high = "n/a";
        eprintln!(
            "  {:<70} extract={:<8} read_info={:<8} probe={:<8}",
            r.file, low, mid, high
        );
    }

    if !failures.is_empty() {
        eprintln!("\n=== FAILURES ===");
        for (file, issues) in &failures {
            eprintln!("  {file}:");
            for issue in issues {
                eprintln!("    - {issue}");
            }
        }
        panic!(
            "{} of {} ICC files had extraction failures",
            failures.len(),
            icc_jpegs.len()
        );
    }

    eprintln!(
        "\nAll {} ICC files extracted successfully at all levels",
        icc_jpegs.len()
    );
}

/// Test specific known-tricky files individually for better error reporting.
mod known_profiles {
    use super::*;

    const WIDE_GAMUT: &str = "wide-gamut";

    fn wide_gamut_dir() -> PathBuf {
        Path::new(IMAGEFLOW_CACHE).join(WIDE_GAMUT)
    }

    fn test_wide_gamut_subdir(subdir: &str) {
        let dir = wide_gamut_dir().join(subdir);
        if !dir.exists() {
            eprintln!("SKIP: {subdir} directory not found");
            return;
        }
        let jpegs = find_jpegs(&dir);
        assert!(!jpegs.is_empty(), "no JPEGs found in {subdir}");

        let mut found_icc = 0;
        for path in &jpegs {
            let data = std::fs::read(path).unwrap();
            let name = rel_path(path);

            // Check if file has ICC at byte level
            if !has_icc_bytes(&data) {
                eprintln!("  SKIP {name}: no ICC_PROFILE signature in file");
                continue;
            }

            // Low-level must find ICC
            let icc = extract_icc_profile(&data);
            assert!(
                icc.is_some(),
                "{name}: extract_icc_profile returned None (but ICC_PROFILE bytes present)"
            );
            let icc_len = icc.unwrap().len();
            assert!(icc_len > 100, "{name}: ICC too short ({icc_len} bytes)");

            // Mid-level must find ICC
            let decoder = zenjpeg::decoder::Decoder::new();
            let info = decoder
                .read_info(&data)
                .unwrap_or_else(|e| panic!("{name}: read_info failed: {e}"));
            assert!(
                info.icc_profile.is_some(),
                "{name}: read_info returned no ICC profile (but extract found {icc_len} bytes)"
            );
            let mid_len = info.icc_profile.as_ref().unwrap().len();
            assert_eq!(
                icc_len, mid_len,
                "{name}: size mismatch extract={icc_len} vs read_info={mid_len}"
            );

            eprintln!("  OK {name}: {icc_len} bytes");
            found_icc += 1;
        }
        assert!(found_icc > 0, "no ICC files found in {subdir}");
    }

    #[test]
    fn adobe_rgb() {
        test_wide_gamut_subdir("adobe-rgb");
    }

    #[test]
    fn display_p3() {
        test_wide_gamut_subdir("display-p3");
    }

    #[test]
    fn prophoto_rgb() {
        test_wide_gamut_subdir("prophoto-rgb");
    }

    #[test]
    fn rec2020_pq() {
        test_wide_gamut_subdir("rec-2020-pq");
    }

    #[test]
    fn gray_gamma_22() {
        test_wide_gamut_subdir("gray-gamma-22");
    }

    #[test]
    fn srgb_reference() {
        test_wide_gamut_subdir("srgb-reference");
    }

    /// Test the repro-icc files from real bug reports.
    #[test]
    fn repro_icc_bug_reports() {
        let dir = Path::new(IMAGEFLOW_CACHE).join("repro-icc");
        if !dir.exists() {
            eprintln!("SKIP: repro-icc directory not found");
            return;
        }
        let jpegs = find_jpegs(&dir);
        if jpegs.is_empty() {
            eprintln!("SKIP: no JPEGs in repro-icc");
            return;
        }

        for path in &jpegs {
            let data = std::fs::read(path).unwrap();
            let name = rel_path(path);

            let icc = extract_icc_profile(&data);
            if let Some(ref profile) = icc {
                let decoder = zenjpeg::decoder::Decoder::new();
                let info = decoder
                    .read_info(&data)
                    .unwrap_or_else(|e| panic!("{name}: read_info failed: {e}"));
                assert!(
                    info.icc_profile.is_some(),
                    "{name}: read_info lost ICC ({} bytes found by extract)",
                    profile.len()
                );
                eprintln!("  OK {name}: {} bytes", profile.len());
            } else {
                eprintln!("  SKIP {name}: no ICC profile (may be intentional)");
            }
        }
    }

    /// Test miscellaneous ICC-tagged files (CMYK, orientation, etc.)
    #[test]
    fn misc_icc_files() {
        let paths = [
            "orientation/Landscape_1.jpg",
            "wrenches.jpg",
            "MarsRGB_v4_sYCC_8bit.jpg",
            "cmyk_logo.jpg",
        ];

        for rel in paths {
            let path = Path::new(IMAGEFLOW_CACHE).join(rel);
            if !path.exists() {
                eprintln!("SKIP: {rel} not found");
                continue;
            }
            let data = std::fs::read(&path).unwrap();

            let icc = extract_icc_profile(&data);
            if let Some(ref profile) = icc {
                let decoder = zenjpeg::decoder::Decoder::new();
                match decoder.read_info(&data) {
                    Ok(info) => {
                        if info.icc_profile.is_some() {
                            eprintln!("  OK {rel}: {} bytes", profile.len());
                        } else {
                            panic!(
                                "{rel}: read_info lost ICC ({} bytes found by extract)",
                                profile.len()
                            );
                        }
                    }
                    Err(e) => {
                        // CMYK and unusual files may fail to parse — report but don't panic
                        eprintln!(
                            "  WARN {rel}: read_info error: {e} (ICC: {} bytes)",
                            profile.len()
                        );
                    }
                }
            } else {
                eprintln!("  SKIP {rel}: no ICC profile");
            }
        }
    }
}
