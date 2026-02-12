//! Sweep test: runs all collected crash reproducer files against zenjpeg.
//!
//! This test walks the crash_repro subdirectories (jpeg_decoder, zune_jpeg_extra,
//! libjpeg_turbo) and tests each JPEG file, reporting panics vs errors vs successes.
//!
//! Files are loaded at runtime (not include_bytes!) to handle the large collection
//! without bloating compile times.

use enough::Unstoppable;
use std::panic::{self, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use zenjpeg::decode::Decoder;

/// Result of testing a single file.
#[derive(Debug)]
enum TestResult {
    /// Decoded successfully
    Ok,
    /// Decoder returned an error (good - graceful rejection)
    Error(String),
    /// Decoder panicked (BAD - this is a bug)
    Panic(String),
    /// Not a JPEG file (skipped)
    NotJpeg,
}

fn test_file(path: &Path) -> TestResult {
    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(e) => return TestResult::Error(format!("read error: {}", e)),
    };

    // Check for JPEG SOI marker
    if data.len() < 2 || data[0] != 0xFF || data[1] != 0xD8 {
        return TestResult::NotJpeg;
    }

    // Test default config
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        Decoder::new().decode(&data, Unstoppable)
    }));
    match result {
        Ok(Ok(_)) => {}
        Ok(Err(e)) => return TestResult::Error(format!("{:?}", e)),
        Err(e) => {
            let msg = e
                .downcast_ref::<String>()
                .map(|s| s.clone())
                .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "unknown panic".to_string());
            return TestResult::Panic(format!("default: {}", msg));
        }
    }

    // Test with fancy upsampling disabled
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        Decoder::new()
            .fancy_upsampling(false)
            .decode(&data, Unstoppable)
    }));
    if let Err(e) = result {
        let msg = e
            .downcast_ref::<String>()
            .map(|s| s.clone())
            .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_else(|| "unknown panic".to_string());
        return TestResult::Panic(format!("fancy_upsampling(false): {}", msg));
    }

    // Test with fancy upsampling enabled
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        Decoder::new()
            .fancy_upsampling(true)
            .decode(&data, Unstoppable)
    }));
    if let Err(e) = result {
        let msg = e
            .downcast_ref::<String>()
            .map(|s| s.clone())
            .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_else(|| "unknown panic".to_string());
        return TestResult::Panic(format!("fancy_upsampling(true): {}", msg));
    }

    TestResult::Ok
}

fn sweep_directory(dir: &Path) -> (Vec<(PathBuf, String)>, usize, usize, usize) {
    let mut panics = Vec::new();
    let mut ok_count = 0;
    let mut error_count = 0;
    let mut skip_count = 0;

    if !dir.exists() {
        return (panics, ok_count, error_count, skip_count);
    }

    // Collect all files recursively
    let mut files = Vec::new();
    collect_files(dir, &mut files);
    files.sort();

    for path in files {
        match test_file(&path) {
            TestResult::Ok => ok_count += 1,
            TestResult::Error(_) => error_count += 1,
            TestResult::Panic(msg) => panics.push((path, msg)),
            TestResult::NotJpeg => skip_count += 1,
        }
    }

    (panics, ok_count, error_count, skip_count)
}

fn collect_files(dir: &Path, files: &mut Vec<PathBuf>) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.is_file() {
                files.push(path);
            } else if path.is_dir() {
                collect_files(&path, files);
            }
        }
    }
}

#[test]
fn sweep_jpeg_decoder_files() {
    let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/crash_repro/jpeg_decoder");
    let (panics, ok, errors, skipped) = sweep_directory(&base);

    eprintln!(
        "\njpeg-decoder sweep: {} ok, {} errors (graceful), {} skipped, {} PANICS",
        ok,
        errors,
        skipped,
        panics.len()
    );
    for (path, msg) in &panics {
        eprintln!("  PANIC: {} - {}", path.file_name().unwrap().to_string_lossy(), msg);
    }

    assert!(
        panics.is_empty(),
        "jpeg-decoder: {} files caused panics: {:?}",
        panics.len(),
        panics
            .iter()
            .map(|(p, m)| format!("{}: {}", p.file_name().unwrap().to_string_lossy(), m))
            .collect::<Vec<_>>()
    );
}

#[test]
fn sweep_zune_jpeg_extra_files() {
    let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/crash_repro/zune_jpeg_extra");
    let (panics, ok, errors, skipped) = sweep_directory(&base);

    eprintln!(
        "\nzune-jpeg-extra sweep: {} ok, {} errors (graceful), {} skipped, {} PANICS",
        ok,
        errors,
        skipped,
        panics.len()
    );
    for (path, msg) in &panics {
        eprintln!("  PANIC: {} - {}", path.file_name().unwrap().to_string_lossy(), msg);
    }

    assert!(
        panics.is_empty(),
        "zune-jpeg-extra: {} files caused panics: {:?}",
        panics.len(),
        panics
            .iter()
            .map(|(p, m)| format!("{}: {}", p.file_name().unwrap().to_string_lossy(), m))
            .collect::<Vec<_>>()
    );
}

#[test]
fn sweep_libjpeg_turbo_files() {
    let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/crash_repro/libjpeg_turbo");
    let (panics, ok, errors, skipped) = sweep_directory(&base);

    eprintln!(
        "\nlibjpeg-turbo sweep: {} ok, {} errors (graceful), {} skipped, {} PANICS",
        ok,
        errors,
        skipped,
        panics.len()
    );
    for (path, msg) in &panics {
        eprintln!("  PANIC: {} - {}", path.file_name().unwrap().to_string_lossy(), msg);
    }

    assert!(
        panics.is_empty(),
        "libjpeg-turbo: {} files caused panics: {:?}",
        panics.len(),
        panics
            .iter()
            .map(|(p, m)| format!("{}: {}", p.file_name().unwrap().to_string_lossy(), m))
            .collect::<Vec<_>>()
    );
}
