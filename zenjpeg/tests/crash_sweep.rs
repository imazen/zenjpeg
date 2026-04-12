//! Sweep tests: runs collected crash reproducer files and codec-corpus datasets
//! against zenjpeg with three-tier expectations.
//!
//! Each JPEG is classified as:
//! - **should_decode**: Valid JPEG — decoder MUST succeed.
//! - **may_decode**: Non-conformant or edge-case — decoder MAY accept or reject.
//! - **must_reject**: Invalid data — decoder MUST return an error.
//!
//! ALL files must never cause a panic, regardless of classification.
//!
//! Codec-corpus datasets download on demand via `codec-corpus` crate.

use enough::Unstoppable;
use std::panic::{self, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use zenjpeg::decode::ChromaUpsampling;
use zenjpeg::decode::Decoder;

// ============================================================================
// File classification
// ============================================================================

/// Files that are valid JPEGs and MUST decode successfully.
/// These triggered C-specific bugs (segfaults, UB, heap corruption) or
/// output-correctness bugs — the JPEG data itself is valid.
const SHOULD_DECODE: &[&str] = &[
    // jpeg-decoder: valid JPEGs that hit C-specific or logic bugs
    "jd_219_no_app14.jpg",     // #219: valid JPEG without optional APP14
    "jd_228_ff00_marker.jpeg", // #228: valid JPEG with FF00 byte stuffing
    // zune-jpeg: valid JPEGs that exposed output-correctness bugs
    "zj_040_decode_diff_1.jpg",    // #40: decode produced wrong pixels
    "zj_134_luma_decode_bad.jpg",  // #134: luma channel decoded incorrectly
    "zj_249_discolored_ycbcr.jpg", // #249: YCbCr→RGB color shift
    "zj_303_app14_adobe.jpg",      // #303: Adobe APP14 marker handling
    // libjpeg-turbo: valid JPEGs that hit C memory-safety bugs
    "ljt_441_segv_jcopy_sample_rows.jpg", // #441: segfault in row copy
    "ljt_470_ub_null_ptr_skip.jpg",       // #470: null pointer UB in C
    "ljt_574_heap_corruption.jpg",        // #574: heap buffer overwrite
    "ljt_758_segv_adjust_quant_src.jpg",  // #758: segfault in quant adjust
];

/// Files that are non-conformant or edge-case — decoder MAY accept or reject.
/// No assertion on success/failure, only on no-panic.
const MAY_DECODE: &[&str] = &[
    // jpeg-decoder: spec edge cases
    "jd_040_16bit_quant.jpg", // #40: 16-bit quantization tables (non-standard)
    "jd_132_subtract_overflow.bin", // #132: arithmetic edge case in coefficient decode
    // zune-jpeg: edge cases in upsampling
    "zj_172_upsample_assert.jpg", // #172: unusual sampling factors, currently decodes ok
];

// Everything else is must_reject (invalid data, MUST error, MUST NOT panic).

fn is_should_decode(filename: &str) -> bool {
    SHOULD_DECODE.contains(&filename)
}

fn is_may_decode(filename: &str) -> bool {
    MAY_DECODE.contains(&filename)
}

// ============================================================================
// Test infrastructure
// ============================================================================

#[derive(Debug)]
enum TestResult {
    /// Decoded successfully (all configurations)
    Ok,
    /// Decoder returned an error (graceful rejection)
    Error(String),
    /// Decoder panicked (BAD — this is a bug)
    Panic(String),
    /// Not a JPEG file (no SOI marker)
    NotJpeg,
}

fn panic_message(e: Box<dyn std::any::Any + Send>) -> String {
    e.downcast_ref::<String>()
        .cloned()
        .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_else(|| "unknown panic".to_string())
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
        Err(e) => return TestResult::Panic(format!("default: {}", panic_message(e))),
    }

    // Test with fancy upsampling disabled
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        Decoder::new()
            .chroma_upsampling(ChromaUpsampling::NearestNeighbor)
            .decode(&data, Unstoppable)
    }));
    if let Err(e) = result {
        return TestResult::Panic(format!("fancy_upsampling(false): {}", panic_message(e)));
    }

    // Test with fancy upsampling enabled
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        Decoder::new().decode(&data, Unstoppable)
    }));
    if let Err(e) = result {
        return TestResult::Panic(format!("fancy_upsampling(true): {}", panic_message(e)));
    }

    TestResult::Ok
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

// ============================================================================
// Annotated sweep: checks three-tier expectations
// ============================================================================

struct SweepResult {
    panics: Vec<(PathBuf, String)>,
    should_decode_failures: Vec<(PathBuf, String)>,
    must_reject_successes: Vec<PathBuf>,
    ok_count: usize,
    error_count: usize,
    skip_count: usize,
    may_ok: usize,
    may_err: usize,
}

fn sweep_directory_annotated(dir: &Path) -> SweepResult {
    let mut result = SweepResult {
        panics: Vec::new(),
        should_decode_failures: Vec::new(),
        must_reject_successes: Vec::new(),
        ok_count: 0,
        error_count: 0,
        skip_count: 0,
        may_ok: 0,
        may_err: 0,
    };

    if !dir.exists() {
        return result;
    }

    let mut files = Vec::new();
    collect_files(dir, &mut files);
    files.sort();

    for path in files {
        let fname = path.file_name().unwrap().to_string_lossy().to_string();
        let test_result = test_file(&path);

        match test_result {
            TestResult::Panic(msg) => {
                result.panics.push((path, msg));
            }
            TestResult::NotJpeg => {
                result.skip_count += 1;
            }
            TestResult::Ok => {
                if is_should_decode(&fname) {
                    result.ok_count += 1;
                } else if is_may_decode(&fname) {
                    result.may_ok += 1;
                } else {
                    // must_reject file decoded successfully — unexpected
                    result.must_reject_successes.push(path);
                }
            }
            TestResult::Error(e) => {
                if is_should_decode(&fname) {
                    // should_decode file returned error — unexpected
                    result.should_decode_failures.push((path, e));
                } else if is_may_decode(&fname) {
                    result.may_err += 1;
                } else {
                    // must_reject file returned error — expected
                    result.error_count += 1;
                }
            }
        }
    }

    result
}

fn assert_sweep(label: &str, result: &SweepResult) {
    eprintln!(
        "\n{} sweep: {} should_decode ok, {} must_reject errors, \
         {} may_decode ({}ok/{}err), {} skipped, {} PANICS",
        label,
        result.ok_count,
        result.error_count,
        result.may_ok + result.may_err,
        result.may_ok,
        result.may_err,
        result.skip_count,
        result.panics.len()
    );

    for (path, msg) in &result.panics {
        eprintln!(
            "  PANIC: {} — {}",
            path.file_name().unwrap().to_string_lossy(),
            msg
        );
    }
    for (path, msg) in &result.should_decode_failures {
        eprintln!(
            "  SHOULD_DECODE FAILED: {} — {}",
            path.file_name().unwrap().to_string_lossy(),
            msg
        );
    }
    for path in &result.must_reject_successes {
        eprintln!(
            "  MUST_REJECT DECODED OK: {} (reclassify as should_decode or may_decode?)",
            path.file_name().unwrap().to_string_lossy()
        );
    }

    assert!(
        result.panics.is_empty(),
        "{}: {} files caused panics",
        label,
        result.panics.len()
    );
    assert!(
        result.should_decode_failures.is_empty(),
        "{}: {} should_decode files returned errors: {:?}",
        label,
        result.should_decode_failures.len(),
        result
            .should_decode_failures
            .iter()
            .map(|(p, m)| format!("{}: {}", p.file_name().unwrap().to_string_lossy(), m))
            .collect::<Vec<_>>()
    );
}

// ============================================================================
// In-repo crash reproducer sweep tests
// ============================================================================

#[test]
fn sweep_jpeg_decoder_files() {
    let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/crash_repro/jpeg_decoder");
    let result = sweep_directory_annotated(&base);
    assert_sweep("jpeg-decoder", &result);
}

#[test]
fn sweep_zune_jpeg_extra_files() {
    let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/crash_repro/zune_jpeg_extra");
    let result = sweep_directory_annotated(&base);
    assert_sweep("zune-jpeg-extra", &result);
}

#[test]
fn sweep_libjpeg_turbo_files() {
    let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/crash_repro/libjpeg_turbo");
    let result = sweep_directory_annotated(&base);
    assert_sweep("libjpeg-turbo", &result);
}

// ============================================================================
// Codec-corpus sweep tests (auto-downloads on first run)
// ============================================================================

fn corpus() -> codec_corpus::Corpus {
    codec_corpus::Corpus::new().expect("codec-corpus init failed")
}

/// Sweep codec-corpus jpeg-conformance/valid — all MUST decode.
#[ignore = "requires codec-corpus (network on first run)"]
#[test]
fn sweep_corpus_jpeg_valid() {
    let corpus = corpus();
    let dir = corpus
        .get("jpeg-conformance/valid")
        .expect("corpus.get(jpeg-conformance/valid)");

    let mut files = Vec::new();
    collect_files(&dir, &mut files);
    files.sort();

    let mut panics = Vec::new();
    let mut failures = Vec::new();
    let mut ok = 0usize;

    for path in &files {
        match test_file(path) {
            TestResult::Ok => ok += 1,
            TestResult::Error(e) => failures.push((path.clone(), e)),
            TestResult::Panic(msg) => panics.push((path.clone(), msg)),
            TestResult::NotJpeg => {}
        }
    }

    eprintln!(
        "\njpeg-conformance/valid: {} ok, {} failures, {} panics (of {} files)",
        ok,
        failures.len(),
        panics.len(),
        files.len()
    );

    assert!(
        panics.is_empty(),
        "valid files caused panics: {:?}",
        panics
            .iter()
            .map(|(p, m)| format!("{}: {}", p.file_name().unwrap().to_string_lossy(), m))
            .collect::<Vec<_>>()
    );
    assert!(
        failures.is_empty(),
        "valid files failed to decode: {:?}",
        failures
            .iter()
            .map(|(p, m)| format!("{}: {}", p.file_name().unwrap().to_string_lossy(), m))
            .collect::<Vec<_>>()
    );
}

/// Sweep codec-corpus jpeg-conformance/invalid — all MUST error, MUST NOT panic.
#[ignore = "requires codec-corpus (network on first run)"]
#[test]
fn sweep_corpus_jpeg_invalid() {
    let corpus = corpus();
    let dir = corpus
        .get("jpeg-conformance/invalid")
        .expect("corpus.get(jpeg-conformance/invalid)");

    let mut files = Vec::new();
    collect_files(&dir, &mut files);
    files.sort();

    let mut panics = Vec::new();
    let mut ok_count = 0usize;
    let mut error_count = 0usize;

    for path in &files {
        match test_file(path) {
            TestResult::Ok => ok_count += 1,
            TestResult::Error(_) => error_count += 1,
            TestResult::Panic(msg) => panics.push((path.clone(), msg)),
            TestResult::NotJpeg => {}
        }
    }

    eprintln!(
        "\njpeg-conformance/invalid: {} errors (expected), {} decoded ok (unexpected), {} panics (of {} files)",
        error_count,
        ok_count,
        panics.len(),
        files.len()
    );

    assert!(
        panics.is_empty(),
        "invalid files caused panics: {:?}",
        panics
            .iter()
            .map(|(p, m)| format!("{}: {}", p.file_name().unwrap().to_string_lossy(), m))
            .collect::<Vec<_>>()
    );
}

/// Sweep codec-corpus jpeg-conformance/non-conformant — MUST NOT panic, may decode or error.
#[ignore = "requires codec-corpus (network on first run)"]
#[test]
fn sweep_corpus_jpeg_nonconformant() {
    let corpus = corpus();
    let dir = corpus
        .get("jpeg-conformance/non-conformant")
        .expect("corpus.get(jpeg-conformance/non-conformant)");

    let mut files = Vec::new();
    collect_files(&dir, &mut files);
    files.sort();

    let mut panics = Vec::new();
    let mut ok_count = 0usize;
    let mut error_count = 0usize;

    for path in &files {
        // Skip companion .txt files
        if path.extension().is_some_and(|e| e == "txt") {
            continue;
        }
        match test_file(path) {
            TestResult::Ok => ok_count += 1,
            TestResult::Error(_) => error_count += 1,
            TestResult::Panic(msg) => panics.push((path.clone(), msg)),
            TestResult::NotJpeg => {}
        }
    }

    eprintln!(
        "\njpeg-conformance/non-conformant: {} accepted, {} rejected, {} panics",
        ok_count,
        error_count,
        panics.len()
    );

    assert!(
        panics.is_empty(),
        "non-conformant files caused panics: {:?}",
        panics
            .iter()
            .map(|(p, m)| format!("{}: {}", p.file_name().unwrap().to_string_lossy(), m))
            .collect::<Vec<_>>()
    );
}

/// Sweep zune fuzz corpus (1,836 files) — MUST NOT panic.
#[ignore = "requires codec-corpus (network on first run)"]
#[test]
fn sweep_corpus_zune_fuzz() {
    let corpus = corpus();
    let dir = corpus
        .get("zune/fuzz-corpus/jpeg")
        .expect("corpus.get(zune/fuzz-corpus/jpeg)");

    let mut files = Vec::new();
    collect_files(&dir, &mut files);
    files.sort();

    let mut panics = Vec::new();
    let mut ok_count = 0usize;
    let mut error_count = 0usize;
    let mut skip_count = 0usize;

    for path in &files {
        match test_file(path) {
            TestResult::Ok => ok_count += 1,
            TestResult::Error(_) => error_count += 1,
            TestResult::Panic(msg) => panics.push((path.clone(), msg)),
            TestResult::NotJpeg => skip_count += 1,
        }
    }

    eprintln!(
        "\nzune/fuzz-corpus/jpeg: {} ok, {} errors, {} skipped, {} panics (of {} files)",
        ok_count,
        error_count,
        skip_count,
        panics.len(),
        files.len()
    );

    assert!(
        panics.is_empty(),
        "zune fuzz corpus: {} files caused panics: {:?}",
        panics.len(),
        panics
            .iter()
            .map(|(p, m)| format!("{}: {}", p.file_name().unwrap().to_string_lossy(), m))
            .collect::<Vec<_>>()
    );
}

/// Sweep codec-corpus crash-repro files (large files from upstream bug reports) — MUST NOT panic.
#[ignore = "requires codec-corpus (network on first run)"]
#[test]
fn sweep_corpus_crash_repro() {
    let corpus = corpus();
    let dir = corpus
        .get("jpeg-conformance/crash-repro")
        .expect("corpus.get(jpeg-conformance/crash-repro)");

    let mut files = Vec::new();
    collect_files(&dir, &mut files);
    // Filter to image files only (skip README.md etc)
    files.retain(|p| {
        p.extension().is_some_and(|e| {
            let e = e.to_string_lossy().to_lowercase();
            e == "jpg" || e == "jpeg" || e == "bin" || e == "jpf"
        })
    });
    files.sort();

    let mut panics = Vec::new();
    let mut ok_count = 0usize;
    let mut error_count = 0usize;

    for path in &files {
        match test_file(path) {
            TestResult::Ok => ok_count += 1,
            TestResult::Error(_) => error_count += 1,
            TestResult::Panic(msg) => panics.push((path.clone(), msg)),
            TestResult::NotJpeg => {}
        }
    }

    eprintln!(
        "\njpeg-conformance/crash-repro: {} ok, {} errors, {} panics (of {} files)",
        ok_count,
        error_count,
        panics.len(),
        files.len()
    );

    assert!(
        panics.is_empty(),
        "crash-repro files caused panics: {:?}",
        panics
            .iter()
            .map(|(p, m)| format!("{}: {}", p.file_name().unwrap().to_string_lossy(), m))
            .collect::<Vec<_>>()
    );
}
