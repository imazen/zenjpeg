//! Common test utilities for zenjpeg tests.
//!
//! Provides path resolution for testdata files without hardcoded paths.
//!
//! ## Environment Variables
//! - `CPP_TESTDATA_DIR`: Path to C++ generated .testdata files
//! - `JPEGLI_TESTDATA`: Path to jpegli testdata directory
//! - `CODEC_CORPUS_DIR`: Path to codec-corpus repository
//!
//! ## Panics
//! Functions panic with helpful messages if resources aren't found.
//! Use `try_*` variants for optional resources.

use std::path::PathBuf;

/// Get path to C++ generated .testdata files.
///
/// # Panics
/// Panics if the file cannot be found. Set `CPP_TESTDATA_DIR` env var.
#[track_caller]
pub fn get_cpp_testdata_path(filename: &str) -> PathBuf {
    try_get_cpp_testdata_path(filename).unwrap_or_else(|| {
        panic!(
            "C++ testdata file '{}' not found.\n\
             Set CPP_TESTDATA_DIR environment variable to the directory containing .testdata files.\n\
             Generate testdata with: GENERATE_RUST_TEST_DATA=1 cjpegli input.png output.jpg",
            filename
        )
    })
}

/// Try to get path to C++ generated .testdata files. Returns None if not found.
pub fn try_get_cpp_testdata_path(filename: &str) -> Option<PathBuf> {
    // Check environment variable first
    if let Ok(dir) = std::env::var("CPP_TESTDATA_DIR") {
        let path = PathBuf::from(dir).join(filename);
        if path.exists() {
            return Some(path);
        }
    }

    // Check relative to manifest dir
    if let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") {
        let candidates = [
            PathBuf::from(&manifest).join("testdata").join(filename),
            PathBuf::from(&manifest).join("../testdata").join(filename),
            PathBuf::from(&manifest).join("../jpegli").join(filename),
            PathBuf::from(&manifest).join("../jpegli-rs/internal/jpegli-cpp").join(filename),
            PathBuf::from(filename),
        ];
        for path in candidates {
            if path.exists() {
                return Some(path);
            }
        }
    }

    None
}

/// Get path to the jpegli testdata directory.
///
/// # Panics
/// Panics if directory cannot be found. Set `JPEGLI_TESTDATA` env var.
#[track_caller]
pub fn get_jpegli_testdata_dir() -> PathBuf {
    try_get_jpegli_testdata_dir().unwrap_or_else(|| {
        panic!(
            "Jpegli testdata directory not found.\n\
             Set JPEGLI_TESTDATA environment variable to the testdata directory.\n\
             Expected structure: $JPEGLI_TESTDATA/jxl/flower/flower_small.rgb.png"
        )
    })
}

/// Try to get path to the jpegli testdata directory. Returns None if not found.
pub fn try_get_jpegli_testdata_dir() -> Option<PathBuf> {
    // Check environment variable first
    if let Ok(dir) = std::env::var("JPEGLI_TESTDATA") {
        let path = PathBuf::from(dir);
        if path.exists() {
            return Some(path);
        }
    }

    // Check relative to manifest dir
    if let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") {
        let candidates = [
            PathBuf::from(&manifest).join("testdata"),
            PathBuf::from(&manifest).join("../testdata"),
            PathBuf::from(&manifest).join("../jpegli/testdata"),
            PathBuf::from(&manifest).join("../jpegli-rs/internal/jpegli-cpp/testdata"),
        ];
        for path in candidates {
            if path.exists() {
                return Some(path);
            }
        }
    }

    None
}

/// Get path to the flower_small test image.
///
/// # Panics
/// Panics if image cannot be found.
#[track_caller]
pub fn get_flower_small_path() -> PathBuf {
    try_get_flower_small_path().unwrap_or_else(|| {
        panic!(
            "Test image flower_small.rgb.png not found.\n\
             Set JPEGLI_TESTDATA environment variable or ensure testdata is available."
        )
    })
}

/// Try to get path to the flower_small test image.
pub fn try_get_flower_small_path() -> Option<PathBuf> {
    let path = try_get_jpegli_testdata_dir()?.join("jxl/flower/flower_small.rgb.png");
    if path.exists() {
        Some(path)
    } else {
        None
    }
}

/// Get path to the codec-corpus directory.
///
/// # Panics
/// Panics if directory cannot be found.
///
/// To fetch codec-corpus:
/// ```bash
/// git clone --depth 1 https://github.com/imazen/codec-corpus ../codec-corpus
/// ```
#[track_caller]
pub fn get_codec_corpus_dir() -> PathBuf {
    try_get_codec_corpus_dir().unwrap_or_else(|| {
        panic!(
            "codec-corpus directory not found.\n\
             Set CODEC_CORPUS_DIR environment variable or clone:\n\
             git clone --depth 1 https://github.com/imazen/codec-corpus ../codec-corpus"
        )
    })
}

/// Try to get path to codec-corpus. Returns None if not found.
pub fn try_get_codec_corpus_dir() -> Option<PathBuf> {
    // Check environment variable first
    if let Ok(dir) = std::env::var("CODEC_CORPUS_DIR") {
        let path = PathBuf::from(dir);
        if path.exists() {
            return Some(path);
        }
    }

    // Check relative to manifest dir
    if let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") {
        let candidates = [
            PathBuf::from(&manifest).join("../codec-corpus"),
            PathBuf::from(&manifest).join("../../codec-corpus"),
            PathBuf::from(&manifest).join("../codec-comparison/codec-corpus"),
            PathBuf::from(&manifest).join("codec-corpus"),
        ];
        for path in candidates {
            if path.exists() {
                return Some(path);
            }
        }
    }

    None
}

/// Get path to kodim01.png from codec-corpus.
#[track_caller]
pub fn get_kodim01_path() -> PathBuf {
    try_get_kodim01_path().unwrap_or_else(|| {
        panic!(
            "kodim01.png not found in codec-corpus.\n\
             Set CODEC_CORPUS_DIR or clone:\n\
             git clone --depth 1 https://github.com/imazen/codec-corpus ../codec-corpus"
        )
    })
}

/// Try to get path to kodim01.png.
pub fn try_get_kodim01_path() -> Option<PathBuf> {
    let path = try_get_codec_corpus_dir()?.join("kodak/kodim01.png");
    if path.exists() {
        Some(path)
    } else {
        None
    }
}

/// Macro to skip test if optional resource is missing (for truly optional tests).
/// Use sparingly - prefer panicking for missing required resources.
#[macro_export]
macro_rules! skip_if_unavailable {
    ($opt:expr, $msg:literal) => {
        match $opt {
            Some(v) => v,
            None => {
                eprintln!("SKIPPING TEST: {}", $msg);
                return;
            }
        }
    };
}
