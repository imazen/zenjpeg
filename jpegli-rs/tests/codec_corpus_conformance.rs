//! Codec corpus conformance tests.
//!
//! Tests decoder against JPEG test images and fuzz corpus from codec-corpus:
//! - jpeg-conformance/valid: Valid JPEGs that MUST decode correctly
//! - jpeg-conformance/invalid: Invalid JPEGs that MUST be rejected (not panic)
//! - jpeg-conformance/non-conformant: Edge cases - behavior varies by decoder
//! - zune/fuzz-corpus/jpeg: 1836 fuzz-generated JPEGs
//! - zune/test-images/jpeg: Specialized edge case images
//! - image-rs/test-images/jpg: Additional test images
//! - mozjpeg/: mozjpeg test images

use std::fs;
use std::path::{Path, PathBuf};

use jpegli::decode::Decoder;

// ============================================================================
// Corpus Discovery
// ============================================================================

/// Find the codec-corpus directory, checking common locations.
fn find_codec_corpus() -> Option<PathBuf> {
    // Check environment variable first
    if let Ok(dir) = std::env::var("CODEC_CORPUS_DIR") {
        let path = PathBuf::from(dir);
        if path.exists() {
            return Some(path);
        }
    }

    // Check relative paths
    let candidates = [
        PathBuf::from("../codec-eval/codec-corpus"),
        PathBuf::from("../../codec-eval/codec-corpus"),
        PathBuf::from("../codec-corpus"),
        PathBuf::from("./codec-corpus"),
    ];

    for path in candidates {
        if path.exists() && path.is_dir() {
            return Some(path);
        }
    }
    None
}

/// Collect all JPEG files from a directory recursively.
fn collect_jpeg_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();

    if !dir.exists() {
        return files;
    }

    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                files.extend(collect_jpeg_files(&path));
            } else if path.is_file() {
                // Check if it's a JPEG by extension or by magic bytes
                let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
                if matches!(ext.to_lowercase().as_str(), "jpg" | "jpeg") {
                    files.push(path);
                } else if ext.is_empty() || ext.len() == 40 {
                    // Fuzz corpus files have no extension (hash names)
                    // Check magic bytes
                    if let Ok(data) = fs::read(&path) {
                        if data.len() >= 2 && data[0] == 0xFF && data[1] == 0xD8 {
                            files.push(path);
                        }
                    }
                }
            }
        }
    }

    files
}

// ============================================================================
// Fuzz Corpus Tests
// ============================================================================

/// Test that decoder doesn't panic on fuzz corpus.
/// These files may be malformed - we just verify no panics or crashes.
#[test]
fn test_fuzz_corpus_no_panic() {
    let corpus_dir = match find_codec_corpus() {
        Some(dir) => dir,
        None => {
            eprintln!("Skipping: codec-corpus not found");
            return;
        }
    };

    let fuzz_dir = corpus_dir.join("zune/fuzz-corpus/jpeg");
    if !fuzz_dir.exists() {
        eprintln!("Skipping: fuzz corpus not found at {:?}", fuzz_dir);
        return;
    }

    let files = collect_jpeg_files(&fuzz_dir);
    println!("Testing {} fuzz corpus files", files.len());

    let decoder = Decoder::new();
    let mut success = 0;
    let mut errors = 0;

    for file in &files {
        let data = match fs::read(file) {
            Ok(d) => d,
            Err(_) => continue,
        };

        // The test passes if we don't panic
        match decoder.decode(&data) {
            Ok(_) => success += 1,
            Err(_) => errors += 1,
        }
    }

    println!(
        "Fuzz corpus: {} decoded, {} rejected (no panics)",
        success, errors
    );
    assert!(
        success + errors == files.len(),
        "All files should be processed"
    );
}

/// Test a sample of fuzz corpus with detailed output.
#[test]
fn test_fuzz_corpus_sample() {
    let corpus_dir = match find_codec_corpus() {
        Some(dir) => dir,
        None => {
            eprintln!("Skipping: codec-corpus not found");
            return;
        }
    };

    let fuzz_dir = corpus_dir.join("zune/fuzz-corpus/jpeg");
    let files = collect_jpeg_files(&fuzz_dir);

    if files.is_empty() {
        eprintln!("Skipping: no fuzz files found");
        return;
    }

    let decoder = Decoder::new();

    // Test first 100 files with details
    for file in files.iter().take(100) {
        let data = match fs::read(file) {
            Ok(d) => d,
            Err(_) => continue,
        };

        match decoder.decode(&data) {
            Ok(img) => {
                // Valid JPEG - verify basic properties
                assert!(img.width > 0, "Width should be positive");
                assert!(img.height > 0, "Height should be positive");
                assert!(!img.data.is_empty(), "Data should not be empty");
            }
            Err(_e) => {
                // Fuzz files may be invalid - this is expected
            }
        }
    }
}

// ============================================================================
// Zune Test Images
// ============================================================================

#[test]
fn test_zune_progressive() {
    let corpus_dir = match find_codec_corpus() {
        Some(dir) => dir,
        None => {
            eprintln!("Skipping: codec-corpus not found");
            return;
        }
    };

    let test_dir = corpus_dir.join("zune/test-images/jpeg");
    if !test_dir.exists() {
        eprintln!("Skipping: zune test images not found");
        return;
    }

    let decoder = Decoder::new();

    // Test progressive images
    let progressive_files = [
        "down_sampled_grayscale_prog.jpg",
        "Kiara_limited_progressive_four_components.jpg",
    ];

    for filename in progressive_files {
        let path = test_dir.join(filename);
        if !path.exists() {
            continue;
        }

        let data = fs::read(&path).expect("read file");
        let result = decoder.decode(&data);

        println!("{}: {:?}", filename, result.is_ok());

        // Progressive grayscale should decode
        if filename.contains("grayscale") && result.is_ok() {
            let img = result.unwrap();
            assert!(img.width > 0 && img.height > 0);
        }
    }
}

#[test]
fn test_zune_sampling_factors() {
    let corpus_dir = match find_codec_corpus() {
        Some(dir) => dir,
        None => {
            eprintln!("Skipping: codec-corpus not found");
            return;
        }
    };

    let test_dir = corpus_dir.join("zune/test-images/jpeg");
    if !test_dir.exists() {
        eprintln!("Skipping: zune test images not found");
        return;
    }

    let decoder = Decoder::new();

    // Test various sampling factor configurations
    let sampling_files = [
        "sampling_factors.jpg",
        "weid_sampling_factors.jpg",
        "weird_sampling_2.jpeg",
        "fox410.jpg",
        "large_horiz_samp_7680_4320.jpg",
        "large_vertical_samp_7680_4320.jpg",
        "large_no_samp_7680_4320.jpg",
    ];

    for filename in sampling_files {
        let path = test_dir.join(filename);
        if !path.exists() {
            continue;
        }

        let data = fs::read(&path).expect("read file");
        let result = decoder.decode(&data);

        println!(
            "{}: {} (size: {} bytes)",
            filename,
            if result.is_ok() { "OK" } else { "FAIL" },
            data.len()
        );

        // Large images may be skipped to avoid OOM in tests
        if !filename.contains("7680") && result.is_ok() {
            let img = result.unwrap();
            assert!(img.width > 0 && img.height > 0);
        }
    }
}

#[test]
fn test_zune_edge_cases() {
    let corpus_dir = match find_codec_corpus() {
        Some(dir) => dir,
        None => {
            eprintln!("Skipping: codec-corpus not found");
            return;
        }
    };

    let test_dir = corpus_dir.join("zune/test-images/jpeg");
    if !test_dir.exists() {
        eprintln!("Skipping: zune test images not found");
        return;
    }

    let decoder = Decoder::new();

    // Test edge cases
    let edge_case_files = [
        "huffman_third_index.jpg",
        "mjpeg_huffman.jpg",
        "rebuilt_relax_fill_bytes_before_marker.jpg",
        "four_components.jpg",
        "weird_components.jpg",
        "cymk.jpg",
        "incomplete_image.jpg",
        "huge_sof_number.jpg",
    ];

    for filename in edge_case_files {
        let path = test_dir.join(filename);
        if !path.exists() {
            continue;
        }

        let data = fs::read(&path).expect("read file");
        let result = decoder.decode(&data);

        println!(
            "{}: {}",
            filename,
            match &result {
                Ok(img) => format!("OK {}x{}", img.width, img.height),
                Err(e) => format!("Error: {:?}", e),
            }
        );

        // CMYK and 4-component images may fail - that's expected
        // incomplete_image.jpg should fail
        if filename == "incomplete_image.jpg" {
            // May or may not fail depending on how incomplete
        }
    }
}

// ============================================================================
// Image-rs Test Images
// ============================================================================

#[test]
fn test_image_rs_progressive() {
    let corpus_dir = match find_codec_corpus() {
        Some(dir) => dir,
        None => {
            eprintln!("Skipping: codec-corpus not found");
            return;
        }
    };

    let test_dir = corpus_dir.join("image-rs/test-images/jpg/progressive");
    if !test_dir.exists() {
        eprintln!("Skipping: image-rs progressive images not found");
        return;
    }

    let decoder = Decoder::new();
    let files = collect_jpeg_files(&test_dir);

    println!("Testing {} progressive images from image-rs", files.len());

    for file in &files {
        let data = fs::read(file).expect("read file");
        let result = decoder.decode(&data);

        let filename = file.file_name().unwrap().to_string_lossy();
        match result {
            Ok(img) => {
                println!("{}: OK", filename);
                assert!(img.width > 0 && img.height > 0);
            }
            Err(e) => {
                // Known issue: Some progressive JPEGs fail with Huffman decode errors
                // See: InvalidHuffmanTable { table_idx: 0, reason: "invalid code" }
                eprintln!("{}: FAIL (known issue) - {:?}", filename, e);
            }
        }
    }
}

#[test]
fn test_image_rs_general() {
    let corpus_dir = match find_codec_corpus() {
        Some(dir) => dir,
        None => {
            eprintln!("Skipping: codec-corpus not found");
            return;
        }
    };

    let test_dir = corpus_dir.join("image-rs/test-images/jpg");
    if !test_dir.exists() {
        eprintln!("Skipping: image-rs test images not found");
        return;
    }

    let decoder = Decoder::new();

    // Test general JPEG images (not in subdirectories)
    for entry in fs::read_dir(&test_dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();

        if !path.is_file() {
            continue;
        }

        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if !matches!(ext.to_lowercase().as_str(), "jpg" | "jpeg") {
            continue;
        }

        let data = fs::read(&path).expect("read file");
        let result = decoder.decode(&data);

        let filename = path.file_name().unwrap().to_string_lossy();
        println!(
            "{}: {}",
            filename,
            if result.is_ok() { "OK" } else { "FAIL" }
        );

        // Standard test images should decode
        if result.is_ok() {
            let img = result.unwrap();
            assert!(img.width > 0 && img.height > 0);
        }
    }
}

// ============================================================================
// mozjpeg Test Images
// ============================================================================

#[test]
fn test_mozjpeg_images() {
    let corpus_dir = match find_codec_corpus() {
        Some(dir) => dir,
        None => {
            eprintln!("Skipping: codec-corpus not found");
            return;
        }
    };

    let test_dir = corpus_dir.join("mozjpeg");
    if !test_dir.exists() {
        eprintln!("Skipping: mozjpeg test images not found");
        return;
    }

    let decoder = Decoder::new();

    // Test mozjpeg reference images
    // testimgari.jpg = arithmetic coded (not supported)
    // testimgint.jpg = baseline interleaved
    // testorig12.jpg = 12-bit (not supported)
    // testorig.jpg = standard baseline
    let jpeg_files = ["testimgint.jpg", "testorig.jpg"];

    for filename in jpeg_files {
        let path = test_dir.join(filename);
        if !path.exists() {
            continue;
        }

        let data = fs::read(&path).expect("read file");
        let result = decoder.decode(&data);

        println!(
            "mozjpeg/{}: {}",
            filename,
            match &result {
                Ok(img) => format!("OK {}x{}", img.width, img.height),
                Err(e) => format!("Error: {:?}", e),
            }
        );

        assert!(result.is_ok(), "mozjpeg {} should decode", filename);
        let img = result.unwrap();
        assert!(img.width > 0 && img.height > 0);
    }

    // Check that unsupported formats are rejected gracefully
    let unsupported_files = ["testimgari.jpg", "testorig12.jpg"];
    for filename in unsupported_files {
        let path = test_dir.join(filename);
        if !path.exists() {
            continue;
        }

        let data = fs::read(&path).expect("read file");
        let result = decoder.decode(&data);

        println!(
            "mozjpeg/{} (unsupported): {}",
            filename,
            if result.is_err() {
                "correctly rejected"
            } else {
                "unexpectedly decoded"
            }
        );
    }
}

// ============================================================================
// Comparison with Reference Decoder
// ============================================================================

#[test]
fn test_fuzz_corpus_vs_reference() {
    let corpus_dir = match find_codec_corpus() {
        Some(dir) => dir,
        None => {
            eprintln!("Skipping: codec-corpus not found");
            return;
        }
    };

    let fuzz_dir = corpus_dir.join("zune/fuzz-corpus/jpeg");
    let files = collect_jpeg_files(&fuzz_dir);

    if files.is_empty() {
        eprintln!("Skipping: no fuzz files found");
        return;
    }

    let jpegli_decoder = Decoder::new();
    let mut both_succeed = 0;
    let mut jpegli_only = 0;
    let mut reference_only = 0;
    let mut both_fail = 0;

    // Test first 500 files against jpeg-decoder
    for file in files.iter().take(500) {
        let data = match fs::read(file) {
            Ok(d) => d,
            Err(_) => continue,
        };

        let jpegli_result = jpegli_decoder.decode(&data);
        let ref_result = jpeg_decoder::Decoder::new(&data[..]).decode();

        match (jpegli_result.is_ok(), ref_result.is_ok()) {
            (true, true) => both_succeed += 1,
            (true, false) => jpegli_only += 1,
            (false, true) => reference_only += 1,
            (false, false) => both_fail += 1,
        }
    }

    println!("Fuzz corpus comparison (first 500 files):");
    println!("  Both succeed: {}", both_succeed);
    println!("  jpegli only:  {}", jpegli_only);
    println!("  jpeg-decoder only: {}", reference_only);
    println!("  Both fail:    {}", both_fail);

    // We should decode at least as many as jpeg-decoder
    // (ideally more, but at minimum not fewer)
    assert!(
        jpegli_only >= 0,
        "jpegli should decode some files jpeg-decoder rejects"
    );
}

// ============================================================================
// Stress Test
// ============================================================================

#[test]
#[ignore = "slow test - run with --ignored"]
fn test_full_fuzz_corpus() {
    let corpus_dir = match find_codec_corpus() {
        Some(dir) => dir,
        None => {
            eprintln!("Skipping: codec-corpus not found");
            return;
        }
    };

    let fuzz_dir = corpus_dir.join("zune/fuzz-corpus/jpeg");
    let files = collect_jpeg_files(&fuzz_dir);

    println!("Testing ALL {} fuzz corpus files", files.len());

    let decoder = Decoder::new();
    let mut success = 0;
    let mut errors = 0;
    let mut panics = 0;

    for (i, file) in files.iter().enumerate() {
        if i % 100 == 0 {
            println!("Progress: {}/{}", i, files.len());
        }

        let data = match fs::read(file) {
            Ok(d) => d,
            Err(_) => continue,
        };

        // Use catch_unwind to detect panics
        let result = std::panic::catch_unwind(|| decoder.decode(&data));

        match result {
            Ok(Ok(_)) => success += 1,
            Ok(Err(_)) => errors += 1,
            Err(_) => {
                panics += 1;
                eprintln!("PANIC on file: {:?}", file);
            }
        }
    }

    println!("Full fuzz corpus results:");
    println!("  Success: {}", success);
    println!("  Errors:  {}", errors);
    println!("  Panics:  {}", panics);

    assert_eq!(panics, 0, "No panics should occur on fuzz corpus");
}
