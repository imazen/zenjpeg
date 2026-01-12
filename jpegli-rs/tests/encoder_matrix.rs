//! Comprehensive encoder configuration matrix test.
//!
//! Tests ALL combinations of jpegli encoder settings and verifies:
//! 1. Valid combinations produce decodable JPEGs with all major decoders
//! 2. Invalid combinations (e.g., progressive + fixed Huffman) error appropriately
//!
//! This is the "loop test" to run while developing to ensure nothing breaks.
//!
//! Run with:
//! ```
//! cargo test --test encoder_matrix -- --nocapture
//! ```

use jpegli::{JpegMode, PixelFormat, Quality, StreamingEncoder, Subsampling};

/// Result of testing one encoder configuration
#[derive(Debug)]
struct MatrixResult {
    config: String,
    encode_result: EncodeOutcome,
    jpegli_decode: bool,
    zune_jpeg: bool,
    file_size: usize,
}

#[derive(Debug, Clone, PartialEq)]
enum EncodeOutcome {
    Success,
    ExpectedError(String),
    UnexpectedError(String),
}

/// Generate a test image
fn generate_test_image(width: usize, height: usize, channels: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height * channels);
    for y in 0..height {
        for x in 0..width {
            data.push((x * 255 / width) as u8);
            if channels >= 3 {
                data.push((y * 255 / height) as u8);
                data.push(((x + y) * 127 / (width + height)) as u8);
            }
            if channels == 4 {
                data.push(255); // Alpha
            }
        }
    }
    data
}

/// Decode with jpegli-rs
fn decode_jpegli(data: &[u8]) -> bool {
    jpegli::Decoder::new().decode(data).is_ok()
}

/// Decode with zune-jpeg
fn decode_zune_jpeg(data: &[u8]) -> bool {
    use zune_jpeg::zune_core::bytestream::ZCursor;
    use zune_jpeg::JpegDecoder;
    let cursor = ZCursor::new(data);
    let mut decoder = JpegDecoder::new(cursor);
    decoder.decode().is_ok()
}

/// Test a single encoder configuration
fn test_config(
    width: u32,
    height: u32,
    pixel_format: PixelFormat,
    mode: JpegMode,
    subsampling: Subsampling,
    optimize_huffman: bool,
    use_xyb: bool,
    quality: f32,
) -> MatrixResult {
    let channels = pixel_format.bytes_per_pixel();
    let data = generate_test_image(width as usize, height as usize, channels);

    let config_name = format!(
        "{:?}/{:?}/{:?}/huff={}/xyb={}/Q{}",
        pixel_format,
        mode,
        subsampling,
        if optimize_huffman { "opt" } else { "fix" },
        use_xyb,
        quality as u32
    );

    let encode_result = StreamingEncoder::new(width, height)
        .pixel_format(pixel_format)
        .mode(mode)
        .subsampling(subsampling)
        .optimize_huffman(optimize_huffman)
        .use_xyb(use_xyb)
        .quality(Quality::from_quality(quality))
        .encode_all(&data);

    match encode_result {
        Ok(jpeg_data) => {
            let jpegli = decode_jpegli(&jpeg_data);
            let zune = decode_zune_jpeg(&jpeg_data);

            MatrixResult {
                config: config_name,
                encode_result: EncodeOutcome::Success,
                jpegli_decode: jpegli,
                zune_jpeg: zune,
                file_size: jpeg_data.len(),
            }
        }
        Err(e) => {
            let err_str = format!("{}", e);
            // Check if this is an expected error
            let is_expected = err_str.contains("Progressive mode with fixed Huffman");

            MatrixResult {
                config: config_name,
                encode_result: if is_expected {
                    EncodeOutcome::ExpectedError(err_str)
                } else {
                    EncodeOutcome::UnexpectedError(err_str)
                },
                jpegli_decode: false,
                zune_jpeg: false,
                file_size: 0,
            }
        }
    }
}

/// Main matrix test
#[test]
fn test_encoder_matrix() {
    let width = 128;
    let height = 128;
    let quality = 85.0;

    println!("\n{}", "=".repeat(90));
    println!("ENCODER CONFIGURATION MATRIX TEST");
    println!("{}\n", "=".repeat(90));

    let mut results: Vec<MatrixResult> = Vec::new();
    let mut pass_count = 0;
    let mut fail_count = 0;

    // Define test matrix
    let _pixel_formats = [PixelFormat::Rgb, PixelFormat::Gray];
    let modes = [JpegMode::Baseline, JpegMode::Progressive];
    let subsamplings = [
        Subsampling::S444,
        Subsampling::S422,
        Subsampling::S420,
        Subsampling::S440,
    ];
    let huffman_opts = [false, true]; // false = fixed, true = optimized
    let xyb_opts = [false, true];

    // Test RGB combinations
    for &mode in &modes {
        for &subsampling in &subsamplings {
            for &optimize_huffman in &huffman_opts {
                // Skip progressive + fixed Huffman (invalid combination, not supported)
                if mode == JpegMode::Progressive && !optimize_huffman {
                    continue;
                }

                for &use_xyb in &xyb_opts {
                    // Skip XYB with subsampling (XYB should use 4:4:4)
                    if use_xyb && subsampling != Subsampling::S444 {
                        continue;
                    }

                    let result = test_config(
                        width,
                        height,
                        PixelFormat::Rgb,
                        mode,
                        subsampling,
                        optimize_huffman,
                        use_xyb,
                        quality,
                    );
                    results.push(result);
                }
            }
        }
    }

    // Test Grayscale combinations (no subsampling, no XYB)
    for &mode in &modes {
        for &optimize_huffman in &huffman_opts {
            // Skip progressive + fixed Huffman (invalid combination, not supported)
            if mode == JpegMode::Progressive && !optimize_huffman {
                continue;
            }

            let result = test_config(
                width,
                height,
                PixelFormat::Gray,
                mode,
                Subsampling::S444, // Grayscale ignores subsampling
                optimize_huffman,
                false, // No XYB for grayscale
                quality,
            );
            results.push(result);
        }
    }

    // Print results table
    println!(
        "{:<55} {:>6} {:>8} {:>8} {:>8}",
        "Configuration", "Status", "jpegli", "zune", "Size"
    );
    println!("{}", "-".repeat(90));

    for result in &results {
        let status = match &result.encode_result {
            EncodeOutcome::Success => {
                if result.jpegli_decode && result.zune_jpeg {
                    pass_count += 1;
                    "OK"
                } else {
                    fail_count += 1;
                    "DECODE?"
                }
            }
            EncodeOutcome::ExpectedError(_) | EncodeOutcome::UnexpectedError(_) => {
                fail_count += 1;
                "ERR!"
            }
        };

        let decode_str = |ok: bool| if ok { "✓" } else { "✗" };
        let size_str = if result.file_size > 0 {
            format!("{}", result.file_size)
        } else {
            "-".to_string()
        };

        println!(
            "{:<55} {:>6} {:>8} {:>8} {:>8}",
            result.config,
            status,
            decode_str(result.jpegli_decode),
            decode_str(result.zune_jpeg),
            size_str
        );

        // Print error details for unexpected failures
        if let EncodeOutcome::UnexpectedError(ref e) = result.encode_result {
            println!("  └─ ERROR: {}", e);
        }
    }

    println!("{}", "-".repeat(90));
    println!("\nSummary: {} passed, {} failures", pass_count, fail_count);

    assert_eq!(fail_count, 0, "There were {} failures!", fail_count);
}

/// Quick smoke test for the most common configurations
#[test]
fn test_common_configurations() {
    let width = 64;
    let height = 64;
    let data = generate_test_image(width, height, 3);

    let configs: Vec<(&str, JpegMode, Subsampling, bool, bool)> = vec![
        // (name, mode, subsampling, optimize_huffman, use_xyb)
        (
            "baseline_444_fixed",
            JpegMode::Baseline,
            Subsampling::S444,
            false,
            false,
        ),
        (
            "baseline_444_opt",
            JpegMode::Baseline,
            Subsampling::S444,
            true,
            false,
        ),
        (
            "baseline_420_opt",
            JpegMode::Baseline,
            Subsampling::S420,
            true,
            false,
        ),
        (
            "progressive_444_opt",
            JpegMode::Progressive,
            Subsampling::S444,
            true,
            false,
        ),
        (
            "progressive_420_opt",
            JpegMode::Progressive,
            Subsampling::S420,
            true,
            false,
        ),
        (
            "xyb_baseline_opt",
            JpegMode::Baseline,
            Subsampling::S444,
            true,
            true,
        ),
        (
            "xyb_progressive_opt",
            JpegMode::Progressive,
            Subsampling::S444,
            true,
            true,
        ),
    ];

    println!("\n=== Common Configuration Smoke Test ===\n");

    for (name, mode, subsampling, optimize_huffman, use_xyb) in configs {
        let result = StreamingEncoder::new(width as u32, height as u32)
            .pixel_format(PixelFormat::Rgb)
            .mode(mode)
            .subsampling(subsampling)
            .optimize_huffman(optimize_huffman)
            .use_xyb(use_xyb)
            .quality(Quality::from_quality(85.0))
            .encode_all(&data);

        match result {
            Ok(jpeg) => {
                let jpegli_ok = decode_jpegli(&jpeg);
                let zune_ok = decode_zune_jpeg(&jpeg);

                let all_ok = jpegli_ok && zune_ok;
                println!(
                    "{:<25}: {} bytes, decoders: {}",
                    name,
                    jpeg.len(),
                    if all_ok { "ALL OK" } else { "SOME FAILED" }
                );

                assert!(all_ok, "{} failed decoder compatibility", name);
            }
            Err(e) => {
                panic!("{} encoding failed: {}", name, e);
            }
        }
    }
}

/// Test that progressive + fixed Huffman produces expected error
#[test]
fn test_progressive_fixed_huffman_errors() {
    let data = vec![128u8; 64 * 64 * 3];

    let result = StreamingEncoder::new(64, 64)
        .pixel_format(PixelFormat::Rgb)
        .mode(JpegMode::Progressive)
        .optimize_huffman(false) // Fixed Huffman
        .encode_all(&data);

    match result {
        Err(e) => {
            let err_str = format!("{}", e);
            assert!(
                err_str.contains("Progressive mode with fixed Huffman"),
                "Expected error about Progressive + fixed Huffman, got: {}",
                err_str
            );
            println!(
                "✓ Progressive + Fixed Huffman correctly errors: {}",
                err_str
            );
        }
        Ok(_) => {
            panic!("Expected error for Progressive + Fixed Huffman, but encoding succeeded!");
        }
    }
}
