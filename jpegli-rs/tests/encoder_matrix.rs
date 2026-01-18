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

use jpegli::{
    decoder::Decoder,
    encoder::{ChromaSubsampling, EncoderConfig, PixelLayout},
};

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

fn encode_rgb(
    width: u32,
    height: u32,
    data: &[u8],
    config: &EncoderConfig,
) -> jpegli::encoder::Result<Vec<u8>> {
    let mut enc = config.encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)?;
    enc.push_packed(data, enough::Unstoppable)?;
    enc.finish()
}

fn encode_gray(
    width: u32,
    height: u32,
    data: &[u8],
    config: &EncoderConfig,
) -> jpegli::encoder::Result<Vec<u8>> {
    let mut enc = config.encode_from_bytes(width, height, PixelLayout::Gray8Srgb)?;
    enc.push_packed(data, enough::Unstoppable)?;
    enc.finish()
}

/// Decode with jpegli-rs
fn decode_jpegli(data: &[u8]) -> bool {
    Decoder::new().decode(data).is_ok()
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
    is_grayscale: bool,
    progressive: bool,
    subsampling: ChromaSubsampling,
    optimize_huffman: bool,
    use_xyb: bool,
    quality: f32,
) -> MatrixResult {
    let channels = if is_grayscale { 1 } else { 3 };
    let data = generate_test_image(width as usize, height as usize, channels);

    let pixel_format_name = if is_grayscale { "Gray" } else { "Rgb" };
    let mode_name = if progressive {
        "Progressive"
    } else {
        "Baseline"
    };
    let sub_name = match subsampling {
        ChromaSubsampling::None => "444",
        ChromaSubsampling::HalfHorizontal => "422",
        ChromaSubsampling::Quarter => "420",
        ChromaSubsampling::HalfVertical => "440",
        _ => "unknown",
    };

    let config_name = format!(
        "{}/{}/{}/huff={}/xyb={}/Q{}",
        pixel_format_name,
        mode_name,
        sub_name,
        if optimize_huffman { "opt" } else { "fix" },
        use_xyb,
        quality as u32
    );

    let mut config = EncoderConfig::new(90.0, ChromaSubsampling::Quarter)
        .quality(quality)
        .progressive(progressive)
        .optimize_huffman(optimize_huffman);

    if is_grayscale {
        config = config.grayscale();
    } else if use_xyb {
        config = config.xyb();
    } else {
        config = config.ycbcr(subsampling);
    }

    let encode_result = if is_grayscale {
        encode_gray(width, height, &data, &config)
    } else {
        encode_rgb(width, height, &data, &config)
    };

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
    let progressives = [false, true];
    let subsamplings = [
        ChromaSubsampling::None,
        ChromaSubsampling::HalfHorizontal,
        ChromaSubsampling::Quarter,
        ChromaSubsampling::HalfVertical,
    ];
    let huffman_opts = [false, true]; // false = fixed, true = optimized
    let xyb_opts = [false, true];

    // Test RGB combinations
    for &progressive in &progressives {
        for &subsampling in &subsamplings {
            for &optimize_huffman in &huffman_opts {
                // Skip progressive + fixed Huffman (invalid combination, not supported)
                if progressive && !optimize_huffman {
                    continue;
                }

                for &use_xyb in &xyb_opts {
                    // Skip XYB with subsampling (XYB should use 4:4:4)
                    if use_xyb && subsampling != ChromaSubsampling::None {
                        continue;
                    }

                    let result = test_config(
                        width,
                        height,
                        false, // not grayscale
                        progressive,
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
    for &progressive in &progressives {
        for &optimize_huffman in &huffman_opts {
            // Skip progressive + fixed Huffman (invalid combination, not supported)
            if progressive && !optimize_huffman {
                continue;
            }

            let result = test_config(
                width,
                height,
                true, // grayscale
                progressive,
                ChromaSubsampling::None, // Grayscale ignores subsampling
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

    let configs: Vec<(&str, bool, ChromaSubsampling, bool, bool)> = vec![
        // (name, progressive, subsampling, optimize_huffman, use_xyb)
        (
            "baseline_444_fixed",
            false,
            ChromaSubsampling::None,
            false,
            false,
        ),
        (
            "baseline_444_opt",
            false,
            ChromaSubsampling::None,
            true,
            false,
        ),
        (
            "baseline_420_opt",
            false,
            ChromaSubsampling::Quarter,
            true,
            false,
        ),
        (
            "progressive_444_opt",
            true,
            ChromaSubsampling::None,
            true,
            false,
        ),
        (
            "progressive_420_opt",
            true,
            ChromaSubsampling::Quarter,
            true,
            false,
        ),
        (
            "xyb_baseline_opt",
            false,
            ChromaSubsampling::None,
            true,
            true,
        ),
        (
            "xyb_progressive_opt",
            true,
            ChromaSubsampling::None,
            true,
            true,
        ),
    ];

    println!("\n=== Common Configuration Smoke Test ===\n");

    for (name, progressive, subsampling, optimize_huffman, use_xyb) in configs {
        let mut config = EncoderConfig::new(90.0, ChromaSubsampling::Quarter)
            .quality(85.0)
            .progressive(progressive)
            .optimize_huffman(optimize_huffman);

        if use_xyb {
            config = config.xyb();
        } else {
            config = config.ycbcr(subsampling);
        }

        let result = encode_rgb(width as u32, height as u32, &data, &config);

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

    let config = EncoderConfig::new(90.0, ChromaSubsampling::Quarter)
        .quality(85.0)
        .progressive(true)
        .optimize_huffman(false); // Fixed Huffman

    let result = encode_rgb(64, 64, &data, &config);

    match result {
        Err(e) => {
            let err_str = format!("{}", e);
            assert!(
                err_str.contains("progressive mode requires optimized Huffman"),
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
