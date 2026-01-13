//! Test that progressive mode requires Huffman optimization

use jpegli::{Error, JpegEncoder, PixelFormat};

#[test]
fn test_progressive_requires_huffman_optimization() {
    let data = vec![128u8; 64 * 64 * 3];

    // Progressive + Fixed Huffman should error
    let result = JpegEncoder::new(64, 64)
        .pixel_format(PixelFormat::Rgb)
        .quality(jpegli::quant::Quality::from_quality(90.0))
        .mode(jpegli::types::JpegMode::Progressive)
        .optimize_huffman(false) // Fixed Huffman
        .encode(&data);

    match result {
        Err(Error::UnsupportedFeature { feature }) => {
            assert!(
                feature.contains("Progressive mode with fixed Huffman"),
                "Expected error message about Progressive + fixed Huffman, got: {}",
                feature
            );
        }
        Ok(_) => panic!("Expected error, but encoding succeeded!"),
        Err(e) => panic!("Expected UnsupportedFeature error, got: {:?}", e),
    }
}

#[test]
fn test_progressive_xyb_requires_huffman_optimization() {
    let data = vec![128u8; 64 * 64 * 3];

    // XYB Progressive + Fixed Huffman should also error
    let result = JpegEncoder::new(64, 64)
        .pixel_format(PixelFormat::Rgb)
        .quality(jpegli::quant::Quality::from_quality(90.0))
        .mode(jpegli::types::JpegMode::Progressive)
        .use_xyb(true)
        .optimize_huffman(false) // Fixed Huffman
        .encode(&data);

    match result {
        Err(Error::UnsupportedFeature { feature }) => {
            assert!(
                feature.contains("Progressive mode with fixed Huffman"),
                "Expected error message about Progressive + fixed Huffman, got: {}",
                feature
            );
        }
        Ok(_) => panic!("Expected error, but encoding succeeded!"),
        Err(e) => panic!("Expected UnsupportedFeature error, got: {:?}", e),
    }
}

#[test]
fn test_baseline_with_fixed_huffman_works() {
    let data = vec![128u8; 64 * 64 * 3];

    // Baseline + Fixed Huffman should work fine
    let result = JpegEncoder::new(64, 64)
        .pixel_format(PixelFormat::Rgb)
        .quality(jpegli::quant::Quality::from_quality(90.0))
        .mode(jpegli::types::JpegMode::Baseline)
        .optimize_huffman(false) // Fixed Huffman is OK for baseline
        .encode(&data);

    assert!(result.is_ok(), "Baseline + Fixed Huffman should work");
}

#[test]
fn test_progressive_with_optimized_huffman_works() {
    let data = vec![128u8; 64 * 64 * 3];

    // Progressive + Optimized Huffman should work
    let result = JpegEncoder::new(64, 64)
        .pixel_format(PixelFormat::Rgb)
        .quality(jpegli::quant::Quality::from_quality(90.0))
        .mode(jpegli::types::JpegMode::Progressive)
        .optimize_huffman(true) // Optimized Huffman
        .encode(&data);

    assert!(
        result.is_ok(),
        "Progressive + Optimized Huffman should work"
    );
}
