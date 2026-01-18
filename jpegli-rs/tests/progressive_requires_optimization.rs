//! Test that progressive mode requires Huffman optimization

use jpegli::encoder::ChromaSubsampling;
use jpegli::encoder::{EncoderConfig, PixelLayout};

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

#[test]
fn test_progressive_requires_huffman_optimization() {
    let data = vec![128u8; 64 * 64 * 3];

    // Progressive + Fixed Huffman should error
    let config = EncoderConfig::new(90.0, ChromaSubsampling::Quarter)
        .quality(90.0)
        .progressive(true)
        .optimize_huffman(false); // Fixed Huffman

    let result = encode_rgb(64, 64, &data, &config);

    match result {
        Err(e) => {
            let err_str = format!("{}", e);
            assert!(
                err_str.contains("progressive mode requires optimized Huffman"),
                "Expected error message about Progressive + fixed Huffman, got: {}",
                err_str
            );
        }
        Ok(_) => panic!("Expected error, but encoding succeeded!"),
    }
}

#[test]
fn test_progressive_xyb_requires_huffman_optimization() {
    let data = vec![128u8; 64 * 64 * 3];

    // XYB Progressive + Fixed Huffman should also error
    let config = EncoderConfig::new(90.0, ChromaSubsampling::Quarter)
        .quality(90.0)
        .progressive(true)
        .xyb()
        .optimize_huffman(false); // Fixed Huffman

    let result = encode_rgb(64, 64, &data, &config);

    match result {
        Err(e) => {
            let err_str = format!("{}", e);
            assert!(
                err_str.contains("progressive mode requires optimized Huffman"),
                "Expected error message about Progressive + fixed Huffman, got: {}",
                err_str
            );
        }
        Ok(_) => panic!("Expected error, but encoding succeeded!"),
    }
}

#[test]
fn test_baseline_with_fixed_huffman_works() {
    let data = vec![128u8; 64 * 64 * 3];

    // Baseline + Fixed Huffman should work fine
    let config = EncoderConfig::new(90.0, ChromaSubsampling::Quarter)
        .quality(90.0)
        .progressive(false)
        .optimize_huffman(false); // Fixed Huffman is OK for baseline

    let result = encode_rgb(64, 64, &data, &config);

    assert!(result.is_ok(), "Baseline + Fixed Huffman should work");
}

#[test]
fn test_progressive_with_optimized_huffman_works() {
    let data = vec![128u8; 64 * 64 * 3];

    // Progressive + Optimized Huffman should work
    let config = EncoderConfig::new(90.0, ChromaSubsampling::Quarter)
        .quality(90.0)
        .progressive(true)
        .optimize_huffman(true); // Optimized Huffman

    let result = encode_rgb(64, 64, &data, &config);

    assert!(
        result.is_ok(),
        "Progressive + Optimized Huffman should work"
    );
}
