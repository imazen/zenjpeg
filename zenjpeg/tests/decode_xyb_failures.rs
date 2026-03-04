//! Regression tests for zenjpeg decoder failures on XYB-encoded JPEGs.
//!
//! These test files were produced by zenjpeg's own encoder in XYB/4:2:0 mode
//! and fail to decode. Found during synthetic training data generation
//! (2026-02-26) where zenjpeg-420-xyb-e2 at specific quality levels produces
//! JPEGs that the decoder rejects.
//!
//! Two error categories:
//! - "invalid Huffman table 0: invalid code" (512x512, q15/q50)
//! - "expected 0xFF for restart marker" (1024x1024, q15/q20/q60)

use zenjpeg::decoder::{Decoder, PixelFormat};

fn try_decode(data: &[u8]) -> Result<(), String> {
    Decoder::new()
        .output_format(PixelFormat::Rgb)
        .decode(data, enough::Unstoppable)
        .map(|_| ())
        .map_err(|e| e.to_string())
}

#[test]
#[ignore = "known bug: XYB Huffman decode failure at 512x512"]
fn xyb_huffman_512_q50() {
    let data = include_bytes!("testdata/decode_failures/xyb_huffman_512_q50.jpg");
    let result = try_decode(data);
    // Currently fails with "invalid Huffman table 0: invalid code"
    // When fixed, remove #[ignore] and assert Ok
    assert!(result.is_ok(), "decode failed: {}", result.unwrap_err());
}

#[test]
#[ignore = "known bug: XYB Huffman decode failure at 512x512"]
fn xyb_huffman_512_q15() {
    let data = include_bytes!("testdata/decode_failures/xyb_huffman_512_q15.jpg");
    let result = try_decode(data);
    assert!(result.is_ok(), "decode failed: {}", result.unwrap_err());
}

#[test]
#[ignore = "known bug: XYB restart marker decode failure at 1024x1024"]
fn xyb_rst_1024_q60() {
    let data = include_bytes!("testdata/decode_failures/xyb_rst_1024_q60.jpg");
    let result = try_decode(data);
    // Currently fails with "expected 0xFF for restart marker"
    assert!(result.is_ok(), "decode failed: {}", result.unwrap_err());
}

#[test]
#[ignore = "known bug: XYB restart marker decode failure at 1024x1024"]
fn xyb_rst_1024_q15() {
    let data = include_bytes!("testdata/decode_failures/xyb_rst_1024_q15.jpg");
    let result = try_decode(data);
    assert!(result.is_ok(), "decode failed: {}", result.unwrap_err());
}

#[test]
#[ignore = "known bug: XYB restart marker decode failure at 1024x1024"]
fn xyb_rst_1024_q20() {
    let data = include_bytes!("testdata/decode_failures/xyb_rst_1024_q20.jpg");
    let result = try_decode(data);
    assert!(result.is_ok(), "decode failed: {}", result.unwrap_err());
}
