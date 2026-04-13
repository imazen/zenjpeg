//! Regression tests for XYB-encoded JPEGs.
//!
//! Two categories of test files:
//!
//! **Permanently corrupted (pre-fix artifacts):**
//! The 512x512 q15/q50 files were encoded BEFORE commit b0cafce fixed
//! `collect_block_frequencies_simd()` — the frequency counter clamped DC
//! categories to 11, but XYB produces categories 12+ at low quality. The
//! Huffman table lacked codes for those categories, producing corrupted
//! bitstreams. These files can never decode and serve as regression tests
//! to verify the decoder rejects them gracefully.
//!
//! **Fixed (post-fix verification):**
//! The 1024x1024 RST files were re-encoded after the fix and decode correctly.
//! They verify that XYB encoding at low quality with restart markers works.

use zenjpeg::decoder::{Decoder, PixelFormat};

fn try_decode(data: &[u8]) -> Result<(), String> {
    Decoder::new()
        .output_format(PixelFormat::Rgb)
        .decode(data, enough::Unstoppable)
        .map(|_| ())
        .map_err(|e| e.to_string())
}

/// Pre-fix corrupted file: frequency counter clamped DC categories to 11,
/// but XYB Q50 needs category 12+. Huffman table has no codes for them.
/// Decoder must reject gracefully (not panic).
#[test]
fn xyb_huffman_512_q50_corrupted_rejects() {
    let data = include_bytes!("testdata/decode_failures/xyb_huffman_512_q50.jpg");
    let result = try_decode(data);
    assert!(
        result.is_err(),
        "pre-fix corrupted XYB file should fail to decode"
    );
    let err = result.unwrap_err();
    assert!(
        err.contains("Huffman") || err.contains("invalid"),
        "error should mention Huffman, got: {err}"
    );
}

/// Pre-fix corrupted file: same root cause as Q50 but at Q15.
#[test]
fn xyb_huffman_512_q15_corrupted_rejects() {
    let data = include_bytes!("testdata/decode_failures/xyb_huffman_512_q15.jpg");
    let result = try_decode(data);
    assert!(
        result.is_err(),
        "pre-fix corrupted XYB file should fail to decode"
    );
    let err = result.unwrap_err();
    assert!(
        err.contains("Huffman") || err.contains("invalid"),
        "error should mention Huffman, got: {err}"
    );
}

/// Post-fix XYB 1024x1024 with restart markers: should decode correctly.
#[test]
fn xyb_rst_1024_q60() {
    let data = include_bytes!("testdata/decode_failures/xyb_rst_1024_q60.jpg");
    let result = try_decode(data);
    assert!(result.is_ok(), "decode failed: {}", result.unwrap_err());
}

#[test]
fn xyb_rst_1024_q15() {
    let data = include_bytes!("testdata/decode_failures/xyb_rst_1024_q15.jpg");
    let result = try_decode(data);
    assert!(result.is_ok(), "decode failed: {}", result.unwrap_err());
}

#[test]
fn xyb_rst_1024_q20() {
    let data = include_bytes!("testdata/decode_failures/xyb_rst_1024_q20.jpg");
    let result = try_decode(data);
    assert!(result.is_ok(), "decode failed: {}", result.unwrap_err());
}
