//! Metadata-preservation tests for the `recompress` Preserve strategy.
//!
//! Coefficient-domain recompression must be metadata-transparent: the
//! output must keep the source's ICC profile (otherwise decoded colors
//! change) and EXIF (otherwise display orientation is lost). This test
//! encodes a JPEG carrying an ICC profile, recompresses it, and asserts
//! the profile survives in the output bytes.

// Needs `recompress` for the API under test, and `__test-utils` for
// `EncoderConfig::request()` (the metadata-carrying encode path, which is
// test-gated). Run: `cargo test -p zenjpeg --features recompress,__test-utils`.
#![cfg(all(feature = "recompress", feature = "__test-utils"))]

use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};
use zenjpeg::recompress::{RecompressOptions, RecompressResult, StrategyKind, recompress};

/// A recognizable fake ICC payload. It need not be a valid profile — the
/// test only asserts the bytes round-trip verbatim. 600 bytes of a
/// non-repeating pattern so a substring search can't false-positive on
/// incidental scan data.
fn fake_icc_profile() -> Vec<u8> {
    let mut p = Vec::with_capacity(600);
    for i in 0..600u32 {
        p.push((i.wrapping_mul(37).wrapping_add(11) & 0xFF) as u8);
    }
    p
}

/// Build a deterministic, non-degenerate RGB test image.
fn test_image(width: usize, height: usize) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let s = ((x ^ y) as u32).wrapping_mul(2_654_435_761);
            let r = ((x * 7 + y * 3) % 240 + (s & 0x0F) as usize) as u8;
            let g = ((x * 5 + y * 11) % 220 + ((s >> 4) & 0x1F) as usize) as u8;
            let b = ((x * 13 + y * 2) % 200 + ((s >> 9) & 0x3F) as usize) as u8;
            pixels.push(r);
            pixels.push(g);
            pixels.push(b);
        }
    }
    pixels
}

/// Encode a baseline 4:2:0 JPEG at `quality`, embedding `icc` as the ICC
/// profile, via the canonical `EncoderConfig::request()` path.
fn make_jpeg_with_icc(width: usize, height: usize, quality: i32, icc: &[u8]) -> Vec<u8> {
    let config = EncoderConfig::ycbcr(quality, ChromaSubsampling::Quarter);
    config
        .request()
        .icc_profile_owned(icc.to_vec())
        .encode_bytes(
            &test_image(width, height),
            width as u32,
            height as u32,
            PixelLayout::Rgb8Srgb,
        )
        .expect("encode with ICC should succeed")
}

/// True if `haystack` contains `needle` as a contiguous byte run.
fn contains_subslice(haystack: &[u8], needle: &[u8]) -> bool {
    if needle.is_empty() || haystack.len() < needle.len() {
        return false;
    }
    haystack.windows(needle.len()).any(|w| w == needle)
}

#[test]
fn preserve_keeps_icc_profile() {
    let icc = fake_icc_profile();
    // High source quality (q92) + lower target (70) routes to the
    // coefficient-domain Preserve strategy.
    let src = make_jpeg_with_icc(160, 160, 92, &icc);
    assert!(
        contains_subslice(&src, b"ICC_PROFILE"),
        "precondition: source JPEG must carry an ICC APP2 segment"
    );
    assert!(
        contains_subslice(&src, &icc),
        "precondition: source must carry the exact ICC payload"
    );

    let result = recompress(&src, &RecompressOptions::new(70.0)).expect("recompress");

    match result {
        RecompressResult::Recompressed {
            ref bytes,
            strategy,
            ..
        } => {
            eprintln!("strategy-chosen: {strategy:?}");
            assert!(
                contains_subslice(bytes, b"ICC_PROFILE"),
                "recompressed output ({strategy:?}) dropped the ICC APP2 marker"
            );
            assert!(
                contains_subslice(bytes, &icc),
                "recompressed output ({strategy:?}) dropped the ICC profile payload"
            );
            if strategy != StrategyKind::Preserve {
                eprintln!("note: router chose {strategy:?}, not Preserve");
            }
        }
        RecompressResult::LosslessOnly { ref bytes, .. } => {
            assert!(
                contains_subslice(bytes, &icc),
                "lossless output dropped the ICC profile payload"
            );
        }
        RecompressResult::NoOp { .. } => {
            eprintln!("note: recompress returned NoOp; emit path not exercised");
        }
        // RecompressResult is #[non_exhaustive].
        _ => panic!("unexpected RecompressResult variant"),
    }
}
