//! Regression tests for S440 + Progressive bug found by fuzzer.
//!
//! Bug: Encoding with non-S444 subsampling + Progressive mode used to fail
//! with "AC symbol not in Huffman table during replay".
//!
//! Root cause: Context assignment for AC scans used scan_idx instead of
//! component_index. This caused misaligned Huffman tables when DC scans
//! were non-interleaved (i.e., S422, S420, S440 modes).
//!
//! Fix: Use component_index for AC context assignment to ensure Y always
//! uses luma table and Cb/Cr use chroma table, regardless of scan order.

use enough::Never;
use jpegli::decode::Decoder;
use jpegli::types::PixelFormat;
use jpegli::{ChromaSubsampling, EncoderConfig, PixelLayout};

/// Regression test for the exact case found by the fuzzer.
#[test]
fn test_s440_progressive_roundtrip() {
    let width = 49u32;
    let height = 255u32;

    // Generate test pixels
    let pixels: Vec<u8> = (0..(width * height * 3)).map(|i| (i % 256) as u8).collect();

    let config = EncoderConfig::new()
        .quality(90.0)
        .ycbcr(ChromaSubsampling::HalfVertical) // S440
        .progressive(true);
    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .expect("encoder setup");
    enc.push_packed(&pixels, Never).expect("push");
    let encoded = enc.finish().expect("encode should succeed");
    eprintln!(
        "Encoded {} bytes for {}x{} S440 Progressive",
        encoded.len(),
        width,
        height
    );

    let decoder = Decoder::new().output_format(PixelFormat::Rgb);
    let decoded = decoder.decode(&encoded).expect("decode should succeed");

    assert_eq!(decoded.width, width);
    assert_eq!(decoded.height, height);
}

/// Test all subsampling modes with progressive encoding
#[test]
fn test_all_subsampling_progressive() {
    let test_cases = [
        (ChromaSubsampling::Full, "S444"),
        (ChromaSubsampling::HalfHorizontal, "S422"),
        (ChromaSubsampling::Quarter, "S420"),
        (ChromaSubsampling::HalfVertical, "S440"),
    ];

    for (subsampling, name) in test_cases {
        let width = 64u32;
        let height = 64u32;

        let pixels: Vec<u8> = (0..(width * height * 3)).map(|i| (i % 256) as u8).collect();

        let config = EncoderConfig::new()
            .quality(90.0)
            .ycbcr(subsampling)
            .progressive(true);
        let mut enc = config
            .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
            .expect("encoder setup");
        enc.push_packed(&pixels, Never).expect("push");

        let encoded = match enc.finish() {
            Ok(data) => data,
            Err(e) => panic!("{} encode failed: {:?}", name, e),
        };

        let decoder = Decoder::new().output_format(PixelFormat::Rgb);
        match decoder.decode(&encoded) {
            Ok(img) => {
                assert_eq!(img.width, width, "{} width mismatch", name);
                assert_eq!(img.height, height, "{} height mismatch", name);
                eprintln!("{} Progressive: OK ({} bytes)", name, encoded.len());
            }
            Err(e) => panic!("{} decode failed: {:?}", name, e),
        }
    }
}

/// Test various image sizes with non-S444 subsampling + progressive
/// Note: Uses MCU-aligned dimensions to avoid unrelated edge cases with odd sizes
#[test]
fn test_progressive_subsampling_various_sizes() {
    // Use MCU-aligned sizes (multiples of 16) to focus on the Huffman table bug
    // Odd dimensions have separate edge case issues unrelated to this fix
    let sizes = [(16, 16), (32, 32), (64, 64), (128, 96), (256, 256)];

    let subsamplings = [
        (ChromaSubsampling::HalfHorizontal, "S422"),
        (ChromaSubsampling::Quarter, "S420"),
        (ChromaSubsampling::HalfVertical, "S440"),
    ];

    for (width, height) in sizes {
        for (subsampling, name) in &subsamplings {
            let pixels: Vec<u8> = (0..(width * height * 3))
                .map(|i| ((i * 7) % 256) as u8)
                .collect();

            let config = EncoderConfig::new()
                .quality(85.0)
                .ycbcr(*subsampling)
                .progressive(true);
            let mut enc = config
                .encode_from_bytes(width as u32, height as u32, PixelLayout::Rgb8Srgb)
                .expect("encoder setup");
            enc.push_packed(&pixels, Never).expect("push");

            let encoded = enc
                .finish()
                .unwrap_or_else(|e| panic!("{}x{} {} encode failed: {:?}", width, height, name, e));

            let decoder = Decoder::new().output_format(PixelFormat::Rgb);
            let decoded = decoder
                .decode(&encoded)
                .unwrap_or_else(|e| panic!("{}x{} {} decode failed: {:?}", width, height, name, e));

            assert_eq!(decoded.width as usize, width);
            assert_eq!(decoded.height as usize, height);
        }
    }
    eprintln!("All size × subsampling combinations passed");
}
