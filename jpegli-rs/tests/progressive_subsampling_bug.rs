//! Test for progressive + subsampling bug.
//!
//! Root cause: encode_progressive_optimized() ignores config.subsampling and
//! always produces 4:4:4 data, but write_frame_header() correctly writes the
//! subsampling factors from config. This causes a mismatch where the header
//! declares e.g. 4:2:0 (Y=2hx2v) but the scan data contains 4:4:4 block counts.
//!
//! Symptom: "Corrupt JPEG data: N extraneous bytes before marker" from djpeg,
//! and all subsampled progressive modes produce the same file size as 4:4:4.

use jpegli::decode::Decoder;
use jpegli::types::{JpegMode, PixelFormat, Subsampling};

use jpegli::{JpegEncoder, Quality};

/// Test that progressive + subsampling produces files that can be decoded
/// by multiple external decoders without corruption errors.
#[test]
fn test_progressive_subsampling_external_decoder_compat() {
    let width = 64u32;
    let height = 64u32;

    // Simple gradient test image
    let mut rgb = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            rgb.push((x * 4) as u8);
            rgb.push((y * 4) as u8);
            rgb.push(128u8);
        }
    }

    let configs = [
        (Subsampling::S444, "S444"),
        (Subsampling::S422, "S422"),
        (Subsampling::S420, "S420"),
        (Subsampling::S440, "S440"),
    ];

    for (subsampling, name) in configs {
        let jpeg = JpegEncoder::new(width, height)
            .pixel_format(PixelFormat::Rgb)
            .mode(JpegMode::Progressive)
            .subsampling(subsampling)
            .optimize_huffman(true)
            .quality(Quality::from_quality(85.0))
            .encode(&rgb)
            .unwrap_or_else(|e| panic!("Progressive {} encode failed: {:?}", name, e));

        // Test with zune-jpeg decoder
        use zune_jpeg::zune_core::bytestream::ZCursor;
        use zune_jpeg::JpegDecoder;
        let zune_result = JpegDecoder::new(ZCursor::new(&jpeg)).decode();

        assert!(
            zune_result.is_ok(),
            "Progressive {} failed zune-jpeg decode: {:?}",
            name,
            zune_result.err()
        );

        // Test with our own decoder
        let our_result = Decoder::new().output_format(PixelFormat::Rgb).decode(&jpeg);

        assert!(
            our_result.is_ok(),
            "Progressive {} failed jpegli-rs decode: {:?}",
            name,
            our_result.err()
        );
    }
}

/// Test that progressive subsampling produces SMALLER files than 4:4:4.
/// If all modes produce the same size, the encoder is ignoring subsampling.
#[test]
fn test_progressive_subsampling_file_sizes() {
    let width = 128u32;
    let height = 128u32;

    // Create a test image with varied content
    let mut rgb = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            rgb.push(((x * 2 + y) % 256) as u8);
            rgb.push(((y * 2 + x) % 256) as u8);
            rgb.push(((x + y) % 256) as u8);
        }
    }

    let encode = |sub: Subsampling| -> usize {
        JpegEncoder::new(width, height)
            .pixel_format(PixelFormat::Rgb)
            .mode(JpegMode::Progressive)
            .subsampling(sub)
            .optimize_huffman(true)
            .quality(Quality::from_quality(85.0))
            .encode(&rgb)
            .expect("encode failed")
            .len()
    };

    let size_444 = encode(Subsampling::S444);
    let size_422 = encode(Subsampling::S422);
    let size_420 = encode(Subsampling::S420);
    let size_440 = encode(Subsampling::S440);

    eprintln!("Progressive file sizes:");
    eprintln!("  S444: {} bytes", size_444);
    eprintln!("  S422: {} bytes", size_422);
    eprintln!("  S420: {} bytes", size_420);
    eprintln!("  S440: {} bytes", size_440);

    // Subsampled modes should be SMALLER than 4:4:4
    // (they have fewer chroma blocks to encode)
    assert!(
        size_422 < size_444,
        "Progressive S422 ({}) should be smaller than S444 ({})",
        size_422,
        size_444
    );

    assert!(
        size_420 < size_444,
        "Progressive S420 ({}) should be smaller than S444 ({})",
        size_420,
        size_444
    );

    assert!(
        size_440 < size_444,
        "Progressive S440 ({}) should be smaller than S444 ({})",
        size_440,
        size_444
    );

    // S420 should be smallest (most subsampling)
    assert!(
        size_420 < size_422,
        "Progressive S420 ({}) should be smaller than S422 ({})",
        size_420,
        size_422
    );

    assert!(
        size_420 < size_440,
        "Progressive S420 ({}) should be smaller than S440 ({})",
        size_420,
        size_440
    );
}

/// Test that baseline subsampling works correctly (control test).
/// If this passes but progressive fails, the bug is in progressive encoder.
#[test]
fn test_baseline_subsampling_works() {
    let width = 128u32;
    let height = 128u32;

    let mut rgb = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            rgb.push(((x * 2 + y) % 256) as u8);
            rgb.push(((y * 2 + x) % 256) as u8);
            rgb.push(((x + y) % 256) as u8);
        }
    }

    let encode = |sub: Subsampling| -> usize {
        JpegEncoder::new(width, height)
            .pixel_format(PixelFormat::Rgb)
            .mode(JpegMode::Baseline)
            .subsampling(sub)
            .optimize_huffman(true)
            .quality(Quality::from_quality(85.0))
            .encode(&rgb)
            .expect("encode failed")
            .len()
    };

    let size_444 = encode(Subsampling::S444);
    let size_420 = encode(Subsampling::S420);

    eprintln!("Baseline file sizes: S444={}, S420={}", size_444, size_420);

    // Baseline S420 should definitely be smaller
    assert!(
        size_420 < size_444,
        "Baseline S420 ({}) should be smaller than S444 ({})",
        size_420,
        size_444
    );
}
