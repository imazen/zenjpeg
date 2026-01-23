//! UltraHDR roundtrip integration tests.
//!
//! Tests the full encode → decode → re-encode workflow.

#![cfg(feature = "ultrahdr")]

use jpegli::decoder::Decoder;
use jpegli::encoder::{ChromaSubsampling, EncoderConfig};
use jpegli::ultrahdr::{
    encode_ultrahdr, reconstruct_hdr, reencode_ultrahdr, tonemapper_from_ultrahdr, GainMapConfig,
    HdrOutputFormat, ToneMapConfig, UhdrColorGamut, UhdrColorTransfer, UhdrPixelFormat,
    UhdrRawImage, UltraHdrExtras, Unstoppable,
};

/// Create a simple HDR test image (linear RGB float).
fn create_test_hdr(width: u32, height: u32) -> UhdrRawImage {
    let mut data = Vec::with_capacity((width * height * 16) as usize);

    for y in 0..height {
        for x in 0..width {
            // Gradient with HDR values (up to 4.0)
            let r = (x as f32 / width as f32) * 4.0;
            let g = (y as f32 / height as f32) * 2.0;
            let b = 0.5f32;
            let a = 1.0f32;

            data.extend_from_slice(&r.to_le_bytes());
            data.extend_from_slice(&g.to_le_bytes());
            data.extend_from_slice(&b.to_le_bytes());
            data.extend_from_slice(&a.to_le_bytes());
        }
    }

    UhdrRawImage::from_data(
        width,
        height,
        UhdrPixelFormat::Rgba32F,
        UhdrColorGamut::Bt709,
        UhdrColorTransfer::Linear,
        data,
    )
    .expect("Failed to create test HDR image")
}

#[test]
fn test_encode_decode_roundtrip() {
    let width = 64;
    let height = 64;
    let hdr = create_test_hdr(width, height);

    // Encode HDR → UltraHDR JPEG
    let jpeg = encode_ultrahdr(
        &hdr,
        &GainMapConfig::default(),
        &ToneMapConfig::default(),
        &EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter),
        75.0,
        Unstoppable,
    )
    .expect("Encoding failed");

    // Verify it's a valid JPEG
    assert_eq!(&jpeg[0..2], &[0xFF, 0xD8], "Should start with SOI");
    assert_eq!(
        &jpeg[jpeg.len() - 2..],
        &[0xFF, 0xD9],
        "Primary image should end with EOI"
    );

    // Decode and check for UltraHDR
    let decoded = Decoder::new().decode(&jpeg).expect("Decoding failed");

    assert_eq!(decoded.width(), width);
    assert_eq!(decoded.height(), height);

    let extras = decoded.extras().expect("Should have extras");
    assert!(extras.is_ultrahdr(), "Should be detected as UltraHDR");

    // Parse metadata
    let (metadata, _) = extras
        .ultrahdr_metadata()
        .expect("Should have metadata")
        .expect("Metadata parsing should succeed");

    // Verify metadata has reasonable values (max_content_boost is [f32; 3] per channel)
    assert!(
        metadata.max_content_boost[0] > 1.0
            || metadata.max_content_boost[1] > 1.0
            || metadata.max_content_boost[2] > 1.0,
        "HDR should have max_content_boost > 1.0 in at least one channel"
    );

    // Decode gain map
    let gainmap = extras
        .decode_gainmap()
        .expect("Should have gain map")
        .expect("Gain map decode should succeed");

    assert!(gainmap.width > 0);
    assert!(gainmap.height > 0);
}

#[test]
fn test_hdr_reconstruction() {
    let width = 32;
    let height = 32;
    let original_hdr = create_test_hdr(width, height);

    // Encode
    let jpeg = encode_ultrahdr(
        &original_hdr,
        &GainMapConfig::default(),
        &ToneMapConfig::default(),
        &EncoderConfig::ycbcr(90.0, ChromaSubsampling::None),
        85.0,
        Unstoppable,
    )
    .expect("Encoding failed");

    // Decode
    let decoded = Decoder::new().decode(&jpeg).expect("Decoding failed");
    let extras = decoded.extras().expect("Should have extras");

    // Reconstruct HDR
    let reconstructed = reconstruct_hdr(
        decoded.pixels(),
        decoded.width(),
        decoded.height(),
        extras,
        4.0, // Standard HDR display boost
        HdrOutputFormat::LinearFloat,
        Unstoppable,
    )
    .expect("HDR reconstruction failed");

    assert_eq!(reconstructed.width, width);
    assert_eq!(reconstructed.height, height);
    assert_eq!(reconstructed.format, UhdrPixelFormat::Rgba32F);

    // Verify HDR values are in reasonable range
    // (exact match not expected due to lossy compression)
    let pixels = &reconstructed.data;
    for i in (0..pixels.len()).step_by(16) {
        let r = f32::from_le_bytes([pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]]);
        let g = f32::from_le_bytes([pixels[i + 4], pixels[i + 5], pixels[i + 6], pixels[i + 7]]);
        let b = f32::from_le_bytes([pixels[i + 8], pixels[i + 9], pixels[i + 10], pixels[i + 11]]);

        // HDR values should be non-negative
        assert!(r >= 0.0, "R should be non-negative: {}", r);
        assert!(g >= 0.0, "G should be non-negative: {}", g);
        assert!(b >= 0.0, "B should be non-negative: {}", b);
    }
}

#[test]
fn test_tonemapper_extraction() {
    let hdr = create_test_hdr(32, 32);

    // Encode
    let jpeg = encode_ultrahdr(
        &hdr,
        &GainMapConfig::default(),
        &ToneMapConfig::default(),
        &EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter),
        75.0,
        Unstoppable,
    )
    .expect("Encoding failed");

    // Decode and extract tonemapper
    let decoded = Decoder::new().decode(&jpeg).expect("Decoding failed");
    let extras = decoded.extras().expect("Should have extras");

    let tonemapper =
        tonemapper_from_ultrahdr(extras).expect("Tonemapper extraction should succeed");

    // Tonemapper should be usable
    let test_input = UhdrRawImage::from_data(
        2,
        2,
        UhdrPixelFormat::Rgba32F,
        UhdrColorGamut::Bt709,
        UhdrColorTransfer::Linear,
        vec![0u8; 64],
    )
    .unwrap();

    let _output = tonemapper
        .apply(&test_input)
        .expect("Tonemapper application should succeed");
}

#[test]
fn test_reencode_ultrahdr() {
    let hdr = create_test_hdr(48, 48);

    // Initial encode
    let original_jpeg = encode_ultrahdr(
        &hdr,
        &GainMapConfig::default(),
        &ToneMapConfig::default(),
        &EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter),
        75.0,
        Unstoppable,
    )
    .expect("Initial encoding failed");

    // Re-encode without modification
    let reencoded = reencode_ultrahdr(
        &original_jpeg,
        4.0,
        None::<fn(&mut UhdrRawImage)>,
        &GainMapConfig::default(),
        &EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter),
        75.0,
        Unstoppable,
    )
    .expect("Re-encoding failed");

    // Verify re-encoded is valid UltraHDR
    let decoded = Decoder::new()
        .decode(&reencoded)
        .expect("Decoding re-encoded failed");

    let extras = decoded.extras().expect("Should have extras");
    assert!(extras.is_ultrahdr(), "Re-encoded should still be UltraHDR");
}

#[test]
fn test_metadata_passthrough() {
    let hdr = create_test_hdr(32, 32);

    // Encode with custom EXIF-like segment
    let jpeg = encode_ultrahdr(
        &hdr,
        &GainMapConfig::default(),
        &ToneMapConfig::default(),
        &EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter),
        75.0,
        Unstoppable,
    )
    .expect("Encoding failed");

    // Decode
    let decoded = Decoder::new().decode(&jpeg).expect("Decoding failed");
    let extras = decoded.extras().expect("Should have extras");

    // Should have XMP with UltraHDR metadata
    let xmp = extras.xmp().expect("Should have XMP");
    assert!(
        xmp.contains("hdrgm:") || xmp.contains("GainMapMax"),
        "XMP should contain gain map metadata"
    );
}
