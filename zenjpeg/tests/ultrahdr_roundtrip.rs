//! UltraHDR roundtrip integration tests.
//!
//! Tests the full encode → decode → re-encode workflow.

#![cfg(feature = "ultrahdr")]

use zenjpeg::decoder::Decoder;
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig};
use zenjpeg::ultrahdr::{
    encode_ultrahdr, tonemapper_from_ultrahdr, GainMapConfig, ToneMapConfig, UhdrColorGamut,
    UhdrColorTransfer, UhdrPixelFormat, UhdrRawImage, UltraHdrExtras, UltraHdrMode,
    UltraHdrReaderConfig, Unstoppable,
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

// NOTE: reconstruct_hdr was removed from the API - use UltraHdrReader instead
#[test]
#[ignore = "reconstruct_hdr API was removed - use UltraHdrReader for HDR reconstruction"]
fn test_hdr_reconstruction() {
    // This test used reconstruct_hdr() which no longer exists.
    // Use UltraHdrReader with UltraHdrMode::Hdr for HDR reconstruction.
    unimplemented!("reconstruct_hdr API was removed");
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

// NOTE: reencode_ultrahdr was removed from the API
#[test]
#[ignore = "reencode_ultrahdr API was removed"]
fn test_reencode_ultrahdr() {
    // This test used reencode_ultrahdr() which no longer exists.
    unimplemented!("reencode_ultrahdr API was removed");
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

#[test]
fn test_gainmap_grayscale_roundtrip() {
    // Test that gainmap grayscale encode/decode preserves pixel values
    let hdr = create_test_hdr(64, 64);

    let jpeg = encode_ultrahdr(
        &hdr,
        &GainMapConfig::default(),
        &ToneMapConfig::default(),
        &EncoderConfig::ycbcr(95.0, ChromaSubsampling::None), // High quality for accurate test
        95.0,                                                 // High gainmap quality
        Unstoppable,
    )
    .expect("Encoding failed");

    let decoded = Decoder::new().decode(&jpeg).expect("Decoding failed");
    let extras = decoded.extras().expect("Should have extras");

    // Decode gainmap
    let gainmap = extras
        .decode_gainmap()
        .expect("Should have gain map")
        .expect("Gain map decode should succeed");

    // Verify gainmap properties
    assert!(gainmap.width > 0, "Gainmap width should be positive");
    assert!(gainmap.height > 0, "Gainmap height should be positive");
    assert_eq!(
        gainmap.channels, 1,
        "Gainmap should be single-channel (grayscale)"
    );

    // Verify gainmap has valid pixel data
    let expected_size = (gainmap.width * gainmap.height) as usize;
    assert_eq!(
        gainmap.data.len(),
        expected_size,
        "Gainmap data size mismatch: {} vs {}",
        gainmap.data.len(),
        expected_size
    );

    // Verify pixel values are not all zero
    let mut non_zero_count = 0;
    for &pixel in &gainmap.data {
        if pixel > 0 {
            non_zero_count += 1;
        }
    }
    assert!(
        non_zero_count > expected_size / 10,
        "Too many zero pixels in gainmap: {}/{}",
        non_zero_count,
        expected_size
    );
}

#[test]
fn test_gainmap_pixel_variance() {
    // Test that gainmap has meaningful variation (not all same value)
    let hdr = create_test_hdr(128, 128);

    let jpeg = encode_ultrahdr(
        &hdr,
        &GainMapConfig::default(),
        &ToneMapConfig::default(),
        &EncoderConfig::ycbcr(90.0, ChromaSubsampling::Quarter),
        85.0,
        Unstoppable,
    )
    .expect("Encoding failed");

    let decoded = Decoder::new().decode(&jpeg).expect("Decoding failed");
    let extras = decoded.extras().expect("Should have extras");
    let gainmap = extras
        .decode_gainmap()
        .expect("Should have gain map")
        .expect("Gain map decode should succeed");

    // Calculate min/max/mean
    let min = *gainmap.data.iter().min().unwrap_or(&0);
    let max = *gainmap.data.iter().max().unwrap_or(&255);
    let sum: u64 = gainmap.data.iter().map(|&p| p as u64).sum();
    let mean = sum as f64 / gainmap.data.len() as f64;

    // Verify there's meaningful variation in the gainmap
    let range = max - min;
    assert!(
        range >= 10,
        "Gainmap should have meaningful variation: min={}, max={}, range={}",
        min,
        max,
        range
    );

    // Mean should be somewhere in the middle (not pegged to extremes)
    assert!(
        mean > 20.0 && mean < 235.0,
        "Gainmap mean should be reasonable: {}",
        mean
    );
}

#[test]
fn test_standalone_grayscale_encode_decode() {
    // Test grayscale JPEG encode/decode independent of UltraHDR
    use zenjpeg::encoder::{EncoderConfig, PixelLayout};

    // Create gradient grayscale image
    let width = 64u32;
    let height = 64u32;
    let mut gray_data = vec![0u8; (width * height) as usize];
    for y in 0..height {
        for x in 0..width {
            // Gradient from dark to light
            gray_data[(y * width + x) as usize] = ((x + y) * 2).min(255) as u8;
        }
    }

    // Encode as grayscale
    let config = EncoderConfig::grayscale(95.0);
    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Gray8Srgb)
        .expect("encoder setup");
    enc.push_packed(&gray_data, Unstoppable).expect("push");
    let jpeg_data = enc.finish().expect("finish encode");

    // Verify JPEG structure
    assert_eq!(&jpeg_data[0..2], &[0xFF, 0xD8], "Should start with SOI");
    assert_eq!(
        &jpeg_data[jpeg_data.len() - 2..],
        &[0xFF, 0xD9],
        "Should end with EOI"
    );

    // Decode and verify
    let decoded = Decoder::new().decode(&jpeg_data).expect("decode failed");
    assert_eq!(decoded.width(), width);
    assert_eq!(decoded.height(), height);

    // Decoded grayscale may be expanded to RGB - check either format
    let decoded_pixels = decoded.pixels();
    let bytes_per_pixel = decoded_pixels.len() / (width * height) as usize;

    // Verify pixel values are approximately correct (lossy compression)
    for y in 0..height {
        for x in 0..width {
            let orig_val = gray_data[(y * width + x) as usize];
            let decoded_val = if bytes_per_pixel == 1 {
                decoded_pixels[(y * width + x) as usize]
            } else {
                // RGB format - take R channel (should equal G and B for grayscale)
                decoded_pixels[((y * width + x) * bytes_per_pixel as u32) as usize]
            };

            // Allow some error due to JPEG compression
            let diff = (orig_val as i32 - decoded_val as i32).unsigned_abs();
            assert!(
                diff < 15,
                "Pixel at ({},{}) differs too much: orig={}, decoded={}, diff={}",
                x,
                y,
                orig_val,
                decoded_val,
                diff
            );
        }
    }
}

/// Test that UltraHdrReader correctly detects UltraHDR files.
///
/// This test currently FAILS because UltraHdrReader's `extract_gainmap_early()`
/// does not find the gain map in this valid UltraHDR file, even though:
/// - The file has correct XMP with hdrgm:Version, hdrgm:GainMapMax, etc.
/// - The file has correct MPF marker with secondary image (type=Undefined=gainmap)
/// - The native `ultrahdr_rs::Decoder` correctly detects it as UltraHDR
///
/// Bug: `extract_gainmap_early()` returns `(None, None)` for this file.
///
/// File structure (verified with xxd):
/// - Offset 0x02: MPF marker (APP2) - indicates 2 images
/// - Offset 0x62: ICC_PROFILE (APP2)
/// - Offset 0x212: XMP (APP1) with hdrgm:* metadata
/// - Offset 0x1a9d: Secondary JPEG (gain map, 801 bytes)
#[test]
fn test_ultrahdr_reader_detection_bug() {
    let sample_path = "tests/images/ultrahdr_sample.jpg";
    let data = std::fs::read(sample_path).expect("Failed to read test file");

    // Verify with ultrahdr-rs crate directly - this WORKS
    let ultrahdr_decoder =
        ultrahdr_rs::Decoder::new(&data).expect("ultrahdr_rs::Decoder creation failed");
    assert!(
        ultrahdr_decoder.is_ultrahdr(),
        "ultrahdr_rs::Decoder should detect this as UltraHDR"
    );
    assert!(
        ultrahdr_decoder.metadata().is_some(),
        "ultrahdr_rs::Decoder should find metadata"
    );
    assert!(
        ultrahdr_decoder.gainmap_jpeg().is_some(),
        "ultrahdr_rs::Decoder should find gainmap JPEG"
    );

    // Now test with UltraHdrReader - this FAILS
    let config = UltraHdrReaderConfig::new()
        .mode(UltraHdrMode::SdrAndGainMap)
        .preserve_metadata(true);

    let reader = Decoder::new()
        .ultrahdr_reader(&data, config)
        .expect("UltraHdrReader creation should succeed");

    // BUG: This assertion fails - UltraHdrReader.is_ultrahdr() returns false
    // even though the file is valid UltraHDR (as proven by ultrahdr_rs::Decoder above)
    assert!(
        reader.is_ultrahdr(),
        "BUG: UltraHdrReader should detect this as UltraHDR, but returns false. \
         The file is valid UltraHDR (confirmed by ultrahdr_rs::Decoder). \
         extract_gainmap_early() is not finding the XMP metadata or MPF gain map."
    );
}

#[test]
fn test_gainmap_various_sizes() {
    // Test gainmap encoding at various resolutions
    for size in [16, 32, 64, 128] {
        let hdr = create_test_hdr(size, size);

        let jpeg = encode_ultrahdr(
            &hdr,
            &GainMapConfig::default(),
            &ToneMapConfig::default(),
            &EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter),
            75.0,
            Unstoppable,
        )
        .unwrap_or_else(|e| panic!("Encoding failed for size {}: {:?}", size, e));

        let decoded = Decoder::new()
            .decode(&jpeg)
            .unwrap_or_else(|e| panic!("Decoding failed for size {}: {:?}", size, e));

        let extras = decoded.extras().expect("Should have extras");
        let gainmap = extras
            .decode_gainmap()
            .expect("Should have gain map")
            .unwrap_or_else(|e| panic!("Gain map decode failed for size {}: {:?}", size, e));

        // Gainmap should be smaller than source (due to scale_factor)
        assert!(
            gainmap.width <= size && gainmap.height <= size,
            "Gainmap should be <= source size: {}x{} vs {}x{}",
            gainmap.width,
            gainmap.height,
            size,
            size
        );

        // Verify it's valid grayscale
        assert_eq!(gainmap.channels, 1, "Should be grayscale");
        assert_eq!(
            gainmap.data.len(),
            (gainmap.width * gainmap.height) as usize,
            "Data size mismatch"
        );
    }
}
