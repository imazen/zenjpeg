//! UltraHDR roundtrip integration tests.
//!
//! Tests the full encode → decode → re-encode workflow.
#![cfg(feature = "ultrahdr")]

use enough::Unstoppable;
use ultrahdr_core::pixel_buffer_from_vec;
use zenjpeg::decoder::Decoder;
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig};
use zenjpeg::ultrahdr::{
    GainMapConfig, ToneMapConfig, UhdrColorGamut, UhdrColorTransfer, UhdrPixelFormat, UhdrRawImage,
    UltraHdrExtras, UltraHdrMode, UltraHdrReaderConfig, encode_ultrahdr, tonemapper_from_ultrahdr,
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

    pixel_buffer_from_vec(
        data,
        width,
        height,
        UhdrPixelFormat::RgbaF32,
        UhdrColorGamut::Bt709,
        UhdrColorTransfer::Linear,
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
    let decoded = Decoder::new()
        .decode(&jpeg, Unstoppable)
        .expect("Decoding failed");

    assert_eq!(decoded.width(), width);
    assert_eq!(decoded.height(), height);

    let extras = decoded.extras().expect("Should have extras");
    assert!(extras.is_ultrahdr(), "Should be detected as UltraHDR");

    // Parse metadata
    let (metadata, _) = extras
        .ultrahdr_metadata()
        .expect("Should have metadata")
        .expect("Metadata parsing should succeed");

    // Verify metadata has reasonable values (gain_map_max is log2 [f64; 3] per channel)
    assert!(
        metadata.channels[0].max as f32 > 0.0
            || metadata.channels[1].max as f32 > 0.0
            || metadata.channels[2].max as f32 > 0.0,
        "HDR should have gain_map_max > 0.0 (log2 boost > 1.0) in at least one channel"
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
    let decoded = Decoder::new()
        .decode(&jpeg, Unstoppable)
        .expect("Decoding failed");
    let extras = decoded.extras().expect("Should have extras");

    let tonemapper =
        tonemapper_from_ultrahdr(extras).expect("Tonemapper extraction should succeed");

    // Tonemapper should be usable
    let test_input = pixel_buffer_from_vec(
        vec![0u8; 64],
        2,
        2,
        UhdrPixelFormat::RgbaF32,
        UhdrColorGamut::Bt709,
        UhdrColorTransfer::Linear,
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

/// Smoke test for the idiot-proof convenience entry: a single call should
/// produce a valid Ultra HDR JPEG from an HDR `PixelBuffer` with no config
/// assembly required.
#[test]
fn encode_ultrahdr_luma_smoke() {
    use ultrahdr_core::gainmap::HdrOutputFormat;
    use zenjpeg::ultrahdr::{decode_ultrahdr, decode_ultrahdr_hdr, encode_ultrahdr_luma};

    let hdr = create_test_hdr(32, 32);
    let bytes = encode_ultrahdr_luma(&hdr).expect("encode_ultrahdr_luma should succeed");

    // Output must be a JPEG with SOI/EOI bookends and UltraHDR XMP.
    assert_eq!(&bytes[0..2], &[0xFF, 0xD8]);
    assert_eq!(&bytes[bytes.len() - 2..], &[0xFF, 0xD9]);

    // SDR-decode round-trip via convenience entry: should yield Rgba8 sRGB.
    let sdr = decode_ultrahdr(&bytes).expect("decode_ultrahdr should succeed");
    assert_eq!(sdr.width(), 32);
    assert_eq!(sdr.height(), 32);

    // HDR-decode at 4× boost via convenience entry: linear f32 RGBA.
    let hdr_back = decode_ultrahdr_hdr(&bytes, 4.0, HdrOutputFormat::LinearFloat)
        .expect("decode_ultrahdr_hdr should succeed");
    assert_eq!(hdr_back.width(), 32);
    assert_eq!(hdr_back.height(), 32);
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
    let decoded = Decoder::new()
        .decode(&jpeg, Unstoppable)
        .expect("Decoding failed");
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

    let decoded = Decoder::new()
        .decode(&jpeg, Unstoppable)
        .expect("Decoding failed");
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

    let decoded = Decoder::new()
        .decode(&jpeg, Unstoppable)
        .expect("Decoding failed");
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
    let decoded = Decoder::new()
        .decode(&jpeg_data, Unstoppable)
        .expect("decode failed");
    assert_eq!(decoded.width(), width);
    assert_eq!(decoded.height(), height);

    // Decoded grayscale may be expanded to RGB - check either format
    let decoded_pixels = decoded.pixels_u8().unwrap();
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
/// Verifies that both ultrahdr_rs::Decoder and zenjpeg's UltraHdrReader
/// correctly detect and parse UltraHDR JPEGs with gain map metadata in
/// the secondary JPEG's XMP (the modern format used by libultrahdr).
#[test]
fn test_ultrahdr_reader_detection_bug() {
    // Generate a baseline UltraHDR JPEG (UltraHdrReader requires baseline)
    let hdr = create_test_hdr(64, 64);
    let data = encode_ultrahdr(
        &hdr,
        &GainMapConfig::default(),
        &ToneMapConfig::default(),
        &EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter).progressive(false),
        75.0,
        Unstoppable,
    )
    .expect("Encoding failed");

    // Verify with ultrahdr-rs crate directly
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

    // Test with UltraHdrReader
    let config = UltraHdrReaderConfig::new()
        .mode(UltraHdrMode::SdrAndGainMap)
        .preserve_metadata(true);

    let reader = Decoder::new()
        .ultrahdr_reader(&data, config)
        .expect("UltraHdrReader creation should succeed");

    assert!(
        reader.is_ultrahdr(),
        "UltraHdrReader should detect this as UltraHDR"
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
            .decode(&jpeg, Unstoppable)
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

// ============================================================================
// Coverage tests for previously-untested public HDR functions.
// Audit (2026-04-26): the existing suite covered encode_ultrahdr,
// decode_ultrahdr, decode_ultrahdr_hdr, encode_ultrahdr_luma,
// tonemapper_from_ultrahdr, and create_hdr_reconstructor. These tests
// fill the gap for encode_ultrahdr_with_curve, encode_ultrahdr_with_tonemapper,
// create_gainmap_computer, encode_with_gainmap, and encode_with_gainmap_format.
// ============================================================================

#[test]
fn encode_ultrahdr_with_curve_smoke() {
    // Closes #71 — the LumaToneMap path. Bt2446C is the published default
    // (used by encode_ultrahdr_luma). Driving it explicitly here means the
    // single-channel splitter path stays exercised even if encode_ultrahdr_luma
    // changes its default curve.
    use zenjpeg::ultrahdr::encode_ultrahdr_with_curve;
    use zentone::Bt2446C;

    let hdr = create_test_hdr(64, 64);
    let jpeg = encode_ultrahdr_with_curve(
        &hdr,
        &Bt2446C::new(1000.0, 203.0),
        &GainMapConfig::default(),
        &EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter),
        75.0,
        Unstoppable,
    )
    .expect("encode_ultrahdr_with_curve should succeed");

    assert_eq!(&jpeg[0..2], &[0xFF, 0xD8]);
    assert_eq!(&jpeg[jpeg.len() - 2..], &[0xFF, 0xD9]);

    let decoded = Decoder::new().decode(&jpeg, Unstoppable).unwrap();
    let extras = decoded.extras().expect("extras");
    assert!(extras.is_ultrahdr(), "should land as a real UltraHDR JPEG");
}

#[test]
fn encode_ultrahdr_with_tonemapper_round_trip() {
    // Encode once via encode_ultrahdr; pull the AdaptiveTonemapper out of the
    // resulting gain map; re-encode a (modified) HDR pair using that learned
    // tonemapper. This is the use case for editing HDR content without
    // re-deriving the SDR curve.
    use zenjpeg::ultrahdr::encode_ultrahdr_with_tonemapper;

    let hdr_first = create_test_hdr(64, 64);
    let jpeg_first = encode_ultrahdr(
        &hdr_first,
        &GainMapConfig::default(),
        &ToneMapConfig::default(),
        &EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter),
        75.0,
        Unstoppable,
    )
    .expect("first encode");

    let decoded_first = Decoder::new().decode(&jpeg_first, Unstoppable).unwrap();
    let extras = decoded_first.extras().expect("extras");
    let tonemapper = tonemapper_from_ultrahdr(extras).expect("tonemapper extraction");

    // "Edited" HDR — same shape, different content.
    let hdr_second = create_test_hdr(64, 64);
    let jpeg_second = encode_ultrahdr_with_tonemapper(
        &hdr_second,
        &tonemapper,
        &GainMapConfig::default(),
        &EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter),
        75.0,
        Unstoppable,
    )
    .expect("re-encode with learned tonemapper");

    assert_eq!(&jpeg_second[0..2], &[0xFF, 0xD8]);
    let decoded_second = Decoder::new().decode(&jpeg_second, Unstoppable).unwrap();
    let extras2 = decoded_second.extras().expect("extras");
    assert!(extras2.is_ultrahdr());
}

/// Build a synthetic SDR `PixelBuffer` (Rgba8 / sRGB) the same shape as the
/// HDR test image. Used by the encode_with_gainmap_* coverage tests, which
/// need a paired HDR+SDR to feed `compute_gainmap` — we don't have a public
/// `tonemap_to_pixel_buffer` to derive SDR from HDR ourselves.
fn create_test_sdr(width: u32, height: u32) -> ultrahdr_core::PixelBuffer {
    let mut data = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            // Slightly different gradient than the HDR so the gain map
            // doesn't collapse to identity.
            data.push(((x * 255) / width.max(1)) as u8);
            data.push(((y * 200) / height.max(1)) as u8);
            data.push(128);
            data.push(0xFF);
        }
    }
    pixel_buffer_from_vec(
        data,
        width,
        height,
        UhdrPixelFormat::Rgba8,
        UhdrColorGamut::Bt709,
        UhdrColorTransfer::Srgb,
    )
    .expect("synthetic SDR PixelBuffer")
}

#[test]
fn create_gainmap_computer_constructs() {
    // The streaming gain-map computer is the alternative to compute_gainmap()
    // for memory-bounded callers. Building one validates that the descriptor
    // negotiation works for the canonical HDR-RGBA-F32 + Bt709 inputs.
    use ultrahdr_core::ColorPrimaries;
    use zenjpeg::ultrahdr::create_gainmap_computer;

    let _row_encoder =
        create_gainmap_computer(128, 128, &GainMapConfig::default(), ColorPrimaries::Bt709)
            .expect("create_gainmap_computer Bt709 ok");

    // Bt2020 gamut path (HDR10 source primary) — separate code branch in
    // RowEncoder::new for the gamut-conversion matrix.
    let _row_encoder_2020 =
        create_gainmap_computer(128, 128, &GainMapConfig::default(), ColorPrimaries::Bt2020)
            .expect("create_gainmap_computer Bt2020 ok");
}

#[test]
fn encode_with_gainmap_default_format() {
    // encode_with_gainmap delegates to encode_with_gainmap_format with
    // GainMapEncodingFormat::Both. Drive it with synthetic HDR + SDR via the
    // public compute_gainmap to exercise the lower-level entry point.
    use zenjpeg::ultrahdr::{compute_gainmap, encode_with_gainmap};

    let hdr = create_test_hdr(64, 64);
    let sdr = create_test_sdr(64, 64);
    let (gainmap, metadata) = compute_gainmap(&hdr, &sdr, &GainMapConfig::default(), &Unstoppable)
        .expect("gainmap compute");

    let jpeg = encode_with_gainmap(
        &sdr,
        &gainmap,
        &metadata,
        &EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter),
        75.0,
        Unstoppable,
    )
    .expect("encode_with_gainmap");

    assert_eq!(&jpeg[0..2], &[0xFF, 0xD8]);
    let decoded = Decoder::new().decode(&jpeg, Unstoppable).unwrap();
    let extras = decoded.extras().expect("extras");
    assert!(
        extras.is_ultrahdr(),
        "default format should land as UltraHDR"
    );
}

#[test]
fn encode_with_gainmap_format_iso21496() {
    // The format-explicit overload lets callers pick the metadata serialization.
    // Iso21496 is the ISO 21496-1 box-only path (no XMP); useful for callers
    // that care about minimum metadata size on the wire.
    use zenjpeg::ultrahdr::{GainMapEncodingFormat, compute_gainmap, encode_with_gainmap_format};

    let hdr = create_test_hdr(64, 64);
    let sdr = create_test_sdr(64, 64);
    let (gainmap, metadata) = compute_gainmap(&hdr, &sdr, &GainMapConfig::default(), &Unstoppable)
        .expect("gainmap compute");

    let jpeg_iso = encode_with_gainmap_format(
        &sdr,
        &gainmap,
        &metadata,
        &EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter),
        75.0,
        GainMapEncodingFormat::Iso21496,
        Unstoppable,
    )
    .expect("encode_with_gainmap_format Iso21496");
    assert_eq!(&jpeg_iso[0..2], &[0xFF, 0xD8]);

    // Both-format and Iso-only outputs differ in size: Both embeds XMP +
    // ISO box, Iso embeds only the box. Sanity-check that the two paths
    // produce DIFFERENT bytes (otherwise the format knob is silently a no-op).
    let jpeg_both = encode_with_gainmap_format(
        &sdr,
        &gainmap,
        &metadata,
        &EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter),
        75.0,
        GainMapEncodingFormat::Both,
        Unstoppable,
    )
    .expect("encode_with_gainmap_format Both");
    assert_ne!(
        jpeg_iso, jpeg_both,
        "Iso21496 and Both should produce different bytes"
    );
}
