//! Integration tests for metadata compatibility with popular Rust crates.
//!
//! Tests verify that jpegli-rs output is compatible with:
//! - `kamadak-exif`: EXIF parsing (verifies JPEG structure doesn't confuse parsers)
//! - `img-parts`: JPEG segment manipulation (ICC profile extraction/injection)
//! - `ultrahdr`: Ultra HDR JPEG encoding (using jpegli as base encoder)

use jpegli::encoder::{ChromaSubsampling, EncoderConfig};
use rgb::RGB;

/// Create a test image with a gradient pattern.
fn create_test_image(width: u32, height: u32) -> Vec<RGB<u8>> {
    let mut pixels = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        for x in 0..width {
            let r = ((x * 255) / width) as u8;
            let g = ((y * 255) / height) as u8;
            let b = (((x + y) * 127) / (width + height)) as u8;
            pixels.push(RGB::new(r, g, b));
        }
    }
    pixels
}

/// Encode a test image to JPEG bytes.
fn encode_test_jpeg(width: u32, height: u32, quality: f32) -> Vec<u8> {
    let pixels = create_test_image(width, height);
    let config = EncoderConfig::new(quality, ChromaSubsampling::Quarter);
    let mut encoder = config.encode_from_rgb::<RGB<u8>>(width, height).unwrap();
    encoder.push_packed(&pixels, enough::Unstoppable).unwrap();
    encoder.finish().unwrap()
}

/// Encode with an ICC profile attached.
fn encode_with_icc_profile(width: u32, height: u32, quality: f32, icc: &[u8]) -> Vec<u8> {
    let pixels = create_test_image(width, height);
    let config = EncoderConfig::new(quality, ChromaSubsampling::Quarter).icc_profile(icc);
    let mut encoder = config.encode_from_rgb::<RGB<u8>>(width, height).unwrap();
    encoder.push_packed(&pixels, enough::Unstoppable).unwrap();
    encoder.finish().unwrap()
}

/// Encode with EXIF data attached using native API.
fn encode_with_exif(width: u32, height: u32, quality: f32, exif: &[u8]) -> Vec<u8> {
    let pixels = create_test_image(width, height);
    let config = EncoderConfig::new(quality, ChromaSubsampling::Quarter).exif(exif);
    let mut encoder = config.encode_from_rgb::<RGB<u8>>(width, height).unwrap();
    encoder.push_packed(&pixels, enough::Unstoppable).unwrap();
    encoder.finish().unwrap()
}

/// Encode with XMP data attached using native API.
fn encode_with_xmp(width: u32, height: u32, quality: f32, xmp: &[u8]) -> Vec<u8> {
    let pixels = create_test_image(width, height);
    let config = EncoderConfig::new(quality, ChromaSubsampling::Quarter).xmp(xmp);
    let mut encoder = config.encode_from_rgb::<RGB<u8>>(width, height).unwrap();
    encoder.push_packed(&pixels, enough::Unstoppable).unwrap();
    encoder.finish().unwrap()
}

/// Encode with all metadata types attached.
fn encode_with_all_metadata(
    width: u32,
    height: u32,
    quality: f32,
    exif: &[u8],
    xmp: &[u8],
    icc: &[u8],
) -> Vec<u8> {
    let pixels = create_test_image(width, height);
    let config = EncoderConfig::new(quality, ChromaSubsampling::Quarter)
        .exif(exif)
        .xmp(xmp)
        .icc_profile(icc);
    let mut encoder = config.encode_from_rgb::<RGB<u8>>(width, height).unwrap();
    encoder.push_packed(&pixels, enough::Unstoppable).unwrap();
    encoder.finish().unwrap()
}

// ============================================================================
// Native EXIF/XMP API tests
// ============================================================================

mod native_metadata_tests {
    use super::*;
    use img_parts::{jpeg::Jpeg, ImageEXIF, ImageICC};

    /// Create minimal EXIF TIFF structure (without the Exif\0\0 prefix).
    fn create_minimal_exif_tiff() -> Vec<u8> {
        let mut tiff = Vec::new();
        // TIFF header (little-endian)
        tiff.extend_from_slice(&[0x49, 0x49]); // II = little-endian
        tiff.extend_from_slice(&[0x2A, 0x00]); // Magic number
        tiff.extend_from_slice(&[0x08, 0x00, 0x00, 0x00]); // Offset to first IFD
                                                           // Minimal IFD0 with 0 entries
        tiff.extend_from_slice(&[0x00, 0x00]); // Number of entries
        tiff.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // Offset to next IFD (none)
        tiff
    }

    #[test]
    fn native_exif_embeds_correctly() {
        let exif_tiff = create_minimal_exif_tiff();
        let jpeg_data = encode_with_exif(128, 128, 80.0, &exif_tiff);

        // Parse with img-parts
        let jpeg = Jpeg::from_bytes(jpeg_data.clone().into()).expect("Should parse JPEG");

        // Extract EXIF - img-parts includes the Exif\0\0 prefix in extracted data
        let extracted = jpeg.exif().expect("EXIF should be present");

        // img-parts may or may not include the Exif\0\0 prefix depending on version
        // Check that our TIFF data is present somewhere in the extracted data
        let tiff_start = if extracted.starts_with(b"Exif\0\0") {
            &extracted[6..]
        } else {
            extracted.as_ref()
        };

        assert_eq!(
            tiff_start,
            exif_tiff.as_slice(),
            "EXIF TIFF data should match"
        );

        // Also verify by scanning raw JPEG bytes for our TIFF header
        let tiff_header = &[0x49, 0x49, 0x2A, 0x00]; // II + magic
        let has_tiff = jpeg_data.windows(4).any(|w| w == tiff_header);
        assert!(has_tiff, "TIFF header should be present in JPEG");
    }

    #[test]
    fn native_xmp_embeds_correctly() {
        let xmp = br#"<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/">
      <dc:creator>jpegli-rs test</dc:creator>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"#;

        let jpeg_data = encode_with_xmp(128, 128, 80.0, xmp);

        // Check that XMP APP1 marker is present
        // XMP uses http://ns.adobe.com/xap/1.0/\0 namespace
        let xmp_namespace = b"http://ns.adobe.com/xap/1.0/\0";

        let has_xmp = jpeg_data
            .windows(xmp_namespace.len())
            .any(|w| w == xmp_namespace);
        assert!(has_xmp, "XMP namespace should be present in JPEG");

        // Verify the XMP content is embedded
        let xmp_content_start = br#"<?xpacket begin"#;
        let has_content = jpeg_data
            .windows(xmp_content_start.len())
            .any(|w| w == xmp_content_start);
        assert!(has_content, "XMP content should be present");
    }

    #[test]
    fn all_metadata_in_correct_order() {
        let exif_tiff = create_minimal_exif_tiff();
        let xmp = b"<xmp>test</xmp>";
        let icc = vec![0x42u8; 128];

        let jpeg_data = encode_with_all_metadata(128, 128, 80.0, &exif_tiff, xmp, &icc);

        // Find positions of each marker type
        let exif_pos = jpeg_data.windows(6).position(|w| w == b"Exif\0\0");
        let xmp_pos = jpeg_data
            .windows(29)
            .position(|w| w == b"http://ns.adobe.com/xap/1.0/\0");
        let icc_pos = jpeg_data.windows(12).position(|w| w == b"ICC_PROFILE\0");

        assert!(exif_pos.is_some(), "EXIF should be present");
        assert!(xmp_pos.is_some(), "XMP should be present");
        assert!(icc_pos.is_some(), "ICC should be present");

        // Verify order: EXIF < XMP < ICC
        let exif_pos = exif_pos.unwrap();
        let xmp_pos = xmp_pos.unwrap();
        let icc_pos = icc_pos.unwrap();

        assert!(
            exif_pos < xmp_pos,
            "EXIF ({exif_pos}) should come before XMP ({xmp_pos})"
        );
        assert!(
            xmp_pos < icc_pos,
            "XMP ({xmp_pos}) should come before ICC ({icc_pos})"
        );
    }

    #[test]
    fn native_exif_compatible_with_kamadak_exif() {
        use std::io::Cursor;

        let exif_tiff = create_minimal_exif_tiff();
        let jpeg_data = encode_with_exif(128, 128, 80.0, &exif_tiff);

        // kamadak-exif should be able to read it
        let mut cursor = Cursor::new(&jpeg_data);
        let reader = exif::Reader::new();

        // We're embedding minimal/empty EXIF, so it might parse with 0 fields
        // The important thing is it doesn't error with "invalid JPEG"
        match reader.read_from_container(&mut cursor) {
            Ok(_exif) => {
                // Successfully parsed - may have 0 fields (our test data is minimal)
                // This is fine - the parse succeeded
            }
            Err(e) => {
                // Should not be a JPEG structure error
                let err_str = format!("{e}");
                assert!(
                    !err_str.contains("invalid") && !err_str.contains("malformed"),
                    "Should not have JPEG structure error: {e}"
                );
            }
        }
    }

    #[test]
    fn native_metadata_compatible_with_img_parts() {
        let exif_tiff = create_minimal_exif_tiff();
        let icc = vec![0x42u8; 256];

        // Use native API to embed both
        let jpeg_data = encode_with_all_metadata(128, 128, 80.0, &exif_tiff, b"", &icc);

        let jpeg = Jpeg::from_bytes(jpeg_data.into()).expect("Should parse");

        // Both should be extractable
        assert!(jpeg.exif().is_some(), "EXIF should be extractable");
        assert!(jpeg.icc_profile().is_some(), "ICC should be extractable");
    }

    #[test]
    fn native_metadata_compatible_with_ultrahdr() {
        use ultrahdr::jpeg::parse_jpeg_segments;

        let exif_tiff = create_minimal_exif_tiff();
        let xmp = b"<xmp>test</xmp>";
        let icc = vec![0x42u8; 64];

        let jpeg_data = encode_with_all_metadata(128, 128, 80.0, &exif_tiff, xmp, &icc);

        // Parse with ultrahdr
        let segments = parse_jpeg_segments(&jpeg_data).expect("Should parse");

        // Check for APP1 EXIF
        let has_exif = segments
            .iter()
            .any(|s| s.marker == 0xE1 && s.data.starts_with(b"Exif\0\0"));
        assert!(has_exif, "EXIF APP1 should be present");

        // Check for APP1 XMP
        let has_xmp = segments
            .iter()
            .any(|s| s.marker == 0xE1 && s.data.starts_with(b"http://ns.adobe.com/xap/1.0/\0"));
        assert!(has_xmp, "XMP APP1 should be present");

        // Check for APP2 ICC
        let has_icc = segments
            .iter()
            .any(|s| s.marker == 0xE2 && s.data.starts_with(b"ICC_PROFILE\0"));
        assert!(has_icc, "ICC APP2 should be present");
    }
}

// ============================================================================
// kamadak-exif integration tests
// ============================================================================

mod kamadak_exif_tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn jpegli_output_parseable_by_exif_crate() {
        // Encode a JPEG with jpegli
        let jpeg_data = encode_test_jpeg(256, 256, 85.0);

        // Verify it's a valid JPEG that exif crate can attempt to parse
        let mut cursor = Cursor::new(&jpeg_data);
        let reader = exif::Reader::new();

        // jpegli doesn't write EXIF, so we expect no EXIF data
        // but the parse should not panic or produce a malformed JPEG error
        match reader.read_from_container(&mut cursor) {
            Ok(exif) => {
                // Unexpected: jpegli produced EXIF data
                panic!("Unexpected EXIF data found: {:?}", exif.fields().count());
            }
            Err(exif::Error::NotFound(_)) => {
                // Expected: no EXIF data in jpegli output
            }
            Err(e) => {
                // Check it's not a JPEG format error
                let err_str = format!("{e}");
                assert!(
                    !err_str.contains("invalid") && !err_str.contains("malformed"),
                    "JPEG structure error from exif crate: {e}"
                );
            }
        }
    }

    #[test]
    fn progressive_jpegli_parseable() {
        // Progressive JPEGs have different structure
        let pixels = create_test_image(128, 128);
        let config = EncoderConfig::new(80.0, ChromaSubsampling::Quarter).progressive(true);
        let mut encoder = config.encode_from_rgb::<RGB<u8>>(128, 128).unwrap();
        encoder.push_packed(&pixels, enough::Unstoppable).unwrap();
        let jpeg_data = encoder.finish().unwrap();

        let mut cursor = Cursor::new(&jpeg_data);
        let reader = exif::Reader::new();

        match reader.read_from_container(&mut cursor) {
            Ok(_) => panic!("Unexpected EXIF data"),
            Err(exif::Error::NotFound(_)) => { /* Expected */ }
            Err(e) => {
                let err_str = format!("{e}");
                assert!(
                    !err_str.contains("invalid") && !err_str.contains("malformed"),
                    "Progressive JPEG structure error: {e}"
                );
            }
        }
    }

    #[test]
    fn various_subsampling_modes_parseable() {
        for subsampling in [
            ChromaSubsampling::None,
            ChromaSubsampling::Quarter,
            ChromaSubsampling::HalfHorizontal,
            ChromaSubsampling::HalfVertical,
        ] {
            let pixels = create_test_image(64, 64);
            let config = EncoderConfig::new(75.0, subsampling);
            let mut encoder = config.encode_from_rgb::<RGB<u8>>(64, 64).unwrap();
            encoder.push_packed(&pixels, enough::Unstoppable).unwrap();
            let jpeg_data = encoder.finish().unwrap();

            let mut cursor = Cursor::new(&jpeg_data);
            let reader = exif::Reader::new();

            match reader.read_from_container(&mut cursor) {
                Ok(_) => panic!("Unexpected EXIF data"),
                Err(exif::Error::NotFound(_)) => { /* Expected */ }
                Err(e) => {
                    let err_str = format!("{e}");
                    assert!(
                        !err_str.contains("invalid") && !err_str.contains("malformed"),
                        "Subsampling {subsampling:?} caused JPEG structure error: {e}"
                    );
                }
            }
        }
    }
}

// ============================================================================
// img-parts integration tests
// ============================================================================

mod img_parts_tests {
    use super::*;
    use img_parts::{jpeg::Jpeg, ImageEXIF, ImageICC};

    #[test]
    fn jpegli_output_parseable_by_img_parts() {
        let jpeg_data = encode_test_jpeg(256, 256, 85.0);

        // img-parts should be able to parse jpegli output
        let jpeg = Jpeg::from_bytes(jpeg_data.clone().into())
            .expect("img-parts failed to parse jpegli JPEG");

        // Verify basic structure
        assert!(!jpeg.segments().is_empty(), "JPEG should have segments");

        // No EXIF by default
        assert!(jpeg.exif().is_none(), "jpegli should not add EXIF");

        // No ICC by default
        assert!(jpeg.icc_profile().is_none(), "No ICC profile expected");
    }

    #[test]
    fn icc_profile_extractable_by_img_parts() {
        // Create a minimal sRGB-like ICC profile (simplified for testing)
        let test_icc = create_test_icc_profile();

        let jpeg_data = encode_with_icc_profile(128, 128, 80.0, &test_icc);

        let jpeg = Jpeg::from_bytes(jpeg_data.into())
            .expect("img-parts failed to parse jpegli JPEG with ICC");

        // Extract ICC profile
        let extracted_icc = jpeg.icc_profile().expect("ICC profile should be present");

        // Verify it matches what we embedded
        assert_eq!(
            extracted_icc.as_ref(),
            test_icc.as_slice(),
            "Extracted ICC profile should match embedded one"
        );
    }

    #[test]
    fn large_icc_profile_chunked_correctly() {
        // Create a large ICC profile (>64KB to test chunking)
        let large_icc = create_large_test_icc_profile(100_000);

        let jpeg_data = encode_with_icc_profile(64, 64, 75.0, &large_icc);

        let jpeg = Jpeg::from_bytes(jpeg_data.into())
            .expect("img-parts failed to parse jpegli JPEG with large ICC");

        let extracted_icc = jpeg
            .icc_profile()
            .expect("Large ICC profile should be present");

        assert_eq!(
            extracted_icc.len(),
            large_icc.len(),
            "Large ICC profile should be fully preserved"
        );
        assert_eq!(
            extracted_icc.as_ref(),
            large_icc.as_slice(),
            "Large ICC profile content should match"
        );
    }

    #[test]
    fn exif_can_be_injected_into_jpegli_output() {
        let jpeg_data = encode_test_jpeg(128, 128, 85.0);

        let mut jpeg =
            Jpeg::from_bytes(jpeg_data.into()).expect("img-parts failed to parse jpegli JPEG");

        // Create minimal EXIF data (just the header)
        let exif_data = create_minimal_exif();
        jpeg.set_exif(Some(exif_data.clone().into()));

        // Re-encode
        let mut output = Vec::new();
        jpeg.encoder()
            .write_to(&mut output)
            .expect("Failed to write modified JPEG");

        // Parse again and verify EXIF is present
        let jpeg2 = Jpeg::from_bytes(output.into()).expect("Failed to parse modified JPEG");
        let extracted_exif = jpeg2
            .exif()
            .expect("EXIF should be present after injection");
        assert_eq!(extracted_exif.as_ref(), exif_data.as_slice());
    }

    #[test]
    fn progressive_jpegli_compatible_with_img_parts() {
        let pixels = create_test_image(256, 256);
        let config = EncoderConfig::new(85.0, ChromaSubsampling::Quarter).progressive(true);
        let mut encoder = config.encode_from_rgb::<RGB<u8>>(256, 256).unwrap();
        encoder.push_packed(&pixels, enough::Unstoppable).unwrap();
        let jpeg_data = encoder.finish().unwrap();

        // img-parts should parse progressive jpegli JPEG
        let jpeg = Jpeg::from_bytes(jpeg_data.clone().into())
            .expect("img-parts should parse progressive jpegli JPEG");

        // Verify we can still do metadata operations on progressive JPEGs
        // img-parts abstracts away scan internals, so we test the API works
        assert!(!jpeg.segments().is_empty(), "Should have segments");

        // Verify ICC can be added to progressive JPEG
        let mut jpeg_mut = jpeg;
        let test_icc = vec![0x42u8; 128];
        jpeg_mut.set_icc_profile(Some(test_icc.clone().into()));

        let mut output = Vec::new();
        jpeg_mut
            .encoder()
            .write_to(&mut output)
            .expect("Should write");

        // Re-parse and verify ICC survived
        let jpeg2 = Jpeg::from_bytes(output.into()).expect("Should re-parse");
        let extracted = jpeg2.icc_profile().expect("ICC should be present");
        assert_eq!(extracted.as_ref(), test_icc.as_slice());

        // Verify original is actually progressive by checking raw bytes for SOF2
        let has_sof2 = jpeg_data.windows(2).any(|w| w[0] == 0xFF && w[1] == 0xC2); // SOF2 = progressive
        assert!(has_sof2, "JPEG should contain progressive SOF2 marker");
    }

    /// Create a minimal valid ICC profile for testing.
    fn create_test_icc_profile() -> Vec<u8> {
        // Minimal ICC profile header (128 bytes) + minimal tag table
        // This is not a real usable profile, just structurally valid for testing
        let mut profile = vec![0u8; 128];

        // Profile size (big-endian)
        let size: u32 = 128;
        profile[0..4].copy_from_slice(&size.to_be_bytes());

        // Preferred CMM type
        profile[4..8].copy_from_slice(b"jpe ");

        // Profile version (4.3)
        profile[8..12].copy_from_slice(&[4, 0x30, 0, 0]);

        // Device class: Display
        profile[12..16].copy_from_slice(b"mntr");

        // Color space: RGB
        profile[16..20].copy_from_slice(b"RGB ");

        // PCS: XYZ
        profile[20..24].copy_from_slice(b"XYZ ");

        // Profile signature
        profile[36..40].copy_from_slice(b"acsp");

        profile
    }

    /// Create a large ICC profile for chunking tests.
    fn create_large_test_icc_profile(size: usize) -> Vec<u8> {
        let mut profile = create_test_icc_profile();
        // Extend with padding
        profile.resize(size, 0x55);
        // Update size in header
        let size_bytes = (size as u32).to_be_bytes();
        profile[0..4].copy_from_slice(&size_bytes);
        profile
    }

    /// Create minimal EXIF data for testing.
    fn create_minimal_exif() -> Vec<u8> {
        // EXIF header + minimal TIFF structure
        let mut exif = Vec::new();

        // EXIF identifier
        exif.extend_from_slice(b"Exif\0\0");

        // TIFF header (little-endian)
        exif.extend_from_slice(&[0x49, 0x49]); // II = little-endian
        exif.extend_from_slice(&[0x2A, 0x00]); // Magic number
        exif.extend_from_slice(&[0x08, 0x00, 0x00, 0x00]); // Offset to first IFD

        // Minimal IFD0 with 0 entries
        exif.extend_from_slice(&[0x00, 0x00]); // Number of entries
        exif.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // Offset to next IFD (none)

        exif
    }
}

// ============================================================================
// ultrahdr integration tests
// ============================================================================

mod ultrahdr_tests {
    use super::*;
    use ultrahdr::jpeg::parse_jpeg_segments;
    use ultrahdr::metadata::xmp::{create_xmp_app1_marker, generate_xmp};
    use ultrahdr::GainMapMetadata;

    #[test]
    fn jpegli_output_parseable_as_ultrahdr_base() {
        let jpeg_data = encode_test_jpeg(256, 256, 85.0);

        // Parse with ultrahdr's JPEG parser
        let segments = parse_jpeg_segments(&jpeg_data).expect("ultrahdr should parse jpegli JPEG");

        // Verify structure
        assert!(segments.len() >= 2, "Should have at least SOI and EOI");
        assert_eq!(segments[0].marker, 0xD8, "First marker should be SOI");
        assert_eq!(
            segments.last().unwrap().marker,
            0xD9,
            "Last marker should be EOI"
        );
    }

    #[test]
    fn xmp_metadata_can_be_injected() {
        let jpeg_data = encode_test_jpeg(128, 128, 80.0);

        // Create gain map metadata
        let metadata = GainMapMetadata {
            min_content_boost: [1.0; 3],
            max_content_boost: [4.0; 3],
            gamma: [1.0; 3],
            offset_sdr: [0.015625; 3],
            offset_hdr: [0.015625; 3],
            hdr_capacity_min: 1.0,
            hdr_capacity_max: 4.0,
            use_base_color_space: true,
        };

        // Generate XMP
        let xmp = generate_xmp(&metadata, 5000);

        // Create APP1 marker
        let app1 = create_xmp_app1_marker(&xmp);

        // Inject after SOI
        let mut result = Vec::with_capacity(jpeg_data.len() + app1.len());
        result.extend_from_slice(&jpeg_data[..2]); // SOI
        result.extend_from_slice(&app1); // XMP APP1
        result.extend_from_slice(&jpeg_data[2..]); // Rest of JPEG

        // Parse and verify XMP is present
        let segments = parse_jpeg_segments(&result).expect("Should parse modified JPEG");

        let has_xmp_app1 = segments
            .iter()
            .any(|s| s.marker == 0xE1 && s.data.starts_with(b"http://ns.adobe.com/xap/1.0/\0"));

        assert!(has_xmp_app1, "XMP APP1 marker should be present");
    }

    #[test]
    fn jpegli_with_icc_compatible_with_ultrahdr() {
        let test_icc = vec![0u8; 256]; // Minimal ICC for testing

        let jpeg_data = encode_with_icc_profile(128, 128, 80.0, &test_icc);

        // Parse and verify ICC is present
        let segments =
            parse_jpeg_segments(&jpeg_data).expect("ultrahdr should parse jpegli+ICC JPEG");

        // Look for APP2 ICC marker
        let has_icc = segments
            .iter()
            .any(|s| s.marker == 0xE2 && s.data.starts_with(b"ICC_PROFILE\0"));

        assert!(has_icc, "ICC profile APP2 marker should be present");
    }

    #[test]
    fn progressive_jpegli_usable_as_ultrahdr_base() {
        let pixels = create_test_image(256, 256);
        let config = EncoderConfig::new(85.0, ChromaSubsampling::Quarter).progressive(true);
        let mut encoder = config.encode_from_rgb::<RGB<u8>>(256, 256).unwrap();
        encoder.push_packed(&pixels, enough::Unstoppable).unwrap();
        let jpeg_data = encoder.finish().unwrap();

        // Parse with ultrahdr
        let segments =
            parse_jpeg_segments(&jpeg_data).expect("ultrahdr should parse progressive jpegli");

        // Progressive should have multiple SOS markers
        let sos_count = segments.iter().filter(|s| s.marker == 0xDA).count();
        assert!(
            sos_count > 1,
            "Progressive JPEG should have multiple scans for ultrahdr"
        );
    }

    #[test]
    fn segment_reconstruction_preserves_jpegli_data() {
        use ultrahdr::jpeg::reconstruct_jpeg;

        let original = encode_test_jpeg(64, 64, 75.0);

        // Parse and reconstruct
        let segments = parse_jpeg_segments(&original).expect("Should parse");
        let reconstructed = reconstruct_jpeg(&segments);

        // Lengths should match (reconstruction shouldn't add/remove data)
        assert_eq!(
            original.len(),
            reconstructed.len(),
            "Reconstructed JPEG should have same length"
        );

        // Data should be identical
        assert_eq!(
            original, reconstructed,
            "Reconstructed JPEG should be byte-identical"
        );
    }

    #[test]
    fn ultrahdr_segment_injection_workflow() {
        use ultrahdr::jpeg::{insert_segment_after_soi, JpegSegment};

        let original = encode_test_jpeg(128, 128, 80.0);

        // Create a custom APP11 marker (used by some HDR formats)
        let custom_segment = JpegSegment {
            marker: 0xEB, // APP11
            data: b"TestData".to_vec(),
            offset: 0,
        };

        // Inject segment
        let modified =
            insert_segment_after_soi(&original, &custom_segment).expect("Injection should work");

        // Parse and verify
        let segments = parse_jpeg_segments(&modified).expect("Should parse modified JPEG");

        let has_custom = segments
            .iter()
            .any(|s| s.marker == 0xEB && s.data == b"TestData");

        assert!(has_custom, "Custom APP11 segment should be present");
    }
}

// ============================================================================
// Cross-crate compatibility tests
// ============================================================================

mod cross_crate_tests {
    use super::*;
    use img_parts::{jpeg::Jpeg, ImageEXIF, ImageICC};
    use std::io::Cursor;

    #[test]
    fn icc_roundtrip_jpegli_to_img_parts_to_exif_crate() {
        // Create ICC profile
        let test_icc = vec![0x42u8; 512];

        // Encode with jpegli
        let jpeg_data = encode_with_icc_profile(128, 128, 80.0, &test_icc);

        // Parse with img-parts
        let jpeg =
            Jpeg::from_bytes(jpeg_data.clone().into()).expect("img-parts should parse jpegli+ICC");

        let extracted_icc = jpeg.icc_profile().expect("ICC should be extractable");
        assert_eq!(extracted_icc.len(), test_icc.len());

        // Also verify exif crate doesn't choke on it
        let mut cursor = Cursor::new(&jpeg_data);
        let reader = exif::Reader::new();

        match reader.read_from_container(&mut cursor) {
            Ok(_) => panic!("Unexpected EXIF"),
            Err(exif::Error::NotFound(_)) => { /* Expected */ }
            Err(e) => {
                let err_str = format!("{e}");
                assert!(
                    !err_str.contains("invalid"),
                    "exif crate failed on ICC JPEG: {e}"
                );
            }
        }
    }

    #[test]
    fn img_parts_modified_jpeg_parseable_by_ultrahdr() {
        use ultrahdr::jpeg::parse_jpeg_segments;

        // Create jpegli JPEG
        let jpeg_data = encode_test_jpeg(128, 128, 80.0);

        // Modify with img-parts (add EXIF)
        let mut jpeg = Jpeg::from_bytes(jpeg_data.into()).expect("img-parts parse");
        jpeg.set_exif(Some(b"Exif\0\0TESTEXIF".to_vec().into()));

        let mut modified = Vec::new();
        jpeg.encoder()
            .write_to(&mut modified)
            .expect("img-parts write");

        // Parse with ultrahdr
        let segments =
            parse_jpeg_segments(&modified).expect("ultrahdr should parse img-parts output");

        // Verify EXIF APP1 present
        let has_exif = segments
            .iter()
            .any(|s| s.marker == 0xE1 && s.data.starts_with(b"Exif\0\0"));

        assert!(
            has_exif,
            "EXIF injected by img-parts should be parseable by ultrahdr"
        );
    }
}
