//! Tests for ICC profile extraction from real-world JPEG files.
//!
//! Tests read from imageflow's local cache at
//! `~/work/imageflow/.image-cache/sources/imageflow-resources/`.
//! If the cache is not present, tests are skipped.

use zenjpeg::color::icc::extract_icc_profile;

const REC2020_PATH: &str = "/home/lilith/work/imageflow/.image-cache/sources/imageflow-resources/test_inputs/wide-gamut/rec-2020-pq/flickr_2a68670c58131566.jpg";
const DISPLAY_P3_PATH: &str = "/home/lilith/work/imageflow/.image-cache/sources/imageflow-resources/test_inputs/wide-gamut/display-p3/flickr_c585e5e91ff47e1c.jpg";
const CANON_SRGB_PATH: &str = "/home/lilith/work/imageflow/.image-cache/sources/imageflow-resources/test_inputs/wide-gamut/srgb-reference/canon_eos_5d_mark_iv/wmc_81b268fc64ea796c.jpg";
const ADOBE_RGB_PATH: &str = "/home/lilith/work/imageflow/.image-cache/sources/imageflow-resources/test_inputs/wide-gamut/adobe-rgb/flickr_0119a8378404ece9.jpg";

/// Helper: read file or skip test.
fn read_or_skip(path: &str) -> Vec<u8> {
    match std::fs::read(path) {
        Ok(data) => data,
        Err(_) => {
            eprintln!("SKIP: cached image not found at {path}");
            // Return empty to let caller handle
            vec![]
        }
    }
}

// ============================================================================
// Low-level: extract_icc_profile (raw APP2 scanner)
// ============================================================================

#[test]
fn extract_icc_rec2020() {
    let data = read_or_skip(REC2020_PATH);
    if data.is_empty() { return; }

    let result = extract_icc_profile(&data);
    assert!(result.is_some(), "extract_icc_profile returned None for Rec.2020 PQ JPEG");
    let icc = result.unwrap();
    eprintln!("Rec.2020 ICC: {} bytes", icc.len());
    assert!(icc.len() > 100, "ICC too short: {} bytes", icc.len());
}

#[test]
fn extract_icc_display_p3() {
    let data = read_or_skip(DISPLAY_P3_PATH);
    if data.is_empty() { return; }

    let result = extract_icc_profile(&data);
    assert!(result.is_some(), "extract_icc_profile returned None for Display P3 JPEG");
    let icc = result.unwrap();
    eprintln!("Display P3 ICC: {} bytes", icc.len());
    assert!(icc.len() > 100, "ICC too short: {} bytes", icc.len());
}

#[test]
fn extract_icc_adobe_rgb() {
    let data = read_or_skip(ADOBE_RGB_PATH);
    if data.is_empty() { return; }

    let result = extract_icc_profile(&data);
    assert!(result.is_some(), "extract_icc_profile returned None for Adobe RGB JPEG");
    let icc = result.unwrap();
    eprintln!("Adobe RGB ICC: {} bytes", icc.len());
    assert!(icc.len() > 100, "ICC too short: {} bytes", icc.len());
}

#[test]
fn extract_icc_canon_srgb() {
    let data = read_or_skip(CANON_SRGB_PATH);
    if data.is_empty() { return; }

    let result = extract_icc_profile(&data);
    assert!(result.is_some(), "extract_icc_profile returned None for Canon sRGB JPEG");
    let icc = result.unwrap();
    eprintln!("Canon sRGB ICC: {} bytes", icc.len());
    assert!(icc.len() > 100, "ICC too short: {} bytes", icc.len());
}

// ============================================================================
// Mid-level: read_info (parser pipeline)
// ============================================================================

#[test]
fn read_info_returns_icc_rec2020() {
    let data = read_or_skip(REC2020_PATH);
    if data.is_empty() { return; }

    let decoder = zenjpeg::decoder::Decoder::new();
    let info = decoder.read_info(&data).expect("read_info failed");
    eprintln!("read_info: has_icc={}, icc.is_some()={}", info.has_icc_profile, info.icc_profile.is_some());
    if let Some(ref icc) = info.icc_profile {
        eprintln!("read_info ICC: {} bytes", icc.len());
    }
    assert!(
        info.icc_profile.is_some(),
        "read_info returned None for ICC on Rec.2020 PQ JPEG"
    );
}

#[test]
fn read_info_returns_icc_display_p3() {
    let data = read_or_skip(DISPLAY_P3_PATH);
    if data.is_empty() { return; }

    let decoder = zenjpeg::decoder::Decoder::new();
    let info = decoder.read_info(&data).expect("read_info failed");
    assert!(
        info.icc_profile.is_some(),
        "read_info returned None for ICC on Display P3 JPEG"
    );
}

#[test]
fn read_info_returns_icc_adobe_rgb() {
    let data = read_or_skip(ADOBE_RGB_PATH);
    if data.is_empty() { return; }

    let decoder = zenjpeg::decoder::Decoder::new();
    let info = decoder.read_info(&data).expect("read_info failed");
    assert!(
        info.icc_profile.is_some(),
        "read_info returned None for ICC on Adobe RGB JPEG"
    );
}

// ============================================================================
// High-level: zencodec trait probe (what zencodecs/imageflow uses)
// Requires `zencodec` feature.
// ============================================================================

#[cfg(feature = "zencodec")]
#[test]
fn zencodec_probe_returns_icc_rec2020() {
    let data = read_or_skip(REC2020_PATH);
    if data.is_empty() { return; }

    let config = zenjpeg::JpegDecoderConfig::new();
    let info = config.probe_header(&data).expect("probe_header failed");
    eprintln!("probe: icc.is_some() = {}", info.source_color.icc_profile.is_some());
    if let Some(ref icc) = info.source_color.icc_profile {
        eprintln!("probe ICC: {} bytes", icc.len());
    }
    assert!(
        info.source_color.icc_profile.is_some(),
        "probe_header returned None for ICC on Rec.2020 PQ JPEG"
    );
}

// ============================================================================
// Full decode: zencodec trait decode (what imageflow's zen pipeline uses)
// Requires `zencodec` feature.
// ============================================================================

#[cfg(feature = "zencodec")]
#[test]
fn zencodec_decode_returns_icc_rec2020() {
    let data = read_or_skip(REC2020_PATH);
    if data.is_empty() { return; }

    let config = zenjpeg::JpegDecoderConfig::new();
    let output = config.decode(&data).expect("decode failed");
    let icc = &output.info().source_color.icc_profile;
    eprintln!("decode: icc.is_some() = {}", icc.is_some());
    if let Some(icc_data) = icc {
        eprintln!("decode ICC: {} bytes", icc_data.len());
    }
    assert!(
        icc.is_some(),
        "decode returned None for ICC on Rec.2020 PQ JPEG"
    );
}

// ============================================================================
// Synthetic tests (no external files needed)
// ============================================================================

/// Build a minimal JPEG byte sequence containing an ICC profile in APP2.
fn build_jpeg_with_icc(icc_payload: &[u8]) -> Vec<u8> {
    let mut jpeg = vec![0xFF, 0xD8]; // SOI

    // APP0 (JFIF header)
    let app0_data = b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00";
    let app0_len = (app0_data.len() + 2) as u16;
    jpeg.extend_from_slice(&[0xFF, 0xE0]);
    jpeg.extend_from_slice(&app0_len.to_be_bytes());
    jpeg.extend_from_slice(app0_data);

    // APP2 with ICC_PROFILE signature, single chunk
    let sig = b"ICC_PROFILE\x00";
    let chunk_num: u8 = 1;
    let total_chunks: u8 = 1;
    let app2_len = (sig.len() + 2 + icc_payload.len() + 2) as u16;
    jpeg.extend_from_slice(&[0xFF, 0xE2]);
    jpeg.extend_from_slice(&app2_len.to_be_bytes());
    jpeg.extend_from_slice(sig);
    jpeg.push(chunk_num);
    jpeg.push(total_chunks);
    jpeg.extend_from_slice(icc_payload);

    // SOS (start of scan) -- triggers end of marker scan
    jpeg.extend_from_slice(&[0xFF, 0xDA]);
    // EOI
    jpeg.extend_from_slice(&[0xFF, 0xD9]);

    jpeg
}

#[test]
fn extract_icc_synthetic_single_chunk() {
    let icc_payload: Vec<u8> = (0..=255).cycle().take(1024).collect();
    let jpeg = build_jpeg_with_icc(&icc_payload);

    let result = extract_icc_profile(&jpeg);
    assert!(result.is_some(), "extract_icc_profile returned None for synthetic JPEG");
    let extracted = result.unwrap();
    assert_eq!(extracted.len(), icc_payload.len());
    assert_eq!(extracted, icc_payload);
}

/// Build a JPEG with multi-chunk ICC profile.
fn build_jpeg_with_multi_chunk_icc(icc_payload: &[u8], chunk_size: usize) -> Vec<u8> {
    let mut jpeg = vec![0xFF, 0xD8]; // SOI

    // APP0 (JFIF)
    let app0_data = b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00";
    let app0_len = (app0_data.len() + 2) as u16;
    jpeg.extend_from_slice(&[0xFF, 0xE0]);
    jpeg.extend_from_slice(&app0_len.to_be_bytes());
    jpeg.extend_from_slice(app0_data);

    let total_chunks = (icc_payload.len() + chunk_size - 1) / chunk_size;
    for (i, chunk) in icc_payload.chunks(chunk_size).enumerate() {
        let sig = b"ICC_PROFILE\x00";
        let chunk_num = (i + 1) as u8;
        let app2_len = (sig.len() + 2 + chunk.len() + 2) as u16;
        jpeg.extend_from_slice(&[0xFF, 0xE2]);
        jpeg.extend_from_slice(&app2_len.to_be_bytes());
        jpeg.extend_from_slice(sig);
        jpeg.push(chunk_num);
        jpeg.push(total_chunks as u8);
        jpeg.extend_from_slice(chunk);
    }

    // SOS
    jpeg.extend_from_slice(&[0xFF, 0xDA]);
    jpeg.extend_from_slice(&[0xFF, 0xD9]);

    jpeg
}

#[test]
fn extract_icc_synthetic_multi_chunk() {
    let icc_payload: Vec<u8> = (0..=255).cycle().take(6600).collect();
    let jpeg = build_jpeg_with_multi_chunk_icc(&icc_payload, 2200);

    let result = extract_icc_profile(&jpeg);
    assert!(result.is_some(), "extract_icc_profile returned None for multi-chunk ICC");
    let extracted = result.unwrap();
    assert_eq!(extracted.len(), icc_payload.len());
    assert_eq!(extracted, icc_payload);
}

#[test]
fn extract_icc_with_interleaved_app1() {
    let icc_payload: Vec<u8> = (0..=255).cycle().take(1024).collect();

    let mut jpeg = vec![0xFF, 0xD8]; // SOI

    // APP0 (JFIF)
    let app0_data = b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00";
    let app0_len = (app0_data.len() + 2) as u16;
    jpeg.extend_from_slice(&[0xFF, 0xE0]);
    jpeg.extend_from_slice(&app0_len.to_be_bytes());
    jpeg.extend_from_slice(app0_data);

    // APP1 (EXIF) -- before ICC
    let exif_data = b"Exif\x00\x00dummy exif data here";
    let app1_len = (exif_data.len() + 2) as u16;
    jpeg.extend_from_slice(&[0xFF, 0xE1]);
    jpeg.extend_from_slice(&app1_len.to_be_bytes());
    jpeg.extend_from_slice(exif_data);

    // APP2 with ICC
    let sig = b"ICC_PROFILE\x00";
    let app2_len = (sig.len() + 2 + icc_payload.len() + 2) as u16;
    jpeg.extend_from_slice(&[0xFF, 0xE2]);
    jpeg.extend_from_slice(&app2_len.to_be_bytes());
    jpeg.extend_from_slice(sig);
    jpeg.push(1);
    jpeg.push(1);
    jpeg.extend_from_slice(&icc_payload);

    // SOS
    jpeg.extend_from_slice(&[0xFF, 0xDA]);
    jpeg.extend_from_slice(&[0xFF, 0xD9]);

    let result = extract_icc_profile(&jpeg);
    assert!(result.is_some(), "extract_icc_profile failed with APP1 before APP2");
    assert_eq!(result.unwrap().len(), icc_payload.len());
}
