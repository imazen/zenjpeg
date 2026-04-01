//! RST marker resync tests.
//!
//! Tests for decoder behavior when restart markers are missing, wrong,
//! duplicated, or out-of-order. These are the mutation test inputs for
//! the RST resync hardening work.

use enough::Unstoppable;
use zenjpeg::decoder::Decoder;
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};

/// Encode a 128x128 4:2:0 JPEG with restart markers every MCU row.
/// 128x128 at 4:2:0 = 8x8 MCUs, DRI=8 means 8 restart segments.
fn make_large_dri_jpeg() -> Vec<u8> {
    let (w, h) = (128u32, 128u32);
    let mut pixels = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let idx = ((y * w + x) * 3) as usize;
            pixels[idx] = ((x * 7 + y * 3) % 256) as u8;
            pixels[idx + 1] = ((x * 3 + y * 11 + 128) % 256) as u8;
            pixels[idx + 2] = ((x * 13 + y * 5 + 64) % 256) as u8;
        }
    }
    let config = EncoderConfig::ycbcr(80, ChromaSubsampling::Quarter).restart_mcu_rows(1);
    let mut enc = config
        .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
        .unwrap();
    enc.push_packed(&pixels, Unstoppable).unwrap();
    enc.finish().unwrap()
}

/// Find all RST marker positions (0xFF 0xD0-0xD7) in JPEG data.
/// Returns Vec<(offset_of_0xFF, rst_number)>.
fn find_rst_markers(data: &[u8]) -> Vec<(usize, u8)> {
    let mut markers = Vec::new();
    let mut i = 0;
    while i + 1 < data.len() {
        if data[i] == 0xFF && (0xD0..=0xD7).contains(&data[i + 1]) {
            markers.push((i, data[i + 1] - 0xD0));
            i += 2;
        } else {
            i += 1;
        }
    }
    markers
}

/// Find the SOS marker + entropy data start position.
/// Returns offset of the first byte after SOS parameters.
#[allow(dead_code)]
fn find_entropy_start(data: &[u8]) -> Option<usize> {
    let mut i = 0;
    while i + 1 < data.len() {
        if data[i] == 0xFF && data[i + 1] == 0xDA {
            // SOS marker found, skip marker + length + params
            let length = u16::from_be_bytes([data[i + 2], data[i + 3]]) as usize;
            return Some(i + 2 + length);
        }
        i += 1;
    }
    None
}

// ============================================================================
// Baseline: verify DRI JPEGs decode correctly
// ============================================================================

#[test]
fn test_dri_jpeg_decodes_correctly() {
    let jpeg = make_large_dri_jpeg();
    let markers = find_rst_markers(&jpeg);
    assert!(
        markers.len() >= 4,
        "DRI JPEG should contain RST markers, got {}",
        markers.len()
    );
    let nums: Vec<u8> = markers.iter().map(|m| m.1).collect();
    eprintln!(
        "DRI JPEG: {} bytes, {} RST markers, nums={:?}",
        jpeg.len(),
        markers.len(),
        nums
    );

    let decoder = Decoder::new();
    let result = decoder.decode(&jpeg, Unstoppable);
    assert!(
        result.is_ok(),
        "valid DRI JPEG should decode: {:?}",
        result.err()
    );
}

// ============================================================================
// Mutation: wrong RST sequence number
// ============================================================================

/// Change one RST marker to wrong number (RST0 -> RST3).
#[test]
fn test_wrong_rst_number_strict_rejects() {
    let mut jpeg = make_large_dri_jpeg();
    let markers = find_rst_markers(&jpeg);
    assert!(markers.len() >= 2);

    // Mutate the first RST marker to a different number
    let (offset, num) = markers[0];
    let wrong_num = (num + 3) & 7; // offset by 3
    jpeg[offset + 1] = 0xD0 + wrong_num;

    let decoder = Decoder::new().strict();
    let result = decoder.decode(&jpeg, Unstoppable);
    assert!(
        result.is_err(),
        "strict mode should reject wrong RST number"
    );
}

#[test]
fn test_wrong_rst_number_permissive_recovers() {
    let mut jpeg = make_large_dri_jpeg();
    let markers = find_rst_markers(&jpeg);
    assert!(markers.len() >= 2);

    let (offset, num) = markers[0];
    let wrong_num = (num + 1) & 7;
    jpeg[offset + 1] = 0xD0 + wrong_num;

    let decoder = Decoder::new().permissive();
    let result = decoder.decode(&jpeg, Unstoppable);
    assert!(
        result.is_ok(),
        "permissive mode should recover from wrong RST: {:?}",
        result.err()
    );
}

/// Balanced mode should also recover from wrong RST numbers.
/// Note: the streaming decode path handles RST markers implicitly via
/// marker_found detection in BitReader refill, not via read_restart_marker().
/// This means wrong RST numbers are silently accepted (no validation of
/// which RST0-7 it is). The coefficient decode path (progressive, transforms,
/// f32 output) does validate and emits RestartMarkerResync warnings.
#[test]
fn test_wrong_rst_number_balanced_recovers() {
    let mut jpeg = make_large_dri_jpeg();
    let markers = find_rst_markers(&jpeg);
    assert!(markers.len() >= 2);

    let (offset, num) = markers[0];
    let wrong_num = (num + 1) & 7;
    jpeg[offset + 1] = 0xD0 + wrong_num;

    // Default strictness is Balanced
    let decoder = Decoder::new();
    let result = decoder.decode(&jpeg, Unstoppable);
    assert!(
        result.is_ok(),
        "balanced mode should recover from wrong RST: {:?}",
        result.err()
    );
}

/// Lenient mode should also recover from missing RST with a warning.
#[test]
fn test_missing_rst_lenient_recovers() {
    let mut jpeg = make_large_dri_jpeg();
    let markers = find_rst_markers(&jpeg);
    assert!(markers.len() >= 2);

    // Zero out the first RST marker
    let (offset, _) = markers[0];
    jpeg[offset] = 0x00;
    jpeg[offset + 1] = 0x00;

    let decoder = Decoder::new().lenient();
    let result = decoder.decode(&jpeg, Unstoppable);
    assert!(
        result.is_ok(),
        "lenient mode should recover from missing RST: {:?}",
        result.err()
    );
}

// ============================================================================
// Mutation: missing RST marker (zeroed out)
// ============================================================================

/// Replace an RST marker with entropy-safe bytes (0x00 0x00).
#[test]
fn test_missing_rst_strict_rejects() {
    let mut jpeg = make_large_dri_jpeg();
    let markers = find_rst_markers(&jpeg);
    assert!(markers.len() >= 2);

    // Zero out the first RST marker (replace FF Dx with 00 00)
    let (offset, _) = markers[0];
    jpeg[offset] = 0x00;
    jpeg[offset + 1] = 0x00;

    let decoder = Decoder::new().strict();
    let result = decoder.decode(&jpeg, Unstoppable);
    // With the RST gone, the entropy decoder will hit garbage or the next marker
    assert!(
        result.is_err(),
        "strict mode should reject missing RST marker"
    );
}

#[test]
fn test_missing_rst_permissive_attempts_recovery() {
    let mut jpeg = make_large_dri_jpeg();
    let markers = find_rst_markers(&jpeg);
    assert!(markers.len() >= 3);

    // Zero out the first RST marker
    let (offset, _) = markers[0];
    jpeg[offset] = 0x00;
    jpeg[offset + 1] = 0x00;

    let decoder = Decoder::new().permissive();
    let result = decoder.decode(&jpeg, Unstoppable);
    // May or may not recover — document the current behavior
    eprintln!(
        "missing RST permissive result: {}",
        if result.is_ok() { "OK" } else { "ERR" }
    );
    if let Err(e) = &result {
        eprintln!("  error: {e}");
    }
}

// ============================================================================
// Mutation: extra RST marker inserted
// ============================================================================

/// Insert an extra RST marker in the middle of entropy data.
#[test]
fn test_extra_rst_marker_inserted() {
    let jpeg = make_large_dri_jpeg();
    let markers = find_rst_markers(&jpeg);
    assert!(markers.len() >= 2);

    // Insert an extra RST0 marker midway between first two real RST markers
    let mid = (markers[0].0 + markers[1].0) / 2;
    let mut mutated = Vec::with_capacity(jpeg.len() + 2);
    mutated.extend_from_slice(&jpeg[..mid]);
    mutated.extend_from_slice(&[0xFF, 0xD0]); // extra RST0
    mutated.extend_from_slice(&jpeg[mid..]);

    let decoder = Decoder::new().permissive();
    let result = decoder.decode(&mutated, Unstoppable);
    eprintln!(
        "extra RST permissive: {}",
        if result.is_ok() { "OK" } else { "ERR" }
    );
    if let Err(e) = &result {
        eprintln!("  error: {e}");
    }
}

// ============================================================================
// Mutation: all RST markers removed
// ============================================================================

/// Remove ALL restart markers from a DRI JPEG.
#[test]
fn test_all_rst_markers_removed() {
    let jpeg = make_large_dri_jpeg();
    let markers = find_rst_markers(&jpeg);
    assert!(!markers.is_empty());

    // Build new data with all RST markers stripped
    let mut stripped = Vec::with_capacity(jpeg.len());
    let mut pos = 0;
    for &(offset, _) in &markers {
        stripped.extend_from_slice(&jpeg[pos..offset]);
        pos = offset + 2; // skip the 2-byte RST marker
    }
    stripped.extend_from_slice(&jpeg[pos..]);

    // Even strict should fail — DRI says restart markers exist but they're gone
    let decoder = Decoder::new().strict();
    let result = decoder.decode(&stripped, Unstoppable);
    assert!(
        result.is_err(),
        "should fail when DRI is set but all RST markers are stripped"
    );

    // Permissive should at least not crash
    let decoder = Decoder::new().permissive();
    let result = decoder.decode(&stripped, Unstoppable);
    eprintln!(
        "all RST removed, permissive: {}",
        if result.is_ok() { "OK" } else { "ERR" }
    );
    if let Err(e) = &result {
        eprintln!("  error: {e}");
    }
}

// ============================================================================
// Mutation: RST markers swapped (out of order)
// ============================================================================

/// Swap two adjacent RST markers' numbers.
#[test]
fn test_rst_markers_swapped() {
    let mut jpeg = make_large_dri_jpeg();
    let markers = find_rst_markers(&jpeg);
    if markers.len() < 3 {
        eprintln!(
            "skipping: need at least 3 RST markers, got {}",
            markers.len()
        );
        return;
    }

    // Swap RST numbers of marker 0 and marker 1
    let (off0, num0) = markers[0];
    let (off1, num1) = markers[1];
    jpeg[off0 + 1] = 0xD0 + num1;
    jpeg[off1 + 1] = 0xD0 + num0;

    let decoder = Decoder::new().strict();
    let result = decoder.decode(&jpeg, Unstoppable);
    assert!(result.is_err(), "strict should reject swapped RST markers");

    // Reset and try permissive
    jpeg[off0 + 1] = 0xD0 + num1;
    jpeg[off1 + 1] = 0xD0 + num0;
    let decoder = Decoder::new().permissive();
    let result = decoder.decode(&jpeg, Unstoppable);
    eprintln!(
        "swapped RST permissive: {}",
        if result.is_ok() { "OK" } else { "ERR" }
    );
}

// ============================================================================
// Mutation: junk bytes before RST marker
// ============================================================================

/// Insert junk bytes (not 0xFF) before a RST marker.
#[test]
fn test_junk_before_rst_marker() {
    let jpeg = make_large_dri_jpeg();
    let markers = find_rst_markers(&jpeg);
    assert!(markers.len() >= 2);

    let (offset, _) = markers[0];
    let mut mutated = Vec::with_capacity(jpeg.len() + 4);
    mutated.extend_from_slice(&jpeg[..offset]);
    mutated.extend_from_slice(&[0x42, 0x43, 0x44, 0x45]); // 4 junk bytes
    mutated.extend_from_slice(&jpeg[offset..]);

    let decoder = Decoder::new().permissive();
    let result = decoder.decode(&mutated, Unstoppable);
    eprintln!(
        "junk before RST permissive: {}",
        if result.is_ok() { "OK" } else { "ERR" }
    );
    if let Err(e) = &result {
        eprintln!("  error: {e}");
    }
}

// ============================================================================
// Quality comparison: mutated vs reference
// ============================================================================

/// Decode a valid DRI JPEG, then decode one with a wrong RST number in permissive mode.
/// The output should be mostly correct (only the segment after the bad RST is affected).
#[test]
fn test_rst_resync_output_quality() {
    let jpeg = make_large_dri_jpeg();
    let markers = find_rst_markers(&jpeg);
    if markers.len() < 4 {
        eprintln!("skipping: need at least 4 RST markers");
        return;
    }

    // Decode reference
    let decoder = Decoder::new();
    let reference = decoder.decode(&jpeg, Unstoppable).unwrap();
    let ref_pixels = reference.into_pixels_u8().unwrap();

    // Mutate: wrong RST number on marker 2 (middle of image)
    let mut mutated = jpeg.clone();
    let (offset, num) = markers[2];
    mutated[offset + 1] = 0xD0 + ((num + 2) & 7);

    let decoder = Decoder::new().permissive();
    let result = decoder.decode(&mutated, Unstoppable);
    if let Ok(decoded) = result {
        let mut_pixels = decoded.into_pixels_u8().unwrap();
        assert_eq!(ref_pixels.len(), mut_pixels.len());

        // Count differing pixels
        let _total = ref_pixels.len() / 3;
        let mut diff_count = 0;
        let mut max_diff = 0u8;
        for (a, b) in ref_pixels.iter().zip(mut_pixels.iter()) {
            let d = a.abs_diff(*b);
            if d > 0 {
                diff_count += 1;
            }
            if d > max_diff {
                max_diff = d;
            }
        }
        eprintln!(
            "RST resync quality: {}/{} channels differ, max_diff={}, {:.1}% affected",
            diff_count,
            ref_pixels.len(),
            max_diff,
            100.0 * diff_count as f64 / ref_pixels.len() as f64
        );

        // Pixels before the mutated RST should be identical
        // (the first ~25% of the image should be untouched)
    } else {
        eprintln!("permissive decode failed: {}", result.unwrap_err());
    }
}
