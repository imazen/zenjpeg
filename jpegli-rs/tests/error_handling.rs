//! Error handling tests.
//!
//! Tests matching C++ jpegli error handling scenarios from error_handling_test.cc
//! and various *_test.cc files.

#[path = "../src/test_utils.rs"]
mod test_utils;

use test_utils::{generate_gradient_d, TestImage};

use jpegli::{
    decoder::Decoder,
    encoder::{ChromaSubsampling, EncoderConfig, PixelLayout},
};

// Helper to encode RGB data with v2 API
fn encode_rgb(width: u32, height: u32, data: &[u8]) -> jpegli::encoder::Result<Vec<u8>> {
    let config = EncoderConfig::new(90.0, ChromaSubsampling::Quarter);
    let mut enc = config.encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)?;
    enc.push_packed(data, enough::Unstoppable)?;
    enc.finish()
}

fn encode_rgb_q(
    width: u32,
    height: u32,
    data: &[u8],
    quality: impl Into<jpegli::encoder::Quality>,
) -> jpegli::encoder::Result<Vec<u8>> {
    let config = EncoderConfig::new(quality, ChromaSubsampling::Quarter);
    let mut enc = config.encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)?;
    enc.push_packed(data, enough::Unstoppable)?;
    enc.finish()
}

fn encode_gray(width: u32, height: u32, data: &[u8]) -> jpegli::encoder::Result<Vec<u8>> {
    let config = EncoderConfig::new(90.0, ChromaSubsampling::Quarter).grayscale();
    let mut enc = config.encode_from_bytes(width, height, PixelLayout::Gray8Srgb)?;
    enc.push_packed(data, enough::Unstoppable)?;
    enc.finish()
}

fn encode_progressive(width: u32, height: u32, data: &[u8]) -> jpegli::encoder::Result<Vec<u8>> {
    let config = EncoderConfig::new(90.0, ChromaSubsampling::Quarter).progressive(true);
    let mut enc = config.encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)?;
    enc.push_packed(data, enough::Unstoppable)?;
    enc.finish()
}

// ============================================================================
// Encoder Input Validation Tests
// ============================================================================

#[test]
fn test_encode_zero_width() {
    let img = TestImage::new(0, 64, 3);
    let result = encode_rgb(0, 64, &img.pixels);
    assert!(result.is_err(), "Should reject zero width");
}

#[test]
fn test_encode_zero_height() {
    let img = TestImage::new(64, 0, 3);
    let result = encode_rgb(64, 0, &img.pixels);
    assert!(result.is_err(), "Should reject zero height");
}

#[test]
fn test_encode_zero_dimensions() {
    let img = TestImage::new(0, 0, 3);
    let result = encode_rgb(0, 0, &img.pixels);
    assert!(result.is_err(), "Should reject zero dimensions");
}

#[test]
fn test_encode_dimension_mismatch() {
    // Pixels for 64x64 but encoder configured for 128x128
    let img = generate_gradient_d(64, 64, 3);
    let result = encode_rgb(128, 128, &img.pixels);
    assert!(result.is_err(), "Should reject mismatched dimensions");
}

#[test]
fn test_encode_empty_input() {
    let result = encode_rgb(64, 64, &[]);
    assert!(result.is_err(), "Should reject empty input");
}

#[test]
fn test_encode_partial_input() {
    // Only half the required pixels
    let partial = vec![128u8; 64 * 64]; // Need 64*64*3
    let result = encode_rgb(64, 64, &partial);
    assert!(result.is_err(), "Should reject partial input");
}

// ============================================================================
// Encoder Quality Validation Tests
// ============================================================================

#[test]
fn test_encode_quality_boundary_low() {
    let img = generate_gradient_d(64, 64, 3);
    let result = encode_rgb_q(64, 64, &img.pixels, 1.0);
    assert!(result.is_ok(), "Q1 should be valid");
}

#[test]
fn test_encode_quality_boundary_high() {
    let img = generate_gradient_d(64, 64, 3);
    let result = encode_rgb_q(64, 64, &img.pixels, 100.0);
    assert!(result.is_ok(), "Q100 should be valid");
}

// ============================================================================
// Decoder Input Validation Tests
// ============================================================================

#[test]
fn test_decode_empty_input() {
    let decoder = Decoder::new();
    let result = decoder.decode(&[]);
    assert!(result.is_err(), "Should reject empty input");
}

#[test]
fn test_decode_single_byte() {
    let decoder = Decoder::new();
    let result = decoder.decode(&[0xFF]);
    assert!(result.is_err(), "Should reject single byte");
}

#[test]
fn test_decode_only_soi() {
    let decoder = Decoder::new();
    let result = decoder.decode(&[0xFF, 0xD8]);
    assert!(result.is_err(), "Should reject SOI-only input");
}

#[test]
fn test_decode_only_soi_eoi() {
    let decoder = Decoder::new();
    let result = decoder.decode(&[0xFF, 0xD8, 0xFF, 0xD9]);
    assert!(
        result.is_err(),
        "Should reject minimal SOI+EOI (no image data)"
    );
}

#[test]
fn test_decode_missing_soi() {
    // Valid-looking structure but wrong start marker
    let bad = vec![0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46];
    let decoder = Decoder::new();
    let result = decoder.decode(&bad);
    assert!(result.is_err(), "Should reject missing SOI");
}

#[test]
fn test_decode_wrong_magic() {
    // Not a JPEG at all
    let png_header = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
    let decoder = Decoder::new();
    let result = decoder.decode(&png_header);
    assert!(result.is_err(), "Should reject PNG data");
}

#[test]
fn test_decode_random_data() {
    let random: Vec<u8> = (0..1000).map(|i| ((i * 17 + 31) % 256) as u8).collect();
    let decoder = Decoder::new();
    let result = decoder.decode(&random);
    assert!(result.is_err(), "Should reject random data");
}

// ============================================================================
// Truncated Input Tests
// ============================================================================

fn create_test_jpeg() -> Vec<u8> {
    let img = generate_gradient_d(64, 64, 3);
    encode_rgb(64, 64, &img.pixels).expect("encode failed")
}

#[test]
fn test_decode_truncated_header() {
    let jpeg = create_test_jpeg();
    // Truncate in the header area
    let truncated = &jpeg[..20.min(jpeg.len())];
    let decoder = Decoder::new();
    let result = decoder.decode(truncated);
    assert!(result.is_err(), "Should reject truncated header");
}

#[test]
fn test_decode_truncated_tables() {
    let jpeg = create_test_jpeg();
    // Truncate after APP0 but before scan data
    let truncated = &jpeg[..100.min(jpeg.len())];
    let decoder = Decoder::new();
    let result = decoder.decode(truncated);
    assert!(result.is_err(), "Should reject truncated tables");
}

#[test]
fn test_decode_truncated_scan_data() {
    let jpeg = create_test_jpeg();
    // Remove last 100 bytes (likely scan data)
    if jpeg.len() > 200 {
        let truncated = &jpeg[..jpeg.len() - 100];
        let decoder = Decoder::new();
        // May succeed with partial data or fail - implementation dependent
        let _ = decoder.decode(truncated);
    }
}

#[test]
fn test_decode_truncated_before_eoi() {
    let jpeg = create_test_jpeg();
    // Remove just the EOI marker
    if jpeg.len() > 2 {
        let truncated = &jpeg[..jpeg.len() - 2];
        let decoder = Decoder::new();
        // May succeed (EOI optional) or fail
        let _ = decoder.decode(truncated);
    }
}

// ============================================================================
// Corrupted Marker Tests
// ============================================================================

#[test]
fn test_decode_corrupted_dqt_length() {
    let mut jpeg = create_test_jpeg();
    // Find DQT marker and corrupt length
    if let Some(pos) = jpeg.windows(2).position(|w| w == [0xFF, 0xDB]) {
        if pos + 4 < jpeg.len() {
            // Set impossibly large length
            jpeg[pos + 2] = 0xFF;
            jpeg[pos + 3] = 0xFF;
        }
    }
    let decoder = Decoder::new();
    let result = decoder.decode(&jpeg);
    assert!(result.is_err(), "Should reject corrupted DQT length");
}

#[test]
fn test_decode_corrupted_dht_length() {
    let mut jpeg = create_test_jpeg();
    // Find DHT marker and corrupt length
    if let Some(pos) = jpeg.windows(2).position(|w| w == [0xFF, 0xC4]) {
        if pos + 4 < jpeg.len() {
            jpeg[pos + 2] = 0xFF;
            jpeg[pos + 3] = 0xFF;
        }
    }
    let decoder = Decoder::new();
    let result = decoder.decode(&jpeg);
    assert!(result.is_err(), "Should reject corrupted DHT length");
}

#[test]
fn test_decode_corrupted_sof_length() {
    let mut jpeg = create_test_jpeg();
    // Find SOF0 marker and corrupt length
    if let Some(pos) = jpeg.windows(2).position(|w| w == [0xFF, 0xC0]) {
        if pos + 4 < jpeg.len() {
            jpeg[pos + 2] = 0x00;
            jpeg[pos + 3] = 0x01; // Length too small
        }
    }
    let decoder = Decoder::new();
    let result = decoder.decode(&jpeg);
    assert!(result.is_err(), "Should reject corrupted SOF length");
}

// ============================================================================
// Invalid SOF Parameter Tests
// ============================================================================

#[test]
fn test_decode_zero_width_in_sof() {
    let mut jpeg = create_test_jpeg();
    // Find SOF0 marker and set width to 0
    if let Some(pos) = jpeg.windows(2).position(|w| w == [0xFF, 0xC0]) {
        // SOF structure: length(2) + precision(1) + height(2) + width(2)
        if pos + 9 < jpeg.len() {
            jpeg[pos + 7] = 0x00; // Width MSB
            jpeg[pos + 8] = 0x00; // Width LSB
        }
    }
    let decoder = Decoder::new();
    let result = decoder.decode(&jpeg);
    // Implementation may accept or reject corrupted SOF - just ensure no panic
    // Some decoders may recover from this corruption
    let _ = result;
}

#[test]
fn test_decode_zero_height_in_sof() {
    let mut jpeg = create_test_jpeg();
    // Find SOF0 marker and set height to 0
    if let Some(pos) = jpeg.windows(2).position(|w| w == [0xFF, 0xC0]) {
        if pos + 7 < jpeg.len() {
            jpeg[pos + 5] = 0x00; // Height MSB
            jpeg[pos + 6] = 0x00; // Height LSB
        }
    }
    let decoder = Decoder::new();
    let result = decoder.decode(&jpeg);
    // Note: Some decoders handle DNL (Define Number of Lines) which allows height=0
    // so this may succeed or fail depending on implementation
    let _ = result;
}

// ============================================================================
// Invalid Component Tests
// ============================================================================

#[test]
fn test_decode_zero_components() {
    let mut jpeg = create_test_jpeg();
    // Find SOF0 and set component count to 0
    if let Some(pos) = jpeg.windows(2).position(|w| w == [0xFF, 0xC0]) {
        if pos + 9 < jpeg.len() {
            jpeg[pos + 9] = 0x00; // Num components = 0
        }
    }
    let decoder = Decoder::new();
    let result = decoder.decode(&jpeg);
    assert!(result.is_err(), "Should reject zero components");
}

#[test]
fn test_decode_too_many_components() {
    let mut jpeg = create_test_jpeg();
    // Find SOF0 and set component count to invalid value
    if let Some(pos) = jpeg.windows(2).position(|w| w == [0xFF, 0xC0]) {
        if pos + 9 < jpeg.len() {
            jpeg[pos + 9] = 0xFF; // 255 components (invalid)
        }
    }
    let decoder = Decoder::new();
    let result = decoder.decode(&jpeg);
    assert!(result.is_err(), "Should reject too many components");
}

// ============================================================================
// Restart Marker Tests
// ============================================================================

#[test]
fn test_decode_spurious_restart_markers() {
    let mut jpeg = create_test_jpeg();
    // Insert restart markers in the middle of scan data
    if let Some(sos_pos) = jpeg.windows(2).position(|w| w == [0xFF, 0xDA]) {
        // Find scan data start (after SOS header)
        let scan_start = sos_pos + 14; // Approximate SOS header size
        if scan_start + 50 < jpeg.len() {
            // Insert RST0 marker
            jpeg.insert(scan_start + 20, 0xD0);
            jpeg.insert(scan_start + 20, 0xFF);
        }
    }
    let decoder = Decoder::new();
    // This may succeed or fail depending on restart interval handling
    let _ = decoder.decode(&jpeg);
}

// ============================================================================
// Byte-Stuffing Tests
// ============================================================================

#[test]
fn test_decode_missing_stuffing_byte() {
    let mut jpeg = create_test_jpeg();
    // Find 0xFF 0x00 sequence and remove the 0x00
    for i in 0..jpeg.len() - 1 {
        if jpeg[i] == 0xFF && jpeg[i + 1] == 0x00 {
            jpeg.remove(i + 1);
            break;
        }
    }
    let decoder = Decoder::new();
    // May produce incorrect output or fail
    let _ = decoder.decode(&jpeg);
}

// ============================================================================
// Progressive Mode Error Tests
// ============================================================================

#[test]
fn test_decode_progressive_truncated() {
    let img = generate_gradient_d(64, 64, 3);
    let jpeg = encode_progressive(64, 64, &img.pixels).expect("encode failed");

    // Truncate progressive JPEG (missing later scans)
    let truncated = &jpeg[..jpeg.len() / 2];
    let decoder = Decoder::new();
    // Should fail or produce incomplete image
    let _ = decoder.decode(truncated);
}

// ============================================================================
// Memory Safety Tests
// ============================================================================

#[test]
fn test_decode_huge_dimensions_in_sof() {
    let mut jpeg = create_test_jpeg();
    // Find SOF0 and set huge dimensions
    if let Some(pos) = jpeg.windows(2).position(|w| w == [0xFF, 0xC0]) {
        if pos + 9 < jpeg.len() {
            // Set 65535x65535 dimensions
            jpeg[pos + 5] = 0xFF;
            jpeg[pos + 6] = 0xFF;
            jpeg[pos + 7] = 0xFF;
            jpeg[pos + 8] = 0xFF;
        }
    }
    let decoder = Decoder::new();
    // Should reject or handle gracefully without OOM
    let result = decoder.decode(&jpeg);
    assert!(result.is_err(), "Should reject impossibly large dimensions");
}

// ============================================================================
// Pixel Format Error Tests
// ============================================================================

#[test]
fn test_encode_wrong_pixel_format_data() {
    // Tell encoder it's grayscale but give RGB data
    let rgb_data = vec![128u8; 64 * 64 * 3];
    let result = encode_gray(64, 64, &rgb_data);
    // This should either fail or handle the mismatch
    // Behavior is implementation-defined
    let _ = result;
}

// ============================================================================
// Concurrent Access Tests (if applicable)
// ============================================================================

#[test]
fn test_encode_decode_concurrent() {
    use std::thread;

    let handles: Vec<_> = (0..4)
        .map(|i| {
            thread::spawn(move || {
                let size = 32 + i * 16;
                let img = generate_gradient_d(size, size, 3);
                let jpeg = encode_rgb(size, size, &img.pixels).expect("encode failed");

                let decoder = Decoder::new();
                let decoded = decoder.decode(&jpeg).expect("decode failed");
                assert_eq!(decoded.width, size);
                assert_eq!(decoded.height, size);
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("thread panicked");
    }
}

// ============================================================================
// Stress Tests
// ============================================================================

#[test]
fn test_encode_many_small_images() {
    for i in 0..100 {
        let size = 8 + (i % 24);
        let img = generate_gradient_d(size, size, 3);
        let jpeg = encode_rgb(size, size, &img.pixels).expect("encode failed");
        assert!(jpeg.len() > 50, "JPEG {} too small", i);
    }
}

#[test]
fn test_decode_many_images() {
    let decoder = Decoder::new();

    for i in 0..50 {
        let size = 16 + (i % 48);
        let img = generate_gradient_d(size, size, 3);
        let jpeg = encode_rgb(size, size, &img.pixels).expect("encode failed");

        let decoded = decoder.decode(&jpeg).expect("decode failed");
        assert_eq!(decoded.width, size, "Width mismatch on iteration {}", i);
        assert_eq!(decoded.height, size, "Height mismatch on iteration {}", i);
    }
}
