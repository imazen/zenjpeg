//! Tests for the YCbCr f32 streaming API.
//!
//! These tests verify the decoder's `decode_to_ycbcr_f32()` method and
//! the encoder's `push_ycbcr_strip_f32()` methods.

use jpegli::{Decoder, JpegEncoder, PixelFormat, Quality, Subsampling};

/// Helper to create a test RGB image with a gradient pattern.
fn create_test_rgb(width: usize, height: usize) -> Vec<u8> {
    let mut data = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            data[idx] = ((x * 255) / width.max(1)) as u8; // R
            data[idx + 1] = ((y * 255) / height.max(1)) as u8; // G
            data[idx + 2] = 128; // B
        }
    }
    data
}

/// Helper to convert RGB to YCbCr in centered range [-128, 127].
fn rgb_to_ycbcr_centered(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let r = r as f32;
    let g = g as f32;
    let b = b as f32;

    // Standard JPEG YCbCr conversion coefficients
    let y = 0.299 * r + 0.587 * g + 0.114 * b;
    let cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0;
    let cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0;

    // Return in centered range (subtract 128 from all)
    (y - 128.0, cb - 128.0, cr - 128.0)
}

/// Test that decode_to_ycbcr_f32 returns valid YCbCr planes.
#[test]
fn test_decode_to_ycbcr_f32_basic() {
    let width = 64;
    let height = 64;

    // Create and encode a test image
    let rgb = create_test_rgb(width, height);
    let jpeg = JpegEncoder::new(width as u32, height as u32)
        .pixel_format(PixelFormat::Rgb)
        .quality(Quality::from_quality(90.0))
        .subsampling(Subsampling::S444)
        .encode(&rgb)
        .unwrap();

    // Decode to YCbCr f32
    let decoder = Decoder::new();
    let ycbcr = decoder.decode_to_ycbcr_f32(&jpeg).unwrap();

    // Verify dimensions
    assert_eq!(ycbcr.width, width as u32);
    assert_eq!(ycbcr.height, height as u32);
    assert_eq!(ycbcr.y.len(), width * height);
    assert_eq!(ycbcr.cb.len(), width * height);
    assert_eq!(ycbcr.cr.len(), width * height);

    // Verify values are in expected range (roughly [-128, 127])
    let y_min = ycbcr.y.iter().cloned().fold(f32::INFINITY, f32::min);
    let y_max = ycbcr.y.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    assert!(y_min >= -130.0, "Y min {} out of range", y_min);
    assert!(y_max <= 130.0, "Y max {} out of range", y_max);

    let cb_min = ycbcr.cb.iter().cloned().fold(f32::INFINITY, f32::min);
    let cb_max = ycbcr.cb.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    assert!(cb_min >= -130.0, "Cb min {} out of range", cb_min);
    assert!(cb_max <= 130.0, "Cb max {} out of range", cb_max);
}

/// Test that encoding from YCbCr produces the same result as encoding from RGB.
#[test]
fn test_encode_ycbcr_parity_with_rgb() {
    let width = 64;
    let height = 64;

    // Create test RGB data
    let rgb = create_test_rgb(width, height);

    // Convert to YCbCr planes (centered range)
    let mut y_plane = vec![0.0f32; width * height];
    let mut cb_plane = vec![0.0f32; width * height];
    let mut cr_plane = vec![0.0f32; width * height];

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let rgb_idx = idx * 3;
            let (y_val, cb_val, cr_val) =
                rgb_to_ycbcr_centered(rgb[rgb_idx], rgb[rgb_idx + 1], rgb[rgb_idx + 2]);
            y_plane[idx] = y_val;
            cb_plane[idx] = cb_val;
            cr_plane[idx] = cr_val;
        }
    }

    // Encode from RGB
    let jpeg_rgb = JpegEncoder::new(width as u32, height as u32)
        .pixel_format(PixelFormat::Rgb)
        .quality(Quality::from_quality(90.0))
        .subsampling(Subsampling::S444)
        .encode(&rgb)
        .unwrap();

    // Encode from YCbCr
    let strip_height = 16; // Typical strip height
    let mut encoder_ycbcr = JpegEncoder::new(width as u32, height as u32)
        .pixel_format(PixelFormat::Rgb) // Still need to specify format for other params
        .quality(Quality::from_quality(90.0))
        .subsampling(Subsampling::S444)
        .start()
        .unwrap();

    // Push strips
    for strip_y in (0..height).step_by(strip_height) {
        let strip_rows = strip_height.min(height - strip_y);
        let start = strip_y * width;
        let end = start + strip_rows * width;

        encoder_ycbcr
            .push_ycbcr_strip_f32(
                &y_plane[start..end],
                &cb_plane[start..end],
                &cr_plane[start..end],
                strip_rows,
            )
            .unwrap();
    }

    let jpeg_ycbcr = encoder_ycbcr.finish().unwrap();

    // Decode both and compare
    let decoder = Decoder::new().output_format(PixelFormat::Rgb);
    let decoded_rgb = decoder.decode(&jpeg_rgb).unwrap();
    let decoded_ycbcr = decoder.decode(&jpeg_ycbcr).unwrap();

    // Should be similar (not identical due to floating-point differences)
    let mut max_diff = 0u8;
    for i in 0..decoded_rgb.data.len() {
        let diff = (decoded_rgb.data[i] as i16 - decoded_ycbcr.data[i] as i16).unsigned_abs() as u8;
        max_diff = max_diff.max(diff);
    }

    // Allow some difference due to FP precision
    assert!(
        max_diff <= 2,
        "RGB and YCbCr paths differ too much: max_diff={}",
        max_diff
    );
}

/// Test decode_to_ycbcr_f32 with subsampled image (4:2:0).
#[test]
fn test_decode_to_ycbcr_f32_420() {
    let width = 64;
    let height = 64;

    // Create and encode a test image with 4:2:0
    let rgb = create_test_rgb(width, height);
    let jpeg = JpegEncoder::new(width as u32, height as u32)
        .pixel_format(PixelFormat::Rgb)
        .quality(Quality::from_quality(90.0))
        .subsampling(Subsampling::S420)
        .encode(&rgb)
        .unwrap();

    // Decode to YCbCr f32
    let decoder = Decoder::new();
    let ycbcr = decoder.decode_to_ycbcr_f32(&jpeg).unwrap();

    // Chroma should be upsampled to full resolution
    assert_eq!(ycbcr.y.len(), width * height);
    assert_eq!(ycbcr.cb.len(), width * height);
    assert_eq!(ycbcr.cr.len(), width * height);
}

/// Test that ICC profile is passed through.
#[test]
fn test_decode_to_ycbcr_f32_icc_passthrough() {
    let width = 64;
    let height = 64;

    // Create and encode a test image
    let rgb = create_test_rgb(width, height);
    let jpeg = JpegEncoder::new(width as u32, height as u32)
        .pixel_format(PixelFormat::Rgb)
        .quality(Quality::from_quality(90.0))
        .encode(&rgb)
        .unwrap();

    // Decode to YCbCr f32
    let decoder = Decoder::new();
    let ycbcr = decoder.decode_to_ycbcr_f32(&jpeg).unwrap();

    // Standard JPEG without embedded ICC should have None
    // (This test mainly verifies the field exists and doesn't crash)
    // In practice, most JPEGs don't have ICC profiles
    let _ = ycbcr.icc_profile;
}

/// Test error handling for grayscale images.
#[test]
fn test_decode_to_ycbcr_f32_grayscale_error() {
    let width = 64;
    let height = 64;

    // Create and encode a grayscale image
    let gray: Vec<u8> = (0..width * height).map(|i| (i % 256) as u8).collect();
    let jpeg = JpegEncoder::new(width as u32, height as u32)
        .pixel_format(PixelFormat::Gray)
        .quality(Quality::from_quality(90.0))
        .encode(&gray)
        .unwrap();

    // Decode to YCbCr f32 should fail for grayscale
    let decoder = Decoder::new();
    let result = decoder.decode_to_ycbcr_f32(&jpeg);
    assert!(result.is_err(), "Should fail for grayscale images");
}

/// Test roundtrip: encode RGB -> decode YCbCr -> re-encode YCbCr -> decode RGB.
#[test]
fn test_ycbcr_f32_roundtrip() {
    let width = 64;
    let height = 64;

    // Create original RGB
    let original_rgb = create_test_rgb(width, height);

    // Encode at high quality
    let jpeg1 = JpegEncoder::new(width as u32, height as u32)
        .pixel_format(PixelFormat::Rgb)
        .quality(Quality::from_quality(95.0))
        .subsampling(Subsampling::S444)
        .encode(&original_rgb)
        .unwrap();

    // Decode to YCbCr
    let decoder = Decoder::new();
    let ycbcr = decoder.decode_to_ycbcr_f32(&jpeg1).unwrap();

    // Re-encode from YCbCr
    let strip_height = 16;
    let mut encoder = JpegEncoder::new(width as u32, height as u32)
        .pixel_format(PixelFormat::Rgb)
        .quality(Quality::from_quality(95.0))
        .subsampling(Subsampling::S444)
        .start()
        .unwrap();

    for strip_y in (0..height).step_by(strip_height) {
        let strip_rows = strip_height.min(height - strip_y);
        let start = strip_y * width;
        let end = start + strip_rows * width;

        encoder
            .push_ycbcr_strip_f32(
                &ycbcr.y[start..end],
                &ycbcr.cb[start..end],
                &ycbcr.cr[start..end],
                strip_rows,
            )
            .unwrap();
    }

    let jpeg2 = encoder.finish().unwrap();

    // Decode final result
    let final_decoded = decoder
        .output_format(PixelFormat::Rgb)
        .decode(&jpeg2)
        .unwrap();

    // Should be visually similar to original (compression is lossy)
    let mut max_diff = 0u8;
    for i in 0..original_rgb.len() {
        let diff = (original_rgb[i] as i16 - final_decoded.data[i] as i16).unsigned_abs() as u8;
        max_diff = max_diff.max(diff);
    }

    // Two rounds of compression will have more loss
    assert!(
        max_diff <= 15,
        "Roundtrip quality loss too high: max_diff={}",
        max_diff
    );
}
