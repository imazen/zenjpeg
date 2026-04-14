//! Tests for the push-based callback decode API (decode_rows / decode_rows_f32).

use crate::test_utils::generate_gradient_d;
use enough::Unstoppable;
use zenjpeg::{
    decoder::{Decoder, PixelFormat},
    encoder::{ChromaSubsampling, EncoderConfig, PixelLayout},
};

// ============================================================================
// Helpers
// ============================================================================

fn encode_rgb(w: u32, h: u32, quality: f32, sub: ChromaSubsampling) -> Vec<u8> {
    let img = generate_gradient_d(w, h, 3);
    let config = EncoderConfig::ycbcr(quality, sub).progressive(false);
    let mut enc = config
        .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
        .unwrap();
    enc.push_packed(&img.pixels, Unstoppable).unwrap();
    enc.finish().unwrap()
}

fn encode_rgb_progressive(w: u32, h: u32, quality: f32) -> Vec<u8> {
    let img = generate_gradient_d(w, h, 3);
    let config = EncoderConfig::ycbcr(quality, ChromaSubsampling::Quarter).progressive(true);
    let mut enc = config
        .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
        .unwrap();
    enc.push_packed(&img.pixels, Unstoppable).unwrap();
    enc.finish().unwrap()
}

fn encode_gray(w: u32, h: u32, quality: f32) -> Vec<u8> {
    let img = generate_gradient_d(w, h, 1);
    let config = EncoderConfig::grayscale(quality);
    let mut enc = config
        .encode_from_bytes(w, h, PixelLayout::Gray8Srgb)
        .unwrap();
    enc.push_packed(&img.pixels, Unstoppable).unwrap();
    enc.finish().unwrap()
}

// ============================================================================
// decode_rows: byte-identical to decode()
// ============================================================================

#[test]
fn callback_rgb_matches_decode() {
    let jpeg = encode_rgb(64, 48, 90.0, ChromaSubsampling::None);

    // Full decode
    let expected = Decoder::new()
        .output_format(PixelFormat::Rgb)
        .decode(&jpeg, Unstoppable)
        .unwrap();
    let expected_pixels = expected.pixels_u8().unwrap();

    // Callback decode
    let mut callback_pixels = Vec::new();
    let info = Decoder::new()
        .decode_rows(
            &jpeg,
            PixelFormat::Rgb,
            |row| {
                assert_eq!(row.width(), 64);
                assert_eq!(row.format(), PixelFormat::Rgb);
                assert_eq!(row.as_bytes().len(), 64 * 3);
                callback_pixels.extend_from_slice(row.as_bytes());
                Ok(())
            },
            Unstoppable,
        )
        .unwrap();

    assert_eq!(info.dimensions.width, 64);
    assert_eq!(info.dimensions.height, 48);
    assert_eq!(callback_pixels.len(), expected_pixels.len());
    assert_eq!(callback_pixels, expected_pixels);
}

#[test]
fn callback_rgba_matches_decode() {
    let jpeg = encode_rgb(32, 32, 85.0, ChromaSubsampling::Quarter);

    let expected = Decoder::new()
        .output_format(PixelFormat::Rgba)
        .decode(&jpeg, Unstoppable)
        .unwrap();
    let expected_pixels = expected.pixels_u8().unwrap();

    let mut callback_pixels = Vec::new();
    Decoder::new()
        .decode_rows(
            &jpeg,
            PixelFormat::Rgba,
            |row| {
                assert_eq!(row.format(), PixelFormat::Rgba);
                assert_eq!(row.as_bytes().len(), 32 * 4);
                callback_pixels.extend_from_slice(row.as_bytes());
                Ok(())
            },
            Unstoppable,
        )
        .unwrap();

    assert_eq!(callback_pixels.len(), expected_pixels.len());
    assert_eq!(callback_pixels, expected_pixels);
}

#[test]
fn callback_bgr_matches_decode() {
    let jpeg = encode_rgb(48, 32, 90.0, ChromaSubsampling::None);

    let expected = Decoder::new()
        .output_format(PixelFormat::Bgr)
        .decode(&jpeg, Unstoppable)
        .unwrap();
    let expected_pixels = expected.pixels_u8().unwrap();

    let mut callback_pixels = Vec::new();
    Decoder::new()
        .decode_rows(
            &jpeg,
            PixelFormat::Bgr,
            |row| {
                callback_pixels.extend_from_slice(row.as_bytes());
                Ok(())
            },
            Unstoppable,
        )
        .unwrap();

    assert_eq!(callback_pixels, expected_pixels);
}

#[test]
fn callback_bgra_matches_decode() {
    let jpeg = encode_rgb(32, 32, 90.0, ChromaSubsampling::Quarter);

    let expected = Decoder::new()
        .output_format(PixelFormat::Bgra)
        .decode(&jpeg, Unstoppable)
        .unwrap();
    let expected_pixels = expected.pixels_u8().unwrap();

    let mut callback_pixels = Vec::new();
    Decoder::new()
        .decode_rows(
            &jpeg,
            PixelFormat::Bgra,
            |row| {
                callback_pixels.extend_from_slice(row.as_bytes());
                Ok(())
            },
            Unstoppable,
        )
        .unwrap();

    assert_eq!(callback_pixels, expected_pixels);
}

#[test]
fn callback_bgrx_matches_decode() {
    let jpeg = encode_rgb(32, 32, 90.0, ChromaSubsampling::None);

    let expected = Decoder::new()
        .output_format(PixelFormat::Bgrx)
        .decode(&jpeg, Unstoppable)
        .unwrap();
    let expected_pixels = expected.pixels_u8().unwrap();

    let mut callback_pixels = Vec::new();
    Decoder::new()
        .decode_rows(
            &jpeg,
            PixelFormat::Bgrx,
            |row| {
                callback_pixels.extend_from_slice(row.as_bytes());
                Ok(())
            },
            Unstoppable,
        )
        .unwrap();

    assert_eq!(callback_pixels, expected_pixels);
}

#[test]
fn callback_gray_matches_decode() {
    let jpeg = encode_gray(64, 64, 90.0);

    let expected = Decoder::new()
        .output_format(PixelFormat::Gray)
        .decode(&jpeg, Unstoppable)
        .unwrap();
    let expected_pixels = expected.pixels_u8().unwrap();

    let mut callback_pixels = Vec::new();
    let info = Decoder::new()
        .decode_rows(
            &jpeg,
            PixelFormat::Gray,
            |row| {
                assert_eq!(row.format(), PixelFormat::Gray);
                assert_eq!(row.as_gray().len(), 64);
                callback_pixels.extend_from_slice(row.as_bytes());
                Ok(())
            },
            Unstoppable,
        )
        .unwrap();

    assert_eq!(info.dimensions.width, 64);
    assert_eq!(info.dimensions.height, 64);
    assert_eq!(callback_pixels.len(), expected_pixels.len());
    assert_eq!(callback_pixels, expected_pixels);
}

// ============================================================================
// Progressive JPEG (buffered mode internally)
// ============================================================================

#[test]
fn callback_progressive_matches_decode() {
    let jpeg = encode_rgb_progressive(64, 48, 85.0);

    let expected = Decoder::new()
        .output_format(PixelFormat::Rgb)
        .decode(&jpeg, Unstoppable)
        .unwrap();
    let expected_pixels = expected.pixels_u8().unwrap();

    let mut callback_pixels = Vec::new();
    Decoder::new()
        .decode_rows(
            &jpeg,
            PixelFormat::Rgb,
            |row| {
                callback_pixels.extend_from_slice(row.as_bytes());
                Ok(())
            },
            Unstoppable,
        )
        .unwrap();

    assert_eq!(callback_pixels, expected_pixels);
}

// ============================================================================
// Row index tracking
// ============================================================================

#[test]
fn callback_row_indices_sequential() {
    let jpeg = encode_rgb(32, 24, 90.0, ChromaSubsampling::None);

    let mut indices = Vec::new();
    Decoder::new()
        .decode_rows(
            &jpeg,
            PixelFormat::Rgb,
            |row| {
                indices.push(row.row_index());
                Ok(())
            },
            Unstoppable,
        )
        .unwrap();

    let expected: Vec<usize> = (0..24).collect();
    assert_eq!(indices, expected);
}

// ============================================================================
// Early abort
// ============================================================================

#[test]
fn callback_early_abort() {
    let jpeg = encode_rgb(32, 100, 90.0, ChromaSubsampling::None);

    let mut rows_received = 0;
    let result = Decoder::new().decode_rows(
        &jpeg,
        PixelFormat::Rgb,
        |row| {
            rows_received += 1;
            if row.row_index() >= 9 {
                Err(zenjpeg::decoder::Error::internal("abort"))
            } else {
                Ok(())
            }
        },
        Unstoppable,
    );

    assert!(result.is_err());
    assert_eq!(rows_received, 10); // rows 0..=9
}

// ============================================================================
// Typed accessors
// ============================================================================

#[test]
fn callback_as_rgb_accessor() {
    let jpeg = encode_rgb(8, 8, 95.0, ChromaSubsampling::None);

    Decoder::new()
        .decode_rows(
            &jpeg,
            PixelFormat::Rgb,
            |row| {
                let rgb = row.as_rgb();
                assert_eq!(rgb.len(), 8);
                // Each pixel should have reasonable values
                // Verify we got valid pixels (non-zero for non-black image)
                assert!(!rgb.is_empty());
                Ok(())
            },
            Unstoppable,
        )
        .unwrap();
}

#[test]
fn callback_as_rgba_accessor() {
    let jpeg = encode_rgb(8, 8, 95.0, ChromaSubsampling::None);

    Decoder::new()
        .decode_rows(
            &jpeg,
            PixelFormat::Rgba,
            |row| {
                let rgba = row.as_rgba();
                assert_eq!(rgba.len(), 8);
                // Alpha should be 255 (opaque)
                for px in rgba {
                    assert_eq!(px.a, 255);
                }
                Ok(())
            },
            Unstoppable,
        )
        .unwrap();
}

#[test]
#[should_panic(expected = "as_rgb() called on")]
fn callback_as_rgb_wrong_format_panics() {
    let jpeg = encode_gray(8, 8, 95.0);

    Decoder::new()
        .decode_rows(
            &jpeg,
            PixelFormat::Gray,
            |row| {
                let _ = row.as_rgb(); // Should panic
                Ok(())
            },
            Unstoppable,
        )
        .unwrap();
}

// ============================================================================
// decode_rows_f32
// ============================================================================

#[test]
fn callback_f32_rgba_basic() {
    let jpeg = encode_rgb(32, 32, 90.0, ChromaSubsampling::None);

    let mut row_count = 0;
    let mut total_floats = 0;
    let info = Decoder::new()
        .decode_rows_f32(
            &jpeg,
            PixelFormat::RgbaF32,
            |row| {
                assert_eq!(row.width(), 32);
                assert_eq!(row.format(), PixelFormat::RgbaF32);
                let data = row.as_slice();
                assert_eq!(data.len(), 32 * 4);
                // Verify values are in reasonable range (RGBA: RGB in ~[0,1], A=1.0)
                for chunk in data.chunks_exact(4) {
                    for &v in &chunk[..3] {
                        assert!(
                            (-0.1..=1.1).contains(&v),
                            "RGB value {} out of range at row {}",
                            v,
                            row.row_index()
                        );
                    }
                    assert!(
                        (chunk[3] - 1.0).abs() < 0.01,
                        "alpha {} != 1.0 at row {}",
                        chunk[3],
                        row.row_index()
                    );
                }
                total_floats += data.len();
                row_count += 1;
                Ok(())
            },
            Unstoppable,
        )
        .unwrap();

    assert_eq!(info.dimensions.width, 32);
    assert_eq!(info.dimensions.height, 32);
    assert_eq!(row_count, 32);
    assert_eq!(total_floats, 32 * 32 * 4);
}

#[test]
fn callback_f32_gray_basic() {
    let jpeg = encode_gray(32, 32, 90.0);

    let mut row_count = 0;
    let info = Decoder::new()
        .decode_rows_f32(
            &jpeg,
            PixelFormat::GrayF32,
            |row| {
                assert_eq!(row.width(), 32);
                assert_eq!(row.format(), PixelFormat::GrayF32);
                let data = row.as_slice();
                assert_eq!(data.len(), 32);
                // Verify values are in reasonable range
                for &v in data {
                    assert!(
                        (-0.1..=1.1).contains(&v),
                        "gray value {} out of range at row {}",
                        v,
                        row.row_index()
                    );
                }
                row_count += 1;
                Ok(())
            },
            Unstoppable,
        )
        .unwrap();

    assert_eq!(info.dimensions.width, 32);
    assert_eq!(info.dimensions.height, 32);
    assert_eq!(row_count, 32);
}

// ============================================================================
// Error handling: unsupported formats
// ============================================================================

#[test]
fn callback_rejects_f32_format() {
    let jpeg = encode_rgb(8, 8, 90.0, ChromaSubsampling::None);

    let result = Decoder::new().decode_rows(&jpeg, PixelFormat::RgbaF32, |_| Ok(()), Unstoppable);
    assert!(result.is_err());
}

#[test]
fn callback_f32_rejects_u8_format() {
    let jpeg = encode_rgb(8, 8, 90.0, ChromaSubsampling::None);

    let result = Decoder::new().decode_rows_f32(&jpeg, PixelFormat::Rgb, |_| Ok(()), Unstoppable);
    assert!(result.is_err());
}

// ============================================================================
// Non-MCU-aligned dimensions
// ============================================================================

#[test]
fn callback_non_mcu_aligned() {
    // 13x11 — not 8-aligned
    let jpeg = encode_rgb(13, 11, 90.0, ChromaSubsampling::None);

    let expected = Decoder::new()
        .output_format(PixelFormat::Rgb)
        .decode(&jpeg, Unstoppable)
        .unwrap();
    let expected_pixels = expected.pixels_u8().unwrap();

    let mut callback_pixels = Vec::new();
    let info = Decoder::new()
        .decode_rows(
            &jpeg,
            PixelFormat::Rgb,
            |row| {
                assert_eq!(row.width(), 13);
                callback_pixels.extend_from_slice(row.as_bytes());
                Ok(())
            },
            Unstoppable,
        )
        .unwrap();

    assert_eq!(info.dimensions.width, 13);
    assert_eq!(info.dimensions.height, 11);
    assert_eq!(callback_pixels, expected_pixels);
}

// ============================================================================
// 4:2:0 subsampling
// ============================================================================

#[test]
fn callback_420_matches_decode() {
    let jpeg = encode_rgb(64, 64, 85.0, ChromaSubsampling::Quarter);

    let expected = Decoder::new()
        .output_format(PixelFormat::Rgb)
        .decode(&jpeg, Unstoppable)
        .unwrap();
    let expected_pixels = expected.pixels_u8().unwrap();

    let mut callback_pixels = Vec::new();
    Decoder::new()
        .decode_rows(
            &jpeg,
            PixelFormat::Rgb,
            |row| {
                callback_pixels.extend_from_slice(row.as_bytes());
                Ok(())
            },
            Unstoppable,
        )
        .unwrap();

    assert_eq!(callback_pixels, expected_pixels);
}
