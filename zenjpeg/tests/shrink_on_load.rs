//! Integration tests for shrink-on-load JPEG decoding.
//!
//! Tests dimension correctness, quality, edge cases, and various subsampling modes.

use enough::Unstoppable;
use zenjpeg::decoder::{DctScale, Decoder, ShrinkHint};
use zenjpeg::encode::encoder_types::{ChromaSubsampling, PixelLayout};
use zenjpeg::encode::EncoderConfig;

// ============================================================================
// Test helpers
// ============================================================================

/// Create a test JPEG from RGB pixel data at given quality.
fn encode_test_jpeg(
    width: u32,
    height: u32,
    pixels: &[u8],
    quality: f32,
    subsampling: ChromaSubsampling,
) -> Vec<u8> {
    let config = EncoderConfig::ycbcr(quality, subsampling).progressive(false);
    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .expect("encode failed");
    enc.push_packed(pixels, Unstoppable).expect("push failed");
    enc.finish().expect("finish failed")
}

/// Create a test JPEG from grayscale data.
fn encode_gray_jpeg(width: u32, height: u32, pixels: &[u8], quality: f32) -> Vec<u8> {
    let config = EncoderConfig::grayscale(quality).progressive(false);
    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Gray8Srgb)
        .expect("encode failed");
    enc.push_packed(pixels, Unstoppable).expect("push failed");
    enc.finish().expect("finish failed")
}

/// Create progressive test JPEG.
fn encode_progressive_jpeg(
    width: u32,
    height: u32,
    pixels: &[u8],
    quality: f32,
    subsampling: ChromaSubsampling,
) -> Vec<u8> {
    let config = EncoderConfig::ycbcr(quality, subsampling).progressive(true);
    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .expect("encode failed");
    enc.push_packed(pixels, Unstoppable).expect("push failed");
    enc.finish().expect("finish failed")
}

/// Generate a smooth gradient test image (RGB).
fn gradient_rgb(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = vec![0u8; (width * height * 3) as usize];
    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 3) as usize;
            pixels[idx] = ((x * 255) / width.max(1)) as u8;
            pixels[idx + 1] = ((y * 255) / height.max(1)) as u8;
            pixels[idx + 2] = (((x + y) * 127) / (width + height).max(1)) as u8;
        }
    }
    pixels
}

/// Generate a smooth gradient test image (Gray).
fn gradient_gray(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = vec![0u8; (width * height) as usize];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            pixels[idx] = (((x + y) * 255) / (width + height).max(1)) as u8;
        }
    }
    pixels
}

// ============================================================================
// Dimension correctness tests
// ============================================================================

#[test]
fn shrink_dimensions_exact_444() {
    let pixels = gradient_rgb(256, 256);
    let jpeg = encode_test_jpeg(256, 256, &pixels, 90.0, ChromaSubsampling::None);

    for &scale in &DctScale::ALL {
        let result = Decoder::new()
            .shrink(ShrinkHint::ExactScale(scale))
            .decode(&jpeg, Unstoppable)
            .unwrap_or_else(|e| panic!("decode at {} failed: {e}", scale));

        let expected_w = scale.scaled_dimension(256);
        let expected_h = scale.scaled_dimension(256);

        assert_eq!(
            result.width(),
            expected_w,
            "width mismatch at {}: expected {expected_w}, got {}",
            scale,
            result.width(),
        );
        assert_eq!(
            result.height(),
            expected_h,
            "height mismatch at {}: expected {expected_h}, got {}",
            scale,
            result.height(),
        );
    }
}

#[test]
fn shrink_dimensions_exact_420() {
    let pixels = gradient_rgb(256, 256);
    let jpeg = encode_test_jpeg(256, 256, &pixels, 90.0, ChromaSubsampling::Quarter);

    for &scale in &DctScale::ALL {
        let result = Decoder::new()
            .shrink(ShrinkHint::ExactScale(scale))
            .decode(&jpeg, Unstoppable)
            .unwrap_or_else(|e| panic!("decode at {} failed: {e}", scale));

        let expected_w = scale.scaled_dimension(256);
        let expected_h = scale.scaled_dimension(256);

        assert_eq!(result.width(), expected_w, "width mismatch at {scale}");
        assert_eq!(result.height(), expected_h, "height mismatch at {scale}");
    }
}

#[test]
fn shrink_dimensions_non_aligned() {
    // 100x75: not 8-aligned, not MCU-aligned for 4:2:0
    let pixels = gradient_rgb(100, 75);
    let jpeg = encode_test_jpeg(100, 75, &pixels, 90.0, ChromaSubsampling::Quarter);

    for &scale in &DctScale::ALL {
        let result = Decoder::new()
            .shrink(ShrinkHint::ExactScale(scale))
            .decode(&jpeg, Unstoppable)
            .unwrap_or_else(|e| panic!("decode at {} failed: {e}", scale));

        let expected_w = scale.scaled_dimension(100);
        let expected_h = scale.scaled_dimension(75);

        assert_eq!(result.width(), expected_w, "width mismatch at {scale}");
        assert_eq!(result.height(), expected_h, "height mismatch at {scale}");
    }
}

#[test]
fn shrink_dimensions_grayscale() {
    let pixels = gradient_gray(128, 96);
    let jpeg = encode_gray_jpeg(128, 96, &pixels, 90.0);

    for &scale in &DctScale::ALL {
        let result = Decoder::new()
            .shrink(ShrinkHint::ExactScale(scale))
            .decode(&jpeg, Unstoppable)
            .unwrap_or_else(|e| panic!("grayscale at {} failed: {e}", scale));

        let expected_w = scale.scaled_dimension(128);
        let expected_h = scale.scaled_dimension(96);

        assert_eq!(result.width(), expected_w, "width mismatch at {scale}");
        assert_eq!(result.height(), expected_h, "height mismatch at {scale}");
    }
}

#[test]
fn shrink_fit_within_selects_correct_scale() {
    let pixels = gradient_rgb(800, 600);
    let jpeg = encode_test_jpeg(800, 600, &pixels, 90.0, ChromaSubsampling::None);

    // Request 100x75 → 1/8 gives 100x75, exact fit
    let result = Decoder::new()
        .shrink(ShrinkHint::FitWithin {
            width: 100,
            height: 75,
        })
        .decode(&jpeg, Unstoppable)
        .unwrap();
    assert_eq!(result.width(), DctScale::Eighth.scaled_dimension(800));
    assert_eq!(result.height(), DctScale::Eighth.scaled_dimension(600));

    // Request 201x151 → 1/4 gives 200x150 (too small!) → 1/2 gives 400x300
    let result = Decoder::new()
        .shrink(ShrinkHint::FitWithin {
            width: 201,
            height: 151,
        })
        .decode(&jpeg, Unstoppable)
        .unwrap();
    assert_eq!(result.width(), DctScale::Half.scaled_dimension(800));
    assert_eq!(result.height(), DctScale::Half.scaled_dimension(600));
}

#[test]
fn full_scale_unchanged() {
    // DctScale::Full should produce identical dimensions to no shrink
    let pixels = gradient_rgb(256, 192);
    let jpeg = encode_test_jpeg(256, 192, &pixels, 90.0, ChromaSubsampling::None);

    let ref_result = Decoder::new().decode(&jpeg, Unstoppable).unwrap();

    let shrunk = Decoder::new()
        .shrink(ShrinkHint::ExactScale(DctScale::Full))
        .decode(&jpeg, Unstoppable)
        .unwrap();

    assert_eq!(ref_result.width(), shrunk.width());
    assert_eq!(ref_result.height(), shrunk.height());
}

// ============================================================================
// Quality / pixel data tests
// ============================================================================

#[test]
fn shrink_full_matches_reference() {
    // Full scale shrink should produce nearly identical output to normal decode
    let pixels = gradient_rgb(64, 64);
    let jpeg = encode_test_jpeg(64, 64, &pixels, 95.0, ChromaSubsampling::None);

    let ref_result = Decoder::new().decode(&jpeg, Unstoppable).unwrap();
    let shrunk = Decoder::new()
        .shrink(ShrinkHint::ExactScale(DctScale::Full))
        .decode(&jpeg, Unstoppable)
        .unwrap();

    let ref_px = ref_result.pixels_u8().unwrap();
    let shrunk_px = shrunk.pixels_u8().unwrap();

    assert_eq!(ref_px.len(), shrunk_px.len(), "pixel buffer size mismatch");

    // Should be identical (same IDCT path, same dimensions)
    let max_diff = ref_px
        .iter()
        .zip(shrunk_px.iter())
        .map(|(a, b)| (*a as i32 - *b as i32).unsigned_abs())
        .max()
        .unwrap_or(0);

    assert_eq!(max_diff, 0, "Full scale should match reference exactly");
}

#[test]
fn shrink_half_produces_valid_pixels() {
    let pixels = gradient_rgb(128, 128);
    let jpeg = encode_test_jpeg(128, 128, &pixels, 95.0, ChromaSubsampling::None);

    let result = Decoder::new()
        .shrink(ShrinkHint::ExactScale(DctScale::Half))
        .decode(&jpeg, Unstoppable)
        .unwrap();

    let px = result.pixels_u8().unwrap();
    let expected_w = DctScale::Half.scaled_dimension(128) as usize;
    let expected_h = DctScale::Half.scaled_dimension(128) as usize;

    assert_eq!(
        px.len(),
        expected_w * expected_h * 3,
        "buffer size mismatch"
    );

    // All pixels should be valid (not zero-filled artifacts)
    // In a gradient image, most pixels should be non-zero
    let nonzero = px.iter().filter(|&&v| v > 0).count();
    assert!(
        nonzero > px.len() / 2,
        "too many zero pixels: {nonzero}/{} — likely broken IDCT",
        px.len(),
    );
}

#[test]
fn shrink_quarter_produces_valid_pixels() {
    let pixels = gradient_rgb(128, 128);
    let jpeg = encode_test_jpeg(128, 128, &pixels, 95.0, ChromaSubsampling::None);

    let result = Decoder::new()
        .shrink(ShrinkHint::ExactScale(DctScale::Quarter))
        .decode(&jpeg, Unstoppable)
        .unwrap();

    let px = result.pixels_u8().unwrap();
    let expected_w = DctScale::Quarter.scaled_dimension(128) as usize;
    let expected_h = DctScale::Quarter.scaled_dimension(128) as usize;
    assert_eq!(px.len(), expected_w * expected_h * 3);

    let nonzero = px.iter().filter(|&&v| v > 0).count();
    assert!(nonzero > px.len() / 2, "too many zero pixels");
}

#[test]
fn shrink_eighth_produces_valid_pixels() {
    let pixels = gradient_rgb(128, 128);
    let jpeg = encode_test_jpeg(128, 128, &pixels, 95.0, ChromaSubsampling::None);

    let result = Decoder::new()
        .shrink(ShrinkHint::ExactScale(DctScale::Eighth))
        .decode(&jpeg, Unstoppable)
        .unwrap();

    let px = result.pixels_u8().unwrap();
    let expected_w = DctScale::Eighth.scaled_dimension(128) as usize;
    let expected_h = DctScale::Eighth.scaled_dimension(128) as usize;
    assert_eq!(px.len(), expected_w * expected_h * 3);

    let nonzero = px.iter().filter(|&&v| v > 0).count();
    assert!(nonzero > px.len() / 2, "too many zero pixels");
}

// ============================================================================
// Subsampling variants
// ============================================================================

#[test]
fn shrink_422_all_scales() {
    let pixels = gradient_rgb(256, 256);
    let jpeg = encode_test_jpeg(256, 256, &pixels, 90.0, ChromaSubsampling::HalfHorizontal);

    for &scale in &DctScale::ALL {
        let result = Decoder::new()
            .shrink(ShrinkHint::ExactScale(scale))
            .decode(&jpeg, Unstoppable)
            .unwrap_or_else(|e| panic!("4:2:2 at {} failed: {e}", scale));

        assert_eq!(result.width(), scale.scaled_dimension(256));
        assert_eq!(result.height(), scale.scaled_dimension(256));
        assert!(result.pixels_u8().unwrap().len() > 0);
    }
}

#[test]
fn shrink_420_all_scales() {
    let pixels = gradient_rgb(256, 256);
    let jpeg = encode_test_jpeg(256, 256, &pixels, 90.0, ChromaSubsampling::Quarter);

    for &scale in &DctScale::ALL {
        let result = Decoder::new()
            .shrink(ShrinkHint::ExactScale(scale))
            .decode(&jpeg, Unstoppable)
            .unwrap_or_else(|e| panic!("4:2:0 at {} failed: {e}", scale));

        assert_eq!(result.width(), scale.scaled_dimension(256));
        assert_eq!(result.height(), scale.scaled_dimension(256));
        assert!(result.pixels_u8().unwrap().len() > 0);
    }
}

// ============================================================================
// Progressive JPEG
// ============================================================================

#[test]
fn shrink_progressive_all_scales() {
    let pixels = gradient_rgb(128, 128);
    let jpeg = encode_progressive_jpeg(128, 128, &pixels, 90.0, ChromaSubsampling::Quarter);

    for &scale in &DctScale::ALL {
        let result = Decoder::new()
            .shrink(ShrinkHint::ExactScale(scale))
            .decode(&jpeg, Unstoppable)
            .unwrap_or_else(|e| panic!("progressive at {} failed: {e}", scale));

        let expected_w = scale.scaled_dimension(128);
        let expected_h = scale.scaled_dimension(128);

        assert_eq!(result.width(), expected_w, "progressive width at {scale}");
        assert_eq!(result.height(), expected_h, "progressive height at {scale}");
    }
}

// ============================================================================
// Scanline reader API
// ============================================================================

#[test]
fn scanline_reader_shrink_dimensions() {
    let pixels = gradient_rgb(256, 256);
    let jpeg = encode_test_jpeg(256, 256, &pixels, 90.0, ChromaSubsampling::None);

    let reader = Decoder::new()
        .shrink(ShrinkHint::ExactScale(DctScale::Half))
        .scanline_reader(&jpeg)
        .unwrap();

    assert_eq!(reader.width(), DctScale::Half.scaled_dimension(256));
    assert_eq!(reader.height(), DctScale::Half.scaled_dimension(256));
}

#[test]
fn scanline_reader_shrink_read_all_rows() {
    let pixels = gradient_rgb(128, 128);
    let jpeg = encode_test_jpeg(128, 128, &pixels, 90.0, ChromaSubsampling::None);

    let mut reader = Decoder::new()
        .shrink(ShrinkHint::ExactScale(DctScale::Quarter))
        .scanline_reader(&jpeg)
        .unwrap();

    let width = reader.width() as usize;
    let height = reader.height() as usize;

    let mut output = vec![0u8; width * height * 3];
    let mut rows_read = 0;

    while rows_read < height {
        let remaining = height - rows_read;
        let out =
            imgref::ImgRefMut::new(&mut output[rows_read * width * 3..], width * 3, remaining);
        let count = reader.read_rows_rgb8(out).unwrap();
        if count == 0 {
            break;
        }
        rows_read += count;
    }

    assert_eq!(rows_read, height, "didn't read all rows");
}

// ============================================================================
// Edge cases
// ============================================================================

#[test]
fn shrink_small_image() {
    // 8x8: single MCU. At 1/8, produces 1x1.
    let pixels = gradient_rgb(8, 8);
    let jpeg = encode_test_jpeg(8, 8, &pixels, 95.0, ChromaSubsampling::None);

    let result = Decoder::new()
        .shrink(ShrinkHint::ExactScale(DctScale::Eighth))
        .decode(&jpeg, Unstoppable)
        .unwrap();

    assert_eq!(result.width(), 1);
    assert_eq!(result.height(), 1);
    assert_eq!(result.pixels_u8().unwrap().len(), 3); // 1x1 RGB
}

#[test]
fn shrink_tall_narrow_image() {
    // 16x128 — skinny tall image
    let pixels = gradient_rgb(16, 128);
    let jpeg = encode_test_jpeg(16, 128, &pixels, 90.0, ChromaSubsampling::None);

    let result = Decoder::new()
        .shrink(ShrinkHint::ExactScale(DctScale::Quarter))
        .decode(&jpeg, Unstoppable)
        .unwrap();

    assert_eq!(result.width(), DctScale::Quarter.scaled_dimension(16));
    assert_eq!(result.height(), DctScale::Quarter.scaled_dimension(128));
}

#[test]
fn shrink_wide_short_image() {
    // 256x16 — wide short image
    let pixels = gradient_rgb(256, 16);
    let jpeg = encode_test_jpeg(256, 16, &pixels, 90.0, ChromaSubsampling::None);

    let result = Decoder::new()
        .shrink(ShrinkHint::ExactScale(DctScale::Half))
        .decode(&jpeg, Unstoppable)
        .unwrap();

    assert_eq!(result.width(), DctScale::Half.scaled_dimension(256));
    assert_eq!(result.height(), DctScale::Half.scaled_dimension(16));
}

#[test]
fn shrink_various_non_mcu_aligned_sizes() {
    // Test multiple non-aligned sizes
    for (w, h) in [(100, 75), (33, 17), (1, 1), (7, 9), (255, 127)] {
        let pixels = gradient_rgb(w, h);
        let jpeg = encode_test_jpeg(w, h, &pixels, 90.0, ChromaSubsampling::None);

        for &scale in &DctScale::ALL {
            let result = Decoder::new()
                .shrink(ShrinkHint::ExactScale(scale))
                .decode(&jpeg, Unstoppable)
                .unwrap_or_else(|e| panic!("{w}x{h} at {scale} failed: {e}"));

            assert_eq!(
                result.width(),
                scale.scaled_dimension(w),
                "{w}x{h} at {scale}",
            );
            assert_eq!(
                result.height(),
                scale.scaled_dimension(h),
                "{w}x{h} at {scale}",
            );
        }
    }
}

// ============================================================================
// JpegInfo.available_scales
// ============================================================================

#[test]
fn jpeg_info_available_scales() {
    let pixels = gradient_rgb(640, 480);
    let jpeg = encode_test_jpeg(640, 480, &pixels, 90.0, ChromaSubsampling::None);

    let info = Decoder::new().read_info(&jpeg).unwrap();

    assert_eq!(info.available_scales.len(), 4);

    let (scale, dims) = info.available_scales[0];
    assert_eq!(scale, DctScale::Eighth);
    assert_eq!(dims.width, 80);
    assert_eq!(dims.height, 60);

    let (scale, dims) = info.available_scales[1];
    assert_eq!(scale, DctScale::Quarter);
    assert_eq!(dims.width, 160);
    assert_eq!(dims.height, 120);

    let (scale, dims) = info.available_scales[2];
    assert_eq!(scale, DctScale::Half);
    assert_eq!(dims.width, 320);
    assert_eq!(dims.height, 240);

    let (scale, dims) = info.available_scales[3];
    assert_eq!(scale, DctScale::Full);
    assert_eq!(dims.width, 640);
    assert_eq!(dims.height, 480);
}
