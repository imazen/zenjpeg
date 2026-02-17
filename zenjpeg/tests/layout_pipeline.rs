//! Integration tests for the layout pipeline module.
//!
//! Tests lossless path (DCT-domain transforms), lossy path (decode→resize→encode),
//! metadata preservation, and edge cases.

#![cfg(feature = "layout")]

use enough::Unstoppable;
use zenjpeg::decoder::Decoder;
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};
use zenjpeg::layout::LayoutConfig;

/// Create a test JPEG with the given dimensions and a noise+patches pattern.
/// Uses LCG PRNG for reproducibility (not gradient — see CLAUDE.md benchmark rules).
fn make_test_jpeg(width: u32, height: u32) -> Vec<u8> {
    let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter).progressive(true);
    let pixel_count = (width * height) as usize;
    let mut pixels = vec![0u8; pixel_count * 3];

    // LCG PRNG for reproducible noise
    let mut rng: u32 = 0x1234_5678;
    let next = |rng: &mut u32| -> u8 {
        *rng = rng.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        (*rng >> 16) as u8
    };

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize * 3;
            // Noise base
            let r = next(&mut rng);
            let g = next(&mut rng);
            let b = next(&mut rng);

            // Patches: solid color blocks in some regions
            if x < width / 4 && y < height / 4 {
                pixels[idx] = 200;
                pixels[idx + 1] = 50;
                pixels[idx + 2] = 50;
            } else if x >= 3 * width / 4 && y >= 3 * height / 4 {
                pixels[idx] = 50;
                pixels[idx + 1] = 50;
                pixels[idx + 2] = 200;
            } else {
                pixels[idx] = r;
                pixels[idx + 1] = g;
                pixels[idx + 2] = b;
            }
        }
    }

    let mut encoder = config
        .request()
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .unwrap();
    encoder.push_packed(&pixels, Unstoppable).unwrap();
    encoder.finish().unwrap()
}

/// Create a test JPEG with 4:4:4 subsampling (MCU = 8x8).
fn make_test_jpeg_444(width: u32, height: u32) -> Vec<u8> {
    let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::None).progressive(true);
    let pixel_count = (width * height) as usize;
    let mut pixels = vec![128u8; pixel_count * 3];

    let mut rng: u32 = 0xDEAD_BEEF;
    let next = |rng: &mut u32| -> u8 {
        *rng = rng.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        (*rng >> 16) as u8
    };

    for i in 0..pixel_count * 3 {
        pixels[i] = next(&mut rng);
    }

    let mut encoder = config
        .request()
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .unwrap();
    encoder.push_packed(&pixels, Unstoppable).unwrap();
    encoder.finish().unwrap()
}

// =============================================================================
// Lossless path tests
// =============================================================================

#[test]
fn lossless_identity_returns_copy() {
    let jpeg = make_test_jpeg(64, 64);
    let result = LayoutConfig::new(85.0)
        .request(&jpeg)
        .execute(&Unstoppable)
        .unwrap();

    assert!(result.lossless, "identity should use lossless path");
    assert_eq!(result.width, 64);
    assert_eq!(result.height, 64);
    // Identity returns a copy of the input
    assert_eq!(result.data, jpeg);
}

#[test]
fn lossless_rotate_90() {
    // MCU-aligned for 4:2:0 (16x16 MCU)
    let jpeg = make_test_jpeg(64, 48);
    let result = LayoutConfig::new(85.0)
        .request(&jpeg)
        .rotate_90()
        .execute(&Unstoppable)
        .unwrap();

    assert!(result.lossless, "rotation-only should use lossless path");
    assert_eq!(result.width, 48, "rotate 90 swaps dimensions");
    assert_eq!(result.height, 64);
    assert_ne!(result.data, jpeg, "rotated data should differ");
}

#[test]
fn lossless_rotate_180() {
    let jpeg = make_test_jpeg(64, 64);
    let result = LayoutConfig::new(85.0)
        .request(&jpeg)
        .rotate_180()
        .execute(&Unstoppable)
        .unwrap();

    assert!(result.lossless);
    assert_eq!(result.width, 64);
    assert_eq!(result.height, 64);
    assert_ne!(result.data, jpeg);
}

#[test]
fn lossless_rotate_270() {
    let jpeg = make_test_jpeg(64, 48);
    let result = LayoutConfig::new(85.0)
        .request(&jpeg)
        .rotate_270()
        .execute(&Unstoppable)
        .unwrap();

    assert!(result.lossless);
    assert_eq!(result.width, 48);
    assert_eq!(result.height, 64);
}

#[test]
fn lossless_flip_horizontal() {
    let jpeg = make_test_jpeg(64, 64);
    let result = LayoutConfig::new(85.0)
        .request(&jpeg)
        .flip_h()
        .execute(&Unstoppable)
        .unwrap();

    assert!(result.lossless);
    assert_eq!(result.width, 64);
    assert_eq!(result.height, 64);
    assert_ne!(result.data, jpeg);
}

#[test]
fn lossless_flip_vertical() {
    let jpeg = make_test_jpeg(64, 64);
    let result = LayoutConfig::new(85.0)
        .request(&jpeg)
        .flip_v()
        .execute(&Unstoppable)
        .unwrap();

    assert!(result.lossless);
    assert_eq!(result.width, 64);
    assert_eq!(result.height, 64);
}

#[test]
fn lossless_auto_orient_exif6() {
    // EXIF 6 = Rotate 90 CW
    let jpeg = make_test_jpeg(64, 48);
    let result = LayoutConfig::new(85.0)
        .request(&jpeg)
        .auto_orient(6)
        .execute(&Unstoppable)
        .unwrap();

    assert!(result.lossless);
    assert_eq!(result.width, 48);
    assert_eq!(result.height, 64);
}

#[test]
fn lossless_auto_orient_exif1_is_noop() {
    // EXIF 1 = Normal (no transform needed)
    let jpeg = make_test_jpeg(64, 64);
    let result = LayoutConfig::new(85.0)
        .request(&jpeg)
        .auto_orient(1)
        .execute(&Unstoppable)
        .unwrap();

    assert!(result.lossless);
    assert_eq!(result.data, jpeg, "EXIF 1 should return identical data");
}

#[test]
fn lossless_composed_rotations() {
    // Rotate90 + Rotate90 = Rotate180
    let jpeg = make_test_jpeg(64, 64);
    let result = LayoutConfig::new(85.0)
        .request(&jpeg)
        .rotate_90()
        .rotate_90()
        .execute(&Unstoppable)
        .unwrap();

    assert!(result.lossless);
    // 90+90 = 180, no dimension swap
    assert_eq!(result.width, 64);
    assert_eq!(result.height, 64);
}

#[test]
fn lossless_output_is_valid_jpeg() {
    let jpeg = make_test_jpeg(64, 48);
    let result = LayoutConfig::new(85.0)
        .request(&jpeg)
        .rotate_90()
        .execute(&Unstoppable)
        .unwrap();

    // Verify the output can be decoded
    let decoded = Decoder::new().decode(&result.data, Unstoppable).unwrap();
    assert_eq!(decoded.width(), 48);
    assert_eq!(decoded.height(), 64);
}

// =============================================================================
// Lossy path tests
// =============================================================================

#[test]
fn lossy_fit_downscale() {
    let jpeg = make_test_jpeg(128, 128);
    let result = LayoutConfig::new(85.0)
        .request(&jpeg)
        .fit(64, 64)
        .execute(&Unstoppable)
        .unwrap();

    assert!(!result.lossless, "resize should use lossy path");
    assert_eq!(result.width, 64);
    assert_eq!(result.height, 64);

    // Verify output is valid JPEG
    let decoded = Decoder::new().decode(&result.data, Unstoppable).unwrap();
    assert_eq!(decoded.width(), 64);
    assert_eq!(decoded.height(), 64);
}

#[test]
fn lossy_fit_aspect_ratio_preserved() {
    // 128x64 → fit 64x64 → should produce 64x32 (preserve aspect ratio)
    let jpeg = make_test_jpeg(128, 64);
    let result = LayoutConfig::new(85.0)
        .request(&jpeg)
        .fit(64, 64)
        .execute(&Unstoppable)
        .unwrap();

    assert!(!result.lossless);
    assert_eq!(result.width, 64);
    assert_eq!(result.height, 32, "aspect ratio should be preserved");
}

#[test]
fn lossy_within_no_upscale() {
    // 64x64 → within 256x256 → should stay 64x64 (Within doesn't upscale)
    let jpeg = make_test_jpeg(64, 64);
    let result = LayoutConfig::new(85.0)
        .request(&jpeg)
        .within(256, 256)
        .execute(&Unstoppable)
        .unwrap();

    // Goes through lossy path (conservative detection) but dimensions should match
    assert_eq!(result.width, 64);
    assert_eq!(result.height, 64);
}

#[test]
fn lossy_within_downscale() {
    // 128x128 → within 64x64 → should produce 64x64
    let jpeg = make_test_jpeg(128, 128);
    let result = LayoutConfig::new(85.0)
        .request(&jpeg)
        .within(64, 64)
        .execute(&Unstoppable)
        .unwrap();

    assert!(!result.lossless);
    assert_eq!(result.width, 64);
    assert_eq!(result.height, 64);
}

#[test]
fn lossy_orient_plus_resize() {
    // Rotate + resize should go through lossy path
    let jpeg = make_test_jpeg(128, 64);
    let result = LayoutConfig::new(85.0)
        .request(&jpeg)
        .auto_orient(6) // Rotate 90
        .fit(32, 32)
        .execute(&Unstoppable)
        .unwrap();

    assert!(!result.lossless);
    // After rotate 90: 64x128, then fit 32x32 → preserving aspect: 16x32 or 32x32 depending
    // on zenlayout's interpretation. Just verify it's valid and within bounds.
    assert!(result.width <= 32);
    assert!(result.height <= 32);

    let decoded = Decoder::new().decode(&result.data, Unstoppable).unwrap();
    assert_eq!(decoded.width(), result.width);
    assert_eq!(decoded.height(), result.height);
}

#[test]
fn lossy_output_smaller_than_input() {
    let jpeg = make_test_jpeg(128, 128);
    let original_size = jpeg.len();

    let result = LayoutConfig::new(85.0)
        .request(&jpeg)
        .fit(32, 32)
        .execute(&Unstoppable)
        .unwrap();

    // Smaller dimensions should generally produce smaller file
    assert!(
        result.data.len() < original_size,
        "32x32 should be smaller than 128x128: {} >= {}",
        result.data.len(),
        original_size
    );
}

// =============================================================================
// Edge cases
// =============================================================================

#[test]
fn small_image_lossless() {
    // Minimum reasonable size (8x8 = one MCU for 4:4:4)
    let jpeg = make_test_jpeg_444(8, 8);
    let result = LayoutConfig::new(85.0)
        .request(&jpeg)
        .rotate_180()
        .execute(&Unstoppable)
        .unwrap();

    assert!(result.lossless);
    assert_eq!(result.width, 8);
    assert_eq!(result.height, 8);
}

#[test]
fn small_image_lossy_resize() {
    let jpeg = make_test_jpeg(64, 64);
    let result = LayoutConfig::new(85.0)
        .request(&jpeg)
        .fit(16, 16)
        .execute(&Unstoppable)
        .unwrap();

    assert!(!result.lossless);
    assert_eq!(result.width, 16);
    assert_eq!(result.height, 16);

    // Should produce valid JPEG
    let decoded = Decoder::new().decode(&result.data, Unstoppable).unwrap();
    assert_eq!(decoded.width(), 16);
}

#[test]
fn config_reuse_across_images() {
    let config = LayoutConfig::new(85.0);

    let jpeg1 = make_test_jpeg(64, 64);
    let jpeg2 = make_test_jpeg(128, 128);

    let r1 = config
        .request(&jpeg1)
        .flip_h()
        .execute(&Unstoppable)
        .unwrap();
    let r2 = config
        .request(&jpeg2)
        .fit(32, 32)
        .execute(&Unstoppable)
        .unwrap();

    assert!(r1.lossless);
    assert!(!r2.lossless);
    assert_eq!(r1.width, 64);
    assert_eq!(r2.width, 32);
}

#[test]
fn custom_filter() {
    let jpeg = make_test_jpeg(128, 128);
    let result = LayoutConfig::new(85.0)
        .with_filter(zenresize::Filter::Lanczos)
        .request(&jpeg)
        .fit(64, 64)
        .execute(&Unstoppable)
        .unwrap();

    assert!(!result.lossless);
    assert_eq!(result.width, 64);
    assert_eq!(result.height, 64);
}

#[test]
fn custom_subsampling_444() {
    let jpeg = make_test_jpeg(128, 128);
    let result = LayoutConfig::new(85.0)
        .with_subsampling(ChromaSubsampling::None)
        .request(&jpeg)
        .fit(64, 64)
        .execute(&Unstoppable)
        .unwrap();

    assert_eq!(result.width, 64);
    assert_eq!(result.height, 64);

    // Verify output decodes
    Decoder::new()
        .decode(&result.data, Unstoppable)
        .unwrap();
}

#[test]
fn non_progressive_output() {
    let jpeg = make_test_jpeg(128, 128);
    let result = LayoutConfig::new(85.0)
        .with_progressive(false)
        .request(&jpeg)
        .fit(64, 64)
        .execute(&Unstoppable)
        .unwrap();

    assert_eq!(result.width, 64);
    Decoder::new()
        .decode(&result.data, Unstoppable)
        .unwrap();
}

// =============================================================================
// Lossless roundtrip consistency
// =============================================================================

#[test]
fn lossless_rotate_roundtrip() {
    // Rotate 90 four times should produce identical coefficients
    // (though re-optimized Huffman tables may differ in size)
    let jpeg = make_test_jpeg(64, 64);

    let r1 = LayoutConfig::new(85.0)
        .request(&jpeg)
        .rotate_90()
        .execute(&Unstoppable)
        .unwrap();
    let r2 = LayoutConfig::new(85.0)
        .request(&r1.data)
        .rotate_90()
        .execute(&Unstoppable)
        .unwrap();
    let r3 = LayoutConfig::new(85.0)
        .request(&r2.data)
        .rotate_90()
        .execute(&Unstoppable)
        .unwrap();
    let r4 = LayoutConfig::new(85.0)
        .request(&r3.data)
        .rotate_90()
        .execute(&Unstoppable)
        .unwrap();

    // After 4 rotations, should have same dimensions
    assert_eq!(r4.width, 64);
    assert_eq!(r4.height, 64);

    // Decode both and compare pixel values (should be near-identical
    // since all transforms are lossless DCT-domain)
    let orig = Decoder::new().decode(&jpeg, Unstoppable).unwrap();
    let round = Decoder::new().decode(&r4.data, Unstoppable).unwrap();

    let orig_px = orig.into_pixels_u8().unwrap();
    let round_px = round.into_pixels_u8().unwrap();
    assert_eq!(orig_px.len(), round_px.len());

    // Max pixel difference should be 0 (true lossless)
    let max_diff: u8 = orig_px
        .iter()
        .zip(round_px.iter())
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap_or(0);
    assert_eq!(max_diff, 0, "4x rotate 90 should be pixel-perfect lossless");
}

#[test]
fn lossless_flip_roundtrip() {
    // flip_h twice should return identical output
    let jpeg = make_test_jpeg(64, 64);

    let r1 = LayoutConfig::new(85.0)
        .request(&jpeg)
        .flip_h()
        .execute(&Unstoppable)
        .unwrap();
    let r2 = LayoutConfig::new(85.0)
        .request(&r1.data)
        .flip_h()
        .execute(&Unstoppable)
        .unwrap();

    let orig = Decoder::new().decode(&jpeg, Unstoppable).unwrap();
    let round = Decoder::new().decode(&r2.data, Unstoppable).unwrap();

    let orig_px = orig.into_pixels_u8().unwrap();
    let round_px = round.into_pixels_u8().unwrap();

    let max_diff: u8 = orig_px
        .iter()
        .zip(round_px.iter())
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap_or(0);
    assert_eq!(max_diff, 0, "double flip_h should be pixel-perfect lossless");
}

// =============================================================================
// UltraHDR gain map preservation tests
// =============================================================================

/// UltraHDR XMP metadata for test images.
const ULTRAHDR_XMP: &str = r#"<?xpacket begin='' id='W5M0MpCehiHzreSzNTczkc9d'?>
<x:xmpmeta xmlns:x='adobe:ns:meta/'>
  <rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'>
    <rdf:Description rdf:about=''
      xmlns:hdrgm='http://ns.adobe.com/hdr-gain-map/1.0/'
      hdrgm:Version='1.0'
      hdrgm:GainMapMax='4.0'/>
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end='w'?>"#;

/// Create a fake UltraHDR JPEG: primary (with gain map XMP) + gain map JPEG appended after EOI.
fn make_ultrahdr_jpeg(
    primary_w: u32,
    primary_h: u32,
    gainmap_w: u32,
    gainmap_h: u32,
) -> Vec<u8> {
    let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter).progressive(true);

    // Encode primary with UltraHDR XMP
    let primary_pixels = make_noise_pixels(primary_w, primary_h, 0x1234_5678);
    let mut primary_encoder = config
        .request()
        .xmp(ULTRAHDR_XMP.as_bytes())
        .encode_from_bytes(primary_w, primary_h, PixelLayout::Rgb8Srgb)
        .unwrap();
    primary_encoder
        .push_packed(&primary_pixels, Unstoppable)
        .unwrap();
    let primary_jpeg = primary_encoder.finish().unwrap();

    // Encode gain map (grayscale-like, but RGB for simplicity)
    let gm_pixels = make_noise_pixels(gainmap_w, gainmap_h, 0xDEAD_BEEF);
    let gm_config = EncoderConfig::ycbcr(75.0, ChromaSubsampling::Quarter);
    let mut gm_encoder = gm_config
        .request()
        .encode_from_bytes(gainmap_w, gainmap_h, PixelLayout::Rgb8Srgb)
        .unwrap();
    gm_encoder.push_packed(&gm_pixels, Unstoppable).unwrap();
    let gm_jpeg = gm_encoder.finish().unwrap();

    // Concatenate: primary + gain map (simple UltraHDR structure)
    let mut ultrahdr = primary_jpeg;
    ultrahdr.extend_from_slice(&gm_jpeg);
    ultrahdr
}

/// Generate noise pixel data for test images.
fn make_noise_pixels(width: u32, height: u32, seed: u32) -> Vec<u8> {
    let pixel_count = (width * height) as usize;
    let mut pixels = vec![0u8; pixel_count * 3];
    let mut rng = seed;
    let next = |rng: &mut u32| -> u8 {
        *rng = rng.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        (*rng >> 16) as u8
    };
    for p in pixels.iter_mut() {
        *p = next(&mut rng);
    }
    pixels
}

#[test]
fn ultrahdr_lossless_identity_preserves_gainmap() {
    let ultrahdr = make_ultrahdr_jpeg(64, 64, 32, 32);

    let result = LayoutConfig::new(85.0)
        .request(&ultrahdr)
        .execute(&Unstoppable)
        .unwrap();

    assert!(result.lossless);
    assert_eq!(result.width, 64);
    assert_eq!(result.height, 64);

    // The output should still contain the gain map (secondary JPEG after EOI)
    // Identity lossless returns a copy, which includes the appended gain map
    assert_eq!(result.data, ultrahdr, "identity should preserve the entire stream");
}

#[test]
fn ultrahdr_lossless_rotate_preserves_gainmap() {
    let ultrahdr = make_ultrahdr_jpeg(64, 48, 32, 24);

    let result = LayoutConfig::new(85.0)
        .request(&ultrahdr)
        .rotate_180()
        .execute(&Unstoppable)
        .unwrap();

    assert!(result.lossless);
    assert_eq!(result.width, 64);
    assert_eq!(result.height, 48);

    // Output should be larger than just the primary (gain map appended)
    // Verify the output contains a secondary JPEG by checking for two SOI markers
    let soi_count = result
        .data
        .windows(2)
        .filter(|w| w[0] == 0xFF && w[1] == 0xD8)
        .count();
    assert!(
        soi_count >= 2,
        "output should contain at least 2 JPEG streams (primary + gain map), got {soi_count}"
    );

    // The output should be decodable as a regular JPEG (just the primary)
    let decoded = Decoder::new()
        .decode(&result.data, Unstoppable)
        .unwrap();
    assert_eq!(decoded.width(), 64);
    assert_eq!(decoded.height(), 48);
}

#[test]
fn ultrahdr_lossy_resize_preserves_gainmap() {
    let ultrahdr = make_ultrahdr_jpeg(128, 128, 64, 64);

    let result = LayoutConfig::new(85.0)
        .request(&ultrahdr)
        .fit(64, 64)
        .execute(&Unstoppable)
        .unwrap();

    assert!(!result.lossless);
    assert_eq!(result.width, 64);
    assert_eq!(result.height, 64);

    // Output should contain the resized gain map (two SOI markers)
    let soi_count = result
        .data
        .windows(2)
        .filter(|w| w[0] == 0xFF && w[1] == 0xD8)
        .count();
    assert!(
        soi_count >= 2,
        "lossy resize should preserve gain map: got {soi_count} SOI markers"
    );

    // Primary should be decodable at target dimensions
    let decoded = Decoder::new()
        .decode(&result.data, Unstoppable)
        .unwrap();
    assert_eq!(decoded.width(), 64);
    assert_eq!(decoded.height(), 64);
}

#[test]
fn non_ultrahdr_jpeg_unchanged_by_gainmap_detection() {
    // Regular JPEG without gain map XMP should work exactly as before
    let jpeg = make_test_jpeg(64, 64);

    let result = LayoutConfig::new(85.0)
        .request(&jpeg)
        .rotate_180()
        .execute(&Unstoppable)
        .unwrap();

    assert!(result.lossless);
    assert_eq!(result.width, 64);
    assert_eq!(result.height, 64);

    // Should have exactly one SOI (no gain map appended)
    let soi_count = result
        .data
        .windows(2)
        .filter(|w| w[0] == 0xFF && w[1] == 0xD8)
        .count();
    assert_eq!(soi_count, 1, "non-UltraHDR should have exactly 1 SOI");
}
