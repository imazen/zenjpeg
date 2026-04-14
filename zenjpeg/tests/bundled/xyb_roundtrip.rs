//! Regression test for issue #4: XYB encoder produces undecodable JPEGs.
//!
//! The frequency counter in `collect_block_frequencies_simd` clamped DC categories
//! to 11 via `.min(11)`, but the actual encoder wrote unclamped categories (12+).
//! This meant the optimized Huffman table lacked codes for categories 12+,
//! corrupting the bitstream with (code=0, len=0) writes.
//!
//! XYB at low quality produces DC differences > ±2047 (wider dynamic range than
//! YCbCr), triggering DC categories 12-15 that standard YCbCr never hits.

use zenjpeg::decoder::Decoder;
use zenjpeg::encoder::{EncoderConfig, PixelLayout, XybSubsampling};

/// Generate a noise+patches test image that produces varied DC coefficients.
/// Uses deterministic seeded "random" to avoid needing external test images.
fn generate_test_image(width: usize, height: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; width * height * 3];
    let mut seed: u32 = 0xDEAD_BEEF;

    for y in 0..height {
        for x in 0..width {
            // LCG pseudo-random
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            let r = ((seed >> 16) & 0xFF) as u8;
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            let g = ((seed >> 16) & 0xFF) as u8;
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            let b = ((seed >> 16) & 0xFF) as u8;

            // Mix patches of solid color with noise for varied DC coefficients
            let patch_x = x / 32;
            let patch_y = y / 32;
            let patch_id = (patch_x * 7 + patch_y * 13) % 5;

            let idx = (y * width + x) * 3;
            match patch_id {
                0 => {
                    // Pure noise
                    rgb[idx] = r;
                    rgb[idx + 1] = g;
                    rgb[idx + 2] = b;
                }
                1 => {
                    // Bright red patch with noise
                    rgb[idx] = 200u8.wrapping_add(r / 4);
                    rgb[idx + 1] = r / 8;
                    rgb[idx + 2] = r / 8;
                }
                2 => {
                    // Dark blue patch with noise
                    rgb[idx] = g / 16;
                    rgb[idx + 1] = g / 16;
                    rgb[idx + 2] = 50u8.wrapping_add(b / 4);
                }
                3 => {
                    // High contrast (near-black / near-white alternating)
                    let bright = ((x + y) % 2 == 0) as u8 * 240;
                    rgb[idx] = bright.wrapping_add(r / 16);
                    rgb[idx + 1] = bright.wrapping_add(g / 16);
                    rgb[idx + 2] = bright.wrapping_add(b / 16);
                }
                _ => {
                    // Green gradient with noise
                    let gy = (y * 255 / height) as u8;
                    rgb[idx] = r / 8;
                    rgb[idx + 1] = gy.wrapping_add(g / 8);
                    rgb[idx + 2] = r / 8;
                }
            }
        }
    }
    rgb
}

/// Encode with XYB 4:2:0 at quality levels that previously produced corrupt output,
/// then verify the result decodes successfully.
#[test]
fn xyb_420_roundtrip_all_qualities() {
    let width = 512u32;
    let height = 512u32;
    let rgb = generate_test_image(width as usize, height as usize);

    for quality in [15, 20, 50, 60, 75, 80, 85, 90, 95] {
        let config = EncoderConfig::xyb(quality, XybSubsampling::BQuarter);
        let encoded = config
            .encode_bytes(&rgb, width, height, PixelLayout::Rgb8Srgb)
            .unwrap_or_else(|e| panic!("XYB 420 q{quality} encode failed: {e}"));

        // This used to fail with "invalid Huffman code" or "expected restart marker"
        let decoded = Decoder::new().decode(&encoded, enough::Unstoppable);
        assert!(
            decoded.is_ok(),
            "XYB 420 q{quality} roundtrip decode failed: {}",
            decoded.unwrap_err()
        );
    }
}

/// XYB allows both allow_16bit_quant_tables settings, but always uses SOF1.
/// SOF1 is required for XYB DC categories, independent of quant precision.
#[test]
fn xyb_allows_baseline_quant() {
    // force_baseline() works for XYB (clamps quant, still uses SOF1)
    let config = EncoderConfig::xyb(50, XybSubsampling::BQuarter).force_baseline();
    assert!(!config.is_allow_16bit_quant_tables());

    // allow_16bit_quant_tables(false) works for XYB
    let config = EncoderConfig::xyb(50, XybSubsampling::BQuarter).allow_16bit_quant_tables(false);
    assert!(!config.is_allow_16bit_quant_tables());

    // allow_16bit_quant_tables(true) also works
    let config = EncoderConfig::xyb(50, XybSubsampling::BQuarter).allow_16bit_quant_tables(true);
    assert!(config.is_allow_16bit_quant_tables());

    // XYB defaults to allow_16bit=false (quant values >255 have no quality impact)
    let config = EncoderConfig::xyb(50, XybSubsampling::BQuarter);
    assert!(!config.is_allow_16bit_quant_tables());
}

/// Non-progressive XYB output uses SOF1 (extended sequential), even with 8-bit quant tables.
/// SOF1 is required because XYB DC categories can exceed baseline's limit of 11.
/// Progressive XYB uses SOF2 (progressive) which also supports extended DC categories.
#[test]
fn xyb_sequential_uses_sof1() {
    // Force sequential mode (progressive uses SOF2 which is fine for XYB)
    let config = EncoderConfig::xyb(50, XybSubsampling::BQuarter).progressive(false);
    assert!(!config.is_allow_16bit_quant_tables());

    let rgb = generate_test_image(64, 64);
    let encoded = config
        .encode_bytes(&rgb, 64, 64, PixelLayout::Rgb8Srgb)
        .expect("encode should succeed");

    // SOF1 = 0xFFC1, SOF0 = 0xFFC0
    let has_sof1 = encoded.windows(2).any(|w| w == [0xFF, 0xC1]);
    let has_sof0 = encoded.windows(2).any(|w| w == [0xFF, 0xC0]);
    assert!(has_sof1, "Sequential XYB must use SOF1 (extended)");
    assert!(!has_sof0, "Sequential XYB must not use SOF0 (baseline)");
}

/// XYB full resolution (no subsampling) roundtrip — verify no regression.
#[test]
fn xyb_full_roundtrip() {
    let width = 256u32;
    let height = 256u32;
    let rgb = generate_test_image(width as usize, height as usize);

    for quality in [15, 50, 85] {
        let config = EncoderConfig::xyb(quality, XybSubsampling::Full);
        let encoded = config
            .encode_bytes(&rgb, width, height, PixelLayout::Rgb8Srgb)
            .unwrap_or_else(|e| panic!("XYB full q{quality} encode failed: {e}"));

        let decoded = Decoder::new().decode(&encoded, enough::Unstoppable);
        assert!(
            decoded.is_ok(),
            "XYB full q{quality} roundtrip decode failed: {}",
            decoded.unwrap_err()
        );
    }
}
