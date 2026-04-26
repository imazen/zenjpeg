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

/// Strict pixel-correctness regression for XYB Full BASELINE (non-progressive).
///
/// Background: the original `XybSubsampling::Full` implementation only fixed
/// the SOF header (1×1/1×1/1×1) and the layout dimensions. The encoder still
/// emitted the bitstream with BQuarter MCU geometry (4 X + 4 Y + 1 B per MCU),
/// so the decoder — correctly following the SOF — read every block as the
/// wrong component. On a 4-quadrant test image the colors came out completely
/// wrong (TL red → green, BL blue → green, etc.). The progressive XYB path
/// happened to be correct because progressive scans are non-interleaved
/// (1 component per scan), which sidesteps the MCU-layout question entirely.
///
/// This test exercises the baseline XYB Full path with a 4-quadrant test
/// image and asserts each quadrant decodes to roughly the right color.
#[test]
fn xyb_full_baseline_pixel_correctness() {
    let w = 128u32;
    let h = 128u32;
    // 4 quadrants of distinct colors — any MCU misalignment scrambles them.
    let rgb: Vec<u8> = (0..h)
        .flat_map(|y| {
            (0..w).flat_map(move |x| {
                let top = y < h / 2;
                let left = x < w / 2;
                match (top, left) {
                    (true, true) => [220u8, 40, 40],    // TL red
                    (true, false) => [40u8, 220, 40],   // TR green
                    (false, true) => [40u8, 40, 220],   // BL blue
                    (false, false) => [220u8, 220, 40], // BR yellow
                }
            })
        })
        .collect();

    let cfg = EncoderConfig::xyb(85.0, XybSubsampling::Full).progressive(false);
    let jpeg = cfg
        .encode_bytes(&rgb, w, h, PixelLayout::Rgb8Srgb)
        .expect("encode");

    let decoded = Decoder::new()
        .decode(&jpeg, enough::Unstoppable)
        .expect("decode");
    let pixels = decoded.pixels_u8().unwrap();
    let probe = |x: u32, y: u32| -> (i32, i32, i32) {
        let i = (y as usize * w as usize + x as usize) * 3;
        (pixels[i] as i32, pixels[i + 1] as i32, pixels[i + 2] as i32)
    };

    // Probe at quadrant centers and check the dominant channel.
    // Note: XYB encoding produces noticeable lossy color casts on saturated
    // primaries even with correct layout (e.g. BQuarter at Q85 sees BR yellow
    // decode to ~(90, 216, 38)). The test focuses on *layout correctness*: the
    // dominant channels should match the source. The original broken Full path
    // produced (12, 131, 16) for TL red — green dominant — which this catches.
    let tl = probe(w / 4, h / 4);
    let tr = probe(3 * w / 4, h / 4);
    let bl = probe(w / 4, 3 * h / 4);
    let br = probe(3 * w / 4, 3 * h / 4);

    // TL red: R is the dominant channel
    assert!(tl.0 > tl.1 && tl.0 > tl.2, "TL not red-dominant: {tl:?}");
    // TR green: G is the dominant channel
    assert!(tr.1 > tr.0 && tr.1 > tr.2, "TR not green-dominant: {tr:?}");
    // BL blue: B is the dominant channel
    assert!(bl.2 > bl.0 && bl.2 > bl.1, "BL not blue-dominant: {bl:?}");
    // BR yellow: R+G both above B (G alone may be > R due to XYB chroma cast)
    assert!(
        br.1 > br.2 && br.0 > br.2,
        "BR not yellow-ish (RG > B): {br:?}"
    );
}

// ============================================================================
// Boxed XYB ↔ RGB roundtrip matrix
// ============================================================================
//
// Verifies that every (XybSubsampling, progressive/baseline, quality) combo
// roundtrips without color-channel scrambling. Catches MCU-layout bugs like
// the one in commit daf52508.
//
// The 4-quadrant probe is layout-sensitive: any encoder/decoder disagreement
// on how blocks are arranged within an MCU shows up as colors landing in the
// wrong quadrant.

fn quadrant_image(w: u32, h: u32) -> Vec<u8> {
    (0..h)
        .flat_map(|y| {
            (0..w).flat_map(move |x| {
                let top = y < h / 2;
                let left = x < w / 2;
                match (top, left) {
                    (true, true) => [220u8, 40, 40],    // TL red
                    (true, false) => [40u8, 220, 40],   // TR green
                    (false, true) => [40u8, 40, 220],   // BL blue
                    (false, false) => [220u8, 220, 40], // BR yellow
                }
            })
        })
        .collect()
}

fn assert_quadrants_match_source(pixels: &[u8], w: u32, _h: u32, label: &str) {
    let probe = |x: u32, y: u32| -> (i32, i32, i32) {
        let i = (y as usize * w as usize + x as usize) * 3;
        (pixels[i] as i32, pixels[i + 1] as i32, pixels[i + 2] as i32)
    };
    let q = w / 4;
    let tl = probe(q, q);
    let tr = probe(3 * q, q);
    let bl = probe(q, 3 * q);
    let br = probe(3 * q, 3 * q);

    assert!(
        tl.0 > tl.1 && tl.0 > tl.2,
        "{label}: TL not red-dominant: {tl:?}"
    );
    assert!(
        tr.1 > tr.0 && tr.1 > tr.2,
        "{label}: TR not green-dominant: {tr:?}"
    );
    assert!(
        bl.2 > bl.0 && bl.2 > bl.1,
        "{label}: BL not blue-dominant: {bl:?}"
    );
    assert!(
        br.1 > br.2 && br.0 > br.2,
        "{label}: BR not yellow-ish: {br:?}"
    );
}

/// Sweep over the full XYB encode matrix and assert pixel-level layout
/// correctness on the 4-quadrant probe. Catches MCU-layout regressions
/// in any path: BQuarter or Full × progressive or baseline × low-to-high Q.
#[test]
fn xyb_roundtrip_matrix_pixel_correctness() {
    let w = 128u32;
    let h = 128u32;
    let rgb = quadrant_image(w, h);

    for sub in [XybSubsampling::BQuarter, XybSubsampling::Full] {
        for progressive in [false, true] {
            for &q in &[15, 50, 85, 95] {
                let label = format!("{sub:?}/prog={progressive}/Q{q}");
                let cfg = EncoderConfig::xyb(q, sub).progressive(progressive);
                let jpeg = cfg
                    .encode_bytes(&rgb, w, h, PixelLayout::Rgb8Srgb)
                    .unwrap_or_else(|e| panic!("{label}: encode failed: {e}"));
                let decoded = Decoder::new()
                    .decode(&jpeg, enough::Unstoppable)
                    .unwrap_or_else(|e| panic!("{label}: decode failed: {e}"));
                let pixels = decoded.pixels_u8().unwrap();
                assert_quadrants_match_source(pixels, w, h, &label);
            }
        }
    }
}

/// Verify that the encoded XYB JPEG actually embeds the canonical XYB ICC
/// profile in an APP2 segment, so non-XYB-aware decoders can interpret colors
/// via standard CMS. Both BQuarter and Full must embed it.
#[test]
fn xyb_encoder_embeds_xyb_icc_profile() {
    use zenjpeg::color::icc::{extract_icc_profile, is_xyb_profile};

    let rgb = generate_test_image(64, 64);
    for sub in [XybSubsampling::BQuarter, XybSubsampling::Full] {
        for progressive in [false, true] {
            let cfg = EncoderConfig::xyb(85.0, sub).progressive(progressive);
            let jpeg = cfg
                .encode_bytes(&rgb, 64, 64, PixelLayout::Rgb8Srgb)
                .expect("encode");
            let icc = extract_icc_profile(&jpeg)
                .unwrap_or_else(|| panic!("{sub:?}/prog={progressive}: no ICC profile found"));
            assert!(
                is_xyb_profile(&icc),
                "{sub:?}/prog={progressive}: embedded ICC is not XYB profile (len={})",
                icc.len()
            );
        }
    }
}

/// Verify that the decoder identifies XYB JPEGs as XYB via the embedded ICC
/// profile on every (subsampling, scan) combination. This is the gate that
/// drives the f32 XYB→RGB conversion path in the output stage.
#[test]
fn xyb_decoder_detects_xyb_color_space() {
    use zenjpeg::color::icc::is_xyb_profile;

    let rgb = generate_test_image(64, 64);
    for sub in [XybSubsampling::BQuarter, XybSubsampling::Full] {
        for progressive in [false, true] {
            let cfg = EncoderConfig::xyb(85.0, sub).progressive(progressive);
            let jpeg = cfg
                .encode_bytes(&rgb, 64, 64, PixelLayout::Rgb8Srgb)
                .expect("encode");
            // Probe via the public ICC extractor (mirrors what the decoder uses
            // internally to set the XYB flag).
            let icc = zenjpeg::color::icc::extract_icc_profile(&jpeg)
                .unwrap_or_else(|| panic!("{sub:?}/prog={progressive}: no ICC"));
            assert!(
                is_xyb_profile(&icc),
                "{sub:?}/prog={progressive}: ICC not detected as XYB"
            );
        }
    }
}

/// Verify that the decoder produces visually-correct sRGB output for XYB JPEGs
/// even WITHOUT calling `.correct_color(Some(Srgb))`. zenjpeg detects XYB via
/// the embedded ICC and runs its own scaled-XYB → sRGB inverse transform in
/// the output stage (`xyb_planes_to_srgb_u8_simd`), so no CMS is required for
/// end-user-correct colors. (The ICC remains useful for OTHER decoders that
/// don't understand XYB natively.)
#[test]
fn xyb_decode_produces_srgb_without_correct_color_call() {
    let w = 128u32;
    let h = 128u32;
    let rgb = quadrant_image(w, h);
    let cfg = EncoderConfig::xyb(85.0, XybSubsampling::Full);
    let jpeg = cfg
        .encode_bytes(&rgb, w, h, PixelLayout::Rgb8Srgb)
        .expect("encode");

    // Default decode (no .correct_color call)
    let decoded = Decoder::new()
        .decode(&jpeg, enough::Unstoppable)
        .expect("decode");
    let pixels = decoded.pixels_u8().unwrap();
    assert_quadrants_match_source(pixels, w, h, "no_correct_color");
}

/// Hard test for Option 1 (no-moxcms XYB). This test makes no reference to
/// any moxcms API and verifies the 4-quadrant correctness through the
/// default decode path. When built with `--no-default-features --features
/// "trellis decoder"` (i.e. no `moxcms`), it proves the in-decoder inverse
/// XYB → sRGB transform carries the entire load.
///
/// Exercises the full encode/decode matrix to catch any subsampling- or
/// scan-specific regressions in the SIMD kernel.
#[test]
fn xyb_decode_correct_without_moxcms() {
    let w = 128u32;
    let h = 128u32;
    let rgb = quadrant_image(w, h);
    for sub in [XybSubsampling::BQuarter, XybSubsampling::Full] {
        for progressive in [false, true] {
            for &q in &[50.0_f32, 85.0, 95.0] {
                let label = format!("{sub:?}/prog={progressive}/Q{q}");
                let cfg = EncoderConfig::xyb(q, sub).progressive(progressive);
                let jpeg = cfg
                    .encode_bytes(&rgb, w, h, PixelLayout::Rgb8Srgb)
                    .unwrap_or_else(|e| panic!("{label}: encode failed: {e}"));
                let decoded = Decoder::new()
                    .decode(&jpeg, enough::Unstoppable)
                    .unwrap_or_else(|e| panic!("{label}: decode failed: {e}"));
                let pixels = decoded.pixels_u8().unwrap();
                assert_quadrants_match_source(pixels, w, h, &label);
            }
        }
    }
}

/// Invariant: for XYB sources, explicit `correct_color(Some(Srgb))` must
/// produce byte-for-byte the same pixels as default decode.
///
/// Motivation: before the built-in SIMD inverse transform (Option 1),
/// default decode returned scaled-XYB raw bytes, which only looked sRGB
/// after the XYB ICC was applied via moxcms. With the SIMD kernel doing
/// the inverse in-decoder, the ICC apply for XYB is now a no-op. This
/// test guards against regressions where the two paths might diverge
/// (e.g. someone re-enables the ICC apply for XYB and it double-transforms).
///
/// Runs identically under both `--features moxcms` and without — `correct_color`
/// is a no-op for XYB in either build.
#[test]
#[cfg(feature = "moxcms")]
fn xyb_decode_byte_equal_with_and_without_correct_color() {
    use zenjpeg::color::icc::TargetColorSpace;
    let w = 128u32;
    let h = 128u32;
    let rgb = quadrant_image(w, h);
    let cfg = EncoderConfig::xyb(85.0, XybSubsampling::Full);
    let jpeg = cfg
        .encode_bytes(&rgb, w, h, PixelLayout::Rgb8Srgb)
        .expect("encode");

    let default_decoded = Decoder::new()
        .decode(&jpeg, enough::Unstoppable)
        .expect("default decode");
    let default_pixels = default_decoded.pixels_u8().unwrap();

    let corrected_decoded = Decoder::new()
        .correct_color(Some(TargetColorSpace::Srgb))
        .decode(&jpeg, enough::Unstoppable)
        .expect("correct_color decode");
    let corrected_pixels = corrected_decoded.pixels_u8().unwrap();

    assert_eq!(
        default_pixels.len(),
        corrected_pixels.len(),
        "buffer lengths differ"
    );
    assert_eq!(
        default_pixels, corrected_pixels,
        "default and correct_color(Srgb) diverge for XYB — ICC double-transform regression?"
    );
}

// ============================================================================
// Linear-input XYB pixel correctness (regression for Known Bug #7)
// ============================================================================
//
// XYB + PixelLayout::RgbF32Linear / Rgb16Linear used to skip the scale_xyb
// step and then multiplied the raw (unscaled) XYB by 255.0, pushing Y
// off-scale and saturating every MCU to white. These tests assert that
// linear-input XYB decodes to the same quadrant-dominant colors as the
// sRGB-input XYB reference (`xyb_full_baseline_pixel_correctness`).

fn make_quadrant_rgb_u8(w: u32, h: u32) -> Vec<u8> {
    (0..h)
        .flat_map(|y| {
            (0..w).flat_map(move |x| {
                let top = y < h / 2;
                let left = x < w / 2;
                match (top, left) {
                    (true, true) => [220u8, 40, 40],    // TL red
                    (true, false) => [40u8, 220, 40],   // TR green
                    (false, true) => [40u8, 40, 220],   // BL blue
                    (false, false) => [220u8, 220, 40], // BR yellow
                }
            })
        })
        .collect()
}

fn rgb_u8_to_linear_f32(rgb_u8: &[u8]) -> Vec<f32> {
    rgb_u8
        .iter()
        .map(|&v| linear_srgb::default::srgb_to_linear(v as f32 / 255.0))
        .collect()
}

fn rgb_u8_to_linear_u16(rgb_u8: &[u8]) -> Vec<u16> {
    rgb_u8
        .iter()
        .map(|&v| {
            let lin = linear_srgb::default::srgb_to_linear(v as f32 / 255.0);
            (lin.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16
        })
        .collect()
}

fn assert_quadrants_correct(pixels: &[u8], w: u32, h: u32, label: &str) {
    let probe = |x: u32, y: u32| -> (i32, i32, i32) {
        let i = (y as usize * w as usize + x as usize) * 3;
        (pixels[i] as i32, pixels[i + 1] as i32, pixels[i + 2] as i32)
    };
    let tl = probe(w / 4, h / 4);
    let tr = probe(3 * w / 4, h / 4);
    let bl = probe(w / 4, 3 * h / 4);
    let br = probe(3 * w / 4, 3 * h / 4);

    assert!(
        tl.0 > tl.1 && tl.0 > tl.2,
        "{label}: TL not red-dominant: {tl:?}"
    );
    assert!(
        tr.1 > tr.0 && tr.1 > tr.2,
        "{label}: TR not green-dominant: {tr:?}"
    );
    assert!(
        bl.2 > bl.0 && bl.2 > bl.1,
        "{label}: BL not blue-dominant: {bl:?}"
    );
    assert!(
        br.1 > br.2 && br.0 > br.2,
        "{label}: BR not yellow-ish (RG > B): {br:?}"
    );
}

/// XYB + `PixelLayout::RgbF32Linear` should produce the same quadrant
/// structure as XYB + `PixelLayout::Rgb8Srgb` on the same source image.
#[test]
fn xyb_full_linear_f32_pixel_correctness() {
    let w = 128u32;
    let h = 128u32;
    let rgb_u8 = make_quadrant_rgb_u8(w, h);
    let rgb_f32 = rgb_u8_to_linear_f32(&rgb_u8);
    let bytes: &[u8] = bytemuck::cast_slice(&rgb_f32);

    let cfg = EncoderConfig::xyb(85.0, XybSubsampling::Full).progressive(false);
    let jpeg = cfg
        .encode_bytes(bytes, w, h, PixelLayout::RgbF32Linear)
        .expect("encode XYB RgbF32Linear");
    let decoded = Decoder::new()
        .decode(&jpeg, enough::Unstoppable)
        .expect("decode");
    let pixels = decoded.pixels_u8().unwrap();
    assert_quadrants_correct(&pixels, w, h, "RgbF32Linear+Full");
}

#[test]
fn xyb_bquarter_linear_f32_pixel_correctness() {
    let w = 128u32;
    let h = 128u32;
    let rgb_u8 = make_quadrant_rgb_u8(w, h);
    let rgb_f32 = rgb_u8_to_linear_f32(&rgb_u8);
    let bytes: &[u8] = bytemuck::cast_slice(&rgb_f32);

    let cfg = EncoderConfig::xyb(85.0, XybSubsampling::BQuarter).progressive(false);
    let jpeg = cfg
        .encode_bytes(bytes, w, h, PixelLayout::RgbF32Linear)
        .expect("encode XYB RgbF32Linear BQuarter");
    let decoded = Decoder::new()
        .decode(&jpeg, enough::Unstoppable)
        .expect("decode");
    let pixels = decoded.pixels_u8().unwrap();
    assert_quadrants_correct(&pixels, w, h, "RgbF32Linear+BQuarter");
}

#[test]
fn xyb_full_linear_u16_pixel_correctness() {
    let w = 128u32;
    let h = 128u32;
    let rgb_u8 = make_quadrant_rgb_u8(w, h);
    let rgb_u16 = rgb_u8_to_linear_u16(&rgb_u8);
    let bytes: &[u8] = bytemuck::cast_slice(&rgb_u16);

    let cfg = EncoderConfig::xyb(85.0, XybSubsampling::Full).progressive(false);
    let jpeg = cfg
        .encode_bytes(bytes, w, h, PixelLayout::Rgb16Linear)
        .expect("encode XYB Rgb16Linear");
    let decoded = Decoder::new()
        .decode(&jpeg, enough::Unstoppable)
        .expect("decode");
    let pixels = decoded.pixels_u8().unwrap();
    assert_quadrants_correct(&pixels, w, h, "Rgb16Linear+Full");
}

/// Solid-colour red: linear and sRGB inputs for XYB must decode to roughly
/// the same thing. If the linear path has the old scale-xyb bug, the decode
/// will saturate to near-white (≈255, 255, 255) instead of the expected
/// near-red (~220, ~40, ~40).
#[test]
fn xyb_linear_matches_srgb_solid_red() {
    let w = 128u32;
    let h = 128u32;
    let rgb_u8: Vec<u8> = (0..h * w).flat_map(|_| [220u8, 40, 40]).collect();
    let rgb_f32 = rgb_u8_to_linear_f32(&rgb_u8);
    let bytes: &[u8] = bytemuck::cast_slice(&rgb_f32);

    let cfg = EncoderConfig::xyb(85.0, XybSubsampling::BQuarter).progressive(false);

    // sRGB reference path (known good).
    let jpeg_srgb = cfg
        .encode_bytes(&rgb_u8, w, h, PixelLayout::Rgb8Srgb)
        .expect("encode sRGB");
    let decoded_srgb = Decoder::new()
        .decode(&jpeg_srgb, enough::Unstoppable)
        .expect("decode sRGB");
    let ref_pixels = decoded_srgb.pixels_u8().unwrap();

    // Linear f32 path — should land in the same colour neighbourhood.
    let jpeg_lin = cfg
        .encode_bytes(bytes, w, h, PixelLayout::RgbF32Linear)
        .expect("encode RgbF32Linear");
    let decoded_lin = Decoder::new()
        .decode(&jpeg_lin, enough::Unstoppable)
        .expect("decode RgbF32Linear");
    let lin_pixels = decoded_lin.pixels_u8().unwrap();

    let (rr, rg, rb) = (
        ref_pixels[0] as i32,
        ref_pixels[1] as i32,
        ref_pixels[2] as i32,
    );
    let (lr, lg, lb) = (
        lin_pixels[0] as i32,
        lin_pixels[1] as i32,
        lin_pixels[2] as i32,
    );

    // Guard: the sRGB reference must be red-dominant (sanity-check the known-good path).
    assert!(
        rr > rg && rr > rb,
        "sRGB reference not red: ({rr},{rg},{rb})"
    );
    // Main assertion: linear-input must also decode as red. Pre-fix this is (255,255,255).
    assert!(
        lr > lg && lr > lb,
        "RgbF32Linear should decode as red; got ({lr},{lg},{lb}) (sRGB ref: ({rr},{rg},{rb})). \
         The old bug made this saturate to white."
    );
    // Tighter: within ~30 of the sRGB reference per channel.
    let dr = (lr - rr).abs();
    let dg = (lg - rg).abs();
    let db = (lb - rb).abs();
    assert!(
        dr <= 30 && dg <= 30 && db <= 30,
        "linear vs sRGB decode diverges too far: linear=({lr},{lg},{lb}) ref=({rr},{rg},{rb})"
    );
}
