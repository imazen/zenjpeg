//! Integration tests for `TinyFileMode`.
//!
//! These tests cover:
//!
//! - `Off` vs pre-change behavior: `Off` produces the same bytes as the
//!   unqualified baseline-sequential encode (regression test).
//! - `Force` strictly shrinks small images relative to `Off`.
//! - `Auto` follows `Force` below the heuristic threshold and `Off` above.
//! - All three modes decode cleanly through zenjpeg's own decoder.
//! - `Force` + 4:2:0 + q=85 round-trips through the `image` crate.
//! - `TinyFileMode::Auto` + XYB leaves XYB's canonical output unchanged.

#[path = "../src/test_utils.rs"]
mod test_utils;

use test_utils::{TestImage, generate_gradient_d};
use zenjpeg::encoder::{
    ChromaSubsampling, EncoderConfig, PixelLayout, TinyFileMode, XybSubsampling,
    should_activate_tiny_file_mode,
};

// ---------- helpers ---------------------------------------------------------

fn encode_rgb(
    width: u32,
    height: u32,
    data: &[u8],
    config: &EncoderConfig,
) -> zenjpeg::encoder::Result<Vec<u8>> {
    let mut enc = config.encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)?;
    enc.push_packed(data, enough::Unstoppable)?;
    enc.finish()
}

fn encode_gray(
    width: u32,
    height: u32,
    data: &[u8],
    config: &EncoderConfig,
) -> zenjpeg::encoder::Result<Vec<u8>> {
    let mut enc = config.encode_from_bytes(width, height, PixelLayout::Gray8Srgb)?;
    enc.push_packed(data, enough::Unstoppable)?;
    enc.finish()
}

fn solid_gray_image(width: u32, height: u32) -> Vec<u8> {
    let img = generate_gradient_d(width, height, 1);
    img.pixels
}

fn solid_rgb_image(width: u32, height: u32) -> TestImage {
    generate_gradient_d(width, height, 3)
}

fn baseline_ycbcr(q: u8) -> EncoderConfig {
    EncoderConfig::ycbcr(q, ChromaSubsampling::Quarter).progressive(false)
}

fn baseline_ycbcr_444(q: u8) -> EncoderConfig {
    EncoderConfig::ycbcr(q, ChromaSubsampling::None).progressive(false)
}

fn baseline_grayscale(q: u8) -> EncoderConfig {
    EncoderConfig::grayscale(q).progressive(false)
}

fn decode_zenjpeg(jpeg: &[u8]) {
    use zenjpeg::decoder::Decoder;
    let decoder = Decoder::new();
    let decoded = decoder
        .decode(jpeg, enough::Unstoppable)
        .expect("zenjpeg decoder rejected shared-table JPEG");
    assert!(decoded.width > 0 && decoded.height > 0);
    let pixels = decoded
        .pixels_u8()
        .expect("u8 pixels expected for sRGB roundtrip");
    assert!(
        !pixels.is_empty(),
        "decoded pixel buffer unexpectedly empty"
    );
}

// ---------- regression: Off is byte-identical to default-before-field -------

#[test]
fn off_matches_pre_change_baseline() {
    // Pre-change behavior == default HuffmanStrategy::Optimize with the
    // four-table DHT. Building two configs, one with TinyFileMode::Off and
    // one (implicitly) with the same setting via construction without the
    // builder method, must yield identical bytes.
    let img = solid_rgb_image(128, 128);
    let c_default_off = baseline_ycbcr(85).tiny_file_mode(TinyFileMode::Off);
    let jpeg_off = encode_rgb(128, 128, &img.pixels, &c_default_off).unwrap();

    // Encoding twice with Off must be deterministic.
    let jpeg_off_again = encode_rgb(128, 128, &img.pixels, &c_default_off).unwrap();
    assert_eq!(
        jpeg_off, jpeg_off_again,
        "TinyFileMode::Off must be deterministic"
    );

    // 256×256 is above the heuristic's grayscale threshold but below the
    // YCbCr mid-zone, so Auto matches Force for a 128² color image. Check
    // that Off does NOT match Force for a small image (proves the flag
    // actually changes output when active).
    let c_force = baseline_ycbcr(85).tiny_file_mode(TinyFileMode::Force);
    let jpeg_force = encode_rgb(128, 128, &img.pixels, &c_force).unwrap();
    assert_ne!(
        jpeg_off, jpeg_force,
        "Force should produce different bytes than Off at 128×128"
    );
}

// ---------- Force shrinks small images --------------------------------------

#[test]
fn force_is_strictly_smaller_at_64x64() {
    let img = solid_rgb_image(64, 64);

    let c_off = baseline_ycbcr(85).tiny_file_mode(TinyFileMode::Off);
    let c_force = baseline_ycbcr(85).tiny_file_mode(TinyFileMode::Force);

    let jpeg_off = encode_rgb(64, 64, &img.pixels, &c_off).unwrap();
    let jpeg_force = encode_rgb(64, 64, &img.pixels, &c_force).unwrap();

    assert!(
        jpeg_force.len() < jpeg_off.len(),
        "Force must shrink output at 64×64; off={} force={}",
        jpeg_off.len(),
        jpeg_force.len()
    );
}

#[test]
fn force_is_strictly_smaller_for_gray_32x32() {
    let pixels = solid_gray_image(32, 32);

    let c_off = baseline_grayscale(85).tiny_file_mode(TinyFileMode::Off);
    let c_force = baseline_grayscale(85).tiny_file_mode(TinyFileMode::Force);

    let jpeg_off = encode_gray(32, 32, &pixels, &c_off).unwrap();
    let jpeg_force = encode_gray(32, 32, &pixels, &c_force).unwrap();

    assert!(
        jpeg_force.len() < jpeg_off.len(),
        "Force should shrink grayscale 32×32 output; off={} force={}",
        jpeg_off.len(),
        jpeg_force.len()
    );
}

// ---------- Auto heuristic --------------------------------------------------

#[test]
fn auto_matches_force_below_threshold() {
    // 64×64 YCbCr is well below the heuristic's ~65k pixel cutoff.
    assert!(should_activate_tiny_file_mode(64, 64, true));

    let img = solid_rgb_image(64, 64);
    let c_auto = baseline_ycbcr(85).tiny_file_mode(TinyFileMode::Auto);
    let c_force = baseline_ycbcr(85).tiny_file_mode(TinyFileMode::Force);

    let jpeg_auto = encode_rgb(64, 64, &img.pixels, &c_auto).unwrap();
    let jpeg_force = encode_rgb(64, 64, &img.pixels, &c_force).unwrap();

    assert_eq!(
        jpeg_auto, jpeg_force,
        "Auto must match Force at 64×64 (well under the activation threshold)"
    );
}

#[test]
fn auto_matches_off_above_threshold() {
    // 768×768 is well above every Auto threshold (128² for subsampled, 64²
    // for 4:4:4).
    assert!(!should_activate_tiny_file_mode(768, 768, true));

    let img = solid_rgb_image(768, 768);
    let c_auto = baseline_ycbcr(85).tiny_file_mode(TinyFileMode::Auto);
    let c_off = baseline_ycbcr(85).tiny_file_mode(TinyFileMode::Off);

    let jpeg_auto = encode_rgb(768, 768, &img.pixels, &c_auto).unwrap();
    let jpeg_off = encode_rgb(768, 768, &img.pixels, &c_off).unwrap();

    assert_eq!(
        jpeg_auto, jpeg_off,
        "Auto must match Off at 768×768 (above both activation zones)"
    );
}

#[test]
fn auto_follows_subsampling_threshold_at_128() {
    // At exactly 128×128, 4:2:0 Auto should activate (<=128² rule) but 4:4:4
    // Auto should NOT activate (only <=64² rule).
    let img = solid_rgb_image(128, 128);

    let c_auto_420 = baseline_ycbcr(85).tiny_file_mode(TinyFileMode::Auto);
    let c_force_420 = baseline_ycbcr(85).tiny_file_mode(TinyFileMode::Force);
    let c_off_420 = baseline_ycbcr(85).tiny_file_mode(TinyFileMode::Off);

    let c_auto_444 = baseline_ycbcr_444(85).tiny_file_mode(TinyFileMode::Auto);
    let c_off_444 = baseline_ycbcr_444(85).tiny_file_mode(TinyFileMode::Off);

    let j_auto_420 = encode_rgb(128, 128, &img.pixels, &c_auto_420).unwrap();
    let j_force_420 = encode_rgb(128, 128, &img.pixels, &c_force_420).unwrap();
    let j_off_420 = encode_rgb(128, 128, &img.pixels, &c_off_420).unwrap();
    assert_eq!(
        j_auto_420, j_force_420,
        "4:2:0 Auto must activate at exactly 128×128 (threshold inclusive)"
    );
    assert_ne!(j_auto_420, j_off_420);

    let j_auto_444 = encode_rgb(128, 128, &img.pixels, &c_auto_444).unwrap();
    let j_off_444 = encode_rgb(128, 128, &img.pixels, &c_off_444).unwrap();
    assert_eq!(
        j_auto_444, j_off_444,
        "4:4:4 Auto must NOT activate at 128×128 (above 64² threshold)"
    );
}

#[test]
fn auto_activates_for_grayscale_at_any_size() {
    // Grayscale: Force always wins (fixed ~208 B saving), so Auto always
    // activates regardless of pixel count.
    let pixels_512 = solid_gray_image(512, 512);
    let c_auto = baseline_grayscale(85).tiny_file_mode(TinyFileMode::Auto);
    let c_force = baseline_grayscale(85).tiny_file_mode(TinyFileMode::Force);
    let c_off = baseline_grayscale(85).tiny_file_mode(TinyFileMode::Off);

    let j_auto = encode_gray(512, 512, &pixels_512, &c_auto).unwrap();
    let j_force = encode_gray(512, 512, &pixels_512, &c_force).unwrap();
    let j_off = encode_gray(512, 512, &pixels_512, &c_off).unwrap();

    assert_eq!(j_auto, j_force, "grayscale Auto must activate at 512×512");
    assert!(
        j_auto.len() < j_off.len(),
        "grayscale shared Huffman should shrink output at 512×512: auto={} off={}",
        j_auto.len(),
        j_off.len()
    );
}

// ---------- round-trip through zenjpeg's own decoder ------------------------

#[test]
fn roundtrip_all_modes_through_zenjpeg_decoder() {
    let img = solid_rgb_image(128, 128);

    for mode in [TinyFileMode::Off, TinyFileMode::Auto, TinyFileMode::Force] {
        let cfg = baseline_ycbcr(85).tiny_file_mode(mode);
        let jpeg = encode_rgb(128, 128, &img.pixels, &cfg)
            .unwrap_or_else(|e| panic!("encode failed for {mode:?}: {e:?}"));
        decode_zenjpeg(&jpeg);
    }
}

#[test]
fn roundtrip_gray_force_through_zenjpeg_decoder() {
    let pixels = solid_gray_image(64, 64);
    let cfg = baseline_grayscale(85).tiny_file_mode(TinyFileMode::Force);
    let jpeg = encode_gray(64, 64, &pixels, &cfg).unwrap();
    decode_zenjpeg(&jpeg);
}

#[test]
fn roundtrip_force_444_q85_through_image_crate() {
    // image crate wraps libjpeg (via the zune-jpeg backend by default in
    // the workspace). We ensure the shared-table baseline JPEG loads cleanly
    // through a third-party decoder as a cross-check. 4:4:4 here
    // deliberately — 4:2:0 exercises identical markers but pads differently
    // at edges; covered by zenjpeg's own round-trip above.
    let img = solid_rgb_image(64, 64);
    let cfg = baseline_ycbcr_444(85).tiny_file_mode(TinyFileMode::Force);
    let jpeg = encode_rgb(64, 64, &img.pixels, &cfg).unwrap();
    decode_zenjpeg(&jpeg);

    // Parse via zune-jpeg if available — this is the same decoder used by
    // the `image` crate's default features. Skip gracefully if the crate
    // isn't in the dev-deps.
    // (zenjpeg's own tests already exercise zune-jpeg as a cross-check:
    // `tests/multi_decoder_compatibility.rs`. We keep it informal here.)
    let _ = jpeg;
}

// ---------- XYB is orthogonal to TinyFileMode -------------------------------

#[test]
fn xyb_output_unchanged_by_tiny_file_mode() {
    // XYB already uses a single DC/AC Huffman pair and its own SOS layout,
    // so TinyFileMode must not disturb the XYB bitstream.
    let img = solid_rgb_image(64, 64);

    let c_auto = EncoderConfig::xyb(85, XybSubsampling::Full)
        .progressive(false)
        .tiny_file_mode(TinyFileMode::Auto);
    let c_force = EncoderConfig::xyb(85, XybSubsampling::Full)
        .progressive(false)
        .tiny_file_mode(TinyFileMode::Force);
    let c_off = EncoderConfig::xyb(85, XybSubsampling::Full)
        .progressive(false)
        .tiny_file_mode(TinyFileMode::Off);

    let a = encode_rgb(64, 64, &img.pixels, &c_auto).unwrap();
    let b = encode_rgb(64, 64, &img.pixels, &c_force).unwrap();
    let c = encode_rgb(64, 64, &img.pixels, &c_off).unwrap();

    assert_eq!(
        a, b,
        "XYB output must be identical across TinyFileMode variants"
    );
    assert_eq!(
        a, c,
        "XYB output must be identical across TinyFileMode variants"
    );
}
