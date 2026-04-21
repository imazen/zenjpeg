//! Integration coverage for the public `BoundaryRd` API (issue #91).
//! See also `boundary_rd_hash_lock.rs` for the byte-identity
//! default-path gate.
//!
//! This test lives on the public surface: `EncoderConfig::boundary_rd`
//! is the only addition. Automatic per-image-class preset selection is
//! deferred to issue #103.

use enough::Unstoppable;
use zenjpeg::encoder::{
    BoundaryRd, BoundaryRdConfig, ChromaSubsampling, EncoderConfig, PixelLayout,
};

/// A small noise+patches image — the CLAUDE.md-approved test generator.
fn gen_noise_patches(w: usize, h: usize, seed: u64) -> Vec<u8> {
    let mut out = vec![0u8; w * h * 3];
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    for y in 0..h {
        for x in 0..w {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            let n = s.wrapping_mul(0x2545_F491_4F6C_DD1D);
            let noise = ((n >> 32) & 0xFF) as u8;
            let patch_y = y / 24 % 2;
            let patch_x = x / 24 % 2;
            let base: u8 = if patch_y == patch_x { 200 } else { 40 };
            let v = base.saturating_add(noise / 8);
            let i = (y * w + x) * 3;
            out[i] = v;
            out[i + 1] = v;
            out[i + 2] = v;
        }
    }
    out
}

/// Bright checkerboard — produces strong seam artifacts that boundary-RD
/// specifically targets.
fn gen_checkerboard(w: usize, h: usize, cell: usize) -> Vec<u8> {
    let mut out = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let on = ((x / cell) + (y / cell)) % 2 == 0;
            let v: u8 = if on { 240 } else { 20 };
            let i = (y * w + x) * 3;
            out[i] = v;
            out[i + 1] = v;
            out[i + 2] = v;
        }
    }
    out
}

fn encode_rgb8(rgb: &[u8], w: u32, h: u32, cfg: EncoderConfig) -> Vec<u8> {
    let mut enc = cfg
        .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
        .expect("encode config");
    enc.push_packed(rgb, Unstoppable).expect("push rows");
    enc.finish().expect("finish")
}

fn decode_rgb8(jpeg: &[u8]) -> (Vec<u8>, u32, u32) {
    use zune_core::bytestream::ZCursor;
    use zune_core::colorspace::ColorSpace;
    use zune_core::options::DecoderOptions;
    use zune_jpeg::JpegDecoder;

    let options = DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::RGB);
    let mut decoder = JpegDecoder::new_with_options(ZCursor::new(jpeg), options);
    let pixels = decoder.decode().expect("decode");
    let (w, h) = decoder.dimensions().expect("dimensions");
    (pixels, w as u32, h as u32)
}

// ---------------------------------------------------------------------------
// Default-path byte identity (the single-most-important guarantee).
// ---------------------------------------------------------------------------

/// `BoundaryRd::Off` (the default) must produce byte-for-byte identical
/// output to a config that never touched the boundary-RD API at all.
/// Guards against accidental hot-path plumbing regressions.
#[test]
fn off_equals_baseline_byte_identical() {
    let (w, h) = (128usize, 128usize);
    let rgb = gen_noise_patches(w, h, 0x5151_abcd);
    let baseline = EncoderConfig::ycbcr(85f32, ChromaSubsampling::Quarter);
    let with_off =
        EncoderConfig::ycbcr(85f32, ChromaSubsampling::Quarter).boundary_rd(BoundaryRd::Off);

    let a = encode_rgb8(&rgb, w as u32, h as u32, baseline);
    let b = encode_rgb8(&rgb, w as u32, h as u32, with_off);
    assert_eq!(a, b, "BoundaryRd::Off must be byte-identical to default");
}

// ---------------------------------------------------------------------------
// On(default config) produces decodable JPEGs of the expected dimensions.
// ---------------------------------------------------------------------------

#[test]
fn on_default_config_decodes() {
    let (w, h) = (128usize, 128usize);
    let rgb = gen_noise_patches(w, h, 0xdead_1234);
    let cfg = EncoderConfig::ycbcr(85f32, ChromaSubsampling::Quarter)
        .boundary_rd(BoundaryRd::On(BoundaryRdConfig::default()));
    let bytes = encode_rgb8(&rgb, w as u32, h as u32, cfg);
    let (dec, dw, dh) = decode_rgb8(&bytes);
    assert_eq!((dw, dh), (w as u32, h as u32));
    assert_eq!(dec.len(), w * h * 3);
}

// ---------------------------------------------------------------------------
// Several manually-tuned configs produce decodable output.
// ---------------------------------------------------------------------------

#[test]
fn manual_configs_decode() {
    let (w, h) = (128usize, 128usize);
    let rgb = gen_noise_patches(w, h, 0xbeef_babe);
    // Covers the "photo-mild", "phase5 left-only", and "phase5
    // left+above" points of interest from the #91 tuning evidence
    // (see benchmarks/boundary_rd_*_2026-04-20.md).
    let configs = [
        BoundaryRdConfig::new()
            .with_alpha(1.0)
            .with_threshold(0.1)
            .with_aq_shrink(0.7)
            .with_max_retries(1)
            .with_above(false),
        BoundaryRdConfig::new()
            .with_alpha(1.0)
            .with_threshold(0.05)
            .with_aq_shrink(0.5)
            .with_max_retries(2)
            .with_above(false),
        BoundaryRdConfig::new()
            .with_alpha(1.0)
            .with_threshold(0.05)
            .with_aq_shrink(0.5)
            .with_max_retries(2)
            .with_above(true),
    ];
    for cfg in configs {
        let encoder_cfg =
            EncoderConfig::ycbcr(85f32, ChromaSubsampling::Quarter).boundary_rd(BoundaryRd::On(cfg));
        let bytes = encode_rgb8(&rgb, w as u32, h as u32, encoder_cfg);
        let (dec, dw, dh) = decode_rgb8(&bytes);
        assert_eq!((dw, dh), (w as u32, h as u32));
        assert_eq!(dec.len(), w * h * 3);
    }
}

// ---------------------------------------------------------------------------
// Manual override with strong alpha + above works.
// ---------------------------------------------------------------------------

#[test]
fn manual_config_decodes() {
    let (w, h) = (128usize, 128usize);
    let rgb = gen_noise_patches(w, h, 42);
    let cfg = EncoderConfig::ycbcr(85f32, ChromaSubsampling::Quarter).boundary_rd(BoundaryRd::On(
        BoundaryRdConfig::new()
            .with_alpha(2.0)
            .with_threshold(0.05)
            .with_aq_shrink(0.5)
            .with_max_retries(2)
            .with_above(true),
    ));
    let bytes = encode_rgb8(&rgb, w as u32, h as u32, cfg);
    let (dec, _, _) = decode_rgb8(&bytes);
    assert_eq!(dec.len(), w * h * 3);
}

// ---------------------------------------------------------------------------
// Functional behavior: enabling boundary-RD with the aggressive
// left+above preset on a checkerboard must produce different bytes from
// Off. We don't measure BBS directly here — the internal metric is
// `pub(crate)` and gated behind `__test-utils`. The committed CSVs at
// `benchmarks/` are the quantitative record.
// ---------------------------------------------------------------------------

#[test]
fn on_left_above_differs_from_off_on_checkerboard() {
    let (w, h) = (128usize, 128usize);
    let rgb = gen_checkerboard(w, h, 10);

    let off = EncoderConfig::ycbcr(80f32, ChromaSubsampling::Quarter).boundary_rd(BoundaryRd::Off);
    let on = EncoderConfig::ycbcr(80f32, ChromaSubsampling::Quarter).boundary_rd(BoundaryRd::On(
        BoundaryRdConfig::new()
            .with_alpha(1.0)
            .with_threshold(0.05)
            .with_aq_shrink(0.5)
            .with_max_retries(2)
            .with_above(true),
    ));

    let bytes_off = encode_rgb8(&rgb, w as u32, h as u32, off);
    let bytes_on = encode_rgb8(&rgb, w as u32, h as u32, on);
    assert_ne!(
        bytes_off, bytes_on,
        "BoundaryRd::On(left+above) must change output on a checkerboard"
    );

    // And both must still decode.
    let (dec_off, _, _) = decode_rgb8(&bytes_off);
    let (dec_on, _, _) = decode_rgb8(&bytes_on);
    assert_eq!(dec_off.len(), w * h * 3);
    assert_eq!(dec_on.len(), w * h * 3);
}
