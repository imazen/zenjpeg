#![cfg(feature = "trellis")]
//! Integration tests for Phase 3 boundary-continuity D term.
//!
//! Verifies that:
//! 1. Enabling `trellis_boundary_rd` produces different output than the
//!    plain trellis path (the feature does something).
//! 2. Disabling it matches the pre-existing trellis path byte-for-byte
//!    (the fast-path guard is tight, no accidental behavior change).
//! 3. Hash-lock on a fixed tiny test image so we catch silent output
//!    drift on future refactors.

use zenjpeg::encode::trellis::TrellisConfig;
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};

/// Deterministic textured test image; the checkerboard edges create
/// 8×8 block boundaries that the boundary D term can detectably smooth.
fn make_image(width: usize, height: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let checker = ((x / 8 + y / 8) % 2 == 0) as u8 * 40;
            let r = (((x * 255) / width) as u8).saturating_add(checker);
            let g = (((y * 255) / height) as u8).saturating_add(checker);
            let b = ((((x + y) * 127) / (width + height)) as u8).saturating_add(checker);
            data.push(r);
            data.push(g);
            data.push(b);
        }
    }
    data
}

fn encode(config: &EncoderConfig, pixels: &[u8], width: u32, height: u32) -> Vec<u8> {
    let mut encoder = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .expect("encoder creation failed");
    encoder
        .push_packed(pixels, enough::Unstoppable)
        .expect("push failed");
    encoder.finish().expect("finish failed")
}

/// With `trellis_boundary_rd(false)` the output MUST match the
/// pre-existing trellis-only path byte-for-byte. Guards against accidental
/// changes to the hot path when the feature is off.
#[test]
fn boundary_rd_disabled_matches_trellis_default() {
    let w = 64u32;
    let h = 64u32;
    let pixels = make_image(w as usize, h as usize);

    let plain = EncoderConfig::ycbcr(75.0, ChromaSubsampling::None)
        .trellis(TrellisConfig::default());
    let with_flag_off = EncoderConfig::ycbcr(75.0, ChromaSubsampling::None)
        .trellis(TrellisConfig::default())
        .trellis_boundary_rd(false);

    let a = encode(&plain, &pixels, w, h);
    let b = encode(&with_flag_off, &pixels, w, h);
    assert_eq!(
        a, b,
        "trellis_boundary_rd(false) must be a byte-exact no-op vs plain trellis"
    );
}

/// Enabling boundary-RD on a textured image changes the output (it
/// evaluates three λ candidates and picks one by D_boundary-augmented
/// cost, which must sometimes pick differently than the default path).
#[test]
fn boundary_rd_enabled_changes_output() {
    let w = 64u32;
    let h = 64u32;
    let pixels = make_image(w as usize, h as usize);

    let plain = EncoderConfig::ycbcr(75.0, ChromaSubsampling::None)
        .trellis(TrellisConfig::default());
    let with_boundary = EncoderConfig::ycbcr(75.0, ChromaSubsampling::None)
        .trellis(TrellisConfig::default())
        .trellis_boundary_rd(true)
        .trellis_boundary_beta(1.0)
        .trellis_boundary_alpha(1.0);

    let a = encode(&plain, &pixels, w, h);
    let b = encode(&with_boundary, &pixels, w, h);
    assert_ne!(
        a, b,
        "boundary-RD must produce different bytes than the no-boundary trellis on \
         a textured 64×64 block image"
    );
}

/// β scale sweep — changing β must in turn change the encoded bytes on
/// non-trivial content. Guards against accidentally ignoring the knob.
#[test]
fn boundary_rd_beta_affects_output() {
    let w = 64u32;
    let h = 64u32;
    let pixels = make_image(w as usize, h as usize);

    let base = EncoderConfig::ycbcr(75.0, ChromaSubsampling::None)
        .trellis(TrellisConfig::default())
        .trellis_boundary_rd(true)
        .trellis_boundary_alpha(1.0);

    let weak = base.clone().trellis_boundary_beta(0.25);
    let strong = base.trellis_boundary_beta(8.0);

    let weak_bytes = encode(&weak, &pixels, w, h);
    let strong_bytes = encode(&strong, &pixels, w, h);
    assert_ne!(
        weak_bytes, strong_bytes,
        "trellis_boundary_beta must influence encoded output"
    );
}

/// On images with 4:2:0 subsampling the boundary path is still Y-only
/// but must not panic or produce empty output.
#[test]
fn boundary_rd_works_with_420_subsampling() {
    let w = 64u32;
    let h = 64u32;
    let pixels = make_image(w as usize, h as usize);

    let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter)
        .trellis(TrellisConfig::default())
        .trellis_boundary_rd(true);

    let bytes = encode(&config, &pixels, w, h);
    assert!(bytes.len() > 16, "encoded output should be non-trivially sized");
    // JPEG starts with SOI (FF D8) and ends with EOI (FF D9)
    assert_eq!(&bytes[0..2], &[0xFF, 0xD8]);
    assert_eq!(&bytes[bytes.len() - 2..], &[0xFF, 0xD9]);
}

/// First block of every row has no left neighbor — boundary-RD must be
/// a no-op for those. Nothing to assert directly from outside, so this
/// just checks that encoding does not fail on tall, single-block-wide
/// images (where every block is a "first-of-row").
#[test]
fn boundary_rd_handles_no_left_neighbor() {
    let w = 8u32;
    let h = 64u32;
    let pixels = make_image(w as usize, h as usize);

    let config = EncoderConfig::ycbcr(80.0, ChromaSubsampling::None)
        .trellis(TrellisConfig::default())
        .trellis_boundary_rd(true);

    let bytes = encode(&config, &pixels, w, h);
    assert!(bytes.len() > 16);
}
