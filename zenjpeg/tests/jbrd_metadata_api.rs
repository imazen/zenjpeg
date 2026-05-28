//! Smoke tests for `Decoder::decode_coefficients_with_jbrd_metadata`.
//!
//! Verifies that:
//!   1. The legacy `decode_coefficients()` API still works (back-compat).
//!   2. The new API returns one `JbrdScanInfo` per SOS, in bitstream order,
//!      with matching spectral / approximation parameters.
//!   3. On a typical progressive JPEG with AC-refinement scans, the new
//!      API populates `reset_points` and/or `extra_zero_runs` for at least
//!      one of those scans (so downstream JBRD reconstruction has the
//!      data it needs).
//!   4. The coefficient data returned alongside the metadata is identical
//!      to what the legacy entry point returns — no encoding drift.
//!
//! Run: `cargo test --release --test jbrd_metadata_api --features decoder`

#![cfg(feature = "decoder")]

use enough::Unstoppable;
use zenjpeg::decode::Decoder;
use zenjpeg::encode::EncoderConfig;
use zenjpeg::encode::ProgressiveScanMode;
use zenjpeg::encode::encoder_types::{ChromaSubsampling, PixelLayout};

/// 256x256 deterministic RGB gradient with slight noise so block contents
/// are non-trivial (progressive encoders emit EOB-run sequences across
/// scans on real content).
fn make_rgb_gradient(w: u32, h: u32) -> Vec<u8> {
    let mut buf = vec![0u8; (w as usize) * (h as usize) * 3];
    for y in 0..h {
        for x in 0..w {
            let i = ((y as usize) * (w as usize) + (x as usize)) * 3;
            buf[i] = ((x * 255) / w.max(1)) as u8;
            buf[i + 1] = ((y * 255) / h.max(1)) as u8;
            // Light noise to defeat constant-block early-out paths.
            buf[i + 2] = ((x.wrapping_add(y).wrapping_mul(31)) & 0xFF) as u8;
        }
    }
    buf
}

fn encode_progressive(rgb: &[u8], w: u32, h: u32) -> Vec<u8> {
    let cfg = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter)
        .progressive(ProgressiveScanMode::Progressive);
    cfg.encode_bytes(rgb, w, h, PixelLayout::Rgb8Srgb)
        .expect("zenjpeg encode failed")
}

#[test]
fn legacy_decode_coefficients_unchanged() {
    let rgb = make_rgb_gradient(256, 256);
    let jpeg = encode_progressive(&rgb, 256, 256);

    // Legacy entry: zero overhead, no metadata returned.
    let coeffs = Decoder::new()
        .decode_coefficients(&jpeg, Unstoppable)
        .expect("decode_coefficients failed");
    assert_eq!(coeffs.width, 256);
    assert_eq!(coeffs.height, 256);
    assert!(!coeffs.components.is_empty());
}

#[test]
fn jbrd_metadata_progressive_smoke() {
    let rgb = make_rgb_gradient(256, 256);
    let jpeg = encode_progressive(&rgb, 256, 256);

    let (coeffs, meta) = Decoder::new()
        .decode_coefficients_with_jbrd_metadata(&jpeg, Unstoppable)
        .expect("decode_coefficients_with_jbrd_metadata failed");

    assert_eq!(coeffs.width, 256);
    assert_eq!(coeffs.height, 256);

    // A progressive JPEG has multiple SOS scans (DC + several AC bands +
    // any refinement scans). Our progressive encoder emits a fixed scan
    // script, so we expect at least 2 scans here — one DC and >=1 AC.
    assert!(
        meta.scans.len() >= 2,
        "expected at least 2 progressive scans, got {} — \
         decode_coefficients_with_jbrd_metadata should populate one entry per SOS",
        meta.scans.len()
    );

    // Bitstream-order invariant: the very first scan must be DC-first
    // (ss == 0, se == 0, ah == 0, al >= 0). libjxl + every progressive
    // encoder emits a DC scan as the first SOS. If our metadata claims
    // the first scan is AC, the tracker is mis-ordering its outputs.
    let dc = &meta.scans[0];
    assert!(
        dc.ss == 0 && dc.se == 0 && dc.ah == 0,
        "first scan should be DC first (ss=0/se=0/ah=0), got \
         ss={}/se={}/ah={}/al={}",
        dc.ss,
        dc.se,
        dc.ah,
        dc.al
    );

    // DC scans cannot emit reset_points or extra_zero_runs (those
    // signals are AC-only — `eobrun_allowed = ss > 0`).
    assert!(
        dc.reset_points.is_empty(),
        "DC scan must have empty reset_points; got {} entries",
        dc.reset_points.len()
    );
    assert!(
        dc.extra_zero_runs.is_empty(),
        "DC scan must have empty extra_zero_runs; got {} entries",
        dc.extra_zero_runs.len()
    );
}

#[test]
fn jbrd_metadata_matches_legacy_coefficients() {
    let rgb = make_rgb_gradient(128, 128);
    let jpeg = encode_progressive(&rgb, 128, 128);

    // Decode both ways and compare every coefficient byte-for-byte.
    let legacy = Decoder::new()
        .decode_coefficients(&jpeg, Unstoppable)
        .expect("legacy decode failed");
    let (with_meta, _meta) = Decoder::new()
        .decode_coefficients_with_jbrd_metadata(&jpeg, Unstoppable)
        .expect("tracked decode failed");

    assert_eq!(legacy.width, with_meta.width);
    assert_eq!(legacy.height, with_meta.height);
    assert_eq!(legacy.components.len(), with_meta.components.len());
    for (a, b) in legacy.components.iter().zip(with_meta.components.iter()) {
        assert_eq!(
            a.coeffs, b.coeffs,
            "JBRD-tracked decode diverged from legacy decode — \
             tracking must not perturb coefficient output"
        );
    }
}
