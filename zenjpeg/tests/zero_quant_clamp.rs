//! Tests for zero quantization value clamping (issue #42).
//!
//! IJG libjpeg-9b at Q=1 produces quantization tables with zero values.
//! libjpeg-turbo silently clamps these to 1. zenjpeg should do the same
//! in non-Strict modes (Balanced, Lenient, Permissive).

#![cfg(feature = "decoder")]

use enough::Unstoppable;
use zenjpeg::decoder::{Decoder, Strictness};

/// Fixture: 1x1 grayscale JPEG from IJG libjpeg-9b at Q=1 with zero quant values.
const FIXTURE: &[u8] = include_bytes!("testdata/all_the_images/quant_zero_libjpeg9b.jpg");

#[test]
fn balanced_decodes_zero_quant() {
    // Balanced is the default strictness
    let result = Decoder::new().decode(FIXTURE, Unstoppable);
    assert!(
        result.is_ok(),
        "Balanced mode should clamp zero quant to 1, got: {:?}",
        result.err()
    );
}

#[test]
fn lenient_decodes_zero_quant() {
    let result = Decoder::new()
        .strictness(Strictness::Lenient)
        .decode(FIXTURE, Unstoppable);
    assert!(
        result.is_ok(),
        "Lenient mode should clamp zero quant to 1, got: {:?}",
        result.err()
    );
}

#[test]
fn permissive_decodes_zero_quant() {
    let result = Decoder::new()
        .strictness(Strictness::Permissive)
        .decode(FIXTURE, Unstoppable);
    assert!(
        result.is_ok(),
        "Permissive mode should clamp zero quant to 1, got: {:?}",
        result.err()
    );
}

#[test]
fn strict_rejects_zero_quant() {
    let result = Decoder::new()
        .strictness(Strictness::Strict)
        .decode(FIXTURE, Unstoppable);
    assert!(
        result.is_err(),
        "Strict mode should reject zero quant values"
    );
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("quantization value is zero"),
        "Error should mention zero quant value, got: {err}"
    );
}
