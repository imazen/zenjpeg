//! zencodec conformance: truncated/EOF input must categorize as incomplete.
//!
//! `zencodec_testkit::check_decode_truncation_series` feeds a known-good
//! encoded JPEG, truncates it at a deterministic series of prefixes, decodes
//! each through the dyn-erased `DecoderConfig` path, and asserts the resulting
//! `ErrorCategory` is in the incomplete-input set — never a panic, OOM, or
//! `Internal`. This is the truncation/EOF half of the codec taxonomy check
//! merged in zencodec PR #112; wired here per the owner's request that every
//! codec's test suite exercise it.
//!
//! Gated on `zencodec` (the trait-impl feature), matching the sibling
//! `emit_integration` / `orientation_descriptor` bundled tests. CI runs this
//! feature (`.github/workflows/ci.yml` — "Test (all user features)" includes
//! `zencodec`), so this test is not a silent skip.

#![cfg(feature = "zencodec")]

use zencodec::encode::{EncodeJob as _, Encoder as _, EncoderConfig as _};
use zenjpeg::{JpegDecoderConfig, JpegEncoderConfig};
use zenpixels::{PixelDescriptor, PixelSlice};

/// A small, valid baseline JPEG produced through the zencodec trait chain.
/// 8×8 tightly-packed RGB8 sRGB checkerboard (one full MCU) — big enough to
/// carry real SOI/SOF/DQT/DHT/SOS/EOI structure for the truncation walker.
fn valid_jpeg() -> Vec<u8> {
    let (w, h) = (8u32, 8u32);
    let mut pixels = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let on = (x + y) % 2 == 0;
            let (r, g, b) = if on { (200, 40, 40) } else { (40, 40, 200) };
            pixels.extend_from_slice(&[r, g, b]);
        }
    }
    let slice = PixelSlice::new(&pixels, w, h, (w * 3) as usize, PixelDescriptor::RGB8_SRGB)
        .expect("rgb8 slice");
    JpegEncoderConfig::new()
        .job()
        .encoder()
        .expect("encoder build")
        .encode(slice)
        .expect("encode")
        .into_vec()
}

#[test]
fn truncation_series_categorizes_as_incomplete_input() {
    let valid = valid_jpeg();
    zencodec_testkit::check_decode_truncation_series(JpegDecoderConfig::new(), &valid)
        .expect("truncated JPEG input must categorize as incomplete, never panic/OOM/Internal");
}
