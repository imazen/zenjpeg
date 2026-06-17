//! Regression test for #155: a downstream caller can NAME `ErrorKind` variants
//! to classify decode errors.
//!
//! This deliberately uses ONLY the public re-export path
//! (`zenjpeg::decoder::ErrorKind`) — the same surface a server crate sees —
//! so that demoting the re-export back to `pub(crate)` would fail to compile.

use zenjpeg::decoder::{Decoder, ErrorKind};
use zenjpeg::encoder::Unstoppable;

/// `Error::kind()` returns a `&ErrorKind` whose variants can be matched by name.
#[test]
fn error_kind_variants_are_nameable_and_matchable() {
    // Garbage input — not a JPEG — produces a datastream error.
    let not_a_jpeg = [0u8; 16];
    let err = Decoder::new()
        .decode(&not_a_jpeg, Unstoppable)
        .expect_err("decoding non-JPEG bytes must fail");

    // The whole point of #155: a downstream can `match` on named variants to
    // map errors onto e.g. HTTP status codes, instead of substring-matching
    // `to_string()`. Match against several named variants.
    let classified = match err.kind() {
        ErrorKind::ImageTooLarge { .. } => "413",
        ErrorKind::AllocationFailed { .. } => "500",
        ErrorKind::InvalidJpegData { .. }
        | ErrorKind::InvalidMarker { .. }
        | ErrorKind::TruncatedData { .. }
        | ErrorKind::InvalidBufferSize { .. } => "400",
        // `ErrorKind` is `#[non_exhaustive]`; downstream code must keep a
        // catch-all. This arm proves the type is genuinely non-exhaustive
        // from outside the crate.
        _ => "400",
    };

    assert_eq!(
        classified, "400",
        "garbage input should classify as a 400-class error"
    );
}

/// A decompression-bomb rejection produces a nameable `ImageTooLarge` kind.
#[test]
fn max_pixels_limit_is_classifiable() {
    use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};

    // Encode a small image, then decode it with an impossibly low pixel cap.
    let pixels = [200u8; 32 * 32 * 3];
    let mut enc = EncoderConfig::ycbcr(80.0, ChromaSubsampling::Quarter)
        .encode_from_bytes(32, 32, PixelLayout::Rgb8Srgb)
        .expect("config builds");
    enc.push_packed(&pixels, Unstoppable).expect("push rows");
    let jpeg = enc.finish().expect("encode");

    let err = Decoder::new()
        .max_pixels(16) // 32x32 = 1024 px > 16
        .decode(&jpeg, Unstoppable)
        .expect_err("decode must reject input over the pixel cap");

    assert!(
        matches!(err.kind(), ErrorKind::ImageTooLarge { .. }),
        "over-cap decode should be ImageTooLarge, got: {:?}",
        err.kind()
    );
}
