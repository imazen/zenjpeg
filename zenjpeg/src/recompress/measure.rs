//! zensim-A measurement helpers used when [`crate::recompress::api::Budget`]
//! permits an IQA pass.
//!
//! `score_recompression(source_jpeg, recompressed_jpeg)` returns the
//! zensim-A of the recompressed bytes vs the **source** bytes. That
//! is the generation-loss signal — *not* the cumulative score against
//! the unknown reference. The router uses calibration to convert
//! generation loss + estimated source-cumulative into projected
//! cumulative zensim-A.

use crate::decoder::{DecodeConfig, OutputTarget};
use enough::Unstoppable;
use zensim::{RgbSlice, Zensim, ZensimProfile};

use crate::recompress::error::Error;

/// The zensim profile the calibration tables were fit against and the
/// dial the recompressor targets: **Profile A** (`ZensimProfile::A`), the
/// user-facing general-purpose profile in the local zensim. Pinned to the
/// `A` variant explicitly — NOT `ZensimProfile::latest()` — so a future
/// zensim rotation of `latest()` cannot silently shift the metric scale
/// out from under the baked GEXP / achieved-quality tables. Profile A is a
/// distinct profile; it is NOT any `PreviewV0_*` variant.
const PROFILE: ZensimProfile = ZensimProfile::A;

/// Score a recompressed JPEG against the source JPEG it came from.
/// Both inputs are decoded to RGB8 internally. Returns the [`PROFILE`]
/// score in `[0, 100]`.
pub fn score_recompression(source_jpeg: &[u8], recompressed_jpeg: &[u8]) -> Result<f32, Error> {
    let source = decode_rgb8(source_jpeg)?;
    let dist = decode_rgb8(recompressed_jpeg)?;
    if source.width != dist.width || source.height != dist.height {
        return Err(Error::Internal("source and recompressed dimensions differ"));
    }
    score_pixels_pair(&source, &dist)
}

/// Score a recompressed JPEG against a *known* reference (the
/// original uncompressed pixels). The caller is responsible for
/// passing RGB8 pixels at the same dimensions as the JPEG.
pub fn score_against_reference(
    reference_rgb8: &[u8],
    width: u32,
    height: u32,
    recompressed_jpeg: &[u8],
) -> Result<f32, Error> {
    let dist = decode_rgb8(recompressed_jpeg)?;
    if dist.width != width || dist.height != height {
        return Err(Error::Internal(
            "reference and recompressed dimensions differ",
        ));
    }
    let reference = DecodedRgb8 {
        width,
        height,
        bytes: reference_rgb8.to_vec(),
    };
    score_pixels_pair(&reference, &dist)
}

struct DecodedRgb8 {
    width: u32,
    height: u32,
    bytes: Vec<u8>,
}

fn decode_rgb8(jpeg: &[u8]) -> Result<DecodedRgb8, Error> {
    let cfg = DecodeConfig::new().output_target(OutputTarget::Srgb8);
    let decoded = cfg
        .decode(jpeg, Unstoppable)
        .map_err(|e| Error::Zenjpeg(format!("decode for scoring: {e}")))?;
    let pixels = decoded
        .pixels_u8()
        .ok_or(Error::Internal("decoded u8 pixels missing"))?
        .to_vec();
    Ok(DecodedRgb8 {
        width: decoded.width,
        height: decoded.height,
        bytes: pixels,
    })
}

fn score_pixels_pair(a: &DecodedRgb8, b: &DecodedRgb8) -> Result<f32, Error> {
    let w = a.width as usize;
    let h = a.height as usize;
    if a.bytes.len() != w * h * 3 || b.bytes.len() != w * h * 3 {
        return Err(Error::Internal(
            "rgb8 byte length mismatch with declared dimensions",
        ));
    }
    // zensim wants `&[[u8; 3]]`. Bytemuck-style cast via chunks_exact.
    let a_pixels: &[[u8; 3]] = bytemuck_rgb_view(&a.bytes);
    let b_pixels: &[[u8; 3]] = bytemuck_rgb_view(&b.bytes);
    let source = RgbSlice::new(a_pixels, w, h);
    let dist = RgbSlice::new(b_pixels, w, h);
    let z = Zensim::new(PROFILE);
    let result = z
        .compute(&source, &dist)
        .map_err(|e| Error::Zensim(format!("{e}")))?;
    Ok(result.score() as f32)
}

/// View a `&[u8]` of length `3 * N` as `&[[u8; 3]; N]` cast to slice.
/// `[u8; 3]` has the same memory layout as 3 consecutive `u8`s with
/// no padding, so this is sound provided length divides by 3.
fn bytemuck_rgb_view(bytes: &[u8]) -> &[[u8; 3]] {
    // Manual cast that avoids the bytemuck dep — `[u8; 3]` has align 1
    // and size 3, so the layout is the same as `[u8]` chunked by 3.
    // Using `std::slice::from_raw_parts` would need `unsafe`; we
    // forbid that crate-wide. Use `as_chunks` (stable since Rust 1.88).
    let (chunks, rem) = bytes.as_chunks::<3>();
    debug_assert!(rem.is_empty(), "rgb8 byte length is not divisible by 3");
    chunks
}
