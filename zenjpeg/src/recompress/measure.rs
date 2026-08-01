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
///
/// Part of the `recompress-expert` surface. The closed loop itself now
/// measures through [`MeasureCtx`] (which builds the reference pyramid
/// once instead of per pass), so without `recompress-expert` this is
/// intentionally unreferenced.
#[cfg_attr(not(feature = "recompress-expert"), allow(dead_code))]
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
///
/// Part of the `recompress-expert` surface (see [`score_recompression`]).
#[cfg_attr(not(feature = "recompress-expert"), allow(dead_code))]
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
    use zencodec::CategorizedError;
    let cfg = DecodeConfig::new().output_target(OutputTarget::Srgb8);
    let decoded = cfg.decode(jpeg, Unstoppable).map_err(|e| {
        // Capture the decode error's real category before flattening it to a
        // `String` — avoids collapsing every decode failure to `Internal`.
        let category = e.category();
        Error::ZenjpegCategorized {
            message: format!("decode for scoring: {e}"),
            category,
        }
    })?;
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

#[cfg_attr(not(feature = "recompress-expert"), allow(dead_code))]
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

/// Per-8×8-block mean of a full-resolution zensim diffmap, on the
/// **full-blocks-only** grid: `blocks_w = width / 8`, `blocks_h =
/// height / 8` (truncating). Right/bottom edge slivers of non-multiple-
/// of-8 images have no entry — consumers must treat out-of-grid blocks
/// as unmeasured (and leave them alone).
#[derive(Debug, Clone)]
pub(crate) struct BlockErrorMap {
    /// Row-major `blocks_w × blocks_h` per-block mean diffmap values.
    pub(crate) errors: Vec<f32>,
    pub(crate) blocks_w: usize,
    pub(crate) blocks_h: usize,
}

impl BlockErrorMap {
    /// Measured error for the block at `(bx, by)` in *coefficient-grid*
    /// coordinates, or `None` when the block lies outside the measured
    /// grid (MCU padding / partial edge blocks).
    pub(crate) fn get(&self, bx: usize, by: usize) -> Option<f32> {
        if bx < self.blocks_w && by < self.blocks_h {
            Some(self.errors[by * self.blocks_w + bx])
        } else {
            None
        }
    }
}

/// Reusable measurement context for a closed-loop `recompress` call:
/// the source decoded once + its zensim reference pyramid built once,
/// then scored against any number of candidate JPEGs.
///
/// This replaces per-pass [`score_recompression`] in the closed loop —
/// that helper re-decodes the source and rebuilds the reference pyramid
/// on every call, which the loop paid once per pass. Scores are computed
/// with the same [`PROFILE`] (Profile A) the calibration tables were fit
/// against; `compute_with_ref_and_diffmap` additionally yields the
/// per-pixel diffmap that the per-block refinement pass consumes.
pub(crate) struct MeasureCtx {
    z: Zensim,
    pre: zensim::PrecomputedReference,
    width: u32,
    height: u32,
}

impl MeasureCtx {
    /// Decode `source_jpeg` and build the reference pyramid.
    pub(crate) fn new(source_jpeg: &[u8]) -> Result<Self, Error> {
        let source = decode_rgb8(source_jpeg)?;
        let w = source.width as usize;
        let h = source.height as usize;
        if source.bytes.len() != w * h * 3 {
            return Err(Error::Internal(
                "rgb8 byte length mismatch with declared dimensions",
            ));
        }
        let pixels = bytemuck_rgb_view(&source.bytes);
        let slice = RgbSlice::new(pixels, w, h);
        let z = Zensim::new(PROFILE);
        let pre = z
            .precompute_reference(&slice)
            .map_err(|e| Error::Zensim(format!("precompute reference: {e}")))?;
        Ok(Self {
            z,
            pre,
            width: source.width,
            height: source.height,
        })
    }

    /// Decode `candidate_jpeg`, score it against the source (generation
    /// loss, [`PROFILE`] scale), and aggregate the diffmap to per-block
    /// means.
    pub(crate) fn score_with_blocks(
        &self,
        candidate_jpeg: &[u8],
    ) -> Result<(f32, BlockErrorMap), Error> {
        use zensim::DiffmapWeighting;
        let dist = decode_rgb8(candidate_jpeg)?;
        if dist.width != self.width || dist.height != self.height {
            return Err(Error::Internal("source and recompressed dimensions differ"));
        }
        let w = self.width as usize;
        let h = self.height as usize;
        if dist.bytes.len() != w * h * 3 {
            return Err(Error::Internal(
                "rgb8 byte length mismatch with declared dimensions",
            ));
        }
        let pixels = bytemuck_rgb_view(&dist.bytes);
        let slice = RgbSlice::new(pixels, w, h);
        let res = self
            .z
            .compute_with_ref_and_diffmap(&self.pre, &slice, DiffmapWeighting::Trained)
            .map_err(|e| Error::Zensim(format!("compute_with_ref_and_diffmap: {e}")))?;
        let blocks = aggregate_diffmap_to_blocks(res.diffmap(), w, h);
        Ok((res.score() as f32, blocks))
    }
}

/// Aggregate a full-resolution diffmap (`width × height` f32, row-major)
/// into per-8×8-block means on the full-blocks-only grid. Same policy as
/// the target-zq loop's aggregation (`encode/zq.rs`): truncating grid,
/// edge slivers unmeasured.
fn aggregate_diffmap_to_blocks(diffmap: &[f32], width: usize, height: usize) -> BlockErrorMap {
    const BLOCK: usize = 8;
    let blocks_w = width / BLOCK;
    let blocks_h = height / BLOCK;
    let mut errors = Vec::with_capacity(blocks_w * blocks_h);
    for by in 0..blocks_h {
        for bx in 0..blocks_w {
            let mut sum = 0.0f64;
            for ly in 0..BLOCK {
                let y = by * BLOCK + ly;
                let row = &diffmap[y * width + bx * BLOCK..y * width + bx * BLOCK + BLOCK];
                for &v in row {
                    sum += v as f64;
                }
            }
            errors.push((sum / (BLOCK * BLOCK) as f64) as f32);
        }
    }
    BlockErrorMap {
        errors,
        blocks_w,
        blocks_h,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregate_diffmap_truncates_edge_slivers() {
        // 17×9 image → 2×1 full blocks; the 1-px right sliver and 1-px
        // bottom sliver are unmeasured.
        let width = 17;
        let height = 9;
        let mut dm = vec![0.0f32; width * height];
        // Left block all 1.0, right block all 3.0, slivers poisoned with
        // huge values that must NOT leak into any block mean.
        for y in 0..height {
            for x in 0..width {
                dm[y * width + x] = if y >= 8 || x >= 16 {
                    1000.0
                } else if x < 8 {
                    1.0
                } else {
                    3.0
                };
            }
        }
        let map = aggregate_diffmap_to_blocks(&dm, width, height);
        assert_eq!((map.blocks_w, map.blocks_h), (2, 1));
        assert!((map.get(0, 0).unwrap() - 1.0).abs() < 1e-6);
        assert!((map.get(1, 0).unwrap() - 3.0).abs() < 1e-6);
        assert_eq!(map.get(2, 0), None, "sliver column is unmeasured");
        assert_eq!(map.get(0, 1), None, "sliver row is unmeasured");
    }
}
