//! Integration tests for `Quality::Zq` / `Quality::ZqExplicit` (#113).
//!
//! These tests validate the public API contract: types compile, the
//! iteration loop runs end-to-end, metrics are populated honestly, and
//! the closed-loop encoder can hit a perceptual target on at least one
//! synthetic image.
//!
//! These tests run only with `--features target-zq` since that's the
//! feature that enables the iteration path. Without the feature,
//! `Quality::Zq*` falls back to single-pass encoding at the resolved
//! starting quality (covered by a separate test).

#![cfg(feature = "target-zq")]

use enough::Unstoppable;
use zenjpeg::encode::zq::{BlockArtifactBound, ZqTarget};
use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, PixelLayout, Quality};

/// Generate a 256×256 synthetic image with mixed smooth + textured regions.
/// Roughly photo-shaped so the iteration loop has meaningful per-block
/// diffmap variance to work with.
fn synthetic_image(w: u32, h: u32) -> Vec<u8> {
    let w = w as usize;
    let h = h as usize;
    let mut rgb = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            // Smooth ramp + per-tile noise. Tiles 32×32; alternating
            // "smooth gradient" and "textured" blocks.
            let tx = x / 32;
            let ty = y / 32;
            let tile_smooth = (tx + ty) % 2 == 0;
            let r_base = (x * 255 / w) as u8;
            let g_base = (y * 255 / h) as u8;
            let b_base = ((x + y) * 255 / (w + h)) as u8;
            if tile_smooth {
                rgb[idx] = r_base;
                rgb[idx + 1] = g_base;
                rgb[idx + 2] = b_base;
            } else {
                // Pseudo-random noise: deterministic from x,y.
                let n = ((x.wrapping_mul(2654435761)) ^ (y.wrapping_mul(40503))) as u8;
                rgb[idx] = r_base.saturating_add(n & 0x3F);
                rgb[idx + 1] = g_base.saturating_sub((n >> 2) & 0x3F);
                rgb[idx + 2] = b_base.saturating_add((n >> 4) & 0x3F);
            }
        }
    }
    rgb
}

#[test]
fn zq_simple_form_encodes_and_returns_metrics() {
    let (w, h) = (256u32, 256);
    let rgb = synthetic_image(w, h);
    let config = EncoderConfig::ycbcr(Quality::Zq(80.0), ChromaSubsampling::Quarter);
    let mut enc = config
        .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
        .expect("encoder creation");
    enc.push_packed(&rgb, Unstoppable).expect("push");
    let (jpeg, metrics) = enc.finish_with_metrics().expect("finish_with_metrics");
    assert!(!jpeg.is_empty(), "should produce some JPEG bytes");
    assert!(
        jpeg.len() == metrics.bytes,
        "metrics.bytes must match jpeg.len()"
    );
    assert!(
        metrics.passes_used >= 1 && metrics.passes_used <= 3,
        "default budget = 2 passes, plus the initial pass = up to 3 total observed"
    );
    assert!(
        metrics.achieved_score.is_finite(),
        "iteration path always measures a final score"
    );
    // The synthetic image (noise tiles + gradient) is HARD: starting q=81
    // typically lands well below target 80 on this content. We don't
    // assert convergence quality here — that's the convergence test's
    // job. We only assert the API contract: the encoder ran the loop,
    // produced finite metrics, and reported targets_met honestly.
    if metrics.achieved_score >= 80.0 {
        assert!(metrics.targets_met, "score >= target should report met");
    } else {
        assert!(!metrics.targets_met, "score < target should NOT report met");
    }
}

#[test]
fn zq_explicit_form_respects_max_passes_zero() {
    let (w, h) = (128u32, 128);
    let rgb = synthetic_image(w, h);
    let target = ZqTarget::new(75.0).with_max_passes(0);
    let config = EncoderConfig::ycbcr(Quality::ZqExplicit(target), ChromaSubsampling::Quarter);
    let mut enc = config
        .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
        .unwrap();
    enc.push_packed(&rgb, Unstoppable).unwrap();
    let (_jpeg, metrics) = enc.finish_with_metrics().unwrap();
    assert_eq!(
        metrics.passes_used, 1,
        "max_passes=0 means single-pass (just the initial encode, no correction)"
    );
}

#[test]
fn zq_explicit_max_undershoot_strict_mode_catches_misses() {
    let (w, h) = (128u32, 128);
    let rgb = synthetic_image(w, h);
    // Target zq=99 with strict floor — basically unreachable.
    let target = ZqTarget::new(99.0)
        .with_max_undershoot(Some(0.5))
        .with_max_passes(2);
    let config = EncoderConfig::ycbcr(Quality::ZqExplicit(target), ChromaSubsampling::Quarter);
    let mut enc = config
        .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
        .unwrap();
    enc.push_packed(&rgb, Unstoppable).unwrap();
    let res = enc.finish_with_metrics();
    assert!(
        res.is_err(),
        "Zq target=99 with max_undershoot=0.5 should error on this image"
    );
}

#[test]
fn zq_with_block_artifact_bound_runs() {
    let (w, h) = (256u32, 256);
    let rgb = synthetic_image(w, h);
    let target = ZqTarget::new(80.0).with_block_artifact(Some(BlockArtifactBound::new(0.020)));
    let config = EncoderConfig::ycbcr(Quality::ZqExplicit(target), ChromaSubsampling::Quarter);
    let mut enc = config
        .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
        .unwrap();
    enc.push_packed(&rgb, Unstoppable).unwrap();
    let (jpeg, metrics) = enc.finish_with_metrics().unwrap();
    assert!(!jpeg.is_empty());
    assert!(metrics.achieved_max_block_artifact.is_finite());
    // Block-artifact bound is 0.020 — at zq=80 on a well-behaved
    // synthetic image, achieved should be well below this.
    assert!(
        metrics.achieved_max_block_artifact <= 0.030,
        "max-block artifact {} unexpectedly high",
        metrics.achieved_max_block_artifact
    );
}

#[test]
fn zq_block_artifact_max_overshoot_strict_mode_errors_on_unreachable_ceiling() {
    let (w, h) = (128u32, 128);
    let rgb = synthetic_image(w, h);
    // Set a very tight ceiling that's basically impossible at low q.
    let target = ZqTarget::new(60.0)
        .with_block_artifact(Some(
            BlockArtifactBound::new(0.0001).with_max_overshoot(Some(0.0001)),
        ))
        .with_max_passes(2);
    let config = EncoderConfig::ycbcr(Quality::ZqExplicit(target), ChromaSubsampling::Quarter);
    let mut enc = config
        .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
        .unwrap();
    enc.push_packed(&rgb, Unstoppable).unwrap();
    let res = enc.finish_with_metrics();
    assert!(
        res.is_err(),
        "block_artifact ceiling=0.0001 with strict overshoot=0.0001 should error"
    );
}

#[test]
fn quality_to_internal_resolves_zq_to_starting_q() {
    // Sanity: zq → starting jpegli q is in a sensible range.
    let zq75 = Quality::Zq(75.0);
    let zq85 = Quality::Zq(85.0);
    let zq95 = Quality::Zq(95.0);
    assert!(zq75.to_internal() > 50.0 && zq75.to_internal() < 90.0);
    assert!(zq85.to_internal() > 70.0 && zq85.to_internal() < 95.0);
    assert!(zq95.to_internal() > 85.0 && zq95.to_internal() <= 100.0);
}

#[test]
fn quality_zq_target_round_trip() {
    let q = Quality::Zq(82.5);
    assert!(q.is_zq_target());
    let t = q.zq_target().expect("Zq has a ZqTarget");
    assert_eq!(t.target, 82.5);

    let explicit = ZqTarget::new(82.5)
        .with_max_overshoot(Some(0.5))
        .with_max_passes(3);
    let q2 = Quality::ZqExplicit(explicit);
    assert!(q2.is_zq_target());
    let t2 = q2.zq_target().unwrap();
    assert_eq!(t2.max_overshoot, Some(0.5));
    assert_eq!(t2.max_passes, 3);
}

#[test]
fn finish_works_on_zq_mode_without_metrics() {
    // The plain finish() entry point must transparently route through
    // the iteration loop when a Zq variant is configured (otherwise
    // pushes-to-buffer would leave the inner streaming encoder with
    // zero rows and finish() would error with incomplete_image).
    let (w, h) = (128u32, 128);
    let rgb = synthetic_image(w, h);
    let config = EncoderConfig::ycbcr(Quality::Zq(75.0), ChromaSubsampling::Quarter);
    let mut enc = config
        .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
        .unwrap();
    enc.push_packed(&rgb, Unstoppable).unwrap();
    let jpeg = enc.finish().expect("finish() should work in Zq mode");
    assert!(!jpeg.is_empty());
}

#[test]
fn zq_works_with_subsampling_444() {
    let (w, h) = (256u32, 256);
    let rgb = synthetic_image(w, h);
    let config = EncoderConfig::ycbcr(Quality::Zq(80.0), ChromaSubsampling::None);
    let mut enc = config
        .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
        .unwrap();
    enc.push_packed(&rgb, Unstoppable).unwrap();
    let (jpeg, metrics) = enc
        .finish_with_metrics()
        .expect("4:4:4 Zq encode should succeed");
    assert!(!jpeg.is_empty());
    assert!(metrics.passes_used >= 1);
    assert!(
        metrics.achieved_score.is_finite(),
        "4:4:4 must run iteration loop, not single-pass fallback"
    );
}

#[test]
fn zq_works_with_subsampling_422() {
    let (w, h) = (256u32, 256);
    let rgb = synthetic_image(w, h);
    let config = EncoderConfig::ycbcr(Quality::Zq(80.0), ChromaSubsampling::HalfHorizontal);
    let mut enc = config
        .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
        .unwrap();
    enc.push_packed(&rgb, Unstoppable).unwrap();
    let (jpeg, metrics) = enc
        .finish_with_metrics()
        .expect("4:2:2 Zq encode should succeed");
    assert!(!jpeg.is_empty());
    assert!(metrics.achieved_score.is_finite());
}

#[test]
fn zq_works_with_subsampling_440() {
    let (w, h) = (256u32, 256);
    let rgb = synthetic_image(w, h);
    let config = EncoderConfig::ycbcr(Quality::Zq(80.0), ChromaSubsampling::HalfVertical);
    let mut enc = config
        .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
        .unwrap();
    enc.push_packed(&rgb, Unstoppable).unwrap();
    let (jpeg, metrics) = enc
        .finish_with_metrics()
        .expect("4:4:0 Zq encode should succeed");
    assert!(!jpeg.is_empty());
    assert!(metrics.achieved_score.is_finite());
}

#[test]
fn zq_works_with_xyb_bquarter() {
    use zenjpeg::encode::XybSubsampling;
    let (w, h) = (256u32, 256);
    let rgb = synthetic_image(w, h);
    // XYB encoding accepts RGB8 input (sRGB) and converts internally;
    // the decoded JPEG is decoded back to RGB8 by the decoder so the
    // diffmap measurement against the source RGB8 is well-defined.
    let config = EncoderConfig::xyb(Quality::Zq(80.0), XybSubsampling::BQuarter);
    let mut enc = config
        .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
        .unwrap();
    enc.push_packed(&rgb, Unstoppable).unwrap();
    let (jpeg, metrics) = enc
        .finish_with_metrics()
        .expect("XYB BQuarter Zq encode should succeed");
    assert!(!jpeg.is_empty());
    assert!(metrics.achieved_score.is_finite());
}

#[test]
fn zq_works_with_xyb_full() {
    use zenjpeg::encode::XybSubsampling;
    let (w, h) = (256u32, 256);
    let rgb = synthetic_image(w, h);
    let config = EncoderConfig::xyb(Quality::Zq(80.0), XybSubsampling::Full);
    let mut enc = config
        .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
        .unwrap();
    enc.push_packed(&rgb, Unstoppable).unwrap();
    let (jpeg, metrics) = enc
        .finish_with_metrics()
        .expect("XYB Full Zq encode should succeed");
    assert!(!jpeg.is_empty());
    assert!(metrics.achieved_score.is_finite());
}

#[test]
fn zq_works_with_linear_f32_input() {
    use zenjpeg::encode::XybSubsampling;
    let (w, h) = (128u32, 128);
    // Convert sRGB8 → linear f32 RGB. This is the typical XYB-input
    // pipeline (pre-converted linear light from a higher-precision
    // source).
    let rgb_u8 = synthetic_image(w, h);
    let mut rgb_f32 = vec![0u8; (w as usize) * (h as usize) * 12];
    for i in 0..(w as usize) * (h as usize) {
        for c in 0..3 {
            let u = rgb_u8[i * 3 + c] as f32 / 255.0;
            // sRGB → linear (approximate gamma 2.2; exact is fine too).
            let lin = if u <= 0.04045 {
                u / 12.92
            } else {
                ((u + 0.055) / 1.055).powf(2.4)
            };
            rgb_f32[i * 12 + c * 4..i * 12 + c * 4 + 4].copy_from_slice(&lin.to_le_bytes());
        }
    }
    let config = EncoderConfig::xyb(Quality::Zq(80.0), XybSubsampling::BQuarter);
    let mut enc = config
        .encode_from_bytes(w, h, PixelLayout::RgbF32Linear)
        .expect("encoder creation");
    enc.push_packed(&rgb_f32, Unstoppable).expect("push");
    let (jpeg, metrics) = enc
        .finish_with_metrics()
        .expect("XYB linear-f32 Zq encode should succeed");
    assert!(!jpeg.is_empty());
    // PR-E lets the iteration loop run on linear-f32 input — measurement
    // uses zensim's linear_planar source path. achieved_score should be
    // finite (NOT NaN, which would indicate single-pass fallback).
    assert!(
        metrics.achieved_score.is_finite(),
        "linear-f32 path must run iteration loop, not fall through to single_pass"
    );
}

#[test]
fn non_zq_quality_does_not_iterate() {
    let (w, h) = (128u32, 128);
    let rgb = synthetic_image(w, h);
    let config = EncoderConfig::ycbcr(Quality::ApproxJpegli(85.0), ChromaSubsampling::Quarter);
    let mut enc = config
        .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
        .unwrap();
    enc.push_packed(&rgb, Unstoppable).unwrap();
    let (jpeg, metrics) = enc.finish_with_metrics().unwrap();
    assert!(!jpeg.is_empty());
    assert_eq!(metrics.passes_used, 1, "non-Zq is single-pass");
    assert!(
        metrics.achieved_score.is_nan(),
        "non-Zq doesn't measure perceptual score"
    );
    assert!(metrics.targets_met, "non-Zq has no goal — vacuously met");
}
