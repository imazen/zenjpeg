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

// ---- Quality::ZqPicker: realtime one-shot (2026-07-16) ----

/// The one-shot picker produces a valid JPEG and its metrics carry the
/// one-shot signature: the score is PREDICTED, not measured, so
/// `achieved_score` is `NaN` and exactly one pass runs. This is the
/// distinguishing contract vs the measuring loop.
#[test]
fn zq_picker_one_shot_predicts_without_measuring() {
    let (w, h) = (256u32, 256);
    let rgb = synthetic_image(w, h);
    let config = EncoderConfig::ycbcr(Quality::ZqPicker(85.0), ChromaSubsampling::Quarter);
    let mut enc = config
        .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
        .expect("encoder creation");
    enc.push_packed(&rgb, Unstoppable).expect("push");
    let (jpeg, metrics) = enc.finish_with_metrics().expect("finish_with_metrics");

    assert!(!jpeg.is_empty(), "one-shot must produce JPEG bytes");
    assert_eq!(
        jpeg.len(),
        metrics.bytes,
        "metrics.bytes must match jpeg.len()"
    );
    assert_eq!(
        metrics.passes_used, 1,
        "one-shot is a single encode — no correction passes"
    );
    assert!(
        metrics.achieved_score.is_nan(),
        "one-shot PREDICTS the config; it does not decode+measure, so \
         achieved_score is NaN (this is the signature that distinguishes it \
         from the Zq measuring loop)"
    );
    assert!(
        metrics.targets_met,
        "one-shot reports met (it trusts the prediction; there is no measured miss)"
    );

    // The bytes must be a real, decodable JPEG at the right dimensions.
    let img = zenjpeg::decode::Decoder::new()
        .decode(&jpeg, Unstoppable)
        .expect("one-shot output must decode");
    assert_eq!((img.width(), img.height()), (w, h), "decoded dimensions");
}

/// One-shot and iterative are observably different paths on the SAME source:
/// the picker one-shot never measures (`achieved_score` NaN), while the Zq
/// loop always does (finite). If a regression routed `ZqPicker` through the
/// loop — or `Zq` through the one-shot — this catches it.
#[test]
fn zq_picker_one_shot_differs_from_iterative_loop() {
    let (w, h) = (256u32, 256);
    let rgb = synthetic_image(w, h);

    let encode = |q: Quality| {
        let mut enc = EncoderConfig::ycbcr(q, ChromaSubsampling::Quarter)
            .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
            .unwrap();
        enc.push_packed(&rgb, Unstoppable).unwrap();
        enc.finish_with_metrics().unwrap().1
    };

    let one_shot = encode(Quality::ZqPicker(85.0));
    let iterative = encode(Quality::Zq(85.0));

    assert!(
        one_shot.achieved_score.is_nan(),
        "ZqPicker must take the no-measure one-shot path"
    );
    assert!(
        iterative.achieved_score.is_finite(),
        "Zq must take the measuring loop path"
    );
}

/// Smooth-gradient sibling of [`synthetic_image`] — easy content with a high
/// zensim ceiling (the loop reaches most floors in one correction).
fn smooth_image(w: u32, h: u32) -> Vec<u8> {
    let (w, h) = (w as usize, h as usize);
    let mut rgb = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            rgb[idx] = (x * 255 / w) as u8;
            rgb[idx + 1] = (y * 255 / h) as u8;
            rgb[idx + 2] = ((x + y) * 255 / (w + h)) as u8;
        }
    }
    rgb
}

/// Heavier-texture sibling — a denser deterministic-noise field over the ramp,
/// so the zensim ceiling is lower and the loop has to work harder to lift q.
fn textured_image(w: u32, h: u32) -> Vec<u8> {
    let (w, h) = (w as usize, h as usize);
    let mut rgb = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            let r_base = (x * 255 / w) as u8;
            let g_base = (y * 255 / h) as u8;
            let b_base = ((x + y) * 255 / (w + h)) as u8;
            // Denser noise than synthetic_image (applied every pixel, wider range).
            let n = ((x.wrapping_mul(2654435761))
                ^ (y.wrapping_mul(40503))
                ^ ((x + y).wrapping_mul(2246822519))) as u8;
            rgb[idx] = r_base.wrapping_add(n & 0x7F);
            rgb[idx + 1] = g_base.wrapping_add((n >> 1) & 0x7F);
            rgb[idx + 2] = b_base.wrapping_add((n >> 2) & 0x7F);
        }
    }
    rgb
}

/// 27-cell k2/k3 convergence census for the closed-loop `Quality::Zq` encoder
/// (the native per-block zensim-diffmap target loop, #113).
///
/// A 3-content × 9-target grid (27 cells). `Zq` targets a zensim FLOOR (achieved
/// ≥ target; overshoot is more quality and is fine), so "converged" = the loop
/// reaches at least `target − TOL` within the pass budget. Convergence-within-k
/// is measured by running each cell at a k2 budget (`max_passes=1` → ≤2 total
/// passes incl. the initial encode) and a k3 budget (`max_passes=2` → ≤3 total)
/// separately.
///
/// Floor targets sit in each content's own reachable range — probed by pushing
/// the loop toward an unreachable-high floor with a generous budget and reading
/// the ceiling it lands at — so a non-converging cell is a loop defect, not an
/// unreachable floor. Generous by design (#113 calibration is approximate); the
/// gate catches a broken/absent per-block secant, with the full census printed.
///
/// The per-block loop REDISTRIBUTES bits but cannot raise the global quality
/// level, so a calibrated start that undershoots the floor (dense texture) used
/// to freeze the score below target at any pass count. The global-q correction
/// (#113, `zq.rs`) fixes that: an undershooting pass 0 now secant-raises q toward
/// the floor before redistribution. With it, all 27 cells reach the floor within
/// k3; the few k2 misses are texture cells whose bad calibration start needs two
/// secant steps to climb. Generous by design (#113 calibration is approximate);
/// the gate catches a regression in the per-block secant OR the global-q climb.
#[test]
fn zq_convergence_census_27_cells_k2_k3() {
    type Gen = fn(u32, u32) -> Vec<u8>;
    let contents: [(&str, Gen); 3] = [
        ("mixed", synthetic_image),
        ("smooth", smooth_image),
        ("textured", textured_image),
    ];
    const N: usize = 9;
    const TOL: f32 = 3.0; // floor tolerance: achieved must reach >= target - TOL
    let (w, h) = (192u32, 192u32);

    let encode = |rgb: &[u8], t: f32, passes: u8| -> (f32, u8) {
        let target = ZqTarget::new(t).with_max_passes(passes);
        let config = EncoderConfig::ycbcr(Quality::ZqExplicit(target), ChromaSubsampling::Quarter);
        let mut enc = config
            .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
            .expect("encoder creation");
        enc.push_packed(rgb, Unstoppable).expect("push");
        let (_jpeg, m) = enc.finish_with_metrics().expect("finish_with_metrics");
        (m.achieved_score, m.passes_used)
    };

    let (mut k2, mut k3) = (0usize, 0usize);
    let total = contents.len() * N;
    for (cname, make) in contents {
        let rgb = make(w, h);
        // Probe the ceiling (max reachable floor) with a generous budget.
        let ceiling = encode(&rgb, 99.0, 4).0;
        assert!(
            ceiling.is_finite() && ceiling > 55.0,
            "{cname}: implausible ceiling {ceiling}"
        );
        // 9 floor targets across [ceiling-22, ceiling-2]: reachable, but high
        // enough that the loop must push quality UP from the calibrated start.
        for i in 0..N {
            let t = (ceiling - 22.0) + 20.0 * (i as f32) / (N as f32 - 1.0);
            let (a2, _) = encode(&rgb, t, 1); // <=2 total passes
            let (a3, _) = encode(&rgb, t, 2); // <=3 total passes
            let in2 = a2.is_finite() && a2 >= t - TOL;
            let in3 = a3.is_finite() && a3 >= t - TOL;
            if in2 {
                k2 += 1;
            }
            if in3 {
                k3 += 1;
            }
            eprintln!(
                "census {cname:9} floor={t:>5.1} -> k2={a2:>6.2}({in2}) k3={a3:>6.2}({in3})  (ceil {ceiling:.1})"
            );
        }
    }
    eprintln!("CENSUS(27): k3(<=3 passes) {k3}/{total} · k2(<=2 passes) {k2}/{total}");

    // Pre-registered gates (generous, #113): the per-block secant reaches the
    // floor within a 3-pass budget on the clear majority; k2 is the fast-path
    // yield (calibrated start + one correction).
    assert!(
        k3 >= total - 1,
        "zq loop reached the floor within k3 (≤3 passes) on only {k3}/{total} cells \
         (the global-q correction should reach every floor; see census)"
    );
    assert!(
        k2 >= (total * 2) / 3,
        "only {k2}/{total} cells reached the floor within k2 (≤2 passes)"
    );
}
