//! Real-codec production gate for the zenjpeg target loop
//! ([`zenjpeg::target_quality::encode_with_target`]).
//!
//! The unit tests in `src/target_quality.rs` prove the secant math with synthetic
//! monotone curves. This integration test closes the loop on the ACTUAL codec and
//! an ACTUAL perceptual metric: it wires real `JpegEncoderConfig` encode →
//! `JpegDecoderConfig` decode → `fast_ssim2::compute_ssimulacra2` into the loop,
//! then checks convergence across a spread of targets — the k2/k3-style census
//! the production gate wants (here against SSIMULACRA2; the zensim-specific
//! 27-cell census is the follow-on, wired the same way with a `zensim` dev-dep).
//!
//! Requires the `zencodec` feature (the `codec` module's re-exported types). Run:
//!   cargo test -p zenjpeg --features zencodec --test target_quality_real
#![cfg(feature = "zencodec")]

use std::borrow::Cow;

use fast_ssim2::{LinearRgbImage, compute_ssimulacra2, srgb_u8_to_linear};
use imgref::Img;
use rgb::Rgb;
use zencodec::decode::{Decode as _, DecodeJob as _, DecoderConfig as _};

use zenpixels::PixelDescriptor;

use zenjpeg::target_quality::{TargetOptions, encode_with_target};
use zenjpeg::{JpegDecoderConfig, JpegEncoderConfig};

/// Packed RGB8 bytes of a textured test image whose SSIMULACRA2 score responds
/// monotonically to JPEG quality: smooth gradients (low-freq) PLUS an XOR/checker
/// high-frequency layer (which quantization degrades), so the score actually moves
/// with q instead of pinning near 100 the way a flat gradient would.
fn textured_rgb8(w: usize, h: usize) -> Vec<u8> {
    let mut px = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            let grad_r = (x * 255 / w.max(1)) as u8;
            let grad_g = (y * 255 / h.max(1)) as u8;
            // MODERATE mid-frequency texture: enough that quality matters, not so
            // harsh that even q90 craters (which would collapse the achievable
            // SSIMULACRA2 range and make targeting meaningless). Coarser XOR
            // (÷3) + modest amplitude gives a healthy ~q20..q90 score spread.
            let hf = ((((x / 3) ^ (y / 3)) & 0x1F) as u8).wrapping_mul(3);
            let checker = if (x / 8 + y / 8) % 2 == 0 { 18u8 } else { 0 };
            px.push(grad_r.wrapping_add(checker));
            px.push(grad_g);
            px.push(128u8.wrapping_add(hf));
        }
    }
    px
}

fn to_linear(bytes: &[u8], w: usize, h: usize) -> LinearRgbImage {
    let pixels: Vec<[f32; 3]> = bytes
        .as_chunks::<3>()
        .0
        .iter()
        .map(|c| {
            [
                srgb_u8_to_linear(c[0]),
                srgb_u8_to_linear(c[1]),
                srgb_u8_to_linear(c[2]),
            ]
        })
        .collect();
    LinearRgbImage::new(pixels, w, h)
}

fn encode_at(rgb8: &[u8], w: usize, h: usize, q: f32) -> Vec<u8> {
    let pixels: Vec<Rgb<u8>> = rgb8
        .as_chunks::<3>()
        .0
        .iter()
        .map(|c| Rgb {
            r: c[0],
            g: c[1],
            b: c[2],
        })
        .collect();
    let img = Img::new(pixels, w, h);
    let enc = JpegEncoderConfig::new().with_calibrated_quality(q.round().clamp(1.0, 100.0));
    enc.encode(zenpixels::PixelSlice::from(img.as_ref()).into())
        .expect("encode jpeg")
        .data()
        .to_vec()
}

/// Decode a JPEG to packed RGB8 bytes (+ dims).
fn decode_rgb8(bytes: &[u8]) -> (Vec<u8>, usize, usize) {
    let out = JpegDecoderConfig::new()
        .job()
        .decoder(Cow::Borrowed(bytes), &[PixelDescriptor::RGB8_SRGB])
        .expect("build decoder")
        .decode()
        .expect("decode jpeg");
    let px = out.pixels();
    let (w, h) = (px.width() as usize, px.rows() as usize);
    let raw = px
        .as_contiguous_bytes()
        .expect("decoded RGB8 is contiguous")
        .to_vec();
    (raw, w, h)
}

/// One real trial: encode `ref_bytes` at `q`, decode, score with SSIMULACRA2.
fn encode_decode_score(ref_bytes: &[u8], w: usize, h: usize, q: f32) -> (Vec<u8>, f64) {
    let jpeg = encode_at(ref_bytes, w, h, q);
    let (dec_bytes, dw, dh) = decode_rgb8(&jpeg);
    assert_eq!((dw, dh), (w, h), "decode changed dimensions");
    let score = compute_ssimulacra2(to_linear(ref_bytes, w, h), to_linear(&dec_bytes, w, h))
        .unwrap_or(-100.0);
    (jpeg, score)
}

#[test]
fn converges_on_real_jpeg_ssim2() {
    let (w, h) = (192usize, 192usize);
    let reference = textured_rgb8(w, h);

    // Sanity: the metric IS responsive to quality on this content.
    let (_, s_lo) = encode_decode_score(&reference, w, h, 20.0);
    let (_, s_hi) = encode_decode_score(&reference, w, h, 90.0);
    assert!(
        s_hi > s_lo + 2.0,
        "SSIMULACRA2 not responsive to quality: q20={s_lo:.2} q90={s_hi:.2}"
    );
    assert!(s_hi <= 100.5, "score out of range: {s_hi:.2}");

    let opts = TargetOptions {
        min_quality: 5.0,
        max_quality: 98.0,
        tolerance: 1.5,
        max_encodes: 10,
        q_start: None,
        // Inherit the shipped `quality_step` rather than pinning one here, so
        // this test exercises the default the crate actually ships and does not
        // go stale again the next time a field is added.
        ..Default::default()
    };
    let target = (s_lo + s_hi) / 2.0; // comfortably inside the achievable band

    let out = encode_with_target(target, &opts, |q| {
        Ok::<_, String>(encode_decode_score(&reference, w, h, q))
    })
    .expect("loop ran");

    // Returned bytes must be a real JPEG that re-decodes and re-scores at the
    // reported score (the loop returns the winning iterate's ACTUAL bytes).
    assert!(!out.data.is_empty(), "no bytes returned");
    assert_eq!(&out.data[0..2], &[0xFF, 0xD8], "not a JPEG (no SOI marker)");
    let (redec, _, _) = decode_rgb8(&out.data);
    let rescore =
        compute_ssimulacra2(to_linear(&reference, w, h), to_linear(&redec, w, h)).unwrap_or(-100.0);
    assert!(
        (rescore - out.search.score).abs() < 1e-6,
        "returned bytes ({rescore:.4}) are not the reported iterate ({:.4})",
        out.search.score
    );

    // Contract: return the LOWEST quality whose score reaches `target -
    // tolerance` (the smallest file at the target). So the achieved score must be
    // AT OR ABOVE the band floor — it may overshoot the target when the codec's
    // quality granularity is coarse (that overshoot is the correct, smallest
    // reaching iterate, not an error). A loose upper bound catches a broken
    // search that just pins at max quality.
    assert!(out.search.converged, "did not reach the target band");
    assert!(
        out.search.score >= target - opts.tolerance,
        "below the band floor: got {:.2}, target {target:.2}, tol {}",
        out.search.score,
        opts.tolerance
    );
    assert!(
        out.search.score <= target + 8.0,
        "overshot far past target (search likely pinned high): got {:.2}, target {target:.2}",
        out.search.score
    );
    assert!(out.search.encodes <= opts.max_encodes, "over budget");
}

/// k2/k3-style convergence census across a spread of interior targets.
#[test]
fn convergence_census_across_targets() {
    let (w, h) = (192usize, 192usize);
    let reference = textured_rgb8(w, h);
    let (_, s_lo) = encode_decode_score(&reference, w, h, 15.0);
    let (_, s_hi) = encode_decode_score(&reference, w, h, 95.0);

    let opts = TargetOptions {
        min_quality: 5.0,
        max_quality: 98.0,
        tolerance: 1.5,
        max_encodes: 12,
        q_start: None,
        // See the note above: inherit the shipped `quality_step`.
        ..Default::default()
    };

    let lo = s_lo + (s_hi - s_lo) * 0.15;
    let hi = s_lo + (s_hi - s_lo) * 0.85;
    let targets: Vec<f64> = (0..6).map(|i| lo + (hi - lo) * (i as f64) / 5.0).collect();

    let (mut converged, mut within_k2, mut within_k3) = (0usize, 0usize, 0usize);
    for &t in &targets {
        let out = encode_with_target(t, &opts, |q| {
            Ok::<_, String>(encode_decode_score(&reference, w, h, q))
        })
        .expect("loop ran");
        if out.search.converged {
            converged += 1;
            if out.search.encodes <= 3 {
                within_k2 += 1;
            }
            if out.search.encodes <= 4 {
                within_k3 += 1;
            }
        }
        eprintln!(
            "census target={t:.2} -> score={:.2} q={:.0} converged={} encodes={}",
            out.search.score, out.search.quality, out.search.converged, out.search.encodes
        );
    }
    let n = targets.len();
    eprintln!(
        "CENSUS: converged {converged}/{n} · k3(<=4enc) {within_k3}/{n} · k2(<=3enc) {within_k2}/{n}"
    );

    // A working secant reaches the interior band on the clear majority.
    assert!(
        converged >= n - 1,
        "secant loop converged on only {converged}/{n} interior targets"
    );
}
