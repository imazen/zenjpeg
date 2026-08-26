//! Zq seed head for [`crate::target_quality`] — predicts the *starting*
//! quality (`TargetOptions::q_start`) for a zensim-targeted encode from
//! zenanalyze content features, so the bracketed secant search converges in
//! fewer encode→decode→score iterations than the content-blind
//! [`crate::target_quality::anchor_guess`].
//!
//! Mirrors zenavif's `q0_head` (the per-codec-loop-ownership exemplar): a
//! deterministic fitted-constants head — no model file — over a
//! piecewise-linear (hinge) target basis with feature×target interactions.
//!
//! # Fit provenance (2026-08-26)
//!
//! `scripts/fit_zq_seed.py` on the 07-01 canonical picker set
//! (`canonical-picker-2026-07-01-zensimA/zenjpeg_lossy`, zensimA-profile
//! `score_zensim`): 96,894 full 7-point per-rendition q→zensim curves,
//! PAVA-isotonized and inverse-labeled at targets {40,45,…,90} (927,217
//! train labels; 27,684 coarse 3-point-plan curves excluded). Greedy
//! forward selection by leave-one-origin-out p90 on train only.
//!
//! Pre-registered wave: `benchmarks/zq_seed_wave_2026-08-26.md`. The
//! unclamped head FAILED the frozen G-Z2 gate (25.7% fewer encodes but 189
//! deep-undershoot convergence regressions); the registered arm B clamps
//! the seed to `[anchor−18, anchor+12]` (L=18 = the largest L in the frozen
//! grid with ZERO regressions). **Shipped verdict (validate, 559,596
//! cells): mean encodes 4.53 → 3.92 (−13.5%), converged 559,596/559,596 —
//! no regression. G-Z2 PASS.** Tables: `benchmarks/zq_seed_fit_2026-08-26.tsv`,
//! `benchmarks/zq_armB_grid_2026-08-26.tsv`.
//!
//! # Registered limitation
//!
//! Labels are zensimA-era `score_zensim`; a runtime loop may steer a newer
//! bake. The seed only needs to land near the basin — the search corrects
//! the residual — and the clamp bounds the damage of any model drift. A
//! refit on current-model rescoring is a registered follow-up data task.
//!
//! # Scope
//!
//! Fitted for `[0,100]`-scaled perceptual (zensim) targets in 40–90; the
//! basis input is clamped to that band (the search's own extrapolation
//! handles the rest). `None` on any non-finite input — the caller falls
//! back to the anchor curve, so the head can only ever *re-seed* the
//! search, never break it.

use crate::target_quality::anchor_guess;

/// The six zenanalyze features, in fit order. `dct_compressibility_uv`
/// (index 1) and `distinct_color_bins` (index 3) are `ln_1p`-transformed
/// inside [`predict_q0_from_features`] — pass RAW values.
pub const ZQ_FEATURES: [zenanalyze::feature::AnalysisFeature; 6] = [
    zenanalyze::feature::AnalysisFeature::FlatColorBlockRatio,
    zenanalyze::feature::AnalysisFeature::DctCompressibilityUV,
    zenanalyze::feature::AnalysisFeature::SpectralSlopeY,
    zenanalyze::feature::AnalysisFeature::DistinctColorBins,
    zenanalyze::feature::AnalysisFeature::GrayscaleScore,
    zenanalyze::feature::AnalysisFeature::SkinToneFraction,
];

/// Fitted robust-L1 coefficients (fit_zq_seed.py 2026-08-26). Layout:
/// `[const, tn, h50, h60, h70, h80, h85, logpx_n, f_0..f_5, f_0*tn..f_5*tn,
/// f_0*h80..f_5*h80]` with `tn=(t−65)/25`, `h_k=max(t−k,0)/10`,
/// `logpx_n=(ln(px)−13)/3`.
const ZQ_COEFS: [f64; 26] = [
    36.752419364192620,    // const
    18.532795461862516,    // tn
    7.471654457918908,     // h50
    10.488342837677967,    // h60
    -8.120355846244466,    // h70
    -8.690436307114338,    // h80
    -3.6566196740897556,   // h85
    0.0035565671303173455, // logpx_n
    -17.620623388914616,   // flat_color_block_ratio
    3.436723078983692,     // ln_1p(dct_compressibility_uv)
    -16.85493783310687,    // spectral_slope_y
    1.9475242187926536,    // ln_1p(distinct_color_bins)
    -6.786372929520951,    // grayscale_score
    -4.6579855184998396,   // skin_tone_fraction
    -3.345171080091729,    // fcbr*tn
    1.4250166987945871,    // dcuv*tn
    -4.45295171168348,     // ssy*tn
    0.130007394190472,     // dcb*tn
    0.5174768442810931,    // gs*tn
    -0.17685600917547373,  // stf*tn
    20.242273889207684,    // fcbr*h80
    -4.520650281804192,    // dcuv*h80
    20.418726350287514,    // ssy*h80
    -1.4890129479408634,   // dcb*h80
    4.626467480003019,     // gs*h80
    5.387196900167114,     // stf*h80
];

/// Fitted target band; basis inputs clamp here.
const ZQ_T_MIN: f64 = 40.0;
const ZQ_T_MAX: f64 = 90.0;
/// Registered arm-B safety clamp around the anchor curve (L = 18 below,
/// 12 above — the frozen-rule pick with zero convergence regressions).
const ZQ_CLAMP_BELOW: f32 = 18.0;
const ZQ_CLAMP_ABOVE: f32 = 12.0;

/// Pure evaluation of the fitted head on already-extracted feature values
/// (in [`ZQ_FEATURES`] order, RAW — transforms applied here). Returns the
/// seed quality clamped to `[anchor−18, anchor+12]` and then `[1, 100]`.
/// `None` if any input is non-finite — the caller keeps the anchor curve.
#[must_use]
pub fn predict_q0_from_features(features: &[f32; 6], target: f64, pixels: u64) -> Option<f32> {
    if !target.is_finite() || features.iter().any(|f| !f.is_finite()) {
        return None;
    }
    let t = target.clamp(ZQ_T_MIN, ZQ_T_MAX);
    let tn = (t - 65.0) / 25.0;
    let h = |k: f64| (t - k).max(0.0) / 10.0;
    let logpx_n = (f64::from(u32::try_from(pixels.max(1)).unwrap_or(u32::MAX)).ln() - 13.0) / 3.0;

    let mut fv = [0.0f64; 6];
    for (i, f) in features.iter().enumerate() {
        fv[i] = f64::from(*f);
    }
    fv[1] = fv[1].max(0.0).ln_1p(); // dct_compressibility_uv
    fv[3] = fv[3].max(0.0).ln_1p(); // distinct_color_bins

    let mut x = [0.0f64; 26];
    x[0] = 1.0;
    x[1] = tn;
    x[2] = h(50.0);
    x[3] = h(60.0);
    x[4] = h(70.0);
    x[5] = h(80.0);
    x[6] = h(85.0);
    x[7] = logpx_n;
    let h80 = h(80.0);
    for i in 0..6 {
        x[8 + i] = fv[i];
        x[14 + i] = fv[i] * tn;
        x[20 + i] = fv[i] * h80;
    }
    let q0: f64 = ZQ_COEFS.iter().zip(x.iter()).map(|(c, v)| c * v).sum();

    let anchor = anchor_guess(target);
    #[allow(clippy::cast_possible_truncation)]
    let q0 = (q0 as f32)
        .clamp(anchor - ZQ_CLAMP_BELOW, anchor + ZQ_CLAMP_ABOVE)
        .clamp(1.0, 100.0);
    Some(q0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Golden vector, computed independently by the fit script's Python
    /// pipeline: raw features [0.25, 3.5, -1.2, 40.0, 0.1, 0.05], t=72,
    /// px=300000 → raw head 98.6607, anchor 81.76 → upper clamp → 93.76.
    #[test]
    fn golden_matches_fit_pipeline() {
        let q0 =
            predict_q0_from_features(&[0.25, 3.5, -1.2, 40.0, 0.1, 0.05], 72.0, 300_000).unwrap();
        assert!((q0 - 93.76).abs() < 1e-3, "q0 = {q0}");
    }

    #[test]
    fn non_finite_inputs_return_none() {
        assert!(
            predict_q0_from_features(&[f32::NAN, 0.0, 0.0, 0.0, 0.0, 0.0], 70.0, 1000).is_none()
        );
        assert!(predict_q0_from_features(&[0.0; 6], f64::NAN, 1000).is_none());
    }

    #[test]
    fn always_inside_anchor_band_and_valid_quality() {
        for t in [0.0, 40.0, 55.0, 72.0, 90.0, 100.0] {
            for f in [[0.0f32; 6], [1.0; 6], [0.9, 50.0, -3.0, 5000.0, 1.0, 1.0]] {
                let q0 = predict_q0_from_features(&f, t, 250_000).unwrap();
                let a = anchor_guess(t);
                assert!(q0 >= (a - ZQ_CLAMP_BELOW).clamp(1.0, 100.0) - 1e-4);
                assert!(q0 <= (a + ZQ_CLAMP_ABOVE).clamp(1.0, 100.0) + 1e-4);
                assert!((1.0..=100.0).contains(&q0));
            }
        }
    }

    #[test]
    fn band_edges_clamp_target_basis() {
        // Below/above the fitted band the basis input pins to the edge, so
        // only the anchor clamp moves the result.
        let f = [0.2f32, 2.0, -0.5, 100.0, 0.3, 0.1];
        let lo = predict_q0_from_features(&f, 40.0, 100_000).unwrap();
        let lo2 = predict_q0_from_features(&f, 10.0, 100_000).unwrap();
        let a40 = anchor_guess(40.0);
        let a10 = anchor_guess(10.0);
        assert!((lo - lo2).abs() <= (a40 - a10).abs() + 1e-4);
    }
}
