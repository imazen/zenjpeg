//! Strategy router.
//!
//! Maps `(SourceAnalysis, target_zensim_a, budget)` to one of:
//! `NoOp`, `Lossless`, or a concrete `(StrategyKind, params)` tuple. The
//! decision is calibration-table-driven; there is no candidate enumeration
//! unless the caller asked for it via [`Budget::MaxIterations`] or
//! [`Budget::MaxTime`].

use crate::recompress::api::{Budget, LosslessReason, NoOpReason, StrategyKind};
use crate::recompress::calibration::{CalibrationLookup, CellCi, EncoderClass, StrategyChoice};
use crate::recompress::error::Error;
use crate::recompress::source::SourceAnalysis;

/// zensim-A band within which we treat the source as already-at-target
/// and return `NoOp`.
const ZENSIM_A_NOOP_BAND: f32 = 1.5;

/// If recompression's *best* projected output is ≥ this fraction of the
/// source, we prefer the lossless re-pack.
const LOSSLESS_SIZE_FALLBACK_THRESHOLD: f32 = 0.98;

/// Margin by which lossless must beat the best recompression candidate
/// on projected size before we prefer it. Lossless *overshoots* the
/// quality target — it delivers the source's full quality, not the
/// (lower) target the user asked for — so it should win only when it is
/// *meaningfully* smaller, never on a near-tie. This also makes the
/// picker robust to calibration noise: `estimate_lossless` returns a
/// flat 0.94 guess, and per-encoder recompression ratios fit from a
/// small corpus can land within a rounding error of it. Without the
/// margin, a 0.0003 difference between a noisy recompression projection
/// and the flat lossless guess flips the decision and ships a
/// barely-compressed, quality-overshooting file instead of a real
/// target-hitting recompression. (Discovered 2026-05-29 during the
/// Lever-2 mozjpeg refit: q65→t50 cells projected Tuned 0.9403 vs
/// Lossless 0.9400, tipped to lossless@0.99 when Tuned actually
/// achieves 0.78 on target.)
const LOSSLESS_PREFER_MARGIN: f32 = 0.03;

/// Preserve is functional as of 2026-05-28 via the
/// `strategies::preserve_emit` minimal JPEG emitter built on
/// zenjpeg's public entropy + huffman primitives. The router
/// considers it as a candidate.
const PRESERVE_AVAILABLE: bool = true;

/// Inputs to the router.
#[derive(Debug)]
pub struct RouterInput<'a> {
    pub analysis: &'a SourceAnalysis,
    /// The quality the user asked for. Used for the NoOp gate (is the
    /// source already at-or-below what they want?).
    pub target_zensim_a: f32,
    /// The quality the encoder should *aim* at, after the delivery-
    /// confidence shift. `>= target_zensim_a`. Used for strategy
    /// selection and parameter derivation so the chosen quantile of
    /// achieved quality clears `target_zensim_a`. For
    /// [`crate::recompress::api::Confidence::P50`] this equals `target_zensim_a`.
    pub effective_target: f32,
    pub budget: Budget,
}

/// Router output.
#[derive(Debug, Clone)]
pub enum RouterOutput {
    NoOp {
        reason: NoOpReason,
    },
    Lossless {
        reason: LosslessReason,
    },
    Strategy {
        kind: StrategyKind,
        params: StrategyParams,
        projected_zensim_a: f32,
    },
}

/// Parameter pack passed to the chosen strategy.
#[derive(Debug, Clone, Copy)]
pub struct StrategyParams {
    /// Target IJG quality for IJG-family re-encodes.
    pub target_ijg_q: u8,
    /// Target butteraugli distance for jpegli-family re-encodes.
    pub target_ba_distance: f32,
    /// Cell confidence — used by strategies that have a fallback path
    /// (e.g., Preserve can ask Tuned to take over if CI is `Empty`).
    pub ci: CellCi,
    /// The **effective** zensim-A target the strategy aims at — the user
    /// target after the delivery-confidence shift and the source-quality
    /// cap (see `decide_strategy`), NOT the raw user target. Strategies
    /// use this to decide how much quality headroom they have to spend, so
    /// the AQ gate scales with the confidence setting (e.g. P25 lowers the
    /// effective target, freeing more headroom; P90 raises it).
    pub target_zensim_a: f32,
    /// Calibrated projection of the cumulative zensim-A this strategy
    /// will deliver vs the reference. Preserve uses
    /// `projected_zensim_a - target_zensim_a` as its AQ headroom: AQ
    /// fires only when the projected overshoot covers AQ's expected
    /// quality cost.
    pub projected_zensim_a: f32,
    /// The zensim-A dial fed to `target_zensim_a_to_ijg_q` /
    /// `target_zensim_a_to_ba_distance` to derive the quality params.
    /// The closed loop ([`Budget::MaxIterations`] > 1) bumps this and
    /// re-derives the params when a measured pass lands short of target.
    pub dial_zensim_a: f32,
}

pub fn decide_strategy(input: &RouterInput<'_>) -> Result<RouterOutput, Error> {
    let source_estimate = input.analysis.estimated_zensim_a_vs_reference();

    // 1. NoOp gate uses the USER target: if the source is already
    // at-or-below what the user asked for, recompressing can only make
    // it worse. The confidence shift must NOT inflate this gate —
    // otherwise a source that comfortably clears the user's target
    // would NoOp just because it can't reach the (higher) internal aim.
    if source_estimate <= input.target_zensim_a + ZENSIM_A_NOOP_BAND {
        return Ok(RouterOutput::NoOp {
            reason: NoOpReason::SourceAlreadyMeetsTarget,
        });
    }

    // Strategy selection aims at the EFFECTIVE target (confidence-
    // shifted, capped at the source's own quality — we can't recompress
    // to a higher quality than the source has). When the effective
    // target approaches the source quality, projected ratios approach
    // 1.0 and the picker correctly falls through to Lossless.
    let target = input
        .effective_target
        .min(source_estimate)
        .clamp(0.0, 100.0);

    // 2. Estimate every strategy at the effective target.
    let encoder = EncoderClass::from_family(input.analysis.encoder);
    let estimates = CalibrationLookup::SEED.estimate_all(
        encoder,
        input.analysis.subsampling,
        source_estimate,
        target,
    );

    // 3. Among Preserve/Deblock/Tuned, pick the smallest projected size
    // among candidates that hit the effective target within ±2 zensim-A.
    // If none qualifies, fall back to the candidate that is *closest* to
    // target (highest projected zensim-A) — that's the best
    // recompression the calibrated table says is achievable. The final
    // lossless guard below catches the case where that candidate would
    // still inflate.
    let recompression_candidates: Vec<&StrategyChoice> = estimates
        .iter()
        .filter(|c| !matches!(c.kind, StrategyKind::Lossless))
        .filter(|c| PRESERVE_AVAILABLE || !matches!(c.kind, StrategyKind::Preserve))
        .collect();

    // Prefer candidates that hit target.
    let on_target = recompression_candidates
        .iter()
        .filter(|c| c.estimate.projected_zensim_a + 2.0 >= target)
        .copied()
        .min_by(|a, b| {
            a.estimate
                .projected_size_ratio
                .partial_cmp(&b.estimate.projected_size_ratio)
                .unwrap_or(core::cmp::Ordering::Equal)
        });

    // Best-effort: closest projection to target among all candidates.
    let closest = recompression_candidates.iter().copied().max_by(|a, b| {
        a.estimate
            .projected_zensim_a
            .partial_cmp(&b.estimate.projected_zensim_a)
            .unwrap_or(core::cmp::Ordering::Equal)
    });

    let lossless = estimates
        .iter()
        .find(|c| matches!(c.kind, StrategyKind::Lossless))
        .copied();

    // 4. Picker:
    //   - prefer on-target candidate if its projected size beats lossless.
    //   - if no on-target candidate exists, take the closest projection
    //     IF its projected ratio is < LOSSLESS_SIZE_FALLBACK_THRESHOLD
    //     (i.e., we actually save bytes).
    //   - otherwise lossless.
    let candidate = on_target.or(closest);
    let take_lossless = match (candidate, lossless) {
        (None, _) => true,
        (Some(c), Some(l)) => {
            // Prefer lossless only when it is MEANINGFULLY smaller than
            // the best recompression (by `LOSSLESS_PREFER_MARGIN`), or
            // when recompression would barely save / inflate (the
            // independent 0.98 guard). A near-tie goes to recompression,
            // which hits the requested target instead of overshooting
            // quality at the source's size.
            c.estimate.projected_size_ratio
                >= l.estimate.projected_size_ratio + LOSSLESS_PREFER_MARGIN
                || c.estimate.projected_size_ratio >= LOSSLESS_SIZE_FALLBACK_THRESHOLD
        }
        (Some(c), None) => c.estimate.projected_size_ratio >= LOSSLESS_SIZE_FALLBACK_THRESHOLD,
    };

    if take_lossless {
        return Ok(RouterOutput::Lossless {
            reason: LosslessReason::RecompressionWouldInflateAtTarget,
        });
    }

    let chosen = candidate.unwrap();

    // Dial the chosen strategy at its inverse-calibrated dial (so it
    // *achieves* the target) when the per-encoder table provided one;
    // otherwise dial the target directly.
    let dial = chosen.estimate.dial_zensim_a.unwrap_or(target);

    let params = StrategyParams {
        target_ijg_q: crate::recompress::target::target_zensim_a_to_ijg_q(dial),
        target_ba_distance: crate::recompress::target::target_zensim_a_to_ba_distance(dial),
        ci: chosen.estimate.ci,
        target_zensim_a: target,
        projected_zensim_a: chosen.estimate.projected_zensim_a,
        dial_zensim_a: dial,
    };

    let _ = input.budget; // budget shapes strategy internals, not routing

    Ok(RouterOutput::Strategy {
        kind: chosen.kind,
        params,
        projected_zensim_a: chosen.estimate.projected_zensim_a,
    })
}
