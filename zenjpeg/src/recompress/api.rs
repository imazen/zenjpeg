//! Frozen public API surface.
//!
//! Adding fields is allowed via `#[non_exhaustive]`. Removing or renaming any
//! type, variant, or field here is a major-version break.

use core::time::Duration;

use crate::recompress::error::Error;
use crate::recompress::router::{RouterInput, RouterOutput, decide_strategy};
use crate::recompress::source::analyze_source;
use crate::recompress::strategies;

/// Closed-loop (Lever 4): stop bumping once the predicted achieved
/// quality is within this many zensim-A of the target. Avoids spending an
/// extra encode to claw back a fraction of a point. Shared with the
/// diffmap refinement pass (`refine` module), which applies the same
/// acceptance rule to its candidates.
pub(in crate::recompress) const CLOSED_LOOP_TOL: f32 = 1.0;

/// Iteration cap for [`Budget::MaxTime`] (which has no explicit count).
/// `MaxIterations(n)` uses its own `n`.
const CLOSED_LOOP_MAX_ITERS: u32 = 4;

/// Closed-loop dial bump: given the current dial and the predicted
/// achieved quality `a_hat` that fell short of `target`, return the next
/// (higher) dial to retry at — or `None` when the dial is saturated and
/// another pass can't make meaningful progress (so the loop must stop).
///
/// The bump covers the shortfall (`target − a_hat`) with a 1.0 floor so
/// it always moves; the dial is capped at 100. Pulled out of the loop so
/// the termination/saturation behaviour is unit-tested in isolation.
fn next_dial(cur_dial: f32, a_hat: f32, target: f32) -> Option<f32> {
    let bump = (target - a_hat).max(1.0);
    let new_dial = (cur_dial + bump).min(100.0);
    if new_dial <= cur_dial + 0.5 {
        None // saturated — no meaningful headroom left
    } else {
        Some(new_dial)
    }
}

/// User-facing configuration for [`recompress`].
///
/// Constructed via [`RecompressOptions::new`] then optionally adjusted
/// with the `with_*` builder methods. The struct is *not*
/// `#[non_exhaustive]` so callers can pattern-match on it in tests —
/// new options will be added behind `with_*` setters that default to
/// no-op when the field is `Option`.
#[derive(Debug, Clone)]
pub struct RecompressOptions {
    /// Target zensim Profile A value the recompressed output should hit
    /// against the **original unknown reference**. Range `[0.0, 100.0]`,
    /// higher = closer to source. 100 = identical (and impossible to
    /// achieve in this product — pick 95 instead).
    ///
    /// **Calibration-accuracy scope:** per-encoder achieved-quality
    /// tables cover jpegli, libjpeg-turbo, and mozjpeg sources at 4:2:0,
    /// fit on 50 CID22-512 references. Under-target delivery on the n=50
    /// validation corpus is ~3.6 % (jpegli), 7 % (mozjpeg), 8.6 % (turbo)
    /// at the default [`Budget::OneShot`]; the closed loop
    /// ([`Budget::MaxIterations`] > 1) trims mozjpeg→5.6 % / turbo→8.0 %.
    /// The no-size-regression invariant holds for all encoders. Other
    /// encoders (Photoshop, Pillow, …) and subsamplings (4:2:2 / 4:4:0)
    /// fall through to the analytical estimate. See
    /// `docs/MULTI_ENCODER_VALIDATION.md`.
    pub target_zensim_a: f32,

    /// Wall-clock / iteration budget. Default is [`Budget::OneShot`] — no
    /// IQA measurements, no candidate enumeration.
    pub budget: Budget,

    /// Delivery confidence — the fraction of images for which the output
    /// is guaranteed to actually reach `target_zensim_a`. Because
    /// calibration has content-dependent variance, a [`Confidence::P50`]
    /// target means ~half of images land below it. Higher confidence
    /// aims the encoder higher (larger files) so more images clear the
    /// bar. Default is [`Confidence::P50`].
    pub confidence: Confidence,
}

impl RecompressOptions {
    /// Construct a one-shot, P50-confidence configuration targeting
    /// `target_zensim_a`. `target_zensim_a` is clamped to `[0, 100]` at
    /// use time, not here — callers may pass out-of-range values and
    /// observe the resulting [`Error::TargetOutOfRange`].
    pub fn new(target_zensim_a: f32) -> Self {
        Self {
            target_zensim_a,
            budget: Budget::OneShot,
            confidence: Confidence::P50,
        }
    }

    /// Replace the budget. Returns `self` for builder chaining.
    pub fn with_budget(mut self, budget: Budget) -> Self {
        self.budget = budget;
        self
    }

    /// Replace the delivery confidence. Returns `self` for chaining.
    pub fn with_confidence(mut self, confidence: Confidence) -> Self {
        self.confidence = confidence;
        self
    }
}

/// Delivery confidence: the fraction of images for which the
/// recompressed output is guaranteed to reach the requested
/// `target_zensim_a` against the original.
///
/// Implemented as an upward shift of the *internal* target the encoder
/// aims at, sized from the calibration residual distribution
/// (`benchmarks/cid22_15img_fitted_calibration_2026-05-28.tsv`): the
/// lower tail of `achieved − projected` is −2.8 at p25, −13.7 at p10,
/// −19.0 at p5. To guarantee the user's target at confidence C, aim
/// `target + shift(C)` so the C-quantile of achieved quality clears it.
///
/// **The shifts are provisional** — derived from a 15-image,
/// jpegli-family-only sweep. They will tighten as the calibration
/// corpus grows (more images, more encoders). Higher confidence ⇒
/// larger output (or a fall-through to lossless when the headroom is
/// gone).
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum Confidence {
    /// ~25% of images reach the target — aggressive, smallest files,
    /// accepts that most images land slightly under (−5.1 internal aim).
    /// Use when bytes matter more than a hard quality floor.
    P25,
    /// Median — ~50% of images reach the target. Default.
    P50,
    /// ~75% of images reach the target (+2.8 zensim-A internal aim).
    P75,
    /// ~90% of images reach the target (+13.7 internal aim).
    P90,
    /// ~95% of images reach the target (+19.0 internal aim).
    P95,
}

impl Confidence {
    /// Signed shift (zensim-A) applied to the internal target so the
    /// chosen quantile of achieved quality lands at the user's target.
    /// `shift(C) = −quantile_{1−C}(residual)` where `residual =
    /// achieved − projected` from the calibration sweep. Negative for
    /// P25 (aim lower), zero for P50, positive above.
    pub fn target_shift(self) -> f32 {
        match self {
            Confidence::P25 => -5.1,
            Confidence::P50 => 0.0,
            Confidence::P75 => 2.8,
            Confidence::P90 => 13.7,
            Confidence::P95 => 19.0,
        }
    }
}

impl Default for RecompressOptions {
    fn default() -> Self {
        Self::new(80.0)
    }
}

/// Time / iteration budget for the recompression call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Budget {
    /// No IQA loop, no candidate enumeration. Strategy + params picked
    /// from the calibration table; the chosen strategy runs once and
    /// returns. This is the production default.
    #[default]
    OneShot,

    /// Up to `N` candidate encodes; each measured against the source.
    /// Cumulative zensim-A vs reference is still inferred from the
    /// table — measurement only refines the size estimate.
    MaxIterations(u32),

    /// Wall-clock cap. Honors the deadline in the closed loop
    /// (`MaxIterations` semantics, bounded by elapsed wall-clock; the
    /// deadline is checked between passes, not mid-encode).
    MaxTime(Duration),
}

/// Outcome of a [`recompress`] call.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum RecompressResult {
    /// Source was recompressed. `bytes` is the new JPEG payload.
    Recompressed {
        bytes: Vec<u8>,
        /// Which path was chosen.
        strategy: StrategyKind,
        /// Model-predicted zensim-A vs the original reference (NOT a
        /// measurement). In [`Budget::OneShot`] this is the only signal
        /// available.
        projected_zensim_a: f32,
        /// Measured zensim-A vs source (i.e. *generation loss*). Present
        /// only when [`Budget::MaxIterations`] or [`Budget::MaxTime`]
        /// allowed an IQA pass.
        measured_zensim_a: Option<f32>,
        /// `output_len / input_len`. Guaranteed ≤ 1.0 unless the
        /// chosen strategy was overridden by an expert caller.
        source_to_output_ratio: f32,
    },

    /// No recompression was beneficial at the target. The output is the
    /// `crate::lossless::restructure` repacking of the source —
    /// pixel-identical, no quality loss, no size regression.
    LosslessOnly {
        bytes: Vec<u8>,
        reason: LosslessReason,
    },

    /// The source already meets the target; no work was done. Caller
    /// should treat the input as the output.
    NoOp { reason: NoOpReason },
}

impl RecompressResult {
    /// The output JPEG bytes, when the call produced new bytes
    /// ([`Recompressed`](RecompressResult::Recompressed) or
    /// [`LosslessOnly`](RecompressResult::LosslessOnly)). Returns `None`
    /// for [`NoOp`](RecompressResult::NoOp) — there the **source bytes are
    /// the output**, so the ergonomic one-liner is:
    ///
    /// ```no_run
    /// # use zenjpeg::recompress::{recompress, RecompressOptions};
    /// # let jpeg: Vec<u8> = Vec::new();
    /// let result = recompress(&jpeg, &RecompressOptions::new(80.0))?;
    /// let out: &[u8] = result.output_bytes().unwrap_or(&jpeg);
    /// # Ok::<(), zenjpeg::recompress::Error>(())
    /// ```
    pub fn output_bytes(&self) -> Option<&[u8]> {
        match self {
            RecompressResult::Recompressed { bytes, .. }
            | RecompressResult::LosslessOnly { bytes, .. } => Some(bytes),
            RecompressResult::NoOp { .. } => None,
        }
    }
}

/// Which strategy produced a [`RecompressResult::Recompressed`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum StrategyKind {
    /// Coefficient-domain edit (quant-table scale, AQ zero-bias, optional
    /// scan restructure). Zero pixel-domain generation loss.
    Preserve,
    /// Full decode + content-aware deblock + zenjpeg re-encode.
    Deblock,
    /// Full decode + zenjpeg high-quality re-encode (XYB-aware).
    Tuned,
    /// `crate::lossless::restructure` only (no coefficient changes).
    Lossless,
}

/// Why [`RecompressResult::LosslessOnly`] was returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum LosslessReason {
    /// At the requested target, every recompression strategy produces
    /// output ≥ source size. Lossless re-pack is strictly better.
    RecompressionWouldInflateAtTarget,
    /// Target is so close to source quality that any recompression would
    /// add measurable generation loss for no size gain.
    TargetTooCloseToSource,
}

/// Why [`RecompressResult::NoOp`] was returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum NoOpReason {
    /// The estimated source zensim-A vs original is already ≤ target,
    /// i.e. the source is at most as good as the target — recompressing
    /// would make it *worse*, not better.
    SourceAlreadyMeetsTarget,
}

/// Recompress `jpeg_bytes` to the target encoded in `opts`. See
/// [`RecompressOptions`] and [`RecompressResult`].
///
/// # Errors
///
/// Returns [`Error`] if the input cannot be parsed as JPEG, if the
/// underlying zenjpeg decode/encode fails, or if a strategy stage
/// reports an internal inconsistency.
pub fn recompress(jpeg_bytes: &[u8], opts: &RecompressOptions) -> Result<RecompressResult, Error> {
    if !(0.0..=100.0).contains(&opts.target_zensim_a) {
        return Err(Error::TargetOutOfRange(opts.target_zensim_a));
    }

    let analysis = analyze_source(jpeg_bytes)?;

    // Delivery confidence shifts the internal aim upward so the chosen
    // quantile of achieved quality clears the user's target. P50 is a
    // no-op shift. Clamped to [0, 100]; the router additionally caps it
    // at the source's own quality.
    let effective_target =
        (opts.target_zensim_a + opts.confidence.target_shift()).clamp(0.0, 100.0);

    let router_input = RouterInput {
        analysis: &analysis,
        target_zensim_a: opts.target_zensim_a,
        effective_target,
        budget: opts.budget,
    };

    match decide_strategy(&router_input)? {
        RouterOutput::NoOp { reason } => Ok(RecompressResult::NoOp { reason }),

        RouterOutput::Lossless { reason } => {
            let bytes = strategies::lossless::run_lossless(jpeg_bytes, &analysis)?;
            // Guard: if even the lossless re-pack produced more bytes
            // than the source (rare but observed on mozjpeg-style
            // already-optimized inputs), return the source bytes
            // unchanged so the no-size-regression invariant holds at
            // the byte level, not just at the perceptual level.
            if bytes.len() > jpeg_bytes.len() {
                return Ok(RecompressResult::LosslessOnly {
                    bytes: jpeg_bytes.to_vec(),
                    reason,
                });
            }
            Ok(RecompressResult::LosslessOnly { bytes, reason })
        }

        RouterOutput::Strategy {
            kind,
            params,
            projected_zensim_a,
        } => {
            // Defensive: the router shouldn't emit `Lossless` through the
            // Strategy arm, but honor the contract if it does.
            if matches!(kind, StrategyKind::Lossless) {
                let bytes = strategies::lossless::run_lossless(jpeg_bytes, &analysis)?;
                return Ok(RecompressResult::LosslessOnly {
                    bytes,
                    reason: LosslessReason::RecompressionWouldInflateAtTarget,
                });
            }

            // Run the chosen strategy at a given param set. Preserve has a
            // roundtrip-decode + operating-window guard; on `Internal`
            // failure it falls back to Tuned, reported as the kind that
            // actually ran.
            let run_strategy = |p: &crate::recompress::router::StrategyParams| -> Result<
                (strategies::StrategyOutcome, StrategyKind),
                Error,
            > {
                match kind {
                    StrategyKind::Preserve => {
                        match strategies::preserve::run_preserve(jpeg_bytes, &analysis, p, opts) {
                            Ok(o) => Ok((o, StrategyKind::Preserve)),
                            Err(Error::Internal(_)) => Ok((
                                strategies::tuned::run_tuned(jpeg_bytes, &analysis, p, opts)?,
                                StrategyKind::Tuned,
                            )),
                            Err(e) => Err(e),
                        }
                    }
                    StrategyKind::Deblock => Ok((
                        strategies::deblock::run_deblock(jpeg_bytes, &analysis, p, opts)?,
                        StrategyKind::Deblock,
                    )),
                    _ => Ok((
                        strategies::tuned::run_tuned(jpeg_bytes, &analysis, p, opts)?,
                        StrategyKind::Tuned,
                    )),
                }
            };

            // Budget → measurement + iteration cap. `OneShot` runs once,
            // no IQA (production default). `MaxIterations(1)` measures
            // once (no bump). `MaxIterations(n>1)` / `MaxTime` enable the
            // closed loop: measure generation loss vs source, predict the
            // achieved quality, and bump the dial when a pass lands short.
            // `BudgetState` carries the wall-clock deadline for `MaxTime`
            // (checked after each pass; it does not preempt an in-flight
            // encode); `hard_cap` backstops the iteration count so neither
            // a generous `MaxTime` nor dial non-monotonicity can spin.
            // Measurement (generation loss vs source) requires zensim, which
            // is only linked under `recompress-iqa`. Without it the closed
            // loop never measures → behaves like OneShot regardless of Budget.
            #[cfg(feature = "recompress-iqa")]
            let measure = matches!(opts.budget, Budget::MaxIterations(_) | Budget::MaxTime(_));
            #[cfg(not(feature = "recompress-iqa"))]
            let measure = false;
            let hard_cap = match opts.budget {
                Budget::MaxIterations(n) => n.max(1),
                Budget::MaxTime(_) => CLOSED_LOOP_MAX_ITERS,
                Budget::OneShot => 1,
            };
            let mut budget_state = crate::recompress::budget::BudgetState::new(opts.budget);

            // Measurement context: source decoded + reference pyramid
            // built ONCE, reused by every measured pass and by the
            // diffmap refinement below. (The previous per-pass
            // `score_recompression` re-decoded the source and rebuilt
            // the pyramid on every iteration.)
            #[cfg(feature = "recompress-iqa")]
            let measure_ctx = if measure {
                Some(crate::recompress::measure::MeasureCtx::new(jpeg_bytes)?)
            } else {
                None
            };

            let encoder_class =
                crate::recompress::calibration::EncoderClass::from_family(analysis.encoder);
            let source_est = analysis.estimated_zensim_a_vs_reference();
            let target = params.target_zensim_a;

            // Closed loop: each pass predicts achieved-vs-original from the
            // measured generation loss and the cell's expected generation
            // loss (`A_hat = anchor + SLOPE·(g − g_exp)`). `anchor` is the
            // calibration's expected achieved at the current dial; bumping
            // the dial by Δ raises achieved ≈ Δ (the inverse calibration
            // aims achieved ≈ dial). Keep the smallest output that clears
            // target; if none does, the closest one.
            let mut cur = params;
            let mut anchor = projected_zensim_a;
            let mut best: Option<(Vec<u8>, StrategyKind, Option<f32>, f32)> = None;

            // Companion state the diffmap refinement needs about the
            // winning candidate: the params/anchor it was produced at
            // and its measured per-block error map.
            #[cfg(feature = "recompress-iqa")]
            struct BestExt {
                params: crate::recompress::router::StrategyParams,
                anchor: f32,
                block_map: crate::recompress::measure::BlockErrorMap,
            }
            #[cfg(feature = "recompress-iqa")]
            let mut best_ext: Option<BestExt> = None;

            // Expected generation-loss for this cell, looked up ONCE on the
            // `target` axis — that is the axis the GEXP tables were fit on
            // (median g when the router aims at `target`). It must NOT be
            // indexed by the dial: inverse calibration routinely moves the
            // dial off the target, and the dial also bumps during the loop,
            // so a dial lookup reads a cell GEXP was never fit for.
            let gexp = crate::recompress::calibration::per_encoder::gexp_lookup(
                encoder_class,
                source_est,
                target,
            );

            for iter in 0..hard_cap {
                // A refinement pass that errors must not discard an
                // already-good earlier result; only propagate if the very
                // first pass failed (nothing to fall back to).
                let (outcome, akind) = match run_strategy(&cur) {
                    Ok(v) => v,
                    Err(e) if best.is_some() => {
                        let _ = e;
                        break;
                    }
                    Err(e) => return Err(e),
                };
                budget_state.note_iteration();
                #[cfg(feature = "recompress-iqa")]
                let (g, block_map) = match measure_ctx.as_ref() {
                    Some(ctx) => {
                        let (score, blocks) = ctx.score_with_blocks(&outcome.bytes)?;
                        (Some(score), Some(blocks))
                    }
                    None => (None, None),
                };
                #[cfg(not(feature = "recompress-iqa"))]
                let g: Option<f32> = None;
                let a_hat = match (g, gexp) {
                    // Per-image correction: shift the calibration anchor by
                    // the measured generation-loss deviation from the cell
                    // median (within-cell slope ≈ 1.1, corr 0.80).
                    (Some(gv), Some(ge)) => {
                        anchor + crate::recompress::calibration::per_encoder::GEXP_SLOPE * (gv - ge)
                    }
                    // OneShot (no measurement) or no GEXP table for this
                    // encoder: trust the calibration anchor.
                    _ => anchor,
                };

                // Prefer the smallest output among passes that clear
                // target (a_hat ≥ target − TOL); otherwise the highest
                // a_hat (closest to target).
                let replace = match &best {
                    None => true,
                    Some((bbytes, _, _, ba)) => {
                        let cur_ok = a_hat >= target - CLOSED_LOOP_TOL;
                        let best_ok = *ba >= target - CLOSED_LOOP_TOL;
                        match (cur_ok, best_ok) {
                            (true, true) => outcome.bytes.len() < bbytes.len(),
                            (true, false) => true,
                            (false, true) => false,
                            (false, false) => a_hat > *ba,
                        }
                    }
                };
                if replace {
                    best = Some((outcome.bytes.clone(), akind, g, a_hat));
                    #[cfg(feature = "recompress-iqa")]
                    {
                        best_ext = block_map.map(|blocks| BestExt {
                            params: cur,
                            anchor,
                            block_map: blocks,
                        });
                    }
                }

                // Stop: no measurement (OneShot), target reached, iteration
                // backstop hit, or the wall-clock / iteration budget is
                // spent (`may_measure` honors `MaxTime`'s deadline).
                if !measure
                    || a_hat >= target - CLOSED_LOOP_TOL
                    || iter + 1 >= hard_cap
                    || !budget_state.may_measure()
                {
                    break;
                }
                // Bump the dial to cover the shortfall and re-derive params.
                let Some(new_dial) = next_dial(cur.dial_zensim_a, a_hat, target) else {
                    break; // dial saturated — can't push higher
                };
                let new_q = crate::recompress::target::target_zensim_a_to_ijg_q(new_dial);
                // No-progress guard: the strategies quantize off
                // `target_ijg_q`, and the q-curve plateaus at the top (dial
                // ≥ 90 → q 100). If the bump doesn't change q, the re-encode
                // is byte-identical — stop rather than burn iterations (and
                // inflate `anchor`) on an unchanged output.
                if new_q == cur.target_ijg_q {
                    break;
                }
                anchor += new_dial - cur.dial_zensim_a;
                cur.dial_zensim_a = new_dial;
                cur.target_ijg_q = new_q;
                cur.target_ba_distance =
                    crate::recompress::target::target_zensim_a_to_ba_distance(new_dial);
            }

            // Diffmap-guided per-block refinement (Lever 5): when the
            // winning candidate is Preserve, was measured, and overshoots
            // the target with iteration budget left, spend the remaining
            // passes converting measured per-block slack into bytes (and
            // un-zeroing blocks the measured map flags). See the
            // `refine` module docs. Refinement errors never discard the
            // incumbent — worst case it returns nothing.
            #[cfg(feature = "recompress-iqa")]
            {
                let incumbent = best
                    .as_ref()
                    .map(|(bytes, kind, _, a_hat)| (*kind, bytes.len(), *a_hat));
                if let Some((StrategyKind::Preserve, incumbent_len, incumbent_a_hat)) = incumbent
                    && let Some(ctx) = measure_ctx.as_ref()
                    && let Some(ext) = best_ext.as_ref()
                    && let Some(ge) = gexp
                {
                    let max_passes = hard_cap.saturating_sub(budget_state.iterations_used);
                    let refined = crate::recompress::refine::refine_preserve(
                        jpeg_bytes,
                        &analysis,
                        &ext.params,
                        ctx,
                        crate::recompress::refine::RefineInputs {
                            anchor: ext.anchor,
                            gexp: ge,
                            target,
                            incumbent_len,
                            incumbent_a_hat,
                            block_map: &ext.block_map,
                            max_passes,
                        },
                        &mut budget_state,
                    )
                    .unwrap_or(None);
                    if let Some(r) = refined {
                        best = Some((r.bytes, StrategyKind::Preserve, Some(r.g), r.a_hat));
                    }
                }
            }

            let (bytes, actual_kind, measured_g, a_hat) =
                best.expect("closed loop runs at least one pass");

            let ratio = bytes.len() as f32 / jpeg_bytes.len() as f32;
            if ratio >= 1.0 {
                // Calibration said this strategy would win on size; it
                // didn't. Honor the no-size-regression invariant by
                // falling back to lossless — and if even lossless
                // inflates, fall back to the source bytes verbatim.
                let lossless = strategies::lossless::run_lossless(jpeg_bytes, &analysis)?;
                let bytes = if lossless.len() > jpeg_bytes.len() {
                    jpeg_bytes.to_vec()
                } else {
                    lossless
                };
                return Ok(RecompressResult::LosslessOnly {
                    bytes,
                    reason: LosslessReason::RecompressionWouldInflateAtTarget,
                });
            }

            Ok(RecompressResult::Recompressed {
                bytes,
                strategy: actual_kind,
                // Report the prediction for the *shipped* bytes: `a_hat`
                // folds the measured generation loss into the calibration
                // anchor and tracks dial bumps. For `OneShot` (no
                // measurement, no bump) `a_hat == projected_zensim_a`, so
                // this is identical to the router's projection there.
                projected_zensim_a: a_hat,
                measured_zensim_a: measured_g,
                source_to_output_ratio: ratio,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn next_dial_bumps_by_shortfall() {
        // a_hat 5 below target → bump by 5.
        assert_eq!(next_dial(50.0, 55.0, 60.0), Some(55.0));
    }

    #[test]
    fn next_dial_has_minimum_bump_of_one() {
        // Tiny shortfall still moves at least 1.0 so the loop progresses.
        assert_eq!(next_dial(50.0, 59.8, 60.0), Some(51.0));
    }

    #[test]
    fn next_dial_caps_at_100() {
        // Large shortfall is clamped to the dial ceiling.
        assert_eq!(next_dial(95.0, 40.0, 90.0), Some(100.0));
    }

    #[test]
    fn next_dial_saturates_to_none() {
        // Already at (or within 0.5 of) the ceiling → no headroom → stop.
        assert_eq!(next_dial(100.0, 40.0, 90.0), None);
        assert_eq!(next_dial(99.6, 40.0, 90.0), None);
    }

    #[test]
    fn next_dial_progress_is_monotone_and_terminating() {
        // Iterating from any start always strictly increases the dial
        // until it saturates, so the closed loop cannot spin forever.
        let mut d = 30.0_f32;
        let mut steps = 0;
        // a_hat held pessimistically far below target (worst case).
        while let Some(nd) = next_dial(d, 0.0, 90.0) {
            assert!(nd > d, "dial must strictly increase");
            d = nd;
            steps += 1;
            assert!(steps < 200, "must terminate");
        }
        assert!(d >= 99.5, "terminates only at the ceiling");
    }
}
