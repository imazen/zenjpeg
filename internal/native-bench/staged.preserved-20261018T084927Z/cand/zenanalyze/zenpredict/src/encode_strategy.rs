//! Trial-encode strategy + budget for multi-cell picker disambiguation.
//!
//! A K=1 picker's top-1 prediction is free (no encode), but for multi-cell codecs
//! (webp/avif) it can mis-pick a content-underdetermined knob — webp's 5-way `filter`
//! is only ~32% predictable from content, vs ~94% for `method` and ~62% for
//! `sharp_yuv` (held-out, 2026-06-29). A *multi-shot* pass trial-encodes the picker's
//! top-K ranked candidates (which differ only in the ambiguous knob) and keeps the
//! best by the codec's metric — accurate, but it costs extra encodes whose cost scales
//! with image size.
//!
//! Two orthogonal controls:
//! - [`PickerStrategy`] — the codec's *mode*: trust the pick, force trials, or adapt.
//! - [`EncodeBudget`] — a *multi-axis resource ceiling* the **application** sets
//!   (max passes / trial pixels / wall-time ms). The most restrictive binds.
//!
//! The per-encode time estimate is **not** in the budget — the application shouldn't
//! have to guess it. The *codec* owns a cost model, so it passes its own
//! `est_ms_per_encode` into [`EncodeBudget::resolve`] when it resolves the ms axis.
//! [`EncodeBudget::time_exhausted`] is the runtime safety net for when that estimate
//! undershoots: the codec checks its measured clock and stops.

/// The codec's trial-encode mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PickerStrategy {
    /// Trust the picker's top-1: a single encode, no trials (strict real-time).
    OneShot,
    /// Always trial-encode up to the budget (quality-first).
    MultiShot,
    /// Let the budget decide — size/time-adaptive. The recommended default: a large
    /// image collapses to one-shot (no trial fits the pixel/time ceiling), a small one
    /// trials up to the budget.
    Auto,
}

/// The application's encode profile: a latency × effort point on the
/// fastest→aggressive spectrum. Drives codec routing (real-time profiles prefer fast
/// codecs — see the zenpicker meta-router), the per-codec trial [`PickerStrategy`], and
/// the codec's internal effort tier (`Fastest` / `Balanced` / `Aggressive` — the codec
/// matches the variant). Only the four sensible combos exist (no `RealtimeAggressive`
/// — too slow for real-time; no `QueuedFastest` — no reason to queue the fastest path).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodeMode {
    /// Real-time, minimum latency: fastest codecs + lowest effort, one-shot.
    RealtimeFastest,
    /// Real-time, more effort within the latency budget: one-shot, mid effort.
    RealtimeBalanced,
    /// Queued / offline, balanced effort: multi-shot trials allowed (size/time-adaptive).
    QueuedBalanced,
    /// Queued / offline, maximum quality: full multi-shot + offline metric-K-verify,
    /// highest effort, all codecs viable.
    QueuedAggressive,
}

impl EncodeMode {
    /// Whether this is a latency-sensitive (real-time) profile.
    pub fn is_realtime(self) -> bool {
        matches!(
            self,
            EncodeMode::RealtimeFastest | EncodeMode::RealtimeBalanced
        )
    }

    /// The default per-codec trial strategy this profile implies. Real-time profiles
    /// stay one-shot (trials add latency); queued profiles trial — `Auto` adapts to
    /// size, `Aggressive` forces the full multi-shot.
    pub fn strategy(self) -> PickerStrategy {
        match self {
            EncodeMode::RealtimeFastest | EncodeMode::RealtimeBalanced => PickerStrategy::OneShot,
            EncodeMode::QueuedBalanced => PickerStrategy::Auto,
            EncodeMode::QueuedAggressive => PickerStrategy::MultiShot,
        }
    }
}

/// Multi-axis resource ceiling for trial encodes, set by the **application**. Any
/// subset of axes may be set; the **most restrictive binds**. All-unset = unbounded
/// (capped only by the candidate count). No per-encode-time estimate lives here — the
/// codec supplies that to [`Self::resolve`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EncodeBudget {
    /// Hard cap on trial encodes. `None` = not bounded by pass count.
    pub max_passes: Option<u16>,
    /// Pixel ceiling across trial encodes (size-adaptive: a large image fits fewer
    /// passes). `None` = not bounded by pixels.
    pub max_trial_pixels: Option<u64>,
    /// Wall-time ceiling (ms) for trials. `None` = not bounded by time.
    pub max_ms: Option<u32>,
}

impl EncodeBudget {
    /// Bound by a hard pass count only.
    pub const fn passes(n: u16) -> Self {
        Self {
            max_passes: Some(n),
            max_trial_pixels: None,
            max_ms: None,
        }
    }
    /// Bound by a trial-pixel ceiling only (size-adaptive).
    pub const fn trial_pixels(px: u64) -> Self {
        Self {
            max_passes: None,
            max_trial_pixels: Some(px),
            max_ms: None,
        }
    }
    /// Bound by a wall-time ceiling only.
    pub const fn milliseconds(ms: u32) -> Self {
        Self {
            max_passes: None,
            max_trial_pixels: None,
            max_ms: Some(ms),
        }
    }

    /// How many of the picker's `n_candidates` ranked cells to trial-encode under
    /// `strategy`. Binds the most restrictive set axis: `max_passes`,
    /// `max_trial_pixels / image_pixels`, and `max_ms / est_ms_per_encode` — where
    /// `est_ms_per_encode` is the **codec's own** estimate for this image (codecs have
    /// cost models; pass `0` if unknown, and rely on [`Self::time_exhausted`] at
    /// runtime instead). Floored at 1 (the top pick always encodes), capped at
    /// `n_candidates`. `OneShot` is always 1.
    pub fn resolve(
        &self,
        strategy: PickerStrategy,
        n_candidates: usize,
        image_pixels: u64,
        est_ms_per_encode: u32,
    ) -> usize {
        let avail = n_candidates.max(1);
        match strategy {
            PickerStrategy::OneShot => 1,
            PickerStrategy::MultiShot | PickerStrategy::Auto => {
                let mut p = avail; // unbounded start; every set axis tightens it
                if let Some(mp) = self.max_passes {
                    p = p.min(mp as usize);
                }
                // `checked_div` folds in the "divisor > 0" guard (returns `None` at 0).
                if let Some(px) = self.max_trial_pixels
                    && let Some(per) = px.checked_div(image_pixels)
                {
                    p = p.min(per as usize);
                }
                if let Some(ms) = self.max_ms
                    && let Some(per) = ms.checked_div(est_ms_per_encode)
                {
                    p = p.min(per as usize);
                }
                p.max(1).min(avail)
            }
        }
    }

    /// Runtime safety net: `true` once the measured wall-time budget is spent. The
    /// codec calls this in its trial loop with the elapsed time since it began
    /// encoding this image, catching estimate undershoot. Always `false` when `max_ms`
    /// is unset.
    pub fn time_exhausted(&self, elapsed_ms: u32) -> bool {
        self.max_ms.is_some_and(|ms| elapsed_ms >= ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use PickerStrategy::*;

    #[test]
    fn one_shot_ignores_budget() {
        assert_eq!(EncodeBudget::passes(5).resolve(OneShot, 5, 1_000, 10), 1);
    }

    #[test]
    fn pass_and_candidate_caps() {
        assert_eq!(EncodeBudget::passes(3).resolve(MultiShot, 5, 0, 0), 3);
        assert_eq!(EncodeBudget::passes(3).resolve(MultiShot, 2, 0, 0), 2);
    }

    #[test]
    fn trial_pixels_is_size_adaptive() {
        let (small, large) = (256 * 256, 4096u64 * 4096);
        assert_eq!(
            EncodeBudget::trial_pixels(3 * small).resolve(Auto, 4, small, 0),
            3
        );
        assert_eq!(
            EncodeBudget::trial_pixels(3 * small).resolve(Auto, 4, large, 0),
            1
        );
    }

    #[test]
    fn ms_axis_uses_codec_estimate() {
        // 300ms ceiling, codec estimates 100ms/encode -> 3 passes
        assert_eq!(EncodeBudget::milliseconds(300).resolve(Auto, 5, 0, 100), 3);
        // tight ceiling -> one-shot
        assert_eq!(EncodeBudget::milliseconds(50).resolve(Auto, 5, 0, 100), 1);
        // no estimate (0) -> ms axis skipped at resolve; time_exhausted handles runtime
        assert_eq!(
            EncodeBudget::milliseconds(50).resolve(MultiShot, 5, 0, 0),
            5
        );
    }

    #[test]
    fn most_restrictive_axis_binds() {
        // passes allows 5, time allows 2 (codec est 100ms, ceiling 200ms) -> 2
        let b = EncodeBudget {
            max_passes: Some(5),
            max_ms: Some(200),
            max_trial_pixels: None,
        };
        assert_eq!(b.resolve(MultiShot, 8, 0, 100), 2);
    }

    #[test]
    fn time_exhausted_is_runtime_safety() {
        let b = EncodeBudget::milliseconds(200);
        assert!(!b.time_exhausted(100));
        assert!(b.time_exhausted(200));
        assert!(!EncodeBudget::passes(3).time_exhausted(99_999)); // no time axis
    }

    #[test]
    fn degenerate_inputs_stay_valid() {
        assert_eq!(EncodeBudget::passes(5).resolve(MultiShot, 0, 0, 0), 1);
        assert_eq!(EncodeBudget::passes(0).resolve(MultiShot, 5, 0, 0), 1);
        assert_eq!(EncodeBudget::default().resolve(MultiShot, 4, 9_999, 9), 4);
    }

    #[test]
    fn encode_mode_maps_to_strategy() {
        assert_eq!(EncodeMode::RealtimeFastest.strategy(), OneShot);
        assert_eq!(EncodeMode::RealtimeBalanced.strategy(), OneShot);
        assert_eq!(EncodeMode::QueuedBalanced.strategy(), Auto);
        assert_eq!(EncodeMode::QueuedAggressive.strategy(), MultiShot);
        assert!(EncodeMode::RealtimeFastest.is_realtime());
        assert!(!EncodeMode::QueuedAggressive.is_realtime());
    }
}
