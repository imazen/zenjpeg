//! Codec-side rescue plumbing for the two-shot pass framework.
//!
//! Encode + verify themselves stay codec-side (zenpredict has no
//! decoder, no zensim, no awareness of jpegli-q / lambda / chroma
//! scaling). What zenpredict contributes is the decision logic and
//! shared types so every codec wrapper writes the same orchestration
//! instead of inventing its own threshold rule:
//!
//! ```ignore
//! let top2 = predictor.argmin_masked_top_k::<2>(&features, &mask, transform, offsets)?;
//! let first = top2[0].expect("mask non-empty");
//! let bytes_0 = codec::encode(first);
//! let achieved = codec::verify(&bytes_0, &source);
//!
//! match zenpredict::should_rescue(achieved, target_zq, &policy) {
//!     RescueDecision::Ship => bytes_0,
//!     RescueDecision::Rescue => {
//!         let bytes_1 = match policy.strategy {
//!             RescueStrategy::SecondBestPick => codec::encode(top2[1].unwrap_or(first)),
//!             RescueStrategy::ConservativeBump => codec::encode_bumped(first, target_zq, achieved),
//!             RescueStrategy::KnownGoodFallback => codec::encode_known_good(target_zq),
//!         };
//!         codec::ship_best_of(bytes_0, bytes_1)
//!     }
//! }
//! ```
//!
//! The two-shot bound is enforced by the codec — this module does
//! not loop.
//!
//! Rescue is **step 4** of the picker safety pipeline — see
//! [`crate::picker_safety`] for the full canonical order (unachievable-zone
//! resolve → knob vetoes → masked argmin → rescue) and how the pre-argmin
//! overlays compose. Note the two fallback paths are distinct: a zone fallback
//! (step 1) fires when the target is *physically unreachable* for the size
//! class (deterministic, pre-encode); [`RescueStrategy::KnownGoodFallback`]
//! fires when the picker *was consulted and undershot* (post-encode).

#[non_exhaustive]
#[derive(Clone, Copy, Debug)]
pub struct RescuePolicy {
    /// Trigger rescue when `achieved < target − rescue_threshold`.
    /// Units match the codec's quality scale (zensim points). Default
    /// `3.0` is a placeholder pending held-out p99 zensim shortfall
    /// calibration.
    pub rescue_threshold: f32,
    pub strategy: RescueStrategy,
}

impl Default for RescuePolicy {
    fn default() -> Self {
        Self {
            rescue_threshold: 3.0,
            strategy: RescueStrategy::ConservativeBump,
        }
    }
}

impl RescuePolicy {
    /// Construct a `RescuePolicy` from the bake's
    /// `zentrain.safety_compact` summary, falling back to defaults
    /// for fields the bake didn't populate. Use this instead of
    /// [`RescuePolicy::default`] in production codecs — the
    /// bake-tuned threshold is calibration-anchored to the actual
    /// shortfall distribution of the training corpus.
    ///
    /// `profile_strict = false` reads
    /// `safety_compact.rescue_threshold_default` (or 3.0 fallback);
    /// `profile_strict = true` reads the strict-tier threshold
    /// (typically tighter, e.g. 1.0 pp). Both fields live in the
    /// `_reserved` tail of the v1 [`crate::SafetyCompact`] today;
    /// future bumps to `version` will surface them as named fields.
    ///
    /// Returns the default policy when the bake has no
    /// `safety_compact` (older bakes).
    pub fn from_bake(model: &crate::Model, profile_strict: bool) -> Self {
        let mut policy = Self::default();
        let Some(safety) = model.safety_compact() else {
            return policy;
        };
        // For v1 SafetyCompact, both rescue thresholds live in
        // `_reserved` as two trailing f32s. Read defensively.
        let rescue_default = f32::from_le_bytes(
            safety._reserved[0..4]
                .try_into()
                .expect("4-byte slice → [u8;4]"),
        );
        let rescue_strict = f32::from_le_bytes(
            safety._reserved[4..8]
                .try_into()
                .expect("4-byte slice → [u8;4]"),
        );
        let chosen = if profile_strict {
            rescue_strict
        } else {
            rescue_default
        };
        if chosen.is_finite() && chosen > 0.0 {
            policy.rescue_threshold = chosen;
        }
        policy
    }
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RescueStrategy {
    /// Bump quality (codec-defined: jpegli-q, lambda, distance) and
    /// disable lossy chroma options. Single deterministic transform
    /// of the first-pick `EncoderConfig`.
    ConservativeBump,
    /// Use the cached second-best argmin. Codec falls through to
    /// `ConservativeBump` if the second-best's predicted bytes are
    /// too close to the first-best (no margin → unlikely to help).
    SecondBestPick,
    /// Hard-coded "always-safe" config for the target band — used
    /// when the picker's confidence is too low to trust either pick.
    KnownGoodFallback,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RescueDecision {
    Ship,
    Rescue,
}

/// Predicate the codec calls after pass-0 verify to decide whether
/// to run a rescue pass. Non-finite `achieved` is treated as
/// verify-failed and forces rescue.
///
/// # Examples
///
/// ```
/// use zenpredict::{RescueDecision, RescuePolicy, should_rescue};
///
/// let policy = RescuePolicy::default();   // 3.0 pp threshold
/// // achieved 83 vs target 85 (gap 2 pp) → ship.
/// assert_eq!(should_rescue(83.0, 85.0, &policy), RescueDecision::Ship);
/// // achieved 81 vs target 85 (gap 4 pp) → rescue.
/// assert_eq!(should_rescue(81.0, 85.0, &policy), RescueDecision::Rescue);
/// // verify failed (NaN) → rescue.
/// assert_eq!(should_rescue(f32::NAN, 85.0, &policy), RescueDecision::Rescue);
/// ```
pub fn should_rescue(achieved: f32, target: f32, policy: &RescuePolicy) -> RescueDecision {
    if !achieved.is_finite() {
        return RescueDecision::Rescue;
    }
    if achieved < target - policy.rescue_threshold {
        RescueDecision::Rescue
    } else {
        RescueDecision::Ship
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ships_when_within_threshold() {
        let policy = RescuePolicy::default();
        assert_eq!(should_rescue(83.0, 85.0, &policy), RescueDecision::Ship);
        assert_eq!(should_rescue(85.0, 85.0, &policy), RescueDecision::Ship);
        assert_eq!(should_rescue(90.0, 85.0, &policy), RescueDecision::Ship);
    }

    #[test]
    fn rescues_on_threshold_breach() {
        let policy = RescuePolicy::default();
        assert_eq!(should_rescue(81.0, 85.0, &policy), RescueDecision::Rescue);
    }

    #[test]
    fn rescues_on_nonfinite_achieved() {
        let policy = RescuePolicy::default();
        assert_eq!(
            should_rescue(f32::NAN, 85.0, &policy),
            RescueDecision::Rescue
        );
        assert_eq!(
            should_rescue(f32::INFINITY, 85.0, &policy),
            RescueDecision::Rescue
        );
        assert_eq!(
            should_rescue(f32::NEG_INFINITY, 85.0, &policy),
            RescueDecision::Rescue
        );
    }

    #[test]
    fn custom_threshold_overrides_default() {
        let policy = RescuePolicy {
            rescue_threshold: 1.0,
            strategy: RescueStrategy::SecondBestPick,
        };
        assert_eq!(should_rescue(83.0, 85.0, &policy), RescueDecision::Rescue);
        assert_eq!(should_rescue(84.5, 85.0, &policy), RescueDecision::Ship);
    }
}
