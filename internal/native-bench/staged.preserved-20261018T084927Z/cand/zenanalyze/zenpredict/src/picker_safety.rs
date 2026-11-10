//! Picker safety pipeline — the one canonical order a codec composes the
//! safety overlays in, so every deployed single-encode picker enforces the
//! same bake-time-validated bounds the same way.
//!
//! The safety system is three cooperating layers, applied in a fixed order:
//!
//! 1. **Unachievable-zone resolve** ([`UnachievableZones::resolve`]) — the
//!    size discriminant. If `target_zq` exceeds the achievable ceiling for the
//!    image's size class, the target is physically unreachable: short-circuit
//!    to the declared fallback knobset (best-achievable cell + scalar). No
//!    argmin, no veto — there is nothing to pick among.
//! 2. **Knob vetoes** ([`apply_knob_vetoes`]) — feature-gated per-cell mask
//!    that bounds the catastrophic RD tail (forbid value `V` on a categorical
//!    axis when feature `F {op} threshold`). Applied to the reach mask.
//! 3. **Masked argmin** (codec) — pick min predicted bytes over
//!    `reach & ~veto`; if the vetoes stranded the row (no cell left), fall
//!    back to the un-vetoed reach mask so a row is never stranded.
//! 4. **Two-shot rescue** (codec, post-encode — [`crate::rescue`]) — after the
//!    pass-0 encode + verify, if the achieved quality undershoots `target_zq`
//!    by more than the bake-tuned threshold, run one rescue pass
//!    ([`crate::should_rescue`] + [`crate::RescueStrategy`]); the
//!    [`crate::FallbackEntry`] table backs `KnownGoodFallback`.
//!
//! Steps **1-2 are pre-argmin and pure** (no encoder, no I/O) — this module
//! composes them into a single [`resolve_pre_argmin`] call returning a
//! [`PreArgminDecision`]. Steps **3-4 are the codec's** (it owns the encoder,
//! the verify, and the never-strand fallback around its argmin).
//!
//! ## Granular quality targeting (runtime contract)
//!
//! The picker is conditioned on a **continuous** quality target: the training
//! input carries `zq_norm = target_zq / 100` plus the `zq_norm²`,
//! `zq_norm·log_px`, and `zq_norm·feature` interaction terms, and the student
//! is a LeakyReLU MLP (piecewise-linear ⇒ continuous in its inputs) whose
//! quality-relevant outputs carry no snap-to-discrete `OutputSpec`. The
//! deployed picker therefore smoothly targets *any* granular quality value,
//! not just the training-grid points — verified monotonic (predicted bytes
//! rise smoothly with the target, 100% monotone across sampled images, no
//! discontinuities).
//!
//! For that to hold at inference the codec MUST build the engineered input
//! from the **raw, unsnapped** target: `zq_norm = target_zq / 100` for the
//! caller's granular `target_zq`, and the matching `zq_norm²` /
//! `zq_norm·log_px` / `zq_norm·feature` terms — never round `target_zq` to the
//! training grid first (that would step the picker and defeat granular
//! targeting). The unachievable-zone check above is likewise continuous in
//! `target_zq` (a strict `target_zq > ceiling_zq` compare), so the granular
//! boundary is exact, not grid-quantized.
//!
//! Distinction between the two fallback paths (both yield a knobset, but they
//! fire on different signals and must not be conflated):
//!
//! - **Zone fallback** (step 1) — the target is *physically unreachable* for
//!   the size class. Deterministic, pre-encode, size-keyed. The picker is not
//!   wrong; the request is impossible, so encode the best-achievable knobset.
//! - **Rescue `KnownGoodFallback`** (step 4) — the picker *was consulted* and
//!   its pass-0 pick *undershot* (OOD input, low confidence). Post-encode,
//!   zq-keyed via [`crate::fallback_for`].
//!
//! ## Worked example
//!
//! ```rust
//! use zenpredict::{
//!     AllowedMask, PreArgminDecision, ScoreTransform, UnachievableZones,
//!     argmin_masked, resolve_pre_argmin,
//! };
//! # fn encode(_cell: usize, _scalar: Option<f32>) {}
//! # fn run(zones: &UnachievableZones, vetoes: &[zenpredict::KnobVeto<'_>],
//! #        features: &[f32], predicted_bytes: &[f32], target_zq: f32, reach: &[bool]) {
//! // `reach` = cells whose predicted metric >= target_zq (the caller's mask).
//! let mut allowed = reach.to_vec();              // veto pass mutates a copy
//! match resolve_pre_argmin(features, target_zq, zones, vetoes, &mut allowed) {
//!     PreArgminDecision::ZoneFallback(fb) => {
//!         // physically unreachable target -> declared best-achievable knobset
//!         encode(fb.cell, fb.scalar);
//!     }
//!     PreArgminDecision::Argmin => {
//!         let pick = argmin_masked(predicted_bytes, &AllowedMask::new(&allowed),
//!                                  ScoreTransform::Exp, None)
//!             // never-strand: vetoes emptied the mask -> revert to reach
//!             .or_else(|| argmin_masked(predicted_bytes, &AllowedMask::new(reach),
//!                                       ScoreTransform::Exp, None));
//!         if let Some(cell) = pick { encode(cell, None); }
//!     }
//! }
//! // ...then pass-0 verify + crate::should_rescue(...) for step 4.
//! # }
//! ```

use crate::knob_veto::{KnobVeto, VetoOp, ZQ_VETO_SENTINEL, apply_knob_vetoes};
use crate::unachievable_zone::{UnachievableZones, ZoneFallback};

/// Outcome of the pre-argmin safety pipeline (steps 1-2).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PreArgminDecision {
    /// `target_zq` is physically unreachable for the image's size class.
    /// Encode this declared fallback knobset directly — do **not** argmin.
    ZoneFallback(ZoneFallback),
    /// No unachievable zone fired. The vetoes (if any) have been applied to
    /// the caller's `allowed` mask; proceed to the masked argmin (with the
    /// never-strand fallback to the un-vetoed reach mask).
    Argmin,
}

/// Compose the pre-argmin safety overlays in the canonical order:
/// unachievable-zone resolve **then** knob vetoes (see the [module docs](self)).
///
/// - Returns [`PreArgminDecision::ZoneFallback`] when
///   [`UnachievableZones::resolve`] fires for `(features, target_zq)` — the
///   target is unreachable for the size class, so the vetoes are **not**
///   applied (there is no pick to constrain) and the caller encodes the
///   fallback knobset.
/// - Otherwise applies [`apply_knob_vetoes`] to `allowed` in place and returns
///   [`PreArgminDecision::Argmin`].
///
/// `allowed` is the picker's mask backing slice — typically a **copy** of the
/// reach mask (`predicted metric >= target_zq`). Pass a copy and keep the
/// original reach mask so the codec can compose the never-strand fallback
/// around its argmin (revert to reach when the vetoes empty the mask); see the
/// [module docs](self) example. `features` is the bake's feat_cols-ordered
/// vector (the same the picker forward pass consumes). Panic-free: every
/// out-of-range index in either overlay is skipped.
pub fn resolve_pre_argmin(
    features: &[f32],
    target_zq: f32,
    zones: &UnachievableZones,
    vetoes: &[KnobVeto<'_>],
    allowed: &mut [bool],
) -> PreArgminDecision {
    // Step 1: zone resolve runs FIRST. An unreachable target has no valid
    // pick, so vetoes (which only constrain a pick) are irrelevant — skip them
    // and hand back the declared fallback.
    if let Some(fb) = zones.resolve(features, target_zq) {
        return PreArgminDecision::ZoneFallback(fb);
    }
    // Step 2: feature-gated knob vetoes mask the catastrophic-tail cells.
    // (apply_knob_vetoes skips `__zq__` sentinel vetoes — feat_idx out of range.)
    apply_knob_vetoes(features, vetoes, allowed);
    // Step 2b: `__zq__` vetoes gate on the caller's target quality (the sentinel
    // feat_idx), which the feature-indexed pass above can't see. Apply them here
    // using `target_zq`. NaN-safe: a non-finite target fires neither comparison.
    for v in vetoes {
        if v.feat_idx == ZQ_VETO_SENTINEL {
            let fires = match v.op {
                VetoOp::LessThan => target_zq < v.threshold,
                VetoOp::GreaterThan => target_zq > v.threshold,
            };
            if fires {
                for &c in v.cells {
                    if let Some(slot) = allowed.get_mut(c as usize) {
                        *slot = false;
                    }
                }
            }
        }
    }
    PreArgminDecision::Argmin
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knob_veto::VetoOp;
    use crate::unachievable_zone::parse_unachievable_zones;
    use alloc::vec::Vec;

    // tiny/small/medium/large-ish table: feat_pixel_count at idx 0, one zone
    // (pixels <= 1e6 -> ceiling 90, fallback cell 2 @ scalar 9).
    fn zones_one() -> UnachievableZones {
        let mut blob = Vec::new();
        blob.extend_from_slice(&0u16.to_le_bytes()); // pixels_feat_idx = 0
        blob.push(1); // n_zones
        blob.extend_from_slice(&1_000_000.0f32.to_le_bytes()); // pixel_upper
        blob.extend_from_slice(&90.0f32.to_le_bytes()); // ceiling_zq
        blob.push(2); // fallback_cell
        blob.extend_from_slice(&9.0f32.to_le_bytes()); // fallback_scalar
        parse_unachievable_zones(&blob).unwrap()
    }

    #[test]
    fn unreachable_target_short_circuits_to_zone_fallback_skipping_vetoes() {
        let zones = zones_one();
        // A veto that WOULD fire (feat[1] > 0.5 forbids cell 0) — must be
        // skipped because the zone fires first.
        let veto = KnobVeto {
            feat_idx: 1,
            op: VetoOp::GreaterThan,
            threshold: 0.5,
            cells: &[0],
        };
        let features = [500_000.0, 1.0]; // pixels in zone; veto-feature would fire
        let mut allowed = [true, true, true];
        let d = resolve_pre_argmin(&features, 94.0, &zones, &[veto], &mut allowed);
        assert_eq!(
            d,
            PreArgminDecision::ZoneFallback(ZoneFallback {
                cell: 2,
                scalar: Some(9.0)
            })
        );
        // vetoes NOT applied — mask untouched (the zone short-circuits).
        assert_eq!(allowed, [true, true, true]);
    }

    #[test]
    fn reachable_target_applies_vetoes_then_argmin() {
        let zones = zones_one();
        let veto = KnobVeto {
            feat_idx: 1,
            op: VetoOp::GreaterThan,
            threshold: 0.5,
            cells: &[0],
        };
        let features = [500_000.0, 1.0]; // in zone class, but target reachable
        let mut allowed = [true, true, true];
        // target 80 <= ceiling 90 -> not a zone -> vetoes apply.
        let d = resolve_pre_argmin(&features, 80.0, &zones, &[veto], &mut allowed);
        assert_eq!(d, PreArgminDecision::Argmin);
        assert_eq!(allowed, [false, true, true]); // cell 0 vetoed
    }

    #[test]
    fn no_zones_no_vetoes_is_plain_argmin() {
        let zones = UnachievableZones::default();
        let mut allowed = [true, true];
        let d = resolve_pre_argmin(&[123.0], 99.0, &zones, &[], &mut allowed);
        assert_eq!(d, PreArgminDecision::Argmin);
        assert_eq!(allowed, [true, true]);
    }

    #[test]
    fn zq_sentinel_veto_gates_on_target_quality() {
        // `__zq__` veto (sentinel feat_idx): forbid cell 0 when target_zq > 62.5.
        // The plain feature pass can't see target_zq; resolve_pre_argmin applies it.
        let zones = UnachievableZones::default();
        let veto = KnobVeto {
            feat_idx: ZQ_VETO_SENTINEL,
            op: VetoOp::GreaterThan,
            threshold: 62.5,
            cells: &[0],
        };
        // target 70 > 62.5 → fires → cell 0 denied.
        let mut a = [true, true, true];
        let d = resolve_pre_argmin(&[1.0], 70.0, &zones, &[veto], &mut a);
        assert_eq!(d, PreArgminDecision::Argmin);
        assert_eq!(a, [false, true, true]);
        // target 50 < 62.5 → does not fire.
        let mut a2 = [true, true, true];
        resolve_pre_argmin(&[1.0], 50.0, &zones, &[veto], &mut a2);
        assert_eq!(a2, [true, true, true]);
        // The feature-indexed pass alone (apply_knob_vetoes) must SKIP the
        // sentinel (out of range) — features has no index 0xFFFF.
        let mut a3 = [true, true, true];
        crate::apply_knob_vetoes(&[1.0], &[veto], &mut a3);
        assert_eq!(a3, [true, true, true]);
    }

    #[test]
    fn granular_target_respected_at_zone_boundary() {
        // The boundary uses the continuous target_zq, not a grid step: 90.0 is
        // reachable (== ceiling), 90.01 is not. Smooth, granular handling.
        let zones = zones_one();
        let features = [500_000.0];
        let mut a1 = [true, true, true];
        assert_eq!(
            resolve_pre_argmin(&features, 90.0, &zones, &[], &mut a1),
            PreArgminDecision::Argmin
        );
        let mut a2 = [true, true, true];
        assert!(matches!(
            resolve_pre_argmin(&features, 90.01, &zones, &[], &mut a2),
            PreArgminDecision::ZoneFallback(_)
        ));
    }
}
