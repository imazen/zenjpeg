//! Expert-only knobs for codec calibration and picker training.
//!
//! Anything in this module is **unstable**: it may change in any patch
//! release without semver justification, and is **not part of the
//! public API contract**. Reach for it only when:
//!
//! 1. Sweeping parameter combinations to feed a picker / regression /
//!    calibration training pipeline.
//! 2. Diagnosing codec behaviour by overriding speed-preset defaults.
//! 3. Wiring a future `predict` feature that selects [`InternalParams`]
//!    via a baked MLP.
//!
//! Everything in here lives behind the `__expert` cargo feature, whose
//! double-underscore signals "private — do not depend on this in
//! production code." Default builds expose only stable public knobs
//! ([`crate::EncoderConfig::with_quality`], etc.).
//!
//! # Migration TODO
//!
//! This struct is currently a placeholder. The full audit of which
//! `EncoderConfig` knobs added since zenjpeg 0.8.4 should move into
//! `InternalParams` is tracked at:
//! <https://github.com/imazen/zenjpeg/issues> (TBD — pending the
//! Phase B/C/1.5 picker calibration agent's WIP landing first).
//!
//! Candidate knobs identified so far (need verification + migration):
//! - the SA-optimized piecewise quant tables v4 (PR #121) — unsure
//!   if these are exposed via EncoderConfig or stay internal.
//! - the `analyze::adaptive` likelihood-driven knob selectors (the
//!   composites that were just stub-ported in b82dbe32).
//! - boundary-rd refinement parameters beyond the existing
//!   `BoundaryRd::On|Off` toggle.
//! - target_zq tuning knobs (currently behind `target-zq` feature).

/// Expert override knobs for the JPEG encoder.
///
/// **Currently a placeholder.** Will be populated in a follow-up PR
/// once the audit of post-0.8.4 EncoderConfig surface is complete.
/// Each future field will be `Option<T>` so `None` (the [`Default`])
/// keeps existing behaviour and `Some(_)` overrides.
///
/// `#[non_exhaustive]` — fields may be added in any patch release.
#[non_exhaustive]
#[derive(Default, Clone, Debug)]
pub struct InternalParams {
    // intentionally empty for now; see module-level docs.
}
