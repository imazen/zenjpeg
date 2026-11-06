//! Per-output activation, clamp, snap-to-discrete, sentinel, and
//! sparse hand-tune overrides.
//!
//! Forward-pass output is raw — no activation, no clamping, no
//! rounding. For codec pickers many outputs need a final stage:
//!
//! - **Activation** — sigmoid / sigmoid-then-linear-remap / clamped
//!   exp / round-to-int / pass-through.
//! - **Bounds clamp** — inclusive `[low, high]`, applied after the
//!   activation. Handles the "this output represents `q ∈ [0, 100]`"
//!   case where the regressor sometimes overshoots.
//! - **Snap to discrete set** — codec parameters are often few-valued
//!   (`filter_sharpness ∈ {0..7}`, `partition_limit ∈ {0, 50, 100}`,
//!   `multi_pass_stats ∈ {0, 1}`). The regressor predicts a
//!   continuous value; the spec snaps to the nearest legal value.
//! - **Sentinel** — when the snapped output equals a designated
//!   sentinel value (typically `-1.0`, `f32::NAN`, etc.), the spec
//!   surfaces "use codec default" to the caller instead of a number.
//! - **Sparse overrides** — bake-time hard-coded `(output_index,
//!   value)` pairs that override the model entirely. Lets a
//!   maintainer hand-tune one output without retraining.
//!
//! [`OutputSpec`] carries these knobs as a 32-byte POD record per
//! output. The bake writes one `OutputSpec` per output, pooling
//! discrete-set values into a separate f32 section.
//!
//! `predict_with_specs` runs the forward pass, then for each output
//! `i` applies, in order:
//!
//! 1. `transform` (Identity / Sigmoid / SigmoidScaled / Exp / Round)
//! 2. `bounds` clamp (inclusive)
//! 3. snap to nearest value in the discrete set, if non-empty
//! 4. sentinel match: if the result equals `sentinel`, surface
//!    [`OutputValue::Default`] instead of a number
//! 5. sparse override: if `i` appears in the override list, replace
//!    the result with the override value (or sentinel if the override
//!    is `f32::NAN`)
//!
//! When no `OutputSpec` section is present in the bake the raw
//! [`crate::Predictor::predict`] path stays cheap; `predict_with_specs`
//! is a separate API so callers that don't need the post-processing
//! pay nothing for it.

use bytemuck::{Pod, Zeroable};

use crate::bounds::FeatureBound;

/// Per-output activation kind. One byte on the wire; the variant
/// determines how `transform_params` is interpreted (often `[0.0,
/// 0.0]`, sometimes `[low, high]` for [`OutputTransform::SigmoidScaled`]).
///
/// Applied BEFORE clamping/snapping so the bounds and discrete sets
/// describe the post-activation domain that consumers care about.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
#[non_exhaustive]
pub enum OutputTransform {
    /// Pass through unchanged. The default; outputs are returned raw.
    #[default]
    Identity = 0,
    /// `1 / (1 + exp(-x))`. Maps `(-∞, ∞) → (0, 1)`.
    Sigmoid = 1,
    /// Sigmoid then linear remap to `[low, high]`. Use for outputs
    /// like "q ∈ [0, 100]" where you want a bounded regression.
    /// `transform_params = [low, high]`.
    SigmoidScaled = 2,
    /// `exp(x)` clamped to `x ∈ [-30, 30]` so finite f32 stays finite.
    /// Useful for log-domain regressors emitting `log(bytes)`.
    Exp = 3,
    /// Round to nearest integer (banker's rounding via `f32::round`).
    /// Apply before the discrete-set snap when both are needed —
    /// rounding first lets the snap operate on integers.
    Round = 4,
}

impl OutputTransform {
    /// Decode the wire byte. Unknown variants surface as
    /// [`OutputTransform::Identity`] so a trainer that emits a
    /// future variant doesn't break old runtimes outright; the
    /// parser logs but doesn't fail. We rejected this in the bake
    /// validator instead — see `zenpredict_bake::BakeError::UnknownOutputTransform`.
    pub fn from_byte(b: u8) -> Option<Self> {
        match b {
            0 => Some(Self::Identity),
            1 => Some(Self::Sigmoid),
            2 => Some(Self::SigmoidScaled),
            3 => Some(Self::Exp),
            4 => Some(Self::Round),
            _ => None,
        }
    }

    /// Apply the activation to a single value, given the spec's
    /// `transform_params`. Pure function.
    pub fn apply(self, x: f32, params: [f32; 2]) -> f32 {
        match self {
            Self::Identity => x,
            Self::Sigmoid => sigmoid(x),
            Self::SigmoidScaled => {
                let s = sigmoid(x);
                let lo = params[0];
                let hi = params[1];
                lo + s * (hi - lo)
            }
            Self::Exp => {
                // Clamp the input domain so `exp(x)` stays finite.
                // f32::MAX ≈ 3.4e38, log(MAX) ≈ 88.7; ±30 keeps the
                // dynamic range generous while ruling out infinities
                // from a model emitting unbounded scores.
                let clamped = x.clamp(-30.0, 30.0);
                #[cfg(feature = "std")]
                {
                    clamped.exp()
                }
                #[cfg(not(feature = "std"))]
                {
                    // no_std + alloc has no `f32::exp`. Mirror
                    // `argmin::ScoreTransform::Exp`'s degraded path.
                    let _ = clamped;
                    x
                }
            }
            Self::Round => round_no_std(x),
        }
    }
}

/// `round_ties_away_from_zero` for `f32`, available in both std and
/// no_std builds. `f32::round` is a `std`-only method, so we
/// reimplement it via the bit-level "add ±0.5 then truncate" pattern.
/// Matches IEEE-754 `roundToIntegralTiesAway` semantics, which is
/// what `f32::round` uses.
#[inline]
fn round_no_std(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    let bias = if x >= 0.0 { 0.5 } else { -0.5 };
    // (x + bias) as i64 truncates toward zero. Cast back to f32.
    // Saturates for values outside i64 range, but those round to
    // ±inf in f32 anyway and the user almost certainly didn't mean
    // to ship a picker that emits 9.2e18-ish values.
    ((x + bias) as i64) as f32
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    // Numerically stable sigmoid: avoids overflow in exp(-x) when x
    // is very negative. The inputs are usually small in practice but
    // codec features can include unscaled byte counts so we prefer
    // the safer branch here.
    if x >= 0.0 {
        #[cfg(feature = "std")]
        {
            let z = (-x).exp();
            1.0 / (1.0 + z)
        }
        #[cfg(not(feature = "std"))]
        {
            let _ = x;
            0.5
        }
    } else {
        #[cfg(feature = "std")]
        {
            let z = x.exp();
            z / (1.0 + z)
        }
        #[cfg(not(feature = "std"))]
        {
            let _ = x;
            0.5
        }
    }
}

/// Per-output post-processing record. 32 bytes on the wire —
/// `n_outputs * 32` bytes per bake when the section is present.
///
/// Layout:
///
/// ```text
/// 0..8     bounds: FeatureBound       (low: f32, high: f32)
/// 8..9     transform: u8              (OutputTransform variant)
/// 9..12    _pad: [u8; 3]              (zero)
/// 12..20   transform_params: [f32; 2] (zero unless SigmoidScaled)
/// 20..24   discrete_set_offset: u32   (in f32 units, into the discrete_sets section)
/// 24..28   discrete_set_len: u32      (count of f32 values; 0 = no snap)
/// 28..32   sentinel: f32              (NaN bit pattern = "no sentinel")
/// ```
///
/// The `pad` bytes MUST be zero in shipped bakes — current parsers
/// ignore them but a future revision may use one for an extension flag.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct OutputSpec {
    pub bounds: FeatureBound,
    pub transform: u8,
    pub _pad: [u8; 3],
    pub transform_params: [f32; 2],
    pub discrete_set_offset: u32,
    pub discrete_set_len: u32,
    pub sentinel: f32,
}

const _: () = assert!(core::mem::size_of::<OutputSpec>() == 32);

impl OutputSpec {
    /// "No-op" spec: identity transform, infinite bounds, no discrete
    /// snap, sentinel as NaN (i.e. no sentinel match).
    ///
    /// Used as the "absent OutputSpec section" default by
    /// [`crate::Predictor::predict_with_specs`] and as a starting
    /// point for builder-style construction.
    pub fn passthrough() -> Self {
        Self {
            bounds: FeatureBound::new(f32::NEG_INFINITY, f32::INFINITY),
            transform: OutputTransform::Identity as u8,
            _pad: [0; 3],
            transform_params: [0.0, 0.0],
            discrete_set_offset: 0,
            discrete_set_len: 0,
            sentinel: f32::NAN,
        }
    }

    /// Return the transform variant, falling back to Identity for
    /// unrecognized bytes (forward compatibility).
    pub fn transform(&self) -> OutputTransform {
        OutputTransform::from_byte(self.transform).unwrap_or(OutputTransform::Identity)
    }

    /// `true` iff the bake declared a sentinel value. Stored as
    /// `f32::NAN` for "no sentinel"; any finite or ±∞ value here
    /// counts as "yes, match this exact value to surface Default".
    pub fn has_sentinel(&self) -> bool {
        !self.sentinel.is_nan()
    }
}

/// One output of [`crate::Predictor::predict_with_specs`]. Either a
/// concrete value (after transform / clamp / snap / sparse override)
/// or "use codec default" — the latter when the rounded value matched
/// the spec's sentinel, or when a sparse override entry was `NaN`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum OutputValue {
    /// A concrete predicted value the caller should use. Always
    /// finite (NaN propagates to `Default`).
    Override(f32),
    /// "Use the codec's default for this parameter." Sentinel hit OR
    /// a `NaN` sparse override.
    Default,
}

impl OutputValue {
    /// Extract the value if this is an `Override`, else `None`.
    pub fn value(self) -> Option<f32> {
        match self {
            Self::Override(v) => Some(v),
            Self::Default => None,
        }
    }

    /// `true` if this is `Override`; `false` for `Default`.
    pub fn is_override(self) -> bool {
        matches!(self, Self::Override(_))
    }
}

/// Sparse hand-tune override. `(output_index, value)` — value is
/// `f32::NAN` when the maintainer wants to force the
/// "use codec default" outcome for that output.
///
/// 8 bytes on the wire (`u32 idx + f32 value`); the bake's
/// `sparse_overrides` section is `n_overrides * 8` bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct SparseOverride {
    pub idx: u32,
    pub value: f32,
}

const _: () = assert!(core::mem::size_of::<SparseOverride>() == 8);

impl SparseOverride {
    pub const fn new(idx: u32, value: f32) -> Self {
        Self { idx, value }
    }
}

/// Apply one [`OutputSpec`] to a raw forward-pass value. Pure
/// function; no allocation. Used internally by `predict_with_specs`
/// and exposed for callers that want to run the pipeline manually.
///
/// `discrete_pool` is the bake's flat discrete-sets section; the
/// spec's `(discrete_set_offset, discrete_set_len)` selects the
/// snap target slice.
///
/// # Order of operations
///
/// 1. `transform.apply(raw, transform_params)`
/// 2. clamp into `bounds`
/// 3. snap to nearest value in the spec's discrete-set slice, if
///    `discrete_set_len > 0`
/// 4. if `has_sentinel()` and the result matches `sentinel`,
///    return [`OutputValue::Default`]
///
/// Sparse overrides are NOT applied here — they're applied by
/// `predict_with_specs` after walking every output, so a sparse
/// entry can override a sentinel hit.
///
/// # Returns
///
/// [`OutputValue::Default`] only on sentinel match; otherwise
/// [`OutputValue::Override`] with the post-pipeline value.
pub fn apply_spec(spec: &OutputSpec, raw: f32, discrete_pool: &[f32]) -> OutputValue {
    // 1. activation
    let after_transform = spec.transform().apply(raw, spec.transform_params);

    // 2. clamp; treat NaN as "out of range" — clamp drops NaN to low
    //    by virtue of `max(low)` evaluating to `low` for NaN inputs.
    let clamped = if after_transform.is_nan() {
        spec.bounds.low
    } else {
        after_transform.max(spec.bounds.low).min(spec.bounds.high)
    };

    // 3. snap to discrete set, if any
    let snapped = if spec.discrete_set_len > 0 {
        let off = spec.discrete_set_offset as usize;
        let len = spec.discrete_set_len as usize;
        let end = off.saturating_add(len);
        if end <= discrete_pool.len() {
            snap_to_nearest(clamped, &discrete_pool[off..end])
        } else {
            clamped
        }
    } else {
        clamped
    };

    // 4. sentinel match — `==` against a finite/inf sentinel is exact;
    //    NaN sentinels are interpreted as "no sentinel" via
    //    `has_sentinel()`. Use `to_bits` for exact-pattern compare so
    //    -0.0 vs 0.0 doesn't collide; codec sentinels in practice are
    //    "-1.0" or "u32::MAX" which never collide with valid outputs.
    if spec.has_sentinel() && snapped.to_bits() == spec.sentinel.to_bits() {
        return OutputValue::Default;
    }
    OutputValue::Override(snapped)
}

#[inline]
fn snap_to_nearest(x: f32, set: &[f32]) -> f32 {
    debug_assert!(!set.is_empty(), "caller checked");
    let mut best = set[0];
    let mut best_dist = (x - best).abs();
    for &v in &set[1..] {
        let d = (x - v).abs();
        if d < best_dist {
            best_dist = d;
            best = v;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_spec_size_is_32_bytes() {
        assert_eq!(core::mem::size_of::<OutputSpec>(), 32);
    }

    #[test]
    fn sparse_override_size_is_8_bytes() {
        assert_eq!(core::mem::size_of::<SparseOverride>(), 8);
    }

    #[test]
    fn passthrough_spec_returns_input() {
        let spec = OutputSpec::passthrough();
        let r = apply_spec(&spec, 0.42, &[]);
        assert_eq!(r, OutputValue::Override(0.42));
    }

    #[test]
    fn identity_transform_is_identity() {
        assert_eq!(OutputTransform::Identity.apply(2.5, [0.0, 0.0]), 2.5);
    }

    #[cfg(feature = "std")]
    #[test]
    fn sigmoid_transform_at_zero_is_half() {
        let v = OutputTransform::Sigmoid.apply(0.0, [0.0, 0.0]);
        assert!((v - 0.5).abs() < 1e-6);
    }

    #[cfg(feature = "std")]
    #[test]
    fn sigmoid_scaled_remaps() {
        let v = OutputTransform::SigmoidScaled.apply(0.0, [0.0, 100.0]);
        assert!((v - 50.0).abs() < 1e-4);
    }

    #[cfg(feature = "std")]
    #[test]
    fn exp_clamp_caps_at_30() {
        let v = OutputTransform::Exp.apply(1000.0, [0.0, 0.0]);
        assert!(v.is_finite());
        let cap = OutputTransform::Exp.apply(30.0, [0.0, 0.0]);
        assert!((v - cap).abs() < 1e-3);
    }

    #[test]
    fn round_transform() {
        assert_eq!(OutputTransform::Round.apply(1.4, [0.0, 0.0]), 1.0);
        assert_eq!(OutputTransform::Round.apply(1.6, [0.0, 0.0]), 2.0);
        assert_eq!(OutputTransform::Round.apply(-0.6, [0.0, 0.0]), -1.0);
    }

    #[test]
    fn bounds_clamp_works() {
        let mut spec = OutputSpec::passthrough();
        spec.bounds = FeatureBound::new(0.0, 10.0);
        assert_eq!(apply_spec(&spec, -5.0, &[]), OutputValue::Override(0.0));
        assert_eq!(apply_spec(&spec, 5.0, &[]), OutputValue::Override(5.0));
        assert_eq!(apply_spec(&spec, 50.0, &[]), OutputValue::Override(10.0));
    }

    #[test]
    fn discrete_snap_picks_nearest() {
        let mut spec = OutputSpec::passthrough();
        spec.bounds = FeatureBound::new(0.0, 7.0);
        spec.discrete_set_offset = 0;
        spec.discrete_set_len = 8;
        let pool = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        assert_eq!(apply_spec(&spec, 3.4, &pool), OutputValue::Override(3.0));
        // Midpoint case — nearest-search ties to the first encountered.
        assert_eq!(apply_spec(&spec, 3.5, &pool), OutputValue::Override(3.0));
        assert_eq!(apply_spec(&spec, 3.51, &pool), OutputValue::Override(4.0));
    }

    #[test]
    fn sentinel_match_returns_default() {
        let mut spec = OutputSpec::passthrough();
        spec.bounds = FeatureBound::new(-1.0, 100.0);
        spec.sentinel = -1.0;
        assert_eq!(apply_spec(&spec, -1.0, &[]), OutputValue::Default);
        assert_eq!(apply_spec(&spec, 0.0, &[]), OutputValue::Override(0.0));
    }

    #[test]
    fn sentinel_after_snap_returns_default() {
        // Round to nearest int, snap to {-1, 0, 1, 2, …, 7}, sentinel
        // at -1 means "model voted for -1 (which a snap_set member),
        // so surface as Default".
        let mut spec = OutputSpec::passthrough();
        spec.transform = OutputTransform::Round as u8;
        spec.bounds = FeatureBound::new(-1.0, 7.0);
        spec.discrete_set_offset = 0;
        spec.discrete_set_len = 9;
        spec.sentinel = -1.0;
        let pool = [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        assert_eq!(apply_spec(&spec, -0.6, &pool), OutputValue::Default);
        assert_eq!(apply_spec(&spec, 3.7, &pool), OutputValue::Override(4.0));
    }

    #[test]
    fn nan_input_clamps_to_low() {
        let mut spec = OutputSpec::passthrough();
        spec.bounds = FeatureBound::new(0.0, 10.0);
        assert_eq!(apply_spec(&spec, f32::NAN, &[]), OutputValue::Override(0.0));
    }
}
