//! Per-feature numeric bounds + out-of-distribution detection.

use bytemuck::{Pod, Zeroable};

/// Inclusive low/high bounds for one input feature.
///
/// Bakes ship a `[FeatureBound; n_inputs]` table (typically the
/// `p01` / `p99` quantiles of each feature on the training corpus)
/// and consumers call [`first_out_of_distribution`] before scoring
/// to detect inputs the model wasn't trained on. Codecs typically
/// fall through to a known-good rescue strategy on a hit; scoring
/// crates may surface the OOD index as a metric warning.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct FeatureBound {
    pub low: f32,
    pub high: f32,
}

impl FeatureBound {
    pub const fn new(low: f32, high: f32) -> Self {
        Self { low, high }
    }

    /// `true` when `value` is finite and inside `[low, high]`.
    #[inline]
    pub fn contains(&self, value: f32) -> bool {
        value.is_finite() && value >= self.low && value <= self.high
    }
}

/// Index of the first feature outside its bounds, or `None` when
/// the entire vector is in-distribution.
///
/// Treats NaN / Inf as out-of-distribution — those inputs can't
/// be modelled and should always force fallback. `bounds.len()`
/// must equal `features.len()`; mismatched lengths panic in debug
/// builds and short-circuit at the shorter length in release.
///
/// # Examples
///
/// ```
/// use zenpredict::{FeatureBound, first_out_of_distribution};
///
/// let bounds = [
///     FeatureBound::new(0.0, 1.0),
///     FeatureBound::new(-1.0, 1.0),
/// ];
/// assert_eq!(first_out_of_distribution(&[0.5, 0.0], &bounds), None);
/// assert_eq!(first_out_of_distribution(&[2.0, 0.0], &bounds), Some(0));
/// assert_eq!(first_out_of_distribution(&[0.5, f32::NAN], &bounds), Some(1));
/// ```
pub fn first_out_of_distribution(features: &[f32], bounds: &[FeatureBound]) -> Option<usize> {
    debug_assert_eq!(
        features.len(),
        bounds.len(),
        "feature vector length must match bounds length",
    );
    for (i, (f, b)) in features.iter().zip(bounds.iter()).enumerate() {
        if !b.contains(*f) {
            return Some(i);
        }
    }
    None
}

/// Bound on a single output prediction. Same wire shape as
/// [`FeatureBound`] (two `f32`s); separate type alias so the API
/// distinguishes "input feature went out of training range" from
/// "model emitted a prediction outside the training-distribution
/// envelope" at the call site.
///
/// Built by the trainer over the held-out validation set: per
/// output dim, take the `(p01, p99)` of predicted values. Codec
/// checks model output against this AFTER `predict()` and BEFORE
/// argmin — any out-of-range output indicates the MLP is
/// extrapolating past its training envelope; route to
/// `RescueStrategy::KnownGoodFallback`.
#[cfg(feature = "advanced")]
pub type OutputBound = FeatureBound;

/// Index of the first model output outside its training-distribution
/// envelope, or `None` when every prediction is in-range.
///
/// Mirrors [`first_out_of_distribution`] but operates on the
/// `Predictor::predict` output rather than the input feature
/// vector. Treats NaN / Inf as out-of-distribution.
///
/// # Examples
///
/// ```
/// use zenpredict::{OutputBound, output_first_out_of_distribution};
///
/// let bounds = [
///     OutputBound::new(4.0, 12.0),    // bytes_log[0] expected range
///     OutputBound::new(50.0, 95.0),   // predicted_zensim[0] expected range
/// ];
/// assert_eq!(output_first_out_of_distribution(&[8.5, 80.0], &bounds), None);
/// // Picker hallucinated zq=200 (impossible). Codec routes to fallback.
/// assert_eq!(output_first_out_of_distribution(&[8.5, 200.0], &bounds), Some(1));
/// // Picker emitted NaN (numeric loss). Out of distribution.
/// assert_eq!(output_first_out_of_distribution(&[f32::NAN, 80.0], &bounds), Some(0));
/// ```
#[cfg(feature = "advanced")]
pub fn output_first_out_of_distribution(
    predictions: &[f32],
    bounds: &[OutputBound],
) -> Option<usize> {
    debug_assert_eq!(
        predictions.len(),
        bounds.len(),
        "predictions length must match output_bounds length",
    );
    for (i, (p, b)) in predictions.iter().zip(bounds.iter()).enumerate() {
        if !b.contains(*p) {
            return Some(i);
        }
    }
    None
}
