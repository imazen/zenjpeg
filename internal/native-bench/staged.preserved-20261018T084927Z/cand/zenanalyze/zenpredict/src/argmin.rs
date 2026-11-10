//! Generic decision math: masked argmin, top-K, confidence,
//! threshold mask, score transforms, and additive offsets.
//!
//! None of this is codec-specific — `argmin_masked` is just "argmin
//! over a slice with a boolean filter," `ArgminOffsets` is just
//! "uniform additive plus per-output additive in the post-transform
//! score space." The names that *were* picker-flavored in the
//! pre-zenpredict codebase (`CostAdjust`, `additive_bytes`,
//! `per_output_offset`, `reach_gate_mask`) generalize cleanly here.

use crate::error::PredictError;

/// Boolean filter over a score slice. `true` means "this index may
/// be picked." Bit-packing isn't worth the API complexity for the
/// 10s–100s of outputs real bakes have.
///
/// All argmin entry points require `mask.len() >= predictions.len()`
/// (or `>= n` for the `_with_scorer` family). Mismatched lengths
/// **panic** in debug AND release — short masks used to silently
/// deny high-index cells, which masked real bugs. Build an
/// all-allowed mask via [`AllowedMask::new`] with a `&[true; N]`
/// slice when you want to admit every cell of a known length.
#[derive(Clone, Copy, Debug)]
pub struct AllowedMask<'a> {
    pub allowed: &'a [bool],
}

impl<'a> AllowedMask<'a> {
    pub fn new(allowed: &'a [bool]) -> Self {
        Self { allowed }
    }

    /// `true` only when `idx < self.len()` AND the mask entry is
    /// `true`. Out-of-range indices return `false`; argmin entries
    /// reject mismatched lengths up front, so this only fires in
    /// hand-constructed call paths that bypass the assert.
    pub fn is_allowed(&self, idx: usize) -> bool {
        self.allowed.get(idx).copied().unwrap_or(false)
    }

    pub fn len(&self) -> usize {
        self.allowed.len()
    }

    pub fn is_empty(&self) -> bool {
        self.allowed.is_empty()
    }
}

/// Score-domain transform applied before offsets are added and
/// before argmin runs. Default is `Identity`.
///
/// Codecs whose model emits log-bytes (the typical zenjpeg shape)
/// pass `Exp` — argmin then runs in raw-byte space, which is what
/// matters when an offsets table mixes per-output overhead in
/// linear bytes.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub enum ScoreTransform {
    /// Argmin over the raw model outputs (no transform). Use when
    /// outputs are already in the argmin-target space — perceptual
    /// distances, raw counts, etc.
    #[default]
    Identity,
    /// Apply `exp` (clamped to [-30, 30] input range to keep
    /// numerics finite) before adding offsets and running argmin.
    /// Lets log-domain regressors mix with linear-domain offsets.
    ///
    /// Computed via `f32::exp` under `std` and `libm::expf` under
    /// no_std (`libm` is an unconditional dependency), so both build
    /// configurations apply a true `exp` and produce the same
    /// linear-space argmin — `ArgminOffsets` in linear-byte space mix
    /// correctly with the exponentiated scores either way.
    Exp,
}

impl ScoreTransform {
    /// Apply the transform to a single score. Used by the argmin
    /// implementation; not typically called by consumers.
    #[inline]
    pub fn apply(self, score: f32) -> f32 {
        match self {
            Self::Identity => score,
            Self::Exp => clamped_exp(score),
        }
    }
}

/// Caller-supplied additive cost adjustments applied in the
/// post-transform score space, before argmin.
///
/// `uniform` is added to every output's score (e.g. caller's ICC /
/// EXIF / XMP overhead in a log-bytes-via-Exp picker). It's the
/// same across all outputs so it doesn't change argmin on its own
/// — but combined with `per_output` (e.g. XYB intrinsic ICC vs
/// YCbCr-no-ICC) it can shift the pick.
#[derive(Clone, Copy, Debug, Default)]
pub struct ArgminOffsets<'a> {
    /// Added to every output's score after the transform.
    pub uniform: f32,
    /// Optional per-output additive — when present must equal the
    /// argmin's working slice length (full `n_outputs` for
    /// `argmin_masked`, the sub-range length for `*_in_range`).
    pub per_output: Option<&'a [f32]>,
}

impl<'a> ArgminOffsets<'a> {
    pub fn uniform(uniform: f32) -> Self {
        Self {
            uniform,
            per_output: None,
        }
    }

    /// Validate the per-output length against the argmin's working
    /// slice length. Returns Ok when no per-output table is set.
    pub(crate) fn validate(&self, expected: usize) -> Result<(), PredictError> {
        if let Some(po) = self.per_output
            && po.len() != expected
        {
            return Err(PredictError::OffsetsLenMismatch {
                expected,
                got: po.len(),
            });
        }
        Ok(())
    }
}

/// Argmin over `predictions`, restricted by `mask`. Score for index
/// `i` is `transform(predictions[i]) + uniform + per_output[i]`.
///
/// Returns `None` when no entry is allowed by the mask.
///
/// Argmin in `Identity` space without offsets reduces to a simple
/// `f32::min` linear scan — the offsets / transform branches are
/// only walked when the caller actually opts in.
///
/// # Contract
///
/// - **Mask length:** `mask.len() >= predictions.len()` is required.
///   Violating this panics in both debug and release; passing a
///   shorter mask used to silently deny high-index cells, which
///   masked real bugs.
/// - **NaN scores:** any prediction that yields NaN after transform
///   (or the closure in `_with_scorer`) is **silently skipped**.
///   `NaN < x` is false in IEEE-754, so NaN cells are never picked.
///   If every allowed cell scores NaN, the function returns `None`
///   indistinguishably from "no allowed cells" — callers needing
///   that distinction should pre-validate predictions.
/// - **Tie-breaking:** when two cells have identical post-offset
///   scores, the **lower index wins** (deterministic, first-encountered).
///
/// # Examples
///
/// ```
/// use zenpredict::{AllowedMask, ScoreTransform, argmin};
///
/// let scores = [3.0_f32, 1.0, 4.0, 1.5, 9.0];
/// let mask_data = [true, false, true, true, true];
/// let mask = AllowedMask::new(&mask_data);
/// let pick = argmin::argmin_masked(&scores, &mask, ScoreTransform::Identity, None);
/// assert_eq!(pick, Some(3)); // index 1's 1.0 is masked out, next-lowest is 3 (1.5)
/// ```
pub fn argmin_masked(
    predictions: &[f32],
    mask: &AllowedMask<'_>,
    transform: ScoreTransform,
    offsets: Option<&ArgminOffsets<'_>>,
) -> Option<usize> {
    assert!(
        mask.len() >= predictions.len(),
        "argmin_masked: mask.len() ({}) < predictions.len() ({}) — short masks used to silently deny high-index cells",
        mask.len(),
        predictions.len(),
    );
    let mut best_idx: Option<usize> = None;
    let mut best_score: f32 = f32::INFINITY;

    for (i, &raw) in predictions.iter().enumerate() {
        if !mask.is_allowed(i) {
            continue;
        }
        let score = score_at(raw, i, transform, offsets);
        if score < best_score {
            best_score = score;
            best_idx = Some(i);
        }
    }
    best_idx
}

/// Top-`K` lowest-scoring indices over `predictions` that the mask
/// permits, ascending (best first). Slots beyond the number of
/// allowed entries are `None`. `K` is generic to keep the call
/// site allocation-free; in practice `K = 2` is what codec rescue
/// logic wants (cached second-best), and `K = 3` is what the
/// "predict-top-K then encode-verify" picker path wants.
///
/// On the default surface, so a per-codec picker or [`zenpicker`] can
/// request the top-K candidate output indices for the proven
/// "predict-top-K then encode-verify" path **without** re-implementing
/// the masking / score-transform / NaN / tie-break contract in the
/// consumer — that contract lives here, once, alongside
/// [`argmin_masked`]. Same masking, score-transform, and range options
/// as [`argmin_masked`]. The closure-scorer variants `*_with_scorer`
/// and the confidence helpers stay behind `advanced`.
///
/// Same NaN, tie-breaking, and mask-length contract as
/// [`argmin_masked`].
///
/// [`zenpicker`]: https://docs.rs/zenpicker
///
/// # Examples
///
/// ```
/// use zenpredict::{AllowedMask, ScoreTransform, argmin};
///
/// let scores = [3.0_f32, 1.0, 4.0, 1.5, 9.0];
/// let mask_data = [true; 5];
/// let mask = AllowedMask::new(&mask_data);
/// let top = argmin::argmin_masked_top_k::<3>(&scores, &mask, ScoreTransform::Identity, None);
/// assert_eq!(top, [Some(1), Some(3), Some(0)]); // 1.0, 1.5, 3.0
/// ```
pub fn argmin_masked_top_k<const K: usize>(
    predictions: &[f32],
    mask: &AllowedMask<'_>,
    transform: ScoreTransform,
    offsets: Option<&ArgminOffsets<'_>>,
) -> [Option<usize>; K] {
    assert!(
        mask.len() >= predictions.len(),
        "argmin_masked_top_k: mask.len() ({}) < predictions.len() ({})",
        mask.len(),
        predictions.len(),
    );

    let mut top: [(f32, usize); K] = [(f32::INFINITY, usize::MAX); K];
    let mut count: usize = 0;

    let mut consider = |score: f32, idx: usize| {
        // Same NaN contract as `argmin_masked`: a NaN-scoring cell is silently skipped, never
        // occupying a top-K slot. Without this, `score.is_nan()` cells (for which every `<`
        // comparison is false) would slide into an empty slot below simply because the
        // shift-loop's `top[i - 1].0 > score` never fires to displace it.
        if score.is_nan() {
            return;
        }
        if count < K {
            let mut i = count;
            while i > 0 && top[i - 1].0 > score {
                top[i] = top[i - 1];
                i -= 1;
            }
            top[i] = (score, idx);
            count += 1;
        } else if K > 0 && score < top[K - 1].0 {
            let mut i = K - 1;
            while i > 0 && top[i - 1].0 > score {
                top[i] = top[i - 1];
                i -= 1;
            }
            top[i] = (score, idx);
        }
    };

    for (i, &raw) in predictions.iter().enumerate() {
        if !mask.is_allowed(i) {
            continue;
        }
        let score = score_at(raw, i, transform, offsets);
        consider(score, i);
    }

    let mut out: [Option<usize>; K] = [None; K];
    for slot in 0..count {
        out[slot] = Some(top[slot].1);
    }
    out
}

/// Argmin over `predictions[range.0..range.1]`, masked by `mask`
/// (whose `len()` must equal `range.1 - range.0`). Returned index
/// is *within the sub-range* (0..(range.1 - range.0)).
pub fn argmin_masked_in_range(
    predictions: &[f32],
    range: (usize, usize),
    mask: &AllowedMask<'_>,
    transform: ScoreTransform,
    offsets: Option<&ArgminOffsets<'_>>,
) -> Option<usize> {
    let (start, end) = range;
    let slice = predictions.get(start..end)?;
    argmin_masked(slice, mask, transform, offsets)
}

/// Top-`K` over a sub-range `predictions[range.0..range.1]`, masked
/// by `mask` (whose `len()` must be `>= range.1 - range.0`).
/// Returned indices are *within the sub-range*. Same support
/// guarantee as [`argmin_masked_top_k`].
///
/// Returns all-`None` when the range is out of bounds (`end >
/// predictions.len()` or `start > end`) — the in-range argmin
/// scalar form returns `None` on the same condition.
pub fn argmin_masked_top_k_in_range<const K: usize>(
    predictions: &[f32],
    range: (usize, usize),
    mask: &AllowedMask<'_>,
    transform: ScoreTransform,
    offsets: Option<&ArgminOffsets<'_>>,
) -> [Option<usize>; K] {
    let (start, end) = range;
    if end > predictions.len() || start > end {
        return [None; K];
    }
    argmin_masked_top_k::<K>(&predictions[start..end], mask, transform, offsets)
}

/// Argmin under a caller-supplied score function. `scorer(i)` is
/// invoked once per `i` where `mask.is_allowed(i)`; returns the
/// scalar score to compare. Smallest score wins. Returns `None`
/// when no entry is allowed.
///
/// The closure-based form is for cases the
/// `transform + offsets` shape can't express cleanly: RD-vs-time
/// (`bytes + μ·ms` reading from two hybrid heads), multi-metric
/// pickers (selecting one metric's sub-range with a runtime index),
/// or codec-specific saturating clamps.
///
/// `n` is the number of candidate indices; the caller binds the
/// model output (or any other source) inside the closure.
///
/// # Contract
///
/// - `mask.len() >= n` is required; violation panics (debug + release).
/// - The closure is invoked **once per allowed index**, in ascending
///   index order. Side effects fire that many times.
/// - **NaN scores are silently skipped** (NaN never compares less
///   than any finite value). If every allowed cell scores NaN, `None`
///   is returned indistinguishably from "no allowed cells."
/// - **Tie-breaking:** lowest index wins (deterministic).
/// - Closure panics propagate; no unwinding protection.
///
/// # Examples
///
/// RD-vs-time argmin with hybrid-heads outputs (#56):
///
/// ```
/// use zenpredict::{AllowedMask, argmin};
///
/// // bytes_log[0..3] then time[3..6] (per-cell, hybrid-heads layout).
/// let out = [10.0_f32, 11.0, 12.0,    // bytes_log per cell
///            5.0, 8.0, 12.0];         // ms per cell
/// let mask_data = [true, true, true];
/// let mask = AllowedMask::new(&mask_data);
/// let mu = 100.0_f32;
/// let pick = argmin::argmin_masked_with_scorer(3, &mask, |i| {
///     let bytes = out[i].exp();
///     let ms    = out[3 + i];
///     bytes + mu * ms
/// });
/// // Cell 0: e^10 ≈ 22 026 + 500 = 22 526
/// // Cell 1: e^11 ≈ 59 874 + 800 = 60 674
/// // Cell 2: e^12 ≈ 162 754 + 1200 = 163 954
/// assert_eq!(pick, Some(0));
/// ```
#[cfg(feature = "advanced")]
pub fn argmin_masked_with_scorer<F>(n: usize, mask: &AllowedMask<'_>, scorer: F) -> Option<usize>
where
    F: Fn(usize) -> f32,
{
    assert!(
        mask.len() >= n,
        "argmin_masked_with_scorer: mask.len() ({}) < n ({})",
        mask.len(),
        n,
    );
    let mut best_idx: Option<usize> = None;
    let mut best_score: f32 = f32::INFINITY;
    for i in 0..n {
        if !mask.is_allowed(i) {
            continue;
        }
        let score = scorer(i);
        if score < best_score {
            best_score = score;
            best_idx = Some(i);
        }
    }
    best_idx
}

/// Top-`K` over a caller-supplied score function. Same shape as
/// [`argmin_masked_top_k`] but with a closure replacing the
/// `transform + offsets` score-derivation. Slots beyond the number
/// of mask-allowed entries are `None`.
///
/// Same NaN, tie-breaking, mask-length, and closure-panic contract
/// as [`argmin_masked_with_scorer`].
#[cfg(feature = "advanced")]
pub fn argmin_masked_top_k_with_scorer<const K: usize, F>(
    n: usize,
    mask: &AllowedMask<'_>,
    scorer: F,
) -> [Option<usize>; K]
where
    F: Fn(usize) -> f32,
{
    assert!(
        mask.len() >= n,
        "argmin_masked_top_k_with_scorer: mask.len() ({}) < n ({})",
        mask.len(),
        n,
    );
    let mut top: [(f32, usize); K] = [(f32::INFINITY, usize::MAX); K];
    let mut count: usize = 0;

    for i in 0..n {
        if !mask.is_allowed(i) {
            continue;
        }
        let score = scorer(i);
        if count < K {
            let mut j = count;
            while j > 0 && top[j - 1].0 > score {
                top[j] = top[j - 1];
                j -= 1;
            }
            top[j] = (score, i);
            count += 1;
        } else if K > 0 && score < top[K - 1].0 {
            let mut j = K - 1;
            while j > 0 && top[j - 1].0 > score {
                top[j] = top[j - 1];
                j -= 1;
            }
            top[j] = (score, i);
        }
    }

    let mut out: [Option<usize>; K] = [None; K];
    for slot in 0..count {
        out[slot] = Some(top[slot].1);
    }
    out
}

/// Pick the argmin and report a confidence signal: the score gap
/// to the second-best mask-allowed entry. Returns `(best_idx, gap)`
/// where `gap` is in the same score units argmin used (post-
/// transform, post-offsets).
///
/// `gap = +∞` when only one mask entry is allowed; `0.0` if every
/// score ties at the top. Returns `None` when the mask permits
/// zero entries.
#[cfg(feature = "advanced")]
pub fn pick_with_confidence(
    predictions: &[f32],
    mask: &AllowedMask<'_>,
    transform: ScoreTransform,
    offsets: Option<&ArgminOffsets<'_>>,
) -> Option<(usize, f32)> {
    let top = argmin_masked_top_k::<2>(predictions, mask, transform, offsets);
    pick_confidence_from_top_k(predictions, transform, offsets, top)
}

#[cfg(feature = "advanced")]
pub fn pick_with_confidence_in_range(
    predictions: &[f32],
    range: (usize, usize),
    mask: &AllowedMask<'_>,
    transform: ScoreTransform,
    offsets: Option<&ArgminOffsets<'_>>,
) -> Option<(usize, f32)> {
    let (start, end) = range;
    let slice = predictions.get(start..end)?;
    let top = argmin_masked_top_k::<2>(slice, mask, transform, offsets);
    pick_confidence_from_top_k(slice, transform, offsets, top)
}

#[cfg(feature = "advanced")]
pub(crate) fn pick_confidence_from_top_k(
    predictions: &[f32],
    transform: ScoreTransform,
    offsets: Option<&ArgminOffsets<'_>>,
    top: [Option<usize>; 2],
) -> Option<(usize, f32)> {
    let best = top[0]?;
    let Some(second) = top[1] else {
        return Some((best, f32::INFINITY));
    };
    let s_best = score_at(predictions[best], best, transform, offsets);
    let s_second = score_at(predictions[second], second, transform, offsets);
    Some((best, (s_second - s_best).max(0.0)))
}

/// Fill `out[i] = values[i].is_finite() && values[i] >= floor` — admit
/// only cells whose per-cell attribute meets a **runtime floor**. The
/// canonical use is a **target-quality** constraint: keep cells whose
/// predicted quality (ssim2 / zensim / reach rate) is at least the
/// caller's target. The companion [`mask_at_most`] expresses a ceiling
/// (e.g. a perf / encode-cost limit).
///
/// Both are generic over a caller-supplied per-cell `f32` attribute —
/// the codec owns where the values come from (a model output head, a
/// bake table, or its own config grammar). AND the result into the
/// constraint mask before calling [`argmin_masked`] /
/// [`argmin_masked_top_k`]; combine several (quality floor AND perf
/// ceiling AND …) by ANDing their masks together.
///
/// `NaN` fails the constraint (`is_finite()` is false) — an unknown /
/// missing-data attribute is never admitted. `out.len()` must equal
/// `values.len()` (mismatch panics, same discipline as [`AllowedMask`]).
///
/// # Examples
///
/// ```
/// use zenpredict::argmin::mask_at_least;
///
/// // Per-cell predicted quality; admit cells reaching the target.
/// let quality = [0.99, 0.5, f32::NAN, 0.95];
/// let mut gate = [false; 4];
/// mask_at_least(&quality, 0.95, &mut gate); // quality >= 0.95
/// assert_eq!(gate, [true, false, false, true]);
/// ```
pub fn mask_at_least(values: &[f32], floor: f32, out: &mut [bool]) {
    assert_eq!(
        values.len(),
        out.len(),
        "mask_at_least: values.len() ({}) != out.len() ({})",
        values.len(),
        out.len(),
    );
    for (slot, &v) in out.iter_mut().zip(values.iter()) {
        // NaN fails (is_finite() is false): an unknown attribute is
        // never admitted under a constraint.
        *slot = v.is_finite() && v >= floor;
    }
}

/// Fill `out[i] = values[i].is_finite() && values[i] <= limit` — admit
/// only cells whose per-cell attribute is within a **runtime ceiling**.
/// The canonical use is a **perf / compute limit**: keep only configs
/// whose encode cost (or any cost attribute) is at most the caller's
/// budget, dropping everything too expensive. Mirror of
/// [`mask_at_least`]; see it for the shared contract.
///
/// `NaN` fails the constraint — a cell with unknown cost is never
/// admitted under a limit. `out.len()` must equal `values.len()`.
///
/// # Examples
///
/// ```
/// use zenpredict::argmin::mask_at_most;
///
/// // Per-cell encode cost (e.g. ms, or a relative effort score).
/// let cost = [1.0, 8.0, f32::NAN, 3.0];
/// let mut gate = [false; 4];
/// mask_at_most(&cost, 3.0, &mut gate); // cost <= 3.0
/// assert_eq!(gate, [true, false, false, true]);
/// ```
pub fn mask_at_most(values: &[f32], limit: f32, out: &mut [bool]) {
    assert_eq!(
        values.len(),
        out.len(),
        "mask_at_most: values.len() ({}) != out.len() ({})",
        values.len(),
        out.len(),
    );
    for (slot, &v) in out.iter_mut().zip(values.iter()) {
        *slot = v.is_finite() && v <= limit;
    }
}

#[inline]
fn score_at(
    raw: f32,
    idx: usize,
    transform: ScoreTransform,
    offsets: Option<&ArgminOffsets<'_>>,
) -> f32 {
    let mut s = transform.apply(raw);
    if let Some(o) = offsets {
        s += o.uniform;
        if let Some(po) = o.per_output
            && let Some(&v) = po.get(idx)
        {
            s += v;
        }
    }
    s
}

fn clamped_exp(x: f32) -> f32 {
    let x = x.clamp(-30.0, 30.0);
    #[cfg(feature = "std")]
    {
        x.exp()
    }
    #[cfg(not(feature = "std"))]
    {
        // `core` has no `f32::exp`, but `libm` is an unconditional
        // dependency (it backs `ln` / `ln_1p` in `feature_transform`),
        // so no_std computes a true `exp` — same linear-space argmin as
        // the std path, no silent degradation.
        libm::expf(x)
    }
}
