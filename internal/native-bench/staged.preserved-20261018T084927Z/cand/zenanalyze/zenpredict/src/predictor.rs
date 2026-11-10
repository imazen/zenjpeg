//! Scratch-owning forward-pass wrapper.
//!
//! Owns a [`Model`] plus the scratch buffers that [`forward`] needs,
//! exposes one allocation-free `predict` entry. Both downstream
//! shapes — codec picker, perceptual scorer — wrap a `Predictor`
//! and add their own typed front door.

#[cfg(feature = "advanced")]
use crate::argmin::pick_confidence_from_top_k;
use crate::argmin::{self, AllowedMask, ArgminOffsets, ScoreTransform};
use crate::error::PredictError;
use crate::feature_transform::{
    FeatureTransform, apply_feature_pipeline_expanding, apply_feature_transforms,
};
use crate::inference::forward;
use crate::model::Model;
#[cfg(feature = "advanced")]
use crate::output_spec::{OutputValue, apply_spec};

/// Scratch-owning forward-pass wrapper. Allocations happen in
/// [`Predictor::new`]; `predict` and the `argmin_*` family are
/// allocation-free hot paths.
///
/// `Predictor` **borrows** the `Model` rather than owning it. This
/// lets a single `Model` live in `static OnceLock<Model>` and be
/// shared by per-thread `Predictor` instances without a `Mutex` —
/// `Model` is `Sync + Send` (read-only owned bytes), while
/// `Predictor` owns the mutable scratch buffers per-thread.
///
/// Single lifetime `'a`: the lifetime of the model borrow. For
/// `static OnceLock<Model>` it's `'static`.
pub struct Predictor<'a> {
    model: &'a Model,
    scratch_a: alloc::vec::Vec<f32>,
    scratch_b: alloc::vec::Vec<f32>,
    output: alloc::vec::Vec<f32>,
    /// Post-processed output buffer for [`Self::predict_with_specs`].
    /// Sized to `n_outputs` at construction; reused across calls.
    /// Gated behind `advanced` — `predict_with_specs` lives under
    /// the same feature, so the buffer only exists when callable.
    #[cfg(feature = "advanced")]
    spec_output: alloc::vec::Vec<OutputValue>,
    /// Scratch buffer for the `_transformed` family. Sized to
    /// `n_inputs` when the bake declared scalar-only
    /// `feature_transforms`, to `model.expanded_input_dim()` when an
    /// expander variant is present (resized lazily on first call),
    /// and to zero otherwise (the no-transform path forwards the
    /// input slice without copying). Reused across calls.
    feat_scratch: alloc::vec::Vec<f32>,
    /// Scratch view of per-feature param slices, rebuilt on each
    /// expander-path call. Stored on `Predictor` to avoid an
    /// allocation per `predict_transformed` invocation; the underlying
    /// storage lives in `Model::feature_transform_params()`.
    feat_param_refs: alloc::vec::Vec<&'a [f32]>,
}

impl<'a> Predictor<'a> {
    /// Construct a predictor borrowing `model`. The model lives at
    /// the caller's choosing — typically inside a
    /// `static OnceLock<Model>` so the same parsed bake is shared
    /// across many `Predictor` instances (one per thread for the
    /// lock-free path, or one per call site for ad-hoc use).
    ///
    /// Example (static singleton, lock-free per-thread):
    /// ```ignore
    /// use std::cell::RefCell;
    /// use std::sync::OnceLock;
    /// use zenpredict::{Model, Predictor};
    ///
    /// static BAKE: &[u8] = include_bytes!("picker.bin");
    /// static MODEL: OnceLock<Model> = OnceLock::new();
    ///
    /// thread_local! {
    ///     static PREDICTOR: RefCell<Predictor<'static>> =
    ///         RefCell::new(Predictor::new(MODEL.get_or_init(
    ///             || Model::from_bytes(BAKE).unwrap()
    ///         )));
    /// }
    /// ```
    pub fn new(model: &'a Model) -> Self {
        let need = model.scratch_len();
        let n_out = model.n_outputs();
        let n_in = model.n_inputs();
        let feat_scratch_len = if model.has_expander_feature_transforms() {
            model.expanded_input_dim()
        } else if model.feature_transforms().is_some() {
            n_in
        } else {
            0
        };
        let feat_param_refs_cap = if model.has_expander_feature_transforms() {
            n_in
        } else {
            0
        };
        Self {
            model,
            scratch_a: alloc::vec![0.0; need],
            scratch_b: alloc::vec![0.0; need],
            output: alloc::vec![0.0; n_out],
            #[cfg(feature = "advanced")]
            spec_output: alloc::vec![OutputValue::Default; n_out],
            feat_scratch: alloc::vec![0.0; feat_scratch_len],
            feat_param_refs: alloc::vec::Vec::with_capacity(feat_param_refs_cap),
        }
    }

    pub fn n_inputs(&self) -> usize {
        self.model.n_inputs()
    }

    pub fn n_outputs(&self) -> usize {
        self.model.n_outputs()
    }

    pub fn schema_hash(&self) -> u64 {
        self.model.schema_hash()
    }

    pub fn model(&self) -> &Model {
        self.model
    }

    /// Run forward pass. Returns the output vector — for log-bytes
    /// regressors, this is `n_outputs` log-bytes-per-config; for
    /// zensim's V0_4 scorer, this is `[distance]`.
    ///
    /// Scratch buffers (`scratch_a`, `scratch_b`) are reused across
    /// calls without zeroing — every layer's matmul writes biases
    /// into the destination buffer **before** accumulating, so stale
    /// data from a prior call never leaks into the result. Calling
    /// `predict` twice with the same `features` is deterministic.
    pub fn predict(&mut self, features: &[f32]) -> Result<&[f32], PredictError> {
        forward(
            self.model,
            features,
            &mut self.scratch_a,
            &mut self.scratch_b,
            &mut self.output,
        )?;
        Ok(&self.output)
    }

    /// Run forward pass and apply the bake's per-output specs and
    /// sparse overrides. Returns one [`OutputValue`] per model
    /// output: either an `Override(f32)` carrying the post-processed
    /// value, or `Default` indicating "use the codec's built-in
    /// default for this parameter".
    ///
    /// Pipeline per output `i`:
    ///
    /// 1. forward pass yields `raw[i]`
    /// 2. `output_specs[i]` (if present) applies activation, clamp,
    ///    snap-to-discrete, then sentinel match (see
    ///    [`crate::output_spec::apply_spec`])
    /// 3. sparse overrides: any `(i, value)` entry replaces the
    ///    pipeline result with `Override(value)`, or with `Default`
    ///    when `value.is_nan()`
    ///
    /// When the bake has no `output_specs` section (raw-passthrough
    /// bake), every output is returned as `Override(raw[i])` —
    /// equivalent to [`Self::predict`] wrapped in `Override`. Sparse
    /// overrides still apply.
    ///
    /// # Errors
    ///
    /// Same as [`Self::predict`]: feature-length mismatch, etc. The
    /// spec pipeline itself is infallible — invalid (offset, len)
    /// pairs were rejected at load time.
    #[cfg(feature = "advanced")]
    pub fn predict_with_specs(&mut self, features: &[f32]) -> Result<&[OutputValue], PredictError> {
        forward(
            self.model,
            features,
            &mut self.scratch_a,
            &mut self.scratch_b,
            &mut self.output,
        )?;
        let specs = self.model.output_specs();
        let pool = self.model.discrete_sets();
        // Two paths: with-specs (apply per-output pipeline) vs.
        // without (raw passthrough → Override). We keep the loop
        // tight in either case.
        if specs.is_empty() {
            for (slot, &raw) in self.spec_output.iter_mut().zip(self.output.iter()) {
                *slot = OutputValue::Override(raw);
            }
        } else {
            for (i, slot) in self.spec_output.iter_mut().enumerate() {
                *slot = apply_spec(&specs[i], self.output[i], pool);
            }
        }
        // Sparse overrides land last so a maintainer can force a
        // specific output even when the spec pipeline produced
        // something different.
        for entry in self.model.sparse_overrides() {
            let i = entry.idx as usize;
            // Bounds were validated at load time but defend in
            // depth.
            if i < self.spec_output.len() {
                self.spec_output[i] = if entry.value.is_nan() {
                    OutputValue::Default
                } else {
                    OutputValue::Override(entry.value)
                };
            }
        }
        Ok(&self.spec_output)
    }

    /// Like [`Self::predict`], but applies the bake's
    /// `zentrain.feature_transforms` (issue #52) to each feature
    /// element before standardize + forward pass. **This is the
    /// recommended entry point for codec runtimes** — calling
    /// [`Self::predict`] on a bake whose `FEATURE_TRANSFORMS`
    /// declares any non-identity transform produces silently-wrong
    /// predictions because the trainer's scaler stats already
    /// reflect the post-transform distribution.
    ///
    /// When the bake omits `feature_transforms` (or every entry is
    /// `Identity`), this method forwards the caller's slice without
    /// copying — same allocation profile as [`Self::predict`].
    pub fn predict_transformed(&mut self, features: &[f32]) -> Result<&[f32], PredictError> {
        // Apply transforms inline so the forward call can borrow
        // `scratch_a`/`scratch_b`/`output` mutably without aliasing
        // through a helper that also borrows `self`.
        //
        // Three cases, in priority order:
        //
        // 1. No transforms metadata ⇒ forward `features` unchanged.
        //
        // 2. All transforms are scalar (Identity / Log / Log1p /
        //    Signed* / Clip* / Winsor* / QuantileBins) ⇒ apply
        //    per-feature into `feat_scratch[..features.len()]`,
        //    forward the (raw-length) scratch. Same allocation
        //    profile as `predict`.
        //
        // 3. At least one transform is an expander
        //    ([`FeatureTransform::Sinusoidal`]) ⇒ allocate
        //    `feat_scratch` to `model.expanded_input_dim()` slots,
        //    apply via `apply_feature_pipeline_expanding`, forward
        //    the expanded scratch. The model's first layer
        //    `in_dim` must equal `expanded_input_dim()` — this
        //    is enforced at bake time by the composer, so the
        //    runtime can trust it.
        let Some(transforms) = self.model.feature_transforms() else {
            forward(
                self.model,
                features,
                &mut self.scratch_a,
                &mut self.scratch_b,
                &mut self.output,
            )?;
            return Ok(&self.output);
        };
        if features.len() != transforms.len() {
            return Err(PredictError::FeatureLenMismatch {
                expected: transforms.len(),
                got: features.len(),
            });
        }

        let has_expander = self.model.has_expander_feature_transforms();
        if has_expander {
            // Expanding path. Per-feature params must be present —
            // Sinusoidal requires its frequency list.
            let params = self
                .model
                .feature_transform_params()
                .ok_or(PredictError::UnexpectedExpanderInScalarPipeline { feature_index: 0 })?;
            let expanded_dim = self.model.expanded_input_dim();
            if self.feat_scratch.len() < expanded_dim {
                self.feat_scratch.resize(expanded_dim, 0.0);
            }
            let dst = &mut self.feat_scratch[..expanded_dim];
            // Build a `&[&[f32]]` view over the owned per-feature params.
            // `feat_param_refs` is a stable scratch field on Predictor;
            // re-fill it here rather than allocating a fresh Vec.
            self.feat_param_refs.clear();
            self.feat_param_refs
                .extend(params.iter().map(|v| v.as_slice()));
            apply_feature_pipeline_expanding(transforms, &self.feat_param_refs, features, dst)?;
            forward(
                self.model,
                dst,
                &mut self.scratch_a,
                &mut self.scratch_b,
                &mut self.output,
            )?;
            return Ok(&self.output);
        }

        // Scalar path.
        if self.feat_scratch.len() < features.len() {
            self.feat_scratch.resize(features.len(), 0.0);
        }
        let dst = &mut self.feat_scratch[..features.len()];
        match self.model.feature_transform_params() {
            Some(params) => {
                debug_assert_eq!(params.len(), transforms.len());
                for i in 0..features.len() {
                    // Scalar contract — `apply_with_params` panics for
                    // expander variants, but `has_expander` already
                    // proved there aren't any in this branch.
                    dst[i] = transforms[i].apply_with_params(features[i], &params[i]);
                }
            }
            None => apply_feature_transforms(transforms, features, dst)?,
        }
        forward(
            self.model,
            dst,
            &mut self.scratch_a,
            &mut self.scratch_b,
            &mut self.output,
        )?;
        Ok(&self.output)
    }

    /// Like [`Self::predict_with_specs`], but applies the bake's
    /// `zentrain.feature_transforms` (issue #52) to each feature
    /// element before forward pass. See
    /// [`Self::predict_transformed`] for the rationale on which
    /// entry point codec runtimes should call.
    #[cfg(feature = "advanced")]
    pub fn predict_with_specs_transformed(
        &mut self,
        features: &[f32],
    ) -> Result<&[OutputValue], PredictError> {
        // Same dispatch as `predict_transformed`: scalar bakes use
        // the fast path; expander bakes route through
        // `apply_feature_pipeline_expanding` with an expanded scratch
        // buffer. The output specs / discrete sets / sparse overrides
        // post-processing runs once at the bottom regardless of
        // which pipeline shape filled `self.output`.
        if let Some(transforms) = self.model.feature_transforms() {
            if features.len() != transforms.len() {
                return Err(PredictError::FeatureLenMismatch {
                    expected: transforms.len(),
                    got: features.len(),
                });
            }
            if self.model.has_expander_feature_transforms() {
                let params = self
                    .model
                    .feature_transform_params()
                    .ok_or(PredictError::UnexpectedExpanderInScalarPipeline { feature_index: 0 })?;
                let expanded_dim = self.model.expanded_input_dim();
                if self.feat_scratch.len() < expanded_dim {
                    self.feat_scratch.resize(expanded_dim, 0.0);
                }
                let dst = &mut self.feat_scratch[..expanded_dim];
                self.feat_param_refs.clear();
                self.feat_param_refs
                    .extend(params.iter().map(|v| v.as_slice()));
                apply_feature_pipeline_expanding(transforms, &self.feat_param_refs, features, dst)?;
                forward(
                    self.model,
                    dst,
                    &mut self.scratch_a,
                    &mut self.scratch_b,
                    &mut self.output,
                )?;
            } else {
                if self.feat_scratch.len() < features.len() {
                    self.feat_scratch.resize(features.len(), 0.0);
                }
                let dst = &mut self.feat_scratch[..features.len()];
                match self.model.feature_transform_params() {
                    Some(params) => {
                        debug_assert_eq!(params.len(), transforms.len());
                        for i in 0..features.len() {
                            dst[i] = transforms[i].apply_with_params(features[i], &params[i]);
                        }
                    }
                    None => apply_feature_transforms(transforms, features, dst)?,
                }
                forward(
                    self.model,
                    dst,
                    &mut self.scratch_a,
                    &mut self.scratch_b,
                    &mut self.output,
                )?;
            }
        } else {
            forward(
                self.model,
                features,
                &mut self.scratch_a,
                &mut self.scratch_b,
                &mut self.output,
            )?;
        }
        let specs = self.model.output_specs();
        let pool = self.model.discrete_sets();
        if specs.is_empty() {
            for (slot, &raw) in self.spec_output.iter_mut().zip(self.output.iter()) {
                *slot = OutputValue::Override(raw);
            }
        } else {
            for (i, slot) in self.spec_output.iter_mut().enumerate() {
                *slot = apply_spec(&specs[i], self.output[i], pool);
            }
        }
        for entry in self.model.sparse_overrides() {
            let i = entry.idx as usize;
            if i < self.spec_output.len() {
                self.spec_output[i] = if entry.value.is_nan() {
                    OutputValue::Default
                } else {
                    OutputValue::Override(entry.value)
                };
            }
        }
        Ok(&self.spec_output)
    }

    /// Per-feature transforms declared by the bake. Convenience
    /// re-export of [`Model::feature_transforms`] so callers that
    /// already hold a `Predictor` don't need to thread the model
    /// through.
    ///
    /// [`Model::feature_transforms`]: crate::Model::feature_transforms
    pub fn feature_transforms(&self) -> Option<&[FeatureTransform]> {
        self.model.feature_transforms()
    }

    /// Pick the argmin output index over the masked set.
    pub fn argmin_masked(
        &mut self,
        features: &[f32],
        mask: &AllowedMask<'_>,
        transform: ScoreTransform,
        offsets: Option<&ArgminOffsets<'_>>,
    ) -> Result<Option<usize>, PredictError> {
        self.predict(features)?;
        if let Some(o) = offsets {
            o.validate(self.output.len())?;
        }
        Ok(argmin::argmin_masked(
            &self.output,
            mask,
            transform,
            offsets,
        ))
    }

    /// Argmin over a sub-range of the output vector. Hybrid-heads
    /// pickers lay outputs out as
    /// `[bytes[0..n_cells], scalar1[0..n_cells], scalar2[0..n_cells], …]`;
    /// the argmin of interest runs over `bytes` only.
    pub fn argmin_masked_in_range(
        &mut self,
        features: &[f32],
        range: (usize, usize),
        mask: &AllowedMask<'_>,
        transform: ScoreTransform,
        offsets: Option<&ArgminOffsets<'_>>,
    ) -> Result<Option<usize>, PredictError> {
        self.predict(features)?;
        let (start, end) = range;
        if end > self.output.len() || start > end {
            return Err(PredictError::OutputDimMismatch {
                expected: self.output.len(),
                got: end,
            });
        }
        if let Some(o) = offsets {
            o.validate(end - start)?;
        }
        Ok(argmin::argmin_masked(
            &self.output[start..end],
            mask,
            transform,
            offsets,
        ))
    }

    /// Run the forward pass, then return the top-`K` lowest-scoring
    /// output indices the `mask` permits, ascending (best first).
    /// Slots beyond the number of mask-allowed entries are `None`.
    ///
    /// On the default surface, so a per-codec picker or `zenpicker` can
    /// run the proven "predict-top-K then encode-verify" path (`K = 3`
    /// typically) at runtime without re-implementing the ranking
    /// contract. Same masking + score-transform + offsets options as
    /// [`Self::argmin_masked`]; same NaN / tie-break / mask-length
    /// contract as [`argmin::argmin_masked_top_k`].
    pub fn argmin_masked_top_k<const K: usize>(
        &mut self,
        features: &[f32],
        mask: &AllowedMask<'_>,
        transform: ScoreTransform,
        offsets: Option<&ArgminOffsets<'_>>,
    ) -> Result<[Option<usize>; K], PredictError> {
        self.predict(features)?;
        if let Some(o) = offsets {
            o.validate(self.output.len())?;
        }
        Ok(argmin::argmin_masked_top_k::<K>(
            &self.output,
            mask,
            transform,
            offsets,
        ))
    }

    /// Top-`K` over a sub-range of the output vector (hybrid-heads
    /// layout — see [`Self::argmin_masked_in_range`]). Returned
    /// indices are within the sub-range. Same support guarantee as
    /// [`Self::argmin_masked_top_k`].
    pub fn argmin_masked_top_k_in_range<const K: usize>(
        &mut self,
        features: &[f32],
        range: (usize, usize),
        mask: &AllowedMask<'_>,
        transform: ScoreTransform,
        offsets: Option<&ArgminOffsets<'_>>,
    ) -> Result<[Option<usize>; K], PredictError> {
        self.predict(features)?;
        let (start, end) = range;
        if end > self.output.len() || start > end {
            return Err(PredictError::OutputDimMismatch {
                expected: self.output.len(),
                got: end,
            });
        }
        if let Some(o) = offsets {
            o.validate(end - start)?;
        }
        Ok(argmin::argmin_masked_top_k::<K>(
            &self.output[start..end],
            mask,
            transform,
            offsets,
        ))
    }

    /// Pick + log-domain confidence (gap to second-best). Returns
    /// `None` when the mask permits zero entries.
    #[cfg(feature = "advanced")]
    pub fn pick_with_confidence(
        &mut self,
        features: &[f32],
        mask: &AllowedMask<'_>,
        transform: ScoreTransform,
        offsets: Option<&ArgminOffsets<'_>>,
    ) -> Result<Option<(usize, f32)>, PredictError> {
        self.predict(features)?;
        if let Some(o) = offsets {
            o.validate(self.output.len())?;
        }
        let top = argmin::argmin_masked_top_k::<2>(&self.output, mask, transform, offsets);
        Ok(pick_confidence_from_top_k(
            &self.output,
            transform,
            offsets,
            top,
        ))
    }

    #[cfg(feature = "advanced")]
    pub fn pick_with_confidence_in_range(
        &mut self,
        features: &[f32],
        range: (usize, usize),
        mask: &AllowedMask<'_>,
        transform: ScoreTransform,
        offsets: Option<&ArgminOffsets<'_>>,
    ) -> Result<Option<(usize, f32)>, PredictError> {
        self.predict(features)?;
        let (start, end) = range;
        if end > self.output.len() || start > end {
            return Err(PredictError::OutputDimMismatch {
                expected: self.output.len(),
                got: end,
            });
        }
        if let Some(o) = offsets {
            o.validate(end - start)?;
        }
        let slice = &self.output[start..end];
        let top = argmin::argmin_masked_top_k::<2>(slice, mask, transform, offsets);
        Ok(pick_confidence_from_top_k(slice, transform, offsets, top))
    }

    /// Run forward pass, then argmin under a caller-supplied score
    /// function (#55). The closure is called for each `i` where
    /// `mask.is_allowed(i)` and reads from the model's output slice
    /// `out` to compute a scalar score; smallest score wins.
    ///
    /// Use this when a `ScoreTransform + ArgminOffsets` adjustment
    /// can't express the desired reduction — e.g. RD-vs-time
    /// (`bytes + μ·ms` reading from two hybrid heads), multi-metric
    /// pickers (selecting one metric's sub-range with a runtime
    /// index), or codec-specific saturating clamps.
    ///
    /// Returns `None` when no entry is allowed by the mask.
    /// `mask.len()` must be ≥ `n_outputs`.
    #[cfg(feature = "advanced")]
    pub fn argmin_masked_with_scorer<F>(
        &mut self,
        features: &[f32],
        mask: &AllowedMask<'_>,
        scorer: F,
    ) -> Result<Option<usize>, PredictError>
    where
        F: Fn(&[f32], usize) -> f32,
    {
        self.predict(features)?;
        let out = &self.output[..];
        Ok(argmin::argmin_masked_with_scorer(out.len(), mask, |i| {
            scorer(out, i)
        }))
    }

    /// Top-`K` argmin under a caller-supplied score function.
    /// Slots beyond the number of mask-allowed entries are `None`.
    #[cfg(feature = "advanced")]
    pub fn argmin_masked_top_k_with_scorer<const K: usize, F>(
        &mut self,
        features: &[f32],
        mask: &AllowedMask<'_>,
        scorer: F,
    ) -> Result<[Option<usize>; K], PredictError>
    where
        F: Fn(&[f32], usize) -> f32,
    {
        self.predict(features)?;
        let out = &self.output[..];
        Ok(argmin::argmin_masked_top_k_with_scorer::<K, _>(
            out.len(),
            mask,
            |i| scorer(out, i),
        ))
    }

    /// Top-`K` argmin with scorer over a sub-range of the output
    /// vector. Slots beyond the number of mask-allowed entries are
    /// `None`. The scorer's `out` is the full predict output (so
    /// the closure can read any head); `i` is the cell index within
    /// the sub-range.
    #[cfg(feature = "advanced")]
    pub fn argmin_masked_top_k_with_scorer_in_range<const K: usize, F>(
        &mut self,
        features: &[f32],
        range: (usize, usize),
        mask: &AllowedMask<'_>,
        scorer: F,
    ) -> Result<[Option<usize>; K], PredictError>
    where
        F: Fn(&[f32], usize) -> f32,
    {
        self.predict(features)?;
        let (start, end) = range;
        if end > self.output.len() || start > end {
            return Err(PredictError::OutputDimMismatch {
                expected: self.output.len(),
                got: end,
            });
        }
        let out = &self.output[..];
        Ok(argmin::argmin_masked_top_k_with_scorer::<K, _>(
            end - start,
            mask,
            |i| scorer(out, i),
        ))
    }

    /// Argmin with scorer over a sub-range of the output vector.
    /// Hybrid-heads pickers expose multiple per-cell heads; this
    /// lets the caller restrict the argmin to e.g. the bytes head
    /// while the closure freely reads from any head. The closure's
    /// `i` is **within the sub-range** (`0..(range.1 - range.0)`);
    /// the scorer's `out` slice is the full predictor output, so
    /// the closure can compute `out[N_CELLS + i]` for the time head
    /// while ranking by cell index `i`.
    #[cfg(feature = "advanced")]
    pub fn argmin_masked_with_scorer_in_range<F>(
        &mut self,
        features: &[f32],
        range: (usize, usize),
        mask: &AllowedMask<'_>,
        scorer: F,
    ) -> Result<Option<usize>, PredictError>
    where
        F: Fn(&[f32], usize) -> f32,
    {
        self.predict(features)?;
        let (start, end) = range;
        if end > self.output.len() || start > end {
            return Err(PredictError::OutputDimMismatch {
                expected: self.output.len(),
                got: end,
            });
        }
        let out = &self.output[..];
        Ok(argmin::argmin_masked_with_scorer(end - start, mask, |i| {
            scorer(out, i)
        }))
    }
}
