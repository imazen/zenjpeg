//! Errors raised by model loading, metadata access, and inference.

use core::fmt;

#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PredictError {
    /// Header magic bytes don't match `ZNPR`.
    BadMagic { found: [u8; 4] },
    /// Format version not supported by this build. v3 is the only
    /// version this crate parses; older bakes must be migrated via
    /// `zentrain/tools/migrate_znpr_v2_to_v3.py`.
    UnsupportedVersion { version: u16, expected: u16 },
    /// Bytes ran out before a section completed parsing.
    Truncated {
        offset: usize,
        want: usize,
        have: usize,
    },
    /// A `Section` (offset, len) addressed bytes outside the file.
    SectionOutOfRange {
        what: &'static str,
        offset: u32,
        len: u32,
        file_len: usize,
    },
    /// A typed slice view requested alignment the input bytes
    /// don't satisfy. Wrap `include_bytes!` in an `#[repr(C, align(16))]`
    /// struct (see crate docs) to guarantee alignment for f32/u32/f64
    /// payloads.
    SectionMisaligned {
        what: &'static str,
        offset: u32,
        required_align: usize,
    },
    /// `weight_dtype` byte was not 0 (f32), 1 (f16), or 2 (i8).
    UnknownWeightDtype { byte: u8 },
    /// `activation` byte was not a recognized variant.
    UnknownActivation { byte: u8 },
    /// Layer `n` declared `in_dim` that doesn't match the prior
    /// layer's `out_dim` (or, for layer 0, doesn't match `n_inputs`).
    LayerDimMismatch {
        layer: usize,
        expected_in: usize,
        got_in: usize,
    },
    /// Final layer's `out_dim` doesn't match the header's `n_outputs`.
    OutputDimMismatch { expected: usize, got: usize },
    /// A header dimension was zero where it must be positive.
    ZeroDimension { what: &'static str },
    /// A dimension multiplication overflowed `usize` during parse —
    /// e.g. `n_inputs * out_dim`, `n_layers * sizeof(LayerEntry)`,
    /// `expected_count * 4` for f32 sections. Crafted bakes can trip
    /// this on 32-bit targets (`i686-unknown-linux-gnu`,
    /// `wasm32-unknown-unknown`); safe parsers reject rather than wrap.
    DimensionOverflow { what: &'static str },
    /// Caller passed a feature vector of the wrong length to
    /// `Predictor::predict` or related entries.
    FeatureLenMismatch { expected: usize, got: usize },
    /// `ArgminOffsets::per_output` length didn't match `n_outputs`
    /// (or the sub-range length, when using the `_in_range` family).
    OffsetsLenMismatch { expected: usize, got: usize },
    /// Schema hash baked into the model didn't match the value the
    /// caller pinned via [`Model::from_bytes_with_schema`].
    ///
    /// [`Model::from_bytes_with_schema`]: crate::Model::from_bytes_with_schema
    SchemaHashMismatch { expected: u64, got: u64 },
    /// A metadata entry's key bytes weren't valid UTF-8.
    MetadataKeyNotUtf8 { offset: usize },
    /// A metadata entry's value type didn't match what the typed
    /// accessor (`get_utf8`, `get_numeric`, `get_bytes`) expected.
    MetadataTypeMismatch {
        key_len: usize,
        expected: crate::MetadataType,
        got: crate::MetadataType,
    },
    /// `get_utf8` saw a value that wasn't valid UTF-8.
    MetadataValueNotUtf8 { key_len: usize },
    /// `zentrain.feature_transforms` value carried a token that
    /// wasn't one of `identity` / `log` / `log1p`.
    UnknownFeatureTransform,
    /// `zentrain.feature_transforms` line count didn't equal
    /// `n_inputs` (or the runtime helper's `transforms` slice
    /// disagreed with `features.len()`).
    FeatureTransformsLenMismatch { expected: usize, got: usize },
    /// A scalar-only pipeline (`apply_feature_transforms`) was called
    /// with an expander variant (today: only `FeatureTransform::Sinusoidal`)
    /// in the transforms list. Use `apply_feature_pipeline_expanding`,
    /// or call the expander-aware Predictor entry point that routes
    /// automatically.
    UnexpectedExpanderInScalarPipeline { feature_index: usize },
    /// A `zenpicker.knob_vetoes` record carried an op byte that wasn't
    /// `0` (LessThan) or `1` (GreaterThan). Reject rather than silently
    /// mis-apply a safety bound.
    UnknownVetoOp { byte: u8 },
}

impl fmt::Display for PredictError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::BadMagic { found } => {
                write!(f, "zenpredict: bad magic, expected ZNPR, found {found:?}")
            }
            Self::UnsupportedVersion { version, expected } => write!(
                f,
                "zenpredict: format version {version} not supported (expected {expected})"
            ),
            Self::Truncated { offset, want, have } => write!(
                f,
                "zenpredict: truncated at offset {offset}, wanted {want} bytes, have {have}"
            ),
            Self::SectionOutOfRange {
                what,
                offset,
                len,
                file_len,
            } => write!(
                f,
                "zenpredict: section `{what}` (offset={offset}, len={len}) out of range \
                 for file_len={file_len}"
            ),
            Self::SectionMisaligned {
                what,
                offset,
                required_align,
            } => write!(
                f,
                "zenpredict: section `{what}` at offset {offset} is not {required_align}-aligned"
            ),
            Self::UnknownWeightDtype { byte } => {
                write!(f, "zenpredict: unknown weight dtype byte {byte:#x}")
            }
            Self::UnknownActivation { byte } => {
                write!(f, "zenpredict: unknown activation byte {byte:#x}")
            }
            Self::LayerDimMismatch {
                layer,
                expected_in,
                got_in,
            } => write!(
                f,
                "zenpredict: layer {layer} expected in_dim {expected_in}, got {got_in}"
            ),
            Self::OutputDimMismatch { expected, got } => write!(
                f,
                "zenpredict: final layer out_dim {got} != header n_outputs {expected}"
            ),
            Self::ZeroDimension { what } => write!(f, "zenpredict: zero dimension in `{what}`"),
            Self::DimensionOverflow { what } => {
                write!(f, "zenpredict: dimension overflow in `{what}`")
            }
            Self::FeatureLenMismatch { expected, got } => write!(
                f,
                "zenpredict: feature vector length {got} != n_inputs {expected}"
            ),
            Self::OffsetsLenMismatch { expected, got } => write!(
                f,
                "zenpredict: ArgminOffsets::per_output length {got} != expected {expected}"
            ),
            Self::SchemaHashMismatch { expected, got } => write!(
                f,
                "zenpredict: schema_hash mismatch: expected {expected:#018x}, got {got:#018x}"
            ),
            Self::MetadataKeyNotUtf8 { offset } => write!(
                f,
                "zenpredict: metadata key at offset {offset} is not valid UTF-8"
            ),
            Self::MetadataTypeMismatch {
                key_len,
                expected,
                got,
            } => write!(
                f,
                "zenpredict: metadata type mismatch (key len {key_len}): expected {expected:?}, got {got:?}"
            ),
            Self::MetadataValueNotUtf8 { key_len } => write!(
                f,
                "zenpredict: metadata value (key len {key_len}) declared utf8 but failed validation"
            ),
            Self::UnknownFeatureTransform => write!(
                f,
                "zenpredict: feature_transforms value carried an unknown token \
                 (expected one of identity/log/log1p)"
            ),
            Self::FeatureTransformsLenMismatch { expected, got } => write!(
                f,
                "zenpredict: feature_transforms length {got} != expected {expected}"
            ),
            Self::UnexpectedExpanderInScalarPipeline { feature_index } => write!(
                f,
                "zenpredict: scalar feature-transform pipeline encountered an \
                 expander variant at feature index {feature_index} \
                 (use apply_feature_pipeline_expanding or the auto-routing \
                 Predictor entry point)"
            ),
            Self::UnknownVetoOp { byte } => write!(
                f,
                "zenpredict: knob_vetoes op byte {byte:#x} not recognized (expected 0=< or 1=>)"
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for PredictError {}
