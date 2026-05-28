//! Error type for the recompression pipeline.

use thiserror::Error;

/// All failure modes [`crate::recompress::recompress`] can surface.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    /// `target_zensim_a` was outside `[0.0, 100.0]`.
    #[error("target_zensim_a {0} is out of range [0, 100]")]
    TargetOutOfRange(f32),

    /// JPEG header parse / probe failed.
    #[error("failed to probe source JPEG: {0}")]
    Probe(String),

    /// Source has a JPEG feature we do not currently support
    /// (e.g. 12-bit samples, CMYK with non-Adobe transform).
    #[error("unsupported source JPEG: {0}")]
    Unsupported(&'static str),

    /// zenjpeg encode/decode reported an internal error.
    #[error("zenjpeg I/O error: {0}")]
    Zenjpeg(String),

    /// zensim scoring failed (e.g. malformed pixel buffer).
    #[error("zensim error: {0}")]
    Zensim(String),

    /// Internal invariant violated — a strategy returned a result that
    /// disagreed with what its calibration predicted.
    #[error("internal: {0}")]
    Internal(&'static str),
}

// `crate::error` is private but the same type is re-exported from
// `crate::encoder::Error` (and `crate::decoder::Error`). Hook into
// that re-export.
impl From<crate::encoder::Error> for Error {
    fn from(e: crate::encoder::Error) -> Self {
        Error::Zenjpeg(format!("{e}"))
    }
}
