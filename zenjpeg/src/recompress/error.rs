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

/// Codec-agnostic classification of recompression failures (zencodec #103).
///
/// The downstream-tool arms (`Zenjpeg` / `Zensim`) flatten their typed cause to
/// a `String`, so the underlying category can't be delegated; they report as
/// [`Internal`](zencodec::ErrorCategory::Internal) — a failure inside the
/// recompress pipeline, not attributable to the caller's request.
impl zencodec::CategorizedError for Error {
    fn codec_name(&self) -> Option<&'static str> {
        Some("zenjpeg")
    }

    fn category(&self) -> zencodec::ErrorCategory {
        use zencodec::ErrorCategory as C;
        match self {
            // Caller asked for a target outside [0, 100].
            Self::TargetOutOfRange(_) => C::InvalidParameters,
            // Probing the source JPEG header failed → unreadable / bad input.
            Self::Probe(_) => C::MalformedImage,
            // The source uses a JPEG feature the recompressor doesn't handle.
            Self::Unsupported(_) => C::UnsupportedImageFeature,
            // Downstream zenjpeg / zensim failures (typed cause flattened to a
            // String) and broken internal invariants are all internal faults.
            Self::Zenjpeg(_) | Self::Zensim(_) | Self::Internal(_) => C::Internal,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zencodec::{CategorizedError, ErrorCategory as C};

    #[test]
    fn category_mapping() {
        assert_eq!(Error::TargetOutOfRange(101.0).codec_name(), Some("zenjpeg"));
        assert_eq!(
            Error::TargetOutOfRange(101.0).category(),
            C::InvalidParameters
        );
        assert_eq!(Error::Probe("eof".into()).category(), C::MalformedImage);
        assert_eq!(
            Error::Unsupported("12-bit").category(),
            C::UnsupportedImageFeature
        );
        assert_eq!(Error::Zenjpeg("io".into()).category(), C::Internal);
        assert_eq!(Error::Zensim("score".into()).category(), C::Internal);
        assert_eq!(Error::Internal("invariant").category(), C::Internal);
    }
}
