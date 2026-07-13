//! Error type for the recompression pipeline.

use thiserror::Error;

/// All failure modes [`crate::recompress::recompress`] can surface.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    /// `target_zensim_a` was outside `[0.0, 100.0]`.
    #[error("target_zensim_a {0} is out of range [0, 100]")]
    TargetOutOfRange(f32),

    /// JPEG header parse / probe failed on structurally-broken input (has a
    /// SOI but is otherwise malformed, e.g. missing quant tables). Truncated
    /// / too-short input uses [`ProbeTruncated`](Self::ProbeTruncated)
    /// instead, which categorizes as `UnexpectedEof` rather than
    /// `MalformedImage`.
    #[error("failed to probe source JPEG: {0}")]
    Probe(String),

    /// Source has a JPEG feature we do not currently support
    /// (e.g. 12-bit samples, CMYK with non-Adobe transform).
    #[error("unsupported source JPEG: {0}")]
    Unsupported(&'static str),

    /// zenjpeg encode/decode reported an internal error whose category could
    /// not be determined (legacy flattened path; prefer
    /// [`ZenjpegCategorized`](Self::ZenjpegCategorized) at new call sites).
    #[error("zenjpeg I/O error: {0}")]
    Zenjpeg(String),

    /// zensim scoring failed (e.g. malformed pixel buffer).
    #[error("zensim error: {0}")]
    Zensim(String),

    /// Internal invariant violated — a strategy returned a result that
    /// disagreed with what its calibration predicted.
    #[error("internal: {0}")]
    Internal(&'static str),

    /// JPEG header parse / probe failed because the input ended before a
    /// complete header could be read (too short, or truncated mid-header).
    /// Distinct from [`Probe`](Self::Probe): categorizes as `UnexpectedEof`
    /// (caller can retry with more data) rather than `MalformedImage`.
    #[error("failed to probe source JPEG (truncated): {0}")]
    ProbeTruncated(String),

    /// A downstream zenjpeg encode/decode failure whose original
    /// [`zencodec::ErrorCategory`] is preserved (as opposed to
    /// [`Zenjpeg`](Self::Zenjpeg), which flattens to `Internal`). Used at
    /// call sites where the underlying `crate::error::Error` implements
    /// [`zencodec::CategorizedError`] and the category is captured before
    /// the typed cause is formatted away to a `String`.
    #[error("zenjpeg error: {message}")]
    ZenjpegCategorized {
        message: String,
        category: zencodec::ErrorCategory,
    },
}

// `crate::error` is private but the same type is re-exported from
// `crate::encoder::Error` (and `crate::decoder::Error`). Hook into
// that re-export.
//
// Captures `e.category()` (via `crate::error::Error`'s `CategorizedError`
// impl) before formatting `e` away to a `String`, so the original category
// survives instead of collapsing to `Internal`.
impl From<crate::encoder::Error> for Error {
    fn from(e: crate::encoder::Error) -> Self {
        use zencodec::CategorizedError;
        let category = e.category();
        Error::ZenjpegCategorized {
            message: format!("{e}"),
            category,
        }
    }
}

/// Codec-agnostic classification of recompression failures (zencodec #103).
///
/// The legacy `Zenjpeg` / `Zensim` arms flatten their typed cause to a
/// `String` with no category alongside it, so they report as
/// [`Internal`](zencodec::ErrorCategory::Internal) — a failure inside the
/// recompress pipeline, not attributable to the caller's request.
/// `ZenjpegCategorized` avoids that by carrying the category captured before
/// flattening (see its `From<crate::encoder::Error>` construction site).
impl zencodec::CategorizedError for Error {
    fn codec_name(&self) -> Option<&'static str> {
        Some("zenjpeg")
    }

    fn category(&self) -> zencodec::ErrorCategory {
        use zencodec::{
            ErrorCategory as C, ImageError as Img, InternalKind as Int, InvalidKind as Inv,
            RequestError as Req, UnsupportedImageKind as UImg,
        };
        match self {
            // Caller asked for a target outside [0, 100].
            Self::TargetOutOfRange(_) => C::Request(Req::Invalid(Inv::Parameters)),
            // Probing the source JPEG header failed on structurally-broken
            // (but not truncated) input → unreadable / bad input.
            Self::Probe(_) => C::Image(Img::Malformed),
            // Probing hit EOF before a complete header — caller can retry
            // with more data, distinct from malformed content.
            Self::ProbeTruncated(_) => C::Image(Img::UnexpectedEof),
            // The source uses a JPEG feature the recompressor doesn't handle.
            Self::Unsupported(_) => C::Image(Img::Unsupported(UImg::Feature)),
            // Legacy flattened downstream zenjpeg / zensim failures and
            // broken internal invariants are all internal faults.
            Self::Zenjpeg(_) | Self::Zensim(_) | Self::Internal(_) => C::Internal(Int::Bug),
            // Category preserved from the original typed cause.
            Self::ZenjpegCategorized { category, .. } => *category,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zencodec::{CategorizedError, ErrorCategory as C};

    #[test]
    fn category_mapping() {
        use zencodec::{
            ImageError as Img, InternalKind as Int, InvalidKind as Inv, RequestError as Req,
            UnsupportedImageKind as UImg,
        };
        assert_eq!(Error::TargetOutOfRange(101.0).codec_name(), Some("zenjpeg"));
        assert_eq!(
            Error::TargetOutOfRange(101.0).category(),
            C::Request(Req::Invalid(Inv::Parameters))
        );
        assert_eq!(
            Error::Probe("eof".into()).category(),
            C::Image(Img::Malformed)
        );
        assert_eq!(
            Error::ProbeTruncated("too short".into()).category(),
            C::Image(Img::UnexpectedEof)
        );
        assert_eq!(
            Error::Unsupported("12-bit").category(),
            C::Image(Img::Unsupported(UImg::Feature))
        );
        assert_eq!(
            Error::Zenjpeg("io".into()).category(),
            C::Internal(Int::Bug)
        );
        assert_eq!(
            Error::Zensim("score".into()).category(),
            C::Internal(Int::Bug)
        );
        assert_eq!(
            Error::Internal("invariant").category(),
            C::Internal(Int::Bug)
        );
        assert_eq!(
            Error::ZenjpegCategorized {
                message: "malformed".into(),
                category: C::Image(Img::Malformed),
            }
            .category(),
            C::Image(Img::Malformed)
        );
    }

    /// `From<crate::encoder::Error>` must preserve the source error's
    /// category instead of collapsing every zenjpeg encode/decode failure to
    /// `Internal` (the bug this variant fixes).
    #[test]
    fn from_encoder_error_preserves_category() {
        use zencodec::ImageError as Img;
        let encoder_err = crate::error::Error::invalid_jpeg_data("bad SOI");
        let recompress_err: Error = encoder_err.into();
        assert_eq!(recompress_err.category(), C::Image(Img::Malformed));
        assert!(matches!(recompress_err, Error::ZenjpegCategorized { .. }));
    }
}
