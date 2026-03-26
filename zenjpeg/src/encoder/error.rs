//! Encoder error types.
//!
//! The encoder uses a hierarchical error structure:
//! - [`ArgumentError`](crate::error::ArgumentError) - Invalid arguments
//! - [`ResourceError`](crate::error::ResourceError) - Memory/IO failures
//! - [`EncoderArgError`] - Invalid encoder-specific arguments
//! - [`EncoderStateError`] - Encoder state/usage errors

use alloc::string::String;
use core::fmt;
use thiserror::Error;
use whereat::{AtTrace, AtTraceBoxed, AtTraceable};

// Re-export shared error types
pub use crate::error::{ArgumentError, ResourceError};

/// Result type for encoder operations.
pub type Result<T> = core::result::Result<T, Error>;

// ============================================================================
// Encoder-specific: Argument errors
// ============================================================================

/// Encoder-specific argument errors.
///
/// These indicate invalid encoder configuration, not runtime failures.
#[derive(Debug, Clone, PartialEq, Error)]
#[non_exhaustive]
pub enum EncoderArgError {
    /// Invalid quality parameter.
    #[error("invalid quality {value}: must be in {valid_range}")]
    InvalidQuality { value: f32, valid_range: &'static str },

    /// Invalid scan script for progressive encoding.
    #[error("invalid scan script: {0}")]
    InvalidScanScript(String),

    /// Invalid encoder configuration.
    #[error("invalid encoder configuration: {0}")]
    InvalidConfig(String),

    /// Stride too small for image width.
    #[error("stride {stride} is too small for width {width} pixels")]
    StrideTooSmall { width: u32, stride: usize },
}

// ============================================================================
// Encoder-specific: State errors
// ============================================================================

/// Encoder state/usage errors.
///
/// These indicate incorrect API usage sequence, not invalid input.
#[derive(Debug, Clone, PartialEq, Error)]
#[non_exhaustive]
pub enum EncoderStateError {
    /// Pushed more rows than image height.
    #[error("pushed {pushed} rows but image height is only {height}")]
    TooManyRows { height: u32, pushed: u32 },

    /// Encoding finished without all rows pushed.
    #[error("encoding finished after {pushed} rows but image height is {height}")]
    IncompleteImage { height: u32, pushed: u32 },
}

// ============================================================================
// Encoder ErrorKind - Composed from shared + encoder-specific
// ============================================================================

/// The specific kind of encoder error.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ErrorKind {
    /// Invalid shared argument from caller.
    Argument(ArgumentError),

    /// Invalid encoder-specific argument.
    EncoderArg(EncoderArgError),

    /// Resource exhaustion or I/O failure.
    Resource(ResourceError),

    /// Encoder state/usage error.
    State(EncoderStateError),

    /// ICC color management error.
    Icc(String),

    /// Internal error (should not happen in correct usage).
    Internal { reason: &'static str },

    /// Operation was cancelled.
    Cancelled,
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Argument(e) => write!(f, "{}", e),
            Self::EncoderArg(e) => write!(f, "{}", e),
            Self::Resource(e) => write!(f, "{}", e),
            Self::State(e) => write!(f, "{}", e),
            Self::Icc(reason) => write!(f, "ICC error: {}", reason),
            Self::Internal { reason } => write!(f, "internal error: {}", reason),
            Self::Cancelled => write!(f, "operation cancelled"),
        }
    }
}

// ============================================================================
// Encoder Error - Main error type with location tracking
// ============================================================================

/// Errors that can occur during JPEG encoding.
///
/// Use [`Error::kind()`] to match on the specific error variant.
#[derive(Debug)]
pub struct Error {
    kind: ErrorKind,
    trace: AtTraceBoxed,
}

impl Error {
    /// Create a new error with the given kind, capturing the current location.
    #[track_caller]
    pub fn new(kind: ErrorKind) -> Self {
        Self {
            kind,
            trace: AtTraceBoxed::capture(),
        }
    }

    /// Create a new error without capturing a trace.
    #[inline]
    pub const fn new_untraced(kind: ErrorKind) -> Self {
        Self {
            kind,
            trace: AtTraceBoxed::new(),
        }
    }

    /// Get the kind of error.
    #[inline]
    pub fn kind(&self) -> &ErrorKind {
        &self.kind
    }

    /// Convert into the error kind, discarding the trace.
    #[inline]
    pub fn into_kind(self) -> ErrorKind {
        self.kind
    }

    // ========================================================================
    // Convenience constructors - Shared argument errors
    // ========================================================================

    #[track_caller]
    pub fn invalid_dimensions(width: u32, height: u32, reason: &'static str) -> Self {
        Self::new(ErrorKind::Argument(ArgumentError::InvalidDimensions {
            width,
            height,
            reason,
        }))
    }

    #[track_caller]
    pub fn invalid_color_format(reason: &'static str) -> Self {
        Self::new(ErrorKind::Argument(ArgumentError::InvalidColorFormat {
            reason,
        }))
    }

    #[track_caller]
    pub fn invalid_buffer_size(expected: usize, actual: usize) -> Self {
        Self::new(ErrorKind::Argument(ArgumentError::InvalidBufferSize {
            expected,
            actual,
        }))
    }

    #[track_caller]
    pub fn unsupported_feature(feature: &'static str) -> Self {
        Self::new(ErrorKind::Argument(ArgumentError::UnsupportedFeature {
            feature,
        }))
    }

    #[track_caller]
    pub fn unsupported_pixel_format(format: crate::types::PixelFormat) -> Self {
        Self::new(ErrorKind::Argument(ArgumentError::UnsupportedPixelFormat {
            format,
        }))
    }

    // ========================================================================
    // Convenience constructors - Encoder argument errors
    // ========================================================================

    #[track_caller]
    pub fn invalid_quality(value: f32, valid_range: &'static str) -> Self {
        Self::new(ErrorKind::EncoderArg(EncoderArgError::InvalidQuality {
            value,
            valid_range,
        }))
    }

    #[track_caller]
    pub fn invalid_scan_script(reason: String) -> Self {
        Self::new(ErrorKind::EncoderArg(EncoderArgError::InvalidScanScript(
            reason,
        )))
    }

    #[track_caller]
    pub fn invalid_config(reason: String) -> Self {
        Self::new(ErrorKind::EncoderArg(EncoderArgError::InvalidConfig(
            reason,
        )))
    }

    #[track_caller]
    pub fn stride_too_small(width: u32, stride: usize) -> Self {
        Self::new(ErrorKind::EncoderArg(EncoderArgError::StrideTooSmall {
            width,
            stride,
        }))
    }

    // ========================================================================
    // Convenience constructors - Resource errors
    // ========================================================================

    #[track_caller]
    pub fn allocation_failed(bytes: usize, context: &'static str) -> Self {
        Self::new(ErrorKind::Resource(ResourceError::AllocationFailed {
            bytes,
            context,
        }))
    }

    #[track_caller]
    pub fn size_overflow(context: &'static str) -> Self {
        Self::new(ErrorKind::Resource(ResourceError::SizeOverflow { context }))
    }

    #[track_caller]
    pub fn image_too_large(pixels: u64, limit: u64) -> Self {
        Self::new(ErrorKind::Resource(ResourceError::ImageTooLarge {
            pixels,
            limit,
        }))
    }

    #[track_caller]
    pub fn io_error(reason: String) -> Self {
        Self::new(ErrorKind::Resource(ResourceError::IoError { reason }))
    }

    // ========================================================================
    // Convenience constructors - State errors
    // ========================================================================

    #[track_caller]
    pub fn too_many_rows(height: u32, pushed: u32) -> Self {
        Self::new(ErrorKind::State(EncoderStateError::TooManyRows {
            height,
            pushed,
        }))
    }

    #[track_caller]
    pub fn incomplete_image(height: u32, pushed: u32) -> Self {
        Self::new(ErrorKind::State(EncoderStateError::IncompleteImage {
            height,
            pushed,
        }))
    }

    // ========================================================================
    // Convenience constructors - Other errors
    // ========================================================================

    #[track_caller]
    pub fn icc_error(reason: String) -> Self {
        Self::new(ErrorKind::Icc(reason))
    }

    #[track_caller]
    pub fn internal(reason: &'static str) -> Self {
        Self::new(ErrorKind::Internal { reason })
    }

    #[track_caller]
    pub fn cancelled() -> Self {
        Self::new(ErrorKind::Cancelled)
    }
}

impl AtTraceable for Error {
    fn trace_mut(&mut self) -> &mut AtTrace {
        self.trace.get_or_insert_mut()
    }

    fn trace(&self) -> Option<&AtTrace> {
        self.trace.as_ref()
    }

    fn fmt_message(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.kind, f)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.kind, f)
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    #[track_caller]
    fn from(err: std::io::Error) -> Self {
        Self::io_error(err.to_string())
    }
}

impl From<enough::StopReason> for Error {
    #[track_caller]
    fn from(_: enough::StopReason) -> Self {
        Self::cancelled()
    }
}

/// Convert from internal error type.
impl From<crate::error::Error> for Error {
    #[track_caller]
    fn from(err: crate::error::Error) -> Self {
        use crate::error::ErrorKind as EK;
        let kind = match err.into_kind() {
            // Shared argument errors
            EK::InvalidDimensions {
                width,
                height,
                reason,
            } => ErrorKind::Argument(ArgumentError::InvalidDimensions {
                width,
                height,
                reason,
            }),
            EK::InvalidColorFormat { reason } => {
                ErrorKind::Argument(ArgumentError::InvalidColorFormat { reason })
            }
            EK::InvalidBufferSize { expected, actual } => {
                ErrorKind::Argument(ArgumentError::InvalidBufferSize { expected, actual })
            }
            EK::UnsupportedFeature { feature } => {
                ErrorKind::Argument(ArgumentError::UnsupportedFeature { feature })
            }
            EK::UnsupportedPixelFormat { format } => {
                ErrorKind::Argument(ArgumentError::UnsupportedPixelFormat { format })
            }

            // Encoder-specific argument errors
            EK::InvalidQuality { value, valid_range } => {
                ErrorKind::EncoderArg(EncoderArgError::InvalidQuality { value, valid_range })
            }
            EK::InvalidScanScript(reason) => {
                ErrorKind::EncoderArg(EncoderArgError::InvalidScanScript(reason))
            }
            EK::InvalidConfig(reason) => {
                ErrorKind::EncoderArg(EncoderArgError::InvalidConfig(reason))
            }
            EK::StrideTooSmall { width, stride } => {
                ErrorKind::EncoderArg(EncoderArgError::StrideTooSmall { width, stride })
            }

            // Resource errors
            EK::AllocationFailed { bytes, context } => {
                ErrorKind::Resource(ResourceError::AllocationFailed { bytes, context })
            }
            EK::SizeOverflow { context } => {
                ErrorKind::Resource(ResourceError::SizeOverflow { context })
            }
            EK::ImageTooLarge { pixels, limit } => {
                ErrorKind::Resource(ResourceError::ImageTooLarge { pixels, limit })
            }
            EK::IoError { reason } => ErrorKind::Resource(ResourceError::IoError { reason }),

            // State errors
            EK::TooManyRows { height, pushed } => {
                ErrorKind::State(EncoderStateError::TooManyRows { height, pushed })
            }
            EK::IncompleteImage { height, pushed } => {
                ErrorKind::State(EncoderStateError::IncompleteImage { height, pushed })
            }

            // Other shared
            EK::IccError(reason) => ErrorKind::Icc(reason),
            EK::InternalError { reason } => ErrorKind::Internal { reason },
            EK::Cancelled => ErrorKind::Cancelled,

            // Decoder-specific errors should not occur in encoder - convert to internal error
            EK::InvalidJpegData { reason } | EK::TruncatedData { context: reason } => {
                ErrorKind::Internal { reason }
            }
            EK::InvalidMarker { marker, context } => ErrorKind::Internal {
                reason: if marker == 0 {
                    context
                } else {
                    "invalid marker"
                },
            },
            EK::InvalidHuffmanTable { reason, .. } | EK::InvalidQuantTable { reason, .. } => {
                ErrorKind::Internal { reason }
            }
            EK::DecodeError(reason) => ErrorKind::Internal {
                reason: if reason.is_empty() {
                    "decode error"
                } else {
                    "unexpected decode error"
                },
            },
            EK::TooManyScans { .. } => ErrorKind::Internal {
                reason: "too many scans",
            },
        };
        Self::new(kind)
    }
}

// Implement Clone manually since AtTrace doesn't implement Clone
impl Clone for Error {
    fn clone(&self) -> Self {
        Self {
            kind: self.kind.clone(),
            trace: AtTraceBoxed::new(), // Don't clone the trace
        }
    }
}

// Implement PartialEq based on kind only
impl PartialEq for Error {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind
    }
}
