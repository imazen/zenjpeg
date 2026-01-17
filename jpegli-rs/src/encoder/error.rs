//! Encoder error types.

use alloc::string::String;
use core::fmt;
use whereat::{AtTrace, AtTraceBoxed, AtTraceable};

/// Result type for encoder operations.
pub type Result<T> = core::result::Result<T, Error>;

/// Errors that can occur during JPEG encoding.
///
/// Use [`Error::kind()`] to match on the specific error variant.
#[derive(Debug)]
pub struct Error {
    kind: ErrorKind,
    trace: AtTraceBoxed,
}

/// The specific kind of encoder error.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ErrorKind {
    /// Invalid input dimensions (zero or too large).
    InvalidDimensions {
        width: u32,
        height: u32,
        reason: &'static str,
    },
    /// Invalid quality parameter.
    InvalidQuality {
        value: f32,
        valid_range: &'static str,
    },
    /// Invalid color space or pixel format combination.
    InvalidColorFormat { reason: &'static str },
    /// Input buffer has wrong size.
    InvalidBufferSize { expected: usize, actual: usize },
    /// Unsupported JPEG feature.
    UnsupportedFeature { feature: &'static str },
    /// Internal error (should not happen in correct usage).
    InternalError { reason: &'static str },
    /// I/O error during encoding.
    IoError { reason: String },
    /// ICC color management error.
    IccError(String),
    /// Invalid scan script for progressive encoding.
    InvalidScanScript(String),
    /// Memory allocation failed.
    AllocationFailed { bytes: usize, context: &'static str },
    /// Size calculation overflowed.
    SizeOverflow { context: &'static str },
    /// Image exceeds maximum pixel limit.
    ImageTooLarge { pixels: u64, limit: u64 },
    /// Operation was cancelled.
    Cancelled,
    /// Pixel format not supported.
    UnsupportedPixelFormat { format: crate::types::PixelFormat },
    /// Invalid encoder configuration.
    InvalidConfig(String),
    /// Stride too small for image width.
    StrideTooSmall { width: u32, stride: usize },
    /// Pushed more rows than image height.
    TooManyRows { height: u32, pushed: u32 },
    /// Encoding finished without all rows pushed.
    IncompleteImage { height: u32, pushed: u32 },
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

    // Convenience constructors

    #[track_caller]
    pub fn invalid_dimensions(width: u32, height: u32, reason: &'static str) -> Self {
        Self::new(ErrorKind::InvalidDimensions {
            width,
            height,
            reason,
        })
    }

    #[track_caller]
    pub fn invalid_quality(value: f32, valid_range: &'static str) -> Self {
        Self::new(ErrorKind::InvalidQuality { value, valid_range })
    }

    #[track_caller]
    pub fn invalid_color_format(reason: &'static str) -> Self {
        Self::new(ErrorKind::InvalidColorFormat { reason })
    }

    #[track_caller]
    pub fn invalid_buffer_size(expected: usize, actual: usize) -> Self {
        Self::new(ErrorKind::InvalidBufferSize { expected, actual })
    }

    #[track_caller]
    pub fn unsupported_feature(feature: &'static str) -> Self {
        Self::new(ErrorKind::UnsupportedFeature { feature })
    }

    #[track_caller]
    pub fn internal(reason: &'static str) -> Self {
        Self::new(ErrorKind::InternalError { reason })
    }

    #[track_caller]
    pub fn io_error(reason: String) -> Self {
        Self::new(ErrorKind::IoError { reason })
    }

    #[track_caller]
    pub fn icc_error(reason: String) -> Self {
        Self::new(ErrorKind::IccError(reason))
    }

    #[track_caller]
    pub fn invalid_scan_script(reason: String) -> Self {
        Self::new(ErrorKind::InvalidScanScript(reason))
    }

    #[track_caller]
    pub fn allocation_failed(bytes: usize, context: &'static str) -> Self {
        Self::new(ErrorKind::AllocationFailed { bytes, context })
    }

    #[track_caller]
    pub fn size_overflow(context: &'static str) -> Self {
        Self::new(ErrorKind::SizeOverflow { context })
    }

    #[track_caller]
    pub fn image_too_large(pixels: u64, limit: u64) -> Self {
        Self::new(ErrorKind::ImageTooLarge { pixels, limit })
    }

    #[track_caller]
    pub fn cancelled() -> Self {
        Self::new(ErrorKind::Cancelled)
    }

    #[track_caller]
    pub fn unsupported_pixel_format(format: crate::types::PixelFormat) -> Self {
        Self::new(ErrorKind::UnsupportedPixelFormat { format })
    }

    #[track_caller]
    pub fn invalid_config(reason: String) -> Self {
        Self::new(ErrorKind::InvalidConfig(reason))
    }

    #[track_caller]
    pub fn stride_too_small(width: u32, stride: usize) -> Self {
        Self::new(ErrorKind::StrideTooSmall { width, stride })
    }

    #[track_caller]
    pub fn too_many_rows(height: u32, pushed: u32) -> Self {
        Self::new(ErrorKind::TooManyRows { height, pushed })
    }

    #[track_caller]
    pub fn incomplete_image(height: u32, pushed: u32) -> Self {
        Self::new(ErrorKind::IncompleteImage { height, pushed })
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

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimensions {
                width,
                height,
                reason,
            } => {
                write!(f, "invalid dimensions {}x{}: {}", width, height, reason)
            }
            Self::InvalidQuality { value, valid_range } => {
                write!(f, "invalid quality {}: must be in {}", value, valid_range)
            }
            Self::InvalidColorFormat { reason } => {
                write!(f, "invalid color format: {}", reason)
            }
            Self::InvalidBufferSize { expected, actual } => {
                write!(
                    f,
                    "invalid buffer size: expected {} bytes, got {}",
                    expected, actual
                )
            }
            Self::UnsupportedFeature { feature } => {
                write!(f, "unsupported feature: {}", feature)
            }
            Self::InternalError { reason } => {
                write!(f, "internal error: {}", reason)
            }
            Self::IoError { reason } => {
                write!(f, "I/O error: {}", reason)
            }
            Self::IccError(reason) => {
                write!(f, "ICC error: {}", reason)
            }
            Self::InvalidScanScript(reason) => {
                write!(f, "invalid scan script: {}", reason)
            }
            Self::AllocationFailed { bytes, context } => {
                write!(f, "allocation of {} bytes failed while {}", bytes, context)
            }
            Self::SizeOverflow { context } => {
                write!(f, "size calculation overflow while {}", context)
            }
            Self::ImageTooLarge { pixels, limit } => {
                write!(
                    f,
                    "image too large: {} pixels exceeds limit of {}",
                    pixels, limit
                )
            }
            Self::Cancelled => write!(f, "operation cancelled"),
            Self::UnsupportedPixelFormat { format } => {
                write!(f, "pixel format {:?} not supported", format)
            }
            Self::InvalidConfig(reason) => {
                write!(f, "invalid encoder configuration: {}", reason)
            }
            Self::StrideTooSmall { width, stride } => {
                write!(
                    f,
                    "stride {} is too small for width {} pixels",
                    stride, width
                )
            }
            Self::TooManyRows { height, pushed } => {
                write!(
                    f,
                    "pushed {} rows but image height is only {}",
                    pushed, height
                )
            }
            Self::IncompleteImage { height, pushed } => {
                write!(
                    f,
                    "encoding finished after {} rows but image height is {}",
                    pushed, height
                )
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

#[cfg(feature = "std")]
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
            EK::InvalidDimensions {
                width,
                height,
                reason,
            } => ErrorKind::InvalidDimensions {
                width,
                height,
                reason,
            },
            EK::InvalidQuality { value, valid_range } => {
                ErrorKind::InvalidQuality { value, valid_range }
            }
            EK::InvalidColorFormat { reason } => ErrorKind::InvalidColorFormat { reason },
            EK::InvalidBufferSize { expected, actual } => {
                ErrorKind::InvalidBufferSize { expected, actual }
            }
            EK::UnsupportedFeature { feature } => ErrorKind::UnsupportedFeature { feature },
            EK::InternalError { reason } => ErrorKind::InternalError { reason },
            EK::IoError { reason } => ErrorKind::IoError { reason },
            EK::IccError(reason) => ErrorKind::IccError(reason),
            EK::InvalidScanScript(reason) => ErrorKind::InvalidScanScript(reason),
            EK::AllocationFailed { bytes, context } => {
                ErrorKind::AllocationFailed { bytes, context }
            }
            EK::SizeOverflow { context } => ErrorKind::SizeOverflow { context },
            EK::ImageTooLarge { pixels, limit } => ErrorKind::ImageTooLarge { pixels, limit },
            EK::Cancelled => ErrorKind::Cancelled,
            EK::UnsupportedPixelFormat { format } => ErrorKind::UnsupportedPixelFormat { format },
            EK::InvalidConfig(reason) => ErrorKind::InvalidConfig(reason),
            EK::StrideTooSmall { width, stride } => ErrorKind::StrideTooSmall { width, stride },
            EK::TooManyRows { height, pushed } => ErrorKind::TooManyRows { height, pushed },
            EK::IncompleteImage { height, pushed } => ErrorKind::IncompleteImage { height, pushed },
            // Decoder-specific errors should not occur in encoder - convert to internal error
            EK::InvalidJpegData { reason } | EK::TruncatedData { context: reason } => {
                ErrorKind::InternalError { reason }
            }
            EK::InvalidMarker { marker, context } => ErrorKind::InternalError {
                reason: if marker == 0 {
                    context
                } else {
                    "invalid marker"
                },
            },
            EK::InvalidHuffmanTable { reason, .. } | EK::InvalidQuantTable { reason, .. } => {
                ErrorKind::InternalError { reason }
            }
            EK::DecodeError(reason) => ErrorKind::InternalError {
                reason: if reason.is_empty() {
                    "decode error"
                } else {
                    "unexpected decode error"
                },
            },
            EK::TooManyScans { .. } => ErrorKind::InternalError {
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
