//! Encoder error types.

use alloc::string::String;
use core::fmt;

/// Result type for encoder operations.
pub type Result<T> = core::result::Result<T, Error>;

/// Errors that can occur during JPEG encoding.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Error {
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

impl fmt::Display for Error {
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
    fn from(err: std::io::Error) -> Self {
        Self::IoError {
            reason: err.to_string(),
        }
    }
}

impl From<enough::StopReason> for Error {
    fn from(_: enough::StopReason) -> Self {
        Self::Cancelled
    }
}

/// Convert from internal error type.
impl From<crate::error::Error> for Error {
    fn from(err: crate::error::Error) -> Self {
        use crate::error::Error as E;
        match err {
            E::InvalidDimensions {
                width,
                height,
                reason,
            } => Self::InvalidDimensions {
                width,
                height,
                reason,
            },
            E::InvalidQuality { value, valid_range } => Self::InvalidQuality { value, valid_range },
            E::InvalidColorFormat { reason } => Self::InvalidColorFormat { reason },
            E::InvalidBufferSize { expected, actual } => {
                Self::InvalidBufferSize { expected, actual }
            }
            E::UnsupportedFeature { feature } => Self::UnsupportedFeature { feature },
            E::InternalError { reason } => Self::InternalError { reason },
            E::IoError { reason } => Self::IoError { reason },
            E::IccError(reason) => Self::IccError(reason),
            E::InvalidScanScript(reason) => Self::InvalidScanScript(reason),
            E::AllocationFailed { bytes, context } => Self::AllocationFailed { bytes, context },
            E::SizeOverflow { context } => Self::SizeOverflow { context },
            E::ImageTooLarge { pixels, limit } => Self::ImageTooLarge { pixels, limit },
            E::Cancelled => Self::Cancelled,
            E::UnsupportedPixelFormat { format } => Self::UnsupportedPixelFormat { format },
            E::InvalidConfig(reason) => Self::InvalidConfig(reason),
            E::StrideTooSmall { width, stride } => Self::StrideTooSmall { width, stride },
            E::TooManyRows { height, pushed } => Self::TooManyRows { height, pushed },
            E::IncompleteImage { height, pushed } => Self::IncompleteImage { height, pushed },
            // Decoder-specific errors should not occur in encoder - convert to internal error
            E::InvalidJpegData { reason } | E::TruncatedData { context: reason } => {
                Self::InternalError { reason }
            }
            E::EndOfScanData => Self::InternalError {
                reason: "unexpected end of scan data",
            },
            E::InvalidMarker { marker, context } => Self::InternalError {
                reason: if marker == 0 {
                    context
                } else {
                    "invalid marker"
                },
            },
            E::InvalidHuffmanTable { reason, .. } | E::InvalidQuantTable { reason, .. } => {
                Self::InternalError { reason }
            }
            E::DecodeError(reason) => Self::InternalError {
                reason: if reason.is_empty() {
                    "decode error"
                } else {
                    "unexpected decode error"
                },
            },
            E::TooManyScans { .. } => Self::InternalError {
                reason: "too many scans",
            },
        }
    }
}
