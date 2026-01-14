//! Decoder error types.

use alloc::string::String;
use core::fmt;

/// Result type for decoder operations.
pub type Result<T> = core::result::Result<T, Error>;

/// Errors that can occur during JPEG decoding.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Error {
    /// Invalid input dimensions.
    InvalidDimensions {
        width: u32,
        height: u32,
        reason: &'static str,
    },
    /// Invalid color space or pixel format combination.
    InvalidColorFormat { reason: &'static str },
    /// Output buffer has wrong size.
    InvalidBufferSize { expected: usize, actual: usize },
    /// Invalid JPEG data (corrupted or not a JPEG).
    InvalidJpegData { reason: &'static str },
    /// Input data is truncated or corrupted.
    TruncatedData { context: &'static str },
    /// Invalid marker in JPEG stream.
    InvalidMarker { marker: u8, context: &'static str },
    /// Invalid Huffman table.
    InvalidHuffmanTable { table_idx: u8, reason: &'static str },
    /// Invalid quantization table.
    InvalidQuantTable { table_idx: u8, reason: &'static str },
    /// Unsupported JPEG feature.
    UnsupportedFeature { feature: &'static str },
    /// Internal error (should not happen in correct usage).
    InternalError { reason: &'static str },
    /// I/O error during decoding.
    IoError { reason: String },
    /// ICC color management error.
    IccError(String),
    /// Memory allocation failed.
    AllocationFailed { bytes: usize, context: &'static str },
    /// Size calculation overflowed.
    SizeOverflow { context: &'static str },
    /// Image exceeds maximum pixel limit.
    ImageTooLarge { pixels: u64, limit: u64 },
    /// Too many progressive scans (DoS protection).
    TooManyScans { count: usize, limit: usize },
    /// Operation was cancelled.
    Cancelled,
    /// Pixel format not supported.
    UnsupportedPixelFormat { format: crate::types::PixelFormat },
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
            Self::InvalidJpegData { reason } => {
                write!(f, "invalid JPEG data: {}", reason)
            }
            Self::TruncatedData { context } => {
                write!(f, "truncated data while {}", context)
            }
            Self::InvalidMarker { marker, context } => {
                write!(f, "invalid marker 0x{:02X} while {}", marker, context)
            }
            Self::InvalidHuffmanTable { table_idx, reason } => {
                write!(f, "invalid Huffman table {}: {}", table_idx, reason)
            }
            Self::InvalidQuantTable { table_idx, reason } => {
                write!(f, "invalid quantization table {}: {}", table_idx, reason)
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
            Self::TooManyScans { count, limit } => {
                write!(f, "too many scans: {} exceeds limit of {}", count, limit)
            }
            Self::Cancelled => write!(f, "operation cancelled"),
            Self::UnsupportedPixelFormat { format } => {
                write!(f, "pixel format {:?} not supported", format)
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
            E::InvalidColorFormat { reason } => Self::InvalidColorFormat { reason },
            E::InvalidBufferSize { expected, actual } => {
                Self::InvalidBufferSize { expected, actual }
            }
            E::InvalidJpegData { reason } => Self::InvalidJpegData { reason },
            E::TruncatedData { context } => Self::TruncatedData { context },
            E::InvalidMarker { marker, context } => Self::InvalidMarker { marker, context },
            E::InvalidHuffmanTable { table_idx, reason } => {
                Self::InvalidHuffmanTable { table_idx, reason }
            }
            E::InvalidQuantTable { table_idx, reason } => {
                Self::InvalidQuantTable { table_idx, reason }
            }
            E::UnsupportedFeature { feature } => Self::UnsupportedFeature { feature },
            E::InternalError { reason } => Self::InternalError { reason },
            E::IoError { reason } => Self::IoError { reason },
            E::IccError(reason) => Self::IccError(reason),
            E::DecodeError(reason) => Self::InvalidJpegData {
                reason: if reason.is_empty() {
                    "decode error"
                } else {
                    "decoding failed"
                },
            },
            E::AllocationFailed { bytes, context } => Self::AllocationFailed { bytes, context },
            E::SizeOverflow { context } => Self::SizeOverflow { context },
            E::ImageTooLarge { pixels, limit } => Self::ImageTooLarge { pixels, limit },
            E::TooManyScans { count, limit } => Self::TooManyScans { count, limit },
            E::Cancelled => Self::Cancelled,
            E::UnsupportedPixelFormat { format } => Self::UnsupportedPixelFormat { format },
            // EndOfScanData is expected during progressive decoding, not a real error
            E::EndOfScanData => Self::InternalError {
                reason: "unexpected end of scan data",
            },
            // Encoder-specific errors should not occur in decoder
            E::InvalidQuality { .. } => Self::InternalError {
                reason: "invalid quality (encoder error)",
            },
            E::InvalidScanScript(_) => Self::InternalError {
                reason: "invalid scan script (encoder error)",
            },
            E::InvalidConfig(_) => Self::InternalError {
                reason: "invalid config (encoder error)",
            },
            E::StrideTooSmall { .. } => Self::InternalError {
                reason: "stride too small (encoder error)",
            },
            E::TooManyRows { .. } => Self::InternalError {
                reason: "too many rows (encoder error)",
            },
            E::IncompleteImage { .. } => Self::InternalError {
                reason: "incomplete image (encoder error)",
            },
        }
    }
}
