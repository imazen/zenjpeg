//! Decoder error types.

use alloc::string::String;
use core::fmt;
use whereat::{AtTrace, AtTraceBoxed, AtTraceable};

/// Result type for decoder operations.
pub type Result<T> = core::result::Result<T, Error>;

/// Errors that can occur during JPEG decoding.
///
/// Use [`Error::kind()`] to match on the specific error variant.
#[derive(Debug)]
pub struct Error {
    kind: ErrorKind,
    trace: AtTraceBoxed,
}

/// The specific kind of decoder error.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ErrorKind {
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
    pub fn invalid_color_format(reason: &'static str) -> Self {
        Self::new(ErrorKind::InvalidColorFormat { reason })
    }

    #[track_caller]
    pub fn invalid_buffer_size(expected: usize, actual: usize) -> Self {
        Self::new(ErrorKind::InvalidBufferSize { expected, actual })
    }

    #[track_caller]
    pub fn invalid_jpeg_data(reason: &'static str) -> Self {
        Self::new(ErrorKind::InvalidJpegData { reason })
    }

    #[track_caller]
    pub fn truncated_data(context: &'static str) -> Self {
        Self::new(ErrorKind::TruncatedData { context })
    }

    #[track_caller]
    pub fn invalid_marker(marker: u8, context: &'static str) -> Self {
        Self::new(ErrorKind::InvalidMarker { marker, context })
    }

    #[track_caller]
    pub fn invalid_huffman_table(table_idx: u8, reason: &'static str) -> Self {
        Self::new(ErrorKind::InvalidHuffmanTable { table_idx, reason })
    }

    #[track_caller]
    pub fn invalid_quant_table(table_idx: u8, reason: &'static str) -> Self {
        Self::new(ErrorKind::InvalidQuantTable { table_idx, reason })
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
    pub fn too_many_scans(count: usize, limit: usize) -> Self {
        Self::new(ErrorKind::TooManyScans { count, limit })
    }

    #[track_caller]
    pub fn cancelled() -> Self {
        Self::new(ErrorKind::Cancelled)
    }

    #[track_caller]
    pub fn unsupported_pixel_format(format: crate::types::PixelFormat) -> Self {
        Self::new(ErrorKind::UnsupportedPixelFormat { format })
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
            EK::InvalidColorFormat { reason } => ErrorKind::InvalidColorFormat { reason },
            EK::InvalidBufferSize { expected, actual } => {
                ErrorKind::InvalidBufferSize { expected, actual }
            }
            EK::InvalidJpegData { reason } => ErrorKind::InvalidJpegData { reason },
            EK::TruncatedData { context } => ErrorKind::TruncatedData { context },
            EK::InvalidMarker { marker, context } => ErrorKind::InvalidMarker { marker, context },
            EK::InvalidHuffmanTable { table_idx, reason } => {
                ErrorKind::InvalidHuffmanTable { table_idx, reason }
            }
            EK::InvalidQuantTable { table_idx, reason } => {
                ErrorKind::InvalidQuantTable { table_idx, reason }
            }
            EK::UnsupportedFeature { feature } => ErrorKind::UnsupportedFeature { feature },
            EK::InternalError { reason } => ErrorKind::InternalError { reason },
            EK::IoError { reason } => ErrorKind::IoError { reason },
            EK::IccError(reason) => ErrorKind::IccError(reason),
            EK::DecodeError(reason) => ErrorKind::InvalidJpegData {
                reason: if reason.is_empty() {
                    "decode error"
                } else {
                    "decoding failed"
                },
            },
            EK::AllocationFailed { bytes, context } => {
                ErrorKind::AllocationFailed { bytes, context }
            }
            EK::SizeOverflow { context } => ErrorKind::SizeOverflow { context },
            EK::ImageTooLarge { pixels, limit } => ErrorKind::ImageTooLarge { pixels, limit },
            EK::TooManyScans { count, limit } => ErrorKind::TooManyScans { count, limit },
            EK::Cancelled => ErrorKind::Cancelled,
            EK::UnsupportedPixelFormat { format } => ErrorKind::UnsupportedPixelFormat { format },
            // Encoder-specific errors should not occur in decoder
            EK::InvalidQuality { .. } => ErrorKind::InternalError {
                reason: "invalid quality (encoder error)",
            },
            EK::InvalidScanScript(_) => ErrorKind::InternalError {
                reason: "invalid scan script (encoder error)",
            },
            EK::InvalidConfig(_) => ErrorKind::InternalError {
                reason: "invalid config (encoder error)",
            },
            EK::StrideTooSmall { .. } => ErrorKind::InternalError {
                reason: "stride too small (encoder error)",
            },
            EK::TooManyRows { .. } => ErrorKind::InternalError {
                reason: "too many rows (encoder error)",
            },
            EK::IncompleteImage { .. } => ErrorKind::InternalError {
                reason: "incomplete image (encoder error)",
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
