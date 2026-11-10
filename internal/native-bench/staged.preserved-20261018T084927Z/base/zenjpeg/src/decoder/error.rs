//! Decoder error types.
//!
//! The decoder uses a hierarchical error structure:
//! - [`ArgumentError`](crate::error::ArgumentError) - Invalid arguments
//! - [`ResourceError`](crate::error::ResourceError) - Memory/IO failures
//! - [`DatastreamError`] - Malformed JPEG data

use alloc::string::String;
use core::fmt;
use thiserror::Error;
use whereat::{AtTrace, AtTraceBoxed, AtTraceable};

// Re-export shared error types
pub use crate::error::{ArgumentError, ResourceError};

/// Result type for decoder operations.
pub type Result<T> = core::result::Result<T, Error>;

// ============================================================================
// Decoder-specific: Datastream errors
// ============================================================================

/// Errors caused by malformed or corrupted JPEG data.
///
/// These indicate the input JPEG is invalid, not a bug in calling code.
#[derive(Debug, Clone, PartialEq, Error)]
#[non_exhaustive]
pub enum DatastreamError {
    /// Invalid JPEG data (corrupted or not a JPEG).
    #[error("invalid JPEG data: {reason}")]
    InvalidJpegData { reason: &'static str },

    /// Input data is truncated or corrupted.
    #[error("truncated data while {context}")]
    TruncatedData { context: &'static str },

    /// Invalid marker in JPEG stream.
    #[error("invalid marker 0x{marker:02X} while {context}")]
    InvalidMarker { marker: u8, context: &'static str },

    /// Invalid Huffman table.
    #[error("invalid Huffman table {table_idx}: {reason}")]
    InvalidHuffmanTable { table_idx: u8, reason: &'static str },

    /// Invalid quantization table.
    #[error("invalid quantization table {table_idx}: {reason}")]
    InvalidQuantTable { table_idx: u8, reason: &'static str },

    /// Too many progressive scans (DoS protection).
    #[error("too many scans: {count} exceeds limit of {limit}")]
    TooManyScans { count: usize, limit: usize },
}

// ============================================================================
// Decoder ErrorKind - Composed from shared + decoder-specific
// ============================================================================

/// The specific kind of decoder error.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ErrorKind {
    /// Invalid argument from caller.
    Argument(ArgumentError),

    /// Resource exhaustion or I/O failure.
    Resource(ResourceError),

    /// Malformed JPEG datastream.
    Datastream(DatastreamError),

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
            Self::Resource(e) => write!(f, "{}", e),
            Self::Datastream(e) => write!(f, "{}", e),
            Self::Icc(reason) => write!(f, "ICC error: {}", reason),
            Self::Internal { reason } => write!(f, "internal error: {}", reason),
            Self::Cancelled => write!(f, "operation cancelled"),
        }
    }
}

// ============================================================================
// Decoder Error - Main error type with location tracking
// ============================================================================

/// Errors that can occur during JPEG decoding.
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
    // Convenience constructors - Argument errors
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
    // Convenience constructors - Datastream errors
    // ========================================================================

    #[track_caller]
    pub fn invalid_jpeg_data(reason: &'static str) -> Self {
        Self::new(ErrorKind::Datastream(DatastreamError::InvalidJpegData {
            reason,
        }))
    }

    #[track_caller]
    pub fn truncated_data(context: &'static str) -> Self {
        Self::new(ErrorKind::Datastream(DatastreamError::TruncatedData {
            context,
        }))
    }

    #[track_caller]
    pub fn invalid_marker(marker: u8, context: &'static str) -> Self {
        Self::new(ErrorKind::Datastream(DatastreamError::InvalidMarker {
            marker,
            context,
        }))
    }

    #[track_caller]
    pub fn invalid_huffman_table(table_idx: u8, reason: &'static str) -> Self {
        Self::new(ErrorKind::Datastream(DatastreamError::InvalidHuffmanTable {
            table_idx,
            reason,
        }))
    }

    #[track_caller]
    pub fn invalid_quant_table(table_idx: u8, reason: &'static str) -> Self {
        Self::new(ErrorKind::Datastream(DatastreamError::InvalidQuantTable {
            table_idx,
            reason,
        }))
    }

    #[track_caller]
    pub fn too_many_scans(count: usize, limit: usize) -> Self {
        Self::new(ErrorKind::Datastream(DatastreamError::TooManyScans {
            count,
            limit,
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
            // Argument errors
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

            // Datastream errors
            EK::InvalidJpegData { reason } => {
                ErrorKind::Datastream(DatastreamError::InvalidJpegData { reason })
            }
            EK::TruncatedData { context } => {
                ErrorKind::Datastream(DatastreamError::TruncatedData { context })
            }
            EK::InvalidMarker { marker, context } => {
                ErrorKind::Datastream(DatastreamError::InvalidMarker { marker, context })
            }
            EK::InvalidHuffmanTable { table_idx, reason } => {
                ErrorKind::Datastream(DatastreamError::InvalidHuffmanTable { table_idx, reason })
            }
            EK::InvalidQuantTable { table_idx, reason } => {
                ErrorKind::Datastream(DatastreamError::InvalidQuantTable { table_idx, reason })
            }
            EK::TooManyScans { count, limit } => {
                ErrorKind::Datastream(DatastreamError::TooManyScans { count, limit })
            }
            EK::DecodeError(reason) => ErrorKind::Datastream(DatastreamError::InvalidJpegData {
                reason: if reason.is_empty() {
                    "decode error"
                } else {
                    "decoding failed"
                },
            }),

            // Other shared
            EK::IccError(reason) => ErrorKind::Icc(reason),
            EK::InternalError { reason } => ErrorKind::Internal { reason },
            EK::Cancelled => ErrorKind::Cancelled,

            // Encoder-specific errors should not occur in decoder
            EK::InvalidQuality { .. } => ErrorKind::Internal {
                reason: "invalid quality (encoder error)",
            },
            EK::InvalidScanScript(_) => ErrorKind::Internal {
                reason: "invalid scan script (encoder error)",
            },
            EK::InvalidConfig(_) => ErrorKind::Internal {
                reason: "invalid config (encoder error)",
            },
            EK::StrideTooSmall { .. } => ErrorKind::Internal {
                reason: "stride too small (encoder error)",
            },
            EK::TooManyRows { .. } => ErrorKind::Internal {
                reason: "too many rows (encoder error)",
            },
            EK::IncompleteImage { .. } => ErrorKind::Internal {
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
