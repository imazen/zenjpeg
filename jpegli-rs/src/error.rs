//! Error types for jpegli.
//!
//! Errors are organized hierarchically:
//! - [`ArgumentError`] - Invalid arguments from the user
//! - [`ResourceError`] - Memory/IO failures
//! - Decoder-specific errors in [`crate::decoder::error`]
//! - Encoder-specific errors in [`crate::encoder::error`]

use alloc::string::String;
use core::fmt;
use thiserror::Error;
use whereat::{AtTrace, AtTraceBoxed, AtTraceable};

/// Result type for jpegli operations.
pub type Result<T> = core::result::Result<T, Error>;

// ============================================================================
// ScanRead - Control flow for entropy-coded scan reading
// ============================================================================

/// Result of reading from an entropy-coded scan.
///
/// This distinguishes between successful reads, normal end-of-scan conditions,
/// and truncated data. End-of-scan is not an error - it's the expected signal
/// that a marker was encountered and the current scan is complete.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanRead<T> {
    /// Successfully read the value.
    Value(T),
    /// Reached end of entropy-coded segment (marker encountered).
    /// This is normal during progressive JPEG decoding between scans.
    EndOfScan,
    /// Data was truncated (end of input without finding a marker).
    /// Caller can choose to treat this as an error or attempt partial decode.
    Truncated,
}

impl<T> ScanRead<T> {
    /// Returns the value if `Value`, otherwise returns the provided default.
    #[inline]
    pub fn unwrap_or(self, default: T) -> T {
        match self {
            Self::Value(v) => v,
            Self::EndOfScan | Self::Truncated => default,
        }
    }

    /// Returns the value if `Value`, otherwise computes it from a closure.
    #[inline]
    pub fn unwrap_or_else<F: FnOnce() -> T>(self, f: F) -> T {
        match self {
            Self::Value(v) => v,
            Self::EndOfScan | Self::Truncated => f(),
        }
    }

    /// Returns `true` if this is `EndOfScan`.
    #[inline]
    pub fn is_end_of_scan(&self) -> bool {
        matches!(self, Self::EndOfScan)
    }

    /// Returns `true` if this is `Truncated`.
    #[inline]
    pub fn is_truncated(&self) -> bool {
        matches!(self, Self::Truncated)
    }

    /// Returns `true` if this is `Value`.
    #[inline]
    pub fn is_value(&self) -> bool {
        matches!(self, Self::Value(_))
    }

    /// Maps the value if `Value`, passes through `EndOfScan` and `Truncated`.
    #[inline]
    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> ScanRead<U> {
        match self {
            Self::Value(v) => ScanRead::Value(f(v)),
            Self::EndOfScan => ScanRead::EndOfScan,
            Self::Truncated => ScanRead::Truncated,
        }
    }
}

/// Result type for entropy-coded scan reads.
///
/// - `Ok(ScanRead::Value(v))` - Successfully read a value
/// - `Ok(ScanRead::EndOfScan)` - Normal end of scan (marker found)
/// - `Ok(ScanRead::Truncated)` - Data ended without marker (caller decides how to handle)
/// - `Err(e)` - Actual error (corruption, internal error, etc.)
pub type ScanResult<T> = Result<ScanRead<T>>;

// ============================================================================
// Shared Error Types - Used by both encoder and decoder
// ============================================================================

/// Errors caused by invalid arguments from the caller.
///
/// These indicate bugs in the calling code, not runtime failures.
#[derive(Debug, Clone, PartialEq, Error)]
#[non_exhaustive]
pub enum ArgumentError {
    /// Invalid image dimensions (zero or exceeds limits).
    #[error("invalid dimensions {width}x{height}: {reason}")]
    InvalidDimensions {
        width: u32,
        height: u32,
        reason: &'static str,
    },

    /// Invalid color space or pixel format combination.
    #[error("invalid color format: {reason}")]
    InvalidColorFormat { reason: &'static str },

    /// Buffer size doesn't match expected size.
    #[error("invalid buffer size: expected {expected} bytes, got {actual}")]
    InvalidBufferSize { expected: usize, actual: usize },

    /// Feature not supported by this codec.
    #[error("unsupported feature: {feature}")]
    UnsupportedFeature { feature: &'static str },

    /// Pixel format not supported for this operation.
    #[error("pixel format {format:?} not supported")]
    UnsupportedPixelFormat { format: crate::types::PixelFormat },
}

/// Errors caused by resource exhaustion or I/O failures.
///
/// These are runtime failures, not bugs in calling code.
#[derive(Debug, Clone, PartialEq, Error)]
#[non_exhaustive]
pub enum ResourceError {
    /// Memory allocation failed.
    #[error("allocation of {bytes} bytes failed while {context}")]
    AllocationFailed { bytes: usize, context: &'static str },

    /// Size calculation overflowed.
    #[error("size calculation overflow while {context}")]
    SizeOverflow { context: &'static str },

    /// Image exceeds maximum pixel limit.
    #[error("image too large: {pixels} pixels exceeds limit of {limit}")]
    ImageTooLarge { pixels: u64, limit: u64 },

    /// I/O operation failed.
    #[error("I/O error: {reason}")]
    IoError { reason: String },
}

// ============================================================================
// Internal ErrorKind - Flat enum for internal use
// ============================================================================

/// The specific kind of error that occurred (internal flat enum).
///
/// This is used internally and by the `From` implementations.
/// Public APIs use [`decoder::ErrorKind`](crate::decoder::ErrorKind) or
/// [`encoder::ErrorKind`](crate::encoder::ErrorKind).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ErrorKind {
    // === Shared: Argument errors ===
    /// Invalid input dimensions (zero or too large).
    InvalidDimensions {
        width: u32,
        height: u32,
        reason: &'static str,
    },
    /// Invalid color space or pixel format combination.
    InvalidColorFormat { reason: &'static str },
    /// Input buffer has wrong size.
    InvalidBufferSize { expected: usize, actual: usize },
    /// Unsupported JPEG feature.
    UnsupportedFeature { feature: &'static str },
    /// Pixel format not yet supported for this operation.
    UnsupportedPixelFormat { format: crate::types::PixelFormat },

    // === Shared: Resource errors ===
    /// Memory allocation failed (OOM or limit exceeded).
    AllocationFailed { bytes: usize, context: &'static str },
    /// Size calculation overflowed.
    SizeOverflow { context: &'static str },
    /// Image exceeds maximum pixel limit.
    ImageTooLarge { pixels: u64, limit: u64 },
    /// I/O error during encoding/decoding.
    IoError { reason: String },

    // === Shared: Other ===
    /// ICC color management error.
    IccError(String),
    /// Internal error (should not happen in correct usage).
    InternalError { reason: &'static str },
    /// Operation was cancelled via Stop trait.
    Cancelled,

    // === Decoder-specific: Datastream errors ===
    /// Invalid JPEG data (corrupted or not a JPEG).
    InvalidJpegData { reason: &'static str },
    /// Input data is truncated or corrupted.
    TruncatedData { context: &'static str },
    /// Invalid marker or segment in JPEG stream.
    InvalidMarker { marker: u8, context: &'static str },
    /// Invalid Huffman table.
    InvalidHuffmanTable { table_idx: u8, reason: &'static str },
    /// Invalid quantization table.
    InvalidQuantTable { table_idx: u8, reason: &'static str },
    /// Too many progressive scans.
    TooManyScans { count: usize, limit: usize },
    /// Decode error from JPEG decoder.
    DecodeError(String),

    // === Encoder-specific: Argument errors ===
    /// Invalid quality parameter.
    InvalidQuality { value: f32, valid_range: &'static str },
    /// Invalid scan script for progressive encoding.
    InvalidScanScript(String),
    /// Invalid encoder configuration.
    InvalidConfig(String),
    /// Stride too small for image width.
    StrideTooSmall { width: u32, stride: usize },

    // === Encoder-specific: State errors ===
    /// Pushed more rows than image height.
    TooManyRows { height: u32, pushed: u32 },
    /// Encoding finished without all rows pushed.
    IncompleteImage { height: u32, pushed: u32 },
}

impl ErrorKind {
    /// Convert to ArgumentError if this is an argument error variant.
    pub fn as_argument_error(&self) -> Option<ArgumentError> {
        match self {
            Self::InvalidDimensions {
                width,
                height,
                reason,
            } => Some(ArgumentError::InvalidDimensions {
                width: *width,
                height: *height,
                reason,
            }),
            Self::InvalidColorFormat { reason } => {
                Some(ArgumentError::InvalidColorFormat { reason })
            }
            Self::InvalidBufferSize { expected, actual } => {
                Some(ArgumentError::InvalidBufferSize {
                    expected: *expected,
                    actual: *actual,
                })
            }
            Self::UnsupportedFeature { feature } => {
                Some(ArgumentError::UnsupportedFeature { feature })
            }
            Self::UnsupportedPixelFormat { format } => {
                Some(ArgumentError::UnsupportedPixelFormat { format: *format })
            }
            _ => None,
        }
    }

    /// Convert to ResourceError if this is a resource error variant.
    pub fn as_resource_error(&self) -> Option<ResourceError> {
        match self {
            Self::AllocationFailed { bytes, context } => Some(ResourceError::AllocationFailed {
                bytes: *bytes,
                context,
            }),
            Self::SizeOverflow { context } => Some(ResourceError::SizeOverflow { context }),
            Self::ImageTooLarge { pixels, limit } => Some(ResourceError::ImageTooLarge {
                pixels: *pixels,
                limit: *limit,
            }),
            Self::IoError { reason } => Some(ResourceError::IoError {
                reason: reason.clone(),
            }),
            _ => None,
        }
    }
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            // Argument errors
            Self::InvalidDimensions {
                width,
                height,
                reason,
            } => write!(f, "invalid dimensions {}x{}: {}", width, height, reason),
            Self::InvalidColorFormat { reason } => write!(f, "invalid color format: {}", reason),
            Self::InvalidBufferSize { expected, actual } => {
                write!(
                    f,
                    "invalid buffer size: expected {} bytes, got {}",
                    expected, actual
                )
            }
            Self::UnsupportedFeature { feature } => write!(f, "unsupported feature: {}", feature),
            Self::UnsupportedPixelFormat { format } => {
                write!(f, "pixel format {:?} not supported", format)
            }

            // Resource errors
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
            Self::IoError { reason } => write!(f, "I/O error: {}", reason),

            // Other shared
            Self::IccError(reason) => write!(f, "ICC error: {}", reason),
            Self::InternalError { reason } => write!(f, "internal error: {}", reason),
            Self::Cancelled => write!(f, "operation cancelled"),

            // Decoder-specific
            Self::InvalidJpegData { reason } => write!(f, "invalid JPEG data: {}", reason),
            Self::TruncatedData { context } => write!(f, "truncated data while {}", context),
            Self::InvalidMarker { marker, context } => {
                write!(f, "invalid marker 0x{:02X} while {}", marker, context)
            }
            Self::InvalidHuffmanTable { table_idx, reason } => {
                write!(f, "invalid Huffman table {}: {}", table_idx, reason)
            }
            Self::InvalidQuantTable { table_idx, reason } => {
                write!(f, "invalid quantization table {}: {}", table_idx, reason)
            }
            Self::TooManyScans { count, limit } => {
                write!(f, "too many scans: {} exceeds limit of {}", count, limit)
            }
            Self::DecodeError(reason) => write!(f, "decode error: {}", reason),

            // Encoder-specific
            Self::InvalidQuality { value, valid_range } => {
                write!(f, "invalid quality {}: must be in {}", value, valid_range)
            }
            Self::InvalidScanScript(reason) => write!(f, "invalid scan script: {}", reason),
            Self::InvalidConfig(reason) => write!(f, "invalid encoder configuration: {}", reason),
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

// ============================================================================
// From implementations for ErrorKind
// ============================================================================

impl From<ArgumentError> for ErrorKind {
    fn from(err: ArgumentError) -> Self {
        match err {
            ArgumentError::InvalidDimensions {
                width,
                height,
                reason,
            } => Self::InvalidDimensions {
                width,
                height,
                reason,
            },
            ArgumentError::InvalidColorFormat { reason } => Self::InvalidColorFormat { reason },
            ArgumentError::InvalidBufferSize { expected, actual } => {
                Self::InvalidBufferSize { expected, actual }
            }
            ArgumentError::UnsupportedFeature { feature } => Self::UnsupportedFeature { feature },
            ArgumentError::UnsupportedPixelFormat { format } => {
                Self::UnsupportedPixelFormat { format }
            }
        }
    }
}

impl From<ResourceError> for ErrorKind {
    fn from(err: ResourceError) -> Self {
        match err {
            ResourceError::AllocationFailed { bytes, context } => {
                Self::AllocationFailed { bytes, context }
            }
            ResourceError::SizeOverflow { context } => Self::SizeOverflow { context },
            ResourceError::ImageTooLarge { pixels, limit } => {
                Self::ImageTooLarge { pixels, limit }
            }
            ResourceError::IoError { reason } => Self::IoError { reason },
        }
    }
}

// ============================================================================
// Error - Main error type with location tracking
// ============================================================================

/// Errors that can occur during JPEG encoding/decoding.
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

    /// Create a new error without capturing a trace (for hot paths).
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

    /// Create an invalid dimensions error.
    #[track_caller]
    pub fn invalid_dimensions(width: u32, height: u32, reason: &'static str) -> Self {
        Self::new(ErrorKind::InvalidDimensions {
            width,
            height,
            reason,
        })
    }

    /// Create an invalid color format error.
    #[track_caller]
    pub fn invalid_color_format(reason: &'static str) -> Self {
        Self::new(ErrorKind::InvalidColorFormat { reason })
    }

    /// Create an invalid buffer size error.
    #[track_caller]
    pub fn invalid_buffer_size(expected: usize, actual: usize) -> Self {
        Self::new(ErrorKind::InvalidBufferSize { expected, actual })
    }

    /// Create an unsupported feature error.
    #[track_caller]
    pub fn unsupported_feature(feature: &'static str) -> Self {
        Self::new(ErrorKind::UnsupportedFeature { feature })
    }

    /// Create an unsupported pixel format error.
    #[track_caller]
    pub fn unsupported_pixel_format(format: crate::types::PixelFormat) -> Self {
        Self::new(ErrorKind::UnsupportedPixelFormat { format })
    }

    // ========================================================================
    // Convenience constructors - Resource errors
    // ========================================================================

    /// Create an allocation failed error.
    #[track_caller]
    pub fn allocation_failed(bytes: usize, context: &'static str) -> Self {
        Self::new(ErrorKind::AllocationFailed { bytes, context })
    }

    /// Create a size overflow error.
    #[track_caller]
    pub fn size_overflow(context: &'static str) -> Self {
        Self::new(ErrorKind::SizeOverflow { context })
    }

    /// Create an image too large error.
    #[track_caller]
    pub fn image_too_large(pixels: u64, limit: u64) -> Self {
        Self::new(ErrorKind::ImageTooLarge { pixels, limit })
    }

    /// Create an I/O error.
    #[track_caller]
    pub fn io_error(reason: String) -> Self {
        Self::new(ErrorKind::IoError { reason })
    }

    // ========================================================================
    // Convenience constructors - Other shared errors
    // ========================================================================

    /// Create an ICC error.
    #[track_caller]
    pub fn icc_error(reason: String) -> Self {
        Self::new(ErrorKind::IccError(reason))
    }

    /// Create an internal error.
    #[track_caller]
    pub fn internal(reason: &'static str) -> Self {
        Self::new(ErrorKind::InternalError { reason })
    }

    /// Create a cancelled error.
    #[track_caller]
    pub fn cancelled() -> Self {
        Self::new(ErrorKind::Cancelled)
    }

    // ========================================================================
    // Convenience constructors - Decoder-specific errors
    // ========================================================================

    /// Create an invalid JPEG data error.
    #[track_caller]
    pub fn invalid_jpeg_data(reason: &'static str) -> Self {
        Self::new(ErrorKind::InvalidJpegData { reason })
    }

    /// Create a truncated data error.
    #[track_caller]
    pub fn truncated_data(context: &'static str) -> Self {
        Self::new(ErrorKind::TruncatedData { context })
    }

    /// Create an invalid marker error.
    #[track_caller]
    pub fn invalid_marker(marker: u8, context: &'static str) -> Self {
        Self::new(ErrorKind::InvalidMarker { marker, context })
    }

    /// Create an invalid Huffman table error.
    #[track_caller]
    pub fn invalid_huffman_table(table_idx: u8, reason: &'static str) -> Self {
        Self::new(ErrorKind::InvalidHuffmanTable { table_idx, reason })
    }

    /// Create an invalid quantization table error.
    #[track_caller]
    pub fn invalid_quant_table(table_idx: u8, reason: &'static str) -> Self {
        Self::new(ErrorKind::InvalidQuantTable { table_idx, reason })
    }

    /// Create a too many scans error.
    #[track_caller]
    pub fn too_many_scans(count: usize, limit: usize) -> Self {
        Self::new(ErrorKind::TooManyScans { count, limit })
    }

    /// Create a decode error.
    #[track_caller]
    pub fn decode_error(reason: String) -> Self {
        Self::new(ErrorKind::DecodeError(reason))
    }

    // ========================================================================
    // Convenience constructors - Encoder-specific errors
    // ========================================================================

    /// Create an invalid quality error.
    #[track_caller]
    pub fn invalid_quality(value: f32, valid_range: &'static str) -> Self {
        Self::new(ErrorKind::InvalidQuality { value, valid_range })
    }

    /// Create an invalid scan script error.
    #[track_caller]
    pub fn invalid_scan_script(reason: String) -> Self {
        Self::new(ErrorKind::InvalidScanScript(reason))
    }

    /// Create an invalid config error.
    #[track_caller]
    pub fn invalid_config(reason: String) -> Self {
        Self::new(ErrorKind::InvalidConfig(reason))
    }

    /// Create a stride too small error.
    #[track_caller]
    pub fn stride_too_small(width: u32, stride: usize) -> Self {
        Self::new(ErrorKind::StrideTooSmall { width, stride })
    }

    /// Create a too many rows error.
    #[track_caller]
    pub fn too_many_rows(height: u32, pushed: u32) -> Self {
        Self::new(ErrorKind::TooManyRows { height, pushed })
    }

    /// Create an incomplete image error.
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

// ============================================================================
// From implementations for Error
// ============================================================================

impl From<ArgumentError> for Error {
    #[track_caller]
    fn from(err: ArgumentError) -> Self {
        Self::new(err.into())
    }
}

impl From<ResourceError> for Error {
    #[track_caller]
    fn from(err: ResourceError) -> Self {
        Self::new(err.into())
    }
}

impl From<enough::StopReason> for Error {
    #[track_caller]
    fn from(_: enough::StopReason) -> Self {
        Self::cancelled()
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

impl From<crate::foundation::aligned_alloc::AllocError> for Error {
    #[track_caller]
    fn from(err: crate::foundation::aligned_alloc::AllocError) -> Self {
        match err {
            crate::foundation::aligned_alloc::AllocError::OutOfMemory => {
                Self::allocation_failed(0, "adaptive quantization")
            }
            crate::foundation::aligned_alloc::AllocError::Overflow => {
                Self::size_overflow("adaptive quantization size calculation")
            }
        }
    }
}

// ============================================================================
// Clone and PartialEq for Error
// ============================================================================

impl Clone for Error {
    fn clone(&self) -> Self {
        Self {
            kind: self.kind.clone(),
            trace: AtTraceBoxed::new(), // Don't clone the trace
        }
    }
}

impl PartialEq for Error {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use whereat::ResultAtTraceableExt;

    #[test]
    fn test_error_size() {
        let size = core::mem::size_of::<Error>();
        println!("\n=== ERROR SIZES ===");
        println!("Error: {} bytes", size);
        println!("ErrorKind: {} bytes", core::mem::size_of::<ErrorKind>());
        println!(
            "ArgumentError: {} bytes",
            core::mem::size_of::<ArgumentError>()
        );
        println!(
            "ResourceError: {} bytes",
            core::mem::size_of::<ResourceError>()
        );
        assert!(size <= 48, "Error is {} bytes, consider optimizing", size);
    }

    #[test]
    fn test_argument_error_display() {
        let err = ArgumentError::InvalidDimensions {
            width: 0,
            height: 100,
            reason: "width cannot be zero",
        };
        assert!(err.to_string().contains("width cannot be zero"));
    }

    #[test]
    fn test_resource_error_display() {
        let err = ResourceError::AllocationFailed {
            bytes: 1024,
            context: "allocating buffer",
        };
        assert!(err.to_string().contains("1024 bytes"));
    }

    #[test]
    fn test_error_from_argument_error() {
        let arg_err = ArgumentError::InvalidDimensions {
            width: 0,
            height: 100,
            reason: "width cannot be zero",
        };
        let err: Error = arg_err.into();
        assert!(matches!(err.kind(), ErrorKind::InvalidDimensions { .. }));
    }

    #[test]
    fn test_error_has_trace() {
        let err = Error::invalid_dimensions(0, 100, "width cannot be zero");
        assert!(!err.trace.is_empty());
    }

    #[test]
    fn test_error_trace_propagation() {
        fn inner() -> Result<()> {
            Err(Error::invalid_dimensions(0, 100, "width cannot be zero"))
        }

        fn outer() -> Result<()> {
            inner().at()?;
            Ok(())
        }

        let err = outer().unwrap_err();
        assert!(
            err.trace.frame_count() >= 1,
            "trace should have at least 1 entry"
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err: Error = io_err.into();
        assert!(matches!(err.kind(), ErrorKind::IoError { .. }));
    }
}
