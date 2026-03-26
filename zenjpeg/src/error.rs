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
use whereat::At;
use whereat::at;

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

// These helper methods are part of the prerelease decoder API.
// Some are unused internally but provided for external callers.
#[allow(dead_code)]
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
// ErrorKind - Flat enum for all error variants (derives thiserror::Error)
// ============================================================================

/// The specific kind of error that occurred.
///
/// This is the inner error type wrapped by [`Error`] (which is `At<ErrorKind>`).
/// Use [`Error::error()`](At::error) to access the kind, or pattern-match
/// after calling [`Error::into_inner()`](At::into_inner).
#[derive(Debug, Clone, PartialEq, Error)]
#[non_exhaustive]
pub enum ErrorKind {
    // === Shared: Argument errors ===
    /// Invalid input dimensions (zero or too large).
    #[error("invalid dimensions {width}x{height}: {reason}")]
    InvalidDimensions {
        width: u32,
        height: u32,
        reason: &'static str,
    },
    /// Invalid color space or pixel format combination.
    #[error("invalid color format: {reason}")]
    InvalidColorFormat { reason: &'static str },
    /// Input buffer has wrong size.
    #[error("invalid buffer size: expected {expected} bytes, got {actual}")]
    InvalidBufferSize { expected: usize, actual: usize },
    /// Unsupported JPEG feature.
    #[error("unsupported feature: {feature}")]
    UnsupportedFeature { feature: &'static str },
    /// Pixel format not yet supported for this operation.
    #[error("pixel format {format:?} not supported")]
    UnsupportedPixelFormat { format: crate::types::PixelFormat },

    // === Shared: Resource errors ===
    /// Memory allocation failed (OOM or limit exceeded).
    #[error("allocation of {bytes} bytes failed while {context}")]
    AllocationFailed { bytes: usize, context: &'static str },
    /// Size calculation overflowed.
    #[error("size calculation overflow while {context}")]
    SizeOverflow { context: &'static str },
    /// Image exceeds maximum pixel limit.
    #[error("image too large: {pixels} pixels exceeds limit of {limit}")]
    ImageTooLarge { pixels: u64, limit: u64 },
    /// I/O error during encoding/decoding.
    #[error("I/O error: {reason}")]
    IoError { reason: String },

    // === Shared: Other ===
    /// ICC color management error.
    #[error("ICC error: {0}")]
    IccError(String),
    /// Internal error (should not happen in correct usage).
    #[error("internal error: {reason}")]
    InternalError { reason: &'static str },
    /// Operation was cancelled via Stop trait.
    #[error("operation cancelled")]
    Cancelled,

    // === Decoder-specific: Datastream errors ===
    /// Invalid JPEG data (corrupted or not a JPEG).
    #[error("invalid JPEG data: {reason}")]
    InvalidJpegData { reason: &'static str },
    /// Input data is truncated or corrupted.
    #[error("truncated data while {context}")]
    TruncatedData { context: &'static str },
    /// Invalid marker or segment in JPEG stream.
    #[error("invalid marker 0x{marker:02X} while {context}")]
    InvalidMarker { marker: u8, context: &'static str },
    /// Invalid Huffman table.
    #[error("invalid Huffman table {table_idx}: {reason}")]
    InvalidHuffmanTable { table_idx: u8, reason: &'static str },
    /// Invalid quantization table.
    #[error("invalid quantization table {table_idx}: {reason}")]
    InvalidQuantTable { table_idx: u8, reason: &'static str },
    /// Too many progressive scans.
    #[error("too many scans: {count} exceeds limit of {limit}")]
    TooManyScans { count: usize, limit: usize },
    /// Decode error from JPEG decoder.
    #[error("decode error: {0}")]
    DecodeError(String),

    // === Encoder-specific: Argument errors ===
    /// Invalid quality parameter.
    #[error("invalid quality {value}: must be in {valid_range}")]
    InvalidQuality {
        value: f32,
        valid_range: &'static str,
    },
    /// Invalid scan script for progressive encoding.
    #[error("invalid scan script: {0}")]
    InvalidScanScript(String),
    /// Invalid encoder configuration.
    #[error("invalid encoder configuration: {0}")]
    InvalidConfig(String),
    /// Stride too small for image width.
    #[error("stride {stride} is too small for width {width} pixels")]
    StrideTooSmall { width: u32, stride: usize },

    // === Encoder-specific: State errors ===
    /// Pushed more rows than image height.
    #[error("pushed {pushed} rows but image height is only {height}")]
    TooManyRows { height: u32, pushed: u32 },
    /// Encoding finished without all rows pushed.
    #[error("encoding finished after {pushed} rows but image height is {height}")]
    IncompleteImage { height: u32, pushed: u32 },

    // === Unsupported codec operation (from zencodec) ===
    /// Unsupported codec operation.
    #[error("unsupported operation: {0}")]
    UnsupportedOperation(zencodec::UnsupportedOperation),
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
            ResourceError::ImageTooLarge { pixels, limit } => Self::ImageTooLarge { pixels, limit },
            ResourceError::IoError { reason } => Self::IoError { reason },
        }
    }
}

// ============================================================================
// Error - Main error type: At<ErrorKind> newtype with location tracking
// ============================================================================

/// Errors that can occur during JPEG encoding/decoding.
///
/// This is a newtype around `At<ErrorKind>` providing zero-cost stacktraces
/// via `whereat`. Use [`Error::error()`] to inspect the [`ErrorKind`], or
/// [`Error::into_inner()`] to destructure.
///
/// Traces propagate automatically through `?` when using `ResultAtExt::at()`.
#[derive(Debug)]
pub struct Error(pub At<ErrorKind>);

impl Error {
    /// Wrap an `ErrorKind` with location tracking.
    #[track_caller]
    #[inline]
    pub fn new(kind: ErrorKind) -> Self {
        Self(at!(kind))
    }

    /// Wrap an `ErrorKind` without capturing a trace (for hot paths).
    #[inline]
    pub const fn new_untraced(kind: ErrorKind) -> Self {
        Self(At::wrap(kind))
    }

    /// Get the kind of error.
    #[inline]
    pub fn kind(&self) -> &ErrorKind {
        self.0.error()
    }

    /// Convert into the error kind, discarding the trace.
    #[inline]
    pub fn into_kind(self) -> ErrorKind {
        self.0.decompose().0
    }

    /// Access the inner `At<ErrorKind>` for trace inspection.
    #[inline]
    pub fn inner(&self) -> &At<ErrorKind> {
        &self.0
    }

    /// Consume and return the inner `At<ErrorKind>`.
    #[inline]
    pub fn into_inner(self) -> At<ErrorKind> {
        self.0
    }

    /// Add the caller's location to the trace (for propagation).
    #[track_caller]
    #[inline]
    pub fn at(self) -> Self {
        Self(self.0.at())
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

// ============================================================================
// Standard trait implementations for Error
// ============================================================================

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

impl core::error::Error for Error {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        self.0.error().source()
    }
}

impl PartialEq for Error {
    fn eq(&self, other: &Self) -> bool {
        self.0.error() == other.0.error()
    }
}

// ============================================================================
// From implementations for Error
// ============================================================================

impl From<ErrorKind> for Error {
    #[track_caller]
    #[inline]
    fn from(kind: ErrorKind) -> Self {
        Self::new(kind)
    }
}

impl From<At<ErrorKind>> for Error {
    #[inline]
    fn from(at: At<ErrorKind>) -> Self {
        Self(at)
    }
}

impl From<Error> for At<ErrorKind> {
    #[inline]
    fn from(err: Error) -> Self {
        err.0
    }
}

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

impl From<zencodec::LimitExceeded> for Error {
    #[track_caller]
    fn from(err: zencodec::LimitExceeded) -> Self {
        use zencodec::LimitExceeded;
        match err {
            LimitExceeded::Width { actual, .. } => {
                Self::invalid_dimensions(actual, 0, "width exceeds limit")
            }
            LimitExceeded::Height { actual, .. } => {
                Self::invalid_dimensions(0, actual, "height exceeds limit")
            }
            LimitExceeded::Pixels { actual, max } => Self::image_too_large(actual, max),
            LimitExceeded::Memory { actual, max } => Self::new(ErrorKind::AllocationFailed {
                bytes: actual as usize,
                context: if max > 0 {
                    "memory limit exceeded"
                } else {
                    "allocation failed"
                },
            }),
            _ => Self::decode_error(format!("{err}")),
        }
    }
}

impl From<zencodec::UnsupportedOperation> for Error {
    #[track_caller]
    fn from(op: zencodec::UnsupportedOperation) -> Self {
        Self::new(ErrorKind::UnsupportedOperation(op))
    }
}

impl From<std::io::Error> for Error {
    #[track_caller]
    fn from(err: std::io::Error) -> Self {
        Self::io_error(err.to_string())
    }
}

#[cfg(feature = "ultrahdr")]
impl From<ultrahdr_core::Error> for Error {
    #[track_caller]
    fn from(err: ultrahdr_core::Error) -> Self {
        use ultrahdr_core::Error as UhdrError;
        match err {
            UhdrError::Stopped(reason) => Self::from(reason),
            UhdrError::InvalidDimensions(w, h) => {
                Self::invalid_dimensions(w, h, "invalid dimensions for UltraHDR")
            }
            UhdrError::DimensionMismatch { .. } => Self::decode_error(err.to_string()),
            UhdrError::AllocationFailed(bytes) => {
                Self::allocation_failed(bytes, "UltraHDR operation")
            }
            UhdrError::LimitExceeded(msg) => Self::decode_error(msg),
            _ => Self::decode_error(err.to_string()),
        }
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
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use whereat::ResultAtExt;

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
        // At<ErrorKind> = ErrorKind + 8 bytes (boxed trace pointer)
        // Error newtype adds zero overhead
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
        assert!(err.0.frame_count() >= 1);
    }

    #[test]
    fn test_error_trace_propagation() {
        fn inner() -> Result<()> {
            Err(Error::invalid_dimensions(0, 100, "width cannot be zero"))
        }

        fn outer() -> Result<()> {
            inner().map_err(|e| e.at())?;
            Ok(())
        }

        let err = outer().unwrap_err();
        assert!(
            err.0.frame_count() >= 2,
            "trace should have at least 2 entries (inner + outer), got {}",
            err.0.frame_count()
        );
    }

    #[test]
    fn test_error_kind_is_error_trait() {
        // ErrorKind implements core::error::Error via thiserror
        fn assert_error<E: core::error::Error>(_: &E) {}
        let kind = ErrorKind::Cancelled;
        assert_error(&kind);
    }

    #[test]
    fn test_at_error_kind_roundtrip() {
        let err = Error::cancelled();
        let at: At<ErrorKind> = err.into();
        let err2: Error = at.into();
        assert_eq!(err2.kind(), &ErrorKind::Cancelled);
    }

    #[test]
    fn test_result_at_ext() {
        fn inner() -> core::result::Result<(), At<ErrorKind>> {
            Err(at!(ErrorKind::Cancelled))
        }

        fn outer() -> core::result::Result<(), At<ErrorKind>> {
            inner().at()?;
            Ok(())
        }

        let err = outer().unwrap_err();
        assert!(err.frame_count() >= 2);
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err: Error = io_err.into();
        assert!(matches!(err.kind(), ErrorKind::IoError { .. }));
    }
}
