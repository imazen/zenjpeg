//! Error types for jpegli.

use alloc::string::String;
use core::fmt;
use whereat::{AtTrace, AtTraceBoxed, AtTraceable};

/// Result type for jpegli operations.
pub type Result<T> = core::result::Result<T, Error>;

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

/// Errors that can occur during JPEG encoding/decoding.
///
/// Use [`Error::kind()`] to match on the specific error variant.
#[derive(Debug)]
pub struct Error {
    kind: ErrorKind,
    trace: AtTraceBoxed,
}

/// The specific kind of error that occurred.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ErrorKind {
    /// Invalid input dimensions (zero or too large).
    InvalidDimensions {
        /// Width provided
        width: u32,
        /// Height provided
        height: u32,
        /// Reason for invalidity
        reason: &'static str,
    },
    /// Invalid quality parameter.
    InvalidQuality {
        /// Value provided
        value: f32,
        /// Valid range description
        valid_range: &'static str,
    },
    /// Invalid color space or pixel format combination.
    InvalidColorFormat {
        /// Description of the issue
        reason: &'static str,
    },
    /// Input buffer has wrong size.
    InvalidBufferSize {
        /// Expected size in bytes
        expected: usize,
        /// Actual size in bytes
        actual: usize,
    },
    /// Invalid JPEG data (corrupted or not a JPEG).
    InvalidJpegData {
        /// Description of the issue
        reason: &'static str,
    },
    /// Input data is truncated or corrupted.
    TruncatedData {
        /// Context where truncation was detected
        context: &'static str,
    },
    /// Invalid marker or segment in JPEG stream.
    InvalidMarker {
        /// The marker byte encountered
        marker: u8,
        /// Context
        context: &'static str,
    },
    /// Invalid Huffman table.
    InvalidHuffmanTable {
        /// Table index
        table_idx: u8,
        /// Description of the issue
        reason: &'static str,
    },
    /// Invalid quantization table.
    InvalidQuantTable {
        /// Table index
        table_idx: u8,
        /// Description of the issue
        reason: &'static str,
    },
    /// Unsupported JPEG feature.
    UnsupportedFeature {
        /// Description of unsupported feature
        feature: &'static str,
    },
    /// Internal error (should not happen in correct usage).
    InternalError {
        /// Description
        reason: &'static str,
    },
    /// I/O error during encoding/decoding.
    IoError {
        /// Description
        reason: String,
    },
    /// ICC color management error.
    IccError(String),
    /// Decode error from JPEG decoder.
    DecodeError(String),
    /// Invalid scan script for progressive encoding.
    InvalidScanScript(String),
    /// Memory allocation failed (OOM or limit exceeded).
    AllocationFailed {
        /// Number of bytes requested
        bytes: usize,
        /// Context where allocation failed
        context: &'static str,
    },
    /// Size calculation overflowed.
    SizeOverflow {
        /// Context where overflow occurred
        context: &'static str,
    },
    /// Image exceeds maximum pixel limit.
    ImageTooLarge {
        /// Total pixels in image
        pixels: u64,
        /// Maximum allowed pixels
        limit: u64,
    },
    /// Too many progressive scans.
    TooManyScans {
        /// Number of scans encountered
        count: usize,
        /// Maximum allowed
        limit: usize,
    },
    /// Operation was cancelled via Stop trait.
    Cancelled,
    /// Pixel format not yet supported for this operation.
    UnsupportedPixelFormat {
        /// The pixel format that was attempted
        format: crate::types::PixelFormat,
    },
    /// Invalid encoder configuration.
    InvalidConfig(String),
    /// Stride too small for image width.
    StrideTooSmall {
        /// Image width
        width: u32,
        /// Stride provided
        stride: usize,
    },
    /// Pushed more rows than image height.
    TooManyRows {
        /// Image height
        height: u32,
        /// Rows already pushed
        pushed: u32,
    },
    /// Encoding finished without all rows pushed.
    IncompleteImage {
        /// Image height
        height: u32,
        /// Rows actually pushed
        pushed: u32,
    },
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

    // Convenience constructors for common errors

    /// Create an invalid dimensions error.
    #[track_caller]
    pub fn invalid_dimensions(width: u32, height: u32, reason: &'static str) -> Self {
        Self::new(ErrorKind::InvalidDimensions {
            width,
            height,
            reason,
        })
    }

    /// Create an invalid quality error.
    #[track_caller]
    pub fn invalid_quality(value: f32, valid_range: &'static str) -> Self {
        Self::new(ErrorKind::InvalidQuality { value, valid_range })
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

    /// Create an unsupported feature error.
    #[track_caller]
    pub fn unsupported_feature(feature: &'static str) -> Self {
        Self::new(ErrorKind::UnsupportedFeature { feature })
    }

    /// Create an internal error.
    #[track_caller]
    pub fn internal(reason: &'static str) -> Self {
        Self::new(ErrorKind::InternalError { reason })
    }

    /// Create an I/O error.
    #[track_caller]
    pub fn io_error(reason: String) -> Self {
        Self::new(ErrorKind::IoError { reason })
    }

    /// Create an ICC error.
    #[track_caller]
    pub fn icc_error(reason: String) -> Self {
        Self::new(ErrorKind::IccError(reason))
    }

    /// Create a decode error.
    #[track_caller]
    pub fn decode_error(reason: String) -> Self {
        Self::new(ErrorKind::DecodeError(reason))
    }

    /// Create an invalid scan script error.
    #[track_caller]
    pub fn invalid_scan_script(reason: String) -> Self {
        Self::new(ErrorKind::InvalidScanScript(reason))
    }

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

    /// Create a too many scans error.
    #[track_caller]
    pub fn too_many_scans(count: usize, limit: usize) -> Self {
        Self::new(ErrorKind::TooManyScans { count, limit })
    }

    /// Create a cancelled error.
    #[track_caller]
    pub fn cancelled() -> Self {
        Self::new(ErrorKind::Cancelled)
    }

    /// Create an unsupported pixel format error.
    #[track_caller]
    pub fn unsupported_pixel_format(format: crate::types::PixelFormat) -> Self {
        Self::new(ErrorKind::UnsupportedPixelFormat { format })
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
            Self::DecodeError(reason) => {
                write!(f, "decode error: {}", reason)
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
            Self::TooManyScans { count, limit } => {
                write!(f, "too many scans: {} exceeds limit of {}", count, limit)
            }
            Self::Cancelled => {
                write!(f, "operation cancelled")
            }
            Self::UnsupportedPixelFormat { format } => {
                write!(f, "pixel format {:?} not yet supported", format)
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

// Implement Clone manually since AtTrace doesn't implement Clone
impl Clone for Error {
    fn clone(&self) -> Self {
        Self {
            kind: self.kind.clone(),
            trace: AtTraceBoxed::new(), // Don't clone the trace
        }
    }
}

// Implement PartialEq based on kind only (trace is not compared)
impl PartialEq for Error {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use whereat::ResultAtTraceableExt;

    #[test]
    fn test_error_size() {
        let size = core::mem::size_of::<Error>();
        println!("\n=== ERROR SIZES ===");
        println!("Error: {} bytes", size);
        println!(
            "Option<Error>: {} bytes",
            core::mem::size_of::<Option<Error>>()
        );
        println!("Result<()>: {} bytes", core::mem::size_of::<Result<()>>());
        println!(
            "core::result::Result<(), Error>: {} bytes",
            core::mem::size_of::<core::result::Result<(), Error>>()
        );
        println!("Box<Error>: {} bytes", core::mem::size_of::<Box<Error>>());
        println!("ErrorKind: {} bytes", core::mem::size_of::<ErrorKind>());
        // Error = ErrorKind (~32 bytes: String payload + discriminant) + AtTraceBoxed (8 bytes)
        // Expected: ~40 bytes on 64-bit platforms
        assert!(size <= 48, "Error is {} bytes, consider optimizing", size);
    }

    #[test]
    fn test_error_display() {
        let err = Error::invalid_dimensions(0, 100, "width cannot be zero");
        assert!(err.to_string().contains("width cannot be zero"));
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
        // Should have 2 trace entries: one from inner, one from outer's .at()
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

    /// Test translating jpegli errors into a different At-style error type
    /// while preserving the full stack trace.
    #[test]
    fn test_trace_preservation_across_error_types() {
        use whereat::At;

        // A different application-level error type using whereat::At
        #[derive(Debug)]
        enum AppErrorKind {
            ImageProcessing(ErrorKind),
            Other(&'static str),
        }

        // Simulate a call chain that creates and propagates a jpegli error
        fn jpegli_inner() -> Result<()> {
            Err(Error::invalid_dimensions(0, 100, "width cannot be zero"))
        }

        fn jpegli_outer() -> Result<()> {
            jpegli_inner().at_str("processing user upload")?;
            Ok(())
        }

        // Convert jpegli::Error to At<AppErrorKind>, preserving the trace
        fn to_app_error(mut err: Error) -> At<AppErrorKind> {
            // Take the trace from jpegli error
            let trace = err.trace.take().unwrap_or_default();

            // Create new At<AppErrorKind> with the transferred trace
            At::from_parts(AppErrorKind::ImageProcessing(err.into_kind()), trace)
        }

        // Run the chain and convert
        let jpegli_err = jpegli_outer().unwrap_err();
        let original_frame_count = jpegli_err.trace.frame_count();

        // Verify we have trace frames before conversion
        assert!(
            original_frame_count >= 1,
            "jpegli error should have trace frames"
        );

        // Convert to app error
        let app_err = to_app_error(jpegli_err);

        // Verify trace was preserved
        assert_eq!(
            app_err.frame_count(),
            original_frame_count,
            "trace frames should be preserved after conversion"
        );

        // Verify we can still iterate the trace
        let frames: Vec<_> = app_err.frames().collect();
        assert!(
            !frames.is_empty(),
            "should be able to iterate trace frames"
        );

        // Verify the error kind was preserved
        assert!(
            matches!(app_err.error(), AppErrorKind::ImageProcessing(_)),
            "error kind should be ImageProcessing"
        );

        // Verify context string is accessible via frames
        let mut found_context = false;
        for frame in app_err.frames() {
            for ctx in frame.contexts() {
                if ctx.as_text() == Some("processing user upload") {
                    found_context = true;
                }
            }
        }
        assert!(found_context, "context string should be preserved in trace");
    }

    /// Test using AtTraceable::into_at() for cleaner error conversion
    #[test]
    fn test_into_at_conversion() {
        use whereat::At;

        #[derive(Debug)]
        struct WrapperError {
            kind: ErrorKind,
        }

        fn create_error() -> Result<()> {
            Err(Error::invalid_quality(150.0, "0.0-100.0"))
        }

        fn propagate() -> Result<()> {
            create_error().at_str("in propagate")?;
            Ok(())
        }

        let jpegli_err = propagate().unwrap_err();
        let original_frames = jpegli_err.trace.frame_count();

        // Use into_at() to convert while preserving trace
        let wrapper: At<WrapperError> = jpegli_err.into_at(|e| WrapperError {
            kind: e.into_kind(),
        });

        // Verify trace preserved
        assert_eq!(
            wrapper.frame_count(),
            original_frames,
            "into_at should preserve all trace frames"
        );

        // Verify error content
        assert!(
            matches!(wrapper.error().kind, ErrorKind::InvalidQuality { .. }),
            "error kind should be preserved"
        );
    }
}
