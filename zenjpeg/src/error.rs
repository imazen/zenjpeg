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
    /// Operation was stopped via the [`Stop`](enough::Stop) trait — either
    /// explicitly cancelled or timed out. The wrapped
    /// [`StopReason`](enough::StopReason) carries which.
    ///
    /// Note: this is the hand-written equivalent of thiserror's
    /// `#[error(transparent)] Cancelled(#[from] enough::StopReason)`. The
    /// literal `transparent`/`#[from]` form can't be used because
    /// `enough::StopReason` does not implement `core::error::Error` (so it
    /// can't act as an error `source`). `#[error("{0}")]` delegates `Display`
    /// to the reason, and the `From<StopReason>` impls below supply the
    /// `#[from]` ergonomics.
    #[error("{0}")]
    Cancelled(enough::StopReason),

    // === Decoder-specific: Datastream errors ===
    /// Invalid JPEG data (corrupted or not a JPEG).
    #[error("invalid JPEG data: {reason}")]
    InvalidJpegData { reason: &'static str },
    /// Data does not begin with a JPEG SOI marker (`0xFF 0xD8`) — this isn't a
    /// JPEG file at all, as opposed to a JPEG that starts correctly but has
    /// corrupted/malformed structure further in
    /// ([`InvalidJpegData`](Self::InvalidJpegData)). Distinguishing this case
    /// lets [`category()`](zencodec::CategorizedError::category) report
    /// `ErrorCategory::Image(ImageError::Unsupported(UnsupportedImageKind::Type))`
    /// instead of `Malformed` — mirrors
    /// [`crate::detect::ProbeError::NotJpeg`], which makes the same
    /// distinction on the lightweight header-probe path (caterr Pattern-B
    /// follow-up finding #2).
    #[error("not a JPEG file: missing SOI marker")]
    NotAJpegFile,
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

    // === Taxonomy refinements (caterr Pattern-B follow-up, zencodec #103) ===
    // These four variants correct category mis-mappings the initial adoption
    // reused from the closest existing constructor (`allocation_failed` /
    // `unsupported_feature`) rather than adding new variants. `ErrorKind` is
    // `#[non_exhaustive]`, so adding variants here is additive — it does not
    // change the shape of any existing variant.
    /// A configured [`ResourceLimits`](zencodec::ResourceLimits) cap was
    /// hit — as opposed to an actual allocation failure
    /// ([`AllocationFailed`](Self::AllocationFailed)). Carries the zencodec
    /// [`LimitKind`](zencodec::LimitKind) so [`category()`](
    /// zencodec::CategorizedError::category) reports
    /// `ErrorCategory::Resource(ResourceError::Limits(kind))` for whichever
    /// cap (memory, output bytes, input bytes, …) was configured and exceeded.
    #[error("resource limit exceeded ({kind:?}): {actual} exceeds configured limit of {limit}")]
    ResourceLimitExceeded {
        kind: zencodec::LimitKind,
        actual: u64,
        limit: u64,
    },
    /// The request was valid and understood, but a configured policy
    /// declined it — e.g. [`JpegDecoderConfig`](crate::JpegDecoderConfig)'s
    /// progressive-JPEG policy rejecting a progressive image. Neither
    /// malformed input nor a missing feature: maps to
    /// `ErrorCategory::Policy(PolicyKind::Decode)`.
    #[error("rejected by policy: {reason}")]
    PolicyRejected { reason: &'static str },
    /// The operation was invoked in an invalid state or out of sequence by
    /// the caller — e.g. `push_rows` changing width/format mid-stream, or
    /// `finish()` called without any prior `push_rows()`. An API-protocol
    /// violation, not bad image data or an unimplemented feature: maps to
    /// `ErrorCategory::InvalidState`.
    #[error("invalid state: {reason}")]
    InvalidState { reason: &'static str },
    /// Pixel-format negotiation on the decode side found no acceptable
    /// [`zenpixels::PixelDescriptor`] channel-type/layout combination.
    /// Distinct from [`UnsupportedPixelFormat`](Self::UnsupportedPixelFormat)
    /// (which carries a [`crate::types::PixelFormat`]) because the decode
    /// push path negotiates raw `(ChannelType, ChannelLayout)` pairs that
    /// don't all map onto that enum; both variants share
    /// `ErrorCategory::UnsupportedPixelFormat`.
    #[error("unsupported pixel format: {reason}")]
    UnsupportedPixelDescriptor { reason: &'static str },
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
/// via `whereat`. Use [`Error::kind()`] to inspect the `ErrorKind`, or
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

    /// Create a cancelled error with reason
    /// [`StopReason::Cancelled`](enough::StopReason::Cancelled).
    ///
    /// To preserve a different [`StopReason`](enough::StopReason) (e.g. a
    /// timeout), convert it directly with `?` or `.into()` — see the
    /// `From<enough::StopReason>` impl.
    #[track_caller]
    pub fn cancelled() -> Self {
        Self::new(ErrorKind::Cancelled(enough::StopReason::Cancelled))
    }

    // ========================================================================
    // Convenience constructors - Decoder-specific errors
    // ========================================================================

    /// Create an invalid JPEG data error.
    #[track_caller]
    pub fn invalid_jpeg_data(reason: &'static str) -> Self {
        Self::new(ErrorKind::InvalidJpegData { reason })
    }

    /// Create a "missing SOI marker, not a JPEG file" error — distinct from
    /// [`Error::invalid_jpeg_data`] (a JPEG that starts correctly but is
    /// malformed further in).
    #[track_caller]
    pub fn not_a_jpeg_file() -> Self {
        Self::new(ErrorKind::NotAJpegFile)
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

    // ========================================================================
    // Convenience constructors - Taxonomy refinements (caterr Pattern-B follow-up)
    // ========================================================================

    /// Create a resource-limit-exceeded error for a configured
    /// [`zencodec::ResourceLimits`] cap (memory, output bytes, input bytes,
    /// …) — distinct from an actual allocation failure
    /// ([`Error::allocation_failed`]).
    #[track_caller]
    pub fn resource_limit_exceeded(kind: zencodec::LimitKind, actual: u64, limit: u64) -> Self {
        Self::new(ErrorKind::ResourceLimitExceeded {
            kind,
            actual,
            limit,
        })
    }

    /// Create a policy-rejected error: the request was valid and understood,
    /// but a configured policy declined it (e.g. progressive JPEG rejected
    /// by decode policy).
    #[track_caller]
    pub fn policy_rejected(reason: &'static str) -> Self {
        Self::new(ErrorKind::PolicyRejected { reason })
    }

    /// Create an invalid-state error: the caller invoked an operation out of
    /// sequence (e.g. `finish()` without `push_rows()`, or changing
    /// width/format mid-stream).
    #[track_caller]
    pub fn invalid_state(reason: &'static str) -> Self {
        Self::new(ErrorKind::InvalidState { reason })
    }

    /// Create an unsupported-pixel-descriptor error: decode-side pixel
    /// format negotiation found no acceptable
    /// `(ChannelType, ChannelLayout)` combination.
    #[track_caller]
    pub fn unsupported_pixel_descriptor(reason: &'static str) -> Self {
        Self::new(ErrorKind::UnsupportedPixelDescriptor { reason })
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

/// Mirrors the `From` impl that thiserror's `#[from]` would generate for the
/// [`ErrorKind::Cancelled`] variant. (Hand-written because `enough::StopReason`
/// does not implement `core::error::Error`; see the variant docs.)
impl From<enough::StopReason> for ErrorKind {
    fn from(reason: enough::StopReason) -> Self {
        ErrorKind::Cancelled(reason)
    }
}

impl From<enough::StopReason> for Error {
    #[track_caller]
    fn from(reason: enough::StopReason) -> Self {
        Self::new(ErrorKind::Cancelled(reason))
    }
}

impl From<zencodec::LimitExceeded> for Error {
    #[track_caller]
    fn from(err: zencodec::LimitExceeded) -> Self {
        use zencodec::LimitExceeded;
        // `.kind()` only borrows, so it's available for the arms below even
        // though the subsequent `match err` consumes `err` by value.
        let kind = err.kind();
        match err {
            LimitExceeded::Width { actual, .. } => {
                Self::invalid_dimensions(actual, 0, "width exceeds limit")
            }
            LimitExceeded::Height { actual, .. } => {
                Self::invalid_dimensions(0, actual, "height exceeds limit")
            }
            LimitExceeded::Pixels { actual, max } => Self::image_too_large(actual, max),
            // Memory/InputSize/OutputSize/Duration/TotalPixels are all
            // configured `ResourceLimits` caps, not allocation failures
            // (`Memory`, previously) or malformed bitstream content (the
            // rest, previously caught by the `decode_error` fallback below).
            // Route them all through the shared `LimitsExceeded(LimitKind)`
            // category using the exceeded kind from `err.kind()`.
            LimitExceeded::Memory { actual, max }
            | LimitExceeded::InputSize { actual, max }
            | LimitExceeded::OutputSize { actual, max }
            | LimitExceeded::Duration { actual, max }
            | LimitExceeded::TotalPixels { actual, max } => {
                Self::resource_limit_exceeded(kind, actual, max)
            }
            LimitExceeded::Frames { actual, max } => {
                Self::resource_limit_exceeded(kind, actual as u64, max as u64)
            }
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
// CategorizedError - coarse, codec-agnostic classification (zencodec #103)
// ============================================================================

/// Call sites for [`ErrorKind::UnsupportedFeature`] fall into two origins that
/// [`category()`](zencodec::CategorizedError::category) must not conflate
/// (caterr Pattern-B follow-up, finding #1): a genuine gap in this codec's
/// JPEG *bitstream* support (arithmetic coding, DNL, extended precision,
/// non-standard block sizes, …) is `Image`-origin — a different codec build
/// might still choke on the same bytes, and the caller can't fix it by
/// changing parameters. A specific zenjpeg *entry point*'s narrower contract
/// (scanline-reader component-count limits, `decode_rows()` dtype
/// restriction, an encoder config combination, a `GainMapRender` mode needing
/// a cargo feature, a color-conversion path that doesn't cover this
/// destination format, …) is `Request`-origin — the caller CAN get a
/// different result by calling a different method or changing config.
///
/// `UnsupportedFeature` is a single flat `&'static str`-payload variant used
/// at ~40 call sites, so the split can't dispatch on a typed sub-enum — there
/// is no `zencodec::UnsupportedOperation` variant matching most of these
/// zenjpeg-specific messages (that enum is closed to a handful of
/// cross-codec operations: row/pull/animation encode-decode, pixel format,
/// multi-image, gain-map encode). Call sites with a clean match against one of
/// those DO use it directly (see `descriptor_to_layout`'s
/// `UnsupportedOperation::PixelFormat` in `codec.rs`) rather than going
/// through this string classifier.
///
/// Instead this matches on stable identifying substrings of the `&'static
/// str` literals already at each call site. **If you add a new
/// `Error::unsupported_feature(...)` call site, or materially reword an
/// existing message, add/update its marker here** — an unmatched
/// Request-origin message silently falls through to `Image`, which is a
/// wrong-but-safe default (still correctly in the "client-supplied-data
/// fault" bucket, just the wrong sub-kind) rather than a compile error.
fn unsupported_feature_is_request_origin(feature: &str) -> bool {
    const REQUEST_ORIGIN_MARKERS: &[&str] = &[
        // Encoder config-combination restrictions.
        "requires optimized Huffman tables",
        "restart segments",
        "gamma-aware chroma",
        "XYB mode only supports",
        "YCbCr input not supported for XYB mode",
        // GainMapRender operation mode (cargo feature gate, unrecognized mode,
        // or a specific render mode unsatisfiable for this call).
        "GainMapRender",
        "ReconstructHdr requested",
        // A specific decode/encode entry point's narrower format/component
        // contract (as opposed to a bitstream-level cap like ">4 components").
        "scanline reader requires",
        "unsupported chroma subsampling for streaming decode",
        "unsupported color conversion",
        "unsupported grayscale",
        "YCbCr planes require",
        "YCbCr output not available for XYB images",
        "YCbCr output requires",
        "multi-channel gain maps",
        "SDR PixelBuffer must be",
        "decode_rows() only supports",
        "decode_rows_f32() only supports",
        "ultrahdr reader only supports baseline JPEG",
        "ultrahdr reader requires",
    ];
    REQUEST_ORIGIN_MARKERS.iter().any(|m| feature.contains(m))
}

/// Map each [`ErrorKind`] variant to exactly one [`zencodec::ErrorCategory`].
///
/// This is the codec-agnostic routing axis: a consumer (an HTTP layer, a retry
/// policy, a logger) reads [`category()`](zencodec::CategorizedError::category)
/// without naming any zenjpeg type. The match is exhaustive — `ErrorKind` is
/// `#[non_exhaustive]` to *callers*, but within this crate a new variant must be
/// categorized here or the build fails, so the mapping stays total.
impl zencodec::CategorizedError for ErrorKind {
    fn codec_name(&self) -> Option<&'static str> {
        Some("zenjpeg")
    }

    fn category(&self) -> zencodec::ErrorCategory {
        use zencodec::ErrorCategory as C;
        use zencodec::ImageError as Img;
        use zencodec::InternalKind as Int;
        use zencodec::InvalidKind as Inv;
        use zencodec::LimitKind as L;
        use zencodec::PolicyKind as Pol;
        use zencodec::RequestError as Req;
        use zencodec::ResourceError as Res;
        use zencodec::UnsupportedImageKind as UImg;
        use zencodec::UnsupportedOperation as Op;
        match self {
            // === Caller-supplied parameters / configuration ===
            // Dimensions (zero or out of range), the colour-format combo, and the
            // encoder dials are caller inputs, not image-data faults.
            Self::InvalidDimensions { .. }
            | Self::InvalidColorFormat { .. }
            | Self::InvalidQuality { .. }
            | Self::InvalidScanScript(_)
            | Self::InvalidConfig(_) => C::Request(Req::Invalid(Inv::Parameters)),

            // === Caller pixel-buffer geometry ===
            // Wrong byte count, or a stride narrower than the row — a buffer
            // layout fault, distinct from a bad config knob.
            Self::InvalidBufferSize { .. } | Self::StrideTooSmall { .. } => {
                C::Request(Req::Invalid(Inv::Buffer))
            }

            // === Pixel-format negotiation found no acceptable format ===
            // `PixelFormat` is a `zencodec::UnsupportedOperation` variant, not
            // its own top-level category, in the new taxonomy — it joins the
            // same Request-origin "unsupported operation" axis as
            // `UnsupportedOperation` below.
            Self::UnsupportedPixelFormat { .. } => C::Request(Req::Unsupported(Op::PixelFormat)),

            // === A valid JPEG feature this codec doesn't implement, OR a
            // caller/API-entry-point-specific restriction === (caterr
            // Pattern-B follow-up finding #1: these were conflated under one
            // flat variant; see `unsupported_feature_is_request_origin` for
            // the origin split.)
            Self::UnsupportedFeature { feature } => {
                if unsupported_feature_is_request_origin(feature) {
                    C::Request(Req::Invalid(Inv::Parameters))
                } else {
                    C::Image(Img::Unsupported(UImg::Feature))
                }
            }

            // === Memory / size acquisition failures ===
            // A real allocation failure is OOM. A size-calculation overflow means
            // the requested buffer cannot be represented at all. The
            // `try_alloc_*` / `checked_size*` helpers in `foundation::alloc`
            // are shared by both encode (caller-declared dimensions) and
            // decode (image-declared dimensions) call sites with no reliable
            // per-call-site provenance in the flat `context: &'static str`
            // payload — auditing which of ~20 generic helper call sites is
            // encode- vs decode-driven would be a separate, much larger sweep.
            // So — mirroring zenpng's overflow → OutOfMemory convention, and
            // per this task's finding #4 — both stay merged under the
            // Resource axis rather than guessing a split; the caller-visible
            // remedy (smaller input / more memory) is identical either way.
            Self::AllocationFailed { .. } | Self::SizeOverflow { .. } => {
                C::Resource(Res::OutOfMemory)
            }

            // === A configured ResourceLimits cap was exceeded ===
            Self::ImageTooLarge { .. } => C::Resource(Res::Limits(L::Pixels)),

            // === I/O / output-sink failure ===
            Self::IoError { .. } => C::Io(zencodec::CodecIoKind::opaque()),

            // === Colour management required ===
            // The sole erroring ICC path is "an ICC must be synthesized for the
            // source CICP but cannot be" — a CMS transform the codec will not
            // perform itself.
            Self::IccError(_) => C::Request(Req::CmsRequired),

            // === Not a JPEG at all (no SOI) === (caterr Pattern-B follow-up
            // finding #2.) Distinct from bitstream corruption further in —
            // mirrors `crate::detect::ProbeError::NotJpeg`, which previously
            // made this same distinction only on the lightweight header-probe
            // path, never on the main decode path.
            Self::NotAJpegFile => C::Image(Img::Unsupported(UImg::Type)),

            // === Malformed / corrupt bitstream content ===
            Self::InvalidJpegData { .. }
            | Self::InvalidMarker { .. }
            | Self::InvalidHuffmanTable { .. }
            | Self::InvalidQuantTable { .. }
            | Self::DecodeError(_) => C::Image(Img::Malformed),

            // === Structural per-image cap on progressive scan count ===
            // (caterr Pattern-B follow-up finding #3.) A `LimitKind::Scans`
            // cap: an anti-DoS ceiling on decode work, not bitstream
            // corruption — the scans themselves are well-formed, there are
            // just pathologically many of them.
            Self::TooManyScans { .. } => C::Resource(Res::Limits(L::Scans)),

            // === Truncated input ===
            Self::TruncatedData { .. } => C::Image(Img::UnexpectedEof),

            // === Caller API-protocol violation (wrong sequence / row count) ===
            Self::TooManyRows { .. } | Self::IncompleteImage { .. } => {
                C::Request(Req::Invalid(Inv::State))
            }

            // === Internal invariant / bug ===
            // A broken invariant in zenjpeg's own logic — always a code
            // defect, never attributable to the caller's request or the image.
            Self::InternalError { .. } => C::Internal(Int::Bug),

            // === Delegate to the wrapped zencodec cause types ===
            // Each carries its own `CategorizedError` impl: `StopReason` splits
            // cancelled vs timed-out, `UnsupportedOperation` splits pixel-format
            // vs operation gaps.
            Self::Cancelled(reason) => reason.category(),
            Self::UnsupportedOperation(op) => op.category(),

            // === Taxonomy refinements (caterr Pattern-B follow-up) ===
            // A configured cap, not an actual allocation failure — carries
            // which cap via the embedded `LimitKind`.
            Self::ResourceLimitExceeded { kind, .. } => C::Resource(Res::Limits(*kind)),
            // Understood-and-declined by policy, not malformed or unsupported.
            // `PolicyRejected` doesn't carry which policy (Decode vs Encode) at
            // the type level; every current call site (`codec.rs`, progressive
            // JPEG rejected by `DecodePolicy`) is decode-side, so this maps to
            // `Policy(Decode)`. If an encode-side `EncodePolicy` rejection is
            // added later, give it its own variant/payload rather than
            // guessing here.
            Self::PolicyRejected { .. } => C::Policy(Pol::Decode),
            // Caller-side API-protocol violation, not bad image data.
            Self::InvalidState { .. } => C::Request(Req::Invalid(Inv::State)),
            // Decode-side descriptor negotiation failure — same operation axis
            // as `UnsupportedPixelFormat` above, different payload shape.
            Self::UnsupportedPixelDescriptor { .. } => {
                C::Request(Req::Unsupported(Op::PixelFormat))
            }
        }
    }
}

/// The public [`Error`] newtype delegates to its [`ErrorKind`]. Consumers holding
/// an `At<ErrorKind>` (via [`Error::into_inner`]) get the same classification for
/// free through zencodec's blanket `impl CategorizedError for At<E>`.
impl zencodec::CategorizedError for Error {
    fn codec_name(&self) -> Option<&'static str> {
        Some("zenjpeg")
    }

    fn category(&self) -> zencodec::ErrorCategory {
        self.kind().category()
    }
}

// ============================================================================
// CodecError envelope bridges (Pattern B) — the zencodec trait boundary
// ============================================================================
//
// zenjpeg's zencodec trait impls (`crate::codec`) use the shared envelope
// `At<zencodec::CodecError>` as their `type Error`, so a generic consumer
// recovers the [`category`](zencodec::CategorizedError::category) **and** the
// codec name through `Dyn*` dispatch — where the concrete error is erased to a
// boxed `dyn Error` (zencodec's `BoxedError`) and only the envelope is
// recoverable (via `CodecErrorExt::codec_error`). The native [`Error`] /
// [`ErrorKind`] stay the detail + category source; these two `From` impls let
// the adapter return the envelope through `?` / `.into()` with no rewrite of the
// fallible internals.
//
// `CodecError::of` reads the category + codec name from the value's
// [`CategorizedError`] impl and keeps the `whereat` trace on the *outside*
// (`At<CodecError>`). The orphan rule permits these (the local `Error` /
// `ErrorKind` appears in the trait reference) but forbids `From<At<ErrorKind>>`
// (both `At<_>` types are foreign), so an already-located `At<ErrorKind>` is
// converted at the call site with `.map_err(zencodec::CodecError::of)` instead.

impl From<Error> for At<zencodec::CodecError> {
    /// Wrap the already-located [`Error`] (`Error(At<ErrorKind>)`) as the shared
    /// envelope, preserving its existing `whereat` trace and reading category +
    /// codec name from the inner [`ErrorKind`].
    #[inline]
    fn from(err: Error) -> Self {
        zencodec::CodecError::of(err.into_inner())
    }
}

impl From<ErrorKind> for At<zencodec::CodecError> {
    /// Locate a bare [`ErrorKind`] (`start_at`) and wrap it as the envelope — for
    /// the adapter's reject stubs / API guards that build an `ErrorKind`
    /// directly rather than going through a located [`Error`].
    #[track_caller]
    #[inline]
    fn from(kind: ErrorKind) -> Self {
        use whereat::ErrorAtExt;
        zencodec::CodecError::of(kind.start_at())
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
        let kind = ErrorKind::Cancelled(enough::StopReason::Cancelled);
        assert_error(&kind);
    }

    #[test]
    fn test_at_error_kind_roundtrip() {
        let err = Error::cancelled();
        let at: At<ErrorKind> = err.into();
        let err2: Error = at.into();
        assert_eq!(
            err2.kind(),
            &ErrorKind::Cancelled(enough::StopReason::Cancelled)
        );
    }

    #[test]
    fn cancelled_preserves_stop_reason() {
        use enough::StopReason;

        // `?`/`.into()` on a StopReason carries the reason into the error
        // (the #[from] ergonomics), instead of collapsing to a unit variant.
        let err: Error = StopReason::TimedOut.into();
        assert_eq!(err.kind(), &ErrorKind::Cancelled(StopReason::TimedOut));
        let err2: Error = StopReason::Cancelled.into();
        assert_eq!(err2.kind(), &ErrorKind::Cancelled(StopReason::Cancelled));

        // Display delegates to the reason (the #[error("{0}")] behavior).
        assert_eq!(err.kind().to_string(), "operation timed out");
        assert_eq!(err2.kind().to_string(), "operation cancelled");

        // ErrorKind::from(reason) mirrors what thiserror's #[from] would emit.
        assert_eq!(
            ErrorKind::from(StopReason::TimedOut),
            ErrorKind::Cancelled(StopReason::TimedOut)
        );
    }

    #[test]
    fn test_result_at_ext() {
        fn inner() -> core::result::Result<(), At<ErrorKind>> {
            Err(at!(ErrorKind::Cancelled(enough::StopReason::Cancelled)))
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

    #[test]
    fn categorized_error_codec_name() {
        use zencodec::CategorizedError;
        let err = Error::invalid_jpeg_data("bad SOI");
        assert_eq!(err.codec_name(), Some("zenjpeg"));
        assert_eq!(err.kind().codec_name(), Some("zenjpeg"));
    }

    #[test]
    fn categorized_error_category_mapping() {
        use enough::StopReason;
        use zencodec::{
            CategorizedError, ErrorCategory as C, ImageError as Img, InternalKind as Int,
            InvalidKind as Inv, LimitKind, PolicyKind as Pol, RequestError as Req,
            ResourceError as Res, UnsupportedImageKind as UImg, UnsupportedOperation as Op,
        };

        // Spot-check every branch of the mapping, including the delegating arms.
        let cases: &[(Error, C)] = &[
            // Caller parameters / config.
            (
                Error::invalid_dimensions(0, 10, "zero width"),
                C::Request(Req::Invalid(Inv::Parameters)),
            ),
            (
                Error::invalid_color_format("rgb+cmyk"),
                C::Request(Req::Invalid(Inv::Parameters)),
            ),
            (
                Error::invalid_quality(200.0, "0..100"),
                C::Request(Req::Invalid(Inv::Parameters)),
            ),
            (
                Error::invalid_scan_script("bad".into()),
                C::Request(Req::Invalid(Inv::Parameters)),
            ),
            (
                Error::invalid_config("bad".into()),
                C::Request(Req::Invalid(Inv::Parameters)),
            ),
            // Pixel-buffer geometry.
            (
                Error::invalid_buffer_size(100, 50),
                C::Request(Req::Invalid(Inv::Buffer)),
            ),
            (
                Error::stride_too_small(64, 16),
                C::Request(Req::Invalid(Inv::Buffer)),
            ),
            // Unsupported feature / format: image-bitstream-origin (stays
            // Image) vs invocation-origin (splits to Request) — caterr
            // Pattern-B follow-up finding #1.
            (
                Error::unsupported_feature("arithmetic coding"),
                C::Image(Img::Unsupported(UImg::Feature)),
            ),
            (
                Error::unsupported_feature("YCbCr output not available for XYB images"),
                C::Request(Req::Invalid(Inv::Parameters)),
            ),
            (
                Error::unsupported_pixel_format(crate::types::PixelFormat::Rgb),
                C::Request(Req::Unsupported(Op::PixelFormat)),
            ),
            (
                Error::unsupported_pixel_descriptor("no acceptable channel layout"),
                C::Request(Req::Unsupported(Op::PixelFormat)),
            ),
            // Memory / size — judged to stay merged (finding #4).
            (
                Error::allocation_failed(1 << 20, "buf"),
                C::Resource(Res::OutOfMemory),
            ),
            (
                Error::size_overflow("w*h*bpp"),
                C::Resource(Res::OutOfMemory),
            ),
            (
                Error::image_too_large(1_000_000, 100),
                C::Resource(Res::Limits(LimitKind::Pixels)),
            ),
            // I/O.
            (
                Error::io_error("disk".into()),
                C::Io(zencodec::CodecIoKind::opaque()),
            ),
            // Colour management.
            (
                Error::icc_error("cannot synthesize".into()),
                C::Request(Req::CmsRequired),
            ),
            // Not a JPEG at all vs malformed further in — finding #2.
            (
                Error::not_a_jpeg_file(),
                C::Image(Img::Unsupported(UImg::Type)),
            ),
            // Malformed bitstream.
            (
                Error::invalid_jpeg_data("bad SOF"),
                C::Image(Img::Malformed),
            ),
            (
                Error::invalid_marker(0xFF, "scan"),
                C::Image(Img::Malformed),
            ),
            (
                Error::invalid_huffman_table(0, "bad"),
                C::Image(Img::Malformed),
            ),
            (
                Error::invalid_quant_table(0, "bad"),
                C::Image(Img::Malformed),
            ),
            (Error::decode_error("boom".into()), C::Image(Img::Malformed)),
            // Structural scan-count ceiling, not malformed content — finding #3.
            (
                Error::too_many_scans(9999, 100),
                C::Resource(Res::Limits(LimitKind::Scans)),
            ),
            // Truncation.
            (
                Error::truncated_data("scan body"),
                C::Image(Img::UnexpectedEof),
            ),
            // API-protocol state.
            (
                Error::too_many_rows(10, 12),
                C::Request(Req::Invalid(Inv::State)),
            ),
            (
                Error::incomplete_image(10, 5),
                C::Request(Req::Invalid(Inv::State)),
            ),
            // Internal.
            (Error::internal("invariant"), C::Internal(Int::Bug)),
            // Taxonomy refinements (caterr Pattern-B follow-up).
            (
                Error::resource_limit_exceeded(zencodec::LimitKind::Memory, 100, 50),
                C::Resource(Res::Limits(LimitKind::Memory)),
            ),
            (
                Error::policy_rejected("progressive JPEG rejected by decode policy"),
                C::Policy(Pol::Decode),
            ),
            (
                Error::invalid_state("finish() called without any push_rows()"),
                C::Request(Req::Invalid(Inv::State)),
            ),
            // Delegated cause types.
            (Error::cancelled(), C::Lifecycle(StopReason::Cancelled)),
            (
                Error::from(StopReason::TimedOut),
                C::Lifecycle(StopReason::TimedOut),
            ),
            (
                Error::from(zencodec::UnsupportedOperation::AnimationEncode),
                C::Request(Req::Unsupported(Op::AnimationEncode)),
            ),
        ];

        for (err, expected) in cases {
            assert_eq!(err.category(), *expected, "category mismatch for {err:?}");
            // The located wrapper classifies identically via the blanket At impl.
            assert_eq!(err.inner().category(), *expected);
        }
    }

    /// Origin split within the single flat `UnsupportedFeature` variant
    /// (caterr Pattern-B follow-up finding #1) — every currently-known
    /// invocation-origin message must classify as `Request`, and a
    /// representative bitstream-feature-gap message must stay `Image`. This
    /// locks in `unsupported_feature_is_request_origin`'s marker list against
    /// silent drift if a call site's message text changes.
    #[test]
    fn unsupported_feature_request_origin_markers() {
        let request_origin_examples = [
            "Progressive mode requires optimized Huffman tables",
            "Smallest scan selection requires optimized Huffman tables",
            "fused parallel encode needs \u{2265}2 restart segments",
            "only RGB/RGBA supported for gamma-aware chroma",
            "XYB mode only supports RGB/RGBA pixel formats",
            "YCbCr input not supported for XYB mode",
            "GainMapRender::ReconstructHdr requires the `ultrahdr` feature",
            "unrecognized GainMapRender mode",
            "ReconstructHdr requested but the XMP has no gain map",
            "scanline reader requires 1 or 3 components in scan",
            "unsupported chroma subsampling for streaming decode",
            "unsupported color conversion",
            "unsupported grayscale \u{2192} color conversion",
            "YCbCr planes require 3-component image",
            "YCbCr output not available for XYB images",
            "YCbCr output requires 3-component image",
            "encode_ultrahdr_with_curve does not support multi-channel gain maps",
            "SDR PixelBuffer must be Rgba8 or Rgb8 for UltraHDR encoding",
            "decode_rows() only supports u8 formats",
            "decode_rows_f32() only supports f32 formats",
            "ultrahdr reader only supports baseline JPEG",
            "ultrahdr reader requires 3-component YCbCr image",
        ];
        for msg in request_origin_examples {
            assert!(
                unsupported_feature_is_request_origin(msg),
                "expected Request-origin classification for {msg:?}"
            );
        }

        let image_origin_examples = [
            "arithmetic coding",
            "12-bit precision JPEG (Extended Sequential) is not yet supported.",
            "lossless JPEG (SOF3) is not supported",
            "more than 4 components",
            "DNL mode (height=0 in SOF) not supported for streaming decode",
            "This JPEG uses non-standard DCT block sizes (DQT table is not 64 entries)",
            "decode_ultrahdr: decoder produced no u8 pixels",
        ];
        for msg in image_origin_examples {
            assert!(
                !unsupported_feature_is_request_origin(msg),
                "expected Image-origin classification for {msg:?}"
            );
        }
    }
}
