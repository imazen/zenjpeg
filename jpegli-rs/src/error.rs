//! Error types for jpegli.

use std::fmt;

/// Result type for jpegli operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur during JPEG encoding/decoding.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Error {
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
    /// Unexpected end of input data.
    UnexpectedEof {
        /// Context where EOF occurred
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
            Self::InvalidJpegData { reason } => {
                write!(f, "invalid JPEG data: {}", reason)
            }
            Self::UnexpectedEof { context } => {
                write!(f, "unexpected end of data while {}", context)
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
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Self::IoError {
            reason: err.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = Error::InvalidDimensions {
            width: 0,
            height: 100,
            reason: "width cannot be zero",
        };
        assert!(err.to_string().contains("width cannot be zero"));
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err: Error = io_err.into();
        assert!(matches!(err, Error::IoError { .. }));
    }
}
