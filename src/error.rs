//! Error types for zenjpeg

use std::fmt;

/// Result type for zenjpeg operations
pub type Result<T> = std::result::Result<T, Error>;

/// Error type for zenjpeg operations
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// Invalid image dimensions
    InvalidDimensions {
        width: usize,
        height: usize,
        reason: &'static str,
    },
    /// Invalid quality value
    InvalidQuality {
        value: f32,
        min: f32,
        max: f32,
    },
    /// Invalid pixel data
    InvalidPixelData {
        expected: usize,
        actual: usize,
    },
    /// Encoding failed
    EncodingFailed {
        stage: &'static str,
        reason: String,
    },
    /// Internal encoder error
    Internal(&'static str),
    /// Invalid Huffman table structure
    InvalidHuffmanTable,
    /// Huffman code length exceeds maximum (16 bits)
    HuffmanCodeLengthOverflow,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidDimensions { width, height, reason } => {
                write!(f, "Invalid dimensions {}x{}: {}", width, height, reason)
            }
            Error::InvalidQuality { value, min, max } => {
                write!(f, "Quality {} out of range [{}, {}]", value, min, max)
            }
            Error::InvalidPixelData { expected, actual } => {
                write!(f, "Expected {} bytes of pixel data, got {}", expected, actual)
            }
            Error::EncodingFailed { stage, reason } => {
                write!(f, "Encoding failed at {}: {}", stage, reason)
            }
            Error::Internal(msg) => write!(f, "Internal error: {}", msg),
            Error::InvalidHuffmanTable => write!(f, "Invalid Huffman table structure"),
            Error::HuffmanCodeLengthOverflow => {
                write!(f, "Huffman code length exceeds maximum (16 bits)")
            }
        }
    }
}

impl std::error::Error for Error {}
