//! Public API for jpegli.
//!
//! This module contains the stable public API. Implementation details
//! are in internal modules (`encode/`, `decode/`, etc.).
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use jpegli::{encode_rgb, decode};
//!
//! // Encode RGB to JPEG
//! let jpeg = encode_rgb(640, 480, &rgb_pixels, 85)?;
//!
//! // Decode JPEG to RGB
//! let image = decode(&jpeg)?;
//! ```
//!
//! # Encoder API
//!
//! ```rust,ignore
//! use jpegli::JpegEncoder;
//!
//! let jpeg = JpegEncoder::new(640, 480)
//!     .quality(85)
//!     .encode(&pixels)?;
//! ```
//!
//! # Decoder API
//!
//! ```rust,ignore
//! use jpegli::Decoder;
//!
//! let image = Decoder::new().decode(&jpeg_data)?;
//! ```

mod decoder;
mod encoder;
mod types;

// Re-export decoder API
pub use decoder::{DecodedImage, DecodedImageF32, DecodedYCbCr, Decoder, DecoderConfig, JpegInfo};

// Re-export encoder API
pub use encoder::{EncoderConfig, JpegEncoder};

// Re-export shared types
pub use types::{
    ChromaDownsampling, ChromaSubsampling, ColorMode, ColorSpace, DownsamplingMethod, JpegMode,
    PixelFormat, PixelLayout, Quality, Subsampling,
};
