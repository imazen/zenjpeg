//! JPEG Decoder - Public API.
//!
//! This module provides everything needed for JPEG decoding.
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use jpegli::decoder::{Decoder, DecodedImage, Result};
//!
//! fn decode_jpeg(data: &[u8]) -> Result<DecodedImage> {
//!     Decoder::new().decode(data)
//! }
//! ```
//!
//! # Decode to specific format
//!
//! ```rust,ignore
//! use jpegli::decoder::{Decoder, PixelFormat, Result};
//!
//! fn decode_rgba(data: &[u8]) -> Result<Vec<u8>> {
//!     let image = Decoder::new()
//!         .output_format(PixelFormat::Rgba)
//!         .decode(data)?;
//!     Ok(image.into_pixels())
//! }
//! ```

mod error;

// === Error types (decoder-specific) ===
pub use error::{Error, Result};

// === Main decoder types ===
pub use crate::decode::{
    DecodedImage, DecodedImageF32, DecodedYCbCr, Decoder, DecoderConfig, JpegInfo,
    ScanlineInfo, ScanlineReader,
};

// === Types used in public structs ===
pub use crate::types::{ColorSpace, Dimensions, JpegMode, PixelFormat, Subsampling};
