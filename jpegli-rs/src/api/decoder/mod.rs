//! JPEG Decoder API.
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
//!
//! # Decode to f32
//!
//! ```rust,ignore
//! use jpegli::decoder::{Decoder, DecodedImageF32, Result};
//!
//! fn decode_hdr(data: &[u8]) -> Result<DecodedImageF32> {
//!     Decoder::new().decode_f32(data)
//! }
//! ```

mod error;

// === Error types ===
pub use error::{Error, Result};

// === Main decoder types ===
pub use crate::decode::{
    DecodedImage, DecodedImageF32, DecodedYCbCr, Decoder, DecoderConfig, JpegInfo,
};

// === Pixel formats (for output) ===
pub use crate::types::PixelFormat;

// === Color space (from decoded image) ===
pub use crate::types::ColorSpace;

// === JPEG mode (from decoded image info) ===
pub use crate::types::JpegMode;
