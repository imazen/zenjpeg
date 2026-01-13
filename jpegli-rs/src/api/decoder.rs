//! JPEG Decoder API.
//!
//! This module provides everything needed for JPEG decoding.
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use jpegli::decoder::{Decoder, DecodedImage};
//!
//! let image: DecodedImage = Decoder::new().decode(&jpeg_data)?;
//! let pixels: &[u8] = image.pixels();
//! ```
//!
//! # Decode to specific format
//!
//! ```rust,ignore
//! use jpegli::decoder::{Decoder, PixelFormat};
//!
//! let image = Decoder::new()
//!     .output_format(PixelFormat::Rgba)
//!     .decode(&jpeg_data)?;
//! ```
//!
//! # Decode to f32
//!
//! ```rust,ignore
//! use jpegli::decoder::Decoder;
//!
//! let image = Decoder::new().decode_f32(&jpeg_data)?;
//! let pixels: &[f32] = image.pixels();
//! ```

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

// === Error types ===
pub use crate::error::{Error, Result};
