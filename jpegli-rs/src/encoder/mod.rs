//! JPEG Encoder - Public API.
//!
//! This module provides everything needed for JPEG encoding.
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use jpegli::encoder::{EncoderConfig, PixelLayout, Result, Unstoppable};
//!
//! fn encode_image(pixels: &[u8]) -> Result<Vec<u8>> {
//!     let config = EncoderConfig::new().quality(85);
//!     let mut enc = config.encode_from_bytes(640, 480, PixelLayout::Rgb8Srgb)?;
//!     enc.push_packed(pixels, Unstoppable)?;
//!     enc.finish()
//! }
//! ```
//!
//! # Cancellation
//!
//! All `push*` methods accept an `impl Stop` parameter for cooperative cancellation.
//! Use `Unstoppable` when you don't need cancellation support:
//!
//! ```rust,ignore
//! use jpegli::encoder::Unstoppable;
//! encoder.push_packed(data, Unstoppable)?;
//! ```
//!
//! For cancellation support, implement the [`Stop`] trait or use types from
//! the `enough` crate directly.

// Note: Currently re-exporting internal error types since the encoder
// types we re-export from crate::encode::v2 use them internally.
// TODO: Create wrapper types or unified error type in the future.

// === Error types ===
// Re-export internal error types since the v2 encoder types use them
pub use crate::error::{Error, Result};

// === Main encoder types ===
pub use crate::encode::v2::{
    BytesEncoder, ChromaSubsampling, ColorMode, DownsamplingMethod, EncoderConfig, PixelLayout,
    Quality, QuantTableConfig, RgbEncoder, Stop, XybSubsampling, YCbCrPlanarEncoder, YCbCrPlanes,
};

#[cfg(feature = "parallel")]
pub use crate::encode::v2::ParallelEncoding;

// === Types used in encoder configuration ===
pub use crate::types::HuffmanMethod;

// === Cancellation support ===
/// Re-exported from `enough` crate. Pass this to `push*` methods when you
/// don't need cancellation support.
pub use enough::Unstoppable;
