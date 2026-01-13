//! JPEG Encoder API.
//!
//! This module provides everything needed for JPEG encoding.
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use jpegli::{EncoderConfig, PixelLayout, Result};
//!
//! fn encode_image(pixels: &[u8]) -> Result<Vec<u8>> {
//!     let config = EncoderConfig::new().quality(85);
//!     let mut enc = config.encode_from_bytes(640, 480, PixelLayout::Rgb8Srgb)?;
//!     enc.push_packed(pixels, enough::Never)?;
//!     enc.finish()
//! }
//! ```

mod error;

// Re-export v2 encoder API
pub use crate::encode::v2::{
    BytesEncoder, ChromaSubsampling, ColorMode, DownsamplingMethod, EncoderConfig, PixelLayout,
    Quality, QuantTableConfig, RgbEncoder, Stop, XybSubsampling, YCbCrPlanes, YCbCrPlanarEncoder,
};

// Error types
pub use error::{Error, Result};
