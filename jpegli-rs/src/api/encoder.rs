//! JPEG Encoder API.
//!
//! This module provides everything needed for JPEG encoding.
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use jpegli::encoder::{JpegEncoder, Quality, PixelFormat};
//!
//! let jpeg = JpegEncoder::new(640, 480)
//!     .quality(Quality::from_quality(85.0))
//!     .pixel_format(PixelFormat::Rgb)
//!     .encode(&rgb_pixels)?;
//! ```
//!
//! # Streaming Encoding
//!
//! ```rust,ignore
//! use jpegli::encoder::{JpegEncoder, Quality};
//!
//! let mut enc = JpegEncoder::new(640, 480)
//!     .quality(Quality::from_quality(85.0))
//!     .start()?;
//!
//! for row in image_rows {
//!     enc.push_row(row)?;
//! }
//! let jpeg = enc.finish()?;
//! ```

// === Main encoder types ===
pub use crate::encode::streaming::StreamingEncoder as JpegEncoder;

// v2 config (dimension-independent, reusable)
pub use crate::encode::v2::EncoderConfig;

// === Quality ===
pub use crate::quant::Quality;

// === Pixel formats ===
pub use crate::encode::v2::PixelLayout;
pub use crate::types::PixelFormat;

// === Subsampling ===
pub use crate::encode::v2::ChromaSubsampling;
pub use crate::types::Subsampling;

// === Color modes ===
pub use crate::encode::v2::ColorMode;
pub use crate::types::ColorSpace;

// === Downsampling methods ===
pub use crate::encode::v2::DownsamplingMethod;
pub use crate::types::ChromaDownsampling;

// === JPEG modes ===
pub use crate::types::JpegMode;

// === Error types ===
pub use crate::error::{Error, Result};
