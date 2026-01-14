//! v2 Encoder API - Streaming encoder with explicit layout and cancellation.
//!
//! This module provides a new encoder API that is:
//! - **Dimension-independent**: Configuration is reusable across images
//! - **Layout-explicit**: Pixel format specified via enum or type
//! - **Streaming**: Push rows incrementally with cancellation support
//! - **Non-generic where possible**: Minimizes monomorphization
//!
//! # Quick Start
//!
//! ```ignore
//! use jpegli::encode::v2::{EncoderConfig, PixelLayout};
//! use enough::Unstoppable;
//!
//! // Create reusable config
//! let config = EncoderConfig::new()
//!     .quality(85)
//!     .progressive(true);
//!
//! // Encode from raw bytes
//! let mut enc = config.encode_from_bytes(1920, 1080, PixelLayout::Rgb8Srgb)?;
//! enc.push_packed(&rgb_bytes, Unstoppable)?;
//! let jpeg = enc.finish()?;
//! ```
//!
//! # Entry Points
//!
//! | Method | Layout From | Use Case |
//! |--------|-------------|----------|
//! | `encode_from_bytes()` | `PixelLayout` enum | Raw byte buffers |
//! | `encode_from_rgb::<P>()` | Type `P` | rgb crate types |
//! | `encode_from_ycbcr_planar()` | Fixed f32 | Video decoder output |
//!
//! # Cancellation
//!
//! All `push*` methods accept an `impl Stop` parameter for cancellation:
//!
//! ```ignore
//! use enough::Unstoppable;
//! use std::sync::atomic::AtomicBool;
//!
//! // No cancellation
//! enc.push_packed(&data, Unstoppable)?;
//!
//! // With AtomicBool
//! let cancel = AtomicBool::new(false);
//! enc.push(&data, rows, stride, &cancel)?;
//! ```

mod config;
mod encoders;
mod types;

pub use config::EncoderConfig;
pub use encoders::{BytesEncoder, Pixel, RgbEncoder, YCbCrPlanarEncoder};
#[cfg(feature = "parallel")]
pub use types::ParallelEncoding;
pub use types::{
    ChromaSubsampling, ColorMode, DownsamplingMethod, PixelLayout, Quality, QuantTableConfig,
    XybSubsampling, YCbCrPlanes,
};

// Re-export edge padding types from crate::types

// Re-export Stop trait for convenience
pub use enough::Stop;
