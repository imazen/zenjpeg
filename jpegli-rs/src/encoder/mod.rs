//! JPEG Encoder - Public API.
//!
//! This module provides everything needed for JPEG encoding.
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use jpegli::encoder::{EncoderConfig, PixelLayout, Unstoppable};
//!
//! let config = EncoderConfig::new().quality(85);
//! let mut enc = config.encode_from_bytes(1920, 1080, PixelLayout::Rgb8Srgb)?;
//! enc.push_packed(&rgb_bytes, Unstoppable)?;
//! let jpeg = enc.finish()?;
//! ```
//!
//! # Entry Points
//!
//! `EncoderConfig` provides three encoder creation methods:
//!
//! | Method | Input Type | Use Case |
//! |--------|------------|----------|
//! | `encode_from_bytes()` | `&[u8]` | Raw byte buffers |
//! | `encode_from_rgb()` | `rgb` crate types | Type-safe pixels |
//! | `encode_from_ycbcr_planar()` | `YCbCrPlanes` | Video pipelines |
//!
//! # Configuration
//!
//! ```rust,ignore
//! use jpegli::encoder::{EncoderConfig, Quality, ChromaSubsampling};
//!
//! let config = EncoderConfig::new()
//!     // Quality (multiple scales available)
//!     .quality(85)                              // 0-100 scale (default: 90)
//!     .quality(Quality::ApproxMozjpeg(80))      // Match mozjpeg output
//!     .quality(Quality::ApproxSsim2(90.0))      // Target SSIMULACRA2 score
//!     .quality(Quality::ApproxButteraugli(1.0)) // Target butteraugli distance
//!
//!     // Encoding mode
//!     .progressive(true)                        // Progressive JPEG (~3% smaller)
//!     .optimize_huffman(true)                   // Optimal Huffman tables (default)
//!
//!     // Color mode (default is 4:4:4)
//!     .ycbcr(ChromaSubsampling::Full)           // 4:4:4 (default, best quality)
//!     .ycbcr(ChromaSubsampling::Quarter)        // 4:2:0 (good compression)
//!     .xyb()                                    // XYB perceptual color space
//!     .grayscale()                              // Single-channel output
//!
//!     // Downsampling
//!     .sharp_yuv(true)                          // Better color edges (~3x slower)
//!
//!     // Metadata
//!     .icc_profile(bytes)                       // Attach ICC profile
//!     .restart_interval(64);                    // MCUs between restart markers
//! ```
//!
//! # Pixel Layouts
//!
//! `PixelLayout` describes the format of raw byte input:
//!
//! | Layout | Bytes/px | Description |
//! |--------|----------|-------------|
//! | `Rgb8Srgb` | 3 | RGB, sRGB gamma (default) |
//! | `Bgr8Srgb` | 3 | BGR, sRGB gamma (Windows/GDI) |
//! | `Rgbx8Srgb` / `Bgrx8Srgb` | 4 | 4th byte ignored |
//! | `Gray8Srgb` | 1 | Grayscale, sRGB gamma |
//! | `Rgb16Linear` / `Rgbx16Linear` | 6/8 | 16-bit linear |
//! | `RgbF32Linear` / `RgbxF32Linear` | 12/16 | Float linear (0.0-1.0) |
//! | `YCbCr8` / `YCbCrF32` | 3/12 | Pre-converted YCbCr |
//!
//! # Cancellation
//!
//! All `push*` methods accept an `impl Stop` parameter for cooperative cancellation:
//!
//! ```rust,ignore
//! use jpegli::encoder::Unstoppable;
//! use std::sync::atomic::AtomicBool;
//!
//! // No cancellation
//! enc.push_packed(&data, Unstoppable)?;
//!
//! // With AtomicBool
//! let cancel = AtomicBool::new(false);
//! enc.push_packed(&data, &cancel)?;
//! ```
//!
//! # Memory Estimation
//!
//! ```rust,ignore
//! let config = EncoderConfig::new().quality(85);
//!
//! // Typical estimate
//! let estimate = config.estimate_memory(1920, 1080);
//!
//! // Guaranteed upper bound
//! let ceiling = config.estimate_memory_ceiling(1920, 1080);
//! ```
//!
//! # Parallel Encoding
//!
//! With the `parallel` feature:
//!
//! ```rust,ignore
//! #[cfg(feature = "parallel")]
//! use jpegli::encoder::ParallelEncoding;
//!
//! let config = EncoderConfig::new()
//!     .quality(85)
//!     .parallel(ParallelEncoding::Auto);
//! ```

// Note: Currently re-exporting internal error types since the encoder
// types we re-export from crate::encode use them internally.
// TODO: Create wrapper types or unified error type in the future.

// === Error types ===
// Re-export internal error types since the encoder types use them
pub use crate::error::{Error, Result};

// === Main encoder types (from encode root modules) ===
pub use crate::encode::byte_encoders::{BytesEncoder, Pixel, RgbEncoder, YCbCrPlanarEncoder};
pub use crate::encode::encoder_config::EncoderConfig;
pub use crate::encode::encoder_types::{
    ChromaSubsampling, ColorMode, DownsamplingMethod, PixelLayout, Quality, QuantTableConfig,
    XybSubsampling, YCbCrPlanes,
};
pub use crate::encode::Stop;

#[cfg(feature = "parallel")]
pub use crate::encode::encoder_types::ParallelEncoding;

// === Types used in encoder configuration ===
pub use crate::types::HuffmanMethod;

// === Cancellation support ===
/// Re-exported from `enough` crate. Pass this to `push*` methods when you
/// don't need cancellation support.
pub use enough::Unstoppable;
