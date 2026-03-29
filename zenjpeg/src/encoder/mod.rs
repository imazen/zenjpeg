//! JPEG Encoder - Public API.
//!
//! This module provides everything needed for JPEG encoding.
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use zenjpeg::encoder::{EncoderConfig, ChromaSubsampling, PixelLayout, Unstoppable};
//!
//! let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter);
//! let mut enc = config.encode_from_bytes(1920, 1080, PixelLayout::Rgb8Srgb)?;
//! enc.push_packed(&rgb_bytes, Unstoppable)?;
//! let jpeg = enc.finish()?;
//! ```
//!
//! # Entry Points
//!
//! `EncoderConfig` provides three constructors for different color modes:
//!
//! | Constructor | Color Mode | Use Case |
//! |-------------|------------|----------|
//! | `EncoderConfig::ycbcr(q, sub)` | YCbCr | Standard JPEG (most compatible) |
//! | `EncoderConfig::xyb(q, b_sub)` | XYB | Perceptual color space |
//! | `EncoderConfig::grayscale(q)` | Grayscale | Single-channel output |
//!
//! Then use `encode_from_bytes()`, `encode_from_rgb()`, or `encode_from_ycbcr_planar()`.
//!
//! # Configuration
//!
//! ```rust,ignore
//! use zenjpeg::encoder::{EncoderConfig, Quality, ChromaSubsampling, XybSubsampling};
//!
//! // YCbCr (standard JPEG) - reusable config
//! let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
//!     .progressive(true)                        // Progressive JPEG (~3% smaller)
//!     .optimize_huffman(true)                   // Optimal Huffman tables (default)
//!     .sharp_yuv(true)                          // Better color edges (~3x slower)
//!     .restart_mcu_rows(4);                      // MCU rows between restart markers (default)
//!
//! // XYB (perceptual color space)
//! let config = EncoderConfig::xyb(85, XybSubsampling::BQuarter)
//!     .progressive(true);
//!
//! // Grayscale
//! let config = EncoderConfig::grayscale(85);
//!
//! // Quality can also use enum variants:
//! let config = EncoderConfig::ycbcr(Quality::ApproxMozjpeg(80), ChromaSubsampling::Quarter);
//! let config = EncoderConfig::ycbcr(Quality::ApproxSsim2(90.0), ChromaSubsampling::None);
//! let config = EncoderConfig::ycbcr(Quality::ApproxButteraugli(1.0), ChromaSubsampling::Quarter);
//! ```
//!
//! # Per-Image Metadata (Three-Layer Pattern)
//!
//! For encoding multiple images with different metadata:
//!
//! ```rust,ignore
//! use zenjpeg::encoder::{EncoderConfig, ChromaSubsampling, Exif, Orientation};
//!
//! // Layer 1: Reusable config (quality, color mode)
//! let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
//!     .auto_optimize(true)
//!     .progressive(true);
//!
//! // Layer 2: Per-image request (metadata)
//! let jpeg1 = config.request()
//!     .icc_profile(&srgb_bytes)
//!     .exif(Exif::build()
//!         .orientation(Orientation::Rotate90)
//!         .copyright("© 2024 Corp"))
//!     .encode(&pixels1, 1920, 1080)?;
//!
//! // Different metadata for each image
//! let jpeg2 = config.request()
//!     .icc_profile(&p3_bytes)
//!     .exif(Exif::build().copyright("Public Domain"))
//!     .encode(&pixels2, 3840, 2160)?;
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
//! | `Rgba8Srgb` / `Rgbx8Srgb` | 4 | RGBA/RGBX, alpha/pad ignored |
//! | `Bgra8Srgb` / `Bgrx8Srgb` | 4 | BGRA/BGRX, alpha/pad ignored |
//! | `Gray8Srgb` | 1 | Grayscale, sRGB gamma |
//! | `Rgb16Linear` / `Rgba16Linear` | 6/8 | 16-bit linear (alpha ignored) |
//! | `RgbF32Linear` / `RgbaF32Linear` | 12/16 | Float linear 0.0-1.0 (alpha ignored) |
//! | `YCbCr8` / `YCbCrF32` | 3/12 | Pre-converted YCbCr |
//!
//! # Cancellation
//!
//! All `push*` methods accept an `impl Stop` parameter for cooperative cancellation:
//!
//! ```rust,ignore
//! use zenjpeg::encoder::Unstoppable;
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
//! use zenjpeg::{EncoderConfig, ChromaSubsampling};
//!
//! let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);
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
//! use zenjpeg::encoder::{EncoderConfig, ChromaSubsampling, ParallelEncoding};
//!
//! let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter)
//!     .parallel(ParallelEncoding::Auto);
//! ```

// Note: Currently re-exporting internal error types since the encoder
// types we re-export from crate::encode use them internally.
// === Error types ===
/// Errors that can occur during JPEG encoding.
pub type EncodeError = crate::error::Error;
/// Result type for encoder operations.
pub type EncodeResult<T> = core::result::Result<T, EncodeError>;
// Keep legacy aliases for backward compatibility
pub use crate::error::{Error, Result};

// === Main encoder types (from encode root modules) ===
pub use crate::encode::Stop;
pub use crate::encode::byte_encoders::{BytesEncoder, Pixel, RgbEncoder, YCbCrPlanarEncoder};
pub use crate::encode::encoder_config::EncoderConfig;
pub use crate::encode::encoder_types::{
    ChromaSubsampling, ColorMode, DownsamplingMethod, HuffmanStrategy, OptimizationPreset,
    PixelLayout, ProgressiveScanMode, Quality, QuantTableConfig, XybSubsampling, YCbCrPlanes,
};
pub use crate::encode::exif::{Exif, ExifFields, Orientation};
pub use crate::encode::request::EncodeRequest;
pub use crate::foundation::alloc::EncodeStats;
pub use crate::types::Limits;

// === Default tables for customization ===
/// Default quantization and zero-bias tables.
///
/// Use these as starting points when creating custom tables.
#[doc(hidden)]
pub use crate::encode::tables;

#[cfg(feature = "parallel")]
pub use crate::encode::encoder_types::ParallelEncoding;

// === mozjpeg-compatible quantization tables ===
pub use crate::encode::tables::presets::{MozjpegTables, QuantTablePreset};

// === Huffman table types ===
/// Pre-built Huffman table set for single-pass encoding.
///
/// By default, the encoder uses general-purpose trained tables that are
/// ~5-12% more efficient than the JPEG Annex K tables.
///
/// Use [`HuffmanTableSet::annex_k()`] for the original JPEG standard tables,
/// or pass custom tables to [`EncoderConfig::custom_huffman_tables()`].
pub use crate::huffman::optimize::HuffmanTableSet;
pub use crate::types::HuffmanMethod;

// === Cancellation support ===
/// Re-exported from `enough` crate. Pass this to `push*` methods when you
/// don't need cancellation support.
pub use enough::Unstoppable;
