#![cfg_attr(not(feature = "unsafe_simd"), forbid(unsafe_code))]

//! # jpegli
//!
//! Pure Rust JPEG encoder with perceptual optimizations.
//!
//! jpegli provides enhanced compression quality compared to standard JPEG
//! through adaptive quantization, optional XYB color space, and other
//! perceptual optimizations.
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use jpegli::encoder::{EncoderConfig, PixelLayout, Unstoppable};
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
//! ## Encoder API
//!
//! All encoder types are in [`encoder`]:
//!
//! ```rust,ignore
//! use jpegli::encoder::{
//!     // Core types
//!     EncoderConfig,          // Builder for encoder configuration
//!     BytesEncoder,           // Encoder for raw byte buffers
//!     RgbEncoder,             // Encoder for rgb crate types
//!     YCbCrPlanarEncoder,     // Encoder for planar YCbCr
//!
//!     // Configuration
//!     Quality,                // Quality settings (ApproxJpegli, ApproxMozjpeg, etc.)
//!     PixelLayout,            // Pixel format for raw bytes
//!     ChromaSubsampling,      // 4:4:4, 4:2:0, 4:2:2, 4:4:0
//!     ColorMode,              // YCbCr, XYB, Grayscale
//!     DownsamplingMethod,     // Box, GammaAware, GammaAwareIterative
//!
//!     // Cancellation
//!     Stop,                   // Trait for cancellation tokens
//!     Unstoppable,            // Use when no cancellation needed
//!
//!     // Results
//!     Error, Result,          // Error handling
//! };
//! ```
//!
//! ### Three Entry Points
//!
//! | Method | Input Type | Use Case |
//! |--------|------------|----------|
//! | [`EncoderConfig::encode_from_bytes`] | `&[u8]` | Raw byte buffers |
//! | [`EncoderConfig::encode_from_rgb`] | `rgb` crate types | Type-safe pixels |
//! | [`EncoderConfig::encode_from_ycbcr_planar`] | [`YCbCrPlanes`](encoder::YCbCrPlanes) | Video pipelines |
//!
//! ### Configuration Options
//!
//! ```rust,ignore
//! let config = EncoderConfig::new()
//!     .quality(85)                              // 0-100 scale
//!     .quality(Quality::ApproxSsim2(90.0))      // Target SSIMULACRA2 score
//!     .quality(Quality::ApproxButteraugli(1.0)) // Target butteraugli distance
//!     .progressive(true)                        // Progressive JPEG (~3% smaller)
//!     .ycbcr(ChromaSubsampling::Quarter)        // 4:2:0 (default)
//!     .xyb()                                    // XYB perceptual color space
//!     .grayscale()                              // Single-channel output
//!     .sharp_yuv(true)                          // Better color edges (~3x slower)
//!     .icc_profile(bytes);                      // Attach ICC profile
//! ```
//!
//! ## Decoder API
//!
//! The decoder is in prerelease. Enable with `features = ["decoder"]`.
//!
//! ```rust,ignore
//! #[cfg(feature = "decoder")]
//! use jpegli::decoder::{Decoder, DecodedImage};
//!
//! let image = Decoder::new().decode(&jpeg_data)?;
//! let pixels: &[u8] = image.pixels();
//! ```
//!
//! ## Feature Flags
//!
//! | Feature | Default | Description |
//! |---------|---------|-------------|
//! | `decoder` | No | Enable decoder API (prerelease) |
//! | `parallel` | No | Multi-threaded encoding via rayon |
//! | `cms-lcms2` | Yes | Color management via lcms2 |
//! | `cms-moxcms` | No | Pure Rust color management |
//! | `unsafe_simd` | No | Raw AVX2/SSE intrinsics (~10-20% faster) |
//!
//! ## Capabilities
//!
//! - **Baseline JPEG**: Standard 8-bit JPEG encoding
//! - **Progressive JPEG**: Multi-scan encoding (~3% smaller files)
//! - **XYB Color Space**: Perceptually optimized for better quality
//! - **Adaptive Quantization**: Content-aware bit allocation
//! - **16-bit / f32 Input**: High bit-depth source support
//! - **Streaming API**: Memory-efficient row-by-row encoding
//! - **Parallel Encoding**: Multi-threaded for large images

// Lint configuration is in workspace Cargo.toml [workspace.lints.clippy]
#![allow(missing_docs)]
#![allow(clippy::module_name_repetitions)]

// ============================================================================
// Public API Modules
// ============================================================================

/// JPEG encoder - public API.
///
/// Contains: `EncoderConfig`, `BytesEncoder`, `RgbEncoder`, `Error`, `Result`, etc.
pub mod encoder;

/// JPEG decoder - public API.
///
/// Contains: `Decoder`, `DecodedImage`, `Error`, `Result`, etc.
///
/// **Note:** The decoder is in prerelease and the API will have breaking changes.
/// Enable with the `decoder` feature flag.
#[cfg(feature = "decoder")]
pub mod decoder;

// ============================================================================
// Internal Implementation Modules
// ============================================================================

// Internal encoder implementation (exposed via test-utils for benchmarks)
#[cfg(feature = "test-utils")]
pub mod encode;
#[cfg(not(feature = "test-utils"))]
pub(crate) mod encode;

// Internal decoder implementation
#[cfg(all(feature = "decoder", feature = "test-utils"))]
pub mod decode;
#[cfg(all(feature = "decoder", not(feature = "test-utils")))]
pub(crate) mod decode;

// Internal shared error type (encoder/decoder have their own public errors)
pub(crate) mod error;

// Internal modules (exposed via test-utils for debugging tools and benchmarks)
#[cfg(feature = "test-utils")]
pub mod color;
#[cfg(not(feature = "test-utils"))]
pub(crate) mod color;

pub(crate) mod encode_simd;

#[cfg(feature = "test-utils")]
pub mod entropy;
#[cfg(not(feature = "test-utils"))]
pub(crate) mod entropy;

#[cfg(feature = "test-utils")]
pub mod foundation;
#[cfg(not(feature = "test-utils"))]
pub(crate) mod foundation;

#[cfg(feature = "test-utils")]
pub mod huffman;
#[cfg(not(feature = "test-utils"))]
pub(crate) mod huffman;

pub(crate) mod quant;

#[cfg(feature = "test-utils")]
pub mod types;
#[cfg(not(feature = "test-utils"))]
pub(crate) mod types;

// Test utilities - public when feature enabled for external test crates
#[cfg(feature = "test-utils")]
pub mod test_utils;
#[cfg(not(feature = "test-utils"))]
pub(crate) mod test_utils;

// Hybrid quantization (jpegli AQ + mozjpeg trellis)
#[cfg(feature = "experimental-hybrid-trellis")]
pub mod hybrid;
