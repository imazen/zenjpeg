//! # jpegli
//!
//! Rust port of jpegli - an improved JPEG encoder and decoder.
//!
//! jpegli provides enhanced compression quality compared to standard JPEG
//! through advanced quantization, optional XYB color space, and other
//! perceptual optimizations.
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use jpegli::encoder::{EncoderConfig, PixelLayout, Result};
//! use enough::Unstoppable;
//!
//! // Encode RGB to JPEG
//! let config = EncoderConfig::new().quality(85);
//! let mut enc = config.encode_from_bytes(640, 480, PixelLayout::Rgb8Srgb)?;
//! enc.push_packed(&rgb_pixels, Unstoppable)?;
//! let jpeg_data = enc.finish()?;
//!
//! // Decode JPEG to RGB
//! let image = jpegli::decoder::Decoder::new().decode(&jpeg_data)?;
//! let pixels: &[u8] = image.pixels();
//! ```
//!
//! ## Features
//!
//! - **Baseline JPEG**: Standard 8-bit JPEG encoding/decoding
//! - **Progressive JPEG**: Multi-scan progressive encoding
//! - **XYB Color Space**: Perceptually optimized color space for better quality
//! - **Adaptive Quantization**: Content-aware quantization for improved detail
//! - **16-bit Support**: High bit-depth input/output
//! - **Parallel Encoding**: Multi-threaded encoding (with `parallel` feature)

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
pub mod decoder;

// ============================================================================
// Internal Implementation Modules
// ============================================================================

// Internal encoder implementation
pub(crate) mod encode;

// Internal decoder implementation
pub(crate) mod decode;

// Internal shared error type (encoder/decoder have their own public errors)
pub(crate) mod error;

// Internal modules
pub(crate) mod color;
pub(crate) mod foundation;
pub(crate) mod quant;
pub(crate) mod types;
pub(crate) mod encode_simd;
pub(crate) mod entropy;
pub(crate) mod huffman;

// Test utilities - public when feature enabled for external test crates
#[cfg(feature = "test-utils")]
pub mod test_utils;
#[cfg(not(feature = "test-utils"))]
pub(crate) mod test_utils;

// Hybrid quantization (jpegli AQ + mozjpeg trellis)
#[cfg(feature = "experimental-hybrid-trellis")]
pub mod hybrid;
