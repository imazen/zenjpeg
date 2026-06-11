#![forbid(unsafe_code)]
// std is unconditionally required — no viable no_std path (752 errors without it).

//! # zenjpeg
//!
//! Pure Rust JPEG encoder and decoder with perceptual optimizations.
//!
//! Provides enhanced compression quality compared to standard JPEG through
//! adaptive quantization, optional XYB color space, and other perceptual
//! optimizations.
//!
//! ## Feature Requirements
//!
//! > **Important:** The decoder requires a feature flag. Add to `Cargo.toml`:
//! > ```toml
//! > [dependencies]
//! > zenjpeg = { version = "0.6", features = ["decoder"] }
//! > ```
//!
//! **Available features:**
//! - `decoder` - Enable JPEG decoding (required for `zenjpeg::decoder` module)
//! - `parallel` - Multi-threaded encoding via rayon

//! - `moxcms` - ICC color management via moxcms (pure Rust)
//! - `ultrahdr` - UltraHDR gain map support
//!
//! See [Feature Flags](#feature-flags) section below for details.
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use zenjpeg::encoder::{EncoderConfig, ChromaSubsampling, PixelLayout, Unstoppable};
//!
//! // Create reusable config (quality + color mode in constructor)
//! let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
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
//! use zenjpeg::encoder::{
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
//! | [`encoder::EncoderConfig::encode_from_bytes`] | `&[u8]` | Raw byte buffers |
//! | [`encoder::EncoderConfig::encode_from_rgb`] | `rgb` crate types | Type-safe pixels |
//! | [`encoder::EncoderConfig::encode_from_ycbcr_planar`] | [`YCbCrPlanes`](encoder::YCbCrPlanes) | Video pipelines |
//!
//! ### Configuration Options
//!
//! ```rust,ignore
//! // YCbCr mode (standard JPEG - most compatible)
//! let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
//!     .progressive(true)                        // Progressive JPEG (~3% smaller)
//!     .sharp_yuv(true)                          // Better color edges (~3x slower)
//!     .icc_profile(bytes);                      // Attach ICC profile
//!
//! // XYB mode (perceptual color space - better quality)
//! let config = EncoderConfig::xyb(85, XybSubsampling::BQuarter)
//!     .progressive(true);
//!
//! // Grayscale mode
//! let config = EncoderConfig::grayscale(85);
//!
//! // Quality can also use enum variants:
//! let config = EncoderConfig::ycbcr(Quality::ApproxSsim2(90.0), ChromaSubsampling::None);
//! let config = EncoderConfig::ycbcr(Quality::ApproxButteraugli(1.0), ChromaSubsampling::Quarter);
//! ```
//!
//! ## Decoder API
//!
//! The decoder is in prerelease (always compiled; the API will have
//! breaking changes).
//!
//! ```rust,ignore
//! use zenjpeg::decoder::{Decoder, DecodedImage};
//!
//! let image = Decoder::new().decode(&jpeg_data)?;
//! let pixels: &[u8] = image.pixels();
//! ```
//!
//! ## Feature Flags
//!
//! | Feature | Default | Description | When to Use |
//! |---------|---------|-------------|-------------|
//! | `decoder` | — | Deprecated no-op (the decoder is always compiled) | Kept so `zenjpeg/decoder` doesn't break downstream |
//! | `std` | — | Legacy (std is always required) | Kept so `zenjpeg/std` doesn't break downstream |
//! | `moxcms` | ❌ No | ICC color management via moxcms (pure Rust) | Color-managed decode pipelines |
//! | `parallel` | ❌ No | Multi-threaded encoding via rayon | Large images (4K+), server workloads |
//! | `ultrahdr` | ❌ No | UltraHDR HDR gain map support | Encoding/decoding HDR JPEGs |
//! | `trellis` | ✅ Yes | Trellis quantization (mozjpeg-style) | Keep enabled for best compression |
//! | `yuv` | ✅ Yes | SharpYUV chroma downsampling | Keep enabled for quality |
//!
//! ### Common Configurations
//!
//! ```toml
//! # Decode + encode (most common)
//! zenjpeg = "0.8"
//!
//! # High-performance server
//! zenjpeg = { version = "0.8", features = ["parallel"] }
//!
//! # UltraHDR support
//! zenjpeg = { version = "0.8", features = ["ultrahdr"] }
//! ```
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

extern crate alloc;

// Error tracing with location tracking
whereat::define_at_crate_info!(path = "zenjpeg/");

// ============================================================================
// Public API Modules
// ============================================================================

/// Fast Gaussian blur preprocessing for improved JPEG compression.
///
/// Applying a mild blur (σ=0.4) before encoding reduces file size ~5% with
/// negligible perceptual quality loss. This module provides zero-dependency
/// blur optimized for this use case.
pub mod blur;

/// JPEG encoder - public API.
///
/// Contains: `EncoderConfig`, `BytesEncoder`, `RgbEncoder`, `Error`, `Result`, etc.
pub mod encoder;

/// Resource estimation heuristics for encoding and decoding.
///
/// Provides min/typical/max estimates for peak memory and time.
pub mod heuristics;

// Codec-analysis metrics (BBS block-boundary score, RD-curve and
// BD-rate harness, sweep orchestration) moved to the internal
// `zenjpeg-bench-utils` crate. They are benchmark infrastructure, not
// part of the shipped encoder library.

/// JPEG encoder detection and quality estimation.
///
/// Identifies which encoder produced a JPEG, estimates its quality level,
/// and extracts structural metadata from header-only parsing (~500 bytes).
pub mod detect;

/// JPEG decoder - public API.
///
/// Contains: `Decoder`, `DecodeResult`, `Error`, `Result`, etc.
///
/// **Note:** The decoder is in prerelease and the API will have breaking changes.
pub mod decoder;

/// UltraHDR support - HDR gain map encoding and decoding.
///
/// Provides integration with `ultrahdr-core` for:
/// - HDR to SDR tonemapping
/// - Gain map computation and application
/// - XMP metadata generation and parsing
/// - Adaptive tonemapper for re-encoding
///
/// Enable with the `ultrahdr` feature flag.
#[cfg(feature = "ultrahdr")]
pub mod ultrahdr;

/// Public JPEG container primitives: zero-copy marker iteration, image
/// boundary detection, and (incrementally, in follow-up commits) MPF
/// parsing, XMP helpers, and the ISO 21496-1 JPEG APP2 envelope.
///
/// This is the single canonical source for "walk a JPEG marker stream";
/// prior scattered implementations in `ultrahdr-core`, `ultrahdr-rs`,
/// and this crate's internal `detect::scanner` consolidate on top of it.
///
/// See [`container::marker`] for the iterator and helpers.
pub mod container;

// ============================================================================
// Internal Implementation Modules
// ============================================================================

// Internal encoder implementation (exposed via test-utils for benchmarks)
#[cfg(feature = "__test-utils")]
pub mod encode;
#[cfg(not(feature = "__test-utils"))]
pub(crate) mod encode;

// Internal decoder implementation
#[cfg(feature = "__test-utils")]
pub mod decode;
#[cfg(not(feature = "__test-utils"))]
pub(crate) mod decode;

// Internal shared error type (encoder/decoder have their own public errors)
pub(crate) mod error;

// Internal modules (exposed via test-utils for debugging tools and benchmarks)
#[cfg(feature = "__test-utils")]
pub mod color;
#[cfg(not(feature = "__test-utils"))]
pub(crate) mod color;

pub(crate) mod encode_simd;

#[cfg(feature = "__test-utils")]
pub mod entropy;
#[cfg(not(feature = "__test-utils"))]
pub(crate) mod entropy;

#[cfg(feature = "__test-utils")]
pub mod foundation;
#[cfg(not(feature = "__test-utils"))]
pub(crate) mod foundation;

#[cfg(feature = "__test-utils")]
pub mod huffman;
#[cfg(not(feature = "__test-utils"))]
pub(crate) mod huffman;

// Make quant accessible for benchmarks when test-utils enabled
#[cfg(feature = "__test-utils")]
#[doc(hidden)]
pub mod quant;
#[cfg(not(feature = "__test-utils"))]
pub(crate) mod quant;

#[cfg(feature = "__test-utils")]
pub mod types;
#[cfg(not(feature = "__test-utils"))]
pub(crate) mod types;

// Test utilities - only compiled when feature enabled (requires std)
#[cfg(feature = "__test-utils")]
pub mod test_utils;

// Post-decode deblocking filters
pub mod deblock;

// Lossless JPEG transforms
pub mod lossless;

// Layout pipeline: lossless transforms + lossy decode→resize→encode
#[cfg(feature = "layout")]
pub mod layout;

// Profiling instrumentation (zero-cost when disabled)
pub mod profile;

// JPEG→JPEG recompression to a target perceptual (zensim Profile A)
// quality with no size regression. Moved in from the standalone
// `zenjpeg-recompress` crate (2026-05-29); reaches the codec's
// `pub(crate)` internals directly. Behind the `recompress` feature so
// base builds are unaffected; the closed loop's `zensim` measurement is
// further gated behind `recompress-iqa`.
#[cfg(feature = "recompress")]
pub mod recompress;

// Image content analysis for `EncoderConfig::adaptive`.
//
// Internal by default — the analyzer output and tier helpers are
// pub(crate); the public surface is the `EncoderConfig::adaptive`
// constructor in `encode::adaptive`.
//
// Exposed as `pub` via `__test-utils` so coefficient's parity
// harness can drive `analyze::analyze_rgb8` directly. Don't rely on
// this surface from a normal consumer.
#[cfg(feature = "__test-utils")]
#[doc(hidden)]
pub mod analyze;
#[cfg(not(feature = "__test-utils"))]
pub(crate) mod analyze;

// zencodec trait implementations
#[cfg(feature = "zencodec")]
mod codec;
#[cfg(feature = "zencodec")]
pub use codec::{
    CmykHandling,
    JpegDecodeJob,
    JpegDecoder,
    JpegDecoderConfig,
    // Backwards compat aliases
    JpegDecoding,
    JpegEncodeJob,
    JpegEncoder,
    JpegEncoderConfig,
    JpegEncoding,
    JpegStreamingDecoder,
};

// zennode pipeline node definitions (EncodeJpeg, DecodeJpeg)
// #[cfg(feature = "zennode")]
// pub mod zennode_defs;
