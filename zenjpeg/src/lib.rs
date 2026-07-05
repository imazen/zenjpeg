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
//! ```rust
//! use zenjpeg::decoder::Decoder;
//! use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout, Unstoppable};
//!
//! # fn main() -> Result<(), Box<dyn core::error::Error>> {
//! // (Make a tiny JPEG to decode.)
//! let pixels = [128u8; 16 * 16 * 3];
//! let mut enc = EncoderConfig::ycbcr(80.0, ChromaSubsampling::Quarter)
//!     .encode_from_bytes(16, 16, PixelLayout::Rgb8Srgb)?;
//! enc.push_packed(&pixels, Unstoppable)?;
//! let jpeg_data: Vec<u8> = enc.finish()?;
//!
//! // Decode with DoS limits + a stop token. `Unstoppable` never cancels;
//! // pass any `&impl zenjpeg::encoder::Stop` for user-initiated cancellation.
//! let decoded = Decoder::new()
//!     .max_pixels(120_000_000) // reject decompression bombs
//!     .decode(&jpeg_data, Unstoppable)?;
//!
//! let (width, height) = decoded.dimensions();
//! let rgb: &[u8] = decoded.pixels_u8().expect("u8 output (default OutputTarget::Srgb8)");
//! assert_eq!((width, height), (16, 16));
//! assert_eq!(rgb.len(), 16 * 16 * 3);
//! # Ok(())
//! # }
//! ```
//!
//! Decode failures carry a [`decoder::ErrorKind`] you can `match` to classify
//! them (e.g. for HTTP status mapping in a server):
//!
//! ```rust
//! use zenjpeg::decoder::{Decoder, ErrorKind};
//! use zenjpeg::encoder::Unstoppable;
//!
//! let not_a_jpeg = [0u8; 8];
//! if let Err(e) = Decoder::new().decode(&not_a_jpeg, Unstoppable) {
//!     match e.kind() {
//!         ErrorKind::ImageTooLarge { .. } => { /* 413 Payload Too Large */ }
//!         ErrorKind::AllocationFailed { .. } => { /* 500 Internal Server Error */ }
//!         _ => { /* 400 Bad Request — corrupt / unsupported input */ }
//!     }
//! }
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

// Image content analysis for `EncoderConfig::adaptive` lives in `zenanalyze`
// directly — the former `zenjpeg::analyze` re-export shim was removed so this
// codec no longer leaks version-specific `zenanalyze` types through its surface.
// Internal callers (`encode::adaptive`) use `zenanalyze::*`; the public surface
// is the `EncoderConfig::adaptive` constructor in `encode::adaptive`.

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

// ============================================================================
// One-shot convenience functions (crate root)
// ============================================================================
//
// Purely-additive free helpers that do the core job — pixels → JPEG and JPEG →
// RGB8 — in a single call with sane defaults, for callers who haven't read the
// builder docs. The encode input is a self-describing [`zenpixels::PixelSlice`]
// (format + dimensions + row stride ride with the pixels), so there is no
// redundant `width`/`height` to keep in sync and no buffer-length mismatch to
// guard against. Reach for the streaming builder API ([`encoder::EncoderConfig`]
// / [`decoder::Decoder`]) when you need chroma-subsampling control, XYB color,
// progressive scans, embedded ICC/EXIF/XMP, resource limits, or cooperative
// cancellation.

/// Encode a [`zenpixels::PixelSlice`] to a baseline JPEG in one call.
///
/// The slice is self-describing: it carries the pixel format (RGB8, RGBA8,
/// grayscale, 16-bit, f32, …), the dimensions, and the row stride, so there is
/// no separate `width`/`height` argument to keep in sync and no buffer-length
/// mismatch to guard against — strided slices are handled directly. `quality`
/// is `0..=100` (higher = better; `85` is a good web default). Encodes standard
/// YCbCr with 4:2:0 chroma — the most widely compatible JPEG configuration.
///
/// Requires the `zencodec` feature (the typed [`zenpixels`] encode path). For
/// 4:4:4 / 4:2:2 chroma, XYB color, progressive scans, embedded ICC/EXIF/XMP,
/// or cooperative cancellation, use the [`encoder::EncoderConfig`] builder.
///
/// # Errors
/// Returns an error if the slice's pixel format is not one the encoder
/// supports, plus any encode error bubbled up from the underlying pipeline.
///
/// ```
/// use zenjpeg::{decode_rgb8, encode};
/// use zenpixels::{PixelDescriptor, PixelSlice};
///
/// // A 16×16 RGB image — dims + stride + format ride with the pixels.
/// let (width, height) = (16u32, 16u32);
/// let rgb: Vec<u8> = (0..width * height)
///     .flat_map(|i| { let v = (i % 256) as u8; [v, 255 - v, 128] })
///     .collect();
/// let img = PixelSlice::new(&rgb, width, height, width as usize * 3, PixelDescriptor::RGB8_SRGB)?;
///
/// // Encode to a baseline JPEG at quality 85 — no separate width/height args.
/// let jpeg = encode(img, 85)?;
///
/// // Decode any JPEG back to tightly-packed RGB8 + dimensions.
/// let (pixels, w, h) = decode_rgb8(&jpeg)?;
///
/// assert_eq!((w, h), (width, height));
/// assert_eq!(pixels.len(), (width * height * 3) as usize); // JPEG is lossy — sizes match, bytes approximate
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[cfg(feature = "zencodec")]
pub fn encode(img: zenpixels::PixelSlice<'_>, quality: u8) -> crate::error::Result<Vec<u8>> {
    // `JpegEncoderConfig::ycbcr(..).encode(slice)` runs the default config → job
    // → encode path: it pulls RGB8 rows from the (possibly strided) slice via
    // `zenpixels-convert` and feeds the streaming encoder. 4:2:0 chroma matches
    // the most widely compatible JPEG configuration.
    let output = crate::codec::JpegEncoderConfig::ycbcr(
        f32::from(quality),
        crate::encoder::ChromaSubsampling::Quarter,
    )
    .encode(img)?;
    Ok(output.into_vec())
}

/// Decode a JPEG (any color space) to tightly-packed 8-bit RGB in one call.
///
/// Returns `(rgb, width, height)` where `rgb` is exactly `width * height * 3`
/// bytes (`R, G, B` per pixel, no stride padding, sRGB). Grayscale, YCbCr and
/// CMYK / YCCK sources are all normalized to 8-bit RGB.
///
/// For f32 / linear-light output, resource limits (decompression-bomb
/// protection), strictness control, preserved metadata, or cooperative
/// cancellation, use the [`decoder::Decoder`] builder.
///
/// # Errors
/// Returns an error if `jpeg` is not a valid / supported JPEG, or a resource
/// limit is exceeded.
///
/// ```
/// use zenjpeg::decode_rgb8;
/// use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};
///
/// // A 16×16 RGB image, tightly packed (width * height * 3 bytes).
/// let (width, height) = (16u32, 16u32);
/// let rgb: Vec<u8> = (0..width * height)
///     .flat_map(|i| { let v = (i % 256) as u8; [v, 255 - v, 128] })
///     .collect();
///
/// // Encode a baseline JPEG via the builder, then round-trip it back to RGB8.
/// let jpeg = EncoderConfig::ycbcr(85u8, ChromaSubsampling::Quarter)
///     .encode_bytes(&rgb, width, height, PixelLayout::Rgb8Srgb)?;
/// let (pixels, w, h) = decode_rgb8(&jpeg)?;
///
/// assert_eq!((w, h), (width, height));
/// assert_eq!(pixels.len(), (width * height * 3) as usize); // JPEG is lossy — sizes match, bytes approximate
/// # Ok::<(), zenjpeg::encoder::EncodeError>(())
/// ```
pub fn decode_rgb8(jpeg: &[u8]) -> crate::error::Result<(Vec<u8>, u32, u32)> {
    // `PixelFormat::Rgb` makes the decoder emit 3-channel RGB for every input
    // (grayscale is expanded, CMYK/YCCK is converted); the default
    // `OutputTarget::Srgb8` keeps it u8.
    let decoded = crate::decoder::Decoder::new()
        .output_format(crate::types::PixelFormat::Rgb)
        .decode(jpeg, enough::Unstoppable)?;
    let (width, height) = decoded.dimensions();
    let rgb = decoded
        .into_pixels_u8()
        .ok_or_else(|| crate::error::Error::internal("decode produced no u8 pixels"))?;
    Ok((rgb, width, height))
}
