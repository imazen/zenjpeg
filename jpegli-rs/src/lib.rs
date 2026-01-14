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
#[doc(hidden)]
pub mod encode;

// Internal decoder implementation
#[doc(hidden)]
pub mod decode;

// Internal shared error type (encoder/decoder have their own public errors)
#[doc(hidden)]
pub mod error;

// Internal modules - hidden but accessible for tests/examples
#[doc(hidden)]
pub mod color;
#[doc(hidden)]
pub mod foundation;
#[doc(hidden)]
pub mod quant;
#[doc(hidden)]
pub mod types;
#[doc(hidden)]
pub mod encode_simd;
#[doc(hidden)]
pub mod entropy;
#[doc(hidden)]
pub mod huffman;
#[doc(hidden)]
pub mod test_utils;

// Hybrid quantization (jpegli AQ + mozjpeg trellis)
#[cfg(feature = "experimental-hybrid-trellis")]
pub mod hybrid;

// ============================================================================
// Internal re-exports for backward compatibility
// ============================================================================

#[doc(hidden)]
pub use encode::chroma;
#[doc(hidden)]
pub use encode::dct;
#[doc(hidden)]
pub use encode::scan_script;

// ============================================================================
// Legacy/Internal Types (hidden from docs)
// ============================================================================

#[doc(hidden)]
pub use types::{
    ChromaDownsampling as LegacyChromaDownsampling, ColorSpace, HuffmanMethod, JpegMode,
    PixelFormat, Subsampling,
};
#[doc(hidden)]
#[deprecated(since = "0.5.0", note = "Use DownsamplingMethod instead")]
pub use types::ChromaDownsampling;

#[doc(hidden)]
pub use quant::Quality as LegacyQuality;
#[doc(hidden)]
pub use quant::quality_conversion::{QualityComparisonMetric, QualityConversion};
#[doc(hidden)]
pub use quant::{CustomQuantMatrices, QuantTable};

#[doc(hidden)]
pub use foundation::AllocationStats;
#[doc(hidden)]
pub use foundation::{aligned_alloc, alloc, bitstream, consts, simd_types};

#[doc(hidden)]
pub use huffman::classic as huffman_classic;
#[doc(hidden)]
pub use huffman::types as huffman_types;

#[doc(hidden)]
pub use decode::idct;
#[doc(hidden)]
pub use decode::idct_int;

#[doc(hidden)]
pub use color::icc;
#[doc(hidden)]
pub use color::xyb;

#[doc(hidden)]
pub use imgref::{Img, ImgRef, ImgRefMut, ImgVec};
