//! # jpegli
//!
//! Rust port of jpegli - an improved JPEG encoder and decoder.
//!
//! jpegli provides enhanced compression quality compared to standard JPEG
//! through advanced quantization, optional XYB color space, and other
//! perceptual optimizations.
//!
//! ## Features
//!
//! - **Baseline JPEG**: Standard 8-bit JPEG encoding/decoding
//! - **Progressive JPEG**: Multi-scan progressive encoding
//! - **XYB Color Space**: Perceptually optimized color space for better quality
//! - **Adaptive Quantization**: Content-aware quantization for improved detail
//! - **16-bit Support**: High bit-depth input/output
//!
//! ## Example
//!
//! ```rust,ignore
//! use jpegli::{Encoder, ColorSpace, Quality};
//!
//! let pixels: &[u8] = &[/* RGB data */];
//! let encoder = Encoder::new()
//!     .width(640)
//!     .height(480)
//!     .color_space(ColorSpace::Rgb)
//!     .jpegli_quality(Quality::from_distance(1.0))
//!     .build()?;
//!
//! let jpeg_data = encoder.encode(pixels)?;
//! ```

// Lint configuration is in workspace Cargo.toml [workspace.lints.clippy]
#![allow(missing_docs)]
#![allow(clippy::module_name_repetitions)]

// ============================================================================
// Module structure
// ============================================================================

// Public modules (stable API)
pub mod decode;
pub mod encode;
pub mod error;
pub mod pixel;
pub mod quality_conversion;
pub mod quant;
pub mod types;

// Internal modules - NOT part of the stable public API.
// These are hidden from documentation and may change without notice.
// Use at your own risk.
#[doc(hidden)]
pub mod adaptive_quant;
#[doc(hidden)]
pub mod adaptive_quant_simd;
#[doc(hidden)]
pub mod alloc;
#[doc(hidden)]
pub mod bitstream;
#[doc(hidden)]
pub mod chroma;
#[doc(hidden)]
pub mod color;
#[doc(hidden)]
pub mod consts;
#[doc(hidden)]
pub mod dct;
#[doc(hidden)]
pub mod encode_simd;
#[doc(hidden)]
pub mod entropy;
#[doc(hidden)]
pub mod huffman;
#[doc(hidden)]
pub mod huffman_classic;
#[doc(hidden)]
pub mod huffman_opt;
#[doc(hidden)]
pub mod huffman_types;
#[doc(hidden)]
pub mod icc;
#[doc(hidden)]
pub mod idct;
#[doc(hidden)]
pub mod scan_script;
#[doc(hidden)]
pub mod simplified_quant;
#[doc(hidden)]
pub mod tone_mapping;
#[doc(hidden)]
pub mod transfer_functions;
#[doc(hidden)]
pub mod xyb;

// Hybrid quantization (jpegli AQ + mozjpeg trellis)
#[cfg(feature = "experimental-hybrid-trellis")]
pub mod hybrid;
#[cfg(feature = "experimental-hybrid-trellis")]
pub mod hybrid_config;

// Test utilities (available for tests and examples)
// Hidden from docs but always available for integration tests
#[doc(hidden)]
pub mod test_utils;

// ============================================================================
// Re-exports for public API
// ============================================================================

pub use error::{Error, Result};
pub use types::{
    ChromaConversion, ColorSpace, HuffmanMethod, JpegMode, OutputDataType, PixelFormat,
    SampleDepth, Subsampling,
};

// Encoder API
pub use encode::{Encoder, EncoderConfig};

// Decoder API
pub use decode::{DecodedImage, DecodedImageF32, Decoder, DecoderConfig};

// Quality settings
pub use quality_conversion::{QualityComparisonMetric, QualityConversion};
pub use quant::{Quality, QuantTable};

// Pixel types for typed encode/decode API
pub use pixel::{Gray16, Gray8, Pixel, RGB16, RGB8, RGBA16, RGBA8};

// Re-export imgref for convenient image buffer handling
pub use imgref::{Img, ImgRef, ImgRefMut, ImgVec};
