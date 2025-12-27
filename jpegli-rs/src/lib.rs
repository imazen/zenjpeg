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
//!     .quality(Quality::from_distance(1.0))
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

// Layer 0: Constants and types
pub mod consts;
pub mod types;

// Layer 1: Pure math functions
pub mod huffman;
pub mod huffman_opt;
pub mod quant;

// Layer 2: Transforms
pub mod color;
pub mod dct;
pub mod idct;
pub mod tone_mapping;
pub mod transfer_functions;
pub mod xyb;

// Layer 3: Bitstream I/O
pub mod bitstream;
pub mod scan_script;

// Layer 4: Stateful components
pub mod entropy;

// Layer 5-6: Pipelines
pub mod decode;
pub mod encode;

// Simplified adaptive quantization (NOT C++ matching - uses arbitrary thresholds)
pub mod simplified_quant;

// Adaptive quantization (placeholder for C++ matching implementation)
pub mod adaptive_quant;

// Quality metrics - re-export from the butteraugli-oxide crate
pub use butteraugli_oxide;

// Error types
pub mod error;

// Safe allocation helpers
pub mod alloc;

// ICC color management
pub mod icc;

// Test utilities (available for tests and examples)
#[cfg(any(test, feature = "test-utils"))]
pub mod test_utils;

// ============================================================================
// Re-exports for public API
// ============================================================================

pub use error::{Error, Result};
pub use types::{ColorSpace, PixelFormat, SampleDepth};

// Encoder API
pub use encode::{Encoder, EncoderConfig};

// Decoder API
pub use decode::{Decoder, DecoderConfig};

// Quality settings
pub use quant::{Quality, QuantTable};
