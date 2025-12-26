//! Forked jpegli encoder for experimental improvements
//!
//! This is a fork of jpegli-rs code to allow experimentation with
//! compression improvements while maintaining the full perceptual
//! quality pipeline (XYB color space, butteraugli-based AQ, etc.)

// Core types and constants
pub mod consts;
pub mod types;
pub mod error;

// Memory allocation
pub mod alloc;

// Transforms
pub mod color;
pub mod dct;
pub mod xyb;
pub mod tone_mapping;
pub mod transfer_functions;

// Quantization
pub mod quant;
pub mod adaptive_quant;
pub mod simplified_quant;

// Entropy coding
pub mod bitstream;
pub mod huffman;
pub mod huffman_opt;
pub mod entropy;
pub mod scan_script;

// ICC color management
pub mod icc;

// Main encoder
pub mod encode;

// Re-exports for convenience
pub use encode::{Encoder, EncoderConfig};
pub use error::{Error, Result};
pub use quant::Quality;
pub use types::{ColorSpace, PixelFormat, SampleDepth};
