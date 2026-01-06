//! Hybrid quantization: jpegli AQ + mozjpeg trellis.
//!
//! This module combines jpegli's adaptive quantization (WHERE to spend bits)
//! with mozjpeg's trellis quantization (HOW to spend bits).
//!
//! Requires the `experimental-hybrid-trellis` feature.

pub mod config;
pub mod core;

// Re-export main types for convenience
pub use config::{
    estimate_hybrid_improvement, should_use_hybrid, HybridConfig, SweepConfig, AQ_MEAN_THRESHOLD,
};
pub use core::{dct_f32_to_i32, hybrid_quantize_block_simple, scale_quant_by_aq};

#[cfg(feature = "experimental-hybrid-trellis")]
pub use core::{hybrid_quantize_block, StandardHuffmanTables};
