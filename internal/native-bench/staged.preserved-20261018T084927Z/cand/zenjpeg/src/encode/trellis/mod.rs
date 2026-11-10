//! Trellis and hybrid quantization for optimal rate-distortion.
//!
//! This module consolidates all trellis/mozjpeg-style quantization code:
//!
//! - **`ac`**: AC coefficient trellis (Viterbi DP) - the core mozjpeg innovation
//! - **`dc`**: DC coefficient trellis optimization
//! - **`rate`**: Huffman rate estimation tables
//! - **`compat`**: `TrellisConfig`, `TrellisSpeedMode`, and `AqCoupling` types
//! - **`hybrid`**: quantization engine (`TrellisContext`) with optional
//!   per-block AQ→lambda coupling
//!
//! Trellis is always compiled but data-gated: with `EncoderConfig::trellis`
//! unset the encoder uses standard zero-bias quantization.

pub mod ac;
#[allow(dead_code)]
pub mod dc;
pub mod rate;

pub mod compat;
pub mod hybrid;

// Re-export main trellis types
pub use ac::trellis_quantize_block;
pub use rate::RateTable;

#[allow(unused_imports)]
pub use dc::{dc_trellis_optimize, dc_trellis_optimize_indexed, simple_quantize_block};

// Re-export compat types
pub use compat::{AqCoupling, TrellisConfig, TrellisSpeedMode};

// Encoder integration helpers (pub(crate) only)
pub(crate) use hybrid::TrellisContext;
