//! Trellis and hybrid quantization for optimal rate-distortion.
//!
//! This module consolidates all trellis/mozjpeg-style quantization code:
//!
//! - **`ac`**: AC coefficient trellis (Viterbi DP) - the core mozjpeg innovation
//! - **`dc`**: DC coefficient trellis optimization
//! - **`rate`**: Huffman rate estimation tables
//! - **`compat`**: mozjpeg-compatible [`TrellisConfig`] and [`TrellisSpeedMode`] types
//! - **`hybrid`**: Combined jpegli AQ + mozjpeg trellis ([`HybridConfig`], [`HybridQuantContext`])
//!
//! # Deletability
//!
//! This entire module can be removed (behind a feature flag) without affecting
//! the core jpegli encoder. When disabled, the encoder falls back to standard
//! zero-bias quantization.

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
pub use compat::{TrellisConfig, TrellisSpeedMode};

// Re-export hybrid types
pub use hybrid::HybridConfig;

// Re-export hybrid core functions

// Encoder integration helpers (pub(crate) only)
pub(crate) use hybrid::HybridQuantContext;
