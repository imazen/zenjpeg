//! Huffman coding module.
//!
//! This module contains Huffman table types, encoding, and optimization:
//! - `encode`: Huffman encoding functions and tables
//! - `classic`: Optimal Huffman table generation (mozjpeg-style algorithms)
//! - `types`: Additional Huffman types and comparisons
//! - `optimize`: Two-pass Huffman optimization with frequency counting and clustering
//! - `trained`: Pre-trained Huffman tables for streaming encoding

pub(crate) mod builtin_tables;
pub mod classic;
pub mod encode;
pub mod optimize;
pub mod trained;
pub mod types;

// Re-export commonly used items from encode (the main huffman.rs functionality)
pub use encode::{
    HuffmanEncodeTable, STD_AC_CHROMINANCE_BITS, STD_AC_CHROMINANCE_VALUES, STD_AC_LUMINANCE_BITS,
    STD_AC_LUMINANCE_VALUES, STD_DC_CHROMINANCE_BITS, STD_DC_CHROMINANCE_VALUES,
    STD_DC_LUMINANCE_BITS, STD_DC_LUMINANCE_VALUES, build_code_lengths,
};

// Decoder-only exports
#[cfg(feature = "decoder")]
pub use encode::HuffmanDecodeTable;

// Re-export from classic

// Re-export from types

// Re-export from optimize (new types from huffman_opt.rs refactor)
