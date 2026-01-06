//! Huffman table optimization for JPEG encoding.
//!
//! **This module is deprecated.** Use `crate::huffman::optimize` instead.
//!
//! This module re-exports types from `huffman::optimize` for backward compatibility.

// Re-export everything from the new location
pub use crate::huffman::optimize::{
    cluster_histograms, ClusterResult, ContextConfig, FrequencyCounter, OptimizedHuffmanTables,
    OptimizedTable, ProgressiveTokenBuffer, RefToken, ScanTokenInfo, Token, TokenBuffer,
};
