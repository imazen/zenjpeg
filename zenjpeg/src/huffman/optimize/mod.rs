//! Huffman table optimization for JPEG encoding.
//!
//! This module implements optimal Huffman table generation from symbol frequency
//! counts, following Section K.2 of the JPEG specification.
//!
//! # Algorithm Comparison: mozjpeg vs jpegli C++
//!
//! This implementation uses the **mozjpeg/libjpeg algorithm** (Section K.2), not the
//! jpegli C++ algorithm. Both produce valid Huffman codes, but differ in approach:
//!
//! ## mozjpeg/libjpeg (this implementation)
//!
//! ```text
//! 1. Classic Huffman merge with `others[]` chain tracking
//! 2. Build tree bottom-up, tracking code lengths via chain traversal
//! 3. Limit to 16 bits using Section K.2 tree manipulation:
//!    - Move symbols from depth > 16 up the tree
//!    - Split shorter codes to maintain valid prefix-free property
//! 4. Remove pseudo-symbol 256 from final table
//! ```
//!
//! **Pros**: Simpler (~100 lines), follows JPEG spec exactly, well-understood
//! **Cons**: O(n²) merge loop (fine for n ≤ 257)
//!
//! ## jpegli C++ (`CreateHuffmanTree` in huffman.cc)
//!
//! ```text
//! 1. Sort symbols by frequency, use two-pointer merge with sentinels
//! 2. If max depth > limit, retry with count_limit *= 2
//!    (artificially boosts low-frequency symbols to reduce tree depth)
//! 3. More complex but potentially faster for large alphabets
//! ```
//!
//! **Pros**: May be faster for large n due to sorted merge, single-pass depth limiting
//! **Cons**: More complex (~150 lines), non-standard retry approach
//!
//! ## Validation Results
//!
//! Tested against 122 C++ jpegli test cases:
//! - **100/122 exact match** (82%)
//! - **22 cases**: mozjpeg produces 1 bit LESS total (better compression)
//! - **0 cases**: mozjpeg worse than jpegli
//!
//! The differences arise from tie-breaking: when two symbols have equal frequency,
//! the algorithms may order them differently, producing different but equally valid trees.
//!
//! # Module Structure
//!
//! - `frequency`: Frequency counting and table generation
//! - `tokens`: Token types for two-pass encoding
//! - `cluster`: Histogram clustering for table optimization
//! - `progressive`: Progressive JPEG tokenization buffer

pub mod cluster;
pub mod frequency;
pub mod progressive;
pub mod tokens;

// Re-export commonly used types
pub use cluster::ContextConfig;
pub use frequency::{FrequencyCounter, HuffmanTableSet, OptimizedTable};
pub use progressive::ProgressiveTokenBuffer;
pub use tokens::{ScanTokenInfo, Token};
