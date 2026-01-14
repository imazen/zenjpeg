//! Low-level foundation utilities.
//!
//! This module contains fundamental utilities used throughout the codebase:
//! - `consts`: JPEG markers, zigzag tables, quantization matrices
//! - `alloc`: Safe allocation helpers for DoS protection
//! - `bitstream`: Low-level bit I/O operations

pub mod aligned_alloc;
pub mod alloc;
pub mod bitstream;
pub mod consts;
pub mod simd_targets;
pub mod simd_types;

// Re-export commonly used items at module level
pub use alloc::{
    checked_size_2d, try_alloc_dct_blocks_tracked, try_alloc_filled, try_alloc_zeroed_f32,
    try_alloc_zeroed_f32_tracked, try_with_capacity_tracked, validate_dimensions, AllocationStats,
    MemoryTracker, DEFAULT_MAX_MEMORY, DEFAULT_MAX_PIXELS, JPEG_MAX_DIMENSION,
    MAX_ICC_PROFILE_SIZE, MAX_SCANS,
};

pub use consts::{
    // DCT constants
    DCT_BLOCK_SIZE,
    DCT_SIZE,
    // JPEG limits
    DC_ALPHABET_SIZE,
    HUFFMAN_ALPHABET_SIZE,
    HUFFMAN_MAX_BIT_LENGTH,
    // Tables
    JPEG_NATURAL_ORDER,
    JPEG_PRECISION,
    JPEG_ZIGZAG_ORDER,
    // JPEG markers
    MARKER_APP0,
    MARKER_APP14,
    MARKER_APP2,
    MARKER_COM,
    MARKER_DHT,
    MARKER_DQT,
    MARKER_DRI,
    MARKER_EOI,
    MARKER_SOF0,
    MARKER_SOF1,
    MARKER_SOF2,
    MARKER_SOI,
    MARKER_SOS,
    MAX_COMPONENTS,
    MAX_DIM_PIXELS,
    MAX_HUFFMAN_CODES,
    MAX_HUFFMAN_TABLES,
    MAX_QUANT_TABLES,
};

pub use bitstream::{BitReader, BitWriter};
