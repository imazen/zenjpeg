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
pub mod simd_types;

// Re-export commonly used items at module level
