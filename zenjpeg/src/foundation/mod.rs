//! Low-level foundation utilities.
//!
//! This module contains fundamental utilities used throughout the codebase:
//! - `consts`: JPEG markers, zigzag tables, quantization matrices
//! - `alloc`: Safe allocation helpers for DoS protection
//! - `bitstream`: Low-level bit I/O operations
//! - `instrumented_vec`: Vec wrapper for allocation profiling (feature-gated)

pub mod aligned_alloc;
pub mod alloc;
pub mod bitstream;
pub mod consts;
pub mod instrumented_vec;
pub mod simd_types;

// Re-export commonly used items at module level
