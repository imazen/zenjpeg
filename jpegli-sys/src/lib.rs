//! FFI bindings to jpegli C++ for comparison testing.
//!
//! This crate provides raw FFI bindings to the C++ jpegli library
//! for testing the Rust port against the original implementation.

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

use libc::c_void;

/// Opaque handle to a jpegli encoder context.
pub type jpegli_encoder = *mut c_void;

/// Opaque handle to a jpegli decoder context.
pub type jpegli_decoder = *mut c_void;

// TODO: Add FFI function declarations once we build the C++ export DLL
// For now, this is a placeholder to allow the workspace to compile.

#[cfg(test)]
mod tests {
    #[test]
    fn test_placeholder() {
        // Placeholder test
        assert!(true);
    }
}
