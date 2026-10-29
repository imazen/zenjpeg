//! FFI bindings to the jpegli C++ reference implementation.
//!
//! When the `internal/jpegli-cpp` submodule is not checked out, build.rs
//! emits `missing_jpegli_cpp` and this crate compiles EMPTY instead of
//! panicking the whole workspace build. Every consumer in zenjpeg is
//! gated behind `--features __ffi-tests`, so plain `cargo test` works
//! from a bare clone; building WITH `__ffi-tests` and no submodule fails
//! loudly at the consumers' `use jpegli_internals_sys::...` imports.
//! Populate with: `git submodule update --init --recursive`.

#[cfg(not(missing_jpegli_cpp))]
mod real;
#[cfg(not(missing_jpegli_cpp))]
pub use real::*;
