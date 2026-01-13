//! Shared types for the public API.
//!
//! This module re-exports types from internal modules to provide
//! a stable public interface.

// Core types from types module
pub use crate::types::{ChromaDownsampling, ColorSpace, JpegMode, PixelFormat, Subsampling};

// Quality from quant module
pub use crate::quant::Quality;

// v2 types that will become the standard
pub use crate::encode::v2::{ChromaSubsampling, ColorMode, DownsamplingMethod, PixelLayout};
