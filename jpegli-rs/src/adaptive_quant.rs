//! Adaptive Quantization for jpegli - backward compatibility re-exports.
//!
//! This module has been refactored. The implementation is now in `quant::aq`.
//! This file provides re-exports for backward compatibility.

// Re-export everything from the new location
pub use crate::quant::aq::{
    compute_aq_strength_map, compute_aq_strength_map_impl, quant_field_to_aq_strength,
    quant_field_to_aq_strength_simd, AQStrengthMap,
};
