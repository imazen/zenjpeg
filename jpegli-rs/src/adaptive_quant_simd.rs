//! SIMD-optimized adaptive quantization functions - backward compatibility re-exports.
//!
//! This module has been refactored. The implementation is now in `quant::aq::simd`.
//! This file provides re-exports for backward compatibility.

// Re-export everything from the new location
pub use crate::quant::aq::simd::{
    compute_pre_erosion_simd, fuzzy_erosion_simd, gamma_modulation_sum_8x8, hf_modulation_sum_8x8,
    masking_sqrt_x8, per_block_modulations_row, per_block_modulations_simd, pre_erosion_pixel_x8,
    pre_erosion_row, ratio_of_derivatives_inv_x8, ratio_of_derivatives_x8,
};

// Re-export test constants for any tests that use them
pub use crate::quant::aq::simd::TEST_INPUTS_RATIO;
