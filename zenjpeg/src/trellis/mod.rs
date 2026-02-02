//! Trellis quantization for optimal rate-distortion.
//!
//! This is the core innovation of mozjpeg over standard libjpeg.
//! Trellis quantization uses dynamic programming to find the optimal
//! quantization decisions that minimize:
//!
//! ```text
//! Cost = Rate + Lambda * Distortion
//! ```
//!
//! where Rate is the Huffman encoding cost and Distortion is the
//! squared error from the original coefficients.
//!
//! Ported from mozjpeg's jcdctmgr.c quantize_trellis().

pub mod ac;
#[allow(dead_code)] // DC trellis API - not yet integrated into main encoder
pub mod dc;
#[allow(dead_code)] // EOB optimization API - not yet integrated into main encoder
pub mod eob;
pub mod rate;

// Re-export main types
pub use ac::trellis_quantize_block;
pub use rate::RateTable;

// Re-export items not yet used from main encoder but part of the trellis API
#[allow(unused_imports)]
pub use ac::trellis_quantize_block_with_eob_info;
#[allow(unused_imports)]
pub use dc::{dc_trellis_optimize, dc_trellis_optimize_indexed, simple_quantize_block};
#[allow(unused_imports)]
pub use eob::{estimate_block_eob_info, optimize_eob_runs, BlockEobInfo};
