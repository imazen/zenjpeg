//! Unstable, opt-in per-block observations for encoder tooling.
//!
//! Capture keeps all blocks in memory. Coefficients are the forward-DCT
//! input to quantization in natural row-major order; levels are in JPEG
//! zigzag order. AQ strength is the input to quantization before any optional boundary
//! refinement, not a normalized multiplier. Entropy attribution is not
//! captured. Capture storage is additional to the encoder's memory estimates
//! and allocation statistics. This interface is for small interactive experiments.

use alloc::sync::Arc;
use alloc::vec::Vec;
use spin::Mutex;

// Custom tables used by the diagnostics viewer.
pub use super::tuning::{EncodingTables, ScalingParams};

/// The image and component grids observed during an encode.
#[derive(Debug)]
pub struct EncodeDiagnostics {
    pub width: u32,
    pub height: u32,
    pub color_path: &'static str,
    pub sampling_factors: Vec<(u8, u8)>,
    pub components: Vec<ComponentDiagnostics>,
}

/// A component's actual base quantization and zero-bias parameters.
#[derive(Debug)]
pub struct ComponentDiagnostics {
    pub component_id: u8,
    pub block_grid: (u32, u32),
    pub quant_table_base: [u16; 64],
    pub zero_bias: [f32; 64],
    pub zero_bias_mul: [f32; 64],
    /// Blocks in raster order, excluding padding outside the image.
    pub blocks: Vec<BlockDiagnostics>,
}

/// One block at quantization time.
#[derive(Debug, Clone)]
pub struct BlockDiagnostics {
    pub coef_pre_quant: [f32; 64],
    pub coef_levels: [i16; 64],
    pub aq_strength: f32,
}

impl Default for BlockDiagnostics {
    fn default() -> Self {
        Self {
            coef_pre_quant: [0.0; 64],
            coef_levels: [0; 64],
            aq_strength: 0.0,
        }
    }
}

/// Owned handle to a capture. After the encoder is finished or dropped,
/// [`Self::try_take`] returns the data without copying the block arrays.
#[derive(Debug)]
pub struct DiagnosticsHandle(pub(crate) Arc<Mutex<EncodeDiagnostics>>);

impl DiagnosticsHandle {
    /// Returns the capture once the encoder releases its reference. If the
    /// encoder is still alive, returns the handle so it can be tried again.
    /// Dropping an unfinished encoder leaves a partial capture.
    pub fn try_take(self) -> Result<EncodeDiagnostics, Self> {
        Arc::try_unwrap(self.0).map(Mutex::into_inner).map_err(Self)
    }
}
