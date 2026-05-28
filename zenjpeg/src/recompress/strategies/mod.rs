//! Strategy implementations. Each module exposes a single `run_*`
//! function returning [`StrategyOutcome`].
//!
//! Every strategy is allowed to fail and the router will then route to
//! `Lossless` per the no-size-regression contract. Strategies that
//! return outputs ≥ source size also trigger that fallback at the API
//! layer (see [`crate::recompress::api::recompress`]).

pub mod deblock;
pub mod lossless;
pub mod preserve;
pub mod preserve_emit;
pub mod tuned;

/// Outcome shared by all recompression strategies.
#[derive(Debug)]
pub struct StrategyOutcome {
    /// New JPEG bytes.
    pub bytes: Vec<u8>,
    /// Optional measured zensim-A vs source — populated only when the
    /// strategy ran an IQA pass.
    pub measured_zensim_a: Option<f32>,
}
