//! Lossless re-pack via `crate::lossless::restructure`.
//!
//! No coefficient changes; only scan structure and Huffman tables are
//! rebuilt. This is the no-regression fallback path.

use crate::lossless::{OutputMode, RestartInterval, RestructureConfig, restructure};
use enough::Unstoppable;

use crate::recompress::error::Error;
use crate::recompress::source::SourceAnalysis;

/// Returns the lossless-restructured JPEG payload. Always progressive +
/// optimized Huffman (zenjpeg picks Huffman optimization by default in
/// progressive mode).
pub fn run_lossless(jpeg_bytes: &[u8], _analysis: &SourceAnalysis) -> Result<Vec<u8>, Error> {
    let cfg = RestructureConfig {
        output_mode: OutputMode::Progressive,
        restart_interval: RestartInterval::None,
        transform: None,
    };
    let out = restructure(jpeg_bytes, &cfg, Unstoppable)?;
    Ok(out)
}
