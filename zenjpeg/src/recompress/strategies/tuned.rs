//! Tuned strategy: full decode + zenjpeg re-encode at calibrated quality.
//!
//! One pixel-domain generation; the highest CPU strategy and the safest
//! fallback when the source has heavy block artifacts or chroma
//! subsampling we want to keep stable.

use crate::decoder::{DecodeConfig, OutputTarget, Subsampling};
use crate::encoder::{
    ChromaSubsampling, EncoderConfig, OptimizationPreset, PixelLayout,
    Unstoppable as EncUnstoppable,
};
use enough::Unstoppable;

use crate::recompress::api::RecompressOptions;
use crate::recompress::error::Error;
use crate::recompress::router::StrategyParams;
use crate::recompress::source::SourceAnalysis;

use super::StrategyOutcome;

pub fn run_tuned(
    jpeg_bytes: &[u8],
    analysis: &SourceAnalysis,
    params: &StrategyParams,
    _opts: &RecompressOptions,
) -> Result<StrategyOutcome, Error> {
    // 1. Decode to RGB8.
    let decode_cfg = DecodeConfig::new().output_target(OutputTarget::Srgb8);
    let decoded = decode_cfg
        .decode(jpeg_bytes, Unstoppable)
        .map_err(|e| Error::Zenjpeg(format!("decode: {e}")))?;

    let pixels = decoded
        .pixels_u8()
        .ok_or(Error::Internal("decoded u8 pixels missing"))?;

    let width = decoded.width;
    let height = decoded.height;

    // 2. Re-encode with matched chroma subsampling + calibrated quality.
    let chroma = source_subsampling_to_encoder(analysis.subsampling);
    // Encoder param set: HybridMaxCompression (jpegli AQ + adaptive
    // trellis + deringing + progressive scan search). The 6-ref RD
    // ablation (benchmarks/zenjpeg_param_rd_6refs_2026-05-28.tsv) showed
    // this beats `auto_optimize` at every quality level (≈4% smaller at
    // zensim 60, 1.8% at 70, 1% at 80) while staying pure YCbCr (broadly
    // decodable). `auto_optimize` (plain hybrid trellis λ=14.5) was NOT
    // the RD optimum. XYB wins more at high quality (≈6% at zensim 80)
    // but changes color handling / decoder compatibility — reserved for
    // a future modern-decoder mode.
    let cfg = EncoderConfig::ycbcr(params.target_ijg_q, chroma)
        .optimization(OptimizationPreset::HybridMaxCompression);

    let mut enc = cfg
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .map_err(|e| Error::Zenjpeg(format!("encode setup: {e}")))?;
    enc.push_packed(pixels, EncUnstoppable)
        .map_err(|e| Error::Zenjpeg(format!("encode push: {e}")))?;
    let bytes = enc
        .finish()
        .map_err(|e| Error::Zenjpeg(format!("encode finish: {e}")))?;

    Ok(StrategyOutcome {
        bytes,
        measured_zensim_a: None,
    })
}

fn source_subsampling_to_encoder(s: Subsampling) -> ChromaSubsampling {
    match s {
        Subsampling::S444 => ChromaSubsampling::None,
        Subsampling::S422 => ChromaSubsampling::HalfHorizontal,
        Subsampling::S420 => ChromaSubsampling::Quarter,
        Subsampling::S440 => ChromaSubsampling::HalfVertical,
        // Future Subsampling variants we don't know about → 4:2:0.
        _ => ChromaSubsampling::Quarter,
    }
}
