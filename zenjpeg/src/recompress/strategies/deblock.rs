//! Deblock strategy: content-aware deblock + re-encode.
//!
//! Identical to the [`super::tuned`] strategy except for the decode-side
//! [`DeblockMode::Auto`] setting, which lets zenjpeg's content
//! classifier choose Knusperli (DCT-domain, low source quality) or the
//! 4-tap boundary filter (mid/high source quality) on a per-image
//! basis. Same encoder config as Tuned (`auto_optimize(true)`, matched
//! chroma subsampling, progressive). One pixel-domain generation pass.
//!
//! When to pick Deblock vs Tuned (router decision): Deblock projects
//! a small zensim-A lift over Tuned at low source quality (estimated
//! source ≤ 50 zensim-A); above that it's neutral or slightly worse
//! and Tuned dominates on size.

use crate::decoder::{DeblockMode, DecodeConfig, OutputTarget, Subsampling};
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

pub fn run_deblock(
    jpeg_bytes: &[u8],
    analysis: &SourceAnalysis,
    params: &StrategyParams,
    _opts: &RecompressOptions,
) -> Result<StrategyOutcome, Error> {
    // 1. Decode with content-aware deblocking enabled.
    let decode_cfg = DecodeConfig::new()
        .output_target(OutputTarget::Srgb8)
        .deblock(DeblockMode::Auto);
    let decoded = decode_cfg
        .decode(jpeg_bytes, Unstoppable)
        .map_err(|e| Error::Zenjpeg(format!("decode: {e}")))?;

    let pixels = decoded
        .pixels_u8()
        .ok_or(Error::Internal("decoded u8 pixels missing"))?;

    let width = decoded.width;
    let height = decoded.height;

    // 2. Re-encode at calibrated quality, matched subsampling.
    let chroma = match analysis.subsampling {
        Subsampling::S444 => ChromaSubsampling::None,
        Subsampling::S422 => ChromaSubsampling::HalfHorizontal,
        Subsampling::S420 => ChromaSubsampling::Quarter,
        Subsampling::S440 => ChromaSubsampling::HalfVertical,
    };

    // Same encoder config as Tuned (HybridMaxCompression — the RD
    // winner over auto_optimize per the param ablation). The Deblock vs
    // Tuned difference lives entirely in the decode-side
    // `DeblockMode::Auto` above.
    let cfg = EncoderConfig::ycbcr(params.target_ijg_q, chroma)
        .optimization(OptimizationPreset::HybridMaxCompression)
        // Entropy-stage exactness on top of the RD-ablated param set:
        // the scan search stays (preset behaviour) and the sequential
        // (+tiny) trials run when the searched progressive output lands
        // under the byte gate — exactly the small-output regime
        // recompress serves. Identical pixels; only bytes improve.
        .progressive(crate::encoder::ProgressiveScanMode::SmallestSearch);

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
