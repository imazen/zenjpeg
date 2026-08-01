//! Preserve strategy — coefficient-domain edit.
//!
//! Decodes the source JPEG's DCT coefficients (via zenjpeg's public
//! `DecodeConfig::decode_coefficients`), scales the quantization
//! tables to a target quality, re-quantizes the coefficients, and
//! emits a new JPEG via [`crate::recompress::strategies::preserve_emit`]. No
//! IDCT/FDCT round trip — generation loss is bounded by the rounding
//! in the requantize step.
//!
//! For sources that are already well-quantized at the target, this
//! beats [`crate::recompress::strategies::tuned::run_tuned`] on both generation
//! loss (zero pixel-domain rounding) and size (fewer Huffman tokens
//! after AQ zero-bias). For sources where the target requires more
//! aggressive quant scaling than the source's tables can express,
//! the router falls back to Tuned.

use crate::decode::{DecodeConfig, SegmentType};
use crate::detect::EncoderFamily;
use crate::types::Subsampling;
use enough::Unstoppable;

use crate::recompress::api::RecompressOptions;
use crate::recompress::aq::build_aq_mask;
use crate::recompress::error::Error;
use crate::recompress::router::StrategyParams;
use crate::recompress::source::SourceAnalysis;

use super::StrategyOutcome;
use super::preserve_emit::{EmitConfig, QuantScale, QuantStrategy, emit_preserved};

/// Convert a target IJG-equivalent quality to a quant-table scale
/// (relative to the source's existing tables). At target_ijg_q = source
/// quality, scale ≈ 1.0 (no change). At lower target, scale > 1.0
/// (tighter quantization).
///
/// The mapping uses the standard IJG quality-to-scale curve:
/// `scale = 5000 / q` for q < 50, `scale = 200 - 2*q` for q >= 50.
/// Source quality is approximated from the source BA distance via the
/// `crate::recompress::target` anchor (which is corpus-fitted).
fn quant_scale_for_target(source: &SourceAnalysis, target_ijg_q: u8) -> QuantScale {
    let target_factor = ijg_quality_to_scale_factor(target_ijg_q);
    // Approximate source IJG-Q from estimated zensim-A. (Inverse of
    // zenjpeg's `quality_to_scale_factor`.)
    let source_zensim = source.estimated_zensim_a_vs_reference();
    // Heuristic: source-quality-from-zensim-A.
    let source_q = zensim_a_to_ijg_q_estimate(source_zensim);
    let source_factor = ijg_quality_to_scale_factor(source_q);
    // Per-component scale = how much *more* quantization the new tables apply.
    let scale = (target_factor / source_factor).max(1.0); // never relax
    QuantScale {
        luma: scale,
        chroma: scale,
    }
}

fn ijg_quality_to_scale_factor(q: u8) -> f32 {
    let q = q.clamp(1, 100) as f32;
    if q < 50.0 {
        5000.0 / q
    } else {
        200.0 - 2.0 * q
    }
}

fn zensim_a_to_ijg_q_estimate(z: f32) -> u8 {
    // Very rough inverse of zenjpeg's source quality estimate.
    let q = match z {
        x if x >= 98.0 => 95,
        x if x >= 95.0 => 90,
        x if x >= 90.0 => 85,
        x if x >= 85.0 => 80,
        x if x >= 80.0 => 75,
        x if x >= 75.0 => 70,
        x if x >= 65.0 => 60,
        x if x >= 55.0 => 50,
        x if x >= 45.0 => 40,
        _ => 30,
    };
    q as u8
}

/// Everything `run_preserve` derives from the input before emitting:
/// decoded coefficients, carried metadata segments, the quant strategy,
/// and the heuristic AQ mask. Factored out so the diffmap-guided
/// refinement pass ([`crate::recompress::refine`], `recompress-iqa`) can
/// rebuild the *identical* emit state for a given `params` and then
/// evolve only the mask.
pub(in crate::recompress) struct PreparedPreserve {
    pub(in crate::recompress) coeffs: crate::decode::DecodedCoefficients,
    pub(in crate::recompress) preserved_segments: Vec<crate::decode::PreservedSegment>,
    pub(in crate::recompress) quant_strategy: QuantStrategy,
    pub(in crate::recompress) aq_mask: Option<super::preserve_emit::AqMask>,
    pub(in crate::recompress) subsampling: Subsampling,
}

pub fn run_preserve(
    jpeg_bytes: &[u8],
    analysis: &SourceAnalysis,
    params: &StrategyParams,
    _opts: &RecompressOptions,
) -> Result<StrategyOutcome, Error> {
    let prepared = prepare_preserve(jpeg_bytes, analysis, params)?;

    // Emit.
    let cfg = EmitConfig {
        quant_strategy: prepared.quant_strategy,
        aq_mask: prepared.aq_mask,
        preserved_segments: prepared.preserved_segments,
    };
    let bytes = emit_preserved(&prepared.coeffs, prepared.subsampling, &cfg)?;

    // Validate the emitted bytes round-trip through zenjpeg's decoder.
    // If they don't, the emitter hit an edge case — return an error so
    // the API layer falls back to Lossless rather than shipping an
    // invalid file. The Lossless fallback guarantees a usable output.
    if DecodeConfig::new().decode(&bytes, Unstoppable).is_err() {
        return Err(Error::Internal(
            "preserve: emitted bytes failed roundtrip decode",
        ));
    }

    Ok(StrategyOutcome {
        bytes,
        measured_zensim_a: None,
    })
}

/// Decode + derive the Preserve emit state for `params` WITHOUT
/// emitting. Deterministic in `(jpeg_bytes, analysis, params)` — two
/// calls with the same inputs produce the same quant strategy and the
/// same heuristic mask, which is what lets the refinement pass
/// reconstruct the state a prior `run_preserve` emitted from.
pub(in crate::recompress) fn prepare_preserve(
    jpeg_bytes: &[u8],
    analysis: &SourceAnalysis,
    params: &StrategyParams,
) -> Result<PreparedPreserve, Error> {
    // Operating window: with the zigzag DQT bug fixed (2026-05-28),
    // Preserve produces pixel-identical output for IDENTITY scale
    // across 4:4:4 / 4:2:0 / 4:2:2 / 4:4:0 with even dimensions.
    // The one remaining edge case is partial-MCU padding at certain
    // odd dimensions (e.g. width=67 with 4:2:0). The
    // roundtrip-decode guard below catches the syntactic failures;
    // the byte gate here only rejects targets that would zero every
    // AC coefficient (encoder structural minimum).
    if params.target_ijg_q < 8 {
        return Err(Error::Internal(
            "preserve: target IJG-Q too low (would zero all AC coefficients)",
        ));
    }

    // 1. Decode source coefficients AND preserved metadata in one pass.
    let (coeffs, extras) = DecodeConfig::new()
        .decode_coefficients_with_extras(jpeg_bytes, Unstoppable)
        .map_err(|e| Error::Zenjpeg(format!("preserve decode coefficients: {e}")))?;
    // Carry every APPn/COM segment EXCEPT MPF. Coefficient-domain
    // recompression is metadata-transparent: dropping these would
    // silently change the decoded colors (ICC profile) and display
    // orientation (EXIF). MPF is excluded because its directory holds
    // byte offsets into embedded images that recompression invalidates.
    let preserved_segments: Vec<_> = extras
        .map(|ex| {
            ex.segments()
                .iter()
                .filter(|s| s.segment_type != SegmentType::Mpf)
                .cloned()
                .collect()
        })
        .unwrap_or_default();

    // 2. Quant-strategy dispatch based on source quality. The
    // head-to-head sweep (benchmarks/preserve_vs_tuned_*.tsv) shows:
    //   - High source-q (est zensim ≥ 75): UniformScale wins on
    //     hit-target accuracy (preserves the source's already-fine
    //     per-frequency weighting). At q=90 → target=70 it's the
    //     ONLY strategy that hits the target precisely; Tuned and
    //     target-quality both undershoot by 1-4 zensim-A.
    //   - Lower source-q (est zensim < 75): TargetQuality wins
    //     because uniform scaling over-preserves coarse source
    //     tables, causing the output to inflate above the source.
    // Quant-strategy dispatch based on source quality. (Tested
    // 2026-05-28: forcing TargetQuality for IJG-family sources craters
    // WORSE than UniformScale on high-q turbo — the "never-finer-than-
    // source" clamp + fine→coarse requantization ratio collapse the
    // coefficients. See docs/MULTI_ENCODER_VALIDATION.md "doing better
    // on turbo". So the source-quality dispatch stands; turbo/mozjpeg
    // route to Tuned via the per-encoder calibration regardless.)
    // Lever 2 (2026-05-29): mozjpeg / ImageMagick sources are encoded
    // with Robidoux-shaped quant tables. Cross-family IJG-std
    // retargeting reshapes the spectrum and craters quality; uniform
    // scaling goes through a lossy source-q estimate. Same-family
    // Robidoux retargeting at the inverse-calibrated dial keeps the
    // spectral shape and hits the target precisely. Route them to it
    // regardless of source quality; keep the IJG dispatch for
    // turbo/Pillow/IJG and everything else.
    let source_zensim = analysis.estimated_zensim_a_vs_reference();
    let is_robidoux_family = matches!(
        analysis.encoder,
        EncoderFamily::Mozjpeg | EncoderFamily::ImageMagick
    );
    let quant_strategy = if is_robidoux_family {
        QuantStrategy::RobidouxTargetQuality {
            target_quality: params.target_ijg_q,
        }
    } else if source_zensim >= 75.0 {
        QuantStrategy::UniformScale(quant_scale_for_target(analysis, params.target_ijg_q))
    } else {
        QuantStrategy::TargetQuality {
            target_ijg_q: params.target_ijg_q,
        }
    };

    // 3. Compute AQ zero-bias mask — but ONLY when we have quality
    // headroom to spend. The AQ ablation
    // (benchmarks/aq_ablation_8refs_2026-05-28.tsv) measured AQ as a
    // consistent trade: ~4% smaller output for ~1 zensim-A lower
    // quality. That's a *good* trade only when calibration projects
    // we'll overshoot the target by more than AQ's expected cost;
    // otherwise AQ pushes the output below target. So we gate AQ on
    // `projected - target >= AQ_HEADROOM_MARGIN`.
    const AQ_HEADROOM_MARGIN: f32 = 2.0;
    let headroom = params.projected_zensim_a - params.target_zensim_a;
    let aq_mask = if headroom >= AQ_HEADROOM_MARGIN {
        build_aq_mask(&coeffs)
    } else {
        None
    };

    let subsampling = match analysis.subsampling {
        Subsampling::S444 | Subsampling::S422 | Subsampling::S420 | Subsampling::S440 => {
            analysis.subsampling
        }
    };

    Ok(PreparedPreserve {
        coeffs,
        preserved_segments,
        quant_strategy,
        aq_mask,
        subsampling,
    })
}
