//! Diffmap-guided per-block refinement for the Preserve strategy
//! (`recompress-iqa` only).
//!
//! The heuristic AQ mask ([`crate::recompress::aq`]) predicts, from
//! coefficient energy alone, which blocks can shed high-frequency AC
//! coefficients. This pass replaces that prediction with measurement.
//! When the closed loop's winning candidate is Preserve, was measured,
//! and overshoots the target with iteration budget left, we score it
//! against the source with a per-pixel zensim diffmap, pool the map to
//! per-8×8-block means ([`crate::recompress::measure::BlockErrorMap`]),
//! and evolve a per-block zero-bias depth:
//!
//! - **Deepen** the zeroed tail one ladder step (`64 → 48 → 32 → 16`,
//!   zigzag indices) in blocks whose measured error sits in the low
//!   tail (≤ p40 of measured blocks) — measured slack becomes bytes.
//! - **Protect** (depth restored to 64, mask cleared) blocks whose
//!   measured error sits in the high tail (≥ p95 AND > 2× median) —
//!   the heuristic zeroed something the metric can see, so undo it.
//!
//! Each refined candidate re-runs only the coefficient-domain emit (no
//! IDCT/FDCT) plus one measurement, and is **kept only when it is both
//! smaller than the incumbent and still clears the target** under the
//! same calibration arithmetic the closed loop uses
//! (`a_hat = anchor + GEXP_SLOPE · (g − gexp)`). Depths only ever move
//! toward the measured evidence, and the first rejected candidate stops
//! the loop, so refinement can never ship anything the closed loop's
//! own acceptance rule would refuse.
//!
//! Blocks outside the measured grid — MCU padding and partial edge
//! blocks (the diffmap grid is full-8×8-blocks-only) — are never
//! touched: edge blocks are exactly where partial-MCU artifacts live.
//!
//! Requires a calibrated generation-loss expectation (`gexp`) for the
//! source's encoder class: without it, `a_hat` cannot see the measured
//! score and the acceptance rule would be blind. The caller skips
//! refinement for uncalibrated encoder classes.

use crate::recompress::budget::BudgetState;
use crate::recompress::calibration::per_encoder::GEXP_SLOPE;
use crate::recompress::error::Error;
use crate::recompress::measure::{BlockErrorMap, MeasureCtx};
use crate::recompress::router::StrategyParams;
use crate::recompress::source::SourceAnalysis;
use crate::recompress::strategies::preserve::prepare_preserve;
use crate::recompress::strategies::preserve_emit::{AqMask, EmitConfig, emit_preserved};

/// Minimum measured overshoot (`a_hat − target`, zensim-A) before a
/// refinement pass is attempted. Mirrors the Preserve strategy's
/// `AQ_HEADROOM_MARGIN`: below this, deepening is likely to dip under
/// target and waste the pass.
pub(crate) const REFINE_MIN_OVERSHOOT: f32 = 2.0;

/// Deepest allowed zero-bias depth (first zeroed zigzag index). The low
/// band (AC 1..16) is never zeroed — matching the most aggressive
/// heuristic tier historically measured as recoverable.
const MIN_DEPTH: usize = 16;

/// Deepen fraction: blocks at or below this percentile of measured
/// block error step one ladder rung deeper.
const DEEPEN_PCTL_NUM: usize = 2;
const DEEPEN_PCTL_DEN: usize = 5; // p40

/// Protect fraction: blocks at or above this percentile AND above
/// 2× the median get their mask cleared entirely.
const PROTECT_PCTL_NUM: usize = 19;
const PROTECT_PCTL_DEN: usize = 20; // p95

/// Inputs describing the incumbent (the closed loop's winning Preserve
/// candidate) and the budget the refinement may spend.
pub(crate) struct RefineInputs<'a> {
    /// Calibration anchor at the incumbent's dial (expected achieved
    /// zensim-A at that dial).
    pub(crate) anchor: f32,
    /// Expected generation loss for the calibration cell (`gexp_lookup`).
    pub(crate) gexp: f32,
    /// Effective target the closed loop aims at.
    pub(crate) target: f32,
    /// Incumbent's encoded size — every kept refinement must beat it.
    pub(crate) incumbent_len: usize,
    /// Incumbent's predicted achieved quality.
    pub(crate) incumbent_a_hat: f32,
    /// Measured per-block error map of the incumbent.
    pub(crate) block_map: &'a BlockErrorMap,
    /// Remaining candidate-encode budget (closed-loop hard cap minus
    /// passes already used).
    pub(crate) max_passes: u32,
}

/// A kept refinement: strictly smaller than the incumbent and still
/// clearing the target.
pub(crate) struct RefineOutcome {
    pub(crate) bytes: Vec<u8>,
    pub(crate) g: f32,
    pub(crate) a_hat: f32,
}

/// Run the refinement loop. Returns `Ok(None)` when no refined
/// candidate was kept (insufficient overshoot, no eligible blocks, or
/// the first refined candidate was rejected). Errors from emit or
/// measurement of *refined* candidates never propagate — they stop the
/// loop with whatever was already kept. Only `prepare_preserve` errors
/// (which mean the incumbent itself could not be reconstructed)
/// propagate, and the caller treats those as "skip refinement".
pub(crate) fn refine_preserve(
    jpeg_bytes: &[u8],
    analysis: &SourceAnalysis,
    params: &StrategyParams,
    ctx: &MeasureCtx,
    inputs: RefineInputs<'_>,
    budget: &mut BudgetState,
) -> Result<Option<RefineOutcome>, Error> {
    if inputs.max_passes == 0
        || inputs.incumbent_a_hat - inputs.target < REFINE_MIN_OVERSHOOT
        || !budget.may_measure()
    {
        return Ok(None);
    }

    // Reconstruct the exact emit state the incumbent was produced from
    // (same params → same quant strategy, same heuristic mask).
    let prepared = prepare_preserve(jpeg_bytes, analysis, params)?;
    let luma = prepared
        .coeffs
        .components
        .first()
        .ok_or(Error::Internal("refine: no luma component"))?;
    let blocks_wide = luma.blocks_wide;
    let n_blocks = luma.num_blocks();

    let mut depths = depths_from_mask(prepared.aq_mask.as_ref(), n_blocks);
    let mut kept: Option<RefineOutcome> = None;
    // The map refinement reads from: the incumbent's map first, then
    // each kept candidate's own map.
    let mut own_map: Option<BlockErrorMap> = None;
    let mut incumbent_len = inputs.incumbent_len;

    for _ in 0..inputs.max_passes {
        if !budget.may_measure() {
            break;
        }
        let map = own_map.as_ref().unwrap_or(inputs.block_map);
        if evolve_depths(&mut depths, map, blocks_wide) == 0 {
            break; // nothing left to deepen or protect
        }
        let cfg = EmitConfig {
            quant_strategy: prepared.quant_strategy,
            aq_mask: Some(mask_from_depths(&depths)),
            preserved_segments: prepared.preserved_segments.clone(),
        };
        let Ok(bytes) = emit_preserved(&prepared.coeffs, prepared.subsampling, &cfg) else {
            break;
        };
        let Ok((g, new_map)) = ctx.score_with_blocks(&bytes) else {
            break;
        };
        budget.note_iteration();

        let a_hat = inputs.anchor + GEXP_SLOPE * (g - inputs.gexp);
        let clears = a_hat >= inputs.target - crate::recompress::api::CLOSED_LOOP_TOL;
        let smaller = bytes.len() < incumbent_len;
        if clears && smaller {
            incumbent_len = bytes.len();
            own_map = Some(new_map);
            kept = Some(RefineOutcome { bytes, g, a_hat });
        } else {
            // Depths are monotone toward the evidence; a rejection means
            // this direction's budget is spent. Keep the incumbent-so-far.
            break;
        }
    }

    Ok(kept)
}

/// Next rung of the zero-bias depth ladder.
fn deeper(depth: usize) -> usize {
    match depth {
        64 => 48,
        48 => 32,
        _ => MIN_DEPTH,
    }
}

/// Per-block zero-bias depths from a (possibly absent) heuristic mask.
/// Depth = first zeroed zigzag index; 64 = untouched block. Assumes
/// tier-shaped masks (a contiguous zeroed tail), which is what
/// `build_aq_mask` produces — for a general mask this reads the lowest
/// set bit.
fn depths_from_mask(mask: Option<&AqMask>, n_blocks: usize) -> Vec<usize> {
    match mask {
        None => vec![64; n_blocks],
        Some(m) => {
            let mut depths = Vec::with_capacity(n_blocks);
            for b in 0..n_blocks {
                let bits = m.get(b).copied().unwrap_or(0);
                depths.push(if bits == 0 {
                    64
                } else {
                    bits.trailing_zeros() as usize
                });
            }
            depths
        }
    }
}

/// Per-block mask from depths: bits `depth..64` set, empty for 64.
fn mask_from_depths(depths: &[usize]) -> AqMask {
    depths
        .iter()
        .map(|&d| {
            if d >= 64 {
                0u64
            } else {
                // Set bits d..64.
                (!0u64) << d
            }
        })
        .collect()
}

struct Thresholds {
    deepen_at: f32,
    protect_at: f32,
    median: f32,
}

/// Percentile thresholds over the measured block errors. `None` when
/// there are no measured blocks.
fn compute_thresholds(errors: &[f32]) -> Option<Thresholds> {
    if errors.is_empty() {
        return None;
    }
    let mut sorted: Vec<f32> = errors.to_vec();
    sorted.sort_by(f32::total_cmp);
    let n = sorted.len();
    // Deepen ("at or below p40") floors its index; protect ("at or
    // above p95") rounds UP so small n doesn't drag the threshold down
    // onto merely-elevated blocks.
    let idx_down = |num: usize, den: usize| ((n - 1) * num) / den;
    let idx_up = |num: usize, den: usize| ((n - 1) * num).div_ceil(den);
    Some(Thresholds {
        deepen_at: sorted[idx_down(DEEPEN_PCTL_NUM, DEEPEN_PCTL_DEN)],
        protect_at: sorted[idx_up(PROTECT_PCTL_NUM, PROTECT_PCTL_DEN)],
        median: sorted[n / 2],
    })
}

/// One evolution step over `depths` given the measured map. Returns the
/// number of blocks changed (0 = converged, caller stops). Blocks
/// outside the measured grid are skipped entirely.
fn evolve_depths(depths: &mut [usize], map: &BlockErrorMap, blocks_wide: usize) -> usize {
    let Some(th) = compute_thresholds(&map.errors) else {
        return 0;
    };
    let mut changed = 0usize;
    for (idx, depth) in depths.iter_mut().enumerate() {
        let bx = idx % blocks_wide;
        let by = idx / blocks_wide;
        let Some(err) = map.get(bx, by) else {
            continue; // MCU padding / edge sliver: never touch
        };
        if err >= th.protect_at && err > 2.0 * th.median {
            if *depth < 64 {
                *depth = 64;
                changed += 1;
            }
        } else if err <= th.deepen_at && *depth > MIN_DEPTH {
            *depth = deeper(*depth);
            changed += 1;
        }
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(errors: Vec<f32>, blocks_w: usize, blocks_h: usize) -> BlockErrorMap {
        assert_eq!(errors.len(), blocks_w * blocks_h);
        BlockErrorMap {
            errors,
            blocks_w,
            blocks_h,
        }
    }

    #[test]
    fn depth_ladder_descends_and_saturates() {
        assert_eq!(deeper(64), 48);
        assert_eq!(deeper(48), 32);
        assert_eq!(deeper(32), 16);
        assert_eq!(deeper(16), 16);
    }

    #[test]
    fn depths_roundtrip_through_masks() {
        let depths = vec![64, 48, 32, 16];
        let mask = mask_from_depths(&depths);
        assert_eq!(mask[0], 0);
        assert_eq!(mask[1], (!0u64) << 48);
        assert_eq!(mask[2], (!0u64) << 32);
        assert_eq!(mask[3], (!0u64) << 16);
        let back = depths_from_mask(Some(&mask), 4);
        assert_eq!(back, depths);
    }

    #[test]
    fn depths_from_absent_mask_are_untouched() {
        assert_eq!(depths_from_mask(None, 3), vec![64, 64, 64]);
    }

    #[test]
    fn evolve_deepens_low_tail_and_protects_high_tail() {
        // 10 blocks in a row: 8 low-error, 1 mid, 1 huge.
        let errors = vec![0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 10.0];
        let m = map(errors, 10, 1);
        let mut depths = vec![48usize; 10];
        let changed = evolve_depths(&mut depths, &m, 10);
        assert!(changed > 0);
        // Low-tail blocks deepened one rung.
        assert_eq!(depths[0], 32);
        // The huge-error block (p95 tail, > 2× median) is protected.
        assert_eq!(depths[9], 64);
        // The mid block is left alone.
        assert_eq!(depths[8], 48);
    }

    #[test]
    fn evolve_skips_unmeasured_padding_blocks() {
        // Coefficient grid 4 wide, measured grid only 2 wide: the right
        // half is MCU padding and must never change.
        let m = map(vec![0.1, 0.1], 2, 1);
        let mut depths = vec![48usize; 4];
        evolve_depths(&mut depths, &m, 4);
        assert_eq!(depths[0], 32, "measured block deepens");
        assert_eq!(depths[1], 32, "measured block deepens");
        assert_eq!(depths[2], 48, "padding block untouched");
        assert_eq!(depths[3], 48, "padding block untouched");
    }

    #[test]
    fn evolve_converges_to_zero_changes() {
        // All-equal low errors: every block deepens until the ladder
        // floor, then no further changes.
        let m = map(vec![0.1; 4], 4, 1);
        let mut depths = vec![64usize; 4];
        let mut rounds = 0;
        while evolve_depths(&mut depths, &m, 4) > 0 {
            rounds += 1;
            assert!(rounds < 10, "must converge");
        }
        assert_eq!(depths, vec![16; 4], "uniform slack deepens to the floor");
    }

    #[test]
    fn uniform_nonzero_errors_never_protect() {
        // Degenerate distribution: protect requires err > 2× median,
        // which a uniform field can't satisfy — everything deepens.
        let m = map(vec![1.0; 4], 4, 1);
        let mut depths = vec![48usize; 4];
        evolve_depths(&mut depths, &m, 4);
        assert_eq!(depths, vec![32; 4]);
    }

    /// End-to-end refinement against real zensim measurement: encode a
    /// textured source, run Preserve, then verify the refinement loop
    /// finds a strictly smaller candidate that still decodes and did
    /// not move closer to the source (zeroing removes information).
    #[test]
    fn refine_shrinks_preserve_output_on_real_measurement() {
        use crate::encode::{ChromaSubsampling, EncoderConfig, PixelLayout};
        use crate::recompress::api::{Budget, RecompressOptions};
        use crate::recompress::calibration::CellCi;
        use crate::recompress::source::analyze_source;
        use crate::recompress::strategies::preserve::run_preserve;
        use enough::Unstoppable;

        // Structured noise across the whole frame (deterministic PRNG):
        // every block carries AC energy through the spectrum, so
        // deepening the zero-bias tail provably removes tokens. Chroma
        // varies smoothly per the DC-only-coverage rule (flat within
        // chroma blocks, changing across them).
        let (w, h) = (128u32, 128u32);
        let mut px = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                let s = (x ^ y.wrapping_shl(3)).wrapping_mul(2654435761u32);
                let luma = 90 + ((s >> 8) & 0x7F) as u8;
                let cb = (16 * (x / 32) + 40) as u8;
                px.push(luma.saturating_add(cb / 4));
                px.push(luma);
                px.push(luma.saturating_add((s & 0x3F) as u8));
            }
        }
        let cfg = EncoderConfig::ycbcr(90, ChromaSubsampling::Quarter);
        let mut enc = cfg.encode_from_bytes(w, h, PixelLayout::Rgb8Srgb).unwrap();
        enc.push_packed(&px, Unstoppable).unwrap();
        let source = enc.finish().unwrap();

        let analysis = analyze_source(&source).expect("source analysis");
        let params = StrategyParams {
            target_ijg_q: 70,
            target_ba_distance: 2.0,
            ci: CellCi::Moderate,
            target_zensim_a: 55.0,
            projected_zensim_a: 80.0,
            dial_zensim_a: 70.0,
        };
        let opts = RecompressOptions::new(55.0);
        let incumbent = run_preserve(&source, &analysis, &params, &opts).expect("preserve runs");

        let ctx = MeasureCtx::new(&source).expect("measure ctx");
        let (g0, map0) = ctx
            .score_with_blocks(&incumbent.bytes)
            .expect("incumbent measures");

        // Anchor far above target: acceptance is effectively "any
        // strictly smaller candidate", which isolates the mask-evolution
        // + emit machinery under test from calibration specifics.
        let anchor = 80.0;
        let mut budget = BudgetState::new(Budget::MaxIterations(8));
        let out = refine_preserve(
            &source,
            &analysis,
            &params,
            &ctx,
            RefineInputs {
                anchor,
                gexp: g0,
                target: 55.0,
                incumbent_len: incumbent.bytes.len(),
                incumbent_a_hat: anchor,
                block_map: &map0,
                max_passes: 4,
            },
            &mut budget,
        )
        .expect("refinement must not error");

        let r = out.expect("textured q90 source at large overshoot must yield a kept refinement");
        assert!(
            r.bytes.len() < incumbent.bytes.len(),
            "kept refinement must be strictly smaller ({} vs {})",
            r.bytes.len(),
            incumbent.bytes.len()
        );
        assert!(
            crate::decode::DecodeConfig::new()
                .decode(&r.bytes, Unstoppable)
                .is_ok(),
            "refined bytes must decode"
        );
        assert!(
            r.g <= g0 + 0.25,
            "zeroing coefficients cannot move the candidate closer to the source \
             (refined g {} vs incumbent g {})",
            r.g,
            g0
        );
        assert!(
            budget.iterations_used >= 1,
            "refinement must account its measurement passes"
        );
    }
}
