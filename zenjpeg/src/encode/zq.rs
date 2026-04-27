//! Target perceptual-quality (`zq`) types for the closed-loop encoder.
//!
//! See [`crate::encode::Quality::Zq`] / [`crate::encode::Quality::ZqExplicit`]
//! for usage. The encoder iteratively encodes the image, measures perceptual
//! error against the source (using zensim diffmap), and adjusts per-block
//! quantization until the target band is met or the iteration budget runs out.
//!
//! # Naming convention
//!
//! Throughout this module:
//! - `_overshoot` = how much the achieved value may exceed the target/ceiling
//! - `_undershoot` = how much the achieved value may fall below it
//!
//! Whether `_overshoot` or `_undershoot` is the "good" direction depends on
//! the constraint:
//!
//! - [`ZqTarget::target`] is a quality FLOOR: overshoot = more quality
//!   (good), undershoot = less quality (bad). Claw-back triggers on
//!   overshoot; failure on undershoot.
//! - [`BlockArtifactBound::ceiling`] is an error CEILING: overshoot = more
//!   artifact (bad), undershoot = less artifact (good). Failure triggers on
//!   overshoot; claw-back on undershoot.
//!
//! Defaults are tuned so the typical caller gets best-effort behavior with
//! no surprise errors. Callers that need strict guarantees opt in via
//! [`ZqTarget::max_undershoot`] (for the floor) or
//! [`BlockArtifactBound::max_overshoot`] (for the ceiling).

/// Explicit target-perceptual-quality specification.
///
/// Used via [`crate::encode::Quality::ZqExplicit`]. For the simple single-
/// knob form, use [`crate::encode::Quality::Zq`] instead — that path uses
/// `ZqTarget::default()` for everything except the target value.
///
/// Per-field naming convention is documented at the [module level][self].
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZqTarget {
    /// IDEAL perceived quality (zq units). The encoder iterates to reach
    /// or exceed this score within the [`Self::max_passes`] budget.
    pub target: f32,

    /// Distance ABOVE target the encoder will accept without further
    /// iteration. `None` = ship the first feasible result (single pass
    /// once target is met). `Some(t)` = if `achieved_score > target + t`,
    /// iterate to recover bytes.
    ///
    /// Smaller value → tighter hug → more iteration cost on average.
    /// Asymmetric by design: there is no over-target tolerance for
    /// failure, only for waste.
    ///
    /// Default: `Some(1.5)`.
    pub max_overshoot: Option<f32>,

    /// Distance BELOW target the encoder will accept as a SUCCESSFUL
    /// encode. `None` = best-effort, never error (default; encoder ships
    /// whatever it managed within `max_passes`). `Some(t)` = if final
    /// `achieved_score < target - t` after exhausting `max_passes`, the
    /// encoder returns `Err`.
    ///
    /// Set this when you NEED a strictness guarantee (archival,
    /// accessibility, SLA-bound serving). Permissive callers leave it
    /// `None` and inspect [`EncodeMetrics::achieved_score`] themselves.
    ///
    /// Default: `None`.
    pub max_undershoot: Option<f32>,

    /// Optional per-block worst-case artifact bound. See
    /// [`BlockArtifactBound`] for the inner knobs.
    pub block_artifact: Option<BlockArtifactBound>,

    /// Diffmap-driven correction-pass budget.
    /// - `0` = single-pass (controller disabled; encoder behaves like
    ///   `Quality::ApproxJpegli` at the resolved starting quality).
    /// - `2` = default; one initial encode plus one correction pass.
    /// - `4` = aggressive; up to four correction passes.
    ///
    /// Each pass costs roughly one base-encode + one zensim diffmap
    /// computation.
    pub max_passes: u8,
}

impl Default for ZqTarget {
    fn default() -> Self {
        Self {
            target: 80.0,
            max_overshoot: Some(1.5),
            max_undershoot: None,
            block_artifact: None,
            max_passes: 2,
        }
    }
}

impl ZqTarget {
    /// Construct a `ZqTarget` with the given target value and all other
    /// fields at their defaults. Equivalent to
    /// `ZqTarget { target, ..Default::default() }`.
    #[must_use]
    pub fn new(target: f32) -> Self {
        Self {
            target,
            ..Default::default()
        }
    }

    /// Builder-style override of [`Self::max_overshoot`].
    #[must_use]
    pub fn with_max_overshoot(mut self, v: Option<f32>) -> Self {
        self.max_overshoot = v;
        self
    }

    /// Builder-style override of [`Self::max_undershoot`].
    #[must_use]
    pub fn with_max_undershoot(mut self, v: Option<f32>) -> Self {
        self.max_undershoot = v;
        self
    }

    /// Builder-style override of [`Self::block_artifact`].
    #[must_use]
    pub fn with_block_artifact(mut self, v: Option<BlockArtifactBound>) -> Self {
        self.block_artifact = v;
        self
    }

    /// Builder-style override of [`Self::max_passes`].
    #[must_use]
    pub fn with_max_passes(mut self, n: u8) -> Self {
        self.max_passes = n;
        self
    }
}

/// Per-block worst-case artifact bound. Companion to [`ZqTarget`].
///
/// `ceiling` is in zensim diffmap units (per-pixel perceptual error,
/// averaged within an 8×8 block). The encoder tries to ensure no block's
/// averaged diffmap exceeds `ceiling`. Like [`ZqTarget::target`], this is
/// best-effort by default — if the budget runs out, the encoder ships
/// what it has. Set [`Self::max_overshoot`] to make the bound strict
/// (encoder errors if the final worst-block exceeds the threshold).
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlockArtifactBound {
    /// Worst-case per-block perceptual error, in zensim diffmap units.
    /// The encoder iterates toward "no block exceeds this".
    pub ceiling: f32,

    /// Distance ABOVE the ceiling the encoder will accept as a
    /// SUCCESSFUL encode. `None` = best-effort, never error.
    /// `Some(t)` = if final max-block-artifact > `ceiling + t` after
    /// `max_passes`, the encoder returns `Err`.
    ///
    /// Set this when you NEED a per-block guarantee (subjective testing,
    /// archival, accessibility).
    ///
    /// Default: `None`.
    pub max_overshoot: Option<f32>,

    /// Distance BELOW the ceiling that triggers byte-recovery iteration.
    /// `None` = no per-block clawback (default; the score-side clawback
    /// via [`ZqTarget::max_overshoot`] usually handles this incidentally).
    /// `Some(t)` = if max-block-artifact < `ceiling - t`, iterate to
    /// recover bytes.
    ///
    /// Surface for completeness + callers who want explicit per-block
    /// clawback semantics.
    ///
    /// Default: `None`.
    pub max_undershoot: Option<f32>,
}

impl BlockArtifactBound {
    /// Construct a `BlockArtifactBound` with the given ceiling and all
    /// other fields at their defaults (best-effort, no errors, no
    /// per-block clawback).
    #[must_use]
    pub fn new(ceiling: f32) -> Self {
        Self {
            ceiling,
            max_overshoot: None,
            max_undershoot: None,
        }
    }

    /// Builder-style override of [`Self::max_overshoot`].
    #[must_use]
    pub fn with_max_overshoot(mut self, v: Option<f32>) -> Self {
        self.max_overshoot = v;
        self
    }

    /// Builder-style override of [`Self::max_undershoot`].
    #[must_use]
    pub fn with_max_undershoot(mut self, v: Option<f32>) -> Self {
        self.max_undershoot = v;
        self
    }
}

/// Outcome of a target-perceptual-quality encode.
///
/// Returned alongside the JPEG bytes from
/// [`crate::encode::BytesEncoder::finish_with_metrics`] (and similar
/// methods on other encoder facades). Callers inspect this when they want
/// to know whether the encoder hit the target, how many passes it took,
/// or what byte budget was actually spent.
#[non_exhaustive]
#[derive(Debug, Clone, Copy)]
pub struct EncodeMetrics {
    /// Final achieved zensim score of the shipped output, computed
    /// against the source via zensim's `Trained` diffmap weighting.
    /// `f32::NAN` if no measurement was performed (e.g. `max_passes=0`
    /// or the decoder feature is unavailable).
    pub achieved_score: f32,

    /// Largest per-block diffmap value (averaged over the 8×8 block) in
    /// the shipped output. `f32::NAN` if not measured.
    pub achieved_max_block_artifact: f32,

    /// Number of encode passes performed, including the initial pass.
    /// `1` means single-pass (no correction).
    pub passes_used: u8,

    /// Encoded JPEG byte count.
    pub bytes: usize,

    /// Whether every active goal was met. `true` iff:
    /// - achieved_score ≥ target, AND
    /// - if `BlockArtifactBound` was set, achieved_max_block_artifact ≤ ceiling
    ///
    /// For non-`Zq` quality variants this is always `true` (no goal).
    pub targets_met: bool,
}

impl EncodeMetrics {
    /// Convenience: the metrics returned for a non-`Zq` quality variant
    /// (no goal, no measurement). Bytes are still recorded.
    pub(crate) fn no_target(bytes: usize) -> Self {
        Self {
            achieved_score: f32::NAN,
            achieved_max_block_artifact: f32::NAN,
            passes_used: 1,
            bytes,
            targets_met: true,
        }
    }
}

/// Maps a user-facing zq value to a starting jpegli-quality estimate for
/// the iteration loop's first pass.
///
/// This is a small lookup table calibrated from the
/// `examples/method_b_real.rs` corpus run (CID22 + screen content). The
/// goal is "first pass lands close enough to target that 1–2 correction
/// passes can refine it"; precision isn't required.
///
/// Identity-ish in the typical user range (zq 70–95 maps to jpegli q 70–95).
/// The iteration loop in `BytesEncoder` corrects from here.
#[must_use]
pub(crate) fn zq_to_starting_jpegli_q(zq: f32) -> f32 {
    // Empirical: streaming-AQ zensim score at jpegli q lands roughly
    // 1–3 zq points above the q value on photos, 1–2 below on dense
    // screen content. Picking q ≈ zq + 1 as the starting point makes
    // pass 1 land at-or-above target on most images, leaving headroom
    // for the controller to claw bytes.
    const ANCHORS: &[(f32, f32)] = &[
        (40.0, 38.0),
        (60.0, 58.0),
        (75.0, 75.0),
        (80.0, 81.0),
        (85.0, 87.0),
        (90.0, 93.0),
        (95.0, 97.0),
    ];

    if zq <= ANCHORS[0].0 {
        return ANCHORS[0].1;
    }
    if zq >= ANCHORS[ANCHORS.len() - 1].0 {
        return ANCHORS[ANCHORS.len() - 1].1.min(100.0);
    }
    for w in ANCHORS.windows(2) {
        let (lo, hi) = (w[0], w[1]);
        if zq >= lo.0 && zq <= hi.0 {
            let t = (zq - lo.0) / (hi.0 - lo.0);
            return lo.1 + t * (hi.1 - lo.1);
        }
    }
    zq // unreachable
}

// ============================================================================
// Internal: iteration loop wiring
// ============================================================================

use super::aq_controller::AqController;

/// Per-iMCU multiplicative-scale controller. Each iMCU row carries a
/// row of per-block scale factors that multiply the streaming-AQ output.
///
/// scale = 1.0 → no change. scale < 1.0 → tighter (more bits).
/// scale > 1.0 → looser (fewer bits). Final AQ is clamped to `[0.0, 0.20]`
/// by the strip processor.
#[derive(Debug)]
struct ScalingController {
    /// `scales[imcu_idx]` = per-block scale factors for that iMCU row.
    scales: alloc::vec::Vec<alloc::vec::Vec<f32>>,
}

impl AqController for ScalingController {
    fn adjust(&mut self, strengths: &mut [f32], imcu_idx: usize) {
        let row = match self.scales.get(imcu_idx) {
            Some(r) => r,
            None => return,
        };
        for (i, s) in strengths.iter_mut().enumerate() {
            let scale = row.get(i).copied().unwrap_or(1.0);
            *s = (*s * scale).clamp(0.0, 0.20);
        }
    }
}

/// Iteration-loop controller hyperparameters. Picked from the
/// `examples/method_b_real.rs` corpus run; small surface so the public
/// API doesn't expose them in v1.
mod hp {
    /// Per-pass cap on absolute scale change per block.
    pub(super) const MAX_SCALE_DELTA: f32 = 0.20;
    /// Lower clamp on cumulative scale.
    pub(super) const MIN_SCALE: f32 = 0.40;
    /// Upper clamp on cumulative scale.
    pub(super) const MAX_SCALE: f32 = 1.80;
}

/// Compute the per-block multiplicative scale schedule for the next pass.
///
/// `prev_scales` and `block_dm` are flat per-block vectors in row-major
/// order, length = blocks_w * blocks_h. The returned scales replace
/// `prev_scales` for the next pass — they are CUMULATIVE, not deltas.
///
/// Policy:
/// - Score below target → tighten the highest-error blocks (scale↓).
/// - Block diffmap above peak ceiling → tighten THOSE blocks specifically.
/// - Score in band, no ceiling violation → loosen the lowest-error
///   blocks (scale↑) to recover bytes.
fn next_scales(
    prev_scales: &[f32],
    block_dm: &[f32],
    current_score: f32,
    current_max_block: f32,
    target: &ZqTarget,
) -> alloc::vec::Vec<f32> {
    let mut sorted = block_dm.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p25 = sorted[sorted.len() / 4];
    let p75 = sorted[(3 * sorted.len()) / 4];
    let max = *sorted.last().unwrap_or(&0.0);

    let score_violation = current_score < target.target;
    let peak_violation = match target.block_artifact {
        Some(b) => current_max_block > b.ceiling,
        None => false,
    };

    // Where to tighten:
    // - peak violation → tighten any block above the ceiling specifically
    // - score violation → tighten the top-25% diffmap tail
    // - both → tighten the union (handled via min)
    let tighten_above = match (peak_violation, score_violation) {
        (true, _) => target.block_artifact.unwrap().ceiling,
        (false, true) => p75,
        _ => f32::INFINITY,
    };

    // Where to loosen (claw back bytes): only if both score and peak are
    // satisfied AND the current score is well into overshoot territory.
    let in_band = !score_violation && !peak_violation;
    let overshoot = current_score - target.target;
    let claw_active = in_band
        && match target.max_overshoot {
            Some(t) => overshoot > t,
            None => false,
        };
    let loosen_below = if claw_active { p25 } else { f32::NEG_INFINITY };

    prev_scales
        .iter()
        .zip(block_dm.iter())
        .map(|(&prev, &dm)| {
            let delta = if dm > tighten_above {
                let frac = ((dm - tighten_above) / (max - tighten_above).max(1e-6)).clamp(0.0, 1.0);
                -hp::MAX_SCALE_DELTA * frac
            } else if dm < loosen_below {
                let frac = ((loosen_below - dm) / loosen_below.max(1e-6)).clamp(0.0, 1.0);
                hp::MAX_SCALE_DELTA * frac
            } else {
                0.0
            };
            (prev + delta).clamp(hp::MIN_SCALE, hp::MAX_SCALE)
        })
        .collect()
}

/// Lay out a flat per-block scale vector into the iMCU-row schedule
/// shape that [`AqController::adjust`] consumes.
///
/// `flat[bx + by * blocks_w]` is the scale for block `(bx, by)`.
/// One iMCU row covers `v_samp` block rows of width `blocks_w`.
fn flat_to_imcu_schedule(
    flat: &[f32],
    blocks_w: usize,
    blocks_h: usize,
    v_samp: usize,
) -> alloc::vec::Vec<alloc::vec::Vec<f32>> {
    let imcu_rows = blocks_h.div_ceil(v_samp);
    let mut out = alloc::vec::Vec::with_capacity(imcu_rows);
    for imcu_idx in 0..imcu_rows {
        let mut row = alloc::vec::Vec::with_capacity(blocks_w * v_samp);
        for vrow in 0..v_samp {
            let by = imcu_idx * v_samp + vrow;
            if by >= blocks_h {
                break;
            }
            for bx in 0..blocks_w {
                row.push(flat[by * blocks_w + bx]);
            }
        }
        out.push(row);
    }
    out
}

/// Aggregate a full-resolution diffmap (`width × height` f32) into a
/// per-block mean. Truncating divides for non-multiples-of-8 dims —
/// the right/bottom edge sliver is dropped (matches the encoder's
/// 8×8 block grid).
fn aggregate_diffmap_to_blocks(
    diffmap: &[f32],
    width: usize,
    height: usize,
) -> alloc::vec::Vec<f32> {
    const BLOCK: usize = 8;
    let blocks_w = width / BLOCK;
    let blocks_h = height / BLOCK;
    let mut out = alloc::vec::Vec::with_capacity(blocks_w * blocks_h);
    for by in 0..blocks_h {
        for bx in 0..blocks_w {
            let mut sum = 0.0f64;
            for ly in 0..BLOCK {
                for lx in 0..BLOCK {
                    let x = bx * BLOCK + lx;
                    let y = by * BLOCK + ly;
                    sum += diffmap[y * width + x] as f64;
                }
            }
            out.push((sum / (BLOCK * BLOCK) as f64) as f32);
        }
    }
    out
}

/// Configuration, dimensions, and pixels captured by a
/// [`crate::encode::BytesEncoder`] in target-zq mode. The iteration loop
/// rebuilds a fresh `BytesEncoder` per pass from these.
///
/// Per-image metadata (ICC, EXIF, XMP) is intentionally NOT carried here
/// — it's injected into the final JPEG bytes by `BytesEncoder` after the
/// iteration loop returns, the same post-encode injection helpers used
/// by the streaming `finish_into` path. The iteration loop itself
/// produces metadata-free JPEGs to keep its inner encodes fast.
#[cfg(feature = "target-zq")]
pub(crate) struct IterationContext<'a> {
    pub(crate) config: &'a crate::encode::EncoderConfig,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) layout: crate::encode::PixelLayout,
    pub(crate) pixels: &'a [u8],
    pub(crate) target: ZqTarget,
}

/// Run the closed-loop target-zq iteration. Returns the JPEG bytes of
/// the best feasible candidate observed (smallest bytes among all
/// passes that meet active goals; if no pass meets goals, the highest-
/// score pass).
///
/// Errors only when:
/// - Encoding errors out at any pass (propagated).
/// - `max_undershoot` is set on `target` and the final achieved score
///   is below `target.target - max_undershoot`.
/// - `block_artifact.max_overshoot` is set and the final max-block
///   artifact exceeds `ceiling + max_overshoot`.
#[cfg(feature = "target-zq")]
pub(crate) fn run_iteration_loop(
    ctx: IterationContext<'_>,
) -> crate::error::Result<(alloc::vec::Vec<u8>, EncodeMetrics)> {
    use crate::encode::{EncoderConfig, Quality};
    use zensim::{RgbSlice, Zensim, ZensimProfile};

    let blocks_w = (ctx.width as usize) / 8;
    let blocks_h = (ctx.height as usize) / 8;
    if blocks_w == 0 || blocks_h == 0 {
        // Image too small to do per-block correction; fall back to the
        // single-pass starting-q encode.
        return run_single_pass(&ctx, None);
    }

    // Vertical sampling factor of the luma plane in iMCUs. Determines
    // how many block rows belong to a single iMCU emission. Covers
    // every supported color mode + subsampling combo:
    //
    //   YCbCr 4:4:4, 4:2:2  → v_samp = 1
    //   YCbCr 4:2:0, 4:4:0  → v_samp = 2
    //   XYB Full            → v_samp = 1
    //   XYB BQuarter        → v_samp = 2  (B is 2×2 downsampled, max v_samp=2)
    //   Grayscale           → v_samp = 1
    let v_samp = match ctx.config.color_mode {
        crate::encode::ColorMode::YCbCr { subsampling } => {
            subsampling.v_samp_factor_luma() as usize
        }
        crate::encode::ColorMode::Xyb { subsampling } => match subsampling {
            crate::encode::XybSubsampling::BQuarter => 2,
            crate::encode::XybSubsampling::Full => 1,
        },
        crate::encode::ColorMode::Grayscale => 1,
    };

    // Build the source ImageSource for zensim. Layout currently must be
    // RGB8 — other layouts would require zenpixels conversion to drive
    // zensim's per-channel pipeline. (XYB encoding accepts RGB8 input
    // and converts internally; the source stays in sRGB8 here.)
    if ctx.layout != crate::encode::PixelLayout::Rgb8Srgb {
        // For non-RGB8 layouts, fall back to single-pass — extending
        // the closed loop to other layouts is future work.
        return run_single_pass(&ctx, None);
    }

    let z = Zensim::new(ZensimProfile::latest());
    let src_chunks: &[[u8; 3]] = bytemuck_chunks(ctx.pixels);
    let src_slice = RgbSlice::new(src_chunks, ctx.width as usize, ctx.height as usize);
    let pre = match z.precompute_reference(&src_slice) {
        Ok(p) => p,
        Err(_) => return run_single_pass(&ctx, None),
    };

    // Pass 0: streaming-AQ baseline. Substitute Quality::Zq* with the
    // resolved starting jpegli q to avoid recursion.
    let starting_q = ctx.config.quality.to_internal();
    let mut pass_config: EncoderConfig = ctx.config.clone();
    pass_config = pass_config.quality(Quality::ApproxJpegli(starting_q));

    let bytes0 = encode_pass(&pass_config, &ctx, None)?;
    let (score0, dm0) = measure(&z, &pre, &bytes0, ctx.width, ctx.height)?;
    let max0 = max_block(&dm0);

    let mut best_bytes = bytes0.clone();
    let mut best_score = score0;
    let mut best_max = max0;
    let mut best_feasible = is_feasible(score0, max0, &ctx.target);
    let mut passes_used: u8 = 1;

    if !is_feasible(score0, max0, &ctx.target) || ctx.target.max_passes == 0 {
        // No iteration budget, or pass 0 already infeasible and we still
        // have to exit cleanly: enter the loop below.
    } else if let Some(t) = ctx.target.max_overshoot {
        if score0 - ctx.target.target <= t {
            // Already in band; ship pass 0.
            return finalize(best_bytes, score0, max0, passes_used, &ctx.target);
        }
    } else {
        // No claw-back budget configured; ship first feasible pass.
        return finalize(best_bytes, score0, max0, passes_used, &ctx.target);
    }

    let mut current_scales = alloc::vec![1.0f32; blocks_w * blocks_h];
    let mut current_dm = dm0;
    let mut current_score = score0;
    let mut current_max = max0;

    for _pass in 1..=ctx.target.max_passes {
        let next = next_scales(
            &current_scales,
            &current_dm,
            current_score,
            current_max,
            &ctx.target,
        );
        let schedule = flat_to_imcu_schedule(&next, blocks_w, blocks_h, v_samp);
        let ctrl: Box<dyn AqController> = Box::new(ScalingController { scales: schedule });
        let bytes_n = encode_pass(&pass_config, &ctx, Some(ctrl))?;
        let (score_n, dm_n) = measure(&z, &pre, &bytes_n, ctx.width, ctx.height)?;
        let max_n = max_block(&dm_n);
        passes_used = passes_used.saturating_add(1);

        let cand_feasible = is_feasible(score_n, max_n, &ctx.target);
        // Best-tracking: prefer feasible over infeasible, then smallest bytes.
        let take = match (best_feasible, cand_feasible) {
            (false, true) => true,
            (true, false) => false,
            (true, true) => bytes_n.len() < best_bytes.len(),
            (false, false) => score_n > best_score,
        };
        if take {
            best_bytes = bytes_n;
            best_score = score_n;
            best_max = max_n;
            best_feasible = cand_feasible;
        }

        // Stop early if we just landed in the comfort band.
        let in_band = cand_feasible
            && match ctx.target.max_overshoot {
                Some(t) => (score_n - ctx.target.target) <= t,
                None => true,
            };
        if in_band {
            break;
        }

        current_scales = next;
        current_dm = dm_n;
        current_score = score_n;
        current_max = max_n;
    }

    finalize(best_bytes, best_score, best_max, passes_used, &ctx.target)
}

#[cfg(feature = "target-zq")]
fn finalize(
    bytes: alloc::vec::Vec<u8>,
    score: f32,
    max_block: f32,
    passes_used: u8,
    target: &ZqTarget,
) -> crate::error::Result<(alloc::vec::Vec<u8>, EncodeMetrics)> {
    // Strict-mode failure checks (see `ZqTarget::max_undershoot` and
    // `BlockArtifactBound::max_overshoot` docs).
    if let Some(slack) = target.max_undershoot {
        if score < target.target - slack {
            return Err(crate::error::Error::invalid_config(alloc::format!(
                "target-zq encoder achieved score {:.3}, below floor {:.3} \
                 (max_undershoot = {:.3}) after {} passes",
                score,
                target.target,
                slack,
                passes_used
            )));
        }
    }
    if let Some(b) = target.block_artifact {
        if let Some(slack) = b.max_overshoot {
            if max_block > b.ceiling + slack {
                return Err(crate::error::Error::invalid_config(alloc::format!(
                    "target-zq encoder achieved max-block-artifact {:.4}, \
                     above ceiling {:.4} (max_overshoot = {:.4}) after {} passes",
                    max_block,
                    b.ceiling,
                    slack,
                    passes_used
                )));
            }
        }
    }
    let targets_met = is_feasible(score, max_block, target);
    let bytes_len = bytes.len();
    Ok((
        bytes,
        EncodeMetrics {
            achieved_score: score,
            achieved_max_block_artifact: max_block,
            passes_used,
            bytes: bytes_len,
            targets_met,
        },
    ))
}

#[cfg(feature = "target-zq")]
fn is_feasible(score: f32, max_block: f32, target: &ZqTarget) -> bool {
    let score_ok = score >= target.target;
    let peak_ok = match target.block_artifact {
        Some(b) => max_block <= b.ceiling,
        None => true,
    };
    score_ok && peak_ok
}

#[cfg(feature = "target-zq")]
fn max_block(dm: &[f32]) -> f32 {
    dm.iter().copied().fold(0.0f32, f32::max)
}

/// Encode pixels via a fresh [`crate::encode::BytesEncoder`] built from
/// `pass_config`. Optional `controller` is installed before pushing.
#[cfg(feature = "target-zq")]
fn encode_pass(
    pass_config: &crate::encode::EncoderConfig,
    ctx: &IterationContext<'_>,
    controller: Option<Box<dyn AqController>>,
) -> crate::error::Result<alloc::vec::Vec<u8>> {
    use enough::Unstoppable;
    let mut enc = pass_config.encode_from_bytes(ctx.width, ctx.height, ctx.layout)?;
    if let Some(c) = controller {
        enc.set_aq_controller(Some(c));
    }
    enc.push_packed(ctx.pixels, Unstoppable)?;
    enc.finish()
}

/// Decode `jpeg`, compute zensim diffmap against `pre`, return
/// (score, per-block diffmap).
#[cfg(feature = "target-zq")]
fn measure(
    z: &zensim::Zensim,
    pre: &zensim::PrecomputedReference,
    jpeg: &[u8],
    width: u32,
    height: u32,
) -> crate::error::Result<(f32, alloc::vec::Vec<f32>)> {
    use enough::Unstoppable;
    use zensim::{DiffmapWeighting, RgbSlice};
    let dec = crate::decode::Decoder::new()
        .decode(jpeg, Unstoppable)
        .map_err(|e| {
            crate::error::Error::invalid_config(alloc::format!(
                "decode for measurement failed: {e}"
            ))
        })?;
    let pixels = dec.into_pixels_u8().ok_or_else(|| {
        crate::error::Error::invalid_config("decoder returned non-u8 pixels".into())
    })?;
    let chunks: &[[u8; 3]] = bytemuck_chunks(&pixels);
    let dec_slice = RgbSlice::new(chunks, width as usize, height as usize);
    let res = z
        .compute_with_ref_and_diffmap(pre, &dec_slice, DiffmapWeighting::Trained)
        .map_err(|e| {
            crate::error::Error::invalid_config(alloc::format!(
                "zensim compute_with_ref_and_diffmap failed: {e}"
            ))
        })?;
    let dm = aggregate_diffmap_to_blocks(res.diffmap(), width as usize, height as usize);
    Ok((res.score() as f32, dm))
}

/// Single-pass fallback: encode at the resolved starting q, no
/// controller, no measurement. Used when target-zq mode can't run the
/// full loop (image too small, layout unsupported, decoder feature off,
/// etc.).
#[cfg(feature = "target-zq")]
fn run_single_pass(
    ctx: &IterationContext<'_>,
    _controller: Option<Box<dyn AqController>>,
) -> crate::error::Result<(alloc::vec::Vec<u8>, EncodeMetrics)> {
    use crate::encode::{EncoderConfig, Quality};
    let starting_q = ctx.config.quality.to_internal();
    let mut pass_config: EncoderConfig = ctx.config.clone();
    pass_config = pass_config.quality(Quality::ApproxJpegli(starting_q));
    let bytes = encode_pass(&pass_config, ctx, None)?;
    let bytes_len = bytes.len();
    Ok((
        bytes,
        EncodeMetrics {
            achieved_score: f32::NAN,
            achieved_max_block_artifact: f32::NAN,
            passes_used: 1,
            bytes: bytes_len,
            targets_met: true,
        },
    ))
}

/// Reinterpret a packed RGB byte slice as `[u8; 3]` chunks. Length must
/// be a multiple of 3; truncates excess.
#[cfg(feature = "target-zq")]
fn bytemuck_chunks(pixels: &[u8]) -> &[[u8; 3]] {
    let n = pixels.len() / 3;
    // SAFETY-equivalent via the `as_chunks` slice method (stable on
    // recent rustc). All bytes in `pixels` are valid; we cast a flat
    // u8 slice to a fixed-stride array slice of the same byte layout.
    let (chunks, _) = pixels.as_chunks::<3>();
    let _ = n;
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zq_target_default_is_sensible() {
        let t = ZqTarget::default();
        assert_eq!(t.target, 80.0);
        assert_eq!(t.max_overshoot, Some(1.5));
        assert_eq!(t.max_undershoot, None);
        assert!(t.block_artifact.is_none());
        assert_eq!(t.max_passes, 2);
    }

    #[test]
    fn zq_target_builder_chains() {
        let t = ZqTarget::new(85.0)
            .with_max_overshoot(Some(0.5))
            .with_max_undershoot(Some(2.0))
            .with_max_passes(3)
            .with_block_artifact(Some(BlockArtifactBound::new(0.012)));
        assert_eq!(t.target, 85.0);
        assert_eq!(t.max_overshoot, Some(0.5));
        assert_eq!(t.max_undershoot, Some(2.0));
        assert_eq!(t.max_passes, 3);
        assert!(t.block_artifact.is_some());
        assert_eq!(t.block_artifact.unwrap().ceiling, 0.012);
    }

    #[test]
    fn block_artifact_bound_default_is_best_effort() {
        let b = BlockArtifactBound::new(0.020);
        assert_eq!(b.ceiling, 0.020);
        assert_eq!(b.max_overshoot, None);
        assert_eq!(b.max_undershoot, None);
    }

    #[test]
    fn zq_calibration_is_monotonic_and_in_range() {
        // Small smoke check: starting q increases with zq, and stays
        // within the jpegli-q range.
        let mut prev = 0.0f32;
        for zq in (40..=95).step_by(5) {
            let q = zq_to_starting_jpegli_q(zq as f32);
            assert!(q > prev, "non-monotonic at zq={zq}: {prev} -> {q}");
            assert!((1.0..=100.0).contains(&q), "out of range at zq={zq}: {q}");
            prev = q;
        }
    }
}
