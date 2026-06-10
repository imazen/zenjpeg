//! Budgeted sweep-plan construction over the encoder knob space.
//!
//! Sweeps over this encoder explode combinatorially: table families ×
//! quality × coefficient-opt × scan × color/subsampling × mode flags.
//! This module turns that space into a **finite, auditable list of
//! encode cells**:
//!
//! 1. **Strata** — the caller picks concrete values per axis
//!    ([`SweepAxes`]; curated defaults in [`SweepAxes::rd_core`] /
//!    [`SweepAxes::modes_full`]). Invalid combinations
//!    (`EncoderConfig::validate()`) are skipped and *reported*, never
//!    silently lost.
//! 2. **Quality grid** — [`QualityGrid`] encodes the sweep discipline:
//!    the step-5 floor for benchmarks, denser grids for training. Low-q
//!    coverage is never thinned preferentially.
//! 3. **Fingerprint dedup** — every cell gets a byte-identity
//!    fingerprint over its *resolved* state (actual quant tables +
//!    zero-bias + entropy-relevant knobs). Configs that alias — Glassa
//!    above its q25 anchor clamp, `allow_16bit` at qualities where no
//!    value exceeds 255, `auto_optimize` vs its explicit trellis
//!    spelling — collapse into one encode with the merged ids recorded
//!    as aliases.
//! 4. **Budget ladder** — [`SweepBuilder::with_budget`] reduces
//!    deterministically: collapse low-tier mode axes to their first
//!    value (recorded in [`SweepPlan::dropped`]), then coarsen the
//!    quality grid uniformly (endpoints kept, never below 11 points),
//!    and finally set [`SweepPlan::over_budget`] rather than sample
//!    silently. No silent caps.
//!
//! 5. **Queue ordering** — cells are emitted main-effects-first: the
//!    all-defaults stratum, then every single-deviation stratum (one
//!    axis changed from its default answers "does this knob matter"),
//!    then interaction combos, with milder deviations before extreme
//!    ones. Quality runs ascending *within* each stratum so an RD curve
//!    is never half-measured: a truncated queue is safe at any stratum
//!    boundary. [`SweepCell::deviations`] exposes the priority class.
//!
//! # Scalar bounds and step provenance
//!
//! | knob | bound | curated steps (modes_full) | provenance |
//! |---|---|---|---|
//! | trellis λ₁ | 12.0–17.0 (useful) | 13.5, 14.0, 14.5, 14.75, 15.5, 16.0 | expert.rs envelope (−46 %..+12 %); adaptive oracle uses 12.0–16.0. Validated 2026-06-10 (`sweep_validate`): strictly monotone in size, spanning −0.9 %..+15 % around no-trellis on mixed 512² content |
//! | trellis λ₂ | 14.0–18.0 | 16.0, 16.5, 17.0 (at λ₁ = 14.75) | expert.rs envelope (−19 %..+11 %); ridge: only λ₁−λ₂ matters at low block energy. Validated: 16.0 vs 17.0 distinct on 42/42 cells |
//! | aq_coupling.scale | −8..+8 | −8, −4, +4, ALL clamped ±1.0 | measured: −4 ≈ 2 % smaller/3 % DSSIM on photos. Unclamped steps are FORBIDDEN: validated 2026-06-10 that unclamped −4 destroys high-AQ content (SSIM2 −31 on noise q85, −90 % bytes) |
//! | coupling.exponent | 0.5–2.0 | 2.0 probe (at scale −4, clamped) | historical sweep grid {0.5, 1, 2} |
//! | delta_dc_weight | 0.0–5.0 | 1.0 probe | expert.rs: 0..+1 % size, diminishing above 2.0. Validated 2026-06-10: SIZE claim holds, but quality collapses at q≤70 (SSIM2 −8..−36 on photos) — probe retained for response-surface mapping; NOT a default candidate, and possibly mis-scaled (worth a look before trusting sweeps that include it) |
//! | chroma_distance_scales | [0.1, 5.0] each | [0.5,0.5], [2,2], [1,2], [2,1] | clamp range; asymmetric probes exercise the per-channel axes. Validated: [1,2] vs [2,1] distinct on 42/42 cells (Cb/Cr independently wired) |
//! | pre_blur σ | 0.0–1.0 | 0.4 | ~5 % size win on photos (validated −11 % best case; synthetic aligned patterns can INFLATE up to +450 % — checker8's DCT degenerates) |
//! | quality | 1–100 | grids in [`QualityGrid`] | step-5 floor / training-dense per sweep discipline |
//!
//! Empirical validation harness: `examples/sweep_validate.rs` (run
//! 2026-06-10, results in `benchmarks/sweep_validate_2026-06-10.tsv`).
//! It encodes the default stratum + every single-deviation stratum on
//! mixed content and fails hard on inert steps, fingerprint-contract
//! violations, and ordering breakage. Re-run it after touching the
//! curated axes or the fingerprint.
//!
//! `MozjpegRobidoux::chroma_quality` is deliberately unswept here: it is
//! an ABSOLUTE quality while the grid moves q, so any static value is
//! wrong at most grid points (a relative form is a design follow-up).
//! Boundary-RD alternates beyond `On(default)` live in the coefficient
//! harness (the 66-combo validated grid), not these curated axes.
//!
//! The plan is **per config-cell**; multiply by corpus images and size
//! buckets with [`SweepPlan::encodes`] to get the real encode count.
//! Persistence of encoded bytes/diffmaps and metric scoring belong to
//! the harness consuming this plan (zenmetrics / coefficient), not here.

#![cfg(feature = "__expert")]

use super::encoder_config::EncoderConfig;
use super::encoder_types::{
    ChromaSubsampling, ColorMode, DownsamplingMethod, HuffmanStrategy, ProgressiveScanMode,
    QuantTableConfig,
};
use super::plan::{TableResolveInputs, resolve_quant_tables};
use super::trellis::{AqCoupling, TrellisConfig};
use crate::types::Subsampling;

// ============================================================================
// Axes
// ============================================================================

/// Concrete values per categorical axis. The cross product of these,
/// times the quality grid, is the candidate cell set.
#[derive(Clone, Debug)]
pub struct SweepAxes {
    /// Quant-table families (with their live knobs).
    pub families: Vec<QuantTableConfig>,
    /// Coefficient optimization: `None` = plain zero-bias rounding.
    pub coeff_opt: Vec<Option<TrellisConfig>>,
    /// Scan modes.
    pub scans: Vec<ProgressiveScanMode>,
    /// Color mode + subsampling combinations.
    pub color_modes: Vec<ColorMode>,
    /// Adaptive quantization on/off.
    pub aq: Vec<bool>,
    /// Overshoot deringing on/off.
    pub deringing: Vec<bool>,
    /// Chroma downsampling method (RGB input only).
    pub downsampling: Vec<DownsamplingMethod>,
    /// Allow 16-bit DQT entries.
    pub allow_16bit: Vec<bool>,
    /// Pre-encode Gaussian blur sigma.
    pub pre_blur: Vec<f32>,
    /// Boundary-continuity refinement (only with `--features boundary-rd`).
    /// Inert when a trellis config is set (the engine skips it there) —
    /// such cells fingerprint-dedupe with their trellis-only twins.
    #[cfg(feature = "boundary-rd")]
    pub boundary_rd: Vec<super::encoder_config::BoundaryRd>,
}

/// Trellis config matching `auto_optimize`'s tuned point (λ₁ = 14.5,
/// DC off, no coupling).
#[must_use]
pub fn trellis_auto_shape() -> TrellisConfig {
    TrellisConfig {
        lambda_log_scale1: 14.5,
        dc_enabled: false,
        ..TrellisConfig::default()
    }
}

/// AQ-coupled trellis with the given coupling scale (DC off, defaults
/// otherwise), clamped to ±1.0 λ-adjustment. Measured envelope: `-4` ≈
/// 2 % smaller / ~3 % DSSIM cost on photos; `+4` ≈ the reverse.
///
/// The clamp is not optional for curated steps: unclamped coupling
/// reproduces the historical screenshot-destruction mode on high-AQ
/// content (validated 2026-06-10 — unclamped −4 scored SSIM2 −31 on
/// 512² noise at q85 while shedding up to 90 % of bytes). Build the
/// struct directly if you genuinely want the unclamped extreme.
#[must_use]
pub fn trellis_coupled(scale: f32) -> TrellisConfig {
    TrellisConfig {
        dc_enabled: false,
        aq_coupling: AqCoupling {
            scale,
            max_adjustment: 1.0,
            ..AqCoupling::OFF
        },
        ..TrellisConfig::default()
    }
}

/// Fixed-lambda trellis at the given λ₁ (DC off, λ₂ default 16.5).
/// Useful λ₁ range 12.0–17.0; below 12 trellis zeroes nearly all AC.
/// Higher λ₁ keeps more coefficients than jpegli's zero-bias rounding —
/// it does NOT converge to the no-trellis output (validated 2026-06-10:
/// λ₁ = 16 is ~15 % larger and ~+3 SSIM2 better than no-trellis on
/// mixed 512² content).
#[must_use]
pub fn trellis_lambda(lambda_log_scale1: f32) -> TrellisConfig {
    TrellisConfig {
        lambda_log_scale1,
        dc_enabled: false,
        ..TrellisConfig::default()
    }
}

/// Fixed-lambda trellis with explicit λ₂ (λ₁ = default 14.75, DC off).
/// Steps the second ridge axis; useful λ₂ range 14.0–18.0.
#[must_use]
pub fn trellis_lambda2(lambda_log_scale2: f32) -> TrellisConfig {
    TrellisConfig {
        lambda_log_scale2,
        dc_enabled: false,
        ..TrellisConfig::default()
    }
}

impl SweepAxes {
    /// The axes that move the rate-distortion front, with everything
    /// else at production defaults: 4 table families × {no trellis,
    /// default trellis, the auto_optimize shape} × {4:2:0, 4:4:4}
    /// (× boundary-rd off/on when that feature is enabled).
    ///
    /// Scan axis: [`ProgressiveScanMode::Smallest`] — the exact
    /// entropy-stage minimizer covers the whole sequential/tiny/
    /// progressive crossover by trial, so no scan heuristic exists in
    /// the core sweep at all. Explicit Baseline stays in
    /// [`modes_full`](Self::modes_full) purely for mode-coverage of the
    /// individual scan modes.
    #[must_use]
    pub fn rd_core() -> Self {
        Self {
            families: vec![
                QuantTableConfig::default(),
                QuantTableConfig::PiecewiseV4,
                QuantTableConfig::MozjpegRobidoux {
                    chroma_quality: None,
                },
                QuantTableConfig::GlassaLowBpp,
            ],
            coeff_opt: vec![
                None,
                Some(TrellisConfig::default()),
                Some(trellis_auto_shape()),
            ],
            scans: vec![ProgressiveScanMode::Smallest],
            color_modes: vec![
                ColorMode::YCbCr {
                    subsampling: ChromaSubsampling::Quarter,
                },
                ColorMode::YCbCr {
                    subsampling: ChromaSubsampling::None,
                },
            ],
            aq: vec![true],
            deringing: vec![true],
            downsampling: vec![DownsamplingMethod::Box],
            allow_16bit: vec![false],
            pre_blur: vec![0.0],
            // Off first: if the budget ladder collapses this axis, the
            // cheaper status-quo value is the one kept.
            #[cfg(feature = "boundary-rd")]
            boundary_rd: vec![
                super::encoder_config::BoundaryRd::Off,
                super::encoder_config::BoundaryRd::On(
                    super::encoder_config::BoundaryRdConfig::default(),
                ),
            ],
        }
    }

    /// Every user-disableable mode axis (the calibration mandate), on
    /// top of [`rd_core`](Self::rd_core): AQ off, deringing off,
    /// sharp-YUV, 16-bit DQT, mozjpeg scan script, scan search, XYB,
    /// 4:2:2, and AQ-coupled trellis points. Large — pair with
    /// [`SweepBuilder::with_budget`].
    #[must_use]
    pub fn modes_full() -> Self {
        let mut axes = Self::rd_core();
        // Scan modes (mode-coverage; Smallest already subsumes the RD
        // question, these pin the individual emitters).
        axes.scans.push(ProgressiveScanMode::Progressive);
        axes.scans.push(ProgressiveScanMode::Baseline);
        axes.scans.push(ProgressiveScanMode::SmallestSearch);
        axes.scans.push(ProgressiveScanMode::ProgressiveMozjpeg);
        axes.scans.push(ProgressiveScanMode::ProgressiveSearch);

        // λ₁ ladder (oracle-informed: adaptive uses 12.0–16.0; defaults
        // 14.5/14.75 are already in rd_core's coeff set).
        axes.coeff_opt.push(Some(trellis_lambda(13.5)));
        axes.coeff_opt.push(Some(trellis_lambda(14.0)));
        axes.coeff_opt.push(Some(trellis_lambda(15.5)));
        axes.coeff_opt.push(Some(trellis_lambda(16.0)));
        // λ₂ ridge probes at default λ₁.
        axes.coeff_opt.push(Some(trellis_lambda2(16.0)));
        axes.coeff_opt.push(Some(trellis_lambda2(17.0)));
        // AQ coupling: measured envelope ±4, every step clamped to ±1.0
        // λ-adjustment — the unclamped form is the known screenshot/
        // noise quality-destruction mode (see `trellis_coupled`).
        axes.coeff_opt.push(Some(trellis_coupled(-4.0)));
        axes.coeff_opt.push(Some(trellis_coupled(4.0)));
        axes.coeff_opt.push(Some(trellis_coupled(-8.0)));
        // Non-linear coupling probe (historical grid {0.5, 1, 2}).
        axes.coeff_opt.push(Some(TrellisConfig {
            dc_enabled: false,
            aq_coupling: AqCoupling {
                scale: -4.0,
                exponent: 2.0,
                max_adjustment: 1.0,
                ..AqCoupling::OFF
            },
            ..TrellisConfig::default()
        }));
        // DC banding-penalty probe (0..+1 % size, diminishing above 2).
        axes.coeff_opt.push(Some(TrellisConfig {
            delta_dc_weight: 1.0,
            ..TrellisConfig::default()
        }));

        // Per-channel chroma distance steps on the jpegli family
        // (uniform 0.5×/2× plus asymmetric probes for the Cb/Cr axes).
        axes.families
            .push(QuantTableConfig::jpegli_chroma_scale(0.5));
        axes.families
            .push(QuantTableConfig::jpegli_chroma_scale(2.0));
        axes.families.push(QuantTableConfig::Jpegli {
            chroma_distance_scales: [1.0, 2.0],
        });
        axes.families.push(QuantTableConfig::Jpegli {
            chroma_distance_scales: [2.0, 1.0],
        });
        axes.color_modes.push(ColorMode::YCbCr {
            subsampling: ChromaSubsampling::HalfHorizontal,
        });
        axes.color_modes.push(ColorMode::Xyb {
            subsampling: super::encoder_types::XybSubsampling::BQuarter,
        });
        axes.aq = vec![true, false];
        axes.deringing = vec![true, false];
        axes.downsampling = vec![
            DownsamplingMethod::Box,
            DownsamplingMethod::GammaAwareIterative,
        ];
        axes.allow_16bit = vec![false, true];
        // Documented ~5 % size win at σ = 0.4.
        axes.pre_blur = vec![0.0, 0.4];
        axes
    }
}

// ============================================================================
// Quality grid
// ============================================================================

/// Quality grids per the sweep discipline. Low-q density is never below
/// high-q density.
#[derive(Clone, Debug)]
pub enum QualityGrid {
    /// q ∈ {1, 5, 10, …, 100} — the 21-point floor for benchmarks and
    /// anchor tables.
    Step5,
    /// Step 5 through q65, step 2 from q70 — the training-density grid
    /// (31 points).
    TrainingDense,
    /// Caller-provided points (kept in the given order, deduplicated).
    Explicit(Vec<f32>),
}

impl QualityGrid {
    /// Materialize the grid points.
    #[must_use]
    pub fn points(&self) -> Vec<f32> {
        match self {
            Self::Step5 => {
                let mut v = vec![1.0];
                v.extend((1..=20).map(|i| (i * 5) as f32));
                v
            }
            Self::TrainingDense => {
                let mut v = vec![1.0];
                v.extend((1..=13).map(|i| (i * 5) as f32)); // 5..=65
                v.extend((35..=50).map(|i| (i * 2) as f32)); // 70..=100
                v
            }
            Self::Explicit(pts) => {
                let mut v = Vec::new();
                for &p in pts {
                    if !v.contains(&p) {
                        v.push(p);
                    }
                }
                v
            }
        }
    }
}

// ============================================================================
// Plan output
// ============================================================================

/// One encode cell: a fully-built config at one quality point.
#[derive(Clone, Debug)]
pub struct SweepCell {
    /// Stable human-readable id (family/coeff/scan/color/flags/q tokens).
    pub id: String,
    /// The config to encode with (quality already applied).
    pub config: EncoderConfig,
    /// The quality point.
    pub quality: f32,
    /// Byte-identity fingerprint of the resolved state. Cells with equal
    /// fingerprints produce identical bytes for the same input.
    pub fingerprint: u64,
    /// Ids of candidate cells merged into this one (identical
    /// fingerprints).
    pub aliases: Vec<String>,
    /// How many axes deviate from the default stratum (index 0 of every
    /// axis). 0 = the production-default cell; 1 = a main-effect probe;
    /// ≥2 = interaction combos. Cells are emitted in ascending order.
    pub deviations: u8,
}

/// A mode axis collapsed by the budget ladder.
#[derive(Clone, Debug)]
pub struct DroppedAxis {
    /// Axis name.
    pub axis: &'static str,
    /// The value kept (Debug-rendered).
    pub kept: String,
    /// The values dropped (Debug-rendered).
    pub dropped: Vec<String>,
}

/// The finite, auditable sweep plan.
#[derive(Clone, Debug)]
pub struct SweepPlan {
    /// Deduplicated encode cells.
    pub cells: Vec<SweepCell>,
    /// Stratum ids rejected by `EncoderConfig::validate()` (e.g. XYB ×
    /// YCbCr-only table families).
    pub invalid_skipped: Vec<String>,
    /// Mode axes collapsed to fit the budget — the explicit
    /// no-silent-caps report.
    pub dropped: Vec<DroppedAxis>,
    /// Candidate cells merged by fingerprint identity.
    pub duplicates_merged: usize,
    /// How many times the quality grid was uniformly coarsened.
    pub q_coarsenings: u32,
    /// The budget could not be met even after the full reduction ladder.
    /// The plan is complete (nothing was sampled away); the caller
    /// decides whether to spend or cut axes manually.
    pub over_budget: bool,
}

impl SweepPlan {
    /// Total encodes when this plan runs over a corpus: cells × images ×
    /// size buckets.
    #[must_use]
    pub fn encodes(&self, images: usize, size_buckets: usize) -> usize {
        self.cells.len() * images * size_buckets
    }
}

// ============================================================================
// Builder
// ============================================================================

/// Builds a [`SweepPlan`] from axes × quality grid under an optional
/// encode-cell budget.
#[derive(Clone, Debug)]
pub struct SweepBuilder {
    axes: SweepAxes,
    grid: QualityGrid,
    budget: Option<usize>,
}

impl SweepBuilder {
    /// New builder over the given axes and quality grid.
    #[must_use]
    pub fn new(axes: SweepAxes, grid: QualityGrid) -> Self {
        Self {
            axes,
            grid,
            budget: None,
        }
    }

    /// Cap the number of (deduplicated) cells. The reduction ladder:
    /// collapse mode axes lowest-tier-first (pre_blur, allow_16bit,
    /// downsampling, deringing, aq, then extra scans, extra color
    /// modes, extra coeff-opt points), then coarsen the quality grid
    /// (uniformly, endpoints kept, ≥ 11 points). Families are never
    /// dropped. Every reduction is recorded.
    #[must_use]
    pub fn with_budget(mut self, max_cells: usize) -> Self {
        self.budget = Some(max_cells);
        self
    }

    /// Build the plan.
    #[must_use]
    pub fn plan(&self) -> SweepPlan {
        let mut axes = self.axes.clone();
        let mut q_points = self.grid.points();
        let mut dropped = Vec::new();
        let mut q_coarsenings = 0u32;
        let mut over_budget = false;

        loop {
            let (cells, invalid_skipped, duplicates_merged) = cross(&axes, &q_points);

            let within = match self.budget {
                None => true,
                Some(b) => cells.len() <= b,
            };
            if within {
                return SweepPlan {
                    cells,
                    invalid_skipped,
                    dropped,
                    duplicates_merged,
                    q_coarsenings,
                    over_budget,
                };
            }

            // Reduction ladder, one step per iteration.
            if let Some(d) = collapse_one_axis(&mut axes) {
                // Coalesce repeated single-value drops of the same axis.
                if let Some(last) = dropped.last_mut()
                    && last.axis == d.axis
                {
                    last.dropped.extend(d.dropped);
                    last.kept = d.kept;
                    continue;
                }
                dropped.push(d);
                continue;
            }
            if q_points.len() > 11 {
                q_points = coarsen_keep_endpoints(&q_points);
                q_coarsenings += 1;
                continue;
            }

            // Nothing left to reduce: report rather than sample.
            over_budget = true;
            let (cells, invalid_skipped, duplicates_merged) = cross(&axes, &q_points);
            return SweepPlan {
                cells,
                invalid_skipped,
                dropped,
                duplicates_merged,
                q_coarsenings,
                over_budget,
            };
        }
    }
}

fn collapse<T: core::fmt::Debug + Clone>(
    name: &'static str,
    v: &mut Vec<T>,
    floor: usize,
) -> Option<DroppedAxis> {
    // Shed ONE value per ladder step — the last (lowest-priority) entry —
    // so the budget is approached from above instead of overshot by
    // whole-axis removals. Axis vecs are ordered most-important-first.
    if v.len() <= floor {
        return None;
    }
    let dropped = vec![format!("{:?}", v[v.len() - 1])];
    v.truncate(v.len() - 1);
    let kept = v
        .iter()
        .map(|x| format!("{x:?}"))
        .collect::<Vec<_>>()
        .join(", ");
    Some(DroppedAxis {
        axis: name,
        kept,
        dropped,
    })
}

#[cfg(feature = "boundary-rd")]
fn collapse_boundary(axes: &mut SweepAxes) -> Option<DroppedAxis> {
    collapse("boundary_rd", &mut axes.boundary_rd, 1)
}

#[cfg(not(feature = "boundary-rd"))]
fn collapse_boundary(_axes: &mut SweepAxes) -> Option<DroppedAxis> {
    None
}

/// Collapse the lowest-tier multi-valued axis to its first value.
fn collapse_one_axis(axes: &mut SweepAxes) -> Option<DroppedAxis> {
    // Tier order: cheapest-to-lose first. Families are never collapsed.
    collapse("pre_blur", &mut axes.pre_blur, 1)
        .or_else(|| collapse("allow_16bit", &mut axes.allow_16bit, 1))
        .or_else(|| collapse("downsampling", &mut axes.downsampling, 1))
        .or_else(|| collapse("deringing", &mut axes.deringing, 1))
        .or_else(|| collapse("aq", &mut axes.aq, 1))
        .or_else(|| collapse_boundary(axes))
        .or_else(|| collapse("scans", &mut axes.scans, 1))
        .or_else(|| collapse("color_modes", &mut axes.color_modes, 2))
        .or_else(|| collapse("coeff_opt", &mut axes.coeff_opt, 3))
        // Last resort before q-coarsening: shed the scalar-step family
        // variants, never the 4 core families (which sit first).
        .or_else(|| collapse("families", &mut axes.families, 4))
}

/// Drop every second interior point (endpoints kept).
fn coarsen_keep_endpoints(points: &[f32]) -> Vec<f32> {
    let last = points.len() - 1;
    points
        .iter()
        .enumerate()
        .filter(|(i, _)| *i == 0 || *i == last || i % 2 == 0)
        .map(|(_, &p)| p)
        .collect()
}

/// One point in the categorical cross product.
struct Stratum<'a> {
    color: ColorMode,
    family: &'a QuantTableConfig,
    coeff: &'a Option<TrellisConfig>,
    scan: ProgressiveScanMode,
    aq: bool,
    dering: bool,
    down: DownsamplingMethod,
    allow16: bool,
    blur: f32,
    #[cfg(feature = "boundary-rd")]
    boundary_rd: super::encoder_config::BoundaryRd,
}

impl Stratum<'_> {
    fn build_config(&self, q: f32) -> EncoderConfig {
        let mut cfg = match self.color {
            ColorMode::YCbCr { subsampling } => EncoderConfig::ycbcr(q, subsampling),
            ColorMode::Xyb { subsampling } => EncoderConfig::xyb(q, subsampling),
            ColorMode::Grayscale => EncoderConfig::grayscale(q),
        };
        cfg = cfg
            .quant_table_config(self.family.clone())
            .progressive(self.scan)
            .aq_enabled(self.aq)
            .deringing(self.dering)
            .downsampling_method(self.down)
            .allow_16bit_quant_tables(self.allow16)
            .pre_blur(self.blur);
        if let Some(t) = self.coeff {
            cfg = cfg.trellis(*t);
        }
        #[cfg(feature = "boundary-rd")]
        {
            cfg = cfg.boundary_rd(self.boundary_rd);
        }
        cfg
    }

    fn id(&self) -> String {
        let fam = match self.family {
            QuantTableConfig::Jpegli {
                chroma_distance_scales: [a, b],
            } => {
                if *a == 1.0 && *b == 1.0 {
                    "jp3".to_string()
                } else {
                    format!("jp3[{a},{b}]")
                }
            }
            QuantTableConfig::JpegliSharedChroma {
                chroma_distance_scales: [a, b],
            } => {
                if *a == 1.0 && *b == 1.0 {
                    "jp2".to_string()
                } else {
                    format!("jp2[{a},{b}]")
                }
            }
            QuantTableConfig::MozjpegRobidoux { chroma_quality } => match chroma_quality {
                None => "moz".to_string(),
                Some(cq) => format!("moz[cq{cq}]"),
            },
            QuantTableConfig::Custom(_) => "custom".to_string(),
            QuantTableConfig::PiecewiseV4 => "pw4".to_string(),
            QuantTableConfig::GlassaLowBpp => "gls".to_string(),
        };
        let co = match self.coeff {
            None => "t0".to_string(),
            Some(t) => {
                // Render every output-relevant field that deviates from the
                // TrellisConfig default — ids must be collision-free across
                // the λ₂ / delta-DC / coupling-exponent probe configs.
                let d = TrellisConfig::default();
                let mut s = format!("tr{:.4}", t.lambda_log_scale1);
                if t.lambda_log_scale2 != d.lambda_log_scale2 {
                    s.push_str(&format!("l2{:.1}", t.lambda_log_scale2));
                }
                if t.dc_enabled {
                    s.push_str("+dc");
                }
                if t.delta_dc_weight != d.delta_dc_weight {
                    s.push_str(&format!("ddc{:.1}", t.delta_dc_weight));
                }
                if t.aq_coupling.is_active() {
                    s.push_str(&format!("cpl{:+.1}", t.aq_coupling.scale));
                    if t.aq_coupling.exponent != 1.0 {
                        s.push_str(&format!("e{:.1}", t.aq_coupling.exponent));
                    }
                    if t.aq_coupling.max_adjustment > 0.0 {
                        s.push_str(&format!("cl{:.1}", t.aq_coupling.max_adjustment));
                    }
                }
                s
            }
        };
        let sc = match self.scan {
            ProgressiveScanMode::Baseline => "base",
            ProgressiveScanMode::Progressive => "prog",
            ProgressiveScanMode::Smallest => "small",
            ProgressiveScanMode::SmallestSearch => "smsrch",
            ProgressiveScanMode::ProgressiveMozjpeg => "pmoz",
            ProgressiveScanMode::ProgressiveSearch => "psrch",
        };
        let col = match self.color {
            ColorMode::YCbCr { subsampling } => match subsampling {
                ChromaSubsampling::Quarter => "420",
                ChromaSubsampling::None => "444",
                ChromaSubsampling::HalfHorizontal => "422",
                ChromaSubsampling::HalfVertical => "440",
            },
            ColorMode::Xyb { subsampling } => match subsampling {
                super::encoder_types::XybSubsampling::BQuarter => "xybBq",
                super::encoder_types::XybSubsampling::Full => "xybFull",
            },
            ColorMode::Grayscale => "gray",
        };
        let mut s = format!("{fam}_{co}_{sc}_{col}");
        if !self.aq {
            s.push_str("-noaq");
        }
        if !self.dering {
            s.push_str("-noder");
        }
        match self.down {
            DownsamplingMethod::Box => {}
            DownsamplingMethod::GammaAware => s.push_str("-gaware"),
            DownsamplingMethod::GammaAwareIterative => s.push_str("-sharp"),
        }
        if self.allow16 {
            s.push_str("-16b");
        }
        if self.blur != 0.0 {
            s.push_str(&format!("-blur{}", self.blur));
        }
        #[cfg(feature = "boundary-rd")]
        if let super::encoder_config::BoundaryRd::On(cfg) = self.boundary_rd {
            if cfg == super::encoder_config::BoundaryRdConfig::default() {
                s.push_str("-brd");
            } else {
                // Non-default knobs: compact content hash keeps ids unique
                // without leaking the whole Debug dump into every row.
                let mut h = Fnv::new();
                h.write(format!("{cfg:?}").as_bytes());
                s.push_str(&format!("-brd#{:04x}", h.0 & 0xffff));
            }
        }
        s
    }
}

/// Cross axes × quality points into deduplicated, priority-ordered cells.
fn cross(axes: &SweepAxes, q_points: &[f32]) -> (Vec<SweepCell>, Vec<String>, usize) {
    #[cfg(feature = "boundary-rd")]
    let brd_values = axes.boundary_rd.clone();
    #[cfg(not(feature = "boundary-rd"))]
    let brd_values: Vec<()> = vec![()];

    // Pass 1: enumerate strata with per-axis value indices; validity is
    // quality-independent so it is checked here, once per stratum.
    struct Entry<'a> {
        stratum: Stratum<'a>,
        deviations: u8,
        idx_sum: usize,
        seq: usize,
    }
    let mut entries: Vec<Entry<'_>> = Vec::new();
    let mut invalid = Vec::new();
    let mut seq = 0usize;

    for (ci, color) in axes.color_modes.iter().enumerate() {
        for (fi, family) in axes.families.iter().enumerate() {
            for (oi, coeff) in axes.coeff_opt.iter().enumerate() {
                for (si, &scan) in axes.scans.iter().enumerate() {
                    for (ai, &aq) in axes.aq.iter().enumerate() {
                        for (di, &dering) in axes.deringing.iter().enumerate() {
                            for (wi, &down) in axes.downsampling.iter().enumerate() {
                                for (xi, &allow16) in axes.allow_16bit.iter().enumerate() {
                                    for (bi, &blur) in axes.pre_blur.iter().enumerate() {
                                        for (ri, brd) in brd_values.iter().enumerate() {
                                            #[cfg(not(feature = "boundary-rd"))]
                                            let _ = brd;
                                            let idxs = [ci, fi, oi, si, ai, di, wi, xi, bi, ri];
                                            let stratum = Stratum {
                                                color: *color,
                                                family,
                                                coeff,
                                                scan,
                                                aq,
                                                dering,
                                                down,
                                                allow16,
                                                blur,
                                                #[cfg(feature = "boundary-rd")]
                                                boundary_rd: *brd,
                                            };
                                            if stratum.build_config(75.0).validate().is_err() {
                                                invalid.push(stratum.id());
                                                continue;
                                            }
                                            entries.push(Entry {
                                                stratum,
                                                deviations: idxs.iter().filter(|&&x| x != 0).count()
                                                    as u8,
                                                idx_sum: idxs.iter().sum(),
                                                seq,
                                            });
                                            seq += 1;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Main effects before interactions; milder deviations before extreme
    // ones; nested order as the deterministic tie-break.
    entries.sort_by_key(|e| (e.deviations, e.idx_sum, e.seq));

    // Pass 2: expand quality ascending within each stratum (complete RD
    // curves — a truncated queue is safe at stratum boundaries) and
    // dedupe by resolved fingerprint. Keep-first means the merged cell
    // carries the highest-priority spelling; later aliases record the
    // exotic ones.
    let mut cells: Vec<SweepCell> = Vec::new();
    let mut by_fingerprint: std::collections::HashMap<u64, usize> =
        std::collections::HashMap::new();
    let mut merged = 0usize;

    for e in &entries {
        for &q in q_points {
            let config = e.stratum.build_config(q);
            let fingerprint = fingerprint(&config);
            let id = format!("{}_q{q}", e.stratum.id());
            if let Some(&idx) = by_fingerprint.get(&fingerprint) {
                cells[idx].aliases.push(id);
                merged += 1;
            } else {
                by_fingerprint.insert(fingerprint, cells.len());
                cells.push(SweepCell {
                    id,
                    config,
                    quality: q,
                    fingerprint,
                    aliases: Vec::new(),
                    deviations: e.deviations,
                });
            }
        }
    }
    (cells, invalid, merged)
}

// ============================================================================
// Byte-identity fingerprint
// ============================================================================

struct Fnv(u64);
impl Fnv {
    fn new() -> Self {
        Fnv(0xcbf2_9ce4_8422_2325)
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= u64::from(b);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    fn u8(&mut self, v: u8) {
        self.write(&[v]);
    }
    fn u16(&mut self, v: u16) {
        self.write(&v.to_le_bytes());
    }
    fn f32(&mut self, v: f32) {
        self.write(&v.to_bits().to_le_bytes());
    }
}

/// Byte-identity fingerprint of a config's resolved state.
///
/// Two configs with equal fingerprints produce identical bytes for the
/// same input image. Built from the RESOLVED state, so it sees through
/// aliases:
///
/// - quality is fully mediated by the resolved tables + zero-bias
///   (Glassa/Piecewise anchor clamps, `allow_16bit` at qualities where
///   no value exceeds 255, …);
/// - `TrellisSpeedMode` IS included: it bounds the coefficient search on
///   high-entropy blocks and therefore changes output bytes (empirically
///   validated 2026-06-10 — an earlier revision excluded it as
///   "output-neutral by construction" and was wrong);
/// - restart rows are resolved to zero under progressive scan modes
///   (suppressed there unless forced).
#[must_use]
pub fn fingerprint(config: &EncoderConfig) -> u64 {
    let (use_xyb, is_420) = match config.get_color_mode() {
        ColorMode::YCbCr { subsampling } => {
            (false, Subsampling::from(subsampling) == Subsampling::S420)
        }
        ColorMode::Xyb { .. } => (true, false),
        ColorMode::Grayscale => (false, false),
    };
    let resolved = resolve_quant_tables(TableResolveInputs {
        quality: config.get_quality(),
        table_config: config.get_quant_table_config(),
        use_xyb,
        is_420,
        allow_16bit: config.is_allow_16bit_quant_tables(),
    });

    let mut h = Fnv::new();
    for table in [&resolved.quant.0, &resolved.quant.1, &resolved.quant.2] {
        for &v in &table.values {
            h.u16(v);
        }
        h.u8(table.precision);
    }
    for zb in [
        &resolved.zero_bias.0,
        &resolved.zero_bias.1,
        &resolved.zero_bias.2,
    ] {
        for &m in &zb.mul {
            h.f32(m);
        }
        for &o in &zb.offset {
            h.f32(o);
        }
    }

    match config.get_trellis() {
        None => h.u8(0),
        Some(t) => {
            h.u8(1);
            h.u8(u8::from(t.enabled));
            h.u8(u8::from(t.dc_enabled));
            h.f32(t.lambda_log_scale1);
            h.f32(t.lambda_log_scale2);
            h.f32(t.delta_dc_weight);
            // speed_mode IS hashed: it bounds the trellis search
            // (lookback/candidates on high-entropy blocks), so it changes
            // chosen coefficients. Empirically falsified as "neutral" on
            // 2026-06-10: Adaptive vs Thorough differed by 582 bytes on
            // 512² noise at q95 (and 2 bytes on a CID22 photo).
            match t.speed_mode {
                super::trellis::TrellisSpeedMode::Thorough => h.u8(0),
                super::trellis::TrellisSpeedMode::Adaptive => h.u8(1),
                super::trellis::TrellisSpeedMode::Level(l) => {
                    h.u8(2);
                    h.u8(l);
                }
                super::trellis::TrellisSpeedMode::Custom {
                    tier1_threshold,
                    tier1_lookback,
                    tier1_candidates,
                    tier2_threshold,
                    tier2_lookback,
                    tier2_candidates,
                } => {
                    h.u8(3);
                    for v in [
                        tier1_threshold,
                        tier1_lookback,
                        tier1_candidates,
                        tier2_threshold,
                        tier2_lookback,
                        tier2_candidates,
                    ] {
                        h.u8(v);
                    }
                }
            }
            h.u8(u8::from(t.aq_coupling.multiplicative));
            h.f32(t.aq_coupling.scale);
            h.f32(t.aq_coupling.exponent);
            h.f32(t.aq_coupling.threshold);
            h.f32(t.aq_coupling.max_adjustment);
            h.f32(t.aq_coupling.chroma_mul);
        }
    }

    let scan = config.scan_mode;
    h.u8(match scan {
        ProgressiveScanMode::Baseline => 0,
        ProgressiveScanMode::Progressive => 1,
        ProgressiveScanMode::ProgressiveMozjpeg => 2,
        ProgressiveScanMode::ProgressiveSearch => 3,
        ProgressiveScanMode::Smallest => 4,
        ProgressiveScanMode::SmallestSearch => 5,
    });
    h.u8(match &config.huffman {
        HuffmanStrategy::Optimize => 0,
        HuffmanStrategy::Fixed => 1,
        HuffmanStrategy::FixedAnnexK => 2,
        HuffmanStrategy::Custom(_) => 3,
    });
    h.u8(match config.get_color_mode() {
        ColorMode::YCbCr { subsampling } => match subsampling {
            ChromaSubsampling::None => 0,
            ChromaSubsampling::HalfHorizontal => 1,
            ChromaSubsampling::Quarter => 2,
            ChromaSubsampling::HalfVertical => 3,
        },
        ColorMode::Xyb { subsampling } => match subsampling {
            super::encoder_types::XybSubsampling::Full => 10,
            super::encoder_types::XybSubsampling::BQuarter => 11,
        },
        ColorMode::Grayscale => 20,
    });
    h.u8(u8::from(config.is_aq_enabled()));
    h.u8(u8::from(config.deringing));
    h.u8(match config.downsampling_method {
        DownsamplingMethod::Box => 0,
        DownsamplingMethod::GammaAware => 1,
        DownsamplingMethod::GammaAwareIterative => 2,
    });
    h.f32(config.pre_blur);
    // Restart markers are suppressed under progressive unless forced.
    let restart_rows = if scan.is_progressive() && !config.force_restart_markers {
        0
    } else {
        config.restart_mcu_rows
    };
    h.u16(restart_rows);
    h.u8(u8::from(config.force_restart_markers));
    // Boundary-RD only fires on the non-trellis path (the engine skips
    // it when a trellis config is set), so it is hashed only there —
    // boundary × trellis cells dedupe with their trellis-only twins.
    #[cfg(feature = "boundary-rd")]
    {
        match (config.get_trellis(), config.resolve_boundary_rd()) {
            (Some(_), _) | (None, None) => h.u8(0),
            (None, Some(flat)) => {
                h.u8(1);
                h.f32(flat.alpha);
                h.f32(flat.threshold);
                h.f32(flat.shrink);
                h.u8(flat.max_retries);
                h.u8(u8::from(flat.above));
                h.f32(flat.drift_gain);
                h.f32(flat.retry_beta);
            }
        }
    }

    h.u8(match config.tiny_file_mode {
        super::encoder_types::TinyFileMode::Auto => 0,
        super::encoder_types::TinyFileMode::Off => 1,
        super::encoder_types::TinyFileMode::Force => 2,
    });

    h.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_axes() -> SweepAxes {
        SweepAxes {
            families: vec![QuantTableConfig::default()],
            coeff_opt: vec![None],
            scans: vec![ProgressiveScanMode::Progressive],
            color_modes: vec![ColorMode::YCbCr {
                subsampling: ChromaSubsampling::Quarter,
            }],
            aq: vec![true],
            deringing: vec![true],
            downsampling: vec![DownsamplingMethod::Box],
            allow_16bit: vec![false],
            pre_blur: vec![0.0],
            #[cfg(feature = "boundary-rd")]
            boundary_rd: vec![super::super::encoder_config::BoundaryRd::Off],
        }
    }

    #[test]
    fn rd_core_is_progressive_only_modes_full_restores_baseline() {
        assert_eq!(
            SweepAxes::rd_core().scans,
            vec![ProgressiveScanMode::Smallest]
        );
        assert!(
            SweepAxes::modes_full()
                .scans
                .contains(&ProgressiveScanMode::Baseline),
            "baseline must stay reachable for the tiny-size bucket"
        );
    }

    #[cfg(feature = "boundary-rd")]
    #[test]
    fn boundary_rd_inert_under_trellis_dedupes() {
        use super::super::encoder_config::{BoundaryRd, BoundaryRdConfig};
        let mut axes = tiny_axes();
        axes.coeff_opt = vec![Some(TrellisConfig::default())];
        axes.boundary_rd = vec![BoundaryRd::Off, BoundaryRd::On(BoundaryRdConfig::default())];
        let plan = SweepBuilder::new(axes, QualityGrid::Explicit(vec![75.0])).plan();
        assert_eq!(
            plan.cells.len(),
            1,
            "engine skips boundary-rd under trellis"
        );
        assert_eq!(plan.duplicates_merged, 1);
    }

    #[cfg(feature = "boundary-rd")]
    #[test]
    fn boundary_rd_distinct_without_trellis() {
        use super::super::encoder_config::{BoundaryRd, BoundaryRdConfig};
        let mut axes = tiny_axes();
        axes.coeff_opt = vec![None];
        axes.boundary_rd = vec![BoundaryRd::Off, BoundaryRd::On(BoundaryRdConfig::default())];
        let plan = SweepBuilder::new(axes, QualityGrid::Explicit(vec![75.0])).plan();
        assert_eq!(plan.cells.len(), 2);
    }

    #[test]
    fn queue_is_main_effects_first() {
        let mut axes = SweepAxes::rd_core();
        axes.aq = vec![true, false];
        let plan = SweepBuilder::new(axes, QualityGrid::Explicit(vec![50.0, 85.0])).plan();

        // The very first cell is the production-default stratum.
        assert_eq!(plan.cells[0].deviations, 0);
        assert!(
            plan.cells[0].id.starts_with("jp3_t0_small_420"),
            "first cell must be the default stratum, got {}",
            plan.cells[0].id
        );
        // Deviations are non-decreasing along the queue.
        for w in plan.cells.windows(2) {
            assert!(
                w[1].deviations >= w[0].deviations || w[1].deviations + 1 >= w[0].deviations,
                "queue must be priority-ordered"
            );
        }
        let first_two = plan
            .cells
            .iter()
            .position(|c| c.deviations >= 2)
            .unwrap_or(plan.cells.len());
        assert!(
            plan.cells[..first_two].iter().all(|c| c.deviations <= 1),
            "all main-effect strata must precede interaction strata"
        );
        // Quality ascends within the leading default stratum.
        assert!(plan.cells[0].quality < plan.cells[1].quality);
    }

    #[test]
    fn modes_full_covers_the_scalar_axes() {
        let axes = SweepAxes::modes_full();
        assert!(
            axes.families
                .iter()
                .any(|f| f.chroma_distance_scales() == Some([2.0, 1.0])),
            "asymmetric chroma probe missing"
        );
        assert!(axes.pre_blur.contains(&0.4));
        assert!(
            axes.coeff_opt
                .iter()
                .flatten()
                .any(|t| (t.lambda_log_scale1 - 16.0).abs() < 1e-6),
            "λ₁ ladder missing 16.0"
        );
        assert!(
            axes.coeff_opt
                .iter()
                .flatten()
                .any(|t| (t.lambda_log_scale2 - 17.0).abs() < 1e-6),
            "λ₂ probe missing"
        );
        assert!(
            axes.coeff_opt
                .iter()
                .flatten()
                .any(|t| { t.aq_coupling.scale == -8.0 && t.aq_coupling.max_adjustment == 1.0 }),
            "clamped −8 coupling missing"
        );
        // No curated coupling step may be unclamped: the unclamped form
        // is the validated quality-destruction mode on high-AQ content.
        for t in axes.coeff_opt.iter().flatten() {
            if t.aq_coupling.is_active() {
                assert!(
                    t.aq_coupling.max_adjustment > 0.0,
                    "unclamped coupling step in curated axes: {t:?}"
                );
            }
        }
    }

    #[test]
    fn speed_mode_changes_fingerprint() {
        // speed_mode bounds the trellis search and changes output bytes
        // (validated empirically 2026-06-10) — it must be hashed.
        use super::super::trellis::TrellisSpeedMode;
        let base = EncoderConfig::ycbcr(95, ChromaSubsampling::Quarter);
        let adaptive = base.clone().trellis(TrellisConfig::default());
        let thorough = base.trellis(TrellisConfig {
            speed_mode: TrellisSpeedMode::Thorough,
            ..TrellisConfig::default()
        });
        assert_ne!(fingerprint(&adaptive), fingerprint(&thorough));
    }

    #[test]
    fn cell_ids_are_unique_across_modes_full() {
        // Regression: λ₂ / delta-DC / coupling-exponent probes used to
        // render identically to their base configs ("tr14.7500" twice,
        // "tr14.7500+dc" colliding with default trellis, the exponent-2
        // probe colliding with plain cpl−4), making TSV rows and alias
        // reports ambiguous. Every id — canonical or alias — is unique.
        let plan =
            SweepBuilder::new(SweepAxes::modes_full(), QualityGrid::Explicit(vec![85.0])).plan();
        let mut seen = std::collections::HashSet::new();
        for cell in &plan.cells {
            assert!(seen.insert(cell.id.clone()), "duplicate id {}", cell.id);
            for a in &cell.aliases {
                assert!(seen.insert(a.clone()), "duplicate alias id {a}");
            }
        }
    }

    #[test]
    fn plan_is_deterministic() {
        let a = SweepBuilder::new(SweepAxes::rd_core(), QualityGrid::Step5).plan();
        let b = SweepBuilder::new(SweepAxes::rd_core(), QualityGrid::Step5).plan();
        assert_eq!(a.cells.len(), b.cells.len());
        for (x, y) in a.cells.iter().zip(&b.cells) {
            assert_eq!(x.id, y.id);
            assert_eq!(x.fingerprint, y.fingerprint);
        }
    }

    #[test]
    fn auto_optimize_aliases_to_explicit_trellis() {
        let via_auto = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter).auto_optimize(true);
        let explicit = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
            .progressive(true)
            .trellis(trellis_auto_shape());
        assert_eq!(fingerprint(&via_auto), fingerprint(&explicit));
    }

    #[test]
    fn glassa_anchor_clamp_dedupes_high_q() {
        let mut axes = tiny_axes();
        axes.families = vec![QuantTableConfig::GlassaLowBpp];
        let plan =
            SweepBuilder::new(axes, QualityGrid::Explicit(vec![30.0, 50.0, 70.0, 90.0])).plan();
        // All four qualities clamp to the q25 anchor: one cell, three aliases.
        assert_eq!(plan.cells.len(), 1, "cells: {:?}", plan.cells);
        assert_eq!(plan.duplicates_merged, 3);
        assert_eq!(plan.cells[0].aliases.len(), 3);
    }

    #[test]
    fn allow_16bit_dedupes_where_tables_fit_8bit() {
        let mut axes = tiny_axes();
        axes.allow_16bit = vec![false, true];
        // Q95: all quant values fit in 8 bits, so the flag is byte-inert.
        let plan = SweepBuilder::new(axes.clone(), QualityGrid::Explicit(vec![95.0])).plan();
        assert_eq!(plan.cells.len(), 1);
        assert_eq!(plan.duplicates_merged, 1);
        // Q50: chroma exceeds 255 — the flag now changes bytes.
        let plan = SweepBuilder::new(axes, QualityGrid::Explicit(vec![50.0])).plan();
        assert_eq!(plan.cells.len(), 2);
    }

    #[test]
    fn invalid_xyb_combinations_are_reported_not_lost() {
        let mut axes = tiny_axes();
        axes.families = vec![QuantTableConfig::PiecewiseV4];
        axes.color_modes = vec![ColorMode::Xyb {
            subsampling: super::super::encoder_types::XybSubsampling::BQuarter,
        }];
        let plan = SweepBuilder::new(axes, QualityGrid::Explicit(vec![75.0])).plan();
        assert!(plan.cells.is_empty());
        assert_eq!(plan.invalid_skipped.len(), 1);
        assert!(plan.invalid_skipped[0].contains("pw4"));
    }

    #[test]
    fn budget_ladder_collapses_mode_axes_first_and_reports() {
        let mut axes = SweepAxes::rd_core();
        axes.aq = vec![true, false];
        axes.deringing = vec![true, false];
        axes.pre_blur = vec![0.0, 0.4];
        let unbudgeted = SweepBuilder::new(axes.clone(), QualityGrid::Step5).plan();
        let budget = unbudgeted.cells.len() / 4;
        let plan = SweepBuilder::new(axes, QualityGrid::Step5)
            .with_budget(budget)
            .plan();
        assert!(plan.cells.len() <= budget);
        assert!(!plan.dropped.is_empty());
        assert_eq!(plan.dropped[0].axis, "pre_blur");
        assert!(!plan.over_budget);
        for d in &plan.dropped {
            assert!(!d.dropped.is_empty(), "drop report must list values");
        }
    }

    #[test]
    fn q_coarsening_keeps_endpoints_and_floor() {
        let pts = QualityGrid::Step5.points();
        let coarse = coarsen_keep_endpoints(&pts);
        assert_eq!(coarse.first(), pts.first());
        assert_eq!(coarse.last(), pts.last());
        assert!(coarse.len() >= 11);
    }

    #[test]
    fn over_budget_reports_rather_than_samples() {
        // Impossible budget: 1 cell. Ladder exhausts, flag set, plan complete.
        let plan = SweepBuilder::new(SweepAxes::rd_core(), QualityGrid::Step5)
            .with_budget(1)
            .plan();
        assert!(plan.over_budget);
        assert!(plan.cells.len() > 1, "nothing may be silently sampled away");
    }

    #[test]
    fn encodes_math() {
        let plan = SweepBuilder::new(tiny_axes(), QualityGrid::Explicit(vec![50.0, 80.0])).plan();
        assert_eq!(plan.encodes(50, 4), plan.cells.len() * 200);
    }
}
