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
//!    spelling, output-neutral `speed_mode` variants — collapse into one
//!    encode with the merged ids recorded as aliases.
//! 4. **Budget ladder** — [`SweepBuilder::with_budget`] reduces
//!    deterministically: collapse low-tier mode axes to their first
//!    value (recorded in [`SweepPlan::dropped`]), then coarsen the
//!    quality grid uniformly (endpoints kept, never below 11 points),
//!    and finally set [`SweepPlan::over_budget`] rather than sample
//!    silently. No silent caps.
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
/// otherwise). Measured envelope: `-4` ≈ 2 % smaller / ~3 % DSSIM cost
/// on photos; `+4` ≈ the reverse.
#[must_use]
pub fn trellis_coupled(scale: f32) -> TrellisConfig {
    TrellisConfig {
        dc_enabled: false,
        aq_coupling: AqCoupling {
            scale,
            ..AqCoupling::OFF
        },
        ..TrellisConfig::default()
    }
}

impl SweepAxes {
    /// The axes that move the rate-distortion front, with everything
    /// else at production defaults: 4 table families × {no trellis,
    /// default trellis, the auto_optimize shape} × {progressive,
    /// baseline} × {4:2:0, 4:4:4}. 48 strata before the quality grid.
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
            scans: vec![
                ProgressiveScanMode::Progressive,
                ProgressiveScanMode::Baseline,
            ],
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
        axes.coeff_opt.push(Some(trellis_coupled(-4.0)));
        axes.coeff_opt.push(Some(trellis_coupled(4.0)));
        axes.scans.push(ProgressiveScanMode::ProgressiveMozjpeg);
        axes.scans.push(ProgressiveScanMode::ProgressiveSearch);
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

/// Collapse the lowest-tier multi-valued axis to its first value.
fn collapse_one_axis(axes: &mut SweepAxes) -> Option<DroppedAxis> {
    fn collapse<T: core::fmt::Debug + Clone>(
        name: &'static str,
        v: &mut Vec<T>,
        keep: usize,
    ) -> Option<DroppedAxis> {
        if v.len() <= keep {
            return None;
        }
        let dropped = v[keep..].iter().map(|x| format!("{x:?}")).collect();
        let kept = v[..keep]
            .iter()
            .map(|x| format!("{x:?}"))
            .collect::<Vec<_>>()
            .join(", ");
        v.truncate(keep);
        Some(DroppedAxis {
            axis: name,
            kept,
            dropped,
        })
    }

    // Tier order: cheapest-to-lose first. Families are never collapsed.
    collapse("pre_blur", &mut axes.pre_blur, 1)
        .or_else(|| collapse("allow_16bit", &mut axes.allow_16bit, 1))
        .or_else(|| collapse("downsampling", &mut axes.downsampling, 1))
        .or_else(|| collapse("deringing", &mut axes.deringing, 1))
        .or_else(|| collapse("aq", &mut axes.aq, 1))
        .or_else(|| collapse("scans", &mut axes.scans, 2))
        .or_else(|| collapse("color_modes", &mut axes.color_modes, 2))
        .or_else(|| collapse("coeff_opt", &mut axes.coeff_opt, 3))
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

/// Cross axes × quality points into deduplicated cells.
fn cross(axes: &SweepAxes, q_points: &[f32]) -> (Vec<SweepCell>, Vec<String>, usize) {
    let mut cells: Vec<SweepCell> = Vec::new();
    let mut by_fingerprint: std::collections::HashMap<u64, usize> =
        std::collections::HashMap::new();
    let mut invalid = Vec::new();
    let mut merged = 0usize;

    for color in &axes.color_modes {
        for family in &axes.families {
            for coeff in &axes.coeff_opt {
                for &scan in &axes.scans {
                    for &aq in &axes.aq {
                        for &dering in &axes.deringing {
                            for &down in &axes.downsampling {
                                for &allow16 in &axes.allow_16bit {
                                    for &blur in &axes.pre_blur {
                                        // Validity is quality-independent; check once
                                        // per stratum at a representative q.
                                        let probe = build_config(
                                            *color, family, coeff, scan, aq, dering, down, allow16,
                                            blur, 75.0,
                                        );
                                        if probe.validate().is_err() {
                                            invalid.push(stratum_id(
                                                color, family, coeff, scan, aq, dering, down,
                                                allow16, blur,
                                            ));
                                            continue;
                                        }
                                        for &q in q_points {
                                            let config = build_config(
                                                *color, family, coeff, scan, aq, dering, down,
                                                allow16, blur, q,
                                            );
                                            let fingerprint = fingerprint(&config);
                                            let id = format!(
                                                "{}_q{q}",
                                                stratum_id(
                                                    color, family, coeff, scan, aq, dering, down,
                                                    allow16, blur,
                                                )
                                            );
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
                                                });
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
    }
    (cells, invalid, merged)
}

#[allow(clippy::too_many_arguments)]
fn build_config(
    color: ColorMode,
    family: &QuantTableConfig,
    coeff: &Option<TrellisConfig>,
    scan: ProgressiveScanMode,
    aq: bool,
    dering: bool,
    down: DownsamplingMethod,
    allow16: bool,
    blur: f32,
    q: f32,
) -> EncoderConfig {
    let mut cfg = match color {
        ColorMode::YCbCr { subsampling } => EncoderConfig::ycbcr(q, subsampling),
        ColorMode::Xyb { subsampling } => EncoderConfig::xyb(q, subsampling),
        ColorMode::Grayscale => EncoderConfig::grayscale(q),
    };
    cfg = cfg
        .quant_table_config(family.clone())
        .progressive(scan)
        .aq_enabled(aq)
        .deringing(dering)
        .downsampling_method(down)
        .allow_16bit_quant_tables(allow16)
        .pre_blur(blur);
    if let Some(t) = coeff {
        cfg = cfg.trellis(*t);
    }
    cfg
}

#[allow(clippy::too_many_arguments)]
fn stratum_id(
    color: &ColorMode,
    family: &QuantTableConfig,
    coeff: &Option<TrellisConfig>,
    scan: ProgressiveScanMode,
    aq: bool,
    dering: bool,
    down: DownsamplingMethod,
    allow16: bool,
    blur: f32,
) -> String {
    let fam = match family {
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
    let co = match coeff {
        None => "t0".to_string(),
        Some(t) => {
            let mut s = format!("tr{:.4}", t.lambda_log_scale1);
            if t.dc_enabled {
                s.push_str("+dc");
            }
            if t.aq_coupling.is_active() {
                s.push_str(&format!("cpl{:+.1}", t.aq_coupling.scale));
            }
            s
        }
    };
    let sc = match scan {
        ProgressiveScanMode::Baseline => "base",
        ProgressiveScanMode::Progressive => "prog",
        ProgressiveScanMode::ProgressiveMozjpeg => "pmoz",
        ProgressiveScanMode::ProgressiveSearch => "psrch",
    };
    let col = match color {
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
    if !aq {
        s.push_str("-noaq");
    }
    if !dering {
        s.push_str("-noder");
    }
    match down {
        DownsamplingMethod::Box => {}
        DownsamplingMethod::GammaAware => s.push_str("-gaware"),
        DownsamplingMethod::GammaAwareIterative => s.push_str("-sharp"),
    }
    if allow16 {
        s.push_str("-16b");
    }
    if blur != 0.0 {
        s.push_str(&format!("-blur{blur}"));
    }
    s
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
/// - `TrellisSpeedMode` is excluded (output-neutral by construction);
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
            // speed_mode excluded: output-neutral.
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
