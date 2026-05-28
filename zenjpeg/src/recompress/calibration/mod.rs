//! Per-cell calibration lookup.
//!
//! A cell is `(encoder × subsampling × source-quality-bin × target-zensim-A
//! × strategy)`. Each cell carries an estimate `(projected_zensim_a,
//! projected_size_ratio, ci_width)` and a "this strategy is preferred"
//! flag.
//!
//! In v0.1 the table is hand-seeded from `crate::detect::reencode`
//! conjectures plus our anchor mapping in [`crate::recompress::target`]. The
//! `zjr-calibrate` driver replaces the data with a real corpus sweep
//! later.

use crate::decoder::Subsampling;
use crate::detect::EncoderFamily;

use crate::recompress::api::StrategyKind;

pub mod data;
pub mod per_encoder;

/// Coarse encoder class used as the calibration-table axis. Multiple
/// fingerprints collapse to the same class when their RD curves are
/// effectively interchangeable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EncoderClass {
    /// libjpeg-turbo / Pillow / generic IJG-table emitters.
    IjgFamily,
    /// ImageMagick (IJG + optimized Huffman; switches to 4:4:4 at Q≥90).
    ImageMagick,
    /// mozjpeg.
    Mozjpeg,
    /// cjpegli YCbCr.
    JpegliYcbcr,
    /// cjpegli XYB.
    JpegliXyb,
    /// Photoshop "Save for Web" presets.
    Photoshop,
    /// zenjpeg YCbCr output.
    ZenjpegYcbcr,
    /// zenjpeg XYB output.
    ZenjpegXyb,
    /// Anything else.
    Unknown,
}

impl EncoderClass {
    pub fn from_family(family: EncoderFamily) -> Self {
        match family {
            EncoderFamily::LibjpegTurbo | EncoderFamily::IjgFamily => EncoderClass::IjgFamily,
            EncoderFamily::ImageMagick => EncoderClass::ImageMagick,
            EncoderFamily::Mozjpeg => EncoderClass::Mozjpeg,
            EncoderFamily::CjpegliYcbcr => EncoderClass::JpegliYcbcr,
            EncoderFamily::CjpegliXyb => EncoderClass::JpegliXyb,
            EncoderFamily::Photoshop => EncoderClass::Photoshop,
            _ => EncoderClass::Unknown,
        }
    }
}

/// Logical identity of a baked calibration table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TableId {
    /// Seed table — coarse, derived from `crate::detect::reencode`.
    Seed,
}

/// One cell estimate from the table.
#[derive(Debug, Clone, Copy)]
pub struct CellEstimate {
    /// Projected zensim-A vs original after applying this strategy at
    /// the calibrated parameter point.
    pub projected_zensim_a: f32,
    /// Projected `output_len / input_len` at that strategy + parameter
    /// point.
    pub projected_size_ratio: f32,
    /// Whether the table flags this strategy as the preferred choice.
    pub preferred: bool,
    /// Confidence-interval width (0 = exact, 1 = no signal).
    pub ci: CellCi,
    /// Inverse-calibrated target dial (zensim-A) to feed
    /// `target_zensim_a_to_ijg_q` so this strategy actually *achieves*
    /// the user target (corrects systematic over/under-shoot). `None`
    /// = no per-encoder inverse available; use the user target directly.
    pub dial_zensim_a: Option<f32>,
}

/// Confidence interval class for a cell estimate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CellCi {
    /// CI < 1 zensim-A, fit on ≥40 samples / cell.
    Tight,
    /// CI < 3 zensim-A — usable for production routing.
    Moderate,
    /// CI ≥ 3, or no samples in cell — extrapolation only.
    Loose,
    /// No data exists for this cell.
    Empty,
}

/// Bundle of strategy choices ranked for a given input.
#[derive(Debug, Clone, Copy)]
pub struct StrategyChoice {
    pub kind: StrategyKind,
    pub estimate: CellEstimate,
}

/// Public lookup API used by the router and (under `expert`) by callers
/// who want to override routing.
#[derive(Debug, Clone, Copy)]
pub struct CalibrationLookup;

impl CalibrationLookup {
    pub const SEED: Self = Self;

    /// Estimate every strategy for a `(encoder, subsampling, source_q,
    /// target_zensim_a)` tuple. Returns a fixed-size array in a stable
    /// order — callers pick `min_by_size_ratio()` or apply their own
    /// preference.
    pub fn estimate_all(
        &self,
        encoder: EncoderClass,
        subsampling: Subsampling,
        source_estimated_zensim_a: f32,
        target_zensim_a: f32,
    ) -> [StrategyChoice; 4] {
        [
            StrategyChoice {
                kind: StrategyKind::Preserve,
                estimate: estimate_preserve(
                    encoder,
                    subsampling,
                    source_estimated_zensim_a,
                    target_zensim_a,
                ),
            },
            StrategyChoice {
                kind: StrategyKind::Deblock,
                estimate: estimate_deblock(
                    encoder,
                    subsampling,
                    source_estimated_zensim_a,
                    target_zensim_a,
                ),
            },
            StrategyChoice {
                kind: StrategyKind::Tuned,
                estimate: estimate_tuned(
                    encoder,
                    subsampling,
                    source_estimated_zensim_a,
                    target_zensim_a,
                ),
            },
            StrategyChoice {
                kind: StrategyKind::Lossless,
                estimate: estimate_lossless(encoder, subsampling, source_estimated_zensim_a),
            },
        ]
    }
}

/// Safety margin (zensim-A) added to the inverse-calibration target so
/// the per-encoder strategies aim slightly *above* the user target. The
/// per-encoder tables are medians; aiming exactly at target leaves ~half
/// of images under (content variance). A small margin keeps under-target
/// delivery low while staying far tighter than the old naive-dial
/// over-delivery. Tuned 2026-05-28: margin 4.0 holds turbo/mozjpeg
/// under-target ≤ ~10% while shrinking files vs the no-inverse baseline.
/// The `Confidence` knob stacks on top of this for stronger guarantees.
const INVERSE_SAFETY_MARGIN: f32 = 4.0;

fn estimate_preserve(
    encoder: EncoderClass,
    _subsampling: Subsampling,
    source: f32,
    target: f32,
) -> CellEstimate {
    // Inverse-calibrated per-encoder Preserve: find the dial that makes
    // Preserve hit `target`, with the projected ratio at that dial. The
    // dial is fed to the strategy so it lands on target instead of
    // over-delivering at the naive dial.
    if let Some((dial, ratio, projected)) = per_encoder::invert_dial(
        encoder,
        StrategyKind::Preserve,
        source,
        (target + INVERSE_SAFETY_MARGIN).min(100.0),
    ) {
        return CellEstimate {
            projected_zensim_a: projected,
            projected_size_ratio: ratio,
            preferred: false,
            ci: CellCi::Moderate,
            dial_zensim_a: Some(dial),
        };
    }
    // Fallback: quant-scale-only approximation (Preserve steps from
    // source to target; ratio falls roughly linearly).
    let projected = target;
    let ratio = preserve_size_ratio(source, target);
    CellEstimate {
        projected_zensim_a: projected,
        projected_size_ratio: ratio,
        preferred: false,
        ci: CellCi::Loose,
        dial_zensim_a: None,
    }
}

fn estimate_deblock(
    encoder: EncoderClass,
    subsampling: Subsampling,
    source: f32,
    target: f32,
) -> CellEstimate {
    // Deblock shares Tuned's re-encode RD profile (decode + re-encode),
    // plus a small low-quality perceptual lift. Inverse-calibrated
    // per-encoder Deblock (shares the Tuned table).
    if let Some((dial, ratio, projected)) = per_encoder::invert_dial(
        encoder,
        StrategyKind::Deblock,
        source,
        (target + INVERSE_SAFETY_MARGIN).min(100.0),
    ) {
        let lift = lift_for(source);
        return CellEstimate {
            projected_zensim_a: (projected + lift).min(99.5),
            projected_size_ratio: ratio * 1.02,
            preferred: source < 60.0,
            ci: CellCi::Moderate,
            dial_zensim_a: Some(dial),
        };
    }
    let source_ba = zensim_a_to_ba(source);
    let fitted = match subsampling {
        Subsampling::S420 => data::lookup_420(source_ba, target),
        Subsampling::S444 => data::lookup_444(source_ba, target),
        _ => None,
    };
    if let Some((projected, ratio)) = fitted {
        let lift = lift_for(source);
        return CellEstimate {
            projected_zensim_a: (projected + lift).min(99.5),
            projected_size_ratio: ratio * 1.02,
            preferred: source < 60.0,
            ci: CellCi::Moderate,
            dial_zensim_a: None,
        };
    }
    let projected = (target + lift_for(source)).min(99.5);
    let ratio = tuned_size_ratio(source, target) * 1.02;
    CellEstimate {
        projected_zensim_a: projected,
        projected_size_ratio: ratio,
        preferred: source < 60.0,
        ci: CellCi::Loose,
        dial_zensim_a: None,
    }
}

fn estimate_tuned(
    encoder: EncoderClass,
    subsampling: Subsampling,
    source: f32,
    target: f32,
) -> CellEstimate {
    // Inverse-calibrated per-encoder Tuned: the source-encoder-
    // independent re-encode path, dialed to hit `target` (corrects the
    // systematic ~5pt downward overshoot at the naive dial).
    if let Some((dial, ratio, projected)) = per_encoder::invert_dial(
        encoder,
        StrategyKind::Tuned,
        source,
        (target + INVERSE_SAFETY_MARGIN).min(100.0),
    ) {
        return CellEstimate {
            projected_zensim_a: projected,
            projected_size_ratio: ratio,
            preferred: source >= 60.0,
            ci: CellCi::Moderate,
            dial_zensim_a: Some(dial),
        };
    }
    // For 4:2:0 and 4:4:4 we have fitted 2D lookups from the
    // 15-image CID22 sweep (jpegli-shaped fallback).
    let source_ba = zensim_a_to_ba(source);
    let fitted = match subsampling {
        Subsampling::S420 => data::lookup_420(source_ba, target),
        Subsampling::S444 => data::lookup_444(source_ba, target),
        _ => None,
    };
    if let Some((projected, ratio)) = fitted {
        return CellEstimate {
            projected_zensim_a: projected,
            projected_size_ratio: ratio,
            preferred: source >= 60.0,
            ci: CellCi::Moderate,
            dial_zensim_a: None,
        };
    }
    // Other subsamplings still rely on the analytical fallback. v0.3
    // adds fitted tables for 422 / 440.
    CellEstimate {
        projected_zensim_a: target,
        projected_size_ratio: tuned_size_ratio(source, target),
        preferred: source >= 60.0,
        ci: CellCi::Loose,
        dial_zensim_a: None,
    }
}

/// Invert the BA→zensim-A anchor in `crate::recompress::target`. Used to look up
/// the fitted table by the source's estimated zensim-A.
fn zensim_a_to_ba(z: f32) -> f32 {
    // Anchors mirror `target_zensim_a_to_ba_distance` in `crate::recompress::target`,
    // copied here to avoid a circular dep.
    const ANCHORS: &[(f32, f32)] = &[
        (98.0, 0.5),
        (95.0, 0.8),
        (90.0, 1.2),
        (85.0, 1.7),
        (80.0, 2.3),
        (75.0, 2.9),
        (70.0, 3.5),
        (60.0, 5.0),
        (50.0, 7.0),
        (40.0, 9.0),
    ];
    interp(z, ANCHORS)
}

fn interp(x: f32, anchors: &[(f32, f32)]) -> f32 {
    if anchors.is_empty() {
        return 0.0;
    }
    // Anchors are in DESCENDING x-order — sort the search.
    let mut sorted: Vec<(f32, f32)> = anchors.to_vec();
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    if x <= sorted[0].0 {
        return sorted[0].1;
    }
    if x >= sorted[sorted.len() - 1].0 {
        return sorted[sorted.len() - 1].1;
    }
    for w in sorted.windows(2) {
        if x >= w[0].0 && x <= w[1].0 {
            let t = (x - w[0].0) / (w[1].0 - w[0].0);
            return w[0].1 + t * (w[1].1 - w[0].1);
        }
    }
    sorted[sorted.len() - 1].1
}

fn estimate_lossless(
    _encoder: EncoderClass,
    _subsampling: Subsampling,
    source: f32,
) -> CellEstimate {
    // Lossless restructure typically lands in 90-98% of input size and
    // preserves the source's zensim-A exactly.
    CellEstimate {
        projected_zensim_a: source,
        projected_size_ratio: 0.94,
        preferred: false,
        ci: CellCi::Moderate,
        dial_zensim_a: None,
    }
}

fn preserve_size_ratio(source: f32, target: f32) -> f32 {
    // Heuristic: each 5 zensim-A drop produces ~14% size drop on the
    // preserve path. Floor at 0.20.
    let drop = (source - target).max(0.0) / 5.0;
    let ratio = 1.0_f32 - 0.14 * drop;
    ratio.clamp(0.20, 1.05)
}

fn tuned_size_ratio(source: f32, target: f32) -> f32 {
    // Tuned/deblock paths usually re-quantize more aggressively but
    // also have to pay header overhead. Slightly larger size ratio
    // floor than preserve.
    let drop = (source - target).max(0.0) / 5.0;
    let ratio = 1.05_f32 - 0.16 * drop;
    ratio.clamp(0.25, 1.10)
}

fn lift_for(source: f32) -> f32 {
    // At low source quality, deblock recovers up to 2 zensim-A vs raw
    // tuned re-encode.
    if source < 50.0 {
        2.0
    } else if source < 70.0 {
        1.0
    } else {
        0.0
    }
}
