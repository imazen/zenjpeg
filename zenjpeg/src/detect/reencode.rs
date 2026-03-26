//! Re-encoding quality recommendations based on empirical calibration.
//!
//! Maps source encoder family + quality level + tolerance → recommended zenjpeg settings
//! for re-encoding with configurable quality loss.
//!
//! Dense 2D calibration grid: 9 tolerance levels × 10 source qualities × 3 encoder families.
//! Bilinear interpolation between calibration points for arbitrary (source_q, tolerance) pairs.
//!
//! Calibration data from 10-image median sweep across libjpeg-turbo, mozjpeg, and cjpegli
//! at Q10-Q90, measuring butteraugli delta and file size ratio.

use super::JpegProbe;
use super::fingerprint::EncoderFamily;
use super::quality::QualityScale;
use crate::encode::encoder_types::{ChromaSubsampling, Quality};

/// Default butteraugli tolerance: barely perceptible degradation.
const DEFAULT_BA_TOLERANCE: f32 = 0.3;

/// Minimum supported tolerance.
const MIN_TOLERANCE: f32 = 0.1;
/// Maximum supported tolerance.
const MAX_TOLERANCE: f32 = 2.0;

/// Recommended settings for re-encoding a JPEG with zenjpeg.
#[derive(Debug, Clone)]
pub struct ReencodeSettings {
    /// Recommended zenjpeg quality.
    pub quality: Quality,
    /// Recommended chroma subsampling (matches source).
    pub subsampling: ChromaSubsampling,
    /// Highest quality that still produces a smaller file than the source.
    ///
    /// `None` means no quality level can guarantee a smaller file — the source
    /// encoder is too efficient at this quality level to beat without visible
    /// quality loss. This is common for low-quality mozjpeg sources (Q10-Q30)
    /// and very low-quality turbo/cjpegli sources (Q10).
    ///
    /// When present, capping quality at this value should produce output
    /// ≤ source size for most images. Calibrated across 10 test images
    /// (conservative: all images must shrink, not just median).
    pub shrink_cap: Option<Quality>,
}

/// Errors from re-encoding quality estimation.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ReencodeError {
    /// Requested tolerance is tighter than achievable for this source.
    ///
    /// Even at the highest quality (Q97), the re-encoding introduces more
    /// degradation than the requested tolerance allows. This typically happens
    /// with high-quality sources (Q85+) at very tight tolerances (<0.2).
    ///
    /// Contains the minimum achievable BA delta and the best-effort settings.
    ToleranceTooTight {
        /// Minimum BA delta achievable at Q97 for this source.
        min_achievable: f32,
        /// Best-effort settings (Q97, matching subsampling).
        best_effort: ReencodeSettings,
    },

    /// Tolerance is zero or negative, which is meaningless for re-encoding.
    InvalidTolerance,
}

impl core::fmt::Display for ReencodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ToleranceTooTight { min_achievable, .. } => {
                write!(
                    f,
                    "tolerance too tight: minimum achievable BA delta is {min_achievable:.2}"
                )
            }
            Self::InvalidTolerance => write!(f, "tolerance must be positive"),
        }
    }
}

impl std::error::Error for ReencodeError {}

/// Estimated source butteraugli distance from detected quality.
///
/// For cjpegli, quality.value IS the BA distance. For IJG/mozjpeg, converts
/// from IJG quality to approximate BA using calibration medians (10 gb82 images).
pub(crate) fn estimated_source_ba(probe: &JpegProbe) -> f32 {
    if probe.quality.scale == QualityScale::ButteraugliDistance {
        return probe.quality.value;
    }
    // IJG Q → approximate BA (median across 10 gb82 images)
    let table: &[(f32, f32)] = match probe.encoder {
        EncoderFamily::Mozjpeg => &[
            (10.0, 11.0),
            (20.0, 7.2),
            (30.0, 5.2),
            (40.0, 4.1),
            (50.0, 3.7),
            (65.0, 3.1),
            (75.0, 3.0),
            (80.0, 2.5),
            (85.0, 2.1),
            (90.0, 2.0),
        ],
        _ => &[
            // IJG/turbo/ImageMagick/Unknown
            (10.0, 9.5),
            (20.0, 5.5),
            (30.0, 4.2),
            (40.0, 3.7),
            (50.0, 3.2),
            (65.0, 3.4),
            (75.0, 2.9),
            (80.0, 2.3),
            (85.0, 2.1),
            (90.0, 1.7),
        ],
    };
    interpolate_1d_ascending(table, probe.quality.value)
}

/// Recommended zenjpeg quality for re-encoding at the default tolerance (≤0.3 BA delta).
pub(crate) fn recommended_q(probe: &JpegProbe) -> f32 {
    // Default tolerance always has valid data for all calibrated sources.
    recommended_q_with_tolerance(probe, DEFAULT_BA_TOLERANCE)
}

/// Recommended zenjpeg quality for re-encoding with a custom BA delta tolerance.
///
/// Performs bilinear interpolation across the 2D calibration grid
/// (tolerance × source quality) for the detected encoder family.
pub(crate) fn recommended_q_with_tolerance(probe: &JpegProbe, ba_tolerance: f32) -> f32 {
    let tol = ba_tolerance.clamp(MIN_TOLERANCE, MAX_TOLERANCE);
    let grid = grid_for_encoder(&probe.encoder, &probe.quality.scale);
    bilinear_lookup(grid, probe.quality.value, tol, &probe.quality.scale)
}

/// Minimum achievable BA delta for this source (at Q97).
fn min_achievable_delta(probe: &JpegProbe) -> f32 {
    let table = min_delta_for_encoder(&probe.encoder, &probe.quality.scale);
    interpolate_1d(table, probe.quality.value, &probe.quality.scale)
}

/// Maximum useful quality when downscaling before re-encoding.
pub(crate) fn quality_ceiling(downscale_ratio: f32) -> f32 {
    // Calibration data: at all tested ratios (1.5x-4x), going from Q90→Q95
    // costs ~40% more bytes for <0.3 BA improvement.
    if downscale_ratio < 1.0 {
        97.0
    } else if downscale_ratio < 1.25 {
        93.0
    } else {
        90.0
    }
}

// ============================================================================
// Calibration grids: tolerance × source quality × encoder family
// ============================================================================
//
// Two grid types:
// 1. Proportional factor grid (PRIMARY): [factor_idx][source_q_idx] → zen Q
//    Uses per-image adaptive thresholds: ba_delta ≤ src_ba × factor.
//    Cell = lowest zen_q where ≥80% of images pass.
//
// 2. Absolute tolerance grid (LEGACY): [tolerance_idx][source_q_idx] → zen Q
//    Uses trimmed mean (top 20% dropped) of ba_delta ≤ tolerance.
//
// Calibrated on 10 gb82 images with auto_optimize.
// Trimmed mean drops top 20% to handle known 4:2:0 encoder bug (#6 in CLAUDE.md)
// where 3/10 images produce catastrophic BA deltas at specific quality levels.

/// Proportional tolerance factors in the calibration grid.
const PROP_FACTORS: &[f32] = &[0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 0.50];

/// Absolute tolerance levels in the calibration grid (legacy, for explicit --tolerance).
const TOLERANCES: &[f32] = &[0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0];

// --- libjpeg-turbo / IJG family ---
const IJG_SRC_QS: &[f32] = &[10.0, 20.0, 30.0, 40.0, 50.0, 65.0, 75.0, 80.0, 85.0, 90.0];

// Proportional factor grid (80% images pass per-image threshold)
const IJG_PROP_GRID: &[&[f32]] = &[
    //                        Q10   Q20   Q30   Q40   Q50   Q65   Q75   Q80   Q85   Q90
    /* f=0.05 */
    &[35.0, 55.0, 80.0, 95.0, 95.0, 97.0, 97.0, 97.0, 97.0, 97.0],
    /* f=0.08 */ &[25.0, 25.0, 70.0, 88.0, 88.0, 93.0, 97.0, 97.0, 97.0, 97.0],
    /* f=0.10 */ &[25.0, 25.0, 50.0, 55.0, 70.0, 80.0, 90.0, 97.0, 97.0, 97.0],
    /* f=0.12 */ &[20.0, 25.0, 45.0, 55.0, 70.0, 80.0, 88.0, 97.0, 97.0, 97.0],
    /* f=0.15 */ &[20.0, 25.0, 35.0, 55.0, 70.0, 80.0, 88.0, 97.0, 97.0, 97.0],
    /* f=0.20 */ &[20.0, 25.0, 35.0, 50.0, 65.0, 70.0, 88.0, 90.0, 97.0, 97.0],
    /* f=0.30 */ &[20.0, 20.0, 25.0, 40.0, 50.0, 55.0, 75.0, 75.0, 90.0, 97.0],
    /* f=0.50 */ &[20.0, 20.0, 20.0, 20.0, 30.0, 40.0, 50.0, 55.0, 70.0, 80.0],
];

// Absolute tolerance grid (trimmed mean, top 20% dropped)
const IJG_GRID: &[&[f32]] = &[
    //                        Q10   Q20   Q30   Q40   Q50   Q65   Q75   Q80   Q85   Q90
    /* tol=0.1 */
    &[20.0, 55.0, 55.0, 60.0, 70.0, 93.0, 95.0, 97.0, 97.0, 97.0],
    /* tol=0.2 */ &[20.0, 55.0, 55.0, 55.0, 70.0, 70.0, 88.0, 97.0, 97.0, 97.0],
    /* tol=0.3 */ &[20.0, 25.0, 35.0, 50.0, 60.0, 70.0, 85.0, 85.0, 88.0, 97.0],
    /* tol=0.4 */ &[20.0, 25.0, 35.0, 50.0, 60.0, 65.0, 70.0, 80.0, 85.0, 97.0],
    /* tol=0.5 */ &[20.0, 25.0, 30.0, 50.0, 55.0, 65.0, 70.0, 75.0, 80.0, 95.0],
    /* tol=0.7 */ &[20.0, 25.0, 30.0, 40.0, 50.0, 50.0, 60.0, 65.0, 75.0, 88.0],
    /* tol=1.0 */ &[20.0, 20.0, 25.0, 35.0, 35.0, 40.0, 50.0, 55.0, 65.0, 80.0],
    /* tol=1.5 */ &[20.0, 20.0, 20.0, 20.0, 20.0, 25.0, 30.0, 50.0, 50.0, 60.0],
    /* tol=2.0 */ &[20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 30.0, 30.0, 50.0],
];

/// Min achievable BA delta per IJG source quality (at Q97, P25 across 10 images).
const IJG_MIN_DELTA: &[(f32, f32)] = &[
    (10.0, 0.00),
    (20.0, 0.00),
    (30.0, 0.00),
    (40.0, 0.00),
    (50.0, 0.00),
    (65.0, 0.00),
    (75.0, 0.01),
    (80.0, 0.00),
    (85.0, 0.00),
    (90.0, 0.03),
];

// --- mozjpeg ---
const MOZ_SRC_QS: &[f32] = &[10.0, 20.0, 30.0, 40.0, 50.0, 65.0, 75.0, 80.0, 85.0, 90.0];

// Proportional factor grid (80% images pass per-image threshold)
const MOZ_PROP_GRID: &[&[f32]] = &[
    //                        Q10   Q20   Q30   Q40   Q50   Q65   Q75   Q80   Q85   Q90
    /* f=0.05 */
    &[20.0, 50.0, 85.0, 85.0, 97.0, 97.0, 97.0, 97.0, 97.0, 97.0],
    /* f=0.08 */ &[20.0, 25.0, 60.0, 85.0, 85.0, 95.0, 97.0, 97.0, 97.0, 97.0],
    /* f=0.10 */ &[20.0, 25.0, 60.0, 80.0, 80.0, 85.0, 97.0, 97.0, 97.0, 97.0],
    /* f=0.12 */ &[20.0, 25.0, 30.0, 40.0, 55.0, 75.0, 97.0, 97.0, 97.0, 97.0],
    /* f=0.15 */ &[20.0, 25.0, 25.0, 40.0, 55.0, 65.0, 90.0, 97.0, 97.0, 97.0],
    /* f=0.20 */ &[20.0, 20.0, 25.0, 30.0, 40.0, 55.0, 85.0, 85.0, 97.0, 97.0],
    /* f=0.30 */ &[20.0, 20.0, 20.0, 25.0, 30.0, 50.0, 70.0, 75.0, 88.0, 97.0],
    /* f=0.50 */ &[20.0, 20.0, 20.0, 20.0, 25.0, 35.0, 45.0, 55.0, 75.0, 85.0],
];

// Absolute tolerance grid (trimmed mean, top 20% dropped)
const MOZ_GRID: &[&[f32]] = &[
    //                        Q10   Q20   Q30   Q40   Q50   Q65   Q75   Q80   Q85   Q90
    /* tol=0.1 */
    &[25.0, 25.0, 90.0, 90.0, 90.0, 93.0, 97.0, 97.0, 97.0, 97.0],
    /* tol=0.2 */ &[20.0, 25.0, 35.0, 75.0, 75.0, 75.0, 85.0, 95.0, 97.0, 97.0],
    /* tol=0.3 */ &[20.0, 20.0, 35.0, 45.0, 55.0, 65.0, 85.0, 85.0, 90.0, 95.0],
    /* tol=0.4 */ &[20.0, 20.0, 30.0, 35.0, 50.0, 65.0, 75.0, 80.0, 85.0, 93.0],
    /* tol=0.5 */ &[20.0, 20.0, 25.0, 35.0, 35.0, 55.0, 70.0, 80.0, 80.0, 88.0],
    /* tol=0.7 */ &[20.0, 20.0, 20.0, 25.0, 35.0, 55.0, 65.0, 70.0, 80.0, 85.0],
    /* tol=1.0 */ &[20.0, 20.0, 20.0, 25.0, 30.0, 45.0, 55.0, 60.0, 75.0, 75.0],
    /* tol=1.5 */ &[20.0, 20.0, 20.0, 25.0, 25.0, 30.0, 40.0, 40.0, 50.0, 55.0],
    /* tol=2.0 */ &[20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 25.0, 30.0, 30.0, 40.0],
];

/// Min achievable BA delta per mozjpeg source quality (at Q97, P25 across 10 images).
const MOZ_MIN_DELTA: &[(f32, f32)] = &[
    (10.0, 0.00),
    (20.0, 0.00),
    (30.0, 0.00),
    (40.0, 0.00),
    (50.0, 0.00),
    (65.0, 0.02),
    (75.0, 0.03),
    (80.0, 0.06),
    (85.0, 0.03),
    (90.0, 0.00),
];

// --- cjpegli / zenjpeg ---
// Source quality points: butteraugli distance (DESCENDING = higher quality).
const JPEGLI_SRC_QS: &[f32] = &[5.8, 4.5, 3.8, 3.5, 3.1, 2.7, 2.3, 2.0, 1.7, 1.3];

// Proportional factor grid (80% images pass per-image threshold)
const JPEGLI_PROP_GRID: &[&[f32]] = &[
    //                      d=5.8 d=4.5 d=3.8 d=3.5 d=3.1 d=2.7 d=2.3 d=2.0 d=1.7 d=1.3
    /* f=0.05 */
    &[35.0, 55.0, 55.0, 55.0, 55.0, 65.0, 75.0, 95.0, 95.0, 95.0],
    /* f=0.08 */ &[30.0, 30.0, 30.0, 40.0, 50.0, 65.0, 75.0, 80.0, 85.0, 90.0],
    /* f=0.10 */ &[25.0, 25.0, 30.0, 40.0, 50.0, 65.0, 75.0, 80.0, 85.0, 90.0],
    /* f=0.12 */ &[25.0, 25.0, 30.0, 40.0, 50.0, 65.0, 75.0, 80.0, 85.0, 90.0],
    /* f=0.15 */ &[20.0, 20.0, 30.0, 40.0, 50.0, 65.0, 75.0, 80.0, 85.0, 90.0],
    /* f=0.20 */ &[20.0, 20.0, 30.0, 40.0, 45.0, 60.0, 75.0, 80.0, 85.0, 90.0],
    /* f=0.30 */ &[20.0, 20.0, 25.0, 30.0, 40.0, 50.0, 70.0, 75.0, 85.0, 90.0],
    /* f=0.50 */ &[20.0, 20.0, 20.0, 25.0, 25.0, 30.0, 55.0, 60.0, 75.0, 85.0],
];

// Absolute tolerance grid (trimmed mean, top 20% dropped)
const JPEGLI_GRID: &[&[f32]] = &[
    //                      d=5.8 d=4.5 d=3.8 d=3.5 d=3.1 d=2.7 d=2.3 d=2.0 d=1.7 d=1.3
    /* tol=0.1 */
    &[35.0, 35.0, 35.0, 40.0, 50.0, 65.0, 75.0, 80.0, 85.0, 90.0],
    /* tol=0.2 */ &[25.0, 25.0, 30.0, 40.0, 50.0, 65.0, 75.0, 80.0, 85.0, 90.0],
    /* tol=0.3 */ &[25.0, 25.0, 30.0, 40.0, 45.0, 60.0, 75.0, 80.0, 85.0, 88.0],
    /* tol=0.4 */ &[25.0, 25.0, 30.0, 40.0, 45.0, 60.0, 70.0, 80.0, 80.0, 85.0],
    /* tol=0.5 */ &[20.0, 20.0, 30.0, 35.0, 45.0, 55.0, 70.0, 75.0, 80.0, 85.0],
    /* tol=0.7 */ &[20.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 65.0, 75.0, 85.0],
    /* tol=1.0 */ &[20.0, 20.0, 25.0, 25.0, 30.0, 40.0, 55.0, 60.0, 60.0, 75.0],
    /* tol=1.5 */ &[20.0, 20.0, 20.0, 20.0, 25.0, 25.0, 40.0, 40.0, 50.0, 60.0],
    /* tol=2.0 */ &[20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 25.0, 30.0, 30.0, 50.0],
];

/// Min achievable BA delta per jpegli source quality (at Q97, P25 across 10 images).
const JPEGLI_MIN_DELTA: &[(f32, f32)] = &[
    (5.8, 0.00),
    (4.5, 0.00),
    (3.8, 0.00),
    (3.5, 0.00),
    (3.1, 0.00),
    (2.7, 0.00),
    (2.3, 0.00),
    (2.0, 0.00),
    (1.7, 0.01),
    (1.3, 0.00),
];

// ============================================================================
// Shrink cap tables: highest Q where trimmed mean size_ratio < 1.0
// ============================================================================

/// Highest zen Q producing smaller output than IJG source (0.0 = cannot shrink).
/// Calibrated on 10 gb82 images (trimmed mean, top 20% dropped).
const IJG_SHRINK_CAP: &[(f32, f32)] = &[
    (10.0, 45.0),
    (20.0, 35.0),
    (30.0, 50.0),
    (40.0, 65.0),
    (50.0, 70.0),
    (65.0, 75.0),
    (75.0, 85.0),
    (80.0, 88.0),
    (85.0, 90.0),
    (90.0, 93.0),
];

/// Highest zen Q producing smaller output than mozjpeg source (0.0 = cannot shrink).
const MOZ_SHRINK_CAP: &[(f32, f32)] = &[
    (10.0, 0.0),
    (20.0, 0.0),
    (30.0, 0.0),
    (40.0, 20.0),
    (50.0, 40.0),
    (65.0, 60.0),
    (75.0, 75.0),
    (80.0, 80.0),
    (85.0, 85.0),
    (90.0, 90.0),
];

/// Highest zen Q producing smaller output than cjpegli source (0.0 = cannot shrink).
const JPEGLI_SHRINK_CAP: &[(f32, f32)] = &[
    (5.8, 0.0),
    (4.5, 0.0),
    (3.8, 25.0),
    (3.5, 35.0),
    (3.1, 50.0),
    (2.7, 65.0),
    (2.3, 75.0),
    (2.0, 80.0),
    (1.7, 85.0),
    (1.3, 90.0),
];

// ============================================================================
// Grid selection and interpolation
// ============================================================================

struct CalibrationGrid {
    src_qs: &'static [f32],
    grid: &'static [&'static [f32]],
}

fn grid_for_encoder(encoder: &EncoderFamily, scale: &QualityScale) -> CalibrationGrid {
    match encoder {
        EncoderFamily::CjpegliYcbcr | EncoderFamily::CjpegliXyb => CalibrationGrid {
            src_qs: JPEGLI_SRC_QS,
            grid: JPEGLI_GRID,
        },
        EncoderFamily::Mozjpeg => CalibrationGrid {
            src_qs: MOZ_SRC_QS,
            grid: MOZ_GRID,
        },
        EncoderFamily::LibjpegTurbo
        | EncoderFamily::ImageMagick
        | EncoderFamily::IjgFamily
        | EncoderFamily::Photoshop
        | EncoderFamily::Unknown => {
            if *scale == QualityScale::ButteraugliDistance {
                CalibrationGrid {
                    src_qs: JPEGLI_SRC_QS,
                    grid: JPEGLI_GRID,
                }
            } else {
                CalibrationGrid {
                    src_qs: IJG_SRC_QS,
                    grid: IJG_GRID,
                }
            }
        }
    }
}

fn min_delta_for_encoder(encoder: &EncoderFamily, scale: &QualityScale) -> &'static [(f32, f32)] {
    match encoder {
        EncoderFamily::CjpegliYcbcr | EncoderFamily::CjpegliXyb => JPEGLI_MIN_DELTA,
        EncoderFamily::Mozjpeg => MOZ_MIN_DELTA,
        EncoderFamily::LibjpegTurbo
        | EncoderFamily::ImageMagick
        | EncoderFamily::IjgFamily
        | EncoderFamily::Photoshop
        | EncoderFamily::Unknown => {
            if *scale == QualityScale::ButteraugliDistance {
                JPEGLI_MIN_DELTA
            } else {
                IJG_MIN_DELTA
            }
        }
    }
}

fn prop_grid_for_encoder(encoder: &EncoderFamily, scale: &QualityScale) -> CalibrationGrid {
    match encoder {
        EncoderFamily::CjpegliYcbcr | EncoderFamily::CjpegliXyb => CalibrationGrid {
            src_qs: JPEGLI_SRC_QS,
            grid: JPEGLI_PROP_GRID,
        },
        EncoderFamily::Mozjpeg => CalibrationGrid {
            src_qs: MOZ_SRC_QS,
            grid: MOZ_PROP_GRID,
        },
        EncoderFamily::LibjpegTurbo
        | EncoderFamily::ImageMagick
        | EncoderFamily::IjgFamily
        | EncoderFamily::Photoshop
        | EncoderFamily::Unknown => {
            if *scale == QualityScale::ButteraugliDistance {
                CalibrationGrid {
                    src_qs: JPEGLI_SRC_QS,
                    grid: JPEGLI_PROP_GRID,
                }
            } else {
                CalibrationGrid {
                    src_qs: IJG_SRC_QS,
                    grid: IJG_PROP_GRID,
                }
            }
        }
    }
}

fn shrink_cap_for_encoder(encoder: &EncoderFamily, scale: &QualityScale) -> &'static [(f32, f32)] {
    match encoder {
        EncoderFamily::CjpegliYcbcr | EncoderFamily::CjpegliXyb => JPEGLI_SHRINK_CAP,
        EncoderFamily::Mozjpeg => MOZ_SHRINK_CAP,
        EncoderFamily::LibjpegTurbo
        | EncoderFamily::ImageMagick
        | EncoderFamily::IjgFamily
        | EncoderFamily::Photoshop
        | EncoderFamily::Unknown => {
            if *scale == QualityScale::ButteraugliDistance {
                JPEGLI_SHRINK_CAP
            } else {
                IJG_SHRINK_CAP
            }
        }
    }
}

/// Highest quality that still produces a smaller file than the source.
///
/// Returns `None` if no quality level can guarantee a smaller file.
/// Returns `Some(q)` where q is the highest safe quality.
fn shrink_cap_q(probe: &JpegProbe) -> Option<f32> {
    let table = shrink_cap_for_encoder(&probe.encoder, &probe.quality.scale);
    let cap = interpolate_1d(table, probe.quality.value, &probe.quality.scale);
    if cap < 1.0 { None } else { Some(cap) }
}

/// Bilinear interpolation across the 2D calibration grid.
///
/// Interpolates in both dimensions (source quality and tolerance)
/// to produce a zen Q recommendation for any (source_q, tolerance) pair.
fn bilinear_lookup(
    grid: CalibrationGrid,
    source_val: f32,
    tolerance: f32,
    scale: &QualityScale,
) -> f32 {
    let is_descending = *scale == QualityScale::ButteraugliDistance;

    // Find bracketing tolerance indices
    let (tol_lo, tol_hi, tol_t) = bracket_tolerance(tolerance);

    // Interpolate source quality at each tolerance level
    let q_lo = interpolate_source(grid.src_qs, grid.grid[tol_lo], source_val, is_descending);
    let q_hi = interpolate_source(grid.src_qs, grid.grid[tol_hi], source_val, is_descending);

    // Interpolate between the two tolerance levels
    let q = q_lo + tol_t * (q_hi - q_lo);
    q.clamp(1.0, 100.0)
}

/// Find the two bracketing tolerance indices and interpolation factor.
fn bracket_tolerance(tol: f32) -> (usize, usize, f32) {
    let tols = TOLERANCES;

    if tol <= tols[0] {
        return (0, 0, 0.0);
    }
    let last = tols.len() - 1;
    if tol >= tols[last] {
        return (last, last, 0.0);
    }

    for i in 0..tols.len() - 1 {
        if tol >= tols[i] && tol <= tols[i + 1] {
            let t = (tol - tols[i]) / (tols[i + 1] - tols[i]);
            return (i, i + 1, t);
        }
    }

    (last, last, 0.0)
}

/// Find the two bracketing proportional factor indices and interpolation fraction.
fn bracket_factor(factor: f32) -> (usize, usize, f32) {
    let factors = PROP_FACTORS;

    if factor <= factors[0] {
        return (0, 0, 0.0);
    }
    let last = factors.len() - 1;
    if factor >= factors[last] {
        return (last, last, 0.0);
    }

    for i in 0..factors.len() - 1 {
        if factor >= factors[i] && factor <= factors[i + 1] {
            let t = (factor - factors[i]) / (factors[i + 1] - factors[i]);
            return (i, i + 1, t);
        }
    }

    (last, last, 0.0)
}

/// Recommended zenjpeg quality using proportional BA tolerance.
///
/// The proportional factor means: allow `ba_delta ≤ src_ba × factor`.
/// This gives constant perceptual impact regardless of source quality.
pub(crate) fn recommended_q_with_factor(probe: &JpegProbe, factor: f32) -> f32 {
    let f = factor.clamp(PROP_FACTORS[0], PROP_FACTORS[PROP_FACTORS.len() - 1]);
    let grid = prop_grid_for_encoder(&probe.encoder, &probe.quality.scale);
    let is_descending = probe.quality.scale == QualityScale::ButteraugliDistance;

    let (f_lo, f_hi, f_t) = bracket_factor(f);
    let q_lo = interpolate_source(
        grid.src_qs,
        grid.grid[f_lo],
        probe.quality.value,
        is_descending,
    );
    let q_hi = interpolate_source(
        grid.src_qs,
        grid.grid[f_hi],
        probe.quality.value,
        is_descending,
    );
    let q = q_lo + f_t * (q_hi - q_lo);
    q.clamp(1.0, 100.0)
}

/// Interpolate source quality within a single tolerance row.
fn interpolate_source(src_qs: &[f32], zen_qs: &[f32], source_val: f32, is_descending: bool) -> f32 {
    if is_descending {
        interpolate_source_descending(src_qs, zen_qs, source_val)
    } else {
        interpolate_source_ascending(src_qs, zen_qs, source_val)
    }
}

/// Interpolate in ascending source quality (IJG/mozjpeg: higher = better).
fn interpolate_source_ascending(src_qs: &[f32], zen_qs: &[f32], val: f32) -> f32 {
    if val <= src_qs[0] {
        if src_qs.len() < 2 {
            return zen_qs[0];
        }
        let slope = (zen_qs[1] - zen_qs[0]) / (src_qs[1] - src_qs[0]);
        return (zen_qs[0] + slope * (val - src_qs[0])).clamp(1.0, 100.0);
    }

    let last = src_qs.len() - 1;
    if val >= src_qs[last] {
        if src_qs.len() < 2 {
            return zen_qs[last];
        }
        let slope = (zen_qs[last] - zen_qs[last - 1]) / (src_qs[last] - src_qs[last - 1]);
        return (zen_qs[last] + slope * (val - src_qs[last])).clamp(1.0, 100.0);
    }

    for i in 0..src_qs.len() - 1 {
        if val >= src_qs[i] && val <= src_qs[i + 1] {
            let t = (val - src_qs[i]) / (src_qs[i + 1] - src_qs[i]);
            return zen_qs[i] + t * (zen_qs[i + 1] - zen_qs[i]);
        }
    }

    zen_qs[zen_qs.len() / 2]
}

/// Interpolate in descending source quality (jpegli BA distance: lower = better).
fn interpolate_source_descending(src_qs: &[f32], zen_qs: &[f32], val: f32) -> f32 {
    // src_qs is DESCENDING (high BA = low quality first)
    if val >= src_qs[0] {
        if src_qs.len() < 2 {
            return zen_qs[0];
        }
        let slope = (zen_qs[1] - zen_qs[0]) / (src_qs[1] - src_qs[0]);
        return (zen_qs[0] + slope * (val - src_qs[0])).clamp(1.0, 100.0);
    }

    let last = src_qs.len() - 1;
    if val <= src_qs[last] {
        if src_qs.len() < 2 {
            return zen_qs[last];
        }
        let slope = (zen_qs[last] - zen_qs[last - 1]) / (src_qs[last] - src_qs[last - 1]);
        return (zen_qs[last] + slope * (val - src_qs[last])).clamp(1.0, 100.0);
    }

    for i in 0..src_qs.len() - 1 {
        if val <= src_qs[i] && val >= src_qs[i + 1] {
            let t = (val - src_qs[i]) / (src_qs[i + 1] - src_qs[i]);
            return zen_qs[i] + t * (zen_qs[i + 1] - zen_qs[i]);
        }
    }

    zen_qs[zen_qs.len() / 2]
}

/// 1D interpolation for min-delta lookup (same as before but on (x,y) pairs).
fn interpolate_1d(table: &[(f32, f32)], val: f32, scale: &QualityScale) -> f32 {
    if *scale == QualityScale::ButteraugliDistance {
        interpolate_1d_descending(table, val)
    } else {
        interpolate_1d_ascending(table, val)
    }
}

fn interpolate_1d_ascending(table: &[(f32, f32)], val: f32) -> f32 {
    if val <= table[0].0 {
        return table[0].1;
    }
    let last = table.len() - 1;
    if val >= table[last].0 {
        if table.len() < 2 {
            return table[last].1;
        }
        let slope = (table[last].1 - table[last - 1].1) / (table[last].0 - table[last - 1].0);
        return table[last].1 + slope * (val - table[last].0);
    }
    for i in 0..table.len() - 1 {
        if val >= table[i].0 && val <= table[i + 1].0 {
            let t = (val - table[i].0) / (table[i + 1].0 - table[i].0);
            return table[i].1 + t * (table[i + 1].1 - table[i].1);
        }
    }
    table[table.len() / 2].1
}

fn interpolate_1d_descending(table: &[(f32, f32)], val: f32) -> f32 {
    if val >= table[0].0 {
        return table[0].1;
    }
    let last = table.len() - 1;
    if val <= table[last].0 {
        if table.len() < 2 {
            return table[last].1;
        }
        let slope = (table[last].1 - table[last - 1].1) / (table[last].0 - table[last - 1].0);
        return table[last].1 + slope * (val - table[last].0);
    }
    for i in 0..table.len() - 1 {
        if val <= table[i].0 && val >= table[i + 1].0 {
            let t = (val - table[i].0) / (table[i + 1].0 - table[i].0);
            return table[i].1 + t * (table[i + 1].1 - table[i].1);
        }
    }
    table[table.len() / 2].1
}

// ============================================================================
// Public API on JpegProbe
// ============================================================================

impl JpegProbe {
    /// Estimated source butteraugli distance.
    ///
    /// For cjpegli sources, returns the detected BA distance directly.
    /// For IJG/mozjpeg sources, converts from IJG quality to approximate BA
    /// using calibration medians.
    #[must_use]
    pub fn estimated_ba(&self) -> f32 {
        estimated_source_ba(self)
    }

    /// Recommended zenjpeg quality for re-encoding this JPEG.
    ///
    /// Returns the lowest quality level that keeps butteraugli delta ≤ 0.3
    /// from the source — barely perceptible degradation.
    ///
    /// For more control over the quality/size tradeoff, use
    /// [`reencode_settings`](Self::reencode_settings) with a custom tolerance.
    ///
    /// ```rust,ignore
    /// let probe = zenjpeg::detect::probe(&source_jpeg)?;
    /// let config = EncoderConfig::ycbcr(
    ///     probe.recommended_quality(),
    ///     probe.recommended_subsampling(),
    /// ).auto_optimize(true);
    /// ```
    #[must_use]
    pub fn recommended_quality(&self) -> Quality {
        Quality::ApproxJpegli(recommended_q(self))
    }

    /// Full re-encoding settings with configurable butteraugli tolerance.
    ///
    /// Returns both quality and subsampling recommendations.
    ///
    /// `ba_tolerance` controls how much quality degradation is acceptable:
    ///
    /// | Tolerance | Meaning | Typical size savings |
    /// |-----------|---------|---------------------|
    /// | 0.1 | Nearly imperceptible | Minimal |
    /// | 0.3 | Barely perceptible (default) | 0-10% |
    /// | 0.5 | Noticeable on close inspection | 5-30% |
    /// | 1.0 | Visible but acceptable | 15-45% |
    /// | 2.0 | Significant quality loss | 30-60% |
    ///
    /// # Errors
    ///
    /// Returns [`ReencodeError::InvalidTolerance`] if tolerance is ≤ 0.
    ///
    /// Returns [`ReencodeError::ToleranceTooTight`] if the requested tolerance
    /// is below what's achievable for this source — even Q97 would exceed it.
    /// The error includes the minimum achievable delta and best-effort settings.
    ///
    /// ```rust,ignore
    /// let probe = zenjpeg::detect::probe(&source_jpeg)?;
    /// let settings = probe.reencode_settings(0.5)?;
    /// let config = EncoderConfig::ycbcr(settings.quality, settings.subsampling)
    ///     .auto_optimize(true);
    /// ```
    pub fn reencode_settings(&self, ba_tolerance: f32) -> Result<ReencodeSettings, ReencodeError> {
        if ba_tolerance <= 0.0 {
            return Err(ReencodeError::InvalidTolerance);
        }

        let sub = ChromaSubsampling::from(self.subsampling);
        let cap = shrink_cap_q(self).map(Quality::ApproxJpegli);

        // Check if tolerance is achievable
        let min_delta = min_achievable_delta(self);
        if ba_tolerance < min_delta {
            return Err(ReencodeError::ToleranceTooTight {
                min_achievable: min_delta,
                best_effort: ReencodeSettings {
                    quality: Quality::ApproxJpegli(97.0),
                    subsampling: sub,
                    shrink_cap: cap,
                },
            });
        }

        let q = recommended_q_with_tolerance(self, ba_tolerance);
        Ok(ReencodeSettings {
            quality: Quality::ApproxJpegli(q),
            subsampling: sub,
            shrink_cap: cap,
        })
    }

    /// Full re-encoding settings with proportional BA tolerance.
    ///
    /// The proportional factor means: allow `ba_delta ≤ src_ba × factor`.
    /// This gives constant perceptual impact regardless of source quality,
    /// unlike absolute tolerance which is too tight for low-quality sources
    /// and too loose for high-quality ones.
    ///
    /// | Factor | Meaning | Typical size savings |
    /// |--------|---------|---------------------|
    /// | 0.05 | Nearly imperceptible | Minimal |
    /// | 0.10 | Barely perceptible | 0-10% |
    /// | 0.15 | Slight degradation (default) | 5-25% |
    /// | 0.30 | Noticeable on inspection | 15-40% |
    /// | 0.50 | Significant quality loss | 30-60% |
    ///
    /// # Errors
    ///
    /// Returns [`ReencodeError::InvalidTolerance`] if factor is ≤ 0.
    ///
    /// Returns [`ReencodeError::ToleranceTooTight`] if the effective tolerance
    /// (`src_ba × factor`) is below what's achievable for this source.
    pub fn reencode_settings_proportional(
        &self,
        factor: f32,
    ) -> Result<ReencodeSettings, ReencodeError> {
        if factor <= 0.0 {
            return Err(ReencodeError::InvalidTolerance);
        }

        let sub = ChromaSubsampling::from(self.subsampling);
        let cap = shrink_cap_q(self).map(Quality::ApproxJpegli);

        // Check if effective tolerance is achievable
        let src_ba = estimated_source_ba(self);
        let effective_tol = src_ba * factor;
        let min_delta = min_achievable_delta(self);
        if effective_tol < min_delta {
            return Err(ReencodeError::ToleranceTooTight {
                min_achievable: min_delta,
                best_effort: ReencodeSettings {
                    quality: Quality::ApproxJpegli(97.0),
                    subsampling: sub,
                    shrink_cap: cap,
                },
            });
        }

        let q = recommended_q_with_factor(self, factor);
        Ok(ReencodeSettings {
            quality: Quality::ApproxJpegli(q),
            subsampling: sub,
            shrink_cap: cap,
        })
    }

    /// Maximum useful quality when downscaling before re-encoding.
    ///
    /// Above this ceiling, additional bytes produce imperceptible quality gains
    /// (~40% more bytes for <0.3 butteraugli improvement).
    ///
    /// `downscale_ratio` is the ratio of input to output dimensions,
    /// e.g., 2.0 means halving width and height (4x fewer pixels).
    #[must_use]
    pub fn quality_ceiling(downscale_ratio: f32) -> Quality {
        Quality::ApproxJpegli(quality_ceiling(downscale_ratio))
    }

    /// Recommended [`ChromaSubsampling`] matching the source JPEG.
    ///
    /// Preserves the source's subsampling mode. If the source uses 4:2:0,
    /// the recommendation is 4:2:0 — re-encoding won't improve chroma resolution
    /// that was already discarded.
    #[must_use]
    pub fn recommended_subsampling(&self) -> ChromaSubsampling {
        ChromaSubsampling::from(self.subsampling)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detect::quality::{Confidence, QualityEstimate};
    use crate::types::{Dimensions, JpegMode, Subsampling};

    fn mock_probe(
        encoder: EncoderFamily,
        quality_value: f32,
        scale: QualityScale,
        subsampling: Subsampling,
    ) -> JpegProbe {
        JpegProbe {
            encoder,
            quality: QualityEstimate {
                value: quality_value,
                scale,
                confidence: Confidence::Exact,
            },
            dimensions: Dimensions::new(512, 512),
            subsampling,
            mode: JpegMode::Progressive,
            num_components: 3,
            scan_count: 10,
            dqt_tables: alloc::vec::Vec::new(),
        }
    }

    // =======================================================================
    // Default tolerance (0.3) tests — verify same results as before
    // =======================================================================

    #[test]
    fn test_turbo_default_recommendations() {
        // IJG_GRID tol=0.3 row (trimmed mean calibration)
        let cases = [
            (10.0, 20.0),
            (20.0, 25.0),
            (30.0, 35.0),
            (40.0, 50.0),
            (50.0, 60.0),
            (65.0, 70.0),
            (75.0, 85.0),
            (80.0, 85.0),
            (85.0, 88.0),
            (90.0, 97.0),
        ];
        for (src_q, expected) in cases {
            let probe = mock_probe(
                EncoderFamily::LibjpegTurbo,
                src_q,
                QualityScale::IjgQuality,
                Subsampling::S420,
            );
            let q = recommended_q(&probe);
            assert!(
                (q - expected).abs() < 0.01,
                "turbo Q{src_q}: expected {expected}, got {q}"
            );
        }
    }

    #[test]
    fn test_mozjpeg_default_recommendations() {
        // MOZ_GRID tol=0.3 row (trimmed mean calibration)
        let cases = [
            (10.0, 20.0),
            (20.0, 20.0),
            (30.0, 35.0),
            (40.0, 45.0),
            (50.0, 55.0),
            (65.0, 65.0),
            (75.0, 85.0),
            (80.0, 85.0),
            (85.0, 90.0),
            (90.0, 95.0),
        ];
        for (src_q, expected) in cases {
            let probe = mock_probe(
                EncoderFamily::Mozjpeg,
                src_q,
                QualityScale::MozjpegQuality,
                Subsampling::S420,
            );
            let q = recommended_q(&probe);
            assert!(
                (q - expected).abs() < 0.01,
                "mozjpeg Q{src_q}: expected {expected}, got {q}"
            );
        }
    }

    #[test]
    fn test_jpegli_default_recommendations() {
        // JPEGLI_GRID tol=0.3 row, using exact grid points from JPEGLI_SRC_QS
        let cases = [
            (5.8, 25.0),
            (4.5, 25.0),
            (3.8, 30.0),
            (3.5, 40.0),
            (3.1, 45.0),
            (2.7, 60.0),
            (2.3, 75.0),
            (2.0, 80.0),
            (1.7, 85.0),
            (1.3, 88.0),
        ];
        for (src_dist, expected) in cases {
            let probe = mock_probe(
                EncoderFamily::CjpegliYcbcr,
                src_dist,
                QualityScale::ButteraugliDistance,
                Subsampling::S444,
            );
            let q = recommended_q(&probe);
            assert!(
                (q - expected).abs() < 0.01,
                "jpegli dist={src_dist}: expected {expected}, got {q}"
            );
        }
    }

    // =======================================================================
    // Tolerance sweep tests
    // =======================================================================

    #[test]
    fn test_higher_tolerance_gives_lower_q() {
        // Across all encoders: higher tolerance → lower zen Q
        for (enc, src_q, scale) in [
            (EncoderFamily::LibjpegTurbo, 85.0, QualityScale::IjgQuality),
            (EncoderFamily::Mozjpeg, 85.0, QualityScale::MozjpegQuality),
            (
                EncoderFamily::CjpegliYcbcr,
                1.8,
                QualityScale::ButteraugliDistance,
            ),
        ] {
            let probe = mock_probe(enc, src_q, scale, Subsampling::S420);
            let mut prev_q = 200.0;
            for tol in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0] {
                let q = recommended_q_with_tolerance(&probe, tol);
                assert!(
                    q <= prev_q + 0.01, // allow tiny floating-point wiggle
                    "{enc:?}: tol={tol}: Q{q} > prev Q{prev_q}"
                );
                prev_q = q;
            }
        }
    }

    #[test]
    fn test_turbo_at_various_tolerances() {
        let probe = mock_probe(
            EncoderFamily::LibjpegTurbo,
            85.0,
            QualityScale::IjgQuality,
            Subsampling::S420,
        );

        // Exact grid points from IJG_GRID
        let q01 = recommended_q_with_tolerance(&probe, 0.1);
        let q03 = recommended_q_with_tolerance(&probe, 0.3);
        let q05 = recommended_q_with_tolerance(&probe, 0.5);
        let q10 = recommended_q_with_tolerance(&probe, 1.0);

        assert!((q01 - 97.0).abs() < 0.01, "tol=0.1: got {q01}");
        assert!((q03 - 88.0).abs() < 0.01, "tol=0.3: got {q03}");
        assert!((q05 - 80.0).abs() < 0.01, "tol=0.5: got {q05}");
        assert!((q10 - 65.0).abs() < 0.01, "tol=1.0: got {q10}");
    }

    #[test]
    fn test_interpolation_between_tolerance_levels() {
        let probe = mock_probe(
            EncoderFamily::LibjpegTurbo,
            90.0,
            QualityScale::IjgQuality,
            Subsampling::S420,
        );

        // tol=0.4→97, tol=0.5→95 (IJG_GRID)
        // Interpolate between 0.4 and 0.5: midpoint at 0.45 → (97+95)/2 = 96
        let q = recommended_q_with_tolerance(&probe, 0.45);
        assert!(
            (q - 96.0).abs() < 0.01,
            "turbo Q90 tol=0.45: expected 96.0, got {q}"
        );
    }

    #[test]
    fn test_interpolation_between_source_qualities() {
        let probe = mock_probe(
            EncoderFamily::LibjpegTurbo,
            57.5, // midpoint of Q50 and Q65
            QualityScale::IjgQuality,
            Subsampling::S420,
        );

        // At tol=0.3: Q50→60, Q65→70, midpoint=65
        let q = recommended_q_with_tolerance(&probe, 0.3);
        assert!(
            (q - 65.0).abs() < 0.1,
            "turbo Q57.5 tol=0.3: expected 65.0, got {q}"
        );
    }

    // =======================================================================
    // reencode_settings() tests
    // =======================================================================

    #[test]
    fn test_reencode_settings_returns_subsampling() {
        let probe = mock_probe(
            EncoderFamily::LibjpegTurbo,
            85.0,
            QualityScale::IjgQuality,
            Subsampling::S422,
        );
        let settings = probe.reencode_settings(0.3).unwrap();
        assert_eq!(settings.subsampling, ChromaSubsampling::HalfHorizontal);
    }

    #[test]
    fn test_reencode_settings_invalid_tolerance() {
        let probe = mock_probe(
            EncoderFamily::LibjpegTurbo,
            85.0,
            QualityScale::IjgQuality,
            Subsampling::S420,
        );
        assert!(matches!(
            probe.reencode_settings(0.0),
            Err(ReencodeError::InvalidTolerance)
        ));
        assert!(matches!(
            probe.reencode_settings(-1.0),
            Err(ReencodeError::InvalidTolerance)
        ));
    }

    #[test]
    fn test_reencode_settings_tolerance_too_tight() {
        // turbo Q90 has min_delta ~0.03 — requesting 0.01 should fail
        let probe = mock_probe(
            EncoderFamily::LibjpegTurbo,
            90.0,
            QualityScale::IjgQuality,
            Subsampling::S420,
        );
        let err = probe.reencode_settings(0.01).unwrap_err();
        match err {
            ReencodeError::ToleranceTooTight {
                min_achievable,
                best_effort,
            } => {
                assert!(min_achievable > 0.02, "min_achievable={min_achievable}");
                assert!(matches!(best_effort.quality, Quality::ApproxJpegli(q) if q == 97.0));
                assert_eq!(best_effort.subsampling, ChromaSubsampling::Quarter);
            }
            _ => panic!("expected ToleranceTooTight, got {err:?}"),
        }
    }

    #[test]
    fn test_reencode_settings_achievable_tolerance() {
        // cjpegli Q50 (BA ~3.2) has min_delta ~0.01 — even tight tolerance works
        let probe = mock_probe(
            EncoderFamily::CjpegliYcbcr,
            3.2,
            QualityScale::ButteraugliDistance,
            Subsampling::S444,
        );
        let settings = probe.reencode_settings(0.1).unwrap();
        assert!(matches!(settings.quality, Quality::ApproxJpegli(q) if q > 20.0));
        assert_eq!(settings.subsampling, ChromaSubsampling::None);
    }

    #[test]
    fn test_reencode_settings_large_tolerance_always_succeeds() {
        // Any large tolerance should succeed for all encoders
        for (enc, src_q, scale) in [
            (EncoderFamily::LibjpegTurbo, 90.0, QualityScale::IjgQuality),
            (EncoderFamily::Mozjpeg, 90.0, QualityScale::MozjpegQuality),
            (
                EncoderFamily::CjpegliYcbcr,
                1.4,
                QualityScale::ButteraugliDistance,
            ),
        ] {
            let probe = mock_probe(enc, src_q, scale, Subsampling::S420);
            assert!(
                probe.reencode_settings(1.0).is_ok(),
                "{enc:?}: tol=1.0 should succeed"
            );
        }
    }

    // =======================================================================
    // Quality ceiling tests
    // =======================================================================

    #[test]
    fn test_quality_ceiling_values() {
        assert!((quality_ceiling(1.5) - 90.0).abs() < 0.01);
        assert!((quality_ceiling(2.0) - 90.0).abs() < 0.01);
        assert!((quality_ceiling(3.0) - 90.0).abs() < 0.01);
        assert!((quality_ceiling(4.0) - 90.0).abs() < 0.01);
        assert!(quality_ceiling(1.0) > 90.0);
        assert!(quality_ceiling(0.5) > 93.0);
    }

    // =======================================================================
    // Subsampling tests
    // =======================================================================

    #[test]
    fn test_recommended_subsampling() {
        let cases = [
            (Subsampling::S444, ChromaSubsampling::None),
            (Subsampling::S420, ChromaSubsampling::Quarter),
            (Subsampling::S422, ChromaSubsampling::HalfHorizontal),
            (Subsampling::S440, ChromaSubsampling::HalfVertical),
        ];
        for (src, expected) in cases {
            let probe = mock_probe(
                EncoderFamily::LibjpegTurbo,
                85.0,
                QualityScale::IjgQuality,
                src,
            );
            assert_eq!(probe.recommended_subsampling(), expected);
        }
    }

    // =======================================================================
    // Monotonicity tests
    // =======================================================================

    #[test]
    fn test_monotonic_across_source_qualities() {
        // Higher source quality should give higher (or equal) zen Q
        for encoder in [EncoderFamily::LibjpegTurbo, EncoderFamily::Mozjpeg] {
            for tol in [0.3, 0.5, 1.0] {
                let mut prev_q = 0.0f32;
                for src_q in [50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0] {
                    let probe = mock_probe(
                        encoder,
                        src_q,
                        match encoder {
                            EncoderFamily::Mozjpeg => QualityScale::MozjpegQuality,
                            _ => QualityScale::IjgQuality,
                        },
                        Subsampling::S420,
                    );
                    let q = recommended_q_with_tolerance(&probe, tol);
                    assert!(
                        q >= prev_q - 0.01,
                        "{encoder:?} tol={tol} Q{src_q}: {q} < prev {prev_q}"
                    );
                    prev_q = q;
                }
            }
        }
    }

    #[test]
    fn test_unknown_encoder_uses_conservative() {
        let probe_unknown = mock_probe(
            EncoderFamily::Unknown,
            85.0,
            QualityScale::IjgQuality,
            Subsampling::S420,
        );
        let probe_turbo = mock_probe(
            EncoderFamily::LibjpegTurbo,
            85.0,
            QualityScale::IjgQuality,
            Subsampling::S420,
        );
        let q_unknown = recommended_q(&probe_unknown);
        let q_turbo = recommended_q(&probe_turbo);
        assert!(
            (q_unknown - q_turbo).abs() < 0.01,
            "unknown={q_unknown}, turbo={q_turbo}"
        );
    }

    #[test]
    fn test_clamped_to_valid_range() {
        let probe = mock_probe(
            EncoderFamily::LibjpegTurbo,
            50.0,
            QualityScale::IjgQuality,
            Subsampling::S420,
        );
        // Extreme tolerance
        let q = recommended_q_with_tolerance(&probe, 10.0);
        assert!((1.0..=100.0).contains(&q), "extreme tol: Q={q}");
        // Very tight tolerance
        let q = recommended_q_with_tolerance(&probe, 0.05);
        assert!((1.0..=100.0).contains(&q), "tight tol: Q={q}");
    }

    #[test]
    fn test_error_display() {
        let err = ReencodeError::InvalidTolerance;
        assert_eq!(err.to_string(), "tolerance must be positive");

        let err = ReencodeError::ToleranceTooTight {
            min_achievable: 0.25,
            best_effort: ReencodeSettings {
                quality: Quality::ApproxJpegli(97.0),
                subsampling: ChromaSubsampling::Quarter,
                shrink_cap: Some(Quality::ApproxJpegli(85.0)),
            },
        };
        assert!(err.to_string().contains("0.25"));
    }

    // =======================================================================
    // Shrink cap tests
    // =======================================================================

    #[test]
    fn test_shrink_cap_turbo() {
        // turbo Q75 → cap should be 85.0 (trimmed mean calibration)
        let probe = mock_probe(
            EncoderFamily::LibjpegTurbo,
            75.0,
            QualityScale::IjgQuality,
            Subsampling::S420,
        );
        let settings = probe.reencode_settings(0.3).unwrap();
        assert!(
            matches!(settings.shrink_cap, Some(Quality::ApproxJpegli(q)) if (q - 85.0).abs() < 0.01),
            "turbo Q75 shrink cap: expected 85.0, got {:?}",
            settings.shrink_cap
        );
    }

    #[test]
    fn test_shrink_cap_turbo_low_q() {
        // turbo Q10 → cap=45.0 (trimmed mean finds some shrink possible)
        let probe = mock_probe(
            EncoderFamily::LibjpegTurbo,
            10.0,
            QualityScale::IjgQuality,
            Subsampling::S420,
        );
        let settings = probe.reencode_settings(0.3).unwrap();
        assert!(
            matches!(settings.shrink_cap, Some(Quality::ApproxJpegli(q)) if (q - 45.0).abs() < 0.01),
            "turbo Q10 shrink cap: expected 45.0, got {:?}",
            settings.shrink_cap
        );
    }

    #[test]
    fn test_shrink_cap_mozjpeg() {
        // mozjpeg Q75 → cap should be 75.0 (trimmed mean calibration)
        let probe = mock_probe(
            EncoderFamily::Mozjpeg,
            75.0,
            QualityScale::MozjpegQuality,
            Subsampling::S420,
        );
        let settings = probe.reencode_settings(0.3).unwrap();
        assert!(
            matches!(settings.shrink_cap, Some(Quality::ApproxJpegli(q)) if (q - 75.0).abs() < 0.01),
            "mozjpeg Q75 shrink cap: expected 75.0, got {:?}",
            settings.shrink_cap
        );
    }

    #[test]
    fn test_shrink_cap_mozjpeg_low_q() {
        // mozjpeg Q20 → no shrink possible
        let probe = mock_probe(
            EncoderFamily::Mozjpeg,
            20.0,
            QualityScale::MozjpegQuality,
            Subsampling::S420,
        );
        let settings = probe.reencode_settings(0.3).unwrap();
        assert!(
            settings.shrink_cap.is_none(),
            "mozjpeg Q20 should have no shrink cap, got {:?}",
            settings.shrink_cap
        );
    }

    #[test]
    fn test_shrink_cap_cjpegli() {
        // cjpegli BA=2.0 → cap should be 80.0 (trimmed mean calibration)
        let probe = mock_probe(
            EncoderFamily::CjpegliYcbcr,
            2.0,
            QualityScale::ButteraugliDistance,
            Subsampling::S444,
        );
        let settings = probe.reencode_settings(0.3).unwrap();
        assert!(
            matches!(settings.shrink_cap, Some(Quality::ApproxJpegli(q)) if (q - 80.0).abs() < 0.01),
            "cjpegli BA=2.0 shrink cap: expected 80.0, got {:?}",
            settings.shrink_cap
        );
    }
}
