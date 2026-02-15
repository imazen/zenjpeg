//! Re-encoding quality recommendations based on empirical calibration.
//!
//! Maps source encoder family + quality level → recommended zenjpeg quality
//! for re-encoding with configurable quality loss tolerance.
//!
//! Two calibrated anchor points per encoder (ba_delta ≤ 0.3 and ≤ 0.5),
//! with linear interpolation/extrapolation for any tolerance value.
//!
//! Calibration data from 25-image sweep across libjpeg-turbo, mozjpeg, and cjpegli
//! at Q50-Q90, measuring butteraugli delta and file size ratio.

use super::fingerprint::EncoderFamily;
use super::quality::QualityScale;
use super::JpegProbe;
use crate::encode::encoder_types::{ChromaSubsampling, Quality};

/// Default butteraugli tolerance: barely perceptible degradation.
const DEFAULT_BA_TOLERANCE: f32 = 0.3;

/// Anchor tolerance for the "tight" calibration tables.
const TIGHT_TOL: f32 = 0.3;
/// Anchor tolerance for the "loose" calibration tables.
const LOOSE_TOL: f32 = 0.5;

/// Recommended zenjpeg quality for re-encoding at the default tolerance (≤0.3 BA delta).
pub(crate) fn recommended_q(probe: &JpegProbe) -> f32 {
    recommended_q_with_tolerance(probe, DEFAULT_BA_TOLERANCE)
}

/// Recommended zenjpeg quality for re-encoding with a custom BA delta tolerance.
///
/// Interpolates between two calibrated anchor points (tol=0.3 and tol=0.5)
/// and extrapolates linearly beyond that range.
pub(crate) fn recommended_q_with_tolerance(probe: &JpegProbe, ba_tolerance: f32) -> f32 {
    let tight = table_for_encoder(&probe.encoder, &probe.quality.scale, Tolerance::Tight);
    let loose = table_for_encoder(&probe.encoder, &probe.quality.scale, Tolerance::Loose);

    let q_tight = interpolate(tight, probe.quality.value, &probe.quality.scale);
    let q_loose = interpolate(loose, probe.quality.value, &probe.quality.scale);

    // Linear interpolation/extrapolation between the two anchor tolerances
    let t = (ba_tolerance - TIGHT_TOL) / (LOOSE_TOL - TIGHT_TOL);
    let q = q_tight + t * (q_loose - q_tight);
    q.clamp(1.0, 100.0)
}

/// Maximum useful quality when downscaling before re-encoding.
///
/// Above this ceiling, additional bytes produce <0.3 BA improvement
/// per ~40% file size increase — diminishing returns.
///
/// Calibrated across 1.5x-4x downscale ratios on 25 images.
/// All ratios converge on Q90 as the ceiling.
pub(crate) fn quality_ceiling(downscale_ratio: f32) -> f32 {
    // Calibration data: at all tested ratios (1.5x-4x), going from Q90→Q95
    // costs ~40% more bytes for <0.3 BA improvement.
    //
    // Full R-D curve shows Q90 is the knee of the curve for all ratios:
    //   1.5x: Q90 BA=2.06 72% → Q95 BA=1.91 103% (+0.15 BA for +43% size)
    //   2.0x: Q90 BA=1.95 51% → Q95 BA=1.70  73% (+0.25 BA for +43% size)
    //   3.0x: Q90 BA=1.75 27% → Q95 BA=1.47  38% (+0.29 BA for +40% size)
    //   4.0x: Q90 BA=1.64 18% → Q95 BA=1.32  25% (+0.33 BA for +38% size)
    if downscale_ratio < 1.0 {
        // Upscaling — no ceiling, user controls quality
        97.0
    } else if downscale_ratio < 1.25 {
        // Minimal resize — treat like re-encode without resize
        93.0
    } else {
        // 1.5x and beyond: Q90 is the universal ceiling
        90.0
    }
}

// ============================================================================
// Per-encoder calibration tables
// ============================================================================

// Each table maps (source_quality_value, recommended_zen_q).
// Source quality is in the encoder's native scale:
//   - IJG/turbo: IJG quality 1-100
//   - mozjpeg: mozjpeg quality 1-100
//   - jpegli: butteraugli distance (lower = better, DESCENDING order)
//
// Two tolerance levels calibrated on 25 gb82 images with auto_optimize:
//   - TIGHT (ba_delta ≤ 0.3): barely perceptible, minimal size savings
//   - LOOSE (ba_delta ≤ 0.5): noticeable only on close inspection, more savings

#[derive(Clone, Copy)]
enum Tolerance {
    /// ba_delta ≤ 0.3 — barely perceptible
    Tight,
    /// ba_delta ≤ 0.5 — noticeable only on close inspection
    Loose,
}

// --- libjpeg-turbo / IJG family ---
// Least efficient source → needs highest zen Q, biggest savings opportunity.

const IJG_TIGHT: &[(f32, f32)] = &[
    (50.0, 65.0),
    (65.0, 70.0),
    (75.0, 85.0),
    (80.0, 88.0),
    (85.0, 90.0),
    (90.0, 95.0),
];

const IJG_LOOSE: &[(f32, f32)] = &[
    (50.0, 55.0),
    (65.0, 65.0),
    (75.0, 75.0),
    (80.0, 75.0),
    (85.0, 80.0),
    (90.0, 88.0),
];

// --- mozjpeg ---
// Trellis-optimized source → needs moderate zen Q.

const MOZ_TIGHT: &[(f32, f32)] = &[
    (50.0, 55.0),
    (65.0, 65.0),
    (75.0, 80.0),
    (80.0, 85.0),
    (85.0, 88.0),
    (90.0, 93.0),
];

const MOZ_LOOSE: &[(f32, f32)] = &[
    (50.0, 50.0),
    (65.0, 55.0),
    (75.0, 70.0),
    (80.0, 75.0),
    (85.0, 80.0),
    (90.0, 85.0),
];

// --- cjpegli / zenjpeg ---
// Same algorithm family → lowest zen Q needed.
// Source quality is butteraugli distance (DESCENDING = higher quality).

const JPEGLI_TIGHT: &[(f32, f32)] = &[
    (3.4, 50.0), // ~cjpegli Q50
    (2.8, 60.0), // ~cjpegli Q65
    (2.4, 75.0), // ~cjpegli Q75
    (2.1, 80.0), // ~cjpegli Q80
    (1.8, 85.0), // ~cjpegli Q85
    (1.4, 88.0), // ~cjpegli Q90
];

const JPEGLI_LOOSE: &[(f32, f32)] = &[
    (3.4, 50.0), // ~cjpegli Q50 (already minimal)
    (2.8, 55.0), // ~cjpegli Q65
    (2.4, 70.0), // ~cjpegli Q75
    (2.1, 75.0), // ~cjpegli Q80
    (1.8, 80.0), // ~cjpegli Q85
    (1.4, 85.0), // ~cjpegli Q90
];

/// Select the appropriate lookup table for this encoder family and tolerance.
fn table_for_encoder(
    encoder: &EncoderFamily,
    scale: &QualityScale,
    tol: Tolerance,
) -> &'static [(f32, f32)] {
    match encoder {
        EncoderFamily::CjpegliYcbcr | EncoderFamily::CjpegliXyb => match tol {
            Tolerance::Tight => JPEGLI_TIGHT,
            Tolerance::Loose => JPEGLI_LOOSE,
        },
        EncoderFamily::Mozjpeg => match tol {
            Tolerance::Tight => MOZ_TIGHT,
            Tolerance::Loose => MOZ_LOOSE,
        },
        // IJG family (turbo, ImageMagick, generic IJG) and unknown encoders
        // use the conservative IJG table (highest Q = safest)
        EncoderFamily::LibjpegTurbo
        | EncoderFamily::ImageMagick
        | EncoderFamily::IjgFamily
        | EncoderFamily::Unknown => {
            // If the quality was detected as butteraugli distance (unusual for IJG
            // family, but possible for unknown encoders), use jpegli table
            if *scale == QualityScale::ButteraugliDistance {
                match tol {
                    Tolerance::Tight => JPEGLI_TIGHT,
                    Tolerance::Loose => JPEGLI_LOOSE,
                }
            } else {
                match tol {
                    Tolerance::Tight => IJG_TIGHT,
                    Tolerance::Loose => IJG_LOOSE,
                }
            }
        }
    }
}

/// Interpolate within a calibration table.
///
/// For IJG/mozjpeg tables: ascending source quality (higher = better).
/// For jpegli table: descending BA distance (lower = better quality).
fn interpolate(table: &[(f32, f32)], source_val: f32, scale: &QualityScale) -> f32 {
    if table.is_empty() {
        return 85.0; // safe default
    }

    let is_distance = *scale == QualityScale::ButteraugliDistance;

    if is_distance {
        // BA distance: table is in DESCENDING order (high BA = low quality first)
        // Source val: lower = better quality = higher zen Q needed
        interpolate_descending(table, source_val)
    } else {
        // IJG/mozjpeg: table is in ASCENDING order (low Q first)
        interpolate_ascending(table, source_val)
    }
}

/// Interpolate in an ascending table (IJG/mozjpeg quality: higher = better).
fn interpolate_ascending(table: &[(f32, f32)], val: f32) -> f32 {
    // Below table range: extrapolate from first two entries
    if val <= table[0].0 {
        if table.len() < 2 {
            return table[0].1;
        }
        let (x0, y0) = table[0];
        let (x1, y1) = table[1];
        let slope = (y1 - y0) / (x1 - x0);
        return (y0 + slope * (val - x0)).clamp(1.0, 100.0);
    }

    // Above table range: extrapolate from last two entries
    let last = table.len() - 1;
    if val >= table[last].0 {
        if table.len() < 2 {
            return table[last].1;
        }
        let (x0, y0) = table[last - 1];
        let (x1, y1) = table[last];
        let slope = (y1 - y0) / (x1 - x0);
        return (y1 + slope * (val - x1)).clamp(1.0, 100.0);
    }

    // Find bracketing entries and interpolate
    for i in 0..table.len() - 1 {
        let (x0, y0) = table[i];
        let (x1, y1) = table[i + 1];
        if val >= x0 && val <= x1 {
            let t = (val - x0) / (x1 - x0);
            return y0 + t * (y1 - y0);
        }
    }

    // Fallback (shouldn't reach here)
    table[table.len() / 2].1
}

/// Interpolate in a descending table (BA distance: lower = better quality).
fn interpolate_descending(table: &[(f32, f32)], val: f32) -> f32 {
    // Above table range (worse quality than our worst calibration point):
    // extrapolate from first two entries
    if val >= table[0].0 {
        if table.len() < 2 {
            return table[0].1;
        }
        let (x0, y0) = table[0];
        let (x1, y1) = table[1];
        let slope = (y1 - y0) / (x1 - x0); // negative slope (lower dist → higher Q)
        return (y0 + slope * (val - x0)).clamp(1.0, 100.0);
    }

    // Below table range (better quality than our best calibration point):
    // extrapolate from last two entries
    let last = table.len() - 1;
    if val <= table[last].0 {
        if table.len() < 2 {
            return table[last].1;
        }
        let (x0, y0) = table[last - 1];
        let (x1, y1) = table[last];
        let slope = (y1 - y0) / (x1 - x0);
        return (y1 + slope * (val - x1)).clamp(1.0, 100.0);
    }

    // Find bracketing entries (descending: table[i].0 > table[i+1].0)
    for i in 0..table.len() - 1 {
        let (x0, y0) = table[i];
        let (x1, y1) = table[i + 1];
        if val <= x0 && val >= x1 {
            let t = (val - x0) / (x1 - x0);
            return y0 + t * (y1 - y0);
        }
    }

    // Fallback
    table[table.len() / 2].1
}

impl JpegProbe {
    /// Recommended zenjpeg quality for re-encoding this JPEG.
    ///
    /// Returns the lowest quality level that keeps butteraugli delta ≤ 0.3
    /// from the source — barely perceptible degradation. Based on empirical
    /// calibration per source encoder family (25 images, 3 encoders, 6 quality levels).
    ///
    /// For more aggressive compression, use
    /// [`recommended_quality_with_tolerance`](Self::recommended_quality_with_tolerance)
    /// with a higher threshold.
    ///
    /// For best results, combine with [`auto_optimize(true)`](crate::encode::EncoderConfig::auto_optimize):
    ///
    /// ```rust,ignore
    /// use zenjpeg::detect;
    /// use zenjpeg::encode::EncoderConfig;
    ///
    /// let probe = detect::probe(&source_jpeg)?;
    /// let config = EncoderConfig::ycbcr(
    ///     probe.recommended_quality(),
    ///     probe.recommended_subsampling(),
    /// ).auto_optimize(true);
    /// ```
    ///
    /// # Encoder-specific behavior
    ///
    /// The recommended quality depends on which encoder produced the source JPEG:
    ///
    /// | Source encoder | Typical zen Q | Notes |
    /// |---------------|--------------|-------|
    /// | libjpeg-turbo | Higher (+10-25) | Least efficient source, biggest savings opportunity |
    /// | mozjpeg | Similar (+0-8) | Already well-optimized, modest savings |
    /// | cjpegli/zenjpeg | Lower (-5 to +0) | Same algorithm family, near-parity |
    ///
    /// Unknown encoders use conservative (higher Q) estimates.
    #[must_use]
    pub fn recommended_quality(&self) -> Quality {
        Quality::ApproxJpegli(recommended_q(self))
    }

    /// Recommended zenjpeg quality with a custom butteraugli tolerance.
    ///
    /// `ba_tolerance` controls how much quality degradation is acceptable:
    ///
    /// | Tolerance | Meaning | Typical size savings |
    /// |-----------|---------|---------------------|
    /// | 0.0 | Exact quality match | Minimal (encoder efficiency only) |
    /// | 0.3 | Barely perceptible (default) | 0-10% vs source |
    /// | 0.5 | Noticeable only on close inspection | 5-30% vs source |
    /// | 1.0 | Visible but acceptable | 15-45% vs source |
    ///
    /// Interpolates between two empirically calibrated anchor points
    /// (0.3 and 0.5) and extrapolates linearly beyond that range.
    ///
    /// ```rust,ignore
    /// use zenjpeg::detect;
    /// use zenjpeg::encode::EncoderConfig;
    ///
    /// let probe = detect::probe(&source_jpeg)?;
    /// // Accept up to 0.5 BA degradation for smaller files
    /// let config = EncoderConfig::ycbcr(
    ///     probe.recommended_quality_with_tolerance(0.5),
    ///     probe.recommended_subsampling(),
    /// ).auto_optimize(true);
    /// ```
    #[must_use]
    pub fn recommended_quality_with_tolerance(&self, ba_tolerance: f32) -> Quality {
        Quality::ApproxJpegli(recommended_q_with_tolerance(self, ba_tolerance))
    }

    /// Maximum useful quality when downscaling before re-encoding.
    ///
    /// Above this ceiling, additional bytes produce imperceptible quality gains
    /// (~40% more bytes for <0.3 butteraugli improvement).
    ///
    /// `downscale_ratio` is the ratio of input to output dimensions,
    /// e.g., 2.0 means halving width and height (4x fewer pixels).
    ///
    /// ```rust,ignore
    /// use zenjpeg::detect;
    /// use zenjpeg::encode::EncoderConfig;
    ///
    /// let probe = detect::probe(&source_jpeg)?;
    /// let q_match = probe.recommended_quality().to_internal();
    /// let q_ceil = detect::JpegProbe::quality_ceiling(2.0).to_internal();
    /// let q = q_match.min(q_ceil);
    ///
    /// let config = EncoderConfig::ycbcr(q, probe.recommended_subsampling())
    ///     .auto_optimize(true);
    /// ```
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
            dqt_tables: Vec::new(),
        }
    }

    #[test]
    fn test_turbo_recommendations() {
        // Exact calibration points
        let cases = [
            (50.0, 65.0),
            (65.0, 70.0),
            (75.0, 85.0),
            (80.0, 88.0),
            (85.0, 90.0),
            (90.0, 95.0),
        ];

        for (src_q, expected_zen_q) in cases {
            let probe = mock_probe(
                EncoderFamily::LibjpegTurbo,
                src_q,
                QualityScale::IjgQuality,
                Subsampling::S420,
            );
            let q = recommended_q(&probe);
            assert!(
                (q - expected_zen_q).abs() < 0.01,
                "turbo Q{src_q}: expected zen Q{expected_zen_q}, got Q{q}"
            );
        }
    }

    #[test]
    fn test_mozjpeg_recommendations() {
        let cases = [
            (50.0, 55.0),
            (65.0, 65.0),
            (75.0, 80.0),
            (80.0, 85.0),
            (85.0, 88.0),
            (90.0, 93.0),
        ];

        for (src_q, expected_zen_q) in cases {
            let probe = mock_probe(
                EncoderFamily::Mozjpeg,
                src_q,
                QualityScale::MozjpegQuality,
                Subsampling::S420,
            );
            let q = recommended_q(&probe);
            assert!(
                (q - expected_zen_q).abs() < 0.01,
                "mozjpeg Q{src_q}: expected zen Q{expected_zen_q}, got Q{q}"
            );
        }
    }

    #[test]
    fn test_jpegli_recommendations() {
        let cases = [
            (3.4, 50.0),
            (2.8, 60.0),
            (2.4, 75.0),
            (2.1, 80.0),
            (1.8, 85.0),
            (1.4, 88.0),
        ];

        for (src_dist, expected_zen_q) in cases {
            let probe = mock_probe(
                EncoderFamily::CjpegliYcbcr,
                src_dist,
                QualityScale::ButteraugliDistance,
                Subsampling::S444,
            );
            let q = recommended_q(&probe);
            assert!(
                (q - expected_zen_q).abs() < 0.01,
                "jpegli dist={src_dist}: expected zen Q{expected_zen_q}, got Q{q}"
            );
        }
    }

    #[test]
    fn test_interpolation_between_points() {
        // turbo Q57.5 = midpoint between Q50→65 and Q65→70
        let probe = mock_probe(
            EncoderFamily::LibjpegTurbo,
            57.5,
            QualityScale::IjgQuality,
            Subsampling::S420,
        );
        let q = recommended_q(&probe);
        assert!(
            (q - 67.5).abs() < 0.01,
            "turbo Q57.5: expected ~67.5, got {q}"
        );

        // jpegli BA 2.6 = midpoint between 2.8→60 and 2.4→75
        let probe = mock_probe(
            EncoderFamily::CjpegliYcbcr,
            2.6,
            QualityScale::ButteraugliDistance,
            Subsampling::S444,
        );
        let q = recommended_q(&probe);
        assert!(
            (q - 67.5).abs() < 0.01,
            "jpegli dist=2.6: expected ~67.5, got {q}"
        );
    }

    #[test]
    fn test_extrapolation_below_range() {
        // turbo Q40 — below table range, should extrapolate
        let probe = mock_probe(
            EncoderFamily::LibjpegTurbo,
            40.0,
            QualityScale::IjgQuality,
            Subsampling::S420,
        );
        let q = recommended_q(&probe);
        // Slope from Q50→65 to Q65→70 is (70-65)/(65-50) = 0.333/Q
        // At Q40: 65 + 0.333*(40-50) = 65 - 3.33 ≈ 61.67
        assert!(q > 55.0 && q < 70.0, "turbo Q40: got {q}, expected ~62");
    }

    #[test]
    fn test_extrapolation_above_range() {
        // turbo Q95 — above table range
        let probe = mock_probe(
            EncoderFamily::LibjpegTurbo,
            95.0,
            QualityScale::IjgQuality,
            Subsampling::S420,
        );
        let q = recommended_q(&probe);
        assert!(q > 95.0 && q <= 100.0, "turbo Q95: got {q}, expected ~97-100");
    }

    #[test]
    fn test_quality_ceiling_values() {
        // All ratios ≥1.5 should give Q90
        assert!((quality_ceiling(1.5) - 90.0).abs() < 0.01);
        assert!((quality_ceiling(2.0) - 90.0).abs() < 0.01);
        assert!((quality_ceiling(3.0) - 90.0).abs() < 0.01);
        assert!((quality_ceiling(4.0) - 90.0).abs() < 0.01);

        // No resize or upscale: higher ceiling
        assert!(quality_ceiling(1.0) > 90.0);
        assert!(quality_ceiling(0.5) > 93.0);
    }

    #[test]
    fn test_unknown_encoder_uses_conservative() {
        // Unknown encoder should use IJG table (highest Q = safest)
        let probe = mock_probe(
            EncoderFamily::Unknown,
            85.0,
            QualityScale::IjgQuality,
            Subsampling::S420,
        );
        let q_unknown = recommended_q(&probe);

        let probe_turbo = mock_probe(
            EncoderFamily::LibjpegTurbo,
            85.0,
            QualityScale::IjgQuality,
            Subsampling::S420,
        );
        let q_turbo = recommended_q(&probe_turbo);

        assert!(
            (q_unknown - q_turbo).abs() < 0.01,
            "unknown should match turbo: unknown={q_unknown}, turbo={q_turbo}"
        );
    }

    #[test]
    fn test_recommended_quality_returns_quality_enum() {
        let probe = mock_probe(
            EncoderFamily::LibjpegTurbo,
            85.0,
            QualityScale::IjgQuality,
            Subsampling::S420,
        );
        let q = probe.recommended_quality();
        assert!(matches!(q, Quality::ApproxJpegli(v) if (v - 90.0).abs() < 0.01));
    }

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
            assert_eq!(
                probe.recommended_subsampling(),
                expected,
                "subsampling {src:?} should map to {expected:?}"
            );
        }
    }

    #[test]
    fn test_tolerance_03_matches_default() {
        // recommended_quality_with_tolerance(0.3) should equal recommended_quality()
        let probe = mock_probe(
            EncoderFamily::LibjpegTurbo,
            85.0,
            QualityScale::IjgQuality,
            Subsampling::S420,
        );
        let q_default = recommended_q(&probe);
        let q_tol03 = recommended_q_with_tolerance(&probe, 0.3);
        assert!(
            (q_default - q_tol03).abs() < 0.01,
            "default={q_default}, tol=0.3={q_tol03}"
        );
    }

    #[test]
    fn test_tolerance_05_gives_lower_q() {
        // Higher tolerance → lower zen Q (more compression, more quality loss)
        for (encoder, src_q, scale) in [
            (EncoderFamily::LibjpegTurbo, 85.0, QualityScale::IjgQuality),
            (EncoderFamily::Mozjpeg, 85.0, QualityScale::MozjpegQuality),
            (EncoderFamily::CjpegliYcbcr, 1.8, QualityScale::ButteraugliDistance),
        ] {
            let probe = mock_probe(encoder, src_q, scale, Subsampling::S420);
            let q_tight = recommended_q_with_tolerance(&probe, 0.3);
            let q_loose = recommended_q_with_tolerance(&probe, 0.5);
            assert!(
                q_loose < q_tight,
                "{encoder:?}: tol=0.3 Q{q_tight} should be > tol=0.5 Q{q_loose}"
            );
        }
    }

    #[test]
    fn test_tolerance_zero_gives_higher_q() {
        // Zero tolerance → higher zen Q (trying to match quality exactly)
        let probe = mock_probe(
            EncoderFamily::LibjpegTurbo,
            85.0,
            QualityScale::IjgQuality,
            Subsampling::S420,
        );
        let q_zero = recommended_q_with_tolerance(&probe, 0.0);
        let q_default = recommended_q_with_tolerance(&probe, 0.3);
        assert!(
            q_zero > q_default,
            "tol=0.0 Q{q_zero} should be > tol=0.3 Q{q_default}"
        );
    }

    #[test]
    fn test_tolerance_large_gives_very_low_q() {
        // Large tolerance → very low zen Q
        let probe = mock_probe(
            EncoderFamily::LibjpegTurbo,
            85.0,
            QualityScale::IjgQuality,
            Subsampling::S420,
        );
        let q_tight = recommended_q_with_tolerance(&probe, 0.3);
        let q_very_loose = recommended_q_with_tolerance(&probe, 1.0);
        assert!(
            q_very_loose < q_tight - 10.0,
            "tol=1.0 Q{q_very_loose} should be much lower than tol=0.3 Q{q_tight}"
        );
    }

    #[test]
    fn test_turbo_shrink_calibration_points() {
        // Verify the shrink (tol=0.5) calibration points
        let cases = [
            (50.0, 55.0),
            (65.0, 65.0),
            (75.0, 75.0),
            (80.0, 75.0),
            (85.0, 80.0),
            (90.0, 88.0),
        ];

        for (src_q, expected_zen_q) in cases {
            let probe = mock_probe(
                EncoderFamily::LibjpegTurbo,
                src_q,
                QualityScale::IjgQuality,
                Subsampling::S420,
            );
            let q = recommended_q_with_tolerance(&probe, 0.5);
            assert!(
                (q - expected_zen_q).abs() < 0.01,
                "turbo Q{src_q} tol=0.5: expected zen Q{expected_zen_q}, got Q{q}"
            );
        }
    }

    #[test]
    fn test_tolerance_interpolation_midpoint() {
        // At tolerance 0.4 (midpoint of 0.3-0.5), result should be midpoint of tight/loose
        let probe = mock_probe(
            EncoderFamily::LibjpegTurbo,
            90.0,
            QualityScale::IjgQuality,
            Subsampling::S420,
        );
        let q_tight = recommended_q_with_tolerance(&probe, 0.3); // 95.0
        let q_loose = recommended_q_with_tolerance(&probe, 0.5); // 88.0
        let q_mid = recommended_q_with_tolerance(&probe, 0.4);
        let expected = (q_tight + q_loose) / 2.0; // 91.5
        assert!(
            (q_mid - expected).abs() < 0.01,
            "turbo Q90 tol=0.4: expected {expected}, got {q_mid}"
        );
    }

    #[test]
    fn test_tolerance_clamped_to_valid_range() {
        // Even with extreme tolerance, Q should stay in [1, 100]
        let probe = mock_probe(
            EncoderFamily::LibjpegTurbo,
            50.0,
            QualityScale::IjgQuality,
            Subsampling::S420,
        );
        let q = recommended_q_with_tolerance(&probe, 5.0);
        assert!(q >= 1.0 && q <= 100.0, "extreme tolerance: Q={q}");

        let q = recommended_q_with_tolerance(&probe, 0.0);
        assert!(q >= 1.0 && q <= 100.0, "zero tolerance: Q={q}");
    }

    #[test]
    fn test_monotonic_recommendations() {
        // Higher source quality should always give higher (or equal) zen Q
        for encoder in [EncoderFamily::LibjpegTurbo, EncoderFamily::Mozjpeg] {
            let mut prev_q = 0.0f32;
            for src_q in [50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0] {
                let probe = mock_probe(
                    encoder,
                    src_q,
                    match encoder {
                        EncoderFamily::Mozjpeg => QualityScale::MozjpegQuality,
                        _ => QualityScale::IjgQuality,
                    },
                    Subsampling::S420,
                );
                let q = recommended_q(&probe);
                assert!(
                    q >= prev_q,
                    "{encoder:?} Q{src_q}: zen Q{q} < prev Q{prev_q} — not monotonic"
                );
                prev_q = q;
            }
        }

        // jpegli: lower distance = higher quality = higher zen Q
        let mut prev_q = 0.0f32;
        for dist in [4.0, 3.4, 2.8, 2.4, 2.1, 1.8, 1.4, 1.0] {
            let probe = mock_probe(
                EncoderFamily::CjpegliYcbcr,
                dist,
                QualityScale::ButteraugliDistance,
                Subsampling::S444,
            );
            let q = recommended_q(&probe);
            assert!(
                q >= prev_q,
                "jpegli dist={dist}: zen Q{q} < prev Q{prev_q} — not monotonic"
            );
            prev_q = q;
        }
    }
}
