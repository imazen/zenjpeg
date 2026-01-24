//! Quality conversion between jpegli and other JPEG encoders.
//!
//! This module provides utilities to convert quality settings from other JPEG encoders
//! (like mozjpeg) to equivalent jpegli quality settings that produce similar visual quality.
//!
//! # Example
//!
//! ```ignore
//! use zenjpeg::{Encoder, QualityConversion, QualityComparisonMetric, Subsampling};
//!
//! // Convert mozjpeg Q85 to equivalent jpegli quality
//! let conversion = QualityConversion::mozjpeg_equivalent(
//!     85,
//!     Subsampling::S444,
//!     QualityComparisonMetric::Dssim,
//! );
//!
//! let encoder = Encoder::new()
//!     .width(800)
//!     .height(600)
//!     .equivalent_quality(conversion);
//! ```

#![allow(dead_code)]

use crate::quant::Quality;
use crate::types::Subsampling;

/// Metric used for quality comparison between encoders.
///
/// Different metrics weight perceptual quality differently. DSSIM is recommended
/// for general use as it correlates well with perceived quality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum QualityComparisonMetric {
    /// DSSIM (Structural Dissimilarity) - lower is better, 0 = identical
    /// Recommended for general use.
    #[default]
    Dssim,
    /// SSIMULACRA2 - higher is better, 100 = identical
    /// More sensitive to fine detail.
    Ssimulacra2,
    /// Butteraugli - lower is better, 0 = identical
    /// Perceptually tuned, same metric used by jpegli internally.
    Butteraugli,
}

/// Quality conversion configuration for matching other encoders.
///
/// Use this to specify a quality value from another encoder (like mozjpeg) and
/// get an equivalent jpegli quality that produces similar visual results.
#[derive(Debug, Clone, Copy)]
pub struct QualityConversion {
    /// The source quality value (typically 1-100 for mozjpeg)
    pub source_quality: u8,
    /// The subsampling mode to use for lookup
    pub subsampling: Subsampling,
    /// The metric used for the approximation tables
    pub metric: QualityComparisonMetric,
    /// Whether this is an exact table lookup or interpolated
    pub is_interpolated: bool,
}

impl QualityConversion {
    /// Creates a quality conversion that attempts to match mozjpeg quality.
    ///
    /// Returns `None` if the quality/subsampling/metric combination is not in
    /// the approximation tables. For guaranteed conversion, use `mozjpeg_equivalent`.
    ///
    /// # Arguments
    ///
    /// * `quality` - mozjpeg quality value (1-100)
    /// * `subsampling` - Target subsampling mode (only S444 and S420 have tables)
    /// * `metric` - Which quality metric to use for comparison
    ///
    /// # Returns
    ///
    /// `Some(QualityConversion)` if exact table lookup is available, `None` otherwise.
    #[must_use]
    pub fn try_mozjpeg_equivalent(
        quality: u8,
        subsampling: Subsampling,
        metric: QualityComparisonMetric,
    ) -> Option<Self> {
        // Q100 is always passthrough
        if quality >= 100 {
            return Some(Self {
                source_quality: 100,
                subsampling,
                metric,
                is_interpolated: false,
            });
        }

        // Check if we have an exact table entry
        let table = get_mapping_table(subsampling, metric)?;

        // Find exact match in table
        if table.iter().any(|&(moz_q, _)| moz_q == quality) {
            Some(Self {
                source_quality: quality,
                subsampling,
                metric,
                is_interpolated: false,
            })
        } else {
            None
        }
    }

    /// Creates a quality conversion that matches mozjpeg quality with fallbacks.
    ///
    /// This always succeeds by:
    /// - Using exact table lookup when available
    /// - Interpolating between nearby table entries when necessary
    /// - Mapping unsupported subsampling modes to the closest supported one
    /// - Q100 always passes through unchanged
    ///
    /// # Arguments
    ///
    /// * `quality` - mozjpeg quality value (1-100)
    /// * `subsampling` - Target subsampling mode
    /// * `metric` - Which quality metric to use for comparison
    #[must_use]
    pub fn mozjpeg_equivalent(
        quality: u8,
        subsampling: Subsampling,
        metric: QualityComparisonMetric,
    ) -> Self {
        // Q100 is always passthrough
        if quality >= 100 {
            return Self {
                source_quality: 100,
                subsampling,
                metric,
                is_interpolated: false,
            };
        }

        // Map subsampling to one we have tables for
        let mapped_subsampling = match subsampling {
            Subsampling::S444 => Subsampling::S444,
            Subsampling::S422 | Subsampling::S420 | Subsampling::S440 => Subsampling::S420,
        };

        // Check if exact entry exists
        let table = get_mapping_table(mapped_subsampling, metric);
        let is_interpolated = match table {
            Some(t) => !t.iter().any(|&(moz_q, _)| moz_q == quality),
            None => true,
        };

        Self {
            source_quality: quality,
            subsampling: mapped_subsampling,
            metric,
            is_interpolated,
        }
    }

    /// Converts to jpegli Quality.
    ///
    /// Uses the approximation tables or interpolation to find the equivalent
    /// jpegli quality value.
    #[must_use]
    pub fn to_jpegli_quality(self) -> Quality {
        // Q100 passthrough
        if self.source_quality >= 100 {
            return Quality::ApproxJpegli(100.0);
        }

        // Get the mapping table
        let table = match get_mapping_table(self.subsampling, self.metric) {
            Some(t) => t,
            None => {
                // No table available, use identity mapping
                return Quality::ApproxJpegli(self.source_quality as f32);
            }
        };

        // Try exact lookup first
        for &(moz_q, jpegli_q) in table {
            if moz_q == self.source_quality {
                return Quality::ApproxJpegli(jpegli_q as f32);
            }
        }

        // Interpolate between nearest entries
        interpolate_quality(self.source_quality, table)
    }
}

/// Interpolates between table entries to find the jpegli quality for a given mozjpeg quality.
fn interpolate_quality(moz_q: u8, table: &[(u8, u8)]) -> Quality {
    // Find the two nearest entries
    let mut lower: Option<(u8, u8)> = None;
    let mut upper: Option<(u8, u8)> = None;

    for &(tbl_moz_q, tbl_jpegli_q) in table {
        if tbl_moz_q <= moz_q {
            match lower {
                None => lower = Some((tbl_moz_q, tbl_jpegli_q)),
                Some((prev_q, _)) if tbl_moz_q > prev_q => lower = Some((tbl_moz_q, tbl_jpegli_q)),
                _ => {}
            }
        }
        if tbl_moz_q >= moz_q {
            match upper {
                None => upper = Some((tbl_moz_q, tbl_jpegli_q)),
                Some((prev_q, _)) if tbl_moz_q < prev_q => upper = Some((tbl_moz_q, tbl_jpegli_q)),
                _ => {}
            }
        }
    }

    match (lower, upper) {
        (Some((l_moz, l_jpegli)), Some((u_moz, u_jpegli))) if l_moz != u_moz => {
            // Linear interpolation
            let t = (moz_q - l_moz) as f32 / (u_moz - l_moz) as f32;
            let jpegli_q = l_jpegli as f32 + t * (u_jpegli as f32 - l_jpegli as f32);
            Quality::ApproxJpegli(jpegli_q)
        }
        (Some((_, jpegli_q)), _) => Quality::ApproxJpegli(jpegli_q as f32),
        (_, Some((_, jpegli_q))) => Quality::ApproxJpegli(jpegli_q as f32),
        (None, None) => {
            // Fallback to identity
            Quality::ApproxJpegli(moz_q as f32)
        }
    }
}

/// Gets the mapping table for a given subsampling and metric combination.
fn get_mapping_table(
    subsampling: Subsampling,
    metric: QualityComparisonMetric,
) -> Option<&'static [(u8, u8)]> {
    match (subsampling, metric) {
        (Subsampling::S444, QualityComparisonMetric::Dssim) => Some(&MOZJPEG_TO_JPEGLI_444_DSSIM),
        (Subsampling::S420, QualityComparisonMetric::Dssim) => Some(&MOZJPEG_TO_JPEGLI_420_DSSIM),
        (Subsampling::S444, QualityComparisonMetric::Ssimulacra2) => {
            Some(&MOZJPEG_TO_JPEGLI_444_SSIMULACRA2)
        }
        (Subsampling::S420, QualityComparisonMetric::Ssimulacra2) => {
            Some(&MOZJPEG_TO_JPEGLI_420_SSIMULACRA2)
        }
        (Subsampling::S444, QualityComparisonMetric::Butteraugli) => {
            Some(&MOZJPEG_TO_JPEGLI_444_BUTTERAUGLI)
        }
        (Subsampling::S420, QualityComparisonMetric::Butteraugli) => {
            Some(&MOZJPEG_TO_JPEGLI_420_BUTTERAUGLI)
        }
        _ => None,
    }
}

// =============================================================================
// Mapping Tables
// =============================================================================
//
// These tables map mozjpeg quality values to equivalent jpegli quality values
// that produce similar visual quality as measured by various metrics.
//
// Format: (mozjpeg_quality, jpegli_quality)
//
// Tables were generated by encoding test images at various quality levels with
// both encoders and finding the jpegli quality that matches mozjpeg's DSSIM/etc.
//
// Note: jpegli is generally more efficient than mozjpeg, so jpegli quality values
// are typically lower than the corresponding mozjpeg values for the same visual quality.

/// mozjpeg to jpegli quality mapping for 4:4:4 subsampling using DSSIM metric.
/// These values were derived from corpus testing on CID22-512 and Kodak datasets.
static MOZJPEG_TO_JPEGLI_444_DSSIM: [(u8, u8); 10] = [
    (30, 28),
    (40, 37),
    (50, 47),
    (60, 55),
    (70, 65),
    (75, 71),
    (80, 77),
    (85, 83),
    (90, 89),
    (95, 94),
];

/// mozjpeg to jpegli quality mapping for 4:2:0 subsampling using DSSIM metric.
static MOZJPEG_TO_JPEGLI_420_DSSIM: [(u8, u8); 10] = [
    (30, 27),
    (40, 36),
    (50, 45),
    (60, 54),
    (70, 64),
    (75, 70),
    (80, 76),
    (85, 82),
    (90, 88),
    (95, 94),
];

/// mozjpeg to jpegli quality mapping for 4:4:4 using SSIMULACRA2 metric.
static MOZJPEG_TO_JPEGLI_444_SSIMULACRA2: [(u8, u8); 10] = [
    (30, 29),
    (40, 38),
    (50, 48),
    (60, 56),
    (70, 66),
    (75, 72),
    (80, 78),
    (85, 84),
    (90, 89),
    (95, 94),
];

/// mozjpeg to jpegli quality mapping for 4:2:0 using SSIMULACRA2 metric.
static MOZJPEG_TO_JPEGLI_420_SSIMULACRA2: [(u8, u8); 10] = [
    (30, 28),
    (40, 37),
    (50, 46),
    (60, 55),
    (70, 65),
    (75, 71),
    (80, 77),
    (85, 83),
    (90, 89),
    (95, 94),
];

/// mozjpeg to jpegli quality mapping for 4:4:4 using Butteraugli metric.
static MOZJPEG_TO_JPEGLI_444_BUTTERAUGLI: [(u8, u8); 10] = [
    (30, 30),
    (40, 39),
    (50, 49),
    (60, 57),
    (70, 67),
    (75, 73),
    (80, 79),
    (85, 85),
    (90, 90),
    (95, 95),
];

/// mozjpeg to jpegli quality mapping for 4:2:0 using Butteraugli metric.
static MOZJPEG_TO_JPEGLI_420_BUTTERAUGLI: [(u8, u8); 10] = [
    (30, 29),
    (40, 38),
    (50, 48),
    (60, 56),
    (70, 66),
    (75, 72),
    (80, 78),
    (85, 84),
    (90, 90),
    (95, 95),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q100_passthrough() {
        let conv = QualityConversion::mozjpeg_equivalent(
            100,
            Subsampling::S444,
            QualityComparisonMetric::Dssim,
        );
        let q = conv.to_jpegli_quality();
        assert_eq!(q.to_internal(), 100.0);
    }

    #[test]
    fn test_try_mozjpeg_equivalent_exact() {
        // Q90 is in the table
        let conv = QualityConversion::try_mozjpeg_equivalent(
            90,
            Subsampling::S444,
            QualityComparisonMetric::Dssim,
        );
        assert!(conv.is_some());
        let conv = conv.unwrap();
        assert!(!conv.is_interpolated);
        let q = conv.to_jpegli_quality();
        assert_eq!(q.to_internal(), 89.0);
    }

    #[test]
    fn test_try_mozjpeg_equivalent_missing() {
        // Q87 is not in the table
        let conv = QualityConversion::try_mozjpeg_equivalent(
            87,
            Subsampling::S444,
            QualityComparisonMetric::Dssim,
        );
        assert!(conv.is_none());
    }

    #[test]
    fn test_mozjpeg_equivalent_interpolation() {
        // Q87 is between Q85 (83) and Q90 (89)
        let conv = QualityConversion::mozjpeg_equivalent(
            87,
            Subsampling::S444,
            QualityComparisonMetric::Dssim,
        );
        assert!(conv.is_interpolated);
        let q = conv.to_jpegli_quality();
        // Should interpolate: 83 + (87-85)/(90-85) * (89-83) = 83 + 0.4 * 6 = 85.4
        let expected = 85.4;
        assert!(
            (q.to_internal() - expected).abs() < 0.5,
            "Expected ~{}, got {}",
            expected,
            q.to_internal()
        );
    }

    #[test]
    fn test_subsampling_fallback() {
        // S422 maps to S420
        let conv = QualityConversion::mozjpeg_equivalent(
            90,
            Subsampling::S422,
            QualityComparisonMetric::Dssim,
        );
        assert_eq!(conv.subsampling, Subsampling::S420);
    }

    #[test]
    fn test_all_metrics() {
        for metric in [
            QualityComparisonMetric::Dssim,
            QualityComparisonMetric::Ssimulacra2,
            QualityComparisonMetric::Butteraugli,
        ] {
            let conv = QualityConversion::mozjpeg_equivalent(85, Subsampling::S444, metric);
            let q = conv.to_jpegli_quality();
            assert!(q.to_internal() >= 80.0 && q.to_internal() <= 90.0);
        }
    }

    #[test]
    fn test_low_quality() {
        // Test extrapolation below table range
        let conv = QualityConversion::mozjpeg_equivalent(
            20,
            Subsampling::S444,
            QualityComparisonMetric::Dssim,
        );
        let q = conv.to_jpegli_quality();
        // Should clamp to lowest table entry (28)
        assert!(q.to_internal() <= 30.0);
    }
}
