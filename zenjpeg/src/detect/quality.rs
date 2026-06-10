//! Quality estimation per encoder family.
//!
//! Three estimation paths:
//! - **IJG family**: Generate all 100 reference tables, find exact match.
//! - **mozjpeg**: Scale Robidoux base tables, find exact match.
//! - **jpegli**: Call `quant_vals_to_distance()` to get butteraugli distance.

use crate::encode::tables::robidoux::{ROBIDOUX_LUMINANCE, quality_to_scale_factor, scale_table};
use crate::foundation::consts::DCT_BLOCK_SIZE;

use super::fingerprint::{EncoderFamily, generate_ijg_table};
use super::scanner::ScanResult;

/// Estimated quality for a JPEG file.
#[derive(Debug, Clone)]
pub struct QualityEstimate {
    /// Quality value (meaning depends on `scale`).
    pub value: f32,
    /// What scale the quality value is on.
    pub scale: QualityScale,
    /// How confident we are in this estimate.
    pub confidence: Confidence,
}

/// The scale used for quality values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum QualityScale {
    /// IJG quality 1-100 (standard JPEG scale).
    IjgQuality,
    /// mozjpeg quality 1-100 (Robidoux preset).
    MozjpegQuality,
    /// Butteraugli distance (lower = better, 1.0 ≈ visually lossless).
    ButteraugliDistance,
    /// Windows GDI+ / System.Drawing quality 1-100.
    ///
    /// Same formula family as [`QualityScale::IjgQuality`] but offset:
    /// Windows quality `q` produces the IJG table at index `q - 1`
    /// (except multiples of 25, which produce the table at `q`).
    WindowsQuality,
}

/// Confidence level for quality estimates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Confidence {
    /// DQT matches a reference table perfectly (SSE = 0).
    Exact,
    /// Nearest match or interpolated — no perfect match found.
    Approximate,
}

/// Estimate quality for the identified encoder family.
pub(crate) fn estimate_quality(scan: &ScanResult, encoder: &EncoderFamily) -> QualityEstimate {
    match encoder {
        EncoderFamily::LibjpegTurbo | EncoderFamily::ImageMagick | EncoderFamily::IjgFamily => {
            estimate_ijg_quality(scan)
        }

        EncoderFamily::Mozjpeg => estimate_mozjpeg_quality(scan),

        EncoderFamily::WindowsImaging => estimate_windows_quality(scan),

        EncoderFamily::CjpegliYcbcr | EncoderFamily::CjpegliXyb => estimate_jpegli_quality(scan),

        // Photoshop uses non-IJG custom tables — match against IJG as a reference
        EncoderFamily::Photoshop | EncoderFamily::Unknown => estimate_ijg_quality_approximate(scan),
    }
}

/// Estimate IJG quality by matching against all 100 reference tables.
fn estimate_ijg_quality(scan: &ScanResult) -> QualityEstimate {
    let luma = match &scan.dqt_tables[0] {
        Some(t) => &t.values,
        None => {
            return QualityEstimate {
                value: 75.0,
                scale: QualityScale::IjgQuality,
                confidence: Confidence::Approximate,
            };
        }
    };

    let chroma = scan.dqt_tables[1].as_ref().map(|t| &t.values);

    // Try exact match first
    for q in 1..=100u8 {
        let ref_luma = generate_ijg_table(q, false);
        if *luma == ref_luma {
            // If we have a chroma table, verify it matches too
            if let Some(chroma_vals) = chroma {
                let ref_chroma = generate_ijg_table(q, true);
                if *chroma_vals == ref_chroma {
                    return QualityEstimate {
                        value: q as f32,
                        scale: QualityScale::IjgQuality,
                        confidence: Confidence::Exact,
                    };
                }
            } else {
                // Grayscale — luma match is sufficient
                return QualityEstimate {
                    value: q as f32,
                    scale: QualityScale::IjgQuality,
                    confidence: Confidence::Exact,
                };
            }
        }
    }

    // No exact match — find closest by SSE on luma table
    find_closest_ijg_quality(luma)
}

/// Find closest IJG quality when no exact match exists.
fn find_closest_ijg_quality(luma: &[u16; 64]) -> QualityEstimate {
    let mut best_q = 75u8;
    let mut best_sse = u64::MAX;

    for q in 1..=100u8 {
        let ref_table = generate_ijg_table(q, false);
        let sse = compute_sse(luma, &ref_table);
        if sse < best_sse {
            best_sse = sse;
            best_q = q;
        }
    }

    QualityEstimate {
        value: best_q as f32,
        scale: QualityScale::IjgQuality,
        confidence: Confidence::Approximate,
    }
}

/// For unknown encoders, try to find the closest IJG quality as a reference point.
fn estimate_ijg_quality_approximate(scan: &ScanResult) -> QualityEstimate {
    let luma = match &scan.dqt_tables[0] {
        Some(t) => &t.values,
        None => {
            return QualityEstimate {
                value: 75.0,
                scale: QualityScale::IjgQuality,
                confidence: Confidence::Approximate,
            };
        }
    };

    find_closest_ijg_quality(luma)
}

/// Estimate Windows (GDI+/WIC) quality from the IJG table index.
///
/// Windows emits byte-exact IJG tables at an internal index `k`. The
/// GDI+ integer quality `q` maps to `k = q - 1`, except `k = q` when
/// `q` is a multiple of 25 (the qualities where `q/100` is exactly
/// representable in binary floating point — verified empirically on a
/// q=1..=100 sweep, zero exceptions). Inverting:
///
/// - `k ∈ {25, 50, 75}`: both `q = k` and `q = k + 1` produce this
///   table; report the round-number `k` (overwhelmingly the more
///   common user input).
/// - `k ∈ {24, 49, 74, 99}`: unreachable from any GDI+ integer
///   quality (only WIC's float `ImageQuality` lands here); report `k`
///   with [`Confidence::Approximate`].
/// - otherwise: report `k + 1`.
fn estimate_windows_quality(scan: &ScanResult) -> QualityEstimate {
    let luma = match &scan.dqt_tables[0] {
        Some(t) => &t.values,
        None => {
            return QualityEstimate {
                value: 75.0,
                scale: QualityScale::WindowsQuality,
                confidence: Confidence::Approximate,
            };
        }
    };
    let chroma = scan.dqt_tables[1].as_ref().map(|t| &t.values);

    // Exact match against IJG tables at every index
    for k in 1..=100u8 {
        if *luma != generate_ijg_table(k, false) {
            continue;
        }
        if let Some(chroma_vals) = chroma
            && *chroma_vals != generate_ijg_table(k, true)
        {
            continue;
        }
        let (value, confidence) = windows_quality_from_ijg_index(k);
        return QualityEstimate {
            value,
            scale: QualityScale::WindowsQuality,
            confidence,
        };
    }

    // No exact match (shouldn't happen — the fingerprint requires an
    // IJG table match before classifying WindowsImaging): closest by SSE.
    let mut best_k = 75u8;
    let mut best_sse = u64::MAX;
    for k in 1..=100u8 {
        let sse = compute_sse(luma, &generate_ijg_table(k, false));
        if sse < best_sse {
            best_sse = sse;
            best_k = k;
        }
    }
    QualityEstimate {
        value: windows_quality_from_ijg_index(best_k).0,
        scale: QualityScale::WindowsQuality,
        confidence: Confidence::Approximate,
    }
}

/// Map a matched IJG table index `k` back to the Windows user quality.
fn windows_quality_from_ijg_index(k: u8) -> (f32, Confidence) {
    match k {
        25 | 50 | 75 | 100 => (k as f32, Confidence::Exact),
        24 | 49 | 74 | 99 => (k as f32, Confidence::Approximate),
        _ => ((k + 1) as f32, Confidence::Exact),
    }
}

/// Estimate mozjpeg quality by matching Robidoux tables.
fn estimate_mozjpeg_quality(scan: &ScanResult) -> QualityEstimate {
    let table = match &scan.dqt_tables[0] {
        Some(t) => &t.values,
        None => {
            return QualityEstimate {
                value: 75.0,
                scale: QualityScale::MozjpegQuality,
                confidence: Confidence::Approximate,
            };
        }
    };

    // Try exact match against Robidoux tables at each quality
    for q in 1..=100u8 {
        let scale = quality_to_scale_factor(q);
        let ref_table = scale_table(&ROBIDOUX_LUMINANCE, scale, true);
        if *table == ref_table {
            return QualityEstimate {
                value: q as f32,
                scale: QualityScale::MozjpegQuality,
                confidence: Confidence::Exact,
            };
        }
    }

    // No exact match — find closest
    let mut best_q = 75u8;
    let mut best_sse = u64::MAX;

    for q in 1..=100u8 {
        let scale = quality_to_scale_factor(q);
        let ref_table = scale_table(&ROBIDOUX_LUMINANCE, scale, true);
        let sse = compute_sse(table, &ref_table);
        if sse < best_sse {
            best_sse = sse;
            best_q = q;
        }
    }

    QualityEstimate {
        value: best_q as f32,
        scale: QualityScale::MozjpegQuality,
        confidence: Confidence::Approximate,
    }
}

/// Estimate jpegli quality by converting DQT tables back to butteraugli distance.
fn estimate_jpegli_quality(scan: &ScanResult) -> QualityEstimate {
    // Need all 3 DQT tables for jpegli
    let y = match &scan.dqt_tables[0] {
        Some(t) => t,
        None => {
            return QualityEstimate {
                value: 1.0,
                scale: QualityScale::ButteraugliDistance,
                confidence: Confidence::Approximate,
            };
        }
    };
    let cb = match &scan.dqt_tables[1] {
        Some(t) => t,
        None => {
            return QualityEstimate {
                value: 1.0,
                scale: QualityScale::ButteraugliDistance,
                confidence: Confidence::Approximate,
            };
        }
    };
    let cr = match &scan.dqt_tables[2] {
        Some(t) => t,
        None => {
            return QualityEstimate {
                value: 1.0,
                scale: QualityScale::ButteraugliDistance,
                confidence: Confidence::Approximate,
            };
        }
    };

    // Convert natural-order values back to zigzag for QuantTable
    // (quant_vals_to_distance expects QuantTable which stores in zigzag order)
    let y_qt = natural_to_quant_table(&y.values, y.precision);
    let cb_qt = natural_to_quant_table(&cb.values, cb.precision);
    let cr_qt = natural_to_quant_table(&cr.values, cr.precision);

    // Decoder-side: we don't know whether the JPEG was encoded with XYB or
    // YCbCr. Default to YCbCr (the dominant case). A future improvement
    // would try both inversions and pick the one with tighter dist_min..dist_max.
    let distance =
        crate::quant::quant_vals_to_distance(&y_qt, &cb_qt, &cr_qt, /* use_xyb = */ false);

    QualityEstimate {
        value: distance,
        scale: QualityScale::ButteraugliDistance,
        confidence: Confidence::Exact,
    }
}

/// Convert natural-order values to a QuantTable (which stores in zigzag order).
fn natural_to_quant_table(natural: &[u16; 64], precision: u8) -> crate::types::QuantTable {
    use crate::foundation::consts::JPEG_ZIGZAG_ORDER;

    let mut zigzag = [0u16; DCT_BLOCK_SIZE];
    for natural_idx in 0..64 {
        let zigzag_idx = JPEG_ZIGZAG_ORDER[natural_idx] as usize;
        zigzag[zigzag_idx] = natural[natural_idx];
    }

    crate::types::QuantTable {
        values: zigzag,
        precision,
    }
}

/// Compute sum of squared errors between two tables.
fn compute_sse(a: &[u16; 64], b: &[u16; 64]) -> u64 {
    let mut sse = 0u64;
    for i in 0..64 {
        let diff = a[i] as i64 - b[i] as i64;
        sse += (diff * diff) as u64;
    }
    sse
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ijg_quality_roundtrip() {
        // Generate an IJG table at Q75, then estimate quality — should be exact Q75
        for q in [10, 25, 50, 75, 85, 90, 95, 100] {
            let luma = generate_ijg_table(q, false);
            let chroma = generate_ijg_table(q, true);

            let scan = mock_scan_two_tables(&luma, &chroma);
            let estimate = estimate_ijg_quality(&scan);

            assert_eq!(
                estimate.value, q as f32,
                "IJG Q{q} roundtrip failed: got {}",
                estimate.value
            );
            assert_eq!(estimate.confidence, Confidence::Exact);
        }
    }

    #[test]
    fn test_mozjpeg_quality_roundtrip() {
        // Generate Robidoux tables at various qualities, verify estimation
        for q in [10, 25, 50, 75, 85, 90, 95, 100] {
            let scale = quality_to_scale_factor(q);
            let table = scale_table(&ROBIDOUX_LUMINANCE, scale, true);

            let scan = mock_scan_identical_tables(&table);
            let estimate = estimate_mozjpeg_quality(&scan);

            assert_eq!(
                estimate.value, q as f32,
                "Mozjpeg Q{q} roundtrip failed: got {}",
                estimate.value
            );
            assert_eq!(estimate.confidence, Confidence::Exact);
        }
    }

    #[test]
    fn test_windows_quality_full_index_sweep() {
        // Every IJG table index k=1..=100 maps to a Windows quality:
        // k+1 normally, k for multiples of 25 (Exact), k Approximate
        // for the GDI+-unreachable indices {24, 49, 74, 99}.
        for k in 1..=100u8 {
            let luma = generate_ijg_table(k, false);
            let chroma = generate_ijg_table(k, true);
            let scan = mock_scan_two_tables(&luma, &chroma);
            let est = estimate_windows_quality(&scan);

            let (want_value, want_conf) = match k {
                25 | 50 | 75 | 100 => (k as f32, Confidence::Exact),
                24 | 49 | 74 | 99 => (k as f32, Confidence::Approximate),
                _ => ((k + 1) as f32, Confidence::Exact),
            };
            assert_eq!(est.scale, QualityScale::WindowsQuality, "k={k}");
            assert_eq!(
                est.value, want_value,
                "k={k}: expected Windows quality {want_value}, got {}",
                est.value
            );
            assert_eq!(est.confidence, want_conf, "k={k}");
        }
    }

    #[test]
    fn test_windows_quality_gdiplus_forward_map_roundtrip() {
        // Forward map (verified on the q=1..=100 z.zr.io sweep,
        // 2026-06-09): GDI+ quality q emits the IJG table at
        // k = q - 1, except k = q when q % 25 == 0. Round-tripping
        // through the estimator must recover q exactly for every
        // quality except the table-identical collisions q ∈ {26, 51,
        // 76} (reported as 25/50/75) and q ∈ {1, 2} (both produce the
        // all-255 k=1 table, reported as 2).
        for q in 1..=100u32 {
            let k = if q % 25 == 0 { q } else { q - 1 }.max(1) as u8;
            let luma = generate_ijg_table(k, false);
            let chroma = generate_ijg_table(k, true);
            let scan = mock_scan_two_tables(&luma, &chroma);
            let est = estimate_windows_quality(&scan);

            let want = match q {
                1 => 2.0,
                26 => 25.0,
                51 => 50.0,
                76 => 75.0,
                other => other as f32,
            };
            assert_eq!(
                est.value, want,
                "GDI+ q={q} (k={k}): expected estimate {want}, got {}",
                est.value
            );
        }
    }

    #[test]
    fn test_approximate_for_non_ijg_table() {
        // Create a table that doesn't match any IJG quality exactly
        let mut table = generate_ijg_table(75, false);
        table[0] += 1; // Modify DC value

        let scan = mock_scan_two_tables(&table, &generate_ijg_table(75, true));
        let estimate = estimate_ijg_quality(&scan);

        assert_eq!(estimate.confidence, Confidence::Approximate);
        // Should still be close to Q75
        assert!((estimate.value - 75.0).abs() < 5.0);
    }

    /// Helper: create a ScanResult with two DQT tables.
    fn mock_scan_two_tables(luma: &[u16; 64], chroma: &[u16; 64]) -> ScanResult {
        use super::super::scanner::{DqtEntry, ScanResult};

        ScanResult {
            dqt_tables: [
                Some(DqtEntry {
                    values: *luma,
                    precision: 0,
                }),
                Some(DqtEntry {
                    values: *chroma,
                    precision: 0,
                }),
                None,
                None,
            ],
            sof: None,
            total_ac_symbols: 0,
            dht_count: 0,
            has_jfif: false,
            jfif_density: None,
            has_icc_profile: false,
            has_adobe: false,
            has_photoshop_iptc: false,
            sos_count: 0,
        }
    }

    /// Helper: create a ScanResult with identical luma/chroma tables (mozjpeg).
    fn mock_scan_identical_tables(table: &[u16; 64]) -> ScanResult {
        mock_scan_two_tables(table, table)
    }
}
