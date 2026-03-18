//! Encoder identification from structural JPEG signals.
//!
//! Implements the decision tree from `detection_heuristics.md`:
//! DQT count → component IDs → table matching → Huffman analysis.

use crate::foundation::consts::MARKER_SOF0;
use crate::quant::{STD_CHROMINANCE_QUANT, STD_LUMINANCE_QUANT};

use super::scanner::ScanResult;

/// Identified encoder family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum EncoderFamily {
    /// IJG tables + standard Huffman (162 AC symbols per table).
    /// Could be libjpeg-turbo or Pillow (structurally identical).
    LibjpegTurbo,

    /// IJG tables + optimized Huffman. Typically ImageMagick.
    /// Switches to 4:4:4 at Q>=90.
    ImageMagick,

    /// IJG tables but cannot distinguish turbo/Pillow/ImageMagick
    /// (e.g., insufficient Huffman info).
    IjgFamily,

    /// Progressive + DQT\[0\]==DQT\[1\] + optimized Huffman.
    /// Uses Robidoux tables by default.
    Mozjpeg,

    /// 3 DQT tables + no JFIF + component IDs 1,2,3.
    CjpegliYcbcr,

    /// 3 DQT tables + APP2 ICC + component IDs R,G,B (82,71,66).
    CjpegliXyb,

    /// Adobe Photoshop: APP14 Adobe marker + APP13 Photoshop 3.0 IPTC,
    /// non-IJG quantization tables (custom per quality preset 0-12).
    Photoshop,

    /// Non-IJG tables, cameras, or other tools.
    Unknown,
}

/// Standard AC symbol count per table (2 tables × 162 symbols = 324 total).
const STANDARD_AC_SYMBOLS_TOTAL: u16 = 324;

/// Identify which encoder family produced this JPEG.
pub(crate) fn identify_encoder(scan: &ScanResult) -> EncoderFamily {
    let sof = match &scan.sof {
        Some(s) => s,
        None => return EncoderFamily::Unknown,
    };

    let dqt_count = count_dqt_tables(scan);

    match dqt_count {
        3 => identify_three_table(scan, sof),
        2 => identify_two_table(scan, sof),
        1 => identify_single_table(scan, sof),
        _ => EncoderFamily::Unknown,
    }
}

/// Count how many DQT tables are defined.
fn count_dqt_tables(scan: &ScanResult) -> u8 {
    scan.dqt_tables.iter().filter(|t| t.is_some()).count() as u8
}

/// 3 DQT tables → jpegli (YCbCr or XYB).
fn identify_three_table(scan: &ScanResult, sof: &super::scanner::SofInfo) -> EncoderFamily {
    // Check component IDs for XYB: R=82, G=71, B=66
    if sof.num_components == 3 {
        let ids: Vec<u8> = sof.components.iter().map(|c| c.0).collect();
        if ids == [82, 71, 66] && scan.has_icc_profile {
            return EncoderFamily::CjpegliXyb;
        }
        if !scan.has_jfif {
            return EncoderFamily::CjpegliYcbcr;
        }
    }

    // 3 tables but doesn't match jpegli patterns
    EncoderFamily::Unknown
}

/// 2 DQT tables → IJG family or mozjpeg.
fn identify_two_table(scan: &ScanResult, sof: &super::scanner::SofInfo) -> EncoderFamily {
    let table0 = match &scan.dqt_tables[0] {
        Some(t) => t,
        None => return EncoderFamily::Unknown,
    };
    let table1 = match &scan.dqt_tables[1] {
        Some(t) => t,
        None => return EncoderFamily::Unknown,
    };

    // Check for mozjpeg: DQT[0] == DQT[1] (identical luma/chroma)
    if table0.values == table1.values {
        // Confirm progressive + reasonable scan count
        let is_progressive = sof.marker != MARKER_SOF0;
        if is_progressive && scan.sos_count >= 4 {
            return EncoderFamily::Mozjpeg;
        }
        // Identical tables but not progressive — unusual, but could still be mozjpeg
        // in baseline mode (rare configuration)
        if is_progressive {
            return EncoderFamily::Mozjpeg;
        }
    }

    // Check if DQT matches IJG formula
    if matches_ijg_tables(&table0.values, &table1.values) {
        return identify_ijg_variant(scan, sof);
    }

    // Check for Photoshop: non-IJG tables + Adobe APP14 + Photoshop 3.0 APP13
    if scan.has_adobe && scan.has_photoshop_iptc {
        return EncoderFamily::Photoshop;
    }

    EncoderFamily::Unknown
}

/// 1 DQT table → grayscale image.
fn identify_single_table(scan: &ScanResult, sof: &super::scanner::SofInfo) -> EncoderFamily {
    if sof.num_components != 1 {
        return EncoderFamily::Unknown;
    }

    let table0 = match &scan.dqt_tables[0] {
        Some(t) => t,
        None => return EncoderFamily::Unknown,
    };

    // Check if it matches IJG luminance table at any quality
    if matches_ijg_luma_table(&table0.values) {
        return identify_ijg_variant(scan, sof);
    }

    EncoderFamily::Unknown
}

/// Check if luma/chroma tables match the IJG formula at any quality 1-100.
fn matches_ijg_tables(luma: &[u16; 64], chroma: &[u16; 64]) -> bool {
    for q in 1..=100u8 {
        let ref_luma = generate_ijg_table(q, false);
        let ref_chroma = generate_ijg_table(q, true);
        if *luma == ref_luma && *chroma == ref_chroma {
            return true;
        }
    }
    false
}

/// Check if a single table matches IJG luminance at any quality 1-100.
fn matches_ijg_luma_table(table: &[u16; 64]) -> bool {
    for q in 1..=100u8 {
        let ref_table = generate_ijg_table(q, false);
        if *table == ref_table {
            return true;
        }
    }
    false
}

/// Distinguish between IJG variants using Huffman table characteristics.
fn identify_ijg_variant(scan: &ScanResult, _sof: &super::scanner::SofInfo) -> EncoderFamily {
    let uses_standard_huffman = scan.total_ac_symbols == STANDARD_AC_SYMBOLS_TOTAL;

    if uses_standard_huffman {
        // libjpeg-turbo and Pillow both use standard Huffman tables
        // They are structurally identical — we report as LibjpegTurbo
        return EncoderFamily::LibjpegTurbo;
    }

    // Optimized Huffman → ImageMagick (baseline) or possibly other
    if scan.dht_count > 0 {
        return EncoderFamily::ImageMagick;
    }

    EncoderFamily::IjgFamily
}

/// Generate an IJG-formula quantization table at a given quality.
///
/// Uses integer arithmetic matching libjpeg-turbo's `jpeg_add_quant_table`:
/// `value = clamp((base * scale_factor + 50) / 100, 1, 255)`
///
/// Returns values in natural (row-major) order.
pub(crate) fn generate_ijg_table(quality: u8, is_chrominance: bool) -> [u16; 64] {
    let q = quality.clamp(1, 100) as u32;
    let scale = if q < 50 { 5000 / q } else { 200 - q * 2 };

    let base = if is_chrominance {
        &STD_CHROMINANCE_QUANT
    } else {
        &STD_LUMINANCE_QUANT
    };

    let mut result = [0u16; 64];
    for i in 0..64 {
        let temp = (base[i] as u32 * scale + 50) / 100;
        let clamped = temp.clamp(1, 255) as u16;
        result[i] = clamped;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_ijg_table_q50_identity() {
        // At Q50, scale = 100%, output should match base table
        let table = generate_ijg_table(50, false);
        assert_eq!(table, STD_LUMINANCE_QUANT);
    }

    #[test]
    fn test_generate_ijg_table_q100_all_ones() {
        // At Q100, scale = 0%, all values become 1 (due to clamp)
        let table = generate_ijg_table(100, false);
        for &v in &table {
            assert_eq!(v, 1);
        }
    }

    #[test]
    fn test_generate_ijg_table_q75() {
        // At Q75, scale = 50%
        let table = generate_ijg_table(75, false);
        // DC coefficient: (16 * 50 + 50) / 100 = 8
        assert_eq!(table[0], 8);
        // Second value: (11 * 50 + 50) / 100 = 6
        assert_eq!(table[1], 6);
    }

    #[test]
    fn test_generate_ijg_uses_integer_arithmetic() {
        // Verify we use integer division, not float rounding
        // At Q75 (scale=50), base=16: (16*50+50)/100 = 8 (integer)
        // Float would give round(8.5) = 9
        let table = generate_ijg_table(75, false);
        assert_eq!(
            table[0], 8,
            "Must use integer arithmetic, not float rounding"
        );
    }

    #[test]
    fn test_robidoux_luma_chroma_identical() {
        // Confirm mozjpeg's Robidoux default has identical luma/chroma
        use crate::encode::tables::robidoux::{ROBIDOUX_CHROMINANCE, ROBIDOUX_LUMINANCE};
        assert_eq!(ROBIDOUX_LUMINANCE, ROBIDOUX_CHROMINANCE);
    }
}
