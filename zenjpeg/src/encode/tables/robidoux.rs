//! Core mozjpeg quantization table data (always compiled).
//!
//! Contains the Robidoux luma/chroma arrays, quality scaling formula, and table
//! generation used by `QuantTableSource::MozjpegDefault`. The full 9-preset API
//! (`QuantTablePreset`, `MozjpegTables`) remains behind the `mozjpeg-tables` feature.

use crate::encode::tuning::{EncodingTables, PerComponent, ScalingParams};

/// DCT block size (8x8 = 64 coefficients)
const DCTSIZE2: usize = 64;

/// Robidoux luminance base table (mozjpeg/ImageMagick default).
///
/// Nicolas Robidoux's psychovisually optimized table for high-frequency
/// detail preservation. Used as the default quant table in both mozjpeg
/// and ImageMagick.
pub(crate) const ROBIDOUX_LUMINANCE: [u16; DCTSIZE2] = [
    16, 16, 16, 18, 25, 37, 56, 85, 16, 17, 20, 27, 34, 40, 53, 75, 16, 20, 24, 31, 43, 62, 91,
    135, 18, 27, 31, 40, 53, 74, 106, 156, 25, 34, 43, 53, 69, 94, 131, 189, 37, 40, 62, 74, 94,
    124, 169, 238, 56, 53, 91, 106, 131, 169, 226, 311, 85, 75, 135, 156, 189, 238, 311, 418,
];

/// Robidoux chrominance base table (same as luminance for this preset).
pub(crate) const ROBIDOUX_CHROMINANCE: [u16; DCTSIZE2] = [
    16, 16, 16, 18, 25, 37, 56, 85, 16, 17, 20, 27, 34, 40, 53, 75, 16, 20, 24, 31, 43, 62, 91,
    135, 18, 27, 31, 40, 53, 74, 106, 156, 25, 34, 43, 53, 69, 94, 131, 189, 37, 40, 62, 74, 94,
    124, 169, 238, 56, 53, 91, 106, 131, 169, 226, 311, 85, 75, 135, 156, 189, 238, 311, 418,
];

/// Convert quality (1-100) to scale factor percentage.
///
/// Matches libjpeg's `jpeg_quality_scaling` function:
/// - q < 50: scale = 5000 / q
/// - q >= 50: scale = 200 - 2*q
pub(crate) fn quality_to_scale_factor(quality: u8) -> u32 {
    let q = quality.clamp(1, 100) as u32;

    if q < 50 { 5000 / q } else { 200 - q * 2 }
}

/// Scale a base quantization table by a percentage factor.
///
/// Formula: scaled[i] = (base[i] * scale + 50) / 100
/// Result is clamped to 1..32767 (or 1..255 if force_baseline).
pub(crate) fn scale_table(
    base: &[u16; DCTSIZE2],
    scale: u32,
    force_baseline: bool,
) -> [u16; DCTSIZE2] {
    let mut result = [0u16; DCTSIZE2];
    let max_val = if force_baseline { 255 } else { 32767 };

    for i in 0..DCTSIZE2 {
        let mut temp = (base[i] as u32 * scale + 50) / 100;

        // Clamp to valid range [1, max_val]
        if temp == 0 {
            temp = 1;
        }
        if temp > max_val {
            temp = max_val;
        }

        result[i] = temp as u16;
    }

    result
}

/// Generate mozjpeg default (Robidoux) encoding tables for a given quality.
///
/// Returns `EncodingTables` with:
/// - Robidoux luma/chroma quant tables scaled to the given quality
/// - Neutral zero-bias (0.0 mul, 0.5 AC offset) for standard rounding
/// - `ScalingParams::Exact` (tables are already quality-scaled)
///
/// The `force_baseline` parameter controls clamping:
/// - `true`: clamp quant values to 255 (baseline JPEG, SOF0)
/// - `false`: allow values up to 32767 (extended JPEG, SOF1)
pub(crate) fn generate_mozjpeg_default_tables(
    quality: u8,
    force_baseline: bool,
) -> Box<EncodingTables> {
    let scale = quality_to_scale_factor(quality);
    let luma = scale_table(&ROBIDOUX_LUMINANCE, scale, force_baseline);
    let chroma = scale_table(&ROBIDOUX_CHROMINANCE, scale, force_baseline);

    // Cb and Cr use the same chrominance table
    let quant = PerComponent {
        c0: std::array::from_fn(|i| luma[i] as f32),
        c1: std::array::from_fn(|i| chroma[i] as f32),
        c2: std::array::from_fn(|i| chroma[i] as f32),
    };

    // Neutral zero-bias for mozjpeg compatibility (standard rounding)
    // - mul of 0.0 means no dead zone scaling
    // - offset of 0.5 gives standard round-half-away-from-zero
    let zero_bias_mul = PerComponent {
        c0: [0.0f32; 64],
        c1: [0.0f32; 64],
        c2: [0.0f32; 64],
    };

    Box::new(EncodingTables {
        quant,
        zero_bias_mul,
        zero_bias_offset_dc: [0.0, 0.0, 0.0],
        zero_bias_offset_ac: [0.5, 0.5, 0.5],
        scaling: ScalingParams::Exact,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_to_scale_factor() {
        assert_eq!(quality_to_scale_factor(50), 100); // Q50 = 100%
        assert_eq!(quality_to_scale_factor(75), 50); // Q75 = 50%
        assert_eq!(quality_to_scale_factor(100), 0); // Q100 = 0%
        assert_eq!(quality_to_scale_factor(25), 200); // Q25 = 200%
        assert_eq!(quality_to_scale_factor(1), 5000); // Q1 = 5000%
    }

    #[test]
    fn test_scale_table_q50_identity() {
        // At Q50, scale = 100%, output should match input
        let scaled = scale_table(&ROBIDOUX_LUMINANCE, 100, false);
        assert_eq!(scaled[0], ROBIDOUX_LUMINANCE[0]);
        assert_eq!(scaled[63], ROBIDOUX_LUMINANCE[63]);
    }

    #[test]
    fn test_scale_table_q100_all_ones() {
        // At Q100, scale = 0%, all values should become 1
        let scaled = scale_table(&ROBIDOUX_LUMINANCE, 0, false);
        for &v in &scaled {
            assert_eq!(v, 1);
        }
    }

    #[test]
    fn test_scale_table_force_baseline_clamp() {
        // With force_baseline, values should be clamped to 255
        let scaled = scale_table(&ROBIDOUX_LUMINANCE, 200, true);
        for &v in &scaled {
            assert!(v <= 255);
            assert!(v >= 1);
        }
    }

    #[test]
    fn test_generate_mozjpeg_default_tables_exact() {
        let tables = generate_mozjpeg_default_tables(85, false);
        assert!(tables.is_exact(), "Should use ScalingParams::Exact");
    }

    #[test]
    fn test_generate_mozjpeg_default_tables_neutral_bias() {
        let tables = generate_mozjpeg_default_tables(85, false);
        // Neutral zero-bias: mul=0, AC offset=0.5, DC offset=0
        for &v in tables.zero_bias_mul.c0.iter() {
            assert_eq!(v, 0.0);
        }
        assert_eq!(tables.zero_bias_offset_dc, [0.0, 0.0, 0.0]);
        assert_eq!(tables.zero_bias_offset_ac, [0.5, 0.5, 0.5]);
    }

    #[test]
    fn test_generate_mozjpeg_default_tables_baseline_clamp() {
        let tables = generate_mozjpeg_default_tables(50, true);
        // All quant values should be <= 255
        for &v in tables.quant.c0.iter() {
            assert!(v <= 255.0);
        }
        for &v in tables.quant.c1.iter() {
            assert!(v <= 255.0);
        }
    }

    #[test]
    fn test_robidoux_luma_chroma_equal() {
        // For Robidoux, luma and chroma tables are identical
        assert_eq!(ROBIDOUX_LUMINANCE, ROBIDOUX_CHROMINANCE);
    }
}
