//! mozjpeg-compatible quantization table presets.
//!
//! This module provides access to the 9 quantization table variants used by mozjpeg,
//! along with the standard quality scaling formula for producing compatible output.
//!
//! # Example
//!
//! ```rust,ignore
//! use zenjpeg::encode::{EncoderConfig, ChromaSubsampling, MozjpegTables, QuantTablePreset};
//!
//! // Generate tables using mozjpeg's Robidoux preset at quality 85
//! let tables = MozjpegTables::generate(85, QuantTablePreset::Robidoux);
//!
//! let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
//!     .quant_tables(tables);
//! ```
//!
//! # Available Presets
//!
//! - `JpegAnnexK` - Standard JPEG Annex K tables (libjpeg default)
//! - `Flat` - Uniform quantization (all values the same base)
//! - `MssimTuned` - MSSIM-optimized on Kodak image set
//! - `Robidoux` - Nicolas Robidoux's tables (mozjpeg/ImageMagick default)
//! - `PsnrHvsM` - PSNR-HVS-M tuned
//! - `Klein` - Klein, Silverstein, Carney (1992)
//! - `Watson` - DCTune (Watson, Taylor, Borthwick 1997)
//! - `Ahumada` - Ahumada, Watson, Peterson (1993)
//! - `Peterson` - Peterson, Ahumada, Watson (1993)

use crate::encode::tuning::{EncodingTables, PerComponent, ScalingParams};

/// DCT block size (8x8 = 64 coefficients)
const DCTSIZE2: usize = 64;

/// Quantization table preset variants from mozjpeg.
///
/// These represent different psychovisual optimization strategies for
/// quantization tables. The default (`Robidoux`) is used by both mozjpeg
/// and ImageMagick.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum QuantTablePreset {
    /// JPEG Annex K (standard libjpeg tables)
    JpegAnnexK,
    /// Flat (uniform quantization - all coefficients have same base value)
    Flat,
    /// MSSIM-tuned on Kodak image set
    MssimTuned,
    /// Robidoux tables (Nicolas Robidoux, mozjpeg/ImageMagick default)
    ///
    /// Psychovisually optimized for high-frequency detail preservation.
    #[default]
    Robidoux,
    /// PSNR-HVS-M tuned
    PsnrHvsM,
    /// Klein, Silverstein, Carney (1992)
    Klein,
    /// Watson, Taylor, Borthwick (DCTune, 1997)
    Watson,
    /// Ahumada, Watson, Peterson (1993)
    Ahumada,
    /// Peterson, Ahumada, Watson (1993)
    Peterson,
}

impl QuantTablePreset {
    /// Alias: ImageMagick tables (same as Robidoux)
    #[allow(non_upper_case_globals)]
    pub const ImageMagick: Self = Self::Robidoux;
}

/// Helper for generating mozjpeg-compatible quantization tables.
///
/// Use this when you want JPEG output that matches mozjpeg's quality/size
/// characteristics for a given quality level.
pub struct MozjpegTables;

impl MozjpegTables {
    /// Generate encoding tables using mozjpeg's quality scaling formula.
    ///
    /// # Arguments
    ///
    /// * `quality` - Quality level 1-100 (higher = better quality, larger files)
    /// * `preset` - Which base table variant to use
    ///
    /// # Returns
    ///
    /// An [`EncodingTables`] with the scaled luminance and chrominance tables,
    /// using `ScalingParams::Exact` (tables are already quality-scaled).
    ///
    /// # Quality Scaling
    ///
    /// The quality value is converted to a scale factor using the standard
    /// libjpeg/mozjpeg formula:
    /// - q < 50: scale = 5000 / q (e.g., q=25 → 200% of base values)
    /// - q >= 50: scale = 200 - 2*q (e.g., q=75 → 50% of base values)
    ///
    /// At q=50, the base tables are used as-is. At q=100, all values become 1.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use zenjpeg::encode::{MozjpegTables, QuantTablePreset};
    ///
    /// // Standard mozjpeg quality 85 with default (Robidoux) tables
    /// let tables = MozjpegTables::generate(85, QuantTablePreset::default());
    ///
    /// // Or with a specific preset
    /// let tables = MozjpegTables::generate(75, QuantTablePreset::MssimTuned);
    /// ```
    #[must_use]
    pub fn generate(quality: u8, preset: QuantTablePreset) -> Box<EncodingTables> {
        Self::generate_ex(quality, preset, false)
    }

    /// Generate encoding tables with baseline clamping control.
    ///
    /// Like [`generate`](Self::generate), but with explicit control over whether
    /// to clamp values to 255 for baseline JPEG compatibility.
    ///
    /// # Arguments
    ///
    /// * `quality` - Quality level 1-100
    /// * `preset` - Which base table variant to use
    /// * `force_baseline` - If true, clamp all values to 255 (baseline JPEG)
    ///
    /// # Notes
    ///
    /// When using `EncoderConfig::force_baseline()`, the encoder will already
    /// clamp quant values. This parameter is mainly useful if you want to
    /// inspect the baseline-clamped tables directly.
    #[must_use]
    pub fn generate_ex(
        quality: u8,
        preset: QuantTablePreset,
        force_baseline: bool,
    ) -> Box<EncodingTables> {
        let scale = quality_to_scale_factor(quality);
        let luma_base = get_luminance_table(preset);
        let chroma_base = get_chrominance_table(preset);

        let luma = scale_table(luma_base, scale, force_baseline);
        let cb = scale_table(chroma_base, scale, force_baseline);
        let cr = cb; // Cb and Cr use the same chrominance table

        // Convert u16 tables to f32 for EncodingTables
        let quant = PerComponent {
            c0: std::array::from_fn(|i| luma[i] as f32),
            c1: std::array::from_fn(|i| cb[i] as f32),
            c2: std::array::from_fn(|i| cr[i] as f32),
        };

        // Use neutral zero-bias for mozjpeg compatibility (standard rounding)
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

    /// Get the raw (unscaled) luminance base table for a preset.
    ///
    /// These are the base values before quality scaling is applied.
    /// At quality 50, these values are used as-is.
    #[must_use]
    pub fn luminance_base(preset: QuantTablePreset) -> &'static [u16; DCTSIZE2] {
        get_luminance_table(preset)
    }

    /// Get the raw (unscaled) chrominance base table for a preset.
    ///
    /// These are the base values before quality scaling is applied.
    /// At quality 50, these values are used as-is.
    #[must_use]
    pub fn chrominance_base(preset: QuantTablePreset) -> &'static [u16; DCTSIZE2] {
        get_chrominance_table(preset)
    }
}

// Core scaling functions imported from mozjpeg_table_data (always compiled).
use super::robidoux::{quality_to_scale_factor, scale_table};

fn get_luminance_table(preset: QuantTablePreset) -> &'static [u16; DCTSIZE2] {
    &STD_LUMINANCE_QUANT_TBL[preset as usize]
}

fn get_chrominance_table(preset: QuantTablePreset) -> &'static [u16; DCTSIZE2] {
    &STD_CHROMINANCE_QUANT_TBL[preset as usize]
}

// =============================================================================
// Quantization Tables (from mozjpeg)
// =============================================================================
// mozjpeg provides 9 different quantization table sets, indexed 0-8.
// Each set has a luminance table and a chrominance table.

/// Number of quantization table variants
const NUM_QUANT_TABLE_VARIANTS: usize = 9;

/// Standard luminance quantization tables (9 variants).
/// Source: mozjpeg jcparam.c std_luminance_quant_tbl
const STD_LUMINANCE_QUANT_TBL: [[u16; DCTSIZE2]; NUM_QUANT_TABLE_VARIANTS] = [
    // 0: JPEG Annex K
    [
        16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69,
        56, 14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81,
        104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
    ],
    // 1: Flat
    [
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
    ],
    // 2: MSSIM-tuned on Kodak
    [
        12, 17, 20, 21, 30, 34, 56, 63, 18, 20, 20, 26, 28, 51, 61, 55, 19, 20, 21, 26, 33, 58, 69,
        55, 26, 26, 26, 30, 46, 87, 86, 66, 31, 33, 36, 40, 46, 96, 100, 73, 40, 35, 46, 62, 81,
        100, 111, 91, 46, 66, 76, 86, 102, 121, 120, 101, 68, 90, 90, 96, 113, 102, 105, 103,
    ],
    // 3: Robidoux (Nicolas Robidoux, used by ImageMagick and mozjpeg)
    [
        16, 16, 16, 18, 25, 37, 56, 85, 16, 17, 20, 27, 34, 40, 53, 75, 16, 20, 24, 31, 43, 62, 91,
        135, 18, 27, 31, 40, 53, 74, 106, 156, 25, 34, 43, 53, 69, 94, 131, 189, 37, 40, 62, 74,
        94, 124, 169, 238, 56, 53, 91, 106, 131, 169, 226, 311, 85, 75, 135, 156, 189, 238, 311,
        418,
    ],
    // 4: PSNR-HVS-M tuned
    [
        9, 10, 12, 14, 27, 32, 51, 62, 11, 12, 14, 19, 27, 44, 59, 73, 12, 14, 18, 25, 42, 59, 79,
        78, 17, 18, 25, 42, 61, 92, 87, 92, 23, 28, 42, 75, 79, 112, 112, 99, 40, 42, 59, 84, 88,
        124, 132, 111, 42, 64, 78, 95, 105, 126, 125, 99, 70, 75, 100, 102, 116, 100, 107, 98,
    ],
    // 5: Klein, Silverstein, Carney (1992)
    [
        10, 12, 14, 19, 26, 38, 57, 86, 12, 18, 21, 28, 35, 41, 54, 76, 14, 21, 25, 32, 44, 63, 92,
        136, 19, 28, 32, 41, 54, 75, 107, 157, 26, 35, 44, 54, 70, 95, 132, 190, 38, 41, 63, 75,
        95, 125, 170, 239, 57, 54, 92, 107, 132, 170, 227, 312, 86, 76, 136, 157, 190, 239, 312,
        419,
    ],
    // 6: Watson, Taylor, Borthwick (DCTune, 1997)
    [
        7, 8, 10, 14, 23, 44, 95, 241, 8, 8, 11, 15, 25, 47, 102, 255, 10, 11, 13, 19, 31, 58, 127,
        255, 14, 15, 19, 27, 44, 83, 181, 255, 23, 25, 31, 44, 72, 136, 255, 255, 44, 47, 58, 83,
        136, 255, 255, 255, 95, 102, 127, 181, 255, 255, 255, 255, 241, 255, 255, 255, 255, 255,
        255, 255,
    ],
    // 7: Ahumada, Watson, Peterson (1993)
    [
        15, 11, 11, 12, 15, 19, 25, 32, 11, 13, 10, 10, 12, 15, 19, 24, 11, 10, 14, 14, 16, 18, 22,
        27, 12, 10, 14, 18, 21, 24, 28, 33, 15, 12, 16, 21, 26, 31, 36, 42, 19, 15, 18, 24, 31, 38,
        45, 53, 25, 19, 22, 28, 36, 45, 55, 65, 32, 24, 27, 33, 42, 53, 65, 77,
    ],
    // 8: Peterson, Ahumada, Watson (1993)
    [
        14, 10, 11, 14, 19, 25, 34, 45, 10, 11, 11, 12, 15, 20, 26, 33, 11, 11, 15, 18, 21, 25, 31,
        38, 14, 12, 18, 24, 28, 33, 39, 47, 19, 15, 21, 28, 36, 43, 51, 59, 25, 20, 25, 33, 43, 54,
        64, 74, 34, 26, 31, 39, 51, 64, 77, 91, 45, 33, 38, 47, 59, 74, 91, 108,
    ],
];

/// Standard chrominance quantization tables (9 variants).
/// Source: mozjpeg jcparam.c std_chrominance_quant_tbl
const STD_CHROMINANCE_QUANT_TBL: [[u16; DCTSIZE2]; NUM_QUANT_TABLE_VARIANTS] = [
    // 0: JPEG Annex K
    [
        17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99,
        99, 47, 66, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    ],
    // 1: Flat
    [
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
    ],
    // 2: MSSIM-tuned on Kodak
    [
        8, 12, 15, 15, 86, 96, 96, 98, 13, 13, 15, 26, 90, 96, 99, 98, 12, 15, 18, 96, 99, 99, 99,
        99, 17, 16, 90, 96, 99, 99, 99, 99, 96, 96, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    ],
    // 3: Robidoux (same as luminance for this preset)
    [
        16, 16, 16, 18, 25, 37, 56, 85, 16, 17, 20, 27, 34, 40, 53, 75, 16, 20, 24, 31, 43, 62, 91,
        135, 18, 27, 31, 40, 53, 74, 106, 156, 25, 34, 43, 53, 69, 94, 131, 189, 37, 40, 62, 74,
        94, 124, 169, 238, 56, 53, 91, 106, 131, 169, 226, 311, 85, 75, 135, 156, 189, 238, 311,
        418,
    ],
    // 4: PSNR-HVS-M tuned
    [
        9, 10, 17, 19, 62, 89, 91, 97, 12, 13, 18, 29, 84, 91, 88, 98, 14, 19, 29, 93, 95, 95, 98,
        97, 20, 26, 84, 88, 95, 95, 98, 94, 26, 86, 91, 93, 97, 99, 98, 99, 99, 100, 98, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 97, 97, 99, 99, 99, 99, 97, 99,
    ],
    // 5: Klein (same as luminance)
    [
        10, 12, 14, 19, 26, 38, 57, 86, 12, 18, 21, 28, 35, 41, 54, 76, 14, 21, 25, 32, 44, 63, 92,
        136, 19, 28, 32, 41, 54, 75, 107, 157, 26, 35, 44, 54, 70, 95, 132, 190, 38, 41, 63, 75,
        95, 125, 170, 239, 57, 54, 92, 107, 132, 170, 227, 312, 86, 76, 136, 157, 190, 239, 312,
        419,
    ],
    // 6: Watson (same as luminance)
    [
        7, 8, 10, 14, 23, 44, 95, 241, 8, 8, 11, 15, 25, 47, 102, 255, 10, 11, 13, 19, 31, 58, 127,
        255, 14, 15, 19, 27, 44, 83, 181, 255, 23, 25, 31, 44, 72, 136, 255, 255, 44, 47, 58, 83,
        136, 255, 255, 255, 95, 102, 127, 181, 255, 255, 255, 255, 241, 255, 255, 255, 255, 255,
        255, 255,
    ],
    // 7: Ahumada (same as luminance)
    [
        15, 11, 11, 12, 15, 19, 25, 32, 11, 13, 10, 10, 12, 15, 19, 24, 11, 10, 14, 14, 16, 18, 22,
        27, 12, 10, 14, 18, 21, 24, 28, 33, 15, 12, 16, 21, 26, 31, 36, 42, 19, 15, 18, 24, 31, 38,
        45, 53, 25, 19, 22, 28, 36, 45, 55, 65, 32, 24, 27, 33, 42, 53, 65, 77,
    ],
    // 8: Peterson (same as luminance)
    [
        14, 10, 11, 14, 19, 25, 34, 45, 10, 11, 11, 12, 15, 20, 26, 33, 11, 11, 15, 18, 21, 25, 31,
        38, 14, 12, 18, 24, 28, 33, 39, 47, 19, 15, 21, 28, 36, 43, 51, 59, 25, 20, 25, 33, 43, 54,
        64, 74, 34, 26, 31, 39, 51, 64, 77, 91, 45, 33, 38, 47, 59, 74, 91, 108,
    ],
];

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
        assert_eq!(quality_to_scale_factor(10), 500); // Q10 = 500%
    }

    #[test]
    fn test_scale_table_q50() {
        // At Q50, scale = 100%, so output should match input
        let base = get_luminance_table(QuantTablePreset::JpegAnnexK);
        let scaled = scale_table(base, 100, false);
        assert_eq!(scaled[0], base[0]);
        assert_eq!(scaled[63], base[63]);
    }

    #[test]
    fn test_scale_table_q100() {
        // At Q100, scale = 0%, so all values should become 1
        let base = get_luminance_table(QuantTablePreset::JpegAnnexK);
        let scaled = scale_table(base, 0, false);
        for &v in &scaled {
            assert_eq!(v, 1);
        }
    }

    #[test]
    fn test_scale_table_force_baseline() {
        // With force_baseline, values should be clamped to 255
        let base = get_luminance_table(QuantTablePreset::Robidoux);
        // At low quality (high scale), some values would exceed 255
        let scaled = scale_table(base, 200, true);
        for &v in &scaled {
            assert!(v <= 255);
            assert!(v >= 1);
        }
    }

    #[test]
    fn test_generate_returns_exact_scaling() {
        let tables = MozjpegTables::generate(85, QuantTablePreset::Robidoux);
        assert!(tables.is_exact(), "Should use ScalingParams::Exact");
    }

    #[test]
    fn test_all_presets_valid() {
        // Verify all presets produce valid tables
        for preset in [
            QuantTablePreset::JpegAnnexK,
            QuantTablePreset::Flat,
            QuantTablePreset::MssimTuned,
            QuantTablePreset::Robidoux,
            QuantTablePreset::PsnrHvsM,
            QuantTablePreset::Klein,
            QuantTablePreset::Watson,
            QuantTablePreset::Ahumada,
            QuantTablePreset::Peterson,
        ] {
            let tables = MozjpegTables::generate(75, preset);
            // Check all quant values are in valid range (1-32767 as f32)
            for &v in tables
                .quant
                .c0
                .iter()
                .chain(tables.quant.c1.iter())
                .chain(tables.quant.c2.iter())
            {
                assert!((1.0..=32767.0).contains(&v), "Invalid quant value: {}", v);
            }
        }
    }

    #[test]
    fn test_base_table_accessors() {
        let luma = MozjpegTables::luminance_base(QuantTablePreset::JpegAnnexK);
        let chroma = MozjpegTables::chrominance_base(QuantTablePreset::JpegAnnexK);

        // Verify first few values match JPEG Annex K
        assert_eq!(luma[0], 16);
        assert_eq!(luma[1], 11);
        assert_eq!(chroma[0], 17);
        assert_eq!(chroma[1], 18);
    }
}
