//! Default quantization and zero-bias tables.
//!
//! These tables are exposed for users who want to customize encoding by
//! modifying the defaults rather than starting from scratch.
//!
//! # Quantization Tables
//!
//! The base quantization matrices are 192 f32 values organized as:
//! - `[0..64]`: Y (luma) component
//! - `[64..128]`: Cb (blue chroma) component
//! - `[128..192]`: Cr (red chroma) component
//!
//! Each 64-element block is in row-major order (NOT zigzag).
//!
//! # Zero-Bias Tables
//!
//! Zero-bias controls coefficient rounding during quantization:
//! - `ZERO_BIAS_MUL_*`: Multiplier tables (higher = more aggressive zeroing)
//! - `ZERO_BIAS_OFFSET_*`: Offset tables
//!
//! The encoder blends between HQ and LQ tables based on quality:
//! - Quality >= 88 (distance <= 1.0): Uses HQ tables
//! - Quality <= 73 (distance >= 3.0): Uses LQ tables
//! - In between: Linear blend
//!
//! # Example
//!
//! ```ignore
//! use zenjpeg::encode::tuning::{EncodingTables, PerComponent};
//! use zenjpeg::encode::tables;
//!
//! // Start with jpegli defaults and modify
//! let mut tables = EncodingTables::default_ycbcr();
//!
//! // Modify: reduce DC quantization for sharper edges
//! tables.quant.c0[0] *= 0.5; // Y
//! tables.quant.c1[0] *= 0.5; // Cb
//! tables.quant.c2[0] *= 0.5; // Cr
//!
//! // Or use mozjpeg-style tables
//! use zenjpeg::encode::{MozjpegTables, QuantTablePreset};
//! let mozjpeg_tables = MozjpegTables::generate(85, QuantTablePreset::Robidoux);
//! ```

/// Core mozjpeg quantization table data (Robidoux arrays, quality scaling).
///
/// Always compiled. Used by `QuantTableSource::MozjpegDefault` to generate
/// Robidoux-based quant tables without enabling the full `mozjpeg-tables` feature.
pub(crate) mod robidoux;

/// mozjpeg-compatible quantization table presets.
///
/// Provides access to the 9 quantization table variants used by mozjpeg
/// (Robidoux, MSSIM, Klein, etc.) with the standard quality scaling formula.
///
/// Requires the `mozjpeg-tables` feature flag.
pub mod presets;

/// Glassa low-BPP optimized quantization tables for extreme compression.
///
/// SA-optimized tables that outperform mozjpeg defaults at Q3-Q25 (low quality).
/// Use for thumbnails, LQIP, and progressive placeholders.
pub mod glassa;

// Re-export base quantization matrices
pub use crate::foundation::consts::{
    BASE_QUANT_MATRIX_STD as BASE_QUANT_STD, BASE_QUANT_MATRIX_XYB as BASE_QUANT_XYB,
    BASE_QUANT_MATRIX_YCBCR as BASE_QUANT_YCBCR,
};

// Re-export global scale factors
pub use crate::foundation::consts::{GLOBAL_SCALE_420, GLOBAL_SCALE_XYB, GLOBAL_SCALE_YCBCR};

// Re-export zero-bias tables
pub use crate::quant::{
    ZERO_BIAS_MUL_XYB, ZERO_BIAS_MUL_XYB_HQ, ZERO_BIAS_MUL_XYB_LQ, ZERO_BIAS_MUL_YCBCR_HQ,
    ZERO_BIAS_MUL_YCBCR_LQ, ZERO_BIAS_OFFSET_XYB, ZERO_BIAS_OFFSET_XYB_AC,
    ZERO_BIAS_OFFSET_YCBCR_AC, ZERO_BIAS_OFFSET_YCBCR_DC,
};

/// Extract the Y (luma) component from a 192-element matrix.
#[inline]
#[must_use]
pub fn luma_from_192(matrix: &[f32; 192]) -> [f32; 64] {
    let mut result = [0.0f32; 64];
    result.copy_from_slice(&matrix[0..64]);
    result
}

/// Extract the Cb (blue chroma) component from a 192-element matrix.
#[inline]
#[must_use]
pub fn cb_from_192(matrix: &[f32; 192]) -> [f32; 64] {
    let mut result = [0.0f32; 64];
    result.copy_from_slice(&matrix[64..128]);
    result
}

/// Extract the Cr (red chroma) component from a 192-element matrix.
#[inline]
#[must_use]
pub fn cr_from_192(matrix: &[f32; 192]) -> [f32; 64] {
    let mut result = [0.0f32; 64];
    result.copy_from_slice(&matrix[128..192]);
    result
}

/// Pack three 64-element components into a 192-element matrix.
#[inline]
#[must_use]
pub fn pack_192(luma: &[f32; 64], cb: &[f32; 64], cr: &[f32; 64]) -> [f32; 192] {
    let mut result = [0.0f32; 192];
    result[0..64].copy_from_slice(luma);
    result[64..128].copy_from_slice(cb);
    result[128..192].copy_from_slice(cr);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_components() {
        let luma = luma_from_192(&BASE_QUANT_YCBCR);
        let cb = cb_from_192(&BASE_QUANT_YCBCR);
        let cr = cr_from_192(&BASE_QUANT_YCBCR);

        // Verify extraction matches original
        assert_eq!(&luma[..], &BASE_QUANT_YCBCR[0..64]);
        assert_eq!(&cb[..], &BASE_QUANT_YCBCR[64..128]);
        assert_eq!(&cr[..], &BASE_QUANT_YCBCR[128..192]);
    }

    #[test]
    fn test_pack_roundtrip() {
        let luma = luma_from_192(&BASE_QUANT_YCBCR);
        let cb = cb_from_192(&BASE_QUANT_YCBCR);
        let cr = cr_from_192(&BASE_QUANT_YCBCR);

        let packed = pack_192(&luma, &cb, &cr);
        assert_eq!(packed, BASE_QUANT_YCBCR);
    }

    #[test]
    fn test_zero_bias_table_sizes() {
        assert_eq!(ZERO_BIAS_MUL_YCBCR_LQ.len(), 192);
        assert_eq!(ZERO_BIAS_MUL_YCBCR_HQ.len(), 192);
        assert_eq!(ZERO_BIAS_OFFSET_YCBCR_DC.len(), 3);
        assert_eq!(ZERO_BIAS_OFFSET_YCBCR_AC.len(), 3);
    }
}
