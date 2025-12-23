//! Quantization tables and quality settings.
//!
//! This module provides:
//! - Standard JPEG quantization tables
//! - jpegli's enhanced quantization matrices
//! - Quality parameter handling (traditional and butteraugli distance)
//! - Adaptive quantization support

use crate::consts::{
    DCT_BLOCK_SIZE, GLOBAL_SCALE_XYB, GLOBAL_SCALE_YCBCR,
    BASE_QUANT_MATRIX_XYB, BASE_QUANT_MATRIX_YCBCR, BASE_QUANT_MATRIX_STD,
    quality_to_distance,
};
use crate::types::ColorSpace;

// Re-export QuantTable from types
pub use crate::types::QuantTable;

/// Standard JPEG luminance quantization table.
/// From ITU-T T.81 (1992) K.1
pub const STD_LUMINANCE_QUANT: [u16; DCT_BLOCK_SIZE] = [
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
];

/// Standard JPEG chrominance quantization table.
/// From ITU-T T.81 (1992) K.2
pub const STD_CHROMINANCE_QUANT: [u16; DCT_BLOCK_SIZE] = [
    17, 18, 24, 47, 99, 99, 99, 99,
    18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
];

/// Quality representation that can be either traditional (1-100) or butteraugli distance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Quality {
    /// Traditional JPEG quality (1-100, where 100 is best)
    Traditional(f32),
    /// Butteraugli distance (0.0 = lossless, higher = more compression)
    /// Typical values: 0.5 = very high quality, 1.0 = high, 2.0 = medium
    Distance(f32),
}

impl Default for Quality {
    fn default() -> Self {
        Self::Traditional(90.0)
    }
}

impl Quality {
    /// Creates a quality setting from traditional JPEG quality (1-100).
    #[must_use]
    pub fn from_quality(q: f32) -> Self {
        Self::Traditional(q.clamp(1.0, 100.0))
    }

    /// Creates a quality setting from butteraugli distance.
    #[must_use]
    pub fn from_distance(d: f32) -> Self {
        Self::Distance(d.max(0.0))
    }

    /// Converts to butteraugli distance.
    #[must_use]
    pub fn to_distance(self) -> f32 {
        match self {
            Self::Traditional(q) => quality_to_distance(q as i32),
            Self::Distance(d) => d,
        }
    }

    /// Converts to traditional quality (approximate).
    #[must_use]
    pub fn to_quality(self) -> f32 {
        match self {
            Self::Traditional(q) => q,
            Self::Distance(d) => distance_to_quality(d),
        }
    }

    /// Converts to linear quality (0.0-1.0 where 1.0 is best).
    #[must_use]
    pub fn to_linear(self) -> f32 {
        let d = self.to_distance();
        // Approximate inverse of linear_quality_to_distance
        if d <= 0.1 {
            1.0
        } else {
            (0.1 / d).min(1.0)
        }
    }
}

/// Converts butteraugli distance to approximate traditional quality.
fn distance_to_quality(distance: f32) -> f32 {
    // Approximate inverse of quality_to_distance
    if distance <= 0.0 {
        100.0
    } else if distance >= 15.0 {
        1.0
    } else {
        // This is a rough approximation
        100.0 - (distance * 6.6).min(99.0)
    }
}

/// Generates a quantization table for the given quality and component.
///
/// # Arguments
/// * `quality` - Quality setting
/// * `component` - Component index (0 = Y/luma, 1+ = chroma)
/// * `color_space` - Color space being used
/// * `use_xyb` - Whether to use XYB-optimized tables
#[must_use]
pub fn generate_quant_table(
    quality: Quality,
    component: usize,
    color_space: ColorSpace,
    use_xyb: bool,
) -> QuantTable {
    let distance = quality.to_distance();

    if use_xyb {
        generate_xyb_quant_table(distance, component)
    } else {
        generate_standard_quant_table(distance, component, color_space)
    }
}

/// Generates a quantization table using jpegli's XYB-optimized matrices.
fn generate_xyb_quant_table(distance: f32, component: usize) -> QuantTable {
    let mut values = [0u16; DCT_BLOCK_SIZE];

    // Select the appropriate base matrix row
    let base_idx = component.min(2) * DCT_BLOCK_SIZE;
    let base = &BASE_QUANT_MATRIX_XYB[base_idx..base_idx + DCT_BLOCK_SIZE];

    // Scale by distance and global scale
    let scale = distance * GLOBAL_SCALE_XYB;

    for (i, &base_val) in base.iter().enumerate() {
        let q = (base_val * scale).round();
        // Clamp to valid quantization values (1-255 for baseline)
        values[i] = (q as u16).clamp(1, 255);
    }

    QuantTable {
        values,
        precision: 0, // 8-bit for baseline
    }
}

/// Generates a quantization table using standard or YCbCr matrices.
fn generate_standard_quant_table(distance: f32, component: usize, color_space: ColorSpace) -> QuantTable {
    let mut values = [0u16; DCT_BLOCK_SIZE];

    // Choose base matrix based on color space
    let (base, global_scale) = if color_space == ColorSpace::YCbCr {
        let base_idx = component.min(2) * DCT_BLOCK_SIZE;
        (&BASE_QUANT_MATRIX_YCBCR[base_idx..base_idx + DCT_BLOCK_SIZE], GLOBAL_SCALE_YCBCR)
    } else {
        // Use standard JPEG tables
        let base_idx = if component == 0 { 0 } else { DCT_BLOCK_SIZE };
        (&BASE_QUANT_MATRIX_STD[base_idx..base_idx + DCT_BLOCK_SIZE], 1.0)
    };

    // Scale by distance
    let scale = distance * global_scale;

    for (i, &base_val) in base.iter().enumerate() {
        let q = (base_val * scale).round();
        values[i] = (q as u16).clamp(1, 255);
    }

    QuantTable {
        values,
        precision: 0,
    }
}

/// Generates a standard JPEG quantization table scaled by quality factor.
///
/// # Arguments
/// * `quality` - Quality 1-100 (100 = best)
/// * `is_chrominance` - True for Cb/Cr tables, false for Y
#[must_use]
pub fn generate_standard_jpeg_table(quality: f32, is_chrominance: bool) -> QuantTable {
    let base_table = if is_chrominance {
        &STD_CHROMINANCE_QUANT
    } else {
        &STD_LUMINANCE_QUANT
    };

    // Standard JPEG quality scaling
    let quality = quality.clamp(1.0, 100.0);
    let scale = if quality < 50.0 {
        5000.0 / quality
    } else {
        200.0 - quality * 2.0
    };

    let mut values = [0u16; DCT_BLOCK_SIZE];
    for (i, &base) in base_table.iter().enumerate() {
        let q = ((base as f32 * scale + 50.0) / 100.0).round();
        values[i] = (q as u16).clamp(1, 255);
    }

    QuantTable {
        values,
        precision: 0,
    }
}

/// Quantizes a DCT coefficient using the given quantization value.
#[inline]
#[must_use]
pub fn quantize(coeff: f32, quant: u16) -> i16 {
    let q = quant as f32;
    (coeff / q).round() as i16
}

/// Dequantizes a coefficient.
#[inline]
#[must_use]
pub fn dequantize(quantized: i16, quant: u16) -> f32 {
    quantized as f32 * quant as f32
}

/// Quantizes a block of DCT coefficients.
pub fn quantize_block(coeffs: &[f32; DCT_BLOCK_SIZE], quant: &[u16; DCT_BLOCK_SIZE]) -> [i16; DCT_BLOCK_SIZE] {
    let mut result = [0i16; DCT_BLOCK_SIZE];
    for i in 0..DCT_BLOCK_SIZE {
        result[i] = quantize(coeffs[i], quant[i]);
    }
    result
}

/// Dequantizes a block of coefficients.
pub fn dequantize_block(quantized: &[i16; DCT_BLOCK_SIZE], quant: &[u16; DCT_BLOCK_SIZE]) -> [f32; DCT_BLOCK_SIZE] {
    let mut result = [0.0f32; DCT_BLOCK_SIZE];
    for i in 0..DCT_BLOCK_SIZE {
        result[i] = dequantize(quantized[i], quant[i]);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_conversion() {
        // Traditional quality 90 should give reasonable distance
        let q = Quality::from_quality(90.0);
        let d = q.to_distance();
        assert!(d > 0.0 && d < 5.0);

        // Distance 1.0 should round-trip approximately
        let q2 = Quality::from_distance(1.0);
        let d2 = q2.to_distance();
        assert!((d2 - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_standard_table_generation() {
        let table_q50 = generate_standard_jpeg_table(50.0, false);
        let table_q90 = generate_standard_jpeg_table(90.0, false);

        // Higher quality should have smaller quantization values
        let sum_q50: u32 = table_q50.values.iter().map(|&v| v as u32).sum();
        let sum_q90: u32 = table_q90.values.iter().map(|&v| v as u32).sum();
        assert!(sum_q90 < sum_q50);
    }

    #[test]
    fn test_quantize_dequantize() {
        let coeff = 123.456f32;
        let quant = 16;

        let quantized = quantize(coeff, quant);
        let recovered = dequantize(quantized, quant);

        // Should be within one quantization step
        assert!((recovered - coeff).abs() < quant as f32);
    }

    #[test]
    fn test_quant_values_in_range() {
        // All generated tables should have values in [1, 255] for baseline
        for q in [10.0, 50.0, 90.0, 100.0] {
            let table = generate_standard_jpeg_table(q, false);
            for &v in &table.values {
                assert!(v >= 1 && v <= 255);
            }
        }
    }

    #[test]
    fn test_xyb_table_generation() {
        let table = generate_quant_table(
            Quality::from_distance(1.0),
            0,
            ColorSpace::Xyb,
            true,
        );

        // All values should be valid
        for &v in &table.values {
            assert!(v >= 1 && v <= 255);
        }
    }
}
