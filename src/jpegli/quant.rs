//! Quantization tables and quality settings.
//!
//! This module provides:
//! - Standard JPEG quantization tables
//! - jpegli's enhanced quantization matrices
//! - Quality parameter handling (traditional and butteraugli distance)
//! - Adaptive quantization support

use crate::jpegli::consts::{
    quality_to_distance, BASE_QUANT_MATRIX_STD, BASE_QUANT_MATRIX_XYB, BASE_QUANT_MATRIX_YCBCR,
    DCT_BLOCK_SIZE, GLOBAL_SCALE_XYB, GLOBAL_SCALE_YCBCR,
};
use crate::jpegli::types::ColorSpace;

// Re-export QuantTable from types
pub use crate::jpegli::types::QuantTable;

/// Per-frequency scaling exponents for non-linear quality scaling.
/// Low frequencies (top-left) use lower exponents for more aggressive scaling,
/// while high frequencies (bottom-right) use 1.0 for linear scaling.
/// From C++ jpegli quant.cc
pub const FREQUENCY_EXPONENT: [f32; DCT_BLOCK_SIZE] = [
    1.00, 0.51, 0.67, 0.74, 1.00, 1.00, 1.00, 1.00, 0.51, 0.66, 0.69, 0.87, 1.00, 1.00, 1.00, 1.00,
    0.67, 0.69, 0.84, 0.83, 0.96, 1.00, 1.00, 1.00, 0.74, 0.87, 0.83, 1.00, 1.00, 0.91, 0.91, 1.00,
    1.00, 1.00, 0.96, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.91, 1.00, 1.00, 1.00, 1.00,
    1.00, 1.00, 1.00, 0.91, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
];

/// Distance threshold where non-linear scaling kicks in.
pub const DIST_THRESHOLD: f32 = 1.5;

/// Distance thresholds for zero-bias blending between HQ and LQ tables.
const DIST_HQ: f32 = 1.0;
const DIST_LQ: f32 = 3.0;

/// Zero-bias multiplier table for YCbCr at low quality (distance >= 3.0).
/// 3 components × 64 coefficients = 192 values.
/// From C++ jpegli quant.cc kZeroBiasMulYCbCrLQ
#[rustfmt::skip]
pub const ZERO_BIAS_MUL_YCBCR_LQ: [f32; 192] = [
    // c = 0 (Y)
    0.0000, 0.0568, 0.3880, 0.6190, 0.6190, 0.4490, 0.4490, 0.6187,
    0.0568, 0.5829, 0.6189, 0.6190, 0.6190, 0.7190, 0.6190, 0.6189,
    0.3880, 0.6189, 0.6190, 0.6190, 0.6190, 0.6190, 0.6187, 0.6100,
    0.6190, 0.6190, 0.6190, 0.6190, 0.5890, 0.3839, 0.7160, 0.6190,
    0.6190, 0.6190, 0.6190, 0.5890, 0.6190, 0.3880, 0.5860, 0.4790,
    0.4490, 0.7190, 0.6190, 0.3839, 0.3880, 0.6190, 0.6190, 0.6190,
    0.4490, 0.6190, 0.6187, 0.7160, 0.5860, 0.6190, 0.6204, 0.6190,
    0.6187, 0.6189, 0.6100, 0.6190, 0.4790, 0.6190, 0.6190, 0.3480,
    // c = 1 (Cb)
    0.0000, 1.1640, 0.9373, 1.1319, 0.8016, 0.9136, 1.1530, 0.9430,
    1.1640, 0.9188, 0.9160, 1.1980, 1.1830, 0.9758, 0.9430, 0.9430,
    0.9373, 0.9160, 0.8430, 1.1720, 0.7083, 0.9430, 0.9430, 0.9430,
    1.1319, 1.1980, 1.1720, 1.1490, 0.8547, 0.9430, 0.9430, 0.9430,
    0.8016, 1.1830, 0.7083, 0.8547, 0.9430, 0.9430, 0.9430, 0.9430,
    0.9136, 0.9758, 0.9430, 0.9430, 0.9430, 0.9430, 0.9430, 0.9430,
    1.1530, 0.9430, 0.9430, 0.9430, 0.9430, 0.9430, 0.9430, 0.9480,
    0.9430, 0.9430, 0.9430, 0.9430, 0.9430, 0.9430, 0.9480, 0.9430,
    // c = 2 (Cr)
    0.0000, 1.3190, 0.4308, 0.4460, 0.0661, 0.0660, 0.2660, 0.2960,
    1.3190, 0.3280, 0.3093, 0.0750, 0.0505, 0.1594, 0.3060, 0.2113,
    0.4308, 0.3093, 0.3060, 0.1182, 0.0500, 0.3060, 0.3915, 0.2426,
    0.4460, 0.0750, 0.1182, 0.0512, 0.0500, 0.2130, 0.3930, 0.1590,
    0.0661, 0.0505, 0.0500, 0.0500, 0.3055, 0.3360, 0.5148, 0.5403,
    0.0660, 0.1594, 0.3060, 0.2130, 0.3360, 0.5060, 0.5874, 0.3060,
    0.2660, 0.3060, 0.3915, 0.3930, 0.5148, 0.5874, 0.3060, 0.3060,
    0.2960, 0.2113, 0.2426, 0.1590, 0.5403, 0.3060, 0.3060, 0.3060,
];

/// Zero-bias multiplier table for YCbCr at high quality (distance <= 1.0).
/// 3 components × 64 coefficients = 192 values.
/// From C++ jpegli quant.cc kZeroBiasMulYCbCrHQ
#[rustfmt::skip]
pub const ZERO_BIAS_MUL_YCBCR_HQ: [f32; 192] = [
    // c = 0 (Y)
    0.0000, 0.0044, 0.2521, 0.6547, 0.8161, 0.6130, 0.8841, 0.8155,
    0.0044, 0.6831, 0.6553, 0.6295, 0.7848, 0.7843, 0.8474, 0.7836,
    0.2521, 0.6553, 0.7834, 0.7829, 0.8161, 0.8072, 0.7743, 0.9242,
    0.6547, 0.6295, 0.7829, 0.8654, 0.7829, 0.6986, 0.7818, 0.7726,
    0.8161, 0.7848, 0.8161, 0.7829, 0.7471, 0.7827, 0.7843, 0.7653,
    0.6130, 0.7843, 0.8072, 0.6986, 0.7827, 0.7848, 0.9508, 0.7653,
    0.8841, 0.8474, 0.7743, 0.7818, 0.7843, 0.9508, 0.7839, 0.8437,
    0.8155, 0.7836, 0.9242, 0.7726, 0.7653, 0.7653, 0.8437, 0.7819,
    // c = 1 (Cb)
    0.0000, 1.0816, 1.0556, 1.2876, 1.1554, 1.1567, 1.8851, 0.5488,
    1.0816, 1.1537, 1.1850, 1.0712, 1.1671, 2.0719, 1.0544, 1.4764,
    1.0556, 1.1850, 1.2870, 1.1981, 1.8181, 1.2618, 1.0564, 1.1191,
    1.2876, 1.0712, 1.1981, 1.4753, 2.0609, 1.0564, 1.2645, 1.0564,
    1.1554, 1.1671, 1.8181, 2.0609, 0.7324, 1.1163, 0.8464, 1.0564,
    1.1567, 2.0719, 1.2618, 1.0564, 1.1163, 1.0040, 1.0564, 1.0564,
    1.8851, 1.0544, 1.0564, 1.2645, 0.8464, 1.0564, 1.0564, 1.0564,
    0.5488, 1.4764, 1.1191, 1.0564, 1.0564, 1.0564, 1.0564, 1.0564,
    // c = 2 (Cr)
    0.0000, 0.5392, 0.6659, 0.8968, 0.6829, 0.6328, 0.5802, 0.4836,
    0.5392, 0.6746, 0.6760, 0.6102, 0.6015, 0.6958, 0.7327, 0.4897,
    0.6659, 0.6760, 0.6957, 0.6543, 0.4396, 0.6330, 0.7081, 0.2583,
    0.8968, 0.6102, 0.6543, 0.5913, 0.6457, 0.5828, 0.5139, 0.3565,
    0.6829, 0.6015, 0.4396, 0.6457, 0.5633, 0.4263, 0.6371, 0.5949,
    0.6328, 0.6958, 0.6330, 0.5828, 0.4263, 0.2847, 0.2909, 0.6629,
    0.5802, 0.7327, 0.7081, 0.5139, 0.6371, 0.2909, 0.6644, 0.6644,
    0.4836, 0.4897, 0.2583, 0.3565, 0.5949, 0.6629, 0.6644, 0.6644,
];

/// Zero-bias offset for DC coefficients (per component).
/// From C++ jpegli quant.cc kZeroBiasOffsetYCbCrDC
pub const ZERO_BIAS_OFFSET_YCBCR_DC: [f32; 3] = [0.0, 0.0, 0.0];

/// Zero-bias offset for AC coefficients (per component).
/// From C++ jpegli quant.cc kZeroBiasOffsetYCbCrAC
pub const ZERO_BIAS_OFFSET_YCBCR_AC: [f32; 3] = [0.59082, 0.58146, 0.57988];

/// Zero-bias parameters for a single DCT block.
///
/// Zero-bias controls how coefficients are rounded toward zero during quantization.
/// A higher multiplier means more aggressive zeroing of small coefficients.
#[derive(Debug, Clone)]
pub struct ZeroBiasParams {
    /// Multiplier per coefficient (64 values)
    pub mul: [f32; DCT_BLOCK_SIZE],
    /// Offset per coefficient (64 values)
    pub offset: [f32; DCT_BLOCK_SIZE],
}

impl Default for ZeroBiasParams {
    fn default() -> Self {
        // Default: all zeros (matches C++ when adaptive quantization is disabled)
        // When AQ is off, C++ sets zero_bias_mul and zero_bias_offset to 0
        // This means no coefficients are biased toward zero based on threshold
        Self {
            mul: [0.0; DCT_BLOCK_SIZE],
            offset: [0.0; DCT_BLOCK_SIZE],
        }
    }
}

impl ZeroBiasParams {
    /// Compute zero-bias parameters for YCbCr color space.
    ///
    /// Blends between HQ and LQ tables based on butteraugli distance.
    /// - distance <= 1.0: Use HQ table
    /// - distance >= 3.0: Use LQ table
    /// - 1.0 < distance < 3.0: Linear blend
    ///
    /// # Arguments
    /// * `distance` - Butteraugli distance (quality parameter)
    /// * `component` - Component index (0=Y, 1=Cb, 2=Cr)
    #[must_use]
    pub fn for_ycbcr(distance: f32, component: usize) -> Self {
        let c = component.min(2);

        // Compute blend factor
        let mix_lq = ((distance - DIST_HQ) / (DIST_LQ - DIST_HQ)).clamp(0.0, 1.0);
        let mix_hq = 1.0 - mix_lq;

        let mut mul = [0.0f32; DCT_BLOCK_SIZE];
        let mut offset = [0.0f32; DCT_BLOCK_SIZE];

        for k in 0..DCT_BLOCK_SIZE {
            let lq = ZERO_BIAS_MUL_YCBCR_LQ[c * DCT_BLOCK_SIZE + k];
            let hq = ZERO_BIAS_MUL_YCBCR_HQ[c * DCT_BLOCK_SIZE + k];
            mul[k] = mix_lq * lq + mix_hq * hq;

            offset[k] = if k == 0 {
                ZERO_BIAS_OFFSET_YCBCR_DC[c]
            } else {
                ZERO_BIAS_OFFSET_YCBCR_AC[c]
            };
        }

        Self { mul, offset }
    }

    /// Compute zero-bias parameters for non-adaptive quantization (simpler default).
    ///
    /// For YCbCr, applies only the offsets without the multiplier blending.
    #[must_use]
    pub fn for_ycbcr_simple(component: usize) -> Self {
        let c = component.min(2);

        let mul = [0.0f32; DCT_BLOCK_SIZE]; // Not used in simple mode

        let mut offset = [0.0f32; DCT_BLOCK_SIZE];
        for k in 0..DCT_BLOCK_SIZE {
            offset[k] = if k == 0 {
                ZERO_BIAS_OFFSET_YCBCR_DC[c]
            } else {
                ZERO_BIAS_OFFSET_YCBCR_AC[c]
            };
        }

        Self { mul, offset }
    }

    /// Apply zero-bias to a coefficient before quantization.
    ///
    /// This adjusts the rounding behavior to favor zeroing small coefficients.
    #[inline]
    #[must_use]
    pub fn apply(&self, coeff: f32, k: usize, quant: f32) -> f32 {
        let threshold = (self.mul[k] + self.offset[k]) * quant;
        if coeff.abs() < threshold {
            0.0
        } else {
            coeff
        }
    }
}

/// Converts butteraugli distance to a per-frequency scale factor.
///
/// This implements jpegli's non-linear quality scaling. At low distances
/// (high quality), scaling is linear. Above DIST_THRESHOLD, scaling becomes
/// non-linear based on the frequency-dependent exponent.
///
/// # Arguments
/// * `distance` - Butteraugli distance (quality parameter)
/// * `freq_idx` - DCT frequency index (0-63, in zigzag order)
///
/// # Returns
/// Scale factor for quantization
#[inline]
#[must_use]
pub fn distance_to_scale(distance: f32, freq_idx: usize) -> f32 {
    if distance < DIST_THRESHOLD {
        return distance;
    }
    let exp = FREQUENCY_EXPONENT[freq_idx];
    let mul = DIST_THRESHOLD.powf(1.0 - exp);
    (0.5 * distance).max(mul * distance.powf(exp))
}

/// Inverse of distance_to_scale - converts scale back to distance.
#[inline]
#[must_use]
pub fn scale_to_distance(scale: f32, freq_idx: usize) -> f32 {
    if scale < DIST_THRESHOLD {
        return scale;
    }
    let exp = 1.0 / FREQUENCY_EXPONENT[freq_idx];
    let mul = DIST_THRESHOLD.powf(1.0 - exp);
    (2.0 * scale).min(mul * scale.powf(exp))
}

/// Infers the butteraugli distance from quantization table values.
///
/// This matches C++ jpegli's `QuantValsToDistance` function.
/// It finds the distance that would produce the given quant tables
/// when using jpegli's quantization formula.
///
/// This is used to compute zero-bias parameters appropriate for the
/// actual quant values, rather than the input distance (which may differ
/// at extreme quality levels where values are clamped to 1).
#[must_use]
pub fn quant_vals_to_distance(
    y_quant: &QuantTable,
    cb_quant: &QuantTable,
    cr_quant: &QuantTable,
) -> f32 {
    use crate::jpegli::consts::{BASE_QUANT_MATRIX_YCBCR, GLOBAL_SCALE_YCBCR};

    const DIST_MAX: f32 = 10000.0;
    const QUANT_MAX: u16 = 255; // baseline JPEG

    let global_scale = GLOBAL_SCALE_YCBCR;

    let mut dist_min = 0.0f32;
    let mut dist_max = DIST_MAX;

    // Process all three components
    let quant_tables = [y_quant, cb_quant, cr_quant];

    for (c, quant) in quant_tables.iter().enumerate() {
        let base_idx = c * DCT_BLOCK_SIZE;
        let base_qm = &BASE_QUANT_MATRIX_YCBCR[base_idx..base_idx + DCT_BLOCK_SIZE];

        for k in 0..DCT_BLOCK_SIZE {
            let mut dmin = 0.0f32;
            let mut dmax = DIST_MAX;
            let invq = 1.0 / base_qm[k] / global_scale;
            let qval = quant.values[k];

            if qval > 1 {
                let scale_min = (qval as f32 - 0.5) * invq;
                dmin = scale_to_distance(scale_min, k);
            }
            if qval < QUANT_MAX {
                let scale_max = (qval as f32 + 0.5) * invq;
                dmax = scale_to_distance(scale_max, k);
            }

            if dmin <= dist_max {
                dist_min = dist_min.max(dmin);
            }
            if dmax >= dist_min {
                dist_max = dist_max.min(dmax);
            }
        }
    }

    // Return the appropriate distance
    if dist_min == 0.0 {
        dist_max
    } else if dist_max >= DIST_MAX {
        dist_min
    } else {
        0.5 * (dist_min + dist_max)
    }
}

/// Standard JPEG luminance quantization table.
/// From ITU-T T.81 (1992) K.1
pub const STD_LUMINANCE_QUANT: [u16; DCT_BLOCK_SIZE] = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113,
    92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
];

/// Standard JPEG chrominance quantization table.
/// From ITU-T T.81 (1992) K.2
pub const STD_CHROMINANCE_QUANT: [u16; DCT_BLOCK_SIZE] = [
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
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
///
/// Uses per-frequency non-linear scaling via `distance_to_scale()` for
/// better quality at the same file size compared to linear scaling.
fn generate_xyb_quant_table(distance: f32, component: usize) -> QuantTable {
    let mut values = [0u16; DCT_BLOCK_SIZE];

    // Select the appropriate base matrix row
    let base_idx = component.min(2) * DCT_BLOCK_SIZE;
    let base = &BASE_QUANT_MATRIX_XYB[base_idx..base_idx + DCT_BLOCK_SIZE];

    for (i, &base_val) in base.iter().enumerate() {
        // Apply per-frequency non-linear scaling
        let scale = distance_to_scale(distance, i) * GLOBAL_SCALE_XYB;
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
///
/// Uses per-frequency non-linear scaling via `distance_to_scale()` for
/// better quality at the same file size compared to linear scaling.
fn generate_standard_quant_table(
    distance: f32,
    component: usize,
    color_space: ColorSpace,
) -> QuantTable {
    let mut values = [0u16; DCT_BLOCK_SIZE];

    // Choose base matrix based on color space
    let (base, global_scale) = if color_space == ColorSpace::YCbCr {
        let base_idx = component.min(2) * DCT_BLOCK_SIZE;
        (
            &BASE_QUANT_MATRIX_YCBCR[base_idx..base_idx + DCT_BLOCK_SIZE],
            GLOBAL_SCALE_YCBCR,
        )
    } else {
        // Use standard JPEG tables
        let base_idx = if component == 0 { 0 } else { DCT_BLOCK_SIZE };
        (
            &BASE_QUANT_MATRIX_STD[base_idx..base_idx + DCT_BLOCK_SIZE],
            1.0,
        )
    };

    for (i, &base_val) in base.iter().enumerate() {
        // Apply per-frequency non-linear scaling
        let scale = distance_to_scale(distance, i) * global_scale;
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
pub fn quantize_block(
    coeffs: &[f32; DCT_BLOCK_SIZE],
    quant: &[u16; DCT_BLOCK_SIZE],
) -> [i16; DCT_BLOCK_SIZE] {
    let mut result = [0i16; DCT_BLOCK_SIZE];
    for i in 0..DCT_BLOCK_SIZE {
        result[i] = quantize(coeffs[i], quant[i]);
    }
    result
}

/// Quantizes a block of DCT coefficients with zero-biasing.
///
/// This matches C++ jpegli's quantization behavior where small coefficients
/// are biased toward zero to improve compression.
///
/// The threshold is: `offset + mul * aq_strength`
/// - If `|coeff/quant| >= threshold`: round normally
/// - Else: set to 0
///
/// For non-adaptive quantization, use aq_strength = 0.0
/// Counter for debugging zero-bias effectiveness
#[cfg(debug_assertions)]
static ZERO_BIAS_DEBUG: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[cfg(debug_assertions)]
static ZERO_BIAS_ZEROS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

pub fn quantize_block_with_zero_bias(
    coeffs: &[f32; DCT_BLOCK_SIZE],
    quant: &[u16; DCT_BLOCK_SIZE],
    zero_bias: &ZeroBiasParams,
    aq_strength: f32,
) -> [i16; DCT_BLOCK_SIZE] {
    let mut result = [0i16; DCT_BLOCK_SIZE];

    for k in 0..DCT_BLOCK_SIZE {
        let q = quant[k] as f32;
        let qval = coeffs[k] / q;

        // Note on scaling: C++ DCT uses 1/64 scaling total, Rust uses 1/8.
        // C++ compensates by using quant_mul = 8/quant in quantization.
        // Net effect: both produce qval = dct_normalized / quant for storage.
        // For threshold comparison, both compare |qval| vs threshold directly.
        let threshold = zero_bias.offset[k] + zero_bias.mul[k] * aq_strength;

        if qval.abs() >= threshold {
            result[k] = qval.round() as i16;
        }
        // else result[k] stays 0
    }
    result
}

/// Alternative: compare with simple quantization
pub fn quantize_block_compare(
    coeffs: &[f32; DCT_BLOCK_SIZE],
    quant: &[u16; DCT_BLOCK_SIZE],
    zero_bias: &ZeroBiasParams,
    aq_strength: f32,
) -> ([i16; DCT_BLOCK_SIZE], usize) {
    let mut result = [0i16; DCT_BLOCK_SIZE];
    let mut zeros_from_bias = 0usize;
    for k in 0..DCT_BLOCK_SIZE {
        let q = quant[k] as f32;
        let qval = coeffs[k] / q;
        let simple_result = qval.round() as i16;
        let threshold = zero_bias.offset[k] + zero_bias.mul[k] * aq_strength;

        if qval.abs() >= threshold {
            result[k] = simple_result;
        } else {
            // Would have been non-zero without zero-biasing
            if simple_result != 0 {
                zeros_from_bias += 1;
            }
        }
    }
    (result, zeros_from_bias)
}

/// Dequantizes a block of coefficients.
pub fn dequantize_block(
    quantized: &[i16; DCT_BLOCK_SIZE],
    quant: &[u16; DCT_BLOCK_SIZE],
) -> [f32; DCT_BLOCK_SIZE] {
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
        let table = generate_quant_table(Quality::from_distance(1.0), 0, ColorSpace::Xyb, true);

        // All values should be valid
        for &v in &table.values {
            assert!(v >= 1 && v <= 255);
        }
    }

    #[test]
    fn test_quant_table_comparison() {
        println!("\n=== Quant Table Comparison (Y channel) ===");
        println!(
            "{:>5} {:>8} {:>10} {:>10} {:>8} {:>8}",
            "Q", "dist", "YCbCr_sum", "XYB_sum", "YCbCr[0]", "XYB[0]"
        );

        for q in [10, 20, 30, 40, 50, 60, 70, 80, 90] {
            let quality = Quality::from_quality(q as f32);
            let distance = quality.to_distance();

            let ycbcr = generate_quant_table(quality, 0, ColorSpace::YCbCr, false);
            let xyb = generate_quant_table(quality, 0, ColorSpace::Xyb, true);

            let ycbcr_sum: u32 = ycbcr.values.iter().map(|&x| x as u32).sum();
            let xyb_sum: u32 = xyb.values.iter().map(|&x| x as u32).sum();

            println!(
                "{:>5} {:>8.2} {:>10} {:>10} {:>8} {:>8}",
                q, distance, ycbcr_sum, xyb_sum, ycbcr.values[0], xyb.values[0]
            );
        }
    }

    #[test]
    fn test_distance_to_scale_linear_region() {
        // Below DIST_THRESHOLD (1.5), scaling should be linear
        for distance in [0.1, 0.5, 1.0, 1.4] {
            for freq_idx in 0..64 {
                let scale = distance_to_scale(distance, freq_idx);
                assert!(
                    (scale - distance).abs() < 1e-6,
                    "Linear region failed: d={}, k={}, scale={}",
                    distance,
                    freq_idx,
                    scale
                );
            }
        }
    }

    #[test]
    fn test_distance_to_scale_nonlinear_region() {
        // Above DIST_THRESHOLD, scaling should be non-linear for some frequencies
        let distance = 3.0;

        // DC coefficient (index 0) has exponent 1.0 - should be close to linear
        let scale_dc = distance_to_scale(distance, 0);
        // The formula with exp=1.0: max(0.5*d, d^1.0) = max(1.5, 3.0) = 3.0
        assert!((scale_dc - distance).abs() < 0.1, "DC scale: {}", scale_dc);

        // Index 1 has exponent 0.51 - should have significant non-linear effect
        let scale_1 = distance_to_scale(distance, 1);
        // exp=0.51, mul=1.5^(1-0.51)=1.5^0.49≈1.22, scale=1.22*3^0.51≈2.15
        // or 0.5*3=1.5, whichever is greater
        assert!(scale_1 > 1.5 && scale_1 < 3.0, "Index 1 scale: {}", scale_1);
    }

    #[test]
    fn test_distance_to_scale_roundtrip() {
        // Test that scale_to_distance inverts distance_to_scale
        for distance in [0.5, 1.0, 2.0, 3.0, 5.0, 10.0] {
            for freq_idx in 0..64 {
                let scale = distance_to_scale(distance, freq_idx);
                let recovered = scale_to_distance(scale, freq_idx);
                assert!(
                    (recovered - distance).abs() < 0.01,
                    "Roundtrip failed: d={}, k={}, scale={}, recovered={}",
                    distance,
                    freq_idx,
                    scale,
                    recovered
                );
            }
        }
    }

    #[test]
    fn test_frequency_exponent_values() {
        // Verify exponent array has expected structure
        assert_eq!(FREQUENCY_EXPONENT.len(), 64);

        // DC coefficient should have exponent 1.0
        assert!((FREQUENCY_EXPONENT[0] - 1.0).abs() < 1e-6);

        // Low frequencies (top-left) should have lower exponents
        assert!(FREQUENCY_EXPONENT[1] < 1.0); // 0.51
        assert!(FREQUENCY_EXPONENT[8] < 1.0); // 0.51

        // High frequencies (bottom-right) should have exponent 1.0
        assert!((FREQUENCY_EXPONENT[63] - 1.0).abs() < 1e-6);
        assert!((FREQUENCY_EXPONENT[62] - 1.0).abs() < 1e-6);
    }

    /// Test that matches C++ DistanceToScale output for specific values.
    /// Reference data generated from instrumented C++ jpegli.
    #[test]
    fn test_distance_to_scale_cpp_reference() {
        // Test cases: (distance, freq_idx, expected_scale)
        // These values should match C++ jpegli exactly
        let test_cases = [
            // Linear region (distance < 1.5)
            (1.0_f32, 0_usize, 1.0_f32),
            (1.0, 1, 1.0),
            (1.0, 63, 1.0),
            // Non-linear region
            (2.0, 0, 2.0), // exp=1.0: max(1.0, 2.0) = 2.0
            (3.0, 0, 3.0), // exp=1.0: max(1.5, 3.0) = 3.0
            (5.0, 0, 5.0), // exp=1.0: linear
        ];

        for (distance, freq_idx, expected) in test_cases {
            let actual = distance_to_scale(distance, freq_idx);
            assert!(
                (actual - expected).abs() < 0.01,
                "Mismatch: distance_to_scale({}, {}) = {}, expected {}",
                distance,
                freq_idx,
                actual,
                expected
            );
        }
    }

    #[test]
    fn test_zero_bias_table_sizes() {
        // Verify table dimensions
        assert_eq!(ZERO_BIAS_MUL_YCBCR_LQ.len(), 192);
        assert_eq!(ZERO_BIAS_MUL_YCBCR_HQ.len(), 192);
        assert_eq!(ZERO_BIAS_OFFSET_YCBCR_DC.len(), 3);
        assert_eq!(ZERO_BIAS_OFFSET_YCBCR_AC.len(), 3);
    }

    #[test]
    fn test_zero_bias_dc_is_zero_in_tables() {
        // DC coefficient (index 0) should have zero multiplier in both tables
        for c in 0..3 {
            assert!(
                ZERO_BIAS_MUL_YCBCR_LQ[c * 64].abs() < 1e-6,
                "LQ DC mul for component {} should be 0, got {}",
                c,
                ZERO_BIAS_MUL_YCBCR_LQ[c * 64]
            );
            assert!(
                ZERO_BIAS_MUL_YCBCR_HQ[c * 64].abs() < 1e-6,
                "HQ DC mul for component {} should be 0, got {}",
                c,
                ZERO_BIAS_MUL_YCBCR_HQ[c * 64]
            );
        }
    }

    #[test]
    fn test_zero_bias_params_default() {
        let params = ZeroBiasParams::default();

        // Default should be all zeros (matches C++ when AQ is disabled)
        for k in 0..64 {
            assert!((params.mul[k]).abs() < 1e-6);
            assert!((params.offset[k]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_zero_bias_for_ycbcr_hq() {
        // At distance <= 1.0, should use HQ table
        let params = ZeroBiasParams::for_ycbcr(0.5, 0);

        // Check some values match HQ table for Y component
        assert!((params.mul[1] - ZERO_BIAS_MUL_YCBCR_HQ[1]).abs() < 1e-5);
        assert!((params.mul[10] - ZERO_BIAS_MUL_YCBCR_HQ[10]).abs() < 1e-5);

        // Check offsets
        assert!((params.offset[0] - ZERO_BIAS_OFFSET_YCBCR_DC[0]).abs() < 1e-5);
        assert!((params.offset[1] - ZERO_BIAS_OFFSET_YCBCR_AC[0]).abs() < 1e-5);
    }

    #[test]
    fn test_zero_bias_for_ycbcr_lq() {
        // At distance >= 3.0, should use LQ table
        let params = ZeroBiasParams::for_ycbcr(5.0, 0);

        // Check some values match LQ table for Y component
        assert!((params.mul[1] - ZERO_BIAS_MUL_YCBCR_LQ[1]).abs() < 1e-5);
        assert!((params.mul[10] - ZERO_BIAS_MUL_YCBCR_LQ[10]).abs() < 1e-5);
    }

    #[test]
    fn test_zero_bias_for_ycbcr_blend() {
        // At distance = 2.0, should be 50/50 blend of HQ and LQ
        let params = ZeroBiasParams::for_ycbcr(2.0, 0);

        // Check a value is between HQ and LQ
        let hq_val = ZERO_BIAS_MUL_YCBCR_HQ[1];
        let lq_val = ZERO_BIAS_MUL_YCBCR_LQ[1];
        let expected = 0.5 * hq_val + 0.5 * lq_val;
        assert!(
            (params.mul[1] - expected).abs() < 1e-5,
            "Expected blend {} (HQ={}, LQ={}), got {}",
            expected,
            hq_val,
            lq_val,
            params.mul[1]
        );
    }

    #[test]
    fn test_zero_bias_for_ycbcr_all_components() {
        // Test all three components
        for c in 0..3 {
            let params = ZeroBiasParams::for_ycbcr(1.5, c);

            // DC offset should match component
            assert!((params.offset[0] - ZERO_BIAS_OFFSET_YCBCR_DC[c]).abs() < 1e-5);

            // AC offsets should match component
            assert!((params.offset[1] - ZERO_BIAS_OFFSET_YCBCR_AC[c]).abs() < 1e-5);
            assert!((params.offset[63] - ZERO_BIAS_OFFSET_YCBCR_AC[c]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_zero_bias_apply() {
        let params = ZeroBiasParams::for_ycbcr(2.0, 0);
        let quant = 16.0;

        // Coefficient below threshold should become zero
        let threshold = (params.mul[1] + params.offset[1]) * quant;
        let small_coeff = threshold * 0.5;
        assert!((params.apply(small_coeff, 1, quant)).abs() < 1e-6);

        // Coefficient above threshold should pass through
        let large_coeff = threshold * 2.0;
        assert!((params.apply(large_coeff, 1, quant) - large_coeff).abs() < 1e-6);
    }

    /// Test zero-bias values against C++ reference data.
    /// These values are computed from C++ jpegli InitQuantizer.
    #[test]
    fn test_zero_bias_cpp_reference() {
        // At distance 2.0 (50% blend), Y component, coefficient 1:
        // LQ[1] = 0.0568, HQ[1] = 0.0044
        // Expected: 0.5 * 0.0568 + 0.5 * 0.0044 = 0.0306
        let params = ZeroBiasParams::for_ycbcr(2.0, 0);
        let expected_mul_1 = 0.5 * 0.0568 + 0.5 * 0.0044;
        assert!(
            (params.mul[1] - expected_mul_1).abs() < 1e-4,
            "Y mul[1] at d=2.0: expected {}, got {}",
            expected_mul_1,
            params.mul[1]
        );

        // Offset for AC should be 0.59082 (Y component)
        assert!(
            (params.offset[1] - 0.59082).abs() < 1e-4,
            "Y offset[1]: expected 0.59082, got {}",
            params.offset[1]
        );
    }
}
