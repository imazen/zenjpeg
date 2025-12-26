//! Quantization table handling for JPEG encoding
//!
//! Supports standard JPEG tables, mozjpeg's optimized tables, and jpegli's
//! perceptually-tuned tables.
//!
//! Also provides zero-bias support for adaptive quantization (from jpegli).

use crate::consts::{MOZJPEG_LUMA_QUANT, STD_CHROMA_QUANT, STD_LUMA_QUANT};

// ============================================================================
// Zero-bias tables from jpegli (for adaptive quantization)
// ============================================================================

/// Distance thresholds for zero-bias blending between HQ and LQ tables.
const DIST_HQ: f32 = 1.0;
const DIST_LQ: f32 = 3.0;

/// Zero-bias multiplier table for YCbCr at low quality (distance >= 3.0).
/// 3 components × 64 coefficients = 192 values.
/// From C++ jpegli quant.cc kZeroBiasMulYCbCrLQ
#[rustfmt::skip]
const ZERO_BIAS_MUL_YCBCR_LQ: [f32; 192] = [
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
const ZERO_BIAS_MUL_YCBCR_HQ: [f32; 192] = [
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
const ZERO_BIAS_OFFSET_YCBCR_DC: [f32; 3] = [0.0, 0.0, 0.0];

/// Zero-bias offset for AC coefficients (per component).
/// From C++ jpegli quant.cc kZeroBiasOffsetYCbCrAC
const ZERO_BIAS_OFFSET_YCBCR_AC: [f32; 3] = [0.59082, 0.58146, 0.57988];

/// Zero-bias parameters for a single DCT block.
///
/// Zero-bias controls how coefficients are rounded toward zero during quantization.
/// A higher multiplier means more aggressive zeroing of small coefficients.
#[derive(Debug, Clone)]
pub struct ZeroBiasParams {
    /// Multiplier per coefficient (64 values)
    pub mul: [f32; 64],
    /// Offset per coefficient (64 values)
    pub offset: [f32; 64],
}

impl Default for ZeroBiasParams {
    fn default() -> Self {
        // Default: all zeros (matches C++ when adaptive quantization is disabled)
        Self {
            mul: [0.0; 64],
            offset: [0.0; 64],
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
    /// Also scales the offset by distance so that high quality (low distance)
    /// has less aggressive zeroing.
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

        // Blend offset based on distance.
        // At high quality (low distance), use threshold close to 0.5 (normal rounding).
        // At low quality (high distance), use jpegli's more aggressive thresholds.
        //
        // The jpegli AC offset (~0.59) is designed to zero more coefficients than normal.
        // We blend between 0.5 (normal) and full offset based on distance.
        let offset_blend = ((distance - DIST_HQ) / (DIST_LQ - DIST_HQ)).clamp(0.0, 1.0);

        let mut mul = [0.0f32; 64];
        let mut offset = [0.0f32; 64];

        for k in 0..64 {
            let lq = ZERO_BIAS_MUL_YCBCR_LQ[c * 64 + k];
            let hq = ZERO_BIAS_MUL_YCBCR_HQ[c * 64 + k];
            mul[k] = mix_lq * lq + mix_hq * hq;

            if k == 0 {
                // DC: keep original offset (0.0)
                offset[k] = ZERO_BIAS_OFFSET_YCBCR_DC[c];
            } else {
                // AC: blend between 0.5 (normal rounding) and jpegli offset (~0.59)
                // This ensures we never preserve MORE coefficients than normal rounding
                let jpegli_offset = ZERO_BIAS_OFFSET_YCBCR_AC[c];
                offset[k] = 0.5 + (jpegli_offset - 0.5) * offset_blend;
            }
        }

        Self { mul, offset }
    }
}

/// Quantize a block of DCT coefficients with zero-biasing.
///
/// This matches C++ jpegli's quantization behavior where small coefficients
/// are biased toward zero to improve compression.
///
/// The threshold is: `offset + mul * aq_strength`
/// - If `|coeff/quant| >= threshold`: round normally
/// - Else: set to 0
///
/// # Arguments
/// * `coeffs` - DCT coefficients (64 values)
/// * `quant` - Quantization table values (64 values)
/// * `zero_bias` - Zero-bias parameters
/// * `aq_strength` - Per-block adaptive quantization strength (0.0-0.2)
pub fn quantize_block_with_zero_bias(
    coeffs: &[f32; 64],
    quant: &[u16; 64],
    zero_bias: &ZeroBiasParams,
    aq_strength: f32,
) -> [i16; 64] {
    let mut result = [0i16; 64];

    for k in 0..64 {
        let q = quant[k] as f32;
        let qval = coeffs[k] / q;

        // Threshold for zeroing: offset + mul * aq_strength
        let threshold = zero_bias.offset[k] + zero_bias.mul[k] * aq_strength;

        if qval.abs() >= threshold {
            result[k] = qval.round() as i16;
        }
        // else result[k] stays 0
    }
    result
}

/// Quantize a block of DCT coefficients (simple version, no zero-bias).
pub fn quantize_block_simple(coeffs: &[f32; 64], quant: &[u16; 64]) -> [i16; 64] {
    let mut result = [0i16; 64];
    for k in 0..64 {
        result[k] = (coeffs[k] / quant[k] as f32).round() as i16;
    }
    result
}

/// Convert traditional JPEG quality (1-100) to butteraugli distance.
///
/// This approximates jpegli's quality-to-distance mapping.
pub fn quality_to_distance(quality: u8) -> f32 {
    let q = quality.clamp(1, 100) as f32;
    if q >= 100.0 {
        0.01
    } else if q >= 95.0 {
        // Very high quality: steep curve
        (100.0 - q) * 0.1
    } else if q >= 70.0 {
        // High to medium: moderate curve
        0.5 + (95.0 - q) * 0.06
    } else {
        // Low quality: steep increase
        2.0 + (70.0 - q) * 0.15
    }
}

/// mozjpeg ImageMagick-style chrominance table (index 3)
/// Uses same values as luma table for better quality.
/// Values >255 are clamped AFTER quality scaling, not in the base table.
const MOZJPEG_CHROMA_QUANT: [u16; 64] = [
    16, 16, 16, 18, 25, 37, 56, 85,
    16, 17, 20, 27, 34, 40, 53, 75,
    16, 20, 24, 31, 43, 62, 91, 135,
    18, 27, 31, 40, 53, 74, 106, 156,
    25, 34, 43, 53, 69, 94, 131, 189,
    37, 40, 62, 74, 94, 124, 169, 238,
    56, 53, 91, 106, 131, 169, 226, 311,  // 311 is valid, will clamp after scaling
    85, 75, 135, 156, 189, 238, 311, 418, // 311, 418 are valid, will clamp after scaling
];

/// Quantization table for a single component
#[derive(Clone, Debug)]
pub struct QuantTable {
    /// Quantization values in natural order
    pub values: [u16; 64],
    /// Table slot (0-3)
    pub slot: u8,
}

impl QuantTable {
    /// Create a quantization table from an array
    pub fn new(values: [u16; 64], slot: u8) -> Self {
        Self { values, slot }
    }

    /// Create standard luminance table at given quality
    pub fn luma_standard(quality: u8) -> Self {
        Self::from_base_table(&STD_LUMA_QUANT, quality, 0)
    }

    /// Create standard chrominance table at given quality
    pub fn chroma_standard(quality: u8) -> Self {
        Self::from_base_table(&STD_CHROMA_QUANT, quality, 1)
    }

    /// Create mozjpeg-style luminance table at given quality
    pub fn luma_mozjpeg(quality: u8) -> Self {
        Self::from_base_table(&MOZJPEG_LUMA_QUANT, quality, 0)
    }

    /// Create mozjpeg-style chrominance table at given quality
    pub fn chroma_mozjpeg(quality: u8) -> Self {
        Self::from_base_table(&MOZJPEG_CHROMA_QUANT, quality, 1)
    }

    /// Scale a base quantization table by quality factor
    fn from_base_table(base: &[u16; 64], quality: u8, slot: u8) -> Self {
        let quality = quality.clamp(1, 100);

        // JPEG quality scaling formula
        let scale = if quality < 50 {
            5000 / quality as u32
        } else {
            200 - 2 * quality as u32
        };

        let mut values = [0u16; 64];
        for i in 0..64 {
            let val = (base[i] as u32 * scale + 50) / 100;
            values[i] = val.clamp(1, 255) as u16;
        }

        Self { values, slot }
    }

    /// Get quantization value at zigzag position
    #[inline]
    pub fn at_zigzag(&self, pos: usize) -> u16 {
        self.values[crate::consts::ZIGZAG[pos]]
    }
}

/// Quantization table set for Y, Cb, Cr components
#[derive(Clone)]
pub struct QuantTableSet {
    /// Luminance quantization table
    pub luma: QuantTable,
    /// Chrominance quantization table
    pub chroma: QuantTable,
}

impl QuantTableSet {
    /// Create standard tables at given quality
    pub fn standard(quality: u8) -> Self {
        Self {
            luma: QuantTable::luma_standard(quality),
            chroma: QuantTable::chroma_standard(quality),
        }
    }

    /// Create mozjpeg-style tables at given quality
    pub fn mozjpeg(quality: u8) -> Self {
        Self {
            luma: QuantTable::luma_mozjpeg(quality),
            chroma: QuantTable::chroma_mozjpeg(quality),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_scaling() {
        // Q50 should give scale factor of 100 (no change)
        let q50 = QuantTable::luma_standard(50);
        assert_eq!(q50.values[0], STD_LUMA_QUANT[0]);

        // Q100 should give scale factor of 0 (minimum values)
        let q100 = QuantTable::luma_standard(100);
        assert_eq!(q100.values[0], 1); // All values clamped to 1

        // Q1 should give large values
        let q1 = QuantTable::luma_standard(1);
        assert!(q1.values[0] > 100);
    }
}
