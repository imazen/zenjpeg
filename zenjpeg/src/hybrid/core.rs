//! Hybrid quantization: jpegli AQ + mozjpeg trellis
//!
//! This combines:
//! - jpegli's adaptive quantization (WHERE to spend bits - per-block AQ strength)
//! - mozjpeg's trellis quantization (HOW to spend bits - rate-distortion optimization)
//!
//! ## Theory
//!
//! AQ analyzes the image globally and assigns per-block "strength" values:
//! - Low AQ (smooth regions): preserve quality, keep more coefficients
//! - High AQ (textured regions): more compression acceptable, zero out more
//!
//! Trellis uses lambda for rate-distortion tradeoff:
//! - Higher lambda = favor rate (smaller files)
//! - Lower lambda = favor distortion (better quality)
//!
//! ## Integration
//!
//! We map AQ strength to trellis lambda adjustment:
//! - AQ=0 → lambda unchanged (baseline)
//! - AQ=0.5 → lambda ~2x (more aggressive compression)
//!
//! This is done by adjusting `lambda_log_scale1` in TrellisConfig:
//! - `new_scale1 = base_scale1 + aq_strength * AQ_LAMBDA_SCALE`

#[cfg(feature = "experimental-hybrid-trellis")]
use mozjpeg_rs::{
    consts::{AC_CHROMINANCE_BITS, AC_CHROMINANCE_VALUES, AC_LUMINANCE_BITS, AC_LUMINANCE_VALUES},
    huffman::{DerivedTable, HuffTable},
    trellis::trellis_quantize_block,
    TrellisConfig,
};

use crate::foundation::consts::DCT_BLOCK_SIZE;

/// Standard AC Huffman tables for trellis rate estimation.
///
/// Trellis quantization needs Huffman tables to estimate bit costs.
/// Using standard JPEG tables is a reasonable approximation when
/// optimized tables aren't available yet.
#[cfg(feature = "experimental-hybrid-trellis")]
pub struct StandardHuffmanTables {
    pub luma_ac: DerivedTable,
    pub chroma_ac: DerivedTable,
}

#[cfg(feature = "experimental-hybrid-trellis")]
impl StandardHuffmanTables {
    /// Create standard Huffman tables for trellis.
    pub fn new() -> Self {
        Self {
            luma_ac: Self::build_luma_ac(),
            chroma_ac: Self::build_chroma_ac(),
        }
    }

    fn build_luma_ac() -> DerivedTable {
        let mut htbl = HuffTable::default();
        htbl.bits.copy_from_slice(&AC_LUMINANCE_BITS);
        for (i, &v) in AC_LUMINANCE_VALUES.iter().enumerate() {
            htbl.huffval[i] = v;
        }
        DerivedTable::from_huff_table(&htbl, false)
            .expect("Standard AC luminance table should be valid")
    }

    fn build_chroma_ac() -> DerivedTable {
        let mut htbl = HuffTable::default();
        htbl.bits.copy_from_slice(&AC_CHROMINANCE_BITS);
        for (i, &v) in AC_CHROMINANCE_VALUES.iter().enumerate() {
            htbl.huffval[i] = v;
        }
        DerivedTable::from_huff_table(&htbl, false)
            .expect("Standard AC chrominance table should be valid")
    }
}

#[cfg(feature = "experimental-hybrid-trellis")]
impl Default for StandardHuffmanTables {
    fn default() -> Self {
        Self::new()
    }
}

/// Scale a quantization table by adaptive quantization strength.
///
/// Higher AQ strength means the block is "less important" (busy/textured area),
/// so we can use coarser quantization (higher quant values = fewer bits).
///
/// # Arguments
/// * `base_quant` - Base quantization table (u16\[64\])
/// * `aq_strength` - Per-block AQ strength from jpegli (typically 0.0 to ~0.5)
///
/// # Returns
/// Scaled quantization table where higher AQ strength = higher quant values
pub fn scale_quant_by_aq(
    base_quant: &[u16; DCT_BLOCK_SIZE],
    aq_strength: f32,
) -> [u16; DCT_BLOCK_SIZE] {
    let mut scaled = [0u16; DCT_BLOCK_SIZE];
    // aq_strength typically ranges from 0.0 to ~0.5
    // strength=0 → multiplier=1.0 (no change)
    // strength=0.5 → multiplier=1.5 (50% coarser)
    let multiplier = 1.0 + aq_strength;

    use wide::f32x8;
    let mul = f32x8::splat(multiplier);
    let one = f32x8::splat(1.0);
    let max_val = f32x8::splat(255.0);

    for chunk in 0..8 {
        let k = chunk * 8;
        let base_f = f32x8::from([
            base_quant[k] as f32,
            base_quant[k + 1] as f32,
            base_quant[k + 2] as f32,
            base_quant[k + 3] as f32,
            base_quant[k + 4] as f32,
            base_quant[k + 5] as f32,
            base_quant[k + 6] as f32,
            base_quant[k + 7] as f32,
        ]);
        let val = (base_f * mul).round().max(one).min(max_val);
        let arr: [f32; 8] = val.into();
        for j in 0..8 {
            scaled[k + j] = arr[j] as u16;
        }
    }

    scaled
}

/// Convert f32 DCT coefficients to i32 for trellis quantization.
///
/// jpegli and mozjpeg use different quantization formulas:
/// - jpegli: quantized = round(DCT * 8 / quantval) (DCT at 1/64 scale)
/// - mozjpeg trellis: quantized = round(DCT / (8 * quantval))
///
/// To make trellis produce the same quantized values as jpegli:
/// - We multiply DCT by 64: trellis sees round((64*DCT) / (8*quantval)) = round(DCT*8 / quantval)
/// - This compensates for both the 1/64 DCT scaling and the trellis's 8× divisor
pub fn dct_f32_to_i32(coeffs: &[f32; DCT_BLOCK_SIZE]) -> [i32; DCT_BLOCK_SIZE] {
    let mut result = [0i32; DCT_BLOCK_SIZE];

    use wide::f32x8;
    let scale = f32x8::splat(64.0);

    for chunk in 0..8 {
        let k = chunk * 8;
        let v = f32x8::from([
            coeffs[k],
            coeffs[k + 1],
            coeffs[k + 2],
            coeffs[k + 3],
            coeffs[k + 4],
            coeffs[k + 5],
            coeffs[k + 6],
            coeffs[k + 7],
        ]);
        let scaled = (v * scale).round();
        let arr: [f32; 8] = scaled.into();
        for j in 0..8 {
            result[k + j] = arr[j] as i32;
        }
    }

    result
}

/// How much to scale lambda per unit of AQ strength.
///
/// This controls the sensitivity of trellis to AQ:
/// - Higher value = more aggressive compression in textured regions
/// - Lower value = more uniform compression across image
///
/// Since lambda = 2^scale1 / ..., adding 1.0 to scale1 doubles lambda.
/// With AQ_LAMBDA_SCALE=2.0 and aq_strength=0.5, lambda increases by 2x.
#[cfg(feature = "experimental-hybrid-trellis")]
const AQ_LAMBDA_SCALE: f32 = 2.0;

/// Hybrid quantization: jpegli AQ + mozjpeg trellis.
///
/// This is the main entry point for hybrid encoding. It adjusts the trellis
/// lambda parameter based on AQ strength:
/// - Higher AQ (textured) → higher lambda → more aggressive compression
/// - Lower AQ (smooth) → lower lambda → preserve quality
///
/// # Arguments
/// * `dct_coeffs` - DCT coefficients in f32 (jpegli format)
/// * `base_quant` - Base quantization table
/// * `aq_strength` - Per-block AQ strength from jpegli (typically 0.0 to 0.5)
/// * `ac_table` - Huffman table for rate estimation
/// * `base_config` - Base trellis configuration (will be adjusted per-block)
///
/// # Returns
/// Quantized coefficients ready for entropy coding
#[cfg(feature = "experimental-hybrid-trellis")]
pub fn hybrid_quantize_block(
    dct_coeffs: &[f32; DCT_BLOCK_SIZE],
    base_quant: &[u16; DCT_BLOCK_SIZE],
    aq_strength: f32,
    ac_table: &DerivedTable,
    base_config: &TrellisConfig,
) -> [i16; DCT_BLOCK_SIZE] {
    // Adjust lambda based on AQ strength:
    // - Higher AQ → increase lambda_log_scale1 → higher lambda → favor compression
    // - aq_strength=0.5 with AQ_LAMBDA_SCALE=2.0 → +1.0 to scale1 → 2x lambda
    let mut config = *base_config;
    config.lambda_log_scale1 += aq_strength * AQ_LAMBDA_SCALE;

    // Convert f32 DCT to i32 (with 8x scaling to match trellis's 8x quant divisor)
    let dct_i32 = dct_f32_to_i32(dct_coeffs);

    // Run trellis quantization with AQ-adjusted lambda
    let mut quantized = [0i16; DCT_BLOCK_SIZE];
    trellis_quantize_block(&dct_i32, &mut quantized, base_quant, ac_table, &config);

    quantized
}

/// Hybrid quantization without trellis (for comparison/testing).
///
/// Scales quant table by AQ but uses simple rounding instead of trellis.
/// This isolates the AQ scaling effect from trellis optimization.
pub fn hybrid_quantize_block_simple(
    dct_coeffs: &[f32; DCT_BLOCK_SIZE],
    base_quant: &[u16; DCT_BLOCK_SIZE],
    aq_strength: f32,
) -> [i16; DCT_BLOCK_SIZE] {
    // 1. Scale quant table by AQ strength
    let scaled_quant = scale_quant_by_aq(base_quant, aq_strength);

    // 2. Simple quantization (divide and round)
    let mut quantized = [0i16; DCT_BLOCK_SIZE];

    use wide::f32x8;
    for chunk in 0..8 {
        let k = chunk * 8;
        let dct = f32x8::from([
            dct_coeffs[k],
            dct_coeffs[k + 1],
            dct_coeffs[k + 2],
            dct_coeffs[k + 3],
            dct_coeffs[k + 4],
            dct_coeffs[k + 5],
            dct_coeffs[k + 6],
            dct_coeffs[k + 7],
        ]);
        let q = f32x8::from([
            scaled_quant[k] as f32,
            scaled_quant[k + 1] as f32,
            scaled_quant[k + 2] as f32,
            scaled_quant[k + 3] as f32,
            scaled_quant[k + 4] as f32,
            scaled_quant[k + 5] as f32,
            scaled_quant[k + 6] as f32,
            scaled_quant[k + 7] as f32,
        ]);
        // DCT uses 1/64 scaling (matching C++), so multiply by 8/quant
        let eight = f32x8::splat(8.0);
        let val = (dct * eight / q).round();
        let arr: [f32; 8] = val.into();
        for j in 0..8 {
            quantized[k + j] = arr[j] as i16;
        }
    }

    quantized
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scale_quant_by_aq() {
        let base = [16u16; 64];

        // Zero AQ strength = no change
        let scaled = scale_quant_by_aq(&base, 0.0);
        assert_eq!(scaled[0], 16);

        // 0.5 AQ strength = 1.5x
        let scaled = scale_quant_by_aq(&base, 0.5);
        assert_eq!(scaled[0], 24);

        // 1.0 AQ strength = 2x
        let scaled = scale_quant_by_aq(&base, 1.0);
        assert_eq!(scaled[0], 32);
    }

    #[test]
    fn test_scale_quant_clamping() {
        let base = [200u16; 64];

        // Should clamp at 255
        let scaled = scale_quant_by_aq(&base, 0.5);
        assert_eq!(scaled[0], 255);

        let base_low = [1u16; 64];
        // Should never go below 1
        let scaled = scale_quant_by_aq(&base_low, 0.0);
        assert_eq!(scaled[0], 1);
    }

    #[test]
    fn test_dct_f32_to_i32() {
        // Function multiplies by 8 for trellis compatibility (see docstring)
        // 127.4 * 8 = 1019.2, rounds to 1019
        let f32_coeffs = [127.4f32; 64];
        let i32_coeffs = dct_f32_to_i32(&f32_coeffs);
        assert_eq!(i32_coeffs[0], 1019);

        // -127.6 * 8 = -1020.8, rounds to -1021
        let f32_coeffs = [-127.6f32; 64];
        let i32_coeffs = dct_f32_to_i32(&f32_coeffs);
        assert_eq!(i32_coeffs[0], -1021);
    }

    #[test]
    fn test_hybrid_quantize_simple() {
        // DC coefficient of 1024 with quant=16 and no AQ
        let mut dct = [0.0f32; 64];
        dct[0] = 1024.0;
        let base_quant = [16u16; 64];

        let quantized = hybrid_quantize_block_simple(&dct, &base_quant, 0.0);
        assert_eq!(quantized[0], 64); // 1024 / 16 = 64

        // With AQ strength 0.5, quant becomes 24
        let quantized = hybrid_quantize_block_simple(&dct, &base_quant, 0.5);
        assert_eq!(quantized[0], 43); // 1024 / 24 = 42.67 -> 43
    }
}
