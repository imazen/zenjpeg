//! Forward DCT (Discrete Cosine Transform) for JPEG encoding
//!
//! This module implements the Loeffler-Ligtenberg-Moschytz algorithm for 8x8 DCT,
//! matching mozjpeg's jfdctint.c (integer slow DCT).
//!
//! The algorithm uses 12 multiplies and 32 adds per 1-D DCT.
//! A 2-D DCT is done by 1-D DCT on rows followed by 1-D DCT on columns.
//!
//! # ⚠️ CRITICAL: OUTPUT SCALING ⚠️
//!
//! The output is scaled up by a factor of 8 compared to a true DCT!
//! This is intentional and matches libjpeg/mozjpeg behavior.
//!
//! - For non-trellis path: descale with `(coef + 4) >> 3` before quantization
//! - For trellis path: pass scaled values directly (trellis multiplies qtable by 8)
//!
//! Reference: C. Loeffler, A. Ligtenberg and G. Moschytz,
//! "Practical Fast 1-D DCT Algorithms with 11 Multiplications",
//! Proc. ICASSP 1989, pp. 988-991.

use crate::consts::{DCTSIZE, DCTSIZE2};

// Fixed-point constants for 13-bit precision (CONST_BITS = 13)
const CONST_BITS: i32 = 13;
const PASS1_BITS: i32 = 2;

// Pre-calculated fixed-point constants: FIX(x) = (x * (1 << CONST_BITS) + 0.5)
const FIX_0_298631336: i32 = 2446;  // FIX(0.298631336)
const FIX_0_390180644: i32 = 3196;  // FIX(0.390180644)
const FIX_0_541196100: i32 = 4433;  // FIX(0.541196100)
const FIX_0_765366865: i32 = 6270;  // FIX(0.765366865)
const FIX_0_899976223: i32 = 7373;  // FIX(0.899976223)
const FIX_1_175875602: i32 = 9633;  // FIX(1.175875602)
const FIX_1_501321110: i32 = 12299; // FIX(1.501321110)
const FIX_1_847759065: i32 = 15137; // FIX(1.847759065)
const FIX_1_961570560: i32 = 16069; // FIX(1.961570560)
const FIX_2_053119869: i32 = 16819; // FIX(2.053119869)
const FIX_2_562915447: i32 = 20995; // FIX(2.562915447)
const FIX_3_072711026: i32 = 25172; // FIX(3.072711026)

/// DESCALE: Right-shift with rounding (used to remove fixed-point scaling)
#[inline]
fn descale(x: i32, n: i32) -> i32 {
    // Round by adding 2^(n-1) before shifting
    (x + (1 << (n - 1))) >> n
}

/// Forward 8x8 DCT using Loeffler algorithm (integer version).
///
/// # ⚠️ CRITICAL: OUTPUT SCALING ⚠️
///
/// Output is scaled by factor of 8 (intentional, matches mozjpeg).
///
/// - For non-trellis: use `descale_for_quant()` before quantization
/// - For trellis: pass directly to `trellis_quantize_block()`
///
/// # Arguments
/// * `samples` - Input 8x8 block of level-shifted pixels (i16, typically -128 to 127)
/// * `coeffs` - Output 8x8 block of DCT coefficients (scaled by 8)
pub fn forward_dct_8x8_int(samples: &[i16; DCTSIZE2], coeffs: &mut [i16; DCTSIZE2]) {
    // Work buffer (we modify in place across both passes)
    let mut data = [0i32; DCTSIZE2];

    // Convert input to i32 for processing
    for i in 0..DCTSIZE2 {
        data[i] = samples[i] as i32;
    }

    // Pass 1: process rows
    // Results are scaled up by sqrt(8) and by 2^PASS1_BITS
    for row in 0..DCTSIZE {
        let base = row * DCTSIZE;

        let tmp0 = data[base] + data[base + 7];
        let tmp7 = data[base] - data[base + 7];
        let tmp1 = data[base + 1] + data[base + 6];
        let tmp6 = data[base + 1] - data[base + 6];
        let tmp2 = data[base + 2] + data[base + 5];
        let tmp5 = data[base + 2] - data[base + 5];
        let tmp3 = data[base + 3] + data[base + 4];
        let tmp4 = data[base + 3] - data[base + 4];

        // Even part (per Loeffler figure 1)
        let tmp10 = tmp0 + tmp3;
        let tmp13 = tmp0 - tmp3;
        let tmp11 = tmp1 + tmp2;
        let tmp12 = tmp1 - tmp2;

        data[base] = (tmp10 + tmp11) << PASS1_BITS;
        data[base + 4] = (tmp10 - tmp11) << PASS1_BITS;

        let z1 = (tmp12 + tmp13) * FIX_0_541196100;
        data[base + 2] = descale(z1 + tmp13 * FIX_0_765366865, CONST_BITS - PASS1_BITS);
        data[base + 6] = descale(z1 + tmp12 * (-FIX_1_847759065), CONST_BITS - PASS1_BITS);

        // Odd part (per Loeffler figure 8)
        let z1 = tmp4 + tmp7;
        let z2 = tmp5 + tmp6;
        let z3 = tmp4 + tmp6;
        let z4 = tmp5 + tmp7;
        let z5 = (z3 + z4) * FIX_1_175875602; // sqrt(2) * c3

        let tmp4 = tmp4 * FIX_0_298631336; // sqrt(2) * (-c1+c3+c5-c7)
        let tmp5 = tmp5 * FIX_2_053119869; // sqrt(2) * ( c1+c3-c5+c7)
        let tmp6 = tmp6 * FIX_3_072711026; // sqrt(2) * ( c1+c3+c5-c7)
        let tmp7 = tmp7 * FIX_1_501321110; // sqrt(2) * ( c1+c3-c5-c7)
        let z1 = z1 * (-FIX_0_899976223);  // sqrt(2) * ( c7-c3)
        let z2 = z2 * (-FIX_2_562915447);  // sqrt(2) * (-c1-c3)
        let z3 = z3 * (-FIX_1_961570560) + z5; // sqrt(2) * (-c3-c5)
        let z4 = z4 * (-FIX_0_390180644) + z5; // sqrt(2) * ( c5-c3)

        data[base + 7] = descale(tmp4 + z1 + z3, CONST_BITS - PASS1_BITS);
        data[base + 5] = descale(tmp5 + z2 + z4, CONST_BITS - PASS1_BITS);
        data[base + 3] = descale(tmp6 + z2 + z3, CONST_BITS - PASS1_BITS);
        data[base + 1] = descale(tmp7 + z1 + z4, CONST_BITS - PASS1_BITS);
    }

    // Pass 2: process columns
    // We remove PASS1_BITS scaling but leave results scaled by factor of 8
    for col in 0..DCTSIZE {
        let tmp0 = data[col] + data[DCTSIZE * 7 + col];
        let tmp7 = data[col] - data[DCTSIZE * 7 + col];
        let tmp1 = data[DCTSIZE + col] + data[DCTSIZE * 6 + col];
        let tmp6 = data[DCTSIZE + col] - data[DCTSIZE * 6 + col];
        let tmp2 = data[DCTSIZE * 2 + col] + data[DCTSIZE * 5 + col];
        let tmp5 = data[DCTSIZE * 2 + col] - data[DCTSIZE * 5 + col];
        let tmp3 = data[DCTSIZE * 3 + col] + data[DCTSIZE * 4 + col];
        let tmp4 = data[DCTSIZE * 3 + col] - data[DCTSIZE * 4 + col];

        // Even part
        let tmp10 = tmp0 + tmp3;
        let tmp13 = tmp0 - tmp3;
        let tmp11 = tmp1 + tmp2;
        let tmp12 = tmp1 - tmp2;

        data[col] = descale(tmp10 + tmp11, PASS1_BITS);
        data[DCTSIZE * 4 + col] = descale(tmp10 - tmp11, PASS1_BITS);

        let z1 = (tmp12 + tmp13) * FIX_0_541196100;
        data[DCTSIZE * 2 + col] = descale(z1 + tmp13 * FIX_0_765366865, CONST_BITS + PASS1_BITS);
        data[DCTSIZE * 6 + col] = descale(z1 + tmp12 * (-FIX_1_847759065), CONST_BITS + PASS1_BITS);

        // Odd part
        let z1 = tmp4 + tmp7;
        let z2 = tmp5 + tmp6;
        let z3 = tmp4 + tmp6;
        let z4 = tmp5 + tmp7;
        let z5 = (z3 + z4) * FIX_1_175875602;

        let tmp4 = tmp4 * FIX_0_298631336;
        let tmp5 = tmp5 * FIX_2_053119869;
        let tmp6 = tmp6 * FIX_3_072711026;
        let tmp7 = tmp7 * FIX_1_501321110;
        let z1 = z1 * (-FIX_0_899976223);
        let z2 = z2 * (-FIX_2_562915447);
        let z3 = z3 * (-FIX_1_961570560) + z5;
        let z4 = z4 * (-FIX_0_390180644) + z5;

        data[DCTSIZE * 7 + col] = descale(tmp4 + z1 + z3, CONST_BITS + PASS1_BITS);
        data[DCTSIZE * 5 + col] = descale(tmp5 + z2 + z4, CONST_BITS + PASS1_BITS);
        data[DCTSIZE * 3 + col] = descale(tmp6 + z2 + z3, CONST_BITS + PASS1_BITS);
        data[DCTSIZE + col] = descale(tmp7 + z1 + z4, CONST_BITS + PASS1_BITS);
    }

    // Copy results to output
    for i in 0..DCTSIZE2 {
        coeffs[i] = data[i] as i16;
    }
}

/// Descale DCT coefficients for non-trellis quantization.
///
/// The integer DCT outputs values scaled by 8. For standard quantization
/// (without trellis), we need to remove this scaling first.
///
/// # Arguments
/// * `coeffs` - DCT coefficients (scaled by 8)
/// * `output` - Descaled coefficients for quantization
#[inline]
pub fn descale_for_quant(coeffs: &[i16; DCTSIZE2], output: &mut [i32; DCTSIZE2]) {
    for i in 0..DCTSIZE2 {
        // (x + 4) >> 3 = divide by 8 with rounding
        output[i] = ((coeffs[i] as i32) + 4) >> 3;
    }
}

/// Scale DCT coefficients for trellis (convert i16 to i32).
///
/// For trellis quantization, we pass the 8x-scaled values directly.
/// Trellis internally multiplies qtable by 8 to match.
///
/// # Arguments
/// * `coeffs` - DCT coefficients (scaled by 8, as i16)
/// * `output` - Same values as i32 for trellis
#[inline]
pub fn scale_for_trellis(coeffs: &[i16; DCTSIZE2], output: &mut [i32; DCTSIZE2]) {
    for i in 0..DCTSIZE2 {
        output[i] = coeffs[i] as i32;
    }
}

/// Quantize DCT coefficients using integer arithmetic.
///
/// For non-trellis path. Input should be descaled DCT (not scaled by 8).
///
/// # Arguments
/// * `coeffs` - Descaled DCT coefficients
/// * `quant` - Quantization table
/// * `output` - Quantized coefficients
pub fn quantize_block_int(coeffs: &[i32; DCTSIZE2], quant: &[u16; DCTSIZE2], output: &mut [i16; DCTSIZE2]) {
    for i in 0..DCTSIZE2 {
        let c = coeffs[i];
        let q = quant[i] as i32;
        // Symmetric rounding: (|c| + q/2) / q, with sign preserved
        output[i] = if c >= 0 {
            ((c + q / 2) / q) as i16
        } else {
            ((c - q / 2) / q) as i16
        };
    }
}

// ============================================================================
// Legacy float-based functions (for backwards compatibility during transition)
// ============================================================================

/// Forward 8x8 DCT on level-shifted input (LEGACY - float version)
///
/// Takes an 8x8 block of pixels (level-shifted by -128) and produces
/// DCT coefficients.
///
/// **DEPRECATED**: Use `forward_dct_8x8_int` for better precision and compression.
pub fn forward_dct_8x8(block: &[i16; 64]) -> [f32; 64] {
    let mut output = [0.0f32; 64];

    // Reference implementation using direct DCT formula
    for v in 0..8 {
        for u in 0..8 {
            let mut sum = 0.0f32;

            for y in 0..8 {
                for x in 0..8 {
                    let pixel = block[y * 8 + x] as f32;
                    let cu = if u == 0 {
                        1.0 / std::f32::consts::SQRT_2
                    } else {
                        1.0
                    };
                    let cv = if v == 0 {
                        1.0 / std::f32::consts::SQRT_2
                    } else {
                        1.0
                    };

                    let cos_u =
                        ((2.0 * x as f32 + 1.0) * u as f32 * std::f32::consts::PI / 16.0).cos();
                    let cos_v =
                        ((2.0 * y as f32 + 1.0) * v as f32 * std::f32::consts::PI / 16.0).cos();

                    sum += cu * cv * pixel * cos_u * cos_v;
                }
            }

            // Scale by 1/4 (standard normalization)
            output[v * 8 + u] = sum / 4.0;
        }
    }

    output
}

/// Quantize DCT coefficients (LEGACY - float version)
pub fn quantize_block(dct: &[f32; 64], quant: &[u16; 64]) -> [i16; 64] {
    let mut output = [0i16; 64];

    for i in 0..64 {
        let q = quant[i] as f32;
        output[i] = (dct[i] / q).round() as i16;
    }

    output
}

/// Level-shift a block of pixels for DCT (subtract 128)
pub fn level_shift(pixels: &[u8; 64]) -> [i16; 64] {
    let mut output = [0i16; 64];
    for i in 0..64 {
        output[i] = pixels[i] as i16 - 128;
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dct_dc_only() {
        // A uniform block should have only DC component
        let block = [0i16; 64]; // level-shifted 128 -> 0
        let dct = forward_dct_8x8(&block);

        // DC should be 0 for uniform 128 block (after level shift)
        assert!(dct[0].abs() < 0.01, "DC = {}", dct[0]);

        // All AC should be 0
        for i in 1..64 {
            assert!(dct[i].abs() < 0.01, "AC[{}] = {}", i, dct[i]);
        }
    }

    #[test]
    fn test_integer_dct_dc_only() {
        // A uniform block should have only DC component
        let samples = [0i16; 64]; // level-shifted 128 -> 0
        let mut coeffs = [0i16; 64];
        forward_dct_8x8_int(&samples, &mut coeffs);

        // DC should be 0 for uniform block (after level shift)
        assert_eq!(coeffs[0], 0, "DC = {}", coeffs[0]);

        // All AC should be 0
        for i in 1..64 {
            assert_eq!(coeffs[i], 0, "AC[{}] = {}", i, coeffs[i]);
        }
    }

    #[test]
    fn test_integer_dct_white_block() {
        // All-white block (255 - 128 = 127)
        let samples = [127i16; 64];
        let mut coeffs = [0i16; 64];
        forward_dct_8x8_int(&samples, &mut coeffs);

        // DC should be large positive (127 * 8 for uniform block)
        // Expected: 127 * 8 = 1016 (the 8x scaling)
        assert!(coeffs[0] > 900, "DC = {} (expected ~1016)", coeffs[0]);

        // All AC should be 0 (uniform block)
        for i in 1..64 {
            assert_eq!(coeffs[i], 0, "AC[{}] = {} (expected 0)", i, coeffs[i]);
        }
    }

    #[test]
    fn test_integer_dct_gradient() {
        // Horizontal gradient block
        let mut samples = [0i16; 64];
        for row in 0..8 {
            for col in 0..8 {
                samples[row * 8 + col] = (col as i16 * 16) - 64; // -64 to 48
            }
        }

        let mut coeffs = [0i16; 64];
        forward_dct_8x8_int(&samples, &mut coeffs);

        // Should have significant AC components
        let ac_energy: i64 = coeffs[1..].iter().map(|&c| (c as i64) * (c as i64)).sum();
        assert!(ac_energy > 1000, "Gradient should have AC energy, got {}", ac_energy);
    }

    #[test]
    fn test_quantize_block_int() {
        let coeffs = [100i32, 50, -30, 0, 200, -150, 10, 5,
                      0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0];
        let quant = [16u16; 64];
        let mut output = [0i16; 64];

        quantize_block_int(&coeffs, &quant, &mut output);

        // 100 / 16 = 6.25 -> 6
        assert_eq!(output[0], 6);
        // 50 / 16 = 3.125 -> 3
        assert_eq!(output[1], 3);
        // -30 / 16 = -1.875 -> -2
        assert_eq!(output[2], -2);
        // 0 / 16 = 0
        assert_eq!(output[3], 0);
    }
}
