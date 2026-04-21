//! SIMD-native data types for efficient block processing.
//!
//! These types store data in raw arrays for `Pod`/`Zeroable`/`const` compatibility.
//! Computation uses scalar loops (portable) or `magetypes` (archmage dispatch).
//! The array storage is layout-compatible with SIMD vectors (32-byte aligned).

#![allow(clippy::wrong_self_convention)] // to_* methods need &self for SIMD types

use archmage::prelude::*;
use magetypes::simd::generic::f32x8 as GenericF32x8;
use magetypes::simd::generic::i32x8 as GenericI32x8;

/// An 8x8 block stored as 8 rows of `[f32; 8]` for SIMD-native access.
///
/// This layout means:
/// - Each row can be loaded into a SIMD vector with a single aligned load
/// - Row-wise operations (DCT, quantization) are trivial
/// - 32-byte aligned for optimal SIMD access
///
/// # Safety
///
/// `Block8x8f` is `Pod` and `Zeroable` because:
/// - `[f32; 8]` is Pod (all f32 bit patterns are valid)
/// - The struct is `#[repr(C, align(32))]` with no padding
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C, align(32))]
pub struct Block8x8f {
    pub rows: [[f32; 8]; 8],
}

impl Block8x8f {
    pub const ZERO: Self = Self {
        rows: [[0.0; 8]; 8],
    };

    /// Create from a flat array (for compatibility with existing code)
    #[inline]
    pub fn from_array(arr: &[f32; 64]) -> Self {
        let mut rows = [[0.0f32; 8]; 8];
        for (row_idx, row) in rows.iter_mut().enumerate() {
            let start = row_idx * 8;
            *row = arr[start..start + 8].try_into().unwrap();
        }
        Self { rows }
    }

    /// Convert to a flat array (for compatibility with existing code)
    #[inline]
    pub fn to_array(&self) -> [f32; 64] {
        let mut arr = [0.0f32; 64];
        for (row_idx, row) in self.rows.iter().enumerate() {
            arr[row_idx * 8..row_idx * 8 + 8].copy_from_slice(row);
        }
        arr
    }

    /// Access a single coefficient
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.rows[row][col]
    }

    /// Set a single coefficient
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: f32) {
        self.rows[row][col] = value;
    }

    /// Multiply all elements by a scalar
    #[inline]
    pub fn scale(&self, factor: f32) -> Self {
        let mut result = Self::ZERO;
        for i in 0..8 {
            for j in 0..8 {
                result.rows[i][j] = self.rows[i][j] * factor;
            }
        }
        result
    }

    /// Element-wise multiply with another block
    #[inline]
    pub fn mul(&self, other: &Self) -> Self {
        let mut result = Self::ZERO;
        for i in 0..8 {
            for j in 0..8 {
                result.rows[i][j] = self.rows[i][j] * other.rows[i][j];
            }
        }
        result
    }

    /// Element-wise add
    #[inline]
    pub fn add(&self, other: &Self) -> Self {
        let mut result = Self::ZERO;
        for i in 0..8 {
            for j in 0..8 {
                result.rows[i][j] = self.rows[i][j] + other.rows[i][j];
            }
        }
        result
    }
}

impl Default for Block8x8f {
    fn default() -> Self {
        Self::ZERO
    }
}

/// An 8x8 block of i16 values stored as 8 rows of `[i16; 8]`.
///
/// Used for quantized DCT coefficients.
#[derive(Clone, Copy, Debug)]
#[repr(C, align(16))]
pub struct Block8x8i16 {
    pub rows: [[i16; 8]; 8],
}

impl Block8x8i16 {
    pub const ZERO: Self = Self { rows: [[0; 8]; 8] };

    /// Create from a flat array
    #[inline]
    pub fn from_array(arr: &[i16; 64]) -> Self {
        let mut rows = [[0i16; 8]; 8];
        for (row_idx, row) in rows.iter_mut().enumerate() {
            let start = row_idx * 8;
            *row = arr[start..start + 8].try_into().unwrap();
        }
        Self { rows }
    }

    /// Convert to a flat array
    #[inline]
    pub fn to_array(&self) -> [i16; 64] {
        let mut arr = [0i16; 64];
        for (row_idx, row) in self.rows.iter().enumerate() {
            arr[row_idx * 8..row_idx * 8 + 8].copy_from_slice(row);
        }
        arr
    }

    /// Access a single coefficient
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> i16 {
        self.rows[row][col]
    }
}

impl Default for Block8x8i16 {
    fn default() -> Self {
        Self::ZERO
    }
}

/// Quantization table stored in SIMD-friendly layout.
///
/// Pre-computes multipliers (8/value) for fast quantization.
/// The 8x factor compensates for DCT's 1/64 scaling (matching C++ jpegli).
#[derive(Clone, Debug)]
#[repr(C, align(32))]
pub struct QuantTableSimd {
    /// Multipliers for quantization (8.0 / quant_value)
    pub mul_rows: [[f32; 8]; 8],
    /// Original values for encoding the JPEG header
    pub values: [u16; 64],
}

/// Zero-bias parameters stored in SIMD-friendly layout.
///
/// Pre-computed thresholds for each coefficient position.
#[derive(Clone, Debug)]
#[repr(C, align(32))]
pub struct ZeroBiasSimd {
    /// offset\[k\] for each coefficient (8 rows of \[f32; 8\])
    pub offset_rows: [[f32; 8]; 8],
    /// mul\[k\] for each coefficient (8 rows of \[f32; 8\])
    pub mul_rows: [[f32; 8]; 8],
}

impl ZeroBiasSimd {
    /// Create from ZeroBiasParams
    pub fn from_params(params: &crate::quant::ZeroBiasParams) -> Self {
        let mut offset_rows = [[0.0f32; 8]; 8];
        let mut mul_rows = [[0.0f32; 8]; 8];
        for row in 0..8 {
            let start = row * 8;
            offset_rows[row] = params.offset[start..start + 8].try_into().unwrap();
            mul_rows[row] = params.mul[start..start + 8].try_into().unwrap();
        }
        Self {
            offset_rows,
            mul_rows,
        }
    }
}

impl QuantTableSimd {
    /// Create from u16 quantization values
    ///
    /// Computes 8.0/quant multipliers for fast quantization.
    /// The 8.0 factor compensates for DCT's 1/64 scaling (matching C++ jpegli).
    pub fn from_values(values: &[u16; 64]) -> Self {
        let mut mul_rows = [[0.0f32; 8]; 8];
        for row in 0..8 {
            let start = row * 8;
            for col in 0..8 {
                mul_rows[row][col] = 8.0 / values[start + col] as f32;
            }
        }
        Self {
            mul_rows,
            values: *values,
        }
    }

    /// Create from f32 quantization values
    ///
    /// Computes 8.0/quant multipliers for fast quantization.
    /// The 8.0 factor compensates for DCT's 1/64 scaling (matching C++ jpegli).
    pub fn from_f32_values(values: &[f32; 64]) -> Self {
        let mut mul_rows = [[0.0f32; 8]; 8];
        let mut u16_values = [0u16; 64];
        for row in 0..8 {
            let start = row * 8;
            for col in 0..8 {
                mul_rows[row][col] = 8.0 / values[start + col];
                u16_values[start + col] = values[start + col].round() as u16;
            }
        }
        Self {
            mul_rows,
            values: u16_values,
        }
    }

    /// Quantize a block using SIMD multiplication
    ///
    /// This is the core optimization: each row is one SIMD multiply with no load overhead.
    #[inline]
    pub fn quantize(&self, block: &Block8x8f) -> Block8x8i32 {
        let mut result = Block8x8i32::ZERO;
        for i in 0..8 {
            for j in 0..8 {
                result.rows[i][j] = (block.rows[i][j] * self.mul_rows[i][j]).round() as i32;
            }
        }
        result
    }

    /// Quantize a block with zero-bias, outputting directly in zigzag order.
    ///
    /// This fuses quantization and zigzag reordering into a single pass,
    /// eliminating the separate natural_to_zigzag_into call.
    #[inline]
    pub fn quantize_with_zero_bias_zigzag(
        &self,
        block: &Block8x8f,
        zero_bias: &ZeroBiasSimd,
        aq_strength: f32,
    ) -> [i16; 64] {
        quantize_block_zigzag(&self.mul_rows, block, zero_bias, aq_strength)
    }

    /// Quantize with a scaled zero-bias vector, outputting directly in zigzag order.
    ///
    /// The `zb_scale` multiplier scales the entire zero-bias threshold:
    /// with the usual formula `|qval| >= offset + mul * aq`, the scaled
    /// form becomes `|qval| >= zb_scale * (offset + mul * aq)`.
    ///
    /// `zb_scale == 1.0` is a strict identity — it produces the same
    /// output as [`Self::quantize_with_zero_bias_zigzag`] for the same
    /// input. Smaller values weaken the zero-bias rule, preserving more
    /// small AC coefficients.
    ///
    /// This is the retry-path entry point for boundary-RD (#91) — the
    /// non-retry hot loop still uses the unscaled variant for codegen
    /// stability (the retry only fires on triggered blocks).
    #[inline]
    pub fn quantize_with_scaled_zero_bias_zigzag(
        &self,
        block: &Block8x8f,
        zero_bias: &ZeroBiasSimd,
        zb_scale: f32,
        aq_strength: f32,
    ) -> [i16; 64] {
        quantize_block_zigzag_scaled(
            &self.mul_rows,
            block,
            zero_bias,
            zb_scale,
            aq_strength,
        )
    }

    #[inline]
    pub fn quantize_with_zero_bias(
        &self,
        block: &Block8x8f,
        zero_bias: &ZeroBiasSimd,
        aq_strength: f32,
    ) -> [i16; 64] {
        quantize_block(&self.mul_rows, block, zero_bias, aq_strength)
    }

    /// Quantize a block from a flat array with zero-bias using pre-computed SIMD tables.
    ///
    /// This avoids the Block8x8f conversion overhead by loading directly from the array.
    #[inline]
    pub fn quantize_array_with_zero_bias(
        &self,
        coeffs: &[f32; 64],
        zero_bias: &ZeroBiasSimd,
        aq_strength: f32,
    ) -> [i16; 64] {
        let mut result = [0i16; 64];
        for row in 0..8 {
            let k = row * 8;
            for col in 0..8 {
                let qval = coeffs[k + col] * self.mul_rows[row][col];
                let threshold =
                    zero_bias.offset_rows[row][col] + zero_bias.mul_rows[row][col] * aq_strength;
                if qval.abs() >= threshold {
                    result[k + col] = fast_round_i32(qval) as i16;
                }
            }
        }
        result
    }
}

/// An 8x8 block of i32 values stored as 8 rows of `[i32; 8]`.
///
/// Used as intermediate during quantization before conversion to i16.
#[derive(Clone, Copy, Debug)]
#[repr(C, align(32))]
pub struct Block8x8i32 {
    pub rows: [[i32; 8]; 8],
}

impl Block8x8i32 {
    pub const ZERO: Self = Self { rows: [[0; 8]; 8] };

    /// Convert to i16 block (with saturation)
    #[inline]
    pub fn to_i16(&self) -> Block8x8i16 {
        let mut result = Block8x8i16::ZERO;
        for i in 0..8 {
            for j in 0..8 {
                result.rows[i][j] = self.rows[i][j].clamp(-32768, 32767) as i16;
            }
        }
        result
    }

    /// Convert to flat i16 array
    #[inline]
    pub fn to_i16_array(&self) -> [i16; 64] {
        self.to_i16().to_array()
    }
}

impl Default for Block8x8i32 {
    fn default() -> Self {
        Self::ZERO
    }
}

/// Magetypes-generic quantize with zigzag output.
///
/// Uses f32x8 for threshold/multiply and i32x8 for blend/round.
/// On x86 AVX2+FMA this is true 256-bit; on NEON/WASM it's polyfilled via pairs.
#[magetypes(v3, neon, wasm128, scalar)]
#[inline(always)]
fn mage_quantize_block_zigzag(
    token: Token,
    block: &Block8x8f,
    mul_rows: &[[f32; 8]; 8],
    zero_bias: &ZeroBiasSimd,
    aq_strength: f32,
) -> [i16; 64] {
    use crate::foundation::consts::JPEG_ZIGZAG_ORDER;

    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    #[allow(non_camel_case_types)]
    type i32x8 = GenericI32x8<Token>;

    let aq_m = f32x8::splat(token, aq_strength);
    let zero_i32 = i32x8::zero(token);
    let mut result = [0i16; 64];

    for row in 0..8 {
        let block_m = f32x8::from_array(token, block.rows[row]);
        let mul_m = f32x8::from_array(token, mul_rows[row]);
        let offset_m = f32x8::from_array(token, zero_bias.offset_rows[row]);
        let bias_mul_m = f32x8::from_array(token, zero_bias.mul_rows[row]);

        let qval = block_m * mul_m;
        let threshold = bias_mul_m.mul_add(aq_m, offset_m);
        let abs_qval = qval.abs();
        let mask = abs_qval.simd_ge(threshold);
        let rounded = qval.to_i32_round();
        let mask_i32 = mask.bitcast_to_i32();
        let blended = i32x8::blend(mask_i32, rounded, zero_i32);

        let arr = blended.to_array();
        let k = row * 8;

        result[JPEG_ZIGZAG_ORDER[k] as usize] = arr[0] as i16;
        result[JPEG_ZIGZAG_ORDER[k + 1] as usize] = arr[1] as i16;
        result[JPEG_ZIGZAG_ORDER[k + 2] as usize] = arr[2] as i16;
        result[JPEG_ZIGZAG_ORDER[k + 3] as usize] = arr[3] as i16;
        result[JPEG_ZIGZAG_ORDER[k + 4] as usize] = arr[4] as i16;
        result[JPEG_ZIGZAG_ORDER[k + 5] as usize] = arr[5] as i16;
        result[JPEG_ZIGZAG_ORDER[k + 6] as usize] = arr[6] as i16;
        result[JPEG_ZIGZAG_ORDER[k + 7] as usize] = arr[7] as i16;
    }

    result
}

/// Magetypes-generic quantize, natural order output.
#[magetypes(v3, neon, wasm128, scalar)]
#[inline(always)]
fn mage_quantize_block(
    token: Token,
    block: &Block8x8f,
    mul_rows: &[[f32; 8]; 8],
    zero_bias: &ZeroBiasSimd,
    aq_strength: f32,
) -> [i16; 64] {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    #[allow(non_camel_case_types)]
    type i32x8 = GenericI32x8<Token>;

    let aq_m = f32x8::splat(token, aq_strength);
    let zero_i32 = i32x8::zero(token);
    let mut result = [0i16; 64];

    for row in 0..8 {
        let block_m = f32x8::from_array(token, block.rows[row]);
        let mul_m = f32x8::from_array(token, mul_rows[row]);
        let offset_m = f32x8::from_array(token, zero_bias.offset_rows[row]);
        let bias_mul_m = f32x8::from_array(token, zero_bias.mul_rows[row]);

        let qval = block_m * mul_m;
        let threshold = bias_mul_m.mul_add(aq_m, offset_m);
        let abs_qval = qval.abs();
        let mask = abs_qval.simd_ge(threshold);
        let rounded = qval.to_i32_round();
        let mask_i32 = mask.bitcast_to_i32();
        let blended = i32x8::blend(mask_i32, rounded, zero_i32);

        let arr = blended.to_array();
        let k = row * 8;

        result[k] = arr[0] as i16;
        result[k + 1] = arr[1] as i16;
        result[k + 2] = arr[2] as i16;
        result[k + 3] = arr[3] as i16;
        result[k + 4] = arr[4] as i16;
        result[k + 5] = arr[5] as i16;
        result[k + 6] = arr[6] as i16;
        result[k + 7] = arr[7] as i16;
    }

    result
}

/// Round to nearest i32, matching wide::f32x8::fast_round_int behavior.
#[inline(always)]
fn fast_round_i32(v: f32) -> i32 {
    // fast_round_int adds/subtracts magic constant to trigger rounding,
    // which is equivalent to roundf for values in the f32 integer range.
    v.round() as i32
}

/// Dispatching quantize with zigzag — magetypes multi-platform dispatch.
#[inline]
fn quantize_block_zigzag(
    mul_rows: &[[f32; 8]; 8],
    block: &Block8x8f,
    zero_bias: &ZeroBiasSimd,
    aq_strength: f32,
) -> [i16; 64] {
    incant!(mage_quantize_block_zigzag(
        block,
        mul_rows,
        zero_bias,
        aq_strength
    ))
}

/// Dispatching scaled-zero-bias quantize with zigzag — magetypes
/// multi-platform dispatch. Only used by boundary-RD's retry path.
#[inline]
fn quantize_block_zigzag_scaled(
    mul_rows: &[[f32; 8]; 8],
    block: &Block8x8f,
    zero_bias: &ZeroBiasSimd,
    zb_scale: f32,
    aq_strength: f32,
) -> [i16; 64] {
    incant!(mage_quantize_block_zigzag_scaled(
        block,
        mul_rows,
        zero_bias,
        zb_scale,
        aq_strength
    ))
}

/// Magetypes-generic quantize with zigzag output and a scaled zero-bias
/// vector. Same SIMD shape as `mage_quantize_block_zigzag`, with one
/// extra f32x8 splat + multiply on the threshold. When `zb_scale ==
/// 1.0`, the resulting threshold is bit-identical to the unscaled form
/// (the FP multiply by 1.0 is an identity in IEEE-754).
#[magetypes(v3, neon, wasm128, scalar)]
#[inline(always)]
fn mage_quantize_block_zigzag_scaled(
    token: Token,
    block: &Block8x8f,
    mul_rows: &[[f32; 8]; 8],
    zero_bias: &ZeroBiasSimd,
    zb_scale: f32,
    aq_strength: f32,
) -> [i16; 64] {
    use crate::foundation::consts::JPEG_ZIGZAG_ORDER;

    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    #[allow(non_camel_case_types)]
    type i32x8 = GenericI32x8<Token>;

    let aq_m = f32x8::splat(token, aq_strength);
    let zb_m = f32x8::splat(token, zb_scale);
    let zero_i32 = i32x8::zero(token);
    let mut result = [0i16; 64];

    for row in 0..8 {
        let block_m = f32x8::from_array(token, block.rows[row]);
        let mul_m = f32x8::from_array(token, mul_rows[row]);
        let offset_m = f32x8::from_array(token, zero_bias.offset_rows[row]);
        let bias_mul_m = f32x8::from_array(token, zero_bias.mul_rows[row]);

        let qval = block_m * mul_m;
        let threshold = bias_mul_m.mul_add(aq_m, offset_m) * zb_m;
        let abs_qval = qval.abs();
        let mask = abs_qval.simd_ge(threshold);
        let rounded = qval.to_i32_round();
        let mask_i32 = mask.bitcast_to_i32();
        let blended = i32x8::blend(mask_i32, rounded, zero_i32);

        let arr = blended.to_array();
        let k = row * 8;

        result[JPEG_ZIGZAG_ORDER[k] as usize] = arr[0] as i16;
        result[JPEG_ZIGZAG_ORDER[k + 1] as usize] = arr[1] as i16;
        result[JPEG_ZIGZAG_ORDER[k + 2] as usize] = arr[2] as i16;
        result[JPEG_ZIGZAG_ORDER[k + 3] as usize] = arr[3] as i16;
        result[JPEG_ZIGZAG_ORDER[k + 4] as usize] = arr[4] as i16;
        result[JPEG_ZIGZAG_ORDER[k + 5] as usize] = arr[5] as i16;
        result[JPEG_ZIGZAG_ORDER[k + 6] as usize] = arr[6] as i16;
        result[JPEG_ZIGZAG_ORDER[k + 7] as usize] = arr[7] as i16;
    }

    result
}

/// Dispatching quantize natural order — magetypes multi-platform dispatch.
#[inline]
fn quantize_block(
    mul_rows: &[[f32; 8]; 8],
    block: &Block8x8f,
    zero_bias: &ZeroBiasSimd,
    aq_strength: f32,
) -> [i16; 64] {
    incant!(mage_quantize_block(block, mul_rows, zero_bias, aq_strength))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build realistic test data for quantize dispatch testing.
    /// Returns (block, quant_table, zero_bias, aq_strength).
    fn quantize_test_data() -> (Block8x8f, QuantTableSimd, ZeroBiasSimd, f32) {
        // Simulate DCT coefficients: DC large, AC coefficients decaying
        let mut coeffs = [0.0f32; 64];
        for i in 0..64 {
            let row = i / 8;
            let col = i % 8;
            let freq = (row + col) as f32;
            // Mix of positive, negative, near-zero values
            coeffs[i] = (100.0 - freq * 8.0) * if i % 3 == 0 { -1.0 } else { 1.0 };
        }
        let block = Block8x8f::from_array(&coeffs);

        // Typical Q85 quant table values
        let mut qvals = [1u16; 64];
        for i in 0..64 {
            qvals[i] = ((i as u16 / 4) + 2).min(255);
        }
        let quant = QuantTableSimd::from_values(&qvals);

        // Realistic zero-bias params
        let mut bias_params = crate::quant::ZeroBiasParams {
            offset: [0.0; 64],
            mul: [0.0; 64],
        };
        for i in 0..64 {
            bias_params.offset[i] = 0.5;
            bias_params.mul[i] = 0.15;
        }
        let zero_bias = ZeroBiasSimd::from_params(&bias_params);

        (block, quant, zero_bias, 1.0)
    }

    /// Test that quantize_block_zigzag produces identical results across all
    /// SIMD dispatch tiers (AVX2+FMA, SSE2 fallback, scalar).
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_quantize_zigzag_dispatch_parity() {
        use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

        let (block, quant, zero_bias, aq) = quantize_test_data();

        // Get the reference result with all SIMD enabled
        let reference = quantize_block_zigzag(&quant.mul_rows, &block, &zero_bias, aq);

        let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
            let result = quantize_block_zigzag(&quant.mul_rows, &block, &zero_bias, aq);
            assert_eq!(
                result, reference,
                "quantize_block_zigzag mismatch at permutation: {perm}"
            );
        });
        eprintln!("quantize_zigzag: {report}");
        assert!(
            report.permutations_run >= 2,
            "expected at least 2 permutations"
        );
    }

    /// Test that quantize_block (natural order) produces identical results
    /// across all SIMD dispatch tiers.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_quantize_natural_dispatch_parity() {
        use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

        let (block, quant, zero_bias, aq) = quantize_test_data();

        let reference = quantize_block(&quant.mul_rows, &block, &zero_bias, aq);

        let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
            let result = quantize_block(&quant.mul_rows, &block, &zero_bias, aq);
            assert_eq!(
                result, reference,
                "quantize_block mismatch at permutation: {perm}"
            );
        });
        eprintln!("quantize_natural: {report}");
        assert!(
            report.permutations_run >= 2,
            "expected at least 2 permutations"
        );
    }

    /// Test that the public quantize_with_zero_bias_zigzag API works across
    /// all tiers (exercises the full dispatch chain through QuantTableSimd).
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_quantize_api_dispatch_parity() {
        use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

        let (block, quant, zero_bias, aq) = quantize_test_data();

        let ref_zigzag = quant.quantize_with_zero_bias_zigzag(&block, &zero_bias, aq);
        let ref_natural = quant.quantize_with_zero_bias(&block, &zero_bias, aq);

        let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
            let zigzag = quant.quantize_with_zero_bias_zigzag(&block, &zero_bias, aq);
            let natural = quant.quantize_with_zero_bias(&block, &zero_bias, aq);
            assert_eq!(zigzag, ref_zigzag, "zigzag API mismatch at: {perm}");
            assert_eq!(natural, ref_natural, "natural API mismatch at: {perm}");
        });
        eprintln!("quantize_api: {report}");
    }

    /// Boundary-RD retry path identity: `quantize_with_scaled_zero_bias_zigzag`
    /// with `zb_scale == 1.0` must be byte-identical to the non-scaled
    /// variant. This is the invariant that keeps `BoundaryRdConfig::default()`
    /// (which resolves to `shrink_zb = 1.0`) byte-identical to the
    /// pre-Task-3 retry output.
    #[test]
    fn scaled_zero_bias_scale_one_is_identity() {
        let (block, quant, zero_bias, aq) = quantize_test_data();
        let reference = quant.quantize_with_zero_bias_zigzag(&block, &zero_bias, aq);
        let scaled = quant.quantize_with_scaled_zero_bias_zigzag(&block, &zero_bias, 1.0, aq);
        assert_eq!(
            scaled, reference,
            "scaled(zb_scale=1.0) must match unscaled quantize"
        );
    }

    /// A smaller `zb_scale` must produce *different* output whenever
    /// the threshold actually bites — if this asserts equal, the retry
    /// knob is a no-op and the Task-3 wiring is broken.
    ///
    /// We build a block of low-magnitude AC coefficients deliberately
    /// close to the threshold so scaling the threshold by 0.25 (vs 1.0
    /// vs 2.0) measurably flips the `|qval| >= threshold` decision for
    /// at least one position.
    #[test]
    fn scaled_zero_bias_scale_below_one_differs() {
        // All-zero block except a handful of AC coeffs near the threshold.
        // With quant values = 16 and zero-bias params offset=0.5, mul=0.15,
        // the unscaled threshold is 0.5 + 0.15*1.0 = 0.65 in quant-space.
        // Since qval = coeff * (8.0 / 16) = coeff * 0.5, a coeff near 1.3
        // lands at qval ≈ 0.65 — exactly on the threshold.
        let mut coeffs = [0.0f32; 64];
        // Seed a mix of coeffs that straddle the threshold at
        // zb_scale=1.0 (threshold=0.65) vs zb_scale=0.25 (threshold=0.1625).
        coeffs[1] = 1.2;
        coeffs[2] = 1.0;
        coeffs[9] = 0.9;
        coeffs[16] = 0.6;
        coeffs[17] = 0.4;
        coeffs[24] = 0.3;
        let block = Block8x8f::from_array(&coeffs);

        let qvals = [16u16; 64];
        let quant = QuantTableSimd::from_values(&qvals);

        let bias_params = crate::quant::ZeroBiasParams {
            offset: [0.5; 64],
            mul: [0.15; 64],
        };
        let zero_bias = ZeroBiasSimd::from_params(&bias_params);
        let aq = 1.0f32;

        let reference = quant.quantize_with_zero_bias_zigzag(&block, &zero_bias, aq);
        let scaled = quant.quantize_with_scaled_zero_bias_zigzag(&block, &zero_bias, 0.25, aq);
        assert_ne!(
            scaled, reference,
            "zb_scale < 1.0 must measurably differ from the unscaled quantize \
             on near-threshold coefficients"
        );
    }

    /// Cross-tier parity for the scaled variant: every SIMD permutation
    /// must agree on the output. Protects against the same class of bug
    /// as `test_quantize_zigzag_dispatch_parity` for the new kernel.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_quantize_scaled_zigzag_dispatch_parity() {
        use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

        let (block, quant, zero_bias, aq) = quantize_test_data();
        let zb_scale = 0.4_f32;

        let reference =
            quantize_block_zigzag_scaled(&quant.mul_rows, &block, &zero_bias, zb_scale, aq);

        let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
            let result =
                quantize_block_zigzag_scaled(&quant.mul_rows, &block, &zero_bias, zb_scale, aq);
            assert_eq!(
                result, reference,
                "quantize_block_zigzag_scaled mismatch at permutation: {perm}"
            );
        });
        eprintln!("quantize_scaled_zigzag: {report}");
        assert!(
            report.permutations_run >= 2,
            "expected at least 2 permutations"
        );
    }

    #[test]
    fn test_block8x8f_roundtrip() {
        let mut arr = [0.0f32; 64];
        for i in 0..64 {
            arr[i] = i as f32 * 1.5;
        }

        let block = Block8x8f::from_array(&arr);
        let result = block.to_array();

        for i in 0..64 {
            assert!((arr[i] - result[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_block8x8f_get_set() {
        let mut block = Block8x8f::ZERO;
        block.set(3, 5, 42.0);
        assert!((block.get(3, 5) - 42.0).abs() < 1e-6);
    }

    #[test]
    fn test_block8x8f_scale() {
        let mut arr = [0.0f32; 64];
        for i in 0..64 {
            arr[i] = i as f32;
        }

        let block = Block8x8f::from_array(&arr);
        let scaled = block.scale(2.0);

        for i in 0..64 {
            let row = i / 8;
            let col = i % 8;
            assert!((scaled.get(row, col) - (i as f32 * 2.0)).abs() < 1e-6);
        }
    }

    #[test]
    fn test_quant_table_simd() {
        let mut values = [1u16; 64];
        for i in 0..64 {
            values[i] = (i + 1) as u16;
        }

        let quant = QuantTableSimd::from_values(&values);

        // Check that multipliers are correct (8.0 / value to compensate for 1/64 DCT scaling)
        for row in 0..8 {
            for col in 0..8 {
                let expected = 8.0 / (row * 8 + col + 1) as f32;
                assert!((quant.mul_rows[row][col] - expected).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_quantize_simple() {
        // Create a block with known values (simulating DCT output at 1/64 scale)
        let mut arr = [0.0f32; 64];
        for i in 0..64 {
            arr[i] = (i + 1) as f32 * 10.0; // 10, 20, 30, ...
        }
        let block = Block8x8f::from_array(&arr);

        // Create quant table where each value equals its position + 1
        let mut values = [1u16; 64];
        for i in 0..64 {
            values[i] = (i + 1) as u16;
        }
        let quant = QuantTableSimd::from_values(&values);

        // Quantize: coeff * 8 / quant_value = (i+1)*10 * 8 / (i+1) = 80
        let result = quant.quantize(&block);

        let arr = result.to_i16_array();
        for i in 0..64 {
            assert_eq!(arr[i], 80, "Mismatch at index {}", i);
        }
    }
}
