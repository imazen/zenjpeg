//! SIMD-native data types for efficient block processing.
//!
//! These types store data in SIMD-friendly layouts to eliminate load/store overhead
//! during DCT and quantization operations.

#![allow(dead_code)]
#![allow(clippy::wrong_self_convention)] // to_* methods need &self for SIMD types

#[cfg(target_arch = "x86_64")]
use archmage::SimdToken;
use wide::{CmpGe, f32x8, i16x8, i32x8};

/// An 8x8 block stored as 8 rows of f32x8 for SIMD-native access.
///
/// This layout means:
/// - Each row is already a SIMD vector (no gather needed)
/// - Row-wise operations (DCT, quantization) are trivial
/// - 32-byte aligned for optimal SIMD access
///
/// # Safety
///
/// `Block8x8f` is `Pod` and `Zeroable` because:
/// - `f32x8` is `#[repr(C)]` containing 8 f32s (all Pod)
/// - The struct is `#[repr(C, align(32))]` with no padding
/// - All bit patterns are valid (f32 allows all patterns including NaN)
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C, align(32))]
pub struct Block8x8f {
    pub rows: [f32x8; 8],
}

impl Block8x8f {
    pub const ZERO: Self = Self {
        rows: [f32x8::ZERO; 8],
    };

    /// Create from a flat array (for compatibility with existing code)
    #[inline]
    pub fn from_array(arr: &[f32; 64]) -> Self {
        let mut rows = [f32x8::ZERO; 8];
        for (row_idx, row) in rows.iter_mut().enumerate() {
            let start = row_idx * 8;
            // Use slice-to-array conversion - zero-cost load from contiguous memory
            let row_slice: [f32; 8] = arr[start..start + 8].try_into().unwrap();
            *row = f32x8::from(row_slice);
        }
        Self { rows }
    }

    /// Convert to a flat array (for compatibility with existing code)
    #[inline]
    pub fn to_array(&self) -> [f32; 64] {
        let mut arr = [0.0f32; 64];
        for (row_idx, row) in self.rows.iter().enumerate() {
            let row_arr: [f32; 8] = (*row).into();
            arr[row_idx * 8..row_idx * 8 + 8].copy_from_slice(&row_arr);
        }
        arr
    }

    /// Access a single coefficient
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> f32 {
        let row_arr: [f32; 8] = self.rows[row].into();
        row_arr[col]
    }

    /// Set a single coefficient
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: f32) {
        let mut row_arr: [f32; 8] = self.rows[row].into();
        row_arr[col] = value;
        self.rows[row] = f32x8::from(row_arr);
    }

    /// Multiply all elements by a scalar
    #[inline]
    pub fn scale(&self, factor: f32) -> Self {
        let scale = f32x8::splat(factor);
        let mut result = Self::ZERO;
        for i in 0..8 {
            result.rows[i] = self.rows[i] * scale;
        }
        result
    }

    /// Element-wise multiply with another block
    #[inline]
    pub fn mul(&self, other: &Self) -> Self {
        let mut result = Self::ZERO;
        for i in 0..8 {
            result.rows[i] = self.rows[i] * other.rows[i];
        }
        result
    }

    /// Element-wise add
    #[inline]
    pub fn add(&self, other: &Self) -> Self {
        let mut result = Self::ZERO;
        for i in 0..8 {
            result.rows[i] = self.rows[i] + other.rows[i];
        }
        result
    }
}

impl Default for Block8x8f {
    fn default() -> Self {
        Self::ZERO
    }
}

/// An 8x8 block of i16 values stored as 8 rows of i16x8.
///
/// Used for quantized DCT coefficients.
#[derive(Clone, Copy, Debug)]
#[repr(C, align(16))]
pub struct Block8x8i16 {
    pub rows: [i16x8; 8],
}

impl Block8x8i16 {
    pub const ZERO: Self = Self {
        rows: [i16x8::ZERO; 8],
    };

    /// Create from a flat array
    #[inline]
    pub fn from_array(arr: &[i16; 64]) -> Self {
        let mut rows = [i16x8::ZERO; 8];
        for (row_idx, row) in rows.iter_mut().enumerate() {
            let start = row_idx * 8;
            // Use slice-to-array conversion - zero-cost load from contiguous memory
            let row_slice: [i16; 8] = arr[start..start + 8].try_into().unwrap();
            *row = i16x8::from(row_slice);
        }
        Self { rows }
    }

    /// Convert to a flat array
    #[inline]
    pub fn to_array(&self) -> [i16; 64] {
        let mut arr = [0i16; 64];
        for (row_idx, row) in self.rows.iter().enumerate() {
            let row_arr: [i16; 8] = (*row).into();
            arr[row_idx * 8..row_idx * 8 + 8].copy_from_slice(&row_arr);
        }
        arr
    }

    /// Access a single coefficient
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> i16 {
        let row_arr: [i16; 8] = self.rows[row].into();
        row_arr[col]
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
    pub mul_rows: [f32x8; 8],
    /// Original values for encoding the JPEG header
    pub values: [u16; 64],
}

/// Zero-bias parameters stored in SIMD-friendly layout.
///
/// Pre-computed thresholds for each coefficient position.
#[derive(Clone, Debug)]
#[repr(C, align(32))]
pub struct ZeroBiasSimd {
    /// offset\[k\] for each coefficient (8 rows of f32x8)
    pub offset_rows: [f32x8; 8],
    /// mul\[k\] for each coefficient (8 rows of f32x8)
    pub mul_rows: [f32x8; 8],
}

impl ZeroBiasSimd {
    /// Create from ZeroBiasParams
    pub fn from_params(params: &crate::quant::ZeroBiasParams) -> Self {
        let mut offset_rows = [f32x8::ZERO; 8];
        let mut mul_rows = [f32x8::ZERO; 8];
        for row in 0..8 {
            let start = row * 8;
            // Zero-cost load from contiguous memory
            let offset_slice: [f32; 8] = params.offset[start..start + 8].try_into().unwrap();
            let mul_slice: [f32; 8] = params.mul[start..start + 8].try_into().unwrap();
            offset_rows[row] = f32x8::from(offset_slice);
            mul_rows[row] = f32x8::from(mul_slice);
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
        let mut mul_rows = [f32x8::ZERO; 8];
        // DCT uses 1/64 scaling, so quantize needs to multiply by 8/quant
        let eight = f32x8::splat(8.0);
        for row in 0..8 {
            let start = row * 8;
            // Convert u16 -> f32 (unavoidable), then SIMD divide
            let row_f32: [f32; 8] = [
                values[start] as f32,
                values[start + 1] as f32,
                values[start + 2] as f32,
                values[start + 3] as f32,
                values[start + 4] as f32,
                values[start + 5] as f32,
                values[start + 6] as f32,
                values[start + 7] as f32,
            ];
            mul_rows[row] = eight / f32x8::from(row_f32);
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
        let mut mul_rows = [f32x8::ZERO; 8];
        let mut u16_values = [0u16; 64];
        // DCT uses 1/64 scaling, so quantize needs to multiply by 8/quant
        let eight = f32x8::splat(8.0);
        for row in 0..8 {
            let start = row * 8;
            // Zero-cost load, SIMD divide
            let values_slice: [f32; 8] = values[start..start + 8].try_into().unwrap();
            mul_rows[row] = eight / f32x8::from(values_slice);
            for col in 0..8 {
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
            // Multiply and round to nearest integer
            let quantized = block.rows[i] * self.mul_rows[i];
            result.rows[i] = quantized.round_int();
        }
        result
    }

    /// Quantize a block with zero-bias using SIMD.
    ///
    /// This is the optimized hot path for encoding. Processes 8 coefficients at a time
    /// with zero additional load/store overhead when data is already in SIMD format.
    ///
    /// # Arguments
    /// * `block` - DCT coefficients in SIMD-native format
    /// * `zero_bias` - Pre-computed zero-bias parameters in SIMD format
    /// * `aq_strength` - Per-block adaptive quantization strength
    ///
    /// # Returns
    /// Quantized coefficients ready for entropy coding
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
    /// Uses unsafe pointer casting for aligned loads - the input must be 64 f32s.
    #[inline]
    pub fn quantize_array_with_zero_bias(
        &self,
        coeffs: &[f32; 64],
        zero_bias: &ZeroBiasSimd,
        aq_strength: f32,
    ) -> [i16; 64] {
        let mut result = [0i16; 64];
        let aq = f32x8::splat(aq_strength);

        for row in 0..8 {
            let k = row * 8;
            // Load 8 coefficients - compiler will optimize this
            let coeffs_simd = f32x8::new([
                coeffs[k],
                coeffs[k + 1],
                coeffs[k + 2],
                coeffs[k + 3],
                coeffs[k + 4],
                coeffs[k + 5],
                coeffs[k + 6],
                coeffs[k + 7],
            ]);

            // qval = coeffs / quant (using pre-computed 1/quant)
            let qval = coeffs_simd * self.mul_rows[row];

            // threshold = offset + mul * aq_strength
            let threshold = zero_bias.offset_rows[row] + zero_bias.mul_rows[row] * aq;

            // |qval| >= threshold ? round(qval) : 0
            let abs_qval = qval.abs();

            // Zero-copy access with fast rounding
            let abs_arr = abs_qval.as_array();
            let thresh_arr = threshold.as_array();
            let rounded = qval.fast_round_int();
            let rounded_arr = rounded.as_array();

            for i in 0..8 {
                if abs_arr[i] >= thresh_arr[i] {
                    result[k + i] = rounded_arr[i] as i16;
                }
            }
        }

        result
    }
}

/// An 8x8 block of i32 values stored as 8 rows of i32x8.
///
/// Used as intermediate during quantization before conversion to i16.
#[derive(Clone, Copy, Debug)]
#[repr(C, align(32))]
pub struct Block8x8i32 {
    pub rows: [i32x8; 8],
}

impl Block8x8i32 {
    pub const ZERO: Self = Self {
        rows: [i32x8::ZERO; 8],
    };

    /// Convert to i16 block (with saturation)
    #[inline]
    pub fn to_i16(&self) -> Block8x8i16 {
        let mut result = Block8x8i16::ZERO;
        for i in 0..8 {
            // Extract i32 values and convert to i16 with saturation
            let row: [i32; 8] = self.rows[i].into();
            result.rows[i] = i16x8::from([
                row[0].clamp(-32768, 32767) as i16,
                row[1].clamp(-32768, 32767) as i16,
                row[2].clamp(-32768, 32767) as i16,
                row[3].clamp(-32768, 32767) as i16,
                row[4].clamp(-32768, 32767) as i16,
                row[5].clamp(-32768, 32767) as i16,
                row[6].clamp(-32768, 32767) as i16,
                row[7].clamp(-32768, 32767) as i16,
            ]);
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

/// Archmage AVX2+FMA quantize with zigzag output. True 256-bit operations.
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn mage_quantize_block_zigzag(
    _token: archmage::X64V3Token,
    block: &Block8x8f,
    mul_rows: &[f32x8; 8],
    zero_bias: &ZeroBiasSimd,
    aq_strength: f32,
) -> [i16; 64] {
    use crate::foundation::consts::JPEG_ZIGZAG_ORDER;
    use magetypes::simd::f32x8 as mf32x8;
    use magetypes::simd::i32x8 as mi32x8;

    let token = _token;
    let aq_m = mf32x8::splat(token, aq_strength);
    let zero_i32 = mi32x8::zero(token);
    let mut result = [0i16; 64];

    for row in 0..8 {
        let block_arr: [f32; 8] = block.rows[row].into();
        let mul_arr: [f32; 8] = mul_rows[row].into();
        let offset_arr: [f32; 8] = zero_bias.offset_rows[row].into();
        let bias_mul_arr: [f32; 8] = zero_bias.mul_rows[row].into();

        let block_m = mf32x8::from_array(token, block_arr);
        let mul_m = mf32x8::from_array(token, mul_arr);
        let offset_m = mf32x8::from_array(token, offset_arr);
        let bias_mul_m = mf32x8::from_array(token, bias_mul_arr);

        let qval = block_m * mul_m;
        let threshold = bias_mul_m.mul_add(aq_m, offset_m);
        let abs_qval = qval.abs();
        let mask = abs_qval.simd_ge(threshold);
        let rounded = qval.to_i32_round();
        let mask_i32 = mask.bitcast_to_i32();
        let blended = mi32x8::blend(mask_i32, rounded, zero_i32);

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

/// Archmage AVX2+FMA quantize, natural order output. True 256-bit operations.
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn mage_quantize_block(
    _token: archmage::X64V3Token,
    block: &Block8x8f,
    mul_rows: &[f32x8; 8],
    zero_bias: &ZeroBiasSimd,
    aq_strength: f32,
) -> [i16; 64] {
    use magetypes::simd::f32x8 as mf32x8;
    use magetypes::simd::i32x8 as mi32x8;

    let token = _token;
    let aq_m = mf32x8::splat(token, aq_strength);
    let zero_i32 = mi32x8::zero(token);
    let mut result = [0i16; 64];

    for row in 0..8 {
        let block_arr: [f32; 8] = block.rows[row].into();
        let mul_arr: [f32; 8] = mul_rows[row].into();
        let offset_arr: [f32; 8] = zero_bias.offset_rows[row].into();
        let bias_mul_arr: [f32; 8] = zero_bias.mul_rows[row].into();

        let block_m = mf32x8::from_array(token, block_arr);
        let mul_m = mf32x8::from_array(token, mul_arr);
        let offset_m = mf32x8::from_array(token, offset_arr);
        let bias_mul_m = mf32x8::from_array(token, bias_mul_arr);

        let qval = block_m * mul_m;
        let threshold = bias_mul_m.mul_add(aq_m, offset_m);
        let abs_qval = qval.abs();
        let mask = abs_qval.simd_ge(threshold);
        let rounded = qval.to_i32_round();
        let mask_i32 = mask.bitcast_to_i32();
        let blended = mi32x8::blend(mask_i32, rounded, zero_i32);

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

/// Scalar fallback quantize with zigzag output (wide crate, 2× SSE2).
#[archmage::autoversion]
fn scalar_quantize_block_zigzag(
    block: &Block8x8f,
    mul_rows: &[f32x8; 8],
    zero_bias: &ZeroBiasSimd,
    aq_strength: f32,
) -> [i16; 64] {
    use crate::foundation::consts::JPEG_ZIGZAG_ORDER;

    let mut result = [0i16; 64];
    let aq = f32x8::splat(aq_strength);
    let zero_i32 = i32x8::ZERO;

    for row in 0..8 {
        let qval = block.rows[row] * mul_rows[row];
        let threshold = zero_bias.offset_rows[row] + zero_bias.mul_rows[row] * aq;
        let abs_qval = qval.abs();
        let mask_f32 = abs_qval.simd_ge(threshold);
        let mask_i32: i32x8 = bytemuck::cast(mask_f32);
        let rounded = qval.fast_round_int();
        let blended = mask_i32.blend(rounded, zero_i32);

        let arr = blended.as_array();
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

/// Scalar fallback quantize, natural order output (wide crate, 2× SSE2).
#[archmage::autoversion]
fn scalar_quantize_block(
    block: &Block8x8f,
    mul_rows: &[f32x8; 8],
    zero_bias: &ZeroBiasSimd,
    aq_strength: f32,
) -> [i16; 64] {
    let mut result = [0i16; 64];
    let aq = f32x8::splat(aq_strength);
    let zero_i32 = i32x8::ZERO;

    for row in 0..8 {
        let qval = block.rows[row] * mul_rows[row];
        let threshold = zero_bias.offset_rows[row] + zero_bias.mul_rows[row] * aq;
        let abs_qval = qval.abs();
        let mask_f32 = abs_qval.simd_ge(threshold);
        let mask_i32: i32x8 = bytemuck::cast(mask_f32);
        let rounded = qval.fast_round_int();
        let blended = mask_i32.blend(rounded, zero_i32);

        let arr = blended.as_array();
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

/// Dispatching quantize with zigzag — tries archmage AVX2, falls back to scalar.
#[inline]
fn quantize_block_zigzag(
    mul_rows: &[f32x8; 8],
    block: &Block8x8f,
    zero_bias: &ZeroBiasSimd,
    aq_strength: f32,
) -> [i16; 64] {
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(token) = archmage::X64V3Token::summon() {
            return mage_quantize_block_zigzag(token, block, mul_rows, zero_bias, aq_strength);
        }
    }
    scalar_quantize_block_zigzag(block, mul_rows, zero_bias, aq_strength)
}

/// Dispatching quantize natural order — tries archmage AVX2, falls back to scalar.
#[inline]
fn quantize_block(
    mul_rows: &[f32x8; 8],
    block: &Block8x8f,
    zero_bias: &ZeroBiasSimd,
    aq_strength: f32,
) -> [i16; 64] {
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(token) = archmage::X64V3Token::summon() {
            return mage_quantize_block(token, block, mul_rows, zero_bias, aq_strength);
        }
    }
    scalar_quantize_block(block, mul_rows, zero_bias, aq_strength)
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
            let row_arr: [f32; 8] = quant.mul_rows[row].into();
            for col in 0..8 {
                let expected = 8.0 / (row * 8 + col + 1) as f32;
                assert!((row_arr[col] - expected).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_quantize_simple() {
        // Create a block with known values (simulating DCT output at 1/64 scale)
        // The quantize function expects coefficients that have been scaled by 1/64 from DCT.
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
