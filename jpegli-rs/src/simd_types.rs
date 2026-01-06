//! SIMD-native data types for efficient block processing.
//!
//! These types store data in SIMD-friendly layouts to eliminate load/store overhead
//! during DCT and quantization operations.

use wide::{f32x8, i16x8, i32x8};

/// An 8x8 block stored as 8 rows of f32x8 for SIMD-native access.
///
/// This layout means:
/// - Each row is already a SIMD vector (no gather needed)
/// - Row-wise operations (DCT, quantization) are trivial
/// - 32-byte aligned for optimal SIMD access
#[derive(Clone, Copy, Debug)]
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
            *row = f32x8::from([
                arr[start],
                arr[start + 1],
                arr[start + 2],
                arr[start + 3],
                arr[start + 4],
                arr[start + 5],
                arr[start + 6],
                arr[start + 7],
            ]);
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
            *row = i16x8::from([
                arr[start],
                arr[start + 1],
                arr[start + 2],
                arr[start + 3],
                arr[start + 4],
                arr[start + 5],
                arr[start + 6],
                arr[start + 7],
            ]);
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
/// Pre-computes multipliers (1/value) for fast quantization.
#[derive(Clone, Debug)]
#[repr(C, align(32))]
pub struct QuantTableSimd {
    /// Reciprocal multipliers for quantization (1.0 / quant_value)
    pub mul_rows: [f32x8; 8],
    /// Original values for encoding the JPEG header
    pub values: [u16; 64],
}

impl QuantTableSimd {
    /// Create from u16 quantization values
    pub fn from_values(values: &[u16; 64]) -> Self {
        let mut mul_rows = [f32x8::ZERO; 8];
        for row in 0..8 {
            let start = row * 8;
            mul_rows[row] = f32x8::from([
                1.0 / values[start] as f32,
                1.0 / values[start + 1] as f32,
                1.0 / values[start + 2] as f32,
                1.0 / values[start + 3] as f32,
                1.0 / values[start + 4] as f32,
                1.0 / values[start + 5] as f32,
                1.0 / values[start + 6] as f32,
                1.0 / values[start + 7] as f32,
            ]);
        }
        Self {
            mul_rows,
            values: *values,
        }
    }

    /// Create from f32 quantization values
    pub fn from_f32_values(values: &[f32; 64]) -> Self {
        let mut mul_rows = [f32x8::ZERO; 8];
        let mut u16_values = [0u16; 64];
        for row in 0..8 {
            let start = row * 8;
            mul_rows[row] = f32x8::from([
                1.0 / values[start],
                1.0 / values[start + 1],
                1.0 / values[start + 2],
                1.0 / values[start + 3],
                1.0 / values[start + 4],
                1.0 / values[start + 5],
                1.0 / values[start + 6],
                1.0 / values[start + 7],
            ]);
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

#[cfg(test)]
mod tests {
    use super::*;

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

        // Check that multipliers are correct
        for row in 0..8 {
            let row_arr: [f32; 8] = quant.mul_rows[row].into();
            for col in 0..8 {
                let expected = 1.0 / (row * 8 + col + 1) as f32;
                assert!((row_arr[col] - expected).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_quantize_simple() {
        // Create a block with known values
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

        // Quantize
        let result = quant.quantize(&block);

        // Each coefficient should be 10 (value * 10 / value = 10)
        let arr = result.to_i16_array();
        for i in 0..64 {
            assert_eq!(arr[i], 10, "Mismatch at index {}", i);
        }
    }
}
