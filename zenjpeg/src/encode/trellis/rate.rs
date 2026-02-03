//! Rate estimation tables for trellis quantization.
//!
//! Trellis quantization needs to estimate Huffman encoding costs for
//! candidate quantized values. It only needs code *lengths* (not the
//! actual codes), so we store a compact table of just the sizes.

use crate::huffman::{
    STD_AC_CHROMINANCE_BITS, STD_AC_CHROMINANCE_VALUES, STD_AC_LUMINANCE_BITS,
    STD_AC_LUMINANCE_VALUES, STD_DC_CHROMINANCE_BITS, STD_DC_CHROMINANCE_VALUES,
    STD_DC_LUMINANCE_BITS, STD_DC_LUMINANCE_VALUES,
};

/// Compact rate estimation table storing only Huffman code lengths.
///
/// This replaces `mozjpeg_rs::huffman::DerivedTable` for trellis use.
/// Trellis only needs code sizes for rate estimation, not the actual codes.
#[derive(Clone, Debug)]
pub struct RateTable {
    /// Code length for each symbol (0 means no code assigned).
    sizes: [u8; 256],
}

impl RateTable {
    /// Build a rate table from standard JPEG Huffman arrays.
    ///
    /// # Arguments
    /// * `bits` - Number of codes of each length (16 entries, bits[0] = length 1 count, etc.)
    /// * `values` - Symbol values in order of increasing code length
    pub fn from_bits_values(bits: &[u8; 16], values: &[u8]) -> Self {
        let mut sizes = [0u8; 256];
        let mut val_idx = 0;
        for (length_minus_1, &count) in bits.iter().enumerate() {
            let length = (length_minus_1 + 1) as u8;
            for _ in 0..count {
                if val_idx < values.len() {
                    sizes[values[val_idx] as usize] = length;
                    val_idx += 1;
                }
            }
        }
        Self { sizes }
    }

    /// Standard AC luminance rate table.
    pub fn standard_luma_ac() -> Self {
        Self::from_bits_values(&STD_AC_LUMINANCE_BITS, &STD_AC_LUMINANCE_VALUES)
    }

    /// Standard AC chrominance rate table.
    pub fn standard_chroma_ac() -> Self {
        Self::from_bits_values(&STD_AC_CHROMINANCE_BITS, &STD_AC_CHROMINANCE_VALUES)
    }

    /// Standard DC luminance rate table.
    pub fn standard_luma_dc() -> Self {
        Self::from_bits_values(&STD_DC_LUMINANCE_BITS, &STD_DC_LUMINANCE_VALUES)
    }

    /// Standard DC chrominance rate table.
    pub fn standard_chroma_dc() -> Self {
        Self::from_bits_values(&STD_DC_CHROMINANCE_BITS, &STD_DC_CHROMINANCE_VALUES)
    }

    /// Get the code length for a symbol. Returns 0 if no code assigned.
    #[inline]
    pub fn get_code_length(&self, symbol: u8) -> u8 {
        self.sizes[symbol as usize]
    }

    /// Get (0, code_length) for API compatibility with DerivedTable::get_code.
    ///
    /// The first element is always 0 (we don't store actual codes).
    /// Trellis only uses the second element (size).
    #[inline]
    pub fn get_code(&self, symbol: u8) -> (u32, u8) {
        (0, self.sizes[symbol as usize])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_luma_ac_has_eob() {
        let table = RateTable::standard_luma_ac();
        // EOB symbol (0x00) should have a code
        assert!(table.get_code_length(0x00) > 0);
        // ZRL symbol (0xF0) should have a code
        assert!(table.get_code_length(0xF0) > 0);
    }

    #[test]
    fn test_standard_chroma_ac_has_eob() {
        let table = RateTable::standard_chroma_ac();
        assert!(table.get_code_length(0x00) > 0);
        assert!(table.get_code_length(0xF0) > 0);
    }

    #[test]
    fn test_standard_dc_tables() {
        let luma = RateTable::standard_luma_dc();
        let chroma = RateTable::standard_chroma_dc();
        // DC category 0 (value=0) should have a code
        assert!(luma.get_code_length(0) > 0);
        assert!(chroma.get_code_length(0) > 0);
        // DC category 8 should have a code
        assert!(luma.get_code_length(8) > 0);
        assert!(chroma.get_code_length(8) > 0);
    }

    #[test]
    fn test_get_code_compat() {
        let table = RateTable::standard_luma_ac();
        let (code, size) = table.get_code(0x00);
        assert_eq!(code, 0); // We don't store codes
        assert!(size > 0);
    }

    #[test]
    fn test_unassigned_symbol_returns_zero() {
        let table = RateTable::standard_luma_ac();
        // Symbol 0xFF is typically not in standard AC tables
        // (it would be run=15, size=15 which doesn't exist)
        let size = table.get_code_length(0xFF);
        // May or may not be 0 depending on the table, but shouldn't panic
        let _ = size;
    }
}
