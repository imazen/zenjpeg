//! Huffman coding for JPEG entropy encoding
//!
//! Provides both standard Huffman tables and optimized table generation.

use crate::consts::ZIGZAG;

/// Standard DC luminance Huffman table (JPEG Annex K)
pub const STD_DC_LUMA_BITS: [u8; 16] = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
pub const STD_DC_LUMA_VALUES: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

/// Standard DC chrominance Huffman table (JPEG Annex K)
pub const STD_DC_CHROMA_BITS: [u8; 16] = [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0];
pub const STD_DC_CHROMA_VALUES: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

/// Standard AC luminance Huffman table (JPEG Annex K)
pub const STD_AC_LUMA_BITS: [u8; 16] = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 125];
pub const STD_AC_LUMA_VALUES: [u8; 162] = [
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08, 0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7,
    0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5,
    0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
    0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
    0xF9, 0xFA,
];

/// Huffman table for encoding
#[derive(Clone)]
pub struct HuffmanTable {
    /// Code lengths for each symbol
    pub bits: [u8; 16],
    /// Symbol values in code length order
    pub values: Vec<u8>,
    /// Encoded codes for each symbol (indexed by symbol value)
    pub codes: [u32; 256],
    /// Code lengths for each symbol (indexed by symbol value)
    pub sizes: [u8; 256],
}

impl HuffmanTable {
    /// Create a Huffman table from BITS and VALUES arrays
    pub fn new(bits: &[u8; 16], values: &[u8]) -> Self {
        let mut table = Self {
            bits: *bits,
            values: values.to_vec(),
            codes: [0; 256],
            sizes: [0; 256],
        };
        table.generate_codes();
        table
    }

    /// Generate code assignments from BITS array
    fn generate_codes(&mut self) {
        let mut code = 0u32;
        let mut si = 0usize;

        for i in 0..16 {
            for _ in 0..self.bits[i] {
                if si < self.values.len() {
                    let sym = self.values[si] as usize;
                    self.codes[sym] = code;
                    self.sizes[sym] = (i + 1) as u8;
                    si += 1;
                    code += 1;
                }
            }
            code <<= 1;
        }
    }

    /// Get the code and size for a symbol
    #[inline]
    pub fn encode(&self, symbol: u8) -> (u32, u8) {
        (self.codes[symbol as usize], self.sizes[symbol as usize])
    }
}

/// Create standard DC luminance Huffman table
pub fn std_dc_luma() -> HuffmanTable {
    HuffmanTable::new(&STD_DC_LUMA_BITS, &STD_DC_LUMA_VALUES)
}

/// Create standard DC chrominance Huffman table
pub fn std_dc_chroma() -> HuffmanTable {
    HuffmanTable::new(&STD_DC_CHROMA_BITS, &STD_DC_CHROMA_VALUES)
}

/// Create standard AC luminance Huffman table
pub fn std_ac_luma() -> HuffmanTable {
    HuffmanTable::new(&STD_AC_LUMA_BITS, &STD_AC_LUMA_VALUES)
}

/// Compute category (bit size) for a DC/AC coefficient difference
#[inline]
pub fn compute_category(value: i16) -> u8 {
    let abs_val = value.unsigned_abs();
    if abs_val == 0 {
        0
    } else {
        16 - abs_val.leading_zeros() as u8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_category() {
        assert_eq!(compute_category(0), 0);
        assert_eq!(compute_category(1), 1);
        assert_eq!(compute_category(-1), 1);
        assert_eq!(compute_category(2), 2);
        assert_eq!(compute_category(3), 2);
        assert_eq!(compute_category(4), 3);
        assert_eq!(compute_category(7), 3);
        assert_eq!(compute_category(255), 8);
    }

    #[test]
    fn test_huffman_table_generation() {
        let table = std_dc_luma();
        // Symbol 0 should have a valid code
        let (code, size) = table.encode(0);
        assert!(size > 0);
        assert!(size <= 16);
    }
}
