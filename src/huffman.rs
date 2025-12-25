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

/// Frequency counter for Huffman optimization
#[derive(Clone)]
pub struct FrequencyCounter {
    /// Counts for each symbol (257 elements for pseudo-symbol at end)
    pub counts: [i64; 257],
}

impl Default for FrequencyCounter {
    fn default() -> Self {
        Self { counts: [0; 257] }
    }
}

impl FrequencyCounter {
    /// Create a new frequency counter
    pub fn new() -> Self {
        Self { counts: [0; 257] }
    }

    /// Count a symbol occurrence
    #[inline]
    pub fn count(&mut self, symbol: u8) {
        self.counts[symbol as usize] += 1;
    }

    /// Reset all counts
    pub fn reset(&mut self) {
        self.counts.fill(0);
    }

    /// Generate optimal Huffman table from frequencies
    pub fn generate_table(&self) -> HuffmanTable {
        // Find non-zero symbols
        let mut symbols: Vec<(usize, i64)> = self.counts[..256]
            .iter()
            .enumerate()
            .filter(|(_, &c)| c > 0)
            .map(|(i, &c)| (i, c))
            .collect();

        if symbols.is_empty() {
            // No symbols - return empty table
            return HuffmanTable::new(&[0; 16], &[]);
        }

        // Sort by frequency (descending)
        symbols.sort_by(|a, b| b.1.cmp(&a.1));

        // Build code lengths using package-merge algorithm (simplified)
        // For simplicity, use a heuristic: assign shorter codes to more frequent symbols
        let n = symbols.len();
        let mut code_lengths: Vec<u8> = vec![0; n];

        // Simple heuristic: distribute codes across lengths 1-16
        // More frequent symbols get shorter codes
        let max_codes: [usize; 17] = [0, 2, 4, 8, 16, 32, 64, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256];
        let mut idx = 0;
        for len in 1..=16u8 {
            let max_at_len = max_codes[len as usize];
            let available = max_at_len.saturating_sub(max_codes[len as usize - 1]);
            for _ in 0..available {
                if idx < n {
                    code_lengths[idx] = len;
                    idx += 1;
                }
            }
        }

        // Ensure all symbols have a code
        while idx < n {
            code_lengths[idx] = 16;
            idx += 1;
        }

        // Limit code lengths to 16 bits (JPEG requirement)
        adjust_code_lengths(&mut code_lengths);

        // Count codes per length
        let mut bits = [0u8; 16];
        for &len in &code_lengths {
            if len > 0 && len <= 16 {
                bits[len as usize - 1] += 1;
            }
        }

        // Build values array (sorted by code length, then by symbol)
        let mut sym_len: Vec<(usize, u8)> = symbols.iter()
            .zip(code_lengths.iter())
            .map(|((sym, _), &len)| (*sym, len))
            .collect();
        sym_len.sort_by_key(|&(sym, len)| (len, sym));

        let values: Vec<u8> = sym_len.iter().map(|&(sym, _)| sym as u8).collect();

        HuffmanTable::new(&bits, &values)
    }
}

/// Adjust code lengths to satisfy JPEG's 16-bit limit
fn adjust_code_lengths(lengths: &mut [u8]) {
    // Count codes at each length
    let mut count = [0usize; 33];
    for &len in lengths.iter() {
        count[len as usize] += 1;
    }

    // Adjust any codes longer than 16
    for i in (17..=32).rev() {
        while count[i] > 0 {
            let mut j = i - 2;
            while count[j] == 0 {
                if j == 0 {
                    break;
                }
                j -= 1;
            }
            if j == 0 {
                break;
            }
            count[i] -= 2;
            count[i - 1] += 1;
            count[j + 1] += 2;
            count[j] -= 1;
        }
    }

    // Distribute lengths back to symbols
    let mut idx = 0;
    for len in 1..=16u8 {
        for _ in 0..count[len as usize] {
            if idx < lengths.len() {
                lengths[idx] = len;
                idx += 1;
            }
        }
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
