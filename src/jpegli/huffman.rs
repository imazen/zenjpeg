//! Huffman table construction and lookup.
//!
//! This module provides:
//! - Standard JPEG Huffman tables (DC and AC for luminance and chrominance)
//! - Huffman table building from symbol frequencies
//! - Lookup table generation for fast encoding/decoding

use crate::jpegli::error::{Error, Result};

/// Maximum code length in bits for JPEG Huffman codes.
pub const MAX_CODE_LENGTH: usize = 16;

/// Maximum number of symbols in a Huffman table.
pub const MAX_SYMBOLS: usize = 256;

/// Standard DC luminance Huffman table (bits).
pub const STD_DC_LUMINANCE_BITS: [u8; 16] = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];

/// Standard DC luminance Huffman table (values).
pub const STD_DC_LUMINANCE_VALUES: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

/// Standard DC chrominance Huffman table (bits).
pub const STD_DC_CHROMINANCE_BITS: [u8; 16] = [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0];

/// Standard DC chrominance Huffman table (values).
pub const STD_DC_CHROMINANCE_VALUES: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

/// Standard AC luminance Huffman table (bits).
pub const STD_AC_LUMINANCE_BITS: [u8; 16] = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 125];

/// Standard AC luminance Huffman table (values).
pub const STD_AC_LUMINANCE_VALUES: [u8; 162] = [
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
    0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
    0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
    0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa,
];

/// Standard AC chrominance Huffman table (bits).
pub const STD_AC_CHROMINANCE_BITS: [u8; 16] = [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 119];

/// Standard AC chrominance Huffman table (values).
pub const STD_AC_CHROMINANCE_VALUES: [u8; 162] = [
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
    0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
    0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
    0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
    0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
    0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
    0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
    0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
    0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa,
];

/// A Huffman encoding table for fast symbol-to-code lookup.
#[derive(Debug, Clone)]
pub struct HuffmanEncodeTable {
    /// Code value for each symbol
    pub codes: [u32; MAX_SYMBOLS],
    /// Code length in bits for each symbol
    pub lengths: [u8; MAX_SYMBOLS],
    /// Number of valid symbols
    pub num_symbols: usize,
}

impl Default for HuffmanEncodeTable {
    fn default() -> Self {
        Self {
            codes: [0; MAX_SYMBOLS],
            lengths: [0; MAX_SYMBOLS],
            num_symbols: 0,
        }
    }
}

impl HuffmanEncodeTable {
    /// Creates a new encoding table from JPEG-format bits and values.
    ///
    /// # Arguments
    /// * `bits` - Number of codes of each length (1-16 bits)
    /// * `values` - Symbol values in order
    pub fn from_bits_values(bits: &[u8; 16], values: &[u8]) -> Result<Self> {
        let mut table = Self::default();

        // Count total symbols
        let total_symbols: usize = bits.iter().map(|&b| b as usize).sum();
        if total_symbols > MAX_SYMBOLS || total_symbols != values.len() {
            return Err(Error::InvalidHuffmanTable {
                table_idx: 0,
                reason: "symbol count mismatch",
            });
        }
        table.num_symbols = total_symbols;

        // Generate codes using JPEG algorithm
        let mut code: u32 = 0;
        let mut symbol_idx = 0;

        for (length_minus_1, &count) in bits.iter().enumerate() {
            let length = (length_minus_1 + 1) as u8;
            for _ in 0..count {
                if symbol_idx >= values.len() {
                    return Err(Error::InvalidHuffmanTable {
                        table_idx: 0,
                        reason: "too many codes for values",
                    });
                }
                let symbol = values[symbol_idx] as usize;
                table.codes[symbol] = code;
                table.lengths[symbol] = length;
                code += 1;
                symbol_idx += 1;
            }
            code <<= 1;
        }

        Ok(table)
    }

    /// Returns the code and length for a symbol.
    #[inline]
    pub fn encode(&self, symbol: u8) -> (u32, u8) {
        let idx = symbol as usize;
        (self.codes[idx], self.lengths[idx])
    }

    /// Creates the standard DC luminance table.
    #[must_use]
    pub fn std_dc_luminance() -> Self {
        Self::from_bits_values(&STD_DC_LUMINANCE_BITS, &STD_DC_LUMINANCE_VALUES)
            .expect("standard table should be valid")
    }

    /// Creates the standard DC chrominance table.
    #[must_use]
    pub fn std_dc_chrominance() -> Self {
        Self::from_bits_values(&STD_DC_CHROMINANCE_BITS, &STD_DC_CHROMINANCE_VALUES)
            .expect("standard table should be valid")
    }

    /// Creates the standard AC luminance table.
    #[must_use]
    pub fn std_ac_luminance() -> Self {
        Self::from_bits_values(&STD_AC_LUMINANCE_BITS, &STD_AC_LUMINANCE_VALUES)
            .expect("standard table should be valid")
    }

    /// Creates the standard AC chrominance table.
    #[must_use]
    pub fn std_ac_chrominance() -> Self {
        Self::from_bits_values(&STD_AC_CHROMINANCE_BITS, &STD_AC_CHROMINANCE_VALUES)
            .expect("standard table should be valid")
    }
}

/// A Huffman decoding table for fast code-to-symbol lookup.
#[derive(Debug, Clone)]
pub struct HuffmanDecodeTable {
    /// Lookup table for short codes (up to FAST_BITS)
    pub fast_lookup: [i16; 1 << Self::FAST_BITS],
    /// Maximum code value for each bit length
    pub maxcode: [i32; MAX_CODE_LENGTH + 1],
    /// Offset for decoding codes of each length
    pub valoffset: [i32; MAX_CODE_LENGTH + 1],
    /// Symbol values
    pub values: Vec<u8>,
    /// Whether the table is valid
    pub valid: bool,
}

impl HuffmanDecodeTable {
    /// Number of bits for fast lookup table.
    pub const FAST_BITS: usize = 9;

    /// Creates a new empty decode table.
    #[must_use]
    pub fn new() -> Self {
        Self {
            fast_lookup: [-1; 1 << Self::FAST_BITS],
            maxcode: [-1; MAX_CODE_LENGTH + 1],
            valoffset: [0; MAX_CODE_LENGTH + 1],
            values: Vec::new(),
            valid: false,
        }
    }

    /// Creates a decoding table from JPEG-format bits and values.
    pub fn from_bits_values(bits: &[u8; 16], values: &[u8]) -> Result<Self> {
        let mut table = Self::new();
        table.values = values.to_vec();

        // Build huffsize and huffcode arrays
        let mut huffsize = vec![0u8; values.len() + 1];
        let mut huffcode = vec![0u32; values.len()];

        // Generate size table
        let mut k = 0;
        for (i, &count) in bits.iter().enumerate() {
            for _ in 0..count {
                huffsize[k] = (i + 1) as u8;
                k += 1;
            }
        }
        huffsize[k] = 0;

        // Generate code table
        let mut code: u32 = 0;
        let mut si = huffsize[0] as usize;
        k = 0;
        while huffsize[k] != 0 {
            while (huffsize[k] as usize) == si {
                huffcode[k] = code;
                code += 1;
                k += 1;
            }
            code <<= 1;
            si += 1;
        }

        // Build maxcode and valoffset tables
        let mut j = 0;
        for i in 1..=MAX_CODE_LENGTH {
            if bits[i - 1] == 0 {
                table.maxcode[i] = -1;
            } else {
                table.valoffset[i] = j as i32 - huffcode[j] as i32;
                j += bits[i - 1] as usize;
                table.maxcode[i] = huffcode[j - 1] as i32;
            }
        }
        table.maxcode[MAX_CODE_LENGTH + 1 - 1] = 0x7FFF_FFFF;

        // Build fast lookup table
        for (k, &code) in huffcode.iter().enumerate() {
            let length = huffsize[k] as usize;
            if length <= Self::FAST_BITS && length > 0 {
                let fast_code = (code as usize) << (Self::FAST_BITS - length);
                let count = 1 << (Self::FAST_BITS - length);
                for m in 0..count {
                    let idx = fast_code + m;
                    // Bounds check for malformed Huffman tables
                    if idx < table.fast_lookup.len() {
                        // Store symbol in lower 8 bits, length in upper 8 bits
                        table.fast_lookup[idx] = (values[k] as i16) | ((length as i16) << 8);
                    }
                }
            }
        }

        table.valid = true;
        Ok(table)
    }

    /// Decodes a symbol from a bit stream using fast lookup.
    /// Returns (symbol, bits_consumed) or None if not in fast table.
    #[inline]
    pub fn fast_decode(&self, bits: u32) -> Option<(u8, u8)> {
        let lookup = self.fast_lookup[(bits >> (32 - Self::FAST_BITS)) as usize];
        if lookup >= 0 {
            let symbol = (lookup & 0xFF) as u8;
            let length = (lookup >> 8) as u8;
            Some((symbol, length))
        } else {
            None
        }
    }

    /// Creates the standard DC luminance decode table.
    #[must_use]
    pub fn std_dc_luminance() -> Self {
        Self::from_bits_values(&STD_DC_LUMINANCE_BITS, &STD_DC_LUMINANCE_VALUES)
            .expect("standard table should be valid")
    }

    /// Creates the standard DC chrominance decode table.
    #[must_use]
    pub fn std_dc_chrominance() -> Self {
        Self::from_bits_values(&STD_DC_CHROMINANCE_BITS, &STD_DC_CHROMINANCE_VALUES)
            .expect("standard table should be valid")
    }

    /// Creates the standard AC luminance decode table.
    #[must_use]
    pub fn std_ac_luminance() -> Self {
        Self::from_bits_values(&STD_AC_LUMINANCE_BITS, &STD_AC_LUMINANCE_VALUES)
            .expect("standard table should be valid")
    }

    /// Creates the standard AC chrominance decode table.
    #[must_use]
    pub fn std_ac_chrominance() -> Self {
        Self::from_bits_values(&STD_AC_CHROMINANCE_BITS, &STD_AC_CHROMINANCE_VALUES)
            .expect("standard table should be valid")
    }
}

impl Default for HuffmanDecodeTable {
    fn default() -> Self {
        Self::new()
    }
}

/// A node in the Huffman tree.
#[derive(Clone)]
struct HuffmanNode {
    /// Total count (frequency)
    total_count: u32,
    /// Left child index (-1 for leaf nodes)
    index_left: i16,
    /// Right child or symbol value (for leaf nodes, the symbol index)
    index_right_or_value: i16,
}

impl HuffmanNode {
    fn new_leaf(count: u32, symbol: i16) -> Self {
        Self {
            total_count: count,
            index_left: -1,
            index_right_or_value: symbol,
        }
    }

    fn is_leaf(&self) -> bool {
        self.index_left < 0
    }
}

/// Set depths recursively from a tree node.
fn set_depth(tree: &[HuffmanNode], node_idx: usize, depth: &mut [u8], level: u8) {
    let node = &tree[node_idx];
    if node.is_leaf() {
        depth[node.index_right_or_value as usize] = level;
    } else {
        set_depth(tree, node.index_left as usize, depth, level + 1);
        set_depth(tree, node.index_right_or_value as usize, depth, level + 1);
    }
}

/// Compare nodes for sorting: by count ascending, then by value descending.
fn compare_nodes(a: &HuffmanNode, b: &HuffmanNode) -> std::cmp::Ordering {
    match a.total_count.cmp(&b.total_count) {
        std::cmp::Ordering::Equal => b.index_right_or_value.cmp(&a.index_right_or_value),
        other => other,
    }
}

/// Builds optimal Huffman code lengths from symbol frequencies.
///
/// This is a port of jpegli's CreateHuffmanTree algorithm which:
/// 1. Builds a Huffman tree with a minimum count threshold
/// 2. If tree exceeds max_length, retries with higher threshold
/// 3. Guarantees all codes fit within max_length bits
///
/// # Arguments
/// * `freqs` - Frequency of each symbol (index = symbol value)
/// * `max_length` - Maximum code length (typically 16 for JPEG)
///
/// # Returns
/// Array of code lengths for each symbol (0 = symbol not present)
pub fn build_code_lengths(freqs: &[u64], max_length: u8) -> Vec<u8> {
    let length = freqs.len();
    let tree_limit = max_length as usize;
    let mut depth = vec![0u8; length];

    // Retry loop with increasing count_limit until tree fits
    let mut count_limit: u32 = 1;
    loop {
        // Build leaf nodes with clamped frequencies
        let mut tree: Vec<HuffmanNode> = Vec::with_capacity(2 * length + 1);

        // Add leaves in reverse order (C++ iterates from length-1 down to 0)
        for i in (0..length).rev() {
            if freqs[i] > 0 {
                let count = (freqs[i] as u32).max(count_limit.saturating_sub(1));
                tree.push(HuffmanNode::new_leaf(count, i as i16));
            }
        }

        let n = tree.len();

        if n == 0 {
            return depth;
        }

        if n == 1 {
            // Single symbol gets depth 1
            depth[tree[0].index_right_or_value as usize] = 1;
            return depth;
        }

        // Sort by count ascending, then by value descending
        tree.sort_by(compare_nodes);

        // Add sentinel nodes (max count, for algorithm termination)
        let sentinel = HuffmanNode {
            total_count: u32::MAX,
            index_left: -1,
            index_right_or_value: -1,
        };
        tree.push(sentinel.clone());
        tree.push(sentinel);

        // Build tree using two-pointer merge
        // i: index into sorted leaves
        // j: index into internal nodes (starts at n+1)
        let mut i = 0;
        let mut j = n + 1;

        for _k in (1..n).rev() {
            // Find two smallest nodes
            let left = if tree[i].total_count <= tree[j].total_count {
                let l = i;
                i += 1;
                l
            } else {
                let l = j;
                j += 1;
                l
            };

            let right = if tree[i].total_count <= tree[j].total_count {
                let r = i;
                i += 1;
                r
            } else {
                let r = j;
                j += 1;
                r
            };

            // Create parent node at the end (replacing sentinel)
            let j_end = tree.len() - 1;
            tree[j_end].total_count = tree[left]
                .total_count
                .saturating_add(tree[right].total_count);
            tree[j_end].index_left = left as i16;
            tree[j_end].index_right_or_value = right as i16;

            // Add new sentinel
            tree.push(HuffmanNode {
                total_count: u32::MAX,
                index_left: -1,
                index_right_or_value: -1,
            });
        }

        // Tree root is at index 2*n - 1
        let root_idx = 2 * n - 1;

        // Reset depths and compute from tree
        for d in &mut depth {
            *d = 0;
        }
        set_depth(&tree, root_idx, &mut depth, 0);

        // Check if we need to retry
        let max_depth = *depth.iter().max().unwrap_or(&0) as usize;
        if max_depth <= tree_limit {
            break;
        }

        // Retry with higher count_limit
        count_limit = count_limit.saturating_mul(2);
        if count_limit > u32::MAX / 2 {
            // Safety limit - just clamp and break
            for d in &mut depth {
                if *d > max_length {
                    *d = max_length;
                }
            }
            break;
        }
    }

    depth
}

/// Converts code lengths to JPEG-format bits and values arrays.
pub fn lengths_to_bits_values(lengths: &[u8]) -> ([u8; 16], Vec<u8>) {
    let mut bits = [0u8; 16];
    let mut symbols_by_length: Vec<Vec<u8>> = vec![Vec::new(); 17];

    for (symbol, &length) in lengths.iter().enumerate() {
        if length > 0 && length <= 16 {
            symbols_by_length[length as usize].push(symbol as u8);
        }
    }

    for i in 1..=16 {
        bits[i - 1] = symbols_by_length[i].len() as u8;
    }

    let values: Vec<u8> = (1..=16)
        .flat_map(|len| symbols_by_length[len].iter().copied())
        .collect();

    (bits, values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_std_dc_luminance_encode() {
        let table = HuffmanEncodeTable::std_dc_luminance();
        // Symbol 0 should have a short code
        let (code, len) = table.encode(0);
        assert!(len > 0 && len <= 16);
        assert!(code < (1 << len));
    }

    #[test]
    fn test_std_ac_luminance_encode() {
        let table = HuffmanEncodeTable::std_ac_luminance();
        // EOB (0x00) should be encoded
        let (code, len) = table.encode(0x00);
        assert!(len > 0 && len <= 16);
    }

    #[test]
    fn test_std_dc_luminance_decode() {
        let table = HuffmanDecodeTable::std_dc_luminance();
        assert!(table.valid);
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let bits = STD_DC_LUMINANCE_BITS;
        let values = STD_DC_LUMINANCE_VALUES;

        let enc = HuffmanEncodeTable::from_bits_values(&bits, &values).unwrap();
        let dec = HuffmanDecodeTable::from_bits_values(&bits, &values).unwrap();

        // Test each symbol
        for &symbol in &values {
            let (code, length) = enc.encode(symbol);
            if length as usize <= HuffmanDecodeTable::FAST_BITS {
                // Shift code to MSB position for fast_decode
                let bits = code << (32 - length);
                if let Some((decoded, dec_len)) = dec.fast_decode(bits) {
                    assert_eq!(decoded, symbol);
                    assert_eq!(dec_len, length);
                }
            }
        }
    }

    #[test]
    fn test_build_code_lengths() {
        let freqs = [100u64, 50, 25, 10, 5];
        let lengths = build_code_lengths(&freqs, 16);

        // Most frequent symbol should have shortest code
        assert!(lengths[0] <= lengths[4]);
    }

    #[test]
    fn test_lengths_to_bits_values() {
        let lengths = [2u8, 2, 3, 3, 3, 0, 0, 0]; // 2 symbols of len 2, 3 of len 3
        let (bits, values) = lengths_to_bits_values(&lengths);

        assert_eq!(bits[1], 2); // 2 symbols of length 2
        assert_eq!(bits[2], 3); // 3 symbols of length 3
        assert_eq!(values.len(), 5);
    }
}
