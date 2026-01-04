//! Shared types for Huffman code generation.
//!
//! This module provides a unified interface for both Huffman algorithms:
//! - mozjpeg/libjpeg classic (Section K.2) - working, well-tested
//! - jpegli C++ style (sorted merge with retry) - needs optimization work
//!
//! The types here ensure both algorithms have the same contract:
//! - Input: 256 symbol frequencies (pseudo-symbol 256 handled internally)
//! - Output: 256 code lengths, all <= 16 bits
//! - Kraft inequality guaranteed

use crate::error::Result;
use crate::huffman::HuffmanEncodeTable;

/// Symbol frequencies for Huffman table generation.
///
/// Contains counts for symbols 0-255. The pseudo-symbol 256 (used to ensure
/// Kraft sum < 2^16) is handled internally by each algorithm.
#[derive(Clone, Debug)]
pub struct SymbolFrequencies {
    /// Frequency count for each symbol 0-255
    counts: [u64; 256],
}

impl Default for SymbolFrequencies {
    fn default() -> Self {
        Self::new()
    }
}

impl SymbolFrequencies {
    /// Creates a new frequency counter with all counts at zero.
    #[must_use]
    pub fn new() -> Self {
        Self { counts: [0; 256] }
    }

    /// Increments the count for a symbol.
    #[inline]
    pub fn count(&mut self, symbol: u8) {
        self.counts[symbol as usize] += 1;
    }

    /// Adds a count for a symbol.
    #[inline]
    pub fn add(&mut self, symbol: u8, count: u64) {
        self.counts[symbol as usize] += count;
    }

    /// Returns the count for a symbol.
    #[must_use]
    pub fn get(&self, symbol: u8) -> u64 {
        self.counts[symbol as usize]
    }

    /// Returns the total number of symbols counted.
    #[must_use]
    pub fn total(&self) -> u64 {
        self.counts.iter().sum()
    }

    /// Returns the number of distinct symbols with non-zero count.
    #[must_use]
    pub fn num_symbols(&self) -> usize {
        self.counts.iter().filter(|&&c| c > 0).count()
    }

    /// Returns true if all counts are zero.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.counts.iter().all(|&c| c == 0)
    }

    /// Resets all counts to zero.
    pub fn reset(&mut self) {
        self.counts.fill(0);
    }

    /// Merges another frequency counter into this one.
    pub fn merge(&mut self, other: &SymbolFrequencies) {
        for i in 0..256 {
            self.counts[i] = self.counts[i].saturating_add(other.counts[i]);
        }
    }

    /// Returns a reference to the raw counts array.
    #[must_use]
    pub fn as_slice(&self) -> &[u64; 256] {
        &self.counts
    }

    /// Creates from a slice of counts (must be exactly 256 elements).
    pub fn from_slice(counts: &[u64]) -> Option<Self> {
        if counts.len() != 256 {
            return None;
        }
        let mut result = Self::new();
        result.counts.copy_from_slice(counts);
        Some(result)
    }
}

/// Code lengths for symbols 0-255.
///
/// Each length is 0 (symbol not present) or 1-16 (valid Huffman code length).
/// The pseudo-symbol 256 is never included in the output.
#[derive(Clone, Debug)]
pub struct CodeLengths {
    /// Code length for each symbol 0-255
    lengths: [u8; 256],
}

impl Default for CodeLengths {
    fn default() -> Self {
        Self::new()
    }
}

impl CodeLengths {
    /// Creates new code lengths with all zeros (no symbols).
    #[must_use]
    pub fn new() -> Self {
        Self { lengths: [0; 256] }
    }

    /// Creates from a length array.
    #[must_use]
    pub fn from_array(lengths: [u8; 256]) -> Self {
        Self { lengths }
    }

    /// Returns the code length for a symbol.
    #[must_use]
    pub fn get(&self, symbol: u8) -> u8 {
        self.lengths[symbol as usize]
    }

    /// Returns a reference to the raw lengths array.
    #[must_use]
    pub fn as_slice(&self) -> &[u8; 256] {
        &self.lengths
    }

    /// Returns the maximum code length.
    #[must_use]
    pub fn max_length(&self) -> u8 {
        *self.lengths.iter().max().unwrap_or(&0)
    }

    /// Computes the Kraft sum: sum(2^(16-length)) for all symbols.
    /// Must be < 2^16 for a valid prefix-free code.
    #[must_use]
    pub fn kraft_sum(&self) -> u64 {
        self.lengths
            .iter()
            .filter(|&&l| l > 0 && l <= 16)
            .map(|&l| 1u64 << (16 - l as u64))
            .sum()
    }

    /// Returns true if this represents a valid Huffman code.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        // All lengths must be 0 or 1-16
        if self.lengths.iter().any(|&l| l > 16) {
            return false;
        }
        // Kraft sum must be <= 2^16 (< for strict prefix-free with pseudo-symbol)
        self.kraft_sum() <= (1 << 16)
    }

    /// Converts to JPEG DHT format (bits, values).
    ///
    /// - `bits[i]` = number of codes with length i+1 (for lengths 1-16)
    /// - `values` = symbols sorted by (length, symbol) order
    #[must_use]
    pub fn to_bits_values(&self) -> ([u8; 16], Vec<u8>) {
        let mut bits = [0u8; 16];
        let mut symbols_by_length: [Vec<u8>; 17] = Default::default();

        for (symbol, &length) in self.lengths.iter().enumerate() {
            if length > 0 && length <= 16 {
                bits[length as usize - 1] += 1;
                symbols_by_length[length as usize].push(symbol as u8);
            }
        }

        // Sort symbols within each length group for canonical ordering
        for syms in &mut symbols_by_length {
            syms.sort_unstable();
        }

        let values: Vec<u8> = (1..=16)
            .flat_map(|len| symbols_by_length[len].iter().copied())
            .collect();

        (bits, values)
    }

    /// Estimates the total bit cost for encoding with these code lengths.
    #[must_use]
    pub fn estimate_cost(&self, frequencies: &SymbolFrequencies) -> u64 {
        (0..256)
            .map(|i| frequencies.get(i as u8) * self.lengths[i] as u64)
            .sum()
    }
}

/// An optimized Huffman table ready for encoding.
///
/// Contains both the fast lookup table and the DHT marker data.
#[derive(Clone, Debug)]
pub struct OptimizedTable {
    /// Fast encoding table (symbol -> code, length)
    pub encode_table: HuffmanEncodeTable,
    /// Number of codes at each length (1-16 bits) for DHT marker
    pub bits: [u8; 16],
    /// Symbol values in code-length order for DHT marker
    pub values: Vec<u8>,
}

impl OptimizedTable {
    /// Creates an optimized table from code lengths.
    pub fn from_code_lengths(lengths: &CodeLengths) -> Result<Self> {
        let (bits, values) = lengths.to_bits_values();
        let encode_table = HuffmanEncodeTable::from_bits_values(&bits, &values)?;
        Ok(Self {
            encode_table,
            bits,
            values,
        })
    }

    /// Returns the code and length for a symbol.
    #[inline]
    pub fn encode(&self, symbol: u8) -> (u32, u8) {
        self.encode_table.encode(symbol)
    }
}

/// Complete set of Huffman tables for JPEG encoding.
///
/// For YCbCr: separate luma/chroma tables (4 total)
/// For XYB: same table for all components (effectively 2, duplicated)
#[derive(Clone, Debug)]
pub struct HuffmanTableSet {
    /// DC table for luminance (component 0)
    pub dc_luma: OptimizedTable,
    /// AC table for luminance (component 0)
    pub ac_luma: OptimizedTable,
    /// DC table for chrominance (components 1, 2)
    pub dc_chroma: OptimizedTable,
    /// AC table for chrominance (components 1, 2)
    pub ac_chroma: OptimizedTable,
}

impl HuffmanTableSet {
    /// Creates a table set where all components use the same tables (for XYB).
    pub fn merged(dc: OptimizedTable, ac: OptimizedTable) -> Self {
        Self {
            dc_luma: dc.clone(),
            ac_luma: ac.clone(),
            dc_chroma: dc,
            ac_chroma: ac,
        }
    }

    /// Creates a table set with separate luma/chroma tables (for YCbCr).
    pub fn luma_chroma(
        dc_luma: OptimizedTable,
        ac_luma: OptimizedTable,
        dc_chroma: OptimizedTable,
        ac_chroma: OptimizedTable,
    ) -> Self {
        Self {
            dc_luma,
            ac_luma,
            dc_chroma,
            ac_chroma,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbol_frequencies_basic() {
        let mut freq = SymbolFrequencies::new();
        assert!(freq.is_empty());

        freq.count(0);
        freq.count(0);
        freq.count(1);

        assert_eq!(freq.get(0), 2);
        assert_eq!(freq.get(1), 1);
        assert_eq!(freq.get(2), 0);
        assert_eq!(freq.total(), 3);
        assert_eq!(freq.num_symbols(), 2);
    }

    #[test]
    fn test_symbol_frequencies_merge() {
        let mut freq1 = SymbolFrequencies::new();
        freq1.count(0);
        freq1.count(1);

        let mut freq2 = SymbolFrequencies::new();
        freq2.count(0);
        freq2.count(2);

        freq1.merge(&freq2);

        assert_eq!(freq1.get(0), 2);
        assert_eq!(freq1.get(1), 1);
        assert_eq!(freq1.get(2), 1);
    }

    #[test]
    fn test_code_lengths_to_bits_values() {
        let mut lengths = CodeLengths::new();
        lengths.lengths[0] = 2;
        lengths.lengths[1] = 2;
        lengths.lengths[2] = 3;
        lengths.lengths[3] = 3;

        let (bits, values) = lengths.to_bits_values();

        assert_eq!(bits[1], 2); // 2 symbols of length 2
        assert_eq!(bits[2], 2); // 2 symbols of length 3
        assert_eq!(values, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_code_lengths_kraft_sum() {
        let mut lengths = CodeLengths::new();
        // Two symbols of length 1 would have Kraft sum = 2^15 + 2^15 = 2^16 (exactly full)
        lengths.lengths[0] = 1;
        lengths.lengths[1] = 1;

        assert_eq!(lengths.kraft_sum(), 1 << 16);
        assert!(lengths.is_valid()); // Exactly 2^16 is valid (equality)
    }

    #[test]
    fn test_code_lengths_estimate_cost() {
        let mut lengths = CodeLengths::new();
        lengths.lengths[0] = 1; // Short code for frequent symbol
        lengths.lengths[1] = 3; // Longer code for rare symbol

        let mut freq = SymbolFrequencies::new();
        freq.add(0, 100);
        freq.add(1, 10);

        // Cost = 100*1 + 10*3 = 130
        assert_eq!(lengths.estimate_cost(&freq), 130);
    }
}
