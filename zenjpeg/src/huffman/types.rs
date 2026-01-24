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

#![allow(dead_code)]

use crate::error::Result;
use crate::huffman::optimize::OptimizedTable;

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

// =============================================================================
// Huffman Algorithm Trait and Implementations
// =============================================================================

/// Which Huffman algorithm to use for code length generation.
///
/// Both algorithms produce valid Huffman codes that satisfy:
/// - All code lengths <= 16 bits
/// - Kraft sum < 2^16 (strictly less, due to pseudo-symbol 256)
///
/// They differ in:
/// - Tie-breaking when symbols have equal frequency
/// - Depth limiting strategy (retry vs tree manipulation)
/// - Performance characteristics
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum HuffmanAlgorithm {
    /// mozjpeg/libjpeg classic algorithm (JPEG Section K.2).
    ///
    /// This is the well-tested, standard algorithm used by libjpeg since 1991.
    /// Uses `others[]` chain tracking and Section K.2 tree manipulation for
    /// depth limiting.
    ///
    /// **Status**: Working, produces valid bitstreams.
    #[default]
    MozjpegClassic,

    /// jpegli C++ algorithm (sorted two-pointer merge with retry).
    ///
    /// This is the algorithm from Google's jpegli in the JPEG XL project.
    /// Uses sorted merge with sentinels. If tree exceeds depth limit, retries
    /// with boosted minimum frequencies.
    ///
    /// **Status**: Needs investigation - may produce larger files than expected.
    JpegliTree,
}

impl HuffmanAlgorithm {
    /// Generates optimal code lengths from symbol frequencies.
    ///
    /// Both algorithms handle the pseudo-symbol 256 internally, so the input
    /// only needs frequencies for symbols 0-255. The output excludes symbol 256.
    ///
    /// # Errors
    /// Returns an error if the algorithm fails (only possible with mozjpeg on overflow).
    pub fn generate_code_lengths(&self, frequencies: &SymbolFrequencies) -> Result<CodeLengths> {
        match self {
            HuffmanAlgorithm::MozjpegClassic => generate_lengths_mozjpeg(frequencies),
            HuffmanAlgorithm::JpegliTree => generate_lengths_jpegli(frequencies),
        }
    }

    /// Generates an optimized table from symbol frequencies.
    pub fn generate_table(&self, frequencies: &SymbolFrequencies) -> Result<OptimizedTable> {
        let lengths = self.generate_code_lengths(frequencies)?;
        let (bits, values) = lengths.to_bits_values();
        OptimizedTable::from_bits_values(bits, values)
    }
}

/// mozjpeg/libjpeg algorithm implementation.
fn generate_lengths_mozjpeg(frequencies: &SymbolFrequencies) -> Result<CodeLengths> {
    use crate::huffman::classic::generate_code_lengths;

    // Convert to the format mozjpeg expects: [i64; 257] with pseudo-symbol
    let mut freq = [0i64; 257];
    for (i, &count) in frequencies.as_slice().iter().enumerate() {
        freq[i] = count as i64;
    }
    // generate_code_lengths sets freq[256] = 1 internally

    let lengths_array = generate_code_lengths(&mut freq)?;
    Ok(CodeLengths::from_array(lengths_array))
}

/// jpegli C++ algorithm implementation.
fn generate_lengths_jpegli(frequencies: &SymbolFrequencies) -> Result<CodeLengths> {
    use crate::huffman::build_code_lengths;

    // Convert to the format jpegli expects: Vec<u64> with pseudo-symbol appended
    let mut freqs: Vec<u64> = frequencies.as_slice().to_vec();
    freqs.push(1); // Pseudo-symbol 256

    let depths = build_code_lengths(&freqs, 16);

    // depths includes symbol 256, but we only want 0-255
    let mut lengths = [0u8; 256];
    lengths.copy_from_slice(&depths[..256]);

    Ok(CodeLengths::from_array(lengths))
}

/// Compares the two algorithms on the same frequencies.
///
/// Returns (mozjpeg_lengths, jpegli_lengths, mozjpeg_cost, jpegli_cost).
/// Lower cost = better compression.
pub fn compare_algorithms(
    frequencies: &SymbolFrequencies,
) -> Result<(CodeLengths, CodeLengths, u64, u64)> {
    let mozjpeg = HuffmanAlgorithm::MozjpegClassic.generate_code_lengths(frequencies)?;
    let jpegli = HuffmanAlgorithm::JpegliTree.generate_code_lengths(frequencies)?;

    let mozjpeg_cost = mozjpeg.estimate_cost(frequencies);
    let jpegli_cost = jpegli.estimate_cost(frequencies);

    Ok((mozjpeg, jpegli, mozjpeg_cost, jpegli_cost))
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

    #[test]
    fn test_algorithm_mozjpeg_produces_valid_code() {
        let mut freq = SymbolFrequencies::new();
        freq.add(0, 1000); // Very common
        freq.add(1, 100);
        freq.add(2, 10);
        freq.add(3, 1);

        let lengths = HuffmanAlgorithm::MozjpegClassic
            .generate_code_lengths(&freq)
            .unwrap();

        assert!(lengths.is_valid());
        assert!(lengths.max_length() <= 16);
        // Most frequent should have shortest code
        assert!(lengths.get(0) <= lengths.get(3));
    }

    #[test]
    fn test_algorithm_jpegli_produces_valid_code() {
        let mut freq = SymbolFrequencies::new();
        freq.add(0, 1000);
        freq.add(1, 100);
        freq.add(2, 10);
        freq.add(3, 1);

        let lengths = HuffmanAlgorithm::JpegliTree
            .generate_code_lengths(&freq)
            .unwrap();

        assert!(lengths.is_valid());
        assert!(lengths.max_length() <= 16);
        // Most frequent should have shortest code
        assert!(lengths.get(0) <= lengths.get(3));
    }

    #[test]
    fn test_algorithm_comparison_same_input() {
        let mut freq = SymbolFrequencies::new();
        // Realistic AC histogram pattern
        freq.add(0, 10000); // EOB - very common
        freq.add(1, 5000); // Small coefficients
        freq.add(17, 3000); // Run=1, size=1
        freq.add(33, 2000); // Run=2, size=1
        for i in 2..16 {
            freq.add(i, 1000 / (i as u64 + 1));
        }

        let (mozjpeg, jpegli, moz_cost, jpg_cost) = compare_algorithms(&freq).unwrap();

        // Both should be valid
        assert!(mozjpeg.is_valid());
        assert!(jpegli.is_valid());

        // Print comparison for debugging
        println!("mozjpeg cost: {} bits", moz_cost);
        println!("jpegli cost:  {} bits", jpg_cost);
        println!(
            "difference:   {} bits ({:.2}%)",
            (moz_cost as i64 - jpg_cost as i64).abs(),
            ((moz_cost as f64 - jpg_cost as f64) / moz_cost as f64 * 100.0).abs()
        );

        // Costs should be close (within 1%)
        let max_diff = (moz_cost.max(jpg_cost) as f64 * 0.01) as u64 + 1;
        assert!(
            (moz_cost as i64 - jpg_cost as i64).unsigned_abs() <= max_diff,
            "Costs differ by more than 1%: mozjpeg={}, jpegli={}",
            moz_cost,
            jpg_cost
        );
    }

    #[test]
    fn test_optimized_table_from_frequencies() {
        let mut freq = SymbolFrequencies::new();
        freq.add(0, 100);
        freq.add(1, 50);
        freq.add(2, 25);

        let table = HuffmanAlgorithm::MozjpegClassic
            .generate_table(&freq)
            .unwrap();

        // Should be able to encode all symbols
        let (code0, len0) = table.encode(0);
        let (code1, len1) = table.encode(1);
        let (_code2, len2) = table.encode(2);

        assert!(len0 > 0);
        assert!(len1 > 0);
        assert!(len2 > 0);
        // Most frequent should have shortest code
        assert!(len0 <= len2);
        // Codes should be distinct
        assert!(code0 != code1 || len0 != len1);
    }
}
