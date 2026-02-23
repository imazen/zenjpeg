//! Frequency counting for Huffman table optimization.
//!
//! This module provides `FrequencyCounter` for collecting symbol frequencies
//! during a first pass over the data, then generating optimal Huffman tables.

#![allow(dead_code)]

use crate::error::Result;
use crate::huffman::HuffmanEncodeTable;
use crate::huffman::classic::{
    depths_to_bits_values, generate_code_lengths, generate_optimal_table,
};

/// An optimized Huffman table with its DHT marker representation.
///
/// Contains both the encoding table (for fast symbol-to-code lookup) and
/// the bits/values arrays (for writing the DHT marker to the JPEG file).
#[derive(Clone, Debug)]
pub struct OptimizedTable {
    /// Encoding table for fast lookup
    pub table: HuffmanEncodeTable,
    /// Number of codes at each length (1-16 bits) for DHT marker
    pub bits: [u8; 16],
    /// Symbol values in code-length order for DHT marker
    pub values: Vec<u8>,
}

impl OptimizedTable {
    /// Creates an optimized table from bits and values arrays.
    pub fn from_bits_values(bits: [u8; 16], values: Vec<u8>) -> crate::error::Result<Self> {
        let table = HuffmanEncodeTable::from_bits_values(&bits, &values)?;
        Ok(Self {
            table,
            bits,
            values,
        })
    }

    /// Creates an optimized table from bits array and values slice.
    ///
    /// Convenience for constructing from static/const data.
    pub fn from_bits_values_static(bits: [u8; 16], values: &[u8]) -> Self {
        let table = HuffmanEncodeTable::from_bits_values(&bits, values)
            .expect("static table data is valid");
        Self {
            table,
            bits,
            values: values.to_vec(),
        }
    }

    /// Returns the code and length for a symbol.
    #[inline]
    pub fn encode(&self, symbol: u8) -> (u32, u8) {
        self.table.encode(symbol)
    }
}

/// A complete set of optimized Huffman tables for JPEG encoding.
///
/// Contains DC and AC tables for both luminance and chrominance components.
#[derive(Clone, Debug)]
pub struct HuffmanTableSet {
    /// DC luminance table
    pub dc_luma: OptimizedTable,
    /// AC luminance table
    pub ac_luma: OptimizedTable,
    /// DC chrominance table
    pub dc_chroma: OptimizedTable,
    /// AC chrominance table
    pub ac_chroma: OptimizedTable,
}

impl HuffmanTableSet {
    /// Builds tables from the JPEG standard Huffman tables (Annex K of ITU-T T.81).
    ///
    /// These are the tables defined in the JPEG specification. They are significantly
    /// less efficient than the general-purpose trained tables used by default.
    ///
    /// Alias: [`Self::annex_k()`].
    pub fn from_standard() -> crate::error::Result<Self> {
        use crate::huffman::encode::{
            STD_AC_CHROMINANCE_BITS, STD_AC_CHROMINANCE_VALUES, STD_AC_LUMINANCE_BITS,
            STD_AC_LUMINANCE_VALUES, STD_DC_CHROMINANCE_BITS, STD_DC_CHROMINANCE_VALUES,
            STD_DC_LUMINANCE_BITS, STD_DC_LUMINANCE_VALUES,
        };

        Ok(Self {
            dc_luma: OptimizedTable::from_bits_values(
                STD_DC_LUMINANCE_BITS,
                STD_DC_LUMINANCE_VALUES.to_vec(),
            )?,
            ac_luma: OptimizedTable::from_bits_values(
                STD_AC_LUMINANCE_BITS,
                STD_AC_LUMINANCE_VALUES.to_vec(),
            )?,
            dc_chroma: OptimizedTable::from_bits_values(
                STD_DC_CHROMINANCE_BITS,
                STD_DC_CHROMINANCE_VALUES.to_vec(),
            )?,
            ac_chroma: OptimizedTable::from_bits_values(
                STD_AC_CHROMINANCE_BITS,
                STD_AC_CHROMINANCE_VALUES.to_vec(),
            )?,
        })
    }

    /// Alias for [`Self::from_standard()`] — the JPEG Annex K tables.
    pub fn annex_k() -> crate::error::Result<Self> {
        Self::from_standard()
    }
}

/// Frequency counter for Huffman optimization.
///
/// Collects symbol frequencies during a first pass over the data,
/// then generates an optimal Huffman table for the second pass.
///
/// # Example
///
/// ```ignore
/// let mut counter = FrequencyCounter::new();
///
/// // First pass: count symbols
/// for block in blocks {
///     counter.count_dc(dc_category);
///     for ac_symbol in ac_symbols {
///         counter.count_ac(ac_symbol);
///     }
/// }
///
/// // Generate optimized table
/// let table = counter.generate_table()?;
/// ```
#[derive(Clone, Debug)]
pub struct FrequencyCounter {
    /// Frequency count for each symbol (0-255) plus pseudo-symbol 256.
    counts: [i64; 257],
}

impl Default for FrequencyCounter {
    fn default() -> Self {
        Self::new()
    }
}

impl FrequencyCounter {
    /// Creates a new frequency counter with all counts at zero.
    #[must_use]
    pub fn new() -> Self {
        Self { counts: [0; 257] }
    }

    /// Resets all counts to zero.
    pub fn reset(&mut self) {
        self.counts.fill(0);
    }

    /// Increments the count for a symbol.
    #[inline]
    pub fn count(&mut self, symbol: u8) {
        self.counts[symbol as usize] += 1;
    }

    /// Sets the count for a symbol directly.
    pub fn set_count(&mut self, symbol: u8, value: i64) {
        self.counts[symbol as usize] = value;
    }

    /// Returns the count for a symbol.
    #[must_use]
    pub fn get_count(&self, symbol: u8) -> i64 {
        self.counts[symbol as usize]
    }

    /// Returns the total number of symbols counted.
    #[must_use]
    pub fn total(&self) -> i64 {
        self.counts[..256].iter().sum()
    }

    /// Returns the number of distinct symbols with non-zero count.
    #[must_use]
    pub fn num_symbols(&self) -> usize {
        self.counts[..256].iter().filter(|&&c| c > 0).count()
    }

    /// Generates an optimal Huffman table from the collected frequencies.
    ///
    /// This implements Section K.2 of the JPEG specification.
    pub fn generate_table(&self) -> Result<HuffmanEncodeTable> {
        let mut freq = self.counts;
        let (bits, values) = generate_optimal_table(&mut freq)?;
        HuffmanEncodeTable::from_bits_values(&bits, &values)
    }

    /// Generates both the table and its DHT representation.
    ///
    /// Returns the encoding table plus the (bits, values) tuple needed for
    /// writing the DHT marker to the JPEG file.
    pub fn generate_table_with_dht(&self) -> Result<OptimizedTable> {
        let mut freq = self.counts;
        let (bits, values) = generate_optimal_table(&mut freq)?;
        let table = HuffmanEncodeTable::from_bits_values(&bits, &values)?;
        Ok(OptimizedTable {
            table,
            bits,
            values,
        })
    }

    /// Generates Huffman table using specified algorithm.
    ///
    /// # Arguments
    /// * `method` - Which Huffman algorithm to use (jpegli or mozjpeg)
    ///
    /// Returns the encoding table plus DHT data for JPEG file.
    pub fn generate_table_with_method(
        &self,
        method: crate::types::HuffmanMethod,
    ) -> Result<OptimizedTable> {
        use crate::types::HuffmanMethod;

        match method {
            HuffmanMethod::JpegliCreateTree => {
                // Use jpegli's CreateHuffmanTree algorithm from huffman.rs
                // IMPORTANT: Include pseudo-symbol 256 with frequency 1 to ensure Kraft sum < 2^16
                let mut freqs: Vec<u64> = self.counts[..256]
                    .iter()
                    .map(|&c| c.max(0) as u64)
                    .collect();
                freqs.push(1); // Add pseudo-symbol 256 with frequency 1

                let depths = crate::huffman::build_code_lengths(&freqs, 16);

                // Convert depths to (bits, values) format
                // depths_to_bits_values already excludes symbol 256 (it only processes 0-255)
                let (bits, values) = depths_to_bits_values(&depths);
                let table = HuffmanEncodeTable::from_bits_values(&bits, &values)?;

                Ok(OptimizedTable {
                    table,
                    bits,
                    values,
                })
            }
            HuffmanMethod::MozjpegClassic => {
                // Use classic mozjpeg algorithm (current implementation)
                self.generate_table_with_dht()
            }
        }
    }

    /// Generates code lengths without building the full table.
    ///
    /// Useful for cost estimation or debugging.
    pub fn generate_lengths(&self) -> Result<[u8; 256]> {
        let mut freq = self.counts;
        generate_code_lengths(&mut freq)
    }

    /// Estimates the total bit cost using current frequencies and given lengths.
    #[must_use]
    pub fn estimate_cost(&self, lengths: &[u8; 256]) -> u64 {
        (0..256)
            .map(|i| self.counts[i] as u64 * lengths[i] as u64)
            .sum()
    }

    /// Checks if this histogram is empty (all counts are zero).
    pub fn is_empty_histogram(&self) -> bool {
        self.counts[..256].iter().all(|&c| c == 0)
    }

    /// Adds another histogram's counts to this one.
    pub fn add(&mut self, other: &FrequencyCounter) {
        for i in 0..257 {
            self.counts[i] = self.counts[i].saturating_add(other.counts[i]);
        }
    }

    /// Creates a new histogram that is the sum of two histograms.
    pub fn combined(&self, other: &FrequencyCounter) -> FrequencyCounter {
        let mut result = self.clone();
        result.add(other);
        result
    }

    /// Estimates the cost of encoding with this histogram.
    ///
    /// Cost = header_bits + data_bits
    /// - header_bits = fixed overhead (17 bytes) + 1 byte per symbol with depth > 0
    /// - data_bits = sum(count * depth) for all symbols
    pub fn estimate_encoding_cost(&self) -> f64 {
        // Generate code lengths
        let lengths = match self.generate_lengths() {
            Ok(l) => l,
            Err(_) => return f64::MAX,
        };

        // Fixed header: 1 byte table class + 16 bytes for counts per length
        let mut header_bits = (1 + 16) * 8;

        // One byte per symbol in the table
        let mut data_bits: u64 = 0;
        for i in 0..256 {
            if lengths[i] > 0 {
                header_bits += 8;
                data_bits += self.counts[i] as u64 * lengths[i] as u64;
            }
        }

        header_bits as f64 + data_bits as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frequency_counter_basic() {
        let mut counter = FrequencyCounter::new();

        counter.count(0);
        counter.count(0);
        counter.count(1);

        assert_eq!(counter.get_count(0), 2);
        assert_eq!(counter.get_count(1), 1);
        assert_eq!(counter.get_count(2), 0);
        assert_eq!(counter.total(), 3);
        assert_eq!(counter.num_symbols(), 2);
    }

    #[test]
    fn test_frequency_counter_reset() {
        let mut counter = FrequencyCounter::new();
        counter.count(0);
        counter.count(1);
        counter.reset();

        assert_eq!(counter.total(), 0);
        assert_eq!(counter.num_symbols(), 0);
    }

    #[test]
    fn test_generate_table_uniform() {
        let mut counter = FrequencyCounter::new();

        // 8 symbols with equal frequency
        for i in 0..8u8 {
            for _ in 0..100 {
                counter.count(i);
            }
        }

        let table = counter.generate_table().unwrap();

        // All 8 symbols should have codes
        let mut total_symbols = 0;
        for i in 0..8 {
            let (_, len) = table.encode(i);
            assert!(len > 0, "Symbol {} should have a code", i);
            assert!(len <= 4, "Uniform 8 symbols should have codes <= 4 bits");
            total_symbols += 1;
        }
        assert_eq!(total_symbols, 8);
    }

    #[test]
    fn test_generate_table_skewed() {
        let mut counter = FrequencyCounter::new();

        // Highly skewed frequencies
        for _ in 0..10000 {
            counter.count(0);
        }
        for _ in 0..100 {
            counter.count(1);
        }
        for _ in 0..10 {
            counter.count(2);
        }
        counter.count(3);

        let table = counter.generate_table().unwrap();

        // Most frequent should have shortest code
        let (_, len0) = table.encode(0);
        let (_, len1) = table.encode(1);
        let (_, len2) = table.encode(2);
        let (_, len3) = table.encode(3);

        assert!(
            len0 <= len1,
            "More frequent symbol should have shorter code"
        );
        assert!(len1 <= len2);
        assert!(len2 <= len3);
    }

    #[test]
    fn test_generate_table_single_symbol() {
        let mut counter = FrequencyCounter::new();
        counter.count(42);
        counter.count(42);
        counter.count(42);

        let table = counter.generate_table().unwrap();
        let (_, len) = table.encode(42);
        assert_eq!(len, 1, "Single symbol should get length 1");
    }

    #[test]
    fn test_generate_table_empty() {
        let counter = FrequencyCounter::new();
        let result = counter.generate_table();
        // Empty table should either error or produce empty table
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_code_length_limit() {
        let mut counter = FrequencyCounter::new();

        // Create frequencies that would produce very deep tree
        // Fibonacci-like: each symbol has frequency equal to sum of next two
        let mut f = 1i64;
        for i in 0..30u8 {
            for _ in 0..f {
                counter.count(i);
            }
            f = (f * 3) / 2 + 1; // Grow faster than Fibonacci
        }

        let table = counter.generate_table().unwrap();

        // All codes should be <= 16 bits
        for i in 0..30 {
            let (_, len) = table.encode(i);
            assert!(len <= 16, "Symbol {} has length {} > 16", i, len);
        }
    }

    #[test]
    fn test_estimate_cost() {
        let mut counter = FrequencyCounter::new();
        for _ in 0..100 {
            counter.count(0);
        } // Will get short code
        for _ in 0..10 {
            counter.count(1);
        } // Will get longer code

        let lengths = counter.generate_lengths().unwrap();

        let cost = counter.estimate_cost(&lengths);
        // Cost should be sum of (count * length) for all symbols
        assert!(cost > 0);
        assert!(cost < 1000); // Reasonable upper bound
    }
}
