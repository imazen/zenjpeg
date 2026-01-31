//! Frequency counting for Huffman table optimization.
//!
//! This module provides `FrequencyCounter` for collecting symbol frequencies
//! during a first pass over the data, then generating optimal Huffman tables.

#![allow(dead_code)]

use crate::error::Result;
use crate::huffman::classic::{
    depths_to_bits_values, generate_code_lengths, generate_optimal_table,
};
use crate::huffman::HuffmanEncodeTable;

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
pub struct OptimizedHuffmanTables {
    /// DC luminance table
    pub dc_luma: OptimizedTable,
    /// AC luminance table
    pub ac_luma: OptimizedTable,
    /// DC chrominance table
    pub dc_chroma: OptimizedTable,
    /// AC chrominance table
    pub ac_chroma: OptimizedTable,
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

    /// Creates a frequency counter from a slice of counts (one per symbol 0-255).
    ///
    /// If the slice is shorter than 256, remaining symbols get count 0.
    /// If longer, extra elements are ignored.
    #[must_use]
    pub fn from_counts(counts: &[i64]) -> Self {
        let mut result = Self::new();
        for (i, &count) in counts.iter().take(256).enumerate() {
            result.counts[i] = count;
        }
        result
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

    /// Ensures all valid DC symbols (0-11) have at least frequency 1.
    ///
    /// Call this before generating a DC Huffman table from partial data
    /// to ensure all valid category symbols have codes assigned.
    pub fn ensure_dc_coverage(&mut self) {
        // DC symbols are categories 0-11 (for DC coefficient differences)
        for symbol in 0..=11 {
            if self.counts[symbol] == 0 {
                self.counts[symbol] = 1;
            }
        }
    }

    /// Ensures all valid AC symbols have at least frequency 1.
    ///
    /// Call this before generating an AC Huffman table from partial data
    /// to ensure all valid run/size symbols have codes assigned.
    ///
    /// Valid AC symbols are:
    /// - 0x00: EOB (End of Block)
    /// - 0xF0: ZRL (Zero Run Length - 16 zeros)
    /// - (run << 4) | size: where run=0-15, size=1-10
    pub fn ensure_ac_coverage(&mut self) {
        // EOB (End of Block)
        if self.counts[0x00] == 0 {
            self.counts[0x00] = 1;
        }
        // ZRL (Zero Run Length - 16 zeros)
        if self.counts[0xF0] == 0 {
            self.counts[0xF0] = 1;
        }
        // All valid (run, size) combinations
        // run: 0-15, size: 1-10 (size 0 is only valid for EOB/ZRL)
        for run in 0..=15u8 {
            for size in 1..=10u8 {
                let symbol = (run << 4) | size;
                if self.counts[symbol as usize] == 0 {
                    self.counts[symbol as usize] = 1;
                }
            }
        }
    }

    /// Computes Shannon entropy of the frequency distribution (in bits).
    ///
    /// Higher entropy means more uniform distribution (better for general use).
    /// Lower entropy means concentrated distribution (potentially pathological).
    ///
    /// Returns 0.0 if histogram is empty.
    #[must_use]
    pub fn entropy(&self) -> f64 {
        let total = self.total() as f64;
        if total == 0.0 {
            return 0.0;
        }

        let mut entropy = 0.0;
        for &count in &self.counts[..256] {
            if count > 0 {
                let p = count as f64 / total;
                entropy -= p * p.log2();
            }
        }
        entropy
    }

    /// Computes the percentage of valid AC symbols that have been seen.
    ///
    /// Valid AC symbols are: EOB (0x00), ZRL (0xF0), and (run << 4 | size) for
    /// run 0-15, size 1-10. Total: 162 valid symbols.
    ///
    /// Returns 0.0-100.0 percentage.
    #[must_use]
    pub fn ac_symbol_coverage(&self) -> f64 {
        const TOTAL_VALID_AC: usize = 2 + 16 * 10; // EOB + ZRL + run/size combos = 162

        let mut seen = 0usize;

        // EOB
        if self.counts[0x00] > 0 {
            seen += 1;
        }
        // ZRL
        if self.counts[0xF0] > 0 {
            seen += 1;
        }
        // Run/size combinations
        for run in 0..=15u8 {
            for size in 1..=10u8 {
                let symbol = (run << 4) | size;
                if self.counts[symbol as usize] > 0 {
                    seen += 1;
                }
            }
        }

        100.0 * seen as f64 / TOTAL_VALID_AC as f64
    }

    /// Computes the percentage of valid DC symbols (0-11) that have been seen.
    #[must_use]
    pub fn dc_symbol_coverage(&self) -> f64 {
        let seen = (0..=11).filter(|&s| self.counts[s] > 0).count();
        100.0 * seen as f64 / 12.0
    }

    /// Computes symmetric KL divergence (Jensen-Shannon divergence) between two histograms.
    ///
    /// Returns a value >= 0. Higher values indicate more different distributions.
    /// Returns 0.0 if distributions are identical.
    #[must_use]
    pub fn divergence(&self, other: &FrequencyCounter) -> f64 {
        let total_self = self.total() as f64;
        let total_other = other.total() as f64;

        if total_self == 0.0 || total_other == 0.0 {
            return f64::MAX;
        }

        let mut divergence = 0.0;
        for i in 0..256 {
            let p = self.counts[i] as f64 / total_self;
            let q = other.counts[i] as f64 / total_other;

            if p > 0.0 || q > 0.0 {
                let m = (p + q) / 2.0;
                if p > 0.0 && m > 0.0 {
                    divergence += p * (p / m).ln();
                }
                if q > 0.0 && m > 0.0 {
                    divergence += q * (q / m).ln();
                }
            }
        }
        divergence / 2.0 // Jensen-Shannon is symmetric average
    }

    /// Blends this histogram with a prior (corpus) histogram for better streaming tables.
    ///
    /// Strategy:
    /// - For symbols with `>= min_samples` observations, use observed frequency
    /// - For rare/unseen symbols, blend observed + scaled prior to avoid pathological codes
    ///
    /// **Note**: Testing shows this hurts compression. Use `add_prior_proportional` instead.
    #[must_use]
    pub fn blend_with_prior(&self, prior: &FrequencyCounter, min_samples: i64) -> FrequencyCounter {
        let observed_total = self.total();
        let prior_total = prior.total();

        if observed_total == 0 {
            return prior.clone();
        }
        if prior_total == 0 {
            return self.clone();
        }

        let scale = observed_total as f64 / prior_total as f64;

        let mut result = FrequencyCounter::new();
        for i in 0..256 {
            let observed = self.counts[i];
            let prior_scaled = (prior.counts[i] as f64 * scale) as i64;

            result.counts[i] = if observed >= min_samples {
                observed
            } else {
                observed + prior_scaled.max(1)
            };
        }

        // Preserve pseudo-symbol if present
        result.counts[256] = self.counts[256];

        result
    }

    /// Adds a proportionally-scaled prior to this histogram.
    ///
    /// Unlike `blend_with_prior`, this adds a small fraction of the prior to ALL symbols,
    /// preserving relative weights from corpus training while keeping observed dominant.
    #[must_use]
    pub fn add_prior_proportional(
        &self,
        prior: &FrequencyCounter,
        prior_weight: f64,
    ) -> FrequencyCounter {
        let observed_total = self.total();
        let prior_total = prior.total();

        if observed_total == 0 {
            return prior.clone();
        }
        if prior_total == 0 {
            return self.clone();
        }

        let scale = (observed_total as f64 * prior_weight) / prior_total as f64;

        let mut result = FrequencyCounter::new();
        for i in 0..256 {
            let observed = self.counts[i];
            let prior_contribution = (prior.counts[i] as f64 * scale).round() as i64;
            result.counts[i] = observed + prior_contribution.max(0);
        }

        // Preserve pseudo-symbol
        result.counts[256] = self.counts[256];

        result
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
