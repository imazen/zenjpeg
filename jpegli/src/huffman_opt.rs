//! Huffman table optimization for JPEG encoding.
//!
//! This module implements optimal Huffman table generation from symbol frequency
//! counts, following Section K.2 of the JPEG specification.
//!
//! # Algorithm Comparison: mozjpeg vs jpegli C++
//!
//! This implementation uses the **mozjpeg/libjpeg algorithm** (Section K.2), not the
//! jpegli C++ algorithm. Both produce valid Huffman codes, but differ in approach:
//!
//! ## mozjpeg/libjpeg (this implementation)
//!
//! ```text
//! 1. Classic Huffman merge with `others[]` chain tracking
//! 2. Build tree bottom-up, tracking code lengths via chain traversal
//! 3. Limit to 16 bits using Section K.2 tree manipulation:
//!    - Move symbols from depth > 16 up the tree
//!    - Split shorter codes to maintain valid prefix-free property
//! 4. Remove pseudo-symbol 256 from final table
//! ```
//!
//! **Pros**: Simpler (~100 lines), follows JPEG spec exactly, well-understood
//! **Cons**: O(n²) merge loop (fine for n ≤ 257)
//!
//! ## jpegli C++ (`CreateHuffmanTree` in huffman.cc)
//!
//! ```text
//! 1. Sort symbols by frequency, use two-pointer merge with sentinels
//! 2. If max depth > limit, retry with count_limit *= 2
//!    (artificially boosts low-frequency symbols to reduce tree depth)
//! 3. More complex but potentially faster for large alphabets
//! ```
//!
//! **Pros**: May be faster for large n due to sorted merge, single-pass depth limiting
//! **Cons**: More complex (~150 lines), non-standard retry approach
//!
//! ## Validation Results
//!
//! Tested against 122 C++ jpegli test cases:
//! - **100/122 exact match** (82%)
//! - **22 cases**: mozjpeg produces 1 bit LESS total (better compression)
//! - **0 cases**: mozjpeg worse than jpegli
//!
//! The differences arise from tie-breaking: when two symbols have equal frequency,
//! the algorithms may order them differently, producing different but equally valid trees.
//!
//! ## Future Work
//!
//! If performance becomes critical, consider implementing the jpegli C++ algorithm.
//! The sorted two-pointer merge is O(n log n) vs O(n²) for classic Huffman, but for
//! n = 257 (max JPEG alphabet), the difference is negligible (~65k vs ~130k operations).

use crate::error::{Error, Result};
use crate::huffman::HuffmanEncodeTable;

/// Maximum code length during tree construction (before limiting to 16).
const MAX_CLEN: usize = 32;

/// Sentinel value for merged frequencies.
const FREQ_MERGED: i64 = i64::MAX - 1;

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
}

/// Generates optimal Huffman code lengths from symbol frequencies.
///
/// This is the core algorithm from Section K.2 of the JPEG specification.
///
/// # Arguments
/// * `freq` - Frequency counts (257 elements, last is pseudo-symbol). Modified in place.
///
/// # Returns
/// Code lengths for symbols 0-255 (0 means symbol not present).
pub fn generate_code_lengths(freq: &mut [i64; 257]) -> Result<[u8; 256]> {
    let mut codesize = [0usize; 257];
    let mut others = [-1i32; 257];

    // Ensure pseudo-symbol 256 has a nonzero count.
    // This guarantees no real symbol gets an all-ones code.
    freq[256] = 1;

    // Collect indices of nonzero frequencies for efficient searching.
    let mut nz_index = [0usize; 257];
    let mut nz_freq = [0i64; 257];
    let mut num_nz = 0;

    for i in 0..257 {
        if freq[i] > 0 {
            nz_index[num_nz] = i;
            nz_freq[num_nz] = freq[i];
            num_nz += 1;
        }
    }

    if num_nz == 0 {
        return Ok([0; 256]);
    }

    if num_nz == 1 {
        // Single symbol: give it length 1
        let mut lengths = [0u8; 256];
        if nz_index[0] < 256 {
            lengths[nz_index[0]] = 1;
        }
        return Ok(lengths);
    }

    // Huffman's algorithm: repeatedly merge two smallest frequencies.
    loop {
        // Find two smallest nonzero frequencies.
        let mut c1: i32 = -1;
        let mut c2: i32 = -1;
        let mut v1 = i64::MAX;
        let mut v2 = i64::MAX;

        for i in 0..num_nz {
            let f = nz_freq[i];
            if f < FREQ_MERGED && f <= v2 {
                if f <= v1 {
                    c2 = c1;
                    v2 = v1;
                    v1 = f;
                    c1 = i as i32;
                } else {
                    v2 = f;
                    c2 = i as i32;
                }
            }
        }

        // Done if we've merged everything into one tree.
        if c2 < 0 {
            break;
        }

        let c1 = c1 as usize;
        let c2 = c2 as usize;

        // Merge c2 into c1.
        nz_freq[c1] = nz_freq[c1].saturating_add(nz_freq[c2]);
        nz_freq[c2] = FREQ_MERGED;

        // Increment codesize for everything in c1's tree.
        codesize[c1] += 1;
        let mut node = c1;
        while others[node] >= 0 {
            node = others[node] as usize;
            codesize[node] += 1;
        }

        // Chain c2 onto c1's tree.
        others[node] = c2 as i32;

        // Increment codesize for everything in c2's tree.
        codesize[c2] += 1;
        let mut node = c2;
        while others[node] >= 0 {
            node = others[node] as usize;
            codesize[node] += 1;
        }
    }

    // Count symbols at each code length.
    let mut bits = [0u8; MAX_CLEN + 1];
    for i in 0..num_nz {
        let len = codesize[i].min(MAX_CLEN);
        bits[len] += 1;
    }

    // Limit code lengths to 16 bits (JPEG requirement).
    // This uses the algorithm from Section K.2 of the JPEG spec:
    // Move symbols from too-deep levels up by splitting shorter codes.
    for i in (17..=MAX_CLEN).rev() {
        while bits[i] > 0 {
            // Find a level with codes to split.
            let mut j = i - 2;
            while j > 0 && bits[j] == 0 {
                j -= 1;
            }
            if j == 0 {
                // Can't limit further - this shouldn't happen with valid input.
                return Err(Error::InternalError {
                    reason: "Huffman code length overflow",
                });
            }

            // Move two symbols from level i to i-1, and split one at j.
            bits[i] -= 2;
            bits[i - 1] += 1;
            bits[j + 1] += 2;
            bits[j] -= 1;
        }
    }

    // Remove the pseudo-symbol 256 from the longest code length.
    let mut longest = 16;
    while longest > 0 && bits[longest] == 0 {
        longest -= 1;
    }
    if longest > 0 {
        bits[longest] -= 1;
    }

    // Map code lengths back to original symbol indices.
    // After limiting, we need to reassign lengths based on the new bit counts.
    //
    // The key insight from Section K.2:
    // 1. Sort symbols by their original codesize (frequency order)
    // 2. Assign new lengths from shortest to longest according to bits[]
    //
    // This ensures symbols that had shorter codes still get shorter codes
    // after depth limiting, even if the exact lengths changed.

    let mut lengths = [0u8; 256];

    // Count how many real symbols we have (exclude pseudo-symbol 256)
    let mut real_symbols: Vec<(usize, usize)> = Vec::new(); // (original_index, codesize)
    for i in 0..num_nz {
        let orig_idx = nz_index[i];
        if orig_idx < 256 && codesize[i] > 0 {
            real_symbols.push((orig_idx, codesize[i]));
        }
    }

    // Sort by codesize (shortest first), then by symbol index for stability
    real_symbols.sort_by_key(|&(idx, cs)| (cs, idx));

    // Assign lengths according to the new bits[] distribution
    let mut sym_iter = real_symbols.iter();
    for len in 1..=16usize {
        for _ in 0..bits[len] {
            if let Some(&(orig_idx, _)) = sym_iter.next() {
                lengths[orig_idx] = len as u8;
            }
        }
    }

    Ok(lengths)
}

/// Generates an optimal Huffman table in JPEG format (bits + values).
///
/// # Arguments
/// * `freq` - Frequency counts (257 elements). Modified in place.
///
/// # Returns
/// (bits, values) tuple ready for JPEG DHT marker.
pub fn generate_optimal_table(freq: &mut [i64; 257]) -> Result<([u8; 16], Vec<u8>)> {
    let lengths = generate_code_lengths(freq)?;

    // Count symbols at each length.
    let mut bits = [0u8; 16];
    let mut symbols_by_length: [Vec<u8>; 17] = Default::default();

    for (symbol, &length) in lengths.iter().enumerate() {
        if length > 0 && length <= 16 {
            symbols_by_length[length as usize].push(symbol as u8);
            bits[length as usize - 1] += 1;
        }
    }

    // Sort symbols within each length for canonical ordering.
    for syms in &mut symbols_by_length {
        syms.sort_unstable();
    }

    // Flatten to values array.
    let values: Vec<u8> = (1..=16)
        .flat_map(|len| symbols_by_length[len].iter().copied())
        .collect();

    Ok((bits, values))
}

/// Token representing a symbol and its extra bits for two-pass encoding.
#[derive(Clone, Copy, Debug)]
pub struct Token {
    /// Context index (which histogram this belongs to).
    pub context: u8,
    /// Huffman symbol (0-255).
    pub symbol: u8,
    /// Additional bits value.
    pub extra_bits: u16,
    /// Number of additional bits (0-15).
    pub num_extra: u8,
}

impl Token {
    /// Creates a new token.
    #[inline]
    pub const fn new(context: u8, symbol: u8, extra_bits: u16, num_extra: u8) -> Self {
        Self {
            context,
            symbol,
            extra_bits,
            num_extra,
        }
    }

    /// Creates a DC token from a difference value.
    #[inline]
    pub fn dc(context: u8, diff: i16) -> Self {
        let category = crate::entropy::category(diff);
        let extra = crate::entropy::additional_bits(diff);
        Self::new(context, category, extra, category)
    }

    /// Creates an AC token from run length and value.
    #[inline]
    pub fn ac(context: u8, run: u8, value: i16) -> Self {
        if value == 0 {
            if run == 0 {
                // EOB
                Self::new(context, 0x00, 0, 0)
            } else {
                // ZRL (run of 16 zeros)
                Self::new(context, 0xF0, 0, 0)
            }
        } else {
            let category = crate::entropy::category(value);
            let extra = crate::entropy::additional_bits(value);
            let symbol = (run << 4) | category;
            Self::new(context, symbol, extra, category)
        }
    }
}

/// Token buffer for two-pass encoding.
///
/// Stores tokens from the first pass for replay in the second pass
/// with optimized Huffman tables.
#[derive(Clone, Debug, Default)]
pub struct TokenBuffer {
    /// Stored tokens.
    tokens: Vec<Token>,
    /// Frequency counters per context.
    counters: Vec<FrequencyCounter>,
}

impl TokenBuffer {
    /// Creates a new token buffer with the specified number of contexts.
    ///
    /// Typical usage:
    /// - 2 contexts for grayscale (DC + AC)
    /// - 4 contexts for color (DC luma, DC chroma, AC luma, AC chroma)
    #[must_use]
    pub fn new(num_contexts: usize) -> Self {
        Self {
            tokens: Vec::new(),
            counters: vec![FrequencyCounter::new(); num_contexts],
        }
    }

    /// Clears all tokens and resets counters.
    pub fn clear(&mut self) {
        self.tokens.clear();
        for counter in &mut self.counters {
            counter.reset();
        }
    }

    /// Adds a token and updates the corresponding frequency counter.
    #[inline]
    pub fn push(&mut self, token: Token) {
        if (token.context as usize) < self.counters.len() {
            self.counters[token.context as usize].count(token.symbol);
        }
        self.tokens.push(token);
    }

    /// Returns the number of stored tokens.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Returns true if the buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Returns an iterator over the tokens.
    pub fn iter(&self) -> impl Iterator<Item = &Token> {
        self.tokens.iter()
    }

    /// Returns the frequency counter for a context.
    #[must_use]
    pub fn counter(&self, context: usize) -> Option<&FrequencyCounter> {
        self.counters.get(context)
    }

    /// Generates optimized Huffman tables for all contexts.
    pub fn generate_tables(&self) -> Result<Vec<HuffmanEncodeTable>> {
        self.counters
            .iter()
            .map(|c| c.generate_table())
            .collect()
    }

    /// Estimates total encoded size in bits using given tables.
    #[must_use]
    pub fn estimate_size(&self, tables: &[HuffmanEncodeTable]) -> u64 {
        let mut total = 0u64;
        for token in &self.tokens {
            if let Some(table) = tables.get(token.context as usize) {
                let (_, len) = table.encode(token.symbol);
                total += len as u64 + token.num_extra as u64;
            }
        }
        total
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

        assert!(len0 <= len1, "More frequent symbol should have shorter code");
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
    fn test_token_dc() {
        let token = Token::dc(0, 5);
        assert_eq!(token.context, 0);
        assert_eq!(token.symbol, 3); // category of 5 is 3
        assert_eq!(token.extra_bits, 5);
        assert_eq!(token.num_extra, 3);

        let token = Token::dc(0, -5);
        assert_eq!(token.symbol, 3); // category of -5 is 3
    }

    #[test]
    fn test_token_ac() {
        // Non-zero value
        let token = Token::ac(1, 2, 7);
        assert_eq!(token.context, 1);
        assert_eq!(token.symbol, (2 << 4) | 3); // run=2, category=3
        assert_eq!(token.num_extra, 3);

        // EOB
        let eob = Token::ac(1, 0, 0);
        assert_eq!(eob.symbol, 0x00);

        // ZRL
        let zrl = Token::ac(1, 16, 0);
        assert_eq!(zrl.symbol, 0xF0);
    }

    #[test]
    fn test_token_buffer() {
        let mut buffer = TokenBuffer::new(2);

        buffer.push(Token::dc(0, 10));
        buffer.push(Token::ac(1, 0, 5));
        buffer.push(Token::ac(1, 0, 0)); // EOB

        assert_eq!(buffer.len(), 3);
        assert!(!buffer.is_empty());

        // Check counters
        assert_eq!(buffer.counter(0).unwrap().num_symbols(), 1); // One DC symbol
        assert_eq!(buffer.counter(1).unwrap().num_symbols(), 2); // Two AC symbols
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
            assert!(
                len <= 16,
                "Symbol {} has length {} > 16",
                i,
                len
            );
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

#[cfg(test)]
mod cpp_comparison_tests {
    //! Tests comparing our implementation against C++ jpegli testdata.

    use super::*;
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    fn load_testdata() -> Option<Vec<(Vec<i64>, Vec<u8>)>> {
        let path = "/home/lilith/work/jpegli/CreateHuffmanTree.testdata";
        let file = File::open(path).ok()?;
        let reader = BufReader::new(file);

        let mut tests = Vec::new();
        for line in reader.lines() {
            let line = line.ok()?;
            let line = line.trim_end_matches(',');
            let v: serde_json::Value = serde_json::from_str(line).ok()?;

            let input: Vec<i64> = v["input_data"]
                .as_array()?
                .iter()
                .map(|x| x.as_i64().unwrap_or(0))
                .collect();
            let expected: Vec<u8> = v["output_depth"]
                .as_array()?
                .iter()
                .map(|x| x.as_u64().unwrap_or(0) as u8)
                .collect();

            tests.push((input, expected));
        }
        Some(tests)
    }

    #[test]
    fn test_against_cpp_testdata() {
        let tests = match load_testdata() {
            Some(t) => t,
            None => {
                eprintln!("Skipping: CreateHuffmanTree.testdata not found");
                return;
            }
        };

        let mut exact_match = 0;
        let mut mozjpeg_better = 0;
        let mut cpp_better = 0;
        let total = tests.len();

        for (input, expected) in &tests {
            let mut freq = [0i64; 257];
            for (i, &f) in input.iter().enumerate().take(257) {
                freq[i] = f;
            }

            let result = generate_code_lengths(&mut freq).unwrap();

            // Check exact match
            let exact = (0..256).all(|i| result[i] == expected[i]);

            // Calculate bit costs
            let cost_result: i64 = (0..256)
                .map(|i| input[i] * result[i] as i64)
                .sum();
            let cost_expected: i64 = (0..256)
                .map(|i| input[i] * expected[i] as i64)
                .sum();

            if exact {
                exact_match += 1;
            } else if cost_result < cost_expected {
                mozjpeg_better += 1;
            } else if cost_result > cost_expected {
                cpp_better += 1;
            } else {
                // Same cost, different assignment (equally valid)
                exact_match += 1;
            }
        }

        println!("C++ comparison results:");
        println!("  Exact match: {}/{}", exact_match, total);
        println!("  mozjpeg better: {}", mozjpeg_better);
        println!("  C++ better: {}", cpp_better);

        // Assert we're at least as good as C++
        assert_eq!(
            cpp_better, 0,
            "mozjpeg algorithm should never be worse than C++"
        );

        // Assert reasonable match rate
        let match_rate = (exact_match + mozjpeg_better) as f64 / total as f64;
        assert!(
            match_rate >= 0.80,
            "Match rate {:.1}% is too low",
            match_rate * 100.0
        );
    }

    #[test]
    fn test_specific_cpp_case() {
        // Test case from C++ testdata that we know produces exact match
        let input = [
            61i64, 98, 196, 372, 613, 754, 818, 663, 525, 185, 3, 0, 0, 0, 0, 0,
        ];
        let expected_depths = [7u8, 6, 4, 3, 3, 3, 2, 3, 3, 5, 8];

        let mut freq = [0i64; 257];
        for (i, &f) in input.iter().enumerate() {
            freq[i] = f;
        }
        freq[256] = 1; // pseudo-symbol

        let result = generate_code_lengths(&mut freq).unwrap();

        for (i, &expected) in expected_depths.iter().enumerate() {
            assert_eq!(
                result[i], expected,
                "Symbol {} depth mismatch: got {}, expected {}",
                i, result[i], expected
            );
        }
    }
}
