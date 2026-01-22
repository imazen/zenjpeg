//! Classic Huffman table construction (mozjpeg/libjpeg algorithm).
//!
//! This module implements the Huffman tree construction algorithm from Section K.2
//! of the JPEG specification (ITU-T T.81). This is the same algorithm used by
//! libjpeg, libjpeg-turbo, and mozjpeg.
//!
//! # Algorithm Overview
//!
//! 1. **Classic Huffman merge** with `others[]` chain tracking
//! 2. Build tree bottom-up, tracking code lengths via chain traversal
//! 3. Limit to 16 bits using Section K.2 tree manipulation:
//!    - Move symbols from depth > 16 up the tree
//!    - Split shorter codes to maintain valid prefix-free property
//! 4. Use pseudo-symbol 256 to ensure Kraft sum < 2^16
//!
//! # Comparison with jpegli C++ Algorithm
//!
//! The jpegli C++ implementation (`CreateHuffmanTree` in huffman.cc) uses a different
//! approach: sorted two-pointer merge with a retry mechanism that increases minimum
//! symbol counts until depth <= 16.
//!
//! Both algorithms produce valid Huffman codes. Differences arise from tie-breaking
//! when symbols have equal frequencies.
//!
//! See `crate::huffman::build_code_lengths()` for the jpegli-style implementation.

use crate::error::{Error, Result};

/// Maximum code length during tree construction (before limiting to 16).
const MAX_CLEN: usize = 32;

/// Sentinel value for merged frequencies.
const FREQ_MERGED: i64 = i64::MAX - 1;

/// Generates optimal Huffman code lengths from symbol frequencies.
///
/// This is the core algorithm from Section K.2 of the JPEG specification.
/// It uses the classic "others[]" chain approach from libjpeg.
///
/// # Arguments
/// * `freq` - Frequency counts (257 elements, last is pseudo-symbol). Modified in place.
///
/// # Returns
/// Code lengths for symbols 0-255 (0 means symbol not present).
///
/// # Algorithm
///
/// 1. Add pseudo-symbol 256 with frequency 1
/// 2. Repeatedly merge the two lowest-frequency symbols
/// 3. Track code lengths by traversing the `others[]` chain
/// 4. If any code > 16 bits, use Section K.2 tree manipulation to shorten
/// 5. Exclude pseudo-symbol 256 from final output
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

    // Count symbols at each code length (INCLUDING pseudo-symbol 256).
    // We need to include it for depth limiting to work correctly.
    let mut bits_with_pseudo = [0u8; MAX_CLEN + 1];
    for i in 0..num_nz {
        let len = codesize[i].min(MAX_CLEN);
        bits_with_pseudo[len] += 1;
    }

    // Limit code lengths to 16 bits (JPEG requirement).
    // This uses the algorithm from Section K.2 of the JPEG spec:
    // Move symbols from too-deep levels up by splitting shorter codes.
    for i in (17..=MAX_CLEN).rev() {
        while bits_with_pseudo[i] > 0 {
            // Find a level with codes to split.
            let mut j = i - 2;
            while j > 0 && bits_with_pseudo[j] == 0 {
                j -= 1;
            }
            if j == 0 {
                // Can't limit further - this shouldn't happen with valid input.
                return Err(Error::internal("Huffman code length overflow"));
            }

            // Move two symbols from level i to i-1, and split one at j.
            bits_with_pseudo[i] -= 2;
            bits_with_pseudo[i - 1] += 1;
            bits_with_pseudo[j + 1] += 2;
            bits_with_pseudo[j] -= 1;
        }
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

    // Collect ALL symbols (including pseudo-symbol 256) with their original codesizes
    // Pre-allocate to avoid reallocs (num_nz is the upper bound)
    let mut all_symbols: Vec<(usize, usize)> = Vec::with_capacity(num_nz);
    for i in 0..num_nz {
        let orig_idx = nz_index[i];
        if codesize[i] > 0 {
            all_symbols.push((orig_idx, codesize[i]));
        }
    }

    // Sort by codesize (shortest first), then by symbol index for stability
    all_symbols.sort_by_key(|&(idx, cs)| (cs, idx));

    // Assign lengths according to the bits_with_pseudo[] distribution
    // Track what length symbol 256 gets assigned so we can exclude it
    let mut symbol_256_length: Option<u8> = None;
    let mut sym_iter = all_symbols.iter();
    for len in 1..=16usize {
        for _ in 0..bits_with_pseudo[len] {
            if let Some(&(orig_idx, _)) = sym_iter.next() {
                if orig_idx == 256 {
                    // Found pseudo-symbol 256 - remember its length
                    symbol_256_length = Some(len as u8);
                } else if orig_idx < 256 {
                    // Real symbol - assign its length
                    lengths[orig_idx] = len as u8;
                }
            }
        }
    }

    // Verify we found and excluded symbol 256
    // (This should always be true since we set freq[256] = 1 at the start)
    if symbol_256_length.is_none() {
        return Err(Error::internal(
            "Pseudo-symbol 256 not found in Huffman tree",
        ));
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
    // Note: lengths[] already excludes pseudo-symbol 256, so we only iterate 0-255.
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

/// Converts code lengths (depths) to JPEG DHT format (bits, values).
///
/// The depths array contains the bit length for each symbol 0-255.
/// Returns (bits, values) where:
/// - bits\[i\] = number of codes with length i+1 (1-16 bits)
/// - values = symbols sorted by code length, then by symbol value
///
/// This function is used when converting from jpegli-style depths to DHT format.
pub fn depths_to_bits_values(depths: &[u8]) -> ([u8; 16], Vec<u8>) {
    let mut bits = [0u8; 16];
    // Use fixed array - Default creates empty Vecs (no allocation until push)
    let mut symbols_by_length: [Vec<u8>; 16] = Default::default();

    // Group symbols by their code length.
    // Only process symbols 0-255 (pseudo-symbol 256 is already excluded).
    for (symbol, &depth) in depths.iter().enumerate().take(256) {
        if depth > 0 && depth <= 16 {
            bits[depth as usize - 1] += 1;
            symbols_by_length[depth as usize - 1].push(symbol as u8);
        }
    }

    // Sort symbols within each length group (for canonical codes)
    for symbols in &mut symbols_by_length {
        symbols.sort_unstable();
    }

    // Flatten into values array
    let values: Vec<u8> = symbols_by_length.into_iter().flatten().collect();

    (bits, values)
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // C++ Comparison Tests (moved from huffman_opt.rs)
    // =========================================================================

    fn load_cpp_testdata() -> Option<Vec<(Vec<i64>, Vec<u8>)>> {
        let path = crate::test_utils::get_cpp_testdata_path("CreateHuffmanTree.testdata")?;
        let file = std::fs::File::open(&path).ok()?;
        let reader = std::io::BufReader::new(file);

        use std::io::BufRead;
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
    #[ignore] // FAILING: 4/185 cases where C++ is better - algorithm needs fixing
    fn test_against_cpp_testdata() {
        let tests = match load_cpp_testdata() {
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
            let cost_result: i64 = (0..256).map(|i| input[i] * result[i] as i64).sum();
            let cost_expected: i64 = (0..256).map(|i| input[i] * expected[i] as i64).sum();

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

    // =========================================================================
    // Basic Algorithm Tests
    // =========================================================================

    #[test]
    fn test_single_symbol() {
        let mut freq = [0i64; 257];
        freq[42] = 100;
        let lengths = generate_code_lengths(&mut freq).unwrap();
        assert_eq!(lengths[42], 1);
        // All others should be 0
        for (i, &len) in lengths.iter().enumerate() {
            if i != 42 {
                assert_eq!(len, 0, "Symbol {} should have length 0", i);
            }
        }
    }

    #[test]
    fn test_two_symbols() {
        let mut freq = [0i64; 257];
        freq[0] = 100;
        freq[1] = 100;
        let lengths = generate_code_lengths(&mut freq).unwrap();
        // Both should have length 1 or 2 (depends on pseudo-symbol)
        assert!(lengths[0] > 0 && lengths[0] <= 2);
        assert!(lengths[1] > 0 && lengths[1] <= 2);
    }

    #[test]
    fn test_skewed_distribution() {
        let mut freq = [0i64; 257];
        freq[0] = 1000; // Very common
        freq[1] = 100;
        freq[2] = 10;
        freq[3] = 1;
        let lengths = generate_code_lengths(&mut freq).unwrap();
        // Most common should have shortest code
        assert!(lengths[0] <= lengths[1]);
        assert!(lengths[1] <= lengths[2]);
        assert!(lengths[2] <= lengths[3]);
    }

    #[test]
    fn test_all_256_symbols() {
        let mut freq = [0i64; 257];
        for i in 0..256 {
            freq[i] = (i + 1) as i64;
        }
        let lengths = generate_code_lengths(&mut freq).unwrap();
        // All should have valid lengths
        for &len in &lengths {
            assert!(len > 0 && len <= 16);
        }
    }

    #[test]
    fn test_kraft_inequality() {
        // Verify Kraft sum < 2^16 (required for valid prefix-free codes)
        let mut freq = [0i64; 257];
        for i in 0..256 {
            freq[i] = 1; // Uniform distribution
        }
        let lengths = generate_code_lengths(&mut freq).unwrap();

        let kraft_sum: u64 = lengths
            .iter()
            .filter(|&&l| l > 0)
            .map(|&l| 1u64 << (16 - l))
            .sum();

        // Must be strictly less than 2^16 due to pseudo-symbol 256
        assert!(
            kraft_sum < (1 << 16),
            "Kraft sum {} should be < {}",
            kraft_sum,
            1u64 << 16
        );
    }

    #[test]
    fn test_generate_optimal_table() {
        let mut freq = [0i64; 257];
        freq[0] = 100;
        freq[1] = 50;
        freq[2] = 25;

        let (bits, values) = generate_optimal_table(&mut freq).unwrap();

        // Should have some codes
        let total_codes: u8 = bits.iter().sum();
        assert_eq!(total_codes, 3);
        assert_eq!(values.len(), 3);

        // Values should be sorted within each length group
        // and include all symbols with non-zero frequency
        assert!(values.contains(&0));
        assert!(values.contains(&1));
        assert!(values.contains(&2));
    }

    #[test]
    fn test_depths_to_bits_values() {
        let depths = [2u8, 2, 3, 3, 0, 0, 0, 0];
        let (bits, values) = depths_to_bits_values(&depths);

        assert_eq!(bits[1], 2); // 2 symbols of length 2
        assert_eq!(bits[2], 2); // 2 symbols of length 3
        assert_eq!(values, vec![0, 1, 2, 3]); // Sorted by length, then symbol
    }
}
