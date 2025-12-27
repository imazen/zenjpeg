//! Huffman table construction for JPEG encoding.
//!
//! This module implements:
//! - Building derived Huffman tables for encoding (Figure C.1-C.3 of JPEG spec)
//! - Generating optimal Huffman tables from symbol frequency counts (Section K.2)
//!
//! The derived table format stores code and length indexed by symbol value,
//! allowing O(1) lookup during encoding.

use crate::error::{Error, Result};

/// Maximum code length allowed by JPEG (16 bits)
pub const MAX_CODE_LENGTH: usize = 16;

/// Maximum code length during Huffman tree construction
const MAX_CLEN: usize = 32;

/// Number of Huffman table slots (0-3)
pub const NUM_HUFF_TBLS: usize = 4;

/// A Huffman table in the raw format (bits + values).
///
/// This is the format stored in the JPEG file and used as input
/// to build derived tables.
#[derive(Clone, Debug)]
pub struct HuffTable {
    /// Number of codes of each length (`bits[0]` is unused, `bits[1-16]` are counts)
    pub bits: [u8; 17],
    /// Symbol values in order of increasing code length
    pub huffval: [u8; 256],
}

impl Default for HuffTable {
    fn default() -> Self {
        Self {
            bits: [0; 17],
            huffval: [0; 256],
        }
    }
}

/// Derived Huffman table optimized for encoding.
///
/// This format allows O(1) lookup of the code for any symbol.
#[derive(Clone, Debug)]
pub struct DerivedTable {
    /// Huffman code for each symbol (indexed by symbol value)
    pub ehufco: [u32; 256],
    /// Code length for each symbol (0 means no code assigned)
    pub ehufsi: [u8; 256],
}

impl Default for DerivedTable {
    fn default() -> Self {
        Self {
            ehufco: [0; 256],
            ehufsi: [0; 256],
        }
    }
}

impl DerivedTable {
    /// Create a new empty derived table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a derived table from a raw Huffman table.
    ///
    /// This implements Figures C.1-C.3 of the JPEG specification.
    ///
    /// # Arguments
    /// * `htbl` - The raw Huffman table (bits + huffval)
    /// * `is_dc` - True for DC tables (max symbol 15), false for AC (max symbol 255)
    ///
    /// # Returns
    /// The derived table ready for encoding, or an error if the table is invalid.
    pub fn from_huff_table(htbl: &HuffTable, is_dc: bool) -> Result<Self> {
        let mut dtbl = Self::new();

        // Figure C.1: make table of Huffman code length for each symbol
        let mut huffsize = [0u8; 257];
        let mut p = 0usize;

        for l in 1..=16 {
            let count = htbl.bits[l] as usize;
            if p + count > 256 {
                return Err(Error::InvalidHuffmanTable);
            }
            for _ in 0..count {
                huffsize[p] = l as u8;
                p += 1;
            }
        }
        let lastp = p;

        // Figure C.2: generate the codes themselves
        let mut huffcode = [0u32; 257];
        let mut code = 0u32;
        let mut si = huffsize[0] as usize;
        p = 0;

        while p < lastp && huffsize[p] != 0 {
            while p < lastp && huffsize[p] as usize == si {
                huffcode[p] = code;
                code += 1;
                p += 1;
            }
            // Check that code still fits in si bits
            if code >= (1 << si) {
                return Err(Error::InvalidHuffmanTable);
            }
            code <<= 1;
            si += 1;
        }

        // Figure C.3: generate encoding tables indexed by symbol
        // Set all codeless symbols to have code length 0
        // (allows detecting attempts to emit undefined symbols)

        // Check max symbol: DC uses 0..15, AC uses 0..255
        let max_symbol = if is_dc { 15 } else { 255 };

        for i in 0..lastp {
            let symbol = htbl.huffval[i] as usize;
            if symbol > max_symbol {
                return Err(Error::InvalidHuffmanTable);
            }
            if dtbl.ehufsi[symbol] != 0 {
                // Duplicate symbol
                return Err(Error::InvalidHuffmanTable);
            }
            dtbl.ehufco[symbol] = huffcode[i];
            dtbl.ehufsi[symbol] = huffsize[i];
        }

        Ok(dtbl)
    }

    /// Get the code and length for a symbol.
    ///
    /// # Returns
    /// (code, length) tuple, or (0, 0) if symbol has no code.
    #[inline]
    pub fn get_code(&self, symbol: u8) -> (u32, u8) {
        let idx = symbol as usize;
        (self.ehufco[idx], self.ehufsi[idx])
    }
}

/// Generate an optimal Huffman table from symbol frequency counts.
///
/// This implements Section K.2 of the JPEG specification. The algorithm
/// builds a Huffman tree from the frequencies, then limits code lengths
/// to 16 bits as required by JPEG.
///
/// # Arguments
/// * `freq` - Frequency count for each symbol (257 elements, last is pseudo-symbol)
///
/// # Returns
/// A raw Huffman table (bits + huffval) ready for encoding.
///
/// # Notes
/// - The frequency array is modified during processing
/// - Symbol 256 is a pseudo-symbol that ensures no real symbol gets all-ones code
#[allow(clippy::needless_range_loop)]
pub fn generate_optimal_table(freq: &mut [i64; 257]) -> Result<HuffTable> {
    let mut htbl = HuffTable::default();
    let mut bits = [0u8; MAX_CLEN + 1];
    let mut bit_pos = [0usize; MAX_CLEN + 1];
    let mut codesize = [0usize; 257];
    let mut nz_index = [0usize; 257];
    let mut others = [-1i32; 257];

    // Ensure pseudo-symbol 256 has a nonzero count
    // This guarantees no real symbol gets all-ones code
    freq[256] = 1;

    // Group nonzero frequencies together for efficient searching
    let mut num_nz_symbols = 0;
    for i in 0..257 {
        if freq[i] > 0 {
            nz_index[num_nz_symbols] = i;
            freq[num_nz_symbols] = freq[i];
            num_nz_symbols += 1;
        }
    }

    // Sentinel values for frequency comparison (like the C code)
    const FREQ_INITIAL_MAX: i64 = 1_000_000_000;
    const FREQ_MERGED: i64 = 1_000_000_001;

    // Huffman's algorithm: repeatedly merge two smallest frequencies
    loop {
        // Find two smallest nonzero frequencies
        let mut c1: i32 = -1;
        let mut c2: i32 = -1;
        let mut v1 = FREQ_INITIAL_MAX;
        let mut v2 = FREQ_INITIAL_MAX;

        for i in 0..num_nz_symbols {
            if freq[i] <= v2 {
                if freq[i] <= v1 {
                    c2 = c1;
                    v2 = v1;
                    v1 = freq[i];
                    c1 = i as i32;
                } else {
                    v2 = freq[i];
                    c2 = i as i32;
                }
            }
        }

        // Done if we've merged everything into one frequency
        if c2 < 0 {
            break;
        }

        let c1 = c1 as usize;
        let c2 = c2 as usize;

        // Merge the two counts/trees
        freq[c1] += freq[c2];
        // Mark c2 as merged (high value so it won't be selected)
        freq[c2] = FREQ_MERGED;

        // Increment codesize of everything in c1's tree branch
        codesize[c1] += 1;
        let mut node = c1;
        while others[node] >= 0 {
            node = others[node] as usize;
            codesize[node] += 1;
        }

        // Chain c2 onto c1's tree branch
        others[node] = c2 as i32;

        // Increment codesize of everything in c2's tree branch
        codesize[c2] += 1;
        let mut node = c2;
        while others[node] >= 0 {
            node = others[node] as usize;
            codesize[node] += 1;
        }
    }

    // Count the number of symbols of each code length
    for i in 0..num_nz_symbols {
        if codesize[i] > MAX_CLEN {
            return Err(Error::HuffmanCodeLengthOverflow);
        }
        bits[codesize[i]] += 1;
    }

    // Limit code lengths to 16 bits (JPEG requirement)
    // This uses the algorithm from Section K.2 of the JPEG spec
    for i in (17..=MAX_CLEN).rev() {
        while bits[i] > 0 {
            // Find length of new prefix to be used
            let mut j = i - 2;
            while j > 0 && bits[j] == 0 {
                j -= 1;
            }
            if j == 0 {
                return Err(Error::HuffmanCodeLengthOverflow);
            }

            bits[i] -= 2; // Remove two symbols
            bits[i - 1] += 1; // One goes to length i-1
            bits[j + 1] += 2; // Two new symbols at length j+1
            bits[j] -= 1; // Symbol at length j becomes a prefix
        }
    }

    // Remove the count for pseudo-symbol 256 from the largest codelength
    let mut i = 16;
    while i > 0 && bits[i] == 0 {
        i -= 1;
    }
    if i > 0 {
        bits[i] -= 1;
    }

    // Copy final symbol counts to output table
    htbl.bits[0] = 0;
    htbl.bits[1..=16].copy_from_slice(&bits[1..=16]);

    // Compute bit_pos AFTER depth limiting - cumulative count of symbols at shorter lengths
    // This gives us the starting position in huffval for each code length
    let mut p = 0usize;
    for i in 1..=16 {
        bit_pos[i] = p;
        p += bits[i] as usize;
    }

    // Create sorted list of symbols by code length.
    //
    // IMPORTANT: After depth limiting, bits[] has been modified but codesize[]
    // still has the original (pre-limiting) values. We must:
    // 1. Sort symbols by their original codesize (to preserve frequency order)
    // 2. Assign new lengths from bits[] (which now has the depth-limited distribution)
    //
    // This ensures symbols that had shorter codes still get shorter codes
    // after depth limiting, even if the exact lengths changed.

    // Build list of (original_nz_index, codesize) pairs, excluding pseudo-symbol 256
    let mut symbols: Vec<(usize, usize)> = (0..num_nz_symbols)
        .filter(|&i| nz_index[i] < 256 && codesize[i] > 0)
        .map(|i| (nz_index[i], codesize[i]))
        .collect();

    // Sort by codesize (shortest first), then by symbol index for stability
    symbols.sort_by_key(|&(idx, cs)| (cs, idx));

    // Assign symbols to huffval according to the new bits[] distribution
    let mut sym_iter = symbols.iter();
    for len in 1..=16usize {
        for _ in 0..bits[len] {
            if let Some(&(orig_idx, _)) = sym_iter.next() {
                let pos = bit_pos[len];
                htbl.huffval[pos] = orig_idx as u8;
                bit_pos[len] += 1;
            }
        }
    }

    Ok(htbl)
}

/// Count symbol frequencies for Huffman optimization.
///
/// This is a helper for building frequency tables that will be passed
/// to `generate_optimal_table`.
#[derive(Clone, Debug)]
pub struct FrequencyCounter {
    /// Frequency count for each symbol (257 elements for pseudo-symbol)
    pub counts: [i64; 257],
}

impl Default for FrequencyCounter {
    fn default() -> Self {
        Self { counts: [0; 257] }
    }
}

impl FrequencyCounter {
    /// Create a new frequency counter.
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset all counts to zero.
    pub fn reset(&mut self) {
        self.counts.fill(0);
    }

    /// Increment the count for a symbol.
    #[inline]
    pub fn count(&mut self, symbol: u8) {
        self.counts[symbol as usize] += 1;
    }

    /// Generate an optimal Huffman table from the collected frequencies.
    pub fn generate_table(&mut self) -> Result<HuffTable> {
        generate_optimal_table(&mut self.counts)
    }
}

/// Create standard DC luminance Huffman table.
pub fn std_dc_luma() -> HuffTable {
    use crate::consts::{DC_LUMINANCE_BITS, DC_LUMINANCE_VALUES};
    let mut htbl = HuffTable::default();
    htbl.bits.copy_from_slice(&DC_LUMINANCE_BITS);
    for (i, &v) in DC_LUMINANCE_VALUES.iter().enumerate() {
        htbl.huffval[i] = v;
    }
    htbl
}

/// Create standard AC luminance Huffman table.
pub fn std_ac_luma() -> HuffTable {
    use crate::consts::{AC_LUMINANCE_BITS, AC_LUMINANCE_VALUES};
    let mut htbl = HuffTable::default();
    htbl.bits.copy_from_slice(&AC_LUMINANCE_BITS);
    for (i, &v) in AC_LUMINANCE_VALUES.iter().enumerate() {
        htbl.huffval[i] = v;
    }
    htbl
}

/// Create standard DC chrominance Huffman table.
pub fn std_dc_chroma() -> HuffTable {
    use crate::consts::{DC_CHROMINANCE_BITS, DC_CHROMINANCE_VALUES};
    let mut htbl = HuffTable::default();
    htbl.bits.copy_from_slice(&DC_CHROMINANCE_BITS);
    for (i, &v) in DC_CHROMINANCE_VALUES.iter().enumerate() {
        htbl.huffval[i] = v;
    }
    htbl
}

/// Create standard AC chrominance Huffman table.
pub fn std_ac_chroma() -> HuffTable {
    use crate::consts::{AC_CHROMINANCE_BITS, AC_CHROMINANCE_VALUES};
    let mut htbl = HuffTable::default();
    htbl.bits.copy_from_slice(&AC_CHROMINANCE_BITS);
    for (i, &v) in AC_CHROMINANCE_VALUES.iter().enumerate() {
        htbl.huffval[i] = v;
    }
    htbl
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consts::{
        AC_CHROMINANCE_BITS, AC_CHROMINANCE_VALUES, AC_LUMINANCE_BITS, AC_LUMINANCE_VALUES,
        DC_CHROMINANCE_BITS, DC_CHROMINANCE_VALUES, DC_LUMINANCE_BITS, DC_LUMINANCE_VALUES,
    };

    fn create_std_dc_luma_table() -> HuffTable {
        let mut htbl = HuffTable::default();
        htbl.bits.copy_from_slice(&DC_LUMINANCE_BITS);
        for (i, &v) in DC_LUMINANCE_VALUES.iter().enumerate() {
            htbl.huffval[i] = v;
        }
        htbl
    }

    fn create_std_ac_luma_table() -> HuffTable {
        let mut htbl = HuffTable::default();
        htbl.bits.copy_from_slice(&AC_LUMINANCE_BITS);
        for (i, &v) in AC_LUMINANCE_VALUES.iter().enumerate() {
            htbl.huffval[i] = v;
        }
        htbl
    }

    #[test]
    fn test_build_dc_derived_table() {
        let htbl = create_std_dc_luma_table();
        let dtbl = DerivedTable::from_huff_table(&htbl, true).unwrap();

        // Verify some known codes from the standard DC luminance table
        // Symbol 0 should have code 00 (2 bits)
        assert_eq!(dtbl.ehufsi[0], 2);
        assert_eq!(dtbl.ehufco[0], 0b00);

        // Symbol 1 should have code 010 (3 bits)
        assert_eq!(dtbl.ehufsi[1], 3);
        assert_eq!(dtbl.ehufco[1], 0b010);

        // Symbol 2 should have code 011 (3 bits)
        assert_eq!(dtbl.ehufsi[2], 3);
        assert_eq!(dtbl.ehufco[2], 0b011);
    }

    #[test]
    fn test_build_ac_derived_table() {
        let htbl = create_std_ac_luma_table();
        let dtbl = DerivedTable::from_huff_table(&htbl, false).unwrap();

        // Symbol 0x00 (EOB) should have code 1010 (4 bits)
        assert_eq!(dtbl.ehufsi[0x00], 4);
        assert_eq!(dtbl.ehufco[0x00], 0b1010);

        // Symbol 0x01 should have code 00 (2 bits)
        assert_eq!(dtbl.ehufsi[0x01], 2);
        assert_eq!(dtbl.ehufco[0x01], 0b00);

        // Symbol 0xF0 (ZRL) should have code 11111111001 (11 bits)
        assert_eq!(dtbl.ehufsi[0xF0], 11);
    }

    #[test]
    fn test_generate_optimal_table_uniform() {
        // Test with uniform frequencies
        let mut freq = [0i64; 257];
        for i in 0..8 {
            freq[i] = 100;
        }

        let htbl = generate_optimal_table(&mut freq).unwrap();

        // Should produce 8 codes (one for each real symbol)
        // Note: The pseudo-symbol 256 is included during tree construction
        // but removed at the end, so we should have exactly 8 symbols
        let total_symbols: u8 = htbl.bits.iter().sum();
        assert_eq!(total_symbols, 8, "Should have 8 symbols total");

        // Most symbols should have short codes (3-4 bits for 8 uniform symbols)
        // The exact distribution depends on how the pseudo-symbol affects the tree
        let short_codes: u8 = htbl.bits[1..=4].iter().sum();
        assert_eq!(
            short_codes, 8,
            "All 8 symbols should have codes of 4 bits or less"
        );
    }

    #[test]
    fn test_generate_optimal_table_skewed() {
        // Test with highly skewed frequencies
        let mut freq = [0i64; 257];
        freq[0] = 1000; // Very common
        freq[1] = 100; // Less common
        freq[2] = 10; // Rare
        freq[3] = 1; // Very rare

        let htbl = generate_optimal_table(&mut freq).unwrap();

        // Most frequent symbol should have shorter code
        let dtbl = DerivedTable::from_huff_table(&htbl, true).unwrap();

        // Symbol 0 should have the shortest code
        let len0 = dtbl.ehufsi[0];
        let len1 = dtbl.ehufsi[1];
        let len2 = dtbl.ehufsi[2];
        let len3 = dtbl.ehufsi[3];

        assert!(len0 <= len1);
        assert!(len1 <= len2);
        assert!(len2 <= len3);
    }

    #[test]
    fn test_frequency_counter() {
        let mut counter = FrequencyCounter::new();

        // Count some symbols
        for _ in 0..100 {
            counter.count(0);
        }
        for _ in 0..50 {
            counter.count(1);
        }
        counter.count(2);

        let htbl = counter.generate_table().unwrap();

        // Verify table was generated
        let total_symbols: u8 = htbl.bits.iter().sum();
        assert_eq!(total_symbols, 3);
    }

    #[test]
    fn test_invalid_table_too_many_symbols() {
        let mut htbl = HuffTable::default();
        // Set impossible counts that would overflow
        htbl.bits[1] = 255;
        htbl.bits[2] = 10;

        let result = DerivedTable::from_huff_table(&htbl, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_dc_symbol_range() {
        let mut htbl = HuffTable::default();
        htbl.bits[1] = 1;
        htbl.huffval[0] = 16; // Invalid for DC (max is 15)

        let result = DerivedTable::from_huff_table(&htbl, true);
        assert!(result.is_err());

        // But valid for AC
        let result = DerivedTable::from_huff_table(&htbl, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_progressive_symbols_in_ac_table() {
        let htbl = create_std_ac_luma_table();
        let dtbl = DerivedTable::from_huff_table(&htbl, false).unwrap();

        // Check essential symbols for progressive refinement encoding
        // Note: Standard AC table from JPEG spec doesn't include all EOBn symbols.
        // Higher EOBn (0x70-0xE0) are for long EOBRUN and may be missing.
        // This is why progressive JPEG should use optimized Huffman tables.

        // EOB0 (0x00) - essential for any JPEG
        assert!(dtbl.ehufsi[0x00] > 0, "EOB0 (0x00) must be present");

        // ZRL (0xF0) - essential for long zero runs
        assert!(dtbl.ehufsi[0xF0] > 0, "ZRL (0xF0) must be present");

        // Run/Size=1 symbols commonly used in progressive refinement
        // At minimum, (0 << 4) | 1 = 0x01 should be present (most common)
        assert!(dtbl.ehufsi[0x01] > 0, "Symbol 0x01 (run=0, size=1) must be present");

        // Document which progressive symbols are missing (informational)
        let mut missing_eobn = Vec::new();
        for n in 1..15u8 {
            let symbol = n << 4;
            if dtbl.ehufsi[symbol as usize] == 0 {
                missing_eobn.push(n);
            }
        }

        if !missing_eobn.is_empty() {
            // This is expected for standard tables - just document it
            println!(
                "Note: Standard AC table missing EOBn for n={:?}. \
                 Use optimized Huffman tables for better progressive compression.",
                missing_eobn
            );
        }
    }

    #[test]
    fn test_chrominance_tables() {
        // Build derived tables for chrominance
        let mut dc_htbl = HuffTable::default();
        dc_htbl.bits.copy_from_slice(&DC_CHROMINANCE_BITS);
        for (i, &v) in DC_CHROMINANCE_VALUES.iter().enumerate() {
            dc_htbl.huffval[i] = v;
        }

        let mut ac_htbl = HuffTable::default();
        ac_htbl.bits.copy_from_slice(&AC_CHROMINANCE_BITS);
        for (i, &v) in AC_CHROMINANCE_VALUES.iter().enumerate() {
            ac_htbl.huffval[i] = v;
        }

        let dc_dtbl = DerivedTable::from_huff_table(&dc_htbl, true).unwrap();
        let ac_dtbl = DerivedTable::from_huff_table(&ac_htbl, false).unwrap();

        // Verify tables were built successfully
        assert!(dc_dtbl.ehufsi[0] > 0);
        assert!(ac_dtbl.ehufsi[0] > 0);
    }
}
