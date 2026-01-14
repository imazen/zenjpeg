//! AC Refinement Scan Parity Tests
//!
//! These tests verify that Rust's AC refinement encoding matches C++ jpegli exactly.
//! AC refinement scans are the most complex part of progressive JPEG encoding.
//!
//! Key differences from AC first scans:
//! - Coefficients are classified as ZERO, PREV_NONZERO, or NEWLY_NONZERO
//! - PREV_NONZERO coefficients emit refinement bits (stored separately)
//! - NEWLY_NONZERO coefficients emit a symbol with sign encoded in bit 1
//! - EOB runs can accumulate refinement bits from multiple blocks
//!
//! See docs/ac_refinement_scan_deep_dive.md for detailed algorithm analysis.

/// Coefficient classification in refinement scans
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoefClass {
    /// Coefficient is zero after shifting by Al
    Zero,
    /// Coefficient was nonzero in previous scan (absval >> Ah != 0)
    PrevNonzero,
    /// Coefficient becomes nonzero in this scan (absval >> Al == 1)
    NewlyNonzero,
}

/// Classifies a coefficient for refinement scan encoding.
///
/// # Arguments
/// * `coef` - The coefficient value
/// * `ah` - Previous approximation level (from previous scan)
/// * `al` - Current approximation level (this scan)
///
/// # Returns
/// The classification of this coefficient
pub fn classify_coefficient(coef: i16, ah: u8, al: u8) -> CoefClass {
    let abs_coef = coef.unsigned_abs();
    let was_nonzero = (abs_coef >> ah) != 0;
    let is_nonzero = (abs_coef >> al) != 0;

    if was_nonzero {
        CoefClass::PrevNonzero
    } else if is_nonzero {
        CoefClass::NewlyNonzero
    } else {
        CoefClass::Zero
    }
}

/// A refinement token as it would be stored in the token stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RefToken {
    /// Symbol byte:
    /// - Bits 7-4: Zero run (0-15)
    /// - Bit 1: Sign for newly-nonzero (0=neg, 1=pos) - C++ encoding
    /// - Bit 0: 1 = newly-nonzero, 0 = EOB/ZRL
    pub symbol: u8,
    /// Count of refinement bits that follow this token
    pub refbits: u8,
}

/// Result of tokenizing a single block for refinement
#[derive(Debug, Clone)]
pub struct BlockTokenization {
    /// Tokens emitted for this block
    pub tokens: Vec<RefToken>,
    /// Refinement bits (one per previously-nonzero coefficient skipped)
    pub refbits: Vec<u8>,
    /// Whether this block ends with EOB condition
    pub ends_with_eob: bool,
    /// Number of newly-nonzero coefficients found
    pub new_nonzeros: usize,
}

/// Tokenizes a single block for AC refinement scan (matches C++ algorithm).
///
/// This implements the exact algorithm from entropy_coding.cc:200-315.
///
/// Key insight from C++: ZRLs are ONLY emitted when we encounter a newly-nonzero
/// coefficient. For blocks that are all-zero or only have prev-nonzero coefficients,
/// no tokens are emitted - they just contribute to the EOB run.
pub fn tokenize_block_refinement(
    block: &[i16; 64],
    ss: u8,
    se: u8,
    ah: u8,
    al: u8,
) -> BlockTokenization {
    let mut tokens = Vec::new();
    let mut refbits = Vec::new();
    let mut r = 0u8; // Run of zeros (positions where coef was AND still is zero)
    let mut pending_refbits: Vec<u8> = Vec::new(); // Refbits accumulated during run
    let mut new_nonzeros = 0usize;

    for k in ss as usize..=se as usize {
        let coef = block[k];
        let abs_coef = coef.unsigned_abs();
        let shifted = abs_coef >> al;
        let was_nonzero = (abs_coef >> ah) != 0;

        if was_nonzero {
            // PREV_NONZERO: This coefficient was coded in a previous scan.
            // Emit refinement bit (the new LSB of the shifted value).
            // This does NOT count as a zero run, but the refbit accumulates.
            let refbit = (shifted & 1) as u8;
            pending_refbits.push(refbit);
            // Do NOT increment r - prev_nonzero doesn't count towards zero run
        } else if shifted == 0 {
            // ZERO: Still zero after shifting - counts towards zero run
            r += 1;
        } else {
            // NEWLY_NONZERO: Was zero before, now nonzero (shifted == 1)
            // This is the ONLY case that emits tokens!

            // First, emit ZRLs for any accumulated zeros > 15
            while r > 15 {
                tokens.push(RefToken {
                    symbol: 0xF0,
                    refbits: pending_refbits.len() as u8,
                });
                refbits.append(&mut pending_refbits);
                r -= 16;
            }

            // Now emit the newly-nonzero token
            let sign = if coef < 0 { 0u8 } else { 1u8 }; // 0=neg, 1=pos

            // C++ encodes sign in bit 1: symbol = (r << 4) + 1 + (sign << 1)
            let symbol = (r << 4) | 1 | (sign << 1);

            tokens.push(RefToken {
                symbol,
                refbits: pending_refbits.len() as u8,
            });
            // Emit pending refbits, then the sign bit
            refbits.append(&mut pending_refbits);
            refbits.push(sign);

            r = 0;
            new_nonzeros += 1;
        }
    }

    // End of block: if there are trailing zeros or pending refbits, this block
    // ends with an EOB condition. The refbits accumulate into the EOB run.
    let ends_with_eob = r > 0 || !pending_refbits.is_empty();
    if ends_with_eob {
        refbits.extend(pending_refbits);
    }

    BlockTokenization {
        tokens,
        refbits,
        ends_with_eob,
        new_nonzeros,
    }
}

/// Encodes the C++ symbol format for new nonzero coefficient.
///
/// C++ uses: symbol = (run << 4) + 1 + ((mask + 1) << 1)
/// where mask = coef >> 31 (all 1s for negative)
///
/// Result:
/// - Negative: symbol = (run << 4) | 1  (bit 1 = 0)
/// - Positive: symbol = (run << 4) | 3  (bit 1 = 1)
pub fn encode_new_nonzero_symbol(run: u8, is_negative: bool) -> u8 {
    if is_negative {
        (run << 4) | 1
    } else {
        (run << 4) | 3
    }
}

/// Extracts run count from symbol.
pub fn extract_run(symbol: u8) -> u8 {
    symbol >> 4
}

/// Extracts sign from symbol (for newly-nonzero tokens).
/// Returns true if positive, false if negative.
pub fn extract_sign(symbol: u8) -> bool {
    (symbol >> 1) & 1 == 1
}

/// For bitstream encoding, the symbol is masked to clear bit 1.
/// The sign is then written as an extra bit.
pub fn symbol_for_huffman(symbol: u8) -> u8 {
    symbol & 0b11111101 // Clear bit 1
}

// =============================================================================
// UNIT TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Coefficient Classification Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_classify_zero() {
        // Coefficient that's zero
        assert_eq!(classify_coefficient(0, 2, 1), CoefClass::Zero);

        // Coefficient that becomes zero after shift
        assert_eq!(classify_coefficient(1, 2, 1), CoefClass::Zero);
        assert_eq!(classify_coefficient(-1, 2, 1), CoefClass::Zero);
    }

    #[test]
    fn test_classify_prev_nonzero() {
        // Coefficient that was nonzero in previous scan
        // At Ah=2, Al=1: value >= 4 was nonzero (4 >> 2 = 1)
        assert_eq!(classify_coefficient(4, 2, 1), CoefClass::PrevNonzero);
        assert_eq!(classify_coefficient(8, 2, 1), CoefClass::PrevNonzero);
        assert_eq!(classify_coefficient(32, 2, 1), CoefClass::PrevNonzero);
        assert_eq!(classify_coefficient(-4, 2, 1), CoefClass::PrevNonzero);
        assert_eq!(classify_coefficient(-32, 2, 1), CoefClass::PrevNonzero);
    }

    #[test]
    fn test_classify_newly_nonzero() {
        // Coefficient that becomes nonzero in THIS scan
        // At Ah=2, Al=1: values 2,3 were zero (>>2=0) but now nonzero (>>1=1)
        assert_eq!(classify_coefficient(2, 2, 1), CoefClass::NewlyNonzero);
        assert_eq!(classify_coefficient(3, 2, 1), CoefClass::NewlyNonzero);
        assert_eq!(classify_coefficient(-2, 2, 1), CoefClass::NewlyNonzero);
        assert_eq!(classify_coefficient(-3, 2, 1), CoefClass::NewlyNonzero);
    }

    // -------------------------------------------------------------------------
    // Symbol Encoding Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_new_nonzero_symbol_positive() {
        // Positive coefficient with 0 zeros
        assert_eq!(encode_new_nonzero_symbol(0, false), 0x03);

        // Positive coefficient with 5 zeros
        assert_eq!(encode_new_nonzero_symbol(5, false), 0x53);

        // Positive coefficient with 14 zeros (max before ZRL)
        assert_eq!(encode_new_nonzero_symbol(14, false), 0xE3);
    }

    #[test]
    fn test_new_nonzero_symbol_negative() {
        // Negative coefficient with 0 zeros
        assert_eq!(encode_new_nonzero_symbol(0, true), 0x01);

        // Negative coefficient with 5 zeros
        assert_eq!(encode_new_nonzero_symbol(5, true), 0x51);

        // Negative coefficient with 14 zeros
        assert_eq!(encode_new_nonzero_symbol(14, true), 0xE1);
    }

    #[test]
    fn test_extract_run() {
        assert_eq!(extract_run(0x00), 0); // EOB
        assert_eq!(extract_run(0x01), 0); // New nonzero, 0 zeros
        assert_eq!(extract_run(0x03), 0); // New nonzero positive, 0 zeros
        assert_eq!(extract_run(0x51), 5); // 5 zeros
        assert_eq!(extract_run(0xF0), 15); // ZRL
    }

    #[test]
    fn test_extract_sign() {
        // Bit 1 = 0 means negative
        assert!(!extract_sign(0x01));
        assert!(!extract_sign(0x51));
        assert!(!extract_sign(0xE1));

        // Bit 1 = 1 means positive
        assert!(extract_sign(0x03));
        assert!(extract_sign(0x53));
        assert!(extract_sign(0xE3));
    }

    #[test]
    fn test_symbol_for_huffman() {
        // Symbol with bit 1 cleared for Huffman lookup
        assert_eq!(symbol_for_huffman(0x01), 0x01); // Already bit 1 = 0
        assert_eq!(symbol_for_huffman(0x03), 0x01); // Bit 1 cleared
        assert_eq!(symbol_for_huffman(0x51), 0x51);
        assert_eq!(symbol_for_huffman(0x53), 0x51);
        assert_eq!(symbol_for_huffman(0xF0), 0xF0); // ZRL unchanged
        assert_eq!(symbol_for_huffman(0x00), 0x00); // EOB unchanged
    }

    // -------------------------------------------------------------------------
    // Block Tokenization Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_empty_block() {
        let block = [0i16; 64];
        let result = tokenize_block_refinement(&block, 1, 63, 2, 1);

        // Empty block = EOB
        assert!(result.tokens.is_empty());
        assert!(result.refbits.is_empty());
        assert!(result.ends_with_eob);
        assert_eq!(result.new_nonzeros, 0);
    }

    #[test]
    fn test_single_prev_nonzero() {
        let mut block = [0i16; 64];
        block[1] = 32; // Was 8 after >>2, now 16 after >>1. Refbit = 16 & 1 = 0

        let result = tokenize_block_refinement(&block, 1, 63, 2, 1);

        // Should just emit refbit, then EOB
        assert!(result.tokens.is_empty()); // No tokens until EOB
        assert_eq!(result.refbits.len(), 1);
        assert_eq!(result.refbits[0], 0); // 16 & 1 = 0
        assert!(result.ends_with_eob);
    }

    #[test]
    fn test_single_newly_nonzero_positive() {
        let mut block = [0i16; 64];
        block[5] = 3; // Was 0 after >>2, becomes 1 after >>1

        let result = tokenize_block_refinement(&block, 1, 63, 2, 1);

        // Should emit: token for new nonzero at position 5 (run=4)
        assert_eq!(result.tokens.len(), 1);

        let token = &result.tokens[0];
        assert_eq!(extract_run(token.symbol), 4); // 4 zeros before
        assert!(extract_sign(token.symbol)); // Positive
        assert_eq!(token.refbits, 0); // No prev-nonzero coefficients skipped

        // Sign bit should be in refbits
        assert_eq!(result.refbits.len(), 1);
        assert_eq!(result.refbits[0], 1); // Positive sign

        assert_eq!(result.new_nonzeros, 1);
    }

    #[test]
    fn test_single_newly_nonzero_negative() {
        let mut block = [0i16; 64];
        block[5] = -3; // Was 0 after >>2, becomes -1 after >>1

        let result = tokenize_block_refinement(&block, 1, 63, 2, 1);

        assert_eq!(result.tokens.len(), 1);

        let token = &result.tokens[0];
        assert_eq!(extract_run(token.symbol), 4);
        assert!(!extract_sign(token.symbol)); // Negative

        assert_eq!(result.refbits.len(), 1);
        assert_eq!(result.refbits[0], 0); // Negative sign
    }

    #[test]
    fn test_prev_nonzero_then_newly_nonzero() {
        let mut block = [0i16; 64];
        block[1] = 32; // Prev nonzero (refbit needed)
        block[5] = 3; // Newly nonzero

        let result = tokenize_block_refinement(&block, 1, 63, 2, 1);

        // Token for newly nonzero should have 1 refbit (from block[1])
        assert_eq!(result.tokens.len(), 1);
        let token = &result.tokens[0];
        assert_eq!(extract_run(token.symbol), 3); // 3 zeros between them
        assert_eq!(token.refbits, 1); // 1 prev-nonzero coefficient skipped

        // Refbits: [refbit from block[1], sign from block[5]]
        assert_eq!(result.refbits.len(), 2);
    }

    #[test]
    fn test_zrl_emission() {
        let mut block = [0i16; 64];
        // Put newly nonzero at position 20 (run = 19 > 15)
        block[20] = 3;

        let result = tokenize_block_refinement(&block, 1, 63, 2, 1);

        // Should emit: ZRL (0xF0) then new nonzero
        assert_eq!(result.tokens.len(), 2);
        assert_eq!(result.tokens[0].symbol, 0xF0); // ZRL
        assert_eq!(result.tokens[0].refbits, 0); // No refbits accumulated

        let new_nz_token = &result.tokens[1];
        assert_eq!(extract_run(new_nz_token.symbol), 3); // Remaining 3 zeros
    }

    #[test]
    fn test_zrl_with_refbits() {
        let mut block = [0i16; 64];
        // 4 prev-nonzero coefficients, then 17 zeros, then newly nonzero
        block[1] = 32;
        block[2] = 32;
        block[3] = 32;
        block[4] = 32;
        // block[5..21] = 0 (17 zeros including the "run" to next)
        block[22] = 3; // Newly nonzero

        let result = tokenize_block_refinement(&block, 1, 63, 2, 1);

        // ZRL should have 4 refbits from the prev-nonzero coefficients
        assert_eq!(result.tokens.len(), 2);
        assert_eq!(result.tokens[0].symbol, 0xF0);
        assert_eq!(result.tokens[0].refbits, 4);

        // Second token is newly nonzero with run = 1 (pos 22 - 16 - 4 - 1 = 1)
        let new_nz_token = &result.tokens[1];
        assert_eq!(new_nz_token.refbits, 0); // No more prev-nonzero after ZRL
    }

    #[test]
    fn test_multiple_prev_nonzero_for_eob() {
        let mut block = [0i16; 64];
        // 5 prev-nonzero coefficients, no newly-nonzero
        for k in 1..6 {
            block[k] = 32 + k as i16;
        }

        let result = tokenize_block_refinement(&block, 1, 63, 2, 1);

        // No tokens (all prev-nonzero just emit refbits)
        assert!(result.tokens.is_empty());

        // 5 refbits (one per prev-nonzero)
        assert_eq!(result.refbits.len(), 5);

        // Should end with EOB
        assert!(result.ends_with_eob);
    }

    // -------------------------------------------------------------------------
    // EOB Run Tests (multi-block)
    // -------------------------------------------------------------------------

    /// Simulates EOB run accumulation across multiple blocks.
    ///
    /// This matches the C++ algorithm from entropy_coding.cc:280-305.
    /// Key behavior: refbits > 255 triggers a split AFTER adding the current block's bits.
    #[derive(Debug)]
    struct EobRunSimulator {
        eob_run: u16,
        accumulated_refbits: Vec<u8>,
        emitted_tokens: Vec<(u8, u16, Vec<u8>)>, // (symbol, eob_run, refbits)
    }

    impl EobRunSimulator {
        fn new() -> Self {
            Self {
                eob_run: 0,
                accumulated_refbits: Vec::new(),
                emitted_tokens: Vec::new(),
            }
        }

        fn accumulate_block(&mut self, block_result: &BlockTokenization) {
            if block_result.ends_with_eob && block_result.tokens.is_empty() {
                // Pure EOB block (no newly-nonzero tokens)
                self.eob_run += 1;

                // C++ adds refbits first, THEN checks limit
                let this_block_refbits = block_result.refbits.clone();
                self.accumulated_refbits.extend(&this_block_refbits);

                // Check limit AFTER adding (C++ uses > 255)
                if self.accumulated_refbits.len() > 255 {
                    // Split: current block starts new EOB run
                    // Remove this block's refbits from accumulated
                    let len = self.accumulated_refbits.len();
                    let this_len = this_block_refbits.len();
                    self.accumulated_refbits.truncate(len - this_len);

                    // Flush the previous run
                    self.eob_run -= 1;
                    if self.eob_run > 0 {
                        self.flush_eob_run();
                    }

                    // Start new run with this block
                    self.eob_run = 1;
                    self.accumulated_refbits = this_block_refbits;
                }

                // Also check max run
                if self.eob_run >= 0x7FFF {
                    self.flush_eob_run();
                }
            } else {
                // Block has tokens - flush EOB run first
                if self.eob_run > 0 {
                    self.flush_eob_run();
                }
                // Then emit block tokens (not tracked in this simulator)
            }
        }

        fn flush_eob_run(&mut self) {
            if self.eob_run == 0 {
                return;
            }

            let symbol = if self.eob_run == 1 {
                0x00
            } else {
                let log2 = 15 - self.eob_run.leading_zeros() as u8;
                log2 << 4
            };

            self.emitted_tokens.push((
                symbol,
                self.eob_run,
                std::mem::take(&mut self.accumulated_refbits),
            ));
            self.eob_run = 0;
        }

        fn finish(&mut self) {
            self.flush_eob_run();
        }
    }

    #[test]
    fn test_eob_run_single_block() {
        let mut sim = EobRunSimulator::new();

        let block = [0i16; 64];
        let result = tokenize_block_refinement(&block, 1, 63, 2, 1);
        sim.accumulate_block(&result);
        sim.finish();

        assert_eq!(sim.emitted_tokens.len(), 1);
        assert_eq!(sim.emitted_tokens[0].0, 0x00); // Single EOB
        assert_eq!(sim.emitted_tokens[0].1, 1);
    }

    #[test]
    fn test_eob_run_multiple_empty() {
        let mut sim = EobRunSimulator::new();

        // 5 empty blocks
        for _ in 0..5 {
            let block = [0i16; 64];
            let result = tokenize_block_refinement(&block, 1, 63, 2, 1);
            sim.accumulate_block(&result);
        }
        sim.finish();

        assert_eq!(sim.emitted_tokens.len(), 1);
        // eob_run = 5, log2(5) = 2, symbol = 0x20
        assert_eq!(sim.emitted_tokens[0].0, 0x20);
        assert_eq!(sim.emitted_tokens[0].1, 5);
    }

    #[test]
    fn test_eob_run_with_refbits() {
        let mut sim = EobRunSimulator::new();

        // 3 blocks, each with 2 prev-nonzero coefficients
        for _ in 0..3 {
            let mut block = [0i16; 64];
            block[1] = 32;
            block[2] = 32;
            let result = tokenize_block_refinement(&block, 1, 63, 2, 1);
            sim.accumulate_block(&result);
        }
        sim.finish();

        assert_eq!(sim.emitted_tokens.len(), 1);
        assert_eq!(sim.emitted_tokens[0].1, 3); // 3 blocks
        assert_eq!(sim.emitted_tokens[0].2.len(), 6); // 6 refbits total (2 per block)
    }

    #[test]
    fn test_eob_run_refbits_limit_split() {
        let mut sim = EobRunSimulator::new();

        // Each block has 60 prev-nonzero coefficients
        // After 5 blocks: 300 refbits > 255, should split
        for _ in 0..5 {
            let mut block = [0i16; 64];
            for k in 1..61 {
                block[k] = 32;
            }
            let result = tokenize_block_refinement(&block, 1, 63, 2, 1);
            sim.accumulate_block(&result);
        }
        sim.finish();

        // Should have split into multiple tokens
        assert!(sim.emitted_tokens.len() >= 2);

        // First token should have close to 255 refbits
        let first_refbits = sim.emitted_tokens[0].2.len();
        assert!(first_refbits <= 255);
    }

    // -------------------------------------------------------------------------
    // Integration Test: Compare with Expected C++ Output
    // -------------------------------------------------------------------------

    /// Test case from C++ instrumentation (placeholder - need real data)
    #[test]
    #[ignore] // Enable when C++ testdata is available
    fn test_matches_cpp_output() {
        // Load C++ testdata
        // let testdata = std::fs::read_to_string("testdata/ACRefinement.testdata").unwrap();

        // Parse each test case
        // For each case, tokenize block and compare tokens/refbits

        // This is the key parity test that will identify the bloat cause
    }

    // -------------------------------------------------------------------------
    // Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_all_zeros_in_range() {
        let mut block = [0i16; 64];
        block[0] = 100; // DC (outside range)

        // Only Ss=1 to Se=5 (all zeros in this range)
        let result = tokenize_block_refinement(&block, 1, 5, 2, 1);

        assert!(result.tokens.is_empty());
        assert!(result.refbits.is_empty());
        assert!(result.ends_with_eob);
    }

    #[test]
    fn test_single_coef_in_range() {
        let mut block = [0i16; 64];
        block[1] = 3; // Newly nonzero at Ss

        let result = tokenize_block_refinement(&block, 1, 1, 2, 1);

        // Single newly-nonzero at the only position
        assert_eq!(result.tokens.len(), 1);
        assert_eq!(extract_run(result.tokens[0].symbol), 0);
        assert!(!result.ends_with_eob); // No trailing zeros
    }

    #[test]
    fn test_max_zeros_before_new_nonzero() {
        let mut block = [0i16; 64];
        block[63] = 3; // Newly nonzero at very end (Ss=1, Se=63)

        let result = tokenize_block_refinement(&block, 1, 63, 2, 1);

        // Should emit: ZRL, ZRL, ZRL (48 zeros), then new nonzero with run=14
        let zrl_count = result.tokens.iter().filter(|t| t.symbol == 0xF0).count();
        assert_eq!(zrl_count, 3); // 48/16 = 3 ZRLs

        let last_token = result.tokens.last().unwrap();
        assert_eq!(extract_run(last_token.symbol), 14); // 62 - 48 = 14
    }

    #[test]
    fn test_refinement_bit_values() {
        let mut block = [0i16; 64];

        // Set up coefficients with known refbit values
        block[1] = 32; // 32 >> 1 = 16, 16 & 1 = 0
        block[2] = 34; // 34 >> 1 = 17, 17 & 1 = 1
        block[3] = 36; // 36 >> 1 = 18, 18 & 1 = 0

        let result = tokenize_block_refinement(&block, 1, 63, 2, 1);

        // All prev-nonzero, so refbits should be [0, 1, 0]
        assert_eq!(result.refbits, vec![0, 1, 0]);
    }
}
