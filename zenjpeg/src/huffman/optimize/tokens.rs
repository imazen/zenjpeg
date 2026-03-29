//! Token types for two-pass Huffman encoding.
//!
//! This module provides token types for capturing symbols and extra bits
//! during a first pass, then replaying them with optimized Huffman tables.

#![allow(dead_code)]
#![allow(clippy::wrong_self_convention)]

use super::frequency::FrequencyCounter;
use crate::error::Result;
use crate::huffman::HuffmanEncodeTable;

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
        let extra = crate::entropy::additional_bits_with_cat(diff, category);
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
            let extra = crate::entropy::additional_bits_with_cat(value, category);
            let symbol = (run << 4) | category;
            Self::new(context, symbol, extra, category)
        }
    }

    /// Serializes to JSON format for C++ comparison.
    #[cfg(feature = "__debug-tokens")]
    pub fn to_debug_json(&self) -> String {
        format!(
            r#"{{"context":{},"symbol":{},"extra_bits":{},"num_extra":{}}}"#,
            self.context, self.symbol, self.extra_bits, self.num_extra
        )
    }
}

/// Token for AC refinement scans in progressive JPEG.
///
/// Refinement scans have special encoding where:
/// - `symbol` encodes the Huffman symbol (EOBn, ZRL, or new nonzero coefficient)
/// - `refbits` counts how many refinement bits follow this token
///
/// This is more compact than `Token` (2 bytes vs 5 bytes) because refinement
/// scans don't need extra_bits - they only emit 1-bit corrections.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RefToken {
    /// Huffman symbol (EOB run indicator or coefficient symbol)
    pub symbol: u8,
    /// Number of refinement bits that follow this token
    pub refbits: u8,
}

impl RefToken {
    /// Creates a new refinement token.
    #[inline]
    pub const fn new(symbol: u8, refbits: u8) -> Self {
        Self { symbol, refbits }
    }

    /// Creates an EOB token with the given run length.
    ///
    /// EOB runs are encoded as:
    /// - Run 1: symbol = 0
    /// - Run 2-3: symbol = 16 + (run - 2)
    /// - Run 4-7: symbol = 32 + (run - 4)
    /// - etc.
    #[inline]
    pub fn eob(run: u16, refbits: u8) -> Self {
        let symbol = if run == 0 {
            0
        } else {
            // EOB run encoding: symbol = (log2(run) << 4) | (run - 2^log2(run))
            let log2 = 15 - run.leading_zeros() as u8;
            (log2 << 4) | ((run - (1 << log2)) as u8 & 0x0F)
        };
        Self::new(symbol, refbits)
    }

    /// Serializes to JSON format for C++ comparison.
    #[cfg(feature = "__debug-tokens")]
    pub fn to_debug_json(&self) -> String {
        format!(r#"{{"symbol":{},"refbits":{}}}"#, self.symbol, self.refbits)
    }
}

/// Metadata for a single progressive scan.
///
/// Each scan in a progressive JPEG has different token storage needs:
/// - DC scans and AC first scans use the main `Token` array
/// - AC refinement scans use separate `RefToken` arrays plus refinement bits
#[derive(Clone, Debug, Default)]
pub struct ScanTokenInfo {
    /// Offset into the main token array (for DC and AC first scans)
    pub token_offset: usize,
    /// Number of tokens for this scan
    pub num_tokens: usize,
    /// Tokens for AC refinement scans (empty for other scan types)
    pub ref_tokens: Vec<RefToken>,
    /// Refinement bits for AC refinement scans (1 bit per byte for simplicity)
    pub refbits: Vec<u8>,
    /// EOB run lengths for refinement scans
    pub eobruns: Vec<u16>,
    /// Restart marker positions (byte offsets into token stream)
    pub restarts: Vec<usize>,
    /// Context ID for this scan (used for histogram lookup)
    pub context: u8,
    /// Spectral selection start (0 for DC, 1-63 for AC)
    pub ss: u8,
    /// Spectral selection end
    pub se: u8,
    /// Successive approximation high bit (0 for first pass)
    pub ah: u8,
    /// Successive approximation low bit
    pub al: u8,
}

impl ScanTokenInfo {
    /// Creates info for a new scan.
    pub fn new(context: u8, ss: u8, se: u8, ah: u8, al: u8) -> Self {
        Self {
            token_offset: 0,
            num_tokens: 0,
            ref_tokens: Vec::new(),
            refbits: Vec::new(),
            eobruns: Vec::new(),
            restarts: Vec::new(),
            context,
            ss,
            se,
            ah,
            al,
        }
    }

    /// Creates info for an AC refinement scan with pre-allocated capacity.
    ///
    /// Estimates based on typical refinement scan characteristics:
    /// - `ref_tokens`: ~1-2 tokens per block on average
    /// - `refbits`: varies widely, estimate ~4 per block
    /// - `eobruns`: ~10% of blocks end with EOB runs
    ///
    /// Uses fallible allocation via `try_reserve`.
    pub fn with_capacity_refinement(
        context: u8,
        ss: u8,
        se: u8,
        ah: u8,
        al: u8,
        num_blocks: usize,
    ) -> Result<Self> {
        use crate::error::Error;

        let mut info = Self::new(context, ss, se, ah, al);

        // Estimate capacities - these are rough estimates to avoid most reallocations
        // Tokens: typically 1-2 per block, use 2 to be safe
        let token_capacity = num_blocks.saturating_mul(2);
        info.ref_tokens
            .try_reserve(token_capacity)
            .map_err(|_| Error::allocation_failed(token_capacity * 2, "refinement tokens"))?;

        // Refbits: varies by content, ~4 per block is reasonable
        let refbits_capacity = num_blocks.saturating_mul(4);
        info.refbits
            .try_reserve(refbits_capacity)
            .map_err(|_| Error::allocation_failed(refbits_capacity, "refinement bits"))?;

        // EOB runs: typically much fewer, ~10% of blocks
        let eobrun_capacity = num_blocks / 10 + 16;
        info.eobruns
            .try_reserve(eobrun_capacity)
            .map_err(|_| Error::allocation_failed(eobrun_capacity * 2, "EOB runs"))?;

        Ok(info)
    }

    /// Returns true if this is an AC refinement scan.
    #[inline]
    pub fn is_refinement(&self) -> bool {
        self.ss > 0 && self.ah > 0
    }

    /// Returns true if this is a DC scan.
    #[inline]
    pub fn is_dc(&self) -> bool {
        self.ss == 0 && self.se == 0
    }

    /// Debug dump of scan statistics
    #[allow(dead_code)]
    pub fn debug_dump(&self, scan_index: usize) {
        if self.is_refinement() {
            eprintln!(
                "=== Rust AC Refinement Scan {} ===\nSs={} Se={} Ah={} Al={}\nnum_blocks=? num_tokens={} num_refbits={} num_eobruns={}",
                scan_index,
                self.ss,
                self.se,
                self.ah,
                self.al,
                self.ref_tokens.len(),
                self.refbits.len(),
                self.eobruns.len()
            );
            // Print first 20 tokens
            eprintln!("TOKENS:");
            for (i, t) in self.ref_tokens.iter().take(20).enumerate() {
                eprintln!("  [{}] symbol=0x{:02x} refbits={}", i, t.symbol, t.refbits);
            }
            if self.ref_tokens.len() > 20 {
                eprintln!("  ... ({} more tokens)", self.ref_tokens.len() - 20);
            }
            eprintln!("=== End Rust AC Refinement Scan {} ===\n", scan_index);
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
        self.counters.iter().map(|c| c.generate_table()).collect()
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
    fn test_ref_token_new() {
        let token = RefToken::new(0x12, 5);
        assert_eq!(token.symbol, 0x12);
        assert_eq!(token.refbits, 5);
    }

    #[test]
    fn test_ref_token_eob() {
        // Run 0 -> symbol 0 (simple EOB)
        let eob0 = RefToken::eob(0, 0);
        assert_eq!(eob0.symbol, 0);

        // Run 1 -> symbol should encode as log2(1)=0, with offset
        let eob1 = RefToken::eob(1, 0);
        assert_eq!(eob1.symbol, 0); // log2(1) = 0, 1 - 1 = 0 -> 0x00

        // Run 2 -> log2(2) = 1, 2 - 2 = 0 -> symbol = (1 << 4) | 0 = 0x10
        let eob2 = RefToken::eob(2, 0);
        assert_eq!(eob2.symbol, 0x10);

        // Run 3 -> log2(3) = 1, 3 - 2 = 1 -> symbol = (1 << 4) | 1 = 0x11
        let eob3 = RefToken::eob(3, 0);
        assert_eq!(eob3.symbol, 0x11);

        // Run 4 -> log2(4) = 2, 4 - 4 = 0 -> symbol = (2 << 4) | 0 = 0x20
        let eob4 = RefToken::eob(4, 0);
        assert_eq!(eob4.symbol, 0x20);
    }

    #[test]
    fn test_scan_token_info() {
        let info = ScanTokenInfo::new(4, 1, 63, 0, 2);
        assert_eq!(info.context, 4);
        assert_eq!(info.ss, 1);
        assert_eq!(info.se, 63);
        assert_eq!(info.ah, 0);
        assert_eq!(info.al, 2);
        assert!(!info.is_refinement()); // ah = 0
        assert!(!info.is_dc()); // ss = 1
    }

    #[test]
    fn test_scan_token_info_dc() {
        let info = ScanTokenInfo::new(0, 0, 0, 0, 1);
        assert!(info.is_dc());
        assert!(!info.is_refinement());
    }

    #[test]
    fn test_scan_token_info_refinement() {
        let info = ScanTokenInfo::new(4, 1, 63, 2, 1);
        assert!(info.is_refinement()); // ss > 0 && ah > 0
        assert!(!info.is_dc());
    }
}
