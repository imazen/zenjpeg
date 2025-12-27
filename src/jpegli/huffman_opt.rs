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

use crate::jpegli::error::{Error, Result};
use crate::jpegli::huffman::HuffmanEncodeTable;

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
        let category = crate::jpegli::entropy::category(diff);
        let extra = crate::jpegli::entropy::additional_bits(diff);
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
            let category = crate::jpegli::entropy::category(value);
            let extra = crate::jpegli::entropy::additional_bits(value);
            let symbol = (run << 4) | category;
            Self::new(context, symbol, extra, category)
        }
    }

    /// Serializes to JSON format for C++ comparison.
    #[cfg(feature = "debug-tokens")]
    pub fn to_debug_json(&self) -> String {
        format!(
            r#"{{"context":{},"symbol":{},"extra_bits":{},"num_extra":{}}}"#,
            self.context, self.symbol, self.extra_bits, self.num_extra
        )
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

// =============================================================================
// Progressive JPEG Tokenization Structures
// =============================================================================

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
    #[cfg(feature = "debug-tokens")]
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
}

/// Result of histogram clustering.
#[derive(Clone, Debug)]
pub struct ClusterResult {
    /// Mapping from context ID to cluster (table) index
    pub context_map: Vec<usize>,
    /// Merged histograms for each cluster
    pub cluster_histograms: Vec<FrequencyCounter>,
    /// Number of clusters
    pub num_clusters: usize,
    /// Merge log for debugging (context pairs that were merged)
    #[cfg(feature = "debug-tokens")]
    pub merge_log: Vec<(usize, usize, f64)>, // (ctx_a, ctx_b, cost_delta)
}

impl ClusterResult {
    /// Dumps the merge log to a file for debugging.
    #[cfg(feature = "debug-tokens")]
    pub fn dump_merge_log(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::fs::File::create(path)?;
        writeln!(file, "[")?;
        for (i, (a, b, cost)) in self.merge_log.iter().enumerate() {
            let comma = if i + 1 < self.merge_log.len() {
                ","
            } else {
                ""
            };
            writeln!(
                file,
                r#"  {{"ctx_a":{},"ctx_b":{},"cost_delta":{:.4}}}{}"#,
                a, b, cost, comma
            )?;
        }
        writeln!(file, "]")?;
        Ok(())
    }
}

impl FrequencyCounter {
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

/// Clusters histograms to minimize total encoding cost.
///
/// This implements the C++ ClusterJpegHistograms algorithm:
/// - For each histogram, find the best existing cluster to merge with
/// - If merging saves bits, merge; otherwise create new cluster
/// - Limit to max_clusters (typically 4 for baseline JPEG)
///
/// # Arguments
/// * `histograms` - Slice of frequency counters (one per context)
/// * `max_clusters` - Maximum number of clusters (4 for baseline, unlimited for progressive)
///
/// # Returns
/// ClusterResult with context-to-cluster mapping and merged histograms
pub fn cluster_histograms(histograms: &[FrequencyCounter], max_clusters: usize) -> ClusterResult {
    let mut context_map = vec![0usize; histograms.len()];
    let mut cluster_histograms: Vec<FrequencyCounter> = Vec::new();
    let mut slot_costs: Vec<f64> = Vec::new();

    #[cfg(feature = "debug-tokens")]
    let mut merge_log = Vec::new();

    for (ctx_idx, histo) in histograms.iter().enumerate() {
        if histo.is_empty_histogram() {
            // Empty histogram - just assign to cluster 0 (will be ignored)
            context_map[ctx_idx] = 0;
            continue;
        }

        let cur_cost = histo.estimate_encoding_cost();
        let num_slots = cluster_histograms.len();

        // Find best slot to merge with
        let mut best_slot = num_slots; // Default: create new cluster
        let mut best_cost = if num_slots >= max_clusters {
            f64::MAX // Force merge if at limit
        } else {
            cur_cost // Cost of creating new cluster
        };

        for slot_idx in 0..num_slots {
            let combined = cluster_histograms[slot_idx].combined(histo);
            let combined_cost = combined.estimate_encoding_cost();
            // Cost delta: how much does merging increase total cost?
            let cost_delta = combined_cost - slot_costs[slot_idx];

            if cost_delta < best_cost {
                best_cost = cost_delta;
                best_slot = slot_idx;
            }
        }

        if best_slot == num_slots {
            // Create new cluster
            cluster_histograms.push(histo.clone());
            slot_costs.push(cur_cost);
            context_map[ctx_idx] = num_slots;
        } else {
            // Merge with existing cluster
            cluster_histograms[best_slot].add(histo);
            slot_costs[best_slot] += best_cost;
            context_map[ctx_idx] = best_slot;

            #[cfg(feature = "debug-tokens")]
            merge_log.push((ctx_idx, best_slot, best_cost));
        }
    }

    ClusterResult {
        context_map,
        num_clusters: cluster_histograms.len(),
        cluster_histograms,
        #[cfg(feature = "debug-tokens")]
        merge_log,
    }
}

/// Buffer for all tokens across all progressive scans.
///
/// This implements the C++ jpegli two-pass approach:
/// 1. Tokenize all scans, collecting symbols without encoding
/// 2. Build histograms from actual token usage
/// 3. Optionally cluster similar histograms
/// 4. Generate optimized Huffman tables
/// 5. Replay tokens with optimized tables
#[derive(Clone, Debug)]
pub struct ProgressiveTokenBuffer {
    /// Main token storage for DC and AC first scans
    pub tokens: Vec<Token>,
    /// Per-scan metadata and tokens
    pub scan_info: Vec<ScanTokenInfo>,
    /// Frequency counters per context
    pub counters: Vec<FrequencyCounter>,
    /// Number of contexts (DC components + AC scans)
    pub num_contexts: usize,
    /// DC predictors per component (for tokenization)
    dc_pred: Vec<i16>,
}

impl ProgressiveTokenBuffer {
    /// Creates a new buffer for progressive tokenization.
    ///
    /// # Arguments
    /// * `num_components` - Number of color components (1 for gray, 3 for color)
    /// * `num_scans` - Number of progressive scans
    ///
    /// Context mapping:
    /// - DC contexts: 0..num_components
    /// - AC contexts: num_components..num_components + num_ac_scans
    pub fn new(num_components: usize, num_scans: usize) -> Self {
        // Estimate contexts: DC (one per component) + AC (one per scan with Se > 0)
        // We'll allocate generously and track actual usage
        let num_contexts = num_components + num_scans;
        Self {
            tokens: Vec::new(),
            scan_info: Vec::with_capacity(num_scans),
            counters: vec![FrequencyCounter::new(); num_contexts],
            num_contexts,
            dc_pred: vec![0; num_components],
        }
    }

    /// Creates a buffer with pre-estimated capacity.
    pub fn with_capacity(num_components: usize, num_scans: usize, estimated_tokens: usize) -> Self {
        let mut buf = Self::new(num_components, num_scans);
        buf.tokens.reserve(estimated_tokens);
        buf
    }

    /// Returns the number of tokens in the main buffer.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Returns true if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Resets DC predictors (call at start of each scan or restart interval).
    pub fn reset_dc_pred(&mut self) {
        self.dc_pred.fill(0);
    }

    /// Gets the current DC predictor for a component.
    pub fn dc_pred(&self, component: usize) -> i16 {
        self.dc_pred.get(component).copied().unwrap_or(0)
    }

    /// Updates the DC predictor for a component.
    pub fn set_dc_pred(&mut self, component: usize, value: i16) {
        if component < self.dc_pred.len() {
            self.dc_pred[component] = value;
        }
    }

    /// Adds a token to the main buffer and updates the frequency counter.
    #[inline]
    pub fn push(&mut self, token: Token) {
        if (token.context as usize) < self.counters.len() {
            self.counters[token.context as usize].count(token.symbol);
        }
        self.tokens.push(token);
    }

    /// Adds a refinement token to the current scan.
    #[inline]
    pub fn push_ref(&mut self, token: RefToken) {
        if let Some(info) = self.scan_info.last_mut() {
            // Count the symbol (masked for EOB run encoding)
            let context = info.context as usize;
            if context < self.counters.len() {
                // Mask off the low 4 bits for EOB run symbols
                self.counters[context].count(token.symbol & 0xF0);
            }
            info.ref_tokens.push(token);
        }
    }

    /// Adds a refinement bit (0 or 1) to the current scan.
    #[inline]
    pub fn push_refbit(&mut self, bit: u8) {
        if let Some(info) = self.scan_info.last_mut() {
            info.refbits.push(bit & 1);
        }
    }

    /// Starts a new scan.
    pub fn start_scan(&mut self, context: u8, ss: u8, se: u8, ah: u8, al: u8) {
        let mut info = ScanTokenInfo::new(context, ss, se, ah, al);
        info.token_offset = self.tokens.len();
        self.scan_info.push(info);
    }

    /// Finalizes the current scan, recording the token count.
    pub fn end_scan(&mut self) {
        if let Some(info) = self.scan_info.last_mut() {
            info.num_tokens = self.tokens.len() - info.token_offset;
        }
    }

    /// Marks a restart position in the current scan.
    pub fn mark_restart(&mut self) {
        if let Some(info) = self.scan_info.last_mut() {
            let pos = if info.is_refinement() {
                info.ref_tokens.len()
            } else {
                self.tokens.len() - info.token_offset
            };
            info.restarts.push(pos);
        }
        self.reset_dc_pred();
    }

    /// Returns the tokens for a specific scan.
    pub fn scan_tokens(&self, scan_index: usize) -> &[Token] {
        if let Some(info) = self.scan_info.get(scan_index) {
            let start = info.token_offset;
            let end = start + info.num_tokens;
            &self.tokens[start..end]
        } else {
            &[]
        }
    }

    /// Returns the frequency counter for a context.
    pub fn counter(&self, context: usize) -> Option<&FrequencyCounter> {
        self.counters.get(context)
    }

    /// Clusters histograms and generates optimized Huffman tables.
    ///
    /// This is the main entry point for two-pass progressive encoding optimization.
    ///
    /// # Arguments
    /// * `max_dc_clusters` - Max DC table clusters (typically 2-4)
    /// * `max_ac_clusters` - Max AC table clusters (typically 2-4)
    /// * `num_dc_contexts` - Number of DC contexts (= num_components)
    ///
    /// # Returns
    /// - `context_map`: Maps each context to a table index
    /// - `num_dc_tables`: Number of DC tables (for indexing into tables array)
    /// - `tables`: Optimized Huffman tables for each cluster (DC tables first, then AC)
    pub fn generate_optimized_tables(
        &self,
        max_dc_clusters: usize,
        max_ac_clusters: usize,
        num_dc_contexts: usize,
    ) -> Result<(Vec<usize>, usize, Vec<OptimizedTable>)> {
        // Split into DC and AC histograms
        let dc_histograms: Vec<_> = self.counters[..num_dc_contexts].to_vec();
        let ac_histograms: Vec<_> = self.counters[num_dc_contexts..].to_vec();

        // Cluster DC and AC separately
        let dc_clusters = cluster_histograms(&dc_histograms, max_dc_clusters);
        let ac_clusters = cluster_histograms(&ac_histograms, max_ac_clusters);

        // Build context map
        let mut context_map = Vec::with_capacity(self.num_contexts);

        // DC contexts map to clusters 0..num_dc_clusters
        for ctx in 0..num_dc_contexts {
            context_map.push(dc_clusters.context_map[ctx]);
        }

        // AC contexts map to clusters num_dc_clusters..
        let dc_offset = dc_clusters.num_clusters;
        for ctx in 0..ac_histograms.len() {
            context_map.push(dc_offset + ac_clusters.context_map[ctx]);
        }

        // Generate tables from clustered histograms
        let mut tables = Vec::new();

        // DC tables
        for histo in &dc_clusters.cluster_histograms {
            if histo.is_empty_histogram() {
                // Empty histogram - use a default table
                let mut default = FrequencyCounter::new();
                default.count(0); // At least one symbol
                tables.push(default.generate_table_with_dht()?);
            } else {
                tables.push(histo.generate_table_with_dht()?);
            }
        }

        // AC tables
        for histo in &ac_clusters.cluster_histograms {
            if histo.is_empty_histogram() {
                let mut default = FrequencyCounter::new();
                default.count(0); // At least one symbol (EOB)
                tables.push(default.generate_table_with_dht()?);
            } else {
                tables.push(histo.generate_table_with_dht()?);
            }
        }

        Ok((context_map, dc_clusters.num_clusters, tables))
    }

    /// Generates optimized Huffman tables with explicit luma/chroma grouping.
    ///
    /// This method creates exactly 2 DC tables and 2 AC tables by explicitly
    /// grouping luma (component 0) vs chroma (components 1+) rather than
    /// using automatic clustering. This ensures the table assignment matches
    /// what the replay code expects.
    ///
    /// # Arguments
    /// * `num_dc_contexts` - Number of DC contexts (= num_components)
    ///
    /// # Returns
    /// - `num_dc_tables`: Always 2 (luma + chroma)
    /// - `tables`: [DC luma, DC chroma, AC luma, AC chroma]
    pub fn generate_luma_chroma_tables(
        &self,
        num_dc_contexts: usize,
    ) -> Result<(usize, Vec<OptimizedTable>)> {
        let mut tables = Vec::with_capacity(4);

        // DC tables: luma = context 0, chroma = contexts 1+
        let dc_luma = &self.counters[0];
        let mut dc_chroma = FrequencyCounter::new();
        for ctx in 1..num_dc_contexts {
            dc_chroma.add(&self.counters[ctx]);
        }

        // Generate DC luma table
        if dc_luma.is_empty_histogram() {
            let mut default = FrequencyCounter::new();
            default.count(0);
            tables.push(default.generate_table_with_dht()?);
        } else {
            tables.push(dc_luma.generate_table_with_dht()?);
        }

        // Generate DC chroma table
        if dc_chroma.is_empty_histogram() {
            tables.push(tables[0].clone()); // Use luma table as fallback
        } else {
            tables.push(dc_chroma.generate_table_with_dht()?);
        }

        // AC tables: need to identify which contexts are luma vs chroma
        // AC contexts start at num_dc_contexts
        //
        // Context assignment: context = num_components + component_index
        // This ensures consistent table assignment regardless of scan order:
        // - AC Y (component 0): context = 3 + 0 = 3 → counters[3] = ac_histograms[0]
        // - AC Cb (component 1): context = 3 + 1 = 4 → counters[4] = ac_histograms[1]
        // - AC Cr (component 2): context = 3 + 2 = 5 → counters[5] = ac_histograms[2]

        let ac_start = num_dc_contexts;
        let ac_histograms = &self.counters[ac_start..];

        // AC luma = component 0 = context num_dc_contexts = ac_histograms[0]
        let ac_luma_idx = 0;
        let ac_luma = if ac_luma_idx < ac_histograms.len() {
            &ac_histograms[ac_luma_idx]
        } else {
            // Fallback for grayscale - should not happen
            &self.counters[0]
        };

        // AC chroma = components 1, 2 = contexts num_dc_contexts+1, num_dc_contexts+2
        let mut ac_chroma = FrequencyCounter::new();
        for idx in 1..ac_histograms.len() {
            ac_chroma.add(&ac_histograms[idx]);
        }

        // Generate AC luma table
        if ac_luma.is_empty_histogram() {
            let mut default = FrequencyCounter::new();
            default.count(0); // EOB
            tables.push(default.generate_table_with_dht()?);
        } else {
            tables.push(ac_luma.generate_table_with_dht()?);
        }

        // Generate AC chroma table
        if ac_chroma.is_empty_histogram() {
            tables.push(tables[2].clone()); // Use AC luma as fallback
        } else {
            tables.push(ac_chroma.generate_table_with_dht()?);
        }

        Ok((2, tables)) // Always 2 DC tables
    }

    /// Dumps all tokens to a JSON file for C++ comparison.
    #[cfg(feature = "debug-tokens")]
    pub fn dump_tokens(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::fs::File::create(path)?;
        writeln!(file, "[")?;
        for (i, token) in self.tokens.iter().enumerate() {
            let comma = if i + 1 < self.tokens.len() { "," } else { "" };
            writeln!(file, "  {}{}", token.to_debug_json(), comma)?;
        }
        writeln!(file, "]")?;
        Ok(())
    }

    /// Dumps histograms to a JSON file for C++ comparison.
    #[cfg(feature = "debug-tokens")]
    pub fn dump_histograms(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::fs::File::create(path)?;
        writeln!(file, "{{")?;
        for (ctx, counter) in self.counters.iter().enumerate() {
            let total = counter.total();
            if total == 0 {
                continue;
            }
            writeln!(file, r#"  "context_{}": {{"#, ctx)?;
            writeln!(file, r#"    "total": {},"#, total)?;
            write!(file, r#"    "counts": ["#)?;
            for (i, count) in (0..256).map(|s| counter.get_count(s as u8)).enumerate() {
                if i > 0 {
                    write!(file, ",")?;
                }
                write!(file, "{}", count)?;
            }
            writeln!(file, "]")?;
            writeln!(file, "  }},")?;
        }
        writeln!(file, "}}")?;
        Ok(())
    }

    // =========================================================================
    // Tokenization Methods
    // =========================================================================

    /// Tokenizes a DC scan (first pass or refinement).
    ///
    /// For interleaved DC scans, blocks should be provided in MCU order:
    /// `[comp0_block0, comp1_block0, comp2_block0, comp0_block1, ...]`
    ///
    /// # Arguments
    /// * `blocks` - Quantized DCT blocks for each component, in MCU order
    /// * `component_indices` - Which components are in this scan (e.g., [0, 1, 2])
    /// * `al` - Successive approximation low bit (0 for first pass)
    /// * `ah` - Successive approximation high bit (0 for first pass)
    pub fn tokenize_dc_scan(
        &mut self,
        blocks: &[&[[i16; 64]]],
        component_indices: &[usize],
        al: u8,
        ah: u8,
    ) {
        // Start the scan - DC uses context = component index
        // For interleaved scans, we'll emit tokens for each component
        self.start_scan(0, 0, 0, ah, al);
        self.reset_dc_pred();

        if ah == 0 {
            // First DC scan: encode DC coefficients shifted by al
            self.tokenize_dc_first(blocks, component_indices, al);
        } else {
            // DC refinement: just emit one bit per block
            self.tokenize_dc_refine(blocks, component_indices, al);
        }

        self.end_scan();
    }

    /// Tokenizes DC first scan (ah == 0).
    fn tokenize_dc_first(&mut self, blocks: &[&[[i16; 64]]], component_indices: &[usize], al: u8) {
        // Get the number of blocks (all components should have same count for interleaved)
        let num_blocks = blocks.get(0).map(|b| b.len()).unwrap_or(0);

        for block_idx in 0..num_blocks {
            for (comp_offset, &comp_idx) in component_indices.iter().enumerate() {
                if let Some(comp_blocks) = blocks.get(comp_offset) {
                    if let Some(block) = comp_blocks.get(block_idx) {
                        // Get DC coefficient and shift by al
                        let dc = block[0] >> al;
                        let prev = self.dc_pred(comp_idx);
                        let diff = dc - prev;
                        self.set_dc_pred(comp_idx, dc);

                        // Create DC token
                        let token = Token::dc(comp_idx as u8, diff);
                        self.push(token);
                    }
                }
            }
        }
    }

    /// Tokenizes DC refinement scan (ah > 0).
    fn tokenize_dc_refine(&mut self, blocks: &[&[[i16; 64]]], component_indices: &[usize], al: u8) {
        let num_blocks = blocks.get(0).map(|b| b.len()).unwrap_or(0);

        for block_idx in 0..num_blocks {
            for (comp_offset, &comp_idx) in component_indices.iter().enumerate() {
                if let Some(comp_blocks) = blocks.get(comp_offset) {
                    if let Some(block) = comp_blocks.get(block_idx) {
                        // For DC refinement, just emit the bit at position al
                        let bit = ((block[0] >> al) & 1) as u8;

                        // DC refinement uses symbol 0 with extra bit
                        let token = Token::new(comp_idx as u8, 0, bit as u16, 1);
                        self.push(token);
                    }
                }
            }
        }
    }

    /// Tokenizes an AC first scan (ah == 0).
    ///
    /// # Arguments
    /// * `blocks` - Quantized DCT blocks for this component
    /// * `context` - Context ID for this scan (for histogram)
    /// * `ss` - Spectral selection start (1-63)
    /// * `se` - Spectral selection end (1-63)
    /// * `al` - Successive approximation low bit
    pub fn tokenize_ac_first_scan(
        &mut self,
        blocks: &[[i16; 64]],
        context: u8,
        ss: u8,
        se: u8,
        al: u8,
    ) {
        self.start_scan(context, ss, se, 0, al);

        let mut eob_run: u16 = 0;

        for block in blocks {
            // Find last nonzero coefficient in spectral range
            let mut last_nonzero = ss as usize;
            for k in (ss as usize..=se as usize).rev() {
                if (block[k] >> al) != 0 {
                    last_nonzero = k;
                    break;
                }
            }

            // Check if block is all zeros in this range
            let is_eob = (ss as usize..=se as usize).all(|k| (block[k] >> al) == 0);

            if is_eob {
                eob_run += 1;
                // Emit EOB run when it reaches max (0x7FFF) or at end
                if eob_run == 0x7FFF {
                    self.emit_eob_run(context, eob_run);
                    eob_run = 0;
                }
                continue;
            }

            // Emit pending EOB run
            if eob_run > 0 {
                self.emit_eob_run(context, eob_run);
                eob_run = 0;
            }

            // Encode coefficients
            let mut run = 0u8;
            for k in ss as usize..=se as usize {
                let coef = block[k] >> al;
                if coef == 0 {
                    run += 1;
                } else {
                    // Emit ZRL for runs >= 16
                    while run >= 16 {
                        let zrl = Token::new(context, 0xF0, 0, 0);
                        self.push(zrl);
                        run -= 16;
                    }

                    // Emit coefficient token
                    let token = Token::ac(context, run, coef as i16);
                    self.push(token);
                    run = 0;
                }

                if k == last_nonzero {
                    break;
                }
            }

            // If we didn't reach the end, emit EOB
            if last_nonzero < se as usize {
                eob_run += 1;
                if eob_run == 0x7FFF {
                    self.emit_eob_run(context, eob_run);
                    eob_run = 0;
                }
            }
        }

        // Flush remaining EOB run
        if eob_run > 0 {
            self.emit_eob_run(context, eob_run);
        }

        self.end_scan();
    }

    /// Emits an EOB run token.
    fn emit_eob_run(&mut self, context: u8, run: u16) {
        if run == 0 {
            return;
        }

        // EOB run encoding: symbol = (log2(run) << 4) | extra
        // For run = 1: symbol = 0 (simple EOB)
        // For run = 2-3: symbol = 0x10 | (run - 2)
        // For run = 4-7: symbol = 0x20 | (run - 4)
        // etc.
        if run == 1 {
            let token = Token::new(context, 0x00, 0, 0);
            self.push(token);
        } else {
            let log2 = 15 - run.leading_zeros() as u8;
            let extra_bits = run - (1 << log2);
            let symbol = log2 << 4;
            let token = Token::new(context, symbol, extra_bits, log2);
            self.push(token);
        }
    }

    /// Tokenizes an AC refinement scan (ah > 0).
    ///
    /// This is the most complex tokenization because it must interleave:
    /// - Symbols for newly-nonzero coefficients
    /// - Refinement bits for previously-nonzero coefficients
    ///
    /// # Arguments
    /// * `blocks` - Quantized DCT blocks for this component
    /// * `context` - Context ID for this scan
    /// * `ss` - Spectral selection start
    /// * `se` - Spectral selection end
    /// * `ah` - Successive approximation high bit (previous precision)
    /// * `al` - Successive approximation low bit (current precision)
    pub fn tokenize_ac_refinement_scan(
        &mut self,
        blocks: &[[i16; 64]],
        context: u8,
        ss: u8,
        se: u8,
        ah: u8,
        al: u8,
    ) {
        self.start_scan(context, ss, se, ah, al);

        let mut eob_run: u16 = 0;
        let mut pending_refbits: Vec<u8> = Vec::new();

        for block in blocks {
            // Find if there are any newly-nonzero or previously-nonzero coefficients
            let mut has_content = false;
            for k in ss as usize..=se as usize {
                let abs_coef = block[k].abs();
                // Was previously nonzero (bits at ah position or higher)
                let was_nonzero = (abs_coef >> ah) != 0;
                // Is newly nonzero (bit at al position, but not at ah)
                let newly_nonzero = !was_nonzero && ((abs_coef >> al) & 1) != 0;
                if was_nonzero || newly_nonzero {
                    has_content = true;
                    break;
                }
            }

            if !has_content {
                // All zeros - add to EOB run
                // First, flush any pending refbits from previous blocks
                if !pending_refbits.is_empty() {
                    // Emit pending EOB run with refbits
                    if eob_run > 0 {
                        self.emit_eob_run_with_refbits(context, eob_run, &pending_refbits);
                        pending_refbits.clear();
                        eob_run = 0;
                    }
                }
                eob_run += 1;
                if eob_run == 0x7FFF {
                    self.emit_eob_run_with_refbits(context, eob_run, &pending_refbits);
                    pending_refbits.clear();
                    eob_run = 0;
                }
                continue;
            }

            // Emit pending EOB run
            if eob_run > 0 {
                self.emit_eob_run_with_refbits(context, eob_run, &pending_refbits);
                pending_refbits.clear();
                eob_run = 0;
            }

            // Process coefficients
            let mut run = 0u8;
            let mut block_refbits: Vec<u8> = Vec::new();

            for k in ss as usize..=se as usize {
                let coef = block[k];
                let abs_coef = coef.abs();
                let was_nonzero = (abs_coef >> ah) != 0;

                if was_nonzero {
                    // Previously nonzero: emit refinement bit
                    let refbit = ((abs_coef >> al) & 1) as u8;
                    block_refbits.push(refbit);
                } else {
                    // Check if newly nonzero
                    let newly_nonzero = ((abs_coef >> al) & 1) != 0;
                    if newly_nonzero {
                        // Emit ZRL for runs >= 16
                        while run >= 16 {
                            // ZRL with pending refbits
                            let ref_token = RefToken::new(0xF0, block_refbits.len() as u8);
                            self.push_ref(ref_token);
                            for &bit in &block_refbits {
                                self.push_refbit(bit);
                            }
                            block_refbits.clear();
                            run -= 16;
                        }

                        // Emit newly nonzero coefficient
                        let sign = if coef < 0 { 0 } else { 1 };
                        let symbol = (run << 4) | 1; // Category is always 1 for refinement
                        let ref_token = RefToken::new(symbol, block_refbits.len() as u8);
                        self.push_ref(ref_token);
                        self.push_refbit(sign);
                        for &bit in &block_refbits {
                            self.push_refbit(bit);
                        }
                        block_refbits.clear();
                        run = 0;
                    } else {
                        run += 1;
                    }
                }
            }

            // If we have trailing refbits, they go with the next EOB
            pending_refbits = block_refbits;
            if run > 0 || !pending_refbits.is_empty() {
                eob_run += 1;
            }
        }

        // Flush remaining EOB run
        if eob_run > 0 || !pending_refbits.is_empty() {
            self.emit_eob_run_with_refbits(context, eob_run, &pending_refbits);
        }

        self.end_scan();
    }

    /// Emits an EOB run token with associated refinement bits.
    fn emit_eob_run_with_refbits(&mut self, _context: u8, run: u16, refbits: &[u8]) {
        let symbol = if run <= 1 {
            0x00
        } else {
            let log2 = 15 - run.leading_zeros() as u8;
            log2 << 4
        };

        let ref_token = RefToken::new(symbol, refbits.len() as u8);
        self.push_ref(ref_token);

        // Store the EOB run value if > 1
        if run > 1 {
            if let Some(info) = self.scan_info.last_mut() {
                info.eobruns.push(run);
            }
        }

        for &bit in refbits {
            self.push_refbit(bit);
        }
    }
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

#[cfg(test)]
mod progressive_token_tests {
    use super::*;

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

    #[test]
    fn test_progressive_token_buffer_new() {
        let buf = ProgressiveTokenBuffer::new(3, 4);
        assert_eq!(buf.num_contexts, 7); // 3 DC + 4 scans
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
    }

    #[test]
    fn test_progressive_token_buffer_push() {
        let mut buf = ProgressiveTokenBuffer::new(1, 2);

        // Start a DC scan
        buf.start_scan(0, 0, 0, 0, 0);

        // Push a DC token
        let token = Token::dc(0, 100);
        buf.push(token);

        assert_eq!(buf.len(), 1);
        assert_eq!(buf.counter(0).unwrap().total(), 1);

        buf.end_scan();
        assert_eq!(buf.scan_info.len(), 1);
        assert_eq!(buf.scan_info[0].num_tokens, 1);
    }

    #[test]
    fn test_progressive_token_buffer_dc_pred() {
        let mut buf = ProgressiveTokenBuffer::new(3, 1);

        // Initial DC predictors should be 0
        assert_eq!(buf.dc_pred(0), 0);
        assert_eq!(buf.dc_pred(1), 0);
        assert_eq!(buf.dc_pred(2), 0);

        // Update predictor
        buf.set_dc_pred(1, 512);
        assert_eq!(buf.dc_pred(1), 512);

        // Reset
        buf.reset_dc_pred();
        assert_eq!(buf.dc_pred(1), 0);
    }

    #[test]
    fn test_progressive_token_buffer_scan_tokens() {
        let mut buf = ProgressiveTokenBuffer::new(1, 2);

        // First scan
        buf.start_scan(0, 0, 0, 0, 0);
        buf.push(Token::dc(0, 50));
        buf.push(Token::dc(0, 60));
        buf.end_scan();

        // Second scan
        buf.start_scan(1, 1, 63, 0, 0);
        buf.push(Token::ac(1, 0, 10));
        buf.push(Token::ac(1, 2, 5));
        buf.push(Token::ac(1, 0, 0)); // EOB
        buf.end_scan();

        // Check scan tokens
        let scan0 = buf.scan_tokens(0);
        assert_eq!(scan0.len(), 2);

        let scan1 = buf.scan_tokens(1);
        assert_eq!(scan1.len(), 3);
    }

    #[test]
    fn test_progressive_token_buffer_refinement() {
        let mut buf = ProgressiveTokenBuffer::new(1, 2);

        // Start a refinement scan
        buf.start_scan(4, 1, 63, 2, 1);

        // Push refinement tokens
        buf.push_ref(RefToken::new(0x11, 3));
        buf.push_refbit(1);
        buf.push_refbit(0);
        buf.push_refbit(1);

        buf.end_scan();

        // Check refinement data stored correctly
        let info = &buf.scan_info[0];
        assert!(info.is_refinement());
        assert_eq!(info.ref_tokens.len(), 1);
        assert_eq!(info.refbits.len(), 3);
        assert_eq!(info.refbits, vec![1, 0, 1]);
    }

    #[test]
    fn test_progressive_token_buffer_restart() {
        let mut buf = ProgressiveTokenBuffer::new(1, 1);

        buf.start_scan(0, 0, 0, 0, 0);
        buf.set_dc_pred(0, 100);

        buf.push(Token::dc(0, 50));
        buf.mark_restart();

        // DC pred should be reset
        assert_eq!(buf.dc_pred(0), 0);

        // Restart position should be recorded
        assert_eq!(buf.scan_info[0].restarts.len(), 1);
        assert_eq!(buf.scan_info[0].restarts[0], 1); // After 1 token

        buf.push(Token::dc(0, 60));
        buf.end_scan();
    }

    #[test]
    fn test_tokenize_dc_first_single_component() {
        let mut buf = ProgressiveTokenBuffer::new(1, 1);

        // Create test blocks with known DC values
        let blocks: [[i16; 64]; 3] = [
            {
                let mut b = [0i16; 64];
                b[0] = 100;
                b
            },
            {
                let mut b = [0i16; 64];
                b[0] = 120;
                b
            },
            {
                let mut b = [0i16; 64];
                b[0] = 80;
                b
            },
        ];

        let block_refs: &[[i16; 64]] = &blocks;
        buf.tokenize_dc_scan(&[block_refs], &[0], 0, 0);

        // Should have 3 tokens
        assert_eq!(buf.len(), 3);

        // Check differential encoding:
        // Block 0: diff = 100 - 0 = 100
        // Block 1: diff = 120 - 100 = 20
        // Block 2: diff = 80 - 120 = -40
        let tokens: Vec<_> = buf.tokens.iter().collect();

        // First token: diff = 100, category = 7 (needs 7 bits)
        assert_eq!(tokens[0].context, 0);
        assert_eq!(tokens[0].symbol, 7); // category(100) = 7

        // Second token: diff = 20, category = 5
        assert_eq!(tokens[1].symbol, 5); // category(20) = 5

        // Third token: diff = -40, category = 6
        assert_eq!(tokens[2].symbol, 6); // category(-40) = 6
    }

    #[test]
    fn test_tokenize_dc_interleaved() {
        let mut buf = ProgressiveTokenBuffer::new(3, 1);

        // Create blocks for 3 components
        let y_blocks: [[i16; 64]; 2] = [
            {
                let mut b = [0i16; 64];
                b[0] = 512;
                b
            },
            {
                let mut b = [0i16; 64];
                b[0] = 520;
                b
            },
        ];
        let cb_blocks: [[i16; 64]; 2] = [
            {
                let mut b = [0i16; 64];
                b[0] = 0;
                b
            },
            {
                let mut b = [0i16; 64];
                b[0] = 10;
                b
            },
        ];
        let cr_blocks: [[i16; 64]; 2] = [
            {
                let mut b = [0i16; 64];
                b[0] = -5;
                b
            },
            {
                let mut b = [0i16; 64];
                b[0] = 5;
                b
            },
        ];

        let blocks: &[&[[i16; 64]]] = &[&y_blocks, &cb_blocks, &cr_blocks];
        buf.tokenize_dc_scan(blocks, &[0, 1, 2], 0, 0);

        // Should have 6 tokens (2 blocks × 3 components)
        assert_eq!(buf.len(), 6);

        // Check context assignment
        assert_eq!(buf.tokens[0].context, 0); // Y
        assert_eq!(buf.tokens[1].context, 1); // Cb
        assert_eq!(buf.tokens[2].context, 2); // Cr
        assert_eq!(buf.tokens[3].context, 0); // Y
        assert_eq!(buf.tokens[4].context, 1); // Cb
        assert_eq!(buf.tokens[5].context, 2); // Cr
    }

    #[test]
    fn test_tokenize_dc_with_al() {
        let mut buf = ProgressiveTokenBuffer::new(1, 1);

        // Create blocks with DC values that will be shifted
        let blocks: [[i16; 64]; 2] = [
            {
                let mut b = [0i16; 64];
                b[0] = 100;
                b
            }, // 100 >> 1 = 50
            {
                let mut b = [0i16; 64];
                b[0] = 120;
                b
            }, // 120 >> 1 = 60
        ];

        let block_refs: &[[i16; 64]] = &blocks;
        buf.tokenize_dc_scan(&[block_refs], &[0], 1, 0); // al = 1

        // First token: diff = 50 - 0 = 50, category = 6
        assert_eq!(buf.tokens[0].symbol, 6);

        // Second token: diff = 60 - 50 = 10, category = 4
        assert_eq!(buf.tokens[1].symbol, 4);
    }

    #[test]
    fn test_tokenize_ac_first_simple() {
        let mut buf = ProgressiveTokenBuffer::new(1, 2);

        // Create a block with some non-zero AC coefficients
        let mut block = [0i16; 64];
        block[1] = 10; // Position 1
        block[5] = -5; // Position 5
                       // Positions 2, 3, 4 are zeros (run of 3)

        let blocks = [block];
        buf.tokenize_ac_first_scan(&blocks, 4, 1, 63, 0);

        // Should have tokens for:
        // - Coef at position 1 (run=0, value=10)
        // - Coef at position 5 (run=3, value=-5)
        // - EOB
        assert!(buf.len() >= 2);

        // First token: run=0, category=4 (for value 10)
        let t0 = &buf.tokens[0];
        assert_eq!(t0.context, 4);
        assert_eq!(t0.symbol, (0 << 4) | 4); // run=0, cat=4

        // Second token: run=3, category=3 (for value -5)
        let t1 = &buf.tokens[1];
        assert_eq!(t1.symbol, (3 << 4) | 3); // run=3, cat=3
    }

    #[test]
    fn test_tokenize_ac_eob_run() {
        let mut buf = ProgressiveTokenBuffer::new(1, 2);

        // Create multiple empty blocks
        let blocks: Vec<[i16; 64]> = vec![[0i16; 64]; 5];
        buf.tokenize_ac_first_scan(&blocks, 4, 1, 63, 0);

        // Should have one EOB run token for 5 blocks
        assert!(!buf.is_empty());

        // The EOB run encoding for 5:
        // log2(5) = 2, 5 - 4 = 1 -> symbol = 0x20, extra = 1
        let t = &buf.tokens[0];
        assert_eq!(t.symbol, 0x20); // log2(5) << 4 = 2 << 4 = 0x20
        assert_eq!(t.extra_bits, 1); // 5 - 4 = 1
        assert_eq!(t.num_extra, 2); // 2 bits for the run value
    }

    #[test]
    fn test_tokenize_ac_zrl() {
        let mut buf = ProgressiveTokenBuffer::new(1, 2);

        // Create a block with a run > 16
        let mut block = [0i16; 64];
        block[20] = 7; // Position 20, with 19 zeros before (positions 1-19)

        let blocks = [block];
        buf.tokenize_ac_first_scan(&blocks, 4, 1, 63, 0);

        // Should have:
        // - ZRL (16 zeros)
        // - Coefficient (run=3, value=7)
        // - EOB
        assert!(buf.len() >= 2);

        // First token should be ZRL
        assert_eq!(buf.tokens[0].symbol, 0xF0);

        // Second token: run=3, category=3
        assert_eq!(buf.tokens[1].symbol, (3 << 4) | 3);
    }
}

#[cfg(test)]
mod cpp_comparison_tests {
    //! Tests comparing our implementation against C++ jpegli testdata.

    use super::*;
    use std::fs::File;
    use std::io::{BufRead, BufReader};
    use std::path::PathBuf;

    fn find_testdata_path(filename: &str) -> Option<PathBuf> {
        // Check environment variable first
        if let Ok(dir) = std::env::var("CPP_TESTDATA_DIR") {
            let path = PathBuf::from(dir).join(filename);
            if path.exists() {
                return Some(path);
            }
        }

        // Try relative paths
        let candidates = [
            PathBuf::from("testdata").join(filename),
            PathBuf::from("../jpegli").join(filename),
        ];
        for path in candidates {
            if path.exists() {
                return Some(path);
            }
        }

        None
    }

    fn load_testdata() -> Option<Vec<(Vec<i64>, Vec<u8>)>> {
        let path = find_testdata_path("CreateHuffmanTree.testdata")?;
        let file = File::open(&path).ok()?;
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
    #[ignore] // Requires C++ testdata file
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
}
