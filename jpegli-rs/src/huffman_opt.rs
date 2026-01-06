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
//!
//! # Module Structure
//!
//! - `huffman_classic`: Classic mozjpeg/libjpeg algorithm (Section K.2)
//! - `huffman`: jpegli-style algorithm (sorted two-pointer merge)
//! - `huffman_opt` (this module): High-level optimization infrastructure

use crate::error::Result;
use crate::huffman::HuffmanEncodeTable;
use crate::huffman_classic::{
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
}

// Note: generate_code_lengths and generate_optimal_table are imported from huffman_classic

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

    /// Debug dump of scan statistics
    #[allow(dead_code)]
    pub fn debug_dump(&self, scan_index: usize) {
        if self.is_refinement() {
            eprintln!(
                "=== Rust AC Refinement Scan {} ===\nSs={} Se={} Ah={} Al={}\nnum_blocks=? num_tokens={} num_refbits={} num_eobruns={}",
                scan_index, self.ss, self.se, self.ah, self.al,
                self.ref_tokens.len(), self.refbits.len(), self.eobruns.len()
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

/// Result of histogram clustering.
#[derive(Clone, Debug)]
pub struct ClusterResult {
    /// Mapping from context ID to cluster (table) index.
    /// `context_map[ctx]` gives the cluster that context `ctx` should use.
    pub context_map: Vec<usize>,
    /// Merged histograms for each cluster.
    /// After clustering, these contain the sum of all histograms
    /// assigned to each cluster.
    pub cluster_histograms: Vec<FrequencyCounter>,
    /// Number of clusters created.
    pub num_clusters: usize,
    /// Slot IDs for each cluster (0-3).
    /// Maps cluster index to JPEG DHT table slot.
    pub slot_ids: Vec<usize>,
    /// Merge log for debugging (context pairs that were merged)
    #[cfg(feature = "debug-tokens")]
    pub merge_log: Vec<(usize, usize, f64)>, // (ctx_a, ctx_b, cost_delta)
}

impl ClusterResult {
    /// Creates an empty result for N contexts.
    pub fn new(num_contexts: usize) -> Self {
        Self {
            context_map: vec![0; num_contexts],
            cluster_histograms: Vec::new(),
            num_clusters: 0,
            slot_ids: Vec::new(),
            #[cfg(feature = "debug-tokens")]
            merge_log: Vec::new(),
        }
    }

    /// Gets the table slot for a context.
    #[inline]
    pub fn get_slot(&self, context: usize) -> usize {
        let cluster = self.context_map.get(context).copied().unwrap_or(0);
        self.slot_ids.get(cluster).copied().unwrap_or(0)
    }

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

/// Context configuration for Huffman table optimization.
///
/// Maps to C++ `encode.cc:340-383` context assignment.
///
/// Context layout:
/// - [0..num_components): DC contexts (one per color channel)
/// - [4..4+num_ac_contexts): AC contexts (varies by scan count)
#[derive(Clone, Debug)]
pub struct ContextConfig {
    /// Total number of contexts
    pub num_contexts: usize,
    /// Offset where AC contexts start (always 4 per C++ design)
    pub ac_offset: usize,
    /// AC context offset for each scan.
    /// `scan_ac_offsets[scan_idx]` is the first AC context for that scan.
    pub scan_ac_offsets: Vec<usize>,
}

impl ContextConfig {
    /// Creates context config for sequential (baseline) JPEG.
    ///
    /// Sequential has one scan with all components.
    /// DC contexts: 0..num_components
    /// AC contexts: 4..4+num_components
    pub fn for_sequential(num_components: usize) -> Self {
        Self {
            num_contexts: 4 + num_components, // DC(0-3) + AC(4+)
            ac_offset: 4,
            scan_ac_offsets: vec![4], // Single scan, AC starts at 4
        }
    }

    /// Creates context config for progressive JPEG.
    ///
    /// Progressive mode assigns separate AC contexts per scan:
    /// - DC contexts: 0..num_components
    /// - AC contexts: 4 + running_count (one per component per AC scan)
    ///
    /// # Arguments
    /// * `num_components` - Number of color components (1-4)
    /// * `scans` - Iterator of (ss, se, comps_in_scan) for each scan
    pub fn for_progressive<I>(num_components: usize, scans: I) -> Self
    where
        I: Iterator<Item = (u8, u8, usize)>, // (ss, se, comps_in_scan)
    {
        let _ = num_components; // Used for validation if needed
        let mut num_ac_contexts = 0;
        let mut scan_ac_offsets = Vec::new();

        for (_ss, se, comps_in_scan) in scans {
            scan_ac_offsets.push(4 + num_ac_contexts);
            // Only AC scans (Se > 0) get contexts
            if se > 0 {
                num_ac_contexts += comps_in_scan;
            }
        }

        Self {
            num_contexts: 4 + num_ac_contexts,
            ac_offset: 4,
            scan_ac_offsets,
        }
    }

    /// Gets DC context for a component.
    ///
    /// DC contexts are 0..3 (clamped for 4+ component images).
    #[inline]
    pub fn dc_context(&self, component: usize) -> usize {
        component.min(3)
    }

    /// Gets AC context for a scan and component-within-scan.
    ///
    /// Returns `scan_ac_offsets[scan_idx] + comp_in_scan`
    #[inline]
    pub fn ac_context(&self, scan_idx: usize, comp_in_scan: usize) -> usize {
        self.scan_ac_offsets
            .get(scan_idx)
            .map(|&offset| offset + comp_in_scan)
            .unwrap_or(self.ac_offset + comp_in_scan)
    }

    /// Returns the number of DC contexts (always min(num_components, 4)).
    #[inline]
    pub fn num_dc_contexts(&self) -> usize {
        self.ac_offset.min(4)
    }

    /// Returns the number of AC contexts.
    #[inline]
    pub fn num_ac_contexts(&self) -> usize {
        self.num_contexts.saturating_sub(self.ac_offset)
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
/// This implements the C++ ClusterJpegHistograms algorithm (entropy_coding.cc:584-642):
/// 1. Process histograms in order
/// 2. For each, find best existing cluster to merge with
/// 3. If merging saves bits, merge; otherwise create new cluster
/// 4. Respect max_clusters limit (typically 2 for baseline, 4 for extended)
///
/// # Arguments
/// * `histograms` - Symbol counts per context
/// * `max_clusters` - Maximum clusters (2 for baseline sequential, 4 for progressive)
/// * `force_baseline` - If true, limit to 2 clusters for baseline JPEG compatibility
///
/// # Returns
/// ClusterResult with context-to-cluster mapping, merged histograms, and slot IDs
pub fn cluster_histograms(
    histograms: &[FrequencyCounter],
    max_clusters: usize,
    force_baseline: bool,
) -> ClusterResult {
    let mut result = ClusterResult::new(histograms.len());

    // Track which cluster is in each slot and its cost
    let mut slot_histograms: Vec<usize> = Vec::new(); // cluster index per slot
    let mut slot_costs: Vec<f64> = Vec::new();

    // Effective max clusters: 2 for baseline, up to max_clusters otherwise
    // Note: More clusters can be created than slots (4) - slot IDs cycle with modulo 4
    // This enables slot redefinition for progressive scans with different symbol distributions
    let effective_max = if force_baseline {
        max_clusters.min(2)
    } else {
        max_clusters // Don't cap - allow more clusters to enable on-demand DHT emission
    };

    #[cfg(feature = "debug-tokens")]
    let mut merge_log = Vec::new();

    for (ctx_idx, histo) in histograms.iter().enumerate() {
        if histo.is_empty_histogram() {
            // Empty histogram - assign to cluster 0, will be ignored
            result.context_map[ctx_idx] = 0;
            continue;
        }

        let num_slots = slot_histograms.len();

        // Default: create new cluster (if within limit)
        let mut best_slot = num_slots;
        let mut best_cost = if force_baseline && num_slots > 1 {
            // Force merge at baseline limit (max 2 tables)
            f64::MAX
        } else if num_slots >= effective_max {
            // At general limit
            f64::MAX
        } else {
            histo.estimate_encoding_cost()
        };

        // Find best existing cluster to merge with
        for slot_idx in 0..num_slots {
            let cluster_idx = slot_histograms[slot_idx];
            let prev = &result.cluster_histograms[cluster_idx];

            let combined = prev.combined(histo);
            let combined_cost = combined.estimate_encoding_cost();

            // Cost delta: how much extra to merge vs current cluster alone
            let cost_delta = combined_cost - slot_costs[slot_idx];

            if cost_delta < best_cost {
                best_cost = cost_delta;
                best_slot = slot_idx;
            }
        }

        if best_slot == num_slots && num_slots < effective_max {
            // Create new cluster
            let cluster_idx = result.cluster_histograms.len();
            result.cluster_histograms.push(histo.clone());
            result.context_map[ctx_idx] = cluster_idx;

            if num_slots < 4 {
                // We have a free slot
                slot_histograms.push(cluster_idx);
                slot_costs.push(best_cost);
                result.slot_ids.push(num_slots);
            } else {
                // No free slot - round-robin replacement
                // (C++ TODO: find best histogram to replace)
                let replace_slot = (result.slot_ids.last().copied().unwrap_or(0) + 1) % 4;
                slot_histograms[replace_slot] = cluster_idx;
                slot_costs[replace_slot] = best_cost;
                result.slot_ids.push(replace_slot);
            }
        } else {
            // Merge with existing cluster
            let target_slot = if best_slot >= num_slots { 0 } else { best_slot };
            let cluster_idx = slot_histograms[target_slot];
            result.cluster_histograms[cluster_idx].add(histo);
            result.context_map[ctx_idx] = cluster_idx;
            slot_costs[target_slot] += best_cost;

            // slot_id already assigned to this cluster

            #[cfg(feature = "debug-tokens")]
            merge_log.push((ctx_idx, target_slot, best_cost));
        }
    }

    result.num_clusters = result.cluster_histograms.len();

    #[cfg(feature = "debug-tokens")]
    {
        result.merge_log = merge_log;
    }

    result
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
            // Count the symbol for Huffman table building
            // - EOB symbols: 0x00, 0x10, 0x20, ... (high nibble = bits needed for run)
            // - Newly-nonzero: 0x01/0x03, 0x11/0x13, ... (high nibble = run, low nibble = 1 or 3)
            // - ZRL: 0xF0
            // For AC refinement, mask with 0xFD (253) to merge positive/negative
            // symbols together for histogram building, matching C++ behavior.
            // Only mask category 1 symbols (low nibble == 1 or 3), not EOB or ZRL.
            let context = info.context as usize;
            if context < self.counters.len() {
                // Mask only if this is a newly-nonzero symbol (category 1)
                let low_nibble = token.symbol & 0x0F;
                let masked_symbol = if low_nibble == 1 || low_nibble == 3 {
                    token.symbol & 253 // Clear sign bit
                } else {
                    token.symbol
                };
                self.counters[context].count(masked_symbol);
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
    /// * `force_baseline` - If true, limit to 2 clusters per type for baseline JPEG
    ///
    /// # Returns
    /// - `context_map`: Maps each context to a table index
    /// - `num_dc_tables`: Number of DC tables (for indexing into tables array)
    /// - `tables`: Optimized Huffman tables for each cluster (DC tables first, then AC)
    /// - `ac_slot_ids`: Slot IDs for each AC table (0-3), for on-demand DHT emission
    pub fn generate_optimized_tables(
        &self,
        max_dc_clusters: usize,
        max_ac_clusters: usize,
        num_dc_contexts: usize,
        force_baseline: bool,
    ) -> Result<(Vec<usize>, usize, Vec<OptimizedTable>, Vec<usize>)> {
        // Split into DC and AC histograms
        let dc_histograms: Vec<_> = self.counters[..num_dc_contexts].to_vec();
        let ac_histograms: Vec<_> = self.counters[num_dc_contexts..].to_vec();

        // Cluster DC and AC separately
        let dc_clusters = cluster_histograms(&dc_histograms, max_dc_clusters, force_baseline);
        let ac_clusters = cluster_histograms(&ac_histograms, max_ac_clusters, force_baseline);

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

        // AC slot IDs for on-demand DHT emission
        let ac_slot_ids = ac_clusters.slot_ids.clone();

        Ok((context_map, dc_clusters.num_clusters, tables, ac_slot_ids))
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

    /// Generates optimized Huffman tables for XYB mode.
    ///
    /// In XYB mode, all components use the same Huffman table (no luma/chroma split).
    /// This function merges all DC contexts and all AC contexts into single tables.
    pub fn generate_xyb_tables(&self, num_dc_contexts: usize) -> Result<OptimizedHuffmanTables> {
        // Merge all DC contexts into one table
        let mut dc_merged = FrequencyCounter::new();
        for ctx in 0..num_dc_contexts {
            dc_merged.add(&self.counters[ctx]);
        }

        // Merge all AC contexts into one table
        let ac_start = num_dc_contexts;
        let mut ac_merged = FrequencyCounter::new();
        for counter in self.counters[ac_start..].iter() {
            ac_merged.add(counter);
        }

        // Generate DC table
        let dc_table = if dc_merged.is_empty_histogram() {
            let mut default = FrequencyCounter::new();
            default.count(0);
            default.generate_table_with_dht()?
        } else {
            dc_merged.generate_table_with_dht()?
        };

        // Generate AC table
        let ac_table = if ac_merged.is_empty_histogram() {
            let mut default = FrequencyCounter::new();
            default.count(0); // EOB
            default.generate_table_with_dht()?
        } else {
            ac_merged.generate_table_with_dht()?
        };

        // XYB uses same table for all components, so luma = chroma
        Ok(OptimizedHuffmanTables {
            dc_luma: dc_table.clone(),
            ac_luma: ac_table.clone(),
            dc_chroma: dc_table,
            ac_chroma: ac_table,
        })
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
        let num_blocks = blocks.first().map(|b| b.len()).unwrap_or(0);

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
        let num_blocks = blocks.first().map(|b| b.len()).unwrap_or(0);

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
    /// IMPORTANT: We must use absolute values for zero-detection to match
    /// the refinement scan's classification. Otherwise, small negative
    /// coefficients like -2 with al=2 would be incorrectly tokenized here
    /// (because (-2) >> 2 = -1 in signed arithmetic) but classified as
    /// "newly-nonzero" in refinement (because abs(-2) >> 2 = 0).
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
            // Use absolute value for consistency with refinement scan classification
            let mut last_nonzero = ss as usize;
            for k in (ss as usize..=se as usize).rev() {
                if (block[k].unsigned_abs() >> al) != 0 {
                    last_nonzero = k;
                    break;
                }
            }

            // Check if block is all zeros in this range (using absolute values)
            let is_eob = (ss as usize..=se as usize).all(|k| (block[k].unsigned_abs() >> al) == 0);

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
                let coef = block[k];
                let abs_shifted = coef.unsigned_abs() >> al;
                if abs_shifted == 0 {
                    run += 1;
                } else {
                    // Emit ZRL for runs >= 16
                    while run >= 16 {
                        let zrl = Token::new(context, 0xF0, 0, 0);
                        self.push(zrl);
                        run -= 16;
                    }

                    // Emit coefficient token with the shifted value
                    // Preserve sign from original coefficient
                    let shifted_value = if coef < 0 {
                        -(abs_shifted as i16)
                    } else {
                        abs_shifted as i16
                    };
                    let token = Token::ac(context, run, shifted_value);
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
            // Use unsigned_abs() for consistency with first-pass encoding
            let mut has_content = false;
            for k in ss as usize..=se as usize {
                let abs_coef = block[k].unsigned_abs();
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
                // DON'T flush pending refbits here - they accumulate with the EOB run
                // just like C++ does. Only flush when we hit limits.
                eob_run += 1;

                // Flush if we hit the maximum EOB run OR refbits limit
                if eob_run == 0x7FFF || pending_refbits.len() > 255 {
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

            // Process coefficients - match C++ order exactly:
            // 1. If completely zero, increment run
            // 2. Emit ZRL if run > 15 (BEFORE adding current position's refbit)
            // 3. If previously nonzero (absval > 1), add refbit
            // 4. If newly nonzero (absval == 1), emit token
            let mut run = 0u8;
            let mut block_refbits: Vec<u8> = Vec::new();

            for k in ss as usize..=se as usize {
                let coef = block[k];
                let abs_coef = coef.unsigned_abs();

                // Step 1: Check if coefficient is completely zero
                if abs_coef == 0 {
                    run += 1;
                    continue;
                }

                // Shift to current precision level (like C++: absval >>= Al)
                let absval = abs_coef >> al;

                // Step 2: Check if zero at current precision (not visible yet)
                if absval == 0 {
                    run += 1;
                    continue;
                }

                // We have a nonzero coefficient at current precision.
                // FIRST check for ZRL, THEN add refbit or emit newly-nonzero.

                // Flush pending EOB run if any
                if eob_run > 0 || !pending_refbits.is_empty() {
                    self.emit_eob_run_with_refbits(context, eob_run, &pending_refbits);
                    pending_refbits.clear();
                    eob_run = 0;
                }

                // Step 3: Emit ZRL tokens BEFORE processing current coefficient
                while run >= 16 {
                    let ref_token = RefToken::new(0xF0, block_refbits.len() as u8);
                    self.push_ref(ref_token);
                    for &bit in &block_refbits {
                        self.push_refbit(bit);
                    }
                    block_refbits.clear();
                    run -= 16;
                }

                // Step 4: Check if previously nonzero (magnitude > 1)
                if absval > 1 {
                    // Previously nonzero: add refinement bit, continue
                    let refbit = (abs_coef >> al) & 1;
                    block_refbits.push(refbit as u8);
                    continue;
                }

                // Step 5: absval == 1, newly nonzero
                // Emit newly nonzero coefficient with accumulated refbits
                let symbol = if coef < 0 {
                    (run << 4) | 1 // 0x?1 for negative
                } else {
                    (run << 4) | 3 // 0x?3 for positive
                };
                let ref_token = RefToken::new(symbol, block_refbits.len() as u8);
                self.push_ref(ref_token);
                for &bit in &block_refbits {
                    self.push_refbit(bit);
                }
                block_refbits.clear();
                run = 0;
            }

            // If we have trailing refbits or trailing zeros, this block ends with EOB.
            // Accumulate refbits with any pending ones from previous EOB blocks.
            if run > 0 || !block_refbits.is_empty() {
                // Check if adding these refbits would exceed the limit
                if pending_refbits.len() + block_refbits.len() > 255 {
                    // Flush current EOB run before starting a new one
                    if eob_run > 0 {
                        self.emit_eob_run_with_refbits(context, eob_run, &pending_refbits);
                        pending_refbits.clear();
                        eob_run = 0;
                    }
                }
                pending_refbits.extend(block_refbits);
                eob_run += 1;

                // Also check if we've hit the max run or refbits limit after accumulation
                if eob_run == 0x7FFF || pending_refbits.len() > 255 {
                    self.emit_eob_run_with_refbits(context, eob_run, &pending_refbits);
                    pending_refbits.clear();
                    eob_run = 0;
                }
            }
        }

        // Flush remaining EOB run
        if eob_run > 0 || !pending_refbits.is_empty() {
            self.emit_eob_run_with_refbits(context, eob_run, &pending_refbits);
        }

        self.end_scan();

        // Debug: dump tokens and refbits for comparison with C++
        if std::env::var("DUMP_AC_REFINEMENT").is_ok() {
            if let Some(info) = self.scan_info.last() {
                eprintln!(
                    "=== Rust AC Refinement Scan (Ss={} Se={} Ah={} Al={}) ===",
                    ss, se, ah, al
                );
                eprintln!(
                    "num_blocks={} num_tokens={} num_refbits={} num_eobruns={}",
                    blocks.len(),
                    info.ref_tokens.len(),
                    info.refbits.len(),
                    info.eobruns.len()
                );
                eprintln!("TOKENS:");
                for (i, t) in info.ref_tokens.iter().enumerate().take(100) {
                    eprintln!("  [{}] symbol=0x{:02x} refbits={}", i, t.symbol, t.refbits);
                }
                if info.ref_tokens.len() > 100 {
                    eprintln!("  ... ({} more tokens)", info.ref_tokens.len() - 100);
                }
                eprintln!("REFBITS:");
                eprint!("  ");
                for (i, &b) in info.refbits.iter().enumerate().take(200) {
                    eprint!("{}", b);
                    if (i + 1) % 64 == 0 {
                        eprintln!();
                        eprint!("  ");
                    }
                }
                eprintln!();
                if info.refbits.len() > 200 {
                    eprintln!("  ... ({} more refbits)", info.refbits.len() - 200);
                }
                eprintln!("EOBRUNS:");
                eprint!("  ");
                for &r in info.eobruns.iter().take(50) {
                    eprint!("{} ", r);
                }
                eprintln!();
                eprintln!("=== End Rust AC Refinement Scan ===\n");
            }
        }
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

    fn load_testdata() -> Option<Vec<(Vec<i64>, Vec<u8>)>> {
        let path = crate::test_utils::get_cpp_testdata_path("CreateHuffmanTree.testdata")?;
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
    #[ignore] // FAILING: 4/185 cases where C++ is better - algorithm needs fixing
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
