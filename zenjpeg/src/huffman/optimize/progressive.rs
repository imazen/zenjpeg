//! Progressive JPEG tokenization buffer.
//!
//! This module provides `ProgressiveTokenBuffer` for two-pass progressive
//! JPEG encoding with optimized Huffman tables.

#![allow(dead_code)]

use crate::error::Result;
use tinyvec::ArrayVec;

use super::cluster::cluster_histograms;
use super::frequency::{FrequencyCounter, HuffmanTableSet, OptimizedTable};
use super::tokens::{RefToken, ScanTokenInfo, Token};

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

    /// Starts a new AC refinement scan with pre-allocated capacity.
    ///
    /// Uses fallible allocation for the internal vectors based on block count.
    pub fn start_scan_for_refinement(
        &mut self,
        context: u8,
        ss: u8,
        se: u8,
        ah: u8,
        al: u8,
        num_blocks: usize,
    ) -> Result<()> {
        let mut info =
            ScanTokenInfo::with_capacity_refinement(context, ss, se, ah, al, num_blocks)?;
        info.token_offset = self.tokens.len();
        self.scan_info.push(info);
        Ok(())
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
    pub fn generate_xyb_tables(&self, num_dc_contexts: usize) -> Result<HuffmanTableSet> {
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
        Ok(HuffmanTableSet {
            dc_luma: dc_table.clone(),
            ac_luma: ac_table.clone(),
            dc_chroma: dc_table,
            ac_chroma: ac_table,
        })
    }

    /// Dumps all tokens to a JSON file for C++ comparison.
    #[cfg(feature = "__debug-tokens")]
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
    #[cfg(feature = "__debug-tokens")]
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
        restart_interval: u16,
    ) {
        // Start the scan - DC uses context = component index
        // For interleaved scans, we'll emit tokens for each component
        self.start_scan(0, 0, 0, ah, al);
        self.reset_dc_pred();

        if ah == 0 {
            // First DC scan: encode DC coefficients shifted by al
            self.tokenize_dc_first(blocks, component_indices, al, restart_interval);
        } else {
            // DC refinement: just emit one bit per block
            self.tokenize_dc_refine(blocks, component_indices, al, restart_interval);
        }

        self.end_scan();
    }

    /// Tokenizes DC first scan (ah == 0).
    fn tokenize_dc_first(
        &mut self,
        blocks: &[&[[i16; 64]]],
        component_indices: &[usize],
        al: u8,
        restart_interval: u16,
    ) {
        // Get the number of blocks (all components should have same count for interleaved)
        let num_blocks = blocks.first().map(|b| b.len()).unwrap_or(0);
        let ri = restart_interval as usize;

        // Iterate using zip to ensure matched lengths without bounds checks
        for block_idx in 0..num_blocks {
            // Insert restart boundary: reset DC prediction, mark position.
            // block_idx counts MCUs (1 block per component per MCU).
            if ri > 0 && block_idx > 0 && block_idx % ri == 0 {
                self.mark_restart();
            }

            for (&comp_idx, &comp_blocks) in component_indices.iter().zip(blocks.iter()) {
                // Use get() for block access - gracefully handles mismatched block counts
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

    /// Tokenizes DC refinement scan (ah > 0).
    fn tokenize_dc_refine(
        &mut self,
        blocks: &[&[[i16; 64]]],
        component_indices: &[usize],
        al: u8,
        restart_interval: u16,
    ) {
        let num_blocks = blocks.first().map(|b| b.len()).unwrap_or(0);
        let ri = restart_interval as usize;

        // Iterate using zip to ensure matched lengths without bounds checks
        for block_idx in 0..num_blocks {
            // Insert restart boundary.
            if ri > 0 && block_idx > 0 && block_idx % ri == 0 {
                self.mark_restart();
            }

            for (&comp_idx, &comp_blocks) in component_indices.iter().zip(blocks.iter()) {
                // Use get() for block access - gracefully handles mismatched block counts
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
    /// * `se` - Spectral selection end (1-63, >= ss)
    /// * `al` - Successive approximation low bit
    ///
    /// # Panics
    /// Panics if `ss == 0`, `se > 63`, or `ss > se`.
    pub fn tokenize_ac_first_scan(
        &mut self,
        blocks: &[[i16; 64]],
        context: u8,
        ss: u8,
        se: u8,
        al: u8,
        restart_interval: u16,
    ) {
        // Validate spectral selection bounds (1-63 for AC, ss <= se)
        // This assertion helps the compiler eliminate bounds checks below.
        assert!(
            ss >= 1 && se <= 63 && ss <= se,
            "invalid spectral selection: ss={}, se={}",
            ss,
            se
        );

        self.start_scan(context, ss, se, 0, al);

        let mut eob_run: u16 = 0;
        let ri = restart_interval as usize;

        // Convert ss/se to usize once, with bounds the compiler can prove
        let ss_idx = ss as usize;
        let se_idx = se as usize;
        let coef_count = se_idx - ss_idx + 1;

        for (block_idx, block) in blocks.iter().enumerate() {
            // Restart boundary: flush pending EOB run, mark position.
            // Each block is one MCU in a non-interleaved scan.
            if ri > 0 && block_idx > 0 && block_idx % ri == 0 {
                if eob_run > 0 {
                    self.emit_eob_run(context, eob_run);
                    eob_run = 0;
                }
                self.mark_restart();
            }

            // Extract the coefficient slice once per block.
            // Since we validated 1 <= ss <= se <= 63, this is guaranteed in-bounds.
            // The compiler can now eliminate bounds checks in the inner loops.
            let coeffs = &block[ss_idx..=se_idx];

            // Find last nonzero coefficient in spectral range (relative to slice start)
            // We search from end to start and track both last_nonzero_rel and found_nonzero in one pass
            let mut last_nonzero_rel = 0usize;
            let mut found_nonzero = false;
            for (i, &coef) in coeffs.iter().enumerate().rev() {
                if (coef.unsigned_abs() >> al) != 0 {
                    last_nonzero_rel = i;
                    found_nonzero = true;
                    break;
                }
            }

            // If no nonzero found, the block is all zeros in this range
            if !found_nonzero {
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

            // Encode coefficients (using relative index within slice)
            let mut run = 0u8;
            for (i, &coef) in coeffs.iter().enumerate() {
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

                if i == last_nonzero_rel {
                    break;
                }
            }

            // If we didn't reach the end, emit EOB
            // (last_nonzero_rel is relative, so compare against coef_count - 1)
            if last_nonzero_rel < coef_count - 1 {
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
    /// * `ss` - Spectral selection start (1-63)
    /// * `se` - Spectral selection end (1-63, >= ss)
    /// * `ah` - Successive approximation high bit (previous precision)
    /// * `al` - Successive approximation low bit (current precision)
    ///
    /// # Errors
    /// Returns an error if memory allocation fails.
    pub fn tokenize_ac_refinement_scan(
        &mut self,
        blocks: &[[i16; 64]],
        context: u8,
        ss: u8,
        se: u8,
        ah: u8,
        al: u8,
        restart_interval: u16,
    ) -> Result<()> {
        use crate::error::Error;

        // Validate spectral selection bounds (1-63 for AC, ss <= se)
        // This helps the compiler eliminate bounds checks below.
        if ss == 0 || se > 63 || ss > se {
            return Err(Error::internal(
                "invalid spectral selection for AC refinement",
            ));
        }

        // Pre-allocate scan storage based on block count
        self.start_scan_for_refinement(context, ss, se, ah, al, blocks.len())?;

        let mut eob_run: u16 = 0;
        let ri = restart_interval as usize;

        // Pre-allocate pending_refbits - max 256 before flush
        let mut pending_refbits: Vec<u8> = Vec::new();
        pending_refbits
            .try_reserve(256)
            .map_err(|_| Error::allocation_failed(256, "pending refinement bits"))?;

        // Use stack-allocated ArrayVec for block_refbits - no heap allocation needed.
        // Max refbits per block = se - ss + 1 <= 63 (spectral range 1-63), so 64 is sufficient.
        // ArrayVec is reused with clear() between blocks.
        let mut block_refbits: ArrayVec<[u8; 64]> = ArrayVec::new();

        // Convert ss/se to usize once, with bounds that the compiler can prove
        let ss_idx = ss as usize;
        let se_idx = se as usize;

        for (block_idx, block) in blocks.iter().enumerate() {
            // Restart boundary: flush pending EOB run + refbits, mark position.
            if ri > 0 && block_idx > 0 && block_idx % ri == 0 {
                if eob_run > 0 || !pending_refbits.is_empty() {
                    self.emit_eob_run_with_refbits(context, eob_run, &pending_refbits);
                    pending_refbits.clear();
                    eob_run = 0;
                }
                self.mark_restart();
            }
            // Extract the coefficient slice once per block.
            // Since we validated ss <= se <= 63 above, this is guaranteed in-bounds.
            // The compiler can now eliminate bounds checks in the inner loop.
            let coeffs = &block[ss_idx..=se_idx];

            // Find if there are any newly-nonzero or previously-nonzero coefficients
            let mut has_content = false;
            for &coef in coeffs {
                let abs_coef = coef.unsigned_abs();
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

            // Pre-compute last position with a newly-nonzero coefficient.
            // ZRL must only be emitted BEFORE this position — after it, any
            // ZRL would be followed by EOB, which djpegli rejects (in_zero_run
            // is only cleared by a newly-nonzero symbol, not by EOB).
            // The C++ jpegli encoder handles this by speculatively writing ZRL
            // tokens and rewinding them if EOB follows (next_eob_token mechanism).
            // We use a simpler approach: don't emit ZRL past the last newly-nonzero.
            let last_newly_nonzero_pos = coeffs.iter().rposition(|&c| {
                let abs_coef = c.unsigned_abs();
                let absval = abs_coef >> al;
                absval == 1
            });

            // Process coefficients - match C++ order exactly:
            // 1. If completely zero, increment run
            // 2. Emit ZRL if run > 15 (BEFORE adding current position's refbit)
            //    BUT only if a newly-nonzero follows later in this block
            // 3. If previously nonzero (absval > 1), add refbit
            // 4. If newly nonzero (absval == 1), emit token
            let mut run = 0u8;
            block_refbits.clear(); // Reuse allocation from previous iteration

            for (pos, &coef) in coeffs.iter().enumerate() {
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

                // Step 3: Emit ZRL tokens BEFORE processing current coefficient,
                // but ONLY if a newly-nonzero coefficient follows later in this
                // block. Otherwise the ZRL would be followed by EOB, which is
                // invalid per djpegli's decoder (in_zero_run check). When we
                // suppress ZRL, the run and refbits accumulate until EOB.
                let may_emit_zrl = last_newly_nonzero_pos.is_some_and(|last_pos| pos <= last_pos);
                while run >= 16 && may_emit_zrl {
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
                    // Note: block_refbits capacity was pre-allocated, no grow_one
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
                pending_refbits.extend(&block_refbits);
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
        if std::env::var("DUMP_AC_REFINEMENT").is_ok()
            && let Some(info) = self.scan_info.last()
        {
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

        Ok(())
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
        if run > 1
            && let Some(info) = self.scan_info.last_mut()
        {
            info.eobruns.push(run);
        }

        for &bit in refbits {
            self.push_refbit(bit);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        buf.tokenize_dc_scan(&[block_refs], &[0], 0, 0, 0);

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
        buf.tokenize_dc_scan(blocks, &[0, 1, 2], 0, 0, 0);

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
        buf.tokenize_dc_scan(&[block_refs], &[0], 1, 0, 0); // al = 1

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
        buf.tokenize_ac_first_scan(&blocks, 4, 1, 63, 0, 0);

        // Should have tokens for:
        // - Coef at position 1 (run=0, value=10)
        // - Coef at position 5 (run=3, value=-5)
        // - EOB
        assert!(buf.len() >= 2);

        // First token: run=0, category=4 (for value 10)
        let t0 = &buf.tokens[0];
        assert_eq!(t0.context, 4);
        assert_eq!(t0.symbol, 4); // run=0, cat=4

        // Second token: run=3, category=3 (for value -5)
        let t1 = &buf.tokens[1];
        assert_eq!(t1.symbol, (3 << 4) | 3); // run=3, cat=3
    }

    #[test]
    fn test_tokenize_ac_eob_run() {
        let mut buf = ProgressiveTokenBuffer::new(1, 2);

        // Create multiple empty blocks
        let blocks: Vec<[i16; 64]> = vec![[0i16; 64]; 5];
        buf.tokenize_ac_first_scan(&blocks, 4, 1, 63, 0, 0);

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
        buf.tokenize_ac_first_scan(&blocks, 4, 1, 63, 0, 0);

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
