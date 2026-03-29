//! Histogram clustering for Huffman table optimization.
//!
//! This module implements the C++ ClusterJpegHistograms algorithm for
//! merging similar symbol frequency histograms to reduce the number of
//! Huffman tables needed in the JPEG file.

#![allow(dead_code)]

use super::frequency::FrequencyCounter;

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
    #[cfg(feature = "__debug-tokens")]
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
            #[cfg(feature = "__debug-tokens")]
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
    #[cfg(feature = "__debug-tokens")]
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

    #[cfg(feature = "__debug-tokens")]
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

            #[cfg(feature = "__debug-tokens")]
            merge_log.push((ctx_idx, target_slot, best_cost));
        }
    }

    result.num_clusters = result.cluster_histograms.len();

    #[cfg(feature = "__debug-tokens")]
    {
        result.merge_log = merge_log;
    }

    result
}
