//! Histogram clustering for Huffman table optimization.
//!
//! This module implements the C++ ClusterJpegHistograms algorithm for
//! merging similar symbol frequency histograms to reduce the number of
//! Huffman tables needed in the JPEG file.

#![allow(dead_code)]

use super::frequency::FrequencyCounter;

/// Strategy for replacing a DHT slot when all 4 are full.
///
/// JPEG allows at most 4 Huffman tables per type (DC/AC). When progressive
/// encoding creates more logical histogram clusters than physical slots,
/// the algorithm must evict one slot to make room. This enum controls
/// which slot is chosen for eviction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SlotReplacement {
    /// Round-robin: cycle through slots 0-3. Matches C++ jpegli behavior.
    #[default]
    RoundRobin,
    /// Replace the slot whose histogram has the fewest total symbols.
    SmallestCount,
    /// Replace the slot whose histogram can be re-merged into another
    /// existing slot at the lowest cost.
    LowestEvictionCost,
    /// Replace the slot that was least recently assigned or updated.
    OldestSlot,
    /// Replace the slot with the highest current encoding cost.
    HighestSlotCost,
    /// Round-robin greedy pass followed by iterative refinement.
    ///
    /// After the standard greedy clustering, performs 1-opt refinement:
    /// for each context, try moving it to every other existing cluster.
    /// Keep any move that reduces total encoding cost. Repeat until
    /// no improvement is found (typically converges in 1-2 passes).
    Refined,
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
/// Uses the default round-robin slot replacement strategy (matching C++ jpegli).
///
/// See [`cluster_histograms_with_strategy`] for alternative replacement strategies.
pub fn cluster_histograms(
    histograms: &[FrequencyCounter],
    max_clusters: usize,
    force_baseline: bool,
) -> ClusterResult {
    cluster_histograms_with_strategy(
        histograms,
        max_clusters,
        force_baseline,
        SlotReplacement::RoundRobin,
    )
}

/// Clusters histograms to minimize total encoding cost, with configurable
/// slot replacement strategy.
///
/// This implements the C++ ClusterJpegHistograms algorithm (entropy_coding.cc:584-642):
/// 1. Process histograms in order
/// 2. For each, find best existing cluster to merge with
/// 3. If merging saves bits, merge; otherwise create new cluster
/// 4. Respect max_clusters limit (typically 2 for baseline, 4 for extended)
///
/// When all 4 DHT slots are occupied and a new cluster is needed,
/// the `strategy` parameter controls which slot is evicted.
///
/// # Arguments
/// * `histograms` - Symbol counts per context
/// * `max_clusters` - Maximum clusters (2 for baseline sequential, 4 for progressive)
/// * `force_baseline` - If true, limit to 2 clusters for baseline JPEG compatibility
/// * `strategy` - Slot replacement strategy when all 4 slots are full
///
/// # Returns
/// ClusterResult with context-to-cluster mapping, merged histograms, and slot IDs
pub fn cluster_histograms_with_strategy(
    histograms: &[FrequencyCounter],
    max_clusters: usize,
    force_baseline: bool,
    strategy: SlotReplacement,
) -> ClusterResult {
    let mut result = ClusterResult::new(histograms.len());

    // Track which cluster is in each slot and its cost
    let mut slot_histograms: Vec<usize> = Vec::new(); // cluster index per slot
    let mut slot_costs: Vec<f64> = Vec::new();
    // Track assignment order for OldestSlot strategy (lower = older)
    let mut slot_age: Vec<usize> = Vec::new();
    let mut age_counter: usize = 0;

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
                slot_age.push(age_counter);
                age_counter += 1;
                result.slot_ids.push(num_slots);
            } else {
                // No free slot — pick which to replace based on strategy
                let replace_slot = pick_replacement_slot(
                    strategy,
                    &result,
                    &slot_histograms,
                    &slot_costs,
                    &slot_age,
                );
                slot_histograms[replace_slot] = cluster_idx;
                slot_costs[replace_slot] = best_cost;
                slot_age[replace_slot] = age_counter;
                age_counter += 1;
                result.slot_ids.push(replace_slot);
            }
        } else {
            // Merge with existing cluster
            let target_slot = if best_slot >= num_slots { 0 } else { best_slot };
            let cluster_idx = slot_histograms[target_slot];
            result.cluster_histograms[cluster_idx].add(histo);
            result.context_map[ctx_idx] = cluster_idx;
            slot_costs[target_slot] += best_cost;

            // Update age on merge (slot was just used)
            if target_slot < slot_age.len() {
                slot_age[target_slot] = age_counter;
                age_counter += 1;
            }

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

    // Post-greedy refinement pass for the Refined strategy
    if strategy == SlotReplacement::Refined {
        refine_clustering(histograms, &mut result);
    }

    result
}

/// Pick which of the 4 occupied slots to replace when a new cluster is needed.
fn pick_replacement_slot(
    strategy: SlotReplacement,
    result: &ClusterResult,
    slot_histograms: &[usize],
    slot_costs: &[f64],
    slot_age: &[usize],
) -> usize {
    debug_assert!(slot_histograms.len() == 4);

    match strategy {
        SlotReplacement::RoundRobin => {
            // C++ jpegli behavior: cycle through slots
            (result.slot_ids.last().copied().unwrap_or(0) + 1) % 4
        }

        SlotReplacement::SmallestCount => {
            // Replace the slot with the fewest total symbols
            (0..4)
                .min_by_key(|&s| result.cluster_histograms[slot_histograms[s]].total())
                .unwrap_or(0)
        }

        SlotReplacement::LowestEvictionCost => {
            // For each slot, compute cost of re-merging its histogram into
            // the best of the other 3 slots. Pick the cheapest eviction.
            let mut best_evict_slot = 0;
            let mut best_evict_cost = f64::MAX;

            for candidate in 0..4 {
                let candidate_histo = &result.cluster_histograms[slot_histograms[candidate]];

                // Find cheapest merge target among the other 3 slots
                let mut cheapest_remerge = f64::MAX;
                for other in 0..4 {
                    if other == candidate {
                        continue;
                    }
                    let other_histo = &result.cluster_histograms[slot_histograms[other]];
                    let combined = other_histo.combined(candidate_histo);
                    let combined_cost = combined.estimate_encoding_cost();
                    let remerge_cost = combined_cost - slot_costs[other];
                    if remerge_cost < cheapest_remerge {
                        cheapest_remerge = remerge_cost;
                    }
                }

                if cheapest_remerge < best_evict_cost {
                    best_evict_cost = cheapest_remerge;
                    best_evict_slot = candidate;
                }
            }

            best_evict_slot
        }

        SlotReplacement::OldestSlot => {
            // Replace the slot that was assigned/updated longest ago
            (0..4)
                .min_by_key(|&s| slot_age.get(s).copied().unwrap_or(0))
                .unwrap_or(0)
        }

        SlotReplacement::HighestSlotCost => {
            // Replace the slot with the highest current encoding cost
            (0..4)
                .max_by(|&a, &b| slot_costs[a].partial_cmp(&slot_costs[b]).unwrap_or(core::cmp::Ordering::Equal))
                .unwrap_or(0)
        }

        SlotReplacement::Refined => {
            // Refined uses RoundRobin for the greedy pass; refinement happens after.
            (result.slot_ids.last().copied().unwrap_or(0) + 1) % 4
        }
    }
}

/// Iterative 1-opt refinement of cluster assignments.
///
/// For each non-empty context, tries moving it to every other cluster.
/// If moving reduces total encoding cost, applies the move. Repeats
/// until a full pass produces no improvement.
///
/// The cluster histograms are rebuilt from the original per-context
/// histograms after each move, so costs are always exact.
fn refine_clustering(
    histograms: &[FrequencyCounter],
    result: &mut ClusterResult,
) {
    // Collect non-empty context indices
    let active: Vec<usize> = (0..histograms.len())
        .filter(|&i| !histograms[i].is_empty_histogram())
        .collect();

    if active.is_empty() || result.num_clusters <= 1 {
        return;
    }

    // Rebuild cluster histograms from scratch (greedy pass accumulates
    // incrementally, which is correct but makes subtract error-prone)
    fn rebuild_cluster_histograms(
        histograms: &[FrequencyCounter],
        context_map: &[usize],
        num_clusters: usize,
        active: &[usize],
    ) -> Vec<FrequencyCounter> {
        let mut clusters = vec![FrequencyCounter::new(); num_clusters];
        for &ctx in active {
            let cluster = context_map[ctx];
            if cluster < num_clusters {
                clusters[cluster].add(&histograms[ctx]);
            }
        }
        clusters
    }

    result.cluster_histograms = rebuild_cluster_histograms(
        histograms,
        &result.context_map,
        result.num_clusters,
        &active,
    );

    // Compute per-cluster costs
    let mut cluster_costs: Vec<f64> = result
        .cluster_histograms
        .iter()
        .map(|h| {
            if h.is_empty_histogram() {
                0.0
            } else {
                h.estimate_encoding_cost()
            }
        })
        .collect();

    let mut total_cost: f64 = cluster_costs.iter().sum();

    // Iterative 1-opt: try moving each context to a better cluster
    const MAX_PASSES: usize = 10;
    for _pass in 0..MAX_PASSES {
        let mut improved = false;

        for &ctx in &active {
            let current_cluster = result.context_map[ctx];
            let ctx_histo = &histograms[ctx];

            // Cost of current cluster WITHOUT this context
            let mut without = result.cluster_histograms[current_cluster].clone();
            without.subtract(ctx_histo);
            let cost_without = if without.is_empty_histogram() {
                0.0 // Cluster becomes empty — saves the whole table
            } else {
                without.estimate_encoding_cost()
            };

            // Savings from removing this context from its current cluster
            let removal_savings = cluster_costs[current_cluster] - cost_without;

            // Try adding to each other cluster
            let mut best_target = current_cluster;
            let mut best_delta = 0.0_f64; // Must improve to move

            for target in 0..result.num_clusters {
                if target == current_cluster {
                    continue;
                }

                let combined = result.cluster_histograms[target].combined(ctx_histo);
                let cost_combined = combined.estimate_encoding_cost();
                let addition_cost = cost_combined - cluster_costs[target];

                // Net delta: cost of adding to target - savings from removing from current
                let delta = addition_cost - removal_savings;
                if delta < best_delta {
                    best_delta = delta;
                    best_target = target;
                }
            }

            if best_target != current_cluster {
                // Apply the move
                result.cluster_histograms[current_cluster].subtract(ctx_histo);
                result.cluster_histograms[best_target].add(ctx_histo);
                result.context_map[ctx] = best_target;

                // Update costs
                cluster_costs[current_cluster] = if result.cluster_histograms[current_cluster].is_empty_histogram() {
                    0.0
                } else {
                    result.cluster_histograms[current_cluster].estimate_encoding_cost()
                };
                cluster_costs[best_target] = result.cluster_histograms[best_target].estimate_encoding_cost();

                total_cost = cluster_costs.iter().sum();
                improved = true;
            }
        }

        if !improved {
            break;
        }
    }

    // Remove empty clusters and compact
    let _ = total_cost; // Used for debugging; suppress warning
    compact_clusters(result, histograms);
}

/// Remove empty clusters and renumber context_map and slot_ids.
fn compact_clusters(result: &mut ClusterResult, histograms: &[FrequencyCounter]) {
    // Find non-empty clusters
    let mut old_to_new = vec![usize::MAX; result.num_clusters];
    let mut new_histograms = Vec::new();
    let mut new_slot_ids = Vec::new();

    for (old_idx, histo) in result.cluster_histograms.iter().enumerate() {
        if !histo.is_empty_histogram() {
            let new_idx = new_histograms.len();
            old_to_new[old_idx] = new_idx;
            new_histograms.push(histo.clone());
            if old_idx < result.slot_ids.len() {
                new_slot_ids.push(result.slot_ids[old_idx]);
            } else {
                new_slot_ids.push(new_idx % 4);
            }
        }
    }

    // If nothing was removed, no compaction needed
    if new_histograms.len() == result.num_clusters {
        return;
    }

    // Remap context_map
    for ctx in 0..result.context_map.len() {
        let old = result.context_map[ctx];
        if old < old_to_new.len() && old_to_new[old] != usize::MAX {
            result.context_map[ctx] = old_to_new[old];
        } else if !histograms[ctx].is_empty_histogram() {
            // Context was in a cluster that got emptied — assign to cluster 0
            result.context_map[ctx] = 0;
        }
    }

    result.cluster_histograms = new_histograms;
    result.slot_ids = new_slot_ids;
    result.num_clusters = result.cluster_histograms.len();
}
