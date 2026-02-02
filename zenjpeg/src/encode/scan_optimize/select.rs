//! Scan selection algorithm for progressive scan optimization.
//!
//! Implements the selection logic from C mozjpeg's `jcmaster.c select_scans()`.
//! Given the estimated sizes of 64 (or 23) candidate scans, determines the
//! optimal Al levels, frequency splits, and DC interleaving.

use super::config::{ScanSearchConfig, ScanSearchResult};

/// Estimated per-scan overhead in bits that the frequency estimator doesn't capture.
///
/// `FrequencyCounter::estimate_encoding_cost()` already includes DHT table data overhead
/// (code lengths + symbol values), but not:
/// - SOS marker: 2 + 2 + 1 + 2*ncomp + 3 = ~10 bytes = 80 bits (single component)
/// - DHT marker envelope: 0xFFC4 + 2-byte length = 4 bytes = 32 bits
/// - Byte alignment padding: ~4 bits average
///
/// We use 150 bits (~19 bytes) to account for these fixed per-scan costs.
/// This is conservative but not so aggressive as to prevent genuinely beneficial
/// SA levels from being selected.
const SCAN_OVERHEAD: usize = 150;

/// Scan selector that processes estimated scan sizes and picks the best configuration.
///
/// Index arithmetic matches C mozjpeg's `jcmaster.c` exactly.
pub(crate) struct ScanSelector {
    config: ScanSearchConfig,
    num_components: u8,

    // Luma scan indices
    luma_freq_split_scan_start: usize,
    num_scans_luma: usize,

    // Chroma scan indices
    num_scans_chroma_dc: usize,
    chroma_freq_split_scan_start: usize,
}

impl ScanSelector {
    /// Create a new selector for the given configuration.
    pub fn new(num_components: u8, config: ScanSearchConfig) -> Self {
        let al_max_luma = config.al_max_luma as usize;
        let al_max_chroma = config.al_max_chroma as usize;
        let num_freq_splits = config.frequency_splits.len();

        // Index calculations matching jcparam.c:775-780
        let num_scans_luma_dc = 1;
        let luma_freq_split_scan_start = num_scans_luma_dc + 3 * al_max_luma + 2;
        let num_scans_luma = luma_freq_split_scan_start + 2 * num_freq_splits + 1;

        let num_scans_chroma_dc = if num_components >= 3 { 3 } else { 0 };
        let chroma_freq_split_scan_start = if num_components >= 3 {
            num_scans_luma + num_scans_chroma_dc + 4 + 6 * al_max_chroma + 2
        } else {
            0
        };

        Self {
            config,
            num_components,
            luma_freq_split_scan_start,
            num_scans_luma,
            num_scans_chroma_dc,
            chroma_freq_split_scan_start,
        }
    }

    /// Process scan sizes and determine the best configuration.
    pub fn select_best(&self, scan_sizes: &[usize]) -> ScanSearchResult {
        let (best_al_luma, best_freq_split_luma) = self.select_luma_params(scan_sizes);

        let (best_al_chroma, best_freq_split_chroma, interleave_chroma_dc) =
            if self.num_components >= 3 {
                self.select_chroma_params(scan_sizes)
            } else {
                (0, 0, false)
            };

        ScanSearchResult {
            best_al_luma,
            best_al_chroma,
            best_freq_split_luma,
            best_freq_split_chroma,
            interleave_chroma_dc,
        }
    }

    /// Select best Al and frequency split for luma.
    /// Matches C mozjpeg's jcmaster.c:786-830.
    fn select_luma_params(&self, scan_sizes: &[usize]) -> (u8, usize) {
        let al_max = self.config.al_max_luma as usize;

        let mut best_al = 0u8;
        let mut best_cost = usize::MAX;

        for al in 0..=al_max {
            let cost = if al == 0 {
                // Cost = base 1-8 + base 9-63 at Al=0
                scan_sizes.get(1).copied().unwrap_or(usize::MAX)
                    + scan_sizes.get(2).copied().unwrap_or(0)
            } else {
                // Bands at this Al level
                let band1_idx = 3 * al + 1; // 1-8 at Al
                let band2_idx = 3 * al + 2; // 9-63 at Al
                let mut c = scan_sizes.get(band1_idx).copied().unwrap_or(0)
                    + scan_sizes.get(band2_idx).copied().unwrap_or(0);

                // Add all refinement costs from this Al down to Al=0
                for i in 0..al {
                    let refine_idx = 3 + 3 * i;
                    c += scan_sizes.get(refine_idx).copied().unwrap_or(0);
                }
                // Each SA level adds a refinement scan with overhead
                // (SOS header ~14 bytes + DHT table ~200 bytes + padding)
                c += al * SCAN_OVERHEAD;
                c
            };

            if al == 0 || cost < best_cost {
                best_cost = cost;
                best_al = al as u8;
            } else {
                // Early termination: if this Al is worse, skip remaining
                break;
            }
        }

        // Find best frequency split
        let full_1_63_idx = self.luma_freq_split_scan_start;
        let mut best_freq_split = 0usize; // 0 = no split
        let mut best_freq_cost = scan_sizes.get(full_1_63_idx).copied().unwrap_or(usize::MAX);

        let freq_start = full_1_63_idx + 1;
        for (i, _split) in self.config.frequency_splits.iter().enumerate() {
            let idx = freq_start + 2 * i;
            // Add per-scan overhead for the extra scan created by splitting
            let cost = scan_sizes.get(idx).copied().unwrap_or(0)
                + scan_sizes.get(idx + 1).copied().unwrap_or(0)
                + SCAN_OVERHEAD;

            if cost < best_freq_cost {
                best_freq_cost = cost;
                best_freq_split = i + 1; // 1-indexed
            }

            // Early termination heuristics from C mozjpeg (jcmaster.c:823-829)
            if i == 2 && best_freq_split == 0 {
                break;
            }
            if i == 3 && best_freq_split != 2 {
                break;
            }
            if i == 4 && best_freq_split != 4 {
                break;
            }
        }

        // Compare freq-split-at-Al=0 against SA cost (Bug 4 from mozjpeg-rs 01fddb9).
        // Frequency splits are measured at Al=0 only. If the best freq split at Al=0
        // beats the best SA configuration, override to Al=0 with the split.
        if best_al > 0 && best_freq_cost < best_cost {
            best_al = 0;
            best_cost = best_freq_cost;
        }
        let _ = best_cost; // suppress unused warning

        (best_al, best_freq_split)
    }

    /// Select best Al, frequency split, and DC interleaving for chroma.
    /// Matches C mozjpeg's jcmaster.c:832-896.
    fn select_chroma_params(&self, scan_sizes: &[usize]) -> (u8, usize, bool) {
        let base = self.num_scans_luma;
        let al_max = self.config.al_max_chroma as usize;

        // DC interleaving decision
        let combined_dc = scan_sizes.get(base).copied().unwrap_or(0);
        let separate_dc = scan_sizes.get(base + 1).copied().unwrap_or(0)
            + scan_sizes.get(base + 2).copied().unwrap_or(0);
        let interleave_chroma_dc = combined_dc <= separate_dc;

        let dc_offset = self.num_scans_chroma_dc; // 3

        let mut best_al = 0u8;
        let mut best_cost = usize::MAX;

        for al in 0..=al_max {
            let cost = if al == 0 {
                let cb_base = base + dc_offset; // 26
                let cr_base = base + dc_offset + 2; // 28
                scan_sizes.get(cb_base).copied().unwrap_or(0)
                    + scan_sizes.get(cb_base + 1).copied().unwrap_or(0)
                    + scan_sizes.get(cr_base).copied().unwrap_or(0)
                    + scan_sizes.get(cr_base + 1).copied().unwrap_or(0)
            } else {
                let band_base = base + dc_offset + 4 + 6 * (al - 1) + 2;
                let mut c = scan_sizes.get(band_base).copied().unwrap_or(0)
                    + scan_sizes.get(band_base + 1).copied().unwrap_or(0)
                    + scan_sizes.get(band_base + 2).copied().unwrap_or(0)
                    + scan_sizes.get(band_base + 3).copied().unwrap_or(0);

                // Add refinement costs
                for i in 0..al {
                    let refine_base = base + dc_offset + 4 + 6 * i;
                    c += scan_sizes.get(refine_base).copied().unwrap_or(0);
                    c += scan_sizes.get(refine_base + 1).copied().unwrap_or(0);
                }
                // Each chroma SA level adds 2 refinement scans (Cb + Cr)
                c += al * 2 * SCAN_OVERHEAD;
                c
            };

            if al == 0 || cost < best_cost {
                best_cost = cost;
                best_al = al as u8;
            } else {
                break;
            }
        }

        // Frequency splits for chroma
        let chroma_full_base = base + dc_offset + 4 + 6 * al_max;
        let mut best_freq_split = 0usize;
        let mut best_freq_cost = scan_sizes.get(chroma_full_base).copied().unwrap_or(0)
            + scan_sizes.get(chroma_full_base + 1).copied().unwrap_or(0);

        let freq_base = self.chroma_freq_split_scan_start;
        for (i, _split) in self.config.frequency_splits.iter().enumerate() {
            let idx = freq_base + 4 * i;
            // Splitting adds 2 extra scans (from 2 full-range to 4 split-range)
            let cost = scan_sizes.get(idx).copied().unwrap_or(0)
                + scan_sizes.get(idx + 1).copied().unwrap_or(0)
                + scan_sizes.get(idx + 2).copied().unwrap_or(0)
                + scan_sizes.get(idx + 3).copied().unwrap_or(0)
                + 2 * SCAN_OVERHEAD;

            if cost < best_freq_cost {
                best_freq_cost = cost;
                best_freq_split = i + 1;
            }

            if i == 2 && best_freq_split == 0 {
                break;
            }
        }

        // Compare freq-split-at-Al=0 against SA cost (same logic as luma).
        if best_al > 0 && best_freq_cost < best_cost {
            best_al = 0;
            best_cost = best_freq_cost;
        }
        let _ = best_cost; // suppress unused warning

        (best_al, best_freq_split, interleave_chroma_dc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_equal_sizes_picks_al0_no_split() {
        let config = ScanSearchConfig::default();
        let selector = ScanSelector::new(3, config);

        // All scans same size: Al=0 should win (fewest scans, no refinement overhead)
        let scan_sizes = vec![100usize; 64];
        let result = selector.select_best(&scan_sizes);

        assert_eq!(result.best_al_luma, 0, "Equal sizes should pick Al=0");
        // With equal costs, full 1-63 should win over splits (fewer scans)
        // Actually, full has cost 100 and each split pair has cost 200, so no split wins.
        assert_eq!(result.best_freq_split_luma, 0, "Equal sizes: no freq split");
    }

    #[test]
    fn test_prefers_lower_cost_al() {
        let config = ScanSearchConfig::default();
        let selector = ScanSelector::new(3, config);

        // Use realistic bit-scale sizes (frequency estimator outputs bits)
        let mut scan_sizes = vec![100_000usize; 64];

        // Make Al=0 baseline expensive
        scan_sizes[1] = 50_000;
        scan_sizes[2] = 50_000;

        // Make Al=1 cheaper: band cost + refinement
        scan_sizes[4] = 30_000;
        scan_sizes[5] = 30_000;
        scan_sizes[3] = 10_000; // Refinement

        // Al=2 even cheaper (must beat Al=0 even with 2*SCAN_OVERHEAD penalty)
        // Al=2 cost = 15000 + 15000 + 5000 + 10000 + 2*2400 = 49800
        // Al=0 cost = 50000 + 50000 = 100000
        scan_sizes[7] = 15_000;
        scan_sizes[8] = 15_000;
        scan_sizes[6] = 5_000;

        // Al=3 worse (defaults to 100_000 each)

        let result = selector.select_best(&scan_sizes);
        assert_eq!(result.best_al_luma, 2, "Should pick Al=2 (cheapest)");
    }

    #[test]
    fn test_chroma_dc_interleaving() {
        let config = ScanSearchConfig::default();
        let selector = ScanSelector::new(3, config);

        let mut scan_sizes = vec![100usize; 64];

        // num_scans_luma = 23 for default config
        // scan_sizes[23] = combined DC, [24] = Cb DC, [25] = Cr DC

        // Interleaved cheaper
        scan_sizes[23] = 50;
        scan_sizes[24] = 30;
        scan_sizes[25] = 30;
        let result = selector.select_best(&scan_sizes);
        assert!(
            result.interleave_chroma_dc,
            "Should interleave when combined DC is cheaper"
        );

        // Separate cheaper
        scan_sizes[23] = 100;
        scan_sizes[24] = 20;
        scan_sizes[25] = 20;
        let result = selector.select_best(&scan_sizes);
        assert!(
            !result.interleave_chroma_dc,
            "Should not interleave when separate DC is cheaper"
        );
    }

    #[test]
    fn test_frequency_split_selection() {
        let config = ScanSearchConfig::default();
        let selector = ScanSelector::new(3, config);

        // Use realistic bit-scale sizes
        let mut scan_sizes = vec![100_000usize; 64];

        // Full 1-63 (scan 12) is expensive
        scan_sizes[12] = 50_000;

        // Split at freq=5 (index 2, scans 17-18) is much cheaper
        // Split cost = 5000 + 5000 + 2400 = 12400 < 50000
        scan_sizes[17] = 5_000;
        scan_sizes[18] = 5_000;

        let result = selector.select_best(&scan_sizes);
        assert!(
            result.best_freq_split_luma > 0,
            "Should pick a frequency split when cheaper"
        );
    }

    #[test]
    fn test_build_final_scans_valid() {
        let config = ScanSearchConfig::default();
        let result = ScanSearchResult {
            best_al_luma: 1,
            best_al_chroma: 0,
            best_freq_split_luma: 0,
            best_freq_split_chroma: 0,
            interleave_chroma_dc: true,
        };

        let scans = result.build_final_scans(3, &config);

        // Verify basic structure
        assert!(!scans.is_empty());

        // First scan should be DC
        assert_eq!(scans[0].ss, 0);
        assert_eq!(scans[0].se, 0);

        // All components should be in range
        for scan in &scans {
            for &c in &scan.components {
                assert!(c < 3, "Component {} out of range", c);
            }
            assert!(scan.se <= 63, "se {} out of range", scan.se);
            assert!(scan.ss <= scan.se || (scan.ss == 0 && scan.se == 0));
        }
    }

    #[test]
    fn test_build_final_scans_with_sa_has_ac_refinement() {
        let config = ScanSearchConfig::default();
        let result = ScanSearchResult {
            best_al_luma: 2,
            best_al_chroma: 1,
            best_freq_split_luma: 0,
            best_freq_split_chroma: 0,
            interleave_chroma_dc: true,
        };

        let scans = result.build_final_scans(3, &config);

        // DC scan should always be full precision (no DC SA)
        assert_eq!(scans[0].al, 0, "DC scan should have Al=0 (no DC SA)");
        assert_eq!(scans[0].ah, 0);

        // No DC refinement scan
        let dc_refines: Vec<_> = scans
            .iter()
            .filter(|s| s.ss == 0 && s.se == 0 && s.ah > 0)
            .collect();
        assert!(dc_refines.is_empty(), "Should have no DC refinement scans");

        // Should have luma AC refinement scans
        let luma_refines: Vec<_> = scans
            .iter()
            .filter(|s| s.components == vec![0] && s.ah > 0 && s.ss > 0)
            .collect();
        assert_eq!(
            luma_refines.len(),
            2,
            "Al=2 needs 2 luma AC refinement scans"
        );

        // Should have chroma AC refinement scans
        let chroma_refines: Vec<_> = scans
            .iter()
            .filter(|s| {
                (s.components == vec![1] || s.components == vec![2]) && s.ah > 0 && s.ss > 0
            })
            .collect();
        assert_eq!(
            chroma_refines.len(),
            2,
            "Al_c=1 needs 1 refinement per chroma = 2 total"
        );
    }

    #[test]
    fn test_grayscale_selection() {
        let config = ScanSearchConfig::default();
        let selector = ScanSelector::new(1, config.clone());

        let scan_sizes = vec![100usize; 23];
        let result = selector.select_best(&scan_sizes);

        assert_eq!(result.best_al_chroma, 0);
        assert!(!result.interleave_chroma_dc);

        // Build final scans for grayscale
        let scans = result.build_final_scans(1, &config);
        assert!(!scans.is_empty());

        // All scans should only have component 0
        for scan in &scans {
            for &c in &scan.components {
                assert_eq!(c, 0, "Grayscale should only have component 0");
            }
        }
    }
}
