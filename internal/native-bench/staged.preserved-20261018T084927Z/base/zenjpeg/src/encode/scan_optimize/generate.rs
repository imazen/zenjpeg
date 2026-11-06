//! Candidate scan generation for progressive scan optimization.
//!
//! Generates 64 candidate scans for YCbCr and 23 for grayscale,
//! matching C mozjpeg's `jpeg_search_progression()` layout exactly.

use super::config::ScanSearchConfig;

/// A lightweight trial scan descriptor.
///
/// Used only during the search phase. The final optimized scans are
/// built as `ProgressiveScan` by `ScanSearchResult::build_final_scans()`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TrialScan {
    /// Component index (0=Y, 1=Cb, 2=Cr).
    pub component: u8,
    /// Spectral selection start (0=DC, 1-63=AC).
    pub ss: u8,
    /// Spectral selection end (0-63).
    pub se: u8,
    /// Successive approximation high bit (previous pass).
    pub ah: u8,
    /// Successive approximation low bit (current pass).
    pub al: u8,
    /// Number of components in this scan (>1 for interleaved DC).
    pub comps_in_scan: u8,
}

impl TrialScan {
    fn dc_all(num_components: u8) -> Self {
        Self {
            component: 0,
            ss: 0,
            se: 0,
            ah: 0,
            al: 0,
            comps_in_scan: num_components,
        }
    }

    fn dc_single(component: u8) -> Self {
        Self {
            component,
            ss: 0,
            se: 0,
            ah: 0,
            al: 0,
            comps_in_scan: 1,
        }
    }

    fn dc_pair(c1: u8, _c2: u8) -> Self {
        Self {
            component: c1,
            ss: 0,
            se: 0,
            ah: 0,
            al: 0,
            comps_in_scan: 2,
        }
    }

    fn ac(component: u8, ss: u8, se: u8, ah: u8, al: u8) -> Self {
        Self {
            component,
            ss,
            se,
            ah,
            al,
            comps_in_scan: 1,
        }
    }

    /// Returns true if this is a DC scan (ss=0, se=0).
    pub fn is_dc(&self) -> bool {
        self.ss == 0 && self.se == 0
    }
}

/// Generate the full set of candidate scans for the search.
///
/// Layout for YCbCr (64 scans with default config):
///
/// **Luma (23 scans):**
/// - 0: DC scan (all components or just luma depending on dc_scan_opt_mode)
/// - 1: Y AC 1-8 at Al=0
/// - 2: Y AC 9-63 at Al=0
/// - 3..12: SA test scans (3 per Al level × 3 levels)
/// - 12: Y full AC 1-63 at Al=0
/// - 13..22: Frequency split tests (5 pairs)
///
/// **Chroma (41 scans):**
/// - 23: Cb+Cr combined DC
/// - 24: Cb DC alone
/// - 25: Cr DC alone
/// - 26-29: Cb/Cr base AC (1-8, 9-63 each)
/// - 30-41: SA test scans (6 per Al level × 2 levels)
/// - 42: Cb full 1-63 at Al=0
/// - 43: Cr full 1-63 at Al=0
/// - 44-63: Chroma frequency splits (5 pairs × 2 components)
pub(crate) fn generate_search_scans(
    num_components: u8,
    config: &ScanSearchConfig,
) -> Vec<TrialScan> {
    if num_components == 1 {
        return generate_grayscale_search_scans(config);
    }

    let mut scans = Vec::with_capacity(64);

    // === LUMA SCANS (23 scans) ===

    // Scan 0: DC scan
    if config.dc_scan_opt_mode == 0 {
        scans.push(TrialScan::dc_all(num_components));
    } else {
        scans.push(TrialScan::dc_single(0));
    }

    // Scans 1-2: Base AC scans at Al=0
    scans.push(TrialScan::ac(0, 1, 8, 0, 0));
    scans.push(TrialScan::ac(0, 9, 63, 0, 0));

    // Scans 3-11: SA test scans (3 per Al level)
    for al in 0..config.al_max_luma {
        scans.push(TrialScan::ac(0, 1, 63, al + 1, al)); // Refinement
        scans.push(TrialScan::ac(0, 1, 8, 0, al + 1)); // 1-8 at Al+1
        scans.push(TrialScan::ac(0, 9, 63, 0, al + 1)); // 9-63 at Al+1
    }

    // Scan 12: Full luma AC 1-63 at Al=0
    scans.push(TrialScan::ac(0, 1, 63, 0, 0));

    // Scans 13-22: Frequency split tests
    for &split in &config.frequency_splits {
        scans.push(TrialScan::ac(0, 1, split, 0, 0));
        scans.push(TrialScan::ac(0, split + 1, 63, 0, 0));
    }

    // === CHROMA SCANS (41 scans) ===
    if num_components >= 3 {
        // DC variants
        scans.push(TrialScan::dc_pair(1, 2)); // Combined Cb+Cr DC
        scans.push(TrialScan::dc_single(1)); // Cb DC alone
        scans.push(TrialScan::dc_single(2)); // Cr DC alone

        // Base AC scans at Al=0
        scans.push(TrialScan::ac(1, 1, 8, 0, 0));
        scans.push(TrialScan::ac(1, 9, 63, 0, 0));
        scans.push(TrialScan::ac(2, 1, 8, 0, 0));
        scans.push(TrialScan::ac(2, 9, 63, 0, 0));

        // SA test scans (6 per Al level)
        for al in 0..config.al_max_chroma {
            scans.push(TrialScan::ac(1, 1, 63, al + 1, al)); // Cb refine
            scans.push(TrialScan::ac(2, 1, 63, al + 1, al)); // Cr refine
            scans.push(TrialScan::ac(1, 1, 8, 0, al + 1)); // Cb 1-8 at Al+1
            scans.push(TrialScan::ac(1, 9, 63, 0, al + 1)); // Cb 9-63 at Al+1
            scans.push(TrialScan::ac(2, 1, 8, 0, al + 1)); // Cr 1-8 at Al+1
            scans.push(TrialScan::ac(2, 9, 63, 0, al + 1)); // Cr 9-63 at Al+1
        }

        // Full chroma AC at Al=0
        scans.push(TrialScan::ac(1, 1, 63, 0, 0));
        scans.push(TrialScan::ac(2, 1, 63, 0, 0));

        // Frequency split tests (5 pairs × 2 components)
        for &split in &config.frequency_splits {
            scans.push(TrialScan::ac(1, 1, split, 0, 0));
            scans.push(TrialScan::ac(1, split + 1, 63, 0, 0));
            scans.push(TrialScan::ac(2, 1, split, 0, 0));
            scans.push(TrialScan::ac(2, split + 1, 63, 0, 0));
        }
    }

    scans
}

/// Generate search scans for grayscale (23 scans).
fn generate_grayscale_search_scans(config: &ScanSearchConfig) -> Vec<TrialScan> {
    let mut scans = Vec::with_capacity(23);

    // Scan 0: DC
    scans.push(TrialScan::dc_single(0));

    // Scans 1-2: Base AC at Al=0
    scans.push(TrialScan::ac(0, 1, 8, 0, 0));
    scans.push(TrialScan::ac(0, 9, 63, 0, 0));

    // SA test scans (3 per Al level)
    for al in 0..config.al_max_luma {
        scans.push(TrialScan::ac(0, 1, 63, al + 1, al));
        scans.push(TrialScan::ac(0, 1, 8, 0, al + 1));
        scans.push(TrialScan::ac(0, 9, 63, 0, al + 1));
    }

    // Full AC 1-63 at Al=0
    scans.push(TrialScan::ac(0, 1, 63, 0, 0));

    // Frequency split tests
    for &split in &config.frequency_splits {
        scans.push(TrialScan::ac(0, 1, split, 0, 0));
        scans.push(TrialScan::ac(0, split + 1, 63, 0, 0));
    }

    scans
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_ycbcr_64_scans() {
        let config = ScanSearchConfig::default();
        let scans = generate_search_scans(3, &config);
        assert_eq!(scans.len(), 64, "YCbCr should generate 64 scans");
    }

    #[test]
    fn test_generate_grayscale_23_scans() {
        let config = ScanSearchConfig::default();
        let scans = generate_search_scans(1, &config);
        assert_eq!(scans.len(), 23, "Grayscale should generate 23 scans");
    }

    #[test]
    fn test_luma_scan_layout() {
        let config = ScanSearchConfig::default();
        let scans = generate_search_scans(3, &config);

        // Scan 0: DC luma only (dc_scan_opt_mode=1 uses separate DC scans)
        assert!(scans[0].is_dc());
        assert_eq!(scans[0].comps_in_scan, 1);

        // Scans 1-2: Y base AC at Al=0
        assert_eq!((scans[1].ss, scans[1].se), (1, 8));
        assert_eq!((scans[1].ah, scans[1].al), (0, 0));
        assert_eq!((scans[2].ss, scans[2].se), (9, 63));
        assert_eq!((scans[2].ah, scans[2].al), (0, 0));

        // Scan 3: refinement 1-63 (Ah=1, Al=0)
        assert_eq!((scans[3].ss, scans[3].se), (1, 63));
        assert_eq!((scans[3].ah, scans[3].al), (1, 0));

        // Scan 4: 1-8 at Al=1
        assert_eq!((scans[4].ss, scans[4].se), (1, 8));
        assert_eq!((scans[4].ah, scans[4].al), (0, 1));

        // Scan 12: Full luma AC 1-63 at Al=0
        assert_eq!((scans[12].ss, scans[12].se), (1, 63));
        assert_eq!((scans[12].ah, scans[12].al), (0, 0));
    }

    #[test]
    fn test_chroma_scan_layout() {
        let config = ScanSearchConfig::default();
        let scans = generate_search_scans(3, &config);

        // Scan 23: Cb+Cr combined DC
        assert!(scans[23].is_dc());
        assert_eq!(scans[23].comps_in_scan, 2);

        // Scan 24: Cb DC alone
        assert!(scans[24].is_dc());
        assert_eq!(scans[24].component, 1);
        assert_eq!(scans[24].comps_in_scan, 1);

        // Scan 25: Cr DC alone
        assert!(scans[25].is_dc());
        assert_eq!(scans[25].component, 2);
        assert_eq!(scans[25].comps_in_scan, 1);

        // Scan 26-27: Cb base AC
        assert_eq!(scans[26].component, 1);
        assert_eq!((scans[26].ss, scans[26].se), (1, 8));
        assert_eq!(scans[27].component, 1);
        assert_eq!((scans[27].ss, scans[27].se), (9, 63));

        // Scan 28-29: Cr base AC
        assert_eq!(scans[28].component, 2);
        assert_eq!(scans[29].component, 2);
    }

    #[test]
    fn test_all_luma_scans_are_component_0() {
        let config = ScanSearchConfig::default();
        let scans = generate_search_scans(3, &config);

        // Luma scans are 0..23 (excluding scan 0 DC which is multi-component)
        for (i, scan) in scans[1..23].iter().enumerate() {
            assert_eq!(
                scan.component,
                0,
                "Luma scan {} (index {}) should be component 0",
                i + 1,
                i + 1
            );
        }
    }
}
