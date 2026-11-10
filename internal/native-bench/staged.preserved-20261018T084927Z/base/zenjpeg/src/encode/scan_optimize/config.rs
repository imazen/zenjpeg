//! Configuration types for progressive scan optimization.
//!
//! Ported from mozjpeg's scan optimization algorithm (jcmaster.c).

use super::super::config::ProgressiveScan;

/// Configuration for the scan optimization search.
///
/// These defaults match C mozjpeg's `optimize_scans` behavior.
#[derive(Debug, Clone)]
pub(crate) struct ScanSearchConfig {
    /// Maximum successive approximation level for luma (default: 3).
    pub al_max_luma: u8,
    /// Maximum successive approximation level for chroma (default: 2).
    pub al_max_chroma: u8,
    /// Frequency split points to test (default: [2, 8, 5, 12, 18]).
    pub frequency_splits: [u8; 5],
    /// DC scan optimization mode (0=interleaved, 1=separate, 2=luma+chroma pair).
    pub dc_scan_opt_mode: u8,
}

impl Default for ScanSearchConfig {
    fn default() -> Self {
        Self {
            al_max_luma: 3,
            al_max_chroma: 2,
            frequency_splits: [2, 8, 5, 12, 18],
            // Use separate DC scans (mode 1) because zenjpeg's progressive
            // tokenizer stores blocks per-component in raster order, not
            // MCU-interleaved. Interleaved DC (mode 0) would require MCU-aware
            // iteration to handle subsampled images correctly.
            dc_scan_opt_mode: 1,
        }
    }
}

/// Results from scan optimization search.
///
/// Contains the selected parameters that produce the smallest progressive JPEG.
#[derive(Debug, Clone)]
pub(crate) struct ScanSearchResult {
    /// Best successive approximation level for luma.
    pub best_al_luma: u8,
    /// Best successive approximation level for chroma.
    pub best_al_chroma: u8,
    /// Best frequency split index for luma (0 = no split, 1-5 = index into frequency_splits).
    pub best_freq_split_luma: usize,
    /// Best frequency split index for chroma.
    pub best_freq_split_chroma: usize,
    /// Whether to interleave chroma DC scans.
    pub interleave_chroma_dc: bool,
}

impl ScanSearchResult {
    /// Build the final optimized scan script from the search results.
    ///
    /// Produces a `Vec<ProgressiveScan>` compatible with zenjpeg's progressive encoder.
    ///
    /// DC scans always use full precision (no DC successive approximation).
    /// DC SA is theoretically valid but adds complexity for negligible savings.
    pub fn build_final_scans(
        &self,
        num_components: u8,
        config: &ScanSearchConfig,
    ) -> Vec<ProgressiveScan> {
        let mut scans = Vec::new();

        // DC scan — always full precision
        if config.dc_scan_opt_mode == 0 {
            // Interleaved DC for all components
            let components: Vec<u8> = (0..num_components).collect();
            scans.push(ProgressiveScan {
                components,
                ss: 0,
                se: 0,
                ah: 0,
                al: 0,
            });
        } else {
            // Separate DC for luma
            scans.push(ProgressiveScan {
                components: vec![0],
                ss: 0,
                se: 0,
                ah: 0,
                al: 0,
            });
        }

        // Luma AC scans.
        // Frequency splits were measured at Al=0 and only apply there.
        // When Al>0, always use full 1-63 range for the initial scan.
        let al = self.best_al_luma;
        if al == 0 && self.best_freq_split_luma > 0 {
            let split = config.frequency_splits[self.best_freq_split_luma - 1];
            scans.push(ProgressiveScan {
                components: vec![0],
                ss: 1,
                se: split,
                ah: 0,
                al: 0,
            });
            scans.push(ProgressiveScan {
                components: vec![0],
                ss: split + 1,
                se: 63,
                ah: 0,
                al: 0,
            });
        } else {
            scans.push(ProgressiveScan {
                components: vec![0],
                ss: 1,
                se: 63,
                ah: 0,
                al,
            });
        }

        // Luma AC refinement scans (from al down to 0)
        for refine_al in (0..al).rev() {
            scans.push(ProgressiveScan {
                components: vec![0],
                ss: 1,
                se: 63,
                ah: refine_al + 1,
                al: refine_al,
            });
        }

        if num_components >= 3 {
            // Chroma DC (only needed when dc_scan_opt_mode != 0)
            if config.dc_scan_opt_mode != 0 {
                if self.interleave_chroma_dc {
                    scans.push(ProgressiveScan {
                        components: vec![1, 2],
                        ss: 0,
                        se: 0,
                        ah: 0,
                        al: 0,
                    });
                } else {
                    for c in 1..=2u8 {
                        scans.push(ProgressiveScan {
                            components: vec![c],
                            ss: 0,
                            se: 0,
                            ah: 0,
                            al: 0,
                        });
                    }
                }
            }

            // Chroma AC scans.
            // Same rule: frequency splits only apply at Al=0.
            let al_c = self.best_al_chroma;
            for comp in 1..=2u8 {
                if al_c == 0 && self.best_freq_split_chroma > 0 {
                    let split = config.frequency_splits[self.best_freq_split_chroma - 1];
                    scans.push(ProgressiveScan {
                        components: vec![comp],
                        ss: 1,
                        se: split,
                        ah: 0,
                        al: 0,
                    });
                    scans.push(ProgressiveScan {
                        components: vec![comp],
                        ss: split + 1,
                        se: 63,
                        ah: 0,
                        al: 0,
                    });
                } else {
                    scans.push(ProgressiveScan {
                        components: vec![comp],
                        ss: 1,
                        se: 63,
                        ah: 0,
                        al: al_c,
                    });
                }
            }

            // Chroma AC refinement
            for refine_al in (0..al_c).rev() {
                for comp in 1..=2u8 {
                    scans.push(ProgressiveScan {
                        components: vec![comp],
                        ss: 1,
                        se: 63,
                        ah: refine_al + 1,
                        al: refine_al,
                    });
                }
            }
        }

        scans
    }
}
