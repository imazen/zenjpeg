//! Progressive scan optimization (mozjpeg-style `optimize_scans`).
//!
//! Tries 64 candidate progressive scan configurations and picks the smallest.
//! This is a **lossless** optimization: decoded pixels are identical, but the
//! progressive scan structure is chosen to minimize file size.
//!
//! Expected savings: 1-3% smaller progressive JPEGs.

mod config;
mod estimate;
mod generate;
mod select;

pub(crate) use config::ScanSearchConfig;

use super::config::ProgressiveScan;
use crate::error::Result;
use crate::foundation::consts::DCT_BLOCK_SIZE;

/// Maximum number of candidates to trial-encode.
///
/// Each candidate requires a full progressive encode pass (tokenize + Huffman
/// optimize + replay). We use the frequency estimator to pre-filter down to
/// this count: the default baseline + the better of the two search winners.
const MAX_TRIAL_ENCODES: usize = 2;

/// Generate candidate progressive scan scripts for trial encoding.
///
/// Uses a shared histogram cache to avoid redundant block-scanning passes
/// across both search phases. Each unique `(component, ss, se, ah, al)`
/// combination is scanned exactly once.
///
/// Strategy:
/// 1. Pre-generate all mixed-SA variants and the Phase 2 trial scan set.
/// 2. Warm the histogram cache with all unique scan keys from both phases.
/// 3. Use cached estimates for Phase 1 (pick best mixed-SA) and Phase 2
///    (pick best uniform-al via mozjpeg search).
/// 4. Compare the two winners by cached estimate and keep only the better one.
/// 5. Always include the default jpegli script as a safety baseline.
/// 6. Return up to `MAX_TRIAL_ENCODES` scripts for trial encoding.
///
/// # Returns
/// Up to `MAX_TRIAL_ENCODES` unique candidate scripts, always including the default.
pub(crate) fn generate_candidate_scripts(
    y_blocks: &[[i16; DCT_BLOCK_SIZE]],
    cb_blocks: &[[i16; DCT_BLOCK_SIZE]],
    cr_blocks: &[[i16; DCT_BLOCK_SIZE]],
    num_components: u8,
) -> Result<Vec<Vec<ProgressiveScan>>> {
    let config = ScanSearchConfig::default();

    // === Pre-generate all candidate scripts for cache warming ===
    let split_points: &[u8] = &config.frequency_splits;
    let al_levels: &[u8] = &[1, 2, 3];

    let mut mixed_sa_scripts: Vec<Vec<ProgressiveScan>> =
        Vec::with_capacity(split_points.len() * al_levels.len());
    for &split in split_points {
        for &al in al_levels {
            let al_c = al.min(config.al_max_chroma);
            mixed_sa_scripts.push(mixed_sa_split_progressive_scans(
                num_components,
                split,
                al,
                al_c,
            ));
        }
    }

    let default_script = default_jpegli_progressive_scans(num_components);

    // === Warm cache: scan blocks once per unique (component, ss, se, ah, al) ===
    let trial_scans = generate::generate_search_scans(num_components, &config);
    let mut cache = estimate::ScanHistogramCache::warm(
        &trial_scans,
        &mixed_sa_scripts,
        y_blocks,
        cb_blocks,
        cr_blocks,
        num_components,
    );

    // === Phase 1: Pick best mixed-SA variant by cached estimate ===
    let mut best_mixed_sa: Option<(Vec<ProgressiveScan>, usize)> = None;
    for script in mixed_sa_scripts {
        let est = cache.estimate_script_cost_cached(&script);
        if best_mixed_sa.as_ref().is_none_or(|(_, best)| est < *best) {
            best_mixed_sa = Some((script, est));
        }
    }

    // === Phase 2: mozjpeg-style 64-candidate search (uniform al) ===
    let scan_sizes = cache.estimate_all_scan_sizes_cached(&trial_scans);
    let selector = select::ScanSelector::new(num_components, config.clone());
    let search_result = selector.select_best(&scan_sizes);
    let optimizer_script = search_result.build_final_scans(num_components, &config);

    // === Phase 3: Pick the better of the two search winners ===
    // The estimator is accurate for relative ranking. Compare both winners
    // and only trial-encode the better one alongside the default baseline.
    let mut candidates: Vec<Vec<ProgressiveScan>> = Vec::with_capacity(MAX_TRIAL_ENCODES);
    candidates.push(default_script);

    // Determine the single best alternative to trial-encode
    let optimizer_est = cache.estimate_script_cost_cached(&optimizer_script);
    let best_alternative = match best_mixed_sa {
        Some((mixed_script, mixed_est)) => {
            if scripts_equivalent(&mixed_script, &optimizer_script) {
                // Same script — just use the optimizer's pick
                Some(optimizer_script)
            } else if mixed_est < optimizer_est {
                // Mixed-SA wins the estimate comparison
                Some(mixed_script)
            } else {
                Some(optimizer_script)
            }
        }
        None => Some(optimizer_script),
    };

    if let Some(alt) = best_alternative
        && !scripts_equivalent(&alt, &candidates[0])
    {
        candidates.push(alt);
    }

    Ok(candidates)
}

/// Check if two scan scripts are structurally equivalent.
fn scripts_equivalent(a: &[ProgressiveScan], b: &[ProgressiveScan]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(sa, sb)| {
        sa.components == sb.components
            && sa.ss == sb.ss
            && sa.se == sb.se
            && sa.ah == sb.ah
            && sa.al == sb.al
    })
}

/// Mixed SA progressive script with a configurable frequency split point.
///
/// Generalizes the default jpegli script: instead of always splitting at
/// frequency 2, splits at an arbitrary point:
/// - AC 1-split at al=0 (low frequency, full precision)
/// - AC (split+1)-63 at the given al level with refinement passes
///
/// This is the key candidate category for the estimator to search within.
/// The default jpegli script is equivalent to split=2, al_luma=2, al_chroma=2.
fn mixed_sa_split_progressive_scans(
    num_components: u8,
    split: u8,
    al_luma: u8,
    al_chroma: u8,
) -> Vec<ProgressiveScan> {
    let nc = num_components as usize;
    let mut scans = Vec::with_capacity(nc * 6);

    // Separate DC scans
    for c in 0..nc {
        scans.push(ProgressiveScan {
            components: vec![c as u8],
            ss: 0,
            se: 0,
            ah: 0,
            al: 0,
        });
    }

    // AC 1-split: full precision (same for all components)
    for c in 0..nc {
        scans.push(ProgressiveScan {
            components: vec![c as u8],
            ss: 1,
            se: split,
            ah: 0,
            al: 0,
        });
    }

    // AC (split+1)-63 first pass at respective al level
    if split < 63 {
        for c in 0..nc {
            let al = if c == 0 { al_luma } else { al_chroma };
            scans.push(ProgressiveScan {
                components: vec![c as u8],
                ss: split + 1,
                se: 63,
                ah: 0,
                al,
            });
        }

        // AC (split+1)-63 refinement passes (from al down to 0)
        let max_al = al_luma.max(al_chroma);
        for refine_al in (0..max_al).rev() {
            for c in 0..nc {
                let al = if c == 0 { al_luma } else { al_chroma };
                if refine_al < al {
                    scans.push(ProgressiveScan {
                        components: vec![c as u8],
                        ss: split + 1,
                        se: 63,
                        ah: refine_al + 1,
                        al: refine_al,
                    });
                }
            }
        }
    }

    scans
}

/// Generate the default jpegli-style progressive scan script.
///
/// Uses the same structure as `ComputedConfig::get_progressive_scan_script()`
/// for non-XYB mode with separate DC scans:
/// - Separate DC scans per component
/// - AC 1-2 at full precision (al=0) per component
/// - AC 3-63 at al=2 per component (successive approximation)
/// - AC 3-63 refinement ah=2→al=1 per component
/// - AC 3-63 refinement ah=1→al=0 per component
fn default_jpegli_progressive_scans(num_components: u8) -> Vec<ProgressiveScan> {
    let nc = num_components as usize;
    let mut scans = Vec::with_capacity(nc * 5);

    // Separate DC scans
    for c in 0..nc {
        scans.push(ProgressiveScan {
            components: vec![c as u8],
            ss: 0,
            se: 0,
            ah: 0,
            al: 0,
        });
    }

    // AC 1-2: full precision
    for c in 0..nc {
        scans.push(ProgressiveScan {
            components: vec![c as u8],
            ss: 1,
            se: 2,
            ah: 0,
            al: 0,
        });
    }

    // AC 3-63 first pass: al=2
    for c in 0..nc {
        scans.push(ProgressiveScan {
            components: vec![c as u8],
            ss: 3,
            se: 63,
            ah: 0,
            al: 2,
        });
    }

    // AC 3-63 refinement: ah=2→al=1
    for c in 0..nc {
        scans.push(ProgressiveScan {
            components: vec![c as u8],
            ss: 3,
            se: 63,
            ah: 2,
            al: 1,
        });
    }

    // AC 3-63 refinement: ah=1→al=0
    for c in 0..nc {
        scans.push(ProgressiveScan {
            components: vec![c as u8],
            ss: 3,
            se: 63,
            ah: 1,
            al: 0,
        });
    }

    scans
}
