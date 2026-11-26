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

/// Maximum number of candidates returned by [`generate_candidate_scripts`].
///
/// Each candidate requires a full progressive encode pass (tokenize + Huffman
/// optimize + replay). We pre-filter via the shared histogram cache down to:
/// 1. The default jpegli script (safety baseline so search never loses to default),
/// 2. The better of the Phase 1 mixed-SA winner and Phase 2 mozjpeg 64-candidate winner,
/// 3. The Phase 3 per-component independent + 3-band script (when distinct).
const MAX_TRIAL_ENCODES: usize = 3;

/// Generate candidate progressive scan scripts for trial encoding.
///
/// Uses a shared histogram cache to avoid redundant block-scanning passes
/// across all search phases. Each unique `(component, ss, se, ah, al)`
/// combination is scanned exactly once.
///
/// Strategy:
/// 1. Pre-generate all mixed-SA variants, Phase 2 trial scans, and Phase 3
///    per-component independent (including 3-band luma) AC candidates.
/// 2. Warm the histogram cache once for all unique scan keys across all phases.
/// 3. Pick best mixed-SA (Phase 1), best uniform-al via mozjpeg search (Phase 2),
///    and best per-component independent AC script (Phase 3, inspired by
///    `jpegli-redux`'s per-component 3-band scan search).
/// 4. Always include the default jpegli script as a safety baseline, plus the
///    better of Phase 1/2 and the Phase 3 per-component winner when distinct.
///
/// # Returns
/// Up to `MAX_TRIAL_ENCODES` unique candidate scripts, always including the default.
#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_candidate_scripts(
    y_blocks: &[[i16; DCT_BLOCK_SIZE]],
    cb_blocks: &[[i16; DCT_BLOCK_SIZE]],
    cr_blocks: &[[i16; DCT_BLOCK_SIZE]],
    num_components: u8,
    interleave_dc_444: bool,
    #[cfg(feature = "parallel")] use_parallel: bool,
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
                interleave_dc_444,
            ));
        }
    }

    let default_script = default_jpegli_progressive_scans(num_components, interleave_dc_444);

    // Pre-generate Phase 3 per-component independent AC candidates (including
    // 3-band luma partitions and al>0 band splits) so all their scan keys are
    // warmed in the exact same single pass.
    let mut per_comp_candidates: Vec<Vec<Vec<ProgressiveScan>>> =
        Vec::with_capacity(num_components as usize);
    let mut flat_per_comp: Vec<Vec<ProgressiveScan>> = Vec::new();
    for c in 0..num_components {
        let cands = per_component_ac_candidates(c, c == 0);
        flat_per_comp.extend(cands.iter().cloned());
        per_comp_candidates.push(cands);
    }

    // === Warm cache: scan blocks once per unique (component, ss, se, ah, al) ===
    let trial_scans = generate::generate_search_scans(num_components, &config);
    let mut cache = estimate::ScanHistogramCache::warm(
        &trial_scans,
        &mixed_sa_scripts,
        &flat_per_comp,
        y_blocks,
        cb_blocks,
        cr_blocks,
        num_components,
        #[cfg(feature = "parallel")]
        use_parallel,
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

    // === Phase 3: Per-component independent + 3-band script ===
    let mut per_comp_script = Vec::new();
    push_dc_scans(&mut per_comp_script, num_components, interleave_dc_444);
    for cands in &per_comp_candidates {
        let mut best_idx = 0usize;
        let mut best_est = usize::MAX;
        for (i, cand) in cands.iter().enumerate() {
            let est = cache.estimate_script_cost_cached(cand);
            if est < best_est {
                best_est = est;
                best_idx = i;
            }
        }
        per_comp_script.extend(cands[best_idx].iter().cloned());
    }

    // === Phase 4: Deduplicate & assemble final trial list ===
    let mut candidates: Vec<Vec<ProgressiveScan>> = Vec::with_capacity(MAX_TRIAL_ENCODES);
    candidates.push(default_script);

    // Better of Phase 1 (mixed-SA) and Phase 2 (mozjpeg 64-cand)
    let optimizer_est = cache.estimate_script_cost_cached(&optimizer_script);
    let best_phase12 = match best_mixed_sa {
        Some((mixed_script, mixed_est)) => {
            if scripts_equivalent(&mixed_script, &optimizer_script) {
                optimizer_script
            } else if mixed_est < optimizer_est {
                mixed_script
            } else {
                optimizer_script
            }
        }
        None => optimizer_script,
    };

    if !candidates
        .iter()
        .any(|existing| scripts_equivalent(existing, &best_phase12))
    {
        candidates.push(best_phase12);
    }

    // Include Phase 3 per-component + 3-band winner when structurally distinct
    if !candidates
        .iter()
        .any(|existing| scripts_equivalent(existing, &per_comp_script))
    {
        candidates.push(per_comp_script);
    }

    Ok(candidates)
}

/// Check if two scan scripts are structurally equivalent.
pub(crate) fn scripts_equivalent(a: &[ProgressiveScan], b: &[ProgressiveScan]) -> bool {
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

/// Append DC scans: interleaved `[0, 1, 2]` on 3-component 4:4:4 (matching
/// `ComputedConfig::get_jpegli_scan_script`), separate per component otherwise.
fn push_dc_scans(scans: &mut Vec<ProgressiveScan>, num_components: u8, interleave_dc_444: bool) {
    if interleave_dc_444 && num_components == 3 {
        scans.push(ProgressiveScan {
            components: vec![0, 1, 2],
            ss: 0,
            se: 0,
            ah: 0,
            al: 0,
        });
    } else {
        for c in 0..num_components {
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

/// Per-component AC scan candidates (adapted from `jpegli-redux`'s BSD-3-Clause
/// per-component progressive scan search).
///
/// Covers 1-, 2-, and 3-band spectral partitions at uniform SA depths (`al=0..=2`
/// for luma, `al=0..=1` for chroma) paired with full-range `1..63` refinement
/// scans, plus mixed-SA low-frequency splits (`al=0` low band + `al>0` high band).
fn per_component_ac_candidates(comp: u8, is_luma: bool) -> Vec<Vec<ProgressiveScan>> {
    let mut scripts = Vec::with_capacity(if is_luma { 25 } else { 9 });

    let mut add_partition = |bands: &[(u8, u8)], depth: u8| {
        let mut s = Vec::with_capacity(bands.len() + depth as usize);
        for &(ss, se) in bands {
            s.push(ProgressiveScan {
                components: vec![comp],
                ss,
                se,
                ah: 0,
                al: depth,
            });
        }
        for al in (1..=depth).rev() {
            s.push(ProgressiveScan {
                components: vec![comp],
                ss: 1,
                se: 63,
                ah: al,
                al: al - 1,
            });
        }
        scripts.push(s);
    };

    let add_hybrid = |scripts: &mut Vec<Vec<ProgressiveScan>>, split: u8, depth: u8| {
        let mut s = Vec::with_capacity(2 + depth as usize);
        s.push(ProgressiveScan {
            components: vec![comp],
            ss: 1,
            se: split,
            ah: 0,
            al: 0,
        });
        s.push(ProgressiveScan {
            components: vec![comp],
            ss: split + 1,
            se: 63,
            ah: 0,
            al: depth,
        });
        for al in (1..=depth).rev() {
            s.push(ProgressiveScan {
                components: vec![comp],
                ss: split + 1,
                se: 63,
                ah: al,
                al: al - 1,
            });
        }
        scripts.push(s);
    };

    if is_luma {
        let parts: [&[(u8, u8)]; 7] = [
            &[(1, 63)],
            &[(1, 5), (6, 63)],
            &[(1, 2), (3, 63)],
            &[(1, 2), (3, 9), (10, 63)],
            &[(1, 8), (9, 63)],
            &[(1, 5), (6, 17), (18, 63)],
            &[(1, 3), (4, 12), (13, 63)],
        ];
        for depth in [0u8, 1, 2] {
            for p in parts {
                add_partition(p, depth);
            }
        }
        for (split, depth) in [(2u8, 2u8), (2, 1), (5, 1), (5, 2)] {
            add_hybrid(&mut scripts, split, depth);
        }
    } else {
        let parts: [&[(u8, u8)]; 3] = [&[(1, 63)], &[(1, 5), (6, 63)], &[(1, 2), (3, 63)]];
        for depth in [0u8, 1] {
            for p in parts {
                add_partition(p, depth);
            }
        }
        for (split, depth) in [(2u8, 2u8), (2, 1), (5, 1)] {
            add_hybrid(&mut scripts, split, depth);
        }
    }

    scripts
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
    interleave_dc_444: bool,
) -> Vec<ProgressiveScan> {
    let nc = num_components as usize;
    let mut scans = Vec::with_capacity(nc * 6);

    push_dc_scans(&mut scans, num_components, interleave_dc_444);

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
/// for non-XYB mode:
/// - Interleaved DC on 4:4:4 (3 components), separate DC otherwise
/// - AC 1-2 at full precision (al=0) per component
/// - AC 3-63 at al=2 per component (successive approximation)
/// - AC 3-63 refinement ah=2→al=1 per component
/// - AC 3-63 refinement ah=1→al=0 per component
fn default_jpegli_progressive_scans(
    num_components: u8,
    interleave_dc_444: bool,
) -> Vec<ProgressiveScan> {
    let nc = num_components as usize;
    let mut scans = Vec::with_capacity(nc * 5);

    push_dc_scans(&mut scans, num_components, interleave_dc_444);

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
