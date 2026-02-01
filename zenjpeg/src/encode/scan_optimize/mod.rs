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
/// optimize + replay). We generate many more candidates than this and use the
/// frequency estimator to pre-filter down to this count.
const MAX_TRIAL_ENCODES: usize = 3;

/// Generate candidate progressive scan scripts for trial encoding.
///
/// Generates a broad set of structurally diverse scan scripts, uses the
/// frequency estimator to rank them within each category, then returns the
/// top candidates for trial encoding.
///
/// Strategy:
/// 1. Generate ~20 mixed-SA variants (different split points × al levels).
///    The estimator is accurate for RELATIVE ranking within similar structures.
/// 2. Pick the best mixed-SA variant by estimate.
/// 3. Run the mozjpeg-style 64-candidate search for the best uniform-al script.
/// 4. Always include the default jpegli script as a safety baseline.
/// 5. Deduplicate and return up to MAX_TRIAL_ENCODES scripts.
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

    // === Phase 1: Generate mixed-SA variants and pick the best by estimate ===
    // These all have the same structure (DC + low-band al=0 + high-band al=K + refinements),
    // so the estimator's relative ranking is reliable.
    let split_points: &[u8] = &config.frequency_splits;
    let al_levels: &[u8] = &[1, 2, 3];

    let mut best_mixed_sa: Option<(Vec<ProgressiveScan>, usize)> = None;

    for &split in split_points {
        for &al in al_levels {
            let al_c = al.min(config.al_max_chroma);
            let script = mixed_sa_split_progressive_scans(num_components, split, al, al_c);
            let est = estimate::estimate_script_cost(&script, y_blocks, cb_blocks, cr_blocks);

            if best_mixed_sa.as_ref().is_none_or(|(_, best)| est < *best) {
                best_mixed_sa = Some((script, est));
            }
        }
    }

    // === Phase 2: mozjpeg-style 64-candidate search (uniform al) ===
    let trial_scans = generate::generate_search_scans(num_components, &config);
    let scan_sizes =
        estimate::estimate_all_scan_sizes(&trial_scans, y_blocks, cb_blocks, cr_blocks);
    let selector = select::ScanSelector::new(num_components, config.clone());
    let search_result = selector.select_best(&scan_sizes);
    let optimizer_script = search_result.build_final_scans(num_components, &config);

    // === Phase 3: Assemble final candidates ===
    let default_script = default_jpegli_progressive_scans(num_components);

    let mut candidates: Vec<Vec<ProgressiveScan>> = Vec::with_capacity(MAX_TRIAL_ENCODES);

    // Always include the default
    candidates.push(default_script);

    // Add optimizer's pick (if different from default)
    if !scripts_equivalent(&optimizer_script, &candidates[0]) {
        candidates.push(optimizer_script);
    }

    // Add best mixed-SA variant (if different from all existing candidates)
    if let Some((mixed_script, _est)) = best_mixed_sa {
        let dominated = candidates
            .iter()
            .any(|c| scripts_equivalent(&mixed_script, c));
        if !dominated && candidates.len() < MAX_TRIAL_ENCODES {
            candidates.push(mixed_script);
        }
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

/// Simplest progressive scan script: no successive approximation.
///
/// Structure: separate DC + full AC 1-63 at al=0 per component.
/// This produces the fewest scans (2 per component) and is optimal when
/// SA overhead (extra scans, refinement passes) isn't worth the compression
/// benefit — common for images with lots of high-frequency content.
fn no_sa_progressive_scans(num_components: u8) -> Vec<ProgressiveScan> {
    let nc = num_components as usize;
    let mut scans = Vec::with_capacity(nc * 2);

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

    // Full AC 1-63 at al=0 per component
    for c in 0..nc {
        scans.push(ProgressiveScan {
            components: vec![c as u8],
            ss: 1,
            se: 63,
            ah: 0,
            al: 0,
        });
    }

    scans
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
