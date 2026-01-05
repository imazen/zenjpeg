//! Debug context mapping in progressive encoding
//!
//! This example shows how contexts map to Huffman tables.

use jpegli::huffman_opt::ContextConfig;

fn main() {
    println!("=== Context Mapping Analysis ===\n");

    // Get the progressive scan script
    let scans = get_progressive_scan_script(true);

    println!("Scan script:");
    for (i, scan) in scans.iter().enumerate() {
        let scan_type = if scan.ss == 0 {
            "DC"
        } else if scan.ah == 0 {
            "AC First"
        } else {
            "AC Refine"
        };
        println!(
            "  Scan {:2}: {:10} Ss={:2}-{:2}, Ah={}, Al={}, comps={:?}",
            i, scan_type, scan.ss, scan.se, scan.ah, scan.al, scan.components
        );
    }
    println!();

    // Create context config matching encode_progressive_optimized
    let num_components = 3;
    let context_config = ContextConfig::for_progressive(
        num_components,
        scans.iter().map(|s| (s.ss, s.se, s.components.len())),
    );

    println!("Context config:");
    println!("  num_dc_contexts: {}", context_config.num_dc_contexts());
    println!("  num_ac_contexts: {}", context_config.num_ac_contexts());
    println!("  ac_offset: {}", context_config.ac_offset);
    println!("  total_contexts: {}", context_config.num_contexts);
    println!();

    // Show context assignments for each scan
    println!("Context assignments:");
    for (scan_idx, scan) in scans.iter().enumerate() {
        if scan.ss == 0 && scan.se == 0 {
            // DC scan
            for (comp_in_scan, &comp_idx) in scan.components.iter().enumerate() {
                let context = context_config.dc_context(comp_idx as usize);
                println!(
                    "  Scan {:2} (DC) comp {}: context {}",
                    scan_idx, comp_idx, context
                );
            }
        } else {
            // AC scan
            for (comp_in_scan, &comp_idx) in scan.components.iter().enumerate() {
                let context = context_config.ac_context(scan_idx, comp_in_scan);
                println!(
                    "  Scan {:2} (AC) comp {}: context {}",
                    scan_idx, comp_idx, context
                );
            }
        }
    }
}

// Simplified scan struct for debugging
struct ProgressiveScan {
    components: Vec<u8>,
    ss: u8,
    se: u8,
    ah: u8,
    al: u8,
}

fn get_progressive_scan_script(is_color: bool) -> Vec<ProgressiveScan> {
    let num_components: u8 = if is_color { 3 } else { 1 };
    let use_refinement = true;

    let mut scans = Vec::new();

    // DC scan (interleaved for all components)
    scans.push(ProgressiveScan {
        components: (0..num_components).collect(),
        ss: 0,
        se: 0,
        ah: 0,
        al: 0,
    });

    // AC scans - iterate scan types first, then components (C++ order)
    if use_refinement {
        // AC 1-2 (per component)
        for c in 0..num_components {
            scans.push(ProgressiveScan {
                components: vec![c],
                ss: 1,
                se: 2,
                ah: 0,
                al: 0,
            });
        }

        // AC 3-63 first pass with Al=2 (per component)
        for c in 0..num_components {
            scans.push(ProgressiveScan {
                components: vec![c],
                ss: 3,
                se: 63,
                ah: 0,
                al: 2,
            });
        }

        // AC 3-63 refinement Ah=2, Al=1 (per component)
        for c in 0..num_components {
            scans.push(ProgressiveScan {
                components: vec![c],
                ss: 3,
                se: 63,
                ah: 2,
                al: 1,
            });
        }

        // AC 3-63 refinement Ah=1, Al=0 (per component)
        for c in 0..num_components {
            scans.push(ProgressiveScan {
                components: vec![c],
                ss: 3,
                se: 63,
                ah: 1,
                al: 0,
            });
        }
    } else {
        // Non-refinement: AC 1-5 then AC 6-63
        for c in 0..num_components {
            scans.push(ProgressiveScan {
                components: vec![c],
                ss: 1,
                se: 5,
                ah: 0,
                al: 0,
            });
        }
        for c in 0..num_components {
            scans.push(ProgressiveScan {
                components: vec![c],
                ss: 6,
                se: 63,
                ah: 0,
                al: 0,
            });
        }
    }

    scans
}
