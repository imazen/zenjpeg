//! Frequency-based cost estimation for candidate scans.
//!
//! Uses Huffman frequency counting to estimate relative scan sizes.
//! This is sufficient for ranking candidates since we only need relative
//! ordering, not exact byte counts.

use super::generate::TrialScan;
use crate::foundation::consts::DCT_BLOCK_SIZE;
use crate::huffman::optimize::FrequencyCounter;

/// Estimate encoded sizes for all candidate scans.
///
/// Returns a vector of estimated sizes (in bits) for each candidate scan.
/// Uses Huffman frequency analysis for AC scans and simple category counting
/// for DC scans.
pub(crate) fn estimate_all_scan_sizes(
    scans: &[TrialScan],
    y_blocks: &[[i16; DCT_BLOCK_SIZE]],
    cb_blocks: &[[i16; DCT_BLOCK_SIZE]],
    cr_blocks: &[[i16; DCT_BLOCK_SIZE]],
) -> Vec<usize> {
    let mut counter = FrequencyCounter::new();
    let mut sizes = Vec::with_capacity(scans.len());

    for scan in scans {
        counter.reset();

        let blocks = match scan.component {
            0 => y_blocks,
            1 => cb_blocks,
            2 => cr_blocks,
            _ => &[],
        };

        let estimated = if scan.is_dc() {
            if scan.comps_in_scan > 1 {
                // Multi-component DC: estimate each component separately and sum
                estimate_dc_scan(&mut counter, y_blocks)
                    + estimate_dc_scan(&mut counter, cb_blocks)
                    + estimate_dc_scan(&mut counter, cr_blocks)
            } else {
                estimate_dc_scan(&mut counter, blocks)
            }
        } else if scan.ah == 0 {
            // AC first scan
            estimate_ac_first_scan(&mut counter, blocks, scan.ss, scan.se, scan.al)
        } else {
            // AC refinement scan
            estimate_ac_refinement_scan(blocks, scan.ss, scan.se, scan.ah, scan.al)
        };

        sizes.push(estimated);
    }

    sizes
}

/// Estimate DC scan cost using Huffman frequency analysis.
///
/// Counts DC delta categories (the standard JPEG DC difference encoding)
/// and estimates encoding cost from the resulting Huffman code lengths.
fn estimate_dc_scan(counter: &mut FrequencyCounter, blocks: &[[i16; DCT_BLOCK_SIZE]]) -> usize {
    counter.reset();

    if blocks.is_empty() {
        return 0;
    }

    let mut prev_dc = 0i16;

    for block in blocks {
        let dc = block[0];
        let diff = dc.wrapping_sub(prev_dc);
        prev_dc = dc;

        // DC category = number of bits needed to represent the difference
        let category = dc_category(diff);
        counter.count(category);
    }

    // Cost = Huffman table overhead + sum(count × code_length) + extra bits
    // For DC, each symbol also carries `category` extra bits for the actual value
    let huffman_cost = counter.estimate_encoding_cost();
    let extra_bits: f64 = blocks
        .iter()
        .scan(0i16, |prev, block| {
            let diff = block[0].wrapping_sub(*prev);
            *prev = block[0];
            Some(dc_category(diff) as f64)
        })
        .sum();

    (huffman_cost + extra_bits) as usize
}

/// Estimate AC first scan cost (ah=0).
///
/// Counts run/value Huffman symbols for coefficients shifted by `al`,
/// within the spectral range [ss, se]. Properly accumulates EOB runs
/// across blocks (progressive JPEG merges consecutive empty blocks into
/// a single EOB run symbol + extra bits, rather than individual EOBs).
fn estimate_ac_first_scan(
    counter: &mut FrequencyCounter,
    blocks: &[[i16; DCT_BLOCK_SIZE]],
    ss: u8,
    se: u8,
    al: u8,
) -> usize {
    counter.reset();

    if blocks.is_empty() {
        return 0;
    }

    let ss = ss as usize;
    let se = se as usize;
    let mut eob_run = 0u32;
    let mut eob_extra_bits = 0usize;

    for block in blocks {
        let mut run = 0u8;
        let mut block_has_nonzero = false;

        for k in ss..=se {
            let coeff = block[k] >> al;
            let abs_coeff = coeff.unsigned_abs();

            if abs_coeff == 0 {
                run += 1;
                continue;
            }

            // First non-zero in this block: flush pending EOB run
            if !block_has_nonzero && eob_run > 0 {
                eob_extra_bits += count_eob_run(counter, eob_run);
                eob_run = 0;
            }
            block_has_nonzero = true;

            // Emit ZRL (16 zero run) symbols for long runs
            while run >= 16 {
                counter.count(0xF0); // ZRL symbol
                run -= 16;
            }

            // Encode run/size symbol
            let size = ac_category(abs_coeff);
            let symbol = (run << 4) | size;
            counter.count(symbol);
            run = 0;
        }

        if run > 0 {
            // Trailing zeros in this block → accumulate into EOB run
            eob_run += 1;
            if eob_run >= 32767 {
                eob_extra_bits += count_eob_run(counter, eob_run);
                eob_run = 0;
            }
        }
    }

    // Flush final EOB run
    if eob_run > 0 {
        eob_extra_bits += count_eob_run(counter, eob_run);
    }

    // Extra bits: each non-zero AC coefficient carries `size` extra bits (includes sign)
    let value_extra_bits: f64 = blocks
        .iter()
        .map(|block| {
            let mut bits = 0.0f64;
            for k in ss..=se {
                let coeff = block[k] >> al;
                let abs_coeff = coeff.unsigned_abs();
                if abs_coeff > 0 {
                    bits += ac_category(abs_coeff) as f64; // value bits include sign
                }
            }
            bits
        })
        .sum();

    (counter.estimate_encoding_cost() + value_extra_bits) as usize + eob_extra_bits
}

/// Estimate AC refinement scan cost (ah > 0).
///
/// In refinement scans, coefficients fall into three categories:
/// 1. Already non-zero from previous pass: contribute 1 refbit each
/// 2. Newly non-zero in this pass: Huffman-coded run/value symbol + 1 sign bit
/// 3. Still zero: part of the run length
///
/// Refbits (1 bit per previously-nonzero coefficient) are NOT Huffman-coded,
/// so we track them separately. EOB runs are accumulated across blocks.
fn estimate_ac_refinement_scan(
    blocks: &[[i16; DCT_BLOCK_SIZE]],
    ss: u8,
    se: u8,
    ah: u8,
    al: u8,
) -> usize {
    if blocks.is_empty() {
        return 0;
    }

    let ss = ss as usize;
    let se = se as usize;
    let mut counter = FrequencyCounter::new();
    let mut total_refbits = 0usize;
    let mut eob_run = 0u32;
    let mut eob_extra_bits = 0usize;

    for block in blocks {
        let mut run = 0u8;
        let mut block_has_newly_sig = false;

        for k in ss..=se {
            let coeff = block[k];
            let abs_coeff = coeff.unsigned_abs();

            // Check if this coefficient was non-zero in the previous pass
            let prev_nonzero = (abs_coeff >> ah) > 0;
            // Check if this coefficient becomes non-zero in the current pass
            let cur_bit = (abs_coeff >> al) & 1;

            if prev_nonzero {
                // Already established: 1 refbit (not Huffman-coded)
                total_refbits += 1;
            } else if cur_bit != 0 {
                // Newly significant: flush pending EOB run
                if !block_has_newly_sig && eob_run > 0 {
                    eob_extra_bits += count_eob_run(&mut counter, eob_run);
                    eob_run = 0;
                }
                block_has_newly_sig = true;

                while run >= 16 {
                    counter.count(0xF0); // ZRL
                    run -= 16;
                }
                // Symbol is (run << 4) | 1 for newly-significant coefficients
                let symbol = (run << 4) | 1;
                counter.count(symbol);
                total_refbits += 1; // sign bit
                run = 0;
            } else {
                // Still zero
                run += 1;
            }
        }

        if !block_has_newly_sig {
            // No newly-significant coefficients: EOB
            eob_run += 1;
            if eob_run >= 32767 {
                eob_extra_bits += count_eob_run(&mut counter, eob_run);
                eob_run = 0;
            }
        } else if run > 0 {
            // Had some newly-significant but trailing unestablished zeros
            eob_run += 1;
            if eob_run >= 32767 {
                eob_extra_bits += count_eob_run(&mut counter, eob_run);
                eob_run = 0;
            }
        }
    }

    // Flush final EOB run
    if eob_run > 0 {
        eob_extra_bits += count_eob_run(&mut counter, eob_run);
    }

    let huffman_cost = counter.estimate_encoding_cost();
    (huffman_cost as usize) + total_refbits + eob_extra_bits
}

/// Per-scan overhead in bits for SOS marker + DHT envelope + byte padding.
///
/// This matches the SCAN_OVERHEAD constant in select.rs and is used by
/// `estimate_script_cost()` for full scan script cost estimation.
const SCAN_OVERHEAD_BITS: usize = 150;

/// Estimate the total encoded cost of a complete progressive scan script.
///
/// Evaluates each scan in the script using the same frequency-based estimation
/// used for individual trial scans, plus per-scan overhead.
///
/// The estimate is accurate for ranking scripts with the same number of scans
/// (relative ordering is reliable). For scripts with different scan counts,
/// there is a systematic bias due to Huffman clustering effects:
/// - 6-9 scans: ratio ~0.99 (nearly exact)
/// - 12 scans: ratio ~1.05 (5% overestimate)
/// - 15 scans: ratio ~1.22 (22% overestimate)
///
/// Cross-structure comparisons (e.g., 9-scan vs 15-scan) must use trial
/// encoding for reliable results.
///
/// Returns the estimated total cost in bits (uncorrected).
pub(crate) fn estimate_script_cost(
    script: &[super::super::config::ProgressiveScan],
    y_blocks: &[[i16; DCT_BLOCK_SIZE]],
    cb_blocks: &[[i16; DCT_BLOCK_SIZE]],
    cr_blocks: &[[i16; DCT_BLOCK_SIZE]],
) -> usize {
    let mut counter = FrequencyCounter::new();
    let mut total = 0usize;

    for scan in script {
        let cost = if scan.ss == 0 && scan.se == 0 {
            // DC scan
            let mut dc_cost = 0;
            for &comp in &scan.components {
                let blocks = match comp {
                    0 => y_blocks,
                    1 => cb_blocks,
                    2 => cr_blocks,
                    _ => continue,
                };
                dc_cost += estimate_dc_scan(&mut counter, blocks);
            }
            dc_cost
        } else if scan.ah == 0 {
            // AC first scan (single component)
            let blocks = match scan.components[0] {
                0 => y_blocks,
                1 => cb_blocks,
                2 => cr_blocks,
                _ => continue,
            };
            estimate_ac_first_scan(&mut counter, blocks, scan.ss, scan.se, scan.al)
        } else {
            // AC refinement scan (single component)
            let blocks = match scan.components[0] {
                0 => y_blocks,
                1 => cb_blocks,
                2 => cr_blocks,
                _ => continue,
            };
            estimate_ac_refinement_scan(blocks, scan.ss, scan.se, scan.ah, scan.al)
        };

        total += cost + SCAN_OVERHEAD_BITS;
    }

    // Note: The raw estimate overestimates scripts with >9 scans due to
    // Huffman clustering effects not being modeled. Calibrated bias:
    //   6-9 scans: ratio ~0.99 (accurate)
    //   12 scans: ratio ~1.05 (5% over)
    //   15 scans: ratio ~1.22 (22% over)
    // A quadratic correction (ratio = 1.0 - 0.001*(s-9) + 0.00633*(s-9)^2)
    // was tested but the per-image variance (±10% at 15 scans) is too wide
    // for reliable cross-structure comparison. The estimate is accurate for
    // ranking scripts with the SAME scan count, which is how the caller uses it.

    total
}

/// Count an EOB run into the frequency counter.
///
/// Progressive JPEG encodes consecutive end-of-block markers as:
/// - n=1: symbol 0x00, 0 extra bits
/// - n=2-3: symbol 0x10, 1 extra bit
/// - n=4-7: symbol 0x20, 2 extra bits
/// - n=2^k .. 2^(k+1)-1: symbol (k<<4), k extra bits
///
/// Returns the number of extra bits for this run.
fn count_eob_run(counter: &mut FrequencyCounter, n: u32) -> usize {
    debug_assert!(n > 0);
    // category = floor(log2(n))
    let category = 31 - n.leading_zeros();
    let symbol = (category as u8) << 4;
    counter.count(symbol);
    category as usize
}

/// Compute the DC category (number of bits) for a DC difference value.
#[inline]
fn dc_category(diff: i16) -> u8 {
    if diff == 0 {
        return 0;
    }
    let abs_diff = diff.unsigned_abs();
    16 - abs_diff.leading_zeros() as u8
}

/// Compute the AC category (number of bits) for an absolute AC coefficient.
#[inline]
fn ac_category(abs_val: u16) -> u8 {
    if abs_val == 0 {
        return 0;
    }
    16 - abs_val.leading_zeros() as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::scan_optimize::generate::generate_search_scans;
    use crate::encode::scan_optimize::ScanSearchConfig;

    #[test]
    fn test_dc_category() {
        assert_eq!(dc_category(0), 0);
        assert_eq!(dc_category(1), 1);
        assert_eq!(dc_category(-1), 1);
        assert_eq!(dc_category(2), 2);
        assert_eq!(dc_category(3), 2);
        assert_eq!(dc_category(4), 3);
        assert_eq!(dc_category(7), 3);
        assert_eq!(dc_category(-7), 3);
        assert_eq!(dc_category(255), 8);
    }

    #[test]
    fn test_ac_category() {
        assert_eq!(ac_category(0), 0);
        assert_eq!(ac_category(1), 1);
        assert_eq!(ac_category(2), 2);
        assert_eq!(ac_category(3), 2);
        assert_eq!(ac_category(255), 8);
        assert_eq!(ac_category(1023), 10);
    }

    #[test]
    fn test_eob_run_encoding() {
        let mut counter = FrequencyCounter::new();

        // n=1: symbol 0x00, 0 extra bits
        assert_eq!(count_eob_run(&mut counter, 1), 0);

        // n=2: symbol 0x10, 1 extra bit
        counter.reset();
        assert_eq!(count_eob_run(&mut counter, 2), 1);

        // n=4: symbol 0x20, 2 extra bits
        counter.reset();
        assert_eq!(count_eob_run(&mut counter, 4), 2);

        // n=100: symbol 0x60 (category 6), 6 extra bits
        counter.reset();
        assert_eq!(count_eob_run(&mut counter, 100), 6);

        // n=32767: symbol 0xe0 (category 14), 14 extra bits
        counter.reset();
        assert_eq!(count_eob_run(&mut counter, 32767), 14);
    }

    #[test]
    fn test_eob_runs_cheaper_than_individual() {
        // 1000 all-zero blocks: EOB run should be much cheaper than 1000 individual EOBs
        let zero_blocks = vec![[0i16; 64]; 1000];

        // Estimate with EOB runs (current implementation)
        let mut counter = FrequencyCounter::new();
        let cost_with_runs = estimate_ac_first_scan(&mut counter, &zero_blocks, 1, 63, 0);

        // The cost should be very small: just one EOB run symbol + extra bits
        // For 1000 blocks: category = floor(log2(1000)) = 9, so symbol 0x90, 9 extra bits
        // Huffman table overhead ~200 bits + 1 symbol code + 9 extra bits
        assert!(
            cost_with_runs < 300,
            "1000 zero-block EOB run should be very cheap, got {}",
            cost_with_runs
        );
    }

    #[test]
    fn test_estimate_zero_blocks() {
        let config = ScanSearchConfig::default();
        let scans = generate_search_scans(3, &config);
        let zero_blocks = vec![[0i16; 64]; 100];

        let sizes = estimate_all_scan_sizes(&scans, &zero_blocks, &zero_blocks, &zero_blocks);

        assert_eq!(sizes.len(), 64);
        // All sizes should be non-negative (some may be 0 for degenerate cases)
        for (i, &size) in sizes.iter().enumerate() {
            // DC scans with zero blocks should still have some overhead
            assert!(
                size < 1_000_000,
                "Scan {} has unreasonably large size: {}",
                i,
                size
            );
        }
    }

    #[test]
    fn test_estimate_produces_valid_sizes() {
        let config = ScanSearchConfig::default();
        let scans = generate_search_scans(3, &config);

        // Create blocks with some realistic-ish data
        let mut y_blocks = vec![[0i16; 64]; 64];
        let mut cb_blocks = vec![[0i16; 64]; 64];
        let cr_blocks = vec![[0i16; 64]; 64];

        for (i, block) in y_blocks.iter_mut().enumerate() {
            block[0] = (i as i16) * 10; // DC values
            block[1] = 5; // Some AC
            block[2] = -3;
            if i % 4 == 0 {
                block[10] = 2;
                block[20] = -1;
            }
        }
        for (i, block) in cb_blocks.iter_mut().enumerate() {
            block[0] = (i as i16) * 5;
            block[1] = 2;
        }

        let sizes = estimate_all_scan_sizes(&scans, &y_blocks, &cb_blocks, &cr_blocks);

        assert_eq!(sizes.len(), 64);

        // DC scan (index 0) should have some cost
        assert!(sizes[0] > 0, "DC scan should have non-zero cost");

        // AC scans for populated Y blocks should have significant cost
        assert!(sizes[1] > 0, "Y AC 1-8 should have non-zero cost with data");
    }

    #[test]
    fn test_dc_scan_monotonic_with_more_blocks() {
        let mut counter = FrequencyCounter::new();

        // More blocks should never produce smaller DC estimate
        let small_blocks = vec![[10i16; 64]; 10];
        let large_blocks = vec![[10i16; 64]; 100];

        let small_cost = estimate_dc_scan(&mut counter, &small_blocks);
        let large_cost = estimate_dc_scan(&mut counter, &large_blocks);

        assert!(
            large_cost >= small_cost,
            "More blocks should cost at least as much: {} < {}",
            large_cost,
            small_cost
        );
    }

    #[test]
    fn test_refinement_has_refbits() {
        // Blocks with non-zero coefficients at ah level should produce refbits
        let mut blocks = vec![[0i16; 64]; 10];
        for block in blocks.iter_mut() {
            // Coefficient with bit 1 set (will be non-zero at ah=1)
            block[1] = 2; // binary 10, so at ah=1 this is non-zero
            block[2] = 3; // binary 11, non-zero at both levels
        }

        let cost = estimate_ac_refinement_scan(&blocks, 1, 63, 1, 0);
        assert!(cost > 0, "Refinement scan should have non-zero cost");
    }
}
