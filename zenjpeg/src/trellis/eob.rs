//! EOB (end-of-block) run optimization for progressive JPEG.
//!
//! Cross-block EOB optimization finds the optimal placement of EOBRUN codes
//! to minimize total encoding cost across a row of blocks.
//!
//! Ported from mozjpeg jcdctmgr.c EOB optimization.

use crate::foundation::consts::{DCT_BLOCK_SIZE, JPEG_NATURAL_ORDER};

use super::ac::jpeg_nbits;
use super::rate::RateTable;

/// Information about a block's EOB status for cross-block optimization.
#[derive(Debug, Clone, Copy)]
pub struct BlockEobInfo {
    /// Cost of making all AC coefficients in this block zero
    pub zero_block_cost: f32,
    /// Cost of encoding this block optimally (with non-zero coefficients)
    pub best_cost: f32,
    /// Cost without the EOB marker (for cross-block chaining)
    pub best_cost_skip: f32,
    /// EOB status: 0 = no EOB needed (last coef at Se), 1 = needs EOB, 2 = all-zero block
    pub requires_eob: u8,
    /// Whether this block has any non-zero AC coefficients
    pub has_nonzero_ac: bool,
}

/// Optimize EOB runs across a row of blocks.
///
/// This function implements cross-block EOB optimization for progressive JPEG.
/// It finds the optimal placement of EOBRUN codes to minimize the total encoding cost.
///
/// # Arguments
/// * `blocks` - Quantized coefficient blocks (will be modified to zero out run blocks)
/// * `block_info` - Per-block EOB information computed during trellis quantization
/// * `ac_table` - AC rate table for EOBRUN cost estimation
/// * `ss` - Spectral selection start (first AC coefficient index)
/// * `se` - Spectral selection end (last AC coefficient index)
///
/// # Returns
/// The number of blocks that were zeroed as part of runs.
#[allow(clippy::needless_range_loop)]
pub fn optimize_eob_runs(
    blocks: &mut [[i16; DCT_BLOCK_SIZE]],
    block_info: &[BlockEobInfo],
    ac_table: &RateTable,
    ss: usize,
    se: usize,
) -> usize {
    let num_blocks = blocks.len();
    if num_blocks == 0 || ss >= se {
        return 0;
    }

    // Accumulated cost arrays for dynamic programming
    let mut accumulated_zero_block_cost = vec![0.0f32; num_blocks + 1];
    let mut accumulated_block_cost = vec![0.0f32; num_blocks + 1];
    let mut block_run_start = vec![0usize; num_blocks];

    accumulated_zero_block_cost[0] = 0.0;
    accumulated_block_cost[0] = 0.0;

    // Forward pass: compute optimal costs
    for bi in 0..num_blocks {
        accumulated_zero_block_cost[bi + 1] =
            accumulated_zero_block_cost[bi] + block_info[bi].zero_block_cost;

        // If this block is all-zero, it can only extend a run
        if block_info[bi].requires_eob == 2 {
            block_run_start[bi] = 0;
            accumulated_block_cost[bi + 1] = accumulated_zero_block_cost[bi + 1];
            continue;
        }

        // Try starting a zero-block run from each previous position
        let mut best_cost = f32::MAX;
        let mut best_start = 0;

        for i in 0..=bi {
            if block_info[i].requires_eob == 2 {
                continue;
            }

            let mut cost = block_info[bi].best_cost_skip;
            cost += accumulated_zero_block_cost[bi] - accumulated_zero_block_cost[i];
            if i > 0 {
                cost += accumulated_block_cost[i];
            }

            // EOBRUN cost
            let zero_block_run = bi - i + block_info[i].requires_eob as usize;
            if zero_block_run > 0 {
                let nbits = jpeg_nbits(zero_block_run as i16) as usize;
                let (_, eobrun_size) = ac_table.get_code((16 * nbits) as u8);
                if eobrun_size > 0 {
                    cost += eobrun_size as f32 + nbits as f32;
                } else {
                    cost += 16.0;
                }
            }

            if cost < best_cost {
                best_cost = cost;
                best_start = i;
            }
        }

        block_run_start[bi] = best_start;
        accumulated_block_cost[bi + 1] = best_cost;
    }

    // Find optimal ending point
    let mut last_block = num_blocks;
    let mut best_cost = f32::MAX;

    for i in 0..=num_blocks {
        if i > 0 && block_info[i - 1].requires_eob == 2 {
            continue;
        }

        let mut cost = accumulated_zero_block_cost[num_blocks] - accumulated_zero_block_cost[i];

        let zero_block_run = num_blocks - i
            + if i > 0 {
                block_info[i - 1].requires_eob as usize
            } else {
                0
            };
        if zero_block_run > 0 && i < num_blocks {
            let nbits = jpeg_nbits(zero_block_run as i16) as usize;
            let (_, eobrun_size) = ac_table.get_code((16 * nbits) as u8);
            if eobrun_size > 0 {
                cost += eobrun_size as f32 + nbits as f32;
            }
        }

        if i > 0 {
            cost += accumulated_block_cost[i];
        }

        if cost < best_cost {
            best_cost = cost;
            last_block = i;
        }
    }

    // Backward pass: zero out blocks that are part of runs
    let mut zeroed_count = 0;
    last_block = last_block.saturating_sub(1);

    let mut bi = num_blocks;
    while bi > 0 {
        bi -= 1;
        while bi >= last_block && bi < num_blocks {
            for j in ss..=se {
                let z = JPEG_NATURAL_ORDER[j] as usize;
                if blocks[bi][z] != 0 {
                    blocks[bi][z] = 0;
                    zeroed_count += 1;
                }
            }
            if bi == 0 {
                break;
            }
            bi -= 1;
        }
        if bi > 0 && bi <= last_block {
            last_block = block_run_start[bi].saturating_sub(1);
        }
    }

    zeroed_count
}

/// Compute EOB info for a single block during trellis quantization.
///
/// Called after trellis quantization to record information needed
/// for cross-block EOB optimization.
pub fn compute_block_eob_info(
    block: &[i16; DCT_BLOCK_SIZE],
    zero_dist: f32,
    best_cost: f32,
    best_cost_skip: f32,
    last_coeff_idx: usize,
    ss: usize,
    se: usize,
) -> BlockEobInfo {
    let has_nonzero_ac = (ss..=se).any(|i| block[JPEG_NATURAL_ORDER[i] as usize] != 0);
    let requires_eob = if !has_nonzero_ac {
        2 // All-zero block
    } else if last_coeff_idx >= se {
        0 // No EOB needed (last coef at end)
    } else {
        1 // Needs EOB
    };

    BlockEobInfo {
        zero_block_cost: zero_dist,
        best_cost,
        best_cost_skip,
        requires_eob,
        has_nonzero_ac,
    }
}

/// Estimate EOB info from an already-quantized block.
///
/// This function estimates the `BlockEobInfo` needed for cross-block EOB optimization
/// when full trellis cost information is not available.
pub fn estimate_block_eob_info(
    block: &[i16; DCT_BLOCK_SIZE],
    ac_table: &RateTable,
    ss: usize,
    se: usize,
) -> BlockEobInfo {
    let mut zero_block_cost = 0.0f32;
    let mut best_cost = 0.0f32;
    let mut last_nonzero_idx = 0usize;
    let mut has_nonzero_ac = false;
    let mut run = 0u8;

    for i in ss..=se {
        let z = JPEG_NATURAL_ORDER[i] as usize;
        let coef = block[z];

        if coef == 0 {
            run = run.saturating_add(1);
        } else {
            has_nonzero_ac = true;
            last_nonzero_idx = i;

            zero_block_cost += (coef as f32) * (coef as f32);

            let nbits = jpeg_nbits(coef);
            let symbol = ((run.min(15)) << 4) | nbits;
            let (_, symbol_size) = ac_table.get_code(symbol);
            best_cost += symbol_size as f32 + nbits as f32;

            let full_zrls = run / 16;
            if full_zrls > 0 {
                let (_, zrl_size) = ac_table.get_code(0xF0);
                best_cost += (full_zrls as f32) * (zrl_size as f32);
            }

            run = 0;
        }
    }

    let requires_eob = if !has_nonzero_ac {
        2
    } else if last_nonzero_idx >= se {
        0
    } else {
        1
    };

    let eob_cost = if requires_eob == 1 {
        let (_, eob_size) = ac_table.get_code(0x00);
        eob_size as f32
    } else {
        0.0
    };

    BlockEobInfo {
        zero_block_cost,
        best_cost: best_cost + eob_cost,
        best_cost_skip: best_cost,
        requires_eob,
        has_nonzero_ac,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_ac_table() -> RateTable {
        RateTable::standard_luma_ac()
    }

    #[test]
    fn test_estimate_block_eob_info_all_zero() {
        let ac_table = create_ac_table();
        let block = [0i16; DCT_BLOCK_SIZE];

        let info = estimate_block_eob_info(&block, &ac_table, 1, 63);

        assert_eq!(info.requires_eob, 2);
        assert!(!info.has_nonzero_ac);
        assert_eq!(info.zero_block_cost, 0.0);
    }

    #[test]
    fn test_estimate_block_eob_info_with_coefficients() {
        let ac_table = create_ac_table();
        let mut block = [0i16; DCT_BLOCK_SIZE];
        block[JPEG_NATURAL_ORDER[1] as usize] = 5;
        block[JPEG_NATURAL_ORDER[2] as usize] = 3;

        let info = estimate_block_eob_info(&block, &ac_table, 1, 63);

        assert_eq!(info.requires_eob, 1);
        assert!(info.has_nonzero_ac);
        assert!(info.zero_block_cost > 0.0);
        assert!(info.best_cost > 0.0);
    }

    #[test]
    fn test_estimate_block_eob_info_last_at_end() {
        let ac_table = create_ac_table();
        let mut block = [0i16; DCT_BLOCK_SIZE];
        block[JPEG_NATURAL_ORDER[63] as usize] = 2;

        let info = estimate_block_eob_info(&block, &ac_table, 1, 63);

        assert_eq!(info.requires_eob, 0);
        assert!(info.has_nonzero_ac);
    }

    #[test]
    fn test_optimize_eob_runs_empty() {
        let ac_table = create_ac_table();
        let blocks: &mut [[i16; DCT_BLOCK_SIZE]] = &mut [];
        let block_info: &[BlockEobInfo] = &[];
        let zeroed = optimize_eob_runs(blocks, block_info, &ac_table, 1, 63);
        assert_eq!(zeroed, 0);
    }

    #[test]
    fn test_optimize_eob_runs_all_zero() {
        let ac_table = create_ac_table();
        let mut blocks = vec![[0i16; DCT_BLOCK_SIZE]; 4];
        let eob_info: Vec<_> = blocks
            .iter()
            .map(|b| estimate_block_eob_info(b, &ac_table, 1, 63))
            .collect();

        let zeroed = optimize_eob_runs(&mut blocks, &eob_info, &ac_table, 1, 63);
        assert_eq!(zeroed, 0); // Already all zero
    }

    #[test]
    fn test_optimize_eob_runs_with_content() {
        let ac_table = create_ac_table();
        let mut blocks = vec![[0i16; DCT_BLOCK_SIZE]; 4];
        for (i, block) in blocks.iter_mut().enumerate() {
            block[JPEG_NATURAL_ORDER[1] as usize] = (i + 1) as i16 * 10;
        }

        let eob_info: Vec<_> = blocks
            .iter()
            .map(|b| estimate_block_eob_info(b, &ac_table, 1, 63))
            .collect();

        let zeroed = optimize_eob_runs(&mut blocks, &eob_info, &ac_table, 1, 63);
        assert!(zeroed <= 4 * 63);
    }

    #[test]
    fn test_optimize_eob_runs_mixed() {
        let ac_table = create_ac_table();
        let mut blocks = vec![[0i16; DCT_BLOCK_SIZE]; 5];
        blocks[0][JPEG_NATURAL_ORDER[1] as usize] = 50;
        // blocks[1], [2] are all zero
        blocks[3][JPEG_NATURAL_ORDER[1] as usize] = 30;
        // blocks[4] is all zero

        let eob_info: Vec<_> = blocks
            .iter()
            .map(|b| estimate_block_eob_info(b, &ac_table, 1, 63))
            .collect();

        assert!(eob_info[0].has_nonzero_ac);
        assert!(!eob_info[1].has_nonzero_ac);
        assert!(!eob_info[2].has_nonzero_ac);
        assert!(eob_info[3].has_nonzero_ac);
        assert!(!eob_info[4].has_nonzero_ac);

        let zeroed = optimize_eob_runs(&mut blocks, &eob_info, &ac_table, 1, 63);
        assert!(zeroed <= 5 * 63);
    }
}
