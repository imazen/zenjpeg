//! DC prediction analysis: measure the effect of DC prediction resets on compression.
//!
//! In JPEG, DC coefficients are differentially encoded: each block stores
//! (DC - prev_DC). At restart markers, prev_DC resets to 0, so the block
//! stores its absolute DC value. This costs more bits if the DC values are
//! smooth (small differences), but can help if there are sharp transitions.
//!
//! This module measures the optimal placement of DC prediction resets.

use crate::foundation::consts::DCT_BLOCK_SIZE;
use crate::huffman::optimize::FrequencyCounter;

use alloc::vec;
use alloc::vec::Vec;

/// Compute the Huffman symbol category for a DC difference value.
/// Category N requires N extra bits. Category 0 = value 0.
#[inline]
fn dc_category(value: i16) -> u8 {
    if value == 0 {
        return 0;
    }
    let abs_val = value.unsigned_abs();
    16 - abs_val.leading_zeros() as u8
}

/// Compute total DC encoding cost (Huffman symbol bits + extra bits) for a given
/// sequence of DC differences, using the optimal Huffman table for those differences.
fn dc_encoding_cost(diffs: &[i16]) -> u64 {
    if diffs.is_empty() {
        return 0;
    }

    let mut counter = FrequencyCounter::new();
    let mut extra_bits_total = 0u64;

    for &d in diffs {
        let cat = dc_category(d);
        counter.count(cat);
        extra_bits_total += cat as u64;
    }

    // Generate optimal code lengths for this distribution
    let lengths = match counter.generate_lengths() {
        Ok(l) => l,
        Err(_) => return u64::MAX,
    };

    // Total = sum(count * code_length) + sum(extra_bits)
    let mut huffman_bits = 0u64;
    for i in 0..256 {
        huffman_bits += counter.get_count(i as u8) as u64 * lengths[i] as u64;
    }

    huffman_bits + extra_bits_total
}

/// Compute the cost of DC encoding for one row of blocks.
/// Returns the sum of `dc_category(diff)` for each DC difference in the row.
/// This is a proxy for encoding cost without needing to build a Huffman table.
fn dc_row_cost_simple(blocks: &[[i16; DCT_BLOCK_SIZE]], start: usize, count: usize) -> (u64, i16) {
    let mut cost = 0u64;
    let mut prev_dc = if start == 0 { 0i16 } else { blocks[start - 1][0] };

    for i in start..start + count {
        let dc = blocks[i][0];
        let diff = dc - prev_dc;
        let cat = dc_category(diff);
        // Cost = Huffman code length (roughly category + 1) + extra bits (category)
        // For a rough estimate, cost ≈ 2 * category + 1 (for category > 0)
        cost += if cat == 0 { 1 } else { (2 * cat + 1) as u64 };
        prev_dc = dc;
    }

    (cost, blocks[start + count - 1][0])
}

/// Compute what the DC row cost would be if prev_dc was reset to 0.
fn dc_row_cost_with_reset(blocks: &[[i16; DCT_BLOCK_SIZE]], start: usize, count: usize) -> u64 {
    let mut cost = 0u64;
    let mut prev_dc = 0i16; // Reset!

    for i in start..start + count {
        let dc = blocks[i][0];
        let diff = dc - prev_dc;
        let cat = dc_category(diff);
        cost += if cat == 0 { 1 } else { (2 * cat + 1) as u64 };
        prev_dc = dc;
    }

    cost
}

/// Result of DC prediction analysis.
#[derive(Clone, Debug)]
pub struct DcAnalysisResult {
    /// Total DC bits with no restarts (continuous prediction)
    pub no_restart_cost: u64,
    /// Total DC bits with fixed restart interval
    pub fixed_restart_costs: Vec<(usize, u64)>, // (interval_in_mcu_rows, cost)
    /// Total DC bits with optimal per-row restart placement
    pub optimal_restart_cost: u64,
    /// Optimal restart positions (MCU row indices where restarts help)
    pub optimal_restart_rows: Vec<usize>,
    /// Total blocks analyzed
    pub total_blocks: usize,
    /// MCU columns
    pub mcu_cols: usize,
    /// MCU rows
    pub mcu_rows: usize,
}

/// Analyze DC prediction across the image to find optimal restart placement.
///
/// For 4:4:4 images only. Analyzes Y, Cb, Cr DC channels together.
pub fn analyze_dc_prediction(
    y_blocks: &[[i16; DCT_BLOCK_SIZE]],
    cb_blocks: &[[i16; DCT_BLOCK_SIZE]],
    cr_blocks: &[[i16; DCT_BLOCK_SIZE]],
    mcu_cols: usize,
    mcu_rows: usize,
    is_color: bool,
) -> DcAnalysisResult {
    // Cost with no restarts at all
    let no_restart = total_dc_cost_no_restart(y_blocks, cb_blocks, cr_blocks, mcu_cols, mcu_rows, is_color);

    // Cost with fixed restart intervals (in MCU rows)
    let mut fixed_costs = Vec::new();
    for &interval in &[1, 2, 4, 8, 16, 32] {
        if interval <= mcu_rows {
            let cost = total_dc_cost_fixed_restart(
                y_blocks, cb_blocks, cr_blocks, mcu_cols, mcu_rows, is_color, interval,
            );
            // Add RST marker overhead: 2 bytes per restart
            let num_restarts = (mcu_rows + interval - 1) / interval - 1;
            let rst_overhead = num_restarts as u64 * 16; // 2 bytes = 16 bits
            fixed_costs.push((interval, cost + rst_overhead));
        }
    }

    // Greedy optimal restart placement:
    // For each MCU row boundary, compute the cost savings of inserting a restart.
    // Insert if it saves more than the RST marker overhead (2 bytes = 16 bits).
    let (optimal_cost, optimal_rows) = find_optimal_restarts(
        y_blocks, cb_blocks, cr_blocks, mcu_cols, mcu_rows, is_color,
    );

    DcAnalysisResult {
        no_restart_cost: no_restart,
        fixed_restart_costs: fixed_costs,
        optimal_restart_cost: optimal_cost,
        optimal_restart_rows: optimal_rows,
        total_blocks: y_blocks.len(),
        mcu_cols,
        mcu_rows,
    }
}

fn total_dc_cost_no_restart(
    y_blocks: &[[i16; DCT_BLOCK_SIZE]],
    cb_blocks: &[[i16; DCT_BLOCK_SIZE]],
    cr_blocks: &[[i16; DCT_BLOCK_SIZE]],
    mcu_cols: usize,
    mcu_rows: usize,
    is_color: bool,
) -> u64 {
    let mut cost = 0u64;
    let mut prev_y: i16 = 0;
    let mut prev_cb: i16 = 0;
    let mut prev_cr: i16 = 0;

    for row in 0..mcu_rows {
        for col in 0..mcu_cols {
            let idx = row * mcu_cols + col;
            if idx < y_blocks.len() {
                let dc = y_blocks[idx][0];
                let cat = dc_category(dc - prev_y);
                cost += if cat == 0 { 1 } else { (2 * cat + 1) as u64 };
                prev_y = dc;
            }
            if is_color && idx < cb_blocks.len() {
                let dc_cb = cb_blocks[idx][0];
                let cat = dc_category(dc_cb - prev_cb);
                cost += if cat == 0 { 1 } else { (2 * cat + 1) as u64 };
                prev_cb = dc_cb;

                let dc_cr = cr_blocks[idx][0];
                let cat = dc_category(dc_cr - prev_cr);
                cost += if cat == 0 { 1 } else { (2 * cat + 1) as u64 };
                prev_cr = dc_cr;
            }
        }
    }

    cost
}

fn total_dc_cost_fixed_restart(
    y_blocks: &[[i16; DCT_BLOCK_SIZE]],
    cb_blocks: &[[i16; DCT_BLOCK_SIZE]],
    cr_blocks: &[[i16; DCT_BLOCK_SIZE]],
    mcu_cols: usize,
    mcu_rows: usize,
    is_color: bool,
    interval_rows: usize,
) -> u64 {
    let mut cost = 0u64;

    for band_start in (0..mcu_rows).step_by(interval_rows) {
        let band_end = (band_start + interval_rows).min(mcu_rows);
        let mut prev_y: i16 = 0;
        let mut prev_cb: i16 = 0;
        let mut prev_cr: i16 = 0;

        for row in band_start..band_end {
            for col in 0..mcu_cols {
                let idx = row * mcu_cols + col;
                if idx < y_blocks.len() {
                    let dc = y_blocks[idx][0];
                    let cat = dc_category(dc - prev_y);
                    cost += if cat == 0 { 1 } else { (2 * cat + 1) as u64 };
                    prev_y = dc;
                }
                if is_color && idx < cb_blocks.len() {
                    let dc_cb = cb_blocks[idx][0];
                    let cat = dc_category(dc_cb - prev_cb);
                    cost += if cat == 0 { 1 } else { (2 * cat + 1) as u64 };
                    prev_cb = dc_cb;

                    let dc_cr = cr_blocks[idx][0];
                    let cat = dc_category(dc_cr - prev_cr);
                    cost += if cat == 0 { 1 } else { (2 * cat + 1) as u64 };
                    prev_cr = dc_cr;
                }
            }
        }
    }

    cost
}

/// Find optimal restart marker positions using a greedy approach.
///
/// At each MCU row boundary, decide whether inserting a restart saves bits.
/// A restart resets DC prediction to 0 — this costs more if DC values are
/// close to the previous row's last DC, but saves if there's a big jump.
fn find_optimal_restarts(
    y_blocks: &[[i16; DCT_BLOCK_SIZE]],
    cb_blocks: &[[i16; DCT_BLOCK_SIZE]],
    cr_blocks: &[[i16; DCT_BLOCK_SIZE]],
    mcu_cols: usize,
    mcu_rows: usize,
    is_color: bool,
) -> (u64, Vec<usize>) {
    let rst_overhead: u64 = 16; // 2 bytes per RST marker

    let mut total_cost = 0u64;
    let mut restart_rows = Vec::new();

    // Track DC prediction state
    let mut prev_y: i16 = 0;
    let mut prev_cb: i16 = 0;
    let mut prev_cr: i16 = 0;

    for row in 0..mcu_rows {
        if row > 0 {
            // Compute cost of this row WITH continuation (no reset)
            let mut cost_continue = 0u64;
            let mut tmp_y = prev_y;
            let mut tmp_cb = prev_cb;
            let mut tmp_cr = prev_cr;
            for col in 0..mcu_cols {
                let idx = row * mcu_cols + col;
                if idx < y_blocks.len() {
                    let dc = y_blocks[idx][0];
                    let cat = dc_category(dc - tmp_y);
                    cost_continue += if cat == 0 { 1 } else { (2 * cat + 1) as u64 };
                    tmp_y = dc;
                }
                if is_color && idx < cb_blocks.len() {
                    let dc_cb = cb_blocks[idx][0];
                    let cat = dc_category(dc_cb - tmp_cb);
                    cost_continue += if cat == 0 { 1 } else { (2 * cat + 1) as u64 };
                    tmp_cb = dc_cb;

                    let dc_cr = cr_blocks[idx][0];
                    let cat = dc_category(dc_cr - tmp_cr);
                    cost_continue += if cat == 0 { 1 } else { (2 * cat + 1) as u64 };
                    tmp_cr = dc_cr;
                }
            }

            // Compute cost of this row WITH reset (restart marker)
            let mut cost_reset = rst_overhead;
            let mut tmp_y_r: i16 = 0;
            let mut tmp_cb_r: i16 = 0;
            let mut tmp_cr_r: i16 = 0;
            for col in 0..mcu_cols {
                let idx = row * mcu_cols + col;
                if idx < y_blocks.len() {
                    let dc = y_blocks[idx][0];
                    let cat = dc_category(dc - tmp_y_r);
                    cost_reset += if cat == 0 { 1 } else { (2 * cat + 1) as u64 };
                    tmp_y_r = dc;
                }
                if is_color && idx < cb_blocks.len() {
                    let dc_cb = cb_blocks[idx][0];
                    let cat = dc_category(dc_cb - tmp_cb_r);
                    cost_reset += if cat == 0 { 1 } else { (2 * cat + 1) as u64 };
                    tmp_cb_r = dc_cb;

                    let dc_cr = cr_blocks[idx][0];
                    let cat = dc_category(dc_cr - tmp_cr_r);
                    cost_reset += if cat == 0 { 1 } else { (2 * cat + 1) as u64 };
                    tmp_cr_r = dc_cr;
                }
            }

            if cost_reset < cost_continue {
                // Insert restart here
                total_cost += cost_reset;
                restart_rows.push(row);
                // Update DC state from reset path
                prev_y = 0;
                prev_cb = 0;
                prev_cr = 0;
            } else {
                total_cost += cost_continue;
            }
        }

        // Process the row to update DC prediction state
        for col in 0..mcu_cols {
            let idx = row * mcu_cols + col;
            if idx < y_blocks.len() {
                prev_y = y_blocks[idx][0];
            }
            if is_color && idx < cb_blocks.len() {
                prev_cb = cb_blocks[idx][0];
                prev_cr = cr_blocks[idx][0];
            }
        }

        // If this was the first row, add its cost
        if row == 0 {
            let mut row_cost = 0u64;
            let mut tmp_y: i16 = 0;
            let mut tmp_cb: i16 = 0;
            let mut tmp_cr: i16 = 0;
            for col in 0..mcu_cols {
                let idx = col;
                if idx < y_blocks.len() {
                    let dc = y_blocks[idx][0];
                    let cat = dc_category(dc - tmp_y);
                    row_cost += if cat == 0 { 1 } else { (2 * cat + 1) as u64 };
                    tmp_y = dc;
                }
                if is_color && idx < cb_blocks.len() {
                    let dc_cb = cb_blocks[idx][0];
                    let cat = dc_category(dc_cb - tmp_cb);
                    row_cost += if cat == 0 { 1 } else { (2 * cat + 1) as u64 };
                    tmp_cb = dc_cb;

                    let dc_cr = cr_blocks[idx][0];
                    let cat = dc_category(dc_cr - tmp_cr);
                    row_cost += if cat == 0 { 1 } else { (2 * cat + 1) as u64 };
                    tmp_cr = dc_cr;
                }
            }
            total_cost += row_cost;
        }
    }

    (total_cost, restart_rows)
}
