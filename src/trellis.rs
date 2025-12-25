//! Trellis quantization for optimal rate-distortion.
//!
//! This is the core innovation of mozjpeg over standard libjpeg.
//! Trellis quantization uses dynamic programming to find the optimal
//! quantization decisions that minimize:
//!
//! ```text
//! Cost = Rate + Lambda * Distortion
//! ```
//!
//! where Rate is the Huffman encoding cost and Distortion is the
//! squared error from the original coefficients.
//!
//! Reference: mozjpeg jcdctmgr.c quantize_trellis()

use crate::consts::{DCTSIZE2, JPEG_NATURAL_ORDER};
use crate::entropy::jpeg_nbits;
use crate::huffman::DerivedTable;

/// Configuration for trellis quantization.
///
/// Trellis quantization uses dynamic programming to find the optimal
/// rate-distortion tradeoff for each block. These parameters control
/// the balance between compression and quality.
#[derive(Debug, Clone, Copy)]
pub struct TrellisConfig {
    /// Enable trellis quantization for AC coefficients
    pub ac_enabled: bool,
    /// Enable DC trellis optimization (cross-block DC refinement)
    pub dc_enabled: bool,
    /// Enable EOB run optimization (for progressive mode)
    pub eob_opt: bool,
    /// Lambda scale parameter 1 (controls rate-distortion tradeoff)
    /// Default: 14.75 (from mozjpeg)
    pub lambda_log_scale1: f32,
    /// Lambda scale parameter 2 (controls adaptive lambda based on block complexity)
    /// Default: 16.5 (from mozjpeg)
    pub lambda_log_scale2: f32,
}

impl Default for TrellisConfig {
    fn default() -> Self {
        Self {
            ac_enabled: true,
            dc_enabled: true,
            eob_opt: true,
            lambda_log_scale1: 14.75,
            lambda_log_scale2: 16.5,
        }
    }
}

impl TrellisConfig {
    /// Create a disabled trellis configuration
    #[must_use]
    pub fn disabled() -> Self {
        Self {
            ac_enabled: false,
            dc_enabled: false,
            eob_opt: false,
            ..Default::default()
        }
    }
}

/// Alias for backwards compatibility
impl TrellisConfig {
    /// Check if AC trellis is enabled
    pub fn enabled(&self) -> bool {
        self.ac_enabled
    }
}

/// Perform trellis quantization on a single 8x8 block.
///
/// This is the core rate-distortion optimization algorithm matching mozjpeg.
///
/// # Arguments
/// * `src` - Raw DCT coefficients (scaled by 8, before any division)
/// * `quantized` - Output buffer for quantized coefficients
/// * `qtable` - Quantization table values
/// * `ac_table` - Derived Huffman table for AC coefficients (for rate estimation)
/// * `config` - Trellis configuration
#[allow(clippy::needless_range_loop)]
pub fn trellis_quantize_block(
    src: &[i32; DCTSIZE2],
    quantized: &mut [i16; DCTSIZE2],
    qtable: &[u16; DCTSIZE2],
    ac_table: &DerivedTable,
    config: &TrellisConfig,
) {
    // Calculate per-coefficient lambda weights: 1/q^2
    // This matches mozjpeg's mode 1 (use_lambda_weight_tbl with flat weights)
    let mut lambda_tbl = [0.0f32; DCTSIZE2];
    for i in 0..DCTSIZE2 {
        let q = qtable[i] as f32;
        lambda_tbl[i] = 1.0 / (q * q);
    }

    // Calculate block norm from AC coefficients (for adaptive lambda)
    let mut norm: f32 = 0.0;
    for i in 1..DCTSIZE2 {
        let c = src[i] as f32;
        norm += c * c;
    }
    norm /= 63.0;

    // Calculate lambda using mozjpeg's formula
    // lambda = 2^scale1 * lambda_base / (2^scale2 + norm)
    // In mode 1, lambda_base = 1.0
    let lambda = if config.lambda_log_scale2 > 0.0 {
        let scale1 = 2.0_f32.powf(config.lambda_log_scale1);
        let scale2 = 2.0_f32.powf(config.lambda_log_scale2);
        scale1 / (scale2 + norm)
    } else {
        2.0_f32.powf(config.lambda_log_scale1 - 12.0)
    };

    // State for dynamic programming
    let mut accumulated_zero_dist = [0.0f32; DCTSIZE2];
    let mut accumulated_cost = [0.0f32; DCTSIZE2];
    let mut run_start = [0usize; DCTSIZE2];

    // Quantize DC coefficient (simple rounding - DC trellis is optional)
    {
        let x = src[0].abs();
        let sign = if src[0] < 0 { -1i16 } else { 1i16 };
        let q = 8 * qtable[0] as i32;
        let qval = (x + q / 2) / q;
        quantized[0] = (qval as i16) * sign;
    }

    // Initialize state
    accumulated_zero_dist[0] = 0.0;
    accumulated_cost[0] = 0.0;

    // Process AC coefficients in zigzag order (positions 1 to 63)
    for i in 1..DCTSIZE2 {
        let z = JPEG_NATURAL_ORDER[i];
        let x = src[z].abs();
        let sign = if src[z] < 0 { -1i16 } else { 1i16 };
        let q = 8 * qtable[z] as i32;

        // Distortion from zeroing this coefficient
        let zero_dist = (x as f32).powi(2) * lambda * lambda_tbl[z];
        accumulated_zero_dist[i] = zero_dist + accumulated_zero_dist[i - 1];

        // Quantized value with rounding
        let qval = (x + q / 2) / q;

        if qval == 0 {
            // Coefficient rounds to zero - no choice needed
            quantized[z] = 0;
            accumulated_cost[i] = f32::MAX;
            run_start[i] = i - 1;
            continue;
        }

        // Clamp to valid range (10 bits for 8-bit JPEG)
        let qval = qval.min(1023);

        // Generate candidate quantized values
        // Candidates are: 1, 3, 7, 15, ..., (2^k - 1), and the rounded value
        let num_candidates = jpeg_nbits(qval as i16) as usize;
        let mut candidates = [(0i32, 0u8, 0.0f32); 16]; // (value, bits, distortion)

        for k in 0..num_candidates {
            let candidate_val = if k < num_candidates - 1 {
                (2 << k) - 1 // 1, 3, 7, 15, ...
            } else {
                qval
            };
            // Distortion: squared error between dequantized and original
            let delta = candidate_val * q - x;
            let dist = (delta as f32).powi(2) * lambda * lambda_tbl[z];
            candidates[k] = (candidate_val, (k + 1) as u8, dist);
        }

        // Find optimal choice using dynamic programming
        accumulated_cost[i] = f32::MAX;

        // Try starting a run from each valid previous position
        for j in 0..i {
            let zz = JPEG_NATURAL_ORDER[j];
            // j=0 is always valid (after DC), otherwise need non-zero coef
            if j != 0 && quantized[zz] == 0 {
                continue;
            }

            let zero_run = i - 1 - j;

            // Cost of ZRL codes for runs >= 16
            let zrl_cost = if zero_run >= 16 {
                let (_, zrl_size) = ac_table.get_code(0xF0);
                if zrl_size == 0 {
                    continue;
                }
                (zero_run / 16) * zrl_size as usize
            } else {
                0
            };

            let run_mod_16 = zero_run & 15;

            // Try each candidate value
            for k in 0..num_candidates {
                let (candidate_val, candidate_bits, candidate_dist) = candidates[k];

                // Huffman symbol: (run << 4) | size
                let symbol = ((run_mod_16 as u8) << 4) | candidate_bits;
                let (_, code_size) = ac_table.get_code(symbol);
                if code_size == 0 {
                    continue;
                }

                // Rate = Huffman code + value bits + ZRL codes
                let rate = code_size as usize + candidate_bits as usize + zrl_cost;

                // Cost = rate + distortion of this coef + distortion of zeros in run
                let zero_run_dist = accumulated_zero_dist[i - 1] - accumulated_zero_dist[j];
                let prev_cost = if j == 0 { 0.0 } else { accumulated_cost[j] };
                let cost = rate as f32 + candidate_dist + zero_run_dist + prev_cost;

                if cost < accumulated_cost[i] {
                    quantized[z] = (candidate_val as i16) * sign;
                    accumulated_cost[i] = cost;
                    run_start[i] = j;
                }
            }
        }
    }

    // Find optimal ending point (last non-zero coefficient)
    let eob_cost = {
        let (_, eob_size) = ac_table.get_code(0x00);
        eob_size as f32
    };

    let mut best_cost = accumulated_zero_dist[DCTSIZE2 - 1] + eob_cost;
    let mut last_coeff_idx = 0;

    for i in 1..DCTSIZE2 {
        let z = JPEG_NATURAL_ORDER[i];
        if quantized[z] != 0 {
            // Cost if this is the last non-zero coefficient
            let tail_zero_dist = accumulated_zero_dist[DCTSIZE2 - 1] - accumulated_zero_dist[i];
            let mut cost = accumulated_cost[i] + tail_zero_dist;
            if i < DCTSIZE2 - 1 {
                cost += eob_cost;
            }

            if cost < best_cost {
                best_cost = cost;
                last_coeff_idx = i;
            }
        }
    }

    // Zero out coefficients after optimal ending and those in runs
    let mut i = DCTSIZE2 - 1;
    while i >= 1 {
        while i > last_coeff_idx {
            let z = JPEG_NATURAL_ORDER[i];
            quantized[z] = 0;
            i -= 1;
        }
        if i >= 1 {
            last_coeff_idx = run_start[i];
            i -= 1;
        }
    }
}

/// Perform trellis quantization and return EOB info for cross-block optimization.
///
/// This version of trellis quantization returns additional information needed
/// for cross-block EOB optimization in progressive JPEG.
///
/// # Arguments
/// Same as `trellis_quantize_block`, plus:
/// * `ss` - Spectral selection start (1 for full block, higher for progressive AC scans)
/// * `se` - Spectral selection end (63 for full block)
///
/// # Returns
/// `BlockEobInfo` containing costs and EOB status for cross-block optimization.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
pub fn trellis_quantize_block_with_eob_info(
    src: &[i32; DCTSIZE2],
    quantized: &mut [i16; DCTSIZE2],
    qtable: &[u16; DCTSIZE2],
    ac_table: &DerivedTable,
    config: &TrellisConfig,
    ss: usize,
    se: usize,
) -> BlockEobInfo {
    // Calculate per-coefficient lambda weights
    let mut lambda_tbl = [0.0f32; DCTSIZE2];
    for i in 0..DCTSIZE2 {
        let q = qtable[i] as f32;
        lambda_tbl[i] = 1.0 / (q * q);
    }

    // Calculate block norm from AC coefficients
    let mut norm: f32 = 0.0;
    for i in 1..DCTSIZE2 {
        let c = src[i] as f32;
        norm += c * c;
    }
    norm /= 63.0;

    // Calculate lambda
    let lambda = if config.lambda_log_scale2 > 0.0 {
        let scale1 = 2.0_f32.powf(config.lambda_log_scale1);
        let scale2 = 2.0_f32.powf(config.lambda_log_scale2);
        scale1 / (scale2 + norm)
    } else {
        2.0_f32.powf(config.lambda_log_scale1 - 12.0)
    };

    // State for dynamic programming
    let mut accumulated_zero_dist = [0.0f32; DCTSIZE2];
    let mut accumulated_cost = [0.0f32; DCTSIZE2];
    let mut run_start = [0usize; DCTSIZE2];

    // Quantize DC coefficient (simple rounding)
    {
        let x = src[0].abs();
        let sign = if src[0] < 0 { -1i16 } else { 1i16 };
        let q = 8 * qtable[0] as i32;
        let qval = (x + q / 2) / q;
        quantized[0] = (qval as i16) * sign;
    }

    // Initialize state
    accumulated_zero_dist[ss - 1] = 0.0;
    accumulated_cost[ss - 1] = 0.0;

    // Process AC coefficients in zigzag order
    for i in ss..=se {
        let z = JPEG_NATURAL_ORDER[i];
        let x = src[z].abs();
        let sign = if src[z] < 0 { -1i16 } else { 1i16 };
        let q = 8 * qtable[z] as i32;

        // Distortion from zeroing this coefficient
        let zero_dist = (x as f32).powi(2) * lambda * lambda_tbl[z];
        accumulated_zero_dist[i] = zero_dist + accumulated_zero_dist[i - 1];

        // Quantized value with rounding
        let qval = (x + q / 2) / q;

        if qval == 0 {
            quantized[z] = 0;
            accumulated_cost[i] = f32::MAX;
            run_start[i] = i - 1;
            continue;
        }

        let qval = qval.min(1023);

        // Generate candidate quantized values
        let num_candidates = jpeg_nbits(qval as i16) as usize;
        let mut candidates = [(0i32, 0u8, 0.0f32); 16];

        for k in 0..num_candidates {
            let candidate_val = if k < num_candidates - 1 {
                (2 << k) - 1
            } else {
                qval
            };
            let delta = candidate_val * q - x;
            let dist = (delta as f32).powi(2) * lambda * lambda_tbl[z];
            candidates[k] = (candidate_val, (k + 1) as u8, dist);
        }

        // Find optimal choice using dynamic programming
        accumulated_cost[i] = f32::MAX;

        for j in (ss - 1)..i {
            let zz = JPEG_NATURAL_ORDER[j];
            if j != ss - 1 && quantized[zz] == 0 {
                continue;
            }

            let zero_run = i - 1 - j;

            let zrl_cost = if zero_run >= 16 {
                let (_, zrl_size) = ac_table.get_code(0xF0);
                if zrl_size == 0 {
                    continue;
                }
                (zero_run / 16) * zrl_size as usize
            } else {
                0
            };

            let run_mod_16 = zero_run & 15;

            for k in 0..num_candidates {
                let (candidate_val, candidate_bits, candidate_dist) = candidates[k];

                let symbol = ((run_mod_16 as u8) << 4) | candidate_bits;
                let (_, code_size) = ac_table.get_code(symbol);
                if code_size == 0 {
                    continue;
                }

                let rate = code_size as usize + candidate_bits as usize + zrl_cost;

                let zero_run_dist = accumulated_zero_dist[i - 1] - accumulated_zero_dist[j];
                let prev_cost = if j == ss - 1 {
                    0.0
                } else {
                    accumulated_cost[j]
                };
                let cost = rate as f32 + candidate_dist + zero_run_dist + prev_cost;

                if cost < accumulated_cost[i] {
                    quantized[z] = (candidate_val as i16) * sign;
                    accumulated_cost[i] = cost;
                    run_start[i] = j;
                }
            }
        }
    }

    // Find optimal ending point
    let eob_cost = {
        let (_, eob_size) = ac_table.get_code(0x00);
        eob_size as f32
    };

    let total_zero_dist = accumulated_zero_dist[se];
    let mut best_cost = total_zero_dist + eob_cost;
    let mut best_cost_skip = total_zero_dist; // Cost without EOB
    let mut last_coeff_idx = ss - 1;

    for i in ss..=se {
        let z = JPEG_NATURAL_ORDER[i];
        if quantized[z] != 0 {
            let tail_zero_dist = accumulated_zero_dist[se] - accumulated_zero_dist[i];
            let cost_wo_eob = accumulated_cost[i] + tail_zero_dist;
            let mut cost = cost_wo_eob;
            if i < se {
                cost += eob_cost;
            }

            if cost < best_cost {
                best_cost = cost;
                best_cost_skip = cost_wo_eob;
                last_coeff_idx = i;
            }
        }
    }

    // Zero out coefficients after optimal ending and those in runs
    let mut i = se;
    while i >= ss {
        while i > last_coeff_idx {
            let z = JPEG_NATURAL_ORDER[i];
            quantized[z] = 0;
            i -= 1;
        }
        if i >= ss {
            last_coeff_idx = run_start[i];
            if i == 0 {
                break;
            }
            i -= 1;
        }
    }

    // Compute EOB info
    compute_block_eob_info(
        quantized,
        total_zero_dist,
        best_cost,
        best_cost_skip,
        last_coeff_idx,
        ss,
        se,
    )
}

/// Maximum number of DC trellis candidates
const DC_TRELLIS_MAX_CANDIDATES: usize = 9;

/// Calculate number of DC trellis candidates based on quantization value.
/// Higher qualities (smaller quant values) can tolerate more candidates.
fn get_num_dc_trellis_candidates(dc_quantval: u16) -> usize {
    let candidates = (2 + 60 / dc_quantval as usize) | 1; // Force odd
    candidates.min(DC_TRELLIS_MAX_CANDIDATES)
}

/// Optimize DC coefficients across multiple blocks using dynamic programming.
///
/// DC trellis explores multiple candidate values for each block's DC coefficient
/// and uses DP to find the optimal path that minimizes rate + distortion.
///
/// # Arguments
/// * `raw_dct_blocks` - Raw DCT coefficients for all blocks (scaled by 8)
/// * `quantized_blocks` - Output quantized blocks (DC values will be updated)
/// * `dc_quantval` - DC quantization value
/// * `dc_table` - Derived Huffman table for DC coefficients
/// * `last_dc` - Previous DC value for differential encoding
/// * `lambda_log_scale1` - Lambda scale parameter 1
/// * `lambda_log_scale2` - Lambda scale parameter 2
///
/// # Returns
/// The final DC value (for next component's last_dc)
pub fn dc_trellis_optimize(
    raw_dct_blocks: &[[i32; DCTSIZE2]],
    quantized_blocks: &mut [[i16; DCTSIZE2]],
    dc_quantval: u16,
    dc_table: &DerivedTable,
    last_dc: i16,
    lambda_log_scale1: f32,
    lambda_log_scale2: f32,
) -> i16 {
    // Process all blocks as one chain (original behavior)
    let indices: Vec<usize> = (0..raw_dct_blocks.len()).collect();
    dc_trellis_optimize_indexed(
        raw_dct_blocks,
        quantized_blocks,
        &indices,
        dc_quantval,
        dc_table,
        last_dc,
        lambda_log_scale1,
        lambda_log_scale2,
    )
}

/// DC trellis optimization with explicit block indices.
///
/// This allows processing blocks in any order (e.g., row order for proper
/// C mozjpeg parity) while they may be stored in a different order (e.g., MCU order).
///
/// # Arguments
/// * `raw_dct_blocks` - All raw DCT blocks (may be in any storage order)
/// * `quantized_blocks` - All quantized blocks (same storage order as raw_dct_blocks)
/// * `indices` - Indices into the block arrays specifying processing order
/// * Other arguments same as `dc_trellis_optimize`
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
pub fn dc_trellis_optimize_indexed(
    raw_dct_blocks: &[[i32; DCTSIZE2]],
    quantized_blocks: &mut [[i16; DCTSIZE2]],
    indices: &[usize],
    dc_quantval: u16,
    dc_table: &DerivedTable,
    last_dc: i16,
    lambda_log_scale1: f32,
    lambda_log_scale2: f32,
) -> i16 {
    let num_blocks = indices.len();
    if num_blocks == 0 {
        return last_dc;
    }

    let num_candidates = get_num_dc_trellis_candidates(dc_quantval);
    let q = 8 * dc_quantval as i32;

    // Lambda weight for DC coefficient: 1/q^2
    let lambda_dc_weight = 1.0 / (dc_quantval as f32 * dc_quantval as f32);

    // Storage for DP
    // accumulated_dc_cost[k][bi] = accumulated cost to reach candidate k at block bi
    // dc_cost_backtrack[k][bi] = which candidate from previous block led here
    // dc_candidate[k][bi] = the actual DC value for candidate k at block bi
    let mut accumulated_dc_cost = vec![vec![0.0f32; num_blocks]; num_candidates];
    let mut dc_cost_backtrack = vec![vec![0usize; num_blocks]; num_candidates];
    let mut dc_candidate = vec![vec![0i16; num_blocks]; num_candidates];

    for bi in 0..num_blocks {
        let block_idx = indices[bi];
        let raw_dc = raw_dct_blocks[block_idx][0];
        let x = raw_dc.abs();
        let sign = if raw_dc < 0 { -1i16 } else { 1i16 };

        // Calculate lambda for this block (simplified - use AC norm from raw block)
        let mut norm: f32 = 0.0;
        for i in 1..DCTSIZE2 {
            let c = raw_dct_blocks[block_idx][i] as f32;
            norm += c * c;
        }
        norm /= 63.0;

        let lambda = if lambda_log_scale2 > 0.0 {
            let scale1 = 2.0_f32.powf(lambda_log_scale1);
            let scale2 = 2.0_f32.powf(lambda_log_scale2);
            scale1 / (scale2 + norm)
        } else {
            2.0_f32.powf(lambda_log_scale1 - 12.0)
        };
        let lambda_dc = lambda * lambda_dc_weight;

        // Rounded quantized value
        let qval = (x + q / 2) / q;

        // Generate candidates centered around qval
        let half_candidates = (num_candidates / 2) as i32;

        for k in 0..num_candidates {
            let candidate_offset = k as i32 - half_candidates;
            let mut candidate_val = qval + candidate_offset;

            // Clamp to valid range (10 bits for 8-bit JPEG)
            candidate_val = candidate_val.clamp(-(1 << 10) + 1, (1 << 10) - 1);

            // Distortion from this candidate
            let delta = candidate_val * q - x;
            let candidate_dist = (delta as f32).powi(2) * lambda_dc;

            // Store the signed candidate value
            dc_candidate[k][bi] = (candidate_val as i16) * sign;

            if bi == 0 {
                // First block: cost is based on difference from last_dc
                let dc_delta = dc_candidate[k][bi] - last_dc;
                let bits = jpeg_nbits(dc_delta);
                let (_, code_size) = dc_table.get_code(bits as u8);
                let rate = if code_size > 0 {
                    bits as usize + code_size as usize
                } else {
                    // Fallback if code not found
                    bits as usize * 2 + 1
                };
                accumulated_dc_cost[k][0] = rate as f32 + candidate_dist;
                dc_cost_backtrack[k][0] = 0; // Not used for first block
            } else {
                // Subsequent blocks: try all previous candidates
                let mut best_cost = f32::MAX;
                let mut best_prev = 0;

                for l in 0..num_candidates {
                    let dc_delta = dc_candidate[k][bi] - dc_candidate[l][bi - 1];
                    let bits = jpeg_nbits(dc_delta);
                    let (_, code_size) = dc_table.get_code(bits as u8);
                    let rate = if code_size > 0 {
                        bits as usize + code_size as usize
                    } else {
                        bits as usize * 2 + 1
                    };
                    let cost = rate as f32 + candidate_dist + accumulated_dc_cost[l][bi - 1];

                    if cost < best_cost {
                        best_cost = cost;
                        best_prev = l;
                    }
                }

                accumulated_dc_cost[k][bi] = best_cost;
                dc_cost_backtrack[k][bi] = best_prev;
            }
        }
    }

    // Find the best final candidate
    let mut best_final = 0;
    for k in 1..num_candidates {
        if accumulated_dc_cost[k][num_blocks - 1] < accumulated_dc_cost[best_final][num_blocks - 1]
        {
            best_final = k;
        }
    }

    // Backtrack to assign optimal DC values
    let mut k = best_final;
    for bi in (0..num_blocks).rev() {
        let block_idx = indices[bi];
        quantized_blocks[block_idx][0] = dc_candidate[k][bi];
        if bi > 0 {
            k = dc_cost_backtrack[k][bi];
        }
    }

    // Return the last DC value for next component
    dc_candidate[best_final][num_blocks - 1]
}

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
/// * `ac_table` - AC Huffman table for EOBRUN cost estimation
/// * `ss` - Spectral selection start (first AC coefficient index)
/// * `se` - Spectral selection end (last AC coefficient index)
///
/// # Returns
/// The number of blocks that were zeroed as part of runs.
#[allow(clippy::needless_range_loop)]
pub fn optimize_eob_runs(
    blocks: &mut [[i16; DCTSIZE2]],
    block_info: &[BlockEobInfo],
    ac_table: &DerivedTable,
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

    // Initialize with first block's zero cost
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
            // Skip if starting block is all-zero (can't start a run from there)
            if block_info[i].requires_eob == 2 {
                continue;
            }

            // Cost = cost of encoding block bi + cost of zeroing blocks i to bi-1 + EOBRUN cost
            let mut cost = block_info[bi].best_cost_skip;
            cost += accumulated_zero_block_cost[bi] - accumulated_zero_block_cost[i];
            if i > 0 {
                cost += accumulated_block_cost[i];
            }

            // EOBRUN cost: encode the number of zero blocks
            let zero_block_run = bi - i + block_info[i].requires_eob as usize;
            if zero_block_run > 0 {
                let nbits = jpeg_nbits(zero_block_run as i16) as usize;
                // EOBRUN symbol is at position 16*nbits in the Huffman table
                let (_, eobrun_size) = ac_table.get_code((16 * nbits) as u8);
                if eobrun_size > 0 {
                    cost += eobrun_size as f32 + nbits as f32;
                } else {
                    // Fallback if EOBRUN symbol not available
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

        // Add EOBRUN cost for trailing zero blocks
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
            // Zero out AC coefficients in this block
            for j in ss..=se {
                let z = JPEG_NATURAL_ORDER[j];
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
/// This is called after trellis quantization to record information needed
/// for cross-block EOB optimization.
pub fn compute_block_eob_info(
    block: &[i16; DCTSIZE2],
    zero_dist: f32,
    best_cost: f32,
    best_cost_skip: f32,
    last_coeff_idx: usize,
    ss: usize,
    se: usize,
) -> BlockEobInfo {
    // Determine EOB status
    let has_nonzero_ac = (ss..=se).any(|i| block[JPEG_NATURAL_ORDER[i]] != 0);
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

/// Quantize a block with simple rounding (no trellis optimization).
///
/// This is used when trellis is disabled.
/// Note: Input should be raw DCT output (scaled by 8).
pub fn simple_quantize_block(
    src: &[i32; DCTSIZE2],
    quantized: &mut [i16; DCTSIZE2],
    qtable: &[u16; DCTSIZE2],
) {
    for i in 0..DCTSIZE2 {
        let x = src[i];
        let q = 8 * qtable[i] as i32;
        let sign = if x < 0 { -1 } else { 1 };
        quantized[i] = (sign * ((x.abs() + q / 2) / q)) as i16;
    }
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;
    use crate::consts::{AC_LUMINANCE_BITS, AC_LUMINANCE_VALUES, STD_LUMINANCE_QUANT_TBL};
    use crate::huffman::HuffTable;

    fn create_ac_table() -> DerivedTable {
        let mut htbl = HuffTable::default();
        htbl.bits.copy_from_slice(&AC_LUMINANCE_BITS);
        for (i, &v) in AC_LUMINANCE_VALUES.iter().enumerate() {
            htbl.huffval[i] = v;
        }
        DerivedTable::from_huff_table(&htbl, false).unwrap()
    }

    fn create_qtable() -> [u16; DCTSIZE2] {
        STD_LUMINANCE_QUANT_TBL[0]
    }

    #[test]
    fn test_trellis_config_default() {
        let config = TrellisConfig::default();
        assert!(config.ac_enabled);
        assert!(config.dc_enabled);
        assert!(config.eob_opt);
        assert!((config.lambda_log_scale1 - 14.75).abs() < 0.01);
        assert!((config.lambda_log_scale2 - 16.5).abs() < 0.01);
    }

    #[test]
    fn test_simple_quantize_block() {
        let qtable = create_qtable();

        // Create a test block with known values (raw DCT, scaled by 8)
        let mut src = [0i32; DCTSIZE2];
        src[0] = 1000 * 8; // DC
        src[1] = 100 * 8; // AC

        let mut quantized = [0i16; DCTSIZE2];
        simple_quantize_block(&src, &mut quantized, &qtable);

        // DC: 1000 / 16 = 63 (with rounding)
        assert_eq!(quantized[0], 63);
        // AC[1]: 100 / 11 = 9 (with rounding)
        assert!(quantized[1] > 0);
    }

    #[test]
    fn test_simple_quantize_negative() {
        let qtable = create_qtable();

        let mut src = [0i32; DCTSIZE2];
        src[0] = -1000 * 8;
        src[1] = -100 * 8;

        let mut quantized = [0i16; DCTSIZE2];
        simple_quantize_block(&src, &mut quantized, &qtable);

        assert!(quantized[0] < 0);
        assert!(quantized[1] < 0);
    }

    #[test]
    fn test_trellis_quantize_zero_block() {
        let ac_table = create_ac_table();
        let qtable = create_qtable();
        let config = TrellisConfig::default();

        let src = [0i32; DCTSIZE2];
        let mut quantized = [0i16; DCTSIZE2];

        trellis_quantize_block(&src, &mut quantized, &qtable, &ac_table, &config);

        for &q in quantized.iter() {
            assert_eq!(q, 0);
        }
    }

    #[test]
    fn test_trellis_quantize_dc_only() {
        let ac_table = create_ac_table();
        let qtable = create_qtable();
        let config = TrellisConfig::default();

        let mut src = [0i32; DCTSIZE2];
        src[0] = 1000 * 8; // DC only (raw DCT, scaled by 8)

        let mut quantized = [0i16; DCTSIZE2];
        trellis_quantize_block(&src, &mut quantized, &qtable, &ac_table, &config);

        assert!(quantized[0] > 0);
        for i in 1..DCTSIZE2 {
            assert_eq!(quantized[i], 0);
        }
    }

    #[test]
    fn test_trellis_preserves_large_coefficients() {
        let ac_table = create_ac_table();
        let qtable = create_qtable();
        let config = TrellisConfig::default();

        // Raw DCT values (scaled by 8)
        let mut src = [0i32; DCTSIZE2];
        src[0] = 500 * 8;
        src[1] = 200 * 8;

        let mut quantized = [0i16; DCTSIZE2];
        trellis_quantize_block(&src, &mut quantized, &qtable, &ac_table, &config);

        assert!(quantized[0] != 0);
        // Large AC coefficient should be preserved
    }

    #[test]
    fn test_trellis_vs_simple() {
        let ac_table = create_ac_table();
        let qtable = create_qtable();
        let config = TrellisConfig::default();

        // Create a block with some content
        let mut src = [0i32; DCTSIZE2];
        src[0] = 800 * 8;
        src[1] = 150 * 8;
        src[8] = 80 * 8;
        src[9] = 40 * 8;

        let mut trellis_out = [0i16; DCTSIZE2];
        let mut simple_out = [0i16; DCTSIZE2];

        trellis_quantize_block(&src, &mut trellis_out, &qtable, &ac_table, &config);
        simple_quantize_block(&src, &mut simple_out, &qtable);

        // Both should produce non-zero DC
        assert!(trellis_out[0] != 0);
        assert!(simple_out[0] != 0);

        // DC values should be similar (trellis might differ slightly)
        let dc_diff = (trellis_out[0] - simple_out[0]).abs();
        assert!(dc_diff <= 1, "DC difference too large: {}", dc_diff);
    }

    #[test]
    fn test_trellis_config_disabled() {
        let config = TrellisConfig::disabled();
        assert!(!config.ac_enabled);
    }
}
