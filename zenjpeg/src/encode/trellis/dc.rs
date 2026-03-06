//! DC coefficient trellis optimization.
//!
//! Optimizes DC coefficients across multiple blocks using dynamic programming.
//! DC trellis explores multiple candidate values for each block's DC coefficient
//! and finds the optimal path that minimizes rate + distortion.
//!
//! Ported from mozjpeg jcdctmgr.c DC trellis optimization.

use crate::foundation::consts::DCT_BLOCK_SIZE;

use super::ac::jpeg_nbits;
use super::rate::RateTable;

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
/// * `dc_table` - Rate table for DC coefficients
/// * `last_dc` - Previous DC value for differential encoding
/// * `lambda_log_scale1` - Lambda scale parameter 1
/// * `lambda_log_scale2` - Lambda scale parameter 2
///
/// # Returns
/// The final DC value (for next component's last_dc)
pub fn dc_trellis_optimize(
    raw_dct_blocks: &[[i32; DCT_BLOCK_SIZE]],
    quantized_blocks: &mut [[i16; DCT_BLOCK_SIZE]],
    dc_quantval: u16,
    dc_table: &RateTable,
    last_dc: i16,
    lambda_log_scale1: f32,
    lambda_log_scale2: f32,
) -> i16 {
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
        0.0,
        None,
    )
}

/// DC trellis optimization with explicit block indices.
///
/// This allows processing blocks in any order (e.g., row order for proper
/// C mozjpeg parity) while they may be stored in a different order.
///
/// # Arguments
/// * `raw_dct_blocks` - All raw DCT blocks (may be in any storage order)
/// * `quantized_blocks` - All quantized blocks (same storage order as raw_dct_blocks)
/// * `indices` - Indices into the block arrays specifying processing order
/// * `delta_dc_weight` - Weight for vertical DC gradient consideration (0.0 = disabled)
/// * `above_row_data` - Raw and quantized DC values from the row above
/// * Other arguments same as `dc_trellis_optimize`
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
pub fn dc_trellis_optimize_indexed(
    raw_dct_blocks: &[[i32; DCT_BLOCK_SIZE]],
    quantized_blocks: &mut [[i16; DCT_BLOCK_SIZE]],
    indices: &[usize],
    dc_quantval: u16,
    dc_table: &RateTable,
    last_dc: i16,
    lambda_log_scale1: f32,
    lambda_log_scale2: f32,
    delta_dc_weight: f32,
    above_row_data: Option<(&[i32], &[i16])>,
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
    let mut accumulated_dc_cost = vec![vec![0.0f32; num_blocks]; num_candidates];
    let mut dc_cost_backtrack = vec![vec![0usize; num_blocks]; num_candidates];
    let mut dc_candidate = vec![vec![0i16; num_blocks]; num_candidates];

    for bi in 0..num_blocks {
        let block_idx = indices[bi];
        let raw_dc = raw_dct_blocks[block_idx][0];
        let x = raw_dc.abs();
        let sign = if raw_dc < 0 { -1i16 } else { 1i16 };

        // Calculate lambda for this block
        let mut norm: f32 = 0.0;
        for i in 1..DCT_BLOCK_SIZE {
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
            let mut candidate_dist = (delta as f32).powi(2) * lambda_dc;

            // Store the signed candidate value
            dc_candidate[k][bi] = (candidate_val as i16) * sign;

            // Take into account DC differences with row above
            if delta_dc_weight > 0.0
                && let Some((raw_dc_above, quantized_dc_above)) = above_row_data
            {
                let dc_above_orig = raw_dc_above[bi];
                let dc_above_recon = quantized_dc_above[bi] as i32 * q;
                let dc_orig = raw_dct_blocks[block_idx][0];
                let dc_recon = dc_candidate[k][bi] as i32 * q;
                let vdelta = (dc_above_orig - dc_orig) - (dc_above_recon - dc_recon);
                let vertical_dist = (vdelta as f32).powi(2) * lambda_dc;
                candidate_dist += delta_dc_weight * (vertical_dist - candidate_dist);
            }

            if bi == 0 {
                // First block: cost is based on difference from last_dc
                let dc_delta = dc_candidate[k][bi] - last_dc;
                let bits = jpeg_nbits(dc_delta);
                let (_, code_size) = dc_table.get_code(bits);
                let rate = if code_size > 0 {
                    bits as usize + code_size as usize
                } else {
                    bits as usize * 2 + 1
                };
                accumulated_dc_cost[k][0] = rate as f32 + candidate_dist;
                dc_cost_backtrack[k][0] = 0;
            } else {
                // Subsequent blocks: try all previous candidates
                let mut best_cost = f32::MAX;
                let mut best_prev = 0;

                for l in 0..num_candidates {
                    let dc_delta = dc_candidate[k][bi] - dc_candidate[l][bi - 1];
                    let bits = jpeg_nbits(dc_delta);
                    let (_, code_size) = dc_table.get_code(bits);
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

    dc_candidate[best_final][num_blocks - 1]
}

/// Quantize a block with simple rounding (no trellis optimization).
///
/// Used when trellis is disabled.
/// Note: Input should be raw DCT output (scaled by 8).
pub fn simple_quantize_block(
    src: &[i32; DCT_BLOCK_SIZE],
    quantized: &mut [i16; DCT_BLOCK_SIZE],
    qtable: &[u16; DCT_BLOCK_SIZE],
) {
    const MAX_COEF_VAL: i32 = (1 << 10) - 1; // 1023
    for i in 0..DCT_BLOCK_SIZE {
        let x = src[i];
        let q = 8 * qtable[i] as i32;
        let sign = if x < 0 { -1 } else { 1 };
        let qval = ((x.abs() + q / 2) / q).min(MAX_COEF_VAL);
        quantized[i] = (sign * qval) as i16;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_dc_table() -> RateTable {
        RateTable::standard_luma_dc()
    }

    fn create_qtable() -> [u16; DCT_BLOCK_SIZE] {
        #[rustfmt::skip]
        let table: [u16; 64] = [
            16, 11, 10, 16,  24,  40,  51,  61,
            12, 12, 14, 19,  26,  58,  60,  55,
            14, 13, 16, 24,  40,  57,  69,  56,
            14, 17, 22, 29,  51,  87,  80,  62,
            18, 22, 37, 56,  68, 109, 103,  77,
            24, 35, 55, 64,  81, 104, 113,  92,
            49, 64, 78, 87, 103, 121, 120, 101,
            72, 92, 95, 98, 112, 100, 103,  99,
        ];
        table
    }

    #[test]
    fn test_simple_quantize_block() {
        let qtable = create_qtable();
        let mut src = [0i32; DCT_BLOCK_SIZE];
        src[0] = 1000 * 8;
        src[1] = 100 * 8;

        let mut quantized = [0i16; DCT_BLOCK_SIZE];
        simple_quantize_block(&src, &mut quantized, &qtable);

        assert_eq!(quantized[0], 63); // 1000 / 16 = 62.5 -> 63
        assert!(quantized[1] > 0);
    }

    #[test]
    fn test_simple_quantize_negative() {
        let qtable = create_qtable();
        let mut src = [0i32; DCT_BLOCK_SIZE];
        src[0] = -1000 * 8;
        src[1] = -100 * 8;

        let mut quantized = [0i16; DCT_BLOCK_SIZE];
        simple_quantize_block(&src, &mut quantized, &qtable);

        assert!(quantized[0] < 0);
        assert!(quantized[1] < 0);
    }

    #[test]
    fn test_dc_trellis_single_block() {
        let dc_table = create_dc_table();
        let qtable = create_qtable();

        let mut raw = [[0i32; DCT_BLOCK_SIZE]; 1];
        raw[0][0] = 500 * 8 * qtable[0] as i32; // Scale appropriately
        let mut quantized = [[0i16; DCT_BLOCK_SIZE]; 1];
        simple_quantize_block(&raw[0], &mut quantized[0], &qtable);

        let last_dc =
            dc_trellis_optimize(&raw, &mut quantized, qtable[0], &dc_table, 0, 14.75, 16.5);

        // Should produce a valid DC value
        assert!(quantized[0][0].unsigned_abs() <= 1023);
        let _ = last_dc;
    }

    #[test]
    fn test_dc_trellis_chain() {
        let dc_table = create_dc_table();
        let qtable = create_qtable();

        let num_blocks = 8;
        let mut raw = vec![[0i32; DCT_BLOCK_SIZE]; num_blocks];
        for (i, block) in raw.iter_mut().enumerate() {
            block[0] = ((i as i32 + 1) * 200) * 8;
            block[1] = 50 * 8;
        }

        let mut quantized = vec![[0i16; DCT_BLOCK_SIZE]; num_blocks];
        for (i, block) in raw.iter().enumerate() {
            simple_quantize_block(block, &mut quantized[i], &qtable);
        }

        let last_dc =
            dc_trellis_optimize(&raw, &mut quantized, qtable[0], &dc_table, 0, 14.75, 16.5);

        // All DC values should be valid
        for block in &quantized {
            assert!(block[0].unsigned_abs() <= 1023);
        }
        let _ = last_dc;
    }

    #[test]
    fn test_dc_trellis_empty_blocks() {
        let dc_table = create_dc_table();
        let raw: &[[i32; DCT_BLOCK_SIZE]] = &[];
        let quantized: &mut [[i16; DCT_BLOCK_SIZE]] = &mut [];

        let last_dc = dc_trellis_optimize(raw, quantized, 16, &dc_table, 42, 14.75, 16.5);
        assert_eq!(last_dc, 42); // Should return input last_dc
    }

    #[test]
    fn test_delta_dc_weight_zero_matches_baseline() {
        let dc_table = create_dc_table();
        let qtable = create_qtable();
        let dc_quantval = qtable[0];
        let num_cols = 4;

        let mut raw_blocks = Vec::new();
        for row in 0..2 {
            for col in 0..num_cols {
                let mut block = [0i32; DCT_BLOCK_SIZE];
                block[0] = ((row * num_cols + col) as i32 * 300 + 100) * 8;
                raw_blocks.push(block);
            }
        }

        // Baseline with weight=0.0 and fake above data
        let mut quantized_baseline = vec![[0i16; DCT_BLOCK_SIZE]; raw_blocks.len()];
        for (i, block) in raw_blocks.iter().enumerate() {
            simple_quantize_block(block, &mut quantized_baseline[i], &qtable);
        }
        let indices_row0: Vec<usize> = (0..num_cols).collect();
        let indices_row1: Vec<usize> = (num_cols..2 * num_cols).collect();

        let last_dc = dc_trellis_optimize_indexed(
            &raw_blocks,
            &mut quantized_baseline,
            &indices_row0,
            dc_quantval,
            &dc_table,
            0,
            14.75,
            16.0,
            0.0,
            None,
        );
        let fake_above_raw = vec![999i32; num_cols];
        let fake_above_quant = vec![99i16; num_cols];
        dc_trellis_optimize_indexed(
            &raw_blocks,
            &mut quantized_baseline,
            &indices_row1,
            dc_quantval,
            &dc_table,
            last_dc,
            14.75,
            16.0,
            0.0,
            Some((&fake_above_raw, &fake_above_quant)),
        );

        // Comparison with None above data
        let mut quantized_no_above = vec![[0i16; DCT_BLOCK_SIZE]; raw_blocks.len()];
        for (i, block) in raw_blocks.iter().enumerate() {
            simple_quantize_block(block, &mut quantized_no_above[i], &qtable);
        }
        let last_dc2 = dc_trellis_optimize_indexed(
            &raw_blocks,
            &mut quantized_no_above,
            &indices_row0,
            dc_quantval,
            &dc_table,
            0,
            14.75,
            16.0,
            0.0,
            None,
        );
        dc_trellis_optimize_indexed(
            &raw_blocks,
            &mut quantized_no_above,
            &indices_row1,
            dc_quantval,
            &dc_table,
            last_dc2,
            14.75,
            16.0,
            0.0,
            None,
        );

        for i in 0..raw_blocks.len() {
            assert_eq!(
                quantized_baseline[i], quantized_no_above[i],
                "Block {} differs with delta_dc_weight=0.0",
                i
            );
        }
    }

    #[test]
    fn test_get_num_dc_trellis_candidates() {
        // High quality (small quant) = more candidates
        assert!(get_num_dc_trellis_candidates(1) >= 5);
        // Low quality (large quant) = fewer candidates
        assert!(get_num_dc_trellis_candidates(100) >= 2);
        // Never exceeds max
        assert!(get_num_dc_trellis_candidates(1) <= DC_TRELLIS_MAX_CANDIDATES);
        // Always odd
        for q in [1, 2, 4, 8, 16, 32, 64, 128] {
            let n = get_num_dc_trellis_candidates(q);
            assert_eq!(n % 2, 1, "Candidates should be odd for q={}", q);
        }
    }
}
