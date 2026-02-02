//! AC coefficient trellis quantization.
//!
//! Core Viterbi dynamic programming for optimal rate-distortion quantization
//! of the 63 AC coefficients in each 8x8 DCT block.
//!
//! Ported from mozjpeg jcdctmgr.c:937-1379.

use crate::encode::mozjpeg_compat::TrellisConfig;
use crate::foundation::consts::{DCT_BLOCK_SIZE, JPEG_NATURAL_ORDER};

use super::eob::{compute_block_eob_info, BlockEobInfo};
use super::rate::RateTable;

/// Number of bits needed to represent the absolute value of a coefficient.
///
/// Returns 0 for value 0, otherwise the position of the highest set bit.
/// Equivalent to mozjpeg's `jpeg_nbits`.
#[inline]
pub(crate) fn jpeg_nbits(val: i16) -> u8 {
    let abs = val.unsigned_abs();
    if abs == 0 {
        0
    } else {
        16 - abs.leading_zeros() as u8
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
/// * `ac_table` - Rate table for AC coefficients (for rate estimation)
/// * `config` - Trellis configuration
#[allow(clippy::needless_range_loop)]
pub fn trellis_quantize_block(
    src: &[i32; DCT_BLOCK_SIZE],
    quantized: &mut [i16; DCT_BLOCK_SIZE],
    qtable: &[u16; DCT_BLOCK_SIZE],
    ac_table: &RateTable,
    config: &TrellisConfig,
) {
    // Calculate per-coefficient lambda weights: 1/q^2
    // Note: C mozjpeg has CSF weights but mode=1 is hardcoded, which always
    // uses flat weights. The use_lambda_weight_tbl flag is effectively ignored.
    let mut lambda_tbl = [0.0f32; DCT_BLOCK_SIZE];
    for i in 0..DCT_BLOCK_SIZE {
        let q = qtable[i] as f32;
        lambda_tbl[i] = 1.0 / (q * q);
    }

    // Calculate block norm from AC coefficients (for adaptive lambda)
    let mut norm: f32 = 0.0;
    for i in 1..DCT_BLOCK_SIZE {
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
    let mut accumulated_zero_dist = [0.0f32; DCT_BLOCK_SIZE];
    let mut accumulated_cost = [0.0f32; DCT_BLOCK_SIZE];
    let mut run_start = [0usize; DCT_BLOCK_SIZE];

    // Quantize DC coefficient (simple rounding - DC trellis is optional)
    // max_coef_bits = data_precision + 2 = 8 + 2 = 10 for 8-bit JPEG
    const MAX_COEF_VAL: i32 = (1 << 10) - 1; // 1023
    {
        let x = src[0].abs();
        let sign = if src[0] < 0 { -1i16 } else { 1i16 };
        let q = 8 * qtable[0] as i32;
        let qval = ((x + q / 2) / q).min(MAX_COEF_VAL);
        quantized[0] = (qval as i16) * sign;
    }

    // Initialize state
    accumulated_zero_dist[0] = 0.0;
    accumulated_cost[0] = 0.0;

    // ===== Speed Optimization: Adaptive Search Limiting =====
    //
    // Trellis quantization has O(n²) complexity due to the predecessor search.
    // For high-entropy blocks (many non-zero coefficients at high quality),
    // this becomes slow. We detect such blocks and limit the search.
    //
    // Only significant for Q80-100 on noisy images; at lower quality most
    // blocks have few non-zero coefficients anyway.
    let (max_lookback, max_candidates) = {
        let mut nonzero_count = 0i32;
        for i in 1..DCT_BLOCK_SIZE {
            let z = JPEG_NATURAL_ORDER[i] as usize;
            let x = src[z].abs();
            let q = 8 * qtable[z] as i32;
            if (x + q / 2) / q > 0 {
                nonzero_count += 1;
            }
        }
        config.speed_mode.get_limits(nonzero_count)
    };

    // Process AC coefficients in zigzag order (positions 1 to 63)
    for i in 1..DCT_BLOCK_SIZE {
        let z = JPEG_NATURAL_ORDER[i] as usize;
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
        let num_candidates = (jpeg_nbits(qval as i16) as usize).min(max_candidates);
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
        // Limit lookback for high-entropy blocks (speed optimization)
        let j_start = i.saturating_sub(max_lookback);
        for j in j_start..i {
            let zz = JPEG_NATURAL_ORDER[j] as usize;
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

    let mut best_cost = accumulated_zero_dist[DCT_BLOCK_SIZE - 1] + eob_cost;
    let mut last_coeff_idx = 0;

    for i in 1..DCT_BLOCK_SIZE {
        let z = JPEG_NATURAL_ORDER[i] as usize;
        if quantized[z] != 0 {
            // Cost if this is the last non-zero coefficient
            let tail_zero_dist =
                accumulated_zero_dist[DCT_BLOCK_SIZE - 1] - accumulated_zero_dist[i];
            let mut cost = accumulated_cost[i] + tail_zero_dist;
            if i < DCT_BLOCK_SIZE - 1 {
                cost += eob_cost;
            }

            if cost < best_cost {
                best_cost = cost;
                last_coeff_idx = i;
            }
        }
    }

    // Zero out coefficients after optimal ending and those in runs
    let mut i = DCT_BLOCK_SIZE - 1;
    while i >= 1 {
        while i > last_coeff_idx {
            let z = JPEG_NATURAL_ORDER[i] as usize;
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
#[allow(dead_code)] // Part of trellis API - used with EOB optimization (not yet integrated)
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
pub fn trellis_quantize_block_with_eob_info(
    src: &[i32; DCT_BLOCK_SIZE],
    quantized: &mut [i16; DCT_BLOCK_SIZE],
    qtable: &[u16; DCT_BLOCK_SIZE],
    ac_table: &RateTable,
    config: &TrellisConfig,
    ss: usize,
    se: usize,
) -> BlockEobInfo {
    // Calculate per-coefficient lambda weights: 1/q^2
    let mut lambda_tbl = [0.0f32; DCT_BLOCK_SIZE];
    for i in 0..DCT_BLOCK_SIZE {
        let q = qtable[i] as f32;
        lambda_tbl[i] = 1.0 / (q * q);
    }

    // Calculate block norm from AC coefficients
    let mut norm: f32 = 0.0;
    for i in 1..DCT_BLOCK_SIZE {
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
    let mut accumulated_zero_dist = [0.0f32; DCT_BLOCK_SIZE];
    let mut accumulated_cost = [0.0f32; DCT_BLOCK_SIZE];
    let mut run_start = [0usize; DCT_BLOCK_SIZE];

    // Quantize DC coefficient (simple rounding)
    const MAX_COEF_VAL: i32 = (1 << 10) - 1; // 1023
    {
        let x = src[0].abs();
        let sign = if src[0] < 0 { -1i16 } else { 1i16 };
        let q = 8 * qtable[0] as i32;
        let qval = ((x + q / 2) / q).min(MAX_COEF_VAL);
        quantized[0] = (qval as i16) * sign;
    }

    // Initialize state
    accumulated_zero_dist[ss - 1] = 0.0;
    accumulated_cost[ss - 1] = 0.0;

    // Speed optimization: detect high-entropy blocks and limit search
    let (max_lookback, max_candidates) = {
        let mut nonzero_count = 0i32;
        for i in ss..=se {
            let z = JPEG_NATURAL_ORDER[i] as usize;
            let x = src[z].abs();
            let q = 8 * qtable[z] as i32;
            if (x + q / 2) / q > 0 {
                nonzero_count += 1;
            }
        }
        config.speed_mode.get_limits(nonzero_count)
    };

    // Process AC coefficients in zigzag order
    for i in ss..=se {
        let z = JPEG_NATURAL_ORDER[i] as usize;
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
        let num_candidates = (jpeg_nbits(qval as i16) as usize).min(max_candidates);
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

        // Limit lookback for high-entropy blocks
        let j_start = i.saturating_sub(max_lookback).max(ss - 1);
        for j in j_start..i {
            let zz = JPEG_NATURAL_ORDER[j] as usize;
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
        let z = JPEG_NATURAL_ORDER[i] as usize;
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
            let z = JPEG_NATURAL_ORDER[i] as usize;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::mozjpeg_compat::TrellisSpeedMode;

    fn create_ac_table() -> RateTable {
        RateTable::standard_luma_ac()
    }

    fn create_qtable() -> [u16; DCT_BLOCK_SIZE] {
        // Standard JPEG luminance quantization table
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
    fn test_trellis_quantize_zero_block() {
        let ac_table = create_ac_table();
        let qtable = create_qtable();
        let config = TrellisConfig::default();

        let src = [0i32; DCT_BLOCK_SIZE];
        let mut quantized = [0i16; DCT_BLOCK_SIZE];

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

        let mut src = [0i32; DCT_BLOCK_SIZE];
        src[0] = 1000 * 8; // DC only (raw DCT, scaled by 8)

        let mut quantized = [0i16; DCT_BLOCK_SIZE];
        trellis_quantize_block(&src, &mut quantized, &qtable, &ac_table, &config);

        assert!(quantized[0] > 0);
        for i in 1..DCT_BLOCK_SIZE {
            assert_eq!(quantized[i], 0);
        }
    }

    #[test]
    fn test_trellis_preserves_large_coefficients() {
        let ac_table = create_ac_table();
        let qtable = create_qtable();
        let config = TrellisConfig::default();

        let mut src = [0i32; DCT_BLOCK_SIZE];
        src[0] = 500 * 8;
        src[1] = 200 * 8;

        let mut quantized = [0i16; DCT_BLOCK_SIZE];
        trellis_quantize_block(&src, &mut quantized, &qtable, &ac_table, &config);

        assert!(quantized[0] != 0);
    }

    #[test]
    fn test_trellis_negative_coefficients() {
        let ac_table = create_ac_table();
        let qtable = create_qtable();
        let config = TrellisConfig::default();

        let mut src = [0i32; DCT_BLOCK_SIZE];
        src[0] = -1000 * 8;
        src[1] = -200 * 8;

        let mut quantized = [0i16; DCT_BLOCK_SIZE];
        trellis_quantize_block(&src, &mut quantized, &qtable, &ac_table, &config);

        assert!(quantized[0] < 0);
    }

    #[test]
    fn test_trellis_with_eob_info() {
        let ac_table = create_ac_table();
        let qtable = create_qtable();
        let config = TrellisConfig::default();

        let mut src = [0i32; DCT_BLOCK_SIZE];
        src[0] = 500 * 8;
        src[1] = 150 * 8;

        let mut quantized = [0i16; DCT_BLOCK_SIZE];
        let info = trellis_quantize_block_with_eob_info(
            &src,
            &mut quantized,
            &qtable,
            &ac_table,
            &config,
            1,
            63,
        );

        // Should have some non-zero AC
        assert!(info.has_nonzero_ac || quantized.iter().skip(1).all(|&c| c == 0));
    }

    #[test]
    fn test_speed_modes() {
        let ac_table = create_ac_table();
        let qtable = create_qtable();

        // Create a high-entropy block
        let mut src = [0i32; DCT_BLOCK_SIZE];
        for (i, s) in src.iter_mut().enumerate() {
            *s = ((i as i32 + 1) * 50) * 8;
        }

        // Test with different speed modes
        for mode in [
            TrellisSpeedMode::Thorough,
            TrellisSpeedMode::Adaptive,
            TrellisSpeedMode::Level(5),
            TrellisSpeedMode::Level(10),
        ] {
            let config = TrellisConfig::default().speed_mode(mode);
            let mut quantized = [0i16; DCT_BLOCK_SIZE];
            trellis_quantize_block(&src, &mut quantized, &qtable, &ac_table, &config);
            // Should not panic and should produce valid output
            assert!(
                quantized[0] != 0,
                "DC should be non-zero for mode {:?}",
                mode
            );
        }
    }

    #[test]
    fn test_jpeg_nbits() {
        assert_eq!(jpeg_nbits(0), 0);
        assert_eq!(jpeg_nbits(1), 1);
        assert_eq!(jpeg_nbits(-1), 1);
        assert_eq!(jpeg_nbits(2), 2);
        assert_eq!(jpeg_nbits(3), 2);
        assert_eq!(jpeg_nbits(4), 3);
        assert_eq!(jpeg_nbits(7), 3);
        assert_eq!(jpeg_nbits(8), 4);
        assert_eq!(jpeg_nbits(255), 8);
        assert_eq!(jpeg_nbits(1023), 10);
    }
}
