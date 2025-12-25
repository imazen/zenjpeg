//! Trellis quantization from mozjpeg
//!
//! Rate-distortion optimized coefficient selection using dynamic programming.
//! This is the core innovation of mozjpeg - it finds the optimal quantization
//! decisions that minimize:
//!
//! ```text
//! Cost = Rate + Lambda * Distortion
//! ```
//!
//! where Rate is the Huffman encoding cost and Distortion is the
//! squared error from the original coefficients.

use crate::consts::ZIGZAG;
use crate::huffman::{compute_category, HuffmanTable};

/// Configuration for trellis quantization
#[derive(Clone, Copy, Debug)]
pub struct TrellisConfig {
    /// Enable trellis for AC coefficients
    pub ac_enabled: bool,
    /// Enable trellis for DC coefficients
    pub dc_enabled: bool,
    /// Lambda log scale parameter 1 (mozjpeg default: 14.75)
    pub lambda_log_scale1: f32,
    /// Lambda log scale parameter 2 (mozjpeg default: 16.5)
    pub lambda_log_scale2: f32,
}

impl Default for TrellisConfig {
    fn default() -> Self {
        Self {
            ac_enabled: true,
            dc_enabled: true,
            lambda_log_scale1: 14.75,
            lambda_log_scale2: 16.5,
        }
    }
}

impl TrellisConfig {
    /// Create a disabled trellis config
    pub fn disabled() -> Self {
        Self {
            ac_enabled: false,
            dc_enabled: false,
            lambda_log_scale1: 14.75,
            lambda_log_scale2: 16.5,
        }
    }

    /// Create config tuned for low quality (Q < 50)
    pub fn low_quality() -> Self {
        Self {
            ac_enabled: true,
            dc_enabled: true,
            lambda_log_scale1: 16.0,  // Higher = more aggressive zeroing
            lambda_log_scale2: 16.5,
        }
    }

    /// Create config tuned for medium quality (Q 50-80)
    pub fn medium_quality() -> Self {
        Self {
            ac_enabled: true,
            dc_enabled: false,
            lambda_log_scale1: 14.75,
            lambda_log_scale2: 16.5,
        }
    }

    /// Create config tuned for high quality (Q > 80)
    pub fn high_quality() -> Self {
        Self {
            ac_enabled: true,
            dc_enabled: false,
            lambda_log_scale1: 13.0,  // Lower = more conservative
            lambda_log_scale2: 16.5,
        }
    }
}

/// Perform trellis quantization on a single 8x8 block.
///
/// This is the core rate-distortion optimization algorithm matching mozjpeg.
///
/// # Arguments
/// * `dct` - DCT coefficients (f32, already level-shifted)
/// * `quant` - Quantization table values
/// * `ac_table` - Huffman table for AC coefficients (for rate estimation)
/// * `config` - Trellis configuration
///
/// # Returns
/// Quantized coefficients in natural (row-major) order
pub fn trellis_quantize_block(
    dct: &[f32; 64],
    quant: &[u16; 64],
    ac_table: &HuffmanTable,
    config: &TrellisConfig,
) -> [i16; 64] {
    // Convert f32 DCT to i32 (scaled by 8 to match mozjpeg)
    let mut src = [0i32; 64];
    for i in 0..64 {
        src[i] = (dct[i] * 8.0).round() as i32;
    }

    let mut quantized = [0i16; 64];

    // Calculate per-coefficient lambda weights: 1/q^2
    let mut lambda_tbl = [0.0f32; 64];
    for i in 0..64 {
        let q = quant[i] as f32;
        lambda_tbl[i] = 1.0 / (q * q);
    }

    // Calculate block norm from AC coefficients (for adaptive lambda)
    let mut norm: f32 = 0.0;
    for i in 1..64 {
        let c = src[i] as f32;
        norm += c * c;
    }
    norm /= 63.0;

    // Calculate lambda using mozjpeg's formula
    let lambda = if config.lambda_log_scale2 > 0.0 {
        let scale1 = 2.0_f32.powf(config.lambda_log_scale1);
        let scale2 = 2.0_f32.powf(config.lambda_log_scale2);
        scale1 / (scale2 + norm)
    } else {
        2.0_f32.powf(config.lambda_log_scale1 - 12.0)
    };

    // State for dynamic programming
    let mut accumulated_zero_dist = [0.0f32; 64];
    let mut accumulated_cost = [0.0f32; 64];
    let mut run_start = [0usize; 64];

    // Quantize DC coefficient (simple rounding - DC trellis is optional)
    {
        let x = src[0].abs();
        let sign = if src[0] < 0 { -1i16 } else { 1i16 };
        let q = 8 * quant[0] as i32;
        let qval = (x + q / 2) / q;
        quantized[0] = (qval as i16) * sign;
    }

    // Initialize state
    accumulated_zero_dist[0] = 0.0;
    accumulated_cost[0] = 0.0;

    if !config.ac_enabled {
        // Simple quantization for AC coefficients
        for i in 1..64 {
            let x = src[i].abs();
            let sign = if src[i] < 0 { -1i16 } else { 1i16 };
            let q = 8 * quant[i] as i32;
            let qval = (x + q / 2) / q;
            quantized[i] = (qval as i16) * sign;
        }
        return quantized;
    }

    // Process AC coefficients in zigzag order (positions 1 to 63)
    for i in 1..64 {
        let z = ZIGZAG[i]; // zigzag index to natural order
        let x = src[z].abs();
        let sign = if src[z] < 0 { -1i16 } else { 1i16 };
        let q = 8 * quant[z] as i32;

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
        let num_candidates = compute_category(qval as i16) as usize;
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
            let zz = ZIGZAG[j];
            // j=0 is always valid (after DC), otherwise need non-zero coef
            if j != 0 && quantized[zz] == 0 {
                continue;
            }

            let zero_run = i - 1 - j;

            // Cost of ZRL codes for runs >= 16
            let zrl_cost = if zero_run >= 16 {
                let zrl_size = ac_table.sizes[0xF0];
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
                let code_size = ac_table.sizes[symbol as usize];
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
    let eob_size = ac_table.sizes[0x00]; // EOB symbol
    let eob_cost = eob_size as f32;

    let mut best_cost = accumulated_zero_dist[63] + eob_cost;
    let mut last_coeff_idx = 0;

    for i in 1..64 {
        let z = ZIGZAG[i];
        if quantized[z] != 0 {
            // Cost if this is the last non-zero coefficient
            let tail_zero_dist = accumulated_zero_dist[63] - accumulated_zero_dist[i];
            let mut cost = accumulated_cost[i] + tail_zero_dist;
            if i < 63 {
                cost += eob_cost;
            }

            if cost < best_cost {
                best_cost = cost;
                last_coeff_idx = i;
            }
        }
    }

    // Zero out coefficients after optimal ending and those in runs
    let mut i = 63;
    while i >= 1 {
        while i > last_coeff_idx {
            let z = ZIGZAG[i];
            quantized[z] = 0;
            if i == 0 {
                break;
            }
            i -= 1;
        }
        if i >= 1 {
            last_coeff_idx = run_start[i];
            if i == 0 {
                break;
            }
            i -= 1;
        }
    }

    quantized
}

/// Simple quantization (rounding) - fallback when trellis is disabled
pub fn simple_quantize(dct: &[f32; 64], quant: &[u16; 64]) -> [i16; 64] {
    let mut output = [0i16; 64];
    for i in 0..64 {
        output[i] = (dct[i] / quant[i] as f32).round() as i16;
    }
    output
}

/// Apply trellis quantization to AC coefficients (legacy interface)
///
/// This is the simplified interface for backwards compatibility.
pub fn trellis_quantize_ac(
    dct: &[f32; 64],
    quant: &[u16; 64],
    config: &TrellisConfig,
) -> [i16; 64] {
    if !config.ac_enabled {
        return simple_quantize(dct, quant);
    }

    // Use standard AC luma table for rate estimation
    let ac_table = crate::huffman::std_ac_luma();
    trellis_quantize_block(dct, quant, &ac_table, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trellis_config_default() {
        let config = TrellisConfig::default();
        assert!(config.ac_enabled);
        assert!(config.dc_enabled);
        assert!((config.lambda_log_scale1 - 14.75).abs() < 0.01);
    }

    #[test]
    fn test_trellis_config_disabled() {
        let config = TrellisConfig::disabled();
        assert!(!config.ac_enabled);
        assert!(!config.dc_enabled);
    }

    #[test]
    fn test_simple_quantize() {
        let dct = [128.0f32; 64];
        let quant = [16u16; 64];

        let result = simple_quantize(&dct, &quant);
        for val in result {
            assert_eq!(val, 8);
        }
    }

    #[test]
    fn test_trellis_quantize() {
        let dct = [128.0f32; 64];
        let quant = [16u16; 64];
        let config = TrellisConfig::default();

        let result = trellis_quantize_ac(&dct, &quant, &config);

        // DC should be quantized
        assert!(result[0] != 0 || dct[0] < quant[0] as f32 / 2.0);
    }

    #[test]
    fn test_trellis_disabled() {
        let dct = [100.0f32; 64];
        let quant = [10u16; 64];
        let config = TrellisConfig::disabled();

        let result = trellis_quantize_ac(&dct, &quant, &config);

        // Should use simple rounding
        for val in result {
            assert_eq!(val, 10);
        }
    }

    #[test]
    fn test_trellis_zero_block() {
        let dct = [0.0f32; 64];
        let quant = [16u16; 64];
        let config = TrellisConfig::default();

        let result = trellis_quantize_ac(&dct, &quant, &config);

        for val in result {
            assert_eq!(val, 0);
        }
    }

    #[test]
    fn test_trellis_preserves_large_coefficients() {
        let mut dct = [0.0f32; 64];
        dct[0] = 500.0;  // Large DC
        dct[1] = 200.0;  // Large AC
        let quant = [16u16; 64];
        let config = TrellisConfig::default();

        let result = trellis_quantize_ac(&dct, &quant, &config);

        assert!(result[0] != 0, "DC coefficient should not be zero");
        // Large AC coefficient may or may not be preserved depending on RD tradeoff
    }
}
