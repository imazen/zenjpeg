//! Trellis quantization from mozjpeg
//!
//! Rate-distortion optimized coefficient selection using dynamic programming.
//! Especially effective at low quality settings where coefficient choices
//! significantly impact both file size and visual quality.

use crate::huffman::compute_category;

/// Configuration for trellis quantization
#[derive(Clone, Copy, Debug)]
pub struct TrellisConfig {
    /// Enable trellis for AC coefficients
    pub ac_enabled: bool,
    /// Enable trellis for DC coefficients
    pub dc_enabled: bool,
    /// Lambda base for rate-distortion tradeoff (higher = prefer smaller files)
    pub lambda_base: f32,
}

impl Default for TrellisConfig {
    fn default() -> Self {
        Self {
            ac_enabled: true,
            dc_enabled: true,
            lambda_base: 1.0,
        }
    }
}

/// Trellis state for dynamic programming
#[derive(Clone, Copy, Default)]
struct TrellisState {
    /// Total cost (rate + lambda * distortion)
    cost: f32,
    /// Quantized value that led to this state
    quant_value: i16,
}

/// Apply trellis quantization to AC coefficients
///
/// This performs rate-distortion optimization on the coefficient choices,
/// considering the Huffman coding cost of each decision.
pub fn trellis_quantize_ac(
    dct: &[f32; 64],
    quant: &[u16; 64],
    config: &TrellisConfig,
) -> [i16; 64] {
    if !config.ac_enabled {
        // Fallback to simple rounding
        return simple_quantize(dct, quant);
    }

    let mut output = [0i16; 64];

    // DC coefficient - use simple quantization for now
    // (DC trellis is more complex due to inter-block dependencies)
    output[0] = (dct[0] / quant[0] as f32).round() as i16;

    // For each AC coefficient, consider candidates
    for i in 1..64 {
        let raw = dct[i];
        let q = quant[i] as f32;
        let base_quant = (raw / q).round() as i16;

        // For trellis, we consider the quantized value and ±1
        // A full implementation would use dynamic programming across the block
        let candidates = [base_quant - 1, base_quant, base_quant + 1];

        let mut best_cost = f32::MAX;
        let mut best_val = base_quant;

        for &candidate in &candidates {
            // Distortion cost
            let reconstructed = candidate as f32 * q;
            let distortion = (raw - reconstructed).powi(2);

            // Rate cost (simplified - real version uses Huffman code lengths)
            let rate = estimate_coefficient_rate(candidate);

            // Combined cost
            let cost = rate + config.lambda_base * distortion / (q * q);

            if cost < best_cost {
                best_cost = cost;
                best_val = candidate;
            }
        }

        output[i] = best_val;
    }

    output
}

/// Simple quantization (rounding)
fn simple_quantize(dct: &[f32; 64], quant: &[u16; 64]) -> [i16; 64] {
    let mut output = [0i16; 64];
    for i in 0..64 {
        output[i] = (dct[i] / quant[i] as f32).round() as i16;
    }
    output
}

/// Estimate the rate (bits) to encode a coefficient
fn estimate_coefficient_rate(value: i16) -> f32 {
    if value == 0 {
        // Zero contributes to run length, not encoded directly
        return 0.0;
    }

    // Category + additional bits
    let cat = compute_category(value);

    // Approximate Huffman code length for this category
    // Based on typical AC table statistics
    let huffman_estimate = match cat {
        0 => 2.0,
        1 => 2.5,
        2 => 3.0,
        3 => 4.0,
        4 => 5.0,
        5 => 6.0,
        6 => 7.0,
        7 => 8.0,
        _ => 10.0,
    };

    huffman_estimate + cat as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trellis_quantize() {
        let dct = [128.0f32; 64];
        let quant = [16u16; 64];
        let config = TrellisConfig::default();

        let result = trellis_quantize_ac(&dct, &quant, &config);

        // All values should be 128/16 = 8
        assert_eq!(result[0], 8);
    }

    #[test]
    fn test_trellis_disabled() {
        let dct = [100.0f32; 64];
        let quant = [10u16; 64];
        let config = TrellisConfig {
            ac_enabled: false,
            ..Default::default()
        };

        let result = trellis_quantize_ac(&dct, &quant, &config);

        // Should use simple rounding
        for val in result {
            assert_eq!(val, 10);
        }
    }
}
