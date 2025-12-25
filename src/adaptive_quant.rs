//! Adaptive quantization from jpegli
//!
//! Content-aware bit allocation that adjusts quantization strength per-block
//! based on local image complexity and perceptual importance.
//!
//! This module implements jpegli's approach to adaptive quantization, which
//! excels at high quality settings by preserving detail in important areas
//! while using stronger compression in smooth regions.

/// Configuration for adaptive quantization
#[derive(Clone, Copy, Debug)]
pub struct AdaptiveQuantConfig {
    /// Enable adaptive quantization
    pub enabled: bool,
    /// Overall strength multiplier (0.0 = off, 1.0 = full strength)
    pub strength: f32,
}

impl Default for AdaptiveQuantConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            strength: 1.0,
        }
    }
}

/// Per-block adaptive quantization field
pub struct AqField {
    /// Width in blocks
    pub width_blocks: usize,
    /// Height in blocks
    pub height_blocks: usize,
    /// Per-block quantization multipliers (values around 1.0)
    /// > 1.0 means use stronger quantization (less bits)
    /// < 1.0 means use weaker quantization (more bits)
    pub multipliers: Vec<f32>,
}

impl AqField {
    /// Create a uniform AQ field (no adaptation)
    pub fn uniform(width_blocks: usize, height_blocks: usize) -> Self {
        Self {
            width_blocks,
            height_blocks,
            multipliers: vec![1.0; width_blocks * height_blocks],
        }
    }

    /// Get multiplier for a specific block
    #[inline]
    pub fn get(&self, bx: usize, by: usize) -> f32 {
        self.multipliers[by * self.width_blocks + bx]
    }

    /// Set multiplier for a specific block
    #[inline]
    pub fn set(&mut self, bx: usize, by: usize, value: f32) {
        self.multipliers[by * self.width_blocks + bx] = value;
    }
}

/// Compute adaptive quantization field from Y plane
///
/// This is a simplified version of jpegli's algorithm. The full version uses:
/// - Pre-erosion to find local minima
/// - Fuzzy erosion for smoothing
/// - Per-block modulations based on local variance
/// - Gamma and high-frequency modulations
pub fn compute_aq_field(
    y_plane: &[u8],
    width: usize,
    height: usize,
    config: &AdaptiveQuantConfig,
) -> AqField {
    if !config.enabled {
        let wb = (width + 7) / 8;
        let hb = (height + 7) / 8;
        return AqField::uniform(wb, hb);
    }

    let width_blocks = (width + 7) / 8;
    let height_blocks = (height + 7) / 8;

    let mut field = AqField::uniform(width_blocks, height_blocks);

    // Compute per-block statistics
    for by in 0..height_blocks {
        for bx in 0..width_blocks {
            let multiplier = compute_block_aq_multiplier(y_plane, width, height, bx, by, config);
            field.set(bx, by, multiplier);
        }
    }

    field
}

/// Compute AQ multiplier for a single block
fn compute_block_aq_multiplier(
    y_plane: &[u8],
    width: usize,
    height: usize,
    bx: usize,
    by: usize,
    config: &AdaptiveQuantConfig,
) -> f32 {
    let block_x = bx * 8;
    let block_y = by * 8;

    // Compute local statistics
    let mut sum = 0u32;
    let mut sum_sq = 0u64;
    let mut count = 0u32;

    for dy in 0..8 {
        let y = block_y + dy;
        if y >= height {
            continue;
        }

        for dx in 0..8 {
            let x = block_x + dx;
            if x >= width {
                continue;
            }

            let val = y_plane[y * width + x] as u32;
            sum += val;
            sum_sq += (val * val) as u64;
            count += 1;
        }
    }

    if count == 0 {
        return 1.0;
    }

    // Variance = E[X^2] - E[X]^2
    let mean = sum as f64 / count as f64;
    let variance = (sum_sq as f64 / count as f64) - mean * mean;
    let std_dev = variance.sqrt() as f32;

    // Convert variance to AQ multiplier
    // High variance = high detail = lower multiplier (more bits)
    // Low variance = smooth area = higher multiplier (fewer bits)

    // jpegli uses a complex formula; this is a simplified approximation
    let base_multiplier = if std_dev < 5.0 {
        // Very smooth - use stronger quantization
        1.0 + 0.3 * config.strength
    } else if std_dev < 20.0 {
        // Moderate detail - neutral
        1.0
    } else {
        // High detail - use weaker quantization
        1.0 - 0.2 * config.strength
    };

    base_multiplier.clamp(0.5, 1.5)
}

/// Apply adaptive quantization to a quantization table for a specific block
pub fn apply_aq_to_quant(base_quant: &[u16; 64], aq_multiplier: f32) -> [u16; 64] {
    let mut output = [0u16; 64];

    for i in 0..64 {
        let scaled = (base_quant[i] as f32 * aq_multiplier).round() as u16;
        output[i] = scaled.clamp(1, 255);
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_field() {
        let field = AqField::uniform(10, 10);
        assert_eq!(field.multipliers.len(), 100);
        assert_eq!(field.get(5, 5), 1.0);
    }

    #[test]
    fn test_smooth_block_gets_higher_multiplier() {
        // Create a smooth block (uniform value)
        let smooth_block: Vec<u8> = vec![128; 64];
        let config = AdaptiveQuantConfig::default();

        let mult = compute_block_aq_multiplier(&smooth_block, 8, 8, 0, 0, &config);

        // Smooth blocks should get higher multiplier (stronger quant)
        assert!(mult > 1.0, "Smooth block multiplier: {}", mult);
    }

    #[test]
    fn test_detailed_block_gets_lower_multiplier() {
        // Create a high-detail block (alternating values)
        let mut detailed_block = Vec::with_capacity(64);
        for i in 0..64 {
            detailed_block.push(if i % 2 == 0 { 50 } else { 200 });
        }

        let config = AdaptiveQuantConfig::default();
        let mult = compute_block_aq_multiplier(&detailed_block, 8, 8, 0, 0, &config);

        // Detailed blocks should get lower multiplier (weaker quant)
        assert!(mult < 1.0, "Detailed block multiplier: {}", mult);
    }
}
