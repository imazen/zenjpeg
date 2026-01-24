//! Overshoot Deringing for JPEG encoding.
//!
//! This module implements mozjpeg's overshoot deringing algorithm, which reduces visible
//! ringing artifacts near sharp edges, particularly on white backgrounds.
//!
//! # The Problem
//!
//! JPEG compression uses DCT (Discrete Cosine Transform) which represents 8x8 pixel blocks
//! as sums of cosine waves. Hard edges, like a sharp transition from gray to white, create
//! high-frequency components that are difficult to represent accurately with limited bits.
//!
//! The result is "ringing" - oscillating artifacts near sharp edges that look like halos
//! or waves emanating from the edge. This is especially visible on white backgrounds where
//! the ringing appears as faint gray bands.
//!
//! # The Insight
//!
//! JPEG decoders always clamp output values to the valid range (0-255). This means:
//! - To display white (255), any encoded value ≥ 255 will work after clamping
//! - The encoder can use values outside the displayable range as "headroom"
//!
//! From a DCT perspective, hard edges are like "clipped" waveforms (square waves).
//! Audio engineers know that clipping creates harmonics. Similarly, JPEG struggles with
//! the high frequencies needed to represent perfectly square transitions.
//!
//! # The Solution
//!
//! Instead of encoding a flat plateau at the maximum value, we can create a smooth curve
//! that "overshoots" above the maximum. When decoded:
//! - The peak of the curve (above 255) gets clamped to 255
//! - The result looks identical to the original flat region
//! - But the smooth curve compresses much better than a hard edge!
//!
//! This is similar to "anti-clipping" algorithms in audio processing.
//!
//! # Algorithm Overview
//!
//! 1. **Identify clipped regions**: Find runs of pixels at the maximum value (255)
//!    in the centered sample space (127 after level shift)
//!
//! 2. **Calculate safe overshoot**: Determine how much we can overshoot without:
//!    - Increasing DC too much (would cost bits)
//!    - Causing decoder overflow issues
//!
//! 3. **Smooth the edges**: Replace flat max-value runs with Catmull-Rom spline
//!    interpolation that creates a smooth curve peaking above the max
//!
//! 4. **The result**: Identical visual output after decoding, but better compression
//!
//! # When It Helps
//!
//! - Images with white backgrounds
//! - Text and graphics with hard edges
//! - Any image with saturated regions (pixels at 0 or 255)
//!
//! # References
//!
//! - mozjpeg source: `jcdctmgr.c:416-550`
//! - Related concepts: Anti-clipping, soft-clipping, waveform shaping

use crate::foundation::consts::{DCT_BLOCK_SIZE, JPEG_NATURAL_ORDER};
use crate::foundation::simd_types::Block8x8f;

/// Maximum sample value after level shift (255 - 128 = 127).
const MAX_SAMPLE: f32 = 127.0;

/// Threshold for detecting "at max" pixels (slightly below max to handle float precision).
const MAX_SAMPLE_THRESHOLD: f32 = 126.5;

/// Catmull-Rom spline interpolation (f32 version).
///
/// Interpolates between `value2` and `value3` based on parameter `t` (0.0 to 1.0).
/// `value1` and `value4` are used to determine the tangent slopes at the endpoints.
///
/// The `size` parameter scales the tangents for the length of the interpolated segment.
#[inline]
fn catmull_rom_f32(value1: f32, value2: f32, value3: f32, value4: f32, t: f32, size: f32) -> f32 {
    // Tangents at the endpoints, scaled by segment size
    let tan1 = (value3 - value1) * size;
    let tan2 = (value4 - value2) * size;

    // Hermite basis functions
    let t2 = t * t;
    let t3 = t2 * t;

    let f1 = 2.0 * t3 - 3.0 * t2 + 1.0; // h00: value at t=0
    let f2 = -2.0 * t3 + 3.0 * t2; // h01: value at t=1
    let f3 = t3 - 2.0 * t2 + t; // h10: tangent at t=0
    let f4 = t3 - t2; // h11: tangent at t=1

    value2 * f1 + tan1 * f3 + value3 * f2 + tan2 * f4
}

/// Preprocess an 8x8 block (f32) to reduce ringing artifacts on white backgrounds.
///
/// This function should be called on level-shifted sample data (samples centered around 0)
/// BEFORE the DCT transform. It modifies pixels at the maximum value to create smooth
/// transitions that compress better while producing identical visual output after decoding.
///
/// # Arguments
/// * `data` - Mutable slice of 64 level-shifted f32 samples (-128 to +127). Modified in place.
/// * `dc_quant` - DC quantization value from the quantization table (used to limit overshoot)
///
/// # Algorithm
///
/// 1. Count pixels at max value and calculate their sum
/// 2. If no max pixels or all max pixels, return unchanged
/// 3. Calculate safe overshoot limit based on DC headroom
/// 4. For each run of max-value pixels (in zigzag order):
///    - Calculate slopes from neighboring pixels
///    - Apply Catmull-Rom interpolation to create smooth curve
///    - Clamp peak values to the overshoot limit
pub fn preprocess_deringing_f32(data: &mut [f32; DCT_BLOCK_SIZE], dc_quant: u16) {
    // Calculate sum and count of max-value pixels
    let mut sum: f32 = 0.0;
    let mut max_sample_count: usize = 0;

    for &sample in data.iter() {
        sum += sample;
        if sample >= MAX_SAMPLE_THRESHOLD {
            max_sample_count += 1;
        }
    }

    // If nothing reaches max value, there's nothing to overshoot.
    // If the block is completely at max value, it's already the best case.
    if max_sample_count == 0 || max_sample_count == DCT_BLOCK_SIZE {
        return;
    }

    // Calculate maximum safe overshoot:
    // 1. Don't overshoot more than 31 (arbitrary reasonable limit)
    // 2. Don't overshoot more than 2x the DC quantization (cost/benefit)
    // 3. Stay within DC headroom to avoid overflow
    let dc_limit = 2.0 * dc_quant as f32;
    let headroom = (MAX_SAMPLE * DCT_BLOCK_SIZE as f32 - sum) / max_sample_count as f32;
    let max_overshoot: f32 = MAX_SAMPLE + 31.0_f32.min(dc_limit).min(headroom);

    // Process pixels in zigzag (natural) order
    let mut n: usize = 0;

    while n < DCT_BLOCK_SIZE {
        // Skip pixels that aren't at max value
        if data[JPEG_NATURAL_ORDER[n] as usize] < MAX_SAMPLE_THRESHOLD {
            n += 1;
            continue;
        }

        // Found a max-value pixel; find the extent of this run
        let start = n;
        while n + 1 < DCT_BLOCK_SIZE
            && data[JPEG_NATURAL_ORDER[n + 1] as usize] >= MAX_SAMPLE_THRESHOLD
        {
            n += 1;
        }
        let end = n + 1; // end is exclusive

        // Get values around the edges of the run for slope calculation
        let f1 = data[JPEG_NATURAL_ORDER[start.saturating_sub(1)] as usize];
        let f2 = data[JPEG_NATURAL_ORDER[start.saturating_sub(2)] as usize];

        let l1 = data[JPEG_NATURAL_ORDER[end.min(DCT_BLOCK_SIZE - 1)] as usize];
        let l2 = data[JPEG_NATURAL_ORDER[(end + 1).min(DCT_BLOCK_SIZE - 1)] as usize];

        // Calculate upward slopes at the edges
        let mut fslope = (f1 - f2).max(MAX_SAMPLE - f1);
        let mut lslope = (l1 - l2).max(MAX_SAMPLE - l1);

        // If at the start/end of the block, make the curve symmetric
        if start == 0 {
            fslope = lslope;
        }
        if end == DCT_BLOCK_SIZE {
            lslope = fslope;
        }

        // Apply Catmull-Rom interpolation across the run
        let length = (end - start) as f32;
        let step = 1.0 / (length + 1.0);
        let mut position = step;

        for i in start..end {
            let interpolated = catmull_rom_f32(
                MAX_SAMPLE - fslope,
                MAX_SAMPLE,
                MAX_SAMPLE,
                MAX_SAMPLE - lslope,
                position,
                length,
            );

            data[JPEG_NATURAL_ORDER[i] as usize] = interpolated.ceil().min(max_overshoot);
            position += step;
        }

        n += 1;
    }
}

/// Apply deringing to a Block8x8f (level-shifted f32 block).
///
/// This is a convenience wrapper that extracts the data from Block8x8f,
/// applies deringing, and writes it back.
///
/// # Arguments
/// * `block` - Mutable reference to a Block8x8f containing level-shifted samples
/// * `dc_quant` - DC quantization value from the quantization table
#[inline]
pub fn preprocess_deringing_block(block: &mut Block8x8f, dc_quant: u16) {
    // Convert Block8x8f to flat array
    let mut data = [0.0f32; DCT_BLOCK_SIZE];
    for (row_idx, row) in block.rows.iter().enumerate() {
        let arr: [f32; 8] = (*row).into();
        data[row_idx * 8..row_idx * 8 + 8].copy_from_slice(&arr);
    }

    // Apply deringing
    preprocess_deringing_f32(&mut data, dc_quant);

    // Write back to Block8x8f
    for (row_idx, row) in block.rows.iter_mut().enumerate() {
        let slice: [f32; 8] = data[row_idx * 8..row_idx * 8 + 8].try_into().unwrap();
        *row = slice.into();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_catmull_rom_midpoint() {
        // At t=0.5, should be roughly at the midpoint value
        let result = catmull_rom_f32(100.0, 110.0, 120.0, 130.0, 0.5, 1.0);
        assert!(
            (result - 115.0).abs() < 1.0,
            "Expected ~115, got {}",
            result
        );
    }

    #[test]
    fn test_catmull_rom_endpoints() {
        // At t=0, should be close to value2
        let result0 = catmull_rom_f32(100.0, 110.0, 120.0, 130.0, 0.0, 1.0);
        assert!(
            (result0 - 110.0).abs() < 0.1,
            "Expected 110, got {}",
            result0
        );

        // At t=1, should be close to value3
        let result1 = catmull_rom_f32(100.0, 110.0, 120.0, 130.0, 1.0, 1.0);
        assert!(
            (result1 - 120.0).abs() < 0.1,
            "Expected 120, got {}",
            result1
        );
    }

    #[test]
    fn test_deringing_no_max_pixels() {
        // Block with no pixels at max value - should be unchanged
        let mut data = [64.0f32; DCT_BLOCK_SIZE];
        let original = data;

        preprocess_deringing_f32(&mut data, 16);

        assert_eq!(
            data, original,
            "Block with no max pixels should be unchanged"
        );
    }

    #[test]
    fn test_deringing_all_max_pixels() {
        // Block with all pixels at max value - should be unchanged
        let mut data = [MAX_SAMPLE; DCT_BLOCK_SIZE];
        let original = data;

        preprocess_deringing_f32(&mut data, 16);

        assert_eq!(
            data, original,
            "Block with all max pixels should be unchanged"
        );
    }

    #[test]
    fn test_deringing_creates_overshoot() {
        // Create a block with a run of max pixels in the middle
        let mut data = [0.0f32; DCT_BLOCK_SIZE];

        // Set some pixels to max value (indices 10-15 in natural order)
        for i in 10..16 {
            data[JPEG_NATURAL_ORDER[i] as usize] = MAX_SAMPLE;
        }
        // Set surrounding pixels to create a slope
        data[JPEG_NATURAL_ORDER[8] as usize] = 80.0;
        data[JPEG_NATURAL_ORDER[9] as usize] = 100.0;
        data[JPEG_NATURAL_ORDER[16] as usize] = 100.0;
        data[JPEG_NATURAL_ORDER[17] as usize] = 80.0;

        preprocess_deringing_f32(&mut data, 16);

        // Check that some overshoot occurred
        let mut has_overshoot = false;
        for i in 10..16 {
            if data[JPEG_NATURAL_ORDER[i] as usize] > MAX_SAMPLE {
                has_overshoot = true;
                break;
            }
        }
        assert!(
            has_overshoot,
            "Deringing should create overshoot above MAX_SAMPLE"
        );

        // Check that overshoot is limited
        for i in 10..16 {
            assert!(
                data[JPEG_NATURAL_ORDER[i] as usize] <= MAX_SAMPLE + 31.0,
                "Overshoot should be limited to MAX_SAMPLE + 31"
            );
        }
    }

    #[test]
    fn test_deringing_smooth_curve() {
        // Create a sharp edge transitioning to max
        let mut data = [0.0f32; DCT_BLOCK_SIZE];

        // Ramp up to max then stay at max
        data[JPEG_NATURAL_ORDER[0] as usize] = 20.0;
        data[JPEG_NATURAL_ORDER[1] as usize] = 60.0;
        data[JPEG_NATURAL_ORDER[2] as usize] = 100.0;
        for i in 3..8 {
            data[JPEG_NATURAL_ORDER[i] as usize] = MAX_SAMPLE;
        }
        // Then ramp down
        data[JPEG_NATURAL_ORDER[8] as usize] = 100.0;
        data[JPEG_NATURAL_ORDER[9] as usize] = 60.0;
        data[JPEG_NATURAL_ORDER[10] as usize] = 20.0;

        preprocess_deringing_f32(&mut data, 16);

        // The max-value region should now have a curve
        // The middle values should be higher than the edges of the curve
        let edge_val = data[JPEG_NATURAL_ORDER[3] as usize];
        let mid_val = data[JPEG_NATURAL_ORDER[5] as usize];

        assert!(
            mid_val >= edge_val,
            "Middle of curve should be >= edges: mid={}, edge={}",
            mid_val,
            edge_val
        );
    }

    #[test]
    fn test_deringing_respects_dc_quant_limit() {
        // With a small DC quant value, overshoot should be more limited
        let mut data = [MAX_SAMPLE; DCT_BLOCK_SIZE];
        // Leave one pixel not at max so deringing triggers
        data[0] = 50.0;

        let mut data_small_quant = data;
        let mut data_large_quant = data;

        preprocess_deringing_f32(&mut data_small_quant, 2); // Small DC quant = limited overshoot
        preprocess_deringing_f32(&mut data_large_quant, 32); // Larger DC quant = more overshoot

        // Find max values in each
        let max_small = data_small_quant.iter().copied().fold(f32::MIN, f32::max);
        let max_large = data_large_quant.iter().copied().fold(f32::MIN, f32::max);

        assert!(
            max_small <= max_large,
            "Smaller DC quant should allow less overshoot: small_max={}, large_max={}",
            max_small,
            max_large
        );
    }

    #[test]
    fn test_deringing_block() {
        use wide::f32x8;

        // Create a Block8x8f with some max pixels
        let mut block = Block8x8f {
            rows: [f32x8::ZERO; 8],
        };

        // Set first row to max
        block.rows[0] = f32x8::splat(MAX_SAMPLE);
        // Set second row to a slope
        block.rows[1] = f32x8::from([100.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0]);

        preprocess_deringing_block(&mut block, 16);

        // First row should have some overshoot
        let row0: [f32; 8] = block.rows[0].into();
        let has_overshoot = row0.iter().any(|&v| v > MAX_SAMPLE);
        assert!(has_overshoot, "Block deringing should create overshoot");
    }
}
