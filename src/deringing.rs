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
//! # When It Helps
//!
//! - Images with white backgrounds
//! - Text and graphics with hard edges
//! - Any image with saturated regions (pixels at 0 or 255)
//!
//! # References
//!
//! - mozjpeg source: `jcdctmgr.c:416-550`
//! - Original implementation: mozjpeg-oxide

use crate::consts::{DCTSIZE2, JPEG_NATURAL_ORDER};

/// Center sample value (128 is subtracted during level shift).
/// After level shift, the range is -128 to +127.
const CENTER_SAMPLE: i16 = 128;

/// Maximum sample value after level shift (255 - 128 = 127).
const MAX_SAMPLE: i16 = 255 - CENTER_SAMPLE;

/// Catmull-Rom spline interpolation.
///
/// Interpolates between `value2` and `value3` based on parameter `t` (0.0 to 1.0).
/// `value1` and `value4` are used to determine the tangent slopes at the endpoints.
///
/// The `size` parameter scales the tangents for the length of the interpolated segment.
fn catmull_rom(value1: i16, value2: i16, value3: i16, value4: i16, t: f32, size: i32) -> f32 {
    // Tangents at the endpoints, scaled by segment size
    let tan1 = (value3 as i32 - value1 as i32) * size;
    let tan2 = (value4 as i32 - value2 as i32) * size;

    // Hermite basis functions
    let t2 = t * t;
    let t3 = t2 * t;

    let f1 = 2.0 * t3 - 3.0 * t2 + 1.0; // h00: value at t=0
    let f2 = -2.0 * t3 + 3.0 * t2; // h01: value at t=1
    let f3 = t3 - 2.0 * t2 + t; // h10: tangent at t=0
    let f4 = t3 - t2; // h11: tangent at t=1

    value2 as f32 * f1 + tan1 as f32 * f3 + value3 as f32 * f2 + tan2 as f32 * f4
}

/// Preprocess an 8x8 block to reduce ringing artifacts on white backgrounds.
///
/// This function should be called on level-shifted sample data (samples centered around 0)
/// BEFORE the DCT transform. It modifies pixels at the maximum value to create smooth
/// transitions that compress better while producing identical visual output after decoding.
///
/// # Arguments
/// * `data` - Mutable slice of 64 level-shifted samples (-128 to +127). Modified in place.
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
pub fn preprocess_deringing(data: &mut [i16; DCTSIZE2], dc_quant: u16) {
    // Calculate sum and count of max-value pixels
    let mut sum: i32 = 0;
    let mut max_sample_count: usize = 0;

    for &sample in data.iter() {
        sum += sample as i32;
        if sample >= MAX_SAMPLE {
            max_sample_count += 1;
        }
    }

    // If nothing reaches max value, there's nothing to overshoot.
    // If the block is completely at max value, it's already the best case.
    if max_sample_count == 0 || max_sample_count == DCTSIZE2 {
        return;
    }

    // Calculate maximum safe overshoot:
    // 1. Don't overshoot more than 31 (arbitrary reasonable limit)
    // 2. Don't overshoot more than 2x the DC quantization (cost/benefit)
    // 3. Stay within DC headroom to avoid overflow
    let dc_limit = 2 * dc_quant as i32;
    let headroom = (MAX_SAMPLE as i32 * DCTSIZE2 as i32 - sum) / max_sample_count as i32;
    let max_overshoot: i16 = MAX_SAMPLE + (31.min(dc_limit).min(headroom)) as i16;

    // Process pixels in zigzag (natural) order
    let mut n: usize = 0;

    while n < DCTSIZE2 {
        // Skip pixels that aren't at max value
        if data[JPEG_NATURAL_ORDER[n]] < MAX_SAMPLE {
            n += 1;
            continue;
        }

        // Found a max-value pixel; find the extent of this run
        let start = n;
        while n + 1 < DCTSIZE2 && data[JPEG_NATURAL_ORDER[n + 1]] >= MAX_SAMPLE {
            n += 1;
        }
        let end = n + 1; // end is exclusive

        // Get values around the edges of the run for slope calculation
        // If at boundary, use the available values
        let f1 = data[JPEG_NATURAL_ORDER[start.saturating_sub(1)]];
        let f2 = data[JPEG_NATURAL_ORDER[start.saturating_sub(2)]];

        let l1 = data[JPEG_NATURAL_ORDER[if end < DCTSIZE2 { end } else { DCTSIZE2 - 1 }]];
        let l2 = data[JPEG_NATURAL_ORDER[if end + 1 < DCTSIZE2 {
            end + 1
        } else {
            DCTSIZE2 - 1
        }]];

        // Calculate upward slopes at the edges
        // Use the steeper of: slope from two samples back, or slope to max
        // This ensures we get an upward slope even if the edge is already clipped
        let mut fslope = (f1 - f2).max(MAX_SAMPLE - f1);
        let mut lslope = (l1 - l2).max(MAX_SAMPLE - l1);

        // If at the start/end of the block, make the curve symmetric
        if start == 0 {
            fslope = lslope;
        }
        if end == DCTSIZE2 {
            lslope = fslope;
        }

        // Apply Catmull-Rom interpolation across the run
        // The curve fits better if we treat the first and last points of the run
        // as being just inside the endpoints
        let length = (end - start) as i32;
        let step = 1.0 / (length + 1) as f32;
        let mut position = step;

        for i in start..end {
            // Interpolate a smooth curve that peaks above MAX_SAMPLE
            // Control points: approaching slope, max, max, departing slope
            let interpolated = catmull_rom(
                MAX_SAMPLE - fslope, // virtual point before (for tangent)
                MAX_SAMPLE,          // start of plateau
                MAX_SAMPLE,          // end of plateau
                MAX_SAMPLE - lslope, // virtual point after (for tangent)
                position,
                length,
            );

            // Ceiling and clamp to max overshoot
            let value = interpolated.ceil() as i16;
            data[JPEG_NATURAL_ORDER[i]] = value.min(max_overshoot);

            position += step;
        }

        n += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_catmull_rom_midpoint() {
        // At t=0.5, should be roughly at the midpoint value
        let result = catmull_rom(100, 110, 120, 130, 0.5, 1);
        assert!(
            (result - 115.0).abs() < 1.0,
            "Expected ~115, got {}",
            result
        );
    }

    #[test]
    fn test_catmull_rom_endpoints() {
        // At t=0, should be close to value2
        let result0 = catmull_rom(100, 110, 120, 130, 0.0, 1);
        assert!(
            (result0 - 110.0).abs() < 0.1,
            "Expected 110, got {}",
            result0
        );

        // At t=1, should be close to value3
        let result1 = catmull_rom(100, 110, 120, 130, 1.0, 1);
        assert!(
            (result1 - 120.0).abs() < 0.1,
            "Expected 120, got {}",
            result1
        );
    }

    #[test]
    fn test_deringing_no_max_pixels() {
        // Block with no pixels at max value - should be unchanged
        let mut data = [64i16; DCTSIZE2];
        let original = data;

        preprocess_deringing(&mut data, 16);

        assert_eq!(
            data, original,
            "Block with no max pixels should be unchanged"
        );
    }

    #[test]
    fn test_deringing_all_max_pixels() {
        // Block with all pixels at max value - should be unchanged
        let mut data = [MAX_SAMPLE; DCTSIZE2];
        let original = data;

        preprocess_deringing(&mut data, 16);

        assert_eq!(
            data, original,
            "Block with all max pixels should be unchanged"
        );
    }

    #[test]
    fn test_deringing_creates_overshoot() {
        // Create a block with a run of max pixels in the middle
        let mut data = [0i16; DCTSIZE2];

        // Set some pixels to max value (indices 10-15 in natural order)
        for i in 10..16 {
            data[JPEG_NATURAL_ORDER[i]] = MAX_SAMPLE;
        }
        // Set surrounding pixels to create a slope
        data[JPEG_NATURAL_ORDER[8]] = 80;
        data[JPEG_NATURAL_ORDER[9]] = 100;
        data[JPEG_NATURAL_ORDER[16]] = 100;
        data[JPEG_NATURAL_ORDER[17]] = 80;

        preprocess_deringing(&mut data, 16);

        // Check that some overshoot occurred
        let mut has_overshoot = false;
        for i in 10..16 {
            if data[JPEG_NATURAL_ORDER[i]] > MAX_SAMPLE {
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
                data[JPEG_NATURAL_ORDER[i]] <= MAX_SAMPLE + 31,
                "Overshoot should be limited to MAX_SAMPLE + 31"
            );
        }
    }

    #[test]
    fn test_deringing_respects_dc_quant_limit() {
        // With a small DC quant value, overshoot should be more limited
        let mut data = [0i16; DCTSIZE2];
        for i in 0..DCTSIZE2 {
            data[i] = MAX_SAMPLE;
        }
        // Leave one pixel not at max so deringing triggers
        data[0] = 50;

        let mut data_small_quant = data;
        let mut data_large_quant = data;

        preprocess_deringing(&mut data_small_quant, 2); // Small DC quant = limited overshoot
        preprocess_deringing(&mut data_large_quant, 32); // Larger DC quant = more overshoot allowed

        // Find max values in each
        let max_small = data_small_quant.iter().copied().max().unwrap();
        let max_large = data_large_quant.iter().copied().max().unwrap();

        assert!(
            max_small <= max_large,
            "Smaller DC quant should allow less overshoot: small_max={}, large_max={}",
            max_small,
            max_large
        );
    }
}
