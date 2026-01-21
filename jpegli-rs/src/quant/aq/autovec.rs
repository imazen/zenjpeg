//! Auto-vectorized AQ functions using multiversion.
//!
//! These functions use pure scalar code that the compiler autovectorizes when
//! the `#[multiversion]` attribute enables AVX2/NEON. This is 2-3x faster than
//! using the `wide` crate without global target features.
//!
//! ## Why not `wide`?
//!
//! The `wide` crate uses `cfg(target_feature)` which is compile-time only.
//! Without `-C target-cpu=x86-64-v3`, it falls back to SSE even inside
//! `#[multiversed]` functions. The `multiversion` crate uses `#[target_feature]`
//! which enables autovectorization at the function level.
//!
//! ## Benchmark (2026-01-21)
//!
//! 8x8 f32 transpose (see examples/autovec_transpose.rs):
//! - Naive scalar: 13.31 ns
//! - #[multiversion]: 4.73 ns (2.8x faster)

use multiversion::multiversion;

// ============================================================================
// Constants (same as simd.rs)
// ============================================================================

const K_INPUT_SCALING: f32 = 1.0 / 255.0;
const K_EPSILON_RATIO: f32 = 1e-2;
const K_NUM_OFFSET_RATIO: f32 = K_EPSILON_RATIO / K_INPUT_SCALING / K_INPUT_SCALING;
const K_SG_MUL: f32 = 226.0480446705883;
const K_SG_MUL2: f32 = 1.0 / 73.377132366608819;
const K_INV_LOG2E: f32 = 0.6931471805599453;
const K_SG_RET_MUL: f32 = K_SG_MUL2 * 18.6580932135 * K_INV_LOG2E;
const K_NUM_MUL_RATIO: f32 = K_SG_RET_MUL * 3.0 * K_SG_MUL;
const K_SG_VOFFSET: f32 = 7.14672470003;
const K_VOFFSET_RATIO: f32 = (K_SG_VOFFSET * K_INV_LOG2E + K_EPSILON_RATIO) / K_INPUT_SCALING;
const K_DEN_MUL_RATIO: f32 = K_INV_LOG2E * K_SG_MUL * K_INPUT_SCALING * K_INPUT_SCALING;

const K_MASKING_LOG_OFFSET: f32 = 28.0;
const K_MASKING_MUL: f32 = 211.50759899638012;
const K_MASKING_MUL_SQRT: f32 = 145433.00828779556; // (K_MASKING_MUL * 1e8).sqrt()

const LIMIT: f32 = 0.2;
const MATCH_GAMMA_OFFSET: f32 = 0.019;
const GAMMA_OFFSET: f32 = MATCH_GAMMA_OFFSET / K_INPUT_SCALING;

// ============================================================================
// Scalar primitives (inlined, will be autovectorized)
// ============================================================================

/// Ratio of derivatives - non-inverted version.
/// Compiler will autovectorize this when called in a loop.
#[inline(always)]
fn ratio_of_derivatives(val: f32) -> f32 {
    let v = val.max(0.0);
    let v2 = v * v;
    let num = v2.mul_add(K_NUM_MUL_RATIO, K_NUM_OFFSET_RATIO);
    let den = (v * K_DEN_MUL_RATIO).mul_add(v2, K_VOFFSET_RATIO);
    // den is always positive due to K_VOFFSET_RATIO > 0
    den / num
}

/// Masking sqrt operation.
#[inline(always)]
fn masking_sqrt(v: f32) -> f32 {
    0.25 * v.mul_add(K_MASKING_MUL_SQRT, K_MASKING_LOG_OFFSET).sqrt()
}

/// Single pixel pre-erosion computation.
#[inline(always)]
fn pre_erosion_pixel(pixel: f32, left: f32, right: f32, top: f32, bottom: f32) -> f32 {
    let base = 0.25 * (left + right + top + bottom);
    let ratio = ratio_of_derivatives(pixel + GAMMA_OFFSET);
    let diff = ratio * (pixel - base);
    let diff_sq = (diff * diff).min(LIMIT);
    masking_sqrt(diff_sq)
}

// ============================================================================
// Multiversioned row functions
// ============================================================================

/// Pre-erosion row processor with padded buffers.
///
/// Uses pure scalar code that the compiler autovectorizes to AVX2/NEON.
/// The padded buffers ensure we can use `chunks_exact(8)` without remainder handling.
///
/// # Arguments
/// * `row` - Current row with +1 padding on each side (len = width + 2)
/// * `row_above` - Row above with same padding
/// * `row_below` - Row below with same padding
/// * `width` - Actual image width (not including padding)
/// * `output` - Output buffer (len = width)
#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "x86_64+sse4.1", "aarch64+neon"))]
pub fn pre_erosion_row_autovec(
    row: &[f32],
    row_above: &[f32],
    row_below: &[f32],
    width: usize,
    output: &mut [f32],
) {
    debug_assert_eq!(row.len(), width + 2);
    debug_assert_eq!(row_above.len(), width + 2);
    debug_assert_eq!(row_below.len(), width + 2);
    debug_assert_eq!(output.len(), width);

    if width == 0 {
        return;
    }

    // Process in chunks of 8 for optimal autovectorization
    // Pixel data starts at index 1 due to padding
    let chunks = width / 8;

    for chunk_idx in 0..chunks {
        let x = chunk_idx * 8;
        let buf_x = x + 1; // Offset for padding

        // Process 8 pixels - compiler will autovectorize this
        for i in 0..8 {
            let pixel = row[buf_x + i];
            let left = row[buf_x + i - 1];
            let right = row[buf_x + i + 1];
            let top = row_above[buf_x + i];
            let bottom = row_below[buf_x + i];

            output[x + i] += pre_erosion_pixel(pixel, left, right, top, bottom);
        }
    }

    // Handle remainder
    for x in (chunks * 8)..width {
        let buf_x = x + 1;
        let pixel = row[buf_x];
        let left = row[buf_x - 1];
        let right = row[buf_x + 1];
        let top = row_above[buf_x];
        let bottom = row_below[buf_x];

        output[x] += pre_erosion_pixel(pixel, left, right, top, bottom);
    }
}

/// Alternative: Process using iterator chunks for cleaner code.
/// May or may not autovectorize as well depending on LLVM version.
#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "x86_64+sse4.1", "aarch64+neon"))]
pub fn pre_erosion_row_autovec_iter(
    row: &[f32],
    row_above: &[f32],
    row_below: &[f32],
    width: usize,
    output: &mut [f32],
) {
    debug_assert_eq!(row.len(), width + 2);
    debug_assert_eq!(row_above.len(), width + 2);
    debug_assert_eq!(row_below.len(), width + 2);
    debug_assert_eq!(output.len(), width);

    // Process all pixels using offset iterators
    // Padding ensures all neighbors are valid
    for x in 0..width {
        let buf_x = x + 1;
        let pixel = row[buf_x];
        let left = row[buf_x - 1];
        let right = row[buf_x + 1];
        let top = row_above[buf_x];
        let bottom = row_below[buf_x];

        output[x] += pre_erosion_pixel(pixel, left, right, top, bottom);
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pre_erosion_row_autovec_matches_reference() {
        // Create test data with padding
        let width = 64;
        let mut row = vec![0.0f32; width + 2];
        let mut row_above = vec![0.0f32; width + 2];
        let mut row_below = vec![0.0f32; width + 2];

        // Fill with test pattern
        for i in 0..width + 2 {
            row[i] = (i as f32 * 3.7) % 255.0;
            row_above[i] = (i as f32 * 2.3 + 10.0) % 255.0;
            row_below[i] = (i as f32 * 4.1 + 20.0) % 255.0;
        }

        let mut output1 = vec![0.0f32; width];
        let mut output2 = vec![0.0f32; width];

        // Run both versions
        pre_erosion_row_autovec(&row, &row_above, &row_below, width, &mut output1);
        pre_erosion_row_autovec_iter(&row, &row_above, &row_below, width, &mut output2);

        // Compare results
        for i in 0..width {
            let diff = (output1[i] - output2[i]).abs();
            assert!(
                diff < 1e-5,
                "Mismatch at {}: {} vs {} (diff {})",
                i,
                output1[i],
                output2[i],
                diff
            );
        }
    }

    #[test]
    fn test_pre_erosion_row_autovec_nonzero() {
        let width = 32;
        let row = vec![128.0f32; width + 2];
        let row_above = vec![100.0f32; width + 2];
        let row_below = vec![150.0f32; width + 2];

        let mut output = vec![0.0f32; width];
        pre_erosion_row_autovec(&row, &row_above, &row_below, width, &mut output);

        // Should produce non-zero results
        let sum: f32 = output.iter().sum();
        assert!(sum > 0.0, "Output should be non-zero, got sum={}", sum);
    }
}
