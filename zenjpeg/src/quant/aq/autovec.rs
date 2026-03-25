//! Auto-vectorized AQ functions using archmage `#[autoversion]`.
//!
//! These functions use pure scalar code that the compiler autovectorizes when
//! `#[autoversion]` enables AVX2/NEON target features. This is 2-3x faster than
//! using the `wide` crate without global target features.
//!
//! ## Why not `wide`?
//!
//! The `wide` crate uses `cfg(target_feature)` which is compile-time only.
//! Without `-C target-cpu=x86-64-v3`, it falls back to SSE even inside
//! autoversioned functions. `#[autoversion]` uses `#[target_feature]`
//! which enables autovectorization at the function level.
//!
//! ## Benchmark (2026-01-21)
//!
//! 8x8 f32 transpose (see examples/autovec_transpose.rs):
//! - Naive scalar: 13.31 ns
//! - #[autoversion]: 4.73 ns (2.8x faster)

use archmage::autoversion;

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
#[autoversion]
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
#[autoversion]
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
// Gamma Modulation (autovectorized)
// ============================================================================

// K_BIAS for gamma modulation is different from GAMMA_OFFSET used in pre_erosion
const K_BIAS: f32 = 0.16 / K_INPUT_SCALING; // = 40.8

/// Inverse ratio of derivatives - for gamma modulation.
/// This is num/den instead of den/num used in pre_erosion.
#[inline(always)]
fn ratio_of_derivatives_inv(val: f32) -> f32 {
    let v = val.max(0.0);
    let v2 = v * v;
    let num = v2.mul_add(K_NUM_MUL_RATIO, K_NUM_OFFSET_RATIO);
    let den = (v * K_DEN_MUL_RATIO).mul_add(v2, K_VOFFSET_RATIO);
    num / den
}

/// Compute gamma modulation sum for an 8x8 block using autovectorization.
///
/// # Arguments
/// * `block` - Pointer to block start in buffer
/// * `stride` - Row stride in buffer
/// * `block_y` - Y coordinate of block start
/// * `img_height` - Image height for boundary checks
#[autoversion]
pub fn gamma_modulation_sum_8x8_autovec(
    block: &[f32],
    stride: usize,
    block_y: usize,
    img_height: usize,
) -> f32 {
    let mut sum = 0.0f32;

    for dy in 0..8 {
        let y = block_y + dy;
        if y >= img_height {
            continue;
        }

        let row_start = dy * stride;
        if row_start + 8 > block.len() {
            continue;
        }

        // Process 8 pixels - compiler will autovectorize
        for i in 0..8 {
            let pixel = block[row_start + i];
            sum += ratio_of_derivatives_inv(pixel + K_BIAS);
        }
    }

    sum
}

// ============================================================================
// HF Modulation (autovectorized)
// ============================================================================

/// Compute HF modulation sum for an 8x8 block using autovectorization.
///
/// Computes sum of |p - p_right| + |p - p_below| for all valid pixel pairs.
///
/// # Arguments
/// * `block` - Pointer to block start in buffer (needs +1 column for horizontal diffs)
/// * `stride` - Row stride in buffer
/// * `block_y` - Y coordinate of block start
/// * `img_height` - Image height for boundary checks
#[autoversion]
pub fn hf_modulation_sum_8x8_autovec(
    block: &[f32],
    stride: usize,
    block_y: usize,
    img_height: usize,
) -> f32 {
    let mut h_sum = 0.0f32;
    let mut v_sum = 0.0f32;

    for dy in 0..8 {
        let y = block_y + dy;
        if y >= img_height {
            continue;
        }

        let row_start = dy * stride;

        // Horizontal differences: |p - p_right| for positions 0..7
        if row_start + 9 <= block.len() {
            // Process 7 horizontal differences (mask out 8th)
            for i in 0..7 {
                let p = block[row_start + i];
                let p_right = block[row_start + i + 1];
                h_sum += (p - p_right).abs();
            }
        }

        // Vertical differences: |p - p_below| for first 7 rows
        if dy < 7 && y + 1 < img_height {
            let next_row_start = (dy + 1) * stride;
            if row_start + 8 <= block.len() && next_row_start + 8 <= block.len() {
                // Process 8 vertical differences
                for i in 0..8 {
                    let p = block[row_start + i];
                    let p_below = block[next_row_start + i];
                    v_sum += (p - p_below).abs();
                }
            }
        }
    }

    h_sum + v_sum
}

// ============================================================================
// Per-block modulations row (fully autovectorized)
// ============================================================================

// Fast log2 approximation - must match simd.rs for identical results
#[inline(always)]
fn fast_log2(x: f32) -> f32 {
    if x <= 0.0 {
        return f32::NEG_INFINITY;
    }

    let bits = x.to_bits() as i32;
    let e = (bits >> 23) - 127;

    // Extract mantissa as float in [1, 2)
    let f = f32::from_bits((bits & 0x007FFFFF) as u32 | 0x3F800000);
    let f_minus_1 = f - 1.0;

    // Polynomial approximation using Horner's method (FMA)
    let t = f_minus_1;
    let log2_f = t * 0.2885390082_f32
        .mul_add(t, -0.3606737602)
        .mul_add(t, 0.4808983470)
        .mul_add(t, -0.7213475204)
        .mul_add(t, 1.442695041);

    e as f32 + log2_f
}

// Fast exp2 approximation - must match simd.rs for identical results
#[inline(always)]
fn fast_exp2(x: f32) -> f32 {
    // Clamp to prevent overflow/underflow
    let x = x.clamp(-126.0, 127.0);

    // Split into integer and fractional parts
    let xi = x.floor();
    let xf = x - xi;

    // Minimax polynomial approximation for 2^xf where xf in [0, 1)
    let p = 1.0
        + xf * (0.6931471805599453  // ln(2)
        + xf * (0.24022650695910071 // ln(2)^2/2
        + xf * (0.055504108664821579 // ln(2)^3/6
        + xf * 0.009618129107628477))); // ln(2)^4/24

    // Combine: 2^xi * p(xf) using IEEE 754 bit manipulation
    let xi_i32 = xi as i32;
    let bits = ((xi_i32 + 127) as u32) << 23;
    f32::from_bits(bits) * p
}

/// Process per_block_modulations for a row of blocks using autovectorization.
///
/// This combines ComputeMask, HfModulation, GammaModulation, and final transform,
/// using pure scalar code that the compiler autovectorizes.
#[autoversion]
pub fn per_block_modulations_row_autovec(
    input: &[f32],
    stride: usize,
    _img_width: usize,
    img_height: usize,
    by: usize,
    block_w: usize,
    aq_row: &mut [f32],
    mul: f32,
    add: f32,
) {
    const K_SUM_COEFF: f32 = -2.0052193233688884 * K_INPUT_SCALING / 112.0;
    const K_GAMMA: f32 = -0.15526878023684174 * K_INV_LOG2E;
    const K_SCALE: f32 = K_INPUT_SCALING / 64.0;
    const LOG2_E: f32 = 1.442695041;

    // ComputeMask constants
    const K_MASK_BASE: f32 = -0.74174993;
    const K_MASK_MUL4: f32 = 3.2353257320940401;
    const K_MASK_MUL2: f32 = 12.906028311180409;
    const K_MASK_OFFSET2: f32 = 305.04035728311436;
    const K_MASK_MUL3: f32 = 5.0220313103171232;
    const K_MASK_OFFSET3: f32 = 2.1925739705298404;
    const K_MASK_OFFSET4: f32 = 0.25 * K_MASK_OFFSET3;
    const K_MASK_MUL0: f32 = 0.74760422233706747;

    let y_start = by * 8;

    for bx in 0..block_w {
        let x_start = bx * 8;

        // Get fuzzy erosion value
        let fuzzy_val = aq_row[bx];

        // 1. ComputeMask (inlined)
        let v1 = (fuzzy_val * K_MASK_MUL0).max(1e-3);
        let v2 = 1.0 / (v1 + K_MASK_OFFSET2);
        let v3 = 1.0 / (v1 * v1 + K_MASK_OFFSET3);
        let v4 = 1.0 / (v1 * v1 + K_MASK_OFFSET4);
        let mut out_val = K_MASK_MUL4.mul_add(
            v4,
            K_MASK_MUL2.mul_add(v2, K_MASK_MUL3.mul_add(v3, K_MASK_BASE)),
        );

        // 2. HfModulation
        let block_offset = y_start * stride + x_start;
        let block = &input[block_offset..];
        let hf_sum = hf_modulation_sum_8x8_autovec(block, stride, y_start, img_height);
        out_val += hf_sum * K_SUM_COEFF;

        // 3. GammaModulation
        let gamma_sum = gamma_modulation_sum_8x8_autovec(block, stride, y_start, img_height);
        let overall_ratio = gamma_sum * K_SCALE;
        let log_ratio = if overall_ratio > 0.0 {
            fast_log2(overall_ratio)
        } else {
            0.0
        };
        out_val += K_GAMMA * log_ratio;

        // 4. Final transform
        let quant_field = fast_exp2(out_val * LOG2_E).mul_add(mul, add);
        aq_row[bx] = quant_field;
    }
}

// ============================================================================
// Fuzzy erosion with autovectorized sorting network
// ============================================================================

const FUZZY_MUL0: f32 = 0.125;
const FUZZY_MUL1: f32 = 0.075;
const FUZZY_MUL2: f32 = 0.06;
const FUZZY_MUL3: f32 = 0.05;

/// Compare-and-swap at indices a and b in the 9x8 array.
/// Operates on all 8 lanes in parallel (autovectorizes).
#[inline(always)]
fn cas_idx(v: &mut [[f32; 8]; 9], a: usize, b: usize) {
    for i in 0..8 {
        let min = v[a][i].min(v[b][i]);
        let max = v[a][i].max(v[b][i]);
        v[a][i] = min;
        v[b][i] = max;
    }
}

/// Sorting network to find 4 smallest of 9 values, operating on 8 parallel sets.
/// Returns weighted sum: MUL0*v0 + MUL1*v1 + MUL2*v2 + MUL3*v3 for each set.
#[inline(always)]
fn weighted_min4_of_9_autovec(mut v: [[f32; 8]; 9]) -> [f32; 8] {
    // Sorting network: 19 compare-exchange operations
    // Each operation processes 8 values in parallel (autovectorizes to SIMD)

    // Layer 1
    cas_idx(&mut v, 0, 1);
    cas_idx(&mut v, 2, 3);
    cas_idx(&mut v, 4, 5);
    cas_idx(&mut v, 6, 7);

    // Layer 2
    cas_idx(&mut v, 0, 2);
    cas_idx(&mut v, 1, 3);
    cas_idx(&mut v, 4, 6);
    cas_idx(&mut v, 5, 7);

    // Layer 3
    cas_idx(&mut v, 0, 4);
    cas_idx(&mut v, 1, 5);
    cas_idx(&mut v, 2, 6);
    cas_idx(&mut v, 3, 7);

    // Layer 4
    cas_idx(&mut v, 0, 8);
    cas_idx(&mut v, 4, 8);

    // Layer 5
    cas_idx(&mut v, 1, 2);
    cas_idx(&mut v, 1, 4);
    cas_idx(&mut v, 2, 4);
    cas_idx(&mut v, 3, 4);
    cas_idx(&mut v, 5, 8);

    // Compute weighted sum
    let mut result = [0.0f32; 8];
    for i in 0..8 {
        result[i] = FUZZY_MUL0 * v[0][i]
            + FUZZY_MUL1 * v[1][i]
            + FUZZY_MUL2 * v[2][i]
            + FUZZY_MUL3 * v[3][i];
    }
    result
}

/// Compute fuzzy erosion for blocks using autovectorized sorting network.
/// Uses #[autoversion] for runtime AVX2/SSE/NEON dispatch.
///
/// This is the default fast path - provides runtime SIMD dispatch without
/// requiring `-C target-cpu=native` compile flags.
///
/// Returns the number of blocks processed.
#[autoversion]
pub fn compute_fuzzy_erosion_blocks_autovec(
    pre_erosion_buffer: &[f32],
    pe_w: usize,
    buffer_rows: usize,
    pe_y_base: isize,
    max_filled_row: isize,
    start: usize,
    end: usize,
    out: &mut [f32],
) -> usize {
    let max_x = pe_w as isize - 1;
    let max_y = max_filled_row.max(0);

    let mut processed = 0;
    let mut bx = start;

    // Process 8 blocks at a time
    while bx + 8 <= end {
        let base_cx: [isize; 8] = std::array::from_fn(|i| ((bx + i - start) * 2) as isize);

        let mut sum = [0.0f32; 8];

        // Process 4 sub-pixels per block
        for dy in 0..2isize {
            for dx in 0..2isize {
                let cy = pe_y_base + dy;
                let cx: [isize; 8] = std::array::from_fn(|i| base_cx[i] + dx);

                // Gather 9 neighbors for all 8 blocks
                let mut v: [[f32; 8]; 9] = [[0.0; 8]; 9];
                for (neighbor_idx, (ny, nx)) in [
                    (-1isize, -1isize),
                    (-1, 0),
                    (-1, 1),
                    (0, -1),
                    (0, 0),
                    (0, 1),
                    (1, -1),
                    (1, 0),
                    (1, 1),
                ]
                .iter()
                .enumerate()
                {
                    let py = (cy + ny).clamp(0, max_y) as usize;
                    let buffer_row = py % buffer_rows;
                    let row_offset = buffer_row * pe_w;

                    for lane in 0..8 {
                        let px = (cx[lane] + nx).clamp(0, max_x) as usize;
                        let idx = row_offset + px;
                        v[neighbor_idx][lane] = if idx < pre_erosion_buffer.len() {
                            pre_erosion_buffer[idx]
                        } else {
                            0.0
                        };
                    }
                }

                // Sorting network
                let result = weighted_min4_of_9_autovec(v);
                for i in 0..8 {
                    sum[i] += result[i];
                }
            }
        }

        // Store results
        out[bx..bx + 8].copy_from_slice(&sum);

        bx += 8;
        processed += 8;
    }

    processed
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

    #[test]
    fn test_gamma_modulation_sum_8x8_autovec() {
        // Create 8x8 block with known values
        let stride = 9; // 8 + 1 for horizontal padding
        let mut block = vec![0.0f32; stride * 8];

        // Fill with test pattern
        for dy in 0..8 {
            for dx in 0..8 {
                block[dy * stride + dx] = ((dy * 8 + dx) as f32 * 3.7) % 255.0;
            }
        }

        let sum = gamma_modulation_sum_8x8_autovec(&block, stride, 0, 8);
        assert!(sum > 0.0, "Gamma sum should be positive, got {}", sum);
        assert!(sum.is_finite(), "Gamma sum should be finite");
    }

    #[test]
    fn test_hf_modulation_sum_8x8_autovec() {
        // Create 8x8 block with gradient
        let stride = 9;
        let mut block = vec![0.0f32; stride * 8];

        // Fill with gradient - differences should be non-zero
        for dy in 0..8 {
            for dx in 0..9 {
                block[dy * stride + dx] = (dy * 10 + dx * 5) as f32;
            }
        }

        let sum = hf_modulation_sum_8x8_autovec(&block, stride, 0, 8);

        // With gradient pattern, should have:
        // - 7 horizontal diffs per row × 8 rows = 56 diffs, each = 5
        // - 8 vertical diffs per row × 7 rows = 56 diffs, each = 10
        // Total = 56*5 + 56*10 = 280 + 560 = 840
        let expected = 840.0;
        let diff = (sum - expected).abs();
        assert!(
            diff < 0.1,
            "HF sum mismatch: got {}, expected {}",
            sum,
            expected
        );
    }

    #[test]
    fn test_per_block_modulations_row_autovec() {
        // Create input buffer for one row of blocks
        let block_w = 8;
        let width = block_w * 8;
        let height = 8;
        let stride = width + 1; // +1 for horizontal padding

        let mut input = vec![128.0f32; stride * height];
        // Add some variation
        for y in 0..height {
            for x in 0..width {
                input[y * stride + x] = 100.0 + ((x + y) % 50) as f32;
            }
        }

        // Initialize aq_row with fuzzy erosion values
        let mut aq_row = vec![0.5f32; block_w];

        let mul = 0.841;
        let add = 0.1;

        per_block_modulations_row_autovec(
            &input,
            stride,
            width,
            height,
            0,
            block_w,
            &mut aq_row,
            mul,
            add,
        );

        // Should produce positive values within reasonable range
        for (bx, &val) in aq_row.iter().enumerate() {
            assert!(
                val.is_finite(),
                "Block {} has non-finite value: {}",
                bx,
                val
            );
            assert!(val > 0.0, "Block {} has non-positive value: {}", bx, val);
        }
    }
}
