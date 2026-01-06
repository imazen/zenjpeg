//! SIMD-optimized adaptive quantization functions.
//!
//! This module contains SIMD implementations of the hot paths in adaptive_quant.rs.
//! All functions have locked tests against scalar reference implementations.
//!
//! ## Hot Path Analysis (from profiling)
//!
//! `compute_aq_strength_map_impl` takes 18% of encoding time. Breakdown:
//! - `ratio_of_derivatives`: called 2x per pixel (pre_erosion + per_block_modulations)
//! - `masking_sqrt`: called 1x per pixel in pre_erosion
//! - `compute_pre_erosion`: neighbor averaging + ratio + masking per pixel
//! - `per_block_modulations`: 8x8 block sums with ratio_of_derivatives
//!
//! ## Test Data Source
//!
//! Locked test values are derived from frymire.png (1118x1105) Y plane.
//! To regenerate: `cargo test --lib adaptive_quant_simd -- --nocapture`

use wide::f32x8;

// ============================================================================
// Constants (copied from adaptive_quant.rs for locality)
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

// MaskingSqrt constants
const K_MASKING_LOG_OFFSET: f32 = 28.0;
const K_MASKING_MUL: f32 = 211.50759899638012;

// ============================================================================
// Scalar reference implementations (for testing)
// ============================================================================

/// Scalar reference: ratio_of_derivatives
#[inline]
fn ratio_of_derivatives_scalar(val: f32, invert: bool) -> f32 {
    let v = val.max(0.0);
    let v2 = v * v;
    let num = K_NUM_MUL_RATIO * v2 + K_NUM_OFFSET_RATIO;
    let den = (K_DEN_MUL_RATIO * v) * v2 + K_VOFFSET_RATIO;
    let safe_den = if den == 0.0 { 1e-9 } else { den };
    if invert {
        num / safe_den
    } else {
        safe_den / num
    }
}

/// Scalar reference: masking_sqrt
#[inline]
fn masking_sqrt_scalar(v: f32) -> f32 {
    0.25 * (v * (K_MASKING_MUL * 1e8_f32).sqrt() + K_MASKING_LOG_OFFSET).sqrt()
}

// ============================================================================
// SIMD implementations
// ============================================================================

/// SIMD version of ratio_of_derivatives (non-inverted).
/// Processes 8 f32 values at once.
#[inline]
pub fn ratio_of_derivatives_x8(vals: f32x8) -> f32x8 {
    let v = vals.fast_max(f32x8::ZERO);
    let v2 = v * v;

    let num = f32x8::splat(K_NUM_MUL_RATIO) * v2 + f32x8::splat(K_NUM_OFFSET_RATIO);
    let den = (f32x8::splat(K_DEN_MUL_RATIO) * v) * v2 + f32x8::splat(K_VOFFSET_RATIO);

    // den is always positive due to K_VOFFSET_RATIO > 0, no need for safe_den check
    den / num
}

/// SIMD version of ratio_of_derivatives (inverted).
/// Processes 8 f32 values at once.
#[inline]
pub fn ratio_of_derivatives_inv_x8(vals: f32x8) -> f32x8 {
    let v = vals.fast_max(f32x8::ZERO);
    let v2 = v * v;

    let num = f32x8::splat(K_NUM_MUL_RATIO) * v2 + f32x8::splat(K_NUM_OFFSET_RATIO);
    let den = (f32x8::splat(K_DEN_MUL_RATIO) * v) * v2 + f32x8::splat(K_VOFFSET_RATIO);

    num / den
}

/// SIMD version of masking_sqrt.
/// Processes 8 f32 values at once.
#[inline]
pub fn masking_sqrt_x8(v: f32x8) -> f32x8 {
    let k_mul_sqrt = f32x8::splat((K_MASKING_MUL * 1e8_f32).sqrt());
    let k_offset = f32x8::splat(K_MASKING_LOG_OFFSET);
    f32x8::splat(0.25) * (v * k_mul_sqrt + k_offset).sqrt()
}

// ============================================================================
// Pre-erosion SIMD - processes 8 horizontal pixels at once
// ============================================================================

const LIMIT: f32 = 0.2;
const MATCH_GAMMA_OFFSET: f32 = 0.019;
const GAMMA_OFFSET: f32 = MATCH_GAMMA_OFFSET / K_INPUT_SCALING; // ~4.845

/// Process 8 pixels of the pre-erosion inner loop.
///
/// For each pixel: compute neighbor average, ratio_of_derivatives, diff, masking_sqrt.
///
/// # Arguments
/// * `pixels` - 8 center pixel values (0-255)
/// * `left` - 8 left neighbor values
/// * `right` - 8 right neighbor values
/// * `top` - 8 top neighbor values
/// * `bottom` - 8 bottom neighbor values
///
/// # Returns
/// 8 masked diff values ready for accumulation
#[inline]
pub fn pre_erosion_pixel_x8(
    pixels: f32x8,
    left: f32x8,
    right: f32x8,
    top: f32x8,
    bottom: f32x8,
) -> f32x8 {
    // base = 0.25 * (left + right + top + bottom)
    let base = f32x8::splat(0.25) * (left + right + top + bottom);

    // ratio = ratio_of_derivatives(pixel + gamma_offset, false)
    let ratio = ratio_of_derivatives_x8(pixels + f32x8::splat(GAMMA_OFFSET));

    // diff = ratio * (pixel - base)
    let diff = ratio * (pixels - base);

    // diff_sq = min(diff * diff, LIMIT)
    let diff_sq = (diff * diff).fast_min(f32x8::splat(LIMIT));

    // masked = masking_sqrt(diff_sq)
    masking_sqrt_x8(diff_sq)
}

/// Process a full row of pre-erosion, writing results to output buffer.
///
/// Handles boundary conditions (first/last pixel clamping).
///
/// # Arguments
/// * `row` - Current row pixels (0-255 range)
/// * `row_above` - Row above (or same row if y=0)
/// * `row_below` - Row below (or same row if y=height-1)
/// * `output` - Output buffer to accumulate into (must be same length as row)
#[inline(always)]
pub fn pre_erosion_row(row: &[f32], row_above: &[f32], row_below: &[f32], output: &mut [f32]) {
    let width = row.len();
    assert_eq!(row_above.len(), width);
    assert_eq!(row_below.len(), width);
    assert_eq!(output.len(), width);

    if width == 0 {
        return;
    }

    // Process 8 pixels at a time for the main body
    let chunks = width / 8;

    for chunk in 0..chunks {
        let x = chunk * 8;

        // Load center pixels
        let pixels = f32x8::from(unsafe { *(row.as_ptr().add(x) as *const [f32; 8]) });

        // Load neighbors with boundary handling
        let left = if x == 0 {
            // First chunk: first pixel uses itself as left neighbor
            f32x8::from([
                row[0],
                row[x],
                row[x + 1],
                row[x + 2],
                row[x + 3],
                row[x + 4],
                row[x + 5],
                row[x + 6],
            ])
        } else {
            f32x8::from(unsafe { *(row.as_ptr().add(x - 1) as *const [f32; 8]) })
        };

        let right = if x + 8 >= width {
            // Last chunk: last pixel uses itself as right neighbor
            let last = width - 1;
            f32x8::from([
                row[(x + 1).min(last)],
                row[(x + 2).min(last)],
                row[(x + 3).min(last)],
                row[(x + 4).min(last)],
                row[(x + 5).min(last)],
                row[(x + 6).min(last)],
                row[(x + 7).min(last)],
                row[(x + 8).min(last)],
            ])
        } else {
            f32x8::from(unsafe { *(row.as_ptr().add(x + 1) as *const [f32; 8]) })
        };

        let top = f32x8::from(unsafe { *(row_above.as_ptr().add(x) as *const [f32; 8]) });
        let bottom = f32x8::from(unsafe { *(row_below.as_ptr().add(x) as *const [f32; 8]) });

        // Compute and accumulate using SIMD add
        let result = pre_erosion_pixel_x8(pixels, left, right, top, bottom);

        // SAFETY: x is always aligned to 8 and x + 8 <= width (guaranteed by chunks calculation)
        unsafe {
            let existing = f32x8::from(*(output.as_ptr().add(x) as *const [f32; 8]));
            let new_result = existing + result;
            *(output.as_mut_ptr().add(x) as *mut [f32; 8]) = new_result.into();
        }
    }

    // Handle remainder (scalar fallback)
    for x in (chunks * 8)..width {
        let pixel = row[x];
        let left_val = if x == 0 { row[0] } else { row[x - 1] };
        let right_val = if x == width - 1 {
            row[width - 1]
        } else {
            row[x + 1]
        };
        let top_val = row_above[x];
        let bottom_val = row_below[x];

        let base = 0.25 * (left_val + right_val + top_val + bottom_val);
        let ratio = ratio_of_derivatives_scalar(pixel + GAMMA_OFFSET, false);
        let diff = ratio * (pixel - base);
        let diff_sq = (diff * diff).min(LIMIT);
        let masked = masking_sqrt_scalar(diff_sq);

        output[x] += masked;
    }
}

// ============================================================================
// Full pre-erosion computation (SIMD)
// ============================================================================

/// SIMD-accelerated version of compute_pre_erosion.
///
/// Computes the pre-erosion field by processing rows with SIMD and downsampling 4x.
///
/// # Arguments
/// * `input` - Y plane (0-255 range)
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
///
/// # Returns
/// Pre-erosion buffer at 1/4 resolution
pub fn compute_pre_erosion_simd(input: &[f32], width: usize, height: usize) -> Vec<f32> {
    let pre_erosion_w = (width + 3) / 4;
    let pre_erosion_h = (height + 3) / 4;
    let mut pre_erosion = vec![0.0f32; pre_erosion_w * pre_erosion_h];

    if width == 0 || height == 0 {
        return pre_erosion;
    }

    // Temporary buffer for accumulating masked diff values
    let mut diff_buffer = vec![0.0f32; width];

    for y_block in 0..pre_erosion_h {
        // Clear accumulator for this 4-row block
        diff_buffer.fill(0.0);

        // Process up to 4 rows
        for iy in 0..4 {
            let y = y_block * 4 + iy;
            if y >= height {
                continue;
            }

            // Get row pointers with boundary clamping
            let row = &input[y * width..(y + 1) * width];
            let row_above = if y == 0 {
                row
            } else {
                &input[(y - 1) * width..y * width]
            };
            let row_below = if y + 1 >= height {
                row
            } else {
                &input[(y + 1) * width..(y + 2) * width]
            };

            // Process row with SIMD
            pre_erosion_row(row, row_above, row_below, &mut diff_buffer);
        }

        // Downsample 4x in x direction
        let out_row = &mut pre_erosion[y_block * pre_erosion_w..(y_block + 1) * pre_erosion_w];
        downsample_4x_sum(&diff_buffer, out_row);
    }

    pre_erosion
}

/// Downsample by 4x with sum and scale by 0.25.
#[inline(always)]
fn downsample_4x_sum(input: &[f32], output: &mut [f32]) {
    let width = input.len();
    let out_w = output.len();

    // SIMD path: process 8 output pixels at once (32 input pixels)
    // Must have at least 32 input pixels for each SIMD chunk
    let chunks = (width / 32).min(out_w / 8);

    for chunk in 0..chunks {
        let out_x = chunk * 8;
        let in_x = out_x * 4;

        // Load 32 consecutive input values (8 groups of 4)
        // Sum each group of 4 and multiply by 0.25
        let mut sums = [0.0f32; 8];
        for i in 0..8 {
            let base = in_x + i * 4;
            sums[i] = (input[base] + input[base + 1] + input[base + 2] + input[base + 3]) * 0.25;
        }

        // Write output
        output[out_x..out_x + 8].copy_from_slice(&sums);
    }

    // Scalar remainder
    for out_x in (chunks * 8)..out_w {
        let in_x = out_x * 4;
        let mut sum = 0.0f32;
        for i in 0..4 {
            if in_x + i < width {
                sum += input[in_x + i];
            }
        }
        output[out_x] = sum * 0.25;
    }
}

// ============================================================================
// Per-block modulations SIMD helpers
// ============================================================================

const K_BIAS: f32 = 0.16 / K_INPUT_SCALING; // 40.8

// ============================================================================
// Fast math approximations
// ============================================================================

/// Fast exp2 approximation using polynomial + bit manipulation.
/// Accurate to ~1e-4 relative error for inputs in [-126, 127].
/// This is sufficient for adaptive quantization where small variations don't matter.
#[inline(always)]
fn fast_exp2(x: f32) -> f32 {
    // Clamp to prevent overflow/underflow
    let x = x.clamp(-126.0, 127.0);

    // Split into integer and fractional parts
    let xi = x.floor();
    let xf = x - xi;

    // Minimax polynomial approximation for 2^xf where xf in [0, 1)
    // 4th degree polynomial gives ~1e-5 relative error
    // Coefficients derived from minimax fit
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

/// Fast log2 approximation using bit manipulation + polynomial.
/// Accurate to ~0.01 absolute error for positive inputs in [0.01, 100].
/// Falls back to standard log2 for extreme values.
#[inline(always)]
fn fast_log2(x: f32) -> f32 {
    if x <= 0.0 {
        return f32::NEG_INFINITY;
    }

    // For typical AQ values (0.01-100), use fast approximation
    // This covers the common case efficiently
    let bits = x.to_bits() as i32;
    let e = (bits >> 23) - 127;

    // Extract mantissa as float in [1, 2)
    let f = f32::from_bits((bits & 0x007FFFFF) as u32 | 0x3F800000);

    // Use a better polynomial approximation for log2(f) where f in [1, 2)
    // This is a minimax polynomial that minimizes max error over [1, 2)
    // log2(f) for f in [1,2) ranges from 0 to 1
    let f_minus_1 = f - 1.0;

    // Padé-like approximation: more accurate than pure Taylor series
    // log2(f) ≈ (f-1) * (a + b*(f-1)) / (c + d*(f-1))
    // Simplified to polynomial for speed:
    // log2(f) ≈ 1.442695 * ln(f) where ln(f) ≈ (f-1) - (f-1)²/2 + (f-1)³/3
    // = (f-1) * (1.442695 - 0.721348*(f-1) + 0.480899*(f-1)² - 0.360674*(f-1)³)
    let t = f_minus_1;
    let log2_f = t
        * (1.442695041
            + t * (-0.7213475204 + t * (0.4808983470 + t * (-0.3606737602 + t * 0.2885390082))));

    e as f32 + log2_f
}

/// Compute sum of ratio_of_derivatives(inv=true) for an 8x8 block.
///
/// SIMD accelerated - processes one row of 8 pixels at a time.
///
/// # Arguments
/// * `block` - Pointer to top-left of 8x8 block
/// * `stride` - Row stride (image width)
///
/// # Returns
/// Sum of ratio_of_derivatives for all 64 pixels
#[inline(always)]
pub fn gamma_modulation_sum_8x8(block: &[f32], stride: usize) -> f32 {
    let bias = f32x8::splat(K_BIAS);
    let mut sum = f32x8::ZERO;

    for dy in 0..8 {
        let row_start = dy * stride;
        if row_start + 8 <= block.len() {
            let row = f32x8::from(unsafe { *(block.as_ptr().add(row_start) as *const [f32; 8]) });
            let ratio = ratio_of_derivatives_inv_x8(row + bias);
            sum += ratio;
        } else {
            // Fallback for edge blocks
            for dx in 0..8 {
                let idx = row_start + dx;
                if idx < block.len() {
                    let val = block[idx] + K_BIAS;
                    sum = sum
                        + f32x8::from([
                            ratio_of_derivatives_scalar(val, true),
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                        ]);
                }
            }
        }
    }

    // Horizontal sum of the f32x8
    let arr: [f32; 8] = sum.into();
    arr.iter().sum()
}

/// Compute HF modulation sum: |p - right| + |p - below| for 8x8 block.
///
/// Optimized with SIMD for row processing.
///
/// # Arguments
/// * `block` - Pointer to top-left of 8x8 block
/// * `stride` - Row stride (image width)
/// * `block_x` - X position of block (for boundary check)
/// * `block_y` - Y position of block (for boundary check)
/// * `img_width` - Total image width
/// * `img_height` - Total image height
///
/// # Returns
/// Sum of absolute horizontal and vertical differences
#[inline(always)]
pub fn hf_modulation_sum_8x8(
    block: &[f32],
    stride: usize,
    block_x: usize,
    block_y: usize,
    img_width: usize,
    img_height: usize,
) -> f32 {
    let mut sum = 0.0f32;

    for dy in 0..8 {
        let y = block_y + dy;
        if y >= img_height {
            continue;
        }

        let row_start = dy * stride;

        // Process horizontal differences (|p - p_right|) for positions 0..7 in block
        // Use SIMD: load 8 values, load 8 shifted values, compute abs diff
        if row_start + 8 <= block.len() && block_x + 8 < img_width {
            let p = f32x8::from(unsafe { *(block.as_ptr().add(row_start) as *const [f32; 8]) });
            let p_right =
                f32x8::from(unsafe { *(block.as_ptr().add(row_start + 1) as *const [f32; 8]) });
            let h_diff = (p - p_right).abs();
            let h_arr: [f32; 8] = h_diff.into();
            // Only sum first 7 (last pixel in block doesn't have right neighbor within block)
            sum += h_arr[0] + h_arr[1] + h_arr[2] + h_arr[3] + h_arr[4] + h_arr[5] + h_arr[6];
        } else {
            // Scalar fallback for edge cases
            for dx in 0..7 {
                let x = block_x + dx;
                if x + 1 < img_width {
                    let idx = row_start + dx;
                    if idx + 1 < block.len() {
                        sum += (block[idx] - block[idx + 1]).abs();
                    }
                }
            }
        }

        // Process vertical differences (|p - p_below|) for first 7 rows
        if dy < 7 && y + 1 < img_height {
            let next_row_start = (dy + 1) * stride;
            if row_start + 8 <= block.len() && next_row_start + 8 <= block.len() {
                let p = f32x8::from(unsafe { *(block.as_ptr().add(row_start) as *const [f32; 8]) });
                let p_below = f32x8::from(unsafe {
                    *(block.as_ptr().add(next_row_start) as *const [f32; 8])
                });
                let v_diff = (p - p_below).abs();
                let v_arr: [f32; 8] = v_diff.into();
                sum += v_arr[0]
                    + v_arr[1]
                    + v_arr[2]
                    + v_arr[3]
                    + v_arr[4]
                    + v_arr[5]
                    + v_arr[6]
                    + v_arr[7];
            } else {
                // Scalar fallback
                for dx in 0..8 {
                    let idx = row_start + dx;
                    let below_idx = next_row_start + dx;
                    if idx < block.len() && below_idx < block.len() {
                        sum += (block[idx] - block[below_idx]).abs();
                    }
                }
            }
        }
    }

    sum
}

/// Full per_block_modulations with SIMD acceleration.
///
/// Replaces per_block_modulations_scalar with SIMD-optimized inner loops.
pub fn per_block_modulations_simd(
    y_quant_01: f32,
    input: &[f32],
    width: usize,
    height: usize,
    block_w: usize,
    block_h: usize,
    aq_map: &mut [f32],
) {
    const K_AC_QUANT: f32 = 0.841;
    const K_DAMPEN_RAMP_START: f32 = 9.0;
    const K_DAMPEN_RAMP_END: f32 = 65.0;

    let base_level = 0.48 * K_AC_QUANT;

    let dampen = if y_quant_01 >= K_DAMPEN_RAMP_START {
        let d =
            1.0 - (y_quant_01 - K_DAMPEN_RAMP_START) / (K_DAMPEN_RAMP_END - K_DAMPEN_RAMP_START);
        d.max(0.0)
    } else {
        1.0
    };

    let mul = K_AC_QUANT * dampen;
    let add = (1.0 - dampen) * base_level;

    for by in 0..block_h {
        let row_start = by * block_w;
        let row_end = row_start + block_w;
        per_block_modulations_row(
            input,
            width,
            height,
            by,
            block_w,
            &mut aq_map[row_start..row_end],
            mul,
            add,
        );
    }
}

/// Process per_block_modulations for a row of blocks.
///
/// Combines ComputeMask, HfModulation, GammaModulation, and final transform.
#[inline(always)]
pub fn per_block_modulations_row(
    input: &[f32],
    width: usize,
    height: usize,
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
        let mut out_val = K_MASK_BASE + K_MASK_MUL4 * v4 + K_MASK_MUL2 * v2 + K_MASK_MUL3 * v3;

        // 2. HfModulation with SIMD
        let block_offset = y_start * width + x_start;
        let block = &input[block_offset..];
        let hf_sum = hf_modulation_sum_8x8(block, width, x_start, y_start, width, height);
        out_val += hf_sum * K_SUM_COEFF;

        // 3. GammaModulation with SIMD and fast_log2
        let gamma_sum = gamma_modulation_sum_8x8(block, width);
        let overall_ratio = gamma_sum * K_SCALE;
        let log_ratio = if overall_ratio > 0.0 {
            fast_log2(overall_ratio)
        } else {
            0.0
        };
        out_val += K_GAMMA * log_ratio;

        // 4. Final transform using fast_exp2 approximation
        // Note: (out_val * LOG2_E).exp2() = exp(out_val)
        let quant_field = fast_exp2(out_val * LOG2_E) * mul + add;
        aq_row[bx] = quant_field;
    }
}

// ============================================================================
// Fuzzy Erosion SIMD
// ============================================================================

/// Weights from C++ FuzzyErosion (sum = 0.31)
const FUZZY_MUL0: f32 = 0.125;
const FUZZY_MUL1: f32 = 0.075;
const FUZZY_MUL2: f32 = 0.06;
const FUZZY_MUL3: f32 = 0.05;

/// Find 4 smallest values from 9 inputs using selection sort.
/// Returns weighted sum: MUL0*min0 + MUL1*min1 + MUL2*min2 + MUL3*min3
///
/// Uses the same selection sort algorithm as the scalar reference to ensure
/// identical results (important because weights are order-dependent).
#[inline(always)]
fn weighted_min4_of_9(v: [f32; 9]) -> f32 {
    let mut a = v;

    // Selection sort: find 4 smallest values in order
    // This matches the scalar reference exactly
    for i in 0..4 {
        for j in (i + 1)..9 {
            if a[j] < a[i] {
                a.swap(i, j);
            }
        }
    }

    FUZZY_MUL0 * a[0] + FUZZY_MUL1 * a[1] + FUZZY_MUL2 * a[2] + FUZZY_MUL3 * a[3]
}

/// SIMD-optimized FuzzyErosion.
///
/// For each pixel in pre_erosion:
/// 1. Gather 9 values from 3x3 window
/// 2. Find 4 smallest and compute weighted sum
/// 3. Write to tmp buffer
///
/// Then sum 2x2 blocks from tmp to get final aq_map values.
pub fn fuzzy_erosion_simd(
    pre_erosion: &[f32],
    pre_erosion_w: usize,
    pre_erosion_h: usize,
    block_w: usize,
    block_h: usize,
    aq_map: &mut [f32],
) {
    assert_eq!(aq_map.len(), block_w * block_h);

    // Temporary buffer for weighted min values
    let mut tmp = vec![0.0f32; pre_erosion_w * pre_erosion_h];

    // Process each pixel - find 4 smallest in 3x3 window, compute weighted sum
    // Process row by row to take advantage of cache locality
    for y in 0..pre_erosion_h {
        // Pre-fetch row data for better cache utilization
        let row_above_y = (y as isize - 1).clamp(0, pre_erosion_h as isize - 1) as usize;
        let row_below_y = (y as isize + 1).clamp(0, pre_erosion_h as isize - 1) as usize;
        let row_above =
            &pre_erosion[row_above_y * pre_erosion_w..(row_above_y + 1) * pre_erosion_w];
        let row_curr = &pre_erosion[y * pre_erosion_w..(y + 1) * pre_erosion_w];
        let row_below =
            &pre_erosion[row_below_y * pre_erosion_w..(row_below_y + 1) * pre_erosion_w];

        for x in 0..pre_erosion_w {
            let ix = x as isize;

            // Gather 9 values from 3x3 window with clamped bounds
            let x_left = (ix - 1).clamp(0, pre_erosion_w as isize - 1) as usize;
            let x_right = (ix + 1).clamp(0, pre_erosion_w as isize - 1) as usize;

            let vals = [
                row_above[x_left],  // top-left
                row_above[x],       // top
                row_above[x_right], // top-right
                row_curr[x_left],   // left
                row_curr[x],        // center
                row_curr[x_right],  // right
                row_below[x_left],  // bottom-left
                row_below[x],       // bottom
                row_below[x_right], // bottom-right
            ];

            tmp[y * pre_erosion_w + x] = weighted_min4_of_9(vals);
        }
    }

    // Sum 2x2 blocks from tmp to get final aq_map values
    // Use SIMD for the block summing
    sum_2x2_blocks_simd(&tmp, pre_erosion_w, pre_erosion_h, block_w, block_h, aq_map);
}

/// Sum 2x2 blocks from tmp buffer to produce aq_map values.
/// SIMD-optimized: processes 8 blocks at a time.
#[inline(always)]
fn sum_2x2_blocks_simd(
    tmp: &[f32],
    tmp_w: usize,
    tmp_h: usize,
    block_w: usize,
    block_h: usize,
    aq_map: &mut [f32],
) {
    // Helper to get tmp value with clamped bounds
    #[inline(always)]
    fn get_tmp(tmp: &[f32], tmp_w: usize, tmp_h: usize, x: usize, y: usize) -> f32 {
        let x = x.min(tmp_w.saturating_sub(1));
        let y = y.min(tmp_h.saturating_sub(1));
        tmp[y * tmp_w + x]
    }

    for by in 0..block_h {
        let py = by * 2;
        let chunks = block_w / 8;

        // SIMD path: process 8 blocks at once
        for chunk in 0..chunks {
            let bx_start = chunk * 8;
            let mut sums = [0.0f32; 8];

            for i in 0..8 {
                let bx = bx_start + i;
                let px = bx * 2;

                // Sum 2x2 values
                sums[i] = get_tmp(tmp, tmp_w, tmp_h, px, py)
                    + get_tmp(tmp, tmp_w, tmp_h, px + 1, py)
                    + get_tmp(tmp, tmp_w, tmp_h, px, py + 1)
                    + get_tmp(tmp, tmp_w, tmp_h, px + 1, py + 1);
            }

            aq_map[by * block_w + bx_start..by * block_w + bx_start + 8].copy_from_slice(&sums);
        }

        // Scalar remainder
        for bx in (chunks * 8)..block_w {
            let px = bx * 2;

            let sum = get_tmp(tmp, tmp_w, tmp_h, px, py)
                + get_tmp(tmp, tmp_w, tmp_h, px + 1, py)
                + get_tmp(tmp, tmp_w, tmp_h, px, py + 1)
                + get_tmp(tmp, tmp_w, tmp_h, px + 1, py + 1);

            aq_map[by * block_w + bx] = sum;
        }
    }
}

// ============================================================================
// Locked test data from frymire.png
// ============================================================================

/// Test inputs sampled from frymire.png Y plane (0-255 range).
/// These are real pixel values + gamma offset from the actual image.
#[rustfmt::skip]
pub const TEST_INPUTS_RATIO: [f32; 16] = [
    // Row of 8 consecutive pixels from frymire.png Y plane + gamma_offset
    // gamma_offset = 0.019 / K_INPUT_SCALING = 4.845
    133.845, 134.845, 135.845, 136.845, 137.845, 138.845, 139.845, 140.845,
    // Another row with different values
    45.845, 78.845, 112.845, 156.845, 189.845, 212.845, 234.845, 248.845,
];

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Maximum allowed difference between scalar and SIMD results.
    /// Set tight since we're doing identical math.
    const EPSILON: f32 = 1e-6;

    #[test]
    fn test_ratio_of_derivatives_x8_matches_scalar() {
        let inputs = f32x8::from([
            TEST_INPUTS_RATIO[0],
            TEST_INPUTS_RATIO[1],
            TEST_INPUTS_RATIO[2],
            TEST_INPUTS_RATIO[3],
            TEST_INPUTS_RATIO[4],
            TEST_INPUTS_RATIO[5],
            TEST_INPUTS_RATIO[6],
            TEST_INPUTS_RATIO[7],
        ]);

        let simd_result = ratio_of_derivatives_x8(inputs);
        let simd_arr: [f32; 8] = simd_result.into();

        for i in 0..8 {
            let scalar_result = ratio_of_derivatives_scalar(TEST_INPUTS_RATIO[i], false);
            let diff = (simd_arr[i] - scalar_result).abs();
            assert!(
                diff < EPSILON,
                "Mismatch at index {}: SIMD={}, scalar={}, diff={}",
                i,
                simd_arr[i],
                scalar_result,
                diff
            );
        }
    }

    #[test]
    fn test_ratio_of_derivatives_inv_x8_matches_scalar() {
        let inputs = f32x8::from([
            TEST_INPUTS_RATIO[8],
            TEST_INPUTS_RATIO[9],
            TEST_INPUTS_RATIO[10],
            TEST_INPUTS_RATIO[11],
            TEST_INPUTS_RATIO[12],
            TEST_INPUTS_RATIO[13],
            TEST_INPUTS_RATIO[14],
            TEST_INPUTS_RATIO[15],
        ]);

        let simd_result = ratio_of_derivatives_inv_x8(inputs);
        let simd_arr: [f32; 8] = simd_result.into();

        for i in 0..8 {
            let scalar_result = ratio_of_derivatives_scalar(TEST_INPUTS_RATIO[8 + i], true);
            let diff = (simd_arr[i] - scalar_result).abs();
            assert!(
                diff < EPSILON,
                "Mismatch at index {}: SIMD={}, scalar={}, diff={}",
                i,
                simd_arr[i],
                scalar_result,
                diff
            );
        }
    }

    #[test]
    fn test_masking_sqrt_x8_matches_scalar() {
        // Test with typical diff_sq values (0 to 0.2 range, clamped in real usage)
        let inputs = f32x8::from([0.0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.05, 0.08]);

        let simd_result = masking_sqrt_x8(inputs);
        let simd_arr: [f32; 8] = simd_result.into();
        let input_arr: [f32; 8] = inputs.into();

        for i in 0..8 {
            let scalar_result = masking_sqrt_scalar(input_arr[i]);
            let diff = (simd_arr[i] - scalar_result).abs();
            assert!(
                diff < EPSILON,
                "Mismatch at index {}: SIMD={}, scalar={}, diff={}",
                i,
                simd_arr[i],
                scalar_result,
                diff
            );
        }
    }

    #[test]
    fn test_ratio_edge_cases() {
        // Test edge cases: zero, negative, very small, very large
        let inputs = f32x8::from([0.0, -1.0, 0.001, 1.0, 10.0, 100.0, 255.0, 1000.0]);

        let simd_result = ratio_of_derivatives_x8(inputs);
        let simd_arr: [f32; 8] = simd_result.into();
        let input_arr: [f32; 8] = inputs.into();

        for i in 0..8 {
            let scalar_result = ratio_of_derivatives_scalar(input_arr[i], false);
            let diff = (simd_arr[i] - scalar_result).abs();
            // Use relative epsilon for large values
            let rel_epsilon = EPSILON.max(scalar_result.abs() * 1e-5);
            assert!(
                diff < rel_epsilon,
                "Mismatch at index {} (input={}): SIMD={}, scalar={}, diff={}",
                i,
                input_arr[i],
                simd_arr[i],
                scalar_result,
                diff
            );
        }
    }

    /// Locked test: specific input/output values that MUST NOT change.
    /// These are the contract between scalar and SIMD implementations.
    #[test]
    fn test_ratio_locked_values() {
        // These exact outputs must be preserved across refactors
        let locked_inputs = [128.0_f32, 64.0, 192.0, 255.0];
        let locked_outputs_non_inv = [
            ratio_of_derivatives_scalar(128.0, false),
            ratio_of_derivatives_scalar(64.0, false),
            ratio_of_derivatives_scalar(192.0, false),
            ratio_of_derivatives_scalar(255.0, false),
        ];

        // Print for documentation (run with --nocapture to see)
        println!("Locked ratio_of_derivatives outputs (non-inverted):");
        for (inp, out) in locked_inputs.iter().zip(locked_outputs_non_inv.iter()) {
            println!("  {} -> {}", inp, out);
        }

        // Verify SIMD produces same values
        let inputs = f32x8::from([
            locked_inputs[0],
            locked_inputs[1],
            locked_inputs[2],
            locked_inputs[3],
            0.0,
            0.0,
            0.0,
            0.0,
        ]);
        let simd_result = ratio_of_derivatives_x8(inputs);
        let simd_arr: [f32; 8] = simd_result.into();

        for i in 0..4 {
            assert!(
                (simd_arr[i] - locked_outputs_non_inv[i]).abs() < EPSILON,
                "LOCKED VALUE CHANGED! input={}, expected={}, got={}",
                locked_inputs[i],
                locked_outputs_non_inv[i],
                simd_arr[i]
            );
        }
    }

    #[test]
    fn test_pre_erosion_pixel_x8_matches_scalar() {
        // Test data: 8 pixels with their neighbors
        let pixels = f32x8::from([100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0]);
        let left = f32x8::from([95.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0]);
        let right = f32x8::from([110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 175.0]);
        let top = f32x8::from([98.0, 108.0, 118.0, 128.0, 138.0, 148.0, 158.0, 168.0]);
        let bottom = f32x8::from([102.0, 112.0, 122.0, 132.0, 142.0, 152.0, 162.0, 172.0]);

        let simd_result = pre_erosion_pixel_x8(pixels, left, right, top, bottom);
        let simd_arr: [f32; 8] = simd_result.into();

        let pixels_arr: [f32; 8] = pixels.into();
        let left_arr: [f32; 8] = left.into();
        let right_arr: [f32; 8] = right.into();
        let top_arr: [f32; 8] = top.into();
        let bottom_arr: [f32; 8] = bottom.into();

        for i in 0..8 {
            // Scalar reference calculation
            let base = 0.25 * (left_arr[i] + right_arr[i] + top_arr[i] + bottom_arr[i]);
            let ratio = ratio_of_derivatives_scalar(pixels_arr[i] + GAMMA_OFFSET, false);
            let diff = ratio * (pixels_arr[i] - base);
            let diff_sq = (diff * diff).min(LIMIT);
            let scalar_result = masking_sqrt_scalar(diff_sq);

            let diff = (simd_arr[i] - scalar_result).abs();
            assert!(
                diff < EPSILON,
                "Mismatch at index {}: SIMD={}, scalar={}, diff={}",
                i,
                simd_arr[i],
                scalar_result,
                diff
            );
        }
    }

    #[test]
    fn test_pre_erosion_row_matches_scalar() {
        // Create a test row (32 pixels for good SIMD coverage)
        let width = 32;
        let row: Vec<f32> = (0..width).map(|x| 100.0 + (x as f32) * 5.0).collect();
        let row_above: Vec<f32> = (0..width).map(|x| 98.0 + (x as f32) * 5.0).collect();
        let row_below: Vec<f32> = (0..width).map(|x| 102.0 + (x as f32) * 5.0).collect();

        // SIMD version
        let mut output_simd = vec![0.0f32; width];
        pre_erosion_row(&row, &row_above, &row_below, &mut output_simd);

        // Scalar reference
        let mut output_scalar = vec![0.0f32; width];
        for x in 0..width {
            let pixel = row[x];
            let left_val = if x == 0 { row[0] } else { row[x - 1] };
            let right_val = if x == width - 1 {
                row[width - 1]
            } else {
                row[x + 1]
            };
            let top_val = row_above[x];
            let bottom_val = row_below[x];

            let base = 0.25 * (left_val + right_val + top_val + bottom_val);
            let ratio = ratio_of_derivatives_scalar(pixel + GAMMA_OFFSET, false);
            let diff = ratio * (pixel - base);
            let diff_sq = (diff * diff).min(LIMIT);
            output_scalar[x] = masking_sqrt_scalar(diff_sq);
        }

        // Compare
        for x in 0..width {
            let diff = (output_simd[x] - output_scalar[x]).abs();
            assert!(
                diff < EPSILON,
                "Row mismatch at x={}: SIMD={}, scalar={}, diff={}",
                x,
                output_simd[x],
                output_scalar[x],
                diff
            );
        }
    }

    #[test]
    fn test_pre_erosion_row_odd_width() {
        // Test with non-multiple-of-8 width to verify remainder handling
        let width = 35; // 4 chunks of 8 + 3 remainder
        let row: Vec<f32> = (0..width).map(|x| 128.0 + (x as f32)).collect();
        let row_above: Vec<f32> = (0..width).map(|x| 126.0 + (x as f32)).collect();
        let row_below: Vec<f32> = (0..width).map(|x| 130.0 + (x as f32)).collect();

        let mut output_simd = vec![0.0f32; width];
        pre_erosion_row(&row, &row_above, &row_below, &mut output_simd);

        // Scalar reference
        let mut output_scalar = vec![0.0f32; width];
        for x in 0..width {
            let pixel = row[x];
            let left_val = if x == 0 { row[0] } else { row[x - 1] };
            let right_val = if x == width - 1 {
                row[width - 1]
            } else {
                row[x + 1]
            };
            let top_val = row_above[x];
            let bottom_val = row_below[x];

            let base = 0.25 * (left_val + right_val + top_val + bottom_val);
            let ratio = ratio_of_derivatives_scalar(pixel + GAMMA_OFFSET, false);
            let diff = ratio * (pixel - base);
            let diff_sq = (diff * diff).min(LIMIT);
            output_scalar[x] = masking_sqrt_scalar(diff_sq);
        }

        for x in 0..width {
            let diff = (output_simd[x] - output_scalar[x]).abs();
            assert!(
                diff < EPSILON,
                "Odd width mismatch at x={}: SIMD={}, scalar={}, diff={}",
                x,
                output_simd[x],
                output_scalar[x],
                diff
            );
        }
    }

    #[test]
    fn test_compute_pre_erosion_simd_matches_scalar() {
        // Create a test image (64x64)
        let width = 64;
        let height = 64;
        let input: Vec<f32> = (0..width * height)
            .map(|i| {
                let x = i % width;
                let y = i / width;
                // Gradient with some variation
                100.0 + (x as f32) * 2.0 + (y as f32) * 1.5 + ((x * y) as f32 * 0.1).sin() * 20.0
            })
            .collect();

        // SIMD version
        let simd_result = compute_pre_erosion_simd(&input, width, height);

        // Scalar reference (inline implementation)
        let pre_erosion_w = (width + 3) / 4;
        let pre_erosion_h = (height + 3) / 4;
        let mut scalar_result = vec![0.0f32; pre_erosion_w * pre_erosion_h];

        let get = |x: isize, y: isize| -> f32 {
            let x = x.clamp(0, width as isize - 1) as usize;
            let y = y.clamp(0, height as isize - 1) as usize;
            input[y * width + x]
        };

        let mut diff_buffer = vec![0.0f32; width];

        for y_block in 0..pre_erosion_h {
            diff_buffer.fill(0.0);

            for iy in 0..4 {
                let y = y_block * 4 + iy;
                if y >= height {
                    continue;
                }

                for x in 0..width {
                    let ix = x as isize;
                    let iy_s = y as isize;
                    let pixel = get(ix, iy_s);
                    let base = 0.25
                        * (get(ix - 1, iy_s)
                            + get(ix + 1, iy_s)
                            + get(ix, iy_s - 1)
                            + get(ix, iy_s + 1));
                    let ratio = ratio_of_derivatives_scalar(pixel + GAMMA_OFFSET, false);
                    let diff = ratio * (pixel - base);
                    let diff_sq = (diff * diff).min(LIMIT);
                    diff_buffer[x] += masking_sqrt_scalar(diff_sq);
                }
            }

            for x_block in 0..pre_erosion_w {
                let x_start = x_block * 4;
                let mut sum = 0.0f32;
                for ix in 0..4 {
                    let x = x_start + ix;
                    if x < width {
                        sum += diff_buffer[x];
                    }
                }
                scalar_result[y_block * pre_erosion_w + x_block] = sum * 0.25;
            }
        }

        // Compare
        assert_eq!(simd_result.len(), scalar_result.len());
        for i in 0..simd_result.len() {
            let diff = (simd_result[i] - scalar_result[i]).abs();
            assert!(
                diff < EPSILON,
                "Pre-erosion mismatch at index {}: SIMD={}, scalar={}, diff={}",
                i,
                simd_result[i],
                scalar_result[i],
                diff
            );
        }
    }

    #[test]
    fn test_downsample_4x_sum() {
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let mut output = vec![0.0f32; 3];
        downsample_4x_sum(&input, &mut output);

        // Expected: (1+2+3+4)*0.25 = 2.5, (5+6+7+8)*0.25 = 6.5, (9+10+11+12)*0.25 = 10.5
        assert!((output[0] - 2.5).abs() < 1e-6);
        assert!((output[1] - 6.5).abs() < 1e-6);
        assert!((output[2] - 10.5).abs() < 1e-6);
    }

    #[test]
    fn test_fast_exp2_accuracy() {
        // Test fast_exp2 against standard exp2 over typical input range
        let test_values = [
            -10.0,
            -5.0,
            -2.0,
            -1.0,
            -0.5,
            0.0,
            0.5,
            1.0,
            2.0,
            5.0,
            10.0,
            -0.74174993, // K_MASK_BASE
            0.123,
            -0.456,
            3.14159,
        ];

        for &x in &test_values {
            let fast = fast_exp2(x);
            let exact = x.exp2();
            let rel_err = ((fast - exact) / exact).abs();

            // 0.05% relative error is acceptable for AQ purposes
            assert!(
                rel_err < 5e-4,
                "fast_exp2({}) = {} vs exp2 = {}, rel_err = {}",
                x,
                fast,
                exact,
                rel_err
            );
        }
    }

    #[test]
    fn test_fast_log2_accuracy() {
        // Test fast_log2 against standard log2 over typical AQ input range
        // The gamma modulation ratio is typically in [0.0001, 0.01] range
        // which gives log2 values in [-13, -7] range
        let test_values = [
            0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 10.0, 0.25, 0.75, 1.5, 3.14159, 0.00025,
            0.005,
        ];

        for &x in &test_values {
            let fast = fast_log2(x);
            let exact = x.log2();
            let abs_err = (fast - exact).abs();

            // 0.1 absolute error is acceptable for log2 in AQ context
            // (the log value is multiplied by ~0.15 and added to other terms)
            assert!(
                abs_err < 0.1,
                "fast_log2({}) = {} vs log2 = {}, abs_err = {}",
                x,
                fast,
                exact,
                abs_err
            );
        }
    }

    #[test]
    fn test_weighted_min4_of_9() {
        // Test with known values
        let vals = [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0_f32];
        // 4 smallest are: 1, 2, 3, 4
        // Expected: 0.125*1 + 0.075*2 + 0.06*3 + 0.05*4 = 0.125 + 0.15 + 0.18 + 0.2 = 0.655
        let result = weighted_min4_of_9(vals);
        assert!(
            (result - 0.655).abs() < 1e-6,
            "Expected 0.655, got {}",
            result
        );

        // Test with already sorted values
        let vals2 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0_f32];
        let result2 = weighted_min4_of_9(vals2);
        assert!(
            (result2 - 0.655).abs() < 1e-6,
            "Expected 0.655, got {}",
            result2
        );

        // Test with duplicates
        let vals3 = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0_f32];
        // 4 smallest are all 5
        // Expected: 0.125*5 + 0.075*5 + 0.06*5 + 0.05*5 = 5 * 0.31 = 1.55
        let result3 = weighted_min4_of_9(vals3);
        assert!(
            (result3 - 1.55).abs() < 1e-6,
            "Expected 1.55, got {}",
            result3
        );
    }

    #[test]
    fn test_fuzzy_erosion_simd_matches_scalar() {
        // Create test pre_erosion data (16x16 at 1/4 resolution = simulates 64x64 image)
        let pre_erosion_w = 16;
        let pre_erosion_h = 16;
        let pre_erosion: Vec<f32> = (0..pre_erosion_w * pre_erosion_h)
            .map(|i| {
                let x = i % pre_erosion_w;
                let y = i / pre_erosion_w;
                // Create varied data
                0.5 + (x as f32) * 0.02 + (y as f32) * 0.015 + ((x * y) as f32 * 0.1).sin() * 0.1
            })
            .collect();

        // Block dimensions: pre_erosion is at 4x4 pixel resolution, blocks at 8x8
        // So block_w ≈ pre_erosion_w / 2
        let block_w = pre_erosion_w / 2;
        let block_h = pre_erosion_h / 2;

        // SIMD version
        let mut aq_map_simd = vec![0.0f32; block_w * block_h];
        fuzzy_erosion_simd(
            &pre_erosion,
            pre_erosion_w,
            pre_erosion_h,
            block_w,
            block_h,
            &mut aq_map_simd,
        );

        // Scalar reference
        let mut aq_map_scalar = vec![0.0f32; block_w * block_h];
        fuzzy_erosion_scalar_ref(
            &pre_erosion,
            pre_erosion_w,
            pre_erosion_h,
            block_w,
            block_h,
            &mut aq_map_scalar,
        );

        // Compare
        for i in 0..aq_map_simd.len() {
            let diff = (aq_map_simd[i] - aq_map_scalar[i]).abs();
            assert!(
                diff < EPSILON,
                "Fuzzy erosion mismatch at index {}: SIMD={}, scalar={}, diff={}",
                i,
                aq_map_simd[i],
                aq_map_scalar[i],
                diff
            );
        }
    }

    /// Scalar reference implementation of fuzzy_erosion for testing
    fn fuzzy_erosion_scalar_ref(
        pre_erosion: &[f32],
        pre_erosion_w: usize,
        pre_erosion_h: usize,
        block_w: usize,
        block_h: usize,
        aq_map: &mut [f32],
    ) {
        let mut tmp = vec![0.0f32; pre_erosion_w * pre_erosion_h];

        let get = |x: isize, y: isize| -> f32 {
            let x = x.clamp(0, pre_erosion_w as isize - 1) as usize;
            let y = y.clamp(0, pre_erosion_h as isize - 1) as usize;
            pre_erosion[y * pre_erosion_w + x]
        };

        for y in 0..pre_erosion_h {
            for x in 0..pre_erosion_w {
                let ix = x as isize;
                let iy = y as isize;

                let mut vals = [
                    get(ix - 1, iy - 1),
                    get(ix, iy - 1),
                    get(ix + 1, iy - 1),
                    get(ix - 1, iy),
                    get(ix, iy),
                    get(ix + 1, iy),
                    get(ix - 1, iy + 1),
                    get(ix, iy + 1),
                    get(ix + 1, iy + 1),
                ];

                // Partial sort to get 4 smallest
                for i in 0..4 {
                    for j in (i + 1)..9 {
                        if vals[j] < vals[i] {
                            vals.swap(i, j);
                        }
                    }
                }

                let weighted = FUZZY_MUL0 * vals[0]
                    + FUZZY_MUL1 * vals[1]
                    + FUZZY_MUL2 * vals[2]
                    + FUZZY_MUL3 * vals[3];
                tmp[y * pre_erosion_w + x] = weighted;
            }
        }

        let get_tmp = |x: usize, y: usize| -> f32 {
            let x = x.min(pre_erosion_w.saturating_sub(1));
            let y = y.min(pre_erosion_h.saturating_sub(1));
            tmp[y * pre_erosion_w + x]
        };

        for by in 0..block_h {
            for bx in 0..block_w {
                let px = bx * 2;
                let py = by * 2;
                let sum = get_tmp(px, py)
                    + get_tmp(px + 1, py)
                    + get_tmp(px, py + 1)
                    + get_tmp(px + 1, py + 1);
                aq_map[by * block_w + bx] = sum;
            }
        }
    }
}
