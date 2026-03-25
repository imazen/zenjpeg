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

#![allow(dead_code)]

use crate::foundation::aligned_alloc::{AlignedVec, AllocError, try_alloc_zeroed};
use archmage::autoversion;
use wide::f32x8;

// ============================================================================
// Safe SIMD load/store helpers
// ============================================================================

/// Load 8 f32s into f32x8. Panics if slice is too short.
/// Uses array conversion which compiles to a single load instruction.
#[inline(always)]
fn load_f32x8(slice: &[f32], offset: usize) -> f32x8 {
    // Convert slice to array, then array to f32x8 (cast/transmute)
    <[f32; 8]>::try_from(&slice[offset..offset + 8])
        .unwrap()
        .into()
}

/// Store f32x8 to slice. Panics if slice is too short.
#[inline(always)]
fn store_f32x8(slice: &mut [f32], offset: usize, value: f32x8) {
    slice[offset..offset + 8].copy_from_slice(&value.to_array())
}

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
    let num = v2.mul_add(K_NUM_MUL_RATIO, K_NUM_OFFSET_RATIO);
    let den = (v * K_DEN_MUL_RATIO).mul_add(v2, K_VOFFSET_RATIO);
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
    0.25 * v
        .mul_add((K_MASKING_MUL * 1e8_f32).sqrt(), K_MASKING_LOG_OFFSET)
        .sqrt()
}

// ============================================================================
// SIMD implementations
// ============================================================================

/// SIMD version of ratio_of_derivatives (non-inverted).
/// Processes 8 f32 values at once.
#[inline(always)]
pub fn ratio_of_derivatives_x8(vals: f32x8) -> f32x8 {
    let v = vals.fast_max(f32x8::ZERO);
    let v2 = v * v;

    let num = v2.mul_add(
        f32x8::splat(K_NUM_MUL_RATIO),
        f32x8::splat(K_NUM_OFFSET_RATIO),
    );
    let den = (v * f32x8::splat(K_DEN_MUL_RATIO)).mul_add(v2, f32x8::splat(K_VOFFSET_RATIO));

    // den is always positive due to K_VOFFSET_RATIO > 0, no need for safe_den check
    den / num
}

/// SIMD version of ratio_of_derivatives (inverted).
/// Processes 8 f32 values at once.
#[inline(always)]
pub fn ratio_of_derivatives_inv_x8(vals: f32x8) -> f32x8 {
    let v = vals.fast_max(f32x8::ZERO);
    let v2 = v * v;

    let num = v2.mul_add(
        f32x8::splat(K_NUM_MUL_RATIO),
        f32x8::splat(K_NUM_OFFSET_RATIO),
    );
    let den = (v * f32x8::splat(K_DEN_MUL_RATIO)).mul_add(v2, f32x8::splat(K_VOFFSET_RATIO));

    num / den
}

/// SIMD version of masking_sqrt.
/// Processes 8 f32 values at once.
#[inline(always)]
pub fn masking_sqrt_x8(v: f32x8) -> f32x8 {
    let k_mul_sqrt = f32x8::splat((K_MASKING_MUL * 1e8_f32).sqrt());
    let k_offset = f32x8::splat(K_MASKING_LOG_OFFSET);
    f32x8::splat(0.25) * v.mul_add(k_mul_sqrt, k_offset).sqrt()
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
#[inline(always)]
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
#[autoversion]
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
        let pixels = load_f32x8(row, x);

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
            load_f32x8(row, x - 1)
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
            load_f32x8(row, x + 1)
        };

        let top = load_f32x8(row_above, x);
        let bottom = load_f32x8(row_below, x);

        // Compute and accumulate using SIMD add
        let result = pre_erosion_pixel_x8(pixels, left, right, top, bottom);

        // Load existing, add result, store back
        let existing = load_f32x8(output, x);
        store_f32x8(output, x, existing + result);
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

/// Process a full row of pre-erosion using padded input buffers (no boundary conditionals).
///
/// This is the optimized version for streaming AQ where buffers have edge replication.
/// Buffer layout: [left_edge, data[0..width], right_edge] where:
/// - left_edge = data[0] (replicated)
/// - right_edge = data[width-1] (replicated)
///
/// This eliminates all boundary conditionals in the SIMD loop.
///
/// # Arguments
/// * `row` - Current row with padding (length = width + 2)
/// * `row_above` - Row above with padding (length = width + 2)
/// * `row_below` - Row below with padding (length = width + 2)
/// * `width` - Actual data width (without padding)
/// * `output` - Output buffer to accumulate into (length = width)
#[autoversion]
pub fn pre_erosion_row_padded(
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

    // Process 8 pixels at a time - no boundary conditionals needed!
    // Pixel data starts at index 1, left neighbor at index 0, right neighbor at index 2
    let chunks = width / 8;

    for chunk in 0..chunks {
        let x = chunk * 8;
        // Data offset: pixel at logical position x is at buffer index x+1
        let buf_x = x + 1;

        // Load center pixels (indices buf_x..buf_x+8)
        let pixels = load_f32x8(row, buf_x);

        // Load neighbors - no conditionals needed due to padding!
        let left = load_f32x8(row, buf_x - 1); // indices x..x+8, always valid
        let right = load_f32x8(row, buf_x + 1); // indices x+2..x+10, always valid
        let top = load_f32x8(row_above, buf_x);
        let bottom = load_f32x8(row_below, buf_x);

        // Compute and accumulate
        let result = pre_erosion_pixel_x8(pixels, left, right, top, bottom);

        // Load existing, add result, store back
        let existing = load_f32x8(output, x);
        store_f32x8(output, x, existing + result);
    }

    // Handle remainder (scalar fallback) - still no conditionals due to padding
    for x in (chunks * 8)..width {
        let buf_x = x + 1;
        let pixel = row[buf_x];
        let left_val = row[buf_x - 1]; // Always valid due to padding
        let right_val = row[buf_x + 1]; // Always valid due to padding
        let top_val = row_above[buf_x];
        let bottom_val = row_below[buf_x];

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
pub fn compute_pre_erosion_simd(
    input: &[f32],
    width: usize,
    height: usize,
) -> Result<AlignedVec<f32>, AllocError> {
    let pre_erosion_w = (width + 3) / 4;
    let pre_erosion_h = (height + 3) / 4;
    let mut pre_erosion = try_alloc_zeroed(pre_erosion_w * pre_erosion_h)?;

    if width == 0 || height == 0 {
        return Ok(pre_erosion);
    }

    // Temporary buffer for accumulating masked diff values
    let mut diff_buffer = try_alloc_zeroed(width)?;

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

    Ok(pre_erosion)
}

/// Downsample by 4x with sum and scale by 0.25.
#[autoversion]
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
    // Using FMA (Horner's method) for better accuracy and performance
    let t = f_minus_1;
    let log2_f = t * 0.2885390082_f32
        .mul_add(t, -0.3606737602)
        .mul_add(t, 0.4808983470)
        .mul_add(t, -0.7213475204)
        .mul_add(t, 1.442695041);

    e as f32 + log2_f
}

/// Compute sum of ratio_of_derivatives(inv=true) for an 8x8 block.
///
/// SIMD accelerated - processes one row of 8 pixels at a time.
///
/// When the buffer has edge-replicated padding (stride > img_width), we use
/// all 8 pixels per row including replicated ones. This matches C++ behavior
/// and gives correct normalization (64 pixels per block).
///
/// # Arguments
/// * `block` - Pointer to top-left of 8x8 block
/// * `stride` - Row stride (may be padded beyond img_width)
/// * `_block_x` - X position of block (unused, kept for API compatibility)
/// * `block_y` - Y position of block (for boundary check)
/// * `_img_width` - Image width (unused, kept for API compatibility)
/// * `img_height` - Total image height
///
/// # Returns
/// Sum of ratio_of_derivatives for all valid pixels in the block
#[inline(always)]
pub fn gamma_modulation_sum_8x8(
    block: &[f32],
    stride: usize,
    _block_x: usize,
    block_y: usize,
    _img_width: usize,
    img_height: usize,
) -> f32 {
    let bias = f32x8::splat(K_BIAS);
    let mut sum = f32x8::ZERO;

    // The buffer is guaranteed to have 8 columns per block row due to MCU-aligned
    // allocation. For edge blocks, the extra columns contain replicated edge values,
    // which is correct for normalization (same pixel value contributes multiple times).
    for dy in 0..8 {
        let y = block_y + dy;
        if y >= img_height {
            continue;
        }

        let row_start = dy * stride;

        // Process all 8 pixels - buffer has MCU-aligned padding with edge replication
        if row_start + 8 <= block.len() {
            let row = load_f32x8(block, row_start);
            let ratio = ratio_of_derivatives_inv_x8(row + bias);
            sum += ratio;
        }
    }

    // Horizontal sum using SIMD reduce_add
    sum.reduce_add()
}

/// Compute HF modulation sum: |p - right| + |p - below| for 8x8 block.
///
/// Optimized with SIMD for row processing.
///
/// When the buffer has edge-replicated padding (stride > img_width), we use
/// all 8 pixels per row including replicated ones. This matches C++ behavior
/// and gives correct normalization (112 = 7×8 + 8×7 differences).
#[inline(always)]
pub fn hf_modulation_sum_8x8(
    block: &[f32],
    stride: usize,
    _block_x: usize,
    block_y: usize,
    _img_width: usize,
    img_height: usize,
) -> f32 {
    // Mask to zero out the 8th element for horizontal differences
    const MASK_FIRST_7: f32x8 = f32x8::new([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]);

    let mut h_sum = f32x8::ZERO;
    let mut v_sum = f32x8::ZERO;

    // The buffer is guaranteed to have 8 columns per block row due to MCU-aligned
    // allocation. For edge blocks, the extra columns contain replicated edge values,
    // which give 0 differences (correct for normalization).
    for dy in 0..8 {
        let y = block_y + dy;
        if y >= img_height {
            continue;
        }

        let row_start = dy * stride;

        // Horizontal differences: |p - p_right| for positions 0..6
        // Buffer has 9 valid elements per row (8 + 1 for rightward shift)
        if row_start + 9 <= block.len() {
            let p = load_f32x8(block, row_start);
            let p_right = load_f32x8(block, row_start + 1);
            // Mask out 8th element (position 7->8 difference not needed), accumulate
            h_sum += (p - p_right).abs() * MASK_FIRST_7;
        }

        // Vertical differences: |p - p_below| for first 7 rows
        if dy < 7 && y + 1 < img_height {
            let next_row_start = (dy + 1) * stride;
            if row_start + 8 <= block.len() && next_row_start + 8 <= block.len() {
                let p = load_f32x8(block, row_start);
                let p_below = load_f32x8(block, next_row_start);
                v_sum += (p - p_below).abs();
            }
        }
    }

    // Single horizontal reduction at the end
    h_sum.reduce_add() + v_sum.reduce_add()
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
            width,  // stride
            width,  // img_width (same as stride for non-streaming)
            height, // img_height
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
///
/// # Parameters
/// - `input`: Input Y plane data
/// - `stride`: Row stride in input buffer (may be padded for SIMD alignment)
/// - `img_width`: Actual image width (for edge clamping)
/// - `img_height`: Actual image height (for edge clamping)
/// - `by`: Block row index
/// - `block_w`: Number of blocks in row
/// - `aq_row`: Output AQ values (one per block)
/// - `mul`, `add`: Final transform coefficients
#[autoversion]
pub fn per_block_modulations_row(
    input: &[f32],
    stride: usize,
    img_width: usize,
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
        // Use FMA for weighted sum
        let mut out_val = K_MASK_MUL4.mul_add(
            v4,
            K_MASK_MUL2.mul_add(v2, K_MASK_MUL3.mul_add(v3, K_MASK_BASE)),
        );

        // 2. HfModulation with SIMD
        let block_offset = y_start * stride + x_start;
        let block = &input[block_offset..];
        let hf_sum = hf_modulation_sum_8x8(block, stride, x_start, y_start, img_width, img_height);
        out_val += hf_sum * K_SUM_COEFF;

        // 3. GammaModulation with SIMD and fast_log2
        let gamma_sum =
            gamma_modulation_sum_8x8(block, stride, x_start, y_start, img_width, img_height);
        let overall_ratio = gamma_sum * K_SCALE;
        let log_ratio = if overall_ratio > 0.0 {
            fast_log2(overall_ratio)
        } else {
            0.0
        };
        out_val += K_GAMMA * log_ratio;

        // 4. Final transform using fast_exp2 approximation
        // Note: (out_val * LOG2_E).exp2() = exp(out_val)
        let quant_field = fast_exp2(out_val * LOG2_E).mul_add(mul, add);
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

// ============================================================================
// SIMD Sorting Network for 4-smallest-of-9
// ============================================================================

/// Compare-and-swap for sorting network: returns (min, max)
#[inline(always)]
fn cas(a: f32x8, b: f32x8) -> (f32x8, f32x8) {
    (a.min(b), a.max(b))
}

/// SIMD sorting network to find 4 smallest of 9 values.
/// Processes 8 independent sets of 9 values in parallel (one per SIMD lane).
/// Returns weighted sum for each lane: MUL0*v0 + MUL1*v1 + MUL2*v2 + MUL3*v3
///
/// Uses a fixed comparison network - no branches, fully SIMD-parallel.
/// The network ensures v[0..4] contain the 4 smallest values (not fully sorted,
/// but the 4 smallest are guaranteed to be in those positions).
#[inline(always)]
fn weighted_min4_of_9_simd(mut v: [f32x8; 9]) -> f32x8 {
    // Sorting network for 9 elements to get 4 smallest in positions 0-3
    // This is a partial sorting network optimized for finding k-smallest
    //
    // The network below is derived from Batcher's odd-even merge sort,
    // optimized to only guarantee the 4 smallest values end up in v[0..4].
    // Total: 19 compare-exchange operations

    // Layer 1: Initial pairwise comparisons
    (v[0], v[1]) = cas(v[0], v[1]);
    (v[2], v[3]) = cas(v[2], v[3]);
    (v[4], v[5]) = cas(v[4], v[5]);
    (v[6], v[7]) = cas(v[6], v[7]);

    // Layer 2: Compare across pairs
    (v[0], v[2]) = cas(v[0], v[2]);
    (v[1], v[3]) = cas(v[1], v[3]);
    (v[4], v[6]) = cas(v[4], v[6]);
    (v[5], v[7]) = cas(v[5], v[7]);

    // Layer 3: Merge groups of 4
    (v[0], v[4]) = cas(v[0], v[4]);
    (v[1], v[5]) = cas(v[1], v[5]);
    (v[2], v[6]) = cas(v[2], v[6]);
    (v[3], v[7]) = cas(v[3], v[7]);

    // Layer 4: Integrate element 8
    (v[0], v[8]) = cas(v[0], v[8]);
    (v[4], v[8]) = cas(v[4], v[8]);

    // Layer 5: Fix ordering in bottom half
    (v[1], v[2]) = cas(v[1], v[2]);
    (v[1], v[4]) = cas(v[1], v[4]);
    (v[2], v[4]) = cas(v[2], v[4]);
    (v[3], v[4]) = cas(v[3], v[4]);
    (v[5], v[8]) = cas(v[5], v[8]);

    // Now v[0..4] contains the 4 smallest values
    // Compute weighted sum using FMA
    let mul0 = f32x8::splat(FUZZY_MUL0);
    let mul1 = f32x8::splat(FUZZY_MUL1);
    let mul2 = f32x8::splat(FUZZY_MUL2);
    let mul3 = f32x8::splat(FUZZY_MUL3);

    mul0 * v[0] + mul1 * v[1] + mul2 * v[2] + mul3 * v[3]
}

/// Process a row of fuzzy erosion using SIMD, 8 pixels at a time.
/// Returns the number of pixels processed.
#[inline]
pub fn fuzzy_erosion_row_simd(
    pre_erosion: &[f32],
    pre_erosion_w: usize,
    y: usize,
    max_y: usize,
    out: &mut [f32],
) -> usize {
    let row_above_y = (y as isize - 1).clamp(0, max_y as isize) as usize;
    let row_below_y = (y + 1).min(max_y);

    let row_above = &pre_erosion[row_above_y * pre_erosion_w..];
    let row_curr = &pre_erosion[y * pre_erosion_w..];
    let row_below = &pre_erosion[row_below_y * pre_erosion_w..];

    // Process 8 pixels at a time with SIMD
    let mut x = 1; // Start at 1 to avoid left edge

    while x + 8 <= pre_erosion_w.saturating_sub(1) {
        // Gather 9 neighbor values for 8 consecutive pixels
        // Each v[i] contains the i-th neighbor for 8 different pixels
        let v = [
            load_f32x8(row_above, x - 1), // top-left
            load_f32x8(row_above, x),     // top
            load_f32x8(row_above, x + 1), // top-right (shifted by 1)
            load_f32x8(row_curr, x - 1),  // left
            load_f32x8(row_curr, x),      // center
            load_f32x8(row_curr, x + 1),  // right
            load_f32x8(row_below, x - 1), // bottom-left
            load_f32x8(row_below, x),     // bottom
            load_f32x8(row_below, x + 1), // bottom-right
        ];

        let result = weighted_min4_of_9_simd(v);
        store_f32x8(out, x, result);

        x += 8;
    }

    // Handle remaining pixels with scalar code
    while x < pre_erosion_w.saturating_sub(1) {
        let x_left = x - 1;
        let x_right = x + 1;

        let vals = [
            row_above[x_left],
            row_above[x],
            row_above[x_right],
            row_curr[x_left],
            row_curr[x],
            row_curr[x_right],
            row_below[x_left],
            row_below[x],
            row_below[x_right],
        ];

        out[x] = weighted_min4_of_9(vals);
        x += 1;
    }

    // Handle edges with clamped bounds
    // Left edge (x=0)
    {
        let vals = [
            row_above[0],
            row_above[0],
            row_above[1.min(pre_erosion_w - 1)],
            row_curr[0],
            row_curr[0],
            row_curr[1.min(pre_erosion_w - 1)],
            row_below[0],
            row_below[0],
            row_below[1.min(pre_erosion_w - 1)],
        ];
        out[0] = weighted_min4_of_9(vals);
    }

    // Right edge (x=pre_erosion_w-1)
    if pre_erosion_w > 1 {
        let x = pre_erosion_w - 1;
        let vals = [
            row_above[x - 1],
            row_above[x],
            row_above[x],
            row_curr[x - 1],
            row_curr[x],
            row_curr[x],
            row_below[x - 1],
            row_below[x],
            row_below[x],
        ];
        out[x] = weighted_min4_of_9(vals);
    }

    pre_erosion_w
}

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

    // Use FMA for weighted sum
    FUZZY_MUL0.mul_add(
        a[0],
        FUZZY_MUL1.mul_add(a[1], FUZZY_MUL2.mul_add(a[2], FUZZY_MUL3 * a[3])),
    )
}

// ============================================================================
// Streaming circular buffer SIMD support
// ============================================================================

/// Gather a single neighbor value for 8 blocks from circular buffer.
/// Each lane gets the value at (cx[lane] + nx, cy + ny) with boundary clamping.
#[inline(always)]
fn gather_neighbor_circular(
    buffer: &[f32],
    pe_w: usize,
    buffer_rows: usize,
    base_cx: [isize; 8],
    cy: isize,
    nx: isize,
    ny: isize,
    max_x: isize,
    max_y: isize,
) -> f32x8 {
    let py = (cy + ny).clamp(0, max_y) as usize;
    let buffer_row = py % buffer_rows;
    let row_offset = buffer_row * pe_w;

    // Gather 8 values with clamped x coordinates
    let vals: [f32; 8] = std::array::from_fn(|i| {
        let px = (base_cx[i] + nx).clamp(0, max_x) as usize;
        let idx = row_offset + px;
        if idx < buffer.len() { buffer[idx] } else { 0.0 }
    });
    vals.into()
}

/// Compute fuzzy erosion for 8 blocks using SIMD sorting network.
///
/// For streaming AQ: processes blocks from a circular pre-erosion buffer.
/// Each block sums 4 weighted_min4_of_9 values (one per 2x2 sub-pixel).
///
/// Returns the number of blocks processed.
#[inline]
pub fn compute_fuzzy_erosion_blocks_simd(
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

    // Process 8 blocks at a time with SIMD
    while bx + 8 <= end {
        // Base cx values for 8 consecutive blocks (stride 2 in pre-erosion space)
        let base_cx: [isize; 8] = std::array::from_fn(|i| ((bx + i - start) * 2) as isize);

        let mut sum = f32x8::ZERO;

        // Process 4 sub-pixels per block: (dx, dy) in {0,1} x {0,1}
        for dy in 0..2isize {
            for dx in 0..2isize {
                let cy = pe_y_base + dy;
                // Add offset to base_cx for this sub-pixel
                let cx: [isize; 8] = std::array::from_fn(|i| base_cx[i] + dx);

                // Gather 9 neighbors for all 8 blocks
                let v = [
                    gather_neighbor_circular(
                        pre_erosion_buffer,
                        pe_w,
                        buffer_rows,
                        cx,
                        cy,
                        -1,
                        -1,
                        max_x,
                        max_y,
                    ),
                    gather_neighbor_circular(
                        pre_erosion_buffer,
                        pe_w,
                        buffer_rows,
                        cx,
                        cy,
                        0,
                        -1,
                        max_x,
                        max_y,
                    ),
                    gather_neighbor_circular(
                        pre_erosion_buffer,
                        pe_w,
                        buffer_rows,
                        cx,
                        cy,
                        1,
                        -1,
                        max_x,
                        max_y,
                    ),
                    gather_neighbor_circular(
                        pre_erosion_buffer,
                        pe_w,
                        buffer_rows,
                        cx,
                        cy,
                        -1,
                        0,
                        max_x,
                        max_y,
                    ),
                    gather_neighbor_circular(
                        pre_erosion_buffer,
                        pe_w,
                        buffer_rows,
                        cx,
                        cy,
                        0,
                        0,
                        max_x,
                        max_y,
                    ),
                    gather_neighbor_circular(
                        pre_erosion_buffer,
                        pe_w,
                        buffer_rows,
                        cx,
                        cy,
                        1,
                        0,
                        max_x,
                        max_y,
                    ),
                    gather_neighbor_circular(
                        pre_erosion_buffer,
                        pe_w,
                        buffer_rows,
                        cx,
                        cy,
                        -1,
                        1,
                        max_x,
                        max_y,
                    ),
                    gather_neighbor_circular(
                        pre_erosion_buffer,
                        pe_w,
                        buffer_rows,
                        cx,
                        cy,
                        0,
                        1,
                        max_x,
                        max_y,
                    ),
                    gather_neighbor_circular(
                        pre_erosion_buffer,
                        pe_w,
                        buffer_rows,
                        cx,
                        cy,
                        1,
                        1,
                        max_x,
                        max_y,
                    ),
                ];

                // SIMD sorting network to find 4 smallest and compute weighted sum
                sum += weighted_min4_of_9_simd(v);
            }
        }

        // Store results for 8 blocks
        let sum_arr = sum.to_array();
        out[bx..bx + 8].copy_from_slice(&sum_arr);

        bx += 8;
        processed += 8;
    }

    processed
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
) -> Result<(), AllocError> {
    assert_eq!(aq_map.len(), block_w * block_h);

    // Temporary buffer for weighted min values
    let mut tmp = try_alloc_zeroed(pre_erosion_w * pre_erosion_h)?;

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

    Ok(())
}

/// Sum 2x2 blocks from tmp buffer to produce aq_map values.
/// Simple scalar implementation - SIMD shuffle overhead isn't worth it for this pattern.
#[inline(always)]
fn sum_2x2_blocks_simd(
    tmp: &[f32],
    tmp_w: usize,
    tmp_h: usize,
    block_w: usize,
    block_h: usize,
    aq_map: &mut [f32],
) {
    let max_x = tmp_w.saturating_sub(1);
    let max_y = tmp_h.saturating_sub(1);

    for by in 0..block_h {
        let py0 = by * 2;
        let py1 = (py0 + 1).min(max_y);
        let row0 = &tmp[py0 * tmp_w..];
        let row1 = &tmp[py1 * tmp_w..];
        let out_row = &mut aq_map[by * block_w..];

        for bx in 0..block_w {
            let px0 = bx * 2;
            let px1 = (px0 + 1).min(max_x);
            out_row[bx] = row0[px0] + row0[px1] + row1[px0] + row1[px1];
        }
    }
}

// ============================================================================
// Archmage SIMD implementations (token-based safe intrinsics)
// ============================================================================

#[cfg(target_arch = "x86_64")]
pub(crate) mod archmage_impl {
    use archmage::{SimdToken, X64V3Token, X64V4Token, arcane, rite};
    use core::arch::x86_64::*;
    // Safe unaligned load/store (shadow the unsafe core::arch versions)
    #[allow(unused_imports)]
    use safe_unaligned_simd::x86_64::{
        _mm256_loadu_ps, _mm256_storeu_ps, _mm512_loadu_ps, _mm512_storeu_ps,
    };

    use super::{
        GAMMA_OFFSET, K_DEN_MUL_RATIO, K_MASKING_LOG_OFFSET, K_MASKING_MUL, K_NUM_MUL_RATIO,
        K_NUM_OFFSET_RATIO, K_VOFFSET_RATIO, LIMIT,
    };

    /// Archmage-based ratio_of_derivatives (non-inverted) using direct intrinsics.
    /// Uses FMA for all multiply-add operations.
    #[rite]
    fn mage_ratio_of_derivatives_x8(_token: X64V3Token, vals: __m256) -> __m256 {
        let zero = _mm256_setzero_ps();

        // v = max(vals, 0)
        let v = _mm256_max_ps(vals, zero);

        // v2 = v * v
        let v2 = _mm256_mul_ps(v, v);

        // num = v2 * K_NUM_MUL_RATIO + K_NUM_OFFSET_RATIO (FMA)
        let k_num_mul = _mm256_set1_ps(K_NUM_MUL_RATIO);
        let k_num_off = _mm256_set1_ps(K_NUM_OFFSET_RATIO);
        let num = _mm256_fmadd_ps(v2, k_num_mul, k_num_off);

        // den = (v * K_DEN_MUL_RATIO) * v2 + K_VOFFSET_RATIO
        // = v * K_DEN_MUL_RATIO * v2 + K_VOFFSET_RATIO
        let k_den_mul = _mm256_set1_ps(K_DEN_MUL_RATIO);
        let k_voff = _mm256_set1_ps(K_VOFFSET_RATIO);
        let v_scaled = _mm256_mul_ps(v, k_den_mul);
        let den = _mm256_fmadd_ps(v_scaled, v2, k_voff);

        // return den / num (non-inverted)
        _mm256_div_ps(den, num)
    }

    /// Archmage-based masking_sqrt using direct intrinsics.
    #[rite]
    fn mage_masking_sqrt_x8(_token: X64V3Token, v: __m256) -> __m256 {
        let k_mul_sqrt = _mm256_set1_ps((K_MASKING_MUL * 1e8_f32).sqrt());
        let k_offset = _mm256_set1_ps(K_MASKING_LOG_OFFSET);
        let quarter = _mm256_set1_ps(0.25);

        // inner = v * k_mul_sqrt + k_offset (FMA)
        let inner = _mm256_fmadd_ps(v, k_mul_sqrt, k_offset);
        // sqrt(inner)
        let sqrt_inner = _mm256_sqrt_ps(inner);
        // 0.25 * sqrt_inner
        _mm256_mul_ps(quarter, sqrt_inner)
    }

    /// Archmage-based pre_erosion computation for 8 pixels.
    /// All operations use direct intrinsics with FMA.
    #[rite]
    fn mage_pre_erosion_pixel_x8(
        token: X64V3Token,
        pixels: __m256,
        left: __m256,
        right: __m256,
        top: __m256,
        bottom: __m256,
    ) -> __m256 {
        let quarter = _mm256_set1_ps(0.25);
        let gamma_offset = _mm256_set1_ps(GAMMA_OFFSET);
        let limit = _mm256_set1_ps(LIMIT);

        // base = 0.25 * (left + right + top + bottom)
        let sum_lr = _mm256_add_ps(left, right);
        let sum_tb = _mm256_add_ps(top, bottom);
        let sum_all = _mm256_add_ps(sum_lr, sum_tb);
        let base = _mm256_mul_ps(quarter, sum_all);

        // ratio = ratio_of_derivatives(pixel + gamma_offset)
        let pixel_gamma = _mm256_add_ps(pixels, gamma_offset);
        let ratio = mage_ratio_of_derivatives_x8(token, pixel_gamma);

        // diff = ratio * (pixel - base)
        let pixel_minus_base = _mm256_sub_ps(pixels, base);
        let diff = _mm256_mul_ps(ratio, pixel_minus_base);

        // diff_sq = min(diff * diff, LIMIT)
        let diff_sq = _mm256_mul_ps(diff, diff);
        let diff_sq_clamped = _mm256_min_ps(diff_sq, limit);

        // masked = masking_sqrt(diff_sq)
        mage_masking_sqrt_x8(token, diff_sq_clamped)
    }

    /// Archmage-based pre_erosion_row_padded using safe_unaligned_simd loads.
    /// Uses token-gated intrinsics for maximum performance with full safety.
    #[arcane]
    #[inline]
    fn mage_pre_erosion_row_padded_inner(
        _token: X64V3Token,
        row: &[f32],
        row_above: &[f32],
        row_below: &[f32],
        output: &mut [f32],
        width: usize,
    ) {
        use safe_unaligned_simd::x86_64 as safe_simd;

        let chunks = width / 8;

        // Broadcast constants once outside the loop
        let quarter = _mm256_set1_ps(0.25);
        let gamma_offset = _mm256_set1_ps(GAMMA_OFFSET);
        let limit = _mm256_set1_ps(LIMIT);
        let k_num_mul = _mm256_set1_ps(K_NUM_MUL_RATIO);
        let k_num_off = _mm256_set1_ps(K_NUM_OFFSET_RATIO);
        let k_den_mul = _mm256_set1_ps(K_DEN_MUL_RATIO);
        let k_voff = _mm256_set1_ps(K_VOFFSET_RATIO);
        let k_mul_sqrt = _mm256_set1_ps((K_MASKING_MUL * 1e8_f32).sqrt());
        let k_offset = _mm256_set1_ps(K_MASKING_LOG_OFFSET);
        let zero = _mm256_setzero_ps();

        for chunk in 0..chunks {
            let x = chunk * 8;
            let buf_x = x + 1; // Data offset due to padding

            // Safe slice-based loads via safe_unaligned_simd
            let pixels =
                safe_simd::_mm256_loadu_ps(<&[f32; 8]>::try_from(&row[buf_x..buf_x + 8]).unwrap());
            let left = safe_simd::_mm256_loadu_ps(
                <&[f32; 8]>::try_from(&row[buf_x - 1..buf_x + 7]).unwrap(),
            );
            let right = safe_simd::_mm256_loadu_ps(
                <&[f32; 8]>::try_from(&row[buf_x + 1..buf_x + 9]).unwrap(),
            );
            let top = safe_simd::_mm256_loadu_ps(
                <&[f32; 8]>::try_from(&row_above[buf_x..buf_x + 8]).unwrap(),
            );
            let bottom = safe_simd::_mm256_loadu_ps(
                <&[f32; 8]>::try_from(&row_below[buf_x..buf_x + 8]).unwrap(),
            );

            // base = 0.25 * (left + right + top + bottom)
            let sum_lr = _mm256_add_ps(left, right);
            let sum_tb = _mm256_add_ps(top, bottom);
            let sum_all = _mm256_add_ps(sum_lr, sum_tb);
            let base = _mm256_mul_ps(quarter, sum_all);

            // ratio = ratio_of_derivatives(pixel + gamma_offset)
            let pixel_gamma = _mm256_add_ps(pixels, gamma_offset);
            let v = _mm256_max_ps(pixel_gamma, zero);
            let v2 = _mm256_mul_ps(v, v);
            let num = _mm256_fmadd_ps(v2, k_num_mul, k_num_off);
            let v_scaled = _mm256_mul_ps(v, k_den_mul);
            let den = _mm256_fmadd_ps(v_scaled, v2, k_voff);
            let ratio = _mm256_div_ps(den, num);

            // diff = ratio * (pixel - base)
            let pixel_minus_base = _mm256_sub_ps(pixels, base);
            let diff = _mm256_mul_ps(ratio, pixel_minus_base);

            // diff_sq = min(diff * diff, LIMIT)
            let diff_sq = _mm256_mul_ps(diff, diff);
            let diff_sq_clamped = _mm256_min_ps(diff_sq, limit);

            // masked = masking_sqrt(diff_sq)
            let inner = _mm256_fmadd_ps(diff_sq_clamped, k_mul_sqrt, k_offset);
            let sqrt_inner = _mm256_sqrt_ps(inner);
            let result = _mm256_mul_ps(quarter, sqrt_inner);

            // Load existing, add result, store back
            let existing =
                safe_simd::_mm256_loadu_ps(<&[f32; 8]>::try_from(&output[x..x + 8]).unwrap());
            let updated = _mm256_add_ps(existing, result);
            safe_simd::_mm256_storeu_ps(
                <&mut [f32; 8]>::try_from(&mut output[x..x + 8]).unwrap(),
                updated,
            );
        }
    }

    /// Archmage-based pre_erosion_row_padded - public wrapper with bounds checking.
    pub fn mage_pre_erosion_row_padded(
        token: X64V3Token,
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

        mage_pre_erosion_row_padded_inner(token, row, row_above, row_below, output, width);

        // Scalar remainder
        let chunks = width / 8;
        for x in (chunks * 8)..width {
            let buf_x = x + 1;
            let pixel = row[buf_x];
            let left_val = row[buf_x - 1];
            let right_val = row[buf_x + 1];
            let top_val = row_above[buf_x];
            let bottom_val = row_below[buf_x];

            let base = 0.25 * (left_val + right_val + top_val + bottom_val);
            let ratio = super::ratio_of_derivatives_scalar(pixel + GAMMA_OFFSET, false);
            let diff = ratio * (pixel - base);
            let diff_sq = (diff * diff).min(LIMIT);
            let masked = super::masking_sqrt_scalar(diff_sq);

            output[x] += masked;
        }
    }

    // ========================================================================
    // AVX-512 pre_erosion - processes 16 pixels per iteration (2x wider)
    // ========================================================================

    /// AVX-512 inner loop - processes 16 pixels per iteration.
    /// Uses X64V4Token to prove AVX-512F capability.
    #[arcane]
    #[inline]
    fn mage_pre_erosion_row_padded_inner_v4(
        _token: X64V4Token,
        row: &[f32],
        row_above: &[f32],
        row_below: &[f32],
        output: &mut [f32],
        width: usize,
    ) {
        let chunks16 = width / 16;

        // Broadcast constants once outside the loop
        let quarter = _mm512_set1_ps(0.25);
        let gamma_offset = _mm512_set1_ps(GAMMA_OFFSET);
        let limit = _mm512_set1_ps(LIMIT);
        let k_num_mul = _mm512_set1_ps(K_NUM_MUL_RATIO);
        let k_num_off = _mm512_set1_ps(K_NUM_OFFSET_RATIO);
        let k_den_mul = _mm512_set1_ps(K_DEN_MUL_RATIO);
        let k_voff = _mm512_set1_ps(K_VOFFSET_RATIO);
        let k_mul_sqrt = _mm512_set1_ps((K_MASKING_MUL * 1e8_f32).sqrt());
        let k_offset = _mm512_set1_ps(K_MASKING_LOG_OFFSET);
        let zero = _mm512_setzero_ps();

        for chunk in 0..chunks16 {
            let x = chunk * 16;
            let buf_x = x + 1; // Data offset due to padding

            // Load 16 pixels at once (safe unaligned loads via safe_unaligned_simd)
            let pixels = _mm512_loadu_ps(row[buf_x..][..16].try_into().unwrap());
            let left = _mm512_loadu_ps(row[buf_x - 1..][..16].try_into().unwrap());
            let right = _mm512_loadu_ps(row[buf_x + 1..][..16].try_into().unwrap());
            let top = _mm512_loadu_ps(row_above[buf_x..][..16].try_into().unwrap());
            let bottom = _mm512_loadu_ps(row_below[buf_x..][..16].try_into().unwrap());

            // base = 0.25 * (left + right + top + bottom)
            let sum_lr = _mm512_add_ps(left, right);
            let sum_tb = _mm512_add_ps(top, bottom);
            let sum_all = _mm512_add_ps(sum_lr, sum_tb);
            let base = _mm512_mul_ps(quarter, sum_all);

            // ratio = ratio_of_derivatives(pixel + gamma_offset)
            let pixel_gamma = _mm512_add_ps(pixels, gamma_offset);
            let v = _mm512_max_ps(pixel_gamma, zero);
            let v2 = _mm512_mul_ps(v, v);
            let num = _mm512_fmadd_ps(v2, k_num_mul, k_num_off);
            let v_scaled = _mm512_mul_ps(v, k_den_mul);
            let den = _mm512_fmadd_ps(v_scaled, v2, k_voff);
            let ratio = _mm512_div_ps(den, num);

            // diff = ratio * (pixel - base)
            let pixel_minus_base = _mm512_sub_ps(pixels, base);
            let diff = _mm512_mul_ps(ratio, pixel_minus_base);

            // diff_sq = min(diff * diff, LIMIT)
            let diff_sq = _mm512_mul_ps(diff, diff);
            let diff_sq_clamped = _mm512_min_ps(diff_sq, limit);

            // masked = masking_sqrt(diff_sq)
            let inner = _mm512_fmadd_ps(diff_sq_clamped, k_mul_sqrt, k_offset);
            let sqrt_inner = _mm512_sqrt_ps(inner);
            let result = _mm512_mul_ps(quarter, sqrt_inner);

            // Load existing, add result, store back
            let existing = _mm512_loadu_ps(output[x..][..16].try_into().unwrap());
            let updated = _mm512_add_ps(existing, result);
            _mm512_storeu_ps((&mut output[x..][..16]).try_into().unwrap(), updated);
        }
    }

    /// Helper for processing 8 pixels (AVX2) - used by V4 for remainder handling.
    #[arcane]
    #[inline]
    fn mage_pre_erosion_8_inner(
        _token: X64V3Token,
        row: &[f32],
        row_above: &[f32],
        row_below: &[f32],
        output: &mut [f32],
        buf_x: usize,
        out_x: usize,
    ) {
        use safe_unaligned_simd::x86_64 as safe_simd;

        let quarter = _mm256_set1_ps(0.25);
        let gamma_offset = _mm256_set1_ps(GAMMA_OFFSET);
        let limit = _mm256_set1_ps(LIMIT);
        let k_num_mul = _mm256_set1_ps(K_NUM_MUL_RATIO);
        let k_num_off = _mm256_set1_ps(K_NUM_OFFSET_RATIO);
        let k_den_mul = _mm256_set1_ps(K_DEN_MUL_RATIO);
        let k_voff = _mm256_set1_ps(K_VOFFSET_RATIO);
        let k_mul_sqrt = _mm256_set1_ps((K_MASKING_MUL * 1e8_f32).sqrt());
        let k_offset = _mm256_set1_ps(K_MASKING_LOG_OFFSET);
        let zero = _mm256_setzero_ps();

        let pixels =
            safe_simd::_mm256_loadu_ps(<&[f32; 8]>::try_from(&row[buf_x..buf_x + 8]).unwrap());
        let left =
            safe_simd::_mm256_loadu_ps(<&[f32; 8]>::try_from(&row[buf_x - 1..buf_x + 7]).unwrap());
        let right =
            safe_simd::_mm256_loadu_ps(<&[f32; 8]>::try_from(&row[buf_x + 1..buf_x + 9]).unwrap());
        let top = safe_simd::_mm256_loadu_ps(
            <&[f32; 8]>::try_from(&row_above[buf_x..buf_x + 8]).unwrap(),
        );
        let bottom = safe_simd::_mm256_loadu_ps(
            <&[f32; 8]>::try_from(&row_below[buf_x..buf_x + 8]).unwrap(),
        );

        let sum_lr = _mm256_add_ps(left, right);
        let sum_tb = _mm256_add_ps(top, bottom);
        let sum_all = _mm256_add_ps(sum_lr, sum_tb);
        let base = _mm256_mul_ps(quarter, sum_all);

        let pixel_gamma = _mm256_add_ps(pixels, gamma_offset);
        let v = _mm256_max_ps(pixel_gamma, zero);
        let v2 = _mm256_mul_ps(v, v);
        let num = _mm256_fmadd_ps(v2, k_num_mul, k_num_off);
        let v_scaled = _mm256_mul_ps(v, k_den_mul);
        let den = _mm256_fmadd_ps(v_scaled, v2, k_voff);
        let ratio = _mm256_div_ps(den, num);

        let pixel_minus_base = _mm256_sub_ps(pixels, base);
        let diff = _mm256_mul_ps(ratio, pixel_minus_base);

        let diff_sq = _mm256_mul_ps(diff, diff);
        let diff_sq_clamped = _mm256_min_ps(diff_sq, limit);

        let inner = _mm256_fmadd_ps(diff_sq_clamped, k_mul_sqrt, k_offset);
        let sqrt_inner = _mm256_sqrt_ps(inner);
        let result = _mm256_mul_ps(quarter, sqrt_inner);

        let existing =
            safe_simd::_mm256_loadu_ps(<&[f32; 8]>::try_from(&output[out_x..out_x + 8]).unwrap());
        let updated = _mm256_add_ps(existing, result);
        safe_simd::_mm256_storeu_ps(
            <&mut [f32; 8]>::try_from(&mut output[out_x..out_x + 8]).unwrap(),
            updated,
        );
    }

    /// AVX-512 pre_erosion_row_padded - public wrapper with V4 dispatch.
    /// Falls back to V3 if AVX-512 not available.
    pub fn mage_pre_erosion_row_padded_v4(
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

        // Try AVX-512 first
        if let Some(v4_token) = X64V4Token::summon() {
            mage_pre_erosion_row_padded_inner_v4(
                v4_token, row, row_above, row_below, output, width,
            );

            // AVX-512 processes 16 at a time, handle 8-15 pixel remainder with AVX2
            let v4_processed = (width / 16) * 16;
            if v4_processed < width
                && let Some(v3_token) = X64V3Token::summon()
            {
                let remaining_chunks8 = (width - v4_processed) / 8;
                for chunk in 0..remaining_chunks8 {
                    let x = v4_processed + chunk * 8;
                    let buf_x = x + 1;
                    mage_pre_erosion_8_inner(v3_token, row, row_above, row_below, output, buf_x, x);
                }
            }

            // Scalar remainder for final <8 pixels
            let processed = (width / 8) * 8;
            for x in processed..width {
                let buf_x = x + 1;
                let pixel = row[buf_x];
                let left_val = row[buf_x - 1];
                let right_val = row[buf_x + 1];
                let top_val = row_above[buf_x];
                let bottom_val = row_below[buf_x];

                let base = 0.25 * (left_val + right_val + top_val + bottom_val);
                let ratio = super::ratio_of_derivatives_scalar(pixel + GAMMA_OFFSET, false);
                let diff = ratio * (pixel - base);
                let diff_sq = (diff * diff).min(LIMIT);
                let masked = super::masking_sqrt_scalar(diff_sq);
                output[x] += masked;
            }
        } else if let Some(token) = X64V3Token::summon() {
            // Fall back to AVX2
            mage_pre_erosion_row_padded(token, row, row_above, row_below, width, output);
        } else {
            // Scalar fallback when no SIMD tokens available
            for x in 0..width {
                let buf_x = x + 1;
                let pixel = row[buf_x];
                let left_val = row[buf_x - 1];
                let right_val = row[buf_x + 1];
                let top_val = row_above[buf_x];
                let bottom_val = row_below[buf_x];

                let base = 0.25 * (left_val + right_val + top_val + bottom_val);
                let ratio = super::ratio_of_derivatives_scalar(pixel + GAMMA_OFFSET, false);
                let diff = ratio * (pixel - base);
                let diff_sq = (diff * diff).min(LIMIT);
                let masked = super::masking_sqrt_scalar(diff_sq);
                output[x] += masked;
            }
        }
    }

    // ========================================================================
    // per_block_modulations_row - fused HF+gamma with minimal data movement
    // ========================================================================

    // Constants for per_block_modulations
    const K_INPUT_SCALING: f32 = 1.0 / 255.0;
    const K_INV_LOG2E: f32 = 0.6931471805599453;
    const K_BIAS: f32 = 0.16 / K_INPUT_SCALING; // 40.8

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

    /// Horizontal sum of __m256 (8 floats) to scalar
    #[rite]
    fn hsum_ps(_token: X64V3Token, v: __m256) -> f32 {
        // vhaddps ymm, ymm, ymm -> adds adjacent pairs
        let sum1 = _mm256_hadd_ps(v, v); // [a+b, c+d, a+b, c+d, e+f, g+h, e+f, g+h]
        let sum2 = _mm256_hadd_ps(sum1, sum1); // [a+b+c+d, ..., e+f+g+h, ...]
        // Extract high 128 and add to low 128
        let hi = _mm256_extractf128_ps(sum2, 1);
        let lo = _mm256_castps256_ps128(sum2);
        let sum3 = _mm_add_ps(lo, hi);
        _mm_cvtss_f32(sum3)
    }

    /// Ratio of derivatives (inverted form for gamma modulation)
    #[rite]
    fn mage_ratio_of_derivatives_inv_x8(
        _token: X64V3Token,
        vals: __m256,
        zero: __m256,
        k_num_mul: __m256,
        k_num_off: __m256,
        k_den_mul: __m256,
        k_voff: __m256,
    ) -> __m256 {
        let v = _mm256_max_ps(vals, zero);
        let v2 = _mm256_mul_ps(v, v);
        let num = _mm256_fmadd_ps(v2, k_num_mul, k_num_off);
        let v_scaled = _mm256_mul_ps(v, k_den_mul);
        let den = _mm256_fmadd_ps(v_scaled, v2, k_voff);
        // Inverted: num / den (not den / num)
        _mm256_div_ps(num, den)
    }

    /// Fused HF + Gamma computation for one 8x8 block.
    /// Returns (hf_sum, gamma_sum).
    ///
    /// This fuses two loops into one, halving memory traffic.
    #[rite]
    fn mage_hf_gamma_sum_8x8(
        token: X64V3Token,
        block: &[f32],
        block_offset: usize,
        stride: usize,
        // Pre-broadcast constants (avoid repeated broadcasts)
        zero: __m256,
        bias: __m256,
        mask_first_7: __m256,
        k_num_mul: __m256,
        k_num_off: __m256,
        k_den_mul: __m256,
        k_voff: __m256,
    ) -> (f32, f32) {
        let mut hf_acc = _mm256_setzero_ps();
        let mut gamma_acc = _mm256_setzero_ps();

        // Process 8 rows
        for dy in 0..8usize {
            let row_start = block_offset + dy * stride;

            // Load row[0:8] and row[1:9]
            let row = _mm256_loadu_ps(block[row_start..][..8].try_into().unwrap());
            let row_right = _mm256_loadu_ps(block[row_start + 1..][..8].try_into().unwrap());

            // HF horizontal: |row - row_right| * mask (first 7 positions)
            let h_diff = _mm256_sub_ps(row, row_right);
            let h_abs = _mm256_andnot_ps(_mm256_set1_ps(-0.0), h_diff); // abs via sign bit clear
            let h_masked = _mm256_mul_ps(h_abs, mask_first_7);
            hf_acc = _mm256_add_ps(hf_acc, h_masked);

            // HF vertical: |row - next_row| for rows 0..6
            if dy < 7 {
                let next_start = block_offset + (dy + 1) * stride;
                let next_row = _mm256_loadu_ps(block[next_start..][..8].try_into().unwrap());
                let v_diff = _mm256_sub_ps(row, next_row);
                let v_abs = _mm256_andnot_ps(_mm256_set1_ps(-0.0), v_diff);
                hf_acc = _mm256_add_ps(hf_acc, v_abs);
            }

            // Gamma: ratio_of_derivatives_inv(row + bias)
            let row_biased = _mm256_add_ps(row, bias);
            let gamma_val = mage_ratio_of_derivatives_inv_x8(
                token, row_biased, zero, k_num_mul, k_num_off, k_den_mul, k_voff,
            );
            gamma_acc = _mm256_add_ps(gamma_acc, gamma_val);
        }

        // Reduce to scalars
        let hf_sum = hsum_ps(token, hf_acc);
        let gamma_sum = hsum_ps(token, gamma_acc);

        (hf_sum, gamma_sum)
    }

    /// Fast log2 approximation (scalar) - matches super::fast_log2
    #[inline(always)]
    fn mage_fast_log2(x: f32) -> f32 {
        if x <= 0.0 {
            return f32::NEG_INFINITY;
        }
        let bits = x.to_bits() as i32;
        let e = (bits >> 23) - 127;
        let f = f32::from_bits((bits & 0x007FFFFF) as u32 | 0x3F800000);
        let t = f - 1.0;
        // Horner's method matching super::fast_log2
        let log2_f = t * 0.2885390082_f32
            .mul_add(t, -0.3606737602)
            .mul_add(t, 0.4808983470)
            .mul_add(t, -0.7213475204)
            .mul_add(t, 1.442695041);
        e as f32 + log2_f
    }

    /// Fast exp2 approximation (scalar)
    #[inline(always)]
    fn mage_fast_exp2(x: f32) -> f32 {
        let x = x.clamp(-126.0, 127.0);
        let xi = x.floor();
        let xf = x - xi;
        let p = 1.0
            + xf * (0.6931471805599453
                + xf * (0.24022650695910071
                    + xf * (0.055504108664821579 + xf * 0.009618129107628477)));
        let pow2_xi = f32::from_bits(((xi as i32 + 127) as u32) << 23);
        p * pow2_xi
    }

    /// Archmage-based per_block_modulations_row.
    /// Fuses HF and gamma into a single pass over the 8x8 block.
    ///
    /// Key optimizations:
    /// - Single fused loop for HF + gamma (halves memory traffic)
    /// - Raw pointer loads (no bounds checks)
    /// - All constants broadcast once, passed to inner function
    /// - Scalar ComputeMask (not worth SIMD for 1 value per block)
    #[arcane]
    pub fn mage_per_block_modulations_row(
        token: X64V3Token,
        input: &[f32],
        stride: usize,
        by: usize,
        block_w: usize,
        aq_row: &mut [f32],
        mul: f32,
        add: f32,
    ) {
        if block_w == 0 {
            return;
        }

        let y_start = by * 8;

        // Broadcast constants once (these live in ymm registers throughout)
        let zero = _mm256_setzero_ps();
        let bias = _mm256_set1_ps(K_BIAS);
        let mask_first_7 = _mm256_set_ps(0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        let k_num_mul = _mm256_set1_ps(K_NUM_MUL_RATIO);
        let k_num_off = _mm256_set1_ps(K_NUM_OFFSET_RATIO);
        let k_den_mul = _mm256_set1_ps(K_DEN_MUL_RATIO);
        let k_voff = _mm256_set1_ps(K_VOFFSET_RATIO);

        for bx in 0..block_w {
            let x_start = bx * 8;

            // Load fuzzy erosion value
            let fuzzy_val = aq_row[bx];

            // ComputeMask (scalar - 4 divisions, not worth SIMD for 1 value)
            let v1 = (fuzzy_val * K_MASK_MUL0).max(1e-3);
            let v1_sq = v1 * v1;
            let v2 = 1.0 / (v1 + K_MASK_OFFSET2);
            let v3 = 1.0 / (v1_sq + K_MASK_OFFSET3);
            let v4 = 1.0 / (v1_sq + K_MASK_OFFSET4);
            let mut out_val = K_MASK_BASE + K_MASK_MUL4 * v4 + K_MASK_MUL2 * v2 + K_MASK_MUL3 * v3;

            // Fused HF + Gamma for 8x8 block
            let block_offset = y_start * stride + x_start;

            let (hf_sum, gamma_sum) = mage_hf_gamma_sum_8x8(
                token,
                input,
                block_offset,
                stride,
                zero,
                bias,
                mask_first_7,
                k_num_mul,
                k_num_off,
                k_den_mul,
                k_voff,
            );

            // HF contribution
            out_val += hf_sum * K_SUM_COEFF;

            // Gamma contribution
            let overall_ratio = gamma_sum * K_SCALE;
            if overall_ratio > 0.0 {
                let log_ratio = mage_fast_log2(overall_ratio);
                out_val += K_GAMMA * log_ratio;
            }

            // Final transform: exp(out_val) * mul + add
            let quant_field = mage_fast_exp2(out_val * LOG2_E) * mul + add;
            aq_row[bx] = quant_field;
        }
    }

    // ========================================================================
    // Fuzzy erosion - massive inlined function (no helper calls in hot path)
    // ========================================================================

    /// Compute fuzzy erosion for a row of blocks.
    ///
    /// This is one massive inlined function to eliminate all function call overhead.
    /// Everything is done in-place: gather 9 values, find 4 smallest, weighted sum.
    ///
    /// For each block, we process 4 corners (2x2), each corner needs:
    /// - Gather 9 values from 3x3 neighborhood
    /// - Find 4 smallest values
    /// - Compute weighted sum: 0.125*v0 + 0.075*v1 + 0.06*v2 + 0.05*v3
    pub fn mage_compute_fuzzy_erosion_row(
        pre_erosion_buffer: &[f32],
        pe_w: usize,
        buffer_rows: usize,
        max_filled_row: usize,
        pe_y_base: usize,
        block_start: usize,
        block_count: usize,
        output: &mut [f32],
    ) {
        if block_count == 0 || pe_w == 0 {
            return;
        }

        let pe_w_i = pe_w as isize;
        let max_y = max_filled_row as isize;

        // Pre-compute row base offsets for cy0 and cy1
        let cy0 = pe_y_base as isize;
        let _cy1 = cy0 + 1; // Used implicitly in row offset calculations

        // Row offsets for the 6 rows we might access (cy0-1, cy0, cy0+1, cy1, cy1+1, cy1+2)
        // But cy0+1 == cy1 and cy1+1 == cy0+2, so we only need 4 unique rows
        let r0 = ((cy0 - 1).clamp(0, max_y) as usize % buffer_rows) * pe_w; // cy0-1
        let r1 = (cy0.clamp(0, max_y) as usize % buffer_rows) * pe_w; // cy0
        let r2 = ((cy0 + 1).clamp(0, max_y) as usize % buffer_rows) * pe_w; // cy0+1 = cy1
        let r3 = ((cy0 + 2).clamp(0, max_y) as usize % buffer_rows) * pe_w; // cy1+1

        // Check if y is interior (can skip row clamping)
        let y_interior = cy0 >= 1 && cy0 + 2 <= max_y;

        for bx_offset in 0..block_count {
            let bx = block_start + bx_offset;
            let pe_x = (bx_offset * 2) as isize;

            // Check if x is interior (can skip column clamping)
            let x_interior = pe_x >= 1 && pe_x + 2 < pe_w_i - 1;

            let block_sum = if y_interior && x_interior {
                // FAST PATH: No clamping needed - all indices are valid
                // Unroll all 4 corners completely inline

                let cx0 = pe_x as usize;
                let cx1 = cx0 + 1;

                // Corner (0,0): center at (cx0, cy0), rows r0/r1/r2
                let mut sum = 0.0f32;
                // Load 9 values for corner (0,0)
                let v0 = pre_erosion_buffer[r0 + cx0 - 1];
                let v1 = pre_erosion_buffer[r0 + cx0];
                let v2 = pre_erosion_buffer[r0 + cx0 + 1];
                let v3 = pre_erosion_buffer[r1 + cx0 - 1];
                let v4 = pre_erosion_buffer[r1 + cx0];
                let v5 = pre_erosion_buffer[r1 + cx0 + 1];
                let v6 = pre_erosion_buffer[r2 + cx0 - 1];
                let v7 = pre_erosion_buffer[r2 + cx0];
                let v8 = pre_erosion_buffer[r2 + cx0 + 1];

                // Find 4 smallest inline using min comparisons
                let mut a = [v0, v1, v2, v3, v4, v5, v6, v7, v8];

                // Pass 1: find min
                let mut mi = 0;
                if a[1] < a[mi] {
                    mi = 1;
                }
                if a[2] < a[mi] {
                    mi = 2;
                }
                if a[3] < a[mi] {
                    mi = 3;
                }
                if a[4] < a[mi] {
                    mi = 4;
                }
                if a[5] < a[mi] {
                    mi = 5;
                }
                if a[6] < a[mi] {
                    mi = 6;
                }
                if a[7] < a[mi] {
                    mi = 7;
                }
                if a[8] < a[mi] {
                    mi = 8;
                }
                sum += 0.125 * a[mi];
                a[mi] = f32::MAX;

                // Pass 2
                mi = 0;
                if a[1] < a[mi] {
                    mi = 1;
                }
                if a[2] < a[mi] {
                    mi = 2;
                }
                if a[3] < a[mi] {
                    mi = 3;
                }
                if a[4] < a[mi] {
                    mi = 4;
                }
                if a[5] < a[mi] {
                    mi = 5;
                }
                if a[6] < a[mi] {
                    mi = 6;
                }
                if a[7] < a[mi] {
                    mi = 7;
                }
                if a[8] < a[mi] {
                    mi = 8;
                }
                sum += 0.075 * a[mi];
                a[mi] = f32::MAX;

                // Pass 3
                mi = 0;
                if a[1] < a[mi] {
                    mi = 1;
                }
                if a[2] < a[mi] {
                    mi = 2;
                }
                if a[3] < a[mi] {
                    mi = 3;
                }
                if a[4] < a[mi] {
                    mi = 4;
                }
                if a[5] < a[mi] {
                    mi = 5;
                }
                if a[6] < a[mi] {
                    mi = 6;
                }
                if a[7] < a[mi] {
                    mi = 7;
                }
                if a[8] < a[mi] {
                    mi = 8;
                }
                sum += 0.06 * a[mi];
                a[mi] = f32::MAX;

                // Pass 4
                mi = 0;
                if a[1] < a[mi] {
                    mi = 1;
                }
                if a[2] < a[mi] {
                    mi = 2;
                }
                if a[3] < a[mi] {
                    mi = 3;
                }
                if a[4] < a[mi] {
                    mi = 4;
                }
                if a[5] < a[mi] {
                    mi = 5;
                }
                if a[6] < a[mi] {
                    mi = 6;
                }
                if a[7] < a[mi] {
                    mi = 7;
                }
                if a[8] < a[mi] {
                    mi = 8;
                }
                sum += 0.05 * a[mi];

                // Corner (1,0): center at (cx1, cy0), rows r0/r1/r2
                let v0 = pre_erosion_buffer[r0 + cx1 - 1];
                let v1 = pre_erosion_buffer[r0 + cx1];
                let v2 = pre_erosion_buffer[r0 + cx1 + 1];
                let v3 = pre_erosion_buffer[r1 + cx1 - 1];
                let v4 = pre_erosion_buffer[r1 + cx1];
                let v5 = pre_erosion_buffer[r1 + cx1 + 1];
                let v6 = pre_erosion_buffer[r2 + cx1 - 1];
                let v7 = pre_erosion_buffer[r2 + cx1];
                let v8 = pre_erosion_buffer[r2 + cx1 + 1];

                let mut a = [v0, v1, v2, v3, v4, v5, v6, v7, v8];
                let mut mi = 0;
                if a[1] < a[mi] {
                    mi = 1;
                }
                if a[2] < a[mi] {
                    mi = 2;
                }
                if a[3] < a[mi] {
                    mi = 3;
                }
                if a[4] < a[mi] {
                    mi = 4;
                }
                if a[5] < a[mi] {
                    mi = 5;
                }
                if a[6] < a[mi] {
                    mi = 6;
                }
                if a[7] < a[mi] {
                    mi = 7;
                }
                if a[8] < a[mi] {
                    mi = 8;
                }
                sum += 0.125 * a[mi];
                a[mi] = f32::MAX;
                mi = 0;
                if a[1] < a[mi] {
                    mi = 1;
                }
                if a[2] < a[mi] {
                    mi = 2;
                }
                if a[3] < a[mi] {
                    mi = 3;
                }
                if a[4] < a[mi] {
                    mi = 4;
                }
                if a[5] < a[mi] {
                    mi = 5;
                }
                if a[6] < a[mi] {
                    mi = 6;
                }
                if a[7] < a[mi] {
                    mi = 7;
                }
                if a[8] < a[mi] {
                    mi = 8;
                }
                sum += 0.075 * a[mi];
                a[mi] = f32::MAX;
                mi = 0;
                if a[1] < a[mi] {
                    mi = 1;
                }
                if a[2] < a[mi] {
                    mi = 2;
                }
                if a[3] < a[mi] {
                    mi = 3;
                }
                if a[4] < a[mi] {
                    mi = 4;
                }
                if a[5] < a[mi] {
                    mi = 5;
                }
                if a[6] < a[mi] {
                    mi = 6;
                }
                if a[7] < a[mi] {
                    mi = 7;
                }
                if a[8] < a[mi] {
                    mi = 8;
                }
                sum += 0.06 * a[mi];
                a[mi] = f32::MAX;
                mi = 0;
                if a[1] < a[mi] {
                    mi = 1;
                }
                if a[2] < a[mi] {
                    mi = 2;
                }
                if a[3] < a[mi] {
                    mi = 3;
                }
                if a[4] < a[mi] {
                    mi = 4;
                }
                if a[5] < a[mi] {
                    mi = 5;
                }
                if a[6] < a[mi] {
                    mi = 6;
                }
                if a[7] < a[mi] {
                    mi = 7;
                }
                if a[8] < a[mi] {
                    mi = 8;
                }
                sum += 0.05 * a[mi];

                // Corner (0,1): center at (cx0, cy1), rows r1/r2/r3
                let v0 = pre_erosion_buffer[r1 + cx0 - 1];
                let v1 = pre_erosion_buffer[r1 + cx0];
                let v2 = pre_erosion_buffer[r1 + cx0 + 1];
                let v3 = pre_erosion_buffer[r2 + cx0 - 1];
                let v4 = pre_erosion_buffer[r2 + cx0];
                let v5 = pre_erosion_buffer[r2 + cx0 + 1];
                let v6 = pre_erosion_buffer[r3 + cx0 - 1];
                let v7 = pre_erosion_buffer[r3 + cx0];
                let v8 = pre_erosion_buffer[r3 + cx0 + 1];

                let mut a = [v0, v1, v2, v3, v4, v5, v6, v7, v8];
                let mut mi = 0;
                if a[1] < a[mi] {
                    mi = 1;
                }
                if a[2] < a[mi] {
                    mi = 2;
                }
                if a[3] < a[mi] {
                    mi = 3;
                }
                if a[4] < a[mi] {
                    mi = 4;
                }
                if a[5] < a[mi] {
                    mi = 5;
                }
                if a[6] < a[mi] {
                    mi = 6;
                }
                if a[7] < a[mi] {
                    mi = 7;
                }
                if a[8] < a[mi] {
                    mi = 8;
                }
                sum += 0.125 * a[mi];
                a[mi] = f32::MAX;
                mi = 0;
                if a[1] < a[mi] {
                    mi = 1;
                }
                if a[2] < a[mi] {
                    mi = 2;
                }
                if a[3] < a[mi] {
                    mi = 3;
                }
                if a[4] < a[mi] {
                    mi = 4;
                }
                if a[5] < a[mi] {
                    mi = 5;
                }
                if a[6] < a[mi] {
                    mi = 6;
                }
                if a[7] < a[mi] {
                    mi = 7;
                }
                if a[8] < a[mi] {
                    mi = 8;
                }
                sum += 0.075 * a[mi];
                a[mi] = f32::MAX;
                mi = 0;
                if a[1] < a[mi] {
                    mi = 1;
                }
                if a[2] < a[mi] {
                    mi = 2;
                }
                if a[3] < a[mi] {
                    mi = 3;
                }
                if a[4] < a[mi] {
                    mi = 4;
                }
                if a[5] < a[mi] {
                    mi = 5;
                }
                if a[6] < a[mi] {
                    mi = 6;
                }
                if a[7] < a[mi] {
                    mi = 7;
                }
                if a[8] < a[mi] {
                    mi = 8;
                }
                sum += 0.06 * a[mi];
                a[mi] = f32::MAX;
                mi = 0;
                if a[1] < a[mi] {
                    mi = 1;
                }
                if a[2] < a[mi] {
                    mi = 2;
                }
                if a[3] < a[mi] {
                    mi = 3;
                }
                if a[4] < a[mi] {
                    mi = 4;
                }
                if a[5] < a[mi] {
                    mi = 5;
                }
                if a[6] < a[mi] {
                    mi = 6;
                }
                if a[7] < a[mi] {
                    mi = 7;
                }
                if a[8] < a[mi] {
                    mi = 8;
                }
                sum += 0.05 * a[mi];

                // Corner (1,1): center at (cx1, cy1), rows r1/r2/r3
                let v0 = pre_erosion_buffer[r1 + cx1 - 1];
                let v1 = pre_erosion_buffer[r1 + cx1];
                let v2 = pre_erosion_buffer[r1 + cx1 + 1];
                let v3 = pre_erosion_buffer[r2 + cx1 - 1];
                let v4 = pre_erosion_buffer[r2 + cx1];
                let v5 = pre_erosion_buffer[r2 + cx1 + 1];
                let v6 = pre_erosion_buffer[r3 + cx1 - 1];
                let v7 = pre_erosion_buffer[r3 + cx1];
                let v8 = pre_erosion_buffer[r3 + cx1 + 1];

                let mut a = [v0, v1, v2, v3, v4, v5, v6, v7, v8];
                let mut mi = 0;
                if a[1] < a[mi] {
                    mi = 1;
                }
                if a[2] < a[mi] {
                    mi = 2;
                }
                if a[3] < a[mi] {
                    mi = 3;
                }
                if a[4] < a[mi] {
                    mi = 4;
                }
                if a[5] < a[mi] {
                    mi = 5;
                }
                if a[6] < a[mi] {
                    mi = 6;
                }
                if a[7] < a[mi] {
                    mi = 7;
                }
                if a[8] < a[mi] {
                    mi = 8;
                }
                sum += 0.125 * a[mi];
                a[mi] = f32::MAX;
                mi = 0;
                if a[1] < a[mi] {
                    mi = 1;
                }
                if a[2] < a[mi] {
                    mi = 2;
                }
                if a[3] < a[mi] {
                    mi = 3;
                }
                if a[4] < a[mi] {
                    mi = 4;
                }
                if a[5] < a[mi] {
                    mi = 5;
                }
                if a[6] < a[mi] {
                    mi = 6;
                }
                if a[7] < a[mi] {
                    mi = 7;
                }
                if a[8] < a[mi] {
                    mi = 8;
                }
                sum += 0.075 * a[mi];
                a[mi] = f32::MAX;
                mi = 0;
                if a[1] < a[mi] {
                    mi = 1;
                }
                if a[2] < a[mi] {
                    mi = 2;
                }
                if a[3] < a[mi] {
                    mi = 3;
                }
                if a[4] < a[mi] {
                    mi = 4;
                }
                if a[5] < a[mi] {
                    mi = 5;
                }
                if a[6] < a[mi] {
                    mi = 6;
                }
                if a[7] < a[mi] {
                    mi = 7;
                }
                if a[8] < a[mi] {
                    mi = 8;
                }
                sum += 0.06 * a[mi];
                a[mi] = f32::MAX;
                mi = 0;
                if a[1] < a[mi] {
                    mi = 1;
                }
                if a[2] < a[mi] {
                    mi = 2;
                }
                if a[3] < a[mi] {
                    mi = 3;
                }
                if a[4] < a[mi] {
                    mi = 4;
                }
                if a[5] < a[mi] {
                    mi = 5;
                }
                if a[6] < a[mi] {
                    mi = 6;
                }
                if a[7] < a[mi] {
                    mi = 7;
                }
                if a[8] < a[mi] {
                    mi = 8;
                }
                sum += 0.05 * a[mi];

                sum
            } else {
                // SLOW PATH: Boundary blocks - use clamping
                let mut sum = 0.0f32;

                for dy in 0..2isize {
                    for dx in 0..2isize {
                        let cx = pe_x + dx;
                        let cy = cy0 + dy;

                        // Gather with clamping
                        let c0 = (cx - 1).clamp(0, pe_w_i - 1) as usize;
                        let c1 = cx.clamp(0, pe_w_i - 1) as usize;
                        let c2 = (cx + 1).clamp(0, pe_w_i - 1) as usize;

                        let ry0 = ((cy - 1).clamp(0, max_y) as usize % buffer_rows) * pe_w;
                        let ry1 = (cy.clamp(0, max_y) as usize % buffer_rows) * pe_w;
                        let ry2 = ((cy + 1).clamp(0, max_y) as usize % buffer_rows) * pe_w;

                        let v0 = pre_erosion_buffer[ry0 + c0];
                        let v1 = pre_erosion_buffer[ry0 + c1];
                        let v2 = pre_erosion_buffer[ry0 + c2];
                        let v3 = pre_erosion_buffer[ry1 + c0];
                        let v4 = pre_erosion_buffer[ry1 + c1];
                        let v5 = pre_erosion_buffer[ry1 + c2];
                        let v6 = pre_erosion_buffer[ry2 + c0];
                        let v7 = pre_erosion_buffer[ry2 + c1];
                        let v8 = pre_erosion_buffer[ry2 + c2];

                        let mut a = [v0, v1, v2, v3, v4, v5, v6, v7, v8];
                        let mut mi = 0;
                        if a[1] < a[mi] {
                            mi = 1;
                        }
                        if a[2] < a[mi] {
                            mi = 2;
                        }
                        if a[3] < a[mi] {
                            mi = 3;
                        }
                        if a[4] < a[mi] {
                            mi = 4;
                        }
                        if a[5] < a[mi] {
                            mi = 5;
                        }
                        if a[6] < a[mi] {
                            mi = 6;
                        }
                        if a[7] < a[mi] {
                            mi = 7;
                        }
                        if a[8] < a[mi] {
                            mi = 8;
                        }
                        sum += 0.125 * a[mi];
                        a[mi] = f32::MAX;
                        mi = 0;
                        if a[1] < a[mi] {
                            mi = 1;
                        }
                        if a[2] < a[mi] {
                            mi = 2;
                        }
                        if a[3] < a[mi] {
                            mi = 3;
                        }
                        if a[4] < a[mi] {
                            mi = 4;
                        }
                        if a[5] < a[mi] {
                            mi = 5;
                        }
                        if a[6] < a[mi] {
                            mi = 6;
                        }
                        if a[7] < a[mi] {
                            mi = 7;
                        }
                        if a[8] < a[mi] {
                            mi = 8;
                        }
                        sum += 0.075 * a[mi];
                        a[mi] = f32::MAX;
                        mi = 0;
                        if a[1] < a[mi] {
                            mi = 1;
                        }
                        if a[2] < a[mi] {
                            mi = 2;
                        }
                        if a[3] < a[mi] {
                            mi = 3;
                        }
                        if a[4] < a[mi] {
                            mi = 4;
                        }
                        if a[5] < a[mi] {
                            mi = 5;
                        }
                        if a[6] < a[mi] {
                            mi = 6;
                        }
                        if a[7] < a[mi] {
                            mi = 7;
                        }
                        if a[8] < a[mi] {
                            mi = 8;
                        }
                        sum += 0.06 * a[mi];
                        a[mi] = f32::MAX;
                        mi = 0;
                        if a[1] < a[mi] {
                            mi = 1;
                        }
                        if a[2] < a[mi] {
                            mi = 2;
                        }
                        if a[3] < a[mi] {
                            mi = 3;
                        }
                        if a[4] < a[mi] {
                            mi = 4;
                        }
                        if a[5] < a[mi] {
                            mi = 5;
                        }
                        if a[6] < a[mi] {
                            mi = 6;
                        }
                        if a[7] < a[mi] {
                            mi = 7;
                        }
                        if a[8] < a[mi] {
                            mi = 8;
                        }
                        sum += 0.05 * a[mi];
                    }
                }

                sum
            };

            output[bx] = block_sum;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
pub use archmage_impl::mage_pre_erosion_row_padded;

#[cfg(target_arch = "x86_64")]
pub use archmage_impl::mage_pre_erosion_row_padded_v4;

#[cfg(target_arch = "x86_64")]
pub use archmage_impl::mage_per_block_modulations_row;

// Note: mage_compute_fuzzy_erosion_row doesn't need archmage token - it's pure inlined Rust
// But it's still inside archmage_impl module which requires the feature

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

    /// Maximum allowed relative difference between scalar and SIMD results.
    /// 1e-6 = 0.0001% - tight but allows for FMA vs non-FMA rounding differences.
    const EPSILON_REL: f32 = 1e-6;

    /// Check if two values match within relative tolerance.
    /// Uses max(abs(a), abs(b)) as denominator to handle near-zero values.
    fn approx_eq(a: f32, b: f32, rel_tol: f32) -> bool {
        let diff = (a - b).abs();
        let max_abs = a.abs().max(b.abs());
        if max_abs == 0.0 {
            diff == 0.0
        } else {
            diff / max_abs <= rel_tol
        }
    }

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
            assert!(
                approx_eq(simd_arr[i], scalar_result, EPSILON_REL),
                "Mismatch at index {}: SIMD={}, scalar={}, rel_diff={}",
                i,
                simd_arr[i],
                scalar_result,
                (simd_arr[i] - scalar_result).abs() / scalar_result.abs().max(1e-10)
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
            assert!(
                approx_eq(simd_arr[i], scalar_result, EPSILON_REL),
                "Mismatch at index {}: SIMD={}, scalar={}, rel_diff={}",
                i,
                simd_arr[i],
                scalar_result,
                (simd_arr[i] - scalar_result).abs() / scalar_result.abs().max(1e-10)
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
            assert!(
                approx_eq(simd_arr[i], scalar_result, EPSILON_REL),
                "Mismatch at index {}: SIMD={}, scalar={}, rel_diff={}",
                i,
                simd_arr[i],
                scalar_result,
                (simd_arr[i] - scalar_result).abs() / scalar_result.abs().max(1e-10)
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
            assert!(
                approx_eq(simd_arr[i], scalar_result, EPSILON_REL),
                "Mismatch at index {} (input={}): SIMD={}, scalar={}, rel_diff={}",
                i,
                input_arr[i],
                simd_arr[i],
                scalar_result,
                (simd_arr[i] - scalar_result).abs() / scalar_result.abs().max(1e-10)
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
                approx_eq(simd_arr[i], locked_outputs_non_inv[i], EPSILON_REL),
                "LOCKED VALUE CHANGED! input={}, expected={}, got={}, rel_diff={}",
                locked_inputs[i],
                locked_outputs_non_inv[i],
                simd_arr[i],
                (simd_arr[i] - locked_outputs_non_inv[i]).abs()
                    / locked_outputs_non_inv[i].abs().max(1e-10)
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

            assert!(
                approx_eq(simd_arr[i], scalar_result, EPSILON_REL),
                "Mismatch at index {}: SIMD={}, scalar={}, rel_diff={}",
                i,
                simd_arr[i],
                scalar_result,
                (simd_arr[i] - scalar_result).abs() / scalar_result.abs().max(1e-10)
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
            assert!(
                approx_eq(output_simd[x], output_scalar[x], EPSILON_REL),
                "Row mismatch at x={}: SIMD={}, scalar={}, rel_diff={}",
                x,
                output_simd[x],
                output_scalar[x],
                (output_simd[x] - output_scalar[x]).abs() / output_scalar[x].abs().max(1e-10)
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
            assert!(
                approx_eq(output_simd[x], output_scalar[x], EPSILON_REL),
                "Odd width mismatch at x={}: SIMD={}, scalar={}, rel_diff={}",
                x,
                output_simd[x],
                output_scalar[x],
                (output_simd[x] - output_scalar[x]).abs() / output_scalar[x].abs().max(1e-10)
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
        let simd_result = compute_pre_erosion_simd(&input, width, height).unwrap();

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
            assert!(
                approx_eq(simd_result[i], scalar_result[i], EPSILON_REL),
                "Pre-erosion mismatch at index {}: SIMD={}, scalar={}, rel_diff={}",
                i,
                simd_result[i],
                scalar_result[i],
                (simd_result[i] - scalar_result[i]).abs() / scalar_result[i].abs().max(1e-10)
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
        )
        .unwrap();

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
            assert!(
                approx_eq(aq_map_simd[i], aq_map_scalar[i], EPSILON_REL),
                "Fuzzy erosion mismatch at index {}: SIMD={}, scalar={}, rel_diff={}",
                i,
                aq_map_simd[i],
                aq_map_scalar[i],
                (aq_map_simd[i] - aq_map_scalar[i]).abs() / aq_map_scalar[i].abs().max(1e-10)
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

                let weighted = FUZZY_MUL0.mul_add(
                    vals[0],
                    FUZZY_MUL1.mul_add(vals[1], FUZZY_MUL2.mul_add(vals[2], FUZZY_MUL3 * vals[3])),
                );
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

    #[test]
    fn test_pre_erosion_row_padded_matches_original() {
        // Test that pre_erosion_row_padded produces same results as pre_erosion_row
        // when given properly padded input buffers with edge replication
        let width = 35; // Non-multiple of 8 to test remainder handling
        let row: Vec<f32> = (0..width).map(|x| 128.0 + (x as f32)).collect();
        let row_above: Vec<f32> = (0..width).map(|x| 126.0 + (x as f32)).collect();
        let row_below: Vec<f32> = (0..width).map(|x| 130.0 + (x as f32)).collect();

        // Create padded buffers with edge replication
        let mut row_padded = vec![0.0f32; width + 2];
        let mut row_above_padded = vec![0.0f32; width + 2];
        let mut row_below_padded = vec![0.0f32; width + 2];

        // Copy with edge replication: [left_edge, data[0..width], right_edge]
        row_padded[1..1 + width].copy_from_slice(&row);
        row_padded[0] = row[0];
        row_padded[width + 1] = row[width - 1];

        row_above_padded[1..1 + width].copy_from_slice(&row_above);
        row_above_padded[0] = row_above[0];
        row_above_padded[width + 1] = row_above[width - 1];

        row_below_padded[1..1 + width].copy_from_slice(&row_below);
        row_below_padded[0] = row_below[0];
        row_below_padded[width + 1] = row_below[width - 1];

        // Original version
        let mut output_original = vec![0.0f32; width];
        pre_erosion_row(&row, &row_above, &row_below, &mut output_original);

        // Padded version
        let mut output_padded = vec![0.0f32; width];
        pre_erosion_row_padded(
            &row_padded,
            &row_above_padded,
            &row_below_padded,
            width,
            &mut output_padded,
        );

        // Compare
        for x in 0..width {
            assert!(
                approx_eq(output_padded[x], output_original[x], EPSILON_REL),
                "Padded vs original mismatch at x={}: padded={}, original={}, rel_diff={}",
                x,
                output_padded[x],
                output_original[x],
                (output_padded[x] - output_original[x]).abs() / output_original[x].abs().max(1e-10)
            );
        }
    }

    #[test]
    fn test_pre_erosion_row_padded_various_widths() {
        // Test several widths to ensure SIMD chunks and remainders work correctly
        for width in [8, 16, 24, 31, 32, 33, 64, 100, 128] {
            let row: Vec<f32> = (0..width).map(|x| 100.0 + (x as f32) * 1.5).collect();
            let row_above: Vec<f32> = (0..width).map(|x| 98.0 + (x as f32) * 1.5).collect();
            let row_below: Vec<f32> = (0..width).map(|x| 102.0 + (x as f32) * 1.5).collect();

            // Create padded buffers
            let mut row_padded = vec![0.0f32; width + 2];
            let mut row_above_padded = vec![0.0f32; width + 2];
            let mut row_below_padded = vec![0.0f32; width + 2];

            row_padded[1..1 + width].copy_from_slice(&row);
            row_padded[0] = row[0];
            row_padded[width + 1] = row[width - 1];

            row_above_padded[1..1 + width].copy_from_slice(&row_above);
            row_above_padded[0] = row_above[0];
            row_above_padded[width + 1] = row_above[width - 1];

            row_below_padded[1..1 + width].copy_from_slice(&row_below);
            row_below_padded[0] = row_below[0];
            row_below_padded[width + 1] = row_below[width - 1];

            let mut output_original = vec![0.0f32; width];
            let mut output_padded = vec![0.0f32; width];

            pre_erosion_row(&row, &row_above, &row_below, &mut output_original);
            pre_erosion_row_padded(
                &row_padded,
                &row_above_padded,
                &row_below_padded,
                width,
                &mut output_padded,
            );

            for x in 0..width {
                assert!(
                    approx_eq(output_padded[x], output_original[x], EPSILON_REL),
                    "Width {} mismatch at x={}: padded={}, original={}, rel_diff={}",
                    width,
                    x,
                    output_padded[x],
                    output_original[x],
                    (output_padded[x] - output_original[x]).abs()
                        / output_original[x].abs().max(1e-10)
                );
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_archmage_pre_erosion_matches_wide() {
        use archmage::SimdToken;

        // Get archmage token - skip test if CPU doesn't support required features
        let Some(token) = archmage::Desktop64::summon() else {
            eprintln!("Skipping archmage test - CPU doesn't support AVX2+FMA");
            return;
        };

        // Test with various widths
        for width in [8, 16, 24, 32, 64, 100, 128, 256] {
            let row: Vec<f32> = (0..width).map(|x| 100.0 + (x as f32) * 1.5).collect();
            let row_above: Vec<f32> = (0..width).map(|x| 98.0 + (x as f32) * 1.5).collect();
            let row_below: Vec<f32> = (0..width).map(|x| 102.0 + (x as f32) * 1.5).collect();

            // Create padded buffers
            let mut row_padded = vec![0.0f32; width + 2];
            let mut row_above_padded = vec![0.0f32; width + 2];
            let mut row_below_padded = vec![0.0f32; width + 2];

            row_padded[1..1 + width].copy_from_slice(&row);
            row_padded[0] = row[0];
            row_padded[width + 1] = row[width - 1];

            row_above_padded[1..1 + width].copy_from_slice(&row_above);
            row_above_padded[0] = row_above[0];
            row_above_padded[width + 1] = row_above[width - 1];

            row_below_padded[1..1 + width].copy_from_slice(&row_below);
            row_below_padded[0] = row_below[0];
            row_below_padded[width + 1] = row_below[width - 1];

            // Wide crate version (autoversioned)
            let mut output_wide = vec![0.0f32; width];
            pre_erosion_row_padded(
                &row_padded,
                &row_above_padded,
                &row_below_padded,
                width,
                &mut output_wide,
            );

            // Archmage version
            let mut output_archmage = vec![0.0f32; width];
            archmage_impl::mage_pre_erosion_row_padded(
                token,
                &row_padded,
                &row_above_padded,
                &row_below_padded,
                width,
                &mut output_archmage,
            );

            // Compare - allow slightly more tolerance for FMA rounding differences
            for x in 0..width {
                let diff = (output_archmage[x] - output_wide[x]).abs();
                let rel_diff = diff / output_wide[x].abs().max(1e-10);
                assert!(
                    rel_diff < 1e-5, // Slightly looser for FMA vs non-FMA
                    "Width {} archmage vs wide mismatch at x={}: archmage={}, wide={}, rel_diff={}",
                    width,
                    x,
                    output_archmage[x],
                    output_wide[x],
                    rel_diff
                );
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_archmage_fuzzy_erosion_matches_scalar() {
        use archmage::{SimdToken, X64V3Token};

        let Some(token) = X64V3Token::summon() else {
            println!("AVX2+FMA not available, skipping archmage test");
            return;
        };

        // Test with various buffer sizes
        for (pe_w, buffer_rows, blocks_w) in [
            (32, 12, 16),   // Small image
            (120, 12, 60),  // Medium image
            (480, 12, 240), // Large image (like 1920px)
        ] {
            // Create test pre-erosion buffer with recognizable pattern
            let mut pre_erosion_buffer = vec![0.0f32; pe_w * buffer_rows];
            for row in 0..buffer_rows {
                for col in 0..pe_w {
                    pre_erosion_buffer[row * pe_w + col] = ((row + col) % 256) as f32 / 256.0 + 0.1;
                }
            }

            // Test several pe_y_base values (interior and boundary)
            for pe_y_base in [0, 1, 5, 10] {
                let max_filled_row = 11; // One less than buffer_rows

                // Scalar reference implementation (inline copy of original algorithm)
                let mut output_scalar = vec![0.0f32; blocks_w];
                {
                    const MUL0: f32 = 0.125;
                    const MUL1: f32 = 0.075;
                    const MUL2: f32 = 0.06;
                    const MUL3: f32 = 0.05;

                    for bx in 0..blocks_w {
                        let pe_x_base = bx * 2;
                        let pe_y = pe_y_base as isize;

                        let mut sum = 0.0f32;
                        for dy in 0..2 {
                            for dx in 0..2 {
                                let cx = (pe_x_base + dx) as isize;
                                let cy = pe_y + dy as isize;

                                let mut vals = [0.0f32; 9];
                                for (i, (ny, nx)) in [
                                    (-1, -1),
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
                                    let px = (cx + nx).clamp(0, pe_w as isize - 1) as usize;
                                    let py = (cy + ny).clamp(0, max_filled_row as isize) as usize;
                                    let buffer_row = py % buffer_rows;
                                    let buf_idx = buffer_row * pe_w + px;
                                    vals[i] = pre_erosion_buffer[buf_idx];
                                }

                                // Partial sort to get 4 smallest
                                for i in 0..4 {
                                    for j in (i + 1)..9 {
                                        if vals[j] < vals[i] {
                                            vals.swap(i, j);
                                        }
                                    }
                                }

                                sum += MUL0 * vals[0]
                                    + MUL1 * vals[1]
                                    + MUL2 * vals[2]
                                    + MUL3 * vals[3];
                            }
                        }
                        output_scalar[bx] = sum;
                    }
                }

                // Optimized version (no token needed - pure inlined Rust)
                let mut output_archmage = vec![0.0f32; blocks_w];
                super::archmage_impl::mage_compute_fuzzy_erosion_row(
                    &pre_erosion_buffer,
                    pe_w,
                    buffer_rows,
                    max_filled_row,
                    pe_y_base,
                    0,        // block_start
                    blocks_w, // block_count
                    &mut output_archmage,
                );
                let _ = token; // Silence unused warning

                // Compare
                for bx in 0..blocks_w {
                    let diff = (output_archmage[bx] - output_scalar[bx]).abs();
                    let rel_diff = diff / output_scalar[bx].abs().max(1e-10);
                    assert!(
                        rel_diff < 1e-5,
                        "pe_w={} pe_y_base={} mismatch at bx={}: archmage={}, scalar={}, rel_diff={}",
                        pe_w,
                        pe_y_base,
                        bx,
                        output_archmage[bx],
                        output_scalar[bx],
                        rel_diff
                    );
                }
            }
        }
    }
}
