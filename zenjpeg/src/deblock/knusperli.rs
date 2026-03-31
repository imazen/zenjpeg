//! Knusperli DCT-domain boundary correction.
//!
//! For each pair of adjacent 8x8 blocks, analytically computes the boundary
//! discontinuity in the DCT domain, then applies a linear gradient correction
//! distributed across low-frequency coefficients. The correction is accumulated
//! in a separate buffer and applied once, preventing cascading artifacts.
//!
//! Reference: google/knusperli `output_image.cc:CopyFromJpegComponent()`
//!
//! All arithmetic uses f32 (the original uses 10-bit fixed-point integers).
//!
//! # Optimization: Strip-Based Processing
//!
//! Processes two block-rows at a time instead of full-image buffers.
//! Working set: 2 rows × blocks_wide × 64 × 4 bytes × 2 buffers.
//! For 4K (480 blocks wide): ~480KB — fits in L2 cache.
//! vs. full-image approach: ~63MB for 4K — blows L3.

use crate::decode::idct::inverse_dct_8x8;
use crate::foundation::consts::JPEG_ZIGZAG_ORDER;

use archmage::prelude::*;
use magetypes::simd::backends::F32x8Backend;
use magetypes::simd::generic::f32x8 as GenericF32x8;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Alpha × √2: α(0)×√2 = 1.0, α(k>0)×√2 = √2.
const ALPHA_SQRT2: [f32; 8] = {
    let s = core::f32::consts::SQRT_2;
    [1.0, s, s, s, s, s, s, s]
};

/// Alternating signs: [+1, -1, +1, -1, ...].
const SIGN_ALT: [f32; 8] = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];

/// Index-squared for HF penalty: [0², 1², 2², ..., 7²].
const IDX_SQ: [f32; 8] = [0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0];

/// HF penalty threshold.
const HF_THRESHOLD: f32 = 400.0;

/// Scale factor for accumulated offsets: 1/(2√2).
const OFFSET_SCALE: f32 = 1.0 / (2.0 * core::f32::consts::SQRT_2);

/// Precomputed gradient corrections for the left/top block in a pair.
/// Only 4 non-zero entries — the linear ramp DCT has zero energy above k=3.
///
/// Original C++ 10-bit FP: [318, -285, 81, -32, 0, 0, 0, 0] / 1024
const GRAD_LEFT: [f32; 4] = [
    318.0 / 1024.0,
    -285.0 / 1024.0,
    81.0 / 1024.0,
    -32.0 / 1024.0,
];

/// Gradient for the right/bottom block. Opposite sign convention from left.
///
/// Derived: `GRAD_RIGHT[k] = GRAD_LEFT[k] * -SIGN_ALT[k]`
const GRAD_RIGHT: [f32; 4] = [
    -318.0 / 1024.0,
    -285.0 / 1024.0,
    -81.0 / 1024.0,
    -32.0 / 1024.0,
];

/// HF-attenuated left gradient: `GRAD_LEFT[k] × 0.5^(k+1)`.
/// Applied when HF energy exceeds threshold — cascading halving per frequency.
const GRAD_LEFT_HF: [f32; 4] = [
    318.0 / 1024.0 * 0.5,
    -285.0 / 1024.0 * 0.25,
    81.0 / 1024.0 * 0.125,
    -32.0 / 1024.0 * 0.0625,
];

/// HF-attenuated right gradient: `GRAD_RIGHT[k] × 0.5^(k+1)`.
const GRAD_RIGHT_HF: [f32; 4] = [
    -318.0 / 1024.0 * 0.5,
    -285.0 / 1024.0 * 0.25,
    -81.0 / 1024.0 * 0.125,
    -32.0 / 1024.0 * 0.0625,
];

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Process one component's coefficients with Knusperli boundary correction.
///
/// Takes zigzag-order i16 coefficients (as produced by `decode_coefficients()`),
/// applies DCT-domain boundary corrections between adjacent blocks, then IDCTs
/// all blocks to produce a pixel-domain f32 plane.
///
/// # Arguments
/// * `zigzag_coeffs` — Flat coefficient buffer, `num_blocks * 64` elements, zigzag order
/// * `blocks_wide` — Number of horizontal 8x8 blocks
/// * `blocks_high` — Number of vertical 8x8 blocks
/// * `quant_table` — 64-element quantization table in natural (raster) order
///
/// # Returns
/// Pixel plane as f32 values (level-shifted: 0-255 range), dimensions `(blocks_wide*8) × (blocks_high*8)`.
pub fn process_component(
    zigzag_coeffs: &[i16],
    blocks_wide: usize,
    blocks_high: usize,
    quant_table: &[u16; 64],
) -> alloc::vec::Vec<f32> {
    debug_assert_eq!(zigzag_coeffs.len(), blocks_wide * blocks_high * 64);

    if blocks_wide == 0 || blocks_high == 0 {
        return alloc::vec::Vec::new();
    }

    let row_len = blocks_wide * 64;
    let pw = blocks_wide * 8;
    let ph = blocks_high * 8;

    // Precompute f32 quant table (avoids u16→f32 per block in finalize)
    let quant_f32: [f32; 64] = core::array::from_fn(|i| quant_table[i] as f32);

    // Working buffers: 2 block-rows of (coefficients + offsets).
    // For 4K (bw=480): 2 × 480 × 256 bytes × 2 = 480KB — fits in L2.
    let mut prev_blocks = alloc::vec![0.0f32; row_len];
    let mut prev_offsets = alloc::vec![0.0f32; row_len];
    let mut curr_blocks = alloc::vec![0.0f32; row_len];
    let mut curr_offsets = alloc::vec![0.0f32; row_len];

    let mut plane = alloc::vec![0.0f32; pw * ph];

    // Row 0: dequantize + H corrections (no row above)
    dequantize_row(zigzag_coeffs, 0, blocks_wide, quant_table, &mut prev_blocks);
    // prev_offsets already zeroed by alloc
    correct_h_row(&prev_blocks, &mut prev_offsets, blocks_wide);

    for by in 1..blocks_high {
        dequantize_row(
            zigzag_coeffs,
            by,
            blocks_wide,
            quant_table,
            &mut curr_blocks,
        );
        curr_offsets.fill(0.0);
        correct_h_row(&curr_blocks, &mut curr_offsets, blocks_wide);
        correct_v_between(
            &prev_blocks,
            &mut prev_offsets,
            &curr_blocks,
            &mut curr_offsets,
            blocks_wide,
        );

        // Row by-1 has all corrections (H + V above + V below) — finalize
        finalize_row(
            &prev_blocks,
            &prev_offsets,
            &quant_f32,
            by - 1,
            blocks_wide,
            pw,
            &mut plane,
        );

        core::mem::swap(&mut prev_blocks, &mut curr_blocks);
        core::mem::swap(&mut prev_offsets, &mut curr_offsets);
    }

    // Last row has H + V from above (no row below) — finalize
    finalize_row(
        &prev_blocks,
        &prev_offsets,
        &quant_f32,
        blocks_high - 1,
        blocks_wide,
        pw,
        &mut plane,
    );

    plane
}

// ---------------------------------------------------------------------------
// Dequantization
// ---------------------------------------------------------------------------

/// Dequantize one block-row from zigzag i16 to natural-order f32.
#[inline]
fn dequantize_row(
    zigzag_coeffs: &[i16],
    by: usize,
    blocks_wide: usize,
    quant_table: &[u16; 64],
    row_blocks: &mut [f32],
) {
    let row_start = by * blocks_wide * 64;
    for bx in 0..blocks_wide {
        let src_off = row_start + bx * 64;
        let dst_off = bx * 64;
        for nat in 0..64 {
            let zi = JPEG_ZIGZAG_ORDER[nat] as usize;
            row_blocks[dst_off + nat] =
                zigzag_coeffs[src_off + zi] as f32 * quant_table[nat] as f32;
        }
    }
}

// ---------------------------------------------------------------------------
// Horizontal boundary correction (within one block-row)
// ---------------------------------------------------------------------------

/// Correct vertical boundaries between horizontally adjacent blocks in one row.
#[inline(never)]
fn correct_h_row(blocks: &[f32], offsets: &mut [f32], blocks_wide: usize) {
    incant!(correct_h_row_impl(blocks, offsets, blocks_wide));
}

#[magetypes(v3, scalar)]
#[inline(always)]
fn correct_h_row_impl(token: Token, blocks: &[f32], offsets: &mut [f32], blocks_wide: usize) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;

    let alpha_v = f32x8::from_array(token, ALPHA_SQRT2);
    let sign_v = f32x8::from_array(token, SIGN_ALT);
    let idx_sq_v = f32x8::from_array(token, IDX_SQ);

    for bx in 0..blocks_wide.saturating_sub(1) {
        let bi = bx * 64;
        let bj = (bx + 1) * 64;

        for v in 0..4 {
            let row = v * 8;

            let gi = f32x8::load(token, blocks[bi + row..bi + row + 8].try_into().unwrap());
            let gj = f32x8::load(token, blocks[bj + row..bj + row + 8].try_into().unwrap());

            let (delta, hf) = compute_delta_hf(gi, gj, alpha_v, sign_v, idx_sq_v);

            let (gl, gr) = select_gradient(hf);
            for k in 0..4 {
                offsets[bi + row + k] += delta * gl[k];
                offsets[bj + row + k] += delta * gr[k];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Vertical boundary correction (between two adjacent block-rows)
// ---------------------------------------------------------------------------

/// Correct horizontal boundaries between top and bottom block-rows.
#[inline(never)]
fn correct_v_between(
    top: &[f32],
    top_off: &mut [f32],
    bot: &[f32],
    bot_off: &mut [f32],
    blocks_wide: usize,
) {
    incant!(correct_v_between_impl(
        top,
        top_off,
        bot,
        bot_off,
        blocks_wide
    ));
}

#[magetypes(v3, scalar)]
#[inline(always)]
fn correct_v_between_impl(
    token: Token,
    top: &[f32],
    top_off: &mut [f32],
    bot: &[f32],
    bot_off: &mut [f32],
    blocks_wide: usize,
) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;

    let alpha_v = f32x8::from_array(token, ALPHA_SQRT2);
    let sign_v = f32x8::from_array(token, SIGN_ALT);
    let idx_sq_v = f32x8::from_array(token, IDX_SQ);

    for bx in 0..blocks_wide {
        let off = bx * 64;

        for u in 0..4 {
            let mut gi_arr = [0.0f32; 8];
            let mut gj_arr = [0.0f32; 8];
            for v in 0..8 {
                gi_arr[v] = top[off + v * 8 + u];
                gj_arr[v] = bot[off + v * 8 + u];
            }

            let gi = f32x8::from_array(token, gi_arr);
            let gj = f32x8::from_array(token, gj_arr);

            let (delta, hf) = compute_delta_hf(gi, gj, alpha_v, sign_v, idx_sq_v);

            let (gl, gr) = select_gradient(hf);
            for v in 0..4 {
                top_off[off + v * 8 + u] += delta * gl[v];
                bot_off[off + v * 8 + u] += delta * gr[v];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Compute boundary discontinuity (delta) and HF energy penalty, generic over backend.
#[inline(always)]
fn compute_delta_hf<T: F32x8Backend>(
    gi: GenericF32x8<T>,
    gj: GenericF32x8<T>,
    alpha: GenericF32x8<T>,
    sign: GenericF32x8<T>,
    idx_sq: GenericF32x8<T>,
) -> (f32, f32) {
    // delta = Σ α(k)√2 × (gj[k] - (-1)^k × gi[k])
    let delta_lanes = alpha * (gj - sign * gi);
    let delta = delta_lanes.reduce_add();

    // hf = Σ k² × (gi[k]² + gj[k]²)
    let hf_lanes = idx_sq * (gi * gi + gj * gj);
    let hf = hf_lanes.reduce_add();

    (delta, hf)
}

/// Select normal or HF-attenuated gradient based on penalty energy.
#[inline(always)]
fn select_gradient(hf: f32) -> (&'static [f32; 4], &'static [f32; 4]) {
    if hf > HF_THRESHOLD {
        (&GRAD_LEFT_HF, &GRAD_RIGHT_HF)
    } else {
        (&GRAD_LEFT, &GRAD_RIGHT)
    }
}

// ---------------------------------------------------------------------------
// Finalization: offset application + IDCT + plane write
// ---------------------------------------------------------------------------

/// Apply scaled offsets, clamp to quant intervals, IDCT, and write one
/// block-row to the output pixel plane.
#[inline(never)]
fn finalize_row(
    blocks: &[f32],
    offsets: &[f32],
    quant_f32: &[f32; 64],
    by: usize,
    blocks_wide: usize,
    pw: usize,
    plane: &mut [f32],
) {
    incant!(finalize_row_impl(
        blocks,
        offsets,
        quant_f32,
        by,
        blocks_wide,
        pw,
        plane
    ));
}

#[magetypes(v3, scalar)]
#[inline(always)]
fn finalize_row_impl(
    token: Token,
    blocks: &[f32],
    offsets: &[f32],
    quant_f32: &[f32; 64],
    by: usize,
    blocks_wide: usize,
    pw: usize,
    plane: &mut [f32],
) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;

    let scale_v = f32x8::splat(token, OFFSET_SCALE);
    let half = f32x8::splat(token, 0.5);
    let level_shift = f32x8::splat(token, 128.0);

    let mut block = [0.0f32; 64];

    for bx in 0..blocks_wide {
        let off = bx * 64;

        for k in (0..64).step_by(8) {
            let mid = f32x8::load(token, blocks[off + k..off + k + 8].try_into().unwrap());
            let correction = f32x8::load(token, offsets[off + k..off + k + 8].try_into().unwrap());
            let q = f32x8::load(token, quant_f32[k..k + 8].try_into().unwrap());
            let half_q = q * half;

            let corrected = mid + correction * scale_v;
            let clamped = corrected.max(mid - half_q).min(mid + half_q);

            clamped.store(<&mut [f32; 8]>::try_from(&mut block[k..k + 8]).unwrap());
        }

        let pixels = inverse_dct_8x8(&block);

        for row in 0..8 {
            let src = f32x8::load(token, pixels[row * 8..(row + 1) * 8].try_into().unwrap());
            let shifted = src + level_shift;
            let dst = (by * 8 + row) * pw + bx * 8;
            shifted.store(<&mut [f32; 8]>::try_from(&mut plane[dst..dst + 8]).unwrap());
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_blocks_no_correction() {
        let blocks_wide = 2;
        let blocks_high = 1;
        let num_blocks = 2;

        let mut zigzag = vec![0i16; num_blocks * 64];
        zigzag[0] = 10;
        zigzag[64] = 10;

        let mut quant = [1u16; 64];
        quant[0] = 8;

        let plane = process_component(&zigzag, blocks_wide, blocks_high, &quant);

        assert_eq!(plane.len(), 16 * 8);
        let p0 = plane[0];
        let p1 = plane[8];
        assert!(
            (p0 - p1).abs() < 0.01,
            "Uniform blocks should produce identical pixels: {p0} vs {p1}"
        );
    }

    #[test]
    fn test_discontinuity_reduced() {
        let blocks_wide = 2;
        let blocks_high = 1;

        let mut zigzag = vec![0i16; 2 * 64];
        zigzag[0] = 5;
        zigzag[64] = 20;

        let quant = [8u16; 64];

        let plane = process_component(&zigzag, blocks_wide, blocks_high, &quant);

        let left_edge = plane[7];
        let right_edge = plane[8];

        let gap = (right_edge - left_edge).abs();
        assert!(
            gap < 15.0,
            "Boundary gap should be reduced from 15.0, got {gap}"
        );
    }

    #[test]
    fn test_vertical_boundary() {
        // 1 block wide, 2 blocks high — tests V correction path
        let blocks_wide = 1;
        let blocks_high = 2;

        let mut zigzag = vec![0i16; 2 * 64];
        zigzag[0] = 5; // top block DC
        zigzag[64] = 20; // bottom block DC

        let quant = [8u16; 64];

        let plane = process_component(&zigzag, blocks_wide, blocks_high, &quant);

        // Bottom edge of top block (row 7) and top edge of bottom block (row 8)
        let pw = 8;
        let top_edge = plane[7 * pw]; // row 7, col 0
        let bot_edge = plane[8 * pw]; // row 8, col 0

        let gap = (bot_edge - top_edge).abs();
        assert!(
            gap < 15.0,
            "Vertical boundary gap should be reduced from 15.0, got {gap}"
        );
    }

    #[test]
    fn test_gradient_precomputation() {
        // Verify GRAD_RIGHT = GRAD_LEFT * -SIGN_ALT for k=0..3
        for k in 0..4 {
            let expected = GRAD_LEFT[k] * -SIGN_ALT[k];
            assert!(
                (GRAD_RIGHT[k] - expected).abs() < 1e-7,
                "GRAD_RIGHT[{k}] = {}, expected {expected}",
                GRAD_RIGHT[k]
            );
        }

        // Verify HF gradients = normal × 0.5^(k+1)
        for k in 0..4 {
            let scale = 0.5f32.powi(k as i32 + 1);
            assert!(
                (GRAD_LEFT_HF[k] - GRAD_LEFT[k] * scale).abs() < 1e-7,
                "GRAD_LEFT_HF[{k}] mismatch"
            );
            assert!(
                (GRAD_RIGHT_HF[k] - GRAD_RIGHT[k] * scale).abs() < 1e-7,
                "GRAD_RIGHT_HF[{k}] mismatch"
            );
        }
    }
}
