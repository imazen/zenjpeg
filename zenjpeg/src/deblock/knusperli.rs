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

use crate::decode::idct::inverse_dct_8x8;
use crate::foundation::consts::JPEG_ZIGZAG_ORDER;

use wide::f32x8;

/// DCT representation of a linear ramp from 0 to 1 across 8 pixels.
/// Only the first 4 coefficients are non-zero — high-frequency corrections
/// would introduce ringing rather than smooth the boundary.
///
/// Original C++ uses 10-bit FP: [318, -285, 81, -32, 0, 0, 0, 0]
const LINEAR_GRADIENT: [f32; 8] = [
    318.0 / 1024.0,  // 0.3105
    -285.0 / 1024.0, // -0.2783
    81.0 / 1024.0,   // 0.0791
    -32.0 / 1024.0,  // -0.0313
    0.0,
    0.0,
    0.0,
    0.0,
];

/// Alpha coefficients × √2: α(0)×√2 = 1.0, α(k>0)×√2 = √2.
const ALPHA_SQRT2: [f32; 8] = {
    let s = core::f32::consts::SQRT_2;
    [1.0, s, s, s, s, s, s, s]
};

/// Alternating signs for boundary evaluation: [+1, -1, +1, -1, ...].
/// Right edge of block = Σ α(u) × (-1)^u × coeff[v,u].
const SIGN_ALT: [f32; 8] = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];

/// HF penalty threshold. When energy in high frequencies exceeds this,
/// the correction is halved per frequency to avoid ringing.
const HF_PENALTY_THRESHOLD: f32 = 400.0;

/// Scale factor for applying accumulated offsets: 1/(2√2).
const OFFSET_SCALE: f32 = 1.0 / (2.0 * core::f32::consts::SQRT_2);

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
    let num_blocks = blocks_wide * blocks_high;
    debug_assert_eq!(zigzag_coeffs.len(), num_blocks * 64);

    // Dequantize all blocks to natural order.
    // We store the dequantized mid-point values (coeff × quant_step)
    // and accumulate corrections in a separate offset buffer.
    let mut blocks = alloc::vec![0.0f32; num_blocks * 64];
    let mut offsets = alloc::vec![0.0f32; num_blocks * 64];

    dequantize_all(zigzag_coeffs, num_blocks, quant_table, &mut blocks);

    // Boundary correction passes
    correct_horizontal_boundaries(&blocks, &mut offsets, blocks_wide, blocks_high);
    correct_vertical_boundaries(&blocks, &mut offsets, blocks_wide, blocks_high);

    // Apply offsets with clamping to quantization intervals, then IDCT
    apply_offsets_and_idct(&mut blocks, &offsets, quant_table, blocks_wide, blocks_high)
}

/// Dequantize all blocks from zigzag i16 to natural-order f32.
fn dequantize_all(
    zigzag_coeffs: &[i16],
    num_blocks: usize,
    quant_table: &[u16; 64],
    blocks: &mut [f32],
) {
    for bi in 0..num_blocks {
        let src = &zigzag_coeffs[bi * 64..(bi + 1) * 64];
        let dst = &mut blocks[bi * 64..(bi + 1) * 64];

        for nat in 0..64 {
            let zi = JPEG_ZIGZAG_ORDER[nat] as usize;
            dst[nat] = src[zi] as f32 * quant_table[nat] as f32;
        }
    }
}

/// Horizontal pass: correct vertical boundaries between left/right adjacent blocks.
///
/// For each pair of horizontally adjacent blocks (i, j), compute the boundary
/// discontinuity for rows v=0..3 (low frequencies only), then distribute
/// correction using the linear gradient basis.
fn correct_horizontal_boundaries(
    blocks: &[f32],
    offsets: &mut [f32],
    blocks_wide: usize,
    blocks_high: usize,
) {
    for by in 0..blocks_high {
        for bx in 0..blocks_wide.saturating_sub(1) {
            let bi = by * blocks_wide + bx;
            let bj = bi + 1;
            let bi_off = bi * 64;
            let bj_off = bj * 64;

            for v in 0..4 {
                let row_base = v * 8;

                // Compute boundary discontinuity and HF energy using SIMD.
                // delta_v = Σ_u α(u)√2 × (coeff_j[v,u] - (-1)^u × coeff_i[v,u])
                // hf_penalty = Σ_u u² × (coeff_i² + coeff_j²)
                let gi = f32x8::new(
                    blocks[bi_off + row_base..bi_off + row_base + 8]
                        .try_into()
                        .unwrap(),
                );
                let gj = f32x8::new(
                    blocks[bj_off + row_base..bj_off + row_base + 8]
                        .try_into()
                        .unwrap(),
                );

                let alpha = f32x8::new(ALPHA_SQRT2);
                let sign = f32x8::new(SIGN_ALT);
                let u_sq = f32x8::new([0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0]);

                // delta components per lane
                let delta_lanes = alpha * (gj - sign * gi);
                // Sum all 8 lanes
                let delta_v_initial: f32 = sum_f32x8(delta_lanes);

                // HF penalty
                let hf_lanes = u_sq * (gi * gi + gj * gj);
                let hf_penalty: f32 = sum_f32x8(hf_lanes);

                // Distribute correction with cascading HF penalty halving.
                // For u=0..7, if hf_penalty > 400 then delta_v *= 0.5 before use.
                // The halving accumulates: by u=7, delta_v may have been halved up to 8 times.
                let mut delta_v = delta_v_initial;

                for u in 0..8 {
                    if hf_penalty > HF_PENALTY_THRESHOLD {
                        delta_v *= 0.5;
                    }
                    let corr_sign = SIGN_ALT[u]; // even: +1 for left, odd: -1
                    let correction = delta_v * LINEAR_GRADIENT[u];
                    offsets[bi_off + row_base + u] += correction;
                    // Right block uses opposite sign for even u, same for odd
                    // (C++: u&1 ? 1 : -1 applied to correction)
                    offsets[bj_off + row_base + u] += correction * -corr_sign;
                }
            }
        }
    }
}

/// Vertical pass: correct horizontal boundaries between top/bottom adjacent blocks.
///
/// Same logic as horizontal but transposed: iterate u=0..3, sum over v=0..7.
fn correct_vertical_boundaries(
    blocks: &[f32],
    offsets: &mut [f32],
    blocks_wide: usize,
    blocks_high: usize,
) {
    for by in 0..blocks_high.saturating_sub(1) {
        for bx in 0..blocks_wide {
            let bi = by * blocks_wide + bx;
            let bj = (by + 1) * blocks_wide + bx;
            let bi_off = bi * 64;
            let bj_off = bj * 64;

            for u in 0..4 {
                // For vertical boundaries, we need column u from each block:
                // positions [0*8+u, 1*8+u, 2*8+u, ..., 7*8+u]
                let mut gi_arr = [0.0f32; 8];
                let mut gj_arr = [0.0f32; 8];

                for v in 0..8 {
                    gi_arr[v] = blocks[bi_off + v * 8 + u];
                    gj_arr[v] = blocks[bj_off + v * 8 + u];
                }

                let gi = f32x8::new(gi_arr);
                let gj = f32x8::new(gj_arr);
                let alpha = f32x8::new(ALPHA_SQRT2);
                let sign = f32x8::new(SIGN_ALT);
                let v_sq = f32x8::new([0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0]);

                let delta_lanes = alpha * (gj - sign * gi);
                let delta_u_initial: f32 = sum_f32x8(delta_lanes);

                let hf_lanes = v_sq * (gi * gi + gj * gj);
                let hf_penalty: f32 = sum_f32x8(hf_lanes);

                let mut delta_u = delta_u_initial;

                for v in 0..8 {
                    if hf_penalty > HF_PENALTY_THRESHOLD {
                        delta_u *= 0.5;
                    }
                    let corr_sign = SIGN_ALT[v];
                    let correction = delta_u * LINEAR_GRADIENT[v];
                    offsets[bi_off + v * 8 + u] += correction;
                    offsets[bj_off + v * 8 + u] += correction * -corr_sign;
                }
            }
        }
    }
}

/// Apply accumulated offsets, clamp to quantization intervals, and IDCT all blocks.
///
/// Returns the pixel plane (level-shifted to 0-255 range).
fn apply_offsets_and_idct(
    blocks: &mut [f32],
    offsets: &[f32],
    quant_table: &[u16; 64],
    blocks_wide: usize,
    blocks_high: usize,
) -> alloc::vec::Vec<f32> {
    let num_blocks = blocks_wide * blocks_high;
    let pw = blocks_wide * 8;
    let ph = blocks_high * 8;
    let mut plane = alloc::vec![0.0f32; pw * ph];

    let scale = f32x8::splat(OFFSET_SCALE);

    for bi in 0..num_blocks {
        let block_off = bi * 64;

        // Apply offsets scaled by 1/(2√2), clamp to quant intervals.
        // Process 8 coefficients at a time.
        for k_base in (0..64).step_by(8) {
            let mid = f32x8::new(
                blocks[block_off + k_base..block_off + k_base + 8]
                    .try_into()
                    .unwrap(),
            );
            let off = f32x8::new(
                offsets[block_off + k_base..block_off + k_base + 8]
                    .try_into()
                    .unwrap(),
            );

            // Recompute min/max from original coefficients (avoids storing 2 extra arrays)
            let mut q_arr = [0.0f32; 8];
            for i in 0..8 {
                q_arr[i] = quant_table[k_base + i] as f32;
            }
            let q = f32x8::new(q_arr);
            let half_q = q * f32x8::splat(0.5);

            let corrected = mid + off * scale;
            let clamped = corrected.max(mid - half_q).min(mid + half_q);

            let result: [f32; 8] = clamped.into();
            blocks[block_off + k_base..block_off + k_base + 8].copy_from_slice(&result);
        }

        // IDCT this block
        let block_data: [f32; 64] = blocks[block_off..block_off + 64].try_into().unwrap();
        let pixels = inverse_dct_8x8(&block_data);

        // Write to output plane with +128 level shift
        let bx = bi % blocks_wide;
        let by = bi / blocks_wide;
        for row in 0..8 {
            let dst_off = (by * 8 + row) * pw + bx * 8;
            for col in 0..8 {
                plane[dst_off + col] = pixels[row * 8 + col] + 128.0;
            }
        }
    }

    plane
}

/// Sum all 8 lanes of an f32x8. Uses pairwise reduction for accuracy.
#[inline(always)]
fn sum_f32x8(v: f32x8) -> f32 {
    let a: [f32; 8] = v.into();
    // Pairwise: (a[0]+a[1]) + (a[2]+a[3]) + (a[4]+a[5]) + (a[6]+a[7])
    let s01 = a[0] + a[1];
    let s23 = a[2] + a[3];
    let s45 = a[4] + a[5];
    let s67 = a[6] + a[7];
    (s01 + s23) + (s45 + s67)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_blocks_no_correction() {
        // Two adjacent blocks with identical DC, all AC zero → no boundary discontinuity
        let blocks_wide = 2;
        let blocks_high = 1;
        let num_blocks = 2;

        let mut zigzag = vec![0i16; num_blocks * 64];
        // Set DC coefficient (zigzag position 0) to 10 for both blocks
        zigzag[0] = 10;
        zigzag[64] = 10;

        let mut quant = [1u16; 64];
        quant[0] = 8; // DC quant

        let plane = process_component(&zigzag, blocks_wide, blocks_high, &quant);

        // Both blocks should produce identical DC values (10 * 8 / 8 + 128 = 138)
        assert_eq!(plane.len(), 16 * 8);
        let p0 = plane[0]; // first pixel
        let p1 = plane[8]; // first pixel of second block
        assert!(
            (p0 - p1).abs() < 0.01,
            "Uniform blocks should produce identical pixels: {p0} vs {p1}"
        );
    }

    #[test]
    fn test_discontinuity_reduced() {
        // Two adjacent blocks with different DC values → boundary should be smoothed
        let blocks_wide = 2;
        let blocks_high = 1;

        let mut zigzag = vec![0i16; 2 * 64];
        zigzag[0] = 5; // left block DC = 5
        zigzag[64] = 20; // right block DC = 20

        let quant = [8u16; 64];

        let plane = process_component(&zigzag, blocks_wide, blocks_high, &quant);

        // Right edge of left block (col 7) and left edge of right block (col 8)
        // should be closer together than the uncorrected values
        let left_edge = plane[7]; // col 7
        let right_edge = plane[8]; // col 8

        // Uncorrected: left = 5*8/8+128=133, right = 20*8/8+128=148, gap=15
        // With correction the gap should be smaller
        let gap = (right_edge - left_edge).abs();
        assert!(
            gap < 15.0,
            "Boundary gap should be reduced from 15.0, got {gap}"
        );
    }

    #[test]
    fn test_gradient_constants() {
        // LINEAR_GRADIENT should sum to approximately 0 (it's a zero-mean ramp)
        let sum: f32 = LINEAR_GRADIENT.iter().sum();
        // Actually it sums to the DC component of the ramp, which is 318/1024 ≈ 0.31
        assert!((sum - (318.0 - 285.0 + 81.0 - 32.0) / 1024.0).abs() < 0.001);
    }
}
