//! WASM SIMD128 encode kernels (wasm32 only).
//!
//! Uses `i32x4_dot_i16x8` (pmaddwd equivalent) for matrix multiply and
//! `u8x16_narrow_i16x8_u` for saturating narrow. Processes 8 pixels per
//! iteration. Matches the integer math of the AVX2/NEON kernels exactly
//! for dispatch parity.

#![cfg(target_arch = "wasm32")]

use crate::types::{ForwardCoeffs, PREC, pack_i16_pair};
use archmage::prelude::*;

/// 4:4:4 WASM SIMD128 encode kernel. Returns number of pixels processed
/// (multiple of 8).
#[arcane(import_intrinsics)]
pub(crate) fn rgb_to_yuv444_wasm(
    _token: archmage::Wasm128Token,
    rgb: &[u8],
    y_out: &mut [u8],
    cb_out: &mut [u8],
    cr_out: &mut [u8],
    n: usize,
    coeffs: &ForwardCoeffs,
) -> usize {
    let y_rg = i32x4_splat(pack_i16_pair(coeffs.yr, coeffs.yg));
    let y_b0 = i32x4_splat(pack_i16_pair(coeffs.yb, 0));
    let cb_rg = i32x4_splat(pack_i16_pair(coeffs.cb_r, coeffs.cb_g));
    let cb_b0 = i32x4_splat(pack_i16_pair(coeffs.cb_b, 0));
    let cr_rg = i32x4_splat(pack_i16_pair(coeffs.cr_r, coeffs.cr_g));
    let cr_b0 = i32x4_splat(pack_i16_pair(coeffs.cr_b, 0));

    let y_bias_v = i32x4_splat(coeffs.y_bias);
    let uv_bias_v = i32x4_splat(coeffs.uv_bias);

    // Process 8 pixels per iteration.
    let blocks = n / 8;
    for blk in 0..blocks {
        let base = blk * 8;

        // Deinterleave 8 RGB pixels into R/G/B i16 arrays.
        let mut ra = [0i16; 8];
        let mut ga = [0i16; 8];
        let mut ba = [0i16; 8];
        for i in 0..8 {
            let p = (base + i) * 3;
            ra[i] = rgb[p] as i16;
            ga[i] = rgb[p + 1] as i16;
            ba[i] = rgb[p + 2] as i16;
        }

        // Process 8 pixels as two halves of 4.
        let mut y_i16 = [0i16; 8];
        let mut cb_i16 = [0i16; 8];
        let mut cr_i16 = [0i16; 8];

        for half in 0..2 {
            let off = half * 4;

            // Interleave RG pairs: [R0, G0, R1, G1, R2, G2, R3, G3]
            let rg = i16x8(
                ra[off],
                ga[off],
                ra[off + 1],
                ga[off + 1],
                ra[off + 2],
                ga[off + 2],
                ra[off + 3],
                ga[off + 3],
            );
            // B+zero pairs: [B0, 0, B1, 0, B2, 0, B3, 0]
            let bz = i16x8(ba[off], 0, ba[off + 1], 0, ba[off + 2], 0, ba[off + 3], 0);

            // pmaddwd equivalent: i32x4_dot_i16x8(rg, y_rg) computes
            //   [r0*yr + g0*yg, r1*yr + g1*yg, r2*yr + g2*yg, r3*yr + g3*yg]
            let y_val = i32x4_add(
                i32x4_add(i32x4_dot_i16x8(rg, y_rg), i32x4_dot_i16x8(bz, y_b0)),
                y_bias_v,
            );
            let cb_val = i32x4_add(
                i32x4_add(i32x4_dot_i16x8(rg, cb_rg), i32x4_dot_i16x8(bz, cb_b0)),
                uv_bias_v,
            );
            let cr_val = i32x4_add(
                i32x4_add(i32x4_dot_i16x8(rg, cr_rg), i32x4_dot_i16x8(bz, cr_b0)),
                uv_bias_v,
            );

            // Arithmetic shift right by PREC (integer truncating divide by 2^PREC).
            let y_s = i32x4_shr(y_val, PREC as u32);
            let cb_s = i32x4_shr(cb_val, PREC as u32);
            let cr_s = i32x4_shr(cr_val, PREC as u32);

            // Extract 4 lanes per channel.
            y_i16[off] = i32x4_extract_lane::<0>(y_s).clamp(0, 255) as i16;
            y_i16[off + 1] = i32x4_extract_lane::<1>(y_s).clamp(0, 255) as i16;
            y_i16[off + 2] = i32x4_extract_lane::<2>(y_s).clamp(0, 255) as i16;
            y_i16[off + 3] = i32x4_extract_lane::<3>(y_s).clamp(0, 255) as i16;
            cb_i16[off] = i32x4_extract_lane::<0>(cb_s).clamp(0, 255) as i16;
            cb_i16[off + 1] = i32x4_extract_lane::<1>(cb_s).clamp(0, 255) as i16;
            cb_i16[off + 2] = i32x4_extract_lane::<2>(cb_s).clamp(0, 255) as i16;
            cb_i16[off + 3] = i32x4_extract_lane::<3>(cb_s).clamp(0, 255) as i16;
            cr_i16[off] = i32x4_extract_lane::<0>(cr_s).clamp(0, 255) as i16;
            cr_i16[off + 1] = i32x4_extract_lane::<1>(cr_s).clamp(0, 255) as i16;
            cr_i16[off + 2] = i32x4_extract_lane::<2>(cr_s).clamp(0, 255) as i16;
            cr_i16[off + 3] = i32x4_extract_lane::<3>(cr_s).clamp(0, 255) as i16;
        }

        for i in 0..8 {
            y_out[base + i] = y_i16[i] as u8;
            cb_out[base + i] = cb_i16[i] as u8;
            cr_out[base + i] = cr_i16[i] as u8;
        }
    }
    blocks * 8
}

/// 4:2:0 WASM SIMD128 fused kernel. Processes pairs of 2 rows × 8 pixels.
/// Replicates the exact AVX2 sequence (avg_epu8 vertical, pair-sum
/// horizontal, pmaddwd at PREC+1) for dispatch parity.
#[arcane(import_intrinsics)]
pub(crate) fn rgb_to_yuv420_wasm(
    _token: archmage::Wasm128Token,
    rgb: &[u8],
    y_out: &mut [u8],
    cb_out: &mut [u8],
    cr_out: &mut [u8],
    width: usize,
    height: usize,
    cw: usize,
    coeffs: &ForwardCoeffs,
) {
    // Y plane uses the same 4:4:4 math.
    let y_rg = i32x4_splat(pack_i16_pair(coeffs.yr, coeffs.yg));
    let y_b0 = i32x4_splat(pack_i16_pair(coeffs.yb, 0));
    let y_bias_v = i32x4_splat(coeffs.y_bias);

    // Chroma at PREC+1 with uv_bias_420.
    let cb_rg = i32x4_splat(pack_i16_pair(coeffs.cb_r, coeffs.cb_g));
    let cb_b0 = i32x4_splat(pack_i16_pair(coeffs.cb_b, 0));
    let cr_rg = i32x4_splat(pack_i16_pair(coeffs.cr_r, coeffs.cr_g));
    let cr_b0 = i32x4_splat(pack_i16_pair(coeffs.cr_b, 0));
    let uv_bias_420_v = i32x4_splat(coeffs.uv_bias_420);

    let row_stride = width * 3;
    let col_blocks = width / 8;
    let row_pairs = height / 2;

    for ry in 0..row_pairs {
        let top = ry * 2;
        let bot = top + 1;
        let top_off = top * row_stride;
        let bot_off = bot * row_stride;
        let y_top_off = top * width;
        let y_bot_off = bot * width;
        let cb_row_off = ry * cw;

        for cx in 0..col_blocks {
            let px = cx * 8;

            // --- Y plane for top and bottom rows (8 pixels each) ---
            for (y_dst_off, src_off) in [(y_top_off, top_off), (y_bot_off, bot_off)] {
                let mut ra = [0i16; 8];
                let mut ga = [0i16; 8];
                let mut ba = [0i16; 8];
                for i in 0..8 {
                    let p = src_off + (px + i) * 3;
                    ra[i] = rgb[p] as i16;
                    ga[i] = rgb[p + 1] as i16;
                    ba[i] = rgb[p + 2] as i16;
                }
                for half in 0..2 {
                    let off = half * 4;
                    let rg = i16x8(
                        ra[off],
                        ga[off],
                        ra[off + 1],
                        ga[off + 1],
                        ra[off + 2],
                        ga[off + 2],
                        ra[off + 3],
                        ga[off + 3],
                    );
                    let bz = i16x8(ba[off], 0, ba[off + 1], 0, ba[off + 2], 0, ba[off + 3], 0);
                    let y_val = i32x4_add(
                        i32x4_add(i32x4_dot_i16x8(rg, y_rg), i32x4_dot_i16x8(bz, y_b0)),
                        y_bias_v,
                    );
                    let y_s = i32x4_shr(y_val, PREC as u32);
                    for lane in 0..4 {
                        let v = match lane {
                            0 => i32x4_extract_lane::<0>(y_s),
                            1 => i32x4_extract_lane::<1>(y_s),
                            2 => i32x4_extract_lane::<2>(y_s),
                            _ => i32x4_extract_lane::<3>(y_s),
                        };
                        y_out[y_dst_off + px + off + lane] = v.clamp(0, 255) as u8;
                    }
                }
            }

            // --- Chroma: avg_epu8 vertical, pair-sum horizontal ---
            // Replicates AVX2 exactly via scalar arithmetic (same rounding).
            // This could be SIMDified with v128 avg + narrow, but the scalar
            // integer form is byte-identical and keeps dispatch parity trivial.
            for i in 0..4 {
                let col_a = px + i * 2;
                let col_b = col_a + 1;
                let i00 = top_off + col_a * 3;
                let i01 = top_off + col_b * 3;
                let i10 = bot_off + col_a * 3;
                let i11 = bot_off + col_b * 3;
                // avg_epu8 vertical: (a + b + 1) / 2
                let r_v0 = (rgb[i00] as i32 + rgb[i10] as i32 + 1) / 2;
                let r_v1 = (rgb[i01] as i32 + rgb[i11] as i32 + 1) / 2;
                let g_v0 = (rgb[i00 + 1] as i32 + rgb[i10 + 1] as i32 + 1) / 2;
                let g_v1 = (rgb[i01 + 1] as i32 + rgb[i11 + 1] as i32 + 1) / 2;
                let b_v0 = (rgb[i00 + 2] as i32 + rgb[i10 + 2] as i32 + 1) / 2;
                let b_v1 = (rgb[i01 + 2] as i32 + rgb[i11 + 2] as i32 + 1) / 2;
                // maddubs pair-sum
                let r_ps = r_v0 + r_v1;
                let g_ps = g_v0 + g_v1;
                let b_ps = b_v0 + b_v1;
                cb_out[cb_row_off + cx * 4 + i] = ((r_ps * coeffs.cb_r as i32
                    + g_ps * coeffs.cb_g as i32
                    + b_ps * coeffs.cb_b as i32
                    + coeffs.uv_bias_420)
                    >> (PREC + 1))
                    .clamp(0, 255) as u8;
                cr_out[cb_row_off + cx * 4 + i] = ((r_ps * coeffs.cr_r as i32
                    + g_ps * coeffs.cr_g as i32
                    + b_ps * coeffs.cr_b as i32
                    + coeffs.uv_bias_420)
                    >> (PREC + 1))
                    .clamp(0, 255) as u8;
            }
            // Silence unused-variable warnings when SIMD splats aren't used here
            // (the SIMD chroma path is not vectorized at this granularity).
            let _ = (cb_rg, cb_b0, cr_rg, cr_b0, uv_bias_420_v);
        }
    }

    // Scalar tail: remaining columns not covered by 8-wide SIMD blocks,
    // plus any odd last row.
    let simd_cols = col_blocks * 8;
    crate::encode::rgb_to_yuv420_scalar_tail(
        rgb, y_out, cb_out, cr_out, width, height, cw, simd_cols, coeffs,
    );
}
