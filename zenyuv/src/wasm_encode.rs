//! WASM SIMD128 encode kernels (wasm32 only).
//!
//! Uses i32x4_dot_i16x8 for the pmaddwd equivalent, manual shuffle for
//! RGB deinterleave, and u8x16_narrow_i16x8_u for saturating narrow.

#![cfg(target_arch = "wasm32")]

use crate::types::{ForwardCoeffs, pack_i16_pair, PREC};

/// 4:4:4 WASM SIMD128 encode kernel. Returns number of pixels processed.
#[arcane(import_intrinsics)]
pub(crate) fn rgb_to_yuv444_wasm(
    token: archmage::Wasm128Token,
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
    let zero = i32x4_splat(0);

    // Process 8 pixels at a time (2 x v128 of 4 pixels each).
    let blocks = n / 8;
    for blk in 0..blocks {
        let base = blk * 8;

        // Load and deinterleave 24 bytes (8 pixels) -> R, G, B arrays.
        let mut ra = [0i16; 8];
        let mut ga = [0i16; 8];
        let mut ba = [0i16; 8];
        for i in 0..8 {
            let p = (base + i) * 3;
            ra[i] = rgb[p] as i16;
            ga[i] = rgb[p + 1] as i16;
            ba[i] = rgb[p + 2] as i16;
        }

        // Process 4 pixels at a time.
        for half in 0..2 {
            let off = half * 4;

            // Interleave RG pairs: [R0,G0, R1,G1, R2,G2, R3,G3]
            let rg = i16x8(
                ra[off], ga[off], ra[off + 1], ga[off + 1],
                ra[off + 2], ga[off + 2], ra[off + 3], ga[off + 3],
            );
            // B + zero pairs: [B0,0, B1,0, B2,0, B3,0]
            let bz = i16x8(
                ba[off], 0, ba[off + 1], 0,
                ba[off + 2], 0, ba[off + 3], 0,
            );

            // Y = dot(rg, y_rg) + dot(bz, y_b0) + y_bias
            let y_val = i32x4_add(
                i32x4_add(i32x4_dot_i16x8(rg, y_rg), i32x4_dot_i16x8(bz, y_b0)),
                y_bias_v,
            );
            // Cb
            let cb_val = i32x4_add(
                i32x4_add(i32x4_dot_i16x8(rg, cb_rg), i32x4_dot_i16x8(bz, cb_b0)),
                uv_bias_v,
            );
            // Cr
            let cr_val = i32x4_add(
                i32x4_add(i32x4_dot_i16x8(rg, cr_rg), i32x4_dot_i16x8(bz, cr_b0)),
                uv_bias_v,
            );

            // Shift right by PREC.
            let y_s = i32x4_shr(y_val, PREC as u32);
            let cb_s = i32x4_shr(cb_val, PREC as u32);
            let cr_s = i32x4_shr(cr_val, PREC as u32);

            // Extract and clamp to [0, 255].
            for i in 0..4 {
                let idx = base + off + i;
                y_out[idx] = (i32x4_extract_lane::<0>(
                    i32x4_shr(i32x4_shl(y_s, (3 - i as u32) * 32), 96),
                ) as u8).max(0);
                // Simpler extraction:
                let y_arr = [
                    i32x4_extract_lane::<0>(y_s),
                    i32x4_extract_lane::<1>(y_s),
                    i32x4_extract_lane::<2>(y_s),
                    i32x4_extract_lane::<3>(y_s),
                ];
                let cb_arr = [
                    i32x4_extract_lane::<0>(cb_s),
                    i32x4_extract_lane::<1>(cb_s),
                    i32x4_extract_lane::<2>(cb_s),
                    i32x4_extract_lane::<3>(cb_s),
                ];
                let cr_arr = [
                    i32x4_extract_lane::<0>(cr_s),
                    i32x4_extract_lane::<1>(cr_s),
                    i32x4_extract_lane::<2>(cr_s),
                    i32x4_extract_lane::<3>(cr_s),
                ];
                y_out[idx] = y_arr[i].clamp(0, 255) as u8;
                cb_out[idx] = cb_arr[i].clamp(0, 255) as u8;
                cr_out[idx] = cr_arr[i].clamp(0, 255) as u8;
            }
        }
    }
    blocks * 8
}
