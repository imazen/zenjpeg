//! NEON encode kernels (aarch64 only).
//!
//! Uses vld3q_u8 to deinterleave 16 RGB pixels in one instruction, then
//! vmovl/vmlal/vqrshrn for the matrix multiply and narrowing.

#![cfg(target_arch = "aarch64")]

use crate::types::ForwardCoeffs;
use archmage::prelude::*;

/// 4:4:4 NEON encode kernel. Returns number of pixels processed (multiple of 16).
#[arcane(import_intrinsics)]
pub(crate) fn rgb_to_yuv444_neon(
    token: archmage::NeonToken,
    rgb: &[u8],
    y_out: &mut [u8],
    cb_out: &mut [u8],
    cr_out: &mut [u8],
    n: usize,
    coeffs: &ForwardCoeffs,
) -> usize {
    let yr = vdupq_n_s16(coeffs.yr);
    let yg = vdupq_n_s16(coeffs.yg);
    let yb = vdupq_n_s16(coeffs.yb);
    let cb_r = vdupq_n_s16(coeffs.cb_r);
    let cb_g = vdupq_n_s16(coeffs.cb_g);
    let cb_b = vdupq_n_s16(coeffs.cb_b);
    let cr_r = vdupq_n_s16(coeffs.cr_r);
    let cr_g = vdupq_n_s16(coeffs.cr_g);
    let cr_b = vdupq_n_s16(coeffs.cr_b);

    let y_bias_v = vdupq_n_s32(coeffs.y_bias);
    let uv_bias_v = vdupq_n_s32(coeffs.uv_bias);

    let blocks = n / 16;
    for blk in 0..blocks {
        let base = blk * 16;
        let src = &rgb[base * 3..base * 3 + 48];

        // vld3q_u8: deinterleave 48 bytes into 3 x uint8x16_t (R, G, B).
        let rgb_deint = vld3q_u8(<&[u8; 48]>::try_from(src).unwrap());
        let r_u8 = rgb_deint.0;
        let g_u8 = rgb_deint.1;
        let b_u8 = rgb_deint.2;

        // Process low 8 pixels.
        let r_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(r_u8)));
        let g_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(g_u8)));
        let b_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(b_u8)));

        let y_lo = compute_channel_neon(token, r_lo, g_lo, b_lo, yr, yg, yb, y_bias_v);
        let cb_lo = compute_channel_neon(token, r_lo, g_lo, b_lo, cb_r, cb_g, cb_b, uv_bias_v);
        let cr_lo = compute_channel_neon(token, r_lo, g_lo, b_lo, cr_r, cr_g, cr_b, uv_bias_v);

        // Process high 8 pixels.
        let r_hi = vreinterpretq_s16_u16(vmovl_high_u8(r_u8));
        let g_hi = vreinterpretq_s16_u16(vmovl_high_u8(g_u8));
        let b_hi = vreinterpretq_s16_u16(vmovl_high_u8(b_u8));

        let y_hi = compute_channel_neon(token, r_hi, g_hi, b_hi, yr, yg, yb, y_bias_v);
        let cb_hi = compute_channel_neon(token, r_hi, g_hi, b_hi, cb_r, cb_g, cb_b, uv_bias_v);
        let cr_hi = compute_channel_neon(token, r_hi, g_hi, b_hi, cr_r, cr_g, cr_b, uv_bias_v);

        // Narrow i16 -> u8 (saturating) and combine.
        let y_u8 = vcombine_u8(vqmovun_s16(y_lo), vqmovun_s16(y_hi));
        let cb_u8 = vcombine_u8(vqmovun_s16(cb_lo), vqmovun_s16(cb_hi));
        let cr_u8 = vcombine_u8(vqmovun_s16(cr_lo), vqmovun_s16(cr_hi));

        // Store.
        vst1q_u8(
            <&mut [u8; 16]>::try_from(&mut y_out[base..base + 16]).unwrap(),
            y_u8,
        );
        vst1q_u8(
            <&mut [u8; 16]>::try_from(&mut cb_out[base..base + 16]).unwrap(),
            cb_u8,
        );
        vst1q_u8(
            <&mut [u8; 16]>::try_from(&mut cr_out[base..base + 16]).unwrap(),
            cr_u8,
        );
    }
    blocks * 16
}

/// Compute one YCbCr channel for 8 pixels using NEON multiply-accumulate.
/// Returns i16x8 result ready for narrowing.
#[cfg(target_arch = "aarch64")]
#[rite]
fn compute_channel_neon(
    _token: archmage::NeonToken,
    r: core::arch::aarch64::int16x8_t,
    g: core::arch::aarch64::int16x8_t,
    b: core::arch::aarch64::int16x8_t,
    c_r: core::arch::aarch64::int16x8_t,
    c_g: core::arch::aarch64::int16x8_t,
    c_b: core::arch::aarch64::int16x8_t,
    bias: core::arch::aarch64::int32x4_t,
) -> core::arch::aarch64::int16x8_t {
    use core::arch::aarch64::*;

    // Low 4 pixels: i16 * i16 -> i32, accumulate.
    let r_lo = vmovl_s16(vget_low_s16(r));
    let g_lo = vmovl_s16(vget_low_s16(g));
    let b_lo = vmovl_s16(vget_low_s16(b));
    let cr_lo = vmovl_s16(vget_low_s16(c_r));
    let cg_lo = vmovl_s16(vget_low_s16(c_g));
    let cb_lo = vmovl_s16(vget_low_s16(c_b));

    let sum_lo = vaddq_s32(
        vaddq_s32(vmulq_s32(r_lo, cr_lo), vmulq_s32(g_lo, cg_lo)),
        vaddq_s32(vmulq_s32(b_lo, cb_lo), bias),
    );

    // High 4 pixels.
    let r_hi = vmovl_high_s16(r);
    let g_hi = vmovl_high_s16(g);
    let b_hi = vmovl_high_s16(b);
    let cr_hi = vmovl_high_s16(c_r);
    let cg_hi = vmovl_high_s16(c_g);
    let cb_hi = vmovl_high_s16(c_b);

    let sum_hi = vaddq_s32(
        vaddq_s32(vmulq_s32(r_hi, cr_hi), vmulq_s32(g_hi, cg_hi)),
        vaddq_s32(vmulq_s32(b_hi, cb_hi), bias),
    );

    // Shift right by PREC and narrow to i16.
    let shifted_lo = vshrn_n_s32::<{ crate::types::PREC as i32 }>(sum_lo);
    let shifted_hi = vshrn_n_s32::<{ crate::types::PREC as i32 }>(sum_hi);

    vcombine_s16(shifted_lo, shifted_hi)
}
