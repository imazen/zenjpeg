//! AVX2 decode kernels (x86-64 only).
//!
//! YCbCr->RGB via mulhrs_epi16 (fixed-point multiply with rounding).
//! 32 pixels per iteration for 4:4:4.

use crate::types::InverseCoeffs;
use archmage::prelude::*;
use safe_unaligned_simd::x86_64 as safe_simd;

/// 4:4:4 AVX2 decode kernel. Returns number of pixels processed (multiple of 16).
/// We process 16 pixels at a time using 128-bit SSE interleave (simpler and correct).
#[arcane]
pub(crate) fn yuv444_to_rgb_avx2(
    token: archmage::X64V3Token,
    y_plane: &[u8],
    cb_plane: &[u8],
    cr_plane: &[u8],
    rgb: &mut [u8],
    n: usize,
    coeffs: &InverseCoeffs,
) -> usize {
    use core::arch::x86_64::*;

    // Fixed-point coefficients scaled for mulhrs (which does (a*b+0x4000)>>15).
    let y_coeff_v = _mm_set1_epi16(coeffs.y_coeff_i as i16);
    let cr_to_r_v = _mm_set1_epi16(coeffs.cr_to_r_i as i16);
    let cr_to_g_v = _mm_set1_epi16(coeffs.cr_to_g_i as i16);
    let cb_to_g_v = _mm_set1_epi16(coeffs.cb_to_g_i as i16);
    let cb_to_b_v = _mm_set1_epi16(coeffs.cb_to_b_i as i16);
    let y_off_v = _mm_set1_epi16(coeffs.y_offset_i as i16);
    let uv_off_v = _mm_set1_epi16(-128);
    let zero128 = _mm_setzero_si128();

    let blocks = n / 16;
    for blk in 0..blocks {
        let base = blk * 16;

        let y_raw =
            safe_simd::_mm_loadu_si128(<&[u8; 16]>::try_from(&y_plane[base..base + 16]).unwrap());
        let cb_raw =
            safe_simd::_mm_loadu_si128(<&[u8; 16]>::try_from(&cb_plane[base..base + 16]).unwrap());
        let cr_raw =
            safe_simd::_mm_loadu_si128(<&[u8; 16]>::try_from(&cr_plane[base..base + 16]).unwrap());

        // Process low 8 and high 8 pixels.
        let y_lo = _mm_add_epi16(_mm_unpacklo_epi8(y_raw, zero128), y_off_v);
        let y_hi = _mm_add_epi16(_mm_unpackhi_epi8(y_raw, zero128), y_off_v);
        let cb_lo = _mm_add_epi16(_mm_unpacklo_epi8(cb_raw, zero128), uv_off_v);
        let cb_hi = _mm_add_epi16(_mm_unpackhi_epi8(cb_raw, zero128), uv_off_v);
        let cr_lo = _mm_add_epi16(_mm_unpacklo_epi8(cr_raw, zero128), uv_off_v);
        let cr_hi = _mm_add_epi16(_mm_unpackhi_epi8(cr_raw, zero128), uv_off_v);

        // Compute R, G, B via mulhrs (fixed-point multiply with rounding).
        let (r_lo, g_lo, b_lo) = inverse_matrix_sse(
            token, y_lo, cb_lo, cr_lo, y_coeff_v, cr_to_r_v, cr_to_g_v, cb_to_g_v, cb_to_b_v,
        );
        let (r_hi, g_hi, b_hi) = inverse_matrix_sse(
            token, y_hi, cb_hi, cr_hi, y_coeff_v, cr_to_r_v, cr_to_g_v, cb_to_g_v, cb_to_b_v,
        );

        // Pack i16 -> u8 (saturating).
        let r_u8 = _mm_packus_epi16(r_lo, r_hi);
        let g_u8 = _mm_packus_epi16(g_lo, g_hi);
        let b_u8 = _mm_packus_epi16(b_lo, b_hi);

        // Interleave R, G, B into packed RGB (48 bytes for 16 pixels).
        let (out0, out1, out2) = interleave_rgb_sse(token, r_u8, g_u8, b_u8);

        let dst = &mut rgb[base * 3..base * 3 + 48];
        safe_simd::_mm_storeu_si128(<&mut [u8; 16]>::try_from(&mut dst[0..16]).unwrap(), out0);
        safe_simd::_mm_storeu_si128(<&mut [u8; 16]>::try_from(&mut dst[16..32]).unwrap(), out1);
        safe_simd::_mm_storeu_si128(<&mut [u8; 16]>::try_from(&mut dst[32..48]).unwrap(), out2);
    }
    blocks * 16
}

/// Compute R, G, B as i16 from Y, Cb, Cr (all i16, with offsets applied).
#[rite]
fn inverse_matrix_sse(
    _token: archmage::X64V3Token,
    y: core::arch::x86_64::__m128i,
    cb: core::arch::x86_64::__m128i,
    cr: core::arch::x86_64::__m128i,
    y_coeff: core::arch::x86_64::__m128i,
    cr_to_r: core::arch::x86_64::__m128i,
    cr_to_g: core::arch::x86_64::__m128i,
    cb_to_g: core::arch::x86_64::__m128i,
    cb_to_b: core::arch::x86_64::__m128i,
) -> (
    core::arch::x86_64::__m128i,
    core::arch::x86_64::__m128i,
    core::arch::x86_64::__m128i,
) {
    use core::arch::x86_64::*;

    // mulhrs: (a*b + 0x4000) >> 15
    let y_scaled = _mm_mulhrs_epi16(y, y_coeff);

    let r = _mm_add_epi16(y_scaled, _mm_mulhrs_epi16(cr, cr_to_r));
    let g = _mm_add_epi16(
        y_scaled,
        _mm_add_epi16(_mm_mulhrs_epi16(cb, cb_to_g), _mm_mulhrs_epi16(cr, cr_to_g)),
    );
    let b = _mm_add_epi16(y_scaled, _mm_mulhrs_epi16(cb, cb_to_b));

    (r, g, b)
}

/// Interleave 16 R + 16 G + 16 B bytes into 48 packed RGB bytes (3 x __m128i).
#[rite]
fn interleave_rgb_sse(
    _token: archmage::X64V3Token,
    r: core::arch::x86_64::__m128i,
    g: core::arch::x86_64::__m128i,
    b: core::arch::x86_64::__m128i,
) -> (
    core::arch::x86_64::__m128i,
    core::arch::x86_64::__m128i,
    core::arch::x86_64::__m128i,
) {
    use core::arch::x86_64::*;

    // Produce: R0 G0 B0 R1 G1 B1 ... R15 G15 B15 (48 bytes = 3 x 16)
    // out[0] = R0 G0 B0 R1 G1 B1 R2 G2 B2 R3 G3 B3 R4 G4 B4 R5
    // out[1] = G5 B5 R6 G6 B6 R7 G7 B7 R8 G8 B8 R9 G9 B9 R10 G10
    // out[2] = B10 R11 G11 B11 R12 G12 B12 R13 G13 B13 R14 G14 B14 R15 G15 B15

    #[rustfmt::skip]
    let sh_r0 = _mm_setr_epi8(
        0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1, -1, 5,
    );
    #[rustfmt::skip]
    let sh_g0 = _mm_setr_epi8(
        -1, 0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1, -1,
    );
    #[rustfmt::skip]
    let sh_b0 = _mm_setr_epi8(
        -1, -1, 0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1,
    );

    #[rustfmt::skip]
    let sh_r1 = _mm_setr_epi8(
        -1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1, 10, -1,
    );
    #[rustfmt::skip]
    let sh_g1 = _mm_setr_epi8(
        5, -1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1, 10,
    );
    #[rustfmt::skip]
    let sh_b1 = _mm_setr_epi8(
        -1, 5, -1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1,
    );

    #[rustfmt::skip]
    let sh_r2 = _mm_setr_epi8(
        -1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15, -1, -1,
    );
    #[rustfmt::skip]
    let sh_g2 = _mm_setr_epi8(
        -1, -1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15, -1,
    );
    #[rustfmt::skip]
    let sh_b2 = _mm_setr_epi8(
        10, -1, -1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15,
    );

    let out0 = _mm_or_si128(
        _mm_or_si128(_mm_shuffle_epi8(r, sh_r0), _mm_shuffle_epi8(g, sh_g0)),
        _mm_shuffle_epi8(b, sh_b0),
    );
    let out1 = _mm_or_si128(
        _mm_or_si128(_mm_shuffle_epi8(r, sh_r1), _mm_shuffle_epi8(g, sh_g1)),
        _mm_shuffle_epi8(b, sh_b1),
    );
    let out2 = _mm_or_si128(
        _mm_or_si128(_mm_shuffle_epi8(r, sh_r2), _mm_shuffle_epi8(g, sh_g2)),
        _mm_shuffle_epi8(b, sh_b2),
    );

    (out0, out1, out2)
}
