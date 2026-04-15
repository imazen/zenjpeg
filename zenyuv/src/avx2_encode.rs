//! AVX2 encode kernels (x86-64 only).
//!
//! RGB->YCbCr via 15-bit fixed-point matrix using pmaddwd. 32 pixels per iter
//! for 4:4:4, 2x32 fused Y+chroma for 4:2:0.

use crate::types::{ForwardCoeffs, pack_i16_pair};
use archmage::prelude::*;
use safe_unaligned_simd::x86_64 as safe_simd;

/// 4:4:4 AVX2 encode kernel. Returns number of pixels processed (multiple of 32).
#[arcane]
pub(crate) fn rgb_to_yuv444_avx2(
    token: archmage::X64V3Token,
    rgb: &[u8],
    y_out: &mut [u8],
    cb_out: &mut [u8],
    cr_out: &mut [u8],
    n: usize,
    coeffs: &ForwardCoeffs,
) -> usize {
    use core::arch::x86_64::*;

    let y_rg = _mm256_set1_epi32(pack_i16_pair(coeffs.yr, coeffs.yg));
    let y_b0 = _mm256_set1_epi32(pack_i16_pair(coeffs.yb, 0));
    let cb_rg = _mm256_set1_epi32(pack_i16_pair(coeffs.cb_r, coeffs.cb_g));
    let cb_b0 = _mm256_set1_epi32(pack_i16_pair(coeffs.cb_b, 0));
    let cr_rg = _mm256_set1_epi32(pack_i16_pair(coeffs.cr_r, coeffs.cr_g));
    let cr_b0 = _mm256_set1_epi32(pack_i16_pair(coeffs.cr_b, 0));

    let y_bias = _mm256_set1_epi32(coeffs.y_bias);
    let uv_bias = _mm256_set1_epi32(coeffs.uv_bias);

    let blocks = n / 32;
    for blk in 0..blocks {
        let src = &rgb[blk * 96..blk * 96 + 96];
        let row0 = safe_simd::_mm256_loadu_si256(<&[u8; 32]>::try_from(&src[0..32]).unwrap());
        let row1 = safe_simd::_mm256_loadu_si256(<&[u8; 32]>::try_from(&src[32..64]).unwrap());
        let row2 = safe_simd::_mm256_loadu_si256(<&[u8; 32]>::try_from(&src[64..96]).unwrap());
        let (r, g, b) = deinterleave_rgb_avx2(token, row0, row1, row2);

        let (y_lo, y_hi) = matrix_row_avx2(token, r, g, b, y_rg, y_b0, y_bias);
        let (cb_lo, cb_hi) = matrix_row_avx2(token, r, g, b, cb_rg, cb_b0, uv_bias);
        let (cr_lo, cr_hi) = matrix_row_avx2(token, r, g, b, cr_rg, cr_b0, uv_bias);

        store_u8x32_avx2(token, &mut y_out[blk * 32..blk * 32 + 32], y_lo, y_hi);
        store_u8x32_avx2(token, &mut cb_out[blk * 32..blk * 32 + 32], cb_lo, cb_hi);
        store_u8x32_avx2(token, &mut cr_out[blk * 32..blk * 32 + 32], cr_lo, cr_hi);
    }
    blocks * 32
}

/// 4:2:0 AVX2 fused kernel: processes 2 rows x 32 pixels per iter.
#[arcane]
pub(crate) fn rgb_to_yuv420_avx2(
    token: archmage::X64V3Token,
    rgb: &[u8],
    y_out: &mut [u8],
    cb_out: &mut [u8],
    cr_out: &mut [u8],
    width: usize,
    height: usize,
    cw: usize,
    coeffs: &ForwardCoeffs,
) {
    use core::arch::x86_64::*;

    let y_rg = _mm256_set1_epi32(pack_i16_pair(coeffs.yr, coeffs.yg));
    let y_b0 = _mm256_set1_epi32(pack_i16_pair(coeffs.yb, 0));
    let cb_rg = _mm256_set1_epi32(pack_i16_pair(coeffs.cb_r, coeffs.cb_g));
    let cb_b0 = _mm256_set1_epi32(pack_i16_pair(coeffs.cb_b, 0));
    let cr_rg = _mm256_set1_epi32(pack_i16_pair(coeffs.cr_r, coeffs.cr_g));
    let cr_b0 = _mm256_set1_epi32(pack_i16_pair(coeffs.cr_b, 0));

    let y_bias_v = _mm256_set1_epi32(coeffs.y_bias);
    let uv_bias_v = _mm256_set1_epi32(coeffs.uv_bias_420);

    let all_ones = _mm256_set1_epi8(1);
    let row_stride = width * 3;
    let col_blocks = width / 32;

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
            let px = cx * 32;
            let src_top = &rgb[top_off + px * 3..top_off + px * 3 + 96];
            let src_bot = &rgb[bot_off + px * 3..bot_off + px * 3 + 96];

            let t0 = safe_simd::_mm256_loadu_si256(<&[u8; 32]>::try_from(&src_top[0..32]).unwrap());
            let t1 =
                safe_simd::_mm256_loadu_si256(<&[u8; 32]>::try_from(&src_top[32..64]).unwrap());
            let t2 =
                safe_simd::_mm256_loadu_si256(<&[u8; 32]>::try_from(&src_top[64..96]).unwrap());
            let (r_top, g_top, b_top) = deinterleave_rgb_avx2(token, t0, t1, t2);

            let b0 = safe_simd::_mm256_loadu_si256(<&[u8; 32]>::try_from(&src_bot[0..32]).unwrap());
            let b1 =
                safe_simd::_mm256_loadu_si256(<&[u8; 32]>::try_from(&src_bot[32..64]).unwrap());
            let b2 =
                safe_simd::_mm256_loadu_si256(<&[u8; 32]>::try_from(&src_bot[64..96]).unwrap());
            let (r_bot, g_bot, b_bot) = deinterleave_rgb_avx2(token, b0, b1, b2);

            // Y for both rows.
            let (yt_lo, yt_hi) = matrix_row_avx2(token, r_top, g_top, b_top, y_rg, y_b0, y_bias_v);
            store_u8x32_avx2(
                token,
                &mut y_out[y_top_off + px..y_top_off + px + 32],
                yt_lo,
                yt_hi,
            );
            let (yb_lo, yb_hi) = matrix_row_avx2(token, r_bot, g_bot, b_bot, y_rg, y_b0, y_bias_v);
            store_u8x32_avx2(
                token,
                &mut y_out[y_bot_off + px..y_bot_off + px + 32],
                yb_lo,
                yb_hi,
            );

            // Chroma: vertical avg then horizontal pair-sum.
            let r_avg = _mm256_avg_epu8(r_top, r_bot);
            let g_avg = _mm256_avg_epu8(g_top, g_bot);
            let b_avg = _mm256_avg_epu8(b_top, b_bot);
            let r_sum = _mm256_maddubs_epi16(r_avg, all_ones);
            let g_sum = _mm256_maddubs_epi16(g_avg, all_ones);
            let b_sum = _mm256_maddubs_epi16(b_avg, all_ones);

            let zero = _mm256_setzero_si256();
            let (rg_a, rg_b) = interleave_epi16_avx2(token, r_sum, g_sum);
            let (bz_a, bz_b) = interleave_epi16_avx2(token, b_sum, zero);

            // Cb
            let cb_lo = _mm256_add_epi32(
                _mm256_add_epi32(
                    _mm256_madd_epi16(rg_a, cb_rg),
                    _mm256_madd_epi16(bz_a, cb_b0),
                ),
                uv_bias_v,
            );
            let cb_hi = _mm256_add_epi32(
                _mm256_add_epi32(
                    _mm256_madd_epi16(rg_b, cb_rg),
                    _mm256_madd_epi16(bz_b, cb_b0),
                ),
                uv_bias_v,
            );
            let cb_u16 = pack_u16_avx2(
                token,
                _mm256_srai_epi32::<16>(cb_lo),
                _mm256_srai_epi32::<16>(cb_hi),
            );
            let cb_u8 = _mm256_packus_epi16(cb_u16, zero);
            let cb_u8 = _mm256_permute4x64_epi64::<0b11_01_10_00>(cb_u8);
            safe_simd::_mm_storeu_si128(
                <&mut [u8; 16]>::try_from(
                    &mut cb_out[cb_row_off + cx * 16..cb_row_off + cx * 16 + 16],
                )
                .unwrap(),
                _mm256_castsi256_si128(cb_u8),
            );

            // Cr
            let cr_lo = _mm256_add_epi32(
                _mm256_add_epi32(
                    _mm256_madd_epi16(rg_a, cr_rg),
                    _mm256_madd_epi16(bz_a, cr_b0),
                ),
                uv_bias_v,
            );
            let cr_hi = _mm256_add_epi32(
                _mm256_add_epi32(
                    _mm256_madd_epi16(rg_b, cr_rg),
                    _mm256_madd_epi16(bz_b, cr_b0),
                ),
                uv_bias_v,
            );
            let cr_u16 = pack_u16_avx2(
                token,
                _mm256_srai_epi32::<16>(cr_lo),
                _mm256_srai_epi32::<16>(cr_hi),
            );
            let cr_u8 = _mm256_packus_epi16(cr_u16, zero);
            let cr_u8 = _mm256_permute4x64_epi64::<0b11_01_10_00>(cr_u8);
            safe_simd::_mm_storeu_si128(
                <&mut [u8; 16]>::try_from(
                    &mut cr_out[cb_row_off + cx * 16..cb_row_off + cx * 16 + 16],
                )
                .unwrap(),
                _mm256_castsi256_si128(cr_u8),
            );
        }
    }

    // Scalar tail: remaining columns, odd last row, etc.
    crate::encode::rgb_to_yuv420_scalar_tail(
        rgb,
        y_out,
        cb_out,
        cr_out,
        width,
        height,
        cw,
        col_blocks * 32,
        coeffs,
    );
}

/// Y-only AVX2 kernel: computes the Y plane for a 4:2:0 input. Skips all
/// chroma computation — useful when the caller will replace chroma with a
/// gamma-corrected (or other custom) downsampling. Same arithmetic as the
/// Y part of `rgb_to_yuv420_avx2`; the chroma code is simply elided.
#[arcane]
pub(crate) fn rgb_to_yuv420_y_only_avx2(
    token: archmage::X64V3Token,
    rgb: &[u8],
    y_out: &mut [u8],
    width: usize,
    height: usize,
    coeffs: &ForwardCoeffs,
) {
    use core::arch::x86_64::*;

    let y_rg = _mm256_set1_epi32(pack_i16_pair(coeffs.yr, coeffs.yg));
    let y_b0 = _mm256_set1_epi32(pack_i16_pair(coeffs.yb, 0));
    let y_bias_v = _mm256_set1_epi32(coeffs.y_bias);

    let row_stride = width * 3;
    let col_blocks = width / 32;

    for row in 0..height {
        let row_off = row * row_stride;
        let y_row_off = row * width;
        for cx in 0..col_blocks {
            let px = cx * 32;
            let src = &rgb[row_off + px * 3..row_off + px * 3 + 96];
            let t0 = safe_simd::_mm256_loadu_si256(<&[u8; 32]>::try_from(&src[0..32]).unwrap());
            let t1 = safe_simd::_mm256_loadu_si256(<&[u8; 32]>::try_from(&src[32..64]).unwrap());
            let t2 = safe_simd::_mm256_loadu_si256(<&[u8; 32]>::try_from(&src[64..96]).unwrap());
            let (r_v, g_v, b_v) = deinterleave_rgb_avx2(token, t0, t1, t2);
            let (y_lo, y_hi) = matrix_row_avx2(token, r_v, g_v, b_v, y_rg, y_b0, y_bias_v);
            store_u8x32_avx2(
                token,
                &mut y_out[y_row_off + px..y_row_off + px + 32],
                y_lo,
                y_hi,
            );
        }
    }

    // Scalar tail: Y only. Covers the [col_blocks * 32..width) columns that
    // the AVX2 kernel skipped.
    crate::encode::rgb_to_y_scalar_tail(rgb, y_out, width, height, col_blocks * 32, coeffs);
}

// ── AVX2 helper functions ──────────────────────────────────────────────────

/// Deinterleave 96 bytes of packed RGB into three 32-byte plane vectors.
#[rite]
pub(crate) fn deinterleave_rgb_avx2(
    _token: archmage::X64V3Token,
    row0: core::arch::x86_64::__m256i,
    row1: core::arch::x86_64::__m256i,
    row2: core::arch::x86_64::__m256i,
) -> (
    core::arch::x86_64::__m256i,
    core::arch::x86_64::__m256i,
    core::arch::x86_64::__m256i,
) {
    use core::arch::x86_64::*;
    let s02_low = _mm256_permute2x128_si256::<0x20>(row0, row2);
    let s02_high = _mm256_permute2x128_si256::<0x31>(row0, row2);
    #[rustfmt::skip]
    let m0 = _mm256_setr_epi8(
        0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0,
        0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0,
    );
    #[rustfmt::skip]
    let m1 = _mm256_setr_epi8(
        0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0,
        -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1,
    );
    let c0 = _mm256_blendv_epi8(_mm256_blendv_epi8(s02_low, s02_high, m0), row1, m1);
    let c1 = _mm256_blendv_epi8(_mm256_blendv_epi8(s02_high, s02_low, m1), row1, m0);
    let c2 = _mm256_blendv_epi8(_mm256_blendv_epi8(row1, s02_low, m0), s02_high, m1);

    #[rustfmt::skip]
    let sh_c0 = _mm256_setr_epi8(
        0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14, 1, 4, 7, 10, 13,
        0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14, 1, 4, 7, 10, 13,
    );
    #[rustfmt::skip]
    let sh_c1 = _mm256_setr_epi8(
        1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14,
        1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14,
    );
    #[rustfmt::skip]
    let sh_c2 = _mm256_setr_epi8(
        2, 5, 8, 11, 14, 1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15,
        2, 5, 8, 11, 14, 1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15,
    );
    (
        _mm256_shuffle_epi8(c0, sh_c0),
        _mm256_shuffle_epi8(c1, sh_c1),
        _mm256_shuffle_epi8(c2, sh_c2),
    )
}

/// For 32 u8-packed R/G/B inputs, compute one output channel via 15-bit
/// fixed-point matrix, returning two u16x16 halves ready for packus.
#[rite]
pub(crate) fn matrix_row_avx2(
    token: archmage::X64V3Token,
    r: core::arch::x86_64::__m256i,
    g: core::arch::x86_64::__m256i,
    b: core::arch::x86_64::__m256i,
    rg_coef: core::arch::x86_64::__m256i,
    b_coef: core::arch::x86_64::__m256i,
    bias: core::arch::x86_64::__m256i,
) -> (core::arch::x86_64::__m256i, core::arch::x86_64::__m256i) {
    use core::arch::x86_64::*;
    let zero = _mm256_setzero_si256();

    let r_l = _mm256_unpacklo_epi8(r, zero);
    let r_h = _mm256_unpackhi_epi8(r, zero);
    let g_l = _mm256_unpacklo_epi8(g, zero);
    let g_h = _mm256_unpackhi_epi8(g, zero);
    let b_l = _mm256_unpacklo_epi8(b, zero);
    let b_h = _mm256_unpackhi_epi8(b, zero);

    let (rg_a, rg_b) = interleave_epi16_avx2(token, r_l, g_l);
    let (rg_c, rg_d) = interleave_epi16_avx2(token, r_h, g_h);
    let (b_a, b_b) = interleave_epi16_avx2(token, b_l, zero);
    let (b_c, b_d) = interleave_epi16_avx2(token, b_h, zero);

    let lo = _mm256_add_epi32(
        _mm256_add_epi32(
            _mm256_madd_epi16(rg_a, rg_coef),
            _mm256_madd_epi16(b_a, b_coef),
        ),
        bias,
    );
    let mid1 = _mm256_add_epi32(
        _mm256_add_epi32(
            _mm256_madd_epi16(rg_b, rg_coef),
            _mm256_madd_epi16(b_b, b_coef),
        ),
        bias,
    );
    let mid2 = _mm256_add_epi32(
        _mm256_add_epi32(
            _mm256_madd_epi16(rg_c, rg_coef),
            _mm256_madd_epi16(b_c, b_coef),
        ),
        bias,
    );
    let hi = _mm256_add_epi32(
        _mm256_add_epi32(
            _mm256_madd_epi16(rg_d, rg_coef),
            _mm256_madd_epi16(b_d, b_coef),
        ),
        bias,
    );

    let lo_s = _mm256_srai_epi32::<15>(lo);
    let m1_s = _mm256_srai_epi32::<15>(mid1);
    let m2_s = _mm256_srai_epi32::<15>(mid2);
    let hi_s = _mm256_srai_epi32::<15>(hi);

    let u16_lo = pack_u16_avx2(token, lo_s, m1_s);
    let u16_hi = pack_u16_avx2(token, m2_s, hi_s);
    (u16_lo, u16_hi)
}

#[rite]
pub(crate) fn interleave_epi16_avx2(
    _token: archmage::X64V3Token,
    a: core::arch::x86_64::__m256i,
    b: core::arch::x86_64::__m256i,
) -> (core::arch::x86_64::__m256i, core::arch::x86_64::__m256i) {
    use core::arch::x86_64::*;
    let l = _mm256_unpacklo_epi16(a, b);
    let h = _mm256_unpackhi_epi16(a, b);
    (
        _mm256_permute2x128_si256::<0x20>(l, h),
        _mm256_permute2x128_si256::<0x31>(l, h),
    )
}

/// Pack two i32x8 -> u16x16 (saturating) with lane-order fixup.
#[rite]
pub(crate) fn pack_u16_avx2(
    _token: archmage::X64V3Token,
    a: core::arch::x86_64::__m256i,
    b: core::arch::x86_64::__m256i,
) -> core::arch::x86_64::__m256i {
    use core::arch::x86_64::*;
    let p = _mm256_packus_epi32(a, b);
    _mm256_permute4x64_epi64::<0b11_01_10_00>(p)
}

/// Pack two u16x16 -> u8x32 (saturating) and store.
#[rite]
pub(crate) fn store_u8x32_avx2(
    _token: archmage::X64V3Token,
    dst: &mut [u8],
    a: core::arch::x86_64::__m256i,
    b: core::arch::x86_64::__m256i,
) {
    use core::arch::x86_64::*;
    let p = _mm256_packus_epi16(a, b);
    safe_simd::_mm256_storeu_si256(<&mut [u8; 32]>::try_from(&mut dst[..32]).unwrap(), p);
}
