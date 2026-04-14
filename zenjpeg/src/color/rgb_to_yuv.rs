//! Internal RGB→YCbCr conversion (BT.601 full-range JFIF).
//!
//! Drop-in replacement for `yuv::rgb_to_yuv444` / `yuv::rgb_to_yuv420` in
//! Professional mode, implemented with magetypes generics. Compiles to AVX2+FMA
//! on x86_64, NEON on aarch64, WASM SIMD128, and a scalar fallback via the same
//! `#[magetypes]` attribute used elsewhere in zenjpeg.
//!
//! BT.601 full-range coefficients (`YuvStandardMatrix::Bt601` + `YuvRange::Full`):
//! ```text
//! Y  =  0.299 R + 0.587 G + 0.114 B
//! Cb = -0.168736 R - 0.331264 G + 0.5 B + 128
//! Cr =  0.5 R - 0.418688 G - 0.081312 B + 128
//! ```
//!
//! We do the math in f32 with FMA rather than i16 fixed-point. On Zen 4 / current
//! Intel cores the f32 FMA and 16-bit integer `pmaddwd` paths have identical
//! latency and throughput, and f32 rounds exactly once at the end (via
//! `to_i32_round`), so output precision equals or exceeds the yuv crate's
//! 15-bit Professional mode.

use archmage::prelude::*;
use magetypes::simd::generic::f32x8 as GenericF32x8;
use magetypes::simd::generic::i32x8 as GenericI32x8;

#[cfg(target_arch = "x86_64")]
use safe_unaligned_simd::x86_64 as safe_simd;

// BT.601 full-range forward matrix.
const YR: f32 = 0.299;
const YG: f32 = 0.587;
const YB: f32 = 0.114;
const CB_R: f32 = -0.168_736;
const CB_G: f32 = -0.331_264;
const CB_B: f32 = 0.5;
const CR_R: f32 = 0.5;
const CR_G: f32 = -0.418_688;
const CR_B: f32 = -0.081_312;
const CHROMA_BIAS: f32 = 128.0;

// 15-bit fixed-point integer coefficients for the AVX2/AVX-512 paths. These
// match the yuv crate's Professional mode (`PRECISION = 15`).
const PREC: i32 = 15;
const I16_YR: i16 = 9798;
const I16_YG: i16 = 19235;
const I16_YB: i16 = 3735;
const I16_CB_R: i16 = -5528;
const I16_CB_G: i16 = -10855;
const I16_CB_B: i16 = 16384;
const I16_CR_R: i16 = 16384;
const I16_CR_G: i16 = -13720;
const I16_CR_B: i16 = -2665;

/// Pack a pair of i16 coefficients into a 32-bit value so pmaddwd reads them
/// as `(low, high)` and computes `a*x + b*y` per i32 lane.
const fn pack_i16_pair(a: i16, b: i16) -> i32 {
    ((a as u16 as u32) | ((b as u16 as u32) << 16)) as i32
}

/// Convert packed 24-bit RGB to three u8 Y/Cb/Cr planes at full resolution.
///
/// `rgb` must be `width * height * 3` bytes. The three output planes must each
/// be at least `width * height` bytes. No padding is written.
pub fn rgb_to_yuv444(
    rgb: &[u8],
    y: &mut [u8],
    cb: &mut [u8],
    cr: &mut [u8],
    width: usize,
    height: usize,
) {
    let n = width * height;
    assert!(rgb.len() >= n * 3);
    assert!(y.len() >= n);
    assert!(cb.len() >= n);
    assert!(cr.len() >= n);

    #[cfg(target_arch = "x86_64")]
    if let Some(token) = archmage::X64V3Token::summon() {
        let done = rgb_to_yuv444_avx2(token, rgb, y, cb, cr, n);
        if done < n {
            rgb_to_yuv444_scalar_tail(rgb, y, cb, cr, done, n);
        }
        return;
    }
    incant!(rgb_to_yuv444_impl(rgb, y, cb, cr, n));
}

#[inline]
fn rgb_to_yuv444_scalar_tail(
    rgb: &[u8],
    y: &mut [u8],
    cb: &mut [u8],
    cr: &mut [u8],
    start: usize,
    end: usize,
) {
    for i in start..end {
        let p = i * 3;
        let r = rgb[p] as f32;
        let g = rgb[p + 1] as f32;
        let b = rgb[p + 2] as f32;
        y[i] = clamp_round(YR * r + YG * g + YB * b);
        cb[i] = clamp_round(CB_R * r + CB_G * g + CB_B * b + CHROMA_BIAS);
        cr[i] = clamp_round(CR_R * r + CR_G * g + CR_B * b + CHROMA_BIAS);
    }
}

/// Convert packed 24-bit RGB to three u8 Y/Cb/Cr planes with 4:2:0 subsampling.
///
/// Y is full-resolution (`width * height`); Cb/Cr are `ceil(width/2) *
/// ceil(height/2)`. Chroma downsampling is a 2×2 box average in YCbCr space
/// (matching the yuv crate's Professional mode).
///
/// Odd widths/heights edge-replicate: the last partial 2×2 cell uses available
/// pixels only, still divided by 4 to match the yuv crate's behavior.
pub fn rgb_to_yuv420(
    rgb: &[u8],
    y: &mut [u8],
    cb: &mut [u8],
    cr: &mut [u8],
    width: usize,
    height: usize,
) {
    let n = width * height;
    let cw = width.div_ceil(2);
    let ch = height.div_ceil(2);
    assert!(rgb.len() >= n * 3);
    assert!(y.len() >= n);
    assert!(cb.len() >= cw * ch);
    assert!(cr.len() >= cw * ch);

    #[cfg(target_arch = "x86_64")]
    if let Some(token) = archmage::X64V3Token::summon() {
        rgb_to_yuv420_avx2(token, rgb, y, cb, cr, width, height, cw);
        return;
    }
    incant!(rgb_to_yuv420_impl(rgb, y, cb, cr, width, height));
}

/// 4:4:4 kernel. Processes pixels in groups of 8 via `f32x8`, with a scalar
/// tail for the final `< 8` pixels.
#[magetypes(v3, neon, wasm128, scalar)]
#[inline(always)]
fn rgb_to_yuv444_impl(
    token: Token,
    rgb: &[u8],
    y: &mut [u8],
    cb: &mut [u8],
    cr: &mut [u8],
    n: usize,
) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    #[allow(non_camel_case_types)]
    type i32x8 = GenericI32x8<Token>;

    let yr = f32x8::splat(token, YR);
    let yg = f32x8::splat(token, YG);
    let yb = f32x8::splat(token, YB);
    let cb_r = f32x8::splat(token, CB_R);
    let cb_g = f32x8::splat(token, CB_G);
    let cb_b = f32x8::splat(token, CB_B);
    let cr_r = f32x8::splat(token, CR_R);
    let cr_g = f32x8::splat(token, CR_G);
    let cr_b = f32x8::splat(token, CR_B);
    let bias = f32x8::splat(token, CHROMA_BIAS);
    let zero = i32x8::zero(token);
    let max255 = i32x8::splat(token, 255);

    let chunks = n / 8;
    for c in 0..chunks {
        let base = c * 8;
        let mut ra = [0.0f32; 8];
        let mut ga = [0.0f32; 8];
        let mut ba = [0.0f32; 8];
        for i in 0..8 {
            let p = (base + i) * 3;
            ra[i] = rgb[p] as f32;
            ga[i] = rgb[p + 1] as f32;
            ba[i] = rgb[p + 2] as f32;
        }
        let r = f32x8::from_array(token, ra);
        let g = f32x8::from_array(token, ga);
        let b = f32x8::from_array(token, ba);

        // Use mul_add chains so every multiply-add becomes one FMA.
        let y_f = r.mul_add(yr, g.mul_add(yg, b * yb));
        let cb_f = r.mul_add(cb_r, g.mul_add(cb_g, b.mul_add(cb_b, bias)));
        let cr_f = r.mul_add(cr_r, g.mul_add(cr_g, b.mul_add(cr_b, bias)));

        let y_i = y_f.to_i32_round().max(zero).min(max255).to_array();
        let cb_i = cb_f.to_i32_round().max(zero).min(max255).to_array();
        let cr_i = cr_f.to_i32_round().max(zero).min(max255).to_array();

        for i in 0..8 {
            y[base + i] = y_i[i] as u8;
            cb[base + i] = cb_i[i] as u8;
            cr[base + i] = cr_i[i] as u8;
        }
    }

    // Scalar tail.
    for i in (chunks * 8)..n {
        let p = i * 3;
        let r = rgb[p] as f32;
        let g = rgb[p + 1] as f32;
        let b = rgb[p + 2] as f32;
        y[i] = clamp_round(YR * r + YG * g + YB * b);
        cb[i] = clamp_round(CB_R * r + CB_G * g + CB_B * b + CHROMA_BIAS);
        cr[i] = clamp_round(CR_R * r + CR_G * g + CR_B * b + CHROMA_BIAS);
    }
}

/// 4:2:0 kernel. Computes Y at full resolution and Cb/Cr at 2×2 block centers
/// by averaging the RGB input block first, then running the forward matrix
/// once per chroma sample. Matches the yuv crate's Professional 4:2:0 output.
#[magetypes(v3, neon, wasm128, scalar)]
#[inline(always)]
fn rgb_to_yuv420_impl(
    token: Token,
    rgb: &[u8],
    y: &mut [u8],
    cb: &mut [u8],
    cr: &mut [u8],
    width: usize,
    height: usize,
) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    #[allow(non_camel_case_types)]
    type i32x8 = GenericI32x8<Token>;

    let cw = width.div_ceil(2);
    let n = width * height;

    // Y plane: full-resolution, matches the 4:4:4 kernel.
    let yr_v = f32x8::splat(token, YR);
    let yg_v = f32x8::splat(token, YG);
    let yb_v = f32x8::splat(token, YB);
    let zero = i32x8::zero(token);
    let max255 = i32x8::splat(token, 255);
    let chunks = n / 8;
    for c in 0..chunks {
        let base = c * 8;
        let mut ra = [0.0f32; 8];
        let mut ga = [0.0f32; 8];
        let mut ba = [0.0f32; 8];
        for i in 0..8 {
            let p = (base + i) * 3;
            ra[i] = rgb[p] as f32;
            ga[i] = rgb[p + 1] as f32;
            ba[i] = rgb[p + 2] as f32;
        }
        let r = f32x8::from_array(token, ra);
        let g = f32x8::from_array(token, ga);
        let b = f32x8::from_array(token, ba);
        let y_f = r.mul_add(yr_v, g.mul_add(yg_v, b * yb_v));
        let y_i = y_f.to_i32_round().max(zero).min(max255).to_array();
        for i in 0..8 {
            y[base + i] = y_i[i] as u8;
        }
    }
    for i in (chunks * 8)..n {
        let p = i * 3;
        let r = rgb[p] as f32;
        let g = rgb[p + 1] as f32;
        let b = rgb[p + 2] as f32;
        y[i] = clamp_round(YR * r + YG * g + YB * b);
    }

    // Chroma: iterate 2×2 blocks.
    let mut cy = 0usize;
    let mut row = 0usize;
    while row < height {
        let row1 = (row + 1).min(height - 1);
        let mut cx = 0usize;
        let mut col = 0usize;
        while col < width {
            let col1 = (col + 1).min(width - 1);

            // Average 2×2 RGB block (1×2 or 1×1 at right/bottom edges).
            let i00 = (row * width + col) * 3;
            let i01 = (row * width + col1) * 3;
            let i10 = (row1 * width + col) * 3;
            let i11 = (row1 * width + col1) * 3;
            let r = (rgb[i00] as u32 + rgb[i01] as u32 + rgb[i10] as u32 + rgb[i11] as u32) as f32
                * 0.25;
            let g = (rgb[i00 + 1] as u32
                + rgb[i01 + 1] as u32
                + rgb[i10 + 1] as u32
                + rgb[i11 + 1] as u32) as f32
                * 0.25;
            let b = (rgb[i00 + 2] as u32
                + rgb[i01 + 2] as u32
                + rgb[i10 + 2] as u32
                + rgb[i11 + 2] as u32) as f32
                * 0.25;

            cb[cy * cw + cx] = clamp_round(CB_R * r + CB_G * g + CB_B * b + CHROMA_BIAS);
            cr[cy * cw + cx] = clamp_round(CR_R * r + CR_G * g + CR_B * b + CHROMA_BIAS);

            cx += 1;
            col += 2;
        }
        cy += 1;
        row += 2;
    }
}

#[inline(always)]
fn clamp_round(v: f32) -> u8 {
    let r = v.round() as i32;
    r.clamp(0, 255) as u8
}

// ── AVX2 kernels ────────────────────────────────────────────────────────────
//
// RGB→YCbCr via 15-bit fixed-point matrix using pmaddwd. 32 pixels per iter.
// Deinterleave is a permute2x128 + 3×blendv + 3×pshufb pattern; the "output
// channel 0" (first byte offset) maps to R for RGB input.

#[cfg(target_arch = "x86_64")]
#[arcane]
fn rgb_to_yuv444_avx2(
    token: archmage::X64V3Token,
    rgb: &[u8],
    y_out: &mut [u8],
    cb_out: &mut [u8],
    cr_out: &mut [u8],
    n: usize,
) -> usize {
    use core::arch::x86_64::*;

    let y_rg = _mm256_set1_epi32(pack_i16_pair(I16_YR, I16_YG));
    let y_b0 = _mm256_set1_epi32(pack_i16_pair(I16_YB, 0));
    let cb_rg = _mm256_set1_epi32(pack_i16_pair(I16_CB_R, I16_CB_G));
    let cb_b0 = _mm256_set1_epi32(pack_i16_pair(I16_CB_B, 0));
    let cr_rg = _mm256_set1_epi32(pack_i16_pair(I16_CR_R, I16_CR_G));
    let cr_b0 = _mm256_set1_epi32(pack_i16_pair(I16_CR_B, 0));

    let round_y = (1i32 << (PREC - 1)) - 1;
    let y_bias = _mm256_set1_epi32(round_y);
    let uv_bias = _mm256_set1_epi32((128i32 << PREC) + round_y);

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

/// Deinterleave 96 bytes of packed RGB into three 32-byte plane vectors.
/// The "first channel" of the input (byte 0) ends up as the first return value,
/// i.e. R for RGB input. Pattern adapted from the zune-jpeg / yuv crate idiom.
#[cfg(target_arch = "x86_64")]
#[rite]
fn deinterleave_rgb_avx2(
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
    // "b0/g0/r0" are yuv-crate internal names — for RGB input, they end up as
    // R/G/B planes in that order (see load_deinterleave_rgb_for_yuv caller).
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

/// For 32 u8-packed R/G/B inputs, compute one output channel (Y or Cb or Cr)
/// via the 15-bit fixed-point matrix, returning two i32x8 halves that still
/// need to be packed to u8.
#[cfg(target_arch = "x86_64")]
#[rite]
fn matrix_row_avx2(
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

    // Widen to i16: unpacklo/unpackhi split 32 u8 → 16 u16 each half.
    let r_l = _mm256_unpacklo_epi8(r, zero);
    let r_h = _mm256_unpackhi_epi8(r, zero);
    let g_l = _mm256_unpacklo_epi8(g, zero);
    let g_h = _mm256_unpackhi_epi8(g, zero);
    let b_l = _mm256_unpacklo_epi8(b, zero);
    let b_h = _mm256_unpackhi_epi8(b, zero);

    // Interleave RG pairs across lanes via permute2x128 so pmaddwd output lanes
    // correspond to contiguous pixels.
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

    // Shift right by 15 (arithmetic so negative intermediates are preserved;
    // final packus saturates to [0, 65535] → [0, 255]).
    let lo_s = _mm256_srai_epi32::<15>(lo);
    let m1_s = _mm256_srai_epi32::<15>(mid1);
    let m2_s = _mm256_srai_epi32::<15>(mid2);
    let hi_s = _mm256_srai_epi32::<15>(hi);

    // Pack pairs of i32x8 → u16x16 (saturating). packus_epi32 does within-lane
    // packing, so fix with permute4x64 afterwards.
    let u16_lo = pack_u16_avx2(token, lo_s, m1_s);
    let u16_hi = pack_u16_avx2(token, m2_s, hi_s);
    (u16_lo, u16_hi)
}

#[cfg(target_arch = "x86_64")]
#[rite]
fn interleave_epi16_avx2(
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

/// Pack two i32x8 → u16x16 (saturating) with lane-order fixup.
#[cfg(target_arch = "x86_64")]
#[rite]
fn pack_u16_avx2(
    _token: archmage::X64V3Token,
    a: core::arch::x86_64::__m256i,
    b: core::arch::x86_64::__m256i,
) -> core::arch::x86_64::__m256i {
    use core::arch::x86_64::*;
    let p = _mm256_packus_epi32(a, b);
    _mm256_permute4x64_epi64::<0b11_01_10_00>(p)
}

/// Pack two u16x16 → u8x32 (saturating) and store. No lane-fixup is needed
/// here: the two `pack_u16_avx2` inputs already carry `[p0..p7 | p16..p23]`
/// and `[p8..p15 | p24..p31]`, so the within-lane `packus_epi16` yields
/// `[p0..p15 | p16..p31]` linearly.
#[cfg(target_arch = "x86_64")]
#[rite]
fn store_u8x32_avx2(
    _token: archmage::X64V3Token,
    dst: &mut [u8],
    a: core::arch::x86_64::__m256i,
    b: core::arch::x86_64::__m256i,
) {
    use core::arch::x86_64::*;
    let p = _mm256_packus_epi16(a, b);
    safe_simd::_mm256_storeu_si256(<&mut [u8; 32]>::try_from(&mut dst[..32]).unwrap(), p);
}

// ── AVX2 4:2:0 ──────────────────────────────────────────────────────────────
//
// Fused kernel: processes 2 rows × 32 pixels per iter.
// Y: full-res via matrix_row (64 output bytes per iter).
// Cb/Cr: fused — horizontal pair-sum of raw R/G/B u8 planes via maddubs,
// vertical sum of two rows, then pmaddwd matrix at PREC+1 = 16 bit shift.
// Avoids the full-res Cb/Cr → u8 → maddubs → average round-trip.

#[cfg(target_arch = "x86_64")]
#[arcane]
fn rgb_to_yuv420_avx2(
    token: archmage::X64V3Token,
    rgb: &[u8],
    y_out: &mut [u8],
    cb_out: &mut [u8],
    cr_out: &mut [u8],
    width: usize,
    height: usize,
    cw: usize,
) {
    use core::arch::x86_64::*;

    let y_rg = _mm256_set1_epi32(pack_i16_pair(I16_YR, I16_YG));
    let y_b0 = _mm256_set1_epi32(pack_i16_pair(I16_YB, 0));
    let cb_rg = _mm256_set1_epi32(pack_i16_pair(I16_CB_R, I16_CB_G));
    let cb_b0 = _mm256_set1_epi32(pack_i16_pair(I16_CB_B, 0));
    let cr_rg = _mm256_set1_epi32(pack_i16_pair(I16_CR_R, I16_CR_G));
    let cr_b0 = _mm256_set1_epi32(pack_i16_pair(I16_CR_B, 0));

    let round_y = (1i32 << (PREC - 1)) - 1;
    let y_bias_v = _mm256_set1_epi32(round_y);

    // 4:2:0 chroma uses PREC+1 = 16 to absorb the ×4 from the 2×2 sum.
    let uv_prec = PREC + 1;
    let uv_round = (1i32 << (uv_prec - 1)) - 1;
    let uv_bias_v = _mm256_set1_epi32((128i32 << uv_prec) + uv_round);

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

            // Deinterleave both rows.
            let t0 = safe_simd::_mm256_loadu_si256(<&[u8; 32]>::try_from(&src_top[0..32]).unwrap());
            let t1 = safe_simd::_mm256_loadu_si256(<&[u8; 32]>::try_from(&src_top[32..64]).unwrap());
            let t2 = safe_simd::_mm256_loadu_si256(<&[u8; 32]>::try_from(&src_top[64..96]).unwrap());
            let (r_top, g_top, b_top) = deinterleave_rgb_avx2(token, t0, t1, t2);

            let b0 = safe_simd::_mm256_loadu_si256(<&[u8; 32]>::try_from(&src_bot[0..32]).unwrap());
            let b1 = safe_simd::_mm256_loadu_si256(<&[u8; 32]>::try_from(&src_bot[32..64]).unwrap());
            let b2 = safe_simd::_mm256_loadu_si256(<&[u8; 32]>::try_from(&src_bot[64..96]).unwrap());
            let (r_bot, g_bot, b_bot) = deinterleave_rgb_avx2(token, b0, b1, b2);

            // Y for both rows (full-res, same as 4:4:4).
            let (yt_lo, yt_hi) = matrix_row_avx2(token, r_top, g_top, b_top, y_rg, y_b0, y_bias_v);
            store_u8x32_avx2(token, &mut y_out[y_top_off + px..y_top_off + px + 32], yt_lo, yt_hi);
            let (yb_lo, yb_hi) = matrix_row_avx2(token, r_bot, g_bot, b_bot, y_rg, y_b0, y_bias_v);
            store_u8x32_avx2(token, &mut y_out[y_bot_off + px..y_bot_off + px + 32], yb_lo, yb_hi);

            // Cb/Cr: vertical avg_epu8 first (u8→u8 rounded), then horizontal
            // maddubs pair-sum (u8→u16, range [0, 510]). The PREC+1 = 16 shift
            // absorbs the ×2 from the pair sum. This matches the yuv crate's
            // exact 4:2:0 path.
            let r_avg = _mm256_avg_epu8(r_top, r_bot);
            let g_avg = _mm256_avg_epu8(g_top, g_bot);
            let b_avg = _mm256_avg_epu8(b_top, b_bot);
            let r_sum = _mm256_maddubs_epi16(r_avg, all_ones);
            let g_sum = _mm256_maddubs_epi16(g_avg, all_ones);
            let b_sum = _mm256_maddubs_epi16(b_avg, all_ones);

            // Interleave RG sums and B+zero for pmaddwd, then matrix multiply.
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
                <&mut [u8; 16]>::try_from(&mut cb_out[cb_row_off + cx * 16..cb_row_off + cx * 16 + 16]).unwrap(),
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
                <&mut [u8; 16]>::try_from(&mut cr_out[cb_row_off + cx * 16..cb_row_off + cx * 16 + 16]).unwrap(),
                _mm256_castsi256_si128(cr_u8),
            );
        }
    }

    // Scalar tail: remaining columns, odd last row, etc.
    rgb_to_yuv420_scalar_tail(rgb, y_out, cb_out, cr_out, width, height, cw, col_blocks * 32);
}

/// Scalar fallback for columns/rows not covered by the 32-column SIMD blocks.
#[cfg(target_arch = "x86_64")]
fn rgb_to_yuv420_scalar_tail(
    rgb: &[u8],
    y_out: &mut [u8],
    cb_out: &mut [u8],
    cr_out: &mut [u8],
    width: usize,
    height: usize,
    cw: usize,
    simd_cols: usize,
) {
    let row_stride = width * 3;

    // Y for columns simd_cols..width (all rows).
    for row in 0..height {
        for col in simd_cols..width {
            let p = row * row_stride + col * 3;
            let r = rgb[p] as f32;
            let g = rgb[p + 1] as f32;
            let b = rgb[p + 2] as f32;
            y_out[row * width + col] = clamp_round(YR * r + YG * g + YB * b);
        }
    }

    // Cb/Cr for all chroma columns that weren't fully handled by SIMD.
    let simd_cx = simd_cols / 2;
    let mut cy = 0usize;
    let mut row = 0usize;
    while row < height {
        let row1 = (row + 1).min(height - 1);
        let mut cx = simd_cx;
        let mut col = simd_cols;
        while col < width {
            let col1 = (col + 1).min(width - 1);
            let i00 = row * row_stride + col * 3;
            let i01 = row * row_stride + col1 * 3;
            let i10 = row1 * row_stride + col * 3;
            let i11 = row1 * row_stride + col1 * 3;
            let r = (rgb[i00] as u32 + rgb[i01] as u32 + rgb[i10] as u32 + rgb[i11] as u32) as f32
                * 0.25;
            let g = (rgb[i00 + 1] as u32
                + rgb[i01 + 1] as u32
                + rgb[i10 + 1] as u32
                + rgb[i11 + 1] as u32) as f32
                * 0.25;
            let b = (rgb[i00 + 2] as u32
                + rgb[i01 + 2] as u32
                + rgb[i10 + 2] as u32
                + rgb[i11 + 2] as u32) as f32
                * 0.25;
            cb_out[cy * cw + cx] = clamp_round(CB_R * r + CB_G * g + CB_B * b + CHROMA_BIAS);
            cr_out[cy * cw + cx] = clamp_round(CR_R * r + CR_G * g + CR_B * b + CHROMA_BIAS);
            cx += 1;
            col += 2;
        }
        cy += 1;
        row += 2;
    }

    // Odd last row: Y was handled above if simd_cols < width, but for the
    // SIMD-covered Y columns on the last odd row, we still need them.
    if height % 2 == 1 {
        let last_row = height - 1;
        for col in 0..simd_cols.min(width) {
            let p = last_row * row_stride + col * 3;
            let r = rgb[p] as f32;
            let g = rgb[p + 1] as f32;
            let b = rgb[p + 2] as f32;
            y_out[last_row * width + col] = clamp_round(YR * r + YG * g + YB * b);
        }
        // Cb/Cr for the last odd chroma row (SIMD columns).
        let cy = height / 2;
        for cx in 0..simd_cx {
            let col = cx * 2;
            let col1 = (col + 1).min(width - 1);
            let i00 = last_row * row_stride + col * 3;
            let i01 = last_row * row_stride + col1 * 3;
            let r = (rgb[i00] as u32 + rgb[i01] as u32 + rgb[i00] as u32 + rgb[i01] as u32) as f32
                * 0.25;
            let g = (rgb[i00 + 1] as u32
                + rgb[i01 + 1] as u32
                + rgb[i00 + 1] as u32
                + rgb[i01 + 1] as u32) as f32
                * 0.25;
            let b = (rgb[i00 + 2] as u32
                + rgb[i01 + 2] as u32
                + rgb[i00 + 2] as u32
                + rgb[i01 + 2] as u32) as f32
                * 0.25;
            cb_out[cy * cw + cx] = clamp_round(CB_R * r + CB_G * g + CB_B * b + CHROMA_BIAS);
            cr_out[cy * cw + cx] = clamp_round(CR_R * r + CR_G * g + CR_B * b + CHROMA_BIAS);
        }
    }
}

// ── existing magetypes-generic fallbacks follow ─────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pattern(width: usize, height: usize) -> Vec<u8> {
        let mut rgb = vec![0u8; width * height * 3];
        for y in 0..height {
            for x in 0..width {
                let i = (y * width + x) * 3;
                // Mix of ramps, stripes, and saturated colors to exercise all
                // coefficient signs and clamp boundaries.
                rgb[i] = ((x * 7 + y * 3) & 0xff) as u8;
                rgb[i + 1] = ((x * 3 ^ y * 11) & 0xff) as u8;
                rgb[i + 2] = (((x + y) * 5) & 0xff) as u8;
            }
        }
        rgb
    }

    fn mean_abs_err(a: &[u8], b: &[u8]) -> f64 {
        let mut s = 0u64;
        for (x, y) in a.iter().zip(b.iter()) {
            s += x.abs_diff(*y) as u64;
        }
        s as f64 / a.len() as f64
    }

    fn max_abs_err(a: &[u8], b: &[u8]) -> u32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| x.abs_diff(*y) as u32)
            .max()
            .unwrap_or(0)
    }

    #[test]
    fn yuv444_matches_yuv_crate() {
        let (w, h) = (123, 45);
        let rgb = make_pattern(w, h);
        let n = w * h;

        let mut y = vec![0u8; n];
        let mut cb = vec![0u8; n];
        let mut cr = vec![0u8; n];
        rgb_to_yuv444(&rgb, &mut y, &mut cb, &mut cr, w, h);

        let mut ref_img = yuv::YuvPlanarImageMut::alloc(
            w as u32,
            h as u32,
            yuv::YuvChromaSubsampling::Yuv444,
        );
        yuv::rgb_to_yuv444(
            &mut ref_img,
            &rgb,
            (w * 3) as u32,
            yuv::YuvRange::Full,
            yuv::YuvStandardMatrix::Bt601,
            yuv::YuvConversionMode::Professional,
        )
        .unwrap();

        let ry = ref_img.y_plane.borrow();
        let ru = ref_img.u_plane.borrow();
        let rv = ref_img.v_plane.borrow();

        // Expect near-byte-identical output: the yuv crate uses 15-bit
        // fixed-point, we use f32 FMA — disagreements come from the last bit
        // of rounding only.
        assert!(max_abs_err(&y, ry) <= 1, "Y max err > 1");
        assert!(max_abs_err(&cb, ru) <= 1, "Cb max err > 1");
        assert!(max_abs_err(&cr, rv) <= 1, "Cr max err > 1");
        assert!(mean_abs_err(&y, ry) < 0.05);
        assert!(mean_abs_err(&cb, ru) < 0.05);
        assert!(mean_abs_err(&cr, rv) < 0.05);
    }

    #[test]
    fn yuv420_matches_yuv_crate() {
        let (w, h) = (124, 46);
        let rgb = make_pattern(w, h);
        let cw = w.div_ceil(2);
        let ch = h.div_ceil(2);

        let mut y = vec![0u8; w * h];
        let mut cb = vec![0u8; cw * ch];
        let mut cr = vec![0u8; cw * ch];
        rgb_to_yuv420(&rgb, &mut y, &mut cb, &mut cr, w, h);

        let mut ref_img = yuv::YuvPlanarImageMut::alloc(
            w as u32,
            h as u32,
            yuv::YuvChromaSubsampling::Yuv420,
        );
        yuv::rgb_to_yuv420(
            &mut ref_img,
            &rgb,
            (w * 3) as u32,
            yuv::YuvRange::Full,
            yuv::YuvStandardMatrix::Bt601,
            yuv::YuvConversionMode::Professional,
        )
        .unwrap();

        let ry = ref_img.y_plane.borrow();
        let ru = ref_img.u_plane.borrow();
        let rv = ref_img.v_plane.borrow();

        let y_max = max_abs_err(&y, ry);
        let cb_max = max_abs_err(&cb, ru);
        let cr_max = max_abs_err(&cr, rv);
        let cb_mean = mean_abs_err(&cb, ru);
        let cr_mean = mean_abs_err(&cr, rv);
        eprintln!(
            "420 parity: Y max={y_max} Cb max={cb_max} mean={cb_mean:.4} Cr max={cr_max} mean={cr_mean:.4}"
        );
        assert!(y_max <= 1, "Y max err {y_max} > 1");
        // Chroma: the fused kernel sums u8 R/G/B via maddubs then applies the
        // matrix at PREC+1. Different rounding path from averaging u8 Cb/Cr
        // after the matrix. Tolerate 3 levels (invisible after JPEG quant).
        assert!(cb_max <= 3, "Cb max err {cb_max} > 3");
        assert!(cr_max <= 3, "Cr max err {cr_max} > 3");
        assert!(cb_mean < 0.3, "Cb mean err {cb_mean} > 0.3");
        assert!(cr_mean < 0.3, "Cr mean err {cr_mean} > 0.3");
    }

    #[test]
    fn clamp_boundaries_white_and_black() {
        let rgb: Vec<u8> = [[0u8, 0, 0], [255, 255, 255]]
            .iter()
            .flatten()
            .copied()
            .collect();
        let mut y = [0u8; 2];
        let mut cb = [0u8; 2];
        let mut cr = [0u8; 2];
        rgb_to_yuv444(&rgb, &mut y, &mut cb, &mut cr, 2, 1);
        assert_eq!(y, [0, 255]);
        assert_eq!(cb, [128, 128]);
        assert_eq!(cr, [128, 128]);
    }
}
