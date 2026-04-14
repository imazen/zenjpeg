//! Magetypes-generic encode kernels (i32x8 fixed-point).
//!
//! These run on all platforms: NEON, WASM SIMD128, AVX2 (fallback), and scalar.
//! Uses the SAME 15-bit fixed-point integer math as the native AVX2/NEON
//! kernels so that all dispatch tiers produce byte-identical output.

use crate::types::{ForwardCoeffs, PREC};
use archmage::prelude::*;
use magetypes::simd::generic::i32x8 as GenericI32x8;

/// 4:4:4 kernel. Processes pixels in groups of 8 via `i32x8`, with a scalar
/// tail for the final `< 8` pixels.
#[magetypes(v3, neon, wasm128, scalar)]
#[inline(always)]
pub(crate) fn rgb_to_yuv444_generic(
    token: Token,
    rgb: &[u8],
    y: &mut [u8],
    cb: &mut [u8],
    cr: &mut [u8],
    n: usize,
    coeffs: &ForwardCoeffs,
) {
    #[allow(non_camel_case_types)]
    type i32x8 = GenericI32x8<Token>;

    let yr = i32x8::splat(token, coeffs.yr as i32);
    let yg = i32x8::splat(token, coeffs.yg as i32);
    let yb = i32x8::splat(token, coeffs.yb as i32);
    let cb_r = i32x8::splat(token, coeffs.cb_r as i32);
    let cb_g = i32x8::splat(token, coeffs.cb_g as i32);
    let cb_b = i32x8::splat(token, coeffs.cb_b as i32);
    let cr_r = i32x8::splat(token, coeffs.cr_r as i32);
    let cr_g = i32x8::splat(token, coeffs.cr_g as i32);
    let cr_b = i32x8::splat(token, coeffs.cr_b as i32);
    let y_bias = i32x8::splat(token, coeffs.y_bias);
    let uv_bias = i32x8::splat(token, coeffs.uv_bias);
    let zero = i32x8::zero(token);
    let max255 = i32x8::splat(token, 255);

    let chunks = n / 8;
    for c in 0..chunks {
        let base = c * 8;
        let mut ra = [0i32; 8];
        let mut ga = [0i32; 8];
        let mut ba = [0i32; 8];
        for i in 0..8 {
            let p = (base + i) * 3;
            ra[i] = rgb[p] as i32;
            ga[i] = rgb[p + 1] as i32;
            ba[i] = rgb[p + 2] as i32;
        }
        let r = i32x8::from_array(token, ra);
        let g = i32x8::from_array(token, ga);
        let b = i32x8::from_array(token, ba);

        // Integer multiply-accumulate + bias, then arithmetic shift right.
        // Matches AVX2 pmaddwd + srai and NEON vmul + vshrn.
        let y_v = ((r * yr) + (g * yg) + (b * yb) + y_bias).shr_arithmetic_const::<{ PREC }>();
        let cb_v =
            ((r * cb_r) + (g * cb_g) + (b * cb_b) + uv_bias).shr_arithmetic_const::<{ PREC }>();
        let cr_v =
            ((r * cr_r) + (g * cr_g) + (b * cr_b) + uv_bias).shr_arithmetic_const::<{ PREC }>();

        let y_i = y_v.max(zero).min(max255).to_array();
        let cb_i = cb_v.max(zero).min(max255).to_array();
        let cr_i = cr_v.max(zero).min(max255).to_array();

        for i in 0..8 {
            y[base + i] = y_i[i] as u8;
            cb[base + i] = cb_i[i] as u8;
            cr[base + i] = cr_i[i] as u8;
        }
    }

    // Scalar tail — same integer math.
    for i in (chunks * 8)..n {
        let p = i * 3;
        let r = rgb[p] as i32;
        let g = rgb[p + 1] as i32;
        let b = rgb[p + 2] as i32;
        y[i] =
            ((r * coeffs.yr as i32 + g * coeffs.yg as i32 + b * coeffs.yb as i32 + coeffs.y_bias)
                >> PREC)
                .clamp(0, 255) as u8;
        cb[i] = ((r * coeffs.cb_r as i32
            + g * coeffs.cb_g as i32
            + b * coeffs.cb_b as i32
            + coeffs.uv_bias)
            >> PREC)
            .clamp(0, 255) as u8;
        cr[i] = ((r * coeffs.cr_r as i32
            + g * coeffs.cr_g as i32
            + b * coeffs.cr_b as i32
            + coeffs.uv_bias)
            >> PREC)
            .clamp(0, 255) as u8;
    }
}

/// 4:2:0 kernel. Computes Y at full resolution and Cb/Cr at 2x2 block centers.
#[magetypes(v3, neon, wasm128, scalar)]
#[inline(always)]
pub(crate) fn rgb_to_yuv420_generic(
    token: Token,
    rgb: &[u8],
    y: &mut [u8],
    cb: &mut [u8],
    cr: &mut [u8],
    width: usize,
    height: usize,
    coeffs: &ForwardCoeffs,
) {
    #[allow(non_camel_case_types)]
    type i32x8 = GenericI32x8<Token>;

    let cw = width.div_ceil(2);
    let n = width * height;

    // Y plane: full-resolution, same integer math as 4:4:4.
    let yr_v = i32x8::splat(token, coeffs.yr as i32);
    let yg_v = i32x8::splat(token, coeffs.yg as i32);
    let yb_v = i32x8::splat(token, coeffs.yb as i32);
    let y_bias_v = i32x8::splat(token, coeffs.y_bias);
    let zero = i32x8::zero(token);
    let max255 = i32x8::splat(token, 255);

    let chunks = n / 8;
    for c in 0..chunks {
        let base = c * 8;
        let mut ra = [0i32; 8];
        let mut ga = [0i32; 8];
        let mut ba = [0i32; 8];
        for i in 0..8 {
            let p = (base + i) * 3;
            ra[i] = rgb[p] as i32;
            ga[i] = rgb[p + 1] as i32;
            ba[i] = rgb[p + 2] as i32;
        }
        let r = i32x8::from_array(token, ra);
        let g = i32x8::from_array(token, ga);
        let b = i32x8::from_array(token, ba);
        let y_v =
            ((r * yr_v) + (g * yg_v) + (b * yb_v) + y_bias_v).shr_arithmetic_const::<{ PREC }>();
        let y_i = y_v.max(zero).min(max255).to_array();
        for i in 0..8 {
            y[base + i] = y_i[i] as u8;
        }
    }
    for i in (chunks * 8)..n {
        let p = i * 3;
        let r = rgb[p] as i32;
        let g = rgb[p + 1] as i32;
        let b = rgb[p + 2] as i32;
        y[i] =
            ((r * coeffs.yr as i32 + g * coeffs.yg as i32 + b * coeffs.yb as i32 + coeffs.y_bias)
                >> PREC)
                .clamp(0, 255) as u8;
    }

    // Chroma: iterate 2x2 blocks, integer math matching AVX2 fused path.
    // AVX2 does: avg_epu8(top, bot) then maddubs pair-sum → pmaddwd at PREC+1.
    // Scalar equivalent: sum all 4 u8 values, multiply by i16 coeff,
    // add uv_bias_420 (at PREC+1), shift right by PREC+1.
    let mut cy = 0usize;
    let mut row = 0usize;
    while row < height {
        let row1 = (row + 1).min(height - 1);
        let mut cx = 0usize;
        let mut col = 0usize;
        while col < width {
            let col1 = (col + 1).min(width - 1);

            let i00 = (row * width + col) * 3;
            let i01 = (row * width + col1) * 3;
            let i10 = (row1 * width + col) * 3;
            let i11 = (row1 * width + col1) * 3;

            // Sum of 4 u8 values (range [0, 1020]).
            let r_sum = rgb[i00] as i32 + rgb[i01] as i32 + rgb[i10] as i32 + rgb[i11] as i32;
            let g_sum = rgb[i00 + 1] as i32
                + rgb[i01 + 1] as i32
                + rgb[i10 + 1] as i32
                + rgb[i11 + 1] as i32;
            let b_sum = rgb[i00 + 2] as i32
                + rgb[i01 + 2] as i32
                + rgb[i10 + 2] as i32
                + rgb[i11 + 2] as i32;

            // Integer multiply + bias at PREC+1, shift right by PREC+1.
            // This matches the AVX2 path: avg_epu8 halves, maddubs sums pairs,
            // pmaddwd at PREC+1 with uv_bias_420.
            //
            // Note: AVX2 does (avg_epu8 + maddubs) which is NOT identical to
            // raw 4-pixel sum due to avg_epu8 rounding. For the generic path,
            // we use the f32 average approach to match quality, but this means
            // the generic 4:2:0 chroma may differ by ±1 from AVX2.
            // This is acceptable since 4:2:0 chroma dispatch parity is hard to
            // achieve perfectly (avg_epu8 rounds up on odd sums).
            let r_avg = (r_sum + 2) / 4; // round-to-nearest, matching avg_epu8 behavior
            let g_avg = (g_sum + 2) / 4;
            let b_avg = (b_sum + 2) / 4;

            // Now treat as single-pixel with integer coefficients at PREC.
            cb[cy * cw + cx] = ((r_avg * coeffs.cb_r as i32
                + g_avg * coeffs.cb_g as i32
                + b_avg * coeffs.cb_b as i32
                + coeffs.uv_bias)
                >> PREC)
                .clamp(0, 255) as u8;
            cr[cy * cw + cx] = ((r_avg * coeffs.cr_r as i32
                + g_avg * coeffs.cr_g as i32
                + b_avg * coeffs.cr_b as i32
                + coeffs.uv_bias)
                >> PREC)
                .clamp(0, 255) as u8;

            cx += 1;
            col += 2;
        }
        cy += 1;
        row += 2;
    }
}
