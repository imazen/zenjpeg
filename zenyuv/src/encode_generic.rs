//! Magetypes-generic encode kernels (f32x8 FMA).
//!
//! These run on all platforms: NEON, WASM SIMD128, AVX2 (fallback), and scalar.

use archmage::prelude::*;
use crate::types::ForwardCoeffs;
use magetypes::simd::generic::f32x8 as GenericF32x8;
use magetypes::simd::generic::i32x8 as GenericI32x8;

/// 4:4:4 kernel. Processes pixels in groups of 8 via `f32x8`, with a scalar
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
    type f32x8 = GenericF32x8<Token>;
    #[allow(non_camel_case_types)]
    type i32x8 = GenericI32x8<Token>;

    let yr = f32x8::splat(token, coeffs.yr_f);
    let yg = f32x8::splat(token, coeffs.yg_f);
    let yb = f32x8::splat(token, coeffs.yb_f);
    let cb_r = f32x8::splat(token, coeffs.cb_r_f);
    let cb_g = f32x8::splat(token, coeffs.cb_g_f);
    let cb_b = f32x8::splat(token, coeffs.cb_b_f);
    let cr_r = f32x8::splat(token, coeffs.cr_r_f);
    let cr_g = f32x8::splat(token, coeffs.cr_g_f);
    let cr_b = f32x8::splat(token, coeffs.cr_b_f);
    let y_bias = f32x8::splat(token, coeffs.y_bias_f);
    let uv_bias = f32x8::splat(token, coeffs.uv_bias_f);
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

        let y_f = r.mul_add(yr, g.mul_add(yg, b.mul_add(yb, y_bias)));
        let cb_f = r.mul_add(cb_r, g.mul_add(cb_g, b.mul_add(cb_b, uv_bias)));
        let cr_f = r.mul_add(cr_r, g.mul_add(cr_g, b.mul_add(cr_b, uv_bias)));

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
        y[i] = crate::clamp_round(coeffs.yr_f * r + coeffs.yg_f * g + coeffs.yb_f * b + coeffs.y_bias_f);
        cb[i] = crate::clamp_round(coeffs.cb_r_f * r + coeffs.cb_g_f * g + coeffs.cb_b_f * b + coeffs.uv_bias_f);
        cr[i] = crate::clamp_round(coeffs.cr_r_f * r + coeffs.cr_g_f * g + coeffs.cr_b_f * b + coeffs.uv_bias_f);
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
    type f32x8 = GenericF32x8<Token>;
    #[allow(non_camel_case_types)]
    type i32x8 = GenericI32x8<Token>;

    let cw = width.div_ceil(2);
    let n = width * height;

    // Y plane: full-resolution.
    let yr_v = f32x8::splat(token, coeffs.yr_f);
    let yg_v = f32x8::splat(token, coeffs.yg_f);
    let yb_v = f32x8::splat(token, coeffs.yb_f);
    let y_bias_v = f32x8::splat(token, coeffs.y_bias_f);
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
        let y_f = r.mul_add(yr_v, g.mul_add(yg_v, b.mul_add(yb_v, y_bias_v)));
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
        y[i] = crate::clamp_round(coeffs.yr_f * r + coeffs.yg_f * g + coeffs.yb_f * b + coeffs.y_bias_f);
    }

    // Chroma: iterate 2x2 blocks.
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
            let r =
                (rgb[i00] as u32 + rgb[i01] as u32 + rgb[i10] as u32 + rgb[i11] as u32) as f32
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

            cb[cy * cw + cx] = crate::clamp_round(
                coeffs.cb_r_f * r + coeffs.cb_g_f * g + coeffs.cb_b_f * b + coeffs.uv_bias_f,
            );
            cr[cy * cw + cx] = crate::clamp_round(
                coeffs.cr_r_f * r + coeffs.cr_g_f * g + coeffs.cr_b_f * b + coeffs.uv_bias_f,
            );

            cx += 1;
            col += 2;
        }
        cy += 1;
        row += 2;
    }
}
