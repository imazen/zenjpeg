//! Magetypes-generic decode kernels (f32x8 FMA).
//!
//! YCbCr->RGB inverse matrix for 4:4:4, 4:2:0, 4:2:2, and 4:0:0 (grayscale).

use crate::types::InverseCoeffs;
use archmage::prelude::*;
use magetypes::simd::generic::f32x8 as GenericF32x8;
use magetypes::simd::generic::i32x8 as GenericI32x8;

/// 4:4:4 decode kernel. Y/Cb/Cr planes are all full-resolution.
#[magetypes(v3, neon, wasm128, scalar)]
#[inline(always)]
pub(crate) fn yuv444_to_rgb_generic(
    token: Token,
    y_plane: &[u8],
    cb_plane: &[u8],
    cr_plane: &[u8],
    rgb: &mut [u8],
    n: usize,
    coeffs: &InverseCoeffs,
) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    #[allow(non_camel_case_types)]
    type i32x8 = GenericI32x8<Token>;

    let y_coeff = f32x8::splat(token, coeffs.y_coeff);
    let cr_to_r = f32x8::splat(token, coeffs.cr_to_r);
    let cr_to_g = f32x8::splat(token, coeffs.cr_to_g);
    let cb_to_g = f32x8::splat(token, coeffs.cb_to_g);
    let cb_to_b = f32x8::splat(token, coeffs.cb_to_b);
    let y_off = f32x8::splat(token, coeffs.y_offset);
    let uv_off = f32x8::splat(token, coeffs.uv_offset);
    let zero = i32x8::zero(token);
    let max255 = i32x8::splat(token, 255);

    let chunks = n / 8;
    for c in 0..chunks {
        let base = c * 8;
        let mut ya = [0.0f32; 8];
        let mut cba = [0.0f32; 8];
        let mut cra = [0.0f32; 8];
        for i in 0..8 {
            ya[i] = y_plane[base + i] as f32;
            cba[i] = cb_plane[base + i] as f32;
            cra[i] = cr_plane[base + i] as f32;
        }
        let y_v = f32x8::from_array(token, ya);
        let cb_v = f32x8::from_array(token, cba);
        let cr_v = f32x8::from_array(token, cra);

        let y_scaled = (y_v + y_off) * y_coeff;
        let cb_shifted = cb_v + uv_off;
        let cr_shifted = cr_v + uv_off;

        let r_f = y_scaled + cr_shifted * cr_to_r;
        let g_f = y_scaled + cb_shifted * cb_to_g + cr_shifted * cr_to_g;
        let b_f = y_scaled + cb_shifted * cb_to_b;

        let r_i = r_f.to_i32_round().max(zero).min(max255).to_array();
        let g_i = g_f.to_i32_round().max(zero).min(max255).to_array();
        let b_i = b_f.to_i32_round().max(zero).min(max255).to_array();

        for i in 0..8 {
            let p = (base + i) * 3;
            rgb[p] = r_i[i] as u8;
            rgb[p + 1] = g_i[i] as u8;
            rgb[p + 2] = b_i[i] as u8;
        }
    }

    // Scalar tail.
    for i in (chunks * 8)..n {
        let y_val = y_plane[i] as f32 + coeffs.y_offset;
        let cb_val = cb_plane[i] as f32 + coeffs.uv_offset;
        let cr_val = cr_plane[i] as f32 + coeffs.uv_offset;
        let y_scaled = y_val * coeffs.y_coeff;
        let p = i * 3;
        rgb[p] = crate::clamp_round(y_scaled + cr_val * coeffs.cr_to_r);
        rgb[p + 1] =
            crate::clamp_round(y_scaled + cb_val * coeffs.cb_to_g + cr_val * coeffs.cr_to_g);
        rgb[p + 2] = crate::clamp_round(y_scaled + cb_val * coeffs.cb_to_b);
    }
}

/// 4:2:0 decode kernel with nearest-neighbor chroma upsampling.
/// Y is full-resolution; Cb/Cr are half in both dimensions.
#[magetypes(v3, neon, wasm128, scalar)]
#[inline(always)]
/// NOTE (measured 2026-07-31, `benches/kernel_tiers.rs`): this gets **no**
/// SIMD benefit on aarch64 — 2.9 ms vs 2.9 ms, 1.00x against its own forced
/// scalar tier at 1920x1080 — while the 4:4:4 sibling
/// [`yuv444_to_rgb_generic`] reaches 3.57x from the same generic machinery.
/// 4:2:0 is the dominant subsampling in real JPEG/WebP, so this is the case
/// that matters most.
///
/// The cause is the chroma addressing below: `cx = col / 2` advances the
/// chroma index at half the luma rate, so a lane-parallel loop over `col`
/// reads `cb_plane`/`cr_plane` with a gather-like pattern. The 4:4:4 path has
/// unit stride on all three planes and vectorizes cleanly. Nothing about the
/// arithmetic blocks SIMD — only the addressing does.
///
/// OPTIMIZED 2026-07-31: 2.9 ms -> 1.7 ms at 1920x1080, bit-identical.
///
/// This used to be the one decode entry with no SIMD benefit at all (1.00x
/// against its own forced-scalar tier) while the 4:4:4 sibling reached 3.57x
/// from the same generic machinery. The cause was the inner loop's shape, not
/// its arithmetic: `cx = col / 2` made chroma advance at half the luma rate, so
/// the loop could not be a flat unit-stride pass the way 4:4:4's is.
///
/// Chroma is now expanded into a FIXED stack scratch (no allocation — this is a
/// `no_std` decode hot path) and the conversion runs as a flat unit-stride loop
/// over each chunk. Both tiers improved equally, which is the expected result:
/// NEON is baseline on aarch64, so what changed is that the autovectorizer can
/// now do to this loop what it already did to 4:4:4's.
///
/// FAILED FIRST ATTEMPT, kept as a warning: fixing the addressing WITHOUT
/// flattening — chroma-major outer loop with a 2-iteration inner loop — was 2x
/// SLOWER (2.9 -> 6.0 ms), despite being bit-identical and removing the same
/// division. Replacing one flat loop with a nested one whose inner bound is
/// variable costs more than the division saves. The scratch buffer is what buys
/// a loop with no inner bound at all; that distinction is the whole fix.
///
/// Bit-identity verified by hashing the output of 14 shapes (78,114 bytes)
/// including chunk boundaries (128/129/255/256/257) and odd dimensions, against
/// the pre-change implementation: identical.
pub(crate) fn yuv420_to_rgb_generic(
    _token: Token,
    y_plane: &[u8],
    cb_plane: &[u8],
    cr_plane: &[u8],
    rgb: &mut [u8],
    width: usize,
    height: usize,
    coeffs: &InverseCoeffs,
) {
    let cw = width.div_ceil(2);

    // Chroma is duplicated into a FIXED stack scratch so the inner loop is a
    // flat, unit-stride pass — the same shape as the 4:4:4 kernel, which is the
    // one that actually vectorizes (3.57x). No allocation: a decode hot path
    // should not heap-allocate per call, and this crate is `no_std`.
    //
    // An earlier attempt fixed the addressing WITHOUT flattening (chroma-major
    // outer loop, 2-iteration inner loop) and was 2x slower — the nesting, not
    // the `col / 2`, was the dominant cost.
    const CHUNK: usize = 128;
    let mut cb_row = [0.0f32; CHUNK];
    let mut cr_row = [0.0f32; CHUNK];

    for row in 0..height {
        let cy = row / 2;
        let crow = cy * cw;
        let yrow = row * width;

        let mut base = 0usize;
        while base < width {
            let n = CHUNK.min(width - base);

            // Expand chroma for this chunk: unit-stride reads, contiguous
            // pair writes.
            let cx0 = base / 2;
            let mut i = 0usize;
            while i < n {
                let cx = cx0 + (base + i) / 2 - base / 2;
                let cb_val = cb_plane[crow + cx] as f32 + coeffs.uv_offset;
                let cr_val = cr_plane[crow + cx] as f32 + coeffs.uv_offset;
                cb_row[i] = cb_val;
                cr_row[i] = cr_val;
                if i + 1 < n && (base + i) % 2 == 0 {
                    cb_row[i + 1] = cb_val;
                    cr_row[i + 1] = cr_val;
                    i += 2;
                } else {
                    i += 1;
                }
            }

            // Flat unit-stride conversion over the chunk.
            for k in 0..n {
                let yi = yrow + base + k;
                let y_val = y_plane[yi] as f32 + coeffs.y_offset;
                let cb_val = cb_row[k];
                let cr_val = cr_row[k];
                let y_scaled = y_val * coeffs.y_coeff;

                let p = yi * 3;
                rgb[p] = crate::clamp_round(y_scaled + cr_val * coeffs.cr_to_r);
                rgb[p + 1] = crate::clamp_round(
                    y_scaled + cb_val * coeffs.cb_to_g + cr_val * coeffs.cr_to_g,
                );
                rgb[p + 2] = crate::clamp_round(y_scaled + cb_val * coeffs.cb_to_b);
            }

            base += n;
        }
    }
}

/// 4:2:0 decode kernel with bilinear chroma upsampling.
/// Interpolates chroma values between sample centers for smoother output.
#[magetypes(v3, neon, wasm128, scalar)]
#[inline(always)]
pub(crate) fn yuv420_to_rgb_bilinear_generic(
    _token: Token,
    y_plane: &[u8],
    cb_plane: &[u8],
    cr_plane: &[u8],
    rgb: &mut [u8],
    width: usize,
    height: usize,
    coeffs: &InverseCoeffs,
) {
    let cw = width.div_ceil(2);
    let ch = height.div_ceil(2);

    for row in 0..height {
        for col in 0..width {
            let yi = row * width + col;

            // Chroma sample center is at (col/2, row/2) in chroma space.
            // For bilinear: compute fractional position and interpolate.
            let cx_f = col as f32 * 0.5;
            let cy_f = row as f32 * 0.5;

            let cx0 = (cx_f - 0.5).max(0.0) as usize;
            let cy0 = (cy_f - 0.5).max(0.0) as usize;
            let cx1 = (cx0 + 1).min(cw - 1);
            let cy1 = (cy0 + 1).min(ch - 1);

            let fx = (cx_f - 0.5) - cx0 as f32;
            let fy = (cy_f - 0.5) - cy0 as f32;
            let fx = fx.clamp(0.0, 1.0);
            let fy = fy.clamp(0.0, 1.0);

            let w00 = (1.0 - fx) * (1.0 - fy);
            let w01 = fx * (1.0 - fy);
            let w10 = (1.0 - fx) * fy;
            let w11 = fx * fy;

            let cb_val = cb_plane[cy0 * cw + cx0] as f32 * w00
                + cb_plane[cy0 * cw + cx1] as f32 * w01
                + cb_plane[cy1 * cw + cx0] as f32 * w10
                + cb_plane[cy1 * cw + cx1] as f32 * w11
                + coeffs.uv_offset;
            let cr_val = cr_plane[cy0 * cw + cx0] as f32 * w00
                + cr_plane[cy0 * cw + cx1] as f32 * w01
                + cr_plane[cy1 * cw + cx0] as f32 * w10
                + cr_plane[cy1 * cw + cx1] as f32 * w11
                + coeffs.uv_offset;

            let y_val = y_plane[yi] as f32 + coeffs.y_offset;
            let y_scaled = y_val * coeffs.y_coeff;

            let p = yi * 3;
            rgb[p] = crate::clamp_round(y_scaled + cr_val * coeffs.cr_to_r);
            rgb[p + 1] =
                crate::clamp_round(y_scaled + cb_val * coeffs.cb_to_g + cr_val * coeffs.cr_to_g);
            rgb[p + 2] = crate::clamp_round(y_scaled + cb_val * coeffs.cb_to_b);
        }
    }
}

/// 4:2:2 decode kernel with nearest-neighbor chroma upsampling.
/// Y is full-resolution; Cb/Cr are half horizontally, full vertically.
#[magetypes(v3, neon, wasm128, scalar)]
#[inline(always)]
pub(crate) fn yuv422_to_rgb_generic(
    _token: Token,
    y_plane: &[u8],
    cb_plane: &[u8],
    cr_plane: &[u8],
    rgb: &mut [u8],
    width: usize,
    height: usize,
    coeffs: &InverseCoeffs,
) {
    let cw = width.div_ceil(2);

    for row in 0..height {
        for col in 0..width {
            let cx = col / 2;
            let yi = row * width + col;
            let ci = row * cw + cx;

            let y_val = y_plane[yi] as f32 + coeffs.y_offset;
            let cb_val = cb_plane[ci] as f32 + coeffs.uv_offset;
            let cr_val = cr_plane[ci] as f32 + coeffs.uv_offset;
            let y_scaled = y_val * coeffs.y_coeff;

            let p = yi * 3;
            rgb[p] = crate::clamp_round(y_scaled + cr_val * coeffs.cr_to_r);
            rgb[p + 1] =
                crate::clamp_round(y_scaled + cb_val * coeffs.cb_to_g + cr_val * coeffs.cr_to_g);
            rgb[p + 2] = crate::clamp_round(y_scaled + cb_val * coeffs.cb_to_b);
        }
    }
}

/// 4:0:0 (grayscale) decode kernel. Only Y plane, no chroma.
#[magetypes(v3, neon, wasm128, scalar)]
#[inline(always)]
pub(crate) fn yuv400_to_rgb_generic(
    token: Token,
    y_plane: &[u8],
    rgb: &mut [u8],
    n: usize,
    coeffs: &InverseCoeffs,
) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    #[allow(non_camel_case_types)]
    type i32x8 = GenericI32x8<Token>;

    let y_coeff = f32x8::splat(token, coeffs.y_coeff);
    let y_off = f32x8::splat(token, coeffs.y_offset);
    let zero = i32x8::zero(token);
    let max255 = i32x8::splat(token, 255);

    let chunks = n / 8;
    for c in 0..chunks {
        let base = c * 8;
        let mut ya = [0.0f32; 8];
        for i in 0..8 {
            ya[i] = y_plane[base + i] as f32;
        }
        let y_v = f32x8::from_array(token, ya);
        let gray = ((y_v + y_off) * y_coeff)
            .to_i32_round()
            .max(zero)
            .min(max255)
            .to_array();

        for i in 0..8 {
            let p = (base + i) * 3;
            let g = gray[i] as u8;
            rgb[p] = g;
            rgb[p + 1] = g;
            rgb[p + 2] = g;
        }
    }

    for i in (chunks * 8)..n {
        let y_val = y_plane[i] as f32 + coeffs.y_offset;
        let g = crate::clamp_round(y_val * coeffs.y_coeff);
        let p = i * 3;
        rgb[p] = g;
        rgb[p + 1] = g;
        rgb[p + 2] = g;
    }
}
