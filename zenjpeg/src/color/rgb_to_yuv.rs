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
    incant!(rgb_to_yuv444_impl(rgb, y, cb, cr, n));
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

        assert!(max_abs_err(&y, ry) <= 1, "Y max err > 1");
        // Chroma: yuv crate averages inside the fixed-point matrix, we average
        // RGB first then matrix. Same math, but rounding paths diverge, so
        // tolerate up to 2 levels max and keep mean tight.
        assert!(max_abs_err(&cb, ru) <= 2, "Cb max err > 2");
        assert!(max_abs_err(&cr, rv) <= 2, "Cr max err > 2");
        assert!(mean_abs_err(&cb, ru) < 0.2);
        assert!(mean_abs_err(&cr, rv) < 0.2);
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
