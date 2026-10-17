//! Row kernels: decoded component planes → interleaved 8-bit pixels.
//!
//! Every decode path that turns level-shifted integer-IDCT planes into pixels
//! (streaming baseline, fused and wave parallel, the fast integer coefficient
//! paths, the scanline strip reader) calls a [`RowKernel`] chosen once per
//! decode from the frame's colour model and the requested output layout. The
//! paths keep their own loop structure; only this call varies, and it varies
//! per row, never per pixel.
//!
//! The YCbCr arms dispatch to the SIMD kernels in [`super::ycbcr`]. The RGB
//! (identity) arms are clamp-and-interleave loops that `#[autoversion]` lets
//! LLVM vectorize per tier; BGR orders are the same kernels with the plane
//! order swapped.

use super::ycbcr::{
    fused_h2v2_box_ycbcr_to_rgb_u8, rgb_u8_swap_rb_inplace, ycbcr_planes_i16_to_rgb_u8,
    ycbcr_planes_i16_to_xrgba_u8,
};
use archmage::autoversion;

/// How three stored planes become R, G, B (or how one becomes gray).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PlaneMatrix {
    /// One luminance plane.
    Gray,
    /// BT.601 YCbCr. `turbo` selects libjpeg-turbo's 16-bit fixed point
    /// (`IdctMethod::Libjpeg`) over the default 14-bit set.
    YCbCr { turbo: bool },
    /// The planes are R, G, B already.
    Identity,
}

/// Bound on the i16 samples the row kernels and the upsamplers are exact for
/// on every SIMD tier, in the pixel domain (after the +128 level shift):
/// `[-SAMPLE_LIMIT, SAMPLE_LIMIT]`. The AVX2 triangle upsamplers sum four
/// times a sample in i16 and the YCbCr kernels centre chroma in i16 (c - 128);
/// libjpeg-turbo's constants stay inside i32 well past it. Clamped IDCT output
/// is [0, 255]; the unclamped IDCT of f32 output targets bounds its output to
/// this range (`decode::idct_int`).
pub(crate) const SAMPLE_LIMIT: i32 = 2047;

/// Interleaved 8-bit output layouts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Layout8 {
    /// One byte per pixel: gray, or the luma of colour planes.
    Gray,
    Rgb,
    Bgr,
    /// R, G, B, 255.
    Rgbx,
    /// B, G, R, 255.
    Bgrx,
}

impl Layout8 {
    pub(crate) const fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Gray => 1,
            Self::Rgb | Self::Bgr => 3,
            Self::Rgbx | Self::Bgrx => 4,
        }
    }

    /// The layout a `PixelFormat` writes, if it is an 8-bit interleaved one.
    pub(crate) fn from_pixel_format(format: crate::types::PixelFormat) -> Option<Self> {
        use crate::types::PixelFormat;
        match format {
            PixelFormat::Gray => Some(Self::Gray),
            PixelFormat::Rgb => Some(Self::Rgb),
            PixelFormat::Bgr => Some(Self::Bgr),
            PixelFormat::Rgba => Some(Self::Rgbx),
            PixelFormat::Bgra | PixelFormat::Bgrx => Some(Self::Bgrx),
            _ => None,
        }
    }
}

/// One colour model's conversion into one output layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RowKernel {
    matrix: PlaneMatrix,
    layout: Layout8,
}

/// Chunk (in output pixels) for the YCbCr 4bpp half-width path, which expands
/// chroma into stack scratch before the full-width kernel. A multiple of 16 so
/// the SIMD kernel's scalar tail lands on the same pixels as one whole-row call.
const H2_CHUNK: usize = 256;

impl RowKernel {
    pub(crate) const fn new(matrix: PlaneMatrix, layout: Layout8) -> Self {
        Self { matrix, layout }
    }

    pub(crate) const fn layout(self) -> Layout8 {
        self.layout
    }

    /// Whether the fused upsample+convert row kernels (`fused_h2v2_fancy_*`,
    /// `fused_h2v1_fancy_*`, `fused_h1v2_fancy_*`) serve this kernel, and if
    /// so whether each fused row then swaps R and B in place. They implement
    /// YCbCr into packed RGB; packed BGR is that plus the swap, done per row
    /// while it is still in cache (`Some(true)`). Every other model or layout
    /// goes through [`Self::row`] after an explicit upsample (`None`). The
    /// fused kernels are x86-64-v3 only.
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    pub(crate) const fn fused_ycbcr(self) -> Option<bool> {
        match (self.matrix, self.layout) {
            (PlaneMatrix::YCbCr { .. }, Layout8::Rgb) => Some(false),
            (PlaneMatrix::YCbCr { .. }, Layout8::Bgr) => Some(true),
            _ => None,
        }
    }

    /// Convert one row of full-resolution planes. The row width is
    /// `c0.len()`; `c1`/`c2` (ignored for [`PlaneMatrix::Gray`]) and `out`
    /// may be longer than needed.
    #[inline]
    pub(crate) fn row(self, c0: &[i16], c1: &[i16], c2: &[i16], out: &mut [u8]) {
        let n = c0.len();
        let out = &mut out[..n * self.layout.bytes_per_pixel()];
        // Every (matrix, layout) pair is named: a new model must not fall
        // into another model's arm.
        match (self.matrix, self.layout) {
            (PlaneMatrix::Gray, layout) => gray_row(c0, out, layout),
            (matrix @ (PlaneMatrix::YCbCr { .. } | PlaneMatrix::Identity), Layout8::Gray) => {
                color_to_gray_row(matrix, c0, &c1[..n], &c2[..n], out)
            }
            (PlaneMatrix::YCbCr { turbo }, Layout8::Rgb) => {
                ycbcr_planes_i16_to_rgb_u8(c0, &c1[..n], &c2[..n], out, turbo)
            }
            (PlaneMatrix::YCbCr { turbo }, Layout8::Bgr) => {
                ycbcr_planes_i16_to_rgb_u8(c0, &c1[..n], &c2[..n], out, turbo);
                rgb_u8_swap_rb_inplace(out);
            }
            (PlaneMatrix::YCbCr { turbo }, Layout8::Rgbx) => {
                ycbcr_planes_i16_to_xrgba_u8(c0, &c1[..n], &c2[..n], out, false, turbo)
            }
            (PlaneMatrix::YCbCr { turbo }, Layout8::Bgrx) => {
                ycbcr_planes_i16_to_xrgba_u8(c0, &c1[..n], &c2[..n], out, true, turbo)
            }
            (PlaneMatrix::Identity, Layout8::Rgb) => interleave3(c0, &c1[..n], &c2[..n], out),
            (PlaneMatrix::Identity, Layout8::Bgr) => interleave3(&c2[..n], &c1[..n], c0, out),
            (PlaneMatrix::Identity, Layout8::Rgbx) => interleave4(c0, &c1[..n], &c2[..n], out),
            (PlaneMatrix::Identity, Layout8::Bgrx) => interleave4(&c2[..n], &c1[..n], c0, out),
        }
    }

    /// Convert one row whose `c1`/`c2` planes have half the horizontal
    /// resolution of `c0`: chroma sample `i` covers pixels `2i` and `2i + 1`
    /// (nearest-neighbour / box upsampling, libjpeg's `h2v1_upsample`). The
    /// row width is `c0.len()`; the chroma slices need `c0.len().div_ceil(2)`
    /// samples.
    #[inline]
    pub(crate) fn row_h2(self, c0: &[i16], c1: &[i16], c2: &[i16], out: &mut [u8]) {
        let n = c0.len();
        let cn = n.div_ceil(2);
        let out = &mut out[..n * self.layout.bytes_per_pixel()];
        match (self.matrix, self.layout) {
            (PlaneMatrix::Gray, layout) => gray_row(c0, out, layout),
            (PlaneMatrix::YCbCr { turbo }, Layout8::Rgb) => {
                fused_h2v2_box_ycbcr_to_rgb_u8(c0, &c1[..cn], &c2[..cn], out, n, turbo)
            }
            (PlaneMatrix::YCbCr { turbo }, Layout8::Bgr) => {
                fused_h2v2_box_ycbcr_to_rgb_u8(c0, &c1[..cn], &c2[..cn], out, n, turbo);
                rgb_u8_swap_rb_inplace(out);
            }
            (PlaneMatrix::Identity, Layout8::Rgb) => {
                interleave3_h2_rgb(c0, &c1[..cn], &c2[..cn], out)
            }
            (PlaneMatrix::Identity, Layout8::Bgr) => {
                interleave3_h2_bgr(c0, &c1[..cn], &c2[..cn], out)
            }
            (PlaneMatrix::Identity, Layout8::Rgbx) => {
                interleave4_h2_rgb(c0, &c1[..cn], &c2[..cn], out)
            }
            (PlaneMatrix::Identity, Layout8::Bgrx) => {
                interleave4_h2_bgr(c0, &c1[..cn], &c2[..cn], out)
            }
            // YCbCr to 4bpp, and luma: replicate the chroma into stack scratch
            // one chunk at a time, then run the full-resolution kernel.
            (PlaneMatrix::YCbCr { .. }, Layout8::Rgbx | Layout8::Bgrx | Layout8::Gray)
            | (PlaneMatrix::Identity, Layout8::Gray) => {
                let (c1, c2) = (&c1[..cn], &c2[..cn]);
                let bpp = self.layout.bytes_per_pixel();
                let mut e1 = [0i16; H2_CHUNK];
                let mut e2 = [0i16; H2_CHUNK];
                for (start, c0_chunk) in (0..n).step_by(H2_CHUNK).zip(c0.chunks(H2_CHUNK)) {
                    let len = c0_chunk.len();
                    for (i, (a, b)) in e1[..len].iter_mut().zip(&mut e2[..len]).enumerate() {
                        let s = (start + i) / 2;
                        *a = c1[s];
                        *b = c2[s];
                    }
                    self.row(c0_chunk, &e1[..len], &e2[..len], &mut out[start * bpp..]);
                }
            }
        }
    }
}

fn gray_row(c0: &[i16], out: &mut [u8], layout: Layout8) {
    match layout {
        Layout8::Gray => clamp_row(c0, out),
        Layout8::Rgb | Layout8::Bgr => interleave3(c0, c0, c0, out),
        Layout8::Rgbx | Layout8::Bgrx => interleave4(c0, c0, c0, out),
    }
}

/// One byte of luma per pixel from colour planes: the Y plane itself for
/// YCbCr, [`rgb_luma_u8`] for RGB.
fn color_to_gray_row(matrix: PlaneMatrix, c0: &[i16], c1: &[i16], c2: &[i16], out: &mut [u8]) {
    match matrix {
        PlaneMatrix::Identity => {
            for (((o, &r), &g), &b) in out.iter_mut().zip(c0).zip(c1).zip(c2) {
                *o = rgb_luma_u8(clamp_u8(r), clamp_u8(g), clamp_u8(b));
            }
        }
        PlaneMatrix::YCbCr { .. } | PlaneMatrix::Gray => clamp_row(c0, out),
    }
}

#[inline(always)]
fn clamp_u8(v: i16) -> u8 {
    v.clamp(0, 255) as u8
}

#[autoversion]
fn clamp_row(src: &[i16], out: &mut [u8]) {
    for (o, &v) in out.iter_mut().zip(src) {
        *o = v.clamp(0, 255) as u8;
    }
}

#[autoversion]
fn interleave3(c0: &[i16], c1: &[i16], c2: &[i16], out: &mut [u8]) {
    for (((o, &a), &b), &c) in out
        .as_chunks_mut::<3>()
        .0
        .iter_mut()
        .zip(c0)
        .zip(c1)
        .zip(c2)
    {
        o[0] = a.clamp(0, 255) as u8;
        o[1] = b.clamp(0, 255) as u8;
        o[2] = c.clamp(0, 255) as u8;
    }
}

#[autoversion]
fn interleave4(c0: &[i16], c1: &[i16], c2: &[i16], out: &mut [u8]) {
    for (((o, &a), &b), &c) in out
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(c0)
        .zip(c1)
        .zip(c2)
    {
        o[0] = a.clamp(0, 255) as u8;
        o[1] = b.clamp(0, 255) as u8;
        o[2] = c.clamp(0, 255) as u8;
        o[3] = 255;
    }
}

/// Clamp-and-interleave with `c1`/`c2` at half horizontal resolution:
/// chroma sample `i` covers pixels `2i` and `2i + 1`. `$full` is the output
/// byte index of `c0` (0 for R-G-B order, 2 for B-G-R); `c1` always lands at
/// index 1 and `c2` at the remaining one. Generated per channel order and pixel
/// size so each loop stays branch-free for the vectorizer.
macro_rules! interleave_h2 {
    ($name:ident, $bpp:literal, $full:literal) => {
        #[autoversion]
        fn $name(c0: &[i16], c1: &[i16], c2: &[i16], out: &mut [u8]) {
            const C2: usize = 2 - $full;
            let pairs = c0.len() / 2;
            for ((o, a), (&b, &c)) in out
                .chunks_exact_mut(2 * $bpp)
                .zip(c0.chunks_exact(2))
                .zip(c1.iter().zip(c2))
                .take(pairs)
            {
                let (b, c) = (b.clamp(0, 255) as u8, c.clamp(0, 255) as u8);
                for (k, &v) in a.iter().enumerate() {
                    let px = &mut o[k * $bpp..k * $bpp + $bpp];
                    px[$full] = v.clamp(0, 255) as u8;
                    px[1] = b;
                    px[C2] = c;
                    if $bpp == 4 {
                        px[$bpp - 1] = 255;
                    }
                }
            }
            if c0.len() % 2 == 1 {
                let x = c0.len() - 1;
                let px = &mut out[x * $bpp..x * $bpp + $bpp];
                px[$full] = c0[x].clamp(0, 255) as u8;
                px[1] = c1[pairs].clamp(0, 255) as u8;
                px[C2] = c2[pairs].clamp(0, 255) as u8;
                if $bpp == 4 {
                    px[$bpp - 1] = 255;
                }
            }
        }
    };
}
interleave_h2!(interleave3_h2_rgb, 3, 0);
interleave_h2!(interleave3_h2_bgr, 3, 2);
interleave_h2!(interleave4_h2_rgb, 4, 0);
interleave_h2!(interleave4_h2_bgr, 4, 2);

/// One row of decoded 8-bit RGB (three bytes per pixel) into `layout`.
pub(crate) fn rgb8_row(src: &[u8], layout: Layout8, out: &mut [u8]) {
    let n = src.len() / 3;
    let out = &mut out[..n * layout.bytes_per_pixel()];
    match layout {
        Layout8::Rgb => out.copy_from_slice(&src[..n * 3]),
        Layout8::Bgr => {
            out.copy_from_slice(&src[..n * 3]);
            rgb_u8_swap_rb_inplace(out);
        }
        Layout8::Rgbx => super::ycbcr::rgb_u8_to_xrgba_u8(&src[..n * 3], out, false),
        Layout8::Bgrx => super::ycbcr::rgb_u8_to_xrgba_u8(&src[..n * 3], out, true),
        Layout8::Gray => {
            for (o, p) in out.iter_mut().zip(src.as_chunks::<3>().0) {
                *o = rgb_luma_u8(p[0], p[1], p[2]);
            }
        }
    }
}

/// Pixels of 8-bit `source` layout into `target` layout (any number of rows:
/// a tightly packed buffer is one long row). Colour into
/// [`Layout8::Gray`] is the luma of the RGB ([`rgb_luma_u8`]), so callers
/// only pass colour-to-gray for one-component frames, where R = G = B makes
/// it exact. Allocation-free: layouts other than packed RGB or gray go through
/// a stack chunk.
pub(crate) fn relayout_u8(src: &[u8], source: Layout8, dst: &mut [u8], target: Layout8) {
    let n = src.len() / source.bytes_per_pixel();
    let src = &src[..n * source.bytes_per_pixel()];
    let dst = &mut dst[..n * target.bytes_per_pixel()];
    if source == target {
        dst.copy_from_slice(src);
        return;
    }
    match source {
        Layout8::Gray => gray8_row(src, target, dst),
        Layout8::Rgb => rgb8_row(src, target, dst),
        Layout8::Bgr | Layout8::Rgbx | Layout8::Bgrx => {
            const CHUNK: usize = 256;
            let (sbpp, tbpp) = (source.bytes_per_pixel(), target.bytes_per_pixel());
            let swap = matches!(source, Layout8::Bgr | Layout8::Bgrx);
            let mut rgb = [0u8; CHUNK * 3];
            for (s, d) in src.chunks(CHUNK * sbpp).zip(dst.chunks_mut(CHUNK * tbpp)) {
                let m = s.len() / sbpp;
                for (o, p) in rgb[..m * 3]
                    .as_chunks_mut::<3>()
                    .0
                    .iter_mut()
                    .zip(s.chunks_exact(sbpp))
                {
                    *o = if swap {
                        [p[2], p[1], p[0]]
                    } else {
                        [p[0], p[1], p[2]]
                    };
                }
                rgb8_row(&rgb[..m * 3], target, d);
            }
        }
    }
}

/// One row of decoded 8-bit gray into `layout` (replicated into colour).
pub(crate) fn gray8_row(src: &[u8], layout: Layout8, out: &mut [u8]) {
    let bpp = layout.bytes_per_pixel();
    let out = &mut out[..src.len() * bpp];
    match layout {
        Layout8::Gray => out.copy_from_slice(src),
        Layout8::Rgb | Layout8::Bgr => {
            for (o, &v) in out.as_chunks_mut::<3>().0.iter_mut().zip(src) {
                o.copy_from_slice(&[v, v, v]);
            }
        }
        Layout8::Rgbx | Layout8::Bgrx => {
            for (o, &v) in out.as_chunks_mut::<4>().0.iter_mut().zip(src) {
                o.copy_from_slice(&[v, v, v, 255]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// RGB-derived values. One definition each, shared by every reader mode (the
// streaming strip, the coefficient reader and the buffered RGB reader), so the
// modes agree bit for bit.
// ---------------------------------------------------------------------------

/// BT.601 luma of 8-bit R, G, B, as libjpeg-turbo's `rgb_gray_convert`:
/// `FIX(0.299) R + FIX(0.587) G + FIX(0.114) B`, 16-bit fixed point, rounded.
#[inline(always)]
pub(crate) fn rgb_luma_u8(r: u8, g: u8, b: u8) -> u8 {
    ((19595 * r as u32 + 38470 * g as u32 + 7471 * b as u32 + 32768) >> 16) as u8
}

/// BT.601 luma of 8-bit R, G, B, normalized to [0, 1].
#[inline(always)]
pub(crate) fn rgb_luma_f32(r: u8, g: u8, b: u8) -> f32 {
    use crate::foundation::consts::{YCBCR_B_TO_Y, YCBCR_G_TO_Y, YCBCR_R_TO_Y};
    (YCBCR_R_TO_Y * r as f32 + YCBCR_G_TO_Y * g as f32 + YCBCR_B_TO_Y * b as f32) / 255.0
}

/// BT.601 forward transform of 8-bit R, G, B, normalized: Y in [0, 1], Cb and
/// Cr centred on 0.
#[inline(always)]
pub(crate) fn rgb_to_ycbcr_normalized(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let (y, cb, cr) = rgb_to_ycbcr_bt601(r as f32, g as f32, b as f32);
    (y / 255.0, cb / 255.0, cr / 255.0)
}

/// BT.601 forward transform of R, G, B in [0, 255]: Y in [0, 255], Cb and Cr
/// centred on 0. Plain multiply-add (no fused multiply-add, which compiles to
/// a library call on targets without FMA).
#[inline(always)]
pub(crate) fn rgb_to_ycbcr_bt601(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    use crate::foundation::consts::{
        YCBCR_B_TO_CB, YCBCR_B_TO_CR, YCBCR_B_TO_Y, YCBCR_G_TO_CB, YCBCR_G_TO_CR, YCBCR_G_TO_Y,
        YCBCR_R_TO_CB, YCBCR_R_TO_CR, YCBCR_R_TO_Y,
    };
    (
        YCBCR_R_TO_Y * r + YCBCR_G_TO_Y * g + YCBCR_B_TO_Y * b,
        YCBCR_R_TO_CB * r + YCBCR_G_TO_CB * g + YCBCR_B_TO_CB * b,
        YCBCR_R_TO_CR * r + YCBCR_G_TO_CR * g + YCBCR_B_TO_CR * b,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Independent samples per plane from an LCG, in `lo..=hi`.
    fn lcg_plane(n: usize, seed: u32, lo: i16, hi: i16) -> Vec<i16> {
        let mut x = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
        let span = (hi as i32 - lo as i32 + 1) as u32;
        (0..n)
            .map(|_| {
                x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (lo as i32 + ((x >> 8) % span) as i32) as i16
            })
            .collect()
    }

    /// YCbCr to RGB from the formulas, not from the kernels. `turbo`:
    /// libjpeg-turbo's table form (jdcolor.c `build_ycc_rgb_table`,
    /// `FIX(x) = (x * 2^16 + 0.5)` truncated): R = Y + ((FIX(1.402) Cr +
    /// ONE_HALF) >> 16), G = Y + ((-FIX(0.34414) Cb - FIX(0.71414) Cr +
    /// ONE_HALF) >> 16), B = Y + ((FIX(1.772) Cb + ONE_HALF) >> 16). Otherwise
    /// the 14-bit form ((Y << 14) + 2^13 + k Cr...) >> 14 with k = round(x *
    /// 2^14). Samples enter as they are (negative and above 255 included);
    /// the result saturates to [0, 255].
    fn ycbcr_ref(y: i16, cb: i16, cr: i16, turbo: bool) -> [u8; 3] {
        let (y, cb, cr) = (y as i64, cb as i64 - 128, cr as i64 - 128);
        let sat = |v: i64| v.clamp(0, 255) as u8;
        if turbo {
            let fix = |x: f64| (x * 65536.0 + 0.5) as i64;
            let half = 1i64 << 15;
            [
                sat(y + ((fix(1.402) * cr + half) >> 16)),
                sat(y + ((-fix(0.34414) * cb - fix(0.71414) * cr + half) >> 16)),
                sat(y + ((fix(1.772) * cb + half) >> 16)),
            ]
        } else {
            let k = |x: f64| (x * 16384.0).round() as i64;
            let ys = (y << 14) + (1 << 13);
            [
                sat((ys + k(1.402) * cr) >> 14),
                sat((ys - k(0.714136) * cr - k(0.344136) * cb) >> 14),
                sat((ys + k(1.772) * cb) >> 14),
            ]
        }
    }

    /// libjpeg-turbo's `rgb_gray_convert` table (jdcolor.c
    /// `build_rgb_y_table`), built from the BT.601 weights.
    fn luma_table() -> [[u32; 256]; 3] {
        let fix = |x: f64| (x * 65536.0 + 0.5) as u32;
        let mut t = [[0u32; 256]; 3];
        for i in 0..256u32 {
            t[0][i as usize] = fix(0.299) * i;
            t[1][i as usize] = fix(0.587) * i;
            t[2][i as usize] = fix(0.114) * i + (1 << 15);
        }
        t
    }

    fn clamp8(v: i16) -> u8 {
        v.clamp(0, 255) as u8
    }

    fn put(out: &mut Vec<u8>, layout: Layout8, [r, g, b]: [u8; 3], luma: u8) {
        match layout {
            Layout8::Gray => out.push(luma),
            Layout8::Rgb => out.extend([r, g, b]),
            Layout8::Bgr => out.extend([b, g, r]),
            Layout8::Rgbx => out.extend([r, g, b, 255]),
            Layout8::Bgrx => out.extend([b, g, r, 255]),
        }
    }

    fn reference(matrix: PlaneMatrix, layout: Layout8, c: &[Vec<i16>; 3], h2: bool) -> Vec<u8> {
        let t = luma_table();
        let mut out = Vec::new();
        for x in 0..c[0].len() {
            let cx = if h2 { x / 2 } else { x };
            let (rgb, luma) = match matrix {
                PlaneMatrix::Gray => {
                    let v = clamp8(c[0][x]);
                    ([v, v, v], v)
                }
                PlaneMatrix::Identity => {
                    let p = [clamp8(c[0][x]), clamp8(c[1][cx]), clamp8(c[2][cx])];
                    let l = (t[0][p[0] as usize] + t[1][p[1] as usize] + t[2][p[2] as usize]) >> 16;
                    (p, l as u8)
                }
                PlaneMatrix::YCbCr { turbo } => (
                    ycbcr_ref(c[0][x], c[1][cx], c[2][cx], turbo),
                    clamp8(c[0][x]),
                ),
            };
            put(&mut out, layout, rgb, luma);
        }
        out
    }

    const MATRICES: [PlaneMatrix; 4] = [
        PlaneMatrix::Gray,
        PlaneMatrix::Identity,
        PlaneMatrix::YCbCr { turbo: false },
        PlaneMatrix::YCbCr { turbo: true },
    ];
    const LAYOUTS: [Layout8; 5] = [
        Layout8::Gray,
        Layout8::Rgb,
        Layout8::Bgr,
        Layout8::Rgbx,
        Layout8::Bgrx,
    ];

    /// Run `f` once per SIMD dispatch tier the host supports (x86-64), so the
    /// AVX-512, AVX2 and scalar kernels are each checked against the
    /// definitions; once elsewhere.
    fn on_every_tier(f: impl Fn()) {
        #[cfg(target_arch = "x86_64")]
        {
            use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
            let report = for_each_token_permutation(CompileTimePolicy::Warn, |_| f());
            assert!(
                report.permutations_run >= 2,
                "expected several tiers: {report}"
            );
        }
        #[cfg(not(target_arch = "x86_64"))]
        f();
    }

    /// Every (matrix, layout) of `row` and `row_h2` equals the formulas, on
    /// every tier, with independent planes, in range and out of range (strip
    /// rows from the unclamped IDCT of f32 output targets go below 0 and above
    /// 255: the AVX2 YCbCr kernels zero-extended Y there). Widths cover every
    /// SIMD block size and the `row_h2` chunk boundary.
    #[test]
    fn every_kernel_matches_its_definition_on_every_tier() {
        let widths = [1usize, 2, 7, 15, 16, 17, 31, 33, 255, 256, 257, 511, 1001];
        // Negative control: the two YCbCr constant sets disagree on these
        // inputs, so a lost or ignored `turbo` flag cannot pass.
        let c = [
            lcg_plane(1001, 1, 0, 255),
            lcg_plane(1001, 2, 0, 255),
            lcg_plane(1001, 3, 0, 255),
        ];
        let differ = (0..1001)
            .filter(|&i| {
                ycbcr_ref(c[0][i], c[1][i], c[2][i], true)
                    != ycbcr_ref(c[0][i], c[1][i], c[2][i], false)
            })
            .count();
        assert!(
            differ > 0,
            "turbo and default YCbCr references agree on every input"
        );

        on_every_tier(|| {
            let lim = SAMPLE_LIMIT as i16;
            for (lo, hi) in [(0i16, 255i16), (-300, 560), (-lim, lim)] {
                for &n in &widths {
                    let seed = n as u32 * 7 + lo.unsigned_abs() as u32;
                    let c = [
                        lcg_plane(n, seed, lo, hi),
                        lcg_plane(n, seed + 1, lo, hi),
                        lcg_plane(n, seed + 2, lo, hi),
                    ];
                    let cn = n.div_ceil(2);
                    let half = [c[0].clone(), c[1][..cn].to_vec(), c[2][..cn].to_vec()];
                    for matrix in MATRICES {
                        for layout in LAYOUTS {
                            let k = RowKernel::new(matrix, layout);
                            let mut out = vec![0xAAu8; n * layout.bytes_per_pixel()];
                            k.row(&c[0], &c[1], &c[2], &mut out);
                            assert_eq!(
                                out,
                                reference(matrix, layout, &c, false),
                                "row {matrix:?} {layout:?} n={n} range {lo}..={hi}"
                            );
                            let mut out = vec![0xAAu8; n * layout.bytes_per_pixel()];
                            k.row_h2(&half[0], &half[1], &half[2], &mut out);
                            assert_eq!(
                                out,
                                reference(matrix, layout, &half, true),
                                "row_h2 {matrix:?} {layout:?} n={n} range {lo}..={hi}"
                            );
                        }
                    }
                }
            }
        });
    }

    /// `rgb8_row`, `gray8_row` and `relayout_u8` between every pair of
    /// layouts equal their definitions (gray from colour is the
    /// `rgb_gray_convert` luma), on every tier.
    #[test]
    fn row8_helpers_match_definitions() {
        let t = luma_table();
        on_every_tier(|| {
            for n in [1usize, 5, 16, 17, 255, 256, 257, 700] {
                let rgb: Vec<u8> = lcg_plane(n * 3, n as u32, 0, 255)
                    .iter()
                    .map(|&v| v as u8)
                    .collect();
                let gray: Vec<u8> = lcg_plane(n, n as u32 + 9, 0, 255)
                    .iter()
                    .map(|&v| v as u8)
                    .collect();
                for target in LAYOUTS {
                    let want_rgb = {
                        let mut v = Vec::new();
                        for p in rgb.as_chunks::<3>().0 {
                            let l =
                                (t[0][p[0] as usize] + t[1][p[1] as usize] + t[2][p[2] as usize])
                                    >> 16;
                            put(&mut v, target, *p, l as u8);
                        }
                        v
                    };
                    let want_gray = {
                        let mut v = Vec::new();
                        for &g in &gray {
                            put(&mut v, target, [g, g, g], g);
                        }
                        v
                    };
                    let mut out = vec![0xAAu8; n * target.bytes_per_pixel()];
                    rgb8_row(&rgb, target, &mut out);
                    assert_eq!(out, want_rgb, "rgb8_row {target:?} n={n}");
                    let mut out = vec![0xAAu8; n * target.bytes_per_pixel()];
                    gray8_row(&gray, target, &mut out);
                    assert_eq!(out, want_gray, "gray8_row {target:?} n={n}");
                    // relayout_u8 from every source layout of the same pixels.
                    for source in LAYOUTS.into_iter().filter(|l| *l != Layout8::Gray) {
                        let mut src = vec![0u8; n * source.bytes_per_pixel()];
                        rgb8_row(&rgb, source, &mut src);
                        let mut out = vec![0xAAu8; n * target.bytes_per_pixel()];
                        relayout_u8(&src, source, &mut out, target);
                        assert_eq!(out, want_rgb, "relayout {source:?} -> {target:?} n={n}");
                    }
                    let mut out = vec![0xAAu8; n * target.bytes_per_pixel()];
                    relayout_u8(&gray, Layout8::Gray, &mut out, target);
                    assert_eq!(out, want_gray, "relayout Gray -> {target:?} n={n}");
                }
            }
        });
    }

    /// `rgb_luma_u8` is libjpeg-turbo's `rgb_gray_convert` on all 2^24 inputs.
    #[test]
    fn luma_is_libjpeg_turbo_rgb_gray_convert_exhaustive() {
        let t = luma_table();
        let mut mismatches = 0u32;
        for r in 0..256usize {
            for g in 0..256usize {
                for b in 0..256usize {
                    let want = ((t[0][r] + t[1][g] + t[2][b]) >> 16) as u8;
                    mismatches += u32::from(rgb_luma_u8(r as u8, g as u8, b as u8) != want);
                }
            }
        }
        assert_eq!(mismatches, 0);
        let (y, cb, cr) = rgb_to_ycbcr_normalized(255, 255, 255);
        assert!((y - 1.0).abs() < 1e-6 && cb.abs() < 1e-6 && cr.abs() < 1e-6);
    }

    /// Turbo and default constants give different RGB for some (Y, Cb, Cr):
    /// place such triples at every pixel, so a kernel lane or scalar
    /// remainder that ignores (or swaps) the `turbo` flag fails at its own
    /// position. `row` gets a differing triple per pixel; `row_h2` per chroma
    /// sample, both pixels of the pair.
    #[test]
    fn turbo_flag_reaches_every_lane_and_remainder() {
        let mut differing = Vec::new();
        'search: for cr in (0..=255i16).step_by(7) {
            for cb in (0..=255i16).step_by(5) {
                for y in 0..=255i16 {
                    if ycbcr_ref(y, cb, cr, true) != ycbcr_ref(y, cb, cr, false) {
                        differing.push((y, cb, cr));
                        if differing.len() == 61 {
                            break 'search;
                        }
                    }
                }
            }
        }
        assert_eq!(differing.len(), 61, "not enough differing triples");
        on_every_tier(|| {
            for n in [1usize, 15, 16, 17, 31, 32, 33, 47, 255, 256, 257, 511, 1001] {
                let t = |x: usize| differing[x % differing.len()];
                let full = [
                    (0..n).map(|x| t(x).0).collect::<Vec<_>>(),
                    (0..n).map(|x| t(x).1).collect(),
                    (0..n).map(|x| t(x).2).collect(),
                ];
                let cn = n.div_ceil(2);
                let half = [
                    (0..n).map(|x| t(x / 2).0).collect::<Vec<_>>(),
                    (0..cn).map(|x| t(x).1).collect(),
                    (0..cn).map(|x| t(x).2).collect(),
                ];
                for turbo in [false, true] {
                    let matrix = PlaneMatrix::YCbCr { turbo };
                    for layout in [Layout8::Rgb, Layout8::Bgr, Layout8::Rgbx, Layout8::Bgrx] {
                        let k = RowKernel::new(matrix, layout);
                        let bpp = layout.bytes_per_pixel();
                        let mut out = vec![0u8; n * bpp];
                        k.row(&full[0], &full[1], &full[2], &mut out);
                        assert_eq!(
                            out,
                            reference(matrix, layout, &full, false),
                            "row {matrix:?} {layout:?} n={n}"
                        );
                        let mut out = vec![0u8; n * bpp];
                        k.row_h2(&half[0], &half[1], &half[2], &mut out);
                        assert_eq!(
                            out,
                            reference(matrix, layout, &half, true),
                            "row_h2 {matrix:?} {layout:?} n={n}"
                        );
                    }
                }
            }
        });
    }
}
