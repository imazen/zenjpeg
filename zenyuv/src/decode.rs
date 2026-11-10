//! YCbCr->RGB decode functions (public API).
//!
//! Each function dispatches to the best available SIMD kernel at runtime:
//! AVX2 > NEON > WASM SIMD128 > generic (magetypes f32x8).

use crate::types::{InverseCoeffs, Matrix, Range, Strides, assert_plane, rgb_pixel_dims};
use archmage::prelude::*;

/// A plane's samples and stride, checked: padding between rows is never read,
/// and the last row needs none.
#[track_caller]
fn checked_plane<'a>(
    name: &'static str,
    plane: imgref::ImgRef<'a, u8>,
    width: usize,
    height: usize,
) -> (&'a [u8], usize) {
    let (buf, stride) = (*plane.buf(), plane.stride());
    let got = (plane.width(), plane.height());
    assert_plane(name, buf.len(), got, stride, (width, height));
    (buf, stride)
}

/// The packed output's samples, pixel dimensions and stride, checked.
#[track_caller]
fn checked_rgb<'a>(rgb: &'a mut imgref::ImgRefMut<'_, u8>) -> (&'a mut [u8], usize, usize, usize) {
    let dims = (rgb.width(), rgb.height());
    let stride = rgb.stride();
    assert_plane("rgb", rgb.buf().len(), dims, stride, dims);
    let (width, height) = rgb_pixel_dims(dims.0, dims.1).unwrap_or_else(|e| panic!("{e}"));
    (&mut **rgb.buf_mut(), width, height, stride)
}

/// Convert 4:4:4 Y/Cb/Cr planes to packed 24-bit RGB.
///
/// Each plane is a view at its own stride, in samples; padding is neither read nor
/// written, and the last row needs none. `rgb`'s width and stride count samples,
/// three per pixel.
///
/// # Panics
///
/// If a plane's dimensions or storage do not fit `rgb`'s dimensions.
pub fn yuv444_to_rgb(
    y: imgref::ImgRef<'_, u8>,
    cb: imgref::ImgRef<'_, u8>,
    cr: imgref::ImgRef<'_, u8>,
    mut rgb: imgref::ImgRefMut<'_, u8>,
    range: Range,
    matrix: Matrix,
) {
    let (out, width, height, rgb_stride) = checked_rgb(&mut rgb);
    let (y, ys) = checked_plane("y", y, width, height);
    let (cb, cbs) = checked_plane("cb", cb, width, height);
    let (cr, crs) = checked_plane("cr", cr, width, height);
    if width == 0 || height == 0 {
        return;
    }
    let coeffs = InverseCoeffs::new(matrix, range);
    if ys == width && cbs == width && crs == width && rgb_stride == width * 3 {
        let n = width * height;
        incant!(crate::decode_generic::yuv444_to_rgb_generic(
            y, cb, cr, out, n, &coeffs
        ));
        return;
    }
    // The kernel's SIMD lanes and its scalar tail may round a tie differently,
    // so which pixels take the tail is part of the output. Keep the tight form's
    // choice: chunks of 8 over the whole image, the tail only at its end.
    if width % 8 == 0 {
        for row in 0..height {
            let (y, cb, cr) = (&y[row * ys..], &cb[row * cbs..], &cr[row * crs..]);
            let out = &mut out[row * rgb_stride..];
            incant!(crate::decode_generic::yuv444_to_rgb_generic(
                y, cb, cr, out, width, &coeffs
            ));
        }
        return;
    }
    let n = width * height;
    let mut start = 0;
    while start < n {
        let count = (n - start).min(8);
        let (mut ya, mut cba, mut cra, mut px) = ([0u8; 8], [0u8; 8], [0u8; 8], [0u8; 24]);
        for i in 0..count {
            let (row, col) = ((start + i) / width, (start + i) % width);
            ya[i] = y[row * ys + col];
            cba[i] = cb[row * cbs + col];
            cra[i] = cr[row * crs + col];
        }
        let (ya, cba, cra) = (&ya[..], &cba[..], &cra[..]);
        let chunk = &mut px[..];
        incant!(crate::decode_generic::yuv444_to_rgb_generic(
            ya, cba, cra, chunk, count, &coeffs
        ));
        for i in 0..count {
            let (row, col) = ((start + i) / width, (start + i) % width);
            let at = row * rgb_stride + col * 3;
            out[at..at + 3].copy_from_slice(&px[i * 3..i * 3 + 3]);
        }
        start += count;
    }
}

/// Convert 4:2:0 Y/Cb/Cr planes to packed 24-bit RGB (nearest-neighbor chroma upsampling).
///
/// Each plane is a view at its own stride, in samples; padding is neither read nor
/// written, and the last row needs none. `rgb`'s width and stride count samples,
/// three per pixel.
///
/// # Panics
///
/// If a plane's dimensions or storage do not fit `rgb`'s dimensions.
pub fn yuv420_to_rgb(
    y: imgref::ImgRef<'_, u8>,
    cb: imgref::ImgRef<'_, u8>,
    cr: imgref::ImgRef<'_, u8>,
    mut rgb: imgref::ImgRefMut<'_, u8>,
    range: Range,
    matrix: Matrix,
) {
    let (out, width, height, rgb_stride) = checked_rgb(&mut rgb);
    let (cw, ch) = (width.div_ceil(2), height.div_ceil(2));
    let (y, ys) = checked_plane("y", y, width, height);
    let (cb, cbs) = checked_plane("cb", cb, cw, ch);
    let (cr, crs) = checked_plane("cr", cr, cw, ch);
    let strides = Strides {
        rgb: rgb_stride,
        y: ys,
        cb: cbs,
        cr: crs,
    };
    if width == 0 || height == 0 {
        return;
    }
    let coeffs = InverseCoeffs::new(matrix, range);
    incant!(crate::decode_generic::yuv420_to_rgb_generic(
        y, cb, cr, out, width, height, strides, &coeffs
    ));
}

/// Convert 4:2:0 Y/Cb/Cr planes to packed 24-bit RGB (bilinear chroma upsampling).
///
/// Each plane is a view at its own stride, in samples; padding is neither read nor
/// written, and the last row needs none. `rgb`'s width and stride count samples,
/// three per pixel.
///
/// # Panics
///
/// If a plane's dimensions or storage do not fit `rgb`'s dimensions.
pub fn yuv420_to_rgb_bilinear(
    y: imgref::ImgRef<'_, u8>,
    cb: imgref::ImgRef<'_, u8>,
    cr: imgref::ImgRef<'_, u8>,
    mut rgb: imgref::ImgRefMut<'_, u8>,
    range: Range,
    matrix: Matrix,
) {
    let (out, width, height, rgb_stride) = checked_rgb(&mut rgb);
    let (cw, ch) = (width.div_ceil(2), height.div_ceil(2));
    let (y, ys) = checked_plane("y", y, width, height);
    let (cb, cbs) = checked_plane("cb", cb, cw, ch);
    let (cr, crs) = checked_plane("cr", cr, cw, ch);
    let strides = Strides {
        rgb: rgb_stride,
        y: ys,
        cb: cbs,
        cr: crs,
    };
    if width == 0 || height == 0 {
        return;
    }
    let coeffs = InverseCoeffs::new(matrix, range);
    incant!(crate::decode_generic::yuv420_to_rgb_bilinear_generic(
        y, cb, cr, out, width, height, strides, &coeffs
    ));
}

/// Convert 4:2:2 Y/Cb/Cr planes to packed 24-bit RGB.
///
/// Each plane is a view at its own stride, in samples; padding is neither read nor
/// written, and the last row needs none. `rgb`'s width and stride count samples,
/// three per pixel.
///
/// # Panics
///
/// If a plane's dimensions or storage do not fit `rgb`'s dimensions.
pub fn yuv422_to_rgb(
    y: imgref::ImgRef<'_, u8>,
    cb: imgref::ImgRef<'_, u8>,
    cr: imgref::ImgRef<'_, u8>,
    mut rgb: imgref::ImgRefMut<'_, u8>,
    range: Range,
    matrix: Matrix,
) {
    let (out, width, height, rgb_stride) = checked_rgb(&mut rgb);
    let (cw, ch) = (width.div_ceil(2), height);
    let (y, ys) = checked_plane("y", y, width, height);
    let (cb, cbs) = checked_plane("cb", cb, cw, ch);
    let (cr, crs) = checked_plane("cr", cr, cw, ch);
    let strides = Strides {
        rgb: rgb_stride,
        y: ys,
        cb: cbs,
        cr: crs,
    };
    if width == 0 || height == 0 {
        return;
    }
    let coeffs = InverseCoeffs::new(matrix, range);
    incant!(crate::decode_generic::yuv422_to_rgb_generic(
        y, cb, cr, out, width, height, strides, &coeffs
    ));
}

/// Convert a Y plane to packed gray RGB.
///
/// Each plane is a view at its own stride, in samples; padding is neither read nor
/// written, and the last row needs none. `rgb`'s width and stride count samples,
/// three per pixel.
///
/// # Panics
///
/// If a plane's dimensions or storage do not fit `rgb`'s dimensions.
pub fn yuv400_to_rgb(
    y: imgref::ImgRef<'_, u8>,
    mut rgb: imgref::ImgRefMut<'_, u8>,
    range: Range,
    matrix: Matrix,
) {
    let (out, width, height, rgb_stride) = checked_rgb(&mut rgb);
    let (y, ys) = checked_plane("y", y, width, height);
    if width == 0 || height == 0 {
        return;
    }
    let coeffs = InverseCoeffs::new(matrix, range);
    if ys == width && rgb_stride == width * 3 {
        let n = width * height;
        incant!(crate::decode_generic::yuv400_to_rgb_generic(
            y, out, n, &coeffs
        ));
        return;
    }
    // As in `yuv444_to_rgb_strided`: chunks of 8 over the whole image.
    if width % 8 == 0 {
        for row in 0..height {
            let y = &y[row * ys..];
            let out = &mut out[row * rgb_stride..];
            incant!(crate::decode_generic::yuv400_to_rgb_generic(
                y, out, width, &coeffs
            ));
        }
        return;
    }
    let n = width * height;
    let mut start = 0;
    while start < n {
        let count = (n - start).min(8);
        let (mut ya, mut px) = ([0u8; 8], [0u8; 24]);
        for (i, sample) in ya.iter_mut().enumerate().take(count) {
            let (row, col) = ((start + i) / width, (start + i) % width);
            *sample = y[row * ys + col];
        }
        let ya = &ya[..];
        let chunk = &mut px[..];
        incant!(crate::decode_generic::yuv400_to_rgb_generic(
            ya, chunk, count, &coeffs
        ));
        for i in 0..count {
            let (row, col) = ((start + i) / width, (start + i) % width);
            let at = row * rgb_stride + col * 3;
            out[at..at + 3].copy_from_slice(&px[i * 3..i * 3 + 3]);
        }
        start += count;
    }
}
