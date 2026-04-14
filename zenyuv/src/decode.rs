//! YCbCr->RGB decode functions (public API).
//!
//! Each function dispatches to the best available SIMD kernel at runtime:
//! AVX2 > NEON > WASM SIMD128 > generic (magetypes f32x8).

use archmage::prelude::*;
use crate::types::{InverseCoeffs, Matrix, Range};

/// Convert 4:4:4 Y/Cb/Cr planes to packed 24-bit RGB.
///
/// Uses BT.601 full-range by default. All planes must be `width * height`.
/// Output `rgb` must be at least `width * height * 3` bytes.
pub fn yuv444_to_rgb(
    y: &[u8],
    cb: &[u8],
    cr: &[u8],
    rgb: &mut [u8],
    width: usize,
    height: usize,
) {
    yuv444_to_rgb_with(y, cb, cr, rgb, width, height, Range::Full, Matrix::Bt601);
}

/// Convert 4:4:4 Y/Cb/Cr planes to packed 24-bit RGB with specified range and matrix.
pub fn yuv444_to_rgb_with(
    y: &[u8],
    cb: &[u8],
    cr: &[u8],
    rgb: &mut [u8],
    width: usize,
    height: usize,
    range: Range,
    matrix: Matrix,
) {
    let n = width * height;
    assert!(y.len() >= n);
    assert!(cb.len() >= n);
    assert!(cr.len() >= n);
    assert!(rgb.len() >= n * 3);

    let coeffs = InverseCoeffs::new(matrix, range);

    // Note: AVX2 decode via mulhrs_epi16 has i16 overflow for y_coeff >= 1.0
    // (full-range: 32768, limited: 38142). Using generic f32 path for now.
    // TODO: AVX2 decode with i32 arithmetic (pmaddwd) to avoid overflow.

    incant!(crate::decode_generic::yuv444_to_rgb_generic(y, cb, cr, rgb, n, &coeffs));
}

/// Convert 4:2:0 Y/Cb/Cr planes to packed 24-bit RGB (nearest-neighbor chroma upsampling).
///
/// Uses BT.601 full-range by default. Y is `width * height`;
/// Cb/Cr are `ceil(width/2) * ceil(height/2)`.
pub fn yuv420_to_rgb(
    y: &[u8],
    cb: &[u8],
    cr: &[u8],
    rgb: &mut [u8],
    width: usize,
    height: usize,
) {
    yuv420_to_rgb_with(y, cb, cr, rgb, width, height, Range::Full, Matrix::Bt601);
}

/// Convert 4:2:0 Y/Cb/Cr planes to packed 24-bit RGB with specified range and matrix.
pub fn yuv420_to_rgb_with(
    y: &[u8],
    cb: &[u8],
    cr: &[u8],
    rgb: &mut [u8],
    width: usize,
    height: usize,
    range: Range,
    matrix: Matrix,
) {
    let n = width * height;
    let cw = width.div_ceil(2);
    let ch = height.div_ceil(2);
    assert!(y.len() >= n);
    assert!(cb.len() >= cw * ch);
    assert!(cr.len() >= cw * ch);
    assert!(rgb.len() >= n * 3);

    let coeffs = InverseCoeffs::new(matrix, range);
    incant!(crate::decode_generic::yuv420_to_rgb_generic(y, cb, cr, rgb, width, height, &coeffs));
}

/// Convert 4:2:0 Y/Cb/Cr planes to packed 24-bit RGB with bilinear chroma upsampling.
pub fn yuv420_to_rgb_bilinear(
    y: &[u8],
    cb: &[u8],
    cr: &[u8],
    rgb: &mut [u8],
    width: usize,
    height: usize,
) {
    yuv420_to_rgb_bilinear_with(y, cb, cr, rgb, width, height, Range::Full, Matrix::Bt601);
}

/// Convert 4:2:0 Y/Cb/Cr planes to packed 24-bit RGB with bilinear chroma upsampling
/// and specified range and matrix.
pub fn yuv420_to_rgb_bilinear_with(
    y: &[u8],
    cb: &[u8],
    cr: &[u8],
    rgb: &mut [u8],
    width: usize,
    height: usize,
    range: Range,
    matrix: Matrix,
) {
    let n = width * height;
    let cw = width.div_ceil(2);
    let ch = height.div_ceil(2);
    assert!(y.len() >= n);
    assert!(cb.len() >= cw * ch);
    assert!(cr.len() >= cw * ch);
    assert!(rgb.len() >= n * 3);

    let coeffs = InverseCoeffs::new(matrix, range);
    incant!(crate::decode_generic::yuv420_to_rgb_bilinear_generic(y, cb, cr, rgb, width, height, &coeffs));
}

/// Convert 4:2:2 Y/Cb/Cr planes to packed 24-bit RGB.
pub fn yuv422_to_rgb(
    y: &[u8],
    cb: &[u8],
    cr: &[u8],
    rgb: &mut [u8],
    width: usize,
    height: usize,
) {
    yuv422_to_rgb_with(y, cb, cr, rgb, width, height, Range::Full, Matrix::Bt601);
}

/// Convert 4:2:2 Y/Cb/Cr planes to packed 24-bit RGB with specified range and matrix.
pub fn yuv422_to_rgb_with(
    y: &[u8],
    cb: &[u8],
    cr: &[u8],
    rgb: &mut [u8],
    width: usize,
    height: usize,
    range: Range,
    matrix: Matrix,
) {
    let n = width * height;
    let cw = width.div_ceil(2);
    assert!(y.len() >= n);
    assert!(cb.len() >= cw * height);
    assert!(cr.len() >= cw * height);
    assert!(rgb.len() >= n * 3);

    let coeffs = InverseCoeffs::new(matrix, range);
    incant!(crate::decode_generic::yuv422_to_rgb_generic(y, cb, cr, rgb, width, height, &coeffs));
}

/// Convert grayscale (4:0:0) Y plane to packed 24-bit RGB.
pub fn yuv400_to_rgb(
    y: &[u8],
    rgb: &mut [u8],
    width: usize,
    height: usize,
) {
    yuv400_to_rgb_with(y, rgb, width, height, Range::Full, Matrix::Bt601);
}

/// Convert grayscale (4:0:0) Y plane to packed 24-bit RGB with specified range and matrix.
pub fn yuv400_to_rgb_with(
    y: &[u8],
    rgb: &mut [u8],
    width: usize,
    height: usize,
    range: Range,
    matrix: Matrix,
) {
    let n = width * height;
    assert!(y.len() >= n);
    assert!(rgb.len() >= n * 3);

    let coeffs = InverseCoeffs::new(matrix, range);
    incant!(crate::decode_generic::yuv400_to_rgb_generic(y, rgb, n, &coeffs));
}

/// Scalar tail for 4:4:4 decode (used when AVX2 processes partial blocks).
#[allow(dead_code)]
#[inline]
fn yuv444_to_rgb_scalar_tail(
    y_plane: &[u8],
    cb_plane: &[u8],
    cr_plane: &[u8],
    rgb: &mut [u8],
    start: usize,
    end: usize,
    coeffs: &InverseCoeffs,
) {
    for i in start..end {
        let y_val = y_plane[i] as f32 + coeffs.y_offset;
        let cb_val = cb_plane[i] as f32 + coeffs.uv_offset;
        let cr_val = cr_plane[i] as f32 + coeffs.uv_offset;
        let y_scaled = y_val * coeffs.y_coeff;
        let p = i * 3;
        rgb[p] = crate::clamp_round(y_scaled + cr_val * coeffs.cr_to_r);
        rgb[p + 1] = crate::clamp_round(y_scaled + cb_val * coeffs.cb_to_g + cr_val * coeffs.cr_to_g);
        rgb[p + 2] = crate::clamp_round(y_scaled + cb_val * coeffs.cb_to_b);
    }
}
