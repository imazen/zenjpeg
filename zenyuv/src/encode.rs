//! RGB->YCbCr encode functions (public API).
//!
//! Each function dispatches to the best available SIMD kernel at runtime:
//! AVX2 > NEON > WASM SIMD128 > generic (magetypes f32x8).

use archmage::prelude::*;
use crate::types::{ForwardCoeffs, Matrix, Range};

/// Convert packed 24-bit RGB to three u8 Y/Cb/Cr planes at full resolution (4:4:4).
///
/// Uses BT.601 full-range by default. `rgb` must be `width * height * 3` bytes.
/// The three output planes must each be at least `width * height` bytes.
pub fn rgb_to_yuv444(
    rgb: &[u8],
    y: &mut [u8],
    cb: &mut [u8],
    cr: &mut [u8],
    width: usize,
    height: usize,
) {
    rgb_to_yuv444_with(rgb, y, cb, cr, width, height, Range::Full, Matrix::Bt601);
}

/// Convert packed 24-bit RGB to three u8 Y/Cb/Cr planes at full resolution (4:4:4)
/// with the specified range and matrix.
pub fn rgb_to_yuv444_with(
    rgb: &[u8],
    y: &mut [u8],
    cb: &mut [u8],
    cr: &mut [u8],
    width: usize,
    height: usize,
    range: Range,
    matrix: Matrix,
) {
    let n = width * height;
    assert!(rgb.len() >= n * 3);
    assert!(y.len() >= n);
    assert!(cb.len() >= n);
    assert!(cr.len() >= n);

    let coeffs = ForwardCoeffs::new(matrix, range);

    #[cfg(target_arch = "x86_64")]
    if let Some(token) = archmage::X64V3Token::summon() {
        let done = crate::avx2_encode::rgb_to_yuv444_avx2(token, rgb, y, cb, cr, n, &coeffs);
        if done < n {
            rgb_to_yuv444_scalar_tail(rgb, y, cb, cr, done, n, &coeffs);
        }
        return;
    }

    incant!(crate::encode_generic::rgb_to_yuv444_generic(rgb, y, cb, cr, n, &coeffs));
}

/// Convert packed 24-bit RGB to three u8 Y/Cb/Cr planes with 4:2:0 subsampling.
///
/// Uses BT.601 full-range by default. Y is full-resolution (`width * height`);
/// Cb/Cr are `ceil(width/2) * ceil(height/2)`.
pub fn rgb_to_yuv420(
    rgb: &[u8],
    y: &mut [u8],
    cb: &mut [u8],
    cr: &mut [u8],
    width: usize,
    height: usize,
) {
    rgb_to_yuv420_with(rgb, y, cb, cr, width, height, Range::Full, Matrix::Bt601);
}

/// Convert packed 24-bit RGB to three u8 Y/Cb/Cr planes with 4:2:0 subsampling,
/// with the specified range and matrix.
pub fn rgb_to_yuv420_with(
    rgb: &[u8],
    y: &mut [u8],
    cb: &mut [u8],
    cr: &mut [u8],
    width: usize,
    height: usize,
    range: Range,
    matrix: Matrix,
) {
    let n = width * height;
    let cw = width.div_ceil(2);
    let ch = height.div_ceil(2);
    assert!(rgb.len() >= n * 3);
    assert!(y.len() >= n);
    assert!(cb.len() >= cw * ch);
    assert!(cr.len() >= cw * ch);

    let coeffs = ForwardCoeffs::new(matrix, range);

    #[cfg(target_arch = "x86_64")]
    if let Some(token) = archmage::X64V3Token::summon() {
        crate::avx2_encode::rgb_to_yuv420_avx2(token, rgb, y, cb, cr, width, height, cw, &coeffs);
        return;
    }

    incant!(crate::encode_generic::rgb_to_yuv420_generic(rgb, y, cb, cr, width, height, &coeffs));
}

/// Scalar tail for 4:4:4 encode (pixels not covered by SIMD blocks).
#[inline]
pub(crate) fn rgb_to_yuv444_scalar_tail(
    rgb: &[u8],
    y: &mut [u8],
    cb: &mut [u8],
    cr: &mut [u8],
    start: usize,
    end: usize,
    coeffs: &ForwardCoeffs,
) {
    for i in start..end {
        let p = i * 3;
        let r = rgb[p] as f32;
        let g = rgb[p + 1] as f32;
        let b = rgb[p + 2] as f32;
        y[i] = crate::clamp_round(coeffs.yr_f * r + coeffs.yg_f * g + coeffs.yb_f * b + coeffs.y_bias_f);
        cb[i] = crate::clamp_round(coeffs.cb_r_f * r + coeffs.cb_g_f * g + coeffs.cb_b_f * b + coeffs.uv_bias_f);
        cr[i] = crate::clamp_round(coeffs.cr_r_f * r + coeffs.cr_g_f * g + coeffs.cr_b_f * b + coeffs.uv_bias_f);
    }
}
