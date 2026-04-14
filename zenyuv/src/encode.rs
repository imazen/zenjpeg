//! RGB->YCbCr encode functions (public API).
//!
//! Each function dispatches to the best available SIMD kernel at runtime:
//! AVX2 > NEON > WASM SIMD128 > generic (magetypes f32x8).

extern crate alloc;

use crate::types::{ForwardCoeffs, Matrix, Range};
use archmage::prelude::*;

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

    #[cfg(target_arch = "aarch64")]
    if let Some(token) = archmage::NeonToken::summon() {
        let done = crate::neon_encode::rgb_to_yuv444_neon(token, rgb, y, cb, cr, n, &coeffs);
        if done < n {
            rgb_to_yuv444_scalar_tail(rgb, y, cb, cr, done, n, &coeffs);
        }
        return;
    }

    incant!(crate::encode_generic::rgb_to_yuv444_generic(
        rgb, y, cb, cr, n, &coeffs
    ));
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

    incant!(crate::encode_generic::rgb_to_yuv420_generic(
        rgb, y, cb, cr, width, height, &coeffs
    ));
}

/// Compute Y plane only (no Cb/Cr) at full resolution. Used by Sharp YUV
/// which computes chroma separately via iterative optimization.
pub(crate) fn rgb_to_yuv444_y_only(
    rgb: &[u8],
    y: &mut [u8],
    width: usize,
    height: usize,
    range: Range,
    matrix: Matrix,
) {
    let n = width * height;
    assert!(rgb.len() >= n * 3);
    assert!(y.len() >= n);

    let coeffs = ForwardCoeffs::new(matrix, range);

    // Use the full 4:4:4 encode path but discard Cb/Cr.
    // TODO: write a Y-only AVX2 kernel to avoid computing Cb/Cr.
    let mut cb_discard = alloc::vec![0u8; n];
    let mut cr_discard = alloc::vec![0u8; n];

    #[cfg(target_arch = "x86_64")]
    if let Some(token) = archmage::X64V3Token::summon() {
        let done = crate::avx2_encode::rgb_to_yuv444_avx2(
            token,
            rgb,
            y,
            &mut cb_discard,
            &mut cr_discard,
            n,
            &coeffs,
        );
        if done < n {
            rgb_to_yuv444_scalar_tail(rgb, y, &mut cb_discard, &mut cr_discard, done, n, &coeffs);
        }
        return;
    }
    incant!(crate::encode_generic::rgb_to_yuv444_generic(
        rgb,
        y,
        &mut cb_discard,
        &mut cr_discard,
        n,
        &coeffs
    ));
}

/// Scalar tail for 4:4:4 encode (pixels not covered by SIMD blocks).
/// Uses the same 15-bit fixed-point integer math as the SIMD kernels
/// so all dispatch tiers produce identical output.
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
    use crate::types::PREC;
    for i in start..end {
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
