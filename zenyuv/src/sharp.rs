//! Sharp YUV: iterative perceptual chroma optimization.
//!
//! Minimizes reconstruction error in gamma-encoded RGB space by iteratively
//! adjusting Cb/Cr values. Same algorithm as libwebp's SharpYUV, parameterized
//! over gamma transfer function, color matrix, and range.
//!
//! No traits — gamma LUTs are passed as `&GammaLuts`, matrix coefficients as
//! `&ForwardCoeffs` / `&InverseCoeffs`. All functions are `#[inline(always)]`
//! so they can be called from `#[arcane]` SIMD regions in the future.

extern crate alloc;

use crate::types::{ForwardCoeffs, InverseCoeffs, Matrix, Range};

/// Configuration for Sharp YUV chroma optimization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SharpYuvConfig {
    /// Maximum Newton-step iterations per 2×2 block (default: 2).
    /// Uses the L2-optimal step with the correct inverse-matrix Jacobian.
    /// 2 iterations gives better quality than 4 iterations of the
    /// traditional forward-matrix gradient approach.
    pub max_iterations: u32,
    /// Stop early if total reconstruction error drops below this (default: 0.1).
    pub convergence_threshold: f32,
    /// After chroma refinement, run a Y refinement pass to compensate for luma
    /// error introduced by chroma subsampling (default: true).
    ///
    /// When chroma is subsampled to 4:2:0 and upsampled back during decode,
    /// the reconstructed RGB has slightly wrong luma at chroma edges. This pass
    /// adjusts Y to compensate, matching the approach in libwebp's
    /// `SharpYuvUpdateY`.
    pub refine_y: bool,
}

impl Default for SharpYuvConfig {
    fn default() -> Self {
        Self {
            max_iterations: 2,
            convergence_threshold: 0.1,
            refine_y: true,
        }
    }
}

/// Like `rgb_to_yuv420_sharp` but takes a pre-allocated workspace. Use via
/// `YuvContext::encode_sharp_420_*` for automatic workspace management.
pub fn rgb_to_yuv420_sharp_with_workspace(
    rgb: &[u8],
    y: &mut [u8],
    cb: &mut [u8],
    cr: &mut [u8],
    width: usize,
    height: usize,
    range: Range,
    matrix: Matrix,
    config: &SharpYuvConfig,
    ws: &mut SharpYuvWorkspace,
) {
    let n = width * height;
    let cw = width.div_ceil(2);
    let ch = height.div_ceil(2);
    assert!(rgb.len() >= n * 3);
    assert!(y.len() >= n);
    assert!(cb.len() >= cw * ch);
    assert!(cr.len() >= cw * ch);
    assert!(ws.chroma_width() >= cw);

    crate::encode::rgb_to_yuv444_y_only(rgb, y, width, height, range, matrix);

    let fwd = ForwardCoeffs::new(matrix, range);
    let inv = InverseCoeffs::new(matrix, range);

    sharp_iterate_rows_u8(
        rgb, y, cb, cr, width, height, cw, ch, &fwd, &inv, config, ws,
    );
}

/// f32 output: Y via fast SIMD (u8→f32 widen), Cb/Cr directly from iteration f32.
///
/// Y still goes through the SIMD u8 kernel (AVX2 pmaddwd) because that's faster
/// than scalar f32 per-pixel. The u8 Y is widened to f32 for the caller's output
/// AND used by the iteration SoA. Cb/Cr skip u8 entirely — written from the
/// iteration workspace as f32.
pub fn rgb_to_yuv420_sharp_f32(
    rgb: &[u8],
    y_f32: &mut [f32],
    cb_f32: &mut [f32],
    cr_f32: &mut [f32],
    width: usize,
    height: usize,
    range: Range,
    matrix: Matrix,
    config: &SharpYuvConfig,
    ws: &mut SharpYuvWorkspace,
) {
    let n = width * height;
    let cw = width.div_ceil(2);
    let ch = height.div_ceil(2);
    assert!(rgb.len() >= n * 3);
    assert!(y_f32.len() >= n);
    assert!(cb_f32.len() >= cw * ch);
    assert!(cr_f32.len() >= cw * ch);
    assert!(ws.chroma_width() >= cw);

    // Y via fast SIMD → u8 temp, then widen to f32 for caller.
    // This is faster than scalar f32 Y despite the u8→f32 step, because
    // the AVX2 pmaddwd kernel processes 32 pixels/iter vs 1 pixel/iter scalar.
    let mut y_u8 = alloc::vec![0u8; n];
    crate::encode::rgb_to_yuv444_y_only(rgb, &mut y_u8, width, height, range, matrix);
    for i in 0..n {
        y_f32[i] = y_u8[i] as f32;
    }

    let fwd = ForwardCoeffs::new(matrix, range);
    let inv = InverseCoeffs::new(matrix, range);

    // Cb/Cr: iteration produces f32 directly. SoA reads Y from u8 (via workspace).
    sharp_iterate_rows_f32_hybrid(
        rgb, &y_u8, cb_f32, cr_f32, width, height, cw, ch, &fwd, &inv, config, ws,
    );
}

/// Convert packed RGB to Y/Cb/Cr 4:2:0 with Sharp YUV chroma optimization.
///
/// Y is computed at full resolution via the fast SIMD path. Cb/Cr are computed
/// per 2×2 block using iterative refinement that minimizes reconstruction error
/// in gamma-encoded RGB space.
///
/// `luts` controls the gamma transfer (sRGB or libwebp). `range` and `matrix`
/// control the YCbCr color space. `config` tunes the iteration.
pub fn rgb_to_yuv420_sharp(
    rgb: &[u8],
    y: &mut [u8],
    cb: &mut [u8],
    cr: &mut [u8],
    width: usize,
    height: usize,
    range: Range,
    matrix: Matrix,
    config: &SharpYuvConfig,
) {
    let n = width * height;
    let cw = width.div_ceil(2);
    let ch = height.div_ceil(2);
    assert!(rgb.len() >= n * 3);
    assert!(y.len() >= n);
    assert!(cb.len() >= cw * ch);
    assert!(cr.len() >= cw * ch);

    let fwd = ForwardCoeffs::new(matrix, range);
    let inv = InverseCoeffs::new(matrix, range);

    // Compute Y at full resolution via fast SIMD path.
    crate::encode::rgb_to_yuv444_y_only(rgb, y, width, height, range, matrix);

    // Per-row SoA workspace. Caller can pre-allocate via SharpYuvWorkspace
    // and pass it across strip calls to avoid per-call allocation.
    let mut workspace = SharpYuvWorkspace::new(cw);
    sharp_iterate_rows_u8(
        rgb,
        y,
        cb,
        cr,
        width,
        height,
        cw,
        ch,
        &fwd,
        &inv,
        config,
        &mut workspace,
    );
}

/// Refine pre-computed Cb/Cr using Sharp YUV iteration.
///
/// Unlike `rgb_to_yuv420_sharp`, this does NOT compute initial Cb/Cr from the
/// forward matrix. Instead, it takes the caller's pre-computed Cb/Cr as the
/// starting point and runs Newton-step refinement to minimize reconstruction
/// error in sRGB space.
///
/// Use this when the initial chroma was computed with a different averaging
/// model (e.g., gamma-corrected averaging) but the reconstruction should
/// still use the standard BT.601/BT.709 inverse matrix.
///
/// Y must be at full resolution (`width * height`). Cb/Cr must be at chroma
/// resolution (`ceil(width/2) * ceil(height/2)`).
pub fn refine_chroma_420_u8(
    rgb: &[u8],
    y: &[u8],
    cb: &mut [u8],
    cr: &mut [u8],
    width: usize,
    height: usize,
    range: Range,
    matrix: Matrix,
    config: &SharpYuvConfig,
) {
    let n = width * height;
    let cw = width.div_ceil(2);
    let ch = height.div_ceil(2);
    assert!(rgb.len() >= n * 3);
    assert!(y.len() >= n);
    assert!(cb.len() >= cw * ch);
    assert!(cr.len() >= cw * ch);

    let inv = InverseCoeffs::new(matrix, range);
    let fwd = ForwardCoeffs::new(matrix, range);

    let mut ws = SharpYuvWorkspace::new(cw);
    refine_iterate_rows_u8(
        rgb, y, cb, cr, width, height, cw, ch, &fwd, &inv, config, &mut ws,
    );
}

/// Like `refine_chroma_420_u8` but takes a pre-allocated workspace.
pub fn refine_chroma_420_u8_with_workspace(
    rgb: &[u8],
    y: &[u8],
    cb: &mut [u8],
    cr: &mut [u8],
    width: usize,
    height: usize,
    range: Range,
    matrix: Matrix,
    config: &SharpYuvConfig,
    ws: &mut SharpYuvWorkspace,
) {
    let n = width * height;
    let cw = width.div_ceil(2);
    let ch = height.div_ceil(2);
    assert!(rgb.len() >= n * 3);
    assert!(y.len() >= n);
    assert!(cb.len() >= cw * ch);
    assert!(cr.len() >= cw * ch);
    assert!(ws.chroma_width() >= cw);

    let inv = InverseCoeffs::new(matrix, range);
    let fwd = ForwardCoeffs::new(matrix, range);

    refine_iterate_rows_u8(
        rgb, y, cb, cr, width, height, cw, ch, &fwd, &inv, config, ws,
    );
}

/// Refine Y plane to compensate for luma error from chroma subsampling.
///
/// After Cb/Cr are refined, the reconstruction `inv(Y, upsample(Cb), upsample(Cr))`
/// produces slightly wrong luma at chroma edges because the upsampled chroma
/// doesn't perfectly reconstruct the original RGB. This function adjusts each Y
/// value to compensate, matching libwebp's `SharpYuvUpdateY` approach.
///
/// Algorithm per pixel:
/// 1. Upsample Cb/Cr from the 4:2:0 grid (nearest-neighbor to the co-sited position)
/// 2. Reconstruct RGB via the inverse color matrix: `RGB_rec = inv(Y, Cb_up, Cr_up)`
/// 3. Compute forward luma of both original and reconstructed RGB
/// 4. Adjust: `Y += clamp(target_luma - reconstructed_luma, -delta_max, +delta_max)`
///
/// Returns the total absolute luma error (sum of |adjustments|) for convergence
/// checking.
pub fn refine_y_420_u8(
    rgb: &[u8],
    y: &mut [u8],
    cb: &[u8],
    cr: &[u8],
    width: usize,
    height: usize,
    range: Range,
    matrix: Matrix,
) -> u64 {
    let n = width * height;
    let cw = width.div_ceil(2);
    let ch = height.div_ceil(2);
    assert!(rgb.len() >= n * 3);
    assert!(y.len() >= n);
    assert!(cb.len() >= cw * ch);
    assert!(cr.len() >= cw * ch);

    let fwd = ForwardCoeffs::new(matrix, range);
    let inv = InverseCoeffs::new(matrix, range);

    refine_y_rows(rgb, y, cb, cr, width, height, cw, ch, &fwd, &inv, range)
}

/// Inner loop: refine Y row by row.
fn refine_y_rows(
    rgb: &[u8],
    y: &mut [u8],
    cb: &[u8],
    cr: &[u8],
    width: usize,
    height: usize,
    cw: usize,
    _ch: usize,
    fwd: &ForwardCoeffs,
    inv: &InverseCoeffs,
    range: Range,
) -> u64 {
    let uv_center = inv.uv_offset.abs(); // 128.0

    // Forward luma coefficients for computing target/reconstructed luma.
    let yr = fwd.yr_f;
    let yg = fwd.yg_f;
    let yb = fwd.yb_f;
    let y_bias = fwd.y_bias_f;

    // Inverse matrix coefficients.
    let y_coeff = inv.y_coeff;
    let y_off = inv.y_offset;
    let cr_to_r = inv.cr_to_r;
    let cr_to_g = inv.cr_to_g;
    let cb_to_g = inv.cb_to_g;
    let cb_to_b = inv.cb_to_b;

    // Y range limits derived from the range parameter.
    let (y_min, y_max) = match range {
        Range::Full => (0.0f32, 255.0f32),
        Range::Limited => (16.0f32, 235.0f32),
    };

    let mut total_diff: u64 = 0;

    for row in 0..height {
        let cy = row / 2; // chroma row (nearest-neighbor vertical)

        for col in 0..width {
            let cx = col / 2; // chroma column (nearest-neighbor horizontal)
            let chroma_idx = cy * cw + cx;

            // Upsampled chroma (nearest-neighbor from 4:2:0 grid).
            let cb_val = cb[chroma_idx] as f32;
            let cr_val = cr[chroma_idx] as f32;
            let cb_c = cb_val - uv_center;
            let cr_c = cr_val - uv_center;

            // Current Y value.
            let pixel_idx = row * width + col;
            let y_cur = y[pixel_idx] as f32;

            // Reconstruct RGB from current YCbCr.
            let y_adj = y_coeff * (y_cur + y_off);
            let rec_r = (y_adj + cr_to_r * cr_c).clamp(0.0, 255.0);
            let rec_g = (y_adj + cr_to_g * cr_c + cb_to_g * cb_c).clamp(0.0, 255.0);
            let rec_b = (y_adj + cb_to_b * cb_c).clamp(0.0, 255.0);

            // Forward luma of reconstructed RGB.
            let rec_luma = yr * rec_r + yg * rec_g + yb * rec_b + y_bias;

            // Original RGB.
            let rgb_idx = pixel_idx * 3;
            let orig_r = rgb[rgb_idx] as f32;
            let orig_g = rgb[rgb_idx + 1] as f32;
            let orig_b = rgb[rgb_idx + 2] as f32;

            // Forward luma of original RGB (this is the target Y).
            let target_luma = yr * orig_r + yg * orig_g + yb * orig_b + y_bias;

            // Luma error: how much Y needs to shift to compensate.
            let diff = target_luma - rec_luma;

            // Apply bounded adjustment to Y.
            let new_y = (y_cur + diff).clamp(y_min, y_max);
            y[pixel_idx] = new_y.round() as u8;

            total_diff += diff.abs() as u64;
        }
    }

    total_diff
}

/// Pre-allocated workspace for sharp YUV iteration. Reuse across calls to
/// avoid per-strip allocation overhead (~36KB per call for 1024px width).
pub struct SharpYuvWorkspace {
    y0s: alloc::vec::Vec<f32>,
    y1s: alloc::vec::Vec<f32>,
    y2s: alloc::vec::Vec<f32>,
    y3s: alloc::vec::Vec<f32>,
    or0: alloc::vec::Vec<f32>,
    og0: alloc::vec::Vec<f32>,
    ob0: alloc::vec::Vec<f32>,
    or1: alloc::vec::Vec<f32>,
    og1: alloc::vec::Vec<f32>,
    ob1: alloc::vec::Vec<f32>,
    or2: alloc::vec::Vec<f32>,
    og2: alloc::vec::Vec<f32>,
    ob2: alloc::vec::Vec<f32>,
    or3: alloc::vec::Vec<f32>,
    og3: alloc::vec::Vec<f32>,
    ob3: alloc::vec::Vec<f32>,
    cb_f: alloc::vec::Vec<f32>,
    cr_f: alloc::vec::Vec<f32>,
}

impl SharpYuvWorkspace {
    /// Returns the chroma width this workspace is sized for.
    pub fn chroma_width(&self) -> usize {
        self.y0s.len()
    }

    /// Allocate workspace for chroma width `cw`.
    pub fn new(cw: usize) -> Self {
        Self {
            y0s: alloc::vec![0.0f32; cw],
            y1s: alloc::vec![0.0f32; cw],
            y2s: alloc::vec![0.0f32; cw],
            y3s: alloc::vec![0.0f32; cw],
            or0: alloc::vec![0.0f32; cw],
            og0: alloc::vec![0.0f32; cw],
            ob0: alloc::vec![0.0f32; cw],
            or1: alloc::vec![0.0f32; cw],
            og1: alloc::vec![0.0f32; cw],
            ob1: alloc::vec![0.0f32; cw],
            or2: alloc::vec![0.0f32; cw],
            og2: alloc::vec![0.0f32; cw],
            ob2: alloc::vec![0.0f32; cw],
            or3: alloc::vec![0.0f32; cw],
            og3: alloc::vec![0.0f32; cw],
            ob3: alloc::vec![0.0f32; cw],
            cb_f: alloc::vec![0.0f32; cw],
            cr_f: alloc::vec![0.0f32; cw],
        }
    }
}

/// Refinement iteration: loads initial Cb/Cr from caller's buffers, extracts
/// SoA data, runs Newton steps, writes refined Cb/Cr back.
fn refine_iterate_rows_u8(
    rgb: &[u8],
    y: &[u8],
    cb: &mut [u8],
    cr: &mut [u8],
    width: usize,
    height: usize,
    cw: usize,
    ch: usize,
    fwd: &ForwardCoeffs,
    inv: &InverseCoeffs,
    config: &SharpYuvConfig,
    ws: &mut SharpYuvWorkspace,
) {
    for cy_idx in 0..ch {
        let row_top = cy_idx * 2;
        let row_bot = (row_top + 1).min(height - 1);
        // Extract SoA data (Y positions + original RGB) without computing
        // initial Cb/Cr — we use the caller's pre-computed values instead.
        extract_soa_row_no_chroma(
            rgb,
            y,
            row_top,
            row_bot,
            width,
            cw,
            &mut ws.y0s,
            &mut ws.y1s,
            &mut ws.y2s,
            &mut ws.y3s,
            &mut ws.or0,
            &mut ws.og0,
            &mut ws.ob0,
            &mut ws.or1,
            &mut ws.og1,
            &mut ws.ob1,
            &mut ws.or2,
            &mut ws.og2,
            &mut ws.ob2,
            &mut ws.or3,
            &mut ws.og3,
            &mut ws.ob3,
        );
        // Load caller's pre-computed Cb/Cr as f32 initial values.
        let row_off = cy_idx * cw;
        for cx_idx in 0..cw {
            ws.cb_f[cx_idx] = cb[row_off + cx_idx] as f32;
            ws.cr_f[cx_idx] = cr[row_off + cx_idx] as f32;
        }
        sharp_iterate_all_blocks(
            &ws.y0s[..cw],
            &ws.y1s[..cw],
            &ws.y2s[..cw],
            &ws.y3s[..cw],
            &ws.or0[..cw],
            &ws.og0[..cw],
            &ws.ob0[..cw],
            &ws.or1[..cw],
            &ws.og1[..cw],
            &ws.ob1[..cw],
            &ws.or2[..cw],
            &ws.og2[..cw],
            &ws.ob2[..cw],
            &ws.or3[..cw],
            &ws.og3[..cw],
            &ws.ob3[..cw],
            &mut ws.cb_f[..cw],
            &mut ws.cr_f[..cw],
            inv,
            fwd,
            config.max_iterations,
            config.convergence_threshold,
        );
        for cx_idx in 0..cw {
            cb[row_off + cx_idx] = clamp_u8(ws.cb_f[cx_idx]);
            cr[row_off + cx_idx] = clamp_u8(ws.cr_f[cx_idx]);
        }
    }
}

/// Core iteration loop, outputting u8 Cb/Cr (with clamp+round narrowing).
fn sharp_iterate_rows_u8(
    rgb: &[u8],
    y: &[u8],
    cb: &mut [u8],
    cr: &mut [u8],
    width: usize,
    height: usize,
    cw: usize,
    ch: usize,
    fwd: &ForwardCoeffs,
    inv: &InverseCoeffs,
    config: &SharpYuvConfig,
    ws: &mut SharpYuvWorkspace,
) {
    for cy_idx in 0..ch {
        let row_top = cy_idx * 2;
        let row_bot = (row_top + 1).min(height - 1);
        extract_soa_row(
            rgb,
            y,
            row_top,
            row_bot,
            width,
            cw,
            &mut ws.y0s,
            &mut ws.y1s,
            &mut ws.y2s,
            &mut ws.y3s,
            &mut ws.or0,
            &mut ws.og0,
            &mut ws.ob0,
            &mut ws.or1,
            &mut ws.og1,
            &mut ws.ob1,
            &mut ws.or2,
            &mut ws.og2,
            &mut ws.ob2,
            &mut ws.or3,
            &mut ws.og3,
            &mut ws.ob3,
            &mut ws.cb_f,
            &mut ws.cr_f,
            fwd,
        );
        sharp_iterate_all_blocks(
            &ws.y0s[..cw],
            &ws.y1s[..cw],
            &ws.y2s[..cw],
            &ws.y3s[..cw],
            &ws.or0[..cw],
            &ws.og0[..cw],
            &ws.ob0[..cw],
            &ws.or1[..cw],
            &ws.og1[..cw],
            &ws.ob1[..cw],
            &ws.or2[..cw],
            &ws.og2[..cw],
            &ws.ob2[..cw],
            &ws.or3[..cw],
            &ws.og3[..cw],
            &ws.ob3[..cw],
            &mut ws.cb_f[..cw],
            &mut ws.cr_f[..cw],
            inv,
            fwd,
            config.max_iterations,
            config.convergence_threshold,
        );
        let row_off = cy_idx * cw;
        for cx_idx in 0..cw {
            cb[row_off + cx_idx] = clamp_u8(ws.cb_f[cx_idx]);
            cr[row_off + cx_idx] = clamp_u8(ws.cr_f[cx_idx]);
        }
    }
}

/// Hybrid f32: Y from u8 SIMD kernel (via SoA), Cb/Cr written as f32 directly.
fn sharp_iterate_rows_f32_hybrid(
    rgb: &[u8],
    y_u8: &[u8],
    cb_f32: &mut [f32],
    cr_f32: &mut [f32],
    width: usize,
    height: usize,
    cw: usize,
    ch: usize,
    fwd: &ForwardCoeffs,
    inv: &InverseCoeffs,
    config: &SharpYuvConfig,
    ws: &mut SharpYuvWorkspace,
) {
    for cy_idx in 0..ch {
        let row_top = cy_idx * 2;
        let row_bot = (row_top + 1).min(height - 1);
        extract_soa_row(
            rgb,
            y_u8,
            row_top,
            row_bot,
            width,
            cw,
            &mut ws.y0s,
            &mut ws.y1s,
            &mut ws.y2s,
            &mut ws.y3s,
            &mut ws.or0,
            &mut ws.og0,
            &mut ws.ob0,
            &mut ws.or1,
            &mut ws.og1,
            &mut ws.ob1,
            &mut ws.or2,
            &mut ws.og2,
            &mut ws.ob2,
            &mut ws.or3,
            &mut ws.og3,
            &mut ws.ob3,
            &mut ws.cb_f,
            &mut ws.cr_f,
            fwd,
        );
        sharp_iterate_all_blocks(
            &ws.y0s[..cw],
            &ws.y1s[..cw],
            &ws.y2s[..cw],
            &ws.y3s[..cw],
            &ws.or0[..cw],
            &ws.og0[..cw],
            &ws.ob0[..cw],
            &ws.or1[..cw],
            &ws.og1[..cw],
            &ws.ob1[..cw],
            &ws.or2[..cw],
            &ws.og2[..cw],
            &ws.ob2[..cw],
            &ws.or3[..cw],
            &ws.og3[..cw],
            &ws.ob3[..cw],
            &mut ws.cb_f[..cw],
            &mut ws.cr_f[..cw],
            inv,
            fwd,
            config.max_iterations,
            config.convergence_threshold,
        );
        // Write Cb/Cr as f32 directly — no u8 narrowing.
        let row_off = cy_idx * cw;
        cb_f32[row_off..row_off + cw].copy_from_slice(&ws.cb_f[..cw]);
        cr_f32[row_off..row_off + cw].copy_from_slice(&ws.cr_f[..cw]);
    }
}

/// Extract one chroma row of 2×2 block data into SoA arrays + box-average
/// initial Cb/Cr.
///
/// The bulk path processes the inner blocks where `x+1 < width` (no edge
/// clamping), using contiguous row slices cast to `&[u8; N]` so LLVM can
/// prove no aliasing and vectorize the u8→f32 + FMA chain.
///
/// The last block (if width is odd) handles edge replication separately.
fn extract_soa_row(
    rgb: &[u8],
    y_plane: &[u8],
    row_top: usize,
    row_bot: usize,
    width: usize,
    cw: usize,
    y0s: &mut [f32],
    y1s: &mut [f32],
    y2s: &mut [f32],
    y3s: &mut [f32],
    or0: &mut [f32],
    og0: &mut [f32],
    ob0: &mut [f32],
    or1: &mut [f32],
    og1: &mut [f32],
    ob1: &mut [f32],
    or2: &mut [f32],
    og2: &mut [f32],
    ob2: &mut [f32],
    or3: &mut [f32],
    og3: &mut [f32],
    ob3: &mut [f32],
    cb_f: &mut [f32],
    cr_f: &mut [f32],
    fwd: &ForwardCoeffs,
) {
    // Pre-slice the two Y rows and two RGB rows as contiguous spans.
    // This eliminates per-pixel index arithmetic and lets LLVM see
    // contiguous memory access patterns.
    let y_top = &y_plane[row_top * width..row_top * width + width];
    let y_bot = &y_plane[row_bot * width..row_bot * width + width];
    let rgb_top = &rgb[row_top * width * 3..row_top * width * 3 + width * 3];
    let rgb_bot = &rgb[row_bot * width * 3..row_bot * width * 3 + width * 3];

    // Bulk: all blocks where x+1 < width (no edge clamping needed).
    // For width=1024, bulk_cw=512 (all blocks). For width=1023, bulk_cw=511.
    let bulk_cw = width / 2;

    extract_soa_bulk(
        rgb_top, rgb_bot, y_top, y_bot, bulk_cw, y0s, y1s, y2s, y3s, or0, og0, ob0, or1, og1, ob1,
        or2, og2, ob2, or3, og3, ob3, cb_f, cr_f, fwd,
    );

    // Edge: last block if width is odd (x1 = x0, replicates last column).
    if cw > bulk_cw {
        let cx = bulk_cw;
        let x0 = cx * 2;
        // x1 clamped to width-1 = x0 for odd width
        y0s[cx] = y_top[x0] as f32;
        y1s[cx] = y_top[x0] as f32; // replicate
        y2s[cx] = y_bot[x0] as f32;
        y3s[cx] = y_bot[x0] as f32;
        let ri = x0 * 3;
        or0[cx] = rgb_top[ri] as f32;
        og0[cx] = rgb_top[ri + 1] as f32;
        ob0[cx] = rgb_top[ri + 2] as f32;
        or1[cx] = or0[cx]; // replicate
        og1[cx] = og0[cx];
        ob1[cx] = ob0[cx];
        or2[cx] = rgb_bot[ri] as f32;
        og2[cx] = rgb_bot[ri + 1] as f32;
        ob2[cx] = rgb_bot[ri + 2] as f32;
        or3[cx] = or2[cx];
        og3[cx] = og2[cx];
        ob3[cx] = ob2[cx];
        let r_avg = (or0[cx] + or1[cx] + or2[cx] + or3[cx]) * 0.25;
        let g_avg = (og0[cx] + og1[cx] + og2[cx] + og3[cx]) * 0.25;
        let b_avg = (ob0[cx] + ob1[cx] + ob2[cx] + ob3[cx]) * 0.25;
        cb_f[cx] = fwd.cb_r_f * r_avg + fwd.cb_g_f * g_avg + fwd.cb_b_f * b_avg + fwd.uv_bias_f;
        cr_f[cx] = fwd.cr_r_f * r_avg + fwd.cr_g_f * g_avg + fwd.cr_b_f * b_avg + fwd.uv_bias_f;
    }
}

/// Extract one chroma row of 2×2 block data into SoA arrays WITHOUT computing
/// initial Cb/Cr. Used by `refine_chroma_420_u8` where the caller provides
/// pre-computed chroma values.
fn extract_soa_row_no_chroma(
    rgb: &[u8],
    y_plane: &[u8],
    row_top: usize,
    row_bot: usize,
    width: usize,
    cw: usize,
    y0s: &mut [f32],
    y1s: &mut [f32],
    y2s: &mut [f32],
    y3s: &mut [f32],
    or0: &mut [f32],
    og0: &mut [f32],
    ob0: &mut [f32],
    or1: &mut [f32],
    og1: &mut [f32],
    ob1: &mut [f32],
    or2: &mut [f32],
    og2: &mut [f32],
    ob2: &mut [f32],
    or3: &mut [f32],
    og3: &mut [f32],
    ob3: &mut [f32],
) {
    let y_top = &y_plane[row_top * width..row_top * width + width];
    let y_bot = &y_plane[row_bot * width..row_bot * width + width];
    let rgb_top = &rgb[row_top * width * 3..row_top * width * 3 + width * 3];
    let rgb_bot = &rgb[row_bot * width * 3..row_bot * width * 3 + width * 3];

    let bulk_cw = width / 2;

    extract_soa_bulk_no_chroma(
        rgb_top, rgb_bot, y_top, y_bot, bulk_cw, y0s, y1s, y2s, y3s, or0, og0, ob0, or1, og1, ob1,
        or2, og2, ob2, or3, og3, ob3,
    );

    // Edge: last block if width is odd.
    if cw > bulk_cw {
        let cx = bulk_cw;
        let x0 = cx * 2;
        y0s[cx] = y_top[x0] as f32;
        y1s[cx] = y_top[x0] as f32;
        y2s[cx] = y_bot[x0] as f32;
        y3s[cx] = y_bot[x0] as f32;
        let ri = x0 * 3;
        or0[cx] = rgb_top[ri] as f32;
        og0[cx] = rgb_top[ri + 1] as f32;
        ob0[cx] = rgb_top[ri + 2] as f32;
        or1[cx] = or0[cx];
        og1[cx] = og0[cx];
        ob1[cx] = ob0[cx];
        or2[cx] = rgb_bot[ri] as f32;
        og2[cx] = rgb_bot[ri + 1] as f32;
        ob2[cx] = rgb_bot[ri + 2] as f32;
        or3[cx] = or2[cx];
        og3[cx] = og2[cx];
        ob3[cx] = ob2[cx];
    }
}

/// Bulk SoA extraction without chroma computation.
#[archmage::autoversion]
fn extract_soa_bulk_no_chroma(
    rgb_top: &[u8],
    rgb_bot: &[u8],
    y_top: &[u8],
    y_bot: &[u8],
    bulk_cw: usize,
    y0s: &mut [f32],
    y1s: &mut [f32],
    y2s: &mut [f32],
    y3s: &mut [f32],
    or0: &mut [f32],
    og0: &mut [f32],
    ob0: &mut [f32],
    or1: &mut [f32],
    og1: &mut [f32],
    ob1: &mut [f32],
    or2: &mut [f32],
    og2: &mut [f32],
    ob2: &mut [f32],
    or3: &mut [f32],
    og3: &mut [f32],
    ob3: &mut [f32],
) {
    for cx in 0..bulk_cw {
        let x0 = cx * 2;
        y0s[cx] = y_top[x0] as f32;
        y1s[cx] = y_top[x0 + 1] as f32;
        y2s[cx] = y_bot[x0] as f32;
        y3s[cx] = y_bot[x0 + 1] as f32;

        let ri = x0 * 3;
        or0[cx] = rgb_top[ri] as f32;
        og0[cx] = rgb_top[ri + 1] as f32;
        ob0[cx] = rgb_top[ri + 2] as f32;
        or1[cx] = rgb_top[ri + 3] as f32;
        og1[cx] = rgb_top[ri + 4] as f32;
        ob1[cx] = rgb_top[ri + 5] as f32;
        or2[cx] = rgb_bot[ri] as f32;
        og2[cx] = rgb_bot[ri + 1] as f32;
        ob2[cx] = rgb_bot[ri + 2] as f32;
        or3[cx] = rgb_bot[ri + 3] as f32;
        og3[cx] = rgb_bot[ri + 4] as f32;
        ob3[cx] = rgb_bot[ri + 5] as f32;
    }
}

/// Bulk inner loop: contiguous row slices, no edge checks.
/// `#[autoversion]` generates AVX2/SSE/NEON variants. LLVM can vectorize
/// the u8→f32 widening and the box-average FMA because:
/// - Input slices are pre-bounded (no per-pixel bounds checks)
/// - Each block reads at stride 6 in RGB / stride 2 in Y (regular pattern)
/// - Output slices are written sequentially (no aliasing between them)
#[archmage::autoversion]
fn extract_soa_bulk(
    rgb_top: &[u8],
    rgb_bot: &[u8],
    y_top: &[u8],
    y_bot: &[u8],
    bulk_cw: usize,
    y0s: &mut [f32],
    y1s: &mut [f32],
    y2s: &mut [f32],
    y3s: &mut [f32],
    or0: &mut [f32],
    og0: &mut [f32],
    ob0: &mut [f32],
    or1: &mut [f32],
    og1: &mut [f32],
    ob1: &mut [f32],
    or2: &mut [f32],
    og2: &mut [f32],
    ob2: &mut [f32],
    or3: &mut [f32],
    og3: &mut [f32],
    ob3: &mut [f32],
    cb_f: &mut [f32],
    cr_f: &mut [f32],
    fwd: &ForwardCoeffs,
) {
    // Coefficients as locals so LLVM can hoist them to registers.
    let cb_r = fwd.cb_r_f;
    let cb_g = fwd.cb_g_f;
    let cb_b = fwd.cb_b_f;
    let cr_r = fwd.cr_r_f;
    let cr_g = fwd.cr_g_f;
    let cr_b = fwd.cr_b_f;
    let uv_bias = fwd.uv_bias_f;

    for cx in 0..bulk_cw {
        // Pixel coordinates. x0 = 2*cx, x1 = 2*cx+1. Both in bounds
        // because bulk_cw = width/2, so x1 = 2*cx+1 < width.
        let x0 = cx * 2;

        // Y: stride-2 in the pre-sliced row.
        y0s[cx] = y_top[x0] as f32;
        y1s[cx] = y_top[x0 + 1] as f32;
        y2s[cx] = y_bot[x0] as f32;
        y3s[cx] = y_bot[x0 + 1] as f32;

        // RGB: stride-6 in the pre-sliced row (3 bytes/pixel, 2 pixels/block).
        let ri = x0 * 3;
        let r0 = rgb_top[ri] as f32;
        let g0 = rgb_top[ri + 1] as f32;
        let b0 = rgb_top[ri + 2] as f32;
        let r1 = rgb_top[ri + 3] as f32;
        let g1 = rgb_top[ri + 4] as f32;
        let b1 = rgb_top[ri + 5] as f32;
        let r2 = rgb_bot[ri] as f32;
        let g2 = rgb_bot[ri + 1] as f32;
        let b2 = rgb_bot[ri + 2] as f32;
        let r3 = rgb_bot[ri + 3] as f32;
        let g3 = rgb_bot[ri + 4] as f32;
        let b3 = rgb_bot[ri + 5] as f32;

        or0[cx] = r0;
        og0[cx] = g0;
        ob0[cx] = b0;
        or1[cx] = r1;
        og1[cx] = g1;
        ob1[cx] = b1;
        or2[cx] = r2;
        og2[cx] = g2;
        ob2[cx] = b2;
        or3[cx] = r3;
        og3[cx] = g3;
        ob3[cx] = b3;

        // Box-average initial Cb/Cr.
        let r_avg = (r0 + r1 + r2 + r3) * 0.25;
        let g_avg = (g0 + g1 + g2 + g3) * 0.25;
        let b_avg = (b0 + b1 + b2 + b3) * 0.25;
        cb_f[cx] = cb_r * r_avg + cb_g * g_avg + cb_b * b_avg + uv_bias;
        cr_f[cx] = cr_r * r_avg + cr_g * g_avg + cr_b * b_avg + uv_bias;
    }
}

/// Iterative refinement over a row of blocks. Processes 8 blocks in parallel
/// via magetypes `f32x8` — same codegen on AVX2 (native), NEON (2×f32x4),
/// WASM128 (2×f32x4), and scalar (8×f32).
///
/// ## L2-optimal Newton step (math)
///
/// We minimize L2 reconstruction error: `L = Σ_pixels ||orig_RGB - inv(Y, Cb, Cr)||²`
///
/// For Cb, only G and B channels are affected (∂R/∂Cb = 0 in the BT.601 inverse):
///   ∂L/∂Cb = -2 Σ [err_g · cb_to_g + err_b · cb_to_b]
///   ∂²L/∂Cb² = 2 · 4 · (cb_to_g² + cb_to_b²)    [4 pixels per block]
///
/// Newton step: δCb = -∂L/∂Cb / ∂²L/∂Cb²
///            = Σ(err_g · cb_to_g + err_b · cb_to_b) / (4 · (cb_to_g² + cb_to_b²))
///
/// This is exact for the unclamped L2 problem. Clamping to [0, 255] handles
/// out-of-gamut cases. Converges in 2 iterations vs 4+ for the forward-matrix
/// gradient with hand-tuned damping.
///
/// Similarly for Cr: ∂R/∂Cr = cr_to_r, ∂G/∂Cr = cr_to_g, ∂B/∂Cr = 0.
fn sharp_iterate_all_blocks(
    y0: &[f32],
    y1: &[f32],
    y2: &[f32],
    y3: &[f32],
    or0: &[f32],
    og0: &[f32],
    ob0: &[f32],
    or1: &[f32],
    og1: &[f32],
    ob1: &[f32],
    or2: &[f32],
    og2: &[f32],
    ob2: &[f32],
    or3: &[f32],
    og3: &[f32],
    ob3: &[f32],
    cb_f: &mut [f32],
    cr_f: &mut [f32],
    inv: &InverseCoeffs,
    _fwd: &ForwardCoeffs,
    max_iterations: u32,
    _convergence_threshold: f32,
) {
    incant!(sharp_iterate_simd(
        y0,
        y1,
        y2,
        y3,
        or0,
        og0,
        ob0,
        or1,
        og1,
        ob1,
        or2,
        og2,
        ob2,
        or3,
        og3,
        ob3,
        cb_f,
        cr_f,
        inv,
        max_iterations
    ));
}

use archmage::prelude::*;
use magetypes::simd::generic::f32x8 as GenericF32x8;

#[magetypes(v3, neon, wasm128, scalar)]
#[inline(always)]
fn sharp_iterate_simd(
    token: Token,
    y0: &[f32],
    y1: &[f32],
    y2: &[f32],
    y3: &[f32],
    or0: &[f32],
    og0: &[f32],
    ob0: &[f32],
    or1: &[f32],
    og1: &[f32],
    ob1: &[f32],
    or2: &[f32],
    og2: &[f32],
    ob2: &[f32],
    or3: &[f32],
    og3: &[f32],
    ob3: &[f32],
    cb_f: &mut [f32],
    cr_f: &mut [f32],
    inv: &InverseCoeffs,
    max_iterations: u32,
) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;

    let n = cb_f.len();
    let chunks = n / 8;

    let uv_center_v = f32x8::splat(token, inv.uv_offset.abs());
    let y_coeff_v = f32x8::splat(token, inv.y_coeff);
    let y_off_v = f32x8::splat(token, inv.y_offset);
    let cr_to_r_v = f32x8::splat(token, inv.cr_to_r);
    let cr_to_g_v = f32x8::splat(token, inv.cr_to_g);
    let cb_to_g_v = f32x8::splat(token, inv.cb_to_g);
    let cb_to_b_v = f32x8::splat(token, inv.cb_to_b);
    let zero_v = f32x8::splat(token, 0.0);
    let max_v = f32x8::splat(token, 255.0);

    let cb_denom = 4.0 * (inv.cb_to_g * inv.cb_to_g + inv.cb_to_b * inv.cb_to_b);
    let cr_denom = 4.0 * (inv.cr_to_r * inv.cr_to_r + inv.cr_to_g * inv.cr_to_g);
    let cb_inv_denom_v = f32x8::splat(token, 1.0 / cb_denom);
    let cr_inv_denom_v = f32x8::splat(token, 1.0 / cr_denom);

    for _ in 0..max_iterations {
        // SIMD: 8 blocks at a time.
        for c in 0..chunks {
            let base = c * 8;
            let cb_v = f32x8::from_slice(token, &cb_f[base..]);
            let cr_v = f32x8::from_slice(token, &cr_f[base..]);
            let cb_c = cb_v - uv_center_v;
            let cr_c = cr_v - uv_center_v;
            let mut cb_num = f32x8::splat(token, 0.0);
            let mut cr_num = f32x8::splat(token, 0.0);

            // 4 pixel positions, unrolled. Each loads 8 values from the SoA arrays.
            macro_rules! pixel {
                ($y:expr, $or:expr, $og:expr, $ob:expr) => {{
                    let yv = f32x8::from_slice(token, &$y[base..]);
                    let y_adj = y_coeff_v * (yv + y_off_v);
                    let rec_r = (y_adj + cr_to_r_v * cr_c).max(zero_v).min(max_v);
                    let rec_g = (y_adj + cr_to_g_v * cr_c + cb_to_g_v * cb_c)
                        .max(zero_v)
                        .min(max_v);
                    let rec_b = (y_adj + cb_to_b_v * cb_c).max(zero_v).min(max_v);
                    let or_v = f32x8::from_slice(token, &$or[base..]);
                    let og_v = f32x8::from_slice(token, &$og[base..]);
                    let ob_v = f32x8::from_slice(token, &$ob[base..]);
                    let er = or_v - rec_r;
                    let eg = og_v - rec_g;
                    let eb = ob_v - rec_b;
                    cb_num = cb_num + eg * cb_to_g_v + eb * cb_to_b_v;
                    cr_num = cr_num + er * cr_to_r_v + eg * cr_to_g_v;
                }};
            }
            pixel!(y0, or0, og0, ob0);
            pixel!(y1, or1, og1, ob1);
            pixel!(y2, or2, og2, ob2);
            pixel!(y3, or3, og3, ob3);

            let new_cb = (cb_v + cb_num * cb_inv_denom_v).max(zero_v).min(max_v);
            let new_cr = (cr_v + cr_num * cr_inv_denom_v).max(zero_v).min(max_v);

            // Store back. from_slice loads 8, we need store 8.
            let cb_arr = new_cb.to_array();
            let cr_arr = new_cr.to_array();
            cb_f[base..base + 8].copy_from_slice(&cb_arr);
            cr_f[base..base + 8].copy_from_slice(&cr_arr);
        }

        // Scalar tail for remaining blocks.
        let tail_start = chunks * 8;
        let uv_center = inv.uv_offset.abs();
        for i in tail_start..n {
            let cb_c = cb_f[i] - uv_center;
            let cr_c = cr_f[i] - uv_center;
            let mut cb_num = 0.0f32;
            let mut cr_num = 0.0f32;
            macro_rules! pixel_scalar {
                ($yv:expr, $or:expr, $og:expr, $ob:expr) => {{
                    let y_adj = inv.y_coeff * ($yv + inv.y_offset);
                    let rec_r = (y_adj + inv.cr_to_r * cr_c).clamp(0.0, 255.0);
                    let rec_g = (y_adj + inv.cr_to_g * cr_c + inv.cb_to_g * cb_c).clamp(0.0, 255.0);
                    let rec_b = (y_adj + inv.cb_to_b * cb_c).clamp(0.0, 255.0);
                    let er = $or - rec_r;
                    let eg = $og - rec_g;
                    let eb = $ob - rec_b;
                    cb_num += eg * inv.cb_to_g + eb * inv.cb_to_b;
                    cr_num += er * inv.cr_to_r + eg * inv.cr_to_g;
                }};
            }
            pixel_scalar!(y0[i], or0[i], og0[i], ob0[i]);
            pixel_scalar!(y1[i], or1[i], og1[i], ob1[i]);
            pixel_scalar!(y2[i], or2[i], og2[i], ob2[i]);
            pixel_scalar!(y3[i], or3[i], og3[i], ob3[i]);
            cb_f[i] = (cb_f[i] + cb_num / cb_denom).clamp(0.0, 255.0);
            cr_f[i] = (cr_f[i] + cr_num / cr_denom).clamp(0.0, 255.0);
        }
    }
}

fn clamp_u8(v: f32) -> u8 {
    v.round().clamp(0.0, 255.0) as u8
}
