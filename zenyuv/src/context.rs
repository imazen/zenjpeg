//! Reusable encoder context with pre-allocated buffers and coefficients.
//!
//! `YuvContext` is method-agnostic: the caller picks box-average vs sharp per
//! call. Temp buffers and sharp workspace are reused across calls; the sharp
//! workspace is lazy-allocated on first use.
//!
//! Supports both u8 output (zenwebp) and f32 output (zenjpeg) — the f32
//! variants currently go through u8 internally (same SIMD kernel) but the API
//! is shaped so we can add a native f32 kernel later without changing callers.

extern crate alloc;
use alloc::vec::Vec;

use crate::gamma::GammaLuts;
use crate::sharp::{SharpYuvConfig, SharpYuvWorkspace};
use crate::types::{ForwardCoeffs, InverseCoeffs, Matrix, Range};

/// Reusable YUV encoder context. Holds precomputed coefficients and gamma LUTs.
///
/// All internal buffers are lazy — allocated on first use of the path that
/// needs them. Non-sharp u8 callers pay zero allocation overhead.
///
/// - u8 box-average: zero alloc (writes directly to caller's buffers)
/// - f32 box-average: lazy u8 temp buffers (for u8→f32 conversion)
/// - u8/f32 sharp: lazy u8 temps + lazy sharp workspace (18 f32 arrays)
pub struct YuvContext {
    pub(crate) fwd: ForwardCoeffs,
    pub(crate) inv: InverseCoeffs,
    range: Range,
    matrix: Matrix,
    /// Gamma LUTs (1KB, always present — cheap).
    luts: GammaLuts,
    /// Temp u8 planes — only allocated for f32 output paths.
    f32_temps: Option<F32Temps>,
    /// Sharp workspace — only allocated on first sharp call.
    sharp_ws: Option<SharpYuvWorkspace>,
}

/// Temp u8 planes for f32 output paths (encode u8 → convert to f32).
struct F32Temps {
    y: Vec<u8>,
    cb: Vec<u8>,
    cr: Vec<u8>,
    max_pixels: usize,
    max_c_size: usize,
}

impl YuvContext {
    /// Create a context for the given color space, sized for images up to
    /// `max_width × max_strip_height` pixels per call.
    ///
    /// For whole-image encoding, set `max_strip_height = image_height`.
    /// For strip encoding (zenjpeg), set `max_strip_height = strip_height` (typically 16).
    /// Create a context for the given color space.
    /// No buffers allocated until first use — zero cost for u8 box-average.
    pub fn new(range: Range, matrix: Matrix) -> Self {
        Self {
            fwd: ForwardCoeffs::new(matrix, range),
            inv: InverseCoeffs::new(matrix, range),
            range,
            matrix,
            luts: GammaLuts::srgb(),
            f32_temps: None,
            sharp_ws: None,
        }
    }

    /// Set gamma LUTs (default is sRGB). Call `GammaLuts::libwebp()` for VP8.
    pub fn set_gamma_luts(&mut self, luts: GammaLuts) {
        self.luts = luts;
    }

    // ── Box-average (non-sharp) ─────────────────────────────────────────

    /// Box-average 4:2:0 encode. u8 output.
    pub fn encode_420_u8(
        &mut self,
        rgb: &[u8],
        y: &mut [u8],
        cb: &mut [u8],
        cr: &mut [u8],
        width: usize,
        height: usize,
    ) {
        crate::encode::rgb_to_yuv420_with(rgb, y, cb, cr, width, height, self.range, self.matrix);
    }

    /// Box-average 4:4:4 encode. u8 output.
    pub fn encode_444_u8(
        &mut self,
        rgb: &[u8],
        y: &mut [u8],
        cb: &mut [u8],
        cr: &mut [u8],
        width: usize,
        height: usize,
    ) {
        crate::encode::rgb_to_yuv444_with(rgb, y, cb, cr, width, height, self.range, self.matrix);
    }

    /// Box-average 4:2:0 encode. f32 output (for zenjpeg's DCT pipeline).
    /// Currently goes through u8 internally; shaped for a future native f32 kernel.
    pub fn encode_420_f32(
        &mut self,
        rgb: &[u8],
        y: &mut [f32],
        cb: &mut [f32],
        cr: &mut [f32],
        width: usize,
        height: usize,
    ) {
        let n = width * height;
        let cw = width.div_ceil(2);
        let ch = height.div_ceil(2);
        let c_size = cw * ch;
        let range = self.range;
        let matrix = self.matrix;
        let temps = self.ensure_f32_temps(n, c_size);
        crate::encode::rgb_to_yuv420_with(
            rgb, &mut temps.y[..n], &mut temps.cb[..c_size], &mut temps.cr[..c_size],
            width, height, range, matrix,
        );
        u8_to_f32(&temps.y[..n], &mut y[..n]);
        u8_to_f32(&temps.cb[..c_size], &mut cb[..c_size]);
        u8_to_f32(&temps.cr[..c_size], &mut cr[..c_size]);
    }

    // ── Sharp YUV ───────────────────────────────────────────────────────

    /// Sharp 4:2:0 encode. u8 output.
    pub fn encode_sharp_420_u8(
        &mut self,
        rgb: &[u8],
        y: &mut [u8],
        cb: &mut [u8],
        cr: &mut [u8],
        width: usize,
        height: usize,
        config: &SharpYuvConfig,
    ) {
        let cw = width.div_ceil(2);
        self.ensure_sharp_ws(cw);
        crate::sharp::rgb_to_yuv420_sharp_with_workspace(
            rgb, y, cb, cr, width, height,
            self.range, self.matrix,
            &self.luts, config,
            self.sharp_ws.as_mut().unwrap(),
        );
    }

    /// Sharp 4:2:0 encode. f32 output. No u8 intermediate — Y computed as f32,
    /// Cb/Cr written directly from the iteration workspace.
    pub fn encode_sharp_420_f32(
        &mut self,
        rgb: &[u8],
        y: &mut [f32],
        cb: &mut [f32],
        cr: &mut [f32],
        width: usize,
        height: usize,
        config: &SharpYuvConfig,
    ) {
        let cw = width.div_ceil(2);
        self.ensure_sharp_ws(cw);
        let range = self.range;
        let matrix = self.matrix;
        crate::sharp::rgb_to_yuv420_sharp_f32(
            rgb, y, cb, cr, width, height,
            range, matrix,
            &self.luts, config,
            self.sharp_ws.as_mut().unwrap(),
        );
    }

    /// Lazy-allocate sharp workspace on first use.
    fn ensure_sharp_ws(&mut self, cw: usize) {
        if self.sharp_ws.is_none() || self.sharp_ws.as_ref().unwrap().chroma_width() < cw {
            self.sharp_ws = Some(SharpYuvWorkspace::new(cw));
        }
    }

    /// Lazy-allocate f32 temp buffers on first use.
    fn ensure_f32_temps(&mut self, max_pixels: usize, max_c_size: usize) -> &mut F32Temps {
        if self.f32_temps.is_none()
            || self.f32_temps.as_ref().unwrap().max_pixels < max_pixels
            || self.f32_temps.as_ref().unwrap().max_c_size < max_c_size
        {
            self.f32_temps = Some(F32Temps {
                y: alloc::vec![0u8; max_pixels],
                cb: alloc::vec![0u8; max_c_size],
                cr: alloc::vec![0u8; max_c_size],
                max_pixels,
                max_c_size,
            });
        }
        self.f32_temps.as_mut().unwrap()
    }
}

#[inline]
fn u8_to_f32(src: &[u8], dst: &mut [f32]) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d = *s as f32;
    }
}
