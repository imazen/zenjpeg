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

use crate::YuvError;
use crate::sharp::{SharpYuvConfig, SharpYuvWorkspace};
use crate::types::{Matrix, Range};

/// The pixel limit (`width × height`) the `try_*` entry points enforce
/// unless [`YuvContext::with_max_pixels`] sets another: 120 MP, above
/// common 108 MP camera output.
pub const DEFAULT_MAX_PIXELS: u64 = 120_000_000;

/// Reusable YUV encoder context. Holds precomputed coefficients and gamma LUTs.
///
/// All internal buffers are lazy — allocated on first use of the path that
/// needs them. Non-sharp u8 callers pay zero allocation overhead.
///
/// - u8 box-average: zero alloc (writes directly to caller's buffers)
/// - f32 box-average: lazy u8 temp buffers (for u8→f32 conversion)
/// - u8/f32 sharp: lazy u8 temps + lazy sharp workspace (18 f32 arrays)
pub struct YuvContext {
    range: Range,
    matrix: Matrix,
    /// Pixel limit for the `try_*` entry points.
    max_pixels: u64,
    /// Temp u8 planes — heap-allocated lazily for f32 output paths.
    f32_temps: Option<alloc::boxed::Box<F32Temps>>,
    /// Sharp workspace — heap-allocated lazily on first sharp call (432+ bytes).
    sharp_ws: Option<alloc::boxed::Box<SharpYuvWorkspace>>,
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
    /// Create a context for the given color space.
    ///
    /// No buffers are allocated until first use — zero cost for u8
    /// box-average. The f32 temps and the sharp workspace are sized lazily
    /// from the dimensions of the first call that needs them and regrown on
    /// demand; there is no up-front `max_width × max_strip_height` sizing.
    pub fn new(range: Range, matrix: Matrix) -> Self {
        Self {
            range,
            matrix,
            max_pixels: DEFAULT_MAX_PIXELS,
            f32_temps: None,
            sharp_ws: None,
        }
    }

    /// Set the pixel limit (`width × height`) the `try_*` entry points
    /// enforce. The default is [`DEFAULT_MAX_PIXELS`]. The infallible
    /// methods do not check it.
    #[must_use]
    pub fn with_max_pixels(mut self, max_pixels: u64) -> Self {
        self.max_pixels = max_pixels;
        self
    }

    // ── Box-average (non-sharp) ─────────────────────────────────────────

    /// Box-average 4:2:0 encode. u8 output.
    ///
    /// # Panics
    ///
    /// If a buffer is shorter than the dimensions need.
    /// [`Self::try_encode_420_u8`] returns an error instead.
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

    /// Y-only 4:2:0 encode. u8 output. Useful when the caller provides custom
    /// chroma downsampling (e.g., gamma-corrected averaging) and wants to
    /// avoid the cost of computing and writing chroma that will be overwritten.
    ///
    /// # Panics
    ///
    /// If a buffer is shorter than the dimensions need.
    /// [`Self::try_encode_420_y_only_u8`] returns an error instead.
    pub fn encode_420_y_only_u8(&mut self, rgb: &[u8], y: &mut [u8], width: usize, height: usize) {
        crate::encode::rgb_to_yuv420_y_only_with(rgb, y, width, height, self.range, self.matrix);
    }

    /// Box-average 4:4:4 encode. u8 output.
    ///
    /// # Panics
    ///
    /// If a buffer is shorter than the dimensions need.
    /// [`Self::try_encode_444_u8`] returns an error instead.
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
    ///
    /// # Panics
    ///
    /// If a buffer is shorter than the dimensions need.
    /// [`Self::try_encode_420_f32`] returns an error instead.
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
            rgb,
            &mut temps.y[..n],
            &mut temps.cb[..c_size],
            &mut temps.cr[..c_size],
            width,
            height,
            range,
            matrix,
        );
        u8_to_f32(&temps.y[..n], &mut y[..n]);
        u8_to_f32(&temps.cb[..c_size], &mut cb[..c_size]);
        u8_to_f32(&temps.cr[..c_size], &mut cr[..c_size]);
    }

    // ── Sharp YUV ───────────────────────────────────────────────────────

    /// Sharp 4:2:0 encode. u8 output.
    ///
    /// # Panics
    ///
    /// If a buffer is shorter than the dimensions need.
    /// [`Self::try_encode_sharp_420_u8`] returns an error instead.
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
        let range = self.range;
        let matrix = self.matrix;
        crate::sharp::rgb_to_yuv420_sharp_with_workspace(
            rgb,
            y,
            cb,
            cr,
            width,
            height,
            range,
            matrix,
            config,
            self.sharp_ws.as_mut().unwrap(),
        );
    }

    /// Sharp 4:2:0 encode. f32 output. No u8 intermediate — Y computed as f32,
    /// Cb/Cr written directly from the iteration workspace.
    ///
    /// # Panics
    ///
    /// If a buffer is shorter than the dimensions need.
    /// [`Self::try_encode_sharp_420_f32`] returns an error instead.
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
            rgb,
            y,
            cb,
            cr,
            width,
            height,
            range,
            matrix,
            config,
            self.sharp_ws.as_mut().unwrap(),
        );
    }

    /// Decode YUV 4:4:4 to RGB u8.
    #[allow(dead_code)]
    pub(crate) fn decode_444_to_rgb(
        &self,
        y: &[u8],
        cb: &[u8],
        cr: &[u8],
        rgb: &mut [u8],
        w: usize,
        h: usize,
    ) {
        crate::decode::yuv444_to_rgb_with(y, cb, cr, rgb, w, h, self.range, self.matrix);
    }

    /// Decode YUV 4:2:0 to RGB u8 (nearest-neighbor chroma upsampling).
    #[allow(dead_code)]
    pub(crate) fn decode_420_to_rgb(
        &self,
        y: &[u8],
        cb: &[u8],
        cr: &[u8],
        rgb: &mut [u8],
        w: usize,
        h: usize,
    ) {
        crate::decode::yuv420_to_rgb_with(y, cb, cr, rgb, w, h, self.range, self.matrix);
    }

    /// Decode YUV 4:2:0 to RGB u8 (bilinear chroma upsampling).
    #[allow(dead_code)]
    pub(crate) fn decode_420_bilinear_to_rgb(
        &self,
        y: &[u8],
        cb: &[u8],
        cr: &[u8],
        rgb: &mut [u8],
        w: usize,
        h: usize,
    ) {
        crate::decode::yuv420_to_rgb_bilinear_with(y, cb, cr, rgb, w, h, self.range, self.matrix);
    }

    /// Decode YUV 4:2:2 to RGB u8.
    #[allow(dead_code)]
    pub(crate) fn decode_422_to_rgb(
        &self,
        y: &[u8],
        cb: &[u8],
        cr: &[u8],
        rgb: &mut [u8],
        w: usize,
        h: usize,
    ) {
        crate::decode::yuv422_to_rgb_with(y, cb, cr, rgb, w, h, self.range, self.matrix);
    }

    /// Decode YUV 4:0:0 (grayscale) to RGB u8.
    #[allow(dead_code)]
    pub(crate) fn decode_400_to_rgb(&self, y: &[u8], rgb: &mut [u8], w: usize, h: usize) {
        crate::decode::yuv400_to_rgb_with(y, rgb, w, h, self.range, self.matrix);
    }

    // ── Fallible entry points ───────────────────────────────────────────
    //
    // Each checks the pixel limit and every buffer length the infallible
    // method would assert, and allocates any context-owned scratch
    // fallibly, before writing anything. Then it runs the infallible method,
    // so the output is the same bytes.

    /// [`Self::encode_420_u8`], returning an error instead of panicking.
    pub fn try_encode_420_u8(
        &mut self,
        rgb: &[u8],
        y: &mut [u8],
        cb: &mut [u8],
        cr: &mut [u8],
        width: usize,
        height: usize,
    ) -> Result<(), YuvError> {
        let d = self.dims(width, height, Chroma::Half)?;
        d.check(rgb.len(), y.len(), Some((cb.len(), cr.len())))?;
        self.encode_420_u8(rgb, y, cb, cr, width, height);
        Ok(())
    }

    /// [`Self::encode_420_y_only_u8`], returning an error instead of
    /// panicking.
    pub fn try_encode_420_y_only_u8(
        &mut self,
        rgb: &[u8],
        y: &mut [u8],
        width: usize,
        height: usize,
    ) -> Result<(), YuvError> {
        let d = self.dims(width, height, Chroma::Half)?;
        d.check(rgb.len(), y.len(), None)?;
        self.encode_420_y_only_u8(rgb, y, width, height);
        Ok(())
    }

    /// [`Self::encode_444_u8`], returning an error instead of panicking.
    pub fn try_encode_444_u8(
        &mut self,
        rgb: &[u8],
        y: &mut [u8],
        cb: &mut [u8],
        cr: &mut [u8],
        width: usize,
        height: usize,
    ) -> Result<(), YuvError> {
        let d = self.dims(width, height, Chroma::Full)?;
        d.check(rgb.len(), y.len(), Some((cb.len(), cr.len())))?;
        self.encode_444_u8(rgb, y, cb, cr, width, height);
        Ok(())
    }

    /// [`Self::encode_420_f32`], returning an error instead of panicking or
    /// aborting on a failed scratch allocation.
    pub fn try_encode_420_f32(
        &mut self,
        rgb: &[u8],
        y: &mut [f32],
        cb: &mut [f32],
        cr: &mut [f32],
        width: usize,
        height: usize,
    ) -> Result<(), YuvError> {
        let d = self.dims(width, height, Chroma::Half)?;
        d.check(rgb.len(), y.len(), Some((cb.len(), cr.len())))?;
        self.try_reserve_f32_temps(d.pixels, d.chroma)?;
        self.encode_420_f32(rgb, y, cb, cr, width, height);
        Ok(())
    }

    /// [`Self::encode_sharp_420_u8`], returning an error instead of
    /// panicking or aborting on a failed workspace allocation.
    ///
    /// The sharp kernel still makes per-call scratch allocations of up to
    /// two bytes per pixel, bounded by the pixel limit.
    #[allow(clippy::too_many_arguments)]
    pub fn try_encode_sharp_420_u8(
        &mut self,
        rgb: &[u8],
        y: &mut [u8],
        cb: &mut [u8],
        cr: &mut [u8],
        width: usize,
        height: usize,
        config: &SharpYuvConfig,
    ) -> Result<(), YuvError> {
        let d = self.dims(width, height, Chroma::Half)?;
        d.check(rgb.len(), y.len(), Some((cb.len(), cr.len())))?;
        self.try_reserve_sharp_ws(width.div_ceil(2))?;
        self.encode_sharp_420_u8(rgb, y, cb, cr, width, height, config);
        Ok(())
    }

    /// [`Self::encode_sharp_420_f32`], returning an error instead of
    /// panicking or aborting on a failed workspace allocation.
    ///
    /// The sharp kernel still makes a per-call scratch allocation of one
    /// byte per pixel, bounded by the pixel limit.
    #[allow(clippy::too_many_arguments)]
    pub fn try_encode_sharp_420_f32(
        &mut self,
        rgb: &[u8],
        y: &mut [f32],
        cb: &mut [f32],
        cr: &mut [f32],
        width: usize,
        height: usize,
        config: &SharpYuvConfig,
    ) -> Result<(), YuvError> {
        let d = self.dims(width, height, Chroma::Half)?;
        d.check(rgb.len(), y.len(), Some((cb.len(), cr.len())))?;
        self.try_reserve_sharp_ws(width.div_ceil(2))?;
        self.encode_sharp_420_f32(rgb, y, cb, cr, width, height, config);
        Ok(())
    }

    /// Pixel and plane sizes for `width × height`, within the pixel limit.
    fn dims(&self, width: usize, height: usize, chroma: Chroma) -> Result<Dims, YuvError> {
        let pixels = (width as u64)
            .checked_mul(height as u64)
            .ok_or(YuvError::Overflow)?;
        if pixels > self.max_pixels {
            return Err(YuvError::TooManyPixels {
                pixels,
                max: self.max_pixels,
            });
        }
        let n = width.checked_mul(height).ok_or(YuvError::Overflow)?;
        let c = match chroma {
            Chroma::Full => n,
            Chroma::Half => width
                .div_ceil(2)
                .checked_mul(height.div_ceil(2))
                .ok_or(YuvError::Overflow)?,
        };
        Ok(Dims {
            pixels: n,
            rgb: n.checked_mul(3).ok_or(YuvError::Overflow)?,
            chroma: c,
        })
    }

    /// Grow the f32-path temps fallibly, so `ensure_f32_temps` keeps them.
    fn try_reserve_f32_temps(
        &mut self,
        max_pixels: usize,
        max_c_size: usize,
    ) -> Result<(), YuvError> {
        if self
            .f32_temps
            .as_ref()
            .is_some_and(|t| t.max_pixels >= max_pixels && t.max_c_size >= max_c_size)
        {
            return Ok(());
        }
        let zeros = |n: usize| {
            let mut v = Vec::new();
            v.try_reserve_exact(n).map_err(|_| YuvError::OutOfMemory)?;
            v.resize(n, 0u8);
            Ok::<_, YuvError>(v)
        };
        self.f32_temps = Some(alloc::boxed::Box::new(F32Temps {
            y: zeros(max_pixels)?,
            cb: zeros(max_c_size)?,
            cr: zeros(max_c_size)?,
            max_pixels,
            max_c_size,
        }));
        Ok(())
    }

    /// Grow the sharp workspace fallibly, so `ensure_sharp_ws` keeps it.
    fn try_reserve_sharp_ws(&mut self, cw: usize) -> Result<(), YuvError> {
        if self
            .sharp_ws
            .as_ref()
            .is_some_and(|ws| ws.chroma_width() >= cw)
        {
            return Ok(());
        }
        let ws = SharpYuvWorkspace::try_new(cw).ok_or(YuvError::OutOfMemory)?;
        self.sharp_ws = Some(alloc::boxed::Box::new(ws));
        Ok(())
    }

    // ── Internal ────────────────────────────────────────────────────────

    /// Lazy-allocate sharp workspace on first use.
    fn ensure_sharp_ws(&mut self, cw: usize) {
        if self.sharp_ws.is_none() || self.sharp_ws.as_ref().unwrap().chroma_width() < cw {
            self.sharp_ws = Some(alloc::boxed::Box::new(SharpYuvWorkspace::new(cw)));
        }
    }

    /// Lazy-allocate f32 temp buffers on first use.
    fn ensure_f32_temps(&mut self, max_pixels: usize, max_c_size: usize) -> &mut F32Temps {
        if self.f32_temps.is_none()
            || self.f32_temps.as_ref().unwrap().max_pixels < max_pixels
            || self.f32_temps.as_ref().unwrap().max_c_size < max_c_size
        {
            self.f32_temps = Some(alloc::boxed::Box::new(F32Temps {
                y: alloc::vec![0u8; max_pixels],
                cb: alloc::vec![0u8; max_c_size],
                cr: alloc::vec![0u8; max_c_size],
                max_pixels,
                max_c_size,
            }));
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

/// Chroma plane size relative to luma.
#[derive(Clone, Copy)]
enum Chroma {
    Full,
    Half,
}

/// Element counts a conversion needs.
struct Dims {
    pixels: usize,
    rgb: usize,
    chroma: usize,
}

impl Dims {
    fn check(&self, rgb: usize, y: usize, chroma: Option<(usize, usize)>) -> Result<(), YuvError> {
        let need = |buffer, needed: usize, len: usize| {
            if len < needed {
                Err(YuvError::BufferTooSmall {
                    buffer,
                    needed,
                    len,
                })
            } else {
                Ok(())
            }
        };
        need("rgb", self.rgb, rgb)?;
        need("y", self.pixels, y)?;
        if let Some((cb, cr)) = chroma {
            need("cb", self.chroma, cb)?;
            need("cr", self.chroma, cr)?;
        }
        Ok(())
    }
}
