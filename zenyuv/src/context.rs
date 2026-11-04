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
use crate::types::{Matrix, Range, Strides, check_plane, rgb_pixel_dims};

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
    pub(crate) range: Range,
    pub(crate) matrix: Matrix,
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

    // ── Conversions ─────────────────────────────────────────────────────
    //
    // Every plane is a view at its own stride, in samples; padding is neither read
    // nor written, and the last row of a plane needs none. `rgb` is packed: its width
    // and stride count samples, three per pixel, so a row pitch need not be a whole
    // number of pixels. Each method checks the pixel limit and every plane against
    // `rgb`'s dimensions before writing anything, and allocates context-owned scratch
    // fallibly, so a refusal writes and allocates nothing.

    /// Box-average 4:2:0 encode. u8 output.
    pub fn encode_420_u8(
        &mut self,
        rgb: imgref::ImgRef<'_, u8>,
        mut y: imgref::ImgRefMut<'_, u8>,
        mut cb: imgref::ImgRefMut<'_, u8>,
        mut cr: imgref::ImgRefMut<'_, u8>,
    ) -> Result<(), YuvError> {
        let (strides, width, height) = self.geometry(rgb, &y, Some((&cb, &cr)), Chroma::Half)?;
        if width == 0 || height == 0 {
            return Ok(());
        }
        crate::encode::rgb_to_yuv420(
            rgb.buf(),
            y.buf_mut(),
            cb.buf_mut(),
            cr.buf_mut(),
            width,
            height,
            strides,
            self.range,
            self.matrix,
        );
        Ok(())
    }

    /// Y-only 4:2:0 encode. u8 output. Useful when the caller provides custom
    /// chroma downsampling (e.g., gamma-corrected averaging) and wants to
    /// avoid the cost of computing and writing chroma that will be overwritten.
    pub fn encode_420_y_only_u8(
        &mut self,
        rgb: imgref::ImgRef<'_, u8>,
        mut y: imgref::ImgRefMut<'_, u8>,
    ) -> Result<(), YuvError> {
        let (strides, width, height) = self.geometry::<u8>(rgb, &y, None, Chroma::Half)?;
        if width == 0 || height == 0 {
            return Ok(());
        }
        crate::encode::rgb_to_yuv420_y_only(
            rgb.buf(),
            y.buf_mut(),
            width,
            height,
            strides.rgb,
            strides.y,
            self.range,
            self.matrix,
        );
        Ok(())
    }

    /// Box-average 4:4:4 encode. u8 output.
    pub fn encode_444_u8(
        &mut self,
        rgb: imgref::ImgRef<'_, u8>,
        mut y: imgref::ImgRefMut<'_, u8>,
        mut cb: imgref::ImgRefMut<'_, u8>,
        mut cr: imgref::ImgRefMut<'_, u8>,
    ) -> Result<(), YuvError> {
        let (strides, width, height) = self.geometry(rgb, &y, Some((&cb, &cr)), Chroma::Full)?;
        if width == 0 || height == 0 {
            return Ok(());
        }
        crate::encode::rgb_to_yuv444(
            rgb.buf(),
            y.buf_mut(),
            cb.buf_mut(),
            cr.buf_mut(),
            width,
            height,
            strides,
            self.range,
            self.matrix,
        );
        Ok(())
    }

    /// Box-average 4:2:0 encode. f32 output (for zenjpeg's DCT pipeline).
    /// Currently goes through u8 internally; shaped for a future native f32 kernel.
    pub fn encode_420_f32(
        &mut self,
        rgb: imgref::ImgRef<'_, u8>,
        mut y: imgref::ImgRefMut<'_, f32>,
        mut cb: imgref::ImgRefMut<'_, f32>,
        mut cr: imgref::ImgRefMut<'_, f32>,
    ) -> Result<(), YuvError> {
        let (strides, width, height) = self.geometry(rgb, &y, Some((&cb, &cr)), Chroma::Half)?;
        if width == 0 || height == 0 {
            return Ok(());
        }
        let (cw, ch) = (width.div_ceil(2), height.div_ceil(2));
        let (n, c_size) = (width * height, cw * ch);
        self.try_reserve_f32_temps(n, c_size)?;
        let (range, matrix) = (self.range, self.matrix);
        let Some(temps) = self.f32_temps.as_mut() else {
            return Err(YuvError::OutOfMemory);
        };
        crate::encode::rgb_to_yuv420(
            rgb.buf(),
            &mut temps.y[..n],
            &mut temps.cb[..c_size],
            &mut temps.cr[..c_size],
            width,
            height,
            Strides {
                rgb: strides.rgb,
                y: width,
                cb: cw,
                cr: cw,
            },
            range,
            matrix,
        );
        u8_rows_to_f32(&temps.y[..n], width, &mut y);
        u8_rows_to_f32(&temps.cb[..c_size], cw, &mut cb);
        u8_rows_to_f32(&temps.cr[..c_size], cw, &mut cr);
        Ok(())
    }

    // ── Sharp YUV ───────────────────────────────────────────────────────

    /// Sharp 4:2:0 encode. u8 output.
    ///
    /// The sharp kernel still makes a per-call scratch allocation of up to two
    /// bytes per pixel, for chroma it discards, bounded by the pixel limit.
    pub fn encode_sharp_420_u8(
        &mut self,
        rgb: imgref::ImgRef<'_, u8>,
        y: imgref::ImgRefMut<'_, u8>,
        cb: imgref::ImgRefMut<'_, u8>,
        cr: imgref::ImgRefMut<'_, u8>,
        config: &SharpYuvConfig,
    ) -> Result<(), YuvError> {
        let (_, width, height) = self.geometry(rgb, &y, Some((&cb, &cr)), Chroma::Half)?;
        if width == 0 || height == 0 {
            return Ok(());
        }
        self.try_reserve_sharp_ws(width.div_ceil(2))?;
        let (range, matrix) = (self.range, self.matrix);
        let Some(ws) = self.sharp_ws.as_mut() else {
            return Err(YuvError::OutOfMemory);
        };
        crate::sharp::rgb_to_yuv420_sharp(rgb, y, cb, cr, range, matrix, config, ws);
        Ok(())
    }

    /// Sharp 4:2:0 encode. f32 output: Y computed via the u8 kernel and widened,
    /// Cb/Cr written directly from the iteration workspace.
    ///
    /// The sharp kernel still makes a per-call scratch allocation of one byte per
    /// pixel, bounded by the pixel limit.
    pub fn encode_sharp_420_f32(
        &mut self,
        rgb: imgref::ImgRef<'_, u8>,
        y: imgref::ImgRefMut<'_, f32>,
        cb: imgref::ImgRefMut<'_, f32>,
        cr: imgref::ImgRefMut<'_, f32>,
        config: &SharpYuvConfig,
    ) -> Result<(), YuvError> {
        let (_, width, height) = self.geometry(rgb, &y, Some((&cb, &cr)), Chroma::Half)?;
        if width == 0 || height == 0 {
            return Ok(());
        }
        self.try_reserve_sharp_ws(width.div_ceil(2))?;
        let (range, matrix) = (self.range, self.matrix);
        let Some(ws) = self.sharp_ws.as_mut() else {
            return Err(YuvError::OutOfMemory);
        };
        crate::sharp::rgb_to_yuv420_sharp_f32(rgb, y, cb, cr, range, matrix, config, ws);
        Ok(())
    }

    /// Check the pixel limit, then every plane against `rgb`'s dimensions, in that
    /// order. Returns the strides, a left-out plane's as 0, and the pixel dimensions.
    fn geometry<T>(
        &self,
        rgb: imgref::ImgRef<'_, u8>,
        y: &imgref::ImgRefMut<'_, T>,
        chroma: Option<(&imgref::ImgRefMut<'_, T>, &imgref::ImgRefMut<'_, T>)>,
        subsampling: Chroma,
    ) -> Result<(Strides, usize, usize), YuvError> {
        let (sample_width, height) = (rgb.width(), rgb.height());
        let (width, height) = rgb_pixel_dims(sample_width, height)?;
        self.check_pixel_limit(width, height)?;
        let dims = |p: &imgref::ImgRefMut<'_, T>| (p.width(), p.height());
        check_plane(
            "rgb",
            rgb.buf().len(),
            (sample_width, height),
            rgb.stride(),
            (sample_width, height),
        )?;
        check_plane("y", y.buf().len(), dims(y), y.stride(), (width, height))?;
        let mut strides = Strides {
            rgb: rgb.stride(),
            y: y.stride(),
            cb: 0,
            cr: 0,
        };
        if let Some((cb, cr)) = chroma {
            let needed = match subsampling {
                Chroma::Full => (width, height),
                Chroma::Half => (width.div_ceil(2), height.div_ceil(2)),
            };
            check_plane("cb", cb.buf().len(), dims(cb), cb.stride(), needed)?;
            check_plane("cr", cr.buf().len(), dims(cr), cr.stride(), needed)?;
            strides.cb = cb.stride();
            strides.cr = cr.stride();
        }
        Ok((strides, width, height))
    }

    /// Refuse an image over the pixel limit.
    fn check_pixel_limit(&self, width: usize, height: usize) -> Result<(), YuvError> {
        let pixels = (width as u64)
            .checked_mul(height as u64)
            .ok_or(YuvError::Overflow)?;
        if pixels > self.max_pixels {
            return Err(YuvError::TooManyPixels {
                pixels,
                max: self.max_pixels,
            });
        }
        width.checked_mul(height).ok_or(YuvError::Overflow)?;
        Ok(())
    }

    /// Grow the f32-path temps fallibly.
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

    /// Grow the sharp workspace fallibly.
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
}

/// Widen tight u8 rows of `src`, `width` samples each, into `dst`'s rows.
fn u8_rows_to_f32(src: &[u8], width: usize, dst: &mut imgref::ImgRefMut<'_, f32>) {
    if width == 0 {
        return;
    }
    for (row, out) in src.chunks_exact(width).zip(dst.rows_mut()) {
        u8_to_f32(row, out);
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
