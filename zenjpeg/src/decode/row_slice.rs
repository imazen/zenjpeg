//! Row slice types for the push-based callback decode API.
//!
//! [`RowSlice`] wraps a single row of decoded u8 pixels with format metadata,
//! providing typed accessors for safe use in callbacks. [`RowSliceF32`] is the
//! equivalent for f32 pixel data.

use crate::types::PixelFormat;

/// A single row of decoded u8 pixels passed to the [`decode_rows`](super::DecodeConfig::decode_rows) callback.
///
/// Borrows data from the decoder's internal strip buffer (zero-copy).
/// The slice is only valid for the duration of the callback invocation.
///
/// # Example
///
/// ```rust,ignore
/// decoder.decode_rows(&jpeg_data, PixelFormat::Rgb, |row| {
///     let rgb = row.as_rgb();
///     for pixel in rgb {
///         // Process each pixel...
///     }
///     Ok(())
/// }, enough::Unstoppable)?;
/// ```
pub struct RowSlice<'a> {
    data: &'a [u8],
    row_index: usize,
    width: usize,
    format: PixelFormat,
}

impl<'a> RowSlice<'a> {
    /// Creates a new `RowSlice`.
    pub(super) fn new(data: &'a [u8], row_index: usize, width: usize, format: PixelFormat) -> Self {
        Self {
            data,
            row_index,
            width,
            format,
        }
    }

    /// Returns the raw pixel bytes.
    ///
    /// Length is `width * format.bytes_per_pixel()`.
    #[inline]
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        self.data
    }

    /// Returns the row as RGB pixels.
    ///
    /// # Panics
    ///
    /// Panics if the format is not [`PixelFormat::Rgb`].
    #[inline]
    #[must_use]
    pub fn as_rgb(&self) -> &[rgb::RGB<u8>] {
        assert_eq!(
            self.format,
            PixelFormat::Rgb,
            "as_rgb() called on {:?} row",
            self.format
        );
        bytemuck::cast_slice(self.data)
    }

    /// Returns the row as RGBA pixels.
    ///
    /// # Panics
    ///
    /// Panics if the format is not [`PixelFormat::Rgba`].
    #[inline]
    #[must_use]
    pub fn as_rgba(&self) -> &[rgb::RGBA<u8>] {
        assert_eq!(
            self.format,
            PixelFormat::Rgba,
            "as_rgba() called on {:?} row",
            self.format
        );
        bytemuck::cast_slice(self.data)
    }

    /// Returns the row as grayscale bytes.
    ///
    /// # Panics
    ///
    /// Panics if the format is not [`PixelFormat::Gray`].
    #[inline]
    #[must_use]
    pub fn as_gray(&self) -> &[u8] {
        assert_eq!(
            self.format,
            PixelFormat::Gray,
            "as_gray() called on {:?} row",
            self.format
        );
        self.data
    }

    /// Returns the 0-based row index within the output image (respects crop).
    #[inline]
    #[must_use]
    pub fn row_index(&self) -> usize {
        self.row_index
    }

    /// Returns the width in pixels.
    #[inline]
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Returns the pixel format.
    #[inline]
    #[must_use]
    pub fn format(&self) -> PixelFormat {
        self.format
    }
}

/// A single row of decoded f32 pixels passed to the [`decode_rows_f32`](super::DecodeConfig::decode_rows_f32) callback.
///
/// Borrows data from the decoder's internal strip buffer (zero-copy).
/// The slice is only valid for the duration of the callback invocation.
pub struct RowSliceF32<'a> {
    data: &'a [f32],
    row_index: usize,
    width: usize,
    format: PixelFormat,
}

impl<'a> RowSliceF32<'a> {
    /// Creates a new `RowSliceF32`.
    pub(super) fn new(
        data: &'a [f32],
        row_index: usize,
        width: usize,
        format: PixelFormat,
    ) -> Self {
        Self {
            data,
            row_index,
            width,
            format,
        }
    }

    /// Returns the raw f32 pixel data.
    ///
    /// Length is `width * format.num_channels()` (e.g., `width * 4` for RgbaF32).
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        self.data
    }

    /// Returns the 0-based row index within the output image (respects crop).
    #[inline]
    #[must_use]
    pub fn row_index(&self) -> usize {
        self.row_index
    }

    /// Returns the width in pixels.
    #[inline]
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Returns the pixel format.
    #[inline]
    #[must_use]
    pub fn format(&self) -> PixelFormat {
        self.format
    }
}
