//! Encoder implementations for v2 API.

use std::io::Write;
use std::marker::PhantomData;

use enough::Stop;

use super::config::EncoderConfig;
use super::types::{PixelLayout, YCbCrPlanes};
use crate::error::{Error, Result};

/// Encoder for raw byte input with explicit pixel layout.
pub struct BytesEncoder {
    config: EncoderConfig,
    layout: PixelLayout,
    width: u32,
    height: u32,
    rows_pushed: u32,
    output: Vec<u8>,
    // TODO: Replace with actual encoding state
    pixel_buffer: Vec<u8>,
}

impl BytesEncoder {
    pub(crate) fn new(
        config: EncoderConfig,
        width: u32,
        height: u32,
        layout: PixelLayout,
    ) -> Result<Self> {
        // Validate dimensions
        if width == 0 || height == 0 {
            return Err(Error::InvalidDimensions {
                width,
                height,
                reason: "dimensions cannot be zero",
            });
        }

        // Check for overflow
        let pixel_count = (width as u64) * (height as u64);
        if pixel_count > u32::MAX as u64 {
            return Err(Error::InvalidDimensions {
                width,
                height,
                reason: "dimensions too large",
            });
        }

        Ok(Self {
            config,
            layout,
            width,
            height,
            rows_pushed: 0,
            output: Vec::new(),
            pixel_buffer: Vec::new(),
        })
    }

    /// Push rows with explicit stride.
    ///
    /// - `data`: Raw pixel bytes
    /// - `rows`: Number of scanlines to push
    /// - `stride_bytes`: Bytes per row in buffer (>= width * bytes_per_pixel)
    /// - `stop`: Cancellation token (use `enough::Never` if not needed)
    pub fn push(
        &mut self,
        data: &[u8],
        rows: usize,
        stride_bytes: usize,
        stop: impl Stop,
    ) -> Result<()> {
        // Check cancellation
        if stop.should_stop() {
            return Err(Error::Cancelled);
        }

        let bpp = self.layout.bytes_per_pixel();
        let min_stride = self.width as usize * bpp;

        // Validate stride
        if stride_bytes < min_stride {
            return Err(Error::StrideTooSmall {
                width: self.width,
                stride: stride_bytes,
            });
        }

        // Validate row count
        let new_total = self.rows_pushed + rows as u32;
        if new_total > self.height {
            return Err(Error::TooManyRows {
                height: self.height,
                pushed: new_total,
            });
        }

        // Validate buffer size
        let expected_size = rows * stride_bytes;
        if data.len() < expected_size {
            return Err(Error::InvalidBufferSize {
                expected: expected_size,
                actual: data.len(),
            });
        }

        // Copy row data (stripping padding if needed)
        for row in 0..rows {
            if stop.should_stop() {
                return Err(Error::Cancelled);
            }

            let src_start = row * stride_bytes;
            let src_end = src_start + min_stride;
            self.pixel_buffer.extend_from_slice(&data[src_start..src_end]);
        }

        self.rows_pushed = new_total;
        Ok(())
    }

    /// Push contiguous (packed) data.
    ///
    /// Stride is assumed to be `width * bytes_per_pixel`.
    /// Rows inferred from `data.len() / (width * bytes_per_pixel)`.
    pub fn push_packed(&mut self, data: &[u8], stop: impl Stop) -> Result<()> {
        let bpp = self.layout.bytes_per_pixel();
        let row_bytes = self.width as usize * bpp;

        if row_bytes == 0 {
            return Err(Error::InvalidDimensions {
                width: self.width,
                height: self.height,
                reason: "row size is zero",
            });
        }

        let rows = data.len() / row_bytes;
        if rows == 0 && !data.is_empty() {
            return Err(Error::InvalidBufferSize {
                expected: row_bytes,
                actual: data.len(),
            });
        }

        self.push(data, rows, row_bytes, stop)
    }

    // === Status ===

    /// Get image width.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Get image height.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Get number of rows pushed so far.
    #[must_use]
    pub fn rows_pushed(&self) -> u32 {
        self.rows_pushed
    }

    /// Get number of rows remaining.
    #[must_use]
    pub fn rows_remaining(&self) -> u32 {
        self.height - self.rows_pushed
    }

    /// Get the pixel layout.
    #[must_use]
    pub fn layout(&self) -> PixelLayout {
        self.layout
    }

    // === Finish ===

    /// Finish encoding, return JPEG bytes.
    pub fn finish(self) -> Result<Vec<u8>> {
        if self.rows_pushed != self.height {
            return Err(Error::IncompleteImage {
                height: self.height,
                pushed: self.rows_pushed,
            });
        }

        // TODO: Actually encode the image using the existing encoder infrastructure
        // For now, use the legacy encoder
        self.encode_with_legacy()
    }

    /// Finish encoding to Write destination.
    pub fn finish_to<W: Write>(self, mut output: W) -> Result<W> {
        let jpeg = self.finish()?;
        output.write_all(&jpeg)?;
        Ok(output)
    }

    /// Internal: encode using legacy encoder
    fn encode_with_legacy(self) -> Result<Vec<u8>> {
        use crate::JpegEncoder;
        use crate::quant::Quality as LegacyQuality;

        let quality = LegacyQuality::from_quality(self.config.quality.to_internal());
        let pixel_format = self.layout.to_legacy();
        let subsampling = match self.config.color_mode {
            super::types::ColorMode::YCbCr { subsampling } => subsampling.to_legacy(),
            super::types::ColorMode::Xyb { .. } => crate::types::Subsampling::S444,
            super::types::ColorMode::Grayscale => crate::types::Subsampling::S444,
        };

        let mut encoder = JpegEncoder::new(self.width, self.height)
            .quality(quality)
            .pixel_format(pixel_format)
            .subsampling(subsampling)
            .optimize_huffman(self.config.optimize_huffman)
            .chroma_downsampling(self.config.downsampling_method.to_legacy());

        if self.config.progressive {
            encoder = encoder.progressive(true);
        }

        if matches!(self.config.color_mode, super::types::ColorMode::Xyb { .. }) {
            encoder = encoder.use_xyb(true);
        }

        encoder.encode(&self.pixel_buffer)
    }
}

/// Marker trait for supported rgb crate pixel types.
pub trait Pixel: Copy + 'static + bytemuck::Pod {
    /// Equivalent PixelLayout for this type.
    const LAYOUT: PixelLayout;
}

// Implement Pixel for rgb crate types
impl Pixel for rgb::RGB<u8> {
    const LAYOUT: PixelLayout = PixelLayout::Rgb8Srgb;
}
impl Pixel for rgb::RGBA<u8> {
    const LAYOUT: PixelLayout = PixelLayout::Rgbx8Srgb;
}
impl Pixel for rgb::Bgr<u8> {
    const LAYOUT: PixelLayout = PixelLayout::Bgr8Srgb;
}
impl Pixel for rgb::Bgra<u8> {
    const LAYOUT: PixelLayout = PixelLayout::Bgrx8Srgb;
}
impl Pixel for rgb::Gray<u8> {
    const LAYOUT: PixelLayout = PixelLayout::Gray8Srgb;
}

impl Pixel for rgb::RGB<u16> {
    const LAYOUT: PixelLayout = PixelLayout::Rgb16Linear;
}
impl Pixel for rgb::RGBA<u16> {
    const LAYOUT: PixelLayout = PixelLayout::Rgbx16Linear;
}
impl Pixel for rgb::Gray<u16> {
    const LAYOUT: PixelLayout = PixelLayout::Gray16Linear;
}

impl Pixel for rgb::RGB<f32> {
    const LAYOUT: PixelLayout = PixelLayout::RgbF32Linear;
}
impl Pixel for rgb::RGBA<f32> {
    const LAYOUT: PixelLayout = PixelLayout::RgbxF32Linear;
}
impl Pixel for rgb::Gray<f32> {
    const LAYOUT: PixelLayout = PixelLayout::GrayF32Linear;
}

/// Encoder for rgb crate pixel types.
///
/// Type parameter P determines pixel layout at compile time.
/// For RGBA/BGRA types, 4th channel is ignored.
pub struct RgbEncoder<P: Pixel> {
    inner: BytesEncoder,
    _marker: PhantomData<P>,
}

impl<P: Pixel> RgbEncoder<P> {
    pub(crate) fn new(config: EncoderConfig, width: u32, height: u32) -> Result<Self> {
        let inner = BytesEncoder::new(config, width, height, P::LAYOUT)?;
        Ok(Self {
            inner,
            _marker: PhantomData,
        })
    }

    /// Push rows with explicit stride (in pixels).
    ///
    /// - `data`: Pixel slice
    /// - `rows`: Number of scanlines to push
    /// - `stride`: Pixels per row in buffer (>= width)
    /// - `stop`: Cancellation token
    pub fn push(
        &mut self,
        data: &[P],
        rows: usize,
        stride: usize,
        stop: impl Stop,
    ) -> Result<()> {
        let stride_bytes = stride * std::mem::size_of::<P>();
        let bytes = bytemuck::cast_slice(data);
        self.inner.push(bytes, rows, stride_bytes, stop)
    }

    /// Push contiguous (packed) data.
    ///
    /// Stride assumed to be `width`. Rows inferred from `data.len() / width`.
    pub fn push_packed(&mut self, data: &[P], stop: impl Stop) -> Result<()> {
        let bytes = bytemuck::cast_slice(data);
        self.inner.push_packed(bytes, stop)
    }

    // === Status ===

    /// Get image width.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.inner.width()
    }

    /// Get image height.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.inner.height()
    }

    /// Get number of rows pushed so far.
    #[must_use]
    pub fn rows_pushed(&self) -> u32 {
        self.inner.rows_pushed()
    }

    /// Get number of rows remaining.
    #[must_use]
    pub fn rows_remaining(&self) -> u32 {
        self.inner.rows_remaining()
    }

    // === Finish ===

    /// Finish encoding, return JPEG bytes.
    pub fn finish(self) -> Result<Vec<u8>> {
        self.inner.finish()
    }

    /// Finish encoding to Write destination.
    pub fn finish_to<W: Write>(self, output: W) -> Result<W> {
        self.inner.finish_to(output)
    }
}

/// Encoder for planar f32 YCbCr input.
///
/// Use when you have pre-converted YCbCr from video decoders, etc.
/// Skips RGB->YCbCr conversion entirely.
///
/// Only valid with `ColorMode::YCbCr`. XYB mode requires RGB input.
pub struct YCbCrPlanarEncoder {
    config: EncoderConfig,
    width: u32,
    height: u32,
    rows_pushed: u32,
    y_plane: Vec<f32>,
    cb_plane: Vec<f32>,
    cr_plane: Vec<f32>,
}

impl YCbCrPlanarEncoder {
    pub(crate) fn new(config: EncoderConfig, width: u32, height: u32) -> Result<Self> {
        // Validate dimensions
        if width == 0 || height == 0 {
            return Err(Error::InvalidDimensions {
                width,
                height,
                reason: "dimensions cannot be zero",
            });
        }

        Ok(Self {
            config,
            width,
            height,
            rows_pushed: 0,
            y_plane: Vec::new(),
            cb_plane: Vec::new(),
            cr_plane: Vec::new(),
        })
    }

    /// Push full-resolution planes. Encoder subsamples chroma as needed.
    ///
    /// - `planes`: Y, Cb, Cr plane data with per-plane strides
    /// - `rows`: Number of luma rows to push
    /// - `stop`: Cancellation token
    pub fn push(&mut self, planes: &YCbCrPlanes<'_>, rows: usize, stop: impl Stop) -> Result<()> {
        if stop.should_stop() {
            return Err(Error::Cancelled);
        }

        // Validate row count
        let new_total = self.rows_pushed + rows as u32;
        if new_total > self.height {
            return Err(Error::TooManyRows {
                height: self.height,
                pushed: new_total,
            });
        }

        // Copy Y plane
        for row in 0..rows {
            if stop.should_stop() {
                return Err(Error::Cancelled);
            }
            let src_start = row * planes.y_stride;
            let src_end = src_start + self.width as usize;
            if src_end > planes.y.len() {
                return Err(Error::InvalidBufferSize {
                    expected: src_end,
                    actual: planes.y.len(),
                });
            }
            self.y_plane.extend_from_slice(&planes.y[src_start..src_end]);
        }

        // Copy Cb plane (full resolution, will be subsampled later)
        for row in 0..rows {
            let src_start = row * planes.cb_stride;
            let src_end = src_start + self.width as usize;
            if src_end > planes.cb.len() {
                return Err(Error::InvalidBufferSize {
                    expected: src_end,
                    actual: planes.cb.len(),
                });
            }
            self.cb_plane
                .extend_from_slice(&planes.cb[src_start..src_end]);
        }

        // Copy Cr plane (full resolution, will be subsampled later)
        for row in 0..rows {
            let src_start = row * planes.cr_stride;
            let src_end = src_start + self.width as usize;
            if src_end > planes.cr.len() {
                return Err(Error::InvalidBufferSize {
                    expected: src_end,
                    actual: planes.cr.len(),
                });
            }
            self.cr_plane
                .extend_from_slice(&planes.cr[src_start..src_end]);
        }

        self.rows_pushed = new_total;
        Ok(())
    }

    /// Push with pre-subsampled chroma.
    ///
    /// Cb/Cr are already at target chroma resolution.
    /// `y_rows` is luma row count; chroma rows derived from ChromaSubsampling.
    pub fn push_subsampled(
        &mut self,
        planes: &YCbCrPlanes<'_>,
        y_rows: usize,
        stop: impl Stop,
    ) -> Result<()> {
        // For now, delegate to push() - subsampling handling will be added later
        // TODO: Properly handle pre-subsampled input
        self.push(planes, y_rows, stop)
    }

    // === Status ===

    /// Get image width.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Get image height.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Get number of rows pushed so far.
    #[must_use]
    pub fn rows_pushed(&self) -> u32 {
        self.rows_pushed
    }

    /// Get number of rows remaining.
    #[must_use]
    pub fn rows_remaining(&self) -> u32 {
        self.height - self.rows_pushed
    }

    // === Finish ===

    /// Finish encoding, return JPEG bytes.
    pub fn finish(self) -> Result<Vec<u8>> {
        if self.rows_pushed != self.height {
            return Err(Error::IncompleteImage {
                height: self.height,
                pushed: self.rows_pushed,
            });
        }

        // TODO: Implement actual planar YCbCr encoding
        // For now, return an error indicating this is not yet implemented
        Err(Error::UnsupportedFeature {
            feature: "planar YCbCr encoding not yet implemented in v2 API",
        })
    }

    /// Finish encoding to Write destination.
    pub fn finish_to<W: Write>(self, mut output: W) -> Result<W> {
        let jpeg = self.finish()?;
        output.write_all(&jpeg)?;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use enough::Never;
    use rgb::RGB;

    #[test]
    fn test_bytes_encoder_basic() {
        let config = EncoderConfig::new().quality(85);
        let mut enc = config
            .encode_from_bytes(8, 8, PixelLayout::Rgb8Srgb)
            .unwrap();

        // Create 8x8 red image
        let pixels = vec![255u8, 0, 0].repeat(64);
        enc.push_packed(&pixels, Never).unwrap();

        let jpeg = enc.finish().unwrap();
        assert!(!jpeg.is_empty());
        assert_eq!(&jpeg[0..2], &[0xFF, 0xD8]); // JPEG SOI marker
    }

    #[test]
    fn test_rgb_encoder_basic() {
        let config = EncoderConfig::new().quality(85);
        let mut enc = config.encode_from_rgb::<RGB<u8>>(8, 8).unwrap();

        // Create 8x8 green image
        let pixels: Vec<RGB<u8>> = vec![RGB::new(0, 255, 0); 64];
        enc.push_packed(&pixels, Never).unwrap();

        let jpeg = enc.finish().unwrap();
        assert!(!jpeg.is_empty());
        assert_eq!(&jpeg[0..2], &[0xFF, 0xD8]); // JPEG SOI marker
    }

    #[test]
    fn test_stride_validation() {
        let config = EncoderConfig::new();
        let mut enc = config
            .encode_from_bytes(100, 10, PixelLayout::Rgb8Srgb)
            .unwrap();

        // Stride too small (less than width * 3)
        let result = enc.push(&[0u8; 100], 1, 100, Never);
        assert!(matches!(result, Err(Error::StrideTooSmall { .. })));
    }

    #[test]
    fn test_too_many_rows() {
        let config = EncoderConfig::new();
        let mut enc = config
            .encode_from_bytes(8, 4, PixelLayout::Rgb8Srgb)
            .unwrap();

        let row_data = vec![0u8; 8 * 3];

        // Push all 4 rows
        for _ in 0..4 {
            enc.push_packed(&row_data, Never).unwrap();
        }

        // Try to push one more
        let result = enc.push_packed(&row_data, Never);
        assert!(matches!(result, Err(Error::TooManyRows { .. })));
    }

    #[test]
    fn test_incomplete_image() {
        let config = EncoderConfig::new();
        let mut enc = config
            .encode_from_bytes(8, 8, PixelLayout::Rgb8Srgb)
            .unwrap();

        // Only push 4 rows
        let rows_data = vec![0u8; 8 * 3 * 4];
        enc.push_packed(&rows_data, Never).unwrap();

        // Try to finish
        let result = enc.finish();
        assert!(matches!(result, Err(Error::IncompleteImage { .. })));
    }
}
