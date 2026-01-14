//! Encoder implementations for v2 API.

// Allow use of deprecated StreamingEncoder internally - v2 API wraps it
#![allow(deprecated)]

use core::marker::PhantomData;

#[cfg(feature = "std")]
use std::io::Write;

use enough::Stop;

use super::config::EncoderConfig;
use super::types::{PixelLayout, YCbCrPlanes};
use crate::encode::streaming::StreamingEncoder;
use crate::error::{Error, Result};

/// Encoder for raw byte input with explicit pixel layout.
///
/// This encoder wraps `StreamingEncoder` to provide true streaming encoding
/// without buffering the entire image in memory.
pub struct BytesEncoder {
    /// v2 config (kept for ICC profile injection)
    config: EncoderConfig,
    /// Pixel layout
    layout: PixelLayout,
    /// Image dimensions
    width: u32,
    height: u32,
    /// Inner streaming encoder (handles actual encoding)
    inner: StreamingEncoder,
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

        // Build and start the streaming encoder with config from v2
        let inner = Self::build_streaming_encoder(&config, width, height, layout)?;

        Ok(Self {
            config,
            layout,
            width,
            height,
            inner,
        })
    }

    /// Build a StreamingEncoder from v2 config.
    fn build_streaming_encoder(
        config: &EncoderConfig,
        width: u32,
        height: u32,
        layout: PixelLayout,
    ) -> Result<StreamingEncoder> {
        use crate::encode::streaming::StreamingEncoder as SE;
        use crate::quant::Quality as LegacyQuality;

        let quality = LegacyQuality::from_quality(config.quality.to_internal());
        let pixel_format = layout.to_legacy();
        let subsampling = match config.color_mode {
            super::types::ColorMode::YCbCr { subsampling } => subsampling.to_legacy(),
            super::types::ColorMode::Xyb { .. } => crate::types::Subsampling::S444,
            super::types::ColorMode::Grayscale => crate::types::Subsampling::S444,
        };

        let mut builder = SE::new(width, height)
            .quality(quality)
            .pixel_format(pixel_format)
            .subsampling(subsampling)
            .optimize_huffman(config.optimize_huffman)
            .chroma_downsampling(config.downsampling_method.to_legacy())
            .restart_interval(config.restart_interval);

        if config.progressive {
            builder = builder.progressive(true);
        }

        if matches!(config.color_mode, super::types::ColorMode::Xyb { .. }) {
            builder = builder.use_xyb(true);
        }

        #[cfg(feature = "parallel")]
        if config.parallel.is_some() {
            // ParallelEncoding::Auto means enable parallel encoding
            // Future variants may have different behaviors
            builder = builder.parallel(true);
        }

        builder.start()
    }

    /// Push rows with explicit stride.
    ///
    /// - `data`: Raw pixel bytes
    /// - `rows`: Number of scanlines to push
    /// - `stride_bytes`: Bytes per row in buffer (>= width * bytes_per_pixel)
    /// - `stop`: Cancellation token (use `enough::Unstoppable` if not needed)
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
        let current_rows = self.inner.rows_pushed() as u32;
        let new_total = current_rows + rows as u32;
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

        // Push rows to streaming encoder
        if stride_bytes == min_stride {
            // Packed data - can push directly
            self.inner
                .push_rows_with_stop(&data[..rows * min_stride], rows, &stop)?;
        } else {
            // Strided data - push row by row
            for row in 0..rows {
                if stop.should_stop() {
                    return Err(Error::Cancelled);
                }

                let src_start = row * stride_bytes;
                let src_end = src_start + min_stride;
                self.inner
                    .push_row_with_stop(&data[src_start..src_end], &stop)?;
            }
        }

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
        self.inner.rows_pushed() as u32
    }

    /// Get number of rows remaining.
    #[must_use]
    pub fn rows_remaining(&self) -> u32 {
        self.height - self.inner.rows_pushed() as u32
    }

    /// Get the pixel layout.
    #[must_use]
    pub fn layout(&self) -> PixelLayout {
        self.layout
    }

    // === Finish ===

    /// Finish encoding, return JPEG bytes.
    pub fn finish(self) -> Result<Vec<u8>> {
        let rows_pushed = self.inner.rows_pushed() as u32;
        if rows_pushed != self.height {
            return Err(Error::IncompleteImage {
                height: self.height,
                pushed: rows_pushed,
            });
        }

        // Finish streaming encoder
        let mut jpeg = self.inner.finish()?;

        // Inject ICC profile if present
        if let Some(ref icc_data) = self.config.icc_profile {
            jpeg = inject_icc_profile(jpeg, icc_data);
        }

        Ok(jpeg)
    }

    /// Finish encoding to Write destination.
    #[cfg(feature = "std")]
    pub fn finish_to<W: Write>(self, mut output: W) -> Result<W> {
        let jpeg = self.finish()?;
        output.write_all(&jpeg)?;
        Ok(output)
    }

    /// Finish encoding, appending JPEG bytes to an existing Vec.
    ///
    /// Useful for no_std environments or buffer reuse.
    pub fn finish_to_vec(self, output: &mut Vec<u8>) -> Result<()> {
        let jpeg = self.finish()?;
        output.extend_from_slice(&jpeg);
        Ok(())
    }
}

/// ICC profile signature for APP2 marker.
const ICC_PROFILE_SIGNATURE: &[u8; 12] = b"ICC_PROFILE\0";

/// Maximum ICC profile bytes per APP2 marker segment.
/// APP2 max length is 65535, minus 2 (length) - 12 (signature) - 2 (chunk info) = 65519.
const MAX_ICC_BYTES_PER_MARKER: usize = 65519;

/// Inject an ICC profile into a JPEG, writing proper APP2 marker chunks.
///
/// Inserts APP2 markers right after SOI (and any existing APP0/APP1 markers).
/// Large profiles are automatically chunked per ICC spec.
fn inject_icc_profile(jpeg: Vec<u8>, icc_data: &[u8]) -> Vec<u8> {
    if icc_data.is_empty() {
        return jpeg;
    }

    // Find insertion point: after SOI and any APP0/APP1 markers
    let insert_pos = find_icc_insert_position(&jpeg);

    // Build ICC APP2 marker segments
    let icc_markers = build_icc_markers(icc_data);

    // Construct new JPEG with ICC markers inserted
    let mut result = Vec::with_capacity(jpeg.len() + icc_markers.len());
    result.extend_from_slice(&jpeg[..insert_pos]);
    result.extend_from_slice(&icc_markers);
    result.extend_from_slice(&jpeg[insert_pos..]);

    result
}

/// Find the position to insert ICC markers (after SOI and APP0/APP1).
fn find_icc_insert_position(jpeg: &[u8]) -> usize {
    // Start after SOI marker (2 bytes)
    let mut pos = 2;

    // Skip any existing APP0 (JFIF) and APP1 (EXIF) markers
    while pos + 4 <= jpeg.len() {
        if jpeg[pos] != 0xFF {
            break;
        }

        let marker = jpeg[pos + 1];
        // APP0 = 0xE0, APP1 = 0xE1
        if marker == 0xE0 || marker == 0xE1 {
            // Get segment length (big-endian, includes length bytes)
            let length = ((jpeg[pos + 2] as usize) << 8) | (jpeg[pos + 3] as usize);
            pos += 2 + length;
        } else {
            break;
        }
    }

    pos
}

/// Build ICC profile APP2 marker segments with proper chunking.
fn build_icc_markers(icc_data: &[u8]) -> Vec<u8> {
    let num_chunks = (icc_data.len() + MAX_ICC_BYTES_PER_MARKER - 1) / MAX_ICC_BYTES_PER_MARKER;
    let mut markers = Vec::new();

    let mut offset = 0;
    for chunk_num in 0..num_chunks {
        let chunk_size = (icc_data.len() - offset).min(MAX_ICC_BYTES_PER_MARKER);

        // APP2 marker
        markers.push(0xFF);
        markers.push(0xE2); // APP2

        // Length: 2 (length field) + 12 (signature) + 2 (chunk info) + data
        let segment_length = 2 + 12 + 2 + chunk_size;
        markers.push((segment_length >> 8) as u8);
        markers.push(segment_length as u8);

        // ICC_PROFILE signature
        markers.extend_from_slice(ICC_PROFILE_SIGNATURE);

        // Chunk number (1-based) and total chunks
        markers.push((chunk_num + 1) as u8);
        markers.push(num_chunks as u8);

        // ICC data chunk
        markers.extend_from_slice(&icc_data[offset..offset + chunk_size]);

        offset += chunk_size;
    }

    markers
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
    pub fn push(&mut self, data: &[P], rows: usize, stride: usize, stop: impl Stop) -> Result<()> {
        let stride_bytes = stride * core::mem::size_of::<P>();
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
    #[cfg(feature = "std")]
    pub fn finish_to<W: Write>(self, output: W) -> Result<W> {
        self.inner.finish_to(output)
    }

    /// Finish encoding, appending JPEG bytes to an existing Vec.
    ///
    /// Useful for no_std environments or buffer reuse.
    pub fn finish_to_vec(self, output: &mut Vec<u8>) -> Result<()> {
        self.inner.finish_to_vec(output)
    }
}

/// Encoder for planar f32 YCbCr input.
///
/// Use when you have pre-converted YCbCr from video decoders, etc.
/// Skips RGB->YCbCr conversion entirely.
///
/// Only valid with `ColorMode::YCbCr`. XYB mode requires RGB input.
pub struct YCbCrPlanarEncoder {
    #[allow(dead_code)] // Will be used when finish() is implemented
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
            self.y_plane
                .extend_from_slice(&planes.y[src_start..src_end]);
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
    #[cfg(feature = "std")]
    pub fn finish_to<W: Write>(self, mut output: W) -> Result<W> {
        let jpeg = self.finish()?;
        output.write_all(&jpeg)?;
        Ok(output)
    }

    /// Finish encoding, appending JPEG bytes to an existing Vec.
    ///
    /// Useful for no_std environments or buffer reuse.
    pub fn finish_to_vec(self, output: &mut Vec<u8>) -> Result<()> {
        let jpeg = self.finish()?;
        output.extend_from_slice(&jpeg);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use enough::Unstoppable;
    use rgb::RGB;

    #[test]
    fn test_bytes_encoder_basic() {
        let config = EncoderConfig::new().quality(85);
        let mut enc = config
            .encode_from_bytes(8, 8, PixelLayout::Rgb8Srgb)
            .unwrap();

        // Create 8x8 red image
        let pixels = vec![255u8, 0, 0].repeat(64);
        enc.push_packed(&pixels, Unstoppable).unwrap();

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
        enc.push_packed(&pixels, Unstoppable).unwrap();

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
        let result = enc.push(&[0u8; 100], 1, 100, Unstoppable);
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
            enc.push_packed(&row_data, Unstoppable).unwrap();
        }

        // Try to push one more
        let result = enc.push_packed(&row_data, Unstoppable);
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
        enc.push_packed(&rows_data, Unstoppable).unwrap();

        // Try to finish
        let result = enc.finish();
        assert!(matches!(result, Err(Error::IncompleteImage { .. })));
    }

    #[test]
    fn test_icc_profile_injection() {
        // Small fake ICC profile (just for testing structure)
        let fake_icc = vec![0u8; 1000];

        let config = EncoderConfig::new()
            .quality(85)
            .icc_profile(fake_icc.clone());
        let mut enc = config
            .encode_from_bytes(8, 8, PixelLayout::Rgb8Srgb)
            .unwrap();

        let pixels = vec![128u8; 8 * 8 * 3];
        enc.push_packed(&pixels, Unstoppable).unwrap();

        let jpeg = enc.finish().unwrap();

        // Verify JPEG structure
        assert_eq!(&jpeg[0..2], &[0xFF, 0xD8]); // SOI

        // Find APP2 ICC profile marker
        let mut found_icc = false;
        let mut pos = 2;
        while pos + 4 < jpeg.len() {
            if jpeg[pos] == 0xFF && jpeg[pos + 1] == 0xE2 {
                // APP2 marker - check for ICC signature
                if jpeg.len() > pos + 16 && &jpeg[pos + 4..pos + 16] == b"ICC_PROFILE\0" {
                    found_icc = true;
                    // Verify chunk numbers
                    assert_eq!(jpeg[pos + 16], 1); // chunk 1
                    assert_eq!(jpeg[pos + 17], 1); // of 1 total
                    break;
                }
            }
            if jpeg[pos] == 0xFF && jpeg[pos + 1] != 0x00 && jpeg[pos + 1] != 0xFF {
                let len = ((jpeg[pos + 2] as usize) << 8) | (jpeg[pos + 3] as usize);
                pos += 2 + len;
            } else {
                pos += 1;
            }
        }
        assert!(found_icc, "ICC profile APP2 marker not found");
    }

    #[test]
    fn test_icc_profile_chunking() {
        // Large ICC profile that requires multiple chunks
        let large_icc = vec![0xABu8; 100_000]; // > 65519 bytes

        let config = EncoderConfig::new().quality(85).icc_profile(large_icc);
        let mut enc = config
            .encode_from_bytes(8, 8, PixelLayout::Rgb8Srgb)
            .unwrap();

        let pixels = vec![128u8; 8 * 8 * 3];
        enc.push_packed(&pixels, Unstoppable).unwrap();

        let jpeg = enc.finish().unwrap();

        // Count APP2 ICC chunks
        let mut chunk_count = 0;
        let mut pos = 2;
        while pos + 4 < jpeg.len() {
            if jpeg[pos] == 0xFF && jpeg[pos + 1] == 0xE2 {
                if jpeg.len() > pos + 16 && &jpeg[pos + 4..pos + 16] == b"ICC_PROFILE\0" {
                    chunk_count += 1;
                    let chunk_num = jpeg[pos + 16];
                    let total_chunks = jpeg[pos + 17];
                    assert_eq!(chunk_num as usize, chunk_count);
                    assert_eq!(total_chunks, 2); // 100000 / 65519 = 2 chunks
                }
            }
            if jpeg[pos] == 0xFF && jpeg[pos + 1] != 0x00 && jpeg[pos + 1] != 0xFF {
                let len = ((jpeg[pos + 2] as usize) << 8) | (jpeg[pos + 3] as usize);
                pos += 2 + len;
            } else {
                pos += 1;
            }
        }
        assert_eq!(chunk_count, 2, "Expected 2 ICC chunks for 100KB profile");
    }

    #[test]
    fn test_finish_to_vec() {
        let config = EncoderConfig::new().quality(85);
        let mut enc = config.encode_from_rgb::<RGB<u8>>(8, 8).unwrap();

        let pixels: Vec<RGB<u8>> = vec![RGB::new(100, 150, 200); 64];
        enc.push_packed(&pixels, Unstoppable).unwrap();

        // Finish to existing vec
        let mut output = Vec::new();
        enc.finish_to_vec(&mut output).unwrap();

        assert!(!output.is_empty());
        assert_eq!(&output[0..2], &[0xFF, 0xD8]); // JPEG SOI marker
    }

    #[test]
    fn test_finish_to_vec_append() {
        let config = EncoderConfig::new().quality(85);
        let mut enc = config.encode_from_rgb::<RGB<u8>>(8, 8).unwrap();

        let pixels: Vec<RGB<u8>> = vec![RGB::new(100, 150, 200); 64];
        enc.push_packed(&pixels, Unstoppable).unwrap();

        // Finish to vec with existing content
        let mut output = vec![0xDE, 0xAD, 0xBE, 0xEF];
        let prefix_len = output.len();
        enc.finish_to_vec(&mut output).unwrap();

        // Verify prefix preserved
        assert_eq!(&output[0..4], &[0xDE, 0xAD, 0xBE, 0xEF]);
        // Verify JPEG appended
        assert_eq!(&output[prefix_len..prefix_len + 2], &[0xFF, 0xD8]);
    }

    #[test]
    fn test_icc_roundtrip_extraction() {
        // Test that we can extract the same ICC profile we injected
        let original_icc: Vec<u8> = (0..=255).cycle().take(3000).collect();

        let config = EncoderConfig::new()
            .quality(85)
            .icc_profile(original_icc.clone());
        let mut enc = config
            .encode_from_bytes(8, 8, PixelLayout::Rgb8Srgb)
            .unwrap();

        let pixels = vec![100u8; 8 * 8 * 3];
        enc.push_packed(&pixels, Unstoppable).unwrap();

        let jpeg = enc.finish().unwrap();

        // Extract ICC profile using the existing extraction function
        let extracted = crate::color::icc::extract_icc_profile(&jpeg);
        assert!(extracted.is_some(), "Failed to extract ICC profile");
        assert_eq!(
            extracted.unwrap(),
            original_icc,
            "Extracted ICC doesn't match original"
        );
    }
}
