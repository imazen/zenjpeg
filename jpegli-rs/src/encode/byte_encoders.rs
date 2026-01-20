//! Encoder implementations for v2 API.

use core::marker::PhantomData;

#[cfg(feature = "std")]
use std::io::Write;

use enough::Stop;

use super::encoder_config::EncoderConfig;
use super::encoder_types::{PixelLayout, YCbCrPlanes};
use super::streaming::StreamingEncoder;
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
            return Err(Error::invalid_dimensions(
                width,
                height,
                "dimensions cannot be zero",
            ));
        }

        // Check for overflow
        let pixel_count = (width as u64) * (height as u64);
        if pixel_count > u32::MAX as u64 {
            return Err(Error::invalid_dimensions(
                width,
                height,
                "dimensions too large",
            ));
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
        let pixel_format = layout.into();
        let subsampling = match config.color_mode {
            super::encoder_types::ColorMode::YCbCr { subsampling } => subsampling.into(),
            super::encoder_types::ColorMode::Xyb { .. } => crate::types::Subsampling::S444,
            super::encoder_types::ColorMode::Grayscale => crate::types::Subsampling::S444,
        };

        let mut builder = SE::new(width, height)
            .quality(quality)
            .pixel_format(pixel_format)
            .subsampling(subsampling)
            .optimize_huffman(config.optimize_huffman)
            .chroma_downsampling(config.downsampling_method)
            .restart_interval(config.restart_interval);

        // Apply custom encoding tables if configured
        if let Some(ref tables) = config.tables {
            builder = builder.encoding_tables(tables.clone());
        }

        if config.progressive {
            builder = builder.progressive(true);
        }

        if matches!(
            config.color_mode,
            super::encoder_types::ColorMode::Xyb { .. }
        ) {
            builder = builder.use_xyb(true);
        }

        // Always pass deringing setting (StreamingEncoder defaults to true)
        builder = builder.deringing(config.deringing);

        builder = builder.allow_16bit_quant_tables(config.allow_16bit_quant_tables);

        #[cfg(feature = "parallel")]
        if config.parallel.is_some() {
            // ParallelEncoding::Auto means enable parallel encoding
            // Future variants may have different behaviors
            builder = builder.parallel(true);
        }

        // Apply trellis quantization config if set
        #[cfg(feature = "experimental-hybrid-trellis")]
        if let Some(ref trellis) = config.trellis {
            builder = builder.trellis(*trellis);
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
            return Err(Error::cancelled());
        }

        let bpp = self.layout.bytes_per_pixel();
        let min_stride = self.width as usize * bpp;

        // Validate stride
        if stride_bytes < min_stride {
            return Err(Error::stride_too_small(self.width, stride_bytes));
        }

        // Validate row count
        let current_rows = self.inner.rows_pushed() as u32;
        let new_total = current_rows + rows as u32;
        if new_total > self.height {
            return Err(Error::too_many_rows(self.height, new_total));
        }

        // Validate buffer size
        let expected_size = rows * stride_bytes;
        if data.len() < expected_size {
            return Err(Error::invalid_buffer_size(expected_size, data.len()));
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
                    return Err(Error::cancelled());
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
            return Err(Error::invalid_dimensions(
                self.width,
                self.height,
                "row size is zero",
            ));
        }

        let rows = data.len() / row_bytes;
        if rows == 0 && !data.is_empty() {
            return Err(Error::invalid_buffer_size(row_bytes, data.len()));
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
            return Err(Error::incomplete_image(self.height, rows_pushed));
        }

        // Finish streaming encoder
        let mut jpeg = self.inner.finish()?;

        // Inject metadata in order: EXIF first (right after SOI), then XMP, then ICC
        // This matches standard JPEG metadata ordering
        if let Some(ref exif) = self.config.exif_data {
            if let Some(exif_bytes) = exif.to_bytes() {
                jpeg = inject_exif(jpeg, &exif_bytes);
            }
        }

        if let Some(ref xmp_data) = self.config.xmp_data {
            jpeg = inject_xmp(jpeg, xmp_data);
        }

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

/// EXIF signature for APP1 marker.
const EXIF_SIGNATURE: &[u8; 6] = b"Exif\0\0";

/// Maximum EXIF data bytes per APP1 marker segment.
/// APP1 max length is 65535, minus 2 (length) - 6 (signature) = 65527.
const MAX_EXIF_BYTES: usize = 65527;

/// XMP namespace signature for APP1 marker.
const XMP_NAMESPACE: &[u8; 29] = b"http://ns.adobe.com/xap/1.0/\0";

/// Maximum XMP data bytes per APP1 marker segment.
/// APP1 max length is 65535, minus 2 (length) - 29 (namespace) = 65504.
const MAX_XMP_BYTES: usize = 65504;

/// Inject EXIF data into a JPEG as APP1 marker, right after SOI.
fn inject_exif(jpeg: Vec<u8>, exif_data: &[u8]) -> Vec<u8> {
    if exif_data.is_empty() {
        return jpeg;
    }

    // Truncate if too large
    let exif_len = exif_data.len().min(MAX_EXIF_BYTES);

    // Build EXIF APP1 marker
    let mut marker = Vec::with_capacity(4 + 6 + exif_len);
    marker.push(0xFF);
    marker.push(0xE1); // APP1

    // Length: 2 (length field) + 6 (signature) + data
    let segment_length = 2 + 6 + exif_len;
    marker.push((segment_length >> 8) as u8);
    marker.push(segment_length as u8);

    // EXIF signature
    marker.extend_from_slice(EXIF_SIGNATURE);

    // EXIF data
    marker.extend_from_slice(&exif_data[..exif_len]);

    // Insert after SOI (2 bytes)
    let mut result = Vec::with_capacity(jpeg.len() + marker.len());
    result.extend_from_slice(&jpeg[..2]); // SOI
    result.extend_from_slice(&marker);
    result.extend_from_slice(&jpeg[2..]);

    result
}

/// Inject XMP data into a JPEG as APP1 marker.
///
/// Inserts after SOI and any existing APP1 (EXIF) markers.
fn inject_xmp(jpeg: Vec<u8>, xmp_data: &[u8]) -> Vec<u8> {
    if xmp_data.is_empty() {
        return jpeg;
    }

    // Truncate if too large
    let xmp_len = xmp_data.len().min(MAX_XMP_BYTES);

    // Build XMP APP1 marker
    let mut marker = Vec::with_capacity(4 + 29 + xmp_len);
    marker.push(0xFF);
    marker.push(0xE1); // APP1

    // Length: 2 (length field) + 29 (namespace) + data
    let segment_length = 2 + 29 + xmp_len;
    marker.push((segment_length >> 8) as u8);
    marker.push(segment_length as u8);

    // XMP namespace
    marker.extend_from_slice(XMP_NAMESPACE);

    // XMP data
    marker.extend_from_slice(&xmp_data[..xmp_len]);

    // Find insertion point: after SOI and any existing EXIF APP1 markers
    let insert_pos = find_xmp_insert_position(&jpeg);

    // Construct new JPEG with XMP marker inserted
    let mut result = Vec::with_capacity(jpeg.len() + marker.len());
    result.extend_from_slice(&jpeg[..insert_pos]);
    result.extend_from_slice(&marker);
    result.extend_from_slice(&jpeg[insert_pos..]);

    result
}

/// Find the position to insert XMP marker (after SOI and EXIF APP1).
fn find_xmp_insert_position(jpeg: &[u8]) -> usize {
    // Start after SOI marker (2 bytes)
    let mut pos = 2;

    // Skip any existing EXIF APP1 markers
    while pos + 4 <= jpeg.len() {
        if jpeg[pos] != 0xFF {
            break;
        }

        let marker = jpeg[pos + 1];
        // APP1 = 0xE1
        if marker == 0xE1 {
            // Check if it's EXIF (not XMP)
            if pos + 10 <= jpeg.len() && &jpeg[pos + 4..pos + 10] == b"Exif\0\0" {
                // Get segment length (big-endian, includes length bytes)
                let length = ((jpeg[pos + 2] as usize) << 8) | (jpeg[pos + 3] as usize);
                pos += 2 + length;
                continue;
            }
        }
        break;
    }

    pos
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
///
/// # YCbCr Value Range
///
/// Input values should be in the centered range:
/// - Y: 0.0 to 255.0 (luma)
/// - Cb, Cr: -128.0 to 127.0 (centered chroma)
///
/// This matches the output of standard RGB→YCbCr conversion with BT.601 coefficients.
///
/// # Streaming
///
/// Data can be pushed in any row count - the encoder buffers partial strips
/// internally and flushes when a complete strip is accumulated.
pub struct YCbCrPlanarEncoder {
    /// v2 config (kept for metadata injection)
    config: EncoderConfig,
    /// Image width
    width: u32,
    /// Image height
    height: u32,
    /// Chroma subsampling configuration
    subsampling: super::encoder_types::ChromaSubsampling,
    /// Strip height for MCU alignment (8 for 4:4:4, 16 for 4:2:0)
    strip_height: usize,
    /// Total rows received so far
    total_rows_pushed: usize,
    /// Y plane buffer (accumulates until strip_height rows)
    y_buffer: Vec<f32>,
    /// Cb plane buffer
    cb_buffer: Vec<f32>,
    /// Cr plane buffer
    cr_buffer: Vec<f32>,
    /// Number of rows currently buffered
    buffered_rows: usize,
    /// Inner streaming encoder (handles actual encoding)
    inner: StreamingEncoder,
}

impl YCbCrPlanarEncoder {
    pub(crate) fn new(config: EncoderConfig, width: u32, height: u32) -> Result<Self> {
        // Validate dimensions
        if width == 0 || height == 0 {
            return Err(Error::invalid_dimensions(
                width,
                height,
                "dimensions cannot be zero",
            ));
        }

        // Check for overflow
        let pixel_count = (width as u64) * (height as u64);
        if pixel_count > u32::MAX as u64 {
            return Err(Error::invalid_dimensions(
                width,
                height,
                "dimensions too large",
            ));
        }

        // Extract subsampling from color mode
        let subsampling = match config.color_mode {
            super::encoder_types::ColorMode::YCbCr { subsampling } => subsampling,
            _ => {
                return Err(Error::invalid_config(
                    "YCbCrPlanarEncoder requires YCbCr color mode".into(),
                ))
            }
        };

        // Build the streaming encoder
        let inner = Self::build_streaming_encoder(&config, width, height)?;

        // Get strip height from inner encoder (8 for 4:4:4, 16 for 4:2:0)
        let strip_height = inner.strip_height();

        // Allocate buffers for one strip
        let width_usize = width as usize;
        let buffer_size = width_usize * strip_height;

        Ok(Self {
            config,
            width,
            height,
            subsampling,
            strip_height,
            total_rows_pushed: 0,
            y_buffer: vec![0.0f32; buffer_size],
            cb_buffer: vec![0.0f32; buffer_size],
            cr_buffer: vec![0.0f32; buffer_size],
            buffered_rows: 0,
            inner,
        })
    }

    /// Build a StreamingEncoder from v2 config for YCbCr planar input.
    fn build_streaming_encoder(
        config: &EncoderConfig,
        width: u32,
        height: u32,
    ) -> Result<StreamingEncoder> {
        use crate::quant::Quality as LegacyQuality;
        use crate::types::PixelFormat;

        let quality = LegacyQuality::from_quality(config.quality.to_internal());
        let subsampling = match config.color_mode {
            super::encoder_types::ColorMode::YCbCr { subsampling } => subsampling.into(),
            _ => crate::types::Subsampling::S444,
        };

        // Use RGB pixel format - the streaming encoder will accept YCbCr data
        // via push_ycbcr_strip_f32, but needs a pixel format for buffer sizing
        let mut builder = StreamingEncoder::new(width, height)
            .quality(quality)
            .pixel_format(PixelFormat::Rgb) // Buffer sizing only
            .subsampling(subsampling)
            .optimize_huffman(config.optimize_huffman)
            .chroma_downsampling(config.downsampling_method)
            .restart_interval(config.restart_interval);

        // Apply custom encoding tables if configured
        if let Some(ref tables) = config.tables {
            builder = builder.encoding_tables(tables.clone());
        }

        if config.progressive {
            builder = builder.progressive(true);
        }

        // Always pass deringing setting (StreamingEncoder defaults to true)
        builder = builder.deringing(config.deringing);

        builder = builder.allow_16bit_quant_tables(config.allow_16bit_quant_tables);

        #[cfg(feature = "parallel")]
        if config.parallel.is_some() {
            builder = builder.parallel(true);
        }

        builder.start()
    }

    /// Push full-resolution planes. Encoder subsamples chroma as needed.
    ///
    /// All three planes must be at full luma resolution (width × rows).
    /// The encoder will perform chroma subsampling according to the configured
    /// `ChromaSubsampling` mode.
    ///
    /// Data can be pushed in any amount - the encoder buffers partial strips
    /// internally and flushes when a complete strip is accumulated.
    ///
    /// # Arguments
    /// - `planes`: Y, Cb, Cr plane data with per-plane strides
    /// - `rows`: Number of luma rows to push
    /// - `stop`: Cancellation token (use `Unstoppable` if not needed)
    ///
    /// # Value Range
    /// - Y: 0.0 to 255.0
    /// - Cb, Cr: -128.0 to 127.0
    pub fn push(&mut self, planes: &YCbCrPlanes<'_>, rows: usize, stop: impl Stop) -> Result<()> {
        if stop.should_stop() {
            return Err(Error::cancelled());
        }

        let width = self.width as usize;

        // Validate row count
        let new_total = self.total_rows_pushed + rows;
        if new_total > self.height as usize {
            return Err(Error::too_many_rows(self.height, new_total as u32));
        }

        let mut src_row = 0;
        while src_row < rows {
            if stop.should_stop() {
                return Err(Error::cancelled());
            }

            // How many rows can we add to the buffer?
            let rows_to_add = (rows - src_row).min(self.strip_height - self.buffered_rows);

            // Copy rows to buffer
            for i in 0..rows_to_add {
                let buf_offset = (self.buffered_rows + i) * width;
                let src_row_idx = src_row + i;

                // Copy Y
                let y_src_start = src_row_idx * planes.y_stride;
                let y_src_end = y_src_start + width;
                if y_src_end > planes.y.len() {
                    return Err(Error::invalid_buffer_size(y_src_end, planes.y.len()));
                }
                self.y_buffer[buf_offset..buf_offset + width]
                    .copy_from_slice(&planes.y[y_src_start..y_src_end]);

                // Copy Cb
                let cb_src_start = src_row_idx * planes.cb_stride;
                let cb_src_end = cb_src_start + width;
                if cb_src_end > planes.cb.len() {
                    return Err(Error::invalid_buffer_size(cb_src_end, planes.cb.len()));
                }
                self.cb_buffer[buf_offset..buf_offset + width]
                    .copy_from_slice(&planes.cb[cb_src_start..cb_src_end]);

                // Copy Cr
                let cr_src_start = src_row_idx * planes.cr_stride;
                let cr_src_end = cr_src_start + width;
                if cr_src_end > planes.cr.len() {
                    return Err(Error::invalid_buffer_size(cr_src_end, planes.cr.len()));
                }
                self.cr_buffer[buf_offset..buf_offset + width]
                    .copy_from_slice(&planes.cr[cr_src_start..cr_src_end]);
            }

            self.buffered_rows += rows_to_add;
            src_row += rows_to_add;
            self.total_rows_pushed += rows_to_add;

            // Flush if we have a complete strip or this is the final strip
            let remaining_image_rows = self.height as usize - self.inner.rows_pushed();
            if self.buffered_rows >= self.strip_height || self.buffered_rows >= remaining_image_rows
            {
                self.flush_buffer()?;
            }
        }

        Ok(())
    }

    /// Flush buffered rows to the inner encoder.
    fn flush_buffer(&mut self) -> Result<()> {
        if self.buffered_rows == 0 {
            return Ok(());
        }

        let width = self.width as usize;
        let data_len = self.buffered_rows * width;

        self.inner.push_ycbcr_strip_f32(
            &self.y_buffer[..data_len],
            &self.cb_buffer[..data_len],
            &self.cr_buffer[..data_len],
            self.buffered_rows,
        )?;

        self.buffered_rows = 0;
        Ok(())
    }

    /// Push with pre-subsampled chroma.
    ///
    /// Use this when your chroma planes are already at the target subsampled resolution.
    /// Y plane is still at full resolution.
    ///
    /// **Note:** Unlike `push()`, this method does not buffer partial strips.
    /// For best results, push complete strips (multiples of `strip_height` rows,
    /// which is 8 for 4:4:4 or 16 for 4:2:0).
    ///
    /// # Arguments
    /// - `planes`: Y at full resolution, Cb/Cr at subsampled resolution
    /// - `y_rows`: Number of luma rows to push
    /// - `stop`: Cancellation token
    ///
    /// # Chroma Dimensions
    /// The expected chroma dimensions depend on the subsampling mode:
    /// - 4:4:4 (None): cb/cr at full width × full height
    /// - 4:2:2 (HalfHorizontal): cb/cr at width/2 × full height
    /// - 4:2:0 (Quarter): cb/cr at width/2 × height/2
    pub fn push_subsampled(
        &mut self,
        planes: &YCbCrPlanes<'_>,
        y_rows: usize,
        stop: impl Stop,
    ) -> Result<()> {
        if stop.should_stop() {
            return Err(Error::cancelled());
        }

        // Check that we don't have buffered full-resolution data
        // (can't mix push() and push_subsampled())
        if self.buffered_rows > 0 {
            return Err(Error::internal(
                "cannot mix push() and push_subsampled() - flush first",
            ));
        }

        let width = self.width as usize;

        // Calculate chroma dimensions based on subsampling
        let (chroma_width, chroma_v_factor) = match self.subsampling {
            super::encoder_types::ChromaSubsampling::None => (width, 1),
            super::encoder_types::ChromaSubsampling::HalfHorizontal => ((width + 1) / 2, 1),
            super::encoder_types::ChromaSubsampling::Quarter => ((width + 1) / 2, 2),
            super::encoder_types::ChromaSubsampling::HalfVertical => (width, 2),
        };
        let chroma_rows = (y_rows + chroma_v_factor - 1) / chroma_v_factor;

        // Validate row count
        let new_total = self.total_rows_pushed + y_rows;
        if new_total > self.height as usize {
            return Err(Error::too_many_rows(self.height, new_total as u32));
        }

        // Check if input is already contiguous
        let y_contiguous = planes.y_stride == width;
        let cb_contiguous = planes.cb_stride == chroma_width;
        let cr_contiguous = planes.cr_stride == chroma_width;

        if y_contiguous && cb_contiguous && cr_contiguous {
            // Fast path: data is already contiguous
            let y_len = width * y_rows;
            let c_len = chroma_width * chroma_rows;

            if planes.y.len() < y_len {
                return Err(Error::invalid_buffer_size(y_len, planes.y.len()));
            }
            if planes.cb.len() < c_len {
                return Err(Error::invalid_buffer_size(c_len, planes.cb.len()));
            }
            if planes.cr.len() < c_len {
                return Err(Error::invalid_buffer_size(c_len, planes.cr.len()));
            }

            self.inner.push_ycbcr_strip_f32_subsampled(
                &planes.y[..y_len],
                &planes.cb[..c_len],
                &planes.cr[..c_len],
                y_rows,
            )?;
        } else {
            // Slow path: copy strided data to contiguous buffers
            let mut y_buf = vec![0.0f32; width * y_rows];
            let mut cb_buf = vec![0.0f32; chroma_width * chroma_rows];
            let mut cr_buf = vec![0.0f32; chroma_width * chroma_rows];

            // Copy Y plane
            for row in 0..y_rows {
                if stop.should_stop() {
                    return Err(Error::cancelled());
                }
                let dst_start = row * width;
                let dst_end = dst_start + width;
                let src_start = row * planes.y_stride;
                let src_end = src_start + width;
                if src_end > planes.y.len() {
                    return Err(Error::invalid_buffer_size(src_end, planes.y.len()));
                }
                y_buf[dst_start..dst_end].copy_from_slice(&planes.y[src_start..src_end]);
            }

            // Copy chroma planes
            for row in 0..chroma_rows {
                let dst_start = row * chroma_width;
                let dst_end = dst_start + chroma_width;

                let cb_src_start = row * planes.cb_stride;
                let cb_src_end = cb_src_start + chroma_width;
                if cb_src_end > planes.cb.len() {
                    return Err(Error::invalid_buffer_size(cb_src_end, planes.cb.len()));
                }
                cb_buf[dst_start..dst_end].copy_from_slice(&planes.cb[cb_src_start..cb_src_end]);

                let cr_src_start = row * planes.cr_stride;
                let cr_src_end = cr_src_start + chroma_width;
                if cr_src_end > planes.cr.len() {
                    return Err(Error::invalid_buffer_size(cr_src_end, planes.cr.len()));
                }
                cr_buf[dst_start..dst_end].copy_from_slice(&planes.cr[cr_src_start..cr_src_end]);
            }

            self.inner
                .push_ycbcr_strip_f32_subsampled(&y_buf, &cb_buf, &cr_buf, y_rows)?;
        }

        self.total_rows_pushed += y_rows;
        Ok(())
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

    /// Get number of rows pushed so far (including buffered rows).
    #[must_use]
    pub fn rows_pushed(&self) -> u32 {
        self.total_rows_pushed as u32
    }

    /// Get number of rows remaining.
    #[must_use]
    pub fn rows_remaining(&self) -> u32 {
        self.height - self.rows_pushed()
    }

    // === Finish ===

    /// Finish encoding, return JPEG bytes.
    pub fn finish(mut self) -> Result<Vec<u8>> {
        // Check if all rows were pushed
        if self.total_rows_pushed != self.height as usize {
            return Err(Error::incomplete_image(
                self.height,
                self.total_rows_pushed as u32,
            ));
        }

        // Flush any remaining buffered rows
        self.flush_buffer()?;

        // Finish streaming encoder
        let mut jpeg = self.inner.finish()?;

        // Inject metadata in order: EXIF first (right after SOI), then XMP, then ICC
        if let Some(ref exif) = self.config.exif_data {
            if let Some(exif_bytes) = exif.to_bytes() {
                jpeg = inject_exif(jpeg, &exif_bytes);
            }
        }

        if let Some(ref xmp_data) = self.config.xmp_data {
            jpeg = inject_xmp(jpeg, xmp_data);
        }

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::ChromaSubsampling;
    use crate::error::ErrorKind;
    use enough::Unstoppable;
    use rgb::RGB;

    #[test]
    fn test_bytes_encoder_basic() {
        let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter);
        let mut enc = config
            .encode_from_bytes(8, 8, PixelLayout::Rgb8Srgb)
            .unwrap();

        // Create 8x8 red image
        let pixels = [255u8, 0, 0].repeat(64);
        enc.push_packed(&pixels, Unstoppable).unwrap();

        let jpeg = enc.finish().unwrap();
        assert!(!jpeg.is_empty());
        assert_eq!(&jpeg[0..2], &[0xFF, 0xD8]); // JPEG SOI marker
    }

    #[test]
    fn test_rgb_encoder_basic() {
        let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter);
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
        let config = EncoderConfig::ycbcr(90.0, ChromaSubsampling::None);
        let mut enc = config
            .encode_from_bytes(100, 10, PixelLayout::Rgb8Srgb)
            .unwrap();

        // Stride too small (less than width * 3)
        let result = enc.push(&[0u8; 100], 1, 100, Unstoppable);
        assert!(matches!(
            result.as_ref().map_err(|e| e.kind()),
            Err(ErrorKind::StrideTooSmall { .. })
        ));
    }

    #[test]
    fn test_too_many_rows() {
        let config = EncoderConfig::ycbcr(90.0, ChromaSubsampling::None);
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
        assert!(matches!(
            result.as_ref().map_err(|e| e.kind()),
            Err(ErrorKind::TooManyRows { .. })
        ));
    }

    #[test]
    fn test_incomplete_image() {
        let config = EncoderConfig::ycbcr(90.0, ChromaSubsampling::None);
        let mut enc = config
            .encode_from_bytes(8, 8, PixelLayout::Rgb8Srgb)
            .unwrap();

        // Only push 4 rows
        let rows_data = vec![0u8; 8 * 3 * 4];
        enc.push_packed(&rows_data, Unstoppable).unwrap();

        // Try to finish
        let result = enc.finish();
        assert!(matches!(
            result.as_ref().map_err(|e| e.kind()),
            Err(ErrorKind::IncompleteImage { .. })
        ));
    }

    #[test]
    fn test_icc_profile_injection() {
        // Small fake ICC profile (just for testing structure)
        let fake_icc = vec![0u8; 1000];

        let config =
            EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter).icc_profile(fake_icc.clone());
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

        let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter).icc_profile(large_icc);
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
            if jpeg[pos] == 0xFF
                && jpeg[pos + 1] == 0xE2
                && jpeg.len() > pos + 16
                && &jpeg[pos + 4..pos + 16] == b"ICC_PROFILE\0"
            {
                chunk_count += 1;
                let chunk_num = jpeg[pos + 16];
                let total_chunks = jpeg[pos + 17];
                assert_eq!(chunk_num as usize, chunk_count);
                assert_eq!(total_chunks, 2); // 100000 / 65519 = 2 chunks
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
        let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter);
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
        let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter);
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

        let config =
            EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter).icc_profile(original_icc.clone());
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

    // =========================================================================
    // YCbCrPlanarEncoder tests
    // =========================================================================

    /// Helper: Convert RGB to YCbCr f32 using BT.601 coefficients.
    fn rgb_to_ycbcr_f32(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
        let r = r as f32;
        let g = g as f32;
        let b = b as f32;

        // BT.601 coefficients
        let y = 0.299 * r + 0.587 * g + 0.114 * b;
        let cb = -0.168736 * r - 0.331264 * g + 0.5 * b;
        let cr = 0.5 * r - 0.418688 * g - 0.081312 * b;

        (y, cb, cr)
    }

    #[test]
    fn test_ycbcr_planar_encoder_basic() {
        use crate::encode::YCbCrPlanes;

        let width = 8usize;
        let height = 8usize;

        // Create YCbCr data for a solid red image
        let mut y_plane = vec![0.0f32; width * height];
        let mut cb_plane = vec![0.0f32; width * height];
        let mut cr_plane = vec![0.0f32; width * height];

        for i in 0..(width * height) {
            let (y, cb, cr) = rgb_to_ycbcr_f32(255, 0, 0); // Red
            y_plane[i] = y;
            cb_plane[i] = cb;
            cr_plane[i] = cr;
        }

        let planes = YCbCrPlanes {
            y: &y_plane,
            y_stride: width,
            cb: &cb_plane,
            cb_stride: width,
            cr: &cr_plane,
            cr_stride: width,
        };

        let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter);
        let mut enc = config
            .encode_from_ycbcr_planar(width as u32, height as u32)
            .unwrap();

        enc.push(&planes, height, Unstoppable).unwrap();

        let jpeg = enc.finish().unwrap();
        assert!(!jpeg.is_empty());
        assert_eq!(&jpeg[0..2], &[0xFF, 0xD8]); // JPEG SOI marker
    }

    #[test]
    fn test_ycbcr_planar_encoder_gradient() {
        use crate::encode::YCbCrPlanes;

        let width = 64usize;
        let height = 64usize;

        // Create YCbCr data for a horizontal gradient (black to white)
        let mut y_plane = vec![0.0f32; width * height];
        let mut cb_plane = vec![0.0f32; width * height];
        let mut cr_plane = vec![0.0f32; width * height];

        for row in 0..height {
            for col in 0..width {
                let gray = (col * 255 / (width - 1)) as u8;
                let (y, cb, cr) = rgb_to_ycbcr_f32(gray, gray, gray);
                let idx = row * width + col;
                y_plane[idx] = y;
                cb_plane[idx] = cb;
                cr_plane[idx] = cr;
            }
        }

        let planes = YCbCrPlanes {
            y: &y_plane,
            y_stride: width,
            cb: &cb_plane,
            cb_stride: width,
            cr: &cr_plane,
            cr_stride: width,
        };

        // Test with 4:4:4 subsampling
        let config = EncoderConfig::ycbcr(90, ChromaSubsampling::None);
        let mut enc = config
            .encode_from_ycbcr_planar(width as u32, height as u32)
            .unwrap();

        enc.push(&planes, height, Unstoppable).unwrap();

        let jpeg = enc.finish().unwrap();
        assert!(!jpeg.is_empty());
        assert_eq!(&jpeg[0..2], &[0xFF, 0xD8]);
    }

    #[test]
    fn test_ycbcr_planar_encoder_strided_input() {
        use crate::encode::YCbCrPlanes;

        let width = 8usize;
        let height = 8usize;
        let stride = 16usize; // Larger stride than width

        // Create YCbCr data with padding (stride > width)
        let mut y_plane = vec![0.0f32; stride * height];
        let mut cb_plane = vec![0.0f32; stride * height];
        let mut cr_plane = vec![0.0f32; stride * height];

        for row in 0..height {
            for col in 0..width {
                let (y, cb, cr) = rgb_to_ycbcr_f32(0, 255, 0); // Green
                let idx = row * stride + col;
                y_plane[idx] = y;
                cb_plane[idx] = cb;
                cr_plane[idx] = cr;
            }
            // Rest of the row (padding) is zeros
        }

        let planes = YCbCrPlanes {
            y: &y_plane,
            y_stride: stride,
            cb: &cb_plane,
            cb_stride: stride,
            cr: &cr_plane,
            cr_stride: stride,
        };

        let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter);
        let mut enc = config
            .encode_from_ycbcr_planar(width as u32, height as u32)
            .unwrap();

        enc.push(&planes, height, Unstoppable).unwrap();

        let jpeg = enc.finish().unwrap();
        assert!(!jpeg.is_empty());
        assert_eq!(&jpeg[0..2], &[0xFF, 0xD8]);
    }

    #[test]
    fn test_ycbcr_planar_encoder_multiple_pushes() {
        use crate::encode::YCbCrPlanes;

        // Use 4:4:4 which has 8-row strips, allowing 8-row pushes
        let width = 16usize;
        let height = 32usize;
        let rows_per_push = 8usize;

        // Create full image YCbCr data
        let mut y_plane = vec![0.0f32; width * height];
        let mut cb_plane = vec![0.0f32; width * height];
        let mut cr_plane = vec![0.0f32; width * height];

        for row in 0..height {
            for col in 0..width {
                // Different color for each 8-row strip
                let strip = row / 8;
                let (r, g, b) = match strip {
                    0 => (255, 0, 0),   // Red
                    1 => (0, 255, 0),   // Green
                    2 => (0, 0, 255),   // Blue
                    _ => (255, 255, 0), // Yellow
                };
                let (y, cb, cr) = rgb_to_ycbcr_f32(r, g, b);
                let idx = row * width + col;
                y_plane[idx] = y;
                cb_plane[idx] = cb;
                cr_plane[idx] = cr;
            }
        }

        // Use 4:4:4 subsampling which has 8-row strip height
        let config = EncoderConfig::ycbcr(85, ChromaSubsampling::None);
        let mut enc = config
            .encode_from_ycbcr_planar(width as u32, height as u32)
            .unwrap();

        // Push in 4 chunks of 8 rows each
        for chunk in 0..4 {
            let start_row = chunk * rows_per_push;
            let start_idx = start_row * width;
            let end_idx = start_idx + rows_per_push * width;

            let planes = YCbCrPlanes {
                y: &y_plane[start_idx..end_idx],
                y_stride: width,
                cb: &cb_plane[start_idx..end_idx],
                cb_stride: width,
                cr: &cr_plane[start_idx..end_idx],
                cr_stride: width,
            };

            enc.push(&planes, rows_per_push, Unstoppable).unwrap();
            assert_eq!(enc.rows_pushed(), ((chunk + 1) * rows_per_push) as u32);
        }

        let jpeg = enc.finish().unwrap();
        assert!(!jpeg.is_empty());
        assert_eq!(&jpeg[0..2], &[0xFF, 0xD8]);
    }

    #[test]
    fn test_ycbcr_planar_encoder_incomplete_image() {
        use crate::encode::YCbCrPlanes;

        // Use 4:4:4 with 8-row strip height for easier testing
        let width = 8usize;
        let height = 16usize;

        // Only create data for half the image
        let half_height = 8usize;
        let y_plane = vec![128.0f32; width * half_height];
        let cb_plane = vec![0.0f32; width * half_height];
        let cr_plane = vec![0.0f32; width * half_height];

        let planes = YCbCrPlanes {
            y: &y_plane,
            y_stride: width,
            cb: &cb_plane,
            cb_stride: width,
            cr: &cr_plane,
            cr_stride: width,
        };

        // Use 4:4:4 subsampling which has 8-row strip height
        let config = EncoderConfig::ycbcr(85, ChromaSubsampling::None);
        let mut enc = config
            .encode_from_ycbcr_planar(width as u32, height as u32)
            .unwrap();

        // Only push half the rows (8 of 16)
        enc.push(&planes, half_height, Unstoppable).unwrap();

        // Try to finish - should fail because only 8 of 16 rows pushed
        let result = enc.finish();
        assert!(matches!(
            result.as_ref().map_err(|e| e.kind()),
            Err(ErrorKind::IncompleteImage { .. })
        ));
    }

    #[test]
    fn test_ycbcr_planar_encoder_subsampled_444() {
        use crate::encode::YCbCrPlanes;

        let width = 16usize;
        let height = 16usize;

        // For 4:4:4, chroma is same size as luma
        let y_plane: Vec<f32> = (0..width * height).map(|i| (i % 256) as f32).collect();
        let cb_plane = vec![0.0f32; width * height];
        let cr_plane = vec![0.0f32; width * height];

        let planes = YCbCrPlanes {
            y: &y_plane,
            y_stride: width,
            cb: &cb_plane,
            cb_stride: width,
            cr: &cr_plane,
            cr_stride: width,
        };

        let config = EncoderConfig::ycbcr(85, ChromaSubsampling::None);
        let mut enc = config
            .encode_from_ycbcr_planar(width as u32, height as u32)
            .unwrap();

        enc.push_subsampled(&planes, height, Unstoppable).unwrap();

        let jpeg = enc.finish().unwrap();
        assert!(!jpeg.is_empty());
        assert_eq!(&jpeg[0..2], &[0xFF, 0xD8]);
    }

    #[test]
    fn test_ycbcr_planar_encoder_subsampled_420() {
        use crate::encode::YCbCrPlanes;

        let width = 16usize;
        let height = 16usize;
        let chroma_width = (width + 1) / 2; // 8
        let chroma_height = (height + 1) / 2; // 8

        // For 4:2:0, chroma is half size in both dimensions
        let y_plane: Vec<f32> = (0..width * height).map(|i| (i % 256) as f32).collect();
        let cb_plane = vec![0.0f32; chroma_width * chroma_height];
        let cr_plane = vec![0.0f32; chroma_width * chroma_height];

        let planes = YCbCrPlanes {
            y: &y_plane,
            y_stride: width,
            cb: &cb_plane,
            cb_stride: chroma_width,
            cr: &cr_plane,
            cr_stride: chroma_width,
        };

        let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter);
        let mut enc = config
            .encode_from_ycbcr_planar(width as u32, height as u32)
            .unwrap();

        enc.push_subsampled(&planes, height, Unstoppable).unwrap();

        let jpeg = enc.finish().unwrap();
        assert!(!jpeg.is_empty());
        assert_eq!(&jpeg[0..2], &[0xFF, 0xD8]);
    }

    #[test]
    fn test_ycbcr_planar_encoder_requires_ycbcr_mode() {
        // Try to create planar encoder with XYB mode - should fail
        let config = EncoderConfig::xyb(85, crate::encode::encoder_types::XybSubsampling::BQuarter);
        let result = config.encode_from_ycbcr_planar(8, 8);
        assert!(result.is_err());
    }

    #[test]
    fn test_ycbcr_planar_encoder_status_methods() {
        use crate::encode::YCbCrPlanes;

        let width = 16u32;
        let height = 32u32;

        let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter);
        let enc = config.encode_from_ycbcr_planar(width, height).unwrap();

        assert_eq!(enc.width(), width);
        assert_eq!(enc.height(), height);
        assert_eq!(enc.rows_pushed(), 0);
        assert_eq!(enc.rows_remaining(), height);

        // Push some rows
        let y_plane = vec![128.0f32; 16 * 16];
        let cb_plane = vec![0.0f32; 16 * 16];
        let cr_plane = vec![0.0f32; 16 * 16];

        let planes = YCbCrPlanes {
            y: &y_plane,
            y_stride: 16,
            cb: &cb_plane,
            cb_stride: 16,
            cr: &cr_plane,
            cr_stride: 16,
        };

        let mut enc = config.encode_from_ycbcr_planar(width, height).unwrap();
        enc.push(&planes, 16, Unstoppable).unwrap();

        assert_eq!(enc.rows_pushed(), 16);
        assert_eq!(enc.rows_remaining(), 16);
    }

    #[test]
    fn test_ycbcr_planar_encoder_with_icc_profile() {
        use crate::encode::YCbCrPlanes;

        let width = 8usize;
        let height = 8usize;

        let y_plane = vec![128.0f32; width * height];
        let cb_plane = vec![0.0f32; width * height];
        let cr_plane = vec![0.0f32; width * height];

        let planes = YCbCrPlanes {
            y: &y_plane,
            y_stride: width,
            cb: &cb_plane,
            cb_stride: width,
            cr: &cr_plane,
            cr_stride: width,
        };

        // Create config with ICC profile
        let fake_icc = vec![0xABu8; 1000];
        let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter).icc_profile(fake_icc);
        let mut enc = config
            .encode_from_ycbcr_planar(width as u32, height as u32)
            .unwrap();

        enc.push(&planes, height, Unstoppable).unwrap();

        let jpeg = enc.finish().unwrap();

        // Verify ICC profile was injected
        let mut found_icc = false;
        let mut pos = 2;
        while pos + 4 < jpeg.len() {
            if jpeg[pos] == 0xFF
                && jpeg[pos + 1] == 0xE2
                && jpeg.len() > pos + 16
                && &jpeg[pos + 4..pos + 16] == b"ICC_PROFILE\0"
            {
                found_icc = true;
                break;
            }
            if jpeg[pos] == 0xFF && jpeg[pos + 1] != 0x00 && jpeg[pos + 1] != 0xFF {
                let len = ((jpeg[pos + 2] as usize) << 8) | (jpeg[pos + 3] as usize);
                pos += 2 + len;
            } else {
                pos += 1;
            }
        }
        assert!(found_icc, "ICC profile should be present in output");
    }

    #[test]
    fn test_ycbcr_planar_encoder_odd_width() {
        use crate::encode::YCbCrPlanes;

        // Non-8-aligned width (tests partial block handling)
        let width = 13usize;
        let height = 17usize;

        let y_plane: Vec<f32> = (0..width * height).map(|i| (i % 256) as f32).collect();
        let cb_plane = vec![0.0f32; width * height];
        let cr_plane = vec![0.0f32; width * height];

        let planes = YCbCrPlanes {
            y: &y_plane,
            y_stride: width,
            cb: &cb_plane,
            cb_stride: width,
            cr: &cr_plane,
            cr_stride: width,
        };

        // Test with 4:4:4 (strip height 8)
        let config = EncoderConfig::ycbcr(85, ChromaSubsampling::None);
        let mut enc = config
            .encode_from_ycbcr_planar(width as u32, height as u32)
            .unwrap();

        enc.push(&planes, height, Unstoppable).unwrap();

        let jpeg = enc.finish().unwrap();
        assert!(!jpeg.is_empty());
        assert_eq!(&jpeg[0..2], &[0xFF, 0xD8]);
    }

    #[test]
    fn test_ycbcr_planar_encoder_single_row_pushes() {
        use crate::encode::YCbCrPlanes;

        // Push one row at a time - tests buffering
        let width = 16usize;
        let height = 24usize;

        let y_plane: Vec<f32> = (0..width * height).map(|i| (i % 256) as f32).collect();
        let cb_plane = vec![0.0f32; width * height];
        let cr_plane = vec![0.0f32; width * height];

        // Use 4:4:4 (strip height 8)
        let config = EncoderConfig::ycbcr(85, ChromaSubsampling::None);
        let mut enc = config
            .encode_from_ycbcr_planar(width as u32, height as u32)
            .unwrap();

        // Push one row at a time
        for row in 0..height {
            let start = row * width;
            let end = start + width;
            let planes = YCbCrPlanes {
                y: &y_plane[start..end],
                y_stride: width,
                cb: &cb_plane[start..end],
                cb_stride: width,
                cr: &cr_plane[start..end],
                cr_stride: width,
            };
            enc.push(&planes, 1, Unstoppable).unwrap();
        }

        let jpeg = enc.finish().unwrap();
        assert!(!jpeg.is_empty());
        assert_eq!(&jpeg[0..2], &[0xFF, 0xD8]);
    }

    #[test]
    fn test_ycbcr_planar_encoder_420_partial_pushes() {
        use crate::encode::YCbCrPlanes;

        // 4:2:0 has 16-row strip height - test partial push buffering
        let width = 16usize;
        let height = 32usize;

        let y_plane: Vec<f32> = (0..width * height).map(|i| (i % 256) as f32).collect();
        let cb_plane = vec![0.0f32; width * height];
        let cr_plane = vec![0.0f32; width * height];

        let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter);
        let mut enc = config
            .encode_from_ycbcr_planar(width as u32, height as u32)
            .unwrap();

        // Push in chunks smaller than strip height (16)
        // Push 5 + 5 + 6 = 16 rows (one strip)
        // Then 8 + 8 = 16 rows (one strip)
        let push_sizes = [5, 5, 6, 8, 8];
        let mut offset = 0;
        for &rows in &push_sizes {
            let start = offset * width;
            let end = start + rows * width;
            let planes = YCbCrPlanes {
                y: &y_plane[start..end],
                y_stride: width,
                cb: &cb_plane[start..end],
                cb_stride: width,
                cr: &cr_plane[start..end],
                cr_stride: width,
            };
            enc.push(&planes, rows, Unstoppable).unwrap();
            offset += rows;
        }

        let jpeg = enc.finish().unwrap();
        assert!(!jpeg.is_empty());
        assert_eq!(&jpeg[0..2], &[0xFF, 0xD8]);
    }
}
