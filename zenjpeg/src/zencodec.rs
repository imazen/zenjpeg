//! zencodec-types trait implementations for zenjpeg.
//!
//! Provides [`JpegEncoderConfig`] and [`JpegDecoderConfig`] types that implement
//! the 4-layer encode/decode traits from zencodec-types, wrapping the native
//! zenjpeg API.
//!
//! The native API remains untouched — this is a thin adapter layer.
//!
//! # Trait mapping
//!
//! | zencodec-types | zenjpeg adapter |
//! |----------------|-----------------|
//! | `EncoderConfig` | [`JpegEncoderConfig`] |
//! | `EncodeJob<'a>` | [`JpegEncodeJob`] |
//! | `Encoder` | [`JpegEncoder`] |
//! | `FrameEncoder` | [`JpegFrameEncoder`] |
//! | `DecoderConfig` | [`JpegDecoderConfig`] |
//! | `DecodeJob<'a>` | [`JpegDecodeJob`] |
//! | `Decoder` | [`JpegDecoder`] |
//! | `FrameDecoder` | [`JpegFrameDecoder`] |

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use zencodec_types::{
    CodecCapabilities, DecodeFrame, DecodeOutput, EncodeOutput, ImageFormat, ImageInfo,
    ImageMetadata, OutputInfo, PixelData, PixelDescriptor, PixelSlice, PixelSliceMut,
    ResourceLimits, Stop,
};

use crate::encode::encoder_config::EncoderConfig;
use crate::encode::encoder_types::{ChromaSubsampling, PixelLayout, Quality};
use crate::encode::exif::Exif;
use crate::error::Error;

// ── Backwards compat aliases ─────────────────────────────────────────────────

/// Alias for backwards compatibility within the `zencodec` feature gate.
pub type JpegEncoding = JpegEncoderConfig;
/// Alias for backwards compatibility within the `zencodec` feature gate.
pub type JpegDecoding = JpegDecoderConfig;

// ============================================================================
// Encode side: EncoderConfig → EncodeJob → Encoder / FrameEncoder
// ============================================================================

/// JPEG encoder configuration implementing [`zencodec_types::EncoderConfig`].
///
/// Wraps [`EncoderConfig`] with the zencodec trait interface.
/// Defaults to YCbCr 4:2:0 at quality 85.
#[derive(Clone, Debug)]
pub struct JpegEncoderConfig {
    inner: EncoderConfig,
    quality: f32,
}

impl JpegEncoderConfig {
    /// Create a default YCbCr 4:2:0 config at quality 85.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter),
            quality: 85.0,
        }
    }

    /// Create a YCbCr config with quality and subsampling.
    #[must_use]
    pub fn ycbcr(quality: f32, subsampling: ChromaSubsampling) -> Self {
        Self {
            inner: EncoderConfig::ycbcr(quality, subsampling),
            quality,
        }
    }

    /// Create a grayscale config with quality.
    #[must_use]
    pub fn grayscale(quality: f32) -> Self {
        Self {
            inner: EncoderConfig::grayscale(quality),
            quality,
        }
    }

    /// Enable progressive JPEG encoding.
    #[must_use]
    pub fn with_progressive(mut self, enable: bool) -> Self {
        self.inner = self.inner.progressive(enable);
        self
    }

    /// Enable SharpYUV chroma downsampling (better edges, slower).
    #[must_use]
    pub fn with_sharp_yuv(mut self, enable: bool) -> Self {
        self.inner = self.inner.sharp_yuv(enable);
        self
    }

    /// Set chroma subsampling mode.
    #[must_use]
    pub fn with_subsampling(self, subsampling: ChromaSubsampling) -> Self {
        Self {
            inner: EncoderConfig::ycbcr(self.quality, subsampling),
            ..self
        }
    }

    /// Access the underlying [`EncoderConfig`].
    #[must_use]
    pub fn inner(&self) -> &EncoderConfig {
        &self.inner
    }

    /// Mutable access to the underlying [`EncoderConfig`].
    pub fn inner_mut(&mut self) -> &mut EncoderConfig {
        &mut self.inner
    }
}

impl Default for JpegEncoderConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Supported encode pixel formats (native, zero-conversion paths).
static ENCODE_DESCRIPTORS: &[PixelDescriptor] = &[
    PixelDescriptor::RGB8_SRGB,
    PixelDescriptor::RGBA8_SRGB,
    PixelDescriptor::GRAY8_SRGB,
    PixelDescriptor::BGRA8_SRGB,
    PixelDescriptor::RGB16_SRGB,
    PixelDescriptor::RGBA16_SRGB,
    PixelDescriptor::GRAY16_SRGB,
    PixelDescriptor::RGBF32_LINEAR,
    PixelDescriptor::RGBAF32_LINEAR,
    PixelDescriptor::GRAYF32_LINEAR,
];

impl zencodec_types::EncoderConfig for JpegEncoderConfig {
    type Error = Error;
    type Job<'a> = JpegEncodeJob<'a>;

    fn format() -> ImageFormat {
        ImageFormat::Jpeg
    }

    fn supported_descriptors() -> &'static [PixelDescriptor] {
        ENCODE_DESCRIPTORS
    }

    fn capabilities() -> &'static CodecCapabilities {
        static CAPS: CodecCapabilities = CodecCapabilities::new()
            .with_encode_icc(true)
            .with_encode_exif(true)
            .with_encode_xmp(true)
            .with_encode_cancel(true)
            .with_native_gray(true)
            .with_quality_range(0.0, 100.0);
        &CAPS
    }

    fn with_calibrated_quality(mut self, quality: f32) -> Self {
        let q = quality.clamp(0.0, 100.0);
        self.quality = q;
        self.inner = self.inner.quality(Quality::ApproxJpegli(q));
        self
    }

    fn calibrated_quality(&self) -> Option<f32> {
        Some(self.quality)
    }

    fn job(&self) -> Self::Job<'_> {
        JpegEncodeJob {
            config: self,
            stop: None,
            metadata: None,
            limits: ResourceLimits::none(),
        }
    }
}

// ── Encode job ──────────────────────────────────────────────────────────────

/// Per-operation JPEG encode job.
///
/// Created by [`JpegEncoderConfig::job()`]. Borrows temporary data (stop token,
/// metadata) and is consumed by creating an [`JpegEncoder`] or [`JpegFrameEncoder`].
pub struct JpegEncodeJob<'a> {
    config: &'a JpegEncoderConfig,
    stop: Option<&'a dyn Stop>,
    metadata: Option<&'a ImageMetadata<'a>>,
    limits: ResourceLimits,
}

impl<'a> zencodec_types::EncodeJob<'a> for JpegEncodeJob<'a> {
    type Error = Error;
    type Encoder = JpegEncoder<'a>;
    type FrameEncoder = JpegFrameEncoder;

    fn with_stop(mut self, stop: &'a dyn Stop) -> Self {
        self.stop = Some(stop);
        self
    }

    fn with_metadata(mut self, meta: &'a ImageMetadata<'a>) -> Self {
        self.metadata = Some(meta);
        self
    }

    fn with_limits(mut self, limits: ResourceLimits) -> Self {
        self.limits = limits;
        self
    }

    fn encoder(self) -> Self::Encoder {
        JpegEncoder {
            config: self.config,
            stop: self.stop,
            metadata: self.metadata,
            buffer: None,
        }
    }

    fn frame_encoder(self) -> Result<Self::FrameEncoder, Self::Error> {
        Err(Error::unsupported_feature(
            "JPEG does not support animation encoding",
        ))
    }
}

// ── Encoder ─────────────────────────────────────────────────────────────────

/// Single-image JPEG encoder implementing [`zencodec_types::Encoder`].
///
/// Supports one-shot `encode()` and row-level `push_rows()` + `finish()`.
pub struct JpegEncoder<'a> {
    config: &'a JpegEncoderConfig,
    stop: Option<&'a dyn Stop>,
    metadata: Option<&'a ImageMetadata<'a>>,
    /// Accumulated rows for push_rows path (None = not started, Some = buffering).
    buffer: Option<RowBuffer>,
}

/// Internal buffer for accumulating pushed rows.
struct RowBuffer {
    data: Vec<u8>,
    width: u32,
    total_rows: u32,
    descriptor: PixelDescriptor,
}

impl<'a> JpegEncoder<'a> {
    /// Build an EncodeRequest from current config + metadata.
    fn build_request(&self) -> crate::encode::request::EncodeRequest<'a> {
        let mut req = self.config.inner.request();
        if let Some(meta) = self.metadata {
            if let Some(icc) = meta.icc_profile {
                req = req.icc_profile(icc);
            }
            if let Some(exif) = meta.exif {
                req = req.exif(Exif::raw(exif));
            }
            if let Some(xmp) = meta.xmp {
                req = req.xmp(xmp);
            }
        }
        if let Some(stop) = self.stop {
            req = req.stop(stop);
        }
        req
    }

    /// Encode from raw bytes with a known pixel layout.
    fn encode_bytes_inner(
        &self,
        data: &[u8],
        width: u32,
        height: u32,
        layout: PixelLayout,
    ) -> Result<EncodeOutput, Error> {
        let req = self.build_request();
        let output = req.encode_bytes(data, width, height, layout)?;
        Ok(EncodeOutput::new(output, ImageFormat::Jpeg))
    }
}

/// Map a PixelDescriptor to a zenjpeg PixelLayout.
fn descriptor_to_layout(desc: PixelDescriptor) -> Result<PixelLayout, Error> {
    use zencodec_types::{ChannelLayout, ChannelType, TransferFunction};
    match (desc.channel_type, desc.layout, desc.transfer) {
        (ChannelType::U8, ChannelLayout::Rgb, TransferFunction::Srgb) => Ok(PixelLayout::Rgb8Srgb),
        (ChannelType::U8, ChannelLayout::Rgba, TransferFunction::Srgb) => {
            Ok(PixelLayout::Rgba8Srgb)
        }
        (ChannelType::U8, ChannelLayout::Bgra, TransferFunction::Srgb) => {
            Ok(PixelLayout::Bgra8Srgb)
        }
        (ChannelType::U8, ChannelLayout::Gray, TransferFunction::Srgb) => {
            Ok(PixelLayout::Gray8Srgb)
        }
        (ChannelType::U16, ChannelLayout::Rgb, _) => Ok(PixelLayout::Rgb16Linear),
        (ChannelType::U16, ChannelLayout::Rgba, _) => Ok(PixelLayout::Rgba16Linear),
        (ChannelType::U16, ChannelLayout::Gray, _) => Ok(PixelLayout::Gray16Linear),
        (ChannelType::F32, ChannelLayout::Rgb, TransferFunction::Linear) => {
            Ok(PixelLayout::RgbF32Linear)
        }
        (ChannelType::F32, ChannelLayout::Rgba, TransferFunction::Linear) => {
            Ok(PixelLayout::RgbaF32Linear)
        }
        (ChannelType::F32, ChannelLayout::Gray, TransferFunction::Linear) => {
            Ok(PixelLayout::GrayF32Linear)
        }
        _ => Err(Error::unsupported_feature(
            "unsupported pixel format for JPEG encoding",
        )),
    }
}

/// Collect contiguous pixel data from a (possibly strided) PixelSlice.
fn collect_contiguous(pixels: &PixelSlice<'_>) -> Vec<u8> {
    let bpp = pixels.descriptor().bytes_per_pixel();
    let row_bytes = pixels.width() as usize * bpp;
    let mut buf = Vec::with_capacity(row_bytes * pixels.rows() as usize);
    for y in 0..pixels.rows() {
        buf.extend_from_slice(pixels.row(y));
    }
    buf
}

impl<'a> zencodec_types::Encoder for JpegEncoder<'a> {
    type Error = Error;

    fn encode(self, pixels: PixelSlice<'_>) -> Result<EncodeOutput, Error> {
        let layout = descriptor_to_layout(pixels.descriptor())?;
        let width = pixels.width();
        let height = pixels.rows();

        // Collect contiguous bytes
        let data = collect_contiguous(&pixels);
        self.encode_bytes_inner(&data, width, height, layout)
    }

    fn push_rows(&mut self, rows: PixelSlice<'_>) -> Result<(), Error> {
        let desc = rows.descriptor();
        let width = rows.width();

        match &mut self.buffer {
            None => {
                // First push — initialize buffer
                let bpp = desc.bytes_per_pixel();
                let row_bytes = width as usize * bpp;
                let mut data = Vec::with_capacity(row_bytes * rows.rows() as usize * 4); // estimate
                for y in 0..rows.rows() {
                    data.extend_from_slice(rows.row(y));
                }
                self.buffer = Some(RowBuffer {
                    data,
                    width,
                    total_rows: rows.rows(),
                    descriptor: desc,
                });
            }
            Some(buf) => {
                // Validate consistency
                if buf.width != width || buf.descriptor != desc {
                    return Err(Error::unsupported_feature(
                        "push_rows: width or format changed between calls",
                    ));
                }
                for y in 0..rows.rows() {
                    buf.data.extend_from_slice(rows.row(y));
                }
                buf.total_rows += rows.rows();
            }
        }
        Ok(())
    }

    fn finish(mut self) -> Result<EncodeOutput, Error> {
        let buf = self
            .buffer
            .take()
            .ok_or_else(|| Error::unsupported_feature("finish() called without any push_rows()"))?;
        let layout = descriptor_to_layout(buf.descriptor)?;
        self.encode_bytes_inner(&buf.data, buf.width, buf.total_rows, layout)
    }

    fn encode_from(
        self,
        _source: &mut dyn FnMut(u32, PixelSliceMut<'_>) -> usize,
    ) -> Result<EncodeOutput, Error> {
        Err(Error::unsupported_feature(
            "JPEG encode_from() is not yet implemented; use encode() instead",
        ))
    }
}

// ── FrameEncoder (animation — unsupported for JPEG) ─────────────────────────

/// JPEG frame encoder — always returns an error since JPEG doesn't support animation.
pub struct JpegFrameEncoder;

impl zencodec_types::FrameEncoder for JpegFrameEncoder {
    type Error = Error;

    fn push_frame(&mut self, _pixels: PixelSlice<'_>, _duration_ms: u32) -> Result<(), Error> {
        Err(Error::unsupported_feature(
            "JPEG does not support animation",
        ))
    }

    fn begin_frame(&mut self, _duration_ms: u32) -> Result<(), Error> {
        Err(Error::unsupported_feature(
            "JPEG does not support animation",
        ))
    }

    fn push_rows(&mut self, _rows: PixelSlice<'_>) -> Result<(), Error> {
        Err(Error::unsupported_feature(
            "JPEG does not support animation",
        ))
    }

    fn end_frame(&mut self) -> Result<(), Error> {
        Err(Error::unsupported_feature(
            "JPEG does not support animation",
        ))
    }

    fn pull_frame(
        &mut self,
        _duration_ms: u32,
        _source: &mut dyn FnMut(u32, PixelSliceMut<'_>) -> usize,
    ) -> Result<(), Error> {
        Err(Error::unsupported_feature(
            "JPEG does not support animation",
        ))
    }

    fn finish(self) -> Result<EncodeOutput, Error> {
        Err(Error::unsupported_feature(
            "JPEG does not support animation",
        ))
    }
}

// ============================================================================
// Decode side: DecoderConfig → DecodeJob → Decoder / FrameDecoder
// ============================================================================

/// JPEG decoder configuration implementing [`zencodec_types::DecoderConfig`].
///
/// Wraps [`crate::decode::DecodeConfig`] with the zencodec trait interface.
#[derive(Clone, Debug)]
pub struct JpegDecoderConfig {
    #[cfg(feature = "decoder")]
    inner: crate::decode::DecodeConfig,
    #[allow(dead_code)]
    limits: ResourceLimits,
}

impl JpegDecoderConfig {
    /// Create a default decoder config.
    #[must_use]
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "decoder")]
            inner: crate::decode::DecodeConfig::new(),
            limits: ResourceLimits::none(),
        }
    }
}

impl Default for JpegDecoderConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Supported decode pixel formats (native, zero-conversion output).
static DECODE_DESCRIPTORS: &[PixelDescriptor] = &[
    PixelDescriptor::RGB8_SRGB,
    PixelDescriptor::RGBA8_SRGB,
    PixelDescriptor::GRAY8_SRGB,
    PixelDescriptor::BGRA8_SRGB,
];

impl zencodec_types::DecoderConfig for JpegDecoderConfig {
    type Error = Error;
    type Job<'a> = JpegDecodeJob<'a>;

    fn format() -> ImageFormat {
        ImageFormat::Jpeg
    }

    fn supported_descriptors() -> &'static [PixelDescriptor] {
        DECODE_DESCRIPTORS
    }

    fn capabilities() -> &'static CodecCapabilities {
        static CAPS: CodecCapabilities = CodecCapabilities::new()
            .with_decode_icc(true)
            .with_decode_exif(true)
            .with_decode_xmp(true)
            .with_decode_cancel(true)
            .with_native_gray(true)
            .with_cheap_probe(true);
        &CAPS
    }

    fn job(&self) -> Self::Job<'_> {
        JpegDecodeJob {
            config: self,
            stop: None,
            limits: ResourceLimits::none(),
        }
    }

    fn probe_header(&self, data: &[u8]) -> Result<ImageInfo, Self::Error> {
        #[cfg(feature = "decoder")]
        {
            let info = self.inner.read_info(data)?;
            Ok(to_image_info(&info))
        }
        #[cfg(not(feature = "decoder"))]
        {
            let _ = data;
            Err(Error::unsupported_feature(
                "decoder feature required for probing",
            ))
        }
    }
}

// ── Decode job ──────────────────────────────────────────────────────────────

/// Per-operation JPEG decode job.
///
/// Created by [`JpegDecoderConfig::job()`]. Borrows a stop token and is
/// consumed by creating a [`JpegDecoder`] or [`JpegFrameDecoder`].
pub struct JpegDecodeJob<'a> {
    config: &'a JpegDecoderConfig,
    stop: Option<&'a dyn Stop>,
    limits: ResourceLimits,
}

impl<'a> zencodec_types::DecodeJob<'a> for JpegDecodeJob<'a> {
    type Error = Error;
    type Decoder = JpegDecoder<'a>;
    type FrameDecoder = JpegFrameDecoder;

    fn with_stop(mut self, stop: &'a dyn Stop) -> Self {
        self.stop = Some(stop);
        self
    }

    fn with_limits(mut self, limits: ResourceLimits) -> Self {
        self.limits = limits;
        self
    }

    fn output_info(&self, data: &[u8]) -> Result<OutputInfo, Self::Error> {
        #[cfg(feature = "decoder")]
        {
            let info = self.config.inner.read_info(data)?;
            let native_format = match info.num_components {
                1 => PixelDescriptor::GRAY8_SRGB,
                _ => PixelDescriptor::RGB8_SRGB,
            };
            Ok(OutputInfo::full_decode(
                info.dimensions.width,
                info.dimensions.height,
                native_format,
            ))
        }
        #[cfg(not(feature = "decoder"))]
        {
            let _ = data;
            Err(Error::unsupported_feature("decoder feature required"))
        }
    }

    fn decoder(self) -> Self::Decoder {
        JpegDecoder {
            config: self.config,
            stop: self.stop,
            limits: self.limits,
        }
    }

    fn frame_decoder(self, _data: &[u8]) -> Result<Self::FrameDecoder, Self::Error> {
        Err(Error::unsupported_feature(
            "JPEG does not support animation decoding",
        ))
    }
}

// ── Decoder ─────────────────────────────────────────────────────────────────

/// One-shot JPEG decoder implementing [`zencodec_types::Decoder`].
pub struct JpegDecoder<'a> {
    config: &'a JpegDecoderConfig,
    stop: Option<&'a dyn Stop>,
    limits: ResourceLimits,
}

impl<'a> JpegDecoder<'a> {
    /// Build a DecodeConfig with limit overrides applied.
    #[cfg(feature = "decoder")]
    fn build_config(&self) -> crate::decode::DecodeConfig {
        let mut cfg = self.config.inner.clone();
        if let Some(max) = self.limits.max_pixels {
            cfg = cfg.max_pixels(max);
        }
        if let Some(bytes) = self.limits.max_memory_bytes {
            cfg = cfg.max_memory(bytes);
        }
        cfg
    }
}

impl<'a> zencodec_types::Decoder for JpegDecoder<'a> {
    type Error = Error;

    fn decode(self, data: &[u8]) -> Result<DecodeOutput, Error> {
        #[cfg(feature = "decoder")]
        {
            use crate::types::PixelFormat;
            use imgref::ImgVec;
            use rgb::Gray;

            let mut cfg = self.build_config();
            cfg = cfg.preserve_all();

            let stop = self.stop.unwrap_or(&enough::Unstoppable);
            let result = cfg.decode(data, stop)?;

            let w = result.width();
            let h = result.height();
            let format = result.format();

            // Extract metadata before consuming pixels
            let mut info = ImageInfo::new(w, h, ImageFormat::Jpeg);
            if let Some(extras) = result.extras() {
                if let Some(icc) = extras.icc_profile() {
                    info = info.with_icc_profile(icc.to_vec());
                }
                if let Some(exif) = extras.exif() {
                    info = info.with_exif(exif.to_vec());
                }
                if let Some(xmp) = extras.xmp() {
                    info = info.with_xmp(xmp.as_bytes().to_vec());
                }
            }

            let pixel_data = match format {
                PixelFormat::Gray => {
                    let pixels_u8 = result.into_pixels_u8().unwrap_or_default();
                    let gray: Vec<Gray<u8>> = pixels_u8.iter().map(|&v| Gray::new(v)).collect();
                    PixelData::Gray8(ImgVec::new(gray, w as usize, h as usize))
                }
                PixelFormat::Rgb => {
                    let pixels_u8 = result.into_pixels_u8().unwrap_or_default();
                    let rgb = bytes_to_rgb(&pixels_u8);
                    PixelData::Rgb8(ImgVec::new(rgb, w as usize, h as usize))
                }
                _ => {
                    // For other formats, best effort — treat as RGB bytes
                    let pixels_u8 = result.into_pixels_u8().unwrap_or_default();
                    let rgb = bytes_to_rgb(&pixels_u8);
                    PixelData::Rgb8(ImgVec::new(rgb, w as usize, h as usize))
                }
            };

            Ok(DecodeOutput::new(pixel_data, info))
        }

        #[cfg(not(feature = "decoder"))]
        {
            let _ = data;
            Err(Error::unsupported_feature("decoder feature required"))
        }
    }

    fn decode_into(self, data: &[u8], mut dst: PixelSliceMut<'_>) -> Result<ImageInfo, Error> {
        #[cfg(feature = "decoder")]
        {
            use imgref::ImgRefMut;
            use zencodec_types::{ChannelLayout, ChannelType};

            let cfg = self.build_config();
            let info_result = self.config.inner.read_info(data)?;
            let info = to_image_info(&info_result);

            let mut reader = cfg.scanline_reader(data)?;
            let width = reader.width() as usize;

            let desc = dst.descriptor();
            match (desc.channel_type, desc.layout) {
                (ChannelType::U8, ChannelLayout::Rgb) => {
                    for y in 0..dst.rows() {
                        if reader.is_finished() {
                            break;
                        }
                        let row = dst.row_mut(y);
                        let n = (row.len() / 3).min(width);
                        let out = ImgRefMut::new(row, n * 3, 1);
                        let count = reader.read_rows_rgb8(out)?;
                        if count == 0 {
                            break;
                        }
                    }
                    Ok(info)
                }
                (ChannelType::U8, ChannelLayout::Rgba) => {
                    for y in 0..dst.rows() {
                        if reader.is_finished() {
                            break;
                        }
                        let row = dst.row_mut(y);
                        let n = (row.len() / 4).min(width);
                        let out = ImgRefMut::new(row, n * 4, 1);
                        let count = reader.read_rows_rgba8(out)?;
                        if count == 0 {
                            break;
                        }
                    }
                    Ok(info)
                }
                (ChannelType::U8, ChannelLayout::Gray) => {
                    for y in 0..dst.rows() {
                        if reader.is_finished() {
                            break;
                        }
                        let row = dst.row_mut(y);
                        let n = row.len().min(width);
                        let out = ImgRefMut::new(row, n, 1);
                        let count = reader.read_rows_gray8(out)?;
                        if count == 0 {
                            break;
                        }
                    }
                    Ok(info)
                }
                (ChannelType::U8, ChannelLayout::Bgra) => {
                    // Decode as RGBA, then swizzle
                    for y in 0..dst.rows() {
                        if reader.is_finished() {
                            break;
                        }
                        let row = dst.row_mut(y);
                        let n = (row.len() / 4).min(width);
                        let out = ImgRefMut::new(row, n * 4, 1);
                        let count = reader.read_rows_rgbx8(out)?;
                        if count == 0 {
                            break;
                        }
                        // Swizzle RGBX → BGRA in-place
                        for i in 0..n {
                            let base = i * 4;
                            let (r, g, b) = (row[base], row[base + 1], row[base + 2]);
                            row[base] = b;
                            row[base + 1] = g;
                            row[base + 2] = r;
                            row[base + 3] = 255;
                        }
                    }
                    Ok(info)
                }
                _ => {
                    // Unsupported format for decode_into — fall back to decode + convert
                    Err(Error::unsupported_feature(
                        "unsupported pixel format for decode_into; use decode() instead",
                    ))
                }
            }
        }
        #[cfg(not(feature = "decoder"))]
        {
            let _ = (data, dst);
            Err(Error::unsupported_feature("decoder feature required"))
        }
    }

    fn decode_rows(
        self,
        data: &[u8],
        sink: &mut dyn FnMut(u32, PixelSlice<'_>),
    ) -> Result<ImageInfo, Error> {
        #[cfg(feature = "decoder")]
        {
            use imgref::ImgRefMut;

            let cfg = self.build_config();
            let info_result = self.config.inner.read_info(data)?;
            let info = to_image_info(&info_result);

            let mut reader = cfg.scanline_reader(data)?;
            let width = reader.width() as usize;
            let mut row_buf = vec![0u8; width * 3];
            let mut row_idx: u32 = 0;

            while !reader.is_finished() {
                let out = ImgRefMut::new(&mut row_buf, width * 3, 1);
                let count = reader.read_rows_rgb8(out)?;
                if count == 0 {
                    break;
                }
                // Create PixelSlice for this row
                let slice = PixelSlice::new(
                    &row_buf,
                    width as u32,
                    1,
                    width * 3,
                    PixelDescriptor::RGB8_SRGB,
                )
                .map_err(|_| Error::unsupported_feature("buffer error in decode_rows"))?;
                sink(row_idx, slice);
                row_idx += 1;
            }
            Ok(info)
        }
        #[cfg(not(feature = "decoder"))]
        {
            let _ = (data, sink);
            Err(Error::unsupported_feature("decoder feature required"))
        }
    }
}

// ── FrameDecoder (animation — unsupported for JPEG) ─────────────────────────

/// JPEG frame decoder — always returns an error since JPEG doesn't support animation.
pub struct JpegFrameDecoder;

impl zencodec_types::FrameDecoder for JpegFrameDecoder {
    type Error = Error;

    fn next_frame(&mut self) -> Result<Option<DecodeFrame>, Error> {
        Err(Error::unsupported_feature(
            "JPEG does not support animation",
        ))
    }

    fn next_frame_into(
        &mut self,
        _dst: PixelSliceMut<'_>,
        _prior_frame: Option<u32>,
    ) -> Result<Option<ImageInfo>, Error> {
        Err(Error::unsupported_feature(
            "JPEG does not support animation",
        ))
    }

    fn next_frame_rows(
        &mut self,
        _sink: &mut dyn FnMut(u32, PixelSlice<'_>),
    ) -> Result<Option<ImageInfo>, Error> {
        Err(Error::unsupported_feature(
            "JPEG does not support animation",
        ))
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Convert JpegInfo to zencodec_types ImageInfo.
#[cfg(feature = "decoder")]
fn to_image_info(info: &crate::decode::JpegInfo) -> ImageInfo {
    let mut img_info = ImageInfo::new(
        info.dimensions.width,
        info.dimensions.height,
        ImageFormat::Jpeg,
    );

    if let Some(ref icc) = info.icc_profile {
        img_info = img_info.with_icc_profile(icc.clone());
    }
    if let Some(ref exif) = info.exif {
        img_info = img_info.with_exif(exif.clone());
    }
    if let Some(ref xmp) = info.xmp {
        img_info = img_info.with_xmp(xmp.as_bytes().to_vec());
    }

    img_info
}

/// Convert raw bytes (3 bytes per pixel) to Vec<Rgb<u8>>.
fn bytes_to_rgb(bytes: &[u8]) -> Vec<rgb::Rgb<u8>> {
    bytes
        .chunks_exact(3)
        .map(|c| rgb::Rgb {
            r: c[0],
            g: c[1],
            b: c[2],
        })
        .collect()
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use imgref::{Img, ImgExt};
    use rgb::{Gray, Rgb, Rgba};
    use zencodec_types::{
        DecodeJob as _, Decoder as _, DecoderConfig as _, EncodeJob as _, Encoder as _,
        EncoderConfig as _,
    };

    #[test]
    fn encoding_default_roundtrip() {
        let enc = JpegEncoderConfig::new().with_calibrated_quality(80.0);
        let pixels = vec![
            Rgb {
                r: 128,
                g: 64,
                b: 32,
            };
            64
        ];
        let img = Img::new(pixels.as_slice(), 8, 8);
        let output = enc.encode_rgb8(img.as_ref()).unwrap();
        assert!(!output.bytes().is_empty());
        assert_eq!(output.format(), ImageFormat::Jpeg);
        // Verify it starts with JPEG SOI marker
        assert_eq!(&output.bytes()[0..2], &[0xFF, 0xD8]);
    }

    #[test]
    fn encoding_with_metadata() {
        let enc = JpegEncoderConfig::new().with_calibrated_quality(85.0);
        let pixels: Vec<Rgb<u8>> = vec![Rgb { r: 255, g: 0, b: 0 }; 16];
        let img = Img::new(pixels.as_slice(), 4, 4);

        let icc = b"fake icc profile data";
        let meta = ImageMetadata::default().with_icc(icc.as_slice());
        let output = enc
            .job()
            .with_metadata(&meta)
            .encoder()
            .encode(PixelSlice::from(img.as_ref()))
            .unwrap();
        assert!(!output.bytes().is_empty());
    }

    #[test]
    fn encoding_gray8() {
        let enc = JpegEncoderConfig::grayscale(90.0);
        let pixels = vec![Gray::new(128u8); 64];
        let img = Img::new(pixels.as_slice(), 8, 8);
        let output = enc.encode_gray8(img.as_ref()).unwrap();
        assert!(!output.bytes().is_empty());
        assert_eq!(output.format(), ImageFormat::Jpeg);
    }

    #[test]
    fn encoding_rgba8_strips_alpha() {
        let enc = JpegEncoderConfig::new().with_calibrated_quality(85.0);
        let pixels = vec![
            Rgba {
                r: 100,
                g: 150,
                b: 200,
                a: 128,
            };
            64
        ];
        let img = Img::new(pixels.as_slice(), 8, 8);
        let output = enc.encode_rgba8(img.as_ref()).unwrap();
        assert!(!output.bytes().is_empty());
    }

    #[test]
    fn push_rows_encode() {
        let enc = JpegEncoderConfig::new().with_calibrated_quality(85.0);
        let pixels: Vec<Rgb<u8>> = vec![
            Rgb {
                r: 128,
                g: 64,
                b: 32,
            };
            8 * 8
        ];
        let img = Img::new(pixels.as_slice(), 8, 8);
        let slice = PixelSlice::from(img.as_ref());

        let mut encoder = enc.job().encoder();
        // Push 4 rows, then 4 more
        let top = slice.sub_rows(0, 4);
        let bottom = slice.sub_rows(4, 4);
        encoder.push_rows(top).unwrap();
        encoder.push_rows(bottom).unwrap();
        let output = encoder.finish().unwrap();
        assert!(!output.bytes().is_empty());
        assert_eq!(&output.bytes()[0..2], &[0xFF, 0xD8]);
    }

    #[cfg(feature = "decoder")]
    #[test]
    fn decode_roundtrip() {
        // Encode
        let enc = JpegEncoderConfig::new().with_calibrated_quality(95.0);
        let pixels = vec![
            Rgb {
                r: 200,
                g: 100,
                b: 50,
            };
            64
        ];
        let img = Img::new(pixels.as_slice(), 8, 8);
        let encoded = enc.encode_rgb8(img.as_ref()).unwrap();

        // Decode
        let dec = JpegDecoderConfig::new();
        let output = dec.decode(encoded.bytes()).unwrap();
        assert_eq!(output.info().width, 8);
        assert_eq!(output.info().height, 8);
        assert_eq!(output.info().format, ImageFormat::Jpeg);
    }

    #[cfg(feature = "decoder")]
    #[test]
    fn probe_info() {
        let enc = JpegEncoderConfig::new().with_calibrated_quality(85.0);
        let pixels = vec![Rgb { r: 0, g: 0, b: 0 }; 100];
        let img = Img::new(pixels.as_slice(), 10, 10);
        let encoded = enc.encode_rgb8(img.as_ref()).unwrap();

        let dec = JpegDecoderConfig::new();
        let info = dec.probe_header(encoded.bytes()).unwrap();
        assert_eq!(info.width, 10);
        assert_eq!(info.height, 10);
        assert_eq!(info.format, ImageFormat::Jpeg);
    }

    #[cfg(feature = "decoder")]
    #[test]
    fn decode_into_rgb8() {
        let enc = JpegEncoderConfig::new().with_calibrated_quality(95.0);
        let pixels = vec![
            Rgb {
                r: 200,
                g: 100,
                b: 50
            };
            64
        ];
        let img = Img::new(pixels.as_slice(), 8, 8);
        let encoded = enc.encode_rgb8(img.as_ref()).unwrap();

        let dec = JpegDecoderConfig::new();
        let dst = vec![Rgb { r: 0u8, g: 0, b: 0 }; 64];
        let mut dst_img = imgref::ImgVec::new(dst, 8, 8);
        let info = dec
            .decode_into_rgb8(encoded.bytes(), dst_img.as_mut())
            .unwrap();
        assert_eq!(info.width, 8);
        assert_eq!(info.height, 8);
    }

    #[cfg(feature = "decoder")]
    #[test]
    fn decode_rows_callback() {
        let enc = JpegEncoderConfig::new().with_calibrated_quality(95.0);
        let pixels = vec![
            Rgb {
                r: 200,
                g: 100,
                b: 50
            };
            64
        ];
        let img = Img::new(pixels.as_slice(), 8, 8);
        let encoded = enc.encode_rgb8(img.as_ref()).unwrap();

        let dec = JpegDecoderConfig::new();
        let mut row_count = 0u32;
        let info = dec
            .job()
            .decoder()
            .decode_rows(encoded.bytes(), &mut |_row_idx, _slice| {
                row_count += 1;
            })
            .unwrap();
        assert_eq!(info.width, 8);
        assert_eq!(info.height, 8);
        assert_eq!(row_count, 8);
    }
}
