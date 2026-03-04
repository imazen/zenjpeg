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
//! | `EncodeRgb8` etc. | [`JpegEncoder`] |
//! | `FrameEncodeRgb8` etc. | [`JpegFrameEncoder`] |
//! | `DecoderConfig` | [`JpegDecoderConfig`] |
//! | `DecodeJob<'a>` | [`JpegDecodeJob`] |
//! | `Decode` | [`JpegDecoder`] |
//! | `FrameDecode` | [`JpegFrameDecoder`] |

extern crate alloc;
use alloc::vec::Vec;

use rgb::{Gray, Rgb, Rgba};
use zencodec_types::{
    DecodeFrame, DecodeOutput, EncodeOutput, ImageFormat, ImageInfo, MetadataView, OutputInfo,
    ResourceLimits, Stop,
};
use zenpixels::{PixelBuffer, PixelDescriptor, PixelSlice, PixelSliceMut};

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

    /// Set encoding quality using calibrated perceptual scale.
    #[must_use]
    pub fn with_calibrated_quality(mut self, quality: f32) -> Self {
        let q = quality.clamp(0.0, 100.0);
        self.quality = q;
        self.inner = self.inner.quality(Quality::ApproxJpegli(q));
        self
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

    /// Convenience: encode RGB8 pixels with this config.
    pub fn encode_rgb8(&self, img: imgref::ImgRef<'_, Rgb<u8>>) -> Result<EncodeOutput, Error> {
        use zencodec_types::{EncodeJob as _, EncodeRgb8 as _, EncoderConfig as _};
        self.job().encoder()?.encode_rgb8(PixelSlice::from(img))
    }

    /// Convenience: encode RGBA8 pixels with this config.
    pub fn encode_rgba8(&self, img: imgref::ImgRef<'_, Rgba<u8>>) -> Result<EncodeOutput, Error> {
        use zencodec_types::{EncodeJob as _, EncodeRgba8 as _, EncoderConfig as _};
        self.job().encoder()?.encode_rgba8(PixelSlice::from(img))
    }

    /// Convenience: encode Gray8 pixels with this config.
    pub fn encode_gray8(&self, img: imgref::ImgRef<'_, Gray<u8>>) -> Result<EncodeOutput, Error> {
        use zencodec_types::{EncodeGray8 as _, EncodeJob as _, EncoderConfig as _};
        self.job().encoder()?.encode_gray8(PixelSlice::from(img))
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

    fn with_generic_quality(mut self, quality: f32) -> Self {
        let q = quality.clamp(0.0, 100.0);
        self.quality = q;
        self.inner = self.inner.quality(Quality::ApproxJpegli(q));
        self
    }

    fn generic_quality(&self) -> Option<f32> {
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
    metadata: Option<&'a MetadataView<'a>>,
    limits: ResourceLimits,
}

impl<'a> zencodec_types::EncodeJob<'a> for JpegEncodeJob<'a> {
    type Error = Error;
    type Enc = JpegEncoder<'a>;
    type FrameEnc = JpegFrameEncoder;

    fn with_stop(mut self, stop: &'a dyn Stop) -> Self {
        self.stop = Some(stop);
        self
    }

    fn with_metadata(mut self, meta: &'a MetadataView<'a>) -> Self {
        self.metadata = Some(meta);
        self
    }

    fn with_limits(mut self, limits: ResourceLimits) -> Self {
        self.limits = limits;
        self
    }

    fn encoder(self) -> Result<Self::Enc, Self::Error> {
        Ok(JpegEncoder {
            config: self.config,
            stop: self.stop,
            metadata: self.metadata,
            limits: self.limits,
            buffer: None,
        })
    }

    fn frame_encoder(self) -> Result<Self::FrameEnc, Self::Error> {
        Err(Error::unsupported_feature(
            "JPEG does not support animation encoding",
        ))
    }
}

// ── Encoder ─────────────────────────────────────────────────────────────────

/// Single-image JPEG encoder.
///
/// Implements per-format encode traits (`EncodeRgb8`, `EncodeRgba8`, etc.)
/// and provides inherent `push_rows()` + `finish()` for streaming encode.
pub struct JpegEncoder<'a> {
    config: &'a JpegEncoderConfig,
    stop: Option<&'a dyn Stop>,
    metadata: Option<&'a MetadataView<'a>>,
    limits: ResourceLimits,
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
        // Pre-flight limit checks
        self.limits.check_dimensions(width, height).map_err(|_| {
            Error::image_too_large(
                width as u64 * height as u64,
                self.limits.max_pixels.unwrap_or(0),
            )
        })?;
        let estimated_mem = width as u64 * height as u64 * layout.bytes_per_pixel() as u64;
        self.limits.check_memory(estimated_mem).map_err(|_| {
            Error::allocation_failed(estimated_mem as usize, "memory limit exceeded")
        })?;

        let req = self.build_request();
        let output = req.encode_bytes(data, width, height, layout)?;
        Ok(EncodeOutput::new(output, ImageFormat::Jpeg))
    }

    /// Push rows for streaming encode.
    pub fn push_rows<P>(&mut self, rows: PixelSlice<'_, P>) -> Result<(), Error> {
        let rows = rows.erase();
        let desc = rows.descriptor();
        let width = rows.width();

        match &mut self.buffer {
            None => {
                // First push — initialize buffer with contiguous row data
                let bpp = desc.bytes_per_pixel();
                let row_bytes = width as usize * bpp;
                let mut data = Vec::with_capacity(row_bytes * rows.rows() as usize * 4); // estimate
                data.extend_from_slice(&rows.contiguous_bytes());
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
                buf.data.extend_from_slice(&rows.contiguous_bytes());
                buf.total_rows += rows.rows();
            }
        }
        Ok(())
    }

    /// Finish a streaming encode started with `push_rows()`.
    pub fn finish(mut self) -> Result<EncodeOutput, Error> {
        let buf = self
            .buffer
            .take()
            .ok_or_else(|| Error::unsupported_feature("finish() called without any push_rows()"))?;
        let layout = descriptor_to_layout(buf.descriptor)?;
        self.encode_bytes_inner(&buf.data, buf.width, buf.total_rows, layout)
    }

    /// Type-erased single-shot encode (for backwards compat).
    pub fn encode<P>(self, pixels: PixelSlice<'_, P>) -> Result<EncodeOutput, Error> {
        let pixels = pixels.erase();
        let layout = descriptor_to_layout(pixels.descriptor())?;
        let width = pixels.width();
        let height = pixels.rows();
        let data = pixels.contiguous_bytes();
        self.encode_bytes_inner(&data, width, height, layout)
    }
}

/// Map a PixelDescriptor to a zenjpeg PixelLayout.
fn descriptor_to_layout(desc: PixelDescriptor) -> Result<PixelLayout, Error> {
    use zenpixels::{ChannelLayout, ChannelType, TransferFunction};
    match (desc.channel_type(), desc.layout(), desc.transfer()) {
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

// ── Per-format encode trait impls ────────────────────────────────────────────

impl zencodec_types::EncodeRgb8 for JpegEncoder<'_> {
    type Error = Error;
    fn encode_rgb8(self, pixels: PixelSlice<'_, Rgb<u8>>) -> Result<EncodeOutput, Error> {
        let w = pixels.width();
        let h = pixels.rows();
        let data = pixels.contiguous_bytes();
        self.encode_bytes_inner(&data, w, h, PixelLayout::Rgb8Srgb)
    }
}

impl zencodec_types::EncodeRgba8 for JpegEncoder<'_> {
    type Error = Error;
    fn encode_rgba8(self, pixels: PixelSlice<'_, Rgba<u8>>) -> Result<EncodeOutput, Error> {
        let w = pixels.width();
        let h = pixels.rows();
        let data = pixels.contiguous_bytes();
        self.encode_bytes_inner(&data, w, h, PixelLayout::Rgba8Srgb)
    }
}

impl zencodec_types::EncodeGray8 for JpegEncoder<'_> {
    type Error = Error;
    fn encode_gray8(self, pixels: PixelSlice<'_, Gray<u8>>) -> Result<EncodeOutput, Error> {
        let w = pixels.width();
        let h = pixels.rows();
        let data = pixels.contiguous_bytes();
        self.encode_bytes_inner(&data, w, h, PixelLayout::Gray8Srgb)
    }
}

impl zencodec_types::EncodeRgb16 for JpegEncoder<'_> {
    type Error = Error;
    fn encode_rgb16(self, pixels: PixelSlice<'_, Rgb<u16>>) -> Result<EncodeOutput, Error> {
        let w = pixels.width();
        let h = pixels.rows();
        let data = pixels.contiguous_bytes();
        self.encode_bytes_inner(&data, w, h, PixelLayout::Rgb16Linear)
    }
}

impl zencodec_types::EncodeRgba16 for JpegEncoder<'_> {
    type Error = Error;
    fn encode_rgba16(self, pixels: PixelSlice<'_, Rgba<u16>>) -> Result<EncodeOutput, Error> {
        let w = pixels.width();
        let h = pixels.rows();
        let data = pixels.contiguous_bytes();
        self.encode_bytes_inner(&data, w, h, PixelLayout::Rgba16Linear)
    }
}

impl zencodec_types::EncodeGray16 for JpegEncoder<'_> {
    type Error = Error;
    fn encode_gray16(self, pixels: PixelSlice<'_, Gray<u16>>) -> Result<EncodeOutput, Error> {
        let w = pixels.width();
        let h = pixels.rows();
        let data = pixels.contiguous_bytes();
        self.encode_bytes_inner(&data, w, h, PixelLayout::Gray16Linear)
    }
}

impl zencodec_types::EncodeRgbF32 for JpegEncoder<'_> {
    type Error = Error;
    fn encode_rgb_f32(self, pixels: PixelSlice<'_, Rgb<f32>>) -> Result<EncodeOutput, Error> {
        let w = pixels.width();
        let h = pixels.rows();
        let data = pixels.contiguous_bytes();
        self.encode_bytes_inner(&data, w, h, PixelLayout::RgbF32Linear)
    }
}

impl zencodec_types::EncodeRgbaF32 for JpegEncoder<'_> {
    type Error = Error;
    fn encode_rgba_f32(self, pixels: PixelSlice<'_, Rgba<f32>>) -> Result<EncodeOutput, Error> {
        let w = pixels.width();
        let h = pixels.rows();
        let data = pixels.contiguous_bytes();
        self.encode_bytes_inner(&data, w, h, PixelLayout::RgbaF32Linear)
    }
}

impl zencodec_types::EncodeGrayF32 for JpegEncoder<'_> {
    type Error = Error;
    fn encode_gray_f32(self, pixels: PixelSlice<'_, Gray<f32>>) -> Result<EncodeOutput, Error> {
        let w = pixels.width();
        let h = pixels.rows();
        let data = pixels.contiguous_bytes();
        self.encode_bytes_inner(&data, w, h, PixelLayout::GrayF32Linear)
    }
}

// ── FrameEncoder (animation — unsupported for JPEG) ─────────────────────────

/// JPEG frame encoder — always returns an error since JPEG doesn't support animation.
pub struct JpegFrameEncoder;

impl zencodec_types::FrameEncodeRgb8 for JpegFrameEncoder {
    type Error = Error;

    fn push_frame_rgb8(
        &mut self,
        _pixels: PixelSlice<'_, Rgb<u8>>,
        _duration_ms: u32,
    ) -> Result<(), Error> {
        Err(Error::unsupported_feature(
            "JPEG does not support animation",
        ))
    }

    fn finish_rgb8(self) -> Result<EncodeOutput, Error> {
        Err(Error::unsupported_feature(
            "JPEG does not support animation",
        ))
    }
}

impl zencodec_types::FrameEncodeRgba8 for JpegFrameEncoder {
    type Error = Error;

    fn push_frame_rgba8(
        &mut self,
        _pixels: PixelSlice<'_, Rgba<u8>>,
        _duration_ms: u32,
    ) -> Result<(), Error> {
        Err(Error::unsupported_feature(
            "JPEG does not support animation",
        ))
    }

    fn finish_rgba8(self) -> Result<EncodeOutput, Error> {
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

    /// Convenience: probe image header with this config.
    pub fn probe_header(&self, data: &[u8]) -> Result<ImageInfo, Error> {
        use zencodec_types::{DecodeJob as _, DecoderConfig as _};
        self.job().probe(data)
    }

    /// Convenience: probe full image metadata (may be expensive).
    pub fn probe_full(&self, data: &[u8]) -> Result<ImageInfo, Error> {
        use zencodec_types::{DecodeJob as _, DecoderConfig as _};
        self.job().probe_full(data)
    }

    /// Convenience: decode image with this config.
    pub fn decode(&self, data: &[u8]) -> Result<DecodeOutput, Error> {
        use zencodec_types::{Decode as _, DecodeJob as _, DecoderConfig as _};
        self.job().decoder(data, &[])?.decode()
    }

    /// Convenience: decode into a pre-allocated RGB8 buffer.
    pub fn decode_into_rgb8(
        &self,
        data: &[u8],
        dst: imgref::ImgRefMut<'_, Rgb<u8>>,
    ) -> Result<ImageInfo, Error> {
        use zencodec_types::{DecodeJob as _, DecoderConfig as _};
        self.job()
            .decoder(data, &[])?
            .decode_into(PixelSliceMut::from(dst))
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
    PixelDescriptor::RGBF32_LINEAR,
    PixelDescriptor::GRAYF32_LINEAR,
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

    fn job(&self) -> Self::Job<'_> {
        JpegDecodeJob {
            config: self,
            stop: None,
            limits: ResourceLimits::none(),
            crop_hint: None,
            orientation: zencodec_types::OrientationHint::default(),
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
    crop_hint: Option<(u32, u32, u32, u32)>,
    orientation: zencodec_types::OrientationHint,
}

impl<'a> zencodec_types::DecodeJob<'a> for JpegDecodeJob<'a> {
    type Error = Error;
    type Dec = JpegDecoder<'a>;
    type StreamDec = JpegNoStreaming;
    type FrameDec = JpegFrameDecoder;

    fn with_stop(mut self, stop: &'a dyn Stop) -> Self {
        self.stop = Some(stop);
        self
    }

    fn with_limits(mut self, limits: ResourceLimits) -> Self {
        self.limits = limits;
        self
    }

    fn with_crop_hint(mut self, x: u32, y: u32, width: u32, height: u32) -> Self {
        self.crop_hint = Some((x, y, width, height));
        self
    }

    fn with_orientation(mut self, hint: zencodec_types::OrientationHint) -> Self {
        self.orientation = hint;
        self
    }

    fn probe(&self, data: &[u8]) -> Result<ImageInfo, Self::Error> {
        #[cfg(feature = "decoder")]
        {
            let info = self.config.inner.read_info(data)?;
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

    fn output_info(&self, data: &[u8]) -> Result<OutputInfo, Self::Error> {
        #[cfg(feature = "decoder")]
        {
            let info = self.config.inner.read_info(data)?;
            let native_format = match info.num_components {
                1 => PixelDescriptor::GRAY8_SRGB,
                _ => PixelDescriptor::RGB8_SRGB,
            };
            let mut w = info.dimensions.width;
            let mut h = info.dimensions.height;

            let mut out = OutputInfo::full_decode(w, h, native_format);

            // Determine if orientation correction should be applied.
            let will_orient = will_auto_orient(self.orientation);
            if will_orient {
                if let Some(ref exif) = info.exif {
                    if let Some(orient_val) = crate::lossless::parse_exif_orientation(exif) {
                        let orient = zencodec_types::Orientation::from_exif(orient_val as u16);
                        if orient.swaps_dimensions() {
                            core::mem::swap(&mut w, &mut h);
                        }
                        out = OutputInfo::full_decode(w, h, native_format)
                            .with_orientation_applied(orient);
                    }
                }
            }

            // Report crop that will be applied (may be MCU-snapped by the decoder).
            if let Some((x, y, cw, ch)) = self.crop_hint {
                out = out.with_crop_applied([x, y, cw, ch]);
            }

            Ok(out)
        }
        #[cfg(not(feature = "decoder"))]
        {
            let _ = data;
            Err(Error::unsupported_feature("decoder feature required"))
        }
    }

    fn decoder(
        self,
        data: &'a [u8],
        preferred: &[PixelDescriptor],
    ) -> Result<Self::Dec, Self::Error> {
        Ok(JpegDecoder {
            config: self.config,
            stop: self.stop,
            limits: self.limits,
            crop_hint: self.crop_hint,
            orientation: self.orientation,
            data,
            preferred: preferred.to_vec(),
        })
    }

    fn streaming_decoder(
        self,
        _data: &'a [u8],
        _preferred: &[PixelDescriptor],
    ) -> Result<Self::StreamDec, Self::Error> {
        Err(Error::unsupported_feature(
            "JPEG does not support streaming decode",
        ))
    }

    fn frame_decoder(
        self,
        _data: &'a [u8],
        _preferred: &[PixelDescriptor],
    ) -> Result<Self::FrameDec, Self::Error> {
        Err(Error::unsupported_feature(
            "JPEG does not support animation decoding",
        ))
    }
}

/// Whether the given orientation hint means we should auto-orient during decode.
fn will_auto_orient(hint: zencodec_types::OrientationHint) -> bool {
    use zencodec_types::OrientationHint;
    match hint {
        OrientationHint::Preserve => false,
        OrientationHint::Correct | OrientationHint::CorrectAndTransform(_) => true,
        OrientationHint::ExactTransform(_) => false,
        _ => false,
    }
}

// ── Decoder ─────────────────────────────────────────────────────────────────

/// One-shot JPEG decoder implementing [`zencodec_types::Decode`].
pub struct JpegDecoder<'a> {
    config: &'a JpegDecoderConfig,
    stop: Option<&'a dyn Stop>,
    limits: ResourceLimits,
    crop_hint: Option<(u32, u32, u32, u32)>,
    orientation: zencodec_types::OrientationHint,
    data: &'a [u8],
    preferred: Vec<PixelDescriptor>,
}

impl<'a> JpegDecoder<'a> {
    /// Build a DecodeConfig with limit overrides and hints applied.
    #[cfg(feature = "decoder")]
    fn build_config(&self) -> crate::decode::DecodeConfig {
        let mut cfg = self.config.inner.clone();
        if let Some(max) = self.limits.max_pixels {
            cfg = cfg.max_pixels(max);
        }
        if let Some(bytes) = self.limits.max_memory_bytes {
            cfg = cfg.max_memory(bytes);
        }
        if let Some((x, y, w, h)) = self.crop_hint {
            cfg = cfg.crop(crate::decode::CropRegion::pixels(x, y, w, h));
        }
        // Map OrientationHint to auto_orient flag.
        // zenjpeg's auto_orient reads EXIF and applies lossless DCT rotation.
        if !will_auto_orient(self.orientation) {
            cfg = cfg.auto_orient(false);
        }
        cfg
    }

    /// Decode into a pre-allocated buffer.
    pub fn decode_into<P>(
        self,
        dst: PixelSliceMut<'_, P>,
    ) -> Result<ImageInfo, Error> {
        let data = self.data;
        let mut dst = dst.erase();
        #[cfg(feature = "decoder")]
        {
            use imgref::ImgRefMut;
            use zenpixels::{ChannelLayout, ChannelType};

            let cfg = self.build_config();
            let info_result = self.config.inner.read_info(data)?;
            let info = to_image_info(&info_result);

            let mut reader = cfg.scanline_reader(data)?;
            let width = reader.width() as usize;

            let desc = dst.descriptor();
            match (desc.channel_type(), desc.layout()) {
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

    /// Decode rows into a sink (streaming decode).
    pub fn decode_rows(
        self,
        sink: &mut dyn zencodec_types::DecodeRowSink,
    ) -> Result<ImageInfo, Error> {
        let data = self.data;
        #[cfg(feature = "decoder")]
        {
            use imgref::ImgRefMut;

            let cfg = self.build_config();
            let info_result = self.config.inner.read_info(data)?;
            let info = to_image_info(&info_result);

            let mut reader = cfg.scanline_reader(data)?;
            let width = reader.width() as usize;
            let descriptor = if info_result.num_components == 1 {
                PixelDescriptor::GRAY8_SRGB
            } else {
                PixelDescriptor::RGB8_SRGB
            };
            let channels = descriptor.bytes_per_pixel();
            let row_bytes = width * channels;
            let mut y: u32 = 0;

            while !reader.is_finished() {
                let mut ps = sink.demand(y, 1, width as u32, descriptor);
                let buf = ps.row_mut(0);
                let out = ImgRefMut::new(&mut buf[..row_bytes], row_bytes, 1);
                let count = match channels {
                    1 => reader.read_rows_gray8(out)?,
                    _ => reader.read_rows_rgb8(out)?,
                };
                if count == 0 {
                    break;
                }
                y += 1;
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

impl zencodec_types::Decode for JpegDecoder<'_> {
    type Error = Error;

    fn decode(self) -> Result<DecodeOutput, Error> {
        #[cfg(feature = "decoder")]
        {
            use crate::decode::OutputTarget;
            use crate::types::PixelFormat;
            use zenpixels::ChannelType;

            let data = self.data;
            let preferred = &self.preferred;

            // Check if caller wants f32 output
            let wants_f32 = preferred
                .iter()
                .any(|d| d.channel_type() == ChannelType::F32);

            let mut cfg = self.build_config();
            cfg = cfg.preserve_all();

            if wants_f32 {
                cfg = cfg.output_target(OutputTarget::LinearF32);
            }

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
                    if let Some(orient) = crate::lossless::parse_exif_orientation(exif) {
                        info = info.with_orientation(zencodec_types::Orientation::from_exif(
                            orient as u16,
                        ));
                    }
                    info = info.with_exif(exif.to_vec());
                }
                if let Some(xmp) = extras.xmp() {
                    info = info.with_xmp(xmp.as_bytes().to_vec());
                }
            }

            let buf = if wants_f32 {
                // f32 linear output path
                let pixels_f32 = result.into_pixels_f32().unwrap_or_default();
                match format {
                    PixelFormat::Gray => {
                        let gray: Vec<Gray<f32>> =
                            pixels_f32.iter().map(|&v| Gray::new(v)).collect();
                        PixelBuffer::from_pixels(gray, w, h)
                            .map_err(|_| Error::internal("pixel count mismatch"))?
                            .with_descriptor(PixelDescriptor::GRAYF32_LINEAR)
                            .into()
                    }
                    _ => {
                        let rgb: Vec<Rgb<f32>> = pixels_f32
                            .chunks_exact(3)
                            .map(|c| Rgb {
                                r: c[0],
                                g: c[1],
                                b: c[2],
                            })
                            .collect();
                        PixelBuffer::from_pixels(rgb, w, h)
                            .map_err(|_| Error::internal("pixel count mismatch"))?
                            .with_descriptor(PixelDescriptor::RGBF32_LINEAR)
                            .into()
                    }
                }
            } else {
                // u8 sRGB output path (default — JPEG is 8-bit, so lossless)
                match format {
                    PixelFormat::Gray => {
                        let pixels_u8 = result.into_pixels_u8().unwrap_or_default();
                        let gray: Vec<Gray<u8>> = pixels_u8.iter().map(|&v| Gray::new(v)).collect();
                        PixelBuffer::from_pixels(gray, w, h)
                            .map_err(|_| Error::internal("pixel count mismatch"))?
                            .with_descriptor(PixelDescriptor::GRAY8_SRGB)
                            .into()
                    }
                    PixelFormat::Rgb => {
                        let pixels_u8 = result.into_pixels_u8().unwrap_or_default();
                        let rgb = bytes_to_rgb(&pixels_u8);
                        PixelBuffer::from_pixels(rgb, w, h)
                            .map_err(|_| Error::internal("pixel count mismatch"))?
                            .with_descriptor(PixelDescriptor::RGB8_SRGB)
                            .into()
                    }
                    _ => {
                        let pixels_u8 = result.into_pixels_u8().unwrap_or_default();
                        let rgb = bytes_to_rgb(&pixels_u8);
                        PixelBuffer::from_pixels(rgb, w, h)
                            .map_err(|_| Error::internal("pixel count mismatch"))?
                            .with_descriptor(PixelDescriptor::RGB8_SRGB)
                            .into()
                    }
                }
            };

            Ok(DecodeOutput::new(buf, info))
        }

        #[cfg(not(feature = "decoder"))]
        {
            Err(Error::unsupported_feature("decoder feature required"))
        }
    }
}

// ── StreamingDecode (unsupported for JPEG) ──────────────────────────────────

/// JPEG streaming decoder stub — JPEG does not support streaming decode.
pub struct JpegNoStreaming;

impl zencodec_types::StreamingDecode for JpegNoStreaming {
    type Error = Error;

    fn next_batch(&mut self) -> Result<Option<(u32, PixelSlice<'_>)>, Self::Error> {
        unreachable!("JPEG does not support streaming decode")
    }

    fn info(&self) -> &ImageInfo {
        unreachable!("JPEG does not support streaming decode")
    }
}

// ── FrameDecoder (animation — unsupported for JPEG) ─────────────────────────

/// JPEG frame decoder — always returns an error since JPEG doesn't support animation.
pub struct JpegFrameDecoder;

impl zencodec_types::FrameDecode for JpegFrameDecoder {
    type Error = Error;

    fn next_frame(&mut self) -> Result<Option<DecodeFrame>, Error> {
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
        if let Some(orient) = crate::lossless::parse_exif_orientation(exif) {
            img_info =
                img_info.with_orientation(zencodec_types::Orientation::from_exif(orient as u16));
        }
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
    use zencodec_types::{DecodeJob as _, DecoderConfig as _, EncodeJob as _, EncoderConfig as _};

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
        let meta = MetadataView::default().with_icc(icc.as_slice());
        let output = enc
            .job()
            .with_metadata(&meta)
            .encoder()
            .unwrap()
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

        let mut encoder = enc.job().encoder().unwrap();
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
    fn decode_rows_sink() {
        struct CountSink {
            buf: Vec<u8>,
            row_count: u32,
        }
        impl zencodec_types::DecodeRowSink for CountSink {
            fn demand(
                &mut self,
                _y: u32,
                height: u32,
                width: u32,
                descriptor: PixelDescriptor,
            ) -> PixelSliceMut<'_> {
                self.row_count += 1;
                let bpp = descriptor.bytes_per_pixel();
                let stride = width as usize * bpp;
                let needed = height as usize * stride;
                self.buf.resize(needed, 0);
                PixelSliceMut::new(&mut self.buf, width, height, stride, descriptor)
                    .expect("buffer sized correctly")
            }
        }

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
        let mut sink = CountSink {
            buf: Vec::new(),
            row_count: 0,
        };
        let info = dec
            .job()
            .decoder(encoded.bytes(), &[])
            .unwrap()
            .decode_rows(&mut sink)
            .unwrap();
        assert_eq!(info.width, 8);
        assert_eq!(info.height, 8);
        assert_eq!(sink.row_count, 8);
    }
}
