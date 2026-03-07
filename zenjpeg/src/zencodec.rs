//! zencodec-types trait implementations for zenjpeg.
//!
//! Provides [`JpegEncoderConfig`] and [`JpegDecoderConfig`] types that implement
//! the encode/decode trait hierarchy from zencodec-types, wrapping the native
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
//! | `FrameEncoder` | `()` (JPEG has no animation) |
//! | `DecoderConfig` | [`JpegDecoderConfig`] |
//! | `DecodeJob<'a>` | [`JpegDecodeJob`] |
//! | `Decode` | [`JpegDecoder`] |
//! | `StreamingDecode` | [`JpegStreamingDecoder`] |
//! | `FrameDecode` | `Unsupported<Error>` (JPEG has no animation) |

extern crate alloc;
use alloc::vec::Vec;

use rgb::{Gray, Rgb};
use zc::{
    ImageFormat, ImageInfo, MetadataView, ResourceLimits, UnsupportedOperation,
    Unsupported,
};
use zc::encode::{EncodeCapabilities, EncodeOutput};
use zc::decode::{DecodeCapabilities, DecodeOutput, OutputInfo};
use zenpixels::{PixelBuffer, PixelDescriptor, PixelSlice};

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
// Encode side: EncoderConfig → EncodeJob → Encoder
// ============================================================================

/// JPEG encode capabilities.
static JPEG_ENCODE_CAPS: EncodeCapabilities = EncodeCapabilities::new()
    .with_icc(true)
    .with_exif(true)
    .with_xmp(true)
    .with_cancel(true)
    .with_lossy(true)
    .with_row_level(true)
    .with_native_gray(true)
    .with_native_16bit(true)
    .with_native_f32(true)
    .with_quality_range(0.0, 100.0);

/// JPEG encoder configuration implementing [`zc::encode::EncoderConfig`].
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

    /// Convenience: encode pixels with this config via the type-erased path.
    pub fn encode(&self, pixels: PixelSlice<'_>) -> Result<EncodeOutput, Error> {
        use zc::encode::{EncodeJob as _, Encoder as _, EncoderConfig as _};
        self.job().encoder()?.encode(pixels)
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

impl zc::encode::EncoderConfig for JpegEncoderConfig {
    type Error = Error;
    type Job<'a> = JpegEncodeJob<'a>;

    fn format() -> ImageFormat {
        ImageFormat::Jpeg
    }

    fn supported_descriptors() -> &'static [PixelDescriptor] {
        ENCODE_DESCRIPTORS
    }

    fn capabilities() -> &'static EncodeCapabilities {
        &JPEG_ENCODE_CAPS
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
/// metadata) and is consumed by creating a [`JpegEncoder`].
pub struct JpegEncodeJob<'a> {
    config: &'a JpegEncoderConfig,
    stop: Option<&'a dyn enough::Stop>,
    metadata: Option<&'a MetadataView<'a>>,
    limits: ResourceLimits,
}

impl<'a> zc::encode::EncodeJob<'a> for JpegEncodeJob<'a> {
    type Error = Error;
    type Enc = JpegEncoder<'a>;
    type FrameEnc = ();

    fn with_stop(mut self, stop: &'a dyn enough::Stop) -> Self {
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
        Err(UnsupportedOperation::AnimationEncode.into())
    }
}

// ── Encoder ─────────────────────────────────────────────────────────────────

/// Single-image JPEG encoder implementing [`zc::encode::Encoder`].
///
/// Supports both one-shot `encode()` and streaming `push_rows()` + `finish()`.
pub struct JpegEncoder<'a> {
    config: &'a JpegEncoderConfig,
    stop: Option<&'a dyn enough::Stop>,
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
}

impl zc::encode::Encoder for JpegEncoder<'_> {
    type Error = Error;

    fn reject(op: UnsupportedOperation) -> Self::Error {
        Error::from(op)
    }

    fn preferred_strip_height(&self) -> u32 {
        // JPEG MCU height: 16 for 4:2:0 (2x2 chroma), 8 for 4:4:4/4:2:2
        16
    }

    fn encode(self, pixels: PixelSlice<'_>) -> Result<EncodeOutput, Error> {
        let layout = descriptor_to_layout(pixels.descriptor())?;
        let width = pixels.width();
        let height = pixels.rows();
        let data = pixels.contiguous_bytes();
        self.encode_bytes_inner(&data, width, height, layout)
    }

    fn push_rows(&mut self, rows: PixelSlice<'_>) -> Result<(), Error> {
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

    fn finish(mut self) -> Result<EncodeOutput, Error> {
        let buf = self
            .buffer
            .take()
            .ok_or_else(|| Error::unsupported_feature("finish() called without any push_rows()"))?;
        let layout = descriptor_to_layout(buf.descriptor)?;
        self.encode_bytes_inner(&buf.data, buf.width, buf.total_rows, layout)
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

// ============================================================================
// Decode side: DecoderConfig → DecodeJob → Decoder / StreamingDecoder
// ============================================================================

/// JPEG decode capabilities.
static JPEG_DECODE_CAPS: DecodeCapabilities = DecodeCapabilities::new()
    .with_icc(true)
    .with_exif(true)
    .with_xmp(true)
    .with_cancel(true)
    .with_cheap_probe(true)
    .with_row_level(true)
    .with_native_gray(true)
    .with_native_f32(true);

/// JPEG decoder configuration implementing [`zc::decode::DecoderConfig`].
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

    /// Access the underlying [`DecodeConfig`](crate::decode::DecodeConfig).
    #[cfg(feature = "decoder")]
    #[must_use]
    pub fn inner(&self) -> &crate::decode::DecodeConfig {
        &self.inner
    }

    /// Mutable access to the underlying [`DecodeConfig`](crate::decode::DecodeConfig).
    #[cfg(feature = "decoder")]
    pub fn inner_mut(&mut self) -> &mut crate::decode::DecodeConfig {
        &mut self.inner
    }

    /// Convenience: probe image header with this config.
    pub fn probe_header(&self, data: &[u8]) -> Result<ImageInfo, Error> {
        use zc::decode::{DecodeJob as _, DecoderConfig as _};
        self.job().probe(data)
    }

    /// Convenience: probe full image metadata (may be expensive).
    pub fn probe_full_metadata(&self, data: &[u8]) -> Result<ImageInfo, Error> {
        use zc::decode::{DecodeJob as _, DecoderConfig as _};
        self.job().probe_full(data)
    }

    /// Convenience: decode image with this config.
    pub fn decode(&self, data: &[u8]) -> Result<DecodeOutput, Error> {
        use zc::decode::{Decode as _, DecodeJob as _, DecoderConfig as _};
        self.job().decoder(data, &[])?.decode()
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

impl zc::decode::DecoderConfig for JpegDecoderConfig {
    type Error = Error;
    type Job<'a> = JpegDecodeJob<'a>;

    fn format() -> ImageFormat {
        ImageFormat::Jpeg
    }

    fn supported_descriptors() -> &'static [PixelDescriptor] {
        DECODE_DESCRIPTORS
    }

    fn capabilities() -> &'static DecodeCapabilities {
        &JPEG_DECODE_CAPS
    }

    fn job(&self) -> Self::Job<'_> {
        JpegDecodeJob {
            config: self,
            stop: None,
            limits: ResourceLimits::none(),
            crop_hint: None,
            orientation: zc::OrientationHint::default(),
        }
    }
}

// ── Decode job ──────────────────────────────────────────────────────────────

/// Per-operation JPEG decode job.
///
/// Created by [`JpegDecoderConfig::job()`]. Borrows a stop token and is
/// consumed by creating a [`JpegDecoder`] or [`JpegStreamingDecoder`].
pub struct JpegDecodeJob<'a> {
    config: &'a JpegDecoderConfig,
    stop: Option<&'a dyn enough::Stop>,
    limits: ResourceLimits,
    crop_hint: Option<(u32, u32, u32, u32)>,
    orientation: zc::OrientationHint,
}

impl<'a> zc::decode::DecodeJob<'a> for JpegDecodeJob<'a> {
    type Error = Error;
    type Dec = JpegDecoder<'a>;
    type StreamDec = JpegStreamingDecoder<'a>;
    type FrameDec = Unsupported<Error>;

    fn with_stop(mut self, stop: &'a dyn enough::Stop) -> Self {
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

    fn with_orientation(mut self, hint: zc::OrientationHint) -> Self {
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
            if will_orient
                && let Some(ref exif) = info.exif
                && let Some(orient_val) = crate::lossless::parse_exif_orientation(exif)
            {
                let orient = zc::Orientation::from_exif(orient_val as u16);
                if orient.swaps_dimensions() {
                    core::mem::swap(&mut w, &mut h);
                }
                out = OutputInfo::full_decode(w, h, native_format).with_orientation_applied(orient);
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
        data: &'a [u8],
        preferred: &[PixelDescriptor],
    ) -> Result<Self::StreamDec, Self::Error> {
        #[cfg(feature = "decoder")]
        {
            let cfg = build_decode_config(
                &self.config.inner,
                &self.limits,
                self.crop_hint,
                self.orientation,
            );
            let header = self.config.inner.read_info(data)?;
            let info = to_image_info(&header);
            let reader = cfg.scanline_reader(data)?;

            // Select output format based on preference and component count
            let descriptor = select_decode_descriptor(preferred, header.num_components);

            Ok(JpegStreamingDecoder {
                reader,
                info,
                descriptor,
                row_buf: Vec::new(),
                current_row: 0,
            })
        }
        #[cfg(not(feature = "decoder"))]
        {
            let _ = (data, preferred);
            Err(Error::unsupported_feature(
                "decoder feature required for streaming decode",
            ))
        }
    }

    fn frame_decoder(
        self,
        _data: &'a [u8],
        _preferred: &[PixelDescriptor],
    ) -> Result<Self::FrameDec, Self::Error> {
        Err(UnsupportedOperation::AnimationDecode.into())
    }
}

/// Whether the given orientation hint means we should auto-orient during decode.
fn will_auto_orient(hint: zc::OrientationHint) -> bool {
    use zc::OrientationHint;
    match hint {
        OrientationHint::Preserve => false,
        OrientationHint::Correct | OrientationHint::CorrectAndTransform(_) => true,
        OrientationHint::ExactTransform(_) => false,
        _ => false,
    }
}

/// Build a DecodeConfig with limit overrides and hints applied.
#[cfg(feature = "decoder")]
fn build_decode_config(
    inner: &crate::decode::DecodeConfig,
    limits: &ResourceLimits,
    crop_hint: Option<(u32, u32, u32, u32)>,
    orientation: zc::OrientationHint,
) -> crate::decode::DecodeConfig {
    let mut cfg = inner.clone();
    if let Some(max) = limits.max_pixels {
        cfg = cfg.max_pixels(max);
    }
    if let Some(bytes) = limits.max_memory_bytes {
        cfg = cfg.max_memory(bytes);
    }
    if let Some((x, y, w, h)) = crop_hint {
        cfg = cfg.crop(crate::decode::CropRegion::pixels(x, y, w, h));
    }
    if !will_auto_orient(orientation) {
        cfg = cfg.auto_orient(false);
    }
    cfg
}

/// Select the appropriate pixel descriptor for decode output based on preferences
/// and component count.
#[cfg(feature = "decoder")]
fn select_decode_descriptor(preferred: &[PixelDescriptor], num_components: u8) -> PixelDescriptor {
    use zenpixels::{ChannelLayout, ChannelType};

    let is_gray = num_components == 1;

    // Check if caller has a preference we can satisfy
    for &desc in preferred {
        let ch = desc.channel_type();
        let layout = desc.layout();

        match (is_gray, ch, layout) {
            (true, ChannelType::U8, ChannelLayout::Gray) => return PixelDescriptor::GRAY8_SRGB,
            (true, ChannelType::F32, ChannelLayout::Gray) => return PixelDescriptor::GRAYF32_LINEAR,
            (false, ChannelType::U8, ChannelLayout::Rgb) => return PixelDescriptor::RGB8_SRGB,
            (false, ChannelType::U8, ChannelLayout::Rgba) => return PixelDescriptor::RGBA8_SRGB,
            (false, ChannelType::U8, ChannelLayout::Bgra) => return PixelDescriptor::BGRA8_SRGB,
            (false, ChannelType::F32, ChannelLayout::Rgb) => return PixelDescriptor::RGBF32_LINEAR,
            (false, ChannelType::F32, ChannelLayout::Rgba) => {
                return PixelDescriptor::RGBAF32_LINEAR;
            }
            _ => {}
        }
    }

    // Default: u8, RGB or Gray
    if is_gray {
        PixelDescriptor::GRAY8_SRGB
    } else {
        PixelDescriptor::RGB8_SRGB
    }
}

// ── Decoder ─────────────────────────────────────────────────────────────────

/// One-shot JPEG decoder implementing [`zc::decode::Decode`].
pub struct JpegDecoder<'a> {
    config: &'a JpegDecoderConfig,
    stop: Option<&'a dyn enough::Stop>,
    limits: ResourceLimits,
    crop_hint: Option<(u32, u32, u32, u32)>,
    orientation: zc::OrientationHint,
    data: &'a [u8],
    preferred: Vec<PixelDescriptor>,
}

impl zc::decode::Decode for JpegDecoder<'_> {
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

            let limits = self.limits;
            let mut cfg = build_decode_config(
                &self.config.inner,
                &limits,
                self.crop_hint,
                self.orientation,
            );
            cfg = cfg.preserve_all();

            if wants_f32 {
                cfg = cfg.output_target(OutputTarget::LinearF32);
            }

            // Check max_width/max_height before full decode (header parse is cheap)
            if limits.max_width.is_some() || limits.max_height.is_some() {
                let header = cfg.read_info(data)?;
                limits.check_dimensions(header.dimensions.width, header.dimensions.height)?;
            }

            let stop = self.stop.unwrap_or(&enough::Unstoppable);
            let mut result = cfg.decode(data, stop)?;

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
                        info = info.with_orientation(zc::Orientation::from_exif(
                            orient as u16,
                        ));
                    }
                    info = info.with_exif(exif.to_vec());
                }
                if let Some(xmp) = extras.xmp() {
                    info = info.with_xmp(xmp.as_bytes().to_vec());
                }
            }

            // Take extras before consuming result for pixels
            let jpeg_extras = result.take_extras();

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

            let mut output = DecodeOutput::new(buf, info);
            if let Some(extras) = jpeg_extras {
                output = output.with_extras(extras);
            }
            Ok(output)
        }

        #[cfg(not(feature = "decoder"))]
        {
            Err(Error::unsupported_feature("decoder feature required"))
        }
    }
}

// ── StreamingDecode ─────────────────────────────────────────────────────────

/// Streaming JPEG decoder implementing [`zc::decode::StreamingDecode`].
///
/// Wraps zenjpeg's `ScanlineReader` to yield scanline batches via `next_batch()`.
/// Each batch is a single row of decoded pixels.
pub struct JpegStreamingDecoder<'a> {
    #[cfg(feature = "decoder")]
    reader: crate::decode::ScanlineReader<'a>,
    info: ImageInfo,
    descriptor: PixelDescriptor,
    /// Reusable row buffer for decoded pixel data.
    row_buf: Vec<u8>,
    current_row: u32,
    #[cfg(not(feature = "decoder"))]
    _phantom: core::marker::PhantomData<&'a ()>,
}

impl zc::decode::StreamingDecode for JpegStreamingDecoder<'_> {
    type Error = Error;

    fn next_batch(&mut self) -> Result<Option<(u32, PixelSlice<'_>)>, Error> {
        #[cfg(feature = "decoder")]
        {
            use imgref::ImgRefMut;
            use zenpixels::{ChannelLayout, ChannelType};

            if self.reader.is_finished() {
                return Ok(None);
            }

            let width = self.reader.width() as usize;
            let bpp = self.descriptor.bytes_per_pixel();
            let row_bytes = width * bpp;
            self.row_buf.resize(row_bytes, 0);

            let out = ImgRefMut::new(&mut self.row_buf, row_bytes, 1);
            let ch_type = self.descriptor.channel_type();
            let ch_layout = self.descriptor.layout();

            let count = match (ch_type, ch_layout) {
                (ChannelType::U8, ChannelLayout::Gray) => self.reader.read_rows_gray8(out)?,
                (ChannelType::U8, ChannelLayout::Rgb) => self.reader.read_rows_rgb8(out)?,
                (ChannelType::U8, ChannelLayout::Rgba) => self.reader.read_rows_rgba8(out)?,
                (ChannelType::U8, ChannelLayout::Bgra) => {
                    self.reader.read_rows_bgra8(out)?
                }
                (ChannelType::F32, ChannelLayout::Gray) => {
                    // f32 gray requires f32 output buffer
                    let float_count = width;
                    let float_bytes = float_count * 4;
                    self.row_buf.resize(float_bytes, 0);
                    let float_slice: &mut [f32] = bytemuck::cast_slice_mut(&mut self.row_buf);
                    let f_out = ImgRefMut::new(float_slice, width, 1);
                    self.reader.read_rows_gray_f32(f_out)?
                }
                (ChannelType::F32, ChannelLayout::Rgb | ChannelLayout::Rgba) => {
                    // f32 RGBA uses 4 channels
                    let channels = if ch_layout == ChannelLayout::Rgba {
                        4
                    } else {
                        // read_rows_rgba_f32 always writes 4 channels
                        4
                    };
                    let float_count = width * channels;
                    let float_bytes = float_count * 4;
                    self.row_buf.resize(float_bytes, 0);
                    let float_slice: &mut [f32] = bytemuck::cast_slice_mut(&mut self.row_buf);
                    let f_out = ImgRefMut::new(float_slice, width * channels, 1);
                    self.reader.read_rows_rgba_f32(f_out)?
                }
                _ => {
                    return Err(Error::unsupported_feature(
                        "unsupported pixel format for streaming decode",
                    ));
                }
            };

            if count == 0 {
                return Ok(None);
            }

            let y = self.current_row;
            self.current_row += count as u32;

            let stride = row_bytes;
            let slice = PixelSlice::new(
                &self.row_buf[..row_bytes * count],
                width as u32,
                count as u32,
                stride,
                self.descriptor,
            )
            .map_err(|_| Error::internal("streaming decode: pixel slice construction failed"))?;

            Ok(Some((y, slice)))
        }

        #[cfg(not(feature = "decoder"))]
        {
            Err(Error::unsupported_feature(
                "decoder feature required for streaming decode",
            ))
        }
    }

    fn info(&self) -> &ImageInfo {
        &self.info
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Convert JpegInfo to zc ImageInfo.
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
                img_info.with_orientation(zc::Orientation::from_exif(orient as u16));
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
    use zc::encode::{
        EncodeJob as _, Encoder as _, EncoderConfig as _,
    };

    #[test]
    fn encoding_default_roundtrip() {
        let enc = JpegEncoderConfig::new().with_calibrated_quality(80.0);
        let pixels: Vec<Rgb<u8>> = vec![
            Rgb {
                r: 128,
                g: 64,
                b: 32,
            };
            64
        ];
        let img = Img::new(pixels.as_slice(), 8, 8);
        let output = enc.encode(PixelSlice::from(img.as_ref()).into()).unwrap();
        assert!(!output.data().is_empty());
        assert_eq!(output.format(), ImageFormat::Jpeg);
        // Verify it starts with JPEG SOI marker
        assert_eq!(&output.data()[0..2], &[0xFF, 0xD8]);
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
            .encode(PixelSlice::from(img.as_ref()).into())
            .unwrap();
        assert!(!output.data().is_empty());
    }

    #[test]
    fn encoding_gray8() {
        let enc = JpegEncoderConfig::grayscale(90.0);
        let pixels = vec![Gray::new(128u8); 64];
        let img = Img::new(pixels.as_slice(), 8, 8);
        let output = enc.encode(PixelSlice::from(img.as_ref()).into()).unwrap();
        assert!(!output.data().is_empty());
        assert_eq!(output.format(), ImageFormat::Jpeg);
    }

    #[test]
    fn encoding_rgba8_strips_alpha() {
        let enc = JpegEncoderConfig::new().with_calibrated_quality(85.0);
        let pixels: Vec<Rgba<u8>> = vec![
            Rgba {
                r: 100,
                g: 150,
                b: 200,
                a: 128,
            };
            64
        ];
        let img = Img::new(pixels.as_slice(), 8, 8);
        let output = enc.encode(PixelSlice::from(img.as_ref()).into()).unwrap();
        assert!(!output.data().is_empty());
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
        let slice: PixelSlice<'_> = PixelSlice::from(img.as_ref()).into();

        let mut encoder = enc.job().encoder().unwrap();
        // Push 4 rows, then 4 more
        let top = slice.sub_rows(0, 4);
        let bottom = slice.sub_rows(4, 4);
        encoder.push_rows(top).unwrap();
        encoder.push_rows(bottom).unwrap();
        let output = encoder.finish().unwrap();
        assert!(!output.data().is_empty());
        assert_eq!(&output.data()[0..2], &[0xFF, 0xD8]);
    }

    #[cfg(feature = "decoder")]
    #[test]
    fn decode_roundtrip() {
        // Encode
        let enc = JpegEncoderConfig::new().with_calibrated_quality(95.0);
        let pixels: Vec<Rgb<u8>> = vec![
            Rgb {
                r: 200,
                g: 100,
                b: 50,
            };
            64
        ];
        let img = Img::new(pixels.as_slice(), 8, 8);
        let encoded = enc.encode(PixelSlice::from(img.as_ref()).into()).unwrap();

        // Decode
        let dec = JpegDecoderConfig::new();
        let output = dec.decode(encoded.data()).unwrap();
        assert_eq!(output.info().width, 8);
        assert_eq!(output.info().height, 8);
        assert_eq!(output.info().format, ImageFormat::Jpeg);
    }

    #[cfg(feature = "decoder")]
    #[test]
    fn probe_info() {
        let enc = JpegEncoderConfig::new().with_calibrated_quality(85.0);
        let pixels: Vec<Rgb<u8>> = vec![Rgb { r: 0, g: 0, b: 0 }; 100];
        let img = Img::new(pixels.as_slice(), 10, 10);
        let encoded = enc.encode(PixelSlice::from(img.as_ref()).into()).unwrap();

        let dec = JpegDecoderConfig::new();
        let info = dec.probe_header(encoded.data()).unwrap();
        assert_eq!(info.width, 10);
        assert_eq!(info.height, 10);
        assert_eq!(info.format, ImageFormat::Jpeg);
    }

    #[cfg(feature = "decoder")]
    #[test]
    fn streaming_decode_roundtrip() {
        use zc::decode::{DecodeJob as _, DecoderConfig as _, StreamingDecode as _};

        // Encode a test image
        let enc = JpegEncoderConfig::new().with_calibrated_quality(95.0);
        let pixels: Vec<Rgb<u8>> = vec![
            Rgb {
                r: 200,
                g: 100,
                b: 50,
            };
            16 * 16
        ];
        let img = Img::new(pixels.as_slice(), 16, 16);
        let encoded = enc.encode(PixelSlice::from(img.as_ref()).into()).unwrap();

        // Stream decode
        let dec = JpegDecoderConfig::new();
        let mut stream = dec
            .job()
            .streaming_decoder(encoded.data(), &[PixelDescriptor::RGB8_SRGB])
            .unwrap();

        assert_eq!(stream.info().width, 16);
        assert_eq!(stream.info().height, 16);

        let mut total_rows = 0u32;
        while let Some((y, batch)) = stream.next_batch().unwrap() {
            assert_eq!(y, total_rows);
            assert_eq!(batch.width(), 16);
            total_rows += batch.rows();
        }
        assert_eq!(total_rows, 16);
    }

    // ── Encoder trait roundtrip tests ────────────────────────────────

    /// Helper: encode via the type-erased Encoder trait, verify output is valid JPEG.
    fn encoder_trait_roundtrip(pixels: zenpixels::PixelSlice<'_>) {
        use zc::encode::Encoder;
        let config = JpegEncoderConfig::new().with_calibrated_quality(75.0);
        let encoder = config.job().encoder().unwrap();
        let output = encoder.encode(pixels).unwrap();
        assert!(!output.is_empty());
        assert_eq!(output.format(), ImageFormat::Jpeg);
        assert_eq!(&output.data()[0..2], &[0xFF, 0xD8]);
    }

    #[test]
    fn encoder_trait_rgb8() {
        let pixels: Vec<Rgb<u8>> = (0..16 * 16)
            .map(|i| Rgb {
                r: (i % 256) as u8,
                g: ((i * 3) % 256) as u8,
                b: ((i * 7) % 256) as u8,
            })
            .collect();
        let img = Img::new(pixels.as_slice(), 16, 16);
        encoder_trait_roundtrip(zenpixels::PixelSlice::from(img.as_ref()).into());
    }

    #[test]
    fn encoder_trait_rgba8() {
        let pixels: Vec<Rgba<u8>> = (0..16 * 16)
            .map(|i| Rgba {
                r: (i % 256) as u8,
                g: 128,
                b: 64,
                a: 255,
            })
            .collect();
        let img = Img::new(pixels.as_slice(), 16, 16);
        encoder_trait_roundtrip(zenpixels::PixelSlice::from(img.as_ref()).into());
    }

    #[test]
    fn encoder_trait_gray8() {
        let pixels: Vec<Gray<u8>> = (0..16 * 16).map(|i| Gray((i % 256) as u8)).collect();
        let img = Img::new(pixels.as_slice(), 16, 16);
        encoder_trait_roundtrip(zenpixels::PixelSlice::from(img.as_ref()).into());
    }

    #[test]
    fn encoder_trait_rgb16() {
        let pixels: Vec<Rgb<u16>> = (0..16 * 16)
            .map(|i| Rgb {
                r: (i * 256) as u16,
                g: ((i * 3 * 256) % 65536) as u16,
                b: 0,
            })
            .collect();
        let img = Img::new(pixels.as_slice(), 16, 16);
        encoder_trait_roundtrip(zenpixels::PixelSlice::from(img.as_ref()).into());
    }

    #[test]
    fn encoder_trait_rgba16() {
        let pixels: Vec<Rgba<u16>> = (0..16 * 16)
            .map(|i| Rgba {
                r: (i * 256) as u16,
                g: 32768,
                b: 16384,
                a: 65535,
            })
            .collect();
        let img = Img::new(pixels.as_slice(), 16, 16);
        encoder_trait_roundtrip(zenpixels::PixelSlice::from(img.as_ref()).into());
    }

    #[test]
    fn encoder_trait_gray16() {
        let pixels: Vec<Gray<u16>> = (0..16 * 16).map(|i| Gray((i * 256) as u16)).collect();
        let img = Img::new(pixels.as_slice(), 16, 16);
        encoder_trait_roundtrip(zenpixels::PixelSlice::from(img.as_ref()).into());
    }

    #[test]
    fn encoder_trait_rgb_f32() {
        let pixels: Vec<Rgb<f32>> = (0..16 * 16)
            .map(|i| {
                let t = i as f32 / 255.0;
                Rgb {
                    r: t,
                    g: t * 0.5,
                    b: t * 0.25,
                }
            })
            .collect();
        let img = Img::new(pixels.as_slice(), 16, 16);
        encoder_trait_roundtrip(zenpixels::PixelSlice::from(img.as_ref()).into());
    }

    #[test]
    fn encoder_trait_rgba_f32() {
        let pixels: Vec<Rgba<f32>> = (0..16 * 16)
            .map(|i| {
                let t = i as f32 / 255.0;
                Rgba {
                    r: t,
                    g: t * 0.5,
                    b: t * 0.25,
                    a: 1.0,
                }
            })
            .collect();
        let img = Img::new(pixels.as_slice(), 16, 16);
        encoder_trait_roundtrip(zenpixels::PixelSlice::from(img.as_ref()).into());
    }

    #[test]
    fn encoder_trait_gray_f32() {
        let pixels: Vec<Gray<f32>> = (0..16 * 16).map(|i| Gray(i as f32 / 255.0)).collect();
        let img = Img::new(pixels.as_slice(), 16, 16);
        encoder_trait_roundtrip(zenpixels::PixelSlice::from(img.as_ref()).into());
    }

    #[test]
    fn encoder_trait_dyn_encoder() {
        let pixels: Vec<Rgb<u8>> = vec![
            Rgb {
                r: 100,
                g: 150,
                b: 200,
            };
            32 * 32
        ];
        let img = Img::new(pixels.as_slice(), 32, 32);
        let config = JpegEncoderConfig::new().with_calibrated_quality(80.0);
        let dyn_enc = config.job().dyn_encoder().unwrap();
        let output = dyn_enc.encode(zenpixels::PixelSlice::from(img.as_ref()).into()).unwrap();
        assert!(!output.is_empty());
        assert_eq!(output.format(), ImageFormat::Jpeg);
    }

    #[test]
    fn capabilities_encode() {
        use zc::encode::EncoderConfig;
        let caps = JpegEncoderConfig::capabilities();
        assert!(caps.icc());
        assert!(caps.exif());
        assert!(caps.xmp());
        assert!(caps.cancel());
        assert!(caps.lossy());
        assert!(!caps.lossless());
        assert!(!caps.animation());
        assert!(caps.row_level());
        assert!(caps.native_gray());
        assert!(caps.quality_range().is_some());
    }

    #[cfg(feature = "decoder")]
    #[test]
    fn capabilities_decode() {
        use zc::decode::DecoderConfig;
        let caps = JpegDecoderConfig::capabilities();
        assert!(caps.icc());
        assert!(caps.exif());
        assert!(caps.xmp());
        assert!(caps.cancel());
        assert!(caps.cheap_probe());
        assert!(caps.row_level());
        assert!(caps.native_gray());
        assert!(!caps.animation());
    }

    #[test]
    fn decode_trait_max_width_enforced() {
        use zc::decode::{Decode as _, DecodeJob as _, DecoderConfig as _};

        // Encode a 32x32 test image
        let pixels: Vec<Rgb<u8>> = vec![
            Rgb {
                r: 128,
                g: 64,
                b: 32
            };
            32 * 32
        ];
        let img = Img::new(pixels.as_slice(), 32, 32);
        let encoded = JpegEncoderConfig::new()
            .encode(PixelSlice::from(img.as_ref()).into())
            .unwrap();

        // Decode with max_width=10 should fail
        let dec = JpegDecoderConfig::new();
        let limits = ResourceLimits::none().with_max_width(10);
        let result = dec
            .job()
            .with_limits(limits)
            .decoder(encoded.data(), &[])
            .unwrap()
            .decode();
        assert!(result.is_err(), "should reject image wider than max_width");
    }

    #[test]
    fn decode_trait_max_height_enforced() {
        use zc::decode::{Decode as _, DecodeJob as _, DecoderConfig as _};

        let pixels: Vec<Rgb<u8>> = vec![
            Rgb {
                r: 128,
                g: 64,
                b: 32
            };
            32 * 32
        ];
        let img = Img::new(pixels.as_slice(), 32, 32);
        let encoded = JpegEncoderConfig::new()
            .encode(PixelSlice::from(img.as_ref()).into())
            .unwrap();

        let dec = JpegDecoderConfig::new();
        let limits = ResourceLimits::none().with_max_height(10);
        let result = dec
            .job()
            .with_limits(limits)
            .decoder(encoded.data(), &[])
            .unwrap()
            .decode();
        assert!(
            result.is_err(),
            "should reject image taller than max_height"
        );
    }

    #[test]
    fn decode_trait_generous_dimensions_ok() {
        use zc::decode::{Decode as _, DecodeJob as _, DecoderConfig as _};

        let pixels: Vec<Rgb<u8>> = vec![
            Rgb {
                r: 128,
                g: 64,
                b: 32
            };
            32 * 32
        ];
        let img = Img::new(pixels.as_slice(), 32, 32);
        let encoded = JpegEncoderConfig::new()
            .encode(PixelSlice::from(img.as_ref()).into())
            .unwrap();

        let dec = JpegDecoderConfig::new();
        let limits = ResourceLimits::none()
            .with_max_width(1000)
            .with_max_height(1000);
        let result = dec
            .job()
            .with_limits(limits)
            .decoder(encoded.data(), &[])
            .unwrap()
            .decode();
        assert!(
            result.is_ok(),
            "generous limits should not reject 32x32 image"
        );
    }

    #[test]
    fn frame_encoder_returns_unsupported() {
        let config = JpegEncoderConfig::new();
        let result = config.job().frame_encoder();
        assert!(result.is_err());
    }

    #[cfg(feature = "decoder")]
    #[test]
    fn frame_decoder_returns_unsupported() {
        use zc::decode::{DecodeJob as _, DecoderConfig as _};

        let dec = JpegDecoderConfig::new();
        let result = dec.job().frame_decoder(&[], &[]);
        assert!(result.is_err());
    }
}
