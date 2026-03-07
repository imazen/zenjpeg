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
//! | `FullFrameEncoder` | `()` (JPEG has no animation) |
//! | `DecoderConfig` | [`JpegDecoderConfig`] |
//! | `DecodeJob<'a>` | [`JpegDecodeJob`] |
//! | `Decode` | [`JpegDecoder`] |
//! | `StreamingDecode` | [`JpegStreamingDecoder`] |
//! | `FullFrameDecode` | `Unsupported<Error>` (JPEG has no animation) |

extern crate alloc;
use alloc::borrow::Cow;
use alloc::vec::Vec;

use rgb::{Gray, Rgb};
use zc::decode::{DecodeCapabilities, DecodeOutput, OutputInfo};
use zc::encode::{EncodeCapabilities, EncodeOutput};
use zc::{ImageFormat, ImageInfo, MetadataView, ResourceLimits, Unsupported, UnsupportedOperation};
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
    .with_enforces_max_pixels(true)
    .with_enforces_max_memory(true)
    .with_quality_range(0.0, 100.0)
    .with_effort_range(0, 2);

/// JPEG encoder configuration implementing [`zc::encode::EncoderConfig`].
///
/// Wraps [`EncoderConfig`] with the zencodec trait interface.
/// Defaults to YCbCr 4:2:0 at quality 85.
#[derive(Clone, Debug)]
pub struct JpegEncoderConfig {
    inner: EncoderConfig,
    quality: f32,
    effort: i32,
}

impl JpegEncoderConfig {
    /// Create a default YCbCr 4:2:0 config at quality 85.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter),
            quality: 85.0,
            effort: 1,
        }
    }

    /// Create a YCbCr config with quality and subsampling.
    #[must_use]
    pub fn ycbcr(quality: f32, subsampling: ChromaSubsampling) -> Self {
        Self {
            inner: EncoderConfig::ycbcr(quality, subsampling),
            quality,
            effort: 1,
        }
    }

    /// Create a grayscale config with quality.
    #[must_use]
    pub fn grayscale(quality: f32) -> Self {
        Self {
            inner: EncoderConfig::grayscale(quality),
            quality,
            effort: 1,
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

    /// Apply effort level, returning a modified config.
    fn effective_config(&self) -> EncoderConfig {
        use crate::encode::encoder_types::OptimizationPreset;
        let preset = match self.effort {
            0 => OptimizationPreset::JpegliBaseline,
            2 => OptimizationPreset::HybridMaxCompression,
            _ => OptimizationPreset::HybridProgressive,
        };
        self.inner.clone().optimization(preset)
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
    PixelDescriptor::RGBX8_SRGB,
    PixelDescriptor::BGRX8_SRGB,
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

    fn with_generic_effort(mut self, effort: i32) -> Self {
        self.effort = effort.clamp(0, 2);
        self
    }

    fn generic_effort(&self) -> Option<i32> {
        Some(self.effort)
    }

    fn job(&self) -> Self::Job<'_> {
        JpegEncodeJob {
            config: self,
            stop: None,
            metadata: None,
            limits: ResourceLimits::none(),
            policy: None,
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
    policy: Option<zc::encode::EncodePolicy>,
}

impl<'a> zc::encode::EncodeJob<'a> for JpegEncodeJob<'a> {
    type Error = Error;
    type Enc = JpegEncoder<'a>;
    type FullFrameEnc = ();

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

    fn with_policy(mut self, policy: zc::encode::EncodePolicy) -> Self {
        self.policy = Some(policy);
        self
    }

    fn encoder(self) -> Result<Self::Enc, Self::Error> {
        Ok(JpegEncoder {
            effective_config: self.config.effective_config(),
            stop: self.stop,
            metadata: self.metadata,
            limits: self.limits,
            policy: self.policy,
            accumulator: None,
        })
    }

    fn full_frame_encoder(self) -> Result<Self::FullFrameEnc, Self::Error> {
        Err(UnsupportedOperation::AnimationEncode.into())
    }
}

// ── Encoder ─────────────────────────────────────────────────────────────────

/// Single-image JPEG encoder implementing [`zc::encode::Encoder`].
///
/// Supports one-shot `encode()`, streaming `push_rows()` + `finish()`,
/// and the `encode_srgba8()` convenience method.
pub struct JpegEncoder<'a> {
    effective_config: EncoderConfig,
    stop: Option<&'a dyn enough::Stop>,
    metadata: Option<&'a MetadataView<'a>>,
    limits: ResourceLimits,
    policy: Option<zc::encode::EncodePolicy>,
    /// Accumulated rows for push_rows path. The native BytesEncoder requires
    /// total height at creation time, which the zc trait doesn't provide upfront,
    /// so we accumulate and then stream through the native encoder in finish().
    accumulator: Option<RowAccumulator>,
}

/// Internal buffer for accumulating pushed rows.
struct RowAccumulator {
    data: Vec<u8>,
    width: u32,
    total_rows: u32,
    layout: PixelLayout,
    descriptor: PixelDescriptor,
}

impl<'a> JpegEncoder<'a> {
    /// Build an EncodeRequest from current config + metadata, applying policy.
    fn build_request(&self) -> crate::encode::request::EncodeRequest<'_> {
        let mut req = self.effective_config.request();
        if let Some(meta) = self.metadata {
            let policy = self.policy.unwrap_or_default();
            if policy.resolve_icc(true) {
                if let Some(icc) = meta.icc_profile {
                    req = req.icc_profile(icc);
                }
            }
            if policy.resolve_exif(true) {
                if let Some(exif) = meta.exif {
                    req = req.exif(Exif::raw(exif));
                }
            }
            if policy.resolve_xmp(true) {
                if let Some(xmp) = meta.xmp {
                    req = req.xmp(xmp);
                }
            }
        }
        if let Some(stop) = self.stop {
            req = req.stop(stop);
        }
        req
    }

    /// Pre-flight limit checks.
    fn check_limits(&self, width: u32, height: u32, layout: PixelLayout) -> Result<(), Error> {
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
        Ok(())
    }

    /// Check output size limits after encoding.
    fn check_output_size(&self, output: &[u8]) -> Result<(), Error> {
        self.limits
            .check_output_size(output.len() as u64)
            .map_err(|_| {
                Error::allocation_failed(output.len(), "output exceeds max_output_bytes limit")
            })?;
        Ok(())
    }

    /// One-shot encode from raw bytes.
    fn encode_bytes_inner(
        &self,
        data: &[u8],
        width: u32,
        height: u32,
        layout: PixelLayout,
    ) -> Result<EncodeOutput, Error> {
        self.check_limits(width, height, layout)?;
        let req = self.build_request();
        let output = req.encode_bytes(data, width, height, layout)?;
        self.check_output_size(&output)?;
        Ok(EncodeOutput::new(output, ImageFormat::Jpeg))
    }

    /// Stream accumulated rows through the native BytesEncoder.
    fn encode_accumulated(&self, acc: RowAccumulator) -> Result<EncodeOutput, Error> {
        self.check_limits(acc.width, acc.total_rows, acc.layout)?;

        let req = self.build_request();
        let stop = self.stop.unwrap_or(&enough::Unstoppable);
        let mut enc = req.encode_from_bytes(acc.width, acc.total_rows, acc.layout)?;
        // Stream through native encoder — it processes MCU rows as they arrive
        enc.push_packed(&acc.data, stop)?;
        let output = enc.finish()?;
        self.check_output_size(&output)?;
        Ok(EncodeOutput::new(output, ImageFormat::Jpeg))
    }
}

impl zc::encode::Encoder for JpegEncoder<'_> {
    type Error = Error;

    fn reject(op: UnsupportedOperation) -> Self::Error {
        Error::from(op)
    }

    fn preferred_strip_height(&self) -> u32 {
        16
    }

    fn encode(self, pixels: PixelSlice<'_>) -> Result<EncodeOutput, Error> {
        let layout = descriptor_to_layout(pixels.descriptor())?;
        let width = pixels.width();
        let height = pixels.rows();
        let data = pixels.contiguous_bytes();
        self.encode_bytes_inner(&data, width, height, layout)
    }

    fn encode_srgba8(
        self,
        data: &mut [u8],
        make_opaque: bool,
        width: u32,
        height: u32,
        stride_pixels: u32,
    ) -> Result<EncodeOutput, Error> {
        if make_opaque {
            for chunk in data.chunks_exact_mut(4) {
                chunk[3] = 255;
            }
        }
        let layout = PixelLayout::Rgba8Srgb;
        self.check_limits(width, height, layout)?;
        let req = self.build_request();
        let stop = self.stop.unwrap_or(&enough::Unstoppable);
        let stride_bytes = stride_pixels as usize * 4;
        let mut enc = req.encode_from_bytes(width, height, layout)?;
        enc.push(data, height as usize, stride_bytes, stop)?;
        let output = enc.finish()?;
        self.check_output_size(&output)?;
        Ok(EncodeOutput::new(output, ImageFormat::Jpeg))
    }

    fn push_rows(&mut self, rows: PixelSlice<'_>) -> Result<(), Error> {
        let desc = rows.descriptor();
        let layout = descriptor_to_layout(desc)?;
        let width = rows.width();
        let data = rows.contiguous_bytes();

        match &mut self.accumulator {
            None => {
                let bpp = desc.bytes_per_pixel();
                let row_bytes = width as usize * bpp;
                let estimated_total = row_bytes * rows.rows() as usize * 4;
                let mut buf = Vec::new();
                buf.try_reserve(estimated_total)
                    .map_err(|_| Error::allocation_failed(estimated_total, "push_rows buffer"))?;
                buf.extend_from_slice(&data);
                self.accumulator = Some(RowAccumulator {
                    data: buf,
                    width,
                    total_rows: rows.rows(),
                    layout,
                    descriptor: desc,
                });
            }
            Some(acc) => {
                if acc.width != width || acc.descriptor != desc {
                    return Err(Error::unsupported_feature(
                        "push_rows: width or format changed between calls",
                    ));
                }
                acc.data.extend_from_slice(&data);
                acc.total_rows += rows.rows();
            }
        }
        Ok(())
    }

    fn finish(mut self) -> Result<EncodeOutput, Error> {
        let acc = self
            .accumulator
            .take()
            .ok_or_else(|| Error::unsupported_feature("finish() called without any push_rows()"))?;
        self.encode_accumulated(acc)
    }

    fn encode_from(
        self,
        source: &mut dyn FnMut(u32, PixelSliceMut<'_>) -> usize,
    ) -> Result<EncodeOutput, Error> {
        // Pull-based encode: allocate a strip buffer, call source repeatedly,
        // feed rows to the native streaming encoder.
        // The challenge: we don't know dimensions upfront from the trait.
        // Probe with a small buffer to discover width, then accumulate.
        let _ = source;
        Err(Self::reject(UnsupportedOperation::PullEncode))
    }
}

/// Map a PixelDescriptor to a zenjpeg PixelLayout.
fn descriptor_to_layout(desc: PixelDescriptor) -> Result<PixelLayout, Error> {
    use zenpixels::{AlphaMode, ChannelLayout, ChannelType, TransferFunction};

    match (desc.channel_type(), desc.layout(), desc.transfer()) {
        (ChannelType::U8, ChannelLayout::Rgb, TransferFunction::Srgb) => Ok(PixelLayout::Rgb8Srgb),
        (ChannelType::U8, ChannelLayout::Rgba, TransferFunction::Srgb) => {
            // Distinguish RGBA (has alpha) from RGBX (padding byte)
            if desc.alpha() == Some(AlphaMode::Undefined) {
                Ok(PixelLayout::Rgbx8Srgb)
            } else {
                Ok(PixelLayout::Rgba8Srgb)
            }
        }
        (ChannelType::U8, ChannelLayout::Bgra, TransferFunction::Srgb) => {
            if desc.alpha() == Some(AlphaMode::Undefined) {
                Ok(PixelLayout::Bgrx8Srgb)
            } else {
                Ok(PixelLayout::Bgra8Srgb)
            }
        }
        (ChannelType::U8, ChannelLayout::Gray, TransferFunction::Srgb) => {
            Ok(PixelLayout::Gray8Srgb)
        }
        (
            ChannelType::U16,
            ChannelLayout::Rgb,
            TransferFunction::Srgb | TransferFunction::Unknown,
        ) => Ok(PixelLayout::Rgb16Linear),
        (
            ChannelType::U16,
            ChannelLayout::Rgba,
            TransferFunction::Srgb | TransferFunction::Unknown,
        ) => Ok(PixelLayout::Rgba16Linear),
        (
            ChannelType::U16,
            ChannelLayout::Gray,
            TransferFunction::Srgb | TransferFunction::Unknown,
        ) => Ok(PixelLayout::Gray16Linear),
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
    .with_native_f32(true)
    .with_enforces_max_pixels(true)
    .with_enforces_max_memory(true);

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
        self.job().decoder(Cow::Borrowed(data), &[])?.decode()
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
    PixelDescriptor::RGBX8_SRGB,
    PixelDescriptor::BGRX8_SRGB,
    PixelDescriptor::RGBF32_LINEAR,
    PixelDescriptor::RGBAF32_LINEAR,
    PixelDescriptor::GRAYF32_LINEAR,
];

impl zc::decode::DecoderConfig for JpegDecoderConfig {
    type Error = Error;
    type Job<'a> = JpegDecodeJob<'a>;

    fn formats() -> &'static [ImageFormat] {
        &[ImageFormat::Jpeg]
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
            policy: None,
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
    policy: Option<zc::decode::DecodePolicy>,
}

impl<'a> zc::decode::DecodeJob<'a> for JpegDecodeJob<'a> {
    type Error = Error;
    type Dec = JpegDecoder<'a>;
    type StreamDec = JpegStreamingDecoder<'a>;
    type FullFrameDec = Unsupported<Error>;

    fn with_stop(mut self, stop: &'a dyn enough::Stop) -> Self {
        self.stop = Some(stop);
        self
    }

    fn with_limits(mut self, limits: ResourceLimits) -> Self {
        self.limits = limits;
        self
    }

    fn with_policy(mut self, policy: zc::decode::DecodePolicy) -> Self {
        self.policy = Some(policy);
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
            // Check input size limits
            self.check_input_size(data)?;
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
            self.check_input_size(data)?;
            let info = self.config.inner.read_info(data)?;
            let native_format = match info.num_components {
                1 => PixelDescriptor::GRAY8_SRGB,
                _ => PixelDescriptor::RGB8_SRGB,
            };
            let mut w = info.dimensions.width;
            let mut h = info.dimensions.height;

            let mut out = OutputInfo::full_decode(w, h, native_format);

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
        data: Cow<'a, [u8]>,
        preferred: &[PixelDescriptor],
    ) -> Result<Self::Dec, Self::Error> {
        self.check_input_size(&data)?;
        Ok(JpegDecoder {
            config: self.config,
            stop: self.stop,
            limits: self.limits,
            crop_hint: self.crop_hint,
            orientation: self.orientation,
            policy: self.policy,
            data,
            preferred: preferred.to_vec(),
        })
    }

    fn push_decoder(
        self,
        data: Cow<'a, [u8]>,
        sink: &mut dyn zc::decode::DecodeRowSink,
        preferred: &[PixelDescriptor],
    ) -> Result<OutputInfo, Self::Error> {
        #[cfg(feature = "decoder")]
        {
            push_decoder_native(self, data, sink, preferred)
        }
        #[cfg(not(feature = "decoder"))]
        {
            let _ = (data, sink, preferred);
            Err(Error::unsupported_feature(
                "decoder feature required for push_decoder",
            ))
        }
    }

    fn streaming_decoder(
        self,
        data: Cow<'a, [u8]>,
        preferred: &[PixelDescriptor],
    ) -> Result<Self::StreamDec, Self::Error> {
        #[cfg(feature = "decoder")]
        {
            // ScanlineReader borrows data with lifetime 'a, so we need &'a [u8].
            // Cow::Borrowed carries &'a [u8]; Cow::Owned can't provide 'a.
            let data: &'a [u8] = match data {
                Cow::Borrowed(slice) => slice,
                Cow::Owned(_) => {
                    return Err(Error::unsupported_feature(
                        "streaming decode requires borrowed data (use Cow::Borrowed)",
                    ));
                }
            };
            self.check_input_size(data)?;
            let cfg = build_decode_config(
                &self.config.inner,
                &self.limits,
                self.crop_hint,
                self.orientation,
                self.policy.as_ref(),
            );
            let header = self.config.inner.read_info(data)?;
            let info = to_image_info(&header);
            let reader = cfg.scanline_reader(data)?;

            let descriptor = select_decode_descriptor(preferred, header.num_components);
            let mcu_height = reader.luma_rows_per_mcu();

            Ok(JpegStreamingDecoder {
                reader,
                info,
                descriptor,
                row_buf: Vec::new(),
                current_row: 0,
                mcu_height: mcu_height as u32,
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

    fn full_frame_decoder(
        self,
        _data: Cow<'a, [u8]>,
        _preferred: &[PixelDescriptor],
    ) -> Result<Self::FullFrameDec, Self::Error> {
        Err(UnsupportedOperation::AnimationDecode.into())
    }
}

impl JpegDecodeJob<'_> {
    /// Check input data size against limits.
    fn check_input_size(&self, data: &[u8]) -> Result<(), Error> {
        self.limits
            .check_input_size(data.len() as u64)
            .map_err(|_| {
                Error::allocation_failed(data.len(), "input exceeds max_input_bytes limit")
            })?;
        Ok(())
    }
}

/// Native streaming push_decoder using ScanlineReader.
///
/// Decodes MCU rows on the fly and pushes them into the sink, avoiding the
/// full-image allocation that `push_decoder_via_full_decode` requires.
/// Peak memory is reduced from full image size to one MCU-row strip
/// (typically 8 or 16 rows × width × bytes-per-pixel).
#[cfg(feature = "decoder")]
fn push_decoder_native<'a>(
    job: JpegDecodeJob<'a>,
    data: Cow<'a, [u8]>,
    sink: &mut dyn zc::decode::DecodeRowSink,
    preferred: &[PixelDescriptor],
) -> Result<OutputInfo, Error> {
    use imgref::ImgRefMut;
    use zenpixels::{ChannelLayout, ChannelType};

    let wrap = |e: zc::decode::SinkError| Error::io_error(e.to_string());

    // ScanlineReader borrows data with lifetime 'a.
    let data_ref: &'a [u8] = match data {
        Cow::Borrowed(slice) => slice,
        Cow::Owned(_) => {
            return Err(Error::unsupported_feature(
                "push_decoder requires borrowed data (use Cow::Borrowed)",
            ));
        }
    };
    job.check_input_size(data_ref)?;

    // Build decode config with limits, crop, orientation, policy
    let cfg = build_decode_config(
        &job.config.inner,
        &job.limits,
        job.crop_hint,
        job.orientation,
        job.policy.as_ref(),
    );

    // Probe header for component count (needed for descriptor selection)
    let header = job.config.inner.read_info(data_ref)?;

    // Create the streaming scanline reader
    let mut reader = cfg.scanline_reader(data_ref)?;

    let width = reader.width() as usize;
    let height = reader.height() as usize;
    let mut descriptor = select_decode_descriptor(preferred, header.num_components);
    let mcu_height = reader.luma_rows_per_mcu();

    let ch_type = descriptor.channel_type();
    let ch_layout = descriptor.layout();

    // read_rows_rgba_f32 always outputs 4 channels, so if caller requested
    // RGBF32 we must upgrade to RGBAF32 to match the actual output layout.
    if ch_type == ChannelType::F32 && ch_layout == ChannelLayout::Rgb {
        descriptor = PixelDescriptor::RGBAF32_LINEAR;
    }

    let bpp = descriptor.bytes_per_pixel();
    let row_bytes = width * bpp;

    // Tell the sink what's coming
    sink.begin(width as u32, height as u32, descriptor)
        .map_err(wrap)?;

    // Allocate a temp buffer for one MCU-row strip
    let strip_bytes = row_bytes * mcu_height;
    let mut strip_buf: Vec<u8> = Vec::new();
    strip_buf
        .try_reserve(strip_bytes)
        .map_err(|_| Error::allocation_failed(strip_bytes, "push_decoder strip buffer"))?;
    strip_buf.resize(strip_bytes, 0);

    let mut y = 0u32;

    while !reader.is_finished() {
        // Decode the next batch of rows into our strip buffer
        let remaining = height - y as usize;
        let batch_max = remaining.min(mcu_height);

        let count = match (ch_type, ch_layout) {
            (ChannelType::U8, ChannelLayout::Gray) => {
                let out = ImgRefMut::new(
                    &mut strip_buf[..row_bytes * batch_max],
                    row_bytes,
                    batch_max,
                );
                reader.read_rows_gray8(out)?
            }
            (ChannelType::U8, ChannelLayout::Rgb) => {
                let out = ImgRefMut::new(
                    &mut strip_buf[..row_bytes * batch_max],
                    row_bytes,
                    batch_max,
                );
                reader.read_rows_rgb8(out)?
            }
            (ChannelType::U8, ChannelLayout::Rgba) => {
                let out = ImgRefMut::new(
                    &mut strip_buf[..row_bytes * batch_max],
                    row_bytes,
                    batch_max,
                );
                if descriptor.alpha() == Some(zenpixels::AlphaMode::Undefined) {
                    reader.read_rows_rgbx8(out)?
                } else {
                    reader.read_rows_rgba8(out)?
                }
            }
            (ChannelType::U8, ChannelLayout::Bgra) => {
                let out = ImgRefMut::new(
                    &mut strip_buf[..row_bytes * batch_max],
                    row_bytes,
                    batch_max,
                );
                if descriptor.alpha() == Some(zenpixels::AlphaMode::Undefined) {
                    reader.read_rows_bgrx8(out)?
                } else {
                    reader.read_rows_bgra8(out)?
                }
            }
            (ChannelType::F32, ChannelLayout::Gray) => {
                let float_slice: &mut [f32] =
                    bytemuck::cast_slice_mut(&mut strip_buf[..row_bytes * batch_max]);
                let f_out = ImgRefMut::new(float_slice, width, batch_max);
                reader.read_rows_gray_f32(f_out)?
            }
            (ChannelType::F32, ChannelLayout::Rgb | ChannelLayout::Rgba) => {
                // read_rows_rgba_f32 always writes 4 f32 channels; descriptor
                // was already upgraded to RGBAF32 above, so row_bytes matches.
                let float_slice: &mut [f32] =
                    bytemuck::cast_slice_mut(&mut strip_buf[..row_bytes * batch_max]);
                let f_out = ImgRefMut::new(float_slice, width * 4, batch_max);
                reader.read_rows_rgba_f32(f_out)?
            }
            _ => {
                return Err(Error::unsupported_feature(
                    "unsupported pixel format for push_decoder",
                ));
            }
        };

        if count == 0 {
            break;
        }

        // Get a buffer from the sink for these rows
        let mut dst = sink
            .provide_next_buffer(y, count as u32, width as u32, descriptor)
            .map_err(wrap)?;

        // Copy decoded rows into the sink's buffer
        for row in 0..count as u32 {
            let src_start = row as usize * row_bytes;
            let src_row = &strip_buf[src_start..src_start + row_bytes];
            dst.row_mut(row).copy_from_slice(src_row);
        }
        drop(dst);

        y += count as u32;
    }

    sink.finish().map_err(wrap)?;

    Ok(OutputInfo::full_decode(
        width as u32,
        height as u32,
        descriptor,
    ))
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
    policy: Option<&zc::decode::DecodePolicy>,
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

    // Map threading policy
    match limits.threading {
        zc::ThreadingPolicy::SingleThread => {
            cfg = cfg.num_threads(1);
        }
        zc::ThreadingPolicy::LimitOrSingle { max_threads } => {
            cfg = cfg.num_threads(max_threads as usize);
        }
        zc::ThreadingPolicy::LimitOrAny {
            preferred_max_threads,
        } => {
            cfg = cfg.num_threads(preferred_max_threads as usize);
        }
        _ => {} // Balanced, Unlimited — use default (auto)
    }

    // Map decode policy to strictness and metadata preservation
    if let Some(pol) = policy {
        if let Some(strict) = pol.strict {
            if strict {
                cfg = cfg.strict();
            }
        }
        if let Some(false) = pol.allow_truncated {
            cfg = cfg.strict();
        }
        // Map metadata policy to PreserveConfig
        let mut preserve = crate::decode::PreserveConfig::all();
        if let Some(false) = pol.allow_icc {
            preserve = preserve.icc(crate::decode::IccPreserve::None);
        }
        if let Some(false) = pol.allow_exif {
            preserve = preserve.exif(false);
        }
        if let Some(false) = pol.allow_xmp {
            preserve = preserve.xmp(false);
        }
        cfg = cfg.preserve(preserve);
    }

    cfg
}

/// Select the appropriate pixel descriptor for decode output.
#[cfg(feature = "decoder")]
fn select_decode_descriptor(preferred: &[PixelDescriptor], num_components: u8) -> PixelDescriptor {
    use zenpixels::{ChannelLayout, ChannelType};

    let is_gray = num_components == 1;

    for &desc in preferred {
        let ch = desc.channel_type();
        let layout = desc.layout();

        match (is_gray, ch, layout) {
            (true, ChannelType::U8, ChannelLayout::Gray) => return PixelDescriptor::GRAY8_SRGB,
            (true, ChannelType::F32, ChannelLayout::Gray) => {
                return PixelDescriptor::GRAYF32_LINEAR;
            }
            (false, ChannelType::U8, ChannelLayout::Rgb) => return PixelDescriptor::RGB8_SRGB,
            (false, ChannelType::U8, ChannelLayout::Rgba) => {
                // Check if it's RGBX or RGBA
                if desc.alpha() == Some(zenpixels::AlphaMode::Undefined) {
                    return PixelDescriptor::RGBX8_SRGB;
                }
                return PixelDescriptor::RGBA8_SRGB;
            }
            (false, ChannelType::U8, ChannelLayout::Bgra) => {
                if desc.alpha() == Some(zenpixels::AlphaMode::Undefined) {
                    return PixelDescriptor::BGRX8_SRGB;
                }
                return PixelDescriptor::BGRA8_SRGB;
            }
            (false, ChannelType::F32, ChannelLayout::Rgb) => return PixelDescriptor::RGBF32_LINEAR,
            (false, ChannelType::F32, ChannelLayout::Rgba) => {
                return PixelDescriptor::RGBAF32_LINEAR;
            }
            _ => {}
        }
    }

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
    policy: Option<zc::decode::DecodePolicy>,
    data: Cow<'a, [u8]>,
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

            let wants_f32 = preferred
                .iter()
                .any(|d| d.channel_type() == ChannelType::F32);

            let limits = self.limits;
            let mut cfg = build_decode_config(
                &self.config.inner,
                &limits,
                self.crop_hint,
                self.orientation,
                self.policy.as_ref(),
            );
            cfg = cfg.preserve_all();

            if wants_f32 {
                cfg = cfg.output_target(OutputTarget::LinearF32);
            }

            // Check max_width/max_height before full decode
            if limits.max_width.is_some() || limits.max_height.is_some() {
                let header = cfg.read_info(&data)?;
                limits.check_dimensions(header.dimensions.width, header.dimensions.height)?;
            }

            let stop = self.stop.unwrap_or(&enough::Unstoppable);
            let mut result = cfg.decode(&data, stop)?;

            let w = result.width();
            let h = result.height();
            let format = result.format();

            // Extract metadata
            let mut info = ImageInfo::new(w, h, ImageFormat::Jpeg);
            if let Some(extras) = result.extras() {
                if let Some(icc) = extras.icc_profile() {
                    info = info.with_icc_profile(icc.to_vec());
                }
                if let Some(exif) = extras.exif() {
                    if let Some(orient) = crate::lossless::parse_exif_orientation(exif) {
                        info = info.with_orientation(zc::Orientation::from_exif(orient as u16));
                    }
                    info = info.with_exif(exif.to_vec());
                }
                if let Some(xmp) = extras.xmp() {
                    info = info.with_xmp(xmp.as_bytes().to_vec());
                }
            }

            let jpeg_extras = result.take_extras();

            // Build PixelBuffer with zero-copy where possible
            let buf = if wants_f32 {
                let pixels_f32 = result.into_pixels_f32().unwrap_or_default();
                match format {
                    PixelFormat::Gray => {
                        // Gray<f32> is repr(transparent) over f32 — safe to reinterpret
                        let gray: Vec<Gray<f32>> =
                            pixels_f32.iter().map(|&v| Gray::new(v)).collect();
                        PixelBuffer::from_pixels(gray, w, h)
                            .map_err(|_| Error::internal("pixel count mismatch"))?
                            .with_descriptor(PixelDescriptor::GRAYF32_LINEAR)
                            .into()
                    }
                    _ => {
                        // f32 RGB: 3 floats per pixel, reinterpret as Rgb<f32>
                        let pixel_count = (w as usize) * (h as usize);
                        if pixels_f32.len() == pixel_count * 3 {
                            // Zero-copy: Vec<f32> → Vec<Rgb<f32>> via bytemuck
                            let raw_bytes = bytemuck::cast_vec::<f32, u8>(pixels_f32);
                            PixelBuffer::from_vec(raw_bytes, w, h, PixelDescriptor::RGBF32_LINEAR)
                                .map_err(|_| Error::internal("pixel buffer creation failed"))?
                                .into()
                        } else {
                            // RGBA f32 fallback
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
                }
            } else {
                let pixels_u8 = result.into_pixels_u8().unwrap_or_default();
                match format {
                    PixelFormat::Gray => {
                        // Zero-copy: Vec<u8> is already Gray8 layout
                        PixelBuffer::from_vec(pixels_u8, w, h, PixelDescriptor::GRAY8_SRGB)
                            .map_err(|_| Error::internal("pixel buffer creation failed"))?
                            .into()
                    }
                    PixelFormat::Rgb => {
                        // Zero-copy: Vec<u8> is already packed RGB8 layout
                        PixelBuffer::from_vec(pixels_u8, w, h, PixelDescriptor::RGB8_SRGB)
                            .map_err(|_| Error::internal("pixel buffer creation failed"))?
                            .into()
                    }
                    _ => {
                        // Other formats (Rgba, Bgra, etc.) — pass raw bytes
                        let desc = match format {
                            PixelFormat::Rgba => PixelDescriptor::RGBA8_SRGB,
                            PixelFormat::Bgra => PixelDescriptor::BGRA8_SRGB,
                            _ => PixelDescriptor::RGB8_SRGB,
                        };
                        PixelBuffer::from_vec(pixels_u8, w, h, desc)
                            .map_err(|_| Error::internal("pixel buffer creation failed"))?
                            .into()
                    }
                }
            };

            let mut output = DecodeOutput::new(buf, info);
            if let Some(extras) = jpeg_extras {
                output = output.with_extras(extras);
            }

            // Check output size limits
            let output_bytes = output.pixels().rows() as u64
                * output.pixels().width() as u64
                * output.pixels().descriptor().bytes_per_pixel() as u64;
            self.limits.check_output_size(output_bytes).map_err(|_| {
                Error::allocation_failed(
                    output_bytes as usize,
                    "decoded output exceeds max_output_bytes limit",
                )
            })?;

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
/// Each batch contains one MCU-row worth of decoded pixels (8 or 16 rows).
pub struct JpegStreamingDecoder<'a> {
    #[cfg(feature = "decoder")]
    reader: crate::decode::ScanlineReader<'a>,
    info: ImageInfo,
    descriptor: PixelDescriptor,
    /// Reusable row buffer for decoded pixel data (sized for MCU-row batches).
    row_buf: Vec<u8>,
    current_row: u32,
    /// MCU height in pixels (8 or 16 depending on subsampling).
    mcu_height: u32,
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
            // Allocate for MCU-row batch instead of single row
            let batch_rows = self.mcu_height as usize;
            let batch_bytes = row_bytes * batch_rows;
            self.row_buf.resize(batch_bytes, 0);

            let ch_type = self.descriptor.channel_type();
            let ch_layout = self.descriptor.layout();

            let count = match (ch_type, ch_layout) {
                (ChannelType::U8, ChannelLayout::Gray) => {
                    let out =
                        ImgRefMut::new(&mut self.row_buf[..batch_bytes], row_bytes, batch_rows);
                    self.reader.read_rows_gray8(out)?
                }
                (ChannelType::U8, ChannelLayout::Rgb) => {
                    let out =
                        ImgRefMut::new(&mut self.row_buf[..batch_bytes], row_bytes, batch_rows);
                    self.reader.read_rows_rgb8(out)?
                }
                (ChannelType::U8, ChannelLayout::Rgba) => {
                    if self.descriptor.alpha() == Some(zenpixels::AlphaMode::Undefined) {
                        let out =
                            ImgRefMut::new(&mut self.row_buf[..batch_bytes], row_bytes, batch_rows);
                        self.reader.read_rows_rgbx8(out)?
                    } else {
                        let out =
                            ImgRefMut::new(&mut self.row_buf[..batch_bytes], row_bytes, batch_rows);
                        self.reader.read_rows_rgba8(out)?
                    }
                }
                (ChannelType::U8, ChannelLayout::Bgra) => {
                    if self.descriptor.alpha() == Some(zenpixels::AlphaMode::Undefined) {
                        let out =
                            ImgRefMut::new(&mut self.row_buf[..batch_bytes], row_bytes, batch_rows);
                        self.reader.read_rows_bgrx8(out)?
                    } else {
                        let out =
                            ImgRefMut::new(&mut self.row_buf[..batch_bytes], row_bytes, batch_rows);
                        self.reader.read_rows_bgra8(out)?
                    }
                }
                (ChannelType::F32, ChannelLayout::Gray) => {
                    let float_count = width * batch_rows;
                    let float_bytes = float_count * 4;
                    self.row_buf.resize(float_bytes, 0);
                    let float_slice: &mut [f32] = bytemuck::cast_slice_mut(&mut self.row_buf);
                    let f_out = ImgRefMut::new(float_slice, width, batch_rows);
                    self.reader.read_rows_gray_f32(f_out)?
                }
                (ChannelType::F32, ChannelLayout::Rgb | ChannelLayout::Rgba) => {
                    // read_rows_rgba_f32 always writes 4 channels
                    let channels = 4;
                    let float_count = width * channels * batch_rows;
                    let float_bytes = float_count * 4;
                    self.row_buf.resize(float_bytes, 0);
                    let float_slice: &mut [f32] = bytemuck::cast_slice_mut(&mut self.row_buf);
                    let f_out = ImgRefMut::new(float_slice, width * channels, batch_rows);
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

            let actual_bytes = row_bytes * count;
            let stride = row_bytes;
            let slice = PixelSlice::new(
                &self.row_buf[..actual_bytes],
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
            img_info = img_info.with_orientation(zc::Orientation::from_exif(orient as u16));
        }
        img_info = img_info.with_exif(exif.clone());
    }
    if let Some(ref xmp) = info.xmp {
        img_info = img_info.with_xmp(xmp.as_bytes().to_vec());
    }

    img_info
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::borrow::Cow;
    use imgref::{Img, ImgExt};
    use rgb::{Gray, Rgb, Rgba};
    use zc::encode::{EncodeJob as _, Encoder as _, EncoderConfig as _};

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
    fn encoding_with_policy_strips_metadata() {
        let enc = JpegEncoderConfig::new().with_calibrated_quality(85.0);
        let pixels: Vec<Rgb<u8>> = vec![Rgb { r: 255, g: 0, b: 0 }; 16];
        let img = Img::new(pixels.as_slice(), 4, 4);

        let icc = b"fake icc profile data";
        let meta = MetadataView::default().with_icc(icc.as_slice());
        let policy = zc::encode::EncodePolicy::strict();

        let output = enc
            .job()
            .with_metadata(&meta)
            .with_policy(policy)
            .encoder()
            .unwrap()
            .encode(PixelSlice::from(img.as_ref()).into())
            .unwrap();
        // Should succeed but ICC may be stripped by strict policy
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
        let top = slice.sub_rows(0, 4);
        let bottom = slice.sub_rows(4, 4);
        encoder.push_rows(top).unwrap();
        encoder.push_rows(bottom).unwrap();
        let output = encoder.finish().unwrap();
        assert!(!output.data().is_empty());
        assert_eq!(&output.data()[0..2], &[0xFF, 0xD8]);
    }

    #[test]
    fn effort_levels() {
        let enc = JpegEncoderConfig::new()
            .with_generic_quality(85.0)
            .with_generic_effort(0); // Fast
        assert_eq!(enc.generic_effort(), Some(0));

        let enc = enc.with_generic_effort(2); // Max
        assert_eq!(enc.generic_effort(), Some(2));

        // Effort clamped to range
        let enc = enc.with_generic_effort(99);
        assert_eq!(enc.generic_effort(), Some(2));
    }

    #[cfg(feature = "decoder")]
    #[test]
    fn decode_roundtrip() {
        use zc::decode::{Decode as _, DecodeJob as _, DecoderConfig as _};
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

        let dec = JpegDecoderConfig::new();
        let output = dec.decode(encoded.data()).unwrap();
        assert_eq!(output.info().width, 8);
        assert_eq!(output.info().height, 8);
        assert_eq!(output.info().format, ImageFormat::Jpeg);
    }

    #[cfg(feature = "decoder")]
    #[test]
    fn decode_zero_copy_rgb8() {
        use zc::decode::{Decode as _, DecodeJob as _, DecoderConfig as _};
        let enc = JpegEncoderConfig::new().with_calibrated_quality(95.0);
        let pixels: Vec<Rgb<u8>> = vec![
            Rgb {
                r: 200,
                g: 100,
                b: 50
            };
            64
        ];
        let img = Img::new(pixels.as_slice(), 8, 8);
        let encoded = enc.encode(PixelSlice::from(img.as_ref()).into()).unwrap();

        let dec = JpegDecoderConfig::new();
        let output = dec
            .job()
            .decoder(Cow::Borrowed(encoded.data()), &[PixelDescriptor::RGB8_SRGB])
            .unwrap()
            .decode()
            .unwrap();
        // Output should be RGB8 — the native format
        assert_eq!(output.descriptor(), PixelDescriptor::RGB8_SRGB);
        let pixel_data = output.pixels();
        assert_eq!(pixel_data.width(), 8);
        assert_eq!(pixel_data.rows(), 8);
        // 8*8*3 = 192 bytes
        assert!(pixel_data.as_contiguous_bytes().is_some());
    }

    #[cfg(feature = "decoder")]
    #[test]
    fn probe_info() {
        use zc::decode::{DecodeJob as _, DecoderConfig as _};
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

        let dec = JpegDecoderConfig::new();
        let mut stream = dec
            .job()
            .streaming_decoder(Cow::Borrowed(encoded.data()), &[PixelDescriptor::RGB8_SRGB])
            .unwrap();

        assert_eq!(stream.info().width, 16);
        assert_eq!(stream.info().height, 16);

        let mut total_rows = 0u32;
        while let Some((y, batch)) = stream.next_batch().unwrap() {
            assert_eq!(y, total_rows);
            assert_eq!(batch.width(), 16);
            // Each batch should be MCU-row sized (multiple rows)
            assert!(batch.rows() >= 1);
            total_rows += batch.rows();
        }
        assert_eq!(total_rows, 16);
    }

    #[cfg(feature = "decoder")]
    #[test]
    fn streaming_decode_batches_mcu_rows() {
        use zc::decode::{DecodeJob as _, DecoderConfig as _, StreamingDecode as _};

        // Create a larger image to see MCU batching
        let enc = JpegEncoderConfig::new().with_calibrated_quality(85.0);
        let pixels: Vec<Rgb<u8>> = vec![
            Rgb {
                r: 128,
                g: 64,
                b: 32
            };
            64 * 64
        ];
        let img = Img::new(pixels.as_slice(), 64, 64);
        let encoded = enc.encode(PixelSlice::from(img.as_ref()).into()).unwrap();

        let dec = JpegDecoderConfig::new();
        let mut stream = dec
            .job()
            .streaming_decoder(Cow::Borrowed(encoded.data()), &[PixelDescriptor::RGB8_SRGB])
            .unwrap();

        let mut batch_count = 0;
        let mut total_rows = 0u32;
        while let Some((_y, batch)) = stream.next_batch().unwrap() {
            batch_count += 1;
            total_rows += batch.rows();
        }
        assert_eq!(total_rows, 64);
        // With MCU batching, we should have fewer batches than rows
        // (64 rows / 16 rows per MCU = ~4 batches for 4:2:0)
        assert!(
            batch_count < 64,
            "expected MCU-row batching, got {batch_count} batches for 64 rows"
        );
    }

    // ── Encoder trait roundtrip tests ────────────────────────────────

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
        let output = dyn_enc
            .encode(zenpixels::PixelSlice::from(img.as_ref()).into())
            .unwrap();
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
        assert!(caps.native_16bit());
        assert!(caps.native_f32());
        assert!(caps.enforces_max_pixels());
        assert!(caps.enforces_max_memory());
        assert!(caps.quality_range().is_some());
        assert!(caps.effort_range().is_some());
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
        assert!(caps.native_f32());
        assert!(caps.enforces_max_pixels());
        assert!(caps.enforces_max_memory());
        assert!(!caps.animation());
    }

    #[test]
    fn decode_trait_max_width_enforced() {
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
        let limits = ResourceLimits::none().with_max_width(10);
        let result = dec
            .job()
            .with_limits(limits)
            .decoder(Cow::Borrowed(encoded.data()), &[])
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
            .decoder(Cow::Borrowed(encoded.data()), &[])
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
            .decoder(Cow::Borrowed(encoded.data()), &[])
            .unwrap()
            .decode();
        assert!(
            result.is_ok(),
            "generous limits should not reject 32x32 image"
        );
    }

    #[test]
    fn full_frame_encoder_returns_unsupported() {
        let config = JpegEncoderConfig::new();
        let result = config.job().full_frame_encoder();
        assert!(result.is_err());
    }

    #[cfg(feature = "decoder")]
    #[test]
    fn full_frame_decoder_returns_unsupported() {
        use zc::decode::{DecodeJob as _, DecoderConfig as _};

        let dec = JpegDecoderConfig::new();
        let result = dec.job().full_frame_decoder(Cow::Borrowed(&[]), &[]);
        assert!(result.is_err());
    }
}
