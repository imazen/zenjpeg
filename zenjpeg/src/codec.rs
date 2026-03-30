//! zencodec trait implementations for zenjpeg.
//!
//! Provides [`JpegEncoderConfig`] and [`JpegDecoderConfig`] types that implement
//! the encode/decode trait hierarchy from zencodec, wrapping the native
//! zenjpeg API.
//!
//! The native API remains untouched — this is a thin adapter layer.
//!
//! # Trait mapping
//!
//! | zencodec | zenjpeg adapter |
//! |----------------|-----------------|
//! | `EncoderConfig` | [`JpegEncoderConfig`] |
//! | `EncodeJob<'a>` | [`JpegEncodeJob`] |
//! | `Encoder` | [`JpegEncoder`] |
//! | `AnimationFrameEncoder` | `()` (JPEG has no animation) |
//! | `DecoderConfig` | [`JpegDecoderConfig`] |
//! | `DecodeJob<'a>` | [`JpegDecodeJob`] |
//! | `Decode` | [`JpegDecoder`] |
//! | `StreamingDecode` | [`JpegStreamingDecoder`] |
//! | `AnimationFrameDecode` | `Unsupported<Error>` (JPEG has no animation) |

extern crate alloc;
use alloc::borrow::Cow;
use alloc::vec::Vec;

use rgb::{Gray, Rgb};
use zencodec::decode::{DecodeCapabilities, DecodeOutput, OutputInfo};
use zencodec::encode::{EncodeCapabilities, EncodeOutput};
use zencodec::{
    ImageFormat, ImageInfo, Metadata, ResourceLimits, Unsupported, UnsupportedOperation,
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
// Encode side: EncoderConfig → EncodeJob → Encoder
// ============================================================================

/// JPEG encode capabilities.
static JPEG_ENCODE_CAPS: EncodeCapabilities = EncodeCapabilities::new()
    .with_icc(true)
    .with_exif(true)
    .with_xmp(true)
    .with_stop(true)
    .with_lossy(true)
    .with_push_rows(true)
    .with_encode_from(true)
    .with_native_gray(true)
    .with_native_16bit(true)
    .with_native_f32(true)
    .with_enforces_max_pixels(true)
    .with_enforces_max_memory(true)
    .with_quality_range(0.0, 100.0)
    .with_effort_range(0, 2)
    .with_threads_supported_range(1, if cfg!(feature = "parallel") { 32 } else { 1 });

/// JPEG encoder configuration implementing [`zencodec::encode::EncoderConfig`].
///
/// Wraps [`EncoderConfig`] with the zencodec trait interface.
/// Defaults to YCbCr 4:2:0 at quality 85.
#[derive(Clone, Debug)]
pub struct JpegEncoderConfig {
    inner: EncoderConfig,
    quality: f32,
    effort: i32,
    /// Original generic quality value passed to `with_generic_quality()`.
    /// Stored separately because the calibration mapping is not invertible.
    generic_quality_input: Option<f32>,
}

impl JpegEncoderConfig {
    /// Create a default YCbCr 4:2:0 config at quality 85.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter),
            quality: 85.0,
            effort: 1,
            generic_quality_input: None,
        }
    }

    /// Create a YCbCr config with quality and subsampling.
    #[must_use]
    pub fn ycbcr(quality: f32, subsampling: ChromaSubsampling) -> Self {
        Self {
            inner: EncoderConfig::ycbcr(quality, subsampling),
            quality,
            effort: 1,
            generic_quality_input: None,
        }
    }

    /// Create a grayscale config with quality.
    #[must_use]
    pub fn grayscale(quality: f32) -> Self {
        Self {
            inner: EncoderConfig::grayscale(quality),
            quality,
            effort: 1,
            generic_quality_input: None,
        }
    }

    /// Create from a named optimization preset.
    ///
    /// Available presets: `"mozjpeg_baseline"`, `"mozjpeg_progressive"`,
    /// `"mozjpeg_max"`, `"jpegli_baseline"`, `"jpegli_progressive"`,
    /// `"hybrid_baseline"`, `"hybrid_progressive"`, `"hybrid_max"`.
    ///
    /// Returns `None` for unrecognized preset names.
    ///
    /// Requires the `trellis` feature (uses [`ExpertConfig`](crate::encode::search::ExpertConfig)).
    #[cfg(feature = "trellis")]
    #[must_use]
    pub fn from_preset(preset_name: &str, quality: f32) -> Option<Self> {
        use crate::encode::encoder_types::OptimizationPreset;
        use crate::encode::search::ExpertConfig;

        let preset = match preset_name {
            "mozjpeg_baseline" => OptimizationPreset::MozjpegBaseline,
            "mozjpeg_progressive" => OptimizationPreset::MozjpegProgressive,
            "mozjpeg_max" => OptimizationPreset::MozjpegMaxCompression,
            "jpegli_baseline" => OptimizationPreset::JpegliBaseline,
            "jpegli_progressive" => OptimizationPreset::JpegliProgressive,
            "hybrid_baseline" => OptimizationPreset::HybridBaseline,
            "hybrid_progressive" => OptimizationPreset::HybridProgressive,
            "hybrid_max" => OptimizationPreset::HybridMaxCompression,
            _ => return None,
        };

        let expert = ExpertConfig::from_preset(preset, quality);
        let color_mode = crate::encode::encoder_types::ColorMode::YCbCr {
            subsampling: ChromaSubsampling::Quarter,
        };
        let inner = expert.to_encoder_config(color_mode);

        Some(Self {
            inner,
            quality,
            effort: 1,
            generic_quality_input: None,
        })
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
        use zencodec::encode::{EncodeJob as _, Encoder as _, EncoderConfig as _};
        self.clone().job().encoder()?.encode(pixels)
    }

    /// Apply effort level, returning a modified config.
    fn effective_config(&self) -> EncoderConfig {
        use crate::encode::encoder_types::OptimizationPreset;
        let preset = match self.effort {
            0 => OptimizationPreset::JpegliBaseline,
            #[cfg(feature = "trellis")]
            2 => OptimizationPreset::HybridMaxCompression,
            #[cfg(feature = "trellis")]
            _ => OptimizationPreset::HybridProgressive,
            #[cfg(not(feature = "trellis"))]
            _ => OptimizationPreset::JpegliProgressive,
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

/// Map generic quality (libjpeg-turbo scale) to jpegli native quality.
///
/// Calibrated on CID22-512 corpus (209 images) to produce the same median
/// SSIMULACRA2 as libjpeg-turbo at each quality level.
fn calibrated_jpeg_quality(generic_q: f32) -> f32 {
    // generic_quality → jpegli native quality
    const TABLE: &[(f32, f32)] = &[
        (5.0, 5.0),
        (10.0, 5.0),
        (15.0, 5.9),
        (20.0, 11.8),
        (25.0, 16.3),
        (30.0, 20.2),
        (35.0, 24.3),
        (40.0, 28.8),
        (45.0, 36.5),
        (50.0, 43.8),
        (55.0, 49.7),
        (60.0, 54.7),
        (65.0, 60.5),
        (70.0, 65.8),
        (72.0, 69.1),
        (75.0, 72.6),
        (78.0, 76.0),
        (80.0, 77.6),
        (82.0, 80.3),
        (85.0, 84.1),
        (87.0, 86.0),
        (90.0, 89.6),
        (92.0, 91.5),
        (95.0, 95.1),
        (97.0, 98.0),
        (99.0, 99.0),
    ];
    interp_quality(TABLE, generic_q)
}

/// Piecewise linear interpolation with clamping at table bounds.
fn interp_quality(table: &[(f32, f32)], x: f32) -> f32 {
    if x <= table[0].0 {
        return table[0].1;
    }
    if x >= table[table.len() - 1].0 {
        return table[table.len() - 1].1;
    }
    for i in 1..table.len() {
        if x <= table[i].0 {
            let (x0, y0) = table[i - 1];
            let (x1, y1) = table[i];
            let t = (x - x0) / (x1 - x0);
            return y0 + t * (y1 - y0);
        }
    }
    table[table.len() - 1].1
}

impl zencodec::encode::EncoderConfig for JpegEncoderConfig {
    type Error = Error;
    type Job = JpegEncodeJob;

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
        let clamped = quality.clamp(0.0, 100.0);
        self.generic_quality_input = Some(clamped);
        let q = calibrated_jpeg_quality(clamped);
        self.quality = q;
        self.inner = self.inner.quality(Quality::ApproxJpegli(q));
        self
    }

    fn generic_quality(&self) -> Option<f32> {
        Some(self.generic_quality_input.unwrap_or(self.quality))
    }

    fn with_generic_effort(mut self, effort: i32) -> Self {
        self.effort = effort.clamp(0, 2);
        self
    }

    fn generic_effort(&self) -> Option<i32> {
        Some(self.effort)
    }

    fn job(self) -> Self::Job {
        JpegEncodeJob {
            config: self,
            stop: None,
            metadata: None,
            limits: ResourceLimits::none(),
            policy: None,
            image_size: None,
        }
    }
}

// ── Encode job ──────────────────────────────────────────────────────────────

/// Per-operation JPEG encode job.
///
/// Created by [`JpegEncoderConfig::job()`]. Borrows temporary data (stop token,
/// metadata) and is consumed by creating a [`JpegEncoder`].
pub struct JpegEncodeJob {
    config: JpegEncoderConfig,
    stop: Option<zencodec::StopToken>,
    metadata: Option<Metadata>,
    limits: ResourceLimits,
    policy: Option<zencodec::encode::EncodePolicy>,
    /// Image dimensions, set via `with_canvas_size`. When known, enables true
    /// streaming in `push_rows` → `finish` (no full-image accumulation).
    image_size: Option<(u32, u32)>,
}

impl zencodec::encode::EncodeJob for JpegEncodeJob {
    type Error = Error;
    type Enc = JpegEncoder;
    type AnimationFrameEnc = ();

    fn with_stop(mut self, stop: zencodec::StopToken) -> Self {
        self.stop = Some(stop);
        self
    }

    fn with_metadata(mut self, meta: Metadata) -> Self {
        self.metadata = Some(meta);
        self
    }

    fn with_limits(mut self, limits: ResourceLimits) -> Self {
        self.limits = limits;
        self
    }

    fn with_policy(mut self, policy: zencodec::encode::EncodePolicy) -> Self {
        self.policy = Some(policy);
        self
    }

    fn with_canvas_size(mut self, width: u32, height: u32) -> Self {
        self.image_size = Some((width, height));
        self
    }

    fn encoder(self) -> Result<JpegEncoder, Self::Error> {
        #[allow(unused_mut)]
        let mut cfg = self.config.effective_config();

        // Map threading policy to parallel encoding config
        #[cfg(feature = "parallel")]
        {
            use zencodec::ThreadingPolicy;
            match self.limits.threading {
                ThreadingPolicy::SingleThread => {
                    // Explicitly do not enable parallel — leave cfg.parallel as None
                }
                ThreadingPolicy::LimitOrSingle { max_threads } => {
                    if max_threads > 1 {
                        cfg = cfg.parallel(crate::encode::ParallelEncoding::Auto);
                    }
                }
                ThreadingPolicy::LimitOrAny { .. }
                | ThreadingPolicy::Balanced
                | ThreadingPolicy::Unlimited => {
                    cfg = cfg.parallel(crate::encode::ParallelEncoding::Auto);
                }
                _ => {}
            }
        }

        Ok(JpegEncoder {
            effective_config: cfg,
            stop: self.stop,
            metadata: self.metadata,
            limits: self.limits,
            policy: self.policy,
            accumulator: None,
            streaming_enc: None,
            image_size: self.image_size,
        })
    }

    fn animation_frame_encoder(self) -> Result<Self::AnimationFrameEnc, Self::Error> {
        Err(UnsupportedOperation::AnimationEncode.into())
    }
}

// ── Encoder ─────────────────────────────────────────────────────────────────

/// Single-image JPEG encoder implementing [`zencodec::encode::Encoder`].
///
/// Supports one-shot `encode()`, streaming `push_rows()` + `finish()`,
/// and the `encode_srgba8()` convenience method.
pub struct JpegEncoder {
    effective_config: EncoderConfig,
    stop: Option<zencodec::StopToken>,
    metadata: Option<Metadata>,
    limits: ResourceLimits,
    policy: Option<zencodec::encode::EncodePolicy>,
    /// Accumulated rows for push_rows path (fallback when dimensions unknown).
    accumulator: Option<RowAccumulator>,
    /// Native streaming encoder — used when dimensions are known via
    /// `with_canvas_size`. Streams rows directly without accumulation.
    /// Forces baseline mode (progressive requires buffering all coefficients).
    streaming_enc: Option<crate::encode::byte_encoders::BytesEncoder>,
    /// Image dimensions from `with_canvas_size`.
    image_size: Option<(u32, u32)>,
}

/// Internal buffer for accumulating pushed rows.
struct RowAccumulator {
    data: Vec<u8>,
    width: u32,
    total_rows: u32,
    layout: PixelLayout,
    descriptor: PixelDescriptor,
}

impl JpegEncoder {
    /// Get a reference to the stop token, defaulting to Unstoppable.
    fn stop_ref(&self) -> &dyn enough::Stop {
        match self.stop {
            Some(ref s) => s,
            None => &enough::Unstoppable,
        }
    }

    /// Build an EncodeRequest from current config + metadata, applying policy.
    fn build_request(&self) -> crate::encode::request::EncodeRequest<'_> {
        self.build_request_from(&self.effective_config)
    }

    /// Build an EncodeRequest from a specific config + metadata, applying policy.
    fn build_request_from<'b>(
        &'b self,
        config: &'b EncoderConfig,
    ) -> crate::encode::request::EncodeRequest<'b> {
        let mut req = config.request();
        if let Some(ref meta) = self.metadata {
            let policy = self.policy.unwrap_or_default();
            if policy.resolve_icc(true)
                && let Some(ref icc) = meta.icc_profile
            {
                req = req.icc_profile(icc);
            }
            if policy.resolve_exif(true)
                && let Some(ref exif) = meta.exif
            {
                req = req.exif(Exif::raw(exif.to_vec()));
            }
            if policy.resolve_xmp(true)
                && let Some(ref xmp) = meta.xmp
            {
                req = req.xmp(xmp);
            }
        }
        if let Some(ref stop) = self.stop {
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
        let stop = self.stop_ref();
        let mut enc = req.encode_from_bytes(acc.width, acc.total_rows, acc.layout)?;
        // Stream through native encoder — it processes MCU rows as they arrive
        enc.push_packed(&acc.data, stop)?;
        let output = enc.finish()?;
        self.check_output_size(&output)?;
        Ok(EncodeOutput::new(output, ImageFormat::Jpeg))
    }
}

impl zencodec::encode::Encoder for JpegEncoder {
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
        let stop = self.stop_ref();
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

        // Streaming path: dimensions known, push directly to native encoder.
        if let Some((img_w, img_h)) = self.image_size {
            if self.streaming_enc.is_none() {
                self.check_limits(img_w, img_h, layout)?;
                // Force baseline + fixed Huffman for true streaming-through.
                // Progressive buffers all coefficients; optimized Huffman
                // buffers all blocks for two-pass frequency counting.
                // Fixed Huffman writes blocks immediately as they arrive.
                let streaming_config = self
                    .effective_config
                    .clone()
                    .progressive(false)
                    .optimize_huffman(false);
                let req = self.build_request_from(&streaming_config);
                let enc = req.encode_from_bytes(img_w, img_h, layout)?;
                self.streaming_enc = Some(enc);
            }
            let stop: &dyn enough::Stop = match self.stop {
                Some(ref s) => s,
                None => &enough::Unstoppable,
            };
            let enc = self.streaming_enc.as_mut().unwrap();
            // Use as_strided_bytes for zero-copy; BytesEncoder::push handles stride.
            enc.push(
                rows.as_strided_bytes(),
                rows.rows() as usize,
                rows.stride(),
                stop,
            )?;
            return Ok(());
        }

        // Fallback: accumulate rows (dimensions unknown).
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
        // Streaming path: finish the native encoder directly.
        if let Some(enc) = self.streaming_enc.take() {
            let output = enc.finish()?;
            self.check_output_size(&output)?;
            return Ok(EncodeOutput::new(output, ImageFormat::Jpeg));
        }

        // Fallback: accumulation path.
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
        use zenpixels::PixelSliceMut;

        let (img_w, img_h) = self.image_size.ok_or_else(|| {
            Error::unsupported_feature(
                "encode_from requires with_canvas_size (dimensions must be known upfront)",
            )
        })?;

        // Determine pixel layout from the first source callback.
        // We use RGBA8/sRGB as the default descriptor for the pull buffer.
        // The source fills the buffer; we discover the actual format from
        // what it produces. For now, use the descriptor from the config's
        // supported list — JPEG always wants RGB8 or RGBA8.
        let desc = PixelDescriptor::RGB8_SRGB;
        let layout = descriptor_to_layout(desc)?;
        self.check_limits(img_w, img_h, layout)?;

        // Force baseline for streaming (same as push_rows streaming path).
        let streaming_config = self
            .effective_config
            .clone()
            .progressive(false)
            .optimize_huffman(false);
        let req = self.build_request_from(&streaming_config);
        let mut enc = req.encode_from_bytes(img_w, img_h, layout)?;
        let stop = self.stop_ref();

        // Allocate strip buffer: preferred_strip_height rows.
        let strip_h = 16u32.min(img_h); // MCU-aligned strip
        let bpp = desc.bytes_per_pixel();
        let stride = img_w as usize * bpp;
        let buf_size = strip_h as usize * stride;
        let mut buf = alloc::vec![0u8; buf_size];

        let mut y = 0u32;
        while y < img_h {
            let rows_wanted = strip_h.min(img_h - y);
            let slice_size = rows_wanted as usize * stride;

            let mut pixel_buf =
                PixelSliceMut::new(&mut buf[..slice_size], img_w, rows_wanted, stride, desc)
                    .map_err(|e| {
                        let _ = e;
                        Error::unsupported_feature("encode_from buffer")
                    })?;

            let rows_provided = source(y, pixel_buf.sub_rows_mut(0, rows_wanted));
            if rows_provided == 0 {
                break;
            }
            let actual_rows = (rows_provided as u32).min(rows_wanted);

            enc.push(
                &buf[..actual_rows as usize * stride],
                actual_rows as usize,
                stride,
                stop,
            )?;
            y += actual_rows;
        }

        let output = enc.finish()?;
        self.check_output_size(&output)?;
        Ok(EncodeOutput::new(output, ImageFormat::Jpeg))
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
    .with_stop(true)
    .with_cheap_probe(true)
    .with_streaming(true)
    .with_native_gray(true)
    .with_native_f32(true)
    .with_enforces_max_pixels(true)
    .with_enforces_max_memory(true)
    .with_enforces_max_input_bytes(true)
    .with_threads_supported_range(1, if cfg!(feature = "parallel") { 32 } else { 1 });

/// JPEG decoder configuration implementing [`zencodec::decode::DecoderConfig`].
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

    /// Create a decode job by consuming this config.
    ///
    /// This is equivalent to [`DecoderConfig::job(self)`] but available
    /// without importing the trait.
    #[must_use]
    pub fn job_static(self) -> JpegDecodeJob {
        JpegDecodeJob {
            config: self,
            stop: None,
            limits: ResourceLimits::none(),
            crop_hint: None,
            orientation: zencodec::OrientationHint::default(),
            policy: None,
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

    /// Enable post-decode deblocking to reduce JPEG block artifacts.
    ///
    /// Delegates to [`DecodeConfig::deblock()`](crate::decode::DecodeConfig::deblock).
    /// See [`DeblockMode`](crate::decode::DeblockMode) for available modes.
    #[cfg(feature = "decoder")]
    #[must_use]
    pub fn deblock(mut self, mode: crate::decode::DeblockMode) -> Self {
        self.inner = self.inner.deblock(mode);
        self
    }

    /// Convenience: probe image header with this config.
    pub fn probe_header(&self, data: &[u8]) -> Result<ImageInfo, Error> {
        use zencodec::decode::{DecodeJob as _, DecoderConfig as _};
        self.clone().job().probe(data)
    }

    /// Convenience: probe full image metadata (may be expensive).
    pub fn probe_full_metadata(&self, data: &[u8]) -> Result<ImageInfo, Error> {
        use zencodec::decode::{DecodeJob as _, DecoderConfig as _};
        self.clone().job().probe_full(data)
    }

    /// Convenience: decode image with this config.
    pub fn decode(&self, data: &[u8]) -> Result<DecodeOutput, Error> {
        use zencodec::decode::{Decode as _, DecodeJob as _, DecoderConfig as _};
        self.clone()
            .job()
            .decoder(Cow::Borrowed(data), &[])?
            .decode()
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

impl zencodec::decode::DecoderConfig for JpegDecoderConfig {
    type Error = Error;
    type Job<'a> = JpegDecodeJob;

    fn formats() -> &'static [ImageFormat] {
        &[ImageFormat::Jpeg]
    }

    fn supported_descriptors() -> &'static [PixelDescriptor] {
        DECODE_DESCRIPTORS
    }

    fn capabilities() -> &'static DecodeCapabilities {
        &JPEG_DECODE_CAPS
    }

    fn job<'a>(self) -> Self::Job<'a> {
        JpegDecodeJob {
            config: self,
            stop: None,
            limits: ResourceLimits::none(),
            crop_hint: None,
            orientation: zencodec::OrientationHint::default(),
            policy: None,
        }
    }
}

// ── Decode job ──────────────────────────────────────────────────────────────

/// Per-operation JPEG decode job.
///
/// Created by [`JpegDecoderConfig::job()`]. Consumed by creating a
/// [`JpegDecoder`] or [`JpegStreamingDecoder`].
pub struct JpegDecodeJob {
    config: JpegDecoderConfig,
    stop: Option<zencodec::StopToken>,
    limits: ResourceLimits,
    crop_hint: Option<(u32, u32, u32, u32)>,
    orientation: zencodec::OrientationHint,
    policy: Option<zencodec::decode::DecodePolicy>,
}

impl<'a> zencodec::decode::DecodeJob<'a> for JpegDecodeJob {
    type Error = Error;
    type Dec = JpegDecoder<'a>;
    type StreamDec = JpegStreamingDecoder<'a>;
    type AnimationFrameDec = Unsupported<Error>;

    fn with_stop(mut self, stop: zencodec::StopToken) -> Self {
        self.stop = Some(stop);
        self
    }

    fn with_limits(mut self, limits: ResourceLimits) -> Self {
        self.limits = limits;
        self
    }

    fn with_policy(mut self, policy: zencodec::decode::DecodePolicy) -> Self {
        self.policy = Some(policy);
        self
    }

    fn with_crop_hint(mut self, x: u32, y: u32, width: u32, height: u32) -> Self {
        self.crop_hint = Some((x, y, width, height));
        self
    }

    fn with_orientation(mut self, hint: zencodec::OrientationHint) -> Self {
        self.orientation = hint;
        self
    }

    fn probe(&self, data: &[u8]) -> Result<ImageInfo, Self::Error> {
        #[cfg(feature = "decoder")]
        {
            // Check input size limits
            self.check_input_size(data)?;
            let info = self.config.inner.read_info(data)?;
            let mut image_info = to_image_info(&info);
            if let Ok(probe) = crate::detect::probe(data) {
                image_info = image_info.with_source_encoding_details(probe);
            }
            Ok(image_info)
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
            let native_format =
                decode_descriptor(&[], &info, self.config.inner.correct_color.as_ref());
            let mut w = info.dimensions.width;
            let mut h = info.dimensions.height;

            let mut out = OutputInfo::full_decode(w, h, native_format);

            let will_orient = will_auto_orient(self.orientation);
            if will_orient
                && let Some(ref exif) = info.exif
                && let Some(orient_val) = crate::lossless::parse_exif_orientation(exif)
            {
                let orient = zencodec::Orientation::from_exif(orient_val).unwrap_or_default();
                if orient.swaps_axes() {
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
        sink: &mut dyn zencodec::decode::DecodeRowSink,
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
            self.check_input_size(&data)?;
            let cfg = build_decode_config(
                &self.config.inner,
                &self.limits,
                self.crop_hint,
                self.orientation,
                self.policy.as_ref(),
            );
            // read_info borrows data temporarily and returns owned JpegInfo.
            let header = self.config.inner.read_info(&data)?;
            self.check_progressive_policy(header.mode)?;
            let mut info = to_image_info(&header);

            // If auto-orient was applied, report Identity orientation
            if will_auto_orient(self.orientation) {
                info = info.with_orientation(zencodec::Orientation::Identity);
            }

            // scanline_reader_cow accepts both Borrowed and Owned data.
            // When Owned, the reader stores the Vec internally.
            let reader = cfg.scanline_reader_cow(data)?;

            let descriptor =
                decode_descriptor(preferred, &header, self.config.inner.correct_color.as_ref());
            let mcu_height = reader.luma_rows_per_mcu();

            Ok(JpegStreamingDecoder {
                reader,
                info,
                descriptor,
                row_buf: aligned_vec::AVec::new(4),
                current_row: 0,
                mcu_height: mcu_height as u32,
                stop: self.stop,
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

    fn animation_frame_decoder(
        self,
        _data: Cow<'a, [u8]>,
        _preferred: &[PixelDescriptor],
    ) -> Result<Self::AnimationFrameDec, Self::Error> {
        Err(UnsupportedOperation::AnimationDecode.into())
    }
}

impl JpegDecodeJob {
    /// Check input data size against limits.
    fn check_input_size(&self, data: &[u8]) -> Result<(), Error> {
        self.limits
            .check_input_size(data.len() as u64)
            .map_err(|_| {
                Error::allocation_failed(data.len(), "input exceeds max_input_bytes limit")
            })?;
        Ok(())
    }

    /// Check whether the decode policy allows progressive JPEGs.
    ///
    /// Returns an error if the image is progressive and the policy forbids it.
    #[cfg(feature = "decoder")]
    fn check_progressive_policy(&self, mode: crate::types::JpegMode) -> Result<(), Error> {
        if let Some(ref policy) = self.policy {
            let is_progressive = matches!(
                mode,
                crate::types::JpegMode::Progressive | crate::types::JpegMode::ArithmeticProgressive
            );
            if is_progressive && !policy.resolve_progressive(true) {
                return Err(Error::unsupported_feature(
                    "progressive JPEG rejected by decode policy",
                ));
            }
        }
        Ok(())
    }
}

/// Native streaming push_decoder using ScanlineReader.
///
/// Decodes MCU rows on the fly and pushes them into the sink, avoiding the
/// full-image allocation that `helpers::copy_decode_to_sink` requires.
/// Peak memory is reduced from full image size to one MCU-row strip
/// (typically 8 or 16 rows × width × bytes-per-pixel).
#[cfg(feature = "decoder")]
fn push_decoder_native<'a>(
    job: JpegDecodeJob,
    data: Cow<'a, [u8]>,
    sink: &mut dyn zencodec::decode::DecodeRowSink,
    preferred: &[PixelDescriptor],
) -> Result<OutputInfo, Error> {
    use imgref::ImgRefMut;
    use zenpixels::{ChannelLayout, ChannelType};

    let wrap = |e: zencodec::decode::SinkError| Error::io_error(e.to_string());

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
    job.check_progressive_policy(header.mode)?;

    // Create the streaming scanline reader
    let mut reader = cfg.scanline_reader(data_ref)?;

    let width = reader.width() as usize;
    let height = reader.height() as usize;
    let mut descriptor =
        decode_descriptor(preferred, &header, job.config.inner.correct_color.as_ref());
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

    // Allocate a temp buffer for one MCU-row strip.
    // Use aligned_vec for ≥4-byte alignment so f32 casts are safe.
    let strip_bytes = row_bytes * mcu_height;
    let mut strip_buf = aligned_vec::AVec::<u8, aligned_vec::ConstAlign<4>>::new(4);
    strip_buf
        .try_reserve(strip_bytes)
        .map_err(|_| Error::allocation_failed(strip_bytes, "push_decoder strip buffer"))?;
    strip_buf.resize(strip_bytes, 0);

    let mut y = 0u32;

    while !reader.is_finished() {
        // Check cooperative cancellation before decoding each MCU-row strip.
        if let Some(ref stop) = job.stop {
            use enough::Stop;
            stop.check()?;
        }

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

    let mut out = OutputInfo::full_decode(width as u32, height as u32, descriptor);

    // Report orientation if auto-orient was applied
    if will_auto_orient(job.orientation)
        && let Some(ref exif) = header.exif
        && let Some(orient_val) = crate::lossless::parse_exif_orientation(exif)
    {
        let orient = zencodec::Orientation::from_exif(orient_val).unwrap_or_default();
        out = out.with_orientation_applied(orient);
    }

    if let Some((x, y, cw, ch)) = job.crop_hint {
        out = out.with_crop_applied([x, y, cw, ch]);
    }

    Ok(out)
}

/// Whether the given orientation hint means we should auto-orient during decode.
fn will_auto_orient(hint: zencodec::OrientationHint) -> bool {
    use zencodec::OrientationHint;
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
    orientation: zencodec::OrientationHint,
    policy: Option<&zencodec::decode::DecodePolicy>,
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
        zencodec::ThreadingPolicy::SingleThread => {
            cfg = cfg.num_threads(1);
        }
        zencodec::ThreadingPolicy::LimitOrSingle { max_threads } => {
            cfg = cfg.num_threads(max_threads as usize);
        }
        zencodec::ThreadingPolicy::LimitOrAny {
            preferred_max_threads,
        } => {
            cfg = cfg.num_threads(preferred_max_threads as usize);
        }
        _ => {} // Balanced, Unlimited — use default (auto)
    }

    // Map decode policy to strictness and metadata preservation
    if let Some(pol) = policy {
        if let Some(strict) = pol.strict
            && strict
        {
            cfg = cfg.strict();
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

/// One-shot JPEG decoder implementing [`zencodec::decode::Decode`].
pub struct JpegDecoder<'a> {
    config: JpegDecoderConfig,
    stop: Option<zencodec::StopToken>,
    limits: ResourceLimits,
    crop_hint: Option<(u32, u32, u32, u32)>,
    orientation: zencodec::OrientationHint,
    policy: Option<zencodec::decode::DecodePolicy>,
    data: Cow<'a, [u8]>,
    preferred: Vec<PixelDescriptor>,
}

impl zencodec::decode::Decode for JpegDecoder<'_> {
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
            cfg = cfg.preserve_all_metadata();

            if wants_f32 {
                cfg = cfg.output_target(OutputTarget::LinearF32);
            }

            // Check dimension limits and progressive policy before full decode.
            // Header parse is cheap (marker scan only, no entropy decoding).
            let needs_header = limits.max_width.is_some()
                || limits.max_height.is_some()
                || self
                    .policy
                    .as_ref()
                    .is_some_and(|p| p.allow_progressive.is_some());
            if needs_header {
                let header = cfg.read_info(&data)?;
                if limits.max_width.is_some() || limits.max_height.is_some() {
                    limits.check_dimensions(header.dimensions.width, header.dimensions.height)?;
                }
                let is_progressive = matches!(
                    header.mode,
                    crate::types::JpegMode::Progressive
                        | crate::types::JpegMode::ArithmeticProgressive
                );
                if let Some(ref policy) = self.policy
                    && is_progressive
                    && !policy.resolve_progressive(true)
                {
                    return Err(Error::unsupported_feature(
                        "progressive JPEG rejected by decode policy",
                    ));
                }
            }

            let stop: &dyn enough::Stop = match &self.stop {
                Some(s) => s,
                None => &enough::Unstoppable,
            };
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
                        // If auto-orient was applied, report Identity; else source
                        if will_auto_orient(self.orientation) {
                            info = info.with_orientation(zencodec::Orientation::Identity);
                        } else {
                            info = info.with_orientation(
                                zencodec::Orientation::from_exif(orient).unwrap_or_default(),
                            );
                        }
                    }
                    info = info.with_exif(exif.to_vec());
                }
                if let Some(xmp) = extras.xmp() {
                    info = info.with_xmp(xmp.as_bytes().to_vec());
                }
                // Populate resolution from JFIF APP0 density.
                if let Some(jfif) = extras.jfif()
                    && let Some(resolution) = jfif_to_resolution(&jfif)
                {
                    info = info.with_resolution(resolution);
                }
            }

            let jpeg_extras = result.take_extras();

            // Derive correct pixel format descriptor from source color metadata.
            let corrected_cicp = self
                .config
                .inner
                .correct_color
                .as_ref()
                .map(|_| zenpixels::Cicp::SRGB);

            // Build PixelBuffer with zero-copy where possible
            let buf = if wants_f32 {
                let pixels_f32 = result.into_pixels_f32().unwrap_or_default();
                match format {
                    PixelFormat::Gray => {
                        let desc = zencodec::helpers::descriptor_for_decoded_pixels(
                            zenpixels::PixelFormat::GrayF32,
                            &info.source_color,
                            corrected_cicp.as_ref(),
                            zencodec::helpers::IccMatchTolerance::Intent,
                        )
                        .with_transfer(zenpixels::TransferFunction::Linear);
                        let gray: Vec<Gray<f32>> =
                            pixels_f32.iter().map(|&v| Gray::new(v)).collect();
                        PixelBuffer::from_pixels(gray, w, h)
                            .map_err(|_| Error::internal("pixel count mismatch"))?
                            .with_descriptor(desc)
                            .into()
                    }
                    _ => {
                        let pixel_count = (w as usize) * (h as usize);
                        if pixels_f32.len() == pixel_count * 3 {
                            let desc = zencodec::helpers::descriptor_for_decoded_pixels(
                                zenpixels::PixelFormat::RgbF32,
                                &info.source_color,
                                corrected_cicp.as_ref(),
                                zencodec::helpers::IccMatchTolerance::Intent,
                            )
                            .with_transfer(zenpixels::TransferFunction::Linear);
                            let raw_bytes = bytemuck::cast_slice::<f32, u8>(&pixels_f32).to_vec();
                            PixelBuffer::from_vec(raw_bytes, w, h, desc)
                                .map_err(|_| Error::internal("pixel buffer creation failed"))?
                        } else {
                            let desc = zencodec::helpers::descriptor_for_decoded_pixels(
                                zenpixels::PixelFormat::RgbaF32,
                                &info.source_color,
                                corrected_cicp.as_ref(),
                                zencodec::helpers::IccMatchTolerance::Intent,
                            )
                            .with_transfer(zenpixels::TransferFunction::Linear);
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
                                .with_descriptor(desc)
                                .into()
                        }
                    }
                }
            } else {
                let pixels_u8 = result.into_pixels_u8().unwrap_or_default();
                let pf = match format {
                    PixelFormat::Gray => zenpixels::PixelFormat::Gray8,
                    PixelFormat::Rgba => zenpixels::PixelFormat::Rgba8,
                    PixelFormat::Bgra => zenpixels::PixelFormat::Bgra8,
                    _ => zenpixels::PixelFormat::Rgb8,
                };
                let desc = zencodec::helpers::descriptor_for_decoded_pixels(
                    pf,
                    &info.source_color,
                    corrected_cicp.as_ref(),
                    zencodec::helpers::IccMatchTolerance::Intent,
                );
                PixelBuffer::from_vec(pixels_u8, w, h, desc)
                    .map_err(|_| Error::internal("pixel buffer creation failed"))?
            };

            let mut output = DecodeOutput::new(buf, info);
            if let Some(extras) = jpeg_extras {
                output = output.with_extras(extras);
            }
            if let Ok(probe) = crate::detect::probe(&data) {
                output = output.with_source_encoding_details(probe);
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

/// Streaming JPEG decoder implementing [`zencodec::decode::StreamingDecode`].
///
/// Wraps zenjpeg's `ScanlineReader` to yield scanline batches via `next_batch()`.
/// Each batch contains one MCU-row worth of decoded pixels (8 or 16 rows).
pub struct JpegStreamingDecoder<'a> {
    #[cfg(feature = "decoder")]
    reader: crate::decode::ScanlineReader<'a>,
    info: ImageInfo,
    descriptor: PixelDescriptor,
    /// Reusable row buffer for decoded pixel data (sized for MCU-row batches).
    /// 4-byte aligned so bytemuck casts to &mut [f32] are safe.
    row_buf: aligned_vec::AVec<u8, aligned_vec::ConstAlign<4>>,
    current_row: u32,
    /// MCU height in pixels (8 or 16 depending on subsampling).
    mcu_height: u32,
    /// Cooperative cancellation token.
    stop: Option<zencodec::StopToken>,
    #[cfg(not(feature = "decoder"))]
    _phantom: core::marker::PhantomData<&'a ()>,
}

impl zencodec::decode::StreamingDecode for JpegStreamingDecoder<'_> {
    type Error = Error;

    fn next_batch(&mut self) -> Result<Option<(u32, PixelSlice<'_>)>, Error> {
        #[cfg(feature = "decoder")]
        {
            use imgref::ImgRefMut;
            use zenpixels::{ChannelLayout, ChannelType};

            // Check cooperative cancellation before doing work.
            if let Some(ref stop) = self.stop {
                use enough::Stop;
                stop.check()?;
            }

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
    )
    .with_bit_depth(info.precision)
    .with_channel_count(info.num_components);

    if let Some(ref icc) = info.icc_profile {
        img_info = img_info.with_icc_profile(icc.clone());
    }
    if let Some(ref exif) = info.exif {
        if let Some(orient) = crate::lossless::parse_exif_orientation(exif) {
            img_info = img_info
                .with_orientation(zencodec::Orientation::from_exif(orient).unwrap_or_default());
        }
        img_info = img_info.with_exif(exif.clone());
    }
    if let Some(ref xmp) = info.xmp {
        img_info = img_info.with_xmp(xmp.as_bytes().to_vec());
    }

    if let Some(ref jfif) = info.jfif
        && let Some(resolution) = jfif_to_resolution(jfif)
    {
        img_info = img_info.with_resolution(resolution);
    }

    img_info = img_info.with_progressive(matches!(
        info.mode,
        crate::types::JpegMode::Progressive | crate::types::JpegMode::ArithmeticProgressive
    ));

    img_info
}

/// Build a [`SourceColor`] from a JPEG header for descriptor derivation.
#[cfg(feature = "decoder")]
fn source_color_from_header(info: &crate::decode::JpegInfo) -> zencodec::decode::SourceColor {
    let mut sc = zencodec::decode::SourceColor::default();
    if let Some(ref icc) = info.icc_profile {
        sc = sc.with_icc_profile(icc.clone());
    }
    sc
}

/// Derive the correct [`PixelDescriptor`] for decoded JPEG pixels.
///
/// Uses the shared zencodec utility to map source color metadata to a
/// descriptor that accurately reflects the pixel data's color space.
#[cfg(feature = "decoder")]
fn decode_descriptor(
    preferred: &[PixelDescriptor],
    header: &crate::decode::JpegInfo,
    correct_color: Option<&crate::color::icc::TargetColorSpace>,
) -> PixelDescriptor {
    let base = select_decode_descriptor(preferred, header.num_components);
    let sc = source_color_from_header(header);
    let corrected_cicp = correct_color.map(|_| zenpixels::Cicp::SRGB);
    zencodec::helpers::descriptor_for_decoded_pixels(
        base.pixel_format(),
        &sc,
        corrected_cicp.as_ref(),
        zencodec::helpers::IccMatchTolerance::Intent,
    )
}

/// Convert JFIF density info to a zencodec [`Resolution`](zencodec::Resolution).
///
/// Returns `None` for aspect-ratio-only density (unit = 0) or zero densities.
#[cfg(feature = "decoder")]
fn jfif_to_resolution(jfif: &crate::encode::extras::JfifInfo) -> Option<zencodec::Resolution> {
    use crate::encode::extras::DensityUnits;
    use zencodec::ResolutionUnit;

    if jfif.x_density == 0 || jfif.y_density == 0 {
        return None;
    }

    let unit = match jfif.density_units {
        DensityUnits::PixelsPerInch => ResolutionUnit::Inch,
        DensityUnits::PixelsPerCm => ResolutionUnit::Centimeter,
        DensityUnits::None => return None, // aspect ratio only, not real DPI
    };

    Some(zencodec::Resolution {
        x: jfif.x_density as f64,
        y: jfif.y_density as f64,
        unit,
    })
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::borrow::Cow;
    use imgref::{Img, ImgExt};
    use rgb::{Gray, Rgb, Rgba};
    use zencodec::encode::{EncodeJob as _, Encoder as _, EncoderConfig as _};

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
        let meta = Metadata::default().with_icc(icc.as_slice());
        let output = enc
            .job()
            .with_metadata(meta)
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
        let meta = Metadata::default().with_icc(icc.as_slice());
        let policy = zencodec::encode::EncodePolicy::strip_all();

        let output = enc
            .job()
            .with_metadata(meta)
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
        use zencodec::decode::{Decode as _, DecodeJob as _, DecoderConfig as _};
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
        use zencodec::decode::{DecodeJob as _, DecoderConfig as _, StreamingDecode as _};

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
        use zencodec::decode::{DecodeJob as _, DecoderConfig as _, StreamingDecode as _};

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

    #[cfg(feature = "decoder")]
    #[test]
    fn streaming_decode_cow_owned() {
        use zencodec::decode::{DecodeJob as _, DecoderConfig as _, StreamingDecode as _};

        // Encode a test image
        let enc = JpegEncoderConfig::new().with_calibrated_quality(95.0);
        let pixels: Vec<Rgb<u8>> = vec![
            Rgb {
                r: 200,
                g: 100,
                b: 50,
            };
            32 * 32
        ];
        let img = Img::new(pixels.as_slice(), 32, 32);
        let encoded = enc.encode(PixelSlice::from(img.as_ref()).into()).unwrap();

        // First decode with Cow::Borrowed as reference
        let dec = JpegDecoderConfig::new();
        let mut borrowed_stream = dec
            .job()
            .streaming_decoder(Cow::Borrowed(encoded.data()), &[PixelDescriptor::RGB8_SRGB])
            .unwrap();

        let mut borrowed_pixels = Vec::new();
        while let Some((_y, batch)) = borrowed_stream.next_batch().unwrap() {
            borrowed_pixels.extend_from_slice(batch.as_strided_bytes());
        }

        // Now decode with Cow::Owned — the key test
        let owned_data = encoded.data().to_vec();
        let dec2 = JpegDecoderConfig::new();
        let mut owned_stream = dec2
            .job()
            .streaming_decoder(Cow::Owned(owned_data), &[PixelDescriptor::RGB8_SRGB])
            .unwrap();

        assert_eq!(owned_stream.info().width, 32);
        assert_eq!(owned_stream.info().height, 32);

        let mut owned_pixels = Vec::new();
        let mut total_rows = 0u32;
        while let Some((y, batch)) = owned_stream.next_batch().unwrap() {
            assert_eq!(y, total_rows);
            owned_pixels.extend_from_slice(batch.as_strided_bytes());
            total_rows += batch.rows();
        }
        assert_eq!(total_rows, 32);

        // Owned and borrowed paths must produce identical output
        assert_eq!(
            owned_pixels, borrowed_pixels,
            "Cow::Owned output differs from Cow::Borrowed"
        );
    }

    #[cfg(feature = "decoder")]
    #[test]
    fn streaming_decode_cow_owned_is_effectively_static() {
        use zencodec::decode::{DecodeJob as _, DecoderConfig as _, StreamingDecode as _};

        // Encode a test image
        let enc = JpegEncoderConfig::new().with_calibrated_quality(85.0);
        let pixels: Vec<Rgb<u8>> = vec![
            Rgb {
                r: 128,
                g: 64,
                b: 32,
            };
            16 * 16
        ];
        let img = Img::new(pixels.as_slice(), 16, 16);
        let encoded = enc.encode(PixelSlice::from(img.as_ref()).into()).unwrap();

        // Create streaming decoder with owned data inside a scope,
        // then use it outside that scope (proves no external borrow).
        let owned_data = encoded.data().to_vec();
        let dec = JpegDecoderConfig::new();
        let mut stream = dec
            .job()
            .streaming_decoder(Cow::Owned(owned_data), &[PixelDescriptor::RGB8_SRGB])
            .unwrap();

        // The stream should work after the owned_data variable is consumed
        let mut total_rows = 0u32;
        while let Some((_y, batch)) = stream.next_batch().unwrap() {
            total_rows += batch.rows();
        }
        assert_eq!(total_rows, 16);
    }

    // ── Encoder trait roundtrip tests ────────────────────────────────

    fn encoder_trait_roundtrip(pixels: zenpixels::PixelSlice<'_>) {
        use zencodec::encode::Encoder;
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
        use zencodec::encode::EncoderConfig;
        let caps = JpegEncoderConfig::capabilities();
        assert!(caps.icc());
        assert!(caps.exif());
        assert!(caps.xmp());
        assert!(caps.stop());
        assert!(caps.lossy());
        assert!(!caps.lossless());
        assert!(!caps.animation());
        assert!(caps.push_rows());
        assert!(caps.encode_from());
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
        use zencodec::decode::DecoderConfig;
        let caps = JpegDecoderConfig::capabilities();
        assert!(caps.icc());
        assert!(caps.exif());
        assert!(caps.xmp());
        assert!(caps.stop());
        assert!(caps.cheap_probe());
        assert!(caps.streaming());
        assert!(caps.native_gray());
        assert!(caps.native_f32());
        assert!(caps.enforces_max_pixels());
        assert!(caps.enforces_max_memory());
        assert!(caps.enforces_max_input_bytes());
        assert!(!caps.animation());
    }

    #[test]
    fn decode_trait_max_width_enforced() {
        use zencodec::decode::{Decode as _, DecodeJob as _, DecoderConfig as _};

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
        use zencodec::decode::{Decode as _, DecodeJob as _, DecoderConfig as _};

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
        use zencodec::decode::{Decode as _, DecodeJob as _, DecoderConfig as _};

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
    fn animation_frame_encoder_returns_unsupported() {
        let config = JpegEncoderConfig::new();
        let result = config.job().animation_frame_encoder();
        assert!(result.is_err());
    }

    #[cfg(feature = "decoder")]
    #[test]
    fn animation_frame_decoder_returns_unsupported() {
        use zencodec::decode::{DecodeJob as _, DecoderConfig as _};

        let dec = JpegDecoderConfig::new();
        let result = dec.job().animation_frame_decoder(Cow::Borrowed(&[]), &[]);
        assert!(result.is_err());
    }

    /// Regression test: passing the full `supported_descriptors()` list (which
    /// includes f32 types like RGBF32_LINEAR) to `decoder()` must not panic.
    ///
    /// Previously, the f32-to-u8 conversion used `bytemuck::cast_vec::<f32, u8>()`
    /// which requires identical alignment (f32=4, u8=1 — always panics with
    /// AlignmentMismatch).
    #[cfg(feature = "decoder")]
    #[test]
    fn decode_with_full_descriptor_list_no_alignment_panic() {
        use zencodec::decode::{Decode as _, DecodeJob as _, DecoderConfig as _};

        // Encode a small RGB image
        let enc = JpegEncoderConfig::new().with_calibrated_quality(85.0);
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

        // Use the full supported descriptor list (includes f32 types)
        let dec = JpegDecoderConfig::new();
        let preferred = JpegDecoderConfig::supported_descriptors();

        // This must not panic — previously hit bytemuck AlignmentMismatch
        let output = dec
            .job()
            .decoder(Cow::Borrowed(encoded.data()), preferred)
            .unwrap()
            .decode()
            .unwrap();

        assert_eq!(output.info().width, 8);
        assert_eq!(output.info().height, 8);
    }

    #[cfg(feature = "decoder")]
    #[test]
    fn encode_from_pull_basic() {
        use zencodec::decode::{Decode as _, DecodeJob as _, DecoderConfig as _};
        use zencodec::encode::{EncodeJob as _, Encoder as _};
        use zenpixels::PixelSliceMut;

        let width = 32u32;
        let height = 32u32;
        let bpp = 3; // RGB8

        // Generate test pattern: horizontal gradient
        let row_bytes = width as usize * bpp;
        let total_bytes = row_bytes * height as usize;
        let mut src_pixels = alloc::vec![0u8; total_bytes];
        for y in 0..height as usize {
            for x in 0..width as usize {
                let offset = y * row_bytes + x * bpp;
                src_pixels[offset] = (x * 255 / 31) as u8; // R
                src_pixels[offset + 1] = (y * 255 / 31) as u8; // G
                src_pixels[offset + 2] = 128; // B
            }
        }

        let config = JpegEncoderConfig::new().with_generic_quality(85.0);
        let job = config.job().with_canvas_size(width, height);
        let encoder = job.encoder().unwrap();

        let encoded = encoder
            .encode_from(&mut |y, mut buf: PixelSliceMut<'_>| {
                let rows = buf.rows();
                for row in 0..rows {
                    let src_y = y + row;
                    if src_y >= height {
                        return row as usize;
                    }
                    let src_start = src_y as usize * row_bytes;
                    let src_end = src_start + row_bytes;
                    buf.row_mut(row)
                        .copy_from_slice(&src_pixels[src_start..src_end]);
                }
                rows as usize
            })
            .unwrap();

        assert!(!encoded.data().is_empty());
        assert!(encoded.data().len() > 100); // Sanity: not trivially small

        // Verify roundtrip: decode and check dimensions
        let dec = JpegDecoderConfig::new();
        let output = dec
            .job()
            .decoder(Cow::Borrowed(encoded.data()), &[])
            .unwrap()
            .decode()
            .unwrap();
        assert_eq!(output.info().width, width);
        assert_eq!(output.info().height, height);
    }

    #[test]
    fn encode_from_requires_canvas_size() {
        use zencodec::encode::{EncodeJob as _, Encoder as _};
        use zenpixels::PixelSliceMut;

        let config = JpegEncoderConfig::new();
        // No with_canvas_size — should error
        let encoder = config.job().encoder().unwrap();
        let result = encoder.encode_from(&mut |_y, _buf: PixelSliceMut<'_>| 0);
        assert!(result.is_err());
    }
}

#[cfg(test)]
mod streaming_test {
    use super::*;
    use zencodec::encode::{EncodeJob, EncoderConfig};

    #[test]
    fn streaming_encode_same_scope() {
        // The real pattern: config consumed by job, stop borrowed from caller scope.
        // Encoder lives in same scope as stop. No escape needed.
        let stop = enough::Unstoppable;
        let config = JpegEncoderConfig::default();
        let job = config
            .job() // config consumed — no config borrow
            .with_stop(zencodec::StopToken::new(stop)) // owned token
            .with_canvas_size(64, 64);
        let mut enc = job.dyn_encoder().unwrap();
        // enc borrows stop, both in same scope — compiles fine
        let pixels = vec![128u8; 64 * 64 * 4];
        let slice = zenpixels::PixelSlice::new(
            &pixels,
            64,
            64,
            64 * 4,
            zenpixels::PixelDescriptor::RGBA8_SRGB,
        )
        .unwrap();
        enc.push_rows(slice).unwrap();
        let _output = enc.finish().unwrap();
    }

    fn make_job(w: u32, h: u32) -> JpegEncodeJob {
        let config = JpegEncoderConfig::default();
        // Config is consumed by job() via clone. Job doesn't borrow config.
        // No stop set, so 'a = 'static.
        config.job().with_canvas_size(w, h)
    }

    #[test]
    fn job_escapes_scope_without_stop() {
        let job = make_job(64, 64);
        let _enc = job.dyn_encoder().unwrap();
    }
}
