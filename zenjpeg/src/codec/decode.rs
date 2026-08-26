//! zencodec decode-side impls: `JpegDecoderConfig` / `JpegDecodeJob` / `JpegDecoder`.

use alloc::borrow::Cow;
use alloc::vec::Vec;

use rgb::{Gray, Rgb};
use whereat::At;
use zencodec::decode::{DecodeCapabilities, DecodeOutput, OutputInfo};
use zencodec::{
    CodecError, ImageFormat, ImageInfo, ResourceLimits, Unsupported, UnsupportedOperation,
};
use zenpixels::{PixelBuffer, PixelDescriptor};

use crate::error::{Error, ErrorKind};
use crate::types::PixelFormat;

use super::info::{decode_descriptor, populate_info_from_jpeg_extras, to_image_info};
use super::streaming::JpegStreamingDecoder;

static JPEG_DECODE_CAPS: DecodeCapabilities = {
    let caps = DecodeCapabilities::new()
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
    // Ultra HDR gain maps: with the `ultrahdr` feature zenjpeg both surfaces
    // the gain map (GainMapRender::Components) and applies it itself
    // (GainMapRender::ReconstructHdr) — the honest reconstructs_hdr signal.
    #[cfg(feature = "ultrahdr")]
    let caps = caps.with_gain_map(true).with_reconstructs_hdr(true);
    caps
};

/// JPEG decoder configuration implementing [`zencodec::decode::DecoderConfig`].
///
/// Wraps [`crate::decode::DecodeConfig`] with the zencodec trait interface.
#[derive(Clone, Debug)]
pub struct JpegDecoderConfig {
    inner: crate::decode::DecodeConfig,
    #[allow(dead_code)]
    limits: ResourceLimits,
    /// How to handle CMYK/YCCK 4-component JPEGs. See [`CmykHandling`].
    cmyk_handling: CmykHandling,
}

/// How to handle CMYK/YCCK 4-component JPEGs on decode.
///
/// JPEGs with 4 components store ink values (CMYK) rather than light values
/// (RGB). Recovering accurate RGB requires a color management transform
/// driven by an ICC profile — usually embedded in the JPEG, sometimes
/// implied by Adobe `APP14` marker conventions. This enum picks the path.
///
/// Non-exhaustive so future named-profile variants (e.g. SWOP v2, FOGRA39)
/// can be added without a breaking change.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CmykHandling {
    /// Emit raw 4-channel CMYK bytes via [`zenpixels::PixelDescriptor::CMYK8`].
    ///
    /// Byte order is inverted CMYK (Adobe/libjpeg convention: 0 = full ink,
    /// 255 = no ink). The embedded ICC profile is preserved on
    /// `ImageInfo.source_color.icc_profile` so the caller can route the
    /// bytes through a CMS (moxcms, lcms2, etc.) for a color-accurate
    /// CMYK→RGB transform. This is the only faithful option for print
    /// workflows or any decoder that wants to honor the source profile.
    ///
    /// Default. Has no effect on non-CMYK input (grayscale, YCbCr).
    #[default]
    Passthrough,
    /// Convert CMYK→RGB using the naive `R = (1-C)(1-K)` formula, ignoring
    /// any embedded ICC profile.
    ///
    /// Cheap and self-contained but wrong: typical print-profile output
    /// lands 30–50 ΔE off true sRGB. Use only when you control the source
    /// pipeline and know the JPEGs are uncalibrated, or when a CMS is not
    /// available and approximate RGB is better than no RGB.
    BadRgb,
}

impl JpegDecoderConfig {
    /// Create a default decoder config.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: crate::decode::DecodeConfig::new(),
            limits: ResourceLimits::none(),
            cmyk_handling: CmykHandling::Passthrough,
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
            gain_map_render: zencodec::GainMapRender::default(),
            cached_header: None,
        }
    }

    /// Access the underlying [`DecodeConfig`](crate::decode::DecodeConfig).
    #[must_use]
    pub fn inner(&self) -> &crate::decode::DecodeConfig {
        &self.inner
    }

    /// Mutable access to the underlying [`DecodeConfig`](crate::decode::DecodeConfig).
    pub fn inner_mut(&mut self) -> &mut crate::decode::DecodeConfig {
        &mut self.inner
    }

    /// Enable post-decode deblocking to reduce JPEG block artifacts.
    ///
    /// Delegates to [`DecodeConfig::deblock()`](crate::decode::DecodeConfig::deblock).
    /// See [`DeblockMode`](crate::decode::DeblockMode) for available modes.
    #[must_use]
    pub fn deblock(mut self, mode: crate::decode::DeblockMode) -> Self {
        self.inner = self.inner.deblock(mode);
        self
    }

    /// Select how to handle CMYK/YCCK 4-component JPEGs. See [`CmykHandling`].
    ///
    /// Has no effect on non-CMYK input (grayscale, YCbCr).
    #[must_use]
    pub fn cmyk_handling(mut self, handling: CmykHandling) -> Self {
        self.cmyk_handling = handling;
        self
    }

    /// Returns the configured [`CmykHandling`] strategy.
    #[must_use]
    pub fn is_cmyk_handling(&self) -> CmykHandling {
        self.cmyk_handling
    }

    /// Convenience: probe image header with this config.
    ///
    /// Returns the shared [`At<CodecError>`] envelope (the zencodec trait path's
    /// `type Error`) so category + codec name survive type erasure.
    pub fn probe_header(&self, data: &[u8]) -> Result<ImageInfo, At<CodecError>> {
        use zencodec::decode::{DecodeJob as _, DecoderConfig as _};
        self.clone().job().probe(data)
    }

    /// Convenience: probe full image metadata (may be expensive).
    pub fn probe_full_metadata(&self, data: &[u8]) -> Result<ImageInfo, At<CodecError>> {
        use zencodec::decode::{DecodeJob as _, DecoderConfig as _};
        self.clone().job().probe_full(data)
    }

    /// Convenience: decode image with this config.
    pub fn decode(&self, data: &[u8]) -> Result<DecodeOutput, At<CodecError>> {
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
    type Error = At<CodecError>;
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

    fn estimate_decode_resources(
        &self,
        image: &zencodec::estimate::ImageCharacteristics,
        compute: &zencodec::estimate::ComputeEnvironment,
    ) -> zencodec::estimate::ResourceEstimate {
        use zencodec::estimate::{ResourceEstimate, ThreadingInformation};
        // Mirror the codec's `estimate_encode_resources` shape, sourcing the
        // peak (output buffer + MCU strips, plus full-frame coeff storage for
        // progressive/subsampled) and wall time from `heuristics::estimate_decode`.
        // Decode is serial by default (the `parallel` fast paths are opt-in via
        // restart segments); report SERIAL and let `at_cores` scale time.
        //
        // `estimate_decode` keys the output-buffer size off a `PixelFormat`, so
        // map the negotiated descriptor's bytes-per-pixel to the closest format.
        let format = match image.descriptor().bytes_per_pixel() {
            1 => PixelFormat::Gray,
            4 => PixelFormat::Rgba,
            _ => PixelFormat::Rgb,
        };
        let e = crate::heuristics::estimate_decode(image.width(), image.height(), format);
        ResourceEstimate::new(e.peak_memory_bytes, e.time_ms as u64)
            .with_peak_max(e.peak_memory_bytes_max)
            .with_threading(ThreadingInformation::SERIAL)
            .at_cores(compute.cores())
    }

    fn job<'a>(self) -> Self::Job<'a> {
        JpegDecodeJob {
            config: self,
            stop: None,
            limits: ResourceLimits::none(),
            crop_hint: None,
            orientation: zencodec::OrientationHint::default(),
            policy: None,
            gain_map_render: zencodec::GainMapRender::default(),
            cached_header: None,
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
    /// How an Ultra HDR gain-map image is rendered (BaseOnly / ReconstructHdr /
    /// Components). Default `BaseOnly`.
    gain_map_render: zencodec::GainMapRender,
    /// Cached header info from probe(). Avoids re-parsing in push_decoder_native.
    cached_header: Option<crate::decode::JpegInfo>,
}

impl<'a> zencodec::decode::DecodeJob<'a> for JpegDecodeJob {
    type Error = At<CodecError>;
    type Dec = JpegDecoder<'a>;
    type StreamDec = JpegStreamingDecoder<'a>;
    type AnimationFrameDec = Unsupported<At<CodecError>>;

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

    fn with_gain_map_render(mut self, render: zencodec::GainMapRender) -> Self {
        self.gain_map_render = render;
        self
    }

    fn probe(&self, data: &[u8]) -> Result<ImageInfo, Self::Error> {
        {
            // Check input size limits
            self.check_input_size(data)?;
            let info = self.limit_adjusted_inner().read_info(data)?;
            let mut image_info = to_image_info(&info);
            if let Ok(probe) = crate::detect::probe(data) {
                image_info = image_info.with_source_encoding_details(probe);
            }
            Ok(image_info)
        }
    }

    fn output_info(&self, data: &[u8]) -> Result<OutputInfo, Self::Error> {
        {
            self.check_input_size(data)?;
            let info = self.limit_adjusted_inner().read_info(data)?;
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
            gain_map_render: self.gain_map_render,
            data,
            preferred: preferred.to_vec(),
        })
    }

    fn push_decoder(
        mut self,
        data: Cow<'a, [u8]>,
        sink: &mut dyn zencodec::decode::DecodeRowSink,
        preferred: &[PixelDescriptor],
    ) -> Result<OutputInfo, Self::Error> {
        {
            // Pre-cache the header so push_decoder_native can reuse it
            // instead of calling read_info() again.
            if self.cached_header.is_none()
                && let Ok(info) = self.config.inner.read_info(&data)
            {
                self.cached_header = Some(info);
            }
            push_decoder_native(self, data, sink, preferred)
        }
    }

    fn streaming_decoder(
        self,
        data: Cow<'a, [u8]>,
        preferred: &[PixelDescriptor],
    ) -> Result<Self::StreamDec, Self::Error> {
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
    }

    fn animation_frame_decoder(
        self,
        _data: Cow<'a, [u8]>,
        _preferred: &[PixelDescriptor],
    ) -> Result<Self::AnimationFrameDec, Self::Error> {
        Err(ErrorKind::UnsupportedOperation(UnsupportedOperation::AnimationDecode).into())
    }
}

impl JpegDecodeJob {
    /// The inner decode config with this job's resource limits applied —
    /// probes must honor `with_limits` exactly like the decode path does
    /// (a 108 MP corpus file probes fine under a raised `max_pixels` but the
    /// inner config's default cap rejected it).
    fn limit_adjusted_inner(&self) -> crate::decode::DecodeConfig {
        let mut cfg = self.config.inner.clone();
        if let Some(max) = self.limits.max_pixels {
            cfg = cfg.max_pixels(max);
        }
        if let Some(bytes) = self.limits.max_memory_bytes {
            cfg = cfg.max_memory(bytes);
        }
        cfg
    }

    /// Check input data size against limits.
    fn check_input_size(&self, data: &[u8]) -> Result<(), Error> {
        self.limits
            .check_input_size(data.len() as u64)
            .map_err(|_| {
                Error::resource_limit_exceeded(
                    zencodec::LimitKind::InputSize,
                    data.len() as u64,
                    self.limits.max_input_bytes.unwrap_or(0),
                )
            })?;
        Ok(())
    }

    /// Check whether the decode policy allows progressive JPEGs.
    ///
    /// Returns an error if the image is progressive and the policy forbids it.
    fn check_progressive_policy(&self, mode: crate::types::JpegMode) -> Result<(), Error> {
        if let Some(ref policy) = self.policy {
            let is_progressive = matches!(
                mode,
                crate::types::JpegMode::Progressive | crate::types::JpegMode::ArithmeticProgressive
            );
            if is_progressive && !policy.resolve_progressive(true) {
                return Err(Error::policy_rejected(
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
fn push_decoder_native<'a>(
    mut job: JpegDecodeJob,
    data: Cow<'a, [u8]>,
    sink: &mut dyn zencodec::decode::DecodeRowSink,
    preferred: &[PixelDescriptor],
) -> Result<OutputInfo, At<CodecError>> {
    use imgref::ImgRefMut;
    use zenpixels::{ChannelLayout, ChannelType};

    let wrap = |e: zencodec::decode::SinkError| Error::io_error(e.to_string());

    // ScanlineReader is created and dropped within this function, so the
    // slice only needs scope-local lifetime — both Cow::Borrowed and
    // Cow::Owned work since `data` is owned by the function body.
    let data_ref: &[u8] = &data;
    job.check_input_size(data_ref)?;

    // Build decode config with limits, crop, orientation, policy
    let cfg = build_decode_config(
        &job.config.inner,
        &job.limits,
        job.crop_hint,
        job.orientation,
        job.policy.as_ref(),
    );

    // Reuse cached header from push_decoder (avoids re-parsing).
    // Falls back to read_info if cache miss (shouldn't happen in normal flow).
    let header = match job.cached_header.take() {
        Some(h) => h,
        None => job.config.inner.read_info(data_ref)?,
    };
    job.check_progressive_policy(header.mode)?;

    // Raw CMYK output for 4-component JPEGs: the streaming ScanlineReader has
    // no raw-CMYK output path, so fall back to the buffered Decode::decode()
    // path (which honors CmykHandling::Passthrough via PixelFormat::Cmyk)
    // and hand the full frame to the sink in one strip.
    if matches!(job.config.cmyk_handling, CmykHandling::Passthrough) && header.num_components == 4 {
        return push_decoder_via_full_decode(job, data, sink);
    }

    // ── Direct fast path ──────────────────────────────────────────────────
    //
    // When the request is a vanilla full-image decode to an u8 RGB-family or
    // grayscale buffer, bypass `ScanlineReader` and write straight into the
    // sink via `parser.to_pixels_into(...)`. This skips:
    //   - ScanlineReader's `buffered_rgb` Vec allocation (~width*height*3 B)
    //   - The bulk memcpy from buffered_rgb into the sink (~3-4 ms at 4K)
    //   - The duplicate header parse inside `cfg.scanline_reader()`
    let descriptor = decode_descriptor(preferred, &header, job.config.inner.correct_color.as_ref());
    if let Some(direct_format) = direct_path_pixel_format(descriptor, header.num_components)
        && job.crop_hint.is_none()
        && cfg.compute_effective_transform_from_data(data_ref)
            == crate::lossless::LosslessTransform::None
        && !matches!(
            cfg.deblock_mode,
            crate::decode::DeblockMode::Knusperli | crate::decode::DeblockMode::Auto
        )
        && cfg.output_target == crate::decode::OutputTarget::Srgb8
        // Progressive/arithmetic need coefficient storage — the direct path
        // uses streaming decode which only handles baseline sequential.
        && matches!(
            header.mode,
            crate::types::JpegMode::Baseline | crate::types::JpegMode::Extended
        )
    {
        return push_decoder_direct(job, data, sink, descriptor, direct_format, &header);
    }

    // Create the streaming scanline reader
    let mut reader = cfg.scanline_reader(data_ref)?;

    let width = reader.width() as usize;
    let height = reader.height() as usize;
    let mut descriptor = descriptor;
    let mcu_height = reader.luma_rows_per_mcu();

    let ch_type = descriptor.channel_type();
    let ch_layout = descriptor.layout();

    // read_rows_rgba_f32 always outputs 4 channels, so if caller requested
    // RGBF32 we must upgrade to RGBAF32 to match the actual output layout.
    if ch_type == ChannelType::F32 && ch_layout == ChannelLayout::Rgb {
        descriptor = PixelDescriptor::RGBAF32_LINEAR;
    }

    let bpp = descriptor.bytes_per_pixel();
    let _row_bytes = width * bpp;

    // Tell the sink what's coming
    sink.begin(width as u32, height as u32, descriptor)
        .map_err(wrap)?;

    // Strip size: for buffered-mode readers the entire image is already
    // decoded; ask the sink for one big buffer so we make O(1) callbacks
    // instead of O(height/mcu_height). For streaming readers we must stick
    // to one MCU row per call (that's the unit the reader produces).
    let strip_rows = if reader.is_buffered() {
        height
    } else {
        mcu_height
    };
    let mut y = 0u32;

    while !reader.is_finished() {
        // Check cooperative cancellation before decoding each MCU-row strip.
        // `StopReason` is not a `core::error::Error`, so route it through the
        // native `Error` (→ `ErrorKind::Cancelled`) before the envelope bridge.
        if let Some(ref stop) = job.stop {
            use enough::Stop;
            stop.check().map_err(Error::from)?;
        }

        let remaining = height - y as usize;
        let batch_max = remaining.min(strip_rows);
        if batch_max == 0 {
            break;
        }

        // Fast path: borrow the sink's buffer and have the reader write
        // directly into it, skipping the strip_buf staging copy.
        //
        // Safety of row count: read_rows_* only returns fewer than requested
        // rows at EOF, which batch_max already accounts for. In the non-EOF
        // path count always equals batch_max.
        let mut dst = sink
            .provide_next_buffer(y, batch_max as u32, width as u32, descriptor)
            .map_err(wrap)?;
        let dst_stride = dst.stride();
        let dst_bytes = dst.as_strided_bytes_mut();

        let count = match (ch_type, ch_layout) {
            (ChannelType::U8, ChannelLayout::Gray) => {
                let out = ImgRefMut::new(dst_bytes, dst_stride, batch_max);
                reader.read_rows_gray8(out)?
            }
            (ChannelType::U8, ChannelLayout::Rgb) => {
                let out = ImgRefMut::new(dst_bytes, dst_stride, batch_max);
                reader.read_rows_rgb8(out)?
            }
            (ChannelType::U8, ChannelLayout::Rgba) => {
                let out = ImgRefMut::new(dst_bytes, dst_stride, batch_max);
                if descriptor.alpha() == Some(zenpixels::AlphaMode::Undefined) {
                    reader.read_rows_rgbx8(out)?
                } else {
                    reader.read_rows_rgba8(out)?
                }
            }
            (ChannelType::U8, ChannelLayout::Bgra) => {
                let out = ImgRefMut::new(dst_bytes, dst_stride, batch_max);
                if descriptor.alpha() == Some(zenpixels::AlphaMode::Undefined) {
                    reader.read_rows_bgrx8(out)?
                } else {
                    reader.read_rows_bgra8(out)?
                }
            }
            (ChannelType::F32, ChannelLayout::Gray) => {
                let float_slice: &mut [f32] = bytemuck::cast_slice_mut(dst_bytes);
                let float_stride = dst_stride / 4;
                let f_out = ImgRefMut::new(float_slice, float_stride, batch_max);
                reader.read_rows_gray_f32(f_out)?
            }
            (ChannelType::F32, ChannelLayout::Rgb | ChannelLayout::Rgba) => {
                // read_rows_rgba_f32 always writes 4 f32 channels; descriptor
                // was already upgraded to RGBAF32 above.
                let float_slice: &mut [f32] = bytemuck::cast_slice_mut(dst_bytes);
                let float_stride = dst_stride / 4;
                let f_out = ImgRefMut::new(float_slice, float_stride, batch_max);
                reader.read_rows_rgba_f32(f_out)?
            }
            _ => {
                return Err(Error::unsupported_pixel_descriptor(
                    "unsupported pixel format for push_decoder",
                )
                .into());
            }
        };

        drop(dst);

        if count == 0 {
            break;
        }
        debug_assert_eq!(
            count, batch_max,
            "reader short-read: batch_max should match count outside EOF"
        );

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

/// Returns `Some(PixelFormat)` if the descriptor + component count are eligible
/// for the direct decode-into-sink fast path; `None` otherwise.
fn direct_path_pixel_format(
    descriptor: PixelDescriptor,
    num_components: u8,
) -> Option<PixelFormat> {
    use zenpixels::{ChannelLayout, ChannelType};
    let ch = descriptor.channel_type();
    let layout = descriptor.layout();
    let is_gray_src = num_components == 1;
    match (ch, layout) {
        (ChannelType::U8, ChannelLayout::Gray) if is_gray_src => Some(PixelFormat::Gray),
        (ChannelType::U8, ChannelLayout::Rgb) if !is_gray_src => Some(PixelFormat::Rgb),
        (ChannelType::U8, ChannelLayout::Rgba) if !is_gray_src => Some(PixelFormat::Rgba),
        (ChannelType::U8, ChannelLayout::Bgra) if !is_gray_src => {
            // Bgrx and Bgra share the same conversion (alpha=255).
            Some(PixelFormat::Bgra)
        }
        _ => None,
    }
}

/// Direct fast path: read header for dims, decode straight into the sink via
/// `DecodeConfig::decode_into` (skips ScanlineReader's buffered_rgb intermediate).
fn push_decoder_direct<'a>(
    job: JpegDecodeJob,
    data: Cow<'a, [u8]>,
    sink: &mut dyn zencodec::decode::DecodeRowSink,
    descriptor: PixelDescriptor,
    format: PixelFormat,
    header: &crate::decode::JpegInfo,
) -> Result<OutputInfo, At<CodecError>> {
    use enough::Unstoppable;

    let wrap = |e: zencodec::decode::SinkError| Error::io_error(e.to_string());

    let cfg = build_decode_config(
        &job.config.inner,
        &job.limits,
        job.crop_hint,
        job.orientation,
        job.policy.as_ref(),
    );

    let data_ref: &[u8] = &data;
    // Header already parsed by push_decoder_native — reuse it.
    let w = header.dimensions.width;
    let h = header.dimensions.height;

    // Streaming fast path: decode directly into the sink buffer, skipping
    // decode_into()'s redundant header parse. The caller (push_decoder_native)
    // already validated baseline mode, no crop/transform/deblock, u8 RGB-family.
    let stop_ref: &dyn enough::Stop = match &job.stop {
        Some(s) => s,
        None => &Unstoppable,
    };

    sink.begin(w, h, descriptor).map_err(wrap)?;
    let mut dst = sink
        .provide_next_buffer(0, h, w, descriptor)
        .map_err(wrap)?;
    let dst_stride = dst.stride();
    let bpp = format.bytes_per_pixel();
    let row_bytes = w as usize * bpp;

    let decode_into_result = if dst_stride == row_bytes {
        let dst_bytes = dst.as_strided_bytes_mut();
        if let Some(ref stop) = job.stop {
            cfg.decode_streaming_into(
                data_ref,
                format,
                dst_bytes,
                stop.clone(),
                header.num_components,
                header.is_xyb,
            )
        } else {
            cfg.decode_streaming_into(
                data_ref,
                format,
                dst_bytes,
                Unstoppable,
                header.num_components,
                header.is_xyb,
            )
        }
    } else {
        let total = row_bytes * h as usize;
        let mut contiguous = vec![0u8; total];
        let streaming_result = if let Some(ref stop) = job.stop {
            cfg.decode_streaming_into(
                data_ref,
                format,
                &mut contiguous,
                stop.clone(),
                header.num_components,
                header.is_xyb,
            )
        } else {
            cfg.decode_streaming_into(
                data_ref,
                format,
                &mut contiguous,
                Unstoppable,
                header.num_components,
                header.is_xyb,
            )
        };
        match streaming_result {
            Ok(written) => {
                let dst_bytes = dst.as_strided_bytes_mut();
                let copy_rows = (written / row_bytes).min(h as usize);
                for y in 0..copy_rows {
                    let src_start = y * row_bytes;
                    let dst_start = y * dst_stride;
                    dst_bytes[dst_start..dst_start + row_bytes]
                        .copy_from_slice(&contiguous[src_start..src_start + row_bytes]);
                }
                Ok(written)
            }
            Err(e) => Err(e),
        }
    };

    // If decode_into succeeded, we're done.
    if decode_into_result.is_ok() {
        drop(dst);
        sink.finish().map_err(wrap)?;

        let mut out = OutputInfo::full_decode(w, h, descriptor);
        if will_auto_orient(job.orientation)
            && let Some(ref exif) = header.exif
            && let Some(orient_val) = crate::lossless::parse_exif_orientation(exif)
        {
            let orient = zencodec::Orientation::from_exif(orient_val).unwrap_or_default();
            out = out.with_orientation_applied(orient);
        }
        return Ok(out);
    }

    // Fallback: cfg.decode() handles all modes safely.
    drop(dst);
    let cfg = cfg.output_format(format);
    let result = cfg.decode(data_ref, stop_ref)?;
    let decoded_w = result.width();
    let decoded_h = result.height();
    let decoded_stride = result.stride();
    let decoded_bpp = result.bytes_per_pixel();
    let pixels = result
        .into_pixels_u8()
        .ok_or_else(|| Error::internal("push_decoder_direct requires u8 output"))?;

    let out_w = decoded_w.min(w);
    let out_h = decoded_h.min(h);
    let fb_row_bytes = out_w as usize * decoded_bpp;

    // Re-acquire the sink buffer (we dropped it before the fallback decode)
    sink.begin(out_w, out_h, descriptor).map_err(wrap)?;
    let mut dst = sink
        .provide_next_buffer(0, out_h, out_w, descriptor)
        .map_err(wrap)?;
    let dst_stride = dst.stride();
    let dst_bytes = dst.as_strided_bytes_mut();

    for y in 0..out_h as usize {
        let src_start = y * decoded_stride;
        let dst_start = y * dst_stride;
        if src_start + fb_row_bytes <= pixels.len() && dst_start + fb_row_bytes <= dst_bytes.len() {
            dst_bytes[dst_start..dst_start + fb_row_bytes]
                .copy_from_slice(&pixels[src_start..src_start + fb_row_bytes]);
        }
    }
    drop(dst);

    sink.finish().map_err(wrap)?;

    let mut out = OutputInfo::full_decode(w, h, descriptor);
    if will_auto_orient(job.orientation)
        && let Some(ref exif) = header.exif
        && let Some(orient_val) = crate::lossless::parse_exif_orientation(exif)
    {
        let orient = zencodec::Orientation::from_exif(orient_val).unwrap_or_default();
        out = out.with_orientation_applied(orient);
    }
    Ok(out)
}

/// Fallback push_decoder path for raw CMYK output (4-component JPEGs).
///
/// The streaming `ScanlineReader` does not support raw-CMYK output — only
/// pre-converted RGB/BGRA/grayscale. So we run a buffered decode (which
/// honors `CmykHandling::Passthrough`) and then forward the whole frame to
/// the sink as a single strip. This gives up the streaming memory advantage
/// but is the only way to emit raw CMYK today.
fn push_decoder_via_full_decode<'a>(
    job: JpegDecodeJob,
    data: Cow<'a, [u8]>,
    sink: &mut dyn zencodec::decode::DecodeRowSink,
) -> Result<OutputInfo, At<CodecError>> {
    use zencodec::decode::{Decode as _, DecodeJob as _};

    let wrap = |e: zencodec::decode::SinkError| Error::io_error(e.to_string());

    // Build a decoder from the same config and hand it the data. This path
    // runs the full CMYK-aware decode (Decode::decode below overrides
    // output_format to PixelFormat::Cmyk when cmyk_handling is Passthrough
    // and the JPEG is 4-component). Both calls already return the envelope.
    let decoder = job.decoder(data, &[])?;
    let output = decoder.decode()?;
    let info = output.info().clone();
    let pixels = output.pixels();
    let descriptor = pixels.descriptor();
    let w = pixels.width();
    let h = pixels.rows();

    sink.begin(w, h, descriptor).map_err(wrap)?;

    // Hand the sink a single strip containing the whole frame.
    let mut dst = sink
        .provide_next_buffer(0, h, w, descriptor)
        .map_err(wrap)?;
    for row in 0..h {
        dst.row_mut(row).copy_from_slice(pixels.row(row));
    }
    drop(dst);
    sink.finish().map_err(wrap)?;

    let mut out = OutputInfo::full_decode(w, h, descriptor);
    if info.orientation != zencodec::Orientation::Identity {
        out = out.with_orientation_applied(info.orientation);
    }
    Ok(out)
}

/// Whether the given orientation hint means we should auto-orient during decode.
pub(super) fn will_auto_orient(hint: zencodec::OrientationHint) -> bool {
    use zencodec::OrientationHint;
    match hint {
        OrientationHint::Preserve => false,
        OrientationHint::Correct | OrientationHint::CorrectAndTransform(_) => true,
        OrientationHint::ExactTransform(_) => false,
        _ => false,
    }
}

/// Build a DecodeConfig with limit overrides and hints applied.
fn build_decode_config(
    inner: &crate::decode::DecodeConfig,
    limits: &ResourceLimits,
    crop_hint: Option<(u32, u32, u32, u32)>,
    orientation: zencodec::OrientationHint,
    policy: Option<&zencodec::decode::DecodePolicy>,
) -> crate::decode::DecodeConfig {
    let mut cfg = inner.clone();
    // Per-site allocation-fallibility preference travels with the rest of the
    // resource governance: big untrusted output/coefficient buffers stay
    // fallible by default, small bounded MCU scratch stays infallible, and an
    // explicit Fallible/Infallible here overrides every site.
    cfg = cfg.alloc_pref(limits.prefer_fallible_allocations);
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

    // Map threading policy. The zencodec 0.1.19 bump collapsed the
    // previous five variants into `Sequential` / `Parallel` (the thread
    // count knob moved to the caller via `pool.install()`), so we route
    // through `is_parallel()` instead of matching the deprecated
    // variants.
    if !limits.threading.is_parallel() {
        cfg = cfg.num_threads(1);
    }
    // Parallel: leave num_threads at its default (auto).

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
pub(super) fn select_decode_descriptor(
    preferred: &[PixelDescriptor],
    num_components: u8,
) -> PixelDescriptor {
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
    gain_map_render: zencodec::GainMapRender,
    data: Cow<'a, [u8]>,
    preferred: Vec<PixelDescriptor>,
}

impl zencodec::decode::Decode for JpegDecoder<'_> {
    type Error = At<CodecError>;

    fn decode(self) -> Result<DecodeOutput, Self::Error> {
        {
            use crate::decode::OutputTarget;
            use crate::types::PixelFormat;
            use zenpixels::ChannelType;

            // Ultra HDR rendition intent. ReconstructHdr takes a dedicated
            // path when the file actually carries gain-map XMP; an image
            // without one decodes below as the (complete) base image.
            // Components decorates the normal decode at the end. Unknown
            // future modes are refused — never silently mis-rendered.
            match self.gain_map_render {
                zencodec::GainMapRender::BaseOnly | zencodec::GainMapRender::Components => {}
                zencodec::GainMapRender::ReconstructHdr { target_headroom } => {
                    #[cfg(feature = "ultrahdr")]
                    {
                        let has_gain_map_xmp = self
                            .config
                            .inner
                            .read_info(&self.data)
                            .ok()
                            .and_then(|i| i.xmp)
                            .is_some_and(|x| x.contains("hdrgm:"));
                        if has_gain_map_xmp {
                            return self
                                .decode_reconstruct_hdr(target_headroom)
                                .map_err(Into::into);
                        }
                        // No gain map: the base image IS the image.
                    }
                    #[cfg(not(feature = "ultrahdr"))]
                    {
                        let _ = target_headroom;
                        return Err(Error::unsupported_feature(
                            "GainMapRender::ReconstructHdr requires the `ultrahdr` feature",
                        )
                        .into());
                    }
                }
                _ => {
                    return Err(
                        Error::unsupported_feature("unrecognized GainMapRender mode").into(),
                    );
                }
            }

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

            // Single header parse for all pre-decode checks: dimension limits,
            // progressive policy, CMYK detection, and output format hinting.
            // Previously three separate read_info() calls.
            let header = cfg.read_info(&data)?;

            // Dimension limits. `check_dimensions` yields a `LimitExceeded`
            // (foreign — no envelope bridge), so map through the native `Error`.
            if limits.max_width.is_some() || limits.max_height.is_some() {
                limits
                    .check_dimensions(header.dimensions.width, header.dimensions.height)
                    .map_err(Error::from)?;
            }

            // Progressive policy
            let is_progressive = matches!(
                header.mode,
                crate::types::JpegMode::Progressive | crate::types::JpegMode::ArithmeticProgressive
            );
            if let Some(ref policy) = self.policy
                && is_progressive
                && !policy.resolve_progressive(true)
            {
                return Err(
                    Error::policy_rejected("progressive JPEG rejected by decode policy").into(),
                );
            }

            // Passthrough CMYK handling
            let is_raw_cmyk = if matches!(self.config.cmyk_handling, CmykHandling::Passthrough)
                && header.num_components == 4
            {
                cfg = cfg.output_format(PixelFormat::Cmyk);
                true
            } else {
                false
            };

            // Hint the internal decoder to produce BGRA/RGBA/etc. directly in
            // the streaming path, eliminating the post-decode full-buffer swizzle.
            if !wants_f32 && !is_raw_cmyk {
                let descriptor =
                    decode_descriptor(preferred, &header, self.config.inner.correct_color.as_ref());
                if let Some(pf) = direct_path_pixel_format(descriptor, header.num_components) {
                    cfg = cfg.output_format(pf);
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

            // Extract metadata. Source precision comes from the SOF header
            // (always 8 today — the parser rejects 12-bit streams; this
            // stays truthful when 12-bit support lands). Note this is the
            // SOURCE depth: an f32 output buffer still carries only
            // `bit_depth` significant bits (#146).
            let mut info = ImageInfo::new(w, h, ImageFormat::Jpeg)
                .with_bit_depth(header.precision)
                .with_channel_count(header.num_components);
            if let Some(extras) = result.extras() {
                info = populate_info_from_jpeg_extras(info, extras, self.orientation);
            }

            let jpeg_extras = result.take_extras();

            // Derive correct pixel format descriptor from source color metadata.
            let corrected_cicp = self
                .config
                .inner
                .correct_color
                .as_ref()
                .map(|_| zenpixels::ColorProfileSource::Cicp(zenpixels::Cicp::SRGB));

            // Build PixelBuffer with zero-copy where possible
            let buf = if wants_f32 {
                let pixels_f32 = result.into_pixels_f32().unwrap_or_default();
                match format {
                    PixelFormat::Gray => {
                        let desc = zencodec::helpers::descriptor_for_decoded_pixels_v2(
                            zenpixels::PixelFormat::GrayF32,
                            &info.source_color,
                            corrected_cicp.as_ref(),
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
                            let desc = zencodec::helpers::descriptor_for_decoded_pixels_v2(
                                zenpixels::PixelFormat::RgbF32,
                                &info.source_color,
                                corrected_cicp.as_ref(),
                            )
                            .with_transfer(zenpixels::TransferFunction::Linear);
                            let raw_bytes = bytemuck::cast_slice::<f32, u8>(&pixels_f32).to_vec();
                            PixelBuffer::from_vec(raw_bytes, w, h, desc)
                                .map_err(|_| Error::internal("pixel buffer creation failed"))?
                        } else {
                            let desc = zencodec::helpers::descriptor_for_decoded_pixels_v2(
                                zenpixels::PixelFormat::RgbaF32,
                                &info.source_color,
                                corrected_cicp.as_ref(),
                            )
                            .with_transfer(zenpixels::TransferFunction::Linear);
                            let rgb: Vec<Rgb<f32>> = pixels_f32
                                .as_chunks::<3>()
                                .0
                                .iter()
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
            } else if is_raw_cmyk {
                // Raw CMYK output: 4 bytes per pixel in inverted CMYK order
                // (mozjpeg convention). The descriptor carries ColorModel::Cmyk
                // so downstream consumers route through a CMS with the ICC
                // profile rather than interpreting bytes as RGB.
                let pixels_u8 = result.into_pixels_u8().unwrap_or_default();
                info.has_alpha = false;
                info.source_color.channel_count = Some(4);
                let desc = zenpixels::PixelDescriptor::CMYK8;
                PixelBuffer::from_vec(pixels_u8, w, h, desc)
                    .map_err(|_| Error::internal("pixel buffer creation failed"))?
            } else {
                let pixels_u8 = result.into_pixels_u8().unwrap_or_default();
                let pf = match format {
                    PixelFormat::Gray => zenpixels::PixelFormat::Gray8,
                    PixelFormat::Rgba => zenpixels::PixelFormat::Rgba8,
                    PixelFormat::Bgra => zenpixels::PixelFormat::Bgra8,
                    _ => zenpixels::PixelFormat::Rgb8,
                };
                let desc = zencodec::helpers::descriptor_for_decoded_pixels_v2(
                    pf,
                    &info.source_color,
                    corrected_cicp.as_ref(),
                );
                PixelBuffer::from_vec(pixels_u8, w, h, desc)
                    .map_err(|_| Error::internal("pixel buffer creation failed"))?
            };

            let mut output = DecodeOutput::new(buf, info);

            // GainMapRender::Components: surface the decoded gain map (pixels
            // + ISO 21496-1 parameters) as `zencodec::decode::DecodedGainMap`
            // in the output extras. Absent on non-gain-map images.
            #[cfg(feature = "ultrahdr")]
            if matches!(self.gain_map_render, zencodec::GainMapRender::Components)
                && let Some(ref extras) = jpeg_extras
            {
                // The gain map is surfaced in the same orientation as the base
                // buffer above (#151).
                let base_transform = cfg.compute_effective_transform_from_data(&data);
                if let Some(dgm) = decode_gain_map_components(extras, base_transform)? {
                    output = output.with_extras(dgm);
                }
            }
            #[cfg(not(feature = "ultrahdr"))]
            if matches!(self.gain_map_render, zencodec::GainMapRender::Components) {
                return Err(Error::unsupported_feature(
                    "GainMapRender::Components requires the `ultrahdr` feature",
                )
                .into());
            }

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
                Error::resource_limit_exceeded(
                    zencodec::LimitKind::OutputSize,
                    output_bytes,
                    self.limits.max_output_bytes.unwrap_or(0),
                )
            })?;

            Ok(output)
        }
    }
}

#[cfg(feature = "ultrahdr")]
impl JpegDecoder<'_> {
    /// Dedicated `GainMapRender::ReconstructHdr` path: decode the SDR base,
    /// decode the MPF gain-map image, apply it at the requested headroom, and
    /// return a linear HDR buffer (1.0 = SDR white, 203 nits).
    ///
    /// Envelope obligation (see `GainMapRender::ReconstructHdr`): the output
    /// [`ImageInfo`]'s `SourceColor` carries the derived peak as
    /// `content_light_level` and a mastering display built from the gain
    /// map's alternate-image capacity, so a downstream native-HDR encode is
    /// complete.
    fn decode_reconstruct_hdr(self, target_headroom: Option<f32>) -> Result<DecodeOutput, Error> {
        use crate::ultrahdr::UltraHdrExtras;
        use ultrahdr_core::gainmap::{HdrOutputFormat, apply_gainmap};

        /// SDR reference white (cd/m²) — 1.0 in the linear output maps here.
        const SDR_WHITE_NITS: f32 = 203.0;

        let limits = self.limits;
        // Crop hints are optional ("decoder may ignore"): a cropped base
        // cannot be aligned with the full-frame gain map under normalized
        // sampling, so this path decodes full-frame instead of mis-rendering
        // the gain field (#151).
        let mut cfg = build_decode_config(
            &self.config.inner,
            &limits,
            None,
            self.orientation,
            self.policy.as_ref(),
        );
        cfg = cfg.preserve_all_metadata();
        // The transform the requested decode would bake (EXIF when
        // auto-orienting, composed with any user decode_transform). The
        // reconstruction itself happens in stored space — upright base +
        // stored gain map — and this permutation is baked into the finished
        // HDR buffer at the end, so orientation is a pure pixel permutation
        // of one canonical reconstruction (byte-identical to `Preserve` plus
        // an external bake, the same construction as #149).
        let base_transform = cfg.compute_effective_transform_from_data(&self.data);
        cfg = cfg.auto_orient(false);
        cfg.decode_transform = None;
        // apply_gainmap consumes RGBA8 SDR input.
        cfg = cfg.output_format(crate::types::PixelFormat::Rgba);

        let header = cfg.read_info(&self.data)?;
        if limits.max_width.is_some() || limits.max_height.is_some() {
            limits.check_dimensions(header.dimensions.width, header.dimensions.height)?;
        }

        let stop: &dyn enough::Stop = match &self.stop {
            Some(s) => s,
            None => &enough::Unstoppable,
        };
        let mut result = cfg.decode(&self.data, stop)?;
        let w = result.width();
        let h = result.height();

        let extras = result.take_extras().ok_or_else(|| {
            Error::icc_error("Ultra HDR reconstruction: decoder produced no extras".into())
        })?;
        let (metadata, _) = extras.ultrahdr_metadata().ok_or_else(|| {
            Error::unsupported_feature("ReconstructHdr requested but the XMP has no gain map")
        })??;
        let gainmap = extras.decode_gainmap().ok_or_else(|| {
            Error::unsupported_feature("ReconstructHdr requested but the MPF has no gain-map image")
        })??;

        let pixels_u8 = result
            .into_pixels_u8()
            .ok_or_else(|| Error::internal("decoder produced no u8 pixels"))?;
        let sdr = PixelBuffer::from_vec(pixels_u8, w, h, zenpixels::PixelDescriptor::RGBA8_SRGB)
            .map_err(|_| Error::internal("pixel buffer creation failed"))?;

        // Output form: honor an f16 preference; default linear f32 RGBA.
        let wants_f16 = self
            .preferred
            .iter()
            .any(|d| d.channel_type() == zenpixels::ChannelType::F16);
        let format = if wants_f16 {
            HdrOutputFormat::LinearF16
        } else {
            HdrOutputFormat::LinearFloat
        };

        // `None` = full reconstruction at the gain map's encoded maximum,
        // via the canonical rounding route shared across adapters (heic#20 —
        // the previous `exp2` of the f32-cast stops double-rounded and could
        // land 1 ULP from the heic adapter's boost for identical params).
        let capacity_max = ultrahdr_core::full_reconstruction_boost(&metadata);
        let display_boost = target_headroom.unwrap_or(capacity_max).max(1.0);

        let hdr = apply_gainmap(&sdr, &gainmap, &metadata, display_boost, format, stop)
            .map_err(|e| Error::icc_error(alloc::format!("gain-map apply failed: {e}")))?;
        drop(sdr);
        // Bake the requested orientation into the finished reconstruction —
        // a pure pixel permutation of the stored-space result (#151), done
        // in place (cycle-following) so no second full-image HDR buffer is
        // allocated. Allocating fallback for any future descriptor or
        // orientation the in-place path declines.
        let hdr = match lossless_to_orientation(base_transform) {
            zencodec::Orientation::Identity => hdr,
            o => {
                let mut buf = hdr;
                match zenpixels_convert::orient::apply_orientation_in_place(&mut buf, o) {
                    Ok(()) => buf,
                    Err(_) => zenpixels_convert::orient::apply_orientation(buf.as_slice(), o),
                }
            }
        };

        let mut info = ImageInfo::new(hdr.width(), hdr.height(), ImageFormat::Jpeg);
        info = populate_info_from_jpeg_extras(info, &extras, self.orientation);

        // Envelope: the CONTENT light level is MEASURED from the
        // reconstructed pixels via the zenpixels measurement owner
        // (`CllMeasure::measure_max`, MaxRGB per CTA-861.3, BT.2408 anchor —
        // appendix AA: the gain map's declared capacity is a range BOUND,
        // usually wrong about actual content, so deriving CLL from it
        // over-states the content whenever the range isn't fully used).
        // MaxFALL comes from the same scan. The mastering display keeps the
        // capacity-derived peak: it describes what the encoding can EXPRESS
        // (a capability/config property), not what the content contains.
        // The f16 output form measures as None (the owner takes f32) and
        // falls back to the capacity-derived value.
        let capacity_nits = SDR_WHITE_NITS * capacity_max.min(display_boost);
        let measured = {
            use zenpixels_convert::hdr::measure::{CllMeasure, LightLevelMethod};
            zenpixels::hdr::ContentLightLevel::measure_max(
                hdr.as_slice(),
                zenpixels::hdr::DiffuseWhite::BT2408,
                LightLevelMethod::MaxRgb,
            )
        };
        info.source_color.content_light_level = Some(match measured {
            Some(cll) => zencodec::ContentLightLevel::new(
                cll.max_content_light_level,
                cll.max_frame_average_light_level,
            ),
            None => zencodec::ContentLightLevel::new(capacity_nits as u16, 0),
        });
        info.source_color.mastering_display = Some(zencodec::MasteringDisplay::new(
            [[0.640, 0.330], [0.300, 0.600], [0.150, 0.060]],
            [0.3127, 0.3290],
            capacity_nits,
            0.005,
        ));

        let mut output = DecodeOutput::new(hdr, info).with_extras(extras);
        if let Ok(probe) = crate::detect::probe(&self.data) {
            output = output.with_source_encoding_details(probe);
        }

        let output_bytes = output.pixels().rows() as u64
            * output.pixels().width() as u64
            * output.pixels().descriptor().bytes_per_pixel() as u64;
        limits.check_output_size(output_bytes).map_err(|_| {
            Error::resource_limit_exceeded(
                zencodec::LimitKind::OutputSize,
                output_bytes,
                limits.max_output_bytes.unwrap_or(0),
            )
        })?;

        Ok(output)
    }
}

/// Map the decoder's lossless transform to the equivalent `Orientation`
/// (both follow EXIF display semantics — `Rotate90` is 90° clockwise).
#[cfg(feature = "ultrahdr")]
fn lossless_to_orientation(t: crate::lossless::LosslessTransform) -> zencodec::Orientation {
    use crate::lossless::LosslessTransform as T;
    use zencodec::Orientation as O;
    match t {
        T::None => O::Identity,
        T::FlipHorizontal => O::FlipH,
        T::FlipVertical => O::FlipV,
        T::Transpose => O::Transpose,
        T::Transverse => O::Transverse,
        T::Rotate90 => O::Rotate90,
        T::Rotate180 => O::Rotate180,
        T::Rotate270 => O::Rotate270,
    }
}

/// Permute gain-map pixels by the same lossless transform the base decode
/// baked, keeping the (base, gain map) pair aligned for normalized-coordinate
/// sampling (#151). No-op for `LosslessTransform::None`.
#[cfg(feature = "ultrahdr")]
fn orient_gain_map(
    mut gm: ultrahdr_core::GainMap,
    transform: crate::lossless::LosslessTransform,
) -> ultrahdr_core::GainMap {
    if transform == crate::lossless::LosslessTransform::None {
        return gm;
    }
    gm.data = crate::decode::transform_interleaved(
        &gm.data,
        gm.width as usize,
        gm.height as usize,
        usize::from(gm.channels),
        transform,
    );
    if transform.swaps_dimensions() {
        core::mem::swap(&mut gm.width, &mut gm.height);
    }
    gm
}

/// Decode the Ultra HDR gain-map components for `GainMapRender::Components`.
///
/// Returns `Ok(None)` when the image carries no gain map (Components on a
/// plain JPEG surfaces nothing); errors only when a gain map is present but
/// malformed. The returned map is full-frame and oriented to match the base
/// buffer (`base_transform`); a crop hint crops only the base.
#[cfg(feature = "ultrahdr")]
fn decode_gain_map_components(
    extras: &crate::decode::DecodedExtras,
    base_transform: crate::lossless::LosslessTransform,
) -> Result<Option<zencodec::decode::DecodedGainMap>, Error> {
    use crate::ultrahdr::UltraHdrExtras;

    let Some(metadata) = extras.ultrahdr_metadata() else {
        return Ok(None);
    };
    let (metadata, _) = metadata?;
    let Some(gainmap) = extras.decode_gainmap() else {
        return Ok(None);
    };
    let gainmap = orient_gain_map(gainmap?, base_transform);

    let desc = if gainmap.channels == 3 {
        zenpixels::PixelDescriptor::RGB8_SRGB
    } else {
        zenpixels::PixelDescriptor::GRAY8_SRGB
    };
    let pixels = PixelBuffer::from_vec(gainmap.data, gainmap.width, gainmap.height, desc)
        .map_err(|_| Error::internal("gain-map pixel buffer creation failed"))?;
    let gm_info =
        zencodec::GainMapInfo::new(metadata, gainmap.width, gainmap.height, gainmap.channels);
    Ok(Some(zencodec::decode::DecodedGainMap::new(pixels, gm_info)))
}
