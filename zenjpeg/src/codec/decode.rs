//! zencodec decode-side impls: `JpegDecoderConfig` / `JpegDecodeJob` / `JpegDecoder`.

use alloc::borrow::Cow;
use alloc::vec::Vec;

use whereat::At;
use zencodec::decode::{DecodeCapabilities, DecodeOutput, OutputInfo};
use zencodec::sample::ExactDepth;
use zencodec::{
    CodecError, ImageFormat, ImageInfo, ResourceLimits, Unsupported, UnsupportedOperation,
};
use zenpixels::{ChannelOrder, PixelBuf, PixelFormat as ZenPixelFormat, SampleType, SignalRange};

use crate::error::{Error, ErrorKind};
use crate::types::PixelFormat;

use super::info::{
    decode_descriptor, needs_plane_reads, populate_info_from_jpeg_extras, to_image_info,
};
use super::streaming::JpegStreamingDecoder;

#[cfg(feature = "controlled-decode")]
mod controlled;

static JPEG_DECODE_CAPS: DecodeCapabilities = {
    let caps = DecodeCapabilities::new()
        .with_incremental(true)
        .with_icc(true)
        .with_exif(true)
        .with_xmp(true)
        .with_stop(true)
        .with_cheap_probe(true)
        .with_streaming(true)
        .with_native_gray(true)
        .with_native_16bit(true)
        .with_native_f32(true)
        .with_enforces_max_pixels(true)
        .with_enforces_max_memory(true)
        .with_enforces_max_input_bytes(true)
        .with_inventory(cfg!(feature = "inventory"))
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
    /// Emit raw 4-channel CMYK with [`zenpixels::ChannelOrder::CmykInverted`].
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
    /// This is equivalent to [`DecoderConfig::job`](zencodec::decode::DecoderConfig::job) but available
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

    /// Probe recognized metadata through the primary image's EOI without
    /// decoding pixels. Requires the complete primary image; excludes trailers,
    /// nested images and unrecognized carriers. The metadata matches preserved
    /// native extras, including late carriers, and is not a claim that a late
    /// profile participated in native pixel color conversion.
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
static DECODE_DESCRIPTORS: &[ZenPixelFormat] = &[
    ZenPixelFormat::RGB8_SRGB,
    ZenPixelFormat::RGBA8_SRGB,
    ZenPixelFormat::GRAY8_SRGB,
    ZenPixelFormat::BGRA8_SRGB,
    ZenPixelFormat::RGBX8_SRGB,
    ZenPixelFormat::BGRX8_SRGB,
    ZenPixelFormat::RGB16_SRGB,
    ZenPixelFormat::RGBA16_SRGB,
    ZenPixelFormat::GRAY16_SRGB,
    ZenPixelFormat::RGBF32_LINEAR,
    ZenPixelFormat::RGBAF32_LINEAR,
    ZenPixelFormat::GRAYF32_LINEAR,
];

impl zencodec::decode::DecoderConfig for JpegDecoderConfig {
    type Error = At<CodecError>;
    type Job<'a> = JpegDecodeJob;

    fn formats() -> &'static [ImageFormat] {
        &[ImageFormat::Jpeg]
    }

    fn supported_descriptors() -> &'static [ZenPixelFormat] {
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
        let format = match image.descriptor().layout.bytes_per_pixel() {
            1 | 2 => PixelFormat::Gray,
            4 | 8 => PixelFormat::Rgba,
            _ => PixelFormat::Rgb,
        };
        let e = crate::heuristics::estimate_decode(image.width(), image.height(), format);
        // The model counts one byte per sample. U16 exits hold one more byte per
        // sample in their output buffer.
        let wide_output = if image.descriptor().layout.sample == SampleType::U16 {
            u64::from(image.width())
                .saturating_mul(u64::from(image.height()))
                .saturating_mul(image.descriptor().layout.bytes_per_pixel() as u64 / 2)
        } else {
            0
        };
        ResourceEstimate::new(
            e.peak_memory_bytes.saturating_add(wide_output),
            e.time_ms as u64,
        )
        .with_peak_max(e.peak_memory_bytes_max.saturating_add(wide_output))
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
/// Created by [`DecoderConfig::job`](zencodec::decode::DecoderConfig::job). Consumed by creating a
/// [`JpegDecoder`] or [`JpegStreamingDecoder`].
pub struct JpegDecodeJob {
    config: JpegDecoderConfig,
    /// Cancellation (and progress) control, polled as a `Stop`.
    stop: Option<zencodec::SharedPulse>,
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

    fn with_pulse(mut self, pulse: zencodec::SharedPulse) -> Self {
        self.stop = Some(pulse);
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

    fn try_set_policy(
        &mut self,
        policy: zencodec::decode::DecodePolicy,
    ) -> Result<(), zencodec::UnsupportedOperation> {
        // Partial-image success and strictness across every probe/decode path
        // are not qualified. Refuse before changing the stored job.
        if policy.allow_truncated == Some(true) || policy.strict.is_some() {
            return Err(zencodec::UnsupportedOperation::DecodePolicy);
        }
        self.policy = Some(policy);
        Ok(())
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

    /// A byte-exact map of `data` as this job's decode path reads it; see
    /// `codec/inventory.rs` for what each disposition means here. It follows
    /// the job's policy and strictness, orientation hint, gain-map render
    /// and pixel, width and height limits, and assumes `max_memory_bytes`
    /// is large enough for the decode. Where the decode depends on the
    /// output format the caller later asks for, dispositions describe the
    /// default output and the part's detail names the difference.
    #[cfg(feature = "inventory")]
    fn inventory(&self, data: &[u8]) -> Result<zencodec::inventory::Inventory, At<CodecError>> {
        if let Some(stop) = &self.stop {
            enough::Stop::check(stop).map_err(Error::from)?;
        }
        use whereat::ErrorAtExt;
        use zencodec::CategorizedError;
        self.limits
            .check_input_size(data.len() as u64)
            .map_err(|cause| {
                CodecError::from_parts(
                    Some("zenjpeg"),
                    cause.category(),
                    alloc::boxed::Box::new(cause),
                )
                .start_at()
            })?;
        let opts = super::inventory::Options {
            allow_exif: self
                .policy
                .as_ref()
                .is_none_or(|p| p.allow_exif != Some(false)),
            allow_icc: self
                .policy
                .as_ref()
                .is_none_or(|p| p.allow_icc != Some(false)),
            allow_xmp: self
                .policy
                .as_ref()
                .is_none_or(|p| p.allow_xmp != Some(false)),
            auto_orient: will_auto_orient(self.orientation),
            render: self.inventory_render(data),
            max_pixels: self.limit_adjusted_inner().get_max_pixels(),
            // `probe()` reads the header with the inner config;
            // `build_decode_config` raises the decode to Strict.
            probe_strictness: self.config.inner.strictness,
            decode_strictness: match &self.policy {
                Some(p) if p.strict == Some(true) || p.allow_truncated == Some(false) => {
                    crate::decode::Strictness::Strict
                }
                _ => self.config.inner.strictness,
            },
            allow_progressive: self
                .policy
                .as_ref()
                .is_none_or(|p| p.resolve_progressive(true)),
            max_width: self.limits.max_width,
            max_height: self.limits.max_height,
        };
        match super::inventory::inventory(
            data,
            opts,
            self.stop
                .as_ref()
                .map_or(&enough::Unstoppable as &dyn enough::Stop, |s| s),
        ) {
            Ok(inv) => Ok(inv),
            Err(super::inventory::Failed::Parts(cause)) => Err(CodecError::from_parts(
                Some("zenjpeg"),
                cause.category(),
                alloc::boxed::Box::new(cause),
            )
            .start_at()),
            Err(super::inventory::Failed::Stopped(reason)) => Err(Error::from(reason).into()),
        }
    }

    fn probe(&self, data: &[u8]) -> Result<ImageInfo, Self::Error> {
        {
            // Check input size limits
            self.check_input_size(data)?;
            let info = read_job_info(&self.limit_adjusted_inner(), data, self.policy.as_ref())?;
            #[cfg_attr(not(feature = "metadata"), allow(unused_mut))]
            let mut image_info = to_image_info(&info);
            #[cfg(feature = "metadata")]
            if let Ok(probe) = crate::detect::probe(data) {
                image_info = image_info.with_source_encoding_details(probe);
            }
            Ok(image_info)
        }
    }

    fn probe_full(&self, data: &[u8]) -> Result<ImageInfo, Self::Error> {
        self.check_input_size(data)?;
        let stop: &dyn enough::Stop = match &self.stop {
            Some(stop) => stop,
            None => &enough::Unstoppable,
        };
        let info = self.limit_adjusted_inner().read_full_info_preserving(
            data,
            &metadata_preserve(self.policy.as_ref()),
            stop,
        )?;
        let mut image_info = to_image_info(&info);
        // Full-file discovery distinguishes an ordinary primary image from
        // embedded gain-map/multi-image content, even when the signal is late.
        // Header probes deliberately retain Unknown.
        stop.check().map_err(Error::from)?;
        super::info::populate_complete_container_scope(data, &mut image_info);
        stop.check().map_err(Error::from)?;
        #[cfg(feature = "metadata")]
        if let Ok(probe) = crate::detect::probe(data) {
            image_info = image_info.with_source_encoding_details(probe);
        }
        Ok(image_info)
    }

    fn output_info(&self, data: &[u8]) -> Result<OutputInfo, Self::Error> {
        {
            self.check_input_size(data)?;
            let info = read_job_info(&self.limit_adjusted_inner(), data, self.policy.as_ref())?;
            let native_format =
                decode_descriptor(&[], &info, self.config.inner.correct_color.as_ref())?;
            let mut w = info.dimensions.width;
            let mut h = info.dimensions.height;

            let mut out = OutputInfo::full_decode(w, h, native_format.clone());

            let will_orient = will_auto_orient(self.orientation);
            if will_orient && let Some(orient_val) = crate::decode::find_exif_orientation(data) {
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
        preferred: &[ZenPixelFormat],
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
        preferred: &[ZenPixelFormat],
    ) -> Result<OutputInfo, Self::Error> {
        {
            // Pre-cache the header so push_decoder_native can reuse it
            // instead of calling read_info() again.
            if self.cached_header.is_none()
                && let Ok(info) = read_job_info(&self.config.inner, &data, self.policy.as_ref())
            {
                self.cached_header = Some(info);
            }
            push_decoder_native(self, data, sink, preferred)
        }
    }

    fn streaming_decoder(
        self,
        data: Cow<'a, [u8]>,
        preferred: &[ZenPixelFormat],
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
            let mut header = read_job_info(&self.config.inner, &data, self.policy.as_ref())?;
            // Streaming pixels use full-file orientation selection, including
            // post-scan EXIF. Report that same packet rather than header-only EXIF.
            if cfg.preserve.exif {
                header.exif = crate::decode::first_exif_segment(&data).map(|s| s.payload.to_vec());
            }
            self.check_progressive_policy(header.mode)?;
            let mut info = to_image_info(&header);

            // If auto-orient was applied, report Identity orientation
            if will_auto_orient(self.orientation) {
                info = info.with_orientation(zencodec::Orientation::Identity);
            }

            let descriptor =
                decode_descriptor(preferred, &header, self.config.inner.correct_color.as_ref())?;

            let cfg = if descriptor.layout.sample == zenpixels::SampleType::F32 {
                cfg.output_target(crate::decode::OutputTarget::LinearF32)
            } else if u16_uses_native_codes(&descriptor, header.precision) {
                cfg.output_target(crate::decode::OutputTarget::Native16)
            } else {
                cfg
            };

            // scanline_reader_cow accepts both Borrowed and Owned data.
            // When Owned, the reader stores the Vec internally.
            let reader = if needs_plane_reads(&descriptor) {
                cfg.for_plane_reads().scanline_reader_cow(data)?
            } else {
                cfg.scanline_reader_cow(data)?
            };
            info.width = reader.width();
            info.height = reader.height();
            let mcu_height = reader.luma_rows_per_mcu();

            Ok(JpegStreamingDecoder {
                reader,
                origin: info.source_color.color_origin(),
                info,
                descriptor,
                row_buf: aligned_vec::AVec::new(4),
                current_row: 0,
                mcu_height: mcu_height as u32,
                precision: header.precision,
                stop: self.stop,
            })
        }
    }

    fn animation_frame_decoder(
        self,
        _data: Cow<'a, [u8]>,
        _preferred: &[ZenPixelFormat],
    ) -> Result<Self::AnimationFrameDec, Self::Error> {
        Err(ErrorKind::UnsupportedOperation(UnsupportedOperation::AnimationDecode).into())
    }

    /// Incremental (sans-I/O) decode: the header is reported as soon as it
    /// probes, and the image decodes through `push_decoder` once the whole
    /// file is offered (a native baseline decoder that streams MCU rows is planned; progressive, arithmetic, CMYK and transformed outputs need the whole file in any case).
    fn dyn_incremental_decoder(
        self,
        preferred: &[ZenPixelFormat],
    ) -> Result<
        alloc::boxed::Box<zencodec::decode::DynIncrementalDecoder>,
        zencodec::decode::BoxedError,
    > {
        Ok(zencodec::decode::erase_incremental(
            zencodec::helpers::WholeFileIncremental::new(self, preferred),
        ))
    }

    #[cfg(feature = "controlled-decode")]
    fn dyn_controlled_decoder(
        self,
        preferred: &[ZenPixelFormat],
        storage: zencodec::storage::StorageBudget,
    ) -> Result<
        alloc::boxed::Box<zencodec::decode::DynControlledDecoder>,
        zencodec::decode::BoxedError,
    > {
        controlled::start(self, preferred, storage)
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

    /// The job's `GainMapRender` as the decode acts on it (see `decode`).
    #[cfg(feature = "inventory")]
    fn inventory_render(&self, data: &[u8]) -> super::inventory::Render {
        use super::inventory::Render;
        match self.gain_map_render {
            zencodec::GainMapRender::BaseOnly => Render::Base,
            #[cfg(feature = "ultrahdr")]
            zencodec::GainMapRender::Components => Render::Components,
            #[cfg(feature = "ultrahdr")]
            zencodec::GainMapRender::ReconstructHdr { .. } => Render::Reconstruct,
            _ => {
                let _ = data;
                Render::Refused
            }
        }
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
    preferred: &[ZenPixelFormat],
) -> Result<OutputInfo, At<CodecError>> {
    use imgref::ImgRefMut;
    use zenpixels::{ChannelOrder, SampleType};

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
    let mut header = match job.cached_header.take() {
        Some(h) => h,
        None => read_job_info(&job.config.inner, data_ref, job.policy.as_ref())?,
    };
    if cfg.preserve.exif {
        header.exif = crate::decode::first_exif_segment(data_ref).map(|s| s.payload.to_vec());
    }
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
    let descriptor =
        decode_descriptor(preferred, &header, job.config.inner.correct_color.as_ref())?;
    let mut begun = false;
    if let Some(direct_format) = direct_path_pixel_format(&descriptor, header.num_components)
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
        let (w, h) = (header.dimensions.width, header.dimensions.height);
        sink.begin(w, h, descriptor.clone()).map_err(wrap)?;
        if sink.max_buffer_rows().get() >= h {
            return push_decoder_direct(job, data, sink, descriptor, direct_format, &header);
        }
        begun = true;
    }

    let cfg = if u16_uses_native_codes(&descriptor, header.precision) {
        cfg.output_target(crate::decode::OutputTarget::Native16)
    } else {
        cfg
    };

    // Create the streaming scanline reader
    let mut reader = if needs_plane_reads(&descriptor) {
        cfg.for_plane_reads().scanline_reader(data_ref)?
    } else {
        cfg.scanline_reader(data_ref)?
    };

    let width = reader.width() as usize;
    let height = reader.height() as usize;
    let mut descriptor = descriptor;
    let mcu_height = reader.luma_rows_per_mcu();

    let ch_type = descriptor.layout.sample;
    let ch_layout = descriptor.layout.order;

    // read_rows_rgba_f32 always outputs 4 channels, so if caller requested
    // RGBF32 we must upgrade to RGBAF32 to match the actual output layout.
    if ch_type == SampleType::F32 && ch_layout == ChannelOrder::Rgb {
        descriptor = ZenPixelFormat::new(
            zenpixels::SampleLayout::RGBAF32,
            descriptor.color.with_alpha(zenpixels::AlphaMode::Straight),
        );
    }

    let row_bytes = width * descriptor.layout.bytes_per_pixel();
    if !begun {
        sink.begin(width as u32, height as u32, descriptor.clone())
            .map_err(wrap)?;
    }

    // Strip size: for buffered-mode readers the entire image is already
    // decoded; ask the sink for one big buffer so we make O(1) callbacks
    // instead of O(height/mcu_height). For streaming readers we must stick
    // to one MCU row per call (that's the unit the reader produces).
    let strip_rows = if reader.is_buffered() {
        height
    } else {
        mcu_height
    }
    .min(sink.max_buffer_rows().get() as usize);
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
            .provide_next_buffer(y, batch_max as u32, width as u32, descriptor.clone())
            .map_err(wrap)?;
        let dst_stride = dst.stride();
        let dst_bytes = dst.as_strided_bytes_mut();

        let count = match (ch_type, ch_layout) {
            (SampleType::U8, ChannelOrder::Gray) => {
                let out = ImgRefMut::new_stride(dst_bytes, row_bytes, batch_max, dst_stride);
                reader.read_rows_gray8(out)?
            }
            (SampleType::U8, ChannelOrder::Rgb) => {
                let out = ImgRefMut::new_stride(dst_bytes, row_bytes, batch_max, dst_stride);
                reader.read_rows_rgb8(out)?
            }
            (SampleType::U8, ChannelOrder::Rgba | ChannelOrder::Rgbx) => {
                let out = ImgRefMut::new_stride(dst_bytes, row_bytes, batch_max, dst_stride);
                if descriptor.layout.order.has_padding_slot() {
                    reader.read_rows_rgbx8(out)?
                } else {
                    reader.read_rows_rgba8(out)?
                }
            }
            (SampleType::U8, ChannelOrder::Bgra | ChannelOrder::Bgrx) => {
                let out = ImgRefMut::new_stride(dst_bytes, row_bytes, batch_max, dst_stride);
                if descriptor.layout.order.has_padding_slot() {
                    reader.read_rows_bgrx8(out)?
                } else {
                    reader.read_rows_bgra8(out)?
                }
            }
            (SampleType::F32, ChannelOrder::Gray) => {
                let float_slice: &mut [f32] = bytemuck::cast_slice_mut(dst_bytes);
                let float_stride = dst_stride / 4;
                let f_out =
                    ImgRefMut::new_stride(float_slice, row_bytes / 4, batch_max, float_stride);
                reader.read_rows_gray_f32(f_out)?
            }
            (SampleType::F32, ChannelOrder::Rgb | ChannelOrder::Rgba) => {
                // read_rows_rgba_f32 always writes 4 f32 channels; descriptor
                // was already upgraded to RGBAF32 above.
                let float_slice: &mut [f32] = bytemuck::cast_slice_mut(dst_bytes);
                let float_stride = dst_stride / 4;
                let f_out =
                    ImgRefMut::new_stride(float_slice, row_bytes / 4, batch_max, float_stride);
                reader.read_rows_rgba_f32(f_out)?
            }
            (SampleType::U16, ChannelOrder::Gray | ChannelOrder::Rgb | ChannelOrder::Rgba) => {
                read_canonical_u16_rows(
                    &mut reader,
                    dst_bytes,
                    dst_stride,
                    batch_max,
                    ch_layout,
                    header.precision,
                )?
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
    descriptor: &ZenPixelFormat,
    num_components: u8,
) -> Option<PixelFormat> {
    use zenpixels::{ChannelOrder, SampleType};
    let ch = descriptor.layout.sample;
    let layout = descriptor.layout.order;
    let is_gray_src = num_components == 1;
    match (ch, layout) {
        (SampleType::U8, ChannelOrder::Gray) if is_gray_src => Some(PixelFormat::Gray),
        (SampleType::U8, ChannelOrder::Rgb) if !is_gray_src => Some(PixelFormat::Rgb),
        (SampleType::U8, ChannelOrder::Rgba | ChannelOrder::Rgbx) if !is_gray_src => {
            Some(PixelFormat::Rgba)
        }
        (SampleType::U8, ChannelOrder::Bgra | ChannelOrder::Bgrx) if !is_gray_src => {
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
    descriptor: ZenPixelFormat,
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

    let mut dst = sink
        .provide_next_buffer(0, h, w, descriptor.clone())
        .map_err(wrap)?;
    let dst_stride = dst.stride();
    let bpp = format.bytes_per_pixel();
    let row_bytes = w as usize * bpp;

    let decode_into_result = if dst_stride == row_bytes {
        let dst_bytes = dst.as_strided_bytes_mut();
        if let Some(ref stop) = job.stop {
            cfg.decode_streaming_into(data_ref, format, dst_bytes, stop, header.num_components)
        } else {
            cfg.decode_streaming_into(
                data_ref,
                format,
                dst_bytes,
                &Unstoppable,
                header.num_components,
            )
        }
    } else {
        let total = row_bytes * h as usize;
        let mut contiguous =
            crate::foundation::alloc::try_alloc_vec(total, "strided JPEG sink output")?;
        let streaming_result = if let Some(ref stop) = job.stop {
            cfg.decode_streaming_into(
                data_ref,
                format,
                &mut contiguous,
                stop,
                header.num_components,
            )
        } else {
            cfg.decode_streaming_into(
                data_ref,
                format,
                &mut contiguous,
                &Unstoppable,
                header.num_components,
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
    let descriptor = pixels.format().clone();
    let w = pixels.width();
    let h = pixels.rows();

    sink.begin(w, h, descriptor.clone()).map_err(wrap)?;

    let cap = sink.max_buffer_rows().get();
    let mut y = 0;
    while y < h {
        let rows = cap.min(h - y);
        let mut dst = sink
            .provide_next_buffer(y, rows, w, descriptor.clone())
            .map_err(wrap)?;
        for row in 0..rows {
            dst.row_mut(row).copy_from_slice(pixels.row(y + row));
        }
        y += rows;
    }
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
fn metadata_preserve(
    policy: Option<&zencodec::decode::DecodePolicy>,
) -> crate::decode::PreserveConfig {
    let mut preserve = crate::decode::PreserveConfig::all();
    if let Some(policy) = policy {
        if policy.allow_exif == Some(false) {
            preserve = preserve.exif(false);
        }
        if policy.allow_xmp == Some(false) {
            preserve = preserve.xmp(false);
        }
        if policy.allow_icc == Some(false) {
            preserve = preserve.icc(crate::decode::IccPreserve::None);
        }
    }
    preserve
}

fn read_job_info(
    config: &crate::decode::DecodeConfig,
    data: &[u8],
    policy: Option<&zencodec::decode::DecodePolicy>,
) -> Result<crate::decode::JpegInfo, Error> {
    config.read_info_preserving(data, &metadata_preserve(policy))
}

fn check_progressive_policy(
    header: &crate::decode::JpegInfo,
    policy: Option<&zencodec::decode::DecodePolicy>,
) -> Result<(), Error> {
    let progressive = matches!(
        header.mode,
        crate::types::JpegMode::Progressive | crate::types::JpegMode::ArithmeticProgressive
    );
    if progressive && policy.is_some_and(|p| !p.resolve_progressive(true)) {
        return Err(Error::policy_rejected(
            "progressive JPEG rejected by decode policy",
        ));
    }
    Ok(())
}

fn build_decode_config(
    inner: &crate::decode::DecodeConfig,
    limits: &ResourceLimits,
    crop_hint: Option<(u32, u32, u32, u32)>,
    orientation: zencodec::OrientationHint,
    policy: Option<&zencodec::decode::DecodePolicy>,
) -> crate::decode::DecodeConfig {
    let mut cfg = inner.clone().preserve(metadata_preserve(policy));
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
    }

    cfg
}

/// A decode exit for a preference, if this frame shape has one.
fn supported_exit(desc: &ZenPixelFormat, is_gray: bool) -> Option<ZenPixelFormat> {
    Some(match (is_gray, desc.layout.sample, desc.layout.order) {
        (true, SampleType::U8, ChannelOrder::Gray) => ZenPixelFormat::GRAY8_SRGB,
        (true, SampleType::U16, ChannelOrder::Gray) => ZenPixelFormat::GRAY16_SRGB,
        (true, SampleType::F32, ChannelOrder::Gray) => ZenPixelFormat::GRAYF32_LINEAR,
        (false, SampleType::U8, ChannelOrder::Rgb) => ZenPixelFormat::RGB8_SRGB,
        (false, SampleType::U8, ChannelOrder::Rgbx) => ZenPixelFormat::RGBX8_SRGB,
        (false, SampleType::U8, ChannelOrder::Rgba) => ZenPixelFormat::RGBA8_SRGB,
        (false, SampleType::U8, ChannelOrder::Bgrx) => ZenPixelFormat::BGRX8_SRGB,
        (false, SampleType::U8, ChannelOrder::Bgra) => ZenPixelFormat::BGRA8_SRGB,
        (false, SampleType::U16, ChannelOrder::Rgb) => ZenPixelFormat::RGB16_SRGB,
        (false, SampleType::U16, ChannelOrder::Rgba) => ZenPixelFormat::RGBA16_SRGB,
        (false, SampleType::F32, ChannelOrder::Rgb) => ZenPixelFormat::RGBF32_LINEAR,
        (false, SampleType::F32, ChannelOrder::Rgba) => ZenPixelFormat::RGBAF32_LINEAR,
        _ => return None,
    })
}

/// Select the appropriate pixel descriptor for decode output.
///
/// The first supported preference wins. A frame above eight bits first looks for
/// a preference that holds its samples, so a list offering U16 or F32 anywhere
/// doesn't narrow; a list offering only U8 asked for U8. Without a supported
/// preference the frame's own depth decides, so it leaves as canonical U16.
pub(super) fn select_decode_descriptor(
    preferred: &[ZenPixelFormat],
    num_components: u8,
    precision: u8,
) -> ZenPixelFormat {
    let is_gray = num_components == 1;
    let mut exits = preferred
        .iter()
        .filter_map(|desc| supported_exit(desc, is_gray));
    if precision > 8
        && let Some(deep) = exits.clone().find(|d| d.layout.sample != SampleType::U8)
    {
        return deep;
    }
    if let Some(first) = exits.next() {
        return first;
    }

    match (is_gray, precision > 8) {
        (true, false) => ZenPixelFormat::GRAY8_SRGB,
        (false, false) => ZenPixelFormat::RGB8_SRGB,
        (true, true) => ZenPixelFormat::GRAY16_SRGB,
        (false, true) => ZenPixelFormat::RGB16_SRGB,
    }
}

/// Whether a U16 exit reads the frame's native codes.
///
/// An eight-bit frame reads the U8 rows every other exit shows and widens them,
/// so asking for U16 cannot change which image comes out. Lossless frames below
/// eight bits have no exact route through U8, so they read codes too.
pub(super) fn u16_uses_native_codes(descriptor: &ZenPixelFormat, precision: u8) -> bool {
    descriptor.layout.sample == zenpixels::SampleType::U16 && precision != 8
}

/// Depth evidence for canonical U16 samples widened from `precision`-bit codes.
pub(super) fn u16_exact_depth(precision: u8) -> Result<ExactDepth, Error> {
    ExactDepth::new(SampleType::U16, SignalRange::Full, precision, 16)
        .map_err(|_| Error::internal("JPEG sample precision has no U16 depth statement"))
}

/// Exact full-range widening of `precision`-bit codes to canonical U16.
fn u16_code_map(precision: u8) -> Result<impl Fn(u16) -> u16, Error> {
    if !(1..=16).contains(&precision) {
        return Err(Error::internal(
            "JPEG sample precision has no exact U16 widening",
        ));
    }
    let max = (1u32 << precision) - 1;
    Ok(move |sample| ((u32::from(sample).min(max) * u32::from(u16::MAX) + max / 2) / max) as u16)
}

/// Read up to `rows` rows of canonical U16 samples into strided bytes.
///
/// `output` is aligned for U16 and each row holds `width × channels` samples. The
/// reader must be configured as [`u16_uses_native_codes`] says. Returns the rows
/// written. Row padding is not touched.
pub(super) fn read_canonical_u16_rows(
    reader: &mut crate::decode::ScanlineReader<'_>,
    output: &mut [u8],
    stride: usize,
    rows: usize,
    layout: ChannelOrder,
    precision: u8,
) -> Result<usize, Error> {
    let width = reader.width() as usize;
    let channels = layout.channels();
    let row_bytes = width * channels * 2;
    let map = u16_code_map(precision)?;
    for y in 0..rows {
        let row = &mut output[y * stride..y * stride + row_bytes];
        if precision == 8 {
            let bytes = imgref::ImgRefMut::new(row, row_bytes, 1);
            let count = match layout {
                ChannelOrder::Gray => reader.read_rows_gray8(bytes)?,
                ChannelOrder::Rgb => reader.read_rows_rgb8(bytes)?,
                _ => reader.read_rows_rgba8(bytes)?,
            };
            if count == 0 {
                return Ok(y);
            }
            // Backwards: each sample's bytes land at or past its unread position.
            for x in (0..width * channels).rev() {
                let word = map(u16::from(row[x]));
                row[x * 2..x * 2 + 2].copy_from_slice(&word.to_ne_bytes());
            }
            continue;
        }
        let count = if layout == ChannelOrder::Gray {
            let samples: &mut [u16] = bytemuck::cast_slice_mut(row);
            reader.read_rows_gray16(imgref::ImgRefMut::new(samples, width, 1))?
        } else {
            let pixels: &mut [rgb::RGB<u16>] = bytemuck::cast_slice_mut(&mut row[..width * 6]);
            reader.read_rows_rgb16(imgref::ImgRefMut::new(pixels, width, 1))?
        };
        if count == 0 {
            return Ok(y);
        }
        let samples: &mut [u16] = bytemuck::cast_slice_mut(row);
        let coded = if layout == ChannelOrder::Gray {
            width
        } else {
            width * 3
        };
        for sample in &mut samples[..coded] {
            *sample = map(*sample);
        }
        if layout == ChannelOrder::Rgba {
            // Backwards: each pixel lands at or past its unread position.
            for x in (0..width).rev() {
                samples.copy_within(x * 3..x * 3 + 3, x * 4);
                samples[x * 4 + 3] = u16::MAX;
            }
        }
    }
    Ok(rows)
}

/// Canonical U16 samples in storage that can move into the returned buffer.
fn canonical_u16_samples(
    samples: impl ExactSizeIterator<Item = u16>,
    precision: u8,
    what: &'static str,
) -> Result<Vec<u16>, Error> {
    let map = u16_code_map(precision)?;
    let mut output = crate::foundation::alloc::try_alloc_vec(samples.len(), what)?;
    for (dst, src) in output.iter_mut().zip(samples) {
        *dst = map(src);
    }
    Ok(output)
}

// ── Decoder ─────────────────────────────────────────────────────────────────

/// One-shot JPEG decoder implementing [`zencodec::decode::Decode`].
pub struct JpegDecoder<'a> {
    config: JpegDecoderConfig,
    /// Cancellation (and progress) control, polled as a `Stop`.
    stop: Option<zencodec::SharedPulse>,
    limits: ResourceLimits,
    crop_hint: Option<(u32, u32, u32, u32)>,
    orientation: zencodec::OrientationHint,
    policy: Option<zencodec::decode::DecodePolicy>,
    gain_map_render: zencodec::GainMapRender,
    data: Cow<'a, [u8]>,
    preferred: Vec<ZenPixelFormat>,
}

impl zencodec::decode::Decode for JpegDecoder<'_> {
    type Error = At<CodecError>;

    fn decode(self) -> Result<DecodeOutput, Self::Error> {
        {
            use crate::decode::OutputTarget;
            use crate::types::PixelFormat;
            use zenpixels::SampleType;

            // Ultra HDR rendition intent. ReconstructHdr takes a dedicated
            // path when resolved metadata identifies a gain map; an image
            // without one decodes below as the (complete) base image.
            // Components decorates the normal decode at the end. Unknown
            // future modes are refused — never silently mis-rendered.
            match self.gain_map_render {
                zencodec::GainMapRender::BaseOnly | zencodec::GainMapRender::Components => {}
                zencodec::GainMapRender::ReconstructHdr { target_headroom } => {
                    #[cfg(feature = "ultrahdr")]
                    {
                        if let Some(output) = self.decode_reconstruct_hdr(target_headroom)? {
                            return Ok(output);
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

            let limits = self.limits;
            let mut cfg = build_decode_config(
                &self.config.inner,
                &limits,
                self.crop_hint,
                self.orientation,
                self.policy.as_ref(),
            );

            // Single header parse for all pre-decode checks: dimension limits,
            // progressive policy, CMYK detection, and output format hinting.
            // Previously three separate read_info() calls.
            let header = read_job_info(&cfg, &data, self.policy.as_ref())?;

            // Choose precision from the first supported preference, just like
            // layout selection. A later float alternative must not override U8.
            let selected =
                select_decode_descriptor(preferred, header.num_components, header.precision);
            let wants_f32 = selected.layout.sample == SampleType::F32;
            if wants_f32 {
                cfg = cfg.output_target(OutputTarget::LinearF32);
            }

            // Dimension limits. `check_dimensions` yields a `LimitExceeded`
            // (foreign — no envelope bridge), so map through the native `Error`.
            if limits.max_width.is_some() || limits.max_height.is_some() {
                limits
                    .check_dimensions(header.dimensions.width, header.dimensions.height)
                    .map_err(Error::from)?;
            }

            check_progressive_policy(&header, self.policy.as_ref())?;

            // Passthrough CMYK handling
            let is_raw_cmyk = if matches!(self.config.cmyk_handling, CmykHandling::Passthrough)
                && header.num_components == 4
            {
                cfg = cfg.output_format(PixelFormat::Cmyk);
                true
            } else {
                false
            };
            // CMYK has no U16 descriptor. A deeper frame leaves as normalized F32,
            // which holds every code, unless the caller explicitly requested inverted CMYK8.
            let raw_cmyk_f32 = is_raw_cmyk
                && header.precision > 8
                && !wants_f32
                && !preferred
                    .iter()
                    .any(|d| d.layout == zenpixels::SampleLayout::CMYK8_INVERTED);
            if raw_cmyk_f32 {
                cfg = cfg.output_target(OutputTarget::SrgbF32);
            }

            let wants_u16 = !is_raw_cmyk && selected.layout.sample == SampleType::U16;
            if wants_u16 {
                cfg = cfg.output_format(match selected.layout.order {
                    zenpixels::ChannelOrder::Gray => PixelFormat::Gray,
                    zenpixels::ChannelOrder::Rgba => PixelFormat::Rgba,
                    _ => PixelFormat::Rgb,
                });
                if u16_uses_native_codes(&selected, header.precision) {
                    cfg = cfg.output_target(OutputTarget::Native16);
                }
            }

            // Hint the internal decoder to produce BGRA/RGBA/etc. directly in
            // the streaming path, eliminating the post-decode full-buffer swizzle.
            if !wants_f32 && !is_raw_cmyk {
                let descriptor = decode_descriptor(
                    preferred,
                    &header,
                    self.config.inner.correct_color.as_ref(),
                )?;
                if let Some(pf) = direct_path_pixel_format(&descriptor, header.num_components) {
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

            // Extract metadata. Source precision comes from the SOF header:
            // 8 or 12 for DCT, 2-16 for lossless. Note this is the SOURCE depth:
            // U16 and f32 output buffers still carry only `bit_depth` significant
            // bits (#146).
            let mut info = ImageInfo::new(w, h, ImageFormat::Jpeg)
                .with_bit_depth(header.precision)
                .with_channel_count(header.num_components);
            if let Some(extras) = result.extras() {
                info = populate_info_from_jpeg_extras(info, extras, self.orientation);
            }

            let jpeg_extras = result.take_extras();

            // Stored ICC and current pixel interpretation are separate when a
            // JPEG transform has already changed the samples.
            let describe = |layout| {
                let base = zencodec::helpers::codec_spec(layout).with_transfer(
                    if layout.sample == SampleType::F32 {
                        zenpixels::Transfer::Linear
                    } else {
                        zenpixels::Transfer::Srgb
                    },
                );
                super::info::decoded_pixel_spec(
                    base,
                    &info.source_color,
                    header.is_xyb,
                    self.config.inner.correct_color.as_ref(),
                )
            };
            let what = "JPEG output pixel buffer";
            let buffer_error = crate::color::icc::pixel_buffer_error(0, what);
            let buf = if wants_f32 {
                let mut samples = result.into_pixels_f32().unwrap_or_default();
                let count = (w as usize) * (h as usize);
                let layout = if format == PixelFormat::Gray {
                    zenpixels::SampleLayout::GRAYF32
                } else if samples.len() == count * 3 && selected.layout.order == ChannelOrder::Rgba
                {
                    samples.try_reserve_exact(count).map_err(|_| {
                        Error::allocation_failed(
                            count.saturating_mul(4),
                            "expanding RGB float to RGBA",
                        )
                    })?;
                    samples.resize(count * 4, 1.0);
                    for x in (0..count).rev() {
                        samples.copy_within(x * 3..x * 3 + 3, x * 4);
                        samples[x * 4 + 3] = 1.0;
                    }
                    zenpixels::SampleLayout::RGBAF32
                } else if samples.len() == count * 3 {
                    zenpixels::SampleLayout::RGBF32
                } else {
                    zenpixels::SampleLayout::RGBAF32
                };
                PixelBuf::from_samples(samples, w, h, describe(layout)?).map_err(buffer_error)?
            } else if raw_cmyk_f32 || is_raw_cmyk {
                let sample = if raw_cmyk_f32 {
                    SampleType::F32
                } else {
                    SampleType::U8
                };
                let layout = zenpixels::SampleLayout::new(sample, ChannelOrder::CmykInverted);
                let desc = describe(layout)?;
                info.has_alpha = false;
                info.source_color.channel_count = Some(4);
                if raw_cmyk_f32 {
                    PixelBuf::from_samples(result.into_pixels_f32().unwrap_or_default(), w, h, desc)
                        .map_err(buffer_error)?
                } else {
                    PixelBuf::from_bytes(result.into_pixels_u8().unwrap_or_default(), w, h, desc)
                        .map_err(buffer_error)?
                }
            } else if wants_u16 {
                let desc = describe(selected.layout)?;
                let bytes = match result.pixels_u16() {
                    Some(codes) => canonical_u16_samples(
                        codes.iter().copied(),
                        header.precision,
                        "U16 output",
                    )?,
                    None => canonical_u16_samples(
                        result
                            .pixels_u8()
                            .ok_or_else(|| Error::internal("decoder produced no integer pixels"))?
                            .iter()
                            .map(|&sample| u16::from(sample)),
                        header.precision,
                        "U16 output",
                    )?,
                };
                let mut buffer = PixelBuf::from_samples(bytes, w, h, desc).map_err(buffer_error)?;
                let mut evidence = buffer.facts();
                evidence.depth = Some(u16_exact_depth(header.precision)?);
                buffer.assume_facts(evidence);
                buffer
            } else {
                let filler = selected.layout.order.has_padding_slot();
                let pf = match format {
                    PixelFormat::Gray => zenpixels::SampleLayout::GRAY8,
                    PixelFormat::Rgba if filler => zenpixels::SampleLayout::RGBX8,
                    PixelFormat::Bgra if filler => zenpixels::SampleLayout::BGRX8,
                    PixelFormat::Rgba => zenpixels::SampleLayout::RGBA8,
                    PixelFormat::Bgra => zenpixels::SampleLayout::BGRA8,
                    _ => zenpixels::SampleLayout::RGB8,
                };
                PixelBuf::from_bytes(
                    result.into_pixels_u8().unwrap_or_default(),
                    w,
                    h,
                    describe(pf)?,
                )
                .map_err(buffer_error)?
            };

            let buf = buf.with_origin(info.source_color.color_origin());
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
                if let Some(dgm) =
                    decode_gain_map_components(extras, base_transform, cfg.preserve.xmp)?
                {
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
            #[cfg(feature = "metadata")]
            if let Ok(probe) = crate::detect::probe(&data) {
                output = output.with_source_encoding_details(probe);
            }

            // Check output size limits
            let output_bytes = output.pixels().rows() as u64
                * output.pixels().width() as u64
                * output.pixels().format().layout.bytes_per_pixel() as u64;
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
    fn decode_reconstruct_hdr(
        &self,
        target_headroom: Option<f32>,
    ) -> Result<Option<DecodeOutput>, Error> {
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
        // The transform the requested decode would bake (EXIF when
        // auto-orienting, composed with any user decode_transform). The
        // reconstruction itself happens in stored space — upright base +
        // stored gain map — and this permutation is baked into the finished
        // HDR buffer at the end, so orientation is a pure pixel permutation
        // of one canonical reconstruction (byte-identical to `Preserve` plus
        // an external bake, the same construction as #149).
        let base_transform = cfg.compute_effective_transform_from_data(&self.data);
        // Gain-map parameters describe the stored base color. Perform requested
        // output gamut conversion only after applying the gain map.
        cfg = cfg.auto_orient(false);
        cfg.correct_color = None;
        cfg.decode_transform = None;
        // apply_gainmap consumes RGBA8 SDR input.
        cfg = cfg.output_format(crate::types::PixelFormat::Rgba);

        let header = read_job_info(&cfg, &self.data, self.policy.as_ref())?;
        check_progressive_policy(&header, self.policy.as_ref())?;
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
        let resolved = crate::ultrahdr::metadata_with_xmp(&extras, cfg.preserve.xmp);
        if resolved.is_none() && !cfg.preserve.xmp && extras.gainmap().is_some() {
            return Err(Error::policy_rejected(
                "gain-map reconstruction needs metadata denied by decode policy",
            ));
        }
        if resolved.is_none() && !extras.is_ultrahdr() {
            return Ok(None);
        }
        let (metadata, _) = resolved.ok_or_else(|| {
            Error::unsupported_feature(
                "ReconstructHdr requested but no gain-map parameters are readable",
            )
        })??;
        let gainmap = crate::ultrahdr::decode_gainmap_with_xmp(&extras, cfg.preserve.xmp)
            .ok_or_else(|| {
                Error::unsupported_feature(
                    "ReconstructHdr requested but the MPF has no gain-map image",
                )
            })??;

        let pixels_u8 = result
            .into_pixels_u8()
            .ok_or_else(|| Error::internal("decoder produced no u8 pixels"))?;
        // Denying extraction cannot turn tagged pixels into assumed sRGB.
        if cfg.preserve.icc == crate::decode::IccPreserve::None
            && crate::container::marker::MarkerIter::for_decoder_metadata(&self.data).any(|span| {
                span.kind == crate::container::marker::MarkerKind::App(2)
                    && span.payload.starts_with(b"ICC_PROFILE\0")
            })
        {
            return Err(Error::policy_rejected(
                "HDR reconstruction needs ICC metadata denied by decode policy",
            ));
        }
        let (primaries, transfer) = crate::ultrahdr::base_color(extras.icc_profile())?;
        let descriptor = zenpixels::PixelFormat::RGBA8_SRGB
            .with_primaries(primaries)
            .with_transfer(transfer);
        let sdr = PixelBuf::from_bytes(pixels_u8, w, h, descriptor)
            .map_err(|_| Error::internal("pixel buffer creation failed"))?;

        let sdr = crate::ultrahdr::prepare_base_pixels(sdr, extras.icc_profile())?;

        // Output form: honor an f16 preference; default linear f32 RGBA.
        let wants_f16 = self
            .preferred
            .iter()
            .any(|d| d.layout.sample == zenpixels::SampleType::F16);
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
        let hdr = if let Some(target) = self.config.inner.correct_color {
            use crate::color::icc::TargetColorSpace;
            let primaries = match target {
                TargetColorSpace::Srgb => zenpixels::Primaries::Bt709,
                TargetColorSpace::DisplayP3 => zenpixels::Primaries::DisplayP3,
                TargetColorSpace::Rec2020 => zenpixels::Primaries::Bt2020,
            };
            let descriptor = hdr.format().clone().with_primaries(primaries);
            if &descriptor == hdr.format() {
                hdr
            } else {
                let converter = zenpixels_convert::RowPlan::build(
                    hdr.format().clone(),
                    descriptor.clone(),
                    &zenpixels_convert::ConvertPolicy::default()
                        .with_gamut(zenpixels_convert::GamutPolicy::PreserveExtended),
                    None,
                )
                .map_err(crate::ultrahdr::ultrahdr_to_zenjpeg_error)?;
                let mut output = PixelBuf::new(w, h, descriptor)
                    .map_err(|_| Error::internal("HDR color output allocation failed"))?;
                let src = hdr.as_view();
                let mut dst = output.as_view_mut();
                for row in 0..h {
                    stop.check()?;
                    converter
                        .convert_row(src.row(row), dst.row_mut(row), w, row)
                        .map_err(crate::ultrahdr::ultrahdr_to_zenjpeg_error)?;
                }
                output
            }
        } else {
            hdr
        };
        // Bake orientation only after reconstructing in stored coordinates.
        let hdr = orient_reconstructed_hdr(hdr, lossless_to_orientation(base_transform))?;

        let mut info = ImageInfo::new(hdr.width(), hdr.height(), ImageFormat::Jpeg);
        info = populate_info_from_jpeg_extras(info, &extras, self.orientation);

        // Content light uses the reconstructed native-gamut MaxRGB.
        // Mastering peak describes the encoded alternate capacity.
        let capacity_nits = SDR_WHITE_NITS * capacity_max;
        info.source_color.content_light_level = Some(reconstructed_max_rgb(&hdr.as_view())?);
        let zenpixels::ColorSpace::Rgb { primaries, .. } = hdr.format().color else {
            return Err(Error::unsupported_feature(
                "unknown reconstructed HDR primaries",
            ));
        };
        let (red, green, blue) = primaries
            .chromaticity()
            .ok_or_else(|| Error::unsupported_feature("unknown reconstructed HDR primaries"))?;
        info.source_color.mastering_display = Some(zenpixels::hdr::MasteringDisplay::new(
            [[red.0, red.1], [green.0, green.1], [blue.0, blue.1]],
            [0.3127, 0.3290],
            capacity_nits,
            0.005,
        ));

        let hdr =
            hdr.with_origin(info.source_color.color_origin())
                .with_hdr(zenpixels::HdrMetadata {
                    cll: info.source_color.content_light_level,
                    mastering: info.source_color.mastering_display,
                });
        let mut output = DecodeOutput::new(hdr, info).with_extras(extras);
        #[cfg(feature = "metadata")]
        if let Ok(probe) = crate::detect::probe(&self.data) {
            output = output.with_source_encoding_details(probe);
        }

        let output_bytes = output.pixels().rows() as u64
            * output.pixels().width() as u64
            * output.pixels().format().layout.bytes_per_pixel() as u64;
        limits.check_output_size(output_bytes).map_err(|_| {
            Error::resource_limit_exceeded(
                zencodec::LimitKind::OutputSize,
                output_bytes,
                limits.max_output_bytes.unwrap_or(0),
            )
        })?;

        Ok(Some(output))
    }
}

/// Measure the existing linear RGBA rendition in its own gamut.
#[cfg(feature = "ultrahdr")]
fn reconstructed_max_rgb(
    view: &zenpixels::PixelView<'_>,
) -> Result<zenpixels::ContentLightLevel, Error> {
    let mut max = 0.0f32;
    let mut sum = 0.0f64;
    let mut measure = |r: f32, g: f32, b: f32| {
        let v = r.max(g).max(b).max(0.0);
        max = max.max(v);
        sum += f64::from(v);
    };
    let buffer_error = || crate::color::icc::pixel_buffer_error(0, "JPEG HDR light level");
    for y in 0..view.rows() {
        if view.format().layout.sample == SampleType::F16 {
            let row = view.row_as::<u16>(y).map_err(buffer_error())?;
            let to_f32 = |bits| half::f16::from_bits(bits).to_f32();
            for &[r, g, b, _] in row.as_chunks::<4>().0 {
                measure(to_f32(r), to_f32(g), to_f32(b));
            }
        } else {
            let row = view.row_as::<f32>(y).map_err(buffer_error())?;
            for &[r, g, b, _] in row.as_chunks::<4>().0 {
                measure(r, g, b);
            }
        }
    }
    let scale = f64::from(zenpixels::DiffuseWhite::BT2408.nits());
    let len = f64::from(view.width()) * f64::from(view.rows());
    Ok(zenpixels::ContentLightLevel::new(
        (f64::from(max) * scale + 0.5) as u16,
        (sum / len * scale + 0.5) as u16,
    ))
}

/// Permute the current packed HDR raster, then re-describe the same storage.
#[cfg(feature = "ultrahdr")]
fn orient_reconstructed_hdr(
    mut image: PixelBuf,
    orientation: zencodec::Orientation,
) -> Result<PixelBuf, Error> {
    if orientation == zencodec::Orientation::Identity {
        return Ok(image);
    }
    let (w, h) = (image.width(), image.height());
    let (ow, oh) = orientation.output_dimensions(w, h);
    let bpp = image.format().layout.bytes_per_pixel();
    let pixels = w as usize * h as usize;
    let mut view = image.as_view_mut();
    let Some(bytes) = view.as_contiguous_bytes_mut() else {
        return zenpixels_convert::apply_orientation(&image.as_view(), orientation)
            .map_err(crate::ultrahdr::ultrahdr_to_zenjpeg_error);
    };
    let next = |i: usize| {
        let (x, y) =
            orientation.forward_map((i % w as usize) as u32, (i / w as usize) as u32, w, h);
        y as usize * ow as usize + x as usize
    };
    let swap = |bytes: &mut [u8], a: usize, b: usize| {
        for j in 0..bpp {
            bytes.swap(a * bpp + j, b * bpp + j);
        }
    };
    if orientation.inverse() == orientation && (w == h || !orientation.swaps_axes()) {
        for i in 0..pixels {
            let other = next(i);
            if i < other {
                swap(bytes, i, other);
            }
        }
    } else {
        let mut seen: Vec<u8> =
            crate::foundation::alloc::try_alloc_vec(pixels.div_ceil(8), "JPEG HDR orientation")?;
        for start in 0..pixels {
            let mut i = start;
            while seen[i / 8] & (1 << (i % 8)) == 0 {
                seen[i / 8] |= 1 << (i % 8);
                i = next(i);
                if i == start {
                    break;
                }
                swap(bytes, start, i);
            }
        }
    }

    let mut parts = image.into_parts();
    parts.width = ow;
    parts.height = oh;
    parts.stride = ow as usize * bpp;
    let oriented = PixelBuf::from_parts(parts).map_err(crate::color::icc::pixel_buffer_error(
        pixels * bpp,
        "JPEG HDR orientation",
    ))?;
    Ok(oriented)
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
    allow_xmp: bool,
) -> Result<Option<zencodec::decode::DecodedGainMap>, Error> {
    let Some(metadata) = crate::ultrahdr::metadata_with_xmp(extras, allow_xmp) else {
        return Ok(None);
    };
    let (metadata, _) = metadata?;
    let Some(gainmap) = crate::ultrahdr::decode_gainmap_with_xmp(extras, allow_xmp) else {
        return Ok(None);
    };
    let gainmap = orient_gain_map(gainmap?, base_transform);

    let layout = if gainmap.channels == 3 {
        zenpixels::SampleLayout::RGB8
    } else {
        zenpixels::SampleLayout::GRAY8
    };
    let desc = ZenPixelFormat::new(
        layout,
        zenpixels::ColorSpace::Untagged {
            channels: gainmap.channels,
            alpha: zenpixels::AlphaMode::None,
        },
    );
    let pixels = PixelBuf::from_bytes(gainmap.data, gainmap.width, gainmap.height, desc)
        .map_err(|_| Error::internal("gain-map pixel buffer creation failed"))?;
    let gm_info =
        zencodec::GainMapInfo::new(metadata, gainmap.width, gainmap.height, gainmap.channels);
    zencodec::decode::DecodedGainMap::try_new(pixels, gm_info)
        .map(Some)
        .map_err(|err| Error::decode_error(alloc::format!("gain-map components: {err}")))
}
