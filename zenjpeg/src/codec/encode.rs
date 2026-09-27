//! zencodec encode-side impls: `JpegEncoderConfig` / `JpegEncodeJob` / `JpegEncoder`.

use alloc::vec::Vec;

use whereat::At;
use zencodec::encode::{EncodeCapabilities, EncodeOutput};
use zencodec::{CodecError, ImageFormat, Metadata, ResourceLimits, UnsupportedOperation};
use zenpixels::{PixelDescriptor, PixelSlice, PixelSliceMut};

use crate::encode::encoder_config::EncoderConfig;
use crate::encode::encoder_types::{ChromaSubsampling, PixelLayout, Quality};
use crate::encode::exif::Exif;
use crate::error::{Error, ErrorKind};

// ============================================================================
// Encode side: EncoderConfig → EncodeJob → Encoder
// ============================================================================

/// JPEG encode capabilities.
static JPEG_ENCODE_CAPS: EncodeCapabilities = EncodeCapabilities::new()
    .with_icc(true)
    .with_exif(true)
    .with_xmp(true)
    // JPEG has no CICP carrier: color is signaled only via an embedded APP2
    // ICC profile. Declare this explicitly so `resolve_color_emit` knows a
    // CICP-only source must synthesize an ICC rather than emit CICP. The two
    // carrier flags (`cicp_is_valid_carrier` / `cicp_safe_sole_carrier`) stay
    // at their `false` defaults for the same reason.
    .with_cicp(false)
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
    /// Uses [`ExpertConfig`](crate::encode::expert::ExpertConfig).
    #[must_use]
    pub fn from_preset(preset_name: &str, quality: f32) -> Option<Self> {
        use crate::encode::encoder_types::OptimizationPreset;
        use crate::encode::expert::ExpertConfig;

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
    ///
    /// Returns the shared [`At<CodecError>`] envelope (the same `type Error` the
    /// zencodec trait path uses), so the category + codec name survive type
    /// erasure; recover the native [`Error`] detail via
    /// [`CodecError::detail`](zencodec::CodecError::detail) when needed.
    pub fn encode(&self, pixels: PixelSlice<'_>) -> Result<EncodeOutput, At<CodecError>> {
        use zencodec::encode::{EncodeJob as _, Encoder as _, EncoderConfig as _};
        self.clone().job().encoder()?.encode(pixels)
    }

    /// Apply effort level, returning a modified config.
    ///
    /// Clamps `self.effort` into zenjpeg's real 0..=2 tiers at the point of
    /// use: `0` = `JpegliBaseline`, `2` = `HybridMaxCompression`, anything
    /// else (including the default `1` and any out-of-range value accepted
    /// by [`with_generic_effort`](Self::with_generic_effort)) = `HybridProgressive`.
    fn effective_config(&self) -> EncoderConfig {
        use crate::encode::encoder_types::OptimizationPreset;
        let preset = match self.effort.clamp(0, 2) {
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
    type Error = At<CodecError>;
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
        // `generic_quality` is the codec-agnostic 0..100 dial. For zenjpeg it
        // means "target a zensim Profile A score of `quality`": A is the
        // canonical codec-target metric (`ZensimProfile::codec_target()`), so a
        // caller asking for generic quality 70 gets an encode whose achieved
        // zensim:A score lands at ~70 regardless of content. When the
        // zensim-targeting machinery is compiled in (`target-zq`), route to the
        // `Quality::Zq` closed loop (warm-started by the picker under
        // `__picker-research`); otherwise fall back to the CID22-calibrated
        // jpegli-native approximation of the same perceptual target.
        #[cfg(feature = "target-zq")]
        {
            self.quality = clamped;
            self.inner = self.inner.quality(Quality::Zq(clamped));
        }
        #[cfg(not(feature = "target-zq"))]
        {
            let q = calibrated_jpeg_quality(clamped);
            self.quality = q;
            self.inner = self.inner.quality(Quality::ApproxJpegli(q));
        }
        self
    }

    fn generic_quality(&self) -> Option<f32> {
        Some(self.generic_quality_input.unwrap_or(self.quality))
    }

    /// Honor a [`Fidelity`](zencodec::encode::Fidelity) target as natively as
    /// JPEG allows. zenjpeg has **no lossless codestream**, but it *does* honor
    /// all three lossy targets natively: a SSIMULACRA2 score, a butteraugli
    /// max-norm distance, and its own jpegli quality dial.
    ///
    /// - `Lossless` → JPEG cannot be exact; best-effort highest quality
    ///   (`ApproxJpegli(100)`). `resolved_target_fidelity` reports the lossy
    ///   quality, never `Lossless`.
    /// - `Lossy(CodecSpecificQuality(q))` → the raw jpegli quality dial
    ///   (`Quality::ApproxJpegli`), bypassing the generic calibration.
    /// - `Lossy(ApproxSsim2(s))` → native `Quality::ApproxSsim2` (single-pass
    ///   SSIM2-calibrated).
    /// - `Lossy(ApproxButteraugli(d))` → native `Quality::ApproxButteraugli`
    ///   (single-pass jpegli-distance encode).
    fn with_fidelity(mut self, fidelity: zencodec::encode::Fidelity) -> Self {
        use zencodec::encode::{Fidelity, LossyTarget};
        match fidelity {
            Fidelity::Lossless => {
                self.quality = 100.0;
                self.generic_quality_input = None;
                self.inner = self.inner.quality(Quality::ApproxJpegli(100.0));
            }
            Fidelity::Lossy(LossyTarget::CodecSpecificQuality(q)) => {
                self.quality = q;
                self.generic_quality_input = Some(q);
                self.inner = self.inner.quality(Quality::ApproxJpegli(q));
            }
            Fidelity::Lossy(LossyTarget::ApproxSsim2(s)) => {
                self.quality = s;
                self.generic_quality_input = Some(s);
                self.inner = self.inner.quality(Quality::ApproxSsim2(s));
            }
            Fidelity::Lossy(LossyTarget::ApproxButteraugli(d)) => {
                self.generic_quality_input = None;
                self.inner = self.inner.quality(Quality::ApproxButteraugli(d));
            }
            // `Fidelity` / `LossyTarget` are `#[non_exhaustive]`: a future lossy
            // target falls back to the jpegli quality dial at a sane default.
            _ => {
                self.quality = 85.0;
                self.generic_quality_input = None;
                self.inner = self.inner.quality(Quality::ApproxJpegli(85.0));
            }
        }
        self
    }

    /// Report the native target the inner config will actually encode to. JPEG
    /// is never lossless, so this always returns a `Lossy` fidelity — read back
    /// from the inner [`Quality`], so a metric target round-trips as itself.
    fn resolved_target_fidelity(&self) -> Option<zencodec::encode::Fidelity> {
        use zencodec::encode::Fidelity;
        Some(match self.inner.get_quality() {
            Quality::ApproxSsim2(s) => Fidelity::ssim2(s),
            Quality::ApproxButteraugli(d) => Fidelity::butteraugli(d),
            Quality::ApproxJpegli(q) => Fidelity::codec_quality(q),
            // Mozjpeg / Zq / future variants → report on the codec quality scale.
            other => Fidelity::codec_quality(other.to_internal()),
        })
    }

    /// Set the generic-effort accept signal.
    ///
    /// zenjpeg supports exactly three real effort tiers, mapped at
    /// point-of-use in [`effective_config`](Self::effective_config): `0` =
    /// `JpegliBaseline` (fastest), `1` = `HybridProgressive` (default), `2` =
    /// `HybridMaxCompression` (slowest/smallest). Out-of-tier values (e.g.
    /// `99`) are accepted and clamped into `HybridProgressive` at encode
    /// time, but the raw value passed here is stored verbatim and echoed
    /// back by [`generic_effort`](Self::generic_effort) — fleet
    /// accept-signal callers that set-then-get to confirm the config
    /// accepted their input must see their own value back, not a silently
    /// clamped one.
    fn with_generic_effort(mut self, effort: i32) -> Self {
        self.effort = effort;
        self
    }

    fn generic_effort(&self) -> Option<i32> {
        Some(self.effort)
    }

    fn estimate_encode_resources(
        &self,
        image: &zencodec::estimate::ImageCharacteristics,
        compute: &zencodec::estimate::ComputeEnvironment,
    ) -> zencodec::estimate::ResourceEstimate {
        use zencodec::estimate::ResourceEstimate;
        let e = crate::heuristics::estimate_encode(image.width(), image.height(), self.inner());
        // `heuristics::estimate_encode` models the encoder's working set (calibrated
        // as VmHWM marginal, i.e. above the held input). The zencodec convention
        // (`ResourceEstimate::conservative`) is total peak = input buffer (held by the
        // caller throughout the encode) + working, so add the caller's input buffer.
        let input = image.input_bytes();
        ResourceEstimate::new(e.peak_memory_bytes.saturating_add(input), e.time_ms as u64)
            .with_peak_max(e.peak_memory_bytes_max.saturating_add(input))
            .with_threading(crate::heuristics::encode_threading_info())
            .at_cores(compute.cores())
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
    type Error = At<CodecError>;
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

        // Map threading policy to parallel encoding config. We use the
        // `is_parallel()` accessor instead of matching individual variants
        // so that the code keeps working across zencodec 0.1.x releases
        // (e.g. the 0.1.19 bump deprecated `SingleThread`/`LimitOrSingle`/
        // `LimitOrAny`/`Balanced`/`Unlimited` in favor of
        // `Sequential`/`Parallel`).
        #[cfg(feature = "parallel")]
        {
            if self.limits.threading.is_parallel() {
                cfg = cfg.parallel(crate::encode::ParallelEncoding::Auto);
            }
            // Otherwise (Sequential): leave cfg.parallel as None.
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
        Err(ErrorKind::UnsupportedOperation(UnsupportedOperation::AnimationEncode).into())
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
    ///
    /// `channel_count` is the channel count of the pixels being encoded
    /// (1 = gray, 3 = RGB, 4 = RGBA); it lets the color-emit resolver
    /// suppress ICC synthesis for grayscale, where an RGB profile would
    /// recolor the image. Pass `None` when the count is not yet known.
    fn build_request(
        &self,
        channel_count: Option<u8>,
    ) -> Result<crate::encode::request::EncodeRequest<'_>, Error> {
        self.build_request_from(&self.effective_config, channel_count)
    }

    /// Build an EncodeRequest from a specific config + metadata, applying policy.
    ///
    /// The color carrier decision (which ICC bytes JPEG embeds, if any) is
    /// resolved through [`zencodec::resolve_color_emit`] under the job's
    /// [`ColorEmitPolicy`](zencodec::ColorEmitPolicy): JPEG's only color
    /// carrier is an APP2 ICC profile (no CICP carrier), so a CICP-only
    /// source synthesizes an ICC via zenpixels-convert's
    /// `synthesize_icc_for_cicp` instead of silently producing an untagged
    /// (sRGB-assumed) JPEG. EXIF/XMP pass through verbatim — the blessed
    /// `with_metadata_policy` path has already filtered them (sub-field EXIF
    /// retention + orientation-tag reconciliation) before the bytes get here.
    ///
    /// Errors when the plan needs an ICC synthesized for the source CICP and
    /// this build can't produce one (see the `cms` feature): JPEG has no CICP
    /// carrier, so encoding without the ICC would misrepresent the image.
    fn build_request_from<'b>(
        &'b self,
        config: &'b EncoderConfig,
        channel_count: Option<u8>,
    ) -> Result<crate::encode::request::EncodeRequest<'b>, Error> {
        let mut req = config.request();
        if let Some(ref meta) = self.metadata {
            let policy = self.policy.unwrap_or_default();
            if policy.resolve_icc(true) {
                let color_policy = policy.resolve_color(zencodec::ColorEmitPolicy::Balanced);
                let mut src = zencodec::SourceColor::default();
                if let Some(n) = channel_count {
                    src = src.with_channel_count(n);
                }
                if let Some(c) = meta.cicp {
                    src = src.with_cicp(c);
                }
                if let Some(ref icc) = meta.icc_profile {
                    src = src.with_icc_profile(icc.clone());
                }
                let plan = zencodec::resolve_color_emit(&src, &JPEG_ENCODE_CAPS, color_policy);
                match plan.icc {
                    zencodec::IccDisposition::KeepSource => {
                        if let Some(ref icc) = meta.icc_profile {
                            req = req.icc_profile(icc);
                        }
                    }
                    zencodec::IccDisposition::SynthesizeFrom(cicp) => {
                        use zenpixels_convert::icc_profiles::{
                            SynthesizedIcc, synthesize_icc_for_cicp,
                        };
                        // `Profile` → embed; `NotNeeded` (sRGB default) →
                        // nothing. Every other outcome is an ERROR, not a
                        // silent skip: JPEG has NO CICP carrier, so an
                        // embedded APP2 ICC is the ONLY way this color
                        // survives — emitting without it would misrepresent
                        // the image as sRGB. The `zencodec` feature carries
                        // zenpixels-convert's icc-db blob (full ITU-T H.273
                        // grid incl PQ/HLG, no moxcms), so the only
                        // unsynthesizable CICPs are outside the assigned
                        // H.273 grid (reserved / unassigned code points).
                        match synthesize_icc_for_cicp(cicp) {
                            SynthesizedIcc::Profile(bytes) => {
                                req = req.icc_profile_owned(bytes.into_owned());
                            }
                            SynthesizedIcc::NotNeeded => {}
                            outcome => {
                                return Err(Error::icc_error(alloc::format!(
                                    "this image's color (CICP primaries {} / transfer {}) \
                                     needs a synthesized ICC profile that cannot be \
                                     produced ({outcome:?}) — the CICP is outside the \
                                     assigned H.273 grid; supply an ICC profile in the \
                                     metadata or drop the CICP",
                                    cicp.color_primaries,
                                    cicp.transfer_characteristics
                                )));
                            }
                        }
                    }
                    zencodec::IccDisposition::Drop => {}
                    // `IccDisposition` is #[non_exhaustive]; conservatively
                    // keep the source ICC rather than silently dropping it.
                    _ => {
                        if let Some(ref icc) = meta.icc_profile {
                            req = req.icc_profile(icc);
                        }
                    }
                }
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

            // HDR content-light-level (`meta.content_light_level` — CTA-861.3
            // MaxCLL/MaxFALL) and mastering display (`meta.mastering_display` —
            // SMPTE ST 2086) have NO native JPEG marker and no standard
            // JPEG/XMP schema, so they are intentionally not emitted here — a
            // documented format limitation, not a silent give-up. The HDR
            // *peak* is not lost for true-HDR output: `ultrahdr::encode` carries
            // it as the gain-map headroom (`alternate_hdr_headroom`, measured
            // from the actual content ≈ `log2(MaxCLL / 203-nit diffuse white)`),
            // so no forwarded MaxCLL is needed. MaxFALL and the mastering volume
            // have no carrier in JPEG or UltraHDR and are dropped.
            let _ = (&meta.content_light_level, &meta.mastering_display);
        }
        if let Some(ref stop) = self.stop {
            req = req.stop(stop);
        }
        Ok(req)
    }

    /// Pre-flight limit checks.
    fn check_limits(&self, width: u32, height: u32, layout: PixelLayout) -> Result<(), Error> {
        self.limits.check_dimensions(width, height).map_err(|_| {
            Error::image_too_large(
                width as u64 * height as u64,
                self.limits.max_pixels.unwrap_or(0),
            )
        })?;
        // Honest pre-flight: gate on the calibrated peak estimate — the
        // encoder's working set (`heuristics::estimate_encode`, VmHWM-marginal
        // calibrated, 2026-06-23 sweep, `estimate_memory_ceiling`-backed safe
        // upper bound) PLUS the input buffer held for the encode's duration —
        // not just the `w*h*bpp` input buffer, which under-states the real
        // peak severalfold. Same input+working convention as
        // `estimate_encode_resources`.
        let input_bytes = width as u64 * height as u64 * layout.bytes_per_pixel() as u64;
        let est = crate::heuristics::estimate_encode(width, height, &self.effective_config);
        let estimated_mem = est.peak_memory_bytes.saturating_add(input_bytes);
        self.limits.check_memory(estimated_mem).map_err(|_| {
            Error::resource_limit_exceeded(
                zencodec::LimitKind::Memory,
                estimated_mem,
                self.limits.max_memory_bytes.unwrap_or(0),
            )
        })?;
        Ok(())
    }

    /// Check output size limits after encoding.
    fn check_output_size(&self, output: &[u8]) -> Result<(), Error> {
        self.limits
            .check_output_size(output.len() as u64)
            .map_err(|_| {
                Error::resource_limit_exceeded(
                    zencodec::LimitKind::OutputSize,
                    output.len() as u64,
                    self.limits.max_output_bytes.unwrap_or(0),
                )
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
        let req = self.build_request(Some(layout.channels() as u8))?;
        let output = req.encode_bytes(data, width, height, layout)?;
        self.check_output_size(&output)?;
        Ok(EncodeOutput::new(output, ImageFormat::Jpeg))
    }

    /// Stream accumulated rows through the native BytesEncoder.
    fn encode_accumulated(&self, acc: RowAccumulator) -> Result<EncodeOutput, Error> {
        self.check_limits(acc.width, acc.total_rows, acc.layout)?;

        let req = self.build_request(Some(acc.layout.channels() as u8))?;
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
    type Error = At<CodecError>;

    fn reject(op: UnsupportedOperation) -> Self::Error {
        ErrorKind::UnsupportedOperation(op).into()
    }

    fn preferred_strip_height(&self) -> u32 {
        16
    }

    fn encode(self, pixels: PixelSlice<'_>) -> Result<EncodeOutput, Self::Error> {
        let layout = descriptor_to_layout(pixels.descriptor())?;
        let width = pixels.width();
        let height = pixels.rows();
        let data = pixels.contiguous_bytes();
        self.encode_bytes_inner(&data, width, height, layout)
            .map_err(Into::into)
    }

    fn encode_srgba8(
        self,
        data: &mut [u8],
        make_opaque: bool,
        width: u32,
        height: u32,
        stride_pixels: u32,
    ) -> Result<EncodeOutput, Self::Error> {
        if make_opaque {
            for chunk in data.as_chunks_mut::<4>().0 {
                chunk[3] = 255;
            }
        }
        let layout = PixelLayout::Rgba8Srgb;
        self.check_limits(width, height, layout)?;
        let req = self.build_request(Some(layout.channels() as u8))?;
        let stop = self.stop_ref();
        let stride_bytes = stride_pixels as usize * 4;
        let mut enc = req.encode_from_bytes(width, height, layout)?;
        enc.push(data, height as usize, stride_bytes, stop)?;
        let output = enc.finish()?;
        self.check_output_size(&output)?;
        Ok(EncodeOutput::new(output, ImageFormat::Jpeg))
    }

    fn push_rows(&mut self, rows: PixelSlice<'_>) -> Result<(), Self::Error> {
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
                let req =
                    self.build_request_from(&streaming_config, Some(layout.channels() as u8))?;
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
                    return Err(Error::invalid_state(
                        "push_rows: width or format changed between calls",
                    )
                    .into());
                }
                acc.data.extend_from_slice(&data);
                acc.total_rows += rows.rows();
            }
        }
        Ok(())
    }

    fn finish(mut self) -> Result<EncodeOutput, Self::Error> {
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
            .ok_or_else(|| Error::invalid_state("finish() called without any push_rows()"))?;
        self.encode_accumulated(acc).map_err(Into::into)
    }

    fn encode_from(
        self,
        source: &mut dyn FnMut(u32, PixelSliceMut<'_>) -> usize,
    ) -> Result<EncodeOutput, Self::Error> {
        use zenpixels::PixelSliceMut;

        // Invoked out of sequence (canvas size must be set before pulling rows)
        // — an API-protocol violation, not an unsupported feature (caterr
        // Pattern-B follow-up finding #1 investigation).
        let (img_w, img_h) = self.image_size.ok_or_else(|| {
            Error::invalid_state(
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
        let req = self.build_request_from(&streaming_config, Some(layout.channels() as u8))?;
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

            // `buf`/`stride`/`rows_wanted` are all computed just above from our
            // own `strip_h`/`bpp`/`img_w` arithmetic, not caller-supplied — a
            // failure here means that internal sizing is inconsistent, i.e. a
            // bug in this function, not an unsupported feature (caterr
            // Pattern-B follow-up finding #1 investigation).
            let mut pixel_buf =
                PixelSliceMut::new(&mut buf[..slice_size], img_w, rows_wanted, stride, desc)
                    .map_err(|e| {
                        let _ = e;
                        Error::internal("encode_from: internal pixel buffer construction failed")
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
        // A caller-requested pixel format/descriptor this codec doesn't
        // negotiate — the dedicated `UnsupportedOperation::PixelFormat` axis,
        // not the generic string-payload `unsupported_feature` (caterr
        // Pattern-B follow-up finding #1 investigation).
        _ => Err(zencodec::UnsupportedOperation::PixelFormat.into()),
    }
}

// ============================================================================
// Decode side: DecoderConfig → DecodeJob → Decoder / StreamingDecoder
// ============================================================================
