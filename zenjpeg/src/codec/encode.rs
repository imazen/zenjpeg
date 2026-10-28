//! zencodec encode-side impls: `JpegEncoderConfig` / `JpegEncodeJob` / `JpegEncoder`.

use alloc::sync::Arc;
use alloc::vec::Vec;

use whereat::At;
use zencodec::encode::{EncodeCapabilities, EncodeOutput};
use zencodec::{CodecError, ImageFormat, Metadata, ResourceLimits, UnsupportedOperation};
use zenpixels::{
    Cicp, ColorContext, ColorModel, ColorPrimaries, ColorProfileSource, PixelDescriptor,
    PixelSlice, PixelSliceMut, TransferFunction,
};

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
    .with_streaming_output(true) // fixed-table sequential configurations; see `begin`
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
    /// Settings the caller made explicitly. They are applied after the
    /// effort preset (see `effective_config`), so the preset cannot undo them.
    overrides: Overrides,
}

/// Explicit settings that outrank the effort preset.
#[derive(Clone, Debug, Default)]
struct Overrides {
    /// A caller supplied native settings through `inner_mut`.
    native_config: bool,
    /// From `from_preset`: replaces the effort-derived preset.
    preset: Option<crate::encode::encoder_types::OptimizationPreset>,
    /// From `with_progressive`.
    progressive: Option<bool>,
    /// From `with_huffman`.
    huffman: Option<crate::encode::encoder_types::HuffmanStrategy>,
    /// From `EncoderConfig::with_prefer_streaming`: fixed tables and a
    /// sequential scan unless chosen explicitly above.
    prefer_streaming: Option<bool>,
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
            overrides: Overrides::default(),
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
            overrides: Overrides::default(),
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
            overrides: Overrides::default(),
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
            overrides: Overrides {
                preset: Some(preset),
                ..Overrides::default()
            },
        })
    }

    /// Enable progressive JPEG encoding. Overrides the scan mode the
    /// effort level would choose.
    #[must_use]
    pub fn with_progressive(mut self, enable: bool) -> Self {
        self.inner = self.inner.progressive(enable);
        self.overrides.progressive = Some(enable);
        self
    }

    /// Choose the Huffman tables, overriding the effort level's choice
    /// (optimized per image).
    ///
    /// Accepts a `bool` (`true`: optimized, `false`: fixed general-purpose
    /// tables, which make larger files), a
    /// [`HuffmanStrategy`](crate::encoder::HuffmanStrategy), or a custom
    /// table set. Optimized tables need every block before the scan can be
    /// written, so their output arrives at `finish`. With fixed or custom
    /// tables and a sequential scan (`with_progressive(false)`, or effort
    /// 0), the encoder streams: `Encoder::begin` reports
    /// [`OutputTiming::streaming`](zencodec::encode::OutputTiming::streaming)
    /// and the header and each MCU row become available through
    /// `pending_output` as rows are pushed.
    #[must_use]
    pub fn with_huffman(
        mut self,
        strategy: impl Into<crate::encode::encoder_types::HuffmanStrategy>,
    ) -> Self {
        let strategy = strategy.into();
        self.inner = self.inner.huffman(strategy.clone());
        self.overrides.huffman = Some(strategy);
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
    ///
    /// Treats the native configuration as explicit: generic effort and a
    /// streaming preference do not replace its optimization choices.
    pub fn inner_mut(&mut self) -> &mut EncoderConfig {
        self.overrides.native_config = true;
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
        if self.overrides.native_config {
            return self.inner.clone();
        }
        use crate::encode::encoder_types::OptimizationPreset;
        let preset = self
            .overrides
            .preset
            .unwrap_or(match self.effort.clamp(0, 2) {
                0 => OptimizationPreset::JpegliBaseline,
                2 => OptimizationPreset::HybridMaxCompression,
                _ => OptimizationPreset::HybridProgressive,
            });
        let mut cfg = self.inner.clone().optimization(preset);
        // Explicit settings outrank the preset. Progressive first: turning
        // it on selects optimized tables, which an explicit Huffman choice
        // may then replace.
        // A streaming preference fills in what the caller left open.
        let stream = self.overrides.prefer_streaming == Some(true);
        match self.overrides.progressive {
            Some(progressive) => cfg = cfg.progressive(progressive),
            None if stream => cfg = cfg.progressive(false),
            None => {}
        }
        match &self.overrides.huffman {
            Some(huffman) => cfg = cfg.huffman(huffman.clone()),
            None if stream => {
                cfg = cfg.huffman(crate::encode::encoder_types::HuffmanStrategy::Fixed);
            }
            None => {}
        }
        cfg
    }
}

impl Default for JpegEncoderConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Supported encode pixel formats.
///
/// The native 16-bit layouts hold linear light, like the f32 ones.
/// sRGB-encoded 16-bit samples are decoded to linear f32 on the way in
/// ([`srgb16_layout`]); the linear 16-bit descriptors need no conversion.
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
    linear16(zenpixels::ChannelLayout::Rgb, None),
    linear16(
        zenpixels::ChannelLayout::Rgba,
        Some(zenpixels::AlphaMode::Straight),
    ),
    linear16(zenpixels::ChannelLayout::Gray, None),
];

const fn linear16(
    layout: zenpixels::ChannelLayout,
    alpha: Option<zenpixels::AlphaMode>,
) -> PixelDescriptor {
    PixelDescriptor::new(
        zenpixels::ChannelType::U16,
        layout,
        alpha,
        TransferFunction::Linear,
    )
}

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
        // Map the generic quality dial through the fixed CID22 calibration.
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
            // Mozjpeg / future variants → report on the codec quality scale.
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
    /// `true`: fixed general-purpose Huffman tables and a sequential scan,
    /// unless `with_huffman` / `with_progressive` chose otherwise, so
    /// `Encoder::begin` can stream (files grow by the fixed tables' cost).
    fn with_prefer_streaming(mut self, prefer: bool) -> Self {
        self.overrides.prefer_streaming = Some(prefer);
        self
    }

    fn prefers_streaming(&self) -> Option<bool> {
        Some(self.overrides.prefer_streaming.unwrap_or(false))
    }

    fn with_generic_effort(mut self, effort: i32) -> Self {
        // A later effort request replaces a named preset.
        self.effort = effort;
        self.overrides.preset = None;
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
            pulse: None,
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
#[derive(Clone)]
pub struct JpegEncodeJob {
    config: JpegEncoderConfig,
    pulse: Option<zencodec::SharedPulse>,
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

    fn with_pulse(mut self, pulse: zencodec::SharedPulse) -> Self {
        self.pulse = Some(pulse);
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

    fn try_set_policy(
        &mut self,
        policy: zencodec::encode::EncodePolicy,
    ) -> Result<(), zencodec::UnsupportedOperation> {
        use zencodec::{ColorEmitPolicy, encode::EncodePolicy};
        if policy == EncodePolicy::none() {
            return Ok(());
        }
        // Qualify supplied color carriers; other emission requirements must
        // not silently fall through to the legacy best-effort setter.
        if policy != EncodePolicy::none().with_color(ColorEmitPolicy::Verbatim) {
            return Err(zencodec::UnsupportedOperation::EncodePolicy);
        }
        self.policy = Some(policy);
        Ok(())
    }

    fn with_canvas_size(mut self, width: u32, height: u32) -> Self {
        self.image_size = Some((width, height));
        self
    }

    /// The plan [`Encoder::begin_declared`](zencodec::encode::Encoder::begin_declared)
    /// returns for this job and declaration: one ordered row pass, with the
    /// output timing of the native row encoder.
    ///
    /// Runs the same validation, limits, colour binding and native builder
    /// geometry as `begin_declared`, without starting native pixel or
    /// coefficient storage. Configuration and metadata preparation may
    /// allocate (a synthesized ICC profile, for example); this is not a memory
    /// bound. `declaration.color` is the current interpretation of the rows
    /// that will follow; retain this exact job, because the plan describes it
    /// and nothing else. JPEG never patches earlier bytes, so the declared
    /// access does not change the layout. Colour carriers are not reported
    /// yet (`carriers: None`, unknown). Errors are the job's own
    /// `At<CodecError>`, boxed.
    fn plan_encode(
        &self,
        declaration: &zencodec::encode::EncodeDeclaration<'_>,
    ) -> Result<zencodec::encode::EncodePlan, zencodec::encode::BoxedError> {
        let start = zencodec::encode::EncodeJob::encoder(self.clone())
            .and_then(|encoder| encoder.prepare_rows(declaration).map_err(Into::into))
            .map_err(|e| Box::new(e) as zencodec::encode::BoxedError)?;
        Ok(zencodec::encode::EncodePlan::rows(start.timing))
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
            // A supplied native configuration already chose its parallel
            // mode. The generic default must not replace that choice.
            if self.limits.threading.is_parallel() && !self.config.overrides.native_config {
                cfg = cfg.parallel(crate::encode::ParallelEncoding::Auto);
            }
            // Otherwise (Sequential): leave cfg.parallel as None.
        }

        Ok(JpegEncoder {
            effective_config: cfg,
            pulse: self.pulse,
            metadata: self.metadata,
            limits: self.limits,
            policy: self.policy,
            accumulator: None,
            streaming_enc: None,
            image_size: self.image_size,
            row_descriptor: None,
            sent: 0,
            begun: false,
            bound: None,
            checked_context: None,
            srgb16_rows: Vec::new(),
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
    /// Cancellation and progress control. The native encoder polls it as a
    /// `Stop`; nothing reports progress through it yet.
    pulse: Option<zencodec::SharedPulse>,
    metadata: Option<Metadata>,
    limits: ResourceLimits,
    policy: Option<zencodec::encode::EncodePolicy>,
    /// Accumulated rows for push_rows path (fallback when dimensions unknown).
    accumulator: Option<RowAccumulator>,
    /// Native row encoder — used when dimensions are known, from `begin`
    /// or `with_canvas_size`. Takes rows without accumulating pixels and
    /// encodes with the configuration as given: fixed-table sequential
    /// output streams (after `begin`), optimized tables and progressive
    /// scans keep their blocks until `finish`.
    streaming_enc: Option<crate::encode::byte_encoders::BytesEncoder>,
    /// The descriptor rows must arrive in, once the row encoder exists.
    row_descriptor: Option<PixelDescriptor>,
    /// Bytes taken through `consume_output`, for the output-size limit.
    sent: u64,
    /// `begin` succeeded: the image is declared.
    begun: bool,
    /// Image dimensions from `begin` or `with_canvas_size`.
    image_size: Option<(u32, u32)>,
    /// The colour description the header was (or will be) written from,
    /// fixed when the row encode starts.
    bound: Option<ColorBinding>,
    /// The last row colour context found consistent with `bound`.
    checked_context: Option<Arc<ColorContext>>,
    /// Reused linear f32 rows for 16-bit sRGB input (`decode_srgb16`).
    srgb16_rows: Vec<u8>,
}

/// Internal buffer for accumulating pushed rows.
struct RowAccumulator {
    data: Vec<u8>,
    width: u32,
    total_rows: u32,
    layout: PixelLayout,
    descriptor: PixelDescriptor,
}

/// What a row encode decides before native startup.
struct RowStart {
    layout: PixelLayout,
    binding: ColorBinding,
    timing: zencodec::encode::OutputTiming,
}

/// What the stored samples mean, as a reader of the written file must
/// interpret them.
#[derive(Clone, Debug, PartialEq)]
enum StoredColor {
    /// The pixels assert nothing beyond the format default (sRGB).
    Default,
    /// This profile describes the stored samples.
    Icc(Arc<[u8]>),
    /// These code points describe the stored samples.
    Cicp(Cicp),
}

impl StoredColor {
    fn parts(&self) -> (Option<&[u8]>, Option<Cicp>) {
        match self {
            Self::Default => (None, None),
            Self::Icc(icc) => (Some(icc), None),
            Self::Cicp(cicp) => (None, Some(*cicp)),
        }
    }
}

/// Which description the colour carrier is written from.
#[derive(Clone, Debug, PartialEq)]
enum ColorBinding {
    /// The job metadata, which the pixels agree with or do not contradict.
    Metadata,
    /// The stored samples' own interpretation.
    Stored(StoredColor),
}

/// Linear-light layouts, which the encoder stores sRGB-encoded.
fn stores_linear_as_srgb(layout: PixelLayout) -> bool {
    layout.is_16bit() || (layout.is_float() && !layout.is_ycbcr())
}

/// Whether the encoder's sRGB encoding changes what the input samples mean:
/// linear input does, 16-bit sRGB input (decoded to linear first) does not.
fn transfer_changes(layout: PixelLayout, descriptor: PixelDescriptor) -> bool {
    stores_linear_as_srgb(layout) && descriptor.transfer() != TransferFunction::Srgb
}

/// U8 bytes whose interpretation only a retained ICC gives (`pixel_layout`).
fn is_profile_tagged(descriptor: PixelDescriptor) -> bool {
    descriptor_to_layout(descriptor).is_err() && srgb16_layout(descriptor).is_none()
}

/// `primaries` with the sRGB transfer: what sRGB-encoded storage of linear
/// samples in those primaries means.
fn srgb_encoded(primaries: ColorPrimaries, gray: bool) -> Result<StoredColor, Error> {
    if gray {
        return Ok(StoredColor::Cicp(Cicp::new(2, 13, 0, true)));
    }
    primaries
        .to_cicp()
        .map(|code| StoredColor::Cicp(Cicp::new(code, 13, 0, true)))
        .ok_or_else(|| {
            Error::icc_error(
                "these primaries have no code point to synthesize a profile from; attach \
                 their ICC profile to the pixels"
                    .into(),
            )
        })
}

/// The primaries and transfer a description identifies. `None` for an
/// unidentified profile or code points zenpixels cannot name; an empty
/// description is the sRGB default.
fn identify(
    (icc, cicp): (Option<&[u8]>, Option<Cicp>),
) -> Option<(ColorPrimaries, TransferFunction)> {
    match (icc, cicp) {
        (Some(icc), _) => ColorProfileSource::Icc(icc).resolve(),
        (None, Some(cicp)) => ColorProfileSource::Cicp(cicp).resolve(),
        (None, None) => Some((ColorPrimaries::Bt709, TransferFunction::Srgb)),
    }
}

/// Whether two descriptions mean the same: identical profile bytes or code
/// points, or the same identified primaries and transfer (only the transfer
/// for grayscale samples).
fn same_meaning(
    a: (Option<&[u8]>, Option<Cicp>),
    b: (Option<&[u8]>, Option<Cicp>),
    gray: bool,
) -> bool {
    if a.0.is_some() && a.0 == b.0 || a.0.is_none() && b.0.is_none() && a.1 == b.1 {
        return true;
    }
    match (identify(a), identify(b)) {
        (Some(a), Some(b)) => a.1 == b.1 && (gray || a.0 == b.0),
        _ => false,
    }
}

/// Whether a description is identified as something other than the sRGB
/// default: non-default code points, or a profile zenpixels identifies.
fn is_identified_non_default((icc, cicp): (Option<&[u8]>, Option<Cicp>), gray: bool) -> bool {
    match icc.map(|icc| ColorProfileSource::Icc(icc).resolve()) {
        Some(Some((primaries, transfer))) => {
            transfer != TransferFunction::Srgb || !gray && primaries != ColorPrimaries::Bt709
        }
        Some(None) | None => cicp.is_some_and(|c| !is_default_meaning((None, Some(c)), gray)),
    }
}

/// Whether a description is the sRGB default an untagged JPEG already means.
fn is_default_meaning((icc, cicp): (Option<&[u8]>, Option<Cicp>), gray: bool) -> bool {
    match (icc, cicp) {
        (None, None) => true,
        (None, Some(c)) => {
            matches!(c.transfer_characteristics, 2 | 13)
                && (gray || matches!(c.color_primaries, 1 | 2))
        }
        (Some(icc), _) => {
            zenpixels::icc::is_common_srgb(icc)
                || matches!(
                    ColorProfileSource::Icc(icc).resolve(),
                    Some((primaries, TransferFunction::Srgb))
                        if gray || primaries == ColorPrimaries::Bt709
                )
        }
    }
}

impl JpegEncoder {
    /// Get a reference to the stop token, defaulting to Unstoppable.
    fn stop_ref(&self) -> &dyn enough::Stop {
        match self.pulse {
            Some(ref s) => s,
            None => &enough::Unstoppable,
        }
    }

    /// Everything `begin_declared` decides before native startup: layout,
    /// colour carrier and output timing. `plan_encode` reports it, and
    /// startup begins the native encoder from the same value.
    fn prepare_rows(
        &self,
        declaration: &zencodec::encode::EncodeDeclaration<'_>,
    ) -> Result<RowStart, Error> {
        // JPEG output is append-only: seek-back access permits nothing more.
        let (width, height, descriptor, context) = (
            declaration.width,
            declaration.height,
            declaration.descriptor,
            declaration.color,
        );
        self.stop_ref().check()?;
        if self.image_size.is_some_and(|s| s != (width, height)) {
            return Err(Error::invalid_state(
                "begin: size differs from with_canvas_size",
            ));
        }
        let layout = self.pixel_layout(descriptor)?;
        self.check_row_filters(layout)?;
        self.check_limits(width, height, layout)?;
        let binding = self.bind_color(descriptor, context, layout)?;
        let lag = self
            .build_request(layout, &binding)?
            .plan_byte_output(width, height, layout)?;
        Ok(RowStart {
            layout,
            timing: self.timing(lag, height),
            binding,
        })
    }

    /// Row output timing for a native lag (`None`: everything at `finish`).
    ///
    /// Keeps the configuration's established conservative lag bound even
    /// when gray input uses smaller, single-component coding units. The bound
    /// describes availability, not coefficient/restart geometry. Holding every
    /// row would be no stream at all: the header alone is not worth a chunked
    /// response.
    fn timing(&self, lag: Option<u32>, height: u32) -> zencodec::encode::OutputTiming {
        use zencodec::encode::OutputTiming;
        let configured_lag = match self.effective_config.color_mode {
            crate::encode::encoder_types::ColorMode::YCbCr {
                subsampling: ChromaSubsampling::Quarter,
            } => 31,
            _ => 15,
        };
        match lag.map(|lag| lag.max(configured_lag)) {
            Some(lag) if lag < height => OutputTiming::streaming(lag),
            _ => OutputTiming::AT_FINISH,
        }
    }

    /// Build an EncodeRequest from the current config, the bound colour
    /// carrier and the job's metadata, applying policy.
    fn build_request<'b>(
        &'b self,
        layout: PixelLayout,
        binding: &'b ColorBinding,
    ) -> Result<crate::encode::request::EncodeRequest<'b>, Error> {
        self.build_request_from(&self.effective_config, layout, binding)
    }

    /// Nonlinear byte encoders consume stored sample values without a transfer
    /// conversion. Qualify profile-tagged U8 input only in YCbCr mode and only
    /// when the matching ICC will actually be retained. XYB has a separate
    /// interpretation transform and is deliberately not admitted here.
    fn pixel_layout(&self, desc: PixelDescriptor) -> Result<PixelLayout, Error> {
        use zenpixels::{
            AlphaMode, ChannelLayout, ChannelType, ColorPrimaries, SignalRange, TransferFunction,
        };
        if let Ok(layout) = descriptor_to_layout(desc) {
            return Ok(layout);
        }
        if let Some(layout) = srgb16_layout(desc) {
            return Ok(layout);
        }
        if desc.channel_type() != ChannelType::U8
            || desc.signal_range != SignalRange::Full
            || desc.alpha() == Some(AlphaMode::Premultiplied)
            || !matches!(
                desc.transfer(),
                TransferFunction::Unknown | TransferFunction::Gamma22
            )
            || !matches!(
                self.effective_config.color_mode,
                crate::encode::encoder_types::ColorMode::YCbCr { .. }
            )
        {
            return Err(zencodec::UnsupportedOperation::PixelFormat.into());
        }
        let Some(zencodec::Metadata {
            icc_profile: Some(icc),
            cicp,
            ..
        }) = &self.metadata
        else {
            return Err(Error::icc_error(
                "profile-tagged bytes require retained ICC metadata".into(),
            ));
        };
        if zenpixels::icc::profile_color_space(icc) != Some(zenpixels::ColorModel::Rgb) {
            return Err(Error::icc_error(
                "RGB samples require an RGB ICC carrier".into(),
            ));
        }
        let resolved = zenpixels::ColorProfileSource::Icc(icc).resolve();
        let compatible = match resolved {
            Some((primaries, transfer)) => {
                (desc.primaries == ColorPrimaries::Unknown || desc.primaries == primaries)
                    && (desc.transfer() == TransferFunction::Unknown || desc.transfer() == transfer)
            }
            None => {
                desc.primaries == ColorPrimaries::Unknown
                    && desc.transfer() == TransferFunction::Unknown
            }
        };
        if !compatible {
            return Err(Error::icc_error(
                "descriptor contradicts the retained ICC interpretation".into(),
            ));
        }
        let policy = self.policy.unwrap_or_default();
        let mut source = zencodec::SourceColor::default().with_icc_profile(icc.clone());
        source.cicp = *cicp;
        if !policy.resolve_icc(true) {
            return Err(Error::encode_policy_rejected(
                "the policy forbids the ICC profile these profile-tagged bytes need",
            ));
        }
        if !matches!(
            zencodec::resolve_color_emit(
                &source,
                &JPEG_ENCODE_CAPS,
                policy.resolve_color(zencodec::ColorEmitPolicy::Balanced)
            )
            .icc,
            zencodec::IccDisposition::KeepSource
        ) {
            return Err(Error::icc_error(
                "profile-tagged bytes require keeping their ICC carrier".into(),
            ));
        }
        match desc.layout() {
            ChannelLayout::Rgb => Ok(PixelLayout::Rgb8Srgb),
            ChannelLayout::Rgba if desc.alpha() == Some(AlphaMode::Undefined) => {
                Ok(PixelLayout::Rgbx8Srgb)
            }
            ChannelLayout::Rgba => Ok(PixelLayout::Rgba8Srgb),
            ChannelLayout::Bgra if desc.alpha() == Some(AlphaMode::Undefined) => {
                Ok(PixelLayout::Bgrx8Srgb)
            }
            ChannelLayout::Bgra => Ok(PixelLayout::Bgra8Srgb),
            // Gray profile and RGB/gray carrier compatibility need separate qualification.
            _ => Err(zencodec::UnsupportedOperation::PixelFormat.into()),
        }
    }

    fn check_declared_context(
        &self,
        descriptor: PixelDescriptor,
        context: Option<&zenpixels::ColorContext>,
    ) -> Result<(), Error> {
        if matches!(
            descriptor.transfer(),
            zenpixels::TransferFunction::Unknown | zenpixels::TransferFunction::Gamma22
        ) && descriptor.channel_type() == zenpixels::ChannelType::U8
            && let Some(context) = context
            && (context.cicp.is_some()
                || context.icc.as_ref()
                    != self.metadata.as_ref().and_then(|m| m.icc_profile.as_ref()))
        {
            return Err(Error::icc_error(
                "current pixel context differs from the retained ICC metadata".into(),
            ));
        }
        Ok(())
    }

    /// What the stored samples mean, from the pixels being encoded: their
    /// colour context, then the primaries their descriptor asserts.
    ///
    /// Linear layouts are stored sRGB-encoded, so their interpretation is the
    /// same primaries with the sRGB transfer: a profile or code points that
    /// described the linear samples no longer apply, and an unidentified
    /// profile cannot be carried across that transform.
    fn stored_color(
        &self,
        descriptor: PixelDescriptor,
        context: Option<&ColorContext>,
        layout: PixelLayout,
    ) -> Result<StoredColor, Error> {
        let gray = layout.is_grayscale();
        let linear = transfer_changes(layout, descriptor);
        // BT.709 is what every untagged buffer is labelled with, so only other
        // primaries are an assertion. Grayscale has no primaries.
        let asserted = match descriptor.primaries {
            _ if gray => None,
            ColorPrimaries::Unknown | ColorPrimaries::Bt709 => None,
            primaries => Some(primaries),
        };
        let refuse = |reason: &str| Err(Error::icc_error(reason.into()));
        if let Some(icc) = context.and_then(|c| c.icc.as_ref()) {
            let channels = if gray {
                ColorModel::Gray
            } else {
                ColorModel::Rgb
            };
            if zenpixels::icc::profile_color_space(icc) != Some(channels) {
                return refuse("the pixels' ICC profile does not describe their channels");
            }
            let identified = ColorProfileSource::Icc(icc).resolve();
            if let Some(cicp) = context.and_then(|c| c.cicp)
                && (identified.is_none() || identified != ColorProfileSource::Cicp(cicp).resolve())
            {
                return refuse("the pixels' ICC profile and code points disagree");
            }
            if asserted.is_some() && identified.map(|(p, _)| p) != asserted {
                return refuse("the descriptor's primaries contradict the pixels' ICC profile");
            }
            if !linear {
                return Ok(StoredColor::Icc(Arc::clone(icc)));
            }
            return match identified {
                Some((primaries, TransferFunction::Linear)) => srgb_encoded(primaries, gray),
                Some(_) => refuse("the pixels' ICC profile does not describe linear samples"),
                None => refuse(
                    "an unidentified ICC profile cannot describe the sRGB-encoded samples \
                     stored for linear input",
                ),
            };
        }
        if let Some(cicp) = context.and_then(|c| c.cicp) {
            if cicp.matrix_coefficients != 0 || !cicp.full_range {
                return refuse("YCbCr or narrow-range code points do not describe RGB samples");
            }
            if asserted.is_some() && ColorPrimaries::from_cicp(cicp.color_primaries) != asserted {
                return refuse("the descriptor's primaries contradict the pixels' code points");
            }
            if !linear {
                return Ok(StoredColor::Cicp(cicp));
            }
            if cicp.transfer_characteristics != 8 {
                return refuse("the pixels' code points do not describe linear samples");
            }
            return Ok(StoredColor::Cicp(Cicp::new(
                cicp.color_primaries,
                13,
                0,
                true,
            )));
        }
        match asserted {
            None => Ok(StoredColor::Default),
            Some(primaries) => srgb_encoded(primaries, gray),
        }
    }

    /// Decide which description the colour carrier is written from.
    ///
    /// Job metadata labels pixels without an explicit colour context, even when
    /// their descriptor names other primaries (the established contract, also
    /// zenpng's `explicit_icc_is_not_overridden_by_descriptor_derived_cicp`).
    /// It must agree with an explicit context: a profile kept after a
    /// conversion is a stale carrier. Without colour metadata the stored
    /// samples' own interpretation is written, so a checked request, which
    /// carries no metadata, keeps the pixels' colour.
    fn bind_color(
        &self,
        descriptor: PixelDescriptor,
        context: Option<&ColorContext>,
        layout: PixelLayout,
    ) -> Result<ColorBinding, Error> {
        if is_profile_tagged(descriptor) {
            // Profile-tagged bytes: `pixel_layout` admitted them only with a
            // matching retained ICC, and the context must be that profile.
            self.check_declared_context(descriptor, context)?;
            return Ok(ColorBinding::Metadata);
        }
        let gray = layout.is_grayscale();
        let stored = self.stored_color(descriptor, context, layout)?;
        let metadata = self
            .metadata
            .as_ref()
            .filter(|m| m.icc_profile.is_some() || m.cicp.is_some())
            .map(|m| (m.icc_profile.as_deref(), m.cicp));
        if matches!(
            self.effective_config.color_mode,
            crate::encode::encoder_types::ColorMode::Xyb { .. }
        ) && (!is_default_meaning(stored.parts(), gray)
            || metadata.is_some_and(|m| !is_default_meaning(m, gray)))
        {
            // XYB converts from sRGB primaries and writes its own profile.
            return Err(Error::icc_error(
                "XYB encoding assumes sRGB primaries; convert these pixels first".into(),
            ));
        }
        let Some(metadata) = metadata else {
            return Ok(ColorBinding::Stored(stored));
        };
        let explicit = context.is_some_and(|c| c.icc.is_some() || c.cicp.is_some());
        if !explicit {
            if transfer_changes(layout, descriptor)
                && matches!(identify(metadata), Some((_, TransferFunction::Linear)))
            {
                return Err(Error::icc_error(
                    "the colour metadata describes linear samples, which this encoder \
                     stores sRGB-encoded"
                        .into(),
                ));
            }
            return Ok(ColorBinding::Metadata);
        }
        if same_meaning(metadata, stored.parts(), gray) {
            Ok(ColorBinding::Metadata)
        } else {
            Err(Error::icc_error(
                "the colour metadata contradicts the pixels' current colour".into(),
            ))
        }
    }

    /// Rows after the header must mean what the header says. Rows without a
    /// colour context assert nothing.
    fn check_rows_color(
        &mut self,
        rows: &PixelSlice<'_>,
        layout: PixelLayout,
    ) -> Result<(), Error> {
        let Some(context) = rows.color_context() else {
            return Ok(());
        };
        if self
            .checked_context
            .as_ref()
            .is_some_and(|checked| Arc::ptr_eq(checked, context))
        {
            return Ok(());
        }
        let descriptor = rows.descriptor();
        if is_profile_tagged(descriptor) {
            self.check_declared_context(descriptor, Some(context))?;
        } else {
            let stored = self.stored_color(descriptor, Some(context), layout)?;
            let header = match &self.bound {
                Some(ColorBinding::Stored(header)) => header.parts(),
                Some(ColorBinding::Metadata) => self
                    .metadata
                    .as_ref()
                    .map_or((None, None), |m| (m.icc_profile.as_deref(), m.cicp)),
                None => return Err(Error::internal("row colour checked before binding")),
            };
            if stored != StoredColor::Default
                && !same_meaning(header, stored.parts(), layout.is_grayscale())
            {
                return Err(Error::icc_error(
                    "the rows' colour context differs from the colour declared before \
                     the header; declare it with begin_declared"
                        .into(),
                ));
            }
        }
        self.checked_context = Some(Arc::clone(context));
        Ok(())
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
        layout: PixelLayout,
        binding: &'b ColorBinding,
    ) -> Result<crate::encode::request::EncodeRequest<'b>, Error> {
        let mut req = config.request();
        let policy = self.policy.unwrap_or_default();
        let gray = layout.is_grayscale();
        let (icc, cicp) = match binding {
            ColorBinding::Metadata => self
                .metadata
                .as_ref()
                .map_or((None, None), |m| (m.icc_profile.as_ref(), m.cicp)),
            ColorBinding::Stored(StoredColor::Default) => (None, None),
            ColorBinding::Stored(StoredColor::Icc(icc)) => (Some(icc), None),
            ColorBinding::Stored(StoredColor::Cicp(cicp)) => (None, Some(*cicp)),
        };
        let mut wrote_icc = false;
        if policy.resolve_icc(true) && (icc.is_some() || cicp.is_some()) {
            let color_policy = policy.resolve_color(zencodec::ColorEmitPolicy::Balanced);
            // The channel count lets the resolver suppress ICC synthesis for
            // grayscale, where an RGB profile would recolor the image.
            let mut src =
                zencodec::SourceColor::default().with_channel_count(layout.channels() as u8);
            if let Some(c) = cicp {
                src = src.with_cicp(c);
            }
            if let Some(icc) = icc {
                src = src.with_icc_profile(Arc::clone(icc));
            }
            let plan = zencodec::resolve_color_emit(&src, &JPEG_ENCODE_CAPS, color_policy);
            match plan.icc {
                zencodec::IccDisposition::KeepSource => {
                    if let Some(icc) = icc {
                        req = req.icc_profile(icc);
                        wrote_icc = true;
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
                            wrote_icc = true;
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
                    if let Some(icc) = icc {
                        req = req.icc_profile(icc);
                        wrote_icc = true;
                    }
                }
            }
        }
        // JPEG's only colour carrier is the ICC profile. A plan or policy that
        // writes none for a description identified as non-sRGB would relabel
        // the image. An unidentified profile can still be stripped by an
        // explicit policy, as before.
        if !wrote_icc && is_identified_non_default((icc.map(|i| &i[..]), cicp), gray) {
            if !policy.resolve_icc(true) {
                return Err(Error::encode_policy_rejected(
                    "the policy forbids the ICC profile, JPEG's only carrier for this \
                     image's colour",
                ));
            }
            return Err(Error::icc_error(
                "the policy leaves no carrier for this image's colour: JPEG can only \
                 describe it with an embedded ICC profile"
                    .into(),
            ));
        }
        if let Some(ref meta) = self.metadata {
            let oriented = meta.orientation != zencodec::Orientation::Identity;
            if policy.resolve_exif(true) {
                if let Some(ref exif) = meta.exif {
                    req = req.exif(Exif::raw(exif.to_vec()));
                } else if oriented {
                    // JPEG carries orientation only in EXIF. A field-only
                    // orientation (a caller-set field, or one from a format
                    // whose orientation is a container property) gets a
                    // minimal EXIF holding just the tag.
                    let mut exif = zencodec::exif::Exif::new(zencodec::exif::TextEncoding::Ascii);
                    exif.set_orientation(meta.orientation);
                    req = req.exif(Exif::raw(zenmeta::exif::serialize(&exif)));
                }
            } else if oriented {
                return Err(Error::encode_policy_rejected(
                    "the policy forbids EXIF, JPEG's only carrier for this image's orientation",
                ));
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
        if let Some(ref pulse) = self.pulse {
            req = req.stop(pulse);
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
        binding: &ColorBinding,
    ) -> Result<EncodeOutput, Error> {
        self.check_limits(width, height, layout)?;
        let req = self.build_request(layout, binding)?;
        let output = req.encode_bytes(data, width, height, layout)?;
        self.check_output_size(&output)?;
        Ok(EncodeOutput::new(output, ImageFormat::Jpeg))
    }

    /// One-shot encode of 16-bit sRGB-encoded pixels. Each strip is decoded
    /// to linear f32 ([`decode_srgb16`]) and pushed before the next, so the
    /// converted copy holds one strip, not the image. The bytes equal those of
    /// a single push of the whole converted image (no pre-blur applies to the
    /// f32 layouts).
    fn encode_srgb16(
        &self,
        pixels: &PixelSlice<'_>,
        layout: PixelLayout,
        binding: &ColorBinding,
    ) -> Result<EncodeOutput, Error> {
        let (width, height) = (pixels.width(), pixels.rows());
        self.check_limits(width, height, layout)?;
        let mut enc = self
            .build_request(layout, binding)?
            .encode_from_bytes(width, height, layout)?;
        let stop = self.stop_ref();
        let row_bytes = width as usize * layout.bytes_per_pixel();
        let strip = zencodec::encode::Encoder::preferred_strip_height(self);
        let mut linear = Vec::new();
        let mut y = 0;
        while y < height {
            let rows = strip.min(height - y);
            decode_srgb16(&pixels.sub_rows(y, rows), &mut linear)?;
            enc.push(&linear, rows as usize, row_bytes, stop)?;
            y += rows;
        }
        let output = enc.finish()?;
        self.check_output_size(&output)?;
        Ok(EncodeOutput::new(output, ImageFormat::Jpeg))
    }

    /// Create the native row encoder for a `width` x `height` image in
    /// `descriptor`, with the configuration as given. The first rows' colour
    /// context is the declaration when `begin` was not called.
    fn start_rows(
        &mut self,
        width: u32,
        height: u32,
        descriptor: PixelDescriptor,
        context: Option<&ColorContext>,
    ) -> Result<(), Error> {
        let layout = self.pixel_layout(descriptor)?;
        self.check_row_filters(layout)?;
        self.check_limits(width, height, layout)?;
        let binding = self.bind_color(descriptor, context, layout)?;
        let enc = self
            .build_request(layout, &binding)?
            .encode_from_bytes(width, height, layout)?;
        self.streaming_enc = Some(enc);
        self.row_descriptor = Some(descriptor);
        self.bound = Some(binding);
        Ok(())
    }

    /// Pre-blur filters the whole image before encoding, and only a packed
    /// RGB8 push applies it (one-shot and accumulated encodes). Row encodes
    /// cannot, so they refuse rather than write a file that differs from the
    /// one-shot encode of the same pixels.
    fn check_row_filters(&self, layout: PixelLayout) -> Result<(), Error> {
        if self.effective_config.pre_blur > 0.0 && layout == PixelLayout::Rgb8Srgb {
            return Err(Error::unsupported_feature(
                "pre-blur filters the whole image; encode it one-shot",
            ));
        }
        Ok(())
    }

    /// The usage paths are exclusive: no one-shot or pull encode once `begin`
    /// or `push_rows` has started a row encode.
    fn check_one_shot(&self) -> Result<(), Error> {
        if self.begun || self.streaming_enc.is_some() || self.accumulator.is_some() {
            return Err(Error::invalid_state(
                "encode after begin or push_rows: the usage paths are exclusive",
            ));
        }
        Ok(())
    }

    /// The output-size limit over everything produced so far: bytes sent
    /// plus `more` still to come.
    fn check_total_output(&self, more: usize) -> Result<(), Error> {
        let total = self.sent.saturating_add(more as u64);
        self.limits.check_output_size(total).map_err(|_| {
            Error::resource_limit_exceeded(
                zencodec::LimitKind::OutputSize,
                total,
                self.limits.max_output_bytes.unwrap_or(0),
            )
        })
    }

    /// Stream accumulated rows through the native BytesEncoder.
    fn encode_accumulated(&self, acc: RowAccumulator) -> Result<EncodeOutput, Error> {
        self.check_limits(acc.width, acc.total_rows, acc.layout)?;

        let binding = self
            .bound
            .as_ref()
            .ok_or_else(|| Error::internal("accumulated rows without a colour binding"))?;
        let req = self.build_request(acc.layout, binding)?;
        let stop = self.stop_ref();
        let mut enc = req.encode_from_bytes(acc.width, acc.total_rows, acc.layout)?;
        // Stream through native encoder — it processes MCU rows as they arrive
        enc.push_packed(&acc.data, stop)?;
        let output = enc.finish()?;
        self.check_output_size(&output)?;
        Ok(EncodeOutput::new(output, ImageFormat::Jpeg))
    }
}

impl JpegEncoder {
    /// Start the row encoder for a declaration: what `begin`, `begin_output`
    /// and `begin_declared` share. The native encoder's own lag is checked
    /// against the prepared plan before any byte is exposed.
    fn start_declared(
        &mut self,
        declaration: &zencodec::encode::EncodeDeclaration<'_>,
    ) -> Result<zencodec::encode::OutputTiming, At<CodecError>> {
        let (width, height, descriptor) = (
            declaration.width,
            declaration.height,
            declaration.descriptor,
        );
        if self.begun {
            return Err(Error::invalid_state("begin called twice").into());
        }
        if self.streaming_enc.is_some() || self.accumulator.is_some() {
            return Err(Error::invalid_state("begin after push_rows").into());
        }
        // Build and check everything before storing any of it: a failed
        // `begin` leaves the encoder as it was.
        let start = self.prepare_rows(declaration)?;
        let mut enc = self
            .build_request(start.layout, &start.binding)?
            .encode_from_bytes(width, height, start.layout)?;
        let native = self.timing(
            enc.can_stream_output().then(|| enc.output_lag_rows()),
            height,
        );
        if native != start.timing {
            return Err(Error::internal("the native row encoder disagrees with its plan").into());
        }
        if start.timing.streams() {
            enc.start_output()?;
            self.check_total_output(enc.pending_output().len())?;
        }
        self.image_size = Some((width, height));
        self.streaming_enc = Some(enc);
        self.row_descriptor = Some(descriptor);
        self.bound = Some(start.binding);
        self.checked_context = None;
        self.begun = true;
        Ok(start.timing)
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
        self.check_one_shot()?;
        let descriptor = pixels.descriptor();
        let layout = self.pixel_layout(descriptor)?;
        let binding = self.bind_color(descriptor, pixels.color_context().map(|c| &**c), layout)?;
        if srgb16_layout(descriptor).is_some() {
            return self
                .encode_srgb16(&pixels, layout, &binding)
                .map_err(Into::into);
        }
        let contiguous = pixels.contiguous_bytes();
        self.encode_bytes_inner(&contiguous, pixels.width(), pixels.rows(), layout, &binding)
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
        self.check_one_shot()?;
        if make_opaque {
            for chunk in data.as_chunks_mut::<4>().0 {
                chunk[3] = 255;
            }
        }
        let layout = PixelLayout::Rgba8Srgb;
        self.check_limits(width, height, layout)?;
        let binding = self.bind_color(PixelDescriptor::RGBA8_SRGB, None, layout)?;
        let req = self.build_request(layout, &binding)?;
        let stop = self.stop_ref();
        let stride_bytes = stride_pixels as usize * 4;
        let mut enc = req.encode_from_bytes(width, height, layout)?;
        enc.push(data, height as usize, stride_bytes, stop)?;
        let output = enc.finish()?;
        self.check_output_size(&output)?;
        Ok(EncodeOutput::new(output, ImageFormat::Jpeg))
    }

    /// Declare the image and start the row encoder with the configuration
    /// as given. Output streams when that configuration allows it:
    /// sequential, fixed or custom Huffman tables (`optimize_huffman(false)`),
    /// not XYB or RGB mode, no MPF gain-map index.
    /// The header, with its metadata segments, is pending at once; each MCU
    /// row follows as it is encoded, at most `2 x strip height - 1` rows
    /// behind (16-row strips for 4:2:0, so 31 rows). Other configurations
    /// report `AT_FINISH`.
    ///
    /// The header's colour carrier is decided here, before any row: rows
    /// pushed later must not carry a colour context that means something
    /// else. Use [`begin_declared`](Self::begin_declared) to declare that context.
    fn begin(
        &mut self,
        width: u32,
        height: u32,
        descriptor: PixelDescriptor,
    ) -> Result<zencodec::encode::OutputTiming, Self::Error> {
        self.start_declared(&zencodec::encode::EncodeDeclaration::new(
            width, height, descriptor,
        ))
    }

    /// Declare the image and the current colour context of the rows that
    /// will follow, then start the row encoder.
    ///
    /// The same as [`begin_output`](Self::begin_output), with the rows'
    /// colour declared before the header is written. The header carries that
    /// colour: the context's ICC profile verbatim, or a profile synthesized
    /// for its code points or for the primaries the descriptor asserts. Rows
    /// pushed later must mean the same thing. Returns the plan
    /// [`EncodeJob::plan_encode`](zencodec::encode::EncodeJob::plan_encode)
    /// reports for the same job and declaration.
    fn begin_declared(
        &mut self,
        declaration: &zencodec::encode::EncodeDeclaration<'_>,
    ) -> Result<zencodec::encode::EncodePlan, Self::Error> {
        Ok(zencodec::encode::EncodePlan::rows(
            self.start_declared(declaration)?,
        ))
    }

    /// JPEG never patches earlier bytes: forward and seek-back destinations
    /// get the same layout and the same timing as [`begin`](Self::begin).
    fn begin_output(
        &mut self,
        width: u32,
        height: u32,
        descriptor: PixelDescriptor,
        access: zencodec::output_patch::OutputAccess,
    ) -> Result<zencodec::encode::OutputTiming, Self::Error> {
        self.start_declared(
            &zencodec::encode::EncodeDeclaration::new(width, height, descriptor)
                .with_access(access),
        )
    }

    fn pending_output(&self) -> &[u8] {
        match &self.streaming_enc {
            Some(enc) => enc.pending_output(),
            None => &[],
        }
    }

    fn consume_output(&mut self, n: usize) {
        if let Some(enc) = self.streaming_enc.as_mut() {
            let n = n.min(enc.pending_output().len());
            enc.consume_output(n);
            self.sent += n as u64;
        }
    }

    fn take_output(&mut self, into: &mut Vec<u8>) {
        if let Some(enc) = self.streaming_enc.as_mut() {
            self.sent += enc.pending_output().len() as u64;
            enc.take_output(into);
        }
    }

    fn push_rows(&mut self, rows: PixelSlice<'_>) -> Result<(), Self::Error> {
        let desc = rows.descriptor();
        let layout = self.pixel_layout(desc)?;

        // Dimensions known (`begin` or `with_canvas_size`): push straight
        // into the native encoder, which encodes MCU rows as they fill.
        if let Some((img_w, img_h)) = self.image_size {
            if self.streaming_enc.is_none() {
                self.start_rows(img_w, img_h, desc, rows.color_context().map(|c| &**c))?;
            }
            self.check_rows_color(&rows, layout)?;
            if self.row_descriptor != Some(desc) {
                return Err(Error::invalid_state(
                    "push_rows: pixel format differs from the one declared",
                )
                .into());
            }
            let stop: &dyn enough::Stop = match self.pulse {
                Some(ref s) => s,
                None => &enough::Unstoppable,
            };
            let enc = self.streaming_enc.as_mut().expect("started above");
            if srgb16_layout(desc).is_some() {
                decode_srgb16(&rows, &mut self.srgb16_rows)?;
                enc.push(
                    &self.srgb16_rows,
                    rows.rows() as usize,
                    rows.width() as usize * layout.bytes_per_pixel(),
                    stop,
                )?;
            } else {
                // Use as_strided_bytes for zero-copy; BytesEncoder::push handles stride.
                enc.push(
                    rows.as_strided_bytes(),
                    rows.rows() as usize,
                    rows.stride(),
                    stop,
                )?;
            }
            let pending = enc.pending_output().len();
            self.check_total_output(pending)?;
            return Ok(());
        }

        // Fallback: accumulate rows (dimensions unknown).
        if self.accumulator.is_some() {
            self.check_rows_color(&rows, layout)?;
        }
        let width = rows.width();
        let contiguous;
        let data: &[u8] = if srgb16_layout(desc).is_some() {
            decode_srgb16(&rows, &mut self.srgb16_rows)?;
            &self.srgb16_rows
        } else {
            contiguous = rows.contiguous_bytes();
            &contiguous
        };
        match &mut self.accumulator {
            None => {
                self.bound =
                    Some(self.bind_color(desc, rows.color_context().map(|c| &**c), layout)?);
                let bpp = desc.bytes_per_pixel();
                let row_bytes = width as usize * bpp;
                let estimated_total = row_bytes * rows.rows() as usize * 4;
                let mut buf = Vec::new();
                buf.try_reserve(estimated_total)
                    .map_err(|_| Error::allocation_failed(estimated_total, "push_rows buffer"))?;
                buf.extend_from_slice(data);
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
                acc.data.extend_from_slice(data);
                acc.total_rows += rows.rows();
            }
        }
        Ok(())
    }

    fn finish(mut self) -> Result<EncodeOutput, Self::Error> {
        // A cancelled encode reports its stop reason, not the row count a
        // push cut short left behind.
        self.stop_ref().check().map_err(Error::from)?;
        // Row encoder: finish it. With streaming output this is only the
        // part not yet taken through `consume_output`.
        if let Some(enc) = self.streaming_enc.take() {
            let output = enc.finish()?;
            self.check_total_output(output.len())?;
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
        self.check_one_shot()?;

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
        let layout = self.pixel_layout(desc)?;
        self.check_row_filters(layout)?;
        self.check_limits(img_w, img_h, layout)?;
        let binding = self.bind_color(desc, None, layout)?;

        // The configuration as given: rows are encoded as each strip
        // arrives; optimized tables or progressive scans keep their blocks
        // until the end.
        let req = self.build_request(layout, &binding)?;
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
        // The native 16-bit layouts are linear light; sRGB-encoded 16-bit
        // samples are decoded to linear f32 first (`srgb16_layout`).
        (
            ChannelType::U16,
            ChannelLayout::Rgb,
            TransferFunction::Linear | TransferFunction::Unknown,
        ) => Ok(PixelLayout::Rgb16Linear),
        (
            ChannelType::U16,
            ChannelLayout::Rgba,
            TransferFunction::Linear | TransferFunction::Unknown,
        ) => Ok(PixelLayout::Rgba16Linear),
        (
            ChannelType::U16,
            ChannelLayout::Gray,
            TransferFunction::Linear | TransferFunction::Unknown,
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

/// The native layout 16-bit sRGB-encoded input is encoded through. No native
/// layout holds it as stored, so its rows are decoded to linear f32 on the
/// way in ([`decode_srgb16`]); the encoder's sRGB encoding then restores
/// them at f32 precision.
fn srgb16_layout(desc: PixelDescriptor) -> Option<PixelLayout> {
    use zenpixels::{ChannelLayout, ChannelType, TransferFunction};
    if desc.channel_type() != ChannelType::U16 || desc.transfer() != TransferFunction::Srgb {
        return None;
    }
    match desc.layout() {
        ChannelLayout::Rgb => Some(PixelLayout::RgbF32Linear),
        ChannelLayout::Rgba => Some(PixelLayout::RgbaF32Linear),
        ChannelLayout::Gray => Some(PixelLayout::GrayF32Linear),
        _ => None,
    }
}

/// Decode 16-bit sRGB-encoded rows to the linear f32 rows of
/// [`srgb16_layout`], packed, into `out`. Alpha is scaled, not decoded.
fn decode_srgb16(rows: &PixelSlice<'_>, out: &mut Vec<u8>) -> Result<(), Error> {
    use linear_srgb::default::{srgb_u16_to_linear_rgba_slice, srgb_u16_to_linear_slice};
    let desc = rows.descriptor();
    let samples = rows.width() as usize * desc.layout().channels();
    let rgba = desc.layout() == zenpixels::ChannelLayout::Rgba;
    let len = samples
        .checked_mul(4 * rows.rows() as usize)
        .ok_or_else(|| Error::allocation_failed(usize::MAX, "16-bit sRGB rows"))?;
    out.clear();
    out.try_reserve(len)
        .map_err(|_| Error::allocation_failed(len, "16-bit sRGB rows"))?;
    let mut encoded: Vec<u16> = Vec::new();
    let mut linear: Vec<f32> = Vec::new();
    encoded
        .try_reserve(samples)
        .and_then(|()| linear.try_reserve(samples))
        .map_err(|_| Error::allocation_failed(samples * 6, "16-bit sRGB row"))?;
    linear.resize(samples, 0.0);
    for y in 0..rows.rows() {
        encoded.clear();
        encoded.extend(
            rows.row(y)
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| u16::from_ne_bytes(*b)),
        );
        if rgba {
            srgb_u16_to_linear_rgba_slice(&encoded, &mut linear);
        } else {
            srgb_u16_to_linear_slice(&encoded, &mut linear);
        }
        for value in &linear {
            out.extend_from_slice(&value.to_ne_bytes());
        }
    }
    Ok(())
}

// ============================================================================
// Decode side: DecoderConfig → DecodeJob → Decoder / StreamingDecoder
// ============================================================================
