//! `EncoderConfig::auto_for` — content-adaptive config from a single image.
//!
//! Takes any [`PixelSlice`] plus a target [`Quality`] and returns an
//! `EncoderConfig` tuned for that image. The target can be metric-
//! native (`Quality::ApproxButteraugli(1.5)`,
//! `Quality::ApproxSsim2(82.0)`) — the variant tells `auto_for` both
//! the JPEG-q-scale to dial in *and* which oracle decision tree to
//! consult; or it can be a plain JPEG-q scalar (`Quality::ApproxJpegli`,
//! `Quality::ApproxMozjpeg`), in which case ssim2 is the default
//! optimization metric.
//!
//! ## Caller-side constraints
//!
//! Pass [`AutoForOptions`] to [`EncoderConfig::auto_for_with`] to
//! constrain what the dispatch is allowed to pick (XYB on/off,
//! sequential vs progressive, restart-marker density for parallel
//! decode, future iteration budget). [`EncoderConfig::auto_for`]
//! is the no-options shorthand and uses [`AutoForOptions::default`].
//!
//! # Status
//!
//! The full dispatch — generating an if/else tree from the oracle
//! `selector_tree_rules.json` — is gated on the `gen_auto_for.py`
//! codegen rewrite (see `auto_for_design.md`). Until that lands, this
//! function uses an analyzer-signal heuristic (chroma sharpness ⇒
//! subsampling, natural likelihood ⇒ sharp_yuv, derived likelihood ⇒
//! deringing). When the codegen lands, only the body of
//! [`auto_for_internal`] changes — the public signature is final.

use crate::analyze::{AnalyzerOutput, analyze};
use crate::encode::encoder_config::EncoderConfig;
use crate::encode::encoder_types::{ChromaSubsampling, ProgressiveScanMode, Quality};
use zenpixels::PixelSlice;

/// Caller-side capability + preference constraints for [`EncoderConfig::auto_for_with`].
///
/// Each field narrows what the dispatch is allowed to pick. Defaults
/// match what zenjpeg returns from a plain `EncoderConfig::ycbcr(q,
/// _)` call: progressive JPEG, no XYB, no restart markers, single-pass
/// encode. Override per call when you need broader compatibility,
/// faster decode, or a tighter quality target.
///
/// `#[non_exhaustive]` so future fields don't break callers — always
/// build via `AutoForOptions::default()` + builder methods.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct AutoForOptions {
    /// Allow the dispatch to select XYB color space when it's a clear
    /// quality win on a given image. **Default: `false`** — XYB still
    /// has decoder-compatibility gaps in the wild (some browsers,
    /// older mobile decoders, embedded systems). Set to `true` when
    /// you control the decoder side or are willing to accept fallback.
    pub allow_xyb: bool,

    /// Allow progressive scan ordering (multi-pass DCT, ~3-5% smaller
    /// at the cost of multi-pass decode + higher memory). **Default:
    /// `true`.** Set to `false` to force baseline (sequential) JPEG —
    /// useful when callers need single-pass, low-latency decode (e.g.
    /// hardware decoders, streaming use cases, GIF89a-style fast
    /// playback).
    pub allow_progressive: bool,

    /// Insert restart markers every `N` MCU rows. `0` disables restart
    /// markers entirely (smaller files, no parallel decode). Other
    /// values cost +0.02-0.16% bpp at row-aligned intervals (per
    /// zenjpeg's own measurements) and unlock parallel + error-recovery
    /// decode paths. **Default: `0`** (no markers).
    ///
    /// Note: progressive JPEG suppresses restart markers by default
    /// regardless of this setting (they cost ~10% in progressive
    /// mode for no benefit). Combine `allow_progressive: false` with
    /// `restart_mcu_rows: > 0` for the parallel-fast-decode shape.
    pub restart_mcu_rows: u16,

    /// Maximum search iterations the encoder is allowed to use for
    /// quality-targeting (a future BD-RD or zensim_iters loop). `0`
    /// disables iterative search entirely — single-pass encode at the
    /// dispatched config. Higher values cost more encode time but
    /// hit metric-native targets (butteraugli distance, ssim2 score)
    /// more tightly. **Default: `0`** (single-pass).
    ///
    /// Reserved for future use; currently a no-op in the heuristic
    /// dispatch. The signature is locked in now so callers can adopt
    /// it stably.
    pub max_iterations: u32,
}

impl Default for AutoForOptions {
    fn default() -> Self {
        Self {
            allow_xyb: false,
            allow_progressive: true,
            restart_mcu_rows: 0,
            max_iterations: 0,
        }
    }
}

impl AutoForOptions {
    /// Preset for fast / parallel decode: forces baseline (sequential)
    /// scan and inserts restart markers every 4 MCU rows. Trades ~3-5%
    /// bpp + 0.04% restart overhead for single-pass parallel-decodable
    /// output. Use when decoder throughput matters more than file size
    /// (e.g. server-side preview pipelines, embedded systems).
    #[must_use]
    pub const fn fast_decode() -> Self {
        Self {
            allow_xyb: false,
            allow_progressive: false,
            restart_mcu_rows: 4,
            max_iterations: 0,
        }
    }

    /// Preset for best metric quality regardless of decoder
    /// compatibility or encode time: enables XYB, allows progressive,
    /// no restart markers, up to 8 search iterations for
    /// metric-native targets. Use when you control the decoder and
    /// want the smallest file at a given target metric value.
    #[must_use]
    pub const fn best_quality() -> Self {
        Self {
            allow_xyb: true,
            allow_progressive: true,
            restart_mcu_rows: 0,
            max_iterations: 8,
        }
    }

    /// Builder: allow / forbid XYB color space.
    #[must_use]
    pub const fn allow_xyb(mut self, v: bool) -> Self {
        self.allow_xyb = v;
        self
    }

    /// Builder: allow / forbid progressive scan ordering.
    #[must_use]
    pub const fn allow_progressive(mut self, v: bool) -> Self {
        self.allow_progressive = v;
        self
    }

    /// Builder: set restart-marker MCU-row interval. `0` disables.
    #[must_use]
    pub const fn restart_mcu_rows(mut self, rows: u16) -> Self {
        self.restart_mcu_rows = rows;
        self
    }

    /// Builder: set the maximum search-iteration budget for
    /// metric-targeting. `0` = single-pass encode.
    #[must_use]
    pub const fn max_iterations(mut self, n: u32) -> Self {
        self.max_iterations = n;
        self
    }
}

impl EncoderConfig {
    /// Build a content-adaptive `EncoderConfig` for `image` targeting
    /// `quality`, using [`AutoForOptions::default`]. The `quality`
    /// argument carries both the target value (a JPEG-q scalar, an
    /// SSIMULACRA2 score, or a butteraugli distance) and the implicit
    /// optimization metric (via the `Quality` variant).
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// // Target butteraugli distance 1.5 — picks the butter-trained
    /// // tree internally, returns a config tuned for that metric.
    /// let config = EncoderConfig::auto_for(image, Quality::ApproxButteraugli(1.5))?;
    ///
    /// // Target SSIM2 score 82 — picks the ssim2-trained tree.
    /// let config = EncoderConfig::auto_for(image, Quality::ApproxSsim2(82.0))?;
    ///
    /// // Plain quality scalar — defaults to ssim2-optimized.
    /// let config = EncoderConfig::auto_for(image, 75.0)?;
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error string when `image`'s descriptor isn't
    /// convertible to RGB8 (e.g. CMYK without a CMS plugin loaded
    /// into `zenpixels-convert::RowConverter`).
    pub fn auto_for(image: PixelSlice<'_>, quality: impl Into<Quality>) -> Result<Self, String> {
        Self::auto_for_with(image, quality, AutoForOptions::default())
    }

    /// Like [`EncoderConfig::auto_for`] but with caller-side
    /// constraints — XYB permission, progressive permission, restart
    /// marker density, future iteration budget. See [`AutoForOptions`].
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// // Fast-decode pipeline: sequential JPEG with restart markers
    /// // every 4 MCU rows for parallel decoding.
    /// let cfg = EncoderConfig::auto_for_with(
    ///     image,
    ///     Quality::ApproxSsim2(82.0),
    ///     AutoForOptions::fast_decode(),
    /// )?;
    ///
    /// // Allow XYB but force baseline scan for compatibility.
    /// let cfg = EncoderConfig::auto_for_with(
    ///     image,
    ///     75.0,
    ///     AutoForOptions::default()
    ///         .allow_xyb(true)
    ///         .allow_progressive(false),
    /// )?;
    /// ```
    pub fn auto_for_with(
        image: PixelSlice<'_>,
        quality: impl Into<Quality>,
        options: AutoForOptions,
    ) -> Result<Self, String> {
        let quality = quality.into();
        let metric = AutoForMetric::from_quality(&quality);
        let features = analyze(image)?;
        Ok(auto_for_internal(&features, quality, metric, options))
    }
}

/// Which oracle tree the dispatch should consult. Inferred from the
/// `Quality` variant — metric-native targets pick their own tree,
/// plain JPEG-q targets default to ssim2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AutoForMetric {
    Ssim2,
    Butter,
}

impl AutoForMetric {
    fn from_quality(q: &Quality) -> Self {
        match q {
            Quality::ApproxButteraugli(_) => AutoForMetric::Butter,
            Quality::ApproxSsim2(_) | Quality::ApproxJpegli(_) | Quality::ApproxMozjpeg(_) => {
                AutoForMetric::Ssim2
            }
        }
    }
}

/// Heuristic dispatch. Replaced wholesale when the tree codegen lands
/// (see `auto_for_design.md`). Choices are conservative — they avoid
/// known regressions from the 2026-04-25 oracle (e.g. 4:4:4 on flat
/// content costs −1.0 to −1.35 zensim) without claiming to find the
/// frontier. Caller-side constraints from `options` clip the search
/// space (XYB off, sequential only, restart marker density).
fn auto_for_internal(
    features: &AnalyzerOutput,
    quality: Quality,
    _metric: AutoForMetric,
    options: AutoForOptions,
) -> EncoderConfig {
    // Subsampling: 4:4:4 only when chroma carries real detail.
    // Threshold matches the 2026-04-25 knob-eval finding that 4:4:4
    // is +0.76 zensim on HighDetail (high cb/cr peaks AND high
    // chroma_complexity) but harmful elsewhere.
    let high_chroma_detail = (features.cb_peak_sharpness > 10.0
        || features.cr_peak_sharpness > 10.0)
        && features.chroma_complexity > 0.05;
    let subsampling = if high_chroma_detail {
        ChromaSubsampling::None
    } else {
        ChromaSubsampling::Quarter
    };

    // sharp_yuv: helps natural / detailed content. Hurts text/screen.
    let sharp_yuv = features.natural_likelihood > 0.5
        && features.screen_content_likelihood < 0.4
        && features.text_likelihood < 0.4;

    // Progressive: universally a small bpp win at zero quality cost.
    // Skipped when caller forbids it (fast-decode pipelines).
    let scan_mode = if options.allow_progressive {
        ProgressiveScanMode::Progressive
    } else {
        ProgressiveScanMode::Baseline
    };

    // XYB: gated entirely on caller permission until the tree codegen
    // gives us the per-bucket signals to know when XYB is the win
    // (the 2026-04-25 oracle showed mixed XYB results — winner on
    // some PhotoNatural cells, regressor on others). Conservative:
    // don't pick it heuristically even when allowed.
    let _allow_xyb = options.allow_xyb; // wired when tree dispatch lands

    let cfg = EncoderConfig::ycbcr(quality, subsampling)
        .progressive(scan_mode)
        .sharp_yuv(sharp_yuv)
        .deringing(true)
        .restart_mcu_rows(options.restart_mcu_rows);

    // max_iterations: reserved for the future BD-RD / zensim_iters
    // loop. Currently a no-op since the heuristic dispatch is
    // single-pass; surfacing it now locks the API shape.
    let _max_iter = options.max_iterations;

    cfg
}

#[cfg(test)]
mod tests {
    use super::*;
    use zenpixels::PixelDescriptor;

    fn slice(rgb: &[u8], w: u32, h: u32) -> PixelSlice<'_> {
        PixelSlice::new(rgb, w, h, (w as usize) * 3, PixelDescriptor::RGB8_SRGB).unwrap()
    }

    #[test]
    fn flat_image_builds_config() {
        let rgb = vec![128u8; 64 * 64 * 3];
        let _cfg = EncoderConfig::auto_for(slice(&rgb, 64, 64), 75.0).unwrap();
    }

    #[test]
    fn butter_target_doesnt_panic() {
        let rgb = vec![64u8; 64 * 64 * 3];
        let _cfg =
            EncoderConfig::auto_for(slice(&rgb, 64, 64), Quality::ApproxButteraugli(1.5)).unwrap();
    }

    #[test]
    fn ssim2_target_doesnt_panic() {
        let rgb = vec![200u8; 32 * 32 * 3];
        let _cfg =
            EncoderConfig::auto_for(slice(&rgb, 32, 32), Quality::ApproxSsim2(82.0)).unwrap();
    }

    #[test]
    fn fast_decode_preset_round_trips() {
        use crate::encode::PixelLayout;
        use enough::Unstoppable;

        // Need ≥ 4 MCU rows so restart_mcu_rows=4 actually emits at
        // least one RST marker. At 4:2:0 the MCU is 16 px tall, so
        // a 96-row image gives 6 MCU rows = 1 restart segment + tail.
        let w: u32 = 96;
        let h: u32 = 96;
        let mut rgb = vec![0u8; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 3) as usize;
                rgb[i] = (x * 2) as u8;
                rgb[i + 1] = (y * 2) as u8;
                rgb[i + 2] = ((x ^ y) * 2) as u8;
            }
        }

        let cfg =
            EncoderConfig::auto_for_with(slice(&rgb, w, h), 75.0, AutoForOptions::fast_decode())
                .unwrap();
        let mut enc = cfg.encode_from_bytes(w, h, PixelLayout::Rgb8Srgb).unwrap();
        enc.push_packed(&rgb, Unstoppable).unwrap();
        let jpeg = enc.finish().unwrap();
        assert_eq!(&jpeg[0..2], &[0xFF, 0xD8]);
        assert_eq!(&jpeg[jpeg.len() - 2..], &[0xFF, 0xD9]);
        // Fast-decode preset enables RST markers — find at least one
        // RST byte (FF D0..D7) in the stream. Skip the SOI/SOS/data
        // header by scanning from offset 100.
        // RST markers are FF D0..D7. Scan for them or for the DRI
        // segment (FF DD <len> <interval>) that announces them.
        let has_dri = jpeg.windows(2).any(|w| w[0] == 0xFF && w[1] == 0xDD);
        let has_rst = jpeg
            .windows(2)
            .any(|w| w[0] == 0xFF && (0xD0..=0xD7).contains(&w[1]));
        assert!(
            has_dri || has_rst,
            "fast_decode preset should emit DRI segment or RST markers; jpeg.len()={}",
            jpeg.len()
        );
    }

    #[test]
    fn auto_for_round_trips_through_encoder() {
        use crate::encode::PixelLayout;
        use enough::Unstoppable;

        let w: u32 = 32;
        let h: u32 = 32;
        let mut rgb = vec![0u8; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 3) as usize;
                rgb[i] = (x * 8) as u8;
                rgb[i + 1] = (y * 8) as u8;
                rgb[i + 2] = ((x ^ y) * 8) as u8;
            }
        }

        let cfg = EncoderConfig::auto_for(slice(&rgb, w, h), 75.0).unwrap();
        let mut enc = cfg.encode_from_bytes(w, h, PixelLayout::Rgb8Srgb).unwrap();
        enc.push_packed(&rgb, Unstoppable).unwrap();
        let jpeg = enc.finish().unwrap();
        assert!(jpeg.len() > 100);
        assert_eq!(&jpeg[0..2], &[0xFF, 0xD8]);
        assert_eq!(&jpeg[jpeg.len() - 2..], &[0xFF, 0xD9]);
    }

    #[test]
    fn options_builder_chains() {
        let opts = AutoForOptions::default()
            .allow_xyb(true)
            .allow_progressive(false)
            .restart_mcu_rows(8)
            .max_iterations(4);
        assert!(opts.allow_xyb);
        assert!(!opts.allow_progressive);
        assert_eq!(opts.restart_mcu_rows, 8);
        assert_eq!(opts.max_iterations, 4);
    }

    #[test]
    fn presets_have_expected_shape() {
        let fast = AutoForOptions::fast_decode();
        assert!(!fast.allow_progressive);
        assert!(fast.restart_mcu_rows > 0);

        let best = AutoForOptions::best_quality();
        assert!(best.allow_xyb);
        assert!(best.allow_progressive);
        assert!(best.max_iterations > 0);
    }
}
