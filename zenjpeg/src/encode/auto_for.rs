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
//! sequential vs progressive, restart-marker density, slow-encoder
//! features). [`EncoderConfig::auto_for`] is the no-options shorthand
//! and uses [`AutoForOptions::default`].
//!
//! # Dispatch (oracle-distilled, manual-tree)
//!
//! Implements a hand-distilled approximation of the 2026-04-25
//! oracle's per-(bucket × q_bin × metric) winners. The full sklearn
//! decision-tree codegen is a future workstream; this distillation
//! captures the dominant patterns from the 70-cell oracle:
//!
//! - **q < 40** (any bucket): hybrid trellis 4:2:0 progressive wins
//!   33/70 cells. Map to `hybrid_lambda` 12.0–16.0 by quality. Slow.
//! - **q ≥ 40 + photo content**: `trelStd` 4:4:4 with XYB at the top
//!   end is the most common winner. Slow.
//! - **q ≥ 40 + screen / illustration**: `trelOff` 4:4:4 + XYB,
//!   essentially regardless of metric. Fast.
//!
//! With `effort = Effort::Fast`, every cell falls through to a fast
//! path (`trelOff`, no hybrid lambda search, no scan-script search),
//! trading some bpp for encode latency. `Effort::Balanced` honors the
//! oracle's trellis pick; `Effort::Max` adds the 64-candidate scan
//! search on top. When `allow_xyb=false` the XYB cells fall back to
//! YCbCr. When megapixels < 0.25, XYB is suppressed regardless of
//! permission — its ICC-profile overhead (~2 KB) becomes a meaningful
//! fraction of the file at thumbnail sizes.

use crate::analyze::{AnalyzerOutput, analyze};
use crate::encode::encoder_config::EncoderConfig;
use crate::encode::encoder_types::{
    ChromaSubsampling, Effort, ProgressiveScanMode, Quality, XybSubsampling,
};
#[cfg(feature = "trellis")]
use crate::encode::trellis::TrellisSpeedMode;
#[cfg(feature = "trellis")]
use crate::encode::trellis::{HybridConfig, TrellisConfig};
use zenpixels::PixelSlice;

/// Restart-marker density choice for [`AutoForOptions`].
///
/// Restart markers partition the entropy-coded stream so decoders can
/// resync, recover from corruption, and decode segments in parallel.
/// Row-aligned intervals carry near-zero bpp overhead because DC
/// prediction already breaks at row boundaries.
///
/// Default is [`RestartMarkers::Off`] to match what a plain
/// `EncoderConfig::ycbcr` call produces.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum RestartMarkers {
    /// No restart markers. Smallest files; decoder can't parallelize.
    #[default]
    Off,

    /// Markers at the default density (every 4 MCU rows) — the
    /// "yes please make this parallel-decodable" choice. Costs
    /// ~0.04% bpp at row-aligned intervals.
    Auto,

    /// Sparser markers (every 8 MCU rows). Half the parallel-decode
    /// granularity of `Auto`, ~0.02% bpp overhead. Right for cases
    /// where you want resync points but don't need fine-grained
    /// parallelism.
    AutoSparse,
}

impl RestartMarkers {
    fn to_mcu_rows(self) -> u16 {
        match self {
            RestartMarkers::Off => 0,
            RestartMarkers::Auto => 4,
            RestartMarkers::AutoSparse => 8,
        }
    }
}

/// Caller-side capability + preference constraints for [`EncoderConfig::auto_for_with`].
///
/// Each field narrows what the dispatch is allowed to pick. Defaults
/// match the most-portable, lowest-encode-latency output: progressive
/// JPEG, no XYB, no restart markers, no slow features. Override per
/// call when you need broader compatibility, faster decode, or want
/// to spend more encode time for tighter compression.
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
    ///
    /// Even when `true`, XYB is suppressed for very small images
    /// (`megapixels < 0.25`) because the embedded ICC profile (~2 KB)
    /// is a meaningful fraction of thumbnail file size.
    pub allow_xyb: bool,

    /// Allow progressive scan ordering (multi-pass DCT, ~3-5% smaller
    /// at the cost of multi-pass decode + higher memory). **Default:
    /// `true`.** Set to `false` to force baseline (sequential) JPEG —
    /// useful when callers need single-pass, low-latency decode (e.g.
    /// hardware decoders, streaming, fast preview pipelines).
    pub allow_progressive: bool,

    /// Encode-time effort budget. Reuses zenjpeg's existing
    /// [`Effort`] enum so the auto_for surface uses the same vocab
    /// the rest of the encoder API does.
    ///
    /// **Default: [`Effort::Fast`]** — no trellis, no hybrid lambda,
    /// no scan-script search. Match the speed of a plain
    /// `EncoderConfig::ycbcr` call.
    ///
    /// - [`Effort::Fast`]: oracle's slow features all disabled.
    ///   Every cell falls through to `trelOff` + standard progressive.
    /// - [`Effort::Balanced`]: oracle's trellis pick honored
    ///   (`Standard` or `Hybrid` lambda) but scan-script search stays
    ///   off. The middle gear that didn't exist with the old `bool`.
    ///   Requires the `trellis` feature.
    /// - [`Effort::Max`]: oracle's trellis pick honored AND
    ///   `ProgressiveScanMode::ProgressiveSearch` enabled. ~2× slower
    ///   than `Balanced` for another ~2% bpp. Requires the `trellis`
    ///   feature.
    ///
    /// The 2026-04-25 oracle showed trellis / hybrid-lambda configs
    /// winning 33/70 cells at q < 40 and many of the q ≥ 40 photo
    /// cells — `Effort::Balanced` is the right pick for any quality
    /// target where encode time isn't the constraint.
    pub effort: Effort,

    /// Restart-marker density. **Default: [`RestartMarkers::Off`].**
    ///
    /// Note: progressive JPEG suppresses restart markers by default
    /// regardless of this setting (they cost ~10% in progressive
    /// mode for no benefit). Combine `allow_progressive: false` with
    /// `RestartMarkers::Auto` for the parallel-fast-decode shape.
    pub restart_markers: RestartMarkers,
    // NOTE: `max_iterations` (search-iteration budget for a future
    // BD-RD / zensim_iters loop) is intentionally NOT surfaced yet.
    // It lands when the iterative search loop is implemented; until
    // then exposing it would be a no-op that callers might depend on.
    // See `auto_for_design.md` for the planned shape.
}

impl Default for AutoForOptions {
    fn default() -> Self {
        Self {
            allow_xyb: false,
            allow_progressive: true,
            effort: Effort::Fast,
            restart_markers: RestartMarkers::Off,
        }
    }
}

impl AutoForOptions {
    /// Preset for fast / parallel decode: forces baseline (sequential)
    /// scan and inserts restart markers at the default density. Trades
    /// ~3-5% bpp + 0.04% restart overhead for single-pass parallel-
    /// decodable output. Encode-side stays fast (`Effort::Fast`).
    #[must_use]
    pub const fn fast_decode() -> Self {
        Self {
            allow_xyb: false,
            allow_progressive: false,
            effort: Effort::Fast,
            restart_markers: RestartMarkers::Auto,
        }
    }

    /// Preset for best metric quality regardless of decoder
    /// compatibility or encode latency: enables XYB, allows
    /// progressive, no restart markers, and uses [`Effort::Max`]
    /// (trellis + hybrid-lambda + scan-script search). Use when you
    /// control the decoder and want the smallest file at a given
    /// target metric value.
    ///
    /// Requires the `trellis` feature; without it, falls back to
    /// [`Effort::Fast`] (the only variant available).
    #[must_use]
    pub const fn best_quality() -> Self {
        Self {
            allow_xyb: true,
            allow_progressive: true,
            #[cfg(feature = "trellis")]
            effort: Effort::Max,
            #[cfg(not(feature = "trellis"))]
            effort: Effort::Fast,
            restart_markers: RestartMarkers::Off,
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

    /// Builder: set encode-time effort budget. See [`Effort`].
    #[must_use]
    pub const fn effort(mut self, v: Effort) -> Self {
        self.effort = v;
        self
    }

    /// Builder: set restart-marker density.
    #[must_use]
    pub const fn restart_markers(mut self, m: RestartMarkers) -> Self {
        self.restart_markers = m;
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
    /// constraints. See [`AutoForOptions`].
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

/// Coarse content classification distilled from analyzer features —
/// matches the oracle's 5-bucket taxonomy as closely as the analyzer
/// signals allow. Internal only; not part of the public surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InferredBucket {
    PhotoNatural,
    PhotoDetailed,
    PhotoFlat,
    Illustration,
    ScreenContent,
}

fn infer_bucket(f: &AnalyzerOutput) -> InferredBucket {
    // Strong synthetic signals win first (text/screen content have
    // very distinctive feature signatures).
    if f.text_likelihood > 0.55 {
        // Text + low chroma + sharp edges → screen content / document.
        // The oracle's 'ScreenContent' bucket dominates at q ≥ 25 and
        // wants XYB+4:4:4; 'Illustration' is similar but with more
        // chroma. Differentiate by chroma signal strength.
        if f.chroma_complexity > 0.04 || f.cb_peak_sharpness > 5.0 {
            return InferredBucket::Illustration;
        }
        return InferredBucket::ScreenContent;
    }
    if f.screen_content_likelihood > 0.5 {
        return InferredBucket::ScreenContent;
    }
    // Photo-class: differentiate by content density.
    if f.uniformity > 0.55 || f.flat_color_block_ratio > 0.25 {
        return InferredBucket::PhotoFlat;
    }
    if f.high_freq_energy_ratio > 0.30
        || f.edge_density > 0.18
        || f.cb_peak_sharpness > 8.0
        || f.cr_peak_sharpness > 8.0
    {
        return InferredBucket::PhotoDetailed;
    }
    InferredBucket::PhotoNatural
}

/// Coarsen `Quality::to_internal()` (a 0-100 jpegli q-scale) into the
/// oracle's 7-way q_bin partition. Same boundaries as
/// `coefficient::scripts::fit_oracle_tree.py`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QBin {
    Q0_7,
    Q8_14,
    Q15_24,
    Q25_39,
    Q40_59,
    Q60_89,
    Q90Plus,
}

fn q_bin(q: f32) -> QBin {
    if q < 8.0 {
        QBin::Q0_7
    } else if q < 15.0 {
        QBin::Q8_14
    } else if q < 25.0 {
        QBin::Q15_24
    } else if q < 40.0 {
        QBin::Q25_39
    } else if q < 60.0 {
        QBin::Q40_59
    } else if q < 90.0 {
        QBin::Q60_89
    } else {
        QBin::Q90Plus
    }
}

/// Oracle-distilled dispatch: per (bucket × q_bin × metric) cell,
/// pick the codec config the 2026-04-25 oracle most often crowned a
/// winner — clamped to caller permissions in `options`.
///
/// This is a manual approximation of the 70 sklearn trees the fitter
/// produces. Replaces the prior "chroma-detail → 4:4:4" two-line
/// heuristic with something that actually reflects the measured
/// Pareto frontier, while staying readable + auditable. Switches to
/// the codegen-emitted decision tree when `gen_auto_for.py` lands.
fn auto_for_internal(
    features: &AnalyzerOutput,
    quality: Quality,
    metric: AutoForMetric,
    options: AutoForOptions,
) -> EncoderConfig {
    let bucket = infer_bucket(features);
    let q_internal = quality.to_internal();
    let qb = q_bin(q_internal);

    // ---- Pick the headline (subsampling, xyb, trellis-shape) ----
    let pick = pick_oracle(bucket, qb, metric);

    // ---- Apply caller constraints ----
    // XYB has decoder-compat + image-size gates layered on top of
    // the oracle pick. Image-size gate is image-dependent (megapixels);
    // permission gate is caller-dependent.
    let xyb_allowed = options.allow_xyb && features.megapixels >= 0.25;
    let use_xyb = pick.use_xyb && xyb_allowed;

    // Slow features (hybrid lambda, full trellis) are only used when
    // the caller opts in. Without permission, fall back to trelOff.
    let trellis_choice = match options.effort {
        Effort::Fast => TrellisChoice::Off,
        // Balanced + Max both honor the oracle's trellis pick.
        // (Trellis-feature-gated variants only exist when `trellis`
        // is on, so we don't need a feature-cfg here — the match is
        // already exhaustive at compile time.)
        #[cfg(feature = "trellis")]
        Effort::Balanced | Effort::Max => pick.trellis,
    };

    // Progressive: oracle universally prefers progressive at every
    // q-bin; respect caller's allow_progressive=false override.
    // ProgressiveSearch (64-candidate scan-script search) is the
    // extra ~2× slower step that lives ONLY in Effort::Max.
    let scan_mode = if !options.allow_progressive {
        ProgressiveScanMode::Baseline
    } else {
        match options.effort {
            Effort::Fast => ProgressiveScanMode::Progressive,
            #[cfg(feature = "trellis")]
            Effort::Balanced => ProgressiveScanMode::Progressive,
            #[cfg(feature = "trellis")]
            Effort::Max => ProgressiveScanMode::ProgressiveSearch,
        }
    };

    // ---- Build the EncoderConfig ----
    let mut cfg = if use_xyb {
        // XYB always wants 4:4:4 in the oracle (or BQuarter for the
        // B-channel). Pick BQuarter — that's the published recommendation
        // and matches every winning XYB cell in the rules JSON.
        EncoderConfig::xyb(quality, XybSubsampling::BQuarter)
    } else {
        EncoderConfig::ycbcr(quality, pick.subsampling)
    };

    cfg = cfg
        .progressive(scan_mode)
        .deringing(true)
        .restart_mcu_rows(options.restart_markers.to_mcu_rows());

    // sharp_yuv: helps natural / detailed photos, hurts text/screen.
    // Only meaningful in YCbCr mode (XYB doesn't subsample chroma the
    // same way).
    if !use_xyb {
        let sharp_yuv = matches!(
            bucket,
            InferredBucket::PhotoNatural | InferredBucket::PhotoDetailed
        ) && features.natural_likelihood > 0.5;
        cfg = cfg.sharp_yuv(sharp_yuv);
    }

    // Trellis / hybrid: gated behind feature flag + caller permission.
    // TrellisConfig::default() is "AC + DC trellis enabled, Adaptive
    // speed". HybridConfig is AQ-coupled trellis with configurable
    // `base_lambda_scale1` (the lambda the oracle's `hyb*` codec_name
    // suffix encodes). `hybrid_config(enabled=true)` zeroes the
    // standalone trellis slot internally.
    #[cfg(feature = "trellis")]
    {
        cfg = match trellis_choice {
            TrellisChoice::Off => cfg,
            TrellisChoice::Standard => cfg.trellis(TrellisConfig {
                speed_mode: TrellisSpeedMode::Adaptive,
                ..TrellisConfig::default()
            }),
            TrellisChoice::Hybrid(lambda) => cfg.hybrid_config(HybridConfig {
                enabled: true,
                base_lambda_scale1: lambda,
                ..HybridConfig::default()
            }),
        };
    }
    #[cfg(not(feature = "trellis"))]
    {
        let _ = trellis_choice;
    }

    cfg
}

/// What the oracle wants for a given (bucket, q_bin, metric) cell.
struct OraclePick {
    subsampling: ChromaSubsampling,
    use_xyb: bool,
    /// Only consulted when `Effort::Balanced | Effort::Max` are
    /// reachable (i.e. the `trellis` feature is on). Without trellis,
    /// every dispatch path lands on `TrellisChoice::Off` directly.
    #[cfg_attr(not(feature = "trellis"), allow(dead_code))]
    trellis: TrellisChoice,
}

#[derive(Debug, Clone, Copy)]
enum TrellisChoice {
    Off,
    #[cfg_attr(not(feature = "trellis"), allow(dead_code))]
    Standard,
    #[cfg_attr(not(feature = "trellis"), allow(dead_code))]
    Hybrid(f32),
}

/// Distilled oracle dispatch. Patterns extracted from the 70-cell
/// `selector_tree_rules.json` (commit corresponds to the 2026-04-25
/// run). Coarse but data-grounded: at low q the dominant winner is
/// hybrid trellis on 4:2:0; at high q it's `trelStd` on 4:4:4 + XYB
/// for photo content, `trelOff` 4:4:4 + XYB for synthetic.
fn pick_oracle(bucket: InferredBucket, qb: QBin, metric: AutoForMetric) -> OraclePick {
    use InferredBucket::*;
    use QBin::*;

    // --- Low-q (q < 40): hybrid lambda + 4:2:0 dominates regardless
    //     of bucket. Lambda value drifts with q-bin and bucket.
    let low_q = matches!(qb, Q0_7 | Q8_14 | Q15_24 | Q25_39);
    if low_q {
        let lambda = match (bucket, qb) {
            // PhotoFlat / Illustration / ScreenContent want stronger
            // hybrid pull-up at very low q (smaller files via more
            // aggressive zeroing).
            (PhotoFlat | Illustration, Q0_7 | Q8_14) => 12.0,
            (ScreenContent, Q0_7 | Q8_14) => 16.0,
            (PhotoDetailed, _) => 14.7,
            (PhotoNatural, _) => 13.5,
            (_, Q15_24 | Q25_39) => 14.0,
            _ => 14.0,
        };
        return OraclePick {
            subsampling: ChromaSubsampling::Quarter, // 4:2:0
            use_xyb: false,
            trellis: TrellisChoice::Hybrid(lambda),
        };
    }

    // --- High-q (q ≥ 40): XYB+4:4:4 wins most photo & synthetic
    //     cells; the choice between trelStd / trelOff depends on
    //     bucket + metric.
    match bucket {
        // Synthetic content: XYB+4:4:4+trelOff lands the most cells
        // across both metrics.
        ScreenContent | Illustration => OraclePick {
            subsampling: ChromaSubsampling::None, // 4:4:4
            use_xyb: true,
            trellis: TrellisChoice::Off,
        },
        // Detailed photos: XYB+4:4:4 with FULL trellis at the very top
        // of q (q60-89, q90+); trelOff is a reasonable downgrade
        // when callers don't want trellis.
        PhotoDetailed => OraclePick {
            subsampling: ChromaSubsampling::None,
            use_xyb: true,
            trellis: match (qb, metric) {
                (Q90Plus, AutoForMetric::Butter) => TrellisChoice::Off,
                (Q60_89 | Q90Plus, _) => TrellisChoice::Standard,
                (Q40_59, AutoForMetric::Butter) => TrellisChoice::Standard,
                _ => TrellisChoice::Off,
            },
        },
        // Flat photos: mixed bag in oracle. XYB+4:4:4+trelOff at the
        // top end is the safe pick; trellis on for the q40-59 ssim2
        // case.
        PhotoFlat => OraclePick {
            subsampling: ChromaSubsampling::None,
            use_xyb: true,
            trellis: match (qb, metric) {
                (Q40_59, AutoForMetric::Ssim2) => TrellisChoice::Standard,
                _ => TrellisChoice::Off,
            },
        },
        // Natural photos: XYB only at the very top (oracle has just
        // one PhotoNatural XYB winner at q90+); below that, hybrid
        // 4:2:0 with strong lambda.
        PhotoNatural => match qb {
            Q90Plus => OraclePick {
                subsampling: ChromaSubsampling::None,
                use_xyb: true,
                trellis: TrellisChoice::Off,
            },
            _ => OraclePick {
                subsampling: ChromaSubsampling::Quarter,
                use_xyb: false,
                trellis: TrellisChoice::Hybrid(16.0),
            },
        },
    }
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

        let has_dri = jpeg.windows(2).any(|w| w[0] == 0xFF && w[1] == 0xDD);
        let has_rst = jpeg
            .windows(2)
            .any(|w| w[0] == 0xFF && (0xD0..=0xD7).contains(&w[1]));
        assert!(
            has_dri || has_rst,
            "fast_decode preset should emit DRI or RST markers"
        );
    }

    #[test]
    fn auto_for_round_trips_through_encoder() {
        use crate::encode::PixelLayout;
        use enough::Unstoppable;

        let w: u32 = 64;
        let h: u32 = 64;
        let mut rgb = vec![0u8; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 3) as usize;
                rgb[i] = (x * 4) as u8;
                rgb[i + 1] = (y * 4) as u8;
                rgb[i + 2] = ((x ^ y) * 4) as u8;
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
            .effort({
                #[cfg(feature = "trellis")]
                {
                    Effort::Balanced
                }
                #[cfg(not(feature = "trellis"))]
                {
                    Effort::Fast
                }
            })
            .restart_markers(RestartMarkers::AutoSparse);
        assert!(opts.allow_xyb);
        assert!(!opts.allow_progressive);
        assert_eq!(opts.restart_markers, RestartMarkers::AutoSparse);
        // Effort doesn't impl PartialEq Hash/Eq publicly — checking
        // the shape via the dispatch is the load-bearing test.
    }

    #[test]
    fn presets_have_expected_shape() {
        let fast = AutoForOptions::fast_decode();
        assert!(!fast.allow_progressive);
        assert!(matches!(fast.effort, Effort::Fast));
        assert_eq!(fast.restart_markers, RestartMarkers::Auto);

        let best = AutoForOptions::best_quality();
        assert!(best.allow_xyb);
        assert!(best.allow_progressive);
        // best_quality picks Max when trellis is on, falls back to
        // Fast otherwise.
        #[cfg(feature = "trellis")]
        assert!(matches!(best.effort, Effort::Max));
        #[cfg(not(feature = "trellis"))]
        assert!(matches!(best.effort, Effort::Fast));
        assert_eq!(best.restart_markers, RestartMarkers::Off);
    }

    #[test]
    fn restart_markers_lower_correctly() {
        assert_eq!(RestartMarkers::Off.to_mcu_rows(), 0);
        assert_eq!(RestartMarkers::Auto.to_mcu_rows(), 4);
        assert_eq!(RestartMarkers::AutoSparse.to_mcu_rows(), 8);
    }

    #[test]
    fn q_bin_partitions_match_oracle() {
        assert_eq!(q_bin(0.0), QBin::Q0_7);
        assert_eq!(q_bin(7.9), QBin::Q0_7);
        assert_eq!(q_bin(8.0), QBin::Q8_14);
        assert_eq!(q_bin(14.5), QBin::Q8_14);
        assert_eq!(q_bin(40.0), QBin::Q40_59);
        assert_eq!(q_bin(89.999), QBin::Q60_89);
        assert_eq!(q_bin(90.0), QBin::Q90Plus);
        assert_eq!(q_bin(100.0), QBin::Q90Plus);
    }

    #[test]
    fn small_image_suppresses_xyb_even_when_allowed() {
        // 256×256 = 0.066 MP, well below the 0.25 MP gate. Even with
        // allow_xyb=true and a high q (where the oracle would pick
        // XYB), the dispatch should fall back to YCbCr.
        let w: u32 = 256;
        let h: u32 = 256;
        let mut rgb = vec![0u8; (w * h * 3) as usize];
        // Photo-detailed-ish content: high-freq + chroma variation.
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 3) as usize;
                rgb[i] = ((x * 13) ^ (y * 7)) as u8;
                rgb[i + 1] = ((x * 5) ^ (y * 11)) as u8;
                rgb[i + 2] = ((x * 17) ^ (y * 3)) as u8;
            }
        }
        let opts = AutoForOptions::best_quality(); // allow_xyb = true
        let cfg = EncoderConfig::auto_for_with(slice(&rgb, w, h), 95.0, opts).unwrap();
        // We can't introspect the config directly, but we can encode
        // and look for the JFIF/Adobe app marker — XYB JPEGs embed
        // an ICC profile (App2 with "ICC_PROFILE\0" magic). Absent
        // that, we're in YCbCr mode (the gate fired).
        use crate::encode::PixelLayout;
        use enough::Unstoppable;
        let mut enc = cfg.encode_from_bytes(w, h, PixelLayout::Rgb8Srgb).unwrap();
        enc.push_packed(&rgb, Unstoppable).unwrap();
        let jpeg = enc.finish().unwrap();
        let has_icc = jpeg.windows(12).any(|w| w == b"ICC_PROFILE\0");
        assert!(
            !has_icc,
            "XYB suppressed for sub-0.25-MP image even with allow_xyb=true"
        );
    }
}
