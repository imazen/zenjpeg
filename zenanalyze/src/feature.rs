//! Public API surface: a stable, opaque, set-composable feature
//! interface that lets multiple codecs share one analysis pass.
//!
//! **Stability contract: there is no 0.2.x.** Every change in 0.1.x
//! is additive — new variants on `#[non_exhaustive]` enums, new
//! parallel functions, never a signature change to a shipped item.
//! See `CLAUDE.md`.
//!
//! Stability contract:
//! - [`AnalysisFeature`] discriminants are **immutable once shipped**.
//!   A retired variant keeps its `u16` slot forever; new variants get
//!   the next sequential number. Never reuse a number — that would
//!   silently break callers persisting [`FeatureSet`] bits to disk or
//!   wire.
//! - [`FeatureSet`] storage is opaque. Future versions may grow the
//!   backing bitset; the public ops are `const fn` set math.
//! - [`FeatureValue`] is `#[non_exhaustive]` — future structured
//!   feature types (small histograms, vectors) can land via additional
//!   variants without a major bump.
//!
//! Composition pattern:
//! ```ignore
//! // Each codec exposes the features it cares about.
//! const JPEG_FEATURES: FeatureSet = FeatureSet::new()
//!     .with(AnalysisFeature::Variance)
//!     .with(AnalysisFeature::EdgeDensity)
//!     .with(AnalysisFeature::DctCompressibilityY);
//! const WEBP_FEATURES: FeatureSet = FeatureSet::new()
//!     .with(AnalysisFeature::Variance)
//!     .with(AnalysisFeature::AlphaPresent)
//!     .with(AnalysisFeature::AlphaBimodalScore);
//!
//! // Orchestrator unions and runs once.
//! let needed = JPEG_FEATURES.union(WEBP_FEATURES);
//! // analyze(slice, &AnalysisQuery::new(needed)) → AnalysisResults …
//! ```
//!
//! Design note — typed-feature path (deferred):
//!
//! A separate sealed-trait `Feature` + `IsActive` system was
//! prototyped alongside this enum, with one zero-sized struct per
//! feature (`pub struct Variance(_)`). It would let callers write
//! `results.get_typed::<Variance>() -> Option<f32>` (no runtime
//! variant match) and would make `FeatureSet::just_typed::<F>()`
//! refuse to compile when `F` is a retired feature. Deferred to a
//! later round so we can settle the dynamic surface first; the macro
//! that generates the witnesses is straightforward when the enum
//! variants are stable. **Action item before adding it:** decide
//! whether the 30 extra public type names are worth the compile-time
//! gating, or whether `#[deprecated]` warnings on the enum variant +
//! `None` at runtime is enough.

/// Single-source-of-truth macro: generates the [`AnalysisFeature`]
/// enum, its `id` / `from_u16` / `is_active` / `name` impls, the
/// internal [`RawAnalysis`] dense struct that the SIMD tiers write
/// into, the `RawAnalysis::into_results` translator, and the
/// [`FeatureSet::SUPPORTED`] preset — all in lockstep so adding a
/// feature is a one-line edit at the invocation site.
///
/// Per-feature row syntax:
/// `$(#[$attr:meta])* $Variant = $id : $type => $field`
/// The `$field` ident is the snake_case name (also used for
/// `AnalysisFeature::name`'s string return via `stringify!`).
macro_rules! features_table {
    (
        $(
            $(#[$variant_attr:meta])*
            $variant:ident = $id:literal : $ty:ty => $field:ident
        ),* $(,)?
    ) => {
        /// Stable identifier for every feature zenanalyze can compute.
        ///
        /// Discriminants are explicit sequential `u16` values that
        /// **must never change** once shipped. Adding a feature uses
        /// the next free number; retiring a feature keeps its number
        /// reserved (the variant stays for ABI stability). Persisted
        /// [`FeatureSet`]s round-trip through major versions correctly.
        ///
        /// `#[non_exhaustive]` so adding variants is not a breaking
        /// change; callers must use a `_` arm in any `match`.
        #[non_exhaustive]
        #[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
        #[repr(u16)]
        pub enum AnalysisFeature {
            $(
                $(#[$variant_attr])*
                $variant = $id,
            )*
            // Future features take the next free sequential id (30, 31, …).
        }

        impl AnalysisFeature {
            /// The feature's stable `u16` discriminant — the wire
            /// format. Sequential since 0, immutable once shipped.
            /// Retired ids stay reserved (their variant is removed
            /// from the enum but the number is never recycled). New
            /// variants take the next free number.
            ///
            /// Sidecars / Python fitters / JSON dumps that key
            /// features by stable id should read this.
            #[inline]
            pub const fn id(self) -> u16 {
                self as u16
            }

            /// Inverse of [`Self::id`]. Returns `None` for unknown
            /// numbers — including ids whose variants are retired
            /// or gated behind a cargo feature that isn't enabled
            /// in this build. Sidecars deserializing a wire-format
            /// id must accept the `None` arm gracefully (older /
            /// newer / cfg-disabled builds).
            #[inline]
            #[allow(unused_doc_comments)]
            pub const fn from_u16(n: u16) -> Option<Self> {
                // `unused_doc_comments` allow: variants in the
                // table carry `///` docstrings that the macro forwards
                // here so that `#[cfg(...)]` attrs stay in lockstep;
                // doc comments don't render on match arms but are
                // valid syntax there.
                match n {
                    $(
                        $(#[$variant_attr])*
                        $id => Some(Self::$variant),
                    )*
                    _ => None,
                }
            }

            /// Run-time check: is this feature *currently* computable
            /// in this build? Crate-internal — public callers learn
            /// "feature is missing" by getting `None` from
            /// [`AnalysisResults::get`].
            #[inline]
            pub(crate) const fn is_active(self) -> bool {
                // Today every variant in this build is active. The
                // experimental cargo feature gates whole variants out
                // at compile time, so a variant existing implies it's
                // active. Variant-level deprecation that keeps the
                // variant in the enum but marks it inactive would
                // wire its arm here.
                match self {
                    _ => true,
                }
            }

            /// Stable, machine-readable name (snake_case). Matches
            /// the field names in the internal [`RawAnalysis`] so
            /// JSON sidecars and downstream Python fitters keep
            /// working. Generated from the field-name token via
            /// `stringify!`.
            #[allow(unused_doc_comments)]
            pub const fn name(self) -> &'static str {
                match self {
                    $(
                        $(#[$variant_attr])*
                        Self::$variant => stringify!($field),
                    )*
                }
            }
        }

        impl FeatureSet {
            /// The set of [`AnalysisFeature`]s this build can compute.
            ///
            /// Built by walking the `features_table!` rows in order;
            /// per-row `#[cfg(feature = "...")]` propagates here, so
            /// disabling a cargo feature shrinks `SUPPORTED`
            /// automatically. Always intersect a caller's wish-list
            /// against `FeatureSet::SUPPORTED` before passing to
            /// [`AnalysisQuery`] — asking for an unsupported feature
            /// isn't an error, just yields `None` from
            /// [`AnalysisResults::get`].
            ///
            /// Use this rather than enumerating every variant by hand.
            /// It's the only "all features" entry point in the public
            /// API; there is intentionally no `FeatureSet::all()`
            /// because production callers should request only what
            /// they need.
            #[allow(unused_doc_comments, unused_mut, unused_assignments)]
            pub const SUPPORTED: Self = {
                // Const block with `let mut` so individual `with`
                // calls can be cfg-gated. The chain form
                // `Self::new().with(X).with(Y)` doesn't allow
                // attributes on individual method calls; this form
                // does. Per-row doc comments forward here too — they
                // don't render on a block expression but are valid
                // syntax. `unused_mut` covers the
                // `experimental`-fully-disabled build where every
                // gated arm vanishes.
                let mut s = Self::new();
                $(
                    $(#[$variant_attr])*
                    {
                        s = s.with(AnalysisFeature::$variant);
                    }
                )*
                s
            };
        }

        /// Dense flat record the SIMD tiers write to with zero
        /// overhead. Each SIMD inner loop already produces a single
        /// `f32`/`u32`/`bool` per feature; storing those into named
        /// struct fields is one mov per feature and lets LLVM keep
        /// accumulators in registers across the loop boundary.
        ///
        /// `pub(crate)` only — never crosses the public API.
        /// Translated once at the end of `analyze_features` into the
        /// sparse [`AnalysisResults`] based on the caller's requested
        /// [`FeatureSet`] (only requested features are copied).
        /// Default = zero / `false` for every field. Cfg-gated rows in
        /// the table become cfg-gated fields here.
        ///
        /// Generated from the [`features_table!`] invocation in
        /// lockstep with [`AnalysisFeature`] and
        /// [`FeatureSet::SUPPORTED`].
        #[derive(Default, Debug, Clone, Copy)]
        pub(crate) struct RawAnalysis {
            $(
                $(#[$variant_attr])*
                pub $field: $ty,
            )*
        }

        impl RawAnalysis {
            /// Translate this dense struct into the sparse public
            /// results, copying **only** the features the caller
            /// asked for. Iteration is in id order, so
            /// [`AnalysisResults`]' insertion-sort hits the
            /// append-at-end fast path on every call. Cfg-gated rows
            /// drop out of the copy list when their feature is
            /// disabled.
            #[allow(unused_doc_comments)]
            pub(crate) fn into_results(
                self,
                requested: FeatureSet,
                geometry: ImageGeometry,
                source_descriptor: zenpixels::PixelDescriptor,
            ) -> AnalysisResults {
                let mut r = AnalysisResults::new(requested, geometry, source_descriptor);
                $(
                    $(#[$variant_attr])*
                    {
                        if requested.contains(AnalysisFeature::$variant) {
                            r.set(AnalysisFeature::$variant, self.$field);
                        }
                    }
                )*
                r
            }
        }

    };
}

// ---------------------- Single source of truth -----------------------
// Editing this table updates the AnalysisFeature enum, the
// `id`/`from_u16`/`is_active`/`name` impls, the RawAnalysis dense
// record, the `into_results` translator, and FeatureSet::SUPPORTED —
// all in lockstep. Per-row format:
//   /// docstring
//   Variant = id : type => raw_field_name,
//
// The Rust `$ty` and the snake-case field name double as the
// AnalysisResults value type and the `name()` string. Adding a
// feature here is the only edit needed inside feature.rs.

features_table! {
    // ---------------- Tier 1: sparse stripe scan (pixel_budget) ------
    /// `f32`. Luma variance on the BT.601 [0, 255] scale.
    Variance = 0 : f32 => variance,
    /// `f32`. Fraction of sampled interior pixels with `|∇L| > 20`.
    EdgeDensity = 1 : f32 => edge_density,
    /// `f32`. `√(Var(Cb) + Var(Cr))` over sampled pixels.
    ChromaComplexity = 2 : f32 => chroma_complexity,
    /// `f32`. Mean `|∇Cb|` over horizontally-paired sampled pixels.
    CbSharpness = 3 : f32 => cb_sharpness,
    /// `f32`. Mean `|∇Cr|` over horizontally-paired sampled pixels.
    CrSharpness = 4 : f32 => cr_sharpness,
    /// `f32`. Fraction of 8×8 blocks with luma variance < 25.
    Uniformity = 5 : f32 => uniformity,
    /// `f32`. Fraction of 8×8 blocks with R, G, B ranges all ≤ 4.
    FlatColorBlockRatio = 6 : f32 => flat_color_block_ratio,
    /// `f32`. Hasler-Süsstrunk M3 colourfulness.
    #[cfg(feature = "experimental")]
    Colourfulness = 7 : f32 => colourfulness,
    /// `f32`. Variance of the 5-tap Laplacian over sampled luma.
    #[cfg(feature = "experimental")]
    LaplacianVariance = 8 : f32 => laplacian_variance,
    /// `f32`. `log10(1 + max_var / max(1, mean_var))` over 8×8 blocks.
    #[cfg(feature = "experimental")]
    VarianceSpread = 9 : f32 => variance_spread,

    // ---------------- Palette: always full-scan ----------------------
    /// `u32`. Distinct 5-bit-per-channel RGB bins observed.
    DistinctColorBins = 10 : u32 => distinct_color_bins,
    // id 11 reserved (was `DistinctColorBinsChao1`, removed pre-0.1.0).
    // Today's full-pass scan made the Chao1 correction term degenerate
    // — it always returned the same value as `DistinctColorBins`, with
    // no path to diverge until a budget-sampled palette variant is
    // implemented. Keeping a permanently-equal field invited callers
    // to write `if chao1 > distinct` checks that would silently break
    // when the divergence-producing path lands. The id is reserved so
    // the future budget-sampled variant can re-introduce the variant
    // at the same number for FeatureSet wire-format compatibility.
    /// `f32`. `DistinctColorBins / min(pixel_count, 32 768)`.
    #[cfg(feature = "experimental")]
    PaletteDensity = 12 : f32 => palette_density,

    // ---------------- Tier 2: per-channel per-axis chroma ------------
    /// `f32`. Cb horizontal gradient energy / 1e5.
    CbHorizSharpness = 13 : f32 => cb_horiz_sharpness,
    /// `f32`. Cb vertical gradient energy / 1e5.
    CbVertSharpness = 14 : f32 => cb_vert_sharpness,
    /// `f32`. Cb peak gradient magnitude. Calibrated so natural
    /// photographic content lands `< 100`; saturated synthetic
    /// inputs (alternating-channel chroma stripes) can reach
    /// `~666`. Renormalising onto a stable `[0, 100]` ceiling is a
    /// follow-up calibration task — until then, code that wants
    /// "is this peak unusually high?" should compare against a
    /// content-class-appropriate threshold rather than clamping.
    CbPeakSharpness = 15 : f32 => cb_peak_sharpness,
    /// `f32`. Cr horizontal gradient energy / 1e5.
    CrHorizSharpness = 16 : f32 => cr_horiz_sharpness,
    /// `f32`. Cr vertical gradient energy / 1e5.
    CrVertSharpness = 17 : f32 => cr_vert_sharpness,
    /// `f32`. Cr peak gradient magnitude. Same calibration story as
    /// [`Self::CbPeakSharpness`] — natural photographs `< 100`,
    /// saturated synthetic content up to `~666`.
    CrPeakSharpness = 18 : f32 => cr_peak_sharpness,

    // ---------------- Tier 3: DCT energy + entropy -------------------
    /// `f32`. `Σ AC[k≥16] / Σ AC[k∈1..16]` over sampled 8×8 luma blocks.
    HighFreqEnergyRatio = 19 : f32 => high_freq_energy_ratio,
    /// `f32`. Shannon entropy of a 32-bin luma histogram, in bits.
    LumaHistogramEntropy = 20 : f32 => luma_histogram_entropy,
    /// `f32`. Mean libwebp α on sampled luma 8×8 DCT blocks. Higher
    /// = harder to compress (more spread AC, fewer near-zero coefs).
    /// **Range:** theoretical `[0, ~8064]` from `256 * last_non_zero
    /// / max_count`; on real photo corpora the median sits at ~16,
    /// p90 at ~30. Downstream calibration must NOT clamp / normalise
    /// against 255 (earlier docs were wrong about that).
    #[cfg(feature = "experimental")]
    DctCompressibilityY = 21 : f32 => dct_compressibility_y,
    /// `f32`. Same shape on chroma DCT blocks, `max(α_cb, α_cr)` per
    /// block. Same `[0, ~8064]` theoretical range; chroma values run
    /// lower in practice (median ~5 on photos). Scale matches
    /// [`Self::DctCompressibilityY`].
    #[cfg(feature = "experimental")]
    DctCompressibilityUV = 22 : f32 => dct_compressibility_uv,
    /// `f32`. Fraction `[0, 1]` of sampled blocks matching another.
    /// Strongest single photo-vs-screen-content discriminator on
    /// real corpora.
    ///
    /// **Empirical AUC = 0.880** for screen-vs-photo on a 219-image
    /// labeled corpus (cid22 + clic2025 + gb82 + gb82-sc + imageflow
    /// + kadid10k + qoi-benchmark) — higher than every other shipped
    /// feature, including the derived [`Self::ScreenContentLikelihood`]
    /// (AUC 0.83). Photos: p50 = 0.002, p90 = 0.037. Screens: p50 =
    /// 0.726, p90 = 0.906. **Recommended operating threshold:
    /// `patch_fraction >= 0.27`** (F1 = 0.769, P = 0.91, R = 0.67)
    /// for screen-like classification.
    ///
    /// Originally validated on a smaller 50-photo / 10-screen pilot
    /// corpus where the default-budget classifier scored ROC-AUC =
    /// 0.978 (vs 0.980 at dense scan) and Spearman ρ between
    /// default and dense = 0.887. The 219-image labeled-corpus AUC
    /// is lower because that corpus includes screen-like
    /// illustrations and edge-case "uniform photos" that genuinely
    /// sit closer to the boundary; the rank order remains correct
    /// and the 1024-block default budget is fit for codec dispatch
    /// as shipped.
    ///
    /// Bumping `hf_max_blocks` to 4096+ tightens absolute error
    /// further but doesn't materially improve the classifier — see
    /// `docs/calibration-corpus-2026-04-27.md` for the full
    /// per-class distribution and AUC ranking.
    ///
    /// Gated behind `experimental` for 0.1.0 because no in-tree
    /// codec consumes it yet; promote when a consumer wires up.
    #[cfg(feature = "experimental")]
    PatchFraction = 23 : f32 => patch_fraction,

    // ---------------- Alpha (always full-scan when present) ----------
    /// `bool`. True iff the source has straight (unassociated) alpha.
    AlphaPresent = 24 : bool => alpha_present,
    /// `f32`. Fraction of sampled pixels with alpha < 255.
    AlphaUsedFraction = 25 : f32 => alpha_used_fraction,
    /// `f32`. Bimodal-ness of the alpha histogram, `[0, 0.5]`.
    AlphaBimodalScore = 26 : f32 => alpha_bimodal_score,

    // ---------------- Derived likelihoods ----------------------------
    /// `f32`. Soft score: rendered text / document content.
    ///
    /// Theoretical range `[0, 1]`. **Empirical max on a 219-image
    /// labeled corpus: 0.71** — the formula's three sub-components
    /// (low entropy + high edge density + low chroma) don't all max
    /// simultaneously on real content. Recommended operating
    /// threshold: `text_likelihood >= 0.30` (F1 = 0.682, AUC = 0.774
    /// for screen-like classification). Do not threshold at `>= 0.8`
    /// — nothing fires there. See
    /// `docs/calibration-corpus-2026-04-27.md`.
    #[cfg(feature = "composites")]
    TextLikelihood = 27 : f32 => text_likelihood,
    /// `f32`. Soft score: UI / chart / synthetic content.
    ///
    /// Theoretical range `[0, 1]`. **Empirical max on a 219-image
    /// labeled corpus: 0.70** — typical screen content has flat
    /// blocks and low chroma (which drive the formula's 0.6 + 0.1
    /// weights to 1) but a moderate distinct-color count (>= 4000
    /// bins) which forces the `palette_small` term to 0, capping the
    /// total at ~0.70. Recommended operating threshold:
    /// `screen_content_likelihood >= 0.60` (F1 = 0.750, P = 0.86,
    /// R = 0.67). For the strongest single screen-vs-photo
    /// discriminator, see [`Self::PatchFraction`] (AUC = 0.88) which
    /// outperforms this derived likelihood (AUC = 0.83). See
    /// `docs/calibration-corpus-2026-04-27.md`.
    #[cfg(feature = "composites")]
    ScreenContentLikelihood = 28 : f32 => screen_content_likelihood,
    /// `f32`. Soft score: natural photographic content.
    ///
    /// Theoretical range `[0, 1]`. **Empirical max on a 219-image
    /// labeled corpus: 0.69**, and photos cleanly separate from
    /// screens at a low threshold: `natural_likelihood >= 0.06`
    /// gives F1 = 0.924, P = 0.876, R = 0.977 for photo
    /// classification. Equivalent to a probability — high values
    /// reliably mean photo. See
    /// `docs/calibration-corpus-2026-04-27.md`.
    #[cfg(feature = "composites")]
    NaturalLikelihood = 29 : f32 => natural_likelihood,

    // ---------------- Quick-path palette signals --------------------
    /// `u32`. Smallest power-of-2 indexed-palette **bit-width** that
    /// fits the source's distinct 5-bit-per-channel RGB bins, encoded
    /// as the bit count:
    ///
    /// - `2`  ⇒ ≤ 4 colours      (1 BPP indexed, 2 BPP file)
    /// - `4`  ⇒ ≤ 16 colours     (4 BPP indexed)
    /// - `8`  ⇒ ≤ 256 colours    (8 BPP indexed — GIF, PNG-PLTE, WebP-lossless)
    /// - `0`  ⇒ > 256 colours    (truecolor required)
    ///
    /// `u32`-typed for storage compatibility with the [`FeatureValue`]
    /// enum's existing `U32` variant; the value range is `{0, 2, 4, 8}`.
    ///
    /// Computed via an early-exit scan that bails as soon as the
    /// running distinct-count exceeds 256. The scan reuses the same
    /// `#[autoversion]` + `chunks_exact(24)` skeleton as the full
    /// `DistinctColorBins` scan, plus a per-pixel running-count check.
    ///
    /// **Net win for paletteable detection in mixed workloads.**
    /// Measured at 8 MP, Ryzen 9 7950X, runtime archmage dispatch,
    /// in-context through `analyze_features_rgb8`:
    ///
    /// | Image | `DistinctColorBins` (full) | This (quick) | Ratio |
    /// |-------|---------------------------:|-------------:|------:|
    /// | Truecolor (>22 K bins) | 6.65 ms | **3.23 ms** | **2.06×** |
    /// | Small palette (16 colours) | 6.08 ms | 6.44 ms | 0.94× |
    /// | Solid (1 colour) | 6.40 ms | 7.08 ms | 0.90× |
    ///
    /// The remaining 6–10 % loss on small-palette content is per-pixel
    /// branch overhead the unconditional full scan doesn't pay; on
    /// truecolor content the early-exit at the 257th distinct bin
    /// dwarfs that. For codec orchestrators that don't know the
    /// content class up front, this is a net win because web-typical
    /// traffic skews truecolor.
    ///
    /// Drives GIF / PNG-indexed / WebP-lossless palette-mode decisions.
    /// Gated behind `experimental` for 0.1.0 because no in-tree
    /// indexed-codec consumer exists yet.
    #[cfg(feature = "experimental")]
    IndexedPaletteWidth = 30 : u32 => indexed_palette_width,

    /// `bool`. Convenience shorthand: `IndexedPaletteWidth != 0` —
    /// the source fits in 256 colours and an indexed-mode codec can
    /// represent it without quantization. Drives the binary
    /// "encode as indexed?" decision when the caller doesn't care
    /// about the exact width. Gated behind `experimental`.
    #[cfg(feature = "experimental")]
    PaletteFitsIn256 = 31 : bool => palette_fits_in_256,

    // ---------------- Depth tier (source-direct HDR / bit-depth) -----
    /// `f32`. Peak luminance over sampled pixels in nits. Computed
    /// against source samples directly (no `RowConverter` tonemap),
    /// honoring the descriptor's transfer function. SDR sources hit
    /// ~80 nits; PQ ~10 000; HLG ~1 000.
    #[cfg(feature = "experimental")]
    PeakLuminanceNits = 32 : f32 => peak_luminance_nits,
    /// `f32`. 99th-percentile luminance in nits (robust against single
    /// hot pixels).
    #[cfg(feature = "experimental")]
    P99LuminanceNits = 33 : f32 => p99_luminance_nits,
    /// `f32`. HDR headroom in stops: `log2(peak_nits / 80)`. SDR ⇒ 0;
    /// 1 000-nit HLG ⇒ ~3.6; 10 000-nit PQ ⇒ ~6.97.
    #[cfg(feature = "experimental")]
    HdrHeadroomStops = 34 : f32 => hdr_headroom_stops,
    /// `f32`. Fraction `[0, 1]` of sampled pixels above 100 nits.
    #[cfg(feature = "experimental")]
    HdrPixelFraction = 35 : f32 => hdr_pixel_fraction,
    /// `f32`. Largest single-channel linear value across sampled
    /// pixels. `> 1.0` ⇒ source carries above-sRGB values that would
    /// clip if narrowed to sRGB primaries.
    #[cfg(feature = "experimental")]
    WideGamutPeak = 36 : f32 => wide_gamut_peak,
    /// `f32`. Fraction `[0, 1]` of sampled pixels with at least one
    /// channel above 1.0 in linear light.
    #[cfg(feature = "experimental")]
    WideGamutFraction = 37 : f32 => wide_gamut_fraction,
    /// `u32`. Effective bit depth: smallest power-of-2 quantization
    /// grid the sampled values populate. `{8, 10, 12, 14, 16, 32}`.
    /// For u8 sources always 8; u8-promoted u16 detected as 8 via the
    /// low-byte distinct-count probe.
    #[cfg(feature = "experimental")]
    EffectiveBitDepth = 38 : u32 => effective_bit_depth,
    /// `bool`. `true` iff peak luminance well exceeds the SDR threshold
    /// AND the source transfer function can carry HDR
    /// (`Pq` / `Hlg` / `Linear`). Catches the hard case the standard
    /// tiers miss: PQ-encoded content whose tonemapped rendition looks
    /// like SDR but whose source carries far more dynamic range.
    #[cfg(feature = "experimental")]
    HdrPresent = 39 : bool => hdr_present,
    /// `f32`. Fraction `[0, 1]` of sampled pixels whose linear-RGB,
    /// projected from the source primaries into BT.709 / sRGB, has
    /// every channel within `[-ε, 1 + ε]`. **Descriptor-gap signal:**
    /// `1.0` ⇒ the source declares wider primaries (P3 / Rec.2020 /
    /// AdobeRGB) but its pixels actually live in the sRGB sub-gamut,
    /// so codecs can encode it with sRGB primaries and save bits on
    /// the colour-metadata + drop the gamut-extended encoder modes.
    /// For sRGB-declared sources this is trivially `1.0`. Threshold
    /// of `≥ 0.99` is a reasonable "downcast safe" cutoff.
    #[cfg(feature = "experimental")]
    GamutCoverageSrgb = 46 : f32 => gamut_coverage_srgb,
    /// `f32`. Same shape, projecting into Display P3. **Descriptor-
    /// gap signal:** for a Rec.2020-declared source, `1.0` here means
    /// the content is encodable in P3 (smaller container than
    /// Rec.2020). Useful as a middle tier when the image isn't sRGB-
    /// safe but doesn't actually use the full Rec.2020 gamut either.
    #[cfg(feature = "experimental")]
    GamutCoverageP3 = 47 : f32 => gamut_coverage_p3,
    /// `f32`. Fraction `[0, 1]` of sampled luma 8×8 blocks where
    /// ≥ 90 % of AC energy lives in the lowest-zigzag positions —
    /// **smooth-content / gradient signal**. Drives JXL
    /// `with_force_strategy` (DCT16 / DCT32 selection — large
    /// transforms pay off when most energy is in the lowest
    /// frequencies) and zenrav1e deblock-strength scaling. Distinct
    /// from `high_freq_energy_ratio` (global mean): this is per-
    /// block-thresholded, robust to a few high-detail blocks dragging
    /// the mean.
    #[cfg(feature = "experimental")]
    GradientFraction = 48 : f32 => gradient_fraction,

    // ---------------- Tier 1 piggyback: cheap secondary signals -----
    /// `f32`. Fraction `[0, 1]` of sampled pixels whose
    /// `max(|R-G|, |G-B|, |R-B|) ≤ 4`. ≥ 0.99 ⇒ effectively grayscale.
    /// Drives zenjpeg `ColorMode::Grayscale`, png/avif/jxl single-
    /// channel encode paths.
    #[cfg(feature = "experimental")]
    GrayscaleScore = 40 : f32 => grayscale_score,

    // ---------------- Tier 3 piggyback: AQ-map signals --------------
    /// `f32`. Mean of `log10(1 + Σ AC²)` over sampled luma 8×8 blocks.
    /// Image-average busyness signal; AQ orchestrators read this for
    /// the global "how textured is the image" baseline.
    #[cfg(feature = "experimental")]
    AqMapMean = 41 : f32 => aq_map_mean,
    /// `f32`. Standard deviation of the same per-block log-AC-energy.
    /// Drives zenjpeg hybrid trellis lambda scaling, webp
    /// segments+sns_strength, avif vaq_strength — high std ⇒
    /// heterogeneous content where AQ pays off.
    #[cfg(feature = "experimental")]
    AqMapStd = 42 : f32 => aq_map_std,
    /// `f32`. Robust luma noise floor estimate, normalized to
    /// `[0, 1]`. 10th percentile of √(low-AC-energy / 15) across
    /// sampled luma 8×8 blocks ÷ 32. Drives zenjpeg `pre_blur`,
    /// jxl `noise/denoise`, webp `sns_strength`.
    #[cfg(feature = "experimental")]
    NoiseFloorY = 43 : f32 => noise_floor_y,
    /// `f32`. Same on chroma — `max(p10_cb, p10_cr)`. Drives chroma-
    /// channel denoise scheduling.
    #[cfg(feature = "experimental")]
    NoiseFloorUV = 44 : f32 => noise_floor_uv,
    /// `f32`. Soft `[0, 1]` score: rendered line art / engineering
    /// drawings / two-tone diagrams. Combines Otsu bimodality of the
    /// 32-bin luma histogram, top-2-bin coverage, and low-entropy
    /// gate via a conservative `min` combinator. Drives webp
    /// `Preset::Drawing`, jxl modular path selection, png palette
    /// preference. Distinct from `ScreenContentLikelihood` (which is
    /// driven by palette and high-frequency energy).
    ///
    /// Behind the `composites` cargo feature: the combinator
    /// coefficients are calibration-driven and may drift in 0.1.x.
    #[cfg(feature = "composites")]
    LineArtScore = 45 : f32 => line_art_score,

    /// `f32`. Fraction `[0, 1]` of sampled pixels in the canonical
    /// chrominance-only skin-tone region. The chroma gates are
    /// **invariant to skin pigmentation** (Cb / Cr quantify hue, not
    /// brightness), so the same thresholds work across light, medium,
    /// and dark skin. Luma covers a wide range to span every tone:
    ///
    /// - `Y  ∈ [40,  240]` — spans deep shadow on dark skin to bright
    ///   highlight on light skin without rejecting either end
    /// - `Cb ∈ [77,  127]` — Chai & Ngan (1999) chrominance bound
    /// - `Cr ∈ [133, 173]` — Chai & Ngan (1999) chrominance bound
    ///
    /// Computed per-pixel in Tier 1 alongside the existing grayscale
    /// counter, reusing the BT.601 fixed-point YCbCr conversion. Zero
    /// added allocations; ~2 ns/pixel on a 7950X.
    ///
    /// **One-direction signal.** Non-zero fraction is strong evidence
    /// of a natural photograph (humans, animals, food). Zero fraction
    /// is **not** evidence against a photograph — landscapes,
    /// architecture, and macro shots without skin tones all score
    /// zero. Use as a positive-only confirmation, never as a negative
    /// classifier.
    ///
    /// **Why YCbCr instead of CIELAB.** CIELAB skin classifiers
    /// (Garcia & Tziritas 1999) outperform YCbCr by ~2 percentage
    /// points on standard skin-detection benchmarks but cost a
    /// non-linear sRGB → XYZ → LAB conversion per pixel. The Chai-Ngan
    /// YCbCr classifier is within ~5 % of LAB at zero extra
    /// arithmetic — already paid for by `chroma_complexity`,
    /// `cb_sharpness`, and the BT.601 luma the analyzer needs anyway.
    ///
    /// **Empirical ranges** (from a 219-image labeled corpus —
    /// `docs/calibration-corpus-2026-04-27.md`):
    ///
    /// - `photo_natural`:   p10 = 0.009, p50 = 0.130, p90 = 0.543
    /// - `photo_portrait`:  p10 = 0.047, p50 = 0.241, p90 = 0.541 — 94 % > 1 %
    /// - `photo_detailed`:  p10 = 0.021, p50 = 0.300, p90 = 0.470
    /// - `screen_document`: p10 = 0.000, p50 = 0.006, p90 = 0.077
    /// - `screen_ui`:       p10 = 0.000, p50 = 0.027, p90 = 0.127
    /// - `illustration`:    p10 = 0.001, p50 = 0.081, p90 = 0.323
    ///
    /// AUC = `0.799` for photo-vs-other classification — comparable
    /// to [`Self::NaturalLikelihood`] (0.814).
    ///
    /// **Operating threshold:** `skin_tone_fraction >= 0.05` gives
    /// `P = 0.89, R = 0.76, F1 = 0.82` for photo classification.
    /// Lower thresholds (`> 0`) maximize recall (`F1 = 0.882`).
    ///
    /// References: Chai & Ngan, "Face segmentation using skin-color
    /// map in videophone applications", IEEE TCSVT 1999;
    /// Vezhnevets et al., "A Survey on Pixel-Based Skin Color
    /// Detection Techniques", Graphicon 2003.
    #[cfg(feature = "experimental")]
    SkinToneFraction = 49 : f32 => skin_tone_fraction,

    /// `f32`. Standard deviation of luma gradient magnitudes across
    /// pixels that crossed the [`Self::EdgeDensity`] threshold
    /// (`|∇L|² > 400`, i.e. `|∇L| > 20`). Range `[0, ~150]` on the
    /// 0–255 luma scale.
    ///
    /// Tier 1 piggyback: the same SIMD edge sweep that produces
    /// `edge_density` accumulates `Σ g` and `Σ g²` over the threshold-
    /// crossing subset. Stddev is computed at row close from those
    /// running sums. Returns `0.0` if zero edges crossed (smooth
    /// image or below-threshold-only gradients).
    ///
    /// **Physical signal.** Natural photographs have edges anti-
    /// aliased by lens MTF + sensor pixel pitch — gradient magnitudes
    /// cluster tightly around the optical cutoff (typical stddev
    /// ~`8–18` on 0–255 luma). Digital artwork has either no edges
    /// (smooth gradients), single-pixel edges (line art — already
    /// caught by [`Self::LineArtScore`]), or **bimodal** gradients
    /// from variable-pressure brushwork or stylization (typical
    /// stddev `> 25`). JPEG-roundtripped artwork's blocking artifacts
    /// also widen this stddev relative to a pristine PNG photograph.
    ///
    /// **Empirical ranges** (from a 219-image labeled corpus —
    /// `docs/calibration-corpus-2026-04-27.md`):
    ///
    /// - `photo_natural`:   p10 = 15.9, p50 = 24.2, p90 = 31.8
    /// - `photo_portrait`:  p10 = 15.3, p50 = 20.7, p90 = 27.0
    /// - `photo_detailed`:  p10 = 18.2, p50 = 23.1, p90 = 32.0
    /// - `illustration`:    p10 = 12.9, p50 = 20.9, p90 = 26.7
    /// - `screen_document`: p10 = 42.0, p50 = 55.3, p90 = 57.5
    /// - `screen_ui`:       p10 = 31.6, p50 = 42.1, p90 = 54.4
    ///
    /// AUC = `0.843` for screen-vs-photo classification (high values →
    /// screen content). The strongest single screen-content signal
    /// after [`Self::PatchFraction`].
    ///
    /// **Operating thresholds:**
    /// - `edge_slope_stdev > 35` ⇒ very likely screen / chart / UI
    /// - `15 ≤ edge_slope_stdev ≤ 32` ⇒ photographic-edge distribution
    /// - `< 15` with low [`Self::EdgeDensity`] ⇒ smooth content
    ///   (illustrations or low-detail photos overlap here, ~13–27)
    ///
    /// Tracks the issue #123 proposal `EdgeSlopeStdev`.
    #[cfg(feature = "experimental")]
    EdgeSlopeStdev = 50 : f32 => edge_slope_stdev,
}

/// A scalar feature value — discriminated by the value type, not by
/// the feature it came from. Most callers want [`Self::to_f32`] (the
/// lossless coercion that maps `Bool(false) → 0.0`, `Bool(true) →
/// 1.0`, and `U32(n) → n as f32`).
///
/// `#[non_exhaustive]` so future structured feature types (small
/// histograms, vectors) can land via additional variants without a
/// major bump. Today the variants cover every scalar field on the
/// legacy `AnalyzerOutput`.
#[non_exhaustive]
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum FeatureValue {
    F32(f32),
    U32(u32),
    Bool(bool),
}

impl FeatureValue {
    /// Type-checked accessor: returns `Some(x)` only if the value is
    /// `F32(_)`. Use when the caller knows the underlying type.
    #[inline]
    pub const fn as_f32(self) -> Option<f32> {
        match self {
            Self::F32(x) => Some(x),
            _ => None,
        }
    }

    /// Type-checked accessor for `U32`.
    #[inline]
    pub const fn as_u32(self) -> Option<u32> {
        match self {
            Self::U32(x) => Some(x),
            _ => None,
        }
    }

    /// Type-checked accessor for `Bool`.
    #[inline]
    pub const fn as_bool(self) -> Option<bool> {
        match self {
            Self::Bool(x) => Some(x),
            _ => None,
        }
    }

    /// Lossless coercion to `f32`. `Bool(false) → 0.0`, `Bool(true) →
    /// 1.0`, `U32(n) → n as f32` (lossless for `n ≤ 2²⁴`, which
    /// covers every current and foreseeable feature). Convenience for
    /// ML pipelines / threshold comparisons that don't care about the
    /// underlying type.
    #[inline]
    pub fn to_f32(self) -> f32 {
        match self {
            Self::F32(x) => x,
            Self::U32(x) => x as f32,
            Self::Bool(false) => 0.0,
            Self::Bool(true) => 1.0,
        }
    }
}

impl From<f32> for FeatureValue {
    fn from(x: f32) -> Self {
        Self::F32(x)
    }
}
impl From<u32> for FeatureValue {
    fn from(x: u32) -> Self {
        Self::U32(x)
    }
}
impl From<bool> for FeatureValue {
    fn from(x: bool) -> Self {
        Self::Bool(x)
    }
}

/// Opaque set of [`AnalysisFeature`]s, supporting `const fn` set math.
///
/// Backed by a 256-bit (4 × `u64`) presence bitmap indexed by the
/// feature's `u16` discriminant. The size is an internal detail —
/// future versions may grow it transparently. Public callers only see
/// the set ops and [`Self::contains`].
///
/// Deliberately no `all()` constructor: callers must enumerate the
/// features they actually need. "Compute everything" is rarely the
/// right choice and disables the runtime dispatch optimisation that
/// skips entire passes when none of their outputs were requested.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct FeatureSet {
    bits: [u64; 4],
}

impl FeatureSet {
    /// Empty set.
    pub const fn new() -> Self {
        Self { bits: [0; 4] }
    }

    /// Singleton set containing exactly one feature.
    pub const fn just(f: AnalysisFeature) -> Self {
        Self::new().with(f)
    }

    /// Return a copy with `f` inserted. `const fn` so it composes in
    /// `const`-context preset definitions.
    pub const fn with(mut self, f: AnalysisFeature) -> Self {
        let id = f as u16 as usize;
        self.bits[id >> 6] |= 1u64 << (id & 63);
        self
    }

    /// Return a copy with `f` removed.
    pub const fn without(mut self, f: AnalysisFeature) -> Self {
        let id = f as u16 as usize;
        self.bits[id >> 6] &= !(1u64 << (id & 63));
        self
    }

    /// Set union (`A ∪ B`).
    pub const fn union(mut self, other: Self) -> Self {
        let mut i = 0;
        while i < 4 {
            self.bits[i] |= other.bits[i];
            i += 1;
        }
        self
    }

    /// Set intersection (`A ∩ B`).
    pub const fn intersect(mut self, other: Self) -> Self {
        let mut i = 0;
        while i < 4 {
            self.bits[i] &= other.bits[i];
            i += 1;
        }
        self
    }

    /// Set difference (`A − B`).
    pub const fn difference(mut self, other: Self) -> Self {
        let mut i = 0;
        while i < 4 {
            self.bits[i] &= !other.bits[i];
            i += 1;
        }
        self
    }

    /// `true` iff `self` and `other` share any feature.
    pub const fn intersects(self, other: Self) -> bool {
        let mut i = 0;
        while i < 4 {
            if self.bits[i] & other.bits[i] != 0 {
                return true;
            }
            i += 1;
        }
        false
    }

    /// `true` iff `self` contains every feature in `other`.
    pub const fn contains_all(self, other: Self) -> bool {
        let mut i = 0;
        while i < 4 {
            if self.bits[i] & other.bits[i] != other.bits[i] {
                return false;
            }
            i += 1;
        }
        true
    }

    /// `true` iff the set contains `f`.
    pub const fn contains(self, f: AnalysisFeature) -> bool {
        let id = f as u16 as usize;
        (self.bits[id >> 6] >> (id & 63)) & 1 != 0
    }

    /// `true` iff the set has no features.
    pub const fn is_empty(self) -> bool {
        let mut i = 0;
        while i < 4 {
            if self.bits[i] != 0 {
                return false;
            }
            i += 1;
        }
        true
    }

    /// Number of features in the set.
    pub const fn len(self) -> u32 {
        let mut total = 0;
        let mut i = 0;
        while i < 4 {
            total += self.bits[i].count_ones();
            i += 1;
        }
        total
    }

    /// Iterate the contained features in ascending [`AnalysisFeature::id`]
    /// order.
    ///
    /// Skips ids that don't correspond to a variant in this build —
    /// retired discriminants are never re-mapped, and cfg-disabled
    /// experimental variants legitimately return `None` from
    /// [`AnalysisFeature::from_u16`]. This means the iterator length
    /// is **at most** [`Self::len`]; on a cross-feature build it
    /// equals it. Sidecars / Python fitters / harness code can use
    /// this to walk every supported feature without hand-listing the
    /// enum.
    pub fn iter(self) -> FeatureSetIter {
        FeatureSetIter {
            bits: self.bits,
            next_id: 0,
        }
    }
    // `SUPPORTED` is generated by the [`features_table!`] invocation
    // above so it stays in lockstep with the enum.
}

/// Iterator over the [`AnalysisFeature`]s in a [`FeatureSet`], in
/// ascending [`AnalysisFeature::id`] order. Built by
/// [`FeatureSet::iter`].
pub struct FeatureSetIter {
    bits: [u64; 4],
    next_id: u16,
}

impl Iterator for FeatureSetIter {
    type Item = AnalysisFeature;
    fn next(&mut self) -> Option<Self::Item> {
        // Walk bits in id-ascending order, skipping ids the bitset
        // doesn't have set, AND ids whose variants aren't in this
        // build (cfg-gated experimentals on default features).
        loop {
            let id = self.next_id as usize;
            if id >= 256 {
                return None;
            }
            let bit = (self.bits[id >> 6] >> (id & 63)) & 1;
            self.next_id += 1;
            if bit == 1
                && let Some(f) = AnalysisFeature::from_u16(id as u16)
            {
                return Some(f);
            }
        }
    }
}

impl IntoIterator for FeatureSet {
    type Item = AnalysisFeature;
    type IntoIter = FeatureSetIter;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl Default for FeatureSet {
    fn default() -> Self {
        Self::new()
    }
}

// --- Internal: tier-membership constants used by the runtime const-bool
// dispatch in `analyze_with`. Not public — codecs name their features
// individually, not via tier bundles, so the tier split stays an
// implementation detail and can be refactored without breaking callers.

/// Palette features whose computation requires an **exact** distinct-
/// colour count. Asking for any of these forces the full-scan path
/// (every pixel walked). Const-block style so individual `.with()`
/// calls can be cfg-gated when their variant is experimental.
pub(crate) const PALETTE_FULL_FEATURES: FeatureSet = {
    let mut s = FeatureSet::new();
    s = s.with(AnalysisFeature::DistinctColorBins);
    #[cfg(feature = "experimental")]
    {
        s = s.with(AnalysisFeature::PaletteDensity);
        // GrayscaleScore is computed on the same full-scan walk that
        // builds the palette histogram. It needs 100 % coverage —
        // stripe-sampling at ~5 % budget would let a single colour
        // pixel slip past the gate ~95 % of the time and produce a
        // false-positive grayscale classification.
        s = s.with(AnalysisFeature::GrayscaleScore);
    }
    s
};

/// Palette features that only need the running count, with an early-
/// exit at 256. On photographic content these typically resolve in
/// ~10 rows; on indexed content they walk to the end. Both members
/// are experimental in 0.1.0; this set is empty when the feature is
/// off, which makes the quick-path dispatch never fire.
#[allow(unused_mut, unused_assignments)]
pub(crate) const PALETTE_QUICK_FEATURES: FeatureSet = {
    let mut s = FeatureSet::new();
    #[cfg(feature = "experimental")]
    {
        s = s.with(AnalysisFeature::IndexedPaletteWidth);
        s = s.with(AnalysisFeature::PaletteFitsIn256);
    }
    s
};

/// Union: any palette feature triggers the palette pass. The quick
/// path runs when the request is exclusively in `PALETTE_QUICK_FEATURES`;
/// any overlap with `PALETTE_FULL_FEATURES` forces the full path
/// (which produces both signal classes).
pub(crate) const PALETTE_FEATURES: FeatureSet = PALETTE_FULL_FEATURES.union(PALETTE_QUICK_FEATURES);

/// Tier 1 "extras" — the optional accumulators that elevate the
/// SIMD kernel from `Minimal` to `Full`. When the requested
/// `FeatureSet` doesn't intersect this set, `accumulate_row_simd`
/// is dispatched as `<FULL = false>` and skips the per-chunk
/// luma_sum / Hasler-Süsstrunk M3 (rg/yb) / skin-tone / edge-slope
/// accumulators — and `extract_tier1_into` skips the separate
/// Laplacian SIMD row pass entirely. Drops ~10 lane-wise f32x8
/// accumulators on AVX2, freeing register pressure on the Tier 1
/// hot path.
///
/// Driven by zenjpeg's actual `ADAPTIVE_FEATURES` query — neither
/// `Variance`, `Colourfulness`, `LaplacianVariance`,
/// `SkinToneFraction`, nor `EdgeSlopeStdev` is in that set, so
/// every zenjpeg analyze call lands in the `Minimal` bucket.
#[allow(unused_mut, unused_assignments)]
pub(crate) const TIER1_EXTRAS_FEATURES: FeatureSet = {
    let mut s = FeatureSet::new();
    s = s.with(AnalysisFeature::Variance);
    #[cfg(feature = "experimental")]
    {
        s = s.with(AnalysisFeature::Colourfulness);
        s = s.with(AnalysisFeature::LaplacianVariance);
        s = s.with(AnalysisFeature::SkinToneFraction);
        s = s.with(AnalysisFeature::EdgeSlopeStdev);
    }
    s
};

/// Subset of [`TIER1_EXTRAS_FEATURES`] gated by the
/// `accumulate_row_simd` `FULL` const-bool: luma stats (Variance) +
/// Hasler M3 (Colourfulness) + edge-slope batching
/// (EdgeSlopeStdev) + the separate Laplacian SIMD pass
/// (LaplacianVariance). `SkinToneFraction` is peeled off into
/// [`TIER1_SKIN_FEATURES`] so the two halves share register
/// pressure on AVX2 only when both are requested.
#[allow(unused_mut, unused_assignments)]
pub(crate) const TIER1_FULL_FEATURES: FeatureSet = {
    let mut s = FeatureSet::new();
    s = s.with(AnalysisFeature::Variance);
    #[cfg(feature = "experimental")]
    {
        s = s.with(AnalysisFeature::Colourfulness);
        s = s.with(AnalysisFeature::LaplacianVariance);
        s = s.with(AnalysisFeature::EdgeSlopeStdev);
    }
    s
};

/// Subset of [`TIER1_EXTRAS_FEATURES`] gated by the
/// `accumulate_row_simd` `SKIN` const-bool: BT.601 chroma matrix
/// (2 fma chains) + 6 Chai-Ngan threshold compares + 5 mask
/// AND-chain + masked counter. Independent of `FULL` — a caller
/// that only wants `SkinToneFraction` dispatches with
/// `<*, false, true>` and skips luma stats / Hasler M3 entirely.
#[allow(unused_mut, unused_assignments)]
pub(crate) const TIER1_SKIN_FEATURES: FeatureSet = {
    let mut s = FeatureSet::new();
    #[cfg(feature = "experimental")]
    {
        s = s.with(AnalysisFeature::SkinToneFraction);
    }
    s
};

pub(crate) const TIER2_FEATURES: FeatureSet = FeatureSet::new()
    .with(AnalysisFeature::CbHorizSharpness)
    .with(AnalysisFeature::CbVertSharpness)
    .with(AnalysisFeature::CbPeakSharpness)
    .with(AnalysisFeature::CrHorizSharpness)
    .with(AnalysisFeature::CrVertSharpness)
    .with(AnalysisFeature::CrPeakSharpness);

#[allow(unused_mut, unused_assignments)]
pub(crate) const TIER3_FEATURES: FeatureSet = {
    let mut s = FeatureSet::new();
    s = s.with(AnalysisFeature::HighFreqEnergyRatio);
    s = s.with(AnalysisFeature::LumaHistogramEntropy);
    #[cfg(feature = "experimental")]
    {
        s = s.with(AnalysisFeature::DctCompressibilityY);
        s = s.with(AnalysisFeature::DctCompressibilityUV);
        s = s.with(AnalysisFeature::PatchFraction);
        s = s.with(AnalysisFeature::AqMapMean);
        s = s.with(AnalysisFeature::AqMapStd);
        s = s.with(AnalysisFeature::NoiseFloorY);
        s = s.with(AnalysisFeature::NoiseFloorUV);
        s = s.with(AnalysisFeature::GradientFraction);
    }
    #[cfg(feature = "composites")]
    {
        s = s.with(AnalysisFeature::LineArtScore);
    }
    s
};

pub(crate) const ALPHA_FEATURES: FeatureSet = FeatureSet::new()
    .with(AnalysisFeature::AlphaPresent)
    .with(AnalysisFeature::AlphaUsedFraction)
    .with(AnalysisFeature::AlphaBimodalScore);

/// Source-direct depth tier. Reads source samples without going
/// through `RowConverter` so HDR / wide-gamut / high-bit-depth signals
/// survive the analysis. All members are experimental in 0.1.0; the
/// set is empty when the cargo feature is off, which makes the depth-
/// tier dispatch never fire in default builds.
#[allow(unused_mut, unused_assignments)]
pub(crate) const DEPTH_FEATURES: FeatureSet = {
    let mut s = FeatureSet::new();
    #[cfg(feature = "experimental")]
    {
        s = s.with(AnalysisFeature::PeakLuminanceNits);
        s = s.with(AnalysisFeature::P99LuminanceNits);
        s = s.with(AnalysisFeature::HdrHeadroomStops);
        s = s.with(AnalysisFeature::HdrPixelFraction);
        s = s.with(AnalysisFeature::WideGamutPeak);
        s = s.with(AnalysisFeature::WideGamutFraction);
        s = s.with(AnalysisFeature::EffectiveBitDepth);
        s = s.with(AnalysisFeature::HdrPresent);
        s = s.with(AnalysisFeature::GamutCoverageSrgb);
        s = s.with(AnalysisFeature::GamutCoverageP3);
    }
    s
};

#[allow(unused_mut, unused_assignments)]
pub(crate) const DERIVED_FEATURES: FeatureSet = {
    let mut s = FeatureSet::new();
    #[cfg(feature = "composites")]
    {
        s = s.with(AnalysisFeature::TextLikelihood);
        s = s.with(AnalysisFeature::ScreenContentLikelihood);
        s = s.with(AnalysisFeature::NaturalLikelihood);
    }
    s
};

// --- Derived-feature dependency closures ----------------------------
//
// Derived likelihoods are computed from leaf features in other tiers.
// The dispatch axes use these supersets to ensure the *correct*
// dependencies run, not the over-approximation "any DERIVED triggers
// every tier".
//
// `compute_derived_likelihoods` is independently gated by const bools
// so an axis-table mistake here can't escape — a likelihood whose deps
// weren't computed is left at the field's default and dropped by
// `into_results`. Layered defense: caller sees `None`, never garbage.

/// Features whose computation reads from Tier 3 outputs
/// (`luma_histogram_entropy`). Includes:
/// - All [`TIER3_FEATURES`] proper.
/// - [`AnalysisFeature::TextLikelihood`] (uses `luma_histogram_entropy`).
/// - [`AnalysisFeature::NaturalLikelihood`] (uses `luma_histogram_entropy`).
///
/// `ScreenContentLikelihood` is **not** here — it's palette + T1 only.
#[allow(unused_mut, unused_assignments)]
pub(crate) const T3_NEEDED_BY: FeatureSet = {
    let mut s = TIER3_FEATURES;
    #[cfg(feature = "composites")]
    {
        s = s.with(AnalysisFeature::TextLikelihood);
        s = s.with(AnalysisFeature::NaturalLikelihood);
    }
    s
};

/// Features whose computation reads from palette outputs
/// (`distinct_color_bins`). Includes:
/// - All [`PALETTE_FEATURES`] proper.
/// - [`AnalysisFeature::ScreenContentLikelihood`] (uses `distinct_color_bins`).
/// - [`AnalysisFeature::NaturalLikelihood`] (uses `distinct_color_bins`).
///
/// `TextLikelihood` is **not** here — it's T3-entropy + T1 only.
#[allow(unused_mut, unused_assignments)]
pub(crate) const PAL_NEEDED_BY: FeatureSet = {
    let mut s = PALETTE_FEATURES;
    #[cfg(feature = "composites")]
    {
        s = s.with(AnalysisFeature::ScreenContentLikelihood);
        s = s.with(AnalysisFeature::NaturalLikelihood);
    }
    s
};

// `RawAnalysis` and `into_results` are generated by the
// `features_table!` invocation at the top of this file.

// ---------------------- AnalysisQuery ----------------------------

/// A request to the analyzer: what features to compute.
///
/// Opaque — the only public knob is the [`FeatureSet`]. Tier sampling
/// budgets (`pixel_budget`, `hf_max_blocks`) are **invariants of the
/// crate**, not per-call knobs: the convergence-trained defaults are
/// the only values that ship, and shipped oracle / threshold tables
/// are calibrated against them. Refining them is a release-level
/// decision, not a caller decision.
///
/// Use [`Self::new`] to build. There is intentionally no `Default`,
/// no fluent budget setters, and no `full()` shortcut — callers must
/// enumerate the features they need, and "compute everything" must be
/// expressed by enumerating every feature explicitly. The runtime
/// dispatcher uses the requested set to skip entire passes whose
/// outputs aren't needed.
#[derive(Clone, Debug)]
pub struct AnalysisQuery {
    features: FeatureSet,
}

impl AnalysisQuery {
    /// Build a query for `features`.
    ///
    /// `features` typically comes from `const`-context union of each
    /// codec's preset, e.g. `JPEG_FEATURES.union(WEBP_FEATURES)`.
    pub const fn new(features: FeatureSet) -> Self {
        Self { features }
    }

    /// The feature set this query asks for.
    pub const fn features(&self) -> FeatureSet {
        self.features
    }
}

/// Crate-internal: the canonical sampling budgets.
///
/// These are **not** exposed on [`AnalysisQuery`]. They're invariants
/// the crate maintains so shipped oracle / threshold tables stay
/// calibrated. Updating them is a release-level decision (with a
/// retrain or recalibration of every downstream consumer); callers
/// don't get to override per-call.
///
/// Tests / oracle re-extraction that genuinely need different
/// sampling go through the `__internal_with_overrides` ctor —
/// double-underscored, `#[doc(hidden)]`, **not** a stable API.
pub(crate) const DEFAULT_PIXEL_BUDGET: usize = 500_000;
pub(crate) const DEFAULT_HF_MAX_BLOCKS: usize = 1024;

#[doc(hidden)]
impl AnalysisQuery {
    /// **Unstable. Tests / oracle re-extraction only.** Lets the
    /// caller override the otherwise-invariant sampling budgets. No
    /// stability guarantee — may be removed or renamed at any time.
    /// Production code that calls this is wrong; if you think you
    /// need it, file an issue describing the use case.
    pub fn __internal_with_overrides(
        features: FeatureSet,
        pixel_budget: usize,
        hf_max_blocks: usize,
    ) -> InternalQuery {
        InternalQuery {
            features,
            pixel_budget,
            hf_max_blocks,
        }
    }
}

/// **Unstable.** Backdoor for oracle / convergence-study work that
/// needs non-default sampling budgets. Constructed only via
/// [`AnalysisQuery::__internal_with_overrides`]; consumed by an
/// internal entry point that public callers don't see.
#[doc(hidden)]
pub struct InternalQuery {
    pub(crate) features: FeatureSet,
    pub(crate) pixel_budget: usize,
    pub(crate) hf_max_blocks: usize,
}

// ---------------------- AnalysisResults ---------------------------

/// Image geometry — width / height / megapixels / aspect ratio.
/// Returned alongside [`AnalysisResults`] regardless of which features
/// were requested (it's a property of the input, not the analysis).
#[derive(Copy, Clone, Debug)]
pub struct ImageGeometry {
    width: u32,
    height: u32,
}

impl ImageGeometry {
    /// Construct from raw width/height. Crate-internal entry; public
    /// callers receive `ImageGeometry` from [`AnalysisResults::geometry`]
    /// rather than building it themselves.
    pub(crate) const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    /// Image width in pixels.
    pub const fn width(self) -> u32 {
        self.width
    }
    /// Image height in pixels.
    pub const fn height(self) -> u32 {
        self.height
    }
    /// `width × height` as `u64` to avoid overflow on giant images.
    pub const fn pixels(self) -> u64 {
        self.width as u64 * self.height as u64
    }
    /// `pixels / 1e6` as f32. Lossy for >2²⁴ MP (>16 trillion pixels).
    pub fn megapixels(self) -> f32 {
        self.pixels() as f32 / 1_000_000.0
    }
    /// `width / max(1, height)`. Returns 0 if height is 0.
    pub fn aspect_ratio(self) -> f32 {
        if self.height == 0 {
            0.0
        } else {
            self.width as f32 / self.height as f32
        }
    }
}

/// Opaque container for one analysis pass's outputs.
///
/// Query individual features with [`Self::get`], passing the
/// [`AnalysisFeature`] you want. Returns `None` if:
/// - The feature wasn't in the requested set, or
/// - The feature is retired ([`AnalysisFeature::is_active`] = false), or
/// - The pass that produces it failed (e.g. image too small).
///
/// Storage is opaque — internally a sparse `Vec<(AnalysisFeature,
/// FeatureValue)>` sized to the requested set, **no over-allocation**.
/// Future versions may switch the backing layout (packed parallel
/// arrays, sorted slice, etc.) without breaking the public API.
pub struct AnalysisResults {
    requested: FeatureSet,
    geometry: ImageGeometry,
    source_descriptor: zenpixels::PixelDescriptor,
    /// One entry per *populated* feature, sorted ascending by
    /// `AnalysisFeature::id`. Vec capacity is preallocated to
    /// `requested.len()` so the analyzer's `set` calls never realloc.
    /// Lookup by linear scan over up to ~30 entries (one cache line).
    values: Vec<(AnalysisFeature, FeatureValue)>,
}

impl AnalysisResults {
    /// Build an empty `AnalysisResults` for the given `requested` set,
    /// image geometry, and source descriptor. Crate-internal — public
    /// callers receive results from `analyze_features`, never
    /// construct them directly.
    pub(crate) fn new(
        requested: FeatureSet,
        geometry: ImageGeometry,
        source_descriptor: zenpixels::PixelDescriptor,
    ) -> Self {
        Self {
            requested,
            geometry,
            source_descriptor,
            values: Vec::with_capacity(requested.len() as usize),
        }
    }

    /// Crate-internal: write a value.
    ///
    /// `debug_assert!`s that `f` is in the requested set — analyzer
    /// code is wrong if it tries to store unrequested features (the
    /// dispatcher should have skipped that work entirely). In release
    /// the call silently no-ops to avoid storing data the caller
    /// didn't ask for.
    ///
    /// If `f` is already present (e.g. a tier wrote it twice), the
    /// later value wins; the entry stays sorted by id.
    pub(crate) fn set(&mut self, f: AnalysisFeature, v: impl Into<FeatureValue>) {
        debug_assert!(
            self.requested.contains(f),
            "analyzer wrote unrequested feature {:?} (id={}) — dispatcher gating is broken",
            f,
            f.id()
        );
        if !self.requested.contains(f) {
            return;
        }
        let v = v.into();
        // Insertion-sort by id. Up to ~30 entries; binary search +
        // memmove would be the same cost as a linear walk here.
        let mut i = 0;
        while i < self.values.len() {
            match self.values[i].0.id().cmp(&f.id()) {
                core::cmp::Ordering::Less => i += 1,
                core::cmp::Ordering::Equal => {
                    self.values[i].1 = v;
                    return;
                }
                core::cmp::Ordering::Greater => break,
            }
        }
        self.values.insert(i, (f, v));
    }

    /// The feature set the caller asked for. Useful for asserting
    /// "did the analyzer compute what I asked for" or for joining
    /// results from multiple analyses.
    pub const fn requested(&self) -> FeatureSet {
        self.requested
    }

    /// Image geometry for the analysed input.
    pub const fn geometry(&self) -> ImageGeometry {
        self.geometry
    }

    /// The source [`zenpixels::PixelDescriptor`] the analyzer ingested.
    ///
    /// Codecs (zenavif / zenjxl / zenwebp / zenjpeg) read this to
    /// drive bit-depth, primaries, transfer-function, color-model,
    /// alpha-mode, and signal-range encode decisions — the analyzer
    /// doesn't surface every descriptor field as a separate feature
    /// because the descriptor is already a small `Copy` value with a
    /// stable shape; callers can pull whatever they need with one
    /// accessor instead of querying ten boolean features.
    ///
    /// For "is the image actually using the gamut?" / "does the bit
    /// depth carry information?" decisions, pair this with the
    /// `tier_depth` features ([`AnalysisFeature::EffectiveBitDepth`],
    /// [`AnalysisFeature::HdrPresent`],
    /// [`AnalysisFeature::WideGamutFraction`], …) which read the
    /// pixel data, not just the metadata.
    #[inline]
    pub const fn source_descriptor(&self) -> zenpixels::PixelDescriptor {
        self.source_descriptor
    }

    /// Look up one feature's value. `None` if not requested, retired,
    /// or computation failed.
    #[inline]
    pub fn get(&self, f: AnalysisFeature) -> Option<FeatureValue> {
        // Linear scan — Vec is sorted by id but at ≤ 30 entries the
        // branch-predictor handles a linear walk faster than a
        // binary search.
        self.values.iter().find(|(k, _)| *k == f).map(|(_, v)| *v)
    }

    /// Convenience: get and coerce to `f32`. Returns `None` if the
    /// feature isn't present, or `Some(0.0)` for `Bool(false)`,
    /// `Some(1.0)` for `Bool(true)`, `Some(n as f32)` for `U32(n)`.
    /// Callers that want strict typed access should use
    /// [`Self::get`] + [`FeatureValue::as_f32`] (or `as_u32` /
    /// `as_bool`).
    #[inline]
    pub fn get_f32(&self, f: AnalysisFeature) -> Option<f32> {
        self.get(f).map(FeatureValue::to_f32)
    }
}

impl core::fmt::Debug for AnalysisResults {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut d = f.debug_struct("AnalysisResults");
        d.field("requested", &self.requested);
        d.field("geometry", &self.geometry);
        for (feature, v) in &self.values {
            d.field(feature.name(), v);
        }
        d.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_set_basic_ops() {
        let a = FeatureSet::just(AnalysisFeature::Variance);
        let b = FeatureSet::just(AnalysisFeature::EdgeDensity);
        let u = a.union(b);
        assert!(u.contains(AnalysisFeature::Variance));
        assert!(u.contains(AnalysisFeature::EdgeDensity));
        // Pick a third unrelated variant to spot-check non-membership.
        // ChromaComplexity is unflagged so the test compiles regardless
        // of the experimental feature gate.
        assert!(!u.contains(AnalysisFeature::ChromaComplexity));
        assert_eq!(u.len(), 2);
        assert!(a.intersects(u));
        assert!(u.contains_all(a));
        assert!(!a.contains_all(u));
    }

    #[test]
    fn feature_value_roundtrip() {
        let v: FeatureValue = 1.5f32.into();
        assert_eq!(v.as_f32(), Some(1.5));
        assert_eq!(v.as_u32(), None);
        assert_eq!(v.to_f32(), 1.5);

        let v: FeatureValue = 7u32.into();
        assert_eq!(v.as_u32(), Some(7));
        assert_eq!(v.to_f32(), 7.0);

        let v: FeatureValue = true.into();
        assert_eq!(v.as_bool(), Some(true));
        assert_eq!(v.to_f32(), 1.0);
    }

    /// Discriminant ids that have been retired and reserved — `from_u16`
    /// must return `None` for these so callers don't get a value at a
    /// slot whose meaning has changed.
    const RESERVED_RETIRED_IDS: &[u16] = &[
        // id 11 was `DistinctColorBinsChao1`, removed pre-0.1.0
        // because the full-pass scan made it permanently equal to
        // `DistinctColorBins`.
        11,
    ];

    #[test]
    fn discriminants_round_trip() {
        // Sequential 0..32 with retired-id and cfg-disabled holes.
        // Active variants round-trip through id() / from_u16; retired
        // and cfg-disabled ids return None. Retired ids must always
        // return None (asserted explicitly); cfg-disabled ones either
        // return Some (when the cargo feature is on) or None (when off)
        // — both legal.
        for id in 0..64u16 {
            if RESERVED_RETIRED_IDS.contains(&id) {
                assert!(
                    AnalysisFeature::from_u16(id).is_none(),
                    "id {id} is retired but from_u16 returned Some — \
                     don't recycle retired discriminants"
                );
                continue;
            }
            // cfg-disabled experimental ids legitimately return None.
            if let Some(f) = AnalysisFeature::from_u16(id) {
                assert_eq!(f.id(), id);
            }
        }
        assert!(AnalysisFeature::from_u16(64).is_none());
        assert!(AnalysisFeature::from_u16(255).is_none());
    }

    #[test]
    fn analysis_query_constructor_only() {
        // `AnalysisQuery` exposes only the features it was constructed
        // with; sampling budgets are crate invariants, not knobs.
        let q = AnalysisQuery::new(FeatureSet::just(AnalysisFeature::Variance));
        assert!(q.features().contains(AnalysisFeature::Variance));
        assert!(!q.features().contains(AnalysisFeature::EdgeDensity));
    }

    #[test]
    fn raw_analysis_round_trip() {
        // Sanity: filling RawAnalysis and translating drops the
        // unrequested fields and keeps the requested ones, with the
        // right typing per feature.
        let raw = RawAnalysis {
            variance: 12.5,
            distinct_color_bins: 4096,
            alpha_present: true,
            edge_density: 0.5, // not requested below
            ..Default::default()
        };

        let requested = FeatureSet::just(AnalysisFeature::Variance)
            .with(AnalysisFeature::DistinctColorBins)
            .with(AnalysisFeature::AlphaPresent);
        let r = raw.into_results(
            requested,
            ImageGeometry::new(64, 64),
            zenpixels::PixelDescriptor::RGB8_SRGB,
        );

        assert_eq!(r.get_f32(AnalysisFeature::Variance), Some(12.5));
        assert_eq!(
            r.get(AnalysisFeature::DistinctColorBins),
            Some(FeatureValue::U32(4096))
        );
        assert_eq!(
            r.get(AnalysisFeature::AlphaPresent),
            Some(FeatureValue::Bool(true))
        );
        // Was zeroed by Default and never requested → None.
        assert_eq!(r.get(AnalysisFeature::EdgeDensity), None);
    }

    #[test]
    fn supported_set_covers_all_active_variants() {
        // Every active variant must round-trip through SUPPORTED.
        // Skip both retired ids (RESERVED_RETIRED_IDS) and ids whose
        // variants are cfg-gated out of this build — `from_u16` already
        // returns `None` for both kinds of holes, so a single
        // `if let Some(f) = …` walk handles them uniformly.
        let mut active = 0u32;
        for id in 0..64u16 {
            if RESERVED_RETIRED_IDS.contains(&id) {
                continue;
            }
            let Some(f) = AnalysisFeature::from_u16(id) else {
                // cfg-disabled experimental variant — legitimately
                // absent in this build.
                continue;
            };
            assert!(
                FeatureSet::SUPPORTED.contains(f),
                "{:?} (id={}) is missing from FeatureSet::SUPPORTED",
                f,
                id
            );
            active += 1;
        }
        assert_eq!(FeatureSet::SUPPORTED.len(), active);
    }

    #[test]
    fn analysis_results_get_and_set() {
        let requested = FeatureSet::just(AnalysisFeature::Variance)
            .with(AnalysisFeature::DistinctColorBins)
            .with(AnalysisFeature::AlphaPresent);
        let mut r = AnalysisResults::new(
            requested,
            ImageGeometry::new(1920, 1080),
            zenpixels::PixelDescriptor::RGB8_SRGB,
        );
        // Set values for everything in the requested set, deliberately
        // out of id-order to exercise the insertion-sort.
        r.set(AnalysisFeature::AlphaPresent, true);
        r.set(AnalysisFeature::Variance, 42.0_f32);
        r.set(AnalysisFeature::DistinctColorBins, 1234_u32);
        // Overwrite (later wins)
        r.set(AnalysisFeature::Variance, 43.0_f32);

        assert_eq!(
            r.get(AnalysisFeature::Variance),
            Some(FeatureValue::F32(43.0))
        );
        assert_eq!(
            r.get(AnalysisFeature::DistinctColorBins),
            Some(FeatureValue::U32(1234))
        );
        assert_eq!(
            r.get(AnalysisFeature::AlphaPresent),
            Some(FeatureValue::Bool(true))
        );
        // Requested but never set (no analysis ran) → None.
        // ChromaComplexity is unflagged so the assertion compiles
        // regardless of the experimental cargo feature.
        assert_eq!(r.get(AnalysisFeature::ChromaComplexity), None);
        // Not requested at all → None
        assert_eq!(r.get(AnalysisFeature::EdgeDensity), None);

        // get_f32 coercion
        assert_eq!(r.get_f32(AnalysisFeature::Variance), Some(43.0));
        assert_eq!(r.get_f32(AnalysisFeature::DistinctColorBins), Some(1234.0));
        assert_eq!(r.get_f32(AnalysisFeature::AlphaPresent), Some(1.0));

        // Geometry round-trips
        assert_eq!(r.geometry().width(), 1920);
        assert_eq!(r.geometry().pixels(), 1920 * 1080);
        assert!((r.geometry().aspect_ratio() - 1920.0 / 1080.0).abs() < 1e-6);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "analyzer wrote unrequested feature")]
    fn set_unrequested_feature_panics_in_debug() {
        let mut r = AnalysisResults::new(
            FeatureSet::just(AnalysisFeature::Variance),
            ImageGeometry::new(64, 64),
            zenpixels::PixelDescriptor::RGB8_SRGB,
        );
        // EdgeDensity wasn't requested — debug_assert! must fire.
        r.set(AnalysisFeature::EdgeDensity, 0.0_f32);
    }

    #[test]
    fn feature_set_iter_visits_every_set_member_in_id_order() {
        let s = FeatureSet::just(AnalysisFeature::DistinctColorBins)
            .with(AnalysisFeature::Variance)
            .with(AnalysisFeature::EdgeDensity);
        let collected: Vec<_> = s.iter().collect();
        // Iter is ascending by id; Variance=0, EdgeDensity=1, DistinctColorBins=10.
        assert_eq!(
            collected,
            vec![
                AnalysisFeature::Variance,
                AnalysisFeature::EdgeDensity,
                AnalysisFeature::DistinctColorBins,
            ]
        );

        // SUPPORTED iterates with len == count.
        let n = FeatureSet::SUPPORTED.iter().count();
        assert_eq!(n as u32, FeatureSet::SUPPORTED.len());

        // Empty set yields nothing.
        assert_eq!(FeatureSet::new().iter().count(), 0);

        // IntoIterator trait impl mirrors iter().
        let s2 = FeatureSet::just(AnalysisFeature::Variance);
        let v: Vec<_> = s2.into_iter().collect();
        assert_eq!(v, vec![AnalysisFeature::Variance]);
    }

    #[test]
    fn analysis_feature_id_is_public_and_stable() {
        // Wire-format guarantee: ids are sequential u16, immutable
        // once shipped. Public callers (Python fitters, sidecars)
        // must be able to read this.
        assert_eq!(AnalysisFeature::Variance.id(), 0);
        assert_eq!(AnalysisFeature::EdgeDensity.id(), 1);
        assert_eq!(AnalysisFeature::DistinctColorBins.id(), 10);
        assert_eq!(AnalysisFeature::AlphaPresent.id(), 24);

        // from_u16 is the public inverse.
        assert_eq!(
            AnalysisFeature::from_u16(0),
            Some(AnalysisFeature::Variance)
        );
        assert_eq!(AnalysisFeature::from_u16(11), None); // retired
        assert_eq!(AnalysisFeature::from_u16(9999), None);
    }

    #[test]
    fn tier_bundles_are_disjoint() {
        // Each feature should belong to at most one tier bundle.
        let bundles = [
            PALETTE_FEATURES,
            TIER2_FEATURES,
            TIER3_FEATURES,
            ALPHA_FEATURES,
            DERIVED_FEATURES,
        ];
        for (i, a) in bundles.iter().enumerate() {
            for b in bundles.iter().skip(i + 1) {
                assert!(!a.intersects(*b), "tier bundles overlap (this is a bug)");
            }
        }
    }
}
