//! Image content analyzers for adaptive codec decisions.
//!
//! Computes the numeric features used by oracle-trained decision trees
//! (originally `coefficient::scripts/fit_oracle_tree.py`) to drive
//! per-image encoder configuration. Every shipped feature is wired
//! into at least one fitted tree's splits per the 2026-04-25 audit.
//!
//! # Public API
//!
//! All public types are opaque. Construction goes through the
//! [`feature`] module:
//!
//! - [`feature::AnalysisFeature`]: stable identifier per feature
//!   (`#[non_exhaustive]`, `#[repr(u16)]`, sequential discriminants —
//!   ids are immutable once shipped).
//! - [`feature::FeatureValue`]: `F32` / `U32` / `Bool` tagged value
//!   with `to_f32` lossless coercion.
//! - [`feature::FeatureSet`]: opaque bitset with full `const fn` set
//!   math. The only "all features" entry is
//!   [`feature::FeatureSet::SUPPORTED`] — there is intentionally no
//!   `all()`. Production callers enumerate what they need.
//! - [`feature::AnalysisQuery`]: opaque request handle. Sampling
//!   budgets are crate invariants, not per-call knobs.
//! - [`feature::ImageGeometry`]: opaque `width`/`height`/
//!   `pixels`/`megapixels`/`aspect_ratio` accessors.
//! - [`feature::AnalysisResults`]: opaque queryable container.
//!
//! # Entry points
//!
//! - [`analyze_features`] — preferred. Takes any [`PixelSlice`] and
//!   an [`feature::AnalysisQuery`]. Native zero-copy on RGB8/RGB8_SRGB
//!   inputs; one-row scratch + `RowConverter` on anything else (every
//!   `zenpixels` descriptor: RGBA/BGRA u8/u16, GRAY u8/u16, RGB/RGBA
//!   f32 linear, PQ/HLG/Bt709, BT.709/DisplayP3/BT.2020/AdobeRGB).
//! - [`analyze_features_rgb8`] — convenience for packed RGB8 buffers.
//!
//! # Composition pattern
//!
//! ```ignore
//! use zenanalyze::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};
//!
//! const JPEG_FEATURES: FeatureSet = FeatureSet::new()
//!     .with(AnalysisFeature::Variance)
//!     .with(AnalysisFeature::EdgeDensity)
//!     .with(AnalysisFeature::DctCompressibilityY);
//! const WEBP_FEATURES: FeatureSet = FeatureSet::new()
//!     .with(AnalysisFeature::Variance)
//!     .with(AnalysisFeature::AlphaPresent);
//!
//! // Orchestrator unions, runs once, hands results down.
//! let needed = JPEG_FEATURES.union(WEBP_FEATURES);
//! let r = zenanalyze::analyze_features(slice, &AnalysisQuery::new(needed))?;
//! let var = r.get_f32(AnalysisFeature::Variance);
//! ```
//!
//! # Threshold contract — iterating during 0.1.x
//!
//! Numeric thresholds and normalization scales in this crate are
//! converging during the 0.1.x line. Downstream consumers that
//! compile-in fitted models (oracle decision trees, content
//! selectors) must pin to a specific zenanalyze patch version and
//! re-validate / retrain whenever they bump it — feature outputs
//! change between patches whenever a real bug is fixed.
//!
//! Every breaking numeric change ships with a CHANGELOG entry under
//! the version it landed in. When the algorithm stabilizes (1.0),
//! the contract will freeze. Sampling budgets (Tier 1/2 stripe step,
//! Tier 3 block cap) are part of that contract — not exposed as
//! per-call knobs.
//!
//! # Tier architecture
//!
//! Five passes, each gated by what the requested [`feature::FeatureSet`]
//! actually needs. None of them materialize a full RGB8 buffer:
//!
//! | Pass        | Iterates over            | Reads               | Cost (4 MP) | Drives                                    |
//! |-------------|--------------------------|---------------------|-------------|-------------------------------------------|
//! | Tier 1      | Stripe-sampled rows      | RGB8 (via RowStream) | ~1 ms      | luma stats, edges, chroma, uniformity, grayscale_score |
//! | Tier 2      | 3-row sliding window     | RGB8                 | ~2 ms      | per-axis Cb/Cr sharpness                  |
//! | Tier 3      | Sampled 8×8 DCT blocks   | RGB8                 | ~3 ms      | DCT energy, entropy, AQ map, noise floor, line-art, patch fraction |
//! | Palette     | Full-image (every pixel) | RGB8                 | ~1 ms      | distinct_color_bins                       |
//! | Alpha       | Stride-sampled rows      | **Source bytes**     | ~0.3 ms    | alpha presence / used / bimodal           |
//! | tier_depth  | Stride-sampled rows      | **Source bytes**     | ~0.5 ms HDR / ~0 ms SDR | peak nits, headroom, bit depth, HDR-present |
//!
//! Tier 1/2/3 + Palette read RGB8, going through `RowStream`'s `Native`
//! zero-copy path on RGB8 layouts and `RowConverter` row-by-row on
//! everything else. Alpha and `tier_depth` read source samples
//! directly — that's load-bearing for HDR (RowConverter doesn't
//! tonemap PQ/HLG; its narrowing clips to [0, 1] sRGB-display) and
//! for the alpha pass (avoids a precision-losing pre-multiply round-
//! trip on u16/f32 alpha).
//!
//! ## Why a separate `tier_depth`
//!
//! The standard tiers' threshold contract is calibrated on display-
//! space RGB8 bytes. HDR content's source samples encode dynamic
//! range that the RGB8 narrowing destroys; a 4000-nit PQ source and
//! a 100-nit-clipped SDR source produce byte-identical RGB8 streams.
//! Routing the depth tier through RowStream would have made it
//! literally impossible to surface HDR-aware features.
//!
//! Adding `tier_depth` as a fifth `const bool` axis to the dispatch
//! table would have doubled it from 16 to 32 monomorphizations for
//! near-zero benefit — the depth tier reads source bytes, not RGB8
//! rows, and shares no inner loop with T1/T2/T3 to specialize. So
//! it's wired as a runtime branch outside the const-bool dispatch.
//!
//! ## Empirical calibration (corpus-eval 2026-04-27)
//!
//! Pre-0.1.0-ship calibration baseline measured on a 219-image
//! labeled corpus from `coefficient/benchmarks/classifier-eval/labels.tsv`
//! (174 photo, 36 screen, 9 illustration, 44 marked synthetic;
//! pooled from cid22-train/val, clic2025-1024, gb82, gb82-sc,
//! imageflow, kadid10k, qoi-benchmark). Full per-class
//! distributions, ROC-AUC ranking for every feature, recommended
//! operating thresholds, and the recalibration candidates that
//! were considered and rejected are recorded in
//! `docs/calibration-corpus-2026-04-27.md`.
//!
//! Top-line empirical findings:
//!
//! - **Strongest single screen-vs-photo discriminator**:
//!   [`feature::AnalysisFeature::PatchFraction`] (AUC = 0.880,
//!   F1 = 0.769 at `>= 0.27`).
//! - **Strongest photo classifier**:
//!   [`feature::AnalysisFeature::NaturalLikelihood`] (F1 = 0.924
//!   at `>= 0.06`).
//! - **Near-deterministic line-art signal**:
//!   [`feature::AnalysisFeature::LineArtScore`] `> 0`
//!   (F1 = 0.978).
//! - **Derived likelihoods empirically saturate at ~0.70**, not
//!   1.0 — operating thresholds live in the 0.3–0.6 band, not 0.8+.
//!
//! Spearman ρ |≥ 0.85| pairs (mostly structural, all kept):
//!
//! - `distinct_color_bins` ↔ `palette_density`: ρ = +1.00 — derived.
//! - `flat_color_block_ratio` ↔ `screen_content_likelihood`: ρ =
//!   +0.99 — structural, the former is the latter's primary input.
//! - `uniformity` ↔ `aq_map_mean`: ρ = −0.97. Both measure block
//!   flatness; drive different knobs (T1 fast-path vs T3 AQ-driven
//!   trellis-λ).
//! - `chroma_complexity` ↔ `colourfulness`: ρ = +0.97. Both
//!   quantify chroma spread; one is normalised, the other is the
//!   raw Hasler-Süsstrunk M3 published scale — keep both for
//!   ergonomics.
//! - `aq_map_mean` ↔ `noise_floor_y`: ρ = +0.91. Both fall when
//!   blocks are flat. Different downstream knobs (zenjpeg trellis-λ
//!   vs `pre_blur`), keep both.
//! - `noise_floor_y` ↔ `screen_content_likelihood`: ρ = −0.89.
//!   Screen content is clean-by-construction; photos carry sensor
//!   noise. Different passes, different knobs.
//! - `cb_sharpness` ↔ `cr_sharpness`: ρ = +0.89. Co-vary by
//!   construction in natural content.
//!
//! No deletion candidates were found. Numeric drift in 0.1.x
//! patches is permitted by the threshold contract above; the
//! committed `docs/calibration-corpus-2026-04-27.md` is the
//! pre-ship baseline against which patch drift can be compared.

// `#[archmage::autoversion]` generates dispatch trampolines that
// don't always reference the unversioned base function.
#![allow(dead_code)]

mod alpha;
pub mod feature;
pub(crate) mod luma;
mod palette;
pub(crate) mod row_stream;
pub(crate) mod tier1;
pub(crate) mod tier2_chroma;
pub(crate) mod tier3;
#[cfg(feature = "experimental")]
pub(crate) mod tier_depth;

use core::fmt;

use zenpixels::{PixelDescriptor, PixelSlice};

use row_stream::RowStream;

/// Errors returned from the public analyzer entries.
///
/// Variants are stable; new variants will be added behind
/// `#[non_exhaustive]` so a `match` on this type must include a `_`
/// arm. The `Display` form is suitable for surfacing to end users
/// (concise, no internal type names); the `source()` chain points at
/// the underlying cause for `Convert` and `Internal` variants.
#[non_exhaustive]
#[derive(Debug)]
pub enum AnalyzeError {
    /// `zenpixels_convert::RowConverter` couldn't build a converter
    /// for the source descriptor (e.g. CMYK without a CMS plugin
    /// loaded, or an unsupported alpha mode). The analyzer accepts
    /// every layout `RowConverter` can ingest, so this variant is
    /// reserved for genuinely-unsupported descriptors.
    Convert(String),
    /// Caller-supplied raw bytes don't form a valid `PixelSlice` for
    /// the declared dimensions / descriptor (e.g. the buffer is
    /// too short, the stride is below the per-row byte minimum, or
    /// alignment doesn't match the channel type). Used by the
    /// `try_analyze_features_rgb8` and any future `try_*` entry that
    /// doesn't accept a pre-validated [`PixelSlice`]. Production
    /// callers can pattern-match on this variant to distinguish
    /// "user input was malformed" from "system ran out of resources"
    /// without parsing the message.
    InvalidInput(String),
    /// Heap allocation refused — the analyzer's working buffers are
    /// proportional to image width × tier (largest single chunk is
    /// `8 × width × 3` for the Tier 3 block-row scratch on a 4 K
    /// image, so ~96 KiB; total per-call working set runs ~265 KiB
    /// at 4 K).
    ///
    /// Currently **never returned** by the public APIs because every
    /// allocation is infallible (`vec![]` / `Box::new`); the variant
    /// exists so that a future `try_*` entry that does fallible
    /// allocation (`Vec::try_reserve`, etc.) has a stable, pattern-
    /// matchable home for the OOM signal. Production callers handling
    /// untrusted images should match this variant explicitly so the
    /// switch from infallible to fallible internals is a non-event.
    ///
    /// `bytes_requested` is the size of the rejected allocation when
    /// known; `None` means the failing path didn't carry the size to
    /// the error site.
    OutOfMemory {
        /// Size of the rejected allocation, in bytes. `None` when
        /// the failing path didn't carry the size into the error.
        bytes_requested: Option<usize>,
    },
    /// An unexpected failure that doesn't fit the other variants.
    /// Strings are kept opaque on purpose — production code should
    /// not pattern-match on the message.
    Internal(String),
}

impl fmt::Display for AnalyzeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AnalyzeError::Convert(msg) => write!(f, "row conversion setup failed: {msg}"),
            AnalyzeError::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            AnalyzeError::OutOfMemory { bytes_requested } => match bytes_requested {
                Some(n) => write!(f, "out of memory: requested {n} bytes"),
                None => write!(f, "out of memory"),
            },
            AnalyzeError::Internal(msg) => write!(f, "internal error: {msg}"),
        }
    }
}

impl core::error::Error for AnalyzeError {}

/// Run the analyzer over any [`PixelSlice`] for the requested feature
/// set. Returns an opaque [`feature::AnalysisResults`] holding only
/// the features the query asked for.
///
/// `query` is built from a [`feature::FeatureSet`] composed via
/// `const fn` set math:
///
/// ```ignore
/// use zenanalyze::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};
/// const MY_FEATURES: FeatureSet = FeatureSet::new()
///     .with(AnalysisFeature::Variance)
///     .with(AnalysisFeature::EdgeDensity);
/// let results = zenanalyze::analyze_features(slice, &AnalysisQuery::new(MY_FEATURES))?;
/// let var = results.get_f32(AnalysisFeature::Variance);
/// ```
///
/// Sampling budgets (Tier 1/2 stripe sampling, Tier 3 block cap) are
/// crate invariants — see [`feature::AnalysisQuery`] for the rationale
/// and the `#[doc(hidden)]` `__internal_with_overrides` backdoor for
/// tests / oracle re-extraction.
///
/// # Dispatch
///
/// Inspects the requested [`feature::FeatureSet`] and computes four
/// `const bool` axes (`PAL`/`T2`/`T3`/`ALPHA`). Sixteen
/// monomorphized [`analyze_specialized`] instantiations cover every
/// combination; inside each, unrequested tiers become straight-line
/// const-eval'd dead code. A caller asking for only `Variance` runs
/// Tier 1 and nothing else.
///
/// # Layered defense against garbage outputs
///
/// 1. Dispatch axes are unioned with derived-feature dependency
///    closures ([`feature::T3_NEEDED_BY`], [`feature::PAL_NEEDED_BY`])
///    so e.g. asking for `ScreenContentLikelihood` correctly gates
///    the palette pass on (it reads `distinct_color_bins`).
/// 2. [`tier3::compute_derived_likelihoods`] is itself const-bool
///    gated — even if the dispatch axes drift, a likelihood whose
///    deps weren't computed is left at default and never written.
/// 3. [`feature::AnalysisResults::get`] only returns values that
///    were both requested and written. Unrequested or unwritten
///    features come back as `None`. **Caller never sees garbage.**
///
/// # Wide gamut, HDR, and bit-depth policy
///
/// **The analyzer accepts every layout `zenpixels_convert::RowConverter`
/// can ingest.** Specifically:
///
/// - **24-bpp packed RGB** (RGB8 / RGB8_SRGB / RGB8 with Display P3 /
///   Rec.2020 / AdobeRGB primaries / non-sRGB transfer) — passes through
///   the zero-copy [`row_stream::RowStream`] `Native` path, byte-for-byte.
///   The analyzer's BT.601 luma weights produce slightly-different
///   numbers for non-sRGB primaries, and that's the principled outcome
///   — wide-gamut content has more saturated colour, so chroma /
///   colourfulness signals legitimately read higher.
/// - **RGBA8 / BGRA8 / Rgbx8 / Bgrx8** — alpha is analysed by the
///   separate alpha pass that reads the source descriptor directly.
///   The RGB tiers see a row-converted RGB8 view; for opaque pixels
///   that's identity, for translucent pixels it depends on the
///   converter's straight-alpha policy.
/// - **RGB16 / RGBA16** — converted to RGB8 by taking the high byte of
///   each channel. Crucially: an 8-bit image promoted to 16 bits via
///   the standard `u8 * 257` (or `u8 << 8`) doubling produces the same
///   high byte back, so analyzer features on a u8-promoted u16 source
///   are *bit-identical* to the original u8 (locked by tests). Genuine
///   16-bit content's extra precision feeds derived signals where
///   it's algorithmically meaningful (palette bins are 5-bit-per-
///   channel; sub-LSB drift in the bottom 8 bits doesn't change
///   uniformity / variance categorically).
/// - **f32 RGB / RGBA (linear or HDR)** — converted to display-space
///   RGB8 via the descriptor's transfer function. For SDR f32 content
///   normalized to `[0, 1]`, the round-trip from a u8 promotion is
///   bit-identical. For PQ / HLG / linear-light HDR, the analyzer
///   measures the SDR rendition — that's the principled choice for
///   features whose threshold contract is calibrated on display-space
///   bytes; HDR-aware analysis (peak luminance, HDR pixel fraction,
///   wide-gamut peak) lives in the future `tier_depth` work
///   (issue #120) and is computed directly on the source samples.
/// - **Gray8 / Gray16 / GrayF32 / GrayA{8,16,F32}** — converted to RGB8
///   by replicating the luma channel. Chroma signals are then trivially
///   zero, which is correct.
///
/// The point: per-image codec decisions don't usually break on a few
/// LSBs of luma drift. They break on the analyzer refusing to run.
/// The previous "opt-in via `analyze_features_lossy_convert`" gate was
/// over-defensive — every supported wide-gamut / HDR / 16-bit source
/// either round-trips through conversion as identity (u8-promoted) or
/// produces a rendering that is the legitimate signal for SDR-
/// calibrated features. We accept everything, document the cases
/// where the rendering differs from a byte-for-byte input, and give
/// codecs a stable feature surface.
///
/// # Errors
///
/// - [`AnalyzeError::Convert`] if `RowConverter` doesn't support the
///   source descriptor (e.g. CMYK without a CMS plugin loaded). The
///   set of accepted descriptors is `zenpixels-convert`'s problem,
///   not the analyzer's.
/// - [`AnalyzeError::Internal`] for unexpected failures (e.g. a
///   malformed `PixelSlice` whose row stride doesn't match its
///   declared format).
pub fn analyze_features(
    slice: PixelSlice<'_>,
    query: &feature::AnalysisQuery,
) -> Result<feature::AnalysisResults, AnalyzeError> {
    let features = query.features();
    let pal = features.intersects(feature::PAL_NEEDED_BY);
    let t2 = features.intersects(feature::TIER2_FEATURES);
    let t3 = features.intersects(feature::T3_NEEDED_BY);
    let alpha = features.intersects(feature::ALPHA_FEATURES);

    // Pick the palette path: full-precision scan if any "exact count"
    // palette feature was requested; otherwise the early-exit scan
    // (much faster on photographic content). Computed once per call;
    // analyze_specialized_raw const-folds nothing on this axis.
    let palette_full_required = features.intersects(feature::PALETTE_FULL_FEATURES);
    // GrayscaleScore lives on the palette tier (full-scan, 100 %
    // coverage) but its per-pixel max/min gate isn't free — only run
    // the gate when the caller actually requested it.
    #[cfg(feature = "experimental")]
    let palette_wants_grayscale = features.contains(feature::AnalysisFeature::GrayscaleScore);
    #[cfg(not(feature = "experimental"))]
    let palette_wants_grayscale = false;

    // Tier 1 dispatch: skip the separate Laplacian SIMD row pass
    // when LaplacianVariance isn't requested. Load-bearing for
    // orchestrator-style callers like zenjpeg's ADAPTIVE_FEATURES,
    // which doesn't ask for LaplacianVariance and currently pays
    // the full cost of the separate SIMD row walk anyway.
    #[cfg(feature = "experimental")]
    let tier1_wants_laplacian = features.contains(feature::AnalysisFeature::LaplacianVariance);
    #[cfg(not(feature = "experimental"))]
    let tier1_wants_laplacian = false;

    // Tier 1 full-kernel gate: flip on for `Variance` /
    // `Colourfulness` / `EdgeSlopeStdev` / `LaplacianVariance` — the
    // accumulators inside `accumulate_row_simd`'s `if FULL` block.
    // See `feature::TIER1_FULL_FEATURES`.
    let tier1_full_kernel = features.intersects(feature::TIER1_FULL_FEATURES);
    // Tier 1 skin-gate: peeled off `wants_full_kernel` so callers
    // that only want `SkinToneFraction` don't pay for Variance /
    // Colourfulness / edge-slope, and vice versa. The BT.601 chroma
    // matrix + Chai-Ngan thresholds spent 12 vmovups + 13 broadcasts
    // of register-spill traffic when bundled — see `cargo asm` audit.
    let tier1_wants_skin = features.intersects(feature::TIER1_SKIN_FEATURES);

    // Depth tier is a runtime axis (not const-bool) — adding it to
    // the 16-arm dispatch would double to 32 arms for one extra
    // pass with negligible LLVM monomorphization wins. The depth
    // tier reads source bytes directly, no shared inner loop with
    // T1/T2/T3 to specialize, so a runtime branch costs nothing.
    let run_depth = features.intersects(feature::DEPTH_FEATURES);
    // Source descriptor — captured up front so we can hand it back to
    // codecs verbatim via `AnalysisResults::source_descriptor()` even
    // after the analyzer's RowConverter / RowStream consumes the slice.
    let source_descriptor = slice.descriptor();

    macro_rules! dispatch {
        ($pal:literal, $t2:literal, $t3:literal, $a:literal) => {{
            let (raw, geometry) = analyze_specialized_raw::<$pal, $t2, $t3, $a>(
                slice,
                feature::DEFAULT_PIXEL_BUDGET,
                feature::DEFAULT_HF_MAX_BLOCKS,
                palette_full_required,
                palette_wants_grayscale,
                tier1_wants_laplacian,
                tier1_full_kernel,
                tier1_wants_skin,
                run_depth,
            )?;
            Ok(raw.into_results(features, geometry, source_descriptor))
        }};
    }
    match (pal, t2, t3, alpha) {
        (false, false, false, false) => dispatch!(false, false, false, false),
        (false, false, false, true) => dispatch!(false, false, false, true),
        (false, false, true, false) => dispatch!(false, false, true, false),
        (false, false, true, true) => dispatch!(false, false, true, true),
        (false, true, false, false) => dispatch!(false, true, false, false),
        (false, true, false, true) => dispatch!(false, true, false, true),
        (false, true, true, false) => dispatch!(false, true, true, false),
        (false, true, true, true) => dispatch!(false, true, true, true),
        (true, false, false, false) => dispatch!(true, false, false, false),
        (true, false, false, true) => dispatch!(true, false, false, true),
        (true, false, true, false) => dispatch!(true, false, true, false),
        (true, false, true, true) => dispatch!(true, false, true, true),
        (true, true, false, false) => dispatch!(true, true, false, false),
        (true, true, false, true) => dispatch!(true, true, false, true),
        (true, true, true, false) => dispatch!(true, true, true, false),
        (true, true, true, true) => dispatch!(true, true, true, true),
    }
}

/// Const-bool-monomorphized analyzer body. Returns the dense
/// [`feature::RawAnalysis`] + [`feature::ImageGeometry`] — the public
/// [`analyze_features`] entry post-converts via
/// [`feature::RawAnalysis::into_results`] using the caller's
/// [`feature::FeatureSet`]. Each `if PAL { … }` arm is dead code in
/// instantiations where the axis is `false`; LLVM const-evals the
/// predicate and DCEs the branch entirely.
///
/// Tier 1 is always-on: if a caller dispatches through here they
/// almost always want at least one Tier 1 feature (variance /
/// edge_density / colourfulness / etc.). Adding a fifth axis to skip
/// it doubles the dispatch table for marginal benefit; defer until
/// there's data showing real callers asking for zero T1 features.
///
/// Sampling budgets `pixel_budget` / `hf_max_blocks` are runtime
/// values for the test / oracle override path. `analyze_features`
/// always passes the canonical [`feature::DEFAULT_PIXEL_BUDGET`] /
/// [`feature::DEFAULT_HF_MAX_BLOCKS`] constants.
fn analyze_specialized_raw<const PAL: bool, const T2: bool, const T3: bool, const ALPHA: bool>(
    slice: PixelSlice<'_>,
    pixel_budget: usize,
    hf_max_blocks: usize,
    palette_full_required: bool,
    palette_wants_grayscale: bool,
    tier1_wants_laplacian: bool,
    tier1_full_kernel: bool,
    tier1_wants_skin: bool,
    run_depth: bool,
) -> Result<(feature::RawAnalysis, feature::ImageGeometry), AnalyzeError> {
    let width = slice.width();
    let height = slice.rows();
    let geometry = feature::ImageGeometry::new(width, height);

    let alpha_stats = if ALPHA {
        alpha::scan_alpha(&slice, pixel_budget)
    } else {
        Default::default()
    };

    // Source-direct depth tier — no RowStream / RowConverter; reads
    // descriptor samples and decodes through the transfer function.
    // Behind the `experimental` cargo feature; gated off entirely
    // when the feature is disabled (then `run_depth` is dead-code-
    // eliminated since DEPTH_FEATURES is empty in that build).
    #[cfg(feature = "experimental")]
    let depth_stats = if run_depth {
        tier_depth::scan_depth(&slice, pixel_budget)
    } else {
        Default::default()
    };
    // Suppress unused-var warning when experimental is off.
    let _ = run_depth;

    let mut stream = RowStream::new(slice).map_err(AnalyzeError::Convert)?;

    // Palette routing: if any "full-precision" palette feature was
    // requested (DistinctColorBins / Chao1 / PaletteDensity), run
    // the full scan. Otherwise — when only quick-path features
    // (IndexedPaletteWidth / PaletteFitsIn256) are asked — use the
    // early-exit scan that bails as soon as the running count
    // exceeds 256. On photos this typically returns within ~10
    // image rows.
    let palette_stats = if PAL {
        if palette_full_required {
            palette::scan_palette(&mut stream, palette_wants_grayscale)
        } else {
            palette::scan_palette_quick(&mut stream)
        }
    } else {
        Default::default()
    };

    let mut raw = feature::RawAnalysis::default();

    if width >= 2 && height >= 2 {
        // Tier 1 dispatch knobs — currently just `wants_laplacian`,
        // which lets orchestrator-style callers (e.g. zenjpeg's
        // `ADAPTIVE_FEATURES`) skip the separate Laplacian SIMD row
        // pass when `LaplacianVariance` isn't requested.
        // `TIER1_EXTRAS_FEATURES` is the union of all features whose
        // accumulators add cost beyond the Tier 1 baseline; for now
        // only LaplacianVariance gates a runtime branch, but the
        // dispatch struct is laid out to grow more knobs (skin /
        // edge-slope / colourfulness const-fold) without churning
        // call sites.
        let t1_dispatch = tier1::Tier1Dispatch {
            wants_laplacian: tier1_wants_laplacian,
            wants_full_kernel: tier1_full_kernel,
            wants_skin: tier1_wants_skin,
        };
        tier1::extract_tier1_into_dispatch(&mut raw, &mut stream, pixel_budget, t1_dispatch);
        if T2 && width >= 3 && height >= 3 {
            tier2_chroma::populate_tier2(&mut raw, &mut stream, pixel_budget);
        }
        if T3 && width >= 8 && height >= 8 {
            tier3::populate_tier3(&mut raw, &mut stream, hf_max_blocks);
        }
        // Layered defense: const-bool gated. Refuses to write a
        // likelihood whose deps weren't computed, regardless of what
        // the dispatch axes decided.
        tier3::compute_derived_likelihoods::<T3, PAL>(&mut raw);
    }

    if ALPHA {
        raw.alpha_present = alpha_stats.present;
        raw.alpha_used_fraction = alpha_stats.used_fraction;
        raw.alpha_bimodal_score = alpha_stats.bimodal_score;
    }

    if PAL {
        // Quick-path signals — populated by both scan paths but the
        // RawAnalysis fields are gated behind `experimental` since
        // no in-tree codec consumes them yet.
        #[cfg(feature = "experimental")]
        {
            raw.indexed_palette_width = palette_stats.indexed_width;
            raw.palette_fits_in_256 = palette_stats.fits_in_256;
        }
        // Full-path-only signals — quick-path leaves these at 0; the
        // `into_results` filter drops them so callers who didn't ask
        // for them get `None` instead of a misleading 0.
        if palette_full_required {
            raw.distinct_color_bins = palette_stats.distinct;
            // PaletteDensity is experimental (derived from the
            // unflagged DistinctColorBins; no consumer wires it yet).
            #[cfg(feature = "experimental")]
            {
                let pixel_count = (width as f64) * (height as f64);
                let denom = pixel_count.clamp(1.0, 32_768.0);
                raw.palette_density =
                    (raw.distinct_color_bins as f64 / denom).clamp(0.0, 1.0) as f32;
                // GrayscaleScore is computed on the same full-scan walk
                // as the distinct-bin histogram. 100 % coverage is
                // load-bearing — the score is used downstream as a
                // binary classifier (`>= 0.99` ⇒ encode as grayscale),
                // and stripe-sampling at ~5 % budget would let one
                // colour pixel slip past the gate ~95 % of the time.
                raw.grayscale_score = if palette_stats.total_pixels > 0 {
                    let gray = palette_stats
                        .total_pixels
                        .saturating_sub(palette_stats.non_grayscale);
                    (gray as f64 / palette_stats.total_pixels as f64) as f32
                } else {
                    0.0
                };
            }
        }
        let _ = palette_stats; // silence unused on the all-experimental-off path
    }

    // Depth tier writeback (experimental only — the fields and the
    // `tier_depth` module are both `cfg(feature = "experimental")`).
    #[cfg(feature = "experimental")]
    if run_depth {
        raw.peak_luminance_nits = depth_stats.peak_nits;
        raw.p99_luminance_nits = depth_stats.p99_nits;
        raw.hdr_headroom_stops = depth_stats.headroom_stops;
        raw.hdr_pixel_fraction = depth_stats.hdr_pixel_fraction;
        raw.wide_gamut_peak = depth_stats.wide_gamut_peak;
        raw.wide_gamut_fraction = depth_stats.wide_gamut_fraction;
        raw.effective_bit_depth = depth_stats.effective_bit_depth;
        raw.hdr_present = depth_stats.hdr_present;
        raw.gamut_coverage_srgb = depth_stats.gamut_coverage_srgb;
        raw.gamut_coverage_p3 = depth_stats.gamut_coverage_p3;
    }

    Ok((raw, geometry))
}

#[doc(hidden)]
/// **Unstable. Tests / oracle re-extraction only.** Run every tier
/// with caller-supplied sampling budgets, returning the dense
/// internal record. No stable API, no public path.
pub fn __analyze_internal(
    slice: PixelSlice<'_>,
    query: &feature::InternalQuery,
) -> Result<feature::AnalysisResults, AnalyzeError> {
    let run_depth = query.features.intersects(feature::DEPTH_FEATURES);
    let source_descriptor = slice.descriptor();
    let (raw, geometry) = analyze_specialized_raw::<true, true, true, true>(
        slice,
        query.pixel_budget,
        query.hf_max_blocks,
        true, // override path always uses full palette scan
        true, // and includes the grayscale gate (test path wants every signal)
        true, // and the Laplacian SIMD pass
        true, // and the Tier 1 full kernel (luma stats / Hasler M3 / edge slope)
        true, // and the Tier 1 skin gate
        run_depth,
    )?;
    Ok(raw.into_results(query.features, geometry, source_descriptor))
}

#[cfg(test)]
/// Crate-internal full-fat analysis for tests: always runs every
/// tier so test assertions can read any field. Returns the dense
/// record + geometry; tests wrap it in `tests::TestOutput`.
pub(crate) fn analyze_full_raw_for_test(
    slice: PixelSlice<'_>,
    full_budgets: bool,
) -> Result<(feature::RawAnalysis, feature::ImageGeometry), AnalyzeError> {
    let pb = if full_budgets {
        usize::MAX
    } else {
        feature::DEFAULT_PIXEL_BUDGET
    };
    let hf = if full_budgets {
        4096
    } else {
        feature::DEFAULT_HF_MAX_BLOCKS
    };
    // Tests want every signal — always run the full palette path,
    // and the depth tier when it's compiled in.
    let run_depth = cfg!(feature = "experimental");
    analyze_specialized_raw::<true, true, true, true>(
        slice, pb, hf, true, true, true, true, true, run_depth,
    )
}

/// Convenience entry for callers holding a packed RGB8 buffer plus a
/// query. **Panics** on length mismatch or stride-construction failure.
///
/// Use this when the input is known-good (e.g. coming from a freshly
/// decoded image where length and stride are guaranteed by
/// construction). For untrusted input — anything where the buffer
/// length or dimensions came in over a wire / from disk / from an
/// FFI boundary — prefer [`try_analyze_features_rgb8`], which surfaces
/// the same problems through [`AnalyzeError::InvalidInput`] /
/// [`AnalyzeError::OutOfMemory`] instead of unwinding.
pub fn analyze_features_rgb8(
    rgb: &[u8],
    width: u32,
    height: u32,
    query: &feature::AnalysisQuery,
) -> feature::AnalysisResults {
    let w = width as usize;
    let h = height as usize;
    assert_eq!(
        rgb.len(),
        w * h * 3,
        "analyze_features_rgb8: RGB8 buffer size mismatch"
    );
    let stride = w * 3;
    let slice = PixelSlice::new(rgb, width, height, stride, PixelDescriptor::RGB8_SRGB)
        .expect("RGB8 PixelSlice from packed buffer");
    analyze_features(slice, query).expect("analyze never fails on RGB8")
}

/// Fallible parallel of [`analyze_features_rgb8`]. Returns
/// [`AnalyzeError::InvalidInput`] when the buffer length doesn't
/// match `width * height * 3` or the resulting stride is invalid;
/// returns [`AnalyzeError::OutOfMemory`] when (in a future fallible-
/// allocation build) the analyzer's working buffers can't be reserved.
///
/// **Recommended for production code that handles untrusted images.**
/// The internals share an analyzer pass with [`analyze_features_rgb8`];
/// the only difference is the error-reporting contract.
///
/// Today this fn never returns `OutOfMemory` because every internal
/// allocation is infallible — but the variant is part of the
/// [`AnalyzeError`] surface (it's `#[non_exhaustive]`) so that a
/// future minor release can tighten the internals to use
/// `Vec::try_reserve` / `Box::try_new` without breaking the public
/// signature. Production callers should match it explicitly today.
///
/// # Errors
///
/// - [`AnalyzeError::InvalidInput`] for buffer-length mismatches or
///   slice-construction failures.
/// - [`AnalyzeError::OutOfMemory`] from future fallible-alloc paths.
/// - [`AnalyzeError::Convert`] / [`AnalyzeError::Internal`] are not
///   reachable on the RGB8 fast path today, but listed here so a
///   `match` over the enum stays complete.
pub fn try_analyze_features_rgb8(
    rgb: &[u8],
    width: u32,
    height: u32,
    query: &feature::AnalysisQuery,
) -> Result<feature::AnalysisResults, AnalyzeError> {
    let w = width as usize;
    let h = height as usize;
    let need = w
        .checked_mul(h)
        .and_then(|wh| wh.checked_mul(3))
        .ok_or_else(|| {
            AnalyzeError::InvalidInput(format!(
                "{width}×{height} pixels overflows usize when computing buffer length"
            ))
        })?;
    if rgb.len() != need {
        return Err(AnalyzeError::InvalidInput(format!(
            "RGB8 buffer length mismatch: expected {need} bytes for {width}×{height}, got {}",
            rgb.len()
        )));
    }
    let stride = w * 3;
    let slice = PixelSlice::new(rgb, width, height, stride, PixelDescriptor::RGB8_SRGB)
        .map_err(|e| AnalyzeError::InvalidInput(format!("PixelSlice::new failed: {e:?}")))?;
    analyze_features(slice, query)
}

#[cfg(test)]
mod tests;
