# zenanalyze [![CI](https://img.shields.io/github/actions/workflow/status/imazen/zenjpeg/ci.yml?style=flat-square&label=CI)](https://github.com/imazen/zenjpeg/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/zenanalyze?style=flat-square)](https://crates.io/crates/zenanalyze) [![lib.rs](https://img.shields.io/crates/v/zenanalyze?style=flat-square&label=lib.rs&color=blue)](https://lib.rs/crates/zenanalyze) [![docs.rs](https://img.shields.io/docsrs/zenanalyze?style=flat-square)](https://docs.rs/zenanalyze) [![license](https://img.shields.io/crates/l/zenanalyze?style=flat-square)](https://github.com/imazen/zenjpeg#license)

Streaming image content analyzer for adaptive codec pipelines. One pass over a
`zenpixels::PixelSlice` extracts the numeric features that decision trees,
selectors, and per-image encoder configurators consume — variance, edge density,
chroma sharpness, palette population, DCT energy, alpha statistics, and derived
text/screen-content/natural likelihoods.

```toml
[dependencies]
zenanalyze = "0.1"
```

## Why

Modern adaptive codecs (zenjpeg, zenwebp, zenpng, future zen formats) all want
the same handful of cheap content features to pick a quality knob — *is this a
photograph or a screenshot? does it have alpha? is it palette-friendly?* —
before they run their expensive encoder. Re-deriving that from scratch in each
codec means three copies of the same Tier 1 SIMD scan, three slightly different
threshold contracts, and three independent oracle retrains every time the math
moves.

zenanalyze is the shared single-pass scanner: codecs ask for the feature set
they care about, the orchestrator unions the requests, and one walk over the
image returns every signal. Tiers gate themselves out when their outputs aren't
needed.

## Quick start

```rust
use zenanalyze::{
    analyze_features,
    feature::{AnalysisFeature, AnalysisQuery, FeatureSet},
};

const JPEG_FEATURES: FeatureSet = FeatureSet::new()
    .with(AnalysisFeature::Variance)
    .with(AnalysisFeature::EdgeDensity)
    .with(AnalysisFeature::HighFreqEnergyRatio);

const WEBP_FEATURES: FeatureSet = FeatureSet::new()
    .with(AnalysisFeature::Variance)
    .with(AnalysisFeature::AlphaPresent)
    .with(AnalysisFeature::AlphaUsedFraction);

let needed = JPEG_FEATURES.union(WEBP_FEATURES);
let results = analyze_features(slice, &AnalysisQuery::new(needed))?;

let variance = results.get_f32(AnalysisFeature::Variance);
let alpha    = results.get(AnalysisFeature::AlphaPresent)
                      .and_then(|v| v.as_bool());
```

`AnalysisFeature` is `#[non_exhaustive]` with stable `u16` discriminants —
retired ids stay reserved, new ids are sequential. `FeatureSet` has full
`const fn` set math (`union`, `intersect`, `difference`) so per-codec presets
compose at compile time. `AnalysisQuery` is intentionally opaque: sampling
budgets are crate invariants, not per-call knobs.

For a packed RGB8 buffer the convenience entry skips the `PixelSlice` ceremony:

```rust
let results = zenanalyze::analyze_features_rgb8(&rgb_bytes, width, height, &query);
```

Codecs that need the source's `PixelDescriptor` for encode-side decisions
(bit depth, primaries, transfer function, color model, alpha mode, signal
range) read it back without having to retain the original `PixelSlice`:

```rust
let descriptor = results.source_descriptor();
match descriptor.primaries {
    zenpixels::ColorPrimaries::Bt709     => /* sRGB-class */,
    zenpixels::ColorPrimaries::DisplayP3 => /* P3 wide-gamut */,
    zenpixels::ColorPrimaries::Bt2020    => /* Rec.2020 / HDR */,
    zenpixels::ColorPrimaries::AdobeRgb  => /* AdobeRGB */,
    _ => /* unknown / future */,
}
```

## What it computes

| Feature | Type | Description |
|---|---|---|
| `Variance` | f32 | Luma variance on the BT.601 [0, 255] scale. |
| `EdgeDensity` | f32 | Fraction of sampled interior pixels with `|∇L| > 20`. |
| `ChromaComplexity` | f32 | `√(Var(Cb) + Var(Cr))` over sampled pixels. |
| `CbSharpness` / `CrSharpness` | f32 | Mean per-axis chroma gradient. |
| `Uniformity` | f32 | Fraction of 8×8 blocks with luma variance < 25. |
| `FlatColorBlockRatio` | f32 | Fraction of 8×8 blocks with R/G/B ranges all ≤ 4. |
| `DistinctColorBins` | u32 | Distinct 5-bit-per-channel RGB bins observed. |
| `Cb*Sharpness` / `Cr*Sharpness` (×3 each) | f32 | Per-channel per-axis chroma sharpness. |
| `HighFreqEnergyRatio` | f32 | DCT AC energy ratio over sampled 8×8 luma blocks. |
| `LumaHistogramEntropy` | f32 | Shannon entropy of a 32-bin luma histogram (bits). |
| `AlphaPresent` / `AlphaUsedFraction` / `AlphaBimodalScore` | bool / f32 / f32 | Straight-alpha statistics. |
| `TextLikelihood` / `ScreenContentLikelihood` / `NaturalLikelihood` | f32 | Soft `[0, 1]` content-class scores. |

Behind the `experimental` cargo feature, organized by what they drive:

**Codec-orchestrator gap-fillers** (added 2026-04-27 from the codec-knob inventory; corpus-validated on CID22 / CLIC2025 / gb82):

| Feature | Drives |
|---|---|
| `GrayscaleScore` | zenjpeg `ColorMode::Grayscale`, png/avif/jxl gray paths |
| `AqMapMean` / `AqMapStd` | zenjpeg hybrid trellis λ, webp segments + sns_strength, avif vaq |
| `NoiseFloorY` / `NoiseFloorUV` | zenjpeg `pre_blur`, jxl `noise/denoise`, webp `sns_strength` |
| `LineArtScore` | webp `Preset::Drawing`, jxl modular path, png palette preference |

**Source-direct HDR / bit-depth tier** (read source samples without RowConverter, since RowConverter doesn't tonemap):

| Feature | Drives |
|---|---|
| `PeakLuminanceNits` / `P99LuminanceNits` | HDR encoder peak / target nits |
| `HdrHeadroomStops` / `HdrPixelFraction` | HDR vs SDR encode-mode selection |
| `WideGamutPeak` / `WideGamutFraction` | Gamut-clipping decisions before sRGB narrowing |
| `EffectiveBitDepth` | avif/jxl `bit_depth`, png `near_lossless_bits` |
| `HdrPresent` | Composite HDR-mode trigger |

**Research-stage**: `Colourfulness`, `LaplacianVariance`, `VarianceSpread`, `PaletteDensity`, `DctCompressibilityY`, `DctCompressibilityUV`, `PatchFraction`, `IndexedPaletteWidth`, `PaletteFitsIn256`. Their numeric scale or definition may change in 0.1.x patches.

## Wide gamut, HDR, and bit depth

`analyze_features` accepts every layout `zenpixels-convert::RowConverter` can
ingest — RGB8 / RGBA8 / BGRA8, RGB16 / RGBA16, RGB-F32 / RGBA-F32 (linear, sRGB,
PQ, HLG), grayscale variants, all primaries (sRGB / Display P3 / Rec.2020 /
AdobeRGB). One entry, no opt-in step.

The principle: per-image codec decisions don't usually break on a few LSBs of
luma drift, they break on the analyzer refusing to run.

**u8-promotion invariance** is locked by tests. An RGB8 image promoted to
RGB16 via the standard `u8 * 257` doubling, or to RGBF32 via `u8 / 255.0`,
produces *bit-identical* features to the original RGB8 source. Codecs that
upgrade from u8 to wider formats internally don't see different analyzer
answers. (Verified by garb's exact-identity narrowing
`(u16 * 255 + 32768) >> 16` for `u16 = u8 * 257`.)

**Wide gamut adapts the values, not the API.** RGB8 with Display P3 / Rec.2020
/ AdobeRGB primaries passes through the zero-copy `Native` row path. The
BT.601 luma weights produce slightly different numbers vs sRGB content — and
that's the principled outcome, since wide-gamut content is more saturated,
chroma signals legitimately read higher.

**HDR f32 inputs** are converted through their declared transfer function to
display-space RGB8 — a tonemapped SDR rendition, since the analyzer's
threshold contract is calibrated on display-space bytes. HDR-aware features
(peak luminance, HDR pixel fraction, wide-gamut peak) live in the future
`tier_depth` work tracked under issue #120; they read the source samples
directly, without going through `RowConverter`.

The only error you can get from a well-formed `PixelSlice` is
`AnalyzeError::Convert(...)` when a descriptor isn't supported by
`zenpixels-convert` (e.g. CMYK without a CMS plugin loaded).

## How it's organized

Three tiers, all dispatched through the requested `FeatureSet` at runtime:

- **Tier 1** — staggered-stripe sample (≈500K pixels, ~1 ms at 4K). Variance,
  edge density, chroma complexity, 8×8 uniformity / flat-block ratio, optional
  Hasler-Süsstrunk colourfulness and Laplacian variance.
- **Tier 2** — full-image 3-row sliding window. Per-channel per-axis chroma
  sharpness; forked from evalchroma 1.0.3.
- **Tier 3** — sampled 8×8 DCT blocks (1024-block budget). High-frequency energy
  ratio, 32-bin luma entropy, optional libwebp α compressibility, optional
  patch-fraction perceptual-hash detector.

A separate **palette pass** runs always-full-scan when any palette feature was
requested (counts converge slowly under sub-sampling). An **alpha pass** scans
the source descriptor's straight-alpha channel directly without going through
RowConverter — no precision loss for u16/f32 alpha.

## Threshold contract

Numeric thresholds and normalization scales are converging during the 0.1.x
line. Downstream consumers that compile-in fitted models (oracle decision
trees, content selectors) must pin to a specific zenanalyze patch version and
re-validate / retrain whenever they bump it. Every breaking numeric change
ships with a CHANGELOG entry. The contract freezes at 1.0.

## Test surface

100+ tests covering math invariants on synthetic inputs (solid colours,
horizontal bands, uniform luma distribution, palette-locked images), the full
16-arm dispatch matrix, every supported pixel format (RGB8 / RGBA8 / BGRA8 /
RGB16 / RGBA16 / RGB-F32 / RGBA-F32 / Gray8 / Gray16 / GrayF32 / GrayA8 /
GrayA16 / GrayAF32), tier sizes from 1×1 to 2048×2048, deterministic-input
bit-equality (catches accumulator non-determinism), and `AnalyzeError` Display
/ source coverage. Math locks use absolute tolerances chosen to clear ULP-level
f32 noise from SIMD tree reductions but catch any genuine architecture
divergence.

> Note on coverage tooling: the SIMD kernels in `tier1.rs`, `palette.rs`,
> `tier2_chroma.rs`, and `tier3.rs` use `#[magetypes(... v4, v3, neon,
> wasm128, scalar)]` to generate one source-level monomorphization per
> architecture tier. At runtime archmage's `incant!` dispatches to whichever
> the CPU supports; the other variants stay compiled but unreachable. Line-
> coverage tools count each variant separately, so the raw percentage on
> these files looks ≈30% on x86_64. Real coverage of executable code paths
> (counted on the dispatched variant only) is ≥95% across every module.

## License

AGPL-3.0-only OR LicenseRef-Imazen-Commercial. Commercial licensing available
from imazen — contact `lilith@imazen.io`.
