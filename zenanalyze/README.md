# zenanalyze [![CI](https://img.shields.io/github/actions/workflow/status/imazen/zenjpeg/ci.yml?style=flat-square&label=CI)](https://github.com/imazen/zenjpeg/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/zenanalyze?style=flat-square)](https://crates.io/crates/zenanalyze) [![lib.rs](https://img.shields.io/crates/v/zenanalyze?style=flat-square&label=lib.rs&color=blue)](https://lib.rs/crates/zenanalyze) [![docs.rs](https://img.shields.io/docsrs/zenanalyze?style=flat-square)](https://docs.rs/zenanalyze) [![license](https://img.shields.io/crates/l/zenanalyze?style=flat-square)](https://github.com/imazen/zenjpeg#license)

Streaming image content analyzer for adaptive codec pipelines. One pass over a
`zenpixels::PixelSlice` extracts the numeric features that decision trees,
selectors, and per-image encoder configurators consume — variance, edge density,
chroma sharpness, palette population, DCT energy, alpha statistics, and (behind
opt-in cargo features) classifier-style content-class likelihoods plus
source-direct HDR / wide-gamut / bit-depth signals that codecs use to **detect
when a descriptor over-promises and the actual pixel content is encodable in
something smaller**.

```toml
[dependencies]
zenanalyze = "0.1"
# Opt in to source-direct HDR / wide-gamut / descriptor-gap signals:
# zenanalyze = { version = "0.1", features = ["experimental"] }
# Opt in to classifier-style composite scores (text / screen / natural /
# line-art likelihoods). The raw signals these consume are stable; the
# composite coefficients are calibration-driven and may drift in 0.1.x:
# zenanalyze = { version = "0.1", features = ["composites"] }
```

## Cargo features

| feature | what it gates | stability |
|---|---|---|
| _(default)_ | Stable raw signals: variance, edge density, chroma sharpness, DCT energy, alpha, palette, distinct-color bins, etc. | Numeric drift in 0.1.x bounded by the threshold contract; signatures frozen |
| `experimental` | Research-stage signals: PatchFraction, AqMapMean/Std, NoiseFloorY/UV, GradientFraction, source-direct HDR / wide-gamut / bit-depth | Metric definition or scale may change; opt in only if you re-validate per patch |
| `composites` | Classifier-style scores: TextLikelihood, ScreenContentLikelihood, NaturalLikelihood, LineArtScore | Hand-tuned weighted combinators; the combination coefficients drift as the corpus calibration matures. The raw signals they consume are stable. |

## Why

Modern adaptive codecs (zenjpeg, zenwebp, zenpng, zenavif, zenjxl) all want the
same handful of cheap content features to pick a quality knob — *is this a
photograph or a screenshot? does it have alpha? is it palette-friendly? is the
HDR flag stale? does this Rec.2020 file actually use the wider gamut?* — before
they run their expensive encoder. Re-deriving that from scratch in each codec
means three copies of the same Tier 1 SIMD scan, three slightly different
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
retired ids stay reserved, new ids are sequential, the wire format never breaks.
`FeatureSet` has full `const fn` set math (`union`, `intersect`, `difference`)
so per-codec presets compose at compile time. `AnalysisQuery` is intentionally
opaque: sampling budgets are crate invariants, not per-call knobs.

For a packed RGB8 buffer the convenience entry skips the `PixelSlice` ceremony:

```rust
// Panicking — for known-good inputs (freshly decoded buffers).
let r = zenanalyze::analyze_features_rgb8(&rgb_bytes, w, h, &q);

// Fallible — for untrusted input. Returns AnalyzeError::InvalidInput on
// length / stride mismatch and AnalyzeError::OutOfMemory on (future)
// fallible-allocation paths.
let r = zenanalyze::try_analyze_features_rgb8(&rgb_bytes, w, h, &q)?;
```

Codecs read the source `PixelDescriptor` directly off the result for encode-side
metadata decisions:

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

`FeatureSet::iter()` walks the contained features in `AnalysisFeature::id()`
order — convenient for sidecars, harnesses, and Python fitters that need to
enumerate the surface without hand-listing variants.

## What it computes

**Default features** (every codec gets these):

| Feature | Type | Description |
|---|---|---|
| `Variance` | f32 | Luma variance on the BT.601 [0, 255] scale. |
| `EdgeDensity` | f32 | Fraction of sampled interior pixels with `\|∇L\| > 20`. |
| `ChromaComplexity` | f32 | `√(Var(Cb) + Var(Cr))` over sampled pixels. |
| `CbSharpness` / `CrSharpness` | f32 | Mean per-axis chroma gradient. |
| `Uniformity` | f32 | Fraction of 8×8 blocks with luma variance < 25. |
| `FlatColorBlockRatio` | f32 | Fraction of 8×8 blocks with R/G/B ranges all ≤ 4. |
| `DistinctColorBins` | u32 | Distinct 5-bit-per-channel RGB bins observed. |
| `Cb*Sharpness` / `Cr*Sharpness` (Horiz/Vert/Peak) | f32 | Per-channel per-axis chroma sharpness. |
| `HighFreqEnergyRatio` | f32 | DCT AC energy ratio over sampled 8×8 luma blocks. |
| `LumaHistogramEntropy` | f32 | Shannon entropy of a 32-bin luma histogram (bits). |
| `AlphaPresent` / `AlphaUsedFraction` / `AlphaBimodalScore` | bool / f32 / f32 | Straight-alpha statistics. |

**Behind the `composites` cargo feature** (calibration-driven, may drift in 0.1.x):

| Feature | Type | Description |
|---|---|---|
| `TextLikelihood` / `ScreenContentLikelihood` / `NaturalLikelihood` | f32 | Soft `[0, 1]` content-class scores. Combine raw signals (variance, edge density, distinct-color bins, etc.) via hand-tuned weights. |
| `LineArtScore` | f32 | Soft `[0, 1]` rendered-line-art score from luma histogram bimodality + entropy. |

**Behind the `experimental` cargo feature**, organised by what they drive:

### Codec-orchestrator gap-fillers

| Feature | Drives |
|---|---|
| `GrayscaleScore` | zenjpeg `ColorMode::Grayscale`, AVIF `Yuv400`, png/jxl gray paths (~30–40% smaller for B&W) |
| `AqMapMean` / `AqMapStd` | zenjpeg hybrid trellis λ, webp segments + sns_strength, avif vaq |
| `NoiseFloorY` / `NoiseFloorUV` | zenjpeg `pre_blur`, jxl `noise/denoise`, webp `sns_strength`, zenrav1e `film_grain` |
| `LineArtScore` _(behind `composites`)_ | webp `Preset::Drawing`, jxl `splines` / `patches`, png palette preference |
| `GradientFraction` | jxl `with_force_strategy` (DCT16 / DCT32 selection), zenrav1e deblock strength |
| `SkinToneFraction` | photo-vs-other dispatch (one-direction signal, AUC 0.80) — webp `Preset::Photo`, jxl perceptual presets, jpeg chroma-aware quant |
| `EdgeSlopeStdev` | screen-vs-photo dispatch (AUC 0.84, second only to `PatchFraction`) — webp `Preset::Drawing` vs `Photo`, jxl modular vs VarDCT |

### Source-direct HDR / wide-gamut / bit-depth tier

These read source samples without going through `RowConverter`, since
`RowConverter` doesn't tonemap — a 4000-nit PQ source and a 100-nit-clipped
SDR source would otherwise produce byte-identical RGB8 streams.

| Feature | Drives |
|---|---|
| `PeakLuminanceNits` / `P99LuminanceNits` | AVIF `clli`, JXL `intensity_target`, HDR encoder peak |
| `HdrHeadroomStops` / `HdrPixelFraction` | HDR vs SDR encode-mode selection |
| `WideGamutPeak` / `WideGamutFraction` | "Linear value > 1.0" detection |
| `GamutCoverageSrgb` / `GamutCoverageP3` | **Descriptor-gap signal** — if a Rec.2020 source's pixels all live in the sRGB sub-gamut, encode at sRGB primaries and save the wide-gamut metadata + encoder modes |
| `EffectiveBitDepth` | AVIF / JXL `bit_depth`, png `near_lossless_bits` (catches u8-promoted u16) |
| `HdrPresent` | Composite "transfer claims HDR AND pixels are actually bright" — catches stale HDR flags |

### Research-stage

`Colourfulness`, `LaplacianVariance`, `VarianceSpread`, `PaletteDensity`,
`DctCompressibilityY`, `DctCompressibilityUV`, `PatchFraction`,
`IndexedPaletteWidth`, `PaletteFitsIn256`. Numeric scale or definition may
change in 0.1.x patches.

## Descriptor-gap detection

The analyzer's job is to spot the gap between what the descriptor *promises*
and what the data *actually carries*, so encoders don't bloat files paying
for capacity the source doesn't need.

| Gap | Signal |
|---|---|
| RGB declared, content is grayscale | `GrayscaleScore ≥ 0.99` |
| Wider primaries declared, content fits sRGB | `GamutCoverageSrgb ≥ 0.99` |
| Rec.2020 declared, content fits Display P3 | `GamutCoverageP3 ≥ 0.99` |
| HDR transfer declared, content is SDR | `HdrPresent == false` |
| u16 declared, content is u8-promoted | `EffectiveBitDepth == 8` |
| RGBA declared, alpha is constant 1.0 | `AlphaUsedFraction == 0` |
| Standard 8×8 transforms, content is smooth | `GradientFraction ≥ 0.5` (use larger DCTs) |
| HDR flag set, peak is dim | `PeakLuminanceNits < 200 && HdrPresent` (mis-tagged HDR) |

Each one is a place where a codec orchestrator can downcast metadata + encoder
modes before encoding, saving real bytes on real corpora.

## Wide gamut, HDR, and bit depth

`analyze_features` accepts every layout `zenpixels-convert::RowConverter` can
ingest — RGB8 / RGBA8 / BGRA8, RGB16 / RGBA16, RGB-F32 / RGBA-F32 (linear, sRGB,
PQ, HLG), grayscale variants, all primaries (sRGB / Display P3 / Rec.2020 /
AdobeRGB). One entry, no opt-in step. The principle: per-image codec decisions
don't usually break on a few LSBs of luma drift, they break on the analyzer
refusing to run.

**u8-promotion invariance is locked by tests.** An RGB8 image promoted to
RGB16 via the standard `u8 * 257` doubling, or to RGBF32 via `u8 / 255.0`,
produces *bit-identical* features to the original RGB8 source. Codecs that
upgrade from u8 to wider formats internally don't see different analyzer
answers. (Verified by garb's exact-identity narrowing
`(u16 * 255 + 32768) >> 16` for `u16 = u8 * 257`.)

**Wide gamut adapts the values, not the API.** RGB8 with Display P3 / Rec.2020
/ AdobeRGB primaries passes through the zero-copy `Native` row path with its
bytes intact. The standard tiers pick the **right luma matrix per source
primaries**: BT.601 weights for sRGB / BT.709 (preserving the trained-threshold
baseline — coefficient's existing thresholds were calibrated against this
matrix on sRGB content), BT.2020 weights for Rec.2020 sources, the Y row of
each primary set's RGB→XYZ matrix for Display P3 / AdobeRGB. Fixed-point
integer-luma scales are normalised to the same sum-220 libwebp baseline so a
pure-white pixel hits the same histogram bin regardless of source primaries —
what differs is the per-channel weight that lands it there. No conversion, no
clipping, just the right matrix. See `src/luma.rs`.

**HDR f32 / linear inputs.** Standard tiers see what an SDR display would
show — `RowConverter` clips out-of-[0, 1] linear values, applies the sRGB
OETF, and narrows to u8. That's the legitimate input for SDR-calibrated
thresholds; tonemapping a 4 000-nit highlight into a visible mid-tone
before measuring "high-frequency-energy ratio" would just lie about what's
there. The above-clip signal lives in `tier_depth`, which reads the source
samples directly via `PixelSlice::row` (bypassing `RowConverter` entirely)
and decodes through the descriptor's transfer function — sRGB / BT.709 /
Gamma 2.2 / Linear / PQ / HLG — to linear nits. Two views of the same
source: the SDR-display view for trained thresholds, the source-direct view
for HDR / wide-gamut signal.

The `tier_depth` reference convention is stable across 0.1.x:

| Transfer | Linear 1.0 maps to | Convention |
|---|---|---|
| `Srgb` / `Bt709` / `Gamma22` / `Linear` | 80 nits | sRGB display reference (IEC 61966-2-1) |
| `Pq` | 10 000 nits | SMPTE ST 2084 absolute |
| `Hlg` | 1 000 nits | nominal HLG broadcast |

The standard tiers' threshold contract is calibrated on display-space RGB8
bytes; the depth tier surfaces the additional metadata-gap and HDR signals
that the RGB8 narrowing destroys.

## Errors

```rust
#[non_exhaustive]
pub enum AnalyzeError {
    Convert(String),                                  // RowConverter setup failed
    InvalidInput(String),                             // user-supplied bad layout / length
    OutOfMemory { bytes_requested: Option<usize> },   // future fallible-alloc path
    Internal(String),                                 // unexpected
}
```

Production code handling untrusted images should pattern-match on
`InvalidInput` / `OutOfMemory` explicitly. Today every internal allocation is
infallible (so `OutOfMemory` is reserved, never returned by current builds);
the variant is part of the public surface so a future minor that flips
internals to `Vec::try_reserve` doesn't break anyone's `match`.

## How it's organised

Five passes, each gated by what the requested `FeatureSet` actually needs:

| Pass | Iterates over | Reads | Cost (4 MP) | Drives |
|---|---|---|---|---|
| Tier 1 | Stripe-sampled rows | RGB8 | ~1 ms | luma stats, edges, chroma, uniformity, grayscale |
| Tier 2 | 3-row sliding window | RGB8 | ~2 ms | per-axis Cb/Cr sharpness |
| Tier 3 | Sampled 8×8 DCT blocks | RGB8 | ~3 ms | DCT energy, entropy, AQ map, noise floor, line-art, gradient, patch fraction |
| Palette | Full image | RGB8 | ~1 ms | distinct color bins |
| Alpha | Stride-sampled rows | **Source bytes** | ~0.3 ms | alpha presence / used / bimodal |
| `tier_depth` (experimental) | Stride-sampled rows | **Source bytes** | ~0.5 ms HDR, ~0 SDR-fast-path | HDR / wide-gamut / bit-depth / gamut-coverage |

Tier 1/2/3 + Palette read RGB8 via `RowStream`, which has three internal paths:

- **Native** (zero-copy) — RGB8-byte-layout-compatible inputs. Sub-slice straight from the source.
- **StripAlpha8** (zero RowConverter, scratch-only) — RGBA8 / BGRA8 / Rgbx8 / Bgrx8. Tight strip-and-maybe-swap into the row scratch. Skips the RowConverter alloc + plan + per-row CPU work.
- **Convert** — everything else (16-bit, f32, grayscale, CMYK, …) goes through `RowConverter` row-by-row.

Alpha and `tier_depth` always read source bytes directly, never through
`RowStream` — a load-bearing detail for HDR (RowConverter doesn't tonemap;
its narrowing clips PQ / HLG into sRGB-display).

## Performance

Release build, AVX2, no `target-cpu=native`, full `FeatureSet::SUPPORTED`:

| Input | 4 MP | RowStream path |
|---|---|---|
| RGB8 / Rgbx8 with sRGB / wide-gamut primaries | 9.5 ms | `Native` (zero-copy slice subindex) |
| RGBA8 | 10.9 ms | `StripAlpha8` (garb SIMD strip) |
| BGRA8 | 12.0 ms | `StripAlpha8` (garb SIMD strip + swap) |
| RGB16 | 24.7 ms | `Convert` (zenpixels-convert RowConverter) |
| RGBA16 | 28.6 ms | `Convert` (zenpixels-convert handles strip + narrow) |

The RGBA8 strip uses `garb::bytes::rgba_to_rgb` / `bgra_to_rgb`
(SIMD-dispatched via `archmage::incant!`) — measured 7× faster than
the previous in-tree scalar strip on a 2048-px row, dropping the
RGBA8 overhead vs RGB8 baseline from +5.8 ms to +1.4 ms. The 16-bit
input paths cost more because RowConverter does transfer-function-
aware narrowing — that's the correct tool for genuinely
heterogeneous input.

Per-call working-set memory is ~265 KB across ~7 allocations (largest single
chunk is the Tier 1 stripe scratch at 9 × width × 3 = 108 KB at 4 K). All
allocations are infallible today; the `OutOfMemory` variant exists so a
future minor can flip them without API breakage.

## Empirical operating thresholds

Picked on a 219-image labeled corpus from coefficient
(`benchmarks/classifier-eval/labels.tsv`, spanning cid22-train/val,
clic2025-1024, gb82, gb82-sc, imageflow, kadid10k, qoi-benchmark). 174
photo, 36 screen, 9 illustration, 44 marked synthetic. F1 / AUC are
for binary screen-vs-photo classification.

**For codec-orchestrator dispatch ("is this a screen or a photo?"):**

| signal | threshold | F1 | AUC | notes |
|---|---|---|---|---|
| `line_art_score > 0` | any nonzero | 0.978 | 0.750 | near-deterministic — line art ⇒ screen-like |
| `natural_likelihood >= 0.06` | photo detection | 0.924 | 0.814 | high precision photo classifier |
| `patch_fraction >= 0.27` | screen detection | **0.769** | **0.880** | **strongest single screen discriminator** |
| `edge_slope_stdev >= 35` | screen detection | — | **0.844** | **second-strongest screen discriminator** — photos cluster 15–32, screens 32–58 |
| `screen_content_likelihood >= 0.60` | screen detection | 0.750 | 0.831 | derived from flat blocks + palette + chroma |
| `flat_color_block_ratio >= 0.53` | screen detection | 0.750 | 0.838 | raw — same F1 as the derived `_likelihood` |
| `skin_tone_fraction >= 0.05` | photo detection | 0.824 | 0.799 | one-direction (presence ⇒ photo); pigmentation-invariant Chai-Ngan YCbCr |
| `text_likelihood >= 0.30` | text detection | 0.682 | 0.774 | weaker but real |
| `grayscale_score >= 0.99` | grayscale dispatch | — | — | encoder gap-filler, near-binary on real grayscale |

**Note:** the three `*_likelihood` features empirically saturate at
~0.70 (not 1.0) on real content, because each is a weighted sum of
clamped sub-components that don't simultaneously max on real images.
**Don't threshold them at `>= 0.8` — nothing will fire.** Operating
points are in the 0.3–0.6 band. The exact corpus maxes are:

- `text_likelihood` max **0.71**
- `screen_content_likelihood` max **0.70**
- `natural_likelihood` max **0.69**

**For descriptor-gap detection** the thresholds are content-physical
(see the "Descriptor-gap detection" table above): `GrayscaleScore >= 0.99`,
`GamutCoverageSrgb >= 0.99`, etc. Those are spec-driven, not corpus-fit.

The full per-class distributions, ROC-AUC ranking for every feature,
Spearman redundancy matrix, and the recalibration findings that were
considered and rejected are recorded in
[`docs/calibration-corpus-2026-04-27.md`](docs/calibration-corpus-2026-04-27.md).
That file is the pre-0.1.0-ship empirical baseline; subsequent 0.1.x
patches that drift numerics should compare against it.

## Threshold contract

Numeric thresholds and normalisation scales drift during 0.1.x. Downstream
consumers that compile-in fitted models (oracle decision trees, content
selectors) must pin to a specific zenanalyze patch version and re-validate
when they bump it.

**There is no 0.2.x.** Every change in 0.1.x is additive — new variants on
`#[non_exhaustive]` enums, new parallel functions, never a signature change to
a shipped item. See `CLAUDE.md`.

## Test surface

130+ tests covering math invariants on synthetic inputs (solid colours,
horizontal bands, uniform luma distribution, palette-locked images, two-tone
line drawings, smooth gradients, pure noise), the full 16-arm dispatch matrix,
every supported pixel format (3 channel-types × 6 transfers × 4 primaries × 2
alpha = 144 sanity-matrix combinations), tier sizes from 1×1 to 4096×4096,
deterministic-input bit-equality (catches accumulator non-determinism),
u8-promotion bit-equality across u16 / f32 sources, HDR-survival (PQ ~1000-nit
content preserved end-to-end where standard tiers would have clipped to SDR),
gamut-coverage projections (saturated Rec.2020 green correctly fails sRGB
coverage), and `AnalyzeError` Display / source coverage. Math locks use
absolute tolerances chosen to clear ULP-level f32 noise from SIMD tree
reductions but catch any genuine architecture divergence.

> Note on coverage tooling: the SIMD kernels in `tier1.rs`, `palette.rs`,
> `tier2_chroma.rs`, and `tier3.rs` use `#[magetypes(... v4, v3, neon,
> wasm128, scalar)]` to generate one source-level monomorphisation per
> architecture tier. At runtime archmage's `incant!` dispatches to whichever
> the CPU supports; the other variants stay compiled but unreachable. Line-
> coverage tools count each variant separately, so the raw percentage on
> these files looks ≈30 % on x86_64. Real coverage of executable code paths
> (counted on the dispatched variant only) is ≥95 % across every module.

## License

AGPL-3.0-only OR LicenseRef-Imazen-Commercial. Commercial licensing available
from imazen — contact `lilith@imazen.io`.
