# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial release. Image content analyzers extracted from `zenjpeg::analyze`
  so other zen codecs can share the same oracle-trained feature pipeline.
- `analyze(PixelSlice)` and `analyze_rgb8(&[u8], w, h)` entry points returning
  `AnalyzerOutput`.
- **Alpha analysis** (`alpha_present`, `alpha_used_fraction`, `alpha_bimodal_score`).
  Mirrors zenwebp's classifier alpha-histogram bimodality detector — bypasses
  the RowStream RGB8 conversion and reads the source `PixelSlice` alpha
  channel directly via `PixelFormat`-keyed byte offsets. Supports RGBA8/BGRA8/
  RGBA16/RGBAF32/GrayA8/GrayA16/GrayAF32; premultiplied alpha is treated as
  opaque (`alpha_present = false`) per coefficient's convention. Stride-sampled
  to share the `pixel_budget`.
- **Cross-codec features** added per the prior-art audit (Hasler-Süsstrunk 2003,
  Pech-Pacheco, libwebp `analysis_enc.c`, libjxl modular):
  - `colourfulness` — Hasler-Süsstrunk M3 colourfulness metric
    (σ_rg² + σ_yb² and μ_rg + μ_yb in opponent space). Drives JXL VarDCT
    chroma quant scale, GIF palette size, JPEG chroma subsampling.
  - `laplacian_variance` — variance of discrete 5-tap Laplacian over
    sampled luma. Classical Pech-Pacheco blur/sharpness axis. Drives JXL
    noise synthesis, AVIF film-grain, JPEG trellis effort.
  - `variance_spread` — `(max − min) / (max + 1)` of per-block luma
    variance, free piggyback on the existing tier-1 SIMD block-stats
    kernel. Captures spatial heterogeneity. Drives JPEG AQ, WebP SNS,
    AVIF cdef strength.
  - `dct_compressibility_y` — **libwebp `GetAlpha` exact algorithm** now:
    per-block 64-bin histogram of `|AC|/16`, α = `256 * last_non_zero / max_count`,
    averaged across sampled blocks. Folded into tier-3's existing DCT pass.
  - `dct_compressibility_uv` — **real chroma DCT pass** now: extends the
    block-extract loop to compute Cb/Cr per pixel (BT.601 integer-quantized,
    matching the luma scale), runs the same separable 1D-DCT + transpose,
    computes per-block α, takes `max(α_cb, α_cr)` per block. Reuses
    `dct1d_8` and `dct2d_8` so it's still LLVM-autovec-friendly. **Histogram
    bin divisor = 8 for chroma vs 16 for luma** because chroma DCT
    coefficients run ~half luma's; finer binning lifts chroma α from p50≈1
    into the same dynamic range as luma α (p50≈5).
  - `palette_density` — `distinct_color_bins / min(pixel_count, 32 768)`,
    a **scale-aware** fraction of populated 5-bit-per-channel bin space.
    Avoids the absolute-count saturation that makes raw `distinct_color_bins`
    drift with image resolution. Real-corpus discrimination: photos p50 ≈
    0.06–0.09, screen content p50 ≈ 0.009 — clean ~7× separation.
  - `distinct_color_bins_chao1` — **Chao1 estimator** (Chao 1984) of the
    true full-image distinct-bin count, derived from singletons and
    doubletons in the budget-sampled histogram. Reduces sampling error
    on multi-megapixel images at default budget: corpus convergence
    study (139 images, 1-50% sampling rates) shows raw count p50 error
    27%/13%/4% at 25%/50%/100% sampling vs Chao1 p50 error 12%/4%/0% —
    **2-3× improvement**. At full sampling, Chao1 = raw count.
    Implementation: 32 KB per-bin u8 saturating-counter array tracked
    alongside the existing presence-bitset, no extra row fetches.
  - `patch_fraction` — **real perceptual-hash patch detection** now:
    each sampled block produces a 32-bit DCT signature from the lowest
    16 zigzag-positioned AC coefficients (sign bit + above-threshold
    bit per coefficient). Sort-and-sweep across the small sampled set
    (≤256 by default) gives the fraction of blocks whose signature
    matches at least one other block. Real perceptual matching, ~10 µs
    overhead total.

  Implementation notes: colourfulness and Laplacian fold into tier-1's
  per-row SIMD pass (4 + 3 new f64 accumulators). The Laplacian
  precomputes BT.601 luma into 3 row-scratch buffers in a single linear
  pass before the 5-tap stencil — avoids computing luma 5× per pixel.
  variance_spread piggybacks on the existing tier-1 block-stats kernel.
  dct_compressibility_y reuses tier-3's already-extracted DCT
  coefficients. Net cost: +1.2 ms on a 4 MP image versus the pre-feature
  baseline.
- `analyze_with(slice, &AnalyzerConfig)` and `analyze_rgb8_with(...)` entry
  points accepting an explicit budget config. Mirrors the
  `coefficient::analysis::feature_extract::ExtractConfig` surface so callers
  migrating from coefficient see the same shape.
- `AnalyzerConfig` (`#[non_exhaustive]`) with `pixel_budget` (default
  500_000, the trained value) and `hf_max_blocks` (default 256, the trained
  value). `AnalyzerConfig::full()` disables sampling for reference scans.
- `AnalyzerOutput` is now `#[non_exhaustive]` so future tier additions
  (HDR luminance histogram, Oklab chroma stats, etc.) can land without a
  0.x-major bump.
- Pixel-format coverage tests across the descriptor matrix every zen* codec
  uses at its encoder boundary: RGB8/RGBA8/BGRA8, RGB16/RGBA16, RGBF32/RGBAF32,
  GRAY8/GRAY16/GRAYF32 — all sRGB and linear transfer paths exercised through
  `RowConverter`.
- CI now runs `cargo test -p zenanalyze` on every Test platform (ubuntu,
  ubuntu-arm, macos, macos-intel, windows, windows-arm), `cross test` on
  i686, and `cargo build` on wasm32-wasip1 (scalar + SIMD128).
- Sanity tests across image-size regimes (`tiny_*`, `medium_*`, `large_*`):
  defends well-formedness invariants (geometry, range bounds on every
  feature) at 1×1, 2×2, 3×3, 7×7, 8×8, 8×16, 16×8, 256×256, 512×256,
  2048×2048. Includes a determinism test (same input → bit-identical
  output) and a timing tripwire on 2048×2048 default-budget analyze
  (must complete in <1 s — flags accidentally O(N²) regressions or
  budget-plumbing breakage).

### Fixed (breaking — feature numerics shift)

The 0.1.x line is explicitly *not* a frozen contract; downstream
consumers that compile-in fitted models (oracle decision trees,
selectors) must pin to a specific zenanalyze patch version and
re-validate / retrain when bumping. The fixes below all change Tier 2
or Tier 3 feature outputs.

- **Tier 2 Cb/Cr asymmetry repaired.** `rgb_to_ycbcr_q` no longer
  halves Cb/Cr when `cr < 0`. Previously, gradient energies for
  cool/green-dominant edges were systematically half those of
  warm/red-dominant edges of equal magnitude. `cb_peak_sharpness`
  and `cr_peak_sharpness` may now be up to 2× higher on average for
  images whose dominant color edges fall in the `cr < 0` half-plane.
- **Tier 2 odd-width edge-column drop fixed.** `span = (width − 2) / 2`
  → `(width − 1) / 2`. Pre-fix, odd widths (3, 5, 7, …) systematically
  dropped the rightmost triplet column from horizontal chroma
  sampling. Most affected: tiny images and any image whose right
  edge holds the dominant chroma transition.
- **Tier 2 trailing-fragment drop fixed.** Trailing partial fragments
  are now flushed if they have any rows (`> 0`), not only if they
  exceed 16 triplet-rows (`> 16`). Pre-fix, **any image with
  `height ≤ 32`** silently emitted zero for every Tier 2 field, and
  taller images undersampled the bottom by up to ~32 pixel rows.
- **Tier 3 high-frequency split changed from raster `k ≥ 16` to
  zigzag `k ≥ 16`.** Previously the metric used JPEG raster order,
  which puts `(u=4, v=0)` ("highest horizontal frequency, zero
  vertical") on the *low* side and `(u=0, v=2)` ("low horizontal,
  low-mid vertical") on the *high* side — biasing toward vertical
  detail. The new zigzag split matches JPEG's natural "drop high AC
  first" scan order and is symmetric in horizontal/vertical detail.
- **Threshold contract relaxed during 0.1.x.** `lib.rs` doc no longer
  calls the contract "frozen". It now states the contract is
  iterating and downstream consumers must pin patch versions.

### Documentation

- Rewrote the misleading "the codegen pass can emit `if features.<name>`"
  claim in `AnalyzerOutput` doc. There is no codegen today: zenjpeg's
  `encode::adaptive::infer_bucket` hand-distills the oracle into
  thresholds. Doc now says so explicitly.

### Downstream impact

- `zenjpeg::encode::adaptive::infer_bucket` carries hand-derived
  cutoffs against the *old* analyzer outputs (`cb_peak_sharpness > 5.0`,
  `> 8.0`; `high_freq_energy_ratio > 0.30`). These are flagged
  `CALIBRATION-PENDING` in source — neither bucket choice is
  catastrophic (PhotoDetailed and PhotoNatural both ship sensible
  configs), but a corpus sweep should tighten the boundary before the
  next adaptive-quality recalibration pass.
- coefficient continues to vendor its own `evalchroma_ext.rs` copy.
  When it migrates to depend on zenanalyze, it picks up these fixes
  automatically.

### BREAKING — opaque feature-set API is now the only API

Legacy `AnalyzerOutput` / `AnalyzerConfig` / `analyze` / `analyze_with`
/ `analyze_rgb8` / `analyze_rgb8_with` are deleted from the public
surface. The only public entries are:

- `analyze_features(slice, &AnalysisQuery) -> Result<AnalysisResults, _>`
- `analyze_features_rgb8(rgb, w, h, &AnalysisQuery) -> AnalysisResults`

All public types are opaque (no `pub` fields anywhere on
[`feature::AnalysisQuery`] / [`feature::AnalysisResults`] /
[`feature::ImageGeometry`]). Sampling budgets are crate invariants;
the `#[doc(hidden)]` `__internal_with_overrides` backdoor exists for
oracle re-extraction tests but is not public-stable.

SIMD tier signatures changed from `&mut AnalyzerOutput` to
`&mut feature::RawAnalysis`. The dense flat record is now
`pub(crate)` and macro-generated alongside the rest of the feature
table — adding a feature is a single row at `features_table!`.

Side-effect-free property guaranteed: requesting feature *F* alone
produces the same numeric value as requesting *{F, …}*. Verified by
the `requesting_more_features_does_not_change_existing_values` test,
which runs every supported feature alone and against
`FeatureSet::SUPPORTED` and asserts bit-equality.

`compute_derived_likelihoods` is const-bool gated on `<T3, PAL>`
axes — likelihoods whose dependencies didn't run are left at default
and dropped by `into_results`. Dispatch axes use precomputed
[`feature::T3_NEEDED_BY`] / [`feature::PAL_NEEDED_BY`] supersets so
asking for a derived feature gates the right tier on. Layered defense:
even if dispatch is wrong, the gate prevents garbage; even if the
gate were bypassed, `into_results` only emits requested features.
**Caller never sees garbage — only `Some(real_value)` or `None`.**

### Opaque feature-set API (initial draft)

New public types (in module `feature`, draft — not yet promoted to
the crate root):

- `AnalysisFeature` — `#[non_exhaustive] #[repr(u16)]` enum with
  sequential discriminants `0..30`. Discriminants are **immutable
  once shipped**; retired variants keep their slot. `name()` is the
  only public accessor (matches the legacy `AnalyzerOutput` field
  names for JSON-sidecar continuity); `id` / `from_u16` /
  `is_active` are `pub(crate)`.
- `FeatureValue` — `#[non_exhaustive]` `Copy` enum (`F32` / `U32` /
  `Bool`) with type-checked `as_*` accessors and a lossless
  `to_f32()` coercion.
- `FeatureSet` — opaque `[u64; 4]` bitset, full `const fn` set math
  (`union` / `intersect` / `difference` / `intersects` / `contains`
  / `contains_all` / `is_empty` / `len`). The only "all features"
  entry point is `FeatureSet::SUPPORTED` (which intentionally may
  shrink with future cargo-feature gating); there is no `all()`.
- `AnalysisQuery` — opaque request handle. **Only public knob is
  the feature set** — sampling budgets are crate invariants
  calibrated against shipped oracle / threshold tables, exposed
  only via a `#[doc(hidden)]` `__internal_with_overrides` backdoor
  for tests / oracle re-extraction.
- `ImageGeometry` — opaque, `width()` / `height()` / `pixels()` /
  `megapixels()` / `aspect_ratio()`.
- `AnalysisResults` — opaque container; sparse
  `Vec<(AnalysisFeature, FeatureValue)>` storage sized to the
  requested set, no over-allocation. Public API is
  `get(feature) -> Option<FeatureValue>`, `get_f32` convenience,
  `requested()`, `geometry()`, `Debug`.

New entry points (additive — legacy `analyze` / `analyze_with` /
`AnalyzerOutput` / `AnalyzerConfig` retained for now):

- `analyze_features(slice, &AnalysisQuery) -> Result<AnalysisResults, _>`
- `analyze_features_rgb8(rgb, w, h, &AnalysisQuery) -> AnalysisResults`

Single-source-of-truth `features_table!` macro: one invocation in
`feature.rs` generates the `AnalysisFeature` enum, its
`id`/`from_u16`/`is_active`/`name` impls, the `pub(crate)` dense
`RawAnalysis` SIMD-target struct, the `RawAnalysis::into_results`
sparse-translator, `FeatureSet::SUPPORTED`, and
`From<&AnalyzerOutput> for RawAnalysis` — all in lockstep. Adding a
feature is one row at the table.

`analyze_features` dispatches via four `const bool` axes
(`PAL`/`T2`/`T3`/`ALPHA`). Each axis becomes straight-line
const-eval inside the monomorphized `analyze_specialized` body —
unrequested tiers are dead-code, no runtime branch, no tier-stats
computation. Sixteen monomorphizations cover all combinations.
Tier 1 is always-on (cheap, almost every caller wants something
from it). Measured at 8 MP, default target-cpu, runtime archmage
dispatch:

| Query | Time | ms/MP | vs legacy |
|-------|-----:|------:|----------:|
| Legacy (all features always) | 14.10 ms | 1.68 | 1.00× |
| **`Variance` only** | **3.35 ms** | **0.40** | **4.2×** |
| `DistinctColorBins` only | 6.55 ms | 0.78 | 2.2× |
| `DctCompressibilityY` only | 7.46 ms | 0.89 | 1.9× |
| `FeatureSet::SUPPORTED` (all) | 12.77 ms | 1.52 | 1.10× |

`AnalyzerConfig::pixel_budget` and `hf_max_blocks` demoted from
`pub` to `pub(crate)`. The legacy struct is now `Default`-only from
outside the crate; arbitrary budget overrides were silently
producing features that drifted from oracle-trained thresholds.

### Default — `hf_max_blocks` back to 1024 (accurate DCT features)

The cross-block f32x8 DCT cut Tier 3's per-block cost ~8-15× vs the
prior scalar separable DCT, so the convergence-elbow setting (1024
blocks, 4-6 % p50 error on DCT-energy features) now costs only ~+1 ms
at 8 MP over the 256-block sloppy default (11-16 % p50 error). The
earlier session reverted the default to 256 *because* of the scalar
DCT's perf cliff at 1024 (~+3 ms); with SIMD that pressure is gone,
and accurate features are unconditionally the right default.

| `hf_max_blocks` | DCT-energy p50 err | `patch_fraction` p50 err | 8 MP cost |
|---:|---:|---:|---:|
| 256 (old default) | 11-16 % | ~100 % | 0.95 ms |
| **1024 (new default)** | **4-6 %** | ~57 % | **1.95 ms** |
| 4096 | converged | structural limit | ~4 ms |

`patch_fraction` is still sample-limited at any block cap below
dense-scan and stays at 57 % p50 error at 1024 — acceptable for
content classification when combined with the other DCT features;
bump to 4096+ via `AnalyzerConfig::hf_max_blocks` if that single
feature dominates a downstream decision.

Total 8 MP runtime moves 9.40 → 11.38 ms (1.36 ms/MP) under the
accurate default. Non-Tier-3 work is still 1.08 ms/MP — the Tier 3
DCT pass at 1024 blocks accounts for the entire delta.

### Performance — palette flag-array + chunked autoversion

Replaced the 4 KB `[u64; 512]` presence bitset with a 32 KB `[u8; 32_768]`
flag array indexed by the 15-bit RGB-bin id. Each pixel does one
unconditional `flags[idx] = 1` byte store instead of the bitset's
read-modify-write `bits[i>>6] |= 1<<(i&63)` chain. Wrapped the entire
scan-and-count in a single `#[autoversion(v4x, v4, v3, neon, scalar)]`
function — one runtime dispatch per `scan_palette` call covering the
row loop, chunked-by-24-byte index compute, all the byte stores, and
the final `iter().sum()` reduction together.

The chunk pattern is a fixed-size `[u8; 24]` view with 8 unrolled
index computes per chunk. That fixed-size proof + the lack of
inter-iteration dependencies lets the v3/v4 autovectorizer issue the
shifts and ORs as a SIMD batch and emit the 8 byte stores together.
The 32 KB-byte reduction at the end SIMD-sums on the same v4 / v4x
path.

Standalone palette-only bench (4096×2048, 50-iter median):

| variant | photo input | saturated input | ratio |
|---------|-------------|-----------------|-------|
| bitset RMW (scalar) | 6.05 ms | 5.65 ms | 1.00× |
| bitset SIMD index | 6.08 ms | 5.69 ms | 1.00× *(no win — RMW dominates)* |
| flag array (scalar) | 4.12 ms | 4.82 ms | **1.47× / 1.17×** |
| flag array SIMD index | 4.59 ms | 5.20 ms | 1.32× / 1.09× *(slower than scalar)* |

Integrated zenanalyze 8 MP RGB8 (Ryzen 9 7950X, default target-cpu,
runtime archmage dispatch):

- Before this change: 12.13 ms (1.45 ms/MP)
- After flag array, no autoversion: 10.80 ms (1.29 ms/MP)
- After autoversion + chunks-of-24: **9.40 ms (1.12 ms/MP)**

Tier 3 contributes ~0.95 ms; the non-Tier-3 work runs at exactly
1.00 ms/MP. We're now at the pre-stated 1 ms/MP target for all
non-DCT analyzer work, and within ~12 % overall. **`-C target-cpu=
native` is now 10.86 ms — *slower* than the default-target build at
9.40 ms** because autoversion specializes the palette path to v4x
while letting the rest of the binary keep its smaller code-cache
footprint at v3. Confirmed bench, not measurement noise.

Three findings from this round, validated by bench:

- **SIMD index-compute is *slower* than scalar with the flag array.**
  An earlier magetypes `u32x8` shift+OR path that fed 8 stores per
  chunk ran ~10–15 % slower than the plain scalar loop. Once the
  bitset RMW dependency was gone, LLVM's autovec on the scalar path
  was already near-optimal — the deinterleave to arrays + vector
  load + store back round-trip costs more than just letting the
  autovectorizer see the linear pattern.
- **`#[autoversion]` per-row dispatch regressed.** Putting autoversion
  on a per-row helper paid the dispatch cost `height` times and went
  *slower* than plain scalar (10.80 → 12.72 ms at 8 MP). Wrapping the
  whole row+count loop reverses that — one dispatch amortizes over
  the entire scan and inside the v4x boundary the autovec is real.
- **`chunks_exact(3)` is too narrow for LLVM.** Switching to
  `chunks_exact(24)` with a `[u8; 24]` fixed-size view + 8 unrolled
  pixel-index computes was the unlock that let the autovectorizer
  see structure. Without that the bare 3-byte loop body looked too
  small to vectorize even under v4x.

The 32 KB working set is the only real cost: 8× larger than the
bitset, but still fits in L1D on every modern x86-64 / ARM core. One
`Box<[u8; 32_768]>` allocation per call, no per-row heap traffic.

### Performance — Tier 3 DCT + palette SIMD round

Tier 3's `dct2d_8` rewritten as a magetypes f32x8 kernel — three planes
batched per `incant!` call so dispatch cost amortizes 3× and the eight
column-of-D vectors stay hot in YMM registers. Each plane is now
`Y_row[v] = Σ_n splat(X[v][n]) * d_col[n]` (8 fmas per row) for the row
pass and `Z_row[v] = Σ_w splat(D[v][w]) * Y_row[w]` for the column pass —
no transpose, no horizontal reduction, no scalar dot products. ~128 fmas
per plane vs ~1920 scalar ops in the old separable-scalar version.

Palette tier moved from a per-pixel scalar loop to an `incant!`-per-row
magetypes u32x8 kernel that computes eight 15-bit
`((r>>3)<<10)|((g>>3)<<5)|(b>>3)` indices in parallel via lane-wise
shifts and ORs, then does eight scalar bitset ORs (the bitset-OR half
stays scalar — random scatter into a 4 KB table doesn't vectorize).

Measured on a Ryzen 9 7950X (release build, default target-cpu —
runtime archmage dispatch, no `-C target-cpu=native`):

| Image | Pre-DCT/palette | Post | Speedup | ms/MP |
|-------|-----------------|------|---------|-------|
| 1024×1024 | 4.49 ms | 4.45 ms | 1.01× | 4.24 |
| 2048×2048 | 7.93 ms | 7.79 ms | 1.02× | 1.86 |
| 4096×2048 | 12.67 ms | 12.13 ms | 1.04× | **1.45** |

Tier 3 alone dropped from ~1.0 ms to ~0.95 ms at 8 MP — small absolute
movement because Tier 3 was already well-bounded by the 256-block cap.
The DCT cleanup is mostly preparation for raising `hf_max_blocks` in a
future revision; at 1024 blocks the kernel cost would otherwise quadruple.

Palette improvement was modest (~5–8 %) because the bitset OR — eight
random 64-bit scatters per chunk — dominates over the index
computation. The SIMD index pipeline does help, but the OR latency is
the structural floor on this tier.

To reach the 1 ms/MP target, the remaining levers from the prior
analysis still apply: rayon parallelism across tiers, or a smarter
palette algorithm that batches the bitset writes (sort-and-sweep, or a
larger flag array with a final pop-count). With `-C target-cpu=native`
the same 8 MP run hits 1.33 ms/MP; that's the LLVM autovec ceiling
without further algorithmic work.

### Performance — full magetypes SIMD pass

Measured on a Ryzen 9 7950X (release build). Adds explicit
`#[magetypes(define(f32x8), v4, v3, neon, wasm128, scalar)]`-driven
SIMD on Tier 1's `accumulate_row` and Laplacian passes; reverts
`hf_max_blocks` back to 256 (1024 didn't fix `patch_fraction`'s
structural sample-sensitivity, just made it ~57 % vs ~100 %); drops
Chao1's redundant 32 KB counter array (raw count IS truth at full
scan).

| Image | Original | Final | Speedup | ms/MP |
|-------|----------|-------|---------|-------|
| 512×512 | 3.31 ms | 2.07 ms | **1.6×** | 7.9 |
| 1024×1024 | 6.82 ms | 4.46 ms | **1.5×** | 4.3 |
| 2048×2048 | 14.1 ms | 7.82 ms | **1.8×** | 1.9 |
| 4096×2048 | 25.3 ms | 12.6 ms | **2.0×** | **1.50** |

**2× faster overall** at 4 MP+ vs the original. ms/MP improves
super-linearly because per-image fixed overheads (alpha scan
allocations, `RowStream` setup) amortize across more pixels.

To reach the 1 ms/MP target, the remaining levers are all bigger:
- **Tier 3 cross-block DCT** (8 blocks per f32x8 batch): est. 3-4 ms
  savings at 4 MP+. Multi-day rewrite.
- **Rayon parallelism** across tiers: est. 2× via multi-core.
  Architectural change.
- **Skip palette full-pass for big images** (use Chao1 + budget at
  >2 MP): saves 4-5 ms but reintroduces palette sampling error.

The full-pass palette tier (~5 ms at 4 MP+) and the bumped block cap
(~3 ms) replace the earlier budget-sampled implementations whose corpus
convergence study showed **never-converging** behaviour for palette
features and 11-16 % p50 error on Tier 3 features. The new pipeline is
slower than the mid-stage 7.93 ms but produces correct outputs:
- exact `distinct_color_bins` and `palette_density` (was 27 % p50 error)
- DCT energy features at 4-6 % p50 error (was 11-16 %)
- usable `patch_fraction` (was 100 % p50 error at 256 blocks)

Tier-by-tier:

- **Tier 2 stride sampling (`pixel_budget` plumbing).** Same
  `compute_stripe_step` pattern Tier 1 uses, now applied to Tier 2.
  Caps Tier 2 at ~0.5 ms (with SIMD) regardless of image size. At
  4096×2048, Tier 2 went from 17.4 ms → 0.48 ms (36×). Pass
  `usize::MAX` as `pixel_budget` to disable.
- **Tier 2 magetypes f32x8 SIMD inner loop.** Replaced the scalar
  every-other-column triplet loop with explicit f32x8 vectorization
  via `#[magetypes(define(f32x8), v4, v3, neon, wasm128, scalar)]`.
  Processes 8 column triplets per iter, deinterleaves each row once
  into per-channel f32 scratch (10 pixels for row0, 8 each for
  row1/row2), then takes overlapping `f32x8::load` slices for the
  a/b/c triplet positions. Periodic reduce to f64 every 32 iters
  bounds precision loss. **2× over scalar** at every size; 4096×2048
  full scan dropped 16.5 ms → 8.5 ms. Also processes every column
  triplet (was every other) — same data, denser sampling, no extra
  cost.
- **Tier 3 fast DCT.** Replaced the per-iteration `.cos()` calls in
  the naive 8×8 DCT with a precomputed `DCT_COEF[8][8]` constant
  matrix and a clean `dct1d_8` matrix-vector helper. LLVM
  auto-vectorizes the 8-element inner products into AVX2 mul-add
  chains. Tier 3 dropped 9-10× at small images and 1.4× at 4 MP+
  (where the entropy-histogram pass dominates over DCT).

- **Tier 1 magetypes f32x8 block-stats kernel.** The 8×8 block stats
  loop (luma uniformity + per-channel flat-color detection) was fully
  scalar. Replaced with a `#[magetypes(define(f32x8), v4, v3, neon,
  wasm128, scalar)]` kernel that processes one block-row (8 pixels)
  per iter using lane-wise `min` / `max` / `mul_add` accumulators
  reduced to scalar at end-of-block. ~10-15% on Tier 1.

Tier 1's `accumulate_row` (luma/chroma/edges) is already
`#[archmage::autoversion]`-vectorized and dominates the remaining ~3.6
ms. Further wins require switching the seven f64 accumulators to
f32x8 with periodic reductions — non-trivial precision work, filed as
follow-up. The color-bins update (gather-bound histogram) is the
other significant remaining cost; SIMD-ing the index calculation
without a scatter primitive is also follow-up.

### Tests

- 5 new regression tests lock the FIXED behavior:
  `tier2_cb_cr_no_longer_asymmetric_under_cr_sign_flip`,
  `tier2_odd_width_5_includes_right_edge_triplet`,
  `tier2_small_height_no_longer_silently_zeroes`,
  `tier2_bottom_edge_partial_fragment_is_counted`,
  `tier3_zigzag_split_is_symmetric_in_horiz_vs_vert_detail`.
  Any regression that re-introduces one of the four bugs fails the
  matching test.
- Tier 1: stripe-sampled variance, edge density, chroma complexity,
  uniformity, flat-color blocks, 5-bit palette bins. `archmage::autoversion`
  on the inner row scan (v3 / NEON / WASM128 / scalar).
- Tier 2: per-channel per-axis chroma sharpness (`cb_horiz`/`cb_vert`/`cb_peak`
  and `cr_*`). Three-row sliding-window port of evalchroma 1.0.3
  `image_sharpness` with fragment-based normalization.
- Tier 3: 32-bin luma histogram entropy, naive 8×8 DCT high-freq energy ratio
  (capped at 256 sampled blocks), derived text/screen/natural likelihoods.
- `RowStream`: pulls RGB8 rows on demand from any `zenpixels::PixelSlice`,
  zero-copy on RGB8/RGB8_SRGB inputs and one-row scratch on every other
  format via `zenpixels-convert::RowConverter`.
