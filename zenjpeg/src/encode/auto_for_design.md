# `EncoderConfig::auto_for` — design notes

This file is the **design rationale**: why we made the choices we did,
post-mortems of earlier approaches, and what's still open.

For the **operational reference** (the analyzer features, the oracle
dispatch table, the `AutoForOptions` semantics, the provenance of
every claim), see
[`coefficient/docs/CODEC_SELECTION_REFERENCE.md`](https://github.com/imazen/coefficient/blob/main/docs/CODEC_SELECTION_REFERENCE.md).

Status: **shipped on main** at commit `7ac74510`. The full sklearn-tree
codegen replacement for the hand-distilled dispatch is still TODO.

## What we shipped

A self-contained, no-deps API that turns
"image bytes + target quality + perceptual-metric preference"
into a tuned `EncoderConfig`:

```rust
let config = EncoderConfig::auto_for(image, Quality::ApproxSsim2(82.0))?;
let config = EncoderConfig::auto_for_with(image, q, AutoForOptions::default()
    .effort(Effort::Balanced)
    .allow_xyb(true))?;
```

The caller doesn't classify the image or know about decision logic.
Zenjpeg owns the analyzer + dispatch end-to-end.

## What we explicitly didn't do

### No public `ContentBucket` enum

The first attempt (commit `912398cf`, force-removed before push to main)
exposed coefficient's 5-way `ImageContentType` taxonomy
(`PhotoNatural`, `PhotoDetailed`, `PhotoFlat`, `ScreenContent`,
`Illustration`) as `pub enum ContentBucket` on zenjpeg's API surface.
That's wrong:

1. The taxonomy is an implementation detail of one classifier (the
   one in `coefficient::analysis::image_adaptive`). A different
   classifier might split or merge categories. Burning the current
   labels into zenjpeg's public ABI commits to that taxonomy
   forever — every change is a breaking release.
2. Callers shouldn't have to classify; they're asking zenjpeg to do
   the work. A function that takes a discrete category as input
   pushes the hard part back to the caller.
3. Even when you have the bucket label, **it's not enough data**.
   See next section.

The bucket lives inside `pick_oracle()` as the internal `InferredBucket`
enum. Inferred from analyzer features via `infer_bucket()`. Not part
of the public surface.

### No bucket-keyed lookup table

The first attempt's dispatch was a 105-arm match on
`(ContentBucket, AutoForMetric, q_bin) → most_frequent_winner_codec`.
That throws away the predictive power that the actual decision
trees encode.

Concrete numbers from the 2026-04-25 oracle (in coefficient,
`scripts/fit_oracle_tree.py`):

| feature set                                       | mean tree accuracy |
|---------------------------------------------------|------------------:|
| 3 features (megapixels, aspect_ratio, bucket_idx) |  ~0.55 |
| 30 features (full Tier 1+2+3 ImageFeatures)       |   0.84 |
| bucket-only (the dropped first attempt)           |  ~0.45 *  |

\* worse than the 3-feature baseline because the rules table picks
ONE winner per (bucket, q_bin, metric) cell; the actual trees split
on `variance`, `high_freq_energy_ratio`, `cr_peak_sharpness`,
`edge_density`, `cb_horiz_sharpness`, `cb_peak_sharpness`,
`luma_histogram_entropy`, etc. and pick among 8–12 leaf classes
per cell.

So the wrong API is even worse than no API.

### No public AnalyzerOutput

`zenjpeg::analyze::AnalyzerOutput` and the per-tier helpers are
`pub(crate)` by default; gated to `pub` only via the `__test-utils`
feature so coefficient's parity harness can drive them. Don't rely
on this surface from a normal consumer.

The features list (and their normalization scales) is the
calibration substrate of the oracle trees. Exposing it would commit
us to keeping every feature stable — even when the next oracle run
shows a different feature set is what actually matters.

## What we built (current state)

### `ImageAnalyzer` — fully encapsulated, internal

Lives in `zenjpeg/src/analyze/`. Layered:

| Tier | File | Outputs |
|---|---|---|
| 1 | `tier1.rs` | variance, edges, chroma stats, uniformity, palette signals |
| 2 | `tier2_chroma.rs` | per-channel per-axis chroma sharpness (forked from `evalchroma 1.0.3`) |
| 3 | `tier3.rs` | luma histogram entropy, DCT high-freq energy ratio, derived likelihoods |

Pull RGB8 rows on demand from any `zenpixels::PixelSlice` via
`row_stream::RowStream`. Native zero-copy on RGB8 inputs;
`zenpixels-convert::RowConverter` runs row-by-row into a single-row
scratch otherwise. **No per-image RGB8 buffer ever materialized**.

Tier 1 holds a 9-row stripe scratch; Tier 2 a 3-row sliding window;
Tier 3 high-freq-energy an 8-row block-row scratch. Total scratch
never exceeds 9 × width × 3 bytes regardless of image height.

### Dispatch — hand-distilled manual tree

`pick_oracle(bucket, q_bin, metric)` returns subsampling +
use_xyb + trellis choice from the dominant winner per cell. Patterns
hand-distilled from the 70-cell `selector_tree_rules.json`:

- **q < 40**: hybrid trellis 4:2:0 progressive (33/70 cells).
  Lambda 12–16 per bucket × q-bin.
- **q ≥ 40, photo content**: XYB+4:4:4 + `Standard` trellis at the
  top of q for detailed photos; `Off` elsewhere.
- **q ≥ 40, synthetic content**: XYB+4:4:4 + `Off`.
- **PhotoNatural exception**: stays on hybrid 4:2:0 until q90+,
  where it crosses to XYB+4:4:4+Off.

This **isn't** the actual sklearn trees. It's a most-frequent-
winner-per-cell distillation. Accuracy in `selector_tree_rules.json`
ranges 0.43–1.00 (mean 0.84); the dispatch picks the modal answer,
which loses on the ~16% of cell-bound images that don't follow the
mode.

### Public API

```rust
impl EncoderConfig {
    pub fn auto_for(
        image: PixelSlice<'_>,
        quality: impl Into<Quality>,
    ) -> Result<Self, String>;

    pub fn auto_for_with(
        image: PixelSlice<'_>,
        quality: impl Into<Quality>,
        options: AutoForOptions,
    ) -> Result<Self, String>;
}

#[non_exhaustive]
pub struct AutoForOptions {
    pub allow_xyb: bool,
    pub allow_progressive: bool,
    pub effort: Effort,             // reuses zenjpeg::Effort
    pub restart_markers: RestartMarkers,
}

#[non_exhaustive]
pub enum RestartMarkers { Off, Auto, AutoSparse }
```

`Quality` carries both the target value AND the implicit metric:
`ApproxSsim2(score)` and `ApproxButteraugli(distance)` pick their
own oracle tree; `ApproxJpegli(q)` and `ApproxMozjpeg(q)` default
to ssim2.

## Open / future work

### Sklearn-tree codegen

The fitter (`coefficient/scripts/fit_oracle_tree.py`) emits the full
serialized tree per cell now (added by prior session at `serialize_node`).
A future `gen_auto_for.py` should walk that JSON and emit
`zenjpeg/src/encode/auto_for_rules.rs` with nested if/else against
`AnalyzerOutput` field names — closing the 16% mode gap.

The hand-distilled `pick_oracle()` body becomes the fallback that
matches when the codegen says "this cell is trivial / single-class".

### `max_iterations`

`AutoForOptions::max_iterations` is intentionally **not surfaced**.
It lands when the BD-RD / `zensim_iters` iterative search loop is
implemented. Until then exposing it would be a no-op callers might
depend on.

### XYB on PhotoNatural q40-89

The oracle's PhotoNatural q40-89 cells uniformly prefer hybrid 4:2:0
over XYB+4:4:4. `infer_bucket()` may misclassify a "natural-looking
but actually detailed" image (foliage, fabric textures) as
PhotoNatural and the dispatch will pick 4:2:0 when 4:4:4 would have
won. Workaround for callers who care: use `EncoderConfig::xyb()`
directly. Long-term fix: the codegen-emitted tree splits on
`high_freq_energy_ratio` and other detail signals, so it'll catch
the misclassification within the dispatch.

### mozjpeg-rs cells

The 2026-04-25 oracle had 1 cell where `mozjpeg-rs-420-e2-v0.5.4` won
(`PhotoFlat / butter / q90+`). Dispatch falls through to a sensible
zenjpeg config. Options for the future:
- Run a small zenjpeg-internal RD loop at q90+ when the cell signals
  "mozjpeg territory" — apply the most aggressive equivalent
  (`auto_optimize` + tuned tables).
- Document the gap and let the caller pick mozjpeg directly when
  they see the relevant feature signature.

## Order of operations (where we are)

1. ✅ `ImageAnalyzer` (Tier 1+2+3) on `pub(crate)`/`__test-utils`.
2. ✅ `EncoderConfig::auto_for` + `auto_for_with` + `AutoForOptions`.
3. ✅ Hand-distilled `pick_oracle()` from oracle JSON.
4. ⬜ `gen_auto_for.py` codegen → `auto_for_rules.rs` if/else trees.
5. ⬜ `max_iterations` once BD-RD search loop lands.

## Verifying changes

Always run the parity example before trusting any analyzer change:

```bash
cargo run --release --example zenjpeg_analyzer_parity \
    /home/lilith/oracle-d2-store/oracle-d2/sources [N]
```

Must report **0.0000 max drift** across all 18 numeric fields. The
trees were trained against the exact normalization scales in
coefficient's reference; any drift silently breaks dispatch
accuracy.

For the dispatch itself, 10 unit tests in `auto_for.rs::tests` cover
preset shapes, builder chaining, q_bin partition boundaries, restart
marker emission round-trip, image-dim XYB gate, and end-to-end
encode round-trips. Both default features and full-feature builds
must pass.
