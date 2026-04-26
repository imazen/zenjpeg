# `EncoderConfig::auto_for` — design notes (issues #94 + #103)

Status: **DESIGN ONLY**, not yet implemented. The first attempt (commit
`912398cf`, force-removed before push to main) shipped a `ContentBucket`
enum + bucket-only lookup table; both decisions were wrong on second
review and are recorded here so the next attempt doesn't repeat them.

## What we're trying to build

A self-contained, no-deps API on zenjpeg that turns
"image bytes + target quality + perceptual-metric preference"
into a tuned `EncoderConfig`:

```rust
let config = EncoderConfig::auto_for(
    rgb_bytes,
    width,
    height,
    target_q,
    AutoForMetric::Ssim2,
);
```

The caller doesn't need to classify the image or know about the
underlying decision logic. zenjpeg owns the analyzer + dispatch
end-to-end.

## What we won't do

### No public `ContentBucket` enum

The first attempt exposed coefficient's 5-way `ImageContentType`
taxonomy (`PhotoNatural`, `PhotoDetailed`, `PhotoFlat`,
`ScreenContent`, `Illustration`) as `pub enum ContentBucket` on
zenjpeg's API surface. That's wrong:

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

### No bucket-keyed lookup table

The first attempt's dispatch was a 105-arm match on
`(ContentBucket, AutoForMetric, q_bin) → most_frequent_winner_codec`.
That throws away the predictive power that the actual decision
trees encode.

Concrete numbers from the 2026-04-25 oracle (in coefficient,
`scripts/fit_oracle_tree.py`):

| feature set                                | mean tree accuracy |
|--------------------------------------------|------------------:|
| 3 features (megapixels, aspect_ratio, bucket_idx) | ~0.55 |
| 30 features (full Tier 1+2+3 ImageFeatures)       |  0.84 |
| bucket-only (the dropped first attempt)           |  ~0.45 *  |

\* worse than the 3-feature baseline because the rules table picks
ONE winner per (bucket, q_bin, metric) cell; the actual tree splits
on `variance`, `high_freq_energy_ratio`, `cr_peak_sharpness`,
`edge_density`, `cb_horiz_sharpness`, `cb_peak_sharpness`,
`luma_histogram_entropy`, etc. and picks among 8-12 leaf classes
per cell.

So the wrong API is even worse than no API.

## What to build instead

### 1. `ImageAnalyzer` — fully encapsulated, internal

A struct that takes RGB pixels and computes whatever features the
trees need:

```rust
pub(crate) struct ImageAnalyzer { /* internal */ }

impl ImageAnalyzer {
    pub(crate) fn analyze(rgb: &[u8], w: u32, h: u32) -> AnalyzerOutput;
}

pub(crate) struct AnalyzerOutput {
    pub variance: f32,
    pub edge_density: f32,
    pub chroma_complexity: f32,
    pub uniformity: f32,
    pub high_freq_energy_ratio: f32,
    pub luma_histogram_entropy: f32,
    pub cb_horiz_sharpness: f32,
    pub cb_peak_sharpness: f32,
    pub cr_peak_sharpness: f32,
    pub natural_likelihood: f32,
    pub text_likelihood: f32,
    pub screen_content_likelihood: f32,
    // ... whatever the latest oracle tree splits on
}
```

The features are NOT public. They're an implementation detail of
the analyzer. Callers never see this struct.

### 2. Source upstream

`coefficient::analysis::feature_extract` (Tier 1 — variance,
edge_density, chroma_complexity, uniformity) is ~700 LOC of
SIMD-friendly Rust with no heavy deps; mostly direct ports of
mozjpeg/jpegli stats.

`coefficient::analysis::evalchroma_ext::populate_tier23` (Tier 2+3
— sharpness breakdown, likelihoods) is ~500 LOC and depends on
`evalchroma` (another sibling crate, ~2k LOC).

For Phase 1 of the zenjpeg-side analyzer, port just Tier 1 — that
gets us to ~0.65 accuracy per the fitter. Phase 2 brings in
Tier 2+3 (or pulls in `evalchroma` as a path-dep) for the full
0.84.

### 3. Generated dispatch — actual tree splits, not bucket lookup

`scripts/gen_auto_for.py` (in coefficient) gets rewritten to walk
each fitted sklearn `tree_` recursively and emit nested if/else
against `AnalyzerOutput` field names:

```rust
// PhotoNatural / Ssim2 / q40-59 — was a flat lookup,
// becomes the actual tree:
if features.high_freq_energy_ratio <= 0.42 {
    if features.variance <= 5800.0 {
        // ...
    } else {
        // ...
    }
} else {
    // ...
}
```

Each leaf returns an `EncoderConfig`. The fitter already serializes
`tree_` recursively (added to `fit_one_tree` in coefficient
2026-04-25); the generator just needs to consume the JSON instead
of looking up the most-frequent class per cell.

The bucket label is no longer the top-level dispatch — the top
level is the q_bin, then the feature splits. If the trees discover
that bucket-like distinctions are useful, they'll split on the
underlying features (variance, edge_density, etc.) themselves.

### 4. Public API

```rust
impl EncoderConfig {
    /// Build a perceptually-tuned config for the given image.
    pub fn auto_for(
        rgb: &[u8],
        width: u32,
        height: u32,
        target_q: u8,
        metric: AutoForMetric,
    ) -> Self {
        let features = ImageAnalyzer::analyze(rgb, width, height);
        auto_for_rules::dispatch(&features, target_q, metric)
    }
}

pub enum AutoForMetric {
    Ssim2,
    Butter,
}
```

That's it. No `ContentBucket`, no exposed feature struct. The
caller hands over pixels and gets a config.

## Integration with coefficient

`coefficient::analysis::recommend_zenjpeg_knobs` stays as the
research-side reference impl that returns coefficient's richer
`ZenjpegKnobs` struct (BD-RD parameters, chroma_distance_scale,
etc. that aren't in the auto_for surface). When zenjpeg's
`auto_for` lands, both implementations reference the same oracle
data and produce equivalent results for the knobs both can express.

## Open question: what about q_bins that won with mozjpeg-rs?

The 2026-04-25 oracle had 1 cell where mozjpeg-rs-420-e2 won
(`PhotoFlat / ssim2 / q90+`). zenjpeg can't dispatch to a different
encoder. Options:
- (current choice) Fall through to a sensible zenjpeg default.
- Run a small zenjpeg-internal RD loop at q90+ when the cell
  signals "mozjpeg territory" — i.e. recognize that we're in a
  regime where the oracle preferred a different encoder, and
  apply zenjpeg's most aggressive equivalent (e.g. `auto_optimize`
  + tuned tables).
- Document the gap and let the caller pick mozjpeg directly when
  they see the relevant feature signature.

## Order of operations

1. Land `ImageAnalyzer` (Tier 1 features) in `zenjpeg/src/analyze/`.
2. Update `gen_auto_for.py` (coefficient) to emit if/else trees
   against `AnalyzerOutput` fields, gated on Tier 1 only for v1.
3. Land the generated `auto_for_rules.rs` (now a real tree, not a
   lookup table) + the public `EncoderConfig::auto_for` API.
4. Phase 2: pull in Tier 2+3 features (port from coefficient or
   path-dep on `evalchroma`).
5. Phase 3: round-trip tests (every (q, metric, plausible feature
   vector) → builds a valid `EncoderConfig` without panic).
