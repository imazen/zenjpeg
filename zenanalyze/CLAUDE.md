# zenanalyze — Claude project guide

## API stability — non-negotiable

**There will never be a 0.2.x.** The crate ships under 0.1.x forever
(or until a 1.0). Every change must fit within the additive contract:

- New `pub fn` / `pub struct` / `pub mod`: OK.
- New variant on a `#[non_exhaustive]` enum (`AnalysisFeature`,
  `FeatureValue`, `AnalyzeError`, etc.): OK.
- New field on a struct that's already private or `#[non_exhaustive]`:
  OK if the existing public constructors stay valid.
- Renaming, removing, or changing the signature of any existing public
  item: **NOT OK.** Even small changes ("rename a parameter for
  clarity", "tighten a return type") are forbidden once the item has
  shipped. Add a parallel item if you need different shape — see
  `try_analyze_features_rgb8` next to `analyze_features_rgb8`.
- Numeric / behavioural drift on existing features in 0.1.x patches is
  expected (the crate-level threshold contract spells it out) — but the
  *signatures* don't move.

If a future need can't be solved additively under 0.1.x, pause and
flag it to the user. Do not propose "ship it in the next major" —
there is no next major.

## Allocation contract

Today every internal allocation is infallible (`vec!` / `Box::new`).
The plan to flip to fallible (`Vec::try_reserve`, etc.) does not
require any signature change — `try_analyze_features_rgb8` and
`AnalyzeError::OutOfMemory { bytes_requested }` are already in place
to surface the OOM. When fallible internals land, no caller has to
recompile.

## Threshold contract

Numeric thresholds and normalisation scales drift during 0.1.x as
features get refined. Downstream consumers that compile-in fitted
models pin to a specific patch and re-validate when they bump.
Documented at the crate-level docstring in `src/lib.rs` and in the
README.

## Tier architecture quick reference

Five passes, gated by the requested `FeatureSet`:

- **Tier 1** — stripe-sampled RGB8 (variance, edges, chroma, uniformity, grayscale).
- **Tier 2** — full-image 3-row sliding window over RGB8 (per-axis Cb/Cr sharpness).
- **Tier 3** — sampled 8×8 DCT blocks on RGB8 (DCT energy, entropy, AQ map, noise floor, line-art, gradient fraction, patch fraction).
- **Palette** — full-image RGB8 (distinct color bins, indexed-palette signals).
- **Alpha** — stride-sampled, **reads source bytes directly** (no RowStream).
- **`tier_depth`** — stride-sampled, **reads source bytes directly** (HDR / wide-gamut / bit-depth signals; HDR signal would not survive RowConverter narrowing).

The Native-vs-Convert decision in `RowStream::new` only applies to
Tier 1/2/3 + Palette. Alpha and `tier_depth` always read the source.

## Don't

- Don't propose 0.2.x.
- Don't change a published function signature, even to "improve" it.
  Add a parallel `try_*` / `with_*` / `_into` variant if needed.
- Don't add new `expect()` / `unwrap()` to public entries that took
  untrusted input. The fallible parallels exist for a reason.
- Don't bake content-class assumptions into the analyzer. The job is
  to surface signals; the consumer (codec orchestrator) decides what
  to do with them.
