# Per-window streaming diffmap measurement (zenjpeg #113 PR-F)

Design doc for the per-window streaming diffmap measurement layer of the
target-zq closed-loop encoder. Context: the iteration loop currently
runs a full-image diffmap per pass, which dominates encode cost on large
images. This doc proposes amortizing the cost via a per-window streaming
consumer in zensim.

Status: design only. No implementation yet. Coordination needed with
upstream zensim before zenjpeg-side wiring.

## Problem

Today's iteration loop in `BytesEncoder::finish_with_metrics` (after
PR-B / #117) does, per pass:

```
encode → decode → compute_with_ref_and_diffmap (FULL IMAGE) → adjust
```

`compute_with_ref_and_diffmap` cost on a 4K image is ~91 ms threaded,
~366 ms single-threaded (per zensim README). For typical 2-pass
encodes, the diffmap cost is ~30–50 % of total time. On larger images
this fraction grows — at 8K it dominates.

## Goal

Replace the full-image measurement with a per-window streaming consumer
that:

1. Accepts strips of distorted pixels as the encoder produces them
   (every K iMCU rows, where K is small — 4 to 8 strips per encode).
2. Maintains rolling `ScaleAccumulators` against a precomputed source
   reference.
3. Emits per-strip "score contribution" deltas the controller can use
   between pass boundaries.
4. Finalizes to the canonical full-image score within FP epsilon.

Cost model:
- One-time: `PrecomputedReference::new` on source (~half a full-image
  diffmap cost). Reused across all passes.
- Per strip: ~`width * STRIP_INNER` pixel cost. STRIP_INNER = 16 in
  zensim today; aligns with one 4:2:0 iMCU row.

For the iteration loop this means: pass cost ≈ encode + IDCT-back +
per-strip metric, all scaling linearly with image area but with smaller
per-pixel constants than full-image diffmap.

## Proposed upstream zensim API

```rust
// New public type in zensim::streaming.

/// Per-strip streaming consumer that accumulates distorted-side
/// contributions against a precomputed source reference. Produces a
/// canonical full-image DiffmapResult on finalize.
pub struct StreamingDiffmap<'a> {
    /// Borrow of the precomputed source pyramid; never modified.
    reference: &'a PrecomputedReference,
    /// Internal per-scale accumulators, shape mirrors the reference's
    /// pyramid.
    scales: Vec<ScaleAccumulators>,
    /// User-supplied diffmap shaping options.
    options: DiffmapOptions,
    /// Rows pushed so far; finalized when this hits image height.
    rows_pushed: usize,
}

impl<'a> StreamingDiffmap<'a> {
    pub fn new(
        reference: &'a PrecomputedReference,
        options: DiffmapOptions,
    ) -> Self;

    /// Push `rows` rows of distorted pixels starting at `strip_y` in
    /// the source coordinate system. `rows` must be a multiple of
    /// STRIP_INNER (16) except possibly the final strip.
    ///
    /// Returns `Some((score_contribution, ...))` when this push
    /// completes a full strip's accumulators; `None` while waiting
    /// for more rows. The score contribution is the delta this strip
    /// adds to the running full-image score.
    pub fn push_distorted_strip(
        &mut self,
        distorted: &impl ImageSource,
        strip_y: usize,
        rows: usize,
    ) -> Option<StripContribution>;

    /// Linear-planar variant for encoder pipelines that already have
    /// linear-RGB f32 strips on hand (no need to interleave back to
    /// RGB8 just to feed the metric).
    pub fn push_distorted_strip_linear_planar(
        &mut self,
        planes: [&[f32]; 3],
        strip_y: usize,
        rows: usize,
        stride: usize,
    ) -> Option<StripContribution>;

    /// Finalize and return the full DiffmapResult. Must be called
    /// after exactly `image_height` rows have been pushed.
    pub fn finalize(self) -> DiffmapResult;

    /// Instantaneous estimate of the full-image score from
    /// already-finalized strip accumulators. Useful for early-stop
    /// heuristics; final value comes from `finalize`.
    pub fn current_score(&self) -> f32;
}

pub struct StripContribution {
    /// Strip's contribution to the running zensim score (signed delta).
    pub score_delta: f32,
    /// Per-block diffmap for the strip rows that just completed
    /// (block-aligned, length = blocks_w * (rows / 8)).
    pub block_diffmap: Vec<f32>,
}
```

## Multi-scale boundary handling

zensim is multi-scale (4 levels, 2× downsampled each, per the v0.2
profile). Scale 3 = 8× downsampled, so features at row Y depend on
rows Y±32 at full resolution. A pure-window measurement that only sees
[Y, Y+STRIP] has wrong features near the strip edges.

The internal `streaming.rs` already handles this with rolling band
buffers — `STRIP_INNER = 16` rows of "valid" output backed by
lookahead/lookback for higher-scale context. The proposed
`StreamingDiffmap` consumer just exposes that internal state machine
publicly; the boundary math is already correct.

**Test-slice validation strategy** (per #113 user direction):

1. Pick a corpus of test images (CID22 + screen content, ~30 images).
2. For each image, encode at q=80, decode.
3. Compute one-shot `compute_with_ref_and_diffmap` → reference score S₀.
4. Compute strip-by-strip via `StreamingDiffmap::push_distorted_strip`
   → finalized score S_n.
5. Assert |S_n − S₀| < ε for ε = 1e-4 (FP rounding tolerance).
6. Repeat with non-aligned slice subsets (e.g. push the first 7 strips,
   skip 1, push the rest) to catch any "valid only when aligned"
   regressions in the boundary handling.

## zenjpeg-side changes (after upstream lands)

`zq.rs::run_iteration_loop` would become:

```rust
let mut streaming = StreamingDiffmap::new(&pre, DiffmapOptions::default());

// Pass 0: encode, push strips into streaming consumer as they're
// finalized by the strip processor.
for strip in encoder.strips() {
    let decoded_strip = idct_back(strip);
    if let Some(contrib) = streaming.push_distorted_strip(&decoded_strip, ...) {
        // Per-strip controller hook: adjust AQ for next strip based on
        // contrib.block_diffmap.
        controller.observe(strip_y, contrib);
    }
}
let result = streaming.finalize();
```

This collapses encode + measure into a single pass with strip-level
feedback. The per-pass full-image diffmap cost vanishes; what's left is
just the IDCT-back of the just-encoded blocks (which we have in
memory anyway).

## Open questions

1. **Per-strip score deltas vs only-finalize semantics?** If callers
   only ever use `finalize()`, the per-strip return value is overhead.
   Maybe have two modes: `StreamingDiffmap::silent()` and
   `::with_strip_feedback()`.

2. **Memory ownership of accumulators.** The internal
   `ScaleAccumulators` are sized by the source pyramid. For a 4K image
   that's ~50 MB of f32 buffers. Should `StreamingDiffmap::new` take
   `&mut PrecomputedReference` to allow pooling, or is allocate-per-call
   fine?

3. **`current_score()` precision.** Strip-level streaming inherently
   has slight pyramid-fusion-order differences from one-shot. The
   "canonical full-image score" comes from `finalize()`; `current_score`
   is an estimate. Does that estimate need a documented error bar?

## Tracking

- zenjpeg #113 PR-F (this design): zenjpeg/docs/zq-streaming-diffmap-design.md
- Upstream zensim issue: TODO (to be filed after this design lands)
- Existing experimental ctor exposure (zensim/--diffmap-public-ctors
  workspace): supersede / discard once StreamingDiffmap lands —
  internal-ctor exposure isn't needed if the public consumer covers
  the use case.

## Order of operations

1. Open upstream zensim issue with this design (link this doc).
2. Land the internal `StreamingDiffmap` API in zensim.
3. Ship a zensim release with the new public API.
4. zenjpeg PR-F: bump zensim dep, replace `compute_with_ref_and_diffmap`
   with the streaming consumer in `run_iteration_loop`.
5. Validate convergence behavior didn't regress (zq_target tests
   should pass unchanged; per-pass time should drop on large images).
