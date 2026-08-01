# Streaming measurement for target-zq (zenjpeg #113 PR-F) — design v2

Revised 2026-07-31. **This version supersedes the v1 draft in place** after
a source-verified coherence review of the proposal against zensim at the
pinned rev (`9d8f73a5`, what zenjpeg builds today) and zensim `main`
(`f316807e`, mid feature/scoring redesign). The v1 draft's API shape and
several of its load-bearing facts did not survive contact with the source;
the corrected facts, the revised upstream proposal, and the implementation
ordering are below.

Status: design + upstream coordination. zensim-side implementation is
**deliberately deferred to the post-freeze surface** (see "Timing");
zenjpeg-side wiring follows the upstream landing.

## Verdict (from the 2026-07-31 coherence review)

Per-strip streaming measurement is **mathematically coherent under
zensim's MLP/linear-projection scoring — if and only if accumulation
happens in feature space and the scoring head runs once at finalize.**

- Every spatial pooling op in the shipped runtime is an f64 sum, a
  power-sum (the "p95"-named fields are L8 power means, not
  percentiles), a weighted-sum-normalized-by-count, or a running max —
  all exactly strip-accumulable. There are **no true percentiles, no
  histograms, and no global normalizations** in the pooling path
  (`9d8f73a5:zensim/src/streaming.rs:297-343`, `metric.rs:630-635`).
- All nonlinearities (p-roots, HF ratios, the MLP head, the α/hybrid
  gate, the PCHIP output spline) apply to the ~372 globally pooled
  scalars once, at finalize (`9d8f73a5:zensim/src/metric.rs:2026-2162`).
- zensim **already ships** strip-based accumulation proven equivalent to
  the one-shot path to <1e-13 rel — `compute_with_ref_streaming_strips`
  at strip geometry 256 inner / 128 margin
  (`9d8f73a5:zensim/src/metric.rs:1386-1440`, `streaming.rs:2635-2660`).
- **Per-strip `score_delta` is mathematically ill-defined** under the
  nonlinear head: the telescoped marginal is order-dependent, confounds
  the strip's own error with coverage renormalization of *all* prior
  pooled features, and partial pools feed the MLP out-of-distribution
  inputs. It must not be in the API. The per-strip **block diffmap is
  clean** — purely local (±40 full-res rows of context under default
  options), no coverage normalization, no head — and it is the only
  per-strip signal the zq controller actually consumes.

## Corrections to the v1 draft (verified against source)

| v1 claim | Reality |
|---|---|
| `STRIP_INNER = 16`, an input cadence | `STRIP_INNER = 32`, an internal band-blocking constant inside a one-shot pass (`9d8f73a5:zensim/src/streaming.rs:33-54`) |
| Multi-scale context ±32 rows | ±40 full-res rows for the default (SSIM-only) diffmap; **±80** for the score side (masked+IW features need a second blur at scale 3) |
| "Rolling band buffers already expose the state machine" | No rolling push machinery exists at the pinned rev. Strips are independently recomputed with 128-row margins (~2× row-work vs inner rows; at v1's 16-row inner it would be ~17×). Zero-overlap streaming exists only in the v2 feature-regime producer on zensim main |
| `PrecomputedReference::new` ~half a diffmap cost | ~25% (4K) to ~34% (8K) of one compare — and zq.rs already builds it once per encode (`zq.rs:668`), so no new saving there |
| `ScaleAccumulators` ~50 MB at 4K | ~1 KB (33 small arrays). The real memory is the distorted-side plane window plus the fused full-res diffmap (~33 MB f32 at 4K) — which should not be materialized at all when only block means are consumed |
| Emit per-strip `score_delta` | Ill-defined under the MLP head; deleted from the proposal |
| Strip path can emit diffmaps | It cannot: the strip aggregator passes no diffmap weights internally; per-band diffmap plumbing exists but is unwired |

Two additional facts that shape the plan:

- **Exactness geometry**: strip inner boundaries must be multiples of 8
  full-res rows (pyramid parity); documented minimum inner height is
  176 rows with ≥128-row margins for exact-vs-one-shot accumulation.
  Byte-exactness is also **per-process**: band layout depends on the
  rayon thread count, so parity harnesses must pin threading.
- **The final rows**: the last ~min(margin, remaining) rows can only
  finalize after the last push — a push consumer must either buffer
  with lag or emit provisional values and correct them.

## Prior art: zensim#16 (closed 2026-04-29) — read before touching this

The v1 draft's upstream ask was already filed as
[imazen/zensim#16](https://github.com/imazen/zensim/issues/16) and
**closed with a reasoned negative result** after synthetic + real-codec
validation of three mechanisms:

| | Option A (buffered streaming) | Option B (scale-0 proxy) | Option D (slice + ZensimScratch) |
|---|---|---|---|
| Per-window signal mid-stream | NaN until finalize | yes (~6% weight mass) | yes (truncated context) |
| Real-codec spatial top-1 | n/a | 33% | 24% |
| Mean SROCC vs ground-truth diffmap | n/a | 0.394 | 0.571 |

Verdict there: *"the multi-scale metric's cross-window psychovisual
masking is fundamental and isn't recoverable from any per-window
computation"* — per-window **canonical score** as a targeting oracle is
dead, and stays dead. Option A's 750-LOC `StreamingDiffmap` (strip-fed,
canonical-equivalent finalize, <1e-4 parity) is parked at
`explored/issue-16-option-a-buffered-streaming` as reference.

**What changed since:** the attribution surface does not compute
per-window scores at all — it spatially decomposes the *global* score's
first-order sensitivity (∂score/∂region of the full-image pooled
features), which sidesteps the cross-window-context problem entirely.
Upstream measured it at M2 0.999–1.000 across block sizes 16–128 px
(the 128 px inversion that plagued the fold is cured), and #69
validated H3 magnitude steering with it in a real closed loop. That —
not a per-window score — is the per-strip signal the revised proposal
carries, so the new issue supersedes #16 rather than reopening it.

## The redesign context (zensim main, freeze plan 2026-07-31)

The metric redesign moves the target surface out from under the v1
proposal:

- `ZensimProfile::codec_target()` is now **Profile B** (deterministic
  linear ensemble); Profile A is deprecated. The frozen dial will be an
  **MLP bake** — per-strip score additivity is permanently unavailable.
- The feature regime moves 372 → **944** (feature_v2), extracted
  **streaming-only** in 128-row kernel strips with 10-row halos, with
  **bit-exact fixed accumulation order** (a push consumer must add
  strips in row order, not merge reassociated partials).
- **The diffmap fold is demoted to visualization-only.** The codec
  steering surface is the **attribution density + summed-area table**
  (`compute_with_ref_score_and_attribution`): signed per-pixel
  first-order ∂score/∂region with O(1) rect queries, valid for
  MLP-class dials (piecewise-linear ⇒ locally exact gradients). The
  validated closed-loop rule is **H3 magnitude steering** (per-tile
  steps ∝ attribution magnitude, capped) — upstream's #69 result found
  a coherent *map* alone did NOT improve target-hitting; the magnitude
  rule did.
- The freeze plan's own Phase-3 item requires a 944-regime **fused
  compare with extractor-side retention hooks** — i.e. upstream must
  build a per-strip-signal surface anyway. The push consumer should BE
  that mechanism, not a parallel one.

## Revised upstream proposal: `StreamingCompare`

Filed as a zensim issue (see Tracking). Regime- and head-agnostic:

```rust
/// Push-based streaming compare. Accumulates in feature space; the
/// profile head runs once at finalize.
pub struct StreamingCompare<'a> {
    reference: &'a PrecomputedReference, // or the v2 ref equivalent
    // internal: rolling distorted-side window (O(width × context) rows),
    // per-scale accumulators, emission cursor lagging the push cursor
    // by the profile's documented context radius.
}

pub struct StreamOptions {
    /// Per-strip signal shape. `Attribution` block sums are the
    /// post-freeze default; the v1 fold ships only as visualization.
    pub strip_signal: StripSignal, // None | BlockDiffmap {..} | Attribution {..}
    pub emit_estimated_score: bool, // off by default
}

impl<'a> StreamingCompare<'a> {
    pub fn new(r: &'a PrecomputedReference, o: StreamOptions) -> Result<Self, ZensimError>;

    /// Rows arrive top-to-bottom, contiguous, any multiple of 8 (16-row
    /// iMCU strips qualify). Feedback covers rows that became FINAL —
    /// which lags the push cursor by the context radius.
    pub fn push_distorted_linear_planar(
        &mut self, planes: [&[f32]; 3], rows: usize, stride: usize,
    ) -> Result<StripFeedback, ZensimError>;

    /// Canonical result — identical (≤ documented ε, thread count
    /// pinned) to the one-shot fused compare for the same profile.
    pub fn finalize(self) -> Result<(ZensimResult, AttributionResult), ZensimError>;
}

#[non_exhaustive]
pub struct StripFeedback {
    /// Newly-finalized full-res rows (empty while lookahead fills).
    pub emitted_rows: core::ops::Range<usize>,
    /// Local per-8×8-block signal for emitted rows. For `Attribution`,
    /// block sums of the signed density (score units; H3-ready).
    pub block_signal: alloc::vec::Vec<f32>,
    /// Score of the covered region so far — NOT the final score, NOT a
    /// delta, NOT additive. NaN until coverage ≥ threshold.
    pub estimated_score_so_far: f32,
    pub coverage: f32,
}
```

Deliberate choices: **no `score_delta`**; feedback keyed to *emitted*
(context-complete) rows; `finalize` returns the fused-compare types so
the streaming path is a drop-in for
`compute_with_ref_score_and_attribution`; the head is versioned
implicitly via `ZensimProfile`.

Interim option (pinned rev, if ever needed before the freeze lands): a
rolling-window `ImageSource` adapter driving
`compute_with_ref_streaming_strips` with multithreading off — correct
and bounded-memory for the *score* side only, at ~2× margin-recompute
overhead; the block map still requires a finalize-time fused map. Any
per-strip *diffmap emission* work at the pinned rev is effort spent on
the surface being retired — don't.

## zenjpeg-side plan

1. **Now (this doc + upstream issue):** coordination only. The zq loop
   keeps its one-shot `compute_with_ref_and_diffmap` measurement; it is
   correct, and its reference pyramid is already amortized across
   passes.
2. **When `StreamingCompare` lands upstream:** replace `measure()` in
   `encode/zq.rs` with the push consumer fed by strip-wise decode of the
   candidate (decoder scanline reader), and wire
   `AqController::observe` (`encode/aq_controller.rs` Layer 3) with
   per-strip `block_signal` feedback. The controller's tighten/loosen
   thresholds are currently full-image block-map percentiles
   (`zq.rs:461-491`) — with per-strip feedback these must come from the
   previous pass or a running estimate: that is a controller redesign,
   and it should adopt **H3 magnitude semantics** (validated upstream
   for MLP-class dials) rather than reusing the percentile rule
   unmeasured.
3. **Known gap to fix in the same PR:** the zq per-block scale grid is
   `width/8 × height/8` truncating (`zq.rs:608`, `546-569`), so partial
   edge blocks are never measured or corrected — and edge MCUs are
   exactly where partial-MCU artifacts live. The recompress refinement
   (2026-07-31) chose the conservative policy (unmeasured ⇒ untouched);
   the zq rewrite should measure partial blocks (partial-region means)
   instead.

## Validation strategy (kept from v1, corrected)

1. Corpus (CID22 + screen content, ~30 images), encode q≈80, decode.
2. One-shot `compute_with_ref_and_diffmap` → S₀ (and fused compare on
   the post-freeze surface).
3. Strip-by-strip via the push consumer → finalized S_n.
4. Assert |S_n − S₀| < 1e-4 **with the rayon thread count pinned**.
5. Non-aligned pushes (7 strips, skip, rest) — this is exactly the test
   that catches boundary-margin bugs; keep it.
6. Run at both the 372/A|B regime and (post-freeze) the 944 regime.

## Timing

Do **not** land `StreamingDiffmap`-as-v1 upstream now: it targets a
signal shape (fold diffmap + additive deltas over 372/A) that the
freeze plan retires, and zensim is mid-endgame with concurrent sessions
landing freeze-gate work. The upstream issue proposes the
`StreamingCompare` skeleton as the push-based delivery of the freeze
plan's own fused-compare retention hooks, so one mechanism serves both.

## Tracking

- zenjpeg #113 PR-F (this doc, v2).
- Upstream: [imazen/zensim#54](https://github.com/imazen/zensim/issues/54)
  "StreamingCompare: push-based streaming compare for encoder closed
  loops" (filed 2026-07-31 from this design; supersedes zensim#16).
- Prior art: [imazen/zensim#16](https://github.com/imazen/zensim/issues/16)
  (closed 2026-04-29) — three explored options with real-codec
  validation; Option A's parked branch
  `explored/issue-16-option-a-buffered-streaming` is a reusable
  substrate for the score-accumulation half.
- Full coherence review (pipeline verification with file:line evidence
  at both revs, op-by-op decomposability table, cost-model corrections):
  produced 2026-07-31; key findings are folded into this doc.
