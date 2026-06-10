# Variant Generation: the zenjpeg approach, and what other zen codecs should adopt

Written 2026-06-10, after the knob-discrimination / exact-entropy /
sweep-planner wave (see `ENCODER_KNOB_SPACE.md` for the rationale history
and `CHANGELOG.md` for the commit trail). zenjpeg is the reference
implementation; this document states the patterns in codec-neutral terms
so zenwebp / zenjxl / zenavif / zenpng can adopt them deliberately.

## The five patterns

### 1. Knobs live on the variant that uses them (discrimination)

A config knob that is dead under some mode is a bug factory: sweeps burn
encodes on inert axes, users get silent no-ops, and pickers train on
noise. Make the type system carry liveness:

```rust
// zenjpeg: the chroma knobs exist only where they act
enum QuantTableConfig {
    Jpegli             { chroma_distance_scales: [f32; 2] },
    MozjpegRobidoux    { chroma_quality: Option<u8> },
    PiecewiseV4,                       // quality drives anchors; no knob
    ...
}
```

Combinations with no defined meaning are **rejected in `validate()`**
(e.g. XYB × YCbCr-trained tables), not silently remapped. Where reality
itself is confused (an inert parameter, a mislabeled channel mapping, a
duplicate quality knob), fix reality rather than modeling the confusion.

### 2. The 100%-knob taxonomy: dominance / trial / metric

Classify every decision by what it changes:

- **Dominance** — one side provably or structurally always wins.
  *Take it directly; zero trials.* (zenjpeg: optimized Huffman over
  fixed; 8-bit DQT when values fit; grayscale tiny-tables — pure header
  pruning with no shared-table cost.) Misclassifying a dominance case as
  a trial wastes CPU; the reverse silently loses bytes.
- **Trial** — the winner is content-dependent but **decoded pixels are
  identical** across candidates. Then `min(bytes)` is exact, 100% of
  the time; no threshold heuristic can beat it. Compute the expensive
  shared state once, serialize candidates, ship the smallest.
  (zenjpeg: `ProgressiveScanMode::Smallest{,Search}` over
  sequential/tiny/progressive; `TinyFileMode::Auto`'s shared-table
  trial; lossless recompress over `OutputMode`s.)
- **Metric** — the decision changes pixels (tables, λ, subsampling,
  color space). Exactness requires *measuring* a perceptual metric per
  candidate (the closed-loop / `Quality::Zq` direction) or accepting a
  trained oracle. This is the irreducible heuristic core; it is what
  sweeps and pickers are FOR. Never pretend a metric knob is a trial
  knob.

**Gate trials in the byte domain, self-measured.** The crossover for
entropy-stage alternatives is governed by output bytes, not pixel
dimensions (degenerate content breaks any pixel proxy). Serialize the
expected winner first; run the other candidates only when its output
lands at or below a provenance-documented byte gate
(`ENTROPY_TRIAL_MAX_BYTES`, currently 32 KiB — raised from 16 KiB when
the validation harness found a 19.8 KB real-photo crossover; see Known
limits). This is self-cost-limiting: small output implies little
entropy content, bounding the trial cost by construction.

### 3. Resolution is a function, introspection calls the same function

`EncoderConfig::resolve_plan(w, h) -> EncodePlan` resolves *every* knob —
including the actual quantization tables — through the **same code** the
encoder runs at construction (`resolve_quant_tables`). The plan cannot
drift from reality because there is no second implementation. Static
plans report only what is statically knowable: content-dependent trial
outcomes are documented as such (`tiny_file_active` reports the
structural `Force` case only), never guessed.

### 4. Byte-identity fingerprints over RESOLVED state

A sweep cell's identity is its resolved state, not its config spelling:
hash the actual tables + zero-bias + entropy-relevant knobs; **exclude**
only inputs fully mediated by hashed state (raw quality; the
`allow_16bit` flag when every value fits 8-bit; boundary-rd when a
trellis config makes it inert). Equal fingerprint ⇒ identical bytes for
the same input ⇒ one encode serves all aliased spellings. In zenjpeg's
`rd_core × Step5` this merges 46% of the naive cross product before any
encode runs.

**Every exclusion must be proven by encode, not by reading code.**
zenjpeg's first fingerprint excluded `TrellisSpeedMode` as
"output-neutral by construction" — it reads like a pure speed knob, but
it bounds the coefficient search on high-entropy blocks, and the
empirical harness (pattern 6) falsified the exclusion in one run: 582
bytes of divergence on 512² noise at q95. Search-effort knobs (speed
levels, candidate counts, lookback limits, zopfli iterations) are
usually output-AFFECTING; treat "neutral" as a claim requiring an
encode-level test that an adversarial input (noise, q95) gets to vote
on.

### 5. Budgeted, ordered, no-silent-caps sweep plans

`SweepAxes` (concrete values per axis, **most-important value first**) ×
`QualityGrid` (step-5 floor; low-q never thinned preferentially) →
deduplicated cells, with:

- **Validity filtering** reported, never silently lost.
- **A budget ladder** that sheds one lowest-priority value at a time
  (lowest-tier axis first, floors protect core values), records every
  drop, coarsens the q-grid uniformly only after axes are at floor, and
  sets `over_budget` instead of sampling silently.
- **Main-effects-first queue ordering**: the all-defaults stratum, then
  every single-deviation stratum (one axis changed = "does this knob
  matter"), then interaction combos; milder deviations before extreme
  ones; quality ascending *within* a stratum so an interrupted run never
  strands a half-measured RD curve. Truncation is safe at stratum
  boundaries; `SweepCell::deviations` exposes the priority class.
- **Scalar steps carry provenance** (module docs table: bound, steps,
  and the measurement each step came from). A bound without provenance
  is a guess; steps without bounds are a dart board.

### 6. Validate the axes empirically before trusting them

Curated steps, fingerprint exclusions, and trial gates are all *claims
about encoder behavior*, and claims drift. Ship a harness that encodes
the **default stratum plus every single-deviation stratum** on a small
mixed corpus (a few real photos + adversarial synthetics: noise,
aligned checkerboard, one tiny image) and hard-fails on:

- an **inert step** (a curated value that never changes output bytes),
- a **fingerprint-contract violation** (equal fingerprint, different
  bytes — checked on real encodes of the alias pairs),
- an **exact-trial contract violation** (a `Smallest`-style mode losing
  to a candidate it claims to subsume),
- **ordering breakage** (defaults-first, deviations non-decreasing),

with soft direction checks (sign/monotonicity per the provenance table)
and per-label Δsize/Δquality aggregates as the report. zenjpeg's
implementation is `examples/sweep_validate.rs` — ~200 cells × 7 images,
32 seconds — and its first run caught five real defects: colliding cell
ids across λ₂/delta-DC probe spellings, the `speed_mode` fingerprint
exclusion, unclamped coupling steps reproducing a known
quality-destruction mode (SSIM2 −31 on noise), a byte-gate
counterexample 24% above the gate on a real CID22 photo, and
`SmallestSearch` losing to the canonical mozjpeg scan script it didn't
trial. Every one of those was invisible to unit tests that only check
plan *structure*. Re-run the harness whenever the axes, the
fingerprint, or a trial gate changes; commit the TSV next to the run
date.

## What each codec should adopt, concretely

The cross-codec contract is `InternalParams` (the per-axis `Option<_>`
partial-merge bundle, zenwebp-shaped) — keep mirroring it. On top of
that, per codec:

| Pattern | zenwebp | zenjxl | zenavif | zenpng |
|---|---|---|---|---|
| Variant-scoped knobs + validate() | segments/partitions config | effort×modular/VarDCT knobs | tile/speed knobs | filter/zopfli knobs |
| Dominance cases | always-on optimized entropy | — audit needed | — audit needed | palette-when-fewer-colors checks |
| Exact trials (pixel-invariant) | lossless: trial entropy backends | modular: trial MA-tree configs at fixed quantization? (audit which stages are pixel-invariant) | OBU/layout-level only (most knobs are metric-class) | **filter-per-row is already exact**; trial zopfli iterations under a byte gate |
| resolve_plan() introspection | yes — port shape directly | yes | yes (auto_tune explains itself) | yes |
| Fingerprint dedup | yes — resolved segment params | yes | yes | yes |
| Sweep planner | port `encode::sweep` shape | port | port | port |

**Consumers:** zenmetrics' sweep driver executes these plans directly —
`zen-metrics sweep --codec zenjpeg --plan rd_core|modes_full
[--plan-budget N]` (zenmetrics commit 2524d81f) asks
`zenjpeg::encode::sweep` for its cells instead of spelling a JSON knob
grid, carries the cell id + resolved-state fingerprint in the
`knob_tuple_json` identity column, and writes the plan's
no-silent-caps manifest to `<output>.plan.json`. A codec that adopts
the planner shape gets fleet execution for free; the knob-vocabulary
translation layer in `zen-metrics-cli/src/sweep/encode.rs` is only
needed for axes the planner doesn't own.

Adoption order that paid off here: **discriminate knobs → add
resolve_plan → fingerprints → sweep planner → exact trials**. The trials
come last because the fingerprint work forces you to learn which knobs
are output-neutral, dominated, or mediated — exactly the classification
the trials need.

## Known limits / open items

- Absolute-valued knobs interact badly with quality grids
  (`MozjpegRobidoux::chroma_quality` is absolute while the grid moves q;
  a relative form is the fix — do not sweep a static absolute value).
- Trial candidates must emit **exactly** what their explicit mode would
  (zenjpeg's first Smallest draft beat explicit Baseline by a 6-byte DRI
  it had silently dropped — the equality contract caught it).
- Byte gates are empirical bounds, not theorems. The 16 KiB entropy-trial
  gate shipped with "7× margin over every observed crossover"; the
  validation harness found a real-photo crossover at 19.8 KB (CID22
  1044329, q10 — baseline 2.0% smaller than progressive). Now 32 KiB,
  with the counterexample recorded at the constant. Expect to revise
  again; what matters is that the gate self-measures and the regret
  above it stays small and shrinking with size.
- "Strictly additive" marker reasoning has a tail: restart markers also
  re-base DC prediction, which on rare content nets out *cheaper* than
  the marker bytes (8 bytes / 0.04% observed once in 42 cells).
  `Smallest` deliberately does not sweep restart-interval space.
- Per-anchor/exotic parameter grids (boundary-rd's 66-combo space)
  belong in the calibration harness (coefficient), not the curated axes.
- Follow-ups tracked in imazen/zenjpeg#143.
