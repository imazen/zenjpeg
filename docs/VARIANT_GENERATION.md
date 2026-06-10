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
(`ENTROPY_TRIAL_MAX_BYTES = 16 KiB`: every observed crossover sits at
≤ ~2.4 KB; 7× margin). This is self-cost-limiting: small output implies
little entropy content, bounding the trial cost by construction.

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
output-neutral knobs (speed modes) and inputs fully mediated by hashed
state (raw quality; the `allow_16bit` flag when every value fits 8-bit;
boundary-rd when a trellis config makes it inert). Equal fingerprint ⇒
identical bytes for the same input ⇒ one encode serves all aliased
spellings. In zenjpeg's `rd_core × Step5` this merges 46% of the naive
cross product before any encode runs.

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
- Per-anchor/exotic parameter grids (boundary-rd's 66-combo space)
  belong in the calibration harness (coefficient), not the curated axes.
- Follow-ups tracked in imazen/zenjpeg#143.
