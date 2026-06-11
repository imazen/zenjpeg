# Variant Generation: the zenjpeg approach, and what other zen codecs should adopt

Written 2026-06-10, extended 2026-06-11 after the empirical-validation
wave and the zenmetrics fleet/job-system integration (see
`ENCODER_KNOB_SPACE.md` for the rationale history and `CHANGELOG.md` for
the commit trail). zenjpeg is the reference implementation; this
document states the patterns in codec-neutral terms so zenwebp / zenjxl
/ zenavif / zenpng can adopt them deliberately.

The shape of the whole thing: the **codec** owns what variants exist
(typed knobs, validity, resolved-state identity, curated axes); the
**executor** (zenmetrics) owns how a cell becomes bytes and scores; the
**fleet/job layer** owns where and when, via compact deterministic
specs; the **image** ships baked binaries and owns nothing. Every
pattern below exists to keep one of those boundaries honest.

## The patterns

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

**Search modes are supremum contracts — test them as such.** A
`Smallest`/`Search` mode's promise is "never worse than any explicit
mode I claim to subsume", byte-for-byte. That is a testable invariant,
and it breaks quietly when a candidate set drifts: zenjpeg's
`SmallestSearch` lost to the explicit mozjpeg scan script by 0.09% on
one cell because the script search never trialed that canonical shape.
The fix is always "add the missing candidate", but only a harness that
encodes the search mode AND every explicit mode side by side will tell
you a candidate is missing. Also keep search modes honest about their
gates: if the sequential trial is byte-gated, the supremum claim holds
only under the gate — say so in the mode's docs rather than implying
totality.

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
- **Curated steps must be safe on content they were NOT measured on.**
  A step measured as "2% smaller / 3% DSSIM on photos" can be a
  quality catastrophe on high-AQ content: zenjpeg's unclamped ±4
  coupling steps scored SSIM2 −31 on noise while shedding 90% of
  bytes — the historical screenshot-destruction mode, re-shipped as a
  curated value. If a knob has a known protective clamp, the curated
  steps carry it; the unclamped extreme stays constructible explicitly
  but never rides the default axes. Enforce with a test ("no active
  coupling without a clamp in curated axes"-shaped).

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

Harness design gotchas, learned the slow way:

- **No degenerate fixtures.** A solid color has no AC coefficients, so
  every coefficient-stage knob produces identical bytes and your
  distinctness assertions pass vacuously. Smooth gradients are nearly
  as bad (repo-wide ban). Use deterministic noise + real photos +
  aligned-pattern screen proxies + one tiny image; each content class
  catches a different failure mode (noise exposed both the speed_mode
  fingerprint bug and the coupling destruction; a real photo broke the
  byte gate; a tiny image exercises header-dominated regimes).
- **Quality floors are content-relative.** Pure noise legitimately
  scores in the low 20s (SSIM2) at 4:2:0 q85 — a single global floor
  either misses corruption or cries wolf. Set floors per content class;
  their job is to catch the negative-score disasters, not to grade.
- **Output rows arrive in completion order, not queue order** (parallel
  execution). Assert content properties, never row positions.
- **Identity must be parseable back to the deviating axis.** The
  harness needs to attribute each cell's deltas to the knob that moved;
  if two different configs render the same id (zenjpeg's λ₂/delta-DC
  collision), attribution silently merges them. A global id-uniqueness
  test over the largest axes set is cheap and load-bearing.

### 7. Cell ids are durable identity (the ledger contract)

The moment plan cells flow into a fleet, their ids stop being debug
labels: they are stored in TSV/parquet identity columns and hashed into
content-addressed `JobId`s, and an executor must re-encode a cell from
its id alone, years later, with no plan spec in hand. That imposes
rules on the id grammar any codec planner emits:

- **Self-describing**: the id encodes every output-relevant axis value
  (`<fam>_<coeff>_<scan>_<color>[-flag…]` in zenjpeg), and a public
  parser (`config_from_cell_id(base_id, q)`) reconstructs the exact
  config. Anything that genuinely cannot be self-described (opaque
  custom table bytes, content-hashed knob bundles) must **error at the
  parser** so it is rejected at declare time, not discovered at execute
  time.
- **Lossless numbers**: render floats with shortest-roundtrip `Display`,
  never fixed precision — `{:.1}` silently truncates off-grid values,
  which is exactly the wrong property for stored identity. (zenjpeg
  switched `tr14.7500` → `tr14.75` on day one of the first consumer;
  day two would have been too late.)
- **Additive-only evolution**: a new deviation gets a new suffix whose
  absence means "default", so every previously stored id stays valid.
  Renaming a token or changing numeric formatting orphans ledger rows —
  treat it like a wire-format break.
- **Renderer and parser move in lockstep**, enforced by a
  grammar-totality test: every id the planner can emit (canonical AND
  alias spellings) must parse back to a config whose resolved-state
  fingerprint (pattern 4) is identical. zenjpeg's
  `cell_ids_roundtrip_to_their_configs` covers rd_core in full plus
  per-axis main-effects over modes_full's value lists — every token
  spelling without the full cross product.
- **The fingerprint rides along and is verified at execute time**:
  identity is `{"cell": <id>, "fp": <fingerprint>, "plan": <name>}`,
  and the executor recomputes the fingerprint after parsing
  (`resolve_verified`). Any renderer/parser drift between the declaring
  and executing builds becomes a loud deterministic failure naming both
  fingerprints — never a silently wrong encode.

## Where each piece lives

Four layers, one compact deterministic spec flowing down. Getting a
piece on the wrong layer either duplicates the knob vocabulary (logic
too high) or breaks completion semantics (logic too low):

| layer | owns | carries |
|---|---|---|
| codec (`encode::sweep`) | what cells exist: axes, validity, fingerprints, ordering, budget ladder, id grammar + parser | the planner |
| executor (zenmetrics `sweep` / `jobexec`) | how a cell becomes bytes + scores; plan expansion at run time | `--plan name [--plan-budget N]`, or one cell id per job |
| fleet / job system | where & when: sharding, leases, retries, completion | the spec (chunk fields or `DesiredJob`s) — never expanded cells |
| docker image | shipping baked binaries | nothing (bake-everything rule) |

Plan *expansion* belongs in the executor because that is the only layer
where the codec's config type exists — anywhere else re-serializes
configs through a knob vocabulary, which is the duplication the planner
kills. The split is safe because `plan(name, budget, q_grid)` is a pure
deterministic function: a three-field spec expands to byte-identical
cell lists on every box.

**Two execution models, one identity.** zenmetrics executes plans both
ways, and choosing the wrong one re-creates the "100k AVIF encodes
never finish" problem:

1. **Chunk mode** (`zen-metrics sweep --codec zenjpeg --plan …`): the
   executor expands at run time; the unit of retry is (image × whole
   plan). Right for GPU-metric fleet runs that complete in one pass.
2. **Job-system mode** (zen-job-core ledger): cells become per-cell
   content-addressed `DesiredJob`s at **declare time** (`zen-metrics
   sweep --plan … --dry-run --emit-cells` →
   `zen_jobctl::declare_encodes`), and completion is `declare → gap →
   run → re-reconcile` against the Parquet ledger. Declaring is
   idempotent (same plan ⇒ same `JobId`s), so a sweep that dies at any
   point converges across any number of partial passes — chunk
   bookkeeping disappears. Right for big/expensive sweeps (AVIF-class)
   that will not finish in one pass. See zenmetrics
   `docs/RUNNING_JOBS.md` §4b for the executor contract.

Identity is the same in both: `{"cell","fp","plan"}` in the
`knob_tuple_json` column / `JobKind::Encode.knobs`. A codec that ships
the planner shape (pattern 7 included) gets both models for free; the
knob-vocabulary translation layer in
`zen-metrics-cli/src/sweep/encode.rs` remains only for axes the planner
doesn't own. One seam to respect: ledger `CellId.q` is `i64`, so
job-system paths require integer q-grids — reject fractional grids at
emit time, never truncate.

## What each codec should adopt, concretely

The cross-codec contract is `InternalParams` (the per-axis `Option<_>`
partial-merge bundle, zenwebp-shaped) — keep mirroring it. On top of
that, per codec:

| Pattern | zenwebp | zenjxl | zenavif | zenpng |
|---|---|---|---|---|
| Variant-scoped knobs + validate() | segments/partitions config | **landed** (noise×lossless rejection et al., 2026-06-11) | tile/speed knobs | filter/zopfli knobs |
| Dominance cases | always-on optimized entropy | — audit in flight | — audit needed | palette-when-fewer-colors checks |
| Exact trials (pixel-invariant) | lossless: trial entropy backends | modular: trial MA-tree configs at fixed quantization? (audit which stages are pixel-invariant) | OBU/layout-level only (most knobs are metric-class) | **filter-per-row is already exact**; trial zopfli iterations under a byte gate |
| resolve_plan() introspection | yes — port shape directly | **landed** (`__expert`, 2026-06-11) | yes (auto_tune explains itself) | yes |
| Fingerprint dedup | yes — resolved segment params | in flight | yes | yes |
| Sweep planner + validation harness | port `encode::sweep` shape | **harness landed**, evidence-driven axes corrections | port | port |

(zenjxl adoption is an active parallel effort — see its own
`VARIANT_GENERATION` adoption doc in that repo for current state rather
than trusting this snapshot.)

**Consumers — two execution models, one identity.** zenmetrics executes
plans through both of its scheduling models, and the difference matters:

1. **Chunk mode** (`zen-metrics sweep --codec zenjpeg --plan
   rd_core|modes_full [--plan-budget N]`, zenmetrics 2524d81f): the
   executor expands the plan at run time; the unit of retry is
   (image × whole plan). Right for GPU-metric fleet runs that complete
   in one pass.
2. **Job-system mode** (zen-job-core ledger): for sweeps that *never*
   complete in one pass (the 100k-cell AVIF problem), cells become
   per-cell content-addressed `DesiredJob`s at **declare time**
   (`zen-metrics sweep --plan … --dry-run --emit-cells`), completion is
   `declare → gap → run → re-reconcile` against the Parquet ledger, and
   chunk bookkeeping disappears. See zenmetrics
   `docs/RUNNING_JOBS.md` §"Plan-driven sweeps".

Both carry the same identity: `{"cell": <stratum-id>, "fp":
<fingerprint>, "plan": <name>}` in the `knob_tuple_json` column /
`JobKind::Encode.knobs`. **That makes the cell-id grammar a durable
contract**: `encode::sweep::config_from_cell_id(base_id, q)`
reconstructs the exact `EncoderConfig` from the id alone (numbers are
shortest-roundtrip `Display` — lossless), so a ledger job is
self-describing and regenerable years later, and the carried `fp` is
verified after parsing so grammar drift fails loudly instead of
encoding the wrong cell. Grammar evolution is additive-only — never
rename a token or change numeric formatting; the
`cell_ids_roundtrip_to_their_configs` test enforces parser totality
over everything the planner emits. (`custom` table bytes and
content-hashed boundary-RD knobs are the two documented
non-self-describing cases.)

A codec that adopts the planner shape gets both execution models for
free; the knob-vocabulary translation layer in
`zen-metrics-cli/src/sweep/encode.rs` is only needed for axes the
planner doesn't own.

### Adoption checklist (the order that paid off)

Each step gates the next; the test named with it is the exit criterion.

1. **Discriminate knobs** — variant-scoped config enums, `validate()`
   rejections for meaningless combos, fix-reality for inert/mislabeled
   knobs. *Gate: invalid combos error; no knob is silently dead.*
2. **`resolve_plan()` introspection** — resolution through the same
   function the encoder runs. *Gate: plan output matches an actual
   encode's choices on spot checks.*
3. **Resolved-state fingerprint** — hash resolved state; prove every
   exclusion BY ENCODE on adversarial content. *Gate: alias pairs
   byte-identical; a distinct-config control differs; search-effort
   knobs hashed unless proven neutral.*
4. **Sweep planner** — curated axes with provenance + protective
   clamps, dedup, validity reporting, main-effects-first ordering,
   one-value-at-a-time budget ladder. *Gate: planner unit tests +
   id-uniqueness over the largest axes.*
5. **Id grammar + parser** (pattern 7) — self-describing ids, lossless
   numbers, `config_from_cell_id` + fp verification. *Gate: the
   grammar-totality roundtrip test.*
6. **Empirical validation harness** (pattern 6) — run it; expect it to
   find real defects (zenjpeg: five on the first run; zenjxl: axes
   corrections on its first run). Fix, re-run, commit the TSV. *Gate:
   ALL HARD CHECKS PASSED.*
7. **Exact trials** — last, because steps 3–6 teach you which knobs are
   dominance/trial/metric class. Byte-domain gates with provenance;
   supremum contracts tested. *Gate: exact-min equality tests + the
   harness's trial-contract checks.*
8. **Executor + fleet wiring** — `--plan` in zenmetrics chunk mode,
   `--dry-run --emit-cells` → `declare_encodes` for the job system,
   jobexec resolution via your parser. Both models or document why not
   (zenmetrics CLAUDE.md guard). *Gate: the e2e test — declare item →
   jobexec stdin → valid bytes; tampered fp → loud failure.*

Steps 1–7 live in the codec repo; step 8 is one PR in zenmetrics once
patterns 4/5/7 exist, because the executor only needs `plan()`,
`fingerprint()`, and `config_from_cell_id()`.

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
