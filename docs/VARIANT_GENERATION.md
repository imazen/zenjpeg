# Variant Generation: the zenjpeg approach, and what other zen codecs should adopt

Written 2026-06-10, extended 2026-06-11 after the empirical-validation
wave, the zenmetrics fleet/job-system integration, and the zenavif
(wrapped-engine) adoption (see
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

## The wrapped-engine patterns (zenavif adoption, 2026-06-10/11)

zenavif was the second full adoption and the first **wrapper codec**:
it drives an engine crate (zenravif → zenrav1e) instead of owning the
encoder, and most of what follows exists because of that boundary.
Every rule below was earned by a failing check or a byte-level proof
(see zenavif's `docs/VARIANT_GENERATION.md` for the audit and run
evidence). They are stated codec-neutrally: zenjxl (jxl-encoder),
zentiff (image-tiff), and any future wrapper hits the same walls.

### 8. Wrapped engines: the resolution mirror is pinned by encode

Pattern 3's ideal — introspection calls the same function the encoder
runs — cannot hold across a crate boundary when the engine keeps its
resolution `pub(crate)` (zenravif's quality→quantizer curve and
per-speed search tables). The fallback is a **mirror**, which is a
second implementation and therefore a drift risk. Make it safe:

- every mirrored constant/curve carries a provenance comment citing
  the engine version and source line;
- every mirror is pinned **by encode** with an alias pair plus a
  near-miss negative control: two inputs the mirror says share a
  mediator must be byte-identical, the adjacent input that maps to a
  different mediator must differ (zenavif pins the quantizer curve
  with q 80.0 ≡ q 80.2 ≠ q 81.0, and the speed tables with
  override==preset ≡ unset ≠ override≠preset);
- the harness re-runs on **every engine dep bump** — that is when
  mirrors die;
- the structural fix is queued, not forgotten: export resolution from
  the engine and delete the mirror (tracked for zenravif 0.1.4).

**Wrapper defaults that claim to follow another knob are claims about
the engine — verify them.** zenavif's `alpha_quality` docs promised
"unset follows the color quality"; the engine's default actually
pinned the alpha quantizer to the quality-80 equivalent, so every
alpha-bearing encode at quality≠80 silently disagreed with the docs.
Forward such defaults explicitly (`unwrap_or(quality)`) and pin the
contract with a three-way encode test: unset ≡ explicit(q) ≠
explicit(engine-default).

### 9. Pin ambient-machine defaults in every cell

Any knob whose default resolves from **ambient machine state** makes
encoded bytes machine-dependent. zenavif's instance: the AV1 tile
count is `min(threads, w·h / min_tile_size²)`, and unset `threads`
substitutes the *host's* core count — default-config encodes are not
byte-reproducible across machines, which silently poisons
content-addressed ledgers and cross-box dedup. Rules:

- the planner pins the knob in every cell (`threads(Some(1))`);
  parallelize across cells, not within them;
- `resolve_plan` reports the machine dependence honestly
  (`TilesResolution::MachineDependent { cap }`) instead of guessing;
- the fingerprint hashes the raw setting and never merges unset
  spellings — not even with another unset spelling, since two hosts
  disagree about what it means.

Audit for siblings: thread counts, detected SIMD level, available
memory, anything `num_cpus`-shaped feeding the bitstream.

### 10. Three ways a knob lies about being alive

Discrimination (pattern 1), continued — all three found by the
zenavif harness or audit, all three invisible to code reading:

- **The neutral-value no-op spelling.** "Enabled at the neutral
  value" can be structurally identical to off: zenavif's VAQ at
  strength 1.0 is byte-identical to VAQ disabled (the still/psy tunes
  always compute the activity mask; the engine skips the rescale at
  1.0). Caught as an inert step in one harness run. Fix the *type*:
  the sweep axis is `Option<scale>` so the no-op cannot be curated,
  and the fingerprint hashes the **active** form
  (`enabled && scale ≠ neutral`).
- **The envelope-dead knob.** A knob can be live in the engine yet
  byte-inert across the product's whole usage envelope: `lru_on_skip`
  only changes decisions when entire restoration units are skip
  blocks, which intra-only still images at the curated speeds never
  produce (28/28 inert comparisons, including content purpose-built
  to maximize skips). De-curate with the evidence recorded in the
  provenance table; keep the probe constructible for
  envelope-expanding sweeps (speed ≤ 1 there).
- **The class-conditional knob.** Alpha knobs are byte-inert without
  an alpha plane; sweeping them on an RGB corpus trips the inert-step
  check *correctly*. Give them a per-content-class preset
  (`modes_full_alpha`) and a per-class harness leg with a two-sided
  check: every class probe must change bytes **on** its class AND
  leave off-class output byte-identical (no coupling into the other
  path). Same shape for animation-only, HDR-only, ICC-conditional
  knobs.

Three more lies from the zenjxl adoption (2026-06-11):

- **The build-feature-dead knob.** A knob can act only inside code the
  cargo-feature set compiles out: jxl-encoder's `lossy_search_seeds`
  acts inside the butteraugli quality loop, which the default
  `__expert` build does not include — curating it is a guaranteed
  inert step. Liveness discrimination extends to the build config:
  curated axes are a function of the feature set, and the provenance
  table says which features a probe needs.
- **The quantization-flattened knob.** A continuous knob can be
  consumed through a quantizer that collapses whole ranges:
  jxl-encoder's tree-sample fraction becomes a pixel stride via
  `ceil(1/f)`, so every override in (0.5, 1.0) is byte-identical to
  0.5 — the e7→e9 schedule ramp (0.5→0.65) was a no-op, and so was
  the probe. Curated steps must land on distinct *effective* values;
  the provenance table records the quantization function, not just
  the bound.
- **The deliberately-dropped knob.** A knob can be plumbed
  config → options → call site and then discarded at the sink for an
  architectural reason documented only in a code comment
  (jxl-encoder's lossless multi-group writer drops lz77/lz77_method:
  the global ANS code would mismatch per-group sections' histograms).
  The effort schedule above it is aspirational; the setter docs must
  say so, and the axis stays out until the architecture supports it
  (jxl-encoder#69).

Also in this family: **backend selection is a knob.** Selecting a
backend the build doesn't contain must fail `validate()` — zenavif's
svtav1 request silently fell back to zenravif, i.e. the config asked
for one encoder and was served by another. And when a backend dies,
its orphaned fields die with it: `matrix_coefficients`' only reader
was the removed svtav1 path, so the field is documented as
informational and fingerprint-excluded (byte-proven) rather than left
implying liveness.

### 11. The single-deviation probe axis

Engines expose many deep binary overrides (zenavif: CDEF, RDO-TX,
SGR, segmentation, bottom-up, LRF, fast-deblock, …). One axis per
override explodes the cross product with interaction combos nobody
asked for. Put them all on **one shared probe axis**: a probe never
combines with another probe, only with the primary axes, so the plan
stays main-effects-shaped by construction. Probe each override **both
ways** — the spelling that equals the preset's derived value
fingerprint-dedupes away (pattern 4 hashes post-override resolved
state), leaving exactly the informative direction per (speed, q)
region with zero curation effort.

### 12. Sweeps feed optimizers: emit features from resolved state

The consumer of all this is a trainer (zentrain) fitting pickers /
MLP optimizers, and it should never parse cell-id strings. Ship the
training bridge with the planner: `feature_columns()` (stable,
append-only names) + `SweepCell::feature_row(input) -> Vec<f64>` —
one numeric column per knob, booleans 0/1, enums as small documented
integers, −1 sentinels for not-applicable. The load-bearing choice:
columns carry **resolved mediators**, not config spellings — the
quantizer rather than raw quality, the post-override search settings
rather than `Option` overrides — so the model generalizes across the
aliases the fingerprint merges instead of learning that q 80.0 and
q 80.2 are different inputs. Training-row identity is
`(image_id, cell_id, fingerprint, features…, bytes, metrics…)`.

### 13. Two validation tiers, sized to the engine

- **Tier 1 — encode-level contract tests** behind the plain `encode`
  feature, in the normal test suite, running in CI on every push: the
  mirror alias pairs, the three-way follows-X pin, knob-liveness spot
  checks (4:2:0 must shrink chroma-textured content). No corpus, no
  expert features, seconds.
- **Tier 2 — the corpus harness** (pattern 6) behind the expert
  feature, re-run on axis/fingerprint/engine-bump changes, TSV
  committed with the date.

For expensive engines (AV1-class encodes cost 100–1000× a JPEG), size
tier 2 down instead of skipping it: small crops (256²) of real
photos, an explicit q subset spanning low-q ({10, 30, 60, 85}), the
default + single-deviation strata only. Minutes, not hours — full
grids belong to the fleet. One ops gotcha that will eat an afternoon:
deep-recursion engines (AV1 partition RDO) overflow rayon's default
2 MB worker stacks; any harness or executor running encodes inside a
rayon pool needs `stack_size(32 MB)`.

## The format-encoder patterns (zenjxl adoption, 2026-06-10/11)

zenjxl wraps an in-org *format encoder* (jxl-encoder), where the
output is a bitstream other implementations must decode. That adds
failure modes a metric-scored sweep can't see, and two of them shipped
as real encoder bugs the harness caught on its first day
(jxl-encoder#68, both fixed same-day upstream).

### 14. The harness must decode what it encodes

Bytes + metric scores are not enough: **hash-locks pin bytes, not
decodability**, so an encoder suite can stay green while emitting
streams no decoder accepts. The harness decode-verifies every cell —
and for lossless cells the decoded pixels must equal the input
**exactly** (the zero-tolerance rule as a hard gate). That gate found
both #68 causes: e9+ lossless streams that zenjxl-decoder, jxl-oxide
AND djxl all rejected.

This pattern audited its own reference implementation: zenjpeg's
harness originally decode-verified only as a side effect of scoring,
hard-gating failures at q85 alone — a decode failure at any other
quality dissolved into a NaN that the aggregates skipped. It now
hard-fails any undecodable cell at any quality (2026-06-11), and its
corpus gained a 509×381 crop because every prior image was
MCU-aligned — JPEG's partial-MCU edge paths were structurally
invisible, pattern 15's exact blind spot. Re-audit your harness
against patterns 14–15 even if it predates them.

Two sub-rules earned scars:

- **Internal consistency is not correctness.** Both bugs had the
  encoder's gather, apply, and section-writer all agreeing with each
  other and all diverging from the spec (ad-hoc `group_id` stream
  numbering vs the decoder's ModularStreamId; a property record
  truncated identically everywhere the encoder looked). No amount of
  encoder-side cross-checking detects that class — only independent
  decoders do. Arbitrate with at least the in-org decoder plus one
  external (jxl-oxide / djxl); unanimous rejection means the
  bitstream is wrong, not the decoders.
- **Decoder fixes don't excuse the encoder.** Before blaming a
  decoder version, re-verify against the *latest* of all of them —
  but when current decoders are unanimous, stop looking for decoder
  bugs (the "should have landed already" hope cost one detour).

### 15. Verify fixes on every content class that failed

The two #68 causes presented identically (same `SectionTooShort`,
same e9+-only, same size gate) and were independent. The first fix
was verified on the noise repro and declared done; the harness re-run
exposed photo cells still red — their encodes were **byte-identical
before/after the fix**, the tell that their trees never hit the fixed
path. Rules:

- A fix is verified when **every cell that failed now passes**, not
  when one repro does. Byte-identical-pre/post on a still-failing
  input means a second cause, not a deeper version of the first.
- **Regression tests must be proven to bite**: flip the fix off and
  watch the test fail before committing it (the group-distinct
  synthetic was checked both ways). A regression test that never
  failed anything is decoration.
- **The corpus must cross the format's partition topology.** 256² and
  512² differ by *code path*, not just pixel count, for any sectioned
  format (one vs four modular groups — both #68 causes were
  structurally invisible on single-group images: one made the
  group_id property constant, the other needed a multi-section walk).
  Pattern 13's small-crop advice has a sharp edge here: crops sized
  inside one section would have hidden both bugs. Include at least
  one image per side of every section/tile/group boundary the format
  has.

### 16. Bisect rule-outs need plumb-checks

Ruling a knob out by toggling it is only valid if the toggle
*reaches the encoder*: jxl-encoder's `with_lz77_method` no-ops on the
lossless path, so "e9+Greedy still fails ⇒ lz77 isn't the culprit"
was a void inference (it also produced byte-identical output — the
tell, again). Every bisect arm carries a **bytes-differ assertion**
against the baseline; an arm that didn't change bytes ruled out
nothing. Same discipline as pattern 4's encode-proven exclusions,
pointed at debugging instead of fingerprints. When the public surface
can't express an arm (unconsumed setter, profile-internal field),
flip it in source behind a temp marker and revert — and when a
field-by-field profile diff is available, *enumerate the delta set
first* instead of bisecting from memory (the e8→e9 diff dump turned
an open-ended hunt into seven candidates).

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

All five codecs have adopted the playbook (each repo's own
`docs/VARIANT_GENERATION.md` is the authoritative status; this snapshot
is from 2026-06-11):

| Pattern | zenwebp | zenjxl | zenavif | zenpng |
|---|---|---|---|---|
| Variant-scoped knobs + validate() | **landed** — mode-discriminated `SweepVariant` (VP8/VP8L); exclusions documented (targets, preset macro-knob, alpha/exact, near_lossless) | **landed** (noise×lossless rejection et al., 2026-06-11) | **landed** (backend/420×RGB/420×16-bit rejections; no-op spellings made untypeable, 2026-06-10/11) | **landed** — near_lossless excluded as metric-class; `parallel` pinned per pattern 9 |
| Dominance cases | always-on optimized entropy | **landed** — container Auto is the only dominance case; audit in its doc | **landed** — container metadata is the dominance class; full audit in its doc | downcast-when-representable (encoder default; format negotiation, not a sweep axis) |
| Exact trials (pixel-invariant) | **audited** — the entire curated lossless (VP8L) space is trial-class by construction; exact lossless trial helper queued | **audited** — whole lossless knob space + lossy entropy-stage knobs (use_ans / histogram strategy / clustering) are trial-class; exact trials belong upstream where state can be shared | **confirmed none** — single-invocation engine + fixed container layout; the predicted empty trial class held | **the entire space is trial-class** (lossless); the zopfli-iteration trial is the `Effort(31+)` region, helper queued |
| resolve_plan() introspection | holds trivially — `SweepVariant::build()` constructs the literal config; no hidden resolution layer | **landed** (`__expert`, 2026-06-11) | **landed** (`PlanInput → EncodePlan`; engine mirrors pinned by encode, pattern 8) | `Compression::effort()` IS the resolution fn; the fingerprint hashes it (`Effort(13)` ≡ `Balanced`) |
| Fingerprint dedup | **landed** — every field hashed; 0 aliases (no plateaus in the quality chain) | **landed** (calibration-plateau q≤20 merges; exclusions encode-proven) | **landed** (threads pinned per pattern 9; every exclusion encode-proven) | **landed** — resolved-effort identity merges spelling aliases |
| Sweep planner + validation harness | **landed** — first harness run fully green (41 cells × 7 images; lossless exact, partial-MB 509×381 leg) | **landed** — mode-discriminated planner + harness with decode/roundtrip gates (caught jxl-encoder#68 ×2 + #69; patterns 14–16) | **landed** — probe-axis planner + harness (caught vaq@1.0 no-op, lru_on_skip envelope-death) + RGBA alpha leg + MLP feature emission | **landed** — exactness harness in the normal suite (~2.5 s); first run caught the downcast-format comparison hazard |
| Id grammar + parser (pattern 7) | **landed** (`vp8-`/`vp8l-`, totality test) | **landed** (bcf67e09 — q- AND distance-grids) | **landed** (a5a564f1 — totality test caught a tokenizer bug) | **landed** (`png-<preset>`/`png-e<n>`) |
| Executor wiring (step 8) | **landed** (zenmetrics 96234317) | **landed** (zenmetrics 57aa2963) | **landed** (zenmetrics 96a31b90) | **landed** (zenmetrics 171ebb04) |

Open items live in each repo's adoption doc; the common queued work is
the exact-trial helpers (zenwebp lossless, zenpng `Effort(31+)`,
zenjxl's upstream in jxl-encoder where encode state can be shared).

**Beyond the original five (2026-06-11):** zengif adopted
(quantizer-dominated metric axes; build-feature liveness structural —
uncompiled-backend ids rejected; ColorQuant's missing dithering knob
structurally excluded; palette-representable content must roundtrip
pixel-exact at q100/d0 through every compiled backend) and zentiff
adopted (all-trial-class ≤16-cell space; first harness run proved
`Predictor::Horizontal` gate-shadowed under `Uncompressed` and
documented PackBits' byte-level-RLE loss on RGB band content) — both
with step-8 zenmetrics wiring open (no `CodecKind` for gif/tiff yet).
zenbitmaps has no variant space to sweep, but it wired
zenpixels-convert's **load-bearing reduction** into its zencodec encode
path (PNM/QOI/BMP/TGA) — raw formats pay full price for dead lanes, so
the bit-exact narrowing wins more there than for any compressed codec.
Remaining codecs are wrappers/decoders/knob-free by design (engines
covered via zenavif/zenjxl; heic/zenraw/zenpdf decode-only;
zenbitmaps/fax knob-free).

(Execution models and the identity contract live in "Where each piece
lives" above and pattern 7 — not repeated here.)

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
6. **Empirical validation harness** (patterns 6 + 14 + 15) — run it;
   expect it to find real defects (zenjpeg: five on the first run;
   zenjxl: two encoder bugs + five mis-curated probes; zenavif: a no-op
   spelling + an envelope-dead knob). Decode-verify EVERY cell (lossless
   modes: exact roundtrip), and make the corpus cross the format's
   partition topology — sections/tiles/groups for sectioned formats,
   non-MCU-aligned dimensions for JPEG-shaped ones. Fix, re-run on every
   content class that failed, commit the TSV. *Gate: ALL HARD CHECKS
   PASSED, decode gate included.*
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

Wrapper codecs (engine behind a crate boundary) take patterns 8–13
alongside steps 1–4; format encoders (your bitstream, others' decoders)
treat patterns 14–16 as additional step-6 gates. Both families were
proven out in one day each (zenavif, zenjxl) once the zenjpeg shape
existed to copy.

## Known limits / open items

- Absolute-valued knobs interact badly with quality grids — sweep them
  as a clamped **delta against the grid q**, never as a static absolute.
  Proven twice: zenavif's `KnobProbe::AlphaQualityDelta` (±25, `Delta(0)`
  fingerprint-aliases the follow-color spelling), and zenjpeg's
  `moz_chroma_deltas` axis (2026-06-11: `moz[cqd-10]` resolves
  `clamp(q−10)` per cell; the id stays q-free, the parser resolves
  against the cell's own q, and the fingerprint mediates it through the
  resolved tables — `cqd-10` at q85 fingerprint-equals absolute `cq75`
  there and `cq40` at q50).
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
