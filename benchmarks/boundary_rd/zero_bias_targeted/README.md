# Zero-bias-shrink targeted validation (Task 7 of #102 rollup)

Date: 2026-04-21
Harness: `zenjpeg/examples/boundary_rd_zero_bias_targeted.rs`
Corpus: 40 images stratified 10 per GPT category (illustration, screen_ui, screen_chart, screen_document) from `~/work/coefficient/scripts/selector_corpus/lineart/zenjpeg_tuning_corpus_gpt.txt`. The original run-time manifest.tsv was pruned.
Total encodes: 320 baseline (OFF) + 1920 candidate = **2240 in 55.4 s**
on the Ryzen 9 7950X.

## Question

Does `zero_bias_shrink < 1.0` provide **consistent per-image Pareto
improvement** over `zero_bias_shrink = 1.0` (off) in any
(content class, Q-range) cell?

## Why a re-run

The earlier broad 40-config sweep
(`benchmarks/boundary_rd/zero_bias_sweep/`) flagged 3 "winning" cells
(bilevel/high zb=0.5, doc-text/high zb=0.5, mixed-vector/high zb=0.3)
using per-cell BD-rate aggregates on 4-6 images. Per-image inspection
showed the BD-rate math was flattering interpolated curves rather than
demonstrating consistent per-image wins — in the doc-text/high zb=0.5
cell, one Gutenberg page was strictly **losing** at every Q level while
the cell-level BD-rate reported -8.94 % SSIM2. A decision on whether
to ship the knob needs per-image accounting at larger sample size.

## Method

### Grid

6 configs. Anchor is `BoundaryRd::Off`. C1 is the current shipped
boundary-RD default (no zb) — the comparison partner the question is
actually asking about.

| tag | α | threshold | aq_shrink | zb_shrink | retries | above |
|----|---|-----------|-----------|-----------|---------|-------|
| C1 | 2.0 | 0.02 | 0.50 | **1.00** | 2 | true |
| C2 | 2.0 | 0.02 | 0.50 | **0.70** | 2 | true |
| C3 | 2.0 | 0.02 | 0.50 | **0.50** | 2 | true |
| C4 | 2.0 | 0.02 | 0.50 | **0.30** | 2 | true |
| C5 | 2.0 | 0.02 | 0.50 | **0.15** | 2 | true |
| C6 | 2.0 | 0.02 | **1.00** | 0.30 | 2 | true |

### Q levels

`{5, 15, 30, 45, 60, 75, 85, 95}` — 8 levels, full range.
Q-ranges: low (≤30), mid (31-60), high (>60).

### Corpus

40 images, 10 per GPT category from the 1,375-image corpus
at `~/work/coefficient/scripts/selector_corpus/lineart/zenjpeg_tuning_corpus_gpt.txt`.
Stratified sample with deterministic seed (SHA-256 keyed with
`"42"`; see the original commit for the manifest (pruned from the repo), with SHA-256
per file for reproducibility). All 40 images are PNG or JPEG,
center-cropped to 512×512 when larger.

GPT categories in this corpus:

- `illustration` (578 in source, 10 sampled)
- `screen_chart` (224 in source, 10 sampled)
- `screen_document` (383 in source, 10 sampled)
- `screen_ui` (190 in source, 10 sampled)

### Decision rule (per-image Pareto)

For each image × (class, Q-range) cell × candidate C ∈ {C2…C6},
compute mean bytes and mean SSIM2 distortion over Q in the band.
Pareto win against C1:

    (bytes_C ≤ 1.02 * bytes_C1 AND ssim2_C ≤ 0.98 * ssim2_C1)
 OR (bytes_C ≤ 0.98 * bytes_C1 AND ssim2_C ≤ 1.02 * ssim2_C1)

(2 % slop per axis for measurement noise; strict win on one axis +
not-worse on the other.)

**Keep the knob** if any (cell, candidate) pair has
`pareto_win_fraction ≥ 0.70` AND
`mean_ssim2_improvement_abs ≥ 0.01`.

**Drop the knob** otherwise.

## Result: DROP

Across all **60 cells × 10 images = 600** image-cell-candidate
measurements, the Pareto rule fires on exactly **one** image:

    screen_chart/mid, image af92100…727efc0b
    bytes_C2 = 2539 vs bytes_C1 = 2519 (ratio 1.0081 — within slop)
    ssim2_C2 = 9.019 vs ssim2_C1 = 9.244 (ratio 0.9757 — within ≤0.98 bar)

C3, C4, C5, C6 on the same image produce **identical** bytes/ssim2
numbers — the retry loop converges to the same output regardless of
`zero_bias_shrink`, meaning the zb knob is doing nothing new on top
of the `aq_shrink=0.5` that all of them share. That single image
pushes the `screen_chart/mid` × Cₓ win fractions to 0.10 (1/10),
still 7× below the 0.70 pass bar.

Every other (cell, candidate) combination has `pareto_win_fraction = 0.00`.

See `per_cell_stats.csv` for the full 60-row summary.

### What's actually happening

`zero_bias_shrink < 1.0` **weakens** the zero-bias rule — it preserves
more small-magnitude AC coefficients that would otherwise be zeroed.
The effect is:

- **SSIM2 distortion improves** slightly (mean improvement 0.01–1.0 pt
  depending on cell) — matching the intended mechanism.
- **Bytes go UP** 1-5 % across every cell, because those preserved
  AC coefficients have to be encoded.

The byte increase always exceeds the 2 % slop, so the strict Pareto
rule never fires. The BD-rate aggregate in the broad sweep interpolated
these two opposing signals and reported "good" numbers in a small
handful of cells — per-image, the improvement is never Pareto.

### C6 sanity check

C6 uses `aq_shrink = 1.0` (disabled) with `zero_bias_shrink = 0.3`.
If `zero_bias_shrink` provided independent signal, C6 should show
cell behaviour distinct from C4 (same zb, but with aq_shrink=0.5).
C6 and C4 show near-identical `mean_bytes_ratio` (≈1.02-1.05) and
`mean_ssim2_improvement` (within 0.003 pt) across all 12 cells —
confirming the two knobs are **effectively collinear** on this
corpus. The aq_shrink=0.5 retry dominates the selection; zb_shrink
adjustments don't move the needle.

## Non-finding: illustration BBS = NaN

One of the ten `illustration` images (`e70742dcdf…d74b66bb90`)
is a near-uniform color region that produces BBS = 0 at every
quality on every config. This makes the cell-level
`mean_bbs_ratio` NaN (0/0) for illustration. It does NOT affect the
per-image Pareto rule, which operates on bytes and SSIM2 only. No
other cells are affected.

## Files

- `per_cell_stats.csv` — 60 rows, per (class, q-range, candidate)
  with `pareto_win_fraction`, `mean_ssim2_improvement_abs`,
  `mean_bytes_ratio`, `mean_bbs_ratio`, and the keep/drop decision.
  This is the headline decision table.

The original run also produced `corpus_manifest.tsv`, `grid.csv`
(2240 rows per image×config×quality), and `per_image_per_cell.csv`
(600 rows of per-image ratios). Those intermediate/raw files were
pruned — the decision summary in `per_cell_stats.csv` is self-contained.

## Reproduction

The harness (`zenjpeg/examples/boundary_rd_zero_bias_targeted.rs`)
and the public `with_zero_bias_shrink` API it exercised are both
removed as part of applying the DROP decision — there is no supported
way to re-run this sweep against the current tree. The evidence
below is the audit trail; restoring the API + harness from git
history (see the commit that introduced this directory) is what's
required to re-run.

The original run used:

```bash
# Against commit 6207b7e9 (tree that still had the zb knob):
cargo build --release -p zenjpeg --features "trellis decoder" \
  --example boundary_rd_zero_bias_targeted
./target/release/examples/boundary_rd_zero_bias_targeted
```

Runtime ~1 minute on a Ryzen 9 7950X. The manifest paths must be
accessible; SHA-256 values in the original manifest gate file-identity
verification.

## Action taken

The `zero_bias_shrink` knob is removed from the public API in a
follow-up commit. The broad sweep results and harness are also
deleted as superseded by this targeted evidence. The targeted harness
is deleted alongside the knob it tested.
