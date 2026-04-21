# BoundaryRdConfig::default() retune (low-Q sweep)

Date: 2026-04-21
Branch: `boundary-rd-low-q-sweep` (stacked on #102 `boundary-rd-rollup`)
Data: `benchmarks/low_q_full/{grid,per_class_per_q,best_per_class_per_q}.csv`
Harness: `zenjpeg/examples/boundary_rd_low_q_sweep.rs`

## TL;DR

Updated `BoundaryRdConfig::default()` from
`alpha=1.0, threshold=0.05, shrink=0.5, max_retries=2, above=false`
(shipped in PR #102) to
`alpha=2.0, threshold=0.02, shrink=0.5, max_retries=2, above=true`.

The new default strictly dominates the prior default on BBS BD-rate across
every (class_bucket, q_range) cell measured. It is **not** a strict Pareto
win — at low Q it trades ~0.17-0.20 points of SSIM2 BD-rate for substantial
extra BBS gain. At mid and high Q, the tradeoff is strictly Pareto.

## Methodology

- Corpus: 18 images total
  - 13 labeled `screenshot` (Screenshot + LineArt classifier labels;
    the 4 synthetic lineart images in `benchmarks/sweep_corpus/lineart/`
    all classify as ScreenContent, so the `screenshot` bucket effectively
    means "non-photo block-grid-aligned content").
  - 5 labeled `photo`.
- Quality levels: 8 (`Q in {5, 15, 30, 45, 60, 75, 85, 95}`),
  partitioned into `low={5,15,30}`, `mid={45,60}`, `high={75,85,95}`.
- Configs: 67 total — 1 baseline (boundary-RD off) and 66 combinations
  across `alpha x threshold x shrink x retries x above`.
- Total encodes: 18 * 8 * 67 = **9648** rows in `grid.csv`.
- Per-cell analysis: BD-rate on BBS (lower = better file/quality tradeoff)
  and SSIM2 (negative = quality gain), aggregated per
  (class_bucket, q_range) in `per_class_per_q.csv`.
- Composite score (for winner selection):
  `score = -BD_BBS - 5 * max(0, BD_SSIM2)` — negative SSIM2 BD-rate
  (a quality gain) is treated as free; positive (a quality loss) is
  penalised 5x.

## Decision

The sweep confirms the `alpha=2.0, threshold=0.02, shrink=0.5, retries=2,
above=true` config strictly dominates the current PR #102 default on BBS
BD-rate across every (class_bucket, q_range) cell.

| class/Q        | old default (a=1.0, t=0.05, above=false) |                  | new default (a=2.0, t=0.02, above=true) |                  | Verdict                                                   |
|----------------|-----------------------------------------:|-----------------:|----------------------------------------:|-----------------:|-----------------------------------------------------------|
|                |                              BBS BD-rate | SSIM2 BD-rate |                             BBS BD-rate | SSIM2 BD-rate |                                                           |
| screenshot/low |                                   -5.29% |           +1.06% |                                  -6.75% |           +1.26% | Larger BBS gain, +0.20% SSIM2 cost                        |
| screenshot/mid |                                   -3.95% |           +0.43% |                                  -4.58% |           +0.42% | Strictly better                                           |
| screenshot/high|                                   -3.61% |           -0.90% |                                  -5.17% |           -1.30% | Strictly Pareto better                                    |
| photo/low      |                                   -5.25% |           +1.30% |                                  -7.98% |           +1.47% | Larger BBS gain, +0.17% SSIM2 cost                        |
| photo/mid      |                                   -5.08% |           -0.24% |                                  -6.77% |           -0.31% | Strictly Pareto better                                    |
| photo/high     |                                   -4.63% |           -0.26% |                                  -5.94% |           -0.36% | Strictly Pareto better                                    |

(Numbers from `benchmarks/low_q_full/per_class_per_q.csv`; full per-image
BD-rate breakdowns + composite scores in the same CSV.)

## Framing

**This is not a universal Pareto win.** At low Q (5, 15, 30), boundary-RD
in both the old and new configurations shows a small SSIM2 BD-rate
regression (+1 to +1.5% at low Q). The new default trades an additional
~0.17-0.20 percentage points of that SSIM2 BD-rate for substantially more
BBS gain.

That tradeoff is the right call because:

1. The feature is motivated by visible boundary blocking between MCUs.
   Blocking artifacts are worst at low Q — precisely where the new
   default trades SSIM2 for BBS most aggressively.
2. BBS (the block-boundary-sensitive metric this work targets) is what
   the feature was designed to reduce. SSIM2 is a general-purpose
   quality metric that weights boundary artifacts less heavily than
   perceptually-measured boundary visibility.
3. At mid Q (45, 60) and high Q (75, 85, 95) — the Q range most
   production web traffic actually uses — the new default is strictly
   Pareto better on both metrics.

If a caller explicitly needs low-Q SSIM2 to be untouched, they can still
construct a `BoundaryRdConfig` manually or set `BoundaryRd::Off`. The
default is the "best we know for unknown content" point on the BBS/SSIM2
curve, not a claim that SSIM2 is never affected.

## Notes on classification

The coefficient-based `ImageClass` classifier labels all four synthetic
lineart images (`flat_blocks.png`, `text_label.png`, `ui_mockup.png`,
`vector_art.png`) as `ScreenContent`. The sweep harness groups
Screenshot, LineArt, and Synthetic into a single `screenshot` bucket;
that bucket effectively covers "non-photo content with sharp
block-grid-aligned edges."

## Dominance matrix note

The `best_per_class_per_q.csv` winner-by-composite-score doesn't always
pick the new default (e.g. `screenshot/low` winner by composite is
`a=0.5, t=0.10, r=1, above=true` because it has lower SSIM2 cost at the
expense of smaller BBS gain). The new default was chosen because it
**strictly dominates the prior default on BBS** across every cell — i.e.
no cell regresses. No other single config in the sweep achieves that. A
future per-image-class preset picker (#103) could select per-bucket
winners from this data directly.
