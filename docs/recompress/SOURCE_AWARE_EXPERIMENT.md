# Source-aware quantization experiment branch

Branch: `experiment/source-aware-quant-rd`.
Design: [SOURCE_AWARE_QUANTIZATION.md](SOURCE_AWARE_QUANTIZATION.md).
First pilot: [results and next iteration](../../benchmarks/source_aware_quant_2026-09-27/README.md).

## Goals and current scope

- [x] Isolate the work on a branch with the design preserved.
- [x] Implement repeatable jpeglish, moz, and classic pixel-encode sweeps.
- [x] Compare generic tables, exact-table controls, table-only hints, and
  candidates informed by source coefficient histograms.
- [x] Record SSIMULACRA2 and Butteraugli against both the source JPEG and
  the original PNG; keep each encoded JPEG and its final quantization tables.
- [x] Compute per-image BD-rate, overlap coverage, regression counts, and
  measured choices at fixed byte budgets. Plot RD curves from saved results.
- [ ] Establish gains on separate development and holdout corpora, including
  screenshots, small/odd dimensions, and external C-encoder sources.
- [ ] Add coefficient-domain explicit-table emission and an identity/lossless
  control; compare it with the three pixel pipelines.
- [ ] Explore source-aware AQ/zero-bias and trellis costs independently of DQT
  selection, then evaluate joint candidates with a bounded IQA budget.
- [ ] Promote demonstrated policies into the shared table resolver and
  per-image request context; recalibrate recompression routing afterward.

The first iteration is a development example using existing APIs, not a new
encoder default or stable API. It starts with the pixel paths because exact
custom tables let all three modes participate without changing the emitter.
The coefficient-domain experiment remains the next independent control.

Working optimization goals:

1. Lower BD-rate under **both** SSIM2 and Butteraugli against the native
   `generic` baseline, assessed separately for each destination mode.
2. Confirm the gain against `exact` too, so table-layout changes cannot be
   mistaken for a quantization improvement.
3. For per-image IQA search, satisfy the actual byte ceiling and retain the
   generic candidate as fallback. Inspect the other metric before accepting a
   metric-specific winner; a blended scalar must not conceal disagreement.
4. Report cumulative and generation-reference results, worst image, regression
   count, overlap coverage and search cost. A mean win alone is insufficient
   for promotion; choose a regression budget on development data before the
   holdout run. No threshold has been claimed as met by the pilot.

Build note: the synced lockfile had one stale `zenpredict -> archmage` edge.
Cargo removed that unused edge; this one-line lockfile correction is included
so the documented `--locked` build works. No dependency versions were changed.

## Run

```sh
ZENJPEG_SKIP_CPP=1 cargo build --release --locked -p zenjpeg \
  --example source_aware_rd --features __test-utils

target/release/examples/source_aware_rd \
  --corpus /path/to/srgb-png-corpus \
  --output /path/to/new-run-directory \
  --limit 3 --sources jpegli,moz,classic --destinations jpegli,moz,classic \
  --source-qualities 80 --qualities 45,55,65,75,80,85,95 \
  --strengths 1,4 --window 0.2 --sampling 420

python3 scripts/source_aware_report.py /path/to/new-run-directory
# Optional standalone SVG figures; uv supplies an isolated plotting environment:
uv run --with matplotlib python scripts/source_aware_report.py \
  /path/to/new-run-directory --plots
```

Use `--sampling 444` for a separate run. No resizing/cropping is done; inputs
are expected to be 8-bit sRGB PNGs with opaque content. Quality numbers follow
each mode's own scale and are not treated as cross-mode perceptual equivalents.
Sources currently use zenjpeg's mode emulations, not the external C encoders.

The output directory must be new. Every point is flushed immediately so failed
runs retain evidence; only a successful run gets a `COMPLETE` marker. Do not
rerun encoding to change report formatting. The report works from saved JSONL.

## Variants and the first hypothesis

- `generic`: normal destination mode's encoder/table pipeline.
- `exact`: identical DQT values and resolved zero-bias, through exact custom
  tables. The harness verifies identical decoded pixels. Custom tables use
  three table slots, so moz's usual two-table output may have different bytes.
  Compare candidates against both generic and exact controls.
- `hint_sN`: select each AC step from a ±20% window around the target, using
  only the source step and a fixed triangular prior on stored levels −16..16.
- `hist_sN`: same bounded search, with the source's actual per-frequency
  coefficient histogram. Source tables are indexed by component table selectors
  and coefficients are explicitly converted from zigzag to natural order.

The candidate objective is added coefficient MSE in fixed target-step units
plus `N * ln(candidate_step / target_step)^2`. It is deliberately a simple
alignment hypothesis, not a JPEG-rate estimator. It includes every legal integer
in the window, so nearby divisors/multiples compete with ordinary entries.
DC stays fixed. Candidate zero-bias remains at the exact baseline's values.
The final encodes and IQA measurements determine whether any apparent scalar
benefit survives AQ, trellis, and pixel round trips.

`encode_ms` measures only each final pixel encode; it excludes source decode,
histogram collection, baseline DQT harvesting, table search, and IQA. It is
diagnostic and must not be presented as end-to-end optimization cost.

## Reading the results

- `points.jsonl`: all raw measurements, JPEG paths, dimensions, bytes, tables.
- `run.json`, copied `Cargo.lock`, and compiled experiment source snapshots:
  arguments, revision/dirty state, metric implementation provenance.
- `bd_rate.csv`: per-image/source-quality/sampling/reference/metric curves,
  compared against generic and exact controls. Negative means fewer bytes.
- `summary.md`: geometric-mean rate changes, worst cells, regression counts.
- `goal_choices.csv`: measured best candidate at each generic point's byte
  budget, once for SSIM2 and once for Butteraugli; includes both metric scores
  so choosing one metric cannot hide damage to the other. This is a per-image
  search oracle over the measured candidates, not a trained one-shot policy.
- `joint_choices.csv`: smallest measured output at each generic point's byte
  ceiling with **both** SSIM2 and Butteraugli no worse than that point. Includes
  the generic fallback and candidates at other sampled qualities. Generation
  and cumulative gates are separate; passing one does not imply passing both.
- `curves_*.svg`: rate–distortion figures, if requested.

BD-rate integrates piecewise-linear **log rate** over overlapping distortion
only, using Pareto frontiers. SSIM2 maps to `100 - score`; Butteraugli uses raw
distance. No extrapolation or arbitrary nonlinear metric conversion. The two
metrics have independent results. Curves with fewer than four Pareto points
are marked sparse and excluded from aggregate claims, although diagnostic
two-point results remain in the CSV. Inspect overlap coverage before drawing
conclusions, and use denser quality sampling to confirm a promising policy.

Generation-reference IQA quantifies drift from the available JPEG. Cumulative
IQA uses the known original and can expose preservation of existing artifacts.
Only generation-reference selection is possible when the original is absent.
Neither should silently substitute for the other.

## Focused verification

```sh
python3 -m unittest discover -s scripts -p test_source_aware_report.py
rustc --edition=2024 --test zenjpeg/examples/source_aware/quant.rs \
  -o /tmp/zenjpeg-source-aware-quant-tests
/tmp/zenjpeg-source-aware-quant-tests
```

The tests cover signed ties, exact divisors including extended-range products,
candidate bounds, zero-frequency stability, known BD-rate ratios, dominated
points, and no-overlap handling. Every sweep verifies exact-control DQT and
decoded-pixel equality at each destination quality.
