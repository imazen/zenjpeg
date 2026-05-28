# benchmarks/ index

This directory holds **committed** sweep results and validation data
that informs calibration and shipping decisions. Every committed file
should be reproducible from a documented `zjr-calibrate` invocation.

## Files

| File | Source command | Notes |
|---|---|---|
| `seed_calibration_smoke_2026-05-28.tsv` | `zjr-calibrate cumulative-sweep --references <synth-ppm> --source-qs 30,60,85 --targets 50,70,85 --subsampling 420` | First smoke run of the v0.1 sweep harness over 3 synthetic 96×96 PPMs (checker / noise / stripes). 27 rows. **Synthetic patterns are degenerate** (per CLAUDE.md gradients-banned rule); the data exists to show the harness runs end-to-end. **Calibration errors of 25–83 zensim-A confirm the seed table needs corpus-driven recalibration** — that is the v0.2 deliverable. Generator: `scripts/make_synthetic_ppm.sh`. |
| `cid22_3img_seed_calibration_2026-05-28.tsv` | `zjr-calibrate cumulative-sweep --references /mnt/v/dataset/cid22/CID22/original/{1001682,1025469,1028637}.png --source-qs 30,50,75,90 --targets 40,55,70,80,90 --subsampling 420` | First **real-image** end-to-end sweep on three CID22 originals. 60 rows: 27 tuned, 3 lossless, 30 NoOp. **No-size-regression invariant holds for every row** (max ratio = 1.0000). **Calibration MAE of 6.94 zensim-A** (max 15.93) — an order of magnitude tighter than the synthetic baseline. v0.1 seed-table baseline. |
| `cid22_15img_seed_sweep_2026-05-28.tsv` | `zjr-calibrate cumulative-sweep --references <15 CID22 PNGs> --source-qs 20,30,40,50,60,70,80,90,95 --targets 30,40,50,60,70,80,85,90 --subsampling 420` | 1080-row 4:2:0 seed-calibration baseline. **MAE 12.67** (analytical seed anchor too coarse — fitted-table replacement lands at 6.78). |
| `cid22_15img_postfit_v2_420_2026-05-28.tsv` | Same command after enabling Tuned `auto_optimize(true)` (zenjpeg `feature = "trellis"`). | 1080 rows. **MAE 11.30** — auto_optimize lifts the achievable target zensim-A by 3-5 points at the high end. |
| `cid22_15img_fitted_calibration_2026-05-28.tsv` | Same command with `src/calibration/data.rs` 2D-table wired into the router. | 1080 rows. **MAE 6.78 — at the irreducible 6.83 per-cell content-variance floor.** This is the v0.2 production calibration. |
| `cid22_15img_seed_sweep_444_2026-05-28.tsv` | Same command at `--subsampling 444`. | 1080 rows of 4:4:4 baseline used to fit the v0.3 4:4:4 lookup table. |
| `aq_ablation_10refs_2026-05-28.tsv` | `cargo run --example aq_ablation` — 10 CID22 refs × 3 source-q × 3 target × {AQ on, AQ off}. | The AQ decision study. Pairs each cell's AQ-on vs AQ-off output. Finding: AQ is a **consistent trade**, ~3-5% smaller for ~0.5-1.7 zensim-A across all cells — never free. Justifies the headroom gate. Includes per-source activity-tier histograms. |
| `cid22_15img_aq_headroom_420_2026-05-28.tsv` | `zjr-calibrate cumulative-sweep` 4:2:0 with tiered AQ + headroom gate (AQ fires only when projected − target ≥ 2 zensim-A). | 4:2:0. 383 preserve outputs. **Under-target delivery 8.4%** (down from 34.6% with always-on AQ — the headline metric). Over-target 86.4% (the safe direction). Zero size regressions. |
| `cid22_15img_aq_headroom_444_2026-05-28.tsv` | Same at `--subsampling 444`. | 4:4:4. 342 preserve outputs. Under-target 4.4%, over-target 91.8%. Zero size regressions. |

## Reproducing a result

Every TSV in this directory should have a matching command in the
table above. To reproduce the smoke baseline:

```bash
bash scripts/make_synthetic_ppm.sh /tmp/zjr-smoke/refs
cargo build --release -p zjr-calibrate
./target/release/zjr-calibrate cumulative-sweep \
    --references /tmp/zjr-smoke/refs \
    --output /tmp/zjr-smoke/cumulative.tsv \
    --source-qs 30,60,85 --targets 50,70,85 --subsampling 420
```

## What's NOT in benchmarks/

- Raw corpus image bytes — pulled from `/mnt/v/input/...` or
  `s3://codec-corpus/...`; never committed.
- The full multi-thousand-row sweep parquet — block storage at
  `/mnt/v/zen/zenjpeg-recompress/sweeps/<date>/` once produced.
  Only summaries and committed TSV smoke artifacts live here.
- Validation pixel diffs / diffmap PNGs — block storage; never
  committed.

## Naming convention

```
<workstream>_<date>.tsv      committed sweep result
<workstream>_<date>.meta.toml optional manifest (sha256, git-rev, command)
<workstream>_<date>.md       human summary / methodology
```

Workstreams currently active:

- `seed_calibration_smoke` — proves the harness runs end-to-end.
- `cid22_qsweep` — TODO (v0.2): cumulative-sweep against
  codec-corpus `sc` references at Q20–Q100 step 2, all four
  subsamplings, all 21 targets.
- `validation_holdout` — TODO (v0.2): 50-image cid22 holdout
  rerun used to gate `[Unreleased]` → `[0.1.x]` releases.

## Theory + cross-check artifacts (2026-05-28)

| File | Source command | Notes |
|---|---|---|
| `aq_direction_10refs_2026-05-28.tsv` | `cargo run --example aq_direction` | flat vs busy AQ targeting, zensim only. 10 refs × 3 src-q × 3 tgt × 4 variants. Flat-targeting 2–13× more byte-efficient at every source quality. Backs `docs/AQ_DIRECTION.md`. |
| `tri_metric_crosscheck_6refs_2026-05-28.tsv` | `cargo run --example tri_metric_gen` + `zen-metrics batch` (butteraugli-gpu, cvvdp, zensim-gpu) | 378 rows: both experiments scored under all 3 metrics. No decision flips. Backs `docs/TRI_METRIC_CROSSCHECK.md`. Columns: ref source_q target_q experiment variant size_ratio zensim butter_pnorm3 cvvdp. |
