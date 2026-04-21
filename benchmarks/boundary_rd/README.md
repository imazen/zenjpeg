# Boundary-RD benchmark evidence (#91 / #102)

This subdirectory holds the committed benchmark evidence for the
boundary-continuity RD refinement feature (`EncoderConfig::boundary_rd`)
— the BD-rate sweeps that motivated `BoundaryRdConfig::default()`,
the SIMD overhead measurement, the line-art sweep corpus, and the
zero-bias-shrink knob sweep.

## Contents

| Path | Role |
|---|---|
| `default_rationale_2026-04-21.md` | Rationale for the current `BoundaryRdConfig::default()` values. Cites the low-Q full-grid sweep (next row). |
| `simd_overhead_2026-04-20.md` | Wall-clock cost of the SIMD boundary-RD helpers (D_b, ac_energy, ref-block pack). |
| `low_q_full/` | Phase-5 full-grid low-Q sweep (`grid.csv`, `per_class_per_q.csv`, `best_per_class_per_q.csv`). Source data for the current default. |
| `zero_bias_sweep/` | Zero-bias-shrink sweep (Task 6): cross of `α × threshold × aq_shrink × zero_bias_shrink × max_retries`. See its local README for decision. |
| `sweep_corpus/` | Committed synthetic line-art PNGs + an external manifest TSV pointing at source-of-truth corpus directories on block storage. |

## Reproduction

All sweeps drive through `zenjpeg/examples/boundary_rd_low_q_sweep.rs`.
The default `--output` path points at `benchmarks/boundary_rd/low_q_full/`
and the default `--lineart-dir` at `benchmarks/boundary_rd/sweep_corpus/lineart/`.

```bash
cargo run --release --example boundary_rd_low_q_sweep \
  --features "trellis decoder" \
  -- --output benchmarks/boundary_rd/low_q_full
```

Raw CSVs are regenerated on each run; the committed copies are the
ones the current default was tuned against. Don't rerun them blindly —
they represent ~8–30 minutes of encodes per sweep.

## Why this subdirectory exists

Prior to the rollup, boundary-RD evidence was scattered across the
top-level `benchmarks/` directory (`bbs_baseline_*`, `boundary_rd_*`,
`low_q_full/`, `sweep_corpus/`), mixed with unrelated benchmarks
(container probing, wide-migration, xyb-perf, RD-exploration). Moving
everything boundary-RD-specific under `benchmarks/boundary_rd/` keeps
the repo root-level `benchmarks/` legible and groups the evidence a
reviewer needs when evaluating changes to `BoundaryRdConfig::default()`.
