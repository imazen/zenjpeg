# zenjpeg Zq seed head — pre-registered wave (2026-08-26)

**Goal (criterion 4)**: a content-aware q0 seed for
`target_quality::search_target` (`TargetOptions.q_start`) that reduces
encode→decode→score iterations vs the content-blind `anchor_guess`, mirroring
zenavif's `q0_head` (the directive exemplar: fitted-constants head, no model
file at runtime).

## Data (frozen)
- `/mnt/v/output/canonical-picker-2026-07-01-zensimA/zenjpeg_lossy/{train,validate}.parquet`
  (266,640 train rows; split is origin-clean by construction — even/odd origin rule).
- Curve key: `(origin_id, cell, width, height)` — per-rendition; q grid
  {5,15,30,50,70,85,95} → 7-point curves, PAVA-isotonized (non-decreasing in q).
- Label: `q*(t)` = leftmost crossing of the isotonic curve at target `t`
  ∈ {40,45,…,90} on `score_zensim` (era profile **zensimA/07-01** — REGISTERED
  LIMITATION: the runtime loop may steer a newer bake; the seed only needs to
  be near the basin, the bracketed search corrects the residual. A refit on
  current-model rescoring is a follow-up data task, not this wave).
- Curves that never reach `t` on the grid: no label at that `t` (skipped, counted).
- q at crossing between grid points: linear interpolation on the isotonic curve
  (matches the sim oracle below).

## Features (frozen pool)
The exemplar candidate list (`fit_q0_head.py CAND_FEATURES`) intersected with
the named `feat_*` columns present in the parquet; `log1p` set carried over.
Greedy forward selection (max 8 features) by leave-one-origin-out p90 |q0−q*|
on TRAIN ONLY. Basis: exemplar's — `[1, tn, hinges(t), logpx_n, f_i, f_i·tn,
f_i·h80]` with `tn=(t−65)/25`, `h_k=max(t−k,0)/10`, `logpx_n=(ln(px)−13)/3`.
Fit: robust-L1 (IRLS) linear, per-origin inverse-count weights.

## Gates (frozen BEFORE any fit runs)
- **G-Z1 (diagnostic, reported not gating)**: validate |q0−q*| p50/p90 per
  target band.
- **G-Z2 (THE decision gate)**: offline secant simulation (verbatim port of
  `search_target` incl. `anchor_guess`, tol 0.5, max_encodes 8, integer-q
  rounding; oracle = the held-out validate curve's isotonic linear interp):
  **mean encodes-to-converge improves ≥ 10% vs the anchor_guess arm, AND
  converged-count does not regress** over all (validate curve × target) cells.
- **G-Z3 (safety, by construction)**: the head is `Option`-returning; any
  feature unavailability → `None` → caller falls back to `anchor_guess`
  (never degrades current behavior).

## Endgame (frozen)
PASS ⇒ consts into `zenjpeg/src/zq_seed.rs` (q0_head-style module, doc'd
provenance, unit tests incl. clamp/monotone/finite), TSV + fit table in
`benchmarks/`, plan/memory updated. FAIL ⇒ the miss is committed here with the
numbers; no consts ship.
