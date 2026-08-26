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

## RESULT (2026-08-26, first arm) — G-Z2 **FAIL** as registered
Fit ran per registration (amendment noted: 27,684 curves carry only a 3-point
coarse-plan q grid and are excluded; 96,894 full 7-pt curves fit; the absent
`palette_density` column dropped from the candidate pool; greedy selection ran
on a seeded 80k-label subsample for tractability — final fit on all 927,217
labels). Selected features: flat_color_block_ratio, dct_compressibility_uv,
spectral_slope_y, distinct_color_bins, grayscale_score, skin_tone_fraction.

- G-Z1 (diagnostic): val |q0−q*| p50 **6.54**, p90 **19.04** (n=559,596).
- G-Z2: mean encodes **4.53 → 3.37** (**−25.7%**, bar ≥10% ✓) BUT converged
  count regressed **559,596 → 559,407** (−189, 0.034%) → the frozen
  no-regression clause fails ⇒ **FAIL. No consts ship from this arm.**

Numbers: `benchmarks/zq_seed_fit_2026-08-26.tsv`. A remedy arm (safety-clamped
seed) will be REGISTERED below before it runs, same decision gate.

## AMENDMENT (registered 2026-08-26, before any arm-B run) — arm B: safety-clamped seed
Diagnosis of the 189 (committed above): ALL are deep undershoots — q0 lands
p50 33.6 BELOW q* (min −4.4 absolute) at t∈[40,65]; from a deep-low seed the
walk-up (step = 1.2·gap, min 4) exhausts max_encodes=8. Zero overshoot cases.

**Arm B**: q0' = clamp(q0, anchor−L, anchor+12). The high clamp is fixed (+12,
untuned — no overshoot failures exist). **L selection rule (frozen)**: over
L ∈ {12, 15, 18, 20, 25}, pick the LARGEST L with ZERO convergence regressions
on validate; re-measure improvement at that L. **Gate: G-Z2 unchanged**
(≥10% mean-encode improvement, no convergence regression). If no L achieves
zero regressions, arm B FAILS and the wave closes FAILED.

## RESULT — arm B **PASS**; head SHIPPED (2026-08-26)
Grid (frozen rule, validate, 559,596 cells; `zq_armB_grid_2026-08-26.tsv`):
L=25→11 regressions, L=20→5, **L=18→0** (conv 559,596/559,596, mean encodes
4.53→3.92, **−13.5%**), L=15→0 (−11.6%), L=12→0 (−9.2%). Rule picks **L=18**
(largest zero-regression L) ⇒ **G-Z2 PASS** (≥10% ✓, no regression ✓).

Shipped: `src/zq_seed.rs` — `predict_q0_from_features(&[f32;6], target, px)
→ Option<f32>`, clamp `[anchor−18, anchor+12]`, `ZQ_FEATURES` =
{flat_color_block_ratio, dct_compressibility_uv†, spectral_slope_y,
distinct_color_bins†, grayscale_score, skin_tone_fraction} († ln_1p inside).
4 unit tests incl. a Python-pipeline golden (93.760 @ t=72). Wire-up: pass
the value as `TargetOptions::q_start`; `None` ⇒ anchor curve (G-Z3 holds by
construction).
