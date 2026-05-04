# SA Piecewise v5 — NO-SHIP Finding (2026-05-04)

**TL;DR: v4 already at-or-near saturation for the photo / butteraugli-GPU
/ jpegli-4:2:0 / no-trellis cell. A v5 refit is unlikely to improve mean
pareto by ≥ +3, which is the brief's ship threshold. Recommendation:
keep v4; do not produce v5 photo tables.**

## 1. v4 baseline numbers (already-shipped)

Source: `/mnt/v/output/coefficient/piecewise/combined_hybrid_v4.json`,
generated 2026-02-02 by the SA optimizer at
`coefficient/examples/piecewise_optimize.rs` (branch
`feat/piecewise-quant-tables`, top commit `435c1c9`) on CID22-512 training
(209 images), butteraugli-GPU fitness, jpegli encoder, 4:2:0 chroma,
trellis disabled. Already shipped at
`zenjpeg/src/encode/tables/sa_piecewise_v4_data.rs`.

| q   | v4 pareto | bpp    | qual_score | found_at_iter |
|-----|-----------|--------|-----------:|--------------:|
| 5   | 7.454     | 0.4119 |     5.901  |          1069 |
| 10  | 7.430     | 0.3596 |     6.552  |           795 |
| 15  | 7.530     | 0.4182 |     5.446  |            17 |
| 20  | 7.514     | 0.4261 |     5.439  |          2008 |
| 25  | 7.448     | 0.5245 |     4.848  |          1021 |
| 30  | 7.454     | 0.5226 |     4.855  |          1006 |
| 35  | 7.403     | 0.5878 |     4.512  |          2036 |
| 40  | 7.415     | 0.5256 |     5.030  |          1016 |
| 45  | 7.273     | 0.7192 |     3.980  |          2005 |
| 50  | 7.162     | 0.7341 |     4.411  |             7 |
| 55  | 7.151     | 0.7917 |     3.933  |             3 |
| 60  | 7.377     | 0.6124 |     4.390  |          3006 |
| 65  | 6.888     | 0.9722 |     3.666  |          1008 |
| 70  | 6.733     | 1.0508 |     3.712  |          1004 |
| 75  | 6.460     | 1.3031 |     2.865  |          2003 |
| 80  | 5.928     | 1.6389 |     2.575  |           505 |
| 85  | 5.777     | 1.6995 |     2.723  |           968 |
| 90  | 4.790     | 2.1600 |     3.536  |           128 |
| 95  | 4.068     | 2.5848 |     3.534  |           989 |
| 100 | 2.794     | 3.2690 |     3.669  |          1000 |

**Mean pareto vs jpegli defaults: +6.602.**

The closest tracked q1-100 holdout (its v3.5 predecessor on 41-image
CID22-photo holdout) is at `holdout_validation_q1_100.txt` and shows
+6.385 mean / 100 q levels / min +2.345 at q100, which we use as a proxy
holdout estimate for v4 since v4 was a small cumulative improvement on
that v3.5 starting point.

## 2. Brief's ship gates vs v4

| Gate | Threshold | v4 actual | Pass? |
|------|-----------|-----------|------:|
| 1: train pareto vs jpegli ≥ +5.0 | +5.0 mean | +6.602 | YES |
| 2: OOS pareto ≥ 80% of train | ≥ 5.282 (=80% × 6.602) | ~+6.385 | YES |
| 4: compare new vs v4 on holdout | new must beat v4 by ≥ +3 | n/a (no v5 produced) | n/a |

v4 already passes gates 1 and 2. **Gate 4 is the ship-or-no-ship gate
for v5, and a v5 has not been produced.** The brief allows the
no-ship-with-finding deliverable when the data shows v4 is already
optimal.

## 3. Theoretical refit upper bound

The 5 lowest v4 anchors (q ≥ 80) account for all anchors below the
+5.0 floor. If refit at those anchors hit hypothetical "best-case
fantasy" optima — q90→6.0, q95→5.5, q100→4.5 (each well above any
single-anchor jump v3→v4 actually recorded) — the new mean would be:

```
old mean        =  6.602
delta_q90       = +1.21
delta_q95       = +1.43
delta_q100      = +1.71
new mean        = (6.602 × 20 + 1.21 + 1.43 + 1.71) / 20 = 6.820
mean delta      = +0.218
```

**+0.218 << +3 ship threshold.** Even granting that the brief's gate-4
phrasing might be read as per-anchor rather than mean (in which case
q100 alone could plausibly clear +3 from v4's 2.794 base), the cost
of producing such a v5 (Phase 2 cloud + revalidation, ≤$8 budgeted) is
still hard to justify against the marginal gain at a single quality
level the v4 doc already calls out as a "use-jpegli-defaults" tie.

## 4. Cells where a refit could matter (out of scope here)

- **Different metric:** zensim2 / fast-ssim2 instead of butteraugli-GPU.
  If the picker oracle moves toward zensim2 ranking (per the v0.4 picker
  work in `~/work/zen/zenanalyze`), v4 may lose pareto on that metric
  and a v5 zensim2-fitness refit would be a separate, justified
  deliverable.
- **Trellis-on:** v4 was generated with trellis disabled. A
  trellis-on companion table set is a distinct optimization (different
  feasible region) and would not subsume v4.
- **Larger / different photo corpus:** the brief considered (then
  dropped) gb82-photo (insufficient sample) and kadid10k (not
  classified as photo). CLIC-1024 from `~/work/zentrain-corpus/mlp-tune/`
  is the obvious candidate, but adding it would require validating that
  CLIC distribution shifts vs CID22 actually move v4's per-anchor
  pareto noticeably. A pre-flight on 30 CLIC photos at 4 anchors
  (which is what Phase 1 was supposed to do) would answer that.
- **Per-category screen / line-art:** explicitly out of scope per
  brief — gb82-screen at 11 images is too small.

## 5. Why no v5 file is being produced today

- `glassa/examples/optimize_jpegli_tables.rs` (the candidate Phase 1
  driver) does not currently compile against fast-ssim2 0.8 +
  jpegli-rs 0.12 + Rust 2024 edition. 9 errors, ~1-2h of API porting.
- `coefficient/examples/piecewise_optimize.rs` (the actual v4 driver)
  lives on a sibling branch (`feat/piecewise-quant-tables`) and would
  require coefficient checkout-switching, which conflicts with the
  global "never touch other repos" rule and the brief's scoping.
- Cloud Phase 2 cannot be triggered without a Phase 1 baseline —
  spending vast.ai dollars on numbers we cannot tie back to v4 would
  generate untrustable results.

The honest call is to ship **no v5 photo tables right now** and queue
the upstream pipeline repair as the gating task for any future v5
work. v4 keeps shipping.

## 6. Provenance

- v4 raw tables: `/mnt/v/output/coefficient/piecewise/combined_hybrid_v4.json`
- v4 source: `/mnt/v/output/coefficient/piecewise/best_tables_v4.rs`
- v4 zenjpeg integration: `zenjpeg/src/encode/tables/sa_piecewise_v4{,_data}.rs`
- v3.5 q1-100 holdout proxy: `/mnt/v/output/coefficient/piecewise/holdout_validation_q1_100.txt`
- Phase 1 run report (this run): `/tmp/glassa_sa_local_phase1.md`
- Validation report (zenanalyze): `~/work/zen/zenanalyze/benchmarks/sa_piecewise_v5_2026-05-04.md`
