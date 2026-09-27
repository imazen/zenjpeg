# zenjpeg target-quality instrument census (2026-08-27) — REGISTERED BEFORE RUNS

GOAL criterion 4: the corpus9 instrument census for zenjpeg's target loop.
zenjpeg's loop = `target_quality::search_target` (caller-driven bracketed
encode→decode→score); default seed = `anchor_guess`; the 2026-08-26 zq wave's
arm-B head SHIPPED as consts (`zq_seed::predict_q0_from_features`, safety
clamp internal) but is INERT — no src consumer wires it (verified by grep).

- Instrument: corpus9 (the family 9-ref set) × t{70,80,88} × k∈{2,3}
  (k = `max_encodes`); tolerance 0 (spend the full budget), emit = last
  trial's bytes (search_target semantics).
- Judge: decoded pixels via the GIT-PINNED zensim (rev = "9d8f73a5a82a944420ca0e040ecfcea0f4afa263") `ZensimProfile::
  latest()` — the same judge `zq_calibrate` uses; the loop and the judge are
  the same model by construction (self-consistent; the zq labels' zensimA-era
  offset is the wave's registered limitation, unchanged here).
- Arms: **A = anchor_guess** (shipped default) and **B = zq_seed q0**
  (in-binary zenanalyze features → `predict_q0_from_features`; None ⇒
  anchor fallback). Family bar for B: ≥15% median-|err| improvement over A
  AND ±2-hit count not regressed. A alone closes the census requirement.
- Harness: `zenjpeg/examples/zensim_census.rs` (this repo, loop-ownership),
  family TSV schema + seed_q column.

## RESULTS (2026-08-27, same day) — census CLOSED; arm B PASSES the family bar

| arm | k | median \|err\| | ±2 hits | photo | nonphoto |
|---|---|---|---|---|---|
| A anchor_guess (shipped default) | 2 | 3.657 | 8/27 | 2.834 | 7.283 |
| A | 3 | 2.556 | 11/27 | 2.311 | 4.747 |
| B zq_seed head | 2 | **1.905** | **14/27** | 1.671 | 4.330 |
| B | 3 | **1.383** | **17/27** | 1.160 | 3.488 |

B vs A: k2 **+47.9%**, k3 **+45.9%** (bar ≥15%), hits improve both k —
**PASS**, and unlike zenwebp/jxl the head improves BOTH classes: zenjpeg's
anchor_guess is the weakest baseline in the family, exactly where a fitted
head pays. Harness `zenjpeg/examples/zensim_census.rs`; cells at
`/mnt/v/output/zenjpeg/instrument-census-2026-08-27/`.

## PROPOSAL — ★ APPROVED + WIRED 2026-08-28 (explicit user yes, AskUserQuestion)
Wire `zq_seed::predict_q0_from_features` as the DEFAULT `TargetOptions::
q_start` source in the target-quality path (feature extraction is in-binary;
`None` fallback keeps anchor_guess — G-J3 shape). The head shipped as consts
2026-08-26 but has zero src consumers; this census is the instrument
evidence for consuming it. Wired 2026-08-28: `zq_seed::predict_q0_from_image` (the in-src image→features→seed composition; census harness now calls it — dedup) + `TargetOptions::seeded_for_image` = the canonical seeded constructor for zensim-target searches; plain `Default` stays seedless. Unit-tested (clamp + degenerate-input fallback + options carry).
