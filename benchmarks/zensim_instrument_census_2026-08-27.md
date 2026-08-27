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
