# Recalibration protocol

How to re-fit every calibration table from a fresh (better) image corpus,
reproducibly, with a validation gate. The whole pipeline is driven by one
script:

```bash
scripts/recalibrate.sh <originals_png_dir> [work_dir] [qstep]
```

It is idempotent per stage and prints a GATE verdict at the end. Read
this doc once; after that the script is the source of truth.

## What gets recalibrated

| artifact | file | how |
|---|---|---|
| per-encoder achieved-quality + size-ratio tables (turbo, mozjpeg, jpegli × preserve, tuned) | `src/calibration/per_encoder.rs` | **auto-spliced** between the `=== BEGIN/END GENERATED TABLES ===` markers by `fit_per_encoder.py` |
| per-encoder source-quality anchors (IJG-Q → zensim-A) | `src/target.rs::ijg_q_to_zensim_a` | emitted to `source_anchors.rs`; **manual paste** |
| jpegli 4:2:0 / 4:4:4 cumulative tables | `src/calibration/data.rs` | emitted to `data_420.rs`; **manual paste** |
| delivery-confidence shifts (P25..P95) | `src/api.rs::Confidence::target_shift` | emitted in the validation report (advisory); **manual paste** |

> **Confidence shifts are currently global but want to be per-encoder
> (v0.3).** The script derives one set from the mixed-encoder
> validation residual. But turbo/mozjpeg route to Tuned and *over*-deliver
> (residual centered above target → the script suggests a *negative*
> P50 shift for the mix), while jpegli's achieved tracks target (≈0).
> A correct model fits `Confidence::target_shift` per encoder class. The
> shipped global values were fit on the cleaner jpegli cumulative
> residuals; treat the script's mixed-set output as a sanity check, not
> a drop-in.
| AQ tier thresholds | `src/aq.rs` | re-tune via `aq_ablation` / `aq_direction` examples if the corpus shifts them; not auto-fit |

The mostly-auto target is `per_encoder.rs`; the small hand-maintained
constants (anchors, data.rs, confidence) are emitted ready-to-paste so a
human reviews the numerical diff before they land.

## Corpus requirements

Per CLAUDE.md sweep discipline — the corpus must span four axes or the
calibration is wrong on the axis someone skipped:

1. **Content**: tiny + photo + screenshot + line-art + mixed. Use named
   corpora (codec-corpus `sc` + CID22 holdout-safe set + GB82
   screenshots). **≥ 40 references per content class** for stable
   percentile fits.
2. **Size**: tiny (≤64²) + small (256²) + medium (~1MP) + large (4K).
   The per-cell achieved tables are size-agnostic only if the corpus
   spans sizes; partial-MCU edge cases live at odd small sizes.
3. **Source quality**: q20–q95 step ≤ 5 (the script default is step 5;
   pass `2` for the full step-2 grid the product targets). The low-q
   band (q20–q40) must be as dense as the high-q band.
4. **Encoder**: libjpeg-turbo (IJG tables), mozjpeg + ImageMagick
   (Robidoux tables), jpegli (distance tables). The script pins
   turbo 3.1.0 + mozjpeg 4.1.5 from the all-the-images docker stages.

Originals must be PNG (the script converts to PPM for turbo/mozjpeg).

## Stages (what the script does)

0. **Prereqs**: checks `zenmetrics` (GPU build), `cjpegli`; extracts the
   pinned encoders from `ati-mozjpeg` / `ati-turbo` docker images via
   tar-stream (WSL docker bind-mount / `docker cp` of `/tmp` is
   unreliable — the tar stream is the robust path).
1. **Encode sources**: each original through turbo/mozjpeg/jpegli at the
   q grid → `<work>/sources/<enc>/<ref>__<enc>__q<Q>.jpg`.
2. **Source-quality anchors**: score every source vs its original with
   `zensim-gpu`, fit per-encoder IJG-Q→zensim-A → `source_anchors.rs`.
3. **Forced-strategy sweeps**: `recompress-sweep --force-strategy
   {preserve,tuned}` per encoder → the per-encoder fit data.
4. **Fit + splice per-encoder tables** into `per_encoder.rs`.
5. **jpegli cumulative tables**: `cumulative-sweep` → `fit_calibration.py`
   → `data_420.rs`.
6. **Rebuild + validation gate**: rebuild, run tests, then run a
   router-chosen `recompress-sweep` per encoder and report under-target
   delivery + size regressions + the derived confidence shifts.

## Gate (must pass before committing)

| metric | threshold | rationale |
|---|---|---|
| size regressions | **= 0** per encoder | the no-inflation invariant is non-negotiable (coefficient algebra guarantees it; a regression means a bug) |
| under-target delivery | **≤ 15 %** per encoder | matches the current turbo 12 % / mozjpeg 8 % / jpegli 4 %; a fresh corpus should not regress this |
| `cargo test --features expert` | all pass | the frozen-API + identity-emit + AQ + confidence invariants hold |

The script prints `GATE: PASS|FAIL`. On FAIL, do **not** paste the new
constants — investigate (usually a source-anchor or per-encoder-table
miss; compare `val_<enc>.tsv` against the prior committed sweep).

## What to commit

After a PASS:

1. `zenjpeg-recompress/src/calibration/per_encoder.rs` (auto-spliced).
2. Paste `source_anchors.rs` → `target.rs::ijg_q_to_zensim_a`,
   `data_420.rs` → `data.rs`, the confidence shifts → `api.rs`.
   Review each numeric diff.
3. Copy the sweeps you want kept to
   `benchmarks/<workstream>_<YYYY-MM-DD>.tsv` and update
   `benchmarks/INDEX.md` with the corpus + command.
4. Update `docs/MULTI_ENCODER_VALIDATION.md` + the compendium with the
   new under-target numbers and corpus size (n per cell).
5. Bump the date stamp in `per_encoder.rs` / CHANGELOG. One commit:
   `recalibrate: <corpus> (<N refs>, turbo 3.1.0 + mozjpeg 4.1.5 + jpegli)`.

## Raw data — keep it

Per CLAUDE.md "always persist encoded variants": the
`<work>/sources/` corpus and all sweep TSVs cost real GPU time. Mirror
them to block storage (`/mnt/v/zen/zenjpeg-recompress/recal-<date>/`)
before deleting the work dir. The committed `benchmarks/*.tsv` are
summaries; the full per-cell sweeps + encoded sources live in block
storage with a `.pointer.md`.

## Reproducibility notes

- Pinned encoders make the source corpus deterministic. Re-extract from
  the same docker tags (turbo 3.1.0, mozjpeg 4.1.5) to reproduce exactly.
- `zenmetrics` GPU scoring is deterministic per build; record the
  `zenmetrics` git commit in the benchmark `.meta`.
- The fitter (`fit_per_encoder.py`) is pure (median over cells) — same
  TSVs in → byte-identical tables out (verified round-trip 2026-05-28).
- Date stamps and `Math.random`-style nondeterminism are absent; the
  pipeline is a pure function of (corpus, encoder versions, q grid).
