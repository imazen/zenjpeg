# zenjpeg lossy picker v0.5 — modes_full + tiny-image content-aware features (clean even/odd split)

Supersedes `zenjpeg_picker_v0.4_clean-imazen26-evenodd_2026-06-26.bin`, which was
trained on `scalar_dense` (trellis-lambda only — it could NOT pick subsampling). v0.5
trains on the full **`modes_full`** knob set, so it picks strategy × subsampling
(gls/jp3/moz/pw4 × 4:2:0/4:2:2/4:4:4/XYB-B-subsampled) — the parallel-audit's #1 fix.

- **Split:** clean origin even/odd (`scripts/picker/origin_split.py`): {0,2,4,6,8}=train,
  {1,3,5}=val, {7,9}=test. No sizing/crop/encode derivative leaks. 414 origins.
- **Corpus:** `clean-picker-corpus-2026-06-26` (all-origin, ≤1 MP) + dense-small
  `clean-picker-small-2026-06-27` merged → 11,535 (rendition, size) keys. Metric **score_zensim** (CPU).
- **DATA_STARVED genuinely cleared (no `--allow-unsafe`):** two root causes fixed —
  (1) tiny renditions were dropped because 13 of the 50 KEEP features are
  percentile/windowed and go NaN below the extractor's deliberate min-sample floor
  (#49). Fixed with **native + mirror-tiled-fill** content features: too-small
  renditions are mirror-tiled to ≥128 px (alternating flips — seamless; min measured
  by recovery, audit 2026-06-28) and the NaN features re-extracted; valid native
  features kept. Each tiny image gets its OWN content-derived percentile values
  (trend-audit-validated; no silent bad values). (2) `large` was demanded but absent
  from the corpus → `SIZE_CLASSES` auto-scoped to the present {tiny, small, medium}.
- **Model:** MLP 110→256×3→54 (hard-example-weighting `emae` to clear a worst-row
  tail), 173,622 params, 339 KB f16 (ZNPR). schema_hash 0x676f03b6e7401d90,
  n_inputs 109 (size_oh=3), n_outputs 54.
- **Held-out TEST (7/9 origins):**
  - argmin (K=1, pure pick): mean 6.71% overhead, argmin_acc 37.1%
  - top-K-verify (rank by predicted bytes, encode-verify K cheapest, pick min actual):
    K=2 3.33% · K=3 1.93% · K=5 **0.683%** · K=6 0.427% · K=7 **0.260%**
  - **val→test gap −0.01pp** — generalizes cleanly (no overfit despite 256³).
  - K=1 is higher than v0.4's scalar_dense (~0.5%) because strategy×subsampling is a
    much harder categorical pick than trellis-only; the proven deployment is top-K-verify.

## INFERENCE CONTRACT (read before wiring)

The tiny-image features in this bin come from **native + mirror-tile-to-128 fill**
(see `zenanalyze/zentrain/TINY_IMAGE_FEATURE_FIX_2026-06-28.md`). For inference to
match training on tiny inputs, feature extraction MUST produce the same values.
- **Once the zenanalyze #49 intrinsic fix lands** (extractor mirror-tiles too-small
  inputs internally), this is AUTOMATIC — the caller just extracts and tiny goes
  through the MLP. No external handling.
- **Until then**, a tiny input hits the extractor floor → NaN → OOD-fallback, so the
  tiny-MLP in this bin is NOT exercised at inference (medium/small are unaffected and
  correct). Do not rely on tiny-image picks until the extractor fix (or a caller-side
  mirror-tile-128 pre-extract) is in place.
