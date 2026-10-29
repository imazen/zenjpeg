# zenjpeg lossy picker v0.4 — CLEAN even/odd-by-origin split (imazen-26)

First picker on the **clean** train/val/test split — by ORIGIN image, last digit:
{0,2,4,6,8}=train, {1,3,5}=val, {7,9}=test (zenmetrics `scripts/picker/origin_split.py`).
No sizing/crop/encode derivative leaks across the split.

- **Corpus:** `clean-picker-corpus-2026-06-26` (imazen-26 representatives, 414 origins,
  Lanczos size-ladder renditions). Sweep: `zenmetrics sweep --plan scalar_dense`
  (q-grid 5,15,30,50,70,85,95), metric **score_zensim** (CPU). Variants persisted to R2.
  Produced by the Hetzner chunk fleet `clean-jpeg-213753` (decode-once chunk mode).
- **Split:** train 212 / val 128 / test 74 origins (35613 / 21772 / 12586 cell rows).
- **Features:** 50 zenanalyze content features (of the picker `_WANTED` set present).
- **Results (held-out TEST = 7/9 origins):**
  - Student argmin (K=1, pure pick): **mean overhead 0.47%**, argmin_acc 88.1%
  - top-2-verify: **0.235%** · top-3-verify: **0.165%** (encode-verify K cheapest, pick min actual)
  - **val→test gap +0.01pp** — generalizes cleanly (no overfit to val).
- **Model:** MLP 110→192×3→39, 102951 params, 208 KB f16 (ZNPR). schema_hash 0x4e4b550b586b23d6.
- Supersedes `zenjpeg_picker_v0.3_2026-05-04.bin` (older even-only data, pre-clean-split).
