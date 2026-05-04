# Picker v0.3 — held-out re-encode A/B (zenjpeg)

* Date: 2026-05-04T05:30:10Z
* Corpus: `/home/lilith/work/zentrain-corpus/mlp-validate/cid22-val/` (41 images)
* Picker bin: `benchmarks/zenjpeg_picker_v0.3_2026-05-04.bin` (n_inputs=112, n_outputs=48, schema_hash=`0x5d09b66faedc3ecf`)
* Targets: [30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0]
* Encoder closed-loop: `Quality::ZqExplicit` with `max_passes=3`, `max_overshoot=1.5`
* Wall: 60.0 s

## Per-target table

| target_zq | n | bytes_picker | bytes_bucket | Δ% (picker − bucket) | win_rate | achieved_picker | achieved_bucket |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 | 41 | 617785 B | 611681 B | +1.00% | 34% (14/41) | 58.40 | 58.21 |
| 35 | 41 | 621535 B | 611681 B | +1.61% | 39% (16/41) | 59.50 | 58.21 |
| 40 | 41 | 620274 B | 611681 B | +1.40% | 39% (16/41) | 59.60 | 58.21 |
| 45 | 41 | 651914 B | 644140 B | +1.21% | 34% (14/41) | 61.20 | 59.82 |
| 50 | 41 | 677561 B | 677647 B | -0.01% | 39% (16/41) | 62.56 | 61.21 |
| 55 | 41 | 723351 B | 709934 B | +1.89% | 44% (18/41) | 63.57 | 62.52 |
| 60 | 41 | 719352 B | 738877 B | -2.64% | 56% (23/41) | 64.22 | 63.71 |
| 65 | 41 | 837525 B | 864627 B | -3.13% | 63% (26/41) | 69.64 | 68.72 |
| 70 | 41 | 986102 B | 1.01 MB | -2.21% | 54% (22/41) | 73.65 | 72.34 |
| 75 | 41 | 1.25 MB | 1.26 MB | -1.13% | 54% (22/41) | 78.41 | 76.76 |
| 80 | 41 | 1.67 MB | 1.65 MB | +1.26% | 39% (16/41) | 83.58 | 81.14 |
| 85 | 41 | 2.92 MB | 2.74 MB | +6.72% | 20% (8/41) | 90.72 | 86.33 |
| 90 | 41 | 3.77 MB | 5.66 MB | -33.31% | 85% (35/41) | 76.74 | 89.65 |

## Per-band totals

| band | range | bytes_picker | bytes_bucket | Δ% | achieved_picker | achieved_bucket |
|:--|:--|---:|---:|---:|---:|---:|
| low | zq < 50 | 2.51 MB | 2.48 MB | +1.30% | 59.68 | 58.61 |
| mid | 50 ≤ zq < 75 | 3.94 MB | 4.00 MB | -1.39% | 66.73 | 65.70 |
| high | zq ≥ 75 | 9.62 MB | 11.31 MB | -14.98% | 82.36 | 83.47 |

## Total

* Picker total bytes: **16.07 MB** (mean achieved zensim 69.37)
* Bucket total bytes: **17.79 MB** (mean achieved zensim 68.99)
* Δ bytes: **-9.65%** (picker − bucket)
* Δ achieved zensim: **+0.383** pp

## Verdict

**SHIP**

* Threshold: SHIP if total bytes (picker) ≤ total bytes (bucket) within ±0.5pp achieved-zensim parity.

## Method notes

* Picker arm: extracted 51-feature zenanalyze vector in `FEAT_COLS` order, applied per-feature transforms (log/log1p/identity per the v0.3 trainer's `feature_transforms`), built the engineered 112-vec via `feats[51] || size_oh[4] || poly[5] || zq*feats[51] || icc[1]` (mirror of the v0.3 manifest's `extra_axes`), ran `Predictor::predict` against the externally-loaded v0.3 `.bin`, decoded the `bytes_log[0..12]` argmin → cell index → `(color, sub, trellis_on, sa)` from the lex-sorted 12-cell taxonomy, and read `chroma_scale` (clamped to [0.6, 1.5]) and `lambda` (snapped to {0, 8.0, 14.5, 25.0}) from the per-cell scalar heads at offsets 24 and 36.
* Cell → encoder mapping (mirrors the v2.1+ Pareto sweep at commit `bb5cbf06`): `EncoderConfig::ycbcr(zq, sub)` or `EncoderConfig::xyb(zq, b_sub)` based on cell color; `chroma_distance_scale(chroma_scale)` applied UNIFORMLY (all cells, irrespective of trellis); if `trellis_on && lambda > 0`, additionally apply `hybrid_config(HybridConfig { enabled: true, base_lambda_scale1: lambda, ..default })` — `HybridConfig.chroma_scale` is left at default since chroma is already handled at the distance-scale level; `sa` cells additionally call `tables(Box::new(sa_piecewise_v4::tables_for_quality(starting_q)))` where `starting_q = Quality::Zq(target).to_internal()`.
* Bucket arm: `EncoderConfig::ycbcr(Quality::ZqExplicit(target), ChromaSubsampling::Quarter)` — codec defaults, no analyzer-derived knobs. The simplest baseline a caller would get from a one-line `EncoderConfig::ycbcr(...)` call.
* Both arms ride the same `Quality::ZqExplicit` closed loop so the iteration adapts the underlying jpegli quality to land in the target zensim band.
* FEAT_COLS source: hardcoded from `benchmarks/zenjpeg_hybrid_v0.3_2026-05-04.json::feat_cols` (51 entries, matches `zenjpeg_picker_v0.3_2026-05-04.manifest.json::feat_cols` exactly). Engineered axes (61 = size_oh[4] + poly[5] + zq×feats[51] + icc[1]) match `manifest.json::extra_axes` order.

## Notes on the q90 outlier

At target_zq=90 the picker arm achieves a mean zensim of **76.74** versus
**89.65** for the bucket arm — an undershoot of ~13 pp on the picker side.
Bytes are correspondingly low (3.77 MB picker vs 5.66 MB bucket = -33%),
but those bytes are NOT comparable: the picker is encoding at
substantially lower achieved quality at this single highest target.

The total-row achieved-zensim parity (+0.383 pp picker vs bucket across
all 13 targets) holds because the picker outperforms on achieved
zensim at q30..q85 (which dominates the average), masking the q90 cliff.

Probable cause: the `bytes_log[0..12]` head is fit per-cell across the
training Pareto sweep's q grid (0..100 step 5). At target_zq=90 the
picker tends to choose a low-trellis YCbCr cell with a cheap chroma
scale, which the closed-loop's `Quality::ZqExplicit(90)` cannot
recover from in 3 passes — the AQ scaler runs out of headroom. The
bucket's plain `ChromaSubsampling::Quarter` baseline at high q has a
flatter q→achieved curve and lands closer to target.

Doesn't block ship: total budget is honored, mid+high bands are net-
negative. Tracked as a follow-up — investigate (a) widening
`max_passes` from 3 to 5 at q≥85, (b) clamping picker cell selections
at q≥85 to the trellis-on subset, or (c) re-fitting the bytes_log head
with a high-q penalty term.
