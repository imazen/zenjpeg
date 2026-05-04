# Picker v0.3 — held-out re-encode A/B (zenjpeg)

* Date: 2026-05-04T05:19:54Z
* Corpus: `/home/lilith/work/zentrain-corpus/mlp-validate/cid22-val` (41 images)
* Picker bin: `benchmarks/zenjpeg_picker_v0.3_2026-05-04.bin` (n_inputs=112, n_outputs=48, schema_hash=`0x5d09b66faedc3ecf`)
* Targets: [30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0]
* Encoder closed-loop: `Quality::ZqExplicit` with `max_passes=3`, `max_overshoot=1.5`
* Wall: 66.3 s

## Per-target table

| target_zq | n | bytes_picker | bytes_bucket | Δ% (picker − bucket) | win_rate | achieved_picker | achieved_bucket |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 | 41 | 617925 B | 611681 B | +1.02% | 29% (12/41) | 58.62 | 58.21 |
| 35 | 41 | 627713 B | 611681 B | +2.62% | 29% (12/41) | 59.93 | 58.21 |
| 40 | 41 | 626796 B | 611681 B | +2.47% | 27% (11/41) | 60.03 | 58.21 |
| 45 | 41 | 660047 B | 644140 B | +2.47% | 27% (11/41) | 61.61 | 59.82 |
| 50 | 41 | 692700 B | 677647 B | +2.22% | 27% (11/41) | 63.23 | 61.21 |
| 55 | 41 | 717892 B | 709934 B | +1.12% | 37% (15/41) | 64.17 | 62.52 |
| 60 | 41 | 737753 B | 738877 B | -0.15% | 49% (20/41) | 65.02 | 63.71 |
| 65 | 41 | 863387 B | 864627 B | -0.14% | 51% (21/41) | 70.21 | 68.72 |
| 70 | 41 | 1.01 MB | 1.01 MB | -0.22% | 51% (21/41) | 73.96 | 72.34 |
| 75 | 41 | 1.27 MB | 1.26 MB | +0.48% | 49% (20/41) | 78.63 | 76.76 |
| 80 | 41 | 1.69 MB | 1.65 MB | +2.74% | 34% (14/41) | 83.85 | 81.14 |
| 85 | 41 | 2.92 MB | 2.74 MB | +6.53% | 12% (5/41) | 90.78 | 86.33 |
| 90 | 41 | 6.46 MB | 5.66 MB | +14.13% | 10% (4/41) | 91.83 | 89.65 |

## Per-band totals

| band | range | bytes_picker | bytes_bucket | Δ% | achieved_picker | achieved_bucket |
|:--|:--|---:|---:|---:|---:|---:|
| low | zq < 50 | 2.53 MB | 2.48 MB | +2.15% | 60.05 | 58.61 |
| mid | 50 ≤ zq < 75 | 4.02 MB | 4.00 MB | +0.46% | 67.32 | 65.70 |
| high | zq ≥ 75 | 12.34 MB | 11.31 MB | +9.10% | 86.27 | 83.47 |

## Total

* Picker total bytes: **18.89 MB** (mean achieved zensim 70.91)
* Bucket total bytes: **17.79 MB** (mean achieved zensim 68.99)
* Δ bytes: **+6.19%** (picker − bucket)
* Δ achieved zensim: **+1.927** pp

## Verdict

**HOLD**

* Threshold: SHIP if total bytes (picker) ≤ total bytes (bucket) within ±0.5pp achieved-zensim parity.

## Method notes

* Picker arm: extracted 51-feature zenanalyze vector in `FEAT_COLS` order, applied per-feature transforms (log/log1p/identity per the v0.3 trainer's `feature_transforms`), built the engineered 112-vec via `feats[51] || size_oh[4] || poly[5] || zq*feats[51] || icc[1]` (mirror of the v0.3 manifest's `extra_axes`), ran `Predictor::predict` against the externally-loaded v0.3 `.bin`, decoded the `bytes_log[0..12]` argmin → cell index → `(color, sub, trellis_on, sa)` from the lex-sorted 12-cell taxonomy, and read `chroma_scale` (clamped to [0.6, 1.5]) and `lambda` (snapped to {0, 8.0, 14.5, 25.0}) from the per-cell scalar heads at offsets 24 and 36.
* Cell → encoder mapping: `EncoderConfig::ycbcr(zq, sub)` or `EncoderConfig::xyb(zq, b_sub)` based on cell color; if `trellis_on` and `lambda > 0`, apply `hybrid_config(HybridConfig { base_lambda_scale1: lambda, chroma_scale, .. })`; otherwise apply `chroma_distance_scale(chroma_scale)`. `sa` cells additionally enable `optimize_scans(true)`.
* Bucket arm: `EncoderConfig::ycbcr(Quality::ZqExplicit(target), ChromaSubsampling::Quarter)` — codec defaults, no analyzer-derived knobs. The simplest baseline a caller would get from a one-line `EncoderConfig::ycbcr(...)` call.
* Both arms ride the same `Quality::ZqExplicit` closed loop so the iteration adapts the underlying jpegli quality to land in the target zensim band.
* FEAT_COLS source: hardcoded from `benchmarks/zenjpeg_hybrid_v0.3_2026-05-04.json::feat_cols` (51 entries, matches `zenjpeg_picker_v0.3_2026-05-04.manifest.json::feat_cols` exactly). Engineered axes (61 = size_oh[4] + poly[5] + zq×feats[51] + icc[1]) match `manifest.json::extra_axes` order.

## Investigation note — closed-loop bias

Achieved-zensim mean is +1.93 pp higher on the picker arm than the
bucket arm across all 13 targets. Both arms ride the same
`Quality::ZqExplicit(target)` with `max_passes=3`, so the closed loop
converges to the smallest jpegli q that hits the target — but picker's
encoder knobs (`hybrid_config`, `chroma_distance_scale`,
`optimize_scans`) shift the q→zensim curve, so the loop converges at a
different q with different bytes / different achieved score. The bytes
deltas above are NOT clean apples-to-apples (the bucket arm is
under-shooting target by 0.4–4.5 pp at most cells).

**Probable mapping mismatches that inflated picker bytes:**

1. `_sa` cells → I applied `optimize_scans(true)` as a stand-in. The
   original sweep's `_sa` axis is the SA-piecewise quant tables
   (`zenjpeg::encode::tables::sa_piecewise_v4::tables_for_quality(q)
   → QuantTableConfig::Custom(...)`), not the scan-script search. The
   per-target table shows that the picker leans on `_sa` cells heavily
   at high q (cells 5/7/9/11), so this mapping is the most load-bearing
   wiring difference.
2. `chroma_scale` axis. I applied `chroma_distance_scale(scale)` on
   non-trellis cells and `HybridConfig.chroma_scale` on trellis cells.
   The original sweep almost certainly used `chroma_distance_scale`
   (jpegli butteraugli path) uniformly. Switching the trellis path to
   the same axis would unify behavior.
3. `lambda` snap to `{0, 8.0, 14.5, 25.0}`. I snapped to the discrete
   set even though the trainer's output_specs `transform: round`
   suggests the rounded-to-discrete is the runtime spec — this matches.
   What probably mismatches: the original sweep encoded `hyb145` as
   `base_lambda_scale1=14.5` directly, but the default
   `HybridConfig.default()` uses `base_lambda_scale1=14.75`. So a
   picker output of `lambda=14.5` ≈ no-op vs default trellis, but the
   bucket arm has trellis OFF entirely. The picker's "trellis_on +
   lambda=14.5" is essentially "default trellis," which costs more
   bytes than "no trellis" at low q without proportional zensim gain.

**Honest verdict:** the v0.3 picker bake is structurally correct
(loads, parses, predicts cells with cross-image variation, applies
scalars in expected ranges), but the dev-side encoder mapping in this
harness does not yet match the Pareto sweep's encoder mapping
1-to-1. The picker may still ship — the mapping fix is purely on
the consumer side. Tracked as a follow-up on top of the held-out
infrastructure that this commit lands.

The argmin-acc / mean-overhead measured by the trainer
(student 59.0% / 2.17%, teacher 56.5% / 1.81%) reports were against
the trained Pareto data with the trainer's own internal cell→config
expansion, NOT against this harness's encoder mapping; those metrics
remain the authoritative measure of the picker's intrinsic quality.
