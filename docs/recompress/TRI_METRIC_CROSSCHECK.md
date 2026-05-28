# Tri-metric cross-check: zensim vs butteraugli vs cvvdp

The AQ-direction and Preserve-vs-Tuned conclusions were originally drawn
against the zensim Profile A metric alone. Because butteraugli and cvvdp
model contrast masking more explicitly than SSIMULACRA2/zensim, we re-ran
both experiments under all three metrics to see whether any
rate-distortion decision flips.

**Verdict: no decision flips. Both conclusions hold under all three
metrics.** There is one quantitative divergence worth recording (below).

## Method

`examples/tri_metric_gen.rs` emits the variant JPEGs for both experiments
(6 CID22 references × 3 source qualities {90,75,60} × 3 targets {70,60,50}),
then `zen-metrics batch` scores every variant under each metric with a
single shared decode per metric:

- `butteraugli_pnorm3_gpu` — lower = better (distance; libjxl 3-norm)
- `cvvdp` (`cvvdp_imazen_v0_0_1`) — higher = better (JOD 0–10, 10 = imperceptible)
- `zensim_gpu` — higher = better (0–100, 100 = identical)

Data: `benchmarks/tri_metric_crosscheck_6refs_2026-05-28.tsv` (378 rows).
GPU: RTX 5070, CUDA runtime.

## Experiment 1 — AQ block-selection direction

Deltas vs the no-AQ control, averaged over all `aqdir` cells. All variants
are at the same uniform quant scale, so only the AQ block-selection differs.

| variant | Δ size (+=smaller) | Δ zensim (+=better) | Δ butter pnorm3 (+=worse) | Δ cvvdp (+=better) |
|---|---|---|---|---|
| `flat_t48` (zero AC 48..64 in flat blocks) | **+0.0097** | −0.226 | +0.0115 | **+0.0002** |
| `busy_t48` (zero AC 48..64 in busy blocks) | +0.0001 | −0.007 | +0.0001 | −0.0000 |
| `busy_t32` (zero AC 32..64 in busy blocks) | +0.0022 | −0.367 | +0.0089 | −0.0002 |

Reading:

- **Flat-targeting saves real size** (+0.0097 ratio); busy-targeting saves
  essentially nothing (`busy_t48` +0.0001) because the quant-table scale-up
  already zeroed busy blocks' high frequencies — exactly as the
  generation-loss theory predicts (`docs/GENERATION_LOSS_THEORY.md` §3, §5).
- **The masking principle does not rescue busy-targeting under the
  masking-aware metrics.** butteraugli and cvvdp both rate `busy_t32` as a
  worse trade than `flat_t48` (more quality cost per byte saved). The HVS
  masking that justifies coarsening busy blocks when *encoding from pixels*
  has nothing to act on here: requantization removed the busy-block
  high-freq first.
- **cvvdp is the most permissive of flat AQ**: it scores `flat_t48`'s
  quality cost at **+0.0002 (slightly positive / imperceptible)**, versus
  zensim's −0.226 and butteraugli's +0.0115. So a cvvdp-driven product
  could run flat AQ more aggressively than zensim allows. This does not
  change the *direction* conclusion; it suggests the production AQ headroom
  gate is, if anything, conservative under cvvdp.

**Conclusion (unchanged): flat-targeting is the correct AQ direction for
recompression, confirmed under zensim, butteraugli, and cvvdp.**

## Experiment 2 — Preserve vs Tuned (generation loss)

54 cells (6 refs × 9 source/target combos). "Quality win" = strategy with
the better metric value at its own operating point.

| | avg size_ratio | zensim wins | butteraugli wins | cvvdp wins |
|---|---|---|---|---|
| `preserve_uniform` | 0.889 | **48 / 54** | **40 / 54** | **39 / 54** |
| `tuned` | 0.725 | 6 / 54 | 14 / 54 | 15 / 54 |

Reading:

- **Preserve delivers higher quality than Tuned on a strong majority of
  cells under every metric** (89 % / 74 % / 72 %). The generation-loss
  theory — Preserve skips the pixel round-trip and its broadband rounding —
  is confirmed *perceptually*, not just in coefficient algebra.
- **Tuned produces smaller files** (0.725 vs 0.889 avg ratio). Tuned's
  `auto_optimize` trellis requantizes harder, landing at a different
  operating point — lower quality, smaller size. This is the
  rate-distortion tradeoff the router arbitrates via the per-strategy
  calibration tables (which project both size and quality).
- **Divergence**: zensim rates Preserve's quality advantage highest
  (89 % win rate); butteraugli and cvvdp rate it real but smaller
  (74 %, 72 %). The masking-aware metrics are somewhat **more forgiving of
  Tuned's pixel-domain generation loss** than zensim is. Practical
  implication: a product tuned purely on zensim slightly over-weights
  Preserve's benefit relative to how cvvdp (the closest proxy to human JOD)
  would score it. The router's strategy ranking should ideally be
  calibrated on cvvdp or a multi-metric blend, not zensim alone — flagged
  for the v0.3 calibration cycle.

**Conclusion (unchanged): coefficient-domain Preserve incurs less
generation loss and lands higher-quality at a given target than the
pixel-domain Tuned path, confirmed under all three metrics. The size
tradeoff (Tuned smaller) is real and is what the calibrated router weighs.**

## Net findings

1. No decision flips: flat-targeting AQ and Preserve's gen-loss advantage
   both survive butteraugli and cvvdp.
2. cvvdp is more permissive of flat AQ than zensim — production AQ gate is
   conservative under cvvdp.
3. zensim slightly over-weights Preserve vs Tuned relative to cvvdp/butter.
   v0.3 should consider recalibrating the strategy router on cvvdp or a
   blend rather than zensim alone.
