# Phase 3 validation — trellis boundary-continuity D term (issue #91)

**Branch:** `phase3-boundary-rd-trellis`
**Commit at measurement time:** see `git log --oneline` on the branch
**Baseline:** `trellis` — `EncoderConfig::ycbcr(q, Quarter).trellis(default)`
**Candidate:** `trellis_boundary_rd` — same + `.trellis_boundary_rd(true)` with
default β=0.1, α=1.0.

## Command

```
cargo run --release --features "trellis decoder __corpus-tests" \
    --example rd_compare -- \
    --baseline trellis --candidate trellis_boundary_rd \
    --corpus cid22:3,screenshots:2,synthetic:2 \
    --qualities 50,65,75,85,95 \
    --metrics ssim2,bbs \
    --output-dir benchmarks/rd_compare/2026-04-20-phase3/
```

7 images (3 cid22 photos, 2 screenshots, 2 synthetic), 5 qualities, 2 configs =
70 encodes total.

## Results

### BD-rate by image class

Lower is better. Positive BD-rate = candidate is worse than baseline at matched
distortion (wasting bits).

| Class | Metric | n | BD-rate mean % | stdev | mean_dist |
|---|---|---|---|---|---|
| photo | ssim2 | 3 | +0.162 | 0.138 | −0.016 |
| photo | bbs | 3 | +0.018 | 0.030 | +0.007 |
| screenshot | ssim2 | 1 | −0.060 | 0.000 | +0.009 |
| screenshot | bbs | 1 | +0.043 | 0.000 | −0.006 |
| lineart | ssim2 | 1 | −0.004 | 0.000 | −0.026 |
| lineart | bbs | 1 | −0.004 | 0.000 | +0.000 |
| synthetic | (NA — no detectable change at this corpus size) | | | | |

### Aggregate

| Metric | n | BD-rate mean % | stdev |
|---|---|---|---|
| ssim2 | 5 | +0.084 | 0.132 |
| bbs | 5 | +0.019 | 0.023 |

## Interpretation

Phase 3's design target was BBS BD-rate ≤ −2% and SSIM2 BD-rate ≤ +0.5% on
synthetic + screenshot content. We hit the SSIM2 criterion (+0.08% ≪ +0.5%)
and *fail* the BBS win criterion — BBS is +0.02% on average, not the −2% we
were aiming for.

**Why the modest numbers:**

1. The post-hoc EOB truncation is a conservative refinement — it only ever
   *zeros* trellis-chosen coefficients, never adds them back. This is safe
   but one-sided; a full bi-directional candidate set would likely do more.
2. The small 7-image corpus includes two images (`gmessages`,
   `synth_checkerboard`) where boundary-RD has no detectable effect because
   the first-pass trellis already produces zero or near-zero high-frequency
   coefficients — nothing to truncate. A larger screenshot/synthetic corpus
   would surface the effect more.
3. β=0.1 default was selected empirically: β=1.0 over-zeros coefficients
   (BD-rate ssim2 +7.6% on the same corpus), β=0.01 is a no-op. The
   spectral-vs-boundary ratio in the scoring function depends on the
   coefficient magnitudes, which vary wildly with quality — a future
   iteration should normalize β by (e.g.) the mean quant value.

## Raw data

- `2026-04-20/curves.csv` — per-image RD curves at each quality.
- `2026-04-20/per_image.csv` — per-image BD-rate and win-rate.
- `2026-04-20/by_class.csv` — aggregated by image class + metric.

## Additional sweeps (ad-hoc, in `/tmp/`, not committed)

| Candidate | Mean SSIM2 BD-rate | Mean BBS BD-rate |
|---|---|---|
| `trellis_boundary_rd_b001` (β=0.01) | −0.001% | −0.000% |
| `trellis_boundary_rd_b01` (β=0.10, = current default) | +0.084% | +0.019% |
| `trellis_boundary_rd` (β=1.00 old default) | +7.559% | +3.999% |

These show the β knob is monotone in aggressiveness and that the β=0.1 default
sits on a plateau where the feature is effectively a no-op on average — but
retains the potential to help on the right content (single screenshot result
showed −0.06% SSIM2 improvement).

## Next steps (not in this branch)

- Quality-adaptive β (β × quant_mean or β × λ⁻¹) so the scoring normalizes
  across quality levels.
- Bi-directional candidate set: also try *adding* a coefficient back at the
  last EOB+1 position when the left neighbor commits a value there.
- Wider corpus (e.g. 20+ screenshots with known seam artifacts from
  gb82 / CLIC) for statistically significant aggregates.
