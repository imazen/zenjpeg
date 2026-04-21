# boundary_rd default config vs baseline — 2026-04-20

Headline evidence for the tuned best-we-know `BoundaryRdConfig::default()`
shipped in `EncoderConfig::boundary_rd(BoundaryRd::On(default))`
(PR #102, closes #91).

The config measured here is the Phase-5 tuned
`α=1.0, threshold=0.05, shrink=0.5, retries=2, above=false` — this is
what `BoundaryRdConfig::default()` returns.

The more conservative "photo-mild" preset
(`α=1.0, threshold=0.1, shrink=0.7, retries=1, above=false`) remains
available as a manually-constructed `BoundaryRdConfig` for callers who
want a photo-safer setting before an automatic per-class classifier
ships in issue #103.

## Command

```
cargo run --release -p zenjpeg --features "trellis decoder" \
  --example rd_compare -- \
    --baseline default --candidate boundary_rd \
    --corpus cid22:2,screenshots:2,synthetic:3 \
    --qualities 65,75,85,95 \
    --metrics ssim2,bbs
```

7 images × 4 qualities × 2 metrics = 56 encodes.

## Per-class BD-rate (from `_by_class.csv`)

Negative BD-rate = candidate wins (saves bits at equal quality, or
adds quality at equal bits).

| class       | metric | n |   BD-rate | direction |
|-------------|--------|--:|----------:|-----------|
| photo       | bbs    | 2 | **−4.48 %** | clean BBS win |
| photo       | ssim2  | 2 |   +0.18 %   | small SSIM2 cost (within #91 +0.5 % guardrail) |
| screenshot  | bbs    | 2 | **−2.97 %** | clean BBS win |
| screenshot  | ssim2  | 2 | **−0.25 %** | clean SSIM2 win |
| lineart     | bbs    | 2 | **−4.05 %** | clean BBS win |
| lineart     | ssim2  | 2 | **−3.23 %** | clean SSIM2 win |
| synthetic   | bbs    | 1 | NA          | BBS saturated to zero on one side |
| synthetic   | ssim2  | 1 | NA          | (paired with synthetic_checkerboard) |

## Honest framing

- **BBS Pareto win across every measurable class.** −2.97 % to −4.48 %
  BD-rate on block-boundary score, which is what this technique is
  designed to optimise.
- **SSIM2 Pareto win on screen-content + line-art / illustration
  content** (−0.25 % to −3.23 %).
- **Photo SSIM2 takes a small Pareto cost** (+0.18 % BD-rate). This is
  the sole regression and is well inside the +0.5 % guardrail from
  the issue #91 brief, but it is real.
- Callers who know the input is photographic can supply a gentler
  `BoundaryRdConfig` manually (e.g.
  `alpha=1.0, threshold=0.1, shrink=0.7, max_retries=1, above=false`)
  until automatic per-class selection ships (issue #103).
- Synthetic content (checkerboard / grid) saturates one side of the
  BBS metric and produces NA for BD-rate. This is expected: BBS is
  computed across blocks, and pure two-tone content has no
  cross-block continuity to score.

## Files

- `boundary_rd_auto_no_hint_2026-04-20_curves.csv` — raw RD curves
  (per-image, per-quality, per-metric) used to compute BD-rate
- `boundary_rd_auto_no_hint_2026-04-20_by_class.csv` — class-aggregate
  BD-rate (the table above is computed from this)
- `boundary_rd_auto_no_hint_2026-04-20_per_image.csv` — per-image
  BD-rate
