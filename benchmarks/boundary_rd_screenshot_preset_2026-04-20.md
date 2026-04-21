# boundary_rd ScreenContent / Illustration preset vs default — 2026-04-20

Headline evidence for the aggressive left+above preset that
`EncoderConfig::boundary_rd(BoundaryRd::Auto)` resolves to when the
caller passes `ImageContentType::ScreenContent` or
`ImageContentType::Illustration` as a hint (PR #102, closes #91).

The preset under test is `phase5_left_above`:
`α=1.0, threshold=0.05, shrink=0.5, retries=2, above=true` —
identical to the photo-flat / Auto-no-hint preset measured in
`boundary_rd_auto_no_hint_2026-04-20.md` except the above-neighbor
(top-edge) D_b term is enabled.

## Command

```
cargo run --release -p zenjpeg --features "trellis decoder" \
  --example rd_compare -- \
    --baseline default --candidate boundary_rd_left_above \
    --corpus cid22:3,screenshots:2,synthetic:2 \
    --qualities 50,65,75,85,95 \
    --metrics ssim2,bbs
```

7 images × 5 qualities × 2 metrics = 70 encodes.

## Per-class BD-rate (from `_by_class.csv`)

Negative BD-rate = candidate wins.

| class       | metric | n |   BD-rate   | direction |
|-------------|--------|--:|------------:|-----------|
| photo       | bbs    | 3 | **−5.32 %** | clean BBS win |
| photo       | ssim2  | 3 |   +0.27 %   | small SSIM2 cost (within #91 +0.5 % guardrail) |
| screenshot  | bbs    | 2 | **−4.18 %** | clean BBS win |
| screenshot  | ssim2  | 2 |   +0.27 %   | small SSIM2 cost |
| lineart     | bbs    | 1 | **−8.55 %** | strongest BBS win in the sweep |
| lineart     | ssim2  | 1 | **−9.26 %** | strongest SSIM2 win in the sweep |
| synthetic   | bbs    | 1 | NA          | BBS saturated to zero on one side |
| synthetic   | ssim2  | 1 | NA          | (paired with synthetic_checkerboard) |

## Honest framing

- **BBS Pareto win on every measurable class** (−4.18 % to −8.55 %),
  the same direction as the no-hint preset but with measurably bigger
  wins on the classes this preset is targeted at (line-art most of
  all).
- **SSIM2 Pareto win on line-art** (−9.26 %) — the designed-for case.
- **SSIM2 small Pareto cost on screen content + photo** (+0.27 %
  each). Within the #91 +0.5 % guardrail, but real. This is why the
  Auto-no-hint preset is the more conservative `photo_mild()` and the
  ScreenContent / Illustration hint is what unlocks `phase5_left_above`
  — the caller is opting in to the SSIM2 tradeoff in exchange for
  the bigger BBS win.
- **Synthetic content** saturates the BBS metric and produces NA for
  BD-rate (same shape as the no-hint sweep — pure two-tone content
  has no cross-block continuity to score).

## Files

- `boundary_rd_screenshot_preset_2026-04-20_curves.csv` — raw RD
  curves used to compute BD-rate
- `boundary_rd_screenshot_preset_2026-04-20_by_class.csv` —
  class-aggregate BD-rate (the table above is computed from this)
- `boundary_rd_screenshot_preset_2026-04-20_per_image.csv` —
  per-image BD-rate
