# boundary_rd aggressive left+above preset vs default — 2026-04-20

Evidence for the aggressive left+above config characterised in this
run:

```rust
BoundaryRd::On(BoundaryRdConfig {
    alpha: 1.0,
    threshold: 0.05,
    shrink: 0.5,
    max_retries: 2,
    above: true,
})
```

This is the same Phase-5 tuned config as the default
(`BoundaryRdConfig::default()` measured in
`boundary_rd_auto_no_hint_2026-04-20.md`) except the above-neighbor
(top-edge) D_b term is enabled. Callers wanting this setting construct
it manually. Automatic per-image-class selection of this versus the
gentler default is deferred to issue #103 (PR #102 ships manual
config only).

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
  the same direction as the default config but with measurably bigger
  wins on the classes this config is targeted at (line-art most of
  all).
- **SSIM2 Pareto win on line-art** (−9.26 %) — the designed-for case.
- **SSIM2 small Pareto cost on screen content + photo** (+0.27 %
  each). Within the #91 +0.5 % guardrail, but real. Callers wanting
  to avoid even this small SSIM2 cost on unknown content should stick
  with `BoundaryRdConfig::default()` (the `above=false` variant).
- **Synthetic content** saturates the BBS metric and produces NA for
  BD-rate (same shape as the default-config sweep — pure two-tone
  content has no cross-block continuity to score).

## Files

- `boundary_rd_screenshot_preset_2026-04-20_curves.csv` — raw RD
  curves used to compute BD-rate
- `boundary_rd_screenshot_preset_2026-04-20_by_class.csv` —
  class-aggregate BD-rate (the table above is computed from this)
- `boundary_rd_screenshot_preset_2026-04-20_per_image.csv` —
  per-image BD-rate
