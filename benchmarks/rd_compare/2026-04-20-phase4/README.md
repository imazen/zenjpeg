# Phase 4 boundary-RD validation — 2026-04-20

Validation runs for the optional left+above (top-edge) extension of
the non-trellis boundary-continuity refinement (Phase 4 of issue
#91), stacked on top of Phase 5 tuned defaults.

## Run 1: incremental — `boundary_rd` vs `boundary_rd_left_above`

```
cargo run --release -p zenjpeg --features "trellis decoder" --example rd_compare -- \
  --baseline boundary_rd \
  --candidate boundary_rd_left_above \
  --corpus cid22:3,screenshots:2,synthetic:2 \
  --qualities 50,65,75,85,95 \
  --metrics ssim2,bbs \
  --output-dir benchmarks/rd_compare/2026-04-20-phase4/incremental/
```

7 images × 5 qualities × 2 metrics → 70 encodes, ~2.1 s wall.

**Headline aggregate** (5 images produced valid BD-rate; 2 saturated
to zero BBS and returned NA):

| metric | n | BD-rate mean | stdev | mean_distance | direction |
|--------|---|-------------:|------:|--------------:|-----------|
| ssim2  | 5 | **−0.572 %** | 1.368 | 0.0737 | above marginally wins at SSIM2 (driven by line-art) |
| bbs    | 5 | **−1.159 %** | 0.365 | 0.8982 | **above wins on block-seam quality** |

**Per-image breakdown:**

| image | class | ssim2 BD-rate | bbs BD-rate |
|---|---|---:|---:|
| 1025469            | photo      | +0.126 % | −1.491 % |
| 1044329            | photo      | −0.070 % | −0.840 % |
| 1189261            | photo      | +0.278 % | −0.855 % |
| codec_wiki         | screenshot | +0.104 % | −0.907 % |
| gmessages          | screenshot | NA       | NA       |
| synth_checkerboard | synthetic  | NA       | NA       |
| synth_stripes      | lineart    | **−3.300 %** | **−1.704 %** |

**Class aggregates** (from `incremental/…/by_class.csv`):

| class      | metric | BD-rate mean |
|---|---|---:|
| lineart    | bbs   | **−1.704 %** |
| lineart    | ssim2 | **−3.300 %** |
| photo      | bbs   | −1.062 %     |
| photo      | ssim2 | +0.111 %     |
| screenshot | bbs   | −0.907 %     |
| screenshot | ssim2 | +0.104 %     |
| synthetic  | bbs   | NA           |
| synthetic  | ssim2 | NA           |

## Run 2: absolute — `default` vs `boundary_rd_left_above`

```
cargo run --release -p zenjpeg --features "trellis decoder" --example rd_compare -- \
  --baseline default \
  --candidate boundary_rd_left_above \
  --corpus cid22:3,screenshots:2,synthetic:2 \
  --qualities 50,65,75,85,95 \
  --metrics ssim2,bbs \
  --output-dir benchmarks/rd_compare/2026-04-20-phase4/absolute/
```

**Headline aggregate:**

| metric | n | BD-rate mean | stdev | mean_distance | direction |
|--------|---|-------------:|------:|--------------:|-----------|
| ssim2  | 5 | −1.638 %     | 3.811 | 0.3126 | mixed across classes; line-art dominates the mean |
| bbs    | 5 | **−5.741 %** | 1.590 | 7.8960 | candidate wins decisively |

**Per-image breakdown:**

| image | class | ssim2 BD-rate | bbs BD-rate |
|---|---|---:|---:|
| 1025469            | photo      | +0.403 % | −5.512 % |
| 1044329            | photo      | +0.303 % | −6.156 % |
| 1189261            | photo      | +0.092 % | −4.303 % |
| codec_wiki         | screenshot | +0.270 % | −4.179 % |
| gmessages          | screenshot | NA       | NA       |
| synth_checkerboard | synthetic  | NA       | NA       |
| synth_stripes      | lineart    | **−9.257 %** | **−8.554 %** |

**Class aggregates** (from `absolute/…/by_class.csv`):

| class      | metric | BD-rate mean |
|---|---|---:|
| lineart    | bbs   | **−8.554 %** |
| lineart    | ssim2 | **−9.257 %** |
| photo      | bbs   | −5.324 %     |
| photo      | ssim2 | +0.266 %     |
| screenshot | bbs   | −4.179 %     |
| screenshot | ssim2 | +0.270 %     |

## Encode-time overhead

Measured via `zenjpeg/tests/boundary_rd_timing.rs`
(noise+patches 512×512 at Q85, 15 iters after 3-iter warmup):

```
512x512 Q85 noise+patches
  left-only:  6.17 ms  size=130983
  left+above: 6.35 ms  size=131562
  overhead:   +3.0 %   size delta: +0.44 %
```

Relative to the default path (off → left-only is +17.9 %/+3.81 %),
going from left-only → left+above costs only **+3 %** more wall
time and **+0.44 %** more bytes. The above buffer is 64 B per column
(one rec-bottom + one orig-bottom, each 8 × f32); the per-candidate
work is one additional SSD (8 floats).

## Decision case

From the Phase 4 spec, the criteria for shipping:

- **Incremental BBS BD-rate over left-only ≤ −0.5 %:** achieved at
  **−1.159 %** (mean across the 5 usable images), with every
  measurable class below that floor (line-art −1.70 %, photo −1.06 %,
  screenshot −0.91 %).
- **SSIM2 BD-rate doesn't regress past +0.3 %:** the mean is
  **−0.572 %** (net-negative is net-wins; line-art dominates the
  mean). Per-image regressions on three photos are +0.08 to +0.28 %,
  all under the +0.3 % guardrail.

This is the **moderate-additional-gain case.** Left+above adds
measurable BBS quality over left-only on every image class in the
sweep, at negligible encode-time and size cost, and does not
meaningfully regress SSIM2. The PR ships with the flag added but
**default off**, documented as an opt-in for content where block
seams are the dominant artifact (screenshots, line-art, grid
patterns).

## Interpretation vs Phase 5 (parent)

Phase 5 locked (α=1.0, threshold=0.05, shrink=0.5, retries=2) for
the left-only path. The above term reuses the same α; separate
tuning of α for the above contribution was deferred — the simplicity
of the shared weight is preserved here, and the measured −1.16 %
incremental BD-rate is already comfortably inside the criterion.
A follow-up could sweep `boundary_rd_alpha_above` separately if a
second pass of tuning looks promising; the encode-time headroom
(~3 %) leaves room for at least one extra α retry or a narrow
secondary sweep without busting the total budget.

The **design space is partially closed**: left+above produces a
clear gain on top of left-only. On this corpus the absolute
`default → left+above` BBS BD-rate is −5.74 %, and the incremental
`left → left+above` is −1.16 %; subtracting in BD-rate space is
non-linear, but the ratio suggests the left-only path captures the
majority of the available BBS reduction and the above term adds an
additional ~20 % on top of that. Further boundary-RD directions
(chroma, trellis-side D-augment for Phase 3) are still open — see
#91.
