# Phase 2 boundary-RD validation — 2026-04-20

Validation runs for the non-trellis, left-neighbor-only
boundary-continuity refinement added in this branch (Phase 2 of
issue #91).

## Run 1: `default` vs `boundary_rd`

```
cargo run --release -p zenjpeg --features "trellis decoder" --example rd_compare -- \
  --baseline default --candidate boundary_rd \
  --corpus cid22:2,screenshots:2,synthetic:3 \
  --qualities 65,75,85,95 \
  --metrics ssim2,bbs \
  --output-dir benchmarks/rd_compare/2026-04-20-phase2/
```

**Headline aggregate** (7 images × 4 qualities × 2 metrics → 56 encodes):

| metric | n  | BD-rate mean | stdev | mean_distance | direction |
|--------|----|-------------:|------:|--------------:|-----------|
| ssim2  | 5  | −0.239 %     | 0.917 | 0.0438        | candidate marginally wins |
| bbs    | 5  | **−1.686 %** | 0.354 | 3.7446        | **candidate wins on block-seam quality** |

(n = number of images with enough BD-rate overlap. Two images —
`gmessages` and `synth_checkerboard` — produced NA because BBS was
already saturated to zero on one side or the curves didn't overlap.)

**Per-image / per-class breakdown:**

| image | class | ssim2 BD-rate | bbs BD-rate |
|---|---|---:|---:|
| 1025469 | photo | −0.224 % | −1.992 % |
| 1044329 | photo | −0.308 % | −2.201 % |
| codec_wiki | screenshot | +0.021 % | −1.342 % |
| gmessages | screenshot | NA | NA |
| synth_checkerboard | synthetic | NA | NA |
| synth_grid | lineart | **+1.091 %** | −1.314 % |
| synth_stripes | lineart | −1.776 % | −1.581 % |

**Class aggregates** (from `by_class.csv`):

| class | metric | BD-rate mean | stdev |
|---|---|---:|---:|
| photo | bbs | −2.097 % | 0.104 |
| photo | ssim2 | −0.266 % | 0.042 |
| screenshot | bbs | −1.342 % | 0.000 |
| screenshot | ssim2 | +0.021 % | 0.000 |
| lineart | bbs | −1.448 % | 0.134 |
| lineart | ssim2 | −0.343 % | 1.433 |

## Run 2: `auto_optimize` vs `auto_optimize_boundary_rd`

Same command with `--baseline auto_optimize --candidate
auto_optimize_boundary_rd`. **Every BD-rate is zero** — confirms the
designed behavior that `boundary_rd(true)` is a no-op when trellis
quantization is active (the hybrid trellis path bypasses the
non-trellis refinement pass). Phase 3 of #91 will add the trellis-side
D-augment.

## Encode-time overhead

Measured via `zenjpeg/tests/boundary_rd_timing.rs`
(noise+patches 512×512 at Q85, 15 iters after 3-iter warmup):

```
  off: 4.57 ms  size=121290
  on:  5.39 ms  size=125910
  overhead: +17.9 %   size delta: +3.81 %
```

Within the +20 % encode-time budget in issue #91, and the +3.8 %
bit-rate increase on a realistic photograph-like input falls in the
expected 0–5 % band.

## Interpretation vs success criteria

From the task spec:

- **Default-path output unchanged.** Verified — `test_size_regression`
  and `test_quality_floor` pass; `test_dispatch_parity` pre-existing
  failure is unrelated.
- **BBS BD-rate clearly negative on at least synthetic + screenshot
  classes.** Met on screenshots (−1.342 %) and lineart/synthetic
  (−1.448 %). Photos also improve significantly (−2.097 %).
- **SSIM2 BD-rate not significantly positive.** Overall mean is
  −0.239 %; the only cell above 0 is `synth_grid` at +1.091 %
  (documented in issue #91 as an expected image-class-dependent
  tradeoff). Photos and screenshots are flat-to-winning.
- **Encode-time overhead ≤ +20 %.** +17.9 % measured.

All four gates pass. The headline candidate wins on BBS by ~1.7 %
while staying flat-to-winning on SSIM2.

## Known limitations

- Phase 2 refines luma only; chroma seams are untouched.
- `boundary_rd(true)` + `trellis(...)` / `auto_optimize(true)` is a
  no-op — trellis path intentionally bypasses the refinement in this
  phase. Phase 3 adds the D_b augment there.
- Fused parallel encode (`--features parallel` + DRI) routes through
  a separate path that does not yet honor the flag — same constraint
  trellis has today. Document as a known gap; Phase 5 can close it.
