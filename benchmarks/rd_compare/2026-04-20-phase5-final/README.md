# Phase 5 final validation — default vs boundary_rd (new defaults)

Re-run of PR #97's headline Phase 2 validation command, unchanged
except for the Phase 5 tuned defaults shipping in this branch
(`α=1.0, threshold=0.05, shrink=0.5, retries=2`).

## Command

```
cargo run --release -p zenjpeg --features "trellis decoder" \
  --example rd_compare -- \
  --baseline default --candidate boundary_rd \
  --corpus cid22:2,screenshots:2,synthetic:3 \
  --qualities 65,75,85,95 \
  --metrics ssim2,bbs \
  --output-dir benchmarks/rd_compare/2026-04-20-phase5-final/
```

## Headline: Phase 2 defaults vs Phase 5 defaults

Same 7 images × 4 qualities × 2 metrics = 56 encodes. The only
difference from `benchmarks/rd_compare/2026-04-20-phase2/` is the four
numeric defaults.

| metric | Phase 2 (PR #97) | Phase 5 (this PR) |  delta   |
|--------|-----------------:|------------------:|---------:|
| ssim2  | −0.239 %         | **−1.272 %**      | −1.03 pt |
| bbs    | −1.686 %         | **−4.005 %**      | −2.32 pt |

BBS BD-rate is **2.4× better** than the Phase 2 guess defaults, and
SSIM2 BD-rate is 5× better (both still net-negative — candidate wins).

## Per-class breakdown

From `by_class.csv`:

| class       | metric | Phase 2   | Phase 5    | delta   |
|-------------|--------|----------:|-----------:|--------:|
| photo       | bbs    | −2.097 %  | **−4.479 %** | −2.38 pt |
| photo       | ssim2  | −0.266 %  | +0.179 %     | +0.44 pt |
| screenshot  | bbs    | −1.342 %  | **−2.975 %** | −1.63 pt |
| screenshot  | ssim2  | +0.021 %  | −0.252 %     | −0.27 pt |
| lineart     | bbs    | −1.448 %  | **−4.047 %** | −2.60 pt |
| lineart     | ssim2  | −0.343 %  | −3.233 %     | −2.89 pt |

Every class sees a 2.0×–2.8× BBS improvement. The only SSIM2
regression is photo at +0.18 % (up from −0.27 % in Phase 2) — still
inside the +0.5 % SSIM2 guardrail from the #91 task spec. Screenshots
went from neutral SSIM2 (+0.02 %) to a clean −0.25 % win. Lineart gets
the biggest boost: BBS −4.05 % and SSIM2 −3.23 % simultaneously.

## Per-image

| image               | class      | bbs Phase 5 | ssim2 Phase 5 |
|---------------------|------------|------------:|--------------:|
| 1025469             | photo      | −3.916 %    | +0.203 %      |
| 1044329             | photo      | −5.041 %    | +0.154 %      |
| codec_wiki          | screenshot | −2.975 %    | −0.252 %      |
| gmessages           | screenshot | NA          | NA            |
| synth_checkerboard  | synthetic  | NA          | NA            |
| synth_grid          | lineart    | −2.663 %    | −1.052 %      |
| synth_stripes       | lineart    | −5.431 %    | −5.414 %      |

`gmessages` and `synth_checkerboard` produce NA because BBS saturates
to zero on one side of the comparison (matching Phase 2). The
remaining 5 images are consistently negative on BBS.

## Honest assessment

- **Technique gain is real and measurable.** Going from −1.69 % to
  −4.00 % BBS BD-rate on the same validation corpus is a meaningful
  codec improvement, not noise.
- **Still opt-in, still not a default-on knob.** The +30 % to +40 %
  encode-time overhead (measured in `boundary_rd_timing.rs`) and the
  +0.18 % photo SSIM2 regression keep this behind
  `EncoderConfig::boundary_rd(true)`.
- **My read of the stack so far:** Phase 2 + Phase 5 together give the
  most bang for the buck on screenshot and lineart content. Phase 3
  (trellis variant) hit essentially zero improvement on the same
  corpus and is orthogonal. Phase 4 (above-neighbor) would further
  extend Phase 2 but doubles the complexity (needs iMCU-row top-edge
  buffering) for a likely fractional additional gain.
- **Per-class opt-out:** If a workflow is photo-only and wants the
  tightest SSIM2 guarantee, it can explicitly set
  `.boundary_rd_max_retries(1)` to get −2.39 % BBS, +0.02 % SSIM2
  (Phase-2-like balance but with the other two tuned knobs).
