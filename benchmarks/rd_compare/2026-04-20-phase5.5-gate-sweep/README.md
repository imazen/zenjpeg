# Phase 5.5 — per-block AQ-strength gate sweep (Run A)

**Date:** 2026-04-20  
**Base:** commit `6f1605c1` (Phase 5 tuned defaults + gate wiring)  
**Corpus:** `cid22:3,screenshots:2,synthetic:2` (7 images; 2 of 7 produce NaN
BD-rate — see "coverage" below)  
**Qualities:** 50, 65, 75, 85, 95  
**Metrics:** SSIM2, BBS  
**Baseline:** `default` (`EncoderConfig::ycbcr(q, Quarter)`)

## Hypothesis

Phase 5 shipped `boundary_rd(true)` with tuned defaults (α=1.0, threshold=0.05,
shrink=0.5, retries=2). Per-image breakdown from PR #99 showed photos
*regress* +0.15%-+0.20% on SSIM2 BD-rate even as BBS improves by 4-5%,
implying boundary-RD fires on blocks where the seam artifact is perceptually
masked (textured regions) and spends bits to no visible benefit. A per-block
AQ-strength gate should skip refinement on blocks where AQ says "this block
is well-masked" (high `aq_strength`) and preserve refinement on smooth blocks
where seams are visible.

**Gate direction question**: In jpegli AQ, higher `aq_strength` means
*more* masked/textured (encoder can quantize harder). So:

- `gate_max < 1.0` → skip refinement when `aq_strength > gate_max`
  (skip textured/well-masked blocks, keep smooth blocks). This is the
  hypothesis direction — if photos regress because refinement fires on
  invisible textured seams, this should help.
- `gate_min > 0.0` → skip refinement when `aq_strength < gate_min`
  (skip smooth blocks, keep textured). This is the *opposite* —
  included to verify we're reading the direction right.

## Results (per-candidate, 5 non-NaN images, mean across classes)

| candidate                     | SSIM2 BD-rate % | BBS BD-rate % |
|-------------------------------|----------------:|--------------:|
| `boundary_rd` (Phase 5, baseline for comparison) | +0.234 | **−3.150** |
| `boundary_rd_gate_max_05`     |          +0.198 |        −1.062 |
| `boundary_rd_gate_max_03`     |      **+0.089** |        −0.439 |
| `boundary_rd_gate_max_02`     |          +0.097 |        −0.153 |
| `boundary_rd_gate_max_01`     |      **+0.015** |    **−0.007** |
| `boundary_rd_gate_min_005`    |          +0.228 |        −3.149 |
| `boundary_rd_gate_min_010`    |          +0.222 |        −3.142 |

(Negative BD-rate = candidate better; positive = worse. Bold = closest to
zero for each column.)

### Reading the table

The `gate_max_*` row sweeps linearly between "no gate"
(`boundary_rd` at row 1) and "refinement effectively disabled"
(`gate_max_01` — 0.01 is below the AQ floor for typical blocks, so
virtually no blocks qualify for refinement). You see the expected
monotonic trade:

- As the gate tightens (0.5 → 0.01), BBS BD-rate shrinks toward zero
  — boundary-RD does less work, so BBS stops improving.
- SSIM2 BD-rate *also* shrinks toward zero, never dipping negative.
  The gate never makes SSIM2 better than the baseline — it only
  reduces how much worse it gets.

The `gate_min_*` variants are indistinguishable from `boundary_rd` (no
gate) at gate values 0.05 and 0.10. In jpegli AQ, almost no blocks have
strength < 0.10 — smooth regions have strength near 0 but jpegli-AQ
modulation produces mostly strengths in the 0.2-1.0 range on real images.
So the lower gate isn't exercised by this corpus. This is not a bug in
the gate, it's a property of the AQ distribution.

## Per-image (SSIM2 + BBS BD-rate %)

| image                  | class      | config                | SSIM2  | BBS    |
|------------------------|------------|-----------------------|--------|--------|
| 1025469                | photo      | boundary_rd           | +0.275 | −4.112 |
|                        |            | gate_max_03           | +0.136 | −1.336 |
|                        |            | gate_max_01           | +0.117 | +0.024 |
| 1044329                | photo      | boundary_rd           | +0.381 | −5.033 |
|                        |            | gate_max_03           | **−0.068** | −0.077 |
|                        |            | gate_max_01           | +0.004 | +0.008 |
| 1189261                | photo      | boundary_rd           | −0.091 | −3.384 |
|                        |            | gate_max_03           | +0.397 | −0.221 |
|                        |            | gate_max_01           | −0.050 | +0.011 |
| codec_wiki             | screenshot | boundary_rd           | +0.183 | −3.295 |
|                        |            | gate_max_03           | −0.020 | −0.560 |
|                        |            | gate_max_01           | +0.004 | −0.080 |
| synth_stripes          | lineart    | boundary_rd           | +0.420 | +0.076 |
|                        |            | gate_max_03           |  0.000 |  0.000 |

(The 7-image corpus includes `gmessages` and `synth_checkerboard`; both
produce NaN BD-rate because their size sweep is nearly flat — the
Bjontegaard area integral is undefined when the candidate and baseline
curves don't overlap on bit-rate. They contribute zero rows to the
aggregate.)

On `1044329` the `gate_max_03` setting actually produces a *small
SSIM2 win* (−0.068%) — the gate helped for that image. But on `1189261`
it makes SSIM2 significantly worse (+0.40% vs −0.09% baseline). The
effect is per-image, not per-class.

## Verdict (honest)

**Negative result: the gate does not close the photo SSIM2 Pareto gap
without proportionally losing the BBS Pareto win.**

Every `gate_max` setting that meaningfully reduces the SSIM2 cost
also reduces the BBS gain by a similar fraction. At `gate_max_01` both
are near zero (we've essentially disabled refinement) — SSIM2 regression
is gone but so is the entire technique's benefit.

In detail:

- `gate_max_05`: 66% of the Phase-5 BBS gain retained, no measurable
  SSIM2 improvement (still +0.20%).
- `gate_max_03`: 14% of BBS gain retained, SSIM2 +0.09% (marginal
  improvement over +0.23% baseline, mostly from one image).
- `gate_max_02`: 5% of BBS gain, SSIM2 +0.10%.
- `gate_max_01`: ~0% of BBS gain, SSIM2 +0.02% (essentially passthrough).

None of these points Pareto-dominates the ungated `boundary_rd` config.
They just let users dial back the technique's strength — which the
existing `boundary_rd_threshold`, `boundary_rd_shrink`, and
`boundary_rd_max_retries` knobs already do, more cleanly.

The `gate_min` direction is even less interesting: on this corpus it
has no effect because typical jpegli AQ distributions don't have many
blocks below 0.10.

## Why the hypothesis was wrong (best guess)

Boundary-RD's SSIM2 "regression" on photos is not about *which blocks*
refinement fires on — it's about what the retry algorithm *does*. Each
retry multiplies `aq_strength` by `shrink=0.5`, producing a quantize
with *less* AQ scaling. This adds bits in a way that lowers D_b but
doesn't necessarily lower reconstruction error in a human-visual sense.
On textured content, AQ's masking tables already put bits in perceptually
important places; forcing a second quantize with shrunken strength
essentially undoes some of AQ's adaptive allocation. SSIM2 is sensitive
to that even when block boundaries look better.

A per-block gate is the wrong knob because it decides *whether to run*
refinement — not *what refinement produces*. A better follow-up would
modify refinement to preserve AQ's frequency-weighted distortion
rather than collapsing it via `shrink`.

## Decision

**Ship no default change.** The new `boundary_rd_aq_gate_max` and
`boundary_rd_aq_gate_min` config knobs are retained — they're a cheap
addition, expose a real dial, and allow future optimizers to use them.
Their defaults (1.0 / 0.0) are no-op, so ungated `boundary_rd(true)`
is byte-identical to Phase 5 (hash-lock test verifies).

## Coverage

7 images loaded: 3 CID22 photos, 2 gb82-sc screenshots
(`codec_wiki`, `gmessages`), 2 synthetics (`synth_checkerboard`,
`synth_stripes`). Of these, `gmessages` and `synth_checkerboard`
produced NaN BD-rate at every candidate — the candidate and baseline
rate-distortion curves don't overlap enough for Bjontegaard integration.
This is the same corpus Phase 5 was tuned on, so the regression and
coverage are directly comparable to PR #99's numbers.

Run B (`2026-04-20-phase5.5-gb82sc/`) re-validates the best candidates
against all 9 valid gb82-sc screenshots.
