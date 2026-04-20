# Phase 5.5 — gb82-sc screenshot validation (Run B)

**Date:** 2026-04-20  
**Base:** commit `cbc1d237` (Phase 5.5 gate configs + multi-candidate harness)  
**Corpus:** all 9 valid gb82-sc screenshots (center-cropped to 512px max side)  
**Qualities:** 50, 65, 75, 85, 95  
**Metrics:** SSIM2, BBS  
**Baseline:** `default` (`EncoderConfig::ycbcr(q, Quarter)`)  
**Candidates:** `boundary_rd` (Phase 5 tuned defaults) + `boundary_rd_gate_max_03`
(best photo-SSIM2 tradeoff from Run A)

## Purpose

Run A used only `codec_wiki` + `gmessages` (2 screenshots) for the
screenshot class. This run validates the boundary-RD technique on a
real, diverse screenshot corpus (9 images) to see whether the SSIM2
regressions seen on photos also appear on screenshots — the content
class the technique was motivated by.

## Corpus coverage

Screenshots loaded (center-cropped to 512×512):

```
codec_wiki, gmessages, graph, gui, imac_dark, imac_g3,
imessage, terminal, windows
```

(`windows95.png` is skipped — unsupported PNG format in the `image`
crate. Now filtered in `rd_compare`'s screenshot loader.)

`gmessages` produced NaN BD-rate (flat rate-distortion curve doesn't
integrate) so all aggregates are over n=8.

## Per-image (BD-rate %, lower-is-better except noted)

| image      | config                  | SSIM2  | BBS    |
|------------|-------------------------|--------|--------|
| codec_wiki | boundary_rd             | +0.183 | −3.295 |
|            | boundary_rd_gate_max_03 | −0.020 | −0.560 |
| graph      | boundary_rd             | +0.499 | −3.018 |
|            | boundary_rd_gate_max_03 | +0.472 | −0.690 |
| gui        | boundary_rd             | +1.362 | −3.090 |
|            | boundary_rd_gate_max_03 | +0.552 | −0.385 |
| imac_dark  | boundary_rd             | +1.769 | −5.123 |
|            | boundary_rd_gate_max_03 | +0.282 | −0.177 |
| imac_g3    | boundary_rd             | **−0.555** | −5.443 |
|            | boundary_rd_gate_max_03 | +0.217 | −0.103 |
| imessage   | boundary_rd             | +0.426 | −4.543 |
|            | boundary_rd_gate_max_03 | +0.224 | −0.359 |
| terminal   | boundary_rd             | +0.796 | −4.336 |
|            | boundary_rd_gate_max_03 | +0.428 | −0.417 |
| windows    | boundary_rd             | **−1.527** | −0.793 |
|            | boundary_rd_gate_max_03 | +0.082 | −0.147 |

(Bold = Pareto win vs baseline on that metric.)

## Aggregates (mean across 8 non-NaN images)

|                                  | SSIM2  | BBS    |
|----------------------------------|-------:|-------:|
| `boundary_rd`            | +0.369 (σ 0.977) | **−3.705** (σ 1.398) |
| `boundary_rd_gate_max_03`| +0.280 (σ 0.183) | −0.355 (σ 0.193) |

## Findings

### The BBS win survives at gb82-sc scale

Ungated `boundary_rd` delivers consistent BBS BD-rate improvement (mean
**−3.7%**, all 8 non-NaN screenshots are BBS wins, 7 of 8 are >−3%).
This is the technique's purpose. **That win is real and reproducible
on a new, diverse corpus.**

### SSIM2 is image-dependent, not content-class-dependent

Unexpectedly, the SSIM2 picture on screenshots is not monotonic:

- 2 of 8 images *win on SSIM2* (`imac_g3` −0.56%, `windows` −1.53%)
- 2 of 8 are near-neutral (codec_wiki +0.18%, imessage +0.43%)
- 4 of 8 regress noticeably (`terminal` +0.80%, `graph` +0.50%,
  `gui` +1.36%, `imac_dark` +1.77%)

Mean is +0.37% SSIM2 BD-rate, but stdev is 0.98 — the class average
hides a bimodal distribution. The "screenshots all win, photos all
lose" narrative from Phase 5 isn't true at this corpus size.

### The gate tames the outliers but kills the BBS win

`gate_max_03` brings the SSIM2 stdev down from 0.98 to 0.18 — no more
`+1.77%` outliers. Mean SSIM2 BD-rate drops slightly (+0.37% → +0.28%),
and more images end up near-neutral.

But the BBS gain collapses from −3.7% to −0.35%. The gate effectively
disables refinement on most blocks, so only ~10% of the BBS benefit
remains.

### No Pareto winner

On this corpus neither candidate Pareto-dominates the other:

- Ungated `boundary_rd`: big BBS win, variable SSIM2 cost
- `gate_max_03`: tiny BBS win, tighter SSIM2 distribution

Per-image, there's no single image where `gate_max_03` Pareto-dominates
`boundary_rd` (better BBS *and* better SSIM2). The gate always costs
most of the BBS gain.

## Decision

Keep Phase 5 defaults as shipped (no gate). The new
`boundary_rd_aq_gate_{max,min}` knobs remain exposed for users/optimizers
who want per-image tuning, but no default change is warranted.

The SSIM2 variance finding is worth noting in the PR description: the
Phase 5 "class-dependent quality tradeoff" claim was based on a tiny
screenshot sample. On a real 8-image screenshot corpus, SSIM2 is **noisy
and image-dependent**, not cleanly better or worse than the
ungated baseline. The BBS gain, however, is reliably present at
roughly the same −3 to −5% magnitude Phase 5 reported.

## Raw data

- `sweep/curves.csv` — per-image, per-config, per-quality
- `sweep/per_image.csv` — per-image × per-metric BD-rate, mean-distance, win-rate
- `sweep/by_class.csv` — class-aggregated (only one class here, `screenshot`)
