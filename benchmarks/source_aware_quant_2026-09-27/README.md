# Source-aware quantization pilot — 2026-09-27

Encoder experiment commit: `006d9783`.
Branch: `experiment/source-aware-quant-rd`.
Status: exploratory pilot; no production defaults changed.

Three CID22-512 validation images (`1025469`, `1044329`, `1189261`), each
encoded with zenjpeg's classic, jpegli, and moz source modes at Q80/4:2:0.
Each source was re-encoded through all three destination modes at
Q45/55/65/75/80/85/95, with six variants: generic, exact, hint/hist at strengths
1 and 4. Total: **1,134 measured output JPEGs**, **189 exact-control checks**,
and **972 per-curve BD-rate comparisons**. These are three independent images,
not 1,134 independent samples. External C encoders and holdout are still pending.

Every exact-table control matched the generic encoder's final DQT values and
decoded pixels. The release build and four integer quantization tests passed;
five report tests passed, including a joint IQA gate that rejects improvement
in one metric when the other worsens.

## Findings

Blind table adaptation is not ready to become a default. Results depend on the
destination mode and metric, and favorable averages conceal per-image losses.

Selected cumulative-reference results versus native `generic`, geometric mean
over the three images × three source modes for each destination:

| Destination / candidate | SSIM2 BD-rate | Valid cells | Butteraugli BD-rate | Valid cells |
|---|---:|---:|---:|---:|
| classic / hint_s1 | +0.34% | 9/9 | −2.91% | 7/9 |
| jpegli / hist_s1 | −0.50% | 9/9 | −1.47% | 8/9 |
| moz / hint_s1 | +0.39% | 9/9 | +1.36% | 8/9 |

Negative BD-rate means fewer bytes at equal quality. The jpegli/hist_s1
Butteraugli worst cell regressed **+7.43%** despite the favorable mean.
Across the full comparison set, 24/972 cells were sparse or lacked usable
overlap; baseline interval coverage fell as low as 28.9%. Inspect the detailed
CSV before interpreting a mean. The seven-quality grid needs densifying around
interesting crossings and plateaus before a policy decision.

The post-analysis joint gate selects the smallest saved candidate that uses
no more bytes and has neither worse SSIM2 nor worse Butteraugli than the generic
point, using the **decoded source JPEG** as reference:

| Destination | Mean byte saving over all 63 goals | Strict byte wins |
|---|---:|---:|
| classic | 0.201% | 11/63 |
| jpegli | 0.109% | 3/63 |
| moz | 0.013% | 3/63 |

These are raw per-budget savings, **not BD-rate**, and include the generic
fallback plus all sampled qualities in the candidate pool. They describe a
measured per-image search oracle and exclude its search/IQA cost. The separate
cumulative-reference gate is offline-only and needs the original; it must not
be presented as a runtime decision rule when only JPEG input is available.

## Next iteration

1. Keep generic output in every IQA candidate set and reject metric tradeoffs
   explicitly rather than hiding them in an aggregate score.
2. Add the coefficient-domain exact-table path and identity/repack control to
   distinguish table-alignment gains from pixel round-trip effects.
3. Investigate AQ/zero-bias independently for jpegli and source-reference
   trellis costs for moz. The current coefficient-MSE candidate score has not
   earned a default in either mode.
4. Add denser target grids, lower/higher source qualities, screenshots, odd
   dimensions, external encoder sources, and a separate holdout before promotion.

## Artifacts and reproduction

Full run (JPEGs, original PNGs, raw points, compiled source snapshots, lockfile,
27 SVG figures, and goal choices):

`/home/lilith/outputs/zenjpeg/source-aware/2026-09-27-pilot/`

Build/run log:
`/home/lilith/outputs/zenjpeg/source-aware/2026-09-27-pilot.log`

The checked-in `run.json` holds the exact arguments. `summary.md`, `bd_rate.csv`,
and `joint_choices.csv` are copied from the saved run. Reformat/rescore the
reports using `scripts/source_aware_report.py`; do not rerun encodes merely to
change output presentation. The earlier 72-point single-image smoke run is
preserved alongside the pilot as `2026-09-27-smoke`.

SHA-256 of full raw `points.jsonl`:
`c1410a01ae09e2a5f1b25af1bddda2c41dbe590187f3399bb19c33a1a47e3d7a`

SHA-256 of `bd_rate.csv`:
`9e83ad7e6e76f308808456bd3ff5e991f848a1a1e5864b1e050a26eeee2f74b9`

Metric versions and decoder settings are captured by `run.json` and the copied
`Cargo.lock`. Metrics are fast-ssim2 SSIMULACRA2 and the Rust Butteraugli crate's
default score, not Butteraugli p-norm3. Both use the same decoded candidate
pixels. BD-rate uses piecewise-linear log-rate integration over common
distortion only. No nonlinear score remapping or extrapolation is used.
