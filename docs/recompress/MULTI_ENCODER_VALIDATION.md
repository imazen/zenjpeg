# Multi-encoder validation: libjpeg-turbo, mozjpeg, jpegli

The v0.2 calibration was fit entirely on jpegli-family sources (the
`cumulative-sweep` harness synthesizes sources with zenjpeg's
`ApproxJpegli` encoder). This document records the validation on **real
libjpeg-turbo and mozjpeg** sources and what it revealed.

## How real sources are generated

`scripts/gen_multiencoder_sources.sh <originals> <out> [qstep]` encodes
each CID22 PNG through three real encoders at a granular q step:

- **libjpeg-turbo** — host `/usr/bin/cjpeg` (PPM in).
- **mozjpeg 4.1.5** — extracted from the `all-the-images` docker
  `mozjpeg` build stage (`docker build --target mozjpeg -t ati-mozjpeg .`),
  pulled to host via a tar stream (`docker run ati-mozjpeg tar -C
  /opt/mozjpeg-4.1.5 -cf - . | tar -x`), run with
  `LD_LIBRARY_PATH=…/lib`. (WSL docker bind-mounts of `/tmp` and
  `docker cp` were both unreliable here; the tar stream is robust.)
- **jpegli** — host `/usr/local/bin/cjpegli` (PNG in).

Sources are named `<refstem>__<encoder>__q<Q>.jpg`. The new
`zjr-calibrate recompress-sweep --sources DIR --originals DIR` subcommand
matches each source to its original by stem prefix, recompresses across a
target grid, and scores output vs the **original** (cumulative quality).

## Detection — PASS

zenjpeg's `detect::probe` correctly identifies all three:

| encoder | detected as | quality scale |
|---|---|---|
| libjpeg-turbo | `LibjpegTurbo` | IjgQuality |
| mozjpeg | `Mozjpeg` | MozjpegQuality |
| jpegli | `CjpegliYcbcr` | ButteraugliDistance |

## Calibration transfer — the finding

Recompress-sweep, 6 CID22 refs × 16 source-q (step 5) × 6 targets,
under-target delivery rate (output landed >2 zensim-A below requested
target):

| source encoder | under-target (v0.2, jpegli-fit) | under-target (after per-encoder source-quality anchors) | size regressions |
|---|---|---|---|
| jpegli (fit on) | 4 % | 4 % | 0 |
| libjpeg-turbo | 76 % | 70 % | 0 |
| mozjpeg | 75 % | 54 % | 0 |

Two distinct facts:

1. **The no-size-regression invariant is encoder-independent** — 0
   regressions across all encoders, exactly as the generation-loss
   theory predicts (it's coefficient algebra, not perception).
2. **Quality calibration does NOT transfer from jpegli to
   turbo/mozjpeg.** On those sources the router massively under-delivers.

## Root cause (two layers)

**Layer 1 — source-quality estimate (fixed).** The v0.2
`encoder_quality_to_estimated_zensim_a` over-estimated turbo/mozjpeg
source quality by ~10 zensim-A (it predicted turbo q90 → 94; measured is
84). The router therefore believed the source was better than it was and
recompressed too aggressively. **Fix shipped**: replaced the shared
anchor + additive-shift model with *measured* per-encoder IJG-Q →
zensim-A curves (`benchmarks/source_quality_vs_original_2026-05-28.tsv`,
median over 6 refs). This dropped mozjpeg under-target 75 % → 54 %.

**Layer 2 — achieved-quality projection table (NOT yet fixed).** The
`CELL_MEDIAN_ZENSIM_A_420` table that projects *achieved* recompressed
quality was also fit on jpegli sources only. A turbo source at a given
estimated quality recompresses to a *lower* achieved quality than a
jpegli source at the same estimate, because turbo's standard 2-table IJG
quantization requantizes differently from jpegli's 3-table
distance-optimized coefficients. This is why turbo barely improved
(76 % → 70 %): its source estimate got fixed, but the achieved-quality
projection is still jpegli-shaped.

## The fix (shipped 2026-05-28) — per-encoder achieved-quality tables

Built per-encoder, per-strategy achieved-quality + size-ratio tables
(`src/calibration/per_encoder.rs`) from forced-strategy recompress-sweeps
on **pinned encoders** (libjpeg-turbo 3.1.0 + mozjpeg 4.1.5, both
extracted from the all-the-images docker stages via tar-stream; jpegli
host `cjpegli`). Re-encode uses the RD-best **HybridMaxCompression**
params (see below). The router projects each strategy from its
per-encoder table and picks the one that lands at-or-above target with
smallest size.

What the tables capture:
- **Preserve craters** on aggressive non-jpegli targets (turbo src90→t50
  achieves ~13, not 50). The router sees the low projection and rejects
  Preserve there.
- **Tuned never craters** (source-encoder-independent re-encode from
  pixels) — it's the escape hatch. Routes aggressive turbo/mozjpeg there.
- For turbo/mozjpeg, **Preserve never wins**: aggressive → craters;
  gentle → dominated by Lossless (which is smaller *and* keeps full
  quality). jpegli keeps Preserve (its distance-optimized coefficients
  requantize gracefully) on the richer 15-image `data::lookup_420/444`.

Result — under-target delivery (output > 2 zensim-A below target):

| source encoder | v0.2 (jpegli-fit) | + source anchors | **+ per-encoder tables** | size regressions |
|---|---|---|---|---|
| jpegli | 4 % | 4 % | **4 %** | 0 |
| libjpeg-turbo 3.1.0 | 76 % | 70 % | **12 %** | 0 |
| mozjpeg 4.1.5 | 75 % | 54 % | **8 %** | 0 |

## Benchmark vs naive deblock

`recompress-sweep --naive-deblock` is the "just decode, deblock, and
re-save at quality Q" baseline (no router, no NoOp/Lossless/Preserve, no
per-encoder calibration). Smart router vs naive, same cells:

| encoder | mode | under-target | size regressions |
|---|---|---|---|
| turbo | **smart** | **12 %** | **0** |
| turbo | naive | 60 % | 146 |
| mozjpeg | **smart** | **8 %** | **0** |
| mozjpeg | naive | 63 % | 254 |
| jpegli | **smart** | **4 %** | **0** |
| jpegli | naive | 53 % | 146 |

Naive deblock under-delivers on the *majority* of cells AND inflates the
file on 146–254 cells (re-encoding a low-quality source at a high target,
or deblock adding spurious detail). The router's NoOp / Lossless /
per-encoder strategy selection is exactly what buys the 0 regressions and
the 4-12 % under-target floor. Artifacts:
`benchmarks/naive_deblock_{turbo,mozjpeg,jpegli}_2026-05-28.tsv` vs
`benchmarks/recompress_sweep_{…}_perenc_2026-05-28.tsv`.

## Encoder params (HybridMaxCompression, not auto_optimize)

The re-encode path (Tuned/Deblock) uses `HybridMaxCompression`, not
`auto_optimize`. The 6-ref encoder RD ablation
(`benchmarks/zenjpeg_param_rd_6refs_2026-05-28.tsv`) measured bytes at
matched zensim vs auto_optimize:

| param set | zensim 60 | zensim 70 | zensim 80 |
|---|---|---|---|
| auto_optimize | 1.000 | 1.000 | 1.000 |
| **HybridMaxCompression** | **0.960** | **0.982** | **0.990** |
| XYB | — | 0.984 | 0.939 |
| jpegli_prog / mozjpeg_max / prog_search | > 1.0 (worse) | | |

HybridMaxCompression is 1-4 % smaller at every quality and stays pure
YCbCr (broadly decodable). XYB wins more at high quality (6 % at zensim
80) but changes color handling / decoder compatibility — reserved for a
future modern-decoder mode.

## Remaining (v0.3)

- **Wider corpus**: the per-encoder tables are n≈6 per cell. Refit on
  ≥40 refs per cell to tighten the noisy cells and the confidence-shift
  residuals.
- **More encoders**: Photoshop "Save for Web", GIMP, Pillow, Apple
  encoders. Each gets a per-encoder table once swept.
- **Subsampling**: the per-encoder tables are 4:2:0. 4:2:2 / 4:4:0 fall
  through to the analytical estimate.
- **XYB modern-decoder mode**: 6 % smaller at high quality, gated behind
  a compatibility flag.

## Artifacts

- `scripts/gen_multiencoder_sources.sh` — source corpus generator
  (pinned turbo 3.1.0 + mozjpeg 4.1.5 + jpegli).
- `benchmarks/source_quality_vs_original_2026-05-28.tsv` — measured
  per-encoder IJG-Q → zensim-A source-quality anchors.
- `benchmarks/recompress_sweep_{turbo,mozjpeg,jpegli}_perenc_2026-05-28.tsv`
  — router-chosen sweep with per-encoder tables (the shipped result).
- `benchmarks/naive_deblock_{turbo,mozjpeg,jpegli}_2026-05-28.tsv` —
  naive-deblock baseline for the head-to-head.
- `benchmarks/zenjpeg_param_rd_6refs_2026-05-28.tsv` — encoder param RD
  ablation (HybridMaxCompression vs auto_optimize vs others).
- `src/calibration/per_encoder.rs` — the baked per-encoder tables.
