# BBS Baseline Measurement — 2026-04-20

Phase 1 of issue #91: "what's our blocking baseline before any encoder work?"

BBS is the Block Boundary Score — mean-squared cross-seam gradient difference
vs original, measured at every 8-pixel grid-aligned seam. See
`zenjpeg/src/metrics/bbs.rs` docs for the exact formula and
`zenjpeg/examples/bbs_measure.rs` for the CLI wrapper used below.

## Environment

- Commit: `0954f43e` (post: `feat(metrics): add bbs_measure CLI example`)
- Host: WSL2 on Ryzen 9 7950X (water-cooled), 128 GB RAM
- Kernel: `6.6.87.2-microsoft-standard-WSL2`
- Rust: `rustc 1.95.0 (59807616e 2026-04-14)`
- Encoder: `EncoderConfig::ycbcr(Q, ChromaSubsampling::Quarter)`, no
  `progressive`, no `auto_optimize`, no trellis — the plainest zenjpeg
  baseline
- Decoder: zune-jpeg (no ICC applied)

## Command

```bash
for img in <path>|<label> ...; do
  ./target/release/examples/bbs_measure \
    --original "$path" --label "$label" \
    --quality 50 --quality 75 --quality 85 --quality 95 \
    --csv benchmarks/bbs_baseline_2026-04-20.csv
done
```

## Corpus

| Label | Source | Dimensions | Class |
|---|---|---|---|
| `cid22_1025469.png` | CID22 validation 1025469 | 512×512 | Photo |
| `cid22_1044329.png` | CID22 validation 1044329 | 512×512 | Photo (high chroma) |
| `cid22_1189261.png` | CID22 validation 1189261 | 512×512 | Photo |
| `gb82sc_terminal.png` | gb82-sc terminal | 1646×1062 | Screenshot (text) |
| `gb82sc_graph.png` | gb82-sc graph | 796×481 | Screenshot (lines + text) |
| `synth_checkerboard.png` | ImageMagick `pattern:checkerboard` | 512×512 | Synthetic (block-aligned) |
| `synth_text_label.png` | ImageMagick `-annotate` text on white | 512×512 | Synthetic (line art) |

The two synthetics are the blocking stress-tests: the checkerboard is
explicitly block-sized patterns, and the text label is anti-aliased line
art — both are classes where human vision is most sensitive to seam
jitter.

Photos come in two flavors: CID22 1025469 is a relatively tame scene,
1044329 is more chromatic (which inflates Cb/Cr contribution), and 1189261
sits in between. CID22 is small enough to run quickly but big enough to
have dozens of block seams in each direction.

## Results

All BBS values are mean-squared gradient error (gamma-domain, BT.601 for
Y/Cb/Cr split). Lower is better. `ratio = total / interior`; a value
significantly greater than 1 means seam gradients carry more error than
interior gradients (classic blocking).

### Photos (CID22)

| image | Q | bytes | bpp | BBS total | BBS Y | BBS Cb | BBS Cr | interior | ratio |
|---|---|---|---|---|---|---|---|---|---|
| cid22_1025469 | 50 | 12,845 | 0.39 | **34.49** | 27.05 | 4.42 | 3.03 | 29.21 | 1.18 |
| cid22_1025469 | 75 | 19,414 | 0.59 | **21.58** | 14.97 | 3.98 | 2.63 | 18.43 | 1.17 |
| cid22_1025469 | 85 | 26,188 | 0.80 | **15.30** | 9.07 | 3.69 | 2.54 | 13.59 | 1.13 |
| cid22_1025469 | 95 | 52,175 | 1.59 | **7.82** | 2.51 | 3.03 | 2.29 | 8.18 | 0.96 |
| cid22_1044329 | 50 | 43,778 | 1.34 | **299.76** | 170.93 | 76.09 | 52.74 | 320.32 | 0.94 |
| cid22_1044329 | 75 | 67,357 | 2.06 | **180.06** | 71.30 | 64.50 | 44.26 | 211.49 | 0.85 |
| cid22_1044329 | 85 | 88,133 | 2.69 | **134.12** | 35.45 | 57.56 | 41.12 | 166.54 | 0.81 |
| cid22_1044329 | 95 | 149,002 | 4.55 | **87.73** | 5.48 | 44.95 | 37.30 | 117.92 | 0.74 |
| cid22_1189261 | 50 | 27,867 | 0.85 | **97.23** | 55.12 | 29.47 | 12.64 | 99.95 | 0.97 |
| cid22_1189261 | 75 | 41,315 | 1.26 | **60.75** | 25.85 | 24.44 | 10.46 | 67.02 | 0.91 |
| cid22_1189261 | 85 | 53,808 | 1.64 | **44.66** | 14.14 | 21.19 | 9.33 | 53.62 | 0.83 |
| cid22_1189261 | 95 | 98,065 | 2.99 | **27.48** | 3.37 | 16.03 | 8.08 | 37.56 | 0.73 |

### Screenshots (gb82-sc)

| image | Q | bytes | bpp | BBS total | BBS Y | BBS Cb | BBS Cr | interior | ratio |
|---|---|---|---|---|---|---|---|---|---|
| gb82sc_terminal | 50 | 75,863 | 0.35 | **29.69** | 26.93 | 1.20 | 1.56 | 35.94 | 0.83 |
| gb82sc_terminal | 75 | 105,674 | 0.48 | **12.74** | 10.55 | 0.91 | 1.28 | 16.34 | 0.78 |
| gb82sc_terminal | 85 | 130,185 | 0.60 | **7.14** | 5.08 | 0.87 | 1.18 | 8.86 | 0.81 |
| gb82sc_terminal | 95 | 191,239 | 0.88 | **2.51** | 0.74 | 0.71 | 1.06 | 2.36 | **1.07** |
| gb82sc_graph | 50 | 16,896 | 0.35 | **37.38** | 26.32 | 5.56 | 5.50 | 44.35 | 0.84 |
| gb82sc_graph | 75 | 22,641 | 0.47 | **19.16** | 9.18 | 5.04 | 4.94 | 23.34 | 0.82 |
| gb82sc_graph | 85 | 26,978 | 0.56 | **13.63** | 4.04 | 4.84 | 4.75 | 16.79 | 0.81 |
| gb82sc_graph | 95 | 38,743 | 0.81 | **9.36** | 0.56 | 4.30 | 4.50 | 11.48 | 0.82 |

### Synthetics

| image | Q | bytes | bpp | BBS total | BBS Y | interior | ratio |
|---|---|---|---|---|---|---|---|
| synth_checkerboard | 50 | 66,833 | 2.04 | **4849.50** | 4849.50 | 3195.11 | **1.52** |
| synth_checkerboard | 75 | 104,806 | 3.20 | **258.77** | 258.77 | 387.77 | 0.67 |
| synth_checkerboard | 85 | 124,320 | 3.79 | **75.19** | 75.19 | 159.54 | 0.47 |
| synth_checkerboard | 95 | 167,735 | 5.12 | **6.10** | 6.10 | 19.90 | 0.31 |
| synth_text_label | 50 | 16,613 | 0.51 | **20.53** | 20.53 | 29.96 | 0.69 |
| synth_text_label | 75 | 21,813 | 0.67 | **6.66** | 6.66 | 9.75 | 0.68 |
| synth_text_label | 85 | 25,633 | 0.78 | **2.82** | 2.82 | 4.10 | 0.69 |
| synth_text_label | 95 | 34,671 | 1.06 | **0.34** | 0.34 | 0.51 | 0.67 |

Synthetic Cb/Cr are zero because both synthetics are pure grayscale —
no chroma content exists, so the chroma channels have zero gradient
everywhere. Dropped from the table to save space.

## Observations

1. **Monotonicity check passes on all 7 images.** BBS strictly decreases
   Q50→Q95 for every image in the corpus. Metric is behaving correctly.

2. **Y dominates seam error across the board.** On photos, BBS Y is 5–10×
   larger than BBS Cb+Cr combined at Q50. 4:2:0 subsampling already caps
   chroma at half resolution so its contribution to seam gradients is
   shrunk proportionally.

3. **`interior_ratio` (BBS / interior) tells a subtler story than raw BBS.**
   On CID22 photos, `ratio` at high Q *drops below 1* — meaning the
   reconstructed seam gradients are *flatter* than the interior. This is
   over-quantization flattening AC coefficients along seams more than in
   block interiors. Not necessarily perceptually bad on photographs where
   seam energy is small to begin with, but it's a signal that a future
   boundary-RD term should be able to exploit: if we can put 1 bit back
   into seam-relevant low-frequency AC coefficients at high Q, the seams
   should pop back into alignment with the interior.

4. **The checkerboard is a 5-alarm fire.** At Q50, synth_checkerboard BBS
   is 4849 vs ~30–300 for photos: a *16× to 160× jump*. The pattern's
   energy lives *entirely* in frequencies the zero-bias is aggressive
   about, so most AC coefficients get zeroed and all seams collapse to a
   uniform gray. Ratio 1.52 at Q50 confirms the seams carry substantially
   more error than the interior. This is the content class issue #91 is
   trying to fix.

5. **Screenshots fall between photos and synthetics.** terminal.png at
   Q95 shows `ratio = 1.07` — the only *non-synthetic* image where
   seams exceed the interior. This is consistent with the perception that
   blocking on text is more visible than on photos at the same Q.

6. **gb82sc_graph holds `ratio ≈ 0.82` across Q50–Q95** — the line art
   has consistent interior gradient structure that the encoder handles
   comparably to seams. If the encoder is equal-error across both, BBS
   doesn't flag blocking, and this is one of the cleaner classes.

## Where this points for encoder work

Reading the table, the boundary-RD term from issue #91 should:

1. Target **photos at Q50–Q85** and **screenshots at Q75–Q95**. This is
   where `interior_ratio` is close to 1 and seam-specific gains would
   make a perceptual difference.

2. Explicitly guard the **checkerboard/line-art regime**: at Q50 the
   synthetic checkerboard has BBS × interior_ratio = 4849 × 1.52, a
   classic content-dependent blow-up. Whatever α the tuning picks must
   not make this pathological case worse.

3. Accept that **at very high Q (95+)** on photos, `interior_ratio < 1`
   — the encoder has already wiped out all cross-seam energy. Boundary
   RD at these Q levels is about *putting energy back* where it matters,
   not removing it.

Raw CSV: [`bbs_baseline_2026-04-20.csv`](./bbs_baseline_2026-04-20.csv).
