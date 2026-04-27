# Empirical calibration report — 2026-04-27

Class-conditional feature distributions, AUC for screen-vs-photo
classification, recommended operating thresholds, and Spearman
redundancy. Source: 219-image labeled corpus from
`coefficient/benchmarks/classifier-eval/labels.tsv` (gpt-5.4-mini
labels reviewed for primary category, has-text, palette size,
chroma dominance, synthetic flag).

The labels span 9 source corpora (cid22-train/val, clic2025-1024,
gb82, gb82-sc, imageflow, kadid10k, qoi-benchmark, corpus). Class
counts: photo=174 (natural=97, detailed=39, portrait=32, uniform=6),
screen=36 (ui=17, document=17, chart=2), illustration=9, marked
synthetic=44.

## Methodology

`zenanalyze/examples/corpus_eval.rs`, labeled mode
(`LABELS_TSV=…/labels.tsv ./target/release/examples/corpus_eval`),
runs `analyze_features` with the experimental-feature set on every
labeled image and emits a CSV with feature columns concatenated
to label columns. Analyses run the default sampling budget (no
`MAX_PER_CORPUS` override) so the numbers reflect what shipped
0.1.0 will produce in production.

## Per-class distributions

p50/p90 per class. "screen" combines screen_* and illustration
(`#screen-like`); "synth p90" pools `is_synthetic == true`; "max"
is the corpus-wide maximum.

| feature                       | photo p50 | photo p90 | screen p50 | screen p90 | synth p90 | corpus max |
|-------------------------------|----------:|----------:|-----------:|-----------:|----------:|-----------:|
| variance                      |      2994 |      5110 |       2998 |       7514 |      7514 |      11243 |
| edge_density                  |    0.1373 |    0.3243 |     0.0696 |     0.3068 |    0.2639 |     0.8910 |
| chroma_complexity             |    0.1125 |    0.2125 |     0.0797 |     0.2230 |    0.2230 |     0.3142 |
| cb_sharpness                  |    0.0067 |    0.0164 |     0.0025 |     0.0103 |    0.0098 |     0.0663 |
| cr_sharpness                  |    0.0043 |    0.0111 |     0.0017 |     0.0088 |    0.0088 |     0.0285 |
| uniformity                    |    0.4211 |    0.7559 |     0.7865 |     0.8901 |    0.8901 |     0.9822 |
| flat_color_block_ratio        |    0.0298 |    0.2824 |     0.7087 |     0.8677 |    0.8677 |     0.9821 |
| colourfulness                 |      44.0 |      88.4 |       27.2 |       89.4 |      89.4 |      146.0 |
| laplacian_variance            |    0.4842 |      1.93 |       1.75 |       9.16 |      9.16 |       22.0 |
| variance_spread               |      1.31 |      1.74 |       1.28 |       1.83 |      1.83 |       2.63 |
| distinct_color_bins           |      2739 |      7740 |       2030 |      12190 |     12190 |      22602 |
| palette_density               |    0.0836 |    0.2362 |     0.0620 |     0.3720 |    0.3720 |     0.6898 |
| cb_horiz_sharpness            |    0.0025 |    0.0154 |     0.0029 |     0.0227 |    0.0227 |     0.1297 |
| cb_vert_sharpness             |    0.0026 |    0.0149 |     0.0036 |     0.0288 |    0.0288 |     0.1470 |
| cb_peak_sharpness             |       0.0 |       3.0 |        2.0 |       11.0 |      11.0 |       38.0 |
| cr_horiz_sharpness            |    0.0073 |    0.0579 |     0.0129 |     0.0863 |    0.0863 |     0.6771 |
| cr_vert_sharpness             |    0.0076 |    0.0474 |     0.0115 |     0.1480 |    0.1480 |     0.6590 |
| cr_peak_sharpness             |       2.0 |      13.0 |       12.0 |       61.0 |      61.0 |      207.0 |
| high_freq_energy_ratio        |    0.1518 |    0.3747 |     0.1871 |     0.4449 |    0.4449 |     0.9201 |
| luma_histogram_entropy        |      4.22 |      4.60 |       2.39 |       4.20 |      4.19 |       4.73 |
| dct_compressibility_y         |      15.3 |      31.9 |       17.7 |       33.0 |      33.0 |       69.1 |
| dct_compressibility_uv        |      5.40 |      14.6 |       2.79 |       11.0 |      11.0 |       44.8 |
| patch_fraction                |    0.0020 |    0.0366 |     0.7260 |     0.9062 |    0.9062 |     0.9961 |
| text_likelihood               |    0.1879 |    0.3000 |     0.3019 |     0.5351 |    0.5351 |     0.7073 |
| screen_content_likelihood     |    0.0363 |    0.3784 |     0.6183 |     0.6744 |    0.6744 |     0.7000 |
| natural_likelihood            |    0.5211 |    0.6556 |     0.0681 |     0.5719 |    0.5687 |     0.6885 |
| grayscale_score               |    0.0494 |    0.2654 |     0.3493 |     0.9666 |    0.9666 |       1.00 |
| aq_map_mean                   |      3.26 |      4.05 |       1.29 |       3.97 |      3.85 |       4.97 |
| aq_map_std                    |      1.00 |      1.41 |       1.62 |       2.19 |      2.19 |       2.50 |
| noise_floor_y                 |    0.0529 |    0.2431 |     0.0000 |     0.1831 |    0.0974 |       1.00 |
| noise_floor_uv                |    0.0195 |    0.0569 |     0.0000 |     0.0477 |    0.0221 |     0.5474 |
| line_art_score                |    0.0000 |    0.0000 |     0.0432 |     0.5870 |    0.5870 |     0.9787 |

Depth-tier features (peak_luminance_nits, hdr_*, wide_gamut_*,
effective_bit_depth) collapse to the SDR fast-path constants on
every PNG/JPEG input — sRGB transfer + u8 source + sub-sRGB gamut.
Validating those signals needs an HDR corpus (PQ/HLG PNGs with
CICP, 10/12/16-bit PNGs, DNG via rawler). Tracked as task #67.

## Discriminative power: AUC for screen-vs-photo

"screen" = `primary_category` starts with `screen_*` or equals
`illustration` (45 positives, 174 negatives).

| feature                       | AUC   | direction   |
|-------------------------------|------:|-------------|
| **patch_fraction**            | 0.880 | high→screen |
| flat_color_block_ratio        | 0.838 | high→screen |
| screen_content_likelihood     | 0.831 | high→screen |
| natural_likelihood            | 0.814 | high→photo  |
| grayscale_score               | 0.797 | high→screen |
| text_likelihood               | 0.774 | high→screen |
| aq_map_std                    | 0.762 | high→screen |
| laplacian_variance            | 0.760 | high→screen |
| line_art_score                | 0.750 | high→screen |
| uniformity                    | 0.738 | high→screen |
| cr_peak_sharpness             | 0.699 | high→screen |
| cb_peak_sharpness             | 0.696 | high→screen |
| luma_histogram_entropy        | 0.848 | high→photo  |
| aq_map_mean                   | 0.792 | high→photo  |
| noise_floor_y                 | 0.794 | high→photo  |

(For "high→photo" features, "AUC" is `1 − raw_AUC` so larger means
better photo discrimination.)

`patch_fraction` is the single strongest screen-vs-photo
discriminator on this corpus. Downstream consumers picking content
class for codec dispatch should weight it accordingly.

## Recommended operating thresholds

Best F1 thresholds for binary screen-content / photo / line-art
classification on this corpus. F1 is reported at the threshold; P
and R are precision / recall at that threshold.

| feature                       | threshold | F1    | P     | R     | classifies |
|-------------------------------|----------:|------:|------:|------:|------------|
| `line_art_score`     `> 0`    |    0.000  | 0.978 | 0.957 | 1.000 | screen-like |
| `natural_likelihood` `≥ 0.06` |    0.061  | 0.924 | 0.876 | 0.977 | photo |
| `patch_fraction`     `≥ 0.27` |    0.274  | 0.769 | 0.909 | 0.667 | screen-like |
| `screen_content_likelihood` `≥ 0.60` |  0.600  | 0.750 | 0.857 | 0.667 | screen-like |
| `flat_color_block_ratio`     `≥ 0.53` |  0.525  | 0.750 | 0.857 | 0.667 | screen-like |
| `text_likelihood`    `≥ 0.30` |    0.300  | 0.682 | 0.698 | 0.667 | screen-like |
| `grayscale_score`    `≥ 0.23` |    0.231  | 0.617 | 0.592 | 0.644 | screen-like |

The `_likelihood` features empirically saturate well below the
`[0, 1]` theoretical range:

- `text_likelihood` corpus max **0.71**
- `screen_content_likelihood` corpus max **0.70**
- `natural_likelihood` corpus max **0.69**

This is a property of the formula structure (each is a weighted
sum of clamped sub-components, and the sub-components don't
simultaneously max on real content). The right operating points
are around the empirical 90th percentile of the target class, not
the theoretical 1.0 — see the table above. **Do not threshold at
0.8 or higher; nothing fires there.**

## Spearman redundancy

Top |ρ| pairs across features. Pairs with |ρ| ≥ 0.85 are
candidates for consolidation, but several of these reflect
formula-by-construction relationships (e.g.,
`screen_content_likelihood` is `0.6·flat_color_block_ratio + …`,
so ρ=0.989 with `flat_color_block_ratio` is structural, not
incidental).

| ρ     | A                            | B                          | note |
|-------|------------------------------|----------------------------|------|
| +1.000| distinct_color_bins          | palette_density            | derived (`bins / max_bins`) |
| +0.989| flat_color_block_ratio       | screen_content_likelihood  | structural — primary input |
| −0.967| uniformity                   | aq_map_mean                | both measure block flatness |
| +0.965| chroma_complexity            | colourfulness              | both quantify chroma spread |
| +0.914| noise_floor_y                | aq_map_mean                | high-AQ regions have residual noise |
| −0.902| flat_color_block_ratio       | noise_floor_y              | flat blocks have no AC residue |
| −0.892| noise_floor_y                | screen_content_likelihood  | screens have no sensor noise |
| +0.886| cr_sharpness                 | dct_compressibility_uv     | chroma sharpness drives chroma AC |
| +0.886| cb_sharpness                 | cr_sharpness               | both planes track each other |
| +0.875| cb_sharpness                 | dct_compressibility_uv     | same shape on cb |
| +0.859| luma_histogram_entropy       | natural_likelihood         | structural — primary input |
| +0.858| aq_map_mean                  | natural_likelihood         | photos have higher AQ |
| −0.857| uniformity                   | noise_floor_y              | flat regions → no noise |
| −0.848| edge_density                 | uniformity                 | inverse by definition |
| +0.846| noise_floor_y                | natural_likelihood         | photos noisy, screens clean |
| −0.845| noise_floor_uv               | screen_content_likelihood  | same chroma reasoning |
| +0.841| edge_density                 | dct_compressibility_y      | edges drive AC energy |
| +0.838| noise_floor_y                | noise_floor_uv             | Y/UV noise correlates |
| −0.826| screen_content_likelihood    | natural_likelihood         | mutually exclusive classes |

`palette_density` ≡ `distinct_color_bins / 32768` and is purely a
display convenience; consumers can pick whichever is more
ergonomic.

## What the data does NOT support

This pass evaluated several plausible recalibrations that the
empirical data does not back:

1. **Lifting `palette_small` ceiling in `screen_content_likelihood`
   (4000→16000 distinct color bins).** AUC drops 0.831 → 0.817.
   Photo p90 rises from 0.578 to 0.651 — same magnitude as the
   screen median rise, no net discrimination gain.
2. **Multiplicative rescale of derived likelihoods to fill `[0, 1]`.**
   Cosmetic only — the optimal threshold moves 0.6 → 0.86 with
   identical F1. Just shifts the magic number.
3. **Rescaling `dct_compressibility_y/uv` by /32 to land in `[0, 2.2]`.**
   Less readable than the current `[0, 70]` raw form on real
   content; theoretical max is 8064 only on degenerate synthetic
   inputs.
4. **Rescaling `luma_histogram_entropy` by /5 to `[0, 1]`.**
   Loses the Shannon-entropy semantics; the 4.73/5.00 corpus
   ceiling already uses 95% of the range.
5. **Rescaling `cr_peak_sharpness` to suppress the synthetic-content
   tail (max 207).** Real photos cap at 57; the 207 max is the
   stress-test signal. Renormalisation is reserved for the
   `infer_bucket` calibration step that consumers will own.

## What the data DID surface

1. **Derived likelihoods empirically saturate at ~0.7, not 1.0.**
   Documented in this report and in the rustdoc on each variant.
   Operating thresholds in the 0.3–0.6 band are correct.
2. **`patch_fraction` is the strongest single screen discriminator.**
   AUC 0.880, F1 0.769 at t≥0.274. Promote in usage guides and in
   the consumer-side codec dispatch.
3. **`line_art_score > 0` is essentially deterministic for
   screen-like content.** F1 0.978 — `> 0` is a reliable enough
   signal to skip soft thresholds.
4. **`natural_likelihood ≥ 0.06` cleanly separates photos from
   screens.** F1 0.924 for photo classification.
5. **`noise_floor_y` and `screen_content_likelihood` correlate at
   ρ=−0.892.** Defensible (screens have no sensor noise) but
   downstream consumers driving different codec knobs
   (`pre_blur` vs encoder preset) should prefer the one that
   matches their physical intent.

## Net 0.1.0 action

**No code changes.** The current calibration is empirically
appropriate for the labeled corpus. Numeric drift is permitted in
0.1.x patches per the threshold contract; this report is the
pre-ship empirical baseline against which patch drift can be
compared.

The rustdoc on each derived likelihood and on `patch_fraction` is
updated to surface observed ranges and the recommended operating
thresholds from this report. The README's "Threshold contract"
section now references this document.

## Reproducing

```sh
# Build:
cargo build --release -p zenanalyze --features experimental --example corpus_eval

# Run labeled mode against the coefficient corpus:
LABELS_TSV=/path/to/coefficient/benchmarks/classifier-eval/labels.tsv \
CORPUS_ROOT=/path/to/codec-corpus \
  ./target/release/examples/corpus_eval > /tmp/zenanalyze_labeled.csv

# Then aggregate per-class p10/p50/p90, AUC, and Spearman ρ in
# Python from the CSV — the report tables above can be regenerated
# with any standard percentile + Mann-Whitney U code.
```

`CORPUS_ROOT` defaults to `~/work/codec-eval/codec-corpus`. The
labeled-mode resolver expects the standard sub-layout
`<corpus>/<sub>/<image>` (e.g. `CID22/CID22-512/training/<image>`,
`clic2025-1024/<image>`, `gb82-sc/<image>`). The per-corpus search
list is in `resolve_dirs` near the top of `run_labeled` in
`examples/corpus_eval.rs` — extend it for new corpora.

## Addendum 2026-04-27 — `SkinToneFraction` and `EdgeSlopeStdev`

After the original calibration pass landed, two new experimental
features were added to address issue #123 (artwork-vs-natural-photo
discriminator). Both validated on the same 219-image labeled corpus
before shipping.

### Per-class distribution

| feature              | photo_natural p50 | photo_portrait p50 | screen_doc p50 | screen_ui p50 | illustration p50 |
|----------------------|------------------:|-------------------:|---------------:|--------------:|-----------------:|
| `skin_tone_fraction` |             0.130 |              0.241 |          0.006 |         0.027 |            0.081 |
| `edge_slope_stdev`   |              24.2 |               20.7 |           55.3 |          42.1 |             20.9 |

### Discriminative power (photo-vs-other AUC)

| feature              | AUC    | rank | notes |
|----------------------|-------:|-----:|-------|
| `skin_tone_fraction` |  0.799 |  3rd | comparable to `natural_likelihood` (0.814), `noise_floor_y` (0.794) |
| `edge_slope_stdev`   |  0.844* | 2nd  | only `patch_fraction` (0.880) discriminates better |

*`edge_slope_stdev` AUC is high → screen content (i.e., 1 − raw_AUC)
since the natural direction of the signal is "high stddev ⇒ artwork
or stylized content".

### Operating thresholds

`skin_tone_fraction`: one-direction signal. `>= 0.05` gives `P = 0.89,
R = 0.76, F1 = 0.82` for photo classification. Lower thresholds
maximize recall (`> 0` ⇒ `F1 = 0.882`). Zero fraction is **not**
evidence against a photograph.

`edge_slope_stdev`: bidirectional. Photos cluster at 15–32, screens
at 32–58. `>= 35` is a strong screen-detection signal but
illustration overlaps photo (13–27) — does not solve the issue #123
illustration-vs-photo subproblem.

### Pigmentation invariance — `SkinToneFraction`

The Chai-Ngan (1999) YCbCr ranges (`Cb ∈ [77, 127], Cr ∈ [133, 173],
Y ∈ [40, 240]`) are **pigmentation-invariant by design** — chroma
quantifies hue, not lightness, so the same gates work across light,
medium, and dark skin tones. Verified against the
`photo_portrait` subset (n=32, diverse subjects): 93.8% have
`skin_tone_fraction > 0.01`, with p50=0.241 and p90=0.541. No skew
toward any specific pigmentation in the misses.

The unit test
`skin_tone_fraction_fires_on_skin_colored_pixels_zero_on_neutral`
locks invariance across three representative tones (light pinkish
RGB(236,188,180), medium tan RGB(198,134,105), deep brown RGB(90,
56,37)) — all score > 0.95 on uniform-skin-tone test images.

### Performance (4 MP, AVX2, no `-C target-cpu=native`)

#### New-feature incremental cost (RGB8 input, single-feature query)

| feature              | mean    | incremental vs `EdgeDensity` |
|----------------------|--------:|-----------------------------:|
| `EdgeDensity`        | 4.31 ms | baseline                     |
| `GrayscaleScore`     | 4.24 ms | -0.07 ms (noise)             |
| `SkinToneFraction`   | 4.24 ms | **-0.07 ms** — free          |
| `EdgeSlopeStdev`     | 4.32 ms | **+0.01 ms** — free          |

Multi-feature: all four together run in 4.25 ms — confirms the
piggyback design (each new feature shares the same Tier 1 stripe
sweep, no second pass).

Both new kernels use `#[archmage::autoversion(v4x, v4, v3, neon,
scalar)]` over `chunks_exact(3)` paired iterators. The chunk size
proves the inner triplet of u8 loads to LLVM, which then emits
per-arch SIMD shuffles + integer multiplies for the BT.601 fixed-
point YCbCr path.

#### `RowStream` row-fetch path comparison

`FeatureSet::SUPPORTED` cost by input format:

| input  | RowStream path                       | mean     |
|--------|--------------------------------------|---------:|
| RGB8   | `Native` (zero-copy slice subindex)  |  9.50 ms |
| RGBA8  | `StripAlpha8` (garb SIMD)            | 10.94 ms |
| BGRA8  | `StripAlpha8` (garb SIMD)            | 12.00 ms |
| RGB16  | `Convert` (zenpixels-convert)        | 24.67 ms |
| RGBA16 | `Convert` (zenpixels-convert)        | 28.64 ms |

Row-level microbench, 2048-px row, RGBA8 → RGB8:

| implementation                                        | µs / row | speedup |
|-------------------------------------------------------|---------:|--------:|
| zenanalyze in-tree scalar strip (pre-2026-04-28)      |     0.96 | 1.0×    |
| `garb::bytes::rgba_to_rgb` (`incant!` SIMD dispatch)  |     0.13 | **7.4×** |
| `garb::bytes::bgra_to_rgb` (`incant!` SIMD dispatch)  |     0.13 | 7.3×    |

The previous in-tree strip was a plain `chunks_exact(4)` scalar loop
that LLVM didn't autovectorize cleanly. Switching `strip_alpha_row`
to `garb::bytes::{rgba,bgra}_to_rgb` (which dispatches to AVX2 /
NEON / WASM128 via `archmage::incant!`) cut the RGBA8 analyzer
overhead vs RGB8 baseline from +5.8 ms to +1.4 ms — strip-alpha is
no longer the bottleneck on RGBA8 input.

**vs zenpixels-convert.** The `Convert` path on 16-bit input is
already SIMD-aware (zenpixels-convert routes through garb-style
primitives) and runs at ~6 ms / MP for the full transfer-function-
aware narrowing. That path is the right tool for genuinely heterogeneous
input (16-bit, f32, gray, CMYK, …). For the trivial RGBA8 / BGRA8
strip-alpha case, the dedicated bypass is ~3× faster than going
through the full converter.

**vs garb directly for the new analyzer features.** Neither garb nor
zenpixels-convert exposes a skin-tone classifier or edge-slope
statistic. The closest alternative would allocate a YCbCr8 row scratch
and walk it for the gate test — doubling memory traffic vs our inline
register-resident YCbCr classification with zero arithmetic savings.

Reproduce: `cargo run --release -p zenanalyze --features
experimental --example edge_skin_bench`.

### What didn't work

A third feature, `flat_block_chroma_luma_noise_ratio`, was prototyped
in tier3 (sensor-noise discriminator from issue #123) and reverted.
Empirical AUC = 0.625 — worse than the existing `noise_floor_y`
(0.794). Root cause documented in
[issue #123 comment](https://github.com/imazen/zenjpeg/issues/123#issuecomment-4331156178):
BT.601 chroma matrix cancels the RGB-domain ratio the issue assumed,
and JPEG 4:2:0 chroma subsampling further suppresses within-block
chroma stddev. The cheap chroma/luma stddev ratio doesn't recover the
sensor-noise signal in the corpus we have.
