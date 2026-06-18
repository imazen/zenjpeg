# JPEG recompression: everything learned

A single reference for the zenjpeg-recompress research: the problem, the
algorithms, every measured finding, the open turbo problem, and — at the
end — what `jxl-encoder` can borrow when transcoding JPEGs to JXL.

Companion docs (detail): `GENERATION_LOSS_THEORY.md`,
`AQ_DIRECTION.md`, `TRI_METRIC_CROSSCHECK.md`,
`MULTI_ENCODER_VALIDATION.md`. Data: `../benchmarks/*.tsv` +
`/mnt/v/zen/zenjpeg-recompress/calibration-n50-2026-05-29/` (pointer in
`../benchmarks/calibration-n50-2026-05-29.pointer.md`).

> **Revision 2026-05-29 — levers 1–5 landed + n=50 recalibration.** The
> open "turbo 12 %" problem of the prior revision is closed. Summary of
> what changed (detail in §5, §8, and the rewritten §10 JXL notes):
> - **Lever 1** — the Preserve fine→coarse crater was a real bug: the
>   2-table layout (turbo/mozjpeg share one chroma quant table) was
>   scaled twice (scale²). Fixed → turbo q90→t70 Preserve 8.1 → 67.5.
> - **Lever 2** — `RobidouxTargetQuality`: same-family requant for
>   mozjpeg/ImageMagick (RD −2.7 % low-q / −1.0 % high-q vs cross-family).
> - **Lever 3** — per-encoder target→dial inverse calibration.
> - **Lever 4** — closed loop on `MaxIterations(n>1)`: the
>   no-clean-reference within-cell signal (§10.3 below).
> - **Lever 5** — recalibrated on **n=50** CID22-512 (was n=6); under-
>   target turbo 13→8.6 %, mozjpeg 8→7 %, jpegli ~4 %, 0 size regressions.
> - **Picker margin** — lossless must beat recompression by a real margin
>   before winning (it overshoots the quality target).

---

## 1. The problem

Given an already-compressed JPEG and a target quality expressed in
**zensim Profile A** units `[0,100]`, produce the smallest output JPEG
whose quality **vs the original (unknown) reference** is at least the
target — and *never* larger than the source. Three things make it hard:

1. **The reference is gone.** We only have the lossy source. All runtime
   measurement is vs the source (generation loss), never vs the original.
   The product target is *cumulative* quality vs the original, which we
   reach only through calibration.
2. **The source is already a JPEG.** Its DCT coefficients carry the prior
   encoder's quantization, block structure, and ringing. Any pixel-domain
   re-encode re-quantizes that noise again — compounding generation loss.
3. **Encoder identity matters.** libjpeg-turbo, mozjpeg, ImageMagick,
   jpegli, Photoshop each use different quant tables and Huffman
   strategies, so the same nominal quality is a different RD point.

---

## 2. The strategy taxonomy

Four strategies on the (generation-loss × bitrate × CPU) surface:

| strategy | mechanism | pixel round-trips | can grow coefficient support |
|---|---|---|---|
| **Lossless** | `zenjpeg::lossless::restructure` — re-pack scans + optimize Huffman, coefficients untouched | 0 | no |
| **Preserve** | coefficient-domain: scale quant tables, re-quantize coefficients, optional AQ zero-bias | 0 | no |
| **Tuned** | decode → zenjpeg re-encode (HybridMaxCompression) | 1 | yes |
| **Deblock** | decode with content-aware deblock → re-encode | 1 | yes |

Plus **NoOp** (source already ≤ target; recompressing can only hurt).

---

## 3. The math of generation loss (why Preserve exists)

Operators on an 8×8 block: `F` = orthonormal DCT, `Q_a` = quantize at
table `a`, `R` = round-and-clamp pixels to `[0,255]`.

**Result 1 — coefficient-domain re-encode at the same table is exactly
idempotent.** `T'_Q(Ĉ) = round(qₛ·Qₛ/Qₛ)·Qₛ = Ĉ`. Zero loss by algebra.
The pixel path `T_Q(Ĉ) = Q_Q(F(R(F⁻¹(Ĉ))))` is **not** idempotent because
`R(F⁻¹(Ĉ)) ≠ F⁻¹(Ĉ)`.

**Result 2 — pixel rounding is broadband and resurrects killed
frequencies.** `R`'s error `|ε| ≤ ½` has energy at every frequency; by
Parseval ≈ 0.29 RMS per coefficient lands in the DCT domain, reviving
coefficients the source had quantized to exact zero. **Support
monotonicity**: coefficient-domain edits can only shrink the support; the
pixel path grows it. The "frequency artifact" of generation loss is
exactly this support growth — strictly a pixel-path phenomenon.

**Result 3 — the [0,255] clamp is a biased error**, reapplied every pixel
generation (a ratchet at highlights/shadows and ringing edges).

**Result 4 — exact-multiple requantize is loss-minimal.** If
`Q_t = k·Qₛ` (integer k), `q_t = round(qₛ/k)` — pure integer level
division, no DCT, no cross-frequency coupling.

**Result 5 — dynamical systems.** Iterating the pixel re-encode converges
to a fixed point of `T_Q` (why generation loss settles after a few
saves). Preserve *is* a projection — lands on the fixed point in one step.

Generation-loss budget: Lossless = 0, Preserve = 0, Tuned = 1,
Deblock = 1. **Measured (tri-metric, §6): Preserve beats Tuned on quality
at a given target on 48/54 (zensim), 40/54 (butter), 39/54 (cvvdp)
cells** — the algebraic advantage is perceptual too, under all three
metrics.

---

## 4. The decision algorithm (router)

```
probe → encoder family, source quality, subsampling
source_est = per-encoder (quality → estimated cumulative zensim-A vs original)
effective_target = clamp(user_target + confidence_shift, 0, 100)

if source_est ≤ user_target + 1.5:           → NoOp
project {Preserve, Deblock, Tuned, Lossless} from per-encoder calibration at effective_target
candidate = smallest-size among strategies with projected ≥ effective_target − 2
            (else highest projected)
if candidate.ratio ≥ lossless.ratio + 0.03  OR  candidate.ratio ≥ 0.98:  → Lossless
else dispatch candidate; if output ≥ source bytes → Lossless fallback
```

**Picker margin (2026-05-29).** Lossless wins only when it is
*meaningfully* smaller than the best recompression (`+0.03`), not on a
near-tie. Rationale: Lossless **overshoots the quality target** — it
ships the source's full quality, not the lower target the user asked for
— and `estimate_lossless` is a flat 0.94 guess, so a noisy 0.0003 gap
between a recompression projection and that guess must not flip a real
target-hitting recompression (22 % savings) to a barely-smaller Lossless.
Net-zero for turbo/jpegli; it stopped mozjpeg tie-flips.

**Closed loop (`Budget::MaxIterations(n>1)`, §8 Lever 4).** The default
`OneShot` runs the chosen strategy once. With `n>1`, each pass measures
generation loss vs source and bumps the dial when it predicts the output
landed short of target (see §10.3 for the signal). `OneShot` output is
byte-identical to pre-loop.

The invariants this enforces, encoder-independently:
- **never larger than source** (Lossless fallback + byte-level guard),
- **NoOp on the user target**, not the confidence-shifted one.

### Confidence (delivery percentile)
`Confidence::{P25,P50,P75,P90,P95}` shifts the internal aim so the chosen
quantile of achieved quality clears the user target. Shifts from the
calibration residual tail (`achieved − projected`): P25 −5.1, P50 0,
P75 +2.8, P90 +13.7, P95 +19.0 zensim-A. Higher confidence ⇒ larger files
(or fall-through to Lossless). **Measured: under-target delivery 34.6 % →
8.4 % moving always-on AQ to the P50 headroom gate.**

### AQ (which blocks to zero-bias)
Per-luma-block tiered classifier on high-band/low-band AC energy ratio:
VeryFlat zeros AC≥32, Flat zeros AC≥48, MidDetail/Detailed untouched.
Gated on quality headroom: AQ only fires when projected − target ≥ 2 (it's
a measured ~3-5 % size for ~1 zensim-A trade — never free). **Direction
finding (§6): flat-targeting beats busy-targeting 2-13× per unit quality
even under masking-aware metrics, because requantization already removed
busy blocks' high-frequency content — the opposite of from-scratch
encoder AQ.**

---

## 5. Per-encoder calibration — and the quant-table families

Quant-table family per encoder (this drives everything below):

| encoder | quant tables | detected as |
|---|---|---|
| libjpeg-turbo, Pillow, IJG | **standard IJG (ITU-T T.81 Annex K)** | `LibjpegTurbo` / `IjgFamily` |
| **mozjpeg, ImageMagick** | **Robidoux** (psychovisual, NOT IJG) | `Mozjpeg` / `ImageMagick` |
| jpegli | distance-optimized perceptual | `CjpegliYcbcr` / `…Xyb` |

Calibration has three layers, all originally jpegli-fit, now per-encoder,
**refit on n=50 CID22-512 (2026-05-29)** — was n=6:
1. **source-quality anchor** (encoder-q → estimated cumulative zensim-A):
   measured per encoder (`target.rs::ijg_q_to_zensim_a`, 16 q-points).
2. **achieved-quality table** (per encoder × strategy × source-est ×
   target → achieved + size-ratio): `per_encoder.rs`, measured from
   forced-strategy sweeps on **pinned encoders** (libjpeg-turbo 3.1.0 +
   mozjpeg 4.1.5, extracted from the all-the-images docker stages).
3. **re-encode params**: HybridMaxCompression (§7).

**Measured under-target delivery (output > 2 zensim-A below target):**

| encoder | jpegli-fit | + src anchors | per-enc n=6 | **per-enc n=50** | + closed loop | naive deblock |
|---|---|---|---|---|---|---|
| jpegli | 4 % | 4 % | 4 % | **3.6 %** | 3.6 % | 53 % |
| libjpeg-turbo | 76 % | 70 % | 12 % | **8.6 %** | 8.0 % | 60 % |
| mozjpeg | 75 % | 54 % | 8 % | **7 %** | 5.6 % | 63 % |

Zero size regressions for the router on all encoders at every column;
naive deblock inflates 146–254 cells. **The router's NoOp / Lossless /
per-encoder selection is the entire value-add over naive recompression.**
The n=6→n=50 step roughly halved turbo/mozjpeg under-target — the prior
miss rate was inflated by bucket-median noise (n≈1–6 per cell), not a real
calibration error.

### What the per-encoder tables revealed
- **For libjpeg-turbo, Preserve still rarely wins.** It is dominated by
  Lossless when gentle (Lossless is smaller AND keeps full quality), and
  Tuned (source-encoder-independent re-encode from pixels) is the
  well-behaved escape that never craters. (The old "always craters"
  framing was the Lever-1 shared-chroma bug, now fixed — see §8.)
- **For mozjpeg/ImageMagick, Robidoux-aware Preserve now competes**
  (Lever 2): same-family requant gives −2.7 % size at matched quality
  (low-q bucket), and the router picks Preserve ~2× more often than with
  the cross-family path.
- **For jpegli, Preserve wins** — its distance-optimized coefficients
  requantize gracefully. jpegli keeps the richer `data::lookup_420/444`
  calibration (4:2:0 regenerated at n=50; 4:4:4 still 15-image).

---

## 6. Tri-metric cross-check (zensim vs butteraugli vs cvvdp)

Both headline decisions re-run under all three metrics on 378 variants
(GPU, `zenmetrics`). **No decision flips.** Notable divergences:
- cvvdp is the most permissive of flat AQ (rates its cost ≈ 0 vs zensim's
  −0.23) — the production AQ gate is conservative under cvvdp.
- zensim over-weights Preserve vs Tuned relative to cvvdp (89 % vs 72 %
  win rate). **The strategy router is calibrated on zensim; v0.3 should
  recalibrate on cvvdp or a blend** — cvvdp is the closest proxy to human
  JOD.

---

## 7. Encoder params — HybridMaxCompression, not auto_optimize

6-ref RD ablation, bytes at matched zensim vs `auto_optimize`:

| param set | zensim 60 | zensim 70 | zensim 80 |
|---|---|---|---|
| auto_optimize | 1.000 | 1.000 | 1.000 |
| **HybridMaxCompression** | **0.960** | **0.982** | **0.990** |
| XYB | — | 0.984 | 0.939 |
| jpegli_prog / mozjpeg_max / prog_search | all > 1.0 (worse) | | |

HybridMaxCompression (jpegli AQ + adaptive trellis + deringing +
progressive scan search) wins 1-4 % at every quality, pure YCbCr. XYB
wins more at high quality (6 % at zensim 80) but changes color/decoder
compatibility — reserved for a future modern-decoder mode.

---

## 8. Closing the turbo/mozjpeg gap — levers 1–5 (done 2026-05-29)

The prior revision left "turbo 12 %" open and listed five ranked levers.
All five shipped; results below. Net: turbo 13→8.0 %, mozjpeg 8→5.6 %
under-target, **zero size regressions** throughout.

The investigation started from a **falsified** hypothesis that turned out
to be a bug: forcing Preserve to standard-IJG TargetQuality tables for
turbo *cratered worse* (src90→t70: 54.8 → 7.2 zensim-A). The collapse
looked like a fine→coarse ratio/clamp problem — and it was, but not where
expected.

1. **Lever 1 — the crater was a real bug, now fixed.** `build_new_quant_
   tables` rebuilt quant tables by iterating per *component*. turbo and
   mozjpeg use a **2-table layout** where Cb and Cr **share one chroma
   table**; the loop scaled that shared table once per chroma component →
   chroma quantized by **scale²**. jpegli's 3-separate-table layout hid
   it. Fix: build each *unique* table exactly once from the ORIGINAL
   coefficients. turbo q90→t70 Preserve **8.1 → 67.5**. (Pixels are
   sacred — a 2× chroma over-quantization is a shipping bug, not a tuning
   knob.)
2. **Lever 2 — Robidoux-aware Preserve for mozjpeg/ImageMagick.** New
   `QuantStrategy::RobidouxTargetQuality` retargets to the Robidoux base
   table (their native family) at the inverse-calibrated dial, instead of
   cross-family IJG-std. Same-family ⇒ the per-position `old/new` ratio is
   near-uniform (no spectral reshape), so the requant acts like a clean
   uniform scale to the *exact* target. **RD −2.7 % size at matched
   achieved quality (low-q bucket), −1.0 % (high-q)** vs the old dispatch.
3. **Lever 3 — per-encoder Tuned/Preserve target→dial inverse
   calibration** (`per_encoder::invert_dial`). Inverts the achieved-vs-
   dial curve so aiming at target T dials the encoder-q that *actually*
   achieves T on that encoder family, correcting the systematic ~5-point
   miss.
4. **Lever 4 — closed loop** (`Budget::MaxIterations(n>1)`). Measures
   generation loss vs source, predicts achieved-vs-original, bumps the
   dial when short. The signal that makes it work is the headline JXL
   transfer note — see §10.3. Measured turbo 8.6→8.0 %, mozjpeg
   7.0→5.6 %, 0 regressions; ~20 % of the residual is Lossless cells with
   no dial to bump (honest source-estimate overshoot).
5. **Lever 5 — n=50 recalibration** (was n=6). Refit per-encoder tables +
   source anchors + the `data.rs` 4:2:0 fallback on 50 CID22-512 refs.
   Halved turbo/mozjpeg under-target; the old miss rate was bucket-median
   noise, not a model error. Also fixed `cumulative-sweep --force-tuned`
   (the smart router was starving the Tuned-fallback fit → degenerate
   grid).

**Remaining (deferred, not core):** `data.rs` 4:4:4 still 15-image; the
`*_GEXP` closed-loop tables are hand-fit (pipeline regen needs the forced
sweep switched to `MaxIterations(1)`); and the corpus is all 512 px — the
4-size discipline (tiny/medium/large) is the next major effort.

---

## 9. Tooling

- `recompress(jpeg, RecompressOptions)` — product API, one entry point.
  `with_budget(MaxIterations(n))` enables the §8 Lever-4 closed loop.
- `zjr-calibrate` — `inspect` / `sweep` / `cumulative-sweep`
  (`--force-tuned` for the `data.rs` fallback fit) / `recompress-sweep`
  (`--force-strategy`, `--naive-deblock`, `--max-iterations N`).
- `scripts/recalibrate.sh <orig_dir> <work> <qstep>` — one-command full
  recalibration: pinned-encoder source gen → source anchors → forced
  sweeps → fit + auto-splice `per_encoder.rs` → `data.rs` 4:2:0 fit →
  rebuild + validation GATE (under-target ≤ 15 %, 0 regressions).
- `scripts/fit_per_encoder.py` (per-encoder tables, auto-spliced),
  `scripts/fit_source_anchors.py` (anchors), `scripts/fit_calibration.py`
  (`data.rs` 4:2:0/4:4:4 from a `--force-tuned` cumulative sweep).
- `scripts/lever4_validate.sh` — regenerates the corpus (persisted
  outside `/tmp`) and runs the iter1-vs-iter3 closed-loop comparison.
- examples (all behind `--features expert`): `preserve_identity_diag`,
  `preserve_vs_tuned`, `aq_ablation`, `aq_direction`, `tri_metric_gen`,
  `zenjpeg_param_rd`, `dump_estimates` (prints `estimate_all` for a cell).

---

## 10. Notes for jxl-encoder: transcoding JPEG → JXL

`jxl-encoder` already does **lossless JPEG→JXL transcoding** (re-coding
the existing 8×8 DCT coefficients into JXL's entropy coder — the headline
JXL feature, bit-exact recoverable). The findings here apply to **lossy**
JPEG→JXL recompression: taking an existing JPEG and producing a *smaller*
JXL at a target perceptual quality. That is the same problem this repo
solves (JPEG→JPEG), and most of it transfers directly. Ordered by how
much each lever moved the needle here.

### 10.1 Stay in the coefficient domain (Results 1–5)
JXL's VarDCT can ingest the JPEG's 8×8 DCT coefficients directly, so a
coefficient-domain re-quantize / re-entropy-code avoids the pixel
round-trip that resurrects killed frequencies (Result 2: ≈0.29 RMS/coeff
of broadband rounding error) and re-ratchets the [0,255] clamp (Result 3).
**Support monotonicity** holds: coefficient edits can only *shrink* the
nonzero set; a pixel round-trip *grows* it. A lossy JPEG→JXL that decodes
to pixels and re-encodes accumulates the exact mosquito-noise / ringing
generation loss in §3. Prefer DCT-domain requant; reserve the pixel path
(JXL's normal VarDCT encode) for cases where you deliberately want a
different transform block size or XYB.

### 10.2 Same-family requant beats reshape — and watch the share-scale bug
This is two findings from Levers 1–2 that JXL must heed because its quant
model is *more* flexible than JPEG's, not less.

- **Same-family (Lever 2).** Retargeting a source to a quant matrix of the
  *same spectral shape* keeps the per-position `old/new` ratio nearly
  constant — a clean uniform rescale to the exact target, minimal
  per-coefficient rounding. Cross-family reshape (imposing a differently-
  shaped matrix) craters quality. Measured here: same-family Robidoux
  requant was **−2.7 % size at matched quality** vs cross-family IJG-std.
  JXL can pick *any* quant matrix per position, so the temptation is to
  impose JXL's "ideal" matrix — **don't**; derive the target matrix by
  scaling the source's own dequant weights (which encode the prior
  encoder's psychovisual shape), then only deviate where you have a
  measured reason.
- **The share-scale bug (Lever 1).** JPEG's 2-table layout (luma + shared
  Cb/Cr) bit us: a per-*component* requant loop scaled the shared chroma
  table **twice** (scale²), silently quantizing chroma 2× too hard.
  jpegli's 3-table layout hid it. JXL has its own sharing (DC quant, a
  global AC scale, per-channel multipliers, named quant matrices reused
  across blocks). **Any requant that scales a shared object must build
  each unique object exactly once from the ORIGINAL, never iterate over
  references to a mutated copy.** This is a *correctness* bug class, not a
  tuning knob — it produced visibly wrong chroma with no error.

### 10.3 The no-clean-reference closed loop (Lever 4) — the key technique
**At transcode time the original is gone; you can only measure generation
loss vs the *source* (`g`), but the product target is quality vs the
*original* (`A`).** This is the central difficulty and JXL faces it
identically. What we learned:

- The *global* relation `A ≈ f(source_estimate, g)` is **weak** — fitting
  it gave recall 0.32 for detecting under-target outputs. Do **not** build
  a transcoder loop on the absolute `g`.
- But the *within-cell* relation is **strong**: holding (source-quality
  bucket, target) fixed, the per-image deviation of `g` from the cell's
  median tracks the per-image deviation of `A` almost **1:1** (corr 0.80,
  slope ≈1.1, 88 % sign concordance). An image that took more generation
  loss than its bucket median lands below the bucket's median achieved.
- So the usable loop is: precompute a per-cell **expected generation loss**
  table `g_exp[source_bucket][target]`; at transcode time predict
  `A_hat = projected_A + slope·(g_measured − g_exp)`; if `A_hat < target`,
  bump the JXL distance/quality and re-encode. Measured win here: mozjpeg
  under-target 7.0→5.6 %, turbo 8.6→8.0 %, **zero size regressions**,
  default one-shot path byte-identical (loop is opt-in).
- **Ceiling to expect:** ~20 % of residual under-target cells are
  "lossless" (source already below target after honest estimation) — no
  dial bumps them. Don't chase those with the loop; surface them as
  "source can't reach target."

JXL's advantage: it can also measure `g` cheaply in its own decoder and
has a continuous distance knob (finer than JPEG's q-ladder), so the bump
step converges in fewer iterations.

### 10.4 Invert the calibration — don't feed the target as the dial (Lever 3)
The achieved quality at a "naive" dial systematically misses target
(~5 zensim-A here). Build the inverse of the achieved-vs-dial curve per
source-encoder family so that aiming at target T selects the JXL distance
that *actually achieves* T on that source. Feeding the user's target
straight in as the JXL distance under-delivers.

### 10.5 AQ for recompression is the *opposite* of AQ for encoding (§4)
From-scratch encoders — including JXL's own adaptive quantization —
coarsen *busy* blocks (contrast masking). But the source JPEG already
stripped busy blocks' high-frequency energy, so a recompression AQ pass
should target the surviving high-freq residue in *flat* blocks, not busy
ones (flat-targeting beat busy-targeting 2–13× per unit quality here, even
under masking-aware cvvdp). **JXL transcoding should not re-run busy-block
AQ on top of what the source already did** — at most, gate it on measured
quality headroom (it's a ~3–5 % size for ~1 quality-point trade, never
free).

### 10.6 Lossless fallback + a real margin (§4 picker)
No-size-regression is free coefficient algebra; JXL's lossless JPEG
transcode is the natural "do no harm" floor. But **lossless overshoots the
quality target** (it ships the source's full quality, not the lower target
asked for), so prefer the lossy recompression unless lossless is
*meaningfully* smaller — not on a near-tie. We require lossless to beat the
best lossy candidate by a real margin (0.03 of source size) before taking
it; a noisy tie otherwise ships a barely-smaller, quality-overshooting
file instead of a real target-hitting transcode.

### 10.7 Delivery confidence as a percentile
Content variance is large (residual lower tail −13.7 zensim-A at p10), so
a single "target quality" silently under-delivers on ~half of images.
Expose P50/P75/P90 — an upward shift of the internal aim sized from the
calibration residual tail — so the caller trades bytes for a delivery
guarantee. Maps directly onto a JXL distance dial.

### 10.8 Pick the metric deliberately, and mind the calibration corpus
- zensim, butteraugli, cvvdp agree on gen-loss *ordering* but disagree on
  RD tradeoffs by meaningful margins. Calibrate the JXL quality model on
  the metric the product optimizes (cvvdp ≈ human JOD); don't assume
  SSIMULACRA2/zensim transfers.
- **n matters more than expected (Lever 5).** At n≈6 refs/cell the
  bucket-median calibration was noisy enough to flip routing on
  thresholds; n=50 roughly halved apparent under-target with *no model
  change*. Fit on ≥50 refs/cell.
- **Size diversity (open here, mandatory for JXL).** Our corpus is all
  512 px. JXL ships across thumbnails→4K; fixed byte overhead (container,
  ICC, signature) and per-pixel behavior diverge by size, so a JPEG→JXL
  model must sweep tiny/small/medium/large or it miscalibrates at the
  extremes.
- **Pinned encoders.** "host cjpeg" is an unknown version; pin the source
  encoders (we use the all-the-images docker stages: libjpeg-turbo 3.1.0 +
  mozjpeg 4.1.5) or the calibration bakes in one build's quirks.
