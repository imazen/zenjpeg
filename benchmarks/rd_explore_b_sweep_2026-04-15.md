# RD exploration: XYB B-channel coarseness sweep, 2026-04-15

Harness: `zenjpeg/examples/rd_explore.rs`
CSV: `benchmarks/rd_explore_b_sweep_2026-04-15.csv` (440 rows)
Corpus: 1 frymire + 6 CID22-512 photos + 3 gb82 graphics, Q ∈ {50, 70, 85, 95}.

Configs (all encode-time):
- `cpp` — cjpegli `-q Q` 4:2:0 baseline
- `zen_default` — `EncoderConfig::ycbcr` + sharp_yuv + deringing
- `zen_auto` — `+ auto_optimize(true)`  (hybrid trellis λ=14.5)
- `zen_auto_dc` — auto + DC trellis on
- `zen_xyb` — `EncoderConfig::xyb` + BQuarter
- `zen_xyb_bN.NN` — xyb with the **B-component base-quant table multiplied by N**

`%vs_cpp` is size delta vs cjpegli; `Δssim` is SSIM2 delta vs cjpegli (positive = better).

## Decoder bug, fixed

The 2026-04-14 run scored every XYB row at ~-60 SSIM2 because `ssim2()` decoded
through `zenjpeg_bench_utils::decode_jpeg_to_rgb`, which is a thin wrapper over
zune-jpeg. zune does **not** apply the embedded XYB ICC profile, so the decoded
RGB was the raw inverse-YCbCr of XYB-encoded coefficients — visually garbage,
ssim2 deeply negative.

Fix: switch the harness to `zenjpeg_bench_utils::decode_jpeg_with_icc`, which
uses `zenjpeg::decoder::Decoder::new().correct_color(Some(TargetColorSpace::Srgb))`.
After the fix XYB scores land in the normal 50–87 range and `zen_xyb` is in
fact a strong RD performer on photos and frymire (see headline below).

There's no bug in the decoder; the bug was the harness picking the wrong decode
path. zune-jpeg behaviour is correct for non-XYB JPEGs and matches its
documented "does not apply ICC" contract. The `decode_jpeg_with_icc` helper
already exists in `zenjpeg-bench-utils/src/lib.rs:1311` for exactly this case
and explicitly warns about XYB.

## Headline: B-channel coarsening is not a Pareto win

Coarsening only the B (blue-yellow) base-quant table beyond the jpegli default
loses SSIM2 much faster than it saves bytes, on every (image class, Q) pair.

Mean-of-class numbers across the sweep:

### Photos (6 CID22-512)

| Q  | config        | bytes | ssim2 | %vs_cpp | Δssim |
|----|---------------|-------|-------|---------|-------|
| 85 | cpp           | 47213 | 79.31 |  +0.0%  | +0.00 |
| 85 | zen_xyb       | 46397 | 77.72 |  -1.7%  | -1.59 |
| 85 | zen_xyb_b1.25 | 46214 | 76.85 |  -2.1%  | -2.47 |
| 85 | zen_xyb_b1.50 | 46095 | 76.13 |  -2.4%  | -3.18 |
| 85 | zen_xyb_b2.00 | 45924 | 74.79 |  -2.7%  | -4.53 |

Going from `zen_xyb` → `zen_xyb_b1.50` saves 0.7% size and costs 1.6 SSIM2 —
about 0.45 SSIM2 per 0.1% saved. Worse than dropping Q.

### Graphics (3 gb82-sc)

| Q  | config        | bytes  | ssim2 | %vs_cpp | Δssim |
|----|---------------|--------|-------|---------|-------|
| 85 | zen_xyb       | 109585 | 82.82 |  -7.4%  | +0.93 |
| 85 | zen_xyb_b1.25 | 109384 | 81.64 |  -7.5%  | -0.25 |
| 85 | zen_xyb_b1.50 | 109250 | 80.50 |  -7.7%  | -1.39 |
| 85 | zen_xyb_b2.00 | 108984 | 79.08 |  -7.9%  | -2.81 |

Same pattern, worse: 0.2% size saved per 1.2 SSIM2 lost.

### frymire (1118×1105 screenshot)

| Q  | config        | bytes  | ssim2 | %vs_cpp | Δssim |
|----|---------------|--------|-------|---------|-------|
| 85 | zen_xyb       | 480943 | 61.88 |  -2.7%  | +11.43 |
| 85 | zen_xyb_b1.25 | 478019 | 60.72 |  -3.3%  | +10.27 |
| 85 | zen_xyb_b1.50 | 475919 | 59.56 |  -3.7%  | +9.11 |
| 85 | zen_xyb_b2.00 | 473122 | 57.26 |  -4.3%  | +6.81 |

frymire is the only place where coarsening flirts with break-even, but `zen_xyb`
already beats cjpegli by 2.7% size and +11.4 SSIM2; coarsening B just gives
back quality for trivial size.

## Why the size lever is so weak

1. **B is already subsampled**. `XybSubsampling::BQuarter` halves B in each
   axis (4:2:0 layout). High-frequency B is mostly eliminated before
   quantisation.
2. **B base-quant is already large vs X/Y in jpegli's table**. Doubling it
   pushes most surviving AC bins to zero across the spectrum, so further
   coarseness mostly destroys low-frequency B (which IS visible) without
   freeing many bits.
3. **The bytes saved are dominated by Y entropy, not B**. Even on photos at
   Q50 where B carries more information, scaling the B table 3× saves <0.5%
   of the file.

A targeted experiment (coarsen *only* high-frequency B AC, leave DC and
low-frequency AC alone) might do better in principle, but the absolute
size headroom in B is so small it's unlikely to clear measurement noise.

## Cross-check vs prior run

`auto_optimize` numbers reproduce the 2026-04-14 conclusions:
- graphics: 12-23% smaller than cjpegli at matched-or-better SSIM2
- photos: +1-2 SSIM2 at ≈matched size
- DC trellis still null (≤0.05 SSIM2, ≤0.1% size delta vs auto)

`zen_xyb` (with the decode fix) is now visible as a real RD competitor:
- frymire: -2.7% size and +11.4 SSIM2 vs cjpegli at Q85
- graphics Q95: -8.3% size and +1.6 SSIM2 vs cjpegli
- photos: ~equal-or-worse on both axes — XYB is not preferred on photos here

## Conclusion

1. **Don't ship a B-coarsening knob.** No factor in {1.25 … 3.0} produced a
   Pareto-win on any image class.
2. **The harness fix matters independently.** Future RD work that touches XYB
   must decode through `decode_jpeg_with_icc` (or any decoder that honours the
   embedded ICC), not `decode_jpeg_to_rgb`. `decode_jpeg_to_rgb` is now
   `#[deprecated]` in `zenjpeg-bench-utils` (commit follows this writeup).
3. **Open follow-up:** if there's interest in B-side wins, target high-frequency
   B AC bins specifically (per-coefficient quant scaling via `scale_coeff`) and
   measure with butteraugli alongside SSIM2 — SSIMULACRA2 may be over-weighting
   chroma here for graphics.

## Second bug found: `XybSubsampling::Full` is silently ignored

A follow-up sweep added a `zen_xyb444` baseline (`EncoderConfig::xyb(q,
XybSubsampling::Full)`) plus a parallel B-coarseness sweep at full-B
resolution. **Every (image, Q, factor) row produced byte-identical and
ssim2-identical output to its `BQuarter` sibling.** CSV:
`benchmarks/rd_explore_xyb444_2026-04-15.csv` (720 rows).

That's not user error — the encoder discards the `XybSubsampling` enum
on the way to the bitstream:

- `zenjpeg/src/encode/serialize.rs:332-344` — `write_frame_header_xyb_ex` writes
  the SOF entries for R/G/B with hard-coded sampling factors `0x22, 0x22, 0x11`.
  No branch on `subsampling`. The CLAUDE.md TODO list calls this out as
  "low priority, always correct for XYB" — but it is *not* "always correct"
  when the public API exposes `XybSubsampling::Full` as a buildable variant.
- `zenjpeg/src/encode/layout.rs:164-171` — `LayoutParams` hard-sets `v_samp = 2`
  whenever `use_xyb`, ignoring the chroma-subsampling input. The XYB enum
  never reaches this layer either.

So `EncoderConfig::xyb(q, XybSubsampling::Full)` builds a config that
*claims* full B but produces a BQuarter bitstream. There is no warning,
no error, no debug assertion. The 4:4:4 XYB rows in this CSV exist
purely as evidence of the bug — they have no independent RD signal.

**Recommended next steps (not done in this commit):**
- Either implement real `XybSubsampling::Full` (R/G/B all `0x11`,
  layout `v_samp=1`, no B downsampler) — straightforward but needs a
  full encode/decode roundtrip path test
- Or remove `XybSubsampling::Full` from the public API and make
  `BQuarter` the only variant
- Update the CLAUDE.md TODO line and the "## Planned Features / Remaining
  Hardening" entry to flag this as a behavioural divergence from the docs,
  not just a cosmetic hardcode.

The B-coarseness conclusion above is unaffected: at 4:2:0 (which is what
both code paths actually emit), no factor in {1.25 … 3.0} produced an RD
win. Whether 4:4:4 XYB *would* have produced a win cannot be answered
until the layout bug is fixed.
