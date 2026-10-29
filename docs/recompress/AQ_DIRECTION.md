# Why recompression AQ targets FLAT blocks, not busy ones

A natural objection: jpegli, mozjpeg, and every modern from-scratch JPEG
encoder use *adaptive quantization* (AQ) that gives **busy/detailed
blocks coarser quantization** — the human visual system masks distortion
in high-activity regions, so you can spend fewer bits there. So shouldn't
recompression AQ also coarsen the busy blocks and protect the flat ones?

**No — and the measurement proves it.** Recompression-via-requantization
is a fundamentally different regime from encoding-from-pixels.

## The measurement

`benchmarks/aq_direction_10refs_2026-05-28.tsv` — 10 CID22 references ×
3 source qualities × 3 targets × 4 AQ variants, all at the **same**
uniform quant scale (so only the AQ block-selection differs):

| variant | block selection | AC zeroed |
|---|---|---|
| `none` | — | none (control) |
| `flat_t48` | flat blocks (high/low AC energy ≤ 8 %) | 48..64 |
| `busy_t48` | busy blocks (ratio > 25 %) | 48..64 |
| `busy_t32` | busy blocks | 32..64 |

Size saved and quality cost vs the `none` control, averaged per source
quality:

| source_q | variant | size saved | quality cost (zensim-A) | efficiency |
|---|---|---|---|---|
| 90 | flat_t48 | +0.0073 | −0.155 | **0.047** |
| 90 | busy_t48 | +0.0001 | −0.014 | ~0 |
| 90 | busy_t32 | +0.0015 | −0.421 | 0.004 |
| 75 | flat_t48 | +0.0088 | −0.223 | **0.039** |
| 75 | busy_t32 | +0.0014 | −0.231 | 0.006 |
| 60 | flat_t48 | +0.0033 | −0.037 | **0.089** |
| 60 | busy_t32 | +0.0012 | −0.030 | 0.040 |

Flat-targeting is **2–13× more byte-efficient per unit quality** at every
source quality, including q90 where requantization is gentlest — the best
case for the masking theory. The theory still loses.

## Why the masking intuition doesn't transfer

The masking principle is about **where to spend a fixed bit budget when
encoding from pixels**. It assumes you have the full-precision DCT
coefficients and are deciding how coarsely to quantize each block.

In recompression, the **quant-table change already does the
masking-style work**:

1. To hit a lower target quality, Preserve scales the quant tables up
   (see `build_new_table`). The high-frequency quant divisors become
   large (50–200+).
2. A **busy block** has large high-frequency coefficients — but dividing
   them by the now-large high-freq quant divisors sends them to **zero
   anyway**. The requantization already shed the busy block's
   high-freq energy.
3. So a busy-targeting AQ pass finds **almost nothing left to remove**
   (`busy_t48` saves +0.0001 ratio — noise). When it reaches lower into
   the spectrum (`busy_t32`) to find surviving energy, that energy is
   structural mid-frequency texture that zensim *does* penalize
   (−0.42 zensim-A at q90) for almost no size gain.
4. A **flat block** retains small high-frequency coefficients that
   survive requantization (grain, subtle gradient structure paired with
   the small low-freq quant positions). Those residuals cost
   run-length-coding bits. Flat-targeting mops them up cheaply — real
   size saving at low perceptual cost.

In short: **requantization is the recompression analog of jpegli's
busy-block coarsening.** It already coarsens the busy blocks. AQ's
residual, complementary job is cleaning up the small high-freq that
survives in flat blocks — which is exactly the opposite block selection
from from-scratch AQ, and exactly right for this regime.

## Caveat

This finding is specific to:
- **The recompression regime** (source is already a JPEG; we requantize
  its coefficients rather than re-encode from pixels).
- **The zensim Profile A metric.** A different perceptual metric that
  models contrast masking more strongly might rate busy-block texture
  loss as cheaper. The conclusion holds for our target metric; it should
  be re-checked if the target metric changes.

The production AQ (`aq::build_aq_mask`) therefore targets flat blocks,
gated by quality headroom (see `aq.rs` module docs). The
`build_aq_mask_busy` function is retained under the `expert` feature for
reproducing this experiment, not for production use.
