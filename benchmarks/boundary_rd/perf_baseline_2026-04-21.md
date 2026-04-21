# boundary_rd perf baseline — 2026-04-21

Honest baseline on a stratified 50-image sample from the lineart-focused
1,375-image corpus at `~/work/coefficient/scripts/selector_corpus/lineart/
zenjpeg_tuning_corpus_gpt.txt`. Seed 42, SplitMix64, 4 GPT categories
balanced, images scaled to max-side=768 MCU-aligned, Q=75 4:2:0, 5
iterations per config per image, median.

The earlier +46.8% number in `simd_overhead_2026-04-20.md` was measured
on `cid22:5` (photo content). Photos are the cheap case: refinement
rarely fires because `D_b` stays below threshold for smooth blocks. The
lineart/illustration content is the DESIGNED-FOR case, and its overhead
is substantially higher.

## Wall-clock overhead (overall)

- **Overall median: +60.4%**
- **p95: +81.9%**

## Per-category wall-clock (N=50 total, 11–13 per cell)

| Category | N | Median overhead | p95 overhead |
|---|---:|---:|---:|
| screen_ui | 13 | +61.0% | +79.2% |
| screen_chart | 13 | +51.2% | +77.2% |
| screen_document | 13 | +55.1% | +75.7% |
| illustration | 11 | +63.3% | +82.8% |

Raw per-image data: `perf_baseline_2026-04-21.csv`

## Callgrind (1 representative image, 30 iters)

Image: `/mnt/v/output/corpus-builder/wikimedia-webshapes/Illustration/
2e573cf50eda55867b119c5c010973e7c097f4f1.png` (screen_ui, 768×768).

- **OFF total: 1,842 M Ir**
- **ON  total: 2,373 M Ir**
- **Delta:  +531 M Ir (+28.8%)**

### Top functions by Ir delta (ON minus OFF)

Ir in megacounts. Full exclusive-Ir listings in `callgrind_baseline_*_2026-04-21.txt`.

| ΔIr (M) | OFF (M) | ON (M) | Function |
|---:|---:|---:|---|
| **192.97** | 0.00 | 192.97 | `StripProcessor::quantize_prev_pending_imcu` source line attribution → the inline refinement body in `quantize_y_with_boundary_rd_impl` (boundary_rd.rs source file) |
| **65.28** | 0.00 | 65.28 | `inverse_dct_8x8` core::num::f32 helper |
| **51.58** | 0.00 | 51.58 | `decode::idct::inverse_dct_8x8` direct body |
| **48.68** | 0.00 | 48.68 | `inverse_dct_8x8` slice iter macros |
| **25.16** | 0.57 | 25.73 | `quantize_y_with_boundary_rd_impl` (strip/mod.rs source) |
| **22.42** | 21.20 | 43.61 | `mage_quantize_block_zigzag_v3` — +22M from ~1 extra retry quantize per triggered block |
| **10.78** | 0.00 | 10.78 | `inverse_dct_8x8` int_macros helper |
| **10.68** | 0.00 | 10.68 | `inverse_dct_8x8` slice index helper |
| **10.50** | 0.00 | 10.50 | `inverse_dct_8x8` ptr::non_null helper |
| **8.28** | 0.00 | 8.28 | `inverse_dct_8x8` cmp helper |
| **7.40** | 96.41 | 103.81 | memcpy (slightly more buffer copies in ON) |
| **7.21** | 0.74 | 7.95 | simd_types quantize_block_zigzag from retry path |
| **6.43** | 6.08 | 12.52 | Another inverse_dct_8x8 path |
| **5.05** | 0.00 | 5.05 | `boundary_distortion` outer function (scalar parts + dispatch) |
| **4.74** | 0.38 | 5.13 | core helper for retry quantize |
| **3.90** | 3.69 | 7.59 | `simd_types::quantize_block_zigzag` (retry-induced +3.9M) |
| **3.41** | 0.00 | 3.41 | `boundary_rd::idct_reference_block` outer |
| **3.12** | 0.00 | 3.12 | archmage token (summon) |
| **2.50** | 0.00 | 2.50 | `__arcane_mage_boundary_distortion` |
| **2.44** | 0.00 | 2.44 | `__arcane_mage_scale_block_x8` |

### Where the 531M Ir goes (attribution summary)

Grouping by what code is actually responsible:

| Bucket | Ir (M) | % of delta |
|---|---:|---:|
| `inverse_dct_8x8` (all helpers rolled up) | ~195 | **37%** |
| Body of `quantize_y_with_boundary_rd_impl` (Option/Vec/copy noise) | ~218 | **41%** |
| Retry `quantize_with_zero_bias_zigzag` | ~34 | **6%** |
| Boundary-distortion math (incl. archmage arcane dispatch) | ~9 | **2%** |
| `idct_reference_block` x8 scaler + IDCT wrapper | ~6 | **1%** |
| misc (memcpy, page fault noise, decode bits) | ~70 | **13%** |
| **Total** | **531** | **100%** |

## Interpretation — where the budget goes

1. **IDCT dominates.** Every block does *two* IDCTs (`idct_reference_block`
   + `idct_quantized_block`) on the non-retry path, and **one additional IDCT
   per retry**. At Q75 on lineart/illustration, refinement triggers
   frequently, so the average is ~2.3 IDCTs/block. The IDCT is already
   archmage-backed, but its body costs ~8M Ir per direct call — multiplied
   by 2× the block count of the image.

2. **Inline body overhead is 40%.** `quantize_y_with_boundary_rd_impl`
   itself is attributing 218M Ir to its source-line range. That's Option
   tag branches on `Vec<Option<[f32;8]>>`, `copy_from_slice` on edges,
   the scalar `left_edge_col`/`right_edge_col`/`top_edge_row`/
   `bottom_edge_row` loops, and `reduce_add` of the SIMD result back to
   a scalar.

3. **The recompute-left-neighbor anti-pattern.** In `quantize_y_with_
   boundary_rd_impl` at strip/mod.rs:1676, every block with `bx > 0`
   recomputes `br::idct_reference_block(&prev_dct)` — the IDCT of the
   PREVIOUS block, whose right edge we already computed when processing
   that block! This alone accounts for ~50% of the reference-IDCT work.
   The earlier agent explicitly flagged this and deferred the fix.

4. **Retry quantize is cheap.** Only ~34M Ir across all retries for a
   768×768 image at Q75. Not a primary target.

5. **The SIMD kernels are clean.** `__arcane_mage_boundary_distortion`
   is 2.5M, `__arcane_mage_scale_block_x8` is 2.4M, `__arcane_mage_ac_
   dct_energy` is below the 100k threshold. The prior SIMD pass did its
   kernel job; the remaining cost is architectural, not kernel.

## Planned phase-2 targets (in priority order)

1. **Cache prev-block's `ref_block` edges** — eliminate the redundant
   `idct_reference_block(&prev_dct)` at strip/mod.rs:1676. The previous
   block's right-edge ORIG column was already computed when that block
   was processed. Store it in an additional entry alongside
   `state.left_edges[bx]`. Expected saving: ~50% of the reference IDCTs
   = ~50M Ir per 768×768 = ~10% of the 531M delta.

2. **Eliminate `Vec<Option<[f32;8]>>` branches.** Replace with a pair
   of `Vec<[f32; 8]>` + a `Vec<u8>` validity bitmap, or a plain
   `Vec<[f32; 8]>` that's cleared to a known sentinel at row boundaries.
   Expected saving: ~5-10M Ir from eliminating the Option tag+branch.

3. **SIMD the edge extraction.** `left_edge_col` / `right_edge_col` /
   `top_edge_row` / `bottom_edge_row` are scalar loops. Each is
   trivially SIMD-able with an `i32gather_ps` or a `load+permute`. But
   the edge-extraction functions themselves aren't visible in the top-20
   — they get inlined into the callsite. Still worth checking what
   assembly LLVM emits and whether vectorizing them helps.

4. **Inline-merge `idct_reference_block` inputs.** The `mage_scale_block_x8`
   packs `Block8x8f.rows[8][8]` into `[f32; 64]`. If we move the
   IDCT to consume `Block8x8f` directly, we skip an 8×f32x8 load/store
   round-trip. Modest saving, but easy.

5. **Reduce the refinement decision to a single SIMD kernel.** Currently
   `boundary_distortion` is called once, then `ac_dct_energy` is called
   separately. Fusing the two shares register state.

## Methodology notes

- CPU: AMD Ryzen 9 7950X, Zen 4, WSL2.
- Compiler: `cargo build --release` (no `target-cpu=native` — runtime
  archmage dispatch).
- Features: `trellis` only (no parallel, no decoder — just raw encode
  path).
- Corpus: 1,375-image zenjpeg_tuning_corpus_gpt.txt (screen_ui 190,
  screen_chart 224, screen_document 383, illustration 578).
- Stratified sample: 13+13+13+11 per category, seed 42 SplitMix64.
- Images scaled max-side=768 Triangle, then cropped to MCU-aligned
  dimensions. Min MCU-aligned = 64×64.
- Encoding: `EncoderConfig::ycbcr(75.0, Quarter)` with
  `boundary_rd(Off)` vs `boundary_rd(On(default))`.
- Per-image: 1 warmup each, then 5 measured OFF and 5 measured ON,
  interleaved per-iteration. Median taken within-image.
- Reproduce: `cargo run --release -p zenjpeg --features trellis --example
  boundary_rd_perf_bench -- --sample 50 --seed 42 --iters 5 --max-side
  768 --tag baseline`.

Callgrind reproduction:

```
cargo build --release -p zenjpeg --features trellis --example boundary_rd_callgrind
IMG='/mnt/v/output/corpus-builder/wikimedia-webshapes/Illustration/\
2e573cf50eda55867b119c5c010973e7c097f4f1.png'
valgrind --tool=callgrind --callgrind-out-file=/tmp/cg_off.out --quiet \
    ./target/release/examples/boundary_rd_callgrind off "$IMG" 30
valgrind --tool=callgrind --callgrind-out-file=/tmp/cg_on.out --quiet \
    ./target/release/examples/boundary_rd_callgrind on  "$IMG" 30
callgrind_annotate --inclusive=no --threshold=100 --auto=no /tmp/cg_off.out \
    > benchmarks/boundary_rd/callgrind_baseline_off_2026-04-21.txt
callgrind_annotate --inclusive=no --threshold=100 --auto=no /tmp/cg_on.out \
    > benchmarks/boundary_rd/callgrind_baseline_on_2026-04-21.txt
```
