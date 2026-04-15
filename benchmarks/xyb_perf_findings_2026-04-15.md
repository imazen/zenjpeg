# XYB encode performance findings, 2026-04-15

After implementing `XybSubsampling::Full` end-to-end, the user asked to
"make XYB fast to encode". This doc records the perf paths investigated
and why none of them yielded a real speedup beyond what's already in tree.

## Baseline (Ryzen 9 7950X, this branch, zenbench `encode_q85_1k_xyb`)

| Config | Mean | vs YCbCr 4:2:0 progressive |
|--------|------|----------------------------|
| YCbCr 4:2:0 progressive | 22.3 ms | baseline |
| YCbCr 4:2:0 baseline +Optimize (buffered) | 9.4 ms | **-58%** |
| YCbCr 4:2:0 baseline +Fixed (streaming-through) | 10.3 ms | -54% |
| XYB BQuarter progressive | 28.9 ms | +30% |
| XYB Full progressive | 34.0 ms | +52% |
| XYB BQuarter baseline | 13.9 ms | -38% |
| XYB Full baseline | 14.6 ms | -35% |
| XYB Full baseline +parallel | 14.6 ms | -35% |
| XYB Full progressive +parallel | 33.9 ms | +52% |

**Headline:** XYB baseline is ~2× faster than XYB progressive. XYB Full
baseline (14.6 ms) is *faster* than YCbCr 4:2:0 progressive (22.3 ms).
The "XYB is slow" reputation is the progressive 2-pass cost, not XYB.

## (b) Profile — where does XYB time go?

`profile_xyb` example + `valgrind --tool=callgrind` on XYB Full
progressive 1024² × 30 iters:

| Function | % of XYB Full insns |
|----------|---------------------|
| `tokenize_ac_refinement_scan` | 14.6% |
| `write_ac_refinement_tokens` | 12.7% (sum across files) |
| `tokenize_ac_first_scan` | 6.7% |
| `srgb_to_scaled_xyb_planes_rgb` | 3.7% |
| `write_ac_first_tokens` | 3.5% (sum) |
| `forward_dct_8x8_wide` | 0.9% |
| `mage_build_refine_masks` | 1.3% |

Progressive entropy (tokenize_ac_* + write_ac_*) is ~40% of cycles.
Color conversion is 3.7%, DCT is <1%.

**Comparison:** YCbCr 4:2:0 progressive uses 9.4B insns; XYB Full uses
14.75B insns. Ratio 1.57× matches the wall-clock gap. Pixel-proportional
to the component count: XYB Full has 3× full-size component planes vs
YCbCr 4:2:0's 1 + 0.25 + 0.25 = 1.5×. The 1.57/1.5 ratio is within
Huffman-table-construction overhead.

There is no XYB-specific bug to fix. Color conversion, DCT, and the XYB
ICC machinery are already cheap in absolute terms.

## (a) Streaming-through for XYB — abandoned

The streaming-through path (`enable_streaming = !Optimize && !Progressive
&& !use_xyb`) skips the buffered "encode-everything-then-output" pass and
emits each strip as it's ready.

To enable it for XYB requires duplicating the XYB header sequence
(write_header_xyb / write_app14_adobe / write_icc_profile /
write_quant_tables_xyb / write_frame_header_xyb_ex / either an XYB-style
2-table or YCbCr-style 4-table Huffman layout / write_scan_header_xyb)
into the streaming-through code path that currently calls only the YCbCr
variants.

**Why abandoned:** the YCbCr streaming-through path is itself **slower**
than buffered+Optimize at 1024² (10.3 ms vs 9.4 ms — see baseline table).
The per-strip BitWriter overhead exceeds the saved buffer-then-encode
pass at this size. Implementing the XYB plumbing for a path that doesn't
even win on YCbCr would be ~hours of refactor for zero or negative win.

If a real speedup is wanted here, the right move is to fix the
streaming-through path's per-strip overhead first (probably by batching
multiple strips before flushing the BitWriter). After that, lifting the
`!use_xyb` gate becomes worthwhile.

## (c) Parallel feature for XYB — no-op

`--features parallel` enables `parallel_dct_y_blocks` for the X-component
DCT pass and the parallel entropy encode path in `blocks.rs::encode_blocks`.

Measured: 0% speedup at every (1024², 4K) × (progressive, baseline)
combination tested. Two reasons:

1. `parallel_dct_y_blocks` short-circuits to sequential when
   `total_blocks < PARALLEL_THRESHOLD = 4096` (`parallel.rs:120`). XYB
   Full strip_height=8, so blocks-per-strip = `blocks_w * 1` — 480 at
   4K-wide, 128 at 1024². Always below threshold.
2. The XYB entropy path runs `encode_sequential_xyb` (in `streaming.rs`)
   which doesn't go through the parallel-aware `blocks.rs` branch. Even
   at large image sizes, entropy stays serial for XYB.

What would actually help:

- **Multi-strip fan-out.** Process N strips concurrently. Architectural
  change (the StripProcessor today is single-threaded).
- **3-way component fan-out.** `pending.{y, cb, cr}` are disjoint mutable
  borrows; X/Y/B DCT loops have no inter-component dependencies and
  could run via `rayon::scope`. *But* DCT is 0.9% of total per the
  callgrind data above. A 3× speedup on 0.9% saves ~0.6%. Not worth the
  cross-thread coordination.
- **XYB entropy parallelization.** `tokenize_ac_*` is the dominant cost.
  Hooking the XYB path into the existing `parallel_entropy_encode_*`
  would need restart markers and (per the test suite) is a substantial
  XYB Huffman-layout change. Plausible but a separate project.

## What did move (already in this branch, prior commits)

- `XybSubsampling::Full` actually emits 4:4:4 XYB now (was silently
  hardcoded to BQuarter). +12 SSIM2 on screenshots, +3 on photos at
  ~+8% size. This is a quality lift, not a speed lift, but it's the
  RD knob users actually care about.
- XYB-Full B-channel handling: `mem::swap` instead of `copy_from_slice`
  in convert. Byte-equivalent, removes a per-strip memcpy. Speed delta
  is at noise level (the copy was a small fraction of total work).

## Recommended user guidance

**For minimum encode time at acceptable quality**, use XYB baseline:

```rust
EncoderConfig::xyb(85, XybSubsampling::Full).progressive(false)
```

That is **35-40% faster** than the default progressive path and still
beats YCbCr 4:2:0 progressive on quality. The default for XYB will stay
progressive (it gives 3-7% smaller files at the cost of speed), but
baseline should be the documented "fast XYB" recipe.

## Open follow-ups (none committed)

- Profile the per-strip BitWriter overhead in streaming-through. If
  batching N strips drops it below buffered+Optimize cost, lift the
  `!use_xyb` gate next.
- Investigate parallel entropy for XYB: the existing `parallel_entropy_encode_444`
  in `parallel.rs` is restart-marker-segmented; XYB's entropy path could
  plausibly be wired through it with a custom Huffman table layout.
- Lower `PARALLEL_THRESHOLD` after measuring rayon overhead at small
  block counts; current 4096 is conservative.
