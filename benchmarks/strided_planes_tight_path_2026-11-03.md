# Tight-path cost of strided planes in zenyuv and zenjpeg (2026-11-03)

Snapshot, not a live claim. Host r7900x (Ryzen 9 7900X, Zen 4); **not measured on
Zen 5**. No `-C target-cpu=native`. Base `762fa712`. RSS not measured here.

## Interleaved A/B: the tightly packed path, final form

A small example built twice with `--features _dev`, against the base's tight API and
the final single form with tight views. Each run is one process taking the best of N
calls; base and change alternate for 21 runs per cell, and the table takes the median
and the minimum of those. Data: x-y-indexed byte pattern, not a photo. The `_dev`
feature enables archmage's testable dispatch on both sides, so `summon()` is slower
than in a normal build by the same amount on each.

| op width height N tier | base median | change median | ratio | min ratio |
|---|---|---|---|---|
| 420 1920 1080 12 scalar | 4723.16us | 4259.27us | 0.9018 | 0.9000 |
| 444 1920 1080 12 scalar | 5991.12us | 3598.24us | 0.6006 | 0.5979 |
| yonly 1920 1080 12 scalar | 1270.28us | 1285.16us | 1.0117 | 1.0270 |
| 420 1920 1080 40 v3 | 368.72us | 372.36us | 1.0099 | 1.0210 |
| 444 1920 1080 40 v3 | 485.02us | 484.69us | 0.9993 | 1.0022 |
| yonly 1920 1080 40 v3 | 267.65us | 267.46us | 0.9993 | 1.0259 |
| sharp 1920 1080 8 v3 | 4899.32us | 4925.62us | 1.0054 | 1.0168 |
| d420 1920 1080 8 v3 | 5519.04us | 5430.08us | 0.9839 | 0.9779 |
| d420 1920 1080 8 scalar | 11816.12us | 11632.98us | 0.9845 | 0.9813 |
| d444 1920 1080 8 v3 | 3171.85us | 3179.79us | 1.0025 | 1.0060 |
| d444 1920 1080 8 scalar | 13722.60us | 13681.58us | 0.9970 | 0.9958 |
| d422 1920 1080 8 v3 | 7258.71us | 7222.65us | 0.9950 | 0.9986 |
| dbil 1920 1080 3 v3 | 21172.16us | 21190.10us | 1.0008 | 1.0054 |
| d400 1920 1080 8 v3 | 1106.54us | 1107.94us | 1.0013 | 1.0149 |
| 420 64 64 4000 scalar | 9.22us | 8.38us | 0.9097 | 0.9047 |
| 420 64 64 4000 v3 | 0.73us | 0.74us | 1.0137 | 1.0139 |
| 444 64 64 4000 v3 | 0.95us | 0.96us | 1.0105 | 1.0106 |
| 420 33 27 8000 v3 | 0.46us | 0.48us | 1.0412 | 1.0217 |
| sharp 64 64 2000 v3 | 12.86us | 12.98us | 1.0094 | 1.0143 |
| d420 64 64 2000 v3 | 11.29us | 10.73us | 0.9504 | 0.9493 |
| 420 4096 2304 12 v3 | 2131.55us | 2145.24us | 1.0064 | 0.9823 |

`d*` rows are decode kernels, reachable only under `_dev`. Reading: at 1920x1080 and
4096x2304 in the v3 tier the median moved by -1.6% to +1.0%. Very small images in v3
carry a fixed per-call cost of about 10 to 20 ns from the geometry checks: +1.4% at
64x64 and +4.1% at 33x27 on 4:2:0. The scalar tier got faster: 4:4:4 by 40% and
4:2:0 by 10%. The 4:2:0 gain comes from bounding each chunk with a slice. The 4:4:4
one was not traced. Causes of the remaining sub-percent movements were not
investigated.

## zenbench: existing benches, before and after

`cargo bench -p zenyuv --bench rgb_to_yuv_bench -- --group=<g>` and
`cargo bench -p zenyuv --bench kernel_tiers --features _dev -- --group=<g>`, one process
per group per tree, sequentially rather than interleaved across trees, and each
group got only 4 rounds. These runs are noisy: trust the A/B above where they differ.
"Change" is `863edc2a`, before the chunk slicing and the single form. "Final" is
`95a0988e`, after the chunk slicing but before the single form, rerun only on the two
groups that had regressed.

| group | base min | base mean | change min | change mean |
|---|---|---|---|---|
| rgb_to_yuv_444_256 | 16.90µs | 18.57µs | 16.84µs | 16.96µs |
| rgb_to_yuv_444_1024 | 272.87µs | 284.67µs | 274.88µs | 277.81µs |
| rgb_to_yuv_420_256 | 12.81µs | 15.55µs | 12.73µs | 13.44µs |
| rgb_to_yuv_420_1024 | 213.60µs | 222.32µs | 212.37µs | 223.39µs |

| group | tier | base min | base mean | change min | change mean | final min | final mean |
|---|---|---|---|---|---|---|---|
| encode_420_u8 | v3 | 413.05µs | 678.16µs | 419.74µs | 718.04µs | 425.15µs | 693.37µs |
| encode_420_u8 | scalar | 5.25ms | 5.28ms | 5.57ms | 5.78ms | 4.74ms | 4.81ms |
| encode_444_u8 | v3 | 553.90µs | 735.43µs | 555.23µs | 686.44µs | not rerun |  |
| encode_444_u8 | scalar | 6.66ms | 6.68ms | 4.08ms | 4.12ms | not rerun |  |
| encode_sharp_420_u8 | v3 | 4.22ms | 4.42ms | 4.23ms | 4.43ms | not rerun |  |
| encode_sharp_420_u8 | scalar | 16.32ms | 16.55ms | 16.18ms | 16.36ms | not rerun |  |
| yuv420_to_rgb | v3 | 6.20ms | 6.32ms | 6.43ms | 6.47ms | 5.92ms | 5.93ms |
| yuv420_to_rgb | scalar | 13.09ms | 13.32ms | 13.11ms | 13.24ms | 12.76ms | 12.83ms |
| yuv444_to_rgb | v3 | 3.52ms | 3.63ms | 3.52ms | 3.65ms | not rerun |  |
| yuv444_to_rgb | scalar | 15.32ms | 15.47ms | 15.40ms | 15.70ms | not rerun |  |

## zenbench: `push_subsampled`

`cargo bench -p zenjpeg --bench ycbcr_planar_push`, min / mean, both trees at their own
code. Base's strided lane is the old path with three allocations per call, at 4 rounds.

| size | tight, base | tight, change | strided, base | strided, change |
|---|---|---|---|---|
| 64x64 | 117.71µs / 124.67µs | 117.47µs / 125.35µs | 117.98µs / 124.48µs | 118.04µs / 121.85µs |
| 1024x1024 | 23.13ms / 23.32ms | 23.10ms / 23.28ms | 23.13ms / 23.32ms | 22.96ms / 23.22ms |
