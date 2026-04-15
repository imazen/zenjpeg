# RD exploration results, 2026-04-14

Harness: `zenjpeg/examples/rd_explore.rs`
CSV: `benchmarks/rd_explore_2026-04-14.csv`
Corpus: 1 frymire + 6 CID22-512 photos + 3 gb82 graphics = 10 images.
Q levels: 50, 70, 85, 95. All 4:2:0 progressive mode for zen_auto*, baseline for zen_default.

All sizes as mean bytes, SSIM2 mean across images. `%vs_cpp` = size delta vs cjpegli.
`Δssim` = SSIM2 delta vs cjpegli (positive = better).

## Graphics (codec_wiki, gmessages, graph)

| Q  | config      | bytes    | ssim2 | %vs_cpp | Δssim |
|----|-------------|----------|-------|---------|-------|
| 50 | cpp         |  75089   | 71.16 | +0.0%   | +0.00 |
| 50 | zen_default |  74827   | 71.42 | -0.3%   | +0.27 |
| 50 | zen_auto    |  60613   | 70.92 | **-19.3%**  | -0.23 |
| 70 | cpp         |  93013   | 77.03 | +0.0%   | +0.00 |
| 70 | zen_default |  92928   | 77.27 | -0.1%   | +0.24 |
| 70 | zen_auto    |  76701   | 77.07 | **-17.5%**  | +0.05 |
| 85 | cpp         | 118306   | 81.89 | +0.0%   | +0.00 |
| 85 | zen_default | 118164   | 82.17 | -0.1%   | +0.29 |
| 85 | zen_auto    | 102809   | 82.52 | **-13.1%**  | +0.63 |
| 95 | cpp         | 166435   | 85.68 | +0.0%   | +0.00 |
| 95 | zen_default | 166217   | 86.14 | -0.1%   | +0.47 |
| 95 | zen_auto    | 149439   | 86.34 | **-10.2%**  | +0.67 |

## Frymire (1118x1105 screenshot)

| Q  | config      | bytes   | ssim2 | %vs_cpp | Δssim |
|----|-------------|---------|-------|---------|-------|
| 50 | cpp         | 269545  | 36.66 | +0.0%   | +0.00 |
| 50 | zen_default | 271843  | 41.37 | +0.9%   | +4.71 |
| 50 | zen_auto    | 250720  | 40.33 | **-7.0%**   | +3.67 |
| 70 | cpp         | 362135  | 44.95 | +0.0%   | +0.00 |
| 70 | zen_default | 363377  | 48.93 | +0.3%   | +3.98 |
| 70 | zen_auto    | 348165  | 49.48 | **-3.9%**   | +4.53 |

## Photos (6 CID22 images)

| Q  | config      | bytes | ssim2 | %vs_cpp | Δssim |
|----|-------------|-------|-------|---------|-------|
| 50 | cpp         | 23764 | 64.21 | +0.0%   | +0.00 |
| 50 | zen_default | 23706 | 64.44 | -0.2%   | +0.23 |
| 50 | zen_auto    | 24010 | 65.56 | +1.0%   | **+1.35** |
| 85 | cpp         | 47213 | 79.27 | +0.0%   | +0.00 |
| 85 | zen_default | 47127 | 79.68 | -0.2%   | +0.41 |
| 85 | zen_auto    | 48039 | 81.41 | +1.8%   | **+2.14** |

## Key findings

1. **`auto_optimize(true)` is the biggest opportunity.** Already implemented, well-validated, but OFF by default. On graphics, 10-19% size reduction at matched or better SSIM2. On photos, +1.0-2.1 SSIM2 at +0.2-1.8% size (Pareto-dominant). On frymire, 2-7% smaller at +3.4 to +4.5 Δssim.

2. **DC trellis is null.** `zen_auto_dc` (hybrid + dc_enabled=true) differs from `zen_auto` by <0.05 SSIM2 and <0.1% size across all 40 (image, Q) rows. Leave it off.

3. **XYB output scored ~-60 SSIM2.** Decoder mismatch likely (ICC handling or comparator assumes sRGB). Not a proven intervention here — needs separate investigation of the XYB decode path before claiming anything.
