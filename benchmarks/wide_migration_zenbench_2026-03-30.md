# wide→magetypes Migration Baseline (zenbench)

- **Date:** 2026-03-30
- **Commit:** 9415d7c6
- **CPU:** Ryzen 9 7950X (WSL2)
- **Tool:** zenbench (interleaved, paired statistics)
- **Command:** `cargo bench -p zenjpeg --bench wide_migration --features "decoder,trellis,parallel" -- --save-baseline=pre-migration`
- **Image:** noise+patches synthetic (deterministic LCG), Q90
- **Total run time:** 598.5s (1095 rounds)

## Encode

### Subsampling × size (RGB8, Q90, sequential)

| Group | base (ms) | prog (ms) | throughput |
|-------|----------|----------|-----------|
| enc/2k/444 | 140.6 ±3.4 | 140.1 ±2.8 | 85-86 MiB/s |
| enc/2k/422 | 120.8 ±8.4 | 119.8 ±7.1 | 99-100 MiB/s |
| enc/2k/420 | 88.1 ±1.3 | 88.2 ±1.9 | 136 MiB/s |
| enc/2k/440 | 102.4 ±1.3 | 102.0 ±1.7 | 117-118 MiB/s |
| enc/4k/444 | 599.5 ±10.8 | 597.8 ±12.6 | 80 MiB/s |
| enc/4k/422 | 421.6 ±2.3 | 422.8 ±4.4 | 114 MiB/s |
| enc/4k/420 | 347.4 ±4.8 | 349.2 ±2.7 | 137-138 MiB/s |
| enc/4k/440 | 423.4 ±7.7 | 427.8 ±5.3 | 112-113 MiB/s |

### Parallel encode (seq vs parallel, baseline)

| Group | seq (ms) | parallel (ms) | change |
|-------|---------|--------------|--------|
| enc/2k/444/par | 140.2 | 139.2 | ~0% |
| enc/2k/422/par | 102.5 | 102.8 | ~0% |
| enc/2k/420/par | 88.8 | 89.1 | ~0% |
| enc/2k/440/par | 100.8 | 101.3 | ~0% |
| enc/4k/444/par | 601.8 | 604.5 | ~0% |
| enc/4k/422/par | 418.3 | 420.7 | +1% |
| enc/4k/420/par | 350.1 | 351.0 | ~0% |
| enc/4k/440/par | 417.9 | 417.2 | ~0% |

Note: Parallel encode shows no benefit on this synthetic image. The parallel path
parallelizes DCT+quantize which is already fast relative to AQ + entropy coding.

### Pixel formats (2k, 420, baseline)

| Format | Time (ms) | vs rgb8 |
|--------|----------|---------|
| rgb8 | 88.5 ±2.1 | baseline |
| rgba8 | 89.4 ±2.0 | +1% |
| bgra8 | 88.4 ±1.1 | ~0% |
| rgb16 | 137.0 ±1.8 | +55% |
| rgbf32 | 144.1 ±1.3 | +63% |

### Grayscale (2k)

| Mode | Time (ms) |
|------|----------|
| gray8 base | 73.8 ±1.0 |
| gray8 prog | 73.9 ±0.9 |

### XYB (2k, baseline)

| Mode | Time (ms) |
|------|----------|
| xyb base | 181.0 ±1.8 |

### Optimization (2k, 420)

| Mode | Time (ms) | vs trellis |
|------|----------|-----------|
| trellis | 560.2 ±7.1 | baseline |
| auto_optimize | 552.3 ±8.3 | -4.5% |

## Decode

### Subsampling × size (default decoder, RGB8 output)

| Group | Time (ms) | throughput |
|-------|----------|-----------|
| dec/2k/444 base | 46.1 ±1.8 | 91 Mpx/s |
| dec/2k/444 prog | 46.0 ±1.5 | 91 Mpx/s |
| dec/2k/422 base | 32.7 ±2.1 | 128 Mpx/s |
| dec/2k/422 prog | 32.0 ±1.6 | 131 Mpx/s |
| dec/2k/420 base | 28.1 ±1.5 | 150 Mpx/s |
| dec/2k/420 prog | 28.0 ±1.6 | 150 Mpx/s |
| dec/2k/440 base | 31.5 ±1.2 | 133 Mpx/s |
| dec/2k/440 prog | 31.5 ±1.7 | 133 Mpx/s |
| dec/4k/444 base | 235.1 ±5.3 | 71 Mpx/s |
| dec/4k/444 prog | 241.3 ±6.0 | 70 Mpx/s |
| dec/4k/422 base | 162.4 ±4.6 | 103 Mpx/s |
| dec/4k/422 prog | 163.5 ±2.8 | 103 Mpx/s |
| dec/4k/420 base | 143.8 ±4.5 | 117 Mpx/s |
| dec/4k/420 prog | 144.0 ±3.0 | 117 Mpx/s |
| dec/4k/440 base | 159.6 ±3.4 | 105 Mpx/s |
| dec/4k/440 prog | 160.0 ±2.8 | 105 Mpx/s |

### Parallel decode (seq vs parallel, baseline only)

| Group | seq (ms) | parallel (ms) | speedup |
|-------|---------|--------------|---------|
| dec/2k/444 | 51.6 | 45.2 | **1.14x** |
| dec/2k/422 | 42.2 | 31.6 | **1.34x** |
| dec/2k/420 | 30.8 | 26.7 | **1.15x** |
| dec/2k/440 | 35.1 | 30.9 | **1.14x** |
| dec/4k/444 | 268.0 | 234.5 | **1.14x** |
| dec/4k/422 | 209.2 | 158.8 | **1.32x** |
| dec/4k/420 | 164.4 | 142.5 | **1.15x** |
| dec/4k/440 | 180.4 | 159.1 | **1.13x** |

### Wave-parallel scanline decode (420 baseline, box filter)

| Group | seq (ms) | wave (ms) | change |
|-------|---------|----------|--------|
| dec/2k/wave | 28.7 | 28.5 | ~0% |
| dec/4k/wave | 181.1 | 180.6 | ~0% |

Note: Wave parallel shows no benefit here — synthetic images with uniform content
don't create enough per-segment work variation for wave scheduling to help.

### IDCT methods (2k, 420 baseline)

| Method | Time (ms) | vs jpegli |
|--------|----------|----------|
| jpegli | 27.5 ±1.4 | baseline |
| libjpeg | 27.9 ±1.1 | +1% |

### Chroma upsampling (2k, 420 baseline)

| Method | Time (ms) | vs triangle |
|--------|----------|------------|
| triangle | 27.2 ±1.3 | baseline |
| nearest | 27.2 ±1.1 | ~0% |

### Deblock modes (2k, 420 baseline)

| Mode | Time (ms) | vs off |
|------|----------|--------|
| off | 27.9 ±1.5 | baseline |
| boundary4tap | 40.2 ±1.4 | +44% |
| knusperli | 123.3 ±2.7 | +342% |

### Output pixel formats (2k, 420 baseline)

| Format | Time (ms) | vs rgb8 |
|--------|----------|---------|
| rgb8 | 27.7 ±2.0 | baseline |
| rgba8 | 30.3 ±1.3 | +9% |
| bgr8 | 29.4 ±0.9 | +6% |
| bgra8 | 30.4 ±1.4 | +10% |
| gray8 | 59.9 ±1.7 | +116% |
| rgb16 | 59.5 ±1.8 | +115% |
| rgbf32 | 60.0 ±1.5 | +117% |

### Scanline vs full decode (2k, 420 baseline)

| Path | Time (ms) | vs full |
|------|----------|---------|
| full_decode | 27.4 ±1.4 | baseline |
| scanline_reader | 28.8 ±1.3 | +5% |

### Grayscale decode (2k)

| Path | Time (ms) |
|------|----------|
| gray → rgb | 47.5 ±1.0 |
| gray → gray | 46.4 ±1.3 |

### Dequant bias (2k, 420 baseline)

| Mode | Time (ms) | vs default |
|------|----------|-----------|
| default (i16 IDCT) | 27.6 ±1.5 | baseline |
| dequant_bias (f32 IDCT) | 103.1 ±2.0 | +274% |

## Notes

- zenbench interleaved measurement eliminates thermal drift between benchmarks
- Noise+patches image (not gradients) for realistic DCT coefficient distribution
- Parallel decode shows 1.13-1.34x speedup (best on 422 at 1.34x)
- Parallel encode shows no benefit (bottleneck is AQ + entropy, not DCT)
- Wave-parallel scanline shows no benefit on uniform synthetic images
- Gray/rgb16/rgbf32 decode ~2x slower — falls through to f32 decode path
- Dequant bias forces f32 IDCT path (3.7x slower than default i16)
