# Decoder Path Matrix

Every decode goes through a series of decisions that select the code path. This doc maps every combination and what image/config properties trigger it.

## Decision layers

| # | Decision | Options | Key condition |
|---|----------|---------|---------------|
| 1 | Entry point | `decode()` / `scanline_reader()` | User's API call |
| 2 | Output precision | u8 / f32 | `output_target` (Srgb8 vs SrgbF32/LinearF32/Precise) |
| 3 | JPEG mode | Baseline / Progressive / Arithmetic | SOF marker in file |
| 4 | Parallel | Fused parallel / Sequential | `parallel` feature + DRI aligned + MCU count ≥ 1024 |
| 5 | Streaming | Direct RGB / Coefficient buffer | Baseline + 3-comp + standard sampling + no transforms/deblock/f32 |
| 6 | IDCT | Jpegli (12-bit) / Libjpeg (13-bit) / f32 | `idct_method`, `LibjpegCompat`, XYB, dequant_bias |
| 7 | Chroma upsample | Nearest / Triangle / LibjpegCompat / HorizontalFancy | `chroma_upsampling` config |
| 8 | Deblock | Off / Boundary4Tap / Knusperli | `deblock_mode` config |
| 9 | Transform | None / 8 lossless transforms | EXIF orientation + `auto_orient` |
| 10 | Color space | YCbCr / XYB / Grayscale / CMYK | Image content |
| 11 | ICC | Apply / Skip | `apply_icc` + embedded profile |
| 12 | Crop | Full / Region | `crop_region` config |

## Major code paths

### Path A: Streaming baseline (fastest)

**Trigger:** Baseline + 3-component YCbCr + standard sampling (4:4:4 / 4:2:0 / 4:2:2) + u8 output + no transforms + no f32 requirements + deblock Off or Boundary4Tap

**Pipeline:** Entropy → IDCT → color convert → RGB u8 (single MCU-row pass, no coefficient storage)

**Files:** `scan.rs:decode_baseline_streaming()`, `scan.rs:534-800`

### Path B: Fused parallel baseline

**Trigger:** Path A conditions + `parallel` feature + DRI MCU-row-aligned + MCU count ≥ 1024 + `num_threads != 1`

**Pipeline:** Parallel entropy → IDCT → color convert per restart segment

**Files:** `fused_parallel.rs`

### Path C: Coefficient → fast i16 output

**Trigger:** Progressive OR arithmetic OR streaming ineligible, but u8 output + non-XYB + no dequant_bias + RGB family format

**Sub-paths:**
- C1: 4:4:4 → `to_pixels_fast_i16()`
- C2: 4:2:0/4:2:2 → `to_pixels_fast_i16_subsampled()`
- C3: Parallel variants of C1/C2

**Files:** `output.rs:212-327` (fast i16), `output.rs:1088-1096` (selection)

### Path D: Coefficient → f32 output

**Trigger:** f32 output target OR dequant_bias OR XYB OR deblock != Off OR non-RGB format

**Pipeline:** Coefficient storage → f32 IDCT per block → optional deblock → upsample f32 → color convert

**Sub-paths:**
- D1: Standard f32 (no deblock)
- D2: f32 + Boundary4Tap deblock
- D3: f32 + Knusperli deblock (replaces IDCT output)
- D4: f32 + dequant_bias (Laplacian bias before IDCT)
- D5: XYB (always f32 IDCT, special color convert)

**Files:** `output.rs:1356-1516` (to_pixels_f32_inner)

### Path E: Buffered scanline fallback

**Trigger:** `scanline_reader()` + (progressive / arithmetic / CMYK / exotic sampling / Knusperli deblock / transform)

**Pipeline:** Full `decode()` → wrap pixels in buffered `ScanlineReader`

**Files:** `mod.rs:scanline_reader_deblock_fallback()`, `mod.rs:scanline_reader_with_transform()`

### Path F: Streaming scanline with deblock

**Trigger:** `scanline_reader()` + Boundary4Tap/AutoStreamable + baseline + standard sampling

**Pipeline:** Streaming decode → i16 boundary filter per MCU row → color convert → RGB u8

**Files:** `scanline.rs:apply_boundary_deblock()`

## What triggers what

### By image property

| Image property | Paths affected |
|----------------|---------------|
| **Baseline JPEG** | A, B, C, D possible |
| **Progressive JPEG** | C, D only (no streaming) |
| **Arithmetic JPEG** | C, D only (no streaming) |
| **4:4:4** | A (streaming), C1 (fast i16) |
| **4:2:0** | A (streaming), C2 (fast i16 subsampled) |
| **4:2:2** | A (streaming), C2 (fast i16 subsampled) |
| **4:4:0 or exotic** | C, D only (no streaming) |
| **Grayscale** | C, D only (streaming requires 3-comp) |
| **CMYK (4-comp)** | D only (coefficient + f32 convert) |
| **XYB color space** | D5 only (forces f32 IDCT) |
| **With DRI + ≥1024 MCUs** | B possible (fused parallel) |
| **No DRI or <1024 MCUs** | A (sequential streaming) |
| **Width/height not 8-aligned** | Same paths, MCU padding handled |
| **12-bit precision** | Error (unsupported in scanline) |
| **DNL (height=0 in SOF)** | Error in scanline, supported in decode |

### By config option

| Config | Effect on path |
|--------|---------------|
| `output_target: Srgb8` (default) | u8 paths (A, B, C) preferred |
| `output_target: SrgbF32` | Forces D (f32 coefficient path) |
| `output_target: SrgbF32Precise` | Forces D4 (dequant_bias) |
| `deblock(Off)` (default) | No effect (zero overhead) |
| `deblock(Boundary4Tap)` | decode(): D2. scanline: F |
| `deblock(Knusperli)` | decode(): D3. scanline: E (fallback) |
| `deblock(Auto)` | Low Q → D3/E. High Q → D2/F |
| `deblock(AutoStreamable)` | Always D2/F (never falls back) |
| `chroma_upsampling(Triangle)` (default) | Jpegli-style filter |
| `chroma_upsampling(LibjpegCompat)` | Libjpeg IDCT auto-selected |
| `chroma_upsampling(NearestNeighbor)` | Box filter (fastest) |
| `idct_method(Jpegli)` (default) | 12-bit fixed-point |
| `idct_method(Libjpeg)` | 13-bit Loeffler |
| `apply_icc(true)` | Post-decode ICC transform |
| `auto_orient(true)` (default) | EXIF rotation → Path E in scanline |
| `crop_region(Some(...))` | Post-decode pixel crop |
| `num_threads(1)` | Disables parallel (B→A) |

## Coverage test matrix

To exercise every major path, test with:

| Test case | Entry | Image | Config | Path |
|-----------|-------|-------|--------|------|
| 1 | decode | baseline 4:2:0 | defaults | A→streaming |
| 2 | decode | baseline 4:4:4 | defaults | A→streaming |
| 3 | decode | baseline 4:2:2 | defaults | A→streaming |
| 4 | decode | progressive 4:2:0 | defaults | C2 |
| 5 | decode | baseline 4:2:0 | deblock=Boundary4Tap | D2 |
| 6 | decode | baseline 4:2:0 | deblock=Knusperli | D3 |
| 7 | decode | baseline 4:2:0 | deblock=Auto, Q20 | D3 (knusperli) |
| 8 | decode | baseline 4:2:0 | deblock=Auto, Q85 | D2 (boundary) |
| 9 | decode | baseline 4:2:0 | dequant_bias=true | D4 |
| 10 | decode | baseline 4:2:0 | output_target=SrgbF32 | D1 |
| 11 | decode | baseline 4:2:0 | chroma=LibjpegCompat | C2 (libjpeg IDCT) |
| 12 | decode | baseline 4:2:0 | chroma=NearestNeighbor | C2 (box filter) |
| 13 | decode | grayscale | defaults | C (no streaming) |
| 14 | decode | CMYK | defaults | D (coefficient) |
| 15 | decode | XYB | defaults | D5 |
| 16 | decode | baseline + EXIF rotation | auto_orient=true | D + transform |
| 17 | decode | baseline + ICC profile | apply_icc=true | A + ICC |
| 18 | decode | baseline 4:2:0 | crop_region | A + crop |
| 19 | scanline | baseline 4:2:0 | defaults | A→streaming |
| 20 | scanline | progressive 4:2:0 | defaults | E (buffered fallback) |
| 21 | scanline | baseline 4:2:0 | deblock=Boundary4Tap | F |
| 22 | scanline | baseline 4:2:0 | deblock=Knusperli | E (fallback) |
| 23 | scanline | baseline 4:2:0 | deblock=AutoStreamable | F |
| 24 | scanline | baseline + EXIF rotation | auto_orient=true | E (transform fallback) |
| 25 | scanline | CMYK | defaults | E (buffered fallback) |
| 26 | decode | baseline 4:2:0 + DRI | parallel feature | B |
| 27 | decode | arithmetic sequential | defaults | C |
