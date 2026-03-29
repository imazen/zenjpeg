# Decoder Path Matrix

Every decode goes through a series of decisions that select the code path. This doc maps the user-facing API to the internal pipeline that runs.

## Quick reference: what you write → what runs

### Fastest (streaming, zero-copy)

```rust
// Path A: streaming baseline. ~0.5ms for 512x512.
let img = Decoder::new().decode(&jpeg, stop)?;
let pixels: &[u8] = img.pixels_u8().unwrap();
```

Runs when: baseline JPEG + YCbCr 3-component + standard sampling (4:4:4, 4:2:0, 4:2:2) + u8 output. No coefficient storage. This is what most web JPEGs hit.

### Parallel (large images)

```rust
// Path B: fused parallel. ~6ms for 4096x4096 (vs 60ms sequential).
// Activates automatically when conditions are met.
let img = Decoder::new()
    .num_threads(0) // 0 = auto (default)
    .decode(&jpeg, stop)?;
```

Runs when: Path A conditions + `parallel` feature + DRI restart markers aligned to MCU rows + image ≥ 1024 MCU blocks. Transparent — same API, same output.

### mozjpeg-compatible

```rust
// Path C: coefficient → fast i16. Pixel-exact with mozjpeg/libjpeg-turbo.
let img = Decoder::new()
    .chroma_upsampling(ChromaUpsampling::Triangle) // auto-selects Libjpeg IDCT
    .decode(&jpeg, stop)?;
```

Uses 13-bit Loeffler IDCT + libjpeg-turbo-compatible chroma upsampling. Max pixel diff ≤ 2 vs mozjpeg on all tested images.

### Deblocked (reduce artifacts)

```rust
// Path D2/F: boundary filter. +1-10 zensim at low Q.
let img = Decoder::new()
    .deblock(DeblockMode::Auto)
    .decode(&jpeg, stop)?;

// Also works with scanline_reader (streaming when possible,
// falls back to buffered when Knusperli needed at low Q):
let mut reader = Decoder::new()
    .deblock(DeblockMode::Auto)
    .scanline_reader(&jpeg)?;
```

| DeblockMode | Quality sweet spot | Speed overhead | Streaming? |
|-------------|-------------------|---------------|------------|
| `Off` | — | 0% | yes |
| `Boundary4Tap` | All Q levels | 5-15% | yes |
| `Knusperli` | Q5-Q30 | 20-40% | fallback to buffered |
| `Auto` | Picks best | varies | falls back when needed |
| `AutoStreamable` | All Q levels | 5-15% | always streaming |

### Maximum reconstruction quality

```rust
// Path D4: dequant bias. +0.4 zensim at Q95, slower.
let img = Decoder::new()
    .dequant_bias(true) // forces f32 IDCT + Laplacian bias
    .decode(&jpeg, stop)?;
// Output is f32 [0,1]:
let pixels: &[f32] = img.pixels_f32().unwrap();
```

Only helps at Q85+. Hurts at Q50 and below. 5-14x slower than default.

### f32 output (HDR, compositing)

```rust
// Path D1: f32 coefficient decode.
let img = Decoder::new()
    .output_target(OutputTarget::SrgbF32) // or LinearF32
    .decode(&jpeg, stop)?;
let pixels: &[f32] = img.pixels_f32().unwrap(); // [0.0, 1.0] range
```

### XYB color space

```rust
// Path D5: XYB always uses f32 IDCT.
let img = Decoder::new()
    .correct_color(Some(TargetColorSpace::Srgb)) // convert XYB → sRGB
    .decode(&xyb_jpeg, stop)?;
```

XYB images (from cjpegli) are auto-detected via ICC profile. Always coefficient path + f32 IDCT.

### Scanline (streaming rows on demand)

```rust
// Path A/F: streaming scanline reader.
let mut reader = Decoder::new()
    .deblock(DeblockMode::AutoStreamable) // guaranteed streaming
    .scanline_reader(&jpeg)?;

let mut row_buf = vec![0u8; reader.width() as usize * 3];
while reader.rows_remaining() > 0 {
    let n = reader.read_rows_rgb8(
        imgref::ImgRefMut::new(&mut row_buf, reader.width() as usize * 3, 1)
    )?;
}
```

Falls back to buffered decode for: progressive, arithmetic, CMYK, EXIF transforms, or Knusperli deblock.

### Lossless transforms

```rust
// Path E (scanline) or D+transform (decode): DCT-domain rotation.
let img = Decoder::new()
    .auto_orient(true) // default: apply EXIF orientation
    .decode(&jpeg, stop)?;
```

Transforms happen in DCT domain (no generation loss). Forces coefficient path.

### Crop region

```rust
use zenjpeg::decode::CropRegion;

let img = Decoder::new()
    .crop(CropRegion::absolute(100, 100, 200, 200))
    .decode(&jpeg, stop)?;
```

Entropy decode runs for full image (DC predictor chain), but IDCT only runs for the crop region.

## Complete API surface

### Builder methods on `Decoder::new()`

| Method | Default | Effect on path |
|--------|---------|---------------|
| `.output_format(PixelFormat)` | `Rgb` | Gray/CMYK force coefficient path |
| `.output_target(OutputTarget)` | `Srgb8` | f32 targets force coefficient path |
| `.chroma_upsampling(ChromaUpsampling)` | `Triangle` | `LibjpegCompat` auto-selects Libjpeg IDCT |
| `.fancy_upsampling(bool)` | `true` | `false` → `NearestNeighbor` |
| `.idct_method(IdctMethod)` | auto | Overrides IDCT selection |
| `.deblock(DeblockMode)` | `Off` | Non-Off forces coefficient or streaming deblock |
| `.dequant_bias(bool)` | `false` | `true` forces f32 precise output |
| `.correct_color(Option<TargetColorSpace>)` | `None` | Post-decode ICC color management |
| `.auto_orient(bool)` | `true` | EXIF rotation in DCT domain |
| `.crop(CropRegion)` | none | Pixel-level crop of output |
| `.num_threads(usize)` | `0` (auto) | `1` disables parallel |
| `.strictness(Strictness)` | `Balanced` | Error tolerance for malformed JPEGs |
| `.max_pixels(u64)` | 256M | Resource limit |
| `.block_smoothing(bool)` | `false` | Progressive rendering smoothing (no effect on final output) |

### Entry points

| Method | Returns | When to use |
|--------|---------|------------|
| `.decode(data, stop)` | `DecodeResult` | Full image decode |
| `.scanline_reader(data)` | `ScanlineReader` | Row-by-row streaming |
| `.scanline_reader_cow(data)` | `ScanlineReader` | Owned or borrowed data |
| `.decode_rows(data, callback)` | per-row callback | Push-style streaming |
| `.decode_rows_f32(data, callback)` | per-row f32 callback | Push-style f32 |
| `.decode_coefficients(data, stop)` | `DecodedCoefficients` | Raw DCT access |
| `.decode_to_ycbcr_f32(data, stop)` | `DecodedYCbCr` | YCbCr planes |

### Output types

| `OutputTarget` | Pixel type | Precision | Speed |
|---------------|------------|-----------|-------|
| `Srgb8` (default) | `u8` | Standard | Fastest |
| `SrgbF32` | `f32` | Standard | ~same |
| `LinearF32` | `f32` | Standard + linearize | ~same |
| `SrgbF32Precise` | `f32` | Laplacian dequant bias | 1.5-2x slower |
| `LinearF32Precise` | `f32` | Laplacian + linearize | 1.5-2x slower |

### Pixel formats

| `PixelFormat` | Bytes/pixel | Notes |
|--------------|-------------|-------|
| `Rgb` (default) | 3 | Fast path |
| `Rgba` | 4 | Alpha = 255 |
| `Bgr` | 3 | Fast path |
| `Bgra` | 4 | Alpha = 255 |
| `Bgrx` | 4 | Padding byte |
| `Gray` | 1 | Forces coefficient path |

## Decision flow

```
User calls decode() or scanline_reader()
│
├─ output_target is f32? ─────────── yes ──→ Path D (coefficient → f32)
├─ dequant_bias? ─────────────────── yes ──→ Path D4 (f32 + bias)
├─ deblock != Off? ────────────────── yes ──→ Path D2/D3 (decode) or F (scanline streaming)
├─ transform (EXIF)? ──────────────── yes ──→ Path D + DCT transform
├─ XYB? ──────────────────────────── yes ──→ Path D5 (f32 XYB)
├─ format is Gray/CMYK? ──────────── yes ──→ Path D (coefficient)
├─ progressive/arithmetic? ────────── yes ──→ Path C (fast i16) or D
├─ exotic sampling? ──────────────── yes ──→ Path C/D (coefficient)
│
├─ parallel eligible? ─────────────── yes ──→ Path B (fused parallel)
│
└─ none of the above ──────────────────────→ Path A (streaming baseline)
```

The default `Decoder::new().decode(&jpeg, stop)` hits Path A for most web JPEGs — the fastest path with no coefficient storage.

## Path properties

| Path | Coefficient storage | IDCT type | Memory | Speed |
|------|-------------------|-----------|--------|-------|
| A: Streaming | none | i16 fused | O(MCU row) | fastest |
| B: Parallel | none | i16 fused | O(MCU row × threads) | fastest large |
| C: Fast i16 | full image | i16 batch | O(image) | fast |
| D: f32 coeff | full image | f32 per-block | O(image) | moderate |
| E: Buffered fallback | full image | varies | O(image × 2) | moderate |
| F: Streaming + deblock | none + 1 row | i16 fused + filter | O(MCU row) | fast |
