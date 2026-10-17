# Decoder Path Matrix

Every decode goes through a series of decisions that select the code path. This doc maps the user-facing API to the internal pipeline that runs.

## Quick reference: what you write → what runs

### Fastest (streaming, zero-copy)

```rust
// Path A: streaming baseline. ~0.5ms for 512x512.
let img = Decoder::new().decode(&jpeg, stop)?;
let pixels: &[u8] = img.pixels_u8().unwrap();
```

Runs when: baseline JPEG + one interleaved first scan + grayscale, YCbCr or RGB components + standard sampling (4:4:4, 4:2:0, 4:2:2 with 1x1 chroma) + u8 output. No coefficient storage. This is what most web JPEGs hit.

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

Falls back to buffered decode for: progressive and arithmetic (decoded to coefficients, then the same strip), and to a pixel buffer for XYB, CMYK, sampling the strip does not model, EXIF transforms, or Knusperli deblock.

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
| `.output_format(PixelFormat)` | `Rgb` | Does not pick the path: `Gray` is served for one-component frames and refused for colour frames (Known Bugs 4); `Cmyk` comes from four-component frames |
| `.output_target(OutputTarget)` | `Srgb8` | `decode()`: f32 targets take Path D. Scanline readers keep their route but never the wave |
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
| `.max_pixels(u64)` | 120M | Resource limit |
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
| `Gray` | 1 | One-component frames, on their own path; refused for colour frames (Known Bugs 4) |

## Decision flow

```
User calls decode() or scanline_reader()
│
├─ output_target is f32? ─────────── yes ──→ Path D (coefficient → f32)
├─ dequant_bias? ─────────────────── yes ──→ Path D4 (f32 + bias)
├─ deblock != Off? ────────────────── yes ──→ Path D2/D3 (decode) or F (scanline streaming)
├─ transform (EXIF)? ──────────────── yes ──→ Path D + DCT transform
├─ progressive/arithmetic? ────────── yes ──→ Path C (fast integer) or D
├─ first scan not interleaved? ────── yes ──→ Path C or D (coefficient)
├─ model/shape has no MCU kernel? ── yes ──→ Path C/D (coefficient; XYB → D5;
│                                            CMYK/YCCK → per-component)
│
├─ parallel eligible? ─────────────── yes ──→ Path B (fused parallel)
│
└─ none of the above ──────────────────────→ Path A (streaming baseline)
```

The model/shape/interleaving checks are one function each in
`decode/color_model.rs` (`PathPlan`); see the next section. The output format
does not pick the path: `Gray` output is served for one-component frames by
whichever path the frame takes (Path A for baseline 1x1 gray) and refused for
three-component frames (CLAUDE.md Known Bugs 4); `Cmyk` is the four-component
per-component path's output.

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

## Colour model and sampling shape

Every path answers "how do these components become colour?" and "which kernel
handles this sampling?" from two values decided once, when SOF is parsed, and
stored on the parser (`decode/color_model.rs`):

- `ColorModel`: Gray, YCbCr, Rgb, Xyb, Cmyk, Ycck. The rule is jpegli's
  `ProcessSOF` (`internal/jpegli-cpp/lib/jpegli/decode_marker.cc`). For three
  components: a JFIF APP0 means YCbCr; else an Adobe APP14 means RGB for
  transform 0 and YCbCr for any other value; else component IDs `'R','G','B'`
  mean RGB, anything else YCbCr. An RGB frame whose ICC profile is the XYB
  profile is XYB. Four components: transform 2 is YCCK, anything else
  Adobe-inverted CMYK.
- `FrameShape`: the chroma layout relative to component 0 (Gray, 4:4:4, 4:2:2,
  4:2:0, 4:4:0, Other) from exact ratios only, the maximum factors, and whether
  the chroma is 1x1 (the MCU every MCU-walking kernel is written for).
  Fractional ratios, asymmetric chroma, chroma above component 0, 4:1:1 and
  oversampled grayscale are `Other`.

Every entry point (`read_info`, `decode`, `decode_into`, `scanline_reader`
borrowed or owned, `decode_coefficients`, the wave reader) reads the header up
to SOF, so all of them report and decode the same model;
`JpegInfo::color_space` and `ScanlineReader::info().color_space` are this
model. An APP0 or APP14 placed after SOF does not change it. The ICC profile
is the exception: `extract_icc_profile` collects the APP2 chunks of the whole
header, up to the first SOS, before marker parsing starts, so an XYB profile
between SOF and SOS still makes an RGB-signalled frame XYB.

| decoder | rule | decided at |
|---|---|---|
| zenjpeg, jpegli | JFIF, then APP14, then IDs | SOF |
| libjpeg-turbo (browsers, Pillow) | same | first SOS (an APP14 between SOF and SOS counts) |
| IJG libjpeg 9 (`djpeg`) | IDs 1,2,3 / 'R','G','B' first, then JFIF, then APP14 | first SOS |

`tests/colorspace_oracles.rs` (`just colorspace-oracles`) checks each of
those decoders against its rule on the same fixtures.

### One kernel family for planes → pixels

The integer paths that walk MCU rows or strips (streaming baseline, fused
parallel, wave parallel, the fast integer coefficient path, the scanline strip)
convert a row of IDCT planes with a `RowKernel` (`color/planes.rs`) picked
once per decode from the model (`ColorModel::row_kernel`) and the output layout
(gray, RGB, BGR, RGBX, BGRX). YCbCr dispatches to the SIMD matrix kernels; RGB
is a clamp-and-interleave (`#[autoversion]`); gray replicates. `row_h2` serves
nearest-neighbour 4:2:2 / 4:2:0, whose chroma covers two pixels. The kernel is
chosen per row, never per pixel. The per-component path (CMYK/YCCK, XYB, and
the shapes no fast path serves) and the f32 coefficient path upsample whole f32
planes and convert them by the model once per image; their 8-bit layouts come
from the same `relayout_u8` the single-pass results use.

### Which paths serve which frames (`PathPlan`)

| path | models | shapes |
|---|---|---|
| streaming baseline (`decode`) | Gray, YCbCr, Rgb | one interleaved scan; Gray 1x1; 4:4:4, 4:2:2, 4:2:0 with 1x1 chroma |
| fused parallel (`parallel`, MCU-row DRI, >= 1024 MCUs) | Gray, YCbCr, Rgb | same; 4:4:4/gray kernel, box for nearest 4:2:2/4:2:0, h2v1 / fancy for triangle |
| wave-parallel scanline (`parallel`, MCU-row DRI, >= 1024 MCUs, `Srgb8` target, deblock off) | YCbCr, Rgb (RGB rows); YCbCr, Gray (native planes) | same as fused; RGB rows (the wave-only reader) only for nearest 4:2:2/4:2:0 |
| fast integer coefficient path (threaded at >= 2,000,000 px) | YCbCr, Rgb | exact chroma ratios 1x1, 2x1, 2x2, 1x2 (4:4:4, 4:2:2, 4:2:0, 4:4:0) with any chroma factors, e.g. (2,2)x3 or (4,2)/(2,1)/(2,1) |
| scanline strip (streaming and coefficient readers) | Gray, YCbCr, Rgb | Gray 1x1; the same ratios with every factor <= 2; the streaming reader needs one interleaved scan, the coefficient reader does not |
| per-component path (f32 planes) | every model | every shape |

The fused decision is taken before any buffer is charged or allocated.

The MCU-walking kernels are written for 1x1 chroma. Shapes with the same
ratios but larger chroma factors used to reach them and now take the fast
integer path or the strip instead: (2,2)/(1,2)/(1,2) and (4,1)/(2,1)/(2,1)
lost the fused kernels (the threaded fast integer path serves them), and the
planar wave state (`read_rows_ycbcr_native_i16` with threads) is not attached
for 4:4:0, (1,2)x3, (2,1)x3, (2,2)/(1,2)/(1,2) or (2,2)/(2,1)/(2,1). This is
performance only; the pixels are the sequential paths'.

### Route and output target

`OutputTarget` changes the IDCT output the strip keeps, not the route: the
streaming and coefficient scanline routes both honour it. With an f32 target
the strip keeps the unclamped IDCT output (ringing below 0 and above 255
survives), f32 reads convert it as is, and 8-bit reads convert it and clamp
the result. So 8-bit reads of one file can differ between targets (by design),
but never between routes, thread counts or SIMD tiers.

Two rules make that hold:

- The unclamped IDCT output is bounded to +/-2047 (`color::planes::SAMPLE_LIMIT`,
  applied in `decode/idct_int.rs` on every tier before the i16 store). A
  legal but extreme coefficient block can ring past 4000; at that size the
  AVX2 h2v2 sums (16 x sample) and the i16 chroma centring overflowed on one
  tier and not another. Measured 2026-10-10 on 392 valid and real-photo
  JPEGs (codec-corpus `jpeg-conformance/{valid,non-conformant}`, `imageflow`,
  `mozjpeg`, `zune/test-images`, `imazen-26`; `SrgbF32` native i16 reads), the
  unclamped samples span [-73, 335] and none reaches the bound; 11 crafted
  files from `jpeg-conformance/{invalid,crash-repro}` do.
- The wave reader holds rows converted from clamped 8-bit samples and has no
  deblock filter, so readers with an f32 target or a deblock mode never take
  the wave route (`compute_wave_state`); they take the strip at every thread
  count.

### Reads derived from RGB

For RGB frames (and readers that hold decoded RGB: sampling shapes the strip
does not serve, CMYK, XYB, transform and deblock fallbacks) the scanline reads
that need luma or YCbCr compute it from the decoded R, G, B with one shared
formula set, so every reader mode returns the same bits. (Progressive
grayscale readers hold a one-byte gray buffer and return it directly.)

- `read_rows_gray8`: libjpeg-turbo's `rgb_gray_convert`,
  `(19595 R + 38470 G + 7471 B + 32768) >> 16`.
- `read_rows_gray_f32` / `read_rows_ycbcr_f32`: the BT.601 matrix
  (`foundation::consts::YCBCR_*`) in f32, normalized.
- `read_rows_gray_linear_f32`: linearize each channel through the exact 8-bit
  sRGB table, then weight.
- `read_rows_ycbcr_native_i16`: an error for RGB, XYB and CMYK/YCCK frames
  (they have no native YCbCr planes) and for readers that hold decoded pixels.
- `Decoder::decode_to_ycbcr_f32`: the BT.601 matrix applied to the R, G, B
  planes.

Integer samples (gray, Y, RGB) linearize through the 8-bit table on every
path; YCbCr frames' `read_rows_rgba_f32` converts the planes to RGB in f32
first. Wave-only readers (`parallel`, nearest 4:2:2/4:2:0, MCU-row DRI,
>= 1024 MCUs, `Srgb8` target, deblock off) keep decoded RGB rows, so on YCbCr
frames the reads that need the upsampled planes (`gray8`, `gray_f32`,
`gray_linear_f32`, `ycbcr_f32`, `rgba_f32`) are refused with an
`UnsupportedFeature` error naming `num_threads(1)` (CLAUDE.md Known Bugs 3).
`read_rows_ycbcr_native_i16` is served from the planar wave; its chroma
padding columns repeat the last real column, as the strip's do.
`decode_rows(Gray)`, `decode_rows_f32` and the zencodec gray/f32 streaming
decode build their reader without the wave-only route.

### Proving a test reaches a path

`tests/colorspace_paths.rs` runs every fixture through every entry point
(`decode` / `decode_into` x pixel format x thread count, scanline reads,
borrowed and owned readers) and compares each against a known answer or
`decode()` at one thread. A test that passes does not show it reached the path
it names: gates send frames elsewhere silently. `scripts/colorspace_mutation_reach.py`
corrupts one site at a time in a copy of the tree (flip one output byte, or
shift an f32 value) and records which tests fail; `just
colorspace-mutation-reach` runs it on a kept copy of the working copy. Its 55
sites are every `RowKernel` call, `rgb8_row`, `gray8_row`, `relayout_u8`, the
in-place RGB -> BGR swap, the scanline reads that derive luma, YCbCr or linear
values or return the gray / Y / Cb / Cr samples, every arm of the
per-component path (u8 and f32: XYB, YCbCr, RGB, gray, CMYK/YCCK) and
`decode_to_ycbcr_f32`'s RGB conversion; on 2026-10-11 every site failed at least one test. Each pass so far found
sites no test reached: the streaming gray loop (hence
`gray_frames_every_entry_point`), the f32 reads of readers that hold decoded
gray or RGB (hence `check_gray_f32_reads` and `rgba_f32` in
`rgb_derived_reads_agree_across_reader_modes`), the f32 linearization of YCbCr
`rgba_f32` (hence `ycbcr_rgba_f32_is_the_planes_through_bt601`), and on
2026-10-11 the gray arms of `read_rows_ycbcr_f32` (now in
`check_gray_f32_reads`), the per-component u8 and f32 arms, which tests only
compared with `decode()` running the same arm (hence
`per_component_u8_and_f32_outputs_agree`), and CMYK/YCCK, which had no test
(hence `cmyk_and_ycck_known_answer`). A path compared only with itself is not
tested: give each site an anchor outside its own arm. Repeat this whenever a
gate or a conversion site changes.

