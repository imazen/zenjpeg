# zenjpeg [![CI](https://img.shields.io/github/actions/workflow/status/imazen/zenjpeg/ci.yml?style=flat-square&label=CI)](https://github.com/imazen/zenjpeg/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/zenjpeg?style=flat-square)](https://crates.io/crates/zenjpeg) [![lib.rs](https://img.shields.io/crates/v/zenjpeg?style=flat-square&label=lib.rs&color=blue)](https://lib.rs/crates/zenjpeg) [![docs.rs](https://img.shields.io/docsrs/zenjpeg?style=flat-square)](https://docs.rs/zenjpeg) [![license](https://img.shields.io/crates/l/zenjpeg?style=flat-square)](https://github.com/imazen/zenjpeg/blob/main/LICENSE)

A pure Rust JPEG encoder and decoder. Heavily inspired by jpegli and mozjpeg, with significant original research: streaming single-pass decode and encode with bounded memory, parallel decode/encode, hybrid trellis quantization, fixed Huffman tables that are 2x better than the JPEG spec defaults, and perceptual quality tuning. A streaming encode uses ~1.8 MB peak for 1080p with fixed Huffman tables — only ~5% larger than optimized Huffman on photo content, yet still beats many popular encoders on perceptual quality. Safe SIMD on x86_64 and aarch64 via archmage tokens. `#![forbid(unsafe_code)]`.

> **Note:** This crate was previously published as `jpegli-rs`. If migrating, update imports from `use jpegli::` to `use zenjpeg::`.

## Quick Start

### Encode

```rust
use zenjpeg::encoder::{EncoderConfig, PixelLayout, ChromaSubsampling, Unstoppable};

let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter);
let mut enc = config.encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)?;
enc.push_packed(&rgb_bytes, Unstoppable)?;
let jpeg_bytes: Vec<u8> = enc.finish()?;
```

### Decode

```rust
use zenjpeg::decoder::Decoder;
use enough::Unstoppable;

let result = Decoder::new().decode(&jpeg_bytes, Unstoppable)?;
let rgb_pixels: &[u8] = result.pixels_u8().expect("u8 output");
let (width, height) = result.dimensions();
```

### Streaming Decode (Row-by-Row)

```rust
use zenjpeg::decoder::Decoder;
use imgref::ImgRefMut;

let mut reader = Decoder::new().scanline_reader(&jpeg_data)?;
let w = reader.width() as usize;
let mut buf = vec![0u8; w * reader.height() as usize * 3];
let mut rows = 0;
while !reader.is_finished() {
    let slice = &mut buf[rows * w * 3..];
    let output = ImgRefMut::new(slice, w * 3, reader.height() as usize - rows);
    rows += reader.read_rows_rgb8(output)?;
}
```

## Heritage

Started as a port of [jpegli](https://github.com/libjxl/libjxl/tree/main/lib/jpegli) from Google's JPEG XL project. After six rewrites it shares ideas but little code with the original.

**From jpegli:** adaptive quantization, XYB color space, perceptual quant tables, zero-bias coefficient rounding.

**From mozjpeg:** overshoot deringing (enabled by default), trellis quantization, hybrid trellis mode. Requires `trellis` feature.

**Our own:** pure safe Rust, streaming row-by-row API, parallel encode/decode, deblocking filters, UltraHDR gain maps, JPEG source detection and re-encoding recommendations.

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `decoder` | **yes** | JPEG decoder (streaming, parallel, deblocking) |
| `trellis` | no | Trellis quantization, `auto_optimize()`, mozjpeg/hybrid presets. Compile error without it. |
| `parallel` | no | Multi-threaded encode/decode via rayon |
| `moxcms` | no | Color management (pure Rust). Required for `.correct_color()` and XYB |
| `ultrahdr` | no | UltraHDR HDR gain map encode/decode |
| `zencodec` | no | zencodec trait implementations for cross-codec pipelines |
| `layout` | no | Lossless transforms + lossy decode→resize→encode pipeline |

`decoder` is on by default. The decoder API is prerelease; expect breaking changes.

```toml
# Encode + decode (most common)
[dependencies]
zenjpeg = "0.6"

# Best compression (trellis + auto_optimize)
[dependencies]
zenjpeg = { version = "0.6", features = ["trellis"] }

# High-performance server
[dependencies]
zenjpeg = { version = "0.6", features = ["parallel", "trellis"] }

# Color-managed decode (XYB, ICC profiles)
[dependencies]
zenjpeg = { version = "0.6", features = ["moxcms"] }
```

## Encoder

### Color Modes

| Constructor | Use Case |
|-------------|----------|
| `EncoderConfig::ycbcr(q, sub)` | Standard JPEG (most compatible) |
| `EncoderConfig::xyb(q, b_sub)` | XYB perceptual color (better quality, needs `moxcms` to decode) |
| `EncoderConfig::grayscale(q)` | Single-channel |

### Entry Points

| Method | Input Type | Use Case |
|--------|------------|----------|
| `encode_from_bytes(w, h, layout)` | `&[u8]` | Raw byte buffers |
| `encode_from_rgb::<P>(w, h)` | `rgb` crate types | `RGB<u8>`, `RGBA<f32>`, etc. |
| `encode_from_ycbcr_planar(w, h)` | `YCbCrPlanes` | Video pipeline output |

All three return a streaming `Encoder`. Push rows with `push_packed()`, finish with `finish()`. One-shot convenience: `config.request().encode(&pixels, w, h)`.

### Builder Methods

| Method | Default | Notes |
|--------|---------|-------|
| `.progressive(bool)` | `true` | ~3% smaller, ~2x slower |
| `.auto_optimize(bool)` | `false` | Best quality/size (hybrid trellis). Requires `trellis` feature |
| `.deringing(bool)` | `true` | Overshoot deringing for documents/graphics |
| `.separate_chroma_tables(bool)` | `true` | 3 quant tables (Y, Cb, Cr) vs 2 |
| `.huffman(strategy)` | `Optimize` | Huffman table strategy |
| `.sharp_yuv(bool)` | `false` | SharpYUV chroma downsampling |

### Quality Options

```rust
use zenjpeg::encoder::{EncoderConfig, Quality, ChromaSubsampling};

// Simple quality scale (0-100)
let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter);

// Target a specific metric
let config = EncoderConfig::ycbcr(Quality::ApproxMozjpeg(80), ChromaSubsampling::Quarter);
let config = EncoderConfig::ycbcr(Quality::ApproxSsim2(90.0), ChromaSubsampling::Quarter);
let config = EncoderConfig::ycbcr(Quality::ApproxButteraugli(1.0), ChromaSubsampling::Quarter);
```

### Trellis Modes (requires `trellis` feature)

**Default (no trellis):** adaptive quantization with perceptual zero-bias. Fast, good quality.

**Hybrid trellis (`auto_optimize(true)`):** combines jpegli AQ with mozjpeg trellis. Best quality/size tradeoff. +1.5 SSIMULACRA2 points vs default at matched file size.

```rust
let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
    .auto_optimize(true); // requires trellis feature
```

**Mozjpeg-compatible presets:** `MozjpegBaseline`, `MozjpegProgressive`, `HybridProgressive`, `HybridMaxCompression` via `ExpertConfig::from_preset()`.

### Per-Image Metadata (Three-Layer Pattern)

For encoding multiple images with the same config but different metadata:

```rust
use zenjpeg::encoder::{EncoderConfig, ChromaSubsampling};

// Layer 1: Reusable config
let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
    .progressive(true);

// Layer 2: Per-image request (metadata, limits, stop token)
let jpeg = config.request()
    .icc_profile(&srgb_icc_bytes)
    .encode(&pixels, 1920, 1080)?;

// Layer 3: Streaming execution
let mut encoder = config.request()
    .icc_profile(&p3_icc_bytes)
    .encode_from_rgb::<rgb::RGB<u8>>(1920, 1080)?;
encoder.push_packed(&pixels, enough::Unstoppable)?;
let jpeg = encoder.finish()?;
```

Request builder methods: `.icc_profile()`, `.exif()`, `.xmp()`, `.stop()`, `.limits()`.

### Pixel Layouts

| Layout | Bytes/px | Notes |
|--------|----------|-------|
| `Rgb8Srgb` | 3 | Default, sRGB gamma |
| `Bgr8Srgb` / `Bgra8Srgb` / `Bgrx8Srgb` | 3/4 | Windows/GDI order |
| `Rgba8Srgb` / `Rgbx8Srgb` | 4 | Alpha/pad ignored |
| `Gray8Srgb` | 1 | Grayscale |
| `Rgb16Linear` / `Rgba16Linear` | 6/8 | 16-bit linear |
| `RgbF32Linear` / `RgbaF32Linear` | 12/16 | HDR float (0.0-1.0) |

## Decoder

### Options

| Method | Default | Effect |
|--------|---------|--------|
| `.chroma_upsampling(method)` | `Triangle` | `NearestNeighbor` for speed. Default matches libjpeg-turbo within max_diff ≤ 3 |
| `.idct_method(method)` | `Jpegli` | `Libjpeg` for pixel-exact mozjpeg match (adds ~37% overhead) |
| `.deblock(mode)` | `Off` | Reduce block artifacts (see [Deblocking](#deblocking)) |
| `.dequant_bias(true)` | `false` | f32 IDCT + Laplacian bias for max reconstruction quality |
| `.output_target(target)` | `Srgb8` | f32 output: `SrgbF32`, `LinearF32`, `SrgbF32Precise` |
| `.output_format(fmt)` | `Rgb` | Pixel format: `Rgb`, `Rgba`, `Bgr`, `Bgra`, `Bgrx`, `Gray` |
| `.correct_color(target)` | `None` | ICC color management (requires `moxcms` feature) |
| `.auto_orient(bool)` | `true` | Apply EXIF orientation in DCT domain |
| `.transform(t)` | none | Lossless rotation/flip during decode |
| `.crop(region)` | none | Pixel-level crop (IDCT skipped outside region) |
| `.num_threads(n)` | `0` (auto) | `1` forces sequential |
| `.strictness(level)` | `Balanced` | `Strict`, `Balanced`, `Lenient`, `Permissive` |
| `.max_pixels(n)` | 100M | DoS protection |
| `.max_memory(n)` | 512 MB | Memory limit |

### Decode Paths

For most web JPEGs, `Decoder::new().decode(&data, stop)` hits the streaming path -- no coefficient storage, one MCU-row pass through entropy/IDCT/color/output. This is the fastest path.

Progressive, CMYK, f32 output, deblocking (Knusperli), and transforms go through the coefficient path. Parallel decode activates automatically when DRI restart markers are present and the image has 1024+ MCU blocks.

See `docs/DECODER_PATHS.md` for the full decision flow and path matrix.

### Output Targets

| `OutputTarget` | Pixel type | Notes |
|---------------|------------|-------|
| `Srgb8` (default) | `u8` | Fastest |
| `SrgbF32` | `f32` | sRGB gamma, 0.0-1.0 |
| `LinearF32` | `f32` | Linear light (for compositing) |
| `SrgbF32Precise` | `f32` | Laplacian dequant bias, 1.5-2x slower |
| `LinearF32Precise` | `f32` | Precise + linearize |

### Scanline Reader Methods

| Method | Bytes/px | Format |
|--------|----------|--------|
| `read_rows_rgb8()` | 3 | R-G-B |
| `read_rows_bgr8()` | 3 | B-G-R |
| `read_rows_rgba8()` / `read_rows_bgra8()` | 4 | With alpha=255 |
| `read_rows_rgbx8()` / `read_rows_bgrx8()` | 4 | With pad=255 |
| `read_rows_rgba_f32()` | 16 | Linear f32 RGBA |
| `read_rows_gray8()` / `read_rows_gray_f32()` | 1/4 | Grayscale |

### Deblocking

JPEG's 8x8 block structure creates visible grid artifacts at low quality. The decoder can reduce these with post-decode filtering.

```rust
use zenjpeg::decoder::{Decoder, DeblockMode};

let result = Decoder::new()
    .deblock(DeblockMode::Auto)
    .decode(&jpeg_data, enough::Unstoppable)?;
```

| DeblockMode | Quality gain (zensim vs original) | Speed | Streaming? |
|-------------|----------------------------------|-------|------------|
| `Off` | — | 0% overhead | yes |
| `Boundary4Tap` | +0.5 at Q90, +2 at Q50, +10 at Q10 | +2% scanline | yes |
| `Knusperli` | +14 at Q5-Q10, hurts at Q70+ | 20-40% slower | falls back to buffered |
| `Auto` | Picks best per quality level | varies | falls back when needed |
| `AutoStreamable` | Boundary4Tap only (streaming-safe) | +2% scanline | always |

All modes work with both `decode()` and `scanline_reader()`. When `scanline_reader()` needs Knusperli, it transparently falls back to coefficient-based decoding.

### Color Management

Requires the `moxcms` feature (pure Rust). Converts the embedded ICC profile to the target color space during decode.

```rust
use zenjpeg::color::icc::TargetColorSpace;

let img = Decoder::new()
    .correct_color(Some(TargetColorSpace::Srgb))
    .decode(&jpeg_data, enough::Unstoppable)?;
```

Default is `None` -- no color conversion. Pixels are returned in the JPEG's native color space.

### Lossless Transforms

Rotate, flip, and transpose by manipulating DCT coefficients directly. No decode to pixels, no re-encode, zero generation loss.

```rust
use zenjpeg::lossless::{transform, apply_exif_orientation, LosslessTransform, TransformConfig};

// Rotate 90 degrees losslessly
let rotated = transform(&jpeg_data, &TransformConfig {
    transform: LosslessTransform::Rotate90,
    ..Default::default()
}, enough::Unstoppable)?;

// Auto-correct EXIF orientation
let oriented = apply_exif_orientation(&jpeg_data, enough::Unstoppable)?;
```

All 8 D4 dihedral group elements: `None`, `FlipHorizontal`, `FlipVertical`, `Transpose`, `Rotate90`, `Rotate180`, `Rotate270`, `Transverse`.

### Cooperative Cancellation

Both encoder and decoder accept `Stop` tokens for graceful shutdown:

```rust
use enough::Unstoppable;

// Never cancel
let image = Decoder::new().decode(&jpeg_data, Unstoppable)?;

// Custom cancellation (e.g., user clicked cancel)
let result = Decoder::new().decode(&jpeg_data, &cancel_token);
```

## Detect API (Encoder Identification)

Identify the source encoder and quality of any JPEG from its headers (~500 bytes, <1us), then get optimal re-encoding settings.

```rust
use zenjpeg::detect::probe;
use zenjpeg::encoder::{EncoderConfig, ChromaSubsampling};

let info = probe(&jpeg_data)?;
println!("Encoder: {:?}, Quality: {:.0}", info.encoder, info.quality.value);

// Get recommended zenjpeg quality to match perceived quality
let config = EncoderConfig::ycbcr(
    info.recommended_quality(),
    info.recommended_subsampling(),
);
```

Detected families: `LibjpegTurbo`, `Mozjpeg`, `CjpegliYcbcr`, `CjpegliXyb`, `ImageMagick`, `IjgFamily`, `Unknown`. Configurable quality/size tradeoff via `info.reencode_settings(tolerance)`.

## Performance

### Encode

Tested on CID22 corpus (337 real photos), size-matched comparison against mozjpeg (Ryzen 9 7950X):

| Mode | vs mozjpeg | Win rate |
|------|-----------|----------|
| `auto_optimize(true)` (trellis) | +0.64 zensim, -0.36 butteraugli | 81% |
| Default (no trellis) | +0.07 zensim, -0.36 butteraugli | -- |

Progressive produces ~3% smaller files at the same quality, takes ~2x longer to encode.

### Decode

Baseline 4:2:0 throughput (zenbench, 10 CID22 photos, Ryzen 9 7950X):

| Decoder | Throughput | vs libjpeg-turbo |
|---------|-----------|-----------------|
| libjpeg-turbo/mozjpeg (C+NASM) | 78.1 MiB/s | -- |
| zenjpeg default (Triangle) | 73.8 MiB/s | 0.94x |
| zenjpeg NearestNeighbor | 80.4 MiB/s | 1.03x |

6% slower than C+NASM on baseline with the default Triangle upsampling (libjpeg-turbo compatible rounding). NearestNeighbor (box filter) matches or beats C. On progressive JPEGs, zenjpeg is **1.35x faster** (46 vs 34 MiB/s) due to the fused single-pass architecture.

Parallel decode (baseline with DRI, `--features parallel`):

| Size | libjpeg-turbo | zenjpeg parallel | ratio |
|------|--------------|-----------------|-------|
| 1024 | 3.86ms | 1.56ms | **0.40x** |
| 2048 | 15.8ms | 3.05ms | **0.19x** |
| 4096 | 65.3ms | 8.74ms | **0.13x** |

Parallel activates automatically with DRI and 1024+ MCU blocks. Use `num_threads(1)` to force sequential.

## Known Limitations

- **Baseline decode speed**: 6% slower than libjpeg-turbo (C+NASM) on baseline JPEGs. Faster on progressive.
- **XYB decode speed**: XYB images use the f32 pipeline; standard JPEGs use fast integer IDCT.
- **XYB file size**: Baseline mode is 2-3% larger than C++ jpegli. Progressive mode matches or beats.
- **Trellis is opt-in**: `auto_optimize()` and mozjpeg presets require `features = ["trellis"]`.

## Table Optimization

The `EncodingTables` API provides fine-grained control over quantization and zero-bias tables for codec research.

```rust
use zenjpeg::encoder::tuning::{EncodingTables, ScalingParams, dct};

let mut tables = EncodingTables::default_ycbcr();
tables.scale_quant(0, 5, 1.2);  // 20% higher quantization at position 5

let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter)
    .tables(Box::new(tables));
```

Helpers: `dct::freq_distance(k)`, `dct::IMPORTANCE_ORDER`, `tables.blend(&other, t)`, `tables.quant.scale_all(f)`.

## C++ Parity

Tested against C++ jpegli on frymire.png (1118x1105) using `jpegli_set_distance()` (3-table mode):

| Metric | Difference |
|--------|------------|
| File size (Q85 seq) | -0.1% |
| File size (Q85 prog) | +0.5% |
| SSIMULACRA2 (Q85) | identical |

When comparing: always use `jpegli_set_distance()`, not `jpeg_set_quality()`. The latter uses 2 chroma tables vs our 3, inflating apparent differences. Use `.separate_chroma_tables(false)` to match 2-table mode.

## Development

```bash
cargo test --release                    # ~930 tests, no external deps
cargo test --release --test cpp_parity_locked  # Quick C++ parity check
cargo test --release -- --ignored       # Full suite (needs C++ build + corpus)
```

## License

Sustainable, large-scale open source work requires a funding model, and I have been
doing this full-time for 15 years. If you are using this for closed-source development
AND make over $1 million per year, you'll need to buy a commercial license at
https://www.imazen.io/pricing

Commercial licenses are similar to the Apache 2 license but company-specific, and on
a sliding scale. You can also use this under the AGPL v3.

## Acknowledgments

Built on ideas from [jpegli](https://github.com/libjxl/libjxl/tree/main/lib/jpegli)
(Google, BSD-3-Clause) and [mozjpeg](https://github.com/nickt/mozjpeg-rs) (Mozilla).
After six rewrites from the initial jpegli port, zenjpeg is an independent project
with its own architecture, streaming pipeline, and quality optimizations.

## AI Disclosure

Developed with assistance from Claude (Anthropic). Extensively tested against
C++ reference with 930+ tests. Report issues at https://github.com/imazen/zenjpeg/issues
