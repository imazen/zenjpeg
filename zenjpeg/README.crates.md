<!-- GENERATED FROM README.md by zenutils gen-readme-crates.sh — DO NOT EDIT. -->

# zenjpeg

zenjpeg is a pure-Rust JPEG encoder and decoder with perceptual optimization. It
began as a port of Google's [jpegli](https://github.com/libjxl/libjxl/tree/main/lib/jpegli)
(from the JPEG XL project) and, after several rewrites, adds streaming single-pass
encode and decode with bounded memory, parallel encode/decode, adaptive
quantization, optional trellis quantization, an optional XYB perceptual color
space, UltraHDR gain maps, lossless DCT-domain transforms, and JPEG→JPEG
recompression. Safe SIMD on x86-64 (AVX2/AVX-512) and aarch64 (NEON) via
[archmage](https://github.com/imazen/archmage) tokens. `#![forbid(unsafe_code)]`,
no C dependencies.

> **Note:** This crate was previously published as `jpegli-rs`. If migrating, update imports from `use jpegli::` to `use zenjpeg::`.

## Quick Start

```toml
[dependencies]
zenjpeg = "0.8"
```

Encode with the `EncoderConfig` builder and decode with `Decoder`. The builder
covers chroma subsampling, XYB color, progressive scans, embedded ICC/EXIF/XMP,
resource limits, and cooperative cancellation.

### Encode (builder)

```rust
use zenjpeg::encoder::{EncoderConfig, PixelLayout, ChromaSubsampling, Unstoppable};

let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);
let mut enc = config.encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)?;
enc.push_packed(&rgb_bytes, Unstoppable)?;
let jpeg_bytes: Vec<u8> = enc.finish()?;
```

### Decode (builder)

```rust
use zenjpeg::decoder::Decoder;
use zenjpeg::encoder::Unstoppable;

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

## End-to-End: decode → re-encode (server-side, with limits + cancellation)

Read a JPEG, decode it under a pixel/memory limit and a cancellation token, then
re-encode the RGB pixels at quality 80. Every type is imported with its real path.

```rust
use std::fs;
use zenjpeg::decoder::Decoder;
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout, Unstoppable};

fn transcode(input_path: &str, output_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let jpeg_bytes = fs::read(input_path)?;

    // Decode with DoS limits + a stop token. `Unstoppable` never cancels;
    // pass any `&impl zenjpeg::encoder::Stop` (e.g. a shared atomic flag) instead
    // to support user-initiated cancellation.
    let decoded = Decoder::new()
        .max_pixels(120_000_000) // reject decompression bombs (120 MP, admits ~108 MP camera photos)
        .max_memory(512 * 1024 * 1024) // cap allocation at 512 MB
        .decode(&jpeg_bytes, Unstoppable)?;

    let (width, height) = decoded.dimensions();
    let rgb: &[u8] = decoded.pixels_u8().expect("u8 output (default OutputTarget::Srgb8)");

    // Re-encode at quality 80.0, 4:2:0 chroma.
    let config = EncoderConfig::ycbcr(80.0, ChromaSubsampling::Quarter);
    let mut enc = config.encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)?;
    enc.push_packed(rgb, Unstoppable)?;
    let out: Vec<u8> = enc.finish()?;

    fs::write(output_path, &out)?;
    Ok(())
}
```

`Unstoppable` is re-exported from `zenjpeg::encoder` and is the same type the
decoder accepts, so a single import covers both the decode and encode calls. To
set all limits in one value (and reuse them via the request builder), use
[`Limits`](#per-image-metadata-three-layer-pattern):
`zenjpeg::encoder::Limits::default().max_pixels(120_000_000).max_memory(512 * 1024 * 1024)`.

## Heritage

Started as a port of [jpegli](https://github.com/libjxl/libjxl/tree/main/lib/jpegli) from Google's JPEG XL project. After six rewrites it shares ideas but little code with the original.

**From jpegli:** adaptive quantization, XYB color space, perceptual quant tables, zero-bias coefficient rounding.

**From mozjpeg:** overshoot deringing (enabled by default), trellis quantization, hybrid trellis mode.

**Our own:** pure safe Rust, streaming row-by-row API, parallel encode/decode, deblocking filters, UltraHDR gain maps, JPEG source detection and re-encoding recommendations.

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `parallel` | no | Multi-threaded encode/decode via rayon |
| `moxcms` | no | Color management (pure Rust). Required for `.correct_color()` and XYB decode |
| `ultrahdr` | no | UltraHDR HDR gain map encode/decode |
| `zencodec` | no | zencodec trait implementations for cross-codec pipelines |
| `layout` | no | Lossless transforms + lossy decode→resize→encode pipeline |
| `recompress` | no | JPEG→JPEG recompression to a target perceptual quality (see [Recompress](#recompress)). Core path; no heavy deps |
| `recompress-iqa` | no | Adds the measured closed loop to `recompress` (pulls in `zensim`) |
| `recompress-expert` | no | Exposes `recompress::expert` internals (unstable, not semver-covered) |
| `target-zq` | no | `Quality::Zq` / `Quality::ZqExplicit` closed-loop perceptual-quality encoder (pulls in `zensim`) |
| `boundary-rd` | no | Opt-in block-boundary continuity refinement (off by default; output is byte-identical unless enabled) |

The **decoder and trellis quantization are always compiled** — the historical
`decoder` and `trellis` feature flags are now no-ops kept only so existing
`features = [...]` lines keep resolving. `auto_optimize()` and the mozjpeg/hybrid
presets need no feature flag; trellis is data-gated and off by default at runtime,
turned on via `auto_optimize(true)`. The decoder API is prerelease — expect
breaking changes.

```toml
# Encode + decode (most common)
[dependencies]
zenjpeg = "0.8"

# High-performance server (parallel encode + decode)
[dependencies]
zenjpeg = { version = "0.8", features = ["parallel"] }

# Color-managed decode (XYB, ICC profiles)
[dependencies]
zenjpeg = { version = "0.8", features = ["moxcms"] }
```

## Encoder

### Color Modes

| Constructor | Use Case |
|-------------|----------|
| `EncoderConfig::ycbcr(q, sub)` | Standard JPEG (most compatible) |
| `EncoderConfig::xyb(q, b_sub)` | XYB perceptual color (better quality, needs `moxcms` to decode) |
| `EncoderConfig::grayscale(q)` | Single-channel |
| `EncoderConfig::rgb(q)` | No color transform — channels stored verbatim as R, G, B at 4:4:4 (Adobe APP14 transform=0). For channel-packed data (e.g. microscopy stains); larger files than YCbCr on photos |

Quality accepts a plain number (`85` or `85.0`, the 0–100 scale) or a `Quality`
variant (see [Quality Options](#quality-options)).

### Entry Points

| Method | Input Type | Use Case |
|--------|------------|----------|
| `encode_from_bytes(w: u32, h: u32, layout)` | `&[u8]` | Raw byte buffers |
| `encode_from_rgb::<P>(w: u32, h: u32)` | `rgb` crate types | `RGB<u8>`, `RGBA<f32>`, etc. |
| `encode_from_ycbcr_planar(w: u32, h: u32)` | `YCbCrPlanes` | Video pipeline output |

Width and height are `u32` (pixels) on every entry point. All three return a streaming `Encoder`. Push rows with `push_packed()`, finish with `finish()`. One-shot convenience: `config.request().encode(&pixels, w, h)`.

### Builder Methods

| Method | Default | Notes |
|--------|---------|-------|
| `.progressive(bool)` | `true` | ~3% smaller, ~2x slower |
| `.auto_optimize(bool)` | `false` | Best quality/size (hybrid trellis) |
| `.deringing(bool)` | `true` | Overshoot deringing for documents/graphics |
| `.separate_chroma_tables(bool)` | `true` | 3 quant tables (Y, Cb, Cr) vs 2 |
| `.huffman(strategy)` | `Optimize` | Huffman table strategy |
| `.sharp_yuv(bool)` | `false` | SharpYUV chroma downsampling |

### Quality Options

```rust
use zenjpeg::encoder::{EncoderConfig, Quality, ChromaSubsampling};

// Simple quality scale (0-100)
let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);

// Target a specific metric
let config = EncoderConfig::ycbcr(Quality::ApproxMozjpeg(80), ChromaSubsampling::Quarter);
let config = EncoderConfig::ycbcr(Quality::ApproxSsim2(90.0), ChromaSubsampling::Quarter);
let config = EncoderConfig::ycbcr(Quality::ApproxButteraugli(1.0), ChromaSubsampling::Quarter);
```

### Trellis Modes

**Default (no trellis):** adaptive quantization with perceptual zero-bias. Fast, good quality.

**Hybrid trellis (`auto_optimize(true)`):** combines jpegli AQ with mozjpeg trellis. Best quality/size tradeoff. +1.5 SSIMULACRA2 points vs default at matched file size.

```rust
let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter)
    .auto_optimize(true);
```

**Mozjpeg-compatible presets:** `MozjpegBaseline`, `MozjpegProgressive`, `HybridProgressive`, `HybridMaxCompression` via `ExpertConfig::from_preset()`.

### Per-Image Metadata (Three-Layer Pattern)

For encoding multiple images with the same config but different metadata:

```rust
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, Limits, Unstoppable};

// Layer 1: Reusable config
let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter)
    .progressive(true);

// Layer 2: Per-image request (metadata, limits, stop token)
let jpeg = config.request()
    .icc_profile(&srgb_icc_bytes)
    .limits(Limits::default().max_output(20 * 1024 * 1024)) // cap encoded size at 20 MB
    .encode(&pixels, 1920, 1080)?;

// Layer 3: Streaming execution
let mut encoder = config.request()
    .icc_profile(&p3_icc_bytes)
    .encode_from_rgb::<rgb::RGB<u8>>(1920, 1080)?;
encoder.push_packed(&pixels, Unstoppable)?;
let jpeg = encoder.finish()?;
```

Request builder methods (on `config.request()`): `.icc_profile(&[u8])`,
`.exif(impl Into<Exif>)`, `.xmp(&[u8])`, `.stop(&dyn Stop)`, and `.limits(Limits)` —
where `Limits` is `zenjpeg::encoder::Limits` built via
`Limits::default().max_pixels(n).max_memory(n).max_output(n)`.

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
| `.strictness(Strictness)` | `Balanced` | `Strictness::{Strict, Balanced, Lenient, Permissive}` (type at `zenjpeg::decoder::Strictness`) |
| `.max_pixels(u64)` | 120M | DoS protection |
| `.max_memory(u64)` | 512 MB | Memory limit |

`Strictness` is `zenjpeg::decoder::Strictness`; `.max_pixels` / `.max_memory` take a
`u64`. Example:

```rust
use zenjpeg::decoder::{Decoder, Strictness};
use zenjpeg::encoder::Unstoppable;

let result = Decoder::new()
    .strictness(Strictness::Strict)   // reject any spec violation or truncation
    .max_pixels(120_000_000)          // 120 MP cap (admits ~108 MP camera photos)
    .max_memory(512 * 1024 * 1024)    // 512 MB cap
    .decode(&jpeg_bytes, Unstoppable)?;
```

### Decode Paths

For most web JPEGs, `Decoder::new().decode(&data, stop)` hits the streaming path -- no coefficient storage, one MCU-row pass through entropy/IDCT/color/output. This is the fastest path.

Progressive, CMYK, f32 output, deblocking (Knusperli), and transforms go through the coefficient path. Parallel decode activates automatically when DRI restart markers are present and the image has 1024+ MCU blocks.

See [docs/DECODER_PATHS.md](https://github.com/imazen/zenjpeg/blob/main/zenjpeg/docs/DECODER_PATHS.md) for the full decision flow and path matrix.

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
use zenjpeg::encoder::Unstoppable;

let result = Decoder::new()
    .deblock(DeblockMode::Auto)
    .decode(&jpeg_data, Unstoppable)?;
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
use zenjpeg::decoder::{Decoder, TargetColorSpace};
use zenjpeg::encoder::Unstoppable;

let img = Decoder::new()
    .correct_color(Some(TargetColorSpace::Srgb))
    .decode(&jpeg_data, Unstoppable)?;
```

Default is `None` -- no color conversion. Pixels are returned in the JPEG's native color space.

### Lossless Transforms

Rotate, flip, and transpose by manipulating DCT coefficients directly. No decode to pixels, no re-encode, zero generation loss.

```rust
use zenjpeg::lossless::{transform, apply_exif_orientation, LosslessTransform, TransformConfig};
use zenjpeg::encoder::Unstoppable;

// Rotate 90 degrees losslessly
let rotated = transform(&jpeg_data, &TransformConfig {
    transform: LosslessTransform::Rotate90,
    ..Default::default()
}, Unstoppable)?;

// Auto-correct EXIF orientation
let oriented = apply_exif_orientation(&jpeg_data, Unstoppable)?;
```

All 8 D4 dihedral group elements: `None`, `FlipHorizontal`, `FlipVertical`, `Transpose`, `Rotate90`, `Rotate180`, `Rotate270`, `Transverse`.

### UltraHDR (requires `ultrahdr` feature)

UltraHDR embeds a gain map inside a standard JPEG so HDR-capable displays get HDR while everything else sees the SDR base image. zenjpeg handles the full stack: encode HDR → UltraHDR JPEG, decode UltraHDR JPEG → HDR pixels.

#### Encode

```rust
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, Unstoppable};
use zenjpeg::ultrahdr::{
    encode_ultrahdr, GainMapConfig, ToneMapConfig, UhdrColorGamut, UhdrColorTransfer,
    UhdrPixelFormat, UhdrRawImage,
};

// Your HDR pixels (linear RGB float, any gamut)
let hdr = UhdrRawImage::from_f32_rgb(
    &hdr_pixels, width, height,
    UhdrPixelFormat::Rgb888, UhdrColorGamut::Bt2100,
    UhdrColorTransfer::Linear,
)?;

// One call: tonemap → encode SDR base → compute gain map → assemble MPF container
let ultrahdr_jpeg = encode_ultrahdr(
    &hdr,
    &GainMapConfig::default(),   // gain map quality/resolution
    &ToneMapConfig::default(),   // SDR tonemapping parameters
    &EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter),
    75.0,                        // gain map JPEG quality
    Unstoppable,
)?;
// ultrahdr_jpeg is a standard JPEG — works everywhere, HDR on supported displays
```

#### Decode (streaming)

```rust
use zenjpeg::decoder::Decoder;
use zenjpeg::ultrahdr::{UltraHdrReaderConfig, UltraHdrMode, GainMapMemory};

let config = UltraHdrReaderConfig::new()
    .mode(UltraHdrMode::Hdr)      // HDR output (applies gain map)
    .display_boost(4.0)           // target display peak brightness ratio
    .memory_strategy(GainMapMemory::Streaming);

let mut reader = Decoder::new().ultrahdr_reader(&jpeg_data, config)?;
let width = reader.dimensions().width as usize;
let mut hdr_row = vec![0.0f32; width * 4]; // RGBA f32 per row

while !reader.is_finished() {
    reader.read_rows(1, None, Some(&mut hdr_row), None)?;
    // hdr_row contains linear f32 RGBA pixels for this row
}
```

#### Decode modes

| `UltraHdrMode` | SDR output | HDR output | Gain map | Use case |
|----------------|-----------|-----------|----------|----------|
| `SdrOnly` | yes | — | — | Fastest, ignore HDR |
| `Hdr` | — | yes | — | HDR display/processing |
| `SdrAndHdr` | yes | yes | — | Preview + HDR pipeline |
| `SdrAndGainMap` | yes | — | yes | Editing, gain map manipulation |

Memory stays bounded regardless of image size — ~500KB peak for SDR-only, ~1MB for HDR mode on 4K images.

#### Detection

```rust
use zenjpeg::decoder::Decoder;
use zenjpeg::encoder::Unstoppable;

let decoded = Decoder::new().decode(&jpeg_data, Unstoppable)?;
if let Some(extras) = decoded.extras() {
    if extras.is_ultrahdr() {
        let (metadata, _) = extras.ultrahdr_metadata().unwrap().unwrap();
        println!("Gain map max boost: {:?}", metadata.gain_map_max);
    }
}
```

Non-UltraHDR JPEGs decode normally — the feature adds zero overhead when no gain map is present.

### Cooperative Cancellation

Both encoder and decoder accept a `Stop` token for graceful shutdown. `Unstoppable`
and the `Stop` trait are both re-exported from `zenjpeg::encoder` (backed by the
`enough` crate) — you do **not** need to depend on `enough` directly. The same
`Unstoppable` value works for every decode and encode call.

The no-cancel case needs nothing beyond zenjpeg:

```rust
use zenjpeg::decoder::Decoder;
use zenjpeg::encoder::Unstoppable; // re-export; no direct `enough` dep needed

let image = Decoder::new().decode(&jpeg_data, Unstoppable)?;
```

For real cancellation, implement the `Stop` trait on your own type (an `AtomicBool`
flag is the usual choice). The trait's one required method is
`check(&self) -> Result<(), StopReason>`; `StopReason` lives in the `enough` crate,
so this case needs `enough` as a direct dependency:

```toml
[dependencies]
zenjpeg = "0.8"
enough = "0.4"
```

```rust
use std::sync::atomic::{AtomicBool, Ordering};
use enough::{Stop, StopReason};
use zenjpeg::decoder::Decoder;

struct CancelFlag(AtomicBool);
impl Stop for CancelFlag {
    fn check(&self) -> Result<(), StopReason> {
        if self.0.load(Ordering::Relaxed) {
            Err(StopReason::Cancelled)
        } else {
            Ok(())
        }
    }
}

let cancel_token = CancelFlag(AtomicBool::new(false));
// ... a watchdog thread can flip the flag: cancel_token.0.store(true, Ordering::Relaxed);
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

## Recompress

Requires the `recompress` feature. Recompress an already-encoded JPEG to a
target **zensim Profile A** quality (`[0, 100]`, higher = closer to the
original) with **minimal generation loss** and **no size regression**. One
entry point routes between `NoOp` / `Lossless` (scan re-pack) / `Preserve`
(coefficient-domain requant, incl. same-family Robidoux retargeting for
mozjpeg/ImageMagick) / `Tuned` / `Deblock`, picking the smallest output that
hits the target, using per-encoder calibration (libjpeg-turbo, mozjpeg,
jpegli) fit on 50 CID22-512 references.

```rust
use zenjpeg::recompress::{recompress, Budget, Confidence, RecompressOptions};

let opts = RecompressOptions::new(80.0)        // target zensim-A, 0..=100
    .with_budget(Budget::OneShot)               // default; no IQA loop
    .with_confidence(Confidence::P50);          // P25/P50/P75/P90/P95 delivery confidence

let result = recompress(&jpeg_bytes, &opts)?;

// `output_bytes()` is `Some` for a recompressed/lossless result, `None` for
// NoOp (source already meets target — keep the input):
let out: &[u8] = result.output_bytes().unwrap_or(&jpeg_bytes);
# Ok::<(), zenjpeg::recompress::Error>(())
```

The result carries `strategy`, the projected quality vs the original, and (with
the IQA loop) a measured generation-loss score. **Budget:** `OneShot` (default,
calibration-only, needs only the `recompress` feature) or — with
`recompress-iqa` — `MaxIterations(n)` / `MaxTime(d)`, which measure generation
loss and bump the dial to land closer to target. **Confidence** shifts the
internal aim so a chosen quantile of images clears the target (content variance
is large, so a bare target under-delivers on ~half the images at P50).

Invariants: output is **never larger than the source** (lossless fallback +
byte-level guard), and the user's target — not the confidence-shifted aim —
gates the `NoOp` decision. See
[docs/recompress/RECOMPRESSION_COMPENDIUM.md](https://github.com/imazen/zenjpeg/blob/main/docs/recompress/RECOMPRESSION_COMPENDIUM.md)
for the strategy taxonomy, generation-loss math, and calibration provenance.

## Performance

Decode is competitive with libjpeg-turbo (C+NASM) on baseline and faster on
progressive; encode lands within ~15–20% of C++ jpegli at matched quality, and
parallel decode scales near-linearly when restart markers are present. All
numbers below were measured on a Ryzen 9 7950X (WSL2) with the default compiler
target (**no** `-C target-cpu=native`). Full methodology, competitor versions,
and pinned-commit reproduction:
**[benchmarks/README.md](https://github.com/imazen/zenjpeg/blob/main/benchmarks/README.md)**.


## Known Limitations

- **Baseline decode speed**: ~6% slower than libjpeg-turbo (C+NASM) on baseline JPEGs with the default Triangle upsampling. Faster on progressive.
- **XYB decode speed**: XYB images use the f32 pipeline; standard JPEGs use the fast integer IDCT.
- **XYB file size**: Baseline mode is 2-3% larger than C++ jpegli. Progressive mode matches or beats.
- **Decoder API is prerelease**: expect breaking changes.

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

## SIMD and Platform Support

SIMD dispatch is handled by [archmage](https://github.com/imazen/archmage) at
runtime — no compile-time target flags required. Supported instruction sets:

- **x86-64**: AVX2/FMA, AVX-512
- **aarch64**: NEON
- **wasm32**: SIMD128 (build with `RUSTFLAGS="-C target-feature=+simd128"`)
- Scalar fallback on all other targets

CI runs on Ubuntu x86-64, Ubuntu ARM64, macOS ARM64, macOS Intel, Windows x86-64, Windows ARM64, and i686 (via `cross`).

## Development

```bash
cargo test --release                    # ~930 tests, no external deps
cargo test --release --test cpp_parity_locked  # Quick C++ parity check
cargo test --release -- --ignored       # Full suite (needs C++ build + corpus)
```

## Acknowledgments

Built on ideas from [jpegli](https://github.com/libjxl/libjxl/tree/main/lib/jpegli)
(Google, BSD-3-Clause) and [mozjpeg](https://github.com/mozilla/mozjpeg)
(Mozilla). After six rewrites from the initial jpegli port, zenjpeg is an
independent project with its own architecture, streaming pipeline, and quality
optimizations.

## AI Disclosure

Developed with assistance from Claude (Anthropic). Extensively tested against the
C++ reference with 930+ tests. Report issues at <https://github.com/imazen/zenjpeg/issues>.

## License

Dual-licensed: [AGPL-3.0](https://github.com/imazen/zenjpeg/blob/main/LICENSE-AGPL3) or [commercial](https://github.com/imazen/zenjpeg/blob/main/LICENSE-COMMERCIAL).

I've maintained and developed open-source image server software -- and the 40+
library ecosystem it depends on -- full-time since 2011. Fifteen years of
continual maintenance, backwards compatibility, support, and the (very rare)
security patch. That kind of stability requires sustainable funding, and
dual-licensing is how we make it work without venture capital or rug-pulls.
Support sustainable and secure software; swap patch tuesday for patch leap-year.

[Our open-source products](https://www.imazen.io/open-source)

**Your options:**

- **Startup license** -- $1 if your company has under $1M revenue and fewer
  than 5 employees. [Get a key](https://www.imazen.io/pricing)
- **Commercial subscription** -- Governed by the Imazen Site-wide Subscription
  License v1.1 or later. Apache 2.0-like terms, no source-sharing requirement.
  Sliding scale by company size.
  [Pricing & 60-day free trial](https://www.imazen.io/pricing)
- **AGPL v3** -- Free and open. Share your source if you distribute.

See [LICENSE-COMMERCIAL](https://github.com/imazen/zenjpeg/blob/main/LICENSE-COMMERCIAL) for details.

## Image tech I maintain

| | |
|:--|:--|
| **Codecs** ¹ | **zenjpeg** · [zenpng] · [zenwebp] · [zengif] · [zenavif] · [zenjxl] · [zenjxl-decoder] · [jxl-encoder] · [zenbitmaps] · [heic] · [zentiff] · [zenpdf] · [zensvg] · [zenjp2] · [zenraw] · [ultrahdr] |
| Codec internals | [zenrav1e] · [rav1d-safe] · [zenravif] · [zenavif-parse] · [zenavif-serialize] |
| Compression | [zenflate] · [zenzop] · [zenzstd] |
| Processing | [zenresize] · [zenquant] · [zenblend] · [zenfilters] · [zensally] · [zentone] |
| Pixels & color | [zenpixels] · [zenpixels-convert] · [linear-srgb] · [garb] · [zenyuv] |
| Pipeline & framework | [zenpipe] · [zencodec] · [zencodecs] · [zenlayout] · [zennode] · [zenwasm] · [zentract] |
| Metrics | [zensim] · [fast-ssim2] · [butteraugli] · [zenmetrics] · [resamplescope-rs] |
| Pickers & ML | [zenanalyze] · [zenpredict] · [zenpicker] · [zenanalyze-api] |
| Test corpora | [codec-corpus] · [imazen-26] |
| Products | [Imageflow] image engine ([.NET][imageflow-dotnet] · [Node][imageflow-node] · [Go][imageflow-go]) · [Imageflow Server] · [ImageResizer] (C#) |

<sub>¹ pure-Rust, `#![forbid(unsafe_code)]` codecs, as of 2026</sub>

### General Rust awesomeness

[zenbench] · [archmage] · [magetypes] · [enough] · [whereat] · [cargo-copter] · [zenutils]

[Open source](https://www.imazen.io/open-source) · [@imazen](https://github.com/imazen) · [@lilith](https://github.com/lilith) · [lib.rs/~lilith](https://lib.rs/~lilith)

[zenpng]: https://github.com/imazen/zenpng
[zenwebp]: https://github.com/imazen/zenwebp
[zengif]: https://github.com/imazen/zengif
[zenavif]: https://github.com/imazen/zenavif
[zenjxl]: https://github.com/imazen/zenjxl
[zenjxl-decoder]: https://github.com/imazen/zenjxl-decoder
[jxl-encoder]: https://github.com/imazen/jxl-encoder
[zenbitmaps]: https://github.com/imazen/zenbitmaps
[heic]: https://github.com/imazen/heic
[zentiff]: https://github.com/imazen/zenextras
[zenpdf]: https://github.com/imazen/zenextras
[zensvg]: https://github.com/imazen/zenextras
[zenjp2]: https://github.com/imazen/zenextras
[zenraw]: https://github.com/imazen/zenraw
[ultrahdr]: https://github.com/imazen/ultrahdr
[zenrav1e]: https://github.com/imazen/zenrav1e
[rav1d-safe]: https://github.com/imazen/rav1d-safe
[zenravif]: https://github.com/imazen/cavif-rs
[zenavif-parse]: https://github.com/imazen/zenavif
[zenavif-serialize]: https://github.com/imazen/zenavif
[zenflate]: https://github.com/imazen/zenflate
[zenzop]: https://github.com/imazen/zenzop
[zenzstd]: https://github.com/imazen/zenzstd
[zenresize]: https://github.com/imazen/zenresize
[zenquant]: https://github.com/imazen/zenquant
[zenblend]: https://github.com/imazen/zenblend
[zenfilters]: https://github.com/imazen/zenpipe
[zensally]: https://github.com/imazen/zensally
[zentone]: https://github.com/imazen/zentone
[zenpixels]: https://github.com/imazen/zenpixels
[zenpixels-convert]: https://github.com/imazen/zenpixels
[linear-srgb]: https://github.com/imazen/linear-srgb
[garb]: https://github.com/imazen/garb
[zenyuv]: https://github.com/imazen/zenjpeg
[zenpipe]: https://github.com/imazen/zenpipe
[zencodec]: https://github.com/imazen/zencodec
[zencodecs]: https://github.com/imazen/zenpipe
[zenlayout]: https://github.com/imazen/zenpipe
[zennode]: https://github.com/imazen/zennode
[zenwasm]: https://github.com/imazen/zenwasm
[zentract]: https://github.com/imazen/zentract
[zensim]: https://github.com/imazen/zensim
[fast-ssim2]: https://github.com/imazen/fast-ssim2
[butteraugli]: https://github.com/imazen/butteraugli
[zenmetrics]: https://github.com/imazen/zenmetrics
[resamplescope-rs]: https://github.com/imazen/resamplescope-rs
[zenanalyze]: https://github.com/imazen/zenanalyze
[zenpredict]: https://github.com/imazen/zenanalyze
[zenpicker]: https://github.com/imazen/zenanalyze
[zenanalyze-api]: https://github.com/imazen/zenanalyze
[codec-corpus]: https://github.com/imazen/codec-corpus
[imazen-26]: https://github.com/imazen/imazen-26
[zenbench]: https://github.com/imazen/zenbench
[archmage]: https://github.com/imazen/archmage
[magetypes]: https://github.com/imazen/archmage
[enough]: https://github.com/imazen/enough
[whereat]: https://github.com/lilith/whereat
[cargo-copter]: https://github.com/imazen/cargo-copter
[zenutils]: https://github.com/imazen/zenutils
[Imageflow]: https://github.com/imazen/imageflow
[Imageflow Server]: https://github.com/imazen/imageflow-dotnet-server
[ImageResizer]: https://github.com/imazen/resizer
[imageflow-dotnet]: https://github.com/imazen/imageflow-dotnet
[imageflow-node]: https://github.com/imazen/imageflow-node
[imageflow-go]: https://github.com/imazen/imageflow-go
