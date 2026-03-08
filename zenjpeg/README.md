# zenjpeg

[![Crates.io](https://img.shields.io/crates/v/zenjpeg.svg)](https://crates.io/crates/zenjpeg)
[![Documentation](https://docs.rs/zenjpeg/badge.svg)](https://docs.rs/zenjpeg)
[![CI](https://github.com/imazen/zenjpeg/actions/workflows/ci.yml/badge.svg)](https://github.com/imazen/zenjpeg/actions/workflows/ci.yml)
[![License: AGPL/Commercial](https://img.shields.io/badge/License-AGPL%2FCommercial-blue.svg)](https://github.com/imazen/zenjpeg/blob/main/LICENSE)

A pure Rust JPEG encoder and decoder with perceptual optimizations.

> **Important:** The decoder requires the `decoder` feature flag:
> ```toml
> [dependencies]
> zenjpeg = { version = "0.6", features = ["decoder"] }
> ```
> See [Feature Flags](#feature-flags) for details.

> **Note:** This crate was previously published as `jpegli-rs`. If migrating, update your imports from `use jpegli::` to `use zenjpeg::`.

## Heritage and Divergence

This project started as a port of [jpegli](https://github.com/libjxl/libjxl/tree/main/lib/jpegli), Google's improved JPEG encoder from the JPEG XL project. After six rewrites it has diverged significantly into an independent project.

**Ideas adopted from jpegli:**
- Adaptive quantization (content-aware bit allocation)
- XYB color space with ICC profiles (progressive mode recommended for best compression)
- Perceptually-tuned quantization tables
- Zero-bias strategies for coefficient rounding

**Ideas adopted from mozjpeg:**
- Overshoot deringing for documents/graphics
- Trellis quantization for optimal coefficient selection
- Hybrid trellis mode (experimental, see Trellis Modes below)

**Where we went our own way:**
- Pure Rust, `#![forbid(unsafe_code)]` unconditionally (SIMD via safe archmage tokens)
- Streaming encoder API for memory efficiency (process images row-by-row)
- Portable SIMD via `wide` crate instead of platform intrinsics
- Parallel encoding support
- UltraHDR support (HDR gain maps for backward-compatible HDR JPEGs)
- Independent optimizations and bug fixes

## Features

- **Pure Rust** - No C/C++ dependencies, builds anywhere Rust does
- **Perceptual optimization** - Adaptive quantization for better visual quality at smaller sizes
- **Trellis quantization** - Optimal coefficient selection from mozjpeg
- **Overshoot deringing** - Eliminates ringing artifacts on documents and graphics (enabled by default)
- **Backward compatible** - Produces standard JPEG files readable by any decoder
- **SIMD accelerated** - Portable SIMD via `wide` crate
- **Streaming API** - Memory-efficient row-by-row encoding for large images
- **Parallel encoding** - Multi-threaded for large images (1024x1024+)
- **Lossless transforms** - Rotate/flip/transpose in DCT domain with zero generation loss
- **JPEG detection** - Identify source encoder, estimate quality, and get optimal re-encoding settings from headers alone (<1µs)
- **UltraHDR support** - Encode/decode HDR gain maps (optional `ultrahdr` feature)
- **Color management** - Optional ICC profile support

## Known Limitations

- **XYB color space** - With progressive mode, matches or beats C++ jpegli file sizes. Baseline mode is 2-3% larger.
- **XYB decoder speed** - XYB images use f32 pipeline; standard JPEG decoding uses fast integer IDCT.

## Trellis Modes

zenjpeg supports three quantization modes:

### Standard (jpegli-style)
Default mode. Uses adaptive quantization with perceptual zero-bias. Good balance of speed and quality.

```rust
let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter);
```

### Standalone Trellis (mozjpeg-style)
Rate-distortion optimized coefficient selection. Typically 10-15% smaller files at equivalent quality.
Slightly slower due to dynamic programming optimization.

```rust
use zenjpeg::encode::{ExpertConfig, OptimizationPreset, ColorMode, ChromaSubsampling};

let expert = ExpertConfig::from_preset(OptimizationPreset::MozjpegBaseline, 85);
let config = expert.to_encoder_config(ColorMode::YCbCr {
    subsampling: ChromaSubsampling::Quarter,
});
```

### Hybrid Trellis (recommended)
Combines jpegli's adaptive quantization with mozjpeg's trellis. **This is our best mode**
and is enabled via `.auto_optimize(true)`:

- **+1.5 SSIM2 points** vs jpegli at matched file size
- **-1.5% to -2% smaller files** at matched quality
- Works across q50-q95 range

```rust
use zenjpeg::encoder::{EncoderConfig, ChromaSubsampling};

// Recommended: use auto_optimize for best results
let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
    .auto_optimize(true);
```

## Quick Start

### Encode

```rust
use zenjpeg::encoder::{EncoderConfig, PixelLayout, ChromaSubsampling, Unstoppable};

// Best quality/size with auto_optimize
let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
    .auto_optimize(true);
let mut enc = config.encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)?;
enc.push_packed(&rgb_bytes, Unstoppable)?;
let jpeg_bytes: Vec<u8> = enc.finish()?;
```

### Decode

Requires `features = ["decoder"]` (prerelease API).

```rust
use zenjpeg::decoder::Decoder;
use enough::Unstoppable;

let result = Decoder::new().decode(&jpeg_bytes, Unstoppable)?;
let rgb_pixels: &[u8] = result.pixels_u8().expect("u8 output");
let (width, height) = result.dimensions();
```

## Resource Limits and Cancellation

### Resource Limits (DoS Protection)

Protect against malicious images that could exhaust memory or CPU:

```rust
use zenjpeg::decoder::Decoder;
use zenjpeg::types::Limits;

// Set limits individually
let decoder = Decoder::new()
    .max_pixels(100_000_000)      // 100 megapixels max
    .max_memory(512_000_000);     // 512 MB max allocation

// Or use Limits struct
let limits = Limits {
    max_pixels: Some(100_000_000),
    max_memory: Some(512_000_000),
    max_output: None,
};
let decoder = Decoder::new().limits(limits);
```

**Default limits:**
- `max_pixels`: 100 megapixels
- `max_memory`: 512 MB

Set to `0` or `None` for unlimited (not recommended for untrusted input).

### Cooperative Cancellation

Use `Stop` tokens for graceful shutdown in long-running operations:

```rust
use enough::{Stop, Unstoppable};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

// Simple case: never cancel
let image = Decoder::new().decode(&jpeg_data, Unstoppable)?;

// Custom stop token (e.g., user clicked cancel button)
struct CancelToken {
    cancelled: Arc<AtomicBool>,
}

impl Stop for CancelToken {
    fn should_stop(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }
}

let cancel = CancelToken {
    cancelled: Arc::new(AtomicBool::new(false)),
};

// Decode with cancellation support
let result = Decoder::new().decode(&jpeg_data, &cancel);

// In another thread: cancel.cancelled.store(true, Ordering::Relaxed);
```

**Encoder cancellation:**
```rust
let mut encoder = config.encode_from_bytes(width, height, layout)?;
encoder.push_packed(&pixels, &cancel_token)?;  // Can be cancelled during push
let jpeg = encoder.finish()?;
```

## Per-Image Metadata (Three-Layer Pattern)

For encoding multiple images with the same config but different metadata:

```rust
use zenjpeg::encoder::{EncoderConfig, ChromaSubsampling, Exif, Orientation};

// Layer 1: Reusable config (quality, color mode, optimization settings)
let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
    .auto_optimize(true)
    .progressive(true);

// Layer 2: Per-image request (metadata, limits, stop token)
// Image 1: sRGB with orientation
let jpeg1 = config.request()
    .icc_profile(&srgb_icc_bytes)
    .exif(Exif::build().orientation(Orientation::Rotate90))
    .encode(&pixels1, 1920, 1080)?;

// Image 2: Display P3 with different metadata
let jpeg2 = config.request()
    .icc_profile(&p3_icc_bytes)
    .exif(Exif::build().copyright("© 2024 Example Corp"))
    .encode(&pixels2, 3840, 2160)?;

// Image 3: No metadata, with cancellation
let jpeg3 = config.request()
    .stop(&cancel_token)
    .encode(&pixels3, 800, 600)?;
```

**Why three layers?**
1. **EncoderConfig** - Reusable settings (quality, color mode, progressive)
2. **EncodeRequest** - Per-image data (ICC profile, EXIF, XMP, limits, stop token)
3. **Encoder** - Streaming execution (push rows, finish)

**Request builder methods:**
- `.icc_profile(&[u8])` - Borrowed ICC profile
- `.icc_profile_owned(Vec<u8>)` - Owned ICC profile
- `.exif(Exif)` - EXIF metadata
- `.xmp(&[u8])` / `.xmp_owned(Vec<u8>)` - XMP metadata
- `.stop(&dyn Stop)` - Cancellation token
- `.limits(Limits)` - Resource limits (encoder future feature)

**Streaming with request:**
```rust
let mut encoder = config.request()
    .icc_profile(&srgb_bytes)
    .encode_from_rgb::<rgb::RGB<u8>>(1920, 1080)?;

encoder.push_packed(&pixels, Unstoppable)?;
let jpeg = encoder.finish()?;
```

## API Reference

### Encoder API

All encoder types are in `zenjpeg::encoder`:

```rust
use zenjpeg::encoder::{
    EncoderConfig, PixelLayout, Quality, ChromaSubsampling, Unstoppable
};
```

#### Quick Start

```rust
use zenjpeg::encoder::{EncoderConfig, PixelLayout, ChromaSubsampling, Unstoppable};

// Create reusable config (quality and color mode set in constructor)
let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
    .progressive(true);

// Encode from raw bytes
let mut enc = config.encode_from_bytes(1920, 1080, PixelLayout::Rgb8Srgb)?;
enc.push_packed(&rgb_bytes, Unstoppable)?;
let jpeg = enc.finish()?;
```

#### Three Encoder Entry Points

| Method | Input Type | Use Case |
|--------|------------|----------|
| `encode_from_bytes(w, h, layout)` | `&[u8]` | Raw byte buffers |
| `encode_from_rgb::<P>(w, h)` | `rgb` crate types | `RGB<u8>`, `RGBA<f32>`, etc. |
| `encode_from_ycbcr_planar(w, h)` | `YCbCrPlanes` | Video decoder output |

#### Examples

```rust
use zenjpeg::encoder::{EncoderConfig, PixelLayout, ChromaSubsampling, Unstoppable};

let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter);

// From raw RGB bytes
let mut enc = config.encode_from_bytes(800, 600, PixelLayout::Rgb8Srgb)?;
enc.push_packed(&rgb_bytes, Unstoppable)?;
let jpeg = enc.finish()?;

// From rgb crate types
use rgb::RGB;
let mut enc = config.encode_from_rgb::<RGB<u8>>(800, 600)?;
enc.push_packed(&pixels, Unstoppable)?;
let jpeg = enc.finish()?;

// From planar YCbCr (video pipelines)
let mut enc = config.encode_from_ycbcr_planar(1920, 1080)?;
enc.push(&planes, num_rows, Unstoppable)?;
let jpeg = enc.finish()?;
```

#### EncoderConfig Constructors

Choose one constructor based on desired color mode:

| Constructor | Color Mode | Use Case |
|-------------|------------|----------|
| `EncoderConfig::ycbcr(q, sub)` | YCbCr | Standard JPEG (most compatible) |
| `EncoderConfig::xyb(q, b_sub)` | XYB | Perceptual color space (better quality) |
| `EncoderConfig::grayscale(q)` | Grayscale | Single-channel output |

#### Builder Methods

| Method | Description | Default |
|--------|-------------|---------|
| `.auto_optimize(bool)` | **Best quality/size** - enables hybrid trellis λ=14.5 | `false` |
| `.progressive(bool)` | Progressive JPEG (3-7% smaller) | `true` |
| `.huffman(impl Into<HuffmanStrategy>)` | Huffman table strategy | `Optimize` |
| `.deringing(bool)` | Overshoot deringing for documents/graphics | `true` |
| `.sharp_yuv(bool)` | SharpYUV downsampling | `false` |
| `.separate_chroma_tables(bool)` | Use 3 quant tables (Y, Cb, Cr) vs 2 (Y, shared) | `true` |
| `.icc_profile(bytes)` | Attach ICC profile | None |
| `.exif(exif)` | Embed EXIF metadata | None |
| `.xmp(data)` | Embed XMP metadata | None |
| `.restart_interval(n)` | MCUs between restart markers | 0 |

#### Quality Options

```rust
use zenjpeg::encoder::{EncoderConfig, Quality, ChromaSubsampling};

// Simple quality scale (0-100)
let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter);

// Quality enum variants
let config = EncoderConfig::ycbcr(
    Quality::ApproxJpegli(85.0),  // Default scale
    ChromaSubsampling::Quarter
);
// Or: Quality::ApproxMozjpeg(80)      - Match mozjpeg output
// Or: Quality::ApproxSsim2(90.0)      - Target SSIMULACRA2 score
// Or: Quality::ApproxButteraugli(1.0) - Target butteraugli distance
```

#### Pixel Layouts

| Layout | Bytes/px | Notes |
|--------|----------|-------|
| `Rgb8Srgb` | 3 | Default, sRGB gamma |
| `Bgr8Srgb` | 3 | Windows/GDI order |
| `Rgba8Srgb` / `Rgbx8Srgb` | 4 | Alpha/pad ignored |
| `Bgra8Srgb` / `Bgrx8Srgb` | 4 | BGR + alpha/pad ignored |
| `Gray8Srgb` | 1 | Grayscale sRGB |
| `Rgb16Linear` / `Rgba16Linear` | 6/8 | 16-bit linear |
| `RgbF32Linear` / `RgbaF32Linear` | 12/16 | HDR float (0.0-1.0) |
| `YCbCr8` / `YCbCrF32` | 3/12 | Pre-converted YCbCr |

#### Chroma Subsampling

```rust
use zenjpeg::encoder::{EncoderConfig, ChromaSubsampling, XybSubsampling};

// YCbCr subsampling
let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter);  // 4:2:0 (best compression)
let config = EncoderConfig::ycbcr(85, ChromaSubsampling::None);     // 4:4:4 (best quality)
let config = EncoderConfig::ycbcr(85, ChromaSubsampling::HalfHorizontal); // 4:2:2
let config = EncoderConfig::ycbcr(85, ChromaSubsampling::HalfVertical);   // 4:4:0

// XYB B-channel subsampling
let config = EncoderConfig::xyb(85, XybSubsampling::BQuarter); // B at 4:2:0
let config = EncoderConfig::xyb(85, XybSubsampling::Full);    // No subsampling
```

#### Resource Estimation

```rust
use zenjpeg::encoder::{EncoderConfig, ChromaSubsampling};

let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter);

// Typical memory estimate
let estimate = config.estimate_memory(1920, 1080);

// Guaranteed upper bound (for resource reservation)
let ceiling = config.estimate_memory_ceiling(1920, 1080);
```

---

### Decoder API

> **Prerelease:** The decoder API is behind the `decoder` feature flag and will have breaking changes.
> Enable with `zenjpeg = { version = "...", features = ["decoder"] }`.

All decoder types are in `zenjpeg::decoder`:

```rust
use zenjpeg::decoder::{Decoder, DecodeResult};
```

#### Basic Decoding

```rust
use zenjpeg::decoder::Decoder;
use enough::Unstoppable;

// Decode to u8 RGB (default)
let result = Decoder::new().decode(&jpeg_data, Unstoppable)?;
let pixels: &[u8] = result.pixels_u8().expect("u8 output");
let (width, height) = result.dimensions();
```

#### High-Precision Decoding (f32)

Use `OutputTarget` for f32 output with different transfer functions:

```rust
use zenjpeg::decoder::{Decoder, OutputTarget};
use enough::Unstoppable;

// sRGB gamma-encoded f32 (0.0-1.0 range)
let result = Decoder::new()
    .output_target(OutputTarget::SrgbF32)
    .decode(&jpeg_data, Unstoppable)?;
let pixels: &[f32] = result.pixels_f32().expect("f32 output");

// Linear light f32 (for compositing, HDR)
let result = Decoder::new()
    .output_target(OutputTarget::LinearF32)
    .decode(&jpeg_data, Unstoppable)?;

// Convert f32 to u8 or u16 when needed
let u8_pixels: Option<Vec<u8>> = result.to_u8();
let u16_pixels: Option<Vec<u16>> = result.to_u16();
```

#### YCbCr Output (Zero Color Conversion)

For video pipelines or re-encoding:

```rust
use zenjpeg::decoder::{Decoder, DecodedYCbCr};

let ycbcr: DecodedYCbCr = Decoder::new().decode_to_ycbcr_f32(&jpeg_data)?;
// Access Y, Cb, Cr planes directly (f32, range [-128, 127])
```

#### Reading JPEG Info Without Decoding

```rust
let info = Decoder::new().read_info(&jpeg_data)?;
println!("{}x{}, {} components", info.width, info.height, info.num_components);
```

#### Decoder Options

| Method | Description | Default |
|--------|-------------|---------|
| `.output_format(fmt)` | Output pixel format | `Rgb` |
| `.fancy_upsampling(bool)` | Smooth chroma upsampling | `true` |
| `.block_smoothing(bool)` | DCT block edge smoothing | `false` |
| `.apply_icc(bool)` | Apply embedded ICC profile | `true` |
| `.dequant_bias(bool)` | Laplacian dequantization biases (see below) | `false` |
| `.auto_orient(bool)` | Apply EXIF orientation in DCT domain | `false` |
| `.transform(t)` | Lossless transform during decode (see below) | None |
| `.max_pixels(n)` | Pixel count limit (DoS protection) | 100M |
| `.max_memory(n)` | Memory limit in bytes | 512 MB |

#### Output Formats

| `PixelFormat` | Bytes/px | Description |
|---------------|----------|-------------|
| `Rgb` | 3 | R-G-B (default) |
| `Bgr` | 3 | B-G-R (Windows/GDI) |
| `Rgba` | 4 | R-G-B-A, alpha = 255 |
| `Bgra` | 4 | B-G-R-A, alpha = 255 |
| `Bgrx` | 4 | B-G-R-X, pad = 255 |
| `Gray` | 1 | Grayscale |

All formats work with buffered decode (`.decode()`), the fast i16 path,
and the streaming scanline reader.

#### Decoded Image Methods

```rust
let image = Decoder::new().decode(&jpeg_data)?;

image.width()           // Image width
image.height()          // Image height
image.dimensions()      // (width, height) tuple
image.pixels()          // &[u8] pixel data
image.bytes_per_pixel() // Bytes per pixel for format
image.stride()          // Bytes per row
```

#### DecoderConfig (Advanced)

```rust
use zenjpeg::decoder::{Decoder, DecoderConfig};

// Most users should use the builder methods instead:
let image = Decoder::new()
    .fancy_upsampling(true)
    .block_smoothing(false)
    .apply_icc(true)
    .dequant_bias(false)
    .max_pixels(100_000_000)
    .max_memory(512 * 1024 * 1024)
    .decode(&jpeg_data)?;

// Or construct DecodeConfig directly:
let decoder = DecodeConfig::default();
```

#### Streaming Decode (Scanline Reader)

Decode row-by-row for minimal memory usage:

```rust
use zenjpeg::decoder::Decoder;
use imgref::ImgRefMut;

let mut reader = Decoder::new().scanline_reader(&jpeg_data)?;
let (w, h) = (reader.width() as usize, reader.height() as usize);
let mut buf = vec![0u8; w * h * 4];

let mut rows = 0;
while !reader.is_finished() {
    let slice = &mut buf[rows * w * 4..];
    let output = ImgRefMut::new(slice, w * 4, h - rows);
    rows += reader.read_rows_bgra8(output)?;
}
```

| Method | Bytes/px | Format |
|--------|----------|--------|
| `read_rows_rgb8()` | 3 | R-G-B |
| `read_rows_bgr8()` | 3 | B-G-R |
| `read_rows_rgbx8()` | 4 | R-G-B-X (pad=255) |
| `read_rows_rgba8()` | 4 | R-G-B-A (A=255) |
| `read_rows_bgra8()` | 4 | B-G-R-A (A=255) |
| `read_rows_bgrx8()` | 4 | B-G-R-X (pad=255) |
| `read_rows_rgba_f32()` | 16 | Linear f32 RGBA |
| `read_rows_gray8()` | 1 | Grayscale u8 |
| `read_rows_gray_f32()` | 4 | Grayscale f32 |

---

### Detect API (Encoder Identification & Re-encoding)

Identify the encoder and quality of any JPEG from its headers (~500 bytes, <1µs),
then get optimal re-encoding settings. Requires `features = ["decoder"]`.

All detect types are in `zenjpeg::detect`:

```rust
use zenjpeg::detect::{probe, EncoderFamily, QualityScale, ReencodeSettings};
```

#### Probing a JPEG

```rust
use zenjpeg::detect::probe;

let info = probe(&jpeg_data)?;
println!("Encoder: {:?}", info.encoder);       // e.g., Mozjpeg, LibjpegTurbo, CjpegliYcbcr
println!("Quality: {:.0} ({:?})", info.quality.value, info.quality.scale);
println!("{}x{}, {:?}, {:?}", info.dimensions.width, info.dimensions.height,
    info.subsampling, info.mode);
```

`probe()` reads only JPEG headers — no entropy decoding, no pixel buffers.

#### Re-encoding Recommendations

Different source encoders need different zenjpeg quality settings to match perceptual
quality. A turbo Q85 JPEG needs zen Q90 to avoid degradation, while a jpegli Q85 only
needs zen Q85. The calibration data comes from a 25-image sweep across three encoder
families at six quality levels, measuring butteraugli delta at 13 zen quality points
(22K data points total).

```rust
use zenjpeg::detect::probe;
use zenjpeg::encoder::{EncoderConfig, ChromaSubsampling};

let info = probe(&source_jpeg)?;

// Default: barely perceptible degradation (BA delta <= 0.3)
let config = EncoderConfig::ycbcr(
    info.recommended_quality(),
    info.recommended_subsampling(),
).auto_optimize(true);
```

#### Configurable Quality/Size Tradeoff

`reencode_settings()` takes a butteraugli tolerance and returns both quality and
subsampling. Higher tolerance = smaller files, more degradation.

| Tolerance | Meaning | Typical size savings |
|-----------|---------|---------------------|
| 0.1 | Nearly imperceptible | Minimal |
| 0.3 | Barely perceptible (default) | 0-10% |
| 0.5 | Noticeable on close inspection | 5-30% |
| 1.0 | Visible but acceptable | 15-45% |
| 2.0 | Significant quality loss | 30-60% |

```rust
use zenjpeg::detect::{probe, ReencodeError};

let info = probe(&source_jpeg)?;

// Aggressive compression — allow 0.5 BA delta
match info.reencode_settings(0.5) {
    Ok(settings) => {
        let config = EncoderConfig::ycbcr(settings.quality, settings.subsampling)
            .auto_optimize(true);
        // encode...
    }
    Err(ReencodeError::ToleranceTooTight { min_achievable, best_effort }) => {
        // Requested tolerance stricter than achievable (e.g., turbo Q90 at tol=0.1)
        // min_achievable: smallest BA delta possible at Q97
        // best_effort: Q97 settings if you want to proceed anyway
        eprintln!("Min achievable delta: {min_achievable:.2}");
    }
    Err(ReencodeError::InvalidTolerance) => {
        // Tolerance was <= 0
    }
}
```

#### Quality Ceilings for Downscaled Re-encodes

When downscaling before re-encoding, there's a quality ceiling above which extra
bytes produce imperceptible gains. At all tested ratios (1.5x-4x), going from Q90
to Q95 costs ~40% more bytes for <0.3 butteraugli improvement.

```rust
use zenjpeg::detect::JpegProbe;

let ceiling = JpegProbe::quality_ceiling(2.0);  // 2x downscale → Q90 ceiling
```

#### Detected Encoder Families

| `EncoderFamily` | Source | Quality Scale |
|-----------------|--------|---------------|
| `LibjpegTurbo` | libjpeg-turbo / IJG | IJG quality (1-100) |
| `Mozjpeg` | mozjpeg | mozjpeg quality (1-100) |
| `CjpegliYcbcr` | jpegli (YCbCr mode) | Butteraugli distance |
| `CjpegliXyb` | jpegli (XYB mode) | Butteraugli distance |
| `ImageMagick` | ImageMagick (optimized Huffman) | IJG quality (1-100) |
| `IjgFamily` | IJG-compatible (other) | IJG quality (1-100) |
| `Unknown` | Unrecognized | IJG quality (estimated) |

## Performance

### Encoding Speed

| Image Size | Sequential | Progressive | Notes |
|------------|------------|-------------|-------|
| 512x512 | 118 MP/s | 58 MP/s | Small images |
| 1024x1024 | 92 MP/s | 36 MP/s | Medium images |
| 2048x2048 | 87 MP/s | 46 MP/s | Large images |

### Sequential vs Progressive

| Quality | Seq Size | Prog Size | Prog Δ | Prog Slowdown |
|---------|----------|-----------|--------|---------------|
| Q50 | 322 KB | 313 KB | **-2.8%** | 2.5x |
| Q70 | 429 KB | 416 KB | **-3.0%** | 2.0x |
| Q85 | 586 KB | 568 KB | **-3.1%** | 2.1x |
| Q95 | 915 KB | 887 KB | **-3.1%** | 2.2x |

**Progressive produces ~3% smaller files** at the same quality, but takes ~2x longer.

**Recommendation:**
- Use **Sequential** for: real-time encoding, high throughput
- Use **Progressive** for: web delivery, storage optimization

### Decoding Speed

The default decode path uses fast integer IDCT (matching zune-jpeg performance).
The f32 pipeline is used for XYB images or when `dequant_bias(true)` is enabled.

| Mode | 2048x2048 | vs zune-jpeg | Notes |
|------|-----------|--------------|-------|
| Scanline 4:2:0 | 4.03ms | **0.99x** | Matches zune-jpeg |
| Scanline 4:4:4 | 5.78ms | **0.91x** | Beats zune-jpeg |
| Buffered fast | 4.72ms | 1.15x | Two-pass overhead |
| Buffered default | 5.51ms | 1.35x | f32 upsampling |

#### Dequantization Bias

`Decoder::new().dequant_bias(true)` enables optimal Laplacian dequantization
biases ([Price & Rabbani 2000](https://doi.org/10.1109/DCC.2000.838190)). This
computes per-coefficient biases from DCT coefficient statistics and applies them
during f32 dequantization, matching C++ jpegli's decoder behavior.

**Tradeoff:** Bypasses the fast integer IDCT path. The quality difference vs the
default integer IDCT is image-dependent and small in either direction:

| Quality | Default SSIM2 | +bias SSIM2 | C++ jpegli | bias vs default |
|---------|---------------|-------------|------------|-----------------|
| Q50 | 37.28 | 35.95 | 36.01 | -1.32 pts |
| Q85 | 50.45 | 50.18 | 50.21 | -0.27 pts |
| Q95 | 53.28 | 53.25 | 53.27 | -0.03 pts |

*(frymire 1118x1105, SSIMULACRA2 vs original, higher = better)*

The bias path consistently tracks C++ jpegli output within 0.02-0.11 SSIMULACRA2
points. Use it when you need decode output to match C++ jpegli, or when processing
pipelines assume jpegli-style reconstruction.

## Table Optimization

The `EncodingTables` API provides fine-grained control over quantization and zero-bias
tables for researching better encoding parameters.

### Quick Start

```rust
use zenjpeg::encoder::{EncoderConfig, ChromaSubsampling};
use zenjpeg::encoder::tuning::{EncodingTables, ScalingParams, dct};

// Start from defaults and modify
let mut tables = EncodingTables::default_ycbcr();

// Scale a specific coefficient (component 0 = Y, k = coefficient index)
tables.scale_quant(0, 5, 1.2);  // 20% higher quantization at position 5

// Or use exact quantization values (no quality scaling)
tables.scaling = ScalingParams::Exact;
tables.quant.c0[0] = 16.0;  // DC quantization for Y

let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter)
    .tables(Box::new(tables));
```

### Understanding the Parameters

**Quantization Tables** (`quant`): 64 coefficients per component (Y/Cb/Cr or X/Y/B)
- Lower values = more precision = larger file
- Higher values = more compression = smaller file
- DC (index 0) affects brightness uniformity
- Low frequencies (indices 1, 8, 9, 16, 17) affect gradients
- High frequencies affect edges and texture

**Zero-Bias Tables** (`zero_bias_mul`, `zero_bias_offset_*`):
- Control rounding behavior during quantization
- `zero_bias_mul[k]` multiplies the dead zone around zero
- Higher values = more aggressive zeroing of small coefficients = smaller files
- `zero_bias_offset_dc/ac` add to the threshold before zeroing

**Scaling Params**:
- `ScalingParams::Scaled { global_scale, frequency_exponents }` - quality-dependent scaling
- `ScalingParams::Exact` - use raw values (must be valid u16 range)

### DCT Coefficient Layout

```
Position in 8x8 block (row-major index k):
 0  1  2  3  4  5  6  7
 8  9 10 11 12 13 14 15
16 17 18 19 20 21 22 23
24 25 26 27 28 29 30 31
32 33 34 35 36 37 38 39
40 41 42 43 44 45 46 47
48 49 50 51 52 53 54 55
56 57 58 59 60 61 62 63

k=0 is DC (average brightness)
k=1,8 are lowest AC frequencies (horizontal/vertical gradients)
k=63 is highest frequency (diagonal detail)
```

Use `dct::freq_distance(k)` to get Manhattan distance from DC (0-14).
Use `dct::IMPORTANCE_ORDER` for coefficients sorted by perceptual impact.

### Research Methodology

#### 1. Corpus-Based Optimization

```rust
use zenjpeg::encoder::tuning::{EncodingTables, dct};

fn evaluate_tables(tables: &EncodingTables, corpus: &[Image]) -> f64 {
    let mut total_score = 0.0;
    for image in corpus {
        let jpeg = encode_with_tables(image, tables);
        let score = ssimulacra2_per_byte(&jpeg, image);  // quality/size
        total_score += score;
    }
    total_score / corpus.len() as f64
}

// Grid search over coefficient k
fn optimize_coefficient(k: usize, component: usize, corpus: &[Image]) {
    let mut best_score = f64::MIN;
    let mut best_value = 1.0;

    for scale in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0] {
        let mut tables = EncodingTables::default_ycbcr();
        tables.scale_quant(component, k, scale);

        let score = evaluate_tables(&tables, corpus);
        if score > best_score {
            best_score = score;
            best_value = scale;
        }
    }
    println!("Coefficient {} best scale: {}", k, best_value);
}
```

#### 2. Gradient-Free Optimization

For automated discovery, use derivative-free optimizers:

```rust
// Using argmin crate with Nelder-Mead
use argmin::solver::neldermead::NelderMead;

fn objective(params: &[f64], corpus: &[Image]) -> f64 {
    let mut tables = EncodingTables::default_ycbcr();

    // Map params to table modifications (e.g., first 10 most impactful coefficients)
    for (i, &scale) in params.iter().enumerate() {
        let k = dct::IMPORTANCE_ORDER[i + 1]; // Skip DC
        tables.scale_quant(0, k, scale as f32); // Y component
    }

    -evaluate_tables(&tables, corpus) // Negative because we minimize
}
```

**Recommended optimizers:**
- **CMA-ES** (Covariance Matrix Adaptation): Best for 10-50 parameters
- **Nelder-Mead**: Good for quick exploration, 5-20 parameters
- **Differential Evolution**: Robust, handles constraints well
- **Bayesian Optimization**: Sample-efficient when evaluations are expensive

#### 3. Image-Adaptive Tables

Different image categories may benefit from different tables:

| Content Type | Strategy |
|--------------|----------|
| Photographs | Lower DC/low-freq quant, preserve gradients |
| Graphics/UI | Higher high-freq quant, preserve edges |
| Text on photos | Balance - preserve both |
| Skin tones | Lower Cb/Cr quant in mid frequencies |

```rust
fn classify_and_encode(image: &Image) -> Vec<u8> {
    let tables = match classify_content(image) {
        ContentType::Photo => tables_optimized_for_photos(),
        ContentType::Graphic => tables_optimized_for_graphics(),
        ContentType::Mixed => EncodingTables::default_ycbcr(),
    };
    encode_with_tables(image, &tables)
}
```

#### 4. Perceptual Weighting

Use quality metrics to weight optimization:

```rust
// SSIMULACRA2 weights certain frequencies more than others
// Butteraugli penalizes different artifacts

fn multi_metric_score(jpeg: &[u8], original: &Image) -> f64 {
    let ssim2 = ssimulacra2(jpeg, original);
    let butteraugli = butteraugli_distance(jpeg, original);
    let size = jpeg.len() as f64;

    // Combine: higher quality, lower butteraugli, smaller size
    (ssim2 * 100.0 - butteraugli * 10.0) / (size / 1000.0)
}
```

### Ideas for Research

1. **Content-aware table selection**: Train a classifier to select optimal tables
2. **Quality-dependent tables**: Different tables for Q50 vs Q90
3. **Resolution-dependent**: High-res images may need different high-freq handling
4. **Per-block adaptive**: Use AQ to modulate per-block quantization
5. **Machine learning**: Use differentiable JPEG approximations to train tables
6. **Genetic algorithms**: Evolve table populations over a corpus
7. **Transfer learning**: Start from optimized tables for similar content

### Available Helpers

```rust
use zenjpeg::encoder::tuning::dct;

// Coefficient analysis
dct::freq_distance(k)       // Manhattan distance from DC (0-14)
dct::row_col(k)             // (row, col) in 8x8 block
dct::to_zigzag(k)           // Row-major to zigzag order
dct::from_zigzag(z)         // Zigzag to row-major
dct::IMPORTANCE_ORDER       // Coefficients by perceptual impact

// Table manipulation
tables.scale_quant(c, k, factor)    // Scale one coefficient
tables.perturb_quant(c, k, delta)   // Add delta to coefficient
tables.blend(&other, t)              // Linear interpolation (0.0-1.0)
tables.quant.scale_component(c, f)   // Scale entire component
tables.quant.scale_all(f)            // Scale all coefficients
```

## Lossless Transforms

Requires `features = ["decoder"]`. Rotate, flip, and transpose JPEGs by manipulating DCT
coefficients directly — no decode to pixels, no re-encode, zero generation loss.

### End-to-End Transform

```rust
use zenjpeg::lossless::{transform, apply_exif_orientation, LosslessTransform, TransformConfig};

// Rotate 90° losslessly
let rotated = transform(&jpeg_data, &TransformConfig {
    transform: LosslessTransform::Rotate90,
    ..Default::default()
}, enough::Unstoppable)?;

// Auto-correct EXIF orientation (resets tag to 1)
let oriented = apply_exif_orientation(&jpeg_data, enough::Unstoppable)?;
```

All metadata (EXIF, ICC, XMP, IPTC) is preserved in the output. Huffman tables are
re-optimized from coefficient frequencies.

### Decode-Time Transform

Apply a transform during decode in DCT-coefficient space, before IDCT. One entropy
decode pass, no re-encoding step.

```rust
use zenjpeg::decoder::Decoder;
use zenjpeg::lossless::LosslessTransform;

// Auto-orient from EXIF (most common use case)
let result = Decoder::new()
    .auto_orient(true)
    .decode(&jpeg_data, enough::Unstoppable)?;

// Explicit transform
let result = Decoder::new()
    .transform(LosslessTransform::Rotate90)
    .decode(&jpeg_data, enough::Unstoppable)?;
```

Works with both buffered `.decode()` and streaming `.scanline_reader()`.

### Available Transforms

All 8 elements of the D4 dihedral group: `None`, `FlipHorizontal`, `FlipVertical`,
`Transpose`, `Rotate90`, `Rotate180`, `Rotate270`, `Transverse`.

Compose with `t1.then(t2)`, invert with `t.inverse()`, map from EXIF with
`LosslessTransform::from_exif_orientation(1..=8)`.

### Edge Handling

For images whose dimensions aren't multiples of the MCU size, the end-to-end pipeline
offers two options via `EdgeHandling`:

- **`TrimPartialBlocks`** (default) — discard partial MCU strips (up to 15px per edge)
- **`RejectPartialBlocks`** — return an error

The decode-time path handles this automatically by rendering the full padded image
and cropping to the visible region.

See [LOSSLESS_TRANSFORMS.md](../LOSSLESS_TRANSFORMS.md) for technical details.

## Overshoot Deringing

**Enabled by default.** This technique was pioneered by [@kornel](https://github.com/kornelski)
in [mozjpeg](https://github.com/mozilla/mozjpeg) and significantly improves quality for
documents, screenshots, and graphics without any quality penalty for photographic content.

### The Problem

JPEG uses DCT (Discrete Cosine Transform) which represents pixel blocks as sums of cosine
waves. Hard edges—like text on a white background—create high-frequency components that
are difficult to represent accurately. The result is "ringing": oscillating artifacts that
look like halos or waves emanating from sharp transitions.

### The Insight

JPEG decoders clamp output values to 0-255. This means to display white (255), any encoded
value ≥255 works identically after clamping. The encoder can exploit this "headroom" above
the displayable range.

### The Solution

Instead of encoding a flat plateau at the maximum value, deringing creates a smooth curve
that "overshoots" above the maximum:
- The peak (above 255) gets clamped to 255 on decode
- The result looks identical to the original
- But the smooth curve compresses much better with fewer artifacts!

This is analogous to "anti-clipping" in audio processing.

### When It Helps Most

- Documents and screenshots with white backgrounds
- Text and graphics with hard edges
- Any image with saturated regions (pixels at 0 or 255)
- UI elements with sharp corners

### Usage

Deringing is **on by default**. To disable it (not recommended):

```rust
let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
    .deringing(false);  // Disable deringing
```

## C++ Parity Status

Tested against C++ jpegli on frymire.png (1118x1105):

| Metric | Rust | C++ | Difference |
|--------|------|-----|------------|
| File size (Q85 seq) | 586.3 KB | 586.7 KB | **-0.1%** |
| File size (Q85 prog) | 568.2 KB | 565.1 KB | **+0.5%** |
| SSIM2 (Q85) | 69.0 | 69.0 | **identical** |

Quality is identical (mean <0.5% difference); file sizes within 2%.

### Comparing with C++ jpegli: 2 vs 3 Quantization Tables

When comparing output between zenjpeg and C++ jpegli, **use `jpegli_set_distance()`
in C++**, not `jpeg_set_quality()`. Here's why:

**The issue:**
- `jpeg_set_quality()` in C++ uses **2 chroma tables** (Cb and Cr share the same table)
- `jpegli_set_distance()` in C++ uses **3 tables** (separate Y, Cb, Cr tables)
- zenjpeg **always uses 3 tables**

Using `jpeg_set_quality()` for comparison will show ~4% file size differences and
different quantization behavior because the encoders are configured differently.

**Correct comparison (FFI):**
```c
// C++ - use distance-based quality (3 tables)
jpegli_set_distance(&cinfo, 1.0, JPEGLI_TRUE);  // distance 1.0 ≈ quality 90

// NOT: jpeg_set_quality(&cinfo, 90, TRUE);  // 2 tables - invalid comparison!
```

**Quality to distance conversion:**
```rust
fn quality_to_distance(q: f32) -> f32 {
    if q >= 100.0 { 0.01 }
    else if q >= 30.0 { 0.1 + (100.0 - q) * 0.09 }
    else { 53.0 / 3000.0 * q * q - 23.0 / 20.0 * q + 25.0 }
}
// q90 → distance 1.0, q75 → distance 2.35
```

With proper distance-based comparison, size and quality differences are typically
within ±2%.

**Matching jpeg_set_quality() behavior:**

If you need output that matches tools using `jpeg_set_quality()` (2 tables),
use the `.separate_chroma_tables(false)` option:

```rust
// Match jpeg_set_quality() behavior (2 tables: Y, shared chroma)
let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
    .separate_chroma_tables(false);
```

## Feature Flags

| Feature | Default | Description | When to Use |
|---------|---------|-------------|-------------|
| `decoder` | ❌ No | **JPEG decoding** - Enables `zenjpeg::decoder` module | **Required** for any decode operations |
| `std` | ✅ Yes | Standard library support | Disable for `no_std` embedded targets |
| `archmage-simd` | N/A | Legacy no-op (archmage is now mandatory) | No effect; kept for backwards compatibility |
| `cms-lcms2` | ✅ Yes | ICC color management via lcms2 | XYB decoding, wide-gamut images |
| `cms-moxcms` | ❌ No | Pure Rust color management | `no_std` or avoid C dependencies |
| `parallel` | ❌ No | Multi-threaded encoding via rayon | Large images (4K+), server workloads |
| `ultrahdr` | ❌ No | UltraHDR HDR gain map support | Encoding/decoding HDR JPEGs |
| `trellis` | ✅ Yes | Trellis quantization (mozjpeg-style) | Keep enabled for best compression |
| `yuv` | ✅ Yes | SharpYUV chroma downsampling | Keep enabled for quality |

By default, the crate uses `#![forbid(unsafe_code)]`. Portable SIMD is provided via the `wide` crate, with token-based archmage intrinsics automatically enabled on x86_64/aarch64 for ~10-20% speedup. The `archmage-simd` feature flag is a legacy no-op kept for backwards compatibility — archmage is now a mandatory dependency.

### Common Configurations

```toml
# Decode + encode (most common)
[dependencies]
zenjpeg = { version = "0.6", features = ["decoder"] }

# Encode only (default)
[dependencies]
zenjpeg = "0.6"

# High-performance server
[dependencies]
zenjpeg = { version = "0.6", features = ["decoder", "parallel"] }

# Embedded / no_std
[dependencies]
zenjpeg = { version = "0.6", default-features = false, features = ["cms-moxcms"] }

# UltraHDR support
[dependencies]
zenjpeg = { version = "0.6", features = ["decoder", "ultrahdr"] }
```

## Encoder Status

| Feature | Status |
|---------|--------|
| Baseline JPEG | Working |
| Progressive JPEG | Working |
| Adaptive quantization | Working |
| Huffman optimization | Working |
| 4:4:4 / 4:2:0 / 4:2:2 / 4:4:0 | Working |
| XYB color space | Working |
| Grayscale | Working |
| Custom quant tables | Working |
| ICC profile embedding | Working |
| YCbCr planar input | Working |

## Decoder Status

> **Prerelease:** Enable with `features = ["decoder"]`. API will have breaking changes.

| Feature | Status |
|---------|--------|
| Baseline JPEG | Working |
| Progressive JPEG | Working |
| All subsampling modes | Working |
| Restart markers | Working |
| ICC profile extraction | Working |
| XYB decoding | Working (with CMS) |
| f32 output | Working |
| Lossless transforms | Working |
| EXIF auto-orient | Working |

## Future Optimization Opportunities

Profiling against C++ jpegli reveals these bottlenecks (2K image, progressive 4:2:0):

| Area | Rust | C++ | Gap | Notes |
|------|------|-----|-----|-------|
| **RGB→YCbCr** | 11.7% | 1.7% | **6.9x** | Biggest opportunity |
| **Adaptive quantization** | 28.6% | 12.1% | **2.4x** | Algorithm efficiency |
| **Huffman freq counting** | 5.7% | 0.5% | **11x** | Already SIMD, still slow |
| DCT | 7.3% | 5.5% | 1.3x | Reasonable |
| Entropy encoding | 10.9% | 35.9% | — | C++ slower here |

**Crates to investigate for RGB→YCbCr:**
- [`yuv`](https://lib.rs/crates/yuv) (0.8.9) - Faster than libyuv, AVX-512/AVX2/SSE/NEON
- [`yuvutils-rs`](https://lib.rs/crates/yuvutils-rs) - AVX2/SSE/NEON, optional AVX-512
- [`dcv-color-primitives`](https://lib.rs/crates/dcv-color-primitives) - AWS, AVX2/NEON

Current gap: Rust is **~20% slower** than C++ jpegli (1.2x median, range 1.05x-1.43x per criterion benchmarks).

## Development

### Verify C++ Parity

```bash
# Quick parity test (no C++ build needed)
cargo test --release --test cpp_parity_locked

# Full comparison (requires C++ jpegli built)
cargo test --release --test comprehensive_cpp_comparison -- --nocapture --ignored
```

### Building C++ Reference (Optional)

```bash
git submodule update --init --recursive
cd internal/jpegli-cpp && mkdir -p build && cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DJPEGXL_ENABLE_TOOLS=ON ..
ninja cjpegli djpegli
```

## License

Sustainable, large-scale open source work requires a funding model, and I have been
doing this full-time for 15 years. If you are using this for closed-source development
AND make over $1 million per year, you'll need to buy a commercial license at
https://www.imazen.io/pricing

Commercial licenses are similar to the Apache 2 license but company-specific, and on
a sliding scale. You can also use this under the AGPL v3.

## Acknowledgments

Originally a port of [jpegli](https://github.com/libjxl/libjxl/tree/main/lib/jpegli)
from the JPEG XL project by Google (BSD-3-Clause). After six rewrites, this is now
an independent project that shares ideas but little code with the original.

## AI Disclosure

Developed with assistance from Claude (Anthropic). Extensively tested against
C++ reference with 340+ tests. Report issues at https://github.com/imazen/zenjpeg/issues
