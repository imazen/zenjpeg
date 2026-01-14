# jpegli-rs

[![Crates.io](https://img.shields.io/crates/v/jpegli-rs.svg)](https://crates.io/crates/jpegli-rs)
[![Documentation](https://docs.rs/jpegli-rs/badge.svg)](https://docs.rs/jpegli-rs)
[![CI](https://github.com/imazen/jpegli-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/imazen/jpegli-rs/actions/workflows/ci.yml)
[![License: AGPL-3.0-or-later](https://img.shields.io/crates/l/jpegli-rs.svg)](LICENSE)

A pure Rust JPEG encoder and decoder with perceptual optimizations.

## Heritage and Divergence

This project was originally inspired by [jpegli](https://github.com/libjxl/libjxl/tree/main/lib/jpegli), Google's improved JPEG encoder from the JPEG XL project. The initial implementation aimed for bit-exact parity with the C++ reference.

However, jpegli-rs has been **rewritten from scratch multiple times** and is now **diverging significantly** from the original. While it retains the core concepts (adaptive quantization, XYB color space support, smart zero-biasing), the implementation details, API design, and optimizations are increasingly our own.

**What we kept from jpegli:**
- Adaptive quantization philosophy (content-aware bit allocation)
- XYB color space support with ICC profiles
- Perceptually-tuned quantization tables
- Zero-bias strategies for coefficient rounding

**Where we're diverging:**
- Pure Rust implementation with no C/C++ dependencies
- New streaming encoder API optimized for memory efficiency
- Different SIMD strategy (portable `wide` crate vs platform intrinsics)
- Parallel encoding support
- Ongoing encoder optimizations independent of upstream

## Features

- **Pure Rust** - No C/C++ dependencies, builds anywhere Rust does
- **Perceptual optimization** - Adaptive quantization for better visual quality at smaller sizes
- **Backward compatible** - Produces standard JPEG files readable by any decoder
- **SIMD accelerated** - Portable SIMD via `wide` crate
- **Streaming API** - Memory-efficient row-by-row encoding
- **Parallel encoding** - Multi-threaded for large images (1024x1024+)
- **Color management** - Optional ICC profile support

## API Reference

### Encoder API

All encoder types are in `jpegli::encoder`:

```rust
use jpegli::encoder::{
    EncoderConfig, PixelLayout, Quality, ChromaSubsampling, Unstoppable
};
```

#### Quick Start

```rust
use jpegli::encoder::{EncoderConfig, PixelLayout, Unstoppable};

// Create reusable config
let config = EncoderConfig::new()
    .quality(85)
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
use jpegli::encoder::{EncoderConfig, PixelLayout, Unstoppable};

let config = EncoderConfig::new().quality(85);

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

#### EncoderConfig Builder Methods

| Method | Description | Default |
|--------|-------------|---------|
| `.quality(q)` | Quality 0-100 or `Quality` enum | 90 |
| `.progressive(bool)` | Progressive JPEG (~3% smaller) | `false` |
| `.optimize_huffman(bool)` | Optimal Huffman tables | `true` |
| `.ycbcr(sub)` | YCbCr with subsampling | `Quarter` (4:2:0) |
| `.xyb()` | XYB perceptual color space | - |
| `.grayscale()` | Single-channel output | - |
| `.sharp_yuv(bool)` | SharpYUV downsampling | `false` |
| `.icc_profile(bytes)` | Attach ICC profile | None |
| `.restart_interval(n)` | MCUs between restart markers | 0 |

#### Quality Options

```rust
use jpegli::encoder::{EncoderConfig, Quality};

// Simple quality scale (0-100)
let config = EncoderConfig::new().quality(85);

// Quality enum variants
let config = EncoderConfig::new()
    .quality(Quality::ApproxJpegli(85.0))     // Default scale
    .quality(Quality::ApproxMozjpeg(80))      // Match mozjpeg output
    .quality(Quality::ApproxSsim2(90.0))      // Target SSIMULACRA2 score
    .quality(Quality::ApproxButteraugli(1.0)); // Target butteraugli distance
```

#### Pixel Layouts

| Layout | Bytes/px | Notes |
|--------|----------|-------|
| `Rgb8Srgb` | 3 | Default, sRGB gamma |
| `Bgr8Srgb` / `Bgrx8Srgb` | 3/4 | Windows/GDI order |
| `Rgbx8Srgb` | 4 | 4th byte ignored |
| `Gray8Srgb` | 1 | Grayscale sRGB |
| `Rgb16Linear` | 6 | 16-bit linear |
| `RgbF32Linear` | 12 | HDR float (0.0-1.0) |
| `YCbCr8` / `YCbCrF32` | 3/12 | Pre-converted YCbCr |

#### Chroma Subsampling

```rust
use jpegli::encoder::{EncoderConfig, ChromaSubsampling};

let config = EncoderConfig::new()
    .ycbcr(ChromaSubsampling::Quarter)        // 4:2:0 (default, best compression)
    .ycbcr(ChromaSubsampling::Full)           // 4:4:4 (best quality)
    .ycbcr(ChromaSubsampling::HalfHorizontal) // 4:2:2
    .ycbcr(ChromaSubsampling::HalfVertical);  // 4:4:0
```

#### Resource Estimation

```rust
use jpegli::encoder::EncoderConfig;

let config = EncoderConfig::new().quality(85);

// Typical memory estimate
let estimate = config.estimate_memory(1920, 1080);

// Guaranteed upper bound (for resource reservation)
let ceiling = config.estimate_memory_ceiling(1920, 1080);
```

---

### Decoder API

> **Prerelease:** The decoder API is behind the `decoder` feature flag and will have breaking changes.
> Enable with `jpegli-rs = { version = "...", features = ["decoder"] }`.

All decoder types are in `jpegli::decoder`:

```rust
use jpegli::decoder::{Decoder, DecodedImage, DecodedImageF32, DecoderConfig};
```

#### Basic Decoding

```rust
// Decode to RGB (default)
let image = Decoder::new().decode(&jpeg_data)?;
let pixels: &[u8] = image.pixels();
let (width, height) = image.dimensions();
```

#### High-Precision Decoding (f32)

Preserves jpegli's 12-bit internal precision:

```rust
let image: DecodedImageF32 = Decoder::new().decode_f32(&jpeg_data)?;
let pixels: &[f32] = image.pixels();  // Values in 0.0-1.0

// Convert to 8-bit or 16-bit when needed
let u8_pixels: Vec<u8> = image.to_u8();
let u16_pixels: Vec<u16> = image.to_u16();
```

#### YCbCr Output (Zero Color Conversion)

For video pipelines or re-encoding:

```rust
use jpegli::decoder::{Decoder, DecodedYCbCr};

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
| `.max_pixels(n)` | Pixel count limit (DoS protection) | 100M |
| `.max_memory(n)` | Memory limit in bytes | 512 MB |

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
use jpegli::decoder::{Decoder, DecoderConfig};

// Most users should use the builder methods instead:
let image = Decoder::new()
    .fancy_upsampling(true)
    .block_smoothing(false)
    .apply_icc(true)
    .max_pixels(100_000_000)
    .max_memory(512 * 1024 * 1024)
    .decode(&jpeg_data)?;

// Or construct DecoderConfig directly:
let config = DecoderConfig::default();
let decoder = Decoder::from_config(config);
```

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

| Decoder | Speed | Notes |
|---------|-------|-------|
| zune-jpeg | 392 MP/s | Integer IDCT, AVX2 |
| jpeg-decoder | 120 MP/s | Integer IDCT |
| **jpegli-rs** | **47 MP/s** | f32 IDCT, 12-bit precision |

The decoder prioritizes precision over speed, matching C++ jpegli's 12-bit pipeline.

## C++ Parity Status

Tested against C++ jpegli on frymire.png (1118x1105):

| Metric | Rust | C++ | Difference |
|--------|------|-----|------------|
| File size (Q85 seq) | 586.3 KB | 586.7 KB | **-0.1%** |
| File size (Q85 prog) | 568.2 KB | 565.1 KB | **+0.5%** |
| SSIM2 (Q85) | 69.0 | 69.0 | **identical** |

Quality is identical; file sizes within 0.5%.

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `decoder` | No | Enable decoder API (prerelease, API will change) |
| `cms-lcms2` | Yes | Color management via lcms2 |
| `cms-moxcms` | No | Pure Rust color management |
| `unsafe_simd` | No | Raw AVX2/SSE intrinsics (~10-20% faster) |
| `test-utils` | Yes | Testing utilities |

SIMD via the `wide` crate is always enabled (portable, safe).

```toml
[dependencies]
jpegli-rs = "0.5"

# Minimal (no CMS):
jpegli-rs = { version = "0.5", default-features = false }

# With unsafe SIMD (x86_64 only):
jpegli-rs = { version = "0.5", features = ["unsafe_simd"] }
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

**AGPL-3.0-or-later**

A commercial license is available from https://imageresizing.net/pricing

## Acknowledgments

Originally inspired by [jpegli](https://github.com/libjxl/libjxl/tree/main/lib/jpegli)
from the JPEG XL project by Google (BSD-3-Clause). This Rust implementation has
been rewritten multiple times and is now an independent project with its own
development trajectory.

## AI Disclosure

Developed with assistance from Claude (Anthropic). Extensively tested against
C++ reference with 340+ tests. Report issues at https://github.com/imazen/jpegli-rs/issues
