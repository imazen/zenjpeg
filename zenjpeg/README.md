# zenjpeg

[![Crates.io](https://img.shields.io/crates/v/zenjpeg.svg)](https://crates.io/crates/zenjpeg)
[![Documentation](https://docs.rs/zenjpeg/badge.svg)](https://docs.rs/zenjpeg)
[![CI](https://github.com/imazen/zenjpeg/actions/workflows/ci.yml/badge.svg)](https://github.com/imazen/zenjpeg/actions/workflows/ci.yml)
[![License: AGPL-3.0-or-later](https://img.shields.io/crates/l/zenjpeg.svg)](LICENSE)

A pure Rust JPEG encoder and decoder with perceptual optimizations.

> **Note:** This crate was previously published as `zenjpeg`. If migrating, update your imports from `use jpegli::` to `use zenjpeg::`.

## Heritage and Divergence

This project started as a port of [jpegli](https://github.com/libjxl/libjxl/tree/main/lib/jpegli), Google's improved JPEG encoder from the JPEG XL project. After six rewrites it has diverged significantly into an independent project.

**Ideas adopted from jpegli:**
- Adaptive quantization (content-aware bit allocation)
- XYB color space with ICC profiles (note: XYB support is currently poor, ~5 SSIMULACRA2 behind C++)
- Perceptually-tuned quantization tables
- Zero-bias strategies for coefficient rounding

**Ideas adopted from mozjpeg:**
- Overshoot deringing for documents/graphics
- Trellis quantization for optimal coefficient selection
- Hybrid approach combining jpegli's AQ with mozjpeg's trellis

**Where we went our own way:**
- Pure Rust, `#![forbid(unsafe_code)]` by default (unsafe SIMD is opt-in)
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
- **UltraHDR support** - Encode/decode HDR gain maps (optional `ultrahdr` feature)
- **Color management** - Optional ICC profile support

## Known Limitations

- **XYB color space** - Currently ~5 SSIMULACRA2 points behind C++ jpegli. Use YCbCr for best quality.
- **Decoder speed** - Prioritizes precision (12-bit pipeline) over speed; ~8x slower than zune-jpeg.

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

Requires `features = ["decoder"]` (prerelease API).

```rust
use zenjpeg::decoder::Decoder;

let image = Decoder::new().decode(&jpeg_bytes)?;
let rgb_pixels: &[u8] = image.pixels();
let (width, height) = image.dimensions();
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
| `.progressive(bool)` | Progressive JPEG (~3% smaller) | `false` |
| `.optimize_huffman(bool)` | Optimal Huffman tables | `true` |
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
| `Bgr8Srgb` / `Bgrx8Srgb` | 3/4 | Windows/GDI order |
| `Rgbx8Srgb` | 4 | 4th byte ignored |
| `Gray8Srgb` | 1 | Grayscale sRGB |
| `Rgb16Linear` | 6 | 16-bit linear |
| `RgbF32Linear` | 12 | HDR float (0.0-1.0) |
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
use zenjpeg::decoder::{Decoder, DecodedImage, DecodedImageF32, DecoderConfig};
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
use zenjpeg::decoder::{Decoder, DecoderConfig};

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
| **zenjpeg** | **47 MP/s** | f32 IDCT, 12-bit precision |

The decoder prioritizes precision over speed, matching C++ jpegli's 12-bit pipeline.

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

Quality is identical; file sizes within 0.5%.

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
within ±1%.

**Matching jpeg_set_quality() behavior:**

If you need output that matches tools using `jpeg_set_quality()` (2 tables),
use the `.separate_chroma_tables(false)` option:

```rust
// Match jpeg_set_quality() behavior (2 tables: Y, shared chroma)
let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
    .separate_chroma_tables(false);
```

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `decoder` | No | Enable decoder API (prerelease, API will change) |
| `ultrahdr` | No | UltraHDR HDR gain map encoding/decoding (requires `decoder`) |
| `cms-lcms2` | Yes | Color management via lcms2 |
| `cms-moxcms` | No | Pure Rust color management |
| `unsafe_simd` | No | Raw AVX2/SSE intrinsics (~10-20% faster) |
| `test-utils` | Yes | Testing utilities |

By default, the crate uses `#![forbid(unsafe_code)]`. SIMD is provided via the safe, portable `wide` crate. Enable `unsafe_simd` for raw intrinsics on x86_64.

```toml
[dependencies]
zenjpeg = "0.11"

# With UltraHDR support:
zenjpeg = { version = "0.11", features = ["ultrahdr"] }

# Minimal (no CMS):
zenjpeg = { version = "0.11", default-features = false }

# With unsafe SIMD (x86_64 only):
zenjpeg = { version = "0.11", features = ["unsafe_simd"] }
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

Current gap: Rust is **~1.6-1.9x slower** than C++ jpegli (fair FFI comparison).

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

Originally a port of [jpegli](https://github.com/libjxl/libjxl/tree/main/lib/jpegli)
from the JPEG XL project by Google (BSD-3-Clause). After six rewrites, this is now
an independent project that shares ideas but little code with the original.

## AI Disclosure

Developed with assistance from Claude (Anthropic). Extensively tested against
C++ reference with 340+ tests. Report issues at https://github.com/imazen/zenjpeg/issues
