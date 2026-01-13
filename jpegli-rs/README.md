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

## Quick Start

```rust
use jpegli::{JpegEncoder, Quality};

// Simple encoding (one-shot)
let jpeg_data = jpegli::encode_rgb(800, 600, &rgb_pixels, 85)?;

// Builder API for more control
let jpeg_data = JpegEncoder::new(800, 600)
    .quality(85)                      // 1-100 scale
    .progressive(true)                // Progressive JPEG (~3% smaller)
    .encode_all(&rgb_pixels)?;

// Decode JPEG to RGB
let image = jpegli::decode(&jpeg_data)?;
let rgb_pixels: &[u8] = image.pixels();
```

### Streaming Encoder (Memory Efficient)

For large images or when reading from a stream:

```rust
use jpegli::JpegEncoder;

let mut encoder = JpegEncoder::new(4096, 4096)
    .quality(85)
    .start()?;

// Push rows incrementally
for row in image_rows {
    encoder.push_row(row)?;
}

let jpeg_data = encoder.finish()?;
```

### Quality Settings

```rust
use jpegli::{JpegEncoder, Quality};

// Traditional 1-100 quality scale
JpegEncoder::new(w, h).quality(85)

// Butteraugli distance (advanced - lower = better quality)
// 1.0 = high quality, 2.0 = medium, 3.0+ = low
JpegEncoder::new(w, h).distance(1.0)

// Quality enum for explicit control
JpegEncoder::new(w, h).quality(Quality::from_distance(1.0))
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

### Parallel Encoding

```rust
// Enable parallel encoding (requires `parallel` feature)
JpegEncoder::new(2048, 2048)
    .quality(85)
    .parallel(true)  // 1.4x speedup on large images
    .encode_all(&pixels)?;
```

| Image Size | Parallel Speedup | Notes |
|------------|------------------|-------|
| 512x512 | 0.69x (slower!) | Overhead exceeds benefit |
| 1024x1024 | 1.11x | Marginal benefit |
| 2048x2048 | **1.40x** | Significant benefit |

**Only use parallel for images 1024x1024 or larger.**

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
| `simd` | Yes | Portable SIMD via `wide` crate |
| `parallel` | No | Multi-threaded encoding (rayon) |
| `cms-lcms2` | Yes | Color management via lcms2 |
| `cms-moxcms` | No | Pure Rust color management |
| `test-utils` | Yes | Testing utilities |

```toml
[dependencies]
jpegli-rs = "0.4"

# Or with parallel encoding:
jpegli-rs = { version = "0.4", features = ["parallel"] }

# Minimal (no CMS):
jpegli-rs = { version = "0.4", default-features = false, features = ["simd"] }
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
| Parallel encoding | Working (1024x1024+) |
| Custom quant tables | Working |

## Decoder Status

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
