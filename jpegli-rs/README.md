# jpegli-rs

[![Crates.io](https://img.shields.io/crates/v/jpegli-rs.svg)](https://crates.io/crates/jpegli-rs)
[![Documentation](https://docs.rs/jpegli-rs/badge.svg)](https://docs.rs/jpegli-rs)
[![CI](https://github.com/imazen/jpegli-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/imazen/jpegli-rs/actions/workflows/ci.yml)
[![License: AGPL-3.0-or-later](https://img.shields.io/crates/l/jpegli-rs.svg)](LICENSE)

Pure Rust implementation of **jpegli** - Google's improved JPEG encoder/decoder from the JPEG XL project.

## Features

- **Pure Rust** - No C/C++ dependencies required
- **Perceptual optimization** - Uses adaptive quantization for better visual quality
- **Backward compatible** - Produces standard JPEG files readable by any decoder
- **SIMD accelerated** - Uses `wide` crate for portable SIMD
- **Color management** - Optional ICC profile support via lcms2 or moxcms

## What is jpegli?

jpegli is Google's improved JPEG encoder that produces smaller files at the same visual quality,
or better quality at the same file size. It achieves this through:

- **Adaptive quantization** - Content-aware bit allocation
- **Improved quantization tables** - Better than standard IJG libjpeg tables
- **XYB color space** (optional) - Perceptually optimized color representation
- **Smart zero-biasing** - Intelligent coefficient rounding

## Usage

```rust
use jpegli::{Encoder, Quality, PixelFormat};

// Encode RGB image data to JPEG
let jpeg_data = Encoder::new()
    .width(800)
    .height(600)
    .pixel_format(PixelFormat::Rgb)
    .quality(Quality::default())  // Q90
    .encode(&rgb_pixels)?;

// Decode JPEG to RGB
let decoded = jpegli::Decoder::new().decode(&jpeg_data)?;
println!("{}x{}", decoded.width, decoded.height);
let rgb_pixels: &[u8] = &decoded.data;
```

## Feature Flags

- `simd` (default) - Enable SIMD acceleration
- `cms-lcms2` - Use lcms2 for color management (C dependency)
- `cms-moxcms` - Use moxcms for color management (pure Rust)

## Encoder Status

The encoder is feature-complete and production-ready:

| Feature | Status |
|---------|--------|
| Baseline JPEG | ✅ Working |
| Progressive JPEG (level 0) | ✅ Working |
| Adaptive quantization | ✅ Matches C++ jpegli |
| Huffman optimization | ✅ Working |
| 4:4:4 / 4:2:0 / 4:2:2 subsampling | ✅ Working |
| XYB color space | ✅ Working (with ICC) |
| Grayscale | ✅ Working |

**Encoder Performance**: ~1-3 MP/s (encoding is compute-intensive due to perceptual optimizations)

## Decoder Status

The decoder is functional with 12-bit internal precision (matching C++ jpegli):

| Feature | Status |
|---------|--------|
| Baseline JPEG | ✅ Working |
| Progressive JPEG | ✅ Working |
| All subsampling modes | ✅ Working |
| ICC profile extraction | ✅ Working |
| XYB decoding (with CMS) | ✅ Working |
| f32 output format | ✅ Working |

**Decoder Performance** (1024x768 image):

| Decoder | Speed | Notes |
|---------|-------|-------|
| zune-jpeg | 392 MP/s | Integer IDCT, AVX2/NEON |
| jpeg-decoder | 120 MP/s | Integer IDCT |
| **jpegli-rs** | **47 MP/s** | f32 IDCT (12-bit precision) |

The decoder is slower than alternatives because it uses a float pipeline for 12-bit precision,
matching C++ jpegli's design. See [Future Goals](#future-goals) for planned optimizations.

## Quality Comparison

Tested on Kodak dataset at various quality levels:

| Quality | jpegli-rs DSSIM | mozjpeg DSSIM | jpegli-rs SSIMULACRA2 | mozjpeg SSIMULACRA2 |
|---------|-----------------|---------------|----------------------|---------------------|
| Q50 | 0.0050 | 0.0060 | 62.5 | 58.0 |
| Q70 | 0.0026 | 0.0034 | 72.4 | 67.8 |
| Q90 | 0.0008 | 0.0011 | 85.0 | 82.0 |

Lower DSSIM is better. Higher SSIMULACRA2 is better.
jpegli-rs achieves **10-17% better quality** at the same file size.

## Development

### Running FFI Comparison Tests

To verify the Rust implementation matches the C++ original:

```bash
# Linux/macOS
./internal/setup-ffi-tests.sh

# Windows
.\internal\setup-ffi-tests.ps1
```

This requires CMake, a C++ compiler, and ~10 minutes for the initial C++ build.
See [internal/README.md](../internal/README.md) for details.

### Running Benchmarks

```bash
# Decoder performance comparison
cargo run --release --example decode_benchmark

# Encoder quality comparison
cargo run --release --example pareto_comparison
```

## Future Goals

### Decoder Optimization (Target: 100+ MP/s)

The current decoder uses f32 arithmetic for 12-bit precision. To reach competitive speeds:

- [ ] Optional integer IDCT path for u8 output
- [ ] Platform-specific SIMD (AVX2, NEON) for hot paths
- [ ] Optimized bit reader with bulk byte loading
- [ ] Multi-threaded decoding for large images

### Encoder Improvements

- [ ] Progressive JPEG level 2 (successive approximation)
- [ ] Parallel block processing
- [ ] Memory-efficient streaming API

## License

**AGPL-3.0-or-later**

A commercial license is available from https://imageresizing.net/pricing

The original jpegli from libjxl is BSD-3-Clause licensed.
This Rust implementation is an independent port, not a derivative work.

## Acknowledgments

This is a Rust port of [jpegli](https://github.com/libjxl/libjxl/tree/main/lib/jpegli)
from the JPEG XL project by Google.

## AI-Generated Code Notice

This crate was developed with significant assistance from Claude (Anthropic).
While extensively tested against the C++ reference implementation with 170+ tests,
not all code paths have been manually reviewed.

Before production use in critical applications:
- Review code paths relevant to your use case
- Run your own validation tests
- Report any issues at https://github.com/imazen/jpegli-rs/issues
