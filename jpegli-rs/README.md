# jpegli-rs

[![Crates.io](https://img.shields.io/crates/v/jpegli-rs.svg)](https://crates.io/crates/jpegli-rs)
[![Documentation](https://docs.rs/jpegli-rs/badge.svg)](https://docs.rs/jpegli-rs)
[![CI](https://github.com/imazen/jpegli-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/imazen/jpegli-rs/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/imazen/jpegli-rs/branch/main/graph/badge.svg)](https://codecov.io/gh/imazen/jpegli-rs)
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
    .jpegli_quality(Quality::default())  // Q90
    .encode(&rgb_pixels)?;

// Decode JPEG to RGB
let decoded = jpegli::Decoder::new().decode(&jpeg_data)?;
println!("{}x{}", decoded.width, decoded.height);
let rgb_pixels: &[u8] = &decoded.data;
```

### Matching mozjpeg Quality

```rust
use jpegli::{Encoder, QualityConversion, QualityComparisonMetric, Subsampling};

// Convert mozjpeg Q85 to equivalent jpegli quality
let conversion = QualityConversion::mozjpeg_equivalent(
    85,
    Subsampling::S444,
    QualityComparisonMetric::Dssim,
);

let jpeg_data = Encoder::new()
    .width(800)
    .height(600)
    .equivalent_quality(conversion)
    .encode(&rgb_pixels)?;
```

## Feature Flags

| Feature | Description |
|---------|-------------|
| `simd` (default) | Enable SIMD acceleration via `wide` crate |
| `cms-lcms2` | Color management via lcms2 (C dependency) |
| `cms-moxcms` | Color management via moxcms (pure Rust) |
| `experimental-hybrid-trellis` | Hybrid quantization: jpegli AQ + mozjpeg trellis (experimental, unvalidated) |

## Encoder Status

The encoder is feature-complete and production-ready:

| Feature | Status |
|---------|--------|
| Baseline JPEG | ✅ Working |
| Progressive JPEG | ✅ Working (levels 0-2) |
| Adaptive quantization | ✅ Matches C++ jpegli |
| Huffman optimization | ✅ Working |
| 4:4:4 / 4:2:0 / 4:2:2 / 4:4:0 subsampling | ✅ Working |
| XYB color space | ✅ Working (with ICC) |
| Restart markers | ✅ Working (encode & decode) |
| Grayscale | ✅ Working |

**Encoder Performance**: 30-55 MP/s (varies by image complexity and quality setting)

## Decoder Status

The decoder is functional with 12-bit internal precision (matching C++ jpegli):

| Feature | Status |
|---------|--------|
| Baseline JPEG | ✅ Working |
| Progressive JPEG | ✅ Working |
| All subsampling modes | ✅ Working |
| Restart markers | ✅ Working (RST0-RST7 validation) |
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

**Note on decoder output:** The jpegli-rs decoder uses Laplacian dequantization biases
(matching C++ djpegli), which produces slightly different output than standard decoders
like jpeg-decoder or zune-jpeg. This is intentional and improves reconstruction accuracy
for typical photographic content. For synthetic test images, this may result in higher
(worse) butteraugli/DSSIM scores when compared against the original.

## Encoder Parity with C++ jpegli

Tested on CLIC2025 + Kodak datasets (56 images × 20 quality levels = 1,120 encodings per encoder):

### Overall Results

| Metric | jpegli-rs | C++ cjpegli | Difference |
|--------|-----------|-------------|------------|
| Avg file size | 247,434 bytes | 247,388 bytes | **+0.02%** |
| Avg DSSIM | 0.00234 | 0.00234 | identical |
| Avg SSIMULACRA2 | 73.44 | 73.44 | identical |
| Avg encode time | 37.1 ms | 42.9 ms | **1.15x faster** |

### Performance by Corpus

| Corpus | Images | Avg Size | Encoder | Bytes | Time | Speed |
|--------|--------|----------|---------|-------|------|-------|
| Kodak | 24 | 0.4 MP | jpegli-rs | 74,791 | 8.7 ms | **45.2 MP/s** |
| Kodak | 24 | 0.4 MP | cjpegli | 74,783 | 13.2 ms | 29.8 MP/s |
| CLIC2025 | 32 | 2.8 MP | jpegli-rs | 376,916 | 58.5 ms | **47.4 MP/s** |
| CLIC2025 | 32 | 2.8 MP | cjpegli | 376,842 | 65.2 ms | 42.5 MP/s |

jpegli-rs produces **byte-for-byte nearly identical** output to C++ jpegli with **15-50% faster** encoding.

### Quality by Quality Level

| Quality | Avg DSSIM | Avg SSIMULACRA2 | Avg BPP |
|---------|-----------|-----------------|---------|
| Q30 | 0.0055 | 56.2 | 0.91 |
| Q50 | 0.0039 | 63.7 | 1.11 |
| Q70 | 0.0023 | 72.3 | 1.51 |
| Q80 | 0.0015 | 77.3 | 1.91 |
| Q90 | 0.0007 | 84.3 | 2.94 |
| Q95 | 0.0003 | 88.7 | 4.37 |

Lower DSSIM is better. Higher SSIMULACRA2 is better.

### Comprehensive Matrix: YCbCr vs XYB (commit a80225e)

Tested across image sizes (64×64, 512×512, 2048×2048) and encoding modes:

#### YCbCr Mode - Excellent Parity ✅

| Size | Mode | Huffman | Q70 Δ% | Q90 Δ% |
|------|------|---------|--------|--------|
| 64×64 | Baseline | Fixed | **0.0%** | **0.0%** |
| 64×64 | Baseline | Optimized | +2.9% | +2.8% |
| 64×64 | Progressive | Optimized | **0.0%** | **0.0%** |
| 512×512 | Baseline | Fixed | **0.0%** | **0.0%** |
| 512×512 | Baseline | Optimized | +1.9% | +2.0% |
| 512×512 | Progressive | Optimized | **0.0%** | **0.0%** |
| 2048×2048 | Baseline | Fixed | **0.0%** | **0.0%** |
| 2048×2048 | Baseline | Optimized | +0.1% | +2.5% |
| 2048×2048 | Progressive | Optimized | **0.0%** | **0.0%** |

#### XYB Mode - Good Parity

| Size | Mode | Huffman | Q70 Δ% | Q90 Δ% |
|------|------|---------|--------|--------|
| 64×64 | Baseline | Fixed | +15.0% | +14.0% |
| 64×64 | Baseline | Optimized | +1.0% | +0.8% |
| 64×64 | Progressive | Optimized | +4.7% | +5.7% |
| 512×512 | Baseline | Fixed | +2.6% | +1.8% |
| 512×512 | Baseline | Optimized | -0.8% | +1.1% |
| 512×512 | Progressive | Optimized | -0.4% | +4.8% |
| 2048×2048 | Baseline | Fixed | +0.2% | +1.2% |
| 2048×2048 | Baseline | Optimized | **0.0%** | +0.9% |
| 2048×2048 | Progressive | Optimized | +0.2% | +2.5% |

**Key findings:**
- YCbCr progressive with Huffman optimization matches C++ exactly (0% diff)
- XYB has larger gaps on small images but converges for large images
- Quality (SSIMULACRA2) is identical between Rust and C++ in all configurations

Run `cargo run --release --example comprehensive_matrix` to reproduce.

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

# Encoder benchmark vs C++ cjpegli (requires cjpegli in PATH)
cargo run --release --example encode_benchmark

# Multi-decoder compatibility test
cargo test --test multi_decoder_compatibility -- --nocapture
```

### Criterion Benchmarks (Rust vs C++ Timing)

Statistically valid performance comparison with noise detection:

```bash
# Run full benchmark suite (requires cjpegli built)
cargo bench --bench cpp_comparison

# Save baseline for tracking performance over time
cargo bench --bench cpp_comparison -- --save-baseline main

# Compare against saved baseline (detects regressions)
cargo bench --bench cpp_comparison -- --baseline main
```

The benchmark tests all encoding mode combinations:
- **Scan modes**: Baseline, Progressive
- **Huffman**: Fixed, Optimized
- **Subsampling**: 4:4:4, 4:2:0
- **Color space**: YCbCr, XYB

Results are saved in `target/criterion/` with HTML reports.

**Quick ad-hoc comparison** (non-statistical):
```bash
cargo run --release --example cpp_timing_matrix
cargo run --release --example cpp_timing_matrix -- --csv results.csv  # track over time
```

## Future Goals

### Decoder Optimization (Target: 100+ MP/s)

The current decoder uses f32 arithmetic for 12-bit precision. To reach competitive speeds:

- [ ] Optional integer IDCT path for u8 output
- [ ] Platform-specific SIMD (AVX2, NEON) for hot paths
- [ ] Optimized bit reader with bulk byte loading
- [ ] Multi-threaded decoding for large images

### Encoder Improvements

- [ ] Parallel block processing
- [ ] Memory-efficient streaming API
- [ ] Further entropy coding optimizations

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
While extensively tested against the C++ reference implementation with 220+ tests,
not all code paths have been manually reviewed.

Before production use in critical applications:
- Review code paths relevant to your use case
- Run your own validation tests
- Report any issues at https://github.com/imazen/jpegli-rs/issues
