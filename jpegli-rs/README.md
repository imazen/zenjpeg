# jpegli-rs

[![Crates.io](https://img.shields.io/crates/v/jpegli-rs.svg)](https://crates.io/crates/jpegli-rs)
[![Documentation](https://docs.rs/jpegli-rs/badge.svg)](https://docs.rs/jpegli-rs)
[![CI](https://github.com/imazen/jpegli-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/imazen/jpegli-rs/actions/workflows/ci.yml)
[![License](https://img.shields.io/crates/l/jpegli-rs.svg)](LICENSE)

Pure Rust implementation of **jpegli** - Google's improved JPEG encoder/decoder from the JPEG XL project.

## Features

- **Pure Rust** - No C/C++ dependencies required
- **Perceptual optimization** - Uses adaptive quantization for better visual quality
- **Backward compatible** - Produces standard JPEG files readable by any decoder
- **SIMD accelerated** - Uses `wide` crate for portable SIMD
- **Color management** - Optional ICC profile support via lcms2

## What is jpegli?

jpegli is Google's improved JPEG encoder that produces smaller files at the same visual quality,
or better quality at the same file size. It achieves this through:

- **Adaptive quantization** - Content-aware bit allocation
- **Improved quantization tables** - Better than standard IJG libjpeg tables
- **XYB color space** (optional) - Perceptually optimized color representation
- **Smart zero-biasing** - Intelligent coefficient rounding

## Usage

```rust
use jpegli_rs::{Encoder, Quality, PixelFormat};

// Encode RGB image data to JPEG
let jpeg_data = Encoder::new()
    .width(800)
    .height(600)
    .pixel_format(PixelFormat::Rgb)
    .quality(Quality::default())  // Q90
    .encode(&rgb_pixels)?;

// Write to file
std::fs::write("output.jpg", &jpeg_data)?;
```

## Feature Flags

- `simd` (default) - Enable SIMD acceleration
- `cms` (default) - Enable color management (uses lcms2)
- `cms-lcms2` - Use lcms2 for color management
- `cms-moxcms` - Use moxcms for color management (pure Rust)

## Performance

jpegli-rs achieves **10-17% better quality** (measured by DSSIM) at the same file size
compared to mozjpeg, or produces **5-8% smaller files** at equivalent quality.

## Acknowledgments

This is a Rust port of [jpegli](https://github.com/libjxl/libjxl/tree/main/lib/jpegli)
from the JPEG XL project by Google.

## License

AGPL-3.0-or-later. A commercial license is available from https://imageresizing.net/pricing

The original jpegli from libjxl is BSD-3-Clause licensed. See LICENSE for full details.

## AI-Generated Code Notice

This crate was developed with significant assistance from Claude (Anthropic).
While extensively tested against the C++ reference implementation with 100+ parity tests,
not all code paths have been manually reviewed.

Before production use in critical applications:
- Review code paths relevant to your use case
- Run your own validation tests
- Report any issues at https://github.com/imazen/jpegli-rs/issues
