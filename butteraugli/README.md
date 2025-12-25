# butteraugli-oxide

[![Crates.io](https://img.shields.io/crates/v/butteraugli-oxide.svg)](https://crates.io/crates/butteraugli-oxide)
[![Documentation](https://docs.rs/butteraugli-oxide/badge.svg)](https://docs.rs/butteraugli-oxide)
[![License](https://img.shields.io/crates/l/butteraugli-oxide.svg)](LICENSE)

Pure Rust implementation of Google's **butteraugli** perceptual image quality metric from [libjxl](https://github.com/libjxl/libjxl).

## What is Butteraugli?

Butteraugli is a psychovisual image quality metric that estimates the perceived difference between two images. Unlike simple metrics like PSNR or MSE, butteraugli models human vision to produce scores that correlate well with subjective quality assessments.

The metric is based on:
- **Opsin dynamics**: Models photosensitive chemical responses in the retina
- **XYB color space**: A hybrid opponent/trichromatic color representation
- **Visual masking**: How image features hide or reveal differences
- **Multi-scale analysis**: Examines differences at multiple frequency bands

## Quality Thresholds

| Score | Interpretation |
|-------|----------------|
| < 1.0 | Images appear identical to most viewers |
| 1.0 - 2.0 | Subtle differences may be noticeable |
| > 2.0 | Visible differences between images |

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
butteraugli-oxide = "0.1"
```

### Input Formats

Two input APIs are provided:

| Function | Input Type | Color Space | Use Case |
|----------|------------|-------------|----------|
| `compute_butteraugli` | `&[u8]` | sRGB (gamma-encoded) | Standard 8-bit images |
| `compute_butteraugli_linear` | `&[f32]` | Linear RGB (0.0-1.0) | HDR, 16-bit, float pipelines |

Both APIs require:
- **Channel order**: RGB (red, green, blue)
- **Layout**: Row-major, interleaved (RGBRGBRGB...)
- **Minimum size**: 8×8 pixels

The sRGB function internally applies gamma decoding before comparison.

### Basic Example

```rust
use butteraugli_oxide::{compute_butteraugli, ButteraugliParams};

// Load two RGB images (u8, 3 bytes per pixel, row-major order)
let original: &[u8] = &[/* original image RGB data */];
let compressed: &[u8] = &[/* compressed image RGB data */];
let width = 640;
let height = 480;

// Compare images
let params = ButteraugliParams::default();
let result = compute_butteraugli(original, compressed, width, height, &params)
    .expect("valid image data");

println!("Butteraugli score: {:.4}", result.score);

if result.score < 1.0 {
    println!("Images appear identical!");
} else if result.score < 2.0 {
    println!("Minor visible differences");
} else {
    println!("Significant visible differences");
}

// Optional: access per-pixel difference map
if let Some(diffmap) = result.diffmap {
    let max_diff = (0..height)
        .flat_map(|y| (0..width).map(move |x| diffmap.get(x, y)))
        .fold(0.0f32, f32::max);
    println!("Maximum local difference: {:.4}", max_diff);
}
```

### Linear RGB Example (HDR/16-bit)

```rust
use butteraugli_oxide::{compute_butteraugli_linear, ButteraugliParams, srgb_to_linear};

// Convert 16-bit image to linear f32
let original_16bit: &[u16] = &[/* 16-bit RGB data */];
let original_linear: Vec<f32> = original_16bit.iter()
    .map(|&v| v as f32 / 65535.0)  // Assuming already linear
    .collect();

// Or convert 8-bit sRGB manually
let original_srgb: &[u8] = &[/* sRGB data */];
let original_linear: Vec<f32> = original_srgb.iter()
    .map(|&v| srgb_to_linear(v))
    .collect();

let result = compute_butteraugli_linear(&original_linear, &compressed_linear, width, height, &ButteraugliParams::default())
    .expect("valid image data");
```

### Custom Parameters

```rust
use butteraugli_oxide::ButteraugliParams;

let params = ButteraugliParams::new()
    .with_hf_asymmetry(1.5)      // Penalize new artifacts more than blurring
    .with_xmul(1.0)              // X channel multiplier (1.0 = neutral)
    .with_intensity_target(250.0); // HDR display brightness in nits
```

### Helper Functions

```rust
use butteraugli_oxide::{score_to_quality, butteraugli_fuzzy_class};

// Convert score to 0-100 quality percentage
let quality = score_to_quality(1.5);  // ~62.5%

// Get fuzzy classification (2.0 = perfect, 1.0 = ok, 0.0 = bad)
let class = butteraugli_fuzzy_class(1.5);  // ~1.25
```

## Features

- **`simd`** (default): Enable SIMD optimizations via the `wide` crate

## Performance

This implementation uses SIMD operations where available for gaussian blur and other compute-intensive operations. Performance is comparable to the C++ implementation for typical image sizes.

## Accuracy

The implementation includes 195 synthetic test cases validated against the C++ libjxl butteraugli. Reference values are captured from C++ and hard-coded for regression testing without requiring FFI bindings at runtime.

## Comparison with Other Crates

| Crate | Type | Notes |
|-------|------|-------|
| `butteraugli-oxide` | Pure Rust | Full implementation, no C++ dependency |
| `butteraugli` | FFI wrapper | Wraps C++ butteraugli library |
| `butteraugli-sys` | FFI bindings | Low-level C++ bindings |

### API Comparison with C++ libjxl

| Feature | C++ butteraugli | butteraugli-oxide |
|---------|-----------------|-------------------|
| Input format | Linear RGB float | sRGB u8 or linear RGB f32 |
| Bit depth | Any (via float) | 8-bit u8 or f32 |
| Color space | Linear RGB only | sRGB (auto-converted) or linear RGB |
| HDR support | Yes | Yes (via `compute_butteraugli_linear`) |
| Channel layout | Planar (separate R, G, B arrays) | Interleaved (RGBRGB...) |

### XYB Color Space Note

**Butteraugli's internal XYB is NOT the same as jpegli's XYB.**

| Aspect | Butteraugli XYB | jpegli XYB |
|--------|-----------------|------------|
| Nonlinearity | Gamma (FastLog2f-based) | Cube root |
| Opsin matrix | Different coefficients | Different coefficients |
| Dynamic sensitivity | Yes (blur-based adaptation) | No |
| XY formula | X = L - M, Y = L + M | X = (L-M)/2, Y = (L+M)/2 |

This crate does NOT accept XYB input directly because there are multiple incompatible XYB definitions. Always provide RGB input and let butteraugli perform its own internal conversion.

## References

- [Original butteraugli repository](https://github.com/google/butteraugli)
- [JPEG XL (libjxl)](https://github.com/libjxl/libjxl) - Contains the reference implementation
- [Butteraugli paper](https://github.com/google/butteraugli/blob/master/doc/butteraugli-theory.pdf)

## AI-Generated Code Notice

This crate was developed with significant assistance from Claude (Anthropic). While the code has been tested against the C++ libjxl butteraugli implementation and passes 195 synthetic test cases with exact numerical parity, **not all code has been manually reviewed or human-audited**.

Before using in production:
- Review critical code paths for your use case
- Run your own validation against expected outputs
- Consider the test suite coverage for your specific requirements

## License

BSD-3-Clause, same as the original libjxl implementation.
