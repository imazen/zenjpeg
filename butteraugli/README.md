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

### Custom Parameters

```rust
use butteraugli_oxide::ButteraugliParams;

let params = ButteraugliParams {
    // Penalize new high-frequency artifacts more than blurring (1.0 = neutral)
    hf_asymmetry: 1.0,
    // Multiplier for X channel differences (1.0 = neutral)
    xmul: 1.0,
    // Display intensity in nits (affects absolute thresholds)
    intensity_target: 80.0,
};
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

## References

- [Original butteraugli repository](https://github.com/google/butteraugli)
- [JPEG XL (libjxl)](https://github.com/libjxl/libjxl) - Contains the reference implementation
- [Butteraugli paper](https://github.com/google/butteraugli/blob/master/doc/butteraugli-theory.pdf)

## License

BSD-3-Clause, same as the original libjxl implementation.
