# zenjpeg Examples

Tools for debugging, benchmarking, and validating the zenjpeg encoder/decoder.

## Quick Reference

| Example | Purpose | Usage |
|---------|---------|-------|
| `jpeg_inspect` | JPEG structure analysis & validation | `cargo run --release --example jpeg_inspect -- image.jpg` |
| `quality_compare` | Compare encoder quality/size | `cargo run --release --example quality_compare -- image.png` |
| `test_libjpeg_compat` | Verify libjpeg compatibility | `cargo run --release --example test_libjpeg_compat` |
| `benchmark_sharp_yuv` | Encoder throughput benchmarks | `cargo run --release --example benchmark_sharp_yuv` |
| `low_q_report` | HTML quality report (Rust vs C++) | `cargo run --release --example low_q_report --features test-utils` |

## JPEG Inspection & Validation

### jpeg_inspect

Comprehensive JPEG analysis tool. Analyzes markers, quantization tables, Huffman tables,
progressive scans, and validates decoding with multiple decoders.

```bash
# Quick overview
cargo run --release --example jpeg_inspect -- image.jpg

# Detailed analysis
cargo run --release --example jpeg_inspect -- --all image.jpg

# Validate with multiple decoders (zune-jpeg, zenjpeg, mozjpeg)
cargo run --release --example jpeg_inspect -- --validate image.jpg

# Batch validate all JPEGs in a directory
cargo run --release --example jpeg_inspect -- --validate /path/to/jpgs/

# Analyze quantization tables
cargo run --release --example jpeg_inspect -- --quant image.jpg

# Analyze Huffman tables
cargo run --release --example jpeg_inspect -- --huffman image.jpg

# Analyze progressive scan structure
cargo run --release --example jpeg_inspect -- --scans progressive.jpg

# Compare two JPEGs
cargo run --release --example jpeg_inspect -- --compare other.jpg image.jpg
```

## Quality & Performance Testing

### quality_compare

Compare encoder quality metrics (DSSIM, SSIMULACRA2, Butteraugli) and file sizes.

```bash
# Single quality level
cargo run --release --example quality_compare -- --quality 80 --encoder zenjpeg-ycbcr image.png

# Compare multiple encoders
cargo run --release --example quality_compare -- --encoder zenjpeg-xyb --metric all image.png

# Sweep quality levels for Pareto curve
cargo run --release --example quality_compare -- --pareto image.png

# Available encoders: zenjpeg-ycbcr, zenjpeg-xyb, cmozjpeg
# Available metrics: dssim, ssim2, butteraugli, all
```

### benchmark_sharp_yuv

Encoder throughput benchmarks at various resolutions.

```bash
cargo run --release --example benchmark_sharp_yuv
# Output: MP/s throughput for 512x512, 1080p, 4K resolutions
```

### low_q_report

Generate HTML report comparing Rust vs C++ jpegli at low quality levels.

```bash
cargo run --release --example low_q_report --features test-utils
# Output: reports/low_q_report.html
```

## XYB Debugging Tools

XYB color space produces 5-11% larger files than C++ jpegli at equivalent quality.
These tools help debug the size difference (quality is identical).

### xyb_parity_test

Compare Rust vs C++ jpegli XYB output (file size and DSSIM).

```bash
cargo run --release --example xyb_parity_test
# Compares multiple Kodak images at Q70/80/90
# Shows size difference and DSSIM delta
```

### xyb_cpp_comparison

Detailed comparison of Rust vs C++ XYB encoding with Butteraugli metrics.

```bash
cargo run --release --example xyb_cpp_comparison
# Shows: quality level, file sizes, butteraugli scores
```

### xyb_vs_ycbcr_butteraugli

Compare XYB vs YCbCr encoding quality using Butteraugli metric.

```bash
cargo run --release --example xyb_vs_ycbcr_butteraugli
# Shows butteraugli scores for XYB vs YCbCr at each quality level
# Lower butteraugli = better quality
```

### xyb_ulp_parity

Test XYB color conversion precision (ULP = units in last place).

```bash
cargo run --release --example xyb_ulp_parity
# Shows max/mean error per XYB channel
# Verifies Rust XYB matches C++ at floating-point precision level
```

### compare_xyb_constants

Compare XYB conversion constants and full-image conversion accuracy vs C++.

```bash
cargo run --release --example compare_xyb_constants --features "test-utils,ffi-tests"
# Tests specific RGB values and full image conversion
# Shows per-channel max difference
```

### xyb_quality_check ⚠️

Compare XYB quality metrics between Rust and C++.

```bash
cargo run --release --example xyb_quality_check
# NOTE: Currently has a decoder bug with progressive XYB JPEGs
# See tests/progressive_xyb_decode.rs for details
```

## Compatibility Testing

### test_libjpeg_compat

Verify zenjpeg output is compatible with libjpeg/djpeg decoder.

```bash
cargo run --release --example test_libjpeg_compat
# Tests various image patterns at multiple quality levels
# Verifies djpeg can decode all outputs
```

### generate_cpp_reference

Generate C++ jpegli reference outputs for comparison testing.

```bash
cargo run --release --example generate_cpp_reference
# Requires cjpegli binary in PATH
```

## Dependencies

Some examples require additional features or external tools:

```bash
# For low_q_report and some XYB tools
cargo run --release --example low_q_report --features test-utils

# For C++ comparison tools
cargo run --release --example compare_xyb_constants --features "test-utils,ffi-tests"
```

External tools used by some examples:
- `cjpegli` / `djpegli` - C++ jpegli encoder/decoder (from libjxl)
- Test images in `internal/jpegli-cpp/testdata/`
- Codec corpus at `~/work/codec-eval/codec-corpus/`

## Known Issues

1. **Progressive XYB decode bug**: zenjpeg decoder fails on progressive XYB JPEGs
   generated by C++ cjpegli. See `tests/progressive_xyb_decode.rs` for test case.
   - Baseline XYB works fine
   - Issue is with non-standard component IDs (R=82, G=71, B=66) in progressive scans

2. **XYB size gap**: 5-11% larger files than C++ jpegli XYB mode (quality is identical, root cause TBD)

## Adding New Examples

When adding debugging tools:
1. Use `jpegli-bench-utils` crate for synthetic images and metrics
2. Follow naming convention: `<category>_<specific>.rs`
3. Add to this README with usage examples
4. Prefer consolidating into existing tools over creating new files
