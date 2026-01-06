# jpegli-rs Project Guide

Pure Rust port of Google's jpegli JPEG encoder/decoder from the JPEG XL project.

## Quick Start

```bash
cargo build --release
cargo test --release
```

## C++ Parity Verification (IMPORTANT)

Run the comprehensive Rust vs C++ jpegli comparison matrix:

```bash
# Requires: git submodule update --init --recursive && build C++ jpegli first
cargo test --release -p jpegli-rs --features ffi-tests --test comprehensive_cpp_comparison -- --nocapture --ignored
```

This produces a table comparing 10 images × 50 quality levels showing:
- **Size**: Rust vs C++ file sizes (expect ~0.1% difference)
- **DSSIM**: Quality metric (expect 0.00% difference)
- **Butteraugli**: Perceptual quality (expect 0.00% difference)
- **Speed**: Encoding time comparison

Other parity tests:
```bash
# Quick locked parity tests (no C++ rebuild needed)
cargo test --release -p jpegli-rs --test cpp_parity_locked

# Encoder configuration matrix (all modes × all decoders)
cargo test --release -p jpegli-rs --features ffi-tests --test encoder_matrix -- --nocapture
```

## Tools

```bash
cargo run --release --example jpeg_inspect -- --validate image.jpg
```

## Project Structure

```
jpegli-rs/
├── jpegli-rs/           # Main library crate
│   ├── src/             # Encoder, decoder, color conversion
│   ├── examples/        # Debugging tools (see examples/README.md)
│   ├── tests/           # Integration tests
│   └── benches/         # Criterion benchmarks
├── jpegli-bench-utils/  # Shared utilities for benchmarks/examples
├── internal/jpegli-cpp/ # C++ jpegli submodule (for parity testing)
└── docs/                # Additional documentation
```

## Examples & Debugging Tools

See **`jpegli-rs/examples/README.md`** for complete documentation.

### Most Useful Tools

| Tool | Purpose |
|------|---------|
| `jpeg_inspect --validate` | Validate JPEGs with multiple decoders |
| `jpeg_inspect --all` | Full JPEG structure analysis |
| `quality_compare` | Compare encoder quality/size metrics |
| `xyb_parity_test` | Compare Rust vs C++ XYB output |
| `test_libjpeg_compat` | Verify libjpeg decoder compatibility |

### XYB Debugging (Quality Gap Investigation)

The XYB color space has a ~5 SSIMULACRA2 quality gap vs C++ jpegli:

```bash
# Compare file sizes and DSSIM
cargo run --release --example xyb_parity_test

# Check XYB conversion precision
cargo run --release --example xyb_ulp_parity

# Compare butteraugli scores
cargo run --release --example xyb_cpp_comparison

# Compare XYB vs YCbCr quality
cargo run --release --example xyb_vs_ycbcr_butteraugli
```

## Known Bugs

1. **Progressive XYB decode** - jpegli-rs decoder fails on progressive XYB JPEGs
   from C++ cjpegli. Baseline XYB works. See `tests/progressive_xyb_decode.rs`.

2. **XYB quality gap** - ~5 SSIMULACRA2 points behind C++ in XYB mode. Root cause TBD.

## Running Tests

```bash
# All tests
cargo test --release

# Specific test file
cargo test --release --test progressive_xyb_decode

# Ignored tests (require external files)
cargo test --release -- --ignored

# With test utilities
cargo test --release --features test-utils
```

## Benchmarks

```bash
# Encoding benchmark
cargo bench --bench encode

# Decoding benchmark
cargo bench --bench decode

# Quick throughput check
cargo run --release --example benchmark_sharp_yuv
```

## C++ Parity Testing

Requires `cjpegli`/`djpegli` binaries from libjxl build:

```bash
# Build C++ tools (in internal/jpegli-cpp)
mkdir -p build && cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DJPEGXL_ENABLE_TOOLS=ON ..
ninja cjpegli djpegli

# Run parity tests
cargo test --release --features ffi-tests
```

## Quality Metrics

**Use DSSIM or SSIMULACRA2, never PSNR.** PSNR doesn't correlate with perceptual quality.

```rust
// DSSIM (lower = better, 0 = identical)
use dssim::Dssim;

// SSIMULACRA2 (higher = better, 100 = identical)
use fast_ssim2::compute_frame_ssimulacra2;

// Butteraugli (lower = better, <1.0 = good)
use butteraugli::compute_butteraugli;
```

## Git Discipline

1. **Commit early, commit often** - Uncommitted work is invisible
2. **Run `cargo fmt` before changes** - Keep formatting commits separate
3. **Commit failing tests first** - Then fix in separate commit
4. **Never loosen test thresholds** - Find the real bug instead

## Feature Flags

```toml
[features]
default = ["simd"]
simd = []           # SIMD optimizations
test-utils = []     # Testing utilities (synthetic images, etc.)
ffi-tests = []      # C++ parity tests (requires jpegli-sys)
```

## Key Files for Debugging

| File | Purpose |
|------|---------|
| `jpegli-rs/src/encode.rs` | Encoder pipeline |
| `jpegli-rs/src/decode.rs` | Decoder pipeline |
| `jpegli-rs/src/xyb.rs` | XYB color conversion |
| `jpegli-rs/src/adaptive_quant.rs` | Adaptive quantization |
| `jpegli-rs/src/huffman.rs` | Huffman encoding |

## External Dependencies

- **Test images**: `internal/jpegli-cpp/testdata/`
- **Codec corpus**: `~/work/codec-eval/codec-corpus/` (Kodak, CID22)
- **C++ reference**: `internal/jpegli-cpp/` submodule

## Detailed Documentation

- `jpegli-rs/examples/README.md` - Examples and debugging tools
- `jpegli-rs/docs/ADAPTIVE_QUANTIZATION.md` - AQ algorithm details
- `internal/jpegli-cpp/jpegli-rs/CLAUDE.md` - Detailed handoff document
- `docs/SECURITY.md` - Security considerations
