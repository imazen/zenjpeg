# jpegli-rs Project Guide

Pure Rust port of Google's jpegli JPEG encoder/decoder from the JPEG XL project.

## API Stability Rules (CRITICAL)

**DO NOT change the public API without explicit approval:**

1. **No re-exports at crate root** - Types stay in their modules (`encoder::EncoderConfig`, not `EncoderConfig`)
2. **No new public types/functions** without approval
3. **No changes to existing function signatures**
4. **Doc links use full paths** - `[`encoder::EncoderConfig`]` not `[`EncoderConfig`]`

## Context Preservation (CRITICAL)

**You may lose context at any time.** Always record findings immediately:

1. **Suspected bugs** → Add to "Known Bugs" section below with file:line references
2. **Code analysis/details** → Save to `CODE.md` with:
   - Relevant code snippets
   - Root cause analysis
   - Proposed fixes
   - C++ reference behavior
3. **Investigation progress** → Commit WIP notes before switching tasks

**Do this BEFORE continuing investigation.** Lost context = wasted work.

## Quick Start

```bash
cargo build --release
cargo test --release
```

## C++ Parity Verification (IMPORTANT)

All features enabled by default. Tests auto-find corpus at `~/work/codec-eval/codec-corpus/`.

### Quick Parity Check
```bash
# Comprehensive test: 10 images × 50 quality levels (live cjpegli FFI)
cargo test --release -p jpegli-rs --test comprehensive_cpp_comparison -- --nocapture --ignored
```

Expected results:
- **Size**: +0.26% (Rust slightly larger)
- **DSSIM**: +0.15% (essentially identical quality)
- **Butteraugli**: -0.00% (identical)

### All C++ Comparison Tests
```bash
# Run ALL comparison tests with live C++ FFI
cargo test --release -p jpegli-rs -- comparison --nocapture --ignored

# Corpus-based comparison (CID22 images)
cargo test --release -p jpegli-rs --test corpus_cpp_comparison -- --nocapture --ignored

# XYB mode comparison (larger differences expected: 0.2-3%)
cargo run --release --example xyb_parity_test

# SSIMULACRA2 comparison (synthetic images)
cargo run --release --example ssim2_comparison
```

### Key Parity Tests
| Test | Command | Expected |
|------|---------|----------|
| comprehensive | `--test comprehensive_cpp_comparison` | Size +0.26%, DSSIM +0.15% |
| corpus | `--test corpus_cpp_comparison` | Size -0.1% |
| xyb | `--example xyb_parity_test` | Size 0.2-3% |
| locked | `--test cpp_parity_locked` | Hash-locked values |
| strip edges | `--test strip_edge_cpp_comparison` | DSSIM <0.6% diff |

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
| `edge_mcu_parity` | Test partial MCU edge handling parity |

### Edge MCU Parity Testing

The `edge_mcu_parity` example and `jpegli_bench_utils` provide tools for testing
edge-case handling in images with non-8-aligned dimensions:

```rust
use jpegli_bench_utils::{tile_edge_columns, create_edge_mcu_test_image, McuEdgeInfo};

// Analyze image for edge characteristics
let info = McuEdgeInfo::analyze(1118, 1105);
// partial_mcu_width = 6, affected_block_pct = 0.71%

// Create test image that amplifies edge bugs
let tiled = tile_edge_columns(&source, 6, 518);
// Now 100% of content comes from edge strip
```

Use `--edge-width N` to test specific column widths (1-7).

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

0. **Debug env var in hot loop (FIXED)** - `jpegli-rs/src/entropy/encoder.rs:798`
   `std::env::var("DEBUG_HUFFMAN_LOOKUP")` was called on every token write.
   Even though the debug code only ran when the env var existed, the syscall
   overhead consumed ~12% of total encode time. Removed entirely.

1. **Progressive XYB decode (FIXED)** - `jpegli-rs/src/decode/mod.rs:1187-1275`
   Progressive DC scans now handle `EndOfScanData` gracefully (same as AC scans).
   Previously failed on XYB with non-standard component IDs (R/G/B = 82/71/66).
   See `tests/progressive_xyb_decode.rs`.

2. **XYB quality gap** - ~5 SSIMULACRA2 points behind C++ in XYB mode. Root cause TBD.

3. **HF modulation index wrap (FIXED)** - `jpegli-rs/src/quant/aq/simd.rs:566`
   Rightmost partial blocks were reading pixels from next row due to missing column check.
   - Added `block_x + 8 <= img_width` guard to vertical SIMD path
   - Affects images where `width % 8 != 0`
   - See `CODE.md` for full analysis

## Planned Features / TODO

### Resource Estimation API (docs/API_DESIGN.md)

For proxy server efficiency: accurate memory and compute cost estimation before encoding.

**Design:**
- `EncoderConfig` - dimension-independent config, reusable across images
- `ResourceEstimate` - returns `peak_bytes` (public), tracks internal metrics
- `InputMethod` enum - OneShot, Streaming, YCbCrDirect, YCbCrSubsampled
- `compute_cost_ms()` - approximate encoding time for current architecture
- `EncodeMetrics` - actual values returned from `finish_with_metrics()`

**Implementation TODO:**
- [ ] Extract `EncoderConfig` from `JpegEncoder` (dimension-independent)
- [ ] Add `InputMethod` enum for different input paths
- [ ] Implement `estimate_resources(width, height, input_method)` with accurate modeling
- [ ] Add allocation tracking (behind feature flag for zero overhead in production)
- [ ] Add `compute_cost_ms()` with CPU detection and calibration
- [ ] Add `EncodeMetrics` struct with actual peak_bytes, alloc_count, total_alloc_bytes, elapsed_ms
- [ ] Add `finish_with_metrics()` to return EncodeMetrics
- [ ] Add `finish_into(buffer)` for zero-copy output to pre-allocated buffer
- [ ] Benchmark to calibrate `compute_cost_ms()` estimates across architectures

**Internal tracking (not public API):**
- Total allocation count
- Total allocated bytes
- Max single allocation size

## Running Tests

### Default Tests (no external dependencies)
```bash
cargo test --release                    # All non-ignored tests (~340 tests)
cargo test --release --test <name>      # Specific test file
```

### Full Test Suite (requires C++, testdata, corpus)

Prerequisites:
1. Build C++ jpegli: `cd internal/jpegli-cpp && mkdir -p build && cd build && cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DJPEGXL_ENABLE_TOOLS=ON .. && ninja cjpegli djpegli`
2. Generate testdata: `GENERATE_RUST_TEST_DATA=1 ./build/tools/cjpegli input.png output.jpg`
3. Corpus available at `~/work/codec-eval/codec-corpus/`

```bash
# Run ALL tests including ignored ones
cargo test --release -- --ignored

# C++ parity tests (most important)
cargo test --release --test parity_enforcement -- --ignored     # 7 tests
cargo test --release --test cpp_comparison -- --ignored         # 4 tests
cargo test --release --test cpp_filesize_comparison -- --ignored

# Corpus-based tests (need corpus-tests feature)
cargo test --release --features corpus-tests -- --ignored

# FFI tests (need ffi-tests feature + C++ build)
cargo test --release --features ffi-tests -- --ignored
```

### Test Categories

| Category | Command | Notes |
|----------|---------|-------|
| Unit tests | `cargo test --release --lib` | 324 tests, no deps |
| Integration | `cargo test --release` | Includes strip parity |
| C++ parity | `cargo test --release -- --ignored` | Needs C++ build |
| Corpus | `--features corpus-tests -- --ignored` | Needs image corpus |
| FFI | `--features ffi-tests` | Direct C++ bindings |

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

## Color Space Constraints

### RGB→YCbCr Matrix: BT.601 Only

**CRITICAL**: The RGB→YCbCr conversion matrix MUST remain BT.601 for cross-decoder compatibility.

When encoding wide-gamut images (Display P3, Adobe RGB, etc.):
- The ICC profile tells decoders how to interpret the *decoded* RGB values
- The YCbCr encoding uses the **same standard BT.601 matrix** as sRGB
- Do NOT use colorspace-specific RGB→YCbCr matrices

**Correct P3 encoding pipeline:**
```
sRGB u8 → linearize → f32 linear sRGB
                          ↓
              gamut expansion (f32 linear)
                          ↓
                   f32 linear P3
                          ↓
              apply P3 gamma → f32 gamma P3
                          ↓
         jpegli (standard BT.601 RGB→YCbCr)  ← NOT a P3-specific matrix
                          ↓
                 JPEG + P3 ICC profile
```

The ICC profile is metadata only - it doesn't change the YCbCr encoding math.

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
default = ["cms", "test-utils"]
decoder = []              # Enable decoder (prerelease, API will change)
parallel = ["dep:rayon"]  # Multi-threaded DCT/quantization
unsafe_simd = []          # Raw AVX2/SSE intrinsics (opt-in)
cms = ["cms-lcms2"]       # Color management
ffi-tests = []            # C++ parity tests (requires jpegli-sys)
corpus-tests = []         # Corpus comparison tests
test-utils = []           # Testing utilities
```

**Decoder:** The decoder API is in prerelease. Enable with `features = ["decoder"]`.
API will have breaking changes.

SIMD via the `wide` crate is always enabled (portable, safe).
The `unsafe_simd` feature enables raw AVX2/SSE intrinsics for ~10-20% speedup on x86_64.

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

- **`jpegli-rs/README.md` - API Reference** (encoder/decoder usage with examples)
- `jpegli-rs/examples/README.md` - Examples and debugging tools
- `jpegli-rs/docs/ADAPTIVE_QUANTIZATION.md` - AQ algorithm details
- `jpegli-rs/docs/API_DESIGN.md` - Full API surface and proposed enhancements
- `internal/jpegli-cpp/jpegli-rs/CLAUDE.md` - Detailed handoff document
- `docs/SECURITY.md` - Security considerations
