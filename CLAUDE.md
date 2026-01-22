# jpegli-rs Project Guide

Pure Rust port of Google's jpegli JPEG encoder/decoder from the JPEG XL project.

## API Stability Rules (CRITICAL)

**DO NOT change the public API without explicit approval:**

1. **No re-exports at crate root** - Types stay in their modules (`encoder::EncoderConfig`, not `EncoderConfig`)
2. **No new public types/functions** without approval
3. **No changes to existing function signatures**
4. **Doc links use full paths** - `[`encoder::EncoderConfig`]` not `[`EncoderConfig`]`

## Performance Rules (CRITICAL)

**Never compromise sequential performance for parallel gains:**

1. Sequential encoding is the default and most common path
2. Parallel refactors must not add overhead to the sequential path
3. Use `#[cfg(feature = "parallel")]` to isolate parallel-only code
4. Benchmark both paths before and after changes
5. Frequency counting and other setup work can be parallelized separately from encoding

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

## Profiling Results (4K image, 2026-01-20)

Run with: `cargo flamegraph --release -p jpegli-rs --example flamegraph_profile -- 4k`
Then: `perf report --stdio --no-children -g none --percent-limit 0.5 2>/dev/null`

| Function | % Time | Notes |
|----------|--------|-------|
| `per_block_modulations_row` | 12.1% | AQ calculation |
| `encode_block_simd` | 12.1% | Entropy encoding |
| `finalize_imcu_aq_with_buffer` | 9.6% | AQ finalization |
| `forward_dct_8x8_wide` | 7.3% | DCT |
| `yuv::avx2::rgb_to_yuv` | 6.9% | Color conversion (yuv crate) |
| `pre_erosion_row` | 5.3% | AQ pre-erosion |
| `memmove_avx512` | 5.3% | Memory ops |
| `preprocess_deringing_f32` | 4.7% | Deringing |
| `collect_block_frequencies_simd` | 4.7% | Huffman freq counting |
| `quantize_block_zigzag` | 4.0% | Quantization |

**By category:**
- **Adaptive Quantization (AQ)**: 27.0% (per_block + finalize + pre_erosion)
- **Entropy Encoding**: 12.1%
- **Color Conversion**: 8.9% (yuv crate + rgb_to_ycbcr)
- **DCT**: 7.3%
- **Memory ops**: 6.1% (memmove + memset)
- **Quantization**: 6.3%
- **Huffman freq counting**: 4.7%
- **Deringing**: 4.7%

**Parallelization status:**
1. AQ calculation (27%) - NOT parallelizable (too fine-grained, see Failed Explorations)
2. Entropy encoding (12%) - already has parallel path
3. DCT (7%) - already has parallel path
4. Quantization (6%) - parallelizable with DCT
5. Frequency counting (5%) - sequential (DC prediction dependency)

## C++ Performance Gap (2026-01-21)

Run with: `cargo bench -p jpegli-rs --bench cpp_comparison`

**WARNING**: The `comprehensive_cpp_comparison` test uses subprocess timing (unfair).
Use the FFI benchmark above for accurate library-to-library comparison.

### Summary

Rust is consistently **1.5x slower** than C++ jpegli (FFI benchmark, 512x512).

**Fair comparison (both at `-C target-cpu=native`):**

| Quality | Rust | C++ FFI | Ratio |
|---------|------|---------|-------|
| q50 | 2.14ms | 1.40ms | 1.53x |
| q75 | 2.26ms | 1.48ms | 1.53x |
| q90 | 2.50ms | 1.70ms | 1.47x |
| q95 | 2.90ms | 1.90ms | 1.53x |

**Without native (Rust SSE2 only, C++ uses Highway runtime dispatch):**

| Quality | Rust | C++ FFI | Ratio |
|---------|------|---------|-------|
| q50 | 2.32ms | 1.49ms | 1.56x |
| q75 | 2.41ms | 1.54ms | 1.56x |

Note: C++ times are similar because Highway uses runtime SIMD dispatch regardless of
compile flags. The `wide` crate uses compile-time `cfg(target_feature)` checks.

### Allocation Optimization (2026-01-21)

Reduced allocations from 33,595 to 5,272 per 10 encodes (84% reduction):
- `Vec::with_capacity` in `generate_code_lengths` (classic.rs:187)
- Fixed array instead of `Vec<Vec>` in `depths_to_bits_values` (classic.rs:272)
- Lazy error creation with `ok_or_else` in progressive.rs:78
- Reusable buffers for YUV conversion and AQ strengths

Remaining 527 allocations/encode are inherent to Huffman table generation (13 scans × ~40 allocations each).

### Root Causes

1. **AQ computation (35% of time)** - Biggest contributor
   - C++ uses Highway SIMD with AVX-512 for all AQ functions
   - Rust uses `wide` crate (AVX2-level, f32x8)
   - `hf_modulation_sum_8x8` still has scalar fallback for rightmost block column

2. **Entropy encoding (14%)**
   - Both use similar algorithms
   - Needs assembly comparison

3. **DCT** - Highway has better AVX-512 optimizations

### Padded AQ Buffers Optimization (2026-01-18)

**Problem**: StreamingAQ was discarding MCU-aligned padding from input strips.

**Solution** (`jpegli-rs/src/quant/aq/streaming.rs`):
- Added `padded_width` field (blocks_w × 8) for MCU-aligned buffer stride
- y_imcu_buffers now allocated with padded_width instead of width
- Pass padded_width to per_block_modulations_row for aligned SIMD access

**Quick benchmark results** (vs previous Rust baseline):
- prog-opt-420: 24-27% faster
- base-opt-420: 46-49% faster
- prog-opt-444: 47-50% faster

Note: These gains are vs previous Rust, NOT vs C++. The 1.6x gap to C++ remains.

### Remaining SIMD Edge Cases

`hf_modulation_sum_8x8` (line 539) still uses scalar fallback for horizontal
differences in rightmost block column due to `block_x + 8 < img_width` check.
This affects ~1.5% of blocks.

To eliminate: would need 1 extra pixel of buffer padding for wraparound reads.

### wide vs archmage SIMD Analysis (2026-01-20)

**Benchmark:** `cargo bench -p jpegli-rs --bench aq_simd --features "archmage-simd,test-utils"`

**Key finding:** The `wide` crate uses `cfg(target_feature)` (compile-time check), NOT
`#[target_feature]` (function-level attribute). This means `#[multiversed]` dispatch
doesn't help - wide falls back to SSE (128-bit xmm) without global AVX2.

**Isolated primitive benchmarks** (misleading):
- `ratio_of_derivatives_x8`: wide 2.1ns, archmage 11.5ns - wide 5.5x faster
- `hf_modulation_sum_8x8`: wide 12.9ns, archmage 9.3ns - archmage 1.4x faster

**Why archmage slower for simple primitives:** `#[target_feature]` prevents inlining,
causing YMM register spills at call boundaries.

**Outer-level benchmarks** (production-representative):
- Without global AVX2 (`-C target-cpu=x86-64`):
  - `pre_erosion_row` width=4096: wide 2.49µs, archmage 1.16µs - **archmage 2.2x faster**
- With global AVX2 (`-C target-cpu=x86-64-v3`):
  - Gap narrows to ~10-17% (wide now uses ymm registers)

**Verification:** `cargo asm` shows 0 ymm usages without global AVX2, 83 ymm usages with it.

**Conclusion:** For production builds without global target-cpu flags, `wide` crate
underperforms significantly. Options:
1. Build with `-C target-cpu=x86-64-v3` (requires AVX2 at runtime)
2. Use archmage with `#[arcane]` macro for AQ functions
3. Use raw intrinsics guarded by runtime feature detection
4. **Use `multiversion` crate for autovectorization** (see below)

### Autovectorization with multiversion (2026-01-21)

**KEY DISCOVERY:** Pure scalar Rust code can be autovectorized to match manual SIMD
by using the `multiversion` crate for runtime dispatch.

**Benchmark results** (8x8 f32 transpose, `jpegli-rs/examples/autovec_transpose.rs`):

| Implementation | Time | Speedup |
|---------------|------|---------|
| Naive scalar | 13.31 ns | 1.0x |
| `#[multiversion]` | 4.73 ns | **2.8x** |

**How it works:**
1. Decorate function with `#[multiversion(targets("x86_64+avx2+fma", "x86_64+sse4.1", "aarch64+neon"))]`
2. Compiler generates separate versions with different `#[target_feature]` attributes
3. Runtime dispatcher picks the best version based on CPU features
4. LLVM autovectorizes each version with the enabled instruction set

**Assembly verification:** The AVX2 version uses `vunpcklps`, `vinsertf128`, `vshufps`,
`vblendps` - the **exact same instructions** as hand-written SIMD!

**When autovectorization works well:**
- Integer operations (excellent)
- Float operations with simple memory patterns (good when multiversion enables AVX2)
- Operations without data-dependent branches

**When autovectorization fails:**
- Complex gather/scatter patterns (without global AVX2)
- Operations with conditional branches in inner loops
- Floating-point requiring strict IEEE semantics (compiler can't reorder)

**Recommended approach:**
```rust
#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "x86_64+sse4.1", "aarch64+neon"))]
fn process(data: &mut [f32]) {
    // Write simple scalar code - compiler autovectorizes
    // NOTE: Simple loops vectorize better than explicit chunks!
    for x in data.iter_mut() {
        *x = x.sqrt();  // Becomes vsqrtps with AVX2
    }
}
```

**IMPORTANT:** Simple loops autovectorize better than explicit chunked loops.
The `for i in 0..8` pattern often prevents vectorization. Let the compiler decide.

**Files:** `jpegli-rs/examples/autovec_transpose.rs`

### Autovec vs Wide Crate AQ Benchmark (2026-01-21)

**Benchmark:** `cargo run --release -p jpegli-rs --example bench_autovec_aq`

**Results** (width=4096, 512 blocks):

| Function | wide | autovec | Result |
|----------|------|---------|--------|
| pre_erosion_row | 2.32 µs/row | 1.19 µs/row | **autovec 1.95x FASTER** |
| gamma_modulation_sum_8x8 | 16.1 ns/block | 34.6 ns/block | wide 2.15x faster |
| hf_modulation_sum_8x8 | 9.4 ns/block | 47.8 ns/block | wide 5.08x faster |
| per_block_modulations_row | 29.8 µs/row | 50.9 µs/row | wide 1.71x faster |

**Key insight:** Autovectorization works well for:
- Long rows with simple iterator loops (pre_erosion_row → 1.95x faster)
- Linear iteration without complex boundary checks

Autovectorization fails for:
- Small 8x8 blocks with nested loops (gamma/hf_modulation → 2-5x slower)
- Loops with per-element boundary checks preventing vectorization

**Integration:** The faster autovec `pre_erosion_row` is now integrated into the
streaming AQ path (`streaming.rs:31`). The slower gamma/hf/per_block functions
continue to use the wide-based SIMD.

**Why wide is slower for pre_erosion:** The `wide` crate uses `cfg(target_feature)`
for compile-time detection. Without global `-C target-cpu=x86-64-v3`, it falls back
to SSE even inside `#[multiversed]` functions. The `multiversion` crate uses
`#[target_feature]` which enables proper AVX2 codegen.

**Files:** `jpegli-rs/src/quant/aq/autovec.rs`, `jpegli-rs/examples/bench_autovec_aq.rs`

## Failed Explorations

### Parallel AQ (2026-01-17)

**Attempted:** Parallelize `per_block_modulations_row` using rayon.

**Why it failed:**
- Per-block AQ computation takes ~0.2 microseconds
- Far too small for rayon thread pool overhead to be worthwhile
- 4K benchmark with threshold=256: **5x slower** than sequential
- Even 8K (33M pixels, 518K blocks) wouldn't benefit

**Analysis:**
- AQ takes 26% of 8K encode time, but only ~15% is parallelizable
- `pre_erosion_row` (6%) has row-to-row accumulation dependency
- `fuzzy_erosion` (5%) needs 3x3 neighborhood lookahead
- Max theoretical speedup with 4 threads: ~10% overall
- After rayon overhead: ~6% realistic gain - not worth complexity

**Conclusion:** The SIMD-optimized sequential path is already efficient. Thread-level parallelism would need coarser granularity (e.g., multiple iMCU rows buffered) to overcome overhead, which conflicts with the streaming architecture.

### Fuzzy Erosion SIMD (2026-01-21)

**Attempted:** SIMD-optimize `compute_fuzzy_erosion_row_into` with archmage.

**Approaches tried:**
1. **Archmage with helper functions**: Created `mage_compute_fuzzy_erosion_row` with separate
   `weighted_4_smallest`, `gather_3x3_clamped`, `gather_3x3_interior` helpers.
   Result: **3x slower** (67ms vs 53ms) - `#[arcane]` prevents inlining, causing YMM register
   spills at every function call boundary.

2. **Massive inlined function**: ~350 lines with all 4 corners fully unrolled, no helper calls.
   Result: **Still slower** (68ms vs 53ms) - instruction cache pressure from code bloat.

**Root cause analysis:**
- The algorithm requires finding 4 smallest from 9 values with index tracking
- Scalar partial sort: `find min → replace with MAX → repeat 4×`
- This creates unpredictable branch patterns that SIMD doesn't help
- Code bloat from unrolling hurts icache more than SIMD helps

**What would actually help:**
- True SIMD sorting network (e.g., bitonic sort for 16 elements)
- Would need to process multiple blocks in parallel, not just vectorize one block
- Complexity not justified for ~5% of encode time

**Files:** `jpegli-rs/src/quant/aq/simd.rs:1377` (massive version, unused),
`jpegli-rs/src/quant/aq/streaming.rs:631` (original scalar, in use)

### AVX-512 Dual-Block DCT (2026-01-21)

**Attempted:** Process two 8x8 blocks simultaneously using AVX-512 (512-bit = 16 floats = 2 blocks).

**Implementation:** Pack two blocks into ZMM registers [A_row_i, B_row_i], do DCT butterflies
with AVX-512 arithmetic, transpose with extract/AVX2/insert pattern.

**Benchmark results:**
- AVX2 single-block: 41.19M blocks/sec
- AVX-512 dual-block: 17.58M blocks/sec (2.3x **slower**)

**Why it failed:**
1. **8x8 blocks fit AVX2 perfectly** - 8 floats = 256 bits, no wasted register space
2. **Transpose cannot be done natively in AVX-512** - `_mm512_unpacklo_ps` operates on 128-bit
   lanes, mixing data between blocks A and B
3. **Extract/insert workaround is expensive** - each transpose requires:
   - 8 `_mm512_extractf32x8_ps` to split ZMM→YMM
   - 48 AVX2 operations (two 8x8 transposes)
   - 16 `_mm512_insertf32x8` to recombine YMM→ZMM
4. **Two transposes per DCT** = 64 extra extract/insert operations
5. **AVX-512 frequency throttling** on some CPUs adds further penalty

**Conclusion:** AVX-512 benefits require naturally 16-wide workloads. 8x8 DCT is inherently
8-wide, making AVX2 the optimal register width. Dual-block packing just adds overhead.

**Files:** `jpegli-rs/src/encode/mage_simd.rs:600-775` (kept for reference, not used in encoder)

## Known Bugs

0. **Debug env var in hot loop (FIXED)** - `jpegli-rs/src/entropy/encoder.rs:798`
   `std::env::var("DEBUG_HUFFMAN_LOOKUP")` was called on every token write.
   Even though the debug code only ran when the env var existed, the syscall
   overhead consumed ~12% of total encode time. Removed entirely.

4. **Eager error evaluation in hot path (FIXED)** - `jpegli-rs/src/entropy/encoder.rs:308-313`
   `ok_or(Error::internal(...))` eagerly evaluates the error argument on every call.
   `Error::new()` → `AtTraceBoxed::capture()` → malloc per call.
   For 8K: ~4M blocks × 2 lookups = 8M unnecessary allocations (11.4% of encode time).
   **Fix**: Changed to `ok_or_else(|| ...)` for lazy evaluation. **13% speedup.**

1. **Progressive XYB decode (FIXED)** - `jpegli-rs/src/decode/mod.rs:1187-1275`
   Progressive DC scans now handle `EndOfScanData` gracefully (same as AC scans).
   Previously failed on XYB with non-standard component IDs (R/G/B = 82/71/66).
   See `tests/progressive_xyb_decode.rs`.

2. **XYB quality gap** - ~5 SSIMULACRA2 points behind C++ in XYB mode. Root cause TBD.

5. **1-pixel partial MCU edge quality gap** - `jpegli-rs/tests/edge_tile_ssim2_comparison.rs`
   Images with width ≡ 1 (mod 8) show -22 to -35 SSIMULACRA2 gap vs C++ jpegli.
   3+ pixel partial edges achieve parity. Investigated AQ edge handling (padded
   buffers, stride vs img_width separation) but gap persists. Root cause likely
   in pre-erosion edge handling or how single-pixel blocks are encoded.
   Run: `cargo test --release -p jpegli-rs --test edge_tile_ssim2_comparison -- --ignored`

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
archmage-simd = ["dep:archmage"]  # Token-based SIMD for AQ (~6% faster)
cms = ["cms-lcms2"]       # Color management
ffi-tests = []            # C++ parity tests (requires jpegli-sys)
corpus-tests = []         # Corpus comparison tests
test-utils = []           # Testing utilities
```

**Decoder:** The decoder API is in prerelease. Enable with `features = ["decoder"]`.
API will have breaking changes.

**SIMD options:**
- Default: `wide` crate (portable, safe) - always enabled
- `archmage-simd`: Token-based safe intrinsics for AQ functions - **~6% faster** on x86_64
- `unsafe_simd`: Raw AVX2/SSE intrinsics - ~10-20% speedup on x86_64

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
