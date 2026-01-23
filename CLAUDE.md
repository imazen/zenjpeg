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

Expected results (using `jpegli_set_distance` for 3-table parity):
- **Size**: ~0% (within ±1%)
- **DSSIM**: ~0% (within ±1%)
- **Butteraugli**: ~0% (within ±1%)

**Note:** Results now use distance-based encoding (`jpegli_set_distance()`) instead
of `jpeg_set_quality()`. This ensures both encoders use 3 quant tables (Y, Cb, Cr).
Previous results showed ~4% differences due to `jpeg_set_quality()` using 2 tables.

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
| comprehensive | `--test comprehensive_cpp_comparison` | Size ~0%, DSSIM ~0% |
| corpus | `--test corpus_cpp_comparison` | Size ~0% |
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

## Profiling Results (4K image, 2026-01-21)

Run with: `cargo flamegraph --release -p jpegli-rs --example flamegraph_profile -- 4k`
Then: `perf report --stdio --no-children -g none --percent-limit 1.0 2>/dev/null`

| Function | % Time | Notes |
|----------|--------|-------|
| `encode_block_simd` | 11.7% | Entropy encoding |
| `per_block_modulations_row` | 9.1% | AQ calculation |
| `preprocess_deringing_f32` | 6.9% | Deringing |
| `build_optimized_tables` | 6.1% | Huffman table building |
| `dct_strip_blocks_to_pending` | 4.5% | DCT |
| `quantize_pending_imcu` | 4.1% | Quantization |
| `yuv::avx2::rgb_to_yuv` | 4.0% | Color conversion (yuv crate) |
| `downsample_2x2_simd_inplace` | 3.0% | Chroma subsampling |
| `finalize_imcu_aq_with_buffer` | 2.8% | AQ finalization (SIMD fuzzy erosion) |
| `pre_erosion_row_autovec_iter` | 2.6% | AQ pre-erosion |
| `memmove_avx512` | 2.6% | Memory ops |

**By category (after SIMD sorting network optimization):**
- **Adaptive Quantization (AQ)**: 14.5% (per_block + finalize + pre_erosion)
- **Entropy Encoding**: 11.7%
- **Deringing**: 6.9%
- **Huffman table building**: 6.1%
- **DCT + Quantization**: 8.6%
- **Color Conversion**: 5.8% (yuv crate + rgb_to_ycbcr)
- **Memory ops**: 2.6%

**Improvement from SIMD sorting network (2026-01-21):**
- `finalize_imcu_aq_with_buffer`: 9.6% → 2.8% (3.4x faster)
- Total AQ: 27% → 14.5% (1.86x faster)

**Parallelization status:**
1. AQ calculation (14.5%) - SIMD-optimized, not worth parallelizing
2. Entropy encoding (12%) - already has parallel path
3. DCT + Quantization (8.6%) - already has parallel path
4. Frequency counting - sequential (DC prediction dependency)

## Decoder Profiling (512x512 image, 2026-01-22)

Run with: `valgrind --tool=callgrind ./target/release/examples/valgrind_decode jpegli 512`

| Function | Instructions | % | Notes |
|----------|--------------|---|-------|
| `idct_int_avx2` | 1.6M | 4.0% | AVX2 IDCT (DC-only check built-in) |
| `upsample_h2v2_i16_fancy_avx2` | 0.7M | 1.7% | Chroma upsampling |
| `to_pixels` | 2.9M | 7.1% | Dequantization |
| `ycbcr_planes_i16_to_rgb_u8_avx2` | 0.8M | 1.9% | YCbCr→RGB |
| `decode_scan` | 1.1M | 2.7% | Entropy decoding |

**Optimization progress (2026-01-22):**
- Started: 60.3M instructions (1.74x vs zune-jpeg)
- After AVX2 upsampling: 46.3M (-23%)
- After AVX2 IDCT: 40.5M (-33% total, 1.17x vs zune)

**Key insight:** Tiered 4x4/8x8 IDCT was counterproductive. The scalar 4x4 IDCT
for sparse blocks took 7M instructions, while AVX2 8x8 IDCT (with built-in DC-only
check) takes only 1.6M. Removed the tiering - always use AVX2 8x8 for non-DC blocks.

**Benchmark results (512x512):**
| Mode | Before | After | Improvement |
|------|--------|-------|-------------|
| Baseline | 1.31ms | 456µs | 65% faster |
| Progressive | 2.03ms | 1.15ms | 43% faster |

## C++ Performance Gap (2026-01-21)

Run with: `cargo bench -p jpegli-rs --bench cpp_comparison`

**WARNING**: The `comprehensive_cpp_comparison` test uses subprocess timing (unfair).
Use the FFI benchmark above for accurate library-to-library comparison.

### Summary

Rust is consistently **1.4x slower** than C++ jpegli (FFI benchmark, 512x512).
Improved from 1.6x after SIMD sorting network optimization.

**Fair comparison (both at `-C target-cpu=native`):**

| Quality | Rust | C++ FFI | Ratio |
|---------|------|---------|-------|
| q50 | 2.22ms | 1.49ms | 1.49x |
| q75 | 2.24ms | 1.53ms | 1.46x |
| q90 | 2.45ms | 1.75ms | 1.40x |
| q95 | 2.74ms | 2.01ms | 1.36x |

Note: Gap was 1.6x before SIMD sorting network. Highway uses runtime SIMD dispatch
regardless of compile flags, while `wide` crate uses compile-time `cfg(target_feature)`.

### Allocation Optimization (2026-01-21)

Reduced allocations from 33,595 to 5,272 per 10 encodes (84% reduction):
- `Vec::with_capacity` in `generate_code_lengths` (classic.rs:187)
- Fixed array instead of `Vec<Vec>` in `depths_to_bits_values` (classic.rs:272)
- Lazy error creation with `ok_or_else` in progressive.rs:78
- Reusable buffers for YUV conversion and AQ strengths

Remaining 527 allocations/encode are inherent to Huffman table generation (13 scans × ~40 allocations each).

### Root Causes

1. **AQ computation (14.5% of time)** - After SIMD sorting network optimization
   - C++ uses Highway SIMD with AVX-512 for all AQ functions
   - Rust uses `wide` crate (AVX2-level, f32x8)
   - `hf_modulation_sum_8x8` still has scalar fallback for rightmost block column

2. **Entropy encoding (12%)**
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

Note: These gains are vs previous Rust, NOT vs C++. Gap to C++ is now 1.4x (was 1.6x before SIMD sorting network).

### Remaining SIMD Edge Cases

`hf_modulation_sum_8x8` (line 539) still uses scalar fallback for horizontal
differences in rightmost block column due to `block_x + 8 < img_width` check.
This affects ~1.5% of blocks.

To eliminate: would need 1 extra pixel of buffer padding for wraparound reads.

### wide vs archmage SIMD Analysis (2026-01-20)

**Benchmark:** `cargo bench -p jpegli-rs --bench aq_simd --features "archmage-simd,test-utils"`

**Key finding:** The `wide` crate usually autovectorizes well, but sometimes picks
intrinsics that LLVM won't re-autovectorize to wider registers. This is operation-
dependent - most `wide` code benefits from `#[multiversed]` dispatch, but some paths
may stay at SSE-width even when AVX2 is enabled at the function level.

**Isolated primitive benchmarks**:
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

**Conclusion:** The `wide` crate autovectorizes well for most operations. When profiling
shows a specific function underperforming, check the assembly - some `wide` intrinsic
choices may not re-autovectorize. Options for those cases:
1. Build with `-C target-cpu=x86-64-v3` (requires AVX2 at runtime)
2. Rewrite as scalar code inside `#[multiversion]` for LLVM autovectorization
3. Use archmage with `#[arcane]` for explicit intrinsics (watch for inlining issues)

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

**Why wide is slower for pre_erosion:** In this specific case, `wide`'s intrinsic
choices don't re-autovectorize to AVX2 inside `#[multiversed]` functions. The scalar
autovec version lets LLVM make optimal choices for each target. This is operation-
dependent - most `wide` code autovectorizes fine, but `pre_erosion_row` hit a case
where scalar + `#[multiversion]` wins.

**Files:** `jpegli-rs/src/quant/aq/autovec.rs`, `jpegli-rs/examples/bench_autovec_aq.rs`

## Decoder Performance Gap (2026-01-22)

Run with: `cargo bench -p jpegli-rs --bench decode_compare`

### Summary

jpegli-rs decoder is **4-5x slower** than zune-jpeg for baseline JPEG and **2-4x slower** for progressive JPEG.

**Baseline JPEG (sequential Huffman):**

| Size | zune-jpeg | jpegli-rs | Ratio |
|------|-----------|-----------|-------|
| 256x256 | 82 µs | 337 µs | 4.1x |
| 512x512 | 285 µs | 1.31 ms | 4.6x |
| 1024x1024 | 946 µs | 4.56 ms | 4.8x |
| 2048x2048 | 3.80 ms | 19.2 ms | 5.0x |

**Progressive JPEG:**

| Size | zune-jpeg | jpegli-rs | Ratio |
|------|-----------|-----------|-------|
| 256x256 | 240 µs | 519 µs | 2.2x |
| 512x512 | 860 µs | 2.03 ms | 2.4x |
| 1024x1024 | 2.46 ms | 10.0 ms | 4.1x |
| 2048x2048 | 9.16 ms | 32.0 ms | 3.5x |

**Throughput comparison (2048x2048):**
- zune-jpeg baseline: 1104 MP/s
- jpegli-rs baseline: 219 MP/s
- zune-jpeg progressive: 458 MP/s
- jpegli-rs progressive: 131 MP/s

### Known Performance Issues

1. **Entropy decoder (10.6% of decode time)** - vs zune's 4%
   - Uses `ScanRead` enum with match on every bit read
   - Branchless `huff_extend` added but control flow overhead remains
   - zune-jpeg uses macros for inline Huffman with no enum matching

2. **Color conversion (20.5% `to_pixels` + 10% YCbCr→RGB)**
   - YCbCr to RGB uses autovectorized i16 path
   - Needs SIMD optimization like encoder's yuv crate

3. **memset overhead (5.57%)**
   - Zeroing coefficient arrays before each block decode
   - Zero-copy architecture could eliminate this

4. **IDCT (4.4% `idct_int_4x4`)**
   - Uses scaled integer IDCT
   - Not bottleneck currently

### Callgrind Analysis (2026-01-22)

Run: `valgrind --tool=callgrind ./target/release/examples/valgrind_decode jpegli 512`

**Totals (512x512 baseline JPEG):**

| Metric | jpegli-rs | zune-jpeg | Ratio |
|--------|-----------|-----------|-------|
| Instructions | 60.3M | 34.6M | **1.74x** |
| Data refs | 16.4M | 9.5M | 1.72x |
| Branches | 9.2M | 4.0M | 2.3x |
| D1 miss rate | 2.3% | 3.7% | better |
| Branch mispredict | 0.8% | 1.5% | better |

**Decode-specific function comparison:**

| Function | jpegli-rs | zune-jpeg | Notes |
|----------|-----------|-----------|-------|
| **Upsampling** | 10.5M (17%) | 115K (0.3%) | **91x more!** Scalar loop |
| **YCbCr→RGB** | 4.4M (7%) | 1.0M (3%) | 4.4x more |
| **IDCT** | 4.2M (7%) | 1.2M (3.5%) | 3.5x more |
| **Entropy decode** | ~1.5M | ~1.7M | similar |
| **memset** | 2.0M (3.4%) | 591K (1.7%) | 3.4x more |

**Root cause:** Upsampling is **91x worse** because `upsample_h2v2_i16_fancy` is fully scalar,
iterating pixel-by-pixel while zune uses AVX2 SIMD (`upsample_horizontal_avx2`).

### Decoder Optimization Path

1. ✅ Branchless huff_extend (done)
2. ✅ AVX2 upsampling (done) - **94% reduction** (10.5M → 0.7M instructions)
3. ✅ Autovectorized YCbCr conversion (done)
4. ✅ Zero-copy coefficient decode (5-7% speedup)
5. 🔲 **AVX2 IDCT** - Now biggest gap: 7M vs zune's 1.2M = **5.8x slower**
6. 🔲 Reduce memset overhead (2.6M instructions, 5.6%)
7. 🔲 Macro-based Huffman (eliminate ScanRead enum)

**Current status (2026-01-22):** 46.3M instructions (was 60.3M), zune is 34.6M.
Gap reduced from 1.74x to 1.34x in instruction count.

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

### Decoder Zero-Copy Architecture (2026-01-22) - IMPLEMENTED

**Problem:** Original decoder returned `([i16; 64], u8)` by value, copying 128 bytes per block.
Smart zeroing alone didn't help because copy dominated memory bandwidth.

**Solution:** Zero-copy `decode_block_into` API where caller provides reusable buffer:
```rust
fn decode_block_into(
    &mut self,
    coeffs: &mut [i16; 64],      // Caller-provided buffer
    prev_coeff_count: u8,        // Zeroing hint from previous block
    component: usize,
    dc_table_idx: usize,
    ac_table_idx: usize,
) -> ScanResult<u8>              // Returns new coeff count
```

**Key insight:** Reusable buffers accumulate state from ALL previous blocks, not just the
immediately previous one. If block N-2 wrote to position X, block N-1 didn't, and block N
doesn't either, position X still has stale data. Fix: track MAXIMUM coefficient count since
last restart marker, not just previous block's count.

**Implementation:**
- `entropy/decoder.rs`: Added `decode_block_into` with smart zeroing
- `decode/parser.rs`: Added `prev_coeff_counts: [u8; 4]` per-component tracking
- `decode/scanline.rs`: Added `coeffs_buf` reusable buffer and max-tracking

**Results (2026-01-22):**
- 512x512: ~5% improvement
- 2048x2048: 6.5% improvement (17.9ms vs 19.2ms)

Memory bandwidth reduction per block:
- Before: 128 bytes zeroing + 128 bytes copy = 256 bytes
- After: ~20 bytes targeted zeroing + 0 bytes copy = ~20 bytes

## Investigation Notes

### DCT Coefficient Parity (2026-01-22) - VERIFIED

**Rust now supports both 2-table and 3-table modes via `separate_chroma_tables` flag.**

The two C++ functions use different quant table configurations:
- `jpeg_set_quality()`: 2 chroma tables (`add_two_chroma_tables=false`) - Cr matrix used for both Cb and Cr
- `jpegli_set_distance()`: 3 tables (`add_two_chroma_tables=true`) - separate Y, Cb, Cr tables

Rust configuration:
- `separate_chroma_tables(true)` (default): 3 tables, matches `jpegli_set_distance()`
- `separate_chroma_tables(false)`: 2 tables, matches `jpeg_set_quality()`

**Root cause of ±1 coefficient differences:**
- Different DCT SIMD implementations (Highway vs wide crate) produce slightly different floating-point intermediates
- **NOT** rounding mode - tested round-to-even in Rust, made no difference (2026-01-22)
- **NOT** DCT scaling - Rust uses 1/8 scaling, C++ uses 1/64, but this is compensated in quantization (`quant_mul = 8/quant` in C++)
- Source is SIMD float precision in intermediate DCT butterfly operations

**Tools added:**
- `jpegli_set_distance` FFI binding in `jpegli-internals-sys`
- `EncoderConfig::distance(f32)` in `jpegli-bench-utils` for distance-based encoding
- `cargo run --release --example compare_dct_coefficients` - DCT coefficient comparison
- `cargo run --release --example coeff_synthetic_test` - Solid color coefficient test

**Verified parity (distance=1.0, kodak/1.png 768x512):**
```
Quantization Tables:
  Table 0: DC rust=3, cpp=3 (MATCH)
  Table 1: DC rust=2, cpp=2 (MATCH)
  Table 2: DC rust=3, cpp=3 (MATCH)

Rust JPEG: 130,270 bytes
C++  JPEG: 130,506 bytes
Size diff: -0.18%
```

**Coefficient differences are normal ±1 rounding, not systematic bugs:**
- 80% of Y blocks differ by ±1 in 1-3 AC coefficients
- Max difference: 6 (single outlier block)
- Cb/Cr: 27-37% blocks differ, all by ±1

### AQ Map Comparison (2026-01-22) - RESOLVED

**AQ maps are 100% identical between Rust and C++ FFI when using matching quant tables.**

Previous analysis using `jpeg_set_quality()` showed spurious differences because
the quant tables didn't match (2 vs 3 tables). With `jpegli_set_distance()`, AQ maps match exactly.

**Verified AQ parity** (FFI mode with distance, flower.png 2268x1512):
```
Mean difference:     0.000000
Mean |difference|:   0.000000
All 53,676 blocks:   0% difference
```

**Remaining ~0.2% size difference source:**
- AQ is identical → same quantization tables
- Difference is in DCT rounding (±1 coefficient differences)
- Both use optimized Huffman (verified: `optimize_coding = 1` in FFI)
- Butteraugli quality within 1% (essentially identical perceptual quality)

## Known Bugs

1. **XYB quality gap** - ~5 SSIMULACRA2 points behind C++ in XYB mode. Root cause TBD.

### Fixed Bugs (historical reference)

- **Debug env var in hot loop** - `entropy/encoder.rs`: Removed `std::env::var()` call from hot path (was 12% overhead)
- **Eager error evaluation** - `entropy/encoder.rs`: Changed `ok_or()` to `ok_or_else()` (13% speedup)
- **Progressive XYB decode** - `decode/mod.rs`: Handle `EndOfScanData` gracefully for non-standard component IDs
- **1-pixel partial MCU edge** - `fast_yuv.rs`, `streaming.rs`: Added edge replication for width ≡ 1 (mod 8)
- **HF modulation index wrap** - `quant/aq/simd.rs`: Added bounds check for rightmost partial blocks

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

### Benchmark Output Rules (CRITICAL)

**NEVER re-run benchmarks just to parse output differently.** Criterion saves structured JSON:

```bash
# Results are stored in target/criterion/<group>/<bench>/new/estimates.json
# Extract results from JSON instead of re-running:
cat target/criterion/decode_compare/jpegli-baseline/512x512/new/estimates.json | jq '.mean.point_estimate'

# If you need terminal output, pipe to a file on FIRST run:
cargo bench --bench decode_compare 2>&1 | tee /tmp/bench-output.txt
```

**JSON structure:** `estimates.json` contains `mean`, `median`, `slope`, `std_dev` with `point_estimate` (nanoseconds) and `confidence_interval`.

**Example extraction:**
```bash
for d in target/criterion/decode_compare/*/512x512/new; do
  name=$(basename $(dirname $(dirname $d)))
  mean=$(cat "$d/estimates.json" | jq -r '.mean.point_estimate')
  echo "$name: $(echo "scale=2; $mean/1000" | bc) µs"
done
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
ultrahdr = ["dep:ultrahdr-core", "decoder"]  # UltraHDR HDR gain map support
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
| `jpegli-rs/src/encode/mod.rs` | Encoder pipeline |
| `jpegli-rs/src/decode/mod.rs` | Decoder pipeline |
| `jpegli-rs/src/color/xyb.rs` | XYB color conversion |
| `jpegli-rs/src/quant/aq/mod.rs` | Adaptive quantization |
| `jpegli-rs/src/huffman/mod.rs` | Huffman encoding |

## External Dependencies

- **Test images**: `internal/jpegli-cpp/testdata/`
- **Codec corpus**: `~/work/codec-eval/codec-corpus/` (Kodak, CID22)
- **C++ reference**: `internal/jpegli-cpp/` submodule

## Detailed Documentation

- **`jpegli-rs/README.md` - API Reference** (encoder/decoder usage with examples)
- `jpegli-rs/examples/README.md` - Examples and debugging tools
- `jpegli-rs/docs/ADAPTIVE_QUANTIZATION.md` - AQ algorithm details
- `jpegli-rs/docs/API_DESIGN.md` - Full API surface and proposed enhancements
- `docs/SECURITY.md` - Security considerations
