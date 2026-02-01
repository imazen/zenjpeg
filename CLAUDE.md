# zenjpeg Project Guide

Pure Rust port of Google's jpegli JPEG encoder/decoder from the JPEG XL project.

## API Stability Rules (CRITICAL)

**DO NOT change the public API without explicit approval:**

1. **No re-exports at crate root** - Types stay in their modules (`encoder::EncoderConfig`, not `EncoderConfig`)
2. **No new public types/functions** without approval
3. **No changes to existing function signatures**
4. **Doc links use full paths** - `[`encoder::EncoderConfig`]` not `[`EncoderConfig`]`

## Pixel Data API Rules (CRITICAL - NON-NEGOTIABLE)

**Every API that touches pixel data MUST follow these rules. NO EXCEPTIONS.**

### Type-Safe Pixel Formats (MANDATORY)

**NEVER use raw `&[u8]` or `&[f32]` for pixel data without explicit format information.**

Use the `rgb` crate types or equivalent to encode channel count, order, and bit depth.

**STRIDE IS ALWAYS REQUIRED. NO EXCEPTIONS.**

Use `imgref::ImgRef`/`ImgRefMut` (preferred) or explicit stride parameter (in pixels, not bytes):

```rust
// BEST - imgref handles stride, type encodes format
fn push_rows(&mut self, rows: ImgRef<'_, rgb::RGB<u8>>) -> Result<()>;
fn push_rows(&mut self, rows: ImgRef<'_, rgb::RGB<u16>>) -> Result<()>;
fn push_rows(&mut self, rows: ImgRef<'_, rgb::RGBA<f32>>) -> Result<()>;
fn read_rows(&mut self, out: ImgRefMut<'_, rgb::RGB<u8>>) -> Result<usize>;

// GOOD - Explicit stride in pixels (not bytes!)
fn push_rows(&mut self, data: &[rgb::RGB<u16>], width: usize, stride_pixels: usize, count: usize) -> Result<()>;
fn read_rows(&mut self, out: &mut [rgb::RGB<u16>], width: usize, stride_pixels: usize, count: usize) -> Result<usize>;

// BAD - No stride, will break on padded buffers
fn push_rows(&mut self, data: &[rgb::RGB<u8>], count: usize) -> Result<()>;

// TERRIBLE - No format, no stride, completely ambiguous
fn push_rows(&mut self, data: &[u8], width: usize) -> Result<()>;
```

### Precision Requirements (MANDATORY)

**zenjpeg is a professional codec. 8-bit internal precision is UNACCEPTABLE.**

- **Internal processing**: 16-bit minimum, 32-bit float preferred
- **DCT/quantization**: f32 or i32, NEVER i16 for intermediates
- **Color conversion**: f32 linear light, NEVER gamma-encoded arithmetic
- **API input/output**: Support u8, u16, f32 - but DOCUMENT the precision implications

```rust
// GOOD - High precision internal, flexible external, stride via imgref
pub fn push_rows(&mut self, rows: ImgRef<'_, rgb::RGB<u8>>) -> Result<()>;   // Converts to f32 internally
pub fn push_rows(&mut self, rows: ImgRef<'_, rgb::RGB<u16>>) -> Result<()>; // Converts to f32 internally
pub fn push_rows(&mut self, rows: ImgRef<'_, rgb::RGB<f32>>) -> Result<()>; // Native precision

// GOOD - Explicit stride alternative
pub fn push_rows(&mut self, data: &[rgb::RGB<u16>], width: usize, stride: usize, count: usize) -> Result<()>;

// BAD - No stride, precision loss hidden
pub fn push_rows(&mut self, rows: &[u8]) -> Result<()>; // What happens to my 16-bit data? What's the stride?
```

### Color Space Awareness (MANDATORY)

**EVERY pixel-touching function must consider color space. Even if "agnostic", document it.**

Think about:
- **Working space**: What color space are we computing in? Linear? Gamma?
- **ICC profile**: Is there an embedded profile? Do we honor it?
- **CICP**: Are we writing/reading CICP tags? (primaries, transfer, matrix)
- **Primaries**: sRGB? P3? Rec.2020? BT.601 vs BT.709 YCbCr matrix?
- **Transfer function**: Linear? sRGB? PQ? HLG?

```rust
// GOOD - Color space explicit, stride via imgref
pub fn push_rows(&mut self, rows: ImgRef<'_, rgb::RGB<f32>>, colorspace: ColorSpace) -> Result<()>;

// GOOD - Explicit stride + colorspace
pub fn push_rows(&mut self, data: &[rgb::RGB<f32>], width: usize, stride: usize, count: usize, colorspace: ColorSpace) -> Result<()>;

// ACCEPTABLE - Documented assumption, still has stride
/// Assumes sRGB primaries with sRGB transfer function.
/// For wide-gamut input, convert to sRGB first or use `push_rows_with_colorspace`.
pub fn push_rows_srgb(&mut self, rows: ImgRef<'_, rgb::RGB<u8>>) -> Result<()>;

// BAD - No stride, no colorspace, no type safety
pub fn push_rows(&mut self, rows: &[u8]) -> Result<()>;
```

### Memory Architecture (MANDATORY)

**WHOLE-IMAGE BUFFERING IS FORBIDDEN. STREAMING ONLY.**

1. **No one-shot APIs** - All encode/decode must be streaming row-by-row
2. **No `Vec<Vec<T>>`** - Use flat buffers with stride, or ring buffers
3. **Borrow, don't clone** - Take `&[T]` not `Vec<T>`, return into caller's buffer
4. **Fallible allocation** - Use `try_reserve()`, return `Result` on OOM
5. **Buffer pool friendly** - Caller provides target buffers, we write into them
6. **Stride always explicit** - Via `imgref` or parameter

```rust
// GOOD - Streaming, borrows, stride via imgref, writes to caller's buffer
pub fn push_rows(&mut self, rows: ImgRef<'_, rgb::RGB<u16>>) -> Result<()>;
pub fn read_rows(&mut self, out: ImgRefMut<'_, rgb::RGB<u16>>) -> Result<usize>;

// GOOD - Explicit stride, caller's buffer
pub fn push_rows(&mut self, data: &[rgb::RGB<u16>], width: usize, stride: usize, count: usize) -> Result<()>;
pub fn read_rows(&mut self, out: &mut [rgb::RGB<u16>], width: usize, stride: usize, max_rows: usize) -> Result<usize>;

// GOOD - Caller provides output buffer, no internal allocation
pub fn finish_into(self, output: &mut Vec<u8>) -> Result<()>;

// BAD - Whole image, allocates, clones, no stride
pub fn encode(image: Vec<Vec<u8>>) -> Vec<u8>;

// BAD - Hidden allocation, not fallible, no stride
pub fn decode(&self) -> Vec<u8>;  // What if it's 100MP? OOM panic!
```

### Ring Buffer / Pool Architecture

Design for buffer reuse. Caller owns all buffers, encoder/decoder writes into them:

```rust
// GOOD - Caller owns input buffer with explicit stride
let mut encoder = StreamingEncoder::new(width, height, config)?;
let stride = (width + 15) & !15;  // Align to 16 pixels for SIMD
let mut row_buf = vec![rgb::RGB::<u16>::default(); stride * batch_size];

for chunk in source.chunks(batch_size) {
    // Copy into caller's buffer (or read directly if source is strided)
    for (i, row) in chunk.iter().enumerate() {
        row_buf[i * stride..(i * stride + width)].copy_from_slice(row);
    }
    encoder.push_rows(&row_buf, width, stride, chunk.len())?;
}

// GOOD - Caller owns output buffer
let mut output = Vec::new();
output.try_reserve(estimated_size)?;  // Fallible!
encoder.finish_into(&mut output)?;

// GOOD - Decoder writes into caller's buffer with stride
let mut decoder = StreamingDecoder::new(&jpeg_data)?;
let out_stride = (decoder.width() + 15) & !15;
let mut out_buf = vec![rgb::RGB::<u8>::default(); out_stride * batch_size];

while !decoder.is_finished() {
    let rows_read = decoder.read_rows(&mut out_buf, decoder.width(), out_stride, batch_size)?;
    process_rows(&out_buf[..rows_read * out_stride]);
}
```

### Summary: The Non-Negotiables

| Rule | Requirement |
|------|-------------|
| **Pixel format** | Type-safe (`rgb::RGB<T>`, `rgb::RGBA<T>`) - NEVER raw `&[u8]` |
| **Stride** | ALWAYS via `imgref` or explicit parameter (pixels, not bytes) - NEVER omitted |
| **Precision** | 16-32 bit internal - NEVER 8-bit arithmetic on pixels |
| **Color space** | Explicit parameter or documented assumption - NEVER silently ignored |
| **Streaming** | Row-by-row only - NEVER whole-image buffering |
| **Allocation** | Fallible (`try_reserve`) - NEVER panic on OOM |
| **Ownership** | Borrow input (`&[T]`), write to caller's output (`&mut [T]`) - NEVER clone |
| **Target buffers** | Caller provides output buffers - NEVER allocate internally when avoidable |

**If you're about to write an API that omits stride, STOP. Add stride. Always.**

**If you're about to write an API that allocates output, STOP. Take a target buffer.**

**If you're about to write an API with raw bytes, STOP. Use `rgb::RGB<T>` or similar.**

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
cargo test --release -p zenjpeg --test comprehensive_cpp_comparison -- --nocapture --ignored
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
cargo test --release -p zenjpeg -- comparison --nocapture --ignored

# Corpus-based comparison (CID22 images)
cargo test --release -p zenjpeg --test corpus_cpp_comparison -- --nocapture --ignored

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
zenjpeg/
├── zenjpeg/           # Main library crate
│   ├── src/             # Encoder, decoder, color conversion
│   ├── examples/        # Debugging tools (see examples/README.md)
│   ├── tests/           # Integration tests
│   └── benches/         # Criterion benchmarks
├── zenjpeg-bench-utils/  # Shared utilities for benchmarks/examples
├── internal/jpegli-cpp/ # C++ jpegli submodule (for parity testing)
└── docs/                # Additional documentation
```

## Examples & Debugging Tools

See **`zenjpeg/examples/README.md`** for complete documentation.

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

### XYB Debugging (Size Gap Investigation)

XYB produces 5-11% larger files than C++ jpegli at equivalent quality (quality is identical):

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

Run with: `cargo flamegraph --release -p zenjpeg --example flamegraph_profile -- 4k`
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

Run with: `cargo bench -p zenjpeg --bench cpp_comparison`

### Summary (2026-02-01)

Rust is **~20% slower** than C++ jpegli (1.2x median, range 1.05x-1.43x per criterion benchmarks).

**Criterion benchmark results (512x512 Q90):**

| Config | Rust | C++ | Ratio |
|--------|------|-----|-------|
| base-420 | 1.37ms | 0.96ms | 1.43x |
| base-444 | 1.85ms | 1.59ms | 1.17x |
| prog-420 | 2.39ms | 1.92ms | 1.25x |
| prog-444 | 3.51ms | 3.34ms | 1.05x |

**Quality parity (comprehensive test, 10 images × 50 quality levels):**

| Metric | Min | Max | Mean |
|--------|-----|-----|------|
| Size Δ% | +0.2% | +1.6% | **+0.63%** |
| DSSIM Δ% | -0.9% | +1.5% | **+0.41%** |
| Butteraugli Δ% | -2.1% | +2.2% | **+0.19%** |

Quality is effectively identical (mean <0.5%); 50/50 quality levels within 5%.

Note: Previous measurements showed 1.4-1.6x slower; improvements came from SIMD sorting
network optimization and allocation reduction.

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

**Solution** (`zenjpeg/src/quant/aq/streaming.rs`):
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

**Benchmark:** `cargo bench -p zenjpeg --bench aq_simd --features "archmage-simd,test-utils"`

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

**Benchmark results** (8x8 f32 transpose, `zenjpeg/examples/autovec_transpose.rs`):

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

**Files:** `zenjpeg/examples/autovec_transpose.rs`

### Autovec vs Wide Crate AQ Benchmark (2026-01-21)

**Benchmark:** `cargo run --release -p zenjpeg --example bench_autovec_aq`

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

**Files:** `zenjpeg/src/quant/aq/autovec.rs`, `zenjpeg/examples/bench_autovec_aq.rs`

## WASM SIMD128 Performance (2026-01-27)

Run with: `just wasm-bench` (or `just wasm-bench-simd` / `just wasm-bench-scalar`)

### Summary

WASM SIMD128 provides **1.6-1.7x encode speedup** and **1.5-2.0x decode speedup** over scalar.
The `wide` crate's f32x4 maps to WASM v128 operations.

**SIMD128 vs Scalar:**

| Size | Encode SIMD | Encode Scalar | **Encode Speedup** | Decode SIMD | Decode Scalar | **Decode Speedup** |
|------|-------------|---------------|--------------------|-------------|---------------|--------------------|
| 64x64 | 40.74 MP/s | 26.13 MP/s | **1.56x** | 103.74 MP/s | 60.57 MP/s | **1.71x** |
| 256x256 | 50.74 MP/s | 30.15 MP/s | **1.68x** | 135.81 MP/s | 67.84 MP/s | **2.00x** |
| 512x512 | 50.88 MP/s | 29.56 MP/s | **1.72x** | 134.07 MP/s | 66.40 MP/s | **2.02x** |
| 1024x1024 | 47.98 MP/s | 29.74 MP/s | **1.61x** | 144.32 MP/s | 92.96 MP/s | **1.55x** |

**Key findings:**
- `wide` crate f32x4 → WASM simd128 v128 mapping works well
- f32x8 becomes two v128 operations (still faster than scalar)
- No runtime CPU feature detection in WASM, so `#[multiversed]` doesn't dispatch
- Build with `RUSTFLAGS="-C target-feature=+simd128"` to enable
- wasmtime needs `--wasm simd` flag to enable SIMD execution

**Build commands:**
```sh
# SIMD128 enabled
CARGO_TARGET_WASM32_WASIP1_RUNNER="wasmtime --wasm simd" \
RUSTFLAGS="-C target-feature=+simd128" \
cargo run --release -p zenjpeg --example wasm_bench \
    --target wasm32-wasip1 --no-default-features --features "std,decoder"

# Scalar (no SIMD)
CARGO_TARGET_WASM32_WASIP1_RUNNER="wasmtime" \
cargo run --release -p zenjpeg --example wasm_bench \
    --target wasm32-wasip1 --no-default-features --features "std,decoder"
```

**Files:** `zenjpeg/examples/wasm_bench.rs`

### WASM SIMD Intrinsics Investigation (2026-01-27)

Investigated whether explicit `core::arch::wasm32` intrinsics could outperform the `wide` crate.

**Findings:**

1. **`wide` crate already uses v128 intrinsics** - When compiled with `+simd128`, the `wide`
   crate's f32x4 wraps v128 directly and uses `f32x4_add`, `f32x4_mul`, etc.

2. **f32x8::transpose has scalar fallback** - On non-AVX targets (including WASM), the
   `wide` crate's `f32x8::transpose` uses a scalar fallback that extracts individual elements.
   Explicit WASM v128 shuffle intrinsics are only 7% faster.

3. **Transpose is only ~20% of DCT time** - Combined with the 7% improvement, explicit
   WASM intrinsics would only improve DCT by ~1.4%. Not worth the complexity.

4. **Archmage is x86_64 only** - The `archmage` crate uses `core::arch::x86_64` intrinsics
   and cannot be used on WASM. No WASM equivalent exists.

**Benchmark results (100K iterations):**

| Operation | SIMD128 | Scalar | Notes |
|-----------|---------|--------|-------|
| 8x8 Transpose (wide) | 12.71 ns | 7.95 ns | Scalar faster! |
| 8x8 Transpose (intrinsics) | 3.00 ns | N/A | 4.2x faster than wide |
| 1D DCT (8 vals) | 0.77 ns | 0.61 ns | Scalar faster |
| Full 8x8 DCT | 125.65 ns | 115.56 ns | Scalar faster |

**Surprising result:** Isolated DCT benchmarks show scalar slightly faster than SIMD.
However, full encoder shows SIMD 1.6x faster. The difference is likely:
- Full encoder benefits from SIMD in quantization, entropy coding, color conversion
- Isolated DCT benchmark has different memory/cache patterns
- wasmtime's SIMD overhead may be amortized over larger operations

**UPDATE: archmage + magetypes comparison (2026-01-27):**

Both archmage 0.2.1 and magetypes 0.1.0 have full WASM SIMD128 support:
- `archmage::Simd128Token` - capability token for WASM
- `magetypes::simd::f32x4` - token-gated SIMD types using v128

| Operation | wide | magetypes | Improvement |
|-----------|------|-----------|-------------|
| Arithmetic (add/sub/mul) | 0.59 ns | 0.52 ns | **14% faster** |
| 4x4 transpose | N/A | 0.75 ns | Native v128 shuffle |
| log2 | 0.37 ns | 0.37 ns | Equal |

**Recommendation:** For WASM-critical paths, consider magetypes over wide for 10-15%
arithmetic improvements. For portable code, wide remains the simpler choice.

**Files:** `zenjpeg/examples/wasm_simd_transpose.rs`, `zenjpeg/examples/wasm_dct_bench.rs`,
`zenjpeg/examples/wasm_magetypes_bench.rs`

## Decoder Performance Gap (2026-01-22)

Run with: `cargo bench -p zenjpeg --bench decode_compare`

### Summary

zenjpeg decoder is **4-5x slower** than zune-jpeg for baseline JPEG and **2-4x slower** for progressive JPEG.

**Baseline JPEG (sequential Huffman):**

| Size | zune-jpeg | zenjpeg | Ratio |
|------|-----------|-----------|-------|
| 256x256 | 82 µs | 337 µs | 4.1x |
| 512x512 | 285 µs | 1.31 ms | 4.6x |
| 1024x1024 | 946 µs | 4.56 ms | 4.8x |
| 2048x2048 | 3.80 ms | 19.2 ms | 5.0x |

**Progressive JPEG:**

| Size | zune-jpeg | zenjpeg | Ratio |
|------|-----------|-----------|-------|
| 256x256 | 240 µs | 519 µs | 2.2x |
| 512x512 | 860 µs | 2.03 ms | 2.4x |
| 1024x1024 | 2.46 ms | 10.0 ms | 4.1x |
| 2048x2048 | 9.16 ms | 32.0 ms | 3.5x |

**Throughput comparison (2048x2048):**
- zune-jpeg baseline: 1104 MP/s
- zenjpeg baseline: 219 MP/s
- zune-jpeg progressive: 458 MP/s
- zenjpeg progressive: 131 MP/s

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

5. **Streaming encoder overhead allocation issue (2026-01-26)**
   - `StreamingEncoder` pre-allocates storage for ALL DCT blocks upfront (`y_blocks`, `cb_blocks`, `cr_blocks`)
   - Location: `zenjpeg/src/encode/strip/mod.rs:459-466`
   - For 4000×3000 image at 4:2:0:
     - `y_blocks`: 187,500 blocks × 128 bytes = **24 MB**
     - `cb_blocks`: 47,000 blocks × 128 bytes = **6 MB**
     - `cr_blocks`: 47,000 blocks × 128 bytes = **6 MB**
     - Total: **~36 MB** just for block storage
   - Additional buffers (`y_strip`, `cb_strip`, `cr_strip`, `all_aq_strengths`) add more
   - Heaptrack verified: 68 MB peak for 4000×3000 UltraHDR encode vs theoretical ~4 MB for true streaming
   - Root cause: Architecture buffers ALL blocks before encoding, even with `optimize_huffman=false`
   - "Streaming" only streams INPUT rows, not OUTPUT blocks
   - Baseline mode with fixed Huffman tables COULD support true streaming (write blocks immediately)
   - Current impl: `streaming.rs:1574-1650` always reads from buffered `strip_output.*_blocks`
   - Fix: Add immediate-write mode for baseline + fixed Huffman (no two-pass needed)

### Callgrind Analysis (2026-01-22)

Run: `valgrind --tool=callgrind ./target/release/examples/valgrind_decode jpegli 512`

**Totals (512x512 baseline JPEG):**

| Metric | zenjpeg | zune-jpeg | Ratio |
|--------|-----------|-----------|-------|
| Instructions | 60.3M | 34.6M | **1.74x** |
| Data refs | 16.4M | 9.5M | 1.72x |
| Branches | 9.2M | 4.0M | 2.3x |
| D1 miss rate | 2.3% | 3.7% | better |
| Branch mispredict | 0.8% | 1.5% | better |

**Decode-specific function comparison:**

| Function | zenjpeg | zune-jpeg | Notes |
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

**Files:** `zenjpeg/src/quant/aq/simd.rs:1377` (massive version, unused),
`zenjpeg/src/quant/aq/streaming.rs:631` (original scalar, in use)

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

**Files:** `zenjpeg/src/encode/mage_simd.rs:600-775` (kept for reference, not used in encoder)

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
- `EncoderConfig::distance(f32)` in `zenjpeg-bench-utils` for distance-based encoding
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

### Visual Diff Interpretation (2026-01-31)

**Repro commands:**
```bash
just xyb-diff                    # XYB mode: C++ | Rust | ΔR×10 | ΔG×10 | ΔB×10
just ycbcr-diff                  # YCbCr mode: same 5-panel layout
just xyb-diff ~/path/to/img.png  # Custom image
```

**XYB mode visual patterns (kodak/1.png q90):**
- **C++ & Rust panels**: Visually identical, slight green tint (normal for this image)
- **ΔR (3rd panel)**: Block patterns with U/M shapes at 8×8 boundaries, sometimes strong
  - Indicates DCT coefficient quantization differences
  - Pattern follows block grid = coefficient rounding at block level
- **ΔG (4th panel)**: Uniform even noise across image
  - Luminance (Y) channel has consistent small differences
  - No block structure = no systematic quantization bias
- **ΔB (5th panel)**: Similar to ΔR but less intense
  - Was severely corrupted before B-channel fix (mean error ~51)
  - Now shows same block-pattern differences as R channel

**XYB numeric results (kodak/1.png 768×512 q90, after AQ channel fix):**
```
Rust: 153752 bytes, C++: 141450 bytes (+8.7%)
Mean |diff|: R=0.237, G=1.017, B=0.104
```
Note: Before fix was +10.8% size, R=0.310, G=1.300, B=0.200.

**YCbCr mode visual patterns:**
- **All diff panels**: Uniform noise pattern, no obvious block structure
- More even distribution of differences across all channels

**YCbCr numeric results (kodak/1.png 768×512 q90):**
```
Rust: 130270 bytes, C++: 143695 bytes (-9.3%)
Mean |diff|: R=2.056, G=1.877, B=2.102
Max  |diff|: R=25, G=17, B=18
```

**Interpretation:**
- XYB: Block patterns in R/B suggest coefficient quantization differences at block boundaries
- YCbCr: Uniform noise suggests consistent but different rounding strategy
- YCbCr produces smaller files but higher per-pixel differences
- XYB produces larger files but lower per-pixel differences (except G channel)

**TODO: Investigate block-boundary coefficient patterns in XYB mode**
- The U/M patterns in ΔR suggest systematic differences in how edge blocks are quantized
- May be related to AQ strength interpolation or zero-bias calculation
- Files: `zenjpeg/src/quant/aq/simd.rs`, `zenjpeg/src/encode/strip/mod.rs:quantize_pending_imcu`

## Known Bugs

(None currently tracked)

### Fixed Bugs (historical reference)

- **XYB file size gap (2026-02-01)** - XYB baseline was 2-3% larger than C++, but this
  was due to Rust getting 2x more progressive savings (5.7-7.3% vs C++'s 3.1-3.6%).
  With progressive mode, Rust XYB matches or beats C++ (-0.3% to -4.3% smaller).
  Resolution: Progressive mode is now recommended for XYB.
  Files: `encode/strip/mod.rs`, `quant/aq/streaming.rs`

- **XYB AQ v_samp mismatch (2026-01-31)** - AQ was initialized with v_samp=1 (from S444
  subsampling), but XYB JPEG uses R:2×2 G:2×2 B:1×1 (max_v_samp_factor=2). This caused
  the AQ to treat each 8-row strip as a full iMCU instead of 16-row iMCUs, producing
  overly conservative quantization. Also affected pending DCT block buffer sizing.
  Fixed in `encode/strip/mod.rs`: both `v_samp` calculations now check `use_xyb`.
  Impact: Size diff 8-18% → 5-11% (3-7pp improvement).
- **XYB AQ using wrong channel (2026-01-31)** - AQ was computed on X channel instead of Y.
  C++ uses `y_channel = jpeg_color_space == JCS_RGB ? 1 : 0`, meaning channel 1 (Y) for XYB.
  Fixed by using `cb_strip` instead of `y_strip` when `use_xyb=true`.
  Impact: Size diff 10.8% → 8.7% (~2pp improvement), all color diffs reduced.
  Files: `encode/strip/mod.rs:779-795`
- **XYB B-channel encoding corruption (2026-01-31)** - B channel had ~51 mean error vs ~0.1 for R/G.
  Root cause: `StripProcessor` created with `use_xyb=false`, then `set_xyb_mode(true)` called,
  but this didn't recalculate B-channel dimensions. Also `c_blocks_h/v` used for B instead of
  `b_blocks_h/v`. Fixed by using `with_xyb()` constructor and adding `padded_b_width`,
  `b_blocks_h`, `b_blocks_v` fields. Files: `streaming.rs`, `strip/mod.rs`, `strip/convert.rs`.
- **Debug env var in hot loop** - `entropy/encoder.rs`: Removed `std::env::var()` call from hot path (was 12% overhead)
- **Eager error evaluation** - `entropy/encoder.rs`: Changed `ok_or()` to `ok_or_else()` (13% speedup)
- **Progressive XYB decode** - `decode/mod.rs`: Handle `EndOfScanData` gracefully for non-standard component IDs
- **1-pixel partial MCU edge** - `fast_yuv.rs`, `streaming.rs`: Added edge replication for width ≡ 1 (mod 8)
- **HF modulation index wrap** - `quant/aq/simd.rs`: Added bounds check for rightmost partial blocks

## Planned Features / TODO

### DONE: LayoutParams Immutable Substruct (completed 2026-01-31)

Eliminated derived-state sync bugs by creating `LayoutParams` (`encode/layout.rs`):
- [x] `LayoutParams` struct with all geometry derived from (subsampling, use_xyb, width, height)
- [x] `StripProcessor` stores `layout: LayoutParams`, removed 12 replaced fields
- [x] `StreamingAQ::new(&LayoutParams, y_quant_01)` — no independent geometry computation
- [x] `estimate_memory_usage/ceiling` use `LayoutParams` (fixes XYB estimates)
- [x] Removed `set_xyb_mode()` (dead code) and `set_strip_stride()` (absorbed into layout)
- [x] Deduplicated chroma/B-channel dimensions: 8 locations now read from `LayoutParams` instead of recomputing. Added `b_strip_height` field, `c_strip_height_for()` and `b_strip_height_for()` helpers for partial final strips.
- [ ] `serialize.rs::write_frame_header_xyb_ex()` still hardcodes 0x22/0x11 (low priority, always correct for XYB)

### DONE: Encoder Invariants & Readability Refactor (completed 2026-02-01)

- [x] Added `total_mcus` (and `h_samp`, `mcu_cols`, `mcu_rows`) to `LayoutParams`
- [x] Introduced `PendingBuffers` struct (replaces 6 `Vec` fields + `pending_current: usize`)
- [x] Made `QuantContext` and `StreamingAQ` non-optional in `StripProcessor`
- [x] Deduplicated chroma quantization via `quantize_chroma_blocks()` free function
- [x] Added `HuffmanStrategy` enum (replaces `optimize_huffman: bool` + `custom_huffman_tables: Option`)
- [x] Extracted `color_convert_strip()` dispatch helper from `process_strip`

### Encoder Hardening TODO

1. **[DONE] Rename `_h`/`_v` block fields to `_w`/`_h` in LayoutParams** — `y_blocks_h` means horizontal
   (width) but `blocks_h` means height. Rename to `y_blocks_w`/`y_blocks_h` etc. so `_w` always
   means width and `_h` always means height. Files: `encode/layout.rs`, all consumers.

2. **[DONE] Add `aq_input_strip()` accessor** — `if use_xyb { &cb_strip } else { &y_strip }` appears 3 times
   (`strip/mod.rs:784, 873, 958`). Extract to single method documenting that `cb_strip` is the
   Y channel in XYB mode. Files: `encode/strip/mod.rs`.

3. **[DONE] Replace `try_into().unwrap()` with debug_assert** — `strip/mod.rs:154` panics if
   `padded_width` invariant breaks. Add bounds assertion. Files: `encode/strip/mod.rs`.

4. **[DONE] Deduplicate 3 process_strip variants** — Extracted `process_strip_common(strip_y,
   actual_strip_height, need_chroma_downsample)` from the shared AQ → downsample → quantize →
   DCT tail. Each public method is now preamble → padding → `process_strip_common()`.
   Files: `encode/strip/mod.rs`.

5. **[DONE] Split `build_jpeg_sequential_into()` into XYB and YCbCr sub-functions** — Split into
   `encode_sequential_xyb()` and `encode_sequential_ycbcr()` with a thin orchestrator.
   Prevents wrong-mode function calls; XYB's lack of frequency collection is explicit in types.
   Files: `encode/streaming.rs`.

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

### CLI Parameter Support

- [x] Quality via `-q` / `--quality` (0-100, maps to quality internally)
- [x] Distance via `-d` / `--distance` (Butteraugli distance, lower = better)

**C++ cjpegli behavior differences based on quality vs distance:**

| Setting | C++ `-q` (quality) | C++ `-d` (distance) |
|---------|-------------------|---------------------|
| Quant tables | 2 tables (Y, shared Cb/Cr) | 3 tables (Y, Cb, Cr) |
| Quant values | Always clamped to 255 | Always clamped to 255 |
| Table selection | `jpeg_set_quality()` | `jpegli_set_distance()` |

**zenjpeg behavior:**
- Default: 3 separate quant tables (matches C++ distance mode)
- Use `separate_chroma_tables(false)` for 2-table mode matching C++ quality mode
- Quant values: Can exceed 255 when `allow_16bit_quant_tables = true` (default)

### 16-bit Quantization Tables (IMPORTANT)

**C++ jpegli API supports 16-bit but CLI forces baseline:**

```cpp
// API signature - force_baseline controls 8-bit vs 16-bit
void jpegli_set_distance(j_compress_ptr cinfo, float distance, boolean force_baseline);
void jpegli_set_quality(j_compress_ptr cinfo, int quality, boolean force_baseline);

// Quant value clamping in quant.cc:585
int quant_max = m->force_baseline ? 255 : 32767U;
```

The `force_baseline` parameter:
- `TRUE` (baseline): Clamp quant values to 255, use 8-bit DQT (SOF0)
- `FALSE` (extended): Allow values up to 32767, use 16-bit DQT when needed (SOF1)

**C++ cjpegli CLI always uses baseline:**
- `jxl::extras::EncodeJpeg()` hardcodes `TRUE` (lib/extras/enc/jpegli.cc:470-472)
- No CLI flag exists to change this
- Default in `jpegli_create_compress()` is `force_baseline = true`

**zenjpeg behavior:**
- `allow_16bit_quant_tables = true` (default): Matches `force_baseline = FALSE`
- `allow_16bit_quant_tables = false`: Matches `force_baseline = TRUE` (cjpegli CLI)
- Precision is auto-selected per-table: 8-bit if max ≤ 255, 16-bit if max > 255
- We DO automatically use 8-bit when no coefficient exceeds 255

**Quality threshold for 16-bit tables:**

The jpegli quant formulas produce chroma values that exceed 255 below Q87.
This is **quality-dependent, not image-dependent** - tested on Kodak corpus (24 images):

| Quality | Max Chroma | Tables |
|---------|------------|--------|
| Q90-100 | ≤200 | 8-bit (both Rust and C++) |
| Q87-89 | 218-254 | 8-bit (both Rust and C++) |
| Q86 | 272 | 16-bit (Rust) or clamped (C++) |
| Q70 | 560 | 16-bit (Rust) or clamped (C++) |
| Q50 | 919 | 16-bit (Rust) or clamped (C++) |

**Practical impact:**
- Default Q90: No difference between Rust and C++
- Web-quality Q70-Q85: Rust uses 16-bit, C++ CLI clamps to 255
- Low-quality Q50: Rust preserves precision, C++ CLI loses ~70% of chroma range

Note: The C++ API supports 16-bit via `force_baseline=FALSE`, but cjpegli CLI
doesn't expose this. Our `allow_16bit_quant_tables=true` default matches the
C++ API capability, not the CLI behavior.

**Why clamping doesn't hurt quality:**

The quant positions that exceed 255 are high-frequency chroma coefficients
(bottom-right corner of DCT matrix). These coefficients quantize to zero
regardless of whether dividing by 255 or 500+.

Tested on Kodak corpus at Q5-Q86:
- **Scan data is byte-for-byte identical** between 16-bit and clamped versions
- **SSIMULACRA2 delta: 0.000** at all quality levels (Q5, Q10, Q20, Q50...)
- Only difference is DQT marker overhead (~128 bytes)

C++ chose baseline compatibility with zero quality impact. Consider defaulting
`allow_16bit_quant_tables=false` to match, since 16-bit provides no benefit.

To match C++ cjpegli CLI, use `.allow_16bit_quant_tables(false)`.

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

## Color Space Architecture (CRITICAL)

### The Fundamental Truth

**Color space is NOT optional metadata. It defines the meaning of pixel values.**

A pixel value of `(255, 128, 0)` means COMPLETELY DIFFERENT COLORS in:
- sRGB (a specific orange)
- Display P3 (a more saturated orange)
- Rec.2020 (an even more saturated orange)
- Linear sRGB (a MUCH brighter orange, wrong gamma)

**If your code ignores color space, it is BROKEN. Period.**

### RGB→YCbCr Matrix: BT.601 Only (JPEG Constraint)

**CRITICAL**: JPEG encoding ALWAYS uses BT.601 RGB→YCbCr, regardless of input gamut.

```
BT.601: Y = 0.299R + 0.587G + 0.114B  (ALWAYS for JPEG)
```

The ICC profile embedded in the JPEG tells decoders what gamut the *decoded* RGB is in.
The YCbCr encoding math doesn't change - only the interpretation of the final RGB.

**Correct wide-gamut encoding pipeline:**
```
Input (any colorspace)
    │
    ▼ [1] Apply EOTF (linearize)
Linear RGB in source primaries
    │
    ▼ [2] Convert primaries (3x3 matrix)
Linear RGB in output primaries
    │
    ▼ [3] Apply OETF (e.g., sRGB gamma)
Gamma-encoded RGB in output primaries
    │
    ▼ [4] BT.601 RGB→YCbCr  ← ALWAYS BT.601!
YCbCr
    │
    ▼ [5] JPEG encode + embed ICC profile
JPEG file (ICC profile describes output primaries)
```

### Color Space Checklist

For EVERY pixel-processing function, answer:

1. **What primaries?** (sRGB/BT.709, P3, Rec.2020, AdobeRGB)
2. **What transfer function?** (Linear, sRGB, PQ, HLG, Gamma 2.2/2.4)
3. **What white point?** (D65, D50, DCI)
4. **Are we doing math?** → Convert to LINEAR first!
5. **Are we storing/transmitting?** → Apply appropriate OETF
6. **Is there an ICC profile?** → Honor it or document why not
7. **Should we write CICP?** → For video-derived content, yes

### UltraHDR Color Pipeline

For UltraHDR, the codec handles ONLY:
- JPEG encoding (SDR base + gain map)
- MPF structure (multi-picture format)
- XMP metadata embedding
- ICC profile embedding

The codec does NOT handle (caller's responsibility):
- HDR to SDR tonemapping
- Gain map computation
- Color space conversions
- ICC profile generation

```rust
// Caller does all color processing
let sdr = my_tonemapper.process(&hdr_linear_p3);  // Caller's tonemapper
let gainmap = compute_gainmap(&hdr, &sdr);         // Caller's gain map
let icc = p3_icc_profile();                        // Caller's ICC

// Codec just encodes and assembles - note stride is ALWAYS provided
let mut encoder = StreamingUltraHdrEncoder::new(w, h, gm_w, gm_h, config)?;
encoder.set_icc_profile(Some(&icc));

let sdr_stride = (w + 15) & !15;  // Caller's buffer layout
let gm_stride = (gm_w + 15) & !15;

for row in 0..height {
    // Push with explicit width and stride
    encoder.push_sdr_rows(&sdr_row_buf, w, sdr_stride, 1)?;
    if row % gm_scale == 0 {
        encoder.push_gainmap_rows(&gm_row_buf, gm_w, gm_stride, 1)?;
    }
}

// Caller provides output buffer
let mut output = Vec::new();
output.try_reserve(estimated_size)?;
encoder.finish_into(&mut output, &metadata)?;
```

### UltraHDR Conformance Notes

zenjpeg uses `ultrahdr-core` for XMP metadata parsing and gain map computation. There are known
differences from Google's libultrahdr reference implementation:

| Behavior | libultrahdr | zenjpeg/ultrahdr-core |
|----------|-------------|----------------------|
| `BaseRenditionIsHDR="True"` | Rejected | Accepted (bug) |
| Required XMP fields validation | Strict | Lenient |
| JPEG boundary detection | JpegScanner | MPF + SOI/EOI fallback |

**Practical impact**: Standard Ultra HDR files work correctly. Edge cases with `BaseRenditionIsHDR="True"`
or missing required XMP fields may behave incorrectly.

See the [ultrahdr README](https://github.com/imazen/ultrahdr#known-differences-from-libultrahdr) for details.

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
default = ["std", "yuv", "archmage-simd"]
decoder = []              # Enable decoder (prerelease, API will change)
parallel = ["dep:rayon"]  # Multi-threaded DCT/quantization
archmage-simd = ["dep:archmage", "dep:magetypes", "dep:safe_unaligned_simd"]  # Token-based SIMD (~10-20% faster)
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
- `archmage-simd` (default): Token-based safe intrinsics via archmage + magetypes - **~10-20% faster** on x86_64

## Key Files for Debugging

| File | Purpose |
|------|---------|
| `zenjpeg/src/encode/mod.rs` | Encoder pipeline |
| `zenjpeg/src/decode/mod.rs` | Decoder pipeline |
| `zenjpeg/src/color/xyb.rs` | XYB color conversion |
| `zenjpeg/src/quant/aq/mod.rs` | Adaptive quantization |
| `zenjpeg/src/huffman/mod.rs` | Huffman encoding |

## External Dependencies

- **Test images**: `internal/jpegli-cpp/testdata/`
- **Codec corpus**: `~/work/codec-eval/codec-corpus/` (Kodak, CID22)
- **C++ reference**: `internal/jpegli-cpp/` submodule

## Detailed Documentation

- **`zenjpeg/README.md` - API Reference** (encoder/decoder usage with examples)
- `zenjpeg/examples/README.md` - Examples and debugging tools
- `zenjpeg/docs/ADAPTIVE_QUANTIZATION.md` - AQ algorithm details
- `zenjpeg/docs/API_DESIGN.md` - Full API surface and proposed enhancements
- `docs/SECURITY.md` - Security considerations
