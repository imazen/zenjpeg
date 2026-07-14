# zenjpeg Tuning History

> **Historical record.** Entries are dated snapshots and are NOT updated
> retroactively; feature flags mentioned in older entries (e.g.
> `archmage-simd`) may since have been removed (SIMD is now always
> compiled). The repo-root `CLAUDE.md` carries the current numbers.

Detailed benchmark data, profiling results, SIMD analysis, and investigation notes
preserved from CLAUDE.md for future reference. This document records the specific
numbers and analysis behind performance decisions -- consult it when revisiting
optimization work or understanding why a particular approach was chosen.

## Encoder Profiling Results (4K image, 2026-01-21)

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
- `finalize_imcu_aq_with_buffer`: 9.6% -> 2.8% (3.4x faster)
- Total AQ: 27% -> 14.5% (1.86x faster)

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
| `ycbcr_planes_i16_to_rgb_u8_avx2` | 0.8M | 1.9% | YCbCr->RGB |
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
| Baseline | 1.31ms | 456us | 65% faster |
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

**Quality parity (comprehensive test, 10 images x 50 quality levels):**

| Metric | Min | Max | Mean |
|--------|-----|-----|------|
| Size delta% | +0.2% | +1.6% | **+0.63%** |
| DSSIM delta% | -0.9% | +1.5% | **+0.41%** |
| Butteraugli delta% | -2.1% | +2.2% | **+0.19%** |

Quality is effectively identical (mean <0.5%); 50/50 quality levels within 5%.

Note: Previous measurements showed 1.4-1.6x slower; improvements came from SIMD sorting
network optimization and allocation reduction.

### Allocation Optimization (2026-01-21)

Reduced allocations from 33,595 to 5,272 per 10 encodes (84% reduction):
- `Vec::with_capacity` in `generate_code_lengths` (classic.rs:187)
- Fixed array instead of `Vec<Vec>` in `depths_to_bits_values` (classic.rs:272)
- Lazy error creation with `ok_or_else` in progressive.rs:78
- Reusable buffers for YUV conversion and AQ strengths

Remaining 527 allocations/encode are inherent to Huffman table generation (13 scans x ~40 allocations each).

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
- Added `padded_width` field (blocks_w x 8) for MCU-aligned buffer stride
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
  - `pre_erosion_row` width=4096: wide 2.49us, archmage 1.16us - **archmage 2.2x faster**
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
| pre_erosion_row | 2.32 us/row | 1.19 us/row | **autovec 1.95x FASTER** |
| gamma_modulation_sum_8x8 | 16.1 ns/block | 34.6 ns/block | wide 2.15x faster |
| hf_modulation_sum_8x8 | 9.4 ns/block | 47.8 ns/block | wide 5.08x faster |
| per_block_modulations_row | 29.8 us/row | 50.9 us/row | wide 1.71x faster |

**Key insight:** Autovectorization works well for:
- Long rows with simple iterator loops (pre_erosion_row -> 1.95x faster)
- Linear iteration without complex boundary checks

Autovectorization fails for:
- Small 8x8 blocks with nested loops (gamma/hf_modulation -> 2-5x slower)
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
- `wide` crate f32x4 -> WASM simd128 v128 mapping works well
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
|------|-----------|---------|-------|
| 256x256 | 82 us | 337 us | 4.1x |
| 512x512 | 285 us | 1.31 ms | 4.6x |
| 1024x1024 | 946 us | 4.56 ms | 4.8x |
| 2048x2048 | 3.80 ms | 19.2 ms | 5.0x |

**Progressive JPEG:**

| Size | zune-jpeg | zenjpeg | Ratio |
|------|-----------|---------|-------|
| 256x256 | 240 us | 519 us | 2.2x |
| 512x512 | 860 us | 2.03 ms | 2.4x |
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

2. **Color conversion (20.5% `to_pixels` + 10% YCbCr->RGB)**
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
   - For 4000x3000 image at 4:2:0:
     - `y_blocks`: 187,500 blocks x 128 bytes = **24 MB**
     - `cb_blocks`: 47,000 blocks x 128 bytes = **6 MB**
     - `cr_blocks`: 47,000 blocks x 128 bytes = **6 MB**
     - Total: **~36 MB** just for block storage
   - Additional buffers (`y_strip`, `cb_strip`, `cr_strip`, `all_aq_strengths`) add more
   - Heaptrack verified: 68 MB peak for 4000x3000 UltraHDR encode vs theoretical ~4 MB for true streaming
   - Root cause: Architecture buffers ALL blocks before encoding, even with `optimize_huffman=false`
   - "Streaming" only streams INPUT rows, not OUTPUT blocks
   - Baseline mode with fixed Huffman tables COULD support true streaming (write blocks immediately)
   - Current impl: `streaming.rs:1574-1650` always reads from buffered `strip_output.*_blocks`
   - Fix: Add immediate-write mode for baseline + fixed Huffman (no two-pass needed)

### Callgrind Analysis (2026-01-22)

Run: `valgrind --tool=callgrind ./target/release/examples/valgrind_decode jpegli 512`

**Totals (512x512 baseline JPEG):**

| Metric | zenjpeg | zune-jpeg | Ratio |
|--------|---------|-----------|-------|
| Instructions | 60.3M | 34.6M | **1.74x** |
| Data refs | 16.4M | 9.5M | 1.72x |
| Branches | 9.2M | 4.0M | 2.3x |
| D1 miss rate | 2.3% | 3.7% | better |
| Branch mispredict | 0.8% | 1.5% | better |

**Decode-specific function comparison:**

| Function | zenjpeg | zune-jpeg | Notes |
|----------|---------|-----------|-------|
| **Upsampling** | 10.5M (17%) | 115K (0.3%) | **91x more!** Scalar loop |
| **YCbCr->RGB** | 4.4M (7%) | 1.0M (3%) | 4.4x more |
| **IDCT** | 4.2M (7%) | 1.2M (3.5%) | 3.5x more |
| **Entropy decode** | ~1.5M | ~1.7M | similar |
| **memset** | 2.0M (3.4%) | 591K (1.7%) | 3.4x more |

**Root cause:** Upsampling is **91x worse** because `upsample_h2v2_i16_fancy` is fully scalar,
iterating pixel-by-pixel while zune uses AVX2 SIMD (`upsample_horizontal_avx2`).

### Decoder Optimization Path

1. Branchless huff_extend (done)
2. AVX2 upsampling (done) - **94% reduction** (10.5M -> 0.7M instructions)
3. Autovectorized YCbCr conversion (done)
4. Zero-copy coefficient decode (5-7% speedup)
5. **AVX2 IDCT** - Now biggest gap: 7M vs zune's 1.2M = **5.8x slower**
6. Reduce memset overhead (2.6M instructions, 5.6%)
7. Macro-based Huffman (eliminate ScanRead enum)

**Current status (2026-01-22):** 46.3M instructions (was 60.3M), zune is 34.6M.
Gap reduced from 1.74x to 1.34x in instruction count.

## Investigation Notes

### DCT Coefficient Parity (2026-01-22) - VERIFIED

**Rust now supports both 2-table and 3-table modes via `separate_chroma_tables` flag.**

The two C++ functions use different quant table configurations:
- `jpeg_set_quality()`: 2 chroma tables (`add_two_chroma_tables=false`) - Cr matrix used for both Cb and Cr
- `jpegli_set_distance()`: 3 tables (`add_two_chroma_tables=true`) - separate Y, Cb, Cr tables

Rust configuration:
- `separate_chroma_tables(true)` (default): 3 tables, matches `jpegli_set_distance()`
- `separate_chroma_tables(false)`: 2 tables, matches `jpeg_set_quality()`

**Root cause of +/-1 coefficient differences:**
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

**Coefficient differences are normal +/-1 rounding, not systematic bugs:**
- 80% of Y blocks differ by +/-1 in 1-3 AC coefficients
- Max difference: 6 (single outlier block)
- Cb/Cr: 27-37% blocks differ, all by +/-1

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
- AQ is identical -> same quantization tables
- Difference is in DCT rounding (+/-1 coefficient differences)
- Both use optimized Huffman (verified: `optimize_coding = 1` in FFI)
- Butteraugli quality within 1% (essentially identical perceptual quality)

### Visual Diff Interpretation (2026-01-31)

**Repro commands:**
```bash
just xyb-diff                    # XYB mode: C++ | Rust | dR*10 | dG*10 | dB*10
just ycbcr-diff                  # YCbCr mode: same 5-panel layout
just xyb-diff ~/path/to/img.png  # Custom image
```

**XYB mode visual patterns (kodak/1.png q90):**
- **C++ & Rust panels**: Visually identical, slight green tint (normal for this image)
- **dR (3rd panel)**: Block patterns with U/M shapes at 8x8 boundaries, sometimes strong
  - Indicates DCT coefficient quantization differences
  - Pattern follows block grid = coefficient rounding at block level
- **dG (4th panel)**: Uniform even noise across image
  - Luminance (Y) channel has consistent small differences
  - No block structure = no systematic quantization bias
- **dB (5th panel)**: Similar to dR but less intense
  - Was severely corrupted before B-channel fix (mean error ~51)
  - Now shows same block-pattern differences as R channel

**XYB numeric results (kodak/1.png 768x512 q90, after AQ channel fix):**
```
Rust: 153752 bytes, C++: 141450 bytes (+8.7%)
Mean |diff|: R=0.237, G=1.017, B=0.104
```
Note: Before fix was +10.8% size, R=0.310, G=1.300, B=0.200.

**YCbCr mode visual patterns:**
- **All diff panels**: Uniform noise pattern, no obvious block structure
- More even distribution of differences across all channels

**YCbCr numeric results (kodak/1.png 768x512 q90):**
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
- The U/M patterns in dR suggest systematic differences in how edge blocks are quantized
- May be related to AQ strength interpolation or zero-bias calculation
- Files: `zenjpeg/src/quant/aq/simd.rs`, `zenjpeg/src/encode/strip/mod.rs:quantize_pending_imcu`

### Hybrid Trellis Rate-Distortion Analysis (2026-02-02)

**Goal:** Determine if AQ-coupled trellis provides better rate-distortion than standalone trellis.

**Hypothesis:** Adjusting trellis lambda per-block based on AQ strength should improve perceptual
quality by spending more bits on smooth areas (low AQ, visible artifacts) and fewer on textured
areas (high AQ, masking effect).

**Implementation:** `HybridConfig::to_trellis_config()` adds `aq_strength * coupling` to
`lambda_log_scale1`. Higher lambda = more aggressive coefficient zeroing.

**Benchmark results** (`cargo run --release --example hybrid_trellis_benchmark`):

| Image | Size | Q | Mode | Bytes | DSSIM | Butteraugli |
|-------|------|---|------|-------|-------|-------------|
| flower_small | 510x532 | 85 | jpegli | 38596 | 0.00072 | 1.918 |
| flower_small | 510x532 | 85 | standalone | 42587 | 0.00053 | 1.809 |
| flower_small | 510x532 | 85 | hybrid(2.0) | 39298 | 0.00069 | 1.789 |
| apple.com | 1920x1080 | 85 | jpegli | 503363 | 0.00052 | 7.389 |
| apple.com | 1920x1080 | 85 | standalone | 469870 | 0.00065 | 7.831 |
| apple.com | 1920x1080 | 85 | hybrid(2.0) | 542594 | 0.00041 | 7.483 |

**Analysis:**
- Hybrid improves DSSIM (lower = better) by 23-49% vs standalone
- But files are 10-30% larger
- On flower_small at Q85: hybrid is 7.7% smaller than standalone but 28.6% worse DSSIM
- On apple.com at Q85: hybrid is 15.5% larger than standalone but 37.1% better DSSIM

**Root cause:** The coupling direction may be inverted. When AQ is high (textured), we increase
lambda (more compression), zeroing more coefficients. This improves DSSIM (perceptual metric that
penalizes texture loss less) but the "saved" bits don't reduce file size -- they're just lost.

**Alternative approaches to explore:**
1. **Reverse coupling** -- decrease lambda for high-AQ blocks (preserve texture, may improve Butteraugli)
2. **Rate-targeting** -- use hybrid to hit a target file size with better quality allocation
3. **Quality-adaptive coupling** -- different coupling strategies at different quality levels
4. **Butteraugli-optimized** -- tune for Butteraugli instead of DSSIM
5. **Block-level rate control** -- redistribute bits from textured to smooth blocks explicitly

**Files:**
- `encode/trellis/hybrid.rs` -- `compute_lambda_adjustment()` does `aq_strength * aq_lambda_scale`
- `encode/trellis/hybrid.rs` -- `to_trellis_config()` applies adjustment to `lambda_log_scale1`
- `examples/hybrid_trellis_benchmark.rs` -- rate-distortion measurement tool

### ExpertConfig Parameter Sensitivity (2026-02-02)

Test: `cargo test --release -p zenjpeg --lib -- search::tests::test_parameter_sensitivity --nocapture`
Image: 256x256 deterministic noise+patches (not gradient), MozjpegBaseline Q85 base = 17,327 bytes.

**Preset baselines (all Q85, 4:2:0):**

| Preset | Bytes | vs MozBase |
|--------|-------|------------|
| MozjpegMaxCompression | 16,979 | -2.0% |
| MozjpegProgressive | 17,043 | -1.6% |
| MozjpegBaseline | 17,327 | -- |
| JpegliBaseline | 18,355 | +5.9% |
| JpegliProgressive | 18,612 | +7.4% |
| HybridBaseline | 23,081 | +33.2% |
| HybridProgressive | 23,455 | +35.4% |
| HybridMaxCompression | 23,130 | +33.5% |

Hybrid presets use jpegli tables + standalone trellis (coupling=0). They don't use hybrid
AQ-coupled trellis by default because `aq_trellis_coupling=0` in all presets.

**Active parameters ranked by max |delta|:**

| Parameter | Range | Min delta | Max delta | Notes |
|-----------|-------|-----------|-----------|-------|
| `tables.quant` (192 vals) | 0.5x-2.0x | -54% | +65% | Primary optimization target |
| `trellis_lambda_log_scale1` | 12.0-17.0 | -46% | +12% | Exponential rate/distortion tradeoff |
| `zero_bias_mul` (jpegli only) | 0.0-1.0 | -14% | +31% | Mozjpeg uses all-zeros (no effect) |
| `trellis_lambda_log_scale2` | 14.0-18.0 | -19% | +11% | Inverse relationship to scale1 |
| `quality` (Scaled only) | Q50-Q95 | -81% | +112% | Zero effect with Exact/mozjpeg tables |
| `trellis_enabled` | on/off | -- | ~15% | Binary toggle |
| `scan_mode` | 4 variants | -- | -2% | ProgressiveSearch best |
| `trellis_delta_dc_weight` | 0.0-5.0 | 0% | +1% | Diminishing above 2.0 |
| `trellis_dc_enabled` | on/off | -- | ~0.1% | Tiny |
| `downsampling_method` | 3 variants | -- | +/-0.2% | Marginal |

**Dead parameters (zero effect regardless of value):**

| Parameter | Root cause | File:line |
|-----------|-----------|-----------|
| `trellis_use_lambda_weight_tbl` | Hardcoded flat 1/q^2 weights | `encode/trellis/ac.rs:47-52` |
| `trellis_num_loops` | Stored but never read (single-pass) | `encode/trellis/ac.rs` (absent) |
| `trellis_speed_mode` | Only search bounds, DP finds same optimum | `encode/trellis/ac.rs:102-113` |
| `aq_trellis_coupling` | **FIXED**: Now affects output (larger files, better DSSIM) | `streaming.rs:263` |
| All `aq_trellis_*` fields | **FIXED**: Now affect lambda adjustment | `encode/trellis/hybrid.rs` |
| `quality` (Exact tables) | Tables pre-scaled; zero-bias all-zeros | `encode/tables/robidoux.rs:99-106` |
| `allow_16bit_quant_tables` | No effect at Q85+ (values <= 255) | -- |
| `deringing` | Only triggers on saturated (255) pixels | `deringing.rs:131-135` |

**Hybrid mode flow (FIXED 2026-02-02):**
1. `ExpertConfig::build_trellis_or_hybrid()` builds `HybridConfig` when coupling > 0
2. `to_encoder_config()` stores it in `config.hybrid_config`
3. `BytesEncoder::build_streaming_encoder()` calls `builder.hybrid_config(config.hybrid_config)`
4. `StreamingEncoder::from_builder()` calls `processor.set_hybrid(builder.hybrid_config)`
5. `StripProcessor::set_hybrid()` creates `HybridQuantContext::new(config)` (Hybrid mode)
6. Quantization uses AQ-coupled lambda adjustment per-block

**Hybrid mode benchmark results (2026-02-02):**
Run: `cargo run --release --example hybrid_trellis_benchmark`

| Image | Q | standalone | hybrid(2.0) | Size delta | DSSIM delta |
|-------|---|------------|-------------|------------|-------------|
| flower_small | 85 | 42587 | 39298 | -7.7% | +28.6% |
| flower_small | 90 | 54278 | 49677 | -8.5% | +31.2% |
| apple.com | 85 | 469870 | 542594 | +15.5% | -37.1% |
| apple.com | 90 | 566481 | 633700 | +11.9% | -32.8% |

**Update (2026-02-02):** Negative coupling NOW WORKS and produces smaller files:
- `aq_trellis_coupling=-4.0`: ~2% smaller files with ~3% DSSIM degradation (photos)
- `aq_trellis_max_adjustment=1.0`: **CRITICAL** for screenshots -- limits quality degradation
  - Without cap: apple.com at coupling=-8.0 -> -24% size, +552% DSSIM (destroyed!)
  - With max_adj=1.0: apple.com at coupling=-8.0 -> +5.3% size, +5.9% DSSIM (acceptable)
- `aq_trellis_multiplicative=true`: Proportional scaling (use smaller values like 0.1)

**Recommended settings:**
- Photos: `coupling=-4.0, max_adjustment=0.0` -> -1.8% size, +3.3% DSSIM
- Mixed/Unknown: `coupling=-8.0, max_adjustment=1.0` -> photos -4%, screenshots protected
- Screenshots: Use `coupling=-1.0` or disable hybrid entirely

**Auto-detection (2026-02-02):** Use AQ statistics to automatically choose settings:
```rust
use zenjpeg::encode::trellis::{adaptive_config, detect_image_type, ImageType};
use zenjpeg::quant::aq::compute_aq_strength_map;

// After computing AQ map...
let (_, _, aq_mean, aq_std) = aq_map.stats();
let image_type = detect_image_type(aq_mean, aq_std); // Photo, Screenshot, or Mixed
let hybrid = adaptive_config(aq_mean, aq_std);       // Returns texture-adaptive HybridConfig
```

Detection uses coefficient of variation (CV = std/mean):
- CV > 1.5 -> Screenshot -> safe_compression()
- CV <= 1.5, mean >= 0.06 -> Photo -> **texture-adaptive coupling**
- Otherwise -> Mixed -> safe_compression()

**Texture-adaptive coupling (2026-02-02):** For photos, coupling scales with AQ mean:
- Low texture (mean <= 0.15): coupling = -4.0 (aggressive)
- High texture: coupling = -4.0 * (0.15 / mean), gentler as texture increases
- Example: mean=0.30 -> coupling=-2.0, mean=0.60 -> coupling=-1.0

**CID22 benchmark (20 images, Butteraugli metric):**
| Mode | Size delta | Butteraugli delta |
|------|------------|-------------------|
| Fixed -4.0 (old) | -9.2% | +10.9% |
| Texture-adaptive | **-3.3%** | **+2.7%** |

Run `cargo run --release --example cid22_hybrid_bench` for full benchmark.

**Validated presets:**
- `HybridConfig::aggressive_compression()` -- photos only (fixed -4.0), risky on high-texture
- `HybridConfig::safe_compression()` -- all content types, max_adj=1.0 protection
- `adaptive_config(mean, std)` -- **recommended**, texture-aware for photos

Run `cargo run --release --example hybrid_auto_detect` to validate detection.

**For optimizers:** Tune `tables.quant` (192 values), `lambda_log_scale1/2` (2 floats),
`zero_bias_mul` (192 values, jpegli only), and `aq_trellis_*` fields for size/quality trade-offs.
Run `cargo run --release --example hybrid_parameter_sweep` for comprehensive analysis.

## Fixed Bugs (historical reference)

- **Double-lambda in hybrid trellis quantization (2026-02-03)** -
  `hybrid_quantize_block()` unconditionally added `aq_strength * AQ_LAMBDA_SCALE` (2.0)
  to `lambda_log_scale1`, but its caller already computed AQ-adjusted lambda via
  `HybridConfig::to_trellis_config()`. With all presets using `coupling=0.0`,
  `to_trellis_config()` returned unadjusted lambda, but `hybrid_quantize_block()` still
  added the hardcoded adjustment. Fix: removed `AQ_LAMBDA_SCALE` constant and the
  redundant adjustment; lambda is now adjusted solely in `to_trellis_config()`.
  With `coupling=0.0`, hybrid mode now produces identical output to standalone trellis.
  Files: `encode/trellis/hybrid.rs`

- **Default EncoderConfig silently enabled hybrid trellis (2026-02-03)** -
  `EncoderConfig::default_internal()` used `HybridConfig::default()` (enabled=true).
  Commit ec3b52c added `else if config.hybrid_config.enabled` to byte_encoders.rs,
  which activated hybrid trellis for all default configs. Caused:
  - `test_trellis_disabled_matches_default` failure (default != disabled)
  - `cpp_parity_locked` Q5 failures (11-13% size regression at extreme low quality)
  - `detect_image_type` doctest wrong example values (Mixed, not Screenshot)
  Fix: Changed default to `HybridConfig::disabled()`. Users must explicitly opt in.
  Files: `encode/encoder_config.rs:157`, `encode/trellis/hybrid.rs:115`

- **Hybrid trellis improvements (2026-02-02)** - Multiple fixes and new features:
  1. Changed condition from `> 0` to `!= 0` to allow negative coupling (smaller files)
  2. Added `aq_trellis_multiplicative` for proportional scaling
  3. Added `aq_trellis_max_adjustment` to cap quality degradation on sensitive images
  4. Added auto-detection: `detect_image_type()` uses CV (std/mean) to classify images
  5. Added `adaptive_config()` returns appropriate HybridConfig for detected image type
  6. Added presets: `aggressive_compression()`, `safe_compression()`, `quality_boost()`
  Results with `coupling=-4.0, max_adjustment=0.0` (photos):
  - flower_small: -2.4% size, +3.4% DSSIM
  Results with `coupling=-8.0, max_adjustment=1.0` (safe):
  - flower_small: -4% size, +7% DSSIM
  - apple.com: -2.5% size, +5.9% DSSIM (protected from 552% degradation!)
  Files: `encode/search.rs`, `encode/trellis/hybrid.rs`, `examples/hybrid_*.rs`

- **XYB file size gap (2026-02-01)** - XYB baseline was 2-3% larger than C++, but this
  was due to Rust getting 2x more progressive savings (5.7-7.3% vs C++'s 3.1-3.6%).
  With progressive mode, Rust XYB matches or beats C++ (-0.3% to -4.3% smaller).
  Resolution: Progressive mode is now recommended for XYB.
  Files: `encode/strip/mod.rs`, `quant/aq/streaming.rs`

- **XYB AQ v_samp mismatch (2026-01-31)** - AQ was initialized with v_samp=1 (from S444
  subsampling), but XYB JPEG uses R:2x2 G:2x2 B:1x1 (max_v_samp_factor=2). This caused
  the AQ to treat each 8-row strip as a full iMCU instead of 16-row iMCUs, producing
  overly conservative quantization. Also affected pending DCT block buffer sizing.
  Fixed in `encode/strip/mod.rs`: both `v_samp` calculations now check `use_xyb`.
  Impact: Size diff 8-18% -> 5-11% (3-7pp improvement).

- **XYB AQ using wrong channel (2026-01-31)** - AQ was computed on X channel instead of Y.
  C++ uses `y_channel = jpeg_color_space == JCS_RGB ? 1 : 0`, meaning channel 1 (Y) for XYB.
  Fixed by using `cb_strip` instead of `y_strip` when `use_xyb=true`.
  Impact: Size diff 10.8% -> 8.7% (~2pp improvement), all color diffs reduced.
  Files: `encode/strip/mod.rs:779-795`

- **XYB B-channel encoding corruption (2026-01-31)** - B channel had ~51 mean error vs ~0.1 for R/G.
  Root cause: `StripProcessor` created with `use_xyb=false`, then `set_xyb_mode(true)` called,
  but this didn't recalculate B-channel dimensions. Also `c_blocks_h/v` used for B instead of
  `b_blocks_h/v`. Fixed by using `with_xyb()` constructor and adding `padded_b_width`,
  `b_blocks_h`, `b_blocks_v` fields. Files: `streaming.rs`, `strip/mod.rs`, `strip/convert.rs`.

- **Debug env var in hot loop** - `entropy/encoder.rs`: Removed `std::env::var()` call from hot path (was 12% overhead)

- **Eager error evaluation** - `entropy/encoder.rs`: Changed `ok_or()` to `ok_or_else()` (13% speedup)

- **Progressive XYB decode** - `decode/mod.rs`: Handle `EndOfScanData` gracefully for non-standard component IDs

- **1-pixel partial MCU edge** - `fast_yuv.rs`, `streaming.rs`: Added edge replication for width = 1 (mod 8)

- **HF modulation index wrap** - `quant/aq/simd.rs`: Added bounds check for rightmost partial blocks

## Compile-Time Profile (2026-06-28, rustc 1.96.0 stable, commit 7afedf4c)

Measured cold, lib-only, default features: `run-heavy --jobs 8 -- cargo build
--release --timings -p zenjpeg`. Frontend/codegen split read from the `sections`
field of the `cargo-timings` HTML; monomorphization from `cargo llvm-lines
--release -p zenjpeg --lib`.

**zenjpeg is genuinely fast to compile in absolute terms** — 15.9 s cold for the
whole dep graph (60 units), peak RSS 0.92 GiB. The interesting part is the *shape*.

**Per-unit (top), with frontend/codegen split:**

| unit | total | frontend | codegen | codegen% |
|------|-------|----------|---------|----------|
| **zenjpeg** | 8.21 s | 3.89 | 4.32 | **52%** |
| zenpixels-convert | 3.60 | 1.06 | 2.54 | 70% |
| zerocopy | 2.70 | 2.61 | 0.09 | 3% |
| magetypes | 2.66 | 2.58 | 0.08 | 3% |
| zenanalyze | 2.53 | 0.61 | 1.92 | 75% |
| linear-srgb | 2.38 | 0.50 | 1.88 | 78% |
| wide | 1.84 | 1.45 | 0.39 | 21% |
| syn | 1.38 | 1.18 | 0.20 | 14% |

Across all 60 units: frontend 19.9 s + codegen 16.5 s of CPU-time (codegen 45%).
zenjpeg itself is **codegen-bound (52%)** — atypical for a Rust library (a
pure-logic crate is usually ~95% frontend). The SIMD-monomorphization +
array-generic shape is what pushes work into LLVM. Note the proc-macro/trait-def
deps (zerocopy, magetypes, syn) are ~100% *frontend* — their codegen cost is paid
later, inside zenjpeg, when the generics are instantiated.

**Monomorphization (`llvm-lines`): 504,370 IR lines / 5,586 fn copies.**

| category | IR lines | share |
|----------|----------|-------|
| all `zenjpeg::*` | 332,232 | 66% |
| stdlib generic monomorph (core/alloc) | 148,581 | 29% |
| └ SIMD kernels (`__arcane_*_avx2`, `_simd`) | 50,840 | 10% |
| └ const match-tables (`huffman::builtin_tables::*`) | 25,498 | 5% |
| └ Display/Debug `::fmt` impls | 16,712 | 3% |

Highest-*copies* offenders are stdlib generics driven by `[T; N]` array code
(8×8 block buffers) and Vec/iterators across many element types:
`core::array::try_from_fn_erased` 71 copies / 9,203 lines, `try_from_fn` 71,
`Vec::drop` 64, `map_fold` closure 52, `Vec::push_mut` 47. zenjpeg's *own* big-IR
functions are mostly single-copy (SIMD kernels + Huffman tables) — large but not
multiplied.

**SIMD-macro exposure (lib):** 55 `#[magetypes]` (all `(v3, neon, wasm128,
scalar)` = 4 tiers each; on x86_64 only v3+scalar reach codegen, neon/wasm are
cfg-stripped), 49 `#[arcane]`, 28 `#[rite]`, 71 `incant!`. This is the
zenjpeg-specific codegen multiplier; each AVX2 kernel is ~2.0–2.7 k IR lines.

**Levers (measured + per `~/work/claudehints/topics/rust-defaults.md`):**
- `[profile.release] incremental = true` — biggest dev-iteration win for warm
  rebuilds of zenjpeg (stable since 1.94). Only helps crates you edit.
- lld is already the default linker (x86_64 Linux, since 1.90) — linking is ~free.
- Cranelift (nightly) would **not** help — it falls back to LLVM for SIMD intrinsics.
- Parallel frontend (`-Z threads`, still nightly in 2026) would attack the 3.89 s
  (of 19.9 s graph-wide) frontend half; not on by default.
- Non-default features (`parallel`, `boundary-rd`, `ultrahdr`, `target-zq`) are
  already off in the measured build — enabling them forks more codegen paths.

### Downstream-consumer build cost (what a dependent crate pays)

A consumer compiles zenjpeg + its **transitive normal deps** once (dev-deps —
criterion/proptest/zune-jpeg/mozjpeg — are NOT paid downstream). `default = []`,
but the **non-optional** dep list is the floor and it is heavy.

Measured cold, `cargo build --release --no-default-features -p zenjpeg` (= the
default consumer graph), rustc 1.96.0, commit f2e0b4f3:
**60 compile units, 14.8 s wall (-j8), 33.6 s CPU-time (17.9 frontend + 15.7
codegen).** 43 unique crates. On a 2-vCPU CI runner expect ~25–33 s wall (less
parallelism; the proc-macro base serializes the early graph).

Heaviest units: zenjpeg 7.8, zenpixels-convert 3.56, magetypes 2.44, zenanalyze
2.44, zerocopy 2.34, linear-srgb 2.31, wide 1.69, garb 1.47, zencodec 1.30,
syn 1.21, zenyuv 1.15.

**Two non-optional subtrees dominate removable cost (~33% of CPU-time) for
features a plain encode/decode consumer doesn't use:**
- `zenanalyze` (git) → `zenpixels-convert`: **~6.0 s** (2.44 + 3.56). Pulled
  unconditionally for `EncoderConfig::adaptive` content analysis. Gating it
  behind an `adaptive` feature removes ~18% of the build **and** un-blocks
  publishing (see below).
- `ultrahdr-core` → `half` → `zerocopy`/`zerocopy-derive` + `wide` → `safe_arch`:
  **~5.4 s** (zerocopy 2.34 + wide 1.69 + zerocopy-derive 0.64 + safe_arch 0.40
  + half 0.30). `ultrahdr-core` 0.5 (pinned rev 3ac20f99) is non-optional for
  container/MPF types and pulled `half` unconditionally; `half`'s f16 drags in
  `zerocopy` (the single heaviest leaf dep at 2.34 s).

  **Fixed upstream (2026-06-28): ultrahdr-core 0.6 (imazen/ultrahdr `708d68a`)
  gates f16/`half` behind an opt-in `f16` feature and drops the `wide` dep.**
  Since zenjpeg already takes ultrahdr-core with `default-features = false`,
  bumping the pin removes all five crates from the default graph — verified by
  resolution: **43 → 37 crates, ~5.4 s CPU (~16%) gone**, zerocopy being the win.
  **Confirmed by a clean build (rc=0):** ultrahdr-core 0.6 requires the
  unpublished `zenpixels-convert 0.2.15` → `zenpixels 0.2.16`. Bumping only the
  ultrahdr-core rev diamonds against the stale pinned `zenanalyze 13d40c3` (still
  on `zenpixels 0.2.14`) — `RowConverter::new` gets two incompatible
  `PixelDescriptor` types (E0308, zenanalyze `row_stream.rs:112`). The fix is the
  documented local-dev pattern: `[patch]` the **whole zen pixel stack** to local
  sibling folders (zenpixels + zenpixels-convert + zencodec + zenanalyze +
  ultrahdr-core), which unifies everything on local `zenpixels 0.2.16`. With that,
  the default graph builds cleanly with all five crates gone (43 → 37). The
  ~5.4 s figure is the removed subtree measured in the consistent crates.io
  baseline; a *full-build total* from the local-patched tree is not comparable
  (local pixel-stack source is heavier than the published crates and the run had
  higher load). **Landing for downstream** still needs a coordinated *published*
  bump of the pixel stack (absolute-path patches can't be committed — CI breaks);
  or do lever #1 (gate `zenanalyze`) first, which removes the diamond's source.

**Publish gate (CRITICAL for downstream): newer zenjpeg is unpublishable.**
crates.io has **0.8.4** (21 non-optional deps, no zenanalyze/zenpixels-convert —
leaner but frozen). Local is **0.8.7**. `cargo publish` of 0.8.x fails because
`zenanalyze = { git, rev }` (non-optional, **no version**) — crates.io forbids
git deps. `zensim` (git, no version) blocks the `target-zq`/`recompress-iqa`
features too. So `cargo add zenjpeg` consumers are stuck on 0.8.4; only git/path
consumers (imazen ecosystem) get 0.8.7 — and pay a git fetch of imazen/zenanalyze
+ imazen/ultrahdr at first build on top of the 60-unit compile.

**Downstream levers, ranked:** (1) make `zenanalyze` optional → −6 s **and**
restores publishability; (2) make `ultrahdr-core`/`half` optional → −5 s, drops
`zerocopy`; (3) publish `zenanalyze`/`zensim` with versions so 0.8.5+ can ship;
(4) the proc-macro base (`syn` 1.21 + `proc-macro2`/`quote` + 8 derive macros) is
a ~3–4 s mostly-unavoidable serial floor — but `zerocopy-derive` is only there via
lever #2's `half` path.

### Why `garb` is codegen-bound (~1.5 s in-graph for a swizzle crate)

Looks surprising — `garb` is "just" RGB↔BGRA byte swizzles — but it's **not** generic
bloat. Isolated measurements (garb 0.2.8, rustc 1.96):

| garb feature set | total | frontend | codegen | note |
|---|---|---|---|---|
| `default,std` (inherent) | 0.76 s | 0.19 | 0.57 (75%) | the floor |
| `experimental,rgb,imgref` | 1.17 s | 0.54 | 0.63 | +0.41 s, mostly frontend |
| in zenjpeg graph | ~1.5 s | — | — | + rgb monomorph + load |

`llvm-lines`: **only 11,808 IR lines / 314 fn copies — tiny.** 255 of 277 functions
are single-copy (count==1), so monomorphization is NOT the cause. The cost is
**function count, not IR size:** garb has **164 `#[arcane]`** (`#[target_feature]`)
functions — 112 in `bytes/avx2.rs` — and a `#[target_feature]` boundary is a hard
optimization unit LLVM can't inline across, so each AVX2 kernel is optimized in
isolation. The top 11 IR offenders are all single-copy AVX2 swizzle kernels:
`rgb_to_abgr_row_v3`, `rgb_to_argb/bgra/rgba_row_v3`, `argb_to_bgr/rgb_row_v3`,
`bgra_to_rgb_row_v3`, `gray_alpha_to_4bpp_*_v3`… i.e. a **combinatorial expansion of
every (src-format → dst-format) pair × the v3/AVX2 tier**. avx2 kernels = 5,049 IR
(43% of garb's total). (neon's 45 + wasm's 72 `#[arcane]` are `cfg(target_arch)`-
stripped on x86_64, so they cost frontend parse only, not codegen.)

This is the same "LLVM-bound on monomorph/fn count, not IR bytes" signature as
zenjpeg's own SIMD kernels — inherent to the write-one-kernel-per-tier archmage
pattern. **The "unexpected" extra ~0.4 s in zenjpeg's graph is feature
unification:** `zenanalyze` + `zenpixels-convert` force garb's `experimental` (+
`rgb` + `imgref`) on (paste!-generated APIs + extra generic surface). garb itself
has no lever short of fewer format-pairs; the only downstream win is dropping
`experimental` if a consumer doesn't need it — and lever #1 (gate `zenanalyze`)
removes garb from the default graph entirely along with its forced features.

#### garb format-gate inventory (2026-06-28): bgr/argb/abgr are dead weight

garb's kernels are concrete `pub fn`s, so they codegen into garb's rlib whether
or not anyone calls them — only a `#[cfg(feature)]` gate *in garb* removes them.
So: which formats does zenjpeg's whole graph actually use? Inventoried every
`garb::` call across all three direct consumers (garb 0.2.8, zenpixels-convert
0.2.14, zenanalyze 13d40c3, zenjpeg `src/`):

| garb format | fns (all tiers) | used in zenjpeg graph? | by whom |
|---|---|---|---|
| rgb / rgba | — | **yes** (heavy) | all three |
| bgra | — | **yes** (`rgb_to_bgra`, `bgra_to_rgb`, `rgba_to_bgra`) | zenjpeg + zpc + zenanalyze |
| gray / gray_alpha | 128 | **yes** (8 fns) | zenpixels-convert |
| **bgr** (3-byte) | 66 | **NO — zero calls** | — |
| **argb** | 77 | **NO — zero calls** | — |
| **abgr** | 41 | **NO — zero calls** | — |

`bgr`+`argb`+`abgr` = **165 of garb's 812 fns (~20%)** with **zero callers** in
the entire zenjpeg dependency graph. On x86_64 that's ~18 AVX2 kernels
(`swap_bgr`/`copy_swap_bgr`/`argb_to_bgr` + `rgb_to_argb`/`rgb_to_abgr`/
`argb_to_rgb`, each ×impl/row/strided) + their scalar twins + dispatch wrappers
= ~10–15 % of garb's codegen (~0.06–0.10 s). Small absolute, but pure dead code,
and the cleanest part of garb to gate — and it'd help *every* garb consumer.

**Recommendation:** add `bgr` / `argb` / `abgr` cargo features to garb (one
`extended-formats` umbrella is fine), default-ON for back-compat, and have
zen consumers take garb with `default-features = false` + only the formats they
use (mirrors how garb already gates `rgb` / `imgref`). **Grayscale is NOT
gateable here** — `gray`/`gray_alpha` (128 fns) is load-bearing: zenpixels-convert
calls 8 gray conversions (`gray_to_rgb/rgba`, `gray_alpha_to_*`, `*_to_gray_bt709`)
and grayscale-JPEG decode needs the gray→rgb expansion. garb *could* expose a
`gray` feature for other consumers, but in this graph it stays on.

Full zenpixels-convert → garb surface (the requested inventory): conversions
`rgba_to_rgb`, `rgb_to_bgra`, `bgra_to_rgb`, `rgb_to_rgba`, `rgba_to_bgra`(+`_inplace_strided`);
gray `gray_to_rgb/rgba`, `gray_to_gray_alpha`, `gray_alpha_to_gray/rgb/rgba`,
`rgb_to_gray_bt709`, `rgba_to_gray_bt709`; depth `convert_{u8,u16,f32}_to_{u16,f32,u8}`;
alpha `premultiply_alpha_rgba_u8_copy`, `(un)premultiply_alpha_f32_copy`;
deinterleave `rgb24_chunk8_to_planes_tokenless_v3`. **No bgr/argb/abgr.**

#### What garb's `experimental` feature actually gates, and who uses it (2026-06-28)

`experimental` (force-enabled by zenanalyze + zenpixels-convert) is a **coarse
umbrella** over 6 sub-areas. Two are load-bearing, four are dead weight:

| experimental sub-area | size | used by |
|---|---|---|
| `deinterleave` module (RGB24→planar) | 1674 lines, 15 SIMD | **zenanalyze** (`rgb24_chunk8_to_planes_scalar` ×3, `…_tokenless_v3`) + **zpc** (`…_tokenless_v3`) |
| `experimental_api` block (mod.rs 799–1571) | 773 lines | **zpc only** — depth `convert_{u8,u16,f32}_to_{…}` (6) + alpha `premultiply_alpha_*`/`unpremultiply_*` (3) |
| `packed` (RGB565 / RGBA4444) | 642 lines | **nobody** |
| `packed_1010102` (10-bit RGB) | 776 lines | **nobody** |
| `experimental_imgref` (imgref.rs block) | gated block | **nobody** |
| `experimental_typed` (typed_rgb.rs block) | gated block | **nobody** |

So `experimental` is **not** optional bloat that could be dropped — it gates the
`deinterleave` module (RGB→planar, exactly what JPEG encode needs) and the
depth-convert + alpha-premultiply API zpc relies on. But because the flag is
coarse, turning it on to get `deinterleave` *also* drags in **~1,400 lines of
unused packed-format code** (`packed.rs` 642 + `packed_1010102.rs` 776) plus the
experimental imgref/typed blocks, and runs `paste!` expansion for the unused
typed/imgref paths.

**Refined garb recommendation:** the win isn't "drop experimental" (it's needed),
it's **split it** — separate `deinterleave`, `depth-convert`/`alpha`,
`packed`, `packed-1010102`, `imgref-experimental`, `typed-experimental` features.
Then zenjpeg's graph enables only `deinterleave` + `depth-convert`/`alpha` and
skips ~1,400 lines of packed-format code that nothing in the graph calls. This is
a bigger and cleaner cut than the bgr/argb/abgr gate above, since the unused
packed modules are entire files, not scattered kernels.

**Why `deinterleave.rs` is 1,674 lines (and what's actually used):** the headline
overstates compile cost — **504 lines (30%) are `#[cfg(test)]`** (line 1170+, not
compiled downstream) + ~280 comment/doc. The ~1,170 production lines are **46
functions = a full RGB↔planar matrix**: dtype {rgb24-u8, rgb48-u16, rgb-f32,
rgba-f32} × direction {forward to_planes (26), reverse planes→rgb (12)} × chunk
{4, 8, 16} × tier {scalar, v3, v3_autovec, tokenless_v3}. zenjpeg's whole graph
calls **only `rgb24_chunk8_to_planes_{scalar,tokenless_v3}` — 2 of 46** — so all
rgb48/u16, all f32 (rgb & rgba), all reverse-direction, and the chunk4/chunk16
variants are unused. The module is big because it's comprehensive (every dtype ×
direction × granularity); the graph touches only the u8-RGB24-chunk8-forward
corner. A `deinterleave` split could go finer still (per-dtype), but the module is
mostly scalar f32 loops (only 15 SIMD attrs total), so the codegen cost of the
unused remainder is modest — the test/doc share is the bigger reason "1,674" looks
alarming.

#### garb ablation study: measured build+link deltas (2026-06-30)

Set up an isolated ablation worktree (`~/work/zen/garb-ablation`, a plain-git copy
of published garb 0.2.8, NOT the real imazen/garb repo — no other-repo rule
violated) + a separate consumer probe crate (`~/work/zen/garb-ablation-consumer`,
depends on it via `path`, zero other deps) to get real numbers instead of
theorizing further. Methodology: `cargo build --release --timings`, min-of-3,
warm dependency cache (only garb's own unit re-measured per step) — same harness
as the compile-time profile above. One methodology bug caught and fixed along
the way: an in-crate `examples/` target silently pulled garb's own dev-deps
(criterion, zenbench) because the vendored `Cargo.toml` (`autoexamples = false`)
initially made the example invisible to cargo, and a naive full-`cargo clean`
before/after each variant repaid the entire proc-macro dependency chain every
time, swamping the signal — moved to a warm-cache delta against a separate probe
crate instead.

**Cumulative ablation (garb's own compile unit, graph's actual feature set: `std,experimental,rgb,imgref`):**

| step | change | time | Δ |
|---|---|---|---|
| A0 | `default,std` only (no experimental) | 0.74 s | — |
| A0′ | **+ graph config** (what zenjpeg's build actually resolves) | **1.18 s** | baseline |
| M1 | − `packed` + `packed_1010102` mods (unused, whole files) | 1.08 s | −0.10 s |
| M2 | − `experimental_imgref` + `experimental_typed` blocks (unused) | 1.02 s | −0.06 s |
| M3 | deinterleave → rgb24-chunk8-forward only (drop rgb48/f32/reverse/chunk4/16) | 0.97 s | −0.05 s |
| **M1+M2+M3 total** | | **0.97 s** | **−0.21 s (−18%)** |

All three cuts landed almost entirely in **frontend** (0.55→0.37 s), barely
touching codegen (0.63→0.60 s) — confirms the earlier claim that ablating unused
*surface area* mainly saves parse/typecheck, not LLVM work.

**M4 attempted and abandoned: bgr/argb/abgr removal is NOT a clean ablation.**
Unlike M1–M3, these kernels are woven through garb's dispatch layers — tier impl
(avx2/scalar/neon/wasm) → row/strided wrappers → public `rgb_to_argb`-style
functions → re-exports referenced from `imgref.rs`/`typed_rgb.rs`. Removing the
concrete functions cascaded through 3 more rounds of dangling-reference errors
across files before being reverted as out of scope for a worktree-copy ablation.
**This is itself the finding for garb's maintainer:** the format-pair kernels
need a real `#[cfg(feature)]` gate authored *in garb* (as recommended above), not
a mechanical deletion — the dispatch web has no clean seam today.

**Codegen floor characterization (opt-level/cgu sweep on the M1+M2+M3 tree):**

| config | time | frontend | codegen |
|---|---|---|---|
| opt=3, cgu=16 (release default) | 0.97 s | 0.37 s | 0.60 s |
| opt=2 | 0.88 s | 0.39 s | 0.49 s |
| opt=1 | 0.71 s | 0.35 s | 0.36 s |
| **opt=0** | **0.40 s** | 0.35 s | **0.05 s** |
| opt=3, cgu=1 | 1.97 s | 0.40 s | 1.57 s (single CGU serializes) |
| opt=3, cgu=256 | 1.02 s | 0.40 s | 0.62 s (no better than 16) |

**The remaining codegen (0.60 s at opt=3) is dominated by optimization work on
the AVX2 kernels, not IR volume** — consistent with the earlier finding that
garb's cost is *function count* (164 non-inlinable `#[target_feature]` fns), not
monomorphization. `cgu=16` (cargo's release default) is already near-optimal;
`cgu=1` is dramatically worse (single-threaded LLVM), `cgu=256` gives nothing
back. Dropping to opt=1 recovers a further ~0.26 s but changes the actual AVX2
codegen quality shipped in the rlib — **not a free lever for a SIMD-perf library.**

**Link cost: negligible, confirmed by direct measurement.** Isolated consumer
probe (garb + tiny `main.rs`, zero other deps), warm-cache delta: BIN
(garb-recompile + link) = 1.04 s vs LIB-only (garb's own unit, same source) =
0.95 s → **link + trivial-binary compile ≈ 0.09 s.** Link is not the bottleneck
anywhere in this picture. A fully-cold build (deps + garb + bin + link from
`cargo clean`) = 3.55 s, but that mostly pays the **shared proc-macro/SIMD-macro
chain** (archmage-macros, syn, magetypes, etc.) once — that cost is amortized
across every crate in a project using archmage, not garb-specific.

**Bottom line vs the "third of a second" target:** cleanly-ablatable dead code
(M1–M3, mechanical and safe) buys **1.18 → 0.97 s (−18%)**. Reaching ~0.33 s
from there requires one of: (a) a real bgr/argb/abgr feature gate authored in
garb (blocked on the dispatch-web refactor identified by the abandoned M4), (b)
dropping optimization level for garb specifically (opt=1 → 0.71 s, opt=0 →
0.40 s — a real runtime-perf tradeoff, not free), or (c) restructuring the
100+ target-feature-gated AVX2 kernels to fewer, more generic dispatch points
(an architectural change, not a config knob). None of these are available as a
zenjpeg-side fix — they're all `imazen/garb` engineering work.

---

# Migrated from CLAUDE.md (2026-07-13)

The following sections were moved here verbatim to keep CLAUDE.md operational.

### Dequantization Bias (2026-02-06)

`Decoder::new().dequant_bias(true)` enables Laplacian dequantization biases
(Price & Rabbani 2000). Computes per-coefficient biases from DCT statistics
and applies them during f32 dequantization. Bypasses fast i16 IDCT path.
Default: off.

**Frymire quality sweep** (1118x1105 photograph, baseline 4:2:0, commit 86e3bef):

| Q | bytes | zenjpeg | zen+bias | cjpegli | zune-jpeg | bias-zen | bias-cpp | maxdif |
|---|-------|---------|----------|---------|-----------|----------|----------|--------|
| 10 | 116K | 5.25 | 1.88 | 1.99 | 5.25 | -3.37 | -0.11 | 1 |
| 20 | 171K | 21.35 | 18.54 | 18.61 | 21.35 | -2.82 | -0.07 | 1 |
| 30 | 219K | 30.59 | 28.57 | 28.59 | 30.59 | -2.03 | -0.03 | 1 |
| 40 | 243K | 34.21 | 32.37 | 32.44 | 34.21 | -1.84 | -0.07 | 1 |
| 50 | 271K | 37.28 | 35.95 | 36.01 | 37.28 | -1.32 | -0.06 | 1 |
| 60 | 309K | 41.07 | 40.07 | 40.10 | 41.07 | -0.99 | -0.03 | 1 |
| 70 | 362K | 45.00 | 44.24 | 44.31 | 45.00 | -0.76 | -0.07 | 1 |
| 80 | 438K | 48.72 | 48.25 | 48.32 | 48.72 | -0.47 | -0.07 | 1 |
| 85 | 494K | 50.45 | 50.18 | 50.21 | 50.45 | -0.27 | -0.03 | 1 |
| 90 | 583K | 51.94 | 51.81 | 51.83 | 51.94 | -0.14 | -0.02 | 1 |
| 95 | 742K | 53.28 | 53.25 | 53.27 | 53.28 | -0.03 | -0.02 | 1 |
| 97 | 848K | 53.71 | 53.68 | 53.73 | 53.71 | -0.03 | -0.05 | 1 |
| 99 | 1034K | 54.00 | 54.03 | 54.07 | 54.00 | +0.03 | -0.05 | 1 |

**CID22 mean** (10 images, 512px, baseline 4:2:0):

| Q | zenjpeg | zen+bias | cjpegli | zune-jpeg | bias-zen | bias-cpp |
|---|---------|----------|---------|-----------|----------|----------|
| 50 | 65.23 | 65.03 | 65.07 | 65.23 | -0.21 | -0.05 |
| 75 | 75.05 | 75.19 | 75.24 | 75.05 | +0.14 | -0.06 |
| 85 | 79.85 | 80.17 | 80.22 | 79.85 | +0.33 | -0.04 |
| 95 | 86.65 | 87.05 | 87.11 | 86.65 | +0.39 | -0.06 |

- `bias-zen`: SSIM2 pt gain over default (positive = better)
- `bias-cpp`: SSIM2 pt gap vs C++ jpegli (negative = C++ better)
- `maxdif`: max pixel diff between zen+bias and cjpegli

**Pairwise SSIMULACRA2** (between decoders, Q85, 6 CID22 images):

| | zenjpeg | zen+bias | cjpegli | zune-jpeg |
|---|---------|----------|---------|-----------|
| zenjpeg | - | 91.39 | 91.42 | 100.00 |
| zen+bias | | - | 94.31 | 91.23 |
| cjpegli | | | - | 91.26 |

**Key findings:**
- zenjpeg default == zune-jpeg (identical output, both integer IDCT)
- zen+bias↔cjpegli similarity: 94.31 vs default↔cjpegli 91.42 (3 pts closer)
- Max pixel diff between zen+bias and cjpegli: always 1 (IDCT rounding only)
- **Image-dependent quality tradeoff**: on CID22 (small, diverse), bias helps
  +0.14 to +0.39 at Q75+. On frymire (large photograph), default integer IDCT
  wins by 0.03-3.37 pts across all qualities. Bias only breaks even at Q99.
- C++ jpegli shows the same pattern: also behind integer IDCT on frymire.
  The f32 IDCT + bias path and integer IDCT path have different rounding
  characteristics; which wins depends on image content.
- bias-cpp gap is consistently tiny (0.02-0.11 pts), confirming zen+bias
  closely matches C++ jpegli decoder behavior regardless of image.

Run: `cargo test --release -p zenjpeg --test dequant_bias_comparison -- --nocapture --ignored`


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

### Linear Iteration for AC Refinement (2026-02-14)

**Attempted:** Replace bitmap-accelerated inner scan loop in `decode_ac_refine` with linear
iteration (k from ss to se), matching zune-jpeg's approach. Goal was to eliminate the
`num_zeros_to_skip < zero_gap` branch that caused 1.08M mispredicts (25.5% of ALL mispredicts).

**Results:** WORSE. Instructions 449M → 469M (+4.4%), mispredicts 4.24M → 7.75M (+83%).

**Why it failed:** Linear iteration visits EVERY position from k to se (~49 positions per block
for band [15,63]), while bitmap visits only nonzero positions (~5-10). Even though individual
branches are more predictable (`coeffs[k] != 0` is 90% false for sparse blocks), the total
branch count is much higher: 49 × ~2.5 branches = ~123 per block vs bitmap's 10 × ~7 = ~70.
The unconditional refinement bit reads in the nonzero case happen the same number of times,
but the zero-position checking adds massive overhead for sparse progressive blocks.

**Conclusion:** Bitmap is fundamentally better for sparse coefficient data. The O(nonzero)
iteration count dominates the per-iteration branch cost.

### Unchecked Bit Reads for AC Refinement (2026-02-14)

**Attempted:** Add `read_bit_unchecked()` (no refill check) with `ensure_n_bits()` pre-fill
before bitmap loops. Save ~2 instructions per bit read by eliminating the `bits_in_buffer == 0`
check in the hot loop.

**Results:** Breaks restart marker handling. 5 test failures including "expected 0xFF for restart
marker" and "invalid Huffman code".

**Why it failed:** Near restart markers, `refill()` returns fewer bits than requested and sets
`marker_found`. The checked `read_bit_refine()` calls `refill()` when buffer empties, which
re-adds zero padding. Unchecked reads consume past the marker boundary. Even with
`saturating_sub` to prevent u8 underflow, the consumed bits corrupt the position for
subsequent Huffman decodes. Safe handling requires tracking available bits vs needed bits per
loop iteration, which adds complexity matching the cost of the original check.

**Conclusion:** The 2-instruction saving per bit read isn't worth the marker boundary complexity.

### Pre-refill AC First Scan (2026-02-15)

**Attempted:** Apply the same `ensure_bits()` + `peek_top(9)` pre-refill pattern (from AC
refine commit 43b24d6) to `decode_ac_first_scan`.

**Results:** Callgrind showed +11% regression (34.8M → 38.6M instructions). AC refine scan
(unchanged code) also regressed +6.4% (162.7M → 173.1M) due to code layout changes from
recompilation. Function is only 2.43% of total — even a 20% improvement saves <0.5%.

**Conclusion:** Not worth pursuing. The function is too small a fraction of total decode
time. Code layout effects from the change outweigh the algorithmic improvement.

### Conditional read_bit_fast in AC Refinement (2026-02-15)

**Attempted:** Add `read_bit_fast()` (no refill check) to bitstream.rs. Use `fast_bits`
boolean in AC refine to choose between `read_bit_fast()` and `read_bit_refine()` per
refinement bit read, based on whether `ensure_bits()` succeeded.

**Results:** All tests passed but callgrind showed 162.7M → 218.7M (+56M, +34%).

**Why it failed:** The per-read `if fast_bits { read_bit_fast() } else { read_bit_refine() }`
branch costs ~2 instructions — exactly the same as the refill check it replaces. Net effect
is zero benefit with added code complexity. The branch predictor handles the refill check
(`bits_in_buffer == 0` is rarely true) just as well as the `fast_bits` check.

**Conclusion:** Cannot eliminate per-bit overhead through branching. Would need fundamentally
different approach (e.g., reading multiple refinement bits in one operation).

### Branchy Coefficient Update in AC Refinement (2026-02-15)

**Attempted:** Replace branchless `c.wrapping_add((bit as i16) * not_set * sign * bit_val)`
with branchy `if bit != 0 && (c & bit_val) == 0 { if c > 0 { +bit_val } else { -bit_val } }`.

**Results:** Callgrind AC refine dropped from 225.2M to 184.5M (-18.1%). But wall-clock
was 5-21% WORSE across all sizes.

**Why it failed:** The `bit != 0` and `c > 0` branches are poorly predicted — coefficient
signs and refinement bits are effectively random. Each misprediction costs ~15 cycles but
counts as only 1 instruction in callgrind. The branchless version has more instructions but
is fully predictable (no branches = no mispredictions). Branch misprediction overhead
dominates instruction-count savings.

**Conclusion:** Callgrind instruction count can be misleading when branch prediction matters.
Branchless is correct for this hot path despite higher instruction count.

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


### Fixed / Resolved Bugs (historical reference)

- **XYB linear-input encoder path saturated to white (FIXED 2026-04-23, commit 9e2348fe)** -
  `EncoderConfig::xyb` + `PixelLayout::RgbF32Linear` or `PixelLayout::Rgb16Linear`
  produced JPEGs that decoded to solid white. The scalar linear-input branch in
  `encode/strip/convert.rs:700` called `linear_rgb_to_xyb_255` (which returns
  UN-scaled XYB for a 0-255-range input) and then multiplied by 255.0 again,
  while the parallel sRGB-input SIMD branch correctly applied
  `scale_xyb(x,y,b) = ((x+offset)*scale)` before the ×255.0 JPEG-range step.
  Fix: linear branch now calls `linear_rgb_to_xyb(r,g,b)` on 0..1 RGB, then
  `scale_xyb()`, then ×255.0 — same pipeline as the sRGB path. Sibling
  `Rgb8Srgb` path unchanged; locked hashes unaffected.
  - Tests: `xyb_linear_matches_srgb_solid_red` et al. in `xyb_roundtrip.rs`.

- **Fused parallel decode bypassed coefficient storage (FIXED 2026-03-31, commit c9b47ec1)** -
  `try_fused_parallel_decode()` didn't check `decode_mode`, so it took the fused parallel
  path (decodes directly to u8 pixels, no coefficient storage) even when
  `DecodeMode::Coefficient` was set. This caused "no decoded data" InternalError for
  `OutputTarget::SrgbF32`, `dequant_bias(true)`, lossless transforms, and Knusperli deblock
  — any path needing coefficient access. Only triggered with `--features parallel` on images
  with DRI (restart markers). Fix: return `Ok(false)` from `try_fused_parallel_decode` when
  `decode_mode == Coefficient`. (`fused_parallel.rs:92`)

- **Progressive decoder truncation near restart markers (FIXED 2026-03-09, commit 08ef601)** -
  Fused `decode_ac_first_scan` and `decode_ac_refine_scan` lacked a bit-by-bit Huffman
  fallback when `peek_bits_refill(16)` failed near restart marker boundaries. When a Huffman
  code > 9 bits occurred in the last 2-3 blocks before a restart marker (0xFF 0xDn), the
  16-bit peek failed because the marker interrupted bitstream refill. The function incorrectly
  treated this as scan truncation, zeroing all remaining AC coefficients. The standard
  `decode_huffman_symbol_lenient` had this fallback but the fused functions did not.
  Triggered at Q91-Q93 (where AC table had codes > 9 bits) with DRI=216 on 576x576 images.
  Fix: added bit-by-bit Huffman decode fallback matching the standard function.
  - Found during investigation of Known Bug #1 (catastrophic auto_optimize quality).
  - Test: `cargo test --release -p zenjpeg --test quality_regression -- diagnostic_coefficient_comparison --nocapture --ignored`

- **Parallel feature skipping deringing (FIXED 2026-03-09)** - `parallel_dct_plane` in
  `encode/parallel.rs` did `extract_block → forward_dct` without applying deringing, while
  the sequential path in `strip/mod.rs:1027-1029` applied `preprocess_deringing_block` before
  DCT. This caused `locked_values` test failures with `--features parallel` — not
  non-determinism, but a deterministic quality regression (deringing silently skipped).
  Fix: pass `deringing: Option<u16>` (dc_quant when enabled) through `parallel_dct_y_blocks`
  into both parallel and sequential DCT plane functions. Deringing is block-local (no
  cross-block dependencies), so it parallelizes trivially.

- **zune-jpeg progressive decode issue (STALE, was Bug #5)** - Originally reported that
  zune-jpeg decoded zenjpeg progressive output as grayscale. Investigation (2026-03-09)
  found 70+ progressive encoding tests pass with zune-jpeg. The AC refinement trailing
  ZRL fix (commit d355648) likely resolved the underlying scan structure issue. The only
  remaining trace is a skip in `chroma_upsample_regression.rs:1038`. Note: zune-jpeg 0.5.12
  still has a separate bug silently skipping AC refinement with DRI (max_diff=224).

- **Grayscale scanline reader panic (FIXED 2026-02-06, commit be24fac)** - Streaming scanline
  reader methods panicked on grayscale images. Fixed by handling 1-component images.

- **XYB 4:2:0 encoder producing undecodable JPEGs (FIXED 2026-03-04, commit b0cafce)** -
  Frequency counter clamped DC categories to 11 (`.min(11)`) but encoder wrote unclamped
  categories. XYB produces DC differences > ±2047 at low quality (categories 12+). Huffman
  table lacked codes for those categories, writing (code=0, len=0) → corrupted bitstream.
  Fix: remove `.min(11)` from `collect_block_frequencies_simd`. Previously-encoded files
  in `testdata/decode_failures/` remain permanently corrupted — tests converted to verify
  graceful rejection (assert decode error, not success).
  - Test: `cargo test --release -p zenjpeg --test xyb_roundtrip`
  - Test: `cargo test --release -p zenjpeg --test decode_xyb_failures`

- **CMYK scanline transform panic (FIXED 2026-03-04, commit bde9f48)** -
  `scanline_reader_with_transform()` had no CMYK check. Non-dimension-swapping transforms
  (e.g., FlipHorizontal) fell through to `from_coefficients()` → `StripProcessor` with
  `[u8; 3]` arrays → index-out-of-bounds at `h_samp[3]`. Fix: route CMYK to buffered
  decode fallback, matching `scanline_reader()`.
  - Test: `cargo test --release -p zenjpeg --test cmyk_transform`

- **False XYB ICC detection for cjpegli JPEGs (FIXED 2026-02-14, commit 744d38a)** -
  `is_xyb_profile()` checked for "jxl " CMM type (bytes 4-7) in ICC profiles, but cjpegli
  writes "jxl " for ALL ICC profiles (including standard sRGB), not just XYB ones. This caused
  every cjpegli JPEG with an ICC profile to be misidentified as XYB, bypassing the fast i16
  decode path and falling through to the f32 XYB→RGB conversion — producing completely wrong
  colors (max_diff=252). Fix: replace "jxl " CMM check with exact-match against the known
  720-byte XYB ICC profile, falling back to "XYB" text search in the profile description.
  Also affected baseline streaming and fused parallel paths (would have returned "no decoded
  data" error for cjpegli images).

- **4:2:0 scanline chroma upsampling at MCU bottom boundaries (FIXED 2026-02-09, commit bd0f8d7)** -
  Bilinear chroma upsampler used edge replication at MCU row bottom boundaries (max ~43
  pixel error for streaming, ~57 for coefficient/transform path). Fix: mirror the existing
  top-boundary fixup for the bottom edge. Coefficient path peeks ahead by IDCT'ing the first
  chroma block row of the next MCU. Streaming path pre-decodes the next MCU row and serves
  corrected chroma through deferred buffers. Boundary max diff now ≤4 (IDCT rounding only).

- **Scanline h2v2 boundary fixup buffer overflow (FIXED 2026-02-09, commit 8f1295f)** -
  `fixup_h2v2_row0()` used hardcoded `[i16; 4096]` stack buffers, panicking on any 4:2:0
  image wider than 8192px (chroma width > 4096). Fix: borrow disjoint struct fields directly
  instead of copying to temp buffers. Closes #1.
- **Progressive MCU-padded storage (FIXED 2026-02-09, commit 29d6d81)** - Progressive decoder
  allocated coefficients with component-based counts (ceil(scaled_w/8)) but output path reads
  with MCU-padded stride (mcu_cols * h_samp). For 4:2:0 with non-MCU-aligned width, caused
  1-block-per-row shift accumulating to max_diff=255. Affected ~20/543 web corpus files.
- **Progressive interleaved DC scan padding (FIXED 2026-02-09, commit 759a4a7)** - Skipping
  entropy data for out-of-bounds MCU padding blocks desynchronized Huffman decoder. Caused
  "invalid Huffman code" parse errors on 80/543 progressive 4:2:0 files.
- See `docs/TUNING_HISTORY.md` for older fixed bugs.

