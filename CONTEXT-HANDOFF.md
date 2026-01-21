# Context Handoff: Archmage SIMD & Performance Profiling

**Date:** 2026-01-20
**Branch:** `feat/remove-deprecated-encoder`
**Last Commit:** `4d3ce27` - feat(archmage-simd): add AQ SIMD functions and DCT dispatch

## Summary

Profiled Rust jpegli vs C++ jpegli to identify performance gaps. Created archmage-based SIMD functions but discovered the **real bottleneck is memory copying, not SIMD instruction choice**.

## Performance Gap

**Benchmark (512x512 images):**

| Quality | Rust | C++ FFI | Ratio |
|---------|------|---------|-------|
| q50 | 2.44ms | 1.46ms | 1.67x |
| q75 | 2.60ms | 1.57ms | 1.66x |
| q90 | 2.78ms | 1.73ms | 1.61x |
| q95 | 3.10ms | 1.91ms | 1.62x |

With `-C target-cpu=native` (enables AVX2/FMA in wide crate).

## Root Cause: Memory Copying (NOT SIMD)

**Cachegrind analysis** (3840x2160 image, 10 iterations):

| Source | D1 Write Miss % | D1 Read Miss % | Notes |
|--------|-----------------|----------------|-------|
| **memcpy** | 30.2% | 7.9% | Buffer rotations |
| **memset** | 14.5% | - | Buffer zeroing |
| **pre_erosion_row** | - | 16.2% | Sliding window |
| **downsample_2x2** | - | 12.3% | Chroma downsampling |
| **yuv crate RGB→YUV** | - | **68.6% LL** | Last-level misses! |

**Key insight:** memcpy + memset = 45% of L1 data cache write misses

### Where the copies happen

`jpegli-rs/src/quant/aq/streaming.rs`:
```rust
// Every row:
self.row_curr.copy_from_slice(row);           // line 364
self.row_prev_prev.copy_from_slice(&self.row_prev);  // line 371
self.pre_erosion_temp.fill(0.0);              // lines 395, 408

// Every 4 rows:
self.pre_erosion_accum.fill(0.0);             // line 439
```

C++ jpegli likely uses **pointer swapping** instead of memcpy for buffer rotation.

## What Was Implemented

### 1. Archmage AQ SIMD Functions (`jpegli-rs/src/encode/mage_simd.rs`)

Created AVX2+FMA implementations using archmage tokens:

```rust
// Ratio of derivatives (used in pre-erosion and per-block modulations)
pub fn mage_ratio_of_derivatives_x8<T: HasAvx2 + HasFma>(token: T, vals: __m256) -> __m256
pub fn mage_ratio_of_derivatives_inv_x8<T: HasAvx2 + HasFma>(token: T, vals: __m256) -> __m256

// Masking sqrt
pub fn mage_masking_sqrt_x8<T: HasAvx2 + HasFma>(token: T, v: __m256) -> __m256

// Pre-erosion pixel computation
pub fn mage_pre_erosion_pixel_x8<T: HasAvx2 + HasFma + Copy>(
    token: T, pixels: __m256, left: __m256, right: __m256, top: __m256, bottom: __m256
) -> __m256

// Block modulation sums
pub fn mage_hf_modulation_sum_8x8<T: HasAvx2 + HasFma + Copy>(...) -> f32
pub fn mage_gamma_modulation_sum_8x8<T: HasAvx2 + HasFma + Copy>(...) -> f32

// Fast math
pub fn mage_fast_exp2_x8<T: HasAvx2 + HasFma>(token: T, x: __m256) -> __m256
pub fn mage_fast_log2_x8<T: HasAvx2 + HasFma>(token: T, x: __m256) -> __m256
```

**Status:** Implemented and tested, but **NOT wired into streaming AQ** because the existing code uses `wide::f32x8` throughout, and the benefit would be minimal given the memory bottleneck.

### 2. Archmage DCT Dispatch (`jpegli-rs/src/encode/strip/mod.rs`)

Wired up archmage DCT with token stored in StripProcessor:

```rust
#[cfg(all(feature = "archmage-simd", target_arch = "x86_64"))]
simd_token: Option<crate::encode::mage_simd::Desktop64>,

// Dispatch helper
fn forward_dct_dispatch(
    token: Option<Desktop64>,
    block: &Block8x8f,
) -> Block8x8f {
    if let Some(t) = token {
        return mage_forward_dct_8x8_wide(t, block);
    }
    crate::encode::dct::simd::forward_dct_8x8_wide(block)
}
```

### 3. Bytemuck for Block8x8f (`jpegli-rs/src/foundation/simd_types.rs`)

Added safe type conversions:
```rust
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C, align(32))]
pub struct Block8x8f { ... }
```

## Recommended Next Steps

### Priority 1: Fix Memory Copying in Streaming AQ

Replace `copy_from_slice` buffer rotations with pointer/index swapping:

```rust
// Current (slow):
self.row_prev_prev.copy_from_slice(&self.row_prev);
self.row_prev.copy_from_slice(&self.row_curr);
self.row_curr.copy_from_slice(row);

// Better: swap buffer indices/pointers
std::mem::swap(&mut self.row_prev_prev, &mut self.row_prev);
std::mem::swap(&mut self.row_prev, &mut self.row_curr);
// Then write directly into row_curr
```

Or use a ring buffer with index rotation instead of three separate buffers.

### Priority 2: Investigate yuv Crate

The `yuv` crate's RGB→YUV conversion has **68% of last-level cache misses**. Options:
1. Profile the yuv crate separately to understand the issue
2. Consider using the archmage `mage_rgb_to_ycbcr_8px` function (already implemented)
3. Check if the yuv crate has better-optimized code paths

### Priority 3: Reduce Buffer Zeroing

The `fill(0.0)` calls are expensive. Consider:
- Lazy zeroing (only zero elements that will be read)
- Reuse buffers without zeroing when safe
- Use `MaybeUninit` for scratch buffers

## Files Changed This Session

```
jpegli-rs/Cargo.toml                   |   5 +   (archmage-simd feature)
jpegli-rs/benches/mage_simd.rs         | 227 +++ (new benchmark)
jpegli-rs/src/encode/dct.rs            |   2 +-
jpegli-rs/src/encode/mage_simd.rs      | 694 +++ (AQ SIMD functions)
jpegli-rs/src/encode/strip/mod.rs      |  65 +  (DCT dispatch)
jpegli-rs/src/foundation/simd_types.rs |   9 +  (bytemuck derives)
```

## How to Run Benchmarks

```bash
# Basic encode benchmark
cargo bench --bench encode -p jpegli-rs@0.9.0 --features archmage-simd -- quality/q/

# C++ comparison (requires jpegli-cpp build)
cargo bench --bench cpp_comparison -p jpegli-rs@0.9.0 --features archmage-simd

# With native CPU features
RUSTFLAGS="-C target-cpu=native" cargo bench ...

# Cache profiling (valgrind)
valgrind --tool=cachegrind ../target/release/examples/flamegraph_profile 512
cg_annotate cachegrind.out.* --auto=yes | head -100
```

## CPU Info

AMD Ryzen 9 7950X with AVX-512 support:
- avx512f, avx512dq, avx512bw, avx512vl, avx512cd
- avx2, fma, bmi1, bmi2

The `wide` crate only uses AVX2 (no AVX-512). The `#[multiversed]` macro dispatches to AVX-512 for outer loop functions, but inner helpers use `wide::f32x8`.

## Tests

All 12 mage_simd tests pass:
```bash
cargo test --release -p jpegli-rs@0.9.0 --features archmage-simd --lib mage_simd
```

## Key Insight

**Don't optimize SIMD instructions until the memory copying is fixed.** The 1.6x performance gap is primarily due to:
1. Buffer rotation via memcpy instead of pointer swapping
2. Excessive buffer zeroing
3. Poor cache behavior in the yuv crate

The archmage SIMD functions are ready but their benefit will be minimal until memory access patterns are improved.
