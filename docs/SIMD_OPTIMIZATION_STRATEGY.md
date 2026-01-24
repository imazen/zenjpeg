# SIMD Optimization Strategy for jpegli-rs

## Current Performance (After Optimizations)

Benchmarks with `-C target-cpu=native` on Rust and `-march=native` on C++ (2K flower.png):

| Mode | Δ Time | Notes |
|------|--------|-------|
| YUV/SEQ/OPT/420 | -5.3% | **Rust is faster!** |
| YUV/PRO/OPT/444 | +10% | Near parity |
| YUV/SEQ/FIX/444 | +72% | Still work to do |
| Overall | +40% | Improved from ~47% |

**Key achievements:**
- YUV 4:2:0 mode is now 5% faster than C++
- Progressive 4:4:4 is within 10% of C++
- AVX2 8x8 transpose is efficient (only 1% of profile time)

## Completed Optimizations

### 1. ✅ AVX2 8x8 Transpose (Implemented)

Used Highway's algorithm with `vunpcklps/vunpckhps` + `vperm2f128`:
- Transpose now takes only 1% of profile time
- Used multiversion macro for efficient dispatch

### 2. ✅ gather_even_odd_x8 (Implemented)

Replaced element-by-element construction with shuffle-based deinterleave.

### 3. ✅ AVX2+FMA Color Conversion (Implemented)

Added FMA to rgb_to_ycbcr_8px_avx2 for faster color space conversion.

## Findings: What Didn't Work

### Parallel SIMD DCT
Attempted to process 8 rows simultaneously but found it **slower** than scalar:
- `f32x8::from([...])` generates 8x `vinsertps` instructions (slow!)
- Extra transpose overhead negated any SIMD benefit
- Scalar row-by-row with AVX2 transpose is actually faster

**Lesson:** The `wide` crate's vector construction is a bottleneck. True AVX2 optimization requires raw intrinsics throughout.

## Remaining Bottlenecks (Profile Data)

For baseline 4:4:4 encoding (2K image):

| Component | % of Time |
|-----------|-----------|
| DCT | 14% |
| Adaptive Quantization | 17% |
| Huffman table optimization | 6% |
| Memory operations | 7% |
| Color conversion | 4% |

### Next Steps
1. **DCT with raw AVX2 intrinsics** - Eliminate wide crate overhead
2. **AQ optimization** - Largest remaining target at 17%
3. **Reduce memory allocations** - memset/memmove at 7%

## Historical: Original Root Causes

### 1. Element-by-Element Vector Construction (Critical)

**Location:** `zenjpeg/src/encode_simd.rs:633`

**Problem:**
```rust
let evens = f32x8::from([a[0], a[2], a[4], a[6], b[0], b[2], b[4], b[6]]);
```

Generates 8x `vinsertps` instructions instead of bulk load + shuffle.

**C++ Highway equivalent:**
```cpp
LoadInterleaved2(d, row_in + 2 * x, v0, v1);  // 2 loads + 2 shuffles
```

**Fix:** Use `std::arch` intrinsics:
```rust
use std::arch::x86_64::*;
unsafe {
    let v0 = _mm256_loadu_ps(ptr);
    let v1 = _mm256_loadu_ps(ptr.add(8));
    // Use vpermps/vshufps to deinterleave
}
```

### 2. DCT Row-by-Row Processing

**Location:** `zenjpeg/src/dct.rs:411-424`

**Problem:**
```rust
for row in 0..8 {
    let mut tmp = [0.0f32; 8];
    for i in 0..8 { tmp[i] = input[row * 8 + i]; }  // copy in
    dct1d_8(&mut tmp);
    for i in 0..8 { output[row * 8 + i] = tmp[i]; } // copy out
}
```

**C++ Highway:** Processes all 8 rows simultaneously with data in YMM registers.

**Fix:** Implement column-major SIMD DCT where each YMM register holds one coefficient position from all 8 rows.

### 3. No FMA (Fused Multiply-Add)

**Problem:** `wide` crate's `a * b + c` generates separate `vmulps` + `vaddps`.

**C++ Highway:** `MulAdd(a, b, c)` generates single `vfmadd231ps`.

**Fix:** Use `_mm256_fmadd_ps` for multiply-add chains in DCT and color conversion.

### 4. Transpose Uses Scalar Scatter

**Location:** `zenjpeg/src/dct.rs:381-403`

**Problem:**
```rust
for col in 0..8 {
    output[col * 8 + 0] = a0[col];
    output[col * 8 + 1] = a1[col];
    // ...
}
```

**Fix:** Use AVX2 8x8 transpose with `vunpcklps`, `vunpckhps`, `vperm2f128`.

## Implementation Plan

### Phase 1: Hot Path Intrinsics (High Impact)

1. **`gather_even_odd_x8`** - Replace with shuffle-based deinterleave
2. **`downsample_2x2_simd_inplace`** - Use bulk loads + shuffles
3. **Color conversion loops** - Add FMA support

### Phase 2: DCT Optimization (Medium Impact)

1. Implement 8-wide column-major DCT
2. SIMD-native 8x8 transpose
3. Fuse DCT + quantization where possible

### Phase 3: Memory Layout (Lower Impact)

1. Ensure 32-byte alignment for AVX2 loads/stores
2. Reduce temporary allocations in hot paths
3. Consider cache-line-aware block processing

## Reference Implementation

C++ Highway patterns to port from `internal/jpegli-cpp/lib/jpegli/`:

| File | Function | Priority |
|------|----------|----------|
| `downsample.cc` | `DownsampleRow2x1` | High |
| `dct-inl.h` | `DCT1DImpl<8>` | High |
| `color_transform.cc` | `RGBToYCbCr` | Medium |
| `encode_streaming.cc` | `QuantizeBlock` | Medium |

## Verification

After each change:
```bash
# Correctness
cargo test --release -p zenjpeg

# Parity with C++
cargo test --release -p zenjpeg --features ffi-tests --test cpp_parity_locked
cargo run --release --features cms --example cpp_parity_matrix

# Performance (Rust-only, for optimization tracking)
RUSTFLAGS="-C target-cpu=native" cargo run --release --example comprehensive_bench
```

## Success Criteria

- Match C++ performance within 10% on 4K images
- Maintain bit-exact output for YCbCr modes
- No regression on small images
