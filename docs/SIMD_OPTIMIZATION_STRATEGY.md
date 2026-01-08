# SIMD Optimization Strategy for jpegli-rs

## Current Performance Gap

Benchmarks with `-march=native` on both Rust and C++ show:

| Image Size | Rust vs C++ |
|------------|-------------|
| 512px | Rust 20-42% faster |
| 2K | Rust 20-100% slower |
| 4K | Rust 46-132% slower |

**Key insight:** Rust wins at small sizes (better setup), C++ wins at large sizes (better SIMD throughput).

## Root Causes Identified

### 1. Element-by-Element Vector Construction (Critical)

**Location:** `jpegli-rs/src/encode_simd.rs:633`

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

**Location:** `jpegli-rs/src/dct.rs:411-424`

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

**Location:** `jpegli-rs/src/dct.rs:381-403`

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
cargo test --release -p jpegli-rs

# Parity with C++
cargo test --release -p jpegli-rs --features ffi-tests --test cpp_parity_locked

# Performance
RUSTFLAGS="-C target-cpu=native" cargo run --release --example cpp_timing_matrix -- --iterations 10
```

## Success Criteria

- Match C++ performance within 10% on 4K images
- Maintain bit-exact output for YCbCr modes
- No regression on small images
