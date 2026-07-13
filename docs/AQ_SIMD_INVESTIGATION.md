# AQ SIMD Investigation: Closing the C++ Performance Gap

> **Historical record (2026-01/02).** The `archmage-simd` cargo feature
> referenced throughout no longer exists — archmage/magetypes SIMD became
> a mandatory dependency (always compiled; there is no toggle). Numbers
> here predate that change and later tuning; consult `docs/TUNING_HISTORY.md`
> and the repo-root `CLAUDE.md` for current performance state.

## Executive Summary

**Problem:** Rust zenjpeg AQ functions consume 24% of encode time vs C++ jpegli's 8% (3x relative overhead).

**Current state (2026-01-21):**
- Default build (wide only): 21.2ms, 73.8 MP/s
- With `archmage-simd` feature: 19.8ms, 79.0 MP/s (**6.6% faster**)
- C++ jpegli target: ~12ms
- **Gap: ~1.65x slower than C++** (improved from 1.6x)

**Root cause identified:**
- `wide` crate uses `cfg(target_feature)` (compile-time check)
- `#[multiversed]` dispatch is useless - all versions use SSE-level code
- Solution: `archmage` with token-based runtime AVX2+FMA dispatch

**Optimizations implemented:**
1. `pre_erosion_row_padded` - raw pointer loads, hoisted constants (2.3x faster isolated)
2. `per_block_modulations_row` - fused HF+gamma loop (halves memory traffic)

**For binary distribution:** Enable `archmage-simd` feature. Provides proper runtime
CPU detection via `Avx2FmaToken::try_new()`.

**Goal:** Match C++ AQ performance by understanding and replicating Highway's SIMD strategy.

---

## Part 1: C++ jpegli AQ Code Locations

### Source Files

```
internal/jpegli-cpp/lib/jpegli/
├── adaptive_quantization.cc    # Main AQ implementation
├── adaptive_quantization.h     # AQ public interface
└── enc.cc                      # Calls ComputeAdaptiveQuantField

internal/jpegli-cpp/lib/jxl/
├── enc_adaptive_quantization.cc  # Core AQ algorithms (shared with JXL)
├── enc_adaptive_quantization.h
├── butteraugli/
│   └── butteraugli.cc          # Perceptual distance (used by AQ)
└── gauss_blur.cc               # Gaussian blur for AQ preprocessing
```

### Key Functions to Analyze

1. **ComputeAdaptiveQuantField** - Entry point, orchestrates AQ
2. **GaborishInverse** - Pre-erosion/sharpening filter
3. **PerBlockModulations** - Per-block quality adjustment
4. **ComputeMask** - Masking for perceptual weighting
5. **DiffPrecompute** - Difference computation for HF modulation

### How to Read Highway SIMD Code

Highway uses a macro-based approach for portable SIMD:

```cpp
#include "hwy/highway.h"
HWY_NAMESPACE_BEGIN

// Highway vector types - width determined at compile/runtime
using D = HWY_CAPPED(float, 16);  // Up to 16 floats (AVX-512)
using V = Vec<D>;

// Operations look like function calls but compile to intrinsics
V result = Add(Mul(a, b), c);     // FMA if available
V loaded = LoadU(d, ptr);          // Unaligned load
StoreU(result, d, out_ptr);        // Unaligned store

HWY_NAMESPACE_END
```

**Key Highway features:**
- `HWY_CAPPED(T, N)` - Vector of up to N elements of type T
- `LoadU`/`StoreU` - Unaligned load/store
- `MulAdd(a, b, c)` - Fused multiply-add: a*b + c
- `IfThenElse(mask, yes, no)` - Blend based on mask
- `TableLookupLanes` - Permute/shuffle

---

## Part 2: Rust AQ Code Locations

### Source Files

```
zenjpeg/src/quant/aq/
├── mod.rs              # AQ module root
├── simd.rs             # SIMD implementations (wide crate)
├── streaming.rs        # Streaming AQ processor
└── tables.rs           # Precomputed tables

zenjpeg/src/encode/
├── strip.rs            # Strip processor (calls AQ)
└── deringing.rs        # Deringing preprocessing
```

### Key Functions

1. **`pre_erosion_row`** - `simd.rs:~200` - Pre-erosion filter
2. **`per_block_modulations_row`** - `simd.rs:~400` - Block modulations
3. **`hf_modulation_sum_8x8`** - `simd.rs:~500` - HF energy calculation
4. **`finalize_imcu_aq_with_buffer`** - `streaming.rs` - Final AQ assembly

### Current SIMD Strategy (wide crate)

```rust
use wide::f32x8;

// wide crate uses cfg(target_feature) - compile-time check
// Does NOT benefit from #[target_feature] or multiversed runtime dispatch
let a = f32x8::from(slice);
let b = f32x8::from(other);
let result = a * b + c;  // May or may not use FMA
```

**Problem:** `wide` checks features at compile time, not runtime. Without `-C target-cpu=x86-64-v3+`, it uses SSE fallbacks.

---

## Part 3: Disassembly Instructions

### Disassembling C++ jpegli (Highway)

```bash
cd /home/lilith/work/zenjpeg/internal/jpegli-cpp

# Build with specific target (default is usually native)
mkdir -p build-v3 && cd build-v3
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-march=x86-64-v3" ..
ninja jpegli-static

# Build with AVX-512
mkdir -p build-v4 && cd build-v4
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-march=x86-64-v4" ..
ninja jpegli-static

# Disassemble specific function
objdump -d -C --no-show-raw-insn build-v4/lib/libjpegli-static.a | \
    grep -A 200 "ComputeAdaptiveQuantField" | head -250

# Or use llvm-objdump for better output
llvm-objdump -d --demangle build-v4/lib/libjpegli-static.a | \
    grep -A 200 "PerBlockModulations" | head -250

# Find all SIMD instructions used
objdump -d build-v4/lib/libjpegli-static.a | grep -E "vmov|vadd|vmul|vfma|vperm" | \
    cut -f3 | sort | uniq -c | sort -rn | head -30
```

### Disassembling Rust (cargo asm)

```bash
cd /home/lilith/work/zenjpeg

# Install cargo-asm if needed
cargo install cargo-asm

# View assembly for specific function (default target)
cargo asm -p zenjpeg --lib "zenjpeg::quant::aq::simd::pre_erosion_row" 2>/dev/null | head -100

# With x86-64-v3 (AVX2 + FMA)
RUSTFLAGS="-C target-cpu=x86-64-v3" cargo asm -p zenjpeg --lib \
    "zenjpeg::quant::aq::simd::pre_erosion_row" 2>/dev/null | head -100

# With x86-64-v4 (AVX-512)
RUSTFLAGS="-C target-cpu=x86-64-v4" cargo asm -p zenjpeg --lib \
    "zenjpeg::quant::aq::simd::pre_erosion_row" 2>/dev/null | head -100

# Count register usage (xmm=128-bit, ymm=256-bit, zmm=512-bit)
RUSTFLAGS="-C target-cpu=x86-64-v4" cargo asm -p zenjpeg --lib \
    "zenjpeg::quant::aq::simd::pre_erosion_row" 2>/dev/null | \
    grep -oE "(xmm|ymm|zmm)[0-9]+" | sort | uniq -c | sort -rn

# List all functions matching pattern
cargo asm -p zenjpeg --lib 2>&1 | grep -i "aq\|erosion\|modulation"
```

### Comparing Instruction Mix

```bash
# Script to compare instruction profiles
compare_asm() {
    local func=$1
    echo "=== Rust default ==="
    cargo asm -p zenjpeg --lib "$func" 2>/dev/null | \
        grep -oE "^[[:space:]]+[a-z]+" | sort | uniq -c | sort -rn | head -15

    echo ""
    echo "=== Rust x86-64-v4 ==="
    RUSTFLAGS="-C target-cpu=x86-64-v4" cargo asm -p zenjpeg --lib "$func" 2>/dev/null | \
        grep -oE "^[[:space:]]+[a-z]+" | sort | uniq -c | sort -rn | head -15
}

compare_asm "zenjpeg::quant::aq::simd::pre_erosion_row"
```

---

## Part 4: Expected Instruction Patterns

### AVX-512 (x86-64-v4) Expectations

For AQ functions processing f32 data:

| Operation | Expected Instruction | Latency | Throughput |
|-----------|---------------------|---------|------------|
| Load 16 floats | `vmovups zmm, [mem]` | 7 | 0.5 |
| Store 16 floats | `vmovups [mem], zmm` | 4 | 1 |
| Multiply | `vmulps zmm, zmm, zmm` | 4 | 0.5 |
| Add | `vaddps zmm, zmm, zmm` | 4 | 0.5 |
| FMA | `vfmadd213ps zmm, zmm, zmm` | 4 | 0.5 |
| Min/Max | `vminps/vmaxps zmm, zmm, zmm` | 4 | 0.5 |
| Blend | `vblendmps zmm{k}, zmm, zmm` | 1 | 0.5 |
| Permute | `vpermps zmm, zmm, zmm` | 3 | 1 |

### AVX2 (x86-64-v3) Expectations

| Operation | Expected Instruction | Latency | Throughput |
|-----------|---------------------|---------|------------|
| Load 8 floats | `vmovups ymm, [mem]` | 7 | 0.5 |
| Store 8 floats | `vmovups [mem], ymm` | 4 | 1 |
| Multiply | `vmulps ymm, ymm, ymm` | 4 | 0.5 |
| Add | `vaddps ymm, ymm, ymm` | 4 | 0.5 |
| FMA | `vfmadd213ps ymm, ymm, ymm` | 4 | 0.5 |

### Red Flags in Assembly

**Bad patterns to look for:**

1. **Scalar operations in hot loops:**
   ```asm
   vmovss xmm0, [rax]      ; scalar load (bad)
   ; should be: vmovups ymm0, [rax] or zmm0
   ```

2. **Missing FMA (separate mul+add):**
   ```asm
   vmulps ymm0, ymm1, ymm2
   vaddps ymm0, ymm0, ymm3  ; should be vfmadd
   ```

3. **Unnecessary shuffles/extracts:**
   ```asm
   vextractf128 xmm1, ymm0, 1  ; extracting half (bad for throughput)
   ```

4. **Register spills:**
   ```asm
   vmovaps [rsp+0x40], ymm0   ; spilling to stack (too many live values)
   ```

5. **SSE in AVX code (causes transitions):**
   ```asm
   movaps xmm0, [rax]   ; legacy SSE encoding
   vmovaps ymm1, [rbx]  ; VEX encoding - mixing is bad
   ```

---

## Part 5: Profiling Methodology

### Accurate Benchmarking Setup

```bash
# 1. Disable CPU frequency scaling
sudo cpupower frequency-set -g performance

# 2. Check current frequency
cat /proc/cpuinfo | grep "cpu MHz" | head -1

# 3. Disable turbo (optional, for consistency)
echo 1 | sudo tee /sys/devices/system/cpu/intel_pturbo/no_turbo

# 4. Pin to single CPU core
taskset -c 0 cargo bench ...
```

### Benchmark Commands

```bash
# FFI comparison (most accurate - same data, no I/O)
RUSTFLAGS="-C target-cpu=x86-64-v4" cargo bench -p zenjpeg --bench cpp_comparison

# Profile with cjpegli-compatible settings
RUSTFLAGS="-C target-cpu=x86-64-v4" cargo run --release -p zenjpeg \
    --example cjpegli_rs_profile -- IMAGE.png -p 0 --num_reps 50

# Flamegraph + perf report
RUSTFLAGS="-C target-cpu=x86-64-v4" cargo flamegraph --release -p zenjpeg \
    --example cjpegli_rs_profile -- IMAGE.png -p 0 --num_reps 50
perf report --stdio --no-children -g none --percent-limit 0.5 2>/dev/null
```

### Test Images

```bash
# 512x512 (FFI benchmark default)
# Built into cpp_comparison benchmark

# 3MP test image (good for profiling)
~/work/codec-eval/codec-corpus/clic2025/validation/097cb426910ba8ce2525dd8bb7fb1777.png
# 1507x2048, RGB, non-interlaced

# CID22 validation images
~/work/codec-eval/codec-corpus/CID22/CID22-512/validation/*.png
```

---

## Part 6: Theory of Performance Gap

### Hypothesis 1: Vector Width

**C++ Highway:** Uses runtime dispatch to select optimal width (AVX-512 zmm when available).

**Rust wide:** Uses `cfg(target_feature)` compile-time check. Without global `-C target-cpu`, defaults to SSE/xmm even on AVX-512 capable CPUs.

**Test:**
```bash
# Check register usage
RUSTFLAGS="-C target-cpu=x86-64-v4" cargo asm ... | grep -c "zmm"
# vs
cargo asm ... | grep -c "zmm"  # Should be 0 without flag
```

### Hypothesis 2: FMA Usage

**C++ Highway:** Explicitly uses `MulAdd()` which compiles to `vfmadd*`.

**Rust wide:** `a * b + c` may not fuse without optimization hints.

**Test:**
```bash
cargo asm ... | grep -c "vfmadd"
# vs
cargo asm ... | grep -E "vmulps|vaddps" | wc -l
```

### Hypothesis 3: Memory Access Patterns

**C++ Highway:** May use prefetch hints, aligned loads when possible.

**Rust wide:** All loads unaligned, no explicit prefetch.

**Test:** Look for `prefetch` instructions in C++ disassembly.

### Hypothesis 4: Loop Unrolling

**C++ Highway:** Highway may unroll loops more aggressively.

**Rust:** LLVM unrolling may differ.

**Test:** Compare loop structure in assembly.

---

## Part 7: Implementation Options

### Option A: Raw Intrinsics with Runtime Dispatch

```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[target_feature(enable = "avx512f")]
unsafe fn pre_erosion_row_avx512(...) {
    // Hand-written AVX-512 intrinsics
    let v = _mm512_loadu_ps(ptr);
    let result = _mm512_fmadd_ps(a, b, c);
    _mm512_storeu_ps(out, result);
}

// Runtime dispatch
fn pre_erosion_row(...) {
    if is_x86_feature_detected!("avx512f") {
        unsafe { pre_erosion_row_avx512(...) }
    } else if is_x86_feature_detected!("avx2") {
        unsafe { pre_erosion_row_avx2(...) }
    } else {
        pre_erosion_row_scalar(...)
    }
}
```

### Option B: archmage Crate

```rust
use archmage::{arcane, Desktop64, HasAvx2, HasFma, mem::avx};

#[arcane]
fn pre_erosion_row<T: HasAvx2 + HasFma>(token: T, ...) {
    // Safe intrinsics with token proving CPU support
    let v = avx::_mm256_loadu_ps(token, slice);
    let result = _mm256_fmadd_ps(a, b, c);  // Safe inside #[arcane]
    avx::_mm256_storeu_ps(token, out, result);
}
```

### Option C: Compile with Global Target

```bash
# Require AVX2+FMA at runtime (Haswell 2013+, Zen 1+)
RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release

# Require AVX-512 at runtime (Skylake-X 2017+, Zen 4+)
RUSTFLAGS="-C target-cpu=x86-64-v4" cargo build --release
```

---

## Part 8: Specific Functions to Optimize

Priority order (by profile % in sequential mode):

1. **`finalize_imcu_aq_with_buffer`** - 11.6%
   - Location: `streaming.rs`
   - Likely memory-bound, check for cache efficiency

2. **`per_block_modulations_row`** - 6.9%
   - Location: `simd.rs:~400`
   - Contains: HF modulation, gamma modulation, masking
   - Key inner function: `hf_modulation_sum_8x8`

3. **`pre_erosion_row`** - 6.1%
   - Location: `simd.rs:~200`
   - Stencil operation: center vs 4 neighbors
   - Good candidate for explicit SIMD

### C++ Reference for Each

```bash
# Find C++ equivalents
grep -rn "PreErosion\|GaborishInverse" internal/jpegli-cpp/lib/
grep -rn "PerBlockModulations\|BlockModulations" internal/jpegli-cpp/lib/
grep -rn "ComputeMask\|HfModulation" internal/jpegli-cpp/lib/
```

---

## Part 9: Validation

After optimization, verify correctness:

```bash
# Run parity tests
cargo test --release -p zenjpeg --test comprehensive_cpp_comparison -- --nocapture --ignored

# Check output matches
cargo run --release --example cjpegli_rs_profile -- test.png out_rust.jpg -p 0 -d 1.0
cjpegli test.png out_cpp.jpg -p 0 -d 1.0
# Compare: should be byte-identical or very close
```

---

## Appendix: Quick Reference

### Profile Commands (justfile)

```bash
just profile              # Default image, 50 iterations
just flamegraph           # With perf flamegraph
just bench-cpp            # FFI comparison benchmark
just parity               # C++ parity test
```

### Key Metrics

| Metric | Current | Target |
|--------|---------|--------|
| AQ % of encode time | 24% | 8% (match C++) |
| Rust/C++ ratio (v4) | 1.33x | 1.0x |
| Rust/C++ ratio (default) | 1.6x | - |

### Files to Modify

- `zenjpeg/src/quant/aq/simd.rs` - Main SIMD implementations
- `zenjpeg/src/quant/aq/streaming.rs` - Streaming processor
- `zenjpeg/Cargo.toml` - Add unsafe_simd or archmage-simd feature

---

## Part 10: Disassembly Comparison Results (2026-01-20)

### Register Usage Comparison

**C++ jpegli AQ (adaptive_quantization.cc.o):**
```
    217 xmm (128-bit)
    147 ymm (256-bit)
     45 zmm (512-bit)
```

**Rust jpegli AQ (pre_erosion_row AVX-512 version):**
```
     52 ymm (256-bit)
     20+ xmm (128-bit)
      0 zmm (512-bit)
```

### Key Finding: Both Use 256-bit in Hot Loops

Despite having AVX-512 available, both implementations primarily use ymm (256-bit) registers
in the AQ hot loops. This is because:

1. **C++ Highway:** Uses `HWY_CAPPED(float, 8)` for 8x8 block operations
2. **Rust wide:** Uses `f32x8` which is always 8 floats

The zmm usage in C++ is for data movement (vmovdqa32 zmm), not computation.

### FMA Instruction Count

| Implementation | FMA Count |
|----------------|-----------|
| C++ AQ total | 41 |
| Rust pre_erosion_row | 9 |
| Rust per_block_modulations_row | 10 |
| **Rust AQ total** | **~19** |

The C++ code has 2x more FMA instructions, suggesting:
1. More aggressive inlining
2. Different polynomial evaluation strategy
3. Better instruction selection by Clang/Highway

### C++ Loop Structure (ComputePreErosion hot path at 0x290)

```asm
# Main loop: processes 8 floats per iteration
.LBB164_14:
    vaddps ymm0, ymm25, [r9+r8*4]      # load + add
    vmulps ymm28, ymm0, ymm0           # square
    vfmadd132ps ymm27, ymm23, ymm28    # FMA
    vfmadd132ps ymm0, ymm21, ymm28     # FMA
    vdivps ymm26, ymm0, ymm27          # division
    # ... more operations
    vmovups [rcx+r8*4], ymm0           # store
    add r8, 8
    cmp rsi, r8
    ja .LBB164_14
```

### Rust Loop Structure (pre_erosion_row hot path)

```asm
# Main loop: similar structure
.LBB164_14:
    vmovups ymm14, [rdi+4*rax]
    vaddps ymm0, ymm15, ymm2
    vmulps ymm9, ymm7, ymm7
    vfmadd231ps ymm8, ymm9, ymm5
    vfmadd213ps ymm7, ymm9, [rip+.LCPI]
    vdivps ymm7, ymm7, ymm16
    # ... more operations
    vmovups [r9+4*rax], ymm2
    add r8, 8
    cmp rcx, r8
    ja .LBB164_14
```

### Conclusions

1. **Vector width is NOT the bottleneck** - both use 256-bit in hot loops
2. **FMA count differs (41 vs ~19)** - C++ has more fused operations
3. **Loop structure is similar** - no major algorithmic difference
4. **Likely causes for remaining gap:**
   - Fewer FMA fusions in Rust (LLVM vs Clang codegen)
   - Different constant loading strategies
   - Inlining decisions (C++ inlines more aggressively)
   - Memory access patterns (C++ may have better cache locality)

### Recommended Next Steps

1. **Profile cache misses:** `perf stat -e cache-misses,L1-dcache-load-misses`
2. **Force more inlining:** Add `#[inline(always)]` to all helper functions
3. **Review polynomial evaluation:** Compare `EvalRationalPolynomial` vs Rust equivalent
4. **Check constant broadcast:** C++ uses `vbroadcastss` with memory operand, Rust uses splat

---

## Part 11: Deep Dive Findings (2026-01-21)

### Cache Performance Comparison

Profiled with `perf stat` on 1507x2048 image, 20 encodes:

| Metric | Rust | C++ (subprocess overhead) |
|--------|------|---------------------------|
| IPC | **2.73** | 1.53 |
| L1 cache miss rate | 8.22% | **2.88%** |
| Branch miss rate | **1.49%** | 8.48% |

**Key finding:** Rust has 3x higher L1 cache miss rate despite higher IPC. This suggests
memory access pattern differences, not instruction throughput, are a significant factor.

### Code Structure Differences

#### 1. Boundary Handling in Inner Loop

**C++ (no conditionals in hot path):**
```cpp
// Input is pre-padded, all loads are safe
for (size_t x = 0; x < xsize; x += Lanes(df)) {
    const auto in = LoadU(df, row_in + x);
    const auto in_r = LoadU(df, row_in + x + 1);  // Always valid due to padding
    const auto in_l = LoadU(df, row_in + x - 1);  // Always valid due to padding
    // ... pure SIMD computation
}
```

**Rust (conditionals per chunk):**
```rust
for chunk in 0..chunks {
    let x = chunk * 8;
    let left = if x == 0 {
        // Construct vector element-by-element for first chunk
        f32x8::from([row[0], row[x], row[x+1], ...])  // SLOW
    } else {
        load_f32x8(row, x - 1)
    };
    // Similar for right neighbor on last chunk
}
```

**Impact:** First and last chunks have slow element-by-element vector construction.

#### 2. Load/Store Overhead

**C++ (direct intrinsic):**
```cpp
const auto in = LoadU(df, row_in + x);  // Single vmovups instruction
```

**Rust (bounds checking + conversion):**
```rust
fn load_f32x8(slice: &[f32], offset: usize) -> f32x8 {
    <[f32; 8]>::try_from(&slice[offset..offset + 8])
        .unwrap()  // Bounds check + panic path
        .into()    // Array to f32x8 conversion
}
```

**Impact:** Extra bounds checking and conversion overhead on every load.

#### 3. Scalar Remainder Loop

```rust
// Rust has scalar fallback for width % 8 != 0
for x in (chunks * 8)..width {
    // Scalar operations with per-pixel conditionals
}
```

C++ avoids this by ensuring input is always padded to SIMD width.

### Performance Improvement Opportunities

1. **Pre-pad input buffers** to eliminate boundary conditionals in hot loop
2. **Use unsafe loads** when buffer padding guarantees safety (behind `unsafe_simd` feature)
3. **Eliminate scalar remainder** by padding width to multiple of 8
4. **Cache-friendly access patterns:** Review buffer layouts for better locality

### Estimated Impact

| Optimization | Estimated Speedup |
|--------------|------------------|
| Remove boundary conditionals | 5-10% |
| Use direct SIMD loads | 3-5% |
| Eliminate scalar remainder | 2-3% |
| Better cache locality | 10-20% |
| **Total potential** | **20-35%** |

### Action Items

1. [x] Add buffer padding in `StreamingAQ` to eliminate boundary checks
2. [x] Implement archmage-simd feature with direct intrinsics for hot paths
3. [ ] Profile individual functions with `perf record` to find exact hotspots
4. [ ] Consider reordering memory accesses for better cache behavior

---

## Part 12: Archmage Integration Results (2026-01-21)

### Implementation Summary

Added `archmage-simd` feature with token-based safe intrinsics for `pre_erosion_row_padded`.

**Key optimizations:**
1. **Raw pointer loads** instead of slice-to-array conversion (eliminates bounds checks)
2. **All constants hoisted** outside loop (broadcast once, reuse)
3. **Inline everything** - `#[inline(always)]` on inner function to avoid call overhead
4. **Padded buffers** - eliminates boundary conditionals in inner loop

### Benchmark Results

**Isolated function benchmark** (`cargo bench -p zenjpeg --bench aq_simd --features "archmage-simd,test-utils"`):

| Width | wide crate | archmage | Speedup |
|-------|-----------|----------|---------|
| 64 | 41ns | 26ns | 1.6x |
| 256 | 149ns | 80ns | 1.9x |
| 1024 | 607ns | 302ns | 2.0x |
| 4096 | 2.8µs | 1.2µs | **2.3x** |

**End-to-end encode benchmark** (1448x1080 image, quality 75, 100 iterations):

| Build | Average | Throughput |
|-------|---------|------------|
| wide only | 21.85ms | 71.6 MP/s |
| archmage-simd | 20.58ms | 76.0 MP/s |

**Improvement: 5.8% faster overall** with archmage.

### Code Structure

**File:** `zenjpeg/src/quant/aq/simd.rs`

```rust
#[cfg(all(feature = "archmage-simd", target_arch = "x86_64"))]
mod archmage_impl {
    use archmage::{arcane, HasAvx2, HasFma};
    use core::arch::x86_64::*;

    #[arcane]
    #[inline(always)]
    fn mage_pre_erosion_row_padded_inner<T: HasAvx2 + HasFma + Copy>(
        _token: T,
        row_ptr: *const f32,
        row_above_ptr: *const f32,
        row_below_ptr: *const f32,
        output_ptr: *mut f32,
        width: usize,
    ) {
        // Broadcast constants once outside loop
        let quarter = _mm256_set1_ps(0.25);
        let gamma_offset = _mm256_set1_ps(GAMMA_OFFSET);
        // ... more constants ...

        for chunk in 0..(width / 8) {
            let x = chunk * 8;
            let buf_x = x + 1;  // Padded buffer offset

            // Direct pointer loads - no bounds checks
            let pixels = _mm256_loadu_ps(row_ptr.add(buf_x));
            let left = _mm256_loadu_ps(row_ptr.add(buf_x - 1));
            let right = _mm256_loadu_ps(row_ptr.add(buf_x + 1));
            let top = _mm256_loadu_ps(row_above_ptr.add(buf_x));
            let bottom = _mm256_loadu_ps(row_below_ptr.add(buf_x));

            // Full computation inlined...
            let result = /* ... */;

            _mm256_storeu_ps(output_ptr.add(x), result);
        }
    }
}
```

**Integration in `streaming.rs`:**

```rust
// Use archmage SIMD when available (2x faster)
#[cfg(all(feature = "archmage-simd", target_arch = "x86_64"))]
if let Some(token) = self.archmage_token {
    mage_pre_erosion_row_padded(token, row_curr, row_above, row_below, width, output);
} else {
    pre_erosion_row_padded(row_curr, row_above, row_below, width, output);
}
```

### Why Archmage is Faster than Wide

1. **Wide uses `cfg(target_feature)`** - compile-time check, not runtime dispatch
   - Without `-C target-cpu=x86-64-v3`, wide falls back to SSE (128-bit xmm)
   - `#[multiversed]` dispatch doesn't help because wide's internal checks are compile-time

2. **Archmage uses `#[target_feature]`** - function-level attribute
   - Enables AVX2+FMA for specific functions via token system
   - Generates ymm (256-bit) code even without global target-cpu flag

3. **Direct pointer loads** eliminate:
   - Slice bounds checking
   - `try_from().unwrap()` conversion overhead
   - Array-to-SIMD type conversion

### Remaining Performance Gap (after Part 12)

With archmage pre_erosion, we improved from 1.6x slower to ~1.5x slower vs C++:

| Metric | Before | After | Target |
|--------|--------|-------|--------|
| pre_erosion (isolated) | 2.8µs | 1.2µs | ~1µs |
| Full encode | 22ms | 20.6ms | ~12ms |

---

## Part 13: Fused per_block_modulations with Archmage (2026-01-21)

### Problem: Separate HF and Gamma Loops

The original `per_block_modulations_row` called two separate functions for each 8x8 block:
1. `hf_modulation_sum_8x8` - loads 8 rows, computes horizontal/vertical differences
2. `gamma_modulation_sum_8x8` - loads same 8 rows again, computes ratio_of_derivatives

This doubled memory traffic for the same data.

### Solution: Fused HF+Gamma Loop

Created `mage_per_block_modulations_row` that fuses both computations into a single 8-row loop:

```rust
#[arcane]
fn mage_hf_gamma_sum_8x8<T: HasAvx2 + HasFma + Copy>(
    token: T,
    block_ptr: *const f32,
    stride: usize,
    // Pre-broadcast constants passed in (not created per-block)
    zero: __m256, bias: __m256, mask_first_7: __m256,
    k_num_mul: __m256, k_num_off: __m256, k_den_mul: __m256, k_voff: __m256,
) -> (f32, f32) {
    let mut hf_acc = _mm256_setzero_ps();
    let mut gamma_acc = _mm256_setzero_ps();

    for dy in 0..8usize {
        let row_ptr = block_ptr.add(dy * stride);
        let row = _mm256_loadu_ps(row_ptr);
        let row_right = _mm256_loadu_ps(row_ptr.add(1));

        // HF horizontal: |row - row_right| * mask
        let h_diff = _mm256_sub_ps(row, row_right);
        let h_abs = _mm256_andnot_ps(_mm256_set1_ps(-0.0), h_diff);
        hf_acc = _mm256_add_ps(hf_acc, _mm256_mul_ps(h_abs, mask_first_7));

        // HF vertical (rows 0-6 only)
        if dy < 7 {
            let next_row = _mm256_loadu_ps(block_ptr.add((dy + 1) * stride));
            let v_abs = _mm256_andnot_ps(_mm256_set1_ps(-0.0), _mm256_sub_ps(row, next_row));
            hf_acc = _mm256_add_ps(hf_acc, v_abs);
        }

        // Gamma: ratio_of_derivatives_inv(row + bias)
        let row_biased = _mm256_add_ps(row, bias);
        let gamma_val = mage_ratio_of_derivatives_inv_x8(...);
        gamma_acc = _mm256_add_ps(gamma_acc, gamma_val);
    }

    (hsum_ps(token, hf_acc), hsum_ps(token, gamma_acc))
}
```

### Key Optimizations

1. **Fused loop** - One pass over 8 rows instead of two (halves memory traffic)
2. **Constants hoisted** - All SIMD constants broadcast once outside the block loop
3. **Constants passed to inner function** - Avoids repeated broadcasts per-block
4. **Raw pointer loads** - No bounds checks, no slice-to-array conversion
5. **Same data reused** - Row loaded once, used for both HF and gamma

### Benchmark Results

**Test:** 1448x1080 image, quality 75, 200 iterations, default target (binary distribution)

| Build | Average | Throughput | vs baseline |
|-------|---------|------------|-------------|
| wide only | 21.19ms | 73.8 MP/s | baseline |
| archmage-simd | 19.79ms | 79.0 MP/s | **6.6% faster** |

### Numerical Parity

Initial implementation had different `fast_log2` approximation, causing 0.06% file size difference.
Fixed by matching the Horner's method coefficients from `super::fast_log2`.
**Both versions now produce byte-identical JPEG output.**

### Multiversion Analysis

Tested removing `#[multiversed]` tags with `-C target-cpu=native`:
- With multiversion: 18.85ms
- Without multiversion: 18.74ms
- **Difference: ~0.6% (within noise)**

For binary distribution, `#[multiversed]` + `wide` is useless because `wide` uses
`cfg(target_feature)` (compile-time), not `#[target_feature]` (function-level).
The multiversion dispatch selects between versions that all use SSE-level code.

**Recommendation:** Use `archmage-simd` feature for binary distribution. The archmage
token system (`Avx2FmaToken::try_new()`) provides proper runtime AVX2+FMA detection.

---

## Part 14: Handoff Notes (2026-01-21)

### Current State

**Performance (binary distribution, default target):**
- wide only: 21.2ms / 73.8 MP/s
- archmage-simd: 19.8ms / 79.0 MP/s (6.6% faster)
- C++ jpegli: ~12ms (target)
- **Gap: ~1.65x slower than C++**

**Archmage coverage:**
- ✅ `pre_erosion_row_padded` - 2.3x faster isolated, ~3% end-to-end
- ✅ `per_block_modulations_row` - fused HF+gamma, ~3% end-to-end
- ❌ `compute_fuzzy_erosion_row_into` - still scalar
- ❌ `quant_field_to_aq_strength` - still scalar loop

### Files Modified

```
zenjpeg/src/quant/aq/simd.rs
├── archmage_impl module (lines 908-1358)
│   ├── mage_pre_erosion_row_padded_inner
│   ├── mage_pre_erosion_row_padded (public)
│   ├── mage_hf_gamma_sum_8x8
│   ├── mage_ratio_of_derivatives_inv_x8
│   ├── mage_fast_log2, mage_fast_exp2
│   └── mage_per_block_modulations_row (public)
└── pub use statements for archmage functions

zenjpeg/src/quant/aq/streaming.rs
├── archmage_token field in StreamingAQ
├── Token initialization in new()
├── Conditional dispatch in compute_and_accumulate_pre_erosion
├── Conditional dispatch in compute_last_row_pre_erosion
└── Conditional dispatch in finalize_imcu_aq_with_buffer
```

### Remaining Optimization Opportunities

1. **`compute_fuzzy_erosion_row_into`** (in streaming.rs)
   - Currently scalar, processes pre_erosion buffer
   - 3x3 window, partial sort to find 4 smallest, weighted sum
   - Could use SIMD for the sorting network

2. **`quant_field_to_aq_strength`** (scalar loop)
   - Simple `1.0 / (1.0 + x)` transform
   - Could process 8 values at once with SIMD

3. **Cache locality** (3x higher L1 miss rate than C++)
   - Profile with `perf stat -e L1-dcache-load-misses`
   - Review buffer access patterns
   - Consider prefetch hints

4. **AVX-512** (currently unused)
   - Both C++ and Rust use ymm (256-bit) in hot loops
   - AVX-512 zmm could help for large images
   - Would need new archmage token type

### Testing Commands

```bash
# Run streaming AQ tests
cargo test -p zenjpeg --features "archmage-simd,test-utils" --lib streaming --release

# Benchmark comparison
cargo build --release -p zenjpeg --example cjpegli_rs_profile --features "test-utils"
./target/release/examples/cjpegli_rs_profile IMAGE.png --disable_output -q 75 --num_reps 200

cargo build --release -p zenjpeg --example cjpegli_rs_profile --features "test-utils,archmage-simd"
./target/release/examples/cjpegli_rs_profile IMAGE.png --disable_output -q 75 --num_reps 200

# Verify numerical parity
md5sum /tmp/out_wide.jpg /tmp/out_mage.jpg  # Should match
```

### Key Learnings

1. **`wide` crate limitation:** Uses `cfg(target_feature)` at compile time, ignores
   `#[target_feature]` function attributes. `#[multiversed]` dispatch is useless.

2. **archmage pattern:** Token proves CPU capability, `#[arcane]` enables intrinsics,
   pass pre-broadcast constants to inner functions to avoid repeated setup.

3. **Fusing loops:** When two functions read same data, fuse into one loop.
   Memory bandwidth is often the bottleneck, not compute.

4. **Numerical parity:** Different polynomial approximations (log2, exp2) can cause
   small differences. Match coefficients exactly for byte-identical output.
