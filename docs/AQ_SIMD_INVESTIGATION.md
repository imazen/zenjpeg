# AQ SIMD Investigation: Closing the C++ Performance Gap

## Executive Summary

**Problem:** Rust jpegli-rs AQ functions consume 24% of encode time vs C++ jpegli's 8% (3x relative overhead).

**Current state (2026-01-20):**
- Default build: Rust 1.6x slower than C++
- With `-C target-cpu=x86-64-v4`: Rust 1.33x slower (40% of gap closed)
- **Key insight:** The relative AQ percentage stays constant even with -v4, ruling out vector width as bottleneck

**Root cause analysis (from disassembly comparison):**
- Both use ymm (256-bit) registers in hot loops, not zmm
- C++ has 41 FMA instructions vs Rust's ~19 in AQ code
- C++ uses `HWY_CAPPED(float, 8)` for 8x8 block ops (same as Rust f32x8)
- Difference is likely instruction scheduling, loop structure, or inlining

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
jpegli-rs/src/quant/aq/
├── mod.rs              # AQ module root
├── simd.rs             # SIMD implementations (wide crate)
├── streaming.rs        # Streaming AQ processor
└── tables.rs           # Precomputed tables

jpegli-rs/src/encode/
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
cd /home/lilith/work/jpegli-rs/internal/jpegli-cpp

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
cd /home/lilith/work/jpegli-rs

# Install cargo-asm if needed
cargo install cargo-asm

# View assembly for specific function (default target)
cargo asm -p jpegli-rs --lib "jpegli::quant::aq::simd::pre_erosion_row" 2>/dev/null | head -100

# With x86-64-v3 (AVX2 + FMA)
RUSTFLAGS="-C target-cpu=x86-64-v3" cargo asm -p jpegli-rs --lib \
    "jpegli::quant::aq::simd::pre_erosion_row" 2>/dev/null | head -100

# With x86-64-v4 (AVX-512)
RUSTFLAGS="-C target-cpu=x86-64-v4" cargo asm -p jpegli-rs --lib \
    "jpegli::quant::aq::simd::pre_erosion_row" 2>/dev/null | head -100

# Count register usage (xmm=128-bit, ymm=256-bit, zmm=512-bit)
RUSTFLAGS="-C target-cpu=x86-64-v4" cargo asm -p jpegli-rs --lib \
    "jpegli::quant::aq::simd::pre_erosion_row" 2>/dev/null | \
    grep -oE "(xmm|ymm|zmm)[0-9]+" | sort | uniq -c | sort -rn

# List all functions matching pattern
cargo asm -p jpegli-rs --lib 2>&1 | grep -i "aq\|erosion\|modulation"
```

### Comparing Instruction Mix

```bash
# Script to compare instruction profiles
compare_asm() {
    local func=$1
    echo "=== Rust default ==="
    cargo asm -p jpegli-rs --lib "$func" 2>/dev/null | \
        grep -oE "^[[:space:]]+[a-z]+" | sort | uniq -c | sort -rn | head -15

    echo ""
    echo "=== Rust x86-64-v4 ==="
    RUSTFLAGS="-C target-cpu=x86-64-v4" cargo asm -p jpegli-rs --lib "$func" 2>/dev/null | \
        grep -oE "^[[:space:]]+[a-z]+" | sort | uniq -c | sort -rn | head -15
}

compare_asm "jpegli::quant::aq::simd::pre_erosion_row"
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
RUSTFLAGS="-C target-cpu=x86-64-v4" cargo bench -p jpegli-rs --bench cpp_comparison

# Profile with cjpegli-compatible settings
RUSTFLAGS="-C target-cpu=x86-64-v4" cargo run --release -p jpegli-rs \
    --example cjpegli_rs_profile -- IMAGE.png -p 0 --num_reps 50

# Flamegraph + perf report
RUSTFLAGS="-C target-cpu=x86-64-v4" cargo flamegraph --release -p jpegli-rs \
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
cargo test --release -p jpegli-rs --test comprehensive_cpp_comparison -- --nocapture --ignored

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

- `jpegli-rs/src/quant/aq/simd.rs` - Main SIMD implementations
- `jpegli-rs/src/quant/aq/streaming.rs` - Streaming processor
- `jpegli-rs/Cargo.toml` - Add unsafe_simd or archmage-simd feature

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
