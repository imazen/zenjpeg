# SIMD Portability Analysis: WASM and ARM NEON Support

This document analyzes Intel-specific SIMD code in jpegli-rs that needs portable alternatives for WASM and ARM NEON targets.

## Executive Summary

| Component | Intel-Specific Code | Portable Alternative | Priority | Status |
|-----------|---------------------|---------------------|----------|--------|
| Integer IDCT | AVX2 transpose + butterfly | `wide` i32x8 + multiversion (1.55x faster!) | HIGH | ✅ Done |
| YCbCr→RGB i16 | AVX2 shuffle + pack | `wide` i32x8 works | HIGH | ✅ Done |
| Even/Odd Gather | AVX2 permute | `wide` f32x8 works | MEDIUM | ✅ Done |
| Forward DCT | AVX2+FMA | `multiversion` aarch64+neon | CRITICAL | ✅ Done |
| RGB→YCbCr encode | AVX2 shuffle | `wide` f32x8 fallback | MEDIUM | ✅ Done |

## Detailed Findings

### 1. Integer IDCT (`decode/idct_int.rs:304-555`) - HIGH PRIORITY

**What it does:** 8x8 inverse DCT for standard JPEG decoding (~40% of decode time)

**Intel-specific operations:**
- `_mm256_permute4x64_epi64` - 64-bit lane permutation
- `_mm256_unpacklo/hi_epi32` - 32-bit interleave
- `_mm256_unpacklo/hi_epi64` - 64-bit interleave
- `_mm256_permute2x128_si256` - 128-bit lane exchange
- `_mm256_packs_epi32` - Pack with saturation
- `_mm256_srai_epi32` - Arithmetic right shift

**Benchmark Results (x86_64 AVX2):**
```
# WITHOUT multiversion (wide compiled without AVX2 target features):
Scalar IDCT: 0.043 µs/block
Wide IDCT:   0.092 µs/block
Wide speedup: 0.46x (SLOWER - wide uses scalar fallback!)

# WITH multiversion targets("x86_64+avx2"):
Scalar IDCT: 0.044 µs/block
Wide IDCT:   0.029 µs/block
Wide speedup: 1.55x (FASTER!)
```

**CRITICAL:** The `wide` crate uses compile-time `#[cfg]` detection, NOT runtime detection.
You MUST use `#[multiversion(targets(...))]` or `#[target_feature(enable = "avx2")]`
for `wide` to use its fast paths.

**Recommendation:**
- **x86_64 AVX2:** Use `wide` i32x8 with multiversion (1.55x faster than scalar)
- **ARM NEON:** Use `wide` i32x8 with multiversion - needs benchmarking
- **WASM:** Use scalar (wide transpose has no WASM SIMD implementation)
- Consider replacing existing AVX2 intrinsics with `wide` + multiversion for simpler code

---

### 2. YCbCr→RGB i16 Conversion (`color/ycbcr.rs:927-1085`) - HIGH PRIORITY

**What it does:** Convert 16 YCbCr pixels to interleaved RGB (decode color conversion)

**Intel-specific operations:**
- `_mm256_shuffle_epi8` (PSHUFB) - Byte shuffle for RGB interleaving
- `_mm256_blendv_epi8` (VPBLENDVB) - Conditional byte select
- `_mm256_permute4x64_epi64` - Lane reordering after packing
- `_mm256_packs/packus_epi32/16` - Pack with saturation

**Portable solution implemented:** See `simd_parity_test.rs::ycbcr_wide`
- Uses `wide::i32x8` for the math (works perfectly)
- Extracts to arrays for RGB interleaving (acceptable overhead)
- **Status: ✅ VERIFIED EXACT MATCH with scalar**

**Performance:** ~same as scalar on x86_64 (wide uses SSE/AVX under hood)

---

### 3. Even/Odd Gather (`encode_simd.rs:245-276`) - MEDIUM PRIORITY

**What it does:** Deinterleave even/odd elements for chroma downsampling

**Intel-specific operations:**
- `_mm256_shuffle_ps` with 0x88/0xDD masks
- `_mm256_permute4x64_epi64` with 0xD8 ordering

**Portable solution implemented:** See `simd_parity_test.rs::gather_wide`
- Direct array construction: `f32x8::from([data[0], data[2], ...])`
- **Status: ✅ VERIFIED EXACT MATCH with scalar**

---

### 4. Forward DCT (`encode/dct.rs:240-512`) - CRITICAL but already handled

**What it does:** 8x8 DCT for JPEG encoding (18-25% of encode time)

**Current status:** Already uses `multiversion` macro with targets:
- `"x86_64+avx2+fma"`
- `"x86_64+sse2"`
- `"aarch64+neon"` ✅

The `wide` crate fallback path is also available. **No additional work needed.**

---

### 5. RGB→YCbCr Encoding (`encode_simd.rs:430-510`) - MEDIUM PRIORITY

**Intel-specific operations:**
- `_mm_shuffle_epi8` - RGB channel extraction (deinterleave)
- `_mm_fmadd_ps` - FMA for color matrix multiply

**Current status:** Has `wide` crate fallback at line 608 (`rgb_to_ycbcr_planes_simd_inplace_fallback`)
- Uses `f32x8::from([r0, r3, r6, ...])` for manual gather
- Uses `f32x8::mul_add()` for FMA operations
- **Already works on all platforms**

---

## Test Infrastructure

### Running Parity Tests

```bash
# Side-by-side Intel vs portable comparison
cargo run --release --example simd_parity_test

# Full C++ parity (requires cjpegli build)
cargo test --release -- comparison --nocapture --ignored
```

### Adding New Portable Implementations

1. Create standalone function (not inside `multiversion`)
2. Add to `simd_parity_test.rs` with scalar reference
3. Verify exact match on x86_64
4. Test on ARM via cross-compilation or CI

---

## Recommended Actions

### Immediate (before 1.0 release)
1. ✅ YCbCr→RGB i16: Already has portable path
2. ✅ Even/odd gather: Already has portable path
3. ✅ Integer IDCT: Implemented with `wide` + multiversion (see `decode/idct_int.rs`)

### Future Optimization
1. ARM NEON-specific paths for transpose (vtrn, vzip instructions)
2. WASM SIMD paths if `wide` crate performance is insufficient
3. Runtime feature detection for ARM SVE

---

## Platform Support Matrix

| Platform | IDCT | YCbCr→RGB | DCT | Status |
|----------|------|-----------|-----|--------|
| x86_64 AVX2 | `wide` i32x8 (**1.77x faster**) | AVX2 | AVX2+FMA | ✅ Done |
| x86_64 SSE2 | `wide` i32x8 | Scalar | SSE2 | ✅ Works |
| x86_64 SSE4.1 | `wide` i32x8 | Multiversion | SSE4.1 | ✅ Works |
| aarch64 NEON | `wide` i32x8 (**1.07x faster**) | Multiversion | NEON | ✅ Tested |
| arm NEON | `wide` i32x8 | Multiversion | NEON | ✅ Targets |
| wasm32 | Scalar/SIMD128 | Portable | Scalar/SIMD128 | ✅ Parity verified |
| wasm32+simd128 | `wide` SIMD | Portable | `wide` SIMD | ✅ Builds (69KB) |

### Benchmark Results (standalone test, Jan 2025)

```
x86_64 AVX2:
  Scalar: 62.5 ns/block
  Wide:   35.3 ns/block
  Speedup: 1.77x

aarch64 NEON (via qemu):
  Scalar: 670.1 ns/block
  Wide:   628.2 ns/block
  Speedup: 1.07x (qemu overhead affects this)

wasm32-wasip1:
  Parity: ✓ Scalar and Wide MATCH exactly
  SIMD128: Requires RUSTFLAGS="-C target-feature=+simd128"
```

### WASM SIMD128 Support

To enable WASM SIMD128 acceleration:
```bash
RUSTFLAGS="-C target-feature=+simd128" cargo build --target wasm32-wasip1
```

Or enable in `.cargo/config.toml`:
```toml
[target.wasm32-wasip1]
rustflags = ["-C", "target-feature=+simd128"]
```

**Runtime Support (2025):** SIMD128 is universally supported:

| Runtime | SIMD128 Since | Notes |
|---------|---------------|-------|
| Chrome | v91 (May 2021) | Full support |
| Firefox | v89 (Jun 2021) | Full support |
| Safari | v16.4 (Mar 2023) | Full support |
| Edge | v91 (May 2021) | Full support |
| Node.js | v16.4 (Jun 2021) | Full support |
| Wasmtime | v0.26 (Mar 2021) | Full support |
| Wasmer | v2.0 (Jun 2021) | Full support |
| wasm3 | v0.5.0 | Interpreter, slower |

**Note:** WASM SIMD detection is compile-time only (no runtime detection).
Binaries built with simd128 require a SIMD-capable runtime.

**Testing:** Use the `wasm-simd` feature to gate SIMD-specific tests:
```bash
RUSTFLAGS="-C target-feature=+simd128" cargo test --target wasm32-wasip1 --features wasm-simd
```

**Key finding:** The `wide` crate REQUIRES `#[multiversion]` or `#[target_feature]`
to use SIMD - it uses compile-time `#[cfg]`, not runtime detection. Without this,
`wide` falls back to scalar and is slower than hand-written scalar code!

---

## Files Modified

- `jpegli-rs/examples/simd_parity_test.rs` - Side-by-side comparison tests
- `docs/SIMD_PORTABILITY.md` - This document

## Related Files

- `jpegli-rs/src/decode/idct_int.rs` - Integer IDCT with AVX2
- `jpegli-rs/src/color/ycbcr.rs` - YCbCr color conversion
- `jpegli-rs/src/encode_simd.rs` - Encoding SIMD functions
- `jpegli-rs/src/encode/dct.rs` - Forward DCT with multiversion
